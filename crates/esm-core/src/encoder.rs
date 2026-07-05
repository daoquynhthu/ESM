//! E-1A sparse encoders.
//!
//! Encoder A (`HashEncoder`) is the raw token/hash control.
//! Encoder B (`CompetitiveEncoder`) is an online sparse competitive encoder.
//! Encoder C (`PredictiveEncoder`) adds local predictive role statistics after observation.
//!
//! Encoders must not read `TargetEvent` during `encode`; target information is only used in `adapt`.

use std::collections::HashMap;

use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};
use crate::rng::mix64;

pub trait SparseEncoder {
    fn name(&self) -> &'static str;
    fn encode(&self, input: &InputEvent) -> SparseCode;
    fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode);
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EncoderKind {
    Hash,
    Competitive,
    Predictive,
}

impl EncoderKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "hash" | "a" | "control" => Some(Self::Hash),
            "competitive" | "b" => Some(Self::Competitive),
            "predictive" | "c" => Some(Self::Predictive),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EncoderConfig {
    pub feature_width: u32,
    pub active_bits: usize,
    pub columns: usize,
    pub column_receptive_cap: usize,
    pub seed: u64,
    pub max_roles: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            feature_width: 65_536,
            active_bits: 16,
            columns: 4096,
            column_receptive_cap: 16,
            seed: 1,
            max_roles: 16,
        }
    }
}

pub fn build_encoder(kind: EncoderKind, cfg: EncoderConfig) -> Box<dyn SparseEncoder> {
    match kind {
        EncoderKind::Hash => Box::new(HashEncoder::new(cfg.feature_width, cfg.active_bits)),
        EncoderKind::Competitive => Box::new(CompetitiveEncoder::new(cfg, 1_000_000)),
        EncoderKind::Predictive => Box::new(PredictiveEncoder::new(cfg)),
    }
}

#[derive(Clone, Debug)]
pub struct HashEncoder {
    feature_width: u32,
    active_bits: usize,
}

impl HashEncoder {
    pub fn new(feature_width: u32, active_bits: usize) -> Self {
        Self { feature_width, active_bits }
    }
}

impl SparseEncoder for HashEncoder {
    fn name(&self) -> &'static str {
        "hash"
    }

    fn encode(&self, input: &InputEvent) -> SparseCode {
        let mut out = Vec::with_capacity(self.active_bits);
        let base = mix64(input.token as u64);
        for i in 0..self.active_bits {
            let h = mix64(base ^ (i as u64).wrapping_mul(0x9e3779b97f4a7c15));
            out.push(FeatureId((h % self.feature_width as u64) as u32));
        }
        SparseCode::new(out)
    }

    fn adapt(&mut self, _input: &InputEvent, _target: &TargetEvent, _code: &SparseCode) {}
}

#[derive(Clone, Debug)]
struct Column {
    usage: u64,
    success_mass: f32,
}

impl Column {
    fn new() -> Self {
        Self { usage: 0, success_mass: 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct CompetitiveEncoder {
    columns: Vec<Column>,
    active_bits: usize,
    feature_offset: u32,
    seed: u64,
    total_activations: u64,
}

#[derive(Copy, Clone, Debug)]
struct SketchTerm {
    value: u64,
    weight: i32,
    fanout: usize,
}

impl CompetitiveEncoder {
    pub fn new(cfg: EncoderConfig, feature_offset: u32) -> Self {
        let mut columns = Vec::with_capacity(cfg.columns);
        for _ in 0..cfg.columns {
            columns.push(Column::new());
        }
        Self {
            columns,
            active_bits: cfg.active_bits,
            feature_offset,
            seed: cfg.seed,
            total_activations: 0,
        }
    }

    fn sketch_terms(input: &InputEvent) -> [SketchTerm; 8] {
        // The control HashEncoder only sees token identity.  The competitive encoder is
        // allowed to use pre-target input context, but never the target role/token.
        [
            SketchTerm { value: 0x00_0000_0000u64 | input.token as u64, weight: 7, fanout: 10 },
            SketchTerm { value: 0x10_0000_0000u64 | input.prev_token as u64, weight: 4, fanout: 6 },
            SketchTerm { value: 0x20_0000_0000u64 | input.context_token as u64, weight: 9, fanout: 12 },
            SketchTerm { value: 0x30_0000_0000u64 | input.position_mod as u64, weight: 3, fanout: 4 },
            SketchTerm {
                value: 0x40_0000_0000u64 | (((input.token as u64) << 32) ^ input.prev_token as u64),
                weight: 5,
                fanout: 6,
            },
            SketchTerm {
                value: 0x50_0000_0000u64 | (((input.token as u64) << 32) ^ input.context_token as u64),
                weight: 10,
                fanout: 12,
            },
            SketchTerm {
                value: 0x60_0000_0000u64 | (((input.prev_token as u64) << 32) ^ input.context_token as u64),
                weight: 5,
                fanout: 6,
            },
            SketchTerm {
                value: 0x70_0000_0000u64 | ((input.step & 0xff) ^ ((input.position_mod as u64) << 16)),
                weight: 1,
                fanout: 2,
            },
        ]
    }

    fn context_key(input: &InputEvent) -> u64 {
        mix64(
            ((input.context_token as u64) << 32)
                ^ ((input.prev_token as u64) << 16)
                ^ input.position_mod as u64,
        )
    }

    fn bump_score(scores: &mut [i32], idx: usize, delta: i32) {
        if let Some(s) = scores.get_mut(idx) {
            *s = s.saturating_add(delta);
        }
    }

    fn projected_scores(&self, input: &InputEvent) -> Vec<i32> {
        let n = self.columns.len().max(1);
        let mut scores = vec![0i32; n];
        for (term_idx, term) in Self::sketch_terms(input).iter().enumerate() {
            for salt in 0..term.fanout {
                let h = mix64(term.value ^ self.seed ^ ((term_idx as u64) << 32) ^ salt as u64);
                let idx = (h % n as u64) as usize;
                Self::bump_score(&mut scores, idx, term.weight * 100);
            }
        }

        // Homeostatic anti-collapse pressure.  This is not a time window; it is a
        // structural relative usage pressure against the current mean exposure.
        let mean_usage = if self.columns.is_empty() {
            0
        } else {
            self.total_activations / self.columns.len() as u64
        };
        for (idx, col) in self.columns.iter().enumerate() {
            if col.usage > mean_usage {
                let excess = (col.usage - mean_usage).min(10_000) as i32;
                scores[idx] = scores[idx].saturating_sub(excess * 20);
            }
            scores[idx] = scores[idx].saturating_add(col.success_mass.round() as i32);
        }
        scores
    }

    fn active_column_indices(&self, input: &InputEvent) -> Vec<usize> {
        let scores = self.projected_scores(input);
        let mut scored: Vec<(usize, i32)> = scores.into_iter().enumerate().collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(self.active_bits);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }
}

impl SparseEncoder for CompetitiveEncoder {
    fn name(&self) -> &'static str {
        "competitive"
    }

    fn encode(&self, input: &InputEvent) -> SparseCode {
        let features = self
            .active_column_indices(input)
            .into_iter()
            .map(|idx| FeatureId(self.feature_offset + idx as u32))
            .collect();
        SparseCode::new(features)
    }

    fn adapt(&mut self, _input: &InputEvent, _target: &TargetEvent, code: &SparseCode) {
        for f in code.as_slice() {
            if f.0 >= self.feature_offset {
                let idx = (f.0 - self.feature_offset) as usize;
                if let Some(col) = self.columns.get_mut(idx) {
                    col.usage = col.usage.saturating_add(1);
                    self.total_activations = self.total_activations.saturating_add(1);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PredictiveEncoder {
    base: CompetitiveEncoder,
    role_counts_by_column: Vec<Vec<u32>>,
    role_counts_by_context: HashMap<u64, Vec<u32>>,
    max_roles: usize,
    predictive_feature_offset: u32,
}

impl PredictiveEncoder {
    pub fn new(cfg: EncoderConfig) -> Self {
        let columns = cfg.columns;
        let max_roles = cfg.max_roles;
        Self {
            base: CompetitiveEncoder::new(cfg, 2_000_000),
            role_counts_by_column: vec![vec![0; max_roles]; columns],
            role_counts_by_context: HashMap::new(),
            max_roles,
            predictive_feature_offset: 3_000_000,
        }
    }

    fn dominant_role(counts: &[u32]) -> Option<(usize, u32, u32)> {
        let mut best_role = 0usize;
        let mut best = 0u32;
        let mut second = 0u32;
        for (role, count) in counts.iter().copied().enumerate() {
            if count > best {
                second = best;
                best = count;
                best_role = role;
            } else if count > second {
                second = count;
            }
        }
        if best == 0 { None } else { Some((best_role, best, second)) }
    }
}

impl SparseEncoder for PredictiveEncoder {
    fn name(&self) -> &'static str {
        "predictive"
    }

    fn encode(&self, input: &InputEvent) -> SparseCode {
        let mut features = self.base.encode(input).as_slice().to_vec();

        // Predictive role-prototype features are based only on previously observed
        // input-context statistics.  The current target is not available in encode.
        let key = CompetitiveEncoder::context_key(input);
        if let Some(counts) = self.role_counts_by_context.get(&key) {
            if let Some((role, best, second)) = Self::dominant_role(counts) {
                // Use an ordinal confidence rule rather than hidden scalar weights.
                if best >= 2 && best >= second.saturating_add(1) {
                    features.push(FeatureId(self.predictive_feature_offset + role as u32));
                }
            }
        }

        SparseCode::new(features)
    }

    fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode) {
        self.base.adapt(input, target, code);
        let role = (target.latent_role as usize) % self.max_roles;

        let key = CompetitiveEncoder::context_key(input);
        let max_roles = self.max_roles;
        let counts = self
            .role_counts_by_context
            .entry(key)
            .or_insert_with(|| vec![0; max_roles]);
        counts[role] = counts[role].saturating_add(1);

        for f in code.as_slice() {
            if f.0 >= self.base.feature_offset && f.0 < self.base.feature_offset + self.base.columns.len() as u32 {
                let idx = (f.0 - self.base.feature_offset) as usize;
                if let Some(counts) = self.role_counts_by_column.get_mut(idx) {
                    counts[role] = counts[role].saturating_add(1);
                    if let Some((dominant, best, second)) = Self::dominant_role(counts) {
                        if dominant == role && best >= second.saturating_add(2) {
                            if let Some(col) = self.base.columns.get_mut(idx) {
                                col.success_mass = (col.success_mass + 0.25).min(50.0);
                            }
                        }
                    }
                }
            }
        }
    }
}
