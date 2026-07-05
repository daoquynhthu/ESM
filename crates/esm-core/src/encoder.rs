//! E-1A sparse encoders.
//!
//! Encoder A (`HashEncoder`) is the raw token/hash control.
//! Encoder B (`CompetitiveEncoder`) is an online sparse competitive encoder.
//! Encoder C (`PredictiveEncoder`) adds local predictive role statistics after observation.
//!
//! Encoders must not read `TargetEvent` during `encode`; target information is only used in `adapt`.

use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};
use crate::rng::{mix64, XorShift64};

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
            let h = mix64(base ^ ((i as u64) * 0x9e3779b97f4a7c15));
            out.push(FeatureId((h % self.feature_width as u64) as u32));
        }
        SparseCode::new(out)
    }

    fn adapt(&mut self, _input: &InputEvent, _target: &TargetEvent, _code: &SparseCode) {}
}

#[derive(Clone, Debug)]
struct Column {
    receptive: Vec<u64>,
    usage: u64,
    success_mass: f32,
}

impl Column {
    fn new(rng: &mut XorShift64, cap: usize) -> Self {
        let mut receptive = Vec::with_capacity(cap);
        for _ in 0..cap {
            receptive.push(mix64(rng.next_u64()));
        }
        Self { receptive, usage: 0, success_mass: 0.0 }
    }

    fn overlap(&self, sketch: &[u64]) -> i32 {
        let mut n = 0;
        for x in sketch {
            if self.receptive.contains(x) {
                n += 1;
            }
        }
        n
    }

    fn adapt_receptive(&mut self, sketch: &[u64], cap: usize) {
        self.usage = self.usage.saturating_add(1);
        for x in sketch {
            if !self.receptive.contains(x) {
                if self.receptive.len() < cap {
                    self.receptive.push(*x);
                } else {
                    let idx = (self.usage as usize + *x as usize) % cap;
                    self.receptive[idx] = *x;
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompetitiveEncoder {
    columns: Vec<Column>,
    active_bits: usize,
    receptive_cap: usize,
    feature_offset: u32,
}

impl CompetitiveEncoder {
    pub fn new(cfg: EncoderConfig, feature_offset: u32) -> Self {
        let mut rng = XorShift64::new(cfg.seed);
        let mut columns = Vec::with_capacity(cfg.columns);
        for _ in 0..cfg.columns {
            columns.push(Column::new(&mut rng, cfg.column_receptive_cap));
        }
        Self {
            columns,
            active_bits: cfg.active_bits,
            receptive_cap: cfg.column_receptive_cap,
            feature_offset,
        }
    }

    fn sketch(input: &InputEvent) -> [u64; 6] {
        let raw = input.input_sketch();
        [
            mix64(raw[0]),
            mix64(raw[1]),
            mix64(raw[2]),
            mix64(raw[3]),
            mix64(raw[4]),
            mix64(raw[5]),
        ]
    }

    fn active_column_indices(&self, input: &InputEvent) -> Vec<usize> {
        let sketch = Self::sketch(input);
        let mut scored: Vec<(usize, i32)> = self
            .columns
            .iter()
            .enumerate()
            .map(|(idx, col)| {
                let overlap = col.overlap(&sketch);
                let usage_penalty = ((col.usage / 32).min(256)) as i32;
                let score = overlap * 1000 - usage_penalty + col.success_mass.round() as i32;
                (idx, score)
            })
            .collect();
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

    fn adapt(&mut self, input: &InputEvent, _target: &TargetEvent, code: &SparseCode) {
        let sketch = Self::sketch(input);
        for f in code.as_slice() {
            if f.0 >= self.feature_offset {
                let idx = (f.0 - self.feature_offset) as usize;
                if let Some(col) = self.columns.get_mut(idx) {
                    col.adapt_receptive(&sketch, self.receptive_cap);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PredictiveEncoder {
    base: CompetitiveEncoder,
    role_counts: Vec<Vec<u32>>,
    max_roles: usize,
}

impl PredictiveEncoder {
    pub fn new(cfg: EncoderConfig) -> Self {
        let columns = cfg.columns;
        let max_roles = cfg.max_roles;
        Self {
            base: CompetitiveEncoder::new(cfg, 2_000_000),
            role_counts: vec![vec![0; max_roles]; columns],
            max_roles,
        }
    }

    fn dominant_role(&self, idx: usize) -> Option<usize> {
        self.role_counts.get(idx).and_then(|counts| {
            counts
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(role, count)| if *count == 0 { usize::MAX } else { role })
        }).filter(|role| *role != usize::MAX)
    }
}

impl SparseEncoder for PredictiveEncoder {
    fn name(&self) -> &'static str {
        "predictive"
    }

    fn encode(&self, input: &InputEvent) -> SparseCode {
        self.base.encode(input)
    }

    fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode) {
        self.base.adapt(input, target, code);
        let sketch = CompetitiveEncoder::sketch(input);
        let role = (target.latent_role as usize) % self.max_roles;
        for f in code.as_slice() {
            if f.0 >= self.base.feature_offset {
                let idx = (f.0 - self.base.feature_offset) as usize;
                if let Some(counts) = self.role_counts.get_mut(idx) {
                    counts[role] = counts[role].saturating_add(1);
                }
                let mismatch = self.dominant_role(idx).map(|r| r != role).unwrap_or(false);
                if mismatch {
                    if let Some(col) = self.base.columns.get_mut(idx) {
                        // Emphasize context-sensitive features when a column is predictively mixed.
                        col.adapt_receptive(&sketch[2..], self.base.receptive_cap);
                    }
                } else if let Some(col) = self.base.columns.get_mut(idx) {
                    col.success_mass = (col.success_mass + 0.05).min(10.0);
                }
            }
        }
    }
}
