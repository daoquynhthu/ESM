use std::collections::HashMap;

use crate::encoder::{Column, EncoderConfig, SparseEncoder, context_key};
use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};
use crate::rng::mix64;

pub fn dominant_role(counts: &[u32], max_roles: usize) -> Option<(usize, u32, u32)> {
    let mut best_role = 0usize;
    let mut best = 0u32;
    let mut second = 0u32;
    for (role, count) in counts.iter().copied().enumerate().take(max_roles) {
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

fn context_sketch_terms(input: &InputEvent) -> [SketchTerm; 8] {
    [
        SketchTerm { value: 0x00_0000_0000u64 | input.token as u64, weight: 2, fanout: 4 },
        SketchTerm { value: 0x10_0000_0000u64 | input.prev_token as u64, weight: 1, fanout: 2 },
        SketchTerm { value: 0x20_0000_0000u64 | input.context_token as u64, weight: 30, fanout: 20 },
        SketchTerm { value: 0x30_0000_0000u64 | input.position_mod as u64, weight: 1, fanout: 2 },
        SketchTerm {
            value: 0x40_0000_0000u64 | (((input.token as u64) << 32) ^ input.prev_token as u64),
            weight: 1,
            fanout: 2,
        },
        SketchTerm {
            value: 0x50_0000_0000u64 | (((input.token as u64) << 32) ^ input.context_token as u64),
            weight: 8,
            fanout: 12,
        },
        SketchTerm {
            value: 0x60_0000_0000u64 | (((input.prev_token as u64) << 32) ^ input.context_token as u64),
            weight: 2,
            fanout: 4,
        },
        SketchTerm { value: 0, weight: 0, fanout: 0 },
    ]
}

#[derive(Copy, Clone, Debug)]
struct SketchTerm {
    value: u64,
    weight: i32,
    fanout: usize,
}

#[derive(Clone, Debug)]
pub struct ContextPredictiveEncoder {
    pub columns: Vec<Column>,
    pub active_bits: usize,
    pub feature_offset: u32,
    pub seed: u64,
    pub total_activations: u64,
    pub role_counts_by_column: Vec<Vec<u32>>,
    pub role_counts_by_context: HashMap<u64, Vec<u32>>,
    pub max_roles: usize,
}

impl ContextPredictiveEncoder {
    pub fn new(cfg: EncoderConfig) -> Self {
        let columns = (0..cfg.columns).map(|_| Column::new()).collect();
        Self {
            columns,
            active_bits: cfg.active_bits,
            feature_offset: 4_000_000,
            seed: cfg.seed,
            total_activations: 0,
            role_counts_by_column: vec![vec![0; cfg.max_roles]; cfg.columns],
            role_counts_by_context: HashMap::new(),
            max_roles: cfg.max_roles,
        }
    }

    fn projected_scores(&self, input: &InputEvent) -> Vec<i32> {
        let n = self.columns.len().max(1);
        let mut scores = vec![0i32; n];
        for (term_idx, term) in context_sketch_terms(input).iter().enumerate() {
            for salt in 0..term.fanout {
                let h = mix64(term.value ^ self.seed ^ ((term_idx as u64) << 32) ^ salt as u64);
                let idx = (h % n as u64) as usize;
                if let Some(s) = scores.get_mut(idx) {
                    *s = s.saturating_add(term.weight * 100);
                }
            }
        }

        // No usage homeostasis: context-dominant encoder needs the SAME
        // columns to fire repeatedly. Homeostasis forces alternation,
        // which destroys the cue→verify column overlap that the ledger
        // depends on.
        for (idx, col) in self.columns.iter().enumerate() {
            scores[idx] = scores[idx].saturating_add(col.success_mass.round() as i32);
            scores[idx] = scores[idx].saturating_add(col.credit_bias);
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

impl SparseEncoder for ContextPredictiveEncoder {
    fn name(&self) -> &'static str {
        "context-predictive"
    }

    fn column_role_margins(&self) -> Vec<u32> {
        self.role_counts_by_column.iter().map(|counts| {
            let (best, second) = counts.iter().copied().fold((0u32, 0u32), |(b, s), c| {
                if c > b { (c, b) } else if c > s { (b, c) } else { (b, s) }
            });
            best.saturating_sub(second)
        }).collect()
    }

    fn feature_offset(&self) -> u32 {
        self.feature_offset
    }

    fn sparse_role_vote(&self, code: &SparseCode) -> Option<(usize, f32)> {
        let mut votes = vec![0u32; self.max_roles];
        let mut total = 0u32;
        for fid in code.as_slice() {
            if fid.0 < self.feature_offset {
                continue;
            }
            let idx = (fid.0 - self.feature_offset) as usize;
            if idx >= self.role_counts_by_column.len() {
                continue;
            }
            let counts = &self.role_counts_by_column[idx];
            let total_for_col: u32 = counts.iter().sum();
            if total_for_col > 0 {
                if let Some((role, _, _)) = dominant_role(counts, self.max_roles) {
                    votes[role] = votes[role].saturating_add(1);
                    total += 1;
                }
            }
        }
        if total == 0 {
            return None;
        }
        let predicted = votes.iter().enumerate()
            .max_by_key(|(_, c)| **c)
            .map(|(r, _)| r)?;
        let confidence = votes[predicted] as f32 / total as f32;
        Some((predicted, confidence))
    }

    fn encode(&self, input: &InputEvent) -> SparseCode {
        let mut features: Vec<FeatureId> = self.active_column_indices(input)
            .into_iter()
            .map(|idx| FeatureId(self.feature_offset + idx as u32))
            .collect();

        let key = context_key(input);
        if let Some(counts) = self.role_counts_by_context.get(&key) {
            if let Some((role, best, second)) = dominant_role(counts, self.max_roles) {
                if best >= 2 && best >= second.saturating_add(1) {
                    features.push(FeatureId(self.feature_offset + 1_000_000 + role as u32));
                }
            }
        }

        SparseCode::new(features)
    }

    fn retrospective_credit(
        &mut self,
        cue_features: &[FeatureId],
        _cue_step: u64,
        verified_role: usize,
    ) {
        // Unconditionally add role counts to ALL cue-step features.
        // Since cue and verify share the same context, the same columns
        // fire at both steps. The extra counts from verify reinforce
        // what the column learned at cue step.
        // CRITICAL: Do NOT modify success_mass. Mass changes cause
        // columns to fire in wrong contexts (cross-context pollution).
        for fid in cue_features {
            if fid.0 < self.feature_offset {
                continue;
            }
            let idx = (fid.0 - self.feature_offset) as usize;
            if idx >= self.role_counts_by_column.len() {
                continue;
            }
            if let Some(counts) = self.role_counts_by_column.get_mut(idx) {
                counts[verified_role] = counts[verified_role].saturating_add(1);
            }
        }
    }

    fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode) {
        let role = (target.latent_role as usize) % self.max_roles;

        for f in code.as_slice() {
            if f.0 >= self.feature_offset && f.0 < self.feature_offset + self.columns.len() as u32 {
                let idx = (f.0 - self.feature_offset) as usize;
                if let Some(col) = self.columns.get_mut(idx) {
                    col.usage = col.usage.saturating_add(1);
                    self.total_activations = self.total_activations.saturating_add(1);
                }
            }
        }

        let key = context_key(input);
        let max_roles = self.max_roles;
        let counts = self
            .role_counts_by_context
            .entry(key)
            .or_insert_with(|| vec![0; max_roles]);
        counts[role] = counts[role].saturating_add(1);

        for f in code.as_slice() {
            if f.0 >= self.feature_offset && f.0 < self.feature_offset + self.columns.len() as u32 {
                let idx = (f.0 - self.feature_offset) as usize;
                if let Some(counts) = self.role_counts_by_column.get_mut(idx) {
                    counts[role] = counts[role].saturating_add(1);
                    if let Some((dominant, best, second)) = dominant_role(counts, self.max_roles) {
                        if dominant == role && best >= second.saturating_add(2) {
                            if let Some(col) = self.columns.get_mut(idx) {
                                col.success_mass = (col.success_mass + 0.25).min(50.0);
                            }
                        }
                    }
                }
            }
        }
    }
}
