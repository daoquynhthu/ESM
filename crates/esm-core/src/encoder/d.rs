//! Encoder D series — archived experimental line.
//!
//! Dual-channel surface + role encoding with anti-Hebbian co-activation penalty
//! and context traces. FAILED Gate E-1A (role_sharing 0.001-0.005, traces had
//! zero effect). Kept for reproducibility; do not use for new experiments.
//!
//! See docs/E1A_EXPERIMENT_REPORT.md §6 for the post-mortem.

use std::collections::HashMap;

use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};
use crate::rng::mix64;

use super::{Column, EncoderConfig, SketchTerm, context_key};
use crate::encoder::SparseEncoder;

// =========================================================================
// D-column internals
// =========================================================================

#[derive(Clone, Debug)]
struct RoleColumn {
    usage: u64,
    success_mass: f32,
    role_counts: Vec<u32>,
}

impl RoleColumn {
    fn new(max_roles: usize) -> Self {
        Self { usage: 0, success_mass: 0.0, role_counts: vec![0; max_roles] }
    }
}

#[derive(Clone, Debug)]
struct ContextTrace {
    id: u64,
    context_key: u64,
    evidence_mass: f32,
    support_mass: f32,
    conflict_mass: f32,
    rent: u64,
    #[allow(dead_code)]
    last_step: u64,
    active: bool,
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

// =========================================================================
// Encoder D
// =========================================================================

#[derive(Clone, Debug)]
pub struct EncoderD {
    surface_columns: Vec<Column>,
    surface_bits: usize,
    surface_offset: u32,
    surface_total: u64,

    role_columns: Vec<RoleColumn>,
    role_bits: usize,
    role_offset: u32,
    role_total: u64,

    co_activation: HashMap<(usize, usize), u64>,

    traces: Vec<ContextTrace>,
    max_traces: usize,

    context_role_counts: HashMap<u64, Vec<u32>>,
    max_roles: usize,

    seed: u64,
    step: u64,

    enable_role_prototypes: bool,
    enable_traces: bool,

    last_surface_active: Vec<usize>,
    last_role_active: Vec<usize>,
}

impl EncoderD {
    pub fn new(cfg: EncoderConfig, role_protos: bool, traces: bool) -> Self {
        let surface_count = cfg.columns;
        let role_count = if role_protos { cfg.role_columns } else { 0 };
        let mut surface = Vec::with_capacity(surface_count);
        for _ in 0..surface_count {
            surface.push(Column::new());
        }
        let mut role = Vec::with_capacity(role_count);
        for _ in 0..role_count {
            role.push(RoleColumn::new(cfg.max_roles));
        }
        Self {
            surface_columns: surface,
            surface_bits: cfg.surface_bits.min(cfg.active_bits),
            surface_offset: 4_000_000,
            surface_total: 0,
            role_columns: role,
            role_bits: if role_protos { cfg.role_bits.min(cfg.active_bits.saturating_sub(cfg.surface_bits)) } else { 0 },
            role_offset: 4_100_000,
            role_total: 0,
            co_activation: HashMap::new(),
            traces: Vec::new(),
            max_traces: (if role_protos { cfg.role_bits } else { cfg.surface_bits }).max(1) * 2,
            context_role_counts: HashMap::new(),
            max_roles: cfg.max_roles,
            seed: cfg.seed,
            step: 0,
            enable_role_prototypes: role_protos,
            enable_traces: traces,
            last_surface_active: Vec::new(),
            last_role_active: Vec::new(),
        }
    }

    fn surface_projected_scores(&self, input: &InputEvent) -> Vec<i32> {
        let n = self.surface_columns.len().max(1);
        let mut scores = vec![0i32; n];
        for (term_idx, term) in Self::surface_terms(input).iter().enumerate() {
            for salt in 0..term.fanout {
                let h = mix64(term.value ^ self.seed ^ ((term_idx as u64) << 32) ^ salt as u64);
                let idx = (h % n as u64) as usize;
                if let Some(s) = scores.get_mut(idx) {
                    *s = s.saturating_add(term.weight * 100);
                }
            }
        }
        let mean = if self.surface_columns.is_empty() { 0 } else { self.surface_total / self.surface_columns.len() as u64 };
        for (idx, col) in self.surface_columns.iter().enumerate() {
            if col.usage > mean {
                let excess = (col.usage - mean).min(10_000) as i32;
                scores[idx] = scores[idx].saturating_sub(excess * 20);
            }
            scores[idx] = scores[idx].saturating_add(col.success_mass.round() as i32);
        }
        scores
    }

    fn surface_terms(input: &InputEvent) -> [SketchTerm; 8] {
        [
            SketchTerm { value: 0x00_0000_0000u64 | input.token as u64, weight: 7, fanout: 10 },
            SketchTerm { value: 0x10_0000_0000u64 | input.prev_token as u64, weight: 4, fanout: 6 },
            SketchTerm { value: 0x20_0000_0000u64 | input.context_token as u64, weight: 9, fanout: 12 },
            SketchTerm { value: 0x30_0000_0000u64 | input.position_mod as u64, weight: 3, fanout: 4 },
            SketchTerm {
                value: 0x40_0000_0000u64 | (((input.token as u64) << 32) ^ input.prev_token as u64),
                weight: 5, fanout: 6,
            },
            SketchTerm {
                value: 0x50_0000_0000u64 | (((input.token as u64) << 32) ^ input.context_token as u64),
                weight: 10, fanout: 12,
            },
            SketchTerm {
                value: 0x60_0000_0000u64 | (((input.prev_token as u64) << 32) ^ input.context_token as u64),
                weight: 5, fanout: 6,
            },
            SketchTerm {
                value: 0x70_0000_0000u64 | ((input.step & 0xff) ^ ((input.position_mod as u64) << 16)),
                weight: 1, fanout: 2,
            },
        ]
    }

    fn role_projected_scores(&self, input: &InputEvent) -> Vec<i32> {
        if self.role_columns.is_empty() {
            return Vec::new();
        }
        let n = self.role_columns.len();
        let mut scores = vec![0i32; n];

        for salt in 0..12 {
            let h = mix64(context_key(input) ^ self.seed ^ 0x8000_0000u64 ^ salt as u64);
            let idx = (h % n as u64) as usize;
            if let Some(s) = scores.get_mut(idx) {
                *s = s.saturating_add(300);
            }
        }
        for &sf in &self.last_surface_active {
            for salt in 0..4 {
                let h = mix64(
                    (sf as u64).wrapping_mul(0x9e3779b97f4a7c15)
                        ^ self.seed ^ 0x9000_0000u64 ^ salt as u64,
                );
                let idx = (h % n as u64) as usize;
                if let Some(s) = scores.get_mut(idx) {
                    *s = s.saturating_add(150);
                }
            }
        }
        if self.enable_traces {
            for trace in &self.traces {
                if !trace.active { continue; }
                for salt in 0..4 {
                    let h = mix64(trace.id ^ self.seed ^ 0xA000_0000u64 ^ salt as u64);
                    let idx = (h % n as u64) as usize;
                    if let Some(s) = scores.get_mut(idx) {
                        let trace_bonus = (trace.support_mass * 100.0) as i32;
                        *s = s.saturating_add(trace_bonus.max(0));
                    }
                }
            }
        }

        let mean = if self.role_columns.is_empty() { 0 } else { self.role_total / self.role_columns.len() as u64 };
        for (idx, col) in self.role_columns.iter().enumerate() {
            if col.usage > mean {
                let excess = (col.usage - mean).min(10_000) as i32;
                scores[idx] = scores[idx].saturating_sub(excess * 20);
            }
            scores[idx] = scores[idx].saturating_add(col.success_mass.round() as i32);
        }
        scores
    }

    fn anti_hebbian_penalty_on_active(active: &[usize], offset: usize, co_activation: &HashMap<(usize, usize), u64>) -> Vec<usize> {
        let mut penalized = Vec::new();
        for i in 0..active.len() {
            let gi = active[i] + offset;
            for j in (i + 1)..active.len() {
                let gj = active[j] + offset;
                let key = if gi < gj { (gi, gj) } else { (gj, gi) };
                if let Some(&count) = co_activation.get(&key) {
                    if count > 50 {
                        penalized.push(active[i]);
                        penalized.push(active[j]);
                    }
                }
            }
        }
        penalized.sort_unstable();
        penalized.dedup();
        penalized
    }

    fn update_traces_prequential(&mut self, input: &InputEvent) {
        if !self.enable_traces || self.max_traces == 0 {
            return;
        }
        let ck = context_key(input);

        if input.context_token != 0 || (input.step % 5 == 0 && self.traces.len() < self.max_traces) {
            let already_exists = self.traces.iter().any(|t| t.active && t.context_key == ck);
            if !already_exists && self.traces.len() < self.max_traces {
                let id = mix64(ck ^ self.seed ^ self.traces.len() as u64);
                self.traces.push(ContextTrace {
                    id,
                    context_key: ck,
                    evidence_mass: 1.0,
                    support_mass: 0.0,
                    conflict_mass: 0.0,
                    rent: 0,
                    last_step: self.step,
                    active: true,
                });
            }
        }

        for trace in &mut self.traces {
            if trace.active {
                trace.rent = trace.rent.saturating_add(1);
                let budget = trace.support_mass.max(2.0) + 2.0;
                if (trace.rent as f32) > budget {
                    trace.active = false;
                }
            }
        }
    }

    fn update_traces_post_observation(&mut self, _input: &InputEvent, _target: &TargetEvent) {
        if !self.enable_traces {
            return;
        }
        let ck = context_key(_input);

        for trace in &mut self.traces {
            if !trace.active { continue; }
            if trace.context_key == ck || (_input.context_token != 0 && (trace.context_key ^ ck) & 0xFFFF_FFFF_0000_0000 == 0) {
                trace.support_mass += 0.5;
                trace.evidence_mass += 0.5;
            } else if trace.context_key & 0xFFFF_FFFF_0000_0000 == ck & 0xFFFF_FFFF_0000_0000 {
                trace.conflict_mass += 0.3;
            }
        }
    }

    fn select_topk(scores: &[i32], k: usize) -> Vec<usize> {
        if scores.is_empty() || k == 0 { return Vec::new(); }
        let mut scored: Vec<(usize, i32)> = scores.iter().copied().enumerate().collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }
}

impl SparseEncoder for EncoderD {
    fn name(&self) -> &'static str {
        if self.enable_role_prototypes && self.enable_traces {
            "d-full"
        } else if self.enable_role_prototypes {
            "d-no-trace"
        } else {
            "d-no-role-proto"
        }
    }

    fn encode(&self, input: &InputEvent) -> SparseCode {
        let mut features = Vec::new();
        let n_surface = self.surface_columns.len();

        let surface_scores = self.surface_projected_scores(input);
        let mut surface_active = Self::select_topk(&surface_scores, self.surface_bits + 4);
        let penalized = Self::anti_hebbian_penalty_on_active(&surface_active, 0, &self.co_activation);
        surface_active.retain(|idx| !penalized.contains(idx));
        surface_active.truncate(self.surface_bits);
        for &idx in &surface_active {
            features.push(FeatureId(self.surface_offset + idx as u32));
        }

        if self.enable_role_prototypes && !self.role_columns.is_empty() {
            let role_scores = self.role_projected_scores(input);
            let mut role_active = Self::select_topk(&role_scores, self.role_bits + 4);
            let penalized = Self::anti_hebbian_penalty_on_active(&role_active, n_surface, &self.co_activation);
            role_active.retain(|idx| !penalized.contains(idx));
            role_active.truncate(self.role_bits);
            for &idx in &role_active {
                features.push(FeatureId(self.role_offset + idx as u32));
            }
        }

        SparseCode::new(features)
    }

    fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, _code: &SparseCode) {
        let role = (target.latent_role as usize) % self.max_roles;

        self.update_traces_prequential(input);

        let surface_scores = self.surface_projected_scores(input);
        self.last_surface_active = Self::select_topk(&surface_scores, self.surface_bits);
        for &idx in &self.last_surface_active {
            if let Some(col) = self.surface_columns.get_mut(idx) {
                col.usage = col.usage.saturating_add(1);
                self.surface_total = self.surface_total.saturating_add(1);
            }
        }

        if self.enable_role_prototypes && !self.role_columns.is_empty() {
            let role_scores = self.role_projected_scores(input);
            self.last_role_active = Self::select_topk(&role_scores, self.role_bits);
            for &idx in &self.last_role_active {
                if let Some(col) = self.role_columns.get_mut(idx) {
                    col.usage = col.usage.saturating_add(1);
                    self.role_total = self.role_total.saturating_add(1);
                    col.role_counts[role] = col.role_counts[role].saturating_add(1);
                    if let Some((_dominant, best, second)) = dominant_role(&col.role_counts) {
                        if best > second.saturating_add(2) {
                            col.success_mass = (col.success_mass + 0.25).min(50.0);
                        }
                    }
                }
            }
            let ck = context_key(input);
            let max_roles = self.max_roles;
            self.context_role_counts.entry(ck)
                .or_insert_with(|| vec![0; max_roles])[role] =
                self.context_role_counts.get(&ck).map(|c| c[role]).unwrap_or(0).saturating_add(1);
        }

        let all_active: Vec<usize> = self.last_surface_active.iter().copied()
            .chain(self.last_role_active.iter().map(|i| i + self.surface_columns.len()))
            .collect();
        for i in 0..all_active.len() {
            for j in (i + 1)..all_active.len() {
                let a = all_active[i];
                let b = all_active[j];
                let key = if a < b { (a, b) } else { (b, a) };
                *self.co_activation.entry(key).or_insert(0) = self.co_activation.get(&key).copied().unwrap_or(0).saturating_add(1);
            }
        }
        if self.step % 1000 == 0 && self.co_activation.len() > 10000 {
            self.co_activation.retain(|_, v| *v > 20);
        }

        self.update_traces_post_observation(input, target);

        self.step = self.step.saturating_add(1);
    }
}
