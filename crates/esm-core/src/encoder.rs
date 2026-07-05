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

#[derive(Clone, Debug)]
pub struct DenseUpdateStats {
    pub loss: f32,
    pub gradient_norm: f32,
}

#[derive(Clone, Debug)]
pub struct DenseReport {
    pub feature_embeddings: HashMap<FeatureId, Vec<f32>>,
    pub feature_credits: HashMap<FeatureId, f32>,
    pub weight_norm: f32,
    pub bias_norm: f32,
}

pub trait SparseEncoder {
    fn name(&self) -> &'static str;
    fn encode(&self, input: &InputEvent) -> SparseCode;
    fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode);

    fn dense_predict_prequential(&self, _code: &SparseCode) -> Option<Vec<f32>> {
        None
    }

    fn dense_adapt(&mut self, _code: &SparseCode, _target: &TargetEvent) -> Option<DenseUpdateStats> {
        None
    }

    fn dense_report(&self) -> Option<DenseReport> {
        None
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EncoderKind {
    Hash,
    Competitive,
    Predictive,
    D,
    DNoTrace,
    DNoRoleProto,
    E0,
}

impl EncoderKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "hash" | "a" | "control" => Some(Self::Hash),
            "competitive" | "b" => Some(Self::Competitive),
            "predictive" | "c" => Some(Self::Predictive),
            "d" | "d-full" | "full-d" => Some(Self::D),
            "d-no-trace" | "d_notrace" => Some(Self::DNoTrace),
            "d-no-role-proto" | "d_noroleproto" => Some(Self::DNoRoleProto),
            "e0" | "encoder-e0" => Some(Self::E0),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EncoderConfig {
    pub feature_width: u32,
    pub active_bits: usize,
    pub surface_bits: usize,
    pub role_bits: usize,
    pub columns: usize,
    pub column_receptive_cap: usize,
    pub role_columns: usize,
    pub seed: u64,
    pub max_roles: usize,
    pub lr: f32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            feature_width: 65_536,
            active_bits: 16,
            surface_bits: 8,
            role_bits: 8,
            columns: 4096,
            column_receptive_cap: 16,
            role_columns: 2048,
            seed: 1,
            max_roles: 16,
            lr: 0.01,
        }
    }
}

pub fn build_encoder(kind: EncoderKind, cfg: EncoderConfig) -> Box<dyn SparseEncoder> {
    match kind {
        EncoderKind::Hash => Box::new(HashEncoder::new(cfg.feature_width, cfg.active_bits)),
        EncoderKind::Competitive => Box::new(CompetitiveEncoder::new(cfg, 1_000_000)),
        EncoderKind::Predictive => Box::new(PredictiveEncoder::new(cfg)),
        EncoderKind::D => Box::new(EncoderD::new(cfg, true, true)),
        EncoderKind::DNoTrace => Box::new(EncoderD::new(cfg, true, false)),
        EncoderKind::DNoRoleProto => Box::new(EncoderD::new(cfg, false, false)),
        EncoderKind::E0 => Box::new(EncoderE0::new(cfg)),
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

pub fn context_key(input: &InputEvent) -> u64 {
    mix64(
        ((input.context_token as u64) << 32)
            ^ ((input.prev_token as u64) << 16)
            ^ input.position_mod as u64,
    )
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
        let key = context_key(input);
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

        let key = context_key(input);
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

// ---------------------------------------------------------------------------
// Dense decoder (linear softmax readout from mean-pooled feature embeddings)
// ---------------------------------------------------------------------------

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[derive(Clone, Debug)]
pub struct DenseDecoder {
    pub embed_dim: usize,
    pub max_roles: usize,
    pub embeddings: HashMap<FeatureId, Vec<f32>>,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub learning_rate: f32,
    pub credit_sums: HashMap<FeatureId, f32>,
    pub credit_counts: HashMap<FeatureId, u64>,
    pub seed: u64,
}

impl DenseDecoder {
    pub fn new(embed_dim: usize, max_roles: usize, learning_rate: f32, seed: u64) -> Self {
        let mut weights = Vec::with_capacity(embed_dim);
        for i in 0..embed_dim {
            let mut row = Vec::with_capacity(max_roles);
            for r in 0..max_roles {
                let h = mix64(seed ^ (i as u64) ^ ((r as u64) << 16) ^ 0xDEAD_BEEF);
                row.push(((h % 1000) as f32 - 500.0) / 5000.0);
            }
            weights.push(row);
        }
        Self {
            embed_dim,
            max_roles,
            embeddings: HashMap::new(),
            weights,
            bias: vec![0.0f32; max_roles],
            learning_rate,
            credit_sums: HashMap::new(),
            credit_counts: HashMap::new(),
            seed: seed.wrapping_add(0xE0),
        }
    }

    pub fn init_embedding(&self, fid: FeatureId) -> Vec<f32> {
        let mut emb = Vec::with_capacity(self.embed_dim);
        for i in 0..self.embed_dim {
            let h = mix64(
                (fid.0 as u64) ^ self.seed ^ (i as u64).wrapping_mul(0x9e3779b97f4a7c15),
            );
            emb.push(((h % 1000) as f32 - 500.0) / 5000.0);
        }
        emb
    }

    pub fn mean_pool(&self, code: &SparseCode) -> Vec<f32> {
        let mut sum = vec![0.0f32; self.embed_dim];
        let mut count = 0usize;
        for f in code.as_slice() {
            if let Some(emb) = self.embeddings.get(f) {
                for i in 0..self.embed_dim {
                    sum[i] += emb[i];
                }
                count += 1;
            }
        }
        if count > 0 {
            let inv = 1.0 / count as f32;
            for i in 0..self.embed_dim {
                sum[i] *= inv;
            }
        }
        sum
    }

    pub fn forward(&self, embed: &[f32]) -> Vec<f32> {
        let mut logits = self.bias.clone();
        for i in 0..self.embed_dim {
            let e = embed[i];
            let w_row = &self.weights[i];
            for r in 0..self.max_roles {
                logits[r] += w_row[r] * e;
            }
        }
        softmax(&logits)
    }

    pub fn backward_and_update(
        &mut self,
        embed: &[f32],
        active_features: &[FeatureId],
        target_role: usize,
    ) -> (f32, f32) {
        let n_active = active_features.len().max(1);

        // Forward
        let mut logits = self.bias.clone();
        for i in 0..self.embed_dim {
            let e = embed[i];
            let w_row = &self.weights[i];
            for r in 0..self.max_roles {
                logits[r] += w_row[r] * e;
            }
        }
        let probs = softmax(&logits);
        let loss = -probs[target_role].max(1e-30).ln();

        // Gradient dL/d_logits[r] = probs[r] - (r == target_role)
        let mut d_logits = vec![0.0f32; self.max_roles];
        for r in 0..self.max_roles {
            d_logits[r] = probs[r] - if r == target_role { 1.0 } else { 0.0 };
        }

        // Gradient norm: weights + bias + embeddings
        let mut grad_norm = 0.0f32;
        for i in 0..self.embed_dim {
            for r in 0..self.max_roles {
                let dw = d_logits[r] * embed[i];
                grad_norm += dw * dw;
            }
        }
        for r in 0..self.max_roles {
            grad_norm += d_logits[r] * d_logits[r];
        }
        let mut d_embed = vec![0.0f32; self.embed_dim];
        for i in 0..self.embed_dim {
            for r in 0..self.max_roles {
                d_embed[i] += d_logits[r] * self.weights[i][r];
            }
        }
        for f in active_features {
            if self.embeddings.contains_key(f) {
                for i in 0..self.embed_dim {
                    let de = d_embed[i] / n_active as f32;
                    grad_norm += de * de;
                }
            }
        }
        grad_norm = grad_norm.sqrt();

        // SGD: weights
        for i in 0..self.embed_dim {
            let e = embed[i];
            for r in 0..self.max_roles {
                self.weights[i][r] -= self.learning_rate * d_logits[r] * e;
            }
        }

        // SGD: bias
        for r in 0..self.max_roles {
            self.bias[r] -= self.learning_rate * d_logits[r];
        }

        // SGD: feature embeddings
        for f in active_features {
            if let Some(emb) = self.embeddings.get_mut(f) {
                for i in 0..self.embed_dim {
                    emb[i] -= self.learning_rate * d_embed[i] / n_active as f32;
                }
            }
        }

        (loss, grad_norm)
    }

    pub fn compute_feature_credit(
        &self,
        code: &SparseCode,
        target: &TargetEvent,
    ) -> (f32, Vec<(FeatureId, f32)>) {
        let role = target.latent_role as usize;

        let full_embed = self.mean_pool(code);
        let full_probs = self.forward(&full_embed);
        let full_loss = -full_probs[role].max(1e-30).ln();

        let mut credits = Vec::new();
        for f in code.as_slice() {
            if !self.embeddings.contains_key(f) {
                continue;
            }
            let mut sum = vec![0.0f32; self.embed_dim];
            let mut count = 0usize;
            for other in code.as_slice() {
                if other == f {
                    continue;
                }
                if let Some(emb) = self.embeddings.get(other) {
                    for i in 0..self.embed_dim {
                        sum[i] += emb[i];
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let inv = 1.0 / count as f32;
                for i in 0..self.embed_dim {
                    sum[i] *= inv;
                }
            }
            let probs = self.forward(&sum);
            let loss_without = -probs[role].max(1e-30).ln();
            credits.push((*f, full_loss - loss_without));
        }

        (full_loss, credits)
    }
}

// ---------------------------------------------------------------------------
// Encoder E0 — predictive v2 sparse encoder + dense decoder (16-dim embeddings,
//               mean-pooling, online SGD, leave-one-out feature credit)
// ---------------------------------------------------------------------------

pub struct EncoderE0 {
    pub base: PredictiveEncoder,
    pub decoder: DenseDecoder,
}

impl EncoderE0 {
    pub fn new(cfg: EncoderConfig) -> Self {
        Self {
            base: PredictiveEncoder::new(cfg),
            decoder: DenseDecoder::new(16, cfg.max_roles, cfg.lr, cfg.seed),
        }
    }
}

impl SparseEncoder for EncoderE0 {
    fn name(&self) -> &'static str {
        "e0"
    }

    fn encode(&self, input: &InputEvent) -> SparseCode {
        self.base.encode(input)
    }

    fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode) {
        self.base.adapt(input, target, code);
    }

    fn dense_predict_prequential(&self, code: &SparseCode) -> Option<Vec<f32>> {
        let embed = self.decoder.mean_pool(code);
        Some(self.decoder.forward(&embed))
    }

    fn dense_adapt(&mut self, code: &SparseCode, target: &TargetEvent) -> Option<DenseUpdateStats> {
        for f in code.as_slice() {
            if !self.decoder.embeddings.contains_key(f) {
                let emb = self.decoder.init_embedding(*f);
                self.decoder.embeddings.insert(*f, emb);
            }
        }

        let embed = self.decoder.mean_pool(code);
        let active_features: Vec<FeatureId> = code.as_slice().to_vec();
        let (loss, grad_norm) =
            self.decoder
                .backward_and_update(&embed, &active_features, target.latent_role as usize);

        let (_, credits) = self.decoder.compute_feature_credit(code, target);
        for (fid, credit) in credits {
            *self.decoder.credit_sums.entry(fid).or_insert(0.0) += credit;
            *self.decoder.credit_counts.entry(fid).or_insert(0) += 1;
        }

        Some(DenseUpdateStats { loss, gradient_norm: grad_norm })
    }

    fn dense_report(&self) -> Option<DenseReport> {
        let mut avg_credits = HashMap::new();
        for (fid, sum) in &self.decoder.credit_sums {
            let count = self.decoder.credit_counts.get(fid).copied().unwrap_or(1).max(1);
            avg_credits.insert(*fid, sum / count as f32);
        }

        let mut weight_norm = 0.0f32;
        for i in 0..self.decoder.embed_dim {
            for r in 0..self.decoder.max_roles {
                weight_norm += self.decoder.weights[i][r] * self.decoder.weights[i][r];
            }
        }
        weight_norm = weight_norm.sqrt();

        let mut bias_norm = 0.0f32;
        for r in 0..self.decoder.max_roles {
            bias_norm += self.decoder.bias[r] * self.decoder.bias[r];
        }
        bias_norm = bias_norm.sqrt();

        Some(DenseReport {
            feature_embeddings: self.decoder.embeddings.clone(),
            feature_credits: avg_credits,
            weight_norm,
            bias_norm,
        })
    }
}

// ---------------------------------------------------------------------------
// Encoder D — dual-channel surface + role, anti-Hebbian, context traces
// ---------------------------------------------------------------------------

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

#[derive(Clone, Debug)]
pub struct EncoderD {
    // Surface columns (sparse projection, same as competitive v2)
    surface_columns: Vec<Column>,
    surface_bits: usize,
    surface_offset: u32,
    surface_total: u64,

    // Role prototype columns (projected from context + surface + traces)
    role_columns: Vec<RoleColumn>,
    role_bits: usize,
    role_offset: u32,
    role_total: u64,

    // Anti-Hebbian co-activation tracking: key = (a, b) where a < b
    co_activation: std::collections::HashMap<(usize, usize), u64>,

    // Context traces
    traces: Vec<ContextTrace>,
    max_traces: usize,

    // Role stats by context (for prototype adaptation)
    context_role_counts: std::collections::HashMap<u64, Vec<u32>>,
    max_roles: usize,

    // Config
    seed: u64,
    step: u64,

    // Ablation
    enable_role_prototypes: bool,
    enable_traces: bool,

    // Cached per step for adapt
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
            co_activation: std::collections::HashMap::new(),
            traces: Vec::new(),
            max_traces: (if role_protos { cfg.role_bits } else { cfg.surface_bits }).max(1) * 2,
            context_role_counts: std::collections::HashMap::new(),
            max_roles: cfg.max_roles,
            seed: cfg.seed,
            step: 0,
            enable_role_prototypes: role_protos,
            enable_traces: traces,
            last_surface_active: Vec::new(),
            last_role_active: Vec::new(),
        }
    }

    // --- surface column helpers (same logic as competitive v2) ---

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
        // Same as CompetitiveEncoder::sketch_terms
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

    // --- role column helpers ---

    fn role_projected_scores(&self, input: &InputEvent) -> Vec<i32> {
        if self.role_columns.is_empty() {
            return Vec::new();
        }
        let n = self.role_columns.len();
        let mut scores = vec![0i32; n];

        // Project from context features + surface active feature hints
        for salt in 0..12 {
            let h = mix64(context_key(input) ^ self.seed ^ 0x8000_0000u64 ^ salt as u64);
            let idx = (h % n as u64) as usize;
            if let Some(s) = scores.get_mut(idx) {
                *s = s.saturating_add(300);
            }
        }
        // Surface feature modulation
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
        // Trace modulation
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

    // --- anti-Hebbian (applied post-selection, only on active set) ---

    fn anti_hebbian_penalty_on_active(active: &[usize], offset: usize, co_activation: &std::collections::HashMap<(usize, usize), u64>) -> Vec<usize> {
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

    // --- context traces ---

    fn update_traces_prequential(&mut self, input: &InputEvent) {
        if !self.enable_traces || self.max_traces == 0 {
            return;
        }
        let ck = context_key(input);

        // Create trace on context-bearing tokens
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

        // Pay rent for all active traces
        for trace in &mut self.traces {
            if trace.active {
                trace.rent = trace.rent.saturating_add(1);
                // Deactivate if rent exceeds support + bootstrap
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
            // Trace matches by context_key similarity (exact match for first version)
            if trace.context_key == ck || (_input.context_token != 0 && (trace.context_key ^ ck) & 0xFFFF_FFFF_0000_0000 == 0) {
                trace.support_mass += 0.5;
                trace.evidence_mass += 0.5;
            } else if trace.context_key & 0xFFFF_FFFF_0000_0000 == ck & 0xFFFF_FFFF_0000_0000 {
                // Partial match — could be conflict
                trace.conflict_mass += 0.3;
            }
        }
    }

    // --- TopK selection ---

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

        // Surface features with anti-Hebbian penalization
        let surface_scores = self.surface_projected_scores(input);
        let mut surface_active = Self::select_topk(&surface_scores, self.surface_bits + 4);
        let penalized = Self::anti_hebbian_penalty_on_active(&surface_active, 0, &self.co_activation);
        surface_active.retain(|idx| !penalized.contains(idx));
        surface_active.truncate(self.surface_bits);
        for &idx in &surface_active {
            features.push(FeatureId(self.surface_offset + idx as u32));
        }

        // Role prototype features with anti-Hebbian penalization
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

        // --- update traces (prequential: before using target for trace creation) ---
        self.update_traces_prequential(input);

        // --- update surface columns ---
        let surface_scores = self.surface_projected_scores(input);
        self.last_surface_active = Self::select_topk(&surface_scores, self.surface_bits);
        for &idx in &self.last_surface_active {
            if let Some(col) = self.surface_columns.get_mut(idx) {
                col.usage = col.usage.saturating_add(1);
                self.surface_total = self.surface_total.saturating_add(1);
            }
        }

        // --- update role columns ---
        if self.enable_role_prototypes && !self.role_columns.is_empty() {
            let role_scores = self.role_projected_scores(input);
            self.last_role_active = Self::select_topk(&role_scores, self.role_bits);
            for &idx in &self.last_role_active {
                if let Some(col) = self.role_columns.get_mut(idx) {
                    col.usage = col.usage.saturating_add(1);
                    self.role_total = self.role_total.saturating_add(1);
                    col.role_counts[role] = col.role_counts[role].saturating_add(1);
                    if let Some((_dominant, best, second)) = dominant_role(&col.role_counts) {
                        // Allocate success mass based on role prediction confidence
                        if best > second.saturating_add(2) {
                            col.success_mass = (col.success_mass + 0.25).min(50.0);
                        }
                    }
                }
            }
            // Context role stats
            let ck = context_key(input);
            let max_roles = self.max_roles;
            self.context_role_counts.entry(ck)
                .or_insert_with(|| vec![0; max_roles])[role] =
                self.context_role_counts.get(&ck).map(|c| c[role]).unwrap_or(0).saturating_add(1);
        }

        // --- anti-Hebbian: track co-activation in this step ---
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
        // Prune old co-activation entries periodically
        if self.step % 1000 == 0 && self.co_activation.len() > 10000 {
            self.co_activation.retain(|_, v| *v > 20);
        }

        // --- post-observation trace update ---
        self.update_traces_post_observation(input, target);

        self.step = self.step.saturating_add(1);
    }
}
