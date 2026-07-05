//! Encoder E series — current experimental line.
//!
//! E0 wraps `PredictiveEncoder` with a dense diagnostic decoder:
//! 16-dim feature embeddings, mean-pooling, linear softmax readout, online SGD,
//! leave-one-out feature credit diagnostics.
//!
//! E1 series wraps `PredictiveEncoder` with an attention-weighted or mean-pooled
//! decoder, with either linear softmax or one-hidden-layer MLP readout, plus
//! credit-gated utility shaping and diagnostic attention metrics.

use std::collections::HashMap;

use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};
use crate::rng::mix64;

use super::{DenseReport, DenseUpdateStats, EncoderConfig, PredictiveEncoder};
use crate::encoder::SparseEncoder;

// =========================================================================
// Utilities
// =========================================================================

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

// =========================================================================
// Pooling and readout mode enums
// =========================================================================

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PoolingMode {
    Mean,
    Attention { top_m: usize },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReadoutMode {
    Linear,
    MLP { hidden_dim: usize },
}

// =========================================================================
// Attention-weighted decoder (supports all 4 pooling × readout combos)
// =========================================================================

#[derive(Clone, Debug)]
pub struct AttentionDecoder {
    pub embed_dim: usize,
    pub max_roles: usize,
    pub pooling: PoolingMode,
    pub readout: ReadoutMode,
    pub learning_rate: f32,

    pub embeddings: HashMap<FeatureId, Vec<f32>>,
    pub attention_key: Vec<f32>,

    // Linear readout:  W[embed_dim][max_roles], bias[max_roles]
    // MLP readout:     W1[embed_dim][hidden_dim], b1[hidden_dim],
    //                  W2[hidden_dim][max_roles], b2[max_roles]
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,

    // Credit tracking (accumulated leave-one-out)
    pub credit_sums: HashMap<FeatureId, f32>,
    pub credit_counts: HashMap<FeatureId, u64>,

    // Attention diagnostic accumulators
    pub attention_mass_base_sum: f64,
    pub attention_mass_proto_sum: f64,
    pub attention_mass_count: u64,

    // Per-step attention data for post-hoc diagnostics (sampled steps)
    pub attention_samples: Vec<AttentionStep>,

    pub seed: u64,
    pub proto_feature_offset: u32,
    pub proto_feature_end: u32,
}

#[derive(Clone, Debug)]
pub struct AttentionStep {
    pub features: Vec<FeatureId>,
    pub weights: Vec<f32>,
    pub target_role: u32,
    pub pooled: Vec<f32>,
}

impl AttentionDecoder {
    pub fn new(
        embed_dim: usize,
        max_roles: usize,
        learning_rate: f32,
        pooling: PoolingMode,
        readout: ReadoutMode,
        seed: u64,
        proto_offset: u32,
        proto_end: u32,
    ) -> Self {
        let hidden_dim = match readout {
            ReadoutMode::Linear => max_roles,
            ReadoutMode::MLP { hidden_dim } => hidden_dim,
        };

        let mut w1 = Vec::with_capacity(embed_dim);
        for i in 0..embed_dim {
            let mut row = Vec::with_capacity(hidden_dim);
            for h in 0..hidden_dim {
                let hval = mix64(seed ^ (i as u64) ^ ((h as u64) << 16) ^ 0xE1A1_0000);
                row.push(((hval % 1000) as f32 - 500.0) / 5000.0);
            }
            w1.push(row);
        }
        let b1 = vec![0.0f32; hidden_dim];

        let mut w2 = Vec::with_capacity(hidden_dim);
        for h in 0..hidden_dim {
            let mut row = Vec::with_capacity(max_roles);
            for r in 0..max_roles {
                let hval = mix64(seed ^ (h as u64) ^ ((r as u64) << 16) ^ 0xE1B2_0000);
                row.push(((hval % 1000) as f32 - 500.0) / 5000.0);
            }
            w2.push(row);
        }
        let b2 = vec![0.0f32; max_roles];

        Self {
            embed_dim,
            max_roles,
            pooling,
            readout,
            learning_rate,
            embeddings: HashMap::new(),
            attention_key: Self::init_key(embed_dim, seed),
            w1,
            b1,
            w2,
            b2,
            credit_sums: HashMap::new(),
            credit_counts: HashMap::new(),
            attention_mass_base_sum: 0.0,
            attention_mass_proto_sum: 0.0,
            attention_mass_count: 0,
            attention_samples: Vec::new(),
            seed: seed.wrapping_add(0xE1),
            proto_feature_offset: proto_offset,
            proto_feature_end: proto_end,
        }
    }

    fn init_key(dim: usize, seed: u64) -> Vec<f32> {
        let mut key = Vec::with_capacity(dim);
        for i in 0..dim {
            let h = mix64(seed ^ (i as u64) ^ 0xE1C3_0000);
            key.push(((h % 1000) as f32 - 500.0) / 5000.0);
        }
        key
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

    /// Pool active feature embeddings into a single vector.
    /// Returns (pooled, vec of (feature, weight) pairs).
    pub fn pool(&self, code: &SparseCode) -> (Vec<f32>, Vec<(FeatureId, f32)>) {
        let active: Vec<(FeatureId, &Vec<f32>)> = code
            .as_slice()
            .iter()
            .filter_map(|f| self.embeddings.get(f).map(|e| (*f, e)))
            .collect();
        if active.is_empty() {
            return (vec![0.0f32; self.embed_dim], Vec::new());
        }
        match self.pooling {
            PoolingMode::Mean => self.mean_pool_inner(&active),
            PoolingMode::Attention { top_m } => self.attn_pool_inner(&active, top_m),
        }
    }

    fn mean_pool_inner(
        &self,
        active: &[(FeatureId, &Vec<f32>)],
    ) -> (Vec<f32>, Vec<(FeatureId, f32)>) {
        let n = active.len().max(1);
        let inv = 1.0 / n as f32;
        let mut pooled = vec![0.0f32; self.embed_dim];
        let mut weights = Vec::with_capacity(active.len());
        for (fid, emb) in active {
            for i in 0..self.embed_dim {
                pooled[i] += emb[i] * inv;
            }
            weights.push((*fid, inv));
        }
        (pooled, weights)
    }

    fn attn_pool_inner(
        &self,
        active: &[(FeatureId, &Vec<f32>)],
        top_m: usize,
    ) -> (Vec<f32>, Vec<(FeatureId, f32)>) {
        let n = active.len();
        let inv_sqrt_d = (1.0 / self.embed_dim as f32).sqrt();

        // Compute scores
        let mut scored: Vec<(usize, f32)> = active
            .iter()
            .enumerate()
            .map(|(idx, (_, emb))| {
                let s: f32 = emb.iter().zip(self.attention_key.iter()).map(|(e, k)| e * k).sum::<f32>()
                    * inv_sqrt_d;
                (idx, s)
            })
            .collect();

        // Select top-m
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_m.min(n);
        scored.truncate(k);

        // Softmax over top-m scores
        let max_s = scored.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scored.iter().map(|(_, s)| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let alphas: Vec<f32> = if sum_exp > 1e-30 {
            exps.iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / k as f32; k]
        };

        // Weighted sum
        let mut pooled = vec![0.0f32; self.embed_dim];
        let mut weights = Vec::with_capacity(k);
        for (i, (orig_idx, _)) in scored.iter().enumerate() {
            let (fid, emb) = active[*orig_idx];
            for j in 0..self.embed_dim {
                pooled[j] += alphas[i] * emb[j];
            }
            weights.push((fid, alphas[i]));
        }
        (pooled, weights)
    }

    /// Forward pass: pooled_embed -> logits -> softmax.
    pub fn forward(&self, embed: &[f32]) -> Vec<f32> {
        match self.readout {
            ReadoutMode::Linear => {
                let mut logits = vec![0.0f32; self.max_roles];
                for r in 0..self.max_roles {
                    logits[r] = self.b2[r]; // w2/b2 used as linear weights when readout=Linear
                    for i in 0..self.embed_dim {
                        logits[r] += embed[i] * self.w2[i][r];
                    }
                }
                softmax(&logits)
            }
            ReadoutMode::MLP { hidden_dim } => {
                let mut hidden = vec![0.0f32; hidden_dim];
                for h in 0..hidden_dim {
                    let mut val = self.b1[h];
                    for i in 0..self.embed_dim {
                        val += embed[i] * self.w1[i][h];
                    }
                    hidden[h] = relu(val);
                }
                let mut logits = vec![0.0f32; self.max_roles];
                for r in 0..self.max_roles {
                    logits[r] = self.b2[r];
                    for h in 0..hidden_dim {
                        logits[r] += hidden[h] * self.w2[h][r];
                    }
                }
                softmax(&logits)
            }
        }
    }

    /// Backward and update all parameters.
    /// Returns (loss, grad_norm).
    pub fn backward_and_update(
        &mut self,
        embed: &[f32],
        active_input: &[FeatureId],
        _attention_weights: &[(FeatureId, f32)],
        target_role: usize,
    ) -> (f32, f32) {
        let n_active = active_input.len().max(1);

        // Forward
        let probs = self.forward(embed);
        let loss = -probs[target_role].max(1e-30).ln();

        // Cross-entropy gradient
        let mut d_logits = vec![0.0f32; self.max_roles];
        for r in 0..self.max_roles {
            d_logits[r] = probs[r] - if r == target_role { 1.0 } else { 0.0 };
        }

        let mut grad_norm = 0.0f32;

        match self.readout {
            ReadoutMode::Linear => {
                // w2 is embed_dim x max_roles, b2 is max_roles
                // d_w2[i][r] = d_logits[r] * embed[i]
                // d_b2[r] = d_logits[r]
                // d_embed[i] = sum_r d_logits[r] * w2[i][r]
                for i in 0..self.embed_dim {
                    for r in 0..self.max_roles {
                        let dw = d_logits[r] * embed[i];
                        grad_norm += dw * dw;
                        self.w2[i][r] -= self.learning_rate * dw;
                    }
                }
                for r in 0..self.max_roles {
                    grad_norm += d_logits[r] * d_logits[r];
                    self.b2[r] -= self.learning_rate * d_logits[r];
                }
            }
            ReadoutMode::MLP { hidden_dim } => {
                // Hidden forward (redo to get hidden values)
                let mut hidden = vec![0.0f32; hidden_dim];
                let mut pre_act = vec![0.0f32; hidden_dim];
                for h in 0..hidden_dim {
                    let mut val = self.b1[h];
                    for i in 0..self.embed_dim {
                        val += embed[i] * self.w1[i][h];
                    }
                    pre_act[h] = val;
                    hidden[h] = relu(val);
                }

                // d_hidden[h] = sum_r d_logits[r] * w2[h][r] (from OLD w2)
                let mut d_hidden = vec![0.0f32; hidden_dim];
                for h in 0..hidden_dim {
                    for r in 0..self.max_roles {
                        d_hidden[h] += d_logits[r] * self.w2[h][r];
                    }
                }
                // d_w2[h][r] = d_logits[r] * hidden[h], then update w2
                for h in 0..hidden_dim {
                    for r in 0..self.max_roles {
                        let dw = d_logits[r] * hidden[h];
                        grad_norm += dw * dw;
                        self.w2[h][r] -= self.learning_rate * dw;
                    }
                }
                for r in 0..self.max_roles {
                    grad_norm += d_logits[r] * d_logits[r];
                    self.b2[r] -= self.learning_rate * d_logits[r];
                }

                // ReLU backprop
                for h in 0..hidden_dim {
                    if pre_act[h] <= 0.0 {
                        d_hidden[h] = 0.0;
                    }
                }

                // d_embed_from_readout[i] = sum_h d_hidden[h] * w1[i][h] (from OLD w1)
                let mut d_embed_readout = vec![0.0f32; self.embed_dim];
                for i in 0..self.embed_dim {
                    for h in 0..hidden_dim {
                        d_embed_readout[i] += d_hidden[h] * self.w1[i][h];
                    }
                }
                // d_w1[i][h] = d_hidden[h] * embed[i], then update w1
                for i in 0..self.embed_dim {
                    for h in 0..hidden_dim {
                        let dw = d_hidden[h] * embed[i];
                        grad_norm += dw * dw;
                        self.w1[i][h] -= self.learning_rate * dw;
                    }
                }
                for h in 0..hidden_dim {
                    grad_norm += d_hidden[h] * d_hidden[h];
                    self.b1[h] -= self.learning_rate * d_hidden[h];
                }

            }
        }

        // For embedding and attention updates, we need d_pooled = gradient w.r.t. pooled embedding.
        // For linear: d_pooled[i] = sum_r d_logits[r] * w2[i][r]
        // For MLP: d_pooled[i] = sum_h d_hidden[h] * w1[i][h]
        let d_pooled: Vec<f32> = match self.readout {
            ReadoutMode::Linear => {
                let mut dp = vec![0.0f32; self.embed_dim];
                for i in 0..self.embed_dim {
                    for r in 0..self.max_roles {
                        dp[i] += d_logits[r] * self.w2[i][r];
                    }
                }
                dp
            }
            ReadoutMode::MLP { hidden_dim } => {
                let mut hidden = vec![0.0f32; hidden_dim];
                let mut pre_act = vec![0.0f32; hidden_dim];
                for h in 0..hidden_dim {
                    let mut val = self.b1[h];
                    for i in 0..self.embed_dim {
                        val += embed[i] * self.w1[i][h];
                    }
                    pre_act[h] = val;
                    hidden[h] = relu(val);
                }
                let mut d_hidden = vec![0.0f32; hidden_dim];
                for h in 0..hidden_dim {
                    for r in 0..self.max_roles {
                        d_hidden[h] += d_logits[r] * self.w2[h][r];
                    }
                }
                for h in 0..hidden_dim {
                    if pre_act[h] <= 0.0 {
                        d_hidden[h] = 0.0;
                    }
                }
                let mut dp = vec![0.0f32; self.embed_dim];
                for i in 0..self.embed_dim {
                    for h in 0..hidden_dim {
                        dp[i] += d_hidden[h] * self.w1[i][h];
                    }
                }
                dp
            }
        };

        // Attention backprop
        let inv_sqrt_d = (1.0 / self.embed_dim as f32).sqrt();
        // rebuild attention weights from stored data — we have them in attention_weights
        // dL/ds_i = alpha_i * (dot(d_pooled, emb_i) - sum_j alpha_j * dot(d_pooled, emb_j))
        // only for top-m features
        let mut d_key = vec![0.0f32; self.embed_dim];
        let mut d_embed_for_update: HashMap<FeatureId, Vec<f32>> = HashMap::new();

        match self.pooling {
            PoolingMode::Mean => {
                // For mean pooling: pooled = (1/n) * sum emb_i
                // d_pooled[i] contributes equally to each active embedding
                // d_emb_i[j] = d_pooled[j] / n_active
                let inv_n = 1.0 / n_active as f32;
                for f in active_input {
                    if self.embeddings.contains_key(f) {
                        let mut de = vec![0.0f32; self.embed_dim];
                        for i in 0..self.embed_dim {
                            de[i] = d_pooled[i] * inv_n;
                            grad_norm += de[i] * de[i];
                        }
                        d_embed_for_update.insert(*f, de);
                    }
                }
            }
            PoolingMode::Attention { top_m } => {
                let active_embs: Vec<(FeatureId, &Vec<f32>)> = active_input
                    .iter()
                    .filter_map(|f| self.embeddings.get(f).map(|e| (*f, e)))
                    .collect();
                let n = active_embs.len();
                let k = top_m.min(n);

                // Recompute scores
                let mut scored: Vec<(usize, f32)> = active_embs
                    .iter()
                    .enumerate()
                    .map(|(idx, (_, emb))| {
                        let s: f32 = emb.iter().zip(self.attention_key.iter())
                            .map(|(e, k)| e * k).sum::<f32>() * inv_sqrt_d;
                        (idx, s)
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(k);

                let max_s = scored.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scored.iter().map(|(_, s)| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let alphas: Vec<f32> = if sum_exp > 1e-30 {
                    exps.iter().map(|e| e / sum_exp).collect()
                } else {
                    vec![1.0 / k as f32; k]
                };

                // dL/dalpha_i * alpha_j stuff
                // First compute weighted_sum = sum_j alpha_j * dot(d_pooled, emb_j)
                let mut weighted_dot_sum = 0.0f32;
                for (i, (orig_idx, _)) in scored.iter().enumerate() {
                    let (_, emb) = active_embs[*orig_idx];
                    let dot: f32 = d_pooled.iter().zip(emb.iter()).map(|(d, e)| d * e).sum();
                    weighted_dot_sum += alphas[i] * dot;
                }

                // For each top-m feature:
                // dL/ds_i = alpha_i * (dot(d_pooled, emb_i) - weighted_dot_sum)
                let mut d_scores: HashMap<usize, f32> = HashMap::new();
                for (i, (orig_idx, _)) in scored.iter().enumerate() {
                    let (_, emb) = active_embs[*orig_idx];
                    let dot: f32 = d_pooled.iter().zip(emb.iter()).map(|(d, e)| d * e).sum();
                    let ds = alphas[i] * (dot - weighted_dot_sum);
                    d_scores.insert(*orig_idx, ds);
                }

                // dL/dkey[k] = sum_i dL/ds_i * emb_i[k] / sqrt(d)
                // dL/demb_i = dL/ds_i * key / sqrt(d) + alpha_i * d_pooled
                for (orig_idx, (fid, emb)) in active_embs.iter().enumerate() {
                    let ds = d_scores.get(&orig_idx).copied().unwrap_or(0.0);
                    let alpha = if let Some(pos) = scored.iter().position(|(oi, _)| *oi == orig_idx) {
                        alphas[pos]
                    } else {
                        0.0
                    };

                    let mut de = vec![0.0f32; self.embed_dim];
                    for i in 0..self.embed_dim {
                        d_key[i] += ds * emb[i] * inv_sqrt_d;
                        de[i] = ds * self.attention_key[i] * inv_sqrt_d + alpha * d_pooled[i];
                        grad_norm += de[i] * de[i];
                    }
                    d_embed_for_update.insert(*fid, de);
                }
            }
        }

        // Update attention key
        for i in 0..self.embed_dim {
            let dk = d_key[i];
            grad_norm += dk * dk;
            self.attention_key[i] -= self.learning_rate * dk;
        }

        // Update embeddings
        for f in active_input {
            if let Some(de) = d_embed_for_update.get(f) {
                if let Some(emb) = self.embeddings.get_mut(f) {
                    for i in 0..self.embed_dim {
                        emb[i] -= self.learning_rate * de[i];
                    }
                }
            }
        }

        grad_norm = grad_norm.sqrt();

        (loss, grad_norm)
    }

    /// Track attention mass by feature type for diagnostics.
    pub fn accumulate_attention_diagnostics(
        &mut self,
        weights: &[(FeatureId, f32)],
    ) {
        let mut base_mass = 0.0f64;
        let mut proto_mass = 0.0f64;
        for (fid, w) in weights {
            let w_f64 = *w as f64;
            if self.proto_feature_offset > 0
                && fid.0 >= self.proto_feature_offset
                && fid.0 < self.proto_feature_end
            {
                proto_mass += w_f64;
            } else {
                base_mass += w_f64;
            }
        }
        self.attention_mass_base_sum += base_mass;
        self.attention_mass_proto_sum += proto_mass;
        self.attention_mass_count = self.attention_mass_count.saturating_add(1);
    }

    /// Store attention step for post-hoc diagnostics.
    pub fn store_attention_step(
        &mut self,
        features: Vec<FeatureId>,
        weights: Vec<f32>,
        target_role: u32,
        pooled: Vec<f32>,
        max_samples: usize,
    ) {
        if self.attention_samples.len() < max_samples {
            self.attention_samples.push(AttentionStep {
                features,
                weights,
                target_role,
                pooled,
            });
        }
    }

    /// Compute feature credit using leave-one-out, recomputing attention for each held-out feature.
    pub fn compute_feature_credit(
        &self,
        code: &SparseCode,
        target: &TargetEvent,
    ) -> (f32, Vec<(FeatureId, f32)>) {
        let role = target.latent_role as usize;

        let (full_embed, _) = self.pool(code);
        let full_probs = self.forward(&full_embed);
        let full_loss = -full_probs[role].max(1e-30).ln();

        let mut credits = Vec::new();
        let features: Vec<FeatureId> = code.as_slice().iter().filter(|f| self.embeddings.contains_key(f)).copied().collect();

        for &f in &features {
            let without_code = SparseCode::new(
                code.as_slice().iter().filter(|&&x| x != f).copied().collect()
            );
            let (embed_without, _) = self.pool(&without_code);
            let probs = self.forward(&embed_without);
            let loss_without = -probs[role].max(1e-30).ln();
            credits.push((f, full_loss - loss_without));
        }

        (full_loss, credits)
    }

    /// Compute attention–credit Pearson correlation over stored samples.
    pub fn compute_attention_credit_correlation(&self) -> f64 {
        let mut pairs = Vec::new();
        for step in &self.attention_samples {
            for (fid, w) in step.features.iter().zip(step.weights.iter()) {
                let count = self.credit_counts.get(fid).copied().unwrap_or(0);
                if count == 0 {
                    continue;
                }
                let avg_credit =
                    self.credit_sums.get(fid).copied().unwrap_or(0.0) / count as f32;
                pairs.push((*w as f64, avg_credit as f64));
            }
        }
        if pairs.len() < 3 {
            return 0.0;
        }
        let n = pairs.len() as f64;
        let sum_w: f64 = pairs.iter().map(|(w, _)| w).sum();
        let sum_c: f64 = pairs.iter().map(|(_, c)| c).sum();
        let sum_ww: f64 = pairs.iter().map(|(w, _)| w * w).sum();
        let sum_cc: f64 = pairs.iter().map(|(_, c)| c * c).sum();
        let sum_wc: f64 = pairs.iter().map(|(w, c)| w * c).sum();
        let num = n * sum_wc - sum_w * sum_c;
        let den = ((n * sum_ww - sum_w * sum_w) * (n * sum_cc - sum_c * sum_c)).sqrt();
        if den < 1e-10 { 0.0 } else { num / den }
    }

    /// Compute average NLL when the top-N attended features are removed.
    /// Returns (avg_nll_without_top1, avg_nll_without_top3, avg_nll_without_top5).
    pub fn compute_nll_without_top(
        &self,
        samples: &[AttentionStep],
    ) -> (f64, f64, f64) {
        let mut nll_sum_1 = 0.0f64;
        let mut nll_sum_3 = 0.0f64;
        let mut nll_sum_5 = 0.0f64;
        let mut cnt_1 = 0u64;
        let mut cnt_3 = 0u64;
        let mut cnt_5 = 0u64;

        for step in samples {
            if step.features.len() < 2 {
                continue;
            }
            let k = step.features.len();

            let mut scored: Vec<(f32, FeatureId)> = step
                .features
                .iter()
                .copied()
                .zip(step.weights.iter().copied())
                .map(|(f, w)| (w, f))
                .collect();
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            if k >= 2 {
                let filt: Vec<FeatureId> = scored.iter().skip(1).map(|(_, f)| *f).collect();
                let code = SparseCode::new(filt);
                let (embed, _) = self.pool(&code);
                let probs = self.forward(&embed);
                let p = probs.get(step.target_role as usize).copied().unwrap_or(0.0).max(1e-10) as f64;
                nll_sum_1 += -p.ln();
                cnt_1 += 1;
            }

            if k >= 4 {
                let filt: Vec<FeatureId> = scored.iter().skip(3).map(|(_, f)| *f).collect();
                let code = SparseCode::new(filt);
                let (embed, _) = self.pool(&code);
                let probs = self.forward(&embed);
                let p = probs.get(step.target_role as usize).copied().unwrap_or(0.0).max(1e-10) as f64;
                nll_sum_3 += -p.ln();
                cnt_3 += 1;
            }

            if k >= 6 {
                let filt: Vec<FeatureId> = scored.iter().skip(5).map(|(_, f)| *f).collect();
                let code = SparseCode::new(filt);
                let (embed, _) = self.pool(&code);
                let probs = self.forward(&embed);
                let p = probs.get(step.target_role as usize).copied().unwrap_or(0.0).max(1e-10) as f64;
                nll_sum_5 += -p.ln();
                cnt_5 += 1;
            }
        }

        (
            if cnt_1 > 0 { nll_sum_1 / cnt_1 as f64 } else { 0.0 },
            if cnt_3 > 0 { nll_sum_3 / cnt_3 as f64 } else { 0.0 },
            if cnt_5 > 0 { nll_sum_5 / cnt_5 as f64 } else { 0.0 },
        )
    }

    /// Compute average attention weight on top-1, top-3, top-5 features across stored samples.
    pub fn compute_top_attention_credits(&self, samples: &[AttentionStep]) -> (f64, f64, f64) {
        let mut top1_credit_sum = 0.0f64;
        let mut top3_credit_sum = 0.0f64;
        let mut top5_credit_sum = 0.0f64;
        let mut count_1 = 0u64;
        let mut count_3 = 0u64;
        let mut count_5 = 0u64;

        for step in samples {
            if step.features.is_empty() {
                continue;
            }
            let mut scored: Vec<(f32, FeatureId)> = step.features.iter().copied()
                .zip(step.weights.iter().copied()).map(|(f, w)| (w, f)).collect();
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Top-1
            if let Some((_, fid)) = scored.first() {
                let cnt = self.credit_counts.get(fid).copied().unwrap_or(0);
                if cnt > 0 {
                    let avg = self.credit_sums.get(fid).copied().unwrap_or(0.0) / cnt as f32;
                    top1_credit_sum += avg as f64;
                    count_1 += 1;
                }
            }

            // Top-3
            for (_, fid) in scored.iter().take(3) {
                let cnt = self.credit_counts.get(fid).copied().unwrap_or(0);
                if cnt > 0 {
                    let avg = self.credit_sums.get(fid).copied().unwrap_or(0.0) / cnt as f32;
                    top3_credit_sum += avg as f64;
                    count_3 += 1;
                }
            }

            // Top-5
            for (_, fid) in scored.iter().take(5) {
                let cnt = self.credit_counts.get(fid).copied().unwrap_or(0);
                if cnt > 0 {
                    let avg = self.credit_sums.get(fid).copied().unwrap_or(0.0) / cnt as f32;
                    top5_credit_sum += avg as f64;
                    count_5 += 1;
                }
            }
        }

        (
            if count_1 > 0 { top1_credit_sum / count_1 as f64 } else { 0.0 },
            if count_3 > 0 { top3_credit_sum / count_3 as f64 } else { 0.0 },
            if count_5 > 0 { top5_credit_sum / count_5 as f64 } else { 0.0 },
        )
    }
}

// =========================================================================
// Dense decoder (linear softmax readout from mean-pooled feature embeddings)
// Kept for E0 reproducibility.
// =========================================================================

// DenseDecoder kept for E0 reproducibility; uses mean pooling + linear readout.

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

        let mut d_logits = vec![0.0f32; self.max_roles];
        for r in 0..self.max_roles {
            d_logits[r] = probs[r] - if r == target_role { 1.0 } else { 0.0 };
        }

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

        for i in 0..self.embed_dim {
            let e = embed[i];
            for r in 0..self.max_roles {
                self.weights[i][r] -= self.learning_rate * d_logits[r] * e;
            }
        }
        for r in 0..self.max_roles {
            self.bias[r] -= self.learning_rate * d_logits[r];
        }
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

// =========================================================================
// Encoder E0 — predictive v2 + dense decoder
// =========================================================================

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
            attention_mass_base: None,
            attention_mass_proto: None,
            top_credit_1: None,
            top_credit_3: None,
            top_credit_5: None,
            attention_credit_corr: None,
            nll_without_top1: None,
            nll_without_top3: None,
            nll_without_top5: None,
            attention_samples: None,
        })
    }
}

// =========================================================================
// Encoder E1 — attention/mean-pooled decoder + optional MLP readout
// =========================================================================

macro_rules! impl_encoder_e1 {
    ($name:ident, $encoder_kind:literal, $pooling:expr, $readout:expr) => {
        pub struct $name {
            pub base: PredictiveEncoder,
            pub decoder: AttentionDecoder,
        }

        impl $name {
            pub fn new(cfg: EncoderConfig) -> Self {
                let proto_off = 3_000_000;
                let proto_end = proto_off + cfg.max_roles as u32;
                Self {
                    base: PredictiveEncoder::new(cfg),
                    decoder: AttentionDecoder::new(
                        16,
                        cfg.max_roles,
                        cfg.lr,
                        $pooling,
                        $readout,
                        cfg.seed,
                        proto_off,
                        proto_end,
                    ),
                }
            }
        }

        impl SparseEncoder for $name {
            fn name(&self) -> &'static str {
                $encoder_kind
            }

            fn encode(&self, input: &InputEvent) -> SparseCode {
                self.base.encode(input)
            }

            fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode) {
                self.base.adapt(input, target, code);
            }

            fn dense_predict_prequential(&self, code: &SparseCode) -> Option<Vec<f32>> {
                let (embed, _) = self.decoder.pool(code);
                Some(self.decoder.forward(&embed))
            }

            fn dense_adapt(&mut self, code: &SparseCode, target: &TargetEvent) -> Option<DenseUpdateStats> {
                for f in code.as_slice() {
                    if !self.decoder.embeddings.contains_key(f) {
                        let emb = self.decoder.init_embedding(*f);
                        self.decoder.embeddings.insert(*f, emb);
                    }
                }

                let (embed, attn_weights) = self.decoder.pool(code);
                let active_features: Vec<FeatureId> = code.as_slice().to_vec();
                let (loss, grad_norm) = self.decoder.backward_and_update(
                    &embed,
                    &active_features,
                    &attn_weights,
                    target.latent_role as usize,
                );

                // Accumulate attention diagnostics
                self.decoder.accumulate_attention_diagnostics(&attn_weights);
                self.decoder.store_attention_step(
                    code.as_slice().to_vec(),
                    attn_weights.iter().map(|(_, w)| *w).collect(),
                    target.latent_role,
                    embed,
                    4096,
                );

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

                // Collect attention diagnostics
                let attn_mass_count = self.decoder.attention_mass_count.max(1);
                let attn_mass_base = self.decoder.attention_mass_base_sum / attn_mass_count as f64;
                let attn_mass_proto = self.decoder.attention_mass_proto_sum / attn_mass_count as f64;

                let top_credits = self.decoder.compute_top_attention_credits(&self.decoder.attention_samples);
                let corr = self.decoder.compute_attention_credit_correlation();
                let (nll_wo_1, nll_wo_3, nll_wo_5) = self.decoder.compute_nll_without_top(&self.decoder.attention_samples);

                Some(DenseReport {
                    feature_embeddings: self.decoder.embeddings.clone(),
                    feature_credits: avg_credits,
                    weight_norm: 0.0,
                    bias_norm: 0.0,
                    attention_mass_base: Some(attn_mass_base),
                    attention_mass_proto: Some(attn_mass_proto),
                    top_credit_1: Some(top_credits.0),
                    top_credit_3: Some(top_credits.1),
                    top_credit_5: Some(top_credits.2),
                    attention_credit_corr: Some(corr),
                    nll_without_top1: Some(nll_wo_1),
                    nll_without_top3: Some(nll_wo_3),
                    nll_without_top5: Some(nll_wo_5),
                    attention_samples: Some(self.decoder.attention_samples.clone()),
                })
            }
        }
    };
}

// E1a: attention + linear
impl_encoder_e1!(EncoderE1a, "e1-attn-linear", PoolingMode::Attention { top_m: 8 }, ReadoutMode::Linear);

// E1b: mean + MLP
impl_encoder_e1!(EncoderE1b, "e1-mean-mlp", PoolingMode::Mean, ReadoutMode::MLP { hidden_dim: 32 });

// E1c: attention + MLP
impl_encoder_e1!(EncoderE1c, "e1-attn-mlp", PoolingMode::Attention { top_m: 8 }, ReadoutMode::MLP { hidden_dim: 32 });

// =========================================================================
// Encoder E2 — credit-gated sparse encoder shaping
// =========================================================================

/// E2 shaping mode.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum E2Mode {
    /// Promote features with positive leave-one-out credit
    CreditPromote,
    /// Promote positive-credit, suppress negative-credit features
    CreditPromoteSuppress,
    /// Use global loss delta instead of per-feature credit (uniform boost/penalty)
    NoLoo,
}

const SHAPE_BIAS: i32 = 100;

macro_rules! impl_encoder_e2 {
    ($name:ident, $encoder_kind:literal, $mode:expr) => {
        pub struct $name {
            pub base: PredictiveEncoder,
            pub decoder: AttentionDecoder,
        }

        impl $name {
            pub fn new(cfg: EncoderConfig) -> Self {
                let proto_off = 3_000_000;
                let proto_end = proto_off + cfg.max_roles as u32;
                Self {
                    base: PredictiveEncoder::new(cfg),
                    decoder: AttentionDecoder::new(
                        16,
                        cfg.max_roles,
                        cfg.lr,
                        PoolingMode::Attention { top_m: 8 },
                        ReadoutMode::MLP { hidden_dim: 32 },
                        cfg.seed,
                        proto_off,
                        proto_end,
                    ),
                }
            }
        }

        impl SparseEncoder for $name {
            fn name(&self) -> &'static str {
                $encoder_kind
            }

            fn encode(&self, input: &InputEvent) -> SparseCode {
                self.base.encode(input)
            }

            fn adapt(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode) {
                self.base.adapt(input, target, code);
            }

            fn dense_predict_prequential(&self, code: &SparseCode) -> Option<Vec<f32>> {
                let (embed, _) = self.decoder.pool(code);
                Some(self.decoder.forward(&embed))
            }

            fn dense_adapt(&mut self, code: &SparseCode, target: &TargetEvent) -> Option<DenseUpdateStats> {
                for f in code.as_slice() {
                    if !self.decoder.embeddings.contains_key(f) {
                        let emb = self.decoder.init_embedding(*f);
                        self.decoder.embeddings.insert(*f, emb);
                    }
                }

                let (embed, attn_weights) = self.decoder.pool(code);
                let active_features: Vec<FeatureId> = code.as_slice().to_vec();
                let (loss, grad_norm) = self.decoder.backward_and_update(
                    &embed,
                    &active_features,
                    &attn_weights,
                    target.latent_role as usize,
                );

                self.decoder.accumulate_attention_diagnostics(&attn_weights);
                self.decoder.store_attention_step(
                    code.as_slice().to_vec(),
                    attn_weights.iter().map(|(_, w)| *w).collect(),
                    target.latent_role,
                    embed,
                    4096,
                );

                // Credit-gated encoder shaping
                let base_off = self.base.base.feature_offset;
                let n_cols = self.base.base.columns.len();

                match $mode {
                    E2Mode::CreditPromote => {
                        let (_, credits) = self.decoder.compute_feature_credit(code, target);
                        for (fid, credit) in credits {
                            *self.decoder.credit_sums.entry(fid).or_insert(0.0) += credit;
                            *self.decoder.credit_counts.entry(fid).or_insert(0) += 1;

                            if fid.0 >= base_off && fid.0 < base_off + n_cols as u32 {
                                let idx = (fid.0 - base_off) as usize;
                                if let Some(col) = self.base.base.columns.get_mut(idx) {
                                    if credit > 0.0 {
                                        col.credit_bias = col.credit_bias.saturating_add(SHAPE_BIAS);
                                    }
                                }
                            }
                        }
                    }
                    E2Mode::CreditPromoteSuppress => {
                        let (_, credits) = self.decoder.compute_feature_credit(code, target);
                        for (fid, credit) in credits {
                            *self.decoder.credit_sums.entry(fid).or_insert(0.0) += credit;
                            *self.decoder.credit_counts.entry(fid).or_insert(0) += 1;

                            if fid.0 >= base_off && fid.0 < base_off + n_cols as u32 {
                                let idx = (fid.0 - base_off) as usize;
                                if let Some(col) = self.base.base.columns.get_mut(idx) {
                                    if credit > 0.0 {
                                        col.credit_bias = col.credit_bias.saturating_add(SHAPE_BIAS);
                                    } else if credit < 0.0 {
                                        col.credit_bias = col.credit_bias.saturating_sub(SHAPE_BIAS);
                                    }
                                }
                            }
                        }
                    }
                    E2Mode::NoLoo => {
                        let baseline = (self.base.max_roles as f32).ln();
                        let improvement = baseline - loss;
                        for fid in code.as_slice() {
                            if fid.0 >= base_off && fid.0 < base_off + n_cols as u32 {
                                let idx = (fid.0 - base_off) as usize;
                                if let Some(col) = self.base.base.columns.get_mut(idx) {
                                    if improvement > 0.0 {
                                        col.credit_bias = col.credit_bias.saturating_add(SHAPE_BIAS);
                                    } else {
                                        col.credit_bias = col.credit_bias.saturating_sub(SHAPE_BIAS);
                                    }
                                }
                            }
                        }
                    }
                }

                Some(DenseUpdateStats { loss, gradient_norm: grad_norm })
            }

            fn dense_report(&self) -> Option<DenseReport> {
                let mut avg_credits = HashMap::new();
                for (fid, sum) in &self.decoder.credit_sums {
                    let count = self.decoder.credit_counts.get(fid).copied().unwrap_or(1).max(1);
                    avg_credits.insert(*fid, sum / count as f32);
                }

                let attn_mass_count = self.decoder.attention_mass_count.max(1);
                let attn_mass_base = self.decoder.attention_mass_base_sum / attn_mass_count as f64;
                let attn_mass_proto = self.decoder.attention_mass_proto_sum / attn_mass_count as f64;

                let top_credits = self.decoder.compute_top_attention_credits(&self.decoder.attention_samples);
                let corr = self.decoder.compute_attention_credit_correlation();
                let (nll_wo_1, nll_wo_3, nll_wo_5) = self.decoder.compute_nll_without_top(&self.decoder.attention_samples);

                Some(DenseReport {
                    feature_embeddings: self.decoder.embeddings.clone(),
                    feature_credits: avg_credits,
                    weight_norm: 0.0,
                    bias_norm: 0.0,
                    attention_mass_base: Some(attn_mass_base),
                    attention_mass_proto: Some(attn_mass_proto),
                    top_credit_1: Some(top_credits.0),
                    top_credit_3: Some(top_credits.1),
                    top_credit_5: Some(top_credits.2),
                    attention_credit_corr: Some(corr),
                    nll_without_top1: Some(nll_wo_1),
                    nll_without_top3: Some(nll_wo_3),
                    nll_without_top5: Some(nll_wo_5),
                    attention_samples: Some(self.decoder.attention_samples.clone()),
                })
            }
        }
    };
}

impl_encoder_e2!(EncoderE2a, "e2-credit-promote", E2Mode::CreditPromote);
impl_encoder_e2!(EncoderE2b, "e2-credit-promote-suppress", E2Mode::CreditPromoteSuppress);
impl_encoder_e2!(EncoderE2c, "e2-no-loo", E2Mode::NoLoo);
