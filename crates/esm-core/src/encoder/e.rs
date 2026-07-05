//! Encoder E series — current experimental line.
//!
//! E0 wraps `PredictiveEncoder` with a dense diagnostic decoder:
//! 16-dim feature embeddings, mean-pooling, linear softmax readout, online SGD,
//! leave-one-out feature credit diagnostics.

use std::collections::HashMap;

use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};
use crate::rng::mix64;

use super::{DenseReport, DenseUpdateStats, EncoderConfig, PredictiveEncoder};
use crate::encoder::SparseEncoder;

// =========================================================================
// Dense decoder (linear softmax readout from mean-pooled feature embeddings)
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
        })
    }
}
