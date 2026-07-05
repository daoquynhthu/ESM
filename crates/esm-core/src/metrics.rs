//! E-1A representation diagnostics.
//!
//! Metrics are computed prequentially where possible: likelihoods are evaluated before the
//! diagnostic count tables are updated with the current target.

use std::collections::HashMap;

use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};

#[derive(Clone, Debug)]
pub struct RoleCounts {
    pub counts: Vec<u64>,
    pub total: u64,
}

impl RoleCounts {
    pub fn new(max_roles: usize) -> Self {
        Self { counts: vec![0; max_roles], total: 0 }
    }

    fn nll(&self, role: usize) -> f64 {
        let k = self.counts.len() as f64;
        let count = self.counts.get(role).copied().unwrap_or(0) as f64;
        let total = self.total as f64;
        -((count + 1.0) / (total + k)).ln()
    }

    pub fn update(&mut self, role: usize) {
        if let Some(x) = self.counts.get_mut(role) {
            *x = x.saturating_add(1);
            self.total = self.total.saturating_add(1);
        }
    }
}

#[derive(Clone, Debug)]
struct CodeSample {
    token: u32,
    role: u32,
    code: SparseCode,
}

#[derive(Clone, Debug)]
pub struct E1aMetrics {
    max_roles: usize,
    sample_limit: usize,
    n: u64,
    token_nll_sum: f64,
    code_nll_sum: f64,
    feature_vote_nll_sum: f64,
    feature_vote_nll_no_proto_sum: f64,
    token_role: HashMap<u32, RoleCounts>,
    code_role: HashMap<u64, RoleCounts>,
    feature_role: HashMap<FeatureId, RoleCounts>,
    feature_usage: HashMap<FeatureId, u64>,
    active_bits_sum: u64,
    dense_nll_sum: f64,
    dense_nll_count: u64,
    samples: Vec<CodeSample>,
    proto_feature_offset: u32,
    proto_feature_end: u32,
}

#[derive(Clone, Debug)]
pub struct E1aReport {
    pub encoder: String,
    pub stream: String,
    pub steps: u64,
    pub token_nll: f64,
    pub code_nll: f64,
    pub feature_vote_nll: f64,
    pub feature_vote_nll_no_proto: f64,
    pub controlled_predictive_info: f64,
    pub controlled_feature_predictive_info: f64,
    pub controlled_feature_predictive_info_no_proto: f64,
    pub same_token_context_split: f64,
    pub role_sharing: f64,
    pub code_entropy: f64,
    pub active_bits_avg: f64,
    pub unique_features: usize,
    pub dense_nll: f64,
    pub dense_cpi: f64,
    pub embedding_role_separation: f64,
    // E1 attention diagnostics
    pub attention_mass_base: f64,
    pub attention_mass_proto: f64,
    pub top_credit_1: f64,
    pub top_credit_3: f64,
    pub top_credit_5: f64,
    pub dense_cpi_without_top1: f64,
    pub dense_cpi_without_top3: f64,
    pub dense_cpi_without_top5: f64,
    pub attention_credit_corr: f64,
}

impl E1aMetrics {
    pub fn new(max_roles: usize, sample_limit: usize) -> Self {
        Self::with_prototype_range(max_roles, sample_limit, 0, 0)
    }

    pub fn with_prototype_range(max_roles: usize, sample_limit: usize, proto_offset: u32, proto_end: u32) -> Self {
        Self {
            max_roles,
            sample_limit,
            n: 0,
            token_nll_sum: 0.0,
            code_nll_sum: 0.0,
            feature_vote_nll_sum: 0.0,
            feature_vote_nll_no_proto_sum: 0.0,
            token_role: HashMap::new(),
            code_role: HashMap::new(),
            feature_role: HashMap::new(),
            feature_usage: HashMap::new(),
            active_bits_sum: 0,
            dense_nll_sum: 0.0,
            dense_nll_count: 0,
            samples: Vec::with_capacity(sample_limit.min(1024)),
            proto_feature_offset: proto_offset,
            proto_feature_end: proto_end,
        }
    }

    pub fn observe_prequential(&mut self, input: &InputEvent, target: &TargetEvent, code: &SparseCode) {
        let role = (target.latent_role as usize) % self.max_roles;
        let code_sig = code.signature();

        let token_nll = self
            .token_role
            .get(&input.token)
            .map(|c| c.nll(role))
            .unwrap_or_else(|| RoleCounts::new(self.max_roles).nll(role));
        let code_nll = self
            .code_role
            .get(&code_sig)
            .map(|c| c.nll(role))
            .unwrap_or_else(|| RoleCounts::new(self.max_roles).nll(role));
        let feature_vote_nll = self.feature_vote_nll(code, role);
        let feature_vote_nll_no_proto = self.feature_vote_nll_filtered(code, role);

        self.token_nll_sum += token_nll;
        self.code_nll_sum += code_nll;
        self.feature_vote_nll_sum += feature_vote_nll;
        self.feature_vote_nll_no_proto_sum += feature_vote_nll_no_proto;
        self.n = self.n.saturating_add(1);
        self.active_bits_sum = self.active_bits_sum.saturating_add(code.len() as u64);

        self.token_role
            .entry(input.token)
            .or_insert_with(|| RoleCounts::new(self.max_roles))
            .update(role);
        self.code_role
            .entry(code_sig)
            .or_insert_with(|| RoleCounts::new(self.max_roles))
            .update(role);

        for f in code.as_slice() {
            *self.feature_usage.entry(*f).or_insert(0) += 1;
            self.feature_role
                .entry(*f)
                .or_insert_with(|| RoleCounts::new(self.max_roles))
                .update(role);
        }

        if self.samples.len() < self.sample_limit {
            self.samples.push(CodeSample { token: input.token, role: target.latent_role, code: code.clone() });
        }
    }

    pub fn observe_dense_prequential(&mut self, target: &TargetEvent, probs: &[f32]) {
        let role = (target.latent_role as usize) % self.max_roles;
        let p = probs.get(role).copied().unwrap_or(0.0).max(1e-10) as f64;
        self.dense_nll_sum += -p.ln();
        self.dense_nll_count = self.dense_nll_count.saturating_add(1);
    }

    pub fn feature_role_counts(&self) -> &HashMap<FeatureId, RoleCounts> {
        &self.feature_role
    }

    pub fn report(&self, encoder: &str, stream: &str) -> E1aReport {
        let n = self.n.max(1) as f64;
        let token_nll = self.token_nll_sum / n;
        let code_nll = self.code_nll_sum / n;
        let feature_vote_nll = self.feature_vote_nll_sum / n;
        let feature_vote_nll_no_proto = self.feature_vote_nll_no_proto_sum / n;
        let has_dense = self.dense_nll_count > 0;
        let dense_nll = if has_dense {
            self.dense_nll_sum / self.dense_nll_count as f64
        } else {
            0.0
        };
        E1aReport {
            encoder: encoder.to_string(),
            stream: stream.to_string(),
            steps: self.n,
            token_nll,
            code_nll,
            feature_vote_nll,
            feature_vote_nll_no_proto,
            controlled_predictive_info: token_nll - code_nll,
            controlled_feature_predictive_info: token_nll - feature_vote_nll,
            controlled_feature_predictive_info_no_proto: token_nll - feature_vote_nll_no_proto,
            same_token_context_split: self.same_token_context_split(),
            role_sharing: self.role_sharing(),
            code_entropy: self.code_entropy(),
            active_bits_avg: self.active_bits_sum as f64 / n,
            unique_features: self.feature_usage.len(),
            dense_nll,
            dense_cpi: if has_dense { token_nll - dense_nll } else { 0.0 },
            embedding_role_separation: 0.0,
            attention_mass_base: 0.0,
            attention_mass_proto: 0.0,
            top_credit_1: 0.0,
            top_credit_3: 0.0,
            top_credit_5: 0.0,
            dense_cpi_without_top1: 0.0,
            dense_cpi_without_top3: 0.0,
            dense_cpi_without_top5: 0.0,
            attention_credit_corr: 0.0,
        }
    }


    fn feature_vote_nll(&self, code: &SparseCode, role: usize) -> f64 {
        let mut role_mass = vec![1.0f64; self.max_roles];
        let mut total_mass = self.max_roles as f64;
        for f in code.as_slice() {
            if let Some(counts) = self.feature_role.get(f) {
                for (r, count) in counts.counts.iter().copied().enumerate() {
                    if let Some(mass) = role_mass.get_mut(r) {
                        *mass += count as f64;
                    }
                }
                total_mass += counts.total as f64;
            }
        }
        let p = role_mass.get(role).copied().unwrap_or(1.0) / total_mass.max(1.0);
        -p.ln()
    }

    fn feature_vote_nll_filtered(&self, code: &SparseCode, role: usize) -> f64 {
        if self.proto_feature_offset >= self.proto_feature_end {
            // No prototype range configured, return the full version
            return self.feature_vote_nll(code, role);
        }
        let mut role_mass = vec![1.0f64; self.max_roles];
        let mut total_mass = self.max_roles as f64;
        for f in code.as_slice() {
            if f.0 >= self.proto_feature_offset && f.0 < self.proto_feature_end {
                continue; // Skip prototype-only features
            }
            if let Some(counts) = self.feature_role.get(f) {
                for (r, count) in counts.counts.iter().copied().enumerate() {
                    if let Some(mass) = role_mass.get_mut(r) {
                        *mass += count as f64;
                    }
                }
                total_mass += counts.total as f64;
            }
        }
        let p = role_mass.get(role).copied().unwrap_or(1.0) / total_mass.max(1.0);
        -p.ln()
    }

    fn same_token_context_split(&self) -> f64 {
        let mut total = 0.0;
        let mut pairs = 0usize;
        for i in 0..self.samples.len() {
            for j in (i + 1)..self.samples.len() {
                let a = &self.samples[i];
                let b = &self.samples[j];
                if a.token == b.token && a.role != b.role {
                    total += 1.0 - a.code.jaccard(&b.code);
                    pairs += 1;
                }
            }
        }
        if pairs == 0 { 0.0 } else { total / pairs as f64 }
    }

    fn role_sharing(&self) -> f64 {
        let mut total = 0.0;
        let mut pairs = 0usize;
        for i in 0..self.samples.len() {
            for j in (i + 1)..self.samples.len() {
                let a = &self.samples[i];
                let b = &self.samples[j];
                if a.role == b.role && a.token != b.token {
                    total += a.code.jaccard(&b.code);
                    pairs += 1;
                }
            }
        }
        if pairs == 0 { 0.0 } else { total / pairs as f64 }
    }

    fn code_entropy(&self) -> f64 {
        let total: u64 = self.feature_usage.values().sum();
        if total == 0 { return 0.0; }
        let total_f = total as f64;
        self.feature_usage
            .values()
            .map(|c| {
                let p = *c as f64 / total_f;
                -p * p.ln()
            })
            .sum()
    }
}

impl E1aReport {
    pub fn set_embedding_role_separation(&mut self, val: f64) {
        self.embedding_role_separation = val;
    }

    pub fn set_e1_diagnostics(
        &mut self,
        mass_base: f64,
        mass_proto: f64,
        top_c1: f64,
        top_c3: f64,
        top_c5: f64,
        cpi_wo_1: f64,
        cpi_wo_3: f64,
        cpi_wo_5: f64,
        corr: f64,
    ) {
        self.attention_mass_base = mass_base;
        self.attention_mass_proto = mass_proto;
        self.top_credit_1 = top_c1;
        self.top_credit_3 = top_c3;
        self.top_credit_5 = top_c5;
        self.dense_cpi_without_top1 = cpi_wo_1;
        self.dense_cpi_without_top3 = cpi_wo_3;
        self.dense_cpi_without_top5 = cpi_wo_5;
        self.attention_credit_corr = corr;
    }

    pub fn to_json_pretty(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"encoder\": \"{}\",\n",
                "  \"stream\": \"{}\",\n",
                "  \"steps\": {},\n",
                "  \"token_nll\": {:.8},\n",
                "  \"code_nll\": {:.8},\n",
                "  \"feature_vote_nll\": {:.8},\n",
                "  \"feature_vote_nll_no_proto\": {:.8},\n",
                "  \"controlled_predictive_info\": {:.8},\n",
                "  \"controlled_feature_predictive_info\": {:.8},\n",
                "  \"controlled_feature_predictive_info_no_proto\": {:.8},\n",
                "  \"same_token_context_split\": {:.8},\n",
                "  \"role_sharing\": {:.8},\n",
                "  \"code_entropy\": {:.8},\n",
                "  \"active_bits_avg\": {:.8},\n",
                "  \"unique_features\": {},\n",
                "  \"dense_nll\": {:.8},\n",
                "  \"dense_cpi\": {:.8},\n",
                "  \"embedding_role_separation\": {:.8},\n",
                "  \"attention_mass_base\": {:.8},\n",
                "  \"attention_mass_proto\": {:.8},\n",
                "  \"top_credit_1\": {:.8},\n",
                "  \"top_credit_3\": {:.8},\n",
                "  \"top_credit_5\": {:.8},\n",
                "  \"dense_cpi_without_top1\": {:.8},\n",
                "  \"dense_cpi_without_top3\": {:.8},\n",
                "  \"dense_cpi_without_top5\": {:.8},\n",
                "  \"attention_credit_corr\": {:.8}\n",
                "}}"
            ),
            escape_json(&self.encoder),
            escape_json(&self.stream),
            self.steps,
            self.token_nll,
            self.code_nll,
            self.feature_vote_nll,
            self.feature_vote_nll_no_proto,
            self.controlled_predictive_info,
            self.controlled_feature_predictive_info,
            self.controlled_feature_predictive_info_no_proto,
            self.same_token_context_split,
            self.role_sharing,
            self.code_entropy,
            self.active_bits_avg,
            self.unique_features,
            self.dense_nll,
            self.dense_cpi,
            self.embedding_role_separation,
            self.attention_mass_base,
            self.attention_mass_proto,
            self.top_credit_1,
            self.top_credit_3,
            self.top_credit_5,
            self.dense_cpi_without_top1,
            self.dense_cpi_without_top3,
            self.dense_cpi_without_top5,
            self.attention_credit_corr
        )
    }
}

/// Measures how well the learned dense embeddings separate different roles.
/// Computes mean pairwise cosine distance between majority-role group centroids.
pub fn compute_embedding_role_separation(
    embeddings: &HashMap<FeatureId, Vec<f32>>,
    feature_role_counts: &HashMap<FeatureId, RoleCounts>,
    max_roles: usize,
) -> f64 {
    if embeddings.is_empty() || max_roles < 2 {
        return 0.0;
    }

    let mut feature_role: HashMap<FeatureId, usize> = HashMap::new();
    for (fid, rc) in feature_role_counts {
        if !embeddings.contains_key(fid) {
            continue;
        }
        let mut best = 0usize;
        let mut best_count = 0u64;
        for (r, c) in rc.counts.iter().enumerate() {
            if *c > best_count {
                best_count = *c;
                best = r;
            }
        }
        if best_count > 0 {
            feature_role.insert(*fid, best);
        }
    }

    let mut role_embeddings: Vec<Vec<Vec<f32>>> = vec![Vec::new(); max_roles];
    for (fid, role) in &feature_role {
        if let Some(emb) = embeddings.get(fid) {
            role_embeddings[*role].push(emb.clone());
        }
    }

    let dim = embeddings.values().next().map(|e| e.len()).unwrap_or(1);
    if dim == 0 {
        return 0.0;
    }

    let role_means: Vec<Option<Vec<f64>>> = role_embeddings
        .iter()
        .map(|embs| {
            if embs.is_empty() {
                return None;
            }
            let mut mean = vec![0.0f64; dim];
            for emb in embs {
                for i in 0..dim {
                    mean[i] += emb[i] as f64;
                }
            }
            let inv = 1.0 / embs.len() as f64;
            for i in 0..dim {
                mean[i] *= inv;
            }
            Some(mean)
        })
        .collect();

    let non_empty: Vec<usize> = role_means
        .iter()
        .enumerate()
        .filter(|(_, m)| m.is_some())
        .map(|(i, _)| i)
        .collect();

    if non_empty.len() < 2 {
        return 0.0;
    }

    let mut total_dist = 0.0f64;
    let mut count = 0usize;
    for a in 0..non_empty.len() {
        let r1 = non_empty[a];
        let m1 = role_means[r1].as_ref().unwrap();
        for b in (a + 1)..non_empty.len() {
            let r2 = non_empty[b];
            let m2 = role_means[r2].as_ref().unwrap();
            let mut dot = 0.0f64;
            let mut n1 = 0.0f64;
            let mut n2 = 0.0f64;
            for i in 0..dim {
                dot += m1[i] * m2[i];
                n1 += m1[i] * m1[i];
                n2 += m2[i] * m2[i];
            }
            let sim = dot / (n1.sqrt() * n2.sqrt()).max(1e-10);
            total_dist += 1.0 - sim;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_dist / count as f64
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
