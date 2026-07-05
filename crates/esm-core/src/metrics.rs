//! E-1A representation diagnostics.
//!
//! Metrics are computed prequentially where possible: likelihoods are evaluated before the
//! diagnostic count tables are updated with the current target.

use std::collections::HashMap;

use crate::event::{InputEvent, TargetEvent};
use crate::feature::{FeatureId, SparseCode};

#[derive(Clone, Debug)]
struct RoleCounts {
    counts: Vec<u64>,
    total: u64,
}

impl RoleCounts {
    fn new(max_roles: usize) -> Self {
        Self { counts: vec![0; max_roles], total: 0 }
    }

    fn nll(&self, role: usize) -> f64 {
        let k = self.counts.len() as f64;
        let count = self.counts.get(role).copied().unwrap_or(0) as f64;
        let total = self.total as f64;
        -((count + 1.0) / (total + k)).ln()
    }

    fn update(&mut self, role: usize) {
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
    token_role: HashMap<u32, RoleCounts>,
    code_role: HashMap<u64, RoleCounts>,
    feature_role: HashMap<FeatureId, RoleCounts>,
    feature_usage: HashMap<FeatureId, u64>,
    active_bits_sum: u64,
    samples: Vec<CodeSample>,
}

#[derive(Clone, Debug)]
pub struct E1aReport {
    pub encoder: String,
    pub stream: String,
    pub steps: u64,
    pub token_nll: f64,
    pub code_nll: f64,
    pub feature_vote_nll: f64,
    pub controlled_predictive_info: f64,
    pub controlled_feature_predictive_info: f64,
    pub same_token_context_split: f64,
    pub role_sharing: f64,
    pub code_entropy: f64,
    pub active_bits_avg: f64,
    pub unique_features: usize,
}

impl E1aMetrics {
    pub fn new(max_roles: usize, sample_limit: usize) -> Self {
        Self {
            max_roles,
            sample_limit,
            n: 0,
            token_nll_sum: 0.0,
            code_nll_sum: 0.0,
            feature_vote_nll_sum: 0.0,
            token_role: HashMap::new(),
            code_role: HashMap::new(),
            feature_role: HashMap::new(),
            feature_usage: HashMap::new(),
            active_bits_sum: 0,
            samples: Vec::with_capacity(sample_limit.min(1024)),
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

        self.token_nll_sum += token_nll;
        self.code_nll_sum += code_nll;
        self.feature_vote_nll_sum += feature_vote_nll;
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

    pub fn report(&self, encoder: &str, stream: &str) -> E1aReport {
        let n = self.n.max(1) as f64;
        let token_nll = self.token_nll_sum / n;
        let code_nll = self.code_nll_sum / n;
        let feature_vote_nll = self.feature_vote_nll_sum / n;
        E1aReport {
            encoder: encoder.to_string(),
            stream: stream.to_string(),
            steps: self.n,
            token_nll,
            code_nll,
            feature_vote_nll,
            controlled_predictive_info: token_nll - code_nll,
            controlled_feature_predictive_info: token_nll - feature_vote_nll,
            same_token_context_split: self.same_token_context_split(),
            role_sharing: self.role_sharing(),
            code_entropy: self.code_entropy(),
            active_bits_avg: self.active_bits_sum as f64 / n,
            unique_features: self.feature_usage.len(),
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
                "  \"controlled_predictive_info\": {:.8},\n",
                "  \"controlled_feature_predictive_info\": {:.8},\n",
                "  \"same_token_context_split\": {:.8},\n",
                "  \"role_sharing\": {:.8},\n",
                "  \"code_entropy\": {:.8},\n",
                "  \"active_bits_avg\": {:.8},\n",
                "  \"unique_features\": {}\n",
                "}}"
            ),
            escape_json(&self.encoder),
            escape_json(&self.stream),
            self.steps,
            self.token_nll,
            self.code_nll,
            self.feature_vote_nll,
            self.controlled_predictive_info,
            self.controlled_feature_predictive_info,
            self.same_token_context_split,
            self.role_sharing,
            self.code_entropy,
            self.active_bits_avg,
            self.unique_features
        )
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
