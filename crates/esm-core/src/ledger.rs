use std::collections::VecDeque;

use crate::feature::{FeatureId, SparseCode};

#[derive(Clone, Debug)]
struct LedgerEntry {
    step: u64,
    features: Vec<FeatureId>,
}

#[derive(Clone, Debug)]
pub struct CausalLedger {
    entries: VecDeque<LedgerEntry>,
    max_len: usize,
}

impl CausalLedger {
    pub fn new(max_len: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_len),
            max_len,
        }
    }

    pub fn record(&mut self, step: u64, code: &SparseCode) {
        if self.entries.len() >= self.max_len {
            self.entries.pop_front();
        }
        self.entries.push_back(LedgerEntry {
            step,
            features: code.as_slice().to_vec(),
        });
    }

    pub fn features_at(&self, step: u64) -> Option<&[FeatureId]> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.step == step)
            .map(|e| e.features.as_slice())
    }
}
