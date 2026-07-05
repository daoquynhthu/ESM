//! Sparse feature IDs and sparse codes.

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FeatureId(pub u32);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SparseCode {
    features: Vec<FeatureId>,
}

impl SparseCode {
    pub fn new(mut features: Vec<FeatureId>) -> Self {
        features.sort_unstable();
        features.dedup();
        Self { features }
    }

    pub fn empty() -> Self {
        Self { features: Vec::new() }
    }

    pub fn as_slice(&self) -> &[FeatureId] {
        &self.features
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn signature(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for f in &self.features {
            h ^= f.0 as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    pub fn jaccard(&self, other: &Self) -> f64 {
        if self.features.is_empty() && other.features.is_empty() {
            return 1.0;
        }
        let mut i = 0;
        let mut j = 0;
        let mut intersection = 0usize;
        let mut union = 0usize;
        while i < self.features.len() || j < other.features.len() {
            match (self.features.get(i), other.features.get(j)) {
                (Some(a), Some(b)) if a == b => {
                    intersection += 1;
                    union += 1;
                    i += 1;
                    j += 1;
                }
                (Some(a), Some(b)) if a < b => {
                    union += 1;
                    i += 1;
                }
                (Some(_), Some(_)) => {
                    union += 1;
                    j += 1;
                }
                (Some(_), None) => {
                    union += 1;
                    i += 1;
                }
                (None, Some(_)) => {
                    union += 1;
                    j += 1;
                }
                (None, None) => break,
            }
        }
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}
