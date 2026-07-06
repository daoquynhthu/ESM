use std::collections::HashSet;

use crate::feature::FeatureId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ClaimPhase {
    Open,
    Verified,
    Failed,
    Retired,
}

#[derive(Clone, Debug)]
pub struct PendingClaim {
    pub id: u64,
    pub cue_step: u64,
    pub issuer_features: Vec<FeatureId>,
    pub condition_key: u64,
    pub expected_role: Option<usize>,
    pub expected_evidence: Vec<FeatureId>,
    pub confidence: f32,
    pub phase: ClaimPhase,
    pub rent_paid: u32,
    pub verification_credit: f32,
    pub match_score: f32,
    pub is_probe: bool,
}

#[derive(Copy, Clone, Debug)]
pub struct ClaimConfig {
    pub max_open_claims: usize,
    pub claims_per_step: usize,
    pub probe_claims_per_step: usize,
    pub rent_per_step: f32,
    pub issue_confidence_floor: f32,
    pub verify_floor: f32,
    pub fail_floor: f32,
    pub verified_gain: f32,
    pub false_alarm_cost: f32,
    pub max_rent_before_retire: f32,
}

impl Default for ClaimConfig {
    fn default() -> Self {
        Self {
            max_open_claims: 256,
            claims_per_step: 8,
            probe_claims_per_step: 2,
            rent_per_step: 0.01,
            issue_confidence_floor: 0.55,
            verify_floor: 0.6,
            fail_floor: 0.4,
            verified_gain: 1.0,
            false_alarm_cost: 0.5,
            max_rent_before_retire: 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedClaim {
    pub issuer_features: Vec<FeatureId>,
    pub credit_role: usize,
}

#[derive(Clone, Debug)]
pub struct VerificationResult {
    pub verified: Vec<VerifiedClaim>,
    pub failed_count: usize,
}

#[derive(Clone, Debug)]
pub struct PendingClaimPool {
    claims: Vec<PendingClaim>,
    next_id: u64,
    issued_this_step: usize,
    probe_issued_this_step: usize,
    cfg: ClaimConfig,
    pub total_issued: u64,
    pub total_verified: u64,
    pub total_failed: u64,
    pub total_retired: u64,
}

fn jaccard_similarity(a: &[FeatureId], b: &[FeatureId]) -> f32 {
    let set_a: HashSet<FeatureId> = a.iter().copied().collect();
    let set_b: HashSet<FeatureId> = b.iter().copied().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

fn contradiction_score(expected: &[FeatureId], actual: &[FeatureId]) -> f32 {
    let set_expected: HashSet<FeatureId> = expected.iter().copied().collect();
    let set_actual: HashSet<FeatureId> = actual.iter().copied().collect();
    let missing = set_expected.difference(&set_actual).count();
    if expected.is_empty() { 0.0 } else { missing as f32 / expected.len() as f32 }
}

impl PendingClaimPool {
    pub fn new(cfg: ClaimConfig) -> Self {
        Self {
            claims: Vec::with_capacity(cfg.max_open_claims),
            next_id: 1,
            issued_this_step: 0,
            probe_issued_this_step: 0,
            cfg,
            total_issued: 0,
            total_verified: 0,
            total_failed: 0,
            total_retired: 0,
        }
    }

    pub fn begin_step(&mut self) {
        self.issued_this_step = 0;
        self.probe_issued_this_step = 0;
    }

    pub fn issue_template_claim(
        &mut self,
        cue_step: u64,
        issuer_features: &[FeatureId],
        condition_key: u64,
        expected_role: usize,
        expected_evidence: &[FeatureId],
        confidence: f32,
    ) -> Option<u64> {
        if self.issued_this_step >= self.cfg.claims_per_step {
            return None;
        }
        if self.open_count() >= self.cfg.max_open_claims {
            self.evict_worst_open();
            if self.open_count() >= self.cfg.max_open_claims {
                return None;
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        self.issued_this_step += 1;

        self.claims.push(PendingClaim {
            id,
            cue_step,
            issuer_features: issuer_features.to_vec(),
            condition_key,
            expected_role: Some(expected_role),
            expected_evidence: expected_evidence.to_vec(),
            confidence,
            phase: ClaimPhase::Open,
            rent_paid: 0,
            verification_credit: 0.0,
            match_score: 0.0,
            is_probe: false,
        });

        self.total_issued += 1;
        Some(id)
    }

    pub fn issue_probe_claim(
        &mut self,
        cue_step: u64,
        issuer_features: &[FeatureId],
        condition_key: u64,
        expected_evidence: &[FeatureId],
    ) -> Option<u64> {
        if self.probe_issued_this_step >= self.cfg.probe_claims_per_step {
            return None;
        }
        if self.issued_this_step >= self.cfg.claims_per_step {
            return None;
        }
        if self.open_count() >= self.cfg.max_open_claims {
            self.evict_worst_open();
            if self.open_count() >= self.cfg.max_open_claims {
                return None;
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        self.issued_this_step += 1;
        self.probe_issued_this_step += 1;

        self.claims.push(PendingClaim {
            id,
            cue_step,
            issuer_features: issuer_features.to_vec(),
            condition_key,
            expected_role: None,
            expected_evidence: expected_evidence.to_vec(),
            confidence: 0.5,
            phase: ClaimPhase::Open,
            rent_paid: 0,
            verification_credit: 0.0,
            match_score: 0.0,
            is_probe: true,
        });

        self.total_issued += 1;
        Some(id)
    }

    /// Verify claims from `cue_step`.
    /// Uses similarity between `current_evidence` (verify-step code) and each
    /// claim's `expected_evidence`. If match_score >= verify_floor → Verified,
    /// credit with `actual_role`. If contradiction_score >= fail_floor → Failed.
    /// Otherwise pays rent and stays open.
    pub fn verify_cue_step(
        &mut self,
        cue_step: u64,
        current_evidence: &[FeatureId],
        actual_role: usize,
    ) -> VerificationResult {
        let mut verified: Vec<VerifiedClaim> = Vec::new();
        let mut failed_count = 0usize;
        let mut to_remove: Vec<usize> = Vec::new();

        for (i, claim) in self.claims.iter_mut().enumerate() {
            if claim.cue_step != cue_step || claim.phase != ClaimPhase::Open {
                continue;
            }

            let ms = jaccard_similarity(current_evidence, &claim.expected_evidence);
            let cs = contradiction_score(&claim.expected_evidence, current_evidence);
            claim.match_score = ms;

            if ms >= self.cfg.verify_floor {
                claim.phase = ClaimPhase::Verified;
                let rent_cost = self.cfg.rent_per_step * claim.rent_paid as f32;
                claim.verification_credit = self.cfg.verified_gain - rent_cost;
                verified.push(VerifiedClaim {
                    issuer_features: claim.issuer_features.clone(),
                    credit_role: actual_role,
                });
                to_remove.push(i);
            } else if cs >= self.cfg.fail_floor {
                claim.phase = ClaimPhase::Failed;
                let rent_cost = self.cfg.rent_per_step * claim.rent_paid as f32;
                claim.verification_credit = -self.cfg.false_alarm_cost - rent_cost;
                failed_count += 1;
                to_remove.push(i);
            }
        }

        for i in to_remove.into_iter().rev() {
            self.claims.swap_remove(i);
        }

        self.total_verified += verified.len() as u64;
        self.total_failed += failed_count as u64;

        VerificationResult { verified, failed_count }
    }

    pub fn pay_rent(&mut self) {
        let mut to_remove: Vec<usize> = Vec::new();
        for (i, claim) in self.claims.iter_mut().enumerate() {
            if claim.phase == ClaimPhase::Open {
                claim.rent_paid += 1;
                let total_rent = self.cfg.rent_per_step * claim.rent_paid as f32;
                if total_rent >= self.cfg.max_rent_before_retire {
                    claim.phase = ClaimPhase::Retired;
                    self.total_retired += 1;
                    to_remove.push(i);
                }
            }
        }
        for i in to_remove.into_iter().rev() {
            self.claims.swap_remove(i);
        }
    }

    fn evict_worst_open(&mut self) {
        if self.claims.is_empty() {
            return;
        }
        if let Some(worst_idx) = self.claims.iter().enumerate()
            .filter(|(_, c)| c.phase == ClaimPhase::Open)
            .min_by(|(_, a), (_, b)| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.rent_paid.cmp(&b.rent_paid).reverse())
            })
            .map(|(i, _)| i)
        {
            self.claims.swap_remove(worst_idx);
            self.total_retired += 1;
        }
    }

    pub fn open_count(&self) -> usize {
        self.claims.iter().filter(|c| c.phase == ClaimPhase::Open).count()
    }

    pub fn total_claims(&self) -> usize {
        self.claims.len()
    }
}
