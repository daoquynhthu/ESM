use std::collections::HashMap;

use esm_core::claims::{ClaimConfig, PendingClaimPool};
use esm_core::encoder::{build_encoder, EncoderConfig, EncoderKind};

use crate::stream::{build_stream, StreamKind};

#[derive(Copy, Clone, Debug)]
pub struct E1bConfig {
    pub stream: StreamKind,
    pub encoder: EncoderKind,
    pub steps: u64,
    pub seed: u64,
    pub active_bits: usize,
    pub columns: usize,
    pub max_roles: usize,
    pub claim_gap: usize,
    pub claim_max_open: usize,
    pub claim_per_step: usize,
    pub claim_probe_per_step: usize,
    pub claim_rent: f32,
    pub claim_verified_gain: f32,
    pub claim_false_alarm_cost: f32,
    pub claim_verify_floor: f32,
    pub claim_fail_floor: f32,
}

impl Default for E1bConfig {
    fn default() -> Self {
        Self {
            stream: StreamKind::DelayedCue,
            encoder: EncoderKind::Predictive,
            steps: 50_000,
            seed: 1,
            active_bits: 16,
            columns: 4096,
            max_roles: 16,
            claim_gap: 0,
            claim_max_open: 256,
            claim_per_step: 8,
            claim_probe_per_step: 2,
            claim_rent: 0.01,
            claim_verified_gain: 1.0,
            claim_false_alarm_cost: 0.5,
            claim_verify_floor: 0.6,
            claim_fail_floor: 0.4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct E1bReport {
    pub encoder: String,
    pub stream: String,
    pub steps: u64,
    pub claim_enabled: bool,
    pub verify_step_nll: f64,
    pub verify_step_accuracy: f64,
    pub voting_nll_at_verify: f64,
    pub voting_accuracy_at_verify: f64,
    pub cue_step_nll: f64,
    pub cue_step_accuracy: f64,
    pub overall_nll: f64,
    pub cue_to_verify_cpi: f64,
    pub avg_cue_features: f64,
    pub avg_verify_features: f64,
    pub avg_shared_features: f64,
    pub claim_verified_rate: f64,
    pub claim_false_alarm_rate: f64,
    pub claim_total_issued: u64,
    pub claim_survival_count: u64,
}

impl E1bReport {
    pub fn to_json_pretty(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"encoder\": \"{}\",\n",
                "  \"stream\": \"{}\",\n",
                "  \"steps\": {},\n",
                "  \"claim_enabled\": {},\n",
                "  \"verify_step_nll\": {:.8},\n",
                "  \"verify_step_accuracy\": {:.8},\n",
                "  \"voting_nll_at_verify\": {:.8},\n",
                "  \"voting_accuracy_at_verify\": {:.8},\n",
                "  \"cue_step_nll\": {:.8},\n",
                "  \"cue_step_accuracy\": {:.8},\n",
                "  \"overall_nll\": {:.8},\n",
                "  \"cue_to_verify_cpi\": {:.8},\n",
                "  \"avg_cue_features\": {:.2},\n",
                "  \"avg_verify_features\": {:.2},\n",
                "  \"avg_shared_features\": {:.4},\n",
                "  \"claim_verified_rate\": {:.6},\n",
                "  \"claim_false_alarm_rate\": {:.6},\n",
                "  \"claim_total_issued\": {},\n",
                "  \"claim_survival_count\": {}\n",
                "}}"
            ),
            self.encoder,
            self.stream,
            self.steps,
            self.claim_enabled,
            self.verify_step_nll,
            self.verify_step_accuracy,
            self.voting_nll_at_verify,
            self.voting_accuracy_at_verify,
            self.cue_step_nll,
            self.cue_step_accuracy,
            self.overall_nll,
            self.cue_to_verify_cpi,
            self.avg_cue_features,
            self.avg_verify_features,
            self.avg_shared_features,
            self.claim_verified_rate,
            self.claim_false_alarm_rate,
            self.claim_total_issued,
            self.claim_survival_count,
        )
    }
}

pub fn run_e1b(cfg: E1bConfig) -> E1bReport {
    let use_claims = cfg.claim_gap > 0;

    let enc_cfg = EncoderConfig {
        active_bits: cfg.active_bits,
        columns: cfg.columns,
        seed: cfg.seed,
        max_roles: cfg.max_roles,
        ..EncoderConfig::default()
    };
    let mut encoder = build_encoder(cfg.encoder, enc_cfg);
    let mut stream = build_stream(cfg.stream, cfg.seed ^ 0x5eed);

    let mut claim_pool = if use_claims {
        Some(PendingClaimPool::new(ClaimConfig {
            max_open_claims: cfg.claim_max_open,
            claims_per_step: cfg.claim_per_step,
            probe_claims_per_step: cfg.claim_probe_per_step,
            rent_per_step: cfg.claim_rent,
            issue_confidence_floor: 0.0,
            verify_floor: cfg.claim_verify_floor,
            fail_floor: cfg.claim_fail_floor,
            verified_gain: cfg.claim_verified_gain,
            false_alarm_cost: cfg.claim_false_alarm_cost,
            max_rent_before_retire: 2.0,
        }))
    } else {
        None
    };

    let mut verify_nll_sum = 0.0f64;
    let mut verify_count = 0u64;
    let mut verify_correct = 0u64;
    let mut voting_nll_sum = 0.0f64;
    let mut voting_correct = 0u64;
    let mut cue_nll_sum = 0.0f64;
    let mut cue_count = 0u64;
    let mut cue_correct = 0u64;
    let mut overall_nll_sum = 0.0f64;
    let mut total_shared_features = 0u64;
    let mut shared_cycles = 0u64;
    let mut total_cue_features = 0u64;
    let mut total_verify_features = 0u64;

    let mut verify_token_role: HashMap<u32, Vec<u64>> = HashMap::new();

    let mut total_claims_verified = 0u64;
    let mut total_claims_failed = 0u64;

    for _step_idx in 0..cfg.steps {
        let (input, target) = stream.next_sample();
        let code = encoder.encode(&input);

        let phase = input.step % 6;

        let overall_p = 1.0 / cfg.max_roles as f64;
        overall_nll_sum += -overall_p.ln();

        if let Some(ref mut cp) = claim_pool {
            cp.begin_step();
        }

        // === Claim issuance at cue step ===
        if let Some(ref mut cp) = claim_pool {
            if phase == 0 {
                if input.token >= 100 && input.token < 100 + cfg.max_roles as u32 {
                    let role = (input.token - 100) as usize;
                    // Template claim: known role from cue token
                    cp.issue_template_claim(
                        input.step,
                        code.as_slice(),
                        input.step,
                        role,
                        code.as_slice(),
                        1.0,
                    );
                } else {
                    // Probe claim: unknown role, predict evidence pattern
                    cp.issue_probe_claim(
                        input.step,
                        code.as_slice(),
                        input.step,
                        code.as_slice(),
                    );
                }
            }
        }

        if let Some(ref mut cp) = claim_pool {
            cp.pay_rent();
        }

        // === Metric collection ===

        if phase == 0 {
            if let Some((predicted, confidence)) = encoder.sparse_role_vote(&code) {
                cue_nll_sum += -(confidence as f64).ln();
                cue_count += 1;
                if predicted == target.latent_role as usize {
                    cue_correct += 1;
                }
            }
        }

        if phase == 5 {
            // Use actual target role at verify (composition streams may not
            // have token→role mapping; cycle_role from cue is unreliable there).
            let actual_role = target.latent_role as usize;

            let verify_p = verify_token_role
                .get(&input.token)
                .and_then(|counts| {
                    let total: u64 = counts.iter().sum();
                    if total == 0 {
                        return None;
                    }
                    let p = counts.get(actual_role).copied().unwrap_or(0) as f64 / total as f64;
                    Some(p)
                })
                .unwrap_or(1.0 / cfg.max_roles as f64);
            verify_nll_sum += -verify_p.max(1e-10).ln();
            verify_count += 1;

            let token_counts = verify_token_role
                .get(&input.token)
                .cloned()
                .unwrap_or_default();
            let predicted = token_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| **c)
                .map(|(r, _)| r)
                .unwrap_or(0);
            if predicted == actual_role {
                verify_correct += 1;
            }

            if let Some((predicted, confidence)) = encoder.sparse_role_vote(&code) {
                voting_nll_sum += -(confidence as f64).ln();
                if predicted == actual_role {
                    voting_correct += 1;
                }
            }
        }

        // === Claim verification at verify step ===
        if let Some(ref mut cp) = claim_pool {
            if phase == 5 && input.step >= cfg.claim_gap as u64 {
                let cue_step = input.step - cfg.claim_gap as u64;
                let actual_role = target.latent_role as usize;
                let result = cp.verify_cue_step(cue_step, code.as_slice(), actual_role);
                total_claims_verified += result.verified.len() as u64;
                total_claims_failed += result.failed_count as u64;

                for verified in &result.verified {
                    encoder.retrospective_credit(
                        &verified.issuer_features,
                        cue_step,
                        verified.credit_role,
                    );
                }

                // Shared feature diagnostics
                if let Some(first) = result.verified.first() {
                    let cue_set: std::collections::HashSet<_> =
                        first.issuer_features.iter().copied().collect();
                    total_cue_features += first.issuer_features.len() as u64;
                    total_verify_features += code.as_slice().len() as u64;
                    let shared: Vec<_> = code
                        .as_slice()
                        .iter()
                        .filter(|f| cue_set.contains(f))
                        .copied()
                        .collect();
                    total_shared_features += shared.len() as u64;
                    shared_cycles += 1;
                }
            }
        }

        // === Adapt ===
        encoder.adapt(&input, &target, &code);

        // === Post-adapt: update token→role counts ===
        if phase == 5 {
            let counts = verify_token_role
                .entry(input.token)
                .or_insert_with(|| vec![0; cfg.max_roles]);
            counts[target.latent_role as usize] = counts[target.latent_role as usize].saturating_add(1);
        }
    }

    let verify_nll = if verify_count > 0 { verify_nll_sum / verify_count as f64 } else { 0.0 };
    let verify_accuracy = if verify_count > 0 { verify_correct as f64 / verify_count as f64 } else { 0.0 };
    let voting_nll = if verify_count > 0 { voting_nll_sum / verify_count as f64 } else { 0.0 };
    let voting_accuracy = if verify_count > 0 { voting_correct as f64 / verify_count as f64 } else { 0.0 };
    let cue_nll = if cue_count > 0 { cue_nll_sum / cue_count as f64 } else { 0.0 };
    let cue_accuracy = if cue_count > 0 { cue_correct as f64 / cue_count as f64 } else { 0.0 };
    let overall_nll = overall_nll_sum / cfg.steps as f64;

    let avg_cue_features = if total_cue_features > 0 { total_cue_features as f64 / shared_cycles as f64 } else { 0.0 };
    let avg_verify_features = if total_verify_features > 0 { total_verify_features as f64 / shared_cycles as f64 } else { 0.0 };
    let avg_shared = if shared_cycles > 0 { total_shared_features as f64 / shared_cycles as f64 } else { 0.0 };

    let uniform_nll = (cfg.max_roles as f64).ln();
    let cue_to_verify_cpi = uniform_nll - verify_nll;

    let total_claims_resolved = total_claims_verified + total_claims_failed;
    let claim_verified_rate = if total_claims_resolved > 0 {
        total_claims_verified as f64 / total_claims_resolved as f64
    } else {
        0.0
    };
    let claim_false_alarm_rate = if total_claims_resolved > 0 {
        total_claims_failed as f64 / total_claims_resolved as f64
    } else {
        0.0
    };

    E1bReport {
        encoder: encoder.name().to_string(),
        stream: "delayed-cue".to_string(),
        steps: cfg.steps,
        claim_enabled: use_claims,
        verify_step_nll: verify_nll,
        verify_step_accuracy: verify_accuracy,
        voting_nll_at_verify: voting_nll,
        voting_accuracy_at_verify: voting_accuracy,
        cue_step_nll: cue_nll,
        cue_step_accuracy: cue_accuracy,
        overall_nll,
        cue_to_verify_cpi,
        avg_cue_features,
        avg_verify_features,
        avg_shared_features: avg_shared,
        claim_verified_rate,
        claim_false_alarm_rate,
        claim_total_issued: total_claims_verified + total_claims_failed,
        claim_survival_count: total_claims_verified + total_claims_failed,
    }
}
