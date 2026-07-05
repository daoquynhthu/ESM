use std::collections::HashMap;

use esm_core::encoder::{build_encoder, EncoderConfig, EncoderKind};
use esm_core::ledger::CausalLedger;

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
    pub ledger_gap: usize,
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
            ledger_gap: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct E1bReport {
    pub encoder: String,
    pub stream: String,
    pub steps: u64,
    pub ledger_enabled: bool,
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
}

impl E1bReport {
    pub fn to_json_pretty(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"encoder\": \"{}\",\n",
                "  \"stream\": \"{}\",\n",
                "  \"steps\": {},\n",
                "  \"ledger_enabled\": {},\n",
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
                "  \"avg_shared_features\": {:.4}\n",
                "}}"
            ),
            self.encoder,
            self.stream,
            self.steps,
            self.ledger_enabled,
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
        )
    }
}

pub fn run_e1b(cfg: E1bConfig) -> E1bReport {
    let use_ledger = cfg.ledger_gap > 0;

    let enc_cfg = EncoderConfig {
        active_bits: cfg.active_bits,
        columns: cfg.columns,
        seed: cfg.seed,
        max_roles: cfg.max_roles,
        ..EncoderConfig::default()
    };
    let mut encoder = build_encoder(cfg.encoder, enc_cfg);
    let mut stream = build_stream(cfg.stream, cfg.seed ^ 0x5eed);
    let mut ledger = if use_ledger {
        Some(CausalLedger::new(cfg.ledger_gap + 4))
    } else {
        None
    };

    // Metrics accumulators
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

    // Token→role counts for CPI computation at verify step
    let mut verify_token_role: HashMap<u32, Vec<u64>> =
        HashMap::new();

    // Track the actual role per cycle (for streams that hide role until verify).
    // At phase 0, infer actual role from the cue token (100→role 0, 101→role 1).
    let mut cycle_role: usize = 0;

    for _step_idx in 0..cfg.steps {
        let (input, target) = stream.next_sample();
        let code = encoder.encode(&input);

        let phase = input.step % 6;
        if phase == 0 && input.token >= 100 && input.token < 100 + cfg.max_roles as u32 {
            // Cue token reveals actual role for delayed-cue streams
            cycle_role = (input.token - 100) as usize;
        }
        // Overall NLL (uniform prior over max_roles)
        let overall_p = 1.0 / cfg.max_roles as f64;
        overall_nll_sum += -overall_p.ln();

        // Record encoder output in ledger (before adapt)
        if let Some(ref mut l) = ledger {
            l.record(input.step, &code);
        }

        // === Metric collection ===

        if phase == 0 {
            // Cue step: sparse role vote
            if let Some((predicted, confidence)) = encoder.sparse_role_vote(&code) {
                cue_nll_sum += -(confidence as f64).ln();
                cue_count += 1;
                if predicted == cycle_role {
                    cue_correct += 1;
                }
            }
        }

        if phase == 5 {
            // Verify step: token-baseline NLL
            let verify_p = verify_token_role
                .get(&input.token)
                .and_then(|counts| {
                    let total: u64 = counts.iter().sum();
                    if total == 0 {
                        return None;
                    }
                    let p = counts.get(cycle_role).copied().unwrap_or(0) as f64 / total as f64;
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
            if predicted == cycle_role {
                verify_correct += 1;
            }

            // Verify step: sparse role vote
            if let Some((predicted, confidence)) = encoder.sparse_role_vote(&code) {
                voting_nll_sum += -(confidence as f64).ln();
                if predicted == cycle_role {
                    voting_correct += 1;
                }
            }
        }

        // === Ledger: intersection-based retrospective credit at verify step ===
        // Only credit columns that fired at BOTH cue AND verify step.
        // This naturally selects for context-generalizing columns and avoids
        // overfitting to cue-specific (token/position) columns.
        if let Some(ref mut l) = ledger {
            if phase == 5 && input.step >= cfg.ledger_gap as u64 {
                let cue_step = input.step - cfg.ledger_gap as u64;
                if let Some(cue_features) = l.features_at(cue_step) {
                    let cue_set: std::collections::HashSet<_> = cue_features.iter().copied().collect();
                    total_cue_features += cue_features.len() as u64;
                    total_verify_features += code.as_slice().len() as u64;
                    let shared: Vec<_> = code
                        .as_slice()
                        .iter()
                        .filter(|f| cue_set.contains(f))
                        .copied()
                        .collect();
                    total_shared_features += shared.len() as u64;
                    shared_cycles += 1;
                    if !shared.is_empty() {
                        encoder.retrospective_credit(&shared, cue_step, cycle_role);
                    }
                }
            }
        }

        // === Adapt ===
        encoder.adapt(&input, &target, &code);

        // === Post-adapt: update token→role counts for next cycle ===
        // Use cycle_role (actual role) to track veridical token→role mapping.
        if phase == 5 {
            let counts = verify_token_role
                .entry(input.token)
                .or_insert_with(|| vec![0; cfg.max_roles]);
            counts[cycle_role] = counts[cycle_role].saturating_add(1);
        }
    }

    let verify_nll = if verify_count > 0 {
        verify_nll_sum / verify_count as f64
    } else {
        0.0
    };
    let verify_accuracy = if verify_count > 0 {
        verify_correct as f64 / verify_count as f64
    } else {
        0.0
    };
    let voting_nll = if verify_count > 0 {
        voting_nll_sum / verify_count as f64
    } else {
        0.0
    };
    let voting_accuracy = if verify_count > 0 {
        voting_correct as f64 / verify_count as f64
    } else {
        0.0
    };
    let cue_nll = if cue_count > 0 {
        cue_nll_sum / cue_count as f64
    } else {
        0.0
    };
    let cue_accuracy = if cue_count > 0 {
        cue_correct as f64 / cue_count as f64
    } else {
        0.0
    };
    let overall_nll = overall_nll_sum / cfg.steps as f64;

    // cue_to_verify_cpi = random_nll - verify_token_nll
    // (how much the token baseline beats random at verify step)
    // Diagnostics
    let avg_cue_features = if total_cue_features > 0 { total_cue_features as f64 / shared_cycles as f64 } else { 0.0 };
    let avg_verify_features = if total_verify_features > 0 { total_verify_features as f64 / shared_cycles as f64 } else { 0.0 };
    let avg_shared = if shared_cycles > 0 { total_shared_features as f64 / shared_cycles as f64 } else { 0.0 };

    let uniform_nll = (cfg.max_roles as f64).ln();
    let cue_to_verify_cpi = uniform_nll - verify_nll;

    E1bReport {
        encoder: encoder.name().to_string(),
        stream: "delayed-cue".to_string(),
        steps: cfg.steps,
        ledger_enabled: use_ledger,
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
    }
}
