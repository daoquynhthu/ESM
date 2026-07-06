use esm_core::encoder::{build_encoder, EncoderConfig, EncoderKind};
use esm_core::genesis::{GenesisConfig, GenesisManager};

use crate::stream::{build_e1d_stream, E1dStreamKind};

#[derive(Copy, Clone, Debug)]
pub struct E1dConfig {
    pub stream: E1dStreamKind,
    pub encoder: EncoderKind,
    pub steps: u64,
    pub seed: u64,
    pub active_bits: usize,
    pub columns: usize,
    pub max_roles: usize,
    // Genesis config
    pub genesis_max_elements: usize,
    pub genesis_max_probes: usize,
    pub genesis_probes_per_step: usize,
    pub genesis_rent_per_step: f32,
    pub genesis_utility_floor: f32,
    pub genesis_parent_coverage_floor: f32,
    pub genesis_parent_utility_floor: f32,
    pub genesis_surprise_floor: f32,
    pub genesis_probe_exploration_fraction: f32,
    pub genesis_coverage_overlap_min: f32,
}

impl Default for E1dConfig {
    fn default() -> Self {
        Self {
            stream: E1dStreamKind::NovelPattern,
            encoder: EncoderKind::Composition,
            steps: 20_000,
            seed: 1,
            active_bits: 16,
            columns: 4096,
            max_roles: 4,
            genesis_max_elements: 256,
            genesis_max_probes: 128,
            genesis_probes_per_step: 2,
            genesis_rent_per_step: 0.01,
            genesis_utility_floor: 0.05,
            genesis_parent_coverage_floor: 0.3,
            genesis_parent_utility_floor: 0.1,
            genesis_surprise_floor: 0.5,
            genesis_probe_exploration_fraction: 0.1,
            genesis_coverage_overlap_min: 0.3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct E1dReport {
    pub encoder: String,
    pub stream: String,
    pub steps: u64,
    pub total_probes_created: u64,
    pub current_probe_count: usize,
    pub active_element_count: usize,
    pub total_retired: u64,
    pub total_promoted: u64,
    pub avg_utility: f64,
    pub avg_rent_paid: f64,
    pub coverage_rate: f64,
    pub steps_with_genesis: u64,
    pub accuracy: f64,
    pub overall_nll: f64,
}

impl E1dReport {
    pub fn to_json_pretty(&self) -> String {
        format!(
            "{{\n\
             \"encoder\": \"{}\",\n\
             \"stream\": \"{}\",\n\
             \"steps\": {},\n\
             \"total_probes_created\": {},\n\
             \"current_probe_count\": {},\n\
             \"active_element_count\": {},\n\
             \"total_retired\": {},\n\
             \"total_promoted\": {},\n\
             \"avg_utility\": {:.6},\n\
             \"avg_rent_paid\": {:.6},\n\
             \"coverage_rate\": {:.6},\n\
             \"steps_with_genesis\": {},\n\
             \"accuracy\": {:.6},\n\
             \"overall_nll\": {:.6}\n\
             }}",
            self.encoder,
            self.stream,
            self.steps,
            self.total_probes_created,
            self.current_probe_count,
            self.active_element_count,
            self.total_retired,
            self.total_promoted,
            self.avg_utility,
            self.avg_rent_paid,
            self.coverage_rate,
            self.steps_with_genesis,
            self.accuracy,
            self.overall_nll,
        )
    }
}

pub fn run_e1d(cfg: E1dConfig) -> E1dReport {
    let enc_cfg = EncoderConfig {
        active_bits: cfg.active_bits,
        columns: cfg.columns,
        seed: cfg.seed,
        max_roles: cfg.max_roles,
        ..EncoderConfig::default()
    };
    let mut encoder = build_encoder(cfg.encoder, enc_cfg);
    let mut stream = build_e1d_stream(cfg.stream, cfg.seed ^ 0x6eed);

    let genesis_cfg = GenesisConfig {
        max_elements: cfg.genesis_max_elements,
        max_probes: cfg.genesis_max_probes,
        probes_per_step: cfg.genesis_probes_per_step,
        rent_per_step: cfg.genesis_rent_per_step,
        utility_floor: cfg.genesis_utility_floor,
        parent_coverage_floor: cfg.genesis_parent_coverage_floor,
        parent_utility_floor: cfg.genesis_parent_utility_floor,
        surprise_floor: cfg.genesis_surprise_floor,
        probe_exploration_fraction: cfg.genesis_probe_exploration_fraction,
        coverage_overlap_min: cfg.genesis_coverage_overlap_min,
    };
    let mut genesis = GenesisManager::new(genesis_cfg);

    let mut correct = 0u64;
    let mut overall_nll_sum = 0.0f64;
    let mut steps_with_genesis = 0u64;

    for _step_idx in 0..cfg.steps {
        let (input, target) = stream.next_sample();
        genesis.step_begin();

        let code = encoder.encode(&input);
        let margins = encoder.column_role_margins();
        let feature_offset = encoder.feature_offset();

        let encoder_surprise = encoder
            .sparse_role_vote(&code)
            .map(|(_, c)| -(c as f64).ln() as f32)
            .unwrap_or(0.0);

        genesis.after_encode(
            code.as_slice(),
            &margins,
            encoder_surprise,
            feature_offset,
            cfg.max_roles,
        );

        // Combined vote
        let encoder_vote = encoder.sparse_role_vote(&code);
        let (element_votes, _, num_voters) = genesis.collect_votes(code.as_slice());

        let (predicted, _confidence) = combine_vote(encoder_vote, &element_votes, num_voters, cfg.max_roles);
        if predicted == target.latent_role as usize {
            correct += 1;
        }

        let uniform_p = 1.0 / cfg.max_roles as f64;
        overall_nll_sum += -uniform_p.ln();

        encoder.adapt(&input, &target, &code);
        genesis.after_adapt(code.as_slice(), target.latent_role as usize);
        genesis.step_end();

        if genesis.genneses_this_step > 0 {
            steps_with_genesis += 1;
        }
    }

    let report = genesis.report();
    let accuracy = correct as f64 / cfg.steps as f64;
    let overall_nll = overall_nll_sum / cfg.steps as f64;

    E1dReport {
        encoder: encoder.name().to_string(),
        stream: stream.name().to_string(),
        steps: cfg.steps,
        total_probes_created: report.total_probes_created,
        current_probe_count: report.current_probe_count,
        active_element_count: report.active_element_count,
        total_retired: report.total_retired,
        total_promoted: report.total_promoted,
        avg_utility: report.avg_utility,
        avg_rent_paid: report.avg_rent_paid,
        coverage_rate: report.coverage_rate,
        steps_with_genesis,
        accuracy,
        overall_nll,
    }
}

fn combine_vote(
    encoder_vote: Option<(usize, f32)>,
    element_votes: &[u32],
    num_voters: usize,
    max_roles: usize,
) -> (usize, f32) {
    let num_elem_roles = element_votes.len();
    match (encoder_vote, num_voters) {
        (Some((enc_role, enc_conf)), 0) => (enc_role, enc_conf),
        (None, 0) => (0, 1.0 / max_roles as f32),
        (None, _) => {
            let total: u32 = element_votes.iter().sum();
            if total == 0 { return (0, 0.0); }
            let pred = element_votes.iter()
                .enumerate()
                .max_by_key(|(_, w)| **w)
                .map(|(r, _)| r)
                .unwrap_or(0);
            let conf = element_votes[pred] as f32 / total as f32;
            (pred, conf)
        }
        (Some((enc_role, enc_conf)), _) => {
            let mut weighted = vec![0.0f32; max_roles.max(num_elem_roles)];
            weighted[enc_role] += enc_conf;

            let elem_total: u32 = element_votes.iter().sum();
            if elem_total > 0 {
                for (r, &w) in element_votes.iter().enumerate() {
                    if r < weighted.len() {
                        weighted[r] += w as f32 / elem_total.max(1) as f32;
                    }
                }
                let pred = weighted.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(r, _)| r)
                    .unwrap_or(0);
                let conf = weighted.iter().sum::<f32>();
                let factor = if conf > 0.0 { 1.0 / conf } else { 1.0 };
                (pred, weighted[pred] * factor)
            } else {
                (enc_role, enc_conf)
            }
        }
    }
}
