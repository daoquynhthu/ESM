//! Gate E-1A representation quality runner.

use esm_core::encoder::{build_encoder, EncoderConfig, EncoderKind};
use esm_core::metrics::{E1aMetrics, E1aReport};

use crate::stream::{build_stream, StreamKind};

#[derive(Copy, Clone, Debug)]
pub struct E1aConfig {
    pub stream: StreamKind,
    pub encoder: EncoderKind,
    pub steps: u64,
    pub seed: u64,
    pub active_bits: usize,
    pub columns: usize,
    pub sample_limit: usize,
    pub max_roles: usize,
}

impl Default for E1aConfig {
    fn default() -> Self {
        Self {
            stream: StreamKind::SameTokenContext,
            encoder: EncoderKind::Hash,
            steps: 10_000,
            seed: 1,
            active_bits: 16,
            columns: 4096,
            sample_limit: 4096,
            max_roles: 16,
        }
    }
}

pub fn run_e1a(cfg: E1aConfig) -> E1aReport {
    let enc_cfg = EncoderConfig {
        active_bits: cfg.active_bits,
        columns: cfg.columns,
        seed: cfg.seed,
        max_roles: cfg.max_roles,
        ..EncoderConfig::default()
    };
    let mut encoder = build_encoder(cfg.encoder, enc_cfg);
    let mut stream = build_stream(cfg.stream, cfg.seed ^ 0x5eed);
    let mut metrics = E1aMetrics::new(cfg.max_roles, cfg.sample_limit);

    for _ in 0..cfg.steps {
        let (input, target) = stream.next_sample();
        let code = encoder.encode(&input);
        metrics.observe_prequential(&input, &target, &code);
        encoder.adapt(&input, &target, &code);
    }

    metrics.report(encoder.name(), stream.name())
}
