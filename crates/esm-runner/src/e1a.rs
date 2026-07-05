//! Gate E-1A representation quality runner.

use esm_core::encoder::{build_encoder, EncoderConfig, EncoderKind};
use esm_core::metrics::{compute_embedding_role_separation, E1aMetrics, E1aReport};

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
    pub lr: f32,
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
            lr: 0.01,
        }
    }
}

fn prototype_range(kind: EncoderKind) -> (u32, u32) {
    match kind {
        EncoderKind::D | EncoderKind::DNoTrace => (4_100_000, 4_100_000 + 2048),
        _ => (0, 0),
    }
}

pub fn run_e1a(cfg: E1aConfig) -> E1aReport {
    let enc_cfg = EncoderConfig {
        active_bits: cfg.active_bits,
        columns: cfg.columns,
        seed: cfg.seed,
        max_roles: cfg.max_roles,
        lr: cfg.lr,
        ..EncoderConfig::default()
    };
    let mut encoder = build_encoder(cfg.encoder, enc_cfg);
    let mut stream = build_stream(cfg.stream, cfg.seed ^ 0x5eed);
    let (proto_off, proto_end) = prototype_range(cfg.encoder);
    let mut metrics = E1aMetrics::with_prototype_range(cfg.max_roles, cfg.sample_limit, proto_off, proto_end);

    for _ in 0..cfg.steps {
        let (input, target) = stream.next_sample();
        let code = encoder.encode(&input);
        metrics.observe_prequential(&input, &target, &code);

        // Dense prediction (prequential: before adapt)
        if let Some(probs) = encoder.dense_predict_prequential(&code) {
            metrics.observe_dense_prequential(&target, &probs);
        }

        encoder.adapt(&input, &target, &code);
        encoder.dense_adapt(&code, &target);
    }

    let mut report = metrics.report(encoder.name(), stream.name());

    // Embedding role separation from learned embeddings
    if let Some(dr) = encoder.dense_report() {
        if !dr.feature_embeddings.is_empty() {
            let sep = compute_embedding_role_separation(
                &dr.feature_embeddings,
                metrics.feature_role_counts(),
                cfg.max_roles,
            );
            report.set_embedding_role_separation(sep);
        }
    }

    report
}
