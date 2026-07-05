# Elastic Sparse Machine (ESM) — Gate E-1A Lab

CPU-first, zero-dependency Rust workspace for engineering online sparse encoders that learn latent-role representations beyond token identity.

**Project status: E-1A FAILED after 10 independent configurations.**

The hypothesis is **NOT SUPPORTED**. The Predictive v2 sparse encoder learns token-frequency features, not role features. Neither advanced decoders (attention, MLP) nor loss-based encoder shaping (credit bias) can extract role information that was never encoded in the sparse code.

---

## Complete experimental record

| Experiment | dense_CPI | Verdict | Failure mode |
|---|---|---|---|
| **v1** `competitive` | — | FAIL | WTA collapse (32 of 4096 features) |
| **v2** `predictive` | — | FAIL | No cross-token role abstraction |
| **D** `d-full` | — | FAIL | Dual-channel regresses vs v2 |
| **E0** `mean+linear` | -1.16 to -0.37 | FAIL | Linear readout cannot extract role |
| **E1a** `attn+linear` | -1.17 to -0.37 | FAIL | Attention mechanism inert |
| **E1b** `mean+MLP` | -1.12 to -0.27 | FAIL | MLP helps (+0.09) but can't cross zero |
| **E1c** `attn+MLP` | -1.12 to -0.27 | FAIL | Attention inert, MLP not enough |
| **E2a** `credit-promote` | -1.12 to -0.27 | FAIL | Matthew effect, hurts feat_CPI |
| **E2b** `promote+suppress` | -1.12 to -0.27 | FAIL | Same |
| **E2c** `no-loo uniform` | -1.12 to -0.27 | FAIL | Catastrophic collapse (159 features) |

**Gate E-1A: FAIL. Do not implement E-1B, E3, or E4.**

---

## Repository structure

```
crates/
  esm-core/          Core data types, encoders, metrics
    src/
      encoder/
        mod.rs       SparseEncoder trait, EncoderKind, hash/competitive/predictive
        d.rs         D series — archived (dual-channel, anti-Hebbian, traces)
        e.rs         E series — E0/E1a/E1b/E1c/E2a/E2b/E2c (archived experiments)
      metrics.rs     E1aMetrics, E1aReport, dense_CPI, embedding_role_separation
      event.rs       InputEvent, TargetEvent (prequential protocol)
      feature.rs     FeatureId, SparseCode (sparse binary codes)
      rng.rs         Deterministic hash-based RNG
  esm-runner/        Experiment harness (e1a.rs) and synthetic streams
  esm-cli/           CLI entry point
  esm-tools/         Development utilities

docs/
  E1A_EXPERIMENT_REPORT.md   Complete experimental record (v1/v2/D/E0/E1/E2)
  ARCHITECTURE.md            Design constraints and architectural decisions
```

---

## Encoder series (all archived)

| Kind | CLI alias | Status |
|---|---|---|
| `HashEncoder` | `hash` | Baseline control |
| `CompetitiveEncoder` | `competitive` | Archived (WTA collapse) |
| `PredictiveEncoder` | `predictive` | Archived (no role abstraction) |
| `EncoderD` | `d` / `d-no-trace` / `d-no-role-proto` | Archived (regresses vs v2) |
| `EncoderE0` | `e0` | Archived (linear readout fails) |
| `EncoderE1a` | `e1a` / `e1-attn-linear` | Archived (attention inert) |
| `EncoderE1b` | `e1b` / `e1-mean-mlp` | Archived (MLP not enough) |
| `EncoderE1c` | `e1c` / `e1-attn-mlp` | Archived (same) |
| `EncoderE2a` | `e2a` / `e2-credit-promote` | Archived (Matthew effect) |
| `EncoderE2b` | `e2b` / `e2-credit-promote-suppress` | Archived (Matthew effect) |
| `EncoderE2c` | `e2c` / `e2-no-loo` | Archived (catastrophic collapse) |

---

## CLI usage

```bash
# Run any experiment (archived encoders still runnable for reproducibility)
cargo run --release -- run e1a --stream <stream> --encoder <kind> [--steps N] [--seed N] [--lr F]

# Streams: same-token-context | role-sharing | delayed-role

# All encoder kinds are runnable but have all failed E-1A.
```

---

## Design constraints

- **CPU-first.** No GPU, no PyTorch/Candle/Burn.
- **Safe Rust only** in core crates. No `unsafe`.
- **Zero external dependencies** in `esm-core`.
- **Prequential protocol:** `encode` never sees `TargetEvent`.
- **No MLP/attention/global hidden state** in the sparse encoder.
- **Decoder is diagnostic only** — does not affect encoding (except E2 variants).

---

## Key scientific finding

The Predictive v2 sparse encoder (projection + homeostasis + context prototypes)
learns token-frequency features, not role features. The `embedding_role_separation`
signal from E0 was a post-hoc artifact of supervised embedding training, not
evidence of genuine role representation in the sparse code. No tested decoder
or shaping mechanism could extract role information beyond the token baseline.
