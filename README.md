# Elastic Sparse Machine (ESM) — Gate E-1A Lab

CPU-first, zero-dependency Rust workspace for engineering online sparse encoders that learn latent-role representations beyond token identity.

**Current status:** E-1A gate **FAIL** — scientific direction **NEEDS REDIRECTION**.

E1 (attention + MLP) proved that the decoder is not the bottleneck. Even a nonlinear readout cannot extract dense_CPI > 0 from Predictive v2's sparse code. The encoder representation itself needs to be shaped by role-prediction loss (Problem B).

---

## Project stage

| Stage | Verdict | Summary |
|---|---|---|
| **v1** `competitive` | FAIL | WTA implementation collapse (32 of 4096 columns active) |
| **v2** `predictive` | FAIL | Anti-collapse fixed, context split works, cross-token abstraction insufficient |
| **D** `d-full` | FAIL | Dual-channel surface+role regresses vs v2; traces inert |
| **E0** `e0` | PARTIAL PASS | Representation existence: PASS (embedding_role_sep 0.7–1.4). Linear readability: FAIL (dense_CPI < 0). |
| **E1a** `e1-attn-linear` | FAIL | Attention+linear ≈ mean+linear — attention mechanism is inert |
| **E1b** `e1-mean-mlp` | FAIL | Mean+MLP improves (+0.09 nats on delayed-role) but cannot cross zero |
| **E1c** `e1-attn-mlp` | FAIL | Attention+MLP ≈ mean+MLP — MLP helps, attention adds nothing |
| **E-1A gate** | **FAIL** | Do not implement E-1B |
| **Next** | **Problem B** | Loss-based encoder utility shaping (credit → column selection) |

---

## Repository structure

```
crates/
  esm-core/          Core data types, encoders, metrics
    src/
      encoder/
        mod.rs       SparseEncoder trait, EncoderKind, hash/competitive/predictive
        d.rs         D series — archived (dual-channel, anti-Hebbian, traces)
        e.rs         E series — E0 (dense decoder), E1a/b/c (attention + MLP)
      metrics.rs     E1aMetrics, dense_CPI, embedding_role_separation, attention diagnostics
      event.rs       InputEvent, TargetEvent (prequential protocol)
      feature.rs     FeatureId, SparseCode (sparse binary codes)
      rng.rs         Deterministic hash-based RNG
  esm-runner/        Experiment harness (e1a.rs) and synthetic streams
  esm-cli/           CLI entry point
  esm-tools/         Development utilities

docs/
  E1A_EXPERIMENT_REPORT.md   Full experimental record (42 runs, all encoder series)
  ARCHITECTURE.md            Design constraints and architectural decisions
```

---

## Encoder series

| Kind | CLI alias | Location | Status |
|---|---|---|---|
| `HashEncoder` | `hash` / `a` / `control` | `encoder::HashEncoder` | Active baseline |
| `CompetitiveEncoder` | `competitive` / `b` | `encoder::CompetitiveEncoder` | Active baseline |
| `PredictiveEncoder` | `predictive` / `c` | `encoder::PredictiveEncoder` | Current best sparse encoder |
| `EncoderD` | `d` / `d-full` / `d-no-trace` / `d-no-role-proto` | `encoder::d::EncoderD` | **Archived** |
| `EncoderE0` | `e0` / `encoder-e0` | `encoder::e::EncoderE0` | Active — mean+linear decoder |
| `EncoderE1a` | `e1a` / `e1-attn-linear` | `encoder::e::EncoderE1a` | Active — attention+linear |
| `EncoderE1b` | `e1b` / `e1-mean-mlp` | `encoder::e::EncoderE1b` | Active — mean+MLP |
| `EncoderE1c` | `e1c` / `e1-attn-mlp` | `encoder::e::EncoderE1c` | Active — attention+MLP |

---

## CLI usage

```bash
# Run a single experiment (prints JSON report to stdout)
cargo run --release -- run e1a --stream <stream> --encoder <kind> [--steps N] [--seed N] [--lr F]

# Streams
#   same-token-context   Same token appears with different latent roles
#   role-sharing         Different tokens share the same latent role
#   delayed-role         Role signal is temporally delayed

# Encoders
#   hash                 Raw token/hash control
#   predictive           Sparse projection + context-key role prototypes (v2)
#   e0                   Predictive + mean-pooled linear decoder
#   e1a / e1-attn-linear Attention top-8 + linear readout
#   e1b / e1-mean-mlp    Mean + one-hidden-layer MLP
#   e1c / e1-attn-mlp    Attention top-8 + one-hidden-layer MLP

# Encoders (archived)
#   d / d-no-trace / d-no-role-proto

# Example
cargo run --release -- run e1a --stream role-sharing --encoder e1c --steps 10000 --lr 0.01
```

---

## Key results (seed 1 / seed 2)

### Dense_CPI — the binding constraint

| Encoder | same-token-context | role-sharing | delayed-role |
|---|---|---|---|
| hash | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| e0 (mean+linear) | -0.413 / -0.420 | -1.157 / -1.160 | -0.372 / -0.372 |
| e1a (attn+linear) | -0.444 / -0.437 | -1.172 / -1.169 | -0.372 / -0.372 |
| e1b (mean+MLP) | **-0.394 / -0.394** | **-1.116 / -1.116** | **-0.283 / -0.271** |
| e1c (attn+MLP) | **-0.396 / -0.393** | **-1.117 / -1.116** | **-0.286 / -0.272** |

### Attention diagnostics (E1c, seed 1)

| Stream | mass_base | mass_proto | top_c1 | attn_corr | cpi_wo1 |
|---|---|---|---|---|---|
| same-token-context | 0.992 | 0.008 | 2.3e-05 | 0.003 | -0.307 |
| role-sharing | 0.996 | 0.004 | -1.8e-06 | -0.005 | -1.012 |
| delayed-role | 0.996 | 0.004 | 1.1e-05 | 0.040 | -0.240 |

### Embedding role separation

| Encoder | same-token-context | role-sharing | delayed-role |
|---|---|---|---|
| e0 | 1.375 / 1.393 | 1.143 / 1.100 | 1.047 / 0.725 |
| e1c | 1.252 / 1.172 | 1.064 / 1.037 | 0.953 / 1.192 |

See `docs/E1A_EXPERIMENT_REPORT.md` for the full 42-experiment matrix and analysis of all encoder series (v1, v2, D, E0, E1a, E1b, E1c).

---

## Design constraints

- **CPU-first.** No GPU, no PyTorch/Candle/Burn.
- **Safe Rust only** in core crates. No `unsafe`.
- **Zero external dependencies** in `esm-core`.
- **Integer IDs** instead of object graph pointers.
- **No `Rc<RefCell<T>>`** in the core graph.
- **No async runtime** in the core.
- **Prequential protocol:** `encode` never sees `TargetEvent`; target is only used in `adapt`.
- **No batch processing:** fully online, one step at a time.
- **No MLP/attention/global hidden state** in the sparse encoder (decoder is diagnostic only and does not affect encoding).
- **E-1A must pass** before ledger/claim/fork/router engineering begins.

---

## Next step: Problem B — loss-based encoder shaping

E1 closed the "fix the readout" hypothesis. The decoder is not the bottleneck.
The next step is to shape the encoder's column selection using loss signal from the dense decoder:

```
Encoder v2 base (predictive projection + context prototypes)
+ Dense decoder (MLP readout, already working at -0.27 nats)
+ Loss-based feedback: dense decoder loss backpropagates into
  the sparse encoder's column selection (utility shaping)

via: credit-gated column utility. Features with positive
leave-one-out credit get their column success_mass boosted;
features with negative credit get it reduced.

Key constraint: the encoder must remain sparse and online.
The shaping signal must not create a second information channel
that leaks TargetEvent into encode.
```
