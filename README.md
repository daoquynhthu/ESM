# Elastic Sparse Machine (ESM) — Gate E-1A Lab

CPU-first, zero-dependency Rust workspace for engineering online sparse encoders that learn latent-role representations beyond token identity.

**Current status:** E-1A gate **FAIL** — but with a revived scientific direction.

The final experiment (E0) proved sparse code features DO learn role-differentiated embedding geometry, but the linear mean-pooling readout cannot extract it. Next step is **Encoder E1** with attention-weighted pooling.

---

## Project stage

| Stage | Verdict | Summary |
|---|---|---|
| **v1** `competitive` | FAIL | WTA implementation collapse (32 of 4096 columns active) |
| **v2** `predictive` | FAIL | Anti-collapse fixed, context split works, cross-token abstraction insufficient |
| **D** `d-full` | FAIL | Dual-channel surface+role regresses vs v2; traces inert |
| **E0** `e0` | **PARTIAL PASS** | **Representation existence: PASS** (embedding_role_sep 0.7–1.4). **Linear readability: FAIL** (dense_CPI < 0). Scientific direction revived. |
| **E-1A gate** | **FAIL** | Do not implement E-1B until readout is fixed |
| **Next** | **E1** | Attention-weighted pooling + one-hidden-layer decoder → target: `dense_CPI > 0` |

---

## Repository structure

```
crates/
  esm-core/          Core data types, encoders, metrics
    src/
      encoder/
        mod.rs       SparseEncoder trait, EncoderKind, hash/competitive/predictive
        d.rs         D series — archived (dual-channel, anti-Hebbian, traces)
        e.rs         E series — current (E0: dense diagnostic decoder)
      metrics.rs     E1aMetrics, dense_CPI, embedding_role_separation
      event.rs       InputEvent, TargetEvent (prequential protocol)
      feature.rs     FeatureId, SparseCode (sparse binary codes)
      rng.rs         Deterministic hash-based RNG
  esm-runner/        Experiment harness (e1a.rs) and synthetic streams
  esm-cli/           CLI entry point
  esm-tools/         Development utilities

docs/
  E1A_EXPERIMENT_REPORT.md   Full experimental record (18 runs, all encoder series)
  ARCHITECTURE.md            Design constraints and architectural decisions
```

---

## Encoder series

| Kind | CLI alias | Location | Status |
|---|---|---|---|
| `HashEncoder` | `hash` / `a` / `control` | `encoder::HashEncoder` | Active baseline |
| `CompetitiveEncoder` | `competitive` / `b` | `encoder::CompetitiveEncoder` | Active baseline |
| `PredictiveEncoder` | `predictive` / `c` | `encoder::PredictiveEncoder` | **Current best sparse encoder** |
| `EncoderD` | `d` / `d-full` / `d-no-trace` / `d-no-role-proto` | `encoder::d::EncoderD` | **Archived** — dual-channel regresses |
| `EncoderE0` | `e0` / `encoder-e0` | `encoder::e::EncoderE0` | **Active experimental** — dense decoder |

> **Note:** D-series encoders are kept for reproducibility but are archived. Do not use for new experiments. They are in a separate submodule (`encoder::d`) and require explicit opt-in.

---

## CLI usage

```bash
# Run a single experiment (prints JSON report to stdout)
cargo run --release -- run e1a --stream <stream> --encoder <kind> [--steps N] [--seed N] [--lr F]

# Streams
#   same-token-context   Same token appears with different latent roles
#   role-sharing         Different tokens share the same latent role
#   delayed-role         Role signal is temporally delayed

# Encoders (active)
#   hash                 Raw token/hash control
#   predictive           Sparse projection + context-key role prototypes (v2)
#   e0                   Predictive + 16-dim feature embeddings + linear softmax + SGD

# Encoders (archived)
#   d                    Dual-channel surface+role + anti-Hebbian + traces
#   d-no-trace           D without context traces
#   d-no-role-proto      D without role prototypes (surface only)

# Example: run E0 on role-sharing with custom learning rate
cargo run --release -- run e1a --stream role-sharing --encoder e0 --steps 10000 --lr 0.01
```

---

## Key results (seed 1 / seed 2)

### Same-token-context

| Encoder | feat_CPI | dense_CPI | embedding_role_sep | context_split |
|---|---|---|---|---|
| hash | 0.036 / 0.036 | — | — | 0.000 |
| predictive | **0.331 / 0.330** | — | — | **0.998** |
| e0 | **0.331 / 0.330** | -0.413 / -0.420 | **1.375 / 1.393** | **0.998** |

### Role-sharing (hardest case)

| Encoder | feat_CPI | dense_CPI | embedding_role_sep | role_sharing |
|---|---|---|---|---|
| hash | 0.077 / 0.077 | — | — | 0.000 |
| predictive | -0.023 / -0.023 | — | — | 0.028 |
| e0 | -0.023 / -0.023 | **-1.157 / -1.160** | **1.143 / 1.100** | 0.028 |

### Delayed-role

| Encoder | feat_CPI | dense_CPI | embedding_role_sep | context_split |
|---|---|---|---|---|
| hash | 0.035 / 0.035 | — | — | 0.000 |
| predictive | -0.126 / -0.133 | — | — | 0.983 |
| e0 | -0.126 / -0.133 | **-0.372 / -0.372** | **1.047 / 0.725** | 0.983 |

See `docs/E1A_EXPERIMENT_REPORT.md` for the full 18-experiment matrix and analysis of all encoder series (v1, v2, D, E0).

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
- **No MLP/attention/global hidden state** in the sparse encoder (E0's dense decoder is diagnostic only and does not affect encoding).
- **E-1A must pass** before ledger/claim/fork/router engineering begins.

---

## Next canonical experiment: Encoder E1

Target: `dense_CPI > 0` on role-sharing stream.

```
Encoder E1:
  Base:           Predictive v2 (unchanged)
  Readout:        attention-weighted pooling (replaces mean pooling)
                  + one-hidden-layer dense decoder (hidden_dim=32 or 64)
  Encoder:        credit-gated sparse utility shaping (weak coupling first)
  Traces:         deferred to E1b
```

The binding constraint is readout architecture. With attention pooling, the decoder can learn which active features carry role signal and which are noise — something mean pooling cannot do. If this achieves `dense_CPI > 0`, the encoder itself may not need modification. If not, loss-based credit shaping of sparse selection (Problem B) becomes necessary.
