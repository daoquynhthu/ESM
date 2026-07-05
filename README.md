# Elastic Sparse Machine (ESM) — Gate E-1A Lab

CPU-first, zero-dependency Rust workspace for engineering online sparse encoders that learn latent-role representations beyond token identity.

**Project status: E-1A PARTIAL PASS (corrected after bug fixes).**

The hypothesis — "a CPU-first, online, sparse competitive encoder can form latent-role
representations beyond token identity" — is **PARTIALLY SUPPORTED**. The Predictive v2
sparse encoder DOES carry readable role information. Previous "FAIL" verdict was caused
by two implementation bugs in the AttentionDecoder (MLP backprop order, extra `/n_active`
on embedding updates) and insufficient training steps (10K), not by a fundamental
representation failure.

**Verdict boundaries (final audit):**
- ✅ E-1A: **PARTIAL PASS** (~FAIL / NOT SUPPORTED no longer valid)
- ✅ E0 embedding_role_separation > 0: **genuine signal** (~artifact designation overturned; 50K/100K dense_CPI confirms role-readable signal)
- ❌ role-sharing stream: **retired as primary metric** (token baseline saturated; use same-token-context or delayed-role instead)
- ⏸️ E-1B bridge validation: **may start** (role likelihoods exist)
- ⏸️ E3/E4: **not yet released** (wait for E-1B passage)

**Frozen: attention-weighted pooling.** Current softmax top-m attention with learned key
is inert (0.99 mass on base features, correlation ≈ 0). No further repair. Use
mean pooling for all future readouts.

**Acceptance criteria for re-verification:**
- token_NLL ≥ 0.20 (streams must not be saturated)
- E0 / E1b dense_CPI > +0.05
- role_sharing > hash baseline

---

## Complete experimental record

| Experiment | dense_CPI (best) | Verdict | Notes |
|---|---|---|---|
| **v1** `competitive` | — | FAIL | WTA collapse (32 of 4096 features) |
| **v2** `predictive` | — | FAIL | No cross-token role abstraction |
| **D** `d-full` | — | FAIL | Dual-channel regresses vs v2 |
| **E0** `mean+linear` 10K | -0.41 | Originally declared FAIL | **Bug: none (DenseDecoder correct)** — just 10K too few |
| **E0** `mean+linear` 50K | **+0.145** | **PASS** ✅ | Corrected: 50K sufficient |
| **E0** `mean+linear` 100K | **+0.244** | **PASS** ✅ | Continues improving |
| **E1a** `attn+linear` 10K | -0.44 | FAIL | Bug: extra /n_active (16x LR reduction) |
| **E1b** `mean+MLP` 10K | -0.39 | Originally FAIL | Bug: MLP backprop order (17% gradient error) |
| **E1b** `mean+MLP` 50K (fixed) | **+0.127** | **PASS** ✅ | Bug fix + 50K = positive |
| **E1c** `attn+MLP` 10K | -0.39 | FAIL | Both bugs |
| **E1c** `attn+MLP` 50K (fixed) | -0.145 | FAIL | Attn mechanism inert even with fixes |
| **E2** all variants | — | FAIL | Credit shaping creates Matthew effect |

**Gate E-1A: PARTIAL PASS (corrected). E-1B bridge validation may proceed; E3/E4 wait.**

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
|---|---|---|---|
| `HashEncoder` | `hash` | Baseline control |
| `CompetitiveEncoder` | `competitive` | Archived (WTA collapse) |
| `PredictiveEncoder` | `predictive` | Active — carries role information (confirmed via dense decoder) |
| `EncoderD` | `d` / `d-no-trace` / `d-no-role-proto` | Archived (regresses vs v2) |
| `EncoderE0` | `e0` | Active — role information confirmed (dense_CPI +0.244 at 100K) |
| `EncoderE1a` | `e1a` / `e1-attn-linear` | Archived (attention inert even after bug fixes) |
| `EncoderE1b` | `e1b` / `e1-mean-mlp` | Active with bug fixes (dense_CPI +0.127 at 50K) |
| `EncoderE1c` | `e1c` / `e1-attn-mlp` | Archived (attention inert, MLP alone sufficient) |
| `EncoderE2a` | `e2a` / `e2-credit-promote` | Archived (Matthew effect) |
| `EncoderE2b` | `e2b` / `e2-credit-promote-suppress` | Archived (Matthew effect) |
| `EncoderE2c` | `e2c` / `e2-no-loo` | Archived (catastrophic collapse) |

---

## CLI usage

```bash
# Run any experiment (archived encoders still runnable for reproducibility)
cargo run --release -- run e1a --stream <stream> --encoder <kind> [--steps N] [--seed N] [--lr F]

# Streams: same-token-context | role-sharing | delayed-role

# Recommended: use 50000+ steps for dense decoder convergence
# E0 and E1b achieve dense_CPI > 0 on same-token-context at 50K steps.
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

## Key scientific finding (corrected)

The Predictive v2 sparse encoder (projection + homeostasis + context prototypes)
DOES form latent-role representations beyond token identity. The `embedding_role_separation`
signal from E0 was genuine evidence of role-differentiated embedding structure.

**Why E-1A was initially declared FAIL:**

1. **Code bug: MLP backprop order** (e.rs:356-362). The `d_hidden` gradient was computed
   from already-updated `w2` weights, creating ~17% gradient corruption. This prevented
   the MLP readout from converging properly.

2. **Code bug: extra `/n_active` on embedding update** (e.rs:585). The effective embedding
   learning rate was 16x lower than intended in the AttentionDecoder, making learning
   extremely slow.

3. **Insufficient training steps** — 10,000 steps provided only ~40 updates per feature.
   At 50,000+ steps, the simple mean-pooled linear decoder (E0) achieves dense_CPI = +0.244.

**Corrected verdict:** The sparse encoder does carry role-readable information, and a
simple linear decoder with sufficient training can extract it. The previously declared
"NOT SUPPORTED" conclusion was an artifact of implementation bugs and insufficient
training steps, not a genuine representation failure.

**Frozen route: attention-weighted pooling.** E1a / E1c (attention variants) are
archived. All future readouts use mean pooling only (E0 linear or E1b mean+MLP).
