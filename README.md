# Elastic Sparse Machine (ESM) — Gate E-1 Lab

CPU-first, zero-dependency Rust workspace for engineering online sparse encoders that learn latent-role representations beyond token identity.

**Project status: E-1A PARTIAL PASS. E-1B bridge: LEDGER INSUFFICIENT (random context prevents cross-cycle transfer).**

The hypothesis — "a CPU-first, online, sparse competitive encoder can form latent-role
representations beyond token identity" — is **PARTIALLY SUPPORTED** (E-1A). The Predictive v2
sparse encoder DOES carry readable role information. Previous "FAIL" verdict was caused
by two implementation bugs in the AttentionDecoder (MLP backprop order, extra `/n_active`
on embedding updates) and insufficient training steps (10K), not by a fundamental
representation failure.

E-1B bridge validation tested three ledger approaches across two encoder architectures:
(1) naive ledger + standard encoder — overfits to cue-specific columns, hurts verify;
(2) intersection ledger + standard encoder — too few shared columns (~5/16) are outvoted;
(3) no-mass ledger + context-dominant encoder — ~15/16 columns shared but random context
prevents cross-cycle transfer. **The core barrier is not ledger design but stream design:
random per-cycle contexts mean columns fire once and are never reused, making ledger
accumulation impossible.** A cycling or fixed-context stream is needed for the ledger
to accumulate meaningful credit across cycles.

**Verdict boundaries (final):**
- ✅ E-1A: **PARTIAL PASS** (~FAIL / NOT SUPPORTED no longer valid)
- ✅ E0 embedding_role_separation > 0: **genuine signal** (~artifact designation overturned; 50K/100K dense_CPI confirms role-readable signal)
- ❌ role-sharing stream: **retired as primary metric** (token baseline saturated; use same-token-context or delayed-role instead)
- ⏸️ E-1B bridge: **LEDGER ALONE INSUFFICIENT** (random context prevents cross-cycle transfer; needs cycling/fixed contexts)
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
| **E-1B** bridge (predictive, naive ledger) | — | ⏸️ **INSUFFICIENT** | Ledger helps cue (+4.5%) but hurts verify (-0.8%). Overfits to cue-specific columns |
| **E-1B** bridge (predictive, intersection ledger) | — | ⏸️ **INSUFFICIENT** | ~5/16 shared columns, 11 non-shared outvote them |
| **E-1B** bridge (context-predictive, ledger) | — | ⏸️ **INSUFFICIENT** | ~15/16 shared columns, but random context prevents cross-cycle transfer |

**Gate E-1A: PARTIAL PASS (corrected). E-1B bridge: LEDGER ALONE INSUFFICIENT (needs cycling/fixed contexts for cross-cycle transfer). E3/E4 wait.**

---

## Repository structure

```
crates/
  esm-core/          Core data types, encoders, metrics
    src/
      encoder/
        mod.rs       SparseEncoder trait, EncoderKind, hash/competitive/predictive
        context.rs   ContextPredictiveEncoder (context-dominant, for E-1B bridge)
        d.rs         D series — archived (dual-channel, anti-Hebbian, traces)
        e.rs         E series — E0/E1a/E1b/E1c/E2a/E2b/E2c (archived experiments)
      ledger.rs      CausalLedger — FIFO ring buffer for E-1B bridge validation
      metrics.rs     E1aMetrics, E1aReport, dense_CPI, embedding_role_separation
      event.rs       InputEvent, TargetEvent (prequential protocol)
      feature.rs     FeatureId, SparseCode (sparse binary codes)
      rng.rs         Deterministic hash-based RNG
  esm-runner/        Experiment harness (e1a.rs, e1b.rs) and synthetic streams
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
| `ContextPredictiveEncoder` | `context` / `context-predictive` / `ctx` | Active — context-dominant variant for E-1B bridge (85% context weight) |

---

## CLI usage

```bash
# Run E-1A experiment (archived encoders still runnable for reproducibility)
cargo run --release -- run e1a --stream <stream> --encoder <kind> [--steps N] [--seed N] [--lr F]

# Run E-1B bridge validation
cargo run --release -- run e1b --encoder predictive|context [--steps N] [--seed N] [--ledger-gap N]

# Streams: same-token-context | role-sharing | delayed-role | delayed-cue-verify-only

# Encoders (new): context / context-predictive / ctx — context-dominant variant

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

## E-1B bridge validation

E-1B tests whether a **causal ledger** can bridge delayed cue→role associations.
The stream separates cue and verification by 5 filler steps. Three approaches tried:

### Approach 1: Naive ledger + standard encoder (predictive)

Reinforces ALL cue-step columns at verify time. Problem: most cue columns are
token/position-specific and don't fire at verify. Their success_mass boost dilutes
generalizing columns.

| Metric | Baseline | With ledger | Delta |
|---|---|---|---|
| Cue-step accuracy | 76.3% | 80.8% | +4.5% ✅ |
| Verify-step accuracy | 51.9% | 51.1% | -0.8% ❌ |

### Approach 2: Intersection ledger + standard encoder

Only credits columns that fire at BOTH cue AND verify (~5.3 of 16). Still hurts
because 11 non-shared columns outvote the 5 shared ones.

| Metric | Baseline | Intersection ledger | Delta |
|---|---|---|---|
| Verify-step accuracy | 51.9% | 51.5% | -0.4% ❌ |

### Approach 3: Context-dominant encoder (`context-predictive`) + no-mass ledger

New encoder with context-weighted sketch terms (context ~85% of column score).
Achieves ~15/16 shared columns between cue and verify. Ledger uses unconditional
role-count addition (no success_mass to avoid cross-context pollution).

| Metric | Baseline | Ledger (counts only) | Delta |
|---|---|---|---|
| Shared columns | ~15/16 | ~15/16 | ✅ |
| Verify-step accuracy | 54.7% | 53.7% | -1.0% ❌ |

### Key finding

The context-dominant encoder successfully maximizes cue→verify column overlap (~15/16).
But the ledger still can't help because **each cycle uses a random context** — the
columns that fire in one cycle never fire again (different context next cycle). The
ledger's role-count additions are wasted on columns that are never reused.

**Root cause: random context prevents cross-cycle transfer.** The ledger can only
accumulate signal if the same columns fire across multiple cycles. A fixed or cycling
context design is needed for the ledger to accumulate meaningful credit.

### New encoder: `ContextPredictiveEncoder`

CLI alias: `context` / `context-predictive` / `ctx`. Context-dominant sketch terms:
- context_token: weight 30, fanout 20 (was 9/12)
- token XOR context: weight 8, fanout 12 (was 10/12)
- All other terms: reduced to weight 1-2, fanout 2-4
- Step XOR position: removed entirely
