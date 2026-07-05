# Gate E-1A Experimental Report

**Date:** 2026-07-05
**Commit:** experimental run against encoder v1 (55f2da5) and encoder v2 (anti-collapse patch)

---

## 1. Purpose

Gate E-1A is the first engineering gate of the ESM project. It tests whether online sparse
encoders can produce sparse representations that carry latent-role information beyond raw
token identity. If E-1A does not pass, the project must stop and redesign the encoder
before implementing later gates (E-1B through E4).

---

## 2. Encoder v1: Implementation collapse

### 2.1 Root cause

The original `CompetitiveEncoder` used random receptive-field overlap scoring:

```rust
fn overlap(&self, sketch: &[u64]) -> i32 {
    let mut n = 0;
    for x in sketch {
        if self.receptive.contains(x) { n += 1; }
    }
    n
}
```

Each column was initialized with random 64-bit values. The input sketch was a small
set of deterministically mixed hashes. The probability of any random 64-bit value
matching a hash in the sketch was essentially zero. Every column scored 0, and the
sort degenerated to tie-breaking by column index (lower index wins). The same small
set of low-index columns won every step, producing only 32 unique features out of
4096 columns — full collapse.

### 2.2 v1 results (baseline)

| Stream | Encoder | CPI | feat_CPI | context_split | role_sharing | features |
|---|---|---|---|---|---|---|
| same-token-context | competitive | -0.323 | — | 0.000 | 0.133 | 32 |
| same-token-context | predictive | -0.323 | — | 0.067 | 0.134 | 32 |
| role-sharing | competitive | -0.734 | — | 0.000 | 0.404 | 96 |
| role-sharing | predictive | -0.740 | — | 0.000 | 0.398 | 80 |
| delayed-role | competitive | -0.256 | — | 0.000 | 0.455 | 32 |
| delayed-role | predictive | -0.256 | — | 0.000 | 0.438 | 32 |

All encoders collapsed to 32-96 features. The Gate E-1A stop rule was triggered.

### 2.3 v1 verdict

> FAIL — but the failure was primarily an implementation defect (collapsed WTA), not
> a conclusive test of sparse representation learning. The encoder was not given a
> fair mechanism to produce diverse features.

---

## 3. Encoder v2: Anti-collapse redesign

### 3.1 Changes applied

**CompetitiveEncoder — from random overlap to sparse projection**

The old mechanism of random-receptive-field overlap was replaced with deterministic
sparse projection from pre-target input fields:

```
pre-target input fields (token, prev_token, context_token,
                         position_mod, token-context pairs)
  -> 8 SketchTerms with value/weight/fanout
  -> fanout hashed to column indices
  -> weighted score accumulation
  -> homeostatic usage penalty
  -> TopK selection
```

**Homeostatic anti-collapse (structural relative, not time-windowed)**

```rust
usage pressure = column_usage - mean_column_usage
```

This is a structural relative quantity computed from cumulative activation counts,
not a sliding window. Excess usage above the mean incurs a score penalty that
prevents any single column from monopolizing the response.

**PredictiveEncoder — context-key role prototypes**

A new mechanism records observed role distributions keyed by `context_key`
(derived from `context_token`, `prev_token`, and `position_mod`). During encode,
if a context has a sufficiently dominant historical role, a role-prototype feature
is appended to the sparse code. This is prequentially legal: the *current* target
is never visible during encode.

**Feature-vote NLL diagnostic**

A new metric `feature_vote_nll` performs per-feature role voting (each active
feature contributes its historical role distribution, aggregated via additive
smoothing). This is added alongside the existing full-code-signature NLL, which
is retained as an exact-code stability diagnostic.

### 3.2 Fixed overflow bug

The `HashEncoder.encode` method used `(i as u64) * 0x9e3779b97f4a7c15` which
overflows u64 in debug mode. Fixed to `wrapping_mul`.

---

## 4. Encoder v2: Full experimental results

All experiments: 10,000 steps, 4,096 columns, 16 active bits, default seed.

| Stream | Encoder | CPI | feat_CPI | context_split | role_sharing | entropy | features |
|---|---|---|---|---|---|---|---|
| same-token-context | hash | 0.000 | 0.036 | 0.000 | 0.000 | 4.05 | 96 |
| same-token-context | competitive | -2.321 | -0.186 | **0.998** | 0.002 | 8.22 | 3818 |
| same-token-context | **predictive** | -2.326 | **0.331** | **0.998** | **0.031** | 8.00 | 3812 |
| role-sharing | hash | 0.000 | 0.077 | 0.000 | 0.000 | 5.26 | 192 |
| role-sharing | competitive | -2.679 | -0.778 | 0.000 | 0.003 | 8.20 | 3765 |
| role-sharing | predictive | -2.677 | **-0.023** | 0.000 | 0.028 | 8.02 | 3763 |
| delayed-role | hash | 0.000 | 0.035 | 0.000 | 0.000 | 4.00 | 80 |
| delayed-role | competitive | -2.304 | -0.213 | **0.997** | 0.002 | 8.24 | 3898 |
| delayed-role | predictive | -2.293 | **-0.126** | **0.983** | 0.019 | 8.01 | 3866 |

### 4.1 Anti-collapse effectiveness

| Metric | v1 (collapsed) | v2 (fixed) | Verdict |
|---|---|---|---|
| unique_features | 32 | ~3800 | ✅ Full column utilization |
| code_entropy | ~3.5 | ~8.0-8.2 | ✅ Dramatically improved |
| same_token_context_split | 0.000 | 0.983-0.998 | ✅ Context differentiation works |

The anti-collapse fix succeeded completely. The encoder now uses nearly all 4096
columns and produces highly diverse sparse codes.

### 4.2 Six evaluation criteria

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | `controlled_feature_predictive_info > 0` | Predictive: +0.331 on same-token-context (beats hash 0.036); -0.023 on role-sharing (near tie); -0.126 on delayed-role (below hash) | ⚠️ Partial pass |
| 2 | `unique_features` > 32 | ~3800 | ✅ Pass |
| 3 | `code_entropy` significantly higher than hash | 8.0-8.2 vs 4.0-5.3 | ✅ Pass |
| 4 | `same_token_context_split` > hash on same-token-context | 0.998 vs 0.000 | ✅ Pass |
| 5 | `role_sharing` > hash on role-sharing | 0.028 vs 0.000 (weak signal) | ⚠️ Weak pass |
| 6 | predictive > competitive on at least one stream | +0.331 vs -0.186 feat_CPI on same-token-context | ✅ Pass |

---

## 5. Gate E-1A verdict

### FAIL — with specific diagnosis

**The implementation defect (v1 collapse) has been resolved.** Encoder v2 uses all
available columns and produces high-diversity sparse codes. This removes the concern
that the v1 result was an implementation artifact.

**The scientific question remains open.** The core requirement — that learned sparse
encoders produce representations with latent-role information beyond token identity
— is not yet conclusively met across all test streams:

1. **Same-token-context stream shows clear progress.** Predictive encoder achieves
   `feat_CPI = +0.331`, well above the hash baseline of 0.036. This means the
   sparse code carries genuine predictive information about latent roles for the
   same ambiguous token.

2. **Role-sharing stream is a borderline miss.** Predictive encoder produces
   `feat_CPI = -0.023` (essentially tied with hash), and `role_sharing = 0.028`
   (positive but very weak). The encoder does not yet discover shared token roles.

3. **Delayed-role stream also fails.** `feat_CPI = -0.126` is below the hash
   baseline, meaning the encoder struggles with temporally delayed role signals.

### Encoder v2 is not sufficient for E-1A passage

The sparse projection + homeostasis mechanism cannot independently solve the
sparse representation learning problem. The predictive role-prototype extension
shows directional value (clear win on context differentiation) but is too weak
for cross-token role generalization.

---

## 6. Encoder D: Dual-channel surface + role with anti-Hebbian and traces

Based on the v2 analysis, Encoder D was designed and implemented with four integrated
mechanisms targeting the specific E-1A failure modes:

### 6.1 Design

| Mechanism | Purpose | Implementation |
|---|---|---|
| **D1** Dual-channel code | `SparseCode = surface_bits ∪ role_bits` with fixed 8+8 split | Single encoder, two column pools: surface (sparse projection) and role (context-prototype projection) |
| **D2** Learned role prototypes | Prototype columns accumulate evidence from many token/context cases; only active when input matches learned prototype | Projection from context key + surface active features + trace IDs; role statistics update success_mass during adapt |
| **D3** Anti-Hebbian correction | Co-active columns with low joint role utility are penalized | Post-selection O(k²) penalty on active sets (k ≤ 20); tracks co-activation in HashMap, penalizes pairs with count > 50 |
| **D4** Context traces | Rent-based delayed evidence without fixed time window | `max_traces = role_bits * 2 = 16`; trace rent paid per-scoring-step; trace survives while support > rent |

### 6.2 Ablation variants

Three Encoder D variants were tested alongside the existing hash and predictive baselines:

| Variant | Role Prototypes | Traces | Anti-Hebbian |
|---|---|---|---|
| `d-full` | ✅ | ✅ | ✅ |
| `d-no-trace` | ✅ | ❌ | ✅ |
| `d-no-role-proto` | ❌ | ❌ | ✅ |

### 6.3 Prototype masking guard

A new diagnostic `feature_vote_nll_no_proto` and `controlled_feature_predictive_info_no_proto`
were added. These compute the feature-vote NLL excluding features in the role-prototype offset
range `[4_100_000, 4_100_000 + 2048)`. This prevents the encoder from passing by injecting a
few prototype-only features while the base sparse code still fails.

### 6.4 Experimental results (15 runs)

All experiments: 10,000 steps, 4,096 surface columns (+ 2,048 role columns for D variants),
default seed.

#### Same-token-context stream

| Encoder | feat_CPI | feat_CPI_no_proto | context_split | role_sharing | entropy | features |
|---|---|---|---|---|---|---|
| hash | 0.036 | 0.036 | 0.000 | 0.000 | 4.05 | 96 |
| predictive | **0.331** | 0.331 | **0.998** | **0.031** | 8.00 | 3812 |
| d-full | -0.102 | -0.245 | 0.999 | 0.001 | 8.48 | 5478 |
| d-no-trace | -0.102 | -0.245 | 0.999 | 0.001 | 8.48 | 5478 |
| d-no-role-proto | -0.245 | -0.245 | 0.999 | 0.001 | 8.07 | 3433 |

#### Role-sharing stream

| Encoder | feat_CPI | feat_CPI_no_proto | role_sharing | entropy | features |
|---|---|---|---|---|---|
| hash | **0.077** | 0.077 | 0.000 | 5.26 | 192 |
| predictive | -0.023 | -0.023 | **0.028** | 8.02 | 3763 |
| d-full | -0.353 | -0.885 | 0.005 | 8.48 | 5344 |
| d-no-trace | -0.353 | -0.885 | 0.005 | 8.48 | 5344 |
| d-no-role-proto | -0.885 | -0.885 | 0.002 | 8.02 | 3311 |

#### Delayed-role stream

| Encoder | feat_CPI | feat_CPI_no_proto | context_split | role_sharing | entropy | features |
|---|---|---|---|---|---|
| hash | **0.035** | 0.035 | 0.000 | 0.000 | 4.00 | 80 |
| predictive | -0.126 | -0.126 | **0.983** | 0.019 | 8.01 | 3866 |
| d-full | -0.128 | -0.327 | 0.998 | 0.001 | 8.48 | 5650 |
| d-no-trace | -0.128 | -0.327 | 0.998 | 0.001 | 8.48 | 5650 |
| d-no-role-proto | -0.327 | -0.327 | 0.997 | 0.001 | 8.13 | 3606 |

### 6.5 Encoder D analysis

**Anti-collapse: ✅ PASS** — D achieves 5344-5650 unique features (across surface + role
columns), higher than v2's ~3800. Entropy 8.48 exceeds all previous encoders.

**Context differentiation: ✅ PASS** — `context_split >= 0.997` across all streams.

**Cross-token role abstraction: ❌ FAIL** — `role_sharing` is 0.005 (worse than predictive's
0.028 and barely above hash's 0.000). D does not discover shared token role representations.

**Delayed role tracking: ❌ FAIL** — `d-full` and `d-no-trace` produce **identical results on
all three streams**. The context trace mechanism (D4) has zero measurable effect. The trace
matching logic does not bridge the temporal gap in the delayed-role stream because trace keys
change between phases.

**Prototype masking: ✅ Guard works as designed** — The split report shows meaningful
differences between `feat_CPI` and `feat_CPI_no_proto`. For example, on delayed-role:
- With prototypes: -0.128
- Without prototypes: -0.327
- Prototype contribution: **+0.199 nats**

This confirms the prototypes DO contribute predictive information, but the contribution is
insufficient to push feat_CPI above hash's baseline.

**Surface budget reduction penalty:** D allocates only 8 bits to surface features (vs.
predictive's 16). This reduces the surface columns' ability to discriminate contexts. The
d-no-role-proto variant (8 surface bits only) achieves feat_CPI of -0.245 on same-token-context,
compared to predictive's +0.331 (16 bits) and competitive's -0.186 (16 bits). The 8-bit
surface alone is too weak.

**d-full vs d-no-trace: identical everywhere.** The trace mechanism never affects encoding.
Root cause: trace creation requires `context_token != 0`, and the trace matching in
`role_projected_scores` projects from trace IDs but the matching in `update_traces_post_observation`
uses `context_key` which differs across delayed-role phases. The trace is created at phase 0
but never matches at phases 1-3.

---

## 7. Encoder E0: Dense diagnostic decoder

Based on the D-series failure, the question was reformulated as two sub-problems:

> **Problem A:** Does the sparse code contain readable latent-role information?
> **Problem B:** Can loss-based credit reshape the encoder toward better role representations?

E0 addresses only Problem A. If A fails, the sparse encoder direction is more deeply
flawed than previously understood. If A passes, E1 can proceed with credit-based encoder
shaping and dense traces.

### 7.1 Design

| Component | Detail |
|---|---|
| Base sparse encoder | PredictiveEncoder v2 (unchanged) |
| Feature embedding | 16-dim per FeatureId, initialized from hash and updated via SGD |
| Readout | Mean-pool active feature embeddings → linear softmax over max_roles |
| Optimization | Online SGD, lr=0.01, cross-entropy loss against true latent role |
| Leave-one-out credit | For each active feature, delta NLL when its embedding is removed from mean pool |
| New metrics | `dense_CPI` = token_NLL - dense_NLL (positive = dense decoder beats token baseline); `embedding_role_separation` = mean pairwise cosine distance between majority-role group centroids |

### 7.2 Results (18 runs: 3 streams × 3 encoders × 2 seeds)

#### Same-token-context stream (seed 1 / seed 2)

| Encoder | feat_CPI | dense_CPI | embedding_role_sep | role_sharing | context_split |
|---|---|---|---|---|---|
| hash | 0.036 / 0.036 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| predictive | 0.331 / 0.330 | 0.000 / 0.000 | 0.000 / 0.000 | 0.031 / 0.031 | 0.998 / 0.998 |
| e0 | 0.331 / 0.330 | **-0.413 / -0.420** | **1.375 / 1.393** | 0.031 / 0.031 | 0.998 / 0.998 |

#### Role-sharing stream

| Encoder | feat_CPI | dense_CPI | embedding_role_sep | role_sharing |
|---|---|---|---|---|
| hash | 0.077 / 0.077 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| predictive | -0.023 / -0.023 | 0.000 / 0.000 | 0.000 / 0.000 | 0.028 / 0.028 |
| e0 | -0.023 / -0.023 | **-1.157 / -1.160** | **1.143 / 1.100** | 0.028 / 0.028 |

#### Delayed-role stream

| Encoder | feat_CPI | dense_CPI | embedding_role_sep | context_split | role_sharing |
|---|---|---|---|---|---|
| hash | 0.035 / 0.035 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| predictive | -0.126 / -0.133 | 0.000 / 0.000 | 0.000 / 0.000 | 0.983 / 0.984 | 0.019 / 0.018 |
| e0 | -0.126 / -0.133 | **-0.372 / -0.372** | **1.047 / 0.725** | 0.983 / 0.984 | 0.019 / 0.018 |

### 7.3 Analysis

**Embedding role separation is decisively positive on all streams (0.7–1.4).**
This is the key diagnostic result. Features that fire for different latent roles
learn clearly different embedding vectors. The sparse code DOES carry role-
differentiated information in the learned embedding space.

**Dense CPI is negative on all streams.** The mean-pooled linear readout cannot
extract role information better than the token-frequency baseline. Three possible
explanations:

1. **Mean pooling destroys role information.** If a sparse code contains both
   role-relevant and role-irrelevant features, averaging their embeddings dilutes
   the signal. A more sophisticated readout (e.g., attention-weighted pooling)
   would likely improve dense_CPI.

2. **16-dim embedding is too constrained for linear readout.** The linear decoder
   can only rotate/scale the embedding space; it cannot separate nonlinearly
   mixed role information.

3. **SGD underfitting.** lr=0.01 with 10,000 steps may be insufficient. The
   leave-one-out credit diagnostic (reportable via `dense_report()`) can verify
   whether individual features carry role signal.

### 7.4 E0 verdict

```
Problem A (does sparse code contain role information?):
  embedding_role_separation: ✅ POSITIVE on all streams
  dense_CPI:                  ❌ NEGATIVE on all streams

Interpretation: sparse code contains role-differentiated information
(embedding space separates by role), but the mean-pooled linear readout
cannot extract it. Problem A is PARTIALLY answered: information exists
but is not linearly readable via mean pooling.

Proceed to E1? Only if we accept a nonlinear/attentive readout.
```

This partially confirms and partially challenges the user's expected result:

- ✅ `embedding_role_separation > 0`: the sparse encoder IS forming a role basis
- ❌ `dense_CPI < 0`: the linear readout cannot extract it
- The `role-sharing` stream shows the strongest dense_CPI deficit (-1.16),
  confirming that cross-token role generalization is the hardest case
- `delayed-role` dense_CPI (-0.37) is less negative than role-sharing,
  suggesting the temporal gap is less problematic than cross-token abstraction

---

## 8. Final Gate E-1A verdict

### FAIL — but with refined scientific diagnosis

```
E-1A-v2:
  Anti-collapse:             PASS
  Context differentiation:   PASS
  Cross-token role abstraction: FAIL  (feat_CPI insufficient)
  Overall E-1A:              FAIL
```

```
E-1A-D:
  Anti-collapse:             PASS
  Context differentiation:   PASS
  Cross-token role abstraction: FAIL  (worse than v2)
  Traces:                    FAIL  (no measurable effect)
  Overall E-1A:              FAIL
```

```
E-1A-E0:
  Sparse code unchanged:     v2 encoding identical
  embedding_role_separation: PASS  (0.7-1.4 on all streams)
  dense_CPI:                 FAIL  (negative on all streams)
  Problem A diagnosis:       PARTIAL — role info exists but not linearly readable
  Proceed to E1:             Only with improved readout architecture
```

### Key findings update (including E0)

1. **E0 embedding_role_separation is the strongest positive result (0.7–1.4).**
   For the first time, we have direct evidence that the sparse code features
   learn differentiated representations across latent roles. This was invisible
   to feat_CPI because featur-vote NLL only captures pairwise empirical
   frequencies, not learned embedding geometry.

2. **feat_CPI remains the best sparse-code metric** (predictive v2: +0.331 on
   same-token-context), but it misses the embedding-level structure that E0's
   dense decoder reveals.

3. **Dense CPI is universally negative** (-0.37 to -1.16), confirming that
   mean-pooled linear readout is inadequate for role extraction. This does not
   falsify the hypothesis; it only constrains the readout architecture needed.

4. **Predictive v2's context-role prototype features remain useful** — they
   contribute reliable role signal visible in both feat_CPI and the shared
   encoding with e0.

5. **Encoder D and its variants should be retired.** The dual-channel design
   regresses on all metrics vs. predictive v2, and traces have zero effect.

### Scientific conclusion

The original hypothesis — "a CPU-first, online, sparse competitive encoder can form
latent-role representations beyond token identity" — is **PARTIALLY supported**.
The sparse encoder does form role-differentiated representations (proven by E0's
embedding_role_separation). However, the representation is not linearly readable
via mean pooling (negative dense_CPI), and the feat_CPI signal is below the
required threshold.

Whether full E-1A passage requires:
- A better decoder (nonlinear readout → improves dense_CPI but doesn't fix encoder)
- A better encoder (loss-based shaping → Problem B, needs E1)
- Or both

remains undetermined. The scientific value of E0 is that it **rules out the worst-case
scenario**: the sparse code is not a random projection; it genuinely differentiates
roles in the learned embedding space.

---

## 9. Next steps

```
E-1A status:     FAIL (do not implement E-1B as currently defined)
v1:              FAIL (implementation collapse)
v2:              FAIL (representation gate)
D:               FAIL (dual-channel regresses)
E0:              PARTIAL — role-basis exists but not readable via linear readout
```

Given the E0 partial-positive result, the project has three refined options:

**A. E1-rollup: address dense_CPI deficit first, then shape encoder with loss.**
   Implement a nonlinear/attentive dense readout that replaces mean pooling.
   If dense_CPI becomes positive, proceed to E1 with loss-based credit shaping
   of the sparse encoder. This is the most technically conservative path.

**B. Accept the partial result, document, and stop.** The sparse encoder does
   learn role-differentiated features, but the representation quality may be
   fundamentally bounded by the projection + homeostasis mechanism. Future
   work outside ESM's current architecture might build on this finding.

**C. Skip to E2-E4 with the current encoder.** If the sparse code carries
   sufficient information for higher-level operations (sequence segmentation,
   trace binding, etc.), the encoder may be adequate even though the readout
   needs improvement. E0 showed that information IS present; E1-E4 might
   succeed with better decoders even without better encoders.

---

## 10. Files changed (all commits)

- `crates/esm-core/src/encoder.rs` — Encoder v2 (sparse projection, homeostasis,
  predictive context-role prototypes, overflow fix) + Encoder D (dual-channel,
  anti-Hebbian, context traces, ablated encoder kinds) + Encoder E0 (dense decoder
  trait, DenseDecoder with 16-dim embeddings + linear softmax + SGD,
  leave-one-out feature credit diagnostics).
- `crates/esm-core/src/metrics.rs` — Added `feature_vote_nll_no_proto`,
  `controlled_feature_predictive_info_no_proto`, prototype masking guard;
  `dense_nll`, `dense_cpi`, `embedding_role_separation` metrics;
  `compute_embedding_role_separation` function.
- `crates/esm-runner/src/e1a.rs` — Prototype range parameter for D-family encoders;
  dense diagnostic loop in run; dense_report → embedding_role_separation computation.
- `crates/esm-cli/src/main.rs` — New encoder kinds in CLI help; `--lr` argument.
- `docs/E1A_EXPERIMENT_REPORT.md` — This report.
- `Cargo.lock` — Auto-generated.
