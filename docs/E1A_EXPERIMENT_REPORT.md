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

## 7. Final Gate E-1A verdict

### FAIL — Encoder D does not pass

```
E-1A-v2:
  Anti-collapse:             PASS
  Context differentiation:   PASS
  Cross-token role abstraction: FAIL
  Delayed role tracking:     FAIL
  Overall E-1A:              FAIL
```

```
E-1A-D:
  Anti-collapse:             PASS
  Context differentiation:   PASS
  Cross-token role abstraction: FAIL  (worse than predictive v2)
  Delayed role tracking:     FAIL  (traces have zero effect)
  Prototype useful?          YES (+0.199 nats on delayed-role)
  Overall E-1A:              FAIL
  Proceed to Encoder D only: NO
```

### Key findings

1. **Predictive v2 remains the best encoder** across all three streams. Its combination of
   full 16-bit sparse projection + context-key role prototypes achieves the best feat_CPI
   on same-token-context (+0.331) and is competitive on role-sharing (-0.023) and delayed-role (-0.126).

2. **Encoder D's dual-channel design with 8+8 split is worse than 16-bit single-channel.**
   Halving the surface budget degrades context discrimination, and the role prototype
   columns do not compensate.

3. **Context traces (D4) need fundamental redesign.** The current trace matching by
   `context_key` cannot bridge temporal gaps. A trace should outlive the key that created it.

4. **Anti-Hebbian co-activation penalty** had no observable effect because surface column
   diversity is already ensured by the sparse projection + homeostasis mechanism.

5. **Role prototypes contribute positive signal** (+0.199 nats on delayed-role) but the
   mechanism is too weak for the required `feat_CPI > hash + 0.05` threshold.

### Scientific conclusion

The hypothesis that "a CPU-first, online, sparse competitive encoder can form latent-role
representations beyond token identity" is **not supported** by the current evidence, across
three encoder designs (v1, v2, D).

The failure is now beyond implementation defects. The sparse projection + homeostasis
mechanism (v2), dual-channel prototypes + anti-Hebbian (D), and context-key role statistics
(predictive v2) all produce insufficient cross-token role abstraction.

---

## 8. Next steps

The project status is:

```
E-1A:        complete
v1:          FAIL (implementation collapse)
v2:          FAIL (representation gate)
D:           FAIL (dual-channel regresses; traces inert)
Overall:     FAIL — do not implement E-1B or later gates
```

Three possible directions:

**A. Retry with fundamentally different encoder paradigm.** Move beyond the projection +
homeostasis framework. Consider sparse dictionary learning with online SGD, explicit
decorrelation objectives, or role-supervised contrastive pressure.

**B. Accept the negative result and document.** If the hypothesis is falsified by three
independent designs across the same toy streams, the project should document this
conclusion transparently. The ESM architecture may need a different approach to
representation formation before re-entering E-1A.

**C. Relax the E-1A gate.** If the ESM project decides that role abstraction can emerge
from segment-level learning rather than encoder-level (i.e., move E-1A to after E-1B/E-1C),
the gate order could be reconsidered. However, this carries the risk that later mechanisms
mask encoder failure.

---

## 9. Files changed (all commits)

- `crates/esm-core/src/encoder.rs` — Encoder v2 (sparse projection, homeostasis,
  predictive context-role prototypes, overflow fix) + Encoder D (dual-channel,
  anti-Hebbian, context traces, ablated encoder kinds).
- `crates/esm-core/src/metrics.rs` — Added `feature_vote_nll_no_proto`,
  `controlled_feature_predictive_info_no_proto`, prototype masking guard.
- `crates/esm-runner/src/e1a.rs` — Prototype range parameter for D-family encoders.
- `crates/esm-cli/src/main.rs` — New encoder kinds in CLI help.
- `docs/E1A_EXPERIMENT_REPORT.md` — This report.
- `Cargo.lock` — Auto-generated.
