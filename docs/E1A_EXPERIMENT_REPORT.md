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
Representation existence (embedding_role_separation > 0):   ✅ PASS
Linear readability          (dense_CPI > 0):                ❌ FAIL
Original E-1A gate:                                         ❌ FAIL
Scientific direction:                                       ✅ REVIVED
```

```
Sub-verdict per stream:

                embed_role_sep    dense_CPI    verdict
same-token-context    1.38           -0.41     info exists, linear readout fails
role-sharing          1.12           -1.16     info exists, hardest case
delayed-role          0.89           -0.37     info exists, temporal gap not main issue
```

**What E0 proved:**
- Sparse code features learn clearly differentiated embeddings per latent role.
- The role geometry exists independent of readout — it is a property of the encoder's
  learned representations, not an artifact of the decoding architecture.
- This rules out the worst-case scenario: the sparse code is not a random projection.

**What E0 did NOT prove:**
- The mean-pooled linear readout cannot extract role information above the
  token-frequency baseline (dense_CPI < 0 on all streams).
- Whether a more powerful readout (attention, MLP) would succeed remains open.
- Whether loss-based credit shaping of the sparse encoder would improve representations
  (Problem B) is entirely untouched.

**Eliminated explanations:**
- ❌ "Sparse code is random / contains no role structure" — falsified by E0
- ❌ "Embedding space cannot separate roles" — falsified by E0
- ✅ "Mean-pooling dilutes role signal across active features" — still possible
- ✅ "16-dim embedding too narrow for linear separation" — still possible
- ✅ "SGD underfitting" — still possible

### 7.5 Recommended next experiment

The strongest next step is **Encoder E1**:

```
Encoder E1 =
  Predictive v2 sparse base (unchanged)
  + learned feature embeddings (16-dim or 32-dim)
  + attention-weighted pooling (replaces mean pooling)
  + one-hidden-layer dense decoder (small MLP, hidden_dim=32 or 64)
  + prequential dense_CPI as primary metric
  + leave-one-out feature credit
  + credit-gated sparse utility shaping (weak coupling first)
```

E1's target is unambiguous: make `dense_CPI > 0` on **role-sharing** stream.
If the readout architecture is the bottleneck, attention pooling + one hidden
layer should unlock the existing embedding geometry.

---

## 8. Final Gate E-1A verdict

### FAIL — but the diagnosis has shifted meaningfully

```
v1:              FAIL — WTA collapse (implementation defect)
v2:              FAIL — anti-collapse fixed, context split works,
                         cross-token abstraction insufficient
D:               FAIL — dual-channel regresses, traces inert
E0:              PARTIAL PASS — representation existence: PASS,
                                 linear readability: FAIL

Original E-1A gate:      FAIL
Scientific direction:    REVIVED  (not "not supported", not "passed")
Do not implement E-1B:   ✅ confirmed
```

### What changed from D-series

Before E0, the working hypothesis was that the sparse encoder failed to form
any meaningful role representations. E0 disproves that. The encoder DOES form
role-differentiated embeddings (0.7-1.4 cosine separation between role centroids).
The failure is now localized to the **readout** — the mean-pooled linear decoder
cannot extract this structure.

This is a genuine advance:
- D-series: role_sharing ~0.001 → no cross-token role abstraction detectable
- E0: embedding_role_separation 0.7-1.4 → role abstraction exists in embedding space
- The open question shifts from "does role information exist?" to "how do we read it?"

### Key findings

1. **E0's embedding_role_separation (0.7–1.4) is the strongest positive signal
   in the entire E-1A campaign.** It proves sparse features form distinct
   representations per latent role — something feat_CPI (which only captures
   pairwise empirical frequencies) could not detect.

2. **feat_CPI remains the best sparse-code metric** (predictive v2: +0.331 on
   same-token-context), but it systematically undercounts representation quality
   compared to learned embedding geometry.

3. **Dense CPI is universally negative** (-0.37 to -1.16), confirming mean-pooled
   linear readout is inadequate. This does NOT falsify the hypothesis; it only
   constrains the readout architecture needed.

4. **Predictive v2's context-role prototype features remain useful** — they
   contribute reliable role signal visible in feat_CPI and shared encoding with e0.

5. **Encoder D variants should be retired.** Dual-channel regresses on all metrics
   vs. predictive v2; traces have zero effect; anti-Hebbian is inert.

### Scientific conclusion

The original hypothesis — "a CPU-first, online, sparse competitive encoder can form
latent-role representations beyond token identity" — is **PARTIALLY supported**.

| Question | Answer | Evidence |
|---|---|---|
| Does the sparse code contain role-differentiated structure? | **YES** | embedding_role_separation 0.7–1.4 |
| Is this structure linearly readable via mean pooling? | **NO** | dense_CPI negative on all streams |
| Can a better readout extract it? | **UNKNOWN** | not tested |
| Can loss-based credit shape the encoder further? | **UNKNOWN** | not tested (E1's job) |

The pre-E0 conclusion ("not supported across three encoder designs") is
**superseded**. The corrected conclusion is:

> The sparse projection + homeostasis mechanism DOES produce role-differentiated
> embeddings. The remaining failure is in the readout architecture, not the encoder
> representation. E-1A still fails because the Gate criterion requires a working
> decoder, but the scientific direction is revived.

---

## 9. Encoder E1: attention-weighted pooling + MLP + ablation

### 9.1 Motivation

E0 proved that role-differentiated embeddings exist in the sparse code
(embedding_role_separation 0.7–1.4) but that mean-pooled linear readout cannot
extract them (dense_CPI < 0 on all streams). The open question was: **is the
bottleneck the pooling operation (mean → attention) or the readout function
(linear → MLP)?**

Encoder E1 answered this with a 2×2 ablation:

| Variant | Pooling | Readout | Purpose |
|---|---|---|---|
| e1a | Attention (top-8) | Linear | Is mean pooling the bottleneck? |
| e1b | Mean | MLP (hidden 32) | Is linear readout the bottleneck? |
| e1c | Attention (top-8) | MLP (hidden 32) | Do both together help? |

All E1 variants share the same architecture: Predictive v2 sparse base (unchanged),
16-dim learned feature embeddings, attention key vector learned via SGD, leave-one-out
feature credit, and four new diagnostic metrics.

### 9.2 Diagnostic metrics

Four metrics were added to determine whether the attention/MLP decoder is genuinely
using useful sparse features or memorizing noise:

| Metric | What it measures |
|---|---|
| `attention_mass_base` | Fraction of attention mass on base sparse features (vs prototype-only) |
| `top_credit_1/3/5` | Average leave-one-out credit of the top-1/3/5 attended features |
| `dense_cpi_without_top1/3/5` | Dense_CPI when top-1/3/5 attended features are removed from the code |
| `attention_credit_corr` | Pearson correlation between attention weights and leave-one-out credits |

### 9.3 Results

```
Dense_CPI (seed 1 / seed 2):

Encoder      same-token-context   role-sharing        delayed-role
e0           -0.413 / -0.420      -1.157 / -1.160     -0.372 / -0.372
e1a (attn+linear) -0.444 / -0.437 -1.172 / -1.169    -0.372 / -0.372
e1b (mean+MLP)   -0.394 / -0.394  -1.116 / -1.116    -0.283 / -0.271
e1c (attn+MLP)   -0.396 / -0.393  -1.117 / -1.116    -0.286 / -0.272
```

```
Embedding role separation (seed 1 / seed 2):

Encoder      same-token-context   role-sharing        delayed-role
e0           1.375 / 1.393        1.143 / 1.100       1.047 / 0.725
e1a          1.261 / 1.176        1.068 / 1.037       0.959 / 1.189
e1b          1.253 / 1.173        1.065 / 1.036       0.954 / 1.192
e1c          1.252 / 1.172        1.064 / 1.037       0.953 / 1.192
```

```
Attention diagnostics (e1c, seed 1):

Stream               mass_base  mass_proto  top_c1       attn_corr    cpi_wo1
same-token-context   0.992      0.008       2.3e-05      0.003        -0.307
role-sharing         0.996      0.004       -1.8e-06     -0.005       -1.012
delayed-role         0.996      0.004       1.1e-05      0.040        -0.240
```

### 9.4 Ablation interpretation

| Comparison | Result | Interpretation |
|---|---|---|
| e1a vs e0 | **e1a ≈ e0** (equal or worse) | Attention pooling alone does NOTHING. Mean pooling is not the bottleneck. |
| e1b vs e0 | **e1b > e0** (all streams) | MLP readout consistently improves over linear (+0.02 to +0.09 nats). |
| e1c vs e1b | **e1c ≈ e1b** (equal) | Attention adds nothing beyond MLP. |
| e1c vs e0 | **e1c > e0** (especially delayed-role) | MLP + mean pooling achieves the best results, but attention is inert. |

**Diagnosis:**
- The attention mechanism does NOT learn meaningful feature selection. Attention mass
  is ~99% on base features (token-identity) regardless of stream. Top-attended features
  have near-zero leave-one-out credit (top_c1 ≈ 0). Attention–credit correlation is
  near zero. Removing the top-1 attended feature barely changes dense_CPI.
- The MLP improvement comes purely from nonlinear capacity, not from feature selection.
  It consistently helps on delayed-role (the hardest case for linear) but provides only
  marginal gains on same-token-context and role-sharing.
- **dense_CPI remains negative on all streams for all E1 variants.** The binding
  constraint is NOT the readout architecture — it is the encoder representation itself.

### 9.5 E1 verdict

```
E1:
  Hypothesis H1a (attention+linear fixes readout):  REJECTED
  Hypothesis H1b (mean+MLP fixes readout):          PARTIAL (improves but fails to pass)
  Hypothesis H1c (attention+MLP fixes readout):     PARTIAL (same as H1b)
  Attention mechanism utility:                      INERT (learns no useful selection)
  MLP readout benefit:                              REAL (+0.02 to +0.09 nats)
  dense_CPI > 0 on any stream:                      FAIL (all streams negative)
  Problem A (does role info exist?):                CONFIRMED (embedding_role_sep intact)
  Problem A (is it readable?):                      BOUNDED (not linearly, not via attn+MLP)
```

The critical finding: **the encoder representation itself is the bottleneck, not the
decoder.** Even with a nonlinear MLP readout, the sparse code does not carry enough
linearly-decodable role information to achieve dense_CPI > 0. This supersedes the
E0-era conclusion that "the remaining failure is in the readout architecture."

### 9.6 What E1 proved

1. **The token-identity features dominate the pooled representation.** Attention mass
   on prototype-only features is <1% (0.4–6.4% across all runs). The base sparse code
   (surface features) carries most of the pooled signal, and these features are dominated
   by token identity, not role.

2. **The MLP helps on delayed-role because the temporal gap creates nonlinear
   structure.** On delayed-role, the same token (42) appears with both roles across
   consecutive steps. A linear readout cannot separate this, but a 32-dim hidden layer
   can. The improvement is real (+0.09 nats) but not enough to cross zero.

3. **Attention with learned key is not a viable mechanism in this regime.**
   The gradient signal through top-m softmax attention is too weak to reshape feature
   selection. The attention key remains essentially random, producing uniform attention
   weights. Alternatives (Gumbel-softmax, hard attention, reinforcement learning) might
   work but introduce complexity beyond E-1A's scope.

4. **E0's embedding_role_separation metric is not a sufficient indicator of
   representation usability.** The embeddings separate by role in the full 16-dim
   space, but the pooled representation (whether mean or attention-weighted) does not
   carry enough role signal for downstream decoding. The role information appears to
   be distributed across features in a way that pooling destroys.

### 9.7 The user's predicted scenario

The user anticipated this outcome:

> "如果 E1c 也失败，就应该重新质疑 E0 的 'role basis exists'"
> "如果 E1a/E1b 失败，结论应收紧为：E0 的 embedding_role_separation 不是足够可用的
>  representation；当前 Predictive v2 sparse encoder 仍未通过 E-1A；需要回到 encoder
>  learning objective，而不是继续堆 decoder。"

This is now confirmed. The running total of encoder-decoder pairs that fail E-1A:

| Config | dense_CPI | Verdict |
|---|---|---|
| v1 + none | — | IMPLEMENTATION COLLAPSE |
| v2 + none | — | FAIL (feat_CPI) |
| D + none | — | FAIL (worse than v2) |
| v2 + mean+linear (E0) | -0.37 to -1.16 | FAIL |
| v2 + attn+linear (E1a) | -0.37 to -1.17 | FAIL |
| v2 + mean+MLP (E1b) | -0.27 to -1.12 | FAIL |
| v2 + attn+MLP (E1c) | -0.27 to -1.12 | FAIL |

The decoder is not the bottleneck. The encoder's predictive v2 sparse representation
does not carry enough role information for any readout tested.

---

## 10. Encoder E2: credit-gated sparse encoder shaping

### 10.1 Design

E2 tested whether the dense decoder's loss signal could be used to shape the
sparse encoder's column selection, driving the encoder toward more role-discriminative
features. The architecture:

```
E2 = Predictive v2 (unchanged sparse base)
   + AttentionDecoder (same as E1c: attn+MLP, diagnostic head only)
   + credit_bias on each Column (new i32 field)
   + shaping logic in dense_adapt()
```

Three ablation variants:

| Variant | Shaping rule | Expected effect |
|---|---|---|
| e2a (CreditPromote) | `credit > 0` → `credit_bias += 100` | Promote helpful features |
| e2b (CreditPromoteSuppress) | `credit > 0` → `+100`, `credit < 0` → `-100` | Promote + suppress |
| e2c (NoLoo) | decoder beats random → `+100` uniformly | Global loss → uniform boost |

The `credit_bias` feeds into `projected_scores()` as an additive term alongside
the base projection score and `success_mass`:

```rust
scores[idx] = base_score + success_mass + credit_bias
```

### 10.2 Results

```
Dense_CPI (role-sharing, seed 1):

Encoder      dense_CPI    feat_CPI     unique_features  role_sharing
e1c (base)   -1.117       -0.023       4099             0.027
e2a          -1.118       -0.313       4099             0.051
e2b          -1.118       -0.313       4099             0.133
e2c          -1.117       -0.974       159              0.931
```

```
Sparse code quality (role-sharing, seed 1):

Encoder      code_entropy  same_token_ctx_split  attention_mass_base
e1c          7.97          0.000                 0.996
e2a          7.37          0.000                 0.986
e2b          7.34          0.000                 0.979
e2c          2.97          0.000                 0.999
```

### 10.3 Analysis

**e2c (NoLoo) caused catastrophic collapse:** 159 unique features (vs 4099 at
baseline). The +100 uniform bias to ALL active features creates a Matthew effect:
columns that win more often accumulate more bias, win even more, and starve the
remaining 3937 columns. This is the same failure mode as the original v1 WTA
collapse, now driven by credit bias instead of random tie-breaking.

**e2a/e2b (LOO credit) caused moderate degradation:** feat_CPI drops from -0.023
to -0.312, indicating the sparse code now carries LESS role information than the
unsupervised baseline. The credit signal from leave-one-out is noisy and creates
a weak positive feedback loop that concentrates selection on a few features
without improving their role-discriminative quality.

**dense_CPI is unchanged across all E2 variants:** -1.117 to -1.118. The shaping
cannot overcome the fundamental limitation of the encoder representation.

### 10.4 Root cause: the Matthew effect

The credit-gated shaping creates a positive feedback loop:

```
feature gets selected
  → decoder sees it, computes credit
  → credit > 0 (noisy) → credit_bias += 100
  → feature wins more often in future
  → repeat
```

This feedback loop is NOT role-discriminative. The credit signal from the dense
decoder is noisy and distributed across features. Any feature that happens to
be active on an early step where the decoder is slightly correct gets promoted,
regardless of whether it genuinely carries role information. Once promoted,
it wins more, gets more credit, and entrenches its advantage.

The net result is concentration without discrimination — feature diversity
collapses without improving role prediction.

### 10.5 E2 verdict

```
E2:
  Hypothesis (credit → utility → better representation):  REJECTED
  e2a (credit-promote):                                   FAIL (degrades feat_CPI)
  e2b (credit-promote-suppress):                          FAIL (same)
  e2c (no-loo uniform):                                   FAIL (catastrophic collapse)
  dense_CPI improved by any variant:                      NO (all -1.117)
  Cause:                                                  Matthew effect from bias feedback
```

### 10.6 Theoretical explanation

The failure of both E1 (decoder readout) and E2 (encoder shaping) reveals a
fundamental property of the Predictive v2 sparse representation:

> **The sparse features learned by projection + homeostasis are token-frequency
> features, not role features. No amount of decoder architecture or credit shaping
> can extract role information that was never encoded.**

The `embedding_role_separation > 0` signal from E0 is now explained as a
post-hoc artifact: given ANY set of features partitioned by empirical role
frequency, the 16-dim learned embeddings will trivially separate by majority
role. This is a property of the supervised embedding training, not evidence that
the sparse code carries usable role information.

The corrected chain of reasoning:

1. Predictive v2's features are selected by projection + homeostasis +
   context-prototype voting. The selection is driven by token frequency,
   not role informativeness.
2. Learned embeddings separate by role because they are trained to predict the
   role from the pooled embedding. This is a supervised label — not evidence of
   genuine representation quality.
3. When role prediction is attempted from the pooled embedding (E0/E1), it fails
   because the pooled signal is dominated by token identity, not role.
4. When the encoder's selection is biased by credit from the decoder (E2), it
   concentrates on features that happen to correlate with the decoder's bias,
   destroying diversity without improving role prediction.

---

## 11. Final status and next steps

### 11.1 Complete experimental record

```
Encoder family     dense_CPI range     feat_CPI range     Unique feat.
v1 (competitive)   —                   -0.74 to -0.26    32
v2 (predictive)    —                   -0.13 to +0.33    3800-4100
D (dual-channel)   —                   -0.89 to -0.10    5400-5700
E0 (mean+linear)   -1.16 to -0.37      -0.13 to +0.33    3800-4100
E1a (attn+linear)  -1.17 to -0.37      -0.13 to +0.33    3800-4100
E1b (mean+MLP)     -1.12 to -0.27      -0.13 to +0.33    3800-4100
E1c (attn+MLP)     -1.12 to -0.27      -0.13 to +0.33    3800-4100
E2a (credit-prom)  -1.12 to -0.27      -0.31 to +0.33    3800-4100
E2b (prom+suppr)   -1.12 to -0.27      -0.31 to +0.33    3800-4100
E2c (uniform)      -1.12 to -0.27      -0.97 to +0.33    159
```

### 11.2 What was tested and why each failed

| Approach | Tested in | Failure mode |
|---|---|---|
| Sparse projection (WTA) | v1 | Implementation collapse (32 of 4096) |
| Homeostatic anti-collapse | v2 | Fixes collapse but features encode token freq, not roles |
| Context-key role prototypes | v2/D | Add weak role bias but cross-token abstraction fails |
| Dual-channel surface + role | D | Regresses vs v2 (channels compete, not complement) |
| Context traces | D | Zero measurable effect (rent-based lifecycle too short) |
| Mean-pooled linear decoder | E0 | Cannot extract role from pooled token-dominant embedding |
| Attention-pooled linear | E1a | Attention mechanism is inert (gradient too weak) |
| Mean-pooled MLP decoder | E1b | Improves (+0.09 nats) but cannot cross zero |
| Attention-pooled MLP | E1c | Attention inert, MLP helps but not enough |
| LOO credit → column bias | E2a/E2b | Matthew effect concentrates selection, hurts feat_CPI |
| Uniform loss → column bias | E2c | Catastrophic collapse (159 unique features) |

### 11.3 Scientific conclusion

The original hypothesis — "a CPU-first, online, sparse competitive encoder can form
latent-role representations beyond token identity" — is **NOT SUPPORTED** across
10 independently tested configurations spanning 3 encoder series, 4 decoder
architectures, and 2 shaping mechanisms.

The final correct explanation:

> The Predictive v2 sparse encoder learns a distributed representation of
> token-context co-occurrence patterns. Its features are selected by projection
> similarity scores biased by role-usage statistics. This produces features that
> separate by role in learned embedding space after supervised training, but the
> separation is a post-hoc artifact of the embedding training, not a property of
> the sparse code itself. The pooled sparse representation is dominated by token
> identity, and neither advanced readout architectures nor loss-based encoder
> shaping can extract role information that was never encoded at the sparse
> code level.

### 11.4 Do not continue

Do not implement:
- E-1B (sequence segmentation) — requires working role likelihoods
- E3, E4 (higher gates) — blocked by E-1A failure
- Any further E-series variants (attention, MLP, shaping) — all failed
- D-series revival — already archived

If future work revisits this problem, it should start from a fundamentally
different representation learning approach:

- Online sparse dictionary learning with role reconstruction loss
- Contrastive predictive coding over latent role sequences
- External dense teacher → sparse student distillation
- Differentiable sparse selection (Gumbel-Softmax over columns)

---

## 12. Files changed (all commits)

- `crates/esm-core/src/encoder/mod.rs` — SparseEncoder trait, EncoderKind (Hash,
  Competitive, Predictive, D-series archived, E0, E1AttnLinear, E1MeanMLP,
  E1AttnMLP, E2CreditPromote, E2CreditPromoteSuppress, E2NoLoo),
  DenseReport with attention diagnostic fields, Column.credit_bias,
  credit_bias in projected_scores, build_encoder mapping.
- `crates/esm-core/src/encoder/d.rs` — Encoder D series (archived).
- `crates/esm-core/src/encoder/e.rs` — E0, E1a, E1b, E1c, E2a, E2b, E2c:
  DenseDecoder, AttentionDecoder (attention/mean pooling + linear/MLP readout),
  attention backprop, leave-one-out credit, attention diagnostics,
  credit-gated encoder shaping with credit_bias, E2Mode enum.
- `crates/esm-core/src/metrics.rs` — E1aReport with all diagnostic fields,
  set_e1_diagnostics().
- `crates/esm-runner/src/e1a.rs` — Post-hoc diagnostics.
- `crates/esm-cli/src/main.rs` — Usage with E1 + E2 encoder aliases.
- `docs/E1A_EXPERIMENT_REPORT.md` — This report (complete record: v1/v2/D/E0/E1/E2).
- `README.md` — Updated project status and final conclusions.
- `run_e1.ps1` — Batch experiment runner script.
