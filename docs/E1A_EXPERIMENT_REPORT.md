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

## 6. Recommendation: Begin Encoder D design

Per the pre-established stop rule:

> If v2 still fails, the conclusion is stronger: it is not a first-implementation
> collapse, but a design failure of the current sparse competitive encoder for E-1A.

The recommended next step is **Encoder D**, exploring explicit decorrelation /
anti-Hebbian / sparse dictionary learning approaches. The predictive encoder's
context-role prototype mechanism from v2 should be retained as a building block.

Candidate directions for Encoder D:

- **Anti-Hebbian decorrelation:** Active columns should inhibit co-active columns
  to force feature diversity, replacing the indirect usage-homeostasis penalty.
- **Sparse dictionary learning with online codebook updates:** Each column learns
  a prototype vector; encoding selects the TopK closest prototypes; adaptation
  updates prototypes toward active inputs and away from competitors.
- **Explicit role-supervised pressure:** Use the observed latent role (prequentially,
  after prediction is fixed) to push feature representations apart for different
  roles — moving beyond passive role-counting toward discriminative feature shaping.

---

## 7. Files changed (this report)

- `crates/esm-core/src/encoder.rs` — Encoder v2: sparse projection, relative
  usage homeostasis, predictive context-role prototypes, overflow fix.
- `crates/esm-core/src/metrics.rs` — Added feature_vote_nll and
  controlled_feature_predictive_info diagnostics.
- `docs/E1A_EXPERIMENT_REPORT.md` — This report.
