# Elastic Sparse Machine (ESM)

CPU-first, zero-dependency Rust workspace for engineering online sparse encoders
that learn latent-role representations beyond token identity.

**Status: E-1A ✅ | E-1B ✅ | E-1C ✅ | E-1D ✅ (all six pass conditions met)**

## Gates

| Gate | Problem | Approach | Status | Report |
|------|---------|----------|--------|--------|
| **E-1A** | Do sparse encoders carry latent-role info? | Predictive v2 + dense decoder probes | ✅ PASS (corrected) | [`docs/E1A_EXPERIMENT_REPORT.md`](docs/E1A_EXPERIMENT_REPORT.md) |
| **E-1B** | Can delayed credit bridge temporal gaps? | PendingClaim pool + Jaccard verification + rent eviction | ✅ PASS (FixedContext/CyclingContext 100%) | [`docs/E1B_PENDINGCLAIM.md`](docs/E1B_PENDINGCLAIM.md) |
| **E-1C** | Does composition (D>1) outperform flat encoding? | CompositionPredictiveEncoder with triple-interaction sketch | ✅ PASS (D=2: 100%, D=3: 99.99% > D=1: 50%) | [`docs/E1C_COMPOSITION_DEPTH.md`](docs/E1C_COMPOSITION_DEPTH.md) |
| **E-1D** | Can the system bootstrap from zero prior knowledge? | ElementStore + GenesisManager + coverage/disagreement tracking | ✅ ALL 6 CONDITIONS MET | [`docs/E1D_GENESIS.md`](docs/E1D_GENESIS.md) |

## CLI

```sh
# E-1B PendingClaim
cargo run -- run e1b --encoder comp --stream fixed-context --steps 50000 --claim-gap 5

# E-1C Composition Depth
cargo run -- run e1a --stream d1 --encoder comp --steps 20000    # control (50%)
cargo run -- run e1a --stream d2 --encoder comp --steps 20000    # pairwise XOR (100%)
cargo run -- run e1a --stream d3 --encoder comp --steps 20000    # triple XOR (99.99%)

# E-1D Genesis (five test streams)
cargo run -- run e1d --e1d-stream empty   --encoder comp --steps 10000
cargo run -- run e1d --e1d-stream novel   --encoder comp --steps 10000
cargo run -- run e1d --e1d-stream rare    --encoder comp --steps 10000
cargo run -- run e1d --e1d-stream weak    --encoder predictive --steps 10000
cargo run -- run e1d --e1d-stream compgap --encoder comp --steps 10000 --genesis-disagreement-rate 0.4

# Legacy E-1A (archived encoders still runnable)
cargo run -- run e1a --stream same-token-context --encoder e0 --steps 50000
```

## Active Encoders

| Kind | CLI alias | Notes |
|------|-----------|-------|
| `PredictiveEncoder` | `predictive` | Sparse projection + context-key role prototypes |
| `ContextPredictiveEncoder` | `context` / `ctx` | Context-dominant (85% weight) for E-1B bridge |
| `CompositionPredictiveEncoder` | `composition` / `comp` | Balanced weights + triple-interaction term (E-1C) |

## Architecture

```
crates/
  esm-core/
    encoder/          SparseEncoder trait + three active encoders
    claims.rs         PendingClaimPool, ClaimConfig, Jaccard verification
    genesis.rs        Element, ElementStore, GenesisManager, CoverageTracker
    event.rs          InputEvent, TargetEvent (prequential protocol)
    feature.rs        FeatureId, SparseCode
    metrics.rs        dense_CPI, E1aMetrics
    rng.rs            Deterministic hash-based RNG
  esm-runner/
    e1a.rs, e1b.rs, e1d.rs    Experiment harnesses
    stream.rs         Synthetic streams (all gates)
  esm-cli/            CLI entry point
  esm-tools/          Development utilities
```

## Design Constraints

- **CPU-first.** No GPU, no PyTorch/Candle/Burn.
- **Safe Rust only** in core crates. No `unsafe`.
- **Zero external dependencies** in `esm-core`.
- **Prequential protocol:** `encode` never sees `TargetEvent`.
- **No MLP/attention/global hidden state** in the sparse encoder.

## Key Scientific Findings

1. **E-1A:** Predictive v2 sparse encoder carries latent-role information (dense_CPI +0.244 at 100K steps). Prior "FAIL" was from MLP backprop bug + insufficient steps.
2. **E-1B:** PendingClaim with Jaccard verification works (100% on fixed/cycling context). Random context prevents cross-cycle transfer — fundamental stream design constraint.
3. **E-1C:** Composition encoder with balanced weights (token^ctx=10 > ctx=9) + triple-interaction term passes D=2/D=3 XOR tasks. Prior encoders fail because top-k selection + homeostasis kills conjunctive columns.
4. **E-1D:** Cold-start bootstrap works through probe creation, rent-based eviction, promotion, and disagreement-triggered genesis. All six pass conditions verified across five test streams.
