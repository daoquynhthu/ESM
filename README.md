# Elastic Sparse Machine (ESM)

This repository is the first engineering workspace for the Elastic Sparse Machine architecture.
It intentionally starts narrow: **Gate E-1A representation lab only**.

The current implementation contains:

- a Rust workspace with `esm-core`, `esm-runner`, `esm-cli`, and `esm-tools` crates;
- deterministic synthetic streams for E-1A;
- Encoder A/B/C controls:
  - `hash`: raw token/hash control;
  - `competitive`: online sparse competitive encoder;
  - `predictive`: predictive sparse encoder with local role statistics;
- prequential diagnostics for sparse representation quality.

The implementation is zero-dependency at this stage. It is designed to compile with stable Rust
when `rustc`/`cargo` are available. The current execution container used to create this repository
has no Rust toolchain installed, so the code has not been compiled in-container.

## Design constraints

- CPU-first.
- Safe Rust only in core crates.
- Integer IDs instead of object graph pointers.
- No `Rc<RefCell<_>>` in the core graph.
- No async runtime in the core.
- No target leakage: encode/predict happens before observe/adapt.
- E-1A must pass before ledger/claim/fork/router engineering begins.

## Intended command

```bash
cargo run -p esm-cli -- run e1a --stream same-token-context --encoder hash --steps 10000
cargo run -p esm-cli -- run e1a --stream same-token-context --encoder competitive --steps 10000
cargo run -p esm-cli -- run e1a --stream same-token-context --encoder predictive --steps 10000
```

## Gate E-1A stop rule

If `competitive` and `predictive` encoders do not beat the raw `hash` control on:

- same-token context split;
- role sharing;
- controlled predictive information;

then ESM scaling should stop and the encoder should be redesigned before implementing later gates.
