# ESM Rust Implementation Plan

## Current scope

This repository implements the first narrow engineering target: **Gate E-1A representation quality**.
No ledger, claim, fork, genesis, router, or full ESM substrate is implemented yet.

This is intentional. The architecture states that E-1A is the main scientific risk. Later gates are not
worth implementing until sparse representation quality beats the raw token/hash control.

## Current crates

- `esm-core`: IDs, events, sparse codes, encoders, metrics.
- `esm-runner`: synthetic streams and E-1A runner.
- `esm-cli`: manual zero-dependency CLI.
- `esm-tools`: reserved for offline inspection.

## Current encoders

### Encoder A: `hash`

Raw token/hash control. It uses the current token only. It is meant to catch cases where the proposed
encoder is merely a complicated hash of token identity.

### Encoder B: `competitive`

Online sparse competitive encoder. It uses a bounded column pool and local WTA. Adaptation happens only
after observation.

### Encoder C: `predictive`

Predictive sparse encoder. It extends the competitive encoder with local latent-role statistics updated
after observation. This is still a toy mechanism, not a final ESM encoder.

## Current streams

- `same-token-context`: same ambiguous token appears under different latent contexts.
- `role-sharing`: different tokens share the same latent role.
- `delayed-role`: role is seeded earlier, then ambiguous tokens appear before disambiguation.

## Diagnostics

- `controlled_predictive_info = token_nll - code_nll` under prequential count models.
- `same_token_context_split`: for same token across different roles, code distance should increase.
- `role_sharing`: for different tokens with the same role, code overlap should increase.
- `code_entropy` and `active_bits_avg`.

## Non-negotiable stop rule

If Encoder B/C do not outperform Encoder A on controlled predictive information and context/role diagnostics,
do not implement Gate E-1B or later. Redesign the encoder.

## Toolchain note

The container used to create this repository does not currently have `rustc` or `cargo`, so this repository
has been scaffolded but not compiled in that environment.
