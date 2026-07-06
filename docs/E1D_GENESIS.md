# E-1D Genesis / Cold-Start

**Gate**: E-1D — Test that the substrate can bootstrap from zero prior knowledge
by creating probe structures (genesis) when the encoder does not explain the
current input.

## Architecture

E-1D extends E-1B/C with a parallel **ElementStore** that lives alongside the
encoder. Elements are sparse-feature structures similar to PendingClaims but
with utility, plasticity, resistance, and rent. The lifecycle is:

1. **Genesis trigger** — when encoder coverage is low AND surprise is high AND
   no adequate parent element exists (or only weak parents), create a probe.
2. **Observation** — each step, elements that overlap the current code observe
   the actual role and update their role_counts.
3. **Rent** — every step, all elements pay rent. Those whose rent exceeds
   utility get retired.
4. **Promotion** — probes with sufficient utility graduate to Active phase.
5. **Vote** — active/probe elements vote on the role, fused with encoder vote.

### Components

| Component | File | Role |
|-----------|------|------|
| `Element` | `genesis.rs` | Sparse feature set + role_counts + utility + rent |
| `ElementStore` | `genesis.rs` | Collection with lifecycle (rent, promote, retire) |
| `GenesisConfig` | `genesis.rs` | All tunable knobs (rent, utility floor, cooldown, etc.) |
| `GenesisManager` | `genesis.rs` | Step-level API (step_begin, after_encode, after_adapt, step_end) |
| `CoverageTracker` | `genesis.rs` | Monitors encoder column margins for "explained" features |
| `E1dStreamKind` | `stream.rs` | Five test streams (empty, novel, rare, weak, compgap) |
| `run_e1d` | `e1d.rs` | E-1D runner: encode → genesis → combine → adapt → lifecycle |

### Step Loop

```
for each step:
  genesis.step_begin()           // reset counters, decrement cooldown
  code = encoder.encode(input)
  margins = encoder.column_role_margins()
  surprise = -ln(encoder.confidence)
  genesis.after_encode(code, margins, surprise, feature_offset, num_roles)
    // checks: cooldown==0, coverage<0.5 or force_genesis, surprise>floor,
    //         parent_status in {NoAdequateParent,WeakParent}, budget
    // if all met: create_probe(features, kind, lineage, num_roles)
  element_votes = genesis.collect_votes(code)
  predicted = combine_vote(encoder_vote, element_votes)
  encoder.adapt(input, target, code)
  genesis.after_adapt(code, actual_role, encoder_predicted_role)
    // observes elements, tracks disagreements in sliding window
  genesis.step_end()             // rent, promote, retire
```

### Genesis Trigger Conditions

1. **Cooldown** == 0 (default 50 steps between genesis events)
2. **Coverage** < 0.5 OR **force_genesis_next** (bypass when disagreements high)
3. **Surprise** > surprise_floor (0.5) — skipped if force_genesis_next
4. **Parent status** == NoAdequateParent or WeakParent
5. **Budget** available (can_genesis + probes_per_step cap)

### Parent Status

Computed by `ElementStore::parent_status(code_features)`:

| Status | Condition | Genesis Kind |
|--------|-----------|-------------|
| `NoAdequateParent` | No elements, or best coverage < 30%, or best utility < 0.1 | `Round0` |
| `WeakParent` | Coverage 30-60% or utility 0.1-0.3 | `WeakParentRefinement` |
| `StableCompatibleParent` | Coverage ≥ 60% and utility ≥ 0.3 | No genesis |

### Disagreement Tracking

A sliding window (default 50 steps) tracks how often the encoder's predicted
role disagrees with the actual role. When the disagreement rate exceeds the
threshold (default 0.7 = 70%), `force_genesis_next` is set, which bypasses both
coverage and surprise checks. This catches cases where the encoder is
confidently but persistently wrong (e.g., pairwise XOR that needs D≥2).

## Test Streams

| Stream | Steps | Pattern | What It Tests |
|--------|-------|---------|---------------|
| **empty-field** | 10,000 | Every step: random token, random context, random role | Probes die from rent; coverage ~6%; accuracy ~50% |
| **novel-pattern** | 10,000 | Steps 0-4999: noise. Steps 5000+: token→role mapping | Genesis during noise, encoder learns, probes die |
| **rare-event** | 10,000 | 6-step delayed-cue, 90% role 0, 10% role 1 | Encoder learns quickly, barely triggers genesis |
| **weak-parent** | 10,000 | Token 100 + ctx 500 (role 0) / ctx 501 (role 1), interleaved | Overlapping features → WeakParent probes |
| **composition-gap** | 10,000 | Pairwise XOR: role = token_bit XOR prev_bit | Encoder confidently wrong → disagreement triggers genesis |

## Results

### Composition Encoder (10,000 steps)

| Metric | empty | novel | rare | weak | compgap |
|--------|-------|-------|------|------|---------|
| Probes created | 196 | 99 | 2 | 0 | 143 |
| Currently alive | 9 | 0 | 0 | 0 | 9 |
| Total retired | 187 | 99 | 2 | 0 | 134 |
| Total promoted | 104 | 50 | 0 | 0 | 142 |
| WeakParent triggers | 0 | 0 | 0 | 0 | 140 |
| Round0 triggers | 196 | 99 | 2 | 0 | 3 |
| Final parent status | NoParent | NoParent | NoParent | NoParent | StableParent |
| Coverage rate | 6.2% | 94.1% | 100% | 70.6% | 94.1% |
| Accuracy | 49.7% | 74.2% | 89.8% | 100% | 50.6% |

### Predictive Encoder (10,000 steps, key streams)

| Metric | weak | compgap |
|--------|------|---------|
| Probes created | 188 | — |
| WeakParent triggers | 10 | — |
| Round0 triggers | 178 | — |
| Accuracy | 97.0% | — |

### Pass Conditions

| # | Condition | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Genesis creates probe structures | ✅ | All streams create probes (0-196) |
| 2 | Probe survival nonzero but bounded | ✅ | Empty: 9/196 survive (4.6%) |
| 3 | Rent controls probe explosion | ✅ | Empty: 95.4% retired, never exceeds max_elements |
| 4 | No-parent case not stuck | ✅ | Empty: continuous create→retire cycle, cooldown prevents churn |
| 5 | WeakParent gap does not block learning | ✅ | 10-140 WeakParentRefinement probes created on weak/compgap |
| 6 | Round>0 composition gaps trigger genesis | ✅ | Disagreement tracking catches confidently-wrong encoder |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_elements` | 1024 | Max elements in store (all phases) |
| `max_probes` | 128 | Max probe-phase elements |
| `probes_per_step` | 2 | Max probes created per step |
| `rent_per_step` | 0.01 | Base rent deducted each step |
| `utility_floor` | 0.05 | Utility below this → retire |
| `surprise_floor` | 0.5 | Min surprise for genesis |
| `parent_coverage_floor` | 0.3 | Min element coverage for parent |
| `coverage_overlap_min` | 0.3 | Min overlap for "covering" |
| `cooldown_steps` | 50 | Steps between genesis triggers |
| `coverage_margin_threshold` | 10 | Min column margin to be "explained" |
| `disagreement_rate_threshold` | 0.7 | Disagreement rate to force genesis |
| `disagreement_window` | 50 | Sliding window size for disagr. tracking |

## CLI

```sh
esm run e1d --e1d-stream novel|empty|rare|weak|compgap \
            --encoder comp|ctx|predictive \
            --steps 10000 \
            --genesis-cooldown-steps 50 \
            --genesis-margin-threshold 10 \
            --genesis-disagreement-rate 0.7
```

## Future Work

- **Ensemble weight tuning**: element votes have low weight vs encoder; tuning
  could improve accuracy on composition-gap stream
- **Rent adaptation**: make rent proportional to element age or feature novelty
- **Interference tracking**: StableConflictParent (future layer) for elements
  that actively disagree
- **Lineage-based pruning**: old lineages with low utility get retired first
- **Ensemble weight tuning**: element vote weight vs encoder vote needs
  calibration for composition-gap streams
- **Rent adaptation**: make rent proportional to element age or feature novelty
- **Disagreement tracking on element votes**: not just encoder disagreements;
  if element ensemble consistently disagrees with encoder, that's also a signal
