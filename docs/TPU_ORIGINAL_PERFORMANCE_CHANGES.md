# Execution-layer performance fork

This source is a performance fork of the embedded DouTPU notebook archive.
The original design documents remain as historical baseline documentation.

- Retain the previous bounded action-feature cache and PAD-only training buckets.
- Subclass only the upstream observation builder, not rules, step, or terminal logic.
  Public snapshots own immutable tuples/bytes. Official opponents request full
  upstream InfoSets on demand; oracle labels are never policy inputs.
- Append public-history tokens once per action, including passes and role markers.
- Pack actor messages into NumPy arrays and insert completed samples in bulk.
  Oversized writes retain the same ring indices and the most recent capacity rows.
- Select actions on-device across every action chunk, return indices and a finite
  flag once, preserve first-index ties and the original interleaved NumPy RNG draws.
  Single-action states still execute finite-score validation.
- Borrow idle training workers for synchronous paired evaluation. Deal order,
  random opponent draws, role exchange, confidence intervals, promotion criteria,
  and paused training games are preserved. Official CPU opponents remain serial.
- Add visible parent-cgroup CPU limits and conservative joint workload presets.
- Record replay/sample/evaluation/checkpoint timing and the sample-update ratio.
- Retain verified fork_run.py migration into a NEW experiment with source locks.

The model, optimizer, losses, and effective batch remain unchanged. Different
actor counts and floating-point reduction layouts do not promise bit-exact
continuation. See V6e_NextStage_Report.md for the measured comparison, validation
scope, and remaining limits. Checkpoint writes remain synchronous; no background
thread receives mutable replay views. In-flight actor games still restart on
resume. No cross-version pipeline, KV cache, shared memory, fused attention, or
buffer donation was added. Never overwrite the original experiment with this source.

## Ordered pipeline stage

- `actor_groups` splits one logical actor round into contiguous worker groups.
  Each worker advances exactly once, with the same frozen policy and epsilon.
  CPU work for an earlier group can overlap inference for a later group.
  Ready-first receives are restored to original worker order before replay writes.
  The whole round is drained before learning, evaluation, saving, or stopping.
- `learner_prefetch` prepares at most one future batch while an independent Adam
  update is executing. It uses a private RNG copy and commits the main RNG only
  when that batch is consumed. Stopping discards unused speculation. Preparation
  errors are deferred until the current optimizer result is accounted for.
- `learning_seconds` measures the whole learning section. With prefetch enabled,
  sample and learner timers overlap and must NOT be added. `actor_seconds` is
  parent dispatch/drain wall time, not total CPU worker execution time. Grouped
  padding statistics are reported per group under `padding.groups`.

The v6e recommendation is in `configs/tpu_v6e_pipeline.json` and
`V6e_Pipeline_Report.md`, with raw A/B/B/A evidence. Smaller inference batches can
change floating-point outcomes; bitwise equality is a tested-window observation,
not a universal guarantee. In particular, four groups did not pass full-state
equality and were slower in the short screen; they are not recommended.
The general TrainConfig defaults remain conservative (one group, no prefetch).
There is no actor/learner policy-version lag and no change to the four optimizer
updates or effective batch of 256. No additional dependency was added.

## Phase-local replay indices

The learner builds eligible physical indices for each role once per phase. The
readiness check and all four samples reuse these arrays; no persistent cache is
kept. The next phase rebuilds them after collection or a version change. Direct
`Replay.sample` callers still rebuild by default. Role order, random draws,
shuffle, weights, history grouping, and prefetch RNG commit semantics are unchanged.

`cycle_end.full_cycle_seconds` measures the completed cycle, including evaluation
and synchronous checkpoints within that cycle. It excludes process startup,
session-start/session-end checkpoints and writing the cycle_end record itself.
`updates_completed` is deliberately distinct from the legacy train event's
`successful_steps`. For throughput, sum updates_completed and divide by summed
full_cycle_seconds over the same warmed window. Do not average per-cycle rates.

The old train event and cycle_seconds keep their pre-evaluation meaning.
replay_index_seconds records index preparation; sample_seconds records the
remaining preparation calls and still overlaps learner_seconds with prefetch.
Checkpoint time is a subset of phase wall times, not another additive phase.
collection_seconds and learning_seconds are host wall intervals, not device
utilization. Process wall time is required to include startup and shutdown costs.

The independent replay_benchmark compares the previous source on a verified
checkpoint, including readiness, four samples and history grouping. It reports
host preparation only, not end-to-end throughput. Round-two test evidence and
limitations are in the delivered Optimization_Review.md.

Tests (after bootstrap, using the pinned upstream rules):

```sh
PYTHONPATH=.:vendor JAX_PLATFORMS=cpu python -m pytest -q tests
```
