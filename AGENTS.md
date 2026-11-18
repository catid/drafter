# Working instructions

`prompt.md` is the authoritative project specification. Read it before starting
work and recheck the relevant requirements after each round of work.

Before running any experiment, benchmark, training job, or evaluation:

- **Optimize Before Long-Running Experiments**: Profile and optimize hot paths,
  device utilization (e.g., multi-GPU sharding, NUMA/core pinning, batched/fused
  ops, zero host syncs), and memory/IO bottlenecks *before* launching
  long-running jobs—never burn hours on unoptimized loops when a validated
  fastpath can run an order of magnitude faster.
- **Mandatory Pre-Experiment Bug-Check Gate**: Always double-check all new or
  modified code, optimizations, configs, launch scripts, and validators for bugs
  *before* running experiments with them. Require parallel independent
  verification (line-by-line review, conformance audit, and
  adversarial/differential testing, including target-device CUDA checks before
  GPU runs) with zero open bugs.
- **Periodic Parallel Bug Checks**: Periodically run independent bug checks in
  parallel across active branches, launch/driver scripts, validators, and live
  experiment artifacts/manifests to catch subtle numerical, state, concurrency,
  provenance, or schema bugs early.
- **Standing 20-Minute Bug-Hunt Rotation**: While any long-running GPU
  training, evaluation, or benchmark job is active, run an independent
  multi-angle parallel bug-hunt and artifact/manifest conformance check every
  20 minutes across active driver processes, output shards, validators, and
  staged next-phase code so defects are caught and fixed before GPU transitions.

After each round of work:

- Thoroughly double-check the changes and results for bugs, using independent
  review and appropriate validation where useful.
- Double-check `prompt.md` to confirm that the work advances the full project,
  including the mandatory reference experiments and reporting.
- Do not substitute fallbacks, smoke tests, pilots, partial implementations, or
  narrower goals for the requested end state. Record incomplete work honestly
  and continue toward the full requirements.
- Claim completion only when the required behavior and artifacts are supported
  by current evidence. A passing test suite alone does not complete the project.
