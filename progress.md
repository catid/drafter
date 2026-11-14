# Handoff: DFlash energy-adapter reference study

Snapshot: **2026-11-14 03:43 UTC**. Read this together with `AGENTS.md` and the
entire `prompt.md` before continuing. The user requested this handoff and a pause
in agent work. **The reference experiment is intentionally still running.**
Do not launch a duplicate job or treat the project as complete.

The user cleared the previous research automation goal and set the current goal
to **"Clean up and prepare for hand-off to another agent"**. Completing this
handoff goal does not complete M0–M4; the next agent should continue the research
objective described below when taking over.

## Objective and working rules

Complete all mandatory **M0–M4 at reference scale**, then determine whether the
adapter improves closed-loop acceptance and net throughput after all costs.
Implementation, tests, smoke runs, and partial experiments are not substitutes
for the required real-model evidence. Optional `--scale full` and M5 are outside
the current task; “complete the full project” means the complete mandatory
reference plan, not changing to full-scale experiments.

The user explicitly requires thorough bug checks and a `prompt.md` conformance
review after each round. This reminder is committed in `AGENTS.md`. Preserve the
specified prompt counts, seeds, generation caps, timing coverage, exact objectives,
and verification rule. Report negative results and sparse bins honestly.

Use the project `.venv`; do not change drivers, CUDA, system packages, or vendored
DFlash. Do not add custom kernels or divert into M5. The required pair is
`Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash-b16`, using Transformers. Cached models
and datasets are already available. The active run uses offline environment
flags. No credential from chat is needed or stored here.

## First action: inspect the existing job

Repository: `/home/catid/drafter`, branch `main`, remote
`git@github.com:catid/drafter.git`. All implementation and completed review code
were already pushed through `c6c0caa3fe491d0f59e80f429a3dfd22d004d7aa`; this
handoff is a subsequent documentation commit. Production implementation remains
the code introduced at `4b9c3c6`.

At the snapshot, these processes were alive:

| Item | Value |
| --- | --- |
| Continuation driver PID | `2946860` |
| Driver `/proc` start ticks | `179200997` |
| Current M3 tuning child PID | `2946875` |
| Child `/proc` start ticks | `179201686` |
| Root tool exec session | `12898` (may not be accessible from another agent/session) |
| Physical GPU for current tune | `0` |
| Current attempt | `fa8c9571901b4cd38b16fb411102a292` |
| Corrected tune start | `2026-11-14T02:16:38.558762+00:00` |

Use actual processes, not this snapshot, to determine current status. The driver
will change its child PID as it advances. Read:

- `artifacts/reference_continuation_state.json`: current child, exact argv,
  environment overrides, attempts, stage gates, and frozen file bindings.
- `artifacts/reference-continuation.log`: driver history.
- `artifacts/reference_continuation_logs/`: logs for each stage.
- `artifacts/current_work_state.json`: operational notes, latest observation, and
  outstanding reporting obligations. Some earlier nested fields are historical;
  prefer `latest_runtime_observation`, current M3 fields, and actual processes.

The previous agent's auxiliary read-only watcher (tool session `64644`, Python
PID `2947199`) was stopped and confirmed absent at 03:43:43 UTC. The experiment
driver and evaluator were not signaled. The next agent owns any further manual
monitoring; no previous auxiliary watcher should be assumed active.

Safe read-only inspection:

```bash
cd /home/catid/drafter
.venv/bin/python - <<'PY'
import json
from pathlib import Path
s = json.loads(Path('artifacts/reference_continuation_state.json').read_text())
print('status:', s['status'])
print('inflight:', s.get('inflight'))
records = [s.get('driver_process')]
if s.get('inflight'):
    records.append(s['inflight'].get('process'))
for record in filter(None, records):
    path = Path('/proc') / str(record['pid'])
    if not path.exists():
        print('missing process:', record['pid'])
        continue
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    print('process:', record['pid'], 'state:', stat[0],
          'start_matches:', int(stat[19]) == record['start_ticks'])
    print('argv:', (path / 'cmdline').read_bytes().decode().split('\0')[:-1])
PY
tail -n 20 artifacts/reference-continuation.log
```

The logs contain an **old intentional termination** from the repetition-detector
correction. A historical `m3_tune exited -15` traceback is not a failure of the
current attempt. Poll the current process/session. A timeout or quiet log is not
a terminal state and does not justify a restart.

## Current progress and automatic continuation

The corrected M3 tune has **79/140 measured requests** and **11/20 complete prompt
bundles** at this snapshot: all 8 standard prompts and the first 3 long prompts
are complete; the fourth long prompt has 2/7 candidates complete. M3 test
evaluation, isolated timing, and M4 are still pending.

Only count the current corrected session:

```text
logical_session: 00df5018aa1946d6821bd38a71de77b8
implementation: 60da3e0c57fee2fbd9cd7ce756c66c56ba5642a63da27ef428b9b30538a459df
manifest: artifacts/m3/tune/manifest-rank0.json
session: artifacts/m3/tune/sessions/baa0c5cee13bb21d83155d21791bbe56bcc00e8995f9880fa5a44278baced860.json
```

The manifest also contains **58 historical rows**. Filter jobs by
`identity.timing_session` equal to the logical session above,
`identity.checkpoint.implementation` equal to the implementation above, and
`status == 'complete'`; do not count every manifest entry. Session `bundles`
records complete comparisons. Final settings will be written to
`artifacts/m3/settings.json` only after all development comparisons finish.

Completed long bundles are `math500-long-0176`, `math500-long-0151`, and
`math500-long-0134`. Fourth prompt `math500-long-0497` has:

- dense, beta 1.0: 15,184 tokens, 7,281 blocks, EOS, 370.459921 seconds;
- dense+sparse, beta 0.5: 13,628 tokens, 4,917 blocks, EOS, 266.296089 seconds.

Both fourth-prompt outputs are unflagged by the repetition detector. Their raw
checksums, identities, token counts/committed streams, finite metrics, acceptance
arithmetic, full-vocabulary mode, and ratio rule passed bounded checks. These are
development observations, not final speed comparisons or semantic correctness
proofs. Earlier complete bundles were also checked. Do not create another audit
script/archive for each request; audit complete stages when available.

`artifacts/reference_continuation_source.txt` already runs the complete sequence:

```text
M3 tune → M3 evaluate → M3 isolated timing/profile → M3 summary gate
        → M4 tune → M4 evaluate → M4 isolated timing/profile → M4 summary gate
```

The driver does not write the final empirical README/SUMMARY/NOTES or complete
the goal. It must be followed by independent final checks and reporting.

| Work | Required schedule |
| --- | --- |
| M3 tune | 140 measured requests + 14 warmups |
| M3 evaluation | 1,920 requests + 64 shared target-only greedy diagnostics |
| M3 isolated timing | 5,760 fresh requests + 1,920 separate profiles; 8 warmups |
| M4 tune | 180 measured requests + 18 warmups |
| M4 evaluation | 2,400 logical rows |
| M4 isolated timing | 7,200 fresh requests + 2,400 separate profiles; 10 warmups |

Development uses 8 standard and 12 long prompts. Test uses 64 standard prompts
(three seeds per positive temperature) and 32 long prompts (one seed). Standard
cap is 2,048; long cap is 24,576. Test evaluation uses process-per-GPU sharding on
GPUs 0–7. Timing uses an isolated GPU 0 process, **all prompts**
(`--timing_prompts 0`), three repeats, one warmup, and a separate full profile
pass. Do not start unrelated GPU work that would interfere with these stages.

M4 may reuse matching M3 plain/bigram acceptance and output evidence only.
Every timing/profile baseline must execute fresh alongside the relevant arms.
An earlier optional timing-reduction question was never answered: **no scope
reduction is authorized**. Runs may take days. Arithmetic scenarios in
`artifacts/reference_compute_plan.json` are limited examples, not reliable ETAs.

## Freeze and recovery rules

All **36 frozen source/artifact bindings** matched at the handoff review. Do not
edit production sources, checkpoints, or bound reports while this driver runs.
`progress.md` is not a bound input. A concrete defect requires investigation and
a reviewed correction; slowness or negative scientific results do not.

```text
driver SHA256:
798a272ff45774e056c0fbacd41366898ca8b0252b00d0267edd80321c73f5d0
plan SHA256:
a5bc136c88537c1923d372eec1af657092444d42ca248dcf65e84fd0f813ec3f
```

If the driver genuinely exits, inspect its state, latest attempt, child/process
group, and logs before acting. It deliberately refuses a duplicate driver or an
unresolved previous `inflight` attempt. **Do not clear state or delete locks just
to make it run.** Resolve any surviving child/unknown outcome first. Preserve
settings, GPU order/count, visibility, and session identities; do not use
`fresh_session` for an ordinary resume.

Only after confirming a clean terminal state, no surviving descendants, and
understanding any failure, the reviewed normal continuation command is:

```bash
.venv/bin/python artifacts/reference_continuation_source.txt --scale reference
```

The driver replays resumable commands; completed compatible work is reused.
`--dry_run` safely prints its plan but does not prove run completion. No restart
is needed or authorized merely because the previous agent paused.

## Completed reference evidence and scientific result

**M0, M1-pre, M1a, corrected M1b, and the M2 protocol passed.**

- M0: 60 upstream/local comparisons matched, covering 7,128 tokens. Both paths
  differ from target-only greedy on 8/20 sequences. BF16 ties, masks, and cache
  rollback were investigated; see `artifacts/m0.json`, `m0_numerics.json`, and
  `m0_current_validation.json`.
- Calibration: 64 train-only prompts, 22,377 hidden vectors; projection
  `[256,4096]`, token features `[151936,64]`. See `artifacts/calibration.json`.
- M1: 2,184 shards, 256,836 blocks, **501,510 primary train output tokens**,
  excluding long and pure-temperature diagnostic traces. Splits are disjoint.
  See `artifacts/m1_audit.json`, `diagnostic_reference.json`, and
  `m1b_validation.json`.
- Long collection has 12 dev/32 test generations. Repetition exclusion leaves
  12 dev/31 primary test outputs; 24 primary test outputs exceed 8K, only 2 exceed
  16K. Do not fill sparse bins artificially.

M2 is a **negative quality result**, not an unfinished fit:

| Split | Zero-adapter grouped KL | Dense | Dense+bias+sparse |
| --- | ---: | ---: | ---: |
| Train | 1.16525348 | 1.92823779 | 1.95928777 |
| Dev | 1.16845300 | 2.01809236 | 2.06024947 |
| Test | 1.21701844 | 2.85879702 | 3.01548795 |

Both fixed fitting phases completed. Extra tuning stopped under the <1%
improvement rule; final dev regression is 76.3%. Two independent reviews found
no proven sign/scaling/objective defect. Do not retune to force a positive result.
The exact full-vocabulary denominator is reconstructed for compact grouped KL;
M4 uses exact live unwarped full-vocabulary KL at unit training temperature/beta.

The original fit finished optimizer work but exact sparse-key accounting exceeded
its 512 MB host limit. Guarded finalization used a 32 GB accounting limit with
**zero optimizer/proximal/chunk writes**, preserving checkpoints, RNG, optimizer,
configuration, and chunk bytes. Evidence:

- `artifacts/m2.json` and `artifacts/m2/{dense,dense_sparse,latest}.pt`;
- `artifacts/m2_validation_reference.json`;
- `artifacts/m2_recovery_validation_reference.json`;
- `artifacts/m2_finalize_recovery/execution.json`.

Exact accounting found 105,410,034 distinct keys, 1,294,937,088 events, 88,666,459
collisions, and 150,394 nonzero buckets. Existing checkpoint fit identities retain
the original configuration. README's 32 GB command is for a **fresh** fit, not a
reason to alter or rerun this completed fit.

## Resolved bug, validation, and remaining reporting limits

The earlier repetition detector missed periods over 64. A 96-token M3 loop and
a 1,508-token M1 test loop exposed the defect. The old M3 attempt was intentionally
stopped and archived. Version 2 scans every eligible exact period; corrected M1
reports and fresh M3 development comparisons use it. Raw M1 traces, calibration,
fitting sources, and M2 checkpoint bytes were preserved.

Evidence is under `artifacts/repetition_detector_review/`. Validation included
**263 passing CPU tests**, complete M3/M4 smoke pipelines, and **16,255** M1
correction invariants. Fresh standard M3 rows exactly reproduced historical
tokens/blocks; the historical long loop also reproduced and is now correctly
flagged. Smoke checks do not constitute reference completion.

The fresh conformance audit found no synthetic substitution, candidate-restricted
primary scoring, upstream-native stochastic sampler substitution, unshifted-rest
objective, reduced test/timing population, or M5 diversion. Source and completed
raw rows support those findings; planned future commands do not prove completion.

Important interpretation limits:

- `diagnostic_reference.json` has `long_context_conclusions_supported=true` based
  only on sufficient 8K–16K coverage. `empty_bins` means physical zero-count bins,
  while `data_sparsity=true` flags inadequate coverage. The global flag does
  **not** support 16K+ inference. Only two dev and two primary test generations
  reach that bin. `reports/SUMMARY.md` already explicitly rules out a stable
  conclusion. Final reporting must state eligibility per bin, retain descriptive
  counts, and mark unsupported 16K+ primary conclusions unavailable.
- The M2 held-out live-full comparison covers 120 slots per split. It is not
  full trace-wide recomputation; hidden-head reconstruction is tolerance-based,
  not bit-exact.
- Current partial dev timings use different realized outputs and are not the
  isolated matched timing evidence required for throughput claims.

## What the next agent must finish

1. Keep monitoring the existing driver. Investigate concrete failures without
   reducing the schedule. Let full M3 and M4 tuning, evaluation, timing/profile,
   and summary gates finish.
2. Audit complete M3 and M4 artifacts: exact populations, identities/checksums,
   frozen development settings, repetition exclusions, prompt-level bootstrap
   intervals, seed variability, realized bins, greedy mismatch reporting, fresh
   baseline timing, phase timing, and memory. Validate summaries against raw
   evidence; do not rely on completion flags alone.
3. Run the prepared independent M4 validator on **actual completed reference
   artifacts**:

   ```bash
   CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/m4_evidence_validator/validate.py \
     --scale reference --artifacts artifacts \
     --output artifacts/m4_validation_reference.json
   ```

   Validator preparation is committed at `83aacf9`: 236 smoke records, including
   96 online requests/947 blocks, passed 29,450 checks plus 36 wrapper fault
   probes and 182 online probes. It verifies schedules, provenance, tuning
   selection, online versions/counters/norms/state bytes, timing and memory.
   JSON flags alone cannot prove tensor immutability, zero initialization,
   absence of buffered gradients, gradient correctness, model exactness, or
   GPU isolation; retain source/test and final operational review for those.
4. Write the final empirical `README.md`, `reports/SUMMARY.md`, `NOTES.md`, and
   `artifacts/milestones.json`. Include negative results, supported/unsupported
   bins, all costs, limitations, and external baselines not run. The driver
   deliberately does not perform this reporting step.
5. Perform a requirement-by-requirement audit of all of `prompt.md`, including
   its 25 required tests and repository/reporting requirements; fix real defects
   and run appropriate validation. Commit and push the completed work.
6. Claim the research project complete only when every mandatory reference
   requirement and final deliverable is proven. The separate handoff task can
   finish while this experiment runs; it is not research project completion.

## Git, local evidence, and SSH

All current implementation sources are tracked. Untracked files are runtime
state/logs/results and obsolete pre-correction archives; do not `git add -A`.
Large checkpoints, trace payloads, and detailed trajectories remain local by
design. A Git clone alone does not contain the experiment evidence needed to
resume this server's run. Preserve this workspace and its artifacts.

The user authorized pushing code and requested the latest forwarded SSH agent
from tmux. At handoff, **tmux session `0`** had the working agent; tmux's global
environment was stale. Resolve the session value dynamically:

```bash
drafter_tmux_agent=$(tmux show-environment -t 0 SSH_AUTH_SOCK)
export SSH_AUTH_SOCK="${drafter_tmux_agent#SSH_AUTH_SOCK=}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=10'
git push origin main
git ls-remote origin refs/heads/main
```

If it expires, inspect current tmux session environments and owned socket paths;
do not inspect private keys. GitHub `main` and local HEAD matched `c6c0caa` before
this handoff. No production source changes were pending.
