# Research status

As of 2026-11-14, reference M0, M1-pre, M1a, and M1b have **passed**. Reference M2
offline fitting is **incomplete pending finalization recovery**; reference M3 and M4 are **not_run**. The completed
dense epoch worsened offline dev loss; final fitting results, closed-loop quality,
online adaptation, and end-to-end speed are still pending.
Synthetic tests and smoke results do not satisfy the real reference gates.

| Milestone | Reference status | Evidence or remaining work |
| --- | --- | --- |
| M0 | passed | Upstream/local reproduction and target-only mismatch investigation |
| M1-pre | passed | 64 train calibration prompts; 22,377 hidden vectors; frozen hidden and token features |
| M1a | passed | 2,184 checksummed shards; 256,836 blocks; all 12 audit requirements pass |
| M1b | passed | 106 diagnostic groups from all 2,184 audited shards; source and calibration provenance validated |
| M2 | incomplete | Both training epochs finished; recovering exact sparse statistics after a memory-limit failure |
| M3 | not_run | Dev selection, four-arm test evaluation, and isolated timing pending |
| M4 | not_run | Dev selection, five-arm online evaluation, and separate isolated timing pending |

The authoritative status snapshot is
[milestones.json](../artifacts/milestones.json). The
[implementation audit](../artifacts/implementation_audit.json) distinguishes code
and test coverage from incomplete reference experiments. Exact commands and stage
gates are in [README](../README.md#reference-execution-m0m4).

The reference pair is Qwen/Qwen3-8B and z-lab/Qwen3-8B-DFlash-b16. Sampling conditions
come from the pinned checkpoint files. The mandatory ratio verifier was used for
stochastic comparisons. Upstream and local matched on all 20 prompts at each of
greedy, warped T=0.7 and pure T=1: 0/7,128 mismatched output-token positions.
Both differed from target-only greedy on 8/20 sequences and 607/2,388 token
positions. Same-prefix investigation reproduced all eight first divergences and
identified BF16 maximum-logit ties, with correct cache lengths and future masks.
See [m0.json](../artifacts/m0.json),
[m0_numerics.json](../artifacts/m0_numerics.json), and the
[current-source validation](../artifacts/m0_current_validation.json) for evidence.

M1-pre used only designated `standard_nonthinking` train prompts. The documented
fixed projection has shape `[256, 4096]`; the embedding-PCA token table has shape
`[151936, 64]`. Projection moments and token-feature statistics were frozen before
trace collection. See [calibration.json](../artifacts/calibration.json).

M1a has collected 501,510 primary train output tokens: 250,790 greedy and 250,720
at the model-card warped `T=0.7` setting. Standard dev/test and pure-temperature
diagnostic generations are also collected. The independent long-thinking set has
12 dev and 32 test generations at `T=0.6, top_k=20, top_p=0.95, min_p=0`, with
thinking enabled. This is out of distribution for the drafter. Long dev/test outputs
contain 151,225/361,144 tokens; none are repetition-dominated, and two reached the
24,576-token cap. All 12 dev and 25 test outputs exceed 8K. Only 2 dev and 3 test
outputs exceed 16K, so that bin has sparse coverage. No adaptation benefit is
established by these lengths. See [m1_audit.json](../artifacts/m1_audit.json).

M1b completed in 313 seconds using eight CPU workers. Its
[validation record](../artifacts/m1b_validation.json) confirms the ordered source
paths and checksums, accepted audit, frozen calibration, prompt manifest, and
recorded source code. The [full diagnostic report](../artifacts/diagnostic_reference.json)
separates conditions, temperatures, splits, output/context bins, and post-rejection
teacher positions.

The following plain-DFlash results use the **test split, on-policy positions,
primary inclusion, and all output offsets**. Overlap is `sum(min(P, q0))` for the
condition's sampling distribution; at greedy temperature it is argmax agreement.
Overlap and target-mass means aggregate positions within prompts before the
prompt-ID bootstrap. Accepted length is the mean over draft rounds, excluding
replacement and bonus tokens; the maximum full block has 15 unknown proposals.
M1 uses one collection seed per prompt and condition, so it does not estimate
seed-to-seed variability.

| Collection condition | Test prompts | Mean accepted length | Sampling overlap, mean [95% CI] | Target mass in DFlash top-256 |
| --- | ---: | ---: | ---: | ---: |
| Standard greedy | 64 | 5.218 | 0.8517 [0.8442, 0.8587] | 99.843% |
| Standard recommended: T=0.7, top-k=20, top-p=0.8 | 64 | 4.873 | 0.8413 [0.8332, 0.8491] | 99.824% |
| Standard pure-temperature T=1 diagnostic | 64 | 4.192 | 0.8183 [0.8077, 0.8282] | 99.830% |
| Long thinking recommended: T=0.6, top-k=20, top-p=0.95 | 32 | 2.499 | 0.7149 [0.6995, 0.7308] | 99.833% |

All listed arms use `min_p=0`. In the recommended standard and long conditions,
the target's top token has mean draft rank 3.649 and 4.259, respectively. At first
rejection it lies within the draft top-256 in 98.95% and 99.50% of cases. These
coverage measurements describe potential reranking/calibration headroom; they do
not establish an adapter improvement or its net compute benefit.

Within long-thinking test traces, on-policy overlap is 0.7397 [0.7199, 0.7620]
at output offsets 4K–8K (31 prompts), and 0.6344 [0.5994, 0.6727] at 8K–16K
(25 prompts). Later bins include only prompts whose outputs reach those offsets.
Only three test prompts reach
16K+, where overlap is 0.6130 [0.5171, 0.7480]; that bin is too sparse for a stable
long-context conclusion. The full report also bins realized input-context length
at 1K, 4K, 8K, 16K, and 32K.

Projected-feature statistics for the same recommended-sampling test groups are
pooled over on-policy logical slots. The calibration reference has coordinate
mean zero and population variance one; a slot is an outlier if any standardized
coordinate has absolute value above 5.

| Condition | Coordinate mean-vector norm | Mean variance / calibration | Median feature L2 norm | Outlier slots |
| --- | ---: | ---: | ---: | ---: |
| Standard non-thinking, T=0.7 | 11.302 | 1.347 | 20.839 | 11.389% |
| Long thinking, T=0.6 | 8.306 | 0.847 | 16.563 | 0.439% |

Calibration uses GSM8K train prompts, standard test uses held-out HumanEval, and
long test uses MATH-500. The conditions also differ in thinking mode and sampling
settings, so their feature and acceptance differences do not isolate a causal
effect of thinking or output length. Long-thinking feature shift does not appear
as a higher standardized outlier rate in these measurements.

M2 used only the primary standard train traces for fitting; long-thinking traces
remain reserved for evaluation. After the single predeclared dense epoch, dev
weighted grouped KL rose from **1.16845 to 2.01809**, a **72.7% deterioration**.
Values remain finite. This negative intermediate result triggered the under-1%
rule stopping extra dense epochs. Completed dense evaluation also worsened train
KL from 1.16525 to 1.92824 and test KL from 1.21702 to 2.85880. The negative result
therefore spans all three splits. Final accounting for the bias/sparse phase remains pending;
no settings or optimizer state were changed in response. The
[dense phase record](../artifacts/m2_dense_interim.json) includes the checkpoint
checksum, parameter statistics, and separate live-full diagnostics from 120 dev
and 120 test slots. The bias/sparse epoch also worsened dev KL to 2.06025
(76.3% above baseline), so extra tuning stopped. Final sparse statistics exceeded
DuckDB's 512 MB memory limit after split evaluation. The saved checkpoint preserves
the completed training state; recovery will recompute evaluation and statistics
with a larger, explicitly recorded accounting limit and no optimizer updates.
Final fitting results remain pending.
Progress is recorded in [NOTES.md](../NOTES.md); the process log is
[m2-reference.log](../artifacts/m2-reference.log). Closed-loop quality and speed
results are pending. The planned M3/M4 timing retains full held-out prompt coverage
and fresh matched baseline measurements.

All 243 CPU tests pass. Complete M3 and M4 smoke runs also pass, including verified
baseline reuse, direct greedy comparisons, fresh isolated timing, sparse norms,
version boundaries, feature-context linkage, and parallel diagnostic parity;
see [validation.json](../artifacts/integrity_smoke_pipeline/validation.json).
The [fresh prompt audit](../artifacts/prompt_followup_audit.json) found and fixed
a timing-resume integrity gap: completion now requires checksummed raw bundles
covering every expected timing and phase-profile request. The active M2 fitting
sources remain unchanged.
CPU smoke paths exercise collection, compact grouped-loss fitting, all required
evaluation arms, and synchronous request-local online updates. Synthetic fitting
and reference-sized random-tensor microbenchmarks are implementation checks;
their results do not establish a reference acceptance or throughput improvement.

External baselines not run: full tree-based Token Recycling, Test-Time Speculation,
and SharpDraft. The required normalized online bigram baseline is implemented and
covered by smoke tests; its reference M3/M4 evaluations remain pending. Full-scale
experiments and optional M5 work have not run.
