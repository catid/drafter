# Research status

The repetition detector now checks exact cycles of any eligible length while
preserving the four-cycle minimum and 50% dominance threshold. Corrected M1
reports exclude one pathological long-test output; all raw traces and trained M2
checkpoints are preserved. A capped M3 dev loop exposed the defect, and those
development comparisons have restarted under the corrected implementation.
See the [confirmed evidence](../artifacts/repetition_detector_review/evidence.json)
and [correction validation](../artifacts/repetition_detector_review/m1_correction_validation.json).

As of 2026-11-14, reference M0, M1-pre, M1a, M1b, and the M2 protocol have **passed**.
M2 produced a **negative offline quality result**: dense and dense+bias+sparse
fitting worsened grouped KL on train, dev, and test. Fresh M3 development selection
is **running** after renewed M2 artifact validation; test evaluation and isolated
timing are pending. M4 is **not_run**.
Closed-loop quality, online adaptation, and end-to-end speed conclusions remain pending.
Synthetic tests and smoke results do not satisfy the real reference gates.

| Milestone | Reference status | Evidence or remaining work |
| --- | --- | --- |
| M0 | passed | Upstream/local reproduction and target-only mismatch investigation |
| M1-pre | passed | 64 train calibration prompts; 22,377 hidden vectors; frozen hidden and token features |
| M1a | passed | 2,184 checksummed shards; 256,836 blocks; all 12 audit requirements pass |
| M1b | passed | 128 corrected diagnostic groups from all 2,184 audited shards; one long-test output separately excluded |
| M2 | passed (protocol); negative quality | Both epochs, complete split evaluation, sparse accounting, and two validators complete |
| M3 | in_progress | Fresh dev selection running; four-arm test evaluation and isolated timing pending |
| M4 | not_run | Dev selection, five-arm online evaluation, and separate isolated timing pending |

The authoritative status snapshot is
[milestones.json](../artifacts/milestones.json). The
[implementation audit](../artifacts/implementation_audit.json) distinguishes code
and test coverage from incomplete reference experiments. Exact commands and stage
gates are in [README](../README.md#reference-execution-m0m4).

The reference pair is Qwen/Qwen3-8B and z-lab/Qwen3-8B-DFlash-b16. Standard
non-thinking sampling defaults come from the target model card; thinking defaults
come from the pinned `generation_config.json`. The mandatory ratio verifier was used for
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
thinking enabled. This is out of distribution for the drafter. Collected long
dev/test outputs contain 151,225/361,144 tokens, and two reached the 24,576-token
cap. The corrected [M1 audit](../artifacts/m1_audit.json) identifies one pathological
test loop (`math500-long-0346`): a 1,508-token cycle occupies 19,806 tokens.
Excluding that entire output leaves **12 dev and 31 test primary generations**,
with 151,225/336,568 tokens. All 12 dev and 24 primary test outputs exceed 8K;
only 2 dev and 2 primary test outputs exceed 16K, so that bin remains sparse.
These lengths establish no adaptation benefit.

M1b was regenerated using eight CPU workers in **331.0782 seconds** from launch
to final log output; the separate validation took **57.8425 seconds**. The
[execution record](../artifacts/repetition_detector_review/m1_correction_execution.json)
distinguishes those timings. The
[correction validator](../artifacts/repetition_detector_review/m1_correction_validation.json)
passed 16,255 checks covering provenance, trace and M2 byte preservation, and
reclassification. The [full diagnostic report](../artifacts/diagnostic_reference.json)
contains 128 groups: all 84 standard and long-dev groups are exactly unchanged;
22 long-test primary groups were recomputed and 22 repetition-excluded groups
were added. The excluded output contributes 67,775 logical slots across on-policy
and teacher positions. The validator did not independently recompute the changed
bootstrap intervals, feature moments, acceptance summaries, or rejection metrics.

The following plain-DFlash results use the **test split, on-policy positions,
primary inclusion, and all output offsets**. Overlap is `sum(min(P, q0))` for the
condition's sampling distribution; at greedy temperature it is argmax agreement.
Overlap and target-mass means aggregate positions within prompts before the
prompt-ID bootstrap. Accepted length is the mean over draft rounds, excluding
replacement and bonus tokens; the maximum full block has 15 unknown proposals.
It counts verified proposals before EOS trimming; a terminal block can therefore
count accepted proposals that are not emitted. Output-length and throughput
metrics use the actual committed output.
M1 uses one collection seed per prompt and condition, so it does not estimate
seed-to-seed variability.

| Collection condition | Test prompts | Mean accepted length | Sampling overlap, mean [95% CI] | Target mass in DFlash top-256 |
| --- | ---: | ---: | ---: | ---: |
| Standard greedy | 64 | 5.218 | 0.8517 [0.8442, 0.8587] | 99.843% |
| Standard recommended: T=0.7, top-k=20, top-p=0.8 | 64 | 4.873 | 0.8413 [0.8332, 0.8491] | 99.824% |
| Standard pure-temperature T=1 diagnostic | 64 | 4.192 | 0.8183 [0.8077, 0.8282] | 99.830% |
| Long thinking recommended: T=0.6, top-k=20, top-p=0.95 | 31 | 2.411 | 0.7115 [0.6973, 0.7256] | 99.829% |

All listed arms use `min_p=0`. In the recommended standard and long conditions,
the target's top token has mean draft rank 3.649 and 4.333, respectively. At first
rejection it lies within the draft top-256 in 98.95% and 99.49% of cases. These
coverage measurements describe potential reranking/calibration headroom; they do
not establish an adapter improvement or its net compute benefit.

Within primary long-thinking test traces, on-policy overlap is 0.7336 [0.7159, 0.7518]
at output offsets 4K–8K (30 prompts), and 0.6245 [0.5927, 0.6547] at 8K–16K
(24 prompts). Later bins include only prompts whose outputs reach those offsets.
Only two primary test prompts reach
16K+, where overlap is 0.5455 [0.5171, 0.5740]; that bin is too sparse for a stable
long-context conclusion. The full report also bins realized input-context length
at 1K, 4K, 8K, 16K, and 32K.

Projected-feature statistics for the same recommended-sampling test groups are
pooled over on-policy logical slots. The calibration reference has coordinate
mean zero and population variance one; a slot is an outlier if any standardized
coordinate has absolute value above 5.

| Condition | Coordinate mean-vector norm | Mean variance / calibration | Median feature L2 norm | Outlier slots |
| --- | ---: | ---: | ---: | ---: |
| Standard non-thinking, T=0.7 | 11.302 | 1.347 | 20.839 | 11.389% |
| Long thinking, T=0.6 | 8.314 | 0.851 | 16.600 | 0.461% |

Calibration uses GSM8K train prompts, standard test uses held-out HumanEval, and
long test uses MATH-500. The conditions also differ in thinking mode and sampling
settings, so their feature and acceptance differences do not isolate a causal
effect of thinking or output length. Long-thinking feature shift does not appear
as a higher standardized outlier rate in these measurements.

M2 completed one predeclared dense epoch followed by one dense+bias+sparse epoch.
Only primary standard train traces were fitted: 501,510 generated tokens from
853 prompts and 1,692 shards. Long-thinking and pure-temperature diagnostic traces
were excluded. Targets and proposals used unwarped unit temperature with
`T_train=beta_train=1`; teacher-forced post-rejection positions retained
`exp(-i/5)` weights. AdamW used learning rate 0.001, W/A weight decay 0.0001,
and no bias decay; sparse SGD used learning rate 0.01 and proximal L1 0.00001.
The final epoch checkpoint of each phase was retained; test results did not select
checkpoints. The under-1% development-improvement rule stopped additional epochs
and sweeps in both phases.

The complete eligible split results are below. Loss is weight-pooled grouped KL;
overlap is slot-pooled, with a union lower bound and grouped upper bound. These
bounds are not confidence intervals, and this aggregation differs from M1's
prompt-level bootstrap. Train, dev, and test contain 1,264,587, 102,975, and 95,160
logical slots, respectively, for every stage.

| Stage | Split | Weighted grouped KL | Union overlap lower bound | Grouped overlap upper bound |
| --- | --- | ---: | ---: | ---: |
| Zero adapter | Train | 1.165253 | 0.490937 | 0.492508 |
| Zero adapter | Dev | 1.168453 | 0.485760 | 0.487724 |
| Zero adapter | Test | 1.217018 | 0.481057 | 0.482343 |
| Dense | Train | 1.928238 | 0.490868 | 0.492709 |
| Dense | Dev | 2.018092 | 0.478438 | 0.480757 |
| Dense | Test | 2.858797 | 0.373201 | 0.374998 |
| Dense+bias+sparse | Train | 1.959288 | 0.496668 | 0.498334 |
| Dense+bias+sparse | Dev | 2.060249 | 0.482248 | 0.484323 |
| Dense+bias+sparse | Test | 3.015488 | 0.353714 | 0.355530 |

Dense dev KL is **72.7% worse** than the zero-adapter baseline; dense+bias+sparse
dev KL is **76.3% worse**. Both fitted stages worsen KL on all three splits.
Dense+bias+sparse modestly improves train overlap, so the negative loss result
does not mean every metric deteriorated. The cause of the adverse fit remains
unproven; passing protocol and consistency checks does not establish an improvement.

Full-vocabulary diagnostics cover only **120 dev slots and 120 test slots** per
stage, with no train slots. The following are subset measurements, not whole-split
full KL or top-1 estimates. Full and grouped KL use the same reconstructed
full-vocabulary proposal `q` on those slots and remain weighted means; overlap
and top-1 agreement are slot means.

| Stage | Split | Full KL | Grouped KL on same subset | Full−grouped KL | Full overlap | Top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Zero adapter | Dev | 1.421697 | 1.420899 | 0.000797 | 0.368720 | 0.416667 |
| Zero adapter | Test | 1.214361 | 1.214064 | 0.000298 | 0.372951 | 0.416667 |
| Dense | Dev | 1.688588 | 1.686864 | 0.001724 | 0.432503 | 0.450000 |
| Dense | Test | 4.076081 | 4.075018 | 0.001063 | 0.163167 | 0.166667 |
| Dense+bias+sparse | Dev | 2.190846 | 2.188882 | 0.001963 | 0.432736 | 0.450000 |
| Dense+bias+sparse | Test | 4.098733 | 4.097775 | 0.000958 | 0.202592 | 0.200000 |

Parameters are FP32. W has shape `[64,256]`, A `[4,64,64]`, b `[151936]`, and
the sparse table S `[16777216]`. Frozen token features remain unchanged; all
learned corrections are zero in the zero-adapter baseline.

| Stage | Parameter | L2 norm | Exact-zero fraction |
| --- | --- | ---: | ---: |
| Dense | W | 1.964834 | 0 |
| Dense | A | 2.332288 | 0 |
| Dense | b | 0 | 1 |
| Dense | S | Absent | — |
| Dense+bias+sparse | W | 2.030185 | 0 |
| Dense+bias+sparse | A | 2.466178 | 0 |
| Dense+bias+sparse | b | 218.121948 | 0 |
| Dense+bias+sparse | S | 0.468677 | 0.991036 |

Exact sparse accounting covers every lookup in all **10,539** sparse-stage
optimizer batches. A collision counts each additional distinct key sharing a
bucket; frequency-weighted collisions count all lookup events in buckets with
more than one distinct key.

| Sparse accounting quantity | Measured value |
| --- | ---: |
| Capacity | 16,777,216 buckets |
| Unique observed keys | 105,410,034 |
| Lookup events | 1,294,937,088 |
| Load factor | 6.282928 unique keys/bucket |
| Occupied buckets | 16,743,575 (99.799484%) |
| Collision count | 88,666,459 |
| Distinct keys in collided buckets | 105,204,151 |
| Frequency-weighted collision count | 1,292,522,353 |
| Final nonzero parameter buckets | 150,394 |
| Allocated sparse parameter bytes | 67,108,864 (64 MiB) |
| Nonzero value bytes only | 601,576 |
| Exact-accounting database bytes | 656,158,720 |
| Accounting chunk disk bytes | 3,242,960,898 |

Observed bucket occupancy and final nonzero parameters measure different things:
99.799% of buckets were accessed, while 99.104% of final parameters are exactly
zero after proximal shrinkage. The nonzero-value byte count excludes key and
index overhead and does not replace the allocated 64 MiB table.

The lookup diagnostic used GPU 0 (`cuda:0`), 4,096 queries per repeat, 3 warmups,
and 15 measured repeats. A deterministic hash sample selected 1,000,000 global
distinct keys, with frequency-weighted queries within that sample. The sorted
exact sampled table occupies 12,000,000 bytes and materializes the fitted
signed-hash value for each sampled key; returned values were bitwise equal.

| Lookup | Synchronized wall median ms | Wall p95 ms | Median bootstrap 95% CI ms | CUDA-event median ms |
| --- | ---: | ---: | --- | ---: |
| Signed feature hash | 0.213684 | 0.284236 | [0.210767, 0.218860] | 0.190848 |
| Sorted exact observed pairs (sample) | 0.043211 | 0.063135 | [0.043013, 0.043777] | 0.022624 |

The sampled exact lookup was faster in this diagnostic. It is not a separately
trained collision-free model or a table of all 105 million observed keys, and
this microbenchmark establishes no end-to-end speedup or quality benefit.

The original fit completed both optimizer epochs, then failed exact aggregation
at DuckDB's 512 MB accounting limit. Recovery repeated sparse train/dev/test
evaluation and accounting with a 32 GB finalizer limit, preserving the original
fit configuration, identity, six source hashes, parameters, optimizer state, RNG,
and committed chunks. Guards recorded **zero optimizer, proximal, or chunk-write
updates**. The reported 889.08 seconds covers this recovery evaluation/accounting
invocation only, not the original fitting runtime.

The [complete M2 report](../artifacts/m2.json),
[recovery execution record](../artifacts/m2_finalize_recovery/execution.json),
[main validator](../artifacts/m2_validation_reference.json), and
[independent recovery validator](../artifacts/m2_recovery_validation_reference.json)
record the evidence. Both validators passed: the main validator checked tensor
health, parameter statistics, provenance, split coverage, metric algebra, and
chunk checksums; the recovery validator checked exact state preservation and
independently aggregated existing database tables. They did not independently
rerun model losses, regroup raw chunks, or prove every key-to-bucket assignment
and key's uniqueness. These offline teacher-forced results establish neither
closed-loop acceptance nor speed.

Fresh M3 development selection is running after renewed M2 artifact validation;
test evaluation and timing remain pending. The
[earlier partial M3 audit](../artifacts/m3_dev_partial_validation_reference.json)
is historical evidence for eight standard dev bundles under the old detector,
not validation of the corrected run. M4 has not run. Planned M3/M4 timing retains
all held-out prompts, three timing repeats, a separate full phase-profile pass,
and fresh matched baseline measurements.
Progress is recorded in [NOTES.md](../NOTES.md).

All 263 CPU tests pass. Complete M3 and M4 smoke runs also pass, including verified
baseline reuse, direct greedy comparisons, fresh isolated timing, sparse norms,
version boundaries, feature-context linkage, and parallel diagnostic parity;
see the [current integration validation](../artifacts/repetition_detector_review/integration_final/validation.json).
The [fresh prompt audit](../artifacts/prompt_followup_audit.json) found and fixed
a timing-resume integrity gap: completion now requires checksummed raw bundles
covering every expected timing and phase-profile request. The six fit-hashed M2
modules remained unchanged through fitting and recovery.
CPU smoke paths exercise collection, compact grouped-loss fitting, all required
evaluation arms, and synchronous request-local online updates. Synthetic fitting
and reference-sized random-tensor microbenchmarks are implementation checks;
their results do not establish a reference acceptance or throughput improvement.

External baselines not run: full tree-based Token Recycling, Test-Time Speculation,
and SharpDraft. The required normalized online bigram baseline is implemented and
covered by smoke tests; its reference M3/M4 evaluations remain pending. Full-scale
experiments and optional M5 work have not run.
