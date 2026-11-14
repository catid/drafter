# Research execution notes

Specification: `prompt.md` (attachment differs only by final newline).

## Startup

Repository initially contained only the specification and had no Git history. The completed environment inventory, package versions, and pinned model revisions are in `artifacts/environment.json`. No system packages, CUDA, or drivers were changed. GPU occupancy is checked before each launch; an unrelated GPU 7 process was present at startup and is no longer present in the latest inventory.

Milestones: M0, M1-pre, M1a, and M1b passed. M2 is running; M3 and M4 are not_run. `artifacts/milestones.json` records milestone evidence. Reference downloads are authorized through the explicit `--allow_download` option. Full scale and M5 are not selected. No adapter improvement result is claimed yet.

## Pinned backend and inspected conditions

DFlash is the unmodified Git submodule at `07ebd93db9f472af339b644bb70221ad8428328a`.
Its actual optional dependency is `local`, declaring torch 2.13.0 and Transformers
5.15.0 on Linux; `uv.lock` freezes the full environment. Python is 3.12.14 in `.venv`.
The installed CUDA wheel runtime is 13.0; the observed driver is 615.71.10.
Measured GPU capabilities and free memory are in the environment inventory.

Target revision: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`.
Drafter revision: `z-lab/Qwen3-8B-DFlash-b16@9b41424b7109f9c5413454f481b09a82b85333f4`.
Downloads used public access and the explicit `python -m edraft.download --scale
reference --allow_download` path. No credential is stored in this repository.

Non-thinking settings come from the target README's Best Practices, line 328:
T=0.7, top_p=0.8, top_k=20, min_p=0. Thinking settings come from the target
`generation_config.json`: T=0.6, top_p=0.95, top_k=20; min_p is absent there and
explicitly 0 in the target README Best Practices line 327. These sources are
pinned at the target revision above. The tokenizer template supports
`enable_thinking=False` and `True`. The drafter card's line 144 specifies thinking
disabled; its checkpoint evaluation uses non-thinking. Thus standard_nonthinking
is supported and long_thinking is supported by the target but out of distribution
for this drafter. Conditions will remain separate. Pure T=1 unwarped is diagnostic.

Current upstream stochastic verification uses the ratio rule. Its vectorized draft
sampling and acceptance uniforms are reproduced for the local adapter-disabled and
zero-weight paths, making seeded positive-temperature parity a hard M0 gate.
Nonzero causal adapters sample left to right using realized tokens. Accepted length
means accepted unknown draft tokens, excluding anchor and replacement/bonus;
upstream `acceptance_lengths` includes the emitted replacement/bonus token.

## Synthetic smoke gate

The first CPU smoke run passed: 1,152 generated tokens across 315 rounds, including
187 replacements and 128 bonus tokens, zero-weight token parity, target-greedy
parity, 84 synchronous online updates, and request-local bigram updates. Measured
synthetic throughput was approximately 1,100 tokens/s (1.05 s); this is not a
real-model throughput estimate. 47 tests passed before reference execution.

## M0 reference evidence

`artifacts/m0.json` records 20 distinct prompts at greedy, model-card warped T=0.7,
and pure-temperature T=1.0. Upstream versus local: 0 mismatches in 7,128 compared
output-token positions across 60 runs. Target-only greedy mismatch is identical
for upstream and local: 8/20 sequences and 607/2,388 token positions (25.42%,
including divergence propagation). These differences are not hidden or counted
as adapter improvements.

`artifacts/m0_numerics.json` investigates all eight first divergences on identical
prefixes. The recorded divergent tokens reproduce, cache lengths agree exactly,
and changing future proposal tokens leaves earlier logits bit-identical. Every
case has a BF16 maximum-logit tie in at least one execution shape; the other path's
top-two margin is 0.125 or 0.25. The evidence supports numerical/tie sensitivity,
with no rollback or mask defect found. Three independent real rollback probes
at accepted lengths 0/1/3 preserve argmax and exact cache lengths.

The held-out real post-normalization hook check covered 60 vectors and 9,116,160
vocabulary entries. BF16 head replay is exact. FP32 reconstruction has maximum
absolute log-probability error 0.11353 and maximum probability error 0.02283;
configured tolerances are logp atol=0.15/rtol=0.01 and probability
atol=0.01/rtol=0.05. All 60 argmax IDs agree. M1 will separately verify serialized
held-out shards. Exact source hooks, shapes, anchor mapping and cache operations
are documented in `artifacts/dflash_inspection.md`.

M0 started before the first project commit, so its original runtime record honestly
reports `uncommitted`. The initial implementation was then committed as `2751da2`.
Subsequent runs record the project commit. GitHub push is pending because the
configured SSH agent is unresponsive (`ssh-add -l` times out); local commits are
preserved and research execution is independent of that connection.

Pre-run estimate: {"scale": "reference", "expected_tokens": 17920, "smoke_tokens_per_second": 1099.9935217379511, "cpu_smoke_extrapolation_seconds": 16.29100503399968, "free_disk_bytes": 33796323827712, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 1800, "smoke_tokens_per_second": 1099.9935217379511, "cpu_smoke_extrapolation_seconds": 1.6363732735044323, "free_disk_bytes": 33796697321472, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 5000, "smoke_tokens_per_second": 1099.9935217379511, "cpu_smoke_extrapolation_seconds": 4.54548131529009, "free_disk_bytes": 33796678717440, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 1800, "smoke_tokens_per_second": 1099.9935217379511, "cpu_smoke_extrapolation_seconds": 1.6363732735044323, "free_disk_bytes": 33796667965440, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 5000, "smoke_tokens_per_second": 1099.9935217379511, "cpu_smoke_extrapolation_seconds": 4.54548131529009, "free_disk_bytes": 33796657319936, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 8192, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 5.29770874314838, "free_disk_bytes": 33796624007168, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

M1 runtime planning from real M0 pilot: {"basis": "measured M0 local model generation, no trace diagnostics cost", "tokens": 7128, "seconds": 76.6340094616171, "tokens_per_second": 93.01353341782449, "calibration_8192_tokens_seconds": 88.07320503781808, "collection_primary_500000_tokens_4GPUs_seconds": 1343.8904577303783, "collection_other_budgets": "standard dev/test + pure diagnostics and 44 long outputs additional, output-length dependent", "planning_range_minutes_4GPUs": [45, 90], "uncertainty": "Range includes unmeasured live diagnostic and shard I/O overhead and variable long-output lengths; refine after first completed shards."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 125000, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 80.83662022626312, "free_disk_bytes": 33796304400384, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 125000, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 80.83662022626312, "free_disk_bytes": 33796304388096, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 125000, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 80.83662022626312, "free_disk_bytes": 33796304388096, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 125000, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 80.83662022626312, "free_disk_bytes": 33796304384000, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 0, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 0.0, "free_disk_bytes": 33790582509568, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 0, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 0.0, "free_disk_bytes": 33789716475904, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 0, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 0.0, "free_disk_bytes": 33788186333184, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 17920, "smoke_tokens_per_second": 1546.3288748357218, "cpu_smoke_extrapolation_seconds": 11.588737875637081, "free_disk_bytes": 33783601463296, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

## Resumed execution, 2026-11-13 23:12 UTC

M0 was repeated after the logging changes: all 60 comparisons and 7,128 tokens
still match upstream, and all 2,369 verification blocks retain their decisions.
`artifacts/m0_current_validation.json` binds this check to source hashes and row
digests. The eight investigated target-only greedy divergences are unchanged.

M1-pre froze the projection and token features from 64 calibration prompts and
22,377 hidden vectors. At 23:10:50 UTC the four M1a workers had completed 250,790
greedy and 250,720 warped-positive primary train tokens (501,510 total), all
standard dev/test generations, and 128/192 pure-temperature diagnostics. Long
thinking collection is additional to these quotas; realized lengths and repetition
checks, not the 24,576-token cap, determine coverage of long-output bins.

GitHub authentication remains unavailable: retrying read-only `git ls-remote`
with the unresponsive agent disabled returned `Permission denied (publickey)`.
Local commits and experiment progress are preserved; no push has succeeded.

Pre-run estimate: {"scale": "reference", "expected_tokens": 1800, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 1.2345860905043082, "free_disk_bytes": 33771529314304, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Updated scoring microbenchmark (2026-11-13T23:13:31.596996+00:00): {"artifact": "artifacts/adapter_benchmark_reference_023b4722f495.json", "label": "synthetic", "source_sha256": "aff049e0f2f71741652b285b63325f37f5720a2ffba01fcbab1899a4ddbd5e49", "median_wall_ms": {"batched_full": 8.893974008969963, "sequential_full": 9.430954465642571, "dense_candidate": 8.111080038361251, "dense_candidate_including_C0_construction": 8.436599979177117}, "interpretation": "reference-size random tensors only; not model acceptance or end-to-end speed evidence"}

M1 remaining-runtime estimate from completed reference shards: diagnostics approximately 118 tokens/s and 2.44 s/generation. Long collection has 11 generations per rank: about 13 minutes at mean realized length 8K, or 38 minutes if all reach 24,576 at that rate, before context slowdown. These are conditional estimates, not upper bounds. The CPU diagnostic aggregation pilot took 1.57 s for 249 blocks; optimization with complete-report parity is in progress before M1b.

M1 monitoring handoff, 2026-11-13 23:15 UTC: all four workers have completed the primary train quotas (250,790 greedy + 250,720 T=0.7 warped tokens), all 256 standard primary dev/test generations, and all 192 T=1 pure-temperature diagnostic generations. The 44 long-thinking generations (12 dev + 32 test) remain separate and run at T=0.6, top_k=20, top_p=0.95, min_p=0 with natural EOS or a 24,576-token cap. No quota or generation setting changed.

M1b CPU optimization is complete: the same 249-block/3,735-slot reference trace pilot reduced accumulation median from 0.308065 s to 0.053294 s (5.78x), and read-plus-accumulate from 1.579253 s to 1.176631 s (1.34x). Every report field was checked against the original scalar accumulator: non-feature statistics and bootstrap results match exactly, while float64 feature reductions agree within 2e-13 absolute/relative tolerance. Fifteen focused tests passed. Artifact: `artifacts/diagnostic_aggregation_benchmark.json`. This is a CPU diagnostic operator result, not model generation throughput. Definitive M1b will wait for M1a acceptance and final integration validation.

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976093184, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976084992, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976084992, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976084992, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976080896, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976080896, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976080896, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Pre-run estimate: {"scale": "reference", "expected_tokens": 62500, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 42.86757258695515, "free_disk_bytes": 33768976080896, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

Long collection expanded at 2026-11-13 23:17:39 UTC from four to eight workers, preserving first-four rank/GPU mapping and appending idle GPUs: physical order `2,3,4,6,0,1,5,7`. The restart uses `--partitions long`; all primary and diagnostic shards remain intact and will be included by the definitive M1a audit. Three complete first-wave long generations stayed on their original logical ranks and resume without duplicate work. Only this project's launcher and four worker PIDs were stopped after confirming no other GPU processes were present. No sampling, seed, prompt, calibration, token-cap, or quota setting changed.

The restart discarded approximately 552.7 GPU-seconds (9.2 GPU-minutes) of unpublished partial work, estimated from time since each worker's last committed shard (47.2, 73.0, 138.1, and 294.4 seconds). No committed trace was removed. The first two long completions had 10,421 and 8,865 tokens, natural EOS, no repetition domination, and 35.85/37.20 tokens per second. At their mean 264.5 seconds/generation, eight-way collection suggests about 25–30 additional minutes including rank imbalance, with substantial uncertainty from future realized lengths. This is an ETA, not a stopping budget. Original process/rank snapshots and occupancy verification are archived in `artifacts/collection_expansion_20261113T231738Z/`.

Final CPU integration validation passed before M1b: 180 pytest tests, complete M3 smoke (32 evaluated / 96 timed requests), complete M4 smoke (40 evaluated / 120 timed requests), four feature-context groups, and identical plain-baseline outputs and accepted lengths across eight shared M3/M4 jobs. Reference fitting remains gated until definitive M1a audit and M1b diagnosis finish.

M1b bounded multiprocessing is implemented with `--workers` (default 1; planned reference diagnosis 8). Each CPU-only worker reads and checksum-validates one audited shard and returns pickle-safe statistics; the parent keeps at most one submitted shard per worker, merges strictly in source order, and computes bootstrap/quantiles once. If prompt/seed metric keys overlap across shards, the parent replays that shard serially before merging any contribution to preserve floating-add order. Audit source paths/checksums remain authoritative. Twenty-two focused tests pass; source is frozen pending renewed full integration after evaluator edits.

Parallel diagnostic benchmark on reference traces (`artifacts/diagnostic_parallel_benchmark.json`): the original four-shard/249-block pilot takes 1.232 s with one worker versus 2.610/2.354 s at four/eight workers including process startup. A larger 64-distinct-shard pilot (3803 blocks) takes 18.492/8.665/6.504 s at one/four/eight workers (2.84x eight-worker speedup). All complete reports match serial: exact non-feature statistics/bootstrap results and 2e-13 absolute/relative tolerance only for float64 feature reductions. This benchmark includes source reading/checksum/schema validation, process startup/transport, accumulation and ordered merge, but excludes final report/bootstrap time. No model-throughput claim is made.

Renewed final integration gate passed at 2026-11-13 23:41 UTC after parallel M1b and evaluator changes: all 222 tests plus fresh complete M3/M4 smoke pipelines, with source hashes unchanged throughout. Evidence: `artifacts/reuse_smoke_pipeline/validation.json`. Diagnostic execution is now ready with eight CPU workers once definitive M1a audit passes.

Definitive M1a audit PASSED on 2026-11-13 at 23:56:36 UTC after all eight long-only workers and launcher exited zero. All 12 requirements passed; accepted source set contains 2,184 checksum-verified shards (256,836 blocks, 58,794,800,457 bytes) and 16 passed serialized held-out head reconstruction checks. Primary standard train output totals are 250,790 greedy + 250,720 recommended warped tokens = 501,510. Standard dev/test have 64 prompts each at T=0,0.7,1.0; pure-temperature train diagnostics have 64 separate prompts. All prompt splits are disjoint and HumanEval remains entirely held out. Evidence: `artifacts/m1_audit.json`.

Long-thinking collection completed 12 dev generations (151,225 tokens; median 11,519.5; range 8,230–24,576) and 32 test generations (361,144 tokens; median 11,183.5; range 2,943–24,576). No generation was repetition dominated. One dev and one test generation hit the 24,576 cap; all other 42 reached natural EOS. All 12 dev and 25 test generations exceeded 8,192 tokens; only 2 dev and 3 test generations exceeded 16,384, so the 16K+ bin remains sparse and does not support strong conclusions. No padding, minimum-length forcing, quota changes, or skipped prompts were used.

M1b CPU diagnosis now starts with eight workers using only the authoritative audited shard set and frozen calibration checksums. The 222-test renewed integration validation passed before this launch.

Pre-run estimate: {"scale": "reference", "expected_tokens": 0, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 0.0, "free_disk_bytes": 33737362960384, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

M1b PASSED its report/provenance gate at 2026-11-14T00:02:39.208486+00:00: `artifacts/diagnostic_reference.json` is measured at reference scale and contains 106 condition/split/bin/policy groups plus 11 generation-length groups. All 2,184 ordered source paths and SHA256 values exactly match the definitive M1a audit. The audit digest, accepted source-set digest, frozen calibration files, frozen prompt manifest, and every recorded implementation source hash were verified. Eight-worker execution completed in 312.96 seconds including the initial source checksum pass and final report/bootstrap. Evidence and report digest: `artifacts/m1b_validation.json`.

Plain-DFlash M1b test-set findings (on-policy, all-offset primary groups; accepted length is the block mean, other metrics below are prompt-aggregated means): standard greedy accepts 5.2178 tokens per block, standard recommended T=0.7 accepts 4.8727, pure T=1 diagnostic accepts 4.1921, and OOD long-thinking T=0.6 accepts 2.4995. Recommended standard stochastic overlap is 0.84127 versus long-thinking 0.71495; target top-1 recall within draft top-256 is 0.998234 and 0.998459 respectively. Unwarped cross-entropy is 0.57628 versus 1.12683, and projected-feature variance ratios relative to frozen standard calibration are 1.34651 versus 0.84731. These are separate-condition baseline diagnostics, not evidence of adapter improvement. Full prompt-bootstrap intervals, ranks, recalls, rejection recoverability, feature statistics, and on/off-policy bins are in the report.

M1b confirms usable >8K coverage (12 dev/25 test generations), but only 2 dev/3 test generations reach 16K+, below the diagnostic minimum of eight generations per split/bin. Both long summaries correctly flag data sparsity. No repetition-dominated output was excluded; 42 generations ended naturally and 2 reached the configured cap. M1a and M1b now permit the authorized primary M2 fit; this agent started no training.

## M2 launch planning, 2026-11-14T00:03:30.209814+00:00

Use the predeclared defaults: batch_blocks=8, one dense epoch then one dense/bias/sparse epoch, lr=1e-3, weight_decay=1e-4, sparse_lr=.01, lambda_l1=1e-5; no additional tuning sweep. There are 21,078 training and 38,291 evaluation batch calls (474,907 block visits). The measured serial 64-shard diagnostic read-and-aggregation pilot suggests roughly 39 minutes for that many block visits, but it includes diagnostic work absent in training and omits GPU loss/backward work. Provisional planning range: 30–120 minutes, to be refined from actual checkpoint progress; this is not a throughput result. Physical GPU 0 is idle. M1a/M1b gates and provenance passed; the latest independent CPU integration passed 222 tests plus full M3/M4 smoke runs. Runtime preflight will append current smoke throughput and free disk space.

Pre-run estimate: {"scale": "reference", "expected_tokens": 500000, "smoke_tokens_per_second": 1457.978519152706, "cpu_smoke_extrapolation_seconds": 342.9405806956412, "free_disk_bytes": 33737360551936, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}

## M2 component profiling decision, 2026-11-14T00:16:00.057060+00:00

A separate benchmark on idle GPU 1 of three audited 8-block batches found packed transfers and vectorized overlap reduced in-memory evaluation from 24.7/28.3/30.9 ms to 7.68/8.04/11.05 ms. Target tensors, weights and grouped loss matched bitwise; final metric differences were at most 4.04e-9. Including sampled shard reads reduced the estimated speedup to 1.15–1.47x. Gross arithmetic savings across the planned batch counts were 12.25–14.56 minutes, with 10.09 minutes of current work already spent and implementation, validation, and restart still needed. Decision: preserve the running fit; no production source, settings, parameters or optimizer trajectory changed. Evidence and the archived one-off source snapshot are in artifacts/m2_component_benchmark/. These are component measurements and estimates, not full-fit throughput or adapter-quality results.

M2 intermediate dense result, 2026-11-14 00:27 UTC: after the single predeclared dense epoch, dev weighted grouped KL is 2.018092362937402 versus zero-weight baseline 1.168452997781549, a relative improvement of -0.7271489454595071 (72.7% worse). Values remain finite. The under-1% rule stops extra dense epochs; the predeclared bias/sparse phase remains required and will run. No hyperparameters, optimizer state, source, or checkpoint were changed. Complete dense train/test evaluation and final sparse results remain pending.

M2 dense phase finalized around 2026-11-14 00:40 UTC. Complete-split grouped KL worsened on train (1.16525348 -> 1.92823779), dev (1.16845300 -> 2.01809236), and test (1.21701844 -> 2.85879702). This is a negative fitting result across all splits. The saved dense checkpoint remains the prescribed final epoch, with W norm 1.96483, A norm 2.33229, bias exactly zero, and no sparse table. Evidence: `artifacts/m2_dense_interim.json`, including checkpoint SHA256 and separate 120-slot dev/test live-full diagnostics. The required bias/sparse phase proceeds unchanged. A read-only objective-consistency review is checking for concrete defects; no hyperparameter tuning or extra optimizer updates are authorized by that review.

M2 dense regression review, 2026-11-14: two independent read-only source reviews found no concrete correction-sign, matrix-orientation, loss-scaling, optimizer-selection, or train/evaluate objective mismatch. Existing passing tests cover independent grouped-loss parameter gradients, analytic W/A descent, and exact resume. Training KL also worsened, so held-out overfitting alone does not explain the result. Fixed-learning-rate overshoot, within-shard batch correlation, and final-iterate drift remain hypotheses; the available epoch-end metrics do not establish a cause. No additional optimizer updates, tests, CUDA work, or production edits were performed by this review.
