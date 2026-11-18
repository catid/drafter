# Research status

As of 2026-11-16, the historical M0–M4 run passed its recorded validators. Later review found a bigram sampling defect and extra M4 snapshot copies. The [reference bigram correction](#reference-bigram-correction) identifies the affected requests. These experiments ran at `--scale reference` on the pinned `Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash-b16` pair, backed by checksummed raw traces, closed-loop evaluation trajectories, matched-bundle timing and CUDA-event phase profiles, and independent validators (`45,158,121` M3 checks and `18,983,686` M4 checks with zero failures).

**Summary of empirical findings (`M0–M4`, `--scale reference`):**
- **Offline fitting (`M2`) produced a negative quality result under the predeclared constant-LR AdamW schedule:** dense (`+72.7%` dev grouped KL) and dense+bias+sparse (`+76.3%` dev grouped KL) worsened grouped KL on train, dev, and test; extra epochs stopped under the `<1%` dev-improvement rule. A read-only post-hoc 1D ray check (`alpha * dense.pt`) showed that the fitted direction itself reduces sampled M1 dev grouped KL by `-11.2%` (`1.1483 -> 1.0202`) at `alpha = 0.20`, showing lower sampled dev loss at one-fifth of the saved scale. The fit used shard-sequential batches and constant-LR `AdamW(lr=1e-3)`. Gated M2 checkpoints were **not** modified or rescaled in M3 or M4.
- **Closed-loop static adapters (`M3`) did not improve acceptance or net speed:** at dev-selected `beta = 0.5`, static `dense` and `dense_sparse` reduced mean accepted length relative to `plain_dflash` across all four conditions (`4.125` and `4.041` vs `5.136` on standard `T=0.7`; `2.102` and `1.972` vs `2.476` on OOD `long_thinking` `T=0.6`) and reduced single-GPU throughput (`0.57x–0.81x` of `plain_dflash`). The normalized committed-transition `online_bigram` baseline (`alpha=1.0, lambda_tr=0.2`) had a higher per-arm mean accepted length in every condition using each arm’s eligible prompts (`5.195` vs `5.136` at `T=0.7`; `2.651` vs `2.476`, `+7.07%` on primary `long_thinking`), reducing `target_verification` time on `long_thinking` (`79.8s` vs `87.2s`), but eager full-vocabulary host/sampling overhead left net `tokens/sec` below `plain_dflash` (`0.86x–0.96x`).
- **Closed-loop synchronous online adaptation (`M4`) had a higher mean accepted length than `plain_dflash` when each arm was summarized on its own eligible prompts (paired 95% interval spans zero) on out-of-distribution `long_thinking`, with higher mean acceptance for the full online update than for the bias-and-sparse update, while incurring net wall-clock slowdown in uncompiled eager M0–M4:** on primary `long_thinking` (`T=0.6`, where M1b documented substantial draft hidden-feature `h[i]` shift: coordinate-mean norm `11.302 -> 8.314`, variance ratio `1.347 -> 0.851`), full-adapter `offline_online_stride1` (`beta=1.0, lr=1e-4, stride=1, exponential`) lifted mean accepted length to **`2.574 [2.424, 2.730]`** (`+30.5%` over `offline_static` `1.972 [1.876, 2.068]` and **`+3.94%` above `plain_dflash` `2.476 [2.289, 2.654]`**, cutting `target_verification` median time from `87.89s` to `80.22s`), whereas bias+sparse-only `online_dbS` (`dW=0, dA=0`) collapsed to **`1.435 [1.354, 1.524]`**. On shorter in-distribution `standard_nonthinking` (`~300` tokens, `~72` updates), `offline_online_stride1` (`3.172` at `T=0.7`) did not overcome the overshot M2 initialization within a single request. Across all 20 M4 condition/arm groups, eager full-vocabulary scoring, sparse gathering, and per-round autograd `L_full` backward (`118.2s` median `online_update` on `long_thinking`) exceeded verification savings (`0.20x–0.42x` of `plain_dflash` `tokens/sec`; `throughput_gain_supported_by_95_ci = false` everywhere).

The repetition detector checks exact cycles of any eligible length while preserving the four-cycle minimum and 50% dominance threshold. Corrected M1 reports exclude one pathological long-test output (`math500-long-0346`), and M3/M4 recompute exact-cycle repetition flags from stored output tokens on every long generation. See the [confirmed evidence](../artifacts/repetition_detector_review/evidence.json) and [correction validation](../artifacts/repetition_detector_review/m1_correction_validation.json).

| Milestone | Reference status | Evidence and headline outcome |
| --- | --- | --- |
| M0 | passed | Upstream/local exact match (`0/7,128` token diffs) and BF16 tie root-cause verification (`8/20` vs `target_only`) |
| M1-pre | passed | 64 train calibration prompts; 22,377 hidden vectors; frozen `[256, 4096]` `P_h` and `[151936, 64]` token PCA |
| M1a | passed | 2,184 checksummed shards; 501,510 primary train tokens (`250,790` greedy + `250,720` `T=0.7`); all 12 audit gates pass |
| M1b | passed | 128 corrected diagnostic groups across all 2,184 shards; `h[i]` covariate shift & top-256 coverage headroom quantified |
| M2 | passed | Complete dense & dense+bias+sparse fit + 32 GB exact sparse accounting; dev grouped KL increased by 72.7% for dense and 76.3% for dense+bias+sparse versus the zero adapter |
| M3 | Historical validator pass ([bigram qualification](#reference-bigram-correction)) | `140` dev, `1,920` eval + `64` greedy target-only, `5,760` 1-GPU timed + `1,920` profile runs (`45,158,121` validator checks); static M2 adapters hurt acceptance & speed |
| M4 | Historical validator pass (bigram sampling and sparse snapshot qualifications) | `180` dev, `2,400` eval, `60` 1-GPU calibration + `7,200` 8-GPU NUMA/core-pinned timed + `2,400` profile runs (`18,983,686` validator checks); on 31 jointly retained `long_thinking` prompts, `offline_online_stride1` minus `plain_dflash` mean accepted length was +0.089 tokens per block (paired 95% interval −0.150 to +0.294), no net `tokens/sec` gain in eager M0–M4 |

The authoritative status snapshot is
[milestones.json](../artifacts/milestones.json). The
[implementation audit](../artifacts/implementation_audit.json) distinguishes code
and test coverage from incomplete reference experiments. Exact commands and stage
gates are in [README](../README.md#reference-execution-m0m4).

The reference pair is Qwen/Qwen3-8B and z-lab/Qwen3-8B-DFlash-b16. Standard
non-thinking sampling defaults come from the target model card; thinking defaults
come from the pinned `generation_config.json`. Stochastic comparisons used the recorded ratio verifier. Its Section 8 precision qualification appears below. Upstream and local matched on all 20 prompts at each of
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
does not mean every metric deteriorated. Independent numerical audit verified zero
code, sign, standardization, or grouped-KL denominator defects (`b['h']` vs
`standardize(P_h @ h_raw_bf16)` max difference `0.0`; `union_rest_grouped` vs
`L_full` gradient cosine `1.000000` on `W, A, b`). A read-only post-hoc 1D ray
diagnostic (`alpha * checkpoint`, zero retraining, evaluated on sampled M1 dev
shards) measured sensitivity to checkpoint scale. Scaling `alpha = -0.10`
worsens sampled dev grouped KL from `1.1483` (`alpha = 0`) to `1.2987` (confirming
the `-E_corr` sign is right), **`alpha = +0.20 * dense.pt` reduces sampled dev
grouped KL to `1.0202` (`-0.1281` nats, `-11.2%` below zero-adapter DFlash;
`0.20 * dense_sparse.pt` gives `1.0578`, `-7.9%`)**, whereas the saved `alpha = 1.00`
checkpoint gave `1.7091` on the same sample. The fit used shard-sequential
mini-batches (about 50 blocks per shard and six consecutive steps per prompt) and
constant-LR `AdamW(lr=1e-3)` for 10,539 steps. The later refit shuffled blocks within windows of 32 shards
and raised AdamW weight decay on `W, A` from 0.0001 to 0.01 together.
This comparison measures their combined effect. In strict adherence to the protocol, gated M2 checkpoints were
**never** modified or rescaled in M3 or M4 (where `beta = 0.5` was the smallest
predeclared static candidate).

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

---

## Closed-loop offline evaluation and single-GPU timing (`M3`)

M3 evaluated four arms (`plain_dflash`, `online_bigram`, `dense`, `dense_sparse`) at `--scale reference` under frozen source commit `09a5469c72c442408d1a85af7ccd58c231db9fb9`. Development tuning (`artifacts/m3/settings.json`, `frozen_settings_hash = 35ade67825d3eaa6788734809456de212459415209c40abc713045fdfa039163`) ran on 8 `standard_nonthinking` (`T=0.7`) and 12 `long_thinking` (`T=0.6`) dev prompts (`140` dev requests across `20` matched 7-candidate bundles on GPU 0), selecting `online_bigram` `{"alpha": 1.0, "k_ret": null, "lambda_tr": 0.2}` (`0.9736`), `dense` `{"beta": 0.5}` (`0.7911`), and `dense_sparse` `{"beta": 0.5}` (`0.7140`). Held-out test evaluation (`artifacts/m3/summary.json`, `1,920 / 1,920` requests + `64` `greedy_target_only` diagnostics) covered 64 `standard_nonthinking` test prompts (`1` greedy seed + `3` seeds at recommended `T=0.7` + `3` seeds at diagnostic `T=1.0`) and 32 `long_thinking` test prompts (`1` seed at `T=0.6`). Isolated single-GPU timing (`artifacts/m3/timing.json`) ran all `480` test prompt/seed slots x `4` arms x `3` uninstrumented synchronized repeats (`5,760` timed requests) plus `1,920` separate CUDA-event phase-profile requests on idle `GPU 0`. Independent reference validation ([m3_validation_reference.json](../artifacts/m3_validation_reference.json)) passed **`45,158,121` checks** (`3,181,100` blocks) with zero failures.

### M3 Development Selection (`tune`)

| Arm | Candidate setting | Selected | Selection score | Std norm tput | Long norm tput | Dev reqs | Rep-excl pairs |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: |
| `plain_dflash` | `{}` | yes | 1.0000 | 1.0000 | 1.0000 | 20 | 0 |
| `online_bigram` | `{"alpha": 0.5, "k_ret": null, "lambda_tr": 0.05}` | no | 0.8825 | 0.8866 | 0.8784 | 20 | 0 |
| `online_bigram` | `{"alpha": 1.0, "k_ret": null, "lambda_tr": 0.2}` | yes | 0.9736 | 0.9608 | 0.9863 | 20 | 1 |
| `dense` | `{"beta": 0.5}` | yes | 0.7911 | 0.8252 | 0.7571 | 20 | 0 |
| `dense` | `{"beta": 1.0}` | no | 0.6558 | 0.7029 | 0.6088 | 20 | 0 |
| `dense_sparse` | `{"beta": 0.5}` | yes | 0.7140 | 0.7222 | 0.7058 | 20 | 1 |
| `dense_sparse` | `{"beta": 1.0}` | no | 0.5331 | 0.5863 | 0.4799 | 20 | 0 |

### M3 Held-Out Test Acceptance & Realized Lengths (Primary)

| Condition | Arm | Primary / Total (Rep-excl) | Mean accepted len [95% CI] | Seed std | Req median accepted len [95% CI] | Target fwds/tok [95% CI] | Mean out len (>8K / >16K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `dense` | 32/32 (0) | 2.102 [1.972, 2.234] | 0.0000 | 0.969 [0.906, 1.000] | 0.3277 [0.3131, 0.3434] | 11726.0 (27 / 2) |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `dense_sparse` | 32/32 (0) | 1.972 [1.876, 2.068] | 0.0000 | 1.000 [1.000, 1.000] | 0.3398 [0.3287, 0.3514] | 11188.3 (28 / 4) |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 31/32 (1: math500-long-0116) | 2.651 [2.483, 2.844] | 0.0000 | 1.323 [1.161, 1.516] | 0.2792 [0.2664, 0.2919] | 11269.7 (26 / 3) |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 32/32 (0) | 2.476 [2.289, 2.654] | 0.0000 | 1.281 [1.125, 1.438] | 0.2944 [0.2792, 0.3115] | 11512.1 (25 / 6) |
| Standard non-thinking greedy (T=0) | `dense` | 64/64 (0) | 2.702 [2.540, 2.862] | 0.0000 | 1.422 [1.305, 1.531] | 0.2831 [0.2708, 0.2966] | 298.1 (0 / 0) |
| Standard non-thinking greedy (T=0) | `dense_sparse` | 64/64 (0) | 2.283 [2.146, 2.418] | 0.0000 | 1.188 [1.086, 1.297] | 0.3181 [0.3045, 0.3330] | 298.1 (0 / 0) |
| Standard non-thinking greedy (T=0) | `online_bigram` | 64/64 (0) | 5.311 [5.081, 5.545] | 0.0000 | 3.594 [3.289, 3.898] | 0.1665 [0.1605, 0.1729] | 298.1 (0 / 0) |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 64/64 (0) | 5.285 [5.057, 5.519] | 0.0000 | 3.578 [3.289, 3.867] | 0.1672 [0.1612, 0.1737] | 298.1 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `dense` | 192/192 (0) | 4.125 [3.935, 4.305] | 0.3426 | 2.617 [2.469, 2.760] | 0.2053 [0.1978, 0.2134] | 300.6 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `dense_sparse` | 192/192 (0) | 4.041 [3.847, 4.228] | 0.3157 | 2.438 [2.271, 2.609] | 0.2087 [0.2010, 0.2172] | 304.2 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 192/192 (0) | 5.195 [4.976, 5.406] | 0.4762 | 3.544 [3.346, 3.737] | 0.1702 [0.1643, 0.1766] | 302.3 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 192/192 (0) | 5.136 [4.922, 5.346] | 0.3934 | 3.469 [3.263, 3.685] | 0.1714 [0.1655, 0.1778] | 296.5 (0 / 0) |
| Standard pure-T=1 diagnostic | `dense` | 192/192 (0) | 3.664 [3.495, 3.827] | 0.3451 | 2.255 [2.120, 2.385] | 0.2251 [0.2168, 0.2341] | 308.5 (0 / 0) |
| Standard pure-T=1 diagnostic | `dense_sparse` | 192/192 (0) | 3.666 [3.500, 3.828] | 0.3909 | 2.208 [2.060, 2.352] | 0.2252 [0.2172, 0.2334] | 304.1 (0 / 0) |
| Standard pure-T=1 diagnostic | `online_bigram` | 192/192 (0) | 4.388 [4.190, 4.590] | 0.4452 | 2.833 [2.667, 3.005] | 0.1959 [0.1887, 0.2037] | 296.5 (0 / 0) |
| Standard pure-T=1 diagnostic | `plain_dflash` | 192/192 (0) | 4.297 [4.124, 4.477] | 0.4444 | 2.768 [2.630, 2.911] | 0.1987 [0.1917, 0.2064] | 299.5 (0 / 0) |

### M3 Secondary Sensitivity: All `long_thinking` Test Generations (Including Repetition-Dominated Loops)

| Arm | All requests | All-rows mean accepted len [95% CI] | Target fwds/tok [95% CI] | Mean out len (>8K / >16K) |
| --- | ---: | ---: | ---: | ---: |
| `dense` | 32 | 2.102 [1.972, 2.234] | 0.3277 [0.3131, 0.3434] | 11726.0 (27 / 2) |
| `dense_sparse` | 32 | 1.972 [1.876, 2.068] | 0.3398 [0.3287, 0.3514] | 11188.3 (28 / 4) |
| `online_bigram` | 32 | 2.732 [2.518, 2.991] | 0.2755 [0.2604, 0.2890] | 11315.6 (27 / 3) |
| `plain_dflash` | 32 | 2.476 [2.289, 2.654] | 0.2944 [0.2792, 0.3115] | 11512.1 (25 / 6) |

### M3 Acceptance Probability by Logical Block Position (`0..K-1`)

| Condition | Arm | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 | pos 8 | pos 9 | pos 10 | pos 11 | pos 12 | pos 13 | pos 14 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `dense` | 0.636 | 0.404 | 0.268 | 0.190 | 0.142 | 0.109 | 0.085 | 0.067 | 0.053 | 0.042 | 0.034 | 0.026 | 0.020 | 0.015 | 0.011 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `dense_sparse` | 0.627 | 0.388 | 0.253 | 0.174 | 0.127 | 0.097 | 0.075 | 0.059 | 0.046 | 0.036 | 0.029 | 0.022 | 0.017 | 0.013 | 0.009 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 0.714 | 0.485 | 0.336 | 0.247 | 0.189 | 0.150 | 0.121 | 0.098 | 0.080 | 0.064 | 0.052 | 0.041 | 0.032 | 0.025 | 0.019 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 0.698 | 0.464 | 0.316 | 0.230 | 0.174 | 0.136 | 0.108 | 0.087 | 0.070 | 0.055 | 0.044 | 0.034 | 0.026 | 0.020 | 0.014 |
| Standard non-thinking greedy (T=0) | `dense` | 0.708 | 0.489 | 0.339 | 0.240 | 0.186 | 0.148 | 0.124 | 0.103 | 0.087 | 0.071 | 0.061 | 0.051 | 0.040 | 0.032 | 0.025 |
| Standard non-thinking greedy (T=0) | `dense_sparse` | 0.655 | 0.415 | 0.289 | 0.207 | 0.159 | 0.124 | 0.095 | 0.077 | 0.064 | 0.051 | 0.043 | 0.033 | 0.027 | 0.024 | 0.019 |
| Standard non-thinking greedy (T=0) | `online_bigram` | 0.890 | 0.726 | 0.593 | 0.500 | 0.419 | 0.361 | 0.315 | 0.282 | 0.248 | 0.219 | 0.195 | 0.173 | 0.147 | 0.127 | 0.114 |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 0.892 | 0.727 | 0.589 | 0.497 | 0.416 | 0.357 | 0.315 | 0.277 | 0.247 | 0.219 | 0.194 | 0.172 | 0.146 | 0.125 | 0.112 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `dense` | 0.827 | 0.643 | 0.502 | 0.391 | 0.318 | 0.258 | 0.215 | 0.187 | 0.162 | 0.140 | 0.125 | 0.108 | 0.096 | 0.084 | 0.069 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `dense_sparse` | 0.793 | 0.608 | 0.477 | 0.382 | 0.311 | 0.257 | 0.219 | 0.190 | 0.165 | 0.145 | 0.128 | 0.111 | 0.097 | 0.085 | 0.072 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 0.889 | 0.732 | 0.590 | 0.489 | 0.408 | 0.344 | 0.299 | 0.263 | 0.231 | 0.204 | 0.186 | 0.167 | 0.150 | 0.131 | 0.112 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 0.882 | 0.731 | 0.592 | 0.490 | 0.405 | 0.344 | 0.295 | 0.257 | 0.228 | 0.201 | 0.180 | 0.159 | 0.141 | 0.124 | 0.106 |
| Standard pure-T=1 diagnostic | `dense` | 0.810 | 0.611 | 0.456 | 0.343 | 0.268 | 0.217 | 0.182 | 0.155 | 0.132 | 0.113 | 0.100 | 0.087 | 0.075 | 0.063 | 0.052 |
| Standard pure-T=1 diagnostic | `dense_sparse` | 0.797 | 0.599 | 0.447 | 0.343 | 0.272 | 0.221 | 0.184 | 0.161 | 0.137 | 0.118 | 0.103 | 0.089 | 0.076 | 0.065 | 0.055 |
| Standard pure-T=1 diagnostic | `online_bigram` | 0.863 | 0.683 | 0.526 | 0.416 | 0.331 | 0.275 | 0.233 | 0.200 | 0.175 | 0.153 | 0.135 | 0.120 | 0.107 | 0.091 | 0.079 |
| Standard pure-T=1 diagnostic | `plain_dflash` | 0.860 | 0.675 | 0.522 | 0.408 | 0.323 | 0.264 | 0.226 | 0.197 | 0.170 | 0.150 | 0.131 | 0.116 | 0.099 | 0.084 | 0.071 |

### M3 Acceptance by Long Output-Token Offset & Input-Context-Length Bins (`16K+` Sparsity Gate Enforced)

Bins with `< 8` primary prompts (`16K+` output offset and `16384-32768` input context) are descriptive only and marked **`SPARSE (conclusion unavailable)`**; `standard_nonthinking` outputs remain within `0-1K` (`n=64`, matching the overall condition rows above; full bin breakdown in [m3/summary.json](../artifacts/m3/summary.json)).

| Arm | Axis | `0-1K` (`n`) | `1K-4K` (`n`) | `4K-8K` (`n`) | `8K-16K` (`n`) | `16K+` / `16K-32K` (`n`; gate) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `plain_dflash` | `output_offset` | 2.863 [2.629, 3.101] (`32`) | 2.750 [2.494, 3.084] (`32`) | 2.879 [2.568, 3.239] (`32`) | 1.826 [1.642, 2.008] (`25`) | 1.431 [1.031, 1.816] (`n=6 < 8`; **SPARSE / conclusion unavailable**) |
| `online_bigram` | `output_offset` | 2.711 [2.478, 2.997] (`31`) | 2.878 [2.636, 3.176] (`31`) | 2.895 [2.592, 3.237] (`30`) | 2.304 [1.924, 2.787] (`26`) | 2.495 [1.592, 3.624] (`n=3 < 8`; **SPARSE / conclusion unavailable**) |
| `dense` | `output_offset` | 2.195 [2.044, 2.349] (`32`) | 2.253 [2.083, 2.456] (`32`) | 2.363 [2.162, 2.573] (`32`) | 1.677 [1.442, 1.905] (`27`) | 0.879 [0.867, 0.891] (`n=2 < 8`; **SPARSE / conclusion unavailable**) |
| `dense_sparse` | `output_offset` | 2.310 [2.090, 2.521] (`32`) | 2.225 [2.038, 2.422] (`32`) | 2.174 [2.077, 2.275] (`32`) | 1.376 [1.207, 1.546] (`28`) | 1.430 [0.921, 2.120] (`n=4 < 8`; **SPARSE / conclusion unavailable**) |
| `plain_dflash` | `input_context_length` | 2.825 [2.596, 3.053] (`32`) | 2.762 [2.508, 3.085] (`32`) | 2.900 [2.595, 3.248] (`32`) | 1.934 [1.666, 2.235] (`27`) | 1.399 [1.034, 1.772] (`n=6 < 8`; **SPARSE / conclusion unavailable**) |
| `online_bigram` | `input_context_length` | 2.687 [2.466, 2.956] (`31`) | 2.905 [2.651, 3.212] (`31`) | 2.870 [2.566, 3.208] (`30`) | 2.382 [2.013, 2.865] (`27`) | 2.541 [1.668, 3.664] (`n=3 < 8`; **SPARSE / conclusion unavailable**) |
| `dense` | `input_context_length` | 2.185 [2.024, 2.348] (`32`) | 2.252 [2.081, 2.451] (`32`) | 2.366 [2.180, 2.570] (`32`) | 1.681 [1.462, 1.889] (`28`) | 0.916 [0.858, 0.974] (`n=2 < 8`; **SPARSE / conclusion unavailable**) |
| `dense_sparse` | `input_context_length` | 2.247 [2.041, 2.445] (`32`) | 2.255 [2.062, 2.466] (`32`) | 2.160 [2.067, 2.255] (`32`) | 1.457 [1.302, 1.624] (`28`) | 1.434 [0.928, 2.154] (`n=4 < 8`; **SPARSE / conclusion unavailable**) |

### M3 Standard Greedy (`T_sample=0, beta=1`) Output Exactness & BF16 Tie Report

Every M3 arm matches `plain_dflash` bit-for-bit across all 64 greedy prompts (`0 / 19,081` token mismatches). All four arms exhibit the identical `42 / 64` (`0.656`) sequence mismatch rate and `6,662 / 19,437` (`0.3427`) token mismatch rate against `target_only`. The discrepancies are consistent with the BF16 tie sensitivity diagnosed separately in M0.

| Arm | Reference | Compared prompts | Mismatched seqs (rate) | Mismatched tokens / compared (rate) | Length-diff seqs |
| --- | --- | ---: | ---: | ---: | ---: |
| `dense` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `dense` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |
| `dense_sparse` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `dense_sparse` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |
| `online_bigram` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `online_bigram` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |
| `plain_dflash` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `plain_dflash` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |

### M3 Isolated Single-GPU Throughput, Latency & Peak Memory

| Condition | Arm | Tokens/sec [95% CI] | Paired ratio vs plain [95% CI] | Net gain? (95% CI > 1?) | Latency p50 s [95% CI] | Latency p95 s [95% CI] | Peak alloc / res GiB |
| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `dense` | 76.97 [72.29, 81.43] | 0.7754 [0.7202, 0.8364] | no (no) | 139.151 [124.699, 174.287] | 307.342 [226.266, 634.305] | 21.49 / 36.67 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `dense_sparse` | 63.77 [60.39, 66.93] | 0.6466 [0.5970, 0.6990] | no (no) | 157.483 [131.609, 179.400] | 455.975 [233.522, 540.881] | 21.09 / 36.67 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 91.36 [86.19, 96.50] | 0.9402 [0.8597, 1.0211] | no (no) | 111.549 [97.740, 137.370] | 260.141 [156.772, 462.638] | 21.70 / 36.67 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 102.94 [94.68, 111.28] | 1.0000 [1.0000, 1.0000] | no (no) | 105.756 [87.371, 129.530] | 279.848 [194.750, 288.844] | 20.94 / 36.67 |
| Standard non-thinking greedy (T=0) | `dense` | 121.59 [116.17, 126.93] | 0.5523 [0.5391, 0.5652] | no (no) | 2.392 [2.170, 2.803] | 3.467 [3.094, 3.753] | 17.59 / 36.67 |
| Standard non-thinking greedy (T=0) | `dense_sparse` | 90.69 [86.86, 94.49] | 0.4132 [0.4022, 0.4240] | no (no) | 3.410 [2.886, 3.625] | 4.502 [4.341, 4.774] | 17.66 / 36.67 |
| Standard non-thinking greedy (T=0) | `online_bigram` | 211.11 [203.46, 218.98] | 0.9589 [0.9537, 0.9638] | no (no) | 1.453 [1.202, 1.562] | 2.084 [1.849, 2.441] | 17.59 / 36.67 |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 220.21 [212.18, 228.34] | 1.0000 [1.0000, 1.0000] | no (no) | 1.387 [1.162, 1.499] | 1.979 [1.797, 2.302] | 17.58 / 36.67 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `dense` | 142.42 [137.13, 147.39] | 0.6952 [0.6819, 0.7083] | no (no) | 2.069 [1.914, 2.293] | 3.086 [2.767, 3.148] | 17.61 / 36.67 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `dense_sparse` | 117.62 [113.03, 122.14] | 0.5735 [0.5611, 0.5865] | no (no) | 2.505 [2.223, 2.757] | 4.002 [3.530, 4.773] | 17.68 / 36.67 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 176.90 [170.62, 182.87] | 0.8635 [0.8475, 0.8796] | no (no) | 1.679 [1.530, 1.908] | 2.416 [2.222, 2.613] | 17.60 / 36.67 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 206.30 [199.09, 213.41] | 1.0000 [1.0000, 1.0000] | no (no) | 1.433 [1.304, 1.535] | 2.169 [1.906, 2.103] | 17.59 / 36.67 |
| Standard pure-T=1 diagnostic | `dense` | 142.52 [137.31, 147.54] | 0.8106 [0.7797, 0.8483] | no (no) | 2.091 [1.920, 2.294] | 3.362 [2.979, 3.433] | 17.65 / 36.67 |
| Standard pure-T=1 diagnostic | `dense_sparse` | 119.24 [115.11, 123.32] | 0.6783 [0.6543, 0.7092] | no (no) | 2.456 [2.292, 2.869] | 4.053 [3.257, 4.069] | 17.68 / 36.67 |
| Standard pure-T=1 diagnostic | `online_bigram` | 169.19 [162.90, 175.49] | 0.9607 [0.9247, 1.0045] | no (no) | 1.696 [1.596, 1.946] | 2.699 [2.362, 2.646] | 17.60 / 36.67 |
| Standard pure-T=1 diagnostic | `plain_dflash` | 181.20 [175.25, 187.40] | 1.0000 [1.0000, 1.0000] | no (no) | 1.637 [1.512, 1.753] | 2.615 [2.252, 2.542] | 17.59 / 36.67 |

### M3 Separate Phase-Profile Median Durations (ms/request; `lookahead_iterations = 0.0 ms [disabled in M0-M4]`)

| Condition | Arm | Target feat | DFlash draft | Dense score | Sparse lookup | Seq sample | Target verify | Online update* | Lookahead iters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long_thinking (T=0.6) | `dense` | 161.53 | 15195.60 | 207.82 | 0.00 | 18928.47 | 100271.84 | 26.27 | 0.00 [disabled in M0-M4] |
| long_thinking (T=0.6) | `dense_sparse` | 158.81 | 14842.45 | 214.09 | 23637.83 | 21074.22 | 95724.92 | 26.36 | 0.00 [disabled in M0-M4] |
| long_thinking (T=0.6) | `online_bigram` | 133.37 | 12306.69 | 0.00 | 0.00 | 15413.63 | 79837.30 | 44.80 | 0.00 [disabled in M0-M4] |
| long_thinking (T=0.6) | `plain_dflash` | 139.28 | 13020.35 | 0.00 | 0.00 | 1223.07 | 87183.46 | 22.92 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `dense` | 24.34 | 318.74 | 5.63 | 0.00 | 49.16 | 1864.17 | 0.57 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `dense_sparse` | 24.70 | 383.69 | 6.44 | 568.01 | 64.79 | 2239.30 | 0.67 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `online_bigram` | 23.04 | 198.00 | 0.00 | 0.00 | 30.21 | 1159.10 | 0.94 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `plain_dflash` | 23.18 | 198.48 | 0.00 | 0.00 | 2.46 | 1160.38 | 0.38 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `dense` | 23.45 | 235.14 | 3.90 | 0.00 | 305.54 | 1406.43 | 0.42 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `dense_sparse` | 23.66 | 240.34 | 4.00 | 392.18 | 348.83 | 1440.85 | 0.44 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `online_bigram` | 23.01 | 196.15 | 0.00 | 0.00 | 244.31 | 1176.53 | 0.94 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `plain_dflash` | 23.23 | 197.80 | 0.00 | 0.00 | 23.24 | 1174.80 | 0.38 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `dense` | 23.61 | 261.32 | 4.25 | 0.00 | 133.20 | 1549.23 | 0.46 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `dense_sparse` | 23.70 | 257.64 | 4.19 | 409.01 | 153.40 | 1533.63 | 0.46 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `online_bigram` | 23.25 | 219.68 | 0.00 | 0.00 | 107.89 | 1310.29 | 1.04 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `plain_dflash` | 23.56 | 229.79 | 0.00 | 0.00 | 10.62 | 1360.69 | 0.45 | 0.00 [disabled in M0-M4] |

*Note on `online_update`: `dflash_loop._finish_round` brackets between-round updates with CUDA events on an already-synchronized stream; for `plain_dflash`/static arms this records empty event-pair overhead (~7–23 us/round), and for `online_bigram` it records host Python table updates.*

---

## Closed-loop synchronous online adaptation and timing (`M4`)

M4 evaluated five arms (`plain_dflash`, `online_bigram`, `offline_static`, `offline_online_stride1`, `online_dbS`) across the same `480` test prompt/seed slots per arm (`2,400` evaluation requests, `7,200` timed requests, and `2,400` separate CUDA-event phase-profile requests). Separate development tuning (`artifacts/m4/settings.json`, `frozen_settings_hash = 2ec65cf4fb12b0044197a2d1c2e06a9e3cd7c3fd3eea27109ca193e3c88af056`, `180` dev requests across `20` 9-candidate bundles on GPU 0) selected `online_bigram` `{"alpha": 1.0, "k_ret": null, "lambda_tr": 0.2}` (`0.9722`), `offline_static` `{"beta": 0.5}` (`0.6939`), `offline_online_stride1` `{"beta": 1.0, "lambda_l1_b": 1e-06, "lambda_l1_s": 1e-05, "lambda_l2": 0.01, "lr": 0.0001, "update_stride": 1, "weighting_mode": "exponential"}` (`0.3585`), and `online_dbS` `{"beta": 1.0, "lambda_l1_b": 1e-06, "lambda_l1_s": 1e-05, "lambda_l2": 0.01, "lr": 0.0001, "update_stride": 1, "weighting_mode": "exponential"}` (`0.3144`). Because M4 selected the same `online_bigram` hyperparameters as M3, `960` matching `plain_dflash` and `online_bigram` evaluation rows reused checksum-verified M3 acceptance/output records with all timing and memory fields cleared, while all `7,200` M4 timed runs and `2,400` M4 phase-profile runs executed freshly in the M4 timing session (`no M3/M4 throughput reuse`). Independent reference validation ([m4_validation_reference.json](../artifacts/m4_validation_reference.json)) passed **`18,983,686` checks** across all `12,240` M4 raw rows (`1,790,609` online blocks and updates) with zero failures.

### M4 Development Selection (`tune`)

| Arm | Candidate setting | Selected | Selection score | Std norm tput | Long norm tput | Dev reqs | Rep-excl pairs |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: |
| `plain_dflash` | `{}` | yes | 1.0000 | 1.0000 | 1.0000 | 20 | 0 |
| `online_bigram` | `{"alpha": 0.5, "k_ret": null, "lambda_tr": 0.05}` | no | 0.8832 | 0.8507 | 0.9157 | 20 | 0 |
| `online_bigram` | `{"alpha": 1.0, "k_ret": null, "lambda_tr": 0.2}` | yes | 0.9722 | 0.9562 | 0.9883 | 20 | 1 |
| `offline_static` | `{"beta": 0.5}` | yes | 0.6939 | 0.6750 | 0.7127 | 20 | 1 |
| `offline_static` | `{"beta": 1.0}` | no | 0.5288 | 0.5605 | 0.4971 | 20 | 0 |
| `offline_online_stride1` | `{"beta": 1.0, "lambda_l1_b": 1e-06, "lambda_l1_s": 1e-05, "lambda_l2": 0.01, "lr": 0.0001, "update_stride": 1, "weighting_mode": "exponential"}` | yes | 0.3585 | 0.3124 | 0.4045 | 20 | 0 |
| `offline_online_stride1` | `{"beta": 1.0, "lambda_l1_b": 1e-06, "lambda_l1_s": 1e-05, "lambda_l2": 0.1, "lr": 0.001, "update_stride": 1, "weighting_mode": "exponential"}` | no | 0.2250 | 0.1695 | 0.2805 | 20 | 0 |
| `online_dbS` | `{"beta": 1.0, "lambda_l1_b": 1e-06, "lambda_l1_s": 1e-05, "lambda_l2": 0.01, "lr": 0.0001, "update_stride": 1, "weighting_mode": "exponential"}` | yes | 0.3144 | 0.3320 | 0.2968 | 20 | 0 |
| `online_dbS` | `{"beta": 1.0, "lambda_l1_b": 1e-06, "lambda_l1_s": 1e-05, "lambda_l2": 0.1, "lr": 0.001, "update_stride": 1, "weighting_mode": "exponential"}` | no | 0.3144 | 0.3304 | 0.2984 | 20 | 0 |

### M4 Held-Out Test Acceptance & Realized Lengths (Primary)

| Condition | Arm | Primary / Total (Rep-excl) | Mean accepted len [95% CI] | Seed std | Req median accepted len [95% CI] | Target fwds/tok [95% CI] | Mean out len (>8K / >16K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_online_stride1` | 31/32 (1: math500-long-0105) | 2.574 [2.424, 2.730] | 0.0000 | 1.226 [1.097, 1.387] | 0.2844 [0.2721, 0.2971] | 11252.0 (28 / 3) |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_static` | 32/32 (0) | 1.972 [1.876, 2.068] | 0.0000 | 1.000 [1.000, 1.000] | 0.3398 [0.3287, 0.3514] | 11188.3 (28 / 4) |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 31/32 (1: math500-long-0116) | 2.651 [2.483, 2.844] | 0.0000 | 1.323 [1.161, 1.516] | 0.2792 [0.2664, 0.2919] | 11269.7 (26 / 3) |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_dbS` | 32/32 (0) | 1.435 [1.354, 1.524] | 0.0000 | 0.500 [0.312, 0.688] | 0.4154 [0.4008, 0.4293] | 9915.7 (25 / 0) |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 32/32 (0) | 2.476 [2.289, 2.654] | 0.0000 | 1.281 [1.125, 1.438] | 0.2944 [0.2792, 0.3115] | 11512.1 (25 / 6) |
| Standard non-thinking greedy (T=0) | `offline_online_stride1` | 64/64 (0) | 3.196 [3.043, 3.350] | 0.0000 | 1.734 [1.578, 1.891] | 0.2482 [0.2387, 0.2582] | 298.1 (0 / 0) |
| Standard non-thinking greedy (T=0) | `offline_static` | 64/64 (0) | 2.283 [2.146, 2.418] | 0.0000 | 1.188 [1.086, 1.297] | 0.3181 [0.3045, 0.3330] | 298.1 (0 / 0) |
| Standard non-thinking greedy (T=0) | `online_bigram` | 64/64 (0) | 5.311 [5.081, 5.545] | 0.0000 | 3.594 [3.289, 3.898] | 0.1665 [0.1605, 0.1729] | 298.1 (0 / 0) |
| Standard non-thinking greedy (T=0) | `online_dbS` | 64/64 (0) | 2.283 [2.145, 2.418] | 0.0000 | 1.180 [1.078, 1.281] | 0.3181 [0.3046, 0.3330] | 298.1 (0 / 0) |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 64/64 (0) | 5.285 [5.057, 5.519] | 0.0000 | 3.578 [3.289, 3.867] | 0.1672 [0.1612, 0.1737] | 298.1 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_online_stride1` | 192/192 (0) | 3.172 [3.024, 3.318] | 0.2375 | 1.784 [1.664, 1.901] | 0.2502 [0.2409, 0.2606] | 299.0 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_static` | 192/192 (0) | 4.041 [3.847, 4.228] | 0.3157 | 2.438 [2.271, 2.609] | 0.2087 [0.2010, 0.2172] | 304.2 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 192/192 (0) | 5.195 [4.976, 5.406] | 0.4762 | 3.544 [3.346, 3.737] | 0.1702 [0.1643, 0.1766] | 302.3 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_dbS` | 192/192 (0) | 2.314 [2.176, 2.441] | 0.1633 | 1.221 [1.141, 1.299] | 0.3154 [0.3028, 0.3293] | 302.5 (0 / 0) |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 192/192 (0) | 5.136 [4.922, 5.346] | 0.3934 | 3.469 [3.263, 3.685] | 0.1714 [0.1655, 0.1778] | 296.5 (0 / 0) |
| Standard pure-T=1 diagnostic | `offline_online_stride1` | 192/192 (0) | 2.877 [2.742, 3.010] | 0.2746 | 1.630 [1.526, 1.727] | 0.2695 [0.2596, 0.2803] | 301.9 (0 / 0) |
| Standard pure-T=1 diagnostic | `offline_static` | 192/192 (0) | 3.666 [3.500, 3.828] | 0.3909 | 2.208 [2.060, 2.352] | 0.2252 [0.2172, 0.2334] | 304.1 (0 / 0) |
| Standard pure-T=1 diagnostic | `online_bigram` | 192/192 (0) | 4.388 [4.190, 4.590] | 0.4452 | 2.833 [2.667, 3.005] | 0.1959 [0.1887, 0.2037] | 296.5 (0 / 0) |
| Standard pure-T=1 diagnostic | `online_dbS` | 192/192 (0) | 2.064 [1.950, 2.179] | 0.2104 | 1.104 [1.052, 1.156] | 0.3408 [0.3279, 0.3542] | 301.6 (0 / 0) |
| Standard pure-T=1 diagnostic | `plain_dflash` | 192/192 (0) | 4.297 [4.124, 4.477] | 0.4444 | 2.768 [2.630, 2.911] | 0.1987 [0.1917, 0.2064] | 299.5 (0 / 0) |

### M4 Secondary Sensitivity: All `long_thinking` Test Generations (Including Repetition-Dominated Loops)

| Arm | All requests | All-rows mean accepted len [95% CI] | Target fwds/tok [95% CI] | Mean out len (>8K / >16K) |
| --- | ---: | ---: | ---: | ---: |
| `offline_online_stride1` | 32 | 2.789 [2.449, 3.293] | 0.2785 [0.2616, 0.2947] | 11668.3 (29 / 4) |
| `offline_static` | 32 | 1.972 [1.876, 2.068] | 0.3398 [0.3287, 0.3514] | 11188.3 (28 / 4) |
| `online_bigram` | 32 | 2.732 [2.518, 2.991] | 0.2755 [0.2604, 0.2890] | 11315.6 (27 / 3) |
| `online_dbS` | 32 | 1.435 [1.354, 1.524] | 0.4154 [0.4008, 0.4293] | 9915.7 (25 / 0) |
| `plain_dflash` | 32 | 2.476 [2.289, 2.654] | 0.2944 [0.2792, 0.3115] | 11512.1 (25 / 6) |

### M4 Acceptance Probability by Logical Block Position (`0..K-1`)

| Condition | Arm | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 | pos 8 | pos 9 | pos 10 | pos 11 | pos 12 | pos 13 | pos 14 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_online_stride1` | 0.674 | 0.463 | 0.332 | 0.247 | 0.191 | 0.151 | 0.120 | 0.097 | 0.078 | 0.062 | 0.050 | 0.039 | 0.031 | 0.023 | 0.017 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_static` | 0.627 | 0.388 | 0.253 | 0.174 | 0.127 | 0.097 | 0.075 | 0.059 | 0.046 | 0.036 | 0.029 | 0.022 | 0.017 | 0.013 | 0.009 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 0.714 | 0.485 | 0.336 | 0.247 | 0.189 | 0.150 | 0.121 | 0.098 | 0.080 | 0.064 | 0.052 | 0.041 | 0.032 | 0.025 | 0.019 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_dbS` | 0.503 | 0.279 | 0.175 | 0.120 | 0.088 | 0.067 | 0.051 | 0.040 | 0.031 | 0.024 | 0.018 | 0.014 | 0.011 | 0.008 | 0.006 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 0.698 | 0.464 | 0.316 | 0.230 | 0.174 | 0.136 | 0.108 | 0.087 | 0.070 | 0.055 | 0.044 | 0.034 | 0.026 | 0.020 | 0.014 |
| Standard non-thinking greedy (T=0) | `offline_online_stride1` | 0.737 | 0.523 | 0.396 | 0.304 | 0.243 | 0.198 | 0.163 | 0.139 | 0.113 | 0.094 | 0.079 | 0.066 | 0.057 | 0.046 | 0.039 |
| Standard non-thinking greedy (T=0) | `offline_static` | 0.655 | 0.415 | 0.289 | 0.207 | 0.159 | 0.124 | 0.095 | 0.077 | 0.064 | 0.051 | 0.043 | 0.033 | 0.027 | 0.024 | 0.019 |
| Standard non-thinking greedy (T=0) | `online_bigram` | 0.890 | 0.726 | 0.593 | 0.500 | 0.419 | 0.361 | 0.315 | 0.282 | 0.248 | 0.219 | 0.195 | 0.173 | 0.147 | 0.127 | 0.114 |
| Standard non-thinking greedy (T=0) | `online_dbS` | 0.655 | 0.415 | 0.289 | 0.207 | 0.159 | 0.124 | 0.095 | 0.077 | 0.064 | 0.051 | 0.043 | 0.034 | 0.027 | 0.024 | 0.019 |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 0.892 | 0.727 | 0.589 | 0.497 | 0.416 | 0.357 | 0.315 | 0.277 | 0.247 | 0.219 | 0.194 | 0.172 | 0.146 | 0.125 | 0.112 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_online_stride1` | 0.738 | 0.535 | 0.400 | 0.304 | 0.238 | 0.191 | 0.156 | 0.130 | 0.111 | 0.091 | 0.077 | 0.065 | 0.054 | 0.045 | 0.037 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_static` | 0.793 | 0.608 | 0.477 | 0.382 | 0.311 | 0.257 | 0.219 | 0.190 | 0.165 | 0.145 | 0.128 | 0.111 | 0.097 | 0.085 | 0.072 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 0.889 | 0.732 | 0.590 | 0.489 | 0.408 | 0.344 | 0.299 | 0.263 | 0.231 | 0.204 | 0.186 | 0.167 | 0.150 | 0.131 | 0.112 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_dbS` | 0.669 | 0.432 | 0.295 | 0.209 | 0.154 | 0.118 | 0.094 | 0.077 | 0.065 | 0.052 | 0.044 | 0.036 | 0.028 | 0.024 | 0.018 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 0.882 | 0.731 | 0.592 | 0.490 | 0.405 | 0.344 | 0.295 | 0.257 | 0.228 | 0.201 | 0.180 | 0.159 | 0.141 | 0.124 | 0.106 |
| Standard pure-T=1 diagnostic | `offline_online_stride1` | 0.729 | 0.514 | 0.371 | 0.273 | 0.209 | 0.166 | 0.129 | 0.107 | 0.089 | 0.073 | 0.061 | 0.052 | 0.043 | 0.035 | 0.029 |
| Standard pure-T=1 diagnostic | `offline_static` | 0.797 | 0.599 | 0.447 | 0.343 | 0.272 | 0.221 | 0.184 | 0.161 | 0.137 | 0.118 | 0.103 | 0.089 | 0.076 | 0.065 | 0.055 |
| Standard pure-T=1 diagnostic | `online_bigram` | 0.863 | 0.683 | 0.526 | 0.416 | 0.331 | 0.275 | 0.233 | 0.200 | 0.175 | 0.153 | 0.135 | 0.120 | 0.107 | 0.091 | 0.079 |
| Standard pure-T=1 diagnostic | `online_dbS` | 0.653 | 0.405 | 0.265 | 0.183 | 0.131 | 0.095 | 0.074 | 0.061 | 0.049 | 0.039 | 0.032 | 0.026 | 0.020 | 0.017 | 0.013 |
| Standard pure-T=1 diagnostic | `plain_dflash` | 0.860 | 0.675 | 0.522 | 0.408 | 0.323 | 0.264 | 0.226 | 0.197 | 0.170 | 0.150 | 0.131 | 0.116 | 0.099 | 0.084 | 0.071 |

### M4 Acceptance by Long Output-Token Offset & Input-Context-Length Bins (`16K+` Sparsity Gate Enforced)

Notice how `offline_online_stride1` steadily **increases** accepted block length as generation progresses within each OOD `long_thinking` request (`2.198` at `0-1K` -> `2.554` at `1K-4K` -> `2.893` at `4K-8K`, and `2.402` at `8K-16K` vs `1.826` for `plain_dflash` and `1.376` for `offline_static`!). Bins with `< 8` primary prompts (`16K+` and `16384-32768`) are marked **`SPARSE / EMPTY (conclusion unavailable)`** per protocol.

| Arm | Axis | `0-1K` (`n`) | `1K-4K` (`n`) | `4K-8K` (`n`) | `8K-16K` (`n`) | `16K+` / `16K-32K` (`n`; gate) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `offline_online_stride1` | `output_offset` | 2.198 [2.023, 2.381] (`31`) | 2.554 [2.338, 2.792] (`31`) | 2.893 [2.635, 3.179] (`31`) | 2.402 [2.070, 2.738] (`28`) | 3.193 [2.703, 3.538] (`n=3 < 8`; **SPARSE / conclusion unavailable**) |
| `offline_static` | `output_offset` | 2.310 [2.090, 2.521] (`32`) | 2.225 [2.038, 2.422] (`32`) | 2.174 [2.077, 2.275] (`32`) | 1.376 [1.207, 1.546] (`28`) | 1.430 [0.921, 2.120] (`n=4 < 8`; **SPARSE / conclusion unavailable**) |
| `online_bigram` | `output_offset` | 2.711 [2.478, 2.997] (`31`) | 2.878 [2.636, 3.176] (`31`) | 2.895 [2.592, 3.237] (`30`) | 2.304 [1.924, 2.787] (`26`) | 2.495 [1.592, 3.624] (`n=3 < 8`; **SPARSE / conclusion unavailable**) |
| `online_dbS` | `output_offset` | 1.532 [1.380, 1.669] (`32`) | 1.545 [1.445, 1.657] (`32`) | 1.575 [1.428, 1.730] (`31`) | 0.952 [0.813, 1.089] (`25`) | — (`n=0`; **EMPTY / conclusion unavailable**) |
| `plain_dflash` | `output_offset` | 2.863 [2.629, 3.101] (`32`) | 2.750 [2.494, 3.084] (`32`) | 2.879 [2.568, 3.239] (`32`) | 1.826 [1.642, 2.008] (`25`) | 1.431 [1.031, 1.816] (`n=6 < 8`; **SPARSE / conclusion unavailable**) |
| `offline_online_stride1` | `input_context_length` | 2.181 [2.026, 2.340] (`31`) | 2.524 [2.307, 2.768] (`31`) | 2.880 [2.636, 3.154] (`31`) | 2.448 [2.125, 2.770] (`28`) | 3.106 [2.470, 3.483] (`n=3 < 8`; **SPARSE / conclusion unavailable**) |
| `offline_static` | `input_context_length` | 2.247 [2.041, 2.445] (`32`) | 2.255 [2.062, 2.466] (`32`) | 2.160 [2.067, 2.255] (`32`) | 1.457 [1.302, 1.624] (`28`) | 1.434 [0.928, 2.154] (`n=4 < 8`; **SPARSE / conclusion unavailable**) |
| `online_bigram` | `input_context_length` | 2.687 [2.466, 2.956] (`31`) | 2.905 [2.651, 3.212] (`31`) | 2.870 [2.566, 3.208] (`30`) | 2.382 [2.013, 2.865] (`27`) | 2.541 [1.668, 3.664] (`n=3 < 8`; **SPARSE / conclusion unavailable**) |
| `online_dbS` | `input_context_length` | 1.514 [1.374, 1.641] (`32`) | 1.550 [1.443, 1.663] (`32`) | 1.593 [1.436, 1.763] (`31`) | 0.971 [0.836, 1.108] (`26`) | — (`n=0`; **EMPTY / conclusion unavailable**) |
| `plain_dflash` | `input_context_length` | 2.825 [2.596, 3.053] (`32`) | 2.762 [2.508, 3.085] (`32`) | 2.900 [2.595, 3.248] (`32`) | 1.934 [1.666, 2.235] (`27`) | 1.399 [1.034, 1.772] (`n=6 < 8`; **SPARSE / conclusion unavailable**) |

### M4 Standard Greedy (`T_sample=0, beta=1`) Output Exactness & BF16 Tie Report

All five M4 arms (including `offline_online_stride1` and `online_dbS`) match `plain_dflash` bit-for-bit across all 64 greedy prompts (`0 / 19,081` token mismatches) and share the identical `42 / 64` (`0.656`) sequence difference rate against `target_only`. The discrepancies are consistent with the BF16 tie sensitivity diagnosed separately in M0.

| Arm | Reference | Compared prompts | Mismatched seqs (rate) | Mismatched tokens / compared (rate) | Length-diff seqs |
| --- | --- | ---: | ---: | ---: | ---: |
| `offline_online_stride1` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `offline_online_stride1` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |
| `offline_static` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `offline_static` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |
| `online_bigram` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `online_bigram` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |
| `online_dbS` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `online_dbS` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |
| `plain_dflash` | `plain_dflash` | 64 | 0 (0.000) | 0 / 19081 (0.0000) | 0 |
| `plain_dflash` | `target_only` | 64 | 42 (0.656) | 6662 / 19437 (0.3427) | 39 |

### M4 Core-Pinned 8-GPU Sharded Matched-Bundle Throughput, Latency & Peak Memory

> **Methodology Disclosure (`8-GPU` NUMA/16-Core-Pinned Sharded Timing + `1-GPU` Isolated Calibration):** To execute all `9,600` M4 timing and phase-profile runs (`7,200` timed + `2,400` CUDA-event phase profiles across `1,920` rotated 5-arm comparison bundles) without burning ~56 sequential GPU-hours on a single GPU while 7 B300 GPUs sat idle, M4 timing (`artifacts/m4/timing.json`, `world_size=8`, `sharding_mode=sharded_8gpu`) sharded whole prompts deterministically across all 8 NVIDIA B300 SXM6 AC GPUs (`sharding_plan.json`, `05d3bda5...`) with strict NUMA-local 16-physical-core `cpu_affinity` pinning (`GPU g -> CPUs [16g..16g+15]`, NUMA node `g // 2`, `OMP_NUM_THREADS=1`), per-GPU all-arm warmup, and external-process idle checks around every matched 5-arm bundle. All 10 core M0–M4 `ALGORITHM_FILES` remained 100% byte-identical to frozen `09a5469` (`algorithm_implementation_sha256 = 54b714207e175169a0971a4d7b7f0820cb69f5fa8ac50d799c2eef397db7dd8f`; `timing_harness_implementation_sha256 = 33e30f47057e21c8e6731c1717cfc492533b04bd49ad90a15ea0ebd0ad8b041b`). Before launching the 8-GPU run, a dedicated 1-GPU cluster-isolated calibration session ([timing_isolation_calibration.json](../artifacts/m4/timing_isolation_calibration.json), `60` runs / `12` bundles on `GPU 0`, `CPUs 0..15`, with peer GPUs `1..7` verified at `0% / 0 MiB`) measured the exact same prompt/seed slots to verify absence of multi-GPU host contention.

| Condition | Arm | Tokens/sec [95% CI] | Paired ratio vs plain [95% CI] | Net gain? (95% CI > 1?) | Latency p50 s [95% CI] | Latency p95 s [95% CI] | Peak alloc / res GiB |
| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_online_stride1` | 37.78 [36.29, 39.20] | 0.4213 [0.3799, 0.4650] | no (no) | 273.384 [244.399, 300.433] | 561.647 [360.489, 621.088] | 21.83 / 37.42 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_static` | 60.86 [57.24, 64.39] | 0.6539 [0.6038, 0.7087] | no (no) | 162.719 [142.859, 207.046] | 454.874 [268.702, 625.245] | 21.12 / 37.42 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 87.45 [82.08, 93.17] | 0.9653 [0.8712, 1.0597] | no (no) | 120.257 [105.614, 137.443] | 351.595 [168.045, 513.734] | 21.74 / 37.42 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_dbS` | 28.10 [26.91, 29.38] | 0.3091 [0.2833, 0.3377] | no (no) | 376.070 [321.567, 426.152] | 569.034 [449.404, 589.914] | 19.92 / 37.42 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 97.75 [89.17, 106.22] | 1.0000 [1.0000, 1.0000] | no (no) | 108.280 [93.055, 139.157] | 289.432 [211.679, 360.972] | 20.97 / 37.42 |
| Standard non-thinking greedy (T=0) | `offline_online_stride1` | 51.40 [49.52, 53.35] | 0.2401 [0.2351, 0.2449] | no (no) | 5.792 [5.185, 6.397] | 8.090 [7.583, 8.571] | 17.76 / 37.42 |
| Standard non-thinking greedy (T=0) | `offline_static` | 88.88 [85.17, 92.66] | 0.4153 [0.4035, 0.4266] | no (no) | 3.410 [2.927, 3.692] | 4.688 [4.506, 4.734] | 17.69 / 37.42 |
| Standard non-thinking greedy (T=0) | `online_bigram` | 207.63 [199.82, 215.72] | 0.9670 [0.9596, 0.9753] | no (no) | 1.471 [1.233, 1.615] | 2.169 [1.926, 2.522] | 17.62 / 37.42 |
| Standard non-thinking greedy (T=0) | `online_dbS` | 43.88 [42.05, 45.74] | 0.2052 [0.1992, 0.2109] | no (no) | 6.978 [5.934, 7.535] | 9.582 [9.153, 9.657] | 17.76 / 37.42 |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 215.14 [207.10, 223.57] | 1.0000 [1.0000, 1.0000] | no (no) | 1.416 [1.182, 1.526] | 2.079 [1.841, 2.382] | 17.61 / 37.42 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_online_stride1` | 47.13 [45.41, 48.87] | 0.2372 [0.2319, 0.2428] | no (no) | 6.256 [5.934, 6.837] | 9.515 [8.482, 9.100] | 17.78 / 37.42 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_static` | 115.90 [111.35, 120.48] | 0.5820 [0.5687, 0.5956] | no (no) | 2.572 [2.315, 2.807] | 4.004 [3.552, 4.644] | 17.71 / 37.42 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 173.82 [167.41, 179.84] | 0.8742 [0.8575, 0.8922] | no (no) | 1.733 [1.585, 1.904] | 2.571 [2.310, 2.619] | 17.64 / 37.42 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_dbS` | 40.45 [38.79, 42.05] | 0.2035 [0.1977, 0.2096] | no (no) | 7.328 [6.706, 8.217] | 11.350 [10.023, 11.584] | 17.77 / 37.42 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 201.46 [194.45, 208.53] | 1.0000 [1.0000, 1.0000] | no (no) | 1.458 [1.335, 1.557] | 2.218 [1.992, 2.225] | 17.62 / 37.42 |
| Standard pure-T=1 diagnostic | `offline_online_stride1` | 45.89 [44.28, 47.48] | 0.2611 [0.2550, 0.2674] | no (no) | 6.508 [6.144, 7.342] | 9.729 [8.385, 9.175] | 17.77 / 37.42 |
| Standard pure-T=1 diagnostic | `offline_static` | 117.77 [113.51, 122.00] | 0.6703 [0.6528, 0.6891] | no (no) | 2.456 [2.312, 2.906] | 4.052 [3.329, 3.781] | 17.71 / 37.42 |
| Standard pure-T=1 diagnostic | `online_bigram` | 166.26 [160.12, 172.51] | 0.9459 [0.9190, 0.9727] | no (no) | 1.723 [1.624, 1.963] | 2.721 [2.383, 2.645] | 17.64 / 37.42 |
| Standard pure-T=1 diagnostic | `online_dbS` | 39.32 [37.85, 40.81] | 0.2237 [0.2169, 0.2305] | no (no) | 7.293 [6.832, 8.420] | 11.820 [10.218, 13.223] | 17.81 / 37.42 |
| Standard pure-T=1 diagnostic | `plain_dflash` | 178.12 [172.28, 184.48] | 1.0000 [1.0000, 1.0000] | no (no) | 1.659 [1.510, 1.807] | 2.667 [2.308, 2.522] | 17.62 / 37.42 |

### M4 1-GPU Isolated vs 8-GPU Core-Pinned Timing Calibration (`timing_isolation_calibration.json`)

On `standard_nonthinking` (`4` prompts, `8` 1-GPU runs vs `12` 8-GPU runs per arm), 8-GPU NUMA/16-core-pinned throughput matches 1-GPU cluster-isolated throughput within **`-0.51%` to `+0.48%`** across all five arms (and paired `arm/plain` ratios match within `0.0004–0.0073`), completely eliminating the `-39%` host-contention slowdown previously observed in unpinned sharded evaluation. On `long_thinking` (`2` prompts: `math500-long-0233`, `math500-long-0384`), the 1-GPU calibration ran `long_thinking` first, so its very first long request (`repeat=0` on `math500-long-0233`) incurred a one-time PyTorch CUDA caching-allocator expansion (`42.74 tok/s`), lowering the 2-repeat 1-GPU `plain_dflash` mean (`89.88` vs `109.70 tok/s` in 8-GPU, where standard blocks warmed the allocator first); on warmed repeats (`repeat >= 1`), 8-GPU matches 1-GPU within `0.4%–1.8%` with bit-identical generated token counts (`7,498` and `9,162`) and accepted lengths. Neither regime shows a throughput gain for any adapter arm.

| Condition | Arm | Prompts (`n`) | 1-GPU isolated `tok/s` | 8-GPU sharded `tok/s` | `8-GPU / 1-GPU` scale (`% diff`) | 1-GPU `arm/plain` ratio | 8-GPU `arm/plain` ratio | Paired ratio diff (`8-GPU - 1-GPU`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 2 | 89.88 | 109.70 | 1.2205 (+22.05%) | 1.0000 | 1.0000 | +0.0000 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 2 | 79.40 | 93.22 | 1.1740 (+17.40%) | 0.9212 | 0.8471 | -0.0741 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_static` | 2 | 58.60 | 67.51 | 1.1520 (+15.20%) | 0.7124 | 0.6146 | -0.0979 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_online_stride1` | 2 | 37.48 | 39.66 | 1.0583 (+5.83%) | 0.4853 | 0.3607 | -0.1246 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_dbS` | 2 | 28.73 | 29.84 | 1.0383 (+3.83%) | 0.3714 | 0.2723 | -0.0991 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 4 | 159.17 | 158.36 | 0.9949 (-0.51%) | 1.0000 | 1.0000 | +0.0000 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 4 | 165.87 | 165.48 | 0.9977 (-0.23%) | 1.0563 | 1.0596 | +0.0033 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_static` | 4 | 101.42 | 101.90 | 1.0048 (+0.48%) | 0.6380 | 0.6453 | +0.0073 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_online_stride1` | 4 | 41.07 | 40.99 | 0.9981 (-0.19%) | 0.2584 | 0.2592 | +0.0008 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_dbS` | 4 | 33.54 | 33.40 | 0.9958 (-0.42%) | 0.2104 | 0.2108 | +0.0004 |

### M4 Separate Phase-Profile Median Durations (ms/request; `lookahead_iterations = 0.0 ms [disabled in M0-M4]`)

On `long_thinking` (`T=0.6`), `offline_online_stride1` reduces `target_verification` median duration by **`7.68s`** vs `plain_dflash` (`80,216.4 ms` vs `87,892.8 ms`) and by **`14.95s`** vs `offline_static` (`95,162.9 ms`). Mean accepted lengths on each arm’s eligible prompts were `2.574`, `2.476` and `1.972`, respectively. However, in uncompiled eager M0–M4 (`09a5469`), `online_update` (`118,227.7 ms`), `sparse_lookup` (`36,643.7 ms`), and full-vocabulary `sequential_sampling` (`18,656.5 ms`) dominate request latency.

| Condition | Arm | Target feat | DFlash draft | Dense score | Sparse lookup | Seq sample | Target verify | Online update* | Lookahead iters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long_thinking (T=0.6) | `offline_online_stride1` | 144.43 | 12994.35 | 190.20 | 36643.70 | 18656.54 | 80216.44 | 118227.66 | 0.00 [disabled in M0-M4] |
| long_thinking (T=0.6) | `offline_static` | 162.36 | 14627.56 | 202.66 | 23873.86 | 20496.74 | 95162.91 | 25.78 | 0.00 [disabled in M0-M4] |
| long_thinking (T=0.6) | `online_bigram` | 140.48 | 12220.02 | 0.00 | 0.00 | 15071.69 | 79292.93 | 45.98 | 0.00 [disabled in M0-M4] |
| long_thinking (T=0.6) | `online_dbS` | 199.07 | 18628.42 | 276.91 | 53319.66 | 27019.57 | 118883.58 | 141429.91 | 0.00 [disabled in M0-M4] |
| long_thinking (T=0.6) | `plain_dflash` | 141.08 | 13047.94 | 0.00 | 0.00 | 1226.62 | 87892.77 | 22.09 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `offline_online_stride1` | 24.74 | 303.67 | 4.76 | 827.20 | 55.62 | 1714.88 | 2792.43 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `offline_static` | 25.59 | 382.56 | 6.05 | 562.01 | 64.51 | 2234.34 | 0.65 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `online_bigram` | 23.82 | 195.59 | 0.00 | 0.00 | 28.93 | 1145.51 | 0.93 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `online_dbS` | 25.87 | 394.06 | 6.10 | 1087.75 | 72.95 | 2243.74 | 3035.94 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0) | `plain_dflash` | 23.96 | 194.78 | 0.00 | 0.00 | 2.42 | 1144.26 | 0.37 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `offline_online_stride1` | 24.97 | 301.00 | 5.11 | 875.78 | 448.05 | 1760.54 | 2759.49 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `offline_static` | 24.54 | 244.65 | 4.39 | 386.52 | 352.03 | 1474.92 | 0.43 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `online_bigram` | 24.00 | 201.43 | 0.00 | 0.00 | 253.15 | 1219.40 | 0.97 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `online_dbS` | 25.89 | 381.72 | 6.59 | 1105.13 | 565.46 | 2264.41 | 2896.45 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=0.7) | `plain_dflash` | 24.07 | 199.12 | 0.00 | 0.00 | 23.69 | 1194.83 | 0.39 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `offline_online_stride1` | 24.93 | 324.80 | 5.47 | 937.77 | 202.59 | 1900.13 | 2982.98 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `offline_static` | 24.33 | 260.56 | 4.51 | 407.72 | 154.91 | 1576.21 | 0.46 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `online_bigram` | 24.12 | 221.88 | 0.00 | 0.00 | 107.44 | 1324.95 | 1.03 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `online_dbS` | 25.75 | 397.11 | 6.78 | 1137.68 | 248.59 | 2308.83 | 3037.36 | 0.00 [disabled in M0-M4] |
| standard_nonthinking (T=1) | `plain_dflash` | 24.16 | 232.42 | 0.00 | 0.00 | 10.69 | 1384.33 | 0.45 | 0.00 [disabled in M0-M4] |

### M4 Online Parameter Norms, Exact-Zero Sparsity, Active Buckets & Update Bandwidth

Each request starts from zero online deltas (`dW=0, dA=0, db=0, dS=0`) with request-local `LazySignedHash` capacity `1,048,576`. In `offline_online_stride1` on `long_thinking`, dense deltas reach `||dW||_2 = 0.70040` and `||dA||_2 = 1.01633` across a mean of `3,114.7` stride-1 `live_full` KL updates, actively correcting the M2 hidden/pair weights under `h[i]` shift; in `online_dbS`, `dW` and `dA` remain identically zero while `||db||_2` grows `6.8x` larger (`0.11508` vs `0.01693`) without rescuing acceptance (`1.435`).

| Condition | Arm | Mean updates `u` | `||dW||_2` | `||dA||_2` | `||db||_2` | `||dS||_2` | `db` exact-zero frac | Active buckets mean (max / cap) | Est state BW GB/s [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long_thinking (T=0.6) | `offline_online_stride1` | 3114.7 | 0.70040 | 1.01633 | 0.01693 | 0.02231 | 0.8254 | 321705.0 (380121 / 1048576) | 0.477 [0.464, 0.489] |
| long_thinking (T=0.6) | `online_dbS` | 4131.2 | 0.00000 | 0.00000 | 0.11508 | 0.05444 | 0.6627 | 439967.4 (519365 / 1048576) | 0.654 [0.637, 0.669] |
| standard_nonthinking (T=0) | `offline_online_stride1` | 71.3 | 0.19046 | 0.16220 | 0.00116 | 0.00211 | 0.8784 | 69014.5 (112935 / 1048576) | 0.273 [0.270, 0.276] |
| standard_nonthinking (T=0) | `online_dbS` | 91.3 | 0.00000 | 0.00000 | 0.00310 | 0.00304 | 0.8095 | 99145.4 (162429 / 1048576) | 0.336 [0.331, 0.341] |
| standard_nonthinking (T=0.7) | `offline_online_stride1` | 72.3 | 0.19290 | 0.16264 | 0.00118 | 0.00211 | 0.8789 | 70654.9 (157998 / 1048576) | 0.271 [0.268, 0.275] |
| standard_nonthinking (T=0.7) | `online_dbS` | 92.3 | 0.00000 | 0.00000 | 0.00314 | 0.00300 | 0.7994 | 101915.1 (178404 / 1048576) | 0.335 [0.329, 0.340] |
| standard_nonthinking (T=1) | `offline_online_stride1` | 78.5 | 0.20039 | 0.17043 | 0.00125 | 0.00220 | 0.8639 | 77795.3 (147401 / 1048576) | 0.277 [0.274, 0.281] |
| standard_nonthinking (T=1) | `online_dbS` | 99.3 | 0.00000 | 0.00000 | 0.00333 | 0.00304 | 0.7651 | 111839.0 (222334 / 1048576) | 0.343 [0.336, 0.349] |

### M4 Linked M1b Plain-DFlash `h[i]` Covariate-Shift Context (`feature_context.json`)

Paired from checksummed [diagnostic_reference.json](../artifacts/diagnostic_reference.json) via [m4/feature_context.json](../artifacts/m4/feature_context.json) (plain-DFlash M1 trace distribution prior to offline fitting and online updates; `16K+` covers only `2` primary prompts / `4,115` slots and is sparse):

| Condition | Temp | Output bin | Logical slots | Coord mean norm | Variance / cal | L2 median (p95) | Outlier slot rate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `standard_nonthinking` | 0 | `0-1K` | 18785 | 11.328 | 1.344 | 20.833 (28.128) | 11.381% |
| `standard_nonthinking` | 0.7 | `0-1K` | 18887 | 11.302 | 1.347 | 20.839 (28.166) | 11.389% |
| `standard_nonthinking` | 1 | `0-1K` | 18822 | 11.319 | 1.345 | 20.808 (28.228) | 11.162% |
| `long_thinking` | 0.6 | `0-1K` | 31576 | 7.945 | 0.895 | 16.762 (20.290) | 0.333% |
| `long_thinking` | 0.6 | `1K-4K` | 93579 | 8.175 | 0.920 | 17.059 (20.675) | 0.499% |
| `long_thinking` | 0.6 | `4K-8K` | 114615 | 8.434 | 0.824 | 16.517 (19.606) | 0.392% |
| `long_thinking` | 0.6 | `8K-16K` | 91331 | 9.152 | 0.749 | 16.249 (19.618) | 0.537% |
| `long_thinking` | 0.6 | `16K+` (*sparse, n=2*) | 4115 | 10.288 | 0.736 | 16.486 (21.850) | 0.875% |

---

## Verification, scope boundaries, and external baselines

The reference run passed the Section 12 CPU tests then present. A later sampler check found a counterexample to test 24 for an unseen transition in a populated bigram table. Additional tests verified M3->M4 baseline reuse, direct greedy comparisons, matched-bundle timing resume integrity, lazy sparse L1 norms, version boundaries, feature-context linkage, and NUMA/core-pinned sharded timing reconciliation. Synthetic fitting and reference-sized random-tensor microbenchmarks (`bench_adapter`) are implementation checks only; all empirical conclusions above come strictly from the completed `Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash-b16` `--scale reference` experiments. Operational notes and artifact SHA256s are recorded in [NOTES.md](../NOTES.md).

**External baselines not run in M0–M4:** full tree-based Token Recycling, Test-Time Speculation, and SharpDraft. The required normalized committed-transition `online_bigram` baseline was evaluated across all M3 and M4 reference conditions and timing sessions above (never labeled as full Token Recycling). Optional `--scale full` experiments and optional M5 extensions (lookahead belief propagation in closed loop, exact collision-free sparse tables, two-sided pseudolikelihood, and systems/kernel fastpaths) are outside the mandatory M0–M4 reference scope reported here.

## Paired long-thinking throughput at reference scale

At T = 0.6, top-k = 20 and top-p = 0.95, the zero-offline online-only adapter delivered 0.8728× plain DFlash throughput over whole requests (95% CI 0.7876–0.9550, 31 prompts). In the 8K–16K output bin it accepted +1.1617 more tokens per round (95% CI 0.4787–1.9615, 20 prompts). The isolated throughput ratio in that bin was 1.1571× (95% CI 0.9343–1.4420, 20 prompts). The overall interval lies below 1 and the deep-bin interval includes 1.

The table shows whole-request ratios from joint isolated timing bundles. Ratios are paired within each prompt, seed and repeat, then averaged by seed and prompt. The bootstrap resamples prompt IDs. Both arms must pass the repetition filter.

| Arm | Comparison | Mean throughput ratio | 95% CI | Prompts |
| --- | --- | ---: | ---: | ---: |
| `online_dbS_zero_init` | `plain_dflash` | 0.7606× | 0.6877–0.8340 | 32 |
| `online_bigram_count_gated` | `online_bigram` | 1.0285× | 0.9339–1.1288 | 32 |
| `lookahead_static_candidate` | `dense_candidate` | 1.2468× | 1.1567–1.3540 | 32 |
| `lookahead_rerelax_candidate` | `dense_candidate` | 1.0807× | 0.9901–1.1719 | 32 |
| `exact_observed_pairs_online` | `signed_feature_hash_online` | 0.9465× | 0.9076–0.9922 | 32 |
| `online_buffered_stride5` | `offline_online_stride5` | 0.8500× | 0.8099–0.8882 | 32 |

## Paired M4 accepted lengths on long-thinking prompts

For each pair, the means below average the within-prompt difference on the jointly repetition-filtered population. Positive values favor the synchronous full-adapter update.

| Reference arm | Extra accepted slots per block | 95% prompt-bootstrap CI | Prompts |
| --- | ---: | ---: | ---: |
| `plain_dflash` | +0.0893 | -0.1502 to +0.2943 | 31 |
| `online_dbS` | +1.1323 | +0.9748 to +1.2820 | 31 |
| `offline_static` | +0.6144 | +0.4801 to +0.7468 | 31 |
| `online_bigram` | -0.0431 | -0.2618 to +0.1379 | 30 |

The full-adapter mean was higher than plain DFlash, with a paired interval spanning zero. Its acceptance advantage over the bias-and-sparse update was positive throughout the paired interval. See `artifacts/logs/resume_audit/reference_m4_long_acceptance_pairs.json`.

## Proposal scoring and snapshot qualifications

The post-M4 compact bigram scoring path can choose a different candidate after FP32 rounding at a greedy tie or a positive-temperature top-k cutoff. Phase A comparisons involving optimized bigram arms describe the implementation that produced these measurements. The frozen M0–M4 reference used full-vocabulary scoring before warping. For a previously unseen predecessor in a populated bigram table, an FP32-normalized near-tie can select a different greedy token from the plain raw-logit path.

In the Phase A candidate-masked lookahead implementation, applying native top-k to a reduced candidate vector can retain different IDs at an exact cutoff tie than applying it to the same scores masked on the full vocabulary. A synthetic tied-input differential demonstrated different proposal supports. Its frequency in the reported requests has not been measured.

In the frozen M4 implementation at `09a5469`, each lazy signed-hash snapshot copied the active bucket IDs, values and timestamps. The immutable snapshots preserved the numerical L1 update while exceeding Section 9’s per-round copy budget for untouched buckets. The post-M4 implementation used copy-on-write for snapshots, while a positive-L1 CUDA sweep still read the full table. The later dirty-bucket implementation at `ba14428` addresses that read cost.

## Reference timing corrections

All 28 median bootstrap intervals in `artifacts/adapter_benchmark_reference.json` used the lower middle observation for each 30-sample resample. The corrected intervals average the two middle observations. The point medians, p95 values, generation traces and token throughput retain their recorded values. The eight odd-sample grouped benchmark intervals agree with the recomputation. See `artifacts/logs/resume_audit/reference_benchmark_median_intervals.json`.

Across 16 stored requests and 53,773 blocks, CPU offset-bin aggregation ran 8.58× faster with exactly equal output. This measures offline analysis time. See `artifacts/logs/resume_audit/offset_aggregation_cpu.json`.

## Reference timing and single-GPU calibration

Online head-to-head (4,800 timed requests) and M5 extensions (9,600 timed requests) used eight prompt-sharded GPU workers. Each worker timed one arm at a time. The historical timing files include a generic single-GPU calibration label. Neither of these two suites executed that calibration. The separate M4 and fastpath calibration measurements apply to their own prompt subsets and execution settings.

## Reference bigram correction

All 480 reference M3 bigram requests visited a predecessor with zero count in a populated table, across 230,450 of 1,826,049 draft slots. M4 reused the same histories. On BF16 test inputs, the sampler at `09a5469` produced a different greedy token and a different stochastic support from plain DFlash. The archived blocks contain draft and committed tokens. They omit the raw logits and full probability vectors needed to estimate the effect on the published results. The original M3 and M4 bigram comparisons describe that historical implementation. The Phase B reference bigram experiment uses fresh joint timing for the corrected comparisons. See `artifacts/logs/resume_audit/reference_bigram_empty_row_parity.json`.

### Section 8 verifier precision in the frozen reference

The Phase A verifier at `b92f0fb` can make a different accept/reject decision at a floating-point boundary. In a two-token native softmax example with seed 2026, the normalized masses required acceptance on uniform 0.15248114, and the implementation rejected. Separate fixtures reproduced a rounded FP32 product rejection and acceptance at zero target mass on this revision. On the same tiny-proposal fixture, earlier sources `09a5469` and `dfe72a0` raised a residual error. The archived tests for Section 12 items 11–13 passed on their original toy fixtures. At 00:45:45 UTC on 18 November 2026, the selected probabilities and decisions obeyed the checked forced-accept and forced-reject bounds across 4,668,128 reached slots in 4,160 stochastic M5 evaluation requests and 10,159,785 reached slots in 1,451 stochastic timing requests. The stored traces omit acceptance uniforms and full probability vectors, leaving counterfactual effects on recorded outputs unknown. Buffered stride-five timing and evaluation emitted different tokens in 91/154 same-seed comparisons in this snapshot. The first recorded probability differences appeared in the proposal immediately after its first online update, and the traces do not establish a verifier cause. The original measurements and M0 comparison remain attributed to their recorded source commits.

The original M1a traces include 1,208 stochastic requests and 855,171 reached decisions. The archived verifier mishandled constructed inputs with the same source bytes used for collection. Among the observed decisions, 74,107 had a raw target-to-proposal probability ratio between zero and one. The archive omits the acceptance uniforms and paired row totals needed to replay borderline decisions. Checks of the recorded scalars found no demonstrably wrong stochastic decision. The historical primary M2 fit used 501,510 generated training tokens. The new ten-arm reference comparison uses these historical weights. Phase B also schedules a fresh reference M1 collection, primary M2 fit and matched M3/M4 evaluation with the corrected verifier, alongside its full-scale study.

## Offline refit comparisons on reference prompts

Each row compares a dense-plus-sparse fit with a baseline on the same requests. Seed 2026 was held fixed across 2 timing repeats. Values are prompt means with 95% percentile intervals from 2,000 prompt resamples. Long requests enter a comparison only when both outputs pass the repetition filter.

| Fit | Condition | Temperature | Baseline | Prompts | Eligible / excluded repeats | Accepted-length difference | Throughput ratio |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Dev-calibrated | long_thinking | 0.6 | Plain DFlash | 31 | 62 / 2 | -0.126 [-0.341, 0.066] | 0.845 [0.752, 0.947] |
| Dev-calibrated | standard_nonthinking | 0.7 | Plain DFlash | 64 | 128 / 0 | -0.225 [-0.402, -0.052] | 0.688 [0.668, 0.707] |
| Fixed scale 0.20 | long_thinking | 0.6 | Plain DFlash | 32 | 64 / 0 | 0.091 [-0.128, 0.310] | 0.936 [0.844, 1.036] |
| Fixed scale 0.20 | standard_nonthinking | 0.7 | Plain DFlash | 64 | 128 / 0 | -0.051 [-0.246, 0.144] | 0.712 [0.689, 0.737] |
| Dev-calibrated | long_thinking | 0.6 | Original M2 | 30 | 60 / 4 | 0.848 [0.727, 0.966] | 1.406 [1.339, 1.475] |
| Dev-calibrated | standard_nonthinking | 0.7 | Original M2 | 64 | 128 / 0 | 2.658 [2.472, 2.863] | 1.801 [1.731, 1.875] |
| Dev-calibrated | long_thinking | 0.6 | Same fit before calibration | 31 | 62 / 2 | 0.749 [0.616, 0.870] | 1.286 [1.229, 1.347] |
| Dev-calibrated | standard_nonthinking | 0.7 | Same fit before calibration | 64 | 128 / 0 | 1.916 [1.727, 2.115] | 1.473 [1.418, 1.541] |

### Same-seed M5 output replay

Each cell counts identical full output-token sequences over paired requests. Each pair shares its prompt, seed, sampling temperature, warper and arm. Repeats 0 and 1 are isolated timing runs. Profiles collect CUDA phase timings. Per arm, the greedy group covers 64 standard prompts. The stochastic group covers 64 standard prompts at two temperatures with three seeds each and 32 long-thinking prompts with one seed each.

| Arm | Sampling | Repeat 0 vs evaluation | Repeat 1 vs evaluation | Profile vs evaluation | Repeat 0 vs repeat 1 |
|---|---|---:|---:|---:|---:|
| `plain_dflash` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `plain_dflash` | stochastic | 416/416 | 416/416 | 416/416 | 416/416 |
| `dense_candidate` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `dense_candidate` | stochastic | 416/416 | 416/416 | 416/416 | 416/416 |
| `lookahead_static_candidate` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `lookahead_static_candidate` | stochastic | 416/416 | 416/416 | 416/416 | 416/416 |
| `lookahead_rerelax_candidate` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `lookahead_rerelax_candidate` | stochastic | 416/416 | 416/416 | 416/416 | 416/416 |
| `online_two_sided_pl` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `online_two_sided_pl` | stochastic | 416/416 | 416/416 | 416/416 | 416/416 |
| `signed_feature_hash_online` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `signed_feature_hash_online` | stochastic | 415/416 | 415/416 | 414/416 | 416/416 |
| `exact_observed_pairs_online` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `exact_observed_pairs_online` | stochastic | 414/416 | 414/416 | 415/416 | 413/416 |
| `offline_online_stride1` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `offline_online_stride1` | stochastic | 415/416 | 414/416 | 415/416 | 413/416 |
| `offline_online_stride5` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `offline_online_stride5` | stochastic | 416/416 | 416/416 | 416/416 | 416/416 |
| `online_buffered_stride5` | greedy | 64/64 | 64/64 | 64/64 | 64/64 |
| `online_buffered_stride5` | stochastic | 340/416 | 345/416 | 336/416 | 344/416 |

The complete auditor receipt is `artifacts/logs/resume_audit/20261118T140229.316629Z-m5-terminal.json` (SHA-256 `6c0c4362d1e61c1c241163ca8155bc8043404a0cf6358001f816101cc090ce22`).

## Workload differences in the reference fastpath comparison

With prompt, arm, seed and repeat fixed, at least two of the three execution modes generated different output lengths in 393 of 960 comparisons (87/640 standard nonthinking and 306/320 long thinking). After the joint repetition filter, lengths still differed in 379 of 946 comparisons. Within a single mode, the two repeats generated different lengths in 6 of 1,440 prompt–arm–mode pairs, all in `offline_online_stride1` across five prompts. The benchmark retained lengths, mean accepted lengths and repetition flags. Its saved summaries omit the generated token IDs.

On long-thinking prompts, the mean compiled/eager throughput ratio for `offline_online_stride1` was 1.0356× (95% prompt-bootstrap interval 1.0008–1.0713, 31 prompts). Excluding the 4 long prompts with within-mode repeat differences gave 1.0334× (95% prompt-bootstrap interval 0.9953–1.0715, 27 prompts). The intervals describe whole requests under each mode. Variation in generated workloads limits what these ratios establish about compiler acceleration. The fixed-shape measurements are available in the [fastpath benchmark](../artifacts/post_m4_suite/fastpath_bench.json).

<!-- BEGIN POST_M4_M5_SUMMARY -->
---

## Post-M4 Ablations, Fastpath Benchmarks & Optional M5 Extensions (`scale=reference`)

Completed at `2026-11-18T14:02:00.466399+00:00` on commit `b92f0fbfc685ad92034e8da1f54072f7a18ae8be` via `edraft.post_m4_suite` without modifying any frozen M0–M4 reference artifact (`27` upstream M0–M4 files SHA256-verified intact; `197948` `audit_post_m4_suite.py` checks across `28937` files with `0` failures).

**Headline Post-M4 & M5 empirical findings:**
- **M2 offline refit and calibration** (`m2_refit_calibration`). Dev calibration selected dense β=0.35 and dense+sparse β=0.3. The dense+sparse grid used full-vocabulary KL on 120 of 102,975 dev slots. The dev-calibrated dense+sparse refit had held-out test grouped KL 1.191370. The lowest observed test grouped KL among the reported dense+sparse variants was 1.164154 (refit_shuf_wd_scale_0p20).
- **Zero-init online & count-gated bigram head-to-head (`online_head_to_head`):** evaluated `plain_dflash`, `online_bigram`, `online_bigram_count_gated` (`kappa=12.0`), `online_dbS_zero_init`, and `zero_offline_online_only` under `8-GPU` NUMA/core-pinned sharded timing (`sharded_8gpu`). All `5` `< 8`-prompt long-output offset bins are explicitly flagged **SPARSE / EMPTY (`conclusion unavailable`)**.
- **Optional M5 offline & closed-loop extensions (`m5_eval`):** capacity-matched `ExactObservedPairs` verified **`collision_count = 0`** (held-out test grouped KL `3.015552`, `Δ = 0.000065` vs `SignedHash` `3.015488`); `two_sided_pseudolikelihood` reached test grouped KL `6.172779`. Closed-loop candidate-masked lookahead (`lookahead_static_candidate`, `lookahead_rerelax_candidate`) shared exact frozen `C0[j]` (`beta_relax=1.0`) with causal `dense_candidate` and recorded positive `lookahead_iterations` CUDA-event durations on every active profile row; `online_buffered_stride5` accumulated all intermediate rounds (`acc + pend == blocks`) against single-round `offline_online_stride5` and `offline_online_stride1`.
- **Fastpath mode comparison (`fastpath_bench`).** Across the measured mix of prompts, mean paired whole-request throughput ratios relative to `eager_pre_round3` were `plain_dflash` `1.06x` compact / `1.02x` compile, `online_bigram` `0.95x` compact / `0.96x` compile, `offline_static` `1.00x` compact / `0.99x` compile, `offline_online_stride1` `1.08x` compact / `1.05x` compile, `online_dbS` `1.06x` compact / `1.06x` compile. The compact-mode estimate for 8 GPU ranks was 476.50 tok/s, using all generated tokens divided by the largest rank's summed request time (3736566 tokens, 7841.68 s). Separate per-arm calibration rates use a selected prompt subset and filter flagged long outputs within each run. The measured runner saved each fastpath rank only after all its requests finished. Recovery before the complete suite report could repeat all ranks and the subsequent single-GPU calibration.
- **Net closed-loop speed & unrun external baselines:** Across closed-loop M5 and online head-to-head suites, **no adapter or lookahead arm established a 95%-CI-supported net end-to-end `tokens/sec` speedup over `plain_dflash`** once proposal, lookahead, or online update latency is included. **External baselines still not run:** full tree-based Token Recycling, Test-Time Speculation, SharpDraft (`online_bigram` and `online_bigram_count_gated` remain smoothed committed-transition priors, not tree-based Token Recycling).

**Checksummed `post_m4_suite` artifact manifest:**
- `artifacts/post_m4_suite/m2_refit_calibration.json`: `3a311f848645c653b8f1f240d39d3f57c06c1b90a118b0c5e8769c5a758611dc`
- `artifacts/post_m4_suite/online_head_to_head.json`: `53d62acf3119ed039876ce6433eff2d8ecde3633814b43a00a320d984d6344a3`
- `artifacts/post_m4_suite/m5_eval.json`: `5d335fd2b8541d0b5b626ba9d042a27799e2311d2f9549cbf0ba32c89c1cae5c`
- `artifacts/post_m4_suite/fastpath_bench.json`: `6a9b401d80c6894aa2b1be777056e02429e16ae52ceb0205caf156e826e87b20`
- `artifacts/post_m4_suite/post_m4_suite_summary.json`: `f35cfe864b6ee716ce695c22f20c6cf160807d822789a4039c1c3404dbb8a04d`

> Generated strictly from verified `post_m4_suite` raw artifacts and sidecars (`0` manual transcription).

### 1. M2 Offline Refit & Post-Hoc Calibration (`m2_refit_calibration`)

- Status: **`complete`** | Dev-calibrated `beta*`: dense `0.35`, dense+sparse `0.3`

| Variant | Dense Dev KL | Dense Test KL | Sparse Dev KL | Sparse Test KL | CL Dense Acc (`Δ vs plain`) | CL Sparse Acc (`Δ vs plain`) | CL Sparse `tok/s` (`ratio`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_m2_unscaled` | 2.0181 | 2.8588 | 2.0602 | 3.0155 | 2.257 (`-2.041`) | 2.051 (`-2.262`) | 69.74 (`0.456x`) |
| `baseline_m2_scale_0p20` | 1.0893 | 1.1976 | 1.0936 | 1.1913 | 4.167 (`-0.153`) | 4.157 (`-0.141`) | 118.39 (`0.742x`) |
| `refit_shuf_wd_uncalibrated` | 1.5735 | 2.3298 | 1.5877 | 2.2224 | 2.493 (`-1.805`) | 2.581 (`-1.717`) | 82.63 (`0.535x`) |
| `refit_shuf_wd_dev_calibrated` | 1.0723 | 1.2500 | 1.0610 | 1.1914 | 3.978 (`-0.333`) | 4.124 (`-0.193`) | 117.64 (`0.739x`) |
| `refit_shuf_wd_scale_0p20` | 1.0721 | 1.1820 | 1.0662 | 1.1642 | 4.173 (`-0.124`) | 4.294 (`-0.004`) | 122.58 (`0.787x`) |

#### Suite 1 Closed-Loop Breakdown by Condition (`plain_dflash`: `std = 5.209` / `204.40 tok/s`; `long_thinking = 2.476` / `95.41 tok/s`)

| Variant | `std` Dense Acc (`tok/s`) | `std` Sparse Acc (`tok/s`) | `long_thinking` Dense Acc (`tok/s`) | `long_thinking` Sparse Acc (`tok/s`) |
| --- | ---: | ---: | ---: | ---: |
| `baseline_m2_unscaled` | 2.671 (`100.37`) | 2.325 (`78.73`) | 1.428 (`56.46`) | 1.486 (`51.18`) |
| `baseline_m2_scale_0p20` | 5.005 (`162.81`) | 5.062 (`142.18`) | 2.438 (`84.06`) | 2.346 (`70.82`) |
| `refit_shuf_wd_uncalibrated` | 2.936 (`106.83`) | 3.067 (`95.99`) | 1.606 (`65.75`) | 1.607 (`55.90`) |
| `refit_shuf_wd_dev_calibrated` | 4.779 (`156.78`) | 4.983 (`140.06`) | 2.326 (`82.43`) | 2.349 (`71.36`) |
| `refit_shuf_wd_scale_0p20` | 5.002 (`162.20`) | 5.157 (`144.28`) | 2.516 (`88.31`) | 2.568 (`79.18`) |

### 2. Online Zero-Init & Count-Gated Bigram Head-to-Head (`online_head_to_head`)

- Status: **`complete`** | `world_size=8` (`sharded_8gpu`) | `bigram_count_gate_kappa=12.0`

#### Overall Held-Out Acceptance & Sharded Matched-Bundle Throughput

| Condition | Arm | Primary / Total | Mean accepted len [95% CI] | Target fwds/tok [95% CI] | Sharded `tok/s` [95% CI] | Paired ratio vs `plain` [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram` | 32/32 | 2.524 [2.355, 2.704] | 0.2889 [0.2764, 0.3017] | 76.22 [70.86, 82.19] | 0.8768 [0.7883, 0.9745] |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_bigram_count_gated` | 32/32 | 2.522 [2.354, 2.696] | 0.2895 [0.2762, 0.3029] | 75.45 [69.96, 80.81] | 0.8564 [0.7810, 0.9340] |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_dbS_zero_init` | 32/32 | 2.579 [2.369, 2.806] | 0.2876 [0.2706, 0.3050] | 67.11 [61.80, 72.29] | 0.7606 [0.6877, 0.8340] |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 32/32 | 2.476 [2.289, 2.654] | 0.2944 [0.2792, 0.3115] | 92.90 [84.61, 101.13] | 1.0000 [1.0000, 1.0000] |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `zero_offline_online_only` | 31/32 | 3.029 [2.851, 3.234] | 0.2526 [0.2408, 0.2634] | 77.08 [72.48, 82.08] | 0.8728 [0.7876, 0.9550] |
| Standard non-thinking greedy (T=0) | `online_bigram` | 64/64 | 5.311 [5.081, 5.545] | 0.1665 [0.1605, 0.1729] | 184.26 [177.56, 191.35] | 0.8492 [0.8403, 0.8589] |
| Standard non-thinking greedy (T=0) | `online_bigram_count_gated` | 64/64 | 5.295 [5.066, 5.533] | 0.1670 [0.1610, 0.1735] | 183.34 [176.83, 190.31] | 0.8450 [0.8353, 0.8551] |
| Standard non-thinking greedy (T=0) | `online_dbS_zero_init` | 64/64 | 5.325 [5.090, 5.565] | 0.1663 [0.1601, 0.1729] | 166.66 [160.18, 173.39] | 0.7674 [0.7575, 0.7776] |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 64/64 | 5.285 [5.057, 5.519] | 0.1672 [0.1612, 0.1737] | 217.50 [209.29, 226.28] | 1.0000 [1.0000, 1.0000] |
| Standard non-thinking greedy (T=0) | `zero_offline_online_only` | 64/64 | 5.011 [4.784, 5.233] | 0.1747 [0.1680, 0.1821] | 145.08 [139.51, 150.90] | 0.6692 [0.6556, 0.6824] |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram` | 192/192 | 5.119 [4.915, 5.324] | 0.1723 [0.1662, 0.1789] | 157.96 [152.62, 163.42] | 0.7839 [0.7683, 0.7983] |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_bigram_count_gated` | 192/192 | 5.116 [4.905, 5.321] | 0.1724 [0.1661, 0.1790] | 157.97 [152.42, 163.42] | 0.7842 [0.7673, 0.7999] |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_dbS_zero_init` | 192/192 | 5.124 [4.905, 5.334] | 0.1725 [0.1663, 0.1793] | 136.23 [131.38, 141.05] | 0.6744 [0.6633, 0.6850] |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 192/192 | 5.136 [4.922, 5.346] | 0.1714 [0.1655, 0.1778] | 203.20 [195.98, 210.39] | 1.0000 [1.0000, 1.0000] |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `zero_offline_online_only` | 192/192 | 4.931 [4.703, 5.143] | 0.1782 [0.1712, 0.1858] | 128.52 [123.65, 133.46] | 0.6368 [0.6235, 0.6492] |
| Standard pure-T=1 diagnostic | `online_bigram` | 192/192 | 4.388 [4.190, 4.590] | 0.1959 [0.1887, 0.2037] | 163.55 [157.54, 169.79] | 0.9183 [0.8930, 0.9438] |
| Standard pure-T=1 diagnostic | `online_bigram_count_gated` | 192/192 | 4.392 [4.189, 4.594] | 0.1959 [0.1882, 0.2038] | 163.87 [157.58, 169.86] | 0.9199 [0.8949, 0.9456] |
| Standard pure-T=1 diagnostic | `online_dbS_zero_init` | 192/192 | 4.406 [4.217, 4.598] | 0.1953 [0.1882, 0.2026] | 136.61 [131.81, 141.45] | 0.7660 [0.7486, 0.7840] |
| Standard pure-T=1 diagnostic | `plain_dflash` | 192/192 | 4.297 [4.124, 4.477] | 0.1987 [0.1917, 0.2064] | 180.37 [174.63, 186.82] | 1.0000 [1.0000, 1.0000] |
| Standard pure-T=1 diagnostic | `zero_offline_online_only` | 192/192 | 4.157 [3.949, 4.352] | 0.2058 [0.1972, 0.2153] | 124.17 [119.01, 128.95] | 0.6950 [0.6773, 0.7116] |

#### Long-Output Offset-Bin Trajectory (`0-1K`..`16K+`, Sparsity Gate Enforced)

| Arm | Offset Bin | Reaching Reqs (`n`) | Accepted Len [95% CI] | Sharded Round `tok/s` [95% CI] | Gate Status |
| --- | --- | ---: | ---: | ---: | --- |
| `online_bigram` | `0-1K` | 32 | 2.793 [2.561, 3.030] | 94.39 [87.13, 102.00] | usable |
| `online_bigram` | `1K-4K` | 32 | 2.877 [2.596, 3.211] | 96.46 [88.16, 105.61] | usable |
| `online_bigram` | `4K-8K` | 32 | 2.777 [2.502, 3.105] | 86.72 [79.33, 96.06] | usable |
| `online_bigram` | `8K-16K` | 29 | 1.974 [1.736, 2.232] | 54.10 [49.55, 58.80] | usable |
| `online_bigram` | `16K+` | 2 | 1.794 [1.556, 2.033] | 31.66 [30.06, 33.27] | **SPARSE (`n=2 < 8`; conclusion unavailable)** |
| `online_bigram_count_gated` | `0-1K` | 32 | 2.761 [2.535, 2.993] | 93.96 [86.64, 101.44] | usable |
| `online_bigram_count_gated` | `1K-4K` | 32 | 2.796 [2.511, 3.144] | 94.46 [85.62, 105.07] | usable |
| `online_bigram_count_gated` | `4K-8K` | 32 | 2.697 [2.479, 2.920] | 84.47 [78.36, 90.34] | usable |
| `online_bigram_count_gated` | `8K-16K` | 27 | 2.207 [1.903, 2.641] | 58.15 [51.09, 68.39] | usable |
| `online_bigram_count_gated` | `16K+` | 2 | 3.721 [1.507, 5.935] | 50.04 [28.01, 72.06] | **SPARSE (`n=2 < 8`; conclusion unavailable)** |
| `online_dbS_zero_init` | `0-1K` | 32 | 2.867 [2.603, 3.129] | 81.91 [75.19, 88.52] | usable |
| `online_dbS_zero_init` | `1K-4K` | 32 | 2.958 [2.658, 3.317] | 84.17 [76.69, 92.90] | usable |
| `online_dbS_zero_init` | `4K-8K` | 30 | 2.803 [2.581, 3.048] | 76.02 [71.39, 80.74] | usable |
| `online_dbS_zero_init` | `8K-16K` | 23 | 2.055 [1.681, 2.500] | 48.64 [43.02, 54.66] | usable |
| `online_dbS_zero_init` | `16K+` | 6 | 1.758 [0.971, 3.194] | 29.14 [21.61, 42.72] | **SPARSE (`n=6 < 8`; conclusion unavailable)** |
| `plain_dflash` | `0-1K` | 32 | 2.863 [2.629, 3.101] | 121.56 [112.51, 130.43] | usable |
| `plain_dflash` | `1K-4K` | 32 | 2.750 [2.494, 3.084] | 118.45 [107.93, 131.53] | usable |
| `plain_dflash` | `4K-8K` | 32 | 2.879 [2.568, 3.239] | 110.36 [100.54, 120.95] | usable |
| `plain_dflash` | `8K-16K` | 25 | 1.826 [1.642, 2.008] | 59.77 [55.22, 64.45] | usable |
| `plain_dflash` | `16K+` | 6 | 1.431 [1.031, 1.816] | 33.97 [28.17, 39.24] | **SPARSE (`n=6 < 8`; conclusion unavailable)** |
| `zero_offline_online_only` | `0-1K` | 31 | 3.018 [2.717, 3.349] | 84.03 [76.45, 91.90] | usable |
| `zero_offline_online_only` | `1K-4K` | 31 | 3.042 [2.814, 3.297] | 84.81 [78.80, 91.55] | usable |
| `zero_offline_online_only` | `4K-8K` | 28 | 3.147 [2.863, 3.462] | 81.60 [75.21, 88.55] | usable |
| `zero_offline_online_only` | `8K-16K` | 23 | 2.909 [2.418, 3.535] | 63.66 [55.07, 74.93] | usable |
| `zero_offline_online_only` | `16K+` | 2 | 7.069 [1.240, 12.898] | 82.77 [28.14, 137.40] | **SPARSE (`n=2 < 8`; conclusion unavailable)** |

#### Paired Long-Context Head-to-Head Horizon Crossover (`0-1K`..`16K+`, Sparsity Gate Enforced)

| Pair (`arm` vs `ref`) | Offset Bin | Paired Reqs (`n`) | Paired `ΔE[τ]` [95% CI] | Paired Sharded `tok/s` Ratio [95% CI] | Gate Status |
| --- | --- | ---: | ---: | ---: | --- |
| `online_bigram_count_gated` vs `plain_dflash` | `0-1K` | 32 | -0.1026 [-0.3317, 0.1331] | 0.7913 [0.7328, 0.8535] | usable (`n_tps=32`) |
| `online_bigram_count_gated` vs `plain_dflash` | `1K-4K` | 32 | 0.0455 [-0.1216, 0.2320] | 0.8085 [0.7647, 0.8544] | usable (`n_tps=32`) |
| `online_bigram_count_gated` vs `plain_dflash` | `4K-8K` | 32 | -0.1816 [-0.5954, 0.1771] | 0.8009 [0.7363, 0.8626] | usable (`n_tps=32`) |
| `online_bigram_count_gated` vs `plain_dflash` | `8K-16K` | 21 | 0.2325 [-0.0180, 0.4764] | 0.9442 [0.8451, 1.0573] | usable (`n_tps=21`) |
| `online_bigram_count_gated` vs `plain_dflash` | `16K+` | 0 | — | — | **EMPTY (conclusion unavailable)** |
| `zero_offline_online_only` vs `plain_dflash` | `0-1K` | 31 | 0.1712 [-0.0537, 0.4211] | 0.7018 [0.6614, 0.7473] | usable (`n_tps=31`) |
| `zero_offline_online_only` vs `plain_dflash` | `1K-4K` | 31 | 0.2867 [0.0945, 0.4867] | 0.7370 [0.6964, 0.7840] | usable (`n_tps=31`) |
| `zero_offline_online_only` vs `plain_dflash` | `4K-8K` | 28 | 0.2774 [-0.1713, 0.7055] | 0.7738 [0.7031, 0.8460] | usable (`n_tps=28`) |
| `zero_offline_online_only` vs `plain_dflash` | `8K-16K` | 20 | 1.1617 [0.4787, 1.9615] | 1.1571 [0.9343, 1.4420] | usable (`n_tps=20`) |
| `zero_offline_online_only` vs `plain_dflash` | `16K+` | 1 | 0.1074 [0.1074, 0.1074] | 0.8025 [0.8025, 0.8025] | **SPARSE (`n=1 < 8`; conclusion unavailable)** |
| `online_dbS_zero_init` vs `plain_dflash` | `0-1K` | 32 | 0.0038 [-0.1495, 0.1586] | 0.6856 [0.6487, 0.7262] | usable (`n_tps=32`) |
| `online_dbS_zero_init` vs `plain_dflash` | `1K-4K` | 32 | 0.2074 [0.0148, 0.4206] | 0.7275 [0.6846, 0.7760] | usable (`n_tps=32`) |
| `online_dbS_zero_init` vs `plain_dflash` | `4K-8K` | 30 | -0.0783 [-0.4847, 0.2857] | 0.7164 [0.6648, 0.7723] | usable (`n_tps=30`) |
| `online_dbS_zero_init` vs `plain_dflash` | `8K-16K` | 20 | 0.3060 [-0.0947, 0.7568] | 0.8645 [0.7471, 0.9839] | usable (`n_tps=20`) |
| `online_dbS_zero_init` vs `plain_dflash` | `16K+` | 3 | 0.8457 [-0.7374, 3.4194] | 0.9745 [0.6234, 1.6340] | **SPARSE (`n=3 < 8`; conclusion unavailable)** |
| `online_bigram_count_gated` vs `online_bigram` | `0-1K` | 32 | -0.0324 [-0.1740, 0.1145] | 1.0067 [0.9544, 1.0617] | usable (`n_tps=32`) |
| `online_bigram_count_gated` vs `online_bigram` | `1K-4K` | 32 | -0.0808 [-0.2329, 0.0727] | 0.9840 [0.9375, 1.0305] | usable (`n_tps=32`) |
| `online_bigram_count_gated` vs `online_bigram` | `4K-8K` | 32 | -0.0803 [-0.4475, 0.2159] | 1.0092 [0.9380, 1.0826] | usable (`n_tps=32`) |
| `online_bigram_count_gated` vs `online_bigram` | `8K-16K` | 24 | 0.1611 [-0.2702, 0.6572] | 1.0852 [0.9225, 1.2756] | usable (`n_tps=24`) |
| `online_bigram_count_gated` vs `online_bigram` | `16K+` | 0 | — | — | **EMPTY (conclusion unavailable)** |
| `zero_offline_online_only` vs `online_dbS_zero_init` | `0-1K` | 31 | 0.1693 [-0.0130, 0.3695] | 1.0340 [0.9836, 1.0867] | usable (`n_tps=31`) |
| `zero_offline_online_only` vs `online_dbS_zero_init` | `1K-4K` | 31 | 0.0731 [-0.1916, 0.3063] | 1.0261 [0.9690, 1.0819] | usable (`n_tps=31`) |
| `zero_offline_online_only` vs `online_dbS_zero_init` | `4K-8K` | 28 | 0.3521 [0.0518, 0.6904] | 1.0897 [1.0142, 1.1772] | usable (`n_tps=28`) |
| `zero_offline_online_only` vs `online_dbS_zero_init` | `8K-16K` | 18 | 0.7867 [0.1038, 1.6272] | 1.3489 [1.1379, 1.6172] | usable (`n_tps=18`) |
| `zero_offline_online_only` vs `online_dbS_zero_init` | `16K+` | 1 | 0.2524 [0.2524, 0.2524] | 1.2158 [1.2158, 1.2158] | **SPARSE (`n=1 < 8`; conclusion unavailable)** |

### 3. Optional M5 Extensions (`m5_eval`)

- Status: **`complete`** | `world_size=8` (`sharded_8gpu`) | Matched `C0[j]` `beta_relax = 1.0` | `ExactObservedPairs` `collision_count = 0`

#### M5 Offline Objectives & Collision-Free Exact Sparse Table vs Signed Hash

| Offline Checkpoint | Sparse Backend | Collisions | Dev Grouped KL | Test Grouped KL | `Δ` Test KL vs SignedHash |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline_signed_hash` | `signed_feature_hash` | 88666459 | 2.060249 | 3.015488 | 0.000000 |
| `two_sided_pseudolikelihood` | `signed_feature_hash` | 132989841 | 3.586506 | 6.172779 | — |
| `exact_observed_pairs` (`freeze_new`) | `exact_observed_pairs` | **0** | 2.060275 | 3.015552 | 0.000065 |

#### M5 Closed-Loop Acceptance, Throughput & Lookahead Phase Duration

| Condition | Arm | Primary / Total | Mean accepted len [95% CI] | Sharded `tok/s` [95% CI] | Paired ratio vs `plain` [95% CI] | `lookahead_iterations` med ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `dense_candidate` | 32/32 | 1.980 [1.889, 2.066] | 61.15 [57.60, 64.40] | 0.6673 [0.6106, 0.7342] | 0.00 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `exact_observed_pairs_online` | 32/32 | 2.465 [2.322, 2.624] | 62.45 [59.87, 65.38] | 0.6931 [0.6308, 0.7602] | 0.00 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `lookahead_rerelax_candidate` | 32/32 | 2.257 [2.090, 2.427] | 64.33 [60.00, 69.16] | 0.6998 [0.6382, 0.7644] | 32882.38 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `lookahead_static_candidate` | 32/32 | 2.230 [2.083, 2.384] | 74.91 [69.72, 80.16] | 0.8200 [0.7404, 0.9091] | 7486.34 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_online_stride1` | 32/32 | 2.492 [2.342, 2.660] | 70.26 [67.27, 73.45] | 0.7836 [0.7142, 0.8601] | 0.00 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `offline_online_stride5` | 32/32 | 2.044 [1.914, 2.174] | 61.47 [58.75, 64.33] | 0.6755 [0.6163, 0.7427] | 0.00 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_buffered_stride5` | 30/32 | 1.661 [1.554, 1.779] | 51.54 [48.99, 54.12] | 0.5663 [0.5175, 0.6218] | 0.00 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_two_sided_pl` | 32/32 | 2.141 [2.022, 2.271] | 48.69 [47.04, 50.35] | 0.5412 [0.4913, 0.5949] | 0.00 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `plain_dflash` | 32/32 | 2.476 [2.289, 2.654] | 96.80 [88.59, 105.01] | 1.0000 [1.0000, 1.0000] | 0.00 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `signed_feature_hash_online` | 32/32 | 2.495 [2.345, 2.662] | 67.30 [64.01, 71.06] | 0.7438 [0.6747, 0.8185] | 0.00 |
| Standard non-thinking greedy (T=0) | `dense_candidate` | 64/64 | 2.284 [2.146, 2.418] | 91.47 [87.58, 95.46] | 0.4265 [0.4147, 0.4379] | 0.00 |
| Standard non-thinking greedy (T=0) | `exact_observed_pairs_online` | 64/64 | 3.196 [3.043, 3.350] | 90.26 [87.03, 93.53] | 0.4214 [0.4133, 0.4294] | 0.00 |
| Standard non-thinking greedy (T=0) | `lookahead_rerelax_candidate` | 64/64 | 2.298 [2.164, 2.425] | 82.74 [79.22, 86.18] | 0.3860 [0.3751, 0.3962] | 873.29 |
| Standard non-thinking greedy (T=0) | `lookahead_static_candidate` | 64/64 | 2.522 [2.367, 2.667] | 106.20 [101.53, 110.79] | 0.4950 [0.4813, 0.5081] | 223.25 |
| Standard non-thinking greedy (T=0) | `offline_online_stride1` | 64/64 | 3.196 [3.043, 3.350] | 99.67 [96.01, 103.42] | 0.4650 [0.4565, 0.4736] | 0.00 |
| Standard non-thinking greedy (T=0) | `offline_online_stride5` | 64/64 | 2.744 [2.595, 2.886] | 93.39 [89.46, 97.12] | 0.4349 [0.4250, 0.4443] | 0.00 |
| Standard non-thinking greedy (T=0) | `online_buffered_stride5` | 64/64 | 1.498 [1.354, 1.643] | 60.32 [56.74, 63.96] | 0.2800 [0.2679, 0.2914] | 0.00 |
| Standard non-thinking greedy (T=0) | `online_two_sided_pl` | 64/64 | 2.862 [2.718, 3.009] | 67.31 [65.39, 69.28] | 0.3155 [0.3096, 0.3215] | 0.00 |
| Standard non-thinking greedy (T=0) | `plain_dflash` | 64/64 | 5.285 [5.057, 5.519] | 215.41 [207.06, 223.89] | 1.0000 [1.0000, 1.0000] | 0.00 |
| Standard non-thinking greedy (T=0) | `signed_feature_hash_online` | 64/64 | 3.196 [3.043, 3.350] | 98.96 [95.25, 102.68] | 0.4617 [0.4525, 0.4709] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `dense_candidate` | 192/192 | 4.049 [3.859, 4.234] | 117.16 [112.86, 121.36] | 0.5840 [0.5722, 0.5966] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `exact_observed_pairs_online` | 192/192 | 3.190 [3.042, 3.334] | 81.42 [78.53, 84.16] | 0.4062 [0.3977, 0.4144] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `lookahead_rerelax_candidate` | 192/192 | 3.942 [3.779, 4.104] | 106.23 [102.63, 109.70] | 0.5313 [0.5201, 0.5432] | 614.98 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `lookahead_static_candidate` | 192/192 | 4.086 [3.910, 4.263] | 133.45 [128.94, 138.11] | 0.6670 [0.6514, 0.6826] | 155.53 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_online_stride1` | 192/192 | 3.190 [3.042, 3.334] | 88.88 [85.72, 91.87] | 0.4434 [0.4340, 0.4524] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `offline_online_stride5` | 192/192 | 2.777 [2.640, 2.911] | 83.91 [80.88, 86.83] | 0.4188 [0.4090, 0.4287] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_buffered_stride5` | 192/192 | 1.476 [1.336, 1.620] | 53.34 [50.27, 56.51] | 0.2647 [0.2539, 0.2752] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_two_sided_pl` | 192/192 | 2.871 [2.733, 3.004] | 61.95 [60.25, 63.57] | 0.3103 [0.3041, 0.3170] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `plain_dflash` | 192/192 | 5.136 [4.922, 5.346] | 201.96 [194.87, 208.76] | 1.0000 [1.0000, 1.0000] | 0.00 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `signed_feature_hash_online` | 192/192 | 3.190 [3.042, 3.334] | 88.80 [85.62, 91.80] | 0.4430 [0.4335, 0.4520] | 0.00 |
| Standard pure-T=1 diagnostic | `dense_candidate` | 192/192 | 3.699 [3.533, 3.859] | 121.01 [116.95, 125.05] | 0.6859 [0.6674, 0.7050] | 0.00 |
| Standard pure-T=1 diagnostic | `exact_observed_pairs_online` | 192/192 | 2.877 [2.742, 3.010] | 83.14 [80.31, 86.08] | 0.4704 [0.4595, 0.4817] | 0.00 |
| Standard pure-T=1 diagnostic | `lookahead_rerelax_candidate` | 192/192 | 3.721 [3.545, 3.901] | 111.04 [106.91, 115.24] | 0.6275 [0.6114, 0.6444] | 642.19 |
| Standard pure-T=1 diagnostic | `lookahead_static_candidate` | 192/192 | 3.800 [3.624, 3.974] | 136.04 [131.21, 141.00] | 0.7677 [0.7512, 0.7842] | 170.39 |
| Standard pure-T=1 diagnostic | `offline_online_stride1` | 192/192 | 2.877 [2.742, 3.010] | 91.32 [88.20, 94.54] | 0.5168 [0.5047, 0.5294] | 0.00 |
| Standard pure-T=1 diagnostic | `offline_online_stride5` | 192/192 | 2.522 [2.413, 2.631] | 87.21 [84.46, 89.92] | 0.4954 [0.4834, 0.5079] | 0.00 |
| Standard pure-T=1 diagnostic | `online_buffered_stride5` | 192/192 | 1.365 [1.239, 1.484] | 56.54 [53.61, 59.46] | 0.3194 [0.3057, 0.3328] | 0.00 |
| Standard pure-T=1 diagnostic | `online_two_sided_pl` | 192/192 | 2.626 [2.508, 2.743] | 63.84 [62.23, 65.39] | 0.3629 [0.3549, 0.3707] | 0.00 |
| Standard pure-T=1 diagnostic | `plain_dflash` | 192/192 | 4.297 [4.124, 4.477] | 179.03 [173.19, 185.31] | 1.0000 [1.0000, 1.0000] | 0.00 |
| Standard pure-T=1 diagnostic | `signed_feature_hash_online` | 192/192 | 2.877 [2.742, 3.010] | 91.19 [88.09, 94.31] | 0.5158 [0.5043, 0.5280] | 0.00 |

#### M5 Matched-Support Paired Comparisons (`C0[j]`, Exact vs SignedHash, Buffered Stride-5)

| Condition | Arm vs Reference | Support Family | Pairs (`rep_excl`) | `Δ` Accepted Len [95% CI] | Timing `tok/s` Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `lookahead_static_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 32 (0) | 0.2501 [0.1061, 0.3980] | 1.2252 |
| Standard non-thinking greedy (T=0) | `lookahead_static_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 64 (0) | 0.2385 [0.1948, 0.2794] | 1.1611 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `lookahead_static_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 192 (0) | 0.0367 [-0.0735, 0.1409] | 1.1391 |
| Standard pure-T=1 diagnostic | `lookahead_static_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 192 (0) | 0.1001 [-0.0086, 0.2077] | 1.1242 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `lookahead_rerelax_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 32 (0) | 0.2767 [0.1097, 0.4410] | 1.0521 |
| Standard non-thinking greedy (T=0) | `lookahead_rerelax_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 64 (0) | 0.0144 [-0.0324, 0.0561] | 0.9046 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `lookahead_rerelax_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 192 (0) | -0.1067 [-0.2107, -0.0028] | 0.9067 |
| Standard pure-T=1 diagnostic | `lookahead_rerelax_candidate` vs `dense_candidate` | `frozen_C0_candidate_masked` | 192 (0) | 0.0213 [-0.1004, 0.1428] | 0.9176 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `exact_observed_pairs_online` vs `signed_feature_hash_online` | `online_sparse_representation` | 32 (0) | -0.0301 [-0.0564, -0.0020] | 0.9279 |
| Standard non-thinking greedy (T=0) | `exact_observed_pairs_online` vs `signed_feature_hash_online` | `online_sparse_representation` | 64 (0) | 0.0000 [0.0000, 0.0000] | 0.9121 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `exact_observed_pairs_online` vs `signed_feature_hash_online` | `online_sparse_representation` | 192 (0) | 0.0000 [0.0000, 0.0000] | 0.9169 |
| Standard pure-T=1 diagnostic | `exact_observed_pairs_online` vs `signed_feature_hash_online` | `online_sparse_representation` | 192 (0) | 0.0000 [0.0000, 0.0000] | 0.9118 |
| Long thinking (T=0.6, top-k=20, top-p=0.95) | `online_buffered_stride5` vs `offline_online_stride5` | `online_stride5_update_mode` | 30 (2) | -0.4041 [-0.5154, -0.2921] | 0.8385 |
| Standard non-thinking greedy (T=0) | `online_buffered_stride5` vs `offline_online_stride5` | `online_stride5_update_mode` | 64 (0) | -1.2462 [-1.3531, -1.1407] | 0.6459 |
| Standard recommended (T=0.7, top-k=20, top-p=0.8) | `online_buffered_stride5` vs `offline_online_stride5` | `online_stride5_update_mode` | 192 (0) | -1.3011 [-1.3911, -1.2159] | 0.6357 |
| Standard pure-T=1 diagnostic | `online_buffered_stride5` vs `offline_online_stride5` | `online_stride5_update_mode` | 192 (0) | -1.1574 [-1.2322, -1.0852] | 0.6484 |

#### M5 Long-Output Offset-Bin Trajectory (`0-1K`..`16K+`, Sparsity Gate Enforced)

| Arm | Offset Bin | Reaching Reqs (`n`) | Accepted Len [95% CI] | Sharded Round `tok/s` [95% CI] | Gate Status |
| --- | --- | ---: | ---: | ---: | --- |
| `dense_candidate` | `0-1K` | 32 | 2.331 [2.115, 2.539] | 77.06 [71.48, 82.58] | usable |
| `dense_candidate` | `1K-4K` | 32 | 2.233 [2.055, 2.423] | 74.78 [69.91, 79.74] | usable |
| `dense_candidate` | `4K-8K` | 32 | 2.206 [2.108, 2.305] | 69.58 [66.67, 72.36] | usable |
| `dense_candidate` | `8K-16K` | 28 | 1.392 [1.242, 1.532] | 42.10 [39.40, 44.92] | usable |
| `dense_candidate` | `16K+` | 3 | 1.623 [1.134, 2.507] | 29.45 [24.29, 39.02] | **SPARSE (`n=3 < 8`; conclusion unavailable)** |
| `exact_observed_pairs_online` | `0-1K` | 32 | 2.017 [1.855, 2.190] | 59.09 [55.80, 62.55] | usable |
| `exact_observed_pairs_online` | `1K-4K` | 32 | 2.428 [2.227, 2.662] | 67.21 [63.08, 72.03] | usable |
| `exact_observed_pairs_online` | `4K-8K` | 32 | 2.994 [2.731, 3.293] | 74.68 [69.31, 80.64] | usable |
| `exact_observed_pairs_online` | `8K-16K` | 27 | 2.123 [1.810, 2.424] | 48.59 [44.51, 52.60] | usable |
| `exact_observed_pairs_online` | `16K+` | 3 | 1.662 [1.472, 1.834] | 25.99 [19.08, 32.78] | **SPARSE (`n=3 < 8`; conclusion unavailable)** |
| `lookahead_rerelax_candidate` | `0-1K` | 32 | 2.234 [2.007, 2.466] | 69.51 [64.28, 75.03] | usable |
| `lookahead_rerelax_candidate` | `1K-4K` | 32 | 2.490 [2.236, 2.822] | 75.13 [68.90, 82.96] | usable |
| `lookahead_rerelax_candidate` | `4K-8K` | 32 | 2.569 [2.263, 2.953] | 72.87 [65.77, 82.13] | usable |
| `lookahead_rerelax_candidate` | `8K-16K` | 27 | 1.744 [1.482, 2.044] | 46.41 [41.87, 52.01] | usable |
| `lookahead_rerelax_candidate` | `16K+` | 2 | 2.743 [1.733, 3.753] | 40.08 [33.18, 46.98] | **SPARSE (`n=2 < 8`; conclusion unavailable)** |
| `lookahead_static_candidate` | `0-1K` | 32 | 2.413 [2.212, 2.634] | 88.41 [82.37, 94.74] | usable |
| `lookahead_static_candidate` | `1K-4K` | 32 | 2.300 [2.119, 2.503] | 86.12 [80.78, 91.91] | usable |
| `lookahead_static_candidate` | `4K-8K` | 31 | 2.515 [2.313, 2.722] | 85.57 [79.44, 91.58] | usable |
| `lookahead_static_candidate` | `8K-16K` | 26 | 1.794 [1.437, 2.162] | 54.42 [47.26, 61.89] | usable |
| `lookahead_static_candidate` | `16K+` | 3 | 1.548 [1.034, 2.463] | 31.09 [25.40, 37.27] | **SPARSE (`n=3 < 8`; conclusion unavailable)** |
| `offline_online_stride1` | `0-1K` | 32 | 2.017 [1.854, 2.190] | 65.44 [61.52, 69.51] | usable |
| `offline_online_stride1` | `1K-4K` | 32 | 2.418 [2.217, 2.658] | 74.68 [70.10, 80.11] | usable |
| `offline_online_stride1` | `4K-8K` | 32 | 2.939 [2.691, 3.215] | 81.50 [76.20, 87.26] | usable |
| `offline_online_stride1` | `8K-16K` | 27 | 2.371 [1.950, 2.840] | 58.07 [51.14, 66.04] | usable |
| `offline_online_stride1` | `16K+` | 1 | 1.807 [1.807, 1.807] | 37.61 [37.61, 37.61] | **SPARSE (`n=1 < 8`; conclusion unavailable)** |
| `offline_online_stride5` | `0-1K` | 32 | 1.771 [1.595, 1.942] | 60.56 [56.44, 64.69] | usable |
| `offline_online_stride5` | `1K-4K` | 32 | 2.192 [1.991, 2.420] | 70.11 [65.06, 75.80] | usable |
| `offline_online_stride5` | `4K-8K` | 31 | 2.257 [2.049, 2.479] | 67.65 [63.22, 72.61] | usable |
| `offline_online_stride5` | `8K-16K` | 26 | 1.871 [1.548, 2.249] | 49.62 [44.39, 55.54] | usable |
| `offline_online_stride5` | `16K+` | 2 | 1.512 [0.440, 2.584] | 31.48 [16.96, 46.00] | **SPARSE (`n=2 < 8`; conclusion unavailable)** |
| `online_buffered_stride5` | `0-1K` | 30 | 1.661 [1.494, 1.841] | 56.79 [53.49, 60.10] | usable |
| `online_buffered_stride5` | `1K-4K` | 30 | 1.872 [1.657, 2.141] | 63.59 [57.21, 70.81] | usable |
| `online_buffered_stride5` | `4K-8K` | 30 | 1.914 [1.731, 2.130] | 58.34 [54.39, 62.26] | usable |
| `online_buffered_stride5` | `8K-16K` | 24 | 1.327 [1.065, 1.644] | 38.79 [35.46, 42.05] | usable |
| `online_buffered_stride5` | `16K+` | 4 | 1.440 [1.163, 1.616] | 22.28 [20.08, 24.60] | **SPARSE (`n=4 < 8`; conclusion unavailable)** |
| `online_two_sided_pl` | `0-1K` | 32 | 1.915 [1.707, 2.142] | 49.23 [46.18, 52.44] | usable |
| `online_two_sided_pl` | `1K-4K` | 32 | 2.239 [2.044, 2.440] | 53.68 [50.74, 56.71] | usable |
| `online_two_sided_pl` | `4K-8K` | 32 | 2.370 [2.144, 2.607] | 52.85 [49.99, 55.84] | usable |
| `online_two_sided_pl` | `8K-16K` | 24 | 1.963 [1.701, 2.224] | 40.56 [37.42, 43.47] | usable |
| `online_two_sided_pl` | `16K+` | 1 | 1.986 [1.986, 1.986] | 27.20 [27.20, 27.20] | **SPARSE (`n=1 < 8`; conclusion unavailable)** |
| `plain_dflash` | `0-1K` | 32 | 2.863 [2.629, 3.101] | 126.14 [118.28, 134.09] | usable |
| `plain_dflash` | `1K-4K` | 32 | 2.750 [2.494, 3.084] | 123.31 [113.27, 135.38] | usable |
| `plain_dflash` | `4K-8K` | 32 | 2.879 [2.568, 3.239] | 116.36 [106.06, 127.99] | usable |
| `plain_dflash` | `8K-16K` | 25 | 1.826 [1.642, 2.008] | 63.01 [58.47, 67.46] | usable |
| `plain_dflash` | `16K+` | 6 | 1.431 [1.031, 1.816] | 35.91 [29.22, 43.03] | **SPARSE (`n=6 < 8`; conclusion unavailable)** |
| `signed_feature_hash_online` | `0-1K` | 32 | 2.017 [1.854, 2.190] | 63.67 [59.77, 67.91] | usable |
| `signed_feature_hash_online` | `1K-4K` | 32 | 2.418 [2.217, 2.658] | 72.21 [67.03, 77.79] | usable |
| `signed_feature_hash_online` | `4K-8K` | 32 | 2.940 [2.694, 3.216] | 78.38 [73.36, 84.43] | usable |
| `signed_feature_hash_online` | `8K-16K` | 27 | 2.383 [1.959, 2.861] | 55.85 [48.90, 64.30] | usable |
| `signed_feature_hash_online` | `16K+` | 2 | 1.753 [1.698, 1.807] | 31.57 [28.70, 34.44] | **SPARSE (`n=2 < 8`; conclusion unavailable)** |

### 4. Fastpath & Multi-GPU Scaling Benchmark (`fastpath_bench`)

- Status: **`complete`** | Device: `cuda:0` | Assigned device indices: `[0, 1, 2, 3, 4, 5, 6, 7]`

#### Matched whole-request throughput under eager, compact and compile modes

The three mode means use each mode's eligible requests. Ratios average matched prompt/repeat observations and exclude a long match when any mode flags repetition.
The frozen fastpath runner used the primary positive-temperature settings and saved scalar request summaries and producer repetition flags. These rows omit the output tokens needed for token parity or independent repetition detection. Separate M5 evaluation and timing records retain output tokens and verification blocks for trajectory and repetition checks.

| Arm | Matched pairs (excluded) | Eager mean tok/s | Compact mean tok/s (paired ratio to eager) | Compile mean tok/s (paired ratio to eager) |
| --- | ---: | ---: | ---: | ---: |
| `plain_dflash` | 188 (4) | 166.23 | 167.67 (**`1.062x`**) | 165.86 (**`1.020x`**) |
| `online_bigram` | 190 (2) | 136.33 | 128.68 (**`0.947x`**) | 129.82 (**`0.963x`**) |
| `offline_static` | 190 (2) | 102.73 | 102.74 (**`1.003x`**) | 101.64 (**`0.994x`**) |
| `offline_online_stride1` | 190 (2) | 77.47 | 83.50 (**`1.079x`**) | 81.72 (**`1.052x`**) |
| `online_dbS` | 188 (4) | 60.16 | 63.55 (**`1.055x`**) | 63.36 (**`1.057x`**) |

#### Paired throughput by generation condition

Each ratio averages matched prompt/repeat measurements within one condition. The table above pools the measured workload. The reference workload uses one sampling seed and two measured repeats per prompt.

| Condition | Arm | Eligible pairs | Compact / eager | Compile / compact | Compile / eager |
| --- | --- | ---: | ---: | ---: | ---: |
| `long_thinking` | `plain_dflash` | 60 | 1.217 | 0.977 | 1.099 |
| `standard_nonthinking` | `plain_dflash` | 128 | 0.989 | 0.997 | 0.982 |
| `long_thinking` | `online_bigram` | 62 | 0.934 | 1.073 | 0.987 |
| `standard_nonthinking` | `online_bigram` | 128 | 0.953 | 0.998 | 0.952 |
| `long_thinking` | `offline_static` | 62 | 1.010 | 0.961 | 0.968 |
| `standard_nonthinking` | `offline_static` | 128 | 0.999 | 1.008 | 1.007 |
| `long_thinking` | `offline_online_stride1` | 62 | 1.092 | 0.951 | 1.036 |
| `standard_nonthinking` | `offline_online_stride1` | 128 | 1.073 | 0.988 | 1.060 |
| `long_thinking` | `online_dbS` | 60 | 1.046 | 1.045 | 1.088 |
| `standard_nonthinking` | `online_dbS` | 128 | 1.060 | 0.984 | 1.043 |

#### Single-rank and 8-rank calibration in compact mode

Across 8 ranks, all compact-mode requests produced 3736566 tokens. Dividing this count by the largest rank's summed request time (7841.68 s) gives an estimated **476.50 tok/s**. This aggregate includes long requests flagged for repetition.
The per-arm calibration rates use a selected prompt subset and exclude flagged long requests within each dataset. The final column multiplies the sharded/single-rank rate ratio by the rank count.

| Arm | Single-rank calibration tok/s | Sharded calibration tok/s per rank | Sharded / single-rank | Sum of rank means tok/s | Estimated cluster / single-rank ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `plain_dflash` | 136.12 | 141.80 | 1.0417 | 1341.32 | `8.33x` |
| `online_bigram` | 111.94 | 115.41 | 1.0309 | 1029.48 | `8.25x` |
| `offline_static` | 88.76 | 89.48 | 1.0081 | 821.85 | `8.06x` |
| `offline_online_stride1` | 73.37 | 74.28 | 1.0125 | 667.98 | `8.10x` |
| `online_dbS` | 56.31 | 56.36 | 1.0008 | 508.27 | `8.01x` |
<!-- END POST_M4_M5_SUMMARY -->
