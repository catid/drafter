# Causal energy adapters for DFlash

This repository tests whether a small learned energy correction improves a parallel
DFlash drafter after its scoring, sampling, verification, and update costs. The full
experiment specification is [prompt.md](prompt.md). M0 and M1-pre have passed at
reference scale; M1a collection is running. Reference M1b and M2–M4 have not run.
No adapter improvement or end-to-end speedup claim is established. See
[the research summary](reports/SUMMARY.md) and
[milestone status](artifacts/milestones.json) for the recorded evidence.

The supported reference pair is `Qwen/Qwen3-8B` with
`z-lab/Qwen3-8B-DFlash-b16`, using Hugging Face Transformers. Source and dependencies
are pinned by the DFlash Git submodule and `uv.lock`. The optional 4B pair is configurable;
Qwen3.5 and server backends are outside M0–M4.

## Setup and execution

```bash
git clone --recurse-submodules git@github.com:catid/drafter.git
cd drafter
uv sync --frozen --python 3.12
uv run --frozen python -m edraft.smoke --scale smoke
uv run --frozen pytest
```

Smoke uses only built-in CPU models, requires no model or dataset downloads, and
does not establish reference milestone acceptance. Without `--allow_download`, all
model loading is cache-only. Full scale requires `--allow_large_run`. Downloads,
revisions, checksums where available, and disk use are recorded in
`artifacts/environment.json`. Runtime notes and exact sampling sources are in
[NOTES.md](NOTES.md). Generated results identify their scale.

The remaining CPU smoke entrypoints can be exercised in a separate output
directory. Collection uses built-in synthetic prompts; training and evaluation
create their own small synthetic fitting fixtures. These commands do not consume
or overwrite the reference calibration and checkpoints.

```bash
uv run --frozen python -m edraft.collect --scale smoke --stage calibrate --output artifacts/smoke_pipeline
uv run --frozen python -m edraft.collect --scale smoke --stage collect --output artifacts/smoke_pipeline
uv run --frozen python -m edraft.collect --scale smoke --stage audit --output artifacts/smoke_pipeline
uv run --frozen python -m edraft.diagnose --scale smoke --output artifacts/smoke_pipeline
uv run --frozen python -m edraft.train --scale smoke --output artifacts/smoke_pipeline
uv run --frozen python -m edraft.eval_closed_loop --scale smoke --stage m3 --mode all --output artifacts/smoke_pipeline
uv run --frozen python -m edraft.eval_closed_loop --scale smoke --stage m4 --mode all --output artifacts/smoke_pipeline
```

## Reference execution, M0–M4

Run the stages in order from the repository root. Commands below use the default
`artifacts/` layout, seed 2026, and the pinned 8B model pair. Every entrypoint accepts
`--config` JSON and command-line overrides; `configs/reference.json` contains
M0-specific fields and is used only for M0 below. Reference compute checks the
passed CPU smoke artifact, appends a runtime estimate to NOTES, and checks free
disk space.

Choose physical GPU indices after inspecting current occupancy. The following
assignments are examples to replace with available devices. Single-process
commands map the chosen physical GPU to `cuda:0`; the launcher maps each listed
physical GPU to one worker. Tuning and timing require one idle GPU. Let sharded
workers finish before timing on their devices.

```bash
nvidia-smi
EDRAFT_SINGLE_GPU=0
EDRAFT_SHARD_GPUS=0,1,2,3
```

Fetch the selected models and prompt datasets explicitly, then build the prompt
manifest. Subsequent commands use cached artifacts; launcher workers run with
Hugging Face offline mode enabled. If everything is already cached, omit the two
download commands.

```bash
uv run --frozen python -m edraft.download --scale reference --allow_download
uv run --frozen python -m edraft.download --scale reference --allow_download --datasets openai/gsm8k openai/openai_humaneval HuggingFaceH4/MATH-500
uv run --frozen python -m edraft.prompts --scale reference
```

M0 compares 20 prompts at greedy, recommended non-thinking sampling, and pure
temperature 1. The numerical investigation covers target-only disagreements and
the real hidden/head reconstruction hook. M1 requires both `m0.json` and
`m0_numerics.json` to pass, including investigation of every target-only mismatch.

```bash
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.m0 --config configs/reference.json --scale reference --device cuda:0
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.investigate_m0 --scale reference --device cuda:0
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.bench_adapter --scale reference --device cuda:0
```

The adapter benchmark compares batched, sequential, and candidate scoring,
including candidate construction and numerical precision. Its reference-sized
random tensors are labelled **synthetic**; these timings do not establish a
real-model throughput gain.

M1-pre freezes the hidden projection and token features using 64 standard train
prompts. M1a then collects plain-DFlash traces with a floor of 250,000 emitted
train tokens at each primary temperature, separately from pure-temperature
diagnostics and the 12 long dev plus 32 long test generations. Audit after the
launcher exits. M1b runs on CPU and reads exactly the checksummed sources accepted
by that audit.

```bash
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.collect --scale reference --stage calibrate --device cuda:0
uv run --frozen python -m edraft.launch --scale reference --task collect --gpus "$EDRAFT_SHARD_GPUS" --train_tokens_per_temperature 250000 --standard_max_new_tokens 1024 --long_max_new_tokens 24576
uv run --frozen python -m edraft.collect --scale reference --stage audit --device cpu --train_tokens_per_temperature 250000 --standard_max_new_tokens 1024 --long_max_new_tokens 24576
uv run --frozen python -m edraft.diagnose --scale reference --device cpu --trace_dir artifacts/traces --m1_audit artifacts/m1_audit.json --workers 8
```

M2 requires a passed `m1_audit.json`, at least 500,000 eligible train output tokens,
both primary temperature floors, disjoint splits, frozen feature provenance, and
M1b checksum coverage of the eligible traces. It fits dense unary/pair parameters
first, then adds bias and sparse parameters. Extra epochs/tuning stop when dev
improvement is below 1%; both required stages and their results are reported.

```bash
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.train --scale reference --device cuda:0 --m1_audit artifacts/m1_audit.json --m1_diagnostic artifacts/diagnostic_reference.json
```

M2 writes `artifacts/m2.json`, `artifacts/m2/dense.pt`, and
`artifacts/m2/dense_sparse.pt`, including loss, held-out grouped/full-KL gap,
parameter sparsity, and sparse collision statistics. Its checkpoint resumes by
default; keep the same fitting configuration when restarting.

M3 tunes only on dev prompts, freezes settings, evaluates test prompts across GPUs,
then measures every arm back-to-back on one idle GPU. The four arms are plain
DFlash, online bigram, dense adapter, and dense plus sparse adapter. Reference
evaluation uses 64 standard test prompts with three seeds at each positive
temperature, plus 32 long test prompts with one seed: 480 requests per arm.
Each standard greedy test prompt also receives one target-only diagnostic.
`greedy_output_comparisons.json` reports every arm against both plain DFlash and
that target-only output, with token/sequence mismatch rates and coverage. Matching
target-only diagnostics are shared with M4; they are outside the timing suite.
Finite-precision mismatches are reported without being conflated with missing
comparison coverage or the mathematical exactness tests.

```bash
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.eval_closed_loop --scale reference --stage m3 --mode tune --device cuda:0 --standard_dev_prompts 8 --long_dev_prompts 12
uv run --frozen python -m edraft.launch --scale reference --task m3 --gpus "$EDRAFT_SHARD_GPUS" --evaluation_max_new_tokens 2048 --long_max_new_tokens 24576
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.eval_closed_loop --scale reference --stage m3 --mode timing --device cuda:0 --timing_prompts 0 --timing_warmups 1 --timing_repeats 3
uv run --frozen python -m edraft.eval_closed_loop --scale reference --stage m3 --mode summarize --device cpu
```

Proceed to M4 only when `artifacts/m3/summary.json` reports `complete`. M4 uses the
same prompt/seed counts with five arms: plain DFlash, online bigram, offline static,
offline plus all online deltas at stride 1, and online bias/sparse deltas only.
It performs separate dev tuning and remeasures both baselines in its own timing
session. Matching M3 baseline output and acceptance records are reused only when
their settings, generation identities, and checksums agree. Different M4 bigram
settings trigger fresh generation. Reused records have unavailable timing and
memory fields; all M4 timing and profile requests execute again. Its summary also
validates the matching M1b feature diagnostics and
writes `artifacts/m4/feature_context.json`; the default source is
`artifacts/diagnostic_reference.json` (`--feature_diagnostic` overrides that path).
Missing or mismatched feature evidence leaves M4 incomplete.

```bash
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.eval_closed_loop --scale reference --stage m4 --mode tune --device cuda:0 --standard_dev_prompts 8 --long_dev_prompts 12
uv run --frozen python -m edraft.launch --scale reference --task m4 --gpus "$EDRAFT_SHARD_GPUS" --evaluation_max_new_tokens 2048 --long_max_new_tokens 24576
CUDA_VISIBLE_DEVICES="$EDRAFT_SINGLE_GPU" uv run --frozen python -m edraft.eval_closed_loop --scale reference --stage m4 --mode timing --device cuda:0 --timing_prompts 0 --timing_warmups 1 --timing_repeats 3
uv run --frozen python -m edraft.eval_closed_loop --scale reference --stage m4 --mode summarize --device cpu
```

Each stage stores frozen `settings.json`, rank-local evaluation manifests, a
`timing.json`, and `summary.json` under `artifacts/m3/` or `artifacts/m4/`.
Large trace/checkpoint files and detailed reference request trajectories remain
local in `artifacts/`; retain them with the manifests to resume an interrupted run.
`--timing_prompts 0` means every required test prompt; a positive subset is a pilot
and cannot complete the milestone. Timing includes request initialization,
tokenization, proposals, verification, and online updates, with a separate CUDA
event pass for phase profiles. Confidence intervals resample prompt IDs after
aggregating seeds, and seed variability is reported separately.

Restart an interrupted command with the same configuration. Completed collection
and evaluation jobs are reused; keep collection GPU order and worker count fixed
to avoid duplicate work. Timing resumes complete all-arm comparison bundles and
reruns every arm of an interrupted bundle together. Use `--fresh_session` on tune
or timing when a new measurement session is intended. Prompt text, generation
caps, model revisions, implementation, or checkpoint changes require settings
that match the new run. A command that finishes with an `incomplete` audit/summary
has not passed its milestone gate.

Full scale is a separate opt-in experiment requiring `--scale full
--allow_large_run`, the larger prompt manifest and collection quotas, and at least
32 long dev prompts for tuning. It has not run. M5 extensions are optional.

## Proposal and exact verification

For frozen post-normalization draft features, project and standardize
`h = (h_raw @ P_h.T - mean) / std` using train calibration statistics. A frozen
embedding PCA supplies token features `E`. With realized previous tokens, the
full-vocabulary correction is

```text
U_all = (H @ W.T) @ E.T + b
g_i(a) = U_all[i,a] + sum_d e(a)^T A[d-1] e(prev_token(i,d)) + sparse_i(a)
s_i(a) = log_softmax(draft_logits_i)[a] - beta_sample * g_i(a)
```

Sparse corrections apply only to the top `M_sparse` IDs selected after dense
correction. The causal correctness path still scores and chooses across the whole
vocabulary. Logical slot zero is the first unknown token; the physical clean
anchor belongs to the verified prefix. For `i-d < 0`, history comes from
`prefix[len(prefix)+i-d]` only if that index exists. Missing history is skipped.

At positive `T_sample`, the same configured top-k/top-p/min-p warper is applied
to corrected proposal scores and target logits, after temperature division.
Verification accepts a proposal with `min(1,P(y)/q(y))`; the first rejection draws
from full-vocabulary `normalize(max(P-q,0))`. Full acceptance emits a target bonus
token. Greedy uses whole-vocabulary lowest-ID argmax, no temperature division,
sampling softmax, or residual sampling.

## Training and state

Offline and online training always use unwarped `T_train=1` and `beta_train=1`.
`P_train=softmax(target_logits)` and `q_train=softmax(l_unit-E_corr)` are independent
of the inference warper. Online training uses exact full-vocabulary KL. Compact
offline `union_rest_grouped` training preserves target probabilities on a stored
union plus their rest mass. Reconstructing `l_unit` from `h_raw_bf16` and the frozen
linear draft head permits an exact corrected proposal partition across all tokens;
the grouped target tail is still an approximation to full token-level KL, whose
gap must be measured.

The signed pair hash is approximate memory with deterministic collision
interference. Online lazy L1 tracks optimizer counter `u` and cumulative thresholds
`cum_thresh[u]`. A read shrinks by `cum_thresh[u]-cum_thresh[last_u]` without mutation;
a write catches up decay and updates only active entries between rounds.

Each request starts with independent zero deltas. Proposal snapshots remain fixed
through draft, sampling, and verification. Online stride `S` updates on every S-th
round using only that firing round, without buffering skipped rounds. Versions
advance only at those boundaries. The standalone lookahead helper uses strictly
positive `T_relax` for beliefs; frozen `C0` candidate sets are separate from evolving
`support` and `clamped` state. Closed-loop lookahead evaluation is optional M5.

The online bigram baseline is a smoothed normalized successor prior with exact
counts by default; it is not a reproduction of full tree-based Token Recycling.
Full Token Recycling, Test-Time Speculation, and SharpDraft have not run.

## Backend and interpretation limits

The local loop supports the inspected original `DFlashDraftModel` with Transformers;
it rejects `DFlash2DraftModel`. The pinned physical block has 16 positions: one
clean anchor plus 15 unknown proposals. Accepted length counts only accepted
unknown proposals, excluding replacement and bonus tokens. The final normalized
draft hidden state feeds the target's frozen linear head. Full hook, cache, mask,
and sampling details are in [the backend inspection](artifacts/dflash_inspection.md).

For the pinned 8B pair, standard non-thinking uses `T=0.7, top_k=20, top_p=0.8,
min_p=0` from the target model card, alongside greedy and pure-temperature-1
diagnostics. Long thinking uses `T=0.6, top_k=20, top_p=0.95, min_p=0`, with thinking
enabled and a 24,576-token cap constrained by available context. Thinking is
supported by the target but out of distribution for this drafter. Realized length
and repetition checks determine usable long-output bins; a cap alone does not
establish long-context evidence. Conditions and temperatures remain separate.

The two DFlash paths agree with each other in M0 but can differ from target-only
greedy execution at BF16 ties. Those measured mismatches and their numerical
investigation are reported in SUMMARY. The 4B pair is configurable but has not
been evaluated here. Changing model pairs requires new source inspection,
calibration, reproduction gates, and matching sampling settings.
