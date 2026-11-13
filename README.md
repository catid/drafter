# Causal energy adapters for DFlash

This repository tests whether a small learned energy correction improves a parallel
DFlash drafter after its scoring, sampling, verification, and update costs. The full
experiment specification is [prompt.md](prompt.md). Work is in progress: M0 is being
validated; M1–M4 have not run. No improvement claim is established.

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
uv run --frozen python -m edraft.download --scale reference --allow_download
uv run --frozen python -m edraft.m0 --config configs/reference.json --scale reference
```

Smoke uses only built-in CPU models, requires no model or dataset downloads, and
does not establish reference milestone acceptance. Without `--allow_download`, all
model loading is cache-only. Full scale requires `--allow_large_run`. Downloads,
revisions, checksums where available, and disk use are recorded in
`artifacts/environment.json`. Runtime notes and exact sampling sources are in
[NOTES.md](NOTES.md). Generated results identify their scale.

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
