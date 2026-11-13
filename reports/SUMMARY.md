# Research status

As of 2026-11-13, reference M0 and M1-pre have **passed**. M1a collection is
**running**; reference M1b, M2, M3, and M4 are **not_run**. There is no measured
conclusion about adapter quality, online adaptation, or end-to-end speed yet.
Synthetic tests and smoke results do not satisfy the real reference gates.

| Milestone | Reference status | Evidence or remaining work |
| --- | --- | --- |
| M0 | passed | Upstream/local reproduction and target-only mismatch investigation |
| M1-pre | passed | 64 train calibration prompts; 22,377 hidden vectors; frozen hidden and token features |
| M1a | running | Four collection workers; final token/split/reconstruction audit pending |
| M1b | not_run | Headroom, realized long-output bins, repetition, and feature-shift report pending |
| M2 | not_run | Primary offline fit and reference loss/sparsity/collision results pending |
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

M1a is collecting at least 500,000 primary train output tokens, split equally
between greedy and the model-card warped `T=0.7` condition. Pure-temperature
diagnostics and long thinking are counted separately. The long condition requires
12 dev and 32 test generations at `T=0.6, top_k=20, top_p=0.95, min_p=0`, with
thinking enabled. This is out of distribution for the drafter. Final realized
lengths and repetition filtering have not been reported; no claim about 8K/16K
adaptation is supported yet. Collection progress is recorded in
[collection-processes.json](../artifacts/collection-processes.json).

CPU smoke paths exercise collection, compact grouped-loss fitting, all required
evaluation arms, and synchronous request-local online updates. Synthetic fitting
and reference-sized random-tensor microbenchmarks are implementation checks;
their results do not establish a reference acceptance or throughput improvement.

External baselines not run: full tree-based Token Recycling, Test-Time Speculation,
and SharpDraft. The required normalized online bigram baseline is implemented and
covered by smoke tests; its reference M3/M4 evaluations remain pending. Full-scale
experiments and optional M5 work have not run.
