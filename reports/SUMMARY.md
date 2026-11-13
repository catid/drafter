# Research status

M0 **passed**. M1-pre is being prepared; M1a, M1b, M2, M3, and M4 are **not_run**.
There is no measured conclusion about adapter quality or end-to-end speed yet.
Synthetic tests cannot satisfy the real reference gates.

The reference pair is Qwen/Qwen3-8B and z-lab/Qwen3-8B-DFlash-b16. Sampling conditions
come from the pinned checkpoint files. The mandatory ratio verifier was used for
stochastic comparisons. Upstream and local matched on all 20 prompts at each of
greedy, warped T=0.7 and pure T=1: 0/7,128 mismatched output-token positions.
Both differed from target-only greedy on 8/20 sequences and 607/2,388 token
positions. Same-prefix investigation reproduced all eight first divergences and
identified BF16 maximum-logit ties, with correct cache lengths and future masks.
See `artifacts/m0.json` and `artifacts/m0_numerics.json` for full evidence.

External baselines not run: full tree-based Token Recycling, Test-Time Speculation,
and SharpDraft. The required online bigram baseline is being implemented locally.
Optional M5 work has not run.
