# Research status

M0 is in progress. M1-pre, M1a, M1b, M2, M3, and M4 are **not_run**.
There is no measured conclusion about adapter quality or end-to-end speed yet.
Synthetic tests cannot satisfy the real reference gates.

The reference pair is Qwen/Qwen3-8B and z-lab/Qwen3-8B-DFlash-b16. Sampling conditions
are being fixed from the pinned checkpoint files, and the mandatory ratio verifier
will be used for stochastic comparisons. The reproduction gate includes at least
20 prompts and separate target-only mismatch rates.

External baselines not run: full tree-based Token Recycling, Test-Time Speculation,
and SharpDraft. The required online bigram baseline is being implemented locally.
Optional M5 work has not run.
