# Research execution notes

Specification: `prompt.md` (attachment differs only by final newline).

## Startup

Repository initially contained only the specification and had no Git history. Environment inventory is in `artifacts/environment.json`; unknown package/model revisions will be filled after local environment setup and inspection. No system packages, CUDA, or drivers will be changed. GPU 7 has an unrelated active process; use GPUs 0–6 only after checking occupancy.

Milestones: M0 in progress; M1–M4 not_run. Reference downloads are authorized through the explicit `--allow_download` option. Full scale and M5 are not selected. No scientific result is claimed yet.

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

Pre-run estimate: {"scale": "reference", "expected_tokens": 17920, "smoke_tokens_per_second": 1099.9935217379511, "cpu_smoke_extrapolation_seconds": 16.29100503399968, "free_disk_bytes": 33796323827712, "caveat": "CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot."}
