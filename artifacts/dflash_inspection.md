# DFlash and reference checkpoint inspection

Inspected on 2026-11-13. Source: `https://github.com/z-lab/dflash`, pinned commit
`07ebd93db9f472af339b644bb70221ad8428328a`. The checkout's actual optional extra is
`local` (not inferred from an older release); it pins Linux `torch==2.13.0`,
`transformers==5.15.0`, and `jinja2==3.1.6`. Base dependencies include
`datasets==5.0.1`. Installed imports were inspected with `inspect.getsource`:
PyTorch reports `2.13.0+cu130`, Transformers `5.15.0`, and the imported
`DFlashDraftModel` is the class from this pinned checkout.

The checkout supports both the original `DFlashDraftModel` and a newer
`DFlash2DraftModel` subclass. The requested checkpoint declares the original
architecture; the local M0–M4 wrapper explicitly rejects the DFlash2 subclass.
No vendored files have been modified.

## Exact checkpoint sources and generation conditions

Target `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`:

- `generation_config.json` specifies thinking temperature `0.6`, `top_p=0.95`,
  `top_k=20`, EOS IDs `[151645,151643]`. `min_p` is absent; target README line
  327 explicitly recommends `MinP=0` for thinking. Thus the recorded thinking
  arm is `T=0.6, top_p=0.95, top_k=20, min_p=0.0`.
- `README.md`, Best Practices line 328, specifies **non-thinking**
  `Temperature=0.7`, `TopP=0.8`, `TopK=20`, `MinP=0`. This is the source for
  standard and long non-thinking stochastic arms; thinking generation defaults
  are not applied to those arms.
- `tokenizer_config.json` names `Qwen2Tokenizer` and its exact chat template
  checks `enable_thinking is defined and enable_thinking is false`. The false
  branch appends an empty `<think>\n\n</think>\n\n` after the assistant prefix.
  Both thinking and non-thinking are therefore supported by the template.
- The target configuration has vocabulary size 151936, hidden width 4096,
  36 layers and `max_position_embeddings=40960`. The README describes 32768
  native context and explains the configured 40960 allocation. Requested
  24576-token long outputs must still fit with the actual input length; the
  wrapper checks total length against the configured maximum. No RoPE scaling
  is introduced by this work.

Drafter `z-lab/Qwen3-8B-DFlash-b16`, revision
`9b41424b7109f9c5413454f481b09a82b85333f4`:

- `config.json` declares `DFlashDraftModel`, physical block size 16,
  five draft layers, hidden width 4096, mask token 151669 and target layer IDs
  `[1,9,17,25,33]`.
- Its README line 144 explicitly states the draft model is used with thinking
  disabled; the examples pass `enable_thinking=False` and greedy sampling.
  The checkpoint's own documented evaluation does not establish thinking
  coverage. **long_thinking is target-supported but out of distribution for
  this drafter**. It must be reported separately from non-thinking conditions.
- No distinct tokenizer configuration is supplied in the downloaded drafter
  metadata. The card uses `Qwen/Qwen3-8B`'s tokenizer, matching the local path.
- Current checkout `dflash/benchmark.py` supports `--reasoning on/off` via the
  target chat template and lists GSM8K, MATH-500, HumanEval, MBPP and MT-Bench.
  That generic current CLI support does not establish that this older
  checkpoint was evaluated in thinking mode.

All file URLs, resolved revisions, SHA256 digests and bytes are recorded by the
explicit downloader in `artifacts/environment.json`.

## Inspected forward and verification path

1. `Qwen3ForCausalLM.forward` calls `Qwen3Model`, exposes requested hidden
   states and applies `target.lm_head` to the final target hidden state.
   `dflash.model.extract_context_feature` concatenates
   `hidden_states[layer_id+1]` for the five selected layers. Its context feature
   shape is `[1, context_chunk_length, 20480]` for this pair.
2. The drafter reduces target context features using `fc`, applies
   `hidden_norm`, and processes a clean anchor plus masked noisy slots with
   full bidirectional draft attention. `DFlashDraftModel.forward` returns
   `self.norm(hidden_states)`. **h_raw is this final RMSNorm output at physical
   slots 1..15, immediately before `target.lm_head`**, shape `[15,4096]` for a
   full round (fewer logical slots at the output cap). `compute_logits` calls
   this linear head. This checkpoint has multiplier 1 and no softcap.
3. These parallel draft features and logits are computed once and remain fixed
   for the proposal round. Adapter pair messages condition on realized sampled
   tokens without recomputing DFlash hidden states.
4. Physical slot zero is the clean anchor, the last verified output token.
   Logical slot `i` maps to physical `i+1`; logical K is 15 at physical B=16.
   The verified prefix includes the anchor, so logical slot zero's d=1 history
   is exactly that anchor. There is no offset-zero interaction.
5. Current upstream positive-temperature draft sampling first divides FP32
   logits by T, gathers exactly top-k if enabled, takes softmax, keeps top-p
   using cumulative mass before each token `< top_p`, renormalizes, scatters
   back to V, then uses batched `torch.multinomial`. Local sampling mirrors this
   inspected numerical convention. The optional min-p extension follows
   top-k/top-p, and identical warpers apply to proposals and target logits.
6. At zero temperature upstream and local paths use full-vocabulary `argmax`,
   with the lowest index selected on ties. No sampling softmax or stochastic
   residual runs at zero temperature.
7. **This pinned upstream now uses ratio rejection sampling**, not independent
   match sampling: `_rejection_sample` draws K uniforms and tests `u*q(y)<P(y)`.
   At first rejection it samples the full-V positive residual; full acceptance
   samples a target bonus. Therefore both greedy and positive-temperature
   upstream/local token parity are hard M0 gates here. A compatibility-only
   independent-match arm is unnecessary for this commit.
8. The target verifies `[anchor,y0,...,y(K-1)]` in one forward. Its logits are K
   proposal conditionals plus a bonus conditional. After m accepted proposals,
   the cache retains the old anchor plus those m proposals and excludes the
   new replacement/bonus anchor until the next forward. Both target and draft
   caches use installed `DynamicCache.activate_past_recording()` and crop by
   negative removal count; Transformers 5.15's positive absolute crop spelling
   is deprecated. The local wrapper verifies the resulting exact cache length.
9. Local `accepted_length=m` counts accepted unknown proposals only; the
   replacement/bonus is excluded. Upstream `acceptance_lengths` records the
   round's produced token count `m+1` (possibly shortened by EOS/output cap).
   These statistics must not be compared without converting the definition.

For disabled and exactly-zero adapters, distributions are independent across
slots. Local draws batch these independent categorical samples and acceptance
uniforms to reproduce the upstream RNG stream token-for-token. Raw draft logits
are used for this identity path to avoid an unnecessary FP32 log-softmax shift;
unwarped `l_unit` is still retained for traces. Nonzero causal adapter scores are
sampled left to right with realized predecessor tokens.

## Implemented preliminary verification (synthetic only)

`tests/test_backend.py` instantiates the actual installed Qwen3 and DFlash
classes with tiny random CPU weights, without downloads. Ten tests passed:
greedy and warped/pure positive-temperature seeded upstream/local parity,
target-only greedy agreement, actual target-cache rollback after 0/1/2 accepted
proposals, bf16 raw-feature/head reconstruction, clean-anchor logical mapping,
EOS and output caps, zero-adapter exact parity, and CPU live online updates with
between-round versions. These tests do not satisfy the real reference M0 gate;
the real checkpoint run is reported separately by `edraft.m0`.
