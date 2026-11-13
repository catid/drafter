# Codex Task: Train and Evaluate a Causal Energy Adapter for DFlash

You are implementing a research pipeline from scratch in this repository. This document is the complete specification. Read it before writing code.

The goal is to determine whether a small, trainable energy adapter improves a parallel DFlash speculative drafter after accounting for the adapter's own compute, sampling, verification, and online-update costs.

The mandatory MVP is:

```text
parallel DFlash block
→ full-vocabulary dense unary correction
→ optional sparse pair correction over a measured top-m set
→ left-to-right sampling with realized-token conditioning
→ exact target verification
→ synchronous between-round online adaptation
```

Closed-loop lookahead relaxation, two-sided pseudolikelihood, full published external baselines beyond the required M3/M4 online bigram baseline, block-length control, Triton kernels, and Qwen3.5 server integration are optional M5 extensions. The standalone lookahead math helper is required earlier so its equations and tests are available in M0–M4, but it is not a closed-loop evaluation arm until M5.

Do not fabricate results. If a model, dataset, backend hook, or benchmark cannot run, record the exact blocker and stop the affected milestone.

---

## 0. Operating rules and run scales

Work through M0–M4 in order. M5 is optional.

Do not change system drivers, CUDA, or system packages. Use a project-local uv environment and pin dependencies.

At startup, write:

```text
NOTES.md
artifacts/environment.json
```

Record:

- OS, Python, PyTorch, CUDA, and driver versions;
- GPU names, compute capabilities, and available memory;
- Transformers, Accelerate, Triton, and Datasets versions;
- DFlash repository URL and commit;
- model identifiers and revisions;
- tokenizer identifiers;
- project git commit;
- visible devices;
- resolved configuration and seeds.

Do not hard-code B300 memory, architecture IDs, interconnect behavior, or throughput.

Every script accepts:

```text
--scale {smoke,reference,full}
```

Smoke scale must run without downloads and must use only the built-in tiny synthetic target and parallel drafter on CPU. It must execute draft, adapter, sampling, target verification, residual correction, bonus-token handling, and online update.

Reference scale uses the supported real model pair and small manifests. If checkpoints are not cached, allow an explicit opt-in download:

```text
--allow_download
```

Passing `--allow_download` is authorized for fetching the default Qwen3 target, the z-lab DFlash drafter, and benchmark prompt datasets on this server. This is the only path that may fetch artifacts; log every download and do not download silently.

Log every downloaded model, revision, URL, checksum if available, and disk usage in `artifacts/environment.json`. Without `--allow_download`, mark the reference milestone blocked rather than silently downloading or substituting another model. All milestone acceptance checks use this scale.

Full scale is opt-in with:

```text
--allow_large_run
```

Long runs must be resumable from a manifest keyed by model pair, prompt, condition, seed, temperature, verification rule, proposal mode, checkpoint, and adapter version. Use atomic writes and rank-local logs.

Before reference or full runs:

1. run smoke scale;
2. measure smoke throughput;
3. estimate runtime;
4. write the estimate to `NOTES.md`;
5. check free disk space.

Label every result as `synthetic`, `smoke`, `reference`, `full`, `not_run`, or `blocked`.

---

## 1. Supported reference backend

The mandatory M0–M4 path uses Hugging Face Transformers and a DFlash pair supported by that backend.

Default pair:

```text
target:  Qwen/Qwen3-8B
drafter: z-lab/Qwen3-8B-DFlash-b16
```

Optional lightweight real pair:

```text
target:  Qwen/Qwen3-4B
drafter: z-lab/Qwen3-4B-DFlash-b16
```

Make both configurable.

Inspect the pinned DFlash checkout and use the optional dependency name actually present in that checkout. Do not assume that a local extra exists.

Do not switch M0–M4 to Qwen3.5, SGLang, or vLLM. Qwen3.5 server integration is an M5 extension.

Before fixing the evaluation conditions, inspect the exact `z-lab/Qwen3-8B-DFlash-b16` model card, evaluation scripts, tokenizer configuration, and local DFlash checkout, plus the target model's `generation_config.json` AND its `README.md` model card. The prompt must not assume either thinking or non-thinking support from memory.

**Source of sampling settings:** On Qwen3 checkpoints, `generation_config.json` holds the thinking-mode defaults (e.g., `temperature=0.6, top_p=0.95, top_k=20`), whereas the non-thinking defaults (e.g., `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0`) appear in the target model card's best-practices section. Take thinking-mode settings from `generation_config.json` and non-thinking settings from the target model card's best-practices text; record both values and their exact file/card sources in `NOTES.md`. Do NOT apply `generation_config.json` thinking defaults to non-thinking arms.

Use these conditions when supported by the inspected checkpoint:

### standard_nonthinking

Supported chat template and ordinary generation length with thinking disabled. Primary stochastic arm uses the target model card's non-thinking temperature and warper; also include greedy (`T_sample=0`) and a clearly labelled pure-temperature (`T_sample=1.0, top_p=1.0, top_k=0, min_p=0.0`) calibration/exactness diagnostic.

### long_thinking

Use `enable_thinking=True` when the target tokenizer/model supports it, even if the DFlash checkpoint was trained primarily for non-thinking mode. Label this condition out-of-distribution for the drafter unless the checkpoint's own evaluation path explicitly includes thinking. Use the target model's `generation_config.json` recommended positive temperature and sampling settings; do not use greedy decoding as the primary thinking condition. Set `max_new_tokens` up to 24K when the context window permits. This is the primary long-output condition for measuring online adaptation when it is available.

Make the warper choice explicit: either use the inspected target `generation_config.json` recommended `top-k`/`top-p`/`min-p` settings and apply that exact warper identically to target and proposal after the adapter correction, or run a clearly labelled pure-temperature ablation with `top_p=1, top_k=0, min_p=0`. Never leave the long-thinking warper implicit.

### long_nonthinking

Use this as an additional condition or fallback when thinking mode is unavailable. Use the target model card's recommended non-thinking positive temperature and warper as the primary stochastic long-output condition when specified, keeping `T_sample=1.0` pure-temperature as a clearly labelled ablation (not the sole long-output stochastic condition). This tests long-output distribution drift but must not be described as a reasoning or chain-of-thought benchmark.

Report the conditions separately and record in `NOTES.md` which condition was actually supported. Never silently substitute one for the other. If a condition produces too few long outputs, report it as data sparsity rather than silently filling bins with repeated or truncated text.

---

## 2. DFlash inspection and local generation loop

Inspect the DFlash source and actual installed model classes before writing wrappers. Record in `NOTES.md`:

1. where target hidden features are produced;
2. where parallel draft logits are produced (including any final normalization layer before the output head);
3. whether draft features are fixed for the proposal round;
4. actual `K`, including any clean anchor;
5. draft sampling at positive temperature;
6. greedy sampling;
7. target verification;
8. cache rollback after partial acceptance;
9. accepted-length definition.

Do not edit vendored DFlash. Reimplement the local generation loop in `src/edraft/dflash_loop.py` with adapter hooks.

M0 must compare upstream DFlash generation against the local adapter-disabled loop under identical prompts, seeds, tokenizer settings, masks, temperature, and verification rule. This upstream-versus-local comparison is the hard reproduction gate at `T_sample=0` and at positive temperature only when the upstream rule is the same ratio rule. If upstream uses another positive-temperature sampler, run it as a separately labelled compatibility diagnostic and use the local ratio-rule loop as the primary plain-DFlash baseline.

Separately compare both DFlash paths with target-only greedy generation and report mismatch rates. Do not automatically treat a target-only mismatch as a local implementation bug: rare floating-point or tie-breaking differences may affect both DFlash paths. If upstream and local disagree with each other, the reproduction gate fails and must be investigated. If they agree with each other but differ from target-only, report the mismatch and inspect cache rollback, sampling masks, and numerical margins before proceeding.

Before enabling the adapter, test target-state rollback for partial acceptance.

---

## 3. Notation and parameterization

Let `V` be vocabulary size, `K` block length, `c` the verified prefix, `h[i]` the fixed DFlash feature, and `draft_logits[i]` the DFlash logits.

Define `h_raw[i]` as the final per-slot DFlash hidden representation after any final normalization and immediately before the linear draft/target output projection (`draft_lm_head`), or the nearest equivalent tensor exposed by the inspected DFlash implementation. Record the exact normalization/head hook and tensor shape in `NOTES.md`.

Use a fixed projection:

```text
h[i] = standardize(P_h @ h_raw[i])
```

where `h_raw[i]` has dimension `d_h`, `P_h` has shape `[p, d_h]`, and `p=256` by default. `P_h` and the standardization statistics are computed once during the M1-pre calibration warmup, then frozen. Store the projected `h[i]` and projection metadata in trace shards.

`W` therefore has shape `[r, p]`. Do not leave the hidden-state source, projection, or standardization implicit.

Define:

```text
l_unit[i] = log_softmax(draft_logits[i])
```

at unit temperature.

Use a fixed token feature table `E` with shape `[V, r]`, default `r=64`. Build `E` with reproducible PCA or randomized SVD from a configurable embedding source. Save the artifact and compute feature statistics on the same M1-pre calibration split only.

Dense parameters:

```text
W: [r, p]
A: [D, r, r]
b: [V]
```

with default `D=4`.

There is no `d=0` interaction. A logical offset `d` is in `1..D` and is stored as `A[d-1]`. Use `A[d-1]` everywhere.

The optional sparse term is:

```text
S(d, previous_token, current_token)
```

It is an approximate hashed pair prior, not collision-free memory.

All adapter parameters start at zero. With the adapter disabled, the local loop must exactly reproduce plain DFlash under the same rule, masks, seed, and temperature.

---

## 4. Cross-boundary history

DFlash block implementations may include a clean anchor at physical index zero. Define logical slot `i=0` as the first unknown masked token, not the physical anchor. The clean anchor is the final verified prefix token `c[-1]` and belongs to the history.

For logical block position `i` and offset `d`:

```text
prev_token(i,d) =
    y[i-d]                    if i-d >= 0
    c[len(c) + i - d]         if i-d < 0 and the prefix token exists
    missing                   otherwise
```

If a history token is `missing`, skip the dense pair feature and do not perform a sparse lookup. Do not substitute a learned or padded token feature unless that behavior is explicitly configured as an ablation.

Thus logical slot zero can use the final verified prefix tokens `c[-1], c[-2], ..., c[-D]`. Record the physical-to-logical position mapping in `NOTES.md` and test it explicitly.

Use this history for dense pair messages, sparse lookup, the online bigram baseline, offline loss, online loss, lookahead initialization, and gradient tests.

---

## 5. Causal proposal

For candidate token `a`, define:

```text
g[i](a) =
    e(a)^T W h[i]
    + b[a]
    + sum over d=1..D of
      (
        e(a)^T A[d-1] e(prev_token(i,d))
        + S(d, prev_token(i,d), a)
      )
```

(skipping terms where `prev_token(i,d)` is `missing`).

The dense correction is full vocabulary. For the causal path, do not read `E` from HBM once per sequential token if the same unary contribution can be precomputed.

Precompute the block unary contribution in one batched operation:

```text
H             = stack(h[0:K])                  # [K, p]
U_all         = (H @ W.T) @ E.T + b            # [K, V]
```

Use the dtype supported by the measured fast path for `E` and the GEMM, with FP32 master parameters and accumulation when practical. Then add only the sequential pair contribution for the current slot:

```text
dense[i] = U_all[i] + E @ pair_feature_messages[i]
```

Use `M_sparse=256` as the default sparse candidate count. Do not reuse `m` for both a message vector and a candidate count.

Benchmark this batched implementation against a sequential full-vocabulary implementation and a candidate-restricted implementation. Do not assume either is faster.

If the sparse term is enabled, construct:

```text
C[i] = TopM(l_unit[i] - beta_sample * dense[i], M_sparse)
```

and evaluate sparse corrections only for `C[i]`. Sparse corrections are zero outside `C[i]`.

Define:

```text
E_corr[i](a) = dense[i](a) + sparse[i](a)
s[i](a) = l_unit[i](a) - beta_sample * E_corr[i](a)
```

For `T_sample > 0`, apply the arm's configured `sampling_warper` identically to both target logits and proposal scores immediately before final normalization:

```text
P[i] = normalize(sampling_warper(target_logits[i] / T_sample))
q[i] = normalize(sampling_warper(s[i] / T_sample))
```

In the milestone evaluations (M1b, M3, M4), the primary stochastic arms use the condition's target-recommended warper (`top_k`/`top_p`/`min_p` from Section 1), while pure temperature sampling (`top_p=1.0, top_k=0, min_p=0.0`) serves as the exactness/calibration diagnostic and fallback unwarped baseline. Never apply truncation to `s[i]` before adding `-beta_sample * E_corr[i]`. Record the exact sampling-warp configuration with every trace and evaluation arm.

Sample left to right. After sampling `y[i]`, use the realized token for every later position.

For `T_sample = 0`:

- do not divide by `T_sample`;
- do not call the sampling softmax;
- choose `argmax` over the entire vocabulary `a in {0,...,V-1}` of `s[i](a)`, not only over `C[i]`; `sparse[i](a)=0` outside `C[i]`;
- break ties by lowest token ID;
- do not call stochastic residual sampling.

The full-vocabulary dense operation is not assumed to be cheap. Benchmark unary projection, pair-message construction, sparse lookup, sequential sampling, and total proposal time.

Provide `dense_full` as the correctness path and `dense_candidate` as a speed ablation.

---

## 6. Sparse pair prior

Implement the dense group first.

Then implement one or both:

### exact_observed_pairs

Store explicit keys `(d, previous_token, current_token)` using an open-addressed or sorted key-value table with configurable capacity and overflow behavior.

### signed_feature_hash

Use a deterministic bucket hash and an independent sign hash. Default offline capacity is `2^24` buckets.

Signed hashing can reduce systematic bias in aggregate estimates, but it does not eliminate deterministic collision interference. Record capacity, unique observed keys, load factor, collision count, frequency-weighted collision count, memory, and performance versus `exact_observed_pairs`.

Apply proximal L1 shrinkage:

```text
x <- sign(x) * max(abs(x) - lr * lambda_l1, 0)
```

Untouched entries must remain exactly zero.

---

## 7. Temperature and optional lookahead

The causal M0–M4 path does not require internal soft beliefs.

Closed-loop lookahead generation is M5 only. Even though closed-loop lookahead benchmarking is deferred to M5, implement the standalone lookahead message/update helper in `src/edraft/adapter.py` during M0–M4. It must expose the pure forward field, backward field, cross-boundary prefix handling, masked hashed messages, and one damped update step without requiring a real DFlash model. This keeps the mathematical unit tests runnable while leaving full lookahead generation optional.

Use separate temperatures:

```text
T_sample
T_relax
```

For lookahead, `T_relax` must be strictly positive. Default `T_relax=1.0`.

### Immutable pre-sampling candidate sets vs. dynamic clamp state

Separate the immutable pre-sampling candidate sets in the block snapshot from the evolving clamp state during sequential sampling:

```text
C0[j]      = TopM(l_unit[j] - beta_relax * U_all[j], M_sparse)   # frozen in ProposalSnapshot
support[j] = C0[j]                                               # active support before clamping slot j
clamped[j] = None                                                # optional realized token ID
```

`C0[j]` and its masks are part of the immutable block snapshot and are NEVER mutated during sampling or clamping. Before slot `j` is sampled, `clamped[j] is None` and `support[j] = C0[j]` (defining sparse message masks and, in `candidate_masked` mode, proposal support). Once slot `i` is sampled and clamped to `y[i]`, set dynamic state `clamped[i] = y[i]`, `support[i] = [y[i]]`, `r[i] = [1.0]` (or one-hot at `y[i]` in `full_vocabulary` mode), and `mu[i] = e(y[i])` without mutating `C0[i]`. A full-vocabulary lookahead proposal is an allowed ablation.

Maintain internal beliefs for unclamped slots (`clamped[j] is None`) explicitly over the active domain (`support[j] = C0[j]` vs. full vocabulary `V`):

```text
if candidate_masked:
    # g[j] has shape [m_j] evaluated on support[j] (= C0[j] while unclamped)
    z_j = gather(l_unit[j], support[j]) - beta_relax * g[j]
    r[j] = softmax(z_j / T_relax)               # shape [m_j], normalized over support[j]
    E_j = E[support[j]]                         # shape [m_j, r]
else:
    # g[j] has shape [V] evaluated over 0..V-1 (with sparse terms zero outside support[j])
    z_j = l_unit[j] - beta_relax * g[j]
    r[j] = softmax(z_j / T_relax)               # shape [V], normalized over V
    E_j = E                                     # shape [V, r]

mu[j] = r[j] @ E_j                              # shape [r]
```

Never confuse candidate beliefs `r[j]` (`[m_j]` or `[V]`) with expected features `mu[j]` (`[r]`).

At each current slot `i` (where `support[i] = C0[i]` prior to clamping `i`):

1. use the latest `g[i]` to form slot scores (`s_C[i] = gather(l_unit[i], C0[i]) - beta_sample * g[i]` when `candidate_masked`, or `s[i] = l_unit[i] - beta_sample * g[i]` over `V` when full-vocabulary);
2. select or sample `y[i]`:
   - **At `T_sample = 0`**:
     - `candidate_masked` lookahead selects deterministic `argmax` over `a in C0[i]` (breaking ties by lowest vocabulary token ID);
     - `full_vocabulary` lookahead selects deterministic `argmax` over all `a in {0,...,V-1}` (breaking ties by lowest token ID).
   - **At `T_sample > 0`**:
     - For `candidate_masked` lookahead, construct a full-vocabulary score tensor with entries outside `C0[i]` set to `-inf` (and `s_C[i] / T_sample` on `C0[i]`), apply the configured `sampling_warper`, and normalize so `q[i](a) = 0` outside `C0[i]` and `sum_{a in C0[i]} q[i](a) = 1.0`.
     - For `full_vocabulary` lookahead, compute `q[i] = normalize(sampling_warper(s[i] / T_sample))` over all `V` tokens.
     - In both modes, `P[i] = normalize(sampling_warper(target_logits[i] / T_sample))` remains full-vocabulary, and the Section 8 rejection residual computes `normalize(max(P[i] - q[i], 0))` over all `V` tokens.
3. clamp slot `i` to `y[i]`: set `clamped[i] = y[i]`, `support[i] = [y[i]]`, `r[i]` singleton `[1.0]` (or one-hot at `y[i]` in `full_vocabulary` mode), and `mu[i] = e(y[i])` while leaving `C0[i]` untouched;
4. re-relax remaining unclamped positions `j > i` (`clamped[j] is None`) if required (`rerelax` mode).

Because the causal MVP scores the full vocabulary, any closed-loop M5 lookahead comparison must use matched support: compare `candidate_masked` lookahead with a causal `dense_candidate` ablation using the identical frozen `C0[j]`, or compare `full_vocabulary` lookahead with the causal full-vocabulary arm. Never attribute a support difference to lookahead quality.

For each synchronous rerelax step, first compute an undamped field `tilde_g_j` from the current forward and backward messages over `support`, then apply damping to every unclamped slot (`clamped[j] is None`):

```text
g_j <- (1 - eta) * g_j + eta * tilde_g_j
```

followed by updating `r[j]` and `mu[j]` over the configured domain as above. Do not silently substitute a causal one-pass update for `rerelax`.

Modes:

```text
off
static
rerelax
```

Default `off`. Defaults for `rerelax` are `R_init=2`, `R_clamp=1`, `eta=0.5`.

### Cross-boundary forward and in-block backward messages (CRITICAL: never wrap negative Python indices!)

For unclamped slot `i` (`support[i] = C0[i]`), token `a`, and relative offset `d in 1..D` (where `r_sup(k, b)` denotes `1.0` if `clamped[k] is not None`, `r[k][b]` in `candidate_masked` mode, and raw full-vocabulary belief `r[k][support[k][b]]` in `full_vocabulary` mode):

- **Forward message (`k = i - d`)**:
  ```text
  if i - d >= 0:
      # In-block predecessor slot k = i - d (either free belief on C0[k] or clamped singleton [y[k]])
      msg_fwd_A(i, d, a) = e(a)^T (A[d-1] @ mu[i-d])
      msg_fwd_S(i, d, a) = I[a in support[i]] * sum_{b=0..|support[i-d]|-1} r_sup(i-d, b) * S(d, support[i-d][b], a)
  else:
      # Cross-boundary verified prefix predecessor (NEVER index mu[i-d] with negative indices!)
      prev = prev_token(i, d)
      if prev is missing:
          msg_fwd_A(i, d, a) = 0.0
          msg_fwd_S(i, d, a) = 0.0
      else:
          msg_fwd_A(i, d, a) = e(a)^T (A[d-1] @ e(prev))
          msg_fwd_S(i, d, a) = I[a in support[i]] * S(d, prev, a)
  ```
- **Backward message (`k = i + d`)**:
  ```text
  if i + d < K:
      # In-block future slot k = i + d (unclamped: support[i+d] = C0[i+d])
      msg_bwd_A(i, d, a) = e(a)^T (A[d-1].T @ mu[i+d])
      msg_bwd_S(i, d, a) = I[a in support[i]] * sum_{b=0..|support[i+d]|-1} r_sup(i+d, b) * S(d, a, support[i+d][b])
  else:
      # Beyond end of draft block: no future tokens exist
      msg_bwd_A(i, d, a) = 0.0
      msg_bwd_S(i, d, a) = 0.0
  ```

Using the masked indicator `I[a in support[i]]` and the raw belief mass `r_sup(k, b)` (NEVER renormalized sub-weights!) ensures `msg_fwd_S` and `msg_bwd_S` are the exact partial derivatives of the single shared masked sparse energy `sum_{a in support[i]} sum_{b in support[k]} r_i(a) r_k(b) S(d, b, a)` in both `candidate_masked` and `full_vocabulary` modes. Use the identical `support` mask and cross-boundary prefix terms in the scalar block energy verified by Test #10.

Do not assert monotonic free-energy descent for arbitrary finite-step synchronous updates. Test shared-energy gradient consistency and small-step descent on a controlled toy problem with line search.

---

## 8. Exact verification

For `T_sample > 0`, target probabilities `P[i]` are the target conditionals at the requested temperature and realized draft prefix. Apply the configured sampling warper identically to target logits and proposal scores immediately before normalization as specified in Section 5 and Section 7.

### Mandatory ratio rule

Use standard speculative rejection sampling:

```text
alpha = min(1, P[i][y[i]] / q[i][y[i]])
```

At the first rejection, sample from:

```text
normalize(max(P[i] - q[i], 0))
```

over the full vocabulary. If all `K` positions are accepted, sample one target bonus token.

The ratio rule is mandatory for all stochastic M0–M4 comparisons.

Define plain-DFlash for the adapter experiments as the local adapter-disabled loop using this ratio rule. If upstream DFlash uses a different positive-temperature sampler, reproduce its native behavior only as a compatibility diagnostic; do not mix its acceptance statistics with the ratio-rule arms. The hard M0 reproduction gate remains upstream-versus-local under the same rule and settings.

Do not use independently sampled target tokens as a performance-verification rule. Independent target/draft matching has collision acceptance `sum_a P[i][a] * q[i][a]` rather than the ratio rule's maximal overlap `sum_a min(P[i][a], q[i][a])`. An upstream implementation that avoids explicit `q` probabilities may use a coupled common-random-number construction, but that is a separate algorithm. Do not call independent match sampling a comparable speculative-verification baseline.

At `T_sample=0`, compare deterministic argmax tokens, emit target argmax at the first mismatch, and never compute ratios or residual samples.

The active adapter snapshot (including `C0[j]` when lookahead is active) must remain immutable through drafting, sampling, and verification.

Record `adapter_version`, proposal mode, verification rule, temperature, draft tokens, accepted length, first rejection, `q[i][y[i]]` for stochastic runs, target probability, decisions, and bonus token.

---

## 9. Offline and online training

Use disjoint train, dev, and evaluation prompts. Use the canonical block-trace schema defined in Section 10 (`M1a`) as the single source of truth for stored shard fields (recording `loss_temperature=1.0` alongside `collection_temperature`).

### Training temperature and full loss

The loss always uses unwarped unit-temperature distributions, even when the generation context used to produce a trace was greedy or warped stochastic.

Trace collection records both:

```text
collection_temperature = the temperature used to generate the realized context
loss_temperature = 1.0
```

For a greedy collection context, compute the target training distribution at `T_train=1.0` conditioned on that realized greedy prefix. Do not replace it with a `T_sample=0` one-hot target.

Offline and online losses always use:

```text
T_train = 1.0
beta_train = 1.0
```

For all training objectives (both offline M2 and online M4), **disable `top_k`/`top_p`/`min_p` warpers** and use exact full-vocabulary `softmax`:

```text
P_train[i] = softmax(target_logits[i] / T_train)
q_train[i] = softmax(
    (l_unit[i] - beta_train * E_corr_train[i]) / T_train
)
```

where `E_corr_train[i]` uses the realized prefix/canvas tokens `prev_token(i, d)` and `C_train[i] = TopM(l_unit[i] - beta_train * dense[i], M_sparse)` when sparse terms are active. Inference-time sampling warpers (`top_k`/`top_p`/`min_p`) and `T_sample` apply ONLY to `P[i]` and `q[i]` during decoding/verification, NEVER to `P_train[i]` or `q_train[i]`.

When full target probabilities and full adapter scores are available:

```text
L_full = sum_i w[i] * KL(P_train[i] || q_train[i])
```

Use `L_full` as the reference objective and as the mandatory online objective in M4 (since live full-vocabulary `target_logits[i]` and `l_unit[i]` are already resident in GPU memory during generation).

### Union-plus-rest grouped objective (offline M2 compact traces)

For compact offline traces, use defaults `M_draft=256` and `M_target=64`. Store a union `U` of the DFlash top-`M_draft` IDs, target top-`M_target` IDs, the draft token, and special tokens. Also store `h_raw[i]` in `bf16` (captured after any final DFlash normalization and immediately before the linear `draft_lm_head` projection) for every slot, plus an immutable artifact reference, revision, dtype, and output-projection configuration for the exact frozen DFlash `draft_lm_head`. During M2, reconstruct the full-vocabulary unit-temperature draft logits with a batched `h_raw[i] @ draft_lm_head.T` (plus head bias if present) operation in FP32 accumulation before computing the adapter-shifted partition. If raw logits rather than log-probabilities are stored for either model, also store that model's exact full-vocabulary log normalizer. Store unwarped unit-temperature target probabilities `P_train[i](u)` on `U` and target rest mass.

Do not renormalize target probabilities over `U`.

Call this `loss_mode=union_rest_grouped`.

During offline training, reconstruct `l_unit[i](v)=log_softmax(draft_lm_head(h_raw[i]))` over all `V` tokens, compute the unwarped unit-temperature adapter-shifted score `s_train_i(v) = l_unit[i](v) - beta_train * E_corr_train[i](v)` over all `V` tokens, and compute its exact log partition:

```text
Z_i = logsumexp_v(s_train_i(v))
q_train_i(u) = exp(s_train_i(u) - Z_i)               for u in U_i
q_train_i_rest = max(1 - sum_{u in U_i} q_train_i(u), epsilon)
P_train_i_rest = max(1 - sum_{u in U_i} P_train_i(u), epsilon)
```

Then minimize the grouped objective:

```text
L_i_grouped =
    sum_{u in U_i} P_train_i(u) * log(P_train_i(u) / q_train_i(u))
    + P_train_i_rest * log(P_train_i_rest / q_train_i_rest)
```

This computes `q_train` on the stored union with the exact full-vocabulary denominator while grouping the unstored target tail into one rest category. It is not identical to the full token-level KL over the target tail; report its gap to `live_full` on a held-out shard. Do not silently replace `q_train_i_rest` with the unmodified DFlash rest mass. If a backend cannot preserve `h_raw` and the frozen draft head, only an unshifted-rest approximation may be used as a clearly labelled fallback. Do not estimate a log partition with a finite Monte Carlo tail sample inside autograd, and never call the fallback exact.

### Weighting

Default, using zero-based logical block positions:

```text
w[i] = exp(-i/gamma)
gamma=5
```

Therefore `w[0]=1.0`.

Ablations:

```text
rejection_relative:
  if first_rejection < K:
    weight 1 for i <= first_rejection
    weight 0.35^(i-first_rejection) for i > first_rejection
  if first_rejection == K:
    weight 1 for every i in 0..K-1

cutoff:
  weight 1 through the first rejection
  weight 0 afterward
```

When all `K` tokens are accepted, `first_rejection=K` and no out-of-range position is accessed.

Post-rejection positions are valid target conditionals for the realized draft canvas but are off-policy relative to the emitted sequence. Label them off-policy teacher data.

Use autograd and FP32 optimizer state. Do not differentiate through sampled IDs or frozen models.

Use an explicit optimizer configuration:

```text
offline W, A, b: AdamW with configurable learning rate and decoupled weight decay
offline S: SGD with proximal L1 by default
online dW, dA, db: configurable, default constant-LR SGD for the small session state
online dS: proximal SGD over deduplicated dirty buckets only
```

Do not apply weight decay to `b` or `S` unless explicitly configured. The L1 threshold operation assumes an optimizer step followed by an explicit proximal operation.

### Numerical precision and analytic gradient test

The frozen target and drafter may run in `bf16` or another configured inference dtype. Adapter master parameters and optimizer states are FP32.

Convert draft logits, target logits, adapter scores, softmax inputs, `q`, `P`, acceptance ratios, and residual distributions to FP32 before numerical operations. Compute residual normalization and probability ratios in FP32, or FP64 in toy exactness tests.

The token feature table `E` may use `bf16` for the forward GEMM only if it is compared against an FP32 reference and the numerical error is reported.

For unwarped `q = softmax((l_unit - beta * E_corr) / T)` against target `P`:

```text
G[i] = sum_a (q[i][a] - P[i][a]) * E[a]

dL/dA[d-1] =
    -(beta / T) * sum_i w[i] *
    outer(G[i], E[prev_token(i,d)])
```

(skipping terms where `prev_token(i,d)` is `missing`). The same `beta/T` factor applies to `W`, `A`, `b`, and `S` gradients. For training, use `beta_train/T_train = 1.0/1.0` in this formula. For a positive-temperature unwarped inference-loss gradient test, substitute `beta_sample/T_sample`. Do not compute this stochastic gradient formula at `T_sample=0` or with nonlinear top-k/top-p warpers enabled.

The analytic gradient test must compare this formula with autograd on the unwarped full-vocabulary `L_full` objective. Do not use `union_rest_grouped` for that test; its grouped tail has a different derivative unless the complete tail is explicitly represented. Test the grouped objective separately by comparing its autograd gradient and its held-out gap to `live_full`. Include a one-step loss-decrease test.

### Online adaptation

Offline parameters are frozen. Each request receives zero-initialized `dW`, `dA`, `db`, and `dS`. Online M4 updates optimize the exact live unwarped full-vocabulary `L_full` objective (`T_train=1.0, beta_train=1.0`).

**Stride semantics:** When `update_stride = S` (e.g., `S = 1` or `S = 5`), an online update executes on every `S`-th verification round using **only that firing round's verification block** (single-round lazy update, with zero gradient or activation buffering across skipped intermediate rounds `1..S-1`, matching TTS lazy stride behavior; buffered multi-round gradient accumulation is an optional M5 ablation).

The frozen offline signed hash may use capacity `2^24`. The online request delta `dS` must not allocate, zero, clone, or shrink a full `2^24`-bucket array per request or per round. Use a fixed GPU-resident signed hash table of capacity `2^20` buckets by default (`~4 MB` FP32 values + `~4 MB` INT32 `last_u` timestamps, allocated once per request, or larger if collision measurements require it). Every executed between-round optimizer step and snapshot update touches only the deduplicated dirty bucket indices accessed in that firing round's block (`<= K * D * M_sparse` buckets), using copy-on-write or versioned active-bucket overlays if needed so untouched buckets are never copied per round.

Track an integer optimizer-update counter `u` (incremented by 1 ONLY when an online update actually executes, so `stride=5` increments once every 5 verification rounds) and a cumulative threshold prefix sum `cum_thresh[u] = sum_{t=1}^u (lr_t * lambda_l1)` (`cum_thresh[0] = 0.0`), which remains exact under both constant and scheduled online learning rates:

- **At read time (during drafting/scoring)**: compute the shrunk value as a pure read-only function without writing to `dS` or `last_u`, preserving snapshot immutability:
  ```text
  delta_thresh = cum_thresh[u_snapshot] - cum_thresh[last_u]
  val_eff = sign(val) * max(abs(val) - delta_thresh, 0)
  ```
- **At write time (only during the between-round update `u -> u + 1`)**: first materialize catch-up shrinkage `cum_thresh[u] - cum_thresh[last_u]` on touched entries, apply the step `u + 1` gradient update and single-step proximal shrinkage `lr_{u+1} * lambda_l1`, and write back the updated values and `last_u = u + 1` into the next round's snapshot. (Maintaining a deduplicated active-bucket index list, sweeping it between rounds, and compacting buckets that hit `0.0` is an allowed simpler implementation; dirty-only shrinkage without catch-up decay is forbidden.) Report active-bucket count, capacity, `u`, and update bandwidth.

Keep `db` as a dense `[V]` FP32 GPU delta (`~600 KB`). Update `db` on GPU using the exact `L_full` gradient across `V`, clone the dense GPU `db` once into the new snapshot at each executed optimizer update boundary, and apply one fused dense GPU proximal-L1 pass (`lr_{u+1} * lambda_l1_b`) per executed optimizer step. Do not round-trip `db` through CPU or convert it to a Python dictionary during generation.

Updates happen only between verification rounds:

1. finish draft;
2. finish sampling;
3. finish target verification;
4. compute single-round update when `round_idx % update_stride == 0` (`1`-based round count, or documented `0`-based convention);
5. create a new immutable snapshot;
6. increment `adapter_version`;
7. use it on the next round.

No asynchronous parameter mutation is allowed in M4. Support update stride 1 and 5. Add proximal L2 decay on dense deltas and L1 shrinkage on `db` and `dS`. Reset request state by default.

### Online bigram baseline

The M3/M4 `online bigram baseline` must define a normalized proposal, not merely a table of successor IDs.

Maintain a per-previous-token successor-count table with:

```text
default per-request bigram table: exact successor counts, no eviction (k_ret = None)
top-k eviction ablation:          k_ret = 256 retained successors by default (configurable)
additive smoothing:               alpha = 1.0 by default (require alpha > 0; reject alpha <= 0 at config validation and tune strictly over positive values)
```

and an update rule based on committed target/output transitions. A transition `(prev_tok, curr_tok)` is added to the table ONLY when both `prev_tok` and `curr_tok` are known valid tokens (never `missing` or synthetic padding).

If `prev_token(i, 1)` is `missing`, do NOT query or update the bigram table; set `tr_prior_i(a) = 0` and return `l_unit[i]` directly so `q_tr[i] = normalize(sampling_warper(l_unit[i] / T_sample))` (or `argmax_V(l_unit[i])` at `T_sample = 0`).

Otherwise, let `N_total` be the exact total transition count for `prev_token(i, 1)`, let `S_ret` be the set of retained successor token IDs (all observed successors in the default exact mode, or `|S_ret| <= k_ret` in the top-k eviction ablation), and define:

```text
N_evicted = max(N_total - sum_{b in S_ret} count(b), 0)
count_eff(a) = count(a)                         if a in S_ret
               N_evicted / (V - |S_ret|)        if a not in S_ret and |S_ret| < V
               0                                if |S_ret| = V
tr_prior_i(a) = log((count_eff(a) + alpha) /
                    (N_total + alpha * V))
```

The evicted mass is spread uniformly over unlisted IDs, so `exp(tr_prior_i)` sums algebraically to `1.0` over all `V` tokens even after top-k eviction. Combine it with the frozen DFlash base logits:

```text
tr_score_i(a) = l_unit[i](a) + lambda_tr * tr_prior_i(a)
q_tr[i] = normalize(sampling_warper(tr_score_i / T_sample))
```

(and `argmax_V(tr_score_i)` with lowest-ID tie-break when `T_sample = 0`). Keep exact total and retained-ID counts; do not silently renormalize over `S_ret`. Because a request capped at 24K output tokens has at most 24,576 transitions, the default per-request implementation retains exact successor counts without eviction (`k_ret = None`), while top-k eviction (`k_ret = 256` default) is a measured memory ablation. If `N_total=0` or `lambda_tr=0`, bypass the prior addition and return `l_unit` directly so empty-table behavior is bit-exact with plain DFlash including tie-breaking. Update the table only between rounds using committed target/output transitions. Keep the table and its update state per request.

This is a documented `online bigram baseline`, not a reproduction of the full published Token Recycling tree algorithm. M5 may add full tree-based Token Recycling beyond this M3/M4 baseline.

---

## 10. Milestones

### M0: environment and baseline

- pin environment and inspect DFlash;
- load Qwen3 reference pair;
- run upstream generation;
- implement local adapter-disabled loop;
- test target-cache rollback;
- compare target-only, upstream DFlash, and local loop.

Acceptance: smoke passes, the real reference path runs, upstream DFlash and the local adapter-disabled loop agree token-for-token on at least 20 prompts under identical settings, and target-only mismatch rates for both paths are reported separately. Blockers are documented.

### M1: trace collection and headroom diagnostic

M1 has three required stages (`M1-pre`, `M1a`, `M1b`).

#### M1-pre: hidden projection calibration

Before writing M1a trace shards, designate a calibration split consisting of a default of 64 `standard_nonthinking` train-split prompts (or a configured larger count). Run a streaming warmup over that split, collect representative `h_raw[i]` tensors, fit `P_h` with incremental PCA or a documented fixed projection, and compute per-coordinate mean and standard deviation. Save:

```text
artifacts/h_proj.pt
```

Freeze `P_h` and its statistics before M1a. Project and standardize `h_raw` on the fly during trace collection. Do not fit the projection using evaluation prompts or future trace shards.

#### M1a: trace collection

Collect plain-DFlash trace shards before fitting any adapter. Split token and prompt budgets explicitly by condition and split so neither long-condition generations nor excluded pure-temperature diagnostic traces dilute M2's primary training budget:

```text
Reference scale target:
  standard_nonthinking (primary-fit traces only: T_collection=0 + model-card warped positive T):
    at least 0.5 million generated output tokens from train-split prompts (excluding pure T=1.0 unwarped diagnostic traces; plus separate dev and >=64-prompt test splits)
  standard_nonthinking (T_collection=1.0 pure-temperature diagnostic):
    counted separately from the 0.5M primary-fit floor
  authorized long condition:
    counted separately, at least 12 complete dev-split generations (for M3/M4 hyperparameter tuning) PLUS at least 32 complete test-split generations

Full scale target:
  standard_nonthinking (primary-fit traces only: T_collection=0 + model-card warped positive T):
    2 to 4 million generated output tokens from train-split prompts (excluding pure T=1.0 diagnostic traces; plus dev/test shards)
  authorized long condition:
    counted separately, at least 32 dev-split generations PLUS at least 96 test-split generations
```

Split prompts by prompt ID into disjoint train, development, and test sets. Keep at least one complete dataset held out for evaluation. Collection must resume after interruption.

Collect traces under the configured generation conditions: `T_sample=0` and the target model card's recommended non-thinking positive temperature with its matching warper for `standard_nonthinking` (plus a clearly labelled `T_sample=1.0` pure-temperature diagnostic); the target model card's recommended non-thinking positive temperature and matching warper for `long_nonthinking` when run (plus `T_sample=1.0` pure-temperature ablation); and `T_recommended` with its matching warper from `generation_config.json` for `long_thinking` when supported. These are collection temperatures for the realized context; the stored training distributions remain unwarped unit-temperature (`P_train`) as specified above. Store the condition and collection temperature so traces from different regimes are never pooled accidentally.

Canonical block-trace schema (store for every block):

```text
prompt_id
condition
seed
collection_temperature
loss_temperature
verification_rule
sampling_warp_config
absolute_output_offset
input_context_length
verified_prefix_tail of at least D tokens
physical_to_logical block mapping
draft_tokens
draft logits or unwarped unit-temperature log-probabilities on the stored union
target unwarped unit-temperature logits/probabilities on the stored union
target unwarped rest mass or exact normalizer
h_raw_bf16[i] (after any final DFlash normalization, immediately before draft_lm_head)
draft_lm_head artifact ID, revision, dtype, bias/tied-weight status, and exact output-projection metadata
projected standardized h[i]
candidate IDs when sparse mode is enabled
accepted length
first rejection index
replacement token
adapter_version
```

Verify on a held-out shard that `draft_lm_head(h_raw_bf16[i])` and stored unwarped unit-temperature probabilities reconstruct live values within the configured `bf16`/FP32 tolerance. Do not train from `T_sample=0` one-hot labels or warped distributions.

#### M1b: diagnostic

Run plain DFlash at `T_sample=0` and the target model card's recommended non-thinking positive temperature with matching warper for `standard_nonthinking`; use pure `T_sample=1.0` without warping as an exactness/calibration diagnostic. For `long_nonthinking`, use the target model card's recommended non-thinking positive temperature and matching warper as the primary stochastic condition when specified, with `T_sample=1.0` pure-temperature as an ablation. For `long_thinking`, use `generation_config.json`'s positive `T_recommended` and matching warper as the primary condition; run `T_sample=0` only as a separately labelled greedy diagnostic and do not treat repetition-dominated greedy output as evidence about the long-thinking regime. Do not pool temperatures or long conditions.

A `max_new_tokens` cap is not evidence that a generation is long. Build prompt templates that request outputs likely to exceed the target bins, such as long code files, large structured documents, exhaustive enumerations, or extended proofs. Use `min_new_tokens` only when supported and only as a documented generation setting. Record realized output lengths and detect pathological repeated n-grams or endless-loop behavior. Report and separately exclude repetition-dominated outputs from the primary long-output analysis; do not fill long bins with artificially repeated text.

If the requested long prompts still produce too few outputs beyond 8K or 16K tokens, mark those bins empty and do not draw conclusions about long-context adaptation.

Report by both output-token offset and input-context-length bin:

```text
output bins: 0-1K, 1K-4K, 4K-8K, 8K-16K, 16K+
context bins: configured and reported from realized input lengths
```

Measure target rank, sampled-target rank, recall at `m in {1,4,16,64,256}`, target mass inside top-m, first-rejection rank, recoverability, mean overlap `sum_a min(P[a],q0[a])`, cross-entropy, realized output length, and accepted length. Also report per-condition summary statistics of the standardized projected features `h[i]` (coordinate mean norm, variance ratio relative to `standard_nonthinking` calibration, L2 norm quantiles, and outlier rate across output-offset bins) so hidden-feature covariate shift in `long_thinking` / `long_nonthinking` is explicitly quantified alongside M4 results.

The optional greedy oracle substitutes a first-rejection token if present in top-m and re-verifies. Label it a greedy diagnostic.

### M2: offline fit

The primary offline fit uses only train-split `standard_nonthinking` traces collected at `T_collection=0` and at the target model card's recommended non-thinking positive temperature with its matching warper, optimizing unwarped `T_train=1.0` and `beta_train=1.0` targets. Pure `T_collection=1.0` unwarped traces may be used as a clearly labelled calibration ablation. Do not include `long_thinking` or `long_nonthinking` traces in the primary fit; those conditions are reserved for M3/M4 distribution-shift and online-adaptation evaluation. Add a separately labelled M5 oracle ablation that trains on long-condition traces only if there is enough data, and never compare that ablation to the default online-adaptation claim without stating the changed training distribution.

Fit dense unary and pair terms first. Add bias and sparse prior afterward.

Report train/dev/test loss, `live_full` versus `union_rest_grouped` gap, overlap, top-1 agreement, norms, exact-zero sparsity, and comparison with zero-weight DFlash.

If improvement is under 1% on the reference split, stop tuning and report it.

### M3: closed-loop evaluation

Reference arms:

```text
plain DFlash
online bigram baseline
dense adapter
dense plus sparse adapter
```

The `online bigram baseline` is required in M3/M4. It must use the same prompts, seeds, verification rule, timing, and target model.

Evaluation prompt and seed counts:

```text
Reference scale:
  64 standard_nonthinking test prompts, 3 stochastic seeds per prompt at T_sample > 0
  32 long-condition test prompts (disjoint from the >=12 long dev prompts), 1 stochastic seed per prompt (to bound reference wall-clock compute; report prompt-level bootstrap CIs)

Full scale:
  at least 128 standard_nonthinking test prompts, 3 stochastic seeds per prompt
  at least 64 long-condition test prompts, 3 stochastic seeds per prompt
```

Every arm shares prompt IDs, seeds, tokenizer settings, block length, temperature, `top-p`/`top-k`/`min-p` settings, and the mandatory ratio verification rule. Reuse `plain DFlash` and `online bigram baseline` logs across M3 and M4 **for acceptance and output metrics only**; always re-measure throughput and latency (`tokens/sec`, phase `ms`, `p50`/`p95`) back-to-back in the same isolated timing session alongside the arms being compared so server drift cannot skew speed comparisons.

Tune `beta`, learning rates, weight decay, L1, online stride, and other adapter hyperparameters only on development prompts (`standard` dev and the `>=12` `long` dev prompts). Tune the `online bigram baseline`'s `lambda_tr` and positive `alpha > 0` (plus `k_ret` around `256` when evaluating the top-k eviction ablation; exact un-evicted counts remain the primary default) with the same development-prompt and timing budget. Freeze every arm's settings before test evaluation.

Use `T_sample=0` (`beta_sample=1`) plus the target model card's recommended non-thinking positive temperature and matching warper as the primary stochastic `standard_nonthinking` arms, keeping pure `T_sample=1.0` unwarped as an exactness/calibration diagnostic. For `long_nonthinking`, use the target model card's recommended non-thinking positive temperature and matching warper as the primary stochastic arm, with `T_sample=1.0` pure-temperature as an ablation. If `long_thinking` is supported, use `generation_config.json`'s recommended positive temperature (`T_recommended`) with its matching warper; greedy `long_thinking` is diagnostic only. Inference `beta` sweeps are optional full-scale ablations.

Measure mean and median accepted length, acceptance by block position, target forwards per generated token, proposal time, verification time, tokens/sec, p50/p95 latency, and peak allocated/reserved memory.

Compute 95% bootstrap intervals by resampling prompt IDs, not individual prompt-seed observations. When multiple seeds per prompt are run (`standard` at reference scale; all conditions at full scale), aggregate seeds within each prompt before bootstrapping and report seed-to-seed variability separately.

Report accepted length and acceptance probability by output-token offset and input-context-length bin.

State whether throughput improved after including proposal cost.

### M4: online adaptation

Reference arms:

```text
plain DFlash
online bigram baseline
offline static
offline + online deltas, stride 1
online db/dS only
```

Full adds stride 5, zero-offline online-only, rejection-relative weighting, and cutoff.

Evaluate the authorized long condition from M1 (`long_thinking` when supported, otherwise `long_nonthinking`, stating explicitly if `long_thinking` was unavailable) as well as `standard_nonthinking`, using the M3 prompt/seed counts (`1` stochastic seed per long test prompt at reference scale; `3` at full scale).

Use the same condition-specific sampling settings as M3 (`T_sample=0` + model-card non-thinking warped positive temperature + `T_sample=1.0` unwarped diagnostic for non-thinking; `generation_config.json` `T_recommended` + matching warper for `long_thinking`). Do not use greedy decoding as the primary long-output condition.

Tune online learning rate, proximal coefficient, L1 coefficient, and stride only on development prompts (`standard` dev + the `>=12` `long` dev prompts). Freeze them before test evaluation.

Report accepted length, acceptance by block position, and acceptance by output offset and context-length bin. Compute 95% intervals by resampling prompt IDs (after aggregating seeds within each prompt whenever `>1` seed is run), and report seed variability where applicable. Re-measure `plain DFlash` and `online bigram baseline` timing in the same isolated benchmark session as the M4 online arms.

Acceptance requires exactness, immutable snapshots, request isolation, update timing, and parameter-norm reporting.

Write `reports/SUMMARY.md` and `README.md`. State which external baselines were not run.

### M5: optional

Lookahead closed-loop evaluation, two-sided verified-stream pseudolikelihood, exact pair table versus signed hash, full tree-based Token Recycling beyond the M3/M4 `online bigram baseline`, Test-Time Speculation, SharpDraft, multi-round buffered stride accumulation, Qwen3.5 server backend, adaptive block length, `torch.compile`, and Triton. The primary lookahead comparison must pair candidate-masked lookahead with a causal `dense_candidate` arm using the identical frozen `C0[j]`; full-vocabulary lookahead is a separate ablation.

---

## 11. Measurement and compute

Use process-per-GPU prompt sharding for collection and evaluation. Basic correctness must run on CPU or one GPU.

Latency measurements use one process on an explicitly selected idle GPU with warmups, synchronized wall-clock timing, CUDA events, repeats, median, p95, and bootstrap intervals.

Measure separately:

```text
target feature extraction
DFlash draft
dense adapter score
sparse lookup
sequential sampling
target verification
online update
```

Do not write custom CUDA or Triton kernels in M0–M4. `torch.compile` is optional after profiling. Triton is optional only after numerical parity and measured speed gates.

---

## 12. Required tests

Implement under `tests/`:

1. zero-weight adapter exactly matches plain DFlash;
2. synthetic CPU smoke path completes without downloads;
3. causal `T_sample=0` never divides by zero, calls sampling softmax, or calls residual sampling;
4. lookahead `T_sample=0` uses positive `T_relax` beliefs, respects the configured `C0[i]` vs. full-`V` greedy argmax domain, updates `support[i]`/`clamped[i]` dynamically without mutating frozen `C0[i]`, and remains deterministic;
5. `W` and `A[d-1]` gradients match autograd on the unwarped full-vocabulary `L_full` objective, and `P_train`/`q_train` remain completely invariant to inference warper settings (`top_k`/`top_p`/`min_p`);
6. one toy descent step lowers loss;
7. cross-boundary prefix tokens reach slot zero in causal, online bigram, and lookahead forward messages (never wrapping negative Python indices `i - d < 0` to the end of the block, and cleanly skipping `missing` history);
8. no self-interaction or offset indexing error (and backward lookahead messages are zero when `i + d >= K`);
9. backward bilinear message uses `A[d-1].T`;
10. backward and forward hashed lookahead messages match the exact partial derivatives of the single masked shared energy using `I[a in support[i]]` and raw belief mass `r_sup` (including cross-boundary prefix terms);
11. exact toy sampling with `V<=8` and `K<=4`;
12. toy exactness covers target mass outside `C0` (including candidate-masked lookahead `-inf` masking before warper), residual, bonus, mandatory ratio verification, per-prefix distributions, and an identical `top_k=3, top_p=0.8` warper applied after adapter correction (plus `min_p` coverage if implemented);
13. tolerances derive from sample size;
14. greedy toy output equals target-only greedy output;
15. real-model greedy mismatch is reported against target-only and plain DFlash;
16. trace shards round-trip;
17. hash and sign functions are deterministic;
18. L1 shrinkage preserves exact zeros, pure-functional read-time evaluation never mutates snapshot tensors or `last_u`, and lazy `cum_thresh[u]` shrinkage matches eager per-update proximal L1 within FP32 tolerance across 10+ multi-step rounds under both `stride=1` and `stride=5` and under both constant and scheduled learning rates;
19. snapshot updates (and sequential clamping of `support[i]`) never mutate the frozen `ProposalSnapshot` or `C0[j]`;
20. `adapter_version` changes only between rounds;
21. online request state is isolated;
22. target-cache rollback preserves target-only behavior;
23. unwarped unit-temperature log-probabilities reconstructed via `draft_lm_head(h_raw_bf16)` match live DFlash `l_unit` within configured `bf16` tolerance on both synthetic and real reference hooks;
24. `online bigram baseline` rejects `alpha <= 0`, handles `missing` history cleanly by falling back to `l_unit`, satisfies `abs(sum(exp(tr_prior)) - 1.0) <= 1e-6` (or dtype-derived tolerance) over all `V` tokens both in exact mode (`k_ret=None`) and after top-`k_ret` eviction (`k_ret=256` default or smaller toy `k_ret` with `N_evicted > 0`), and `N_total=0` or `lambda_tr=0` reproduces plain DFlash bit-for-bit including tie-breaking;
25. M4 online `L_full` update executes single-round updates and increments optimizer counter `u` only on active `stride` boundaries (without buffering intermediate skipped rounds).

---

## 13. Repository layout

```text
pyproject.toml
README.md
NOTES.md
artifacts/
reports/
configs/
src/edraft/
  toy.py
  features.py
  adapter.py
  sparse.py
  dflash_loop.py
  sampler.py
  verify.py
  traces.py
  collect.py
  diagnose.py
  train.py
  eval_closed_loop.py
  online.py
  launch.py
tests/
```

Every script accepts a config plus command-line overrides, logs the resolved config and git commit, records seeds, writes atomic outputs, and uses resumable manifests.

The README must explain:

- proposal equations;
- `T_sample` versus `T_relax` and unwarped `T_train = 1.0` (`P_train`, `q_train`);
- full-vocabulary dense scoring;
- exact ratio residual correction;
- `union_rest_grouped` objective and `h_raw_bf16` reconstruction;
- hashed-pair collision and lazy `cum_thresh` L1 behavior;
- cross-boundary context (including `i - d < 0` prefix boundary rules and `missing` history);
- snapshot/version (`C0[j]` vs. `support[j]`) and single-round stride semantics;
- DFlash/backend limitations;
- exact smoke and reference commands.
