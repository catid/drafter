"""Local DFlash loop with fixed parallel features and causal adapter sampling.

The upstream checkout is never patched. Physical slot zero is the verified
anchor; logical proposal slots are physical 1..B-1. Accepted length here always
excludes that anchor and the replacement/bonus token.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Callable

import torch
from torch.nn import functional as F

from .backend import crop_cache, extract_context_feature, make_cache
from .sampler import (
    SamplingWarper,
    probabilities,
    sample_from_top_scaled,
    sample_token,
    sample_token_compact_with_prob,
    sample_token_tensor,
)
from .verify import CompactProposal, verify_block


@dataclass
class GenerationResult:
    output_ids: torch.Tensor
    num_input_tokens: int
    blocks: list[dict]
    elapsed_seconds: float
    phase_ms: dict[str, float]

    @property
    def output_tokens(self) -> list[int]:
        return self.output_ids[0, self.num_input_tokens:].tolist()

    @property
    def num_output_tokens(self) -> int:
        return self.output_ids.shape[1] - self.num_input_tokens

    @property
    def tokens_per_second(self) -> float:
        return self.num_output_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.


class PhaseTimer:
    """CUDA events avoid synchronizing every slot; final collection synchronizes."""
    def __init__(self, device, enabled):
        self.device = torch.device(device)
        self.enabled = enabled
        self.events = []
        self.totals = {}

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                yield
                b.record()
            self.events.append((name, a, b))
        else:
            start = time.perf_counter()
            yield
            self.totals[name] = self.totals.get(name, 0.) + (time.perf_counter() - start) * 1000

    def finish(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for name, a, b in self.events:
            self.totals[name] = self.totals.get(name, 0.) + a.elapsed_time(b)
        return self.totals


def _is_zero(snapshot):
    if snapshot is None:
        return True
    flag = getattr(snapshot, "is_zero", None)
    if flag is not None:
        return bool(flag() if callable(flag) else flag)
    return (not bool(snapshot.W.count_nonzero()) and not bool(snapshot.A.count_nonzero())
            and not bool(snapshot.b.count_nonzero()) and getattr(snapshot, "sparse", None) is None)


def _sample_proposal(draft_logits, h_raw, prefix, *, temperature, warper, generator,
                     snapshot, projection, beta, bigram, timer, need_training_tensors=True,
                     fastpath=None, _return_token_tensor=False, _allow_compact_q=False):
    K = len(draft_logits)
    probs_fn = None if fastpath is None else fastpath.probabilities
    compact_probs_fn = None if fastpath is None else fastpath.compact_probabilities
    unary_fn = None if fastpath is None else fastpath.unary
    bf16_gemm = bool(fastpath is not None and fastpath.bf16_gemm)
    bigram_inactive = (bigram is None or bigram.lambda_tr == 0 or not bigram.counts)
    adapter_active = not _is_zero(snapshot)
    # Plain evaluation does not pay for adapter-only projection or training transforms.
    l_unit = None
    if need_training_tensors or adapter_active or not bigram_inactive:
        with timer.phase('unit_log_probability'):
            l_unit = draft_logits.float().log_softmax(-1)
    H = None
    if need_training_tensors or adapter_active:
        with timer.phase('feature_projection'):
            H = projection(h_raw) if projection is not None else h_raw.float()
    if K == 0:
        empty_toks = draft_logits.new_empty((0,), dtype=torch.long) if _return_token_tensor else []
        return empty_toks, None, l_unit, H
    if _is_zero(snapshot) and bigram_inactive:
        # Scores are independent in this case. Batch the categorical draws to
        # preserve the inspected upstream RNG stream. The mathematical causal
        # proposal is unchanged; raw logits avoid an unnecessary FP32 shift.
        with timer.phase("sequential_sampling"):
            if temperature == 0:
                token_tensor = draft_logits.argmax(-1)
                return (token_tensor if _return_token_tensor else token_tensor.tolist()), None, l_unit, H
            q = (probabilities if probs_fn is None else probs_fn)(draft_logits, temperature, warper)
            token_tensor = torch.multinomial(q, 1, generator=generator)[:, 0]
            return (token_tensor if _return_token_tensor else token_tensor.tolist()), q, l_unit, H
    U = None
    if adapter_active:
        with timer.phase("dense_adapter_score"):
            U = snapshot.unary(H, unary_fn=unary_fn, bf16_gemm=bf16_gemm)
    if snapshot is None or snapshot.C0 is None:
        device = draft_logits.device
        V = draft_logits.shape[-1]
        D_hist = len(snapshot.A) if adapter_active else (1 if not bigram_inactive else 0)
        prefix_tail = prefix[-D_hist:] if D_hist > 0 else ()
        L_ctx = len(prefix_tail)
        ctx_buf = torch.empty(L_ctx + K, dtype=torch.long, device=device)
        if L_ctx:
            ctx_buf[:L_ctx] = torch.tensor(prefix_tail, dtype=torch.long, device=device)
        offsets_all = torch.arange(1, D_hist + 1, dtype=torch.long, device=device) if adapter_active else None
        compact_k = warper.top_k if (temperature > 0 and 0 < warper.top_k < V) else 0
        k_eff = 1 if (temperature == 0 and V > 0) else compact_k
        slot_probs: torch.Tensor | None = None
        slot_indices = torch.empty((K, compact_k), dtype=torch.long, device=device) if compact_k else None
        q_selected_buf: torch.Tensor | None = None
        full_qs: list[torch.Tensor] | None = [] if (temperature > 0 and not compact_k) else None

        use_bigram_compact = (
            not adapter_active
            and not bigram_inactive
            and bigram.lambda_tr > 0
            and k_eff > 0
        )
        use_sparse_compact = (
            adapter_active
            and beta != 0
            and snapshot.sparse is not None
            and bigram_inactive
            and k_eff > 0
            and snapshot.M_sparse >= 1
            and snapshot.M_sparse + k_eff <= V
        )

        bg_state = None
        bg_base_vals = None
        bg_base_ids = None
        if use_bigram_compact:
            with timer.phase("bigram_score"):
                bg_state = bigram._sync_device_table(device, l_unit.dtype)
                if temperature == 0:
                    bg_base_vals, bg_base_ids = torch.max(l_unit, dim=-1, keepdim=True)
                else:
                    prob_dt = l_unit.dtype if l_unit.dtype == torch.float64 else torch.float32
                    _, bg_base_ids = torch.topk(l_unit.to(prob_dt) / temperature, k_eff, dim=-1)
                    bg_base_vals = l_unit.gather(-1, bg_base_ids)

        precomp_sparse = None
        if use_sparse_compact and snapshot.pair_is_zero():
            with timer.phase("sparse_lookup"):
                precomp_sparse = snapshot.precompute_sparse_pool(
                    l_unit, U, temperature=temperature, k_eff=k_eff, beta=beta
                )

        for i in range(K):
            if use_bigram_compact:
                with timer.phase("bigram_score"):
                    prev_tok_t = ctx_buf[L_ctx + i - 1] if (L_ctx + i) > 0 else None
                    sel_scores, sel_ids = bigram.merge_compact_candidates(
                        l_unit[i], bg_base_vals[i], bg_base_ids[i], prev_tok_t,
                        temperature=temperature, k_eff=k_eff, state=bg_state,
                    )
                with timer.phase("sequential_sampling"):
                    if temperature == 0:
                        tok_t, p_sub, idx_sub, q_sel = sel_ids[0], None, None, None
                    else:
                        tok_t, p_sub, idx_sub, q_sel = sample_from_top_scaled(
                            sel_scores, sel_ids, warper, generator
                        )
            elif use_sparse_compact:
                avail = min(D_hist, L_ctx + i)
                prev_ids = ctx_buf[L_ctx + i - offsets_all[:avail]]
                pre_i = (
                    (precomp_sparse[0][i], precomp_sparse[1][i])
                    if (precomp_sparse is not None and avail > 0)
                    else None
                )
                sel_scores, sel_ids_or_full = snapshot.score_slot_compact_candidates(
                    l_unit[i], U[i], offsets_all[:avail, None], prev_ids,
                    temperature=temperature, k_eff=k_eff, beta=beta,
                    timer=timer, contiguous_offsets=True, bf16_gemm=bf16_gemm,
                    precomputed_ids=pre_i,
                )
                with timer.phase("sequential_sampling"):
                    if sel_scores is None:
                        tok_t, p_sub, idx_sub, q_sel = sample_token_compact_with_prob(
                            sel_ids_or_full, temperature, warper, generator, compact_probs_fn=compact_probs_fn
                        )
                    elif temperature == 0:
                        tok_t, p_sub, idx_sub, q_sel = sel_ids_or_full[0], None, None, None
                    else:
                        tok_t, p_sub, idx_sub, q_sel = sample_from_top_scaled(
                            sel_scores, sel_ids_or_full, warper, generator
                        )
            else:
                if adapter_active:
                    avail = min(D_hist, L_ctx + i)
                    prev_ids = ctx_buf[L_ctx + i - offsets_all[:avail]]
                    score = snapshot.score_slot_tensor(
                        l_unit[i], U[i], offsets_all[:avail, None], prev_ids,
                        beta=beta, timer=timer, contiguous_offsets=True, bf16_gemm=bf16_gemm)
                else:
                    score = l_unit[i]
                if not bigram_inactive:
                    with timer.phase("bigram_score"):
                        prev_tok_t = ctx_buf[L_ctx + i - 1] if (L_ctx + i) > 0 else None
                        score = bigram.score_tensor(score, prev_tok_t)
                with timer.phase("sequential_sampling"):
                    tok_t, p_sub, idx_sub, q_sel = sample_token_compact_with_prob(
                        score, temperature, warper, generator, compact_probs_fn=compact_probs_fn
                    )

            ctx_buf[L_ctx + i] = tok_t
            if p_sub is not None:
                if compact_k:
                    if slot_probs is None:
                        slot_probs = torch.empty((K, compact_k), dtype=p_sub.dtype, device=device)
                        q_selected_buf = torch.empty((K,), dtype=p_sub.dtype, device=device)
                    slot_probs[i] = p_sub
                    slot_indices[i] = idx_sub
                    q_selected_buf[i] = q_sel
                else:
                    full_qs.append(p_sub)
        if temperature == 0:
            q_out = None
        elif compact_k:
            if slot_probs is None:
                q_out = None
            elif _allow_compact_q:
                q_out = CompactProposal(slot_indices, slot_probs, q_selected_buf, V)
            else:
                with timer.phase("sequential_sampling"):
                    q_out = torch.zeros((K, V), dtype=slot_probs.dtype, device=device)
                    q_out.scatter_(-1, slot_indices, slot_probs)
        else:
            q_out = torch.stack(full_qs) if full_qs else None
        token_tensor = ctx_buf[L_ctx:L_ctx + K]
        return (token_tensor if _return_token_tensor else token_tensor.tolist()), q_out, l_unit, H
    tokens, qs = [], []
    for i in range(K):
        score = (snapshot.score_slot(l_unit[i], U[i] if U is not None else None, prefix, tokens, i, beta=beta,
                                     timer=timer, bf16_gemm=bf16_gemm)
                 if snapshot is not None else l_unit[i])
        if bigram is not None:
            with timer.phase("bigram_score"):
                previous = tokens[-1] if tokens else (prefix[-1] if prefix else None)
                score = bigram.score(score, previous)
        with timer.phase("sequential_sampling"):
            token, q_i = sample_token(score, temperature, warper, generator, probs_fn=probs_fn)
        tokens.append(token)
        if q_i is not None:
            qs.append(q_i)
    out_tokens = torch.tensor(tokens, dtype=torch.long, device=draft_logits.device) if _return_token_tensor else tokens
    return out_tokens, torch.stack(qs) if qs else None, l_unit, H


def _record(*, round_idx, offset, input_context_length, prefix, draft_tokens, q, target_logits,
            draft_logits, h_raw, H, verification, snapshot, temperature, warper, head_metadata,
            l_unit=None, need_training_tensors=True):
    K = len(draft_tokens)
    return {
        "round_idx": round_idx,
        "absolute_output_offset": offset,
        "input_context_length": input_context_length,
        "verified_prefix_tail": list(prefix[-max(4, len(snapshot.A) if snapshot is not None else 4):]),
        "prefix": list(prefix),
        "physical_to_logical": {"anchor_physical_index": 0, "logical_to_physical": list(range(1, K + 1))},
        "draft_tokens": list(draft_tokens),
        "draft_logits": draft_logits.detach() if need_training_tensors else None,
        "l_unit": l_unit.detach() if l_unit is not None and need_training_tensors else None,
        "target_logits": target_logits[:K].detach() if need_training_tensors else None,
        "target_bonus_logits": target_logits[K].detach() if need_training_tensors else None,
        "h_raw_bf16": h_raw.detach().to(torch.bfloat16) if need_training_tensors else None,
        "h_raw": h_raw.detach() if need_training_tensors else None,
        "h": H.detach() if H is not None and need_training_tensors else None,
        "draft_lm_head": head_metadata,
        "accepted_length": verification.accepted_length,
        "first_rejection_index": verification.first_rejection,
        "replacement_token": verification.replacement_token,
        "bonus_token": verification.bonus_token,
        "decisions": verification.decisions,
        "proposal_token_probs": verification.proposal_token_probs,
        "target_token_probs": verification.target_token_probs,
        "adapter_version": getattr(snapshot, "version", 0),
        "proposal_mode": "dense_full" if snapshot is not None else "plain_dflash",
        "verification_rule": verification.verification_rule,
        "collection_temperature": temperature,
        "loss_temperature": 1.,
        "sampling_warp_config": warper.to_dict(),
        "off_policy_teacher_positions": list(range(verification.first_rejection + 1, K)),
    }


def _finish_round(record, *, snapshot, online_state, bigram, committed, on_block,
                  retain_tensors, timer):
    if on_block is not None:
        on_block(record)
    version = getattr(snapshot, "version", 0)
    with timer.phase("online_update"):
        if online_state is not None:
            online_state.update(record["round_idx"], record["l_unit"], record["target_logits"],
                                record["h"], record["prefix"], record["draft_tokens"],
                                first_rejection=record["first_rejection_index"])
        if bigram is not None:
            bigram.update(record["prefix"], committed)
    if getattr(snapshot, "version", 0) != version:
        raise RuntimeError("Active proposal snapshot version changed during round")
    record["committed_tokens"] = list(committed)
    record["next_adapter_version"] = getattr(online_state, "version", version)
    if retain_tensors:
        return record
    tensor_fields = {"draft_logits", "l_unit", "target_logits", "target_bonus_logits", "h_raw_bf16", "h_raw", "h", "prefix"}
    return {key: value for key, value in record.items() if key not in tensor_fields}


def generate_reference(
    backend, input_ids: torch.Tensor, max_new_tokens: int, *, temperature: float = 0.,
    warper: SamplingWarper = SamplingWarper(), generator: torch.Generator | None = None,
    adapter=None, online_state=None, projection=None, bigram=None, beta: float = 1.,
    block_size: int | None = None, stop_token_ids=None, on_block: Callable | None = None,
    retain_tensors: bool = False, measure_phases: bool = False, fastpath=None,
) -> GenerationResult:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("The reference loop requires one nonempty unpadded prompt")
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("Invalid generation length/temperature")
    block_size = backend.block_size if block_size is None else block_size
    if block_size < 2:
        raise ValueError("DFlash block_size must include an anchor and at least one proposal")
    if input_ids.shape[1] + max_new_tokens > backend.target.config.max_position_embeddings:
        raise ValueError("Requested prompt plus generation exceeds model context window")
    if adapter is not None and online_state is not None:
        raise ValueError("Pass adapter or online_state, not both")
    device = input_ids.device
    probs_fn = None if fastpath is None else fastpath.probabilities
    timer = PhaseTimer(device, measure_phases)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begun = time.perf_counter()
    n = input_ids.shape[1]
    if max_new_tokens == 0:
        return GenerationResult(input_ids.clone(), n, [], 0., {})
    max_length = n + max_new_tokens
    output_ids = torch.full((1, max_length + 1), backend.drafter.mask_token_id,
                            dtype=torch.long, device=device)
    output_ids[:, :n] = input_ids
    positions = torch.arange(max_length + 1, device=device)[None]
    target_cache, draft_cache = make_cache(backend.target.config), make_cache(backend.drafter.config)
    with torch.no_grad(), timer.phase("target_feature_extraction"):
        output = backend.target(input_ids, position_ids=positions[:, :n], past_key_values=target_cache,
                                use_cache=True, logits_to_keep=1, output_hidden_states=True)
        target_hidden = extract_context_feature(output.hidden_states, backend.drafter.target_layer_ids)
    token, _ = sample_token(output.logits[0, -1], temperature, warper, generator, probs_fn=probs_fn)
    output_ids[0, n] = token
    crop_cache(target_cache, n)
    prompt_tokens = input_ids[0].tolist()
    if bigram is not None:
        # The prefill's first target token is already committed, and its
        # transition from the last prompt token belongs to the request table.
        bigram.update(prompt_tokens, [token])
    committed_history = prompt_tokens + [token]
    stops = set(backend.stop_token_ids if stop_token_ids is None else stop_token_ids)
    stopped = token in stops
    start, round_idx, blocks = n, 0, []
    head_metadata = backend.head_metadata()
    static_snapshot = adapter.snapshot() if adapter is not None else None
    need_training_tensors = on_block is not None or retain_tensors or online_state is not None
    full_prefix_needed = on_block is not None or retain_tensors
    while start + 1 < max_length and not stopped:
        round_idx += 1
        verify_size = min(block_size, max_length - start)
        canvas = output_ids[:, start:start + verify_size].clone()
        snapshot = online_state.snapshot() if online_state is not None else static_snapshot
        history_depth = max(4, len(snapshot.A) if snapshot is not None else 1)
        prefix = list(committed_history) if full_prefix_needed else committed_history[-history_depth:]
        with torch.no_grad(), timer.phase("dflash_draft"):
            pos_slice = positions[:, start - target_hidden.shape[1]:start + verify_size]
            if fastpath is not None and fastpath.config.compile_drafter:
                h_raw, draft_logits = fastpath.draft_block(
                    backend, canvas, target_hidden, pos_slice, draft_cache, start
                )
            else:
                noise = F.embedding(canvas, backend.target.get_input_embeddings().weight) * backend.input_embedding_scale
                h_raw = backend.drafter(
                    target_hidden=target_hidden, noise_embedding=noise,
                    position_ids=pos_slice,
                    past_key_values=draft_cache, use_cache=True,
                )[0, 1:, :]
                crop_cache(draft_cache, start)
                draft_logits = backend.drafter.compute_logits(h_raw[None], backend.draft_lm_head)[0]
        with torch.no_grad():
            token_tensor, q, l_unit, H = _sample_proposal(
                draft_logits, h_raw, prefix, temperature=temperature, warper=warper,
                generator=generator, snapshot=snapshot, projection=projection, beta=beta,
                bigram=bigram, timer=timer, need_training_tensors=need_training_tensors,
                fastpath=fastpath, _return_token_tensor=True, _allow_compact_q=True,
            )
        canvas[0, 1:] = token_tensor
        with torch.no_grad(), timer.phase("target_verification"):
            output = backend.target(canvas, position_ids=positions[:, start:start + verify_size],
                                    past_key_values=target_cache, use_cache=True, output_hidden_states=True)
            tokens = token_tensor.tolist()
            verification = verify_block(token_tensor, q, output.logits[0], temperature, warper, generator,
                                        probs_fn=probs_fn)
        emitted = verification.emitted_tokens[:max_length - start - 1]
        for i, item in enumerate(emitted):
            if item in stops:
                emitted = emitted[:i + 1]
                stopped = True
                break
        output_ids[0, start + 1:start + 1 + len(emitted)] = input_ids.new_tensor(emitted)
        committed_history.extend(emitted)
        record = _record(
            round_idx=round_idx, offset=start + 1 - n, input_context_length=start + 1, prefix=prefix,
            draft_tokens=tokens, q=q, target_logits=output.logits[0], draft_logits=draft_logits,
            h_raw=h_raw, H=H, verification=verification, snapshot=snapshot, temperature=temperature,
            warper=warper, head_metadata=head_metadata, l_unit=l_unit, need_training_tensors=need_training_tensors,
        )
        start += len(emitted)
        crop_cache(target_cache, start)
        with torch.no_grad(), timer.phase("target_feature_extraction"):
            target_hidden = extract_context_feature(output.hidden_states, backend.drafter.target_layer_ids)[:, :len(emitted)]
        record["target_cache_length_after_rollback"] = int(target_cache.get_seq_length())
        blocks.append(_finish_round(record, snapshot=snapshot, online_state=online_state, bigram=bigram,
                                    committed=emitted, on_block=on_block, retain_tensors=retain_tensors, timer=timer))
    phases = timer.finish()
    return GenerationResult(output_ids[:, :min(start + 1, max_length)].clone(), n, blocks,
                            time.perf_counter() - begun, phases)


def generate_toy(
    target, drafter, prefix, max_new_tokens: int, *, block_length: int = 4,
    temperature: float = 0., warper: SamplingWarper = SamplingWarper(), generator=None,
    adapter=None, online_state=None, projection=None, bigram=None, beta=1.,
    on_block=None, retain_tensors=False, measure_phases=False, fastpath=None,
) -> GenerationResult:
    """CPU smoke loop with exactly the same proposal/verification/update helpers."""
    if target.embedding.device.type != "cpu":
        raise ValueError("The built-in smoke backend must run on CPU")
    if max_new_tokens < 0 or block_length < 1:
        raise ValueError("Invalid smoke generation lengths")
    if adapter is not None and online_state is not None:
        raise ValueError("Pass adapter or online_state, not both")
    context, output, blocks = list(prefix), [], []
    n = len(context)
    probs_fn = None if fastpath is None else fastpath.probabilities
    timer = PhaseTimer("cpu", measure_phases)
    begun = time.perf_counter()
    static_snapshot = adapter.snapshot() if adapter is not None else None
    need_training_tensors = on_block is not None or retain_tensors or online_state is not None
    while len(output) < max_new_tokens:
        K = min(block_length, max_new_tokens - len(output))
        snapshot = online_state.snapshot() if online_state is not None else static_snapshot
        history_depth = max(4, len(snapshot.A) if snapshot is not None else 1)
        proposal_prefix = context if on_block is not None or retain_tensors else context[-history_depth:]
        with timer.phase("dflash_draft"):
            drafted = drafter.draft(context, K)
        with torch.no_grad():
            tokens, q, l_unit, H = _sample_proposal(
                drafted.logits, drafted.h_raw, proposal_prefix, temperature=temperature, warper=warper,
                generator=generator, snapshot=snapshot, projection=projection, beta=beta,
                bigram=bigram, timer=timer, need_training_tensors=need_training_tensors,
                fastpath=fastpath, _allow_compact_q=True,
            )
        with timer.phase("target_verification"):
            target_logits = target.verify_logits(context, tokens)
            verification = verify_block(tokens, q, target_logits, temperature, warper, generator,
                                        probs_fn=probs_fn)
        emitted = verification.emitted_tokens[:max_new_tokens - len(output)]
        record = _record(
            round_idx=len(blocks) + 1, offset=len(output), input_context_length=len(context), prefix=proposal_prefix,
            draft_tokens=tokens, q=q, target_logits=target_logits, draft_logits=drafted.logits,
            h_raw=drafted.h_raw, H=H, verification=verification, snapshot=snapshot,
            temperature=temperature, warper=warper, head_metadata=drafter.head_metadata(),
            l_unit=l_unit, need_training_tensors=need_training_tensors,
        )
        blocks.append(_finish_round(record, snapshot=snapshot, online_state=online_state, bigram=bigram,
                                    committed=emitted, on_block=on_block, retain_tensors=retain_tensors, timer=timer))
        context.extend(emitted)
        output.extend(emitted)
    return GenerationResult(torch.tensor([context], dtype=torch.long), n, blocks,
                            time.perf_counter() - begun, timer.finish())
