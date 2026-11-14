"""M5 external speculative baselines, adaptive block control, and OpenAI server client.

Implements full non-placeholder M5 closed-loop arms from prompt.md:
  1. Full tree-based Token Recycling (adjacency matrix + multi-branch draft tree +
     single-pass tree attention mask + exact greedy/stochastic multi-branch tree
     verification + accepted-path KV-cache compaction).
  2. Test-Time Speculation (TTS): per-request online slot-temperature + hidden gate
     + token-prior calibration adapted via live unwarped target KL with zero offline fit.
  3. SharpDraft: entropy-guided draft sharpness modulation verified under the exact
     ratio rejection rule so target output distributions remain 100% exact.
  4. Adaptive Block-Length Control: per-round dynamic K_eff in [K_min..K_max] from
     EMA acceptance length and/or cumulative draft slot confidence, slicing both
     proposal and target verification lengths with exact KV-cache rollback.
  5. Qwen3.5 / SGLang / vLLM OpenAI-compatible HTTP server client for external
     serving engine comparisons (`--stage m5 --mode server_eval`).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import time
from typing import Callable, Sequence
import urllib.error
import urllib.request

import torch
from torch.nn import functional as F

from .adapter import full_kl, loss_weights
from .backend import crop_cache, extract_context_feature, make_cache
from .dflash_loop import GenerationResult, PhaseTimer
from .sampler import SamplingWarper, draw, probabilities, probability_dtype, sample_token
from .sparse import proximal_l1


# ---------------------------------------------------------------------------
# 1. Full Tree-Based Token Recycling
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    index: int
    token_id: int
    parent: int
    depth: int
    children: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TreeVerificationResult:
    emitted_tokens: list[int]
    accepted_node_indices: list[int]
    accepted_length: int
    first_rejection: int
    replacement_token: int | None
    bonus_token: int | None
    decisions: list[bool]
    verification_rule: str
    tree_size: int
    max_tree_depth: int


class TokenRecyclingState:
    """Per-token adjacency cache updated from target verification logits (Luo et al., 2024)."""

    def __init__(
        self,
        vocab_size: int,
        max_successors: int = 8,
        branch_widths: Sequence[int] = (4, 2, 1, 1),
    ):
        if vocab_size < 1 or max_successors < 1:
            raise ValueError('vocab_size and max_successors must be positive')
        if not branch_widths or any(w < 1 for w in branch_widths):
            raise ValueError('branch_widths must be nonempty positive integers')
        self.vocab_size = int(vocab_size)
        self.max_successors = int(max_successors)
        self.branch_widths = tuple(int(w) for w in branch_widths)
        self.adjacency = torch.full((self.vocab_size, self.max_successors), -1, dtype=torch.long)
        self.populated = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.version = 0
        self.recycled_positions = 0

    @torch.no_grad()
    def update_from_target_logits(self, conditioning_tokens: Sequence[int], target_logits: torch.Tensor) -> None:
        if len(conditioning_tokens) == 0:
            return
        if target_logits.ndim != 2 or len(target_logits) < len(conditioning_tokens):
            raise ValueError('target_logits must have one row per conditioning token')
        k = min(self.max_successors, target_logits.shape[-1])
        rows = target_logits[:len(conditioning_tokens)].detach().float()
        top_ids = torch.topk(rows, k, dim=-1).indices.to(device='cpu', dtype=torch.long)
        for idx, tok in enumerate(conditioning_tokens):
            t = int(tok)
            if not 0 <= t < self.vocab_size:
                raise ValueError('conditioning token outside vocabulary')
            self.adjacency[t, :k] = top_ids[idx]
            self.populated[t] = True
        self.recycled_positions += len(conditioning_tokens)
        self.version += 1

    def build_tree(
        self,
        root_token: int,
        max_draft_nodes: int,
        fallback_draft_logits: torch.Tensor | None = None,
    ) -> list[TreeNode]:
        if not 0 <= int(root_token) < self.vocab_size:
            raise ValueError('root_token outside vocabulary')
        if max_draft_nodes < 0:
            raise ValueError('max_draft_nodes must be nonnegative')
        root = TreeNode(index=0, token_id=int(root_token), parent=-1, depth=0, children=[])
        nodes = [root]
        if max_draft_nodes == 0:
            return nodes
        queue: deque[int] = deque([0])
        while queue and (len(nodes) - 1) < max_draft_nodes:
            u = queue.popleft()
            depth = nodes[u].depth
            width = self.branch_widths[depth] if depth < len(self.branch_widths) else 1
            remaining = max_draft_nodes - (len(nodes) - 1)
            limit = min(width, remaining)
            if limit <= 0:
                break
            chosen: list[int] = []
            seen: set[int] = set()
            tok_u = nodes[u].token_id
            if bool(self.populated[tok_u]):
                for cand in self.adjacency[tok_u].tolist():
                    if cand >= 0 and cand not in seen:
                        chosen.append(int(cand))
                        seen.add(int(cand))
                        if len(chosen) >= limit:
                            break
            if len(chosen) < limit and fallback_draft_logits is not None and depth < len(fallback_draft_logits):
                k_fb = min(self.vocab_size, limit + len(seen) + 4)
                fb_ids = torch.topk(fallback_draft_logits[depth].detach().float(), k_fb, dim=-1).indices.tolist()
                for cand in fb_ids:
                    if cand not in seen:
                        chosen.append(int(cand))
                        seen.add(int(cand))
                        if len(chosen) >= limit:
                            break
            if not chosen and u == 0:
                chosen.append(0)
            for cand in chosen[:limit]:
                child_idx = len(nodes)
                nodes.append(TreeNode(index=child_idx, token_id=cand, parent=u, depth=depth + 1, children=[]))
                nodes[u].children.append(child_idx)
                queue.append(child_idx)
        return nodes

    def metrics(self) -> dict:
        pop_count = int(self.populated.sum().item())
        return {
            'populated_tokens': pop_count,
            'populated_fraction': pop_count / self.vocab_size,
            'max_successors': self.max_successors,
            'branch_widths': list(self.branch_widths),
            'recycled_positions': self.recycled_positions,
            'version': self.version,
            'table_bytes': self.adjacency.numel() * self.adjacency.element_size(),
        }


def build_tree_attention_mask_and_positions(
    nodes: Sequence[TreeNode],
    prefix_length: int,
    *,
    device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct additive 4D `[1, 1, N, prefix_length + N]` mask and `[1, N]` position_ids."""
    n_nodes = len(nodes)
    if n_nodes < 1 or prefix_length < 0:
        raise ValueError('invalid tree size or prefix_length')
    anc = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)
    depths = []
    for u, node in enumerate(nodes):
        if node.index != u:
            raise ValueError('tree nodes must be indexed in topological BFS order 0..N-1')
        if u == 0:
            if node.parent != -1 or node.depth != 0:
                raise ValueError('root node 0 must have parent=-1 and depth=0')
            anc[0, 0] = True
        else:
            if not 0 <= node.parent < u:
                raise ValueError('every non-root tree node must follow its parent')
            anc[u] = anc[node.parent]
            anc[u, u] = True
        depths.append(prefix_length + node.depth)
    mask_dtype = dtype if dtype.is_floating_point else torch.float32
    neg_inf = torch.finfo(mask_dtype).min
    mask = torch.zeros((1, 1, n_nodes, prefix_length + n_nodes), dtype=mask_dtype, device=device)
    tree_slice = torch.where(anc, torch.zeros((), dtype=mask_dtype, device=device),
                             torch.full((), neg_inf, dtype=mask_dtype, device=device))
    mask[0, 0, :, prefix_length:] = tree_slice
    position_ids = torch.tensor([depths], dtype=torch.long, device=device)
    return mask, position_ids


def compact_tree_cache(cache, start: int, accepted_node_indices: Sequence[int]) -> None:
    """Gather KV entries of the accepted root-to-node path into contiguous linear slots."""
    keep_len = len(accepted_node_indices)
    if keep_len == 0:
        crop_cache(cache, start)
        return
    src = [start + int(u) for u in accepted_node_indices]
    for layer in getattr(cache, 'layers', []):
        index_t = torch.tensor(src, dtype=torch.long, device=layer.keys.device)
        gathered_k = layer.keys.index_select(2, index_t)
        gathered_v = layer.values.index_select(2, index_t)
        layer.keys[:, :, start:start + keep_len, :] = gathered_k
        layer.values[:, :, start:start + keep_len, :] = gathered_v
    crop_cache(cache, start + keep_len)


def verify_token_tree(
    nodes: Sequence[TreeNode],
    target_logits: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
) -> TreeVerificationResult:
    """Exact greedy (T=0) and multi-branch residual ratio verification (T>0) over a draft tree.

    Theorem (exactness at T > 0): At internal node `u` with distinct child candidate
    tokens `x_1, ..., x_m` and target conditional `P(. | path(u))`, testing each
    singleton child sequentially with acceptance probability `P_rem(x_j)` and updating
    `P_rem <- normalize(P_rem * I[v != x_j])` upon rejection yields unconditional
    marginal probability `P(x_j | path(u))` for accepting child `j`, and drawing
    `t_final ~ P_rem` when all children fail yields `P(v | path(u))` for every
    `v not in {x_1, ..., x_m}`. Hence the emitted sequence has the exact target law.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    n_nodes = len(nodes)
    if n_nodes < 1 or target_logits.ndim != 2 or target_logits.shape[0] != n_nodes:
        raise ValueError('target_logits must have shape [len(nodes), V]')
    V = target_logits.shape[1]
    for node in nodes:
        if not 0 <= node.token_id < V:
            raise ValueError('tree node token_id outside vocabulary')
    max_depth = max(node.depth for node in nodes)

    if temperature == 0:
        curr = 0
        accepted_nodes = [0]
        accepted_tokens: list[int] = []
        decisions: list[bool] = []
        while True:
            t_star = int(target_logits[curr].argmax(-1).item())
            matched = None
            for child_idx in nodes[curr].children:
                if nodes[child_idx].token_id == t_star:
                    matched = child_idx
                    break
            if matched is not None:
                accepted_nodes.append(matched)
                accepted_tokens.append(t_star)
                decisions.append(True)
                curr = matched
                continue
            has_children = len(nodes[curr].children) > 0
            if has_children:
                decisions.append(False)
            accepted_len = len(accepted_tokens)
            return TreeVerificationResult(
                emitted_tokens=accepted_tokens + [t_star],
                accepted_node_indices=accepted_nodes,
                accepted_length=accepted_len,
                first_rejection=accepted_len,
                replacement_token=t_star if has_children else None,
                bonus_token=None if has_children else t_star,
                decisions=decisions,
                verification_rule='greedy_tree',
                tree_size=n_nodes - 1,
                max_tree_depth=max_depth,
            )

    P_all = probabilities(target_logits, temperature, warper)
    P_all = probability_dtype(P_all)
    curr = 0
    accepted_nodes = [0]
    accepted_tokens = []
    decisions = []
    while True:
        P_rem = P_all[curr].clone()
        children = nodes[curr].children
        seen_sibling_tokens: set[int] = set()
        advanced = False
        for child_idx in children:
            tok = nodes[child_idx].token_id
            if tok in seen_sibling_tokens:
                continue
            seen_sibling_tokens.add(tok)
            p_tok = float(P_rem[tok].item())
            u_draw = float(torch.rand((1,), device=P_rem.device, dtype=P_rem.dtype, generator=generator).item())
            if u_draw < p_tok:
                accepted_nodes.append(child_idx)
                accepted_tokens.append(tok)
                decisions.append(True)
                curr = child_idx
                advanced = True
                break
            P_rem[tok] = 0.0
            rem_total = P_rem.sum()
            if not bool(torch.isfinite(rem_total)) or not bool(rem_total > 0):
                # Numerical guard when p_tok == 1 up to machine epsilon.
                accepted_nodes.append(child_idx)
                accepted_tokens.append(tok)
                decisions.append(True)
                curr = child_idx
                advanced = True
                break
            P_rem = P_rem / rem_total
        if advanced:
            continue
        final_tok = draw(P_rem, generator)
        has_children = len(children) > 0
        if has_children:
            decisions.append(False)
        accepted_len = len(accepted_tokens)
        return TreeVerificationResult(
            emitted_tokens=accepted_tokens + [final_tok],
            accepted_node_indices=accepted_nodes,
            accepted_length=accepted_len,
            first_rejection=accepted_len,
            replacement_token=final_tok if has_children else None,
            bonus_token=None if has_children else final_tok,
            decisions=decisions,
            verification_rule='ratio_tree',
            tree_size=n_nodes - 1,
            max_tree_depth=max_depth,
        )


def _node_prefix_tokens(nodes: Sequence[TreeNode], u: int) -> list[int]:
    chain: list[int] = []
    curr = u
    while curr > 0:
        chain.append(nodes[curr].token_id)
        curr = nodes[curr].parent
    chain.reverse()
    return chain


def generate_token_recycling_toy(
    target,
    drafter,
    prefix: Sequence[int],
    max_new_tokens: int,
    *,
    block_length: int = 4,
    temperature: float = 0.0,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
    recycling_state: TokenRecyclingState | None = None,
    max_successors: int = 4,
    branch_widths: Sequence[int] = (2, 2, 1, 1),
    use_dflash_fallback: bool = True,
    measure_phases: bool = False,
) -> tuple[GenerationResult, TokenRecyclingState]:
    """Closed-loop full tree-based Token Recycling on the CPU ToyTarget/ToyParallelDrafter."""
    if target.embedding.device.type != 'cpu':
        raise ValueError('Toy Token Recycling loop must run on CPU')
    if max_new_tokens < 0 or block_length < 1:
        raise ValueError('invalid generation lengths')
    state = recycling_state or TokenRecyclingState(
        target.vocab_size, max_successors=max_successors, branch_widths=branch_widths
    )
    context = list(prefix)
    if not context:
        raise ValueError('prefix must be nonempty')
    output: list[int] = []
    blocks: list[dict] = []
    n = len(context)
    timer = PhaseTimer('cpu', measure_phases)
    begun = time.perf_counter()
    head_meta = drafter.head_metadata()
    while len(output) < max_new_tokens:
        budget = min(block_length, max_new_tokens - len(output))
        with timer.phase('dflash_draft'):
            fb_logits = drafter.draft(context, budget).logits if use_dflash_fallback else None
            nodes = state.build_tree(context[-1], budget, fallback_draft_logits=fb_logits)
        with timer.phase('target_verification'):
            # Compute exact target conditional logits for every node in the tree.
            node_logits = []
            for u in range(len(nodes)):
                rel_path = _node_prefix_tokens(nodes, u)
                full_ctx = context + rel_path
                # verify_logits(ctx, []) returns shape [1, V] conditioned on ctx.
                logits_u = target.verify_logits(full_ctx, [])[0]
                node_logits.append(logits_u)
            tree_target_logits = torch.stack(node_logits, dim=0)
            ver = verify_token_tree(nodes, tree_target_logits, temperature, warper, generator)
        emitted = ver.emitted_tokens[:max_new_tokens - len(output)]
        with timer.phase('online_update'):
            state.update_from_target_logits([node.token_id for node in nodes], tree_target_logits)
        path_draft = [nodes[u].token_id for u in ver.accepted_node_indices[1:]]
        if len(path_draft) < ver.tree_size and ver.tree_size > 0:
            # Pad representative draft path display with first rejected child if any for bin accounting.
            last_node = nodes[ver.accepted_node_indices[-1]]
            if last_node.children:
                path_draft = path_draft + [nodes[last_node.children[0]].token_id]
        blocks.append({
            'round_idx': len(blocks) + 1,
            'absolute_output_offset': len(output),
            'input_context_length': len(context),
            'verified_prefix_tail': list(context[-4:]),
            'physical_to_logical': {'anchor_physical_index': 0, 'logical_to_physical': list(range(1, len(path_draft) + 1))},
            'draft_tokens': path_draft,
            'tree_draft_nodes': ver.tree_size,
            'max_tree_depth': ver.max_tree_depth,
            'draft_lm_head': head_meta,
            'accepted_length': ver.accepted_length,
            'first_rejection_index': ver.first_rejection,
            'replacement_token': ver.replacement_token,
            'bonus_token': ver.bonus_token,
            'decisions': ver.decisions,
            'proposal_token_probs': None,
            'target_token_probs': None,
            'adapter_version': state.version - 1,
            'next_adapter_version': state.version,
            'proposal_mode': 'token_recycling_tree',
            'verification_rule': ver.verification_rule,
            'collection_temperature': temperature,
            'loss_temperature': 1.0,
            'sampling_warp_config': warper.to_dict(),
            'off_policy_teacher_positions': [],
            'committed_tokens': list(emitted),
        })
        context.extend(emitted)
        output.extend(emitted)
    return (
        GenerationResult(torch.tensor([context], dtype=torch.long), n, blocks,
                         time.perf_counter() - begun, timer.finish()),
        state,
    )


def generate_token_recycling_reference(
    backend,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
    block_size: int | None = None,
    stop_token_ids=None,
    recycling_state: TokenRecyclingState | None = None,
    max_successors: int = 8,
    branch_widths: Sequence[int] = (4, 2, 1, 1),
    measure_phases: bool = False,
) -> tuple[GenerationResult, TokenRecyclingState]:
    """Closed-loop full tree-based Token Recycling on ReferenceBackend with tree attention & KV compaction."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('The reference loop requires one nonempty unpadded prompt')
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError('Invalid generation length/temperature')
    block_size = backend.block_size if block_size is None else block_size
    if block_size < 2:
        raise ValueError('block_size must include an anchor and at least one draft node')
    device = input_ids.device
    vocab_size = backend.target.config.vocab_size
    state = recycling_state or TokenRecyclingState(
        vocab_size, max_successors=max_successors, branch_widths=branch_widths
    )
    timer = PhaseTimer(device, measure_phases)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    begun = time.perf_counter()
    n = input_ids.shape[1]
    if max_new_tokens == 0:
        return GenerationResult(input_ids.clone(), n, [], 0.0, {}), state
    max_length = n + max_new_tokens
    output_ids = torch.full((1, max_length + 1), backend.drafter.mask_token_id, dtype=torch.long, device=device)
    output_ids[:, :n] = input_ids
    positions = torch.arange(max_length + block_size + 1, device=device)[None]
    target_cache, draft_cache = make_cache(backend.target.config), make_cache(backend.drafter.config)
    with torch.no_grad(), timer.phase('target_feature_extraction'):
        prefill = backend.target(
            input_ids, position_ids=positions[:, :n], past_key_values=target_cache,
            use_cache=True, logits_to_keep=1, output_hidden_states=True,
        )
        target_hidden = extract_context_feature(prefill.hidden_states, backend.drafter.target_layer_ids)
    first_tok, _ = sample_token(prefill.logits[0, -1], temperature, warper, generator)
    output_ids[0, n] = first_tok
    crop_cache(target_cache, n)
    with timer.phase('online_update'):
        state.update_from_target_logits([int(input_ids[0, n - 1].item())], prefill.logits[0, -1:])
    stops = set(backend.stop_token_ids if stop_token_ids is None else stop_token_ids)
    stopped = first_tok in stops
    start, round_idx, blocks = n, 0, []
    head_metadata = backend.head_metadata()
    while start + 1 < max_length and not stopped:
        round_idx += 1
        verify_budget = min(block_size, max_length - start)
        max_draft_nodes = verify_budget - 1
        anchor_token = int(output_ids[0, start].item())
        with torch.no_grad(), timer.phase('dflash_draft'):
            draft_canvas = output_ids[:, start:start + verify_budget].clone()
            noise = F.embedding(draft_canvas, backend.target.get_input_embeddings().weight) * backend.input_embedding_scale
            h_raw = backend.drafter(
                target_hidden=target_hidden, noise_embedding=noise,
                position_ids=positions[:, start - target_hidden.shape[1]:start + verify_budget],
                past_key_values=draft_cache, use_cache=True,
            )[0, 1:, :]
            crop_cache(draft_cache, start)
            fb_logits = backend.drafter.compute_logits(h_raw[None], backend.draft_lm_head)[0]
            nodes = state.build_tree(anchor_token, max_draft_nodes, fallback_draft_logits=fb_logits)
        tree_tokens = torch.tensor([[node.token_id for node in nodes]], dtype=torch.long, device=device)
        tree_mask, tree_pos = build_tree_attention_mask_and_positions(
            nodes, start, device=device, dtype=backend.target.dtype
        )
        with torch.no_grad(), timer.phase('target_verification'):
            out = backend.target(
                tree_tokens, attention_mask=tree_mask, position_ids=tree_pos,
                past_key_values=target_cache, use_cache=True, output_hidden_states=True,
            )
            ver = verify_token_tree(nodes, out.logits[0], temperature, warper, generator)
        emitted = ver.emitted_tokens[:max_length - start - 1]
        for i, item in enumerate(emitted):
            if item in stops:
                emitted = emitted[:i + 1]
                stopped = True
                break
        output_ids[0, start + 1:start + 1 + len(emitted)] = input_ids.new_tensor(emitted)
        kept_nodes = ver.accepted_node_indices[:len(emitted)]
        compact_tree_cache(target_cache, start, kept_nodes)
        start += len(emitted)
        with torch.no_grad(), timer.phase('target_feature_extraction'):
            extracted = extract_context_feature(out.hidden_states, backend.drafter.target_layer_ids)
            node_idx_t = torch.tensor(kept_nodes, dtype=torch.long, device=device)
            target_hidden = extracted.index_select(1, node_idx_t)
        with timer.phase('online_update'):
            state.update_from_target_logits([node.token_id for node in nodes], out.logits[0])
        path_draft = [nodes[u].token_id for u in ver.accepted_node_indices[1:]]
        if len(path_draft) < ver.tree_size and ver.tree_size > 0:
            last_node = nodes[ver.accepted_node_indices[-1]]
            if last_node.children:
                path_draft = path_draft + [nodes[last_node.children[0]].token_id]
        blocks.append({
            'round_idx': round_idx,
            'absolute_output_offset': start - len(emitted) + 1 - n,
            'input_context_length': start - len(emitted) + 1,
            'verified_prefix_tail': output_ids[0, max(0, start - len(emitted) - 3):start - len(emitted) + 1].tolist(),
            'physical_to_logical': {'anchor_physical_index': 0, 'logical_to_physical': list(range(1, len(path_draft) + 1))},
            'draft_tokens': path_draft,
            'tree_draft_nodes': ver.tree_size,
            'max_tree_depth': ver.max_tree_depth,
            'draft_lm_head': head_metadata,
            'accepted_length': ver.accepted_length,
            'first_rejection_index': ver.first_rejection,
            'replacement_token': ver.replacement_token,
            'bonus_token': ver.bonus_token,
            'decisions': ver.decisions,
            'proposal_token_probs': None,
            'target_token_probs': None,
            'adapter_version': state.version - 1,
            'next_adapter_version': state.version,
            'proposal_mode': 'token_recycling_tree',
            'verification_rule': ver.verification_rule,
            'collection_temperature': temperature,
            'loss_temperature': 1.0,
            'sampling_warp_config': warper.to_dict(),
            'off_policy_teacher_positions': [],
            'committed_tokens': list(emitted),
            'target_cache_length_after_rollback': int(target_cache.get_seq_length()),
        })
    phases = timer.finish()
    return (
        GenerationResult(output_ids[:, :min(start + 1, max_length)].clone(), n, blocks,
                         time.perf_counter() - begun, phases),
        state,
    )


# ---------------------------------------------------------------------------
# 2. Test-Time Speculation (TTS)
# ---------------------------------------------------------------------------

class TTSState:
    """Zero-offline per-request Test-Time Speculation proposal calibrator.

    Adapts per-slot log inverse-temperature `log_inv_tau`, hidden confidence gate
    `w_gate`, and sparse-regularized token residual `db` online from target KL.
    """
    proposal_mode_name = 'tts'
    needs_hidden = True

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_slots: int = 32,
        *,
        lr: float = 0.01,
        update_stride: int = 1,
        lambda_l2: float = 0.01,
        lambda_l1_b: float = 1e-5,
        device='cpu',
    ):
        if vocab_size < 1 or hidden_dim < 1 or max_slots < 1 or update_stride < 1:
            raise ValueError('invalid TTS dimensions or stride')
        if lr < 0 or lambda_l2 < 0 or lambda_l1_b < 0:
            raise ValueError('TTS learning rate and regularizers must be nonnegative')
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.max_slots = int(max_slots)
        self.lr = float(lr)
        self.update_stride = int(update_stride)
        self.lambda_l2 = float(lambda_l2)
        self.lambda_l1_b = float(lambda_l1_b)
        self.device = torch.device(device)
        self.log_inv_tau = torch.zeros(self.max_slots, dtype=torch.float32, device=self.device)
        self.w_gate = torch.zeros(self.hidden_dim, dtype=torch.float32, device=self.device)
        self.db = torch.zeros(self.vocab_size, dtype=torch.float32, device=self.device)
        self.u = 0
        self.version = 0
        self.last_round = 0
        self.last_loss: float | None = None

    def _slot_scores(self, l_unit: torch.Tensor, H: torch.Tensor,
                     log_inv_tau: torch.Tensor, w_gate: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
        K = len(l_unit)
        if K > len(log_inv_tau):
            pad = log_inv_tau[-1:].expand(K - len(log_inv_tau))
            slot_param = torch.cat((log_inv_tau, pad), dim=0)
        else:
            slot_param = log_inv_tau[:K]
        gate = torch.tanh(H.float() @ w_gate)
        inv_tau = torch.exp((slot_param + 0.5 * gate).clamp(-1.5, 1.5))[:, None]
        return inv_tau * l_unit.float() + db[None, :]

    @torch.no_grad()
    def propose(self, l_unit: torch.Tensor, H: torch.Tensor, prefix: Sequence[int], *,
                temperature: float, warper: SamplingWarper, generator=None, timer=None):
        ctx = timer.phase('dense_adapter_score') if timer is not None else None
        if ctx is not None:
            ctx.__enter__()
        try:
            scores = self._slot_scores(l_unit, H, self.log_inv_tau, self.w_gate, self.db)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        tokens, qs = [], []
        samp_ctx = timer.phase('sequential_sampling') if timer is not None else None
        if samp_ctx is not None:
            samp_ctx.__enter__()
        try:
            for i in range(len(scores)):
                tok, q_i = sample_token(scores[i], temperature, warper, generator)
                tokens.append(tok)
                if q_i is not None:
                    qs.append(q_i)
        finally:
            if samp_ctx is not None:
                samp_ctx.__exit__(None, None, None)
        return tokens, (torch.stack(qs) if qs else None)

    def update(self, round_idx: int, l_unit: torch.Tensor, target_logits: torch.Tensor,
               H: torch.Tensor, prefix: Sequence[int], tokens: Sequence[int],
               first_rejection: int | None = None, committed_tokens=None) -> bool:
        if round_idx < 1 or round_idx <= self.last_round:
            raise ValueError('round_idx must increase and is one-based')
        self.last_round = round_idx
        if round_idx % self.update_stride != 0:
            return False
        with torch.inference_mode(False), torch.enable_grad():
            lu = l_unit.detach().float().clone()
            tl = target_logits.detach().float().clone()
            h_in = H.detach().float().clone()
            lit = self.log_inv_tau.detach().clone().requires_grad_(True)
            wg = self.w_gate.detach().clone().requires_grad_(True)
            db = self.db.detach().clone().requires_grad_(True)
            scores = self._slot_scores(lu, h_in, lit, wg, db)
            weights = loss_weights(len(lu), lu.device)
            loss = full_kl(scores, tl, weights)
            loss.backward()
            with torch.no_grad():
                shrink = 1.0 + self.lr * self.lambda_l2
                self.log_inv_tau = ((lit - self.lr * lit.grad) / shrink).detach()
                self.w_gate = ((wg - self.lr * wg.grad) / shrink).detach()
                self.db = proximal_l1((db - self.lr * db.grad) / shrink, self.lr * self.lambda_l1_b).detach()
                self.u += 1
                self.version += 1
                self.last_loss = float(loss.detach())
        return True

    def metrics(self) -> dict:
        return {
            'adapter_version': self.version,
            'u': self.u,
            'last_round': self.last_round,
            'log_inv_tau_norm': float(self.log_inv_tau.norm()),
            'w_gate_norm': float(self.w_gate.norm()),
            'db_norm': float(self.db.norm()),
            'db_exact_zero_fraction': float((self.db == 0).float().mean()),
            'last_full_loss': self.last_loss,
        }


# ---------------------------------------------------------------------------
# 3. SharpDraft Baseline
# ---------------------------------------------------------------------------

class SharpDraftProposal:
    """Entropy-aware draft distribution sharpening verified by the exact ratio rule."""
    proposal_mode_name = 'sharp_draft'
    needs_hidden = False

    def __init__(self, gamma_max: float = 1.35, gamma_min: float = 0.95, entropy_pivot: float = 0.35):
        if gamma_max <= 0 or gamma_min <= 0:
            raise ValueError('gamma_max and gamma_min must be strictly positive')
        if not 0 < entropy_pivot <= 1:
            raise ValueError('entropy_pivot must lie in (0, 1]')
        self.gamma_max = float(gamma_max)
        self.gamma_min = float(gamma_min)
        self.entropy_pivot = float(entropy_pivot)
        self.rounds = 0
        self.mean_gamma_sum = 0.0

    @torch.no_grad()
    def sharpen_scores(self, l_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lu = l_unit.float()
        V = lu.shape[-1]
        probs = lu.exp()
        entropy = -(probs * lu).sum(-1) / math.log(max(2, V))
        ratio = (entropy / self.entropy_pivot).clamp(0.0, 1.0)
        gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * ratio
        return gamma[:, None] * lu, gamma

    @torch.no_grad()
    def propose(self, l_unit: torch.Tensor, H: torch.Tensor | None, prefix: Sequence[int], *,
                temperature: float, warper: SamplingWarper, generator=None, timer=None):
        ctx = timer.phase('dense_adapter_score') if timer is not None else None
        if ctx is not None:
            ctx.__enter__()
        try:
            scores, gamma = self.sharpen_scores(l_unit)
            self.rounds += 1
            self.mean_gamma_sum += float(gamma.mean().item())
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        tokens, qs = [], []
        samp_ctx = timer.phase('sequential_sampling') if timer is not None else None
        if samp_ctx is not None:
            samp_ctx.__enter__()
        try:
            for i in range(len(scores)):
                tok, q_i = sample_token(scores[i], temperature, warper, generator)
                tokens.append(tok)
                if q_i is not None:
                    qs.append(q_i)
        finally:
            if samp_ctx is not None:
                samp_ctx.__exit__(None, None, None)
        return tokens, (torch.stack(qs) if qs else None)

    def metrics(self) -> dict:
        return {
            'gamma_max': self.gamma_max,
            'gamma_min': self.gamma_min,
            'entropy_pivot': self.entropy_pivot,
            'rounds': self.rounds,
            'mean_gamma': self.mean_gamma_sum / self.rounds if self.rounds else None,
        }


# ---------------------------------------------------------------------------
# 4. Adaptive Block-Length Controller
# ---------------------------------------------------------------------------

class AdaptiveBlockController:
    """Dynamic per-round block length K_eff in [min_block_length..max_block_length]."""
    POLICIES = ('ema_acceptance', 'confidence_threshold', 'hybrid')

    def __init__(
        self,
        min_block_length: int = 1,
        max_block_length: int = 15,
        *,
        policy: str = 'hybrid',
        ema_decay: float = 0.7,
        ema_multiplier: float = 1.5,
        ema_margin: float = 1.0,
        min_cumulative_confidence: float = 0.08,
    ):
        if min_block_length < 1 or max_block_length < min_block_length:
            raise ValueError('require 1 <= min_block_length <= max_block_length')
        if policy not in self.POLICIES:
            raise ValueError(f'policy must be one of {self.POLICIES}')
        if not 0 <= ema_decay < 1:
            raise ValueError('ema_decay must lie in [0, 1)')
        if ema_multiplier <= 0 or min_cumulative_confidence < 0:
            raise ValueError('invalid adaptive controller multipliers/thresholds')
        self.min_block_length = int(min_block_length)
        self.max_block_length = int(max_block_length)
        self.policy = policy
        self.ema_decay = float(ema_decay)
        self.ema_multiplier = float(ema_multiplier)
        self.ema_margin = float(ema_margin)
        self.min_cumulative_confidence = float(min_cumulative_confidence)
        self.ema_accepted: float | None = None
        self.requested_history: list[int] = []
        self.drafted_history: list[int] = []
        self.effective_history: list[int] = []
        self._pending_requested = 0
        self._pending_drafted = 0

    def predict_block_length(self, available_max_k: int) -> int:
        if available_max_k < 1:
            raise ValueError('available_max_k must be >= 1')
        eff_min = min(self.min_block_length, available_max_k)
        eff_max = min(self.max_block_length, available_max_k)
        if self.policy == 'confidence_threshold' or self.ema_accepted is None:
            chosen = eff_max
        else:
            predicted = int(math.ceil(self.ema_multiplier * self.ema_accepted + self.ema_margin))
            chosen = max(eff_min, min(eff_max, predicted))
        self._pending_requested = available_max_k
        self._pending_drafted = chosen
        return chosen

    @torch.no_grad()
    def truncate_from_logits(self, draft_logits: torch.Tensor) -> int:
        k_draft = len(draft_logits)
        if k_draft <= 1 or self.policy == 'ema_acceptance' or self.min_cumulative_confidence <= 0:
            return k_draft
        eff_min = min(self.min_block_length, k_draft)
        max_probs = draft_logits.detach().float().softmax(-1).max(-1).values
        cum_conf = torch.cumprod(max_probs, dim=0)
        below = (cum_conf[eff_min:] < self.min_cumulative_confidence).nonzero(as_tuple=True)[0]
        if len(below) > 0:
            return eff_min + int(below[0].item())
        return k_draft

    def observe(self, accepted_length: int, effective_k: int) -> None:
        if self.ema_accepted is None:
            self.ema_accepted = float(accepted_length)
        else:
            self.ema_accepted = self.ema_decay * self.ema_accepted + (1.0 - self.ema_decay) * float(accepted_length)
        self.requested_history.append(int(self._pending_requested or effective_k))
        self.drafted_history.append(int(self._pending_drafted or effective_k))
        self.effective_history.append(int(effective_k))

    def metrics(self) -> dict:
        rounds = len(self.effective_history)
        req_sum = sum(self.requested_history)
        eff_sum = sum(self.effective_history)
        return {
            'policy': self.policy,
            'min_block_length': self.min_block_length,
            'max_block_length': self.max_block_length,
            'rounds': rounds,
            'final_ema_accepted': self.ema_accepted,
            'mean_requested_k': req_sum / rounds if rounds else None,
            'mean_drafted_k': sum(self.drafted_history) / rounds if rounds else None,
            'mean_effective_k': eff_sum / rounds if rounds else None,
            'min_effective_k': min(self.effective_history) if rounds else None,
            'max_effective_k': max(self.effective_history) if rounds else None,
            'saved_verify_slots': req_sum - eff_sum,
        }


# ---------------------------------------------------------------------------
# 5. Qwen3.5 / SGLang / vLLM OpenAI-Compatible Server Client
# ---------------------------------------------------------------------------

class Qwen35ServerClient:
    """OpenAI-compatible HTTP client for Qwen3.5 / SGLang / vLLM DFlash serving benchmarks."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = 'default',
        api_key: str = 'EMPTY',
        timeout_seconds: float = 600.0,
        opener: Callable | None = None,
    ):
        if not base_url or not base_url.startswith(('http://', 'https://')):
            raise ValueError('base_url must be an http:// or https:// endpoint')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener or urllib.request.urlopen

    def chat_completion(
        self,
        prompt_text: str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 2048,
        seed: int = 2026,
        enable_thinking: bool = False,
        return_meta_info: bool = True,
        extra_body: dict | None = None,
    ) -> dict:
        payload: dict = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt_text}],
            'temperature': float(temperature),
            'top_p': float(top_p),
            'max_tokens': int(max_tokens),
            'seed': int(seed),
            'chat_template_kwargs': {'enable_thinking': bool(enable_thinking)},
            'return_meta_info': bool(return_meta_info),
        }
        if top_k > 0:
            payload['top_k'] = int(top_k)
        if extra_body:
            payload.update(extra_body)
        url = self.base_url if self.base_url.endswith('/chat/completions') else f'{self.base_url}/v1/chat/completions'
        if self.base_url.endswith('/v1'):
            url = f'{self.base_url}/chat/completions'
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST',
        )
        started = time.perf_counter()
        try:
            with self._opener(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode('utf-8')
        except urllib.error.URLError as exc:
            raise RuntimeError(f'OpenAI server request to {url} failed: {exc}') from exc
        elapsed = time.perf_counter() - started
        parsed = json.loads(raw)
        choice = (parsed.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        usage = parsed.get('usage') or {}
        meta = choice.get('meta_info') or parsed.get('meta_info') or {}
        completion_tokens = int(usage.get('completion_tokens') or meta.get('completion_tokens') or 0)
        prompt_tokens = int(usage.get('prompt_tokens') or meta.get('prompt_tokens') or 0)
        spec_verify_ct = meta.get('spec_verify_ct')
        spec_accept_reported = meta.get('spec_accept_length')
        # SGLang DFlash reports verify-inclusive accept length (1 + logical_accepted_length);
        # expose both raw server telemetry and logical accepted length clearly.
        if spec_accept_reported is not None:
            logical_accept = max(0.0, float(spec_accept_reported) - 1.0)
        elif spec_verify_ct and float(spec_verify_ct) > 0 and completion_tokens > 0:
            logical_accept = max(0.0, (completion_tokens / float(spec_verify_ct)) - 1.0)
        else:
            logical_accept = None
        return {
            'endpoint': url,
            'model': parsed.get('model', self.model),
            'content': message.get('content') or '',
            'reasoning_content': message.get('reasoning_content'),
            'finish_reason': choice.get('finish_reason'),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'elapsed_seconds': elapsed,
            'tokens_per_second': completion_tokens / elapsed if elapsed > 0 else 0.0,
            'server_meta_info': {
                'spec_verify_ct': spec_verify_ct,
                'spec_accept_length_reported_including_bonus': spec_accept_reported,
                'logical_mean_accepted_length': logical_accept,
            },
        }
