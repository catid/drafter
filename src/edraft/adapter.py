"""Causal energy correction and standalone synchronous lookahead mathematics."""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
import math
from typing import Sequence
import torch
from torch import nn


def prev_token(prefix: Sequence[int], realized: Sequence[int], i: int, d: int) -> int | None:
    if d < 1:
        raise ValueError('offset d must be positive; there is no self-interaction')
    k = i - d
    if k >= 0:
        if k >= len(realized):
            raise ValueError('required realized predecessor is unavailable')
        return int(realized[k])
    index = len(prefix) + k
    return int(prefix[index]) if index >= 0 else None


def top_candidates(scores: torch.Tensor, count: int) -> torch.Tensor:
    if count < 1:
        raise ValueError('candidate count must be positive')
    return torch.topk(scores, min(count, scores.shape[-1]), dim=-1).indices


def block_unary(E, W, b, H, *, bf16_gemm: bool = False, E_bf16: torch.Tensor | None = None):
    if bf16_gemm:
        e_bf = E.bfloat16() if E_bf16 is None else E_bf16
        return ((H.bfloat16() @ W.bfloat16().T) @ e_bf.T).to(b.dtype) + b
    return (H.to(W.dtype) @ W.T) @ E.to(W.dtype).T + b


def _phase(timer, name):
    return nullcontext() if timer is None else timer.phase(name)


def _pair_message(E: torch.Tensor, A: torch.Tensor, predecessors: list[tuple[int, int]],
                  pair_matrix: torch.Tensor | None = None) -> torch.Tensor:
    r = E.shape[1]
    avail = len(predecessors)
    prev_ids = torch.tensor([p for _, p in predecessors], device=A.device, dtype=torch.long)
    E_prev = E.index_select(0, prev_ids).to(A.dtype)
    if pair_matrix is not None and all(d == idx + 1 for idx, (d, _) in enumerate(predecessors)):
        return pair_matrix[:, :avail * r] @ E_prev.reshape(-1)
    if all(d == idx + 1 for idx, (d, _) in enumerate(predecessors)):
        A_sel = A[:avail]
    else:
        d_idx = torch.tensor([d - 1 for d, _ in predecessors], device=A.device, dtype=torch.long)
        A_sel = A.index_select(0, d_idx)
    return torch.bmm(A_sel, E_prev.unsqueeze(-1)).squeeze(-1).sum(0)


def slot_correction(E, A, U_i, l_unit_i, prefix, realized, i, sparse=None, M_sparse=256, beta=1.0, timer=None,
                    *, pair_matrix=None, bf16_gemm: bool = False, E_bf16: torch.Tensor | None = None):
    with _phase(timer, 'dense_pair_score'):
        predecessors = []
        for d in range(1, len(A) + 1):
            previous = prev_token(prefix, realized, i, d)
            if previous is not None:
                predecessors.append((d, previous))
        if predecessors:
            message = _pair_message(E, A, predecessors, pair_matrix=pair_matrix)
            if bf16_gemm:
                e_bf = E.bfloat16() if E_bf16 is None else E_bf16
                dense = U_i + (e_bf @ message.bfloat16()).to(U_i.dtype)
            else:
                dense = U_i + E.to(A.dtype) @ message
        else:
            dense = U_i
    if sparse is None or not predecessors:
        return dense
    with _phase(timer, 'sparse_lookup'):
        candidates = top_candidates(l_unit_i - beta * dense, M_sparse)
        # All logical offsets share one vectorized hash invocation for this slot.
        offsets = torch.tensor([d for d, _ in predecessors], device=dense.device, dtype=torch.long)[:, None]
        previous = torch.tensor([p for _, p in predecessors], device=dense.device, dtype=torch.long)[:, None]
        values = sparse.lookup(offsets, previous, candidates[None, :]).to(dense.dtype).sum(0)
        correction = dense.scatter_add(0, candidates, values)
    return correction


def block_predecessor_pairs(prefix: Sequence[int], tokens: Sequence[int], K: int, D: int, device):
    """Build valid (slot_idx, d_zero_based, prev_token_id) 1D int64 tensors in one host-to-device copy."""
    pairs = []
    for i in range(K):
        for d in range(1, D + 1):
            previous = prev_token(prefix, tokens, i, d)
            if previous is not None:
                pairs.append((i, d - 1, previous))
    if not pairs:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, empty
    packed = torch.tensor(pairs, device=device, dtype=torch.long)
    return packed[:, 0], packed[:, 1], packed[:, 2]


def block_correction(E, A, U, l_unit, prefix, tokens, sparse=None, M_sparse=256, beta=1.0,
                     *, bf16_gemm: bool = False, E_bf16: torch.Tensor | None = None):
    """Vectorized [K, V] causal correction across all slots and offsets 1..D."""
    K = len(U)
    valid_slots, valid_d0, valid_prev = block_predecessor_pairs(prefix, tokens, K, len(A), A.device)
    if valid_slots.numel() == 0:
        return U
    E_cast = E.to(A.dtype)
    E_prev = E_cast.index_select(0, valid_prev)
    msgs = torch.bmm(A.index_select(0, valid_d0), E_prev.unsqueeze(-1)).squeeze(-1)
    slot_msgs = torch.zeros(K, E.shape[1], dtype=A.dtype, device=A.device).index_add(0, valid_slots, msgs)
    if bf16_gemm:
        e_bf = E.bfloat16() if E_bf16 is None else E_bf16
        dense = U + (slot_msgs.bfloat16() @ e_bf.T).to(U.dtype)
    else:
        dense = U + slot_msgs @ E_cast.T
    if sparse is None:
        return dense
    candidates = top_candidates(l_unit - beta * dense, M_sparse)
    cands_pair = candidates.index_select(0, valid_slots)
    values_pair = sparse.lookup((valid_d0 + 1)[:, None], valid_prev[:, None], cands_pair).to(dense.dtype)
    slot_values = torch.zeros(K, candidates.shape[1], dtype=dense.dtype, device=dense.device).index_add(0, valid_slots, values_pair)
    return dense.scatter_add(1, candidates, slot_values)


@dataclass(frozen=True)
class ProposalSnapshot:
    E: torch.Tensor
    W: torch.Tensor
    A: torch.Tensor
    b: torch.Tensor
    version: int = 0
    sparse: object | None = None
    M_sparse: int = 256
    C0: tuple[torch.Tensor, ...] | None = None
    is_zero: bool = False
    _A_flat: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)
    _E_bf16: torch.Tensor | None = field(default=None, repr=False, compare=False, kw_only=True)
    _pair_zero: bool | None = field(default=None, repr=False, compare=False, kw_only=True)

    def pair_is_zero(self) -> bool:
        cached = self._pair_zero
        if cached is None:
            cached = not bool(self.A.count_nonzero())
            object.__setattr__(self, '_pair_zero', cached)
        return cached

    def _pair_matrix(self) -> torch.Tensor:
        cached = self._A_flat
        if cached is None:
            D, r, _ = self.A.shape
            cached = self.A.permute(1, 0, 2).contiguous().view(r, D * r)
            object.__setattr__(self, '_A_flat', cached)
        return cached

    def _token_features_bf16(self) -> torch.Tensor:
        cached = self._E_bf16
        if cached is None or cached.device != self.E.device:
            cached = self.E.bfloat16()
            object.__setattr__(self, '_E_bf16', cached)
        return cached

    @property
    def D(self):
        return len(self.A)

    @property
    def adapter_version(self):
        return self.version

    def unary(self, H, proposal_mode='dense_full', *, unary_fn=None, bf16_gemm: bool = False):
        if proposal_mode == 'dense_full':
            if unary_fn is not None:
                return unary_fn(self.E, self.W, self.b, H)
            return block_unary(
                self.E, self.W, self.b, H,
                bf16_gemm=bf16_gemm,
                E_bf16=self._token_features_bf16() if bf16_gemm else None,
            )
        if proposal_mode != 'dense_candidate' or self.C0 is None:
            raise ValueError('candidate unary requires frozen C0')
        hidden = H.to(self.W.dtype) @ self.W.T
        # Only candidate rows of E are read; no K-by-V dense GEMM occurs here.
        return [self.E[ids].to(self.W.dtype) @ hidden[i] + self.b[ids]
                for i, ids in enumerate(self.C0)]

    def score_slot(self, l_unit_i, U_i, prefix, realized, i, beta=1.0, proposal_mode='dense_full', timer=None,
                   *, bf16_gemm: bool = False):
        if proposal_mode == 'dense_candidate':
            if self.C0 is None:
                raise ValueError('dense_candidate requires the frozen C0 matched-support snapshot')
            ids = self.C0[i]
            if beta == 0 or self.is_zero:
                selected = l_unit_i[ids]
            else:
                if U_i.shape[-1] != len(ids):
                    raise ValueError('candidate scoring requires candidate unary rows; call unary(..., proposal_mode=dense_candidate)')
                unary = U_i
                with _phase(timer, 'dense_pair_score'):
                    predecessors = []
                    for d in range(1, len(self.A) + 1):
                        previous = prev_token(prefix, realized, i, d)
                        if previous is not None:
                            predecessors.append((d, previous))
                    if predecessors:
                        message = _pair_message(self.E, self.A, predecessors, pair_matrix=self._pair_matrix())
                        correction = unary + self.E[ids].to(self.A.dtype) @ message
                    else:
                        correction = unary
                if self.sparse is not None and predecessors:
                    with _phase(timer, 'sparse_lookup'):
                        offsets = torch.tensor([d for d, _ in predecessors], device=ids.device, dtype=torch.long)[:, None]
                        previous_ids = torch.tensor([p for _, p in predecessors], device=ids.device, dtype=torch.long)[:, None]
                        correction = correction + self.sparse.lookup(offsets, previous_ids, ids[None, :]).sum(0)
                selected = l_unit_i[ids] - beta * correction
            return torch.full_like(l_unit_i, -torch.inf).scatter(0, ids, selected)
        if proposal_mode != 'dense_full':
            raise ValueError('unknown proposal mode')
        if beta == 0 or self.is_zero:
            return l_unit_i
        correction = slot_correction(
            self.E, self.A, U_i, l_unit_i, prefix, realized, i,
            self.sparse, self.M_sparse, beta, timer=timer,
            pair_matrix=self._pair_matrix(),
            bf16_gemm=bf16_gemm,
            E_bf16=self._token_features_bf16() if bf16_gemm else None,
        )
        return l_unit_i - beta * correction

    def score_slot_tensor(self, l_unit_i, U_i, offsets: torch.Tensor, prev_ids: torch.Tensor,
                          beta=1.0, proposal_mode='dense_full', timer=None, *,
                          contiguous_offsets: bool = False, bf16_gemm: bool = False):
        """Sync-free causal slot scoring when offsets [avail] or [avail, 1] and prev_ids [avail] reside on device."""
        if proposal_mode != 'dense_full':
            raise ValueError('score_slot_tensor requires dense_full proposal mode')
        offs_1d = offsets.squeeze(-1) if offsets.ndim == 2 and offsets.shape[1] == 1 else offsets
        if offs_1d.ndim != 1 or prev_ids.ndim != 1 or offs_1d.numel() != prev_ids.numel():
            raise ValueError('offsets and prev_ids must be matching 1D predecessor tensors (or offsets [avail, 1])')
        avail = prev_ids.numel()
        if contiguous_offsets and avail > len(self.A):
            raise ValueError('contiguous_offsets length exceeds adapter history depth')
        if beta == 0 or self.is_zero:
            return l_unit_i
        with _phase(timer, 'dense_pair_score'):
            if avail and not self.pair_is_zero():
                r = self.E.shape[1]
                E_prev_2d = self.E.index_select(0, prev_ids).to(self.A.dtype)
                if contiguous_offsets:
                    message = self._pair_matrix()[:, :avail * r] @ E_prev_2d.reshape(-1)
                else:
                    A_sel = self.A.index_select(0, offs_1d - 1)
                    message = torch.bmm(A_sel, E_prev_2d.unsqueeze(-1)).squeeze(-1).sum(0)
                if bf16_gemm:
                    dense = U_i + (self._token_features_bf16() @ message.bfloat16()).to(U_i.dtype)
                else:
                    dense = U_i + self.E.to(self.A.dtype) @ message
            else:
                dense = U_i
        if self.sparse is None or avail == 0:
            correction = dense
        else:
            with _phase(timer, 'sparse_lookup'):
                candidates = top_candidates(l_unit_i - beta * dense, self.M_sparse)
                offs_2d = offs_1d[:, None]
                values = self.sparse.lookup(offs_2d, prev_ids[:, None], candidates[None, :]).to(dense.dtype).sum(0)
                correction = dense.scatter_add(0, candidates, values)
        return l_unit_i - beta * correction

    def precompute_sparse_pool(
        self,
        l_unit: torch.Tensor,
        U: torch.Tensor,
        *,
        temperature: float,
        k_eff: int,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched `[K, M_sparse]` + `[K, k_eff]` candidate extraction when `pair_is_zero()`."""
        if self.M_sparse < 1:
            raise ValueError('candidate count must be positive')
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if k_eff < 1:
            raise ValueError('k_eff must be positive')
        V = l_unit.shape[-1]
        m_eff = min(self.M_sparse, V)
        r_need = 1 if temperature == 0 else k_eff
        r_k = min(r_need, V - m_eff)
        u_all = l_unit - beta * U
        c_ids_all = torch.topk(u_all, m_eff, dim=-1).indices
        if r_k == 0:
            return c_ids_all, c_ids_all.new_empty((*c_ids_all.shape[:-1], 0))
        if temperature == 0 or m_eff < k_eff or m_eff + r_k == V:
            fmax = torch.finfo(u_all.dtype).max
            u_all.clamp_min_(-fmax)
        u_all.scatter_(-1, c_ids_all, -torch.inf)
        if temperature == 0:
            r_ids_all = torch.max(u_all, dim=-1, keepdim=True).indices
        else:
            r_ids_all = torch.topk(u_all, r_k, dim=-1).indices
        return c_ids_all, r_ids_all

    def score_slot_compact_candidates(
        self,
        l_unit_i: torch.Tensor,
        U_i: torch.Tensor,
        offsets: torch.Tensor,
        prev_ids: torch.Tensor,
        *,
        temperature: float,
        k_eff: int,
        beta: float = 1.0,
        timer=None,
        contiguous_offsets: bool = False,
        bf16_gemm: bool = False,
        precomputed_ids: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Zero-sync `[M_sparse + k_eff]` sparse candidate scoring without `[V]` `scatter_add`.

        Returns `(None, full_score)` when `avail == 0` (no sparse lookup) or `beta == 0`/`is_zero`,
        or `(selected_scores[k_eff], selected_ids[k_eff])` (`k_eff == 1` at `temperature == 0`
        with exact lowest-ID tie-breaking; `scaled_top_scores[k_eff]` at `temperature > 0`).
        """
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if k_eff < 1:
            raise ValueError('k_eff must be positive')
        offs_1d = offsets.squeeze(-1) if offsets.ndim == 2 and offsets.shape[1] == 1 else offsets
        if offs_1d.ndim != 1 or prev_ids.ndim != 1 or offs_1d.numel() != prev_ids.numel():
            raise ValueError('offsets and prev_ids must be matching 1D predecessor tensors (or offsets [avail, 1])')
        avail = prev_ids.numel()
        if contiguous_offsets and avail > len(self.A):
            raise ValueError('contiguous_offsets length exceeds adapter history depth')
        if beta == 0 or self.is_zero:
            return None, l_unit_i
        with _phase(timer, 'dense_pair_score'):
            if avail and not self.pair_is_zero():
                r = self.E.shape[1]
                E_prev_2d = self.E.index_select(0, prev_ids).to(self.A.dtype)
                if contiguous_offsets:
                    message = self._pair_matrix()[:, :avail * r] @ E_prev_2d.reshape(-1)
                else:
                    A_sel = self.A.index_select(0, offs_1d - 1)
                    message = torch.bmm(A_sel, E_prev_2d.unsqueeze(-1)).squeeze(-1).sum(0)
                if bf16_gemm:
                    dense = U_i + (self._token_features_bf16() @ message.bfloat16()).to(U_i.dtype)
                else:
                    dense = U_i + self.E.to(self.A.dtype) @ message
            else:
                dense = U_i
        if self.sparse is None or avail == 0:
            return None, l_unit_i - beta * dense
        if self.M_sparse < 1:
            raise ValueError('candidate count must be positive')
        with _phase(timer, 'sparse_lookup'):
            if precomputed_ids is not None:
                c_ids, r_ids = precomputed_ids
            else:
                V = l_unit_i.shape[-1]
                m_eff = min(self.M_sparse, V)
                r_need = 1 if temperature == 0 else k_eff
                r_k = min(r_need, V - m_eff)
                u = l_unit_i - beta * dense
                c_ids = torch.topk(u, m_eff, dim=-1).indices
                if r_k == 0:
                    r_ids = c_ids.new_empty((0,))
                else:
                    if temperature == 0 or m_eff < k_eff or m_eff + r_k == V:
                        fmax = torch.finfo(u.dtype).max
                        u.clamp_min_(-fmax)
                    u.scatter_(-1, c_ids, -torch.inf)
                    if temperature == 0:
                        r_ids = torch.max(u, dim=-1, keepdim=True).indices
                    else:
                        r_ids = torch.topk(u, r_k, dim=-1).indices
            offs_2d = offs_1d[:, None]
            s_vals = self.sparse.lookup(offs_2d, prev_ids[:, None], c_ids[None, :]).to(dense.dtype).sum(0)
            c_scores = l_unit_i.gather(0, c_ids) - beta * (dense.gather(0, c_ids) + s_vals)
            if r_ids.numel():
                r_scores = l_unit_i.gather(0, r_ids) - beta * dense.gather(0, r_ids)
                cand_ids = torch.cat((c_ids, r_ids), dim=0)
                cand_vals = torch.cat((c_scores, r_scores), dim=0)
            else:
                cand_ids = c_ids
                cand_vals = c_scores
            if temperature == 0:
                cand_ids, perm = cand_ids.sort(dim=0)
                cand_vals = cand_vals.gather(0, perm)
                win_pos = torch.argmax(cand_vals, dim=0, keepdim=True)
                return cand_vals.gather(0, win_pos), cand_ids.gather(0, win_pos)
            if r_ids.numel():
                valid_r = ~(r_ids[:, None] == c_ids[None, :]).any(dim=1)
                all_valid = torch.cat((torch.ones_like(c_ids, dtype=torch.bool), valid_r), dim=0)
            else:
                all_valid = torch.ones_like(c_ids, dtype=torch.bool)
            id_key = torch.where(all_valid, cand_ids, torch.full_like(cand_ids, self.E.shape[0]))
            _, perm = id_key.sort(dim=0, stable=True)
            cand_ids = cand_ids.gather(0, perm)
            cand_vals = cand_vals.gather(0, perm)
            valid_asc = all_valid.gather(0, perm)
            prob_dtype = cand_vals.dtype if cand_vals.dtype == torch.float64 else torch.float32
            scaled = cand_vals.to(prob_dtype) / temperature
            fmax = torch.finfo(scaled.dtype).max
            sel_key = torch.where(valid_asc, scaled.clamp_min(-fmax), -torch.inf)
            _, order = torch.sort(sel_key, dim=0, descending=True, stable=True)
            top_order = order[:min(k_eff, cand_ids.shape[0])]
            return scaled.gather(0, top_order), cand_ids.gather(0, top_order)

    def with_candidates(self, C0):
        return ProposalSnapshot(self.E, self.W, self.A, self.b, self.version, self.sparse,
                                self.M_sparse, tuple(c.detach().clone() for c in C0), self.is_zero,
                                _E_bf16=self._E_bf16, _pair_zero=self._pair_zero)


class DenseAdapter(nn.Module):
    def __init__(self, E: torch.Tensor, p: int, D: int = 4, sparse=None, M_sparse: int = 256):
        super().__init__()
        if E.ndim != 2 or p < 1 or D < 1 or M_sparse < 1:
            raise ValueError('invalid adapter dimensions')
        self.register_buffer('E', E.detach().float().clone())
        V, r = E.shape
        self.W = nn.Parameter(torch.zeros(r, p, device=E.device, dtype=torch.float32))
        self.A = nn.Parameter(torch.zeros(D, r, r, device=E.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(V, device=E.device, dtype=torch.float32))
        self.sparse = sparse
        self.M_sparse = M_sparse
        self._cached_E_bf16: torch.Tensor | None = None

    def _token_features_bf16(self) -> torch.Tensor:
        if self._cached_E_bf16 is None or self._cached_E_bf16.device != self.E.device:
            self._cached_E_bf16 = self.E.detach().bfloat16()
        return self._cached_E_bf16

    def unary(self, H, *, bf16_gemm: bool = False):
        return block_unary(
            self.E, self.W, self.b, H,
            bf16_gemm=bf16_gemm,
            E_bf16=self._token_features_bf16() if bf16_gemm else None,
        )

    def correction(self, l_unit, H, prefix, tokens, beta=1.0, *, bf16_gemm: bool = False):
        U = self.unary(H, bf16_gemm=bf16_gemm)
        return block_correction(
            self.E, self.A, U, l_unit, prefix, tokens,
            self.sparse, self.M_sparse, beta,
            bf16_gemm=bf16_gemm,
            E_bf16=self._token_features_bf16() if bf16_gemm else None,
        )

    def training_distributions(self, l_unit, target_logits, H, prefix, tokens):
        # No inference temperature, beta, or warper parameter exists on this API.
        scores = l_unit.float() - self.correction(l_unit.float(), H.float(), prefix, tokens)
        return torch.softmax(target_logits.float(), -1), torch.softmax(scores, -1)

    def full_loss(self, l_unit, target_logits, H, prefix, tokens, weights=None):
        scores = l_unit.float() - self.correction(l_unit.float(), H.float(), prefix, tokens)
        return full_kl(scores, target_logits, weights)

    def snapshot(self, version=0, C0=None):
        sparse = self.sparse.snapshot() if self.sparse is not None else None
        is_zero = all(not bool(torch.count_nonzero(x)) for x in (self.W, self.A, self.b))
        if self.sparse is not None:
            is_zero = is_zero and not bool(torch.count_nonzero(self.sparse.values))
        return ProposalSnapshot(self.E.detach(), self.W.detach().clone(), self.A.detach().clone(),
                                self.b.detach().clone(), version, sparse, self.M_sparse,
                                None if C0 is None else tuple(c.detach().clone() for c in C0), is_zero,
                                _E_bf16=self._cached_E_bf16)


def loss_weights(K, device=None, mode='exponential', gamma=5., first_rejection=None):
    positions = torch.arange(K, device=device, dtype=torch.float32)
    if mode == 'exponential':
        if gamma <= 0:
            raise ValueError('gamma must be positive')
        return torch.exp(-positions / gamma)
    if first_rejection is None or not 0 <= first_rejection <= K:
        raise ValueError('first_rejection must be in 0..K')
    if mode == 'rejection_relative':
        return 0.35 ** (positions - first_rejection).clamp_min(0)
    if mode == 'cutoff':
        return (positions <= first_rejection).float()
    raise ValueError('unknown weighting mode')


def _non_inference(x: torch.Tensor) -> torch.Tensor:
    return x.clone() if x.is_inference() else x


def _effective_slot_weights(weights, n: int, device: torch.device) -> torch.Tensor:
    """Canonicalize any broadcast-compatible `weights` for `(n,)` per-slot losses into a 1-D `(n,)` float32 tensor."""
    if weights is None:
        return loss_weights(n, device)
    w = (
        weights.detach().to(device=device, dtype=torch.float32)
        if isinstance(weights, torch.Tensor)
        else torch.as_tensor(weights, device=device, dtype=torch.float32)
    )
    bshape = torch.broadcast_shapes((n,), tuple(w.shape))
    w_b = w.expand(bshape)
    if w_b.ndim > 1:
        return w_b.sum(dim=tuple(range(w_b.ndim - 1)))
    return w_b.clone() if w_b.numel() != w.numel() else w_b


class _FullKLFunction(torch.autograd.Function):
    """LSE-replay full-vocabulary KL avoiding [N, V] probability buffers saved for backward."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, target_logits: torch.Tensor, weights) -> torch.Tensor:
        t_det = _non_inference(target_logits.detach())
        w_f = _non_inference(_effective_slot_weights(weights, scores.shape[0], scores.device))
        p_log = torch.log_softmax(t_det.float(), dim=-1)
        q_log = torch.log_softmax(scores.detach().float(), dim=-1)
        per_slot = (p_log.exp() * (p_log - q_log)).sum(dim=-1)
        ctx.save_for_backward(scores, t_det, w_f)
        return (per_slot * w_f).sum()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        scores, target_logits, w_f = ctx.saved_tensors
        scale = (grad_out.float() * w_f)[:, None]
        if torch.is_grad_enabled():
            grad_s = scale * (torch.softmax(scores.float(), dim=-1) - torch.softmax(target_logits.float(), dim=-1))
        else:
            grad_s = torch.softmax(scores.float(), dim=-1)
            grad_s.sub_(torch.softmax(target_logits.float(), dim=-1))
            grad_s.mul_(scale)
        return grad_s.to(scores.dtype), None, None


def full_kl(scores, target_logits, weights=None):
    if weights is None:
        weights = loss_weights(len(scores), scores.device)
    if (
        torch.is_grad_enabled()
        and scores.requires_grad
        and scores.ndim == 2
        and not getattr(weights, 'requires_grad', False)
        and not torch.compiler.is_compiling()
    ):
        return _FullKLFunction.apply(scores, target_logits, weights)
    p_log = torch.log_softmax(target_logits.float().detach(), -1)
    q_log = torch.log_softmax(scores.float(), -1)
    per_slot = (p_log.exp() * (p_log - q_log)).sum(-1)
    return (per_slot * weights).sum()


@dataclass(frozen=True)
class PrecollatedGroupedTargets:
    """Pre-packed ragged union/complement targets for zero-loop grouped-KL steps."""
    rows: int
    vocab_size: int
    ids: torch.Tensor
    p: torch.Tensor
    lengths: torch.Tensor
    row: torch.Tensor
    rest_p: torch.Tensor
    full_rows: torch.Tensor
    weights: torch.Tensor

    def pin_memory(self) -> "PrecollatedGroupedTargets":
        return PrecollatedGroupedTargets(
            rows=self.rows,
            vocab_size=self.vocab_size,
            ids=self.ids.pin_memory(),
            p=self.p.pin_memory(),
            lengths=self.lengths.pin_memory(),
            row=self.row.pin_memory(),
            rest_p=self.rest_p.pin_memory(),
            full_rows=self.full_rows.pin_memory(),
            weights=self.weights.pin_memory(),
        )

    def to(self, device, *, non_blocking: bool = False) -> "PrecollatedGroupedTargets":
        target_dev = torch.device(device)
        if self.ids.device == target_dev:
            return self
        return PrecollatedGroupedTargets(
            rows=self.rows,
            vocab_size=self.vocab_size,
            ids=self.ids.to(target_dev, non_blocking=non_blocking),
            p=self.p.to(target_dev, non_blocking=non_blocking),
            lengths=self.lengths.to(target_dev, non_blocking=non_blocking),
            row=self.row.to(target_dev, non_blocking=non_blocking),
            rest_p=self.rest_p.to(target_dev, non_blocking=non_blocking),
            full_rows=self.full_rows.to(target_dev, non_blocking=non_blocking),
            weights=self.weights.to(target_dev, non_blocking=non_blocking),
        )


def collate_grouped_targets(
    union_ids,
    target_union_probabilities,
    target_rest_mass,
    weights=None,
    *,
    vocab_size: int,
    device=None,
) -> PrecollatedGroupedTargets:
    N = len(union_ids)
    V = int(vocab_size)
    if len(target_union_probabilities) != N:
        raise ValueError('scores and union rows must have consistent [N,V] dimensions')
    lengths = [len(ids) for ids in union_ids]
    if any(n > V or len(prob) != n for n, prob in zip(lengths, target_union_probabilities)):
        raise ValueError('union ID and target-probability lengths differ or exceed vocabulary')
    target_dev = (
        torch.device(device)
        if device is not None
        else (union_ids[0].device if N and isinstance(union_ids[0], torch.Tensor) else torch.device('cpu'))
    )
    total_len = sum(lengths)
    lengths_tensor = torch.tensor(lengths, device=target_dev, dtype=torch.long)
    if total_len:
        ids = torch.cat([x.detach().to(device=target_dev, dtype=torch.long) for x in union_ids])
        p = torch.cat([x.detach().to(device=target_dev, dtype=torch.float32) for x in target_union_probabilities])
        row = torch.repeat_interleave(
            torch.arange(N, device=target_dev, dtype=torch.long),
            lengths_tensor,
            output_size=total_len,
        )
    else:
        ids = torch.empty(0, device=target_dev, dtype=torch.long)
        p = torch.empty(0, device=target_dev, dtype=torch.float32)
        row = torch.empty(0, device=target_dev, dtype=torch.long)
    rest_p = (
        target_rest_mass.detach().to(device=target_dev, dtype=torch.float32)
        if isinstance(target_rest_mass, torch.Tensor)
        else torch.stack(
            [torch.as_tensor(x, device=target_dev, dtype=torch.float32).detach() for x in target_rest_mass]
        )
        if N
        else torch.empty(0, device=target_dev, dtype=torch.float32)
    )
    if rest_p.shape != (N,):
        raise ValueError('target rest mass must have one scalar per score row')
    full_list = [i for i, n in enumerate(lengths) if n == V]
    if full_list:
        full_rows = torch.tensor(full_list, device=target_dev, dtype=torch.long)
        if bool((rest_p.index_select(0, full_rows) > 2e-6).any()):
            raise ValueError('nonzero target rest mass is inconsistent with a full-vocabulary union')
        rest_p = rest_p.index_fill(0, full_rows, 0.)
    else:
        full_rows = torch.empty(0, device=target_dev, dtype=torch.long)
    eff_weights = _effective_slot_weights(weights, N, target_dev)
    return PrecollatedGroupedTargets(
        rows=N,
        vocab_size=V,
        ids=ids,
        p=p,
        lengths=lengths_tensor,
        row=row,
        rest_p=rest_p,
        full_rows=full_rows,
        weights=eff_weights,
    )


class _GroupedKLFunction(torch.autograd.Function):
    """Exact full-V grouped KL saving zero [N, V] probability/mask buffers for backward."""

    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        ids: torch.Tensor,
        p: torch.Tensor,
        lengths: torch.Tensor,
        row: torch.Tensor,
        rest_p: torch.Tensor,
        full_rows: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        q_log = torch.log_softmax(scores.detach().float(), dim=-1)
        if ids.numel():
            union_terms = torch.xlogy(p, p) - p * q_log[row, ids]
            union_loss = torch.segment_reduce(union_terms, 'sum', lengths=lengths)
            q_log.index_put_((row, ids), q_log.new_full((ids.numel(),), -torch.inf))
        else:
            union_loss = q_log.new_zeros(scores.shape[0])
        if full_rows.numel():
            q_log.index_put_((full_rows, torch.zeros_like(full_rows)), q_log.new_zeros(full_rows.numel()))
        log_q_rest = torch.logsumexp(q_log, dim=-1)
        rest_loss = torch.where(
            rest_p > 0,
            torch.xlogy(rest_p, rest_p) - rest_p * log_q_rest,
            torch.zeros_like(rest_p),
        )
        ctx.save_for_backward(
            scores,
            _non_inference(ids),
            _non_inference(p),
            _non_inference(lengths),
            _non_inference(row),
            _non_inference(rest_p),
            _non_inference(full_rows),
            _non_inference(weights),
            _non_inference(log_q_rest),
        )
        return ((union_loss + rest_loss) * weights).sum()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        scores, ids, p, lengths, row, rest_p, full_rows, weights, log_q_rest = ctx.saved_tensors
        s = scores.float()
        gw = grad_out.float() * weights
        c_rest = (-gw * rest_p)[:, None]

        if torch.is_grad_enabled():
            q_log = torch.log_softmax(s, dim=-1)
            q = q_log.exp()
            dy_tail = (
                q_log.index_put((row, ids), q_log.new_full((ids.numel(),), -torch.inf))
                if ids.numel()
                else q_log
            )
            if full_rows.numel():
                dy_tail = dy_tail.index_put((full_rows, torch.zeros_like(full_rows)), q_log.new_zeros(full_rows.numel()))
            safe_tail = torch.where((rest_p <= 0)[:, None], torch.zeros_like(dy_tail), dy_tail)
            lse_tail_live = torch.logsumexp(safe_tail, dim=-1, keepdim=True)
            dy = torch.where((rest_p > 0)[:, None], torch.exp(safe_tail - lse_tail_live) * c_rest, torch.zeros_like(q_log))
            if ids.numel():
                dy = dy.index_put((row, ids), -gw.index_select(0, row) * p, accumulate=True)
            grad = dy - q * dy.sum(dim=-1, keepdim=True)
        else:
            lse_tail = torch.where(torch.isfinite(log_q_rest), log_q_rest, torch.zeros_like(log_q_rest))[:, None]
            dy = torch.log_softmax(s, dim=-1)
            q = dy.exp()
            if ids.numel():
                dy.index_put_((row, ids), dy.new_full((ids.numel(),), -torch.inf))
            if full_rows.numel():
                dy.index_put_((full_rows, torch.zeros_like(full_rows)), dy.new_zeros(full_rows.numel()))
            dy.sub_(lse_tail).exp_().mul_(c_rest)
            if ids.numel():
                dy.index_put_((row, ids), -gw.index_select(0, row) * p, accumulate=True)
            dy_sum = dy.sum(dim=-1, keepdim=True)
            q.mul_(dy_sum)
            dy.sub_(q)
            grad = dy
        return grad.to(scores.dtype), None, None, None, None, None, None, None


def _grouped_kl_compile_friendly(scores: torch.Tensor, pre: PrecollatedGroupedTargets) -> torch.Tensor:
    q_log = torch.log_softmax(scores.float(), dim=-1)
    if pre.ids.numel():
        union_terms = torch.xlogy(pre.p, pre.p) - pre.p * q_log[pre.row, pre.ids]
        union_loss = torch.segment_reduce(union_terms, 'sum', lengths=pre.lengths)
        tail_scores = q_log.index_put((pre.row, pre.ids), q_log.new_full((pre.ids.numel(),), -torch.inf))
    else:
        union_loss = q_log.new_zeros(scores.shape[0])
        tail_scores = q_log
    if pre.full_rows.numel():
        tail_scores = tail_scores.index_put(
            (pre.full_rows, torch.zeros_like(pre.full_rows)),
            q_log.new_zeros(pre.full_rows.numel()),
        )
    safe_tail = torch.where((pre.rest_p <= 0)[:, None], torch.zeros_like(tail_scores), tail_scores)
    log_q_rest = torch.logsumexp(safe_tail, dim=-1)
    rest_loss = torch.where(
        pre.rest_p > 0,
        torch.xlogy(pre.rest_p, pre.rest_p) - pre.rest_p * log_q_rest,
        torch.zeros_like(pre.rest_p),
    )
    return ((union_loss + rest_loss) * pre.weights).sum()


def union_rest_grouped_precollated(scores: torch.Tensor, pre: PrecollatedGroupedTargets) -> torch.Tensor:
    """Exact full-V grouped KL on pre-collated ragged targets."""
    if scores.ndim != 2 or scores.shape != (pre.rows, pre.vocab_size):
        raise ValueError('scores and union rows must have consistent [N,V] dimensions')
    target_pre = pre.to(scores.device)
    if (
        torch.is_grad_enabled()
        and scores.requires_grad
        and not target_pre.weights.requires_grad
        and not torch.compiler.is_compiling()
    ):
        return _GroupedKLFunction.apply(
            scores,
            target_pre.ids,
            target_pre.p,
            target_pre.lengths,
            target_pre.row,
            target_pre.rest_p,
            target_pre.full_rows,
            target_pre.weights,
        )
    return _grouped_kl_compile_friendly(scores, target_pre)


def union_rest_grouped(scores, union_ids, target_union_probabilities, target_rest_mass, weights=None):
    """Exact full-V normalization and one batched masked-tail logsumexp."""
    if scores.ndim != 2 or len(union_ids) != len(scores) or len(target_union_probabilities) != len(scores):
        raise ValueError('scores and union rows must have consistent [N,V] dimensions')
    N, V = scores.shape
    weights_t = None if weights is None else weights
    pre = collate_grouped_targets(
        union_ids,
        target_union_probabilities,
        target_rest_mass,
        weights_t if (weights_t is None or not getattr(weights_t, 'requires_grad', False)) else None,
        vocab_size=V,
        device=scores.device,
    )
    if weights_t is not None and getattr(weights_t, 'requires_grad', False):
        q_log = torch.log_softmax(scores.float(), -1)
        union_terms = torch.xlogy(pre.p, pre.p) - pre.p * q_log[pre.row, pre.ids]
        union_loss = torch.segment_reduce(union_terms, 'sum', lengths=pre.lengths)
        tail_scores = q_log.index_put((pre.row, pre.ids), q_log.new_full((pre.ids.numel(),), -torch.inf))
        if pre.full_rows.numel():
            tail_scores = tail_scores.index_put(
                (pre.full_rows, torch.zeros_like(pre.full_rows)),
                q_log.new_zeros(pre.full_rows.numel()),
            )
        safe_tail = torch.where((pre.rest_p <= 0)[:, None], torch.zeros_like(tail_scores), tail_scores)
        log_q_rest = torch.logsumexp(safe_tail, -1)
        rest_loss = torch.where(
            pre.rest_p > 0,
            torch.xlogy(pre.rest_p, pre.rest_p) - pre.rest_p * log_q_rest,
            torch.zeros_like(pre.rest_p),
        )
        return ((union_loss + rest_loss) * weights_t).sum()
    return union_rest_grouped_precollated(scores, pre)


def fused_head_grouped_kl(
    h_raw_bf16: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor | None,
    dense_correction: torch.Tensor,
    pre: PrecollatedGroupedTargets,
    *,
    output_multiplier: float = 1.0,
    bf16_gemm: bool = False,
) -> torch.Tensor:
    """Shift-invariant fused `h_raw_bf16 @ draft_lm_head.T - dense_correction` -> grouped KL."""
    with torch.no_grad():
        if bf16_gemm:
            w_bf = head_weight if head_weight.dtype == torch.bfloat16 else head_weight.bfloat16()
            raw_logits = torch.nn.functional.linear(h_raw_bf16.bfloat16(), w_bf).float()
            if head_bias is not None:
                raw_logits.add_(head_bias.float())
        else:
            raw_logits = torch.nn.functional.linear(
                h_raw_bf16.float(),
                head_weight.float(),
                None if head_bias is None else head_bias.float(),
            )
        if output_multiplier != 1.0:
            raw_logits.mul_(float(output_multiplier))
    return union_rest_grouped_precollated(raw_logits - dense_correction, pre)


@dataclass
class LookaheadState:
    snapshot: ProposalSnapshot
    l_unit: torch.Tensor
    U_all: torch.Tensor
    prefix: tuple[int, ...]
    support: list[torch.Tensor]
    clamped: list[int | None]
    r: list[torch.Tensor]
    mu: list[torch.Tensor]
    g: torch.Tensor  # Full vocabulary representation of the field in both modes.
    domain: str
    T_relax: float
    beta_relax: float

    def raw_support_mass(self, i):
        return self.r[i] if self.domain == 'candidate_masked' else self.r[i][self.support[i]]

    def full_belief(self, i):
        if self.domain == 'full_vocabulary':
            return self.r[i]
        return torch.zeros(self.snapshot.E.shape[0], device=self.g.device, dtype=self.r[i].dtype).scatter(0, self.support[i], self.r[i])

    def clamp(self, i: int, token: int):
        self.clamped[i] = int(token)
        self.support[i] = torch.tensor([token], device=self.g.device, dtype=torch.long)
        if self.domain == 'candidate_masked':
            self.r[i] = torch.ones(1, device=self.g.device, dtype=self.g.dtype)
        else:
            self.r[i] = torch.nn.functional.one_hot(torch.tensor(token, device=self.g.device), self.g.shape[-1]).to(self.g.dtype)
        self.mu[i] = self.snapshot.E[token]

    def sample_slot(self, i, T_sample, beta_sample=1., warper=None, generator=None):
        from .sampler import SamplingWarper, sample_token
        scores = self.l_unit[i] - beta_sample * self.g[i]
        if self.domain == 'candidate_masked':
            ids = self.snapshot.C0[i]
            scores = torch.full_like(scores, -torch.inf).scatter(0, ids, scores[ids])
        token, probs = sample_token(scores, T_sample, warper or SamplingWarper(), generator=generator)
        self.clamp(i, token)
        return token, probs


def initialize_lookahead(snapshot, l_unit, H, prefix, domain='candidate_masked', T_relax=1., beta_relax=1.):
    if T_relax <= 0:
        raise ValueError('T_relax must be strictly positive')
    if domain not in ('candidate_masked', 'full_vocabulary'):
        raise ValueError('unknown belief domain')
    U = snapshot.unary(H)
    if snapshot.C0 is None:
        snapshot = snapshot.with_candidates(top_candidates(l_unit - beta_relax * U, snapshot.M_sparse))
    support = [ids.clone() for ids in snapshot.C0]
    state = LookaheadState(snapshot, l_unit, U, tuple(prefix), support, [None] * len(H), [], [], U.clone(), domain, T_relax, beta_relax)
    for i in range(len(H)):
        scores = l_unit[i] - beta_relax * U[i]
        ids = support[i] if domain == 'candidate_masked' else torch.arange(l_unit.shape[-1], device=l_unit.device)
        belief = torch.softmax(scores[ids] / T_relax, -1)
        state.r.append(belief)
        state.mu.append(belief @ snapshot.E[ids])
    return state


def lookahead_fields(state: LookaheadState):
    """Pure fwd/back fields: transpose A backwards, raw masked sparse mass."""
    snap, E, A = state.snapshot, state.snapshot.E, state.snapshot.A
    forward, backward = [], []
    K = len(state.r)
    for i in range(K):
        f = torch.zeros_like(state.g[i])
        b = torch.zeros_like(f)
        own = state.support[i]
        for d in range(1, len(A) + 1):
            k = i - d
            if k >= 0:
                f = f + E @ (A[d - 1] @ state.mu[k])
                if snap.sparse is not None:
                    table = snap.sparse.lookup(d, state.support[k][:, None], own[None, :])
                    msg = state.raw_support_mass(k) @ table
                    f = f + torch.zeros_like(f).scatter(0, own, msg)
            else:
                previous = prev_token(state.prefix, (), i, d)
                if previous is not None:
                    f = f + E @ (A[d - 1] @ E[previous])
                    if snap.sparse is not None:
                        f = f + torch.zeros_like(f).scatter(0, own, snap.sparse.lookup(d, previous, own))
            k = i + d
            if k < K:
                b = b + E @ (A[d - 1].T @ state.mu[k])
                if snap.sparse is not None:
                    table = snap.sparse.lookup(d, own[:, None], state.support[k][None, :])
                    msg = table @ state.raw_support_mass(k)
                    b = b + torch.zeros_like(b).scatter(0, own, msg)
        forward.append(f)
        backward.append(b)
    return torch.stack(forward), torch.stack(backward)


def lookahead_energy(state: LookaheadState):
    """One shared masked scalar energy used to verify both message derivatives."""
    snap, E, A = state.snapshot, state.snapshot.E, state.snapshot.A
    energy = sum((state.full_belief(i) * state.U_all[i]).sum() for i in range(len(state.r)))
    for i in range(len(state.r)):
        for d in range(1, len(A) + 1):
            k = i - d
            if k >= 0:
                energy = energy + state.mu[i] @ A[d - 1] @ state.mu[k]
                if snap.sparse is not None:
                    pairs = snap.sparse.lookup(d, state.support[k][:, None], state.support[i][None, :])
                    energy = energy + state.raw_support_mass(k) @ pairs @ state.raw_support_mass(i)
            else:
                previous = prev_token(state.prefix, (), i, d)
                if previous is not None:
                    energy = energy + state.mu[i] @ A[d - 1] @ E[previous]
                    if snap.sparse is not None:
                        energy = energy + state.raw_support_mass(i) @ snap.sparse.lookup(d, previous, state.support[i])
    return energy


def damped_lookahead_step(state: LookaheadState, eta=0.5):
    if not 0 < eta <= 1:
        raise ValueError('eta must be in (0,1]')
    forward, backward = lookahead_fields(state)
    tilde = state.U_all + forward + backward
    # All fields are computed before any belief changes: genuinely synchronous.
    fields = (1 - eta) * state.g + eta * tilde
    r_new, mu_new = list(state.r), list(state.mu)
    for i in range(len(state.r)):
        if state.clamped[i] is not None:
            fields[i] = state.g[i]
            continue
        ids = state.support[i] if state.domain == 'candidate_masked' else torch.arange(state.g.shape[-1], device=state.g.device)
        r_new[i] = torch.softmax((state.l_unit[i, ids] - state.beta_relax * fields[i, ids]) / state.T_relax, -1)
        mu_new[i] = r_new[i] @ state.snapshot.E[ids]
    state.g, state.r, state.mu = fields, r_new, mu_new
    return state
