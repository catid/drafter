"""Causal energy correction and standalone synchronous lookahead mathematics."""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
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


def block_unary(E, W, b, H):
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
                    *, pair_matrix=None):
    with _phase(timer, 'dense_pair_score'):
        predecessors = []
        for d in range(1, len(A) + 1):
            previous = prev_token(prefix, realized, i, d)
            if previous is not None:
                predecessors.append((d, previous))
        if predecessors:
            message = _pair_message(E, A, predecessors, pair_matrix=pair_matrix)
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


def block_correction(E, A, U, l_unit, prefix, tokens, sparse=None, M_sparse=256, beta=1.0):
    """Vectorized [K, V] causal correction across all slots and offsets 1..D."""
    K = len(U)
    valid_slots, valid_d0, valid_prev = block_predecessor_pairs(prefix, tokens, K, len(A), A.device)
    if valid_slots.numel() == 0:
        return U
    E_cast = E.to(A.dtype)
    E_prev = E_cast.index_select(0, valid_prev)
    msgs = torch.bmm(A.index_select(0, valid_d0), E_prev.unsqueeze(-1)).squeeze(-1)
    slot_msgs = torch.zeros(K, E.shape[1], dtype=A.dtype, device=A.device).index_add(0, valid_slots, msgs)
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

    def _pair_matrix(self) -> torch.Tensor:
        cached = self._A_flat
        if cached is None:
            D, r, _ = self.A.shape
            cached = self.A.permute(1, 0, 2).contiguous().view(r, D * r)
            object.__setattr__(self, '_A_flat', cached)
        return cached

    @property
    def D(self):
        return len(self.A)

    @property
    def adapter_version(self):
        return self.version

    def unary(self, H, proposal_mode='dense_full', *, unary_fn=None):
        if proposal_mode == 'dense_full':
            return (block_unary if unary_fn is None else unary_fn)(self.E, self.W, self.b, H)
        if proposal_mode != 'dense_candidate' or self.C0 is None:
            raise ValueError('candidate unary requires frozen C0')
        hidden = H.to(self.W.dtype) @ self.W.T
        # Only candidate rows of E are read; no K-by-V dense GEMM occurs here.
        return [self.E[ids].to(self.W.dtype) @ hidden[i] + self.b[ids]
                for i, ids in enumerate(self.C0)]

    def score_slot(self, l_unit_i, U_i, prefix, realized, i, beta=1.0, proposal_mode='dense_full', timer=None):
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
        correction = slot_correction(self.E, self.A, U_i, l_unit_i, prefix, realized, i,
                                     self.sparse, self.M_sparse, beta, timer=timer,
                                     pair_matrix=self._pair_matrix())
        return l_unit_i - beta * correction

    def score_slot_tensor(self, l_unit_i, U_i, offsets: torch.Tensor, prev_ids: torch.Tensor, i: int = 0,
                          beta=1.0, proposal_mode='dense_full', timer=None):
        """Sync-free causal slot scoring when offsets [avail] or [avail, 1] and prev_ids [avail] reside on device."""
        if proposal_mode != 'dense_full':
            raise ValueError('score_slot_tensor requires dense_full proposal mode')
        if beta == 0 or self.is_zero:
            return l_unit_i
        avail = prev_ids.numel()
        with _phase(timer, 'dense_pair_score'):
            if avail:
                r = self.E.shape[1]
                E_prev = self.E.index_select(0, prev_ids).to(self.A.dtype).reshape(-1)
                message = self._pair_matrix()[:, :avail * r] @ E_prev
                dense = U_i + self.E.to(self.A.dtype) @ message
            else:
                dense = U_i
        if self.sparse is None or avail == 0:
            correction = dense
        else:
            with _phase(timer, 'sparse_lookup'):
                candidates = top_candidates(l_unit_i - beta * dense, self.M_sparse)
                offs_2d = offsets[:, None] if offsets.ndim == 1 else offsets
                values = self.sparse.lookup(offs_2d, prev_ids[:, None], candidates[None, :]).to(dense.dtype).sum(0)
                correction = dense.scatter_add(0, candidates, values)
        return l_unit_i - beta * correction

    def with_candidates(self, C0):
        return ProposalSnapshot(self.E, self.W, self.A, self.b, self.version, self.sparse,
                                self.M_sparse, tuple(c.detach().clone() for c in C0), self.is_zero)


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

    def unary(self, H):
        return block_unary(self.E, self.W, self.b, H)

    def correction(self, l_unit, H, prefix, tokens, beta=1.0):
        U = self.unary(H)
        return block_correction(self.E, self.A, U, l_unit, prefix, tokens,
                                self.sparse, self.M_sparse, beta)

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
                                None if C0 is None else tuple(c.detach().clone() for c in C0), is_zero)


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


def full_kl(scores, target_logits, weights=None):
    p_log = torch.log_softmax(target_logits.float().detach(), -1)
    q_log = torch.log_softmax(scores.float(), -1)
    per_slot = (p_log.exp() * (p_log - q_log)).sum(-1)
    if weights is None:
        weights = loss_weights(len(scores), scores.device)
    return (per_slot * weights).sum()


def union_rest_grouped(scores, union_ids, target_union_probabilities, target_rest_mass, weights=None):
    """Exact full-V normalization and one batched masked-tail logsumexp."""
    if scores.ndim != 2 or len(union_ids) != len(scores) or len(target_union_probabilities) != len(scores):
        raise ValueError('scores and union rows must have consistent [N,V] dimensions')
    q_log = torch.log_softmax(scores.float(), -1)
    N, V = q_log.shape
    lengths = [len(ids) for ids in union_ids]
    if any(n > V or len(p) != n for n, p in zip(lengths, target_union_probabilities)):
        raise ValueError('union ID and target-probability lengths differ or exceed vocabulary')
    # Canonical unions are sorted unique ID sets. Ragged rows are concatenated
    # rather than padding to the largest union or launching one kernel per row.
    ids = torch.cat([x.to(device=scores.device, dtype=torch.long) for x in union_ids])
    p = torch.cat([x.detach().to(device=scores.device, dtype=torch.float32) for x in target_union_probabilities])
    lengths_tensor = torch.tensor(lengths, device=scores.device)
    row = torch.repeat_interleave(torch.arange(N, device=scores.device),
                                  lengths_tensor, output_size=sum(lengths))
    union_terms = torch.xlogy(p, p) - p * q_log[row, ids]
    union_loss = torch.segment_reduce(union_terms, 'sum', lengths=lengths_tensor)
    rest_p = (target_rest_mass.detach().to(device=scores.device, dtype=torch.float32)
              if isinstance(target_rest_mass, torch.Tensor) else
              torch.stack([torch.as_tensor(x, device=scores.device, dtype=torch.float32).detach()
                           for x in target_rest_mass]))
    if rest_p.shape != (N,):
        raise ValueError('target rest mass must have one scalar per score row')
    full_rows = [i for i, n in enumerate(lengths) if n == V]
    tail_scores = q_log.index_put((row, ids), q_log.new_full((len(ids),), -torch.inf))
    if full_rows:
        full = torch.tensor(full_rows, device=scores.device)
        if bool((rest_p[full] > 2e-6).any()):
            raise ValueError('nonzero target rest mass is inconsistent with a full-vocabulary union')
        rest_p = rest_p.index_fill(0, full, 0.)
        # logsumexp(-inf,...,-inf) has undefined backward even if multiplied
        # by zero. A constant sentinel gives these zero-mass rows finite math.
        tail_scores = tail_scores.index_put((full, torch.zeros_like(full)), q_log.new_zeros(len(full)))
    log_q_rest = torch.logsumexp(tail_scores, -1)
    rest_loss = torch.xlogy(rest_p, rest_p) - rest_p * log_q_rest
    if weights is None:
        weights = loss_weights(N, scores.device)
    return ((union_loss + rest_loss) * weights).sum()


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
