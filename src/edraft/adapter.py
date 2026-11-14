"""Causal energy correction and standalone synchronous lookahead mathematics."""
from __future__ import annotations

from dataclasses import dataclass
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


def next_token(committed: Sequence[int], i: int, d: int) -> int | None:
    if d < 1:
        raise ValueError('offset d must be positive; there is no self-interaction')
    if i < 0 or i >= len(committed):
        raise ValueError('slot i must index a committed verified position')
    k = i + d
    return int(committed[k]) if k < len(committed) else None


def top_candidates(scores: torch.Tensor, count: int) -> torch.Tensor:
    if count < 1:
        raise ValueError('candidate count must be positive')
    return torch.topk(scores, min(count, scores.shape[-1]), dim=-1).indices


def block_unary(E, W, b, H):
    return (H.to(W.dtype) @ W.T) @ E.to(W.dtype).T + b


def _phase(timer, name):
    return nullcontext() if timer is None else timer.phase(name)


def slot_correction(E, A, U_i, l_unit_i, prefix, realized, i, sparse=None, M_sparse=256, beta=1.0, timer=None):
    with _phase(timer, 'dense_pair_score'):
        message = torch.zeros(E.shape[1], dtype=A.dtype, device=A.device)
        predecessors = []
        for d in range(1, len(A) + 1):
            previous = prev_token(prefix, realized, i, d)
            if previous is not None:
                message = message + A[d - 1] @ E[previous].to(A.dtype)
                predecessors.append((d, previous))
        dense = U_i + E.to(A.dtype) @ message
    if sparse is None or not predecessors:
        return dense
    with _phase(timer, 'sparse_lookup'):
        candidates = top_candidates(l_unit_i - beta * dense, M_sparse)
        # All logical offsets share one vectorized hash invocation for this slot.
        offsets = torch.tensor([d for d, _ in predecessors], device=dense.device)[:, None]
        previous = torch.tensor([p for _, p in predecessors], device=dense.device)[:, None]
        values = sparse.lookup(offsets, previous, candidates[None, :]).to(dense.dtype).sum(0)
        correction = dense + torch.zeros_like(dense).scatter(0, candidates, values)
    return correction


def two_sided_slot_correction(E, A, U_i, l_unit_i, prefix, committed, i,
                              sparse=None, M_sparse=256, beta=1.0, timer=None):
    """Exact two-sided energy correction on committed verified token stream `committed`."""
    if i < 0 or i >= len(committed):
        raise ValueError('two-sided correction requires slot i on the committed verified stream')
    with _phase(timer, 'dense_pair_score'):
        message = torch.zeros(E.shape[1], dtype=A.dtype, device=A.device)
        predecessors = []
        successors = []
        for d in range(1, len(A) + 1):
            previous = prev_token(prefix, committed, i, d)
            if previous is not None:
                message = message + A[d - 1] @ E[previous].to(A.dtype)
                predecessors.append((d, previous))
            successor = next_token(committed, i, d)
            if successor is not None:
                message = message + A[d - 1].T @ E[successor].to(A.dtype)
                successors.append((d, successor))
        dense = U_i + E.to(A.dtype) @ message
    if sparse is None or (not predecessors and not successors):
        return dense
    with _phase(timer, 'sparse_lookup'):
        candidates = top_candidates(l_unit_i - beta * dense, M_sparse)
        values = torch.zeros(len(candidates), dtype=dense.dtype, device=dense.device)
        if predecessors:
            offsets_f = torch.tensor([d for d, _ in predecessors], device=dense.device)[:, None]
            prev_ids = torch.tensor([p for _, p in predecessors], device=dense.device)[:, None]
            values = values + sparse.lookup(offsets_f, prev_ids, candidates[None, :]).to(dense.dtype).sum(0)
        if successors:
            offsets_b = torch.tensor([d for d, _ in successors], device=dense.device)[:, None]
            succ_ids = torch.tensor([s for _, s in successors], device=dense.device)[:, None]
            values = values + sparse.lookup(offsets_b, candidates[None, :], succ_ids).to(dense.dtype).sum(0)
        correction = dense + torch.zeros_like(dense).scatter(0, candidates, values)
    return correction


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
    unary_fn: object | None = None

    @property
    def D(self):
        return len(self.A)

    @property
    def adapter_version(self):
        return self.version

    def unary(self, H, proposal_mode='dense_full'):
        if proposal_mode == 'dense_full':
            if self.unary_fn is not None:
                return self.unary_fn(self.E, self.W, self.b, H)
            return block_unary(self.E, self.W, self.b, H)
        if proposal_mode != 'dense_candidate' or self.C0 is None:
            raise ValueError('candidate unary requires frozen C0')
        work_dtype = torch.promote_types(H.dtype, self.W.dtype)
        hidden = H.to(work_dtype) @ self.W.to(work_dtype).T
        E_w = self.E.to(work_dtype)
        b_w = self.b.to(work_dtype)
        # Only candidate rows of E are read; no K-by-V dense GEMM occurs here.
        return [E_w[ids] @ hidden[i] + b_w[ids]
                for i, ids in enumerate(self.C0)]

    def score_slot(self, l_unit_i, U_i, prefix, realized, i, beta=1.0, proposal_mode='dense_full', timer=None):
        if proposal_mode == 'dense_candidate':
            if self.C0 is None:
                raise ValueError('dense_candidate requires the frozen C0 matched-support snapshot')
            ids = self.C0[i]
            work_dtype = torch.promote_types(l_unit_i.dtype, self.A.dtype)
            if U_i is not None:
                work_dtype = torch.promote_types(work_dtype, U_i.dtype)
            if beta == 0 or self.is_zero:
                selected = l_unit_i[ids].to(work_dtype)
            else:
                if U_i.shape[-1] != len(ids):
                    raise ValueError('candidate scoring requires candidate unary rows; call unary(..., proposal_mode=dense_candidate)')
                unary = U_i.to(work_dtype)
                A_w = self.A.to(work_dtype)
                E_w = self.E.to(work_dtype)
                with _phase(timer, 'dense_pair_score'):
                    message = torch.zeros(E_w.shape[1], dtype=work_dtype, device=self.A.device)
                    predecessors = []
                    for d in range(1, len(A_w) + 1):
                        previous = prev_token(prefix, realized, i, d)
                        if previous is not None:
                            message = message + A_w[d - 1] @ E_w[previous]
                            predecessors.append((d, previous))
                    correction = unary + E_w[ids] @ message
                if self.sparse is not None and predecessors:
                    with _phase(timer, 'sparse_lookup'):
                        offsets = torch.tensor([d for d, _ in predecessors], device=ids.device)[:, None]
                        previous_ids = torch.tensor([p for _, p in predecessors], device=ids.device)[:, None]
                        correction = correction + self.sparse.lookup(offsets, previous_ids, ids[None, :]).to(work_dtype).sum(0)
                selected = l_unit_i[ids].to(work_dtype) - beta * correction
            return torch.full(l_unit_i.shape, -torch.inf, device=selected.device, dtype=selected.dtype).scatter(0, ids, selected)
        if proposal_mode != 'dense_full':
            raise ValueError('unknown proposal mode')
        if beta == 0 or self.is_zero:
            return l_unit_i
        work_dtype = torch.promote_types(l_unit_i.dtype, self.A.dtype)
        if U_i is not None:
            work_dtype = torch.promote_types(work_dtype, U_i.dtype)
        correction = slot_correction(self.E, self.A, U_i.to(work_dtype), l_unit_i.to(work_dtype), prefix, realized, i,
                                     self.sparse, self.M_sparse, beta, timer=timer)
        return l_unit_i.to(correction.dtype) - beta * correction

    def with_candidates(self, C0):
        return ProposalSnapshot(self.E, self.W, self.A, self.b, self.version, self.sparse,
                                self.M_sparse, tuple(c.detach().clone() for c in C0), self.is_zero, self.unary_fn)


class DenseAdapter(nn.Module):
    def __init__(self, E: torch.Tensor, p: int, D: int = 4, sparse=None, M_sparse: int = 256, *, unary_fn=None):
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
        self.unary_fn = unary_fn

    def unary(self, H):
        if self.unary_fn is not None:
            return self.unary_fn(self.E, self.W, self.b, H)
        return block_unary(self.E, self.W, self.b, H)

    def correction(self, l_unit, H, prefix, tokens, beta=1.0):
        U = self.unary(H)
        return torch.stack([slot_correction(self.E, self.A, U[i], l_unit[i], prefix, tokens, i,
                                           self.sparse, self.M_sparse, beta) for i in range(len(H))])

    def training_distributions(self, l_unit, target_logits, H, prefix, tokens):
        # No inference temperature, beta, or warper parameter exists on this API.
        scores = l_unit.float() - self.correction(l_unit.float(), H.float(), prefix, tokens)
        return torch.softmax(target_logits.float(), -1), torch.softmax(scores, -1)

    def full_loss(self, l_unit, target_logits, H, prefix, tokens, weights=None):
        scores = l_unit.float() - self.correction(l_unit.float(), H.float(), prefix, tokens)
        return full_kl(scores, target_logits, weights)

    def two_sided_correction(self, l_unit, H, prefix, committed, eval_slots=None, beta=1.0):
        n_slots = min(len(H), len(committed)) if eval_slots is None else int(eval_slots)
        if n_slots < 1 or n_slots > min(len(H), len(committed)):
            raise ValueError('eval_slots must lie in 1..min(len(H), len(committed))')
        U = self.unary(H[:n_slots])
        return torch.stack([
            two_sided_slot_correction(self.E, self.A, U[i], l_unit[i], prefix, committed, i,
                                      self.sparse, self.M_sparse, beta)
            for i in range(n_slots)
        ])

    def two_sided_loss(self, l_unit, H, prefix, committed, weights=None, eval_slots=None):
        n_slots = min(len(H), len(committed)) if eval_slots is None else int(eval_slots)
        scores = l_unit[:n_slots].float() - self.two_sided_correction(
            l_unit.float(), H.float(), prefix, committed, eval_slots=n_slots, beta=1.0
        )
        return two_sided_pseudolikelihood_loss(scores, committed[:n_slots], weights)

    def snapshot(self, version=0, C0=None):
        sparse = self.sparse.snapshot() if self.sparse is not None else None
        is_zero = all(not bool(torch.count_nonzero(x)) for x in (self.W, self.A, self.b))
        if sparse is not None:
            is_zero = is_zero and not bool(torch.count_nonzero(sparse.values))
        return ProposalSnapshot(self.E.detach(), self.W.detach().clone(), self.A.detach().clone(),
                                self.b.detach().clone(), version, sparse, self.M_sparse,
                                None if C0 is None else tuple(c.detach().clone() for c in C0), is_zero,
                                self.unary_fn)


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


def two_sided_pseudolikelihood_loss(scores, committed_targets: Sequence[int] | torch.Tensor, weights=None):
    """Exact Besag two-sided negative log-pseudolikelihood over all V tokens (`T_train=beta_train=1`)."""
    if scores.ndim != 2 or len(scores) == 0:
        raise ValueError('scores must have shape [L, V] with L >= 1')
    targets = torch.as_tensor(committed_targets, dtype=torch.long, device=scores.device)
    if targets.shape != (len(scores),):
        raise ValueError('committed_targets must have one verified token ID per score row')
    if bool((targets < 0).any() or (targets >= scores.shape[-1]).any()):
        raise ValueError('committed token ID outside vocabulary')
    q_log = torch.log_softmax(scores.float(), -1)
    per_slot_nll = -q_log.gather(-1, targets[:, None])[:, 0]
    if weights is None:
        weights = loss_weights(len(scores), scores.device)
    return (per_slot_nll * weights).sum()


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
    rest_term = torch.where(rest_p > 0, rest_p * log_q_rest, torch.zeros_like(rest_p))
    rest_loss = torch.xlogy(rest_p, rest_p) - rest_term
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
        self.mu[i] = self.snapshot.E[token].to(self.g.dtype)

    def sample_slot(self, i, T_sample, beta_sample=1., warper=None, generator=None):
        from .sampler import SamplingWarper, sample_token
        scores = self.l_unit[i] - beta_sample * self.g[i]
        if self.domain == 'candidate_masked':
            ids = self.snapshot.C0[i]
            scores = torch.full(scores.shape, -torch.inf, device=scores.device, dtype=scores.dtype).scatter(0, ids, scores[ids])
        token, probs = sample_token(scores, T_sample, warper or SamplingWarper(), generator=generator)
        self.clamp(i, token)
        return token, probs


def _valid_real_scalar(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass(frozen=True)
class LookaheadConfig:
    mode: str = 'off'
    domain: str = 'candidate_masked'
    R_init: int = 2
    R_clamp: int = 1
    eta: float = 0.5
    T_relax: float = 1.0
    beta_relax: float | None = None

    def __post_init__(self):
        if self.mode not in ('off', 'static', 'rerelax'):
            raise ValueError("lookahead mode must be 'off', 'static', or 'rerelax'")
        if self.domain not in ('candidate_masked', 'full_vocabulary'):
            raise ValueError("lookahead domain must be 'candidate_masked' or 'full_vocabulary'")
        if not _valid_real_scalar(self.T_relax) or float(self.T_relax) <= 0:
            raise ValueError('T_relax must be finite and strictly positive')
        if not _valid_real_scalar(self.eta) or not (0.0 < float(self.eta) <= 1.0):
            raise ValueError('eta must be finite and lie in (0, 1]')
        if self.beta_relax is not None and not _valid_real_scalar(self.beta_relax):
            raise ValueError('beta_relax must be a finite real number when set')
        if not isinstance(self.R_init, int) or isinstance(self.R_init, bool) or self.R_init < 0:
            raise ValueError('R_init must be a nonnegative integer')
        if not isinstance(self.R_clamp, int) or isinstance(self.R_clamp, bool) or self.R_clamp < 0:
            raise ValueError('R_clamp must be a nonnegative integer')
        if self.mode != 'off' and self.R_init < 1:
            raise ValueError('active lookahead requires R_init >= 1')
        if self.mode == 'rerelax' and self.R_clamp < 1:
            raise ValueError('rerelax lookahead requires R_clamp >= 1')


def compute_frozen_candidates(snapshot: ProposalSnapshot, l_unit: torch.Tensor,
                              U_all: torch.Tensor, beta_relax: float = 1.0) -> tuple[torch.Tensor, ...]:
    """Single authoritative frozen C0[j] constructor shared by causal dense_candidate and lookahead."""
    if not _valid_real_scalar(beta_relax):
        raise ValueError('beta_relax must be finite')
    if snapshot.C0 is not None:
        return snapshot.C0
    work_dtype = torch.promote_types(torch.promote_types(l_unit.dtype, U_all.dtype), torch.float32)
    raw = top_candidates(l_unit.to(work_dtype) - float(beta_relax) * U_all.to(work_dtype), snapshot.M_sparse)
    return tuple(c.detach().clone() for c in raw)


def initialize_lookahead(snapshot, l_unit, H, prefix, domain='candidate_masked',
                         T_relax=1., beta_relax=1., U_precomputed=None):
    if not _valid_real_scalar(T_relax) or float(T_relax) <= 0:
        raise ValueError('T_relax must be finite and strictly positive')
    if not _valid_real_scalar(beta_relax):
        raise ValueError('beta_relax must be finite')
    if domain not in ('candidate_masked', 'full_vocabulary'):
        raise ValueError('unknown belief domain')
    T_relax_f = float(T_relax)
    beta_relax_f = float(beta_relax)
    raw_U = snapshot.unary(H) if U_precomputed is None else U_precomputed
    work_dtype = torch.promote_types(torch.promote_types(l_unit.dtype, raw_U.dtype), torch.float32)
    l_w = l_unit.to(work_dtype)
    U = raw_U.to(work_dtype)
    E_w = snapshot.E.to(work_dtype)
    if snapshot.C0 is None:
        snapshot = snapshot.with_candidates(compute_frozen_candidates(snapshot, l_w, U, beta_relax_f))
    support = [ids.clone() for ids in snapshot.C0]
    state = LookaheadState(snapshot, l_w, U, tuple(prefix), support, [None] * len(H), [], [],
                           U.clone(), domain, T_relax_f, beta_relax_f)
    for i in range(len(H)):
        scores = l_w[i] - beta_relax_f * U[i]
        ids = support[i] if domain == 'candidate_masked' else torch.arange(l_w.shape[-1], device=l_w.device)
        belief = torch.softmax(scores[ids] / T_relax_f, -1).to(work_dtype)
        state.r.append(belief)
        state.mu.append(belief @ E_w[ids])
    return state


def lookahead_fields(state: LookaheadState):
    """Pure fwd/back fields: transpose A backwards, raw masked sparse mass."""
    snap, E, A = state.snapshot, state.snapshot.E, state.snapshot.A
    dtype = state.g.dtype
    E_cast = E.to(dtype)
    A_cast = A.to(dtype)
    K = len(state.r)
    r_dim = E.shape[1]
    msg_fwd = torch.zeros((K, r_dim), dtype=dtype, device=state.g.device)
    msg_bwd = torch.zeros((K, r_dim), dtype=dtype, device=state.g.device)
    sparse_fwd = []
    sparse_bwd = []
    for i in range(K):
        own = state.support[i]
        sf = torch.zeros(len(own), dtype=dtype, device=state.g.device) if snap.sparse is not None else None
        sb = torch.zeros(len(own), dtype=dtype, device=state.g.device) if snap.sparse is not None else None
        for d in range(1, len(A_cast) + 1):
            k = i - d
            if k >= 0:
                msg_fwd[i] = msg_fwd[i] + A_cast[d - 1] @ state.mu[k].to(dtype)
                if snap.sparse is not None:
                    table = snap.sparse.lookup(d, state.support[k][:, None], own[None, :]).to(dtype)
                    sf = sf + state.raw_support_mass(k).to(dtype) @ table
            else:
                previous = prev_token(state.prefix, (), i, d)
                if previous is not None:
                    msg_fwd[i] = msg_fwd[i] + A_cast[d - 1] @ E_cast[previous]
                    if snap.sparse is not None:
                        sf = sf + snap.sparse.lookup(d, previous, own).to(dtype)
            k = i + d
            if k < K:
                msg_bwd[i] = msg_bwd[i] + A_cast[d - 1].T @ state.mu[k].to(dtype)
                if snap.sparse is not None:
                    table = snap.sparse.lookup(d, own[:, None], state.support[k][None, :]).to(dtype)
                    sb = sb + table @ state.raw_support_mass(k).to(dtype)
        sparse_fwd.append(sf)
        sparse_bwd.append(sb)
    forward = msg_fwd @ E_cast.T
    backward = msg_bwd @ E_cast.T
    if snap.sparse is not None:
        for i in range(K):
            own = state.support[i]
            forward[i].scatter_add_(0, own, sparse_fwd[i])
            backward[i].scatter_add_(0, own, sparse_bwd[i])
    return forward, backward


def lookahead_energy(state: LookaheadState):
    """One shared masked scalar energy used to verify both message derivatives."""
    snap = state.snapshot
    dtype = state.g.dtype
    E_cast = snap.E.to(dtype)
    A_cast = snap.A.to(dtype)
    U_cast = state.U_all.to(dtype)
    energy = sum((state.full_belief(i).to(dtype) * U_cast[i]).sum() for i in range(len(state.r)))
    for i in range(len(state.r)):
        mu_i = state.mu[i].to(dtype)
        raw_i = state.raw_support_mass(i).to(dtype)
        for d in range(1, len(A_cast) + 1):
            k = i - d
            if k >= 0:
                energy = energy + mu_i @ A_cast[d - 1] @ state.mu[k].to(dtype)
                if snap.sparse is not None:
                    pairs = snap.sparse.lookup(d, state.support[k][:, None], state.support[i][None, :]).to(dtype)
                    energy = energy + state.raw_support_mass(k).to(dtype) @ pairs @ raw_i
            else:
                previous = prev_token(state.prefix, (), i, d)
                if previous is not None:
                    energy = energy + mu_i @ A_cast[d - 1] @ E_cast[previous]
                    if snap.sparse is not None:
                        energy = energy + raw_i @ snap.sparse.lookup(d, previous, state.support[i]).to(dtype)
    return energy


def damped_lookahead_step(state: LookaheadState, eta=0.5):
    if not _valid_real_scalar(eta) or not (0.0 < float(eta) <= 1.0):
        raise ValueError('eta must be finite and in (0, 1]')
    eta_f = float(eta)
    dtype = state.g.dtype
    E_cast = state.snapshot.E.to(dtype)
    forward, backward = lookahead_fields(state)
    tilde = state.U_all.to(dtype) + forward + backward
    # All fields are computed before any belief changes: genuinely synchronous.
    fields = (1.0 - eta_f) * state.g + eta_f * tilde
    r_new, mu_new = list(state.r), list(state.mu)
    for i in range(len(state.r)):
        if state.clamped[i] is not None:
            fields[i] = state.g[i]
            continue
        ids = state.support[i] if state.domain == 'candidate_masked' else torch.arange(state.g.shape[-1], device=state.g.device)
        r_new[i] = torch.softmax((state.l_unit[i, ids] - state.beta_relax * fields[i, ids]) / state.T_relax, -1).to(dtype)
        mu_new[i] = r_new[i] @ E_cast[ids]
    state.g, state.r, state.mu = fields, r_new, mu_new
    return state
