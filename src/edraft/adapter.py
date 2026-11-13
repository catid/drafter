"""Causal energy correction and standalone synchronous lookahead mathematics."""
from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def D(self):
        return len(self.A)

    @property
    def adapter_version(self):
        return self.version

    def unary(self, H, proposal_mode='dense_full'):
        if proposal_mode == 'dense_full':
            return block_unary(self.E, self.W, self.b, H)
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
                    message = torch.zeros(self.E.shape[1], dtype=self.A.dtype, device=self.A.device)
                    predecessors = []
                    for d in range(1, len(self.A) + 1):
                        previous = prev_token(prefix, realized, i, d)
                        if previous is not None:
                            message = message + self.A[d - 1] @ self.E[previous]
                            predecessors.append((d, previous))
                    correction = unary + self.E[ids] @ message
                if self.sparse is not None and predecessors:
                    with _phase(timer, 'sparse_lookup'):
                        offsets = torch.tensor([d for d, _ in predecessors], device=ids.device)[:, None]
                        previous_ids = torch.tensor([p for _, p in predecessors], device=ids.device)[:, None]
                        correction = correction + self.sparse.lookup(offsets, previous_ids, ids[None, :]).sum(0)
                selected = l_unit_i[ids] - beta * correction
            return torch.full_like(l_unit_i, -torch.inf).scatter(0, ids, selected)
        if proposal_mode != 'dense_full':
            raise ValueError('unknown proposal mode')
        if beta == 0 or self.is_zero:
            return l_unit_i
        correction = slot_correction(self.E, self.A, U_i, l_unit_i, prefix, realized, i,
                                     self.sparse, self.M_sparse, beta, timer=timer)
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
        return torch.stack([slot_correction(self.E, self.A, U[i], l_unit[i], prefix, tokens, i,
                                           self.sparse, self.M_sparse, beta) for i in range(len(H))])

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
    """Exact q partition with one grouped target-tail category (not full KL)."""
    q_log = torch.log_softmax(scores.float(), -1)
    losses = []
    for i, ids in enumerate(union_ids):
        p = target_union_probabilities[i].float().detach()
        log_q_u = q_log[i, ids]
        # Stable complement through logsumexp over the actual unstored vocabulary.
        rest_mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        rest_mask[ids] = False
        log_q_rest = torch.logsumexp(q_log[i, rest_mask], 0) if bool(rest_mask.any()) else scores.new_tensor(-torch.inf)
        rest_p = torch.as_tensor(target_rest_mass[i], device=scores.device, dtype=torch.float32).detach()
        union_loss = torch.xlogy(p, p).sum() - (p * log_q_u).sum()
        rest_loss = torch.where(rest_p > 0, torch.xlogy(rest_p, rest_p) - rest_p * log_q_rest.clamp_min(-1e30), rest_p * 0)
        losses.append(union_loss + rest_loss)
    losses = torch.stack(losses)
    if weights is None:
        weights = loss_weights(len(scores), scores.device)
    return (losses * weights).sum()


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
