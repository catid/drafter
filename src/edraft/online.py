"""Request-local, synchronous live-full adaptation and normalized bigram baseline."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable
import torch

from .adapter import (ProposalSnapshot, block_unary, slot_correction, top_candidates,
                      prev_token, full_kl)
from .sparse import LazySignedHash, SumSparse, hash_keys, proximal_l1


class OnlineBigram:
    """Exact successor counts; optional top-k retained-prior memory ablation.

    Full counts are kept as the authoritative count ledger in the eviction
    diagnostic, so a reappearing retained ID never receives an approximate count.
    Memory reports include that ledger; this ablation does not claim bounded
    memory savings over exact counts.
    """
    def __init__(self, vocab_size: int, alpha: float = 1., lambda_tr: float = 0.1, k_ret: int | None = None):
        if alpha <= 0:
            raise ValueError('alpha must be strictly positive')
        if vocab_size < 1 or (k_ret is not None and k_ret < 1):
            raise ValueError('invalid vocabulary or retention count')
        self.vocab_size, self.alpha, self.lambda_tr, self.k_ret = vocab_size, alpha, lambda_tr, k_ret
        self.counts = defaultdict(Counter)
        self.version = 0

    def update(self, prefix_tail, committed_tokens):
        previous = int(prefix_tail[-1]) if len(prefix_tail) else None
        for token in committed_tokens:
            token = int(token)
            if not 0 <= token < self.vocab_size:
                raise ValueError('committed token outside vocabulary')
            if previous is not None:
                if not 0 <= previous < self.vocab_size:
                    raise ValueError('previous token outside vocabulary')
                self.counts[previous][token] += 1
            previous = token
        if len(committed_tokens):
            self.version += 1

    def prior(self, previous: int | None, device=None, dtype=torch.float32):
        if previous is None:
            return torch.zeros(self.vocab_size, device=device, dtype=dtype)
        if not self.counts.get(int(previous)):
            return torch.full((self.vocab_size,), -torch.log(torch.tensor(float(self.vocab_size))).item(), device=device, dtype=dtype)
        counts = self.counts[int(previous)]
        total = sum(counts.values())
        retained = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        if self.k_ret is not None:
            retained = retained[:self.k_ret]
        evicted = total - sum(count for _, count in retained)
        absent = self.vocab_size - len(retained)
        effective = torch.full((self.vocab_size,), evicted / absent if absent else 0., device=device, dtype=dtype)
        if retained:
            ids = torch.tensor([key for key, _ in retained], device=device, dtype=torch.long)
            effective[ids] = torch.tensor([count for _, count in retained], device=device, dtype=dtype)
        return torch.log((effective + self.alpha) / (total + self.alpha * self.vocab_size))

    def score(self, l_unit, previous: int | None):
        if previous is None or self.lambda_tr == 0 or not self.counts.get(int(previous)):
            return l_unit
        return l_unit + self.lambda_tr * self.prior(previous, device=l_unit.device, dtype=l_unit.dtype)

    def metrics(self):
        retained = sum(min(len(c), self.k_ret) if self.k_ret is not None else len(c) for c in self.counts.values())
        all_pairs = sum(len(c) for c in self.counts.values())
        return {'counted_transitions': sum(sum(c.values()) for c in self.counts.values()),
                'exact_ledger_pairs': all_pairs, 'retained_prior_pairs': retained,
                'ledger_retained_for_exact_counts': True, 'k_ret': self.k_ret}


@dataclass(frozen=True)
class _TrainingSparse:
    offline: object | None
    indices: torch.Tensor
    values: torch.Tensor
    capacity: int

    def lookup(self, d, previous, current):
        buckets, signs = hash_keys(d, previous, current, self.capacity, self.values.device)
        pos = torch.searchsorted(self.indices, buckets)
        delta = self.values[pos] * signs
        return delta if self.offline is None else delta + self.offline.lookup(d, previous, current)


class OnlineState:
    """One request's deltas. Call update only after completing target verification.

    round_idx is one-based. Skipped rounds retain no activations or gradients.
    u counts optimizer steps; version changes exactly once per executed update.
    """
    def __init__(self, offline, lr: float | Callable[[int], float] = 0.01,
                 update_stride: int = 1, lambda_l2: float = 0., lambda_l1_b: float = 0.,
                 lambda_l1_s: float = 0., sparse_capacity: int = 2**20,
                 enable_sparse: bool | None = None, bias_sparse_only: bool = False):
        if update_stride < 1 or min(lambda_l2, lambda_l1_b, lambda_l1_s) < 0:
            raise ValueError('invalid online stride or regularization')
        if not callable(lr) and lr < 0:
            raise ValueError('learning rate must be nonnegative')
        self.offline = offline.snapshot() if hasattr(offline, 'snapshot') else offline
        self.lr, self.update_stride = lr, update_stride
        self.lambda_l2, self.lambda_l1_b, self.lambda_l1_s = lambda_l2, lambda_l1_b, lambda_l1_s
        self.bias_sparse_only = bias_sparse_only
        self.dW = torch.zeros_like(self.offline.W, dtype=torch.float32)
        self.dA = torch.zeros_like(self.offline.A, dtype=torch.float32)
        self.db = torch.zeros_like(self.offline.b, dtype=torch.float32)
        enable_sparse = self.offline.sparse is not None if enable_sparse is None else enable_sparse
        self.dS = LazySignedHash(sparse_capacity, self.db.device) if enable_sparse else None
        self.u = 0
        self.version = 0
        self.last_round = 0
        self.last_loss = None
        self._snapshot = self._make_snapshot()

    def _make_snapshot(self):
        terms = []
        if self.offline.sparse is not None:
            terms.append(self.offline.sparse)
        if self.dS is not None:
            terms.append(self.dS.snapshot())
        sparse = SumSparse(tuple(terms)) if terms else None
        W, A, b = (self.offline.W + self.dW).detach(), (self.offline.A + self.dA).detach(), (self.offline.b + self.db).detach()
        is_zero = self.offline.is_zero and all(not bool(torch.count_nonzero(x)) for x in (self.dW, self.dA, self.db))
        if self.dS is not None:
            is_zero = is_zero and len(self.dS.active) == 0
        return ProposalSnapshot(self.offline.E, W, A, b, self.version, sparse, self.offline.M_sparse, None, is_zero)

    def snapshot(self, C0=None):
        return self._snapshot if C0 is None else self._snapshot.with_candidates(C0)

    def update(self, round_idx, l_unit, target_logits, H, prefix, tokens, weights=None):
        if round_idx < 1 or round_idx <= self.last_round:
            raise ValueError('round_idx must increase and is one-based')
        self.last_round = round_idx
        if round_idx % self.update_stride:
            return False
        # Permit a caller whose model loop uses no_grad or inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            return self._update(l_unit, target_logits, H, prefix, tokens, weights)

    def _update(self, l_unit, target_logits, H, prefix, tokens, weights):
        lr = float(self.lr(self.u + 1) if callable(self.lr) else self.lr)
        if lr < 0:
            raise ValueError('scheduled learning rate must be nonnegative')
        l_unit, target_logits, H = (x.detach().float().clone() for x in (l_unit, target_logits, H))
        if weights is not None:
            weights = weights.detach().clone()
        dW = self.dW.detach().clone().requires_grad_(not self.bias_sparse_only)
        dA = self.dA.detach().clone().requires_grad_(not self.bias_sparse_only)
        db = self.db.detach().clone().requires_grad_(True)
        W, A, b = self.offline.W + dW, self.offline.A + dA, self.offline.b + db
        E = self.offline.E
        U = block_unary(E, W, b, H)
        sparse = self.offline.sparse
        dirty = sparse_values = None
        if self.dS is not None:
            accesses = []
            for i in range(len(H)):
                dense = slot_correction(E, A, U[i], l_unit[i], prefix, tokens, i)
                candidates = top_candidates(l_unit[i] - dense, self.offline.M_sparse)
                for d in range(1, len(A) + 1):
                    previous = prev_token(prefix, tokens, i, d)
                    if previous is not None:
                        accesses.append(hash_keys(d, previous, candidates, self.dS.capacity, E.device)[0])
            dirty = torch.unique(torch.cat(accesses), sorted=True) if accesses else torch.empty(0, device=E.device, dtype=torch.long)
            sparse_values = self.dS.read(dirty).detach().clone().requires_grad_(True)
            sparse = _TrainingSparse(self.offline.sparse, dirty, sparse_values, self.dS.capacity)
        correction = torch.stack([slot_correction(E, A, U[i], l_unit[i], prefix, tokens, i,
                                                 sparse, self.offline.M_sparse) for i in range(len(H))])
        loss = full_kl(l_unit - correction, target_logits, weights)
        loss.backward()
        with torch.no_grad():
            shrink = 1 + lr * self.lambda_l2
            if not self.bias_sparse_only:
                self.dW = (dW - lr * dW.grad) / shrink
                self.dA = (dA - lr * dA.grad) / shrink
            self.db = proximal_l1((db - lr * db.grad) / shrink, lr * self.lambda_l1_b)
            if self.dS is not None:
                gradient = sparse_values.grad if sparse_values.grad is not None else torch.zeros_like(sparse_values)
                self.dS.step(dirty, gradient, lr, self.lambda_l1_s)
            self.u += 1
            self.version += 1
            self.last_loss = float(loss.detach())
            self._snapshot = self._make_snapshot()
        return True

    def metrics(self):
        return {'adapter_version': self.version, 'u': self.u, 'last_round': self.last_round,
                'loss_mode': 'live_full', 'T_train': 1., 'beta_train': 1.,
                'dW_norm': float(self.dW.norm()), 'dA_norm': float(self.dA.norm()),
                'db_norm': float(self.db.norm()), 'db_exact_zero_fraction': float((self.db == 0).float().mean()),
                'active_buckets': len(self.dS.active) if self.dS is not None else 0,
                'sparse_capacity': self.dS.capacity if self.dS is not None else 0,
                'sparse_update_bytes_estimate': self.dS.last_update_bytes if self.dS is not None else 0,
                'sparse_snapshot_bytes': (len(self.dS.active) * 16 + (self.dS.u + 1) * 4) if self.dS is not None else 0,
                'dense_snapshot_bytes': sum(x.numel() * x.element_size() for x in (self._snapshot.W, self._snapshot.A, self._snapshot.b)),
                'sparse_snapshot_strategy': 'compact_active_bucket_copy',
                'sparse_update_strategy': 'dirty_lazy_catchup_and_active_list_compaction',
                'last_full_loss': self.last_loss}
