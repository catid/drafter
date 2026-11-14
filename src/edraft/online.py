"""Request-local, synchronous live-full adaptation and normalized bigram baseline."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Callable
import torch

from .adapter import (ProposalSnapshot, block_predecessor_pairs, block_unary,
                      slot_correction, top_candidates, prev_token, full_kl, loss_weights)
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
        self._cache: dict[int, dict] = {}

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
                self._cache.pop(previous, None)
            previous = token
        if len(committed_tokens):
            self.version += 1

    def _prior_parts(self, prev_int: int, counts: Counter, device, dtype):
        target_device = torch.device('cpu') if device is None else torch.device(device)
        cache_key = (self.alpha, self.k_ret, target_device, dtype)
        bucket = self._cache.get(prev_int)
        total = sum(counts.values())
        if bucket is not None:
            cached = bucket.get(cache_key)
            if cached is not None and cached[0] == total and cached[1] == len(counts):
                return cached[2], cached[3], cached[4]
        else:
            bucket = {}
            self._cache[prev_int] = bucket
        retained = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        if self.k_ret is not None:
            retained = retained[:self.k_ret]
        evicted = total - sum(count for _, count in retained)
        absent = self.vocab_size - len(retained)
        denom = total + self.alpha * self.vocab_size
        base_val = torch.tensor(evicted / absent if absent else 0., device=target_device, dtype=dtype)
        base_lp = torch.log((base_val + self.alpha) / denom)
        if retained:
            ids_t = torch.tensor([key for key, _ in retained], device=target_device, dtype=torch.long)
            ret_counts = torch.tensor([count for _, count in retained], device=target_device, dtype=dtype)
            ret_lp = torch.log((ret_counts + self.alpha) / denom)
        else:
            ids_t = torch.empty(0, device=target_device, dtype=torch.long)
            ret_lp = torch.empty(0, device=target_device, dtype=dtype)
        bucket[cache_key] = (total, len(counts), base_lp, ids_t, ret_lp)
        return base_lp, ids_t, ret_lp

    def prior(self, previous: int | None, device=None, dtype=torch.float32):
        if previous is None:
            return torch.zeros(self.vocab_size, device=device, dtype=dtype)
        prev_int = int(previous)
        counts = self.counts.get(prev_int)
        if not counts:
            return torch.full((self.vocab_size,), -math.log(float(self.vocab_size)), device=device, dtype=dtype)
        base_lp, ids_t, ret_lp = self._prior_parts(prev_int, counts, device, dtype)
        out = base_lp.expand(self.vocab_size).clone()
        if ids_t.numel():
            out[ids_t] = ret_lp
        return out

    def score(self, l_unit, previous: int | None):
        if previous is None or self.lambda_tr == 0:
            return l_unit
        prev_int = int(previous)
        counts = self.counts.get(prev_int)
        if not counts:
            return l_unit
        base_lp, ids_t, ret_lp = self._prior_parts(prev_int, counts, l_unit.device, l_unit.dtype)
        out = l_unit + self.lambda_tr * base_lp
        if ids_t.numel():
            out[ids_t] = l_unit.index_select(0, ids_t) + self.lambda_tr * ret_lp
        return out

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
                 enable_sparse: bool | None = None, bias_sparse_only: bool = False,
                 weighting_mode: str = 'exponential', gamma: float = 5.,
                 fastpath=None):
        if update_stride < 1 or min(lambda_l2, lambda_l1_b, lambda_l1_s) < 0:
            raise ValueError('invalid online stride or regularization')
        if not callable(lr) and lr < 0:
            raise ValueError('learning rate must be nonnegative')
        self.offline = offline.snapshot() if hasattr(offline, 'snapshot') else offline
        self.lr, self.update_stride = lr, update_stride
        self.lambda_l2, self.lambda_l1_b, self.lambda_l1_s = lambda_l2, lambda_l1_b, lambda_l1_s
        if weighting_mode not in ('exponential', 'rejection_relative', 'cutoff') or gamma <= 0:
            raise ValueError('invalid online weighting mode or gamma')
        self.weighting_mode, self.gamma = weighting_mode, gamma
        self.bias_sparse_only = bias_sparse_only
        self.fastpath = fastpath
        self.dW = torch.zeros_like(self.offline.W, dtype=torch.float32)
        self.dA = torch.zeros_like(self.offline.A, dtype=torch.float32)
        self.db = torch.zeros_like(self.offline.b, dtype=torch.float32)
        enable_sparse = self.offline.sparse is not None if enable_sparse is None else enable_sparse
        self.dS = LazySignedHash(sparse_capacity, self.db.device) if enable_sparse else None
        self.u = 0
        self.version = 0
        self.last_round = 0
        self._last_loss_tensor: torch.Tensor | float | None = None
        self.state_bytes_estimate = {'dense_activation_clone': 0, 'dense_optimizer': 0, 'dense_snapshot': 0,
                                     'sparse_update': 0, 'sparse_snapshot': 0}
        self.active_bucket_min = None
        self.active_bucket_max = 0
        self._snapshot = self._make_snapshot()

    @property
    def last_loss(self) -> float | None:
        if self._last_loss_tensor is None:
            return None
        return float(self._last_loss_tensor)

    @last_loss.setter
    def last_loss(self, value: torch.Tensor | float | None):
        self._last_loss_tensor = value

    def _make_snapshot(self):
        terms = []
        if self.offline.sparse is not None:
            terms.append(self.offline.sparse)
        if self.dS is not None:
            terms.append(self.dS.snapshot())
        sparse = SumSparse(tuple(terms)) if terms else None
        W, A, b = (self.offline.W + self.dW).detach(), (self.offline.A + self.dA).detach(), (self.offline.b + self.db).detach()
        if not self.offline.is_zero:
            is_zero = False
        elif self.u == 0:
            is_zero = self.dS is None or len(self.dS.active) == 0
        else:
            is_zero = (self.dS is None or len(self.dS.active) == 0) and not bool(
                torch.count_nonzero(self.db) or torch.count_nonzero(self.dW) or torch.count_nonzero(self.dA)
            )
        return ProposalSnapshot(self.offline.E, W, A, b, self.version, sparse, self.offline.M_sparse, None, is_zero)

    def snapshot(self, C0=None):
        return self._snapshot if C0 is None else self._snapshot.with_candidates(C0)

    def update(self, round_idx, l_unit, target_logits, H, prefix, tokens, weights=None, first_rejection=None):
        if round_idx < 1 or round_idx <= self.last_round:
            raise ValueError('round_idx must increase and is one-based')
        self.last_round = round_idx
        if round_idx % self.update_stride:
            return False
        # Permit a caller whose model loop uses no_grad or inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            if weights is None:
                weights = loss_weights(len(l_unit), l_unit.device, self.weighting_mode, self.gamma, first_rejection)
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
        K = len(H)
        U = block_unary(E, W, b, H)
        valid_slots, valid_d0, valid_prev = block_predecessor_pairs(prefix, tokens, K, len(A), E.device)
        if valid_slots.numel():
            E_cast = E.to(A.dtype)
            E_prev = E_cast.index_select(0, valid_prev)
            msgs = torch.bmm(A.index_select(0, valid_d0), E_prev.unsqueeze(-1)).squeeze(-1)
            slot_msgs = torch.zeros(K, E.shape[1], dtype=A.dtype, device=A.device).index_add(0, valid_slots, msgs)
            dense = U + slot_msgs @ E_cast.T
        else:
            dense = U
        dirty = sparse_values = None
        if self.dS is not None:
            if valid_slots.numel():
                candidates = top_candidates(l_unit - dense, self.offline.M_sparse)
                cands_pair = candidates.index_select(0, valid_slots)
                d_pair = (valid_d0 + 1)[:, None]
                prev_pair = valid_prev[:, None]
                buckets, signs = hash_keys(d_pair, prev_pair, cands_pair, self.dS.capacity, E.device)
                dirty, inverse = torch.unique(buckets.reshape(-1), sorted=True, return_inverse=True)
                sparse_values = self.dS.read(dirty).detach().clone().requires_grad_(True)
                pair_vals = sparse_values[inverse].reshape(buckets.shape) * signs
                if self.offline.sparse is not None:
                    pair_vals = pair_vals + self.offline.sparse.lookup(d_pair, prev_pair, cands_pair)
                slot_vals = torch.zeros(K, candidates.shape[1], dtype=dense.dtype, device=dense.device).index_add(
                    0, valid_slots, pair_vals.to(dense.dtype))
                correction = dense.scatter_add(1, candidates, slot_vals)
            else:
                dirty = torch.empty(0, device=E.device, dtype=torch.long)
                correction = dense
        elif self.offline.sparse is not None and valid_slots.numel():
            candidates = top_candidates(l_unit - dense, self.offline.M_sparse)
            cands_pair = candidates.index_select(0, valid_slots)
            pair_vals = self.offline.sparse.lookup((valid_d0 + 1)[:, None], valid_prev[:, None], cands_pair).to(dense.dtype)
            slot_vals = torch.zeros(K, candidates.shape[1], dtype=dense.dtype, device=dense.device).index_add(
                0, valid_slots, pair_vals)
            correction = dense.scatter_add(1, candidates, slot_vals)
        else:
            correction = dense
        use_fast_kl = (self.fastpath is not None
                       and (self.fastpath.static_block_length is None or K == self.fastpath.static_block_length))
        kl_fn = self.fastpath.full_kl_fn if use_fast_kl else full_kl
        loss = kl_fn(l_unit - correction, target_logits, weights)
        loss.backward()
        with torch.no_grad():
            shrink = 1 + lr * self.lambda_l2
            if not self.bias_sparse_only:
                self.dW = (dW - lr * (dW.grad if dW.grad is not None else torch.zeros_like(dW))) / shrink
                self.dA = (dA - lr * (dA.grad if dA.grad is not None else torch.zeros_like(dA))) / shrink
            self.db = proximal_l1((db - lr * db.grad) / shrink, lr * self.lambda_l1_b)
            if self.dS is not None:
                if sparse_values is not None:
                    gradient = sparse_values.grad if sparse_values.grad is not None else torch.zeros_like(sparse_values)
                else:
                    gradient = torch.empty(0, device=E.device, dtype=torch.float32)
                self.dS.step_unique(dirty, gradient, lr, self.lambda_l1_s)
            self.u += 1
            self.version += 1
            self._last_loss_tensor = loss.detach()
            self._snapshot = self._make_snapshot()
            dense_bytes = sum(x.numel() * x.element_size() for x in (self.dW, self.dA, self.db))
            optimized_bytes = self.db.numel() * self.db.element_size() if self.bias_sparse_only else dense_bytes
            self.state_bytes_estimate['dense_activation_clone'] += 2 * dense_bytes
            # Conceptual reads/writes, not a profiler-derived hardware traffic count.
            self.state_bytes_estimate['dense_optimizer'] += 5 * optimized_bytes + 2 * self.db.numel() * self.db.element_size()
            self.state_bytes_estimate['dense_snapshot'] += 3 * dense_bytes
            if self.dS is not None:
                active = len(self.dS.active)
                snapshot_bytes = active * 16 + (self.dS.u + 1) * 4
                self.state_bytes_estimate['sparse_update'] += self.dS.last_update_bytes
                self.state_bytes_estimate['sparse_snapshot'] += 2 * snapshot_bytes
                self.active_bucket_min = active if self.active_bucket_min is None else min(self.active_bucket_min, active)
                self.active_bucket_max = max(self.active_bucket_max, active)
        return True

    def metrics(self):
        # Evaluate current lazy shrinkage without materializing it into storage.
        effective_dS = self.dS.read(self.dS.active) if self.dS is not None else None
        return {'adapter_version': self.version, 'u': self.u, 'last_round': self.last_round,
                'loss_mode': 'live_full', 'T_train': 1., 'beta_train': 1.,
                'dW_norm': float(self.dW.norm()), 'dA_norm': float(self.dA.norm()),
                'db_norm': float(self.db.norm()),
                'dS_norm': float(effective_dS.norm()) if effective_dS is not None else 0.,
                'dS_norm_definition': 'L2 norm of effective lazy-shrunk request delta at current optimizer u; zero unrepresented buckets; pure read',
                'db_exact_zero_fraction': float((self.db == 0).float().mean()),
                'active_buckets': len(self.dS.active) if self.dS is not None else 0,
                'active_buckets_min_over_updates': self.active_bucket_min,
                'active_buckets_max_over_updates': self.active_bucket_max,
                'state_bytes_estimate_by_operation': dict(self.state_bytes_estimate),
                'state_bytes_estimate_total': sum(self.state_bytes_estimate.values()),
                'state_bytes_definition': 'conceptual FP32 parameter/timestamp/index reads+writes for state clones, SGD/prox steps and snapshots; excludes E/logit/autograd activation traffic and is not hardware HBM traffic',
                'sparse_capacity': self.dS.capacity if self.dS is not None else 0,
                'sparse_update_bytes_estimate': self.dS.last_update_bytes if self.dS is not None else 0,
                'sparse_snapshot_bytes': (len(self.dS.active) * 16 + (self.dS.u + 1) * 4) if self.dS is not None else 0,
                'dense_snapshot_bytes': sum(x.numel() * x.element_size() for x in (self._snapshot.W, self._snapshot.A, self._snapshot.b)),
                'sparse_snapshot_strategy': 'compact_active_bucket_copy',
                'sparse_update_strategy': 'dirty_lazy_catchup_and_active_list_compaction',
                'last_full_loss': self.last_loss}
