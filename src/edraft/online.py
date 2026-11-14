"""Request-local, synchronous live-full adaptation and normalized bigram baseline."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Callable
import torch

from .adapter import (ProposalSnapshot, block_unary, slot_correction, two_sided_slot_correction,
                      top_candidates, prev_token, next_token, full_kl,
                      two_sided_pseudolikelihood_loss, loss_weights)
from .sparse import (LazySignedHash, LazyExactObservedPairs, SumSparse,
                     hash_keys, pack_exact_keys, proximal_l1)


class OnlineBigram:
    """Exact successor counts; optional top-k retained-prior memory ablation.

    Full counts are kept as the authoritative count ledger in the eviction
    diagnostic, so a reappearing retained ID never receives an approximate count.
    Memory reports include that ledger; this ablation does not claim bounded
    memory savings over exact counts.
    """
    def __init__(self, vocab_size: int, alpha: float = 1., lambda_tr: float = 0.1,
                 k_ret: int | None = None, *, count_gate_kappa: float | None = None):
        if alpha <= 0:
            raise ValueError('alpha must be strictly positive')
        if vocab_size < 1 or (k_ret is not None and k_ret < 1):
            raise ValueError('invalid vocabulary or retention count')
        if count_gate_kappa is not None:
            if isinstance(count_gate_kappa, bool) or not math.isfinite(float(count_gate_kappa)) or float(count_gate_kappa) <= 0:
                raise ValueError('count_gate_kappa must be a finite positive float when set')
            count_gate_kappa = float(count_gate_kappa)
        self.vocab_size, self.alpha, self.lambda_tr, self.k_ret = vocab_size, alpha, lambda_tr, k_ret
        self.count_gate_kappa = count_gate_kappa
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

    def effective_lambda(self, previous: int | None) -> float:
        if previous is None or self.lambda_tr == 0:
            return 0.0
        counts = self.counts.get(int(previous))
        if not counts:
            return 0.0
        if self.count_gate_kappa is None:
            return float(self.lambda_tr)
        n_prev = sum(counts.values())
        return float(self.lambda_tr) * (n_prev / (n_prev + self.count_gate_kappa))

    def score(self, l_unit, previous: int | None):
        lam = self.effective_lambda(previous)
        if lam == 0.0:
            return l_unit
        return l_unit + lam * self.prior(previous, device=l_unit.device, dtype=l_unit.dtype)

    def metrics(self):
        retained = sum(min(len(c), self.k_ret) if self.k_ret is not None else len(c) for c in self.counts.values())
        all_pairs = sum(len(c) for c in self.counts.values())
        out = {'counted_transitions': sum(sum(c.values()) for c in self.counts.values()),
               'exact_ledger_pairs': all_pairs, 'retained_prior_pairs': retained,
               'ledger_retained_for_exact_counts': True, 'k_ret': self.k_ret}
        if self.count_gate_kappa is not None:
            out['count_gate_kappa'] = self.count_gate_kappa
        return out


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


@dataclass(frozen=True)
class _TrainingExactSparse:
    offline: object | None
    keys: torch.Tensor
    values: torch.Tensor
    vocab_size: int

    def lookup(self, d, previous, current):
        packed = pack_exact_keys(d, previous, current, self.vocab_size, device=self.values.device)
        if self.keys.numel() == 0:
            delta = torch.zeros(packed.shape, dtype=torch.float32, device=self.values.device)
        else:
            pos = torch.searchsorted(self.keys, packed)
            safe = pos.clamp_max(len(self.keys) - 1)
            present = (pos < len(self.keys)) & (self.keys[safe] == packed)
            delta = torch.where(present, self.values[safe], torch.zeros_like(self.values[safe]))
        return delta if self.offline is None else delta + self.offline.lookup(d, previous, current)


class OnlineState:
    """One request's deltas. Call update only after completing target verification.

    round_idx is one-based. Skipped rounds retain no activations or gradients
    unless M5 `accumulate_stride=True` explicitly buffers compact parameter
    gradients across the `S` rounds of each stride window.
    u counts optimizer steps; version changes exactly once per executed update.
    """
    def __init__(self, offline, lr: float | Callable[[int], float] = 0.01,
                 update_stride: int = 1, lambda_l2: float = 0., lambda_l1_b: float = 0.,
                 lambda_l1_s: float = 0., sparse_capacity: int = 2**20,
                 enable_sparse: bool | None = None, bias_sparse_only: bool = False,
                 weighting_mode: str = 'exponential', gamma: float = 5.,
                 *, sparse_backend: str = 'signed_feature_hash',
                 sparse_overflow_policy: str = 'freeze_new',
                 accumulate_stride: bool = False,
                 buffer_reduction: str = 'mean',
                 loss_mode: str = 'live_full'):
        if update_stride < 1 or min(lambda_l2, lambda_l1_b, lambda_l1_s) < 0:
            raise ValueError('invalid online stride or regularization')
        if not callable(lr) and lr < 0:
            raise ValueError('learning rate must be nonnegative')
        if sparse_backend not in ('signed_feature_hash', 'exact_observed_pairs'):
            raise ValueError("sparse_backend must be 'signed_feature_hash' or 'exact_observed_pairs'")
        if buffer_reduction not in ('mean', 'sum'):
            raise ValueError("buffer_reduction must be 'mean' or 'sum'")
        if loss_mode not in ('live_full', 'two_sided_pseudolikelihood'):
            raise ValueError("loss_mode must be 'live_full' or 'two_sided_pseudolikelihood'")
        self.offline = offline.snapshot() if hasattr(offline, 'snapshot') else offline
        self.lr, self.update_stride = lr, update_stride
        self.lambda_l2, self.lambda_l1_b, self.lambda_l1_s = lambda_l2, lambda_l1_b, lambda_l1_s
        if weighting_mode not in ('exponential', 'rejection_relative', 'cutoff') or gamma <= 0:
            raise ValueError('invalid online weighting mode or gamma')
        self.weighting_mode, self.gamma = weighting_mode, gamma
        self.bias_sparse_only = bias_sparse_only
        self.sparse_backend = sparse_backend
        self.sparse_overflow_policy = sparse_overflow_policy
        self.accumulate_stride = bool(accumulate_stride)
        self.buffer_reduction = buffer_reduction
        self.loss_mode = loss_mode
        self.dW = torch.zeros_like(self.offline.W, dtype=torch.float32)
        self.dA = torch.zeros_like(self.offline.A, dtype=torch.float32)
        self.db = torch.zeros_like(self.offline.b, dtype=torch.float32)
        enable_sparse = self.offline.sparse is not None if enable_sparse is None else enable_sparse
        if not enable_sparse:
            self.dS = None
        elif sparse_backend == 'signed_feature_hash':
            self.dS = LazySignedHash(sparse_capacity, self.db.device)
        else:
            self.dS = LazyExactObservedPairs(
                sparse_capacity,
                vocab_size=self.offline.E.shape[0],
                overflow_policy=sparse_overflow_policy,
                device=self.db.device,
            )
        self.u = 0
        self.version = 0
        self.last_round = 0
        self.last_loss = None
        self._buffered_rounds = 0
        self.buffered_rounds_total = 0
        self._buf_dW: torch.Tensor | None = None
        self._buf_dA: torch.Tensor | None = None
        self._buf_db: torch.Tensor | None = None
        self._buf_sparse_idx: list[torch.Tensor] = []
        self._buf_sparse_grad: list[torch.Tensor] = []
        self._buf_loss = 0.0
        self.state_bytes_estimate = {'dense_activation_clone': 0, 'dense_optimizer': 0, 'dense_snapshot': 0,
                                     'sparse_update': 0, 'sparse_snapshot': 0}
        self.active_bucket_min = None
        self.active_bucket_max = 0
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

    def update(self, round_idx, l_unit, target_logits, H, prefix, tokens,
               weights=None, first_rejection=None, committed_tokens=None):
        if round_idx < 1 or round_idx <= self.last_round:
            raise ValueError('round_idx must increase and is one-based')
        self.last_round = round_idx
        if not self.accumulate_stride and (round_idx % self.update_stride):
            return False
        # Permit a caller whose model loop uses no_grad or inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            return self._update(round_idx, l_unit, target_logits, H, prefix, tokens,
                                weights=weights, first_rejection=first_rejection,
                                committed_tokens=committed_tokens)

    def _compute_round_gradients(self, l_unit, target_logits, H, prefix, tokens,
                                 weights=None, first_rejection=None, committed_tokens=None):
        l_unit = l_unit.detach().float().clone()
        H = H.detach().float().clone()
        target_logits = target_logits.detach().float().clone() if target_logits is not None else None
        dW = self.dW.detach().clone().requires_grad_(not self.bias_sparse_only)
        dA = self.dA.detach().clone().requires_grad_(not self.bias_sparse_only)
        db = self.db.detach().clone().requires_grad_(True)
        W, A, b = self.offline.W + dW, self.offline.A + dA, self.offline.b + db
        E = self.offline.E
        V = E.shape[0]
        dirty = sparse_values = None
        if self.loss_mode == 'two_sided_pseudolikelihood':
            if committed_tokens is None:
                if first_rejection is None or first_rejection == len(tokens):
                    committed = [int(t) for t in tokens]
                else:
                    if target_logits is None:
                        raise ValueError('two_sided_pseudolikelihood requires committed_tokens or target_logits')
                    committed = [int(t) for t in tokens[:first_rejection]] + [int(target_logits[first_rejection].argmax())]
            else:
                committed = [int(t) for t in committed_tokens]
            if not committed:
                raise ValueError('two_sided_pseudolikelihood requires at least one committed verified token')
            n_eval = min(len(H), len(committed))
            U = block_unary(E, W, b, H[:n_eval])
            sparse = self.offline.sparse
            if self.dS is not None:
                accesses = []
                for i in range(n_eval):
                    dense = two_sided_slot_correction(E, A, U[i], l_unit[i], prefix, committed, i)
                    candidates = top_candidates(l_unit[i] - dense, self.offline.M_sparse)
                    for d in range(1, len(A) + 1):
                        previous = prev_token(prefix, committed, i, d)
                        if previous is not None:
                            if self.sparse_backend == 'signed_feature_hash':
                                accesses.append(hash_keys(d, previous, candidates, self.dS.capacity, E.device)[0])
                            else:
                                accesses.append(pack_exact_keys(d, previous, candidates, V, device=E.device))
                        successor = next_token(committed, i, d)
                        if successor is not None:
                            if self.sparse_backend == 'signed_feature_hash':
                                accesses.append(hash_keys(d, candidates, successor, self.dS.capacity, E.device)[0])
                            else:
                                accesses.append(pack_exact_keys(d, candidates, successor, V, device=E.device))
                dirty = torch.unique(torch.cat(accesses), sorted=True) if accesses else torch.empty(0, device=E.device, dtype=torch.long)
                sparse_values = self.dS.read(dirty).detach().clone().requires_grad_(True)
                sparse = (_TrainingSparse(self.offline.sparse, dirty, sparse_values, self.dS.capacity)
                          if self.sparse_backend == 'signed_feature_hash'
                          else _TrainingExactSparse(self.offline.sparse, dirty, sparse_values, V))
            correction = torch.stack([
                two_sided_slot_correction(E, A, U[i], l_unit[i], prefix, committed, i,
                                          sparse, self.offline.M_sparse)
                for i in range(n_eval)
            ])
            if weights is None:
                eff_r = min(first_rejection, n_eval) if first_rejection is not None else None
                weights_t = loss_weights(n_eval, l_unit.device, self.weighting_mode, self.gamma, eff_r)
            else:
                weights_t = weights[:n_eval].detach().clone()
            loss = two_sided_pseudolikelihood_loss(l_unit[:n_eval] - correction, committed[:n_eval], weights_t)
        else:
            if target_logits is None:
                raise ValueError('live_full online update requires target_logits')
            U = block_unary(E, W, b, H)
            sparse = self.offline.sparse
            if self.dS is not None:
                accesses = []
                for i in range(len(H)):
                    dense = slot_correction(E, A, U[i], l_unit[i], prefix, tokens, i)
                    candidates = top_candidates(l_unit[i] - dense, self.offline.M_sparse)
                    for d in range(1, len(A) + 1):
                        previous = prev_token(prefix, tokens, i, d)
                        if previous is not None:
                            if self.sparse_backend == 'signed_feature_hash':
                                accesses.append(hash_keys(d, previous, candidates, self.dS.capacity, E.device)[0])
                            else:
                                accesses.append(pack_exact_keys(d, previous, candidates, V, device=E.device))
                dirty = torch.unique(torch.cat(accesses), sorted=True) if accesses else torch.empty(0, device=E.device, dtype=torch.long)
                sparse_values = self.dS.read(dirty).detach().clone().requires_grad_(True)
                sparse = (_TrainingSparse(self.offline.sparse, dirty, sparse_values, self.dS.capacity)
                          if self.sparse_backend == 'signed_feature_hash'
                          else _TrainingExactSparse(self.offline.sparse, dirty, sparse_values, V))
            correction = torch.stack([slot_correction(E, A, U[i], l_unit[i], prefix, tokens, i,
                                                     sparse, self.offline.M_sparse) for i in range(len(H))])
            if weights is None:
                weights_t = loss_weights(len(l_unit), l_unit.device, self.weighting_mode, self.gamma, first_rejection)
            else:
                weights_t = weights.detach().clone()
            loss = full_kl(l_unit - correction, target_logits, weights_t)
        loss.backward()
        g_dW = None if self.bias_sparse_only else (dW.grad.detach() if dW.grad is not None else torch.zeros_like(dW))
        g_dA = None if self.bias_sparse_only else (dA.grad.detach() if dA.grad is not None else torch.zeros_like(dA))
        g_db = db.grad.detach() if db.grad is not None else torch.zeros_like(db)
        g_sparse = (sparse_values.grad.detach() if sparse_values is not None and sparse_values.grad is not None
                    else (torch.zeros_like(sparse_values) if sparse_values is not None else None))
        return float(loss.detach()), g_dW, g_dA, g_db, dirty, g_sparse

    def _apply_step(self, lr: float, loss_value: float, g_dW, g_dA, g_db, dirty, g_sparse, cloned_rounds: int = 1):
        with torch.no_grad():
            shrink = 1 + lr * self.lambda_l2
            if not self.bias_sparse_only:
                self.dW = (self.dW - lr * g_dW) / shrink
                self.dA = (self.dA - lr * g_dA) / shrink
            self.db = proximal_l1((self.db - lr * g_db) / shrink, lr * self.lambda_l1_b)
            if self.dS is not None:
                self.dS.step(dirty, g_sparse, lr, self.lambda_l1_s)
            self.u += 1
            self.version += 1
            self.last_loss = float(loss_value)
            self._snapshot = self._make_snapshot()
            dense_bytes = sum(x.numel() * x.element_size() for x in (self.dW, self.dA, self.db))
            optimized_bytes = self.db.numel() * self.db.element_size() if self.bias_sparse_only else dense_bytes
            self.state_bytes_estimate['dense_activation_clone'] += 2 * dense_bytes * cloned_rounds
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

    def _update(self, round_idx, l_unit, target_logits, H, prefix, tokens,
                weights=None, first_rejection=None, committed_tokens=None):
        lr_check = float(self.lr(self.u + 1) if callable(self.lr) else self.lr)
        if lr_check < 0:
            raise ValueError('scheduled learning rate must be nonnegative')
        loss_val, g_dW, g_dA, g_db, dirty, g_sparse = self._compute_round_gradients(
            l_unit, target_logits, H, prefix, tokens,
            weights=weights, first_rejection=first_rejection, committed_tokens=committed_tokens,
        )
        if not self.accumulate_stride:
            self._apply_step(lr_check, loss_val, g_dW, g_dA, g_db, dirty, g_sparse, cloned_rounds=1)
            return True
        with torch.no_grad():
            self._buffered_rounds += 1
            self._buf_loss += loss_val
            if not self.bias_sparse_only:
                self._buf_dW = g_dW.clone() if self._buf_dW is None else (self._buf_dW + g_dW)
                self._buf_dA = g_dA.clone() if self._buf_dA is None else (self._buf_dA + g_dA)
            self._buf_db = g_db.clone() if self._buf_db is None else (self._buf_db + g_db)
            if self.dS is not None and dirty is not None and dirty.numel() > 0:
                self._buf_sparse_idx.append(dirty.detach().clone())
                self._buf_sparse_grad.append(g_sparse.detach().clone())
            if round_idx % self.update_stride != 0:
                return False
            denom = float(self._buffered_rounds) if self.buffer_reduction == 'mean' else 1.0
            agg_dW = None if self.bias_sparse_only else (self._buf_dW / denom)
            agg_dA = None if self.bias_sparse_only else (self._buf_dA / denom)
            agg_db = self._buf_db / denom
            if self.dS is not None:
                if self._buf_sparse_idx:
                    agg_dirty = torch.cat(self._buf_sparse_idx)
                    agg_sparse = torch.cat(self._buf_sparse_grad) / denom
                else:
                    agg_dirty = torch.empty(0, dtype=torch.long, device=self.db.device)
                    agg_sparse = torch.empty(0, dtype=torch.float32, device=self.db.device)
            else:
                agg_dirty = agg_sparse = None
            rounds_in_window = self._buffered_rounds
            self.buffered_rounds_total += rounds_in_window
            avg_loss = self._buf_loss / denom
            self._buffered_rounds = 0
            self._buf_loss = 0.0
            self._buf_dW = None
            self._buf_dA = None
            self._buf_db = None
            self._buf_sparse_idx = []
            self._buf_sparse_grad = []
            self._apply_step(lr_check, avg_loss, agg_dW, agg_dA, agg_db, agg_dirty, agg_sparse,
                             cloned_rounds=rounds_in_window)
            return True

    def metrics(self):
        # Evaluate current lazy shrinkage without materializing it into storage.
        effective_dS = self.dS.read(self.dS.active) if self.dS is not None else None
        result = {'adapter_version': self.version, 'u': self.u, 'last_round': self.last_round,
                  'loss_mode': self.loss_mode, 'T_train': 1., 'beta_train': 1.,
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
                  'sparse_snapshot_strategy': 'compact_active_bucket_copy' if self.sparse_backend == 'signed_feature_hash' else 'compact_active_exact_pair_copy',
                  'sparse_update_strategy': 'dirty_lazy_catchup_and_active_list_compaction',
                  'last_full_loss': self.last_loss}
        if self.sparse_backend != 'signed_feature_hash' or self.accumulate_stride or self.loss_mode != 'live_full':
            result['sparse_backend'] = self.sparse_backend
            result['accumulate_stride'] = self.accumulate_stride
            result['buffer_reduction'] = self.buffer_reduction
            result['buffered_pending_rounds'] = self._buffered_rounds
            result['buffered_rounds_accumulated_total'] = self.buffered_rounds_total
            if self.dS is not None and self.sparse_backend == 'exact_observed_pairs':
                result['sparse_overflow_events'] = self.dS.overflow_events
                result['sparse_dropped_unique_keys'] = self.dS.dropped_unique_keys
                result['sparse_peak_active'] = self.dS.peak_active
        return result
