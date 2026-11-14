"""Request-local, synchronous live-full adaptation and normalized bigram baseline."""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Callable
import torch

from .adapter import (ProposalSnapshot, block_predecessor_pairs, block_unary,
                      slot_correction, top_candidates, prev_token, full_kl, loss_weights)
from .optimizers import GramNewtonSchulz
from .sparse import LazySignedHash, SumSparse, hash_keys, proximal_l1


class _BigramRowIndex:
    """Incremental O(1)/O(log k_ret) top-k tracker under monotone +1 count updates."""
    __slots__ = ('k_ret', 'total', 'unique_len', 'retained_sum', 'ids', 'counts_f', 'pos', 'top', 'in_top')

    def __init__(self, k_ret: int | None):
        self.k_ret = k_ret
        self.total = 0
        self.unique_len = 0
        self.retained_sum = 0
        if k_ret is None:
            self.ids: list[int] = []
            self.counts_f: list[float] = []
            self.pos: dict[int, int] = {}
            self.top = None
            self.in_top = None
        else:
            self.ids = None
            self.counts_f = None
            self.pos = None
            self.top: list[tuple[int, int]] = []
            self.in_top: dict[int, int] = {}

    @classmethod
    def from_counter(cls, counts: Counter, k_ret: int | None) -> '_BigramRowIndex':
        row = cls(k_ret)
        row.unique_len = len(counts)
        if not counts:
            return row
        if k_ret is None:
            row.ids = list(counts.keys())
            row.counts_f = [float(v) for v in counts.values()]
            row.pos = {tok: idx for idx, tok in enumerate(row.ids)}
            total = sum(counts.values())
            row.total = total
            row.retained_sum = total
            return row
        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:k_ret]
        row.top = [(-cnt, tok) for tok, cnt in ordered]
        row.in_top = {tok: cnt for tok, cnt in ordered}
        row.total = sum(counts.values())
        row.retained_sum = sum(cnt for _, cnt in ordered)
        return row

    def increment(self, token: int, new_count: int) -> None:
        self.total += 1
        if new_count == 1:
            self.unique_len += 1
        if self.k_ret is None:
            self.retained_sum += 1
            idx = self.pos.get(token)
            if idx is None:
                self.pos[token] = len(self.ids)
                self.ids.append(token)
                self.counts_f.append(float(new_count))
            else:
                self.counts_f[idx] = float(new_count)
            return
        old_count = new_count - 1
        if token in self.in_top:
            pos = bisect_left(self.top, (-old_count, token))
            self.top.pop(pos)
            insort(self.top, (-new_count, token))
            self.in_top[token] = new_count
            self.retained_sum += 1
            return
        cand = (-new_count, token)
        if len(self.top) < self.k_ret:
            insort(self.top, cand)
            self.in_top[token] = new_count
            self.retained_sum += new_count
        elif cand < self.top[-1]:
            neg_ev, ev_tok = self.top.pop()
            del self.in_top[ev_tok]
            self.retained_sum -= (-neg_ev)
            insort(self.top, cand)
            self.in_top[token] = new_count
            self.retained_sum += new_count


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
        self._rows: dict[int, _BigramRowIndex] = {}
        self._cache: dict[int, dict] = {}
        self._ids_bufs: dict[torch.device, torch.Tensor] = {}
        self._counts_bufs: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _borrow_buffers(self, length: int, device: torch.device, dtype: torch.dtype):
        ids_buf = self._ids_bufs.get(device)
        if ids_buf is None or ids_buf.numel() < length:
            cap = max(length, 64 if ids_buf is None else ids_buf.numel() * 2)
            ids_buf = torch.empty(cap, device=device, dtype=torch.long)
            self._ids_bufs[device] = ids_buf
        c_key = (device, dtype)
        c_buf = self._counts_bufs.get(c_key)
        if c_buf is None or c_buf.numel() < length:
            cap = max(length, 64 if c_buf is None else c_buf.numel() * 2)
            c_buf = torch.empty(cap, device=device, dtype=dtype)
            self._counts_bufs[c_key] = c_buf
        return ids_buf[:length], c_buf[:length]

    def update(self, prefix_tail, committed_tokens):
        previous = int(prefix_tail[-1]) if len(prefix_tail) else None
        for token in committed_tokens:
            token = int(token)
            if not 0 <= token < self.vocab_size:
                raise ValueError('committed token outside vocabulary')
            if previous is not None:
                if not 0 <= previous < self.vocab_size:
                    raise ValueError('previous token outside vocabulary')
                row_counts = self.counts[previous]
                new_count = row_counts[token] + 1
                row_counts[token] = new_count
                row_idx = self._rows.get(previous)
                if row_idx is None or row_idx.k_ret != self.k_ret or row_idx.unique_len + (1 if new_count == 1 else 0) != len(row_counts):
                    self._rows[previous] = _BigramRowIndex.from_counter(row_counts, self.k_ret)
                else:
                    row_idx.increment(token, new_count)
                self._cache.pop(previous, None)
            previous = token
        if len(committed_tokens):
            self.version += 1

    def _row_for(self, prev_int: int, counts: Counter) -> _BigramRowIndex:
        row = self._rows.get(prev_int)
        if row is None or row.k_ret != self.k_ret or row.unique_len != len(counts):
            row = _BigramRowIndex.from_counter(counts, self.k_ret)
            self._rows[prev_int] = row
        return row

    def _prior_parts(self, prev_int: int, counts: Counter, device, dtype):
        target_device = torch.device('cpu') if device is None else torch.device(device)
        row = self._row_for(prev_int, counts)
        cache_key = (self.alpha, self.k_ret, target_device, dtype)
        bucket = self._cache.get(prev_int)
        if bucket is not None:
            cached = bucket.get(cache_key)
            if cached is not None and cached[0] == row.total and cached[1] == row.unique_len:
                return cached[2], cached[3], cached[4]
        else:
            bucket = {}
            self._cache[prev_int] = bucket
        total = row.total
        if row.k_ret is None:
            ids_list = row.ids
            counts_list = row.counts_f
        else:
            ids_list = [tok for _, tok in row.top]
            counts_list = [float(-neg_c) for neg_c, _ in row.top]
        ret_len = len(ids_list)
        evicted = total - row.retained_sum
        absent = self.vocab_size - ret_len
        denom = float(total) + float(self.alpha) * float(self.vocab_size)
        base_val = (float(evicted) / float(absent)) if absent else 0.0
        base_lp = math.log((base_val + float(self.alpha)) / denom)
        if ret_len:
            cpu_ids = torch.tensor(ids_list, dtype=torch.long)
            cpu_counts = torch.tensor(counts_list, dtype=dtype)
            if target_device.type == 'cpu':
                ids_t = cpu_ids
                ret_lp = torch.log((cpu_counts + self.alpha) / denom)
            else:
                ids_slice, counts_slice = self._borrow_buffers(ret_len, target_device, dtype)
                ids_slice.copy_(cpu_ids)
                counts_slice.copy_(cpu_counts)
                ids_t = ids_slice.clone()
                ret_lp = torch.log((counts_slice + self.alpha) / denom)
        else:
            ids_t = torch.empty(0, device=target_device, dtype=torch.long)
            ret_lp = torch.empty(0, device=target_device, dtype=dtype)
        bucket[cache_key] = (total, row.unique_len, base_lp, ids_t, ret_lp)
        return base_lp, ids_t, ret_lp

    def prior(self, previous: int | None, device=None, dtype=torch.float32):
        if previous is None:
            return torch.zeros(self.vocab_size, device=device, dtype=dtype)
        prev_int = int(previous)
        counts = self.counts.get(prev_int)
        if not counts:
            return torch.full((self.vocab_size,), -math.log(float(self.vocab_size)), device=device, dtype=dtype)
        base_lp, ids_t, ret_lp = self._prior_parts(prev_int, counts, device, dtype)
        out = torch.full((self.vocab_size,), base_lp, device=device, dtype=dtype)
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
        out = l_unit + (self.lambda_tr * base_lp)
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
                 fastpath=None, optimizer: str = 'sgd',
                 betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8):
        if update_stride < 1 or min(lambda_l2, lambda_l1_b, lambda_l1_s) < 0:
            raise ValueError('invalid online stride or regularization')
        if not callable(lr) and lr < 0:
            raise ValueError('learning rate must be nonnegative')
        if optimizer not in ('sgd', 'atan2', 'rms', 'muon'):
            raise ValueError('online optimizer must be one of sgd, atan2, rms, muon')
        if not (0. <= betas[0] < 1. and 0. <= betas[1] < 1.) or eps <= 0:
            raise ValueError('invalid online optimizer betas or eps')
        self.offline = offline.snapshot() if hasattr(offline, 'snapshot') else offline
        self.lr, self.update_stride = lr, update_stride
        self.lambda_l2, self.lambda_l1_b, self.lambda_l1_s = lambda_l2, lambda_l1_b, lambda_l1_s
        if weighting_mode not in ('exponential', 'rejection_relative', 'cutoff') or gamma <= 0:
            raise ValueError('invalid online weighting mode or gamma')
        self.weighting_mode, self.gamma = weighting_mode, gamma
        self.bias_sparse_only = bias_sparse_only
        self.fastpath = fastpath
        self.optimizer = optimizer
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self._opt_state: dict[str, torch.Tensor] = {}
        self._orthogonalizer = GramNewtonSchulz() if optimizer == 'muon' else None
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

    def _make_snapshot(self, *, precomputed_dense: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        terms = []
        if self.offline.sparse is not None:
            terms.append(self.offline.sparse)
        if self.dS is not None:
            terms.append(self.dS.snapshot())
        sparse = SumSparse(tuple(terms)) if terms else None
        if precomputed_dense is not None:
            W, A, b = precomputed_dense
        else:
            W = (self.offline.W + self.dW).detach()
            A = (self.offline.A + self.dA).detach()
            b = (self.offline.b + self.db).detach()
        if not self.offline.is_zero:
            is_zero = False
        elif self.u == 0:
            is_zero = self.dS is None or len(self.dS.active) == 0
        elif self.dS is not None and len(self.dS.active) > 0:
            is_zero = False
        elif self.bias_sparse_only:
            is_zero = not bool(self.db.any())
        else:
            is_zero = not bool(
                torch.cat((self.db.reshape(-1), self.dW.reshape(-1), self.dA.reshape(-1))).any()
            )
        return ProposalSnapshot(self.offline.E, W, A, b, self.version, sparse, self.offline.M_sparse, None, is_zero,
                                _E_bf16=self.offline._E_bf16)

    def snapshot(self, C0=None):
        return self._snapshot if C0 is None else self._snapshot.with_candidates(C0)

    def _adaptive_dense_step(self, dW: torch.Tensor, gW: torch.Tensor | None,
                             dA: torch.Tensor, gA: torch.Tensor | None,
                             db: torch.Tensor, gb: torch.Tensor, lr: float, step: int) -> int:
        shrink = 1.0 + lr * self.lambda_l2
        beta1, beta2 = self.betas
        bias1 = max(1.0 - beta1 ** step, 1e-16)
        bias2 = max(1.0 - beta2 ** step, 1e-16)
        extra_state_bytes = 0

        def _precond_dir(name: str, grad: torch.Tensor) -> torch.Tensor:
            nonlocal extra_state_bytes
            elem_bytes = grad.numel() * grad.element_size()
            if self.optimizer == 'rms':
                v = self._opt_state.get(f'{name}_v')
                if v is None:
                    v = torch.zeros_like(grad)
                    self._opt_state[f'{name}_v'] = v
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                extra_state_bytes += 2 * elem_bytes
                return grad / (v / bias2).sqrt().add_(self.eps)
            if self.optimizer == 'muon' and grad.ndim >= 2:
                buf = self._opt_state.get(f'{name}_mom')
                if buf is None:
                    buf = torch.zeros_like(grad)
                    self._opt_state[f'{name}_mom'] = buf
                buf.mul_(beta1).add_(grad)
                extra_state_bytes += 2 * elem_bytes
                nesterov = grad.add(buf, alpha=beta1)
                ortho = self._orthogonalizer(nesterov)
                scale = 0.2 * math.sqrt(max(int(ortho.shape[-2]), int(ortho.shape[-1])))
                return (ortho * scale).to(grad.dtype)
            m = self._opt_state.get(f'{name}_m')
            v = self._opt_state.get(f'{name}_v')
            if m is None or v is None:
                m = torch.zeros_like(grad)
                v = torch.zeros_like(grad)
                self._opt_state[f'{name}_m'] = m
                self._opt_state[f'{name}_v'] = v
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            extra_state_bytes += 4 * elem_bytes
            return torch.atan2(m / bias1, (v / bias2).sqrt().add_(self.eps))

        if not self.bias_sparse_only:
            gW_eff = gW if gW is not None else torch.zeros_like(dW)
            gA_eff = gA if gA is not None else torch.zeros_like(dA)
            self.dW = (dW - lr * _precond_dir('dW', gW_eff)) / shrink
            self.dA = (dA - lr * _precond_dir('dA', gA_eff)) / shrink
        # Full-vocabulary bias `db` has simplex-bounded gradients `sum_i w_i (q_i - p_i)`
        # with ~1/V tail noise on unvisited tokens and explicit proximal-L1 shrinkage;
        # use bias-corrected first-moment EMA rather than per-coordinate 1/|gb| scaling
        # so inactive vocabulary coordinates are not amplified and L1 sparsity holds.
        mb = self._opt_state.get('db_m')
        if mb is None:
            mb = torch.zeros_like(gb)
            self._opt_state['db_m'] = mb
        mb.mul_(beta1).add_(gb, alpha=1.0 - beta1)
        extra_state_bytes += 2 * gb.numel() * gb.element_size()
        step_b = mb / bias1
        self.db = proximal_l1((db - lr * step_b) / shrink, lr * self.lambda_l1_b)
        return extra_state_bytes

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

        def _prep_in(x: torch.Tensor, *, as_float: bool = True) -> torch.Tensor:
            t = x.detach()
            if as_float and t.dtype != torch.float32:
                t = t.float()
            return t.clone() if t.is_inference() else t

        l_unit = _prep_in(l_unit)
        target_logits = _prep_in(target_logits)
        H = _prep_in(H)
        if weights is not None:
            weights = _prep_in(weights, as_float=False)
        if self.bias_sparse_only:
            dW = self.dW.detach()
            dA = self.dA.detach()
        else:
            dW = self.dW.detach().clone().requires_grad_(True)
            dA = self.dA.detach().clone().requires_grad_(True)
        db = self.db.detach().clone().requires_grad_(True)
        W = self.offline.W + dW if not self.bias_sparse_only else self._snapshot.W
        A = self.offline.A + dA if not self.bias_sparse_only else self._snapshot.A
        b = self.offline.b + db
        E = self.offline.E
        K = len(H)
        bf16_gemm = bool(self.fastpath is not None and self.fastpath.bf16_gemm)
        E_bf16 = self.offline._token_features_bf16() if bf16_gemm else None
        U = block_unary(E, W, b, H, bf16_gemm=bf16_gemm, E_bf16=E_bf16)
        valid_slots, valid_d0, valid_prev = block_predecessor_pairs(prefix, tokens, K, len(A), E.device)
        if valid_slots.numel():
            if bf16_gemm:
                assert E_bf16 is not None
                E_prev = E_bf16.index_select(0, valid_prev)
                msgs = torch.bmm(A.to(torch.bfloat16).index_select(0, valid_d0), E_prev.unsqueeze(-1)).squeeze(-1).to(A.dtype)
                slot_msgs = torch.zeros(K, E.shape[1], dtype=A.dtype, device=A.device).index_add(0, valid_slots, msgs)
                dense = U + (slot_msgs.to(torch.bfloat16) @ E_bf16.T).to(U.dtype)
            else:
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
            precomputed_dense = None
            extra_opt_bytes = 0
            if self.optimizer == 'sgd':
                if self.fastpath is not None:
                    self.dW, self.dA, self.db, snap_W, snap_A, snap_b = self.fastpath.online_dense_step(
                        dW, dW.grad, dA, dA.grad, db, db.grad,
                        self.offline.W, self.offline.A, self.offline.b,
                        lr=lr, lambda_l2=self.lambda_l2, lambda_l1_b=self.lambda_l1_b,
                        bias_sparse_only=self.bias_sparse_only,
                    )
                    precomputed_dense = (snap_W, snap_A, snap_b)
                else:
                    shrink = 1 + lr * self.lambda_l2
                    if not self.bias_sparse_only:
                        self.dW = (dW - lr * (dW.grad if dW.grad is not None else torch.zeros_like(dW))) / shrink
                        self.dA = (dA - lr * (dA.grad if dA.grad is not None else torch.zeros_like(dA))) / shrink
                    self.db = proximal_l1((db - lr * db.grad) / shrink, lr * self.lambda_l1_b)
            else:
                extra_opt_bytes = self._adaptive_dense_step(dW, dW.grad, dA, dA.grad, db, db.grad, lr, self.u + 1)
            if self.dS is not None:
                if sparse_values is not None:
                    gradient = sparse_values.grad if sparse_values.grad is not None else torch.zeros_like(sparse_values)
                else:
                    gradient = torch.empty(0, device=E.device, dtype=torch.float32)
                self.dS.step_unique(dirty, gradient, lr, self.lambda_l1_s)
            self.u += 1
            self.version += 1
            self._last_loss_tensor = loss.detach()
            self._snapshot = self._make_snapshot(precomputed_dense=precomputed_dense)
            dense_bytes = sum(x.numel() * x.element_size() for x in (self.dW, self.dA, self.db))
            db_bytes = self.db.numel() * self.db.element_size()
            optimized_bytes = db_bytes if self.bias_sparse_only else dense_bytes
            self.state_bytes_estimate['dense_activation_clone'] += 2 * dense_bytes
            # Conceptual reads/writes, not a profiler-derived hardware traffic count.
            self.state_bytes_estimate['dense_optimizer'] += 5 * optimized_bytes + 2 * db_bytes + extra_opt_bytes
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
        out = {'adapter_version': self.version, 'u': self.u, 'last_round': self.last_round,
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
        if self.optimizer != 'sgd':
            out['dense_optimizer'] = self.optimizer
        return out
