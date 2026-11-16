"""Request-local, synchronous live-full adaptation and normalized bigram baseline."""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Callable
import torch

from .adapter import (ProposalSnapshot, analytic_full_kl_and_correction_grad,
                      block_predecessor_pairs, block_unary,
                      slot_correction, two_sided_slot_correction,
                      top_candidates, prev_token, next_token, full_kl,
                      two_sided_pseudolikelihood_loss, loss_weights)
from .optimizers import GramNewtonSchulz
from .sparse import (LazySignedHash, LazyExactObservedPairs, SumSparse,
                     _bucket_from_raw, _hash_raw_and_sign,
                     hash_keys, pack_exact_keys, proximal_l1)


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


class _DeviceBigramState:
    """Device-resident compact transition table for zero-sync `OnlineBigram` scoring."""
    __slots__ = (
        'device', 'dtype', 'hyper_key', 'row_slot', 'base_shift', 'ret_ids', 'ret_shift',
        'ret_valid', 'prev_to_slot', 'first_prev', 'known_dict_len', 'active_rows',
        'max_ret', 'row_cap', 'col_cap', 'dirty_rows',
    )

    def __init__(self, vocab_size: int, device: torch.device, dtype: torch.dtype,
                 hyper_key: tuple[float, int | None, float, float | None], row_cap: int = 64, col_cap: int = 16):
        self.device = device
        self.dtype = dtype
        self.hyper_key = hyper_key
        self.row_cap = max(2, row_cap)
        self.col_cap = max(1, col_cap)
        self.row_slot = torch.zeros(vocab_size, device=device, dtype=torch.long)
        # Row 0 is the inactive sentinel; IEEE-754 `-0.0` is the exact additive identity
        # (`x + (-0.0) == x` bit-for-bit, preserving even `-0.0` and `+/-inf`).
        # On CUDA, PyTorch `bf16/fp16 + python_float` promotes to `opmath_t = float32`
        # before rounding back to `dtype`, whereas CPU casts the scalar to `dtype` first.
        base_dt = (torch.float64 if dtype == torch.float64 else torch.float32) if device.type == 'cuda' else dtype
        self.base_shift = torch.full((self.row_cap,), -0.0, device=device, dtype=base_dt)
        self.ret_ids = torch.zeros((self.row_cap, self.col_cap), device=device, dtype=torch.long)
        self.ret_shift = torch.full((self.row_cap, self.col_cap), -0.0, device=device, dtype=dtype)
        self.ret_valid = torch.zeros((self.row_cap, self.col_cap), device=device, dtype=torch.bool)
        self.prev_to_slot: dict[int, int] = {}
        self.first_prev: int | None = None
        self.known_dict_len = 0
        self.active_rows = 0
        self.max_ret = 0
        self.dirty_rows: set[int] = set()

    def ensure_capacity(self, needed_rows: int, needed_cols: int) -> None:
        if needed_rows <= self.row_cap and needed_cols <= self.col_cap:
            return
        new_row_cap = max(needed_rows, self.row_cap * 2 if needed_rows > self.row_cap else self.row_cap)
        new_col_cap = max(needed_cols, self.col_cap * 2 if needed_cols > self.col_cap else self.col_cap)
        if new_row_cap != self.row_cap:
            new_base = torch.full((new_row_cap,), -0.0, device=self.device, dtype=self.base_shift.dtype)
            new_base[:self.active_rows + 1].copy_(self.base_shift[:self.active_rows + 1])
            self.base_shift = new_base
        new_ids = torch.zeros((new_row_cap, new_col_cap), device=self.device, dtype=torch.long)
        new_shift = torch.full((new_row_cap, new_col_cap), -0.0, device=self.device, dtype=self.dtype)
        new_valid = torch.zeros((new_row_cap, new_col_cap), device=self.device, dtype=torch.bool)
        if self.active_rows > 0:
            active_end = self.active_rows + 1
            new_ids[1:active_end, :self.col_cap].copy_(self.ret_ids[1:active_end, :self.col_cap])
            new_shift[1:active_end, :self.col_cap].copy_(self.ret_shift[1:active_end, :self.col_cap])
            new_valid[1:active_end, :self.col_cap].copy_(self.ret_valid[1:active_end, :self.col_cap])
            if new_col_cap > self.col_cap:
                # Pad newly grown columns of already-synced active rows with their column-0
                # `(ids[0], ret_shift[0])` entry so non-dirty `score_tensor` rows never scatter
                # into token 0, while `new_valid` stays `False` on padded columns.
                new_ids[1:active_end, self.col_cap:new_col_cap] = new_ids[1:active_end, 0:1]
                new_shift[1:active_end, self.col_cap:new_col_cap] = new_shift[1:active_end, 0:1]
        self.ret_ids = new_ids
        self.ret_shift = new_shift
        self.ret_valid = new_valid
        self.row_cap = new_row_cap
        self.col_cap = new_col_cap


class OnlineBigram:
    """Exact successor counts; optional top-k retained-prior memory ablation.

    Full counts are kept as the authoritative count ledger in the eviction
    diagnostic, so a reappearing retained ID never receives an approximate count.
    Memory reports include that ledger; this ablation does not claim bounded
    memory savings over exact counts.

    Contract: mutate the table only through `update()`. Direct edits of `counts`
    (or its inner `Counter`s) bypass dirty tracking for the device-resident
    tables used by `score_tensor`/`merge_compact_candidates`; that is unsupported.
    Removing whole keys, or clearing `counts`, is detected and triggers a rebuild,
    as a convenience.
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
        self._rows: dict[int, _BigramRowIndex] = {}
        self._cache: dict[int, dict] = {}
        self._ids_bufs: dict[torch.device, torch.Tensor] = {}
        self._counts_bufs: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._dev_tables: dict[tuple[torch.device, torch.dtype], _DeviceBigramState] = {}

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
                for st in self._dev_tables.values():
                    st.dirty_rows.add(previous)
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

    def _validated_kappa(self) -> float | None:
        if self.count_gate_kappa is None:
            return None
        if (isinstance(self.count_gate_kappa, bool)
                or not math.isfinite(float(self.count_gate_kappa))
                or float(self.count_gate_kappa) <= 0):
            raise ValueError('count_gate_kappa must be a finite positive float when set')
        return float(self.count_gate_kappa)

    def effective_lambda(self, previous: int | None) -> float:
        kappa = self._validated_kappa()
        if previous is None or self.lambda_tr == 0:
            return 0.0
        counts = self.counts.get(int(previous))
        if not counts:
            return 0.0
        if kappa is None:
            return float(self.lambda_tr)
        n_prev = sum(counts.values())
        return float(self.lambda_tr) * (n_prev / (n_prev + kappa))

    def score(self, l_unit, previous: int | None):
        kappa = self._validated_kappa()
        if previous is None or self.lambda_tr == 0:
            return l_unit
        prev_int = int(previous)
        counts = self.counts.get(prev_int)
        if not counts:
            return l_unit
        base_lp, ids_t, ret_lp = self._prior_parts(prev_int, counts, l_unit.device, l_unit.dtype)
        if kappa is None:
            lam_row = float(self.lambda_tr)
        else:
            total = self._row_for(prev_int, counts).total
            lam_row = float(self.lambda_tr) * (total / (total + kappa))
        if lam_row == 0.0:
            return l_unit
        out = l_unit + (lam_row * base_lp)
        if ids_t.numel():
            out[ids_t] = l_unit.index_select(0, ids_t) + lam_row * ret_lp
        return out

    def _sync_device_table(self, device: torch.device, dtype: torch.dtype) -> _DeviceBigramState:
        key = (device, dtype)
        kappa = self._validated_kappa()
        hyper_key = (float(self.alpha), self.k_ret, float(self.lambda_tr), kappa)
        state = self._dev_tables.get(key)
        need_rebuild = (
            state is None
            or state.hyper_key != hyper_key
            or state.active_rows > len(self.counts)
            or (state.first_prev is not None and not self.counts.get(state.first_prev))
        )
        if not need_rebuild and (
            state.dirty_rows
            or len(self.counts) != state.known_dict_len
            or state.active_rows <= 64
        ):
            if any(not self.counts.get(p) for p in state.prev_to_slot):
                need_rebuild = True
        if need_rebuild:
            for stale_p in [p for p in self._rows if not self.counts.get(p)]:
                self._rows.pop(stale_p, None)
                self._cache.pop(stale_p, None)
            col_init = min(max(self.k_ret, 1), 64) if self.k_ret is not None else 16
            state = _DeviceBigramState(self.vocab_size, device, dtype, hyper_key, col_cap=col_init)
            state.dirty_rows.update(p for p, c in self.counts.items() if c)
            self._dev_tables[key] = state
        if not state.dirty_rows:
            state.known_dict_len = len(self.counts)
            return state
        dirty_list = [p for p in state.dirty_rows if self.counts.get(p)]
        state.dirty_rows.clear()
        if not dirty_list:
            state.known_dict_len = len(self.counts)
            return state

        prepared = []
        new_active = state.active_rows
        needed_max_ret = state.max_ret
        new_prev_ids: list[int] = []
        new_prev_slots: list[int] = []
        for prev_int in dirty_list:
            slot = state.prev_to_slot.get(prev_int)
            if slot is None:
                new_active += 1
                slot = new_active
                state.prev_to_slot[prev_int] = slot
                if state.first_prev is None:
                    state.first_prev = prev_int
                new_prev_ids.append(prev_int)
                new_prev_slots.append(slot)
            row = self._row_for(prev_int, self.counts[prev_int])
            total = row.total
            if row.k_ret is None:
                ids_list = row.ids
                counts_list = row.counts_f
            else:
                ids_list = [tok for _, tok in row.top]
                counts_list = [float(-neg_c) for neg_c, _ in row.top]
            ret_len = len(ids_list)
            if ret_len > needed_max_ret:
                needed_max_ret = ret_len
            evicted = total - row.retained_sum
            absent = self.vocab_size - ret_len
            denom = float(total) + float(self.alpha) * float(self.vocab_size)
            base_val = (float(evicted) / float(absent)) if absent else 0.0
            base_lp = math.log((base_val + float(self.alpha)) / denom)
            prepared.append((slot, total, base_lp, ids_list, counts_list, denom))

        state.ensure_capacity(new_active + 1, max(1, needed_max_ret))
        state.active_rows = new_active
        state.max_ret = needed_max_ret
        width = state.col_cap
        n_dirty = len(prepared)
        slots_cpu = torch.empty(n_dirty, dtype=torch.long)
        base_cpu = torch.empty(n_dirty, dtype=state.base_shift.dtype)
        ids_cpu = torch.empty((n_dirty, width), dtype=torch.long)
        shift_dev_rows = torch.empty((n_dirty, width), device=device, dtype=dtype)
        valid_cpu = torch.zeros((n_dirty, width), dtype=torch.bool)
        lam = float(self.lambda_tr)
        alpha_f = float(self.alpha)
        for idx, (slot, total, base_lp, ids_list, counts_list, denom) in enumerate(prepared):
            lam_row = lam if kappa is None else lam * (total / (total + kappa))
            slots_cpu[idx] = slot
            base_cpu[idx] = lam_row * base_lp
            rlen = len(ids_list)
            ids_t = torch.tensor(ids_list, dtype=torch.long)
            cnt_t = torch.tensor(counts_list, dtype=dtype)
            if rlen > 1:
                ids_t, id_order = ids_t.sort()
                cnt_t = cnt_t.index_select(0, id_order)
            cnt_dev = cnt_t.to(device=device)
            ret_val = lam_row * torch.log((cnt_dev + alpha_f) / denom)
            ids_cpu[idx, :rlen] = ids_t
            shift_dev_rows[idx, :rlen] = ret_val
            valid_cpu[idx, :rlen] = True
            if rlen < width:
                ids_cpu[idx, rlen:] = ids_t[0]
                shift_dev_rows[idx, rlen:] = ret_val[0]

        if new_prev_ids:
            prev_t = torch.tensor(new_prev_ids, dtype=torch.long, device=device)
            pslot_t = torch.tensor(new_prev_slots, dtype=torch.long, device=device)
            state.row_slot.index_copy_(0, prev_t, pslot_t)
        slots_dev = slots_cpu.to(device=device)
        state.base_shift.index_copy_(0, slots_dev, base_cpu.to(device=device))
        state.ret_ids.index_copy_(0, slots_dev, ids_cpu.to(device=device))
        state.ret_shift.index_copy_(0, slots_dev, shift_dev_rows)
        state.ret_valid.index_copy_(0, slots_dev, valid_cpu.to(device=device))
        state.known_dict_len = len(self.counts)
        return state

    def score_tensor(self, l_unit: torch.Tensor, prev_token: torch.Tensor | None) -> torch.Tensor:
        """Zero-sync, 0-ULP device-resident bigram scoring matching `score(l_unit, int(prev_token))`."""
        if prev_token is None or self.lambda_tr == 0 or not self.counts:
            if not self.counts and self._dev_tables:
                self._dev_tables.clear()
            return l_unit
        state = self._sync_device_table(l_unit.device, l_unit.dtype)
        if state.active_rows == 0 or state.max_ret == 0:
            return l_unit
        prev_1d = (
            prev_token.reshape(1)
            if isinstance(prev_token, torch.Tensor)
            else torch.empty(1, dtype=torch.long, device=l_unit.device).fill_(int(prev_token))
        )
        r_1d = state.row_slot.gather(0, prev_1d)
        base_s = state.base_shift.gather(0, r_1d)[0]
        if l_unit.device.type == 'cuda' and l_unit.dtype in (torch.bfloat16, torch.float16):
            shifted = (l_unit.float() + base_s).to(l_unit.dtype)
        else:
            shifted = (l_unit + base_s).to(l_unit.dtype)
        ids_row = state.ret_ids.index_select(0, r_1d)[0, :state.max_ret]
        vals = l_unit.gather(0, ids_row) + state.ret_shift.index_select(0, r_1d)[0, :state.max_ret]
        shifted.scatter_(0, ids_row, vals)
        return shifted

    def merge_compact_candidates(
        self,
        l_unit_i: torch.Tensor,
        base_vals_i: torch.Tensor,
        base_ids_i: torch.Tensor,
        prev_token: torch.Tensor | None,
        *,
        temperature: float,
        k_eff: int,
        state: _DeviceBigramState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-sync, zero-`[V]`-allocation `[k_eff] U [R]` candidate merge (`lambda_tr >= 0`).

        When `temperature == 0` (`k_eff == 1`), returns `(win_score[1], win_id[1])` with
        exact lowest-ID tie-breaking. When `temperature > 0`, returns
        `(scaled_top_scores[k_eff], top_ids[k_eff])`.

        Exactness scope: bit-exact with full-`V` `score` + `probabilities` when
        `N_total == 0` or `lambda_tr == 0` (preserves the batched plain-DFlash `topk`
        order via `stable=True` sort, `prompt.md` section 9) and at `temperature == 0`
        (lowest-ID argmax). For `N_total > 0` it is bit-exact except on exactly tied
        scores straddling a `top_k`/`top_p` truncation boundary, where the reference
        order is implementation-defined (CPU/CUDA `topk` tie order); this path orders
        ties by `base_ids_i` followed by ascending retained IDs. Ratio verification
        stays exact because `verify_block` uses this same proposal `q`.
        """
        if self.lambda_tr < 0:
            raise ValueError("merge_compact_candidates requires nonnegative lambda_tr")
        if self.count_gate_kappa is not None and self.count_gate_kappa <= 0:
            raise ValueError("count_gate_kappa must be strictly positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if k_eff < 1 or k_eff > self.vocab_size:
            raise ValueError("k_eff must satisfy 1 <= k_eff <= vocab_size")
        prob_dtype = l_unit_i.dtype if l_unit_i.dtype == torch.float64 else torch.float32
        if not self.counts and self._dev_tables:
            self._dev_tables.clear()
        if prev_token is None or self.lambda_tr == 0 or not self.counts:
            if temperature == 0:
                return base_vals_i, base_ids_i
            return base_vals_i.to(prob_dtype) / temperature, base_ids_i
        if state is None:
            state = self._sync_device_table(l_unit_i.device, l_unit_i.dtype)
        if state.active_rows == 0 or state.max_ret == 0:
            if temperature == 0:
                return base_vals_i, base_ids_i
            return base_vals_i.to(prob_dtype) / temperature, base_ids_i

        prev_1d = (
            prev_token.reshape(1)
            if isinstance(prev_token, torch.Tensor)
            else torch.empty(1, dtype=torch.long, device=l_unit_i.device).fill_(int(prev_token))
        )
        r_1d = state.row_slot.gather(0, prev_1d)
        base_s = state.base_shift.gather(0, r_1d)[0]
        m_ret = state.max_ret
        r_ids = state.ret_ids.index_select(0, r_1d)[0, :m_ret]
        r_val = state.ret_valid.index_select(0, r_1d)[0, :m_ret]
        r_scores = l_unit_i.gather(0, r_ids) + state.ret_shift.index_select(0, r_1d)[0, :m_ret]
        if base_vals_i.device.type == 'cuda' and base_vals_i.dtype in (torch.bfloat16, torch.float16):
            b_shifted = (base_vals_i.float() + base_s).to(base_vals_i.dtype)
        else:
            b_shifted = (base_vals_i + base_s).to(base_vals_i.dtype)

        overlap = (base_ids_i[:, None] == r_ids[None, :]) & r_val[None, :]
        has_ov = overlap.any(dim=1)
        ov_col = overlap.int().argmax(dim=1)
        b_updated = torch.where(has_ov, r_scores.gather(0, ov_col), b_shifted)
        valid_r = r_val & (~overlap.any(dim=0))

        all_ids = torch.cat((base_ids_i, r_ids), dim=0)
        all_valid = torch.cat((torch.ones_like(has_ov), valid_r), dim=0)

        if temperature == 0:
            all_scores = torch.cat((b_updated, r_scores), dim=0)
            id_key = torch.where(all_valid, all_ids, torch.full_like(all_ids, self.vocab_size))
            _, id_perm = id_key.sort(dim=0, stable=True)
            ids_asc = all_ids.gather(0, id_perm)
            scores_asc = all_scores.gather(0, id_perm)
            valid_asc = all_valid.gather(0, id_perm)
            fmax = torch.finfo(scores_asc.dtype).max
            arg_key = torch.where(valid_asc, scores_asc.clamp_min(-fmax), -torch.inf)
            win_pos = torch.argmax(arg_key, dim=0, keepdim=True)
            return scores_asc.gather(0, win_pos), ids_asc.gather(0, win_pos)

        all_scaled = torch.cat((b_updated, r_scores), dim=0).to(prob_dtype) / temperature
        fmax = torch.finfo(all_scaled.dtype).max
        sel_key = torch.where(all_valid, all_scaled.clamp_min(-fmax), -torch.inf)
        _, sel_pos = torch.sort(sel_key, dim=0, descending=True, stable=True)
        top_pos = sel_pos[:k_eff]
        return all_scaled.gather(0, top_pos), all_ids.gather(0, top_pos)

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
                 fastpath=None, optimizer: str = 'sgd',
                 betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8,
                 *, sparse_backend: str = 'signed_feature_hash',
                 sparse_overflow_policy: str = 'freeze_new',
                 accumulate_stride: bool = False,
                 buffer_reduction: str = 'mean',
                 loss_mode: str = 'live_full',
                 kl_loss_fn: Callable | None = None,
                 analytic_online_grad: bool | None = None):
        if update_stride < 1 or min(lambda_l2, lambda_l1_b, lambda_l1_s) < 0:
            raise ValueError('invalid online stride or regularization')
        if not callable(lr) and lr < 0:
            raise ValueError('learning rate must be nonnegative')
        if optimizer not in ('sgd', 'atan2', 'rms', 'muon'):
            raise ValueError('online optimizer must be one of sgd, atan2, rms, muon')
        if not (0. <= betas[0] < 1. and 0. <= betas[1] < 1.) or eps <= 0:
            raise ValueError('invalid online optimizer betas or eps')
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
        self.fastpath = fastpath
        if analytic_online_grad is not None:
            self.analytic_online_grad = bool(analytic_online_grad)
        elif fastpath is not None:
            fp_cfg = getattr(fastpath, 'config', fastpath)
            self.analytic_online_grad = bool(getattr(fp_cfg, 'analytic_online_grad', True))
        else:
            self.analytic_online_grad = True
        self.optimizer = optimizer
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self._opt_state: dict[str, torch.Tensor] = {}
        self._orthogonalizer = GramNewtonSchulz() if optimizer == 'muon' else None
        self.sparse_backend = sparse_backend
        self.sparse_overflow_policy = sparse_overflow_policy
        self.accumulate_stride = bool(accumulate_stride)
        self.buffer_reduction = buffer_reduction
        self.loss_mode = loss_mode
        self.kl_loss_fn = kl_loss_fn
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
        self._last_loss_tensor: torch.Tensor | float | None = None
        self._buffered_rounds = 0
        self.buffered_rounds_total = 0
        self._buf_dW: torch.Tensor | None = None
        self._buf_dA: torch.Tensor | None = None
        self._buf_db: torch.Tensor | None = None
        self._buf_sparse_idx: list[torch.Tensor] = []
        self._buf_sparse_grad: list[torch.Tensor] = []
        self._buf_loss: torch.Tensor | float = 0.0
        self._buf_had_pairs = False
        self._last_grad_had_pairs = False
        self._offline_is_zero = bool(self.offline.is_zero)
        self._offline_pair_zero = bool(self.offline.pair_is_zero())
        self._any_nonzero_lr_step = False
        self._dA_ever_updated = False
        self._state_bytes_estimate = {'dense_activation_clone': 0, 'dense_optimizer': 0, 'dense_snapshot': 0,
                                      'sparse_update': 0, 'sparse_snapshot': 0}
        self._active_bucket_min: int | None = None
        self._active_bucket_max: int = 0
        self._gpu_sparse_update_bytes: torch.Tensor | None = None
        self._gpu_sparse_snapshot_bytes: torch.Tensor | None = None
        self._gpu_active_min: torch.Tensor | None = None
        self._gpu_active_max: torch.Tensor | None = None
        self._snapshot = self._make_snapshot()

    def _flush_sparse_stats(self) -> None:
        if self._gpu_sparse_update_bytes is None:
            return
        self._state_bytes_estimate['sparse_update'] += int(self._gpu_sparse_update_bytes.item())
        self._state_bytes_estimate['sparse_snapshot'] += int(self._gpu_sparse_snapshot_bytes.item())
        g_min = int(self._gpu_active_min.item())
        g_max = int(self._gpu_active_max.item())
        self._active_bucket_min = g_min if self._active_bucket_min is None else min(self._active_bucket_min, g_min)
        self._active_bucket_max = max(self._active_bucket_max, g_max)
        self._gpu_sparse_update_bytes = None
        self._gpu_sparse_snapshot_bytes = None
        self._gpu_active_min = None
        self._gpu_active_max = None

    @property
    def state_bytes_estimate(self) -> dict:
        self._flush_sparse_stats()
        return self._state_bytes_estimate

    @state_bytes_estimate.setter
    def state_bytes_estimate(self, value: dict) -> None:
        self._flush_sparse_stats()
        self._state_bytes_estimate = value

    @property
    def active_bucket_min(self) -> int | None:
        self._flush_sparse_stats()
        return self._active_bucket_min

    @active_bucket_min.setter
    def active_bucket_min(self, value: int | None) -> None:
        self._flush_sparse_stats()
        self._active_bucket_min = value

    @property
    def active_bucket_max(self) -> int:
        self._flush_sparse_stats()
        return self._active_bucket_max

    @active_bucket_max.setter
    def active_bucket_max(self, value: int) -> None:
        self._flush_sparse_stats()
        self._active_bucket_max = int(value)

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
        ever_dS = self.dS is not None and (
            bool(self.dS._ever_nonzero_step)
            if isinstance(self.dS, LazySignedHash)
            else len(self.dS.active) > 0
        )
        ever_mutated = self._any_nonzero_lr_step or ever_dS
        if not self._offline_is_zero:
            is_zero = False
        else:
            is_zero = not ever_mutated
        if not self._offline_pair_zero:
            pair_zero = False
        elif self.bias_sparse_only:
            pair_zero = True
        else:
            pair_zero = not self._dA_ever_updated
        return ProposalSnapshot(
            self.offline.E, W, A, b, self.version, sparse, self.offline.M_sparse,
            None, is_zero, getattr(self.offline, 'unary_fn', None),
            _E_bf16=getattr(self.offline, '_E_bf16', None), _pair_zero=pair_zero,
        )

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

    def update(self, round_idx, l_unit, target_logits, H, prefix, tokens,
               weights=None, first_rejection=None, committed_tokens=None):
        if round_idx < 1 or round_idx <= self.last_round:
            raise ValueError('round_idx must increase and is one-based')
        self.last_round = round_idx
        if not self.accumulate_stride and (round_idx % self.update_stride):
            return False
        # Permit a caller whose model loop uses no_grad or inference_mode.
        use_analytic = self.analytic_online_grad and self.loss_mode == 'live_full'
        if use_analytic and self.kl_loss_fn is None:
            with torch.inference_mode(False), torch.no_grad():
                return self._update(round_idx, l_unit, target_logits, H, prefix, tokens,
                                    weights=weights, first_rejection=first_rejection,
                                    committed_tokens=committed_tokens)
        with torch.inference_mode(False), torch.enable_grad():
            return self._update(round_idx, l_unit, target_logits, H, prefix, tokens,
                                weights=weights, first_rejection=first_rejection,
                                committed_tokens=committed_tokens)

    def _compute_round_gradients_analytic(
        self, l_unit, target_logits, H, prefix, tokens,
        weights=None, first_rejection=None, *, _dedup_dirty: bool = True,
    ):
        if target_logits is None:
            raise ValueError('live_full online update requires target_logits')

        def _prep_in(x: torch.Tensor, *, as_float: bool = True) -> torch.Tensor:
            t = x.detach()
            if as_float and t.dtype != torch.float32:
                t = t.float()
            return t.clone() if t.is_inference() else t

        with torch.no_grad():
            l_unit = _prep_in(l_unit)
            H = _prep_in(H)
            target_logits = _prep_in(target_logits)
            if weights is not None:
                weights = _prep_in(weights, as_float=False)
            W = self.offline.W + self.dW if not self.bias_sparse_only else self._snapshot.W
            A = self.offline.A + self.dA if not self.bias_sparse_only else self._snapshot.A
            b = self.offline.b + self.db
            E = self.offline.E
            V = E.shape[0]
            K = len(H)
            bf16_gemm = bool(self.fastpath is not None and self.fastpath.bf16_gemm)
            E_bf16 = self.offline._token_features_bf16() if bf16_gemm else None

            U = block_unary(E, W, b, H, bf16_gemm=bf16_gemm, E_bf16=E_bf16)
            valid_slots, valid_d0, valid_prev = block_predecessor_pairs(prefix, tokens, K, len(A), E.device)
            has_valid_pairs = valid_slots.numel() > 0
            self._last_grad_had_pairs = has_valid_pairs
            pair_known_zero = self._offline_pair_zero and (self.bias_sparse_only or not self._dA_ever_updated)

            E_prev = None
            E_prev_bf16 = None
            if has_valid_pairs:
                if bf16_gemm:
                    assert E_bf16 is not None
                    if not pair_known_zero or not self.bias_sparse_only:
                        E_prev_bf16 = E_bf16.index_select(0, valid_prev)
                    if not pair_known_zero:
                        msgs = torch.bmm(
                            A.index_select(0, valid_d0).to(torch.bfloat16),
                            E_prev_bf16.unsqueeze(-1),
                        ).squeeze(-1).to(A.dtype)
                        slot_msgs = torch.zeros(K, E.shape[1], dtype=A.dtype, device=A.device).index_add_(
                            0, valid_slots, msgs
                        )
                        dense = U.add_((slot_msgs.to(torch.bfloat16) @ E_bf16.T).to(U.dtype))
                    else:
                        dense = U
                else:
                    E_cast = self.offline._token_features_for_A()
                    if not pair_known_zero or not self.bias_sparse_only:
                        E_prev = E_cast.index_select(0, valid_prev)
                    if not pair_known_zero:
                        msgs = torch.bmm(A.index_select(0, valid_d0), E_prev.unsqueeze(-1)).squeeze(-1)
                        slot_msgs = torch.zeros(K, E.shape[1], dtype=A.dtype, device=A.device).index_add_(
                            0, valid_slots, msgs
                        )
                        dense = U.add_(slot_msgs @ E_cast.T)
                    else:
                        dense = U
            else:
                dense = U

            candidates = None
            buckets = signs = dirty = inverse = None
            if self.dS is not None:
                if has_valid_pairs:
                    candidates = top_candidates(l_unit - dense, self.offline.M_sparse)
                    cands_pair = candidates.index_select(0, valid_slots)
                    d_pair = (valid_d0 + 1)[:, None]
                    prev_pair = valid_prev[:, None]
                    if self.sparse_backend == 'signed_feature_hash':
                        off_sp = self.offline.sparse
                        if off_sp is not None and hasattr(off_sp, '_can_share_hash') and off_sp._can_share_hash():
                            raw_bucket, signs = _hash_raw_and_sign(d_pair, prev_pair, cands_pair, E.device)
                            buckets = _bucket_from_raw(raw_bucket, self.dS.capacity)
                            pair_vals = self.dS.read(buckets) * signs + off_sp._lookup_from_raw(raw_bucket, signs)
                        else:
                            buckets, signs = hash_keys(d_pair, prev_pair, cands_pair, self.dS.capacity, E.device)
                            pair_vals = self.dS.read(buckets) * signs
                            if off_sp is not None:
                                pair_vals = pair_vals + off_sp.lookup(d_pair, prev_pair, cands_pair)
                    else:
                        packed = pack_exact_keys(d_pair, prev_pair, cands_pair, V, device=E.device)
                        dirty, inverse = torch.unique(packed.reshape(-1), sorted=True, return_inverse=True)
                        pair_vals = self.dS.read(dirty)[inverse].reshape(packed.shape)
                        if self.offline.sparse is not None:
                            pair_vals = pair_vals + self.offline.sparse.lookup(d_pair, prev_pair, cands_pair)
                    slot_vals = torch.zeros(
                        K, candidates.shape[1], dtype=dense.dtype, device=dense.device
                    ).index_add_(0, valid_slots, pair_vals.to(dense.dtype))
                    dense.scatter_add_(1, candidates, slot_vals)
                    scores = l_unit - dense
                else:
                    scores = l_unit - dense
            elif self.offline.sparse is not None and has_valid_pairs:
                candidates = top_candidates(l_unit - dense, self.offline.M_sparse)
                cands_pair = candidates.index_select(0, valid_slots)
                pair_vals = self.offline.sparse.lookup(
                    (valid_d0 + 1)[:, None], valid_prev[:, None], cands_pair
                ).to(dense.dtype)
                slot_vals = torch.zeros(
                    K, candidates.shape[1], dtype=dense.dtype, device=dense.device
                ).index_add_(0, valid_slots, pair_vals)
                dense.scatter_add_(1, candidates, slot_vals)
                scores = l_unit - dense
            else:
                scores = l_unit - dense

            if weights is None:
                weights_t = loss_weights(K, l_unit.device, self.weighting_mode, self.gamma, first_rejection)
            else:
                weights_t = weights

            if self.kl_loss_fn is not None:
                scores_leaf = scores.detach().requires_grad_(True)
                with torch.enable_grad():
                    loss_t = self.kl_loss_fn(scores_leaf, target_logits, weights_t)
                    (grad_scores,) = torch.autograd.grad(loss_t, scores_leaf)
                loss = loss_t.detach()
                G = grad_scores.detach().float().neg_()
            elif self.fastpath is not None and (
                self.fastpath.static_block_length is None or K == self.fastpath.static_block_length
            ):
                loss, G = self.fastpath.analytic_kl_grad_fn(scores, target_logits, weights_t)
            else:
                loss, G = analytic_full_kl_and_correction_grad(scores, target_logits, weights_t)

            g_db = G.sum(dim=0)
            if self.bias_sparse_only:
                g_dW = None
                g_dA = None
            elif bf16_gemm:
                assert E_bf16 is not None
                G_E_bf16 = G.to(torch.bfloat16) @ E_bf16
                g_dW = (G_E_bf16.T @ H.to(torch.bfloat16)).to(self.dW.dtype)
                if has_valid_pairs:
                    G_pair_bf16 = G_E_bf16.index_select(0, valid_slots)
                    outer_pair = torch.bmm(G_pair_bf16.unsqueeze(2), E_prev_bf16.unsqueeze(1)).to(self.dA.dtype)
                    g_dA = torch.zeros_like(self.dA).index_add_(0, valid_d0, outer_pair)
                else:
                    g_dA = torch.zeros_like(self.dA)
            else:
                E_cast = self.offline._token_features_for_A()
                G_E = G @ E_cast
                g_dW = (G_E.T @ H).to(self.dW.dtype)
                if has_valid_pairs:
                    G_pair = G_E.index_select(0, valid_slots)
                    outer_pair = torch.bmm(G_pair.unsqueeze(2), E_prev.unsqueeze(1)).to(self.dA.dtype)
                    g_dA = torch.zeros_like(self.dA).index_add_(0, valid_d0, outer_pair)
                else:
                    g_dA = torch.zeros_like(self.dA)

            if self.dS is None:
                g_sparse = None
            elif not has_valid_pairs:
                dirty = torch.empty(0, device=E.device, dtype=torch.long)
                g_sparse = torch.empty(0, device=E.device, dtype=torch.float32)
            else:
                g_slot_vals = G.gather(1, candidates)
                g_pair_vals = g_slot_vals.index_select(0, valid_slots)
                if self.sparse_backend == 'signed_feature_hash':
                    pair_grads = (g_pair_vals * signs).reshape(-1)
                    if _dedup_dirty:
                        dirty, inv = torch.unique(buckets.reshape(-1), sorted=True, return_inverse=True)
                        g_sparse = torch.zeros(len(dirty), device=E.device, dtype=torch.float32).scatter_add_(
                            0, inv, pair_grads
                        )
                    else:
                        dirty = buckets.reshape(-1)
                        g_sparse = pair_grads
                else:
                    g_sparse = torch.zeros(len(dirty), device=E.device, dtype=torch.float32).scatter_add_(
                        0, inverse, g_pair_vals.reshape(-1)
                    )

            return loss.detach(), g_dW, g_dA, g_db, dirty, g_sparse

    def _compute_round_gradients(self, l_unit, target_logits, H, prefix, tokens,
                                 weights=None, first_rejection=None, committed_tokens=None,
                                 *, _dedup_dirty: bool = True):
        if self.analytic_online_grad and self.loss_mode == 'live_full':
            return self._compute_round_gradients_analytic(
                l_unit, target_logits, H, prefix, tokens,
                weights=weights, first_rejection=first_rejection, _dedup_dirty=_dedup_dirty,
            )

        def _prep_in(x: torch.Tensor, *, as_float: bool = True) -> torch.Tensor:
            t = x.detach()
            if as_float and t.dtype != torch.float32:
                t = t.float()
            return t.clone() if t.is_inference() else t

        l_unit = _prep_in(l_unit)
        H = _prep_in(H)
        target_logits = _prep_in(target_logits) if target_logits is not None else None
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
        V = E.shape[0]
        bf16_gemm = bool(self.fastpath is not None and self.fastpath.bf16_gemm)
        E_bf16 = self.offline._token_features_bf16() if bf16_gemm else None
        dirty = sparse_values = None

        if self.loss_mode == 'two_sided_pseudolikelihood':
            self._last_grad_had_pairs = True
            if committed_tokens is None:
                if first_rejection is None or first_rejection == len(tokens):
                    committed = [int(t) for t in tokens]
                else:
                    if target_logits is None:
                        raise ValueError('two_sided_pseudolikelihood requires committed_tokens or target_logits')
                    committed = [int(t) for t in tokens[:first_rejection]] + [int(target_logits[first_rejection].argmax())]
                if not committed:
                    raise ValueError('two_sided_pseudolikelihood requires at least one committed verified token')
            else:
                committed = [int(t) for t in committed_tokens]
                if not committed or len(H) == 0:
                    self._last_grad_had_pairs = False
                    return None, None, None, None, None, None
            n_eval = min(len(H), len(committed))
            if n_eval <= 0:
                self._last_grad_had_pairs = False
                return None, None, None, None, None, None
            U = block_unary(E, W, b, H[:n_eval], bf16_gemm=bf16_gemm, E_bf16=E_bf16)
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
                weights_t = weights[:n_eval]
            loss = two_sided_pseudolikelihood_loss(l_unit[:n_eval] - correction, committed[:n_eval], weights_t)
        else:
            if target_logits is None:
                raise ValueError('live_full online update requires target_logits')
            K = len(H)
            U = block_unary(E, W, b, H, bf16_gemm=bf16_gemm, E_bf16=E_bf16)
            valid_slots, valid_d0, valid_prev = block_predecessor_pairs(prefix, tokens, K, len(A), E.device)
            self._last_grad_had_pairs = valid_slots.numel() > 0
            if valid_slots.numel():
                if bf16_gemm:
                    assert E_bf16 is not None
                    E_prev = E_bf16.index_select(0, valid_prev)
                    msgs = torch.bmm(A.index_select(0, valid_d0).to(torch.bfloat16), E_prev.unsqueeze(-1)).squeeze(-1).to(A.dtype)
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
            if self.dS is not None:
                if valid_slots.numel():
                    candidates = top_candidates(l_unit - dense, self.offline.M_sparse)
                    cands_pair = candidates.index_select(0, valid_slots)
                    d_pair = (valid_d0 + 1)[:, None]
                    prev_pair = valid_prev[:, None]
                    if self.sparse_backend == 'signed_feature_hash':
                        buckets, signs = hash_keys(d_pair, prev_pair, cands_pair, self.dS.capacity, E.device)
                        dirty, inverse = torch.unique(buckets.reshape(-1), sorted=True, return_inverse=True)
                        sparse_values = self.dS.read(dirty).detach().clone().requires_grad_(True)
                        pair_vals = sparse_values[inverse].reshape(buckets.shape) * signs
                    else:
                        packed = pack_exact_keys(d_pair, prev_pair, cands_pair, V, device=E.device)
                        dirty, inverse = torch.unique(packed.reshape(-1), sorted=True, return_inverse=True)
                        sparse_values = self.dS.read(dirty).detach().clone().requires_grad_(True)
                        pair_vals = sparse_values[inverse].reshape(packed.shape)
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
            if weights is None:
                weights_t = loss_weights(K, l_unit.device, self.weighting_mode, self.gamma, first_rejection)
            else:
                weights_t = weights
            if self.kl_loss_fn is not None:
                kl_fn = self.kl_loss_fn
            elif self.fastpath is not None and (self.fastpath.static_block_length is None or K == self.fastpath.static_block_length):
                kl_fn = self.fastpath.full_kl_fn
            else:
                kl_fn = full_kl
            loss = kl_fn(l_unit - correction, target_logits, weights_t)

        loss.backward()
        g_dW = None if self.bias_sparse_only else (dW.grad.detach() if dW.grad is not None else torch.zeros_like(dW))
        g_dA = None if self.bias_sparse_only else (dA.grad.detach() if dA.grad is not None else torch.zeros_like(dA))
        g_db = db.grad.detach() if db.grad is not None else torch.zeros_like(db)
        if self.dS is None:
            g_sparse = None
        elif sparse_values is not None:
            g_sparse = sparse_values.grad.detach() if sparse_values.grad is not None else torch.zeros_like(sparse_values)
        else:
            g_sparse = torch.empty(0, device=E.device, dtype=torch.float32)
        return loss.detach(), g_dW, g_dA, g_db, dirty, g_sparse

    def _apply_step(self, lr: float, loss_value: torch.Tensor | float, g_dW, g_dA, g_db, dirty, g_sparse,
                    *, cloned_rounds: int = 1, dirty_is_unique: bool = False, had_pairs: bool = True):
        with torch.no_grad():
            precomputed_dense = None
            extra_opt_bytes = 0
            if self.optimizer == 'sgd':
                if self.fastpath is not None:
                    self.dW, self.dA, self.db, snap_W, snap_A, snap_b = self.fastpath.online_dense_step(
                        self.dW, g_dW, self.dA, g_dA, self.db, g_db,
                        self.offline.W, self.offline.A, self.offline.b,
                        lr=lr, lambda_l2=self.lambda_l2, lambda_l1_b=self.lambda_l1_b,
                        bias_sparse_only=self.bias_sparse_only,
                    )
                    precomputed_dense = (snap_W, snap_A, snap_b)
                else:
                    shrink = 1 + lr * self.lambda_l2
                    if not self.bias_sparse_only:
                        self.dW = (self.dW - lr * g_dW) / shrink
                        self.dA = (self.dA - lr * g_dA) / shrink
                    self.db = proximal_l1((self.db - lr * g_db) / shrink, lr * self.lambda_l1_b)
            else:
                extra_opt_bytes = self._adaptive_dense_step(
                    self.dW, g_dW, self.dA, g_dA, self.db, g_db, lr, self.u + 1
                )
            if self.dS is not None:
                # Release OnlineState's internal cached snapshot before mutating dS so CoW undo
                # nodes are recorded ONLY when an external caller holds a prior snapshot.
                self._snapshot = None
                if self.sparse_backend == 'signed_feature_hash' and dirty_is_unique:
                    self.dS.step_unique(dirty, g_sparse, lr, self.lambda_l1_s)
                else:
                    self.dS.step(dirty, g_sparse, lr, self.lambda_l1_s)
            if lr > 0.0:
                self._any_nonzero_lr_step = True
                if not self.bias_sparse_only and had_pairs:
                    self._dA_ever_updated = True
            self.u += 1
            self.version += 1
            self._last_loss_tensor = loss_value
            self._snapshot = self._make_snapshot(precomputed_dense=precomputed_dense)
            dense_bytes = sum(x.numel() * x.element_size() for x in (self.dW, self.dA, self.db))
            db_bytes = self.db.numel() * self.db.element_size()
            optimized_bytes = db_bytes if self.bias_sparse_only else dense_bytes
            self._state_bytes_estimate['dense_activation_clone'] += 2 * dense_bytes * cloned_rounds
            # Conceptual reads/writes, not a profiler-derived hardware traffic count.
            self._state_bytes_estimate['dense_optimizer'] += 5 * optimized_bytes + 2 * db_bytes + extra_opt_bytes
            self._state_bytes_estimate['dense_snapshot'] += 3 * dense_bytes
            if self.dS is not None:
                if isinstance(self.dS, LazySignedHash):
                    act_t = self.dS._active_count
                    upd_t = self.dS._last_update_bytes_t
                    snap2_t = 32 * act_t + 8 * (self.dS.u + 1)
                    if self._gpu_sparse_update_bytes is None:
                        self._gpu_sparse_update_bytes = upd_t.clone()
                        self._gpu_sparse_snapshot_bytes = snap2_t.clone()
                        self._gpu_active_min = act_t.clone()
                        self._gpu_active_max = act_t.clone()
                    else:
                        self._gpu_sparse_update_bytes.add_(upd_t)
                        self._gpu_sparse_snapshot_bytes.add_(snap2_t)
                        torch.minimum(self._gpu_active_min, act_t, out=self._gpu_active_min)
                        torch.maximum(self._gpu_active_max, act_t, out=self._gpu_active_max)
                else:
                    active = len(self.dS.active)
                    snapshot_bytes = active * 16 + (self.dS.u + 1) * 4
                    self._state_bytes_estimate['sparse_update'] += self.dS.last_update_bytes
                    self._state_bytes_estimate['sparse_snapshot'] += 2 * snapshot_bytes
                    self._active_bucket_min = active if self._active_bucket_min is None else min(self._active_bucket_min, active)
                    self._active_bucket_max = max(self._active_bucket_max, active)

    def _update(self, round_idx, l_unit, target_logits, H, prefix, tokens,
                weights=None, first_rejection=None, committed_tokens=None):
        lr_check = float(self.lr(self.u + 1) if callable(self.lr) else self.lr)
        if lr_check < 0:
            raise ValueError('scheduled learning rate must be nonnegative')
        skip_dedup = (
            self.analytic_online_grad
            and self.loss_mode == 'live_full'
            and self.sparse_backend == 'signed_feature_hash'
        )
        loss_val, g_dW, g_dA, g_db, dirty, g_sparse = self._compute_round_gradients(
            l_unit, target_logits, H, prefix, tokens,
            weights=weights, first_rejection=first_rejection, committed_tokens=committed_tokens,
            _dedup_dirty=not skip_dedup,
        )
        if not self.accumulate_stride:
            if loss_val is None:
                return False
            self._apply_step(lr_check, loss_val, g_dW, g_dA, g_db, dirty, g_sparse,
                             cloned_rounds=1, dirty_is_unique=not skip_dedup,
                             had_pairs=self._last_grad_had_pairs)
            return True
        with torch.no_grad():
            if loss_val is not None:
                self._buffered_rounds += 1
                self._buf_loss = self._buf_loss + loss_val
                if self._last_grad_had_pairs:
                    self._buf_had_pairs = True
                if not self.bias_sparse_only:
                    self._buf_dW = g_dW.clone() if self._buf_dW is None else (self._buf_dW + g_dW)
                    self._buf_dA = g_dA.clone() if self._buf_dA is None else (self._buf_dA + g_dA)
                self._buf_db = g_db.clone() if self._buf_db is None else (self._buf_db + g_db)
                if self.dS is not None and dirty is not None and dirty.numel() > 0:
                    self._buf_sparse_idx.append(dirty.detach().clone())
                    self._buf_sparse_grad.append(g_sparse.detach().clone())
            if round_idx % self.update_stride != 0 or self._buffered_rounds == 0:
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
            had_pairs_win = self._buf_had_pairs
            self._buffered_rounds = 0
            self._buf_loss = 0.0
            self._buf_had_pairs = False
            self._buf_dW = None
            self._buf_dA = None
            self._buf_db = None
            self._buf_sparse_idx = []
            self._buf_sparse_grad = []
            self._apply_step(lr_check, avg_loss, agg_dW, agg_dA, agg_db, agg_dirty, agg_sparse,
                             cloned_rounds=rounds_in_window, dirty_is_unique=False,
                             had_pairs=had_pairs_win)
            return True

    def metrics(self):
        # Evaluate current lazy shrinkage without materializing it into storage.
        active_idx = self.dS.active if self.dS is not None else None
        active_len = len(active_idx) if active_idx is not None else 0
        effective_dS = self.dS.read(active_idx) if active_idx is not None else None
        s_bytes = dict(self.state_bytes_estimate)
        result = {'adapter_version': self.version, 'u': self.u, 'last_round': self.last_round,
                  'loss_mode': self.loss_mode, 'T_train': 1., 'beta_train': 1.,
                  'dW_norm': float(self.dW.norm()), 'dA_norm': float(self.dA.norm()),
                  'db_norm': float(self.db.norm()),
                  'dS_norm': float(effective_dS.norm()) if effective_dS is not None else 0.,
                  'dS_norm_definition': 'L2 norm of effective lazy-shrunk request delta at current optimizer u; zero unrepresented buckets; pure read',
                  'db_exact_zero_fraction': float((self.db == 0).float().mean()),
                  'active_buckets': active_len,
                  'active_buckets_min_over_updates': self.active_bucket_min,
                  'active_buckets_max_over_updates': self.active_bucket_max,
                  'state_bytes_estimate_by_operation': s_bytes,
                  'state_bytes_estimate_total': sum(s_bytes.values()),
                  'state_bytes_definition': 'conceptual FP32 parameter/timestamp/index reads+writes for state clones, SGD/prox steps and snapshots; excludes E/logit/autograd activation traffic and is not hardware HBM traffic',
                  'sparse_capacity': self.dS.capacity if self.dS is not None else 0,
                  'sparse_update_bytes_estimate': self.dS.last_update_bytes if self.dS is not None else 0,
                  'sparse_snapshot_bytes': (active_len * 16 + (self.dS.u + 1) * 4) if self.dS is not None else 0,
                  'dense_snapshot_bytes': sum(x.numel() * x.element_size() for x in (self._snapshot.W, self._snapshot.A, self._snapshot.b)),
                  'sparse_snapshot_strategy': 'compact_active_bucket_copy' if self.sparse_backend == 'signed_feature_hash' else 'compact_active_exact_pair_copy',
                  'sparse_update_strategy': 'dirty_lazy_catchup_and_active_list_compaction',
                  'last_full_loss': self.last_loss}
        if self.optimizer != 'sgd':
            result['dense_optimizer'] = self.optimizer
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
