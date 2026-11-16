"""Deterministic signed hashing and read-only lazy proximal snapshots."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from collections import Counter
import weakref
import torch
from torch import nn


def _as_int64_tensor(value, device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.int64 and (device is None or value.device == device):
            return value
        return value.to(dtype=torch.int64, device=device)
    return torch.as_tensor(value, dtype=torch.int64, device=device)


def _hash_raw_and_sign(d, previous, current, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    target_device = torch.device(device) if device is not None else None
    if target_device is None:
        for item in (d, previous, current):
            if isinstance(item, torch.Tensor):
                target_device = item.device
                break
    d_t = _as_int64_tensor(d, target_device)
    prev_t = _as_int64_tensor(previous, target_device)
    curr_t = _as_int64_tensor(current, target_device)
    # Separate prime-modular hashes; no Python randomized hash or platform RNG.
    # Multiply narrow shapes before broadcasting on addition to avoid redundant work.
    prime = 2147483647
    mixed = ((d_t * 73856093 + prev_t * 19349663) + (curr_t * 83492791 + 104729)) % prime
    raw_bucket = (mixed * 48271 + 1) % prime
    sign_hash = ((d_t * 15485863 + prev_t * 32452843) + (curr_t * 49979687 + 67867967)) % prime
    sign = 1 - 2 * (((sign_hash * 40699 + 17) % prime) & 1)
    return raw_bucket, sign.to(torch.float32)


def _bucket_from_raw(raw_bucket: torch.Tensor, capacity: int) -> torch.Tensor:
    if (capacity & (capacity - 1)) == 0:
        return raw_bucket & (capacity - 1)
    return raw_bucket % capacity


def hash_keys(d, previous, current, capacity: int, device=None):
    if capacity <= 0:
        raise ValueError('hash capacity must be positive')
    raw_bucket, signs = _hash_raw_and_sign(d, previous, current, device)
    return _bucket_from_raw(raw_bucket, capacity), signs


def proximal_l1(values: torch.Tensor, threshold) -> torch.Tensor:
    return values.sign() * (values.abs() - threshold).clamp_min(0)


def _gather_1d(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return table.gather(0, idx.reshape(-1)).reshape(idx.shape)


@dataclass(frozen=True)
class FrozenHash:
    values: torch.Tensor
    capacity: int
    lookup_fn: object | None = None

    def _can_share_hash(self) -> bool:
        return self.lookup_fn is None or (torch.is_grad_enabled() and self.values.requires_grad)

    def _lookup_from_raw(self, raw_bucket: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        return _gather_1d(self.values, _bucket_from_raw(raw_bucket, self.capacity)) * signs

    def lookup(self, d, previous, current):
        if self.lookup_fn is not None and not (torch.is_grad_enabled() and self.values.requires_grad):
            return self.lookup_fn(d, previous, current, self.values, self.capacity)
        buckets, signs = hash_keys(d, previous, current, self.capacity, self.values.device)
        return _gather_1d(self.values, buckets) * signs


class SignedHash(nn.Module):
    def __init__(self, capacity: int = 2**24, device=None, *, lookup_fn=None):
        super().__init__()
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.values = nn.Parameter(torch.zeros(capacity, dtype=torch.float32, device=device))
        self.lookup_fn = lookup_fn

    def _can_share_hash(self) -> bool:
        return self.lookup_fn is None or (torch.is_grad_enabled() and self.values.requires_grad)

    def _lookup_from_raw(self, raw_bucket: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        return _gather_1d(self.values, _bucket_from_raw(raw_bucket, self.capacity)) * signs

    def lookup(self, d, previous, current):
        if self.lookup_fn is not None and not (torch.is_grad_enabled() and self.values.requires_grad):
            return self.lookup_fn(d, previous, current, self.values, self.capacity)
        buckets, signs = hash_keys(d, previous, current, self.capacity, self.values.device)
        return _gather_1d(self.values, buckets) * signs

    def snapshot(self):
        # Offline fit stops before decoding; capture once when a request is created.
        return FrozenHash(self.values.detach().clone(), self.capacity, self.lookup_fn)

    @torch.no_grad()
    def proximal_step(self, lr: float, lambda_l1: float):
        self.values.copy_(proximal_l1(self.values, lr * lambda_l1))

    def collision_report(self, keys: list[tuple[int, int, int]]) -> dict:
        counts = Counter(keys)
        occupancy = Counter()
        weighted = Counter()
        for key, frequency in counts.items():
            bucket = int(hash_keys(*key, self.capacity)[0])
            occupancy[bucket] += 1
            weighted[bucket] += frequency
        return {'capacity': self.capacity, 'unique_observed_keys': len(counts),
                'occupied_buckets': len(occupancy), 'load_factor': len(counts) / self.capacity,
                'collision_count': sum(n - 1 for n in occupancy.values()),
                'frequency_weighted_collision_count': sum(weighted[b] for b, n in occupancy.items() if n > 1),
                'memory_bytes': self.values.numel() * self.values.element_size()}


class _LazyUndoNode:
    """Copy-on-write undo link recording pre-mutation values for dirty buckets only."""
    __slots__ = ('indices', 'old_values', 'old_last_u', 'next_node', '__weakref__')

    def __init__(self):
        self.indices: torch.Tensor | None = None
        self.old_values: torch.Tensor | None = None
        self.old_last_u: torch.Tensor | None = None
        self.next_node: _LazyUndoNode | None = None


class LazyHashSnapshot:
    """Immutable lazy-L1 sparse snapshot with O(1) direct head reads and O(|dirty|) CoW undo.

    While this snapshot points to the live head of `LazySignedHash` (`_undo_node.next_node is None`),
    `read(indices)` indexes the GPU table directly in O(1) without cloning the `2^20` arrays
    or executing `torch.searchsorted`. If later `step`/`step_unique` updates mutate dirty buckets
    while an older snapshot remains referenced, the mutated buckets' prior `(values, last_u)` are
    recorded in `_LazyUndoNode` so historical snapshots remain 0-ULP bit-exact.
    """
    __slots__ = (
        'u', 'capacity', '_table_values', '_table_last_u', '_thresh_buf',
        '_undo_node', '_compact_cache',
    )

    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        last_u: torch.Tensor,
        cum_thresh: torch.Tensor,
        u: int,
        capacity: int,
    ):
        object.__setattr__(self, 'u', int(u))
        object.__setattr__(self, 'capacity', int(capacity))
        object.__setattr__(self, '_table_values', None)
        object.__setattr__(self, '_table_last_u', None)
        object.__setattr__(self, '_thresh_buf', cum_thresh)
        object.__setattr__(self, '_undo_node', None)
        object.__setattr__(self, '_compact_cache', (indices, values, last_u))

    @classmethod
    def _from_table(
        cls,
        table_values: torch.Tensor,
        table_last_u: torch.Tensor,
        thresh_buf: torch.Tensor,
        u: int,
        capacity: int,
        undo_node: _LazyUndoNode,
    ) -> 'LazyHashSnapshot':
        inst = object.__new__(cls)
        object.__setattr__(inst, 'u', int(u))
        object.__setattr__(inst, 'capacity', int(capacity))
        object.__setattr__(inst, '_table_values', table_values)
        object.__setattr__(inst, '_table_last_u', table_last_u)
        object.__setattr__(inst, '_thresh_buf', thresh_buf)
        object.__setattr__(inst, '_undo_node', undo_node)
        object.__setattr__(inst, '_compact_cache', None)
        return inst

    def __setattr__(self, name, value):
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name):
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    @property
    def _device(self) -> torch.device:
        if self._table_values is not None:
            return self._table_values.device
        return self._compact_cache[1].device

    def _can_share_hash(self) -> bool:
        return True

    def _lookup_from_raw(self, raw_bucket: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        return self.read(_bucket_from_raw(raw_bucket, self.capacity)) * signs

    def _materialize_compact(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._compact_cache
        if cached is not None:
            return cached
        raw_val = self._table_values
        raw_u = self._table_last_u
        node = self._undo_node
        if node is not None and node.next_node is not None:
            nodes = []
            curr = node
            while curr is not None and curr.next_node is not None:
                nodes.append(curr)
                curr = curr.next_node
            raw_val = raw_val.clone()
            raw_u = raw_u.clone()
            for nd in reversed(nodes):
                if nd.indices is not None and nd.indices.numel() > 0:
                    raw_val.scatter_(0, nd.indices, nd.old_values)
                    raw_u.scatter_(0, nd.indices, nd.old_last_u)
        delta_t = self._thresh_buf[self.u] - self._thresh_buf[raw_u.long()]
        alive = ~((raw_val.abs() - delta_t) <= 0.0)
        act_idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
        compact = (act_idx, raw_val[act_idx].clone(), raw_u[act_idx].clone())
        object.__setattr__(self, '_compact_cache', compact)
        return compact

    @property
    def indices(self) -> torch.Tensor:
        return self._materialize_compact()[0]

    @property
    def values(self) -> torch.Tensor:
        return self._materialize_compact()[1]

    @property
    def last_u(self) -> torch.Tensor:
        return self._materialize_compact()[2]

    @property
    def cum_thresh(self) -> torch.Tensor:
        return self._thresh_buf[:self.u + 1]

    def read(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices if indices.dtype == torch.int64 else indices.long()
        if self._table_values is None:
            flat = idx.reshape(-1)
            comp_idx, comp_val, comp_u = self._compact_cache
            if comp_idx.numel() == 0:
                return torch.zeros(idx.shape, dtype=torch.float32, device=comp_val.device)
            pos = torch.searchsorted(comp_idx, flat)
            safe = pos.clamp_max(len(comp_idx) - 1)
            present = (pos < len(comp_idx)) & (comp_idx.gather(0, safe) == flat)
            effective = proximal_l1(
                comp_val.gather(0, safe),
                self._thresh_buf[self.u] - self._thresh_buf.gather(0, comp_u.gather(0, safe).long()),
            )
            return torch.where(present, effective, 0.0).reshape(idx.shape)

        node = self._undo_node
        if node is None or node.next_node is None:
            flat = idx.reshape(-1)
            raw_val = self._table_values.gather(0, flat)
            raw_u = self._table_last_u.gather(0, flat).long()
            return proximal_l1(raw_val, self._thresh_buf[self.u] - self._thresh_buf.gather(0, raw_u)).reshape(idx.shape)

        flat = idx.reshape(-1)
        if flat.numel() == 0:
            return torch.zeros(idx.shape, dtype=torch.float32, device=self._table_values.device)
        nodes = []
        curr = node
        while curr is not None and curr.next_node is not None:
            nodes.append(curr)
            curr = curr.next_node
        raw_val = self._table_values.gather(0, flat).clone()
        raw_u = self._table_last_u.gather(0, flat).clone()
        for nd in reversed(nodes):
            n_idx = nd.indices
            if n_idx is not None and n_idx.numel() > 0:
                pos = torch.searchsorted(n_idx, flat)
                safe = pos.clamp_max(n_idx.numel() - 1)
                hit = (pos < n_idx.numel()) & (n_idx.gather(0, safe) == flat)
                raw_val = torch.where(hit, nd.old_values.gather(0, safe), raw_val)
                raw_u = torch.where(hit, nd.old_last_u.gather(0, safe), raw_u)
        effective = proximal_l1(raw_val, self._thresh_buf[self.u] - self._thresh_buf.gather(0, raw_u.long()))
        return effective.reshape(idx.shape)

    def lookup(self, d, previous, current):
        indices, signs = hash_keys(d, previous, current, self.capacity, self._device)
        return self.read(indices) * signs

    @property
    def snapshot_bytes(self):
        comp_idx, comp_val, comp_u = self._materialize_compact()
        ct = self.cum_thresh
        return sum(x.numel() * x.element_size() for x in (comp_idx, comp_val, comp_u, ct))


class LazySignedHash:
    """Fixed request table; zero-sync O(|dirty|) updates with CoW snapshots and scheduled lazy L1."""
    def __init__(self, capacity: int = 2**20, device=None):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = int(capacity)
        self.values = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.last_u = torch.zeros(self.capacity, dtype=torch.int32, device=device)
        dev = self.values.device
        self.active_mask = torch.zeros(self.capacity, dtype=torch.bool, device=dev)
        self._grad_accum = torch.zeros(self.capacity, dtype=torch.float32, device=dev)
        self._active_count = torch.zeros((), dtype=torch.int64, device=dev)
        self._last_dirty_count = torch.zeros((), dtype=torch.int64, device=dev)
        self._last_update_bytes_t = torch.zeros((), dtype=torch.int64, device=dev)
        self._manual_last_update_bytes: int | None = None
        self._active_cache: torch.Tensor | None = None
        self._head_ref: weakref.ReferenceType[_LazyUndoNode] | None = None
        self.cum_thresh = [0.0]
        self._cum_thresh_buf = torch.zeros(4096, dtype=torch.float32, device=dev)
        self._cum_thresh_len = 1
        self.u = 0
        self._ever_nonzero_step = False
        if dev.type == 'cpu' and self.capacity > 4096:
            n_blks = (self.capacity + 1023) >> 10
            self._cpu_blk_cnt: torch.Tensor | None = torch.zeros(n_blks, dtype=torch.int32)
            self._cpu_blk_min_exp: torch.Tensor | None = torch.full((n_blks,), torch.inf, dtype=torch.float32)
        else:
            self._cpu_blk_cnt = None
            self._cpu_blk_min_exp = None

    @property
    def active(self) -> torch.Tensor:
        if self._active_cache is None:
            self._active_cache = torch.nonzero(self.active_mask, as_tuple=False).squeeze(-1)
        return self._active_cache

    @property
    def last_update_bytes(self) -> int:
        if self._manual_last_update_bytes is not None:
            return self._manual_last_update_bytes
        if self.u == 0:
            return 0
        return int(self._last_update_bytes_t.item())

    @last_update_bytes.setter
    def last_update_bytes(self, value: int) -> None:
        self._manual_last_update_bytes = int(value)

    def _can_share_hash(self) -> bool:
        return True

    def _lookup_from_raw(self, raw_bucket: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        return self.read(_bucket_from_raw(raw_bucket, self.capacity)) * signs

    def _thresholds(self) -> torch.Tensor:
        n = len(self.cum_thresh)
        if self._cum_thresh_len != n:
            # Allocate a fresh buffer so extant snapshots' views of _thresh_buf remain immutable.
            cap = max(n, self._cum_thresh_buf.numel() * 2)
            new_buf = torch.empty(cap, dtype=torch.float32, device=self.values.device)
            new_buf[:n].copy_(torch.tensor(self.cum_thresh, device=self.values.device, dtype=torch.float32))
            self._cum_thresh_buf = new_buf
            self._cum_thresh_len = n
        return self._cum_thresh_buf[:n]

    def _append_threshold(self, new_thresh: float) -> torch.Tensor:
        self.u += 1
        self.cum_thresh.append(new_thresh)
        if self._cum_thresh_len + 1 == len(self.cum_thresh):
            if self.u >= self._cum_thresh_buf.numel():
                new_buf = torch.empty(self._cum_thresh_buf.numel() * 2, dtype=torch.float32, device=self.values.device)
                new_buf[:self.u].copy_(self._cum_thresh_buf[:self.u])
                self._cum_thresh_buf = new_buf
            self._cum_thresh_buf[self.u : self.u + 1].fill_(new_thresh)
            self._cum_thresh_len = self.u + 1
            return self._cum_thresh_buf[:self.u + 1]
        return self._thresholds()

    def _record_undo_sorted(self, sorted_indices: torch.Tensor) -> None:
        if sorted_indices.numel() == 0 or self._head_ref is None:
            return
        head = self._head_ref()
        if head is None:
            self._head_ref = None
            return
        new_head = _LazyUndoNode()
        head.indices = sorted_indices
        head.old_values = self.values.gather(0, sorted_indices)
        head.old_last_u = self.last_u.gather(0, sorted_indices)
        head.next_node = new_head
        self._head_ref = weakref.ref(new_head)

    def read(self, indices: torch.Tensor):
        idx = indices if indices.dtype == torch.int64 else indices.long()
        thresholds = self._thresholds()
        flat = idx.reshape(-1)
        raw_u = self.last_u.gather(0, flat).long()
        return proximal_l1(
            self.values.gather(0, flat),
            thresholds[self.u] - thresholds.gather(0, raw_u),
        ).reshape(idx.shape)

    def lookup(self, d, previous, current) -> torch.Tensor:
        indices, signs = hash_keys(d, previous, current, self.capacity, self.values.device)
        return self.read(indices) * signs

    def snapshot(self) -> LazyHashSnapshot:
        head = None if self._head_ref is None else self._head_ref()
        if head is None:
            head = _LazyUndoNode()
            self._head_ref = weakref.ref(head)
        thresh = self._thresholds()
        return LazyHashSnapshot._from_table(
            self.values, self.last_u, self._cum_thresh_buf[:len(thresh)],
            self.u, self.capacity, head,
        )

    def _sweep_untouched_l1_decay(self, thresh_buf: torch.Tensor) -> None:
        if self._cpu_blk_cnt is not None:
            thresh_u_t = thresh_buf[self.u]
            thresh_now = float(thresh_u_t)
            cand_blks = torch.nonzero(
                (self._cpu_blk_cnt > 0) & ~(thresh_now < self._cpu_blk_min_exp),
                as_tuple=False,
            ).squeeze(-1)
            for blk in cand_blks.tolist():
                start = blk << 10
                end = min(start + 1024, self.capacity)
                mask_slice = self.active_mask[start:end]
                val_slice = self.values[start:end]
                u_slice = self.last_u[start:end].long()
                delta_slice = thresh_u_t - thresh_buf[u_slice]
                alive_slice = mask_slice & ~((val_slice.abs() - delta_slice) <= 0.0)
                old_cnt = int(self._cpu_blk_cnt[blk].item())
                new_cnt = int(alive_slice.sum().item())
                if new_cnt != old_cnt:
                    self.active_mask[start:end] = alive_slice
                    self._active_count = self._active_count + (new_cnt - old_cnt)
                    self._cpu_blk_cnt[blk] = new_cnt
                if new_cnt == 0:
                    self._cpu_blk_min_exp[blk] = torch.inf
                else:
                    rem_vals = val_slice[alive_slice].abs()
                    rem_birth = thresh_buf[u_slice[alive_slice]]
                    self._cpu_blk_min_exp[blk] = ((rem_vals + rem_birth) * 0.999999 - 1e-38).min()
            return
        delta_t = thresh_buf[self.u] - thresh_buf.gather(0, self.last_u.long())
        still_alive = self.active_mask & ~((self.values.abs() - delta_t) <= 0.0)
        self.active_mask.copy_(still_alive)
        self._active_count = still_alive.long().sum()

    def _update_cpu_blocks_dirty(self, unique_b: torch.Tensor, new_val_u: torch.Tensor, thresh_u_f32: float) -> None:
        was_act = self.active_mask[unique_b]
        now_act = new_val_u != 0
        self.active_mask[unique_b] = now_act
        delta_each = now_act.to(torch.int32) - was_act.to(torch.int32)
        self._active_count = self._active_count + delta_each.to(torch.int64).sum()
        blks_u = unique_b >> 10
        self._cpu_blk_cnt.scatter_add_(0, blks_u, delta_each)
        zero_blks = self._cpu_blk_cnt[blks_u] == 0
        if bool(zero_blks.any()):
            self._cpu_blk_min_exp[blks_u[zero_blks]] = torch.inf
        if bool(now_act.any()):
            act_blks = blks_u[now_act]
            lb = (new_val_u[now_act].abs() + thresh_u_f32) * 0.999999 - 1e-38
            self._cpu_blk_min_exp.scatter_reduce_(0, act_blks, lb, reduce='amin', include_self=True)

    @torch.no_grad()
    def step_unique(self, unique: torch.Tensor, grad: torch.Tensor, lr: float, lambda_l1: float):
        if lr < 0 or lambda_l1 < 0:
            raise ValueError('learning rate and L1 coefficient must be nonnegative')
        dev = self.values.device
        unique = unique.to(device=dev, dtype=torch.int64).reshape(-1)
        grad = grad.to(device=dev, dtype=torch.float32).reshape(-1)
        if unique.shape != grad.shape:
            raise ValueError('unique and grad must share 1D length')
        step_thresh = float(lr) * float(lambda_l1)
        new_thresh = self.cum_thresh[-1] + step_thresh
        self._active_cache = None
        self._manual_last_update_bytes = None

        if unique.numel() == 0:
            thresh_buf = self._append_threshold(new_thresh)
            if step_thresh > 0.0:
                self._sweep_untouched_l1_decay(thresh_buf)
            self._last_dirty_count = torch.zeros((), dtype=torch.int64, device=dev)
            self._last_update_bytes_t = 12 * self._active_count
            return

        if lr > 0.0:
            self._ever_nonzero_step = True
        old = self.read(unique)
        if self._head_ref is not None and self._head_ref() is not None:
            s_idx, _ = unique.sort()
            self._record_undo_sorted(s_idx)
        thresh_buf = self._append_threshold(new_thresh)
        new_val = proximal_l1(old - lr * grad, step_thresh)
        self.values.scatter_(0, unique, new_val)
        self.last_u.index_fill_(0, unique, self.u)

        if self._cpu_blk_cnt is not None:
            self._update_cpu_blocks_dirty(unique, new_val, float(thresh_buf[self.u]))
            if step_thresh > 0.0:
                self._sweep_untouched_l1_decay(thresh_buf)
        elif step_thresh == 0.0:
            was_active = self.active_mask.gather(0, unique)
            now_active = new_val != 0
            self.active_mask.scatter_(0, unique, now_active)
            self._active_count = self._active_count + (now_active.long().sum() - was_active.long().sum())
        else:
            self.active_mask.scatter_(0, unique, new_val != 0)
            self._sweep_untouched_l1_decay(thresh_buf)

        self._last_dirty_count = torch.empty((), dtype=torch.int64, device=dev).fill_(unique.numel())
        self._last_update_bytes_t = 16 * self._last_dirty_count + 12 * self._active_count

    @torch.no_grad()
    def step(self, indices: torch.Tensor, gradients: torch.Tensor, lr: float, lambda_l1: float):
        if lr < 0 or lambda_l1 < 0:
            raise ValueError('learning rate and L1 coefficient must be nonnegative')
        dev = self.values.device
        indices = indices.to(device=dev, dtype=torch.int64).reshape(-1)
        gradients = gradients.to(device=dev, dtype=torch.float32).reshape(-1)
        if indices.shape != gradients.shape:
            raise ValueError('indices and gradients must share 1D length')
        if indices.numel() == 0:
            self.step_unique(indices, gradients, lr, lambda_l1)
            return

        step_thresh = float(lr) * float(lambda_l1)
        new_thresh = self.cum_thresh[-1] + step_thresh
        self._active_cache = None
        self._manual_last_update_bytes = None
        if lr > 0.0:
            self._ever_nonzero_step = True

        sorted_b, _ = indices.sort()
        is_first = torch.ones_like(sorted_b, dtype=torch.bool)
        if sorted_b.numel() > 1:
            is_first[1:] = sorted_b[1:] != sorted_b[:-1]

        self._grad_accum.scatter_add_(0, indices, gradients)
        old = self.read(sorted_b)
        tot_grad = self._grad_accum.gather(0, sorted_b)
        self._grad_accum.index_fill_(0, sorted_b, 0.0)

        self._record_undo_sorted(sorted_b)
        thresh_buf = self._append_threshold(new_thresh)
        new_val = proximal_l1(old - lr * tot_grad, step_thresh)
        self.values.scatter_(0, sorted_b, new_val)
        self.last_u.index_fill_(0, sorted_b, self.u)

        if self._cpu_blk_cnt is not None:
            unique_b = sorted_b[is_first]
            new_val_u = new_val[is_first]
            self._update_cpu_blocks_dirty(unique_b, new_val_u, float(thresh_buf[self.u]))
            if step_thresh > 0.0:
                self._sweep_untouched_l1_decay(thresh_buf)
            self._last_dirty_count = torch.empty((), dtype=torch.int64, device=dev).fill_(unique_b.numel())
        elif step_thresh == 0.0:
            was_active = (self.active_mask.gather(0, sorted_b) & is_first).long().sum()
            now_active_each = new_val != 0
            self.active_mask.scatter_(0, sorted_b, now_active_each)
            now_active = (now_active_each & is_first).long().sum()
            self._active_count = self._active_count + (now_active - was_active)
            self._last_dirty_count = is_first.long().sum()
        else:
            self.active_mask.scatter_(0, sorted_b, new_val != 0)
            self._sweep_untouched_l1_decay(thresh_buf)
            self._last_dirty_count = is_first.long().sum()

        self._last_update_bytes_t = 16 * self._last_dirty_count + 12 * self._active_count


@dataclass(frozen=True)
class SumSparse:
    terms: tuple

    def lookup(self, d, previous, current):
        if len(self.terms) >= 2 and all(
            hasattr(t, '_can_share_hash') and t._can_share_hash() for t in self.terms
        ):
            t0 = self.terms[0]
            dev = t0._device if hasattr(t0, '_device') else t0.values.device
            if all((t._device if hasattr(t, '_device') else t.values.device) == dev for t in self.terms[1:]):
                raw_bucket, signs = _hash_raw_and_sign(d, previous, current, dev)
                acc = t0._lookup_from_raw(raw_bucket, signs)
                for term in self.terms[1:]:
                    acc = acc + term._lookup_from_raw(raw_bucket, signs)
                return acc
        values = [term.lookup(d, previous, current) for term in self.terms]
        return sum(values[1:], values[0])


def pack_exact_keys(d, previous, current, vocab_size: int, device=None) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError('vocab_size must be positive')
    target_device = torch.device(device) if device is not None else None
    if target_device is None:
        for item in (d, previous, current):
            if isinstance(item, torch.Tensor):
                target_device = item.device
                break
    d_t, prev_t, curr_t = torch.broadcast_tensors(
        _as_int64_tensor(d, target_device),
        _as_int64_tensor(previous, target_device),
        _as_int64_tensor(current, target_device),
    )
    return (d_t * vocab_size + prev_t) * vocab_size + curr_t


def unpack_exact_keys(keys: torch.Tensor, vocab_size: int):
    if vocab_size <= 0:
        raise ValueError('vocab_size must be positive')
    keys = torch.as_tensor(keys, dtype=torch.int64)
    stride = vocab_size * vocab_size
    return keys // stride, (keys // vocab_size) % vocab_size, keys % vocab_size


@dataclass(frozen=True)
class FrozenExactPairs:
    """Immutable collision-free sorted key-value table for exact observed pairs."""
    keys: torch.Tensor
    values: torch.Tensor
    vocab_size: int
    capacity: int
    overflow_events: int = 0
    dropped_unique_keys: int = 0

    def read_keys(self, packed_keys: torch.Tensor) -> torch.Tensor:
        flat = packed_keys.reshape(-1)
        if self.keys.numel() == 0 or flat.numel() == 0:
            return torch.zeros(packed_keys.shape, dtype=torch.float32, device=self.values.device)
        pos = torch.searchsorted(self.keys, flat)
        safe = pos.clamp_max(len(self.keys) - 1)
        present = (pos < len(self.keys)) & (self.keys[safe] == flat)
        gathered = self.values[safe]
        return torch.where(present, gathered, torch.zeros_like(gathered)).reshape(packed_keys.shape)

    def lookup(self, d, previous, current) -> torch.Tensor:
        packed = pack_exact_keys(d, previous, current, self.vocab_size, device=self.values.device)
        return self.read_keys(packed)

    @property
    def snapshot_bytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()


class ExactObservedPairs(nn.Module):
    """Collision-free explicit `(d, prev, curr)` sparse table with fixed-shape checkpoint buffers.

    `slot_table` maintains a permutation of `0..capacity-1`:
      - `slot_table[:n_occ]` maps sorted `key_table[:n_occ]` to parameter indices in `self.values`;
      - `slot_table[n_occ:]` holds free parameter indices.
    Unseen keys at evaluation time (and keys beyond capacity under `freeze_new`) evaluate to exact zero.
    """
    OVERFLOW_POLICIES = ('freeze_new', 'error')

    def __init__(self, capacity: int = 2**22, vocab_size: int = 248320,
                 overflow_policy: str = 'freeze_new', device=None):
        super().__init__()
        if capacity <= 0 or vocab_size <= 0:
            raise ValueError('capacity and vocab_size must be positive')
        if overflow_policy not in self.OVERFLOW_POLICIES:
            raise ValueError(f'overflow_policy must be one of {self.OVERFLOW_POLICIES}')
        self.capacity = int(capacity)
        self.vocab_size = int(vocab_size)
        self.overflow_policy = overflow_policy
        self.values = nn.Parameter(torch.zeros(self.capacity, dtype=torch.float32, device=device))
        self.register_buffer('key_table', torch.full((self.capacity,), -1, dtype=torch.int64, device=device))
        self.register_buffer('slot_table', torch.arange(self.capacity, dtype=torch.int64, device=device))
        # [n_occupied, peak_occupied, overflow_events, dropped_unique_keys]
        self.register_buffer('meta_counts', torch.zeros(4, dtype=torch.int64, device=device))

    @property
    def num_occupied(self) -> int:
        return int(self.meta_counts[0].item())

    @property
    def peak_occupied(self) -> int:
        return int(self.meta_counts[1].item())

    @property
    def overflow_events(self) -> int:
        return int(self.meta_counts[2].item())

    @property
    def dropped_unique_keys(self) -> int:
        return int(self.meta_counts[3].item())

    @torch.no_grad()
    def _insert_missing(self, flat_keys: torch.Tensor) -> None:
        if flat_keys.numel() == 0:
            return
        unique_q = torch.unique(flat_keys, sorted=True)
        n_occ = self.num_occupied
        active_keys = self.key_table[:n_occ]
        if n_occ > 0:
            pos = torch.searchsorted(active_keys, unique_q)
            safe = pos.clamp_max(n_occ - 1)
            present = (pos < n_occ) & (active_keys[safe] == unique_q)
            new_keys = unique_q[~present]
        else:
            new_keys = unique_q
        if new_keys.numel() == 0:
            return
        avail = self.capacity - n_occ
        if len(new_keys) > avail:
            if self.overflow_policy == 'error':
                raise OverflowError(
                    f'ExactObservedPairs capacity {self.capacity} exceeded by {len(new_keys) - avail} keys'
                )
            dropped = len(new_keys) - avail
            admitted = new_keys[:avail]
            admitted_set = admitted
            if avail > 0:
                pos_f = torch.searchsorted(admitted_set, flat_keys)
                safe_f = pos_f.clamp_max(avail - 1)
                in_adm = (pos_f < avail) & (admitted_set[safe_f] == flat_keys)
            else:
                in_adm = torch.zeros_like(flat_keys, dtype=torch.bool)
            if n_occ > 0:
                pos_e = torch.searchsorted(active_keys, flat_keys)
                safe_e = pos_e.clamp_max(n_occ - 1)
                in_ext = (pos_e < n_occ) & (active_keys[safe_e] == flat_keys)
            else:
                in_ext = torch.zeros_like(flat_keys, dtype=torch.bool)
            self.meta_counts[2] += int((~(in_adm | in_ext)).sum().item())
            self.meta_counts[3] += int(dropped)
            new_keys = admitted
        if new_keys.numel() == 0:
            return
        m = len(new_keys)
        new_slots = self.slot_table[n_occ:n_occ + m].clone()
        self.values.data[new_slots] = 0.0
        merged_keys = torch.cat((active_keys, new_keys))
        merged_slots = torch.cat((self.slot_table[:n_occ], new_slots))
        order = torch.argsort(merged_keys)
        new_total = n_occ + m
        self.key_table[:new_total] = merged_keys[order]
        self.slot_table[:new_total] = merged_slots[order]
        self.meta_counts[0] = new_total
        if new_total > int(self.meta_counts[1].item()):
            self.meta_counts[1] = new_total

    def lookup_packed(self, packed_keys: torch.Tensor, *, insert_missing: bool | None = None) -> torch.Tensor:
        packed_keys = packed_keys.to(device=self.values.device, dtype=torch.int64)
        flat = packed_keys.reshape(-1)
        if flat.numel() == 0:
            return torch.zeros(packed_keys.shape, dtype=torch.float32, device=self.values.device)
        do_insert = self.training if insert_missing is None else bool(insert_missing)
        if do_insert:
            self._insert_missing(flat.detach())
        n_occ = self.num_occupied
        if n_occ == 0:
            return torch.zeros(packed_keys.shape, dtype=torch.float32, device=self.values.device)
        active_keys = self.key_table[:n_occ]
        active_slots = self.slot_table[:n_occ]
        pos = torch.searchsorted(active_keys, flat)
        safe = pos.clamp_max(n_occ - 1)
        present = (pos < n_occ) & (active_keys[safe] == flat)
        gathered = self.values[active_slots[safe]]
        return torch.where(present, gathered, torch.zeros_like(gathered)).reshape(packed_keys.shape)

    def lookup(self, d, previous, current) -> torch.Tensor:
        packed = pack_exact_keys(d, previous, current, self.vocab_size, device=self.values.device)
        return self.lookup_packed(packed)

    @torch.no_grad()
    def proximal_step(self, lr: float, lambda_l1: float, *, reclaim_zeros: bool = True):
        self.values.copy_(proximal_l1(self.values, lr * lambda_l1))
        n_occ = self.num_occupied
        if not reclaim_zeros or n_occ == 0:
            return
        occ_slots = self.slot_table[:n_occ]
        alive = self.values[occ_slots] != 0
        if bool(alive.all()):
            return
        alive_keys = self.key_table[:n_occ][alive].clone()
        alive_slots = occ_slots[alive].clone()
        dead_slots = occ_slots[~alive].clone()
        free_tail = self.slot_table[n_occ:].clone()
        m_alive = len(alive_keys)
        self.key_table[:m_alive] = alive_keys
        self.key_table[m_alive:n_occ] = -1
        self.slot_table[:m_alive] = alive_slots
        self.slot_table[m_alive:] = torch.cat((dead_slots, free_tail))
        self.meta_counts[0] = m_alive

    def snapshot(self) -> FrozenExactPairs:
        n_occ = self.num_occupied
        if n_occ == 0:
            empty_k = torch.empty(0, dtype=torch.int64, device=self.values.device)
            empty_v = torch.empty(0, dtype=torch.float32, device=self.values.device)
            return FrozenExactPairs(empty_k, empty_v, self.vocab_size, self.capacity,
                                    self.overflow_events, self.dropped_unique_keys)
        keys = self.key_table[:n_occ].detach().clone()
        vals = self.values.detach()[self.slot_table[:n_occ]].clone()
        nonzero = vals != 0
        return FrozenExactPairs(keys[nonzero], vals[nonzero], self.vocab_size, self.capacity,
                                self.overflow_events, self.dropped_unique_keys)

    def collision_report(self, keys: list[tuple[int, int, int]]) -> dict:
        counts = Counter(keys)
        n_occ = self.num_occupied
        return {
            'representation': 'exact_observed_pairs',
            'capacity': self.capacity,
            'vocab_size': self.vocab_size,
            'overflow_policy': self.overflow_policy,
            'unique_observed_keys': len(counts),
            'occupied_buckets': n_occ,
            'peak_occupied_entries': self.peak_occupied,
            'load_factor': n_occ / self.capacity,
            'collision_count': 0,
            'frequency_weighted_collision_count': 0,
            'overflow_events': self.overflow_events,
            'dropped_unique_keys': self.dropped_unique_keys,
            'memory_bytes': (self.values.numel() * self.values.element_size()
                             + self.key_table.numel() * self.key_table.element_size()
                             + self.slot_table.numel() * self.slot_table.element_size()),
            'compact_snapshot_bytes': self.snapshot().snapshot_bytes,
        }


@dataclass(frozen=True)
class LazyExactSnapshot:
    """Immutable compact snapshot of active collision-free observed pairs with lazy L1 shrinkage."""
    keys: torch.Tensor
    values: torch.Tensor
    last_u: torch.Tensor
    cum_thresh: torch.Tensor
    u: int
    vocab_size: int
    capacity: int
    overflow_events: int = 0
    dropped_unique_keys: int = 0

    def read(self, packed_keys: torch.Tensor) -> torch.Tensor:
        flat = packed_keys.reshape(-1)
        if self.keys.numel() == 0 or flat.numel() == 0:
            return torch.zeros(packed_keys.shape, dtype=torch.float32, device=self.values.device)
        pos = torch.searchsorted(self.keys, flat)
        safe = pos.clamp_max(len(self.keys) - 1)
        present = (pos < len(self.keys)) & (self.keys[safe] == flat)
        effective = proximal_l1(
            self.values[safe],
            self.cum_thresh[self.u] - self.cum_thresh[self.last_u[safe].long()],
        )
        return torch.where(present, effective, torch.zeros_like(effective)).reshape(packed_keys.shape)

    def lookup(self, d, previous, current) -> torch.Tensor:
        packed = pack_exact_keys(d, previous, current, self.vocab_size, device=self.values.device)
        return self.read(packed)

    @property
    def snapshot_bytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.keys, self.values, self.last_u, self.cum_thresh))


class LazyExactObservedPairs:
    """Request-local collision-free `(d, prev, curr)` table with exact scheduled lazy L1 shrinkage."""
    OVERFLOW_POLICIES = ('freeze_new', 'error')

    def __init__(self, capacity: int = 2**20, vocab_size: int = 248320,
                 overflow_policy: str = 'freeze_new', device=None):
        if capacity <= 0 or vocab_size <= 0:
            raise ValueError('capacity and vocab_size must be positive')
        if overflow_policy not in self.OVERFLOW_POLICIES:
            raise ValueError(f'overflow_policy must be one of {self.OVERFLOW_POLICIES}')
        self.capacity = int(capacity)
        self.vocab_size = int(vocab_size)
        self.overflow_policy = overflow_policy
        self.keys = torch.empty(0, dtype=torch.int64, device=device)
        self.values = torch.empty(0, dtype=torch.float32, device=device)
        self.last_u = torch.empty(0, dtype=torch.int32, device=device)
        self.cum_thresh = [0.0]
        self.u = 0
        self.last_update_bytes = 0
        self.overflow_events = 0
        self.dropped_unique_keys = 0
        self.peak_active = 0

    @property
    def active(self) -> torch.Tensor:
        return self.keys

    def read(self, packed_keys: torch.Tensor) -> torch.Tensor:
        packed_keys = packed_keys.to(device=self.values.device, dtype=torch.int64)
        flat = packed_keys.reshape(-1)
        if self.keys.numel() == 0 or flat.numel() == 0:
            return torch.zeros(packed_keys.shape, dtype=torch.float32, device=self.values.device)
        thresholds = torch.tensor(self.cum_thresh, device=self.values.device, dtype=torch.float32)
        pos = torch.searchsorted(self.keys, flat)
        safe = pos.clamp_max(len(self.keys) - 1)
        present = (pos < len(self.keys)) & (self.keys[safe] == flat)
        effective = proximal_l1(self.values[safe], thresholds[self.u] - thresholds[self.last_u[safe].long()])
        return torch.where(present, effective, torch.zeros_like(effective)).reshape(packed_keys.shape)

    def lookup(self, d, previous, current) -> torch.Tensor:
        packed = pack_exact_keys(d, previous, current, self.vocab_size, device=self.values.device)
        return self.read(packed)

    def snapshot(self) -> LazyExactSnapshot:
        return LazyExactSnapshot(
            self.keys.clone(),
            self.values.clone(),
            self.last_u.clone(),
            torch.tensor(self.cum_thresh, device=self.values.device, dtype=torch.float32),
            self.u,
            self.vocab_size,
            self.capacity,
            self.overflow_events,
            self.dropped_unique_keys,
        )

    @torch.no_grad()
    def step(self, packed_keys: torch.Tensor, gradients: torch.Tensor, lr: float, lambda_l1: float):
        if lr < 0 or lambda_l1 < 0:
            raise ValueError('learning rate and L1 coefficient must be nonnegative')
        device = self.values.device
        packed_keys = packed_keys.to(device=device, dtype=torch.int64).reshape(-1)
        gradients = gradients.to(device=device, dtype=torch.float32).reshape(-1)
        if packed_keys.shape != gradients.shape:
            raise ValueError('packed_keys and gradients must share 1D length')
        # Reclaim entries that have already decayed to zero at current u before checking capacity.
        if self.keys.numel() > 0:
            alive_now = self.read(self.keys) != 0
            if not bool(alive_now.all()):
                self.keys = self.keys[alive_now]
                self.values = self.values[alive_now]
                self.last_u = self.last_u[alive_now]
        if packed_keys.numel() == 0:
            self.u += 1
            self.cum_thresh.append(self.cum_thresh[-1] + lr * lambda_l1)
            if self.keys.numel() > 0:
                alive_after = self.read(self.keys) != 0
                self.keys = self.keys[alive_after]
                self.values = self.values[alive_after]
                self.last_u = self.last_u[alive_after]
            self.last_update_bytes = self.keys.numel() * 16
            return
        unique, inverse = torch.unique(packed_keys, sorted=True, return_inverse=True)
        grad = torch.zeros(len(unique), device=device, dtype=torch.float32)
        grad.scatter_add_(0, inverse, gradients)
        if self.keys.numel() > 0:
            pos = torch.searchsorted(self.keys, unique)
            safe = pos.clamp_max(len(self.keys) - 1)
            is_existing = (pos < len(self.keys)) & (self.keys[safe] == unique)
        else:
            is_existing = torch.zeros(len(unique), dtype=torch.bool, device=device)
        new_mask = ~is_existing
        new_keys = unique[new_mask]
        new_grads = grad[new_mask]
        avail = self.capacity - len(self.keys)
        if len(new_keys) > avail:
            if self.overflow_policy == 'error':
                raise OverflowError(
                    f'LazyExactObservedPairs capacity {self.capacity} exceeded by {len(new_keys) - avail} keys'
                )
            dropped = len(new_keys) - avail
            self.dropped_unique_keys += int(dropped)
            self.overflow_events += 1
            if avail > 0:
                # Deterministic priority: largest |grad| first, lowest packed key tie-break via stable sort.
                keep_rank = torch.argsort(-new_grads.abs(), stable=True)[:avail]
                keep_order = torch.argsort(new_keys[keep_rank])
                chosen = keep_rank[keep_order]
                admitted_new_keys = new_keys[chosen]
                admitted_new_grads = new_grads[chosen]
            else:
                admitted_new_keys = torch.empty(0, dtype=torch.int64, device=device)
                admitted_new_grads = torch.empty(0, dtype=torch.float32, device=device)
            admitted_keys = torch.cat((unique[is_existing], admitted_new_keys))
            admitted_grads = torch.cat((grad[is_existing], admitted_new_grads))
            order = torch.argsort(admitted_keys)
            admitted_keys = admitted_keys[order]
            admitted_grads = admitted_grads[order]
        else:
            admitted_keys = unique
            admitted_grads = grad
        old = self.read(admitted_keys)
        if self.keys.numel() > 0 and admitted_keys.numel() > 0:
            pos_u = torch.searchsorted(admitted_keys, self.keys)
            safe_u = pos_u.clamp_max(len(admitted_keys) - 1)
            touched = (pos_u < len(admitted_keys)) & (admitted_keys[safe_u] == self.keys)
            untouched_keys = self.keys[~touched]
            untouched_vals = self.values[~touched]
            untouched_u = self.last_u[~touched]
        else:
            untouched_keys = self.keys
            untouched_vals = self.values
            untouched_u = self.last_u
        self.u += 1
        self.cum_thresh.append(self.cum_thresh[-1] + lr * lambda_l1)
        updated_vals = proximal_l1(old - lr * admitted_grads, lr * lambda_l1)
        updated_u = torch.full((len(admitted_keys),), self.u, dtype=torch.int32, device=device)
        merged_keys = torch.cat((untouched_keys, admitted_keys))
        merged_vals = torch.cat((untouched_vals, updated_vals))
        merged_u = torch.cat((untouched_u, updated_u))
        if merged_keys.numel() > 1:
            sort_idx = torch.argsort(merged_keys)
            merged_keys = merged_keys[sort_idx]
            merged_vals = merged_vals[sort_idx]
            merged_u = merged_u[sort_idx]
        self.keys, self.values, self.last_u = merged_keys, merged_vals, merged_u
        if self.keys.numel() > 0:
            alive_final = self.read(self.keys) != 0
            self.keys = self.keys[alive_final]
            self.values = self.values[alive_final]
            self.last_u = self.last_u[alive_final]
        self.peak_active = max(self.peak_active, len(self.keys))
        self.last_update_bytes = admitted_keys.numel() * (8 + 4 + 4 + 4) + self.keys.numel() * 16
