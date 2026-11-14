"""Deterministic signed hashing and read-only lazy proximal snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import torch
from torch import nn


def hash_keys(d, previous, current, capacity: int, device=None):
    if capacity <= 0:
        raise ValueError('hash capacity must be positive')
    d, previous, current = torch.broadcast_tensors(
        torch.as_tensor(d, dtype=torch.int64, device=device),
        torch.as_tensor(previous, dtype=torch.int64, device=device),
        torch.as_tensor(current, dtype=torch.int64, device=device))
    # Separate prime-modular hashes; no Python randomized hash or platform RNG.
    prime = 2147483647
    mixed = (d * 73856093 + previous * 19349663 + current * 83492791 + 104729) % prime
    bucket = ((mixed * 48271 + 1) % prime) % capacity
    sign_hash = (d * 15485863 + previous * 32452843 + current * 49979687 + 67867967) % prime
    sign = 1 - 2 * (((sign_hash * 40699 + 17) % prime) % 2)
    return bucket, sign.to(torch.float32)


def proximal_l1(values: torch.Tensor, threshold) -> torch.Tensor:
    return values.sign() * (values.abs() - threshold).clamp_min(0)


@dataclass(frozen=True)
class FrozenHash:
    values: torch.Tensor
    capacity: int

    def lookup(self, d, previous, current):
        buckets, signs = hash_keys(d, previous, current, self.capacity, self.values.device)
        return self.values[buckets] * signs


class SignedHash(nn.Module):
    def __init__(self, capacity: int = 2**24, device=None):
        super().__init__()
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.values = nn.Parameter(torch.zeros(capacity, dtype=torch.float32, device=device))

    def lookup(self, d, previous, current):
        buckets, signs = hash_keys(d, previous, current, self.capacity, self.values.device)
        return self.values[buckets] * signs

    def snapshot(self):
        # Offline fit stops before decoding; capture once when a request is created.
        return FrozenHash(self.values.detach().clone(), self.capacity)

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


@dataclass(frozen=True)
class LazyHashSnapshot:
    """Compact immutable active-bucket copy; unrepresented buckets are exactly zero.

    The state owns one fixed-capacity GPU table. Snapshots clone only its active
    index/value/timestamp list (the allowed active-bucket representation), never
    the capacity-sized arrays. This cost is exposed in snapshot_bytes.
    """
    indices: torch.Tensor
    values: torch.Tensor
    last_u: torch.Tensor
    cum_thresh: torch.Tensor
    u: int
    capacity: int

    def read(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1)
        if self.indices.numel() == 0:
            return torch.zeros_like(indices, dtype=torch.float32)
        pos = torch.searchsorted(self.indices, flat)
        safe = pos.clamp_max(len(self.indices) - 1)
        present = (pos < len(self.indices)) & (self.indices[safe] == flat)
        effective = proximal_l1(self.values[safe], self.cum_thresh[self.u] - self.cum_thresh[self.last_u[safe].long()])
        return torch.where(present, effective, torch.zeros_like(effective)).reshape(indices.shape)

    def lookup(self, d, previous, current):
        indices, signs = hash_keys(d, previous, current, self.capacity, self.values.device)
        return self.read(indices) * signs

    @property
    def snapshot_bytes(self):
        return sum(x.numel() * x.element_size() for x in (self.indices, self.values, self.last_u, self.cum_thresh))


class LazySignedHash:
    """Fixed request table; dirty-only updates with exact scheduled lazy shrinkage."""
    def __init__(self, capacity: int = 2**20, device=None):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.last_u = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.active = torch.empty(0, dtype=torch.int64, device=device)
        self.cum_thresh = [0.0]
        self.u = 0
        self.last_update_bytes = 0

    def read(self, indices: torch.Tensor):
        thresholds = torch.tensor(self.cum_thresh, device=self.values.device, dtype=torch.float32)
        return proximal_l1(self.values[indices], thresholds[self.u] - thresholds[self.last_u[indices].long()])

    def snapshot(self) -> LazyHashSnapshot:
        # Compact read-only snapshot, independent of future table mutation.
        return LazyHashSnapshot(self.active.clone(), self.values[self.active].clone(),
                                self.last_u[self.active].clone(),
                                torch.tensor(self.cum_thresh, device=self.values.device, dtype=torch.float32),
                                self.u, self.capacity)

    @torch.no_grad()
    def step(self, indices: torch.Tensor, gradients: torch.Tensor, lr: float, lambda_l1: float):
        if lr < 0 or lambda_l1 < 0:
            raise ValueError('learning rate and L1 coefficient must be nonnegative')
        indices = indices.to(device=self.values.device, dtype=torch.int64).reshape(-1)
        gradients = gradients.to(device=self.values.device, dtype=torch.float32).reshape(-1)
        unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
        grad = torch.zeros(len(unique), device=self.values.device, dtype=torch.float32)
        grad.scatter_add_(0, inverse, gradients)
        old = self.read(unique)
        self.u += 1
        self.cum_thresh.append(self.cum_thresh[-1] + lr * lambda_l1)
        self.values[unique] = proximal_l1(old - lr * grad, lr * lambda_l1)
        self.last_u[unique] = self.u
        self.active = torch.unique(torch.cat((self.active, unique)), sorted=True)
        # Compact dead entries without modifying their values/timestamps.
        self.active = self.active[self.read(self.active) != 0]
        self.last_update_bytes = unique.numel() * (4 + 4 + 4 + 4) + self.active.numel() * 12


@dataclass(frozen=True)
class SumSparse:
    terms: tuple

    def lookup(self, d, previous, current):
        values = [term.lookup(d, previous, current) for term in self.terms]
        return sum(values[1:], values[0])


def pack_exact_keys(d, previous, current, vocab_size: int, device=None) -> torch.Tensor:
    if vocab_size <= 0:
        raise ValueError('vocab_size must be positive')
    d_t, prev_t, curr_t = torch.broadcast_tensors(
        torch.as_tensor(d, dtype=torch.int64, device=device),
        torch.as_tensor(previous, dtype=torch.int64, device=device),
        torch.as_tensor(current, dtype=torch.int64, device=device),
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
