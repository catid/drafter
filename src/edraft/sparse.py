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
