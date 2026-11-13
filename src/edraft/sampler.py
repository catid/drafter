"""Sampling primitives shared by proposals and target verification.

Training must not call this module: its distributions are unwarped T=1.
Inference warper order is top-k, top-p, then min-p. Top-k/top-p use the
inspected pinned DFlash convention for seeded upstream reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class SamplingWarper:
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0

    def __post_init__(self):
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def probability_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """FP64 toy calculations stay FP64; inference never uses bf16 probabilities."""
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def mask_scores(scores: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
    """Mask before warping; never mutate the score or immutable candidate tensors."""
    if scores.ndim != 1 or candidate_ids.ndim != 1 or candidate_ids.numel() == 0:
        raise ValueError("expected one score vector and nonempty candidate IDs")
    masked = torch.full_like(scores, -torch.inf)
    return masked.scatter(0, candidate_ids, scores.gather(0, candidate_ids))


def apply_warper(scores: torch.Tensor, warper: SamplingWarper = SamplingWarper()) -> torch.Tensor:
    """Apply truncations to *already corrected and temperature-scaled* scores.

    Works on [..., V]. Top-k keeps exactly k IDs, matching pinned DFlash.
    Top-p keeps each descending token whose preceding cumulative mass is <p.
    """
    scores = probability_dtype(scores)
    if 0 < warper.top_k < scores.shape[-1]:
        selected, indices = torch.topk(scores, warper.top_k, dim=-1)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, indices, selected)
    if warper.top_p < 1:
        probs = torch.softmax(scores, dim=-1)
        sorted_probs, indices = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < warper.top_p
        keep = torch.zeros_like(keep).scatter(-1, indices, keep)
        scores = scores.masked_fill(~keep, -torch.inf)
    if warper.min_p > 0:
        # p(a) < min_p * max p is equivalent in log space, with no softmax.
        cutoff = scores.amax(dim=-1, keepdim=True) + math.log(warper.min_p)
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    return scores


def probabilities(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("sampling probabilities require finite temperature > 0")
    scores = probability_dtype(scores) / temperature
    V = scores.shape[-1]
    indices = None
    if 0 < warper.top_k < V:
        scores, indices = torch.topk(scores, warper.top_k, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    if warper.top_p < 1:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < warper.top_p
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if warper.min_p > 0:
        keep = probs >= warper.min_p * probs.amax(dim=-1, keepdim=True)
        probs = probs * keep
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if indices is not None:
        full_shape = (*probs.shape[:-1], V)
        probs = torch.zeros(full_shape, dtype=probs.dtype, device=probs.device).scatter(-1, indices, probs)
    return probs


def draw(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """Draw one full-vocabulary categorical token."""
    if probs.ndim != 1:
        raise ValueError("draw requires a single probability vector")
    return int(torch.multinomial(probs, 1, generator=generator).item())


def sample_token(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
) -> tuple[int, torch.Tensor | None]:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("sample_token requires a nonempty vocabulary vector")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if temperature == 0:
        # torch.argmax returns the first maximum: lowest full-vocabulary ID.
        # Deliberately no division, warper, sampling softmax, or random draw.
        return int(torch.argmax(scores).item()), None
    probs = probabilities(scores, temperature, warper)
    return draw(probs, generator), probs


sample_score = sample_token
