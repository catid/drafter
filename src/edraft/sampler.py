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


def compact_probabilities(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return `(probs_sub, indices)` on retained support without `[..., V]` scatter.

    Preserves exact 0-ULP tie-breaking parity with `09a5469` and pinned upstream
    DFlash (`probs.sort(descending=True)` + scatter on the compact `[..., top_k]`
    support). When `top_k` is inactive (`0` or `>= V`), returns `(probs_full, None)`.
    """
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
    return probs, indices


def warp_top_scaled_scores(
    scaled_top_scores: torch.Tensor,
    warper: SamplingWarper,
) -> torch.Tensor:
    """Apply `softmax` + `top_p` + `min_p` to pre-selected `[..., top_k]` scaled scores."""
    probs = torch.softmax(scaled_top_scores, dim=-1)
    if warper.top_p < 1:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < warper.top_p
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if warper.min_p > 0:
        keep = probs >= warper.min_p * probs.amax(dim=-1, keepdim=True)
        probs = probs * keep
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def sample_from_top_scaled(
    scaled_top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    warper: SamplingWarper,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-sync categorical draw from pre-selected `[top_k]` scaled scores and indices.

    Returns `(tok_t, probs_sorted, idx_sorted, q_selected)` with ascending-ID CDF order.
    """
    probs_sub = warp_top_scaled_scores(scaled_top_scores, warper)
    idx_sorted, perm = top_indices.sort(dim=-1)
    probs_sorted = probs_sub.gather(-1, perm)
    choice = torch.multinomial(probs_sorted, 1, generator=generator)[0]
    return idx_sorted[choice], probs_sorted, idx_sorted, probs_sorted[choice]


def probabilities(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
) -> torch.Tensor:
    probs, indices = compact_probabilities(scores, temperature, warper)
    if indices is not None:
        V = scores.shape[-1]
        full_shape = (*probs.shape[:-1], V)
        probs = torch.zeros(full_shape, dtype=probs.dtype, device=probs.device).scatter(-1, indices, probs)
    return probs


def draw(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """Draw one full-vocabulary categorical token."""
    if probs.ndim != 1:
        raise ValueError("draw requires a single probability vector")
    return int(torch.multinomial(probs, 1, generator=generator).item())


def draw_tensor(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one full-vocabulary categorical token without host synchronization."""
    if probs.ndim != 1:
        raise ValueError("draw_tensor requires a single probability vector")
    return torch.multinomial(probs, 1, generator=generator)[0]


def sample_token_compact_with_prob(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
    *,
    compact_probs_fn=None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Sync-free categorical draw returning `(tok_t, probs_compact, indices_compact, q_selected)`."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("sample_token_compact requires a nonempty vocabulary vector")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if temperature == 0:
        return torch.argmax(scores), None, None, None
    probs_sub, indices = (compact_probabilities if compact_probs_fn is None else compact_probs_fn)(
        scores, temperature, warper
    )
    if indices is None:
        tok_t = draw_tensor(probs_sub, generator)
        return tok_t, probs_sub, None, probs_sub[tok_t]
    idx_sorted, perm = indices.sort(dim=-1)
    probs_sorted = probs_sub.gather(-1, perm)
    choice = torch.multinomial(probs_sorted, 1, generator=generator)[0]
    return idx_sorted[choice], probs_sorted, idx_sorted, probs_sorted[choice]


def sample_token_compact(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
    *,
    compact_probs_fn=None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Sync-free categorical draw on compact warped support with ascending-ID CDF order.

    Returns `(tok_tensor, probs_compact, indices_compact)`:
    - `temperature == 0`: `(torch.argmax(scores), None, None)` with no division,
      warper, softmax, or RNG consumption.
    - `0 < warper.top_k < V`: sorts retained `indices` ascending before `multinomial`
      so CDF bucket order matches ascending vocabulary IDs `0..V-1`, returning
      `(tok_t, probs_sorted, idx_sorted)` of length `warper.top_k`.
    - Full-vocabulary warper (`indices is None`): returns `(draw_tensor(probs), probs, None)`.
    """
    tok_t, probs_c, idx_c, _ = sample_token_compact_with_prob(
        scores, temperature, warper, generator, compact_probs_fn=compact_probs_fn
    )
    return tok_t, probs_c, idx_c


def sample_token(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
    *,
    probs_fn=None,
) -> tuple[int, torch.Tensor | None]:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("sample_token requires a nonempty vocabulary vector")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if temperature == 0:
        # torch.argmax returns the first maximum: lowest full-vocabulary ID.
        # Deliberately no division, warper, sampling softmax, or random draw.
        return int(torch.argmax(scores).item()), None
    probs = (probabilities if probs_fn is None else probs_fn)(scores, temperature, warper)
    return draw(probs, generator), probs


def sample_token_tensor(
    scores: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
    *,
    probs_fn=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("sample_token_tensor requires a nonempty vocabulary vector")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if temperature == 0:
        # torch.argmax returns the first maximum: lowest full-vocabulary ID.
        # Deliberately no division, warper, sampling softmax, or random draw.
        return torch.argmax(scores), None
    probs = (probabilities if probs_fn is None else probs_fn)(scores, temperature, warper)
    return draw_tensor(probs, generator), probs


sample_score = sample_token
