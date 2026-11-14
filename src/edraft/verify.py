"""Exact ratio rejection verification with full-vocabulary residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch

from .sampler import SamplingWarper, draw, probabilities, probability_dtype


@dataclass(frozen=True)
class VerificationResult:
    emitted_tokens: list[int]
    accepted_length: int
    first_rejection: int
    replacement_token: int | None
    bonus_token: int | None
    decisions: list[bool]
    proposal_token_probs: list[float] | None
    target_token_probs: list[float] | None
    verification_rule: str = "ratio"


def residual_distribution(target_probs: torch.Tensor, proposal_probs: torch.Tensor) -> torch.Tensor:
    """Positive residual normalized over all V, including mass outside proposals."""
    target_probs = probability_dtype(target_probs)
    proposal_probs = probability_dtype(proposal_probs)
    residual = (target_probs - proposal_probs).clamp_min(0)
    total = residual.sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise RuntimeError("rejection has no positive finite residual mass; check P/q precision")
    return residual / total


def verify_block(
    draft_tokens: Sequence[int] | torch.Tensor,
    proposal_probs: torch.Tensor | Sequence[torch.Tensor] | None,
    target_logits: torch.Tensor,
    temperature: float,
    warper: SamplingWarper = SamplingWarper(),
    generator: torch.Generator | None = None,
    verification_rule: str = "ratio",
    *,
    probs_fn=None,
) -> VerificationResult:
    """Verify K proposals against K+1 target conditionals on their realized canvas.

    target_logits[i] conditions on prefix + draft_tokens[:i]; the last row
    supplies the bonus distribution. Accepted length excludes replacement/bonus.
    Post-rejection rows are valid off-policy teacher data, never emitted.
    """
    if verification_rule != "ratio":
        raise ValueError("M0–M4 stochastic verification requires verification_rule='ratio'")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    draft_tensor = draft_tokens if isinstance(draft_tokens, torch.Tensor) else None
    tokens = draft_tensor.tolist() if draft_tensor is not None else [int(x) for x in draft_tokens]
    K = len(tokens)
    if target_logits.ndim != 2 or target_logits.shape[0] != K + 1:
        raise ValueError("target_logits must have shape [K+1, V], including bonus")
    V = target_logits.shape[1]
    if any(token < 0 or token >= V for token in tokens):
        raise ValueError("draft token outside vocabulary")
    if temperature == 0:
        target_ids = target_logits.argmax(-1).tolist()
        for i, token in enumerate(tokens):
            if token != target_ids[i]:
                return VerificationResult(tokens[:i] + [target_ids[i]], i, i, target_ids[i],
                                          None, [True] * i + [False], None, None, "greedy")
        return VerificationResult(tokens + [target_ids[K]], K, K, None, target_ids[K],
                                  [True] * K, None, None, "greedy")

    if proposal_probs is None:
        raise ValueError("stochastic ratio verification requires full proposal distributions")
    q = proposal_probs if isinstance(proposal_probs, torch.Tensor) else torch.stack(list(proposal_probs))
    q = probability_dtype(q)
    if q.shape != (K, V):
        raise ValueError("proposal_probs must have shape [K,V]")
    P = (probabilities if probs_fn is None else probs_fn)(target_logits, temperature, warper)
    if draft_tensor is not None and draft_tensor.device == q.device and draft_tensor.dtype == torch.long:
        token_indices = draft_tensor[:, None]
    else:
        token_indices = torch.tensor(tokens, device=q.device, dtype=torch.long)[:, None]
    q_selected = q.gather(-1, token_indices)[:, 0]
    p_selected = P[:K].gather(-1, token_indices)[:, 0]
    # Transfer both [K] vectors in one host sync and validate before touching RNG.
    qp_cpu = torch.stack((q_selected, p_selected)).tolist()
    proposal_token_probs, target_token_probs = qp_cpu[0], qp_cpu[1]
    if not all(x > 0 and math.isfinite(x) for x in proposal_token_probs):
        raise ValueError("a realized draft token must have positive finite proposal mass")
    # Draw every slot's acceptance uniform before any residual draw, matching
    # the inspected upstream ratio sampler's seeded RNG consumption.
    uniforms = torch.rand((K,), device=P.device, dtype=P.dtype, generator=generator)
    # u*q < P is algebraically u < min(1,P/q), matching upstream.
    decisions_all = (uniforms * q_selected < p_selected).tolist()
    for i, accepted in enumerate(decisions_all):
        if not accepted:
            replacement = draw(residual_distribution(P[i], q[i]), generator)
            return VerificationResult(tokens[:i] + [replacement], i, i, replacement, None,
                                      decisions_all[:i + 1], proposal_token_probs, target_token_probs)
    bonus = draw(P[K], generator)
    return VerificationResult(tokens + [bonus], K, K, None, bonus, decisions_all,
                              proposal_token_probs, target_token_probs)
