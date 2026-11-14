"""Exact ratio rejection verification with full-vocabulary residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch

from .sampler import SamplingWarper, compact_probabilities, draw, probabilities, probability_dtype


@dataclass(frozen=True)
class CompactProposal:
    """Compact `[K, top_k]` proposal distributions avoiding `[K, V]` materialization."""

    slot_indices: torch.Tensor
    slot_probs: torch.Tensor
    q_selected: torch.Tensor
    vocab_size: int

    def to_dense(self) -> torch.Tensor:
        K = self.slot_probs.shape[0]
        out = torch.zeros(
            (K, self.vocab_size),
            dtype=self.slot_probs.dtype,
            device=self.slot_probs.device,
        )
        if K > 0:
            out.scatter_(-1, self.slot_indices, self.slot_probs)
        return out


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
    proposal_probs: torch.Tensor | Sequence[torch.Tensor] | CompactProposal | None,
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
    is_compact_q = isinstance(proposal_probs, CompactProposal)
    if is_compact_q:
        if (
            proposal_probs.vocab_size != V
            or proposal_probs.slot_probs.ndim != 2
            or proposal_probs.slot_probs.shape[0] != K
            or proposal_probs.slot_indices.shape != proposal_probs.slot_probs.shape
            or proposal_probs.slot_indices.dtype != torch.long
            or proposal_probs.q_selected.shape != (K,)
            or proposal_probs.slot_indices.device != proposal_probs.slot_probs.device
            or proposal_probs.q_selected.device != proposal_probs.slot_probs.device
        ):
            raise ValueError("CompactProposal shapes/dtypes/devices must match [K, top_k] and vocab_size=V")
        q = None
        q_selected = probability_dtype(proposal_probs.q_selected)
        q_device = proposal_probs.slot_probs.device
    else:
        q = proposal_probs if isinstance(proposal_probs, torch.Tensor) else torch.stack(list(proposal_probs))
        q = probability_dtype(q)
        if q.shape != (K, V):
            raise ValueError("proposal_probs must have shape [K,V]")
        q_device = q.device

    if probs_fn is None:
        p_sub, p_idx = compact_probabilities(target_logits, temperature, warper)
    else:
        owner = getattr(probs_fn, "__self__", None)
        if owner is not None and hasattr(owner, "compact_probabilities") and getattr(probs_fn, "__name__", None) == "probabilities":
            p_sub, p_idx = owner.compact_probabilities(target_logits, temperature, warper)
        else:
            p_sub, p_idx = probs_fn(target_logits, temperature, warper), None

    if draft_tensor is not None and draft_tensor.device == q_device and draft_tensor.dtype == torch.long:
        token_indices = draft_tensor[:, None]
    else:
        token_indices = torch.tensor(tokens, device=q_device, dtype=torch.long)[:, None]
    if not is_compact_q:
        q_selected = q.gather(-1, token_indices)[:, 0]
    if K == 0:
        p_selected = p_sub.new_empty((0,))
    elif p_idx is None:
        p_tok_idx = token_indices if token_indices.device == p_sub.device else token_indices.to(p_sub.device)
        p_selected = p_sub[:K].gather(-1, p_tok_idx)[:, 0]
    else:
        p_tok_idx = token_indices if token_indices.device == p_idx.device else token_indices.to(p_idx.device)
        match = p_idx[:K] == p_tok_idx
        has_match = match.any(dim=-1)
        match_col = match.int().argmax(dim=-1, keepdim=True)
        p_selected = torch.where(
            has_match,
            p_sub[:K].gather(-1, match_col).squeeze(-1),
            torch.zeros_like(p_sub[:K, 0]),
        )

    def _materialize_p_row(row_idx: int) -> torch.Tensor:
        if p_idx is None:
            return p_sub[row_idx]
        return torch.zeros(V, dtype=p_sub.dtype, device=p_sub.device).scatter_(0, p_idx[row_idx], p_sub[row_idx])

    # Transfer both [K] vectors in one host sync and validate before touching RNG.
    qp_cpu = torch.stack((q_selected, p_selected)).tolist()
    proposal_token_probs, target_token_probs = qp_cpu[0], qp_cpu[1]
    if not all(x > 0 and math.isfinite(x) for x in proposal_token_probs):
        raise ValueError("a realized draft token must have positive finite proposal mass")
    # Draw every slot's acceptance uniform before any residual draw, matching
    # the inspected upstream ratio sampler's seeded RNG consumption.
    uniforms = torch.rand((K,), device=p_sub.device, dtype=p_sub.dtype, generator=generator)
    # u*q < P is algebraically u < min(1,P/q), matching upstream.
    decisions_all = (uniforms * q_selected < p_selected).tolist()
    for i, accepted in enumerate(decisions_all):
        if not accepted:
            pi = _materialize_p_row(i)
            if is_compact_q:
                qi_idx = proposal_probs.slot_indices[i]
                qi_probs = probability_dtype(proposal_probs.slot_probs[i])
                qi = torch.zeros(V, dtype=qi_probs.dtype, device=qi_probs.device).scatter_(0, qi_idx, qi_probs)
            else:
                qi = q[i]
            if not bool((pi - qi).clamp_min(0).sum() > 0):
                p64 = pi.double()
                q64 = qi.double()
                p64_sum, q64_sum = p64.sum(), q64.sum()
                if (bool(torch.isfinite(p64_sum)) and bool(torch.isfinite(q64_sum))
                        and bool((p64 >= 0).all()) and bool((q64 >= 0).all())
                        and abs(float(p64_sum) - 1.0) <= 1e-4 and abs(float(q64_sum) - 1.0) <= 1e-4):
                    p64 = p64 / p64_sum
                    q64 = q64 / q64_sum
                    u64 = float(uniforms[i].item())
                    if u64 * float(q64[tokens[i]].item()) < float(p64[tokens[i]].item()) or not bool((p64 - q64).clamp_min(0).sum() > 0):
                        decisions_all[i] = True
                        continue
                    replacement = draw(residual_distribution(p64, q64).to(p_sub.dtype), generator)
                    return VerificationResult(tokens[:i] + [replacement], i, i, replacement, None,
                                              decisions_all[:i + 1], proposal_token_probs, target_token_probs)
            replacement = draw(residual_distribution(pi, qi), generator)
            return VerificationResult(tokens[:i] + [replacement], i, i, replacement, None,
                                      decisions_all[:i + 1], proposal_token_probs, target_token_probs)
    bonus = draw(_materialize_p_row(K), generator)
    return VerificationResult(tokens + [bonus], K, K, None, bonus, decisions_all,
                              proposal_token_probs, target_token_probs)
