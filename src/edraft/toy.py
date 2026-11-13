"""Download-free CPU target and fixed-feature parallel drafter for smoke tests.

This is synthetic data, never a substitute for reference milestone evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn

from .sampler import SamplingWarper, sample_token


def _normalized(hidden: torch.Tensor) -> torch.Tensor:
    return hidden / (hidden.square().mean(-1, keepdim=True) + 1e-6).sqrt()


class ToyTarget(nn.Module):
    """Frozen second-order target; explicit prefix semantics make rollback testable."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 12, seed: int = 2026,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if vocab_size < 2 or hidden_size < 2:
            raise ValueError("toy vocabulary and hidden size must be >= 2")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("embedding", torch.randn(vocab_size, hidden_size, generator=generator,
                                                       dtype=dtype))
        self.register_buffer("start_hidden", torch.randn(hidden_size, generator=generator, dtype=dtype))
        self.register_buffer("frequencies", torch.linspace(0.13, 0.83, hidden_size, dtype=dtype))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=True, dtype=dtype)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab_size, hidden_size, generator=generator,
                                                 dtype=dtype) / math.sqrt(hidden_size))
            self.lm_head.bias.copy_(torch.randn(vocab_size, generator=generator, dtype=dtype) * 0.1)
        self.requires_grad_(False)
        self.eval()

    def hidden(self, prefix: Sequence[int]) -> torch.Tensor:
        prefix = list(prefix)
        h = self.embedding[int(prefix[-1])] if prefix else self.start_hidden
        if len(prefix) >= 2:
            h = h + 0.35 * self.embedding[int(prefix[-2])]
        h = h + 0.12 * torch.sin(self.frequencies * len(prefix))
        return _normalized(h)

    @torch.no_grad()
    def logits(self, prefix: Sequence[int]) -> torch.Tensor:
        return self.lm_head(self.hidden(prefix))

    @torch.no_grad()
    def verify_logits(self, prefix: Sequence[int], draft_tokens: Sequence[int]) -> torch.Tensor:
        canvas = list(prefix)
        rows = []
        for token in draft_tokens:
            rows.append(self.logits(canvas))
            canvas.append(int(token))
        rows.append(self.logits(canvas))
        return torch.stack(rows)

    @torch.no_grad()
    def generate(self, prefix: Sequence[int], max_new_tokens: int, temperature: float = 0,
                 warper: SamplingWarper = SamplingWarper(),
                 generator: torch.Generator | None = None) -> list[int]:
        context = list(prefix)
        output = []
        for _ in range(max_new_tokens):
            token, _ = sample_token(self.logits(context), temperature, warper, generator)
            output.append(token)
            context.append(token)
        return output


@dataclass(frozen=True)
class ToyDraft:
    logits: torch.Tensor
    h_raw: torch.Tensor
    physical_to_logical: tuple[int, ...]


class ToyParallelDrafter(nn.Module):
    """One batched hidden/head operation; no reads of realized proposal tokens.

    Physical slot zero is the clean anchor (prefix[-1]); returned logical slots
    correspond to physical indices 1..K. h_raw is post-RMS normalization and
    immediately before draft_lm_head, as required by trace reconstruction.
    """

    def __init__(self, target: ToyTarget | None = None, seed: int = 2027):
        super().__init__()
        target = target if target is not None else ToyTarget()
        self.vocab_size = target.vocab_size
        self.hidden_size = target.hidden_size
        self.register_buffer("embedding", target.embedding.detach().clone())
        self.register_buffer("start_hidden", target.start_hidden.detach().clone())
        self.register_buffer("frequencies", target.frequencies.detach().clone())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("slot_direction", torch.randn(self.hidden_size, generator=generator,
                                                           dtype=target.embedding.dtype) * 0.2)
        self.draft_lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=True,
                                       dtype=target.embedding.dtype)
        self.draft_lm_head.load_state_dict(target.lm_head.state_dict())
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def draft(self, prefix: Sequence[int], K: int) -> ToyDraft:
        if K < 1:
            raise ValueError("K must be >= 1 logical unknown positions")
        prefix = list(prefix)
        anchor = self.embedding[int(prefix[-1])] if prefix else self.start_hidden
        if len(prefix) >= 2:
            anchor = anchor + 0.35 * self.embedding[int(prefix[-2])]
        positions = torch.arange(K, dtype=anchor.dtype, device=anchor.device)
        hidden = anchor.unsqueeze(0) + 0.12 * torch.sin(
            (positions[:, None] + len(prefix)) * self.frequencies[None, :])
        hidden = hidden + positions[:, None] * self.slot_direction[None, :]
        h_raw = _normalized(hidden)
        return ToyDraft(self.draft_lm_head(h_raw), h_raw, tuple(range(1, K + 1)))

    def head_metadata(self) -> dict:
        return {"artifact_id": "builtin-toy-linear-head", "revision": "synthetic-v1",
                "dtype": str(self.draft_lm_head.weight.dtype), "bias": True,
                "tied_weights": False, "hook": "post _normalized, pre draft_lm_head",
                "weight_shape": list(self.draft_lm_head.weight.shape)}


class ToyCache:
    """Explicit test cache with the same crop-after-partial-acceptance invariant."""

    def __init__(self, target: ToyTarget, prefix: Sequence[int] = ()):
        self.target = target
        self.tokens = list(prefix)

    def append(self, tokens: Sequence[int]) -> torch.Tensor:
        logits = self.target.verify_logits(self.tokens, tokens)
        self.tokens.extend(int(x) for x in tokens)
        return logits

    def crop(self, length: int) -> None:
        if not 0 <= length <= len(self.tokens):
            raise ValueError("invalid cache crop length")
        del self.tokens[length:]

    def next_logits(self) -> torch.Tensor:
        return self.target.logits(self.tokens)
