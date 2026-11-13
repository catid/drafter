"""Inspected Hugging Face backend for the original Qwen3 DFlash checkpoint.

Artifact downloads belong to the explicit bootstrap command. Model loading here is
strictly local, even when a hub model identifier rather than a snapshot path is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def make_cache(config: Any):
    from transformers import DynamicCache

    cache = DynamicCache(config=config)
    # Transformers 5.15 needs recording for reversible fixed-size cache layers.
    cache.activate_past_recording()
    return cache


def crop_cache(cache: Any, length: int) -> None:
    """Keep exactly ``length`` tokens using the pinned cache's removal API."""
    remove = int(cache.get_seq_length()) - int(length)
    if remove < 0:
        raise ValueError(f"Cannot extend cache from {cache.get_seq_length()} to {length}")
    cache.crop(-remove)
    if int(cache.get_seq_length()) != length:
        raise RuntimeError("Target cache rollback did not preserve requested prefix")


def extract_context_feature(hidden_states, layer_ids) -> torch.Tensor:
    # Index zero is the embedding output; layer i's output is hidden_states[i+1].
    return torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)


@dataclass
class ReferenceBackend:
    target: Any
    drafter: Any
    tokenizer: Any = None
    target_id: str = "Qwen/Qwen3-8B"
    drafter_id: str = "z-lab/Qwen3-8B-DFlash-b16"
    target_revision: str | None = None
    drafter_revision: str | None = None

    def __post_init__(self):
        from dflash.model import DFlash2DraftModel, DFlashDraftModel

        if not isinstance(self.drafter, DFlashDraftModel) or isinstance(self.drafter, DFlash2DraftModel):
            raise ValueError("M0–M4 requires the original DFlashDraftModel checkpoint")
        if self.target.config.model_type != "qwen3":
            raise ValueError("The reference backend currently supports Qwen3 targets only")
        dc = getattr(self.drafter.config, "dflash_config", {})
        self.input_embedding_scale = float(dc.get("input_embedding_scale", 1.0))
        self.output_multiplier = float(dc.get("output_multiplier", 1.0))
        self.final_logit_softcapping = dc.get("final_logit_softcapping")
        if self.final_logit_softcapping:
            raise ValueError("Nonlinear output softcapping cannot use the required linear head reconstruction")
        self.target.eval().requires_grad_(False)
        self.drafter.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        target: str | Path = "Qwen/Qwen3-8B",
        drafter: str | Path = "z-lab/Qwen3-8B-DFlash-b16",
        *,
        target_revision: str | None = None,
        drafter_revision: str | None = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ) -> "ReferenceBackend":
        from dflash.model import DFlashDraftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common = dict(local_files_only=True, dtype=dtype, attn_implementation=attn_implementation)
        target_model = AutoModelForCausalLM.from_pretrained(
            str(target), revision=target_revision, **common
        ).to(device)
        draft_model = DFlashDraftModel.from_pretrained(
            str(drafter), revision=drafter_revision, **common
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(target), revision=target_revision, local_files_only=True)
        return cls(target_model, draft_model, tokenizer, str(target), str(drafter), target_revision, drafter_revision)

    @property
    def device(self) -> torch.device:
        return self.target.device

    @property
    def block_size(self) -> int:
        return self.drafter.block_size

    @property
    def draft_lm_head(self):
        return self.target.get_output_embeddings()

    @property
    def stop_token_ids(self) -> list[int]:
        ids = self.target.generation_config.eos_token_id
        if ids is None and self.tokenizer is not None:
            ids = self.tokenizer.eos_token_id
        return [ids] if isinstance(ids, int) else list(ids or [])

    def tokenize(self, prompt: str, *, enable_thinking: bool = False) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

    def head_metadata(self) -> dict:
        head = self.draft_lm_head
        return {
            "artifact_id": self.target_id, "revision": self.target_revision,
            "module": "Qwen3ForCausalLM.lm_head", "hook": "DFlashDraftModel.norm output, logical slots [1:]",
            "shape": list(head.weight.shape), "dtype": str(head.weight.dtype),
            "bias": head.bias is not None,
            "tied_weights": head.weight.data_ptr() == self.target.get_input_embeddings().weight.data_ptr(),
            "output_multiplier": self.output_multiplier, "final_logit_softcapping": None,
        }

    @torch.no_grad()
    def upstream_generate(self, input_ids, max_new_tokens, *, temperature=0., top_p=1., top_k=0,
                          stop_token_ids=None, return_stats=False):
        from dflash.model import dflash_generate

        return dflash_generate(
            self.drafter, self.target, input_ids, max_new_tokens,
            self.stop_token_ids if stop_token_ids is None else stop_token_ids,
            temperature=temperature, top_p=top_p, top_k=top_k, return_stats=return_stats,
        )

    @torch.no_grad()
    def target_only_generate(self, input_ids, max_new_tokens, *, temperature=0., warper=None,
                             generator=None, stop_token_ids=None):
        from .sampler import SamplingWarper, sample_token

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        warper = SamplingWarper() if warper is None else warper
        stop = set(self.stop_token_ids if stop_token_ids is None else stop_token_ids)
        cache = make_cache(self.target.config)
        pieces = [input_ids]
        current = input_ids
        for _ in range(max_new_tokens):
            start = cache.get_seq_length()
            positions = torch.arange(start, start + current.shape[1], device=input_ids.device)[None]
            out = self.target(current, position_ids=positions, past_key_values=cache,
                              use_cache=True, logits_to_keep=1)
            token, _ = sample_token(out.logits[0, -1], temperature, warper, generator)
            current = input_ids.new_tensor([[token]])
            pieces.append(current)
            if token in stop:
                break
        return torch.cat(pieces, dim=1)

    @torch.no_grad()
    def check_cache_rollback(self, prefix_ids: torch.Tensor, proposed_tokens: torch.Tensor,
                             accepted_length: int, replacement_token: int) -> dict:
        """Compare rolled-back suffix logits with a fresh pass of the valid prefix.

        This verifies a deliberately partial acceptance independently of any
        adapter or acceptance RNG, using the actual installed target/cache class.
        """
        if prefix_ids.shape[0] != 1 or proposed_tokens.ndim != 2:
            raise ValueError("Expected [1,N] token tensors")
        if not 0 <= accepted_length < proposed_tokens.shape[1]:
            raise ValueError("Rollback probe must reject within the proposal")
        cache = make_cache(self.target.config)
        self.target(prefix_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        self.target(proposed_tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
        keep = prefix_ids.shape[1] + accepted_length
        crop_cache(cache, keep)
        replacement = prefix_ids.new_tensor([[replacement_token]])
        rolled = self.target(replacement, past_key_values=cache, use_cache=True, logits_to_keep=1).logits.float()
        valid = torch.cat([prefix_ids, proposed_tokens[:, :accepted_length], replacement], dim=1)
        fresh = self.target(valid, use_cache=False, logits_to_keep=1).logits.float()
        return {
            "cache_length": int(cache.get_seq_length()), "expected_length": valid.shape[1],
            "accepted_length": accepted_length,
            "max_abs_logit_error": float((rolled - fresh).abs().max()),
            "argmax_equal": bool(torch.equal(rolled.argmax(-1), fresh.argmax(-1))),
            "rolled_argmax": int(rolled[0, -1].argmax()),
            "fresh_argmax": int(fresh[0, -1].argmax()),
        }
