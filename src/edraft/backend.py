"""Inspected Hugging Face backend for the original Qwen3 DFlash checkpoint.

Artifact downloads belong to the explicit bootstrap command. Model loading here is
strictly local, even when a hub model identifier rather than a snapshot path is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


_FAST_SDPA_NAME = "edraft_sdpa_clr"
_FAST_SDPA_REGISTERED = False


def register_edraft_sdpa_clr() -> str:
    """Register pure-PyTorch 2.13 `causal_lower_right` + native GQA SDPA backend."""
    global _FAST_SDPA_REGISTERED
    if _FAST_SDPA_REGISTERED:
        return _FAST_SDPA_NAME

    from torch.backends.cuda import SDPAParams, can_use_flash_attention
    from torch.nn.attention.bias import CausalBias, causal_lower_right
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    base_mask_fn = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]

    def _edraft_sdpa_clr_mask(
        batch_size: int,
        q_length: int,
        kv_length: int,
        q_offset: int = 0,
        kv_offset: int = 0,
        mask_function: Any = None,
        attention_mask: torch.Tensor | None = None,
        local_size: int | None = None,
        allow_is_causal_skip: bool = True,
        allow_is_bidirectional_skip: bool = False,
        **kwargs: Any,
    ):
        if (
            allow_is_causal_skip
            and attention_mask is None
            and local_size is None
            and isinstance(q_length, int)
            and isinstance(kv_length, int)
            and isinstance(q_offset, int)
            and isinstance(kv_offset, int)
            and kv_offset == 0
        ):
            if q_length == 1 or q_length == kv_length or q_offset == 0:
                return None
            if 1 < q_length < kv_length and q_offset + q_length == kv_length:
                return causal_lower_right(q_length, kv_length)
        call_kwargs = dict(kwargs)
        if mask_function is not None:
            call_kwargs["mask_function"] = mask_function
        return base_mask_fn(
            batch_size=batch_size,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=kv_offset,
            attention_mask=attention_mask,
            local_size=local_size,
            allow_is_causal_skip=allow_is_causal_skip,
            allow_is_bidirectional_skip=allow_is_bidirectional_skip,
            **call_kwargs,
        )

    def _edraft_sdpa_clr_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        position_bias: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if position_bias is None and not kwargs.get("output_attentions", False) and isinstance(attention_mask, CausalBias):
            kv_groups = int(getattr(module, "num_key_value_groups", 1))
            can_gqa = key.shape[-1] == value.shape[-1] <= 256
            if query.is_cuda:
                allow_native_gqa = can_gqa and can_use_flash_attention(
                    SDPAParams(query, key, value, None, dropout, False, kv_groups > 1)
                )
            else:
                allow_native_gqa = can_gqa
            if kv_groups > 1 and not allow_native_gqa:
                key = repeat_kv(key, kv_groups)
                value = repeat_kv(value, kv_groups)
                enable_gqa = False
            else:
                enable_gqa = kv_groups > 1
            attn_out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout,
                scale=scaling,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
            return attn_out.transpose(1, 2).contiguous(), None
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )

    ALL_MASK_ATTENTION_FUNCTIONS.register(_FAST_SDPA_NAME, _edraft_sdpa_clr_mask)
    if hasattr(ALL_MASK_ATTENTION_FUNCTIONS, "_global_mapping"):
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[_FAST_SDPA_NAME] = _edraft_sdpa_clr_mask
    ALL_ATTENTION_FUNCTIONS.register(_FAST_SDPA_NAME, _edraft_sdpa_clr_forward)
    if hasattr(ALL_ATTENTION_FUNCTIONS, "_global_mapping"):
        ALL_ATTENTION_FUNCTIONS._global_mapping[_FAST_SDPA_NAME] = _edraft_sdpa_clr_forward
    _FAST_SDPA_REGISTERED = True
    return _FAST_SDPA_NAME


def _build_preallocated_cache(config: Any, max_cache_len: int):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    cap = max(64, int(max_cache_len))

    class PreallocatedDynamicLayer(DynamicLayer):
        """Zero-cat in-place KV cache layer backed by contiguous BSHD storage."""

        def __init__(self, initial_capacity: int = cap, **kwargs: Any):
            super().__init__(**kwargs)
            self._initial_capacity = max(64, int(initial_capacity))
            self._capacity = 0
            self._seq_len = 0
            self._k_buf: torch.Tensor | None = None
            self._v_buf: torch.Tensor | None = None

        def _ensure_capacity(
            self,
            batch_size: int,
            num_heads: int,
            needed_len: int,
            head_dim: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            if (
                self._k_buf is None
                or self._v_buf is None
                or self._k_buf.shape[0] != batch_size
                or self._k_buf.shape[2] != num_heads
                or self._k_buf.shape[3] != head_dim
                or self._k_buf.dtype != dtype
                or self._k_buf.device != device
            ):
                alloc_cap = max(self._initial_capacity, int(needed_len))
                self._k_buf = torch.empty((batch_size, alloc_cap, num_heads, head_dim), dtype=dtype, device=device)
                self._v_buf = torch.empty((batch_size, alloc_cap, num_heads, head_dim), dtype=dtype, device=device)
                self._capacity = alloc_cap
                self._seq_len = 0
            elif needed_len > self._capacity:
                new_cap = max(int(needed_len), self._capacity * 2)
                new_k = torch.empty((batch_size, new_cap, num_heads, head_dim), dtype=dtype, device=device)
                new_v = torch.empty((batch_size, new_cap, num_heads, head_dim), dtype=dtype, device=device)
                if self._seq_len > 0:
                    new_k[:, : self._seq_len, :, :].copy_(self._k_buf[:, : self._seq_len, :, :])
                    new_v[:, : self._seq_len, :, :].copy_(self._v_buf[:, : self._seq_len, :, :])
                self._k_buf = new_k
                self._v_buf = new_v
                self._capacity = new_cap

        def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
            self.dtype, self.device = key_states.dtype, key_states.device
            bsz, num_heads, seq_len, head_dim = key_states.shape
            self._ensure_capacity(bsz, num_heads, max(self._initial_capacity, seq_len), head_dim, self.dtype, self.device)
            assert self._k_buf is not None and self._v_buf is not None
            self.keys = self._k_buf[:, :0, :, :].transpose(1, 2)
            self.values = self._v_buf[:, :0, :, :].transpose(1, 2)
            self.is_initialized = True

        def update(
            self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            bsz, num_heads, new_len, head_dim = key_states.shape
            end = self._seq_len + new_len
            self._ensure_capacity(bsz, num_heads, end, head_dim, key_states.dtype, key_states.device)
            assert self._k_buf is not None and self._v_buf is not None
            if new_len > 0:
                self._k_buf[:, self._seq_len : end, :, :].copy_(key_states.transpose(1, 2))
                self._v_buf[:, self._seq_len : end, :, :].copy_(value_states.transpose(1, 2))
            self._seq_len = end
            self.keys = self._k_buf[:, :end, :, :].transpose(1, 2)
            self.values = self._v_buf[:, :end, :, :].transpose(1, 2)
            return self.keys, self.values

        def get_seq_length(self) -> int:
            return self._seq_len if self.is_initialized else 0

        def crop(self, tokens_to_remove: int) -> None:
            if not self.is_initialized or tokens_to_remove == 0:
                return
            if tokens_to_remove > 0:
                if tokens_to_remove >= self._seq_len:
                    return
                new_len = int(tokens_to_remove)
            else:
                new_len = max(0, self._seq_len - (-int(tokens_to_remove)))
            self._seq_len = new_len
            assert self._k_buf is not None and self._v_buf is not None
            self.keys = self._k_buf[:, :new_len, :, :].transpose(1, 2)
            self.values = self._v_buf[:, :new_len, :, :].transpose(1, 2)

        def reset_sequence(self, min_capacity: int = 0) -> None:
            req_cap = int(min_capacity)
            if req_cap > self._initial_capacity:
                self._initial_capacity = req_cap
            self._seq_len = 0
            if self.is_initialized and self._k_buf is not None and self._v_buf is not None:
                if req_cap > self._capacity:
                    bsz, _, num_heads, head_dim = self._k_buf.shape
                    self._k_buf = torch.empty(
                        (bsz, req_cap, num_heads, head_dim), dtype=self._k_buf.dtype, device=self._k_buf.device
                    )
                    self._v_buf = torch.empty(
                        (bsz, req_cap, num_heads, head_dim), dtype=self._v_buf.dtype, device=self._v_buf.device
                    )
                    self._capacity = req_cap
                self.keys = self._k_buf[:, :0, :, :].transpose(1, 2)
                self.values = self._v_buf[:, :0, :, :].transpose(1, 2)

        def reset(self) -> None:
            self.reset_sequence()

        def batch_repeat_interleave(self, repeats: int) -> None:
            if self._seq_len > 0 and self._k_buf is not None and self._v_buf is not None:
                self._k_buf = self._k_buf.repeat_interleave(repeats, dim=0)
                self._v_buf = self._v_buf.repeat_interleave(repeats, dim=0)
                self.keys = self._k_buf[:, : self._seq_len, :, :].transpose(1, 2)
                self.values = self._v_buf[:, : self._seq_len, :, :].transpose(1, 2)

        def batch_select_indices(self, indices: torch.Tensor) -> None:
            if self._seq_len > 0 and self._k_buf is not None and self._v_buf is not None:
                self._k_buf = self._k_buf[indices, ...]
                self._v_buf = self._v_buf[indices, ...]
                self.keys = self._k_buf[:, : self._seq_len, :, :].transpose(1, 2)
                self.values = self._v_buf[:, : self._seq_len, :, :].transpose(1, 2)

    cache = DynamicCache(config=config)
    if len(cache.layers) > 0:
        cache.layers = [
            PreallocatedDynamicLayer(initial_capacity=cap) if type(layer) is DynamicLayer else layer
            for layer in cache.layers
        ]
    else:
        cache.layer_class_to_replicate = lambda: PreallocatedDynamicLayer(initial_capacity=cap)
    cache.activate_past_recording()
    return cache


def make_cache(config: Any, *, max_cache_len: int | None = None, preallocate: bool = False):
    if preallocate or max_cache_len is not None:
        return _build_preallocated_cache(config, max_cache_len or 512)

    from transformers import DynamicCache

    cache = DynamicCache(config=config)
    # Transformers 5.15 needs recording for reversible fixed-size cache layers.
    cache.activate_past_recording()
    return cache


def _reset_preallocated_cache(cache: Any, max_cache_len: int) -> None:
    for layer in getattr(cache, "layers", ()):
        if hasattr(layer, "reset_sequence"):
            layer.reset_sequence(min_capacity=max_cache_len)
        elif hasattr(layer, "reset"):
            layer.reset()


def crop_cache(cache: Any, length: int) -> None:
    """Keep exactly ``length`` tokens using the pinned cache's removal API."""
    cur = int(cache.get_seq_length())
    remove = cur - int(length)
    if remove < 0:
        raise ValueError(f"Cannot extend cache from {cur} to {length}")
    if remove > 0:
        cache.crop(-remove)
        if int(cache.get_seq_length()) != length:
            raise RuntimeError("Target cache rollback did not preserve requested prefix")
    else:
        cache.crop(0)


def extract_context_feature(
    hidden_states,
    layer_ids,
    *,
    max_tokens: int | None = None,
) -> torch.Tensor:
    # Index zero is the embedding output; layer i's output is hidden_states[i+1].
    if max_tokens is not None:
        return torch.cat([hidden_states[i + 1][:, :max_tokens, :] for i in layer_ids], dim=-1)
    return torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)


class _TargetFeatureTap:
    """Lightweight forward hooks on only `drafter.target_layer_ids` (5 layers instead of 37)."""

    def __init__(self, target_model: Any, layer_ids: list[int] | tuple[int, ...]) -> None:
        self.layer_ids = tuple(int(x) for x in layer_ids)
        self.enabled = False
        self._captured: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        base = getattr(target_model, "model", None)
        layers = getattr(base, "layers", None) if base is not None else None
        norm = getattr(base, "norm", None) if base is not None else None
        embed = getattr(base, "embed_tokens", None) if base is not None else None
        self.supported = layers is not None and norm is not None
        if not self.supported:
            return
        num_layers = len(layers)
        unique_ids = sorted(set(self.layer_ids))
        for lid in unique_ids:
            if lid == -1 and embed is not None:
                mod = embed
            elif 0 <= lid < num_layers - 1:
                mod = layers[lid]
            elif lid == num_layers - 1:
                mod = norm
            else:
                self.supported = False
                self.close()
                return

            def _hook(_module: Any, _inputs: Any, output: Any, key: int = lid) -> None:
                if not self.enabled:
                    return
                tensor = output[0] if isinstance(output, tuple) else output
                self._captured[key] = tensor

            self._handles.append(mod.register_forward_hook(_hook))

    def collect(self, *, max_tokens: int | None = None) -> torch.Tensor:
        if max_tokens is not None:
            parts = [self._captured[lid][:, :max_tokens, :] for lid in self.layer_ids]
        else:
            parts = [self._captured[lid] for lid in self.layer_ids]
        self._captured.clear()
        return torch.cat(parts, dim=-1)

    def clear(self) -> None:
        self._captured.clear()

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._captured.clear()


def _set_model_attn_implementation(model: Any, attn_impl: str) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg._attn_implementation = attn_impl
    for mod in model.modules():
        m_cfg = getattr(mod, "config", None)
        if m_cfg is not None and hasattr(m_cfg, "_attn_implementation"):
            m_cfg._attn_implementation = attn_impl


@dataclass
class ReferenceBackend:
    target: Any
    drafter: Any
    tokenizer: Any = None
    target_id: str = "Qwen/Qwen3-8B"
    drafter_id: str = "z-lab/Qwen3-8B-DFlash-b16"
    target_revision: str | None = None
    drafter_revision: str | None = None
    fast_decode: bool | None = None
    _base_attn_impl: str = field(default="sdpa", init=False, repr=False)
    _feature_tap: _TargetFeatureTap | None = field(default=None, init=False, repr=False)
    _pooled_caches: tuple[Any, Any] | None = field(default=None, init=False, repr=False)
    _pooled_in_use: bool = field(default=False, init=False, repr=False)

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
        cur_attn = str(getattr(self.target.config, "_attn_implementation", "sdpa") or "sdpa")
        self._base_attn_impl = "sdpa" if cur_attn == _FAST_SDPA_NAME else cur_attn
        resolved_fast = (
            os.environ.get("EDRAFT_FAST_DECODE", "0") == "1"
            if self.fast_decode is None
            else bool(self.fast_decode)
        )
        self.fast_decode = resolved_fast
        if self.fast_decode:
            self.enable_fast_decode(True)

    def enable_fast_decode(self, enabled: bool = True) -> None:
        self.fast_decode = bool(enabled)
        if self.fast_decode:
            attn_name = register_edraft_sdpa_clr()
            _set_model_attn_implementation(self.target, attn_name)
            if self._feature_tap is None:
                self._feature_tap = _TargetFeatureTap(self.target, self.drafter.target_layer_ids)
        else:
            if self._feature_tap is not None:
                self._feature_tap.clear()
            _set_model_attn_implementation(self.target, self._base_attn_impl)

    def acquire_decode_caches(self, max_cache_len: int) -> tuple[Any, Any, bool]:
        cap = max(64, int(max_cache_len))
        if not self.fast_decode:
            return make_cache(self.target.config), make_cache(self.drafter.config), False
        if not self._pooled_in_use and cap <= 40960:
            if self._pooled_caches is None:
                self._pooled_caches = (
                    make_cache(self.target.config, max_cache_len=cap, preallocate=True),
                    make_cache(self.drafter.config, max_cache_len=cap, preallocate=True),
                )
            else:
                _reset_preallocated_cache(self._pooled_caches[0], cap)
                _reset_preallocated_cache(self._pooled_caches[1], cap)
            self._pooled_in_use = True
            return self._pooled_caches[0], self._pooled_caches[1], True
        return (
            make_cache(self.target.config, max_cache_len=cap, preallocate=True),
            make_cache(self.drafter.config, max_cache_len=cap, preallocate=True),
            False,
        )

    def release_decode_caches(self, pooled: bool) -> None:
        if pooled and self._pooled_caches is not None:
            _reset_preallocated_cache(self._pooled_caches[0], 0)
            _reset_preallocated_cache(self._pooled_caches[1], 0)
            self._pooled_in_use = False

    def can_tap_target_features(self) -> bool:
        return bool(self.fast_decode and self._feature_tap is not None and self._feature_tap.supported)

    @contextmanager
    def capture_target_features(self):
        tap = self._feature_tap if self.can_tap_target_features() else None
        if tap is None:
            yield None
            return
        tap.clear()
        tap.enabled = True
        try:
            yield tap
        finally:
            tap.enabled = False

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
        fast_decode: bool | None = None,
    ) -> "ReferenceBackend":
        from dflash.model import DFlashDraftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        eff_fast = (
            os.environ.get("EDRAFT_FAST_DECODE", "0") == "1"
            if fast_decode is None
            else bool(fast_decode)
        )
        if eff_fast and attn_implementation == "sdpa":
            register_edraft_sdpa_clr()
        common = dict(local_files_only=True, dtype=dtype, attn_implementation=attn_implementation)
        target_model = AutoModelForCausalLM.from_pretrained(
            str(target), revision=target_revision, **common
        ).to(device)
        draft_model = DFlashDraftModel.from_pretrained(
            str(drafter), revision=drafter_revision, **common
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(target), revision=target_revision, local_files_only=True)
        return cls(
            target_model,
            draft_model,
            tokenizer,
            str(target),
            str(drafter),
            target_revision,
            drafter_revision,
            fast_decode=eff_fast,
        )

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
        from .sampler import SamplingWarper, sample_token, sample_token_tensor

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        warper = SamplingWarper() if warper is None else warper
        stop = set(self.stop_token_ids if stop_token_ids is None else stop_token_ids)
        if not self.fast_decode:
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

        n = int(input_ids.shape[1])
        if max_new_tokens == 0:
            return input_ids.clone()
        total_cap = n + max_new_tokens
        cache, _, pooled = self.acquire_decode_caches(total_cap + 2)
        try:
            device = input_ids.device
            out_buf = torch.empty((1, total_cap), dtype=input_ids.dtype, device=device)
            out_buf[:, :n] = input_ids
            positions_all = torch.arange(total_cap, device=device)[None]
            cur_len = n
            current = input_ids
            pos_slice = positions_all[:, :n]
            for _ in range(max_new_tokens):
                out = self.target(
                    current,
                    position_ids=pos_slice,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                tok_t, _ = sample_token_tensor(out.logits[0, -1], temperature, warper, generator)
                token = int(tok_t.item())
                out_buf[0, cur_len] = tok_t
                cur_len += 1
                if token in stop or cur_len >= total_cap:
                    break
                current = out_buf[:, cur_len - 1 : cur_len]
                pos_slice = positions_all[:, cur_len - 1 : cur_len]
            return out_buf[:, :cur_len].clone()
        finally:
            self.release_decode_caches(pooled)

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
        cache = make_cache(self.target.config, preallocate=bool(self.fast_decode))
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
