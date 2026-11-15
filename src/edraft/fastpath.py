"""Opt-in torch.compile and static-shape CUDAGraph wrappers with eager fallback.

All wrappers preserve exact mathematical semantics:
- Default configuration is disabled (`compile=False`, `cuda_graph=False`).
- Static-shape graphs fall back to eager whenever shapes vary (e.g. cap-trimmed
  terminal blocks with K_eff < K) or on CPU (unless `simulate_cpu_graph=True`
  in unit tests).
- Graph outputs are cloned before returning so callers never alias static replay
  buffers and `ProposalSnapshot` tensors are never mutated.
- Random draws (`torch.multinomial`, `torch.rand`) are never captured inside a
  CUDA graph; only deterministic score/probability transforms are captured.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import types
from typing import Any, Callable
import torch
from torch.nn import functional as F

from .adapter import block_unary, full_kl
from .sampler import (
    SamplingWarper,
    compact_probabilities as eager_compact_probabilities,
    probabilities as eager_probabilities,
    probability_dtype,
)


@dataclass(frozen=True)
class FastPathConfig:
    compile: bool = False
    cuda_graph: bool = False
    compile_mode: str = "default"
    simulate_cpu_graph: bool = False
    bf16_gemm: bool = False
    compile_drafter: bool = False
    compile_online: bool = False

    @property
    def enabled(self) -> bool:
        return bool(
            self.compile
            or self.cuda_graph
            or self.simulate_cpu_graph
            or self.bf16_gemm
            or self.compile_drafter
            or self.compile_online
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "compile": bool(self.compile),
            "cuda_graph": bool(self.cuda_graph),
            "compile_mode": self.compile_mode,
        }
        if self.bf16_gemm:
            out["bf16_gemm"] = True
        if self.compile_drafter:
            out["compile_drafter"] = True
        if self.compile_online:
            out["compile_online"] = True
        return out


def _clone_graph_output(out):
    if isinstance(out, tuple):
        return tuple(None if x is None else x.clone() for x in out)
    return out.clone()


def _copy_graph_output(dst, src) -> None:
    if isinstance(dst, tuple):
        for d, s in zip(dst, src):
            if d is not None and s is not None:
                d.copy_(s)
    else:
        dst.copy_(src)


class StaticGraphCallable:
    """Static-shape CUDA graph cache (with optional CPU buffer-replay simulation).

    `fn` must be a pure deterministic tensor function taking `num_tensor_args`
    positional tensors followed by any hashable static arguments `extra_key`.
    """
    def __init__(self, fn: Callable, *, allowed_first_dims: tuple[int, ...] | None = None,
                 warmup_iters: int = 3, simulate_cpu: bool = False):
        if warmup_iters < 1:
            raise ValueError("warmup_iters must be positive")
        self.fn = fn
        self.allowed_first_dims = allowed_first_dims
        self.warmup_iters = warmup_iters
        self.simulate_cpu = simulate_cpu
        self._entries: dict[tuple, dict] = {}
        self.graph_captures = 0
        self.graph_replays = 0
        self.eager_fallbacks = 0

    def __call__(self, *tensors: torch.Tensor, extra_args: tuple = ()):
        if not tensors:
            raise ValueError("StaticGraphCallable requires at least one tensor input")
        first = tensors[0]
        if self.allowed_first_dims is not None and first.ndim >= 1 and first.shape[0] not in self.allowed_first_dims:
            self.eager_fallbacks += 1
            return self.fn(*tensors, *extra_args)
        device = first.device
        if any(t.device != device for t in tensors):
            self.eager_fallbacks += 1
            return self.fn(*tensors, *extra_args)
        if device.type != "cuda" and not self.simulate_cpu:
            self.eager_fallbacks += 1
            return self.fn(*tensors, *extra_args)
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            self.eager_fallbacks += 1
            return self.fn(*tensors, *extra_args)

        key = (tuple((tuple(t.shape), t.dtype, t.device) for t in tensors), extra_args)
        entry = self._entries.get(key)
        if entry is None:
            static_inputs = tuple(torch.empty_like(t) for t in tensors)
            for dst, src in zip(static_inputs, tensors):
                dst.copy_(src)
            if device.type == "cuda":
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
                    for _ in range(self.warmup_iters):
                        _ = self.fn(*static_inputs, *extra_args)
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.device(device), torch.cuda.graph(graph), torch.no_grad():
                    static_output = self.fn(*static_inputs, *extra_args)
                entry = {"kind": "cuda", "graph": graph, "inputs": static_inputs, "output": static_output}
            else:
                with torch.no_grad():
                    sample_out = self.fn(*static_inputs, *extra_args)
                    static_output = _clone_graph_output(sample_out)
                entry = {"kind": "cpu_sim", "inputs": static_inputs, "output": static_output}
            self._entries[key] = entry
            self.graph_captures += 1
            return _clone_graph_output(entry["output"])

        for dst, src in zip(entry["inputs"], tensors):
            dst.copy_(src)
        if entry["kind"] == "cuda":
            entry["graph"].replay()
        else:
            with torch.no_grad():
                _copy_graph_output(entry["output"], self.fn(*entry["inputs"], *extra_args))
        self.graph_replays += 1
        return _clone_graph_output(entry["output"])


def _fresh_compile(fn: Callable, mode: str, *, dynamic: bool = False) -> Callable:
    """Clone `fn.__code__` so each compiled specialization owns an isolated Dynamo guard cache."""
    isolated = types.FunctionType(
        fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    return torch.compile(isolated, mode=mode, fullgraph=False, dynamic=dynamic)


def _make_compiled_unary(mode: str, *, bf16_gemm: bool = False) -> Callable:
    if bf16_gemm:
        def _unary_bf16_impl(E_bf16: torch.Tensor, W: torch.Tensor, b: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
            return ((H.bfloat16() @ W.bfloat16().T) @ E_bf16.T).to(b.dtype) + b

        return _fresh_compile(_unary_bf16_impl, mode)

    def _unary_impl(E: torch.Tensor, W: torch.Tensor, b: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return (H.to(W.dtype) @ W.T) @ E.to(W.dtype).T + b

    return _fresh_compile(_unary_impl, mode)


def _make_compiled_full_kl(mode: str) -> Callable:
    def _full_kl_impl(scores: torch.Tensor, target_logits: torch.Tensor,
                      weights: torch.Tensor | None = None) -> torch.Tensor:
        p_log = torch.log_softmax(target_logits.float().detach(), -1)
        p = p_log.exp()
        q_log = torch.log_softmax(scores.float(), -1)
        valid_p = p > 0
        p_log_safe = torch.where(valid_p, p_log, torch.zeros_like(p_log))
        q_log_safe = torch.where(valid_p, q_log, torch.zeros_like(q_log))
        per_slot = torch.where(valid_p, p * (p_log_safe - q_log_safe), torch.zeros_like(q_log)).sum(-1)
        if weights is None:
            positions = torch.arange(scores.shape[0], device=scores.device, dtype=torch.float32)
            weights = torch.exp(-positions / 5.0)
        return (per_slot * weights).sum()

    return _fresh_compile(_full_kl_impl, mode)


def _make_compiled_online_dense_step(mode: str, *, bias_only: bool = False) -> Callable:
    if bias_only:
        def _bias_step_impl(
            db: torch.Tensor, gb: torch.Tensor, off_b: torch.Tensor, hyper: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            shrink = hyper[0]
            lr_val = hyper[1]
            thresh = hyper[2]
            cand_b = (db - lr_val * gb) / shrink
            new_db = cand_b.sign() * (cand_b.abs() - thresh).clamp_min(0.0)
            return new_db, off_b + new_db

        return _fresh_compile(_bias_step_impl, mode)

    def _dense_step_impl(
        dW: torch.Tensor, gW: torch.Tensor,
        dA: torch.Tensor, gA: torch.Tensor,
        db: torch.Tensor, gb: torch.Tensor,
        off_W: torch.Tensor, off_A: torch.Tensor, off_b: torch.Tensor,
        hyper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shrink = hyper[0]
        lr_val = hyper[1]
        thresh = hyper[2]
        new_dW = (dW - lr_val * gW) / shrink
        new_dA = (dA - lr_val * gA) / shrink
        cand_b = (db - lr_val * gb) / shrink
        new_db = cand_b.sign() * (cand_b.abs() - thresh).clamp_min(0.0)
        return new_dW, new_dA, new_db, off_W + new_dW, off_A + new_dA, off_b + new_db

    return _fresh_compile(_dense_step_impl, mode)


def _make_compiled_probs(temperature: float, warper: SamplingWarper, mode: str) -> Callable:
    temp = float(temperature)
    top_k = int(warper.top_k)
    top_p = float(warper.top_p)
    min_p = float(warper.min_p)

    def _probs_impl(scores: torch.Tensor) -> torch.Tensor:
        s = probability_dtype(scores) / temp
        V = s.shape[-1]
        indices = None
        if 0 < top_k < V:
            s, indices = torch.topk(s, top_k, dim=-1)
        probs = torch.softmax(s, dim=-1)
        if top_p < 1.0:
            sorted_probs, order = probs.sort(dim=-1, descending=True)
            keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
            probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if min_p > 0.0:
            keep = probs >= min_p * probs.amax(dim=-1, keepdim=True)
            probs = probs * keep
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if indices is not None:
            full_shape = (*probs.shape[:-1], V)
            probs = torch.zeros(full_shape, dtype=probs.dtype, device=probs.device).scatter(-1, indices, probs)
        return probs

    return _fresh_compile(_probs_impl, mode)


def _make_compiled_compact_probs(temperature: float, warper: SamplingWarper, mode: str) -> Callable:
    temp = float(temperature)
    top_k = int(warper.top_k)
    top_p = float(warper.top_p)
    min_p = float(warper.min_p)

    def _compact_probs_impl(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        s = probability_dtype(scores) / temp
        V = s.shape[-1]
        indices = None
        if 0 < top_k < V:
            s, indices = torch.topk(s, top_k, dim=-1)
        probs = torch.softmax(s, dim=-1)
        if top_p < 1.0:
            sorted_probs, order = probs.sort(dim=-1, descending=True)
            keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
            probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if min_p > 0.0:
            keep = probs >= min_p * probs.amax(dim=-1, keepdim=True)
            probs = probs * keep
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs, indices

    return _fresh_compile(_compact_probs_impl, mode)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _CompiledDrafterRunner:
    """Stateless compiled DFlash drafter subgraphs around eager DynamicCache updates.

    Never patches `vendor/dflash` and never traces `DynamicCache.update` or
    `crop_cache` inside Dynamo, eliminating KV-cache mutation hazards and
    dynamic-prefix recompilation storms.
    """

    def __init__(self, backend: Any, mode: str) -> None:
        from .backend import crop_cache

        self._crop_cache = crop_cache
        drafter = backend.drafter
        embed_weight = backend.target.get_input_embeddings().weight
        embed_scale = float(backend.input_embedding_scale)
        lm_head = backend.draft_lm_head

        can_decompose = (
            all(hasattr(drafter, attr) for attr in ("layers", "fc", "hidden_norm", "rotary_emb", "norm"))
            and all(
                getattr(layer, "attention_conv", None) is None
                and getattr(layer, "mlp_conv", None) is None
                and not bool(getattr(layer.self_attn, "is_causal", False))
                and getattr(layer.self_attn, "sliding_window", None) is None
                for layer in drafter.layers
            )
        )
        self.decomposed = can_decompose
        if can_decompose:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]

            def _stem(canvas: torch.Tensor, target_hidden: torch.Tensor, position_ids: torch.Tensor):
                noise = F.embedding(canvas, embed_weight) * embed_scale
                t_proj = drafter.hidden_norm(drafter.fc(target_hidden))
                cos, sin = drafter.rotary_emb(noise, position_ids)
                return noise, t_proj, cos, sin

            self._stem = _fresh_compile(_stem, mode, dynamic=True)
            self._layers: list[tuple[Any, Callable, Callable]] = []
            for layer in drafter.layers:
                attn = layer.self_attn
                in_ln = layer.input_layernorm
                post_ln = layer.post_attention_layernorm
                mlp = layer.mlp
                head_dim = int(attn.head_dim)
                scaling = float(attn.scaling)

                def _make_layer_fns(attn_m=attn, in_ln_m=in_ln, post_ln_m=post_ln, mlp_m=mlp,
                                    hd=head_dim, sc=scaling):
                    def _pre(h_in: torch.Tensor, t_proj: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
                        h_norm = in_ln_m(h_in)
                        bsz, q_len = h_norm.shape[:-1]
                        ctx_len = t_proj.shape[1]
                        q = attn_m.q_proj(h_norm).view(bsz, q_len, -1, hd)
                        q = attn_m.q_norm(q).transpose(1, 2)
                        k = torch.cat([attn_m.k_proj(t_proj), attn_m.k_proj(h_norm)], dim=1).view(
                            bsz, ctx_len + q_len, -1, hd
                        )
                        v = torch.cat([attn_m.v_proj(t_proj), attn_m.v_proj(h_norm)], dim=1).view(
                            bsz, ctx_len + q_len, -1, hd
                        )
                        k = attn_m.k_norm(k).transpose(1, 2)
                        v = v.transpose(1, 2)
                        cos_u = cos.unsqueeze(1)
                        sin_u = sin.unsqueeze(1)
                        q = (q * cos_u[..., -q_len:, :]) + (_rotate_half(q) * sin_u[..., -q_len:, :])
                        k = (k * cos_u) + (_rotate_half(k) * sin_u)
                        return q, k, v

                    def _post(residual: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
                        bsz, _, q_len, _ = q.shape
                        attn_out, _ = sdpa_fn(
                            attn_m, q, k, v, None, dropout=0.0, scaling=sc, sliding_window=None
                        )
                        attn_out = attn_m.o_proj(attn_out.reshape(bsz, q_len, -1))
                        h_mid = residual + attn_out
                        return h_mid + mlp_m(post_ln_m(h_mid))

                    return _fresh_compile(_pre, mode, dynamic=True), _fresh_compile(_post, mode, dynamic=True)

                pre_fn, post_fn = _make_layer_fns()
                self._layers.append((attn, pre_fn, post_fn))

            def _head(hidden_states: torch.Tensor):
                h_raw = drafter.norm(hidden_states)[0, 1:, :]
                logits = drafter.compute_logits(h_raw[None], lm_head)[0]
                return h_raw, logits

            self._head = _fresh_compile(_head, mode, dynamic=True)
        else:
            self._drafter = drafter

            def _noise_fn(canvas: torch.Tensor):
                return F.embedding(canvas, embed_weight) * embed_scale

            def _logits_fn(h_raw: torch.Tensor):
                return drafter.compute_logits(h_raw[None], lm_head)[0]

            self._noise_fn = _fresh_compile(_noise_fn, mode, dynamic=True)
            self._logits_fn = _fresh_compile(_logits_fn, mode, dynamic=True)

    def __call__(
        self,
        canvas: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        draft_cache: Any,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.decomposed:
                hidden, t_proj, cos, sin = self._stem(canvas, target_hidden, position_ids)
                for attn, pre_fn, post_fn in self._layers:
                    q, k, v = pre_fn(hidden, t_proj, cos, sin)
                    if draft_cache is not None:
                        k, v = draft_cache.update(
                            k, v, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": None}
                        )
                    hidden = post_fn(hidden, q, k, v)
                if draft_cache is not None:
                    self._crop_cache(draft_cache, start)
                return self._head(hidden)

            noise = self._noise_fn(canvas)
            h_raw = self._drafter(
                target_hidden=target_hidden,
                noise_embedding=noise,
                position_ids=position_ids,
                past_key_values=draft_cache,
                use_cache=True,
            )[0, 1:, :]
            if draft_cache is not None:
                self._crop_cache(draft_cache, start)
            return h_raw, self._logits_fn(h_raw)


class FastPathRuntime:
    """Symmetric optional fastpath dispatcher across plain, bigram, static, and online arms."""
    def __init__(self, config: FastPathConfig = FastPathConfig(), *, static_block_length: int | None = None):
        self.config = config
        self.static_block_length = static_block_length
        self.bf16_gemm = bool(config.bf16_gemm)
        if config.compile:
            self._unary_base = _make_compiled_unary(config.compile_mode, bf16_gemm=self.bf16_gemm)
        elif self.bf16_gemm:
            self._unary_base = lambda E_bf16, W, b, H: block_unary(E_bf16, W, b, H, bf16_gemm=True, E_bf16=E_bf16)
        else:
            self._unary_base = block_unary
        self.full_kl_fn = _make_compiled_full_kl(config.compile_mode) if (config.compile or config.compile_online) else full_kl
        compile_online_active = bool(config.compile or config.compile_online)
        self._online_dense_step_fn = (
            _make_compiled_online_dense_step(config.compile_mode, bias_only=False) if compile_online_active else None
        )
        self._online_bias_step_fn = (
            _make_compiled_online_dense_step(config.compile_mode, bias_only=True) if compile_online_active else None
        )
        self._e_bf16_cache: dict[tuple, torch.Tensor] = {}
        self._drafter_runners: dict[int, _CompiledDrafterRunner] = {}
        self._probs_compiled_cache: dict[tuple[int, float, SamplingWarper], Callable] = {}
        self._compact_probs_compiled_cache: dict[tuple[int, float, SamplingWarper], Callable] = {}
        if config.cuda_graph or config.simulate_cpu_graph:
            allowed_unary = (static_block_length,) if static_block_length is not None else None
            self._unary_graph = StaticGraphCallable(
                lambda E_in, W, b, H: self._unary_base(E_in, W, b, H),
                allowed_first_dims=allowed_unary,
                simulate_cpu=config.simulate_cpu_graph,
            )
            self._probs_graph = StaticGraphCallable(
                lambda scores, temperature, warper: self._probs_callable_for(
                    scores.ndim, temperature, warper
                )(scores),
                simulate_cpu=config.simulate_cpu_graph,
            )
            self._compact_probs_graph = StaticGraphCallable(
                lambda scores, temperature, warper: self._compact_probs_callable_for(
                    scores.ndim, temperature, warper
                )(scores),
                simulate_cpu=config.simulate_cpu_graph,
            )
        else:
            self._unary_graph = None
            self._probs_graph = None
            self._compact_probs_graph = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _cached_e_input(self, E: torch.Tensor) -> torch.Tensor:
        if not self.bf16_gemm:
            return E
        if E.dtype == torch.bfloat16:
            return E
        key = (int(E.data_ptr()), tuple(E.shape), E.dtype, E.device, int(E._version))
        cached = self._e_bf16_cache.get(key)
        if cached is None:
            cached = E.detach().bfloat16()
            self._e_bf16_cache[key] = cached
        return cached

    def _probs_callable_for(self, ndim: int, temperature: float, warper: SamplingWarper) -> Callable:
        if not self.config.compile:
            return lambda s: eager_probabilities(s, temperature, warper)
        key = (int(ndim), float(temperature), warper)
        fn = self._probs_compiled_cache.get(key)
        if fn is None:
            fn = _make_compiled_probs(temperature, warper, self.config.compile_mode)
            self._probs_compiled_cache[key] = fn
        return fn

    def _compact_probs_callable_for(self, ndim: int, temperature: float, warper: SamplingWarper) -> Callable:
        if not self.config.compile:
            return lambda s: eager_compact_probabilities(s, temperature, warper)
        key = (int(ndim), float(temperature), warper)
        fn = self._compact_probs_compiled_cache.get(key)
        if fn is None:
            fn = _make_compiled_compact_probs(temperature, warper, self.config.compile_mode)
            self._compact_probs_compiled_cache[key] = fn
        return fn

    def unary(self, E: torch.Tensor, W: torch.Tensor, b: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        grad_needed = torch.is_grad_enabled() and any(x.requires_grad for x in (E, W, b, H))
        e_in = self._cached_e_input(E)
        if self.static_block_length is not None and H.shape[0] != self.static_block_length:
            if self._unary_graph is not None:
                self._unary_graph.eager_fallbacks += 1
            if grad_needed:
                return block_unary(E, W, b, H, bf16_gemm=self.bf16_gemm, E_bf16=e_in if self.bf16_gemm else None)
            with torch.no_grad():
                return block_unary(E, W, b, H, bf16_gemm=self.bf16_gemm, E_bf16=e_in if self.bf16_gemm else None)
        if self._unary_graph is not None:
            if (H.device.type == "cuda" or self.config.simulate_cpu_graph) and not grad_needed:
                with torch.no_grad():
                    return self._unary_bound_E(e_in, W, b, H)
            self._unary_graph.eager_fallbacks += 1
        if grad_needed:
            return self._unary_base(e_in, W, b, H)
        with torch.no_grad():
            return self._unary_base(e_in, W, b, H)

    def _unary_bound_E(self, E: torch.Tensor, W: torch.Tensor, b: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # Key by E's storage identity & version so E is never copied on every block replay.
        e_id = (int(E.data_ptr()), tuple(E.shape), E.dtype, E.device, int(E._version))
        cache = getattr(self, "_bound_e_graphs", None)
        if cache is None:
            cache = {}
            self._bound_e_graphs = cache
        entry = cache.get(e_id)
        if entry is None:
            E_ref = E.detach()
            allowed = (self.static_block_length,) if self.static_block_length is not None else None
            entry = StaticGraphCallable(
                lambda W_in, b_in, H_in: self._unary_base(E_ref, W_in, b_in, H_in),
                allowed_first_dims=None,
                simulate_cpu=self.config.simulate_cpu_graph,
            )
            cache[e_id] = (E_ref, entry, allowed)
        _, graph_fn, allowed = cache[e_id]
        if allowed is not None and H.shape[0] not in allowed:
            self._unary_graph.eager_fallbacks += 1
            return block_unary(E, W, b, H, bf16_gemm=self.bf16_gemm, E_bf16=E if self.bf16_gemm else None)
        out = graph_fn(W, b, H)
        self._unary_graph.graph_captures = sum(g.graph_captures for _, g, _ in cache.values())
        self._unary_graph.graph_replays = sum(g.graph_replays for _, g, _ in cache.values())
        return out

    def online_dense_step(
        self,
        dW: torch.Tensor,
        gW: torch.Tensor | None,
        dA: torch.Tensor,
        gA: torch.Tensor | None,
        db: torch.Tensor,
        gb: torch.Tensor,
        off_W: torch.Tensor,
        off_A: torch.Tensor,
        off_b: torch.Tensor,
        *,
        lr: float,
        lambda_l2: float,
        lambda_l1_b: float,
        bias_sparse_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shrink = 1.0 + float(lr) * float(lambda_l2)
        thresh = float(lr) * float(lambda_l1_b)
        with torch.no_grad():
            if bias_sparse_only:
                if self._online_bias_step_fn is not None:
                    hyper = torch.tensor([shrink, float(lr), thresh], dtype=torch.float32, device=db.device)
                    new_db, snap_b = self._online_bias_step_fn(db, gb, off_b, hyper)
                    return dW, dA, new_db, (off_W + dW).detach(), (off_A + dA).detach(), snap_b.detach()
                cand_b = (db - lr * gb) / shrink
                new_db = cand_b.sign() * (cand_b.abs() - thresh).clamp_min(0.0)
                return dW, dA, new_db, (off_W + dW).detach(), (off_A + dA).detach(), (off_b + new_db).detach()
            gW_eff = gW if gW is not None else torch.zeros_like(dW)
            gA_eff = gA if gA is not None else torch.zeros_like(dA)
            if self._online_dense_step_fn is not None:
                hyper = torch.tensor([shrink, float(lr), thresh], dtype=torch.float32, device=db.device)
                new_dW, new_dA, new_db, snap_W, snap_A, snap_b = self._online_dense_step_fn(
                    dW, gW_eff, dA, gA_eff, db, gb, off_W, off_A, off_b, hyper
                )
                return new_dW, new_dA, new_db, snap_W.detach(), snap_A.detach(), snap_b.detach()
            new_dW = (dW - lr * gW_eff) / shrink
            new_dA = (dA - lr * gA_eff) / shrink
            cand_b = (db - lr * gb) / shrink
            new_db = cand_b.sign() * (cand_b.abs() - thresh).clamp_min(0.0)
            return new_dW, new_dA, new_db, (off_W + new_dW).detach(), (off_A + new_dA).detach(), (off_b + new_db).detach()

    def draft_block(
        self,
        backend: Any,
        canvas: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        draft_cache: Any,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.compile_drafter:
            from .backend import crop_cache

            noise = F.embedding(canvas, backend.target.get_input_embeddings().weight) * backend.input_embedding_scale
            h_raw = backend.drafter(
                target_hidden=target_hidden,
                noise_embedding=noise,
                position_ids=position_ids,
                past_key_values=draft_cache,
                use_cache=True,
            )[0, 1:, :]
            if draft_cache is not None:
                crop_cache(draft_cache, start)
            draft_logits = backend.drafter.compute_logits(h_raw[None], backend.draft_lm_head)[0]
            return h_raw, draft_logits

        key = id(backend.drafter)
        runner = self._drafter_runners.get(key)
        if runner is None:
            runner = _CompiledDrafterRunner(backend, self.config.compile_mode)
            self._drafter_runners[key] = runner
        return runner(canvas, target_hidden, position_ids, draft_cache, start)

    def probabilities(self, scores: torch.Tensor, temperature: float,
                      warper: SamplingWarper = SamplingWarper()) -> torch.Tensor:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling probabilities require finite temperature > 0")
        if (scores.ndim == 2 and self.static_block_length is not None
                and scores.shape[0] not in (self.static_block_length, self.static_block_length + 1)):
            if self._probs_graph is not None:
                self._probs_graph.eager_fallbacks += 1
            with torch.no_grad():
                return eager_probabilities(scores, temperature, warper)
        with torch.no_grad():
            if self._probs_graph is not None:
                return self._probs_graph(scores, extra_args=(float(temperature), warper))
            return self._probs_callable_for(scores.ndim, temperature, warper)(scores)

    def compact_probabilities(
        self,
        scores: torch.Tensor,
        temperature: float,
        warper: SamplingWarper = SamplingWarper(),
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("sampling probabilities require finite temperature > 0")
        if (scores.ndim == 2 and self.static_block_length is not None
                and scores.shape[0] not in (self.static_block_length, self.static_block_length + 1)):
            if self._compact_probs_graph is not None:
                self._compact_probs_graph.eager_fallbacks += 1
            with torch.no_grad():
                return eager_compact_probabilities(scores, temperature, warper)
        with torch.no_grad():
            if self._compact_probs_graph is not None:
                return self._compact_probs_graph(scores, extra_args=(float(temperature), warper))
            return self._compact_probs_callable_for(scores.ndim, temperature, warper)(scores)

    def stats(self) -> dict:
        probs_captures = (
            (0 if self._probs_graph is None else self._probs_graph.graph_captures)
            + (0 if self._compact_probs_graph is None else self._compact_probs_graph.graph_captures)
        )
        probs_replays = (
            (0 if self._probs_graph is None else self._probs_graph.graph_replays)
            + (0 if self._compact_probs_graph is None else self._compact_probs_graph.graph_replays)
        )
        probs_fallbacks = (
            (0 if self._probs_graph is None else self._probs_graph.eager_fallbacks)
            + (0 if self._compact_probs_graph is None else self._compact_probs_graph.eager_fallbacks)
        )
        return {
            "compile": bool(self.config.compile),
            "cuda_graph": bool(self.config.cuda_graph),
            "bf16_gemm": bool(self.config.bf16_gemm),
            "compile_drafter": bool(self.config.compile_drafter),
            "compile_online": bool(self.config.compile_online),
            "unary_graph_captures": 0 if self._unary_graph is None else self._unary_graph.graph_captures,
            "unary_graph_replays": 0 if self._unary_graph is None else self._unary_graph.graph_replays,
            "unary_eager_fallbacks": 0 if self._unary_graph is None else self._unary_graph.eager_fallbacks,
            "probs_graph_captures": probs_captures,
            "probs_graph_replays": probs_replays,
            "probs_eager_fallbacks": probs_fallbacks,
        }
