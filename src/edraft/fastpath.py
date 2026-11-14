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
from typing import Callable
import torch

from .adapter import block_unary, full_kl
from .sampler import SamplingWarper, probabilities as eager_probabilities, probability_dtype


@dataclass(frozen=True)
class FastPathConfig:
    compile: bool = False
    cuda_graph: bool = False
    compile_mode: str = "default"
    simulate_cpu_graph: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.compile or self.cuda_graph or self.simulate_cpu_graph)

    def to_dict(self) -> dict:
        return {"compile": bool(self.compile), "cuda_graph": bool(self.cuda_graph),
                "compile_mode": self.compile_mode}


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

    def __call__(self, *tensors: torch.Tensor, extra_args: tuple = ()) -> torch.Tensor:
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
                    static_output = sample_out.clone()
                entry = {"kind": "cpu_sim", "inputs": static_inputs, "output": static_output}
            self._entries[key] = entry
            self.graph_captures += 1
            return entry["output"].clone()

        for dst, src in zip(entry["inputs"], tensors):
            dst.copy_(src)
        if entry["kind"] == "cuda":
            entry["graph"].replay()
        else:
            with torch.no_grad():
                entry["output"].copy_(self.fn(*entry["inputs"], *extra_args))
        self.graph_replays += 1
        return entry["output"].clone()


def _fresh_compile(fn: Callable, mode: str) -> Callable:
    """Clone `fn.__code__` so each compiled specialization owns an isolated Dynamo guard cache."""
    isolated = types.FunctionType(
        fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    return torch.compile(isolated, mode=mode, fullgraph=False)


def _make_compiled_unary(mode: str) -> Callable:
    def _unary_impl(E: torch.Tensor, W: torch.Tensor, b: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return (H.to(W.dtype) @ W.T) @ E.to(W.dtype).T + b

    return _fresh_compile(_unary_impl, mode)


def _make_compiled_full_kl(mode: str) -> Callable:
    def _full_kl_impl(scores: torch.Tensor, target_logits: torch.Tensor,
                      weights: torch.Tensor | None = None) -> torch.Tensor:
        p_log = torch.log_softmax(target_logits.float().detach(), -1)
        q_log = torch.log_softmax(scores.float(), -1)
        per_slot = (p_log.exp() * (p_log - q_log)).sum(-1)
        if weights is None:
            positions = torch.arange(scores.shape[0], device=scores.device, dtype=torch.float32)
            weights = torch.exp(-positions / 5.0)
        return (per_slot * weights).sum()

    return _fresh_compile(_full_kl_impl, mode)


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


class FastPathRuntime:
    """Symmetric optional fastpath dispatcher across plain, bigram, static, and online arms."""
    def __init__(self, config: FastPathConfig = FastPathConfig(), *, static_block_length: int | None = None):
        self.config = config
        self.static_block_length = static_block_length
        self._unary_base = _make_compiled_unary(config.compile_mode) if config.compile else block_unary
        self.full_kl_fn = _make_compiled_full_kl(config.compile_mode) if config.compile else full_kl
        self._probs_compiled_cache: dict[tuple[int, float, SamplingWarper], Callable] = {}
        if config.cuda_graph or config.simulate_cpu_graph:
            allowed_unary = (static_block_length,) if static_block_length is not None else None
            self._unary_graph = StaticGraphCallable(
                lambda E, W, b, H: self._unary_base(E, W, b, H),
                allowed_first_dims=allowed_unary,
                simulate_cpu=config.simulate_cpu_graph,
            )
            self._probs_graph = StaticGraphCallable(
                lambda scores, temperature, warper: self._probs_callable_for(
                    scores.ndim, temperature, warper
                )(scores),
                simulate_cpu=config.simulate_cpu_graph,
            )
        else:
            self._unary_graph = None
            self._probs_graph = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _probs_callable_for(self, ndim: int, temperature: float, warper: SamplingWarper) -> Callable:
        if not self.config.compile:
            return lambda s: eager_probabilities(s, temperature, warper)
        key = (int(ndim), float(temperature), warper)
        fn = self._probs_compiled_cache.get(key)
        if fn is None:
            fn = _make_compiled_probs(temperature, warper, self.config.compile_mode)
            self._probs_compiled_cache[key] = fn
        return fn

    def unary(self, E: torch.Tensor, W: torch.Tensor, b: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        grad_needed = torch.is_grad_enabled() and any(x.requires_grad for x in (E, W, b, H))
        if self.static_block_length is not None and H.shape[0] != self.static_block_length:
            if self._unary_graph is not None:
                self._unary_graph.eager_fallbacks += 1
            if grad_needed:
                return block_unary(E, W, b, H)
            with torch.no_grad():
                return block_unary(E, W, b, H)
        if self._unary_graph is not None:
            if (H.device.type == "cuda" or self.config.simulate_cpu_graph) and not grad_needed:
                with torch.no_grad():
                    return self._unary_bound_E(E, W, b, H)
            self._unary_graph.eager_fallbacks += 1
        if grad_needed:
            return self._unary_base(E, W, b, H)
        with torch.no_grad():
            return self._unary_base(E, W, b, H)

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
            return block_unary(E, W, b, H)
        out = graph_fn(W, b, H)
        self._unary_graph.graph_captures = sum(g.graph_captures for _, g, _ in cache.values())
        self._unary_graph.graph_replays = sum(g.graph_replays for _, g, _ in cache.values())
        return out

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

    def stats(self) -> dict:
        return {
            "compile": bool(self.config.compile),
            "cuda_graph": bool(self.config.cuda_graph),
            "unary_graph_captures": 0 if self._unary_graph is None else self._unary_graph.graph_captures,
            "unary_graph_replays": 0 if self._unary_graph is None else self._unary_graph.graph_replays,
            "unary_eager_fallbacks": 0 if self._unary_graph is None else self._unary_graph.eager_fallbacks,
            "probs_graph_captures": 0 if self._probs_graph is None else self._probs_graph.graph_captures,
            "probs_graph_replays": 0 if self._probs_graph is None else self._probs_graph.graph_replays,
            "probs_eager_fallbacks": 0 if self._probs_graph is None else self._probs_graph.eager_fallbacks,
        }
