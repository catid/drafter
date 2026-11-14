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

from dataclasses import asdict, dataclass
from typing import Callable
import torch

from .adapter import block_unary, full_kl
from .sampler import SamplingWarper, probabilities as eager_probabilities


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


def _maybe_compile(fn: Callable, mode: str) -> Callable:
    return torch.compile(fn, mode=mode, fullgraph=False)


class FastPathRuntime:
    """Symmetric optional fastpath dispatcher across plain, bigram, static, and online arms."""
    def __init__(self, config: FastPathConfig = FastPathConfig(), *, static_block_length: int | None = None):
        self.config = config
        self.static_block_length = static_block_length
        unary_base = _maybe_compile(block_unary, config.compile_mode) if config.compile else block_unary
        probs_base = _maybe_compile(eager_probabilities, config.compile_mode) if config.compile else eager_probabilities
        self.full_kl_fn = _maybe_compile(full_kl, config.compile_mode) if config.compile else full_kl
        self._unary_base = unary_base
        self._probs_base = probs_base
        if config.cuda_graph or config.simulate_cpu_graph:
            allowed_unary = (static_block_length,) if static_block_length is not None else None
            self._unary_graph = StaticGraphCallable(
                lambda E, W, b, H: unary_base(E, W, b, H),
                allowed_first_dims=allowed_unary,
                simulate_cpu=config.simulate_cpu_graph,
            )
            self._probs_graph = StaticGraphCallable(
                lambda scores, temperature, warper: probs_base(scores, temperature, warper),
                simulate_cpu=config.simulate_cpu_graph,
            )
        else:
            self._unary_graph = None
            self._probs_graph = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def unary(self, E: torch.Tensor, W: torch.Tensor, b: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        if self._unary_graph is not None:
            # Keep constant E bound via extra_args if unchanged; pass W, b, H as copied tensor inputs
            # only when H matches static_block_length, avoiding copying large [V, r] E on replay.
            if (self.static_block_length is None or H.shape[0] == self.static_block_length) and (
                H.device.type == "cuda" or self.config.simulate_cpu_graph
            ) and not (torch.is_grad_enabled() and any(x.requires_grad for x in (E, W, b, H))):
                return self._unary_bound_E(E, W, b, H)
            self._unary_graph.eager_fallbacks += 1
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
            return self._unary_base(E, W, b, H)
        out = graph_fn(W, b, H)
        self._unary_graph.graph_captures = sum(g.graph_captures for _, g, _ in cache.values())
        self._unary_graph.graph_replays = sum(g.graph_replays for _, g, _ in cache.values())
        return out

    def probabilities(self, scores: torch.Tensor, temperature: float,
                      warper: SamplingWarper = SamplingWarper()) -> torch.Tensor:
        if self._probs_graph is not None:
            if scores.ndim == 2 and self.static_block_length is not None:
                if scores.shape[0] not in (self.static_block_length, self.static_block_length + 1):
                    self._probs_graph.eager_fallbacks += 1
                    return self._probs_base(scores, temperature, warper)
            return self._probs_graph(scores, extra_args=(float(temperature), warper))
        return self._probs_base(scores, temperature, warper)

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
