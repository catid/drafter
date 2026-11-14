"""Optional M5 Triton kernels strictly gated on numerical parity and measured speed.

Conformance rule (`prompt.md` lines 869, 891):
  - Never imported or executed in M0–M4 (`m0.py`, `collect.py`, `diagnose.py`,
    `train.py` M2 default, or `eval_closed_loop.py` `--stage m3|m4`).
  - In M5 (`--stage m5 --m5_triton`), every Triton kernel must pass BOTH:
      1. Runtime numerical parity check against the exact PyTorch reference on
         representative active shapes (`atol` / `rtol`), raising `TritonParityError`
         on any numerical discrepancy when `strict=True`; AND
      2. Synchronized median speed gate (`triton_median_ms < pytorch_median_ms * speed_margin`),
         cleanly rejecting and falling back to the PyTorch reference whenever Triton is
         unavailable, running on CPU, or slower than PyTorch.
"""
from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Callable

import torch

from .adapter import full_kl, loss_weights
from .sparse import hash_keys, proximal_l1

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


class TritonParityError(RuntimeError):
    """Raised when a candidate M5 Triton kernel fails numerical parity against PyTorch."""


@dataclass(frozen=True)
class TritonGateReport:
    kernel_name: str
    enabled: bool
    status: str
    device: str
    parity_passed: bool | None
    max_abs_error: float | None
    atol: float
    rtol: float
    speed_passed: bool | None
    triton_median_ms: float | None
    pytorch_median_ms: float | None
    speedup_ratio: float | None
    fallback: str = 'pytorch_reference'

    def to_dict(self) -> dict:
        return {
            'kernel_name': self.kernel_name,
            'enabled': self.enabled,
            'status': self.status,
            'device': self.device,
            'parity_passed': self.parity_passed,
            'max_abs_error': self.max_abs_error,
            'atol': self.atol,
            'rtol': self.rtol,
            'speed_passed': self.speed_passed,
            'triton_median_ms': self.triton_median_ms,
            'pytorch_median_ms': self.pytorch_median_ms,
            'speedup_ratio': self.speedup_ratio,
            'fallback': self.fallback,
        }


if TRITON_AVAILABLE:
    @triton.jit
    def _signed_hash_lookup_kernel(
        d_ptr,
        prev_ptr,
        curr_ptr,
        values_ptr,
        last_u_ptr,
        cum_thresh_ptr,
        out_ptr,
        n_elements,
        capacity: tl.int64,
        thresh_u: tl.float32,
        has_lazy: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        d_val = tl.load(d_ptr + offs, mask=mask, other=1).to(tl.int64)
        prev_val = tl.load(prev_ptr + offs, mask=mask, other=0).to(tl.int64)
        curr_val = tl.load(curr_ptr + offs, mask=mask, other=0).to(tl.int64)

        prime = tl.full((), 2147483647, dtype=tl.int64)
        mixed = (d_val * 73856093 + prev_val * 19349663 + curr_val * 83492791 + 104729) % prime
        bucket = ((mixed * 48271 + 1) % prime) % capacity
        sign_hash = (d_val * 15485863 + prev_val * 32452843 + curr_val * 49979687 + 67867967) % prime
        bit = ((sign_hash * 40699 + 17) % prime) % 2
        sign = (1 - 2 * bit).to(tl.float32)

        raw = tl.load(values_ptr + bucket, mask=mask, other=0.0).to(tl.float32)
        if has_lazy:
            lu = tl.load(last_u_ptr + bucket, mask=mask, other=0).to(tl.int64)
            base_t = tl.load(cum_thresh_ptr + lu, mask=mask, other=0.0).to(tl.float32)
            delta_t = thresh_u - base_t
            abs_shrunk = tl.maximum(tl.abs(raw) - delta_t, 0.0)
            val_sign = tl.where(raw > 0.0, 1.0, tl.where(raw < 0.0, -1.0, 0.0))
            eff = val_sign * abs_shrunk
        else:
            eff = raw
        tl.store(out_ptr + offs, eff * sign, mask=mask)

    @triton.jit
    def _rowwise_kl_forward_kernel(
        scores_ptr,
        target_ptr,
        out_row_kl_ptr,
        n_cols,
        stride_s_row,
        stride_t_row,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        s_row = scores_ptr + row * stride_s_row
        t_row = target_ptr + row * stride_t_row

        # Pass 1: online logsumexp for both target_logits and scores.
        m_t = -float('inf')
        d_t = 0.0
        m_s = -float('inf')
        d_s = 0.0
        for start in range(0, n_cols, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < n_cols
            tv = tl.load(t_row + cols, mask=mask, other=-float('inf')).to(tl.float32)
            sv = tl.load(s_row + cols, mask=mask, other=-float('inf')).to(tl.float32)

            mt_blk = tl.max(tv, axis=0)
            mt_new = tl.maximum(m_t, mt_blk)
            d_t = d_t * tl.exp(m_t - mt_new) + tl.sum(tl.exp(tv - mt_new), axis=0)
            m_t = mt_new

            ms_blk = tl.max(sv, axis=0)
            ms_new = tl.maximum(m_s, ms_blk)
            d_s = d_s * tl.exp(m_s - ms_new) + tl.sum(tl.exp(sv - ms_new), axis=0)
            m_s = ms_new

        lse_t = m_t + tl.log(d_t)
        lse_s = m_s + tl.log(d_s)

        # Pass 2: exact sum_v exp(tv - lse_t) * ((tv - lse_t) - (sv - lse_s)).
        kl_acc = 0.0
        for start in range(0, n_cols, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < n_cols
            tv = tl.load(t_row + cols, mask=mask, other=-float('inf')).to(tl.float32)
            sv = tl.load(s_row + cols, mask=mask, other=0.0).to(tl.float32)
            log_p = tv - lse_t
            log_q = sv - lse_s
            p = tl.where(mask, tl.exp(log_p), 0.0)
            term = tl.where(mask, p * (log_p - log_q), 0.0)
            kl_acc += tl.sum(term, axis=0)

        tl.store(out_row_kl_ptr + row, kl_acc)


def emulate_signed_hash_kernel_cpu(
    d,
    previous,
    current,
    values: torch.Tensor,
    capacity: int,
    *,
    last_u: torch.Tensor | None = None,
    cum_thresh: torch.Tensor | None = None,
    u: int = 0,
) -> torch.Tensor:
    """Exact int64/fp32 mirror of `_signed_hash_lookup_kernel`'s register equations for CPU parity verification."""
    d_t, prev_t, curr_t = torch.broadcast_tensors(
        torch.as_tensor(d, dtype=torch.int64, device=values.device),
        torch.as_tensor(previous, dtype=torch.int64, device=values.device),
        torch.as_tensor(current, dtype=torch.int64, device=values.device),
    )
    prime = 2147483647
    mixed = (d_t * 73856093 + prev_t * 19349663 + curr_t * 83492791 + 104729) % prime
    bucket = ((mixed * 48271 + 1) % prime) % int(capacity)
    sign_hash = (d_t * 15485863 + prev_t * 32452843 + curr_t * 49979687 + 67867967) % prime
    bit = ((sign_hash * 40699 + 17) % prime) % 2
    sign = (1 - 2 * bit).to(torch.float32)
    raw = values[bucket].to(torch.float32)
    if last_u is not None and cum_thresh is not None:
        delta_t = cum_thresh[u] - cum_thresh[last_u[bucket].long()]
        eff = proximal_l1(raw, delta_t)
    else:
        eff = raw
    return eff * sign


def triton_signed_hash_lookup(
    d,
    previous,
    current,
    values: torch.Tensor,
    capacity: int,
    *,
    last_u: torch.Tensor | None = None,
    cum_thresh: torch.Tensor | None = None,
    u: int = 0,
) -> torch.Tensor:
    if not TRITON_AVAILABLE or values.device.type != 'cuda':
        raise RuntimeError('triton_signed_hash_lookup requires CUDA tensors and installed triton')
    d_t, prev_t, curr_t = torch.broadcast_tensors(
        torch.as_tensor(d, dtype=torch.int64, device=values.device),
        torch.as_tensor(previous, dtype=torch.int64, device=values.device),
        torch.as_tensor(current, dtype=torch.int64, device=values.device),
    )
    out_shape = d_t.shape
    d_flat = d_t.contiguous().reshape(-1)
    prev_flat = prev_t.contiguous().reshape(-1)
    curr_flat = curr_t.contiguous().reshape(-1)
    n_el = d_flat.numel()
    out = torch.empty(n_el, dtype=torch.float32, device=values.device)
    if n_el == 0:
        return out.reshape(out_shape)
    has_lazy = last_u is not None and cum_thresh is not None
    dummy_i32 = last_u if has_lazy else torch.empty(1, dtype=torch.int32, device=values.device)
    dummy_f32 = cum_thresh if has_lazy else torch.empty(1, dtype=torch.float32, device=values.device)
    thresh_u = float(cum_thresh[u].item()) if has_lazy else 0.0
    block = 256
    grid = ((n_el + block - 1) // block,)
    _signed_hash_lookup_kernel[grid](
        d_flat, prev_flat, curr_flat, values, dummy_i32, dummy_f32, out,
        n_el, int(capacity), thresh_u, has_lazy=has_lazy, BLOCK_SIZE=block,
    )
    return out.reshape(out_shape)


def triton_rowwise_kl(scores: torch.Tensor, target_logits: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if not TRITON_AVAILABLE or scores.device.type != 'cuda':
        raise RuntimeError('triton_rowwise_kl requires CUDA tensors and installed triton')
    s = scores.float().contiguous()
    t = target_logits.detach().float().contiguous()
    n_rows, n_cols = s.shape
    row_kl = torch.empty(n_rows, dtype=torch.float32, device=s.device)
    _rowwise_kl_forward_kernel[(n_rows,)](
        s, t, row_kl, n_cols, s.stride(0), t.stride(0), BLOCK_V=1024,
    )
    if weights is None:
        weights = loss_weights(n_rows, s.device)
    return (row_kl * weights).sum()


def _bench_fn_ms(fn: Callable[[], torch.Tensor], device: torch.device, *, warmups: int = 3, repeats: int = 11) -> float:
    for _ in range(warmups):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        if device.type == 'cuda':
            with torch.cuda.device(device):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                fn()
                ev1.record()
            torch.cuda.synchronize(device)
            samples.append(float(ev0.elapsed_time(ev1)))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def evaluate_kernel_gate(
    kernel_name: str,
    reference_fn: Callable[[], torch.Tensor],
    candidate_fn: Callable[[], torch.Tensor] | None,
    *,
    device: torch.device | str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    speed_margin: float = 0.98,
    strict: bool = True,
    warmups: int = 3,
    repeats: int = 11,
    measured_times_ms: tuple[float, float] | None = None,
) -> tuple[Callable[[], torch.Tensor], TritonGateReport]:
    """Run mandatory numerical parity + speed gates; return `(winning_fn, report)`."""
    dev = torch.device(device)
    if candidate_fn is None:
        return reference_fn, TritonGateReport(
            kernel_name=kernel_name,
            enabled=False,
            status='unavailable_device_or_triton',
            device=str(dev),
            parity_passed=None,
            max_abs_error=None,
            atol=atol,
            rtol=rtol,
            speed_passed=None,
            triton_median_ms=None,
            pytorch_median_ms=None,
            speedup_ratio=None,
        )
    with torch.no_grad():
        ref_out = reference_fn().detach().float()
        cand_out = candidate_fn().detach().float()
        max_err = float((ref_out - cand_out).abs().max().item()) if ref_out.numel() else 0.0
        parity_ok = bool(torch.allclose(cand_out, ref_out, atol=atol, rtol=rtol))
    if not parity_ok:
        if strict:
            raise TritonParityError(
                f'Triton kernel {kernel_name} failed numerical parity: max_abs_error={max_err:.3e} > atol={atol:.3e}'
            )
        return reference_fn, TritonGateReport(
            kernel_name=kernel_name,
            enabled=False,
            status='rejected_parity_failure',
            device=str(dev),
            parity_passed=False,
            max_abs_error=max_err,
            atol=atol,
            rtol=rtol,
            speed_passed=None,
            triton_median_ms=None,
            pytorch_median_ms=None,
            speedup_ratio=None,
        )
    if measured_times_ms is not None:
        cand_ms, ref_ms = float(measured_times_ms[0]), float(measured_times_ms[1])
    else:
        ref_ms = _bench_fn_ms(reference_fn, dev, warmups=warmups, repeats=repeats)
        cand_ms = _bench_fn_ms(candidate_fn, dev, warmups=warmups, repeats=repeats)
    speed_ok = bool(cand_ms < ref_ms * speed_margin)
    ratio = (ref_ms / cand_ms) if cand_ms > 0 else None
    if not speed_ok:
        return reference_fn, TritonGateReport(
            kernel_name=kernel_name,
            enabled=False,
            status='rejected_slower_than_pytorch',
            device=str(dev),
            parity_passed=True,
            max_abs_error=max_err,
            atol=atol,
            rtol=rtol,
            speed_passed=False,
            triton_median_ms=cand_ms,
            pytorch_median_ms=ref_ms,
            speedup_ratio=ratio,
        )
    return candidate_fn, TritonGateReport(
        kernel_name=kernel_name,
        enabled=True,
        status='accepted',
        device=str(dev),
        parity_passed=True,
        max_abs_error=max_err,
        atol=atol,
        rtol=rtol,
        speed_passed=True,
        triton_median_ms=cand_ms,
        pytorch_median_ms=ref_ms,
        speedup_ratio=ratio,
        fallback='none',
    )


def build_gated_m5_kernels(
    device: torch.device | str,
    *,
    vocab_size: int = 248320,
    sparse_capacity: int = 2**20,
    M_sparse: int = 256,
    D: int = 4,
    K: int = 15,
    strict: bool = True,
) -> dict[str, TritonGateReport]:
    """Benchmark and parity-gate M5 Triton kernels on `device`, returning gate reports."""
    dev = torch.device(device)
    if dev.type != 'cuda' or not TRITON_AVAILABLE:
        _, hash_rep = evaluate_kernel_gate('signed_hash_lookup', lambda: torch.zeros(1), None, device=dev, strict=strict)
        _, kl_rep = evaluate_kernel_gate('rowwise_full_kl', lambda: torch.zeros(1), None, device=dev, strict=strict)
        return {'signed_hash_lookup': hash_rep, 'rowwise_full_kl': kl_rep}
    gen = torch.Generator(device=dev).manual_seed(2026)
    d_q = torch.arange(1, D + 1, device=dev, dtype=torch.int64)[:, None]
    prev_q = torch.randint(0, vocab_size, (D, 1), generator=gen, device=dev, dtype=torch.int64)
    curr_q = torch.randint(0, vocab_size, (1, M_sparse), generator=gen, device=dev, dtype=torch.int64)
    vals = torch.randn(sparse_capacity, generator=gen, device=dev, dtype=torch.float32)

    def ref_hash():
        b, s = hash_keys(d_q, prev_q, curr_q, sparse_capacity, dev)
        return vals[b] * s

    def cand_hash():
        return triton_signed_hash_lookup(d_q, prev_q, curr_q, vals, sparse_capacity)

    _, hash_rep = evaluate_kernel_gate('signed_hash_lookup', ref_hash, cand_hash, device=dev,
                                       atol=1e-6, rtol=1e-6, strict=strict)
    scores = torch.randn((K, min(vocab_size, 32768)), generator=gen, device=dev, dtype=torch.float32)
    targets = torch.randn_like(scores)
    weights = loss_weights(K, dev)

    def ref_kl():
        return full_kl(scores, targets, weights)

    def cand_kl():
        return triton_rowwise_kl(scores, targets, weights)

    _, kl_rep = evaluate_kernel_gate('rowwise_full_kl', ref_kl, cand_kl, device=dev,
                                     atol=1e-4, rtol=1e-4, strict=strict)
    return {'signed_hash_lookup': hash_rep, 'rowwise_full_kl': kl_rep}


def build_gated_torch_compile(
    device: torch.device | str,
    *,
    vocab_size: int = 248320,
    r: int = 64,
    p: int = 128,
    K: int = 15,
    speed_margin: float = 0.98,
    strict: bool = False,
) -> dict[str, TritonGateReport]:
    """Profile and parity-gate `torch.compile` on dense adapter ops (`prompt.md:869, 891`)."""
    from .adapter import block_unary

    dev = torch.device(device)
    if not hasattr(torch, 'compile'):
        _, rep = evaluate_kernel_gate('torch_compile_block_unary', lambda: torch.zeros(1), None, device=dev, strict=strict)
        return {'torch_compile_block_unary': rep}

    v_bench = min(int(vocab_size), 8192 if dev.type == 'cpu' else int(vocab_size))
    gen = torch.Generator(device=dev).manual_seed(2026)
    E = torch.randn((v_bench, r), generator=gen, device=dev, dtype=torch.float32)
    W = torch.randn((r, p), generator=gen, device=dev, dtype=torch.float32)
    b = torch.randn((v_bench,), generator=gen, device=dev, dtype=torch.float32)
    H = torch.randn((K, p), generator=gen, device=dev, dtype=torch.float32)

    def ref_unary():
        return block_unary(E, W, b, H)

    try:
        compiled_unary = torch.compile(block_unary)
        _ = compiled_unary(E, W, b, H)

        def cand_unary():
            return compiled_unary(E, W, b, H)
    except Exception:
        cand_unary = None

    _, rep = evaluate_kernel_gate(
        'torch_compile_block_unary',
        ref_unary,
        cand_unary,
        device=dev,
        atol=1e-4,
        rtol=1e-4,
        speed_margin=speed_margin,
        warmups=1 if dev.type == 'cpu' else 3,
        repeats=3 if dev.type == 'cpu' else 11,
        strict=strict,
    )
    return {'torch_compile_block_unary': rep}
