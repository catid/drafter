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
        u_step: tl.int64,
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
            thresh_u = tl.load(cum_thresh_ptr + u_step).to(tl.float32)
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
        out_lse_s_ptr,
        out_lse_t_ptr,
        n_cols,
        stride_s_row,
        stride_t_row,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        s_row = scores_ptr + row * stride_s_row
        t_row = target_ptr + row * stride_t_row

        # Pass 1: online logsumexp for both target_logits and scores (safe on all-`-inf` tiles).
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
            mt_safe = tl.where(mt_new > -float('inf'), mt_new, 0.0)
            alpha_t = tl.where(m_t > -float('inf'), tl.exp(m_t - mt_safe), 0.0)
            sum_t = tl.sum(tl.where(mask & (tv > -float('inf')), tl.exp(tv - mt_safe), 0.0), axis=0)
            d_t = d_t * alpha_t + sum_t
            m_t = mt_new

            ms_blk = tl.max(sv, axis=0)
            ms_new = tl.maximum(m_s, ms_blk)
            ms_safe = tl.where(ms_new > -float('inf'), ms_new, 0.0)
            alpha_s = tl.where(m_s > -float('inf'), tl.exp(m_s - ms_safe), 0.0)
            sum_s = tl.sum(tl.where(mask & (sv > -float('inf')), tl.exp(sv - ms_safe), 0.0), axis=0)
            d_s = d_s * alpha_s + sum_s
            m_s = ms_new

        lse_t = tl.where(d_t > 0.0, m_t + tl.log(d_t), -float('inf'))
        lse_s = tl.where(d_s > 0.0, m_s + tl.log(d_s), -float('inf'))

        # Pass 2: exact sum_{v: P(v) > 0} exp(tv - lse_t) * ((tv - lse_t) - (sv - lse_s)).
        kl_acc = 0.0
        for start in range(0, n_cols, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < n_cols
            tv = tl.load(t_row + cols, mask=mask, other=-float('inf')).to(tl.float32)
            sv = tl.load(s_row + cols, mask=mask, other=-float('inf')).to(tl.float32)
            valid_t = mask & (tv > -float('inf')) & (lse_t > -float('inf'))
            lp = tl.where(valid_t, tv - lse_t, 0.0)
            p = tl.where(valid_t, tl.exp(lp), 0.0)
            lq = tl.where(mask & (lse_s > -float('inf')), sv - lse_s, 0.0)
            diff = tl.where(p > 0.0, lp - lq, 0.0)
            term = tl.where(p > 0.0, p * diff, 0.0)
            kl_acc += tl.sum(term, axis=0)

        tl.store(out_lse_s_ptr + row, lse_s)
        tl.store(out_lse_t_ptr + row, lse_t)
        tl.store(out_row_kl_ptr + row, kl_acc)

    @triton.jit
    def _rowwise_kl_backward_kernel(
        scores_ptr,
        target_ptr,
        lse_s_ptr,
        lse_t_ptr,
        row_scale_ptr,
        grad_scores_ptr,
        n_cols,
        stride_s_row,
        stride_t_row,
        stride_g_row,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        s_row = scores_ptr + row * stride_s_row
        t_row = target_ptr + row * stride_t_row
        g_row = grad_scores_ptr + row * stride_g_row
        lse_s = tl.load(lse_s_ptr + row).to(tl.float32)
        lse_t = tl.load(lse_t_ptr + row).to(tl.float32)
        scale = tl.load(row_scale_ptr + row).to(tl.float32)
        row_valid = tl.where(lse_t > -float('inf'), 1.0, 0.0)
        for start in range(0, n_cols, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < n_cols
            sv = tl.load(s_row + cols, mask=mask, other=-float('inf')).to(tl.float32)
            tv = tl.load(t_row + cols, mask=mask, other=-float('inf')).to(tl.float32)
            valid_s = mask & (sv > -float('inf')) & (lse_s > -float('inf'))
            valid_t = mask & (tv > -float('inf')) & (lse_t > -float('inf'))
            q = tl.where(valid_s, tl.exp(tl.where(valid_s, sv - lse_s, 0.0)), 0.0)
            p = tl.where(valid_t, tl.exp(tl.where(valid_t, tv - lse_t, 0.0)), 0.0)
            tl.store(g_row + cols, scale * (row_valid * q - p), mask=mask)


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
    if int(capacity) <= 0:
        raise ValueError('hash capacity must be positive')
    if (last_u is None) != (cum_thresh is None):
        raise ValueError('last_u and cum_thresh must be provided together')
    u_int = int(u)
    if cum_thresh is not None and (u_int < 0 or u_int >= cum_thresh.numel()):
        raise ValueError('step index u out of bounds for cum_thresh')
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
        delta_t = cum_thresh[u_int] - cum_thresh[last_u[bucket].long()]
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
    if int(capacity) <= 0:
        raise ValueError('hash capacity must be positive')
    if (last_u is None) != (cum_thresh is None):
        raise ValueError('last_u and cum_thresh must be provided together')
    has_lazy = last_u is not None and cum_thresh is not None
    u_int = int(u)
    if has_lazy and (u_int < 0 or u_int >= cum_thresh.numel()):
        raise ValueError('step index u out of bounds for cum_thresh')
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
    vals_c = values.contiguous()
    lu_buf = last_u.contiguous() if has_lazy else torch.empty(1, dtype=torch.int32, device=values.device)
    ct_buf = cum_thresh.contiguous() if has_lazy else torch.empty(1, dtype=torch.float32, device=values.device)
    block = 256
    grid = ((n_el + block - 1) // block,)
    _signed_hash_lookup_kernel[grid](
        d_flat, prev_flat, curr_flat, vals_c, lu_buf, ct_buf, out,
        n_el, int(capacity), u_int if has_lazy else 0, has_lazy=has_lazy, BLOCK_SIZE=block,
    )
    return out.reshape(out_shape)


def _non_inference_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.is_inference() and not torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return x.clone()
    return x


class _TritonRowwiseKLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, target_logits: torch.Tensor, emulate_cpu: bool = False) -> torch.Tensor:
        s = _non_inference_tensor(scores.detach().float().contiguous())
        t = _non_inference_tensor(target_logits.detach().float().contiguous())
        n_rows, n_cols = s.shape
        if not emulate_cpu:
            row_kl = torch.empty(n_rows, dtype=torch.float32, device=s.device)
            lse_s = torch.empty(n_rows, dtype=torch.float32, device=s.device)
            lse_t = torch.empty(n_rows, dtype=torch.float32, device=s.device)
            _rowwise_kl_forward_kernel[(n_rows,)](
                s, t, row_kl, lse_s, lse_t, n_cols, s.stride(0), t.stride(0), BLOCK_V=1024,
            )
        else:
            # Exact 1024-column tiled online LSE + KL CPU mirror of `_rowwise_kl_forward_kernel`.
            m_t = torch.full((n_rows,), -float('inf'), dtype=torch.float32, device=s.device)
            d_t = torch.zeros(n_rows, dtype=torch.float32, device=s.device)
            m_s = torch.full((n_rows,), -float('inf'), dtype=torch.float32, device=s.device)
            d_s = torch.zeros(n_rows, dtype=torch.float32, device=s.device)
            for start in range(0, n_cols, 1024):
                tc = t[:, start:start + 1024]
                sc = s[:, start:start + 1024]
                mt_blk = tc.max(dim=-1).values
                mt_new = torch.maximum(m_t, mt_blk)
                mt_safe = torch.where(mt_new > -float('inf'), mt_new, torch.zeros_like(mt_new))
                alpha_t = torch.where(m_t > -float('inf'), torch.exp(m_t - mt_safe), torch.zeros_like(m_t))
                sum_t = torch.where(tc > -float('inf'), torch.exp(tc - mt_safe[:, None]), torch.zeros_like(tc)).sum(dim=-1)
                d_t = d_t * alpha_t + sum_t
                m_t = mt_new

                ms_blk = sc.max(dim=-1).values
                ms_new = torch.maximum(m_s, ms_blk)
                ms_safe = torch.where(ms_new > -float('inf'), ms_new, torch.zeros_like(ms_new))
                alpha_s = torch.where(m_s > -float('inf'), torch.exp(m_s - ms_safe), torch.zeros_like(m_s))
                sum_s = torch.where(sc > -float('inf'), torch.exp(sc - ms_safe[:, None]), torch.zeros_like(sc)).sum(dim=-1)
                d_s = d_s * alpha_s + sum_s
                m_s = ms_new
            lse_t = torch.where(d_t > 0.0, m_t + torch.log(d_t), torch.full_like(m_t, -float('inf')))
            lse_s = torch.where(d_s > 0.0, m_s + torch.log(d_s), torch.full_like(m_s, -float('inf')))
            row_kl = torch.zeros(n_rows, dtype=torch.float32, device=s.device)
            for start in range(0, n_cols, 1024):
                sc = s[:, start:start + 1024]
                tc = t[:, start:start + 1024]
                valid_t = (tc > -float('inf')) & (lse_t[:, None] > -float('inf'))
                lp = torch.where(valid_t, tc - lse_t[:, None], torch.zeros_like(tc))
                p = torch.where(valid_t, torch.exp(lp), torch.zeros_like(tc))
                lq = torch.where(lse_s[:, None] > -float('inf'), sc - lse_s[:, None], torch.zeros_like(sc))
                diff = torch.where(p > 0.0, lp - lq, torch.zeros_like(p))
                row_kl = row_kl + torch.where(p > 0.0, p * diff, torch.zeros_like(p)).sum(-1)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(scores, t, lse_s, lse_t)
        ctx.orig_dtype = scores.dtype
        ctx.emulate_cpu = emulate_cpu
        return row_kl

    @staticmethod
    def backward(ctx, grad_row_kl: torch.Tensor):
        scores, t, lse_s, lse_t = ctx.saved_tensors
        row_scale = grad_row_kl.to(torch.float32).contiguous()
        if torch.is_grad_enabled():
            # Differentiable backward when `create_graph=True` (2nd-order HVPs).
            q = torch.softmax(scores.float(), dim=-1)
            valid_t = (t > -float('inf')) & (lse_t[:, None] > -float('inf'))
            p = torch.where(valid_t, torch.exp(torch.where(valid_t, t - lse_t[:, None], torch.zeros_like(t))), torch.zeros_like(t))
            row_valid = (lse_t > -float('inf')).to(torch.float32)[:, None]
            grad_s = row_scale[:, None] * (row_valid * q - p)
            return grad_s.to(ctx.orig_dtype), None, None
        s = _non_inference_tensor(scores.detach().float().contiguous())
        n_rows, n_cols = s.shape
        if not ctx.emulate_cpu:
            grad_s = torch.empty_like(s)
            _rowwise_kl_backward_kernel[(n_rows,)](
                s, t, lse_s, lse_t, row_scale, grad_s,
                n_cols, s.stride(0), t.stride(0), grad_s.stride(0), BLOCK_V=1024,
            )
        else:
            grad_s = torch.empty_like(s)
            row_valid = (lse_t > -float('inf')).to(torch.float32)[:, None]
            for start in range(0, n_cols, 1024):
                sc = s[:, start:start + 1024]
                tc = t[:, start:start + 1024]
                valid_s = (sc > -float('inf')) & (lse_s[:, None] > -float('inf'))
                valid_t = (tc > -float('inf')) & (lse_t[:, None] > -float('inf'))
                q = torch.where(valid_s, torch.exp(torch.where(valid_s, sc - lse_s[:, None], torch.zeros_like(sc))), torch.zeros_like(sc))
                p = torch.where(valid_t, torch.exp(torch.where(valid_t, tc - lse_t[:, None], torch.zeros_like(tc))), torch.zeros_like(tc))
                grad_s[:, start:start + 1024] = row_scale[:, None] * (row_valid * q - p)
        return grad_s.to(ctx.orig_dtype), None, None


def _run_rowwise_kl(scores: torch.Tensor, target_logits: torch.Tensor,
                    weights: torch.Tensor | float | None, *, emulate_cpu: bool) -> torch.Tensor:
    if scores.ndim != 2 or target_logits.shape != scores.shape:
        raise ValueError('scores and target_logits must share 2D shape [K, V]')
    if scores.shape[1] == 0:
        raise ValueError('vocab dimension must be positive')
    w = loss_weights(len(scores), scores.device) if weights is None else weights
    if isinstance(w, torch.Tensor):
        w = _non_inference_tensor(w)
    if scores.shape[0] == 0:
        s_clean = _non_inference_tensor(scores)
        per_slot = s_clean.float().sum(dim=-1) * 0.0
        return (per_slot * w).sum()
    row_kl = _TritonRowwiseKLFunction.apply(_non_inference_tensor(scores), target_logits, emulate_cpu)
    return (row_kl * w).sum()


def emulate_rowwise_kl_cpu(scores: torch.Tensor, target_logits: torch.Tensor,
                           weights: torch.Tensor | float | None = None) -> torch.Tensor:
    """CPU mirror of differentiable `triton_rowwise_kl` (`_TritonRowwiseKLFunction`)."""
    return _run_rowwise_kl(scores, target_logits, weights, emulate_cpu=True)


def triton_rowwise_kl(scores: torch.Tensor, target_logits: torch.Tensor,
                      weights: torch.Tensor | float | None = None) -> torch.Tensor:
    if not TRITON_AVAILABLE or scores.device.type != 'cuda':
        raise RuntimeError('triton_rowwise_kl requires CUDA tensors and installed triton')
    return _run_rowwise_kl(scores, target_logits, weights, emulate_cpu=False)


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
    grad_parity_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = None,
    extra_parity_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = None,
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
        if extra_parity_fn is not None:
            ref_ex, cand_ex = extra_parity_fn()
            ref_e = ref_ex.detach().float()
            cand_e = cand_ex.detach().float()
            ex_err = float((ref_e - cand_e).abs().max().item()) if ref_e.numel() else 0.0
            max_err = max(max_err, ex_err)
            parity_ok = parity_ok and bool(torch.allclose(cand_e, ref_e, atol=atol, rtol=rtol))
    if grad_parity_fn is not None:
        ref_grad, cand_grad = grad_parity_fn()
        ref_g = ref_grad.detach().float()
        cand_g = cand_grad.detach().float()
        grad_err = float((ref_g - cand_g).abs().max().item()) if ref_g.numel() else 0.0
        max_err = max(max_err, grad_err)
        parity_ok = parity_ok and bool(torch.allclose(cand_g, ref_g, atol=atol, rtol=rtol))
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
    return_callables: bool = False,
):
    """Benchmark and parity-gate M5 Triton kernels on `device` (full vocab forward+backward)."""
    dev = torch.device(device)
    if dev.type != 'cuda' or not TRITON_AVAILABLE:
        _, hash_rep = evaluate_kernel_gate('signed_hash_lookup', lambda: torch.zeros(1), None, device=dev, strict=strict)
        _, kl_rep = evaluate_kernel_gate('rowwise_full_kl', lambda: torch.zeros(1), None, device=dev, strict=strict)
        reports = {'signed_hash_lookup': hash_rep, 'rowwise_full_kl': kl_rep}
        callables = {'signed_hash_lookup': None, 'rowwise_full_kl': None}
        return (callables, reports) if return_callables else reports
    gen = torch.Generator(device=dev).manual_seed(2026)
    d_q = torch.arange(1, D + 1, device=dev, dtype=torch.int64)[:, None]
    prev_q = torch.randint(0, vocab_size, (D, 1), generator=gen, device=dev, dtype=torch.int64)
    curr_q = torch.randint(0, vocab_size, (1, M_sparse), generator=gen, device=dev, dtype=torch.int64)
    vals = torch.randn(sparse_capacity, generator=gen, device=dev, dtype=torch.float32)
    last_u = torch.randint(0, 4, (sparse_capacity,), generator=gen, device=dev, dtype=torch.int32)
    cum_thresh = torch.tensor([0.0, 0.05, 0.12, 0.25], device=dev, dtype=torch.float32)

    def ref_hash():
        b, s = hash_keys(d_q, prev_q, curr_q, sparse_capacity, dev)
        return vals[b] * s

    def cand_hash():
        return triton_signed_hash_lookup(d_q, prev_q, curr_q, vals, sparse_capacity)

    def lazy_hash_pair():
        b, s = hash_keys(d_q, prev_q, curr_q, sparse_capacity, dev)
        ref_lazy = proximal_l1(vals[b], cum_thresh[3] - cum_thresh[last_u[b].long()]) * s
        cand_lazy = triton_signed_hash_lookup(
            d_q, prev_q, curr_q, vals, sparse_capacity,
            last_u=last_u, cum_thresh=cum_thresh, u=3,
        )
        return ref_lazy, cand_lazy

    _, hash_rep = evaluate_kernel_gate('signed_hash_lookup', ref_hash, cand_hash, device=dev,
                                       atol=1e-6, rtol=1e-6, strict=strict,
                                       extra_parity_fn=lazy_hash_pair)
    scores_base = torch.randn((K, int(vocab_size)), generator=gen, device=dev, dtype=torch.float32)
    targets = torch.randn_like(scores_base)
    weights = loss_weights(K, dev)

    def grad_pair():
        with torch.enable_grad():
            s_ref = scores_base.detach().clone().requires_grad_(True)
            s_cand = scores_base.detach().clone().requires_grad_(True)
            full_kl(s_ref, targets, weights).backward()
            triton_rowwise_kl(s_cand, targets, weights).backward()
            return s_ref.grad, s_cand.grad

    def ref_kl():
        with torch.enable_grad():
            s_in = scores_base.detach().clone().requires_grad_(True)
            out = full_kl(s_in, targets, weights)
            out.backward()
            return out.detach()

    def cand_kl():
        with torch.enable_grad():
            s_in = scores_base.detach().clone().requires_grad_(True)
            out = triton_rowwise_kl(s_in, targets, weights)
            out.backward()
            return out.detach()

    _, kl_rep = evaluate_kernel_gate('rowwise_full_kl', ref_kl, cand_kl, device=dev,
                                     atol=1e-4, rtol=1e-4, strict=strict,
                                     grad_parity_fn=grad_pair)
    reports = {'signed_hash_lookup': hash_rep, 'rowwise_full_kl': kl_rep}
    callables = {
        'signed_hash_lookup': triton_signed_hash_lookup if hash_rep.enabled else None,
        'rowwise_full_kl': triton_rowwise_kl if kl_rep.enabled else None,
    }
    return (callables, reports) if return_callables else reports


def build_gated_torch_compile(
    device: torch.device | str,
    *,
    vocab_size: int = 248320,
    r: int = 64,
    p: int = 128,
    K: int = 15,
    speed_margin: float = 0.98,
    strict: bool = False,
    return_callables: bool = False,
):
    """Profile and parity-gate `torch.compile` on dense adapter ops (`prompt.md:869, 891`)."""
    from .adapter import block_unary

    dev = torch.device(device)
    if not hasattr(torch, 'compile'):
        _, rep = evaluate_kernel_gate('torch_compile_block_unary', lambda: torch.zeros(1), None, device=dev, strict=strict)
        reports = {'torch_compile_block_unary': rep}
        return ({'torch_compile_block_unary': None}, reports) if return_callables else reports

    v_bench = min(int(vocab_size), 8192 if dev.type == 'cpu' else int(vocab_size))
    gen = torch.Generator(device=dev).manual_seed(2026)
    E = torch.randn((v_bench, r), generator=gen, device=dev, dtype=torch.float32)
    W = torch.randn((r, p), generator=gen, device=dev, dtype=torch.float32)
    b = torch.randn((v_bench,), generator=gen, device=dev, dtype=torch.float32)
    H = torch.randn((K, p), generator=gen, device=dev, dtype=torch.float32)

    def ref_unary():
        return block_unary(E, W, b, H)

    compiled_unary = None
    try:
        raw_compiled = torch.compile(block_unary, dynamic=True)

        def compiled_unary(E_in, W_in, b_in, H_in):
            if any(t.requires_grad for t in (E_in, W_in, b_in, H_in)) and torch.is_grad_enabled():
                return block_unary(E_in, W_in, b_in, H_in)
            with torch.no_grad():
                return raw_compiled(E_in, W_in, b_in, H_in)

        _ = compiled_unary(E, W, b, H)

        def cand_unary():
            return compiled_unary(E, W, b, H)
    except Exception:
        compiled_unary = None
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
    reports = {'torch_compile_block_unary': rep}
    callables = {'torch_compile_block_unary': compiled_unary if rep.enabled else None}
    return (callables, reports) if return_callables else reports
