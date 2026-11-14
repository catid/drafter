"""Pure-PyTorch spectral and scale-invariant optimizers for dense adapter fitting.

Adapted from `optimizer_experiments` (Polar-Express GramNewtonSchulz, PaceMuon
with decaying iterate EMA and preconditioned pullback, AnchorMuon with SODA
init-anchoring + row-only PMuonEq + NorMuon, and scale-invariant AdamATan2),
with strict `prompt.md` conformance:
- Zero custom CUDA or Triton kernels (pure PyTorch `baddbmm` / elementwise ops).
- 3D lag tensor `A [D, r, r]` is orthogonalized per lag slice as a `[D, r, r]`
  batch of `[r, r]` operators rather than flattened across lags.
- Vector bias `b [V]` never receives implicit weight decay or SODA shrinkage
  unless `bias_weight_decay > 0` is explicitly configured (`prompt.md:597`).
- Sparse tables (`S` / `dS`) remain strictly on SGD + explicit proximal L1.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Callable, Iterable, Iterator

import torch
from torch.optim import Optimizer


_RAW_POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)


def _scale_polar_express_coefficients(
    coefficients: Iterable[Iterable[float]],
    *,
    safety_factor: float = 1.05,
) -> tuple[tuple[float, float, float], ...]:
    rows = [tuple(float(v) for v in row) for row in coefficients]
    out: list[tuple[float, float, float]] = []
    for index, (a, b, c) in enumerate(rows):
        scale = float(safety_factor) if index < len(rows) - 1 else 1.0
        out.append((a / scale, b / (scale**3), c / (scale**5)))
    return tuple(out)


POLAR_EXPRESS_COEFFICIENTS = _scale_polar_express_coefficients(_RAW_POLAR_EXPRESS_COEFFICIENTS)


def is_matrix_like(param: torch.Tensor, *, min_matrix_dim: int = 2) -> bool:
    if not param.is_floating_point() or param.ndim < 2:
        return False
    if min(int(param.shape[-2]), int(param.shape[-1])) >= min_matrix_dim:
        return True
    effective = [int(d) for d in param.shape if int(d) > 1]
    if len(effective) < 2:
        return False
    return min(effective[0], math.prod(effective[1:])) >= min_matrix_dim


def batch_matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    """View a matrix or batched matrix tensor (e.g. W [r, p] or A [D, r, r]) as [B, M, N]."""
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim >= 3 and min(int(tensor.shape[-2]), int(tensor.shape[-1])) >= 2:
        return tensor.reshape(-1, int(tensor.shape[-2]), int(tensor.shape[-1]))
    flat = tensor.reshape(int(tensor.shape[0]), -1)
    return flat.unsqueeze(0)


class GramNewtonSchulz:
    """5-step Polar-Express Gram Newton-Schulz orthogonalizer on [M, N] or [B, M, N]."""

    def __init__(
        self,
        *,
        coefficients: Iterable[Iterable[float]] | None = None,
        epsilon: float = 1e-7,
        reset_iterations: Iterable[int] = (2,),
        compute_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        self.coefficients = tuple(
            tuple(float(v) for v in row) for row in (coefficients or POLAR_EXPRESS_COEFFICIENTS)
        )
        self.epsilon = float(epsilon)
        self.reset_iterations = {int(i) for i in reset_iterations}
        self.compute_dtype = compute_dtype

    def _compute_dtype_for(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype is not None:
            return self.compute_dtype
        return torch.bfloat16 if (x.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32

    @torch.no_grad()
    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.ndim < 2:
            raise ValueError("GramNewtonSchulz expects a tensor with ndim >= 2")
        original_shape = matrix.shape
        original_dtype = matrix.dtype
        x = batch_matrix_view(matrix).to(torch.float32)
        transposed = x.size(-2) > x.size(-1)
        if transposed:
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + self.epsilon)
        x = x.to(self._compute_dtype_for(x))
        if max(x.shape[-2], x.shape[-1]) > min(x.shape[-2], x.shape[-1]):
            x = self._gram_recurrence(x)
        else:
            x = self._standard_recurrence(x)
        if transposed:
            x = x.mT
        return x.to(original_dtype).reshape(original_shape)

    def _standard_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        for a, b, c in self.coefficients:
            gram = x @ x.mT
            poly = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
            x = torch.baddbmm(x, poly, x, beta=a)
        return x

    def _gram_recurrence(self, x: torch.Tensor) -> torch.Tensor:
        gram = x @ x.mT
        eye = torch.eye(gram.size(-1), device=x.device, dtype=x.dtype).expand(gram.size(0), -1, -1).contiguous()
        q: torch.Tensor | None = None
        for i, (a, b, c) in enumerate(self.coefficients):
            if i in self.reset_iterations and i != 0:
                if q is None:
                    raise RuntimeError("Gram Newton-Schulz reset reached without an inverse estimate")
                x = q @ x
                gram = x @ x.mT
                q = None
            z = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
            if i == 0 or i in self.reset_iterations:
                q = z + a * eye
            else:
                if q is None:
                    raise RuntimeError("Gram Newton-Schulz inverse estimate was not initialized")
                q = torch.baddbmm(q, q, z, beta=a)
            if i < len(self.coefficients) - 1 and i + 1 not in self.reset_iterations:
                rz = torch.baddbmm(gram, gram, z, beta=a)
                gram = torch.baddbmm(rz, z, rz, beta=a)
        if q is None:
            raise RuntimeError("Gram Newton-Schulz finished without an inverse estimate")
        return q @ x


def _normuon_row_normalize(
    update: torch.Tensor,
    second_momentum: torch.Tensor,
    *,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    old_norm = update.norm(dim=(-2, -1), keepdim=True)
    row_power = update.square().mean(dim=-1, keepdim=True)
    second_momentum.lerp_(row_power, 1.0 - beta2)
    out = update * torch.rsqrt(second_momentum + eps)
    return out * (old_norm / (out.norm(dim=(-2, -1), keepdim=True) + eps))


class AdamATan2(Optimizer):
    """Scale-invariant AdamATan2 optimizer (`atan2(m_hat, sqrt(v_hat) + eps)`) with decoupled WD."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0 or eps <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid AdamATan2 learning rate, eps, or weight_decay")
        if len(betas) != 2 or not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must contain two values in [0, 1)")
        defaults = dict(lr=float(lr), betas=(float(betas[0]), float(betas[1])), eps=float(eps),
                        weight_decay=float(weight_decay))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                g = p.grad.detach().to(torch.float32)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] = int(state["step"]) + 1
                step = int(state["step"])
                exp_avg = state["exp_avg"].to(device=p.device, dtype=torch.float32)
                exp_avg_sq = state["exp_avg_sq"].to(device=p.device, dtype=torch.float32)
                state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                bias1 = max(1.0 - beta1**step, 1e-16)
                bias2 = max(1.0 - beta2**step, 1e-16)
                denom = (exp_avg_sq / bias2).sqrt().add_(eps)
                update = torch.atan2(exp_avg / bias1, denom)
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss


class PaceMuon(Optimizer):
    """Muon / PaceMuon with batched GramNewtonSchulz, optional PMuonEq, and decaying iterate EMA."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 8e-3,
        fallback_lr: float | None = None,
        fallback_mode: str = "atan2",
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        fallback_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.0,
        pmuoneq_eps: float = 1e-6,
        ema_enabled: bool = True,
        pullback_c: float = 0.05,
        kappa: float = 0.5,
        pace_precond: str = "adam",
        pace_beta2: float = 0.99,
        pace_eps: float = 1e-8,
        min_decay: float = 0.002,
        min_matrix_dim: int = 2,
    ) -> None:
        fallback_lr = lr if fallback_lr is None else fallback_lr
        if lr < 0.0 or fallback_lr < 0.0 or weight_decay < 0.0:
            raise ValueError("learning rates and weight_decay must be non-negative")
        if fallback_mode not in {"adamw", "atan2"}:
            raise ValueError("fallback_mode must be 'adamw' or 'atan2'")
        if pace_precond not in {"none", "adam", "row"}:
            raise ValueError("pace_precond must be 'none', 'adam', or 'row'")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if row_gamma < 0.0 or pullback_c < 0.0 or kappa < 0.0:
            raise ValueError("row_gamma, pullback_c, and kappa must be non-negative")
        prepared = self._prepare_groups(
            params,
            lr=float(lr),
            fallback_lr=float(fallback_lr),
            fallback_mode=fallback_mode,
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            fallback_betas=(float(fallback_betas[0]), float(fallback_betas[1])),
            eps=float(eps),
            pmuoneq_beta=float(pmuoneq_beta),
            row_gamma=float(row_gamma),
            pmuoneq_eps=float(pmuoneq_eps),
            min_matrix_dim=int(min_matrix_dim),
        )
        super().__init__(prepared, defaults={})
        self._orthogonalizer = GramNewtonSchulz()
        self.ema_enabled = bool(ema_enabled)
        self.pullback_c = float(pullback_c) if self.ema_enabled else 0.0
        self.kappa = float(kappa)
        self.pace_precond = str(pace_precond)
        self.pace_beta2 = float(pace_beta2)
        self.pace_eps = float(pace_eps)
        self.min_decay = float(min_decay)
        self.step_index = 0
        self._swapped = False

    @staticmethod
    def _prepare_groups(
        params: Iterable[Any],
        *,
        lr: float,
        fallback_lr: float,
        fallback_mode: str,
        momentum: float,
        weight_decay: float,
        fallback_betas: tuple[float, float],
        eps: float,
        pmuoneq_beta: float,
        row_gamma: float,
        pmuoneq_eps: float,
        min_matrix_dim: int,
    ) -> list[dict[str, Any]]:
        items = list(params)
        if not items:
            raise ValueError("PaceMuon received an empty parameter list")
        groups: list[dict[str, Any]] = []
        seen: set[int] = set()
        if isinstance(items[0], dict):
            for raw in items:
                ps = [p for p in raw["params"] if id(p) not in seen]
                for p in ps:
                    seen.add(id(p))
                if not ps:
                    continue
                wd = float(raw.get("weight_decay", weight_decay))
                explicit = raw.get("use_muon")
                if explicit is not None:
                    use_m = bool(explicit)
                    groups.append(
                        {
                            "params": ps,
                            "use_muon": use_m,
                            "lr": float(raw.get("lr", lr if use_m else fallback_lr)),
                            "base_lr": float(raw.get("lr", lr if use_m else fallback_lr)),
                            "weight_decay": wd,
                            "momentum": float(raw.get("momentum", momentum)),
                            "fallback_mode": str(raw.get("fallback_mode", fallback_mode)),
                            "betas": tuple(raw.get("betas", fallback_betas)),
                            "eps": float(raw.get("eps", eps)),
                            "pmuoneq_beta": float(raw.get("pmuoneq_beta", pmuoneq_beta)),
                            "row_gamma": float(raw.get("row_gamma", row_gamma)),
                            "pmuoneq_eps": float(raw.get("pmuoneq_eps", pmuoneq_eps)),
                        }
                    )
                    continue
                mat = [p for p in ps if is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
                vec = [p for p in ps if not is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
                if mat:
                    base = float(raw.get("lr", lr))
                    groups.append(
                        {
                            "params": mat,
                            "use_muon": True,
                            "lr": base,
                            "base_lr": base,
                            "weight_decay": wd,
                            "momentum": float(raw.get("momentum", momentum)),
                            "pmuoneq_beta": float(raw.get("pmuoneq_beta", pmuoneq_beta)),
                            "row_gamma": float(raw.get("row_gamma", row_gamma)),
                            "pmuoneq_eps": float(raw.get("pmuoneq_eps", pmuoneq_eps)),
                        }
                    )
                if vec:
                    base_v = float(raw.get("fallback_lr", raw.get("lr", fallback_lr)))
                    groups.append(
                        {
                            "params": vec,
                            "use_muon": False,
                            "lr": base_v,
                            "base_lr": base_v,
                            "weight_decay": wd,
                            "fallback_mode": str(raw.get("fallback_mode", fallback_mode)),
                            "betas": tuple(raw.get("betas", fallback_betas)),
                            "eps": float(raw.get("eps", eps)),
                        }
                    )
            return groups

        mat = [p for p in items if is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
        vec = [p for p in items if not is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
        if mat:
            groups.append(
                {
                    "params": mat,
                    "use_muon": True,
                    "lr": lr,
                    "base_lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "pmuoneq_beta": pmuoneq_beta,
                    "row_gamma": row_gamma,
                    "pmuoneq_eps": pmuoneq_eps,
                }
            )
        if vec:
            groups.append(
                {
                    "params": vec,
                    "use_muon": False,
                    "lr": fallback_lr,
                    "base_lr": fallback_lr,
                    "weight_decay": 0.0,
                    "fallback_mode": fallback_mode,
                    "betas": fallback_betas,
                    "eps": eps,
                }
            )
        return groups

    def _decay_t(self, step: int) -> float:
        return max((1.0 + float(step)) ** (-self.kappa), self.min_decay)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        if self._swapped:
            raise RuntimeError("PaceMuon.step() called while weights are swapped to EMA")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_index += 1
        decay = self._decay_t(self.step_index)
        if self.ema_enabled:
            self._pace_before_step()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_fallback_group(group)

        if self.ema_enabled:
            self._pace_after_step(decay)
        return loss

    def _pace_before_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                ema = state.get("pace_ema")
                if ema is None or ema.shape != p.shape or ema.device != p.device:
                    ema = state["pace_ema"] = (
                        ema.to(device=p.device, dtype=torch.float32)
                        if isinstance(ema, torch.Tensor) and ema.shape == p.shape
                        else p.detach().to(torch.float32, copy=True)
                    )
                if self.pullback_c <= 0.0:
                    continue
                pre = state.get("pace_pre")
                if pre is None or pre.shape != p.shape or pre.device != p.device:
                    pre = state["pace_pre"] = torch.empty_like(p, dtype=torch.float32)
                pre.copy_(p.detach())
                g = p.grad.detach().to(torch.float32)
                if self.pace_precond == "adam":
                    v = state.get("pace_v")
                    if v is None or v.shape != p.shape or v.device != p.device:
                        v = state["pace_v"] = (
                            v.to(device=p.device, dtype=torch.float32)
                            if isinstance(v, torch.Tensor) and v.shape == p.shape
                            else torch.zeros_like(p, dtype=torch.float32)
                        )
                    v.mul_(self.pace_beta2).addcmul_(g, g, value=1.0 - self.pace_beta2)
                elif self.pace_precond == "row" and group.get("use_muon", False):
                    g_mv = batch_matrix_view(g)
                    v_row = state.get("pace_v_row")
                    row_shape = (g_mv.shape[0], g_mv.shape[1])
                    if v_row is None or v_row.shape != row_shape or v_row.device != p.device:
                        v_row = state["pace_v_row"] = (
                            v_row.to(device=p.device, dtype=torch.float32)
                            if isinstance(v_row, torch.Tensor) and v_row.shape == row_shape
                            else torch.zeros(row_shape, device=p.device, dtype=torch.float32)
                        )
                    v_row.mul_(self.pace_beta2).add_(g_mv.square().mean(dim=-1), alpha=1.0 - self.pace_beta2)

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        wd = float(group.get("weight_decay", 0.0))
        beta = float(group["momentum"])
        row_gamma = float(group.get("row_gamma", 0.0))
        pmuoneq_beta = float(group.get("pmuoneq_beta", 0.90))
        pmuoneq_eps = float(group.get("pmuoneq_eps", 1e-6))
        for p in group["params"]:
            if p.grad is None:
                continue
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            g = p.grad.detach().to(torch.float32)
            state = self.state[p]
            mom = state.get("momentum_buffer")
            if mom is None or mom.shape != p.shape or mom.device != p.device:
                mom = state["momentum_buffer"] = (
                    mom.to(device=p.device, dtype=torch.float32)
                    if isinstance(mom, torch.Tensor) and mom.shape == p.shape
                    else torch.zeros_like(p, dtype=torch.float32)
                )
            mom.lerp_(g, 1.0 - beta)
            source = torch.lerp(g, mom, beta)
            mat = batch_matrix_view(source)
            if row_gamma > 0.0:
                g_mat = batch_matrix_view(g)
                row_shape = (g_mat.shape[0], g_mat.shape[1])
                row_ema = state.get("pmuoneq_row_ema")
                if row_ema is None or row_ema.shape != row_shape or row_ema.device != p.device:
                    row_ema = state["pmuoneq_row_ema"] = (
                        row_ema.to(device=p.device, dtype=torch.float32)
                        if isinstance(row_ema, torch.Tensor) and row_ema.shape == row_shape
                        else torch.ones(row_shape, device=p.device, dtype=torch.float32)
                    )
                row_ema.mul_(pmuoneq_beta).add_(g_mat.square().mean(dim=-1), alpha=1.0 - pmuoneq_beta)
                row_factor = (row_ema + pmuoneq_eps).pow(-row_gamma)
                mat = mat * row_factor.unsqueeze(-1)
            update = self._orthogonalizer(mat)
            scale = 0.2 * math.sqrt(max(int(update.shape[-2]), int(update.shape[-1])))
            p.add_((update * scale).reshape_as(p).to(p.dtype), alpha=-lr)

    def _step_fallback_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group.get("weight_decay", 0.0))
        mode = str(group.get("fallback_mode", "atan2"))
        for p in group["params"]:
            if p.grad is None:
                continue
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            g = p.grad.detach().to(torch.float32)
            state = self.state[p]
            if "adam_step" not in state:
                state["adam_step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["adam_step"] = int(state["adam_step"]) + 1
            step = int(state["adam_step"])
            exp_avg = state["exp_avg"].to(device=p.device, dtype=torch.float32)
            exp_avg_sq = state["exp_avg_sq"].to(device=p.device, dtype=torch.float32)
            state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq
            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            mhat = exp_avg / max(1.0 - beta1**step, 1e-16)
            vhat = exp_avg_sq / max(1.0 - beta2**step, 1e-16)
            denom = vhat.sqrt().add_(eps)
            if mode == "atan2":
                p.add_(torch.atan2(mhat, denom).to(p.dtype), alpha=-lr)
            else:
                p.addcdiv_(mhat.to(p.dtype), denom.to(p.dtype), value=-lr)

    def _pace_after_step(self, decay: float) -> None:
        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                ema = state.get("pace_ema")
                if ema is None:
                    continue
                if self.pullback_c > 0.0 and "pace_pre" in state:
                    pre = state["pace_pre"]
                    if self.pace_precond == "adam" and "pace_v" in state:
                        bias2 = max(1.0 - self.pace_beta2**self.step_index, 1e-16)
                        lam = (state["pace_v"] / bias2).sqrt().add_(self.pace_eps).reciprocal_()
                        lam.mul_(lr * self.pullback_c * decay).clamp_(max=1.0)
                        diff = (ema - pre).mul_(lam)
                    elif self.pace_precond == "row" and "pace_v_row" in state:
                        bias2 = max(1.0 - self.pace_beta2**self.step_index, 1e-16)
                        lam_row = (state["pace_v_row"] / bias2).sqrt().add_(self.pace_eps).reciprocal_()
                        lam_row.mul_(lr * self.pullback_c * decay).clamp_(max=1.0)
                        diff = (batch_matrix_view(ema - pre) * lam_row.unsqueeze(-1)).reshape_as(p)
                    else:
                        lam_scalar = min(lr * self.pullback_c * decay, 1.0)
                        diff = (ema - pre).mul_(lam_scalar)
                    p.add_(diff.to(p.dtype))
                ema.mul_(1.0 - decay).add_(p.detach().to(torch.float32), alpha=decay)

    @torch.no_grad()
    def swap_to_ema(self) -> None:
        if self._swapped or not self.ema_enabled:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "pace_ema" not in state:
                    continue
                backup = state.get("pace_live_backup")
                if backup is None or backup.shape != p.shape or backup.device != p.device:
                    backup = state["pace_live_backup"] = torch.empty_like(p)
                backup.copy_(p.detach())
                p.copy_(state["pace_ema"].to(device=p.device, dtype=p.dtype))
        self._swapped = True

    @torch.no_grad()
    def swap_to_live(self) -> None:
        if not self._swapped:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "pace_live_backup" not in state:
                    continue
                p.copy_(state["pace_live_backup"].to(device=p.device, dtype=p.dtype))
        self._swapped = False

    @contextmanager
    def use_ema_weights(self) -> Iterator[None]:
        self.swap_to_ema()
        try:
            yield
        finally:
            self.swap_to_live()

    @torch.no_grad()
    def commit_ema(self) -> None:
        """Permanently copy iterate EMA weights into parameters at stage completion."""
        if self._swapped:
            self._swapped = False
            return
        if not self.ema_enabled:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state and "pace_ema" in state:
                    p.copy_(state["pace_ema"].to(device=p.device, dtype=p.dtype))

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["pace_global"] = {
            "step_index": self.step_index,
            "ema_enabled": self.ema_enabled,
            "pullback_c": self.pullback_c,
            "kappa": self.kappa,
            "pace_precond": self.pace_precond,
            "pace_beta2": self.pace_beta2,
            "pace_eps": self.pace_eps,
            "min_decay": self.min_decay,
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        copy_sd = dict(state_dict)
        g = copy_sd.pop("pace_global", {})
        super().load_state_dict(copy_sd)
        self.step_index = int(g.get("step_index", 0))
        self.ema_enabled = bool(g.get("ema_enabled", self.ema_enabled))
        self.pullback_c = float(g.get("pullback_c", self.pullback_c))
        self.kappa = float(g.get("kappa", self.kappa))
        self.pace_precond = str(g.get("pace_precond", self.pace_precond))
        self.pace_beta2 = float(g.get("pace_beta2", self.pace_beta2))
        self.pace_eps = float(g.get("pace_eps", self.pace_eps))
        self.min_decay = float(g.get("min_decay", self.min_decay))
        self._swapped = False


class AnchorMuon(Optimizer):
    """AnchorMuon: SODA init-anchor pull + row-only PMuonEq + GramNS + NorMuon + AdamATan2 fallback."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 8e-3,
        fallback_lr: float | None = None,
        momentum: float = 0.95,
        fallback_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        soda_lambda_scale: float = 0.05,
        soda_lambda_power: float = 1.0,
        pmuoneq_beta: float = 0.90,
        row_gamma: float = 0.35,
        pmuoneq_eps: float = 1e-6,
        normuon_beta2: float = 0.93,
        normuon_eps: float = 1e-10,
        ema_enabled: bool = False,
        kappa: float = 0.5,
        min_decay: float = 0.002,
        min_matrix_dim: int = 2,
    ) -> None:
        fallback_lr = lr if fallback_lr is None else fallback_lr
        if lr < 0.0 or fallback_lr < 0.0 or soda_lambda_scale < 0.0 or soda_lambda_power <= 0.0:
            raise ValueError("invalid AnchorMuon learning rate or SODA parameters")
        items = list(params)
        if not items:
            raise ValueError("AnchorMuon received an empty parameter list")
        groups: list[dict[str, Any]] = []
        seen: set[int] = set()
        if isinstance(items[0], dict):
            for raw in items:
                ps = [p for p in raw["params"] if id(p) not in seen]
                for p in ps:
                    seen.add(id(p))
                if not ps:
                    continue
                wd = float(raw.get("weight_decay", 0.0))
                mat = [p for p in ps if is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
                vec = [p for p in ps if not is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
                if mat:
                    base = float(raw.get("lr", lr))
                    groups.append(
                        {
                            "params": mat,
                            "use_matrix_update": True,
                            "lr": base,
                            "base_lr": base,
                            "weight_decay": wd,
                            "soda_enabled": bool(raw.get("soda_enabled", True)),
                            "momentum": float(raw.get("momentum", momentum)),
                            "pmuoneq_beta": float(raw.get("pmuoneq_beta", pmuoneq_beta)),
                            "row_gamma": float(raw.get("row_gamma", row_gamma)),
                            "pmuoneq_eps": float(raw.get("pmuoneq_eps", pmuoneq_eps)),
                            "normuon_beta2": float(raw.get("normuon_beta2", normuon_beta2)),
                            "normuon_eps": float(raw.get("normuon_eps", normuon_eps)),
                        }
                    )
                if vec:
                    base_v = float(raw.get("fallback_lr", raw.get("lr", fallback_lr)))
                    # Respect prompt.md:597: vector `b` gets SODA pull ONLY if weight_decay > 0 or explicitly set.
                    groups.append(
                        {
                            "params": vec,
                            "use_matrix_update": False,
                            "lr": base_v,
                            "base_lr": base_v,
                            "weight_decay": wd,
                            "soda_enabled": bool(raw.get("soda_enabled", wd > 0.0)),
                            "betas": tuple(raw.get("betas", fallback_betas)),
                            "eps": float(raw.get("eps", eps)),
                        }
                    )
        else:
            mat = [p for p in items if is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
            vec = [p for p in items if not is_matrix_like(p, min_matrix_dim=min_matrix_dim)]
            if mat:
                groups.append(
                    {
                        "params": mat,
                        "use_matrix_update": True,
                        "lr": float(lr),
                        "base_lr": float(lr),
                        "weight_decay": 0.0,
                        "soda_enabled": True,
                        "momentum": float(momentum),
                        "pmuoneq_beta": float(pmuoneq_beta),
                        "row_gamma": float(row_gamma),
                        "pmuoneq_eps": float(pmuoneq_eps),
                        "normuon_beta2": float(normuon_beta2),
                        "normuon_eps": float(normuon_eps),
                    }
                )
            if vec:
                groups.append(
                    {
                        "params": vec,
                        "use_matrix_update": False,
                        "lr": float(fallback_lr),
                        "base_lr": float(fallback_lr),
                        "weight_decay": 0.0,
                        "soda_enabled": False,
                        "betas": (float(fallback_betas[0]), float(fallback_betas[1])),
                        "eps": float(eps),
                    }
                )
        super().__init__(groups, defaults={})
        self._orthogonalizer = GramNewtonSchulz()
        self.soda_lambda_scale = float(soda_lambda_scale)
        self.soda_lambda_power = float(soda_lambda_power)
        self.ema_enabled = bool(ema_enabled)
        self.kappa = float(kappa)
        self.min_decay = float(min_decay)
        self.step_index = 0
        self._swapped = False

    def _soda_weight(self, step: int) -> float:
        if self.soda_lambda_scale <= 0.0:
            return 0.0
        return min(1.0, self.soda_lambda_scale / (float(step + 1) ** self.soda_lambda_power))

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        if self._swapped:
            raise RuntimeError("AnchorMuon.step() called while weights are swapped to EMA")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_index += 1
        soda_w = self._soda_weight(self.step_index)
        decay = max((1.0 + float(self.step_index)) ** (-self.kappa), self.min_decay)

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            soda_on = bool(group.get("soda_enabled", False)) and soda_w > 0.0
            if group.get("use_matrix_update", False):
                beta_m = float(group["momentum"])
                beta_p = float(group["pmuoneq_beta"])
                row_gamma = float(group["row_gamma"])
                pmuoneq_eps = float(group["pmuoneq_eps"])
                normuon_beta2 = float(group["normuon_beta2"])
                normuon_eps = float(group["normuon_eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if self.ema_enabled and ("pace_ema" not in state or state["pace_ema"].shape != p.shape):
                        state["pace_ema"] = p.detach().to(torch.float32, copy=True)
                    if soda_on:
                        anchor = state.get("soda_z0")
                        if anchor is None or anchor.shape != p.shape or anchor.device != p.device:
                            anchor = state["soda_z0"] = (
                                anchor.to(device=p.device, dtype=p.dtype)
                                if isinstance(anchor, torch.Tensor) and anchor.shape == p.shape
                                else p.detach().clone()
                            )
                        p.lerp_(anchor, soda_w)
                    elif wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    g = p.grad.detach().to(torch.float32)
                    mom = state.get("momentum_buffer")
                    if mom is None or mom.shape != p.shape or mom.device != p.device:
                        mom = state["momentum_buffer"] = (
                            mom.to(device=p.device, dtype=torch.float32)
                            if isinstance(mom, torch.Tensor) and mom.shape == p.shape
                            else torch.zeros_like(p, dtype=torch.float32)
                        )
                    mom.lerp_(g, 1.0 - beta_m)
                    source = batch_matrix_view(torch.lerp(g, mom, beta_m))
                    g_mv = batch_matrix_view(g)
                    row_shape = (g_mv.shape[0], g_mv.shape[1])
                    row_ema = state.get("pmuoneq_row_ema")
                    if row_ema is None or row_ema.shape != row_shape or row_ema.device != p.device:
                        row_ema = state["pmuoneq_row_ema"] = (
                            row_ema.to(device=p.device, dtype=torch.float32)
                            if isinstance(row_ema, torch.Tensor) and row_ema.shape == row_shape
                            else torch.ones(row_shape, device=p.device, dtype=torch.float32)
                        )
                    second = state.get("normuon_second")
                    sec_shape = (g_mv.shape[0], g_mv.shape[1], 1)
                    if second is None or second.shape != sec_shape or second.device != p.device:
                        second = state["normuon_second"] = (
                            second.to(device=p.device, dtype=torch.float32)
                            if isinstance(second, torch.Tensor) and second.shape == sec_shape
                            else torch.zeros(sec_shape, device=p.device, dtype=torch.float32)
                        )
                    row_ema.mul_(beta_p).add_(g_mv.square().mean(dim=-1), alpha=1.0 - beta_p)
                    if row_gamma > 0.0:
                        source = source * (row_ema + pmuoneq_eps).pow(-row_gamma).unsqueeze(-1)
                    update = self._orthogonalizer(source)
                    update = update * (0.2 * math.sqrt(max(int(update.shape[-2]), int(update.shape[-1]))))
                    update = _normuon_row_normalize(update, second, beta2=normuon_beta2, eps=normuon_eps)
                    p.add_(update.reshape_as(p).to(p.dtype), alpha=-lr)
                    if self.ema_enabled:
                        ema = state["pace_ema"].to(device=p.device, dtype=torch.float32)
                        state["pace_ema"] = ema
                        ema.mul_(1.0 - decay).add_(p.detach().to(torch.float32), alpha=decay)
            else:
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if self.ema_enabled and ("pace_ema" not in state or state["pace_ema"].shape != p.shape):
                        state["pace_ema"] = p.detach().to(torch.float32, copy=True)
                    if soda_on:
                        anchor = state.get("soda_z0")
                        if anchor is None or anchor.shape != p.shape or anchor.device != p.device:
                            anchor = state["soda_z0"] = (
                                anchor.to(device=p.device, dtype=p.dtype)
                                if isinstance(anchor, torch.Tensor) and anchor.shape == p.shape
                                else p.detach().clone()
                            )
                        p.lerp_(anchor, soda_w)
                    elif wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    g = p.grad.detach().to(torch.float32)
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"] = int(state["step"]) + 1
                    st = int(state["step"])
                    exp_avg = state["exp_avg"].to(device=p.device, dtype=torch.float32)
                    exp_avg_sq = state["exp_avg_sq"].to(device=p.device, dtype=torch.float32)
                    state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    mhat = exp_avg / max(1.0 - beta1**st, 1e-16)
                    vhat = exp_avg_sq / max(1.0 - beta2**st, 1e-16)
                    p.add_(torch.atan2(mhat, vhat.sqrt().add_(eps)).to(p.dtype), alpha=-lr)
                    if self.ema_enabled:
                        ema = state["pace_ema"].to(device=p.device, dtype=torch.float32)
                        state["pace_ema"] = ema
                        ema.mul_(1.0 - decay).add_(p.detach().to(torch.float32), alpha=decay)
        return loss

    @torch.no_grad()
    def swap_to_ema(self) -> None:
        if self._swapped or not self.ema_enabled:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "pace_ema" not in state:
                    continue
                backup = state.get("pace_live_backup")
                if backup is None or backup.shape != p.shape or backup.device != p.device:
                    backup = state["pace_live_backup"] = torch.empty_like(p)
                backup.copy_(p.detach())
                p.copy_(state["pace_ema"].to(device=p.device, dtype=p.dtype))
        self._swapped = True

    @torch.no_grad()
    def swap_to_live(self) -> None:
        if not self._swapped:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "pace_live_backup" not in state:
                    continue
                p.copy_(state["pace_live_backup"].to(device=p.device, dtype=p.dtype))
        self._swapped = False

    @contextmanager
    def use_ema_weights(self) -> Iterator[None]:
        self.swap_to_ema()
        try:
            yield
        finally:
            self.swap_to_live()

    @torch.no_grad()
    def commit_ema(self) -> None:
        """Permanently copy iterate EMA weights into parameters at stage completion."""
        if self._swapped:
            self._swapped = False
            return
        if not self.ema_enabled:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state and "pace_ema" in state:
                    p.copy_(state["pace_ema"].to(device=p.device, dtype=p.dtype))

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["anchor_global"] = {
            "step_index": self.step_index,
            "soda_lambda_scale": self.soda_lambda_scale,
            "soda_lambda_power": self.soda_lambda_power,
            "ema_enabled": self.ema_enabled,
            "kappa": self.kappa,
            "min_decay": self.min_decay,
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        copy_sd = dict(state_dict)
        g = copy_sd.pop("anchor_global", {})
        super().load_state_dict(copy_sd)
        self.step_index = int(g.get("step_index", 0))
        self.soda_lambda_scale = float(g.get("soda_lambda_scale", self.soda_lambda_scale))
        self.soda_lambda_power = float(g.get("soda_lambda_power", self.soda_lambda_power))
        self.ema_enabled = bool(g.get("ema_enabled", self.ema_enabled))
        self.kappa = float(g.get("kappa", self.kappa))
        self.min_decay = float(g.get("min_decay", self.min_decay))
        self._swapped = False


def compute_lr_multiplier(
    step_one_based: int,
    total_steps: int,
    *,
    schedule: str = "constant",
    warmup_fraction: float = 0.05,
    decay_fraction: float = 0.20,
    min_lr_ratio: float = 0.10,
) -> float:
    """Return LR multiplier in [min_lr_ratio, 1.0] for constant, WSD, or cosine schedules."""
    if schedule == "constant" or total_steps <= 1:
        return 1.0
    step = max(1, min(int(step_one_based), int(total_steps)))
    warmup_steps = max(0, int(round(total_steps * warmup_fraction)))
    if warmup_steps > 0 and step <= warmup_steps:
        return max(min_lr_ratio, float(step) / float(warmup_steps))
    if schedule == "wsd":
        decay_steps = max(1, int(round(total_steps * decay_fraction)))
        decay_start = max(warmup_steps, total_steps - decay_steps)
        if step <= decay_start:
            return 1.0
        progress = float(step - decay_start) / float(max(1, total_steps - decay_start))
        return 1.0 - (1.0 - min_lr_ratio) * min(1.0, max(0.0, progress))
    if schedule == "cosine":
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    raise ValueError(f"unknown lr schedule: {schedule}")


def make_online_lr_schedule(
    base_lr: float,
    *,
    schedule: str = "constant",
    warmup_steps: int = 4,
    total_steps: int = 128,
    min_lr_ratio: float = 0.10,
) -> Callable[[int], float]:
    """Build a 1-based `u -> lr_u` schedule for `OnlineState` (`LazySignedHash.cum_thresh[u]`)."""
    if base_lr < 0.0:
        raise ValueError("base_lr must be non-negative")
    if schedule not in {"constant", "warmup_inv_sqrt", "wsd"}:
        raise ValueError(f"unknown online lr schedule: {schedule}")

    def _schedule(u: int) -> float:
        step = max(1, int(u))
        if schedule == "constant":
            return float(base_lr)
        if schedule == "warmup_inv_sqrt":
            w = max(1, int(warmup_steps))
            if step <= w:
                return float(base_lr) * (float(step) / float(w))
            return float(base_lr) * math.sqrt(float(w) / float(step))
        mult = compute_lr_multiplier(
            step,
            max(1, int(total_steps)),
            schedule="wsd",
            warmup_fraction=float(warmup_steps) / float(max(1, total_steps)),
            min_lr_ratio=min_lr_ratio,
        )
        return float(base_lr) * mult

    return _schedule


__all__ = [
    "POLAR_EXPRESS_COEFFICIENTS",
    "GramNewtonSchulz",
    "AdamATan2",
    "PaceMuon",
    "AnchorMuon",
    "is_matrix_like",
    "batch_matrix_view",
    "compute_lr_multiplier",
    "make_online_lr_schedule",
]
