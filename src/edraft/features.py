"""Frozen calibration projections; no evaluation examples are fitted here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os
import torch


@dataclass(frozen=True)
class HiddenProjection:
    matrix: torch.Tensor  # [p, d_h]
    mean: torch.Tensor
    std: torch.Tensor
    metadata: dict

    def __call__(self, h_raw: torch.Tensor) -> torch.Tensor:
        return (h_raw.float() @ self.matrix.to(h_raw.device).T - self.mean.to(h_raw.device)) / self.std.to(h_raw.device)

    def to(self, device) -> 'HiddenProjection':
        return HiddenProjection(self.matrix.to(device), self.mean.to(device), self.std.to(device), self.metadata)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
        torch.save({'matrix': self.matrix.cpu(), 'mean': self.mean.cpu(), 'std': self.std.cpu(), 'metadata': self.metadata}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> 'HiddenProjection':
        return cls(**torch.load(path, map_location='cpu', weights_only=True))


def fit_hidden_projection(batches: Iterable[torch.Tensor], d_h: int, p: int = 256, seed: int = 2026,
                          prompt_ids: list[str] | None = None) -> HiddenProjection:
    """Fixed seeded orthogonal Gaussian projection with streaming train-only moments.

    This is the specification's documented fixed-projection option, not PCA fitted
    to future traces. Moments use a parallel Welford merge in FP64.
    """
    if not 0 < p <= d_h:
        raise ValueError('projection dimension must be in 1..d_h')
    gen = torch.Generator(device='cpu').manual_seed(seed)
    matrix = torch.linalg.qr(torch.randn(d_h, p, generator=gen, dtype=torch.float64), mode='reduced').Q.T.float()
    n = 0
    mean = torch.zeros(p, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for batch in batches:
        x = batch.detach().reshape(-1, d_h).cpu().float().matmul(matrix.T).double()
        if not len(x):
            continue
        bn = len(x)
        bm = x.mean(0)
        bm2 = (x - bm).square().sum(0)
        delta = bm - mean
        total = n + bn
        mean += delta * bn / total
        m2 += bm2 + delta.square() * n * bn / total
        n = total
    if n < 2:
        raise ValueError('at least two calibration hidden vectors are required')
    std = (m2 / n).sqrt().clamp_min(1e-6)
    return HiddenProjection(matrix, mean.float(), std.float(), {
        'method': 'fixed_seeded_orthogonal_gaussian', 'seed': seed, 'p': p,
        'd_h': d_h, 'sample_count': n, 'split': 'train',
        'condition': 'standard_nonthinking', 'prompt_ids': prompt_ids or [],
        'variance_denominator': 'population', 'std_floor': 1e-6,
    })


def fit_token_features(embedding: torch.Tensor, r: int = 64, seed: int = 2026,
                       calibration_token_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Reproducible randomized PCA; calibration-only coordinate standardization."""
    if calibration_token_ids is None or calibration_token_ids.numel() < 2:
        raise ValueError('token feature standardization requires calibration-split token IDs')
    x = embedding.detach().float()
    if not 0 < r <= min(x.shape):
        raise ValueError('invalid token feature rank')
    # PCA uses the frozen embedding source, never test hidden states or tokens.
    centered = x - x.mean(0)
    devices = [x.device.index] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if x.is_cuda:
            torch.cuda.manual_seed(seed)
        _, _, vectors = torch.pca_lowrank(centered, q=min(r + 8, min(x.shape)), center=False, niter=3)
    vectors = vectors[:, :r]
    # Resolve SVD sign ambiguity from the largest loading, not arbitrary backend signs.
    pivots = vectors.abs().argmax(0)
    signs = vectors[pivots, torch.arange(r, device=x.device)].sign()
    vectors = vectors * torch.where(signs == 0, 1, signs)
    projected = centered @ vectors
    calibration = projected[calibration_token_ids.to(x.device)]
    mean, std = calibration.mean(0), calibration.std(0, unbiased=False).clamp_min(1e-6)
    E = ((projected - mean) / std).contiguous()
    return E, {'method': 'randomized_embedding_pca', 'r': r, 'seed': seed,
               'calibration_token_count': calibration_token_ids.numel(),
               'mean': mean.cpu(), 'std': std.cpu(), 'basis': vectors.cpu(),
               'embedding_mean': x.mean(0).cpu()}


def reconstruct_l_unit(h_raw_bf16: torch.Tensor, head: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    logits = h_raw_bf16.float() @ head.float().T
    if bias is not None:
        logits = logits + bias.float()
    return torch.log_softmax(logits, dim=-1)
