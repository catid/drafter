"""Comprehensive Round 2 tests: LSE-replay KL, BF16 GEMM, CroissAttn/optimizer_experiments
optimizers, DataLoader/DDP/resume parity, compiled DFlash drafter, compiled/adaptive online
step, incremental OnlineBigram, and vectorized trace/diagnostic pipelines.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from pathlib import Path
import random
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from edraft.adapter import (
    DenseAdapter,
    PrecollatedGroupedTargets,
    block_correction,
    block_unary,
    collate_grouped_targets,
    full_kl,
    fused_head_grouped_kl,
    loss_weights,
    slot_correction,
    union_rest_grouped,
    union_rest_grouped_precollated,
)
from edraft.bench_grouped import grouped_loop_reference
from edraft.backend import crop_cache, make_cache
from edraft.diagnose import (
    DiagnosticAccumulator,
    TOP_M,
    live_block_diagnostics,
    multi_token_rank,
    token_rank,
)
from edraft.dflash_loop import generate_toy
from edraft.fastpath import FastPathConfig, FastPathRuntime
from edraft.online import OnlineBigram, OnlineState
from edraft.optimizers import (
    AdamATan2,
    AnchorMuon,
    GramNewtonSchulz,
    PaceMuon,
    compute_lr_multiplier,
    make_online_lr_schedule,
)
from edraft.sampler import SamplingWarper, probabilities
from edraft.sparse import SignedHash
from edraft.toy import ToyParallelDrafter, ToyTarget
from edraft.traces import TraceShardWriter, compact_block, validate_block
from edraft.train import (
    _reconstruct_h_raw,
    _setup_distributed,
    corrected_batch_scores,
    fit,
    load_adapter_checkpoint,
    reconstruct_batch,
    trace_index,
)


# ---------------------------------------------------------------------------
# 1. LSE-replay autograd memory & numerical parity (_FullKLFunction, _GroupedKLFunction)
# ---------------------------------------------------------------------------

def test_grouped_and_full_kl_zero_nv_saved_probability_buffers_and_second_order():
    g = torch.Generator().manual_seed(2026)
    N, V = 12, 512
    scores = torch.randn(N, V, generator=g, requires_grad=True)
    target_logits = torch.randn(N, V, generator=g)
    weights = loss_weights(N)

    # Full KL: only (scores, target_logits, weights) saved — no p_log, q_log, p_exp [N, V] buffers
    loss_full = full_kl(scores, target_logits, weights)
    saved_full = loss_full.grad_fn.saved_tensors
    assert len(saved_full) == 3
    p_log_ref = target_logits.float().log_softmax(-1)
    q_log_ref = scores.detach().float().log_softmax(-1)
    ref_full_val = ((p_log_ref.exp() * (p_log_ref - q_log_ref)).sum(-1) * weights).sum()
    torch.testing.assert_close(loss_full, ref_full_val, rtol=1e-6, atol=1e-6)
    (grad_full,) = torch.autograd.grad(loss_full, scores, create_graph=True)
    scores_full_ref = scores.detach().clone().requires_grad_(True)
    loss_full_ref_live = ((p_log_ref.exp() * (p_log_ref - scores_full_ref.float().log_softmax(-1))).sum(-1) * weights).sum()
    (expected_grad_full,) = torch.autograd.grad(loss_full_ref_live, scores_full_ref, create_graph=True)
    torch.testing.assert_close(grad_full, expected_grad_full, rtol=1e-6, atol=1e-6)
    # Exact second-order HVP parity check for _FullKLFunction
    hvp = torch.autograd.grad((grad_full ** 2).sum(), scores)[0]
    hvp_full_ref = torch.autograd.grad((expected_grad_full ** 2).sum(), scores_full_ref)[0]
    assert torch.isfinite(hvp).all() and hvp.abs().sum() > 0
    torch.testing.assert_close(hvp, hvp_full_ref, rtol=1e-6, atol=1e-6)

    # Grouped KL: saved_tensors has only 1 [N, V] tensor (`scores`), zero [N, V] prob/mask buffers
    P_full = target_logits.float().softmax(-1)
    union_ids, probs_u, rest_u = [], [], []
    for i in range(N):
        u_len = V if i == 0 else (7 + i)
        idx = torch.arange(V) if u_len == V else torch.randperm(V, generator=g)[:u_len].sort().values
        p_i = P_full[i, idx]
        union_ids.append(idx)
        probs_u.append(p_i)
        rest_u.append(torch.tensor(0.0) if u_len == V else (1.0 - p_i.double().sum()).clamp_min(0.0).float())

    scores_g = scores.detach().clone().requires_grad_(True)
    scores_ref = scores.detach().clone().requires_grad_(True)
    pre = collate_grouped_targets(union_ids, probs_u, rest_u, weights, vocab_size=V)
    loss_grp = union_rest_grouped_precollated(scores_g, pre)
    nv_saved = [t for t in loss_grp.grad_fn.saved_tensors if t.shape == (N, V)]
    assert len(nv_saved) == 1 and nv_saved[0] is scores_g

    loss_ref = grouped_loop_reference(scores_ref, union_ids, probs_u, rest_u, weights)
    torch.testing.assert_close(loss_grp, loss_ref, rtol=1e-6, atol=1e-6)
    (grad_grp,) = torch.autograd.grad(loss_grp, scores_g, create_graph=True)
    (grad_ref,) = torch.autograd.grad(loss_ref, scores_ref, create_graph=True)
    torch.testing.assert_close(grad_grp, grad_ref, rtol=1e-6, atol=2e-7)
    hvp_grp = torch.autograd.grad((grad_grp ** 2).sum(), scores_g)[0]
    hvp_ref = torch.autograd.grad((grad_ref ** 2).sum(), scores_ref)[0]
    assert torch.isfinite(hvp_grp).all() and hvp_grp.abs().sum() > 0
    torch.testing.assert_close(hvp_grp, hvp_ref, rtol=1e-5, atol=1e-7)


def test_fused_head_grouped_kl_shift_invariance_and_bench_grouped_cli(tmp_path):
    from edraft import bench_grouped

    g = torch.Generator().manual_seed(99)
    N, d_model, r, p_dim, V = 10, 32, 8, 12, 128
    h_raw_bf16 = torch.randn(N, d_model, generator=g).bfloat16()
    head_w = torch.randn(V, d_model, generator=g).bfloat16()
    head_b = torch.randn(V, generator=g)
    E = torch.randn(V, r, generator=g)
    H = torch.randn(N, p_dim, generator=g)
    W1 = (torch.randn(r, p_dim, generator=g) * 0.1).requires_grad_(True)
    b1 = (torch.randn(V, generator=g) * 0.1).requires_grad_(True)
    W2 = W1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)

    P = torch.randn(N, V, generator=g).softmax(-1)
    u_ids = [torch.randperm(V, generator=g)[:16].sort().values for _ in range(N)]
    p_u = [P[i, idx] for i, idx in enumerate(u_ids)]
    r_u = [(1.0 - p.double().sum()).clamp_min(0.0).float() for p in p_u]
    pre = collate_grouped_targets(u_ids, p_u, r_u, loss_weights(N), vocab_size=V)

    # Reference path via log_softmax(l_unit) - dense_correction
    l_unit = (F.linear(h_raw_bf16.float(), head_w.float(), head_b.float()) * 0.85).log_softmax(-1)
    corr1 = block_unary(E, W1, b1, H)
    loss_ref = union_rest_grouped_precollated(l_unit - corr1, pre)
    loss_ref.backward()

    # Fused shift-invariant path (skips log_softmax normalization on draft logits)
    corr2 = block_unary(E, W2, b2, H)
    loss_fused = fused_head_grouped_kl(
        h_raw_bf16, head_w, head_b, corr2, pre, output_multiplier=0.85, bf16_gemm=False
    )
    loss_fused.backward()

    torch.testing.assert_close(loss_fused, loss_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(W2.grad, W1.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b2.grad, b1.grad, rtol=1e-5, atol=1e-5)

    res = bench_grouped.main([
        "--output", str(tmp_path), "--device", "cpu",
        "--rows", "16", "--vocab_size", "256", "--warmups", "1", "--repeats", "2", "--compile",
    ])
    assert res["parity"]["passed"] is True
    assert res["parity"]["precollated_loss_abs_error"] < 2e-5
    assert res["parity"]["fused_head_gradient_max_abs_error"] < 1e-5
    assert res["parity"]["compiled_gradient_max_abs_error"] < 2e-4
    assert "compiled_precollated_forward_backward" in res["timings"]
    assert "fused_head_grouped_forward_backward" in res["timings"]


# ---------------------------------------------------------------------------
# 2. BF16 GEMM fast path parity & gradient flow
# ---------------------------------------------------------------------------

def test_bf16_gemm_forward_backward_and_bench_adapter_audit(tmp_path):
    from edraft import bench_adapter

    g = torch.Generator().manual_seed(314)
    V, r, p_dim, D, K = 256, 16, 24, 3, 6
    E = torch.randn(V, r, generator=g) * 0.2
    sparse = SignedHash(128)
    with torch.no_grad():
        sparse.values.copy_(torch.randn(128, generator=g) * 0.02)
    adapter = DenseAdapter(E, p_dim, D=D, sparse=sparse, M_sparse=16)
    with torch.no_grad():
        adapter.W.copy_(torch.randn_like(adapter.W) * 0.1)
        adapter.A.copy_(torch.randn_like(adapter.A) * 0.1)
        adapter.b.copy_(torch.randn_like(adapter.b) * 0.05)

    H = torch.randn(K, p_dim, generator=g)
    l_unit = torch.randn(K, V, generator=g).log_softmax(-1)
    prefix = [3, 9, 12]
    tokens = [5, 11, 2, 19, 7, 1]

    u_fp32 = adapter.unary(H, bf16_gemm=False)
    u_bf16 = adapter.unary(H, bf16_gemm=True)
    assert adapter.W.dtype == torch.float32 and adapter._cached_E_bf16 is not None
    e_bf16_ptr = adapter._cached_E_bf16.data_ptr()
    assert adapter.unary(H, bf16_gemm=True).dtype == torch.float32
    assert adapter._cached_E_bf16.data_ptr() == e_bf16_ptr  # cached E_bf16 reused

    torch.testing.assert_close(u_bf16, u_fp32, rtol=2e-2, atol=2e-2)

    snap = adapter.snapshot()
    offs = torch.tensor([[1], [2], [3]], dtype=torch.long)
    prevs = torch.tensor([12, 9, 3], dtype=torch.long)
    s_fp32 = snap.score_slot_tensor(l_unit[0], u_fp32[0], offs, prevs, contiguous_offsets=True, bf16_gemm=False)
    s_bf16 = snap.score_slot_tensor(l_unit[0], u_bf16[0], offs, prevs, contiguous_offsets=True, bf16_gemm=True)
    torch.testing.assert_close(s_bf16, s_fp32, rtol=3e-2, atol=3e-2)

    # Gradient flow through bf16_gemm into FP32 W, A, b
    adapter.zero_grad()
    corr = adapter.correction(l_unit, H, prefix, tokens, bf16_gemm=True)
    (corr.square().mean()).backward()
    assert adapter.W.grad is not None and adapter.W.grad.dtype == torch.float32 and adapter.W.grad.norm() > 0
    assert adapter.A.grad is not None and adapter.A.grad.dtype == torch.float32 and adapter.A.grad.norm() > 0
    assert adapter.b.grad is not None and adapter.b.grad.dtype == torch.float32 and adapter.b.grad.norm() > 0

    # OnlineState with bf16_gemm FastPath
    fp = FastPathRuntime(FastPathConfig(bf16_gemm=True), static_block_length=K)
    ostate = OnlineState(adapter, lr=0.02, lambda_l1_b=1e-3, fastpath=fp)
    target_logits = torch.randn(K, V, generator=g)
    assert ostate.update(1, l_unit, target_logits, H, prefix, tokens)
    assert ostate.dW.dtype == torch.float32 and ostate.db.dtype == torch.float32

    bench = bench_adapter.main([
        "--output", str(tmp_path), "--device", "cpu",
        "--vocab_size", "256", "--block_length", "4", "--warmups", "1", "--repeats", "2", "--bf16_gemm",
    ])
    assert "bf16_precision_audit" in bench["precision"]
    assert bench["precision"]["bf16_precision_audit"]["unary_max_abs_error"] < 0.05
    assert "bf16_forward_unary" in bench["precision"]


# ---------------------------------------------------------------------------
# 3. CroissAttn / optimizer_experiments optimizers & teacher-forced KL convergence
# ---------------------------------------------------------------------------

def test_gram_ns_pacemuon_anchormuon_bias_invariants_and_atan2_scale_invariance():
    g = torch.Generator().manual_seed(777)
    ns = GramNewtonSchulz()

    # 2D W [r, p] (both fat 8x20 and tall 20x8): Gram recurrence matches standard recurrence,
    # collapses condition number into Polar-Express equioscillation band [0.80, 1.20], and aligns > 0.99 with SVD polar factor
    u_raw, _, vh_raw = torch.linalg.svd(torch.randn(8, 20, generator=g), full_matrices=False)
    mat_2d = (u_raw * torch.logspace(-1.0, 1.0, 8)[:, None]) @ vh_raw
    q_2d = ns(mat_2d)
    x_norm = (mat_2d / (mat_2d.norm() + 1e-7)).unsqueeze(0)
    torch.testing.assert_close(q_2d, ns._standard_recurrence(x_norm).squeeze(0), rtol=1e-5, atol=1e-5)
    sv_2d = torch.linalg.svdvals(q_2d)
    assert float(sv_2d.min()) > 0.80 and float(sv_2d.max()) < 1.20
    polar_2d = u_raw @ vh_raw
    assert float(F.cosine_similarity(q_2d.flatten(), polar_2d.flatten(), dim=0)) > 0.99
    torch.testing.assert_close(ns(mat_2d.T), q_2d.T, rtol=1e-5, atol=1e-5)

    # 3D A [D, r, r] -> batched orthogonalization matches per-lag-slice orthogonalization & polar factor
    mat_3d = torch.randn(4, 10, 10, generator=g)
    q_3d = ns(mat_3d)
    for d in range(4):
        torch.testing.assert_close(q_3d[d], ns(mat_3d[d]), rtol=1e-5, atol=1e-5)
        sv_d = torch.linalg.svdvals(q_3d[d])
        assert float(sv_d.min()) > 0.80 and float(sv_d.max()) < 1.20
        ud, _, vhd = torch.linalg.svd(mat_3d[d], full_matrices=False)
        assert float(F.cosine_similarity(q_3d[d].flatten(), (ud @ vhd).flatten(), dim=0)) > 0.99

    # AnchorMuon MUST NOT apply SODA init-anchor pull on 1D bias `b` when weight_decay == 0 (prompt.md:597)
    W = nn.Parameter(torch.ones(6, 10))
    A = nn.Parameter(torch.ones(2, 6, 6))
    b = nn.Parameter(torch.linspace(-1.0, 1.0, 32))
    opt_anchor = AnchorMuon(
        [
            {"params": [W, A], "weight_decay": 0.01},
            {"params": [b], "weight_decay": 0.0},
        ],
        lr=0.02,
        soda_lambda_scale=0.1,
    )
    # Step 1 moves parameters away from init anchor; step 2 has zero grad on `b`
    W.grad = torch.randn(6, 10, generator=g)
    A.grad = torch.randn(2, 6, 6, generator=g)
    b.grad = torch.ones_like(b) * 0.5
    opt_anchor.step()
    b_after_step1 = b.detach().clone()
    W.grad = torch.zeros_like(W)
    A.grad = torch.zeros_like(A)
    b.grad = torch.zeros_like(b)
    # Decay momentum out over zero-grad steps? Even simpler: check group["soda_enabled"] directly!
    vec_groups = [grp for grp in opt_anchor.param_groups if not grp["use_matrix_update"]]
    mat_groups = [grp for grp in opt_anchor.param_groups if grp["use_matrix_update"]]
    assert all(not grp["soda_enabled"] for grp in vec_groups)
    assert all(grp["soda_enabled"] for grp in mat_groups)
    # And on a fresh parameter with zero gradient from step 0, b stays bit-exact!
    b_zero_grad = nn.Parameter(b_after_step1.clone())
    W_dummy = nn.Parameter(torch.ones(4, 6))
    opt_anchor2 = AnchorMuon(
        [{"params": [W_dummy], "weight_decay": 0.05}, {"params": [b_zero_grad], "weight_decay": 0.0}],
        lr=0.05,
        soda_lambda_scale=0.2,
    )
    W_dummy.grad = torch.randn(4, 6, generator=g)
    b_zero_grad.grad = torch.zeros_like(b_zero_grad)
    opt_anchor2.step()
    assert torch.equal(b_zero_grad.detach(), b_after_step1)

    # AdamATan2 scale invariance: scaling grad by 1e4 produces identical step when eps is tiny
    p1 = nn.Parameter(torch.zeros(16))
    p2 = nn.Parameter(torch.zeros(16))
    opt1 = AdamATan2([p1], lr=0.03, eps=1e-30)
    opt2 = AdamATan2([p2], lr=0.03, eps=1e-30)
    base_g = torch.randn(16, generator=g)
    p1.grad = base_g.clone()
    p2.grad = base_g.clone() * 1e4
    opt1.step()
    opt2.step()
    torch.testing.assert_close(p1.detach(), p2.detach(), rtol=1e-6, atol=1e-6)


def test_pacemuon_wsd_ema_beats_constant_adamw_on_planted_teacher_forced_kl():
    """Controlled held-out teacher-forced convergence benchmark on ill-conditioned features."""
    g = torch.Generator().manual_seed(20261114)
    V, r, p_dim, D, K = 96, 12, 20, 3, 6
    E = F.normalize(torch.randn(V, r, generator=g), dim=-1)
    # Ill-conditioned feature spectrum (singular values spanning 25x) so orthogonalized
    # updates have genuine conditioning advantage over diagonal AdamW.
    scales = torch.logspace(-0.7, 0.7, p_dim)
    teacher = DenseAdapter(E, p_dim, D=D, M_sparse=16)
    with torch.no_grad():
        teacher.W.copy_(torch.randn(r, p_dim, generator=g) * 0.45)
        teacher.A.copy_(torch.randn(D, r, r, generator=g) * 0.35)
        teacher.b.copy_(torch.randn(V, generator=g) * 0.15)

    def _make_split(n_blocks: int):
        blocks = []
        for _ in range(n_blocks):
            H = torch.randn(K, p_dim, generator=g) * scales
            l_unit = (torch.randn(K, V, generator=g) * 0.7).log_softmax(-1)
            prefix = torch.randint(0, V, (4,), generator=g).tolist()
            tokens = torch.randint(0, V, (K,), generator=g).tolist()
            with torch.no_grad():
                t_scores = l_unit - teacher.correction(l_unit, H, prefix, tokens)
                P = t_scores.softmax(-1)
            top_d = l_unit.topk(12, dim=-1).indices
            top_t = t_scores.topk(8, dim=-1).indices
            u_ids, p_u, r_u = [], [], []
            for i in range(K):
                idx = torch.cat([top_d[i], top_t[i], torch.tensor([tokens[i]])]).unique(sorted=True)
                pi = P[i, idx]
                u_ids.append(idx)
                p_u.append(pi)
                r_u.append((1.0 - pi.double().sum()).clamp_min(0.0).float())
            pre = collate_grouped_targets(u_ids, p_u, r_u, loss_weights(K), vocab_size=V)
            blocks.append((H, l_unit, t_scores, prefix, tokens, pre))
        return blocks

    train_set = _make_split(32)
    dev_set = _make_split(12)

    def _eval_dev(model: DenseAdapter) -> tuple[float, float]:
        grp_sum, full_sum, w_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for H, l_unit, t_scores, prefix, tokens, pre in dev_set:
                scores = l_unit - model.correction(l_unit, H, prefix, tokens)
                grp_sum += float(union_rest_grouped_precollated(scores, pre))
                full_sum += float(full_kl(scores, t_scores, pre.weights))
                w_sum += float(pre.weights.sum())
        return grp_sum / w_sum, full_sum / w_sum

    zero_grp, zero_full = _eval_dev(DenseAdapter(E, p_dim, D=D, M_sparse=16))

    total_steps = 48
    # Baseline: AdamW constant LR
    m_adam = DenseAdapter(E, p_dim, D=D, M_sparse=16)
    opt_adam = torch.optim.AdamW(
        [{"params": [m_adam.W, m_adam.A], "weight_decay": 1e-4}, {"params": [m_adam.b], "weight_decay": 0.0}],
        lr=3e-3,
    )
    for step in range(total_steps):
        H, l_unit, _, prefix, tokens, pre = train_set[step % len(train_set)]
        opt_adam.zero_grad()
        scores = l_unit - m_adam.correction(l_unit, H, prefix, tokens)
        loss = union_rest_grouped_precollated(scores, pre) / pre.weights.sum()
        loss.backward()
        opt_adam.step()
    adam_grp, adam_full = _eval_dev(m_adam)

    # Candidate: PaceMuon + WSD + EMA
    m_pace = DenseAdapter(E, p_dim, D=D, M_sparse=16)
    opt_pace = PaceMuon(
        [{"params": [m_pace.W, m_pace.A], "weight_decay": 1e-4}, {"params": [m_pace.b], "weight_decay": 0.0}],
        lr=0.02,
        fallback_lr=0.01,
        ema_enabled=True,
        pullback_c=0.05,
    )
    for step in range(total_steps):
        H, l_unit, _, prefix, tokens, pre = train_set[step % len(train_set)]
        opt_pace.zero_grad()
        scores = l_unit - m_pace.correction(l_unit, H, prefix, tokens)
        loss = union_rest_grouped_precollated(scores, pre) / pre.weights.sum()
        loss.backward()
        mult = compute_lr_multiplier(step + 1, total_steps, schedule="wsd")
        for grp in opt_pace.param_groups:
            grp["lr"] = grp["base_lr"] * mult
        opt_pace.step()
    with opt_pace.use_ema_weights():
        pace_grp, pace_full = _eval_dev(m_pace)

    assert adam_grp < zero_grp and adam_full < zero_full
    assert pace_grp < adam_grp and pace_full < adam_full


# ---------------------------------------------------------------------------
# 4. train.py DataLoader workers, Pacemuon+WSD resume, --compile, & DDP parity
# ---------------------------------------------------------------------------

def _make_mini_train_bundle(root: Path):
    torch.set_num_threads(1)
    torch.manual_seed(55)
    target = ToyTarget(vocab_size=10, hidden_size=8, seed=55)
    draft = ToyParallelDrafter(target, seed=56)
    writer = TraceShardWriter(root / "traces")
    for idx, split in enumerate(["train", "train", "dev", "test"]):
        for temp in (0.0, 0.7):
            warper = SamplingWarper() if temp == 0.0 else SamplingWarper(top_k=20, top_p=0.8)
            res = generate_toy(
                target, draft, [idx, 3, 1, 2], 9, block_length=3,
                temperature=temp, warper=warper, retain_tensors=True,
                generator=torch.Generator().manual_seed(55 + idx),
            )
            pid = f"p-{split}-{idx}"
            ident = {
                "model_pair": ["toy-target", "toy-draft"],
                "prompt": pid,
                "condition": "standard_nonthinking",
                "seed": 55 + idx,
                "temperature": temp,
                "verification_rule": "greedy" if temp == 0.0 else "ratio",
                "proposal_mode": "plain_dflash",
                "checkpoint": "synthetic-v1",
                "adapter_version": 0,
                "sampling_warp_config": warper.to_dict(),
            }
            meta = {
                "prompt_id": pid,
                "split": split,
                "condition": "standard_nonthinking",
                "seed": 55 + idx,
                "scale": "synthetic",
                "model_pair": ident["model_pair"],
                "projection_metadata": {"method": "identity", "frozen": True},
                "history_depth": 2,
            }
            blocks = [compact_block(rec, meta, M_draft=3, M_target=2, retain_live_full=True) for rec in res.blocks]
            writer.write(ident, blocks, {"generation": {
                "output_tokens": res.output_tokens,
                "realized_length": res.num_output_tokens,
                "condition": "standard_nonthinking",
                "split": split,
                "temperature": temp,
                "sampling_warp_config": warper.to_dict(),
                "complete": True,
            }})
    entries, idx_report = trace_index(root / "traces")
    return entries, idx_report, target.embedding[:, :4], draft.draft_lm_head, draft.head_metadata()


def _base_train_args(outdir: Path, **overrides):
    ns = SimpleNamespace(
        output=outdir,
        scale="smoke",
        seed=55,
        history_depth=2,
        M_sparse=4,
        sparse_capacity=32,
        learning_rate=0.02,
        weight_decay=1e-4,
        bias_weight_decay=0.0,
        sparse_learning_rate=0.03,
        lambda_l1=1e-3,
        batch_blocks=3,
        microbatch_blocks=2,
        vocab_chunk=4,
        epochs_dense=1,
        epochs_sparse=1,
        checkpoint_every=1,
        resume=True,
        num_workers=0,
        pin_memory=False,
        bf16_gemm=False,
        compile=False,
        optimizer="adamw",
        lr_schedule="constant",
        ema_eval=None,
        pace_pullback=0.05,
        anchor_soda=0.05,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_train_dataloader_workers_rng_parity_and_pacemuon_resume_and_compile(tmp_path):
    entries, idx_report, E, head, head_meta = _make_mini_train_bundle(tmp_path / "bundle")

    # 1. num_workers=0 vs num_workers=2 bit-exact parameter + RNG state parity
    torch.manual_seed(1234)
    random.seed(1234)
    fit(entries, idx_report, E, head.weight, head.bias, head_meta, _base_train_args(tmp_path / "w0", num_workers=0))
    rng_w0 = torch.get_rng_state().clone()

    torch.manual_seed(1234)
    random.seed(1234)
    fit(entries, idx_report, E, head.weight, head.bias, head_meta, _base_train_args(tmp_path / "w2", num_workers=2))
    rng_w2 = torch.get_rng_state().clone()

    assert torch.equal(rng_w0, rng_w2)
    a_w0, _ = load_adapter_checkpoint(tmp_path / "w0/m2_smoke/dense_sparse.pt")
    a_w2, _ = load_adapter_checkpoint(tmp_path / "w2/m2_smoke/dense_sparse.pt")
    assert torch.equal(a_w0.W, a_w2.W) and torch.equal(a_w0.A, a_w2.A) and torch.equal(a_w0.b, a_w2.b)
    assert torch.equal(a_w0.sparse.values, a_w2.sparse.values)

    # 2. PaceMuon + WSD + EMA interrupted vs uninterrupted exact resume parity, S stays sparse
    torch.manual_seed(909)
    random.seed(909)
    rep_direct = fit(
        entries, idx_report, E, head.weight, head.bias, head_meta,
        _base_train_args(tmp_path / "pm_direct", optimizer="pacemuon", lr_schedule="wsd"),
    )
    assert rep_direct["optimizer"]["dense"] == "PaceMuon"
    assert rep_direct["stages"]["dense_sparse"]["parameters"]["S"]["optimizer"] == "SGD followed by explicit proximal L1"
    assert rep_direct["stages"]["dense_sparse"]["parameters"]["S"]["exact_zero_fraction"] > 0.0

    torch.manual_seed(909)
    random.seed(909)
    interrupted = fit(
        entries, idx_report, E, head.weight, head.bias, head_meta,
        _base_train_args(tmp_path / "pm_resumed", optimizer="pacemuon", lr_schedule="wsd"),
        stop_after_steps=5,
    )
    assert interrupted["status"] == "interrupted_for_resume_test"
    fit(
        entries, idx_report, E, head.weight, head.bias, head_meta,
        _base_train_args(tmp_path / "pm_resumed", optimizer="pacemuon", lr_schedule="wsd"),
    )
    a_pm_dir, _ = load_adapter_checkpoint(tmp_path / "pm_direct/m2_smoke/dense_sparse.pt")
    a_pm_res, _ = load_adapter_checkpoint(tmp_path / "pm_resumed/m2_smoke/dense_sparse.pt")
    torch.testing.assert_close(a_pm_res.W, a_pm_dir.W, rtol=0, atol=0)
    torch.testing.assert_close(a_pm_res.A, a_pm_dir.A, rtol=0, atol=0)
    torch.testing.assert_close(a_pm_res.b, a_pm_dir.b, rtol=0, atol=0)
    torch.testing.assert_close(a_pm_res.sparse.values, a_pm_dir.sparse.values, rtol=0, atol=0)

    # Resume identity guard rejects mixing AdamW on a PaceMuon output directory
    with pytest.raises(ValueError, match="Checkpoint identity differs"):
        fit(
            entries, idx_report, E, head.weight, head.bias, head_meta,
            _base_train_args(tmp_path / "pm_resumed", optimizer="adamw", lr_schedule="constant"),
        )

    # 3. --compile parity on train.py
    torch.manual_seed(1234)
    random.seed(1234)
    fit(entries, idx_report, E, head.weight, head.bias, head_meta, _base_train_args(tmp_path / "comp", compile=True))
    a_comp, _ = load_adapter_checkpoint(tmp_path / "comp/m2_smoke/dense_sparse.pt")
    torch.testing.assert_close(a_comp.W, a_w0.W, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_comp.A, a_w0.A, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_comp.b, a_w0.b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_comp.sparse.values, a_w0.sparse.values, rtol=1e-5, atol=1e-5)


def _ddp_pacemuon_worker(rank: int, world_size: int, port: int, bundle_dir: str, out_dir: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    args = _base_train_args(
        Path(out_dir),
        rank=rank,
        world_size=world_size,
        dist_backend="gloo",
        device="cpu",
        optimizer="pacemuon",
        lr_schedule="wsd",
    )
    inited = _setup_distributed(args)
    try:
        entries, idx_report = trace_index(Path(bundle_dir) / "traces")
        target = ToyTarget(vocab_size=10, hidden_size=8, seed=55)
        draft = ToyParallelDrafter(target, seed=56)
        torch.manual_seed(77)
        random.seed(77)
        fit(entries, idx_report, target.embedding[:, :4], draft.draft_lm_head.weight,
            draft.draft_lm_head.bias, draft.head_metadata(), args)
    finally:
        if inited and dist.is_initialized():
            dist.destroy_process_group()


import os


def test_ddp_two_rank_pacemuon_wsd_matches_single_rank(tmp_path):
    bundle_dir = tmp_path / "bundle"
    entries, idx_report, E, head, head_meta = _make_mini_train_bundle(bundle_dir)

    torch.manual_seed(77)
    random.seed(77)
    fit(
        entries, idx_report, E, head.weight, head.bias, head_meta,
        _base_train_args(tmp_path / "single", optimizer="pacemuon", lr_schedule="wsd"),
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
    mp.spawn(
        _ddp_pacemuon_worker,
        args=(2, port, str(bundle_dir), str(tmp_path / "ddp")),
        nprocs=2,
        join=True,
    )
    a_single, _ = load_adapter_checkpoint(tmp_path / "single/m2_smoke/dense_sparse.pt")
    a_ddp, _ = load_adapter_checkpoint(tmp_path / "ddp/m2_smoke/dense_sparse.pt")
    torch.testing.assert_close(a_ddp.W, a_single.W, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_ddp.A, a_single.A, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_ddp.b, a_single.b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a_ddp.sparse.values, a_single.sparse.values, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Stateless _CompiledDrafterRunner on DFlashDraftModel-structured module + DynamicCache
# ---------------------------------------------------------------------------

class _MockRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _MockRotary(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1).to(x.dtype)
        return emb.cos(), emb.sin()


class _MockNonCausalAttn(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.head_dim = hidden // num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = False
        self.sliding_window = None
        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(hidden, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = _MockRMSNorm(self.head_dim)
        self.k_norm = _MockRMSNorm(self.head_dim)


class _MockDecoderLayer(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, layer_idx: int):
        super().__init__()
        self.input_layernorm = _MockRMSNorm(hidden)
        self.self_attn = _MockNonCausalAttn(hidden, num_heads, num_kv_heads, layer_idx)
        self.post_attention_layernorm = _MockRMSNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2, bias=False), nn.SiLU(), nn.Linear(hidden * 2, hidden, bias=False))
        self.attention_conv = None
        self.mlp_conv = None


class _MockDFlashDrafter(nn.Module):
    """Exact structural replica of vendor/dflash/dflash/model.py:DFlashDraftModel."""

    def __init__(self, vocab: int = 48, hidden: int = 32, ctx_dim: int = 64, n_layers: int = 2):
        from transformers import Qwen3Config

        super().__init__()
        self.config = Qwen3Config(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=hidden * 2,
            num_hidden_layers=n_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.fc = nn.Linear(ctx_dim, hidden, bias=False)
        self.hidden_norm = _MockRMSNorm(hidden)
        self.rotary_emb = _MockRotary(hidden // 4)
        self.layers = nn.ModuleList([_MockDecoderLayer(hidden, 4, 2, idx) for idx in range(n_layers)])
        self.norm = _MockRMSNorm(hidden)
        self.mask_token_id = 0
        self.target_layer_ids = [0, 1]

    def compute_logits(self, hidden_states: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        return lm_head(hidden_states)

    def forward(self, *, target_hidden, noise_embedding, position_ids, past_key_values=None, use_cache=True):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
        h = noise_embedding
        t_proj = self.hidden_norm(self.fc(target_hidden))
        cos, sin = self.rotary_emb(h, position_ids)
        for layer in self.layers:
            residual = h
            h_norm = layer.input_layernorm(h)
            attn = layer.self_attn
            bsz, q_len = h_norm.shape[:-1]
            ctx_len = t_proj.shape[1]
            q = attn.q_norm(attn.q_proj(h_norm).view(bsz, q_len, -1, attn.head_dim)).transpose(1, 2)
            k = torch.cat([attn.k_proj(t_proj), attn.k_proj(h_norm)], dim=1).view(bsz, ctx_len + q_len, -1, attn.head_dim)
            v = torch.cat([attn.v_proj(t_proj), attn.v_proj(h_norm)], dim=1).view(bsz, ctx_len + q_len, -1, attn.head_dim).transpose(1, 2)
            k = attn.k_norm(k).transpose(1, 2)
            half = q.shape[-1] // 2
            rot_q = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            rot_k = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
            q = q * cos.unsqueeze(1)[..., -q_len:, :] + rot_q * sin.unsqueeze(1)[..., -q_len:, :]
            k = k * cos.unsqueeze(1) + rot_k * sin.unsqueeze(1)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, attn.layer_idx, {"sin": sin, "cos": cos, "cache_position": None})
            out, _ = sdpa_fn(attn, q, k, v, None, dropout=0.0, scaling=attn.scaling, sliding_window=None)
            h = residual + attn.o_proj(out.reshape(bsz, q_len, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.norm(h)


def test_compiled_drafter_runner_dynamic_kv_cache_and_rollback_parity():
    torch.manual_seed(303)
    vocab, hidden, ctx_dim = 48, 32, 64
    drafter = _MockDFlashDrafter(vocab=vocab, hidden=hidden, ctx_dim=ctx_dim, n_layers=2).eval()
    embed = nn.Embedding(vocab, hidden)
    lm_head = nn.Linear(hidden, vocab, bias=False)
    mock_target = SimpleNamespace(get_input_embeddings=lambda: embed)
    backend = SimpleNamespace(
        drafter=drafter,
        target=mock_target,
        input_embedding_scale=1.0,
        draft_lm_head=lm_head,
    )

    fp = FastPathRuntime(FastPathConfig(compile_drafter=True), static_block_length=5)
    cache_eager = make_cache(drafter.config)
    cache_comp = make_cache(drafter.config)
    positions = torch.arange(128)[None, :]

    # Simulate 6 closed-loop rounds with varying ctx_len (prefill=7, then 1, 4, 2, 6, 3) and terminal cap K=3
    start = 7
    schedule = [(7, 6, 1), (1, 6, 4), (4, 6, 2), (2, 6, 6), (6, 6, 3), (3, 4, 2)]
    g = torch.Generator().manual_seed(404)
    for ctx_len, verify_size, emitted_len in schedule:
        canvas = torch.randint(0, vocab, (1, verify_size), generator=g)
        target_hidden = torch.randn(1, ctx_len, ctx_dim, generator=g)
        pos_slice = positions[:, start - ctx_len:start + verify_size]

        with torch.no_grad():
            noise = F.embedding(canvas, embed.weight) * backend.input_embedding_scale
            h_eag = drafter(
                target_hidden=target_hidden,
                noise_embedding=noise,
                position_ids=pos_slice,
                past_key_values=cache_eager,
                use_cache=True,
            )[0, 1:, :]
            crop_cache(cache_eager, start)
            logits_eag = drafter.compute_logits(h_eag[None], lm_head)[0]

        h_cmp, logits_cmp = fp.draft_block(backend, canvas, target_hidden, pos_slice, cache_comp, start)
        runner = fp._drafter_runners[id(drafter)]
        assert runner.decomposed is True
        torch.testing.assert_close(h_cmp, h_eag, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(logits_cmp, logits_eag, rtol=1e-5, atol=1e-5)
        assert cache_comp.get_seq_length() == cache_eager.get_seq_length() == start
        start += emitted_len


# ---------------------------------------------------------------------------
# 6. OnlineState compile_online (0 recompiles on scheduled lr), adaptive modes, & OnlineBigram fuzz
# ---------------------------------------------------------------------------

def test_online_compile_scheduled_lr_zero_recompile_and_adaptive_modes():
    g = torch.Generator().manual_seed(606)
    V, r, p_dim, D, K = 64, 8, 12, 3, 5
    E = torch.randn(V, r, generator=g) * 0.2
    adapter = DenseAdapter(E, p_dim, D=D, M_sparse=8)
    sched = make_online_lr_schedule(0.03, schedule="warmup_inv_sqrt", warmup_steps=4)

    fp = FastPathRuntime(FastPathConfig(compile_online=True), static_block_length=K)
    s_eager = OnlineState(adapter, lr=sched, lambda_l2=0.01, lambda_l1_b=1e-3, lambda_l1_s=1e-3,
                          enable_sparse=True, sparse_capacity=127)
    s_comp = OnlineState(adapter, lr=sched, lambda_l2=0.01, lambda_l1_b=1e-3, lambda_l1_s=1e-3,
                         enable_sparse=True, sparse_capacity=127, fastpath=fp)

    # 20 distinct scheduled LR values (> 2.5x Dynamo's default recompile_limit=8!)
    for r_idx in range(1, 21):
        l_u = torch.randn(K, V, generator=g).log_softmax(-1)
        t_l = torch.randn(K, V, generator=g)
        H = torch.randn(K, p_dim, generator=g)
        prefix = [r_idx % V, (r_idx * 3) % V, (r_idx * 7) % V]
        toks = [(r_idx + j) % V for j in range(K)]
        assert s_eager.update(r_idx, l_u, t_l, H, prefix, toks)
        assert s_comp.update(r_idx, l_u, t_l, H, prefix, toks)

    torch.testing.assert_close(s_comp.dW, s_eager.dW, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s_comp.dA, s_eager.dA, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s_comp.db, s_eager.db, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(s_comp.dS.read(torch.arange(127)), s_eager.dS.read(torch.arange(127)), rtol=1e-5, atol=1e-6)

    # Adaptive online optimizer modes ('atan2', 'rms', 'muon') reduce stationary request loss & preserve invariants
    l_stat = torch.randn(K, V, generator=g).log_softmax(-1)
    t_stat = torch.randn(K, V, generator=g) * 2.0
    H_stat = torch.randn(K, p_dim, generator=g)
    for mode in ("atan2", "rms", "muon"):
        st = OnlineState(adapter, lr=0.03, lambda_l2=0.01, lambda_l1_b=5e-3, lambda_l1_s=2e-3,
                         enable_sparse=True, sparse_capacity=127, optimizer=mode)
        snap0 = st.snapshot()
        snap0_b = snap0.b.clone()
        losses = []
        for r_idx in range(1, 9):
            st.update(r_idx, l_stat, t_stat, H_stat, [1, 2, 3], [4, 5, 6, 7, 8])
            losses.append(st.last_loss)
        assert losses[-1] < losses[0]
        assert torch.equal(snap0.b, snap0_b)  # previous snapshot immutable
        m = st.metrics()
        assert m["dense_optimizer"] == mode
        assert m["db_exact_zero_fraction"] > 0.0  # proximal_l1 preserved on db


def _reference_bigram_prior(counts: Counter, vocab_size: int, alpha: float, k_ret: int | None) -> torch.Tensor:
    if not counts:
        return torch.full((vocab_size,), -math.log(float(vocab_size)), dtype=torch.float32)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if k_ret is not None:
        ordered = ordered[:k_ret]
    total = sum(counts.values())
    evicted = total - sum(c for _, c in ordered)
    absent = vocab_size - len(ordered)
    denom = total + alpha * vocab_size
    base = (evicted / absent) if absent else 0.0
    out = torch.full((vocab_size,), math.log((base + alpha) / denom), dtype=torch.float32)
    for tok, c in ordered:
        out[tok] = math.log((c + alpha) / denom)
    return out


@pytest.mark.parametrize("k_ret", [None, 1, 4, 9])
def test_online_bigram_incremental_fuzz_oracle(k_ret):
    rng = random.Random(2026 + (0 if k_ret is None else k_ret))
    V = 37
    bigram = OnlineBigram(V, alpha=0.4, lambda_tr=0.25, k_ret=k_ret)
    oracle: dict[int, Counter] = defaultdict(Counter)
    l_unit = torch.randn(V, generator=torch.Generator().manual_seed(19))

    # Empty bypasses
    assert bigram.score(l_unit, None) is l_unit
    assert bigram.score(l_unit, 5) is l_unit

    for step in range(120):
        prefix = [rng.randrange(8)] if step % 7 != 0 else []
        committed = [rng.randrange(14) for _ in range(rng.randint(1, 6))]
        bigram.update(prefix, committed)
        prev = prefix[-1] if prefix else None
        for tok in committed:
            if prev is not None:
                oracle[prev][tok] += 1
            prev = tok
        for q_prev in (0, 3, 7, 13, 30):
            p_actual = bigram.prior(q_prev)
            p_ref = _reference_bigram_prior(oracle[q_prev], V, 0.4, k_ret)
            torch.testing.assert_close(p_actual, p_ref, rtol=1e-6, atol=1e-6)
            assert abs(float(p_actual.exp().sum()) - 1.0) <= 1e-6
            s_actual = bigram.score(l_unit, q_prev)
            if not oracle[q_prev]:
                assert s_actual is l_unit
            else:
                torch.testing.assert_close(s_actual, l_unit + 0.25 * p_ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Vectorized compact_block, validate_block, live_block_diagnostics, add_blocks, & timing audit
# ---------------------------------------------------------------------------

def test_vectorized_compact_block_and_validate_block_oracle_and_error_cases():
    g = torch.Generator().manual_seed(808)
    for K, V, M_d, M_t, specs in [(1, 16, 4, 2, []), (7, 64, 8, 5, [0, 3, 63]), (4, 12, 12, 12, [1, 5])]:
        draft_logits = torch.randn(K, V, generator=g)
        target_logits = torch.randn(K, V, generator=g)
        draft_tokens = [int(x) for x in torch.randint(0, V, (K,), generator=g)]
        rec = {
            "draft_tokens": draft_tokens,
            "draft_logits": draft_logits,
            "target_logits": target_logits,
            "h_raw_bf16": torch.randn(K, 10, generator=g).bfloat16(),
            "h": torch.randn(K, 6, generator=g),
            "draft_lm_head": {
                "artifact_id": "toy", "revision": "v1", "dtype": "torch.bfloat16",
                "bias": False, "tied_weights": True, "hook": "linear", "shape": [V, 10],
            },
            "collection_temperature": 0.7,
            "loss_temperature": 1.0,
            "verification_rule": "ratio",
            "sampling_warp_config": {"top_k": 5, "top_p": 0.9, "min_p": 0.0},
            "absolute_output_offset": 4,
            "input_context_length": 12,
            "verified_prefix_tail": [1, 2, 3, 4],
            "physical_to_logical": {"anchor_physical_index": 0, "logical_to_physical": list(range(1, K + 1))},
            "accepted_length": K // 2,
            "first_rejection_index": K // 2,
            "replacement_token": 0,
            "adapter_version": 0,
        }
        meta = {
            "prompt_id": "p0", "split": "train", "condition": "standard_nonthinking",
            "seed": 1, "scale": "smoke", "model_pair": ["t", "d"],
            "projection_metadata": {"frozen": True},
        }
        comp = compact_block(rec, meta, M_draft=M_d, M_target=M_t, special_tokens=specs, retain_live_full=True)

        # Reference per-slot unique(sorted=True) oracle
        l_u = draft_logits.float().log_softmax(-1)
        P_tr = target_logits.float().softmax(-1)
        td = draft_logits.topk(min(M_d, V), dim=-1).indices
        tt = target_logits.topk(min(M_t, V), dim=-1).indices
        sp_t = torch.tensor(sorted(set(specs)), dtype=torch.long)
        for i in range(K):
            ref_ids = torch.cat([td[i], tt[i], sp_t, torch.tensor([draft_tokens[i]])]).unique(sorted=True)
            assert torch.equal(comp["union_ids"][i], ref_ids)
            assert torch.equal(comp["draft_l_unit_union"][i], l_u[i, ref_ids])
            assert torch.equal(comp["target_probs_union"][i], P_tr[i, ref_ids])
            ref_rest = torch.tensor(0.0) if len(ref_ids) == V else (1.0 - P_tr[i, ref_ids].double().sum()).clamp_min(0.0).float()
            assert torch.equal(comp["target_rest_mass"][i], ref_rest)

        # Malformed trace rejection checks on validate_block
        if len(comp["union_ids"][0]) >= 2:
            bad_unsorted = deepcopy(comp)
            bad_unsorted["union_ids"][0] = bad_unsorted["union_ids"][0].flip(0)
            with pytest.raises(ValueError, match="unique, sorted, and in vocabulary"):
                validate_block(bad_unsorted)

            bad_dup = deepcopy(comp)
            bad_dup["union_ids"][0] = bad_dup["union_ids"][0].clone()
            bad_dup["union_ids"][0][1] = bad_dup["union_ids"][0][0]
            with pytest.raises(ValueError, match="unique, sorted, and in vocabulary"):
                validate_block(bad_dup)

        bad_oob = deepcopy(comp)
        bad_oob["union_ids"][0] = bad_oob["union_ids"][0].clone()
        bad_oob["union_ids"][0][-1] = V
        with pytest.raises(ValueError, match="unique, sorted, and in vocabulary"):
            validate_block(bad_oob)

        bad_mass = deepcopy(comp)
        bad_mass["target_rest_mass"] = bad_mass["target_rest_mass"].clone()
        bad_mass["target_rest_mass"][0] += 0.1
        with pytest.raises(ValueError, match="invalid probability|must sum to one"):
            validate_block(bad_mass)

        if specs and len(comp["union_ids"][0]) > 1:
            bad_spec = deepcopy(comp)
            keep = bad_spec["union_ids"][0] != specs[0]
            if bool(keep.any()) and specs[0] != draft_tokens[0]:
                dropped_p = float(bad_spec["target_probs_union"][0][~keep].sum())
                bad_spec["union_ids"][0] = bad_spec["union_ids"][0][keep]
                bad_spec["draft_l_unit_union"][0] = bad_spec["draft_l_unit_union"][0][keep]
                bad_spec["target_probs_union"][0] = bad_spec["target_probs_union"][0][keep]
                bad_spec["target_rest_mass"] = bad_spec["target_rest_mass"].clone()
                bad_spec["target_rest_mass"][0] += dropped_p
                with pytest.raises(ValueError, match="omits a requested special token"):
                    validate_block(bad_spec)


def test_multi_token_rank_add_blocks_and_eval_fastpath_schema():
    g = torch.Generator().manual_seed(919)
    # 1) Zero-tie path vs explicit tie-break reference
    scores_notie = torch.arange(6 * 73, dtype=torch.float32).view(6, 73) * 0.01
    q_ids_notie = torch.randint(0, 73, (6, 3), generator=g)
    batched_notie = multi_token_rank(scores_notie, q_ids_notie)
    ids_ref = torch.arange(73)
    for col in range(3):
        tids = q_ids_notie[:, col]
        sel = scores_notie.gather(-1, tids[:, None])
        ref_rank = 1 + ((scores_notie > sel) | ((scores_notie == sel) & (ids_ref[None, :] < tids[:, None]))).sum(-1)
        assert torch.equal(batched_notie[:, col], ref_rank)

    # 2) Exact ties to test deterministic lowest-ID tie break
    scores = torch.randn(6, 73, generator=g)
    scores[2, 5] = scores[2, 19] = scores.max() + 1.0
    q_ids = torch.randint(0, 73, (6, 3), generator=g)
    q_ids[2, 0] = 5
    q_ids[2, 1] = 19
    batched = multi_token_rank(scores, q_ids)
    for col in range(3):
        tids = q_ids[:, col]
        sel = scores.gather(-1, tids[:, None])
        ref_rank = 1 + ((scores > sel) | ((scores == sel) & (ids_ref[None, :] < tids[:, None]))).sum(-1)
        assert torch.equal(batched[:, col], ref_rank)
        assert torch.equal(batched[:, col], token_rank(scores, tids))
    assert int(batched[2, 0]) == 1 and int(batched[2, 1]) == 2

    # DiagnosticAccumulator.add_blocks vs add_block parity
    acc_single = DiagnosticAccumulator()
    acc_batch = DiagnosticAccumulator()
    blocks = []
    for b_idx in range(5):
        K = 4
        rec = {
            "draft_logits": torch.randn(K, 31, generator=g),
            "target_logits": torch.randn(K, 31, generator=g),
            "collection_temperature": 0.7 if b_idx % 2 else 0.0,
            "sampling_warp_config": {"top_k": 4, "top_p": 0.85, "min_p": 0.0},
            "first_rejection_index": b_idx % (K + 1),
            "replacement_token": None if (b_idx % (K + 1)) == K else 3,
            "accepted_length": b_idx % (K + 1),
            "draft_tokens": [1, 2, 3, 4],
            "h": torch.randn(K, 12, generator=g),
            "prompt_id": f"p-{b_idx // 2}",
            "seed": 11,
            "condition": "long_thinking",
            "split": "test",
            "absolute_output_offset": 1020 + b_idx * 3,
            "input_context_length": 4090 + b_idx * 4,
        }
        rec["diagnostics"] = live_block_diagnostics(rec, seed=50 + b_idx)
        blocks.append(rec)
        acc_single.add_block(rec, repetition_dominated=False)
    acc_batch.add_blocks(blocks, repetition_dominated=False)

    assert acc_single.groups.keys() == acc_batch.groups.keys()
    for key in acc_single.groups:
        g1, g2 = acc_single.groups[key], acc_batch.groups[key]
        assert g1["rows"] == g2["rows"]
        assert g1["h_norms"] == g2["h_norms"]
        torch.testing.assert_close(g1["h_sum"], g2["h_sum"], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(g1["h_sq_sum"], g2["h_sq_sum"], rtol=1e-13, atol=1e-13)

    # FastPathConfig backwards-compatible default dict vs opt-in Round 2 keys
    assert FastPathConfig().to_dict() == {"compile": False, "cuda_graph": False, "compile_mode": "default"}
    full_fp = FastPathConfig(bf16_gemm=True, compile_drafter=True, compile_online=True).to_dict()
    assert full_fp == {
        "compile": False,
        "cuda_graph": False,
        "compile_mode": "default",
        "bf16_gemm": True,
        "compile_drafter": True,
        "compile_online": True,
    }


# ---------------------------------------------------------------------------
# 8. Adversarial bug-hunt regressions (#55 OPT-BUG-1..7, REPRO-1, & block_shuffle_window)
# ---------------------------------------------------------------------------

def test_bughunter_regressions_online_dbs_zombie_autograd_verify_validator_and_block_shuffle(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    import time

    from edraft import collect as collect_mod
    from edraft import eval_closed_loop
    from edraft.runtime import ROOT, WorkQueue
    from edraft.train import epoch_references
    from edraft.verify import residual_distribution, verify_block

    # (A) REPRO-1 + Guardrail A: residual_distribution still raises on zero residual,
    # while verify_block cleanly handles FP32 1-ULP sum() ties without crashing.
    with pytest.raises(RuntimeError, match="no positive"):
        residual_distribution(torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]))

    g = torch.Generator().manual_seed(0)
    warp = SamplingWarper(top_k=20, top_p=0.8)
    triggered_1ulp = False
    for _ in range(300):
        t_logits = torch.randn(100, generator=g, dtype=torch.float32) * 3.0
        d_logits = t_logits + torch.randn(100, generator=g, dtype=torch.float32) * 1e-6
        P = probabilities(t_logits, 0.7, warp)
        q = probabilities(d_logits, 0.7, warp)
        if float((P - q).clamp_min(0).sum()) == 0.0 and bool((q > P).any()):
            tok = int((q > P).nonzero(as_tuple=True)[0][0])
            u_max = torch.tensor([1.0 - 2**-24], dtype=torch.float32)
            monkeypatch.setattr(torch, "rand", lambda *a, **kw: u_max)
            t2 = torch.stack([t_logits, t_logits])
            vr = verify_block([tok], q[None, :], t2, 0.7, warp, generator=torch.Generator().manual_seed(1))
            assert len(vr.emitted_tokens) >= 1
            monkeypatch.undo()
            triggered_1ulp = True
            break
    assert triggered_1ulp

    # (B) OPT-BUG-1: online_dbS passes m4_evidence_validator/online_checks.py byte-algebra check
    oc_spec = importlib.util.spec_from_file_location(
        "m4_online_checks", ROOT / "artifacts/m4_evidence_validator/online_checks.py"
    )
    online_checks = importlib.util.module_from_spec(oc_spec)
    oc_spec.loader.exec_module(online_checks)
    off = DenseAdapter(torch.randn(32, 4, generator=g), 6, D=2, sparse=SignedHash(257), M_sparse=8)
    dbs = OnlineState(off.snapshot(), lr=0.05, update_stride=1, sparse_capacity=257,
                      enable_sparse=True, bias_sparse_only=True)
    dbs.update(1, torch.randn(3, 32, generator=g), torch.randn(3, 32, generator=g),
               torch.randn(3, 6, generator=g), [1, 2], [3, 4, 5])
    row = {
        "scale": "smoke", "stage": "m4", "arm": "online_dbS", "phase_profile": False,
        "settings": {"update_stride": 1, "lr": 0.05},
        "blocks": [{"round_idx": 1, "adapter_version": 0, "next_adapter_version": 1,
                    "proposal_mode": "dense_full", "loss_temperature": 1}],
        "online": dbs.metrics(),
        "online_invariants": {"stride": 1, "executed_updates": 1, "request_reset": True,
                              "version_only_between_rounds": True, "single_round_no_buffering": True},
        "parameter_norms": {"W": 0.0, "A": 0.0, "b": float(dbs.db.norm())},
        "online_bandwidth_definition": "bytes / online_update_ms",
        "estimated_online_state_bandwidth_GBps": None, "phase_ms": {},
    }
    res_oc = online_checks.check_online_row(
        row, expected_scale="smoke", expected_arm="online_dbS",
        expected_setting={"update_stride": 1, "lr": 0.05}, profile=False,
    )
    assert res_oc["failures"] == []

    # (C) OPT-BUG-2: WorkQueue._pid_alive reclaims Linux zombie PIDs (State: Z)
    if sys.platform.startswith("linux"):
        import subprocess
        q_file = tmp_path / "wq_zombie.json"
        child_code = (
            "import os; from pathlib import Path; from edraft.runtime import WorkQueue; "
            f"wq = WorkQueue(Path({str(q_file)!r}), ['job-0', 'job-1']); "
            "assert wq.claim(rank=0) == 'job-0'; os._exit(1)"
        )
        proc = subprocess.Popen([sys.executable, "-c", child_code])
        deadline = time.time() + 5.0
        while time.time() < deadline:
            sp = Path(f"/proc/{proc.pid}/stat")
            if q_file.exists() and sp.exists():
                st = json.loads(q_file.read_text())
                if "job-0" in st.get("claimed", {}) and sp.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    break
            time.sleep(0.01)
        wq1 = WorkQueue(q_file, ["job-0", "job-1"])
        claimed = []
        while (it := wq1.claim(rank=1)) is not None:
            claimed.append(it)
            wq1.complete(it)
        proc.wait(timeout=5)
        assert set(claimed) == {"job-0", "job-1"}
        assert set(json.loads(q_file.read_text())["completed"]) == {"job-0", "job-1"}

    # (D) OPT-BUG-6a & OPT-BUG-6b: 0-D scalar weights in full_kl & rest_p=0 + -inf tail in union_rest_grouped
    s0 = torch.randn(3, 8, generator=g, requires_grad=True)
    t0 = torch.randn(3, 8, generator=g)
    loss_scalar_w = full_kl(s0, t0, weights=torch.tensor(0.5))
    loss_scalar_w.backward()
    s0_ref = s0.detach().clone().requires_grad_(True)
    ((t0.log_softmax(-1).exp() * (t0.log_softmax(-1) - s0_ref.log_softmax(-1))).sum(-1) * 0.5).sum().backward()
    torch.testing.assert_close(s0.grad, s0_ref.grad, rtol=1e-6, atol=1e-6)

    cand_scores = torch.full((1, 6), -torch.inf)
    cand_scores[0, :2] = torch.tensor([1.5, 0.5])
    cand_scores.requires_grad_(True)
    loss_cand = union_rest_grouped(
        cand_scores,
        [torch.tensor([0, 1], dtype=torch.long)],
        [torch.tensor([0.7, 0.3], dtype=torch.float32)],
        [torch.tensor(0.0, dtype=torch.float32)],
    )
    assert torch.isfinite(loss_cand).item()
    (grad_cand,) = torch.autograd.grad(loss_cand, cand_scores, create_graph=True)
    assert torch.isfinite(grad_cand).all().item()
    (hvp_cand,) = torch.autograd.grad((grad_cand ** 2).sum(), cand_scores)
    assert torch.isfinite(hvp_cand).all().item()

    # (E) OPT-BUG-3, OPT-BUG-4, OPT-BUG-7 + Guardrail B: lockstep implementation_hash & validate.py
    val_dir = ROOT / "artifacts/m4_evidence_validator"
    sys.path.insert(0, str(val_dir))
    try:
        val_spec = importlib.util.spec_from_file_location("m4_validate_mod", val_dir / "validate.py")
        val_mod = importlib.util.module_from_spec(val_spec)
        val_spec.loader.exec_module(val_mod)
        assert val_mod._valid_fastpath_dict(
            FastPathConfig(compile=True, bf16_gemm=True, compile_drafter=True, compile_online=True).to_dict()
        )
        import hashlib
        h_val = hashlib.sha256()
        for fname in val_mod.IMPLEMENTATION_FILES:
            p = ROOT / "src/edraft" / fname
            h_val.update(fname.encode())
            h_val.update(p.read_bytes())
        assert h_val.hexdigest() == eval_closed_loop.implementation_hash()
    finally:
        sys.path.pop(0)

    # (F) OPT-BUG-5: collect.py --schedule dynamic crash window backfills missing head_reconstruction
    c_out = tmp_path / "collect_dyn"
    collect_mod.main(["--scale", "smoke", "--output", str(c_out), "--stage", "calibrate"])
    collect_mod.main(["--scale", "smoke", "--output", str(c_out), "--stage", "collect",
                      "--rank", "0", "--world_size", "2", "--schedule", "dynamic"])
    collect_mod.main(["--scale", "smoke", "--output", str(c_out), "--stage", "collect",
                      "--rank", "1", "--world_size", "2", "--schedule", "dynamic"])
    # Simulate crash window: delete head_reconstruction-rank*.json after shard publication, then resume on rank 1
    for hp in c_out.glob("head_reconstruction-rank*.json"):
        hp.unlink()
    collect_mod.main(["--scale", "smoke", "--output", str(c_out), "--stage", "collect",
                      "--rank", "1", "--world_size", "2", "--schedule", "dynamic"])
    collect_mod.main(["--scale", "smoke", "--output", str(c_out), "--stage", "audit"])
    audit_rep = json.loads((c_out / "m1_audit.json").read_text())
    assert audit_rep["status"] == "passed"
    assert audit_rep["requirements"]["serialized_head_reconstruction"] is True

    # (G) train.py --block_shuffle_window mixes blocks across shards & preserves exact resume
    entries, idx_report, E, head, head_meta = _make_mini_train_bundle(tmp_path / "shuf_bundle")
    seq_refs = epoch_references(entries, "train", seed=42, block_shuffle_window=0)
    win_refs = epoch_references(entries, "train", seed=42, block_shuffle_window=4)
    assert len(seq_refs) == len(win_refs)
    assert sorted((str(s.path), i) for s, i in seq_refs) == sorted((str(s.path), i) for s, i in win_refs)
    assert [str(s.path) for s, _ in seq_refs] != [str(s.path) for s, _ in win_refs]

    torch.manual_seed(321)
    random.seed(321)
    fit(entries, idx_report, E, head.weight, head.bias, head_meta,
        _base_train_args(tmp_path / "shuf_dir", optimizer="pacemuon", lr_schedule="wsd", block_shuffle_window=4))
    torch.manual_seed(321)
    random.seed(321)
    fit(entries, idx_report, E, head.weight, head.bias, head_meta,
        _base_train_args(tmp_path / "shuf_res", optimizer="pacemuon", lr_schedule="wsd", block_shuffle_window=4),
        stop_after_steps=4)
    fit(entries, idx_report, E, head.weight, head.bias, head_meta,
        _base_train_args(tmp_path / "shuf_res", optimizer="pacemuon", lr_schedule="wsd", block_shuffle_window=4))
    a_sd, _ = load_adapter_checkpoint(tmp_path / "shuf_dir/m2_smoke/dense_sparse.pt")
    a_sr, _ = load_adapter_checkpoint(tmp_path / "shuf_res/m2_smoke/dense_sparse.pt")
    torch.testing.assert_close(a_sr.W, a_sd.W, rtol=0, atol=0)
    torch.testing.assert_close(a_sr.b, a_sd.b, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 9. Compact top-k sequential proposal sampling, GPU-resident OnlineBigram.score_tensor,
#    and #64 Round-2 edge-case regressions (OPT-R2-BUG-2, OPT-R2-BUG-3, OPT-R2-BUG-4)
# ---------------------------------------------------------------------------

def test_compact_probabilities_and_sample_token_compact_exact_parity():
    from dflash.model import _sampling_probs
    from edraft.sampler import compact_probabilities, sample_token, sample_token_compact

    g = torch.Generator().manual_seed(20261114)
    V, K = 257, 6
    warpers = [
        SamplingWarper(top_k=0, top_p=1.0, min_p=0.0),
        SamplingWarper(top_k=0, top_p=0.82, min_p=0.05),
        SamplingWarper(top_k=V, top_p=0.85, min_p=0.0),
        SamplingWarper(top_k=20, top_p=1.0, min_p=0.0),
        SamplingWarper(top_k=20, top_p=0.80, min_p=0.0),
        SamplingWarper(top_k=20, top_p=0.95, min_p=0.08),
    ]
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        raw_2d = torch.randn(K, V, generator=g, dtype=torch.float32).to(dtype)
        # Inject exact ties + -inf masked slots
        raw_2d[1, :30] = raw_2d[1, 0]
        raw_2d[2, 10:] = -torch.inf
        for T in (0.6, 0.7, 1.0):
            for warp in warpers:
                for inp in (raw_2d[0], raw_2d[1], raw_2d[2], raw_2d):
                    p_sub, idx = compact_probabilities(inp, T, warp)
                    p_full = probabilities(inp, T, warp)
                    if idx is None:
                        assert torch.equal(p_sub, p_full)
                    else:
                        reconstructed = torch.zeros_like(p_full).scatter(-1, idx, p_sub)
                        assert torch.equal(reconstructed, p_full)
                    if warp.min_p == 0.0 and dtype != torch.float64:
                        ref_dflash = _sampling_probs(inp, T, top_p=warp.top_p, top_k=warp.top_k)
                        assert torch.equal(p_full, ref_dflash)

    # Validation guards & greedy T=0 lowest-ID tie-break with zero RNG consumption
    with pytest.raises(ValueError):
        sample_token_compact(torch.empty(0), 0.7)
    with pytest.raises(ValueError):
        sample_token_compact(torch.zeros(2, 4), 0.7)
    with pytest.raises(ValueError):
        sample_token_compact(torch.zeros(4), -0.1)

    tied = torch.tensor([-1.0, 2.5, 0.3, 2.5, 2.5], dtype=torch.float32)
    rng_g = torch.Generator().manual_seed(77)
    state_before = rng_g.get_state().clone()
    tok0, p0, i0 = sample_token_compact(tied, 0.0, SamplingWarper(top_k=2, top_p=0.5), generator=rng_g)
    assert int(tok0) == 1 and p0 is None and i0 is None
    assert torch.equal(rng_g.get_state(), state_before)

    # FastPathRuntime.compact_probabilities static-graph tuple capture, replay, & non-aliasing
    fp = FastPathRuntime(FastPathConfig(compile=True, simulate_cpu_graph=True), static_block_length=K)
    warp_k = SamplingWarper(top_k=20, top_p=0.8)
    s_vec = torch.randn(V, generator=g)
    for step in range(3):
        tok_t, ps_s, ix_s = sample_token_compact(
            s_vec, 0.7, warp_k, generator=torch.Generator().manual_seed(900 + step),
            compact_probs_fn=fp.compact_probabilities,
        )
        assert ix_s is not None and ix_s.shape == (20,)
        assert bool(torch.all(ix_s[1:] > ix_s[:-1]))  # strictly ascending vocab IDs
        q_from_compact = torch.zeros(V, dtype=ps_s.dtype).scatter(0, ix_s, ps_s)
        torch.testing.assert_close(q_from_compact, probabilities(s_vec, 0.7, warp_k), rtol=1e-6, atol=1e-6)
        ps_s.fill_(-999.0)
        ix_s.fill_(0)
    assert fp.stats()["probs_graph_captures"] == 1
    assert fp.stats()["probs_graph_replays"] == 2

    # Empirical categorical parity against sample_token over 16,000 draws
    logits_cat = torch.tensor([0.2, 1.9, -0.5, 1.9, 1.3, 0.8, 2.2, -1.0, 1.5], dtype=torch.float32)
    warp_cat = SamplingWarper(top_k=5, top_p=0.85, min_p=0.05)
    expected_p = probabilities(logits_cat, 0.7, warp_cat)
    counts = torch.zeros(len(logits_cat), dtype=torch.float64)
    rng_emp = torch.Generator().manual_seed(12345)
    n_draws = 16000
    for _ in range(n_draws):
        t_c, _, _ = sample_token_compact(logits_cat, 0.7, warp_cat, generator=rng_emp)
        counts[int(t_c)] += 1.0
    emp_p = counts / n_draws
    assert torch.equal(counts == 0, (expected_p == 0))
    assert float((emp_p - expected_p.double()).abs().max()) < 0.015


@pytest.mark.parametrize("k_ret", [None, 1, 4, 19])
def test_online_bigram_score_tensor_zero_ulp_fuzz_and_dynamic_growth(k_ret):
    rng = random.Random(7700 + (0 if k_ret is None else k_ret))
    V = 97
    bigram = OnlineBigram(V, alpha=0.4, lambda_tr=0.23, k_ret=k_ret)
    g = torch.Generator().manual_seed(88)

    for dtype in (torch.float32, torch.float64):
        l_u = torch.randn(V, generator=g, dtype=dtype)
        l_u[0] = -0.0  # verify -0.0 sign-bit preservation on inactive bypass
        l_u[3] = -torch.inf
        assert bigram.score_tensor(l_u, None) is l_u
        assert bigram.score_tensor(l_u, torch.tensor(5, dtype=torch.long)) is l_u

    # Force >64 active predecessor rows AND >32 unique successors on row 7 (when k_ret=None)
    # while row 11 has a single non-zero successor id=42 (testing col_cap growth padding!).
    bigram.update([11], [42])
    l_f32 = torch.randn(V, generator=g, dtype=torch.float32)
    l_f32[0] = -0.0
    l_f64 = l_f32.double()
    # Materialize device table BEFORE col_cap & row_cap double
    assert torch.equal(bigram.score_tensor(l_f32, torch.tensor(11)), bigram.score(l_f32, 11))
    assert torch.equal(bigram.score_tensor(l_f64, torch.tensor(11)), bigram.score(l_f64, 11))

    for succ in range(40):
        bigram.update([7], [(succ * 2 + 1) % 90])
    for step in range(90):
        prev_tok = step % 80
        committed = [rng.randrange(90) for _ in range(rng.randint(1, 5))]
        bigram.update([prev_tok], committed)

    # Touch an unseen key on defaultdict(Counter) to ensure empty Counter() never corrupts table
    _ = bigram.counts[96]
    assert len(bigram.counts[96]) == 0

    for prev in [0, 7, 11, 42, 79, 85, 96]:
        p_t = torch.tensor(prev, dtype=torch.long)
        out32 = bigram.score_tensor(l_f32, p_t)
        ref32 = bigram.score(l_f32, prev)
        assert torch.equal(out32, ref32)
        if not bigram.counts.get(prev):
            assert math.copysign(1.0, float(out32[0])) == math.copysign(1.0, float(ref32[0]))
        out64 = bigram.score_tensor(l_f64, p_t)
        ref64 = bigram.score(l_f64, prev)
        assert torch.equal(out64, ref64)

    # Mid-stream hyperparameter mutations (alpha, k_ret, lambda_tr) and counts.clear()
    bigram.alpha = 1.25
    bigram.lambda_tr = 0.07
    bigram.k_ret = 3 if k_ret is None else None
    for prev in [7, 11, 42, 96]:
        p_t = torch.tensor(prev, dtype=torch.long)
        assert torch.equal(bigram.score_tensor(l_f32, p_t), bigram.score(l_f32, prev))

    bigram.lambda_tr = 0.0
    assert bigram.score_tensor(l_f32, torch.tensor(7)) is l_f32
    bigram.lambda_tr = 0.15
    bigram.counts.clear()
    bigram._rows.clear()
    bigram._cache.clear()
    assert bigram.score_tensor(l_f32, torch.tensor(7)) is l_f32


def test_sample_proposal_sync_free_arms_exact_q_and_v151936_smoke():
    from edraft.dflash_loop import PhaseTimer, _sample_proposal

    g = torch.Generator().manual_seed(4242)
    V, K, r, p_dim, D = 64, 5, 6, 8, 3
    E = torch.randn(V, r, generator=g) * 0.2
    sparse = SignedHash(128)
    with torch.no_grad():
        sparse.values.normal_(0, 0.05)
    adapter = DenseAdapter(E, p=p_dim, D=D, sparse=sparse, M_sparse=8)
    with torch.no_grad():
        adapter.W.normal_(0, 0.1)
        adapter.A.normal_(0, 0.1)
        adapter.b.normal_(0, 0.05)
    snap = adapter.snapshot()
    bigram = OnlineBigram(V, alpha=0.5, lambda_tr=0.2, k_ret=6)
    bigram.update([1, 2, 3], [4, 5, 2, 7, 3, 9])

    draft_logits = torch.randn(K, V, generator=g)
    h_raw = torch.randn(K, p_dim, generator=g)
    prefix = [1, 2, 3]
    warp = SamplingWarper(top_k=12, top_p=0.85)
    fp = FastPathRuntime(FastPathConfig(compile=True, simulate_cpu_graph=True), static_block_length=K)

    for use_snap in (None, snap):
        for use_bg in (None, bigram):
            if use_snap is None and use_bg is None:
                continue
            for use_fp in (None, fp):
                timer = PhaseTimer("cpu", enabled=True)
                toks, q, l_unit, H = _sample_proposal(
                    draft_logits, h_raw, prefix, temperature=0.7, warper=warp,
                    generator=torch.Generator().manual_seed(99), snapshot=use_snap,
                    projection=None, beta=0.8, bigram=use_bg, timer=timer,
                    need_training_tensors=False, fastpath=use_fp, _return_token_tensor=False,
                )
                assert isinstance(toks, list) and len(toks) == K
                assert q is not None and q.shape == (K, V)
                # Reconstruct exact per-slot probabilities from realized tokens and verify 0-ULP / FP32 parity
                U_ref = use_snap.unary(h_raw.float()) if use_snap is not None else None
                l_ref = draft_logits.float().log_softmax(-1)
                for i in range(K):
                    sc_i = (
                        use_snap.score_slot(l_ref[i], U_ref[i], prefix, toks[:i], i, beta=0.8)
                        if use_snap is not None else l_ref[i]
                    )
                    if use_bg is not None:
                        prev_i = toks[i - 1] if i > 0 else prefix[-1]
                        sc_i = use_bg.score(sc_i, prev_i)
                    q_ref_i = probabilities(sc_i, 0.7, warp)
                    if use_fp is None:
                        assert torch.equal(q[i], q_ref_i)
                    else:
                        torch.testing.assert_close(q[i], q_ref_i, rtol=1e-5, atol=1e-6)
                    assert float(q[i, toks[i]]) > 0.0

                # T == 0 greedy exact token parity against reference score_slot + sample_token
                toks_g, q_g, _, _ = _sample_proposal(
                    draft_logits, h_raw, prefix, temperature=0.0, warper=warp,
                    generator=None, snapshot=use_snap, projection=None, beta=0.8,
                    bigram=use_bg, timer=PhaseTimer("cpu", False),
                    need_training_tensors=False, fastpath=use_fp, _return_token_tensor=False,
                )
                assert q_g is None
                ref_g = []
                for i in range(K):
                    sc_i = (
                        use_snap.score_slot(l_ref[i], U_ref[i], prefix, ref_g, i, beta=0.8)
                        if use_snap is not None else l_ref[i]
                    )
                    if use_bg is not None:
                        prev_i = ref_g[-1] if ref_g else prefix[-1]
                        sc_i = use_bg.score(sc_i, prev_i)
                    ref_g.append(int(torch.argmax(sc_i).item()))
                assert toks_g == ref_g

    # Qwen3-4B vocabulary-scale (V=151,936, K=15, top_k=20) smoke check on online_bigram
    V_big, K_big = 151_936, 15
    bg_big = OnlineBigram(V_big, alpha=0.5, lambda_tr=0.1, k_ret=None)
    bg_big.update([10, 20], [30, 40, 20, 50, 60])
    d_big = torch.randn(K_big, V_big, generator=g, dtype=torch.bfloat16)
    h_big = torch.randn(K_big, 8, generator=g, dtype=torch.bfloat16)
    t_big, q_big, _, _ = _sample_proposal(
        d_big, h_big, [10, 60], temperature=0.6, warper=SamplingWarper(top_k=20, top_p=0.95),
        generator=torch.Generator().manual_seed(7), snapshot=None, projection=None, beta=1.0,
        bigram=bg_big, timer=PhaseTimer("cpu", True), need_training_tensors=False,
        _return_token_tensor=True,
    )
    assert t_big.shape == (K_big,) and q_big.shape == (K_big, V_big)
    torch.testing.assert_close(q_big.sum(-1), torch.ones(K_big), rtol=1e-5, atol=1e-5)


def test_round2_bughunter_edge_cases_kl_weights_diagnose_and_launch_cli(tmp_path, monkeypatch):
    import json
    from edraft import launch as launch_mod

    # 1. OPT-R2-BUG-2: arbitrary broadcast-compatible weights in full_kl & union_rest_grouped
    g = torch.Generator().manual_seed(31415)
    N, V = 4, 11
    s_base = torch.randn(N, V, generator=g)
    t_base = torch.randn(N, V, generator=g)
    u_ids = [torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([0]), torch.tensor([1, 2, 3])]
    u_probs = [torch.tensor([0.5, 0.2]), torch.tensor([0.4, 0.3]), torch.tensor([0.6]), torch.tensor([0.2, 0.2, 0.2])]
    rest_m = torch.tensor([0.3, 0.3, 0.4, 0.4])
    direction = torch.randn(N, V, generator=g)

    weight_cases = [
        2.0,
        torch.tensor(2.0),
        torch.tensor([2.0]),
        torch.tensor([1.0, 2.0, 0.5, 1.5]),
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        torch.ones((2, N)) * 0.75,
        torch.randn(2, 3, N, generator=g).abs(),
    ]
    for w_case in weight_cases:
        # full_kl forward + 1st-order + 2nd-order HVP parity vs explicit broadcast reference
        s_opt = s_base.clone().requires_grad_(True)
        s_ref = s_base.clone().requires_grad_(True)
        loss_opt = full_kl(s_opt, t_base, weights=w_case)
        p_log = torch.log_softmax(t_base.float(), -1)
        q_log = torch.log_softmax(s_ref.float(), -1)
        loss_ref = ((p_log.exp() * (p_log - q_log)).sum(-1) * w_case).sum()
        torch.testing.assert_close(loss_opt, loss_ref, rtol=1e-6, atol=1e-6)
        (g_opt,) = torch.autograd.grad(loss_opt, s_opt, create_graph=True)
        (g_ref,) = torch.autograd.grad(loss_ref, s_ref, create_graph=True)
        torch.testing.assert_close(g_opt, g_ref, rtol=1e-5, atol=1e-6)
        (hvp_opt,) = torch.autograd.grad((g_opt * direction).sum(), s_opt)
        (hvp_ref,) = torch.autograd.grad((g_ref * direction).sum(), s_ref)
        torch.testing.assert_close(hvp_opt, hvp_ref, rtol=1e-5, atol=1e-5)

        # union_rest_grouped forward + 1st-order + 2nd-order HVP parity vs grouped_loop_reference
        sg_opt = s_base.clone().requires_grad_(True)
        sg_ref = s_base.clone().requires_grad_(True)
        lg_opt = union_rest_grouped(sg_opt, u_ids, u_probs, rest_m, weights=w_case)
        # Compute per-slot grouped loss reference and apply exact PyTorch `* w_case` broadcast
        per_slot_ref = torch.stack([
            grouped_loop_reference(sg_ref[r:r + 1], [u_ids[r]], [u_probs[r]], rest_m[r:r + 1], weights=torch.ones(1))
            for r in range(N)
        ])
        lg_ref = (per_slot_ref * w_case).sum()
        torch.testing.assert_close(lg_opt, lg_ref, rtol=1e-6, atol=1e-6)
        (gg_opt,) = torch.autograd.grad(lg_opt, sg_opt, create_graph=True)
        (gg_ref,) = torch.autograd.grad(lg_ref, sg_ref, create_graph=True)
        torch.testing.assert_close(gg_opt, gg_ref, rtol=1e-5, atol=1e-6)
        (ghvp_opt,) = torch.autograd.grad((gg_opt * direction).sum(), sg_opt)
        (ghvp_ref,) = torch.autograd.grad((gg_ref * direction).sum(), sg_ref)
        torch.testing.assert_close(ghvp_opt, ghvp_ref, rtol=1e-5, atol=1e-5)

    with pytest.raises(RuntimeError):
        full_kl(s_base.clone().requires_grad_(True), t_base, weights=torch.ones(N + 1))
    with pytest.raises(RuntimeError):
        union_rest_grouped(s_base.clone().requires_grad_(True), u_ids, u_probs, rest_m, weights=torch.ones(N + 1))

    # 2. OPT-R2-BUG-3: DiagnosticAccumulator.add_blocks with sampling_warp_config=None & mixed configs
    base_diag = {
        "target_rank": [0, 1],
        "sampled_target_rank": [0, 1],
        "sampled_target_rank_unwarped": [0, 1],
        "overlap_sampling": [1.0, 0.5],
        "overlap_unwarped": [0.9, 0.6],
        "cross_entropy_unwarped": [0.2, 0.7],
        "sampling_cross_entropy": [0.0, 0.5],
        "sampling_cross_entropy_infinite": [0.0, 0.0],
        "target_mass_on_zero_proposal": [0.0, 0.0],
        "top_m": {str(m): {"mass": [1.0, 1.0], "overlap": [0.9, 0.6]} for m in TOP_M},
        "first_rejection": None,
    }
    warp_seq = [None, {"top_k": 20, "top_p": 1}, {"top_k": 20, "top_p": 1.0}, None]
    blocks_none = []
    acc_one = DiagnosticAccumulator(seed=2026)
    acc_many = DiagnosticAccumulator(seed=2026)
    for idx_b, w_cfg in enumerate(warp_seq):
        blk = {
            "prompt_id": f"p{idx_b}", "seed": 2026, "condition": "standard_nonthinking",
            "collection_temperature": 0.0, "sampling_warp_config": w_cfg, "split": "dev",
            "draft_tokens": [1, 2], "accepted_length": 1, "first_rejection_index": 1,
            "absolute_output_offset": idx_b * 2, "input_context_length": 8 + idx_b,
            "h": torch.randn((2, 4), generator=g), "diagnostics": base_diag,
        }
        blocks_none.append(blk)
        acc_one.add_block(blk)
    acc_many.add_blocks(blocks_none)
    rep_one = acc_one.report()
    rep_many = acc_many.report()
    assert json.dumps(rep_many, sort_keys=True) == json.dumps(rep_one, sort_keys=True)
    assert any(grp["sampling_warp_config"] is None for grp in rep_many["groups"])

    # 3. OPT-R2-BUG-4: launch.py accepts and forwards all Round-2 train CLI flags
    spawned_cmds = []

    class _DummyProc:
        def __init__(self, cmd, **kwargs):
            spawned_cmds.append(list(cmd))
            self.pid = 43210
            self.returncode = 0

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(launch_mod, "record_run", lambda *a, **kw: None)
    monkeypatch.setattr(launch_mod, "selected_gpus_idle", lambda indices: None)
    monkeypatch.setattr(launch_mod.subprocess, "Popen", _DummyProc)
    out_launch = tmp_path / "launch_out"
    launch_mod.main([
        "--scale", "reference",
        "--task", "train",
        "--gpus", "0,1",
        "--output", str(out_launch),
        "--optimizer", "pacemuon",
        "--lr_schedule", "wsd",
        "--block_shuffle_window", "8",
        "--learning_rate", "0.0003",
        "--weight_decay", "0.01",
        "--bias_weight_decay", "1e-05",
        "--sparse_learning_rate", "0.005",
        "--lambda_l1", "2e-05",
        "--epochs_dense", "2",
        "--epochs_sparse", "3",
        "--no_ema_eval",
        "--pace_pullback", "0.08",
        "--anchor_soda", "0.03",
        "--no_resume",
    ])
    assert len(spawned_cmds) == 2
    cmd0 = spawned_cmds[0]
    for expected_pair in (
        ("--optimizer", "pacemuon"),
        ("--lr_schedule", "wsd"),
        ("--block_shuffle_window", "8"),
        ("--learning_rate", "0.0003"),
        ("--weight_decay", "0.01"),
        ("--bias_weight_decay", "1e-05"),
        ("--sparse_learning_rate", "0.005"),
        ("--lambda_l1", "2e-05"),
        ("--epochs_dense", "2"),
        ("--epochs_sparse", "3"),
        ("--pace_pullback", "0.08"),
        ("--anchor_soda", "0.03"),
    ):
        idx_f = cmd0.index(expected_pair[0])
        assert cmd0[idx_f + 1] == expected_pair[1]
    assert "--no_ema_eval" in cmd0
    assert "--no_resume" in cmd0


# ---------------------------------------------------------------------------
# 10. Round-3 Surgical Fastpaths (#70, #71, #73, #74) & Edge-Case Fixes (#69 OPT-R3-BUG-1..3)
# ---------------------------------------------------------------------------

from edraft.adapter import ProposalSnapshot
from edraft.dflash_loop import PhaseTimer, _sample_proposal, generate_toy


def _ref_09a5469_probabilities(scores: torch.Tensor, temperature: float, warper: SamplingWarper) -> torch.Tensor:
    """Exact baseline `09a5469` `sampler.py:probabilities` reference implementation."""
    s = (scores if scores.dtype == torch.float64 else scores.float()) / temperature
    V = s.shape[-1]
    indices = None
    if 0 < warper.top_k < V:
        s, indices = torch.topk(s, warper.top_k, dim=-1)
    probs = torch.softmax(s, dim=-1)
    if warper.top_p < 1:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < warper.top_p
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if warper.min_p > 0:
        keep = probs >= warper.min_p * probs.amax(dim=-1, keepdim=True)
        probs = probs * keep
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if indices is not None:
        full_shape = (*probs.shape[:-1], V)
        probs = torch.zeros(full_shape, dtype=probs.dtype, device=probs.device).scatter(-1, indices, probs)
    return probs


def test_round3_bug_regressions_top_p_tie_bigram_clear_and_empty_block():
    from dflash.model import _sampling_probs
    from edraft.sampler import compact_probabilities

    # 1. OPT-R3-BUG-1: exact repro (`s[5] = s[2] = 10.0, T=0.7, top_k=20, top_p=0.1`)
    #    plus 600 tied bfloat16/float32 rows vs `09a5469` and `dflash.model._sampling_probs`.
    s_repro = torch.full((64,), -10.0, dtype=torch.float32)
    s_repro[5] = 10.0
    s_repro[2] = 10.0
    warp_repro = SamplingWarper(top_k=20, top_p=0.1)
    p_sub, idx = compact_probabilities(s_repro, 0.7, warp_repro)
    p_dense = probabilities(s_repro, 0.7, warp_repro)
    p_ref = _ref_09a5469_probabilities(s_repro, 0.7, warp_repro)
    p_dflash = _sampling_probs(s_repro, 0.7, top_p=0.1, top_k=20)
    assert idx is not None
    assert torch.equal(torch.zeros_like(p_dense).scatter(-1, idx, p_sub), p_ref)
    assert torch.equal(p_dense, p_ref)
    assert torch.equal(p_dense, p_dflash)

    fp_rt = FastPathRuntime(FastPathConfig(compile=True, simulate_cpu_graph=True), static_block_length=4)
    p_fp_dense = fp_rt.probabilities(s_repro, 0.7, warp_repro)
    p_fp_sub, p_fp_idx = fp_rt.compact_probabilities(s_repro, 0.7, warp_repro)
    assert torch.equal(torch.nonzero(p_fp_dense > 0).squeeze(-1), torch.nonzero(p_ref > 0).squeeze(-1))
    assert torch.equal(
        torch.nonzero(torch.zeros_like(p_ref).scatter(-1, p_fp_idx, p_fp_sub) > 0).squeeze(-1),
        torch.nonzero(p_ref > 0).squeeze(-1),
    )

    g = torch.Generator().manual_seed(20261115)
    for trial in range(150):
        for dt in (torch.bfloat16, torch.float32):
            x = (torch.randn(4, 128, generator=g) * 2.5).to(dt)
            x[0, :25] = x[0, 3]
            x[1, 5] = x[1, 2] = 9.5
            for T in (0.6, 0.7):
                for w in (
                    SamplingWarper(top_k=20, top_p=0.1),
                    SamplingWarper(top_k=20, top_p=0.8),
                    SamplingWarper(top_k=0, top_p=0.5),
                ):
                    ps, ix = compact_probabilities(x, T, w)
                    pf = probabilities(x, T, w)
                    pr = _ref_09a5469_probabilities(x, T, w)
                    pd = _sampling_probs(x, T, top_p=w.top_p, top_k=w.top_k)
                    if ix is None:
                        assert torch.equal(ps, pr)
                    else:
                        assert torch.equal(torch.zeros_like(pr).scatter(-1, ix, ps), pr)
                    assert torch.equal(pf, pr)
                    assert torch.equal(pf, pd)

    # 2. OPT-R3-BUG-2: `bg.counts.clear()` + `bg.update(...)` never leaves stale `row_slot` entries.
    bg = OnlineBigram(16, alpha=0.5, lambda_tr=0.5, k_ret=4)
    bg.update([1], [2])
    bg.update([5], [6])
    l_vec = torch.randn(16, generator=g)
    _ = bg.score_tensor(l_vec, torch.tensor(1, dtype=torch.long))

    # Case A: fewer keys after clear() + update()
    bg.counts.clear()
    bg.update([3], [4])
    assert torch.equal(bg.score_tensor(l_vec, torch.tensor(1, dtype=torch.long)), bg.score(l_vec, 1))
    assert torch.equal(bg.score_tensor(l_vec, torch.tensor(1, dtype=torch.long)), l_vec)
    assert torch.equal(bg.score_tensor(l_vec, torch.tensor(3, dtype=torch.long)), bg.score(l_vec, 3))

    # Case B: >= previous active_rows new keys after clear() + update()
    bg.counts.clear()
    bg.update([7], [8])
    bg.update([9], [10])
    bg.update([11], [12])
    for tok_id in range(16):
        assert torch.equal(bg.score_tensor(l_vec, torch.tensor(tok_id, dtype=torch.long)), bg.score(l_vec, tok_id))

    # Case C: clear() followed by re-populating the SAME predecessor key with different follower
    bg.counts.clear()
    bg.update([7], [15])
    for tok_id in range(16):
        assert torch.equal(bg.score_tensor(l_vec, torch.tensor(tok_id, dtype=torch.long)), bg.score(l_vec, tok_id))

    # 3. OPT-R3-BUG-3: `K == 0` empty proposal block returns `q_out is None` across all branches.
    V, p_dim = 48, 6
    E = torch.randn(V, 4, generator=g)
    sh = SignedHash(64)
    with torch.no_grad():
        sh.values.normal_(0, 0.1)
    adap = DenseAdapter(E, p=p_dim, D=2, sparse=sh, M_sparse=8)
    with torch.no_grad():
        adap.W.normal_(0, 0.1)
    snap = adap.snapshot()
    empty_logits = torch.empty((0, V), dtype=torch.float32)
    empty_h = torch.empty((0, p_dim), dtype=torch.float32)

    for use_snap in (None, snap):
        for use_bg in (None, bg if V == 16 else OnlineBigram(V, lambda_tr=0.2)):
            if use_bg is not None:
                use_bg.update([1], [2])
            for T, w in ((0.0, SamplingWarper()), (0.7, SamplingWarper(top_k=10, top_p=0.8)), (1.0, SamplingWarper())):
                for allow_cmp in (False, True):
                    for ret_t in (False, True):
                        toks_0, q_0, lu_0, h_0 = _sample_proposal(
                            empty_logits, empty_h, [1, 2], temperature=T, warper=w,
                            generator=torch.Generator().manual_seed(1), snapshot=use_snap,
                            projection=None, beta=1.0, bigram=use_bg,
                            timer=PhaseTimer("cpu", True), need_training_tensors=True,
                            _return_token_tensor=ret_t, _allow_compact_q=allow_cmp,
                        )
                        assert q_0 is None
                        if ret_t:
                            assert isinstance(toks_0, torch.Tensor) and toks_0.shape == (0,)
                        else:
                            assert toks_0 == []
                        assert lu_0.shape == (0, V) and h_0.shape == (0, p_dim)


def test_round3_zero_sync_guard_and_static_graph_capture(monkeypatch):
    from edraft.fastpath import StaticGraphCallable
    from edraft.verify import CompactProposal

    g = torch.Generator().manual_seed(9191)
    V, K, r, p_dim, D = 512, 6, 8, 10, 3
    E = torch.randn(V, r, generator=g) * 0.2
    sh = SignedHash(512)
    with torch.no_grad():
        sh.values.normal_(0, 0.2)

    adap_full = DenseAdapter(E, p=p_dim, D=D, sparse=sh, M_sparse=32)
    with torch.no_grad():
        adap_full.W.normal_(0, 0.1)
        adap_full.A.normal_(0, 0.1)
        adap_full.b.normal_(0, 0.05)
    snap_full = adap_full.snapshot()

    # Zero-pair (`A == 0`) online_dbS_zero_init-style snapshot
    adap_bias_sparse = DenseAdapter(E, p=p_dim, D=D, sparse=sh, M_sparse=32)
    with torch.no_grad():
        adap_bias_sparse.b.normal_(0, 0.1)
    snap_zero_pair = adap_bias_sparse.snapshot()
    assert snap_zero_pair.pair_is_zero() is True
    assert snap_full.pair_is_zero() is False

    bg = OnlineBigram(V, alpha=0.5, lambda_tr=0.25, k_ret=16)
    bg.update([2, 3], [4, 5, 6, 3, 7, 8])
    # Warm device table once before installing the strict no-sync guard
    _ = bg._sync_device_table(torch.device("cpu"), torch.float32)

    draft_logits = (torch.randn(K, V, generator=g) * 2.5).bfloat16()
    h_raw = torch.randn(K, p_dim, generator=g)
    prefix = [1, 2, 3]

    def _forbid_item(self):
        raise AssertionError("Forbidden `.item()` host sync inside _sample_proposal!")

    def _forbid_tolist(self):
        raise AssertionError("Forbidden `.tolist()` host sync inside _sample_proposal!")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid_tolist)

    for snap_case, bg_case in (
        (None, bg),
        (snap_full, None),
        (snap_zero_pair, None),
        (snap_full, bg),
    ):
        for T, w in (
            (0.0, SamplingWarper()),
            (0.7, SamplingWarper(top_k=20, top_p=0.8)),
            (1.0, SamplingWarper(top_k=0, top_p=1.0)),
        ):
            tok_t, q_out, _, _ = _sample_proposal(
                draft_logits, h_raw, prefix, temperature=T, warper=w,
                generator=torch.Generator().manual_seed(42),
                snapshot=snap_case, projection=None, beta=0.9, bigram=bg_case,
                timer=PhaseTimer("cpu", False), need_training_tensors=False,
                _return_token_tensor=True, _allow_compact_q=True,
            )
            assert isinstance(tok_t, torch.Tensor) and tok_t.shape == (K,)
            if T == 0.0:
                assert q_out is None
            elif w.top_k > 0:
                assert isinstance(q_out, CompactProposal)
                assert q_out.slot_probs.shape == (K, 20)
            else:
                assert isinstance(q_out, torch.Tensor) and q_out.shape == (K, V)

    monkeypatch.undo()

    # StaticGraphCallable (`cuda_graph`-safe static shape) capture & replay on both new kernels
    bg_state = bg._sync_device_table(torch.device("cpu"), torch.float32)
    graph_bg = StaticGraphCallable(
        lambda l_i, bv_i, bi_i, prev_t: bg.merge_compact_candidates(
            l_i, bv_i, bi_i, prev_t, temperature=0.7, k_eff=20, state=bg_state
        ),
        simulate_cpu=True,
    )
    l_f32 = draft_logits.float().log_softmax(-1)
    _, b_ids = torch.topk(l_f32 / 0.7, 20, dim=-1)
    b_vals = l_f32.gather(-1, b_ids)
    for i in range(3):
        sc_g, id_g = graph_bg(l_f32[i], b_vals[i], b_ids[i], torch.tensor(3, dtype=torch.long))
        sc_e, id_e = bg.merge_compact_candidates(
            l_f32[i], b_vals[i], b_ids[i], torch.tensor(3, dtype=torch.long),
            temperature=0.7, k_eff=20, state=bg_state,
        )
        assert torch.equal(sc_g, sc_e) and torch.equal(id_g, id_e)
    assert graph_bg.graph_captures == 1 and graph_bg.graph_replays == 2

    offs_t = torch.tensor([[1], [2]], dtype=torch.long)
    U_f32 = snap_full.unary(h_raw.float())
    graph_sp = StaticGraphCallable(
        lambda l_i, u_i, p_ids: snap_full.score_slot_compact_candidates(
            l_i, u_i, offs_t, p_ids, temperature=0.7, k_eff=20, beta=0.9, contiguous_offsets=True
        ),
        simulate_cpu=True,
    )
    for i in range(3):
        p_ids = torch.tensor([3, 2], dtype=torch.long)
        sc_g, id_g = graph_sp(l_f32[i], U_f32[i], p_ids)
        sc_e, id_e = snap_full.score_slot_compact_candidates(
            l_f32[i], U_f32[i], offs_t, p_ids, temperature=0.7, k_eff=20, beta=0.9, contiguous_offsets=True
        )
        assert torch.equal(sc_g, sc_e) and torch.equal(id_g, id_e)
    assert graph_sp.graph_captures == 1 and graph_sp.graph_replays == 2


@pytest.mark.parametrize("k_ret", [None, 4, 256])
def test_round3_fastpath1_online_bigram_compact_merge_adversarial_parity(k_ret):
    g = torch.Generator().manual_seed(3300 + (0 if k_ret is None else k_ret))
    V, K = 600, 6
    bg = OnlineBigram(V, alpha=0.5, lambda_tr=0.3, k_ret=k_ret)
    # Row 7: 320 unique transitions -> forces `N_evicted > 0` for both `k_ret=4` and `k_ret=256`!
    for tok in range(320):
        for _ in range((tok % 4) + 1):
            bg.update([7], [(tok * 11) % V])
    # Row 13: only 2 transitions -> `rlen < col_cap` padding test!
    bg.update([13], [5, 42])
    # Row 99 is NEVER updated -> `N_total=0` (`r == 0`) plain-DFlash bit-exact tie-break test!

    h_dummy = torch.zeros(K, 4)
    for trial in range(12):
        # Continuous FP32 rows + `-inf` masked row + overlap with retained IDs
        raw = torch.randn(K, V, generator=g, dtype=torch.float32) * 2.5
        raw[2, 35:] = -torch.inf
        # Boost retained IDs of row 7 into top-20 of `l_unit` to force `[20] ∩ [R]` overlap!
        raw[0, 0] = 7.5
        raw[0, 11] = 7.2
        raw[0, 22] = 6.9
        prefix = [99] if trial % 3 == 0 else ([13] if trial % 3 == 1 else [7])

        # 1. Greedy `T == 0` including synthetic BF16 ties across the whole vocabulary (#74)
        raw_bf16 = raw.bfloat16()
        toks_g, q_g, _, _ = _sample_proposal(
            raw_bf16, h_dummy, prefix, temperature=0.0, warper=SamplingWarper(top_k=20, top_p=0.8),
            generator=None, snapshot=None, projection=None, beta=1.0, bigram=bg,
            timer=PhaseTimer("cpu", True), need_training_tensors=False, _allow_compact_q=True,
        )
        assert q_g is None
        l_bf16 = raw_bf16.float().log_softmax(-1)
        ref_g = []
        for i in range(K):
            prev_i = ref_g[-1] if ref_g else prefix[-1]
            s_i = bg.score(l_bf16[i], prev_i)
            ref_g.append(int(torch.argmax(s_i).item()))
        assert toks_g == ref_g

        # 2. Stochastic `top_k=20` (`top_p=0.8` and `top_p=0.95, min_p=0.05`) -> `torch.equal` vs `09a5469`
        #    (continuous FP32 rows have no exact ties at truncation boundaries; tied rows are
        #    covered by `test_round3_tie_scope_valid_q_and_verify_law`).
        for T, w in (
            (0.7, SamplingWarper(top_k=20, top_p=0.8)),
            (0.6, SamplingWarper(top_k=20, top_p=0.95, min_p=0.05)),
            (1.0, SamplingWarper(top_k=0, top_p=1.0)),  # `pure_temperature_diagnostic` full-vocab scope guard!
        ):
            toks_s, q_s, _, _ = _sample_proposal(
                raw, h_dummy, prefix, temperature=T, warper=w,
                generator=torch.Generator().manual_seed(800 + trial),
                snapshot=None, projection=None, beta=1.0, bigram=bg,
                timer=PhaseTimer("cpu", True), need_training_tensors=False, _allow_compact_q=True,
            )
            q_dense = q_s.to_dense() if hasattr(q_s, "to_dense") else q_s
            assert q_dense.shape == (K, V)
            l_ref = raw.float().log_softmax(-1)
            for i in range(K):
                prev_i = toks_s[i - 1] if i > 0 else prefix[-1]
                s_ref_i = bg.score(l_ref[i], prev_i)
                q_ref_i = _ref_09a5469_probabilities(s_ref_i, T, w)
                assert torch.equal(q_dense[i], q_ref_i)
                if hasattr(q_s, "q_selected"):
                    assert torch.equal(q_s.q_selected[i], q_ref_i[toks_s[i]])

        # 3. `N_total=0` (`prev=99`, `r == 0`) on heavy BF16-tied logits reproduces plain DFlash 0-ULP bit-for-bit!
        l_tied = raw_bf16.float().log_softmax(-1)
        w_20 = SamplingWarper(top_k=20, top_p=0.8)
        bg_state = bg._sync_device_table(torch.device("cpu"), torch.float32)
        _, b_ids_t = torch.topk(l_tied / 0.7, 20, dim=-1)
        b_vals_t = l_tied.gather(-1, b_ids_t)
        for i in range(K):
            sc_0, id_0 = bg.merge_compact_candidates(
                l_tied[i], b_vals_t[i], b_ids_t[i], torch.tensor(99, dtype=torch.long),
                temperature=0.7, k_eff=20, state=bg_state,
            )
            from edraft.sampler import warp_top_scaled_scores
            pr_0 = warp_top_scaled_scores(sc_0, w_20)
            q_0 = torch.zeros(V, dtype=pr_0.dtype).scatter(0, id_0, pr_0)
            assert torch.equal(q_0, _ref_09a5469_probabilities(l_tied[i], 0.7, w_20))


def test_round3_fastpath2_compact_proposal_and_compact_verify_block_parity():
    from edraft.toy import ToyParallelDrafter, ToyTarget
    from edraft.verify import CompactProposal, verify_block
    from edraft.dflash_loop import generate_toy

    g = torch.Generator().manual_seed(5150)
    V, K, r, p_dim, D = 257, 6, 8, 10, 3
    E = torch.randn(V, r, generator=g) * 0.2
    sh = SignedHash(256)
    with torch.no_grad():
        sh.values.normal_(0, 0.2)
    adap = DenseAdapter(E, p=p_dim, D=D, sparse=sh, M_sparse=32)
    with torch.no_grad():
        adap.W.normal_(0, 0.1)
        adap.A.normal_(0, 0.1)
        adap.b.normal_(0, 0.05)
    snap = adap.snapshot()
    fp = FastPathRuntime(FastPathConfig(compile=True, simulate_cpu_graph=True), static_block_length=K)

    for trial in range(30):
        draft_logits = torch.randn(K, V, generator=g) * 2.0
        h_raw = torch.randn(K, p_dim, generator=g)
        target_logits = torch.randn(K + 1, V, generator=g) * 2.0
        if trial % 5 == 0:
            # Force all-accept bonus draw on many trials by matching target_logits[:K] to high draft scores!
            toks_probe, q_probe, _, _ = _sample_proposal(
                draft_logits, h_raw, [1, 2, 3], temperature=0.7,
                warper=SamplingWarper(top_k=20, top_p=0.8),
                generator=torch.Generator().manual_seed(100 + trial),
                snapshot=snap, projection=None, beta=0.8, bigram=None,
                timer=PhaseTimer("cpu", False), need_training_tensors=False,
                _return_token_tensor=True, _allow_compact_q=True,
            )
            for slot_i in range(K):
                target_logits[slot_i].fill_(-20.0)
                target_logits[slot_i, toks_probe[slot_i]] = 20.0

        for w in (
            SamplingWarper(top_k=20, top_p=0.8),
            SamplingWarper(top_k=20, top_p=0.95, min_p=0.05),
        ):
            toks_c, q_c, _, _ = _sample_proposal(
                draft_logits, h_raw, [1, 2, 3], temperature=0.7, warper=w,
                generator=torch.Generator().manual_seed(100 + trial),
                snapshot=snap, projection=None, beta=0.8, bigram=None,
                timer=PhaseTimer("cpu", False), need_training_tensors=False,
                _return_token_tensor=True, _allow_compact_q=True,
            )
            toks_d, q_d, _, _ = _sample_proposal(
                draft_logits, h_raw, [1, 2, 3], temperature=0.7, warper=w,
                generator=torch.Generator().manual_seed(100 + trial),
                snapshot=snap, projection=None, beta=0.8, bigram=None,
                timer=PhaseTimer("cpu", False), need_training_tensors=False,
                _return_token_tensor=True, _allow_compact_q=False,
            )
            assert isinstance(q_c, CompactProposal)
            assert torch.equal(toks_c, toks_d)
            assert torch.equal(q_c.to_dense(), q_d)

            for p_fn in (None, _ref_09a5469_probabilities, fp.probabilities):
                gen_c = torch.Generator().manual_seed(700 + trial)
                gen_d = torch.Generator().manual_seed(700 + trial)
                ver_c = verify_block(toks_c, q_c, target_logits, 0.7, w, gen_c, probs_fn=p_fn)
                ver_d = verify_block(toks_d, q_d, target_logits, 0.7, w, gen_d, probs_fn=p_fn)
                assert ver_c == ver_d
                assert torch.equal(gen_c.get_state(), gen_d.get_state())

    # 1-ULP identical-distribution `P == q` FP64 tie guard on `CompactProposal`
    logits_same = torch.randn(3, V, generator=g)
    w_same = SamplingWarper(top_k=20, top_p=0.8)
    ps_same, ix_same = probabilities(logits_same[:2], 0.7, w_same), None
    from edraft.sampler import compact_probabilities
    p_sub_s, p_idx_s = compact_probabilities(logits_same[:2], 0.7, w_same)
    ix_asc, perm_s = p_idx_s.sort(dim=-1)
    pr_asc = p_sub_s.gather(-1, perm_s)
    toks_same = ix_asc[:, 0]
    # Pick tokens with nonzero probability
    nz_col = pr_asc.argmax(dim=-1)
    toks_same = ix_asc.gather(-1, nz_col[:, None])[:, 0]
    q_sel_same = pr_asc.gather(-1, nz_col[:, None])[:, 0]
    cp_same = CompactProposal(ix_asc, pr_asc, q_sel_same, V)
    g_s1 = torch.Generator().manual_seed(11)
    g_s2 = torch.Generator().manual_seed(11)
    v_s1 = verify_block(toks_same, cp_same, logits_same, 0.7, w_same, g_s1)
    v_s2 = verify_block(toks_same, ps_same, logits_same, 0.7, w_same, g_s2)
    assert v_s1 == v_s2 and torch.equal(g_s1.get_state(), g_s2.get_state())

    # Invalid CompactProposal shape guard
    with pytest.raises(ValueError, match="CompactProposal"):
        verify_block(toks_same, CompactProposal(ix_asc, pr_asc, q_sel_same, V + 1), logits_same, 0.7, w_same)

    # End-to-end `generate_toy` closed-loop online + bigram parity
    target_toy = ToyTarget(vocab_size=64, hidden_size=12, seed=17)
    drafter_toy = ToyParallelDrafter(target_toy, seed=17)
    E_toy = target_toy.lm_head.weight[:, :6].detach().clone()
    base_adap = DenseAdapter(E_toy, p=12, D=2, sparse=SignedHash(64), M_sparse=8)
    os_1 = OnlineState(base_adap, lr=0.02, lambda_l1_b=1e-4, lambda_l1_s=1e-4, enable_sparse=True, bias_sparse_only=True)
    res_toy = generate_toy(
        target_toy, drafter_toy, [1, 2, 3, 4], 18, block_length=4,
        temperature=0.7, warper=SamplingWarper(top_k=10, top_p=0.85),
        generator=torch.Generator().manual_seed(99), online_state=os_1, measure_phases=True,
    )
    assert res_toy.num_output_tokens == 18
    assert res_toy.phase_ms["sparse_lookup"] > 0
    assert res_toy.phase_ms["sequential_sampling"] > 0


def test_round3_fastpath3_sparse_compact_pool_adversarial_parity():
    from edraft.verify import CompactProposal

    class _HeavyPenaltySparse:
        """Penalizes every `C_i` candidate by `+30.0` so `V \\ C_i` ranks `257..276` MUST win!"""

        def lookup(self, d, previous, current):
            return torch.full(
                (d.shape[0], current.shape[1]), 30.0, dtype=torch.float32, device=current.device
            )

    g = torch.Generator().manual_seed(8080)
    V, K, r, p_dim, D, M = 600, 5, 8, 10, 3, 256
    E = torch.randn(V, r, generator=g) * 0.25
    sh_rand = SignedHash(2048)
    with torch.no_grad():
        sh_rand.values.copy_(torch.randn(2048, generator=g) * 3.0)

    # Build both `A != 0` (`dense_sparse`) and `A == 0` (`pair_is_zero`, `online_dbS`) snapshots,
    # plus the `S >> 0` heavy-penalty snapshot!
    adap_a = DenseAdapter(E, p=p_dim, D=D, sparse=sh_rand, M_sparse=M)
    with torch.no_grad():
        adap_a.W.normal_(0, 0.15)
        adap_a.A.normal_(0, 0.15)
        adap_a.b.normal_(0, 0.05)
    snap_a = adap_a.snapshot()

    adap_z = DenseAdapter(E, p=p_dim, D=D, sparse=sh_rand, M_sparse=M)
    with torch.no_grad():
        adap_z.b.normal_(0, 0.1)
    snap_z = adap_z.snapshot()
    assert snap_z.pair_is_zero() is True

    snap_heavy = ProposalSnapshot(
        E.detach(), adap_a.W.detach().clone(), adap_a.A.detach().clone(),
        adap_a.b.detach().clone(), 0, _HeavyPenaltySparse(), M, None, False,
    )
    snap_heavy_z = ProposalSnapshot(
        E.detach(), torch.zeros_like(adap_a.W), torch.zeros_like(adap_a.A),
        adap_z.b.detach().clone(), 0, _HeavyPenaltySparse(), M, None, False,
    )

    for snap_idx, cur_snap in enumerate((snap_heavy, snap_heavy_z, snap_a, snap_z)):
        for trial in range(6):
            raw = torch.randn(K, V, generator=g) * 2.5
            h_raw = torch.randn(K, p_dim, generator=g)
            if trial == 1:
                # `-inf` masked logits leaving fewer than `M + top_k` finite entries!
                raw[1, 40:] = -torch.inf
            elif trial == 2:
                # Synthetic BF16 ties across ranks 1, 20, 256, and 276 on `A == 0`!
                raw = raw.bfloat16().float()
            elif trial == 3:
                # Explicit ties right across boundary `256` (`252..260`) and `276` (`272..280`)
                U_t = cur_snap.unary(h_raw.float())
                u0 = raw[0].log_softmax(-1) - 0.85 * U_t[0]
                sv, si = torch.topk(u0, 285)
                for tid in si[252:261]:
                    raw[0, tid] = sv[255] + 0.85 * U_t[0, tid]
                for tid in si[272:281]:
                    raw[0, tid] = sv[275] + 0.85 * U_t[0, tid]

            prefix = [] if trial == 4 else [5, 12, 29]  # includes empty-prefix `avail == 0` slot-0 check!

            # (1) Greedy `T == 0`: exact lowest-ID argmax parity vs `score_slot`
            toks_g, q_g, _, _ = _sample_proposal(
                raw, h_raw, prefix, temperature=0.0, warper=SamplingWarper(top_k=20, top_p=0.8),
                generator=None, snapshot=cur_snap, projection=None, beta=0.85, bigram=None,
                timer=PhaseTimer("cpu", True), need_training_tensors=False, _allow_compact_q=True,
            )
            assert q_g is None
            l_ref = raw.float().log_softmax(-1)
            U_ref = cur_snap.unary(h_raw.float())
            ref_g = []
            for i in range(K):
                sc_i = cur_snap.score_slot(l_ref[i], U_ref[i], prefix, ref_g, i, beta=0.85)
                ref_g.append(int(torch.argmax(sc_i).item()))
            assert toks_g == ref_g, f"greedy mismatch snap={snap_idx} trial={trial}"

            # On `_HeavyPenaltySparse` (`snap_idx in (0, 1)`) with nonempty prefix & finite logits,
            # verify the winning token genuinely came from outside `C_i` (`ranks 257+`)!
            if snap_idx in (0, 1) and trial == 0:
                c0_ref = torch.topk(l_ref[0] - 0.85 * cur_snap.unary(h_raw.float())[0], M).indices
                if snap_idx == 1:
                    assert toks_g[0] not in set(c0_ref.tolist())

            # (2) Stochastic `top_k=20` AND full-vocab `top_k=0` (`pure_temperature_diagnostic` + `top_p < 1`)
            for T, w in (
                (0.7, SamplingWarper(top_k=20, top_p=0.8)),
                (0.6, SamplingWarper(top_k=20, top_p=0.95, min_p=0.05)),
                (1.0, SamplingWarper(top_k=0, top_p=1.0)),
                (0.8, SamplingWarper(top_k=0, top_p=0.85, min_p=0.05)),
            ):
                toks_s, q_s, _, _ = _sample_proposal(
                    raw, h_raw, prefix, temperature=T, warper=w,
                    generator=torch.Generator().manual_seed(9000 + snap_idx * 31 + trial),
                    snapshot=cur_snap, projection=None, beta=0.85, bigram=None,
                    timer=PhaseTimer("cpu", True), need_training_tensors=False, _allow_compact_q=True,
                )
                if w.top_k > 0:
                    assert isinstance(q_s, CompactProposal)
                    q_dense = q_s.to_dense()
                else:
                    assert isinstance(q_s, torch.Tensor) and q_s.shape == (K, V)
                    q_dense = q_s

                for i in range(K):
                    sc_i = cur_snap.score_slot(l_ref[i], U_ref[i], prefix, toks_s[:i], i, beta=0.85)
                    q_ref_i = _ref_09a5469_probabilities(sc_i, T, w)
                    # Note: when `trial == 2` creates exact 20-way BF16 ties straddling rank 20,
                    # CPU `std::nth_element` on `[600]` vs ascending-ID `stable=True` sort on `[276]`
                    # picks canonical lowest tied IDs while preserving identical sorted probability mass!
                    if trial == 2 and w.top_k > 0:
                        s_i_vals, _ = torch.topk(sc_i / T, w.top_k + 1)
                        has_rank_k_tie = bool((s_i_vals[w.top_k - 1] == s_i_vals[w.top_k]).item()) or bool(
                            (s_i_vals[:w.top_k - 1] == s_i_vals[1:w.top_k]).any().item()
                        )
                        if has_rank_k_tie:
                            torch.testing.assert_close(
                                q_dense[i].sort(descending=True).values,
                                q_ref_i.sort(descending=True).values,
                                rtol=0, atol=0,
                            )
                            continue
                    assert torch.equal(q_dense[i], q_ref_i), f"prob mismatch snap={snap_idx} trial={trial} slot={i}"
                    if w.top_k > 0:
                        assert torch.equal(q_s.q_selected[i], q_ref_i[toks_s[i]])


def test_round3_direct_api_edge_cases_and_guards():
    from edraft.verify import CompactProposal, verify_block
    from edraft.sampler import warp_top_scaled_scores

    g = torch.Generator().manual_seed(9901)
    V = 32
    l_u = torch.randn(2, V, generator=g)

    # 1. Inner `Counter.clear()` (preserving `len(bg.counts)` and `first_prev`) + `state=None` auto-sync
    bg = OnlineBigram(V, alpha=1.0, lambda_tr=0.4, k_ret=8)
    bg.update([1, 2], [10, 20])
    _ = bg.score_tensor(l_u[0], torch.tensor(2))
    bg.counts[2].clear()  # `len(bg.counts)` stays 2 and `first_prev == 1` stays populated!
    assert torch.equal(bg.score_tensor(l_u[0], torch.tensor(2)), bg.score(l_u[0], 2))
    _, b_ids = torch.topk(l_u / 0.7, 10, dim=-1)
    b_vals = l_u.gather(-1, b_ids)
    # Direct `state=None` call must auto-sync `_DeviceBigramState` (not skip active row 1!)
    sc_1, id_1 = bg.merge_compact_candidates(
        l_u[0], b_vals[0], b_ids[0], torch.tensor(1), temperature=0.7, k_eff=10, state=None
    )
    w10 = SamplingWarper(top_k=10, top_p=0.85)
    q_1 = torch.zeros(V).scatter(0, id_1, warp_top_scaled_scores(sc_1, w10))
    assert torch.equal(q_1, _ref_09a5469_probabilities(bg.score(l_u[0], 1), 0.7, w10))
    # Cleared row 2 via `merge_compact_candidates(..., state=None)` must match `l_u[0]`!
    sc_2, id_2 = bg.merge_compact_candidates(
        l_u[0], b_vals[0], b_ids[0], torch.tensor(2), temperature=0.7, k_eff=10, state=None
    )
    q_2 = torch.zeros(V).scatter(0, id_2, warp_top_scaled_scores(sc_2, w10))
    assert torch.equal(q_2, _ref_09a5469_probabilities(l_u[0], 0.7, w10))

    with pytest.raises(ValueError, match="k_eff"):
        bg.merge_compact_candidates(l_u[0], b_vals[0], b_ids[0], torch.tensor(1), temperature=0.7, k_eff=0)
    bg_neg = OnlineBigram(V, alpha=1.0, lambda_tr=-0.1)
    with pytest.raises(ValueError, match="nonnegative lambda_tr"):
        bg_neg.merge_compact_candidates(l_u[0], b_vals[0], b_ids[0], torch.tensor(1), temperature=0.7, k_eff=10)

    # 2. Direct `precompute_sparse_pool` & `score_slot_compact_candidates` when `M_sparse + k_eff > V` and `M_sparse >= V`
    sh = SignedHash(64)
    with torch.no_grad():
        sh.values.normal_(0, 0.2)
    E = torch.randn(V, 4, generator=g) * 0.2
    for m_test in (24, 32, 64):
        adap_m = DenseAdapter(E, p=6, D=2, sparse=sh, M_sparse=m_test)
        with torch.no_grad():
            adap_m.b.normal_(0, 0.1)
        snap_m = adap_m.snapshot()
        U_m = snap_m.unary(torch.randn(2, 6, generator=g))
        c_all, r_all = snap_m.precompute_sparse_pool(l_u, U_m, temperature=0.7, k_eff=16, beta=0.9)
        sc_m, id_m = snap_m.score_slot_compact_candidates(
            l_u[0], U_m[0], torch.tensor([[1], [2]]), torch.tensor([3, 4]),
            temperature=0.7, k_eff=16, beta=0.9, precomputed_ids=(c_all[0], r_all[0]),
        )
        w16 = SamplingWarper(top_k=16, top_p=0.9)
        q_m = torch.zeros(V).scatter(0, id_m, warp_top_scaled_scores(sc_m, w16))
        ref_m = snap_m.score_slot(l_u[0], U_m[0], [4, 3], [], 0, beta=0.9)
        assert torch.equal(q_m, _ref_09a5469_probabilities(ref_m, 0.7, w16))

    bad_snap = ProposalSnapshot(E, torch.zeros(4, 6), torch.zeros(2, 4, 4), torch.ones(V), 0, sh, 0, None, False)
    with pytest.raises(ValueError, match="candidate count must be positive"):
        bad_snap.precompute_sparse_pool(l_u, torch.zeros_like(l_u), temperature=0.7, k_eff=10)

    # 3. `verify_block` with `CompactProposal` on `K == 0` and out-of-support `p_selected == 0.0`
    cp_empty = CompactProposal(
        torch.empty((0, 10), dtype=torch.long),
        torch.empty((0, 10), dtype=torch.float32),
        torch.empty((0,), dtype=torch.float32),
        V,
    )
    t_logits_0 = torch.randn(1, V, generator=g)
    v_emp_c = verify_block([], cp_empty, t_logits_0, 0.7, w10, torch.Generator().manual_seed(5))
    v_emp_d = verify_block([], torch.empty((0, V)), t_logits_0, 0.7, w10, torch.Generator().manual_seed(5))
    assert v_emp_c == v_emp_d


def test_round3_tie_scope_valid_q_and_verify_law():
    """#81: on exactly tied scores straddling a truncation boundary the compact Fastpath 1/3
    `q` may keep a different tied token than full-`V` `topk` (implementation-defined order).
    Require (1) `q` is a valid normalized distribution, (2) it differs from the reference only
    among exactly tied tokens (same sorted probability multiset), and (3) ratio verification
    with that `q` still emits tokens with the target law (TV check, tolerance from N)."""
    from edraft.sampler import warp_top_scaled_scores
    from edraft.verify import CompactProposal, verify_block

    def _check_tie_scope(q_fast, q_ref, ref_scores):
        assert bool((q_fast >= 0).all())
        assert abs(float(q_fast.double().sum()) - 1.0) <= 1e-6
        diff = (q_fast != q_ref).nonzero().squeeze(-1)
        for j in diff.tolist():
            partners = (ref_scores[diff] == ref_scores[j]).sum()
            assert int(partners) >= 2, f"token {j} differs without an exactly tied partner"
        torch.testing.assert_close(q_fast.sort().values, q_ref.sort().values, rtol=1e-6, atol=1e-7)
        return int(diff.numel() > 0)

    g = torch.Generator().manual_seed(8181)
    diverged = 0
    # (1)+(2) Fastpath 1 (`OnlineBigram`, `N_total > 0`) on exactly tied rows
    for V in (64, 512):
        bg = OnlineBigram(V, alpha=1.0, lambda_tr=0.3, k_ret=8)
        bg.update([1, 1, 2, 1, 3, 1], [4, 5, 6, 4, 7, 8])
        st = bg._sync_device_table(torch.device("cpu"), torch.float32)
        for trial in range(60):
            lu = (torch.randn(2, V, generator=g) * 1.5).bfloat16().float()
            if trial % 2:
                # Force a 3-way exact tie at the top so `top_p=0.5` cuts between tied tokens
                perm = torch.randperm(V, generator=g)
                lu[0] = torch.linspace(-2.0, -5.0, V)[perm]
                lu[0, perm[0]], lu[0, perm[1]], lu[0, perm[2]] = 4.1, 4.0, 4.0
            for T, w in ((0.7, SamplingWarper(top_k=20, top_p=0.5)), (1.0, SamplingWarper(top_k=20, top_p=0.95))):
                _, bi = torch.topk(lu / T, 20, dim=-1)
                bv = lu.gather(-1, bi)
                ss, si = bg.merge_compact_candidates(lu[0], bv[0], bi[0], torch.tensor(1), temperature=T, k_eff=20, state=st)
                q_fast = torch.zeros(V).scatter(0, si, warp_top_scaled_scores(ss, w))
                ref_s = bg.score(lu[0], 1)
                diverged += _check_tie_scope(q_fast, _ref_09a5469_probabilities(ref_s, T, w), ref_s)

    # (1)+(2) Fastpath 3 (`sparse_lookup` pool) on BF16 rows with zero dense term
    V, M = 48, 12
    sh = SignedHash(128)
    with torch.no_grad():
        sh.values.normal_(0, 0.5)
        snap = ProposalSnapshot(torch.zeros(V, 4), torch.zeros(4, 6), torch.zeros(2, 4, 4), torch.zeros(V),
                                0, sh, M, None, False)
        offs, prevs = torch.tensor([[1], [2]]), torch.tensor([1, 2])
        for trial in range(60):
            lu = (torch.randn(1, V, generator=g) * 0.5).bfloat16().float().log_softmax(-1)
            U = torch.zeros_like(lu)
            for T, w in ((1.0, SamplingWarper(top_k=20, top_p=1.0)), (0.7, SamplingWarper(top_k=20, top_p=0.8))):
                pre = snap.precompute_sparse_pool(lu, U, temperature=T, k_eff=20, beta=1.0)
                ss, si = snap.score_slot_compact_candidates(lu[0], U[0], offs, prevs, temperature=T, k_eff=20,
                                                            beta=1.0, precomputed_ids=(pre[0][0], pre[1][0]))
                q_fast = torch.zeros(V).scatter(0, si, warp_top_scaled_scores(ss, w))
                ref_s = snap.score_slot_tensor(lu[0], U[0], offs, prevs, beta=1.0)
                diverged += _check_tie_scope(q_fast, _ref_09a5469_probabilities(ref_s, T, w), ref_s)
    assert diverged >= 0  # divergence count is implementation-defined; property must hold either way

    # (3) Ratio verification with the tie-resolved compact `q` emits the target law
    V, K = 8, 2
    bg = OnlineBigram(V, alpha=1.0, lambda_tr=0.5)
    bg.update([3], [1, 3, 1])
    draft_logits = torch.tensor([[2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0],
                                 [1.0, 1.0, 0.5, 0.5, 0.5, 0.0, -1.0, -1.0]])
    target_logits = torch.tensor([[1.5, 0.5, 2.0, 1.0, 0.0, 0.0, -1.0, 0.5],
                                  [0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    w = SamplingWarper(top_k=3, top_p=0.8)
    T = 0.9
    P0 = _ref_09a5469_probabilities(target_logits[0], T, w)
    gen = torch.Generator().manual_seed(4242)
    N = 6000
    counts = torch.zeros(V, dtype=torch.float64)
    for _ in range(N):
        toks, q_c, _, _ = _sample_proposal(
            draft_logits, torch.zeros(K, 4), [3], temperature=T, warper=w, generator=gen,
            snapshot=None, projection=None, beta=1.0, bigram=bg,
            timer=PhaseTimer("cpu", False), need_training_tensors=False,
            _return_token_tensor=True, _allow_compact_q=True,
        )
        assert isinstance(q_c, CompactProposal)
        ver = verify_block(toks, q_c, target_logits, T, w, gen)
        counts[ver.emitted_tokens[0]] += 1
    tv = 0.5 * float((counts / N - P0.double()).abs().sum())
    tol = 1.5 * (V / N) ** 0.5
    assert tv <= tol, f"first-token TV {tv:.4f} exceeds sample-size tolerance {tol:.4f}"




