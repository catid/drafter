"""Comprehensive Round-4 optimization parity, zero-sync, CoW undo, and benchmark tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import gc
import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch

from edraft.adapter import (
    DenseAdapter,
    ProposalSnapshot,
    analytic_full_kl_and_correction_grad,
    block_unary,
    full_kl,
    slot_correction,
)
from edraft.dflash_loop import PhaseTimer, _sample_proposal, generate_toy
from edraft.fastpath import FastPathConfig, FastPathRuntime
from edraft.online import OnlineState
from edraft.runtime import ROOT
from edraft.sampler import SamplingWarper, probabilities, sample_token
from edraft.sparse import (
    ExactObservedPairs,
    FrozenHash,
    LazyHashSnapshot,
    LazySignedHash,
    SignedHash,
    SumSparse,
    hash_keys,
    proximal_l1,
)
from edraft.toy import ToyParallelDrafter, ToyTarget


def _load_online_validator():
    spec = importlib.util.spec_from_file_location(
        "m4_online_checks_r4",
        ROOT / "artifacts/m4_evidence_validator/online_checks.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Opt 1: O(1) Direct-Read LazySignedHash + CoW Undo Chain + Untouched L1 Decay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [53, 257, 4096, 8192])
@pytest.mark.parametrize("scheduled", [False, True])
def test_lazy_signed_hash_cow_snapshots_55_interleaved_steps_and_untouched_l1_decay(
    capacity: int, scheduled: bool
):
    gen = torch.Generator().manual_seed(9001 + capacity + int(scheduled))
    lazy = LazySignedHash(capacity)
    eager = torch.zeros(capacity, dtype=torch.float32)

    # Probe indices covering full table (when small) or dense random subset + boundaries (when 8192)
    if capacity <= 4096:
        probe_idx = torch.arange(capacity, dtype=torch.int64)
    else:
        probe_idx = torch.unique(
            torch.cat([
                torch.tensor([0, 1, 1023, 1024, 1025, 2047, 2048, capacity - 2, capacity - 1]),
                torch.randint(capacity, (1500,), generator=gen),
            ]),
            sorted=True,
        )

    held_snapshots: list[tuple[int, LazyHashSnapshot, torch.Tensor, int]] = []

    # Snapshot at u=0 before any step
    s0 = lazy.snapshot()
    assert torch.equal(s0.read(probe_idx), torch.zeros_like(probe_idx, dtype=torch.float32))
    held_snapshots.append((0, s0, s0.read(probe_idx).clone(), 0))

    for step_idx in range(1, 56):
        lr = (0.25 / (1.0 + 0.05 * step_idx)) if scheduled else 0.18
        # Periodically use aggressive L1 with empty or tiny dirty set so UNTOUCHED buckets decay to exact 0!
        if step_idx in (12, 25, 38, 50):
            lam_l1 = 0.35
        elif step_idx % 7 == 0:
            lam_l1 = 0.0
        elif step_idx % 11 == 0:
            lr = 0.0
            lam_l1 = 0.2
        else:
            lam_l1 = 0.04

        mode = step_idx % 4
        if mode == 0:
            # Empty dirty step with positive L1 decay!
            idx = torch.empty(0, dtype=torch.int64)
            grad = torch.empty(0, dtype=torch.float32)
            lazy.step(idx, grad, lr, lam_l1)
            full_grad = torch.zeros(capacity, dtype=torch.float32)
            dirty_unique_count = 0
        elif mode == 1:
            # Unsorted step_unique!
            k_u = min(capacity, 29)
            perm = torch.randperm(capacity, generator=gen)[:k_u]
            grad = torch.randn(k_u, generator=gen) * 1.4
            lazy.step_unique(perm, grad, lr, lam_l1)
            full_grad = torch.zeros(capacity, dtype=torch.float32).scatter_add_(0, perm, grad)
            dirty_unique_count = k_u
        else:
            # Heavy duplicate indices in step()!
            hot_pool = torch.randint(capacity, (12,), generator=gen)
            picks = torch.randint(len(hot_pool), (48,), generator=gen)
            idx = hot_pool[picks]
            grad = torch.randn(48, generator=gen) * 0.9
            lazy.step(idx, grad, lr, lam_l1)
            full_grad = torch.zeros(capacity, dtype=torch.float32).scatter_add_(0, idx, grad)
            dirty_unique_count = int(torch.unique(idx).numel())

        eager = proximal_l1(eager - lr * full_grad, lr * lam_l1)

        # Verify live read matches eager within multi-step FP32 threshold accumulation tolerance
        live_probe = lazy.read(probe_idx)
        torch.testing.assert_close(live_probe, eager[probe_idx], atol=5e-7, rtol=1e-6)

        # Verify active bucket list & active count match exact live nonzero set!
        live_all = lazy.read(torch.arange(capacity, dtype=torch.int64))
        expected_active = torch.nonzero(live_all != 0, as_tuple=False).squeeze(-1)
        assert torch.equal(lazy.active, expected_active), f"active mismatch at step {step_idx}"
        assert int(lazy._active_count.item()) == int(expected_active.numel())
        assert lazy.last_update_bytes == 16 * dirty_unique_count + 12 * int(expected_active.numel())

        # Capture new head snapshot and verify it is on the O(1) direct-read head!
        snap = lazy.snapshot()
        assert snap._undo_node is not None and snap._undo_node.next_node is None
        snap_probe = snap.read(probe_idx)
        # Head snapshot read MUST be 0-ULP identical to lazy.read!
        assert torch.equal(snap_probe, live_probe)
        held_snapshots.append((step_idx, snap, snap_probe.clone(), int(expected_active.numel())))

    # Verify EVERY historical snapshot held across all 55 interleaved steps is 0-ULP BIT-EXACT!
    for u_step, snap, expected_probe, exp_act_cnt in held_snapshots:
        assert torch.equal(snap.read(probe_idx), expected_probe), f"CoW drift on snapshot u={u_step}"
        # Also test shaped (2D/3D) and int32 index queries
        shaped_q = probe_idx[:12].reshape(3, 4).to(torch.int32)
        assert torch.equal(snap.read(shaped_q), expected_probe[:12].reshape(3, 4))
        # Verify compact properties & snapshot_bytes algebra
        assert snap.indices.numel() == exp_act_cnt
        assert snap.values.numel() == exp_act_cnt
        assert snap.last_u.numel() == exp_act_cnt
        assert snap.cum_thresh.numel() == u_step + 1
        assert snap.snapshot_bytes == 16 * exp_act_cnt + 4 * (u_step + 1)

    with pytest.raises(FrozenInstanceError):
        held_snapshots[-1][1].u = 999  # type: ignore[misc]


def test_lazy_signed_hash_weakref_undo_chain_gc_releases_dropped_snapshots():
    lazy = LazySignedHash(2**20)
    s0 = lazy.snapshot()
    lazy.step(torch.tensor([5, 100000, 5]), torch.tensor([1.0, -2.0, 0.5]), lr=0.1, lambda_l1=0.01)
    assert s0._undo_node.next_node is not None
    assert torch.equal(s0.read(torch.tensor([5, 100000])), torch.tensor([0.0, 0.0]))

    # New snapshot s1 is at the current head
    s1 = lazy.snapshot()
    assert s1._undo_node.next_node is None
    expected_s1 = s1.read(torch.tensor([5, 100000])).clone()

    # Drop s0 and s1, then step: _head_ref dies so zero undo nodes are kept!
    del s0
    del s1
    gc.collect()
    lazy.step(torch.tensor([5]), torch.tensor([-0.5]), lr=0.1, lambda_l1=0.01)
    s2 = lazy.snapshot()
    assert s2._undo_node.next_node is None
    assert not torch.equal(s2.read(torch.tensor([5, 100000])), expected_s1)


# ---------------------------------------------------------------------------
# 2. Opt 2: Analytic vs Reference Autograd Parity Across Full Option Matrix
# ---------------------------------------------------------------------------


def test_analytic_full_kl_and_grad_edge_cases_weights_and_inf_rows():
    gen = torch.Generator().manual_seed(777)
    K, V = 5, 37
    scores = torch.randn(K, V, generator=gen)
    target = torch.randn(K, V, generator=gen)
    # Row 2: all -inf target_logits (inactive row)
    target[2] = -torch.inf
    # Row 3: some -inf entries in both target and scores
    target[3, :10] = -torch.inf
    scores[3, :5] = -torch.inf

    for weights in (
        None,
        torch.tensor(0.4, dtype=torch.float32),
        torch.linspace(0.2, 1.3, K),
        torch.ones((2, K), dtype=torch.float32) * 0.3,
    ):
        s_ref = scores.detach().clone().requires_grad_(True)
        loss_ref = full_kl(s_ref, target, weights)
        loss_ref.backward()
        grad_corr_ref = -s_ref.grad.detach()

        loss_an, G_an = analytic_full_kl_and_correction_grad(scores, target, weights)
        torch.testing.assert_close(loss_an, loss_ref.detach(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(G_an, grad_corr_ref, atol=1e-6, rtol=1e-6)

    # Empty K=0 edge case
    loss_0, G_0 = analytic_full_kl_and_correction_grad(torch.empty((0, 8)), torch.empty((0, 8)))
    assert loss_0.item() == 0.0 and G_0.shape == (0, 8)


@pytest.mark.parametrize("optimizer", ["sgd", "atan2", "rms", "muon"])
@pytest.mark.parametrize("bias_sparse_only", [False, True])
@pytest.mark.parametrize("bf16_gemm", [False, True])
def test_analytic_online_grad_parity_across_optimizers_and_bf16(
    optimizer: str, bias_sparse_only: bool, bf16_gemm: bool
):
    gen = torch.Generator().manual_seed(4200 + hash((optimizer, bias_sparse_only, bf16_gemm)) % 997)
    V, r, p, D, K, M = 53, 8, 10, 4, 5, 9
    E = torch.randn(V, r, generator=gen) / math.sqrt(r)
    off_sparse = SignedHash(128)
    with torch.no_grad():
        off_sparse.values.copy_(torch.randn(128, generator=gen) * 0.03)
    adapter = DenseAdapter(E, p=p, D=D, sparse=off_sparse, M_sparse=M)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(r, p, generator=gen) * 0.03)
        adapter.A.copy_(torch.randn(D, r, r, generator=gen) * 0.03)
        adapter.b.copy_(torch.randn(V, generator=gen) * 0.02)

    fp_an = FastPathRuntime(FastPathConfig(bf16_gemm=bf16_gemm, analytic_online_grad=True), static_block_length=K)
    fp_ref = FastPathRuntime(FastPathConfig(bf16_gemm=bf16_gemm, analytic_online_grad=False), static_block_length=K)

    common_kw = dict(
        lr=0.015,
        update_stride=1,
        lambda_l2=0.01,
        lambda_l1_b=1e-3,
        lambda_l1_s=2e-3,
        sparse_capacity=128,
        enable_sparse=True,
        bias_sparse_only=bias_sparse_only,
        optimizer=optimizer,
    )
    st_an = OnlineState(adapter, fastpath=fp_an, **common_kw)
    st_ref = OnlineState(adapter, fastpath=fp_ref, **common_kw)
    assert st_an.analytic_online_grad is True
    assert st_ref.analytic_online_grad is False

    prefixes = [[], [2], [4, 1, 9, 3, 7]]
    tol = 1e-6

    for r_idx in range(1, 10):
        k_cur = 1 if r_idx == 1 else (K if r_idx < 9 else 3)
        pref = prefixes[(r_idx - 1) % len(prefixes)] if r_idx > 1 else []
        l_u = torch.randn(k_cur, V, generator=gen).log_softmax(-1)
        t_log = torch.randn(k_cur, V, generator=gen)
        H = torch.randn(k_cur, p, generator=gen)
        toks = [int(x) for x in torch.randint(V, (k_cur,), generator=gen).tolist()]

        g_an = st_an._compute_round_gradients(l_u, t_log, H, pref, toks)
        g_rf = st_ref._compute_round_gradients(l_u, t_log, H, pref, toks)
        torch.testing.assert_close(g_an[0], g_rf[0], atol=tol, rtol=tol)
        if bias_sparse_only:
            assert g_an[1] is None and g_rf[1] is None
            assert g_an[2] is None and g_rf[2] is None
        else:
            torch.testing.assert_close(g_an[1], g_rf[1], atol=tol, rtol=tol)
            torch.testing.assert_close(g_an[2], g_rf[2], atol=tol, rtol=tol)
        torch.testing.assert_close(g_an[3], g_rf[3], atol=tol, rtol=tol)
        assert torch.equal(g_an[4], g_rf[4])
        torch.testing.assert_close(g_an[5], g_rf[5], atol=tol, rtol=tol)

        assert st_an.update(r_idx, l_u, t_log, H, pref, toks) is True
        assert st_ref.update(r_idx, l_u, t_log, H, pref, toks) is True

        torch.testing.assert_close(st_an.dW, st_ref.dW, atol=tol, rtol=tol)
        torch.testing.assert_close(st_an.dA, st_ref.dA, atol=tol, rtol=tol)
        torch.testing.assert_close(st_an.db, st_ref.db, atol=tol, rtol=tol)
        torch.testing.assert_close(
            st_an.dS.read(torch.arange(128)),
            st_ref.dS.read(torch.arange(128)),
            atol=tol,
            rtol=tol,
        )
        assert st_an.state_bytes_estimate == st_ref.state_bytes_estimate


@pytest.mark.parametrize("sparse_backend", ["signed_feature_hash", "exact_observed_pairs"])
@pytest.mark.parametrize("accumulate_stride,buffer_reduction", [(False, "mean"), (True, "mean"), (True, "sum")])
@pytest.mark.parametrize("weighting_mode", ["exponential", "rejection_relative", "cutoff"])
def test_analytic_online_grad_parity_sparse_backends_buffered_stride_and_weightings(
    sparse_backend: str, accumulate_stride: bool, buffer_reduction: str, weighting_mode: str
):
    gen = torch.Generator().manual_seed(
        8100 + hash((sparse_backend, accumulate_stride, buffer_reduction, weighting_mode)) % 991
    )
    V, r, p, D, K, M = 41, 6, 8, 3, 4, 7
    E = torch.randn(V, r, generator=gen) / math.sqrt(r)
    if sparse_backend == "signed_feature_hash":
        off_sp = SignedHash(64)
        with torch.no_grad():
            off_sp.values.copy_(torch.randn(64, generator=gen) * 0.02)
    else:
        off_sp = ExactObservedPairs(capacity=64, vocab_size=V)
        off_sp.train()
        _ = off_sp.lookup(torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[3, 4]]))
        with torch.no_grad():
            off_sp.values.copy_(torch.randn(64, generator=gen) * 0.02)
        off_sp.eval()

    adapter = DenseAdapter(E, p=p, D=D, sparse=off_sp, M_sparse=M)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(r, p, generator=gen) * 0.02)
        adapter.A.copy_(torch.randn(D, r, r, generator=gen) * 0.02)
        adapter.b.copy_(torch.randn(V, generator=gen) * 0.01)

    stride = 3 if accumulate_stride else 2
    kw = dict(
        lr=lambda u: 0.02 / math.sqrt(u),
        update_stride=stride,
        lambda_l2=0.01,
        lambda_l1_b=5e-4,
        lambda_l1_s=5e-4,
        sparse_capacity=64,
        enable_sparse=True,
        weighting_mode=weighting_mode,
        gamma=4.0,
        sparse_backend=sparse_backend,
        accumulate_stride=accumulate_stride,
        buffer_reduction=buffer_reduction,
    )
    st_an = OnlineState(adapter, analytic_online_grad=True, **kw)
    st_ref = OnlineState(adapter, analytic_online_grad=False, **kw)

    for r_idx in range(1, 10):
        l_u = torch.randn(K, V, generator=gen).log_softmax(-1)
        t_log = torch.randn(K, V, generator=gen)
        H = torch.randn(K, p, generator=gen)
        pref = [1, 5, 2]
        toks = [int(x) for x in torch.randint(V, (K,), generator=gen).tolist()]
        first_rej = r_idx % (K + 1)

        u1 = st_an.update(r_idx, l_u, t_log, H, pref, toks, first_rejection=first_rej)
        u2 = st_ref.update(r_idx, l_u, t_log, H, pref, toks, first_rejection=first_rej)
        assert u1 == u2
        torch.testing.assert_close(st_an.dW, st_ref.dW, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(st_an.dA, st_ref.dA, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(st_an.db, st_ref.db, atol=1e-6, rtol=1e-6)
        if sparse_backend == "signed_feature_hash":
            torch.testing.assert_close(
                st_an.dS.read(torch.arange(64)),
                st_ref.dS.read(torch.arange(64)),
                atol=1e-6,
                rtol=1e-6,
            )
        else:
            probe_packed = torch.arange(0, V * V * D, 17, dtype=torch.int64)
            torch.testing.assert_close(
                st_an.dS.read(probe_packed),
                st_ref.dS.read(probe_packed),
                atol=1e-6,
                rtol=1e-6,
            )

    m_an = st_an.metrics()
    m_ref = st_ref.metrics()
    assert m_an["u"] == m_ref["u"]
    assert m_an["active_buckets"] == m_ref["active_buckets"]
    assert m_an["last_full_loss"] == pytest.approx(m_ref["last_full_loss"], rel=1e-6, abs=1e-6)


def test_analytic_online_grad_with_compile_online_and_custom_kl_loss_fn():
    from edraft.triton_ops import emulate_rowwise_kl_cpu

    gen = torch.Generator().manual_seed(5511)
    V, r, p, D, K = 37, 6, 8, 3, 4
    E = torch.randn(V, r, generator=gen) / math.sqrt(r)
    adapter = DenseAdapter(E, p=p, D=D, sparse=SignedHash(64), M_sparse=8)
    with torch.no_grad():
        adapter.W.normal_(0, 0.02)
        adapter.A.normal_(0, 0.02)

    fp_comp = FastPathRuntime(FastPathConfig(compile_online=True, analytic_online_grad=True), static_block_length=K)
    st_comp = OnlineState(adapter, lr=0.02, sparse_capacity=64, enable_sparse=True, fastpath=fp_comp)
    st_kl = OnlineState(
        adapter, lr=0.02, sparse_capacity=64, enable_sparse=True,
        kl_loss_fn=emulate_rowwise_kl_cpu, analytic_online_grad=True,
    )
    st_ref = OnlineState(adapter, lr=0.02, sparse_capacity=64, enable_sparse=True, analytic_online_grad=False)

    for r_idx in range(1, 5):
        l_u = torch.randn(K, V, generator=gen).log_softmax(-1)
        t_log = torch.randn(K, V, generator=gen)
        H = torch.randn(K, p, generator=gen)
        pref = [2, 4, 1]
        toks = [int(x) for x in torch.randint(V, (K,), generator=gen).tolist()]
        st_comp.update(r_idx, l_u, t_log, H, pref, toks)
        st_kl.update(r_idx, l_u, t_log, H, pref, toks)
        st_ref.update(r_idx, l_u, t_log, H, pref, toks)

    torch.testing.assert_close(st_comp.dW, st_ref.dW, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(st_comp.db, st_ref.db, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(st_kl.dW, st_ref.dW, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(st_kl.db, st_ref.db, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. Opt 3: Zero Host Syncs Across Multi-Round Online Arms + SumSparse & Proposal Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm_mode",
    ["offline_online_stride1", "zero_offline_online_only", "online_dbS", "online_dbS_zero_init", "zero_lr"],
)
def test_zero_host_syncs_across_multi_round_online_arms(monkeypatch, arm_mode: str):
    gen = torch.Generator().manual_seed(6060)
    V, r, p, D, K, M = 128, 8, 12, 4, 5, 16
    E = torch.randn(V, r, generator=gen) / math.sqrt(r)
    off_sparse = SignedHash(256)
    adapter = DenseAdapter(E, p=p, D=D, sparse=off_sparse, M_sparse=M)
    if arm_mode in ("offline_online_stride1", "online_dbS"):
        with torch.no_grad():
            adapter.W.copy_(torch.randn(r, p, generator=gen) * 0.03)
            adapter.A.copy_(torch.randn(D, r, r, generator=gen) * 0.03)
            adapter.b.copy_(torch.randn(V, generator=gen) * 0.02)
            off_sparse.values.copy_(torch.randn(256, generator=gen) * 0.02)
    base_snap = adapter.snapshot()

    lr_val = 0.0 if arm_mode == "zero_lr" else 0.01
    bias_only = arm_mode in ("online_dbS", "online_dbS_zero_init")
    state = OnlineState(
        base_snap,
        lr=lr_val,
        update_stride=1,
        lambda_l2=0.01,
        lambda_l1_b=1e-4,
        lambda_l1_s=1e-4,
        sparse_capacity=256,
        enable_sparse=True,
        bias_sparse_only=bias_only,
    )

    if arm_mode in ("zero_offline_online_only", "online_dbS_zero_init", "zero_lr"):
        assert state.snapshot().is_zero is True
        assert state.snapshot().pair_is_zero() is True

    # Pre-generate round inputs before installing the strict host-sync trap
    rounds_data = []
    for r_idx in range(1, 10):
        d_logits = torch.randn(K, V, generator=gen)
        t_logits = torch.randn(K, V, generator=gen)
        h_raw = torch.randn(K, p, generator=gen)
        pref = [] if r_idx == 1 else [3, 7, 11, 19]
        toks = [int(x) for x in torch.randint(V, (K,), generator=gen).tolist()]
        rounds_data.append((r_idx, d_logits, t_logits, h_raw, pref, toks))

    rng = torch.Generator().manual_seed(999)
    warper = SamplingWarper(top_k=12, top_p=0.85)
    timer = PhaseTimer("cpu", False)

    def _forbid_sync(op_name: str):
        def _fail(*args, **kwargs):
            raise AssertionError(f"Forbidden host sync operation called during online round: {op_name}")
        return _fail

    # Install strict trap on ALL PyTorch host-synchronizing tensor conversions & dynamic-shape ops!
    monkeypatch.setattr(torch.Tensor, "item", _forbid_sync("Tensor.item"))
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid_sync("Tensor.tolist"))
    monkeypatch.setattr(torch.Tensor, "__bool__", _forbid_sync("Tensor.__bool__"))
    monkeypatch.setattr(torch.Tensor, "__int__", _forbid_sync("Tensor.__int__"))
    monkeypatch.setattr(torch.Tensor, "__float__", _forbid_sync("Tensor.__float__"))
    monkeypatch.setattr(torch, "nonzero", _forbid_sync("torch.nonzero"))
    monkeypatch.setattr(torch, "unique", _forbid_sync("torch.unique"))

    for r_idx, d_logits, t_logits, h_raw, pref, toks in rounds_data:
        snap = state.snapshot()
        _ = snap.pair_is_zero()
        tok_t, q_c, l_u, H_proj = _sample_proposal(
            d_logits, h_raw, pref,
            temperature=0.7, warper=warper, generator=rng,
            snapshot=snap, projection=None, beta=1.0, bigram=None,
            timer=timer, need_training_tensors=True,
            _return_token_tensor=True, _allow_compact_q=True,
        )
        assert isinstance(tok_t, torch.Tensor) and tok_t.shape == (K,)
        assert state.update(r_idx, l_u, t_logits, H_proj, pref, toks) is True

    monkeypatch.undo()

    # Verify post-run zero/pair_zero semantics & validator compatibility!
    if arm_mode == "zero_lr":
        assert state.snapshot().is_zero is True
        assert state.snapshot().pair_is_zero() is True
    elif arm_mode == "online_dbS_zero_init":
        assert state.snapshot().is_zero is False
        assert state.snapshot().pair_is_zero() is True
    elif arm_mode == "zero_offline_online_only":
        assert state.snapshot().is_zero is False
        assert state.snapshot().pair_is_zero() is False

    m = state.metrics()
    assert m["u"] == 9
    assert m["adapter_version"] == 9


def test_hash_keys_sum_sparse_and_cached_pair_loop_0_ulp_parity():
    gen = torch.Generator().manual_seed(31415)
    prime = 2147483647

    # 1. Verify hash_keys 0-ULP parity against explicit arithmetic formula across power-of-2 & prime capacities
    for cap in (1, 2, 17, 64, 257, 2**20, 2**24):
        d = torch.tensor([[1], [2], [3], [4]], dtype=torch.int64)
        prev = torch.tensor([[0], [19], [151935], [42]], dtype=torch.int64)
        curr = torch.randint(0, 151936, (1, 256), generator=gen, dtype=torch.int64)
        b_fast, s_fast = hash_keys(d, prev, curr, cap)

        mixed = ((d * 73856093 + prev * 19349663) + (curr * 83492791 + 104729)) % prime
        b_ref = ((mixed * 48271 + 1) % prime) % cap
        sh = ((d * 15485863 + prev * 32452843) + (curr * 49979687 + 67867967)) % prime
        s_ref = (1 - 2 * (((sh * 40699 + 17) % prime) % 2)).to(torch.float32)
        assert torch.equal(b_fast, b_ref)
        assert torch.equal(s_fast, s_ref)

    # 2. Verify SumSparse shared-hash 0-ULP parity (including custom lookup_fn fallback)
    off = SignedHash(2**12)
    with torch.no_grad():
        off.values.copy_(torch.randn(2**12, generator=gen) * 0.2)
    f_off = off.snapshot()
    lazy = LazySignedHash(2**10)
    lazy.step(torch.randint(2**10, (300,), generator=gen), torch.randn(300, generator=gen), lr=0.3, lambda_l1=0.01)
    s_lazy = lazy.snapshot()

    comb = SumSparse((f_off, s_lazy))
    d_q = torch.arange(1, 5, dtype=torch.int64)[:, None]
    p_q = torch.tensor([7, 13, 0, 99], dtype=torch.int64)[:, None]
    c_q = torch.randint(500, (1, 64), generator=gen, dtype=torch.int64)
    assert torch.equal(comb.lookup(d_q, p_q, c_q), f_off.lookup(d_q, p_q, c_q) + s_lazy.lookup(d_q, p_q, c_q))

    called_custom = [0]
    def _custom_lookup(d_in, p_in, c_in, vals, cap_in):
        called_custom[0] += 1
        b_i, s_i = hash_keys(d_in, p_in, c_in, cap_in, vals.device)
        return vals[b_i] * s_i

    f_custom = FrozenHash(f_off.values, f_off.capacity, lookup_fn=_custom_lookup)
    comb_custom = SumSparse((f_custom, s_lazy))
    assert torch.equal(comb_custom.lookup(d_q, p_q, c_q), comb.lookup(d_q, p_q, c_q))
    assert called_custom[0] == 1

    # 3. Verify dflash_loop _sample_proposal cached rev_tok_buf/rev_E_buf 0-ULP parity against per-slot score_slot oracle
    V, r, p, D, K, M = 300, 8, 10, 4, 6, 32
    E = torch.randn(V, r, generator=gen) * 0.2
    adap = DenseAdapter(E, p=p, D=D, sparse=off, M_sparse=M)
    with torch.no_grad():
        adap.W.normal_(0, 0.1)
        adap.A.normal_(0, 0.1)
        adap.b.normal_(0, 0.05)
    ostate = OnlineState(adap, lr=0.05, lambda_l1_s=1e-3, sparse_capacity=2**10, enable_sparse=True)
    ostate.update(
        1,
        torch.randn(K, V, generator=gen).log_softmax(-1),
        torch.randn(K, V, generator=gen),
        torch.randn(K, p, generator=gen),
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9, 10],
    )
    snap = ostate.snapshot()

    for prefix in ([], [9], [4, 8], [11, 22, 33, 44, 55]):
        for temp, warp in (
            (0.0, SamplingWarper()),
            (0.7, SamplingWarper(top_k=20, top_p=0.8)),
            (1.0, SamplingWarper(top_k=0, top_p=1.0)),
        ):
            raw_logits = torch.randn(K, V, generator=gen) * 2.0
            h_raw = torch.randn(K, p, generator=gen)
            toks_fast, q_fast, l_u, H_p = _sample_proposal(
                raw_logits, h_raw, prefix,
                temperature=temp, warper=warp,
                generator=torch.Generator().manual_seed(777),
                snapshot=snap, projection=None, beta=0.9, bigram=None,
                timer=PhaseTimer("cpu", True), need_training_tensors=True,
                _allow_compact_q=True, enable_round3_compact=True,
            )

            # Independent sequential per-slot oracle:
            # - At T=0 and T=1.0 (top_k=0 full-vocab), sampled tokens AND distributions match 0-ULP!
            # - At T>0 with top_k=20, compact sampling draws from the 20-element top-k vector rather than
            #   the [V]-element scattered vector while inducing the 0-ULP identical conditional q[i] at
            #   every realized prefix toks_fast[:i], AND matches uncached score_slot_compact_candidates token-for-token!
            U_ref = block_unary(snap.E, snap.W, snap.b, H_p)
            if temp == 0.0 or warp.top_k == 0:
                rng_ref = torch.Generator().manual_seed(777)
                ref_toks, ref_qs = [], []
                for i in range(K):
                    sc_i = snap.score_slot(l_u[i], U_ref[i], prefix, ref_toks, i, beta=0.9)
                    t_i, q_i = sample_token(sc_i, temp, warp, generator=rng_ref)
                    ref_toks.append(t_i)
                    if q_i is not None:
                        ref_qs.append(q_i)
                assert toks_fast == ref_toks
                if temp == 0.0:
                    assert q_fast is None
                else:
                    q_dense = q_fast.to_dense() if hasattr(q_fast, "to_dense") else q_fast
                    assert torch.equal(q_dense, torch.stack(ref_qs))
            else:
                from edraft.sampler import sample_from_top_scaled, sample_token_compact_with_prob

                q_dense = q_fast.to_dense() if hasattr(q_fast, "to_dense") else q_fast
                rng_uncached = torch.Generator().manual_seed(777)
                uncached_toks = []
                offs_all = torch.arange(1, D + 1, dtype=torch.long)
                for i in range(K):
                    sc_i = snap.score_slot(l_u[i], U_ref[i], prefix, toks_fast[:i], i, beta=0.9)
                    q_oracle_i = probabilities(sc_i, temp, warp)
                    assert torch.equal(q_dense[i], q_oracle_i)

                    # Also verify uncached score_slot_compact_candidates produces identical token & RNG stream!
                    hist = list(prefix[-D:]) + uncached_toks
                    avail = min(D, len(hist))
                    prev_ids = torch.tensor(hist[-1:-avail - 1:-1], dtype=torch.long) if avail else torch.empty(0, dtype=torch.long)
                    sel_sc, sel_ids = snap.score_slot_compact_candidates(
                        l_u[i], U_ref[i], offs_all[:avail, None], prev_ids,
                        temperature=temp, k_eff=warp.top_k, beta=0.9, contiguous_offsets=True,
                    )
                    if sel_sc is None:
                        t_u, _, _, _ = sample_token_compact_with_prob(sel_ids, temp, warp, rng_uncached)
                    else:
                        t_u, _, _, _ = sample_from_top_scaled(sel_sc, sel_ids, warp, rng_uncached)
                    uncached_toks.append(int(t_u.item()))
                assert toks_fast == uncached_toks


# ---------------------------------------------------------------------------
# 4. Opt 4: Validator Compatibility + post_m4_suite Round-4 Microbenchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm",
    [
        "offline_online_stride1",
        "online_dbS",
        "offline_online_stride5",
        "zero_offline_online_only",
        "online_rejection_relative",
        "online_cutoff",
    ],
)
def test_round4_online_rows_pass_m4_evidence_validator(arm: str):
    validator = _load_online_validator()
    target = ToyTarget(vocab_size=48, hidden_size=12, seed=21)
    drafter = ToyParallelDrafter(target, seed=22)
    E = target.lm_head.weight[:, :6].detach().clone()
    off_sp = SignedHash(257)
    adap = DenseAdapter(E, p=12, D=3, sparse=off_sp, M_sparse=8)
    if arm != "zero_offline_online_only":
        with torch.no_grad():
            adap.W.normal_(0, 0.05)
            adap.A.normal_(0, 0.05)
            adap.b.normal_(0, 0.02)
            off_sp.values.normal_(0, 0.05)

    stride = 5 if arm == "offline_online_stride5" else 1
    w_mode = (
        "rejection_relative" if arm == "online_rejection_relative"
        else ("cutoff" if arm == "online_cutoff" else "exponential")
    )
    setting = {
        "lr": 0.02,
        "update_stride": stride,
        "lambda_l2": 0.01,
        "lambda_l1_b": 1e-4,
        "lambda_l1_s": 1e-4,
        "weighting_mode": w_mode,
    }
    state = OnlineState(
        adap,
        sparse_capacity=257,
        enable_sparse=True,
        bias_sparse_only=(arm == "online_dbS"),
        **setting,
    )
    res = generate_toy(
        target, drafter, [1, 2, 3, 4], 24, block_length=4,
        temperature=0.7, warper=SamplingWarper(top_k=10, top_p=0.8),
        generator=torch.Generator().manual_seed(303),
        online_state=state, measure_phases=True,
    )
    m = state.metrics()
    upd_ms = res.phase_ms.get("online_update", 0.0)
    bw = m["state_bytes_estimate_total"] / (upd_ms * 1e6) if upd_ms > 0 else None
    row = {
        "scale": "smoke",
        "stage": "m4",
        "arm": arm,
        "phase_profile": True,
        "settings": setting,
        "blocks": res.blocks,
        "online": m,
        "online_invariants": {
            "stride": stride,
            "executed_updates": state.u,
            "request_reset": True,
            "version_only_between_rounds": True,
            "single_round_no_buffering": True,
        },
        "parameter_norms": {
            "W": float(state.offline.W.norm()),
            "A": float(state.offline.A.norm()),
            "b": float(state.offline.b.norm()),
        },
        "online_bandwidth_definition": "state_bytes_estimate_total / (online_update_ms * 1e6)",
        "estimated_online_state_bandwidth_GBps": bw,
        "phase_ms": res.phase_ms,
    }
    checked = validator.check_online_row(
        row, expected_scale="smoke", expected_arm=arm, expected_setting=setting, profile=True
    )
    assert checked["failures"] == [], f"Validator failures for {arm}: {checked['failures']}"


def test_post_m4_suite_fastpath_bench_includes_round4_online_microbenchmark(tmp_path):
    from edraft import post_m4_suite

    args = SimpleNamespace(
        scale="smoke",
        seed=17,
        bench_warmups=1,
        bench_repeats=2,
        bigram_count_gate_kappa=12.0,
    )
    r4 = post_m4_suite._measure_round4_online_microbench(args, torch.device("cpu"))
    assert r4["dimensions"]["seeded_active_buckets"] == 180
    assert r4["trajectory_round4"]["steps"] == 64
    assert r4["trajectory_round4"]["total_wall_ms"] > 0.0
    assert r4["speedups"]["online_update_round4_over_pre_round4"] > 0.0
    assert r4["speedups"]["proposal_sum_sparse_round4_over_pre_round4"] > 0.0
    assert r4["speedups"]["sum_sparse_15slot_round4_over_pre_round4"] > 0.0
    assert r4["speedups"]["lazy_hash_step_snapshot_round4_over_pre_round4"] > 0.0

    # End-to-end smoke fastpath_bench report wiring check
    out_dir = tmp_path / "post_m4_r4_smoke"
    summary = post_m4_suite.main([
        "--scale", "smoke",
        "--suite", "fastpath_bench",
        "--artifacts_root", str(tmp_path / "art"),
        "--output", str(out_dir),
        "--bench_warmups", "1",
        "--bench_repeats", "2",
        "--timing_warmups", "1",
        "--timing_repeats", "2",
    ])
    fp_rep = summary["suites"]["fastpath_bench"]
    assert "round4_online_microbenchmark" in fp_rep
    assert fp_rep["round4_online_microbenchmark"]["trajectory_round4"]["steps"] == 64
