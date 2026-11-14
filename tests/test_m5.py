"""Comprehensive mathematical, adversarial, and end-to-end CPU tests for all M5 extensions."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

import pytest
import torch
from torch.nn import functional as F

from edraft import eval_closed_loop, train, triton_ops
from edraft.adapter import (
    DenseAdapter,
    LookaheadConfig,
    compute_frozen_candidates,
    damped_lookahead_step,
    initialize_lookahead,
    lookahead_fields,
    two_sided_pseudolikelihood_loss,
)
from edraft.dflash_loop import generate_toy
from edraft.features import fit_hidden_projection, fit_token_features
from edraft.m5_baselines import (
    AdaptiveBlockController,
    Qwen35ServerClient,
    SharpDraftProposal,
    TTSState,
    TokenRecyclingState,
    TreeNode,
    build_tree_attention_mask_and_positions,
    compact_tree_cache,
    generate_token_recycling_toy,
    verify_token_tree,
)
from edraft.online import OnlineState, proximal_l1
from edraft.sampler import SamplingWarper, probabilities, sample_token
from edraft.sparse import (
    ExactObservedPairs,
    LazyExactObservedPairs,
    SignedHash,
    hash_keys,
    pack_exact_keys,
)
from edraft.toy import ToyParallelDrafter, ToyTarget


def _build_toy_env(seed: int = 101, m_sparse: int = 3):
    target = ToyTarget(vocab_size=8, hidden_size=12, seed=seed)
    drafter = ToyParallelDrafter(target, seed=seed + 1)
    cal = [[i % 8, (i + 2) % 8, (3 * i + 1) % 8] for i in range(16)]
    proj = fit_hidden_projection([drafter.draft(p, 4).h_raw for p in cal], 12, 6, seed=seed)
    E, _ = fit_token_features(target.embedding, 4, seed, torch.tensor(sum(cal, [])))
    sparse = SignedHash(128)
    g = torch.Generator().manual_seed(seed + 2)
    adapter = DenseAdapter(E, 6, 4, sparse=sparse, M_sparse=m_sparse)
    with torch.no_grad():
        adapter.W.copy_(0.35 * torch.randn(adapter.W.shape, generator=g))
        adapter.A.copy_(0.30 * torch.randn(adapter.A.shape, generator=g))
        adapter.b.copy_(0.20 * torch.randn(adapter.b.shape, generator=g))
        adapter.sparse.values.copy_(0.15 * torch.randn(adapter.sparse.values.shape, generator=g))
    adapter.eval()
    return target, drafter, proj, E, adapter


def _direct_target_rollouts(target: ToyTarget, prefix: list[int], length: int,
                            temperature: float, warper: SamplingWarper, n_samples: int, seed: int = 9000):
    counts_pos = torch.zeros(length, target.vocab_size, dtype=torch.float64)
    counts_bigram = torch.zeros(target.vocab_size, target.vocab_size, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_samples):
        ctx = list(prefix)
        emitted = []
        for pos in range(length):
            logits = target.verify_logits(ctx, [])[0]
            tok, _ = sample_token(logits, temperature, warper, gen)
            counts_pos[pos, tok] += 1.0
            if pos > 0:
                counts_bigram[emitted[-1], tok] += 1.0
            ctx.append(tok)
            emitted.append(tok)
    return counts_pos / n_samples, counts_bigram / max(1.0, counts_bigram.sum())


# ---------------------------------------------------------------------------
# 1. Closed-Loop Lookahead & Matched Frozen C0 Support
# ---------------------------------------------------------------------------

def test_lookahead_matched_frozen_c0_and_clamp_immutability():
    target, drafter, proj, _, adapter = _build_toy_env(seed=11, m_sparse=3)
    prefix = [1, 4, 2, 6]
    draft = drafter.draft(prefix, 4)
    l_unit = draft.logits.log_softmax(-1)
    H = proj(draft.h_raw)
    snap = adapter.snapshot()
    beta = 0.8

    res_cand = generate_toy(
        target, drafter, prefix, 4, temperature=0.7,
        warper=SamplingWarper(top_k=4, top_p=0.85),
        generator=torch.Generator().manual_seed(77),
        adapter=adapter, projection=proj, beta=beta,
        proposal_mode='dense_candidate', retain_tensors=True,
    )
    res_la_static = generate_toy(
        target, drafter, prefix, 4, temperature=0.7,
        warper=SamplingWarper(top_k=4, top_p=0.85),
        generator=torch.Generator().manual_seed(77),
        adapter=adapter, projection=proj, beta=beta,
        proposal_mode='dense_candidate',
        lookahead=LookaheadConfig(mode='static', domain='candidate_masked', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=beta),
        retain_tensors=True, measure_phases=True,
    )
    res_la_rerelax = generate_toy(
        target, drafter, prefix, 4, temperature=0.7,
        warper=SamplingWarper(top_k=4, top_p=0.85),
        generator=torch.Generator().manual_seed(77),
        adapter=adapter, projection=proj, beta=beta,
        proposal_mode='dense_candidate',
        lookahead=LookaheadConfig(mode='rerelax', domain='candidate_masked', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=beta),
        retain_tensors=True, measure_phases=True,
    )

    c0_expected = compute_frozen_candidates(snap, l_unit, snap.unary(H), beta_relax=beta)
    for j in range(4):
        assert torch.equal(res_cand.blocks[0]['candidate_ids'][j], c0_expected[j])
        assert torch.equal(res_la_static.blocks[0]['candidate_ids'][j], c0_expected[j])
        assert torch.equal(res_la_rerelax.blocks[0]['candidate_ids'][j], c0_expected[j])
        # Every proposed token under candidate_masked must belong to C0[j].
        assert int(res_cand.blocks[0]['draft_tokens'][j]) in set(c0_expected[j].tolist())
        assert int(res_la_static.blocks[0]['draft_tokens'][j]) in set(c0_expected[j].tolist())
        assert int(res_la_rerelax.blocks[0]['draft_tokens'][j]) in set(c0_expected[j].tolist())

    assert 'lookahead_iterations' in res_la_static.phase_ms
    assert 'lookahead_iterations' in res_la_rerelax.phase_ms

    # Verify C0[j] immutability across sequential clamping steps.
    state = initialize_lookahead(snap, l_unit, H, prefix, domain='candidate_masked', T_relax=1.0, beta_relax=beta)
    c0_before = [c.clone() for c in state.snapshot.C0]
    for _ in range(2):
        damped_lookahead_step(state, eta=0.5)
    warper = SamplingWarper(top_k=2, top_p=0.8)
    gen = torch.Generator().manual_seed(99)
    for i in range(4):
        tok, q_i = state.sample_slot(i, 0.7, beta_sample=beta, warper=warper, generator=gen)
        assert tok in set(c0_before[i].tolist())
        outside = [v for v in range(8) if v not in set(c0_before[i].tolist())]
        assert float(q_i[outside].abs().max()) == 0.0
        assert state.clamped[i] == tok
        assert state.support[i].tolist() == [tok]
        if i + 1 < 4:
            damped_lookahead_step(state, eta=0.5)
    for j in range(4):
        assert torch.equal(state.snapshot.C0[j], c0_before[j])


def test_lookahead_closed_loop_greedy_and_warped_stochastic_exactness():
    target, drafter, proj, _, adapter = _build_toy_env(seed=23, m_sparse=2)
    prefix = [0, 3, 5, 2]
    expected_greedy = target.generate(prefix, 20)

    configs = [
        ('dense_candidate', None),
        ('dense_candidate', LookaheadConfig(mode='static', domain='candidate_masked', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=1.0)),
        ('dense_candidate', LookaheadConfig(mode='rerelax', domain='candidate_masked', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=1.0)),
        ('dense_full', LookaheadConfig(mode='static', domain='full_vocabulary', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=1.0)),
        ('dense_full', LookaheadConfig(mode='rerelax', domain='full_vocabulary', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=1.0)),
    ]
    for mode, la_cfg in configs:
        res = generate_toy(
            target, drafter, prefix, 20, temperature=0.0,
            adapter=adapter, projection=proj, proposal_mode=mode, lookahead=la_cfg,
        )
        assert res.output_tokens == expected_greedy

    # Stochastic distribution exactness even with aggressive C0 (M_sparse=2 out of V=8!) + warper:
    # rejected tokens outside C0 must still be recovered with exact target probability via (P - q)_+!
    warper = SamplingWarper(top_k=4, top_p=0.82, min_p=0.05)
    n_samples = 1400
    gen_len = 6
    ref_pos, ref_bg = _direct_target_rollouts(target, prefix, gen_len, 0.7, warper, n_samples, seed=5000)

    for idx, (mode, la_cfg) in enumerate([
        ('dense_candidate', LookaheadConfig(mode='rerelax', domain='candidate_masked', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=0.9)),
        ('dense_full', LookaheadConfig(mode='rerelax', domain='full_vocabulary', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0, beta_relax=0.9)),
    ]):
        emp_pos = torch.zeros_like(ref_pos)
        emp_bg = torch.zeros_like(ref_bg)
        for s in range(n_samples):
            out = generate_toy(
                target, drafter, prefix, gen_len, temperature=0.7, warper=warper,
                generator=torch.Generator().manual_seed(80000 + 10000 * idx + s),
                adapter=adapter, projection=proj, beta=0.9,
                proposal_mode=mode, lookahead=la_cfg,
            ).output_tokens
            for p_i, tok in enumerate(out):
                emp_pos[p_i, tok] += 1.0
                if p_i > 0:
                    emp_bg[out[p_i - 1], tok] += 1.0
        emp_pos /= n_samples
        emp_bg /= emp_bg.sum()
        max_tv = float(0.5 * (emp_pos - ref_pos).abs().sum(dim=-1).max())
        bg_tv = float(0.5 * (emp_bg - ref_bg).abs().sum())
        assert max_tv < 0.05, f'marginal TV too large ({max_tv}) for {mode}/{la_cfg}'
        assert bg_tv < 0.06, f'bigram TV too large ({bg_tv}) for {mode}/{la_cfg}'


# ---------------------------------------------------------------------------
# 2. Two-Sided Verified-Stream Pseudolikelihood
# ---------------------------------------------------------------------------

def test_two_sided_pseudolikelihood_finite_difference_gradients():
    torch.manual_seed(31)
    V, r, p, D, L = 7, 3, 4, 2, 4
    E = torch.randn(V, r, dtype=torch.float32)
    sparse = ExactObservedPairs(128, vocab_size=V, overflow_policy='error')
    adapter = DenseAdapter(E, p, D, sparse=sparse, M_sparse=V)
    with torch.no_grad():
        adapter.W.copy_(0.25 * torch.randn(r, p))
        # Deliberately asymmetric A[d] so A[d] vs A[d].T transposition bugs are caught!
        adapter.A.copy_(0.35 * torch.randn(D, r, r))
        adapter.b.copy_(0.15 * torch.randn(V))
    adapter.train()

    l_unit = torch.randn(L, V).log_softmax(-1)
    H = torch.randn(L, p)
    prefix = [2, 5, 1]
    committed = [3, 0, 6, 4, 2]  # 4 slots + 1 future committed bonus token!

    # Warm up sparse slot admissions once so keys are registered in ExactObservedPairs, then seed nonzero values.
    _ = adapter.two_sided_loss(l_unit, H, prefix, committed, eval_slots=L)
    with torch.no_grad():
        occ = adapter.sparse.num_occupied
        assert occ > 0
        adapter.sparse.values[:occ].copy_(0.2 * torch.randn(occ))
    adapter.eval()  # freeze admission during central finite-difference perturbation

    adapter.zero_grad(set_to_none=True)
    if adapter.sparse.values.grad is not None:
        adapter.sparse.values.grad.zero_()
    loss = adapter.two_sided_loss(l_unit, H, prefix, committed, eval_slots=L)
    loss.backward()

    eps = 2e-3
    for name, param in [('W', adapter.W), ('A', adapter.A), ('b', adapter.b), ('S', adapter.sparse.values)]:
        analytic = param.grad.detach().clone()
        flat = param.detach().view(-1)
        check_count = min(len(flat), 18)
        for idx in range(check_count):
            orig = float(flat[idx].item())
            with torch.no_grad():
                flat[idx] = orig + eps
                lp = float(adapter.two_sided_loss(l_unit, H, prefix, committed, eval_slots=L).item())
                flat[idx] = orig - eps
                lm = float(adapter.two_sided_loss(l_unit, H, prefix, committed, eval_slots=L).item())
                flat[idx] = orig
            num_grad = (lp - lm) / (2.0 * eps)
            assert abs(float(analytic.view(-1)[idx].item()) - num_grad) < 1e-3, (
                f'gradient mismatch on {name}[{idx}]: analytic={analytic.view(-1)[idx].item()} vs fd={num_grad}'
            )


def test_offline_and_online_two_sided_pseudolikelihood_fit_and_resume(tmp_path):
    train.main([
        '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
        '--objective', 'two_sided_pseudolikelihood',
        '--sparse_backend', 'exact_observed_pairs',
        '--checkpoint_every', '2',
    ])
    # Primary m2_smoke must remain untouched!
    assert not (tmp_path / 'm2_smoke.json').exists()
    out_json = tmp_path / 'm5_two_sided_pseudolikelihood_exact_observed_pairs_smoke.json'
    assert out_json.is_file()
    report = json.loads(out_json.read_text())
    assert report['status'] == 'passed'
    assert report['objective'] == 'two_sided_pseudolikelihood'
    assert report['sparse_backend'] == 'exact_observed_pairs'
    sp_stats = report['stages']['dense_sparse']['sparse_collision_statistics']
    assert sp_stats['active_representation_collision_count'] == 0
    assert 'two_sided_pseudolikelihood' in report['stages']['dense_sparse']['test']

    # Reload fitted ExactObservedPairs checkpoint and run online two-sided PL adaptation.
    ckpt_path = Path(report['checkpoints']['dense_sparse'])
    loaded, meta = train.load_adapter_checkpoint(ckpt_path, device='cpu')
    assert isinstance(loaded.sparse, ExactObservedPairs)
    assert meta['adapter_config']['sparse_backend'] == 'exact_observed_pairs'

    # Verify bit-exact mid-stage interruption + resume on two_sided_pseudolikelihood + exact_observed_pairs:
    from types import SimpleNamespace
    entries, idx_rep = train.trace_index(tmp_path / 'm2_smoke_traces')
    toy_tgt = ToyTarget(vocab_size=8, hidden_size=12, seed=2026)
    toy_dft = ToyParallelDrafter(toy_tgt, seed=2027)
    head_meta = toy_dft.head_metadata()
    head_w, head_b = toy_dft.draft_lm_head.weight, toy_dft.draft_lm_head.bias
    E_ref = loaded.E.detach().clone()

    def _pl_args(out_dir):
        return SimpleNamespace(
            output=out_dir, scale='smoke', seed=2026, history_depth=4, M_sparse=4,
            sparse_capacity=256, learning_rate=0.01, weight_decay=0.0001,
            bias_weight_decay=0.0, sparse_learning_rate=0.02, lambda_l1=1e-5,
            batch_blocks=8, vocab_chunk=8, epochs_dense=1, epochs_sparse=1,
            checkpoint_every=2, resume=True,
            sparse_backend='exact_observed_pairs', sparse_overflow_policy='freeze_new',
            objective='two_sided_pseudolikelihood',
        )

    direct_rep = train.fit(entries, idx_rep, E_ref, head_w, head_b, head_meta, _pl_args(tmp_path / 'pl_direct'))
    first_epoch_steps = (sum(e.blocks for e in entries if e.split == 'train') + 7) // 8
    for stage_tag, stop_step in [('dense', 2), ('dense_sparse', first_epoch_steps + 2)]:
        res_args = _pl_args(tmp_path / f'pl_resumed_{stage_tag}')
        inter = train.fit(entries, idx_rep, E_ref, head_w, head_b, head_meta, res_args, stop_after_steps=stop_step)
        assert inter['status'] == 'interrupted_for_resume_test'
        res_rep = train.fit(entries, idx_rep, E_ref, head_w, head_b, head_meta, res_args)
        d_mod, _ = train.load_adapter_checkpoint(Path(direct_rep['checkpoints']['dense_sparse']))
        r_mod, _ = train.load_adapter_checkpoint(Path(res_rep['checkpoints']['dense_sparse']))
        for k, v in d_mod.state_dict().items():
            torch.testing.assert_close(v, r_mod.state_dict()[k], rtol=0, atol=0)
        d_st = direct_rep['stages']['dense_sparse']['sparse_collision_statistics']
        r_st = res_rep['stages']['dense_sparse']['sparse_collision_statistics']
        for metric_k in ['observed_lookup_events', 'unique_observed_keys', 'occupied_exact_entries']:
            assert d_st[metric_k] == r_st[metric_k]

    target = ToyTarget(vocab_size=8, hidden_size=12, seed=2026)
    drafter = ToyParallelDrafter(target, seed=2027)
    proj = fit_hidden_projection([drafter.draft([0, 1, 2], 4).h_raw for _ in range(8)], 12, 6, seed=2026)
    online = OnlineState(
        loaded, enable_sparse=True, sparse_capacity=64,
        sparse_backend='exact_observed_pairs',
        loss_mode='two_sided_pseudolikelihood',
        lr=5e-3,
    )
    res = generate_toy(
        target, drafter, [1, 3, 5, 2], 24, temperature=0.0,
        online_state=online, projection=proj,
    )
    assert res.output_tokens == target.generate([1, 3, 5, 2], 24)
    assert online.u == len(res.blocks)
    assert online.metrics()['loss_mode'] == 'two_sided_pseudolikelihood'
    assert online.metrics()['sparse_backend'] == 'exact_observed_pairs'


# ---------------------------------------------------------------------------
# 3. ExactObservedPairs vs SignedHash & Scheduled Lazy Exact Sparse Parity
# ---------------------------------------------------------------------------

def test_exact_observed_pairs_zero_collisions_and_overflow_policies():
    V = 32
    cap = 8
    # Find two distinct keys that collide in SignedHash(capacity=8).
    colliding = None
    seen_buckets = {}
    for p in range(V):
        for c in range(V):
            idx, _ = hash_keys(1, torch.tensor([p]), torch.tensor([c]), cap)
            b = int(idx.item())
            if b in seen_buckets and seen_buckets[b] != (p, c):
                colliding = (seen_buckets[b], (p, c))
                break
            seen_buckets[b] = (p, c)
        if colliding is not None:
            break
    assert colliding is not None
    (p1, c1), (p2, c2) = colliding

    sh = SignedHash(cap)
    ex = ExactObservedPairs(cap, vocab_size=V, overflow_policy='error')
    ex.train()
    # Register both keys in ExactObservedPairs and optimize distinct targets.
    opt_sh = torch.optim.SGD([sh.values], lr=0.5)
    opt_ex = torch.optim.SGD([ex.values], lr=0.5)
    for _ in range(10):
        opt_sh.zero_grad()
        loss_sh = (sh.lookup(1, torch.tensor([p1]), torch.tensor([c1])) - 2.0).pow(2).sum() + (
            sh.lookup(1, torch.tensor([p2]), torch.tensor([c2])) + 3.0
        ).pow(2).sum()
        loss_sh.backward()
        opt_sh.step()

        opt_ex.zero_grad()
        loss_ex = (ex.lookup(1, torch.tensor([p1]), torch.tensor([c1])) - 2.0).pow(2).sum() + (
            ex.lookup(1, torch.tensor([p2]), torch.tensor([c2])) + 3.0
        ).pow(2).sum()
        loss_ex.backward()
        opt_ex.step()

    ex.eval()
    v1 = float(ex.lookup(1, torch.tensor([p1]), torch.tensor([c1])).item())
    v2 = float(ex.lookup(1, torch.tensor([p2]), torch.tensor([c2])).item())
    assert abs(v1 - 2.0) < 1e-3 and abs(v2 - (-3.0)) < 1e-3
    # SignedHash cannot represent independent values in the collided bucket!
    sh1 = float(sh.lookup(1, torch.tensor([p1]), torch.tensor([c1])).item())
    sh2 = float(sh.lookup(1, torch.tensor([p2]), torch.tensor([c2])).item())
    assert abs(sh1) == pytest.approx(abs(sh2), abs=1e-6)

    # Test overflow_policy='error' vs 'freeze_new' + zero-slot reclamation.
    small_err = ExactObservedPairs(3, vocab_size=V, overflow_policy='error')
    small_err.train()
    _ = small_err.lookup(1, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    with pytest.raises(OverflowError, match='capacity'):
        _ = small_err.lookup(1, torch.tensor([3]), torch.tensor([3]))

    small_frz = ExactObservedPairs(3, vocab_size=V, overflow_policy='freeze_new')
    small_frz.train()
    _ = small_frz.lookup(1, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    with torch.no_grad():
        small_frz.values[:3].copy_(torch.tensor([0.5, 0.001, -0.5]))
    dropped_out = small_frz.lookup(1, torch.tensor([3, 4]), torch.tensor([3, 4]))
    assert torch.all(dropped_out == 0.0)
    assert small_frz.overflow_events == 2
    assert small_frz.dropped_unique_keys == 2
    # Proximal step shrinks slot 1 (0.001) to exact zero and reclaims its slot!
    small_frz.proximal_step(lr=1.0, lambda_l1=0.05)
    assert small_frz.num_occupied == 2
    _ = small_frz.lookup(1, torch.tensor([9]), torch.tensor([9]))
    assert small_frz.num_occupied == 3


def test_lazy_exact_observed_pairs_matches_eager_dense_table_under_scheduled_l1():
    D, V = 3, 6
    lazy = LazyExactObservedPairs(capacity=D * V * V, vocab_size=V, overflow_policy='error')
    eager = torch.zeros(D + 1, V, V, dtype=torch.float32)
    lambda_l1 = 0.03
    lr0 = 0.25
    gen = torch.Generator().manual_seed(404)

    snapshots = []
    for step in range(1, 17):
        lr_u = lr0 / (step ** 0.5)
        n_keys = int(torch.randint(1, 8, (1,), generator=gen).item())
        ds = torch.randint(1, D + 1, (n_keys,), generator=gen)
        ps = torch.randint(0, V, (n_keys,), generator=gen)
        cs = torch.randint(0, V, (n_keys,), generator=gen)
        grads = 0.4 * torch.randn(n_keys, generator=gen)
        packed = pack_exact_keys(ds, ps, cs, V)
        uniq_packed = torch.unique(packed)
        summed_grads = torch.zeros(len(uniq_packed), dtype=torch.float32)
        for k_i, u_k in enumerate(uniq_packed):
            summed_grads[k_i] = grads[packed == u_k].sum()

        # Eager reference step across ALL (d, p, c) slots in [1..D, 0..V-1, 0..V-1]!
        grad_grid = torch.zeros_like(eager)
        for u_k, g_val in zip(uniq_packed.tolist(), summed_grads.tolist()):
            d_i = u_k // (V * V)
            p_i = (u_k // V) % V
            c_i = u_k % V
            grad_grid[d_i, p_i, c_i] = g_val
        eager = proximal_l1(eager - lr_u * grad_grid, lr_u * lambda_l1)

        # Lazy exact update!
        lazy.step(uniq_packed, summed_grads, lr_u, lambda_l1)
        snap = lazy.snapshot()
        snapshots.append((snap, eager.clone()))

        # Pure non-mutating read check across ALL D * V * V keys!
        values_before_read = lazy.values.clone()
        last_u_before_read = lazy.last_u.clone()
        all_d, all_p, all_c = torch.meshgrid(
            torch.arange(1, D + 1), torch.arange(V), torch.arange(V), indexing='ij'
        )
        got = lazy.read(pack_exact_keys(all_d.reshape(-1), all_p.reshape(-1), all_c.reshape(-1), V))
        expected = eager[1:].reshape(-1)
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
        assert torch.equal(lazy.values, values_before_read)
        assert torch.equal(lazy.last_u, last_u_before_read)

    # Verify historical snapshots remained 100% immutable across all subsequent steps!
    all_d, all_p, all_c = torch.meshgrid(
        torch.arange(1, D + 1), torch.arange(V), torch.arange(V), indexing='ij'
    )
    for snap, expected_grid in snapshots:
        got_snap = snap.lookup(all_d, all_p, all_c)
        torch.testing.assert_close(got_snap, expected_grid[1:], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Multi-Round Buffered Stride Accumulation
# ---------------------------------------------------------------------------

def test_online_buffered_stride5_exact_five_round_gradient_mean():
    _, drafter, proj, _, base_adapter = _build_toy_env(seed=55, m_sparse=4)
    target = ToyTarget(vocab_size=8, hidden_size=12, seed=55)
    lr = 2e-2
    l2 = 0.05
    l1_b = 1e-4
    l1_s = 1e-4

    buffered = OnlineState(
        base_adapter, enable_sparse=True, sparse_capacity=128,
        sparse_backend='exact_observed_pairs',
        lr=lr, update_stride=5, accumulate_stride=True, buffer_reduction='mean',
        lambda_l2=l2, lambda_l1_b=l1_b, lambda_l1_s=l1_s,
    )
    lazy_single = OnlineState(
        base_adapter, enable_sparse=True, sparse_capacity=128,
        sparse_backend='exact_observed_pairs',
        lr=lr, update_stride=5, accumulate_stride=False,
        lambda_l2=l2, lambda_l1_b=l1_b, lambda_l1_s=l1_s,
    )

    snap0 = buffered.snapshot()
    w0, a0, b0 = snap0.W.clone(), snap0.A.clone(), snap0.b.clone()

    # Collect 5 distinct rounds and compute the exact manual 5-round mean gradient wrt round-0 snapshot!
    gw_sum = torch.zeros_like(buffered.dW)
    ga_sum = torch.zeros_like(buffered.dA)
    gb_sum = torch.zeros_like(buffered.db)

    for r_idx in range(1, 6):
        prefix = [r_idx % 8, (r_idx + 2) % 8, (2 * r_idx + 1) % 8]
        draft = drafter.draft(prefix, 4)
        lu = draft.logits.log_softmax(-1)
        tokens = [(r_idx + j) % 8 for j in range(4)]
        tl = target.verify_logits(prefix, tokens)[:4]
        H = proj(draft.h_raw)

        # Manual gradient on dense params wrt initial offline adapter:
        import copy
        ref_mod = copy.deepcopy(base_adapter)
        ref_mod.zero_grad(set_to_none=True)
        loss_r = ref_mod.full_loss(lu, tl, H, prefix, tokens)
        loss_r.backward()
        gw_sum += ref_mod.W.grad.detach()
        ga_sum += ref_mod.A.grad.detach()
        gb_sum += ref_mod.b.grad.detach()

        updated_buf = buffered.update(r_idx, lu, tl, H, prefix, tokens, first_rejection=2)
        updated_lazy = lazy_single.update(r_idx, lu, tl, H, prefix, tokens, first_rejection=2)
        if r_idx < 5:
            assert not updated_buf and not updated_lazy
            assert buffered.u == 0 and buffered.version == 0
            cur = buffered.snapshot()
            assert torch.equal(cur.W, w0) and torch.equal(cur.A, a0) and torch.equal(cur.b, b0)
        else:
            assert updated_buf and updated_lazy
            assert buffered.u == 1 and buffered.version == 1

    lr_1 = lr / (1.0 ** 0.5)
    shrink = 1.0 + lr_1 * l2
    expected_dW = -(lr_1 * (gw_sum / 5.0)) / shrink
    expected_dA = -(lr_1 * (ga_sum / 5.0)) / shrink
    expected_db = proximal_l1(-(lr_1 * (gb_sum / 5.0)) / shrink, lr_1 * l1_b)
    torch.testing.assert_close(buffered.dW, expected_dW, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(buffered.dA, expected_dA, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(buffered.db, expected_db, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(buffered.dW, lazy_single.dW)
    assert buffered.metrics()['buffered_rounds_accumulated_total'] == 5


# ---------------------------------------------------------------------------
# 5. Long-Condition Oracle & Pure-T=1 Calibration Ablations
# ---------------------------------------------------------------------------

def test_m5_long_oracle_and_pure_t1_ablations_isolated_from_m2(tmp_path):
    train.main([
        '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
        '--allow_long_oracle_fit',
    ])
    oracle_json = tmp_path / 'm5_long_oracle_smoke.json'
    assert oracle_json.is_file()
    assert not (tmp_path / 'm2_smoke.json').exists()
    rep_oracle = json.loads(oracle_json.read_text())
    assert rep_oracle['training_distribution'] == 'long_oracle_m5'
    assert 'M5 ablation only' in rep_oracle['ablation_warning']

    train.main([
        '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
        '--allow_pure_t1_ablation',
    ])
    t1_json = tmp_path / 'm5_pure_t1_calibration_smoke.json'
    assert t1_json.is_file()
    rep_t1 = json.loads(t1_json.read_text())
    assert rep_t1['training_distribution'] == 'pure_T1_calibration_ablation_m5'
    assert not (tmp_path / 'm2_smoke.json').exists()

    with pytest.raises(ValueError, match='Cannot combine'):
        train.main([
            '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
            '--allow_long_oracle_fit', '--allow_pure_t1_ablation',
        ])


# ---------------------------------------------------------------------------
# 6. Token Recycling Tree, TTS, SharpDraft, Adaptive Block, Server & Triton
# ---------------------------------------------------------------------------

def test_token_recycling_tree_mask_kv_compaction_and_target_exactness():
    nodes = [
        TreeNode(index=0, token_id=3, parent=-1, depth=0, children=[1, 2]),
        TreeNode(index=1, token_id=5, parent=0, depth=1, children=[3]),
        TreeNode(index=2, token_id=2, parent=0, depth=1, children=[4]),
        TreeNode(index=3, token_id=7, parent=1, depth=2, children=[]),
        TreeNode(index=4, token_id=1, parent=2, depth=2, children=[]),
    ]
    mask, pos = build_tree_attention_mask_and_positions(nodes, prefix_length=6, device='cpu', dtype=torch.float32)
    assert pos.tolist() == [[6, 7, 7, 8, 8]]
    assert mask.shape == (1, 1, 5, 11)
    # Every node attends to all 6 prefix tokens:
    assert torch.all(mask[0, 0, :, :6] == 0.0)
    # Node 3 (parent=1 -> parent=0) attends to tree indices {0, 1, 3}, NOT {2, 4}:
    assert mask[0, 0, 3, 6 + 0].item() == 0.0
    assert mask[0, 0, 3, 6 + 1].item() == 0.0
    assert mask[0, 0, 3, 6 + 3].item() == 0.0
    assert mask[0, 0, 3, 6 + 2].item() == torch.finfo(torch.float32).min
    assert mask[0, 0, 3, 6 + 4].item() == torch.finfo(torch.float32).min

    # Verify compact_tree_cache gathers the exact accepted branch [0, 2, 4] into [start..start+2].
    from transformers import DynamicCache
    cache = DynamicCache()
    k_init = torch.arange(11 * 4, dtype=torch.float32).view(1, 1, 11, 4)
    v_init = (100.0 + torch.arange(11 * 4, dtype=torch.float32)).view(1, 1, 11, 4)
    cache.update(k_init.clone(), v_init.clone(), 0)
    compact_tree_cache(cache, start=6, accepted_node_indices=[0, 2, 4])
    assert cache.get_seq_length() == 9
    layer = cache.layers[0]
    torch.testing.assert_close(layer.keys[:, :, :6, :], k_init[:, :, :6, :])
    torch.testing.assert_close(layer.keys[:, :, 6, :], k_init[:, :, 6 + 0, :])
    torch.testing.assert_close(layer.keys[:, :, 7, :], k_init[:, :, 6 + 2, :])
    torch.testing.assert_close(layer.keys[:, :, 8, :], k_init[:, :, 6 + 4, :])

    # Closed-loop greedy + multi-branch warped stochastic exactness on ToyTarget:
    target = ToyTarget(vocab_size=8, hidden_size=12, seed=71)
    drafter = ToyParallelDrafter(target, seed=72)
    prefix = [2, 6, 1, 4]
    res_greedy, state_g = generate_token_recycling_toy(
        target, drafter, prefix, 24, temperature=0.0,
        branch_widths=(3, 2, 1, 1), measure_phases=True,
    )
    assert res_greedy.output_tokens == target.generate(prefix, 24)
    assert state_g.metrics()['populated_tokens'] > 0

    warper = SamplingWarper(top_k=4, top_p=0.85)
    n_samples = 1500
    gen_len = 5
    ref_pos, ref_bg = _direct_target_rollouts(target, prefix, gen_len, 0.7, warper, n_samples, seed=6100)
    emp_pos = torch.zeros_like(ref_pos)
    emp_bg = torch.zeros_like(ref_bg)
    # Pre-seed adjacency matrix with non-trivial multi-branch transitions so sibling rejection paths fire heavily!
    seeded_adj = TokenRecyclingState(8, max_successors=4, branch_widths=(3, 2, 1))
    seeded_adj.update_from_target_logits([0, 1, 2, 3, 4, 5, 6, 7], torch.randn(8, 8, generator=torch.Generator().manual_seed(9)))
    for s in range(n_samples):
        st = TokenRecyclingState(8, max_successors=4, branch_widths=(3, 2, 1))
        st.adjacency.copy_(seeded_adj.adjacency)
        st.populated.copy_(seeded_adj.populated)
        out, _ = generate_token_recycling_toy(
            target, drafter, prefix, gen_len, temperature=0.7, warper=warper,
            generator=torch.Generator().manual_seed(120000 + s),
            recycling_state=st,
        )
        seq = out.output_tokens
        for p_i, tok in enumerate(seq):
            emp_pos[p_i, tok] += 1.0
            if p_i > 0:
                emp_bg[seq[p_i - 1], tok] += 1.0
    emp_pos /= n_samples
    emp_bg /= emp_bg.sum()
    assert float(0.5 * (emp_pos - ref_pos).abs().sum(dim=-1).max()) < 0.05
    assert float(0.5 * (emp_bg - ref_bg).abs().sum()) < 0.06


def test_tts_sharp_draft_and_adaptive_block_controller():
    target, drafter, proj, _, adapter = _build_toy_env(seed=88, m_sparse=4)
    prefix = [1, 2, 5, 0]

    # TTS reduces target KL across repeated online updates on a consistent context:
    tts = TTSState(vocab_size=8, hidden_dim=6, max_slots=4, lr=0.08, lambda_l2=0.0, lambda_l1_b=0.0)
    draft = drafter.draft(prefix, 4)
    lu = draft.logits.log_softmax(-1)
    tl = target.verify_logits(prefix, [3, 1, 4, 2])[:4]
    H = proj(draft.h_raw)
    tts.update(1, lu, tl, H, prefix, [3, 1, 4, 2])
    first_loss = tts.last_loss
    for r in range(2, 10):
        tts.update(r, lu, tl, H, prefix, [3, 1, 4, 2])
    assert tts.last_loss < first_loss

    # SharpDraft sharpens peaked distributions (gamma > 1.2) more than uniform ones (gamma == gamma_min):
    sd = SharpDraftProposal(gamma_max=1.4, gamma_min=0.95, entropy_pivot=0.4)
    peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).log_softmax(-1)
    flat = torch.zeros(1, 8).log_softmax(-1)
    _, g_peaked = sd.sharpen_scores(peaked)
    _, g_flat = sd.sharpen_scores(flat)
    assert float(g_peaked.item()) > 1.35
    assert float(g_flat.item()) == pytest.approx(0.95, abs=1e-6)

    # AdaptiveBlockController dynamically truncates low-confidence tails and preserves greedy exactness:
    ctrl = AdaptiveBlockController(min_block_length=1, max_block_length=4, policy='hybrid', min_cumulative_confidence=0.45)
    res_adaptive = generate_toy(
        target, drafter, prefix, 24, temperature=0.0,
        adapter=adapter, projection=proj, adaptive_controller=ctrl,
    )
    assert res_adaptive.output_tokens == target.generate(prefix, 24)
    metrics = ctrl.metrics()
    assert metrics['rounds'] == len(res_adaptive.blocks)
    assert 1 <= metrics['min_effective_k'] <= metrics['max_effective_k'] <= 4
    assert metrics['saved_verify_slots'] >= 0


def test_qwen35_server_client_and_resumable_server_eval(tmp_path):
    requests_seen = []

    class MockOpenAIHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length).decode('utf-8'))
            requests_seen.append(body)
            resp = {
                'id': f'cmpl-{len(requests_seen)}',
                'model': body.get('model', 'qwen3.5-mock'),
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'Answer 42', 'reasoning_content': None},
                    'finish_reason': 'stop',
                    'meta_info': {'spec_verify_ct': 8, 'spec_accept_length': 3.75},
                }],
                'usage': {'prompt_tokens': 12, 'completion_tokens': 30, 'total_tokens': 42},
            }
            encoded = json.dumps(resp).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(('127.0.0.1', 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f'http://127.0.0.1:{server.server_port}'
        client = Qwen35ServerClient(base_url, model='Qwen/Qwen3.5-4B')
        one = client.chat_completion('What is 6*7?', temperature=0.7, top_p=0.8, top_k=20, max_tokens=64, enable_thinking=True)
        assert one['completion_tokens'] == 30
        assert one['server_meta_info']['spec_accept_length_reported_including_bonus'] == pytest.approx(3.75)
        assert one['server_meta_info']['logical_mean_accepted_length'] == pytest.approx(2.75)
        assert requests_seen[-1]['chat_template_kwargs'] == {'enable_thinking': True}

        requests_seen.clear()
        eval_closed_loop.main([
            '--stage', 'm5', '--scale', 'smoke', '--mode', 'server_eval',
            '--m5_server_url', base_url, '--m5_server_model', 'Qwen/Qwen3.5-4B',
            '--output', str(tmp_path),
        ])
        first_count = len(requests_seen)
        assert first_count == 8  # 2 std * 3 conds + 2 long_thinking * 1 cond
        rep = json.loads((tmp_path / 'm5' / 'server_eval.json').read_text())
        assert rep['status'] == 'complete'
        assert rep['completed_requests'] == 8

        # Second invocation resumes from manifest with zero new HTTP calls!
        eval_closed_loop.main([
            '--stage', 'm5', '--scale', 'smoke', '--mode', 'server_eval',
            '--m5_server_url', base_url, '--m5_server_model', 'Qwen/Qwen3.5-4B',
            '--output', str(tmp_path),
        ])
        assert len(requests_seen) == first_count
    finally:
        server.shutdown()
        server.server_close()


def test_triton_ops_cpu_emulation_parity_and_runtime_gates():
    from edraft.online import LazySignedHash
    cap = 1024
    gen = torch.Generator().manual_seed(303)
    values = torch.randn(cap, generator=gen, dtype=torch.float32)
    sh = SignedHash(cap)
    with torch.no_grad():
        sh.values.copy_(values)
    prev = torch.randint(0, 500, (17, 1), generator=gen)
    cand = torch.randint(0, 500, (17, 16), generator=gen)
    for d in range(1, 5):
        ref = sh.lookup(d, prev, cand)
        emulated = triton_ops.emulate_signed_hash_kernel_cpu(d, prev, cand, values, cap)
        torch.testing.assert_close(emulated, ref, rtol=0.0, atol=0.0)

    # Also test lazy scheduled L1 register math in emulate_signed_hash_kernel_cpu vs LazySignedHash:
    lazy_sh = LazySignedHash(cap)
    idx_step = torch.tensor([3, 17, 42, 99], dtype=torch.long)
    lazy_sh.step(idx_step, torch.tensor([0.5, -0.8, 0.02, -0.3]), lr=0.2, lambda_l1=0.05)
    lazy_sh.step(torch.tensor([17, 200], dtype=torch.long), torch.tensor([-0.4, 0.6]), lr=0.15, lambda_l1=0.05)
    snap_lazy = lazy_sh.snapshot()
    dense_cum_thresh = torch.tensor(lazy_sh.cum_thresh, dtype=torch.float32)
    for d in range(1, 4):
        ref_lazy = snap_lazy.lookup(d, prev, cand)
        emu_lazy = triton_ops.emulate_signed_hash_kernel_cpu(
            d, prev, cand, lazy_sh.values, cap,
            last_u=lazy_sh.last_u, cum_thresh=dense_cum_thresh, u=lazy_sh.u,
        )
        torch.testing.assert_close(emu_lazy, ref_lazy, rtol=1e-6, atol=1e-6)

    # Parity violation must raise TritonParityError when strict=True, and reject when strict=False:
    x = torch.randn(8, 16)
    with pytest.raises(triton_ops.TritonParityError, match='failed numerical parity'):
        triton_ops.evaluate_kernel_gate(
            'bad_kernel',
            lambda: x * 2.0,
            lambda: x * 2.0 + 1e-2,
            device='cpu', atol=1e-5, rtol=1e-5, strict=True,
        )
    fn_non_strict, rep_non_strict = triton_ops.evaluate_kernel_gate(
        'bad_kernel_soft',
        lambda: x * 2.0,
        lambda: x * 2.0 + 1e-2,
        device='cpu', atol=1e-5, rtol=1e-5, strict=False,
    )
    assert rep_non_strict.enabled is False
    assert rep_non_strict.status == 'rejected_parity_failure'
    torch.testing.assert_close(fn_non_strict(), x * 2.0)

    # Speed gate rejects slower candidate cleanly and returns PyTorch reference:
    fn_slow, rep_slow = triton_ops.evaluate_kernel_gate(
        'slow_candidate',
        lambda: x + 1.0,
        lambda: x + 1.0,
        device='cpu',
        measured_times_ms=(1.5, 1.0),
    )
    assert rep_slow.parity_passed is True
    assert rep_slow.speed_passed is False
    assert rep_slow.enabled is False
    assert rep_slow.status == 'rejected_slower_than_pytorch'
    torch.testing.assert_close(fn_slow(), x + 1.0)

    # Speed gate accepts candidate when parity passes and cand_ms < ref_ms * speed_margin:
    fn_fast, rep_fast = triton_ops.evaluate_kernel_gate(
        'accepted_candidate',
        lambda: x + 1.0,
        lambda: x + 1.0,
        device='cpu',
        measured_times_ms=(0.5, 1.0),
    )
    assert rep_fast.parity_passed is True
    assert rep_fast.speed_passed is True
    assert rep_fast.enabled is True
    assert rep_fast.status == 'accepted'
    assert rep_fast.speedup_ratio == pytest.approx(2.0)
    torch.testing.assert_close(fn_fast(), x + 1.0)


def test_m3_and_m4_never_import_triton_ops_and_m5_end_to_end_smoke_completes(tmp_path):
    # Ensure M4 smoke run never imports edraft.triton_ops when removed from sys.modules:
    saved = sys.modules.pop('edraft.triton_ops', None)
    try:
        eval_closed_loop.main([
            '--stage', 'm4', '--scale', 'smoke', '--mode', 'tune',
            '--device', 'cpu', '--output', str(tmp_path), '--smoke_tokens', '16',
        ])
        assert 'edraft.triton_ops' not in sys.modules
    finally:
        if saved is not None:
            sys.modules['edraft.triton_ops'] = saved

    # Fit M5 long-oracle smoke checkpoint so we also exercise long_oracle_offline in full M5!
    train.main([
        '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
        '--allow_long_oracle_fit',
    ])
    oracle_ckpt = tmp_path / 'm5_long_oracle_smoke' / 'dense_sparse.pt'
    assert oracle_ckpt.is_file()

    # Full end-to-end --stage m5 --scale smoke --mode all across all 17 M5 arms!
    eval_closed_loop.main([
        '--stage', 'm5', '--scale', 'smoke', '--mode', 'all',
        '--device', 'cpu', '--output', str(tmp_path),
        '--smoke_tokens', '16', '--timing_repeats', '2', '--timing_warmups', '1',
        '--m5_oracle_checkpoint', str(oracle_ckpt),
        '--m5_triton', '--m5_torch_compile',
    ])
    summary = json.loads((tmp_path / 'm5' / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['external_baselines_not_run'] == []
    assert summary['greedy_output_comparisons_complete'] is True
    assert summary['greedy_output_comparisons']['numerical_mismatches_observed'] is False
    assert summary['isolated_timing_complete'] is True
    assert summary['triton_gate_report'] is not None
    assert summary['torch_compile_gate_report'] is not None

    families = {c['matched_support_family'] for c in summary['matched_support_comparisons']}
    assert families == {
        'frozen_C0_candidate_masked',
        'full_vocabulary',
        'online_sparse_representation',
        'online_stride5_update_mode',
    }
    compared_arms = {(c['arm'], c['reference_arm']) for c in summary['matched_support_comparisons']}
    assert compared_arms == {
        ('lookahead_static_candidate', 'dense_candidate'),
        ('lookahead_rerelax_candidate', 'dense_candidate'),
        ('lookahead_static_full', 'dense_full'),
        ('lookahead_rerelax_full', 'dense_full'),
        ('exact_observed_pairs_online', 'signed_feature_hash_online'),
        ('online_buffered_stride5', 'offline_online_stride5'),
    }
