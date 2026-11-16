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

    # Full end-to-end --stage m5 --scale smoke --mode all across all 21 M5 arms!
    eval_closed_loop.main([
        '--stage', 'm5', '--scale', 'smoke', '--mode', 'all',
        '--device', 'cpu', '--output', str(tmp_path),
        '--smoke_tokens', '16', '--timing_repeats', '2', '--timing_warmups', '1',
        '--m5_oracle_checkpoint', str(oracle_ckpt),
        '--bigram_count_gate_kappa', '12.0',
        '--m5_triton', '--m5_torch_compile',
    ])
    summary = json.loads((tmp_path / 'm5' / 'summary.json').read_text())
    timed = json.loads((tmp_path / 'm5' / 'timing.json').read_text())
    assert summary['status'] == 'complete'
    assert summary['external_baselines_not_run'] == []
    assert summary['greedy_output_comparisons_complete'] is True
    assert summary['greedy_output_comparisons']['numerical_mismatches_observed'] is False
    assert summary['isolated_timing_complete'] is True
    assert summary['triton_gate_report'] is not None
    assert summary['torch_compile_gate_report'] is not None
    assert summary['settings']['selected']['online_bigram_count_gated']['count_gate_kappa'] == pytest.approx(12.0)

    # Verify M5 offset_bin_slices and head_to_head_online_offset_comparison in both timing.json and summary.json:
    assert len(timed['offset_bin_slices']) > 0
    assert len(summary['offset_bin_slices']) > 0
    for sl in summary['offset_bin_slices']:
        assert [b['bin'] for b in sl['bins']] == ['0-1K', '1K-4K', '4K-8K', '8K-16K', '16K+']
        b0 = sl['bins'][0]
        assert b0['empty_bin'] is False
        assert b0['tokens_per_second']['mean'] > 0
        for higher in sl['bins'][1:]:
            assert higher['empty_bin'] is True
    h2h_pairs = {(c['arm'], c['reference_arm']) for c in summary['head_to_head_online_offset_comparison']}
    assert h2h_pairs == set(eval_closed_loop.M5_ONLINE_HEAD_TO_HEAD_PAIRS)

    families = {c['matched_support_family'] for c in summary['matched_support_comparisons']}
    assert families == {
        'frozen_C0_candidate_masked',
        'full_vocabulary',
        'online_sparse_representation',
        'online_stride5_update_mode',
        'online_bigram_count_gate',
        'online_zero_init_dbS_vs_count_gated_bigram',
        'online_zero_vs_checkpoint_offline_init',
        'online_dbS_zero_vs_checkpoint_offline_init',
    }
    compared_arms = {(c['arm'], c['reference_arm']) for c in summary['matched_support_comparisons']}
    assert compared_arms == {
        ('lookahead_static_candidate', 'dense_candidate'),
        ('lookahead_rerelax_candidate', 'dense_candidate'),
        ('lookahead_static_full', 'dense_full'),
        ('lookahead_rerelax_full', 'dense_full'),
        ('exact_observed_pairs_online', 'signed_feature_hash_online'),
        ('online_buffered_stride5', 'offline_online_stride5'),
        ('online_bigram_count_gated', 'online_bigram'),
        ('online_dbS_zero_init', 'online_bigram_count_gated'),
        ('zero_offline_online_only', 'offline_online_stride1'),
        ('online_dbS_zero_init', 'online_dbS'),
    }


def test_count_gated_online_bigram_and_zero_init_online_dbs_dynamics():
    from edraft.online import OnlineBigram, OnlineState

    for bad_kappa in (0.0, -4.0, float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='count_gate_kappa'):
            OnlineBigram(16, lambda_tr=0.2, alpha=1.0, count_gate_kappa=bad_kappa)

    ungated = OnlineBigram(16, lambda_tr=0.2, alpha=1.0)
    assert 'count_gate_kappa' not in ungated.metrics()
    assert ungated.effective_lambda(3) == 0.0
    ungated.update([3], [5])
    assert ungated.effective_lambda(3) == pytest.approx(0.2)

    gated = OnlineBigram(16, lambda_tr=0.2, alpha=1.0, count_gate_kappa=8.0)
    assert gated.metrics()['count_gate_kappa'] == pytest.approx(8.0)
    # N=0: exact zero weight and zero score perturbation (no Laplace prior distortion!)
    assert gated.effective_lambda(3) == 0.0
    base = torch.linspace(-1.0, 1.0, 16)
    assert gated.score(base, 3) is base

    # N=1 -> 0.2 * 1 / 9; N=8 -> 0.2 * 8 / 16 = 0.1
    gated.update([3], [5])
    assert gated.effective_lambda(3) == pytest.approx(0.2 * 1.0 / 9.0)
    for _ in range(7):
        gated.update([3], [5])
    assert gated.effective_lambda(3) == pytest.approx(0.1)
    torch.testing.assert_close(
        gated.score(base, 3),
        base + 0.1 * gated.prior(3),
    )

    # Verify zero-init online_dbS (`W=0, A=0, b=0, S=0`, `D=4`, `bias_sparse_only=True`):
    torch.manual_seed(77)
    V, d_tok, p, D, K = 12, 4, 5, 4, 4
    E = F.normalize(torch.randn(V, d_tok), dim=-1)
    zero_offline = DenseAdapter(E, p, D, M_sparse=8)
    state = OnlineState(
        zero_offline, enable_sparse=True, sparse_capacity=128,
        lr=0.05, update_stride=1, lambda_l2=0.01, lambda_l1_b=0.0, lambda_l1_s=0.0,
        bias_sparse_only=True,
    )
    snap0 = state.snapshot()
    assert snap0.is_zero is True
    l_unit = torch.randn(K, V).log_softmax(-1)
    H = torch.randn(K, p)
    prefix = [1, 2, 3, 4]
    draft_tokens = [5, 6, 7, 8]
    u0 = snap0.unary(H)
    s0 = torch.stack([snap0.score_slot(l_unit[i], u0[i], prefix, draft_tokens, i) for i in range(K)])
    torch.testing.assert_close(s0, l_unit, rtol=0, atol=0)

    # Target strongly prefers token 9 across slots; online update pushes energy negatively on 9
    target_logits = torch.full((K, V), -4.0)
    target_logits[:, 9] = 6.0
    assert state.update(1, l_unit, target_logits, H, prefix, draft_tokens,
                        first_rejection=1, committed_tokens=[9, 9]) is True
    assert state.u == 1
    assert torch.count_nonzero(state.dW).item() == 0
    assert torch.count_nonzero(state.dA).item() == 0
    # Negative energy on target-favored token 9 increases its proposal score!
    assert float(state.db[9].item()) < 0.0
    snap1 = state.snapshot()
    u1 = snap1.unary(H)
    s1 = torch.stack([snap1.score_slot(l_unit[i], u1[i], prefix, draft_tokens, i) for i in range(K)])
    assert float((s1[:, 9] - l_unit[:, 9]).mean().item()) > 0.0


def test_head_to_head_online_offset_bin_slices_across_all_five_bins():
    from edraft import launch

    # CLI guard checks on eval_closed_loop and launch:
    with pytest.raises(SystemExit):
        eval_closed_loop.main(['--stage', 'm4', '--scale', 'smoke', '--bigram_count_gate_kappa', '8'])
    with pytest.raises(SystemExit):
        eval_closed_loop.main(['--stage', 'm5', '--scale', 'smoke', '--bigram_count_gate_kappa', '0'])
    with pytest.raises(SystemExit):
        launch.main(['--task', 'm4', '--scale', 'smoke', '--bigram_count_gate_kappa', '8'])

    offsets = [0, 1500, 5000, 10000, 18000]  # covers '0-1K', '1K-4K', '4K-8K', '8K-16K', '16K+'
    arms_acc = {
        'plain_dflash': [2, 2, 2, 2, 2],
        'online_bigram': [1, 2, 3, 3, 4],
        'online_bigram_count_gated': [2, 2, 3, 3, 4],
        'offline_online_stride1': [1, 2, 3, 4, 4],
        'zero_offline_online_only': [2, 3, 4, 4, 4],
        'online_dbS': [2, 3, 3, 4, 4],
        'online_dbS_zero_init': [2, 3, 4, 4, 4],
    }

    def make_row(arm, pid, seed, acc_list, *, sec_per_round=0.01, rep_dom=False, repeat=None):
        blocks = []
        for idx, (off, acc) in enumerate(zip(offsets, acc_list)):
            blocks.append({
                'round_idx': idx + 1,
                'absolute_output_offset': off,
                'input_context_length': 128 + off,
                'draft_tokens': [1, 2, 3, 4],
                'accepted_length': acc,
                'committed_tokens': [1] * (acc + 1),
                'round_elapsed_seconds': sec_per_round,
            })
        return {
            'condition': 'long_thinking',
            'temperature': 0.6,
            'sampling_warp_config': {'top_k': 20, 'top_p': 0.95, 'min_p': 0.0},
            'arm': arm,
            'prompt_id': pid,
            'seed': seed,
            'repeat': repeat,
            'phase_profile': False,
            'generation': {'repetition_dominated': rep_dom},
            'blocks': blocks,
        }

    eval_rows, timed_rows = [], []
    for arm, acc_list in arms_acc.items():
        sec = 0.01 if arm != 'online_bigram' else 0.012
        for pid in ('p0', 'p1'):
            eval_rows.append(make_row(arm, pid, 2026, acc_list, sec_per_round=sec))
            for rep in ('0:epoch=e1', '1:epoch=e1'):
                timed_rows.append(make_row(arm, pid, 2026, acc_list, sec_per_round=sec, repeat=rep))
        # Excluded repetition-dominated pair on p_rep:
        eval_rows.append(make_row(arm, 'p_rep', 2026, [4, 4, 4, 4, 4], sec_per_round=sec, rep_dom=True))

    slices = eval_closed_loop.summarize_offset_bin_slices(timed_rows, isolated_timing=True)
    by_arm = {s['arm']: s for s in slices}
    assert set(by_arm) == set(arms_acc)
    z_bins = {b['bin']: b for b in by_arm['online_dbS_zero_init']['bins']}
    assert list(z_bins) == ['0-1K', '1K-4K', '4K-8K', '8K-16K', '16K+']
    for b_label, expected_acc in zip(['0-1K', '1K-4K', '4K-8K', '8K-16K', '16K+'], [2, 3, 4, 4, 4]):
        assert z_bins[b_label]['empty_bin'] is False
        assert z_bins[b_label]['accepted_length']['mean'] == pytest.approx(float(expected_acc))
        assert z_bins[b_label]['tokens_per_second']['mean'] == pytest.approx((expected_acc + 1) / 0.01)

    h2h = eval_closed_loop.compute_head_to_head_online_offset_comparison(eval_rows, timed_rows)
    by_pair = {(c['arm'], c['reference_arm']): c for c in h2h}
    assert set(by_pair) == set(eval_closed_loop.M5_ONLINE_HEAD_TO_HEAD_PAIRS)

    # Check 0-1K vs 16K+ trajectory: online_bigram loses in 0-1K (-1 accepted_length vs plain),
    # while online_bigram_count_gated and online_dbS_zero_init have 0 loss in 0-1K and +2 gain in 16K+!
    bg_vs_plain = {b['bin']: b for b in by_pair[('online_bigram', 'plain_dflash')]['bins']}
    cg_vs_plain = {b['bin']: b for b in by_pair[('online_bigram_count_gated', 'plain_dflash')]['bins']}
    z_vs_cg = {b['bin']: b for b in by_pair[('online_dbS_zero_init', 'online_bigram_count_gated')]['bins']}
    assert by_pair[('online_bigram', 'plain_dflash')]['repetition_excluded_pairs'] == 1
    assert bg_vs_plain['0-1K']['accepted_length_delta']['mean'] == pytest.approx(-1.0)
    assert cg_vs_plain['0-1K']['accepted_length_delta']['mean'] == pytest.approx(0.0)
    assert cg_vs_plain['16K+']['accepted_length_delta']['mean'] == pytest.approx(2.0)
    assert z_vs_cg['4K-8K']['accepted_length_delta']['mean'] == pytest.approx(1.0)
    assert z_vs_cg['4K-8K']['isolated_timing_tokens_per_second']['paired_ratio']['mean'] == pytest.approx(5.0 / 4.0)


def test_verify_fp32_one_ulp_tie_and_guardrail_a_strict_residual(monkeypatch):
    from edraft.sampler import SamplingWarper, probabilities
    from edraft.verify import residual_distribution, verify_block

    # Guardrail A (#56): standalone residual_distribution MUST still raise RuntimeError on zero positive mass!
    with pytest.raises(RuntimeError, match='no positive finite residual mass'):
        residual_distribution(torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]))

    # Exact REPRO-1 (#55) trigger: P and q from real probabilities() with 1-ULP FP32 zero residual,
    # while q[tok] > P[tok] by 1 ULP so u_max = 1 - 2**-24 rejects!
    g = torch.Generator().manual_seed(0)
    warp = SamplingWarper(top_k=20, top_p=0.8)
    triggered = False
    for _ in range(300):
        t_logits = torch.randn(100, generator=g, dtype=torch.float32) * 3.0
        d_logits = t_logits + torch.randn(100, generator=g, dtype=torch.float32) * 1e-6
        P = probabilities(t_logits, 0.7, warp)
        q = probabilities(d_logits, 0.7, warp)
        if float((P - q).clamp_min(0).sum()) == 0.0 and bool((q > P).any()):
            tok = int((q > P).nonzero(as_tuple=True)[0][0])
            # Confirm standalone residual_distribution raises on this exact REPRO-1 pair:
            with pytest.raises(RuntimeError, match='no positive finite residual mass'):
                residual_distribution(P, q)
            # Force uniform draw to u_max = 1 - 2**-24 so verify_block enters the rejection branch:
            u_max = float(torch.tensor(1.0, dtype=torch.float32) - torch.tensor(2**-24, dtype=torch.float32))
            orig_rand = torch.rand
            monkeypatch.setattr(torch, 'rand', lambda shape, **kw: torch.full(shape, u_max,
                                                                              device=kw.get('device'),
                                                                              dtype=kw.get('dtype', torch.float32)))
            try:
                out = verify_block(
                    draft_tokens=[tok],
                    proposal_probs=q.unsqueeze(0),
                    target_logits=torch.stack([t_logits, t_logits]),
                    temperature=0.7,
                    warper=warp,
                    generator=torch.Generator().manual_seed(42),
                )
            finally:
                monkeypatch.setattr(torch, 'rand', orig_rand)
            assert 0 <= out.accepted_length <= 1
            assert len(out.emitted_tokens) in (1, 2)
            triggered = True
            break
    assert triggered, 'Expected exact REPRO-1 1-ULP zero-residual trigger within 300 trials'

    # Also test exact bitwise P == q tie with u_max rejection -> safely flips to accepted:
    logits_tie = torch.linspace(-2.0, 2.0, 16, dtype=torch.float32)
    q_tie = probabilities(logits_tie, 0.7, SamplingWarper(top_k=8, top_p=0.9))
    q_over = q_tie.clone()
    tok0 = int(q_tie.argmax())
    q_over[tok0] = q_tie[tok0] + 1e-7  # tiny unnormalized excess with zero positive (P - q) everywhere
    u_max = float(torch.tensor(1.0, dtype=torch.float32) - torch.tensor(2**-24, dtype=torch.float32))
    orig_rand = torch.rand
    monkeypatch.setattr(torch, 'rand', lambda shape, **kw: torch.full(shape, u_max,
                                                                      device=kw.get('device'),
                                                                      dtype=kw.get('dtype', torch.float32)))
    try:
        out_tie = verify_block(
            draft_tokens=[tok0],
            proposal_probs=q_over.unsqueeze(0),
            target_logits=torch.stack([logits_tie, logits_tie]),
            temperature=0.7,
            warper=SamplingWarper(top_k=8, top_p=0.9),
            generator=torch.Generator().manual_seed(99),
        )
    finally:
        monkeypatch.setattr(torch, 'rand', orig_rand)
    assert len(out_tie.emitted_tokens) in (1, 2)


def test_train_multi_shard_block_shuffle_window_and_bit_exact_resume(tmp_path):
    parser_args = [
        '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
        '--epochs_dense', '1', '--epochs_sparse', '1',
        '--batch_blocks', '8', '--checkpoint_every', '1',
    ]
    with pytest.raises(ValueError, match='block_shuffle_window'):
        train.main([*parser_args, '--block_shuffle_window', '-1'])

    # Run sequential baseline (window=0) and shuffled (window=4):
    train.main([*parser_args, '--block_shuffle_window', '0'])
    train.main([*parser_args, '--block_shuffle_window', '4'])
    seq_report = json.loads((tmp_path / 'm2_smoke.json').read_text())
    shuf_name = 'm5_union_rest_grouped_signed_feature_hash_shuf4_smoke'
    shuf_report = json.loads((tmp_path / f'{shuf_name}.json').read_text())
    assert shuf_report['block_shuffle_window'] == 4
    # Exact 1:1 bijection over blocks -> identical initial baseline & sparse lookup key multiset!
    assert shuf_report['baseline']['dev']['weighted_mean_kl'] == pytest.approx(
        seq_report['baseline']['dev']['weighted_mean_kl']
    )
    seq_sp = seq_report['stages']['dense_sparse']['sparse_collision_statistics']
    shuf_sp = shuf_report['stages']['dense_sparse']['sparse_collision_statistics']
    assert shuf_sp['observed_lookup_events'] == seq_sp['observed_lookup_events']
    assert shuf_sp['unique_observed_keys'] == seq_sp['unique_observed_keys']

    # Verify epoch_references mixes multiple shards/prompts inside a single batch when window=4:
    entries, idx_rep = train.trace_index(tmp_path / 'm2_smoke_traces')
    refs_seq = train.epoch_references(entries, 'train', seed=2026, block_shuffle_window=0)
    refs_shuf = train.epoch_references(entries, 'train', seed=2026, block_shuffle_window=4)
    assert sorted((s.path, i) for s, i in refs_seq) == sorted((s.path, i) for s, i in refs_shuf)
    first_batch_prompts_shuf = {s.prompt_id for s, _ in refs_shuf[:8]}
    assert len(first_batch_prompts_shuf) >= 2

    # Verify mid-epoch --resume bit-exactness (rtol=0, atol=0) AND identity invalidation on window change:
    from types import SimpleNamespace
    from edraft.toy import ToyParallelDrafter, ToyTarget
    from edraft.features import fit_token_features

    target = ToyTarget(vocab_size=8, hidden_size=12, seed=2026)
    draft = ToyParallelDrafter(target, seed=2027)
    calibration = [[i % 8, (i + 3) % 8, (i * 3 + 1) % 8] for i in range(64)]
    E, _ = fit_token_features(target.embedding, 4, 2026, torch.tensor(sum(calibration, [])))
    head, bias, head_meta = draft.draft_lm_head.weight, draft.draft_lm_head.bias, draft.head_metadata()

    def make_ns(out_dir, win, tag='shuf_resume_test'):
        return SimpleNamespace(
            scale='smoke', device='cpu', output=out_dir, seed=2026,
            learning_rate=1e-3, weight_decay=1e-4, bias_weight_decay=0.0,
            sparse_learning_rate=0.01, lambda_l1=1e-5, sparse_capacity=256,
            M_sparse=4, history_depth=4, batch_blocks=4, vocab_chunk=8192,
            epochs_dense=1, epochs_sparse=1, checkpoint_every=1,
            stats_memory_limit='64MB', stats_benchmark_entries=1024,
            sparse_backend='signed_feature_hash', sparse_overflow_policy='freeze_new',
            objective='union_rest_grouped', allow_long_oracle_fit=False,
            allow_pure_t1_ablation=False, block_shuffle_window=win,
            m5_fit_tag=tag, resume=True,
        )

    import random
    out_uninterrupted = tmp_path / 'uninterrupted'
    out_resumed = tmp_path / 'resumed'
    torch.manual_seed(2026)
    random.seed(2026)
    train.fit(entries, idx_rep, E, head, bias, head_meta, make_ns(out_uninterrupted, 4))

    torch.manual_seed(2026)
    random.seed(2026)
    interrupted = train.fit(entries, idx_rep, E, head, bias, head_meta, make_ns(out_resumed, 4), stop_after_steps=3)
    assert interrupted['status'] == 'interrupted_for_resume_test'

    # Changing block_shuffle_window from 4 -> 3 on same checkpoint dir MUST fail identity validation!
    with pytest.raises(ValueError, match='Checkpoint identity differs'):
        train.fit(entries, idx_rep, E, head, bias, head_meta, make_ns(out_resumed, 3))

    # Resuming with matching block_shuffle_window=4 produces rtol=0, atol=0 bit-exact parameters!
    train.fit(entries, idx_rep, E, head, bias, head_meta, make_ns(out_resumed, 4))
    a_full, _ = train.load_adapter_checkpoint(out_uninterrupted / 'shuf_resume_test_smoke' / 'dense_sparse.pt', 'cpu')
    a_res, _ = train.load_adapter_checkpoint(out_resumed / 'shuf_resume_test_smoke' / 'dense_sparse.pt', 'cpu')
    for k, v in a_full.state_dict().items():
        torch.testing.assert_close(v, a_res.state_dict()[k], rtol=0, atol=0)


def test_m5_audit_bug1_matched_c0_identical_across_divergent_tuned_betas(tmp_path, monkeypatch):
    from edraft import adapter as adapter_mod
    from edraft.runtime import common_parser, fingerprint, parse_config

    parser = common_parser('test')
    parser.add_argument('--stage', default='m5')
    parser.add_argument('--smoke_tokens', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--long_max_new_tokens', type=int, default=32)
    args = parse_config(parser, ['--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path)])
    runner = eval_closed_loop.EvaluationRunner(args)
    # Perturb W/A/b strongly so beta=0.5 vs beta=1.0 would change TopM if beta_relax were not shared!
    with torch.no_grad():
        torch.manual_seed(909)
        runner.dense_sparse.W.copy_(torch.randn_like(runner.dense_sparse.W) * 4.0)
        runner.dense_sparse.A.copy_(torch.randn_like(runner.dense_sparse.A) * 4.0)
        runner.dense_sparse.b.copy_(torch.randn_like(runner.dense_sparse.b) * 4.0)

    captured = {}
    orig_c0 = adapter_mod.compute_frozen_candidates

    def spy_c0(snapshot, l_unit, U_all, beta_relax=1.0):
        c0 = orig_c0(snapshot, l_unit, U_all, beta_relax=beta_relax)
        captured.setdefault(current_arm[0], []).append(tuple(x.clone() for x in c0))
        return c0

    monkeypatch.setattr(adapter_mod, 'compute_frozen_candidates', spy_c0)
    monkeypatch.setattr('edraft.dflash_loop.compute_frozen_candidates', spy_c0)

    prompt = {
        'prompt_id': 'synthetic-test-standard_nonthinking-0',
        'split': 'test', 'condition': 'standard_nonthinking',
        'text': 'Synthetic 0', 'dataset': 'builtin', 'reference_selected': True,
    }
    warp = SamplingWarper(top_k=4, top_p=0.9)
    current_arm = ['dense_candidate']
    runner.run(
        {'prompt': prompt, 'arm': 'dense_candidate', 'sampling_name': 'primary',
         'temperature': 0.7, 'warper': warp, 'seed': 2026},
        {'beta': 0.5, 'beta_relax': 1.0},
    )
    current_arm[0] = 'lookahead_rerelax_candidate'
    runner.run(
        {'prompt': prompt, 'arm': 'lookahead_rerelax_candidate', 'sampling_name': 'primary',
         'temperature': 0.7, 'warper': warp, 'seed': 2026},
        {'beta': 1.0, 'beta_relax': 1.0, 'R_init': 2, 'R_clamp': 1, 'eta': 0.5, 'T_relax': 1.0},
    )
    current_arm[0] = 'lookahead_static_candidate'
    runner.run(
        {'prompt': prompt, 'arm': 'lookahead_static_candidate', 'sampling_name': 'primary',
         'temperature': 0.7, 'warper': warp, 'seed': 2026},
        {'beta': 1.0, 'beta_relax': 1.0, 'R_init': 2, 'R_clamp': 1, 'eta': 0.5, 'T_relax': 1.0},
    )

    c0_dense_r1 = captured['dense_candidate'][0]
    c0_rerelax_r1 = captured['lookahead_rerelax_candidate'][0]
    c0_static_r1 = captured['lookahead_static_candidate'][0]
    for j in range(len(c0_dense_r1)):
        assert torch.equal(c0_dense_r1[j], c0_rerelax_r1[j])
        assert torch.equal(c0_dense_r1[j], c0_static_r1[j])

    # Verify validate_frozen_settings rejects mismatched beta_relax across matched-support arms:
    manifest = eval_closed_loop.smoke_manifest()
    bad_settings = {
        'frozen': True, 'stage': 'm5', 'scale': 'smoke',
        'checkpoint_hashes': runner.checkpoint_hashes,
        'prompt_manifest_hash': fingerprint(manifest),
        'development_prompt_ids': [p['prompt_id'] for p in manifest['prompts'] if p['split'] == 'dev'],
        'selected': {
            'plain_dflash': {},
            'dense_candidate': {'beta': 0.5, 'beta_relax': 0.5},
            'lookahead_rerelax_candidate': {'beta': 1.0, 'beta_relax': 1.0, 'R_init': 2, 'R_clamp': 1, 'eta': 0.5, 'T_relax': 1.0},
        },
    }
    with pytest.raises(ValueError, match='identical beta_relax'):
        eval_closed_loop.validate_frozen_settings(
            bad_settings, 'm5', 'smoke', runner.checkpoint_hashes, manifest,
            m5_arms=['plain_dflash', 'dense_candidate', 'lookahead_rerelax_candidate'],
        )


def test_m5_audit_bug2_mixed_dtypes_and_lookahead_scalar_validation():
    import math
    from edraft.adapter import (
        LookaheadConfig, compute_frozen_candidates, damped_lookahead_step,
        initialize_lookahead, lookahead_energy,
    )
    from edraft.sparse import SignedHash

    # Strict validation against NaN / Inf / bool on LookaheadConfig and lookahead functions:
    for bad in (float('nan'), float('inf'), float('-inf'), True, False, 0.0, -1.0):
        with pytest.raises(ValueError):
            LookaheadConfig(mode='static', T_relax=bad)
    for bad_eta in (float('nan'), float('inf'), True, False, 0.0, 1.5):
        with pytest.raises(ValueError):
            LookaheadConfig(mode='static', eta=bad_eta)
    for bad_br in (float('nan'), float('inf'), float('-inf'), True):
        with pytest.raises(ValueError):
            LookaheadConfig(mode='static', beta_relax=bad_br)
    for bad_int in (True, False, -1):
        with pytest.raises(ValueError):
            LookaheadConfig(mode='rerelax', R_init=bad_int, R_clamp=1)
        with pytest.raises(ValueError):
            LookaheadConfig(mode='rerelax', R_init=2, R_clamp=bad_int)

    torch.manual_seed(314)
    V, d_tok, p, D, K = 16, 4, 6, 3, 4
    E = F.normalize(torch.randn(V, d_tok), dim=-1)
    sp = SignedHash(64)
    adapter = DenseAdapter(E, p, D, sparse=sp, M_sparse=6)
    with torch.no_grad():
        sp.values.normal_(0.0, 0.2)
        adapter.W.normal_(0.0, 0.3)
        adapter.A.normal_(0.0, 0.3)
        adapter.b.normal_(0.0, 0.3)

    for input_dtype in (torch.bfloat16, torch.float32, torch.float64):
        l_unit = torch.randn(K, V, dtype=torch.float32).log_softmax(-1).to(input_dtype)
        H = torch.randn(K, p, dtype=input_dtype)
        prefix = [1, 2, 3, 4]
        base_snap = adapter.snapshot()
        u_all = base_snap.unary(H)
        c0 = compute_frozen_candidates(base_snap, l_unit, u_all, beta_relax=1.0)
        cand_snap = base_snap.with_candidates(c0)
        u_cand = cand_snap.unary(H, proposal_mode='dense_candidate')
        realized = []
        for i in range(K):
            sc = cand_snap.score_slot(l_unit[i], u_cand[i], prefix, realized, i,
                                      beta=0.8, proposal_mode='dense_candidate')
            assert sc.shape == (V,)
            assert int(torch.isfinite(sc).sum().item()) == 6
            realized.append(int(sc.argmax().item()))

        for domain in ('candidate_masked', 'full_vocabulary'):
            state = initialize_lookahead(
                base_snap, l_unit, H, prefix,
                domain=domain, T_relax=0.9, beta_relax=1.0,
            )
            e_val = lookahead_energy(state)
            assert math.isfinite(float(e_val.item()))
            tok0, probs0 = state.sample_slot(0, T_sample=0.7, beta_sample=0.8,
                                             warper=SamplingWarper(top_k=4, top_p=0.9),
                                             generator=torch.Generator().manual_seed(11))
            assert 0 <= tok0 < V
            assert torch.isfinite(probs0).all()
            damped_lookahead_step(state, eta=0.5)

    # Also verify initialize_lookahead / compute_frozen_candidates / damped_lookahead_step reject NaN/bool directly:
    snap_chk = adapter.snapshot()
    l_chk = torch.randn(K, V).log_softmax(-1)
    h_chk = torch.randn(K, p)
    with pytest.raises(ValueError):
        compute_frozen_candidates(snap_chk, l_chk, snap_chk.unary(h_chk), beta_relax=float('nan'))
    with pytest.raises(ValueError):
        initialize_lookahead(snap_chk, l_chk, h_chk, [1, 2], T_relax=float('inf'))
    with pytest.raises(ValueError):
        initialize_lookahead(snap_chk, l_chk, h_chk, [1, 2], T_relax=True)
    with pytest.raises(ValueError):
        initialize_lookahead(snap_chk, l_chk, h_chk, [1, 2], beta_relax=float('nan'))
    st_ok = initialize_lookahead(snap_chk, l_chk, h_chk, [1, 2])
    with pytest.raises(ValueError):
        damped_lookahead_step(st_ok, eta=float('nan'))
    with pytest.raises(ValueError):
        damped_lookahead_step(st_ok, eta=True)


def test_m5_audit_bug3_mid_block_eos_on_block_committed_tokens_and_empty_block(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from dflash.model import DFlashDraftModel
    from edraft.backend import ReferenceBackend
    from edraft.dflash_loop import generate_reference
    from edraft.features import fit_hidden_projection, fit_token_features
    from edraft.online import OnlineState
    from edraft.traces import TraceShardWriter, compact_block

    torch.manual_seed(91)
    V_mock = 24
    t_cfg = Qwen3Config(
        vocab_size=V_mock, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
        eos_token_id=23, attn_implementation='sdpa',
    )
    t_cfg._attn_implementation = 'sdpa'
    target = Qwen3ForCausalLM(t_cfg).eval()
    d_cfg = Qwen3Config(
        vocab_size=V_mock, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
        attn_implementation='sdpa',
    )
    d_cfg._attn_implementation = 'sdpa'
    d_cfg.num_target_layers = 3
    d_cfg.dflash_config = {'block_size': 5, 'mask_token_id': 22, 'target_layer_ids': [0, 2]}
    drafter = DFlashDraftModel(d_cfg).eval()
    backend = ReferenceBackend(
        target=target, drafter=drafter,
        target_id='tiny-qwen3', drafter_id='tiny-dflash',
        target_revision='r0', drafter_revision='r0',
    )
    proj = fit_hidden_projection([torch.randn(4, 32) for _ in range(8)], 32, 4, seed=1)
    E, _ = fit_token_features(torch.randn(V_mock, 32), 4, 1, torch.arange(V_mock))

    inp = torch.tensor([[2, 5, 9, 1]], dtype=torch.long)
    seq = backend.target_only_generate(inp, 10, temperature=0.0, stop_token_ids=())[0, 4:].tolist()
    eos_tok = seq[2]  # Force EOS mid-block inside Round 1!
    compacted = []
    meta = {
        'prompt_id': 'eos-p0', 'split': 'train', 'condition': 'standard_nonthinking',
        'seed': 1, 'scale': 'smoke', 'model_pair': ['m-t', 'm-d'],
        'projection_metadata': proj.metadata, 'history_depth': 4,
    }
    res = generate_reference(
        backend, inp, max_new_tokens=10,
        temperature=0.0, warper=SamplingWarper(), stop_token_ids={eos_tok},
        projection=proj, retain_tensors=True,
        on_block=lambda rec: compacted.append(compact_block(rec, meta, M_draft=4, M_target=2, retain_live_full=True)),
    )
    actual_committed = res.blocks[-1]['committed_tokens']
    cb = compacted[-1]
    # Crucial check (#68 M5-BUG-3): committed_tokens is populated BEFORE on_block(record), with zero post-EOS leakage!
    assert 'committed_tokens' in cb
    assert cb['committed_tokens'] == actual_committed
    assert train._block_own_committed(cb) == actual_committed

    # Write shard and verify train._annotate_verified_stream + legacy sidecar fallback + empty block!
    writer = TraceShardWriter(tmp_path / 'eos_traces')
    for split_name in ('train', 'dev', 'test'):
        split_blocks = [dict(b, split=split_name, prompt_id=f'eos-{split_name}') for b in compacted]
        identity = {
            'model_pair': ['m-t', 'm-d'], 'prompt': f'eos-{split_name}', 'condition': 'standard_nonthinking',
            'seed': 1, 'temperature': 0.0, 'verification_rule': 'greedy',
            'proposal_mode': 'plain_dflash', 'checkpoint': 'v1', 'adapter_version': 0,
            'sampling_warp_config': SamplingWarper().to_dict(),
        }
        writer.write(identity, split_blocks, {'generation': {
            'output_tokens': res.output_tokens, 'realized_length': len(res.output_tokens),
            'condition': 'standard_nonthinking', 'split': split_name, 'seed': 1,
            'temperature': 0.0, 'sampling_warp_config': SamplingWarper().to_dict(),
            'partition': 'primary', 'complete': True, 'capped': False, 'eos_reached': True,
        }})
    entries, _ = train.trace_index(tmp_path / 'eos_traces')
    refs = train.epoch_references(entries, 'train')
    loaded_blocks = next(train.batches(refs, 4, history_depth=4))[1]
    assert loaded_blocks[-1]['_verified_committed_window'] == actual_committed
    assert loaded_blocks[-1]['_verified_eval_slots'] == min(cb['block_length'], len(actual_committed))

    # Also test legacy block without committed_tokens using sidecar_generation slicing, AND empty committed=[]:
    legacy_blocks = [dict(b) for b in compacted]
    for lb in legacy_blocks:
        lb.pop('committed_tokens', None)
        lb.pop('_verified_committed_window', None)
        lb.pop('_verified_eval_slots', None)
    train._annotate_verified_stream(
        legacy_blocks, 4,
        sidecar_generation={'output_tokens': res.output_tokens, 'realized_length': len(res.output_tokens)},
    )
    assert legacy_blocks[-1]['_verified_committed_window'] == actual_committed
    assert legacy_blocks[-1]['_verified_eval_slots'] == min(cb['block_length'], len(actual_committed))

    empty_blk = dict(cb, committed_tokens=[])
    empty_blk.pop('_verified_committed_window', None)
    empty_blk.pop('_verified_eval_slots', None)
    train._annotate_verified_stream([empty_blk], 4)
    assert empty_blk['_verified_eval_slots'] == 0
    adapter = DenseAdapter(E, 4, 4, M_sparse=4)
    l_u = torch.randn(cb['block_length'], V_mock).log_softmax(-1)
    loss_empty, w_empty, _ = train.batch_loss(adapter, [empty_blk], l_u, objective='two_sided_pseudolikelihood')
    assert float(w_empty.item()) == 0.0
    assert float(loss_empty.item()) == 0.0

    # OnlineState two_sided_pseudolikelihood with committed_tokens=[] is a clean no-op:
    ostate = OnlineState(adapter, lr=0.05, update_stride=1, loss_mode='two_sided_pseudolikelihood')
    did_step = ostate.update(1, l_u, None, torch.randn(cb['block_length'], 4), [1, 2], [7, 4, 5, 6][:cb['block_length']],
                             first_rejection=0, committed_tokens=[])
    assert did_step is False
    assert ostate.u == 0


def test_m5_triton_autograd_kl_and_runner_active_kernel_dispatch(tmp_path, monkeypatch):
    from edraft import triton_ops
    from edraft.adapter import block_unary, full_kl
    from edraft.runtime import common_parser, parse_config
    from edraft.sparse import hash_keys

    # 1. Verify _TritonRowwiseKLFunction forward + backward parity across dtypes and 1D/broadcast weights:
    torch.manual_seed(777)
    for dt in (torch.bfloat16, torch.float32, torch.float64):
        s_ref = torch.randn(5, 37, dtype=dt, requires_grad=True)
        s_cand = s_ref.detach().clone().requires_grad_(True)
        t_logits = torch.randn(5, 37, dtype=dt)
        for w in (None, torch.linspace(0.2, 1.0, 5), torch.ones(2, 5) * 0.5):
            l_ref = full_kl(s_ref, t_logits, w)
            l_cand = triton_ops.emulate_rowwise_kl_cpu(s_cand, t_logits, w)
            assert l_cand.requires_grad is True
            l_ref.backward()
            l_cand.backward()
            atol = 2e-2 if dt == torch.bfloat16 else 1e-5
            torch.testing.assert_close(l_cand.detach().float(), l_ref.detach().float(), atol=atol, rtol=atol)
            assert s_cand.grad.dtype == dt
            torch.testing.assert_close(s_cand.grad.float(), s_ref.grad.float(), atol=atol, rtol=atol)
            s_ref.grad = None
            s_cand.grad = None

    # 2. Verify evaluate_kernel_gate rejects wrong backward gradient via grad_parity_fn:
    x = torch.randn(4, 8)
    with pytest.raises(triton_ops.TritonParityError):
        triton_ops.evaluate_kernel_gate(
            'bad_grad_kl_strict',
            lambda: x.sum(),
            lambda: x.sum(),
            device='cpu',
            strict=True,
            measured_times_ms=(0.2, 1.0),
            grad_parity_fn=lambda: (torch.ones_like(x), torch.zeros_like(x)),
        )
    _, bad_grad_rep = triton_ops.evaluate_kernel_gate(
        'bad_grad_kl',
        lambda: x.sum(),
        lambda: x.sum(),
        device='cpu',
        strict=False,
        measured_times_ms=(0.2, 1.0),
        grad_parity_fn=lambda: (torch.ones_like(x), torch.zeros_like(x)),
    )
    assert bad_grad_rep.enabled is False
    assert bad_grad_rep.parity_passed is False
    assert bad_grad_rep.status == 'rejected_parity_failure'

    # 3. Verify EvaluationRunner wires accepted gated callables into live proposal & online execution!
    counts = {'signed_hash_lookup': 0, 'rowwise_full_kl': 0, 'torch_compile_block_unary': 0}

    def fast_hash(d, previous, current, values, capacity, *, last_u=None, cum_thresh=None, u=0):
        counts['signed_hash_lookup'] += 1
        return triton_ops.emulate_signed_hash_kernel_cpu(
            d, previous, current, values, capacity, last_u=last_u, cum_thresh=cum_thresh, u=u,
        )

    def fast_kl(scores, target_logits, weights=None):
        counts['rowwise_full_kl'] += 1
        return triton_ops.emulate_rowwise_kl_cpu(scores, target_logits, weights)

    def fast_unary(E, W, b, H):
        counts['torch_compile_block_unary'] += 1
        return block_unary(E, W, b, H)

    accepted_rep = lambda name: triton_ops.TritonGateReport(
        kernel_name=name, enabled=True, status='accepted', device='cpu',
        parity_passed=True, max_abs_error=0.0, atol=1e-5, rtol=1e-5,
        speed_passed=True, triton_median_ms=0.1, pytorch_median_ms=0.5,
        speedup_ratio=5.0,
    )

    monkeypatch.setattr(
        triton_ops, 'build_gated_m5_kernels',
        lambda *a, **kw: (
            ({'signed_hash_lookup': fast_hash, 'rowwise_full_kl': fast_kl},
             {'signed_hash_lookup': accepted_rep('signed_hash_lookup'),
              'rowwise_full_kl': accepted_rep('rowwise_full_kl')})
            if kw.get('return_callables') else
            {'signed_hash_lookup': accepted_rep('signed_hash_lookup'),
             'rowwise_full_kl': accepted_rep('rowwise_full_kl')}
        ),
    )
    monkeypatch.setattr(
        triton_ops, 'build_gated_torch_compile',
        lambda *a, **kw: (
            ({'torch_compile_block_unary': fast_unary},
             {'torch_compile_block_unary': accepted_rep('torch_compile_block_unary')})
            if kw.get('return_callables') else
            {'torch_compile_block_unary': accepted_rep('torch_compile_block_unary')}
        ),
    )

    parser = common_parser('test')
    parser.add_argument('--stage', default='m5')
    parser.add_argument('--smoke_tokens', type=int, default=12)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--long_max_new_tokens', type=int, default=32)
    parser.add_argument('--m5_triton', action='store_true', default=True)
    parser.add_argument('--m5_torch_compile', action='store_true', default=True)
    args = parse_config(parser, ['--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path)])
    runner = eval_closed_loop.EvaluationRunner(args)
    with torch.no_grad():
        runner.dense_sparse.sparse.values.normal_(0.0, 0.2)

    prompt = {
        'prompt_id': 'synthetic-test-standard_nonthinking-0',
        'split': 'test', 'condition': 'standard_nonthinking',
        'text': 'Synthetic 0', 'dataset': 'builtin', 'reference_selected': True,
    }
    for arm in ('dense_full', 'offline_online_stride1', 'zero_offline_online_only'):
        setting = eval_closed_loop.candidate_settings(arm)[0]
        row = runner.run(
            {'prompt': prompt, 'arm': arm, 'sampling_name': 'primary',
             'temperature': 0.7, 'warper': SamplingWarper(top_k=4, top_p=0.9), 'seed': 2026},
            setting,
        )
        assert row['m5_telemetry']['active_dispatched_kernels'] == [
            'rowwise_full_kl', 'signed_hash_lookup', 'torch_compile_block_unary'
        ]
    assert counts['signed_hash_lookup'] > 0
    assert counts['rowwise_full_kl'] > 0
    assert counts['torch_compile_block_unary'] > 0


def test_m5_exact_observed_pairs_zero_collision_reporting_and_beta_calibration(tmp_path):
    # 1. ExactObservedPairs train.py report must record 0 active collisions + counterfactual SignedHash stats (#69 M5-BUG-4):
    train.main([
        '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
        '--sparse_backend', 'exact_observed_pairs', '--sparse_capacity', '256',
        '--m5_fit_tag', 'exact_pairs_audit',
    ])
    exact_rep = json.loads((tmp_path / 'exact_pairs_audit_smoke.json').read_text())
    cstats = exact_rep['stages']['dense_sparse']['sparse_collision_statistics']
    assert cstats['representation'] == 'exact_observed_pairs'
    assert cstats['collision_count'] == 0
    assert cstats['frequency_weighted_collision_count'] == 0
    assert cstats['collision_rate_per_nonempty_bucket'] == 0.0
    assert cstats['frequency_weighted_collision_rate'] == 0.0
    assert 'counterfactual_signed_hash_collision_count' in cstats
    if isinstance(cstats.get('lookup_benchmark'), dict):
        assert cstats['lookup_benchmark']['benchmarked_representation'] == 'counterfactual_signed_feature_hash'

    # 2. Post-fit scalar beta calibration on dev live_full KL + checkpoint scaling (#71):
    train.main([
        '--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path),
        '--block_shuffle_window', '4', '--weight_decay', '1e-2',
        '--calibrate_beta', '--apply_calibrated_beta',
        '--m5_fit_tag', 'm5_calibrated_refit',
    ])
    cal_rep = json.loads((tmp_path / 'm5_calibrated_refit_smoke.json').read_text())
    for st in ('dense', 'dense_sparse'):
        bcal = cal_rep['stages'][st]['beta_calibration']
        assert bcal['metric'] == 'dev_live_full_kl'
        assert bcal['dev_live_full_logical_slots'] > 0
        assert bcal['applied_to_parameters'] is True
        assert bcal['calibrated_kl'] <= bcal['baseline_beta0_kl'] + 1e-9
        assert bcal['calibrated_kl'] <= bcal['uncalibrated_beta1_kl'] + 1e-9
        assert Path(cal_rep['checkpoints'][f'{st}_calibrated']).is_file()
        assert Path(cal_rep['checkpoints'][f'{st}_unscaled']).is_file()

    # Verify standalone --scale_checkpoint_from CLI (e.g. 0.20 * unscaled -> scaled020.pt) and linear score identity:
    unscaled_pt = Path(cal_rep['checkpoints']['dense_sparse_unscaled'])
    scaled020_pt = tmp_path / 'dense_sparse_020.pt'
    train.main([
        '--output', str(tmp_path),
        '--scale_checkpoint_from', str(unscaled_pt),
        '--scale_checkpoint_to', str(scaled020_pt),
        '--checkpoint_scale', '0.20',
    ])
    assert scaled020_pt.is_file()
    a_unscaled, _ = train.load_adapter_checkpoint(unscaled_pt, 'cpu')
    a_scaled, scaled_meta = train.load_adapter_checkpoint(scaled020_pt, 'cpu')
    assert scaled_meta['checkpoint_scale'] == pytest.approx(0.20)
    torch.testing.assert_close(a_scaled.E, a_unscaled.E, rtol=0, atol=0)

    entries, _ = train.trace_index(tmp_path / 'm2_smoke_traces')
    dev_refs = train.epoch_references(entries, 'dev')
    blocks = next(train.batches(dev_refs, 4, history_depth=4))[1]
    l_unit = torch.randn(sum(b['block_length'] for b in blocks), a_unscaled.E.shape[0]).log_softmax(-1)
    s_via_beta = train.corrected_batch_scores(a_unscaled, blocks, l_unit, beta=0.20)
    s_via_ckpt = train.corrected_batch_scores(a_scaled, blocks, l_unit, beta=1.0)
    torch.testing.assert_close(s_via_ckpt, s_via_beta, atol=1e-6, rtol=1e-6)

    # Verify closed-loop M5 evaluation supports --checkpoint_scale and --m5_beta_candidates out of the box:
    eval_closed_loop.main([
        '--stage', 'm5', '--scale', 'smoke', '--mode', 'all',
        '--device', 'cpu', '--output', str(tmp_path / 'm5_scaled_eval'),
        '--smoke_tokens', '12', '--timing_repeats', '2', '--timing_warmups', '1',
        '--m5_arms', 'plain_dflash', 'dense_full', 'zero_offline_online_only',
        'offline_online_stride1', 'online_dbS_zero_init', 'online_bigram_count_gated',
        '--checkpoint_scale', '0.20',
        '--m5_beta_candidates', '0.5', '1.0',
    ])
    scaled_summary = json.loads((tmp_path / 'm5_scaled_eval' / 'm5' / 'summary.json').read_text())
    assert scaled_summary['status'] == 'complete'
    assert scaled_summary['settings']['checkpoint_hashes']['checkpoint_scale'] == pytest.approx(0.20)


def test_m5_r2_triton_kl_inference_mode_hvp_neginf_and_signed_hash_grad():
    """Regression suite for M5-R2-BUG-1 and M5-R2-BUG-2 (#77)."""
    import inspect
    from edraft import triton_ops
    from edraft.adapter import full_kl
    from edraft.sparse import hash_keys, proximal_l1

    torch.manual_seed(2026)

    # 1a. target_logits and weights created inside torch.inference_mode() + call inside inference_mode():
    with torch.inference_mode():
        t_inf = torch.randn(4, 32)
        w_inf = torch.tensor([1.0, 0.8, 0.6, 0.4])
        s_inf = torch.randn(4, 32)
        val_inf = triton_ops.emulate_rowwise_kl_cpu(s_inf, t_inf, w_inf)
        assert torch.isfinite(val_inf)

    s_grad = torch.randn(4, 32, requires_grad=True)
    s_ref = s_grad.detach().clone().requires_grad_(True)
    loss_cand = triton_ops.emulate_rowwise_kl_cpu(s_grad, t_inf, w_inf)
    loss_ref = full_kl(s_ref, t_inf.clone(), w_inf.clone())
    torch.testing.assert_close(loss_cand, loss_ref, atol=1e-6, rtol=1e-6)
    loss_cand.backward()
    loss_ref.backward()
    torch.testing.assert_close(s_grad.grad, s_ref.grad, atol=1e-6, rtol=1e-6)

    # 1b. create_graph=True 2nd-order HVP parity (including weights requiring grad):
    s2 = torch.randn(3, 24, requires_grad=True)
    w2 = torch.tensor([1.0, 0.7, 0.5], requires_grad=True)
    s2_ref = s2.detach().clone().requires_grad_(True)
    w2_ref = w2.detach().clone().requires_grad_(True)
    t2 = torch.randn(3, 24)
    g_s, g_w = torch.autograd.grad(
        triton_ops.emulate_rowwise_kl_cpu(s2, t2, w2), (s2, w2), create_graph=True
    )
    g_s_ref, g_w_ref = torch.autograd.grad(
        full_kl(s2_ref, t2, w2_ref), (s2_ref, w2_ref), create_graph=True
    )
    assert g_s.requires_grad is True
    torch.testing.assert_close(g_s, g_s_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(g_w, g_w_ref, atol=1e-6, rtol=1e-6)
    vec = torch.randn_like(s2)
    hvp_cand = torch.autograd.grad((g_s * vec).sum(), s2)[0]
    hvp_ref = torch.autograd.grad((g_s_ref * vec).sum(), s2_ref)[0]
    torch.testing.assert_close(hvp_cand, hvp_ref, atol=1e-6, rtol=1e-6)

    # 1c. Leading 1024-column -inf tile + scattered in-bounds -inf target logits + n_rows == 0:
    s_tile = torch.randn(2, 2048)
    s_tile[0, :1024] = -float('inf')
    s_tile.requires_grad_(True)
    s_tile_ref = s_tile.detach().clone().requires_grad_(True)
    t_tile = torch.randn(2, 2048)
    t_tile[0, :1024] = -float('inf')
    t_tile[1, :1024] = -float('inf')
    t_tile[1, 1500] = -float('inf')
    out_tile = triton_ops.emulate_rowwise_kl_cpu(s_tile, t_tile)
    out_tile_ref = full_kl(s_tile_ref, t_tile)
    assert torch.isfinite(out_tile)
    torch.testing.assert_close(out_tile, out_tile_ref, atol=1e-5, rtol=1e-5)
    out_tile.backward()
    out_tile_ref.backward()
    assert torch.isfinite(s_tile.grad).all()
    torch.testing.assert_close(s_tile.grad, s_tile_ref.grad, atol=1e-5, rtol=1e-5)

    s_empty = torch.empty(0, 16, requires_grad=True)
    s_empty_ref = torch.empty(0, 16, requires_grad=True)
    t_empty = torch.empty(0, 16)
    out_empty = triton_ops.emulate_rowwise_kl_cpu(s_empty, t_empty)
    out_empty_ref = full_kl(s_empty_ref, t_empty)
    assert float(out_empty.detach()) == 0.0 and float(out_empty_ref.detach()) == 0.0
    g_empty = torch.autograd.grad(out_empty, s_empty, create_graph=True)[0]
    g_empty_ref = torch.autograd.grad(out_empty_ref, s_empty_ref, create_graph=True)[0]
    assert g_empty.shape == (0, 16) and g_empty_ref.shape == (0, 16)
    with pytest.raises(ValueError, match='vocab dimension must be positive'):
        triton_ops.emulate_rowwise_kl_cpu(torch.empty(2, 0), torch.empty(2, 0))

    # 2a. SignedHash.lookup preserves gradients when lookup_fn is attached and grad is enabled:
    calls = {'n': 0}

    def custom_lookup(d, prev, curr, vals, cap):
        calls['n'] += 1
        return triton_ops.emulate_signed_hash_kernel_cpu(d, prev, curr, vals.detach(), cap)

    sh = SignedHash(64, lookup_fn=custom_lookup)
    with torch.no_grad():
        sh.values.normal_(0.0, 0.2)
    d_q = torch.tensor([[1], [2]], dtype=torch.int64)
    p_q = torch.tensor([[3], [5]], dtype=torch.int64)
    c_q = torch.tensor([[2, 4, 6, 7]], dtype=torch.int64)

    out_train = sh.lookup(d_q, p_q, c_q)
    assert calls['n'] == 0
    out_train.sum().backward()
    assert sh.values.grad is not None and float(sh.values.grad.abs().sum()) > 0.0

    with torch.no_grad():
        out_eval = sh.lookup(d_q, p_q, c_q)
    assert calls['n'] == 1
    torch.testing.assert_close(out_eval, out_train.detach(), atol=1e-6, rtol=1e-6)

    snap = sh.snapshot()
    out_snap = snap.lookup(d_q, p_q, c_q)
    assert calls['n'] == 2
    torch.testing.assert_close(out_snap, out_train.detach(), atol=1e-6, rtol=1e-6)

    # 2b. Zero host sync (.item()) in triton_signed_hash_lookup + has_lazy=True parity gate:
    src_text = inspect.getsource(triton_ops.triton_signed_hash_lookup)
    assert '.item()' not in src_text
    lu = torch.randint(0, 4, (64,), dtype=torch.int32)
    ct = torch.tensor([0.0, 0.04, 0.11, 0.22], dtype=torch.float32)
    lazy_cpu = triton_ops.emulate_signed_hash_kernel_cpu(
        d_q, p_q, c_q, sh.values.detach(), 64, last_u=lu, cum_thresh=ct, u=3
    )
    b_idx, signs = hash_keys(d_q, p_q, c_q, 64, sh.values.device)
    lazy_ref = proximal_l1(sh.values.detach()[b_idx], ct[3] - ct[lu[b_idx].long()]) * signs
    torch.testing.assert_close(lazy_cpu, lazy_ref, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='provided together'):
        triton_ops.emulate_signed_hash_kernel_cpu(d_q, p_q, c_q, sh.values.detach(), 64, last_u=lu)
    with pytest.raises(ValueError, match='out of bounds'):
        triton_ops.emulate_signed_hash_kernel_cpu(
            d_q, p_q, c_q, sh.values.detach(), 64, last_u=lu, cum_thresh=ct, u=4
        )
    # Verify extra_parity_fn rejects lazy mismatch even when non-lazy parity passes:
    _, bad_lazy_rep = triton_ops.evaluate_kernel_gate(
        'signed_hash_lookup',
        lambda: out_eval,
        lambda: out_eval,
        device='cpu',
        strict=False,
        measured_times_ms=(0.1, 1.0),
        extra_parity_fn=lambda: (lazy_ref, lazy_ref + 1.0),
    )
    assert bad_lazy_rep.enabled is False
    assert bad_lazy_rep.status == 'rejected_parity_failure'


def test_m5_r2_checkpoint_scale_provenance_and_m3_m4_gate_enforcement(tmp_path):
    """Regression suite for M5-R2-BUG-3 (#78): scaled checkpoints cannot bypass M3/M4 or mix scales."""
    from unittest.mock import patch
    from edraft import eval_closed_loop, train
    from edraft.features import HiddenProjection

    torch.manual_seed(11)
    E = torch.nn.functional.normalize(torch.randn(16, 4), dim=-1)
    # Also verify scale_adapter_inplace works when grad is enabled on leaf nn.Parameters:
    ad_leaf = DenseAdapter(E, 6, 4, sparse=SignedHash(64), M_sparse=4)
    with torch.no_grad():
        ad_leaf.W.fill_(2.0)
        ad_leaf.sparse.values.fill_(1.0)
    train.scale_adapter_inplace(ad_leaf, 0.25)
    assert float(ad_leaf.W.detach()[0, 0]) == pytest.approx(0.5)
    assert float(ad_leaf.sparse.values.detach()[0]) == pytest.approx(0.25)

    proj = HiddenProjection(
        matrix=torch.randn(6, 8),
        mean=torch.zeros(6),
        std=torch.ones(6),
        metadata={'p': 6},
    )
    proj_path = tmp_path / 'h_proj.pt'
    proj.save(proj_path)
    proj_sha = eval_closed_loop.file_sha256(proj_path)
    env = json.loads((eval_closed_loop.ROOT / 'artifacts/environment.json').read_text())
    revs = [env['model_artifacts'][n]['revision'] for n in ('Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16')]

    m2_identity = 'gated_m2_fit_identity_r2'
    rep = {
        'model_pair': ['Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16'],
        'model_revisions': revs,
        'frozen_artifacts': {'hidden_projection_sha256': proj_sha},
    }
    (tmp_path / 'm2').mkdir(parents=True)
    for stage_label in ('dense', 'dense_sparse'):
        sp = SignedHash(64) if stage_label == 'dense_sparse' else None
        ad = DenseAdapter(E, 6, 4, sparse=sp, M_sparse=4)
        with torch.no_grad():
            ad.W.fill_(1.0)
            ad.A.fill_(0.5)
            ad.b.fill_(0.25)
            if sp is not None:
                ad.sparse.values.fill_(0.75)
        ckpt = {
            'schema_version': 1,
            'scale': 'reference',
            'identity': m2_identity,
            'stage': stage_label,
            'epoch': 2,
            'next_batch': 0,
            'adapter_state': {k: v.clone() for k, v in ad.state_dict().items()},
            'adapter_config': {'p': 6, 'r': 4, 'D': 4, 'M_sparse': 4, 'sparse_capacity': 64 if sp else None},
            'report': rep,
            'adapter_version': f'v_{stage_label}',
        }
        torch.save(ckpt, tmp_path / 'm2' / f'{stage_label}.pt')

    (tmp_path / 'm2.json').write_text(json.dumps({'status': 'passed', 'scale': 'reference', 'identity': m2_identity}))
    for prior in ('m3', 'm4'):
        (tmp_path / prior).mkdir(parents=True)
        (tmp_path / prior / 'summary.json').write_text(json.dumps({'status': 'complete', 'scale': 'reference'}))

    scaled_d_0p2 = tmp_path / 'dense_0p2.pt'
    scaled_s_0p2 = tmp_path / 'sparse_0p2.pt'
    scaled_s_0p8 = tmp_path / 'sparse_0p8.pt'
    scaled_d_re1 = tmp_path / 'dense_0p2_times_5.pt'
    scaled_s_re1 = tmp_path / 'sparse_0p2_times_5.pt'
    train.scale_adapter_checkpoint(tmp_path / 'm2/dense.pt', scaled_d_0p2, 0.2)
    train.scale_adapter_checkpoint(tmp_path / 'm2/dense_sparse.pt', scaled_s_0p2, 0.2)
    train.scale_adapter_checkpoint(tmp_path / 'm2/dense_sparse.pt', scaled_s_0p8, 0.8)
    train.scale_adapter_checkpoint(scaled_d_0p2, scaled_d_re1, 5.0)
    train.scale_adapter_checkpoint(scaled_s_0p2, scaled_s_re1, 5.0)

    class DummyBackend:
        block_size = 5
        stop_token_ids = [15]

        class tokenizer:
            chat_template = 'tpl'
            special_tokens_map = {}

            class backend_tokenizer:
                @staticmethod
                def to_str():
                    return '{}'

            @staticmethod
            def get_vocab():
                return {'a': 0}

        class target:
            dtype = torch.bfloat16

            class config:
                max_position_embeddings = 256

                @staticmethod
                def to_dict():
                    return {'v': 16}

            class generation_config:
                @staticmethod
                def to_dict():
                    return {}

    def make_args(stage, d_ckpt, s_ckpt, ckpt_scale=None):
        p = eval_closed_loop.common_parser('repro')
        p.add_argument('--stage', default=stage)
        p.add_argument('--dense_checkpoint', type=Path, default=d_ckpt)
        p.add_argument('--sparse_checkpoint', type=Path, default=s_ckpt)
        p.add_argument('--projection', type=Path, default=proj_path)
        p.add_argument('--checkpoint_scale', type=float, default=ckpt_scale)
        return eval_closed_loop.parse_config(p, ['--scale', 'reference', '--device', 'cpu', '--output', str(tmp_path)])

    with patch('edraft.download.fetch_model', return_value=tmp_path), \
         patch('edraft.backend.ReferenceBackend.from_pretrained', return_value=DummyBackend()):
        # 3a. M3 and M4 strictly reject scaled checkpoints (including 0.2 * 5.0 round-trip):
        for bad_stage in ('m3', 'm4'):
            with pytest.raises(ValueError, match='accepted M2 fit'):
                eval_closed_loop.EvaluationRunner(make_args(bad_stage, scaled_d_0p2, scaled_s_0p2))
            with pytest.raises(ValueError, match='accepted M2 fit'):
                eval_closed_loop.EvaluationRunner(make_args(bad_stage, scaled_d_re1, scaled_s_re1))

        # 3b. M5 rejects mismatched 0.2x dense + 0.8x dense_sparse:
        with pytest.raises(ValueError, match='same fit identity'):
            eval_closed_loop.EvaluationRunner(make_args('m5', scaled_d_0p2, scaled_s_0p8))

        # 3c. M5 accepts matched 0.2x pair and CLI --checkpoint_scale 0.2 with is_m5_custom_checkpoint == True:
        r_file = eval_closed_loop.EvaluationRunner(make_args('m5', scaled_d_0p2, scaled_s_0p2))
        assert r_file.is_m5_custom_checkpoint is True
        r_cli = eval_closed_loop.EvaluationRunner(
            make_args('m5', tmp_path / 'm2/dense.pt', tmp_path / 'm2/dense_sparse.pt', ckpt_scale=0.2)
        )
        assert r_cli.is_m5_custom_checkpoint is True
        assert float(r_cli.dense.W.detach()[0, 0]) == pytest.approx(0.2)

        # Unscaled gated M2 checkpoints (and explicit no-op --checkpoint_scale 1.0) remain False:
        r_m2 = eval_closed_loop.EvaluationRunner(
            make_args('m5', tmp_path / 'm2/dense.pt', tmp_path / 'm2/dense_sparse.pt', ckpt_scale=1.0)
        )
        assert r_m2.is_m5_custom_checkpoint is False



