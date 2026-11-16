"""Unit, differential-parity, adversarial-integrity, and benchmark tests for `_FullVocabLookaheadCache`."""
from __future__ import annotations

import time
import pytest
import torch
import torch.nn.functional as F

from edraft.adapter import (
    DenseAdapter,
    LookaheadConfig,
    _FullVocabLookaheadCache,
    damped_lookahead_step,
    initialize_lookahead,
    lookahead_energy,
    lookahead_fields,
)
from edraft.dflash_loop import PhaseTimer, _sample_proposal, generate_toy
from edraft.sampler import SamplingWarper
from edraft.sparse import ExactObservedPairs, LazySignedHash, SignedHash, SumSparse
from edraft.toy import ToyParallelDrafter, ToyTarget


def _build_fixture(
    *,
    V: int = 48,
    r_dim: int = 8,
    p_dim: int = 10,
    D: int = 4,
    M: int = 8,
    K: int = 6,
    sparse_kind: str = 'signed_hash',
    dtype: torch.dtype = torch.float32,
    seed: int = 101,
):
    gen = torch.Generator().manual_seed(seed)
    E = F.normalize(torch.randn((V, r_dim), generator=gen, dtype=torch.float32), dim=-1)
    if sparse_kind == 'none':
        sp = None
    elif sparse_kind == 'signed_hash':
        sp = SignedHash(512)
        with torch.no_grad():
            sp.values.normal_(0.0, 0.25, generator=gen)
    elif sparse_kind == 'exact_pairs':
        sp = ExactObservedPairs(256, vocab_size=V)
        sp.train()
        d_init = torch.arange(1, D + 1)[:, None]
        p_init = torch.randint(0, V, (D, 12), generator=gen)
        c_init = torch.randint(0, V, (D, 12), generator=gen)
        _ = sp.lookup(d_init, p_init, c_init)
        with torch.no_grad():
            sp.values.normal_(0.0, 0.25, generator=gen)
        sp.eval()
    elif sparse_kind == 'sum_sparse':
        sh = SignedHash(512)
        with torch.no_grad():
            sh.values.normal_(0.0, 0.2, generator=gen)
        lazy = LazySignedHash(256)
        lazy.step(torch.tensor([3, 17, 42, 99], dtype=torch.int64), torch.tensor([0.5, -0.4, 0.3, -0.2]), lr=0.5, lambda_l1=0.02)
        sp = sh
    else:
        raise ValueError(f'unknown sparse_kind: {sparse_kind}')

    adapter = DenseAdapter(E, p=p_dim, D=D, sparse=sp, M_sparse=M)
    with torch.no_grad():
        adapter.W.normal_(0.0, 0.12, generator=gen)
        adapter.A.normal_(0.0, 0.12, generator=gen)
        adapter.b.normal_(0.0, 0.05, generator=gen)
    snap = adapter.snapshot(version=3)
    if sparse_kind == 'sum_sparse':
        snap = type(snap)(
            E=snap.E, W=snap.W, A=snap.A, b=snap.b, version=snap.version,
            sparse=SumSparse((snap.sparse, lazy.snapshot())), M_sparse=snap.M_sparse,
            C0=snap.C0, is_zero=False, unary_fn=snap.unary_fn,
        )
    if dtype == torch.float64:
        snap = type(snap)(
            E=snap.E.double(), W=snap.W.double(), A=snap.A.double(), b=snap.b.double(),
            version=snap.version, sparse=snap.sparse, M_sparse=snap.M_sparse,
            C0=snap.C0, is_zero=False, unary_fn=snap.unary_fn,
        )
    l_unit = torch.randn((K, V), generator=gen, dtype=torch.float64 if dtype == torch.float64 else torch.float32).log_softmax(-1)
    H = torch.randn((K, p_dim), generator=gen, dtype=torch.float64 if dtype == torch.float64 else torch.float32)
    if dtype == torch.bfloat16:
        l_unit = l_unit.bfloat16()
        H = H.bfloat16()
    return snap, l_unit, H


def test_prompt_test4_and_test10_full_vocab_fastpath_exactness():
    """Verify Test #4 (greedy full-V vs C0 domain, immutable C0) and Test #10 (shared energy partial derivatives)."""
    # Test #4 (`prompt.md:902`): zero-weight 0-ULP + nonzero-weight determinism & outside-C0 greedy selection:
    adapter = DenseAdapter(torch.eye(5), 1, D=1, M_sparse=2)
    C0 = [torch.tensor([3, 2]), torch.tensor([1, 4])]
    snap = adapter.snapshot(C0=C0)
    l_unit = torch.tensor([[10.0, 0.0, 2.0, 2.0, -1.0], [9.0, 4.0, 1.0, 4.0, 9.0]])
    orig_c0 = [c.clone() for c in snap.C0]

    st_fast = initialize_lookahead(snap, l_unit, torch.zeros(2, 1), [], domain='full_vocabulary', T_relax=0.8, use_fast_lookahead=True)
    st_ref = initialize_lookahead(snap, l_unit, torch.zeros(2, 1), [], domain='full_vocabulary', T_relax=0.8, use_fast_lookahead=False)
    assert isinstance(st_fast._fast, _FullVocabLookaheadCache)
    assert st_ref._fast is None

    damped_lookahead_step(st_fast, eta=0.5)
    damped_lookahead_step(st_ref, eta=0.5)
    assert torch.equal(st_fast.g, st_ref.g)

    y0_fast, p0_fast = st_fast.sample_slot(0, 0.0)
    y0_ref, p0_ref = st_ref.sample_slot(0, 0.0)
    assert p0_fast is None and p0_ref is None
    assert y0_fast == y0_ref == 0  # outside C0[0] = [3, 2]!
    assert st_fast.clamped[0] == 0 and st_fast.support[0].tolist() == [0]

    damped_lookahead_step(st_fast, eta=0.5)
    damped_lookahead_step(st_ref, eta=0.5)
    y1_fast, _ = st_fast.sample_slot(1, 0.0)
    y1_ref, _ = st_ref.sample_slot(1, 0.0)
    assert y1_fast == y1_ref == 0  # lowest-ID tie break between token 0 and token 4 (both 9.0)
    assert all(torch.equal(a, b) for a, b in zip(orig_c0, snap.C0))

    # Test #10 (`prompt.md:908`): FP64 shared-energy partial derivatives before and after clamping outside C0:
    snap64, l64, H64 = _build_fixture(V=24, r_dim=6, p_dim=8, D=3, M=5, K=5, sparse_kind='exact_pairs', dtype=torch.float64, seed=202)
    st64 = initialize_lookahead(snap64, l64, H64, [7, 11], domain='full_vocabulary', T_relax=0.9, beta_relax=0.8, use_fast_lookahead=True)
    assert isinstance(st64._fast, _FullVocabLookaheadCache)
    damped_lookahead_step(st64, eta=0.5)
    # Clamp slot 1 to a token guaranteed outside C0[1]:
    c0_1 = set(st64.snapshot.C0[1].tolist())
    outside_tok = next(t for t in range(24) if t not in c0_1)
    st64.clamp(1, outside_tok)
    damped_lookahead_step(st64, eta=0.5)
    assert isinstance(st64._fast, _FullVocabLookaheadCache)

    # Verify raw support mass on unclamped slots is strictly < 1.0 (never renormalized over C0):
    for slot_i in (0, 2, 3, 4):
        raw_sum = float(st64.raw_support_mass(slot_i).sum().item())
        assert 0.0 < raw_sum < 1.0

    fwd64, bwd64 = lookahead_fields(st64)
    expected_fields = st64.U_all + fwd64 + bwd64
    r_vars = [r.detach().clone().requires_grad_(True) for r in st64.r]
    st64.r = r_vars
    st64.mu = [r @ st64.snapshot.E for r in st64.r]
    grads = torch.autograd.grad(lookahead_energy(st64), st64.r)
    for slot_i in (0, 2, 3, 4):
        torch.testing.assert_close(grads[slot_i], expected_fields[slot_i], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('sparse_kind', ['none', 'signed_hash', 'exact_pairs', 'sum_sparse'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float64])
def test_full_vocab_fastpath_sequential_and_nonprefix_clamp_parity(sparse_kind, dtype):
    """Verify step-by-step parity of `_FullVocabLookaheadCache` vs uncached eager across dtypes & sparse backends."""
    if dtype == torch.float64 and sparse_kind == 'sum_sparse':
        pytest.skip('covered by float32/bfloat16 sum_sparse and float64 exact_pairs/signed_hash')
    snap, l_unit, H = _build_fixture(V=40, r_dim=8, p_dim=10, D=4, M=6, K=6, sparse_kind=sparse_kind, dtype=dtype, seed=303)
    prefix = [12, 5, 31]
    atol = 1e-12 if dtype == torch.float64 else 1e-5
    rtol = 1e-12 if dtype == torch.float64 else 1e-5

    # Case A: Sequential prefix clamping (alternating tokens inside C0[i] and outside C0[i]):
    st_f = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=0.75, beta_relax=0.85, use_fast_lookahead=True)
    st_r = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=0.75, beta_relax=0.85, use_fast_lookahead=False)
    assert isinstance(st_f._fast, _FullVocabLookaheadCache)

    for _ in range(2):
        damped_lookahead_step(st_f, eta=0.5)
        damped_lookahead_step(st_r, eta=0.5)

    for i in range(6):
        torch.testing.assert_close(st_f.full_slot_scores(i, beta=0.7), st_r.full_slot_scores(i, beta=0.7), atol=atol, rtol=rtol)
        c0_set = set(st_r.snapshot.C0[i].tolist())
        chosen_tok = int(st_r.snapshot.C0[i][0].item()) if i % 2 == 0 else next(t for t in range(40) if t not in c0_set)
        st_f.clamp(i, chosen_tok)
        st_r.clamp(i, chosen_tok)
        if i + 1 < 6:
            damped_lookahead_step(st_f, eta=0.5)
            damped_lookahead_step(st_r, eta=0.5)
            assert isinstance(st_f._fast, _FullVocabLookaheadCache)

    torch.testing.assert_close(st_f.g, st_r.g, atol=atol, rtol=rtol)
    for rf, rr in zip(st_f.r, st_r.r):
        torch.testing.assert_close(rf, rr, atol=atol, rtol=rtol)
    for mf, mr in zip(st_f.mu, st_r.mu):
        torch.testing.assert_close(mf, mr, atol=atol, rtol=rtol)

    # Case B: Out-of-order non-prefix clamping (middle slots 3 and 1 clamped before slot 0):
    st_f2 = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=1.1, beta_relax=0.6, use_fast_lookahead=True)
    st_r2 = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=1.1, beta_relax=0.6, use_fast_lookahead=False)
    damped_lookahead_step(st_f2, eta=0.4)
    damped_lookahead_step(st_r2, eta=0.4)
    for slot_k in (3, 1, 4):
        c0_set = set(st_r2.snapshot.C0[slot_k].tolist())
        tok_k = next(t for t in range(40) if t not in c0_set) if slot_k == 3 else int(st_r2.snapshot.C0[slot_k][-1].item())
        st_f2.clamp(slot_k, tok_k)
        st_r2.clamp(slot_k, tok_k)
        damped_lookahead_step(st_f2, eta=0.6)
        damped_lookahead_step(st_r2, eta=0.6)
        assert isinstance(st_f2._fast, _FullVocabLookaheadCache)
    torch.testing.assert_close(st_f2.g, st_r2.g, atol=atol, rtol=rtol)
    for i in (0, 2, 5):
        torch.testing.assert_close(st_f2.r[i], st_r2.r[i], atol=atol, rtol=rtol)
        torch.testing.assert_close(st_f2.mu[i], st_r2.mu[i], atol=atol, rtol=rtol)


def test_full_vocab_fastpath_alias_immutability_and_adversarial_mutation_fallback():
    """Ensure saved `state.g` references never mutate across steps and in-place mutations fall back cleanly."""
    snap, l_unit, H = _build_fixture(V=32, r_dim=8, p_dim=8, D=3, M=6, K=5, sparse_kind='signed_hash', seed=404)
    st_f = initialize_lookahead(snap, l_unit, H, [2, 9], domain='full_vocabulary', use_fast_lookahead=True)
    st_r = initialize_lookahead(snap, l_unit, H, [2, 9], domain='full_vocabulary', use_fast_lookahead=False)

    # 1. Holding `g0 = state.g` before a step must keep `g0` completely unchanged after `damped_lookahead_step`:
    g0_f = st_f.g
    g0_copy = g0_f.clone()
    damped_lookahead_step(st_f, eta=0.5)
    damped_lookahead_step(st_r, eta=0.5)
    assert torch.equal(g0_f, g0_copy), 'damped_lookahead_step mutated an already-exposed state.g tensor in place!'
    assert not torch.equal(st_f.g, g0_f)
    torch.testing.assert_close(st_f.g, st_r.g, atol=1e-6, rtol=1e-6)

    # 2. In-place mutation of `state.g[0, 0]` must be detected by `_check_fast_integrity` and preserved:
    st_f.g[0, 0] += 3.5
    st_r.g[0, 0] += 3.5
    damped_lookahead_step(st_f, eta=0.5)
    damped_lookahead_step(st_r, eta=0.5)
    assert st_f._fast is None
    torch.testing.assert_close(st_f.g, st_r.g, atol=1e-6, rtol=1e-6)

    # 3. In-place mutation of `state.support[0]` or `state.r[0]` also falls back with exact parity:
    st_f3 = initialize_lookahead(snap, l_unit, H, [2, 9], domain='full_vocabulary', use_fast_lookahead=True)
    st_r3 = initialize_lookahead(snap, l_unit, H, [2, 9], domain='full_vocabulary', use_fast_lookahead=False)
    st_f3.support[0][0] = (st_f3.support[0][0] + 1) % 32
    st_r3.support[0][0] = st_f3.support[0][0]
    damped_lookahead_step(st_f3, eta=0.5)
    damped_lookahead_step(st_r3, eta=0.5)
    assert st_f3._fast is None
    torch.testing.assert_close(st_f3.g, st_r3.g, atol=1e-6, rtol=1e-6)

    # 4. Ragged C0 cleanly bypasses fast cache without error:
    ragged_snap = snap.with_candidates([torch.tensor([0, 1]), torch.tensor([2, 3, 4]), torch.tensor([1]), torch.tensor([0, 2]), torch.tensor([3, 5])])
    st_rag = initialize_lookahead(ragged_snap, l_unit, H, [2], domain='full_vocabulary', use_fast_lookahead=True)
    assert st_rag._fast is None
    damped_lookahead_step(st_rag, eta=0.5)
    assert st_rag.g.shape == (5, 32)


@pytest.mark.parametrize('mode', ['static', 'rerelax'])
@pytest.mark.parametrize('temperature', [0.0, 0.7])
def test_closed_loop_dflash_full_vocab_lookahead_parity(mode, temperature):
    """Verify `_sample_proposal` and `generate_toy` with `domain='full_vocabulary'` match uncached reference."""
    snap, l_unit, H = _build_fixture(V=64, r_dim=8, p_dim=10, D=4, M=8, K=6, sparse_kind='signed_hash', seed=505)
    draft_logits = l_unit.clone()
    warper = SamplingWarper(top_k=12, top_p=0.85) if temperature > 0 else SamplingWarper()
    la_cfg = LookaheadConfig(mode=mode, domain='full_vocabulary', R_init=2, R_clamp=1, eta=0.5, T_relax=0.9)
    timer = PhaseTimer('cpu', enabled=True)

    toks_fast, q_fast, _, _ = _sample_proposal(
        draft_logits, H, [4, 9, 15],
        temperature=temperature, warper=warper,
        generator=torch.Generator().manual_seed(999),
        snapshot=snap, projection=None, beta=0.8, bigram=None, timer=timer,
        lookahead=la_cfg, enable_round3_compact=True,
    )
    toks_ref, q_ref, _, _ = _sample_proposal(
        draft_logits, H, [4, 9, 15],
        temperature=temperature, warper=warper,
        generator=torch.Generator().manual_seed(999),
        snapshot=snap, projection=None, beta=0.8, bigram=None, timer=timer,
        lookahead=la_cfg, enable_round3_compact=False,
    )
    assert toks_fast == toks_ref
    if temperature == 0.0:
        assert q_fast is None and q_ref is None
    else:
        torch.testing.assert_close(q_fast, q_ref, atol=1e-6, rtol=1e-6)

    # Also verify closed-loop `generate_toy` completes cleanly with `lookahead_iterations` recorded:
    target = ToyTarget(vocab_size=12, hidden_size=8, seed=17)
    drafter = ToyParallelDrafter(target, seed=18)
    ad_toy = DenseAdapter(F.normalize(torch.randn(12, 4), dim=-1), p=8, D=2, sparse=SignedHash(64), M_sparse=4)
    with torch.no_grad():
        ad_toy.W.normal_(0.0, 0.1)
        ad_toy.A.normal_(0.0, 0.1)
        ad_toy.sparse.values.normal_(0.0, 0.1)
    res = generate_toy(
        target, drafter, [1, 3, 5], max_new_tokens=10, block_length=4,
        temperature=temperature, warper=warper,
        generator=torch.Generator().manual_seed(42),
        adapter=ad_toy, beta=0.6, lookahead=la_cfg, measure_phases=True,
    )
    assert res.num_output_tokens == 10
    assert res.phase_ms.get('lookahead_iterations', 0.0) > 0.0


def test_full_vocab_lookahead_qwen3_scale_benchmark_and_parity():
    """Benchmark `full_vocabulary` `static` and `rerelax` on realistic `[K=15, V=151936, r=64, D=4, M=256]`."""
    V, r_dim, p_dim, D, M, K = 151_936, 64, 128, 4, 256, 15
    gen = torch.Generator().manual_seed(2026)
    E = F.normalize(torch.randn((V, r_dim), generator=gen, dtype=torch.float32), dim=-1)
    sp = SignedHash(2**16)
    with torch.no_grad():
        sp.values.normal_(0.0, 0.1, generator=gen)
    adapter = DenseAdapter(E, p=p_dim, D=D, sparse=sp, M_sparse=M)
    with torch.no_grad():
        adapter.W.normal_(0.0, 0.05, generator=gen)
        adapter.A.normal_(0.0, 0.05, generator=gen)
        adapter.b.normal_(0.0, 0.02, generator=gen)
    snap = adapter.snapshot(version=1)
    draft_logits = torch.randn((K, V), generator=gen, dtype=torch.bfloat16)
    H = torch.randn((K, p_dim), generator=gen, dtype=torch.float32)
    prefix = [101, 202, 303, 404]
    warper = SamplingWarper(top_k=20, top_p=0.8)

    for mode in ('static', 'rerelax'):
        la_cfg = LookaheadConfig(mode=mode, domain='full_vocabulary', R_init=2, R_clamp=1, eta=0.5, T_relax=1.0)
        timer = PhaseTimer('cpu', enabled=False)

        # Warmup fastpath once:
        _ = _sample_proposal(
            draft_logits, H, prefix, temperature=0.7, warper=warper,
            generator=torch.Generator().manual_seed(1234),
            snapshot=snap, projection=None, beta=0.5, bigram=None, timer=timer,
            lookahead=la_cfg, enable_round3_compact=True,
        )

        t0 = time.perf_counter()
        toks_fast, q_fast, _, _ = _sample_proposal(
            draft_logits, H, prefix, temperature=0.7, warper=warper,
            generator=torch.Generator().manual_seed(5678),
            snapshot=snap, projection=None, beta=0.5, bigram=None, timer=timer,
            lookahead=la_cfg, enable_round3_compact=True,
        )
        fast_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        toks_ref, q_ref, _, _ = _sample_proposal(
            draft_logits, H, prefix, temperature=0.7, warper=warper,
            generator=torch.Generator().manual_seed(5678),
            snapshot=snap, projection=None, beta=0.5, bigram=None, timer=timer,
            lookahead=la_cfg, enable_round3_compact=False,
        )
        eager_ms = (time.perf_counter() - t1) * 1000.0

        max_q_diff = float((q_fast - q_ref).abs().max().item())
        assert toks_fast == toks_ref, f'{mode}: token mismatch {toks_fast} vs {toks_ref}'
        assert max_q_diff <= 1e-5, f'{mode}: max_q_diff={max_q_diff}'
        assert fast_ms < eager_ms, f'{mode}: expected speedup, got fast={fast_ms:.1f}ms vs eager={eager_ms:.1f}ms'


def test_full_vocab_zero_host_sync_clamp_and_all_clamped_alias_detach(monkeypatch):
    """Verify #228 Section 3: zero .item()/.tolist()/.nonzero() syncs in hot loop & all-clamped g_exposed detach."""
    snap, l_unit, H = _build_fixture(V=48, r_dim=8, p_dim=10, D=4, M=6, K=5, sparse_kind='signed_hash', seed=707)
    prefix = [3, 14, 25]

    # Reference uncached run with mixed inside-C0 and outside-C0 clamps:
    st_ref = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=0.85, beta_relax=0.75, use_fast_lookahead=False)
    damped_lookahead_step(st_ref, eta=0.5)
    chosen = []
    for i in range(5):
        c0_set = set(st_ref.snapshot.C0[i].tolist())
        tok = int(st_ref.snapshot.C0[i][0].item()) if i % 2 == 0 else next(t for t in range(48) if t not in c0_set)
        chosen.append(tok)
        st_ref.clamp(i, tok)
        if i + 1 < 5:
            damped_lookahead_step(st_ref, eta=0.5)

    # Warm neighbor/pair index caches once, then ban .item(), .tolist(), and .nonzero() completely:
    st_warm = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=0.85, beta_relax=0.75, use_fast_lookahead=True)
    for i in range(5):
        st_warm.clamp(i, chosen[i])

    def _forbid_sync(*_args, **_kwargs):
        raise RuntimeError('D2H host sync (.item/.tolist/.nonzero) forbidden on full_vocabulary fastpath!')

    monkeypatch.setattr(torch.Tensor, 'item', _forbid_sync)
    monkeypatch.setattr(torch.Tensor, 'tolist', _forbid_sync)
    monkeypatch.setattr(torch.Tensor, 'nonzero', _forbid_sync)

    # 1a. Python-int clamp(i, tok) with zero D2H syncs:
    st_int = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=0.85, beta_relax=0.75, use_fast_lookahead=True)
    damped_lookahead_step(st_int, eta=0.5)
    for i in range(5):
        _ = st_int.full_slot_scores(i, beta=0.6)
        st_int.clamp(i, chosen[i])
        if i + 1 < 5:
            damped_lookahead_step(st_int, eta=0.5)

    # 1b. Pure tensor _clamp_full_vocab_tensor(i, tok_t) with zero D2H or H2D syncs:
    chosen_gpu = [torch.tensor(t, dtype=torch.long) for t in chosen]
    st_ten = initialize_lookahead(snap, l_unit, H, prefix, domain='full_vocabulary', T_relax=0.85, beta_relax=0.75, use_fast_lookahead=True)
    damped_lookahead_step(st_ten, eta=0.5)
    for i in range(5):
        _ = st_ten.full_slot_scores(i, beta=0.6)
        st_ten._clamp_full_vocab_tensor(i, chosen_gpu[i])
        if i + 1 < 5:
            damped_lookahead_step(st_ten, eta=0.5)

    # 2. All-clamped (`n_clamped == K`) g_exposed detach check (#228 Section 3 Item 2):
    g_before_all_clamped = st_int.g
    damped_lookahead_step(st_int, eta=0.5)
    g_after_all_clamped = st_int.g
    assert g_after_all_clamped is not g_before_all_clamped
    assert torch.equal(g_after_all_clamped, g_before_all_clamped)

    monkeypatch.undo()

    # Verify both st_int and st_ten materialize exact g, r, mu, support, and clamped matching st_ref:
    for st_cand in (st_int, st_ten):
        assert st_cand.clamped == chosen
        assert [s.tolist() for s in st_cand.support] == [[t] for t in chosen]
        torch.testing.assert_close(st_cand.g, st_ref.g, atol=1e-5, rtol=1e-5)
        for rc, rr in zip(st_cand.r, st_ref.r):
            torch.testing.assert_close(rc, rr, atol=1e-5, rtol=1e-5)
        for mc, mr in zip(st_cand.mu, st_ref.mu):
            torch.testing.assert_close(mc, mr, atol=1e-5, rtol=1e-5)

