"""End-to-end post-M4 branch integration tests (`opt/pytorch-fastpaths` + `feat/m5-and-full-scale`)."""
from __future__ import annotations

import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import socket
import subprocess
import sys
import torch
import pytest

from edraft.adapter import DenseAdapter, LookaheadConfig
from edraft.dflash_loop import _sample_proposal, generate_toy, PhaseTimer
from edraft.diagnose import OUTPUT_LABELS
from edraft.eval_closed_loop import (
    IMPLEMENTATION_FILES as EVAL_IMPL_FILES,
    implementation_hash,
    main as eval_main,
)
from edraft.features import fit_hidden_projection, fit_token_features
from edraft import launch, post_m4_suite, train
from edraft.m5_baselines import (
    AdaptiveBlockController,
    SharpDraftProposal,
    TTSState,
    generate_token_recycling_toy,
)
from edraft.online import OnlineBigram, OnlineState
from edraft.runtime import ROOT, WorkQueue
from edraft.sampler import SamplingWarper, probabilities
from edraft.sparse import (
    ExactObservedPairs,
    LazyExactObservedPairs,
    LazySignedHash,
    SignedHash,
    SumSparse,
)
from edraft.toy import ToyParallelDrafter, ToyTarget
from edraft.verify import CompactProposal


def _load_validator_module():
    val_dir = ROOT / 'artifacts' / 'm4_evidence_validator'
    if str(val_dir) not in sys.path:
        sys.path.insert(0, str(val_dir))
    spec = importlib.util.spec_from_file_location('m4_validate_mod', val_dir / 'validate.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. OnlineBigram count-confidence gating + Round-3 Fastpath 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('k_ret', [None, 4])
def test_online_bigram_count_gate_kappa_exact_parity_all_paths(k_ret):
    V = 37
    gen = torch.Generator().manual_seed(101)
    l_unit = torch.log_softmax(torch.randn(6, V, generator=gen), dim=-1)

    bg = OnlineBigram(vocab_size=V, alpha=0.5, lambda_tr=0.2, k_ret=k_ret, count_gate_kappa=12.0)
    bg.update([3], [5, 7, 5, 9, 11, 5, 2, 7, 14])
    bg.update([5], [1, 1, 2, 3, 4, 6, 8, 10, 12, 13, 15, 16, 1, 2, 3])
    bg.update([20], [21])

    # Check effective_lambda formula lambda_tr * N / (N + kappa)
    n_3 = sum(bg.counts[3].values())
    assert math.isclose(bg.effective_lambda(3), 0.2 * (n_3 / (n_3 + 12.0)), rel_tol=1e-12)
    assert bg.effective_lambda(19) == 0.0
    assert bg.effective_lambda(None) == 0.0

    for row_idx, prev in enumerate([3, 5, 20, 19, 7, 0]):
        s_py = bg.score(l_unit[row_idx], prev)
        s_ten = bg.score_tensor(l_unit[row_idx], torch.tensor(prev, dtype=torch.long))
        assert torch.equal(s_py, s_ten), f'0-ULP mismatch at prev={prev}'

        # Greedy (temperature=0) compact merge parity
        base_v0, base_i0 = torch.max(l_unit[row_idx], dim=-1, keepdim=True)
        win_sc, win_id = bg.merge_compact_candidates(
            l_unit[row_idx], base_v0, base_i0, torch.tensor(prev, dtype=torch.long),
            temperature=0.0, k_eff=1,
        )
        assert int(win_id.item()) == int(torch.argmax(s_py).item())
        assert torch.allclose(win_sc, s_py[win_id], atol=0.0, rtol=0.0)

        # Stochastic top_k=8, top_p=0.85 compact merge parity
        warper = SamplingWarper(top_k=8, top_p=0.85)
        temperature = 0.7
        _, base_ik = torch.topk(l_unit[row_idx] / temperature, 8, dim=-1)
        base_vk = l_unit[row_idx].gather(-1, base_ik)
        sel_sc, sel_ids = bg.merge_compact_candidates(
            l_unit[row_idx], base_vk, base_ik, torch.tensor(prev, dtype=torch.long),
            temperature=temperature, k_eff=8,
        )
        q_full = probabilities(s_py, temperature, warper)
        sub_w = SamplingWarper(top_k=0, top_p=warper.top_p, min_p=warper.min_p)
        p_sub = probabilities(sel_sc, 1.0, sub_w)
        q_compact = torch.zeros_like(q_full).scatter(0, sel_ids, p_sub)
        assert torch.allclose(q_full, q_compact, atol=1e-7, rtol=1e-6)


def test_online_bigram_hyper_key_invalidation_and_validation():
    V = 19
    l_unit = torch.log_softmax(torch.randn(V, generator=torch.Generator().manual_seed(7)), dim=-1)
    bg = OnlineBigram(vocab_size=V, alpha=1.0, lambda_tr=0.2, k_ret=4, count_gate_kappa=12.0)
    bg.update([2], [4, 4, 5, 6])
    prev_t = torch.tensor(2, dtype=torch.long)

    s_k12 = bg.score_tensor(l_unit, prev_t)
    assert torch.equal(s_k12, bg.score(l_unit, 2))

    # Mutating count_gate_kappa must invalidate _DeviceBigramState via hyper_key
    bg.count_gate_kappa = 2.0
    s_k2 = bg.score_tensor(l_unit, prev_t)
    assert not torch.equal(s_k12, s_k2)
    assert torch.equal(s_k2, bg.score(l_unit, 2))

    bg.count_gate_kappa = None
    s_none = bg.score_tensor(l_unit, prev_t)
    assert torch.equal(s_none, bg.score(l_unit, 2))

    bg.count_gate_kappa = -3.0
    with pytest.raises(ValueError, match='count_gate_kappa'):
        bg.score(l_unit, 2)
    with pytest.raises(ValueError, match='count_gate_kappa'):
        bg.score_tensor(l_unit, prev_t)


# ---------------------------------------------------------------------------
# 2. Closed-loop Fastpaths 2 & 3 across online/M5 modes & sparse backends
# ---------------------------------------------------------------------------

def _toy_env(seed: int = 2026):
    target = ToyTarget(vocab_size=31, hidden_size=16, seed=seed)
    drafter = ToyParallelDrafter(target, seed=seed + 1)
    cal = [[i % 31, (i + 3) % 31, (i * 5 + 1) % 31] for i in range(16)]
    proj = fit_hidden_projection(
        [drafter.draft(p, 4).h_raw for p in cal],
        target.hidden_size, p=6, seed=seed,
        prompt_ids=[f'p-{i}' for i in range(16)],
    )
    E, _ = fit_token_features(
        target.embedding, r=6, seed=seed,
        calibration_token_ids=torch.tensor(sum(cal, [])),
    )
    return target, drafter, proj, E


@pytest.mark.parametrize('mode_name', [
    'online_bigram_count_gated',
    'online_dbS_zero_init',
    'zero_offline_online_only',
    'offline_online_stride5',
    'online_buffered_stride5',
    'online_rejection_relative',
    'online_cutoff',
    'exact_observed_pairs_online',
])
def test_closed_loop_round3_compact_vs_eager_parity_all_modes(mode_name):
    target, drafter, proj, E = _toy_env(seed=314)
    V = target.vocab_size
    prefix = [2, 7, 11, 5]
    warper = SamplingWarper(top_k=6, top_p=0.85)
    temperature = 0.7

    def _build_arm():
        if mode_name == 'online_bigram_count_gated':
            bg = OnlineBigram(V, alpha=0.5, lambda_tr=0.15, k_ret=6, count_gate_kappa=12.0)
            return None, None, bg
        if mode_name in ('online_dbS_zero_init', 'zero_offline_online_only'):
            offline = DenseAdapter(E, p=6, D=4, M_sparse=8)
            st = OnlineState(
                offline, lr=0.05, update_stride=1, lambda_l1_b=1e-6, lambda_l1_s=1e-6,
                sparse_capacity=128, enable_sparse=True,
                bias_sparse_only=(mode_name == 'online_dbS_zero_init'),
            )
            return None, st, None
        offline_sp = (
            ExactObservedPairs(capacity=128, vocab_size=V)
            if mode_name == 'exact_observed_pairs_online'
            else SignedHash(128)
        )
        offline = DenseAdapter(E, p=6, D=4, sparse=offline_sp, M_sparse=8)
        gen = torch.Generator().manual_seed(99)
        with torch.no_grad():
            offline.W.copy_(torch.randn_like(offline.W, generator=gen) * 0.02)
            offline.A.copy_(torch.randn_like(offline.A, generator=gen) * 0.02)
            offline.b.copy_(torch.randn_like(offline.b, generator=gen) * 0.01)
            if isinstance(offline_sp, SignedHash):
                offline_sp.values.copy_(torch.randn_like(offline_sp.values, generator=gen) * 0.02)
            else:
                offline_sp.train()
                _ = offline_sp.lookup([1, 2], [7, 11], [3, 4])
                offline_sp.values.data[:2] = torch.tensor([0.05, -0.04])
                offline_sp.eval()
        kwargs = dict(lr=0.03, lambda_l2=0.01, lambda_l1_b=1e-6, lambda_l1_s=1e-6,
                      sparse_capacity=128, enable_sparse=True)
        if mode_name == 'offline_online_stride5':
            kwargs['update_stride'] = 5
        elif mode_name == 'online_buffered_stride5':
            kwargs['update_stride'] = 5
            kwargs['accumulate_stride'] = True
        elif mode_name == 'online_rejection_relative':
            kwargs['weighting_mode'] = 'rejection_relative'
        elif mode_name == 'online_cutoff':
            kwargs['weighting_mode'] = 'cutoff'
        elif mode_name == 'exact_observed_pairs_online':
            kwargs['sparse_backend'] = 'exact_observed_pairs'
        st = OnlineState(offline, **kwargs)
        return None, st, None

    _, st_r3, bg_r3 = _build_arm()
    res_r3 = generate_toy(
        target, drafter, prefix, 48, temperature=temperature, warper=warper,
        generator=torch.Generator().manual_seed(777),
        online_state=st_r3, bigram=bg_r3,
        projection=proj if st_r3 is not None else None,
        enable_round3_compact=True,
    )

    _, st_eg, bg_eg = _build_arm()
    res_eg = generate_toy(
        target, drafter, prefix, 48, temperature=temperature, warper=warper,
        generator=torch.Generator().manual_seed(777),
        online_state=st_eg, bigram=bg_eg,
        projection=proj if st_eg is not None else None,
        enable_round3_compact=False,
    )

    assert res_r3.output_tokens == res_eg.output_tokens
    assert len(res_r3.blocks) == len(res_eg.blocks)
    for b1, b2 in zip(res_r3.blocks, res_eg.blocks):
        assert b1['draft_tokens'] == b2['draft_tokens']
        assert b1['accepted_length'] == b2['accepted_length']
        assert b1['adapter_version'] == b2['adapter_version']
        assert b1['next_adapter_version'] == b2['next_adapter_version']
    if st_r3 is not None:
        assert st_r3.u > 0
        assert torch.allclose(st_r3.db, st_eg.db, atol=1e-6, rtol=1e-5)


def test_compact_proposal_activated_and_sparse_2d_broadcast_contract():
    target, drafter, proj, E = _toy_env(seed=512)
    V = target.vocab_size

    # Verify 2D/broadcast lookup contract `[avail, 1] x [avail, 1] x [1, M]` across all sparse types
    offs = torch.tensor([[1], [2], [3]], dtype=torch.long)
    prevs = torch.tensor([[4], [9], [12]], dtype=torch.long)
    cands = torch.tensor([[0, 5, 10, 15, 20]], dtype=torch.long)

    sh = SignedHash(64)
    lsh = LazySignedHash(64)
    lsh.step(torch.tensor([1, 5]), torch.tensor([0.2, -0.3]), lr=0.1, lambda_l1=0.0)
    eop = ExactObservedPairs(capacity=32, vocab_size=V)
    eop.train()
    _ = eop.lookup(offs, prevs, cands)
    eop.values.data[:4] = torch.tensor([0.1, -0.2, 0.3, -0.4])
    eop.eval()
    leop = LazyExactObservedPairs(capacity=32, vocab_size=V)
    leop.step(torch.tensor([1 * V * V + 4 * V + 5]), torch.tensor([0.5]), lr=0.1, lambda_l1=0.0)

    for sp in [sh, sh.snapshot(), lsh, lsh.snapshot(), eop, eop.snapshot(), leop, leop.snapshot(),
               SumSparse((eop.snapshot(), leop.snapshot()))]:
        out = sp.lookup(offs, prevs, cands)
        assert out.shape == (3, 5)
        assert out.dtype == torch.float32

    # Verify _sample_proposal emits CompactProposal when _allow_compact_q=True & top_k > 0
    adapter = DenseAdapter(E, p=6, D=4, sparse=eop, M_sparse=6)
    with torch.no_grad():
        adapter.b.add_(0.05)
    snap = adapter.snapshot()
    assert snap.pair_is_zero()
    drafted = drafter.draft([1, 2, 3, 4], 4)
    toks, q, _, _ = _sample_proposal(
        drafted.logits, drafted.h_raw, [1, 2, 3, 4],
        temperature=0.7, warper=SamplingWarper(top_k=5, top_p=0.9),
        generator=torch.Generator().manual_seed(11),
        snapshot=snap, projection=proj, beta=1.0, bigram=None,
        timer=PhaseTimer('cpu', False), _allow_compact_q=True,
    )
    assert isinstance(q, CompactProposal)
    assert len(toks) == 4


def test_m5_invariants_triton_isolation_and_autograd_guard(tmp_path):
    # 1. SignedHash autograd guard bypasses custom lookup_fn when grad_enabled and values.requires_grad
    called = {'count': 0}

    def fake_triton_lookup(values, d, prev, curr, cap):
        called['count'] += 1
        return torch.zeros((1, 3), dtype=torch.float32)

    sh = SignedHash(32)
    sh.lookup_fn = fake_triton_lookup
    with torch.no_grad():
        _ = sh.lookup([[1]], [[2]], [[3, 4, 5]])
    assert called['count'] == 1

    # With grad enabled on nn.Parameter values, lookup_fn MUST be bypassed so autograd flows
    loss = sh.lookup([[1]], [[2]], [[3, 4, 5]]).sum()
    assert called['count'] == 1
    loss.backward()
    assert sh.values.grad is not None and sh.values.grad.abs().sum() > 0

    # 2. two_sided_pseudolikelihood populates committed_tokens before on_block & handles n_eval == 0
    target, drafter, proj, E = _toy_env(seed=808)
    offline = DenseAdapter(E, p=6, D=4, M_sparse=4)
    st_pl = OnlineState(offline, lr=0.02, enable_sparse=True, sparse_capacity=64,
                        loss_mode='two_sided_pseudolikelihood')
    observed_in_callback = []
    res_pl = generate_toy(
        target, drafter, [1, 2, 3, 4], 20, temperature=0.7,
        warper=SamplingWarper(top_k=5, top_p=0.9),
        generator=torch.Generator().manual_seed(19),
        online_state=st_pl, projection=proj,
        on_block=lambda rec: observed_in_callback.append(list(rec['committed_tokens'])),
    )
    assert len(observed_in_callback) == len(res_pl.blocks)
    assert all(len(c) >= 1 for c in observed_in_callback)
    # n_eval == 0 safe (empty committed_tokens does not raise)
    assert st_pl.update(st_pl.last_round + 1, torch.zeros(2, target.vocab_size), None,
                        torch.zeros(2, 6), [1, 2], [3, 4], committed_tokens=[]) is False

    # 3. Lookahead, AdaptiveBlockController, TTSState, SharpDraftProposal, TokenRecycling all intact
    adapter = DenseAdapter(E, p=6, D=4, sparse=SignedHash(64), M_sparse=6)
    la_cfg = LookaheadConfig(mode='rerelax', domain='candidate_masked', R_init=2, R_clamp=1, beta_relax=0.5)
    res_la = generate_toy(
        target, drafter, [1, 2, 3, 4], 16, temperature=0.7,
        warper=SamplingWarper(top_k=5, top_p=0.9),
        generator=torch.Generator().manual_seed(21),
        adapter=adapter, projection=proj, proposal_mode='dense_candidate', lookahead=la_cfg,
    )
    assert res_la.num_output_tokens == 16

    abc = AdaptiveBlockController(max_block_length=4, min_cumulative_confidence=0.1)
    res_abc = generate_toy(target, drafter, [1, 2, 3, 4], 16, temperature=0.7,
                           generator=torch.Generator().manual_seed(22),
                           adapter=adapter, projection=proj, adaptive_controller=abc)
    assert res_abc.num_output_tokens == 16

    tts = TTSState(target.vocab_size, 6, max_slots=4, device=torch.device('cpu'))
    res_tts = generate_toy(target, drafter, [1, 2, 3, 4], 16, temperature=0.7,
                           generator=torch.Generator().manual_seed(23),
                           projection=proj, proposal_controller=tts)
    assert res_tts.num_output_tokens == 16

    sharp = SharpDraftProposal(gamma_max=1.35, gamma_min=0.95, entropy_pivot=0.35)
    res_sh = generate_toy(target, drafter, [1, 2, 3, 4], 16, temperature=0.7,
                          generator=torch.Generator().manual_seed(24),
                          proposal_controller=sharp)
    assert res_sh.num_output_tokens == 16

    res_tr, _ = generate_token_recycling_toy(
        target, drafter, [1, 2, 3, 4], 16, temperature=0.7,
        warper=SamplingWarper(top_k=5, top_p=0.9),
        generator=torch.Generator().manual_seed(25),
    )
    assert res_tr.num_output_tokens == 16

    # 4. Run M3 + M4 closed-loop in a fresh subprocess and assert edraft.triton_ops is NEVER imported
    code = (
        "import sys, edraft.eval_closed_loop as ecl; "
        f"ecl.main(['--scale', 'smoke', '--stage', 'm3', '--mode', 'all', '--output', {str(tmp_path)!r}]); "
        f"ecl.main(['--scale', 'smoke', '--stage', 'm4', '--mode', 'all', '--output', {str(tmp_path)!r}]); "
        "assert 'edraft.triton_ops' not in sys.modules, 'triton_ops leaked into M3/M4!'"
    )
    proc = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert proc.returncode == 0


# ---------------------------------------------------------------------------
# 3. Unified train.py (Round-2/3 + M5 + DDP ExactObservedPairs) & launch.py
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _ddp_train_worker(rank: int, world_size: int, port: int, out_dir: str, objective: str):
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    train.main([
        '--scale', 'smoke',
        '--output', out_dir,
        '--rank', str(rank),
        '--world_size', str(world_size),
        '--dist_backend', 'gloo',
        '--objective', objective,
        '--sparse_backend', 'exact_observed_pairs',
        '--microbatch_blocks', '2',
        '--block_shuffle_window', '4',
        '--m5_fit_tag', f'ddp_{objective}',
    ])


@pytest.mark.parametrize('objective', ['union_rest_grouped', 'two_sided_pseudolikelihood'])
def test_unified_train_microbatch_and_ddp_exact_observed_pairs(tmp_path, objective):
    # Single-process un-microbatched vs microbatch_blocks=2 parity with ExactObservedPairs
    out_single = tmp_path / f'single_{objective}'
    rep_single = train.main([
        '--scale', 'smoke',
        '--output', str(out_single),
        '--objective', objective,
        '--sparse_backend', 'exact_observed_pairs',
        '--optimizer', 'pacemuon',
        '--lr_schedule', 'cosine',
        '--block_shuffle_window', '4',
        '--calibrate_beta',
        '--m5_fit_tag', 'combo',
    ])
    assert rep_single['status'] in ('passed', 'failed')
    ds_single = rep_single['stages']['dense_sparse']
    assert ds_single['sparse_collision_statistics']['collision_count'] == 0
    assert 'beta_calibration' in ds_single

    out_mb = tmp_path / f'mb_{objective}'
    rep_mb = train.main([
        '--scale', 'smoke',
        '--output', str(out_mb),
        '--objective', objective,
        '--sparse_backend', 'exact_observed_pairs',
        '--optimizer', 'pacemuon',
        '--lr_schedule', 'cosine',
        '--block_shuffle_window', '4',
        '--microbatch_blocks', '2',
        '--calibrate_beta',
        '--m5_fit_tag', 'combo',
    ])
    a_single, _ = train.load_adapter_checkpoint(out_single / 'combo_smoke' / 'dense_sparse.pt', 'cpu')
    a_mb, _ = train.load_adapter_checkpoint(out_mb / 'combo_smoke' / 'dense_sparse.pt', 'cpu')
    assert torch.allclose(a_single.W, a_mb.W, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a_single.A, a_mb.A, atol=1e-5, rtol=1e-5)
    assert torch.allclose(a_single.b, a_mb.b, atol=1e-5, rtol=1e-5)
    s1 = a_single.sparse.snapshot()
    s2 = a_mb.sparse.snapshot()
    assert torch.equal(s1.keys, s2.keys)
    assert torch.allclose(s1.values, s2.values, atol=1e-5, rtol=1e-5)

    # 2-rank DDP on gloo with ExactObservedPairs + block_shuffle_window + microbatch_blocks
    out_ddp = tmp_path / f'ddp_{objective}'
    port = _free_port()
    ctx = mp.get_context('spawn')
    procs = [
        ctx.Process(target=_ddp_train_worker, args=(r, 2, port, str(out_ddp), objective))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    ddp_json = json.loads((out_ddp / f'ddp_{objective}_smoke.json').read_text())
    assert ddp_json['stages']['dense_sparse']['sparse_collision_statistics']['collision_count'] == 0


def test_scale_adapter_checkpoint_provenance_and_launch_cli(tmp_path, monkeypatch):
    out = tmp_path / 'fit'
    train.main(['--scale', 'smoke', '--output', str(out)])
    src_ckpt = out / 'm2_smoke' / 'dense.pt'
    s020 = tmp_path / 'dense_0p20.pt'
    s100_roundtrip = tmp_path / 'dense_roundtrip_1p0.pt'

    meta_020 = train.scale_adapter_checkpoint(src_ckpt, s020, 0.20)
    assert math.isclose(meta_020['effective_checkpoint_scale'], 0.20)
    raw_orig = torch.load(src_ckpt, map_location='cpu', weights_only=True)
    raw_020 = torch.load(s020, map_location='cpu', weights_only=True)
    assert math.isclose(raw_020['checkpoint_scale'], 0.20)
    assert raw_020['non_unit_scaled'] is True
    assert raw_020['identity'] != raw_orig['identity']
    assert raw_020['base_identity'] == raw_orig['identity']

    # Scaling 0.20 * 5.0 back to 1.0 still preserves non_unit_scaled provenance (M5-R2-BUG-3)
    train.scale_adapter_checkpoint(s020, s100_roundtrip, 5.0)
    raw_rt = torch.load(s100_roundtrip, map_location='cpu', weights_only=True)
    assert raw_rt['non_unit_scaled'] is True
    assert raw_rt['identity'] != raw_orig['identity']

    # launch.py worker command construction forwarding both Round-2/3 optimization flags and M5 flags
    spawned = []
    real_popen = subprocess.Popen

    class _FakeProc:
        def __init__(self, command, cwd=None, env=None, stdout=None, stderr=None):
            self.pid = 40000 + len(spawned)
            self.returncode = 0
            spawned.append({'command': list(command), 'env': dict(env or {})})

        def poll(self):
            return 0

        def terminate(self):
            pass

    def _popen_router(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and command and command[0] == sys.executable:
            return _FakeProc(command, *args, **kwargs)
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(launch, 'selected_gpus_idle', lambda indices: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', _popen_router)

    launch.main([
        '--task', 'train',
        '--scale', 'full',
        '--allow_large_run',
        '--gpus', '0,1',
        '--optimizer', 'pacemuon',
        '--lr_schedule', 'wsd',
        '--microbatch_blocks', '4',
        '--block_shuffle_window', '32',
        '--bf16_gemm',
        '--sparse_backend', 'exact_observed_pairs',
        '--calibrate_beta',
        '--output', str(tmp_path / 'launch_out'),
    ])
    assert len(spawned) == 2
    cmd0 = spawned[0]['command']
    for expected_flag in ('--optimizer', 'pacemuon', '--lr_schedule', 'wsd',
                          '--microbatch_blocks', '4', '--block_shuffle_window', '32',
                          '--bf16_gemm', '--sparse_backend', 'exact_observed_pairs',
                          '--calibrate_beta', '--allow_large_run'):
        assert expected_flag in cmd0
    assert 'MASTER_PORT' in spawned[0]['env']

    spawned.clear()
    launch.main([
        '--task', 'm5',
        '--eval_mode', 'evaluate',
        '--scale', 'reference',
        '--gpus', '0',
        '--compile',
        '--bigram_count_gate_kappa', '12.0',
        '--checkpoint_scale', '0.20',
        '--m5_arms', 'plain_dflash', 'online_bigram_count_gated', 'online_dbS_zero_init',
        '--output', str(tmp_path / 'launch_m5'),
    ])
    assert len(spawned) == 1
    m5_cmd = spawned[0]['command']
    for expected_flag in ('--stage', 'm5', '--compile', '--bigram_count_gate_kappa', '12.0',
                          '--checkpoint_scale', '0.2'):
        assert expected_flag in m5_cmd

    # WorkQueue dead-PID reclamation
    wq = WorkQueue(tmp_path / 'wq.json', ['item-a', 'item-b'])
    assert wq.claim(rank=0, pid=999_999_999) == 'item-a'
    # Dead PID 999_999_999 is reclaimed automatically on next claim
    assert wq.claim(rank=1) == 'item-a'
    wq.complete('item-a')
    assert wq.claim(rank=1) == 'item-b'


# ---------------------------------------------------------------------------
# 4. Validator & implementation_hash lockstep + --source_repo
# ---------------------------------------------------------------------------

def test_validator_hash_lockstep_and_source_repo_flag(tmp_path):
    val_mod = _load_validator_module()
    assert tuple(EVAL_IMPL_FILES) == tuple(val_mod.IMPLEMENTATION_FILES)
    assert implementation_hash() == implementation_hash('m3') == implementation_hash('m4')
    assert implementation_hash('m5') != implementation_hash('m3')

    # End-to-end M3 -> M4 smoke + validate.py --scale smoke
    art_dir = tmp_path / 'smoke_artifacts'
    eval_main(['--scale', 'smoke', '--stage', 'm3', '--mode', 'all', '--output', str(art_dir)])
    eval_main(['--scale', 'smoke', '--stage', 'm4', '--mode', 'all', '--output', str(art_dir)])

    audit_out = tmp_path / 'audit_current.json'
    rc = val_mod.main([
        '--scale', 'smoke',
        '--repo', str(ROOT),
        '--artifacts', str(art_dir),
        '--output', str(audit_out),
    ])
    assert rc == 0
    audit_json = json.loads(audit_out.read_text())
    assert audit_json['status'] == 'passed'
    assert audit_json['failures'] == []

    # Construct a synthetic 09a5469-layout 11-file source tree (without fastpath.py / optimizers.py)
    legacy_repo = tmp_path / 'legacy_09a5469_repo'
    legacy_edraft = legacy_repo / 'src' / 'edraft'
    legacy_edraft.mkdir(parents=True)
    import hashlib
    h_11 = hashlib.sha256()
    for name in val_mod.IMPLEMENTATION_FILES:
        if name in ('fastpath.py', 'optimizers.py'):
            continue
        data = (ROOT / 'src' / 'edraft' / name).read_bytes()
        (legacy_edraft / name).write_bytes(data)
        h_11.update(name.encode())
        h_11.update(data)
    legacy_impl_hash = h_11.hexdigest()

    # Without --source_repo, missing fastpath.py/optimizers.py in --repo MUST fail
    rc_missing = val_mod.main([
        '--scale', 'smoke',
        '--repo', str(legacy_repo),
        '--artifacts', str(art_dir),
        '--output', str(tmp_path / 'audit_missing.json'),
    ])
    assert rc_missing == 1

    # Re-stamp smoke M3/M4 artifacts with legacy_impl_hash to mimic frozen 09a5469 M3/M4 evidence
    legacy_art = tmp_path / 'legacy_artifacts'
    import shutil
    shutil.copytree(art_dir, legacy_art)

    def _rehash_tree(stage_name: str):
        s_path = legacy_art / stage_name / 'settings.json'
        s_data = json.loads(s_path.read_text())
        s_data['checkpoint_hashes']['implementation'] = legacy_impl_hash
        s_path.write_text(json.dumps(s_data, indent=2) + '\n')
        new_frozen = val_mod.fingerprint(s_data)

        for mode in ('tune', 'evaluate', 'timing'):
            m_path = legacy_art / stage_name / mode / 'manifest-rank0.json'
            m_data = json.loads(m_path.read_text())
            new_jobs = {}
            old_to_new_path = {}
            for old_key, entry in m_data['jobs'].items():
                ident = entry['identity']
                ident['checkpoint'] = s_data['checkpoint_hashes']
                if mode != 'tune':
                    ident['frozen_settings_hash'] = new_frozen
                new_key = val_mod.fingerprint(ident)
                orig_rpath = Path(entry['result_path'])
                local_old_rpath = legacy_art / stage_name / mode / 'rank0' / orig_rpath.name
                row = json.loads(local_old_rpath.read_text())
                row['identity'] = ident
                row['checkpoint_hashes'] = s_data['checkpoint_hashes']
                if stage_name == 'm4' and mode == 'evaluate' and row.get('baseline_reuse', {}).get('status') == 'reused':
                    src_m3_settings = legacy_art / 'm3' / 'settings.json'
                    src_m3_manifest = legacy_art / 'm3' / 'evaluate' / 'manifest-rank0.json'
                    m3_m = json.loads(src_m3_manifest.read_text())
                    m3_s = json.loads(src_m3_settings.read_text())
                    src_ident = dict(ident, stage='m3', frozen_settings_hash=val_mod.fingerprint(m3_s))
                    src_key = val_mod.fingerprint(src_ident)
                    src_rpath = Path(m3_m['jobs'][src_key]['result_path'])
                    row['baseline_reuse'].update({
                        'source_result_path': str(src_rpath.resolve()),
                        'source_result_sha256': val_mod.digest(src_rpath),
                        'source_manifest_path': str(src_m3_manifest.resolve()),
                        'source_manifest_sha256': val_mod.digest(src_m3_manifest),
                        'source_settings_path': str(src_m3_settings.resolve()),
                        'source_settings_sha256': val_mod.digest(src_m3_settings),
                        'source_identity_hash': src_key,
                        'source_frozen_settings_hash': val_mod.fingerprint(m3_s),
                    })
                local_old_rpath.unlink()
                new_rpath = local_old_rpath.with_name(f'{new_key}.json')
                new_rpath.write_text(json.dumps(row, indent=2) + '\n')
                for k_old in (str(orig_rpath), str(orig_rpath.resolve()), str(local_old_rpath), str(local_old_rpath.resolve())):
                    old_to_new_path[k_old] = str(new_rpath)
                new_jobs[new_key] = {
                    'identity': ident,
                    'status': 'complete',
                    'result_path': str(new_rpath),
                    'result_sha256': val_mod.digest(new_rpath),
                }
            m_data['jobs'] = new_jobs
            m_path.write_text(json.dumps(m_data, indent=2) + '\n')

            sess_dir = legacy_art / stage_name / mode / 'sessions'
            if sess_dir.is_dir():
                for sp in list(sess_dir.glob('*.json')):
                    sess = json.loads(sp.read_text())
                    sess['configuration']['checkpoint_hashes'] = s_data['checkpoint_hashes']
                    if mode == 'timing':
                        sess['configuration']['configuration']['frozen_settings_hash'] = new_frozen
                    for b in sess['bundles'].values():
                        b['result_paths'] = [old_to_new_path.get(p, p) for p in b['result_paths']]
                    sp.unlink()
                    new_sp = sess_dir / f"{val_mod.fingerprint(sess['configuration'])}.json"
                    new_sp.write_text(json.dumps(sess, indent=2) + '\n')
                    if mode == 'timing':
                        t_path = legacy_art / stage_name / 'timing.json'
                        t_data = json.loads(t_path.read_text())
                        t_data['frozen_settings_hash'] = new_frozen
                        t_data['raw_evidence']['session_path'] = str(new_sp.resolve())
                        t_data['raw_evidence']['session_sha256'] = val_mod.digest(new_sp)
                        t_data['raw_evidence']['manifest_path'] = str(m_path.resolve())
                        t_data['raw_evidence']['manifest_sha256'] = val_mod.digest(m_path)
                        t_path.write_text(json.dumps(t_data, indent=2) + '\n')

    _rehash_tree('m3')
    _rehash_tree('m4')

    audit_legacy_out = tmp_path / 'audit_legacy_passed.json'
    rc_legacy = val_mod.main([
        '--scale', 'smoke',
        '--repo', str(ROOT),
        '--source_repo', str(legacy_repo),
        '--artifacts', str(legacy_art),
        '--output', str(audit_legacy_out),
    ])
    assert rc_legacy == 0
    legacy_res = json.loads(audit_legacy_out.read_text())
    assert legacy_res['status'] == 'passed'
    assert legacy_res['failures'] == []


# ---------------------------------------------------------------------------
# 5. Post-Gate-E Ablation Runner (`python -m edraft.post_m4_suite`)
# ---------------------------------------------------------------------------

def test_post_m4_suite_smoke_all_and_resume(tmp_path):
    out_dir = tmp_path / 'suite_out'
    summary = post_m4_suite.main([
        '--scale', 'smoke',
        '--suite', 'all',
        '--output', str(out_dir),
        '--bench_warmups', '1',
        '--bench_repeats', '2',
        '--timing_warmups', '1',
        '--timing_repeats', '2',
        '--smoke_tokens', '16',
    ])
    assert summary['status'] == 'complete'
    assert set(summary['executed_suites']) == {
        'fastpath_bench',
        'online_head_to_head',
        'm2_refit_calibration',
    }
    assert summary['implementation_hashes']['m3'] == summary['implementation_hashes']['m4']
    assert summary['implementation_hashes']['m5'] != summary['implementation_hashes']['m3']

    # (a) fastpath_bench checks
    fp = summary['suites']['fastpath_bench']
    assert fp['status'] == 'complete'
    assert fp['grouped_kl_microbenchmark']['parity_passed'] is True
    cl = fp['closed_loop_matched_comparison']
    assert cl['modes'] == ['eager_pre_round3', 'round3_compact', 'round3_compile']
    assert len(cl['paired_speedups']) == len(cl['arms'])
    assert all(row['mean_phase_ms'] for row in cl['summary_by_mode_and_arm'])

    # (b) online_head_to_head checks
    h2h = summary['suites']['online_head_to_head']
    assert h2h['status'] == 'complete'
    assert h2h['bigram_count_gate_kappa'] == 12.0
    assert set(h2h['arms']) == {
        'plain_dflash',
        'online_bigram',
        'online_bigram_count_gated',
        'online_dbS_zero_init',
        'zero_offline_online_only',
    }
    assert h2h['offset_labels'] == list(OUTPUT_LABELS)
    assert h2h['selected_settings']['online_bigram_count_gated']['count_gate_kappa'] == 12.0
    assert len(h2h['evaluation_offset_bin_slices']) > 0
    assert len(h2h['isolated_timing_offset_bin_slices']) > 0
    assert len(h2h['head_to_head_online_offset_comparison']) > 0

    # (c) m2_refit_calibration checks
    cal = summary['suites']['m2_refit_calibration']
    assert cal['status'] == 'complete'
    assert cal['refit_hyperparameters']['block_shuffle_window'] == 32
    assert cal['refit_hyperparameters']['weight_decay'] == 1e-2
    assert cal['refit_hyperparameters']['fixed_checkpoint_scale'] == 0.20
    assert cal['beta_calibration'] is not None
    variant_names = [v['variant'] for v in cal['variants']]
    assert variant_names == [
        'baseline_m2_unscaled',
        'baseline_m2_scale_0p20',
        'refit_shuf_wd_uncalibrated',
        'refit_shuf_wd_dev_calibrated',
        'refit_shuf_wd_scale_0p20',
    ]
    for v in cal['variants']:
        assert math.isfinite(v['offline_dense_kl']['test']['weighted_mean_kl'])
        assert math.isfinite(v['offline_dense_sparse_kl']['test']['weighted_mean_kl'])
        assert 'dense_sparse' in v['closed_loop']

    # Resumed invocation reuses completed sub-suite artifacts instantaneously
    summary_resumed = post_m4_suite.main([
        '--scale', 'smoke',
        '--suite', 'all',
        '--output', str(out_dir),
    ])
    assert summary_resumed['status'] == 'complete'
