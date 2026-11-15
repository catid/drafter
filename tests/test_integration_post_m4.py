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
        assert v['closed_loop']['status'] == 'complete'
        assert 'plain_dflash' in v['closed_loop']
        assert 'dense' in v['closed_loop']
        assert 'dense_sparse' in v['closed_loop']
        assert v['closed_loop']['dense']['effective_beta'] == 1.0
        assert v['closed_loop']['dense_sparse']['effective_beta'] == 1.0

    # Resumed invocation with identical CLI config reuses completed sub-suite artifacts instantaneously
    summary_resumed = post_m4_suite.main([
        '--scale', 'smoke',
        '--suite', 'all',
        '--output', str(out_dir),
        '--bench_warmups', '1',
        '--bench_repeats', '2',
        '--timing_warmups', '1',
        '--timing_repeats', '2',
        '--smoke_tokens', '16',
    ])
    assert summary_resumed['status'] == 'complete'
    for s_name in ('fastpath_bench', 'online_head_to_head', 'm2_refit_calibration'):
        assert (
            summary_resumed['suites'][s_name]['config_fingerprint']
            == summary['suites'][s_name]['config_fingerprint']
        )


# ---------------------------------------------------------------------------
# 6. Phase-1 Review Regression Tests (INT-BUG-1 .. INT-BUG-7)
# ---------------------------------------------------------------------------

def test_int_bug_1_and_2_m2_refit_reference_prereqs_and_matched_bundle(tmp_path, monkeypatch):
    # Create a fake frozen reference artifacts_root and verify:
    # (1) INT-BUG-1: diagnostic_reference.json is linked into ws and passed via --m1_diagnostic to train.main;
    # (2) INT-BUG-2: EvaluationRunner is instantiated ONCE, evaluates plain_dflash + all 5*(dense, dense_sparse)
    #     arms at fixed beta=1.0 (zero dev re-tuning on pre-scaled checkpoints), and matches smoke schema.
    art_root = tmp_path / 'frozen_ref_artifacts'
    (art_root / 'traces').mkdir(parents=True)
    (art_root / 'm2').mkdir(parents=True)
    (art_root / 'm3').mkdir(parents=True)
    (art_root / 'm4').mkdir(parents=True)

    # Seed baseline smoke checkpoints to reuse valid tensor payloads for mocked reference checkpoints
    seed_fit = tmp_path / 'seed_smoke'
    train.main(['--scale', 'smoke', '--output', str(seed_fit)])
    proj_sha = 'proj-sha-ref-123'

    def _make_ref_ckpt(src_pt: Path, dst_pt: Path, stage_lbl: str):
        payload = torch.load(src_pt, map_location='cpu', weights_only=True)
        payload['scale'] = 'reference'
        payload['stage'] = stage_lbl
        payload['report'] = {
            'model_pair': ['Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16'],
            'model_revisions': ['rev-t', 'rev-d'],
            'frozen_artifacts': {'hidden_projection_sha256': proj_sha},
        }
        train.atomic_torch_save(dst_pt, payload)

    _make_ref_ckpt(seed_fit / 'm2_smoke' / 'dense.pt', art_root / 'm2' / 'dense.pt', 'dense')
    _make_ref_ckpt(seed_fit / 'm2_smoke' / 'dense_sparse.pt', art_root / 'm2' / 'dense_sparse.pt', 'dense_sparse')
    (art_root / 'm1_audit.json').write_text(json.dumps({'status': 'passed', 'shards': []}))
    (art_root / 'diagnostic_reference.json').write_text(json.dumps({'status': 'measured', 'scale': 'reference'}))
    (art_root / 'h_proj.pt').write_bytes(b'proj')
    (art_root / 'token_features.pt').write_bytes(b'feat')
    (art_root / 'm2.json').write_text(json.dumps({'status': 'passed', 'scale': 'reference', 'identity': 'id-m2'}))
    (art_root / 'm3' / 'summary.json').write_text(json.dumps({'status': 'complete', 'scale': 'reference'}))
    (art_root / 'm4' / 'summary.json').write_text(json.dumps({'status': 'complete', 'scale': 'reference'}))
    (art_root / 'smoke.json').write_text(json.dumps({'status': 'passed', 'tokens_per_second': 1000.0}))
    (art_root / 'environment.json').write_text(json.dumps({'status': 'recorded'}))
    prompts_payload = {
        'scale': 'reference',
        'prompts': [
            {
                'prompt_id': f'ref-test-{cond}-{i}',
                'split': 'test',
                'condition': cond,
                'text': f'Prompt {i}',
                'dataset': 'test_ds',
                'reference_selected': True,
            }
            for cond, n_req in (('standard_nonthinking', 64), ('long_thinking', 32))
            for i in range(n_req)
        ],
    }
    (art_root / 'prompts.json').write_text(json.dumps(prompts_payload))

    captured_train_cli = {}

    def _fake_train_main(cli_list):
        captured_train_cli['argv'] = list(cli_list)
        out_ws = Path(cli_list[cli_list.index('--output') + 1])
        tag = cli_list[cli_list.index('--m5_fit_tag') + 1]
        # Verify diagnostic_reference.json was linked BEFORE train.main runs!
        diag_arg = Path(cli_list[cli_list.index('--m1_diagnostic') + 1])
        assert diag_arg.is_file(), 'diagnostic_reference.json was not linked into refit workspace!'
        rdir = out_ws / tag
        rdir.mkdir(parents=True, exist_ok=True)
        _make_ref_ckpt(seed_fit / 'm2_smoke' / 'dense.pt', rdir / 'dense.pt', 'dense')
        _make_ref_ckpt(seed_fit / 'm2_smoke' / 'dense_sparse.pt', rdir / 'dense_sparse.pt', 'dense_sparse')
        train.scale_adapter_checkpoint(rdir / 'dense.pt', rdir / 'dense_calibrated.pt', 0.35, preserve_fit_identity=True)
        train.scale_adapter_checkpoint(rdir / 'dense_sparse.pt', rdir / 'dense_sparse_calibrated.pt', 0.35, preserve_fit_identity=True)
        rep = {
            'status': 'passed',
            'stages': {
                'dense': {'beta_calibration': {'optimal_beta': 0.35}},
                'dense_sparse': {'beta_calibration': {'optimal_beta': 0.35}},
            },
        }
        (out_ws / f'{tag}.json').write_text(json.dumps(rep))
        return rep

    runner_instances = []
    seen_runs = []

    class _FakeEvaluationRunner:
        def __init__(self, ns):
            runner_instances.append(ns)
            self.args = ns
            self.device = torch.device('cpu')
            self.tiny = False
            self.model_pair = ['Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16']
            self.model_revisions = ['rev-t', 'rev-d']
            self.checkpoint_hashes = {'projection': proj_sha}
            self.dense, _ = train.load_adapter_checkpoint(ns.dense_checkpoint, 'cpu')
            self.dense_sparse, _ = train.load_adapter_checkpoint(ns.sparse_checkpoint, 'cpu')

        def run(self, job, setting, profile=False):
            beta = setting.get('beta', 1.0)
            if job.get('max_new_tokens_override') is None:
                seen_runs.append((job['arm'], beta, job['prompt']['prompt_id']))
            return {
                'effective_beta': beta,
                'generated_tokens': 32,
                'elapsed_seconds': 0.5,
                'tokens_per_second': 64.0,
                'mean_accepted_length': 2.5 if job['arm'] != 'plain_dflash' else 2.0,
                'median_accepted_length': 2.5 if job['arm'] != 'plain_dflash' else 2.0,
                'target_forwards_per_generated_token': 0.35,
            }

    monkeypatch.setattr(train, 'main', _fake_train_main)
    monkeypatch.setattr(post_m4_suite, 'EvaluationRunner', _FakeEvaluationRunner)
    monkeypatch.setattr(post_m4_suite, 'assert_idle', lambda dev: {'mocked': True})
    monkeypatch.setattr(post_m4_suite, 'assert_no_external_gpu_processes', lambda dev: None)
    monkeypatch.setattr(post_m4_suite, '_prepare_offline_eval_context', lambda a, w, d: ('ctx', {}, None, None, {}))
    monkeypatch.setattr(
        post_m4_suite,
        '_offline_kl_summary',
        lambda ctx, ckpt, dev: {
            'checkpoint': str(ckpt),
            'dev': {'weighted_mean_kl': 0.12},
            'test': {'weighted_mean_kl': 0.13},
        },
    )

    out_suite = tmp_path / 'ref_suite_out'
    monkeypatch.setattr('shutil.disk_usage', lambda p: type('DU', (), {'free': 200 * 2**30})())
    res = post_m4_suite.main([
        '--scale', 'reference',
        '--device', 'cpu',
        '--suite', 'm2_refit_calibration',
        '--artifacts_root', str(art_root),
        '--output', str(out_suite),
        '--timing_repeats', '2',
        '--timing_warmups', '1',
    ])

    assert '--m1_diagnostic' in captured_train_cli['argv']
    assert len(runner_instances) == 1, 'Backend/EvaluationRunner must be instantiated once, not 5x!'
    assert all(b == 1.0 for _, b, _ in seen_runs), 'All checkpoints must be evaluated at fixed beta=1.0!'
    arms_seen = {arm for arm, _, _ in seen_runs}
    assert arms_seen == {'plain_dflash', 'dense', 'dense_sparse'}, 'Both dense and dense_sparse must be evaluated!'
    cal_rep = res['suites']['m2_refit_calibration']
    assert cal_rep['status'] == 'complete'
    scales_seen = [v['closed_loop']['dense']['checkpoint_scale'] for v in cal_rep['variants']]
    assert math.isclose(scales_seen[0], 1.0)
    assert math.isclose(scales_seen[1], 0.20)
    assert math.isclose(scales_seen[2], 1.0)
    assert math.isclose(scales_seen[3], 0.35)
    assert math.isclose(scales_seen[4], 0.20)


def test_int_bug_3_isolated_artifacts_root_and_notes_clean(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from edraft.runtime import preflight, resolve_artifact_path

    notes_path = ROOT / 'NOTES.md'
    notes_before = notes_path.read_bytes()

    isolated_out = tmp_path / 'isolated_output'
    isolated_out.mkdir(parents=True)
    (isolated_out / 'smoke.json').write_text(json.dumps({'status': 'passed', 'tokens_per_second': 500.0}))
    (isolated_out / 'environment.json').write_text(json.dumps({'marker': 'from_isolated_output'}))

    assert resolve_artifact_path('smoke.json', isolated_out) == isolated_out / 'smoke.json'
    assert resolve_artifact_path('environment.json', isolated_out) == isolated_out / 'environment.json'

    env_override_dir = tmp_path / 'env_override'
    env_override_dir.mkdir()
    (env_override_dir / 'environment.json').write_text(json.dumps({'marker': 'from_env_var'}))
    monkeypatch.setenv('EDRAFT_ARTIFACTS_ROOT', str(env_override_dir))
    assert resolve_artifact_path('environment.json', tmp_path / 'nonexistent') == env_override_dir / 'environment.json'

    monkeypatch.setattr('shutil.disk_usage', lambda p: type('DU', (), {'free': 200 * 2**30})())
    rec = preflight(SimpleNamespace(scale='reference', output=isolated_out), 2500)
    assert math.isclose(rec['cpu_smoke_extrapolation_seconds'], 5.0)
    assert (isolated_out / 'preflight_estimates.jsonl').is_file()
    assert notes_path.read_bytes() == notes_before, 'Isolated preflight must NEVER mutate ROOT/NOTES.md!'


def test_int_bug_4_and_5_pilot_status_config_aware_resume_and_threads(tmp_path):
    orig_threads = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        out_dir = tmp_path / 'suite_h2h'
        # 1) Pilot run (--timing_prompts 1) with kappa=4.0 -> MUST be 'pilot_complete', NOT 'complete'
        res_pilot = post_m4_suite.main([
            '--scale', 'smoke',
            '--suite', 'online_head_to_head',
            '--output', str(out_dir),
            '--online_arms', 'plain_dflash', 'online_bigram_count_gated',
            '--bigram_count_gate_kappa', '4.0',
            '--timing_prompts', '1',
            '--timing_repeats', '2',
            '--smoke_tokens', '12',
        ])
        assert torch.get_num_threads() == 3, 'post_m4_suite.main must restore caller CPU thread count!'
        assert res_pilot['status'] == 'pilot_complete'
        assert res_pilot['suites']['online_head_to_head']['status'] == 'pilot_complete'

        # 2) Full run (--timing_prompts 0) with kappa=64.0 and extra arm 'online_dbS_zero_init' on SAME --output:
        #    default --resume MUST NOT return the stale pilot report!
        res_full = post_m4_suite.main([
            '--scale', 'smoke',
            '--suite', 'online_head_to_head',
            '--output', str(out_dir),
            '--online_arms', 'plain_dflash', 'online_bigram_count_gated', 'online_dbS_zero_init',
            '--bigram_count_gate_kappa', '64.0',
            '--timing_prompts', '0',
            '--timing_repeats', '2',
            '--smoke_tokens', '12',
        ])
        assert res_full['status'] == 'complete'
        h2h_full = res_full['suites']['online_head_to_head']
        assert h2h_full['status'] == 'complete'
        assert h2h_full['bigram_count_gate_kappa'] == 64.0
        assert 'online_dbS_zero_init' in h2h_full['arms']

        # 3) Changing m2_refit_calibration hyperparameters on SAME --output invalidates --resume cleanly
        out_m2 = tmp_path / 'suite_m2'
        m2_r1 = post_m4_suite.main([
            '--scale', 'smoke',
            '--suite', 'm2_refit_calibration',
            '--output', str(out_m2),
            '--weight_decay', '0.01',
            '--fixed_checkpoint_scale', '0.20',
            '--smoke_tokens', '12',
        ])
        m2_r2 = post_m4_suite.main([
            '--scale', 'smoke',
            '--suite', 'm2_refit_calibration',
            '--output', str(out_m2),
            '--weight_decay', '0.25',
            '--fixed_checkpoint_scale', '0.50',
            '--smoke_tokens', '12',
        ])
        hp2 = m2_r2['suites']['m2_refit_calibration']['refit_hyperparameters']
        assert math.isclose(hp2['weight_decay'], 0.25)
        assert math.isclose(hp2['fixed_checkpoint_scale'], 0.50)
        assert (
            m2_r2['suites']['m2_refit_calibration']['config_fingerprint']
            != m2_r1['suites']['m2_refit_calibration']['config_fingerprint']
        )
        v_names = [v['variant'] for v in m2_r2['suites']['m2_refit_calibration']['variants']]
        assert 'baseline_m2_scale_0p50' in v_names
        assert 'refit_shuf_wd_scale_0p50' in v_names
    finally:
        torch.set_num_threads(orig_threads)


def test_int_bug_6_train_bf16_gemm_calibrate_beta_and_two_sided_parity(tmp_path):
    for bf16_flag in (False, True):
        out_dir = tmp_path / f'fit_bf16_{int(bf16_flag)}'
        cli = [
            '--scale', 'smoke',
            '--output', str(out_dir),
            '--epochs_dense', '3',
            '--epochs_sparse', '3',
            '--learning_rate', '0.05',
            '--calibrate_beta',
            '--apply_calibrated_beta',
            '--calibrate_beta_grid', '1.0',
            '--m5_fit_tag', 'repro6',
        ]
        if bf16_flag:
            cli.append('--bf16_gemm')
        train.main(cli)
        rep = json.loads((out_dir / 'repro6_smoke.json').read_text())
        st = rep['stages']['dense_sparse']
        pt1 = [pt for pt in st['beta_calibration']['grid_curve'] if abs(pt['beta'] - 1.0) < 1e-9][0]
        cal_gkl = pt1['dev_union_rest_grouped_kl']
        ev_gkl = st['dev']['weighted_mean_kl'] if 'dev' in st else st['eval']['dev']['weighted_mean_kl']
        assert abs(cal_gkl - ev_gkl) < 1e-7, (
            f'bf16_gemm={bf16_flag}: calibrate_beta(1.0)={cal_gkl} disagreed with evaluate()={ev_gkl}'
        )

    # Verify two_sided_pseudolikelihood + --bf16_gemm end-to-end training & evaluation parity
    out_pl = tmp_path / 'fit_pl_bf16'
    rep_pl = train.main([
        '--scale', 'smoke',
        '--output', str(out_pl),
        '--objective', 'two_sided_pseudolikelihood',
        '--bf16_gemm',
        '--epochs_dense', '2',
        '--epochs_sparse', '2',
        '--m5_fit_tag', 'pl_bf16',
    ])
    assert math.isfinite(rep_pl['stages']['dense_sparse']['dev']['weighted_mean_kl'])


def test_int_bug_7_snapshot_vs_adapter_unary_bf16_and_m5_torch_compile():
    from edraft.fastpath import FastPathConfig, FastPathRuntime
    from edraft.triton_ops import build_gated_torch_compile

    gen = torch.Generator().manual_seed(42)
    E = torch.randn(32, 8, generator=gen)
    H = torch.randn(4, 6, generator=gen)

    calls = {'n': 0}

    def custom_4arg_unary(E_in, W_in, b_in, H_in):
        calls['n'] += 1
        return (H_in @ W_in.T) @ E_in.T + b_in + 0.123

    adapter = DenseAdapter(E, p=6, D=4, unary_fn=custom_4arg_unary)
    with torch.no_grad():
        adapter.W.copy_(torch.randn_like(adapter.W, generator=gen) * 0.1)
        adapter.b.copy_(torch.randn_like(adapter.b, generator=gen) * 0.05)
    snap = adapter.snapshot()

    # Under bf16_gemm=False, both DenseAdapter and ProposalSnapshot invoke custom_4arg_unary
    u_ad_fp32 = adapter.unary(H, bf16_gemm=False)
    u_sn_fp32 = snap.unary(H, bf16_gemm=False)
    assert calls['n'] == 2
    assert torch.equal(u_ad_fp32, u_sn_fp32)

    # Under bf16_gemm=True, both bypass un-flagged 4-arg unary_fn and match BF16 block_unary identically
    u_ad_bf16 = adapter.unary(H, bf16_gemm=True)
    u_sn_bf16 = snap.unary(H, bf16_gemm=True)
    assert calls['n'] == 2
    assert torch.equal(u_ad_bf16, u_sn_bf16)

    # Under build_gated_torch_compile(bf16_gemm=True), _supports_bf16_gemm=True is honored by both
    # AND FastPathRuntime(bf16_gemm=True, compile=False) does NOT shadow snapshot.unary_fn!
    tc_fns, _ = build_gated_torch_compile(
        'cpu', vocab_size=32, r=8, p=6, K=4, bf16_gemm=True, speed_margin=100.0, return_callables=True,
    )
    tc_fn = tc_fns['torch_compile_block_unary']
    if tc_fn is not None:
        assert getattr(tc_fn, '_supports_bf16_gemm', False) is True
        adapter.unary_fn = tc_fn
        snap_tc = adapter.snapshot()
        assert torch.allclose(adapter.unary(H, bf16_gemm=True), u_ad_bf16, atol=1e-5, rtol=1e-5)
        assert torch.allclose(snap_tc.unary(H, bf16_gemm=True), u_sn_bf16, atol=1e-5, rtol=1e-5)

        seen_tc = {'count': 0}

        def wrapped_tc(E_in, W_in, b_in, H_in, *, bf16_gemm=False, E_bf16=None):
            seen_tc['count'] += 1
            return tc_fn(E_in, W_in, b_in, H_in, bf16_gemm=bf16_gemm, E_bf16=E_bf16)

        wrapped_tc._supports_bf16_gemm = True
        adapter.unary_fn = wrapped_tc
        fp_bf16_only = FastPathRuntime(FastPathConfig(bf16_gemm=True), static_block_length=4)
        logits = torch.randn(4, 32, generator=gen)
        _sample_proposal(
            logits, H, [1, 2, 3, 4],
            temperature=0.7, warper=SamplingWarper(top_k=5, top_p=0.9),
            generator=torch.Generator().manual_seed(9),
            snapshot=adapter.snapshot(), projection=None, beta=1.0, bigram=None,
            timer=PhaseTimer('cpu', False), fastpath=fp_bf16_only,
        )
        assert seen_tc['count'] == 1, 'FastPathRuntime(bf16_gemm=True, compile=False) must not shadow snapshot.unary_fn!'

