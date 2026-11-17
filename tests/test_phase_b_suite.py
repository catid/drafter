"""End-to-end CPU tests for Phase-B secondary reference ablations and `--scale full` sharding."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edraft import eval_closed_loop, phase_b_suite, prompts
from edraft.runtime import ROOT, atomic_json, fingerprint


def _make_full_scale_synthetic_manifest() -> dict:
    rows = []
    for i in range(64):
        rows.append({
            'prompt_id': f'std-dev-{i:03d}',
            'dataset': 'synthetic_gsm',
            'split': 'dev',
            'condition': 'standard_nonthinking',
            'reference_selected': True,
            'text': f'Standard dev {i}',
        })
    for i in range(6850):
        rows.append({
            'prompt_id': f'std-train-{i:04d}',
            'dataset': 'synthetic_gsm',
            'split': 'train',
            'condition': 'standard_nonthinking',
            'reference_selected': True,
            'text': f'Standard train {i}',
        })
    for i in range(164):
        rows.append({
            'prompt_id': f'he-test-{i:03d}',
            'dataset': 'synthetic_humaneval',
            'split': 'test',
            'condition': 'standard_nonthinking',
            'reference_selected': i < 128,
            'text': f'Standard test {i}',
        })
    for i in range(32):
        rows.append({
            'prompt_id': f'long-dev-{i:03d}',
            'dataset': 'synthetic_math500',
            'split': 'dev',
            'condition': 'long_thinking',
            'reference_selected': True,
            'text': f'Long dev {i}',
        })
    for i in range(96):
        rows.append({
            'prompt_id': f'long-test-{i:03d}',
            'dataset': 'synthetic_math500',
            'split': 'test',
            'condition': 'long_thinking',
            'reference_selected': True,
            'text': f'Long test {i}',
        })
    return {
        'scale': 'full',
        'seed': 2026,
        'complete_heldout_dataset': 'synthetic_humaneval',
        'calibration_ids': [f'std-train-{i:04d}' for i in range(64)],
        'prompts': rows,
    }


def test_bigram_eviction_arms_and_memory_summary_end_to_end(tmp_path):
    ws = tmp_path / 'bigram_evict_ws'
    arms = [
        'plain_dflash',
        'online_bigram',
        'online_bigram_kret256',
        'online_bigram_kret128',
        'online_bigram_count_gated',
    ]
    base = [
        '--scale', 'smoke', '--device', 'cpu', '--stage', 'm5',
        '--output', str(ws),
        '--timing_repeats', '2', '--timing_warmups', '1', '--smoke_tokens', '16',
        '--bigram_k_ret', '2',
        '--bigram_count_gate_kappa', '12.0',
        '--m5_arms', *arms,
    ]
    for r in range(2):
        eval_closed_loop.main([*base, '--mode', 'tune', '--rank', str(r), '--world_size', '2', '--logical_session', 'sess_bg'])
    eval_closed_loop.main([*base, '--mode', 'tune_summarize'])
    settings = json.loads((ws / 'm5' / 'settings.json').read_text())
    assert settings['selected']['online_bigram_kret256']['k_ret'] == 2
    assert settings['selected']['online_bigram_kret128']['k_ret'] == 128
    assert settings['selected']['online_bigram']['k_ret'] is None

    for r in range(2):
        eval_closed_loop.main([*base, '--mode', 'evaluate', '--schedule', 'dynamic', '--rank', str(r), '--world_size', '2'])
    for r in range(2):
        eval_closed_loop.main([*base, '--mode', 'timing', '--rank', str(r), '--world_size', '2', '--logical_session', 'sess_tm'])
    eval_closed_loop.main([*base, '--mode', 'summarize'])

    summary = json.loads((ws / 'm5' / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    bg_mem = summary.get('bigram_memory_ablation', [])
    assert len(bg_mem) > 0
    by_arm = {row['arm']: row for row in bg_mem if row['temperature'] > 0}
    assert by_arm['online_bigram']['k_ret'] is None
    assert by_arm['online_bigram_kret256']['k_ret'] == 2
    assert by_arm['online_bigram_kret256']['retained_prior_pairs_mean'] <= by_arm['online_bigram']['exact_ledger_pairs_mean']
    matched_families = {row['matched_support_family'] for row in summary['matched_support_comparisons']}
    assert 'online_bigram_topk_eviction' in matched_families


def test_full_scale_m3_and_m4_sharded_tune_and_multi_rank_token_weights(tmp_path):
    # 1. Verify smoke and reference M3/M4 still reject world_size > 1 tune & tune_summarize
    for bad_scale in ('smoke', 'reference'):
        for stg in ('m3', 'm4'):
            with pytest.raises(SystemExit):
                eval_closed_loop.main([
                    '--scale', bad_scale, '--device', 'cpu', '--stage', stg,
                    '--mode', 'tune', '--rank', '0', '--world_size', '2',
                    '--output', str(tmp_path / f'reject_{bad_scale}_{stg}'),
                ])
            with pytest.raises(SystemExit):
                eval_closed_loop.main([
                    '--scale', bad_scale, '--device', 'cpu', '--stage', stg,
                    '--mode', 'tune_summarize',
                    '--output', str(tmp_path / f'reject_sum_{bad_scale}_{stg}'),
                ])

    # 2. Verify full-scale M3 & M4 (all 9 full M4 arms!) run sharded tune (W=4) + tune_summarize cleanly
    #    and `compute_tune_sharding_plan` consumes m1_audit.json for M3, then multi-rank M3 tune for M4!
    ws = tmp_path / 'full_tune_ws'
    manifest = _make_full_scale_synthetic_manifest()
    dev_prompts = eval_closed_loop.select_prompts(manifest, 'dev', 'full', standard_dev_prompts=4, long_dev_prompts=32)
    assert len(dev_prompts) == 36

    # Populate m1_audit.json so M3 sharded tune uses `m1_audit_realized_tokens`
    m1_gens = [
        {'prompt_id': p['prompt_id'], 'realized_length': (1000 + 50 * idx) if p['condition'].startswith('long_') else (100 + 10 * idx)}
        for idx, p in enumerate(dev_prompts)
    ]
    atomic_json(ws / 'm1_audit.json', {'scale': 'full', 'status': 'passed', 'generations': m1_gens})

    class _FakeShardedRunner:
        def __init__(self, stage: str, rank: int, world_size: int):
            self.args = SimpleNamespace(
                scale='full',
                allow_large_run=True,
                stage=stage,
                mode='tune',
                rank=rank,
                world_size=world_size,
                seed=2026,
                device='cpu',
                output=ws,
                max_new_tokens=2048,
                long_max_new_tokens=24576,
                smoke_tokens=32,
                standard_dev_prompts=4,
                long_dev_prompts=32,
                timing_warmups=1,
                fresh_session=False,
                logical_session=f'sess_{stage}',
                m5_arms=None,
                m5_oracle_checkpoint=None,
            )
            self.tiny = False
            self.device = eval_closed_loop.torch.device('cpu')
            self.affinity_info = eval_closed_loop.gpu_cpu_affinity(rank, total_cpus=128)
            self.affinity_info['cpu_affinity_applied'] = True
            self.checkpoint_hashes = {
                'dense': 'a' * 64,
                'dense_sparse': 'b' * 64,
                'projection': 'c' * 64,
                'implementation': eval_closed_loop.implementation_hash(stage),
            }
            self.model_pair = ['Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16']
            self.model_revisions = ['rev_t', 'rev_d']
            self.generation_environment = {'synthetic_full_stub': True}
            self.fastpath = None

        def run(self, job, setting, profile=False):
            pid = job['prompt']['prompt_id']
            arm = job['arm']
            toks = 240 if job['prompt']['condition'].startswith('long_') else 64
            bonus = 1.15 if setting.get('update_stride') == 5 or setting.get('beta') == 0.5 or setting.get('lambda_tr') == 0.2 else 1.0
            tps = 100.0 * bonus
            return {
                'scale': 'full',
                'label': 'full',
                'stage': self.args.stage,
                'arm': arm,
                'prompt_id': pid,
                'split': 'dev',
                'condition': job['prompt']['condition'],
                'seed': job['seed'],
                'sampling_name': job['sampling_name'],
                'temperature': job['temperature'],
                'sampling_warp_config': eval_closed_loop.asdict(job['warper']),
                'settings': setting,
                'verification_rule': 'ratio' if job['temperature'] > 0 else 'greedy',
                'model_pair': self.model_pair,
                'model_revisions': self.model_revisions,
                'checkpoint_hashes': self.checkpoint_hashes,
                'generated_tokens': toks,
                'elapsed_seconds': toks / tps,
                'tokens_per_second': tps,
                'mean_accepted_length': 4.5,
                'median_accepted_length': 4.0,
                'target_forwards_per_generated_token': 0.2,
                'phase_ms': {},
                'blocks': [],
                'generation': {'complete': True, 'repetition_dominated': False},
                'output_tokens': [1] * toks,
                'phase_profile': profile,
            }

    # Run 4-rank sharded M3 tune + tune_summarize
    for r in range(4):
        out_r = eval_closed_loop.tune(_FakeShardedRunner('m3', r, 4), manifest)
        assert out_r['status'] == 'tune_rank_complete'
    m3_plan = json.loads((ws / 'm3' / 'tune' / 'sharding_plan.json').read_text())
    assert m3_plan['weight_source'] == 'm1_audit_realized_tokens'

    sum_args_m3 = copy.copy(_FakeShardedRunner('m3', 0, 1).args)
    m3_settings = eval_closed_loop.summarize_sharded_tune(sum_args_m3, manifest)
    assert m3_settings['sharding_mode'] == 'sharded_4gpu'
    assert set(m3_settings['selected']) == set(eval_closed_loop.M3_ARMS)
    eval_closed_loop.validate_frozen_settings(
        m3_settings, 'm3', 'full', m3_settings['checkpoint_hashes'], manifest,
        m3_settings['generation_caps'], m3_settings['model_pair'], m3_settings['model_revisions'],
        m3_settings['generation_environment'],
    )

    # Run 4-rank sharded M4 tune + tune_summarize (9 full-scale arms, weighted from multi-rank m3 tune!)
    for r in range(4):
        out_r4 = eval_closed_loop.tune(_FakeShardedRunner('m4', r, 4), manifest)
        assert out_r4['status'] == 'tune_rank_complete'
    m4_plan = json.loads((ws / 'm4' / 'tune' / 'sharding_plan.json').read_text())
    assert m4_plan['weight_source'] == 'm3_tune_realized_tokens'

    sum_args_m4 = copy.copy(_FakeShardedRunner('m4', 0, 1).args)
    m4_settings = eval_closed_loop.summarize_sharded_tune(sum_args_m4, manifest)
    assert m4_settings['sharding_mode'] == 'sharded_4gpu'
    expected_full_m4 = {
        'plain_dflash', 'online_bigram', 'offline_static', 'offline_online_stride1', 'online_dbS',
        'offline_online_stride5', 'zero_offline_online_only', 'online_rejection_relative', 'online_cutoff',
    }
    assert set(m4_settings['selected']) == expected_full_m4
    eval_closed_loop.validate_frozen_settings(
        m4_settings, 'm4', 'full', m4_settings['checkpoint_hashes'], manifest,
        m4_settings['generation_caps'], m4_settings['model_pair'], m4_settings['model_revisions'],
        m4_settings['generation_environment'],
    )


def test_real_cached_prompts_full_scale_capacity_verification(tmp_path):
    ref_env = ROOT / 'artifacts' / 'environment.json'
    if not ref_env.is_file():
        ref_env = Path('/home/catid/drafter/artifacts/environment.json')
    if not ref_env.is_file():
        pytest.skip('No local reference environment.json cached on this machine')
    env_data = json.loads(ref_env.read_text())
    ds_art = env_data.get('dataset_artifacts', {})
    if not all(Path(ds_art.get(d, {}).get('path', '')).is_dir() for d in prompts.DATASETS):
        pytest.skip('Cached HF benchmark datasets not present on disk')

    out_dir = tmp_path / 'full_prompts_out'
    out_dir.mkdir(parents=True)
    atomic_json(out_dir / 'environment.json', env_data)
    ns = prompts.parse_config(
        prompts.common_parser('test'),
        ['--scale', 'full', '--allow_large_run', '--seed', '2026', '--output', str(out_dir)],
    )
    ns.m5_long_train_prompts = 0
    built = prompts.build_manifest(ns)
    cap_rep = phase_b_suite.verify_full_scale_prompt_capacity(built, 'full')
    assert cap_rep['status'] == 'passed'
    assert cap_rep['counts']['standard_train_prompts'] == 7409
    assert cap_rep['counts']['standard_dev_prompts'] == 64
    assert cap_rep['counts']['standard_test_selected'] == 128
    assert cap_rep['counts']['standard_test_total_heldout'] == 164
    assert cap_rep['counts']['long_dev_prompts'] == 32
    assert cap_rep['counts']['long_test_selected'] == 96


def test_phase_b_suite_smoke_all_cache_hit_and_upstream_immutability(tmp_path, monkeypatch):
    ref_dir = tmp_path / 'frozen_ref_artifacts'
    ref_dir.mkdir(parents=True)
    (ref_dir / 'm4').mkdir(parents=True)
    atomic_json(ref_dir / 'milestones.json', {'m4': 'passed'})
    atomic_json(ref_dir / 'm4' / 'summary.json', {'status': 'complete', 'scale': 'reference'})

    # Guard: --output cannot equal populated --reference_artifacts on non-smoke scales
    with pytest.raises(SystemExit):
        phase_b_suite.main([
            '--scale', 'reference', '--suite', 'ref_secondary_ablations',
            '--output', str(ref_dir), '--reference_artifacts', str(ref_dir),
        ])

    # Fingerprint invariance between `--scale reference` and `--scale full --allow_large_run`
    # for the 3 secondary reference ablations so Step 5 (`--suite all`) is a guaranteed cache hit!
    base_args_ref = SimpleNamespace(
        scale='reference', seed=2026, device='cuda:0', gpus='0,1,2,3,4,5,6,7',
        target='Qwen/Qwen3-8B', drafter='z-lab/Qwen3-8B-DFlash-b16',
        max_new_tokens=2048, long_max_new_tokens=24576, smoke_tokens=32,
        timing_repeats=3, timing_warmups=1, timing_prompts=0,
        bigram_eviction_arms=list(phase_b_suite.DEFAULT_BIGRAM_EVICTION_ARMS),
        bigram_k_ret=256, bigram_count_gate_kappa=12.0,
        full_vocab_lookahead_arms=list(phase_b_suite.DEFAULT_FULL_VOCAB_LOOKAHEAD_ARMS),
        lookahead_dense_checkpoint=None, lookahead_sparse_checkpoint=None, lookahead_checkpoint_scale=None,
        m5_beta_relax=1.0, m5_beta_candidates=[0.5, 1.0],
        pure_t1_tag='m5_pure_t1_calibration', block_shuffle_window=32,
        weight_decay=1e-2, fixed_checkpoint_scale=0.20,
    )
    base_args_full = copy.copy(base_args_ref)
    base_args_full.scale = 'full'
    for s_name in phase_b_suite.REF_SECONDARY_SUITES:
        assert (
            phase_b_suite._suite_config_fingerprint(s_name, base_args_ref, ref_dir)
            == phase_b_suite._suite_config_fingerprint(s_name, base_args_full, ref_dir)
        )

    out_root = tmp_path / 'phase_b_out'
    common_cli = [
        '--scale', 'smoke',
        '--output', str(out_root),
        '--reference_artifacts', str(ref_dir),
        '--timing_repeats', '2',
        '--timing_warmups', '1',
        '--smoke_tokens', '16',
        '--block_shuffle_window', '2',
    ]

    # Run Steps 1..4 individually, then Step 5 (`--suite all`) with compute entrypoints blocked!
    for step_suite in ('bigram_eviction', 'full_vocab_lookahead', 'pure_t1_fit', 'full_scale'):
        res = phase_b_suite.main([*common_cli, '--suite', step_suite])
        assert res['status'] == 'complete'
        assert res['suites'][step_suite]['status'] == 'complete'

    def _forbid(*_a, **_kw):
        raise AssertionError('Expected 100% cache hit in --suite all without recomputing!')

    monkeypatch.setattr(phase_b_suite, '_execute_closed_loop_stage', _forbid)
    monkeypatch.setattr(phase_b_suite, '_run_training_fit_isolated', _forbid)
    monkeypatch.setattr(phase_b_suite, '_bootstrap_full_scale_workspace', _forbid)
    master = phase_b_suite.main([*common_cli, '--suite', 'all'])
    assert master['status'] == 'complete'
    assert set(master['suites']) == {'bigram_eviction', 'full_vocab_lookahead', 'pure_t1_fit', 'full_scale'}
    assert master['upstream_reference_immutability']['status'] == 'intact'
    assert master['upstream_reference_immutability']['verified_files'] == 2

    # Verify M4 in full_scale_pipeline ran cleanly and upstream sentinel mutation is caught
    before = phase_b_suite._snapshot_upstream_hashes(ref_dir)
    atomic_json(ref_dir / 'milestones.json', {'m4': 'tampered'})
    with pytest.raises(RuntimeError, match='mutated'):
        phase_b_suite._verify_upstream_immutability(ref_dir, before)


def test_launch_bigram_k_ret_forwarding_and_worktree_env_bootstrap(tmp_path, monkeypatch):
    from edraft import launch

    with pytest.raises(SystemExit):
        launch.main(['--task', 'm4', '--scale', 'smoke', '--bigram_k_ret', '256'])
    with pytest.raises(SystemExit):
        launch.main(['--task', 'm5_timing', '--scale', 'smoke', '--bigram_k_ret', '0'])

    ws_timing = tmp_path / 'launch_kret_ws'
    eval_closed_loop.main([
        '--stage', 'm5', '--scale', 'smoke', '--mode', 'tune',
        '--device', 'cpu', '--output', str(ws_timing),
        '--m5_arms', 'plain_dflash', 'online_bigram', 'online_bigram_kret256',
        '--bigram_k_ret', '3',
    ])
    eval_closed_loop.main([
        '--stage', 'm5', '--scale', 'smoke', '--mode', 'evaluate',
        '--device', 'cpu', '--output', str(ws_timing),
        '--m5_arms', 'plain_dflash', 'online_bigram', 'online_bigram_kret256',
        '--bigram_k_ret', '3',
    ])
    launch.main([
        '--task', 'm5_timing', '--scale', 'smoke', '--gpus', '0',
        '--output', str(ws_timing), '--timing_repeats', '2', '--timing_warmups', '1',
        '--m5_arms', 'plain_dflash', 'online_bigram', 'online_bigram_kret256',
        '--bigram_k_ret', '3',
    ])
    timed = json.loads((ws_timing / 'm5' / 'timing.json').read_text())
    assert timed['status'] == 'complete'

    # Verify _bootstrap_full_scale_workspace seeds model_artifacts/dataset_artifacts from reference_artifacts
    # even when ROOT/artifacts/environment.json is absent!
    fake_ref = tmp_path / 'fake_ref_artifacts'
    fake_ref.mkdir(parents=True)
    atomic_json(fake_ref / 'environment.json', {
        'models': {'target': 'Qwen/Qwen3-8B', 'drafter': 'z-lab/Qwen3-8B-DFlash-b16'},
        'model_artifacts': {
            'Qwen/Qwen3-8B': {'revision': 'rev-target'},
            'z-lab/Qwen3-8B-DFlash-b16': {'revision': 'rev-drafter'},
        },
        'dataset_artifacts': {
            'openai/gsm8k': {'revision': 'rev-gsm'},
            'openai/openai_humaneval': {'revision': 'rev-he'},
            'HuggingFaceH4/MATH-500': {'revision': 'rev-math'},
        },
    })
    atomic_json(fake_ref / 'smoke.json', {'status': 'passed'})
    atomic_json(fake_ref / 'm0.json', {'status': 'passed', 'scale': 'reference', 'num_distinct_prompts': 20, 'comparisons': {'upstream_local': {'mismatched_tokens': 0}}})
    atomic_json(fake_ref / 'm0_numerics.json', {'status': 'passed', 'head_reconstruction': {'passed': True}, 'findings': [], 'model_revisions': {'Qwen/Qwen3-8B': 'rev-target', 'z-lab/Qwen3-8B-DFlash-b16': 'rev-drafter'}})
    full_ws = tmp_path / 'bootstrapped_full'
    atomic_json(full_ws / 'prompts.json', _make_full_scale_synthetic_manifest())
    fake_root = tmp_path / 'empty_repo_root'
    fake_root.mkdir(parents=True)
    monkeypatch.setattr('edraft.runtime.ROOT', fake_root)
    ns = SimpleNamespace(
        scale='full', seed=2026, device='cpu', target='Qwen/Qwen3-8B',
        drafter='z-lab/Qwen3-8B-DFlash-b16', allow_large_run=True,
        allow_download=False, resume=True, output=full_ws,
    )
    cap = phase_b_suite._bootstrap_full_scale_workspace(ns, fake_ref, full_ws, 'full', [0])
    assert cap['status'] == 'passed'
    boot_env = json.loads((full_ws / 'environment.json').read_text())
    assert boot_env['model_artifacts']['Qwen/Qwen3-8B']['revision'] == 'rev-target'
    assert boot_env['dataset_artifacts']['openai/gsm8k']['revision'] == 'rev-gsm'


def test_eval_bug_3_fastpath_calibration_reconcile_and_suite_forwarding(tmp_path):
    # 1. Exact EVAL-BUG-3 reproduction & verification on FastPath-enabled timing_calibration + summarize
    ws = tmp_path / 'eval_bug3_ws'
    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'all',
        '--smoke_tokens', '8', '--timing_repeats', '2', '--timing_warmups', '1',
        '--no_analytic_online_grad',
        '--output', str(ws),
    ])
    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'timing_calibration',
        '--calibration_repeats', '2', '--smoke_tokens', '8', '--timing_warmups', '1',
        '--no_analytic_online_grad',
        '--output', str(ws),
    ])
    # Reconcile WITH matching --no_analytic_online_grad flag:
    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'summarize',
        '--reconcile_sharded_timing',
        '--calibration_repeats', '2', '--smoke_tokens', '8', '--timing_warmups', '1',
        '--no_analytic_online_grad',
        '--output', str(ws),
    ])
    cal_json = json.loads((ws / 'm3' / 'timing_isolation_calibration.json').read_text())
    assert cal_json['status'] == 'complete'
    assert cal_json['reconciled_with_sharded_timing'] is True
    assert cal_json['fastpath']['analytic_online_grad'] is False

    # Reconcile WITHOUT repeating the CLI flag (inferred from timing.json['fastpath']):
    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'summarize',
        '--reconcile_sharded_timing',
        '--calibration_repeats', '2', '--smoke_tokens', '8', '--timing_warmups', '1',
        '--output', str(ws),
    ])
    # Conflicting CLI fastpath flag (--bf16_gemm) must still be rejected:
    with pytest.raises(ValueError, match='fastpath'):
        eval_closed_loop.main([
            '--scale', 'smoke', '--stage', 'm3', '--mode', 'summarize',
            '--reconcile_sharded_timing',
            '--calibration_repeats', '2', '--smoke_tokens', '8', '--timing_warmups', '1',
            '--bf16_gemm',
            '--output', str(ws),
        ])

    # 2. phase_b_suite fastpath fingerprint separation + sharded forwarding
    ref_dir = tmp_path / 'ref_fp'
    ref_dir.mkdir(parents=True)
    base_ns = SimpleNamespace(
        scale='smoke', seed=2026, device='cpu', gpus='0,1',
        target='Qwen/Qwen3-8B', drafter='z-lab/Qwen3-8B-DFlash-b16',
        max_new_tokens=2048, long_max_new_tokens=24576, smoke_tokens=8,
        timing_repeats=2, timing_warmups=1, timing_prompts=0,
        bigram_eviction_arms=['plain_dflash', 'online_bigram', 'online_bigram_kret256'],
        bigram_k_ret=4, bigram_count_gate_kappa=12.0,
    )
    fp_default = phase_b_suite._suite_config_fingerprint('bigram_eviction', base_ns, ref_dir)
    fp_ns = copy.copy(base_ns)
    fp_ns.analytic_online_grad = False
    fp_active = phase_b_suite._suite_config_fingerprint('bigram_eviction', fp_ns, ref_dir)
    assert fp_default != fp_active

    suite_out = tmp_path / 'suite_fp_out'
    res = phase_b_suite.main([
        '--scale', 'smoke', '--suite', 'bigram_eviction',
        '--gpus', '0,1',
        '--output', str(suite_out), '--reference_artifacts', str(ref_dir),
        '--smoke_tokens', '8', '--timing_repeats', '2', '--timing_warmups', '1',
        '--bigram_eviction_arms', 'plain_dflash', 'online_bigram', 'online_bigram_kret256',
        '--bigram_k_ret', '4',
        '--no_analytic_online_grad',
    ])
    assert res['status'] == 'complete'
    t_json = json.loads(Path(res['suites']['bigram_eviction']['artifacts']['timing_json']).read_text())
    assert t_json['fastpath']['analytic_online_grad'] is False


def test_phase_b_relpath_and_pilot_timing_prompts_semantics(tmp_path, monkeypatch):
    # 1. PHASE-B-PILOT-PROMPTS-BUG-2: _select_test_prompts(timing_prompts=K) selects K per condition
    manifest = _make_full_scale_synthetic_manifest()
    all_test, sel_1, is_pilot_1 = phase_b_suite._select_test_prompts(manifest, 'full', 1)
    assert len(all_test) == 192  # 128 std + 64 long
    assert len(sel_1) == 2
    assert {p['condition'] for p in sel_1} == {'long_thinking', 'standard_nonthinking'}
    assert is_pilot_1 is True

    _, sel_128, is_pilot_128 = phase_b_suite._select_test_prompts(manifest, 'full', 128)
    assert len(sel_128) == 192
    assert is_pilot_128 is False

    # 2. PHASE-B-RELPATH-BUG-1 + pilot_prompts=1 (covers both conditions in pure_t1_fit!) + pilot_prompts=2 resume
    cwd_sub = tmp_path / 'caller_cwd'
    cwd_sub.mkdir(parents=True)
    (cwd_sub / 'rel_ref').mkdir(parents=True)
    monkeypatch.chdir(cwd_sub)

    assert not (ROOT / 'rel_out').exists()
    res_p1 = phase_b_suite.main([
        '--scale', 'smoke', '--suite', 'pure_t1_fit',
        '--output', 'rel_out', '--reference_artifacts', 'rel_ref',
        '--smoke_tokens', '8', '--timing_repeats', '2', '--timing_warmups', '1',
        '--timing_prompts', '1', '--block_shuffle_window', '2',
    ])
    assert not (ROOT / 'rel_out').exists()
    assert (cwd_sub / 'rel_out' / 'phase_b_summary.json').is_file()
    assert res_p1['status'] == 'pilot_complete'
    cl_var = res_p1['suites']['pure_t1_fit']['variants'][0]['closed_loop']
    regimes = set(cl_var['plain_dflash']['by_regime'].keys())
    assert 'long_thinking::thinking_recommended' in regimes
    assert 'standard_nonthinking::recommended' in regimes or 'standard_nonthinking::nonthinking_recommended' in regimes

    # timing_prompts=2 on smoke covers all 2+2=4 test prompts -> status='complete', and --resume must hit cache!
    res_p2 = phase_b_suite.main([
        '--scale', 'smoke', '--suite', 'bigram_eviction',
        '--output', 'rel_out_2', '--reference_artifacts', 'rel_ref',
        '--smoke_tokens', '8', '--timing_repeats', '2', '--timing_warmups', '1',
        '--timing_prompts', '2',
        '--bigram_eviction_arms', 'plain_dflash', 'online_bigram',
    ])
    assert res_p2['status'] == 'complete'

    def _forbid(*_a, **_kw):
        raise AssertionError('Expected _can_resume_report cache hit when timing_prompts>=max_cond_count!')

    monkeypatch.setattr(phase_b_suite, '_execute_closed_loop_stage', _forbid)
    res_p2_cached = phase_b_suite.main([
        '--scale', 'smoke', '--suite', 'bigram_eviction',
        '--output', 'rel_out_2', '--reference_artifacts', 'rel_ref',
        '--smoke_tokens', '8', '--timing_repeats', '2', '--timing_warmups', '1',
        '--timing_prompts', '2',
        '--bigram_eviction_arms', 'plain_dflash', 'online_bigram',
    ])
    assert res_p2_cached['status'] == 'complete'


def test_fast_decode_wiring_across_collect_eval_launch_and_phase_b_suite(tmp_path):
    from edraft import collect, launch

    # 1. collect.parser() accepts --fast_decode (default False)
    c_args_def = collect.prepare_args(collect.parser().parse_args(['--scale', 'smoke', '--output', str(tmp_path / 'c0')]))
    assert c_args_def.fast_decode is False
    c_args_fd = collect.prepare_args(collect.parser().parse_args(['--scale', 'smoke', '--output', str(tmp_path / 'c1'), '--fast_decode']))
    assert c_args_fd.fast_decode is True

    # 2. eval_closed_loop + timing_calibration + reconcile + launch --fast_decode end-to-end
    ws = tmp_path / 'fd_ws'
    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'tune',
        '--device', 'cpu', '--output', str(ws), '--fast_decode',
    ])
    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'evaluate',
        '--device', 'cpu', '--output', str(ws), '--fast_decode',
    ])
    launch.main([
        '--task', 'm3_timing', '--scale', 'smoke', '--gpus', '0',
        '--output', str(ws), '--timing_repeats', '2', '--timing_warmups', '1',
        '--fast_decode',
    ])
    t_rep = json.loads((ws / 'm3' / 'timing.json').read_text())
    assert t_rep['status'] == 'complete'
    assert t_rep['fastpath']['fast_decode'] is True

    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'timing_calibration',
        '--calibration_repeats', '2', '--timing_warmups', '1',
        '--output', str(ws), '--fast_decode',
    ])
    eval_closed_loop.main([
        '--scale', 'smoke', '--stage', 'm3', '--mode', 'summarize',
        '--reconcile_sharded_timing', '--calibration_repeats', '2', '--timing_warmups', '1',
        '--output', str(ws), '--fast_decode',
    ])
    cal_rep = json.loads((ws / 'm3' / 'timing_isolation_calibration.json').read_text())
    assert cal_rep['status'] == 'complete'
    assert cal_rep['fastpath']['fast_decode'] is True

    # 3. phase_b_suite --fast_decode forwards across full_scale (calibrate + collect + m3 + m4)
    ref_dir = tmp_path / 'fd_ref'
    ref_dir.mkdir(parents=True)
    out_fd = tmp_path / 'fd_suite'
    res = phase_b_suite.main([
        '--scale', 'smoke', '--suite', 'full_scale',
        '--output', str(out_fd), '--reference_artifacts', str(ref_dir),
        '--smoke_tokens', '8', '--timing_repeats', '2', '--timing_warmups', '1',
        '--fast_decode',
    ])
    assert res['status'] == 'complete'
    m4_tim = json.loads(Path(res['suites']['full_scale']['m4_closed_loop']['timing_json']).read_text())
    assert m4_tim['fastpath']['fast_decode'] is True


def test_dual_offline_kl_sharding_load_balance_resume_and_parity(tmp_path, monkeypatch):
    import torch
    from edraft import train

    # 1. Verify _dual_offline_kl_regime_weights (default fallback, token_counts without 0/0.0 double count, and real sidecars)
    empty_ws = tmp_path / 'empty_ws'
    empty_ws.mkdir()
    assert phase_b_suite._dual_offline_kl_regime_weights(empty_ws) == (13209, 7687)

    tc_ws = tmp_path / 'tc_ws'
    tc_ws.mkdir()
    atomic_json(tc_ws / 'm1_audit.json', {
        'token_counts': {
            'standard_nonthinking/dev/T=0.0': 20280,
            'standard_nonthinking/dev/T=0': 20280,
            'standard_nonthinking/dev/T=0.7': 19932,
            'standard_nonthinking/test/T=0.0': 19081,
            'standard_nonthinking/test/T=0.7': 19143,
            'standard_nonthinking/dev/T=1.0': 19788,
            'standard_nonthinking/dev/T=1': 19788,
            'standard_nonthinking/test/T=1.0': 19054,
        },
    })
    assert phase_b_suite._dual_offline_kl_regime_weights(tc_ws) == (78436, 38842)

    real_ref = Path('/home/catid/drafter/artifacts')
    if (real_ref / 'm1_audit.json').is_file():
        assert phase_b_suite._dual_offline_kl_regime_weights(real_ref) == (13209, 7687)

    # 2. Verify _plan_dual_offline_kl_shards across 10 checkpoints (5 variants) and 8 GPUs
    ten_ckpts = {f'var{i // 2}::{"dense" if i % 2 == 0 else "dense_sparse"}': tmp_path / f'c{i}.pt' for i in range(10)}
    shards_10 = phase_b_suite._plan_dual_offline_kl_shards(
        ten_ckpts, list(range(8)), primary_weight=13209, pure_t1_weight=7687,
    )
    assert len(shards_10) == 8
    seen_pairs = [(t['regime'], t['key']) for _, _, tasks, _ in shards_10 for t in tasks]
    assert len(seen_pairs) == 20 and len(set(seen_pairs)) == 20
    max_load_10 = max(est for _, _, _, est in shards_10)
    total_load_10 = 10 * (13209 + 7687)
    assert max_load_10 == 28583
    assert total_load_10 / max_load_10 > 7.30  # 7.31x speedup vs 1-GPU sequential (vs 5.0x naive ckpt sharding)
    assert {t['regime'] for t in shards_10[0][2]} == {'primary'}
    assert {t['regime'] for t in shards_10[1][2]} == {'primary'}

    # 8 checkpoints across 8 GPUs -> exact 1x primary + 1x pure_t1 on every rank (8.00x = 100% balance)
    eight_ckpts = {k: v for idx, (k, v) in enumerate(ten_ckpts.items()) if idx < 8}
    shards_8 = phase_b_suite._plan_dual_offline_kl_shards(
        eight_ckpts, list(range(8)), primary_weight=13209, pure_t1_weight=7687,
    )
    assert [est for _, _, _, est in shards_8] == [20896] * 8

    # Fewer tasks than GPUs spawns only needed ranks
    one_ckpt = {'only::dense': tmp_path / 'only.pt'}
    shards_1 = phase_b_suite._plan_dual_offline_kl_shards(one_ckpt, list(range(8)))
    assert len(shards_1) == 2

    # 3. Build a real mini reference-layout workspace on CPU to verify 0-ULP parity, shared_head, and per-rank resume
    from safetensors.torch import save_file as save_safetensors

    ws = tmp_path / 'dual_kl_ref_ws'
    ws.mkdir(parents=True)
    train.main(['--scale', 'smoke', '--device', 'cpu', '--seed', '2026', '--output', str(ws)])
    _, _, smoke_head, smoke_bias, head_meta = train.smoke_bundle(
        SimpleNamespace(
            output=ws, seed=2026, history_depth=4,
            allow_pure_t1_ablation=True, allow_long_oracle_fit=False,
        )
    )

    shard_paths = sorted(
        str(p.resolve())
        for sub in ('m2_smoke_traces', 'm2_smoke_t1_traces')
        for p in (ws / sub).glob('rank*/*.pt')
    )
    assert len(shard_paths) > 0
    (ws / 'traces').symlink_to(ws / 'm2_smoke_traces')
    atomic_json(ws / 'm1_audit.json', {'scale': 'reference', 'status': 'passed', 'shards': shard_paths})

    # Materialize real safetensors head snapshot + environment.json so load_frozen_head runs unmocked
    snap_dir = ws / 'fake_head_snap' / head_meta['revision']
    snap_dir.mkdir(parents=True)
    atomic_json(snap_dir / 'config.json', {'tie_word_embeddings': bool(head_meta['tied_weights'])})
    head_tensors = {'lm_head.weight': smoke_head.detach().cpu()}
    if smoke_bias is not None:
        head_tensors['lm_head.bias'] = smoke_bias.detach().cpu()
    save_safetensors(head_tensors, str(snap_dir / 'model.safetensors'))
    atomic_json(ws / 'environment.json', {
        'model_artifacts': {
            head_meta['artifact_id']: {
                'revision': head_meta['revision'],
                'path': str(snap_dir.resolve()),
            }
        }
    })

    # Create 5 variants = 10 distinct scaled/unscaled checkpoints matching pure_t1_fit
    base_d = ws / 'm2_smoke' / 'dense.pt'
    base_s = ws / 'm2_smoke' / 'dense_sparse.pt'
    ckpts_dir = ws / 'ckpts'
    ckpt_map: dict[str, Path] = {
        'v0_unscaled::dense': base_d,
        'v0_unscaled::dense_sparse': base_s,
    }
    for idx, sc in enumerate((0.20, 0.50, 0.75, 1.15), start=1):
        d_dst = ckpts_dir / f'v{idx}_dense.pt'
        s_dst = ckpts_dir / f'v{idx}_sparse.pt'
        train.scale_adapter_checkpoint(base_d, d_dst, sc)
        train.scale_adapter_checkpoint(base_s, s_dst, sc)
        ckpt_map[f'v{idx}::dense'] = d_dst
        ckpt_map[f'v{idx}::dense_sparse'] = s_dst
    assert len(ckpt_map) == 10

    # Baseline 1-process CPU evaluation
    cpu_args = SimpleNamespace(
        scale='reference', seed=2026, device='cpu',
        target='Qwen/Qwen3-8B', drafter='z-lab/Qwen3-8B-DFlash-b16',
        resume=True,
    )
    ref_prim, ref_t1 = phase_b_suite._evaluate_dual_offline_kl_batch(
        cpu_args, ws, 'reference', ckpt_map, gpu_index=0, gpus=[0],
    )

    # Monkeypatch _spawn_parallel_workers to run _worker_dual_offline_kl in-process on CPU while tracking head loads & spawns
    spawn_history: list[list[tuple[int, int]]] = []
    head_loads_per_worker: list[int] = []
    real_load_head = phase_b_suite.load_frozen_head

    def _fake_spawn(worker_specs, logdir, label, *, scale, device_str, omp_threads=1):
        del logdir, label, scale, device_str, omp_threads
        spawn_history.append([(r, g) for r, g, _ in worker_specs])
        for _rank, _gpu, cmd in worker_specs:
            spec_idx = cmd.index('--_worker_spec') + 1
            out_idx = cmd.index('--_worker_out') + 1
            calls = [0]

            def _counting_head(*a, **kw):
                calls[0] += 1
                return real_load_head(*a, **kw)

            monkeypatch.setattr(phase_b_suite, 'load_frozen_head', _counting_head)
            try:
                phase_b_suite._worker_dual_offline_kl(
                    SimpleNamespace(device='cpu', seed=2026),
                    Path(cmd[spec_idx]),
                    Path(cmd[out_idx]),
                )
            finally:
                monkeypatch.setattr(phase_b_suite, 'load_frozen_head', real_load_head)
            head_loads_per_worker.append(calls[0])
        return {'status': 'complete'}

    monkeypatch.setattr(phase_b_suite, '_spawn_parallel_workers', _fake_spawn)

    cuda_args = SimpleNamespace(
        scale='reference', seed=2026, device='cuda:0',
        target='Qwen/Qwen3-8B', drafter='z-lab/Qwen3-8B-DFlash-b16',
        resume=True,
    )
    sharded_prim, sharded_t1 = phase_b_suite._evaluate_dual_offline_kl_batch(
        cuda_args, ws, 'reference', ckpt_map, gpu_index=0, gpus=list(range(8)),
    )
    assert len(spawn_history) == 1
    assert spawn_history[0] == [(r, r) for r in range(8)]
    # Every worker loads the frozen head ONCE (shared across primary & pure_t1 on mixed-regime workers!)
    assert head_loads_per_worker == [1] * 8
    assert list(sharded_prim.keys()) == list(ref_prim.keys())
    assert list(sharded_t1.keys()) == list(ref_t1.keys())
    assert sharded_prim == ref_prim
    assert sharded_t1 == ref_t1

    # Full cache hit on --resume: 0 workers spawned!
    spawn_history.clear()
    cached_prim, cached_t1 = phase_b_suite._evaluate_dual_offline_kl_batch(
        cuda_args, ws, 'reference', ckpt_map, gpu_index=0, gpus=list(range(8)),
    )
    assert spawn_history == []
    assert cached_prim == ref_prim and cached_t1 == ref_t1

    # Partial resume: delete ONLY rank 3's result file -> spawns ONLY rank 3!
    rank3_files = list((ws / 'dual_offline_kl_eval').glob('result-rank3-*.json'))
    assert len(rank3_files) == 1
    rank3_files[0].unlink()
    spawn_history.clear()
    part_prim, part_t1 = phase_b_suite._evaluate_dual_offline_kl_batch(
        cuda_args, ws, 'reference', ckpt_map, gpu_index=0, gpus=list(range(8)),
    )
    assert spawn_history == [[(3, 3)]]
    assert part_prim == ref_prim and part_t1 == ref_t1

    # Checkpoint modification invalidates ONLY the shards containing that checkpoint!
    train.scale_adapter_checkpoint(base_d, ckpts_dir / 'v4_dense.pt', 0.35)
    spawn_history.clear()
    upd_prim, upd_t1 = phase_b_suite._evaluate_dual_offline_kl_batch(
        cuda_args, ws, 'reference', ckpt_map, gpu_index=0, gpus=list(range(8)),
    )
    assert len(spawn_history) == 1
    assert 1 <= len(spawn_history[0]) <= 2  # at most 2 ranks (primary + pure_t1 for v4::dense) recomputed!
    ref_prim_upd, ref_t1_upd = phase_b_suite._evaluate_dual_offline_kl_batch(
        cpu_args, ws, 'reference', ckpt_map, gpu_index=0, gpus=[0],
    )
    assert upd_prim == ref_prim_upd and upd_t1 == ref_t1_upd

    # --no_resume (resume=False) forces all 8 ranks to recompute
    no_res_args = copy.copy(cuda_args)
    no_res_args.resume = False
    spawn_history.clear()
    nr_prim, nr_t1 = phase_b_suite._evaluate_dual_offline_kl_batch(
        no_res_args, ws, 'reference', ckpt_map, gpu_index=0, gpus=list(range(8)),
    )
    assert spawn_history == [[(r, r) for r in range(8)]]
    assert nr_prim == ref_prim_upd and nr_t1 == ref_t1_upd

    # Legacy `checkpoints` spec + monkeypatched `_prepare_offline_eval_context -> None` compatibility
    monkeypatch.setattr(phase_b_suite, 'apply_gpu_cpu_affinity', lambda *_a, **_kw: {})
    monkeypatch.setattr(phase_b_suite, 'assert_idle', lambda *_a, **_kw: {'ok': True})
    monkeypatch.setattr(phase_b_suite, '_prepare_offline_eval_context', lambda *_a, **_kw: None)
    monkeypatch.setattr(phase_b_suite, '_offline_kl_summary', lambda *_a, **_kw: {'dev': {'weighted_mean_kl': 1.0}, 'test': {'weighted_mean_kl': 1.0}})
    leg_spec = tmp_path / 'legacy_spec.json'
    leg_out = tmp_path / 'legacy_out.json'
    atomic_json(leg_spec, {
        'workspace': str(ws),
        'effective_scale': 'smoke',
        'physical_gpu': 0,
        'checkpoints': {'leg::dense': str(base_d)},
    })
    phase_b_suite._worker_dual_offline_kl(SimpleNamespace(device='cpu', seed=2026), leg_spec, leg_out)
    leg_payload = json.loads(leg_out.read_text())
    assert 'leg::dense' in leg_payload['primary_heldout_kl']
    assert 'leg::dense' in leg_payload['pure_t1_heldout_kl']


def test_bootstrap_full_scale_heals_interrupted_m0_copy(tmp_path, monkeypatch):
    fake_ref = tmp_path / 'ref_art'
    (fake_ref / 'm0').mkdir(parents=True)
    atomic_json(fake_ref / 'environment.json', {
        'models': {'target': 'Qwen/Qwen3-8B', 'drafter': 'z-lab/Qwen3-8B-DFlash-b16'},
        'model_artifacts': {
            'Qwen/Qwen3-8B': {'revision': 'rev-t'},
            'z-lab/Qwen3-8B-DFlash-b16': {'revision': 'rev-d'},
        },
        'dataset_artifacts': {
            'openai/gsm8k': {'revision': 'rev-g'},
            'openai/openai_humaneval': {'revision': 'rev-h'},
            'HuggingFaceH4/MATH-500': {'revision': 'rev-m'},
        },
    })
    atomic_json(fake_ref / 'smoke.json', {'status': 'passed'})
    atomic_json(fake_ref / 'm0.json', {'status': 'passed', 'scale': 'reference', 'num_distinct_prompts': 3})
    atomic_json(fake_ref / 'm0_numerics.json', {'status': 'passed'})
    for i in range(3):
        atomic_json(fake_ref / 'm0' / f'row_{i:02d}.json', {'idx': i})

    # Simulate crash after m0.json + m0_numerics.json were copied, before m0/*.json and m0_provenance.json finished
    full_ws = tmp_path / 'full_ws_interrupted'
    (full_ws / 'm0').mkdir(parents=True)
    atomic_json(full_ws / 'm0.json', {'status': 'passed', 'scale': 'reference', 'num_distinct_prompts': 3})
    atomic_json(full_ws / 'm0_numerics.json', {'status': 'passed'})
    atomic_json(full_ws / 'prompts.json', _make_full_scale_synthetic_manifest())

    fake_root = tmp_path / 'empty_root'
    fake_root.mkdir(parents=True)
    monkeypatch.setattr('edraft.runtime.ROOT', fake_root)
    ns = SimpleNamespace(
        scale='full', seed=2026, device='cpu', target='Qwen/Qwen3-8B',
        drafter='z-lab/Qwen3-8B-DFlash-b16', allow_large_run=True,
        allow_download=False, resume=True, output=full_ws,
    )
    cap = phase_b_suite._bootstrap_full_scale_workspace(ns, fake_ref, full_ws, 'full', [0])
    assert cap['status'] == 'passed'
    assert (full_ws / 'm0_provenance.json').is_file()
    assert len(list((full_ws / 'm0').glob('*.json'))) == 3
    prov = json.loads((full_ws / 'm0_provenance.json').read_text())
    assert prov['m0_row_files_copied'] == 3


def test_pure_t1_closed_loop_shards_quadratic_lpt_balance_and_per_rank_resume(tmp_path, monkeypatch):
    from edraft import train

    # 1. Real reference artifacts check (read-only): <1% quadratic cost spread & exact 4 long prompts/rank across 8 GPUs
    real_ref = Path('/home/catid/drafter/artifacts')
    if (real_ref / 'prompts.json').is_file() and (real_ref / 'm4' / 'timing' / 'sharding_plan.json').is_file():
        pm_real = json.loads((real_ref / 'prompts.json').read_text())
        _, sel_real, _ = phase_b_suite._select_test_prompts(pm_real, 'reference', 0)
        eff_w, shards_real, meta_real = phase_b_suite._plan_closed_loop_prompt_shards(
            sel_real, pm_real, real_ref, 8, timing_repeats=3,
        )
        assert eff_w == 8
        assert meta_real['weight_source'] == 'closed_loop_sharding_plan_quadratic_lpt'
        all_assigned = [p['prompt_id'] for r in range(8) for p in shards_real[r]]
        assert len(all_assigned) == 96 and len(set(all_assigned)) == 96
        for r in range(8):
            assert sum(1 for p in shards_real[r] if p['condition'].startswith('long_')) == 4
            assert sum(1 for p in shards_real[r] if p['condition'] == 'standard_nonthinking') >= 1
        loads_real = [meta_real['rank_estimated_loads'][str(r)] for r in range(8)]
        spread = (max(loads_real) - min(loads_real)) / (sum(loads_real) / 8.0)
        assert spread < 0.01  # 0.46% spread vs 46.19% unweighted manifest snake!

    # 2. Synthetic m1_audit.json fallback with extreme 24,576 vs 1,000 outlier: count guard keeps <= ceil(N_long/W) per rank
    syn_manifest = _make_full_scale_synthetic_manifest()
    _, sel_full, _ = phase_b_suite._select_test_prompts(syn_manifest, 'full', 0)
    m1_ws = tmp_path / 'm1_only_ws'
    m1_ws.mkdir()
    gens = []
    for idx, p in enumerate(sel_full):
        if p['condition'].startswith('long_'):
            rl = 24576 if idx % 8 == 0 else (1200 + 100 * (idx % 7))
        else:
            rl = 200 + 15 * (idx % 10)
        gens.append({'prompt_id': p['prompt_id'], 'split': 'test', 'realized_length': rl})
    atomic_json(m1_ws / 'm1_audit.json', {'scale': 'full', 'status': 'passed', 'generations': gens})
    eff_w_m1, shards_m1, meta_m1 = phase_b_suite._plan_closed_loop_prompt_shards(
        sel_full, syn_manifest, m1_ws, 8, timing_repeats=3,
    )
    assert eff_w_m1 == 8
    assert meta_m1['weight_source'] == 'm1_audit_realized_tokens_lpt'
    for r in range(8):
        # 64 selected long prompts across 8 GPUs -> exact 8 long prompts per rank despite 24,576 outliers!
        assert sum(1 for p in shards_m1[r] if p['condition'].startswith('long_')) == 8

    # Pilot 1-per-condition (`2` prompts across `2` GPUs) leaves zero empty ranks
    _, sel_pilot2, _ = phase_b_suite._select_test_prompts(syn_manifest, 'full', 1)
    eff_w_p2, shards_p2, _ = phase_b_suite._plan_closed_loop_prompt_shards(
        sel_pilot2, syn_manifest, m1_ws, 2, timing_repeats=2,
    )
    assert eff_w_p2 == 2 and all(len(shards_p2[r]) == 1 for r in range(2))

    # 3. End-to-end _evaluate_variants_closed_loop per-rank --resume & stale higher-rank cleanup
    ws = tmp_path / 'var_cl_ws'
    ws.mkdir()
    train.main(['--scale', 'smoke', '--device', 'cpu', '--seed', '2026', '--output', str(ws)])
    variants = [
        ('v_base', ws / 'm2_smoke' / 'dense.pt', ws / 'm2_smoke' / 'dense_sparse.pt'),
    ]
    base_args = SimpleNamespace(
        scale='smoke', seed=2026, device='cpu',
        target='Qwen/Qwen3-8B', drafter='z-lab/Qwen3-8B-DFlash-b16',
        timing_warmups=1, timing_repeats=2, timing_prompts=0,
        smoke_tokens=8, max_new_tokens=64, long_max_new_tokens=128,
        resume=True,
    )
    st_1gpu, out_1gpu = phase_b_suite._evaluate_variants_closed_loop(base_args, ws, 'smoke', variants, [0])
    assert st_1gpu == 'complete'

    # Plant a stale rank-3 file to verify stale ranks >= eff_w are purged before globbing
    stale_dir = ws / 'pure_t1_closed_loop_shards'
    stale_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(stale_dir / 'result-rank3.json', {'stale': True})
    atomic_json(stale_dir / 'spec-rank3.json', {'stale': True})

    real_spawn = phase_b_suite._spawn_parallel_workers
    spawned_ranks: list[list[int]] = []

    def _tracking_spawn(worker_specs, *a, **kw):
        spawned_ranks.append([r for r, _, _ in worker_specs])
        return real_spawn(worker_specs, *a, **kw)

    monkeypatch.setattr(phase_b_suite, '_spawn_parallel_workers', _tracking_spawn)
    st_2gpu, out_2gpu = phase_b_suite._evaluate_variants_closed_loop(base_args, ws, 'smoke', variants, [0, 1])
    assert st_2gpu == 'complete'
    assert spawned_ranks == [[0, 1]]
    assert not (stale_dir / 'result-rank3.json').exists()
    assert not (stale_dir / 'spec-rank3.json').exists()
    assert (stale_dir / 'sharding_plan.json').is_file()
    assert out_2gpu['v_base']['dense']['mean_accepted_length'] == pytest.approx(
        out_1gpu['v_base']['dense']['mean_accepted_length']
    )

    # Full cache hit on --resume: 0 workers spawned
    spawned_ranks.clear()
    st_cached, out_cached = phase_b_suite._evaluate_variants_closed_loop(base_args, ws, 'smoke', variants, [0, 1])
    assert st_cached == 'complete'
    assert spawned_ranks == []
    assert out_cached == out_2gpu

    # Partial resume: delete ONLY result-rank1.json -> spawns ONLY rank 1 while rank 0 stays untouched!
    r0_before = (stale_dir / 'result-rank0.json').read_text()
    (stale_dir / 'result-rank1.json').unlink()
    spawned_ranks.clear()
    st_part, out_part = phase_b_suite._evaluate_variants_closed_loop(base_args, ws, 'smoke', variants, [0, 1])
    assert st_part == 'complete'
    assert spawned_ranks == [[1]]
    assert (stale_dir / 'result-rank0.json').read_text() == r0_before
    for sub in ('plain_dflash', 'dense', 'dense_sparse'):
        assert out_part['v_base'][sub]['requests'] == out_2gpu['v_base'][sub]['requests']
        assert out_part['v_base'][sub]['primary_requests'] == out_2gpu['v_base'][sub]['primary_requests']
        assert out_part['v_base'][sub]['mean_accepted_length'] == out_2gpu['v_base'][sub]['mean_accepted_length']
    assert (
        out_part['v_base']['dense']['paired_accepted_length_delta_vs_plain']
        == out_2gpu['v_base']['dense']['paired_accepted_length_delta_vs_plain']
    )


def test_pure_t1_resume_mtime_bug1_and_1worker_stale_shard_cleanup(tmp_path, monkeypatch):
    import os
    import shutil

    ref_dir = tmp_path / 'ref_dir'
    ref_dir.mkdir(parents=True)
    out_dir = tmp_path / 'phase_b_out'
    cli_2gpu = [
        '--scale', 'smoke',
        '--suite', 'pure_t1_fit',
        '--gpus', '0,1',
        '--output', str(out_dir),
        '--reference_artifacts', str(ref_dir),
        '--smoke_tokens', '8',
        '--timing_repeats', '2',
        '--timing_warmups', '1',
        '--block_shuffle_window', '2',
    ]

    # 1. Initial full 2-worker run via top-level phase_b_suite.main
    res1 = phase_b_suite.main(cli_2gpu)
    assert res1['status'] == 'complete'

    abl_root = out_dir / 'reference_secondary_ablations'
    report_path = abl_root / 'pure_t1_fit.json'
    ws = (abl_root / 'pure_t1_workspace').resolve()
    shard_dir = ws / 'pure_t1_closed_loop_shards'
    assert report_path.is_file()
    assert (shard_dir / 'result-rank0.json').is_file()
    assert (shard_dir / 'result-rank1.json').is_file()

    # 2. Simulate partial crash before pure_t1_fit.json wrote (result-rank1.json missing) + mtime_ns bump
    r0_bytes_before = (shard_dir / 'result-rank0.json').read_text()
    report_path.unlink()
    (shard_dir / 'result-rank1.json').unlink()
    for p_touch in (ws / 'm2_smoke' / 'dense.pt', ws / 'baseline_m2_scaled' / 'dense_scale0p20.pt'):
        st = p_touch.stat()
        os.utime(p_touch, ns=(st.st_atime_ns, st.st_mtime_ns + 60_000_000_000))

    real_spawn = phase_b_suite._spawn_parallel_workers
    spawn_calls: list[tuple[str, list[int]]] = []
    kl_summary_calls = [0]
    real_kl_sum = phase_b_suite._offline_kl_summary

    def _spy_spawn(worker_specs, logdir, label, **kw):
        spawn_calls.append((label, [r for r, _, _ in worker_specs]))
        return real_spawn(worker_specs, logdir, label, **kw)

    def _spy_kl(*a, **kw):
        kl_summary_calls[0] += 1
        return real_kl_sum(*a, **kw)

    monkeypatch.setattr(phase_b_suite, '_spawn_parallel_workers', _spy_spawn)
    monkeypatch.setattr(phase_b_suite, '_offline_kl_summary', _spy_kl)

    res2 = phase_b_suite.main(cli_2gpu)
    assert res2['status'] == 'complete'
    # 0 dual_offline_kl_eval recomputations AND only closed-loop rank 1 respawned!
    assert kl_summary_calls[0] == 0
    assert spawn_calls == [('pure-t1-eval', [1])]
    assert (shard_dir / 'result-rank0.json').read_text() == r0_bytes_before

    # 3. Also verify CUDA-branch (`eff_scale='reference', device='cuda:0'`) partial resume inside run_pure_t1_fit
    #    on mtime bump: 0 `dual-offline-kl-*` workers spawned and only `[1]` `pure-t1-eval` worker spawned!
    from safetensors.torch import save_file as save_safetensors
    from edraft import train

    cuda_ref = tmp_path / 'cuda_ref'
    cuda_ref.mkdir()
    cuda_abl = tmp_path / 'cuda_abl'
    cuda_abl.mkdir()
    cuda_args = SimpleNamespace(
        scale='reference', seed=2026, device='cuda:0', gpus='0,1',
        target='Qwen/Qwen3-8B', drafter='z-lab/Qwen3-8B-DFlash-b16',
        max_new_tokens=64, long_max_new_tokens=128, smoke_tokens=8,
        timing_repeats=2, timing_warmups=1, timing_prompts=1,
        pure_t1_tag='m5_pure_t1_calibration', block_shuffle_window=2,
        weight_decay=1e-2, fixed_checkpoint_scale=0.20,
        compile=False, cuda_graph=False, bf16_gemm=False,
        compile_drafter=False, compile_online=False, fast_decode=False,
        analytic_online_grad=True, resume=True,
    )
    cfg_fp = phase_b_suite._suite_config_fingerprint('pure_t1_fit', cuda_args, cuda_ref)
    c_ws = cuda_abl / f'pure_t1_ws_{cfg_fp[:12]}'
    c_ws.mkdir(parents=True)
    # Populate valid reference-layout smoke artifacts in c_ws so run_pure_t1_fit exercises use_cuda=True on CPU
    _, _, s_head, s_bias, h_meta = train.smoke_bundle(
        SimpleNamespace(output=c_ws, seed=2026, history_depth=4, allow_pure_t1_ablation=True, allow_long_oracle_fit=False)
    )
    train.main(['--scale', 'smoke', '--device', 'cpu', '--seed', '2026', '--output', str(c_ws)])
    shutil.copytree(c_ws / 'm2_smoke', c_ws / 'm2')
    t1_fit_dir = c_ws / 'm5_pure_t1_calibration'
    shutil.copytree(c_ws / 'm2_smoke', t1_fit_dir)
    shutil.copy2(t1_fit_dir / 'dense.pt', t1_fit_dir / 'dense_calibrated.pt')
    shutil.copy2(t1_fit_dir / 'dense_sparse.pt', t1_fit_dir / 'dense_sparse_calibrated.pt')
    atomic_json(c_ws / 'm5_pure_t1_calibration.json', {'status': 'passed', 'scale': 'reference', 'stages': {}})
    s_paths = sorted(str(p.resolve()) for sub in ('m2_smoke_traces', 'm2_smoke_t1_traces') for p in (c_ws / sub).glob('rank*/*.pt'))
    (c_ws / 'traces').symlink_to(c_ws / 'm2_smoke_traces')
    atomic_json(c_ws / 'm1_audit.json', {'scale': 'reference', 'status': 'passed', 'shards': s_paths})
    atomic_json(c_ws / 'prompts.json', _make_full_scale_synthetic_manifest())
    snap_d = c_ws / 'snap' / h_meta['revision']
    snap_d.mkdir(parents=True)
    atomic_json(snap_d / 'config.json', {'tie_word_embeddings': False})
    save_safetensors({'lm_head.weight': s_head.detach().cpu(), 'lm_head.bias': s_bias.detach().cpu()}, str(snap_d / 'model.safetensors'))
    atomic_json(c_ws / 'environment.json', {'model_artifacts': {h_meta['artifact_id']: {'revision': h_meta['revision'], 'path': str(snap_d)}}})

    cuda_spawns: list[tuple[str, list[int]]] = []

    def _fake_cuda_spawn(worker_specs, logdir, label, **kw):
        del logdir, kw
        cuda_spawns.append((label, [r for r, _, _ in worker_specs]))
        for _r, _g, cmd in worker_specs:
            mode = cmd[cmd.index('--_worker_mode') + 1]
            spec_p = Path(cmd[cmd.index('--_worker_spec') + 1])
            out_p = Path(cmd[cmd.index('--_worker_out') + 1])
            if mode == 'dual_offline_kl':
                phase_b_suite._worker_dual_offline_kl(SimpleNamespace(device='cpu', seed=2026), spec_p, out_p)
            else:
                spec_d = json.loads(spec_p.read_text())
                fake_rows = {}
                for ak in ['plain_dflash', *[f'{vn}::{st}' for vn, _, _ in spec_d['variants'] for st in ('dense', 'dense_sparse')]]:
                    fake_rows[ak] = [
                        {
                            'prompt_id': p['prompt_id'], 'condition': p['condition'],
                            'sampling_name': sn, 'temperature': float(tp), 'repeat': rp,
                            'repetition_dominated': False, 'generation': {'repetition_dominated': False},
                            'effective_beta': 1.0, 'generated_tokens': 16, 'elapsed_seconds': 0.1,
                            'tokens_per_second': 160.0, 'mean_accepted_length': 4.0,
                            'median_accepted_length': 4.0, 'target_forwards_per_generated_token': 0.25,
                        }
                        for p in spec_d['prompts']
                        for sn, tp, _ in eval_closed_loop.sampling_conditions(p['condition'], primary_only=p['condition'].startswith('long_'))
                        for rp in range(2)
                    ]
                atomic_json(out_p, {
                    'rank': spec_d['rank'], 'physical_gpu': spec_d['physical_gpu'],
                    'numa_node': 0, 'cpu_affinity': [0], 'cpu_affinity_applied': False,
                    'spec_fingerprint': spec_d['spec_fingerprint'],
                    'variant_scales': [(vn, 1.0, 1.0) for vn, _, _ in spec_d['variants']],
                    'rows_by_key': fake_rows,
                })
        return {'status': 'complete'}

    monkeypatch.setattr(phase_b_suite, '_run_training_fit_isolated', lambda *_a, **_kw: {'status': 'passed', 'stages': {}})
    monkeypatch.setattr(phase_b_suite, '_spawn_parallel_workers', _fake_cuda_spawn)

    rep_c1 = phase_b_suite.run_pure_t1_fit(cuda_args, cuda_ref, cuda_abl)
    assert rep_c1['status'] == 'pilot_complete'
    assert [ranks for _, ranks in cuda_spawns] == [[0, 1], [0, 1]]

    # Delete pure_t1_fit.json + closed-loop result-rank1.json, re-run run_pure_t1_fit -> 0 dual-kl spawns, [1] closed-loop spawn!
    (cuda_abl / 'pure_t1_fit.json').unlink()
    (c_ws / 'pure_t1_closed_loop_shards' / 'result-rank1.json').unlink()
    cuda_spawns.clear()
    rep_c2 = phase_b_suite.run_pure_t1_fit(cuda_args, cuda_ref, cuda_abl)
    assert rep_c2['status'] == 'pilot_complete'
    assert cuda_spawns == [('pure-t1-eval', [1])]

    # 4. #320 Item 1: 1-worker rerun on CPU/smoke purges stale `pure_t1_closed_loop_shards/*.json`
    monkeypatch.setattr(phase_b_suite, '_spawn_parallel_workers', real_spawn)
    variants_1w = [('v_b', ws / 'm2_smoke' / 'dense.pt', ws / 'm2_smoke' / 'dense_sparse.pt')]
    args_1w = SimpleNamespace(
        scale='smoke', seed=2026, device='cpu',
        target='Qwen/Qwen3-8B', drafter='z-lab/Qwen3-8B-DFlash-b16',
        timing_warmups=1, timing_repeats=2, timing_prompts=0,
        smoke_tokens=8, max_new_tokens=64, long_max_new_tokens=128,
        resume=False,
    )
    assert len(list(shard_dir.glob('result-rank*.json'))) == 2
    st_1w, _ = phase_b_suite._evaluate_variants_closed_loop(args_1w, ws, 'smoke', variants_1w, [0])
    assert st_1w == 'complete'
    assert list(shard_dir.glob('*.json')) == []

    # 5. #320 Item 2: Malformed cached JSON list / non-dict elements return None cleanly without raising
    bad_json = tmp_path / 'bad_cache.json'
    for bad_payload in (
        [1, 2, 3],
        {'spec_fingerprint': 'fp', 'variant_scales': [['missing_two']], 'rows_by_key': {}},
        {'spec_fingerprint': 'fp', 'variant_scales': [['v_b', 1.0, 1.0]], 'rows_by_key': {'plain_dflash': ['not_a_dict']}},
    ):
        atomic_json(bad_json, bad_payload)
        assert phase_b_suite._valid_cached_variant_eval_shard(bad_json, 'fp', variants_1w, [], 2) is None
        assert phase_b_suite._valid_cached_dual_kl_shard(bad_json, [{'regime': 'primary', 'key': 'k'}]) is None
