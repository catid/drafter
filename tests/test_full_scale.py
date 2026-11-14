"""Comprehensive CPU verification for `--scale full` (`--allow_large_run`) across M1-M4."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from edraft import collect, eval_closed_loop, launch, prompts, train
from edraft.features import fit_hidden_projection, fit_token_features
from edraft.runtime import common_parser, fingerprint, parse_config, preflight
from edraft.sampler import SamplingWarper
from edraft.toy import ToyParallelDrafter, ToyTarget


def _make_full_scale_prompt_manifest(*, n_train: int = 80, n_std_dev: int = 16, n_std_test: int = 128,
                                     n_long_dev: int = 32, n_long_test: int = 96,
                                     long_train: int = 0) -> dict:
    rows = []
    for i in range(n_train):
        rows.append({
            'prompt_id': f'std-train-{i:04d}',
            'split': 'train',
            'condition': 'standard_nonthinking',
            'dataset': 'gsm8k',
            'text': f'Train question {i}',
            'reference_selected': True,
            'drafter_distribution': 'in_distribution_nonthinking',
        })
    for i in range(n_std_dev):
        rows.append({
            'prompt_id': f'std-dev-{i:04d}',
            'split': 'dev',
            'condition': 'standard_nonthinking',
            'dataset': 'gsm8k',
            'text': f'Dev question {i}',
            'reference_selected': True,
            'drafter_distribution': 'in_distribution_nonthinking',
        })
    for i in range(n_std_test):
        rows.append({
            'prompt_id': f'std-test-{i:04d}',
            'split': 'test',
            'condition': 'standard_nonthinking',
            'dataset': 'humaneval' if i >= n_std_test // 2 else 'gsm8k',
            'text': f'Test question {i}',
            'reference_selected': True,
            'drafter_distribution': 'in_distribution_nonthinking',
        })
    for i in range(n_long_dev):
        rows.append({
            'prompt_id': f'long-dev-{i:04d}',
            'split': 'dev',
            'condition': 'long_thinking',
            'dataset': 'math500',
            'text': f'Long dev problem {i}',
            'reference_selected': True,
            'drafter_distribution': 'out_of_distribution',
        })
    for i in range(n_long_test):
        rows.append({
            'prompt_id': f'long-test-{i:04d}',
            'split': 'test',
            'condition': 'long_thinking',
            'dataset': 'math500',
            'text': f'Long test problem {i}',
            'reference_selected': True,
            'drafter_distribution': 'out_of_distribution',
        })
    for i in range(long_train):
        rows.append({
            'prompt_id': f'long-train-{i:04d}',
            'split': 'train',
            'condition': 'long_thinking',
            'dataset': 'math500',
            'text': f'Long oracle train problem {i}',
            'reference_selected': False,
            'm5_long_oracle_train': True,
            'drafter_distribution': 'out_of_distribution',
        })
    return {
        'schema_version': 1,
        'scale': 'full',
        'complete_heldout_dataset': 'humaneval',
        'dataset_revisions': {'gsm8k': 'rev-1', 'humaneval': 'rev-2', 'math500': 'rev-3'},
        'prompts': rows,
    }


def test_full_scale_preflight_requires_allow_large_run(tmp_path, monkeypatch):
    import edraft.runtime as rt

    parser = common_parser('test')
    with pytest.raises(SystemExit):
        parse_config(parser, ['--scale', 'full', '--output', str(tmp_path)])

    monkeypatch.setattr(rt, 'ROOT', tmp_path)
    (tmp_path / 'artifacts').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'artifacts' / 'smoke.json').write_text(
        json.dumps({'status': 'passed', 'tokens_per_second': 1000.0})
    )
    args_allowed = parse_config(parser, ['--scale', 'full', '--allow_large_run', '--output', str(tmp_path)])
    report = preflight(args_allowed, 2_000_000)
    assert report['expected_tokens'] == 2_000_000
    assert (tmp_path / 'NOTES.md').is_file()


def test_prompts_m5_long_train_carveout_preserves_default_manifest(tmp_path, monkeypatch):
    import edraft.download as dl

    monkeypatch.setattr(prompts, 'ROOT', tmp_path)
    monkeypatch.setattr(prompts, 'record_run', lambda args, name: {})
    monkeypatch.setattr(dl, 'fetch_dataset', lambda dataset, allow_download: tmp_path / dataset.replace('/', '_'))
    (tmp_path / 'artifacts').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'artifacts' / 'environment.json').write_text(json.dumps({
        'dataset_artifacts': {d: {'revision': 'rev'} for d in prompts.DATASETS}
    }))

    def fake_local_rows(path, subset=None, split=None):
        name = Path(path).name
        if 'gsm8k' in name:
            return [{'question': f'GSM train {i}', 'answer': '1'} for i in range(200)]
        if 'humaneval' in name:
            return [{'task_id': f'HE/{i}', 'prompt': f'def f_{i}():\n'} for i in range(164)]
        return [{'problem': f'MATH {i}', 'level': 'Level 4', 'unique_id': f'm-{i}'} for i in range(200)]

    monkeypatch.setattr(prompts, 'local_rows', fake_local_rows)

    base_ns = dict(scale='full', allow_large_run=True, allow_download=False, seed=2026)
    m_default = prompts.build_manifest(SimpleNamespace(**base_ns, output=tmp_path / 'def'))
    m_zero = prompts.build_manifest(SimpleNamespace(**base_ns, m5_long_train_prompts=0, output=tmp_path / 'zero'))
    assert m_default == m_zero

    m_oracle = prompts.build_manifest(SimpleNamespace(**base_ns, m5_long_train_prompts=24, output=tmp_path / 'orc'))
    oracle_rows = [r for r in m_oracle['prompts'] if r.get('m5_long_oracle_train')]
    assert len(oracle_rows) == 24
    assert all(r['split'] == 'train' and r['condition'] == 'long_thinking' for r in oracle_rows)
    assert all(r['reference_selected'] is False for r in oracle_rows)
    eval_ids = {r['prompt_id'] for r in m_oracle['prompts'] if r['split'] in {'dev', 'test'}}
    assert eval_ids.isdisjoint({r['prompt_id'] for r in oracle_rows})


def test_collect_planned_jobs_and_full_scale_token_ceilings(tmp_path):
    manifest = _make_full_scale_prompt_manifest(long_train=10)
    args = SimpleNamespace(
        scale='full',
        allow_large_run=True,
        partitions=['train_primary', 'standard_eval', 'diagnostic', 'long', 'long_oracle_train'],
        standard_max_new_tokens=1024,
        long_max_new_tokens=24576,
        diagnostic_train_prompts=16,
        world_size=2,
        rank=0,
        seed=2026,
    )
    args_r1 = SimpleNamespace(**{**args.__dict__, 'rank': 1})
    jobs_r0 = list(collect.planned_jobs(args, manifest))
    jobs_r1 = list(collect.planned_jobs(args_r1, manifest))
    oracle_r0 = [j['row']['prompt_id'] for j in jobs_r0 if j['partition'] == 'long_oracle_train']
    oracle_r1 = [j['row']['prompt_id'] for j in jobs_r1 if j['partition'] == 'long_oracle_train']
    assert len(oracle_r0) + len(oracle_r1) == 10
    assert set(oracle_r0).isdisjoint(oracle_r1)


def test_train_full_scale_reference_gates(tmp_path):
    manifest = _make_full_scale_prompt_manifest()
    prompts_path = tmp_path / 'prompts.json'
    prompts_path.write_text(json.dumps(manifest))

    target = ToyTarget(vocab_size=8, hidden_size=12, seed=7)
    drafter = ToyParallelDrafter(target, seed=8)
    cal_Ids = [f'std-train-{i:04d}' for i in range(64)]
    cal_h = [drafter.draft([i % 8, (i + 1) % 8], 4).h_raw for i in range(64)]
    proj = fit_hidden_projection(cal_h, 12, 6, seed=7, prompt_ids=cal_Ids)
    proj_path = tmp_path / 'h_proj.pt'
    proj.save(proj_path)

    E, tok_meta = fit_token_features(target.embedding, 4, 7, torch.arange(8))
    tok_meta['calibration_prompt_ids'] = cal_Ids
    feat_path = tmp_path / 'token_features.pt'
    torch.save({'E': E, 'metadata': tok_meta}, feat_path)

    proj_sha = train.file_digest(proj_path)
    feat_sha = train.file_digest(feat_path)
    shard_path = tmp_path / 'rank0' / 's0.pt'
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(b'placeholder')
    shard_sha = train.file_digest(shard_path)

    audit_path = tmp_path / 'm1_audit.json'
    audit_path.write_text(json.dumps({'status': 'passed', 'scale': 'full', 'shards': [str(shard_path)]}))
    diag_path = tmp_path / 'diagnostic_full.json'
    diag_path.write_text(json.dumps({
        'status': 'measured',
        'scale': 'full',
        'source_shards': [{'path': str(shard_path.resolve()), 'sha256': shard_sha}],
    }))

    entry = train.ShardInfo(
        path=shard_path,
        split='train',
        prompt_id='std-train-0000',
        output_tokens=2_400_000,
        blocks=10,
        sha256=shard_sha,
        identity={
            'projection_sha256': proj_sha,
            'token_features_sha256': feat_sha,
            'model_pair': ['t-model', 'd-model'],
        },
    )
    args = SimpleNamespace(
        scale='full',
        m1_audit=audit_path,
        m1_diagnostic=diag_path,
        prompts=prompts_path,
        projection=proj_path,
        features=feat_path,
        target='t-model',
        drafter='d-model',
        allow_long_oracle_fit=False,
        allow_pure_t1_ablation=False,
        max_primary_train_tokens=4_000_000,
    )

    valid_index = {
        'primary': {
            'train': {'generated_output_tokens': 2_400_000, 'prompts': 80, 'shards': 160, 'blocks': 1000},
            'dev': {'generated_output_tokens': 32_000, 'prompts': 16, 'shards': 32, 'blocks': 100},
            'test': {'generated_output_tokens': 256_000, 'prompts': 128, 'shards': 256, 'blocks': 800},
        },
        'primary_train_tokens_by_temperature': {'0.0': 1_200_000, '0.7': 1_200_000},
    }
    train.validate_reference_gate(args, valid_index, [entry], proj, tok_meta)

    under_floor = {
        **valid_index,
        'primary': {**valid_index['primary'], 'train': {**valid_index['primary']['train'], 'generated_output_tokens': 1_999_999}},
    }
    with pytest.raises(RuntimeError, match='fewer than 2000000'):
        train.validate_reference_gate(args, under_floor, [entry], proj, tok_meta)

    over_ceiling = {
        **valid_index,
        'primary': {**valid_index['primary'], 'train': {**valid_index['primary']['train'], 'generated_output_tokens': 4_000_001}},
    }
    with pytest.raises(RuntimeError, match='exceed 4000000 token ceiling'):
        train.validate_reference_gate(args, over_ceiling, [entry], proj, tok_meta)

    under_test = {
        **valid_index,
        'primary': {**valid_index['primary'], 'test': {**valid_index['primary']['test'], 'prompts': 127}},
    }
    with pytest.raises(RuntimeError, match='fewer than 128 standard test prompts'):
        train.validate_reference_gate(args, under_test, [entry], proj, tok_meta)


def test_eval_closed_loop_full_scale_prompts_seeds_and_arms():
    manifest = _make_full_scale_prompt_manifest(n_std_test=128, n_long_dev=32, n_long_test=96)
    dev_selected = eval_closed_loop.select_prompts(manifest, 'dev', 'full', standard_dev_prompts=8, long_dev_prompts=32)
    assert len([p for p in dev_selected if p['condition'] == 'standard_nonthinking']) == 8
    assert len([p for p in dev_selected if p['condition'] == 'long_thinking']) == 32

    with pytest.raises(ValueError, match='>=32 at full scale'):
        eval_closed_loop.select_prompts(manifest, 'dev', 'full', standard_dev_prompts=8, long_dev_prompts=12)

    test_selected = eval_closed_loop.select_prompts(manifest, 'test', 'full')
    std_test = [p for p in test_selected if p['condition'] == 'standard_nonthinking']
    long_test = [p for p in test_selected if p['condition'] == 'long_thinking']
    assert len(std_test) == 128
    assert len(long_test) == 64

    full_m4_arms = eval_closed_loop.arms_for('m4', 'full')
    assert full_m4_arms == (
        'plain_dflash',
        'online_bigram',
        'offline_static',
        'offline_online_stride1',
        'online_dbS',
        'offline_online_stride5',
        'zero_offline_online_only',
        'online_rejection_relative',
        'online_cutoff',
    )

    jobs = eval_closed_loop.evaluation_jobs(test_selected, 'm4', 'full', seed=2026)
    # Per std test prompt: greedy (1 seed) + T=0.7 (3 seeds) + T=1.0 (3 seeds) = 7 * 9 arms = 63 jobs
    # Per long_thinking test prompt: T=0.6 (3 seeds at full scale!) = 3 * 9 arms = 27 jobs
    assert len(jobs) == 128 * 7 * len(full_m4_arms) + 64 * 3 * len(full_m4_arms)


def test_full_scale_m4_extra_arms_execute_cleanly_with_exact_zero_offline_norms(tmp_path):
    parser = common_parser('test')
    parser.add_argument('--stage', default='m4')
    parser.add_argument('--smoke_tokens', type=int, default=24)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--long_max_new_tokens', type=int, default=64)
    args = parse_config(parser, ['--scale', 'smoke', '--device', 'cpu', '--output', str(tmp_path)])
    runner = eval_closed_loop.EvaluationRunner(args)
    prompt = {
        'prompt_id': 'synthetic-test-standard_nonthinking-0',
        'split': 'test',
        'condition': 'standard_nonthinking',
        'text': 'Synthetic 0',
        'dataset': 'builtin',
        'reference_selected': True,
    }
    for arm in ('offline_online_stride5', 'zero_offline_online_only', 'online_rejection_relative', 'online_cutoff'):
        setting = eval_closed_loop.candidate_settings(arm)[0]
        for temp, warp in [(0.0, SamplingWarper()), (0.7, SamplingWarper(top_k=4, top_p=0.85))]:
            row = runner.run(
                {'prompt': prompt, 'arm': arm, 'sampling_name': 'test', 'temperature': temp, 'warper': warp, 'seed': 2026},
                setting,
                profile=True,
            )
            assert row['generation']['complete'] is True
            assert row['online_invariants']['single_round_no_buffering'] is True
            if arm == 'zero_offline_online_only':
                assert row['parameter_norms'] == {'W': 0.0, 'A': 0.0, 'b': 0.0}
            else:
                assert row['parameter_norms']['W'] > 0.0
            if arm == 'offline_online_stride5':
                assert row['online_invariants']['stride'] == 5


def test_beta_sweep_mode_resumable_and_checksum_verified(tmp_path):
    eval_closed_loop.main([
        '--stage', 'm3', '--scale', 'smoke', '--mode', 'tune',
        '--device', 'cpu', '--output', str(tmp_path), '--smoke_tokens', '16',
    ])
    eval_closed_loop.main([
        '--stage', 'm3', '--scale', 'smoke', '--mode', 'beta_sweep',
        '--beta_grid', '0.0', '0.5', '1.0', '--beta_sweep_split', 'dev',
        '--device', 'cpu', '--output', str(tmp_path), '--smoke_tokens', '16',
    ])
    sweep_path = tmp_path / 'm3' / 'beta_sweep.json'
    assert sweep_path.is_file()
    report = json.loads(sweep_path.read_text())
    assert report['status'] == 'complete'
    assert report['beta_grid'] == [0.0, 0.5, 1.0]
    assert set(report['swept_arms']) == {'dense', 'dense_sparse'}
    assert len(report['curves']) > 0

    # Second run reuses manifest without recomputing; tampering a result file triggers checksum error.
    eval_closed_loop.main([
        '--stage', 'm3', '--scale', 'smoke', '--mode', 'beta_sweep',
        '--beta_grid', '0.0', '0.5', '1.0', '--beta_sweep_split', 'dev',
        '--device', 'cpu', '--output', str(tmp_path), '--smoke_tokens', '16',
    ])
    manifest = json.loads((tmp_path / 'm3' / 'beta_sweep' / 'manifest-rank0.json').read_text())
    first_entry = next(iter(manifest['jobs'].values()))
    Path(first_entry['result_path']).write_text('{"corrupted": true}')
    with pytest.raises(ValueError, match='checksum changed'):
        eval_closed_loop.main([
            '--stage', 'm3', '--scale', 'smoke', '--mode', 'beta_sweep',
            '--beta_grid', '0.0', '0.5', '1.0', '--beta_sweep_split', 'dev',
            '--device', 'cpu', '--output', str(tmp_path), '--smoke_tokens', '16',
        ])


def test_launch_full_scale_and_m5_worker_commands(tmp_path, monkeypatch):
    spawned = []

    class DummyProc:
        def __init__(self, command, cwd, env, stdout, stderr):
            self.pid = 42000 + len(spawned)
            self.returncode = 0
            spawned.append({'command': command, 'env': env})

        def poll(self):
            return 0

    monkeypatch.setattr(launch, 'selected_gpus_idle', lambda indices: None)
    monkeypatch.setattr(launch, 'record_run', lambda args, name: {})
    monkeypatch.setattr(launch.subprocess, 'Popen', DummyProc)

    launch.main([
        '--scale', 'full', '--allow_large_run', '--task', 'collect',
        '--gpus', '2,5', '--output', str(tmp_path),
    ])
    assert len(spawned) == 2
    assert '--allow_large_run' in spawned[0]['command']
    idx = spawned[0]['command'].index('--train_tokens_per_temperature')
    assert spawned[0]['command'][idx + 1] == '1000000'
    assert str((launch.ROOT / 'src').resolve()) in spawned[0]['env']['PYTHONPATH']

    spawned.clear()
    launch.main([
        '--scale', 'full', '--allow_large_run', '--task', 'm5',
        '--eval_mode', 'beta_sweep', '--m5_triton',
        '--beta_grid', '0.0', '0.2', '0.5', '1.0',
        '--beta_sweep_split', 'dev',
        '--beta_sweep_arms', 'dense_full', 'dense_candidate',
        '--sparse_overflow_policy', 'error',
        '--m5_server_url', 'http://127.0.0.1:8000',
        '--m5_server_model', 'qwen35-test',
        '--m5_server_api_key', 'tok123',
        '--m5_server_timeout', '120.0',
        '--dense_checkpoint', str(tmp_path / 'custom_dense.pt'),
        '--sparse_checkpoint', str(tmp_path / 'custom_sparse.pt'),
        '--checkpoint_scale', '0.20',
        '--m5_beta_candidates', '0.25', '0.75',
        '--m5_beta_relax', '1.0',
        '--gpus', '3', '--output', str(tmp_path),
    ])
    assert len(spawned) == 1
    cmd = spawned[0]['command']
    assert '--stage' in cmd and cmd[cmd.index('--stage') + 1] == 'm5'
    assert '--mode' in cmd and cmd[cmd.index('--mode') + 1] == 'beta_sweep'
    assert '--m5_triton' in cmd
    assert '--allow_large_run' in cmd
    for flag, expected in [
        ('--sparse_overflow_policy', 'error'),
        ('--beta_sweep_split', 'dev'),
        ('--m5_server_model', 'qwen35-test'),
        ('--m5_server_api_key', 'tok123'),
        ('--m5_server_timeout', '120.0'),
        ('--dense_checkpoint', str(tmp_path / 'custom_dense.pt')),
        ('--sparse_checkpoint', str(tmp_path / 'custom_sparse.pt')),
        ('--checkpoint_scale', '0.2'),
        ('--m5_beta_relax', '1.0'),
    ]:
        assert flag in cmd and cmd[cmd.index(flag) + 1] == expected
    assert '--beta_grid' in cmd
    assert '--beta_sweep_arms' in cmd
    assert '--m5_beta_candidates' in cmd

    # Verify strict launch.py CLI rejections (#69 M5-BUG-3):
    for bad_argv in [
        ['--scale', 'smoke', '--task', 'm5', '--bigram_count_gate_kappa', 'inf'],
        ['--scale', 'smoke', '--task', 'm5', '--bigram_count_gate_kappa', 'nan'],
        ['--scale', 'smoke', '--task', 'm5', '--m5_server_timeout', 'inf'],
        ['--scale', 'smoke', '--task', 'm5', '--checkpoint_scale', '-0.1'],
        ['--scale', 'smoke', '--task', 'm5', '--checkpoint_scale', 'nan'],
        ['--scale', 'smoke', '--task', 'm4', '--m5_triton'],
        ['--scale', 'smoke', '--task', 'm4', '--eval_mode', 'server_eval', '--m5_server_url', 'http://localhost:8000'],
        ['--scale', 'smoke', '--task', 'm3', '--checkpoint_scale', '0.2'],
        ['--scale', 'smoke', '--task', 'collect', '--eval_mode', 'beta_sweep'],
        ['--scale', 'smoke', '--task', 'collect', '--beta_grid', '0.0', '1.0'],
    ]:
        with pytest.raises(SystemExit):
            launch.main(bad_argv)
