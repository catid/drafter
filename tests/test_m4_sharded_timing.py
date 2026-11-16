"""Comprehensive CPU tests for 8-GPU core-pinned sharded M4 timing, 1-GPU isolation calibration, and validator lockstep."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from edraft import collect, diagnose, eval_closed_loop
from edraft.eval_closed_loop import (
    ALGORITHM_FILES,
    FROZEN_09A5469_ALGORITHM_FILES,
    FROZEN_09A5469_ALGORITHM_HASH,
    FROZEN_09A5469_IMPLEMENTATION_HASH,
    algorithm_file_hashes,
    algorithm_hash,
    apply_gpu_cpu_affinity,
    atomic_json,
    candidate_settings,
    compute_timing_sharding_plan,
    evaluation_jobs,
    file_sha256,
    fingerprint,
    gpu_cpu_affinity,
    implementation_hash,
    run_identity,
    validate_checkpoint_hashes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / 'artifacts' / 'm4_evidence_validator' / 'validate.py'


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(REPO_ROOT / 'src')
    return env


def _extract_09a5469_source_repo(dst: Path) -> Path:
    import shutil
    check_git = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'cat-file', '-e', '09a5469c72c442408d1a85af7ccd58c231db9fb9^{commit}'],
        capture_output=True,
        check=False,
    )
    if check_git.returncode == 0:
        dst.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ['git', '-C', str(REPO_ROOT), 'archive', '09a5469c72c442408d1a85af7ccd58c231db9fb9'],
            capture_output=True,
            check=True,
        )
        subprocess.run(['tar', '-x', '-C', str(dst)], input=archive.stdout, check=True)
        return dst
    snapshot = Path('/tmp/edraft_09a5469_source')
    if (snapshot / 'src' / 'edraft').is_dir():
        shutil.copytree(snapshot, dst, dirs_exist_ok=True)
        return dst
    pytest.skip('09a5469 commit archive not available in standalone non-git export')


def _prepare_smoke_m1_m2(root: Path) -> tuple[Path, Path, Path]:
    """Run smoke M1 collection, M1b diagnostics, and M2 training into `root`."""
    trace_dir = root / 'traces'
    prompt_manifest = root / 'prompts.json'
    collect.main([
        '--scale', 'smoke',
        '--output', str(trace_dir),
        '--prompt_manifest', str(prompt_manifest),
        '--device', 'cpu',
    ])
    diag_out = root / 'diagnostic_smoke.json'
    diagnose.main([
        '--scale', 'smoke',
        '--trace_dir', str(trace_dir),
        '--m1_audit', str(trace_dir / 'm1_audit.json'),
        '--output', str(root),
    ])
    proc = subprocess.run(
        [
            sys.executable, '-m', 'edraft.train',
            '--scale', 'smoke',
            '--device', 'cpu',
            '--trace_dir', str(trace_dir),
            '--m1_audit', str(trace_dir / 'm1_audit.json'),
            '--m1_diagnostic', str(diag_out),
            '--prompts', str(prompt_manifest),
            '--output', str(root),
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        check=False,
    )
    assert proc.returncode == 0, f'smoke train failed:\n{proc.stdout}\n{proc.stderr}'
    return trace_dir, prompt_manifest, diag_out


def _run_validator(output_dir: Path, scale: str = 'smoke', extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(VALIDATOR_PATH),
        '--scale', scale,
        '--repo', str(REPO_ROOT),
        '--artifacts', str(output_dir),
        '--output', str(output_dir / 'validation_report.json'),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env(), check=False)


def test_implementation_hash_lockstep_and_09a5469_source_repo_support(tmp_path):
    assert implementation_hash('m3') == implementation_hash('m4') == implementation_hash()
    assert implementation_hash('m5') != implementation_hash('m4')

    frozen_tree = _extract_09a5469_source_repo(tmp_path / 'frozen_09a5469')
    frozen_src = frozen_tree / 'src' / 'edraft'
    assert set(algorithm_file_hashes(frozen_src).keys()) == set(ALGORITHM_FILES)
    assert algorithm_file_hashes(frozen_src) == FROZEN_09A5469_ALGORITHM_FILES
    assert algorithm_hash(frozen_src) == FROZEN_09A5469_ALGORITHM_HASH
    assert implementation_hash('m4', source_dir=frozen_src) == FROZEN_09A5469_IMPLEMENTATION_HASH


def test_checkpoint_hash_provenance_split_and_tamper_rejection(tmp_path, monkeypatch):
    dense = tmp_path / 'dense.pt'
    sparse = tmp_path / 'sparse.pt'
    dense.write_bytes(b'dense-weights')
    sparse.write_bytes(b'sparse-weights')

    current_hashes = {
        'dense': file_sha256(dense),
        'sparse': file_sha256(sparse),
        'implementation': implementation_hash('m4'),
    }
    # 1. Exact current implementation_hash passes.
    validate_checkpoint_hashes(current_hashes, current_hashes)

    # 2. On post-M4 branch where ALGORITHM_FILES have post-M4 fastpaths, FROZEN_09A5469_IMPLEMENTATION_HASH
    #    is rejected unless ALGORITHM_FILES match 09a5469!
    frozen_hashes = {
        'dense': file_sha256(dense),
        'sparse': file_sha256(sparse),
        'implementation': FROZEN_09A5469_IMPLEMENTATION_HASH,
    }
    with pytest.raises(ValueError, match='checkpoint/projection changed since dev tuning'):
        validate_checkpoint_hashes(frozen_hashes, current_hashes)

    # 3. When all 10 ALGORITHM_FILES match 09a5469, FROZEN_09A5469_IMPLEMENTATION_HASH is accepted.
    monkeypatch.setattr(eval_closed_loop, 'algorithm_hash', lambda source_dir=None: FROZEN_09A5469_ALGORITHM_HASH)
    monkeypatch.setattr(eval_closed_loop, 'algorithm_file_hashes', lambda source_dir=None: dict(FROZEN_09A5469_ALGORITHM_FILES))
    validate_checkpoint_hashes(frozen_hashes, current_hashes)

    # 4. Unknown implementation_hash is always rejected.
    bad_hashes = dict(frozen_hashes, implementation='0' * 64)
    with pytest.raises(ValueError, match='checkpoint/projection changed since dev tuning'):
        validate_checkpoint_hashes(bad_hashes, current_hashes)


def test_gpu_cpu_affinity_numa_mapping_and_cpu_test_safety(monkeypatch):
    seen_cpus: set[int] = set()
    for g in range(8):
        info = gpu_cpu_affinity(g, total_cpus=256)
        assert info['physical_gpu'] == g
        assert info['numa_node'] == g // 2
        assert info['cpu_affinity'] == list(range(16 * g, 16 * g + 16))
        assert len(info['cpu_affinity']) == 16
        assert seen_cpus.isdisjoint(info['cpu_affinity'])
        seen_cpus.update(info['cpu_affinity'])
    assert seen_cpus == set(range(128))

    narrow = gpu_cpu_affinity(3, total_cpus=16)
    assert len(narrow['cpu_affinity']) >= 1
    assert all(0 <= c < 16 for c in narrow['cpu_affinity'])

    called = False

    def _forbid_setaffinity(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError('os.sched_setaffinity must not be called on CPU device')

    monkeypatch.setattr(os, 'sched_setaffinity', _forbid_setaffinity)
    meta = apply_gpu_cpu_affinity(torch.device('cpu'), 2)
    assert not called
    assert meta['cpu_affinity_applied'] is False
    assert meta['physical_gpu'] == 2
    assert meta['numa_node'] == 1


def test_snake_sharding_plan_deterministic_and_balanced(tmp_path):
    args = SimpleNamespace(
        output=tmp_path,
        stage='m4',
        scale='smoke',
        seed=2026,
        world_size=2,
        logical_session='smoke-test-session',
        timing_prompts=0,
        timing_repeats=2,
        timing_warmups=1,
        smoke_tokens=16,
        max_new_tokens=2048,
        long_max_new_tokens=24576,
    )
    prompts = [
        {'prompt_id': 'std_a', 'condition': 'standard_nonthinking', 'split': 'test'},
        {'prompt_id': 'std_b', 'condition': 'standard_nonthinking', 'split': 'test'},
        {'prompt_id': 'std_c', 'condition': 'standard_nonthinking', 'split': 'test'},
        {'prompt_id': 'std_d', 'condition': 'standard_nonthinking', 'split': 'test'},
        {'prompt_id': 'long_a', 'condition': 'long_thinking', 'split': 'test'},
        {'prompt_id': 'long_b', 'condition': 'long_thinking', 'split': 'test'},
    ]
    prompt_manifest = {'prompts': prompts}
    selected_cfg = {arm: candidate_settings(arm)[0] for arm in eval_closed_loop.arms_for('m4', 'smoke')}
    settings = {
        'checkpoint_hashes': {'dense': 'h1', 'dense_sparse': 'h2', 'projection': 'h3', 'implementation': implementation_hash('m4')},
        'selected': selected_cfg,
        'model_pair': ['toy-target', 'toy-draft'],
        'model_revisions': ['v1', 'v1'],
        'generation_environment': {'kind': 'unit'},
    }

    plan_fallback = compute_timing_sharding_plan(args, prompt_manifest, settings, prompts)
    assert plan_fallback['weight_source'] == 'prompt_manifest_order_fallback'
    assert set(plan_fallback['ranks']['0']) == {'std_a', 'std_d', 'long_a'}
    assert set(plan_fallback['ranks']['1']) == {'std_b', 'std_c', 'long_b'}

    eval_dir = tmp_path / 'm4' / 'evaluate' / 'rank0'
    eval_dir.mkdir(parents=True)
    frozen_hash = fingerprint(settings)
    stub = SimpleNamespace(
        args=args, tiny=True, checkpoint_hashes=settings['checkpoint_hashes'],
        model_pair=settings['model_pair'], model_revisions=settings['model_revisions'],
        generation_environment=settings['generation_environment'],
    )
    jobs_map = {}
    token_targets = {'std_a': 10, 'std_b': 40, 'std_c': 30, 'std_d': 20, 'long_a': 50, 'long_b': 100}
    for idx, job in enumerate(evaluation_jobs(prompts, 'm4', 'smoke', args.seed)):
        ident = run_identity(stub, job, settings['selected'][job['arm']], 'evaluate', frozen_hash)
        key = fingerprint(ident)
        row_path = eval_dir / f'{idx:04d}.json'
        pid = job['prompt']['prompt_id']
        row_payload = {
            'identity': ident,
            'prompt_id': pid,
            'condition': job['prompt']['condition'],
            'arm': job['arm'],
            'phase_profile': False,
            'generated_tokens': token_targets[pid],
            'generation': {'generated_tokens': token_targets[pid], 'repetition_dominated': False},
        }
        atomic_json(row_path, row_payload)
        jobs_map[key] = {'status': 'complete', 'result_path': str(row_path), 'result_sha256': file_sha256(row_path)}
    atomic_json(tmp_path / 'm4' / 'evaluate' / 'manifest-rank0.json', {'jobs': jobs_map})

    plan_eval = compute_timing_sharding_plan(args, prompt_manifest, settings, prompts)
    assert plan_eval['weight_source'] == 'evaluate_realized_tokens'
    assert set(plan_eval['ranks']['0']) == {'std_b', 'std_a', 'long_b'}
    assert set(plan_eval['ranks']['1']) == {'std_c', 'std_d', 'long_a'}
    assert plan_eval['by_condition']['standard_nonthinking']['ranks']['0'] == ['std_b', 'std_a']
    assert plan_eval['by_condition']['standard_nonthinking']['ranks']['1'] == ['std_c', 'std_d']


def test_end_to_end_single_and_sharded_m4_timing_calibration_and_validators(tmp_path):
    # 1. Legacy W=1 M3 + M4 run must remain 100% intact and pass validate.py.
    out_w1 = tmp_path / 'eval_w1'
    common = ['--scale', 'smoke', '--device', 'cpu']
    eval_closed_loop.main(['--stage', 'm3', '--mode', 'all', '--output', str(out_w1), *common])
    eval_closed_loop.main(['--stage', 'm4', '--mode', 'all', '--output', str(out_w1), *common])
    res_w1 = _run_validator(out_w1, scale='smoke', extra_args=['--source_repo', str(REPO_ROOT)])
    assert res_w1.returncode == 0, f'W=1 validator failed:\n{res_w1.stdout}\n{res_w1.stderr}'

    # 2. Sharded M4 timing via `launch.py --task m4_timing` (W=2 concurrent workers) + 1-GPU isolation calibration!
    out_sharded = tmp_path / 'eval_sharded'
    _prepare_smoke_m1_m2(out_sharded)
    common_sharded = ['--scale', 'smoke', '--device', 'cpu']
    eval_closed_loop.main(['--stage', 'm3', '--mode', 'all', '--output', str(out_sharded), *common_sharded])
    eval_closed_loop.main(['--stage', 'm4', '--mode', 'tune', '--output', str(out_sharded), *common_sharded])
    eval_closed_loop.main(['--stage', 'm4', '--mode', 'evaluate', '--output', str(out_sharded), *common_sharded])

    # Run 1-GPU calibration FIRST (testing calibration-before-sharded auto-reconcile order too!)
    eval_closed_loop.main([
        '--stage', 'm4',
        '--mode', 'timing_calibration',
        '--calibration_repeats', '2',
        '--output', str(out_sharded),
        *common_sharded,
    ])

    launch_cmd = [
        sys.executable, '-m', 'edraft.launch',
        '--task', 'm4_timing',
        '--scale', 'smoke',
        '--gpus', '0,1',
        '--output', str(out_sharded),
        '--logical_session', 'smoke-m4-w2',
    ]
    proc = subprocess.run(launch_cmd, capture_output=True, text=True, env=_subprocess_env(), check=False)
    assert proc.returncode == 0, f'launch.py m4_timing failed:\n{proc.stdout}\n{proc.stderr}'

    timing_json = json.loads((out_sharded / 'm4' / 'timing.json').read_text())
    assert timing_json['status'] == 'complete'
    assert timing_json['world_size'] == 2
    assert timing_json['sharding_mode'] == 'sharded_2gpu'
    assert timing_json['timing_harness_implementation_sha256'] == implementation_hash('m4')
    assert (out_sharded / 'm4' / 'timing' / 'sharding_plan.json').is_file()
    assert (out_sharded / 'm4' / 'timing_isolation_calibration.json').is_file()

    cal_json = json.loads((out_sharded / 'm4' / 'timing_isolation_calibration.json').read_text())
    assert cal_json['status'] == 'complete'
    assert cal_json['reconciled_with_sharded_timing'] is True
    assert cal_json['warmup_completed'] is True
    assert len(cal_json['groups']) > 0

    res_val = _run_validator(out_sharded, scale='smoke')
    assert res_val.returncode == 0, f'Sharded validator failed:\n{res_val.stdout}\n{res_val.stderr}'

    # 3. Test interrupted-bundle resume on rank 1!
    sess_r1_path = out_sharded / 'm4' / 'timing' / 'sessions' / 'smoke-m4-w2-rank1.json'
    sess_r1 = json.loads(sess_r1_path.read_text())
    first_bkey = next(iter(sess_r1['bundles']))
    victim_file = Path(sess_r1['bundles'][first_bkey]['result_paths'][-1])
    victim_file.unlink()
    sess_r1['bundles'][first_bkey]['status'] = 'in_progress'
    sess_r1['bundles'][first_bkey]['result_paths'] = sess_r1['bundles'][first_bkey]['result_paths'][:-1]
    atomic_json(sess_r1_path, sess_r1)

    eval_closed_loop.main([
        '--stage', 'm4', '--mode', 'timing',
        '--rank', '1', '--world_size', '2',
        '--logical_session', 'smoke-m4-w2',
        '--output', str(out_sharded),
        *common_sharded,
    ])
    eval_closed_loop.main(['--stage', 'm4', '--mode', 'summarize', '--output', str(out_sharded), *common_sharded])
    sess_r1_after = json.loads(sess_r1_path.read_text())
    assert len(sess_r1_after['epochs']) == 2
    assert sess_r1_after['bundles'][first_bkey]['status'] == 'complete'
    assert len(sess_r1_after['bundles'][first_bkey]['result_paths']) == 5

    res_resume = _run_validator(out_sharded, scale='smoke')
    assert res_resume.returncode == 0, f'Post-resume validator failed:\n{res_resume.stdout}\n{res_resume.stderr}'

    # 4. Fault-injection tests: validator MUST reject overlapping CPU affinity or tampered calibration math!
    sess_r1_bad = copy.deepcopy(sess_r1_after)
    sess_r1_bad['cpu_affinity'] = list(range(0, 16))
    atomic_json(sess_r1_path, sess_r1_bad)
    timing_bad = json.loads((out_sharded / 'm4' / 'timing.json').read_text())
    timing_bad['raw_evidence']['session_sha256s'][1] = file_sha256(sess_r1_path)
    atomic_json(out_sharded / 'm4' / 'timing.json', timing_bad)
    res_overlap = _run_validator(out_sharded, scale='smoke')
    assert res_overlap.returncode != 0
    assert 'cpu_affinity' in (res_overlap.stdout + res_overlap.stderr)

    atomic_json(sess_r1_path, sess_r1_after)
    eval_closed_loop.main(['--stage', 'm4', '--mode', 'summarize', '--output', str(out_sharded), *common_sharded])

    cal_fresh = json.loads((out_sharded / 'm4' / 'timing_isolation_calibration.json').read_text())
    cal_tampered = copy.deepcopy(cal_fresh)
    cal_tampered['groups'][0]['tps_8gpu_over_1gpu_ratio'] += 0.5
    atomic_json(out_sharded / 'm4' / 'timing_isolation_calibration.json', cal_tampered)
    res_cal_bad = _run_validator(out_sharded, scale='smoke')
    assert res_cal_bad.returncode != 0
    assert 'calibration group' in (res_cal_bad.stdout + res_cal_bad.stderr)
