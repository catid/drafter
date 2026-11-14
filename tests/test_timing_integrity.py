"""Persisted timing evidence must survive independent identity/coverage checks."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from edraft.eval_closed_loop import (arms_for, audit_timing_evidence, candidate_settings,
    comparison_bundle, comparison_session, file_sha256, sampling_conditions, smoke_manifest, timing)
from edraft.runtime import atomic_json, fingerprint


def fake_runner(tmp_path):
    args = SimpleNamespace(output=tmp_path, stage='m3', scale='smoke', seed=8, rank=0, world_size=1,
        max_new_tokens=8, long_max_new_tokens=8, smoke_tokens=8, fresh_session=False,
        timing_repeats=2, timing_warmups=1, timing_prompts=0)
    calls = []
    def run(job, setting, profile=False):
        calls.append((job['arm'], profile))
        return {'arm': job['arm'], 'prompt_id': job['prompt']['prompt_id'],
            'condition': job['prompt']['condition'], 'seed': job['seed'], 'temperature': job['temperature'],
            'sampling_name': job['sampling_name'], 'sampling_warp_config': asdict(job['warper']),
            'settings': setting, 'generation': {'complete': True, 'repetition_dominated': False},
            'phase_profile': profile, 'tokens_per_second': 10., 'elapsed_seconds': .8,
            'phase_ms': {'target_verification': 2.} if profile else {}}
    return SimpleNamespace(args=args, tiny=True, device=torch.device('cpu'), checkpoint_hashes={'dense': 'test'},
        model_pair=['toy', 'toy'], model_revisions=['v1', 'v1'], generation_environment={'kind': 'test'},
        run=run, calls=calls)


@pytest.fixture
def bundle(tmp_path):
    runner = fake_runner(tmp_path)
    prompt = smoke_manifest()['prompts'][-1]
    name, temperature, warper = sampling_conditions('long_thinking')[0]
    jobs = [(dict(prompt=prompt, arm=arm, sampling_name=name, temperature=temperature, warper=warper, seed=8), {})
            for arm in ('plain_dflash', 'online_bigram')]
    path, state, epoch = comparison_session(runner, 'timing', {'test': True})
    rows = comparison_bundle(runner, 'timing', jobs, 'frozen', path, state, epoch, 0)
    return runner, jobs, path, state, rows


@pytest.mark.parametrize('damage,match', [
    ('payload', 'checksum'), ('row_identity', 'identity mismatch'),
    ('manifest_identity', 'manifest identity'), ('missing_checksum', 'checksum'),
    ('wrong_order', 'path'), ('missing_arm', 'coverage'),
    ('profile', 'timing/profile'), ('incomplete_generation', 'incomplete'),
    ('wrong_settings', 'settings mismatch'), ('unknown_epoch', 'unknown epoch'),
])
def test_complete_bundle_never_reuses_damaged_evidence(bundle, damage, match):
    runner, jobs, path, state, rows = bundle
    prior = next(iter(state['bundles'].values()))
    result_path = Path(prior['result_paths'][0])
    manifest_path = runner.args.output / 'm3/timing/manifest-rank0.json'
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['jobs'][fingerprint(rows[0]['identity'])]
    row = json.loads(result_path.read_text())
    if damage == 'wrong_order':
        prior['result_paths'].reverse()
    elif damage == 'missing_arm':
        prior['result_paths'].pop()
    elif damage == 'unknown_epoch':
        prior['epoch'] = 'unrecorded'
    elif damage == 'manifest_identity':
        entry['identity']['seed'] += 1
    elif damage == 'missing_checksum':
        del entry['result_sha256']
    else:
        if damage == 'payload':
            row['tokens_per_second'] = 99999.
        elif damage == 'row_identity':
            row['identity']['seed'] += 1
        elif damage == 'profile':
            row['phase_profile'] = True
        elif damage == 'incomplete_generation':
            row['generation']['complete'] = False
        else:
            row['settings'] = {'beta': 999.}
        atomic_json(result_path, row)
        if damage != 'payload':
            entry['result_sha256'] = file_sha256(result_path)
    atomic_json(manifest_path, manifest)
    atomic_json(path, state)
    path, resumed, epoch = comparison_session(runner, 'timing', {'test': True})
    with pytest.raises(ValueError, match=match):
        comparison_bundle(runner, 'timing', jobs, 'frozen', path, resumed, epoch, 0)
    assert len(runner.calls) == 2


@pytest.mark.parametrize('missing', ['file', 'manifest_entry'])
def test_missing_complete_bundle_evidence_reruns_every_arm(bundle, missing):
    runner, jobs, path, state, rows = bundle
    prior = next(iter(state['bundles'].values()))
    if missing == 'file':
        Path(prior['result_paths'][-1]).unlink()
    else:
        manifest_path = runner.args.output / 'm3/timing/manifest-rank0.json'
        manifest = json.loads(manifest_path.read_text())
        del manifest['jobs'][fingerprint(rows[-1]['identity'])]
        atomic_json(manifest_path, manifest)
    path, resumed, epoch = comparison_session(runner, 'timing', {'test': True})
    new = comparison_bundle(runner, 'timing', jobs, 'frozen', path, resumed, epoch, 0)
    assert [arm for arm, _ in runner.calls] == ['plain_dflash', 'online_bigram'] * 2
    assert all(row['repeat'] != old['repeat'] for row, old in zip(new, rows))


@pytest.fixture
def timing_evidence(tmp_path, monkeypatch):
    runner = fake_runner(tmp_path)
    settings = {'selected': {arm: candidate_settings(arm)[0] for arm in arms_for('m3')}}
    # These tests exercise real persistence/resume/gating, not the already tested
    # numerical summary bootstrap or the generator. Avoid their unrelated cost.
    monkeypatch.setattr('edraft.eval_closed_loop.summarize_rows', lambda *args, **kwargs: [])
    prompts = smoke_manifest()
    report = timing(runner, prompts, settings)
    return runner, prompts, settings, report


def test_timing_completion_requires_every_repeat_and_separate_profile(timing_evidence):
    runner, prompts, settings, report = timing_evidence
    evidence = audit_timing_evidence(runner, prompts, settings, report)
    assert evidence['status'] == 'complete'
    assert evidence['timing_requests'] == 64  # 8 prompt/condition/seed combinations × 4 arms × 2 repeats.
    assert evidence['profile_requests'] == 32
    assert evidence['bundles'] == 24
    calls = len(runner.calls)
    resumed = timing(runner, prompts, settings)
    assert audit_timing_evidence(runner, prompts, settings, resumed)['status'] == 'complete'
    assert len(runner.calls) == calls + 8  # Only the two-condition, four-arm warmups run again.
    assert audit_timing_evidence(runner, prompts, settings, report)['status'] == 'incomplete'  # Old session hash.


@pytest.mark.parametrize('damage', ['flags_only', 'missing_profile', 'profile_checksum',
    'session_checksum', 'manifest_checksum', 'wrong_count', 'prompt_coverage', 'identity'])
def test_timing_gate_rejects_incomplete_or_changed_raw_evidence(timing_evidence, damage):
    runner, prompts, settings, original = timing_evidence
    report = copy.deepcopy(original)
    evidence = report['raw_evidence']
    if damage == 'flags_only':
        del report['raw_evidence']
    elif damage == 'wrong_count':
        report['timed_requests'] -= 1
    elif damage == 'prompt_coverage':
        report['prompt_ids'][-1] = report['prompt_ids'][0]
    elif damage == 'session_checksum':
        Path(evidence['session_path']).write_text('{}')
    elif damage == 'manifest_checksum':
        Path(evidence['manifest_path']).write_text('{}')
    else:
        manifest_path = Path(evidence['manifest_path'])
        manifest = json.loads(manifest_path.read_text())
        entry = next(item for item in manifest['jobs'].values() if item['identity']['phase_profile'])
        result_path = Path(entry['result_path'])
        if damage == 'missing_profile':
            result_path.unlink()
        else:
            row = json.loads(result_path.read_text())
            if damage == 'profile_checksum':
                row['phase_ms']['target_verification'] = 99999.
            else:
                row['identity']['seed'] += 1
            atomic_json(result_path, row)
            if damage == 'identity':
                entry['result_sha256'] = file_sha256(result_path)
                atomic_json(manifest_path, manifest)
                evidence['manifest_sha256'] = file_sha256(manifest_path)
    audit = audit_timing_evidence(runner, prompts, settings, report)
    assert audit['status'] == 'incomplete'
    assert audit['reason']
