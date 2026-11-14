"""Report provenance fixtures are unit data, never model measurements."""
from types import SimpleNamespace
import copy
import json
from dataclasses import asdict
import pytest
from edraft.runtime import atomic_json, fingerprint
from edraft.eval_closed_loop import (file_sha256, sampling_conditions, select_prompts,
    m4_feature_context, EvaluationRunner, smoke_manifest, arms_for, candidate_settings,
    evaluate, aggregate)


def reference_fixture(tmp_path):
    manifest = {'prompts': [dict(prompt_id=f'{condition}-{i}', split='test', condition=condition,
        text=f'Prompt {i}', reference_selected=True) for condition, count in
        [('standard_nonthinking', 64), ('long_thinking', 32)] for i in range(count)]}
    manifest_path = tmp_path / 'prompts.json'
    atomic_json(manifest_path, manifest)
    (tmp_path / 'h_proj.pt').write_bytes(b'unit projection fixture')
    (tmp_path / 'token_features.pt').write_bytes(b'unit feature fixture')
    hashes = {name: file_sha256(tmp_path / name) for name in ('h_proj.pt', 'token_features.pt')}
    calibration = {'identity': {'model_pair': ['target', 'draft'], 'checkpoint': ['target-revision', 'draft-revision'],
        'prompt_manifest_sha256': file_sha256(manifest_path)}, 'artifact_sha256': hashes,
        'resolved_config': {'prompt_manifest': str(manifest_path)}}
    atomic_json(tmp_path / 'calibration.json', calibration)
    shard = tmp_path / 'unit-trace.pt'
    shard.write_bytes(b'accepted unit shard fixture')
    sources = [{'path': str(shard.resolve()), 'sha256': file_sha256(shard),
                'identity_sha256': fingerprint({'unit_identity': 1})}]
    generations, groups = [], []
    features = {'coordinate_mean_norm': .1, 'variance_ratio_to_standard_calibration': 1.2,
                'l2_norm_quantiles': {'p0': 1., 'p25': 2., 'p50': 3., 'p75': 4., 'p95': 5., 'p99': 6., 'p100': 7.},
                'outlier_slot_rate': .01, 'outlier_coordinate_rate': .001,
                'outlier_threshold_abs_coordinate': 5., 'calibration_reference': 'unit-standard-calibration'}
    for prompt in manifest['prompts']:
        for _, temperature, warp in sampling_conditions(prompt['condition']):
            generations.append({'prompt_id': prompt['prompt_id'], 'condition': prompt['condition'], 'split': 'test',
                                'temperature': temperature, 'sampling_warp_config': asdict(warp), 'complete': True})
    for condition in ('standard_nonthinking', 'long_thinking'):
        for _, temperature, warp in sampling_conditions(condition):
            groups.append({'condition': condition, 'collection_temperature': temperature,
                'sampling_warp_config': asdict(warp), 'split': 'test', 'axis': 'output_offset', 'bin': '0-1K',
                'data_policy': 'on_policy', 'inclusion': 'primary', 'logical_slots': 10, 'features': copy.deepcopy(features)})
    audit = {'status': 'passed', 'scale': 'reference', 'calibration': calibration,
             'source_shards': sources, 'generations': generations}
    audit_path = tmp_path / 'm1_audit.json'
    atomic_json(audit_path, audit)
    diagnostic = {'status': 'measured', 'scale': 'reference', 'label': 'reference', 'groups': groups,
        'source_shards': [{'path': source['path'], 'sha256': source['sha256']} for source in sources],
        'audit_provenance': {'audit_path': str(audit_path), 'audit_sha256': file_sha256(audit_path),
            'source_set_sha256': fingerprint(sources), 'frozen_artifact_sha256': hashes,
            'prompt_manifest_sha256': file_sha256(manifest_path)}}
    diagnostic_path = tmp_path / 'diagnostic_reference.json'
    atomic_json(diagnostic_path, diagnostic)
    settings = {'model_pair': ['target', 'draft'], 'model_revisions': ['target-revision', 'draft-revision'],
                'checkpoint_hashes': {'projection': hashes['h_proj.pt']}, 'prompt_manifest_hash': fingerprint(manifest)}
    args = SimpleNamespace(output=tmp_path, stage='m4', scale='reference', feature_diagnostic=None)
    return args, manifest, settings, diagnostic_path, audit_path


def test_m4_links_matching_plain_feature_context_and_explicit_empty_bins(tmp_path):
    args, manifest, settings, diagnostic_path, _ = reference_fixture(tmp_path)
    context = m4_feature_context(args, manifest, settings, select_prompts(manifest, 'test', 'reference'))
    assert context['status'] == 'complete'
    assert context['feature_group_count'] == 4
    assert context['matched_test_prompt_conditions'] == 224
    assert context['diagnostic_sha256'] == file_sha256(diagnostic_path)
    assert context['frozen_artifact_sha256']['h_proj.pt'] == settings['checkpoint_hashes']['projection']
    assert 'not measurements of updated M4 arms' in context['interpretation']
    assert all('16K+' in condition['empty_bins'] for condition in context['condition_bin_coverage'])
    assert context['artifact_sha256'] == file_sha256(context['artifact_path'])


@pytest.mark.parametrize('failure', ['missing_report', 'projection', 'source_set', 'missing_condition', 'missing_statistic', 'model_revision', 'source_prompt'])
def test_m4_feature_context_rejects_missing_or_mismatched_evidence(tmp_path, failure):
    args, manifest, settings, diagnostic_path, audit_path = reference_fixture(tmp_path)
    if failure == 'missing_report':
        diagnostic_path.unlink()
    elif failure == 'projection':
        (tmp_path / 'h_proj.pt').write_bytes(b'changed calibration')
    elif failure == 'model_revision':
        settings['model_revisions'][0] = 'different revision'
    else:
        diagnostic = json.loads(diagnostic_path.read_text())
        if failure == 'source_set':
            diagnostic['source_shards'][0]['sha256'] = 'different source'
        elif failure == 'missing_condition':
            diagnostic['groups'] = [group for group in diagnostic['groups'] if group['condition'] != 'long_thinking']
        elif failure == 'missing_statistic':
            del diagnostic['groups'][0]['features']['coordinate_mean_norm']
        elif failure == 'source_prompt':
            audit = json.loads(audit_path.read_text())
            audit['generations'] = audit['generations'][:-1]
            atomic_json(audit_path, audit)
            diagnostic['audit_provenance']['audit_sha256'] = file_sha256(audit_path)
        atomic_json(diagnostic_path, diagnostic)
    context = m4_feature_context(args, manifest, settings, select_prompts(manifest, 'test', 'reference'))
    assert context['status'] == 'incomplete'
    assert context['reason']
    assert not context['groups']


def test_smoke_m4_recomputes_separate_plain_feature_context_and_requires_it(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path, stage='m4', scale='smoke', seed=91, device='cpu', smoke_tokens=8,
                           rank=0, world_size=1, max_new_tokens=8, long_max_new_tokens=8)
    runner = EvaluationRunner(args)
    manifest = smoke_manifest()
    settings = {'selected': {arm: candidate_settings(arm)[0] for arm in arms_for('m4')},
                'checkpoint_hashes': runner.checkpoint_hashes, 'model_pair': runner.model_pair,
                'model_revisions': runner.model_revisions}
    evaluate(runner, manifest, settings)
    atomic_json(tmp_path / 'm4' / 'timing.json', {'status': 'complete', 'full_reference_prompt_coverage': True,
                                              'frozen_settings_hash': fingerprint(settings)})
    # This test isolates the feature-context gate; raw timing is covered separately.
    monkeypatch.setattr('edraft.eval_closed_loop.audit_timing_evidence', lambda *args: {'status': 'complete'})
    complete = aggregate(args, manifest, settings)
    assert complete['status'] == 'complete'
    assert complete['feature_context_complete']
    assert complete['feature_context']['label'] == 'synthetic'
    assert complete['feature_context']['checkpoint_hashes'] == settings['checkpoint_hashes']
    assert 'not features from updated adapter arms' in complete['feature_context']['interpretation']
    monkeypatch.setattr('edraft.eval_closed_loop.m4_feature_context', lambda *args: {'status': 'incomplete', 'reason': 'missing feature evidence'})
    incomplete = aggregate(args, manifest, settings)
    assert not incomplete['missing_requests'] and incomplete['isolated_timing_complete']
    assert incomplete['status'] == 'incomplete' and not incomplete['feature_context_complete']
