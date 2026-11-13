"""Reuse fixtures are unit evidence, never reference model measurements."""
import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from edraft.eval_closed_loop import (REUSED_UNAVAILABLE_FIELDS, aggregate, arms_for, candidate_settings,
    evaluate, execute_job, file_sha256, generation_caps, sampling_conditions, smoke_manifest, summarize_rows)
from edraft.runtime import atomic_json, fingerprint
from edraft.sampler import SamplingWarper


def make_runner(tmp_path, stage):
    args = SimpleNamespace(output=tmp_path, stage=stage, scale='smoke', seed=17, rank=0, world_size=1,
                           smoke_tokens=8, max_new_tokens=16, long_max_new_tokens=32, device='cpu')
    runner = SimpleNamespace(args=args, tiny=True, device=torch.device('cpu'), calls=[],
        checkpoint_hashes={'dense': 'dense', 'dense_sparse': 'sparse', 'projection': 'projection', 'implementation': 'source-hash'},
        model_pair=['toy-target', 'toy-draft'], model_revisions=['v1', 'v1'])

    def prepare_inputs(prompt, max_new_tokens_override=None):
        cap = min(args.smoke_tokens, max_new_tokens_override or args.smoke_tokens)
        return [1, 2], cap, {'input_token_ids_sha256': fingerprint([1, 2]), 'input_token_count': 2,
            'enable_thinking': prompt['condition'] == 'long_thinking', 'requested_generation_cap': cap,
            'effective_generation_cap': cap, 'generation_environment_sha256': fingerprint(None)}

    runner.prepare_inputs = prepare_inputs
    runner.target_calls = []

    def run_target_only(prompt, prepared):
        runner.target_calls.append(prompt['prompt_id'])
        return {'scale': args.scale, 'origin_stage': args.stage, 'prompt_id': prompt['prompt_id'],
                'input_metadata': prepared[2], 'output_tokens': [1, 2] * 4, 'generated_tokens': 8,
                'generation': {'complete': True, 'eos_reached': False, 'capped': True}}

    runner.run_target_only = run_target_only

    def run(job, setting, profile=False):
        runner.calls.append((copy.deepcopy(job), copy.deepcopy(setting), profile))
        prompt = job['prompt']
        return {'scale': args.scale, 'label': 'synthetic', 'stage': args.stage, 'arm': job['arm'],
            'prompt_id': prompt['prompt_id'], 'split': prompt['split'], 'condition': prompt['condition'],
            'seed': job['seed'], 'sampling_name': job['sampling_name'], 'temperature': job['temperature'],
            'sampling_warp_config': asdict(job['warper']), 'settings': copy.deepcopy(setting),
            'verification_rule': 'ratio' if job['temperature'] else 'greedy',
            'model_pair': runner.model_pair, 'model_revisions': runner.model_revisions,
            'checkpoint_hashes': runner.checkpoint_hashes, 'generated_tokens': 8, 'output_tokens': [1, 2] * 4,
            'input_metadata': prepare_inputs(prompt)[2],
            'elapsed_seconds': 2., 'generation_loop_elapsed_seconds': 1.5, 'tokens_per_second': 4.,
            'phase_ms': {'dflash_draft': 100.}, 'peak_allocated_bytes': 100, 'peak_reserved_bytes': 200,
            'estimated_online_state_bandwidth_GBps': None, 'latency_scope': 'unit measurement',
            'mean_accepted_length': 2., 'median_accepted_length': 2., 'target_forwards_per_generated_token': .25,
            'generation': {'complete': True, 'repetition_dominated': False},
            'blocks': [{'accepted_length': 2, 'draft_tokens': [1, 2, 3], 'absolute_output_offset': 0, 'input_context_length': 4}],
            'online': None, 'bigram': {'transitions': 8} if job['arm'] == 'online_bigram' else None,
            'phase_profile': profile}

    runner.run = run
    return runner


def frozen_settings(runner):
    return {'frozen': True, 'stage': runner.args.stage, 'scale': runner.args.scale,
            'selected': {arm: candidate_settings(arm)[0] for arm in arms_for(runner.args.stage)},
            'checkpoint_hashes': runner.checkpoint_hashes, 'model_pair': runner.model_pair,
            'model_revisions': runner.model_revisions, 'generation_caps': generation_caps(runner.args)}


def source_request(tmp_path, arm='plain_dflash'):
    runner = make_runner(tmp_path, 'm3')
    settings = frozen_settings(runner)
    atomic_json(tmp_path / 'm3' / 'settings.json', settings)
    prompt = next(p for p in smoke_manifest()['prompts'] if p['split'] == 'test' and p['condition'] == 'standard_nonthinking')
    name, temperature, warper = sampling_conditions(prompt['condition'])[1]
    job = dict(prompt=prompt, arm=arm, sampling_name=name, temperature=temperature, warper=warper, seed=17)
    row = execute_job(runner, job, settings['selected'][arm], 'evaluate', fingerprint(settings))
    return job, settings['selected'][arm], row


@pytest.mark.parametrize('arm', ['plain_dflash', 'online_bigram'])
def test_exact_baseline_reuses_outputs_without_calling_runner_and_pins_sources(tmp_path, arm):
    job, setting, source = source_request(tmp_path, arm)
    runner = make_runner(tmp_path, 'm4')
    result = execute_job(runner, job, setting, 'evaluate', 'different-m4-selection-hash')
    assert runner.calls == []
    assert result['identity']['stage'] == result['stage'] == 'm4'
    assert result['identity']['frozen_settings_hash'] == 'different-m4-selection-hash'
    for field in ('output_tokens', 'blocks', 'generation', 'mean_accepted_length', 'target_forwards_per_generated_token', 'bigram'):
        assert result[field] == source[field]
    assert all(result[field] is None for field in REUSED_UNAVAILABLE_FIELDS)
    provenance = result['baseline_reuse']
    assert provenance['status'] == 'reused'
    assert provenance['excluded_identity_fields'] == ['stage', 'frozen_settings_hash']
    for name in ('result', 'manifest', 'settings'):
        assert provenance[f'source_{name}_sha256'] == file_sha256(provenance[f'source_{name}_path'])
    assert json.loads(open(provenance['source_result_path']).read()) == source
    assert execute_job(runner, job, setting, 'evaluate', 'different-m4-selection-hash') == result
    assert not runner.calls


@pytest.mark.parametrize(('change', 'identity_field'), [
    ('seed', 'seed'), ('warper', 'sampling_warp_config'), ('settings', 'setting'),
    ('cap', 'generation_cap'), ('other_cap', 'generation_caps'), ('prompt', 'prompt_payload_hash'),
    ('revision', 'model_revisions'), ('source_code', 'checkpoint'), ('checkpoint', 'checkpoint')])
def test_mismatch_runs_fresh_with_reason_and_preserves_m4_settings(tmp_path, change, identity_field):
    job, setting, _ = source_request(tmp_path, 'online_bigram')
    runner = make_runner(tmp_path, 'm4')
    if change == 'seed':
        job['seed'] += 1
    elif change == 'warper':
        job['warper'] = SamplingWarper(top_k=10, top_p=.8)
    elif change == 'settings':
        setting = candidate_settings('online_bigram')[1]
    elif change == 'cap':
        runner.args.smoke_tokens += 1
    elif change == 'other_cap':
        runner.args.long_max_new_tokens += 1
    elif change == 'prompt':
        job['prompt']['text'] += ' changed'
    elif change == 'revision':
        runner.model_revisions = ['v2', 'v1']
    else:
        runner.checkpoint_hashes['implementation' if change == 'source_code' else 'projection'] = 'changed'
    result = execute_job(runner, job, setting, 'evaluate', 'm4-frozen')
    assert len(runner.calls) == 1
    assert result['baseline_reuse']['status'] == 'fresh'
    assert identity_field in result['baseline_reuse']['mismatched_fields']
    assert result['settings'] == setting
    assert result['tokens_per_second'] == 4.


@pytest.mark.parametrize('tamper', ['result_bytes', 'source_settings', 'missing_checksum'])
def test_unverified_source_cannot_be_reused(tmp_path, tamper):
    job, setting, source = source_request(tmp_path)
    manifest_path = tmp_path / 'm3' / 'evaluate' / 'manifest-rank0.json'
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['jobs'][fingerprint(source['identity'])]
    if tamper == 'result_bytes':
        source['output_tokens'][0] = 7
        atomic_json(entry['result_path'], source)
    elif tamper == 'source_settings':
        path = tmp_path / 'm3' / 'settings.json'
        settings = json.loads(path.read_text())
        settings['extra'] = 'changed after freezing'
        atomic_json(path, settings)
    else:
        del entry['result_sha256']
        atomic_json(manifest_path, manifest)
    runner = make_runner(tmp_path, 'm4')
    result = execute_job(runner, job, setting, 'evaluate', 'm4-frozen')
    assert len(runner.calls) == 1
    assert result['baseline_reuse']['status'] == 'fresh'
    assert 'invalid M3 source evidence' in result['baseline_reuse']['reason']


def test_changed_tokenizer_environment_prevents_baseline_reuse(tmp_path):
    job, setting, _ = source_request(tmp_path)
    runner = make_runner(tmp_path, 'm4')
    runner.generation_environment = {'tokenizer': {'chat_template_sha256': 'changed'}}
    result = execute_job(runner, job, setting, 'evaluate', 'm4-frozen')
    assert len(runner.calls) == 1
    assert result['baseline_reuse']['status'] == 'fresh'
    assert 'frozen source settings' in result['baseline_reuse']['reason']


def test_frozen_selection_rejects_changed_tokenizer_model_environment(tmp_path):
    from edraft.eval_closed_loop import validate_frozen_settings
    runner, prompts = make_runner(tmp_path, 'm4'), smoke_manifest()
    settings = frozen_settings(runner)
    settings.update(generation_environment={'tokenizer': 'original'}, prompt_manifest_hash=fingerprint(prompts),
                    development_prompt_ids=[p['prompt_id'] for p in prompts['prompts'] if p['split'] == 'dev'])
    with pytest.raises(ValueError, match='generation environment changed'):
        validate_frozen_settings(settings, 'm4', 'smoke', runner.checkpoint_hashes, prompts,
                                 generation_environment={'tokenizer': 'changed'})


@pytest.mark.parametrize(('mode', 'profile'), [('tune', False), ('timing', False), ('timing', True), ('evaluate', True)])
def test_tune_timing_and_profiles_never_reuse_m3_baselines(tmp_path, mode, profile):
    job, setting, _ = source_request(tmp_path)
    runner = make_runner(tmp_path, 'm4')
    result = execute_job(runner, job, setting, mode, 'm4-frozen', profile=profile)
    assert len(runner.calls) == 1
    assert 'baseline_reuse' not in result
    assert result['elapsed_seconds'] == 2.
    assert result['phase_profile'] == profile
    assert not hasattr(runner, '_m3_baseline_sources')


def test_reused_and_mixed_rows_keep_acceptance_and_only_observed_latency(tmp_path):
    job, setting, source = source_request(tmp_path)
    reused = execute_job(make_runner(tmp_path, 'm4'), job, setting, 'evaluate', 'm4-frozen')
    report = summarize_rows([reused])[0]
    assert report['metrics']['mean_accepted_length']['mean'] == 2.
    assert report['acceptance_bins']
    assert report['request_latency_p50_seconds'] is None
    assert report['request_latency_p95_seconds'] is None
    assert report['phase_ms_median'] == {}
    assert report['peak_allocated_bytes'] is report['peak_reserved_bytes'] is None
    assert report['reused_acceptance_output_requests'] == report['requests_without_latency'] == 1
    mixed = summarize_rows([reused, source])[0]
    assert mixed['requests_with_latency'] == mixed['requests_without_latency'] == 1
    assert mixed['request_latency_p50_seconds'] == 2.
    unavailable = summarize_rows([reused], isolated_timing=True)[0]
    assert unavailable['metrics']['tokens_per_second']['mean'] is None
    assert unavailable['request_latency_p50_prompt_bootstrap']['estimate'] is None


def test_m4_aggregate_acceptance_complete_with_reused_baselines_and_fresh_timing(tmp_path, monkeypatch):
    prompts = smoke_manifest()
    m3 = make_runner(tmp_path, 'm3')
    settings3 = frozen_settings(m3)
    atomic_json(tmp_path / 'm3' / 'settings.json', settings3)
    evaluate(m3, prompts, settings3)
    m4 = make_runner(tmp_path, 'm4')
    settings4 = frozen_settings(m4)
    evaluated = evaluate(m4, prompts, settings4)
    assert evaluated['baseline_reuse']['reused_requests'] == 16
    assert len(m4.calls) == 24
    monkeypatch.setattr('edraft.eval_closed_loop.m4_feature_context', lambda *args: {'status': 'complete'})
    before_timing = aggregate(m4.args, prompts, settings4)
    assert before_timing['completed_requests'] == 40
    assert not before_timing['missing_requests']
    assert before_timing['status'] == 'incomplete'
    assert not before_timing['isolated_timing_complete']
    atomic_json(tmp_path / 'm4' / 'timing.json', {'status': 'complete', 'full_reference_prompt_coverage': True,
                                              'frozen_settings_hash': fingerprint(settings4)})
    report = aggregate(m4.args, prompts, settings4)
    assert report['status'] == 'complete'
    assert report['baseline_reuse']['reused_requests'] == 16
    for group in report['groups']:
        assert group['metrics']['mean_accepted_length']['mean'] == 2.
        if group['arm'] in ('plain_dflash', 'online_bigram'):
            assert group['requests_without_latency'] == group['requests']
            assert group['request_latency_p50_seconds'] is None
