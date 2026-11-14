from types import SimpleNamespace
import copy
import pytest
import torch
from edraft.runtime import fingerprint
from edraft.eval_closed_loop import (arms_for, candidate_settings, evaluation_jobs, sampling_conditions,
    select_prompts, validate_frozen_settings, EvaluationRunner, smoke_manifest, summarize_rows)


def manifest():
    return {'prompts': [dict(prompt_id=f'{split}-{condition}-{i}', split=split, condition=condition,
        text='test', reference_selected=True) for split, count in [('dev', 64), ('test', 128)]
        for condition in ['standard_nonthinking', 'long_thinking'] for i in range(count)]}


def test_reference_test_counts_and_shared_conditions_seeds_arms():
    prompts = select_prompts(manifest(), 'test', 'reference')
    assert len(prompts) == 96
    m3 = evaluation_jobs(prompts, 'm3', 'reference')
    assert len(m3) == 1920
    assert len(evaluation_jobs(prompts, 'm4', 'reference')) == 2400
    for arm in arms_for('m3'):
        rows = [j for j in m3 if j['arm'] == arm]
        assert sum(j['temperature'] == 0 for j in rows) == 64
        assert sum(j['temperature'] == .7 for j in rows) == 64 * 3
        assert sum(j['temperature'] == 1. for j in rows) == 64 * 3
        assert sum(j['temperature'] == .6 for j in rows) == 32
    assert all(j['warper'].top_p == .95 and j['temperature'] > 0 for j in m3 if j['prompt']['condition'] == 'long_thinking')


def test_dev_selection_and_settings_prevent_test_leakage():
    prompts = manifest()
    selected = select_prompts(prompts, 'dev', 'reference')
    assert len(selected) == 20 and all(p['split'] == 'dev' for p in selected)
    frozen = {'stage': 'm3', 'scale': 'reference', 'frozen': True, 'checkpoint_hashes': {'dense': 'hash'},
              'development_prompt_ids': [p['prompt_id'] for p in selected], 'prompt_manifest_hash': fingerprint(prompts), 'selected': {a: candidate_settings(a)[0] for a in arms_for('m3')}}
    validate_frozen_settings(frozen, 'm3', 'reference', {'dense': 'hash'}, prompts)
    tampered = copy.deepcopy(frozen)
    tampered['development_prompt_ids'].append('test-standard_nonthinking-0')
    with pytest.raises(ValueError, match='non-development'):
        validate_frozen_settings(tampered, 'm3', 'reference', {'dense': 'hash'}, prompts)
    with pytest.raises(ValueError, match='changed'):
        validate_frozen_settings(frozen, 'm3', 'reference', {'dense': 'new-hash'}, prompts)
    with pytest.raises(ValueError, match='long dev'):
        select_prompts(prompts, 'dev', 'reference', long_dev_prompts=11)


def test_each_tuned_arm_has_same_configuration_budget_and_bigram_alpha_positive():
    for stage in ('m3', 'm4'):
        for arm in arms_for(stage):
            assert len(candidate_settings(arm)) == (1 if arm == 'plain_dflash' else 2)
    assert all(s['alpha'] > 0 and s['k_ret'] is None for s in candidate_settings('online_bigram'))


@pytest.mark.parametrize('stage', ['m3', 'm4'])
def test_all_smoke_arms_run_exact_greedy_and_state_isolation(stage):
    args = SimpleNamespace(scale='smoke', seed=31, device='cpu', stage=stage, smoke_tokens=16)
    runner = EvaluationRunner(args)
    prompt = smoke_manifest()['prompts'][-1]
    name, T, warp = sampling_conditions('standard_nonthinking')[0]
    outputs = []
    for arm in arms_for(stage):
        job = dict(prompt=prompt, arm=arm, sampling_name=name, temperature=T, warper=warp, seed=31)
        settings = candidate_settings(arm)[-1]
        row = runner.run(job, settings, profile=True)
        outputs.append(row['output_tokens'])
        assert row['generated_tokens'] == 16
        assert row['effective_beta'] == 1.
        assert row['elapsed_seconds'] >= row['generation_loop_elapsed_seconds']
        if row['online']:
            assert row['online_invariants']['version_only_between_rounds']
            again = runner.run(job, settings)
            assert again['online']['u'] == row['online']['u']
            assert again['online']['db_norm'] == row['online']['db_norm']
        if arm in ('dense_sparse', 'offline_static'):
            assert 'dense_pair_score' in row['phase_ms']
            assert 'sparse_lookup' in row['phase_ms']
            assert 'adapter_causal_score' not in row['phase_ms']
    assert all(out == outputs[0] for out in outputs)


def test_summary_speed_requires_isolated_timing_and_prompt_bootstrap():
    args = SimpleNamespace(scale='smoke', seed=17, device='cpu', stage='m3', smoke_tokens=16)
    runner = EvaluationRunner(args)
    prompt = smoke_manifest()['prompts'][-1]
    name, T, warp = sampling_conditions('long_thinking')[0]
    row = runner.run(dict(prompt=prompt, arm='plain_dflash', sampling_name=name, temperature=T, warper=warp, seed=17), {})
    report = summarize_rows([row])[0]
    assert 'tokens_per_second' not in report['metrics']
    timed = summarize_rows([row], isolated_timing=True)[0]
    assert timed['metrics']['tokens_per_second']['bootstrap_unit'] == 'prompt_id'
    assert timed['acceptance_bins']


def test_plain_and_bigram_do_not_pay_adapter_projection_cost():
    args = SimpleNamespace(scale='smoke', seed=41, device='cpu', stage='m3', smoke_tokens=16)
    runner = EvaluationRunner(args)
    def forbidden_projection(hidden):
        raise AssertionError('baseline evaluated adapter projection')
    runner.projection = forbidden_projection
    prompt = smoke_manifest()['prompts'][-1]
    name, T, warp = sampling_conditions('long_thinking')[0]
    for arm in ('plain_dflash', 'online_bigram'):
        row = runner.run(dict(prompt=prompt, arm=arm, sampling_name=name, temperature=T, warper=warp, seed=41), candidate_settings(arm)[0], profile=True)
        assert 'feature_projection' not in row['phase_ms']
        if arm == 'plain_dflash':
            assert 'unit_log_probability' not in row['phase_ms']


def test_resumed_comparison_reruns_all_arms_after_partial_bundle(tmp_path):
    from edraft.eval_closed_loop import comparison_session, comparison_bundle
    args = SimpleNamespace(output=tmp_path, stage='m3', scale='smoke', seed=8, rank=0,
                           max_new_tokens=16, long_max_new_tokens=16, smoke_tokens=16, fresh_session=False)
    calls = []
    fail = [True]
    def run(job, setting, profile):
        calls.append(job['arm'])
        if job['arm'] == 'online_bigram' and fail[0]:
            fail[0] = False
            raise RuntimeError('simulated interruption')
        return {'arm': job['arm'], 'prompt_id': job['prompt']['prompt_id'],
                'condition': job['prompt']['condition'], 'seed': job['seed'], 'temperature': job['temperature'],
                'sampling_name': job['sampling_name'], 'sampling_warp_config': vars(job['warper']),
                'settings': setting, 'generation': {'complete': True}, 'phase_profile': profile,
                'tokens_per_second': 1.}
    runner = SimpleNamespace(args=args, tiny=True, device=torch.device('cpu'), checkpoint_hashes={'a': 'b'},
                             model_pair=['toy', 'toy'], model_revisions=['v1', 'v1'], run=run)
    prompt = smoke_manifest()['prompts'][-1]
    name, T, warp = sampling_conditions('long_thinking')[0]
    jobs = [(dict(prompt=prompt, arm=a, sampling_name=name, temperature=T, warper=warp, seed=8), {}) for a in ('plain_dflash', 'online_bigram')]
    path, state, epoch = comparison_session(runner, 'timing', {'test': True})
    with pytest.raises(RuntimeError, match='interruption'):
        comparison_bundle(runner, 'timing', jobs, 'frozen', path, state, epoch, 0)
    assert not state['bundles']
    path, state, epoch = comparison_session(runner, 'timing', {'test': True})
    completed = comparison_bundle(runner, 'timing', jobs, 'frozen', path, state, epoch, 0)
    assert calls == ['plain_dflash', 'online_bigram', 'plain_dflash', 'online_bigram']
    path, state, epoch = comparison_session(runner, 'timing', {'test': True})
    resumed = comparison_bundle(runner, 'timing', jobs, 'frozen', path, state, epoch, 0)
    assert resumed == completed
    assert len(calls) == 4
    assert len(state['epochs']) == 3


def test_frozen_settings_bind_prompt_text_and_long_dev_count():
    prompts = manifest()
    selected = select_prompts(prompts, 'dev', 'reference')
    settings = {'stage': 'm3', 'scale': 'reference', 'frozen': True, 'checkpoint_hashes': {'dense': 'a'},
                'prompt_manifest_hash': fingerprint(prompts), 'development_prompt_ids': [p['prompt_id'] for p in selected],
                'selected': {a: candidate_settings(a)[0] for a in arms_for('m3')}}
    changed = copy.deepcopy(prompts)
    changed['prompts'][0]['text'] = 'Different task with same ID'
    with pytest.raises(ValueError, match='manifest changed'):
        validate_frozen_settings(settings, 'm3', 'reference', {'dense': 'a'}, changed)
    settings['development_prompt_ids'] = [p['prompt_id'] for p in selected if p['condition'] == 'standard_nonthinking']
    with pytest.raises(ValueError, match='long development'):
        validate_frozen_settings(settings, 'm3', 'reference', {'dense': 'a'}, prompts)


def test_pilot_timing_cannot_complete_milestone_even_when_all_outputs_exist(tmp_path):
    from edraft.eval_closed_loop import evaluate, aggregate
    from edraft.runtime import atomic_json
    args = SimpleNamespace(scale='smoke', seed=61, device='cpu', stage='m3', smoke_tokens=8,
                           output=tmp_path, rank=0, world_size=1, max_new_tokens=8, long_max_new_tokens=8)
    runner = EvaluationRunner(args)
    prompts = smoke_manifest()
    settings = {'selected': {a: candidate_settings(a)[0] for a in arms_for('m3')},
                'checkpoint_hashes': runner.checkpoint_hashes, 'model_pair': runner.model_pair,
                'model_revisions': runner.model_revisions}
    evaluate(runner, prompts, settings)
    atomic_json(tmp_path / 'm3' / 'timing.json', {'status': 'pilot_complete', 'full_reference_prompt_coverage': False,
                                              'frozen_settings_hash': fingerprint(settings)})
    report = aggregate(args, prompts, settings)
    assert not report['missing_requests']
    assert report['completed_requests'] == report['expected_requests']
    assert not report['isolated_timing_complete']
    assert report['status'] == 'incomplete'
