"""Greedy comparison fixtures exercise reporting, not reference measurements."""
import json

import pytest

from edraft.eval_closed_loop import (aggregate, evaluate, evaluation_jobs, execute_target_only,
    file_sha256, greedy_output_report, greedy_token_mismatch, sampling_conditions, select_prompts, smoke_manifest)
from edraft.runtime import atomic_json, fingerprint
from test_baseline_reuse import frozen_settings, make_runner


def greedy_job():
    prompt = next(p for p in smoke_manifest()['prompts'] if p['split'] == 'test' and p['condition'] == 'standard_nonthinking')
    name, temperature, warper = sampling_conditions(prompt['condition'])[0]
    return dict(prompt=prompt, arm='plain_dflash', sampling_name=name, temperature=temperature, warper=warper, seed=17)


def evaluation_rows(output, stage='m3'):
    manifest = json.loads((output / stage / 'evaluate' / 'manifest-rank0.json').read_text())
    return [json.loads(open(entry['result_path']).read()) for entry in manifest['jobs'].values()]


def test_greedy_mismatch_counts_first_difference_and_unequal_tails():
    equal = greedy_token_mismatch([1, 2], [1, 2])
    assert equal['sequence_equal'] and equal['first_mismatch'] is None
    assert equal['token_mismatch_rate'] == 0.
    different = greedy_token_mismatch([1, 7, 3], [1, 2, 3, 4])
    assert different['first_mismatch'] == 1
    assert different['mismatched_tokens'] == 2 and different['compared_tokens'] == 4
    assert different['token_mismatch_rate'] == .5
    assert different['length_difference'] == -1
    assert greedy_token_mismatch([], [])['sequence_equal']


def test_target_only_runs_once_and_reuses_matching_diagnostic_across_stages(tmp_path):
    m3, m4 = make_runner(tmp_path, 'm3'), make_runner(tmp_path, 'm4')
    generated = execute_target_only(m3, greedy_job())
    reused = execute_target_only(m4, greedy_job())
    assert generated['status'] == 'generated'
    assert reused['status'] == 'reused' and reused['origin_stage'] == 'm3'
    assert len(m3.target_calls) == 1 and not m4.target_calls
    assert not m3.calls and not m4.calls
    assert generated['path'] == reused['path']
    assert reused['sha256'] == file_sha256(reused['path'])
    row = json.loads(open(reused['path']).read())
    assert 'stage' not in row['identity'] and 'frozen_settings_hash' not in row['identity']
    assert row['generation'] == {'complete': True, 'eos_reached': False, 'capped': True}
    assert row['identity']['input_metadata']['effective_generation_cap'] == 8


@pytest.mark.parametrize('change', ['seed', 'revision', 'cap', 'prompt', 'source', 'tokenizer'])
def test_target_only_identity_changes_require_fresh_generation(tmp_path, change):
    execute_target_only(make_runner(tmp_path, 'm3'), greedy_job())
    m4, job = make_runner(tmp_path, 'm4'), greedy_job()
    if change == 'seed':
        job['seed'] += 1
    elif change == 'revision':
        m4.model_revisions[0] = 'new-target-revision'
    elif change == 'cap':
        m4.args.smoke_tokens += 1
    elif change == 'prompt':
        job['prompt']['text'] += ' changed'
    elif change == 'source':
        m4.checkpoint_hashes['implementation'] = 'new-core-source'
    else:
        prepare = m4.prepare_inputs
        def changed_tokenizer(prompt, max_new_tokens_override=None):
            inputs, cap, metadata = prepare(prompt, max_new_tokens_override)
            return inputs, cap, dict(metadata, generation_environment_sha256='changed-tokenizer-config')
        m4.prepare_inputs = changed_tokenizer
    result = execute_target_only(m4, job)
    assert result['status'] == 'generated' and len(m4.target_calls) == 1


def test_corrupt_target_only_diagnostic_is_regenerated(tmp_path):
    generated = execute_target_only(make_runner(tmp_path, 'm3'), greedy_job())
    row = json.loads(open(generated['path']).read())
    row['output_tokens'][0] = 7
    atomic_json(generated['path'], row)
    runner = make_runner(tmp_path, 'm4')
    repaired = execute_target_only(runner, greedy_job())
    assert repaired['status'] == 'generated' and len(runner.target_calls) == 1
    assert 'checksum changed' in repaired['reason']


def test_paired_report_detects_adapter_mismatch_against_both_references(tmp_path):
    runner, prompts = make_runner(tmp_path, 'm3'), smoke_manifest()
    evaluate(runner, prompts, frozen_settings(runner))
    rows = evaluation_rows(tmp_path)
    changed = next(row for row in rows if row['temperature'] == 0 and row['arm'] == 'dense')
    changed['output_tokens'] = [1, 7, 1]
    changed['generated_tokens'] = 3
    jobs = evaluation_jobs(select_prompts(prompts, 'test', 'smoke'), 'm3', 'smoke', 17)
    report = greedy_output_report(rows, jobs, tmp_path)
    assert report['status'] == 'complete'
    assert report['expected_prompts'] == 2
    assert report['expected_comparisons'] == report['completed_comparisons'] == 16
    assert report['numerical_mismatches_observed']
    mismatches = [comparison for comparison in report['comparisons'] if not comparison['sequence_equal']]
    assert {comparison['reference'] for comparison in mismatches} == {'plain_dflash', 'target_only'}
    assert all(row['first_mismatch'] == 1 and row['length_difference'] == -5 for row in mismatches)
    assert all(group['sequence_mismatch_rate'] == .5 for group in report['groups'] if group['arm'] == 'dense')


def test_missing_plain_comparison_is_explicit(tmp_path):
    runner, prompts = make_runner(tmp_path, 'm3'), smoke_manifest()
    evaluate(runner, prompts, frozen_settings(runner))
    rows = evaluation_rows(tmp_path)
    removed = next(row for row in rows if row['temperature'] == 0 and row['arm'] == 'plain_dflash')
    rows.remove(removed)
    jobs = evaluation_jobs(select_prompts(prompts, 'test', 'smoke'), 'm3', 'smoke', 17)
    report = greedy_output_report(rows, jobs, tmp_path)
    assert report['status'] == 'incomplete'
    assert any(row['reason'] == 'missing paired plain-DFlash greedy output' for row in report['missing_comparisons'])
    assert report['completed_comparisons'] < report['expected_comparisons']


def test_missing_target_only_coverage_gates_aggregate_even_with_all_arm_outputs(tmp_path, monkeypatch):
    runner, prompts = make_runner(tmp_path, 'm3'), smoke_manifest()
    settings = frozen_settings(runner)
    evaluate(runner, prompts, settings)
    atomic_json(tmp_path / 'm3' / 'timing.json', {'status': 'complete', 'full_reference_prompt_coverage': True,
                                              'frozen_settings_hash': fingerprint(settings)})
    # Isolate greedy coverage from the independently tested raw-timing gate.
    monkeypatch.setattr('edraft.eval_closed_loop.audit_timing_evidence', lambda *args: {'status': 'complete'})
    before = aggregate(runner.args, prompts, settings)
    assert before['status'] == 'complete'
    source = execute_target_only(runner, greedy_job())
    from pathlib import Path
    Path(source['path']).unlink()
    missing = aggregate(runner.args, prompts, settings)
    assert missing['status'] == 'incomplete'
    assert not missing['missing_requests'] and missing['isolated_timing_complete']
    assert not missing['greedy_output_comparisons_complete']
    assert all(row['reference'] == 'target_only' for row in missing['greedy_output_comparisons']['missing_comparisons'])


def test_numerical_mismatch_is_reported_without_failing_completed_coverage(tmp_path, monkeypatch):
    runner, prompts = make_runner(tmp_path, 'm3'), smoke_manifest()
    settings = frozen_settings(runner)
    evaluate(runner, prompts, settings)
    manifest_path = tmp_path / 'm3' / 'evaluate' / 'manifest-rank0.json'
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['jobs'].values():
        if entry['identity']['temperature'] == 0 and entry['identity']['proposal_mode'] == 'dense':
            row = json.loads(open(entry['result_path']).read())
            row['output_tokens'][0] = 7
            atomic_json(entry['result_path'], row)
            entry['result_sha256'] = file_sha256(entry['result_path'])
            break
    atomic_json(manifest_path, manifest)
    atomic_json(tmp_path / 'm3' / 'timing.json', {'status': 'complete', 'full_reference_prompt_coverage': True,
                                              'frozen_settings_hash': fingerprint(settings)})
    monkeypatch.setattr('edraft.eval_closed_loop.audit_timing_evidence', lambda *args: {'status': 'complete'})
    report = aggregate(runner.args, prompts, settings)
    assert report['status'] == 'complete'
    assert report['greedy_output_comparisons_complete']
    assert report['greedy_output_comparisons']['numerical_mismatches_observed']
