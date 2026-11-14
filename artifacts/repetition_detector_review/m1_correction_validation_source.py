"""CPU-only preservation snapshot and post-regeneration M1 correction audit.

prepare records immutable inputs before the authorized audit/diagnose reruns.
validate independently checks regenerated reports and rehashes every trace.
Neither mode launches collection, diagnosis, fitting, evaluation, or GPU work.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
if ROOT.name == 'artifacts':
    ROOT = ROOT.parent
REVIEW = ROOT / 'artifacts/repetition_detector_review'
ARCHIVE = REVIEW / 'pre_correction'
BASELINE = REVIEW / 'm1_correction_preservation_baseline_v2.json'
ARCHIVE_HASH = 'aebc133b6a1fa77263f5f9dd526e4ce47aa5cadff77b6ca2dec476c7d7177a9e'
EXPECTED_CHANGED_PROMPT = 'math500-long-0346'
GROUP_KEYS = ('condition', 'collection_temperature', 'sampling_warp_config',
              'split', 'axis', 'bin', 'data_policy')
OUTPUT_EDGES = (0, 1024, 4096, 8192, 16384)
OUTPUT_LABELS = ('0-1K', '1K-4K', '4K-8K', '8K-16K', '16K+')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def generation_key(generation):
    return fingerprint({k: generation[k] for k in
        ('prompt_id', 'condition', 'split', 'seed', 'temperature', 'sampling_warp_config')})


def primary(identity):
    temperature, warp = identity['temperature'], identity['sampling_warp_config']
    return identity['condition'] == 'standard_nonthinking' and (temperature == 0 or
        (abs(temperature - .7) < 1e-10 and warp.get('top_k') == 20 and
         warp.get('top_p') == .8 and warp.get('min_p', 0.) == 0.))


def archive_record():
    path = ARCHIVE / 'archive_manifest.json'
    require(digest(path) == ARCHIVE_HASH, 'Pre-correction archive manifest changed')
    manifest = read(path)
    for entry in manifest['files']:
        require(digest(entry['copy']) == entry['sha256'], 'Archive copy changed: ' + entry['copy'])
    return manifest


def affected_shard_evidence(row, context_edges):
    """Reduce raw slot metrics independently, without DiagnosticAccumulator."""
    import torch
    torch.set_num_threads(1)
    path = Path(row['path'])
    require(digest(path) == row['sha256'], 'Affected raw trace changed before evidence capture')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require(fingerprint(payload['metadata']['generation']) == row['generation_sha256'],
            'Affected shard/sidecar generation metadata disagree')
    groups = {}
    for block in payload['blocks']:
        values = {name: block['diagnostics'][name] for name in
            ('target_rank', 'sampled_target_rank', 'sampled_target_rank_unwarped',
             'overlap_sampling', 'overlap_unwarped', 'cross_entropy_unwarped',
             'sampling_cross_entropy', 'sampling_cross_entropy_infinite',
             'target_mass_on_zero_proposal')}
        for m, metrics in block['diagnostics']['top_m'].items():
            for name, metric_values in metrics.items():
                values[f'{name}_at_{m}'] = metric_values
        for i in range(len(block['draft_tokens'])):
            offset = block['absolute_output_offset'] + i
            context = block['input_context_length'] + i
            edge_index = bisect.bisect_right(context_edges, context)
            lower = context_edges[edge_index - 1] if edge_index else 0
            context_label = (f'{lower}-{context_edges[edge_index]}' if edge_index < len(context_edges)
                             else f'{lower}+')
            output_label = OUTPUT_LABELS[bisect.bisect_right(OUTPUT_EDGES[1:], offset)]
            for axis, label in [('all', 'all'), ('output_offset', output_label),
                                ('input_context_length', context_label)]:
                base = {'condition': block['condition'], 'collection_temperature': float(block['collection_temperature']),
                        'sampling_warp_config': block['sampling_warp_config'], 'split': block['split'],
                        'axis': axis, 'bin': label,
                        'data_policy': 'off_policy_teacher' if i > block['first_rejection_index'] else 'on_policy'}
                key = fingerprint(base)
                group = groups.setdefault(key, {'base': base, 'logical_slots': 0, 'metric_totals': {}})
                group['logical_slots'] += 1
                for name, metric_values in values.items():
                    value = metric_values[i]
                    if value is None:
                        continue
                    total = group['metric_totals'].setdefault(name, [0., 0])
                    total[0] += float(value)
                    total[1] += 1
    for group in groups.values():
        group['metrics'] = {name: {'mean': total / count, 'logical_values': count, 'n_prompts': 1}
                            for name, (total, count) in group.pop('metric_totals').items()}
    return {'source_path': str(path), 'source_sha256': row['sha256'],
            'payload_metadata_sha256': fingerprint(payload['metadata']),
            'method': 'Independent scalar traversal of raw stored logical-slot diagnostics; no DiagnosticAccumulator call.',
            'groups': groups}


def prepare():
    require(not BASELINE.exists(), 'Preservation baseline already exists; refusing overwrite')
    manifest = archive_record()
    old_audit = read(ARCHIVE / 'artifacts/m1_audit.json')
    old_diagnostic = read(ARCHIVE / 'artifacts/diagnostic_reference.json')
    old_m2_validation = read(ARCHIVE / 'artifacts/m2_validation_reference.json')
    m2 = read(ROOT / 'artifacts/m2.json')
    require(digest(ROOT / 'artifacts/m2.json') == old_m2_validation['report_sha256'],
            'M2 report differs from the pre-correction validation binding')
    sys.path.insert(0, str(ROOT / 'src'))
    from edraft.diagnose import recompute_generation_diagnostics
    import torch
    records, changed = [], []
    require(len(old_audit['generations']) == len(old_audit['source_shards']), 'Original generation/source coverage differs')
    for source, old_generation in zip(old_audit['source_shards'], old_audit['generations']):
        path = Path(source['path'])
        side_path = path.with_suffix('.pt.json')
        side = read(side_path)
        generation = side['generation']
        require({k: v for k, v in generation.items() if k != 'output_tokens'} == old_generation,
                'Current sidecar metadata differs from pre-correction M1 audit')
        require(side['sha256'] == source['sha256'] and fingerprint(side['identity']) == source['identity_sha256'],
                'Original trace identity/checksum mismatch')
        before = fingerprint(generation)
        current = recompute_generation_diagnostics(generation)
        require(fingerprint(generation) == before, 'Refresh mutated original generation')
        require(recompute_generation_diagnostics(current) == current, 'Refresh is not idempotent')
        expected = {k: v for k, v in current.items() if k != 'output_tokens'}
        record = {'path': str(path), 'sha256': source['sha256'], 'bytes': side['disk_bytes'],
                  'sidecar_path': str(side_path), 'sidecar_sha256': digest(side_path),
                  'identity_sha256': fingerprint(side['identity']), 'generation_sha256': before,
                  'output_tokens_sha256': fingerprint(generation['output_tokens']),
                  'generation_key': generation_key(generation),
                  'expected_refreshed_generation_sha256': fingerprint(expected),
                  'prompt_id': generation['prompt_id'], 'condition': generation['condition'],
                  'split': generation['split'], 'logical_slots': side['logical_slots'],
                  'blocks': side['blocks'], 'primary_eligible': primary(side['identity'])}
        records.append(record)
        if current['repetition_dominated'] != generation['repetition_dominated']:
            changed.append({'prompt_id': current['prompt_id'], 'split': current['split'],
                            'condition': current['condition'], 'old': generation['repetition_dominated'],
                            'new': current['repetition_dominated'], 'periodic_run': current['periodic_run'],
                            'path': str(path), 'logical_slots': side['logical_slots']})
    require(len(changed) == 1 and changed[0]['prompt_id'] == EXPECTED_CHANGED_PROMPT and
            changed[0]['split'] == 'test' and changed[0]['condition'] == 'long_thinking' and
            changed[0]['old'] is False and changed[0]['new'] is True,
            'Unexpected M1 classification correction set')
    fixed = {}
    fixed_paths = [ROOT / 'artifacts/m2.json', ROOT / 'artifacts/m2/latest.pt',
                   ROOT / 'artifacts/calibration.json', ROOT / 'artifacts/prompts.json']
    fixed_paths += [Path(p) for p in m2['checkpoints'].values()]
    fixed_paths += [ROOT / 'artifacts' / name for name in old_audit['calibration']['artifact_sha256']]
    fixed_paths += list((ROOT / 'artifacts/traces').glob('manifest-rank*.json'))
    fixed_paths += [ROOT / 'src/edraft' / name for name in m2['implementation_sha256']]
    for path in sorted(set(fixed_paths)):
        fixed[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        previous = old_m2_validation['evidence'].get(str(path))
        if previous and 'sha256' in previous:
            require(fixed[str(path)]['sha256'] == previous['sha256'],
                    'Fixed input differs from the pre-correction M2 validation binding: ' + str(path))
    for name, expected in m2['implementation_sha256'].items():
        require(fixed[str(ROOT / 'src/edraft' / name)]['sha256'] == expected, 'Fit source changed: ' + name)
    require(not torch.cuda.is_initialized(), 'Prepare unexpectedly initialized CUDA')
    affected = next(row for row in records if row['prompt_id'] == EXPECTED_CHANGED_PROMPT)
    independent_excluded = affected_shard_evidence(affected, manifest['reruns']['diagnose']['resolved_config']['context_edges'])
    baseline = {'schema_version': 1, 'status': 'prepared_before_reference_regeneration',
                'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
                'archive_manifest_sha256': ARCHIVE_HASH,
                'source_sha256': digest(__file__), 'sources': records, 'fixed_files': fixed,
                'source_set_sha256': fingerprint(old_audit['source_shards']),
                'm2_fit_identity': m2['identity'], 'm2_primary_source_fingerprint': m2['trace_index']['source_fingerprint'],
                'm2_trace_index': m2['trace_index'], 'expected_classification_changes': changed,
                'original_audit_sha256': digest(ARCHIVE / 'artifacts/m1_audit.json'),
                'original_diagnostic_sha256': digest(ARCHIVE / 'artifacts/diagnostic_reference.json'),
                'original_group_count': len(old_diagnostic['groups']),
                'independent_affected_shard_evidence': independent_excluded,
                'reruns': manifest['reruns'], 'cuda_initialized': False,
                'raw_hash_policy': 'Original audited SHA-256 retained here; validate rehashes all complete trace bytes independently.'}
    write(BASELINE, baseline)
    return {'status': baseline['status'], 'baseline': str(BASELINE), 'baseline_sha256': digest(BASELINE),
            'sources': len(records), 'fixed_files': len(fixed), 'classification_changes': changed}


def validate(output):
    begun = time.perf_counter()
    archive_record()
    baseline = read(BASELINE)
    require(baseline['source_sha256'] == digest(__file__), 'Validator source changed since preservation snapshot')
    checks = Counter()
    def check(value, label):
        require(value, label)
        checks[label] += 1
    old_audit = read(ARCHIVE / 'artifacts/m1_audit.json')
    old_diagnostic = read(ARCHIVE / 'artifacts/diagnostic_reference.json')
    audit_path = ROOT / 'artifacts/m1_audit.json'
    diagnostic_path = ROOT / 'artifacts/diagnostic_reference.json'
    audit, diagnostic = read(audit_path), read(diagnostic_path)
    check(audit['status'] == 'passed' and audit['scale'] == 'reference', 'm1a_passed_reference')
    check(diagnostic['status'] == 'measured' and diagnostic['scale'] == diagnostic['label'] == 'reference', 'm1b_measured_reference')
    check(digest(audit_path) != baseline['original_audit_sha256'], 'audit_regenerated')
    check(digest(diagnostic_path) != baseline['original_diagnostic_sha256'], 'diagnostic_regenerated')
    for key in ('requirements', 'floors', 'token_counts', 'prompt_counts', 'shards', 'source_shards',
                'long_condition', 'head_reconstruction', 'excluded_shards', 'calibration'):
        check(audit[key] == old_audit[key], 'preserved_audit_' + key)
    for name, expected in baseline['fixed_files'].items():
        path = Path(name)
        check(path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], 'fixed_input_bytes_unchanged')
    expected_by_key = {row['generation_key']: row for row in baseline['sources']}
    check(len(expected_by_key) == len(baseline['sources']) == 2184, 'unique_original_generations')
    actual_by_key = {generation_key(g): g for g in audit['generations']}
    check(len(actual_by_key) == len(audit['generations']) and actual_by_key.keys() == expected_by_key.keys(), 'complete_generation_coverage')
    eligible = []
    for row in baseline['sources']:
        path, side_path = Path(row['path']), Path(row['sidecar_path'])
        check(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'raw_trace_bytes_unchanged')
        check(digest(side_path) == row['sidecar_sha256'], 'sidecar_bytes_unchanged')
        side = read(side_path)
        check(fingerprint(side['generation']) == row['generation_sha256'], 'stored_generation_metadata_unchanged')
        check(fingerprint(side['generation']['output_tokens']) == row['output_tokens_sha256'], 'stored_output_tokens_unchanged')
        check(fingerprint(side['identity']) == row['identity_sha256'], 'trace_identity_unchanged')
        check(fingerprint(actual_by_key[row['generation_key']]) == row['expected_refreshed_generation_sha256'], 'audit_refreshed_generation_matches_prechecked_expectation')
        if row['primary_eligible']:
            eligible.append((str(path), {'identity': side['identity'], 'sha256': side['sha256']}))
    check(fingerprint([item for _, item in sorted(eligible)]) == baseline['m2_primary_source_fingerprint'], 'm2_eligible_training_source_fingerprint_preserved')
    changed = [g for g in audit['generations'] if g['diagnostic_recomputation']['classification_changed']]
    check(len(changed) == 1 and changed[0]['prompt_id'] == EXPECTED_CHANGED_PROMPT and changed[0]['repetition_dominated'], 'one_expected_changed_M1_prompt')
    source_map = [{'path': s['path'], 'sha256': s['sha256']} for s in audit['source_shards']]
    check(diagnostic['source_shards'] == source_map, 'diagnostic_exact_ordered_source_set')
    provenance = diagnostic['audit_provenance']
    check(provenance['audit_sha256'] == digest(audit_path), 'diagnostic_audit_hash_binding')
    check(Path(provenance['audit_path']).resolve() == audit_path.resolve(), 'diagnostic_audit_path_binding')
    check(provenance['source_set_sha256'] == baseline['source_set_sha256'], 'diagnostic_original_source_set_binding')
    check(provenance['frozen_artifact_sha256'] == audit['calibration']['artifact_sha256'], 'diagnostic_calibration_binding')
    check(provenance['prompt_manifest_sha256'] == audit['calibration']['identity']['prompt_manifest_sha256'], 'diagnostic_prompt_binding')
    for mode in ('audit', 'diagnose'):
        run = read(baseline['reruns'][mode]['original_run_record'])
        check(run['config'] == baseline['reruns'][mode]['resolved_config'], mode + '_original_resolved_config_preserved')
        check(run['visible_devices'] == '', mode + '_cpu_only_visible_devices')
        check(fingerprint(run['source_files_sha256']) == run['source_tree_sha256'], mode + '_source_tree_hash_algebra')
        actual_source_paths = {str(p.relative_to(ROOT)) for p in (ROOT / 'src/edraft').glob('*.py')}
        check(set(run['source_files_sha256']) == actual_source_paths, mode + '_complete_source_file_coverage')
        for name, expected in run['source_files_sha256'].items():
            check(digest(ROOT / name) == expected, mode + '_recorded_source_matches_current')
        if mode == 'diagnose':
            check(diagnostic['run'] == run, 'diagnostic_embedded_run_matches_record')
    for key in ('context_bin_edges', 'output_bins', 'outlier_threshold', 'cross_entropy_note', 'off_policy_note'):
        check(diagnostic[key] == old_diagnostic[key], 'diagnostic_reporting_metadata_unchanged_' + key)
    old_groups = {fingerprint({k: g[k] for k in (*GROUP_KEYS, 'inclusion')}): g for g in old_diagnostic['groups']}
    new_groups = {fingerprint({k: g[k] for k in (*GROUP_KEYS, 'inclusion')}): g for g in diagnostic['groups']}
    check(len(new_groups) == len(diagnostic['groups']), 'unique_new_diagnostic_groups')
    for key, group in old_groups.items():
        if not (group['condition'] == 'long_thinking' and group['split'] == 'test'):
            check(new_groups.get(key) == group, 'unaffected_diagnostic_group_bitwise_equal')
    def combine(groups):
        result = defaultdict(dict)
        for group in groups:
            result[fingerprint({k: group[k] for k in GROUP_KEYS})][group['inclusion']] = group
        return result
    old_combined, new_combined = combine(old_diagnostic['groups']), combine(diagnostic['groups'])
    check(old_combined.keys() == new_combined.keys(), 'diagnostic_group_axes_and_policies_preserved')
    independent_excluded = baseline['independent_affected_shard_evidence']['groups']
    check(set(independent_excluded) <= set(old_combined), 'affected_raw_group_keys_in_original_report')
    excluded_slots = 0
    for key, old_members in old_combined.items():
        members = new_combined[key]
        expected_removed = independent_excluded.get(key)
        expected_inclusions = {'primary'}
        if expected_removed is not None:
            expected_inclusions.add('repetition_excluded')
            if old_members['primary']['logical_slots'] == expected_removed['logical_slots']:
                expected_inclusions.remove('primary')
        check(set(members) == expected_inclusions, 'exact_expected_group_inclusions')
        check(sum(g['logical_slots'] for g in old_members.values()) == sum(g['logical_slots'] for g in members.values()), 'logical_slot_conservation_across_inclusion')
        excluded = members.get('repetition_excluded')
        if excluded is None:
            continue
        check(expected_removed is not None and excluded['logical_slots'] == expected_removed['logical_slots'], 'excluded_slots_match_independent_raw_shard')
        check(set(excluded['metrics']) == set(expected_removed['metrics']), 'excluded_metric_key_coverage')
        check(excluded['condition'] == 'long_thinking' and excluded['split'] == 'test', 'exclusion_only_affected_long_test')
        if excluded['axis'] == 'all':
            excluded_slots += excluded['logical_slots']
        old, kept = old_members['primary'], members.get('primary')
        if kept is not None:
            expected_kept_keys = {name for name, metric in old['metrics'].items()
                                  if metric['n_prompts'] - int(name in expected_removed['metrics']) > 0}
            check(set(kept['metrics']) == expected_kept_keys, 'kept_metric_key_coverage')
            for name in expected_kept_keys - set(expected_removed['metrics']):
                check(kept['metrics'][name] == old['metrics'][name], 'unaffected_metric_within_changed_group_unchanged')
        for metric, removed in excluded['metrics'].items():
            check(removed['n_prompts'] == 1, 'excluded_metric_single_prompt')
            check(math.isclose(removed['mean'], expected_removed['metrics'][metric]['mean'], rel_tol=1e-12, abs_tol=1e-12), 'excluded_metric_mean_matches_independent_raw_shard')
            if kept is None or metric not in kept['metrics']:
                continue
            before, after = old['metrics'][metric], kept['metrics'][metric]
            check(before['n_prompts'] == after['n_prompts'] + 1, 'metric_prompt_count_conservation')
            expected_mean = (before['mean'] * before['n_prompts'] - removed['mean']) / after['n_prompts']
            check(math.isclose(after['mean'], expected_mean, rel_tol=1e-10, abs_tol=1e-10), 'removed_prompt_metric_mean_algebra')
    check(excluded_slots == baseline['expected_classification_changes'][0]['logical_slots'], 'exact_affected_shard_slots_excluded')
    generation_groups = defaultdict(list)
    for g in audit['generations']:
        key = fingerprint({k: g[k] for k in ('condition', 'temperature', 'split', 'sampling_warp_config')})
        generation_groups[key].append(g)
    for summary in diagnostic['generation_lengths']:
        key = fingerprint({'condition': summary['condition'], 'temperature': summary['collection_temperature'],
                           'split': summary['split'], 'sampling_warp_config': summary['sampling_warp_config']})
        rows = generation_groups.pop(key)
        kept = [g for g in rows if not (g['condition'].startswith('long_') and g['repetition_dominated'])]
        lengths = [g['realized_length'] for g in kept]
        check(summary['generations'] == len(rows) and summary['primary_generations'] == len(kept) and
              summary['repetition_excluded'] == len(rows) - len(kept), 'generation_summary_inclusion_counts')
        check(summary['realized_length_mean'] == statistics.mean(lengths) and summary['realized_length_median'] == statistics.median(lengths), 'generation_summary_primary_lengths')
        check(summary['primary_generations_reaching_bins'] == {label: sum(n > edge for n in lengths) for edge, label in zip(OUTPUT_EDGES, OUTPUT_LABELS)}, 'generation_summary_primary_bins')
        counts = summary['primary_generations_reaching_bins']
        check(summary['empty_bins'] == [label for label, count in counts.items() if count == 0], 'generation_summary_empty_bins')
        check(summary['minimum_long_generations_per_bin'] == 8, 'generation_summary_minimum_bin_count')
        is_greedy_thinking = summary['condition'] == 'long_thinking' and summary['collection_temperature'] == 0
        check(summary['greedy_long_thinking_diagnostic_only'] == is_greedy_thinking, 'generation_summary_greedy_label')
        check(summary['long_context_conclusions_supported'] == (counts['8K-16K'] >= 8 and not is_greedy_thinking), 'generation_summary_support_gate')
        expected_sparse = any(counts[label] < 8 for label in ('8K-16K', '16K+')) if summary['condition'].startswith('long_') else None
        check(summary['data_sparsity'] == expected_sparse, 'generation_summary_sparsity_gate')
        check(summary['natural_eos_count'] == sum(bool(g.get('eos_reached')) for g in rows) and
              summary['capped_count'] == sum(bool(g.get('capped')) for g in rows), 'generation_summary_realized_termination_counts')
    check(not generation_groups, 'all_generation_summary_conditions_present')
    check('torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized(), 'cuda_uninitialized')
    result = {'scale': 'reference', 'status': 'passed', 'validated_utc': datetime.now(timezone.utc).isoformat(),
              'validation_kind': 'independent_repetition_correction_and_preservation_audit',
              'source_path': str(Path(__file__).resolve()), 'source_sha256': digest(__file__),
              'preservation_baseline': str(BASELINE), 'preservation_baseline_sha256': digest(BASELINE),
              'checks': dict(checks), 'm1_audit': str(audit_path), 'm1_audit_sha256': digest(audit_path),
              'diagnostic_report': str(diagnostic_path), 'diagnostic_report_sha256': digest(diagnostic_path),
              'source_shards': len(baseline['sources']), 'diagnostic_groups': len(diagnostic['groups']),
              'source_set_sha256': baseline['source_set_sha256'],
              'source_tree_sha256': diagnostic['run']['source_tree_sha256'],
              'project_git_commit': diagnostic['run']['project_git_commit'],
              'classification_changes': baseline['expected_classification_changes'],
              'm2_fit_identity_preserved': baseline['m2_fit_identity'], 'cuda_initialized': False,
              'workers': diagnostic['run']['config']['workers'], 'elapsed_seconds': time.perf_counter() - begun,
              'limitations': ['Changed long-test bootstrap intervals, feature moments, acceptance-length/position summaries, rejection ranks, and recoverability fields were not independently recomputed.',
                              'All trace bytes and sidecar metadata were independently rehashed; immutable original labels remain historical evidence.']}
    write(output, result)
    return {'status': 'passed', 'output': str(output), 'sha256': digest(output),
            'checks': sum(checks.values()), 'elapsed_seconds': result['elapsed_seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'validate'))
    parser.add_argument('--output', type=Path, default=REVIEW / 'm1_correction_validation.json')
    args = parser.parse_args()
    print(json.dumps(prepare() if args.mode == 'prepare' else validate(args.output), indent=2))
