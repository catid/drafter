"""Read-only mutation probes for the independent M3 artifact audit, using smoke rows."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'reference', 'full'), default='smoke')
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('/home/catid/drafter'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.scale != 'smoke':
        parser.error('these in-memory mutation fixtures require existing smoke artifacts')
    cfg = SimpleNamespace(artifacts=args.artifacts, repo=args.repo, scale='smoke', seed=2026)
    cases = []

    def record(name, should_pass, operation):
        try:
            result = operation()
            passed = not result.failures
            detail = result.failures[:3]
        except (ValueError, KeyError, TypeError, OSError, IndexError, AttributeError) as error:
            passed, detail = False, [str(error)]
        cases.append({'case': name, 'expected_pass': should_pass, 'actual_pass': passed,
                      'matched_expectation': passed == should_pass, 'detail': detail})

    def row_probe(arm, is_profile=False, mutate=lambda r: None):
        audit = validate.Audit(cfg)
        audit.prepare()
        mode = 'timing' if is_profile else 'evaluate'
        entries = audit.manifests(mode)
        entry = next(e for e in entries.values() if e['identity']['phase_profile'] is is_profile
                     and e['identity']['proposal_mode'] == arm)
        identity = copy.deepcopy(entry['identity'])
        path = Path(entry['result_path']).resolve()
        row = validate.read(path)
        mutate(row)
        payload = json.dumps(row, sort_keys=True).encode()
        virtual_digest = hashlib.sha256(payload).hexdigest()
        entries = copy.deepcopy(entries)
        entries[validate.fingerprint(identity)]['result_sha256'] = virtual_digest
        original = audit.artifact

        def virtual_artifact(requested):
            if Path(requested).resolve() == path:
                audit.files[str(path)] = virtual_digest
                return copy.deepcopy(row)
            return original(requested)

        audit.artifact = virtual_artifact
        audit.row(identity, entries, 'profile' if is_profile else 'evaluate')
        return audit

    for arm in ('plain_dflash', 'online_bigram', 'dense', 'dense_sparse'):
        record('valid_eval_' + arm, True, lambda arm=arm: row_probe(arm, False))
        record('valid_profile_' + arm, True, lambda arm=arm: row_probe(arm, True))

    mutations = [
        ('forbidden_baseline_reuse_on_m3', 'plain_dflash', False, lambda r: r.__setitem__('baseline_reuse', {'status': 'reused'})),
        ('unexpected_online_dict_on_m3', 'dense_sparse', False, lambda r: r.__setitem__('online', {'u': 1})),
        ('legacy_repetition_detector_v1', 'dense', False, lambda r: r['generation'].__setitem__('repetition_detector_version', 1)),
        ('candidate_restricted_proposal_mode', 'dense', False, lambda r: r['blocks'][0].__setitem__('proposal_mode', 'dense_candidate')),
        ('nonzero_block_adapter_version', 'dense_sparse', False, lambda r: r['blocks'][0].__setitem__('next_adapter_version', 1)),
        ('wrong_loss_temperature', 'dense', False, lambda r: r['blocks'][0].__setitem__('loss_temperature', 0.7)),
        ('invalid_draft_tokens_length', 'dense', False, lambda r: r['blocks'][0]['draft_tokens'].append(0)),
        ('invalid_physical_to_logical', 'dense', False, lambda r: r['blocks'][0]['physical_to_logical'].__setitem__('anchor_physical_index', 1)),
        ('corrupted_committed_stream', 'dense', False, lambda r: r['blocks'][0]['committed_tokens'].append(0)),
        ('corrupted_mean_accepted_length', 'dense', False, lambda r: r.__setitem__('mean_accepted_length', r['mean_accepted_length'] + 0.1)),
        ('corrupted_target_forwards_ratio', 'plain_dflash', False, lambda r: r.__setitem__('target_forwards_per_generated_token', 9.9)),
        ('evicted_bigram_k_ret', 'online_bigram', False, lambda r: r['bigram'].__setitem__('k_ret', 256)),
        ('missing_bigram_metadata', 'online_bigram', False, lambda r: r.__setitem__('bigram', None)),
        ('unexpected_phases_on_unprofiled', 'dense', False, lambda r: r.__setitem__('phase_ms', {'dflash_draft': 1.0})),
        ('missing_dense_adapter_score_phase', 'dense', True, lambda r: r['phase_ms'].pop('dense_adapter_score')),
        ('missing_sparse_lookup_phase', 'dense_sparse', True, lambda r: r['phase_ms'].pop('sparse_lookup')),
        ('missing_bigram_score_phase', 'online_bigram', True, lambda r: r['phase_ms'].pop('bigram_score')),
        ('missing_online_update_phase', 'online_bigram', True, lambda r: r['phase_ms'].pop('online_update')),
        ('phase_profile_numeric_false', 'dense', False, lambda r: r.__setitem__('phase_profile', 0)),
        ('wrong_thinking_flag', 'dense', False, lambda r: r['input_metadata'].__setitem__('enable_thinking', not r['input_metadata']['enable_thinking'])),
        ('invalid_output_token_bool', 'dense', False, lambda r: r['output_tokens'].__setitem__(0, True)),
    ]
    for name, arm, is_prof, mut in mutations:
        record(name, False, lambda arm=arm, is_prof=is_prof, mut=mut: row_probe(arm, is_prof, mut))

    def ref_bigram_phase_probe(mutate):
        ref_cfg = SimpleNamespace(artifacts=args.artifacts, repo=args.repo, scale='reference', scope='full', seed=2026)
        audit = validate.Audit(ref_cfg)
        gen_env = {'stop_token_ids': [99], 'context_limit': 32768}
        audit.settings = {
            'model_pair': ['Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16'],
            'model_revisions': ['rev-t', 'rev-d'],
            'checkpoint_hashes': {'dense': 'd', 'dense_sparse': 'ds', 'projection': 'p', 'implementation': 'i'},
            'generation_environment': gen_env,
            'generation_caps': validate.CAPS,
        }
        audit.frozen = 'f' * 64
        prompt = {'prompt_id': 'p0', 'split': 'test', 'condition': 'standard_nonthinking', 'text': 'hi'}
        audit.prompts = {'dev': [], 'test': [prompt]}
        sample = ('greedy', 0.0, {'top_k': 0, 'top_p': 1.0, 'min_p': 0.0})
        setting = {'lambda_tr': 0.05, 'alpha': 0.5, 'k_ret': None}
        identity = audit.identity(prompt, 'online_bigram', setting, sample, 2026, 'timing', 'sess0', 'phase_profile:epoch=e0', True)
        key = validate.fingerprint(identity)
        vpath = (args.artifacts / 'm3/timing/rank0' / f'{key}.json').resolve()
        row = {
            'identity': identity, 'rank': 0, 'scale': 'reference', 'label': 'reference', 'stage': 'm3',
            'arm': 'online_bigram', 'prompt_id': 'p0', 'split': 'test', 'condition': 'standard_nonthinking',
            'seed': 2026, 'temperature': 0.0, 'sampling_name': 'greedy',
            'sampling_warp_config': sample[2], 'settings': setting, 'effective_beta': 1.0,
            'verification_rule': 'greedy', 'model_pair': audit.settings['model_pair'],
            'model_revisions': audit.settings['model_revisions'], 'checkpoint_hashes': audit.settings['checkpoint_hashes'],
            'phase_profile': True, 'timing_session': 'sess0', 'repeat': 'phase_profile:epoch=e0', 'timing_isolated': True,
            'input_metadata': {
                'generation_environment_sha256': validate.fingerprint(gen_env),
                'input_token_count': 10, 'enable_thinking': False, 'input_token_ids_sha256': 'a' * 64,
                'requested_generation_cap': 2048, 'effective_generation_cap': 2048,
            },
            'output_tokens': [7, 99], 'generated_tokens': 2,
            'generation': {'complete': True, 'capped': False, 'eos_reached': True,
                           'repetition_detector_version': 2, 'repetition_dominated': False, 'realized_length': 2},
            'elapsed_seconds': 0.5, 'generation_loop_elapsed_seconds': 0.4, 'tokens_per_second': 4.0,
            'peak_allocated_bytes': 1024, 'peak_reserved_bytes': 2048,
            'phase_ms': {'dflash_draft': 1.0, 'sequential_sampling': 0.5, 'target_verification': 2.0,
                         'target_feature_extraction': 0.3, 'bigram_score': 0.0, 'online_update': 0.0},
            'bigram': {'k_ret': None, 'ledger_retained_for_exact_counts': True,
                       'counted_transitions': 2, 'exact_ledger_pairs': 2, 'retained_prior_pairs': 2},
            'parameter_norms': {},
            'blocks': [{
                'round_idx': 1, 'absolute_output_offset': 1, 'verified_prefix_tail': [7],
                'draft_tokens': [0] * 15,
                'physical_to_logical': {'anchor_physical_index': 0, 'logical_to_physical': list(range(1, 16))},
                'accepted_length': 0, 'decisions': [False], 'first_rejection_index': 0,
                'bonus_token': None, 'replacement_token': 99, 'committed_tokens': [99],
                'proposal_token_probs': None, 'target_token_probs': None,
                'adapter_version': 0, 'next_adapter_version': 0, 'proposal_mode': 'plain_dflash',
                'verification_rule': 'greedy', 'collection_temperature': 0.0, 'loss_temperature': 1.0,
                'sampling_warp_config': sample[2],
            }],
            'mean_accepted_length': 0.0, 'median_accepted_length': 0.0, 'target_forwards_per_generated_token': 1.0,
        }
        mutate(row)
        vdigest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        entries = {key: {'status': 'complete', 'identity': identity, 'result_path': str(vpath), 'result_sha256': vdigest}}
        audit.entry_ranks[('timing', key)] = 0
        audit.artifact = lambda requested: (audit.files.__setitem__(str(vpath), vdigest) or copy.deepcopy(row))
        audit.row(identity, entries, 'profile')
        return audit

    record('ref_zero_ms_bigram_phases_allowed', True, lambda: ref_bigram_phase_probe(lambda r: None))
    record('ref_missing_online_update_rejected', False, lambda: ref_bigram_phase_probe(lambda r: r['phase_ms'].pop('online_update')))
    record('ref_zero_ms_target_verification_rejected', False, lambda: ref_bigram_phase_probe(lambda r: r['phase_ms'].__setitem__('target_verification', 0.0)))

    def full_probe(name):
        audit = validate.Audit(cfg)
        original_artifact, original_manifests = audit.artifact, audit.manifests

        def altered_artifact(path):
            value = original_artifact(path)
            p = Path(path)
            if name.startswith('bundle_') and p.parent.name == 'sessions' and 'tune' in p.parts:
                value = copy.deepcopy(value)
                bundle_key = next(iter(value['bundles']))
                bundle = value['bundles'][bundle_key]
                if name == 'bundle_missing':
                    del value['bundles'][bundle_key]
                elif name == 'bundle_swap_paths':
                    bundle['result_paths'][0], bundle['result_paths'][1] = bundle['result_paths'][1], bundle['result_paths'][0]
                elif name == 'bundle_unknown_epoch':
                    bundle['epoch'] = 'unrecorded'
            elif name == 'corrupted_greedy_mismatch' and p.name == 'greedy_output_comparisons.json':
                value = copy.deepcopy(value)
                value['comparisons'][0]['mismatched_tokens'] += 1
            elif name == 'incomplete_summary' and p.name == 'summary.json':
                value = copy.deepcopy(value)
                value['isolated_timing_complete'] = False
            return value

        def altered_manifests(mode):
            entries = original_manifests(mode)
            if mode == 'evaluate' and name in ('extra_current_identity', 'allowed_historical_identity', 'missing_evaluation_row'):
                entries = copy.deepcopy(entries)
                k = next(iter(entries))
                if name == 'missing_evaluation_row':
                    del entries[k]
                else:
                    extra = copy.deepcopy(entries[k])
                    extra['identity']['seed'] = 9999
                    if name == 'allowed_historical_identity':
                        extra['identity']['frozen_settings_hash'] = 'historical'
                    entries[validate.fingerprint(extra['identity'])] = extra
            return entries

        audit.artifact, audit.manifests = altered_artifact, altered_manifests
        if name == 'changed_input_at_final_recheck':
            original_digest = validate.digest
            with patch.object(validate, 'digest', side_effect=lambda p: '0' * 64 if Path(p).resolve() ==
                              (args.artifacts / 'm3/settings.json').resolve() else original_digest(p)):
                audit.run()
        else:
            audit.run()
        return audit

    for name in ('valid_complete_smoke', 'valid_tune_scope_smoke', 'allowed_historical_identity', 'bundle_missing', 'bundle_swap_paths',
                 'bundle_unknown_epoch', 'extra_current_identity', 'missing_evaluation_row',
                 'corrupted_greedy_mismatch', 'incomplete_summary', 'changed_input_at_final_recheck'):
        if name == 'valid_tune_scope_smoke':
            def tune_scope_probe():
                tcfg = SimpleNamespace(artifacts=args.artifacts, repo=args.repo, scale='smoke', scope='tune', seed=2026)
                audit = validate.Audit(tcfg)
                audit.run()
                return audit
            record(name, True, tune_scope_probe)
        else:
            record(name, name in ('valid_complete_smoke', 'allowed_historical_identity'), lambda name=name: full_probe(name))

    result = {
        'status': 'passed' if all(c['matched_expectation'] for c in cases) else 'failed',
        'scale': 'smoke',
        'label': 'synthetic',
        'cases': cases,
        'source_sha256': {p.name: validate.digest(p) for p in Path(__file__).parent.glob('*.py')},
        'limitations': [
            'Virtual in-memory row/session/manifest/summary mutations; original experiment artifacts remain unchanged.',
            'Row probes check rejection at the row boundary; full probes exercise schedule, greedy comparisons, summary, and final hash checks.',
        ],
    }
    validate.atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'cases': len(cases),
                      'unexpected': [c for c in cases if not c['matched_expectation']]}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
