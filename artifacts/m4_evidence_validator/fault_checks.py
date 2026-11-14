"""Read-only mutation probes for the independent artifact audit, using smoke rows."""
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
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.scale != 'smoke':
        parser.error('these in-memory mutation fixtures require existing smoke artifacts')
    cfg = SimpleNamespace(artifacts=args.artifacts, repo=Path(__file__).resolve().parents[2], scale='smoke', seed=2026)
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

    def row_probe(kind, mutate=lambda r: None):
        audit = validate.Audit(cfg)
        audit.prepare()
        is_profile = kind.endswith('profile')
        mode = 'timing' if is_profile else 'evaluate'
        entries = audit.manifests(mode)
        if is_profile:
            entry = next(e for e in entries.values() if e['identity']['phase_profile']
                         and e['identity']['proposal_mode'] == ('plain_dflash' if kind == 'baseline_profile' else 'online_dbS'))
        else:
            entry = next(e for e in entries.values() if e['identity']['proposal_mode'] ==
                         ('plain_dflash' if kind == 'reused' else 'online_dbS'))
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

    for kind in ('online', 'reused', 'profile', 'baseline_profile'):
        record('valid_' + kind + '_row', True, lambda kind=kind: row_probe(kind))
    for key in ('generation_loop_elapsed_seconds', 'estimated_online_state_bandwidth_GBps'):
        record('reused_' + key, False, lambda key=key: row_probe('reused', lambda r: r.__setitem__(key, 123)))
    record('unrelated_reuse_settings', False, lambda: row_probe('reused', lambda r: r['baseline_reuse'].update(
        source_settings_path=str((args.artifacts / 'm4/timing.json').resolve()),
        source_settings_sha256=validate.digest(args.artifacts / 'm4/timing.json'))))
    mutations = {
        'phase_profile_numeric_false': lambda r: r.__setitem__('phase_profile', 0),
        'wrong_thinking_flag': lambda r: r['input_metadata'].__setitem__('enable_thinking', not r['input_metadata']['enable_thinking']),
        'invalid_input_hash': lambda r: r['input_metadata'].__setitem__('input_token_ids_sha256', 'bad'),
        'wrong_rank': lambda r: r.__setitem__('rank', 1),
        'incomplete_generation': lambda r: r['generation'].__setitem__('complete', False),
        'wrong_eos_flag': lambda r: r['generation'].__setitem__('eos_reached', True),
        'numeric_capped_flag': lambda r: r['generation'].__setitem__('capped', 1),
        'invalid_output_token': lambda r: r['output_tokens'].__setitem__(0, True),
        'wrong_block_warper': lambda r: r['blocks'][0]['sampling_warp_config'].__setitem__('top_p', .1),
        'wrong_update_count': lambda r: r['online'].__setitem__('u', 999),
        'bias_only_dense_delta': lambda r: r['online'].__setitem__('dW_norm', .5),
        'missing_nullable_loss': lambda r: r['online'].pop('last_full_loss'),
    }
    for name, mutation in mutations.items():
        record(name, False, lambda mutation=mutation: row_probe('online', mutation))
    for phase in ('dflash_draft', 'dense_adapter_score', 'sparse_lookup', 'sequential_sampling', 'target_verification', 'online_update'):
        record('missing_profile_' + phase, False, lambda phase=phase: row_probe('profile', lambda r: r['phase_ms'].pop(phase)))
    for phase in ('dflash_draft', 'sequential_sampling', 'target_verification'):
        record('missing_baseline_profile_' + phase, False,
               lambda phase=phase: row_probe('baseline_profile', lambda r: r['phase_ms'].pop(phase)))

    def full_probe(name):
        audit = validate.Audit(cfg)
        original_artifact, original_manifests = audit.artifact, audit.manifests

        def altered_artifact(path):
            value = original_artifact(path)
            if name.startswith('bundle_') and Path(path).parent.name == 'sessions' and 'tune' in Path(path).parts:
                value = copy.deepcopy(value)
                bundle_key = next(iter(value['bundles']))
                bundle = value['bundles'][bundle_key]
                if name == 'bundle_missing':
                    del value['bundles'][bundle_key]
                elif name == 'bundle_swap_paths':
                    bundle['result_paths'][0], bundle['result_paths'][1] = bundle['result_paths'][1], bundle['result_paths'][0]
                elif name == 'bundle_unknown_epoch':
                    bundle['epoch'] = 'unrecorded'
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
                              (args.artifacts / 'm4/settings.json').resolve() else original_digest(p)):
                audit.run()
        else:
            audit.run()
        return audit

    for name in ('valid_complete_smoke', 'allowed_historical_identity', 'bundle_missing', 'bundle_swap_paths',
                 'bundle_unknown_epoch', 'extra_current_identity', 'missing_evaluation_row', 'changed_input_at_final_recheck'):
        record(name, name in ('valid_complete_smoke', 'allowed_historical_identity'), lambda name=name: full_probe(name))
    result = {'status': 'passed' if all(c['matched_expectation'] for c in cases) else 'failed',
              'scale': 'smoke', 'label': 'synthetic', 'cases': cases,
              'source_sha256': {p.name: validate.digest(p) for p in Path(__file__).parent.glob('*.py')},
              'limitations': ['Virtual in-memory row/session/manifest mutations; original experiment artifacts remain unchanged.',
                              'Row probes check rejection at the row boundary; full probes exercise schedule and final hash checks.']}
    validate.atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'cases': len(cases),
                      'unexpected': [c for c in cases if not c['matched_expectation']]}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
