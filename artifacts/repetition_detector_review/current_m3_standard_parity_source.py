"""Bounded CPU-only old/new standard-dev parity audit; no model/code imports."""
import argparse
import collections
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scale', choices=['reference'], default='reference')
args = parser.parse_args()
ROOT = Path('/home/catid/drafter')
OLD_ROOT = ROOT / 'artifacts/repetition_detector_review/pre_correction'
OLD_SESSION = '672dec894a4447c9b2f59eb3411bcdcf'
NEW_SESSION = '00df5018aa1946d6821bd38a71de77b8'
OUT = ROOT / 'artifacts/repetition_detector_review/current_m3_standard_parity.json'
started = time.monotonic()
evidence = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def fingerprint(value):
    return digest(canonical(value).encode())


def read(path):
    payload = path.read_bytes()
    evidence[str(path)] = {'sha256': digest(payload), 'bytes': len(payload)}
    return json.loads(payload)


def condition_key(identity):
    return identity['prompt'], identity['proposal_mode'], canonical(identity['setting'])


def session_record(root, session_id):
    records = [(p, read(p)) for p in (root / 'artifacts/m3/tune/sessions').glob('*.json')]
    selected = [(p, s) for p, s in records if s['logical_session'] == session_id]
    assert len(selected) == 1
    path, record = selected[0]
    assert path.stem == fingerprint(record['configuration'])
    return path, record


def normalize_identity(identity):
    value = deepcopy(identity)
    for key in ('timing_session', 'repeat'):
        value.pop(key)
    value['checkpoint'].pop('implementation')
    return value


TIMING_FIELDS = {'elapsed_seconds', 'generation_loop_elapsed_seconds', 'tokens_per_second',
                 'peak_allocated_bytes', 'peak_reserved_bytes'}


def core(row):
    value = deepcopy(row)
    for key in TIMING_FIELDS | {'generation', 'timing_session', 'repeat'}:
        value.pop(key)
    value['identity'] = normalize_identity(value['identity'])
    value['checkpoint_hashes'].pop('implementation')
    return value


def differences(a, b, path=''):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        found = []
        for key in sorted(a.keys() | b.keys()):
            child = path + '/' + key
            found.extend([child] if key not in a or key not in b else differences(a[key], b[key], child))
        return found
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + '/length']
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differences(x, y, path + '/' + str(i))]
    return [] if a == b else [path]


def exact_run(tokens):
    best, detail = 0, {'start': None, 'end': None, 'period': None, 'complete_cycles': 0}
    for period in range(1, len(tokens) // 4 + 1):
        run = 0
        for i in range(period, len(tokens)):
            run = run + 1 if tokens[i] == tokens[i - period] else 0
            if run >= 3 * period and run + period > best:
                best = run + period
                detail = {'start': i + 1 - best, 'end': i + 1, 'period': period,
                          'complete_cycles': best // period}
    return best, detail


old_manifest = read(OLD_ROOT / 'artifacts/m3/tune/manifest-rank0.json')
new_manifest = read(ROOT / 'artifacts/m3/tune/manifest-rank0.json')
old_session_path, old_session = session_record(OLD_ROOT, OLD_SESSION)
new_session_path, new_session = session_record(ROOT, NEW_SESSION)
old_config, new_config = deepcopy(old_session['configuration']), deepcopy(new_session['configuration'])
old_impl = old_config['checkpoint_hashes'].pop('implementation')
new_impl = new_config['checkpoint_hashes'].pop('implementation')
assert old_config == new_config and old_impl != new_impl
implementation = hashlib.sha256()
for name in ('adapter.py', 'online.py', 'features.py', 'dflash_loop.py', 'sampler.py',
             'verify.py', 'backend.py', 'sparse.py', 'toy.py', 'diagnose.py', 'eval_closed_loop.py'):
    path = ROOT / 'src/edraft' / name
    implementation.update(path.name.encode())
    implementation.update(path.read_bytes())
assert implementation.hexdigest() == new_impl
prompts = read(ROOT / 'artifacts/prompts.json')
assert fingerprint(prompts) == new_config['configuration']['prompt_manifest_hash']
prompt_by_id = {p['prompt_id']: p for p in prompts['prompts']}
standard_ids = [p for p in new_config['configuration']['dev_ids']
                if prompt_by_id[p]['condition'] == 'standard_nonthinking']
assert len(standard_ids) == 8
expected = {(pid, arm, canonical(setting)) for pid in standard_ids
            for arm, settings in new_config['configuration']['candidates'].items() for setting in settings}
assert len(expected) == 56


def selected_rows(manifest, session_id, archive=False):
    result = {}
    for key, entry in manifest['jobs'].items():
        identity = entry['identity']
        if identity['timing_session'] != session_id or identity['prompt'] not in standard_ids:
            continue
        assert entry['status'] == 'complete' and fingerprint(identity) == key
        original = Path(entry['result_path'])
        path = OLD_ROOT / original.relative_to(ROOT) if archive else original
        row = read(path)
        assert evidence[str(path)]['sha256'] == entry['result_sha256']
        assert row['identity'] == identity and row['timing_session'] == session_id
        assert identity['prompt_payload_hash'] == fingerprint(prompt_by_id[identity['prompt']])
        match = condition_key(identity)
        assert match not in result
        result[match] = (row, path, entry)
    assert set(result) == expected, f'Wait for all eight standard bundles: found {len(result)}/56 rows'
    return result


old_rows = selected_rows(old_manifest, OLD_SESSION, archive=True)
new_rows = selected_rows(new_manifest, NEW_SESSION)
for records, session in [(old_rows, old_session), (new_rows, new_session)]:
    epochs = {e['epoch'] for e in session['epochs']}
    for pid in standard_ids:
        entries = [entry for (p, _, _), (_, _, entry) in records.items() if p == pid]
        paths = {entry['result_path'] for entry in entries}
        bundles = [b for b in session['bundles'].values() if set(b['result_paths']) == paths]
        assert len(bundles) == 1 and bundles[0]['status'] == 'complete'
        bundle = bundles[0]
        assert len(bundle['result_paths']) == 7 and bundle['epoch'] in epochs
        assert all(e['identity']['repeat'] == 'dev:epoch=' + bundle['epoch'] for e in entries)

results, discrepancies = [], []
detector_changes = collections.Counter()
for key in sorted(expected):
    old, old_path, old_entry = old_rows[key]
    new, new_path, new_entry = new_rows[key]
    paths = differences(core(old), core(new))
    if paths:
        discrepancies.append({'match': key, 'differing_fields': len(paths), 'first_paths': paths[:30]})
    prior, current = old['generation'], new['generation']
    assert current['repetition_detector_version'] == 2
    span, periodic = exact_run(new['output_tokens'])
    assert current['longest_periodic_run_tokens'] == span and current['periodic_run'] == periodic
    n = len(new['output_tokens'])
    flag = n >= 256 and (span >= max(256, n / 2) or
                         current['repeated_ngram_fraction']['8'] >= .8
                         and current['dominant_ngram_token_coverage']['16'] >= .5)
    assert current['repetition_dominated'] == flag
    diagnostic_differences = differences(prior, current)
    for path in diagnostic_differences:
        detector_changes[path] += 1
    results.append({'prompt_id': key[0], 'arm': key[1], 'settings': new['settings'],
                    'old_path': str(old_path), 'old_sha256': old_entry['result_sha256'],
                    'new_path': str(new_path), 'new_sha256': new_entry['result_sha256'],
                    'output_tokens_equal': old['output_tokens'] == new['output_tokens'],
                    'full_verification_blocks_equal': old['blocks'] == new['blocks'],
                    'parameter_norms_equal': old['parameter_norms'] == new['parameter_norms'],
                    'core_exact_equal': not paths, 'tokens': len(new['output_tokens']),
                    'blocks': len(new['blocks']), 'diagnostic_differences': diagnostic_differences,
                    'classification_changed': prior['repetition_dominated'] != current['repetition_dominated']})
source = Path(__file__).resolve()
report = {'status': 'passed' if not discrepancies else 'differences_require_investigation',
    'scale': args.scale, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Eight standard development prompts x seven candidates; long outputs and later test/timing runs excluded.',
    'source_path': str(source), 'source_sha256': digest(source.read_bytes()),
    'old_session': OLD_SESSION, 'new_session': NEW_SESSION,
    'old_implementation': old_impl, 'new_implementation': new_impl,
    'snapshot': {'old_manifest_jobs': len(old_manifest['jobs']), 'new_manifest_jobs': len(new_manifest['jobs']),
                 'old_session_path': str(old_session_path), 'new_session_path': str(new_session_path),
                 'manifest_and_session_hashes_are_snapshot_only': True},
    'coverage': {'prompts': len(standard_ids), 'candidate_rows': len(results),
                 'blocks': sum(r['blocks'] for r in results), 'tokens': sum(r['tokens'] for r in results)},
    'excluded_comparison_fields': sorted(TIMING_FIELDS) + ['generation (compared separately)',
        'identity.checkpoint.implementation', 'checkpoint_hashes.implementation',
        'timing_session', 'repeat', 'identity.timing_session', 'identity.repeat'],
    'detector_field_change_counts': dict(detector_changes),
    'classification_changes': sum(r['classification_changed'] for r in results),
    'discrepancies': discrepancies, 'rows': results, 'evidence': evidence,
    'limitations': ['Exact recorded outputs/blocks establish parity only for these 56 requests, not all future or long outputs.',
        'No model inference, hidden runtime tensors, RNG state, normalization, or unseen execution was independently inspected.',
        'Timing and peak memory are deliberately not parity requirements; this is not a speed comparison.',
        'Session/manifest files may grow after the captured snapshots; immutable raw rows and their hashes identify the audited scope.'],
    'cuda_initialized': False, 'execution': 'Python standard library only; no model/code imports',
    'elapsed_seconds': time.monotonic() - started}
temporary = None
try:
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=OUT.parent,
                                     prefix=OUT.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, OUT)
finally:
    if temporary is not None:
        temporary.unlink(missing_ok=True)
print(json.dumps({'status': report['status'], 'output': str(OUT), 'sha256': digest(OUT.read_bytes()),
                  'coverage': report['coverage'], 'discrepancies': len(discrepancies),
                  'classification_changes': report['classification_changes'],
                  'detector_field_change_counts': report['detector_field_change_counts']}))
