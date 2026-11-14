"""CPU-only, bounded independent audit of the first corrected M3 long-dev bundle.

This preserves and reruns the earlier in-memory audit; it is not a byte-for-byte
capture of that unsaved interactive command. It imports no experiment/model code.
Writes only its selected report and content-addressed mutable-input snapshots.
Use --output for a fresh report path when retaining prior evidence. All experiment
inputs are left unchanged. Optional --config is a JSON object with scale/output.
Run from the repository:
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python artifacts/m3_first_long_bundle_validation_source.py --scale reference
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, help='Optional JSON configuration with scale/output keys.')
parser.add_argument('--scale', choices=('smoke', 'reference', 'full'), default='reference')
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/m3_first_long_bundle_validation.json')
preliminary, _ = parser.parse_known_args()
config_raw = None
if preliminary.config is not None:
    try:
        config_raw = preliminary.config.read_bytes()
        config_values = json.loads(config_raw)
        if not isinstance(config_values, dict) or set(config_values) - {'scale', 'output'}:
            parser.error('--config must contain a JSON object with only scale/output keys')
        if 'scale' in config_values and config_values['scale'] not in ('smoke', 'reference', 'full'):
            parser.error('configuration scale must be smoke, reference, or full')
        if 'output' in config_values and not isinstance(config_values['output'], str):
            parser.error('configuration output must be a path string')
        parser.set_defaults(**config_values)
    except (OSError, ValueError) as error:
        parser.error(str(error))
args = parser.parse_args()
if args.scale != 'reference':
    parser.error('This fixed seven-row reference audit does not support smoke/full scales.')
OUTPUT = args.output.resolve()
SNAPSHOTS = ROOT / 'artifacts/m3_first_long_bundle_validation_inputs'
SESSION = '00df5018aa1946d6821bd38a71de77b8'
IMPLEMENTATION = '60da3e0c57fee2fbd9cd7ce756c66c56ba5642a63da27ef428b9b30538a459df'
SESSION_PATH = ROOT / 'artifacts/m3/tune/sessions/baa0c5cee13bb21d83155d21791bbe56bcc00e8995f9880fa5a44278baced860.json'
BUNDLE = '5c843a8fdefde35195b7c5fdbdd9d0d1137553b2a0baff7bfcc939e677d559c6'
PROMPT = 'math500-long-0176'
checks = Counter()
evidence = {}
started = time.monotonic()
PROJECT_COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
RESOLVED_CONFIG = {'scale': args.scale, 'config': str(args.config.resolve()) if args.config else None,
                   'output': str(OUTPUT), 'snapshot_directory': str(SNAPSHOTS), 'seed': 2026,
                   'logical_session': SESSION, 'implementation': IMPLEMENTATION,
                   'session_path': str(SESSION_PATH), 'bundle_key': BUNDLE, 'prompt_id': PROMPT}
print(json.dumps({'resolved_config': RESOLVED_CONFIG, 'project_commit': PROJECT_COMMIT, 'seed': 2026}), flush=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def publish_bytes(path, raw):
    """Flush a sibling temporary file, then atomically publish the completed bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require(condition, label):
    if not condition:
        raise AssertionError(label)
    checks[label] += 1


def read(path, mutable=False):
    raw = path.read_bytes()
    record = {'sha256': sha(raw), 'bytes': len(raw), 'mutable': mutable}
    if mutable:
        SNAPSHOTS.mkdir(exist_ok=True)
        saved = SNAPSHOTS / (path.stem + '-' + sha(raw) + '.json')
        if not saved.exists():
            publish_bytes(saved, raw)
        require(saved.read_bytes() == raw, 'captured_mutable_snapshot_bytes')
        record['snapshot_path'] = str(saved.relative_to(ROOT))
    evidence[str(path.relative_to(ROOT))] = record
    return json.loads(raw)


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def exact_periodic(tokens):
    """Exhaustive independent NumPy equality-run scan, all P <= floor(N/4)."""
    values = np.asarray(tokens, dtype=np.int64)
    best_length, best = 0, None
    for period in range(1, len(tokens) // 4 + 1):
        equal = values[period:] == values[:-period]
        if np.count_nonzero(equal) < 3 * period:
            continue
        false_positions = np.flatnonzero(~equal)
        boundaries = np.concatenate(([-1], false_positions, [len(equal)]))
        runs = np.diff(boundaries) - 1
        index = int(runs.argmax())
        run = int(runs[index])
        span = run + period
        if run >= 3 * period and span > best_length:
            start = int(boundaries[index]) + 1
            best_length = span
            best = {'start': start, 'end': start + span, 'period': period,
                    'complete_cycles': span // period}
    return best_length, best


require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'cuda_hidden')
source_path = Path(__file__).resolve()
evidence[str(source_path.relative_to(ROOT))] = {'sha256': sha(source_path.read_bytes())}
if config_raw is not None:
    evidence[str(args.config.resolve())] = {'sha256': sha(config_raw), 'bytes': len(config_raw)}
session = read(SESSION_PATH, mutable=True)
manifest = read(ROOT / 'artifacts/m3/tune/manifest-rank0.json', mutable=True)
prompts = read(ROOT / 'artifacts/prompts.json')
cfg = session['configuration']
plan = cfg['configuration']
prompt = next(p for p in prompts['prompts'] if p['prompt_id'] == PROMPT)
require(fingerprint(cfg) == SESSION_PATH.stem, 'session_configuration_filename')
require(session['logical_session'] == SESSION, 'current_corrected_session')
require(cfg['checkpoint_hashes']['implementation'] == IMPLEMENTATION, 'corrected_implementation_binding')
require(cfg['stage'] == 'm3' and cfg['scale'] == 'reference' and cfg['seed'] == 2026,
        'reference_m3_session_seed')
require(fingerprint(prompts) == plan['prompt_manifest_hash'], 'prompt_manifest_binding')
require(prompt['split'] == 'dev' and prompt['condition'] == 'long_thinking', 'long_dev_prompt')
require(PROMPT in plan['dev_ids'], 'prompt_planned_for_tuning')
bundle = session['bundles'][BUNDLE]
require(bundle['status'] == 'complete', 'bundle_complete')
require(bundle['epoch'] in {e['epoch'] for e in session['epochs']}, 'known_epoch')
require(len(bundle['result_paths']) == len(set(bundle['result_paths'])) == 7, 'seven_unique_rows')
expected_candidates = [(a, s) for a, settings in plan['candidates'].items() for s in settings]
require(len(expected_candidates) == 7, 'seven_planned_candidates')
rows = []
metrics = []
stops = set(cfg['generation_environment']['stop_token_ids'])
warper = {'top_k': 20, 'top_p': 0.95, 'min_p': 0.0}
for raw_path in bundle['result_paths']:
    path = Path(raw_path)
    row = read(path)
    entry = manifest['jobs'][path.stem]
    identity = row['identity']
    require(entry['status'] == 'complete', 'manifest_row_complete')
    require(entry['result_path'] == raw_path and path.parent == ROOT / 'artifacts/m3/tune/rank0', 'raw_path_binding')
    require(evidence[str(path.relative_to(ROOT))]['sha256'] == entry['result_sha256'], 'raw_checksum')
    require(fingerprint(identity) == path.stem and identity == entry['identity'], 'canonical_identity')
    require(identity['prompt_payload_hash'] == fingerprint(prompt), 'prompt_payload_binding')
    require(identity['checkpoint'] == cfg['checkpoint_hashes'], 'checkpoint_metadata_binding')
    require(identity['model_pair'] == cfg['model_pair'] and identity['model_revisions'] == cfg['model_revisions'], 'model_metadata_binding')
    require(identity['mode'] == 'tune' and identity['stage'] == 'm3' and identity['scale'] == 'reference', 'tune_stage_scale')
    require(identity['timing_session'] == SESSION and identity['repeat'] == 'dev:epoch=' + bundle['epoch'], 'session_epoch_binding')
    require(identity['condition'] == 'long_thinking' and identity['seed'] == 2026 and identity['prompt'] == PROMPT, 'prompt_condition_seed')
    require(identity['temperature'] == 0.6 and identity['sampling_name'] == 'thinking_recommended' and
            identity['sampling_warp_config'] == warper and identity['verification_rule'] == 'ratio', 'sampling_and_verification')
    require(identity['generation_cap'] == 24576 and identity['generation_caps'] == cfg['generation_caps'], 'generation_caps')
    require(identity['adapter_version'] == 0 and identity['frozen_settings_hash'] == '' and identity['phase_profile'] is False, 'tune_identity_fields')
    aliases = {'prompt_id': 'prompt', 'arm': 'proposal_mode', 'settings': 'setting', 'checkpoint_hashes': 'checkpoint'}
    for field in ('condition', 'seed', 'temperature', 'sampling_name', 'verification_rule', 'model_pair',
                  'model_revisions', 'stage', 'scale', 'sampling_warp_config', 'timing_session', 'repeat', 'phase_profile'):
        aliases[field] = field
    require(all(row[a] == identity[b] for a, b in aliases.items()), 'row_identity_consistency')
    require(row['split'] == 'dev' and row['label'] == 'reference', 'reference_dev_row')
    meta = row['input_metadata']
    require(meta['generation_environment_sha256'] == fingerprint(cfg['generation_environment']), 'generation_environment_binding')
    require(meta['enable_thinking'] is True and meta['requested_generation_cap'] == 24576 and
            meta['effective_generation_cap'] == min(24576, cfg['generation_environment']['context_limit'] - meta['input_token_count']), 'effective_cap')
    require(row['timing_isolated'] is False and row['phase_profile'] is False and row['phase_ms'] == {}, 'development_timing_scope')
    tokens, generation, blocks = row['output_tokens'], row['generation'], row['blocks']
    n = len(tokens)
    require(n == row['generated_tokens'] == generation['realized_length'] and 0 < n <= meta['effective_generation_cap'], 'token_count_cap')
    require(all(type(t) is int and 0 <= t < 151936 for t in tokens), 'token_range')
    require(generation['complete'] is True and generation['eos_reached'] == (tokens[-1] in stops) and
            not any(t in stops for t in tokens[:-1]), 'complete_first_eos')
    require(generation['capped'] == (n == meta['effective_generation_cap']) and
            (generation['eos_reached'] or generation['capped']), 'termination_at_eos_or_cap')
    for field in ('elapsed_seconds', 'generation_loop_elapsed_seconds', 'tokens_per_second', 'mean_accepted_length',
                  'median_accepted_length', 'target_forwards_per_generated_token'):
        require(math.isfinite(row[field]), 'finite_metrics')
    require(row['elapsed_seconds'] >= row['generation_loop_elapsed_seconds'] > 0, 'latency_nesting')
    require(close(row['tokens_per_second'], n / row['elapsed_seconds']), 'throughput_arithmetic')
    accepted = [b['accepted_length'] for b in blocks]
    require(all(type(a) is int and 0 <= a <= 15 for a in accepted), 'accepted_length_bounds')
    require(close(row['mean_accepted_length'], statistics.mean(accepted)) and
            close(row['median_accepted_length'], statistics.median(accepted)), 'acceptance_arithmetic')
    require(close(row['target_forwards_per_generated_token'], (len(blocks) + 1) / n), 'target_forward_arithmetic')
    require(tokens[1:] == [t for b in blocks for t in b['committed_tokens']], 'committed_stream_reconstruction')
    longest, witness = exact_periodic(tokens)
    require(longest == generation['longest_periodic_run_tokens'] and witness == generation['periodic_run'], 'all_periods_diagnostic_recomputation')
    if witness is not None:
        start, end, period = witness['start'], witness['end'], witness['period']
        require(end - start == longest and end - start >= 4 * period and
                witness['complete_cycles'] == longest // period and
                all(tokens[i] == tokens[i - period] for i in range(start + period, end)), 'every_periodic_witness_token')
    repeated, dominant = {}, {}
    for width in (4, 8, 16):
        counts = Counter(tuple(tokens[i:i + width]) for i in range(max(0, n - width + 1)))
        windows = max(0, n - width + 1)
        repeated[str(width)] = 1 - len(counts) / windows if windows else 0.0
        dominant[str(width)] = min(1.0, max(counts.values(), default=0) * width / max(n, 1))
    require(repeated == generation['repeated_ngram_fraction'] and dominant == generation['dominant_ngram_token_coverage'], 'ngram_diagnostics')
    flagged = n >= 256 and (longest >= max(256, math.ceil(n / 2)) or (repeated['8'] >= 0.8 and dominant['16'] >= 0.5))
    require(generation['repetition_detector_version'] == 2 and generation['repetition_dominated'] == flagged, 'repetition_version_and_flag')
    rows.append(row)
    metrics.append({'path': str(path.relative_to(ROOT)), 'sha256': entry['result_sha256'], 'identity_sha256': path.stem,
                    'arm': row['arm'], 'settings': row['settings'], 'generated_tokens': n, 'verification_blocks': len(blocks),
                    **{k: row[k] for k in ('elapsed_seconds', 'generation_loop_elapsed_seconds', 'tokens_per_second',
                                          'mean_accepted_length', 'median_accepted_length', 'target_forwards_per_generated_token')},
                    'eos_reached': generation['eos_reached'], 'capped': generation['capped'], 'repetition_dominated': flagged,
                    'longest_periodic_run_tokens': longest, 'periodic_fraction': longest / n, 'periodic_run': witness,
                    'output_tokens_sha256_le_int64': sha(np.asarray(tokens, dtype='<i8').tobytes())})

require(Counter((r['arm'], fingerprint(r['settings'])) for r in rows) ==
        Counter((a, fingerprint(s)) for a, s in expected_candidates), 'candidate_multiset')
rotation = plan['dev_ids'].index(PROMPT) % len(expected_candidates)
require([(r['arm'], r['settings']) for r in rows] == expected_candidates[rotation:] + expected_candidates[:rotation], 'rotated_candidate_order')
jobs = [(prompt, r['arm'], r['sampling_name'], r['seed'], r['settings']) for r in rows]
require(fingerprint({'jobs': jobs, 'repeat': 'dev', 'profile': False}) == BUNDLE, 'canonical_bundle_key')
require(all(r['input_metadata'] == rows[0]['input_metadata'] for r in rows), 'matched_input_metadata')
require(len({r['output_tokens'][0] for r in rows}) == 1, 'matched_initial_target_token')
for row in metrics:
    require(sha((ROOT / row['path']).read_bytes()) == row['sha256'], 'raw_files_unchanged_during_audit')
require('torch' not in sys.modules, 'no_torch_import')
elapsed_sum = sum(r['elapsed_seconds'] for r in rows)
report = {
    'status': 'passed', 'validation_kind': 'first_corrected_m3_complete_long_dev_bundle',
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'scale': args.scale, 'resolved_config': RESOLVED_CONFIG, 'seed': 2026, 'project_commit': PROJECT_COMMIT,
    'scope': 'One complete long-thinking development prompt with seven candidates; mandatory M3/M4 remain incomplete.',
    'source_path': str(source_path.relative_to(ROOT)), 'source_sha256': sha(source_path.read_bytes()),
    'source_preservation': 'Reconstructed and freshly rerun after the earlier successful unsaved in-memory audit.',
    'logical_session': SESSION, 'implementation': IMPLEMENTATION, 'prompt_id': PROMPT,
    'bundle_key': BUNDLE, 'bundle': bundle, 'checkpoint_hashes_from_session': cfg['checkpoint_hashes'],
    'model_pair_from_session': cfg['model_pair'], 'model_revisions_from_session': cfg['model_revisions'],
    'input_metadata': rows[0]['input_metadata'], 'checks': dict(checks), 'check_count': sum(checks.values()),
    'candidate_metrics': metrics, 'evidence': evidence,
    'totals': {'candidate_rows': 7, 'generated_tokens': sum(r['generated_tokens'] for r in rows),
               'verification_blocks': sum(len(r['blocks']) for r in rows), 'request_elapsed_seconds': elapsed_sum,
               'generation_loop_elapsed_seconds': sum(r['generation_loop_elapsed_seconds'] for r in rows),
               'eos_rows': sum(r['generation']['eos_reached'] for r in rows),
               'capped_rows': sum(r['generation']['capped'] for r in rows),
               'repetition_dominated_rows': sum(r['generation']['repetition_dominated'] for r in rows)},
    'remaining_long_dev_equal_cost_scenario': {
        'bundles': 11, 'seconds_per_bundle': elapsed_sum, 'seconds': 11 * elapsed_sum, 'hours': 11 * elapsed_sum / 3600,
        'interpretation': 'One-prompt arithmetic only, not a forecast or confidence interval. Ignores work already begun and reporting/serialization overhead. Includes the excluded pathological candidate generation cost; applies only to the remaining eleven M3 long-dev bundles.'},
    'limitations': [
        'Only one long-dev prompt; no final setting, statistical quality, generalization, or full-study speed conclusion follows.',
        'The seven candidates have different realized outputs and lengths from one seed/repeat. Their request-time sum excludes between-request/reporting overhead and is workload evidence, not elapsed job duration; pooled tokens/seconds is not presented as a performance metric.',
        'Development timings are not isolated throughput/profile evidence; mandatory fresh M3/M4 timing remains required.',
        'Mutable manifest and session snapshots are preserved by hash; this audit includes only the seven named current-session rows, never historical pre-correction rows.',
        'Checkpoint and model hashes/revisions are metadata bindings here; this bounded audit does not rehash checkpoint bytes, inspect live tensors, retokenize, replay inference, or establish causal loading.',
        'Acceptance summaries and committed-stream algebra are checked; full probability vectors, stochastic uniforms, residual sampling, and tensor/cache exactness are not independently reconstructed.',
        'The exhaustive NumPy detector is an independent artifact implementation also used in the prior detector experiment; it shares the equality-run algorithmic principle with production and is not a separate scalar all-period oracle on these long rows.',
        'Periodic/n-gram flagging is a mechanical pathology diagnostic, not semantic grading. Unflagged outputs are not thereby proven correct.',
        'Repetition-excluded output still consumes real generation time. One-prompt length and arm variation make extrapolation uncertain.'
    ],
    'execution': {'python': sys.version, 'numpy': np.__version__, 'torch_imported': False,
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'audit_elapsed_seconds': time.monotonic() - started}
}
publish_bytes(OUTPUT, (json.dumps(report, indent=2) + '\n').encode())
print(json.dumps({'path': str(OUTPUT), 'sha256': sha(OUTPUT.read_bytes()), 'source_sha256': report['source_sha256'],
                  'status': report['status'], 'checks': report['check_count'], 'totals': report['totals'],
                  'audit_elapsed_seconds': report['execution']['audit_elapsed_seconds']}))
