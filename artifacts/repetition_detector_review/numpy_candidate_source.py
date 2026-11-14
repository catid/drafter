"""Bounded CPU-only repetition-detector experiment; never edits production data.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' .venv/bin/python <this file>.

Evidence is written only to sibling numpy_candidate_benchmark.json and
numpy_candidate_m1_scan.json. Production diagnostics and source sidecars are
read-only. Exact arbitrary-period coverage follows from P <= N // 4 whenever
at least four complete cycles occur within an N-token output.
"""
from collections import Counter
from pathlib import Path
import datetime
import hashlib
import itertools
import json
import os
import platform
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
M3_PATH = ROOT / 'm3/tune/rank0/f3cd5559bf914c6d809243262867ede41a53dd778d5754c9bcea4644319f9677.json'
M1_PATH = ROOT / 'traces/rank1/390b0d2e74ab69ebdfd300c9e3642d5024dcd394e8f1fe89167b45e06f387240.pt.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), digest(raw)


def exact_numpy(tokens, *, count_filter=True):
    """Exact maximum periodic span with >= 4 cycles; O(N^2) time/O(N) memory.

    For each P, a true run of length R in a[P:] == a[:-P] represents a
    contiguous periodic span R + P beginning at the equality-run's start.
    R >= 3P is precisely the requirement of >= 4 complete cycles. The total
    number of true equalities must be at least 3P, hence the optional filter
    cannot discard a qualifying periodic span. The unfiltered variant
    reproduces the independent first-pass M1 scan algorithm.
    """
    a = np.asarray(tokens, dtype='<i8')
    length = len(a)
    best = {'length': 0, 'period': None, 'start': None, 'end_exclusive': None,
            'complete_cycles': 0}
    old_longest = 0
    qualifying = []
    for period in range(1, length // 4 + 1):
        equal = a[period:] == a[:-period]
        if count_filter and np.count_nonzero(equal) < 3 * period:
            continue
        false_positions = np.flatnonzero(~equal)
        bounds = np.empty(len(false_positions) + 2, dtype=np.int64)
        bounds[0], bounds[-1] = -1, len(equal)
        bounds[1:-1] = false_positions
        runs = np.diff(bounds) - 1
        index = int(runs.argmax())
        run = int(runs[index])
        if run < 3 * period:
            continue
        span = run + period
        start = int(bounds[index]) + 1
        witness = {'length': span, 'period': period, 'start': start,
                   'end_exclusive': start + span, 'complete_cycles': span // period}
        if period <= 64:
            old_longest = max(old_longest, span)
        if span > best['length']:
            best = witness
        if span >= max(256, (length + 1) // 2):
            qualifying.append(witness)
    return {'best': best, 'old_limit_longest': old_longest,
            'qualifying_periods': qualifying}


def scalar_oracle(tokens, max_period=None):
    """Independent straightforward scalar scan, also reproduces old detector."""
    length = len(tokens)
    longest = 0
    upper = length // 4 if max_period is None else min(max_period, length // 2)
    for period in range(1, upper + 1):
        run = 0
        for index in range(period, length):
            run = run + 1 if tokens[index] == tokens[index - period] else 0
            if run >= 3 * period:
                longest = max(longest, run + period)
    return longest


def periodic_dominated(length, longest):
    return length >= 256 and longest >= max(256, (length + 1) // 2)


def check_witness(tokens, witness):
    if not witness['length']:
        return
    start, end, period = witness['start'], witness['end_exclusive'], witness['period']
    assert end - start == witness['length']
    assert end - start >= 4 * period
    assert all(tokens[i] == tokens[i - period] for i in range(start + period, end))


def correctness():
    cases = [list(bits) for n in range(11) for bits in itertools.product(range(2), repeat=n)]
    rng = np.random.default_rng(2026)
    cases += [rng.integers(0, 8, size=int(n)).tolist() for n in rng.integers(0, 513, size=200)]
    for period in (1, 2, 3, 17, 63, 64, 65, 96, 127, 257):
        unit = list(range(period))
        cases.extend([unit * 3, unit * 4, unit * 5 + unit[:period // 2],
                      [-2, -3] + unit * 4 + [-4, -5],
                      unit * 2 + [-9] + unit * 2])
    cases.extend([list(range(65)) * 4, list(range(80)) * 6])
    for tokens in cases:
        expected = scalar_oracle(tokens)
        filtered = exact_numpy(tokens)
        unfiltered = exact_numpy(tokens, count_filter=False)
        assert filtered == unfiltered
        assert filtered['best']['length'] == expected
        assert filtered['old_limit_longest'] == scalar_oracle(tokens, 64)
        check_witness(tokens, filtered['best'])
    return {'status': 'passed', 'case_count': len(cases),
            'coverage': 'all binary sequences of length 0..10; 200 seeded random cases; structured periods 1..257 around 3/4/5-cycle and interruption boundaries',
            'checks': ['candidate equals unfiltered scan', 'longest span equals scalar arbitrary-period oracle',
                       'old-limit value equals original period<=64 loop', 'returned witness has exact equality and >=4 cycles']}


def benchmark():
    real_m3, m3_hash = read_json(M3_PATH)
    real_m1, m1_hash = read_json(M1_PATH)
    rng = np.random.default_rng(2026)
    cases = {
        'real_m3_96_token_loop': real_m3['output_tokens'],
        'real_m1_1508_token_loop': real_m1['generation']['output_tokens'],
        'random_24k': rng.integers(0, 151936, size=24576).tolist(),
        'all_same_24k': [7] * 24576,
        'period96_24k': list(range(96)) * 256,
        'period1508_24k': (list(range(1508)) * 17)[:24576],
        'period6144_four_cycles_24k': list(range(6144)) * 4,
    }
    methods = {
        'old_period64_scalar': lambda t: scalar_oracle(t, 64),
        'numpy_unfiltered_all_periods': lambda t: exact_numpy(t, count_filter=False),
        'numpy_count_filtered_all_periods': exact_numpy,
    }
    rows = []
    for name, tokens in cases.items():
        candidate = exact_numpy(tokens)
        check_witness(tokens, candidate['best'])
        if name == 'real_m3_96_token_loop':
            assert candidate['best'] == {'length': 15477, 'period': 96, 'start': 9099,
                                         'end_exclusive': 24576, 'complete_cycles': 161}
        if name == 'real_m1_1508_token_loop':
            assert candidate['best'] == {'length': 19806, 'period': 1508, 'start': 4770,
                                         'end_exclusive': 24576, 'complete_cycles': 13}
        timings = {}
        for label, method in methods.items():
            method(tokens)
            samples = []
            for _ in range(5):
                begun = time.perf_counter()
                result = method(tokens)
                samples.append(time.perf_counter() - begun)
            if label != 'old_period64_scalar':
                assert result == candidate
            else:
                assert result == candidate['old_limit_longest']
            timings[label] = {'samples_seconds': samples, 'median_seconds': statistics.median(samples),
                              'maximum_seconds': max(samples)}
        rows.append({'case': name, 'length': len(tokens), 'candidate': candidate,
                     'old_periodic_dominated': periodic_dominated(len(tokens), candidate['old_limit_longest']),
                     'new_periodic_dominated': periodic_dominated(len(tokens), candidate['best']['length']),
                     'timings': timings})
    return {'correctness': correctness(), 'warmups_per_case_method': 1, 'repeats': 5,
            'cases': rows, 'real_sources': [
                {'path': str(M3_PATH), 'sha256': m3_hash, 'generation_elapsed_seconds': real_m3['elapsed_seconds']},
                {'path': str(M1_PATH), 'sha256': m1_hash}]}


def scan_m1():
    audit_path = ROOT / 'm1_audit.json'
    audit, audit_hash = read_json(audit_path)
    rows = []
    begun = time.perf_counter()
    for source in audit['source_shards']:
        path = Path(source['path']).with_suffix('.pt.json')
        side, side_hash = read_json(path)
        if not side['identity']['condition'].startswith('long_'):
            continue
        generation = side['generation']
        tokens = generation['output_tokens']
        result = exact_numpy(tokens)
        assert result == exact_numpy(tokens, count_filter=False)
        assert result['old_limit_longest'] == generation['longest_periodic_run_tokens']
        ngram_clause = (generation['repeated_ngram_fraction']['8'] >= .8
                        and generation['dominant_ngram_token_coverage']['16'] >= .5)
        old = len(tokens) >= 256 and (periodic_dominated(len(tokens), result['old_limit_longest']) or ngram_clause)
        new = len(tokens) >= 256 and (periodic_dominated(len(tokens), result['best']['length']) or ngram_clause)
        assert old == generation['repetition_dominated']
        check_witness(tokens, result['best'])
        rows.append({'prompt_id': generation['prompt_id'], 'split': generation['split'],
                     'condition': generation['condition'], 'length': len(tokens),
                     'old_repetition_dominated': old, 'new_repetition_dominated': new,
                     'changed': old != new, 'periodic_scan': result,
                     'sidecar_path': str(path), 'sidecar_sha256': side_hash,
                     'trace_path': source['path'], 'trace_sha256_from_audit': source['sha256'],
                     'output_tokens_sha256_little_endian_int64': digest(np.asarray(tokens, dtype='<i8').tobytes())})
    assert len(rows) == 44
    return {'scope': 'all authorized M1 long output JSON sidecars; tensor traces not loaded',
            'audit_path': str(audit_path), 'audit_sha256': audit_hash,
            'outputs': len(rows), 'tokens': sum(row['length'] for row in rows),
            'split_counts': dict(Counter(row['split'] for row in rows)),
            'changed_count': sum(row['changed'] for row in rows),
            'old_limit_reproduction': 'all 44 old longest-span values and classifications exactly reproduced',
            'filtered_vs_original_scan': 'exact equality for all 44',
            'rows': rows, 'elapsed_seconds': time.perf_counter() - begun}


def main():
    source_sha = digest(Path(__file__).read_bytes())
    metadata = {'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_path': str(Path(__file__).resolve()), 'source_sha256': source_sha,
                'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                'environment': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                'torch_imported': 'torch' in sys.modules,
                'threshold': 'length>=256; periodic span>=max(256,ceil(length/2)); at least four complete cycles',
                'period_coverage': 'all integers 1..floor(length/4), exhaustive and exact',
                'production_files_modified': False}
    assert os.environ.get('OMP_NUM_THREADS') == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for name, report in [('benchmark', benchmark()), ('m1_scan', scan_m1())]:
        path = OUT / f'numpy_candidate_{name}.json'
        path.write_text(json.dumps({**metadata, **report}, indent=2) + '\n')
        print(json.dumps({'artifact': str(path), 'sha256': digest(path.read_bytes())}), flush=True)


if __name__ == '__main__':
    main()
