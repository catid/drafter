"""Exact-cycle regression tests with an independent scalar oracle."""
from copy import deepcopy
import hashlib
import json
import random

import pytest

from edraft import diagnose
from test_traces import live_record


def brute_periodic_run(tokens):
    """Enumerate each possible start and cycle, then compare against its motif.

    This intentionally avoids shifted-array comparisons, run-boundary extraction,
    and the production count filter. Only small sequences use this oracle.
    """
    best = None
    for start in range(len(tokens)):
        for period in range(1, (len(tokens) - start) // 4 + 1):
            end = start + period
            while end < len(tokens) and tokens[end] == tokens[start + (end - start) % period]:
                end += 1
            span = end - start
            if span < 4 * period:
                continue
            candidate = (-span, period, start, end)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return 0, {'start': None, 'end': None, 'period': None, 'complete_cycles': 0}
    negative_span, period, start, end = best
    return -negative_span, {'start': start, 'end': end, 'period': period,
                           'complete_cycles': (end - start) // period}


def test_exact_longest_run_matches_independent_brute_force():
    rng = random.Random(77419)
    cases = [[], [8], [4] * 3, [4] * 4, [2] * 80,
             [1] * 8 + [9] + [2] * 8, [1, 2] * 4 + [9] + [3] * 8,
             list(range(80))]
    for length in range(0, 81, 4):
        for alphabet in (2, 5, 100):
            cases.append([rng.randrange(alphabet) for _ in range(length)])
    for period in range(1, 13):
        motif = list(range(100, 100 + period))
        prefix = list(range(1000, 1000 + rng.randrange(8)))
        tail = motif[:rng.randrange(period)]
        cases.append(prefix + motif * 4 + tail + [9999])
    for tokens in cases:
        expected_length, expected_run = brute_periodic_run(tokens)
        actual = diagnose.generation_diagnostics(tokens)
        assert actual['repetition_detector_version'] == 2
        assert actual['longest_periodic_run_tokens'] == expected_length
        assert actual['periodic_run'] == expected_run
        assert actual['realized_length'] == len(tokens)


@pytest.mark.parametrize('period,prefix_length,cycles,tail', [
    (96, 17, 5, 21),
    (1508, 311, 4, 37),
])
def test_long_primitive_cycles_are_detected_without_a_period_cutoff(period, prefix_length, cycles, tail):
    motif = list(range(period))
    prefix = list(range(10000, 10000 + prefix_length))
    tokens = prefix + motif * cycles + motif[:tail]
    result = diagnose.generation_diagnostics(tokens)
    assert result['repetition_dominated'] is True
    assert result['longest_periodic_run_tokens'] == period * cycles + tail
    assert result['periodic_run'] == {'start': prefix_length, 'end': len(tokens),
                                     'period': period, 'complete_cycles': cycles}
    assert result['dominant_ngram_token_coverage']['16'] < .5
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('cycles,tail,expected', [(3, 0, False), (3, 95, False), (4, 0, True)])
def test_at_least_four_complete_cycles_are_required(cycles, tail, expected):
    motif = list(range(96))
    result = diagnose.generation_diagnostics(motif * cycles + motif[:tail])
    assert result['repetition_dominated'] is expected
    if not expected:
        assert result['longest_periodic_run_tokens'] == 0


@pytest.mark.parametrize('prefix_length,expected', [(383, True), (384, True), (385, False)])
def test_exact_half_output_dominance_boundary(prefix_length, expected):
    tokens = list(range(1000, 1000 + prefix_length)) + list(range(96)) * 4
    result = diagnose.generation_diagnostics(tokens)
    assert result['longest_periodic_run_tokens'] == 384
    assert result['repetition_dominated'] is expected


@pytest.mark.parametrize('tokens,expected_span,expected_flag', [
    (list(range(63)) * 4 + [1000, 1001, 1002], 252, False),
    (list(range(63)) * 4 + [1000, 1001, 1002, 1003], 252, False),
    (list(range(64)) * 4, 256, True),
])
def test_minimum_output_and_periodic_span_boundaries(tokens, expected_span, expected_flag):
    result = diagnose.generation_diagnostics(tokens)
    assert result['longest_periodic_run_tokens'] == expected_span
    assert result['repetition_dominated'] is expected_flag


def test_interrupted_cycles_and_changing_section_bodies_are_not_dominant_loops():
    motif = list(range(96))
    interrupted = motif * 3 + [9999] + motif * 3
    sections = []
    for section in range(20):
        sections.extend(list(range(12)))  # Repeated heading / boilerplate.
        sections.extend(range(1000 + 80 * section, 1080 + 80 * section))
    for tokens in (interrupted, sections):
        result = diagnose.generation_diagnostics(tokens)
        assert result['repetition_dominated'] is False
        assert result['longest_periodic_run_tokens'] < max(256, len(tokens) / 2)


def test_existing_ngram_fallback_remains_effective_without_a_dominant_exact_cycle():
    tokens = []
    for section in range(12):
        tokens.extend([0] * 128 + [1000 + section])
    result = diagnose.generation_diagnostics(tokens)
    assert result['longest_periodic_run_tokens'] == 128
    assert result['repeated_ngram_fraction']['8'] >= .8
    assert result['dominant_ngram_token_coverage']['16'] >= .5
    assert result['repetition_dominated'] is True


def stale_generation(tokens, **extra):
    return {'output_tokens': tokens, 'realized_length': len(tokens),
            'repetition_rule': 'legacy period<=64 rule', 'repetition_dominated': False,
            'longest_periodic_run_tokens': 0, 'complete': True, 'eos_reached': False,
            'capped': True, 'max_new_tokens': len(tokens), 'prompt_id': 'synthetic-long-test',
            'condition': 'long_thinking', 'split': 'test', 'temperature': .7,
            'identity': {'seed': 50, 'checkpoint': 'frozen-toy'}, **extra}


@pytest.mark.parametrize('stored_version', [None, 1])
def test_recomputation_returns_copy_preserves_original_provenance_and_is_idempotent(stored_version):
    generation = stale_generation(list(range(96)) * 4)
    if stored_version is not None:
        generation['repetition_detector_version'] = stored_version
    original = deepcopy(generation)
    refreshed = diagnose.recompute_generation_diagnostics(generation)
    assert generation == original
    assert refreshed is not generation
    assert refreshed['repetition_dominated'] is True
    assert refreshed['repetition_detector_version'] == 2
    for key in ('output_tokens', 'complete', 'eos_reached', 'capped', 'identity',
                'max_new_tokens', 'prompt_id', 'condition', 'split', 'temperature'):
        assert refreshed[key] == original[key]
    assert refreshed['diagnostic_recomputation'] == {
        'basis': 'stored_output_tokens',
        'stored_repetition_detector_version': stored_version if stored_version is not None else 'legacy_period_le_64',
        'stored_repetition_rule': original['repetition_rule'],
        'stored_repetition_dominated': False, 'classification_changed': True,
    }
    assert diagnose.recompute_generation_diagnostics(refreshed) == refreshed


def test_recomputation_can_correct_a_stale_positive_and_handles_absent_old_fields():
    generation = stale_generation(list(range(600)), repetition_dominated=True)
    refreshed = diagnose.recompute_generation_diagnostics(generation)
    assert refreshed['repetition_dominated'] is False
    assert refreshed['diagnostic_recomputation']['stored_repetition_dominated'] is True
    assert refreshed['diagnostic_recomputation']['classification_changed'] is True
    minimal = diagnose.recompute_generation_diagnostics({'output_tokens': [1, 2, 3]})
    assert minimal['realized_length'] == 3
    assert minimal['diagnostic_recomputation']['stored_repetition_rule'] is None
    assert minimal['diagnostic_recomputation']['stored_repetition_dominated'] is None


@pytest.mark.parametrize('generation', [
    {'realized_length': 300, 'repetition_dominated': False},
    {'output_tokens': [1, 2, 3], 'realized_length': 4},
])
def test_recomputation_rejects_missing_tokens_or_inconsistent_lengths(generation):
    original = deepcopy(generation)
    with pytest.raises(ValueError):
        diagnose.recompute_generation_diagnostics(generation)
    assert generation == original


def test_diagnostic_ingestion_reclassifies_stored_false_without_rewriting_trace(tmp_path, live_record):
    from edraft.traces import compact_block, write_shard

    record, metadata, _ = live_record
    metadata = dict(metadata, condition='long_thinking', scale='smoke')
    record = dict(record, diagnostics=diagnose.live_block_diagnostics(record, seed=11))
    block = compact_block(record, metadata)
    rng = random.Random(284)
    motif = [rng.randrange(16) for _ in range(96)]
    generation = stale_generation(motif * 4, prompt_id=metadata['prompt_id'],
                                  seed=metadata['seed'], sampling_warp_config=record['sampling_warp_config'])
    path = tmp_path / 'synthetic-long.pt'
    sidecar = write_shard(path, [block], {'generation': generation})
    original_bytes = path.read_bytes()
    original_sidecar = path.with_suffix('.pt.json').read_bytes()
    accumulator = diagnose.DiagnosticAccumulator()
    diagnose._accumulate_source(accumulator, {'path': str(path), 'sha256': sidecar['sha256']}, 'smoke')
    report = accumulator.report()
    assert accumulator.generations[0]['repetition_dominated'] is True
    assert accumulator.generations[0]['diagnostic_recomputation']['classification_changed'] is True
    assert report['groups'] and {group['inclusion'] for group in report['groups']} == {'repetition_excluded'}
    assert report['generation_lengths'][0]['primary_generations'] == 0
    assert report['generation_lengths'][0]['repetition_excluded'] == 1
    assert path.read_bytes() == original_bytes
    assert path.with_suffix('.pt.json').read_bytes() == original_sidecar
    assert hashlib.sha256(original_bytes).hexdigest() == sidecar['sha256']
