"""Scalar-oracle regression coverage for batched diagnostic accumulation."""
import bisect
from collections import defaultdict
import copy
import json

import pytest
import torch

from edraft.diagnose import (DiagnosticAccumulator, OUTPUT_EDGES, OUTPUT_LABELS,
                            _context_label, live_block_diagnostics, generation_diagnostics)


class ScalarDiagnosticAccumulator(DiagnosticAccumulator):
    """Original per-slot implementation, retained only as a test oracle."""
    def add_block(self, block, repetition_dominated=False):
        if 'diagnostics' not in block:
            raise ValueError('exact live diagnostics are missing; compact union alone cannot recover full ranks or overlap')
        diag, h = block['diagnostics'], block['h'].double().cpu()
        K = len(block['draft_tokens'])
        first = int(block['first_rejection_index'])
        condition = block['condition']
        include_primary = not (condition.startswith('long_') and repetition_dominated)
        arm = (condition, float(block['collection_temperature']), json.dumps(block['sampling_warp_config'], sort_keys=True), block['split'])
        for i in range(K):
            offset = block['absolute_output_offset'] + i
            context = block['input_context_length'] + i
            policy = 'off_policy_teacher' if i > first else 'on_policy'
            for axis, bin_name in [('all', 'all'), ('output_offset', OUTPUT_LABELS[bisect.bisect_right(OUTPUT_EDGES, offset)]),
                                   ('input_context_length', _context_label(context, self.context_edges))]:
                key = (*arm, axis, bin_name, policy, 'primary' if include_primary else 'repetition_excluded')
                group = self.groups.setdefault(key, {'rows': defaultdict(lambda: defaultdict(lambda: [0., 0])), 'n': 0, 'h_sum': torch.zeros(h.shape[-1], dtype=torch.float64),
                    'h_sq_sum': torch.zeros(h.shape[-1], dtype=torch.float64), 'h_norms': [], 'outlier_slots': 0, 'outlier_coordinates': 0,
                    'accepted_length': [], 'first_rejection': [], 'acceptance_by_block_position': defaultdict(list)})
                group['n'] += 1
                group['h_sum'] += h[i]
                group['h_sq_sum'] += h[i].square()
                group['h_norms'].append(float(h[i].norm()))
                outliers = h[i].abs() > self.outlier_threshold
                group['outlier_slots'] += bool(outliers.any())
                group['outlier_coordinates'] += int(outliers.sum())
                for metric in ('target_rank', 'sampled_target_rank', 'sampled_target_rank_unwarped', 'overlap_sampling',
                               'overlap_unwarped', 'cross_entropy_unwarped', 'sampling_cross_entropy',
                               'sampling_cross_entropy_infinite', 'target_mass_on_zero_proposal'):
                    value = diag[metric][i]
                    if value is not None:
                        accumulator = group['rows'][metric][(block['prompt_id'], block['seed'])]
                        accumulator[0] += float(value)
                        accumulator[1] += 1
                for m, stats in diag['top_m'].items():
                    for metric, values in stats.items():
                        accumulator = group['rows'][f'{metric}_at_{m}'][(block['prompt_id'], block['seed'])]
                        accumulator[0] += float(values[i])
                        accumulator[1] += 1
                group['acceptance_by_block_position'][str(i)].append(float(i < first))
                # Record a round once in every bin it actually starts in.
                if i == 0:
                    group['accepted_length'].append(float(block['accepted_length']))
                if i == first and diag['first_rejection'] is not None:
                    group['first_rejection'].append(diag['first_rejection'])


def mixed_blocks(dtype):
    rng = torch.Generator().manual_seed(482)
    arms = [('standard_nonthinking', 0., False),
            ('standard_nonthinking', .7, False),
            ('standard_nonthinking', 1., True),
            ('long_thinking', .6, False),
            ('long_thinking', .6, True),
            ('long_nonthinking', .7, True)]
    for arm_index, (condition, temperature, repetition) in enumerate(arms):
        for prompt_index in range(2):
            for seed_index in range(2):
                for round_index in range(2):
                    index = arm_index + prompt_index + seed_index + round_index
                    K = (1, 3, 15)[index % 3]
                    first = (0, K // 2, K - 1, K)[(index + seed_index) % 4]
                    h = torch.randn(K, 512, generator=rng, dtype=dtype)[:, ::2]
                    h[0, 0] = 5.  # Outliers are strictly above the threshold.
                    h[-1, -1] = 6.
                    draft = torch.randn(K, 19, generator=rng)
                    target = torch.randn(K, 19, generator=rng)
                    warp = {'top_k': 1 if temperature in (.6, .7) else 0,
                            'top_p': .8 if temperature == .7 else 1., 'min_p': 0.}
                    block = {'draft_logits': draft, 'l_unit': draft.log_softmax(-1),
                             'target_logits': target, 'collection_temperature': temperature,
                             'sampling_warp_config': warp, 'first_rejection_index': first,
                             'replacement_token': None if first == K else 2,
                             'accepted_length': first, 'draft_tokens': [0] * K, 'h': h,
                             'prompt_id': f'prompt-{prompt_index}', 'seed': seed_index,
                             'condition': condition, 'split': 'test',
                             'absolute_output_offset': (1023, 4095, 8191, 16383)[index % 4],
                             'input_context_length': (1023, 4095, 8191, 16383, 32767)[index % 5]}
                    block['diagnostics'] = live_block_diagnostics(block, seed=17 + index)
                    yield block, repetition


def assert_reports_equal(actual, expected):
    """Full report comparison: only float64 feature reductions may round differently."""
    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    for new, old in zip(actual['groups'], expected['groups'], strict=True):
        new_features, old_features = new.pop('features'), old.pop('features')
        assert new_features.keys() == old_features.keys()
        for key in new_features:
            value, reference = new_features[key], old_features[key]
            if isinstance(reference, (float, dict)):
                assert value == pytest.approx(reference, abs=2e-13, rel=2e-13)
            else:
                assert value == reference
    # This includes all prompt bootstrap means, CIs, seed variability, counts,
    # bin boundaries, accepted lengths, rejection/recoverability, and exclusions.
    assert actual == expected


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_batched_accumulator_complete_report_matches_original_scalar(dtype):
    scalar, batched = ScalarDiagnosticAccumulator(), DiagnosticAccumulator()
    for block, repetition in mixed_blocks(dtype):
        generation = {'prompt_id': block['prompt_id'], 'seed': block['seed'],
                      'condition': block['condition'], 'temperature': block['collection_temperature'],
                      'split': block['split'], 'sampling_warp_config': block['sampling_warp_config'],
                      'realized_length': block['absolute_output_offset'] + len(block['draft_tokens']),
                      'repetition_dominated': repetition, 'eos_reached': True, 'capped': False}
        for accumulator in (scalar, batched):
            accumulator.add_generation(generation)
            accumulator.add_block(block, repetition_dominated=repetition)
    assert scalar.groups.keys() == batched.groups.keys()
    for key in scalar.groups:
        old, new = scalar.groups[key], batched.groups[key]
        assert old['rows'] == new['rows']
        for field in ('n', 'outlier_slots', 'outlier_coordinates', 'accepted_length',
                      'first_rejection', 'acceptance_by_block_position'):
            assert old[field] == new[field]
        for field in ('h_sum', 'h_sq_sum'):
            torch.testing.assert_close(old[field], new[field], atol=2e-13, rtol=2e-13)
    assert_reports_equal(batched.report(), scalar.report())


def test_missing_sampling_ce_keeps_metric_absent_until_finite_observation():
    block, _ = next(mixed_blocks(torch.float32))
    block['diagnostics']['sampling_cross_entropy'] = [None] * len(block['draft_tokens'])
    scalar, batched = ScalarDiagnosticAccumulator(), DiagnosticAccumulator()
    for accumulator in (scalar, batched):
        accumulator.add_block(block)
        assert all('sampling_cross_entropy' not in g['rows'] for g in accumulator.groups.values())
    block['diagnostics']['sampling_cross_entropy'][0] = 2.25
    for accumulator in (scalar, batched):
        accumulator.add_block(block)
    assert_reports_equal(batched.report(), scalar.report())


@pytest.fixture
def diagnostic_shards(tmp_path):
    from edraft.traces import compact_block, write_shard
    grouped = defaultdict(list)
    for block, repetition in mixed_blocks(torch.float64):
        key = (block['condition'], block['collection_temperature'], block['prompt_id'],
               block['seed'], repetition)
        K = len(block['draft_tokens'])
        block.update({'loss_temperature': 1., 'h_raw_bf16': torch.zeros(K, 8, dtype=torch.bfloat16),
                      'draft_lm_head': {'artifact_id': 'synthetic-head', 'revision': 'fixed',
                                       'dtype': 'torch.bfloat16', 'bias': False, 'tied_weights': False,
                                       'hook': 'synthetic', 'shape': [19, 8]},
                      'verified_prefix_tail': [1, 2, 3, 4], 'adapter_version': 0,
                      'verification_rule': 'greedy' if block['collection_temperature'] == 0 else 'ratio',
                      'physical_to_logical': {'anchor_physical_index': 0, 'logical_to_physical': list(range(1, K + 1))}})
        metadata = {name: block[name] for name in ('prompt_id', 'condition', 'seed', 'split')}
        metadata.update(scale='smoke', model_pair=['toy-target', 'toy-draft'],
                        projection_metadata={'frozen': True, 'calibration_split': 'train'})
        grouped[key].append(compact_block(block, metadata))
    sources = []
    for index, (key, blocks) in enumerate(grouped.items()):
        condition, temperature, prompt, seed, repetition = key
        generation = {'prompt_id': prompt, 'condition': condition, 'temperature': temperature,
                      'seed': seed, 'split': 'test', 'sampling_warp_config': blocks[0]['sampling_warp_config'],
                      'complete': True, 'realized_length': 17000 + index,
                      'repetition_dominated': repetition, 'eos_reached': True, 'capped': False}
        path = tmp_path / f'shard-{index}.pt'
        summary = write_shard(path, blocks, {'generation': generation})
        sources.append({'path': str(path), 'sha256': summary['sha256']})
    return sources


def test_parallel_shard_reports_preserve_all_serial_statistics(diagnostic_shards):
    from edraft.diagnose import accumulate_trace_shards
    serial = accumulate_trace_shards(diagnostic_shards, 'smoke', workers=1)
    parallel = accumulate_trace_shards(diagnostic_shards, 'smoke', workers=4)
    assert parallel.generations == serial.generations
    for key in serial.groups:
        for field in ('rows', 'h_norms', 'first_rejection', 'acceptance_by_block_position'):
            assert parallel.groups[key][field] == serial.groups[key][field]
    assert_reports_equal(parallel.report(), serial.report())


def test_parallel_overlap_replays_before_any_partial_merge(diagnostic_shards, monkeypatch):
    import edraft.diagnose as diagnosis
    sources = diagnostic_shards[:1] * 2
    serial = diagnosis.accumulate_trace_shards(sources, 'smoke', workers=1)
    original = diagnosis._accumulate_source
    replays = []
    def replay(accumulator, source, scale):
        replays.append(source)
        return original(accumulator, source, scale)
    monkeypatch.setattr(diagnosis, '_accumulate_source', replay)
    parallel = diagnosis.accumulate_trace_shards(sources, 'smoke', workers=2)
    assert replays == sources[:1]
    assert_reports_equal(parallel.report(), serial.report())


@pytest.mark.parametrize('failure', ['checksum', 'scale', 'diagnostics'])
def test_parallel_workers_propagate_validation_errors(diagnostic_shards, failure):
    from edraft.diagnose import accumulate_trace_shards
    from edraft.traces import read_shard, write_shard
    source = dict(diagnostic_shards[0])
    if failure == 'checksum':
        payload = read_shard(source['path'])
        payload['metadata']['generation']['realized_length'] += 1
        write_shard(source['path'], payload['blocks'], payload['metadata'])
    elif failure == 'diagnostics':
        payload = read_shard(source['path'])
        del payload['blocks'][0]['diagnostics']
        updated = write_shard(source['path'], payload['blocks'], payload['metadata'])
        source['sha256'] = updated['sha256']
    with pytest.raises(ValueError if failure == 'diagnostics' else RuntimeError,
                       match={'checksum': 'checksum changed', 'scale': 'scale does not match',
                                          'diagnostics': 'diagnostics are missing'}[failure]):
        accumulate_trace_shards([source], 'reference' if failure == 'scale' else 'smoke', workers=2)


def test_parallel_window_bounds_completed_unmerged_results(monkeypatch):
    import edraft.diagnose as diagnosis
    submitted = merged = peak = 0
    merge_order = []
    class ReadyFuture:
        def __init__(self, source):
            self.source = source
        def result(self):
            return {'source': self.source, 'groups': {}, 'generations': []}
    class ImmediatePool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, source, *args):
            nonlocal submitted, peak
            submitted += 1
            peak = max(peak, submitted - merged)
            return ReadyFuture(source)
    original = diagnosis._merge_shard_result
    def merge(accumulator, result, source, scale):
        nonlocal merged
        merge_order.append(source['path'])
        original(accumulator, result, source, scale)
        merged += 1
    monkeypatch.setattr(diagnosis, 'ProcessPoolExecutor', ImmediatePool)
    monkeypatch.setattr(diagnosis, '_merge_shard_result', merge)
    sources = [{'path': str(i), 'sha256': str(i)} for i in range(20)]
    diagnosis.accumulate_trace_shards(sources, 'smoke', workers=3)
    assert peak == 3
    assert merge_order == [source['path'] for source in sources]


def test_parallel_empty_sources_and_invalid_worker_count():
    from edraft.diagnose import accumulate_trace_shards
    assert accumulate_trace_shards([], 'smoke', workers=3).report() == DiagnosticAccumulator().report()
    with pytest.raises(ValueError, match='workers must be positive'):
        accumulate_trace_shards([], 'smoke', workers=0)
