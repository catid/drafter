import json
import pytest
import torch
from edraft.diagnose import (live_block_diagnostics, generation_diagnostics,
    bootstrap_prompt_means, DiagnosticAccumulator, token_rank)


def record(T=1., offset=0, condition='standard_nonthinking', seed=1):
    draft = torch.tensor([[2., 1., 0., -1.], [0., 0., 0., 0.], [1., 2., 3., 4.]])
    target = torch.tensor([[-1., 0., 1., 2.], [0., 2., 1., -1.], [1., 2., 3., 4.]])
    result = {'draft_logits': draft, 'l_unit': draft.log_softmax(-1), 'target_logits': target,
              'collection_temperature': T, 'sampling_warp_config': {'top_k': 0, 'top_p': 1., 'min_p': 0.},
              'first_rejection_index': 1, 'replacement_token': 2, 'accepted_length': 1,
              'draft_tokens': [0, 0, 3], 'h': torch.tensor([[1., -1.], [3., 0.], [6., 0.]]),
              'prompt_id': 'p1', 'seed': seed, 'condition': condition, 'split': 'test',
              'absolute_output_offset': offset, 'input_context_length': 4095}
    result['diagnostics'] = live_block_diagnostics(result, seed=13)
    return result


def test_live_exact_metrics_rank_mass_overlap_and_rng_independence():
    torch.manual_seed(81)
    before = torch.get_rng_state().clone()
    block = record()
    assert torch.equal(before, torch.get_rng_state())
    diagnostics = block['diagnostics']
    assert diagnostics['target_rank'] == [4, 2, 1]
    assert diagnostics['first_rejection']['replacement_token_rank'] == 3
    assert diagnostics['top_m']['1']['target_top1_recall'] == [0., 0., 1.]
    P, q = block['target_logits'].softmax(-1), block['draft_logits'].softmax(-1)
    torch.testing.assert_close(torch.tensor(diagnostics['overlap_unwarped']), torch.minimum(P, q).sum(-1))
    torch.testing.assert_close(torch.tensor(diagnostics['cross_entropy_unwarped']), -(P * q.log()).sum(-1))
    assert diagnostics['top_m']['256']['target_mass_sampling'] == pytest.approx([1., 1., 1.])
    assert live_block_diagnostics(block, seed=13) == diagnostics


def test_warped_support_infinite_cross_entropy_is_explicit_and_json_safe():
    block = record(T=.7)
    block['sampling_warp_config'] = {'top_k': 1, 'top_p': 1., 'min_p': 0.}
    result = live_block_diagnostics(block, seed=1)
    assert result['sampling_cross_entropy_infinite'][0]
    assert result['sampling_cross_entropy'][0] is None
    assert result['overlap_sampling'][0] == 0
    json.dumps(result, allow_nan=False)
    assert result['overlap_unwarped'][0] > 0


def test_repetition_detection_uses_realized_tokens_and_marks_empty_long_bins():
    looping = generation_diagnostics([1, 2, 3, 4] * 200)
    assert looping['repetition_dominated']
    assert looping['realized_length'] == 800
    natural = generation_diagnostics(list(range(3000)))
    assert not natural['repetition_dominated']
    assert natural['output_bins_reached'] == ['0-1K', '1K-4K']
    acc = DiagnosticAccumulator()
    acc.add_generation(dict(natural, condition='long_thinking', temperature=.6, split='test', complete=True, capped=False, eos_reached=True))
    summary = acc.report()['generation_lengths'][0]
    assert summary['empty_bins'] == ['4K-8K', '8K-16K', '16K+']
    assert summary['data_sparsity']
    assert not summary['long_context_conclusions_supported']


def test_bootstrap_aggregates_seeds_before_prompt_ids():
    observations = [{'prompt_id': 'a', 'seed': 1, 'value': 0},
                    {'prompt_id': 'a', 'seed': 2, 'value': 10},
                    {'prompt_id': 'b', 'seed': 1, 'value': 20}]
    result = bootstrap_prompt_means(observations, samples=2000)
    assert result['n_prompts'] == 2
    assert result['mean'] == 12.5
    assert result['ci95'] == [5., 20.]
    assert result['seed_std_mean'] == 2.5


def test_groups_separate_condition_temperature_bins_policy_and_repetition():
    acc = DiagnosticAccumulator()
    acc.add_block(record(T=1., offset=1023))
    acc.add_block(record(T=.7, offset=1023))
    acc.add_block(record(T=.6, offset=8193, condition='long_thinking'), repetition_dominated=True)
    report = acc.report()
    groups = report['groups']
    assert {g['collection_temperature'] for g in groups} == {1., .7, .6}
    one = [g for g in groups if g['collection_temperature'] == 1. and g['axis'] == 'output_offset']
    assert {g['bin'] for g in one} == {'0-1K', '1K-4K'}
    assert {g['data_policy'] for g in one} == {'on_policy', 'off_policy_teacher'}
    assert all(g['inclusion'] == 'repetition_excluded' for g in groups if g['condition'] == 'long_thinking')
    off_policy = next(g for g in groups if g['collection_temperature'] == 1 and g['axis'] == 'all' and g['data_policy'] == 'off_policy_teacher')
    assert off_policy['features']['outlier_slot_rate'] == 1.
    assert off_policy['features']['coordinate_mean_norm'] == 6.
    json.dumps(report, allow_nan=False)


def test_compact_union_cannot_silently_substitute_for_full_diagnostics():
    block = record()
    del block['diagnostics']
    with pytest.raises(ValueError, match='compact union'):
        DiagnosticAccumulator().add_block(block)
