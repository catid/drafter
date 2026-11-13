from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from edraft.adapter import DenseAdapter
from edraft.dflash_loop import generate_toy
from edraft.sampler import SamplingWarper
from edraft.sparse import SignedHash
from edraft.toy import ToyTarget, ToyParallelDrafter
from edraft.traces import TraceShardWriter, compact_block
from edraft.train import (batch_loss, batch_targets, corrected_batch_scores, fit,
                          load_adapter_checkpoint, load_frozen_head, primary_eligible,
                          reconstruct_batch, trace_index, validate_reference_gate)


def _args(output):
    return SimpleNamespace(output=output, scale='smoke', seed=71, history_depth=2, M_sparse=3,
                           sparse_capacity=32, learning_rate=.01, weight_decay=.0001,
                           bias_weight_decay=0., sparse_learning_rate=.02, lambda_l1=.0001,
                           batch_blocks=3, vocab_chunk=3, epochs_dense=1, epochs_sparse=1,
                           checkpoint_every=2, resume=True)


@pytest.fixture
def bundle(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    target = ToyTarget(vocab_size=8, hidden_size=6, seed=71)
    draft = ToyParallelDrafter(target, seed=72)
    writer = TraceShardWriter(tmp_path / 'traces')
    all_blocks = []
    for i, split in enumerate(['train', 'dev', 'test']):
        for temperature in [0., .7]:
            warper = SamplingWarper() if temperature == 0 else SamplingWarper(top_k=20, top_p=.8)
            result = generate_toy(target, draft, [i, 4, 2, 1], 12, block_length=3,
                                   temperature=temperature, warper=warper, retain_tensors=True,
                                   generator=torch.Generator().manual_seed(71))
            identity = {'model_pair': ['toy-target', 'toy-draft'], 'prompt': f'prompt-{split}',
                        'condition': 'standard_nonthinking', 'seed': 71, 'temperature': temperature,
                        'verification_rule': 'greedy' if temperature == 0 else 'ratio',
                        'proposal_mode': 'plain_dflash', 'checkpoint': 'synthetic-v1', 'adapter_version': 0,
                        'sampling_warp_config': warper.to_dict()}
            metadata = {'prompt_id': identity['prompt'], 'split': split, 'condition': identity['condition'],
                        'seed': 71, 'scale': 'synthetic', 'model_pair': identity['model_pair'],
                        'projection_metadata': {'method': 'synthetic-identity', 'frozen': True}, 'history_depth': 2}
            blocks = [compact_block(record, metadata, M_draft=2, M_target=1, retain_live_full=True)
                      for record in result.blocks]
            writer.write(identity, blocks, {'generation': {'output_tokens': result.output_tokens,
                'realized_length': result.num_output_tokens, 'condition': identity['condition'], 'split': split,
                'temperature': temperature, 'sampling_warp_config': warper.to_dict(), 'complete': True}})
            all_blocks.extend(blocks)
    entries, report = trace_index(tmp_path / 'traces')
    return entries, report, target.embedding[:, :3], draft.draft_lm_head, draft.head_metadata(), all_blocks


@pytest.mark.parametrize('change,allowed', [({}, True), ({'temperature': 0.}, True),
    ({'temperature': 1.}, False), ({'condition': 'long_thinking', 'temperature': .6}, False),
    ({'condition': 'long_nonthinking'}, False), ({'sampling_warp_config': {'top_k': 20, 'top_p': .95}}, False),
    ({'sampling_warp_config': {'top_k': 20, 'top_p': .8, 'min_p': .1}}, False)])
def test_primary_fit_context_eligibility(change, allowed):
    base = {'split': 'train', 'condition': 'standard_nonthinking', 'temperature': .7,
            'sampling_warp_config': {'top_k': 20, 'top_p': .8, 'min_p': 0.}}
    assert primary_eligible({**base, **change}, split='train') is allowed
    assert not primary_eligible({**base, 'split': 'test'}, split='train')


def test_batched_teacher_forcing_matches_per_block_scores_and_gradients(bundle):
    _, _, E, head, _, blocks = bundle
    blocks = blocks[:3]
    adapter = DenseAdapter(E, p=6, D=2, sparse=SignedHash(32), M_sparse=3)
    with torch.no_grad():
        adapter.W.normal_(0, .01)
        adapter.A.normal_(0, .02)
        adapter.b.normal_(0, .01)
        adapter.sparse.values.normal_(0, .03)
    l_unit = reconstruct_batch(blocks, head.weight, head.bias, vocab_chunk=3)
    scores = corrected_batch_scores(adapter, blocks, l_unit)
    separate, cursor = [], 0
    for block in blocks:
        K = block['block_length']
        l = l_unit[cursor:cursor + K]
        separate.append(l - adapter.correction(l, block['h'], block['verified_prefix_tail'], block['draft_tokens']))
        cursor += K
    torch.testing.assert_close(scores, torch.cat(separate), atol=2e-6, rtol=2e-6)
    loss, _, _ = batch_loss(adapter, blocks, l_unit)
    params = [adapter.W, adapter.A, adapter.b, adapter.sparse.values]
    actual = torch.autograd.grad(loss, params)
    ids, ps, rest, weights = batch_targets(blocks, 'cpu')
    q = torch.cat(separate).softmax(-1)
    independent = []
    for i, union in enumerate(ids):
        mask = torch.ones(q.shape[-1], dtype=torch.bool)
        mask[union] = False
        pgroup = torch.cat([ps[i], rest[i].reshape(1)])
        qgroup = torch.cat([q[i, union], q[i, mask].sum().reshape(1)])
        independent.append(torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs=pgroup), torch.distributions.Categorical(probs=qgroup)))
    expected_loss = (torch.stack(independent) * weights).sum()
    expected = torch.autograd.grad(expected_loss, params)
    torch.testing.assert_close(loss, expected_loss, atol=2e-6, rtol=2e-6)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_training_never_calls_inference_warpers(bundle, monkeypatch):
    _, _, E, head, _, blocks = bundle
    def forbidden(*a, **kw):
        raise AssertionError('training consulted an inference warper')
    monkeypatch.setattr('edraft.sampler.probabilities', forbidden)
    monkeypatch.setattr('edraft.sampler.apply_warper', forbidden)
    adapter = DenseAdapter(E, p=6, D=2)
    l_unit = reconstruct_batch(blocks[:2], head.weight, head.bias)
    base, _, _ = batch_loss(adapter, blocks[:2], l_unit)
    altered = deepcopy(blocks[:2])
    for block in altered:
        block['collection_temperature'] = 123
        block['sampling_warp_config'] = {'top_k': 1, 'top_p': .01, 'min_p': .99}
    result, _, _ = batch_loss(adapter, altered, l_unit)
    torch.testing.assert_close(base, result, atol=0, rtol=0)


@pytest.mark.parametrize('interruption_stage', ['dense', 'sparse'])
def test_full_fit_and_interrupted_resume_are_parameter_exact(bundle, tmp_path, interruption_stage):
    entries, report, E, head, metadata, _ = bundle
    full_report = fit(entries, report, E, head.weight, head.bias, metadata, _args(tmp_path / 'direct'))
    resumed_args = _args(tmp_path / 'resumed')
    first_epoch_steps = (sum(e.blocks for e in entries if e.split == 'train') + 2) // 3
    steps = 3 if interruption_stage == 'dense' else first_epoch_steps + 1
    interrupted = fit(entries, report, E, head.weight, head.bias, metadata, resumed_args, stop_after_steps=steps)
    assert interrupted['status'] == 'interrupted_for_resume_test'
    resumed_report = fit(entries, report, E, head.weight, head.bias, metadata, resumed_args)
    direct, _ = load_adapter_checkpoint(Path(full_report['checkpoints']['dense_sparse']))
    resumed, checkpoint_metadata = load_adapter_checkpoint(Path(resumed_report['checkpoints']['dense_sparse']))
    for key, value in direct.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    assert checkpoint_metadata['report']['head_metadata'] == metadata
    assert full_report['stages']['dense']['parameters']['b']['norm'] == 0
    assert 'S' not in full_report['stages']['dense']['parameters']
    assert full_report['relative_dev_improvement'] > 0
    diagnostic = full_report['stages']['dense_sparse']['test']['live_full_diagnostic']
    assert diagnostic['logical_slots'] > 0
    assert diagnostic['full_minus_grouped_kl'] >= -1e-6
    assert 0 <= diagnostic['mean_overlap'] <= 1
    direct_stats = full_report['stages']['dense_sparse']['sparse_collision_statistics']
    resumed_stats = resumed_report['stages']['dense_sparse']['sparse_collision_statistics']
    for name in ['observed_lookup_events', 'unique_observed_keys', 'collision_count', 'frequency_weighted_collision_count']:
        assert direct_stats[name] == resumed_stats[name]


def test_trace_index_rejects_prompt_split_leakage_even_in_excluded_condition(bundle, tmp_path):
    entries, _, _, _, _, _ = bundle
    source = entries[0]
    side_path = source.path.with_suffix('.pt.json')
    side = json.loads(side_path.read_text())
    side['identity']['prompt'] = 'prompt-test' if source.split != 'test' else 'prompt-train'
    side['identity']['condition'] = 'long_thinking'
    side['identity']['temperature'] = .6
    side['generation']['condition'] = 'long_thinking'
    side_path.write_text(json.dumps(side))
    with pytest.raises(ValueError, match='leaks across'):
        trace_index(tmp_path / 'traces')


def test_authoritative_audit_paths_exclude_stale_shards(bundle, tmp_path):
    entries, report, *_ = bundle
    accepted = [e.path for e in entries]
    stale = tmp_path / 'traces/rank0/stale.pt'
    stale.write_bytes(b'unpublished')
    selected, selected_report = trace_index(tmp_path / 'traces', accepted_paths=accepted)
    assert selected_report == report
    assert len(selected) == len(entries)


def test_reference_gate_cannot_lower_required_primary_floor(tmp_path):
    args = SimpleNamespace(scale='reference', m1_audit=tmp_path / 'audit.json')
    args.m1_audit.write_text(json.dumps({'status': 'passed', 'scale': 'reference'}))
    with pytest.raises(RuntimeError, match='500000'):
        validate_reference_gate(args, {'primary': {'train': {'generated_output_tokens': 499999}}}, [], None, {})
    args.m1_audit.write_text(json.dumps({'status': 'not_run'}))
    with pytest.raises(RuntimeError, match='M1'):
        validate_reference_gate(args, {}, [], None, {})


def test_load_exact_local_frozen_head_and_reject_wrong_provenance(tmp_path):
    from safetensors.torch import save_file
    revision = 'abc123'
    path = tmp_path / revision
    path.mkdir()
    weight = torch.randn(8, 6, dtype=torch.bfloat16)
    save_file({'lm_head.weight': weight}, path / 'model.safetensors')
    (path / 'config.json').write_text(json.dumps({'tie_word_embeddings': False}))
    metadata = {'artifact_id': str(path), 'revision': revision, 'shape': [8, 6],
                'dtype': 'torch.bfloat16', 'bias': False, 'tied_weights': False}
    loaded, bias = load_frozen_head(metadata, 'cpu')
    assert torch.equal(weight, loaded) and bias is None
    with pytest.raises(ValueError, match='revision'):
        load_frozen_head({**metadata, 'revision': 'different'}, 'cpu')
    with pytest.raises(ValueError, match='dtype'):
        load_frozen_head({**metadata, 'dtype': 'torch.float32'}, 'cpu')


def test_external_sparse_merge_exact_counts_collisions_and_corruption(tmp_path):
    from collections import Counter
    from edraft.sparse import hash_keys
    from edraft.sparse_stats import SparseAccessRecorder, pack_keys
    keys = [(1 + i % 2, i % 8, (3*i+2) % 8) for i in range(40)]
    recorder = SparseAccessRecorder(tmp_path, 'fixture', 8)
    packed = torch.tensor([pack_keys(*key, 8) for key in keys])
    chunks = [recorder.write_batch(packed[:17], 0, 0), recorder.write_batch(packed[17:], 0, 1)]
    table = SignedHash(3)
    table.values.data.copy_(torch.tensor([.3, -.2, .7]))
    result = recorder.finalize(chunks, table, benchmark_entries=8, benchmark_queries=32)
    frequencies = Counter(keys)
    occupancy, weighted = Counter(), Counter()
    for key, count in frequencies.items():
        bucket = int(hash_keys(*key, 3)[0])
        occupancy[bucket] += 1
        weighted[bucket] += count
    assert result['observed_lookup_events'] == len(keys)
    assert result['unique_observed_keys'] == len(frequencies)
    assert result['collision_count'] == sum(n-1 for n in occupancy.values())
    assert result['frequency_weighted_collision_count'] == sum(weighted[k] for k,n in occupancy.items() if n>1)
    assert result['lookup_benchmark']['query_values_equal']
    with pytest.raises(ValueError, match='double-count'):
        recorder.finalize(chunks+chunks, table)
    Path(chunks[0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        recorder.finalize(chunks, table)
