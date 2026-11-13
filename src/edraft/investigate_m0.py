"""Investigate M0 target-only differences on identical prefixes, without downloads."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .backend import ReferenceBackend, crop_cache, make_cache
from .dflash_loop import generate_reference
from .features import reconstruct_l_unit
from .runtime import ROOT, Manifest, atomic_json, common_parser, parse_config, preflight, record_run


def distribution_summary(logits, tokenizer, selected=()):
    logits = logits.float()
    ids = torch.topk(logits, 5).indices.tolist()
    argmax = int(logits.argmax())
    top = logits.max()
    runner_up = logits.topk(2).values[1]
    return {
        'argmax': argmax, 'decoded_argmax': tokenizer.decode([argmax]),
        'top1_minus_top2': float(top - runner_up),
        'number_tied_for_max': int((logits == top).sum()),
        'top5': [{'id': i, 'text': tokenizer.decode([i]), 'logit': float(logits[i])} for i in ids],
        'selected_logits': {str(i): float(logits[i]) for i in selected},
    }


def difference(a, b):
    delta = (a.float() - b.float()).abs()
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'rms': float(delta.square().mean().sqrt()), 'argmax_equal': bool(a.argmax() == b.argmax())}


def cache_difference(a, b):
    if a.get_seq_length() != b.get_seq_length():
        return {'lengths_equal': False, 'a_length': int(a.get_seq_length()), 'b_length': int(b.get_seq_length())}
    keys, values = [], []
    for x, y in zip(a.layers, b.layers):
        keys.append(float((x.keys.float() - y.keys.float()).abs().max()))
        values.append(float((x.values.float() - y.values.float()).abs().max()))
    return {'lengths_equal': True, 'length': int(a.get_seq_length()),
            'max_key_abs_by_layer': keys, 'max_value_abs_by_layer': values,
            'max_key_abs': max(keys), 'max_value_abs': max(values)}


@torch.no_grad()
def sequential_prefix(backend, prompt_ids, generated_prefix):
    cache = make_cache(backend.target.config)
    out = backend.target(prompt_ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                         output_hidden_states=True)
    for token in generated_prefix:
        token_ids = prompt_ids.new_tensor([[token]])
        out = backend.target(token_ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                             output_hidden_states=True)
    return out, cache


@torch.no_grad()
def investigate_row(backend, row):
    offset = row['local_target']['first_mismatch']
    prompt_ids = backend.tokenize(row['prompt_text'], enable_thinking=False)
    n = prompt_ids.shape[1]
    shared = row['local_tokens'][:offset]
    if shared != row['target_tokens'][:offset]:
        raise RuntimeError('The selected offset is not the first divergent token')
    cache = make_cache(backend.target.config)
    backend.target(prompt_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    rollbacks = []
    for block in row['blocks']:
        begin = block['absolute_output_offset']
        canvas = prompt_ids.new_tensor([[row['local_tokens'][begin - 1]] + block['draft_tokens']])
        is_block = begin <= offset < begin + len(block['committed_tokens'])
        before = copy.deepcopy(cache) if is_block else None
        positions = torch.arange(n + begin - 1, n + begin - 1 + canvas.shape[1], device=prompt_ids.device)[None]
        output = backend.target(canvas, position_ids=positions, past_key_values=cache, use_cache=True,
                                output_hidden_states=is_block)
        keep = block['target_cache_length_after_rollback']
        crop_cache(cache, keep)
        rollbacks.append({'round': block['round_idx'], 'length': int(cache.get_seq_length()),
                          'expected': n + begin - 1 + len(block['committed_tokens'])})
        if not is_block:
            continue
        j = offset - begin
        block_logits = output.logits[0, j].float()
        sequential_out, sequential_cache = sequential_prefix(backend, prompt_ids, shared)
        sequential_logits = sequential_out.logits[0, -1].float()
        full_prefix = torch.cat([prompt_ids, prompt_ids.new_tensor([shared])], dim=1)
        fresh = backend.target(full_prefix, use_cache=False, logits_to_keep=1, output_hidden_states=True)
        fresh_logits = fresh.logits[0, -1].float()
        _, prior_sequential_cache = sequential_prefix(backend, prompt_ids, row['local_tokens'][:begin - 1])
        prior_cache_error = cache_difference(before, prior_sequential_cache)

        # Same cached prefix as the speculative block, but only the query at
        # which greedy outputs diverged. This isolates query-shape arithmetic.
        single_cache = copy.deepcopy(before)
        if j:
            backend.target(canvas[:, :j], position_ids=positions[:, :j], past_key_values=single_cache,
                           use_cache=True, logits_to_keep=1)
        single = backend.target(canvas[:, j:j + 1], position_ids=positions[:, j:j + 1],
                                past_key_values=single_cache, use_cache=True, output_hidden_states=True)
        single_logits = single.logits[0, -1].float()
        changed = canvas.clone()
        changed[:, j + 1:] = (changed[:, j + 1:] + 31) % backend.target.config.vocab_size
        mutated = backend.target(changed, position_ids=positions, past_key_values=copy.deepcopy(before),
                                 use_cache=True)
        future_error = difference(block_logits, mutated.logits[0, j])

        # h[-1] is the target's final normalized hidden state. FP32 head math
        # removes final BF16 output quantization while retaining each path's h.
        head_fp32 = backend.draft_lm_head.weight.float()
        block_fp32 = F.linear(output.hidden_states[-1][0, j].float(), head_fp32)
        sequential_fp32 = F.linear(sequential_out.hidden_states[-1][0, -1].float(), head_fp32)
        single_fp32 = F.linear(single.hidden_states[-1][0, -1].float(), head_fp32)
        selected = [row['local_tokens'][offset], row['target_tokens'][offset]]
        stats = {
            'prompt_id': row['identity']['prompt'], 'prompt_text': row['prompt_text'],
            'first_output_divergence': offset, 'round_idx': block['round_idx'], 'logical_slot': j,
            'common_prefix_length': len(shared) + n,
            'conditioning_prefix_verified_equal':
                row['local_tokens'][:begin] + block['draft_tokens'][:j] == shared,
            'recorded_local_token': selected[0], 'recorded_target_token': selected[1],
            'recorded_local_token_reproduced': int(block_logits.argmax()) == selected[0],
            'recorded_target_token_reproduced': int(sequential_logits.argmax()) == selected[1],
            'block': distribution_summary(block_logits, backend.tokenizer, selected),
            'target_sequential': distribution_summary(sequential_logits, backend.tokenizer, selected),
            'fresh_full_prefix': distribution_summary(fresh_logits, backend.tokenizer, selected),
            'single_query_same_prior_cache': distribution_summary(single_logits, backend.tokenizer, selected),
            'fp32_head_block': distribution_summary(block_fp32, backend.tokenizer, selected),
            'fp32_head_target_sequential': distribution_summary(sequential_fp32, backend.tokenizer, selected),
            'fp32_head_single_query': distribution_summary(single_fp32, backend.tokenizer, selected),
            'block_vs_sequential_logits': difference(block_logits, sequential_logits),
            'block_vs_single_query_same_prior_logits': difference(block_logits, single_logits),
            'block_vs_fresh_logits': difference(block_logits, fresh_logits),
            'block_head_bf16_vs_fp32': difference(block_logits, block_fp32),
            'future_tokens_perturbation': future_error,
            'rolled_cache_vs_sequential_prior': prior_cache_error,
            'rollback_lengths': rollbacks,
            'all_rollback_lengths_exact': all(x['length'] == x['expected'] for x in rollbacks),
            'target_attention': {'implementation': backend.target.config._attn_implementation,
                                 'sliding_window': backend.target.config.sliding_window,
                                 'causal': True, 'padded_input': False},
        }
        stats['diagnosis'] = (
            'Same-prefix BF16 numerical/tie sensitivity; future-token masking and exact rollback lengths verified.'
            if stats['conditioning_prefix_verified_equal'] and stats['all_rollback_lengths_exact']
            and future_error['max_abs'] == 0 and stats['recorded_local_token_reproduced']
            and stats['recorded_target_token_reproduced']
            else 'Unresolved: inspect recorded reproduction, cache, or future-mask evidence.'
        )
        return stats
    raise RuntimeError('No recorded verification block contains the first mismatch')


@torch.no_grad()
def inspect_real_head(backend, args):
    from .traces import reconstruction_diagnostic
    # Independent held-out prompt not used in M0's reproduction prompt list.
    prompt = 'Explain the difference between a stable sorting algorithm and a deterministic one, with two examples.'
    captures = []
    def capture(block):
        if len(captures) < 4:
            captures.append({k: block[k].detach().clone() for k in ['h_raw_bf16', 'draft_logits', 'l_unit']})
    generate_reference(backend, backend.tokenize(prompt, enable_thinking=False), 64, on_block=capture)
    head = backend.draft_lm_head.weight
    records = []
    for block in captures:
        h = block['h_raw_bf16']
        reconstructed = reconstruct_l_unit(h, head, backend.draft_lm_head.bias)
        live = block['l_unit'].float()
        delta = (reconstructed - live).abs()
        reprojected_bf16 = backend.draft_lm_head(h).float().log_softmax(-1)
        shared_diagnostic = reconstruction_diagnostic(
            {**block, 'draft_lm_head': backend.head_metadata()}, backend.draft_lm_head,
            atol=args.head_atol, rtol=args.head_rtol,
        )
        records.append({
            'slots': len(h), 'max_abs_logprob_error': float(delta.max()),
            'mean_abs_logprob_error': float(delta.mean()),
            'p99_abs_logprob_error': float(torch.quantile(delta.reshape(-1), .99)),
            'max_probability_error': float((reconstructed.exp() - live.exp()).abs().max()),
            'mean_KL_live_to_reconstructed': float((live.exp() * (live - reconstructed)).sum(-1).mean()),
            'bf16_head_replay_exact': bool(torch.equal(reprojected_bf16, live)),
            'within_configured_tolerance': bool(torch.allclose(reconstructed, live,
                                                              atol=args.head_atol, rtol=args.head_rtol)),
            'shared_trace_diagnostic': shared_diagnostic,
        })
    return {'scale': args.scale, 'prompt_id': 'm0-head-heldout-stable-sort', 'prompt': prompt,
            'hook': backend.head_metadata(), 'atol': args.head_atol, 'rtol': args.head_rtol,
            'reconstruction_dtype': 'FP32 input and weight with FP32 accumulation', 'blocks': records,
            'passed': all(r['within_configured_tolerance'] and r['bf16_head_replay_exact']
                          and r['shared_trace_diagnostic']['status'] == 'passed' for r in records)}


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--head_atol', type=float, default=.15)
    parser.add_argument('--head_rtol', type=float, default=.01)
    args = parse_config(parser)
    if args.scale == 'smoke':
        from .smoke import run_smoke
        return run_smoke(args)
    preflight(args, 5000)
    record_run(args, 'investigate_m0')
    env = json.loads((ROOT / 'artifacts/environment.json').read_text())
    models = env['model_artifacts']
    backend = ReferenceBackend.from_pretrained(models[args.target]['path'], models[args.drafter]['path'],
        target_revision=models[args.target]['revision'], drafter_revision=models[args.drafter]['revision'], device=args.device)
    manifest = Manifest(args.output / 'm0_numerics_manifest-rank0.json')
    rows = []
    for path in sorted((args.output / 'm0').glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('local_target', {}).get('sequence_equal') is False:
            rows.append(row)
    results = []
    for row in sorted(rows, key=lambda x: x['identity']['prompt']):
        identity = {**row['identity'], 'diagnostic': 'same_prefix_numerical_v1'}
        path = args.output / 'm0_numerics' / f'{manifest.key(identity)}.json'
        if manifest.completed(identity):
            result = json.loads(path.read_text())
        else:
            result = investigate_row(backend, row)
            atomic_json(path, result)
            manifest.finish(identity, path)
        results.append(result)
        print(result['prompt_id'], result['diagnosis'],
              'margins', result['block']['top1_minus_top2'], result['target_sequential']['top1_minus_top2'], flush=True)
    head = inspect_real_head(backend, args)
    report = {'scale': args.scale, 'status': 'passed' if head['passed'] and all(
        not r['diagnosis'].startswith('Unresolved') for r in results) else 'failed',
        'mismatch_rows_examined': len(results), 'findings': results, 'head_reconstruction': head,
        'model_revisions': {k: models[k]['revision'] for k in [args.target, args.drafter]},
        'note': 'Snapshot of completed M0 mismatch rows; rerun after M0 finishes to cover later rows.'}
    atomic_json(args.output / 'm0_numerics.json', report)
    print(json.dumps({'status': report['status'], 'rows': len(results), 'head': head}, indent=2), flush=True)
    if report['status'] != 'passed':
        raise RuntimeError('M0 numerical investigation has unresolved findings')


if __name__ == '__main__':
    main()
