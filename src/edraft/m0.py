"""M0 hard reproduction gate against the pinned, unmodified upstream DFlash."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .runtime import (ROOT, Manifest, atomic_json, common_parser, environment, parse_config,
                      preflight, record_run, resolved_config)
from .sampler import SamplingWarper

PROMPTS = [
    'Explain why the sky is blue in three sentences.',
    'Write a Python function that merges two sorted lists and explain its complexity.',
    'What is the sum of the first 100 positive integers? Show the calculation.',
    'Describe the difference between a process and a thread.',
    'Give a simple proof that there are infinitely many prime numbers.',
    'Write a short story about a gardener finding a mysterious key.',
    'Explain how a refrigerator moves heat.',
    'List five ways to make a database query faster and explain each.',
    'Translate into French: The train leaves tomorrow morning at eight.',
    'Describe the water cycle for a primary school student.',
    'Implement binary search in Python with comments.',
    'Compare breadth first search and depth first search.',
    'Why does a leap year usually occur every four years?',
    'Write a SQL query to count orders per customer, including customers with none.',
    'Explain the intuition behind Bayes theorem with an example.',
    'Describe three differences between RNA and DNA.',
    'A rectangle has perimeter 30 and width 5. Find its area.',
    'Suggest a seven day plan for learning basic Spanish.',
    'Explain why floating point addition is not associative.',
    'Write a haiku about autumn rain, followed by a brief explanation.',
]


def mismatch(a, b):
    n = min(len(a), len(b))
    positions = [i for i in range(n) if a[i] != b[i]] + list(range(n, max(len(a), len(b))))
    return {'mismatched_tokens': len(positions), 'compared_tokens': max(len(a), len(b)),
            'sequence_equal': not positions, 'first_mismatch': positions[0] if positions else None}


def run_reference(args):
    from .backend import ReferenceBackend
    from .dflash_loop import generate_reference
    from .download import fetch_model
    if args.num_prompts < 20:
        raise ValueError('M0 reference gate requires at least 20 prompts')
    preflight(args, args.num_prompts * args.max_new_tokens * 7)
    run_provenance = record_run(args, 'm0')
    env = environment(resolved_config(args))
    target_path = fetch_model(args.target, args.allow_download)
    draft_path = fetch_model(args.drafter, args.allow_download)
    # Re-read after authorized downloads pin the revisions.
    env = json.loads((ROOT / 'artifacts/environment.json').read_text())
    target_revision = env['model_artifacts'][args.target]['revision']
    draft_revision = env['model_artifacts'][args.drafter]['revision']
    print(f'Loading reference pair on {args.device}', flush=True)
    backend = ReferenceBackend.from_pretrained(target_path, draft_path, device=args.device,
                                               target_revision=target_revision, drafter_revision=draft_revision)
    outdir = args.output / 'm0'
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(outdir / 'manifest-rank0.json')
    rows = []
    conditions = [('greedy', 0., SamplingWarper()),
                  ('model_card_nonthinking', .7, SamplingWarper(top_k=20, top_p=.8)),
                  ('pure_temperature_diagnostic', 1., SamplingWarper())]
    for index in range(args.num_prompts):
        prompt = PROMPTS[index % len(PROMPTS)]
        if index >= len(PROMPTS):
            prompt += f' Give exactly {index % 5 + 2} paragraphs.'
        input_ids = backend.tokenize(prompt, enable_thinking=False)
        for name, temperature, warper in conditions:
            seed = args.seed + index
            identity = {'model_pair': [args.target, args.drafter], 'prompt': f'm0-{index:03}',
                        'condition': name, 'seed': seed, 'temperature': temperature,
                        'verification_rule': 'ratio' if temperature > 0 else 'greedy',
                        'proposal_mode': 'adapter_disabled', 'checkpoint': [target_revision, draft_revision],
                        'adapter_version': 0, 'sampling_warp_config': asdict(warper),
                        'max_new_tokens': args.max_new_tokens, 'enable_thinking': False,
                        'dflash_commit': env['dflash']['commit'], 'project_git_commit': env['project_git_commit']}
            path = outdir / f'{manifest.key(identity)}.json'
            if manifest.completed(identity):
                rows.append(json.loads(path.read_text()))
                continue
            torch.manual_seed(seed)
            upstream = backend.upstream_generate(input_ids, args.max_new_tokens, temperature=temperature,
                                                   top_p=warper.top_p, top_k=warper.top_k)
            torch.manual_seed(seed)
            local = generate_reference(backend, input_ids, args.max_new_tokens, temperature=temperature,
                                       warper=warper, retain_tensors=False, measure_phases=True)
            upstream_ids = upstream[0, input_ids.shape[1]:].tolist()
            local_ids = local.output_ids[0, input_ids.shape[1]:].tolist()
            row = {'scale': args.scale, 'identity': identity, 'prompt_text': prompt,
                   'upstream_local': mismatch(upstream_ids, local_ids),
                   'upstream_tokens': upstream_ids, 'local_tokens': local_ids,
                   'blocks': local.blocks, 'phase_ms': local.phase_ms,
                   'local_elapsed_seconds': local.elapsed_seconds}
            if temperature == 0:
                target = backend.target_only_generate(input_ids, args.max_new_tokens)
                target_ids = target[0, input_ids.shape[1]:].tolist()
                row.update({'target_tokens': target_ids,
                            'upstream_target': mismatch(upstream_ids, target_ids),
                            'local_target': mismatch(local_ids, target_ids)})
            atomic_json(path, row)
            manifest.finish(identity, path)
            rows.append(row)
            print(f'{index+1}/{args.num_prompts} {name}: parity={row["upstream_local"]["sequence_equal"]}, '
                  f'{len(local_ids)} tokens, {local.elapsed_seconds:.2f}s', flush=True)
            if not row['upstream_local']['sequence_equal']:
                atomic_json(outdir / 'first_failure.json', row)
                raise RuntimeError(f'M0 upstream/local reproduction failed: {path}')
    probe = backend.tokenize(PROMPTS[0], enable_thinking=False)
    proposal = backend.target_only_generate(probe, 4)[..., probe.shape[1]:]
    rollback = [backend.check_cache_rollback(probe, proposal, accepted, 42) for accepted in [0, 1, 3]]
    greedy = [r for r in rows if r['identity']['temperature'] == 0]
    aggregate = {}
    for comparison in ['upstream_local', 'upstream_target', 'local_target']:
        relevant = [r[comparison] for r in rows if comparison in r]
        total = sum(r['compared_tokens'] for r in relevant)
        aggregate[comparison] = {
            'mismatched_tokens': sum(r['mismatched_tokens'] for r in relevant), 'compared_tokens': total,
            'token_mismatch_rate': sum(r['mismatched_tokens'] for r in relevant) / max(total, 1),
            'sequence_mismatch_rate': sum(not r['sequence_equal'] for r in relevant) / len(relevant)}
    status = 'passed' if all(r['argmax_equal'] and r['cache_length'] == r['expected_length'] for r in rollback) else 'failed'
    summary = {'scale': args.scale, 'status': status, 'num_distinct_prompts': args.num_prompts,
               'run_provenance': run_provenance,
               'comparisons': aggregate, 'cache_rollback': rollback, 'head_metadata': backend.head_metadata(),
               'conditions': [{'name': n, 'temperature': t, 'warper': asdict(w)} for n, t, w in conditions],
               'target_mismatch_investigation_required': any(not r['local_target']['sequence_equal'] for r in greedy)}
    atomic_json(args.output / 'm0.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if status != 'passed':
        raise RuntimeError('Real cache rollback gate failed')
    return summary


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--num_prompts', type=int, default=20)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parse_config(parser)
    if args.scale == 'smoke':
        from .smoke import run_smoke
        run_smoke(args)
    else:
        run_reference(args)


if __name__ == '__main__':
    main()
