"""Synthetic microbenchmark of exact grouped-KL reductions; never a model fit."""
from __future__ import annotations

import json
import torch

from .adapter import loss_weights, union_rest_grouped as grouped_vectorized_candidate
from .bench_adapter import assert_idle, measure
from .runtime import Manifest, atomic_json, common_parser, fingerprint, parse_config, preflight, record_run


def grouped_loop_reference(scores, union_ids, target_union_probabilities, target_rest_mass, weights=None):
    """Frozen pre-vectorization implementation, retained as a numerical oracle."""
    q_log = torch.log_softmax(scores.float(), -1)
    losses = []
    for i, ids in enumerate(union_ids):
        p = target_union_probabilities[i].float().detach()
        log_q_u = q_log[i, ids]
        rest_mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        rest_mask[ids] = False
        has_rest = bool(rest_mask.any())
        log_q_rest = torch.logsumexp(q_log[i, rest_mask], 0) if has_rest else scores.new_tensor(0.)
        rest_p = torch.as_tensor(target_rest_mass[i], device=scores.device, dtype=torch.float32).detach()
        if not has_rest:
            if float(rest_p) > 2e-6:
                raise ValueError('nonzero target rest mass is inconsistent with a full-vocabulary union')
            rest_p = torch.zeros_like(rest_p)
        union_loss = torch.xlogy(p, p).sum() - (p * log_q_u).sum()
        rest_loss = torch.where(rest_p > 0, torch.xlogy(rest_p, rest_p) - rest_p * log_q_rest, rest_p * 0)
        losses.append(union_loss + rest_loss)
    if weights is None:
        weights = loss_weights(len(scores), scores.device)
    return (torch.stack(losses) * weights).sum()


def run(args):
    device = torch.device(args.device)
    inventory = assert_idle(device)
    record = record_run(args, 'bench_grouped')
    if args.scale != 'smoke':
        preflight(args, 0)
    torch.set_num_threads(args.cpu_threads)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    N, V = args.rows, args.vocab_size
    scores = (torch.randn(N, V, generator=generator, device=device) * 2).requires_grad_(True)
    target = (torch.randn(N, V, generator=generator, device=device) * 2).softmax(-1)
    lengths = [min(V, max(1, args.union_size - i % 17)) for i in range(N)]
    ids = [torch.randperm(V, device=device, generator=generator)[:n].sort().values for n in lengths]
    p = [target[i, selected] for i, selected in enumerate(ids)]
    rest = torch.stack([(1 - row.double().sum()).clamp_min(0).float() if len(row) < V
                        else row.new_zeros(()) for row in p])
    weights = torch.exp(-(torch.arange(N, device=device) % 15) / 5.)
    functions = {'loop': grouped_loop_reference, 'vectorized': grouped_vectorized_candidate}
    parity = {}
    loss_ref = grouped_loop_reference(scores, ids, p, rest, weights)
    grad_ref = torch.autograd.grad(loss_ref, scores)[0]
    loss_vec = grouped_vectorized_candidate(scores, ids, p, rest, weights)
    grad_vec = torch.autograd.grad(loss_vec, scores)[0]
    parity = {'loss_abs_error': float((loss_ref-loss_vec).detach().abs()),
              'gradient_max_abs_error': float((grad_ref-grad_vec).abs().max()),
              'loss_atol': 2e-5, 'loss_rtol': 2e-5, 'gradient_atol': 2e-6, 'gradient_rtol': 2e-4,
              'passed': bool(torch.allclose(loss_ref, loss_vec, atol=2e-5, rtol=2e-5)
                              and torch.allclose(grad_ref, grad_vec, atol=2e-6, rtol=2e-4))}
    if not parity['passed']:
        raise RuntimeError(f'Grouped vectorization parity failed: {parity}')
    timings = {}
    for name, fn in functions.items():
        for backward in [False, True]:
            def operation(fn=fn, backward=backward):
                if backward:
                    loss = fn(scores, ids, p, rest, weights)
                    torch.autograd.grad(loss, scores)
                else:
                    with torch.no_grad():
                        fn(scores, ids, p, rest, weights)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            timings[f'{name}_{"forward_backward" if backward else "forward"}'] = measure(
                operation, device, args.warmups, args.repeats, args.seed)
            if device.type == 'cuda':
                timings[f'{name}_{"forward_backward" if backward else "forward"}'].update({
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
    old = timings['loop_forward_backward']['synchronized_wall']['median_ms']
    new = timings['vectorized_forward_backward']['synchronized_wall']['median_ms']
    report = {'scale': args.scale, 'data': 'synthetic', 'status': 'measured', 'run': record,
              'inventory': inventory, 'shape': {'rows': N, 'vocab': V, 'union_max': max(lengths),
                                              'union_min': min(lengths)},
              'parity': parity, 'timings': timings, 'forward_backward_speedup': old / new,
              'replacement_gate_passed': parity['passed'] and new < old,
              'scope': 'synthetic grouped-loss operator only; no model training or model throughput claim'}
    identity = {'model_pair': 'synthetic_score_tensors', 'prompt': fingerprint(report['shape']),
                'condition': 'grouped_loss_microbenchmark', 'seed': args.seed, 'temperature': 1.,
                'verification_rule': 'not_applicable', 'proposal_mode': 'full_partition_grouped_loss',
                'checkpoint': record['source_tree_sha256'], 'adapter_version': 0}
    path = args.output / f'grouped_benchmark_{args.scale}.json'
    atomic_json(path, report)
    Manifest(args.output / 'grouped_benchmark_manifest.json').finish(identity, path)
    return report


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--rows', type=int, default=120)
    parser.add_argument('--vocab_size', type=int, default=151936)
    parser.add_argument('--union_size', type=int, default=320)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--cpu_threads', type=int, default=1)
    args = parse_config(parser)
    if min(args.rows, args.vocab_size, args.union_size, args.repeats) < 1 or args.warmups < 0:
        raise ValueError('Invalid grouped benchmark shape or repeat counts')
    if args.scale == 'smoke':
        args.rows, args.vocab_size, args.union_size = min(args.rows, 12), min(args.vocab_size, 32), min(args.union_size, 8)
    result = run(args)
    print(json.dumps({k: result[k] for k in ['scale', 'data', 'parity', 'forward_backward_speedup', 'replacement_gate_passed']}, indent=2))


if __name__ == '__main__':
    main()
