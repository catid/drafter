"""Synthetic microbenchmark of exact grouped-KL reductions; never a model fit."""
from __future__ import annotations

import json
import torch

from .adapter import (collate_grouped_targets, fused_head_grouped_kl, loss_weights,
                      union_rest_grouped as grouped_vectorized_candidate,
                      union_rest_grouped_precollated)
from .bench_adapter import assert_idle, measure
from .fastpath import _fresh_compile
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
    precollated = collate_grouped_targets(ids, p, rest, weights, vocab_size=V, device=device)
    hidden_dim = 64 if args.scale == 'smoke' else 512
    h_raw_bf16 = torch.randn(N, hidden_dim, generator=generator, device=device).bfloat16()
    head_weight = (torch.randn(V, hidden_dim, generator=generator, device=device) * (hidden_dim ** -0.5)).bfloat16()
    dense_corr = (torch.randn(N, V, generator=generator, device=device) * 0.1).requires_grad_(True)

    functions = {
        'loop': grouped_loop_reference,
        'vectorized': grouped_vectorized_candidate,
        'precollated': lambda sc, _i, _p, _r, _w: union_rest_grouped_precollated(sc, precollated),
    }
    if getattr(args, 'compile', False):
        compiled_pre = _fresh_compile(
            lambda sc, pre: union_rest_grouped_precollated(sc, pre),
            'default',
        )
        functions['compiled_precollated'] = lambda sc, _i, _p, _r, _w: compiled_pre(sc, precollated)

    loss_ref = grouped_loop_reference(scores, ids, p, rest, weights)
    grad_ref = torch.autograd.grad(loss_ref, scores)[0]
    loss_vec = grouped_vectorized_candidate(scores, ids, p, rest, weights)
    grad_vec = torch.autograd.grad(loss_vec, scores)[0]
    loss_pre = union_rest_grouped_precollated(scores, precollated)
    grad_pre = torch.autograd.grad(loss_pre, scores)[0]

    fused_scores_ref = (torch.nn.functional.linear(h_raw_bf16.float(), head_weight.float()) - dense_corr)
    loss_fused_ref = grouped_loop_reference(fused_scores_ref, ids, p, rest, weights)
    grad_fused_ref = torch.autograd.grad(loss_fused_ref, dense_corr)[0]
    loss_fused = fused_head_grouped_kl(h_raw_bf16, head_weight, None, dense_corr, precollated)
    grad_fused = torch.autograd.grad(loss_fused, dense_corr)[0]

    comp_passed = True
    comp_loss_err = 0.0
    comp_grad_err = 0.0
    if getattr(args, 'compile', False):
        loss_cmp = functions['compiled_precollated'](scores, ids, p, rest, weights)
        grad_cmp = torch.autograd.grad(loss_cmp, scores)[0]
        comp_loss_err = float((loss_ref - loss_cmp).detach().abs())
        comp_grad_err = float((grad_ref - grad_cmp).abs().max())
        comp_passed = bool(
            torch.allclose(loss_ref, loss_cmp, atol=2e-5, rtol=2e-5)
            and torch.allclose(grad_ref, grad_cmp, atol=2e-6, rtol=2e-4)
        )

    parity = {
        'loss_abs_error': float((loss_ref - loss_vec).detach().abs()),
        'gradient_max_abs_error': float((grad_ref - grad_vec).abs().max()),
        'precollated_loss_abs_error': float((loss_ref - loss_pre).detach().abs()),
        'precollated_gradient_max_abs_error': float((grad_ref - grad_pre).abs().max()),
        'fused_head_loss_abs_error': float((loss_fused_ref - loss_fused).detach().abs()),
        'fused_head_gradient_max_abs_error': float((grad_fused_ref - grad_fused).abs().max()),
        'compiled_loss_abs_error': comp_loss_err if getattr(args, 'compile', False) else None,
        'compiled_gradient_max_abs_error': comp_grad_err if getattr(args, 'compile', False) else None,
        'loss_atol': 2e-5,
        'loss_rtol': 2e-5,
        'gradient_atol': 2e-6,
        'gradient_rtol': 2e-4,
        'passed': bool(
            torch.allclose(loss_ref, loss_vec, atol=2e-5, rtol=2e-5)
            and torch.allclose(grad_ref, grad_vec, atol=2e-6, rtol=2e-4)
            and torch.allclose(loss_ref, loss_pre, atol=2e-5, rtol=2e-5)
            and torch.allclose(grad_ref, grad_pre, atol=2e-6, rtol=2e-4)
            and torch.allclose(loss_fused_ref, loss_fused, atol=2e-5, rtol=2e-5)
            and torch.allclose(grad_fused_ref, grad_fused, atol=2e-6, rtol=2e-4)
            and comp_passed
        ),
    }
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
    for backward in [False, True]:
        def fused_op(backward=backward):
            if backward:
                loss = fused_head_grouped_kl(h_raw_bf16, head_weight, None, dense_corr, precollated)
                torch.autograd.grad(loss, dense_corr)
            else:
                with torch.no_grad():
                    fused_head_grouped_kl(h_raw_bf16, head_weight, None, dense_corr, precollated)
        timings[f'fused_head_grouped_{"forward_backward" if backward else "forward"}'] = measure(
            fused_op, device, args.warmups, args.repeats, args.seed)
    old = timings['loop_forward_backward']['synchronized_wall']['median_ms']
    new = timings['vectorized_forward_backward']['synchronized_wall']['median_ms']
    pre_ms = timings['precollated_forward_backward']['synchronized_wall']['median_ms']
    report = {'scale': args.scale, 'data': 'synthetic', 'status': 'measured', 'run': record,
              'inventory': inventory, 'shape': {'rows': N, 'vocab': V, 'union_max': max(lengths),
                                              'union_min': min(lengths)},
              'parity': parity, 'timings': timings, 'forward_backward_speedup': old / new,
              'precollated_forward_backward_speedup': old / pre_ms,
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


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument('--rows', type=int, default=120)
    parser.add_argument('--vocab_size', type=int, default=151936)
    parser.add_argument('--union_size', type=int, default=320)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--compile', action='store_true', default=False)
    args = parse_config(parser, argv)
    if min(args.rows, args.vocab_size, args.union_size, args.repeats) < 1 or args.warmups < 0:
        raise ValueError('Invalid grouped benchmark shape or repeat counts')
    if args.scale == 'smoke':
        args.rows, args.vocab_size, args.union_size = min(args.rows, 12), min(args.vocab_size, 32), min(args.union_size, 8)
    result = run(args)
    print(json.dumps({k: result[k] for k in ['scale', 'data', 'parity', 'forward_backward_speedup', 'replacement_gate_passed']}, indent=2))
    return result


if __name__ == '__main__':
    main()
