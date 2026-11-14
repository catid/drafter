"""Measured adapter microbenchmarks; random inputs are always labeled synthetic."""
from __future__ import annotations

import json
import os
import statistics
import hashlib
from pathlib import Path
import subprocess
import time
import torch

from .adapter import DenseAdapter, prev_token, top_candidates
from .fastpath import FastPathConfig, FastPathRuntime
from .online import OnlineBigram, OnlineState
from .runtime import (common_parser, parse_config, record_run, resolved_config,
                      atomic_json, preflight, Manifest, fingerprint)
from .sampler import SamplingWarper, sample_token, sample_token_compact, sample_token_tensor
from .sparse import SignedHash


def summarize(samples, seed):
    tensor = torch.tensor(samples, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(samples), (1000, len(samples)), generator=generator)
    bootstrap = tensor[indices].median(-1).values
    return {'median_ms': statistics.median(samples), 'p95_ms': float(torch.quantile(tensor, .95)),
            'median_bootstrap_95_ci_ms': torch.quantile(bootstrap, torch.tensor([.025, .975], dtype=torch.float64)).tolist(),
            'samples_ms': samples}


def measure(function, device, warmups, repeats, seed):
    for _ in range(warmups):
        function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    wall, events = [], []
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
        then = time.perf_counter()
        function()
        if device.type == 'cuda':
            end.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
        wall.append((time.perf_counter() - then) * 1000)
        if device.type == 'cuda':
            events.append(start.elapsed_time(end))
    return {'synchronized_wall': summarize(wall, seed),
            'cuda_event': summarize(events, seed) if events else None}


def assert_idle(device):
    if device.type != 'cuda':
        return None
    # Query before initializing allocations. Never stop another user's process.
    logical_index = device.index or 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    selector = visible.split(',')[logical_index].strip() if visible is not None else str(logical_index)
    row = subprocess.check_output(['nvidia-smi', f'--id={selector}',
        '--query-gpu=memory.used,utilization.gpu,name,compute_cap', '--format=csv,noheader,nounits'], text=True).strip()
    memory, utilization, name, capability = [x.strip() for x in row.split(',')]
    if int(memory) > 128 or int(utilization) > 2:
        raise RuntimeError(f'Benchmark device is not idle: {row}')
    return {'name': name, 'logical_device_index': logical_index, 'nvidia_smi_selector': selector, 'compute_capability': capability, 'initial_used_MiB': int(memory),
            'initial_utilization_percent': int(utilization)}


def assert_no_external_gpu_processes(device):
    """Recheck isolation around each long all-arm comparison bundle."""
    if device.type != 'cuda':
        return
    index = device.index or 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    selector = visible.split(',')[index].strip() if visible is not None else str(index)
    gpu_uuid = subprocess.check_output(['nvidia-smi', f'--id={selector}', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    external = []
    for line in processes.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and fields[0] == gpu_uuid and int(fields[1]) != os.getpid():
            external.append(int(fields[1]))
    if external:
        raise RuntimeError(f'Isolated comparison interrupted: external compute processes {external} appeared on {selector}; current all-arm bundle will be rerun on resume')


def benchmark(args):
    device = torch.device(args.device)
    inventory = assert_idle(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    V, K, r, p, D, M = args.vocab_size, args.block_length, args.rank, args.projection_dim, args.history_depth, args.candidates
    E = torch.randn(V, r, device=device, generator=generator) / r**.5
    H = torch.randn(K, p, device=device, generator=generator)
    l_unit = torch.log_softmax(torch.randn(K, V, device=device, generator=generator), -1)
    sparse = SignedHash(args.hash_capacity, device=device)
    adapter = DenseAdapter(E, p, D, sparse=sparse, M_sparse=M)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(r, p, device=device, generator=generator) * .01)
        adapter.A.copy_(torch.randn(D, r, r, device=device, generator=generator) * .01)
        adapter.b.copy_(torch.randn(V, device=device, generator=generator) * .001)
        sparse.values.copy_(torch.randn(args.hash_capacity, device=device, generator=generator) * .001)
    snapshot = adapter.snapshot()
    U_full = snapshot.unary(H)
    candidates = top_candidates(l_unit - U_full, M)
    snapshot = snapshot.with_candidates(candidates)
    prefix = [int(x) for x in torch.randint(V, (D,), generator=generator, device=device).cpu()]
    fixed_tokens = [int(x) for x in torch.randint(V, (K,), generator=generator, device=device).cpu()]
    warper = SamplingWarper()  # pure-temperature sampling cost, no target model.
    sampler_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    timing = lambda fn: measure(fn, device, args.warmups, args.repeats, args.seed)

    def pair_messages():
        messages = []
        for i in range(K):
            msg = torch.zeros(r, device=device)
            for d in range(1, D + 1):
                previous = prev_token(prefix, fixed_tokens, i, d)
                if previous is not None:
                    msg = msg + snapshot.A[d - 1] @ E[previous]
            messages.append(msg)
        return messages

    messages = pair_messages()
    dense = torch.stack([U_full[i] + E @ messages[i] for i in range(K)])
    sparse_candidates = top_candidates(l_unit - dense, M)

    def sparse_lookup():
        result = []
        for i in range(K):
            ids = sparse_candidates[i]
            predecessors = [(d, prev_token(prefix, fixed_tokens, i, d)) for d in range(1, D + 1)]
            predecessors = [(d, p) for d, p in predecessors if p is not None]
            offsets = torch.tensor([d for d, _ in predecessors], device=device)[:, None]
            previous = torch.tensor([p for _, p in predecessors], device=device)[:, None]
            val = snapshot.sparse.lookup(offsets, previous, ids[None, :]).sum(0)
            result.append(val)
        return result

    def sampling():
        return [sample_token(l_unit[i] - dense[i], 1., warper, generator=sampler_rng)[0] for i in range(K)]

    def total(mode, active_snapshot=None):
        active_snapshot = snapshot if active_snapshot is None else active_snapshot
        unary = active_snapshot.unary(H, proposal_mode='dense_candidate' if mode == 'dense_candidate' else 'dense_full') if mode != 'sequential_full' else None
        realized = []
        for i in range(K):
            U_i = active_snapshot.unary(H[i:i + 1])[0] if mode == 'sequential_full' else unary[i]
            scores = active_snapshot.score_slot(l_unit[i], U_i, prefix, realized, i,
                                         proposal_mode='dense_candidate' if mode == 'dense_candidate' else 'dense_full')
            token, _ = sample_token(scores, 1., warper, generator=sampler_rng)
            realized.append(token)
        return realized

    phases = {'unary_batched_full': timing(lambda: snapshot.unary(H)),
              'unary_sequential_full': timing(lambda: [snapshot.unary(H[i:i + 1]) for i in range(K)]),
              'unary_true_candidate': timing(lambda: snapshot.unary(H, proposal_mode='dense_candidate')),
              'candidate_construction_including_full_unary': timing(lambda: top_candidates(l_unit - snapshot.unary(H), M)),
              'pair_message_construction': timing(pair_messages),
              'dense_pair_projection': timing(lambda: [E @ msg for msg in messages]),
              'sparse_lookup': timing(sparse_lookup),
              'sparse_top_m_selection': timing(lambda: top_candidates(l_unit - dense, M)),
              'sequential_sampling': timing(sampling)}
    target_logits = torch.randn(K, V, device=device, generator=generator)
    online_state = OnlineState(snapshot, lr=.001, lambda_l2=1e-4, lambda_l1_b=1e-5,
                               lambda_l1_s=1e-5, sparse_capacity=min(args.hash_capacity, 2**20), enable_sparse=True)
    online_round = [0]
    def online_step():
        online_round[0] += 1
        online_state.update(online_round[0], l_unit, target_logits, H, prefix, fixed_tokens)
    phases['online_update_step'] = timing(online_step)

    bigram = OnlineBigram(vocab_size=V, alpha=1., lambda_tr=.1, k_ret=min(32, V))
    bigram.update(prefix, fixed_tokens)
    phases['online_bigram_score'] = timing(
        lambda: [bigram.score(l_unit[i], fixed_tokens[i - 1] if i > 0 else prefix[-1]) for i in range(K)]
    )
    bigram_prev_t = torch.tensor(
        [fixed_tokens[i - 1] if i > 0 else prefix[-1] for i in range(K)],
        dtype=torch.long,
        device=device,
    )
    phases['online_bigram_score_tensor'] = timing(
        lambda: [bigram.score_tensor(l_unit[i], bigram_prev_t[i]) for i in range(K)]
    )
    compact_warper = SamplingWarper(top_k=min(20, max(1, V - 1)), top_p=0.8) if V > 1 else SamplingWarper()
    phases['sequential_sampling_compact_topk20'] = timing(
        lambda: [sample_token_compact(l_unit[i] - dense[i], 0.7, compact_warper, generator=sampler_rng)[0] for i in range(K)]
    )

    modes = {mode: timing(lambda mode=mode: total(mode))
             for mode in ('batched_full', 'sequential_full', 'dense_candidate')}

    prefix_t = torch.tensor(prefix, dtype=torch.long, device=device)
    ctx_buf = torch.empty(len(prefix) + K, dtype=torch.long, device=device)
    slot_offsets = [torch.arange(1, min(D, len(prefix) + i) + 1, dtype=torch.long, device=device)
                    for i in range(K)]

    def total_zero_sync(fastpath=None, active_warper=warper, temp=1.):
        bf16_mode = bool(fastpath is not None and fastpath.bf16_gemm)
        unary = fastpath.unary(E, snapshot.W, snapshot.b, H) if fastpath is not None else snapshot.unary(H)
        compact_fn = fastpath.compact_probabilities if fastpath is not None else None
        ctx_buf[:len(prefix)] = prefix_t
        for i in range(K):
            offs = slot_offsets[i]
            prev_ids = ctx_buf[len(prefix) + i - offs]
            scores = snapshot.score_slot_tensor(
                l_unit[i], unary[i], offs, prev_ids, contiguous_offsets=True, bf16_gemm=bf16_mode
            )
            tok_t, _, _ = sample_token_compact(
                scores, temp, active_warper, generator=sampler_rng, compact_probs_fn=compact_fn
            )
            ctx_buf[len(prefix) + i] = tok_t
        return ctx_buf[len(prefix):].tolist()

    modes['batched_full_zero_sync'] = timing(total_zero_sync)
    modes['batched_full_compact_topk'] = timing(lambda: total_zero_sync(None, compact_warper, 0.7))
    if getattr(args, 'compile', False) or getattr(args, 'cuda_graph', False) or getattr(args, 'bf16_gemm', False):
        fastpath = FastPathRuntime(
            FastPathConfig(
                compile=bool(getattr(args, 'compile', False)),
                cuda_graph=bool(getattr(args, 'cuda_graph', False)),
                bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
            ),
            static_block_length=K,
        )
        modes['batched_full_fastpath'] = timing(lambda: total_zero_sync(fastpath))
        online_fast = OnlineState(snapshot, lr=.001, lambda_l2=1e-4, lambda_l1_b=1e-5,
                                  lambda_l1_s=1e-5, sparse_capacity=min(args.hash_capacity, 2**20),
                                  enable_sparse=True, fastpath=fastpath)
        online_fast_round = [0]
        def online_fast_step():
            online_fast_round[0] += 1
            online_fast.update(online_fast_round[0], l_unit, target_logits, H, prefix, fixed_tokens)
        phases['online_update_step_fastpath'] = timing(online_fast_step)

    def candidate_end_to_end():
        # Immutable C0 construction itself incurs one full unary pass; charge it.
        C0 = top_candidates(l_unit - snapshot.unary(H), M)
        active_snapshot = snapshot.with_candidates(C0)
        return total('dense_candidate', active_snapshot)

    modes['dense_candidate_including_C0_construction'] = timing(candidate_end_to_end)
    precision = {'bf16_available': device.type == 'cuda' and torch.cuda.is_bf16_supported()}
    U_bf16_check = snapshot.unary(H, bf16_gemm=True)
    pair_fp32 = torch.stack([E @ msg for msg in messages])
    E_bf16_check = snapshot._token_features_bf16()
    pair_bf16 = torch.stack([(E_bf16_check @ msg.bfloat16()).float() for msg in messages])
    corr_fp32 = adapter.correction(l_unit, H, prefix, fixed_tokens)
    corr_bf16 = adapter.correction(l_unit, H, prefix, fixed_tokens, bf16_gemm=True)
    unary_diff = U_bf16_check - U_full
    pair_diff = pair_bf16 - pair_fp32
    corr_diff = corr_bf16 - corr_fp32
    q_fp32_all = torch.softmax(l_unit - U_full, -1)
    q_bf16_all = torch.softmax(l_unit - U_bf16_check, -1)
    bf16_audit = {
        'unary_max_abs_error': float(unary_diff.abs().max()),
        'unary_mean_abs_error': float(unary_diff.abs().mean()),
        'pair_max_abs_error': float(pair_diff.abs().max()),
        'pair_mean_abs_error': float(pair_diff.abs().mean()),
        'full_correction_max_abs_error': float(corr_diff.abs().max()),
        'full_correction_mean_abs_error': float(corr_diff.abs().mean()),
        'proposal_probability_l1_mean': float((q_fp32_all - q_bf16_all).abs().sum(-1).mean()),
        'unary_corrected_argmax_disagreement_rate': float(
            ((l_unit - U_full).argmax(-1) != (l_unit - U_bf16_check).argmax(-1)).float().mean()
        ),
        'full_corrected_argmax_disagreement_rate': float(
            ((l_unit - corr_fp32).argmax(-1) != (l_unit - corr_bf16).argmax(-1)).float().mean()
        ),
        'accumulation': 'PyTorch BF16 GEMM with FP32 accumulation, BF16 GEMM output then FP32 bias; no custom kernel',
    }
    precision['bf16_precision_audit'] = bf16_audit
    if precision['bf16_available'] or getattr(args, 'bf16_gemm', False):
        H_bf16, W_bf16, E_bf16 = H.bfloat16(), snapshot.W.bfloat16(), E.bfloat16()
        def unary_bf16():
            return ((H_bf16 @ W_bf16.T) @ E_bf16.T).float() + snapshot.b
        precision.update(bf16_audit)
        precision['bf16_forward_unary'] = timing(unary_bf16)
    return {'label': 'synthetic', 'requested_scale': args.scale, 'shape_kind': 'tiny' if args.scale == 'smoke' else 'reference-shape',
            'status': 'measured', 'device': str(device), 'device_inventory': inventory,
            'dimensions': {'V': V, 'K': K, 'r': r, 'p': p, 'D': D, 'M_sparse': M},
            'phases': phases, 'modes': modes, 'precision': precision,
            'notes': ['Random tensors; these measurements do not establish a model acceptance or end-to-end throughput gain.',
                      'Candidate scoring uses a different support; compare computation costs only.',
                      'Report total candidate cost including frozen C0 construction; scoring-only timings also shown.',
                      'All modes include sequential sampling and sparse lookup.'],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None}


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--block_length', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--projection_dim', type=int)
    parser.add_argument('--history_depth', type=int, default=4)
    parser.add_argument('--candidates', type=int)
    parser.add_argument('--hash_capacity', type=int)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--cuda_graph', action='store_true', default=False)
    parser.add_argument('--bf16_gemm', action='store_true', default=False)
    args = parse_config(parser, argv)
    tiny = args.scale == 'smoke'
    for key, value in {'vocab_size': 31 if tiny else 151936, 'block_length': 4 if tiny else 15,
                       'rank': 8 if tiny else 64, 'projection_dim': 8 if tiny else 256,
                       'candidates': 8 if tiny else 256, 'hash_capacity': 257 if tiny else 2**24}.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.repeats < 2 or args.warmups < 1:
        parser.error('require repeats>=2 and warmups>=1')
    record = record_run(args, 'bench_adapter')
    if tiny:
        from .smoke import run_smoke
        run_smoke(args)
    if not tiny:
        preflight(args, args.block_length * args.repeats * 4)
    source_hash = hashlib.sha256(b''.join((Path(__file__).parent / name).read_bytes() for name in ('bench_adapter.py', 'adapter.py', 'sparse.py', 'sampler.py', 'online.py', 'fastpath.py'))).hexdigest()
    identity = {'model_pair': [args.target, args.drafter], 'prompt': 'random_tensor_microbenchmark',
                'condition': 'synthetic', 'seed': args.seed, 'temperature': 1.,
                'verification_rule': 'not_applicable', 'proposal_mode': 'all_measured_modes',
                'checkpoint': {'configuration': fingerprint(resolved_config(args)), 'source_sha256': source_hash}, 'adapter_version': 0}
    manifest = Manifest(args.output / 'adapter_benchmark_manifest.json')
    path = args.output / f'adapter_benchmark_{args.scale}_{fingerprint(identity)[:12]}.json'
    if manifest.completed(identity):
        print(json.dumps({'status': 'already_measured', 'path': str(path)}))
        return json.loads(path.read_text()) if path.exists() else None
    torch.set_num_threads(1)
    with torch.no_grad():
        result = benchmark(args)
    result['run'] = record
    result['source_sha256'] = source_hash
    atomic_json(path, result)
    atomic_json(args.output / f'adapter_benchmark_{args.scale}.json', result)
    manifest.finish(identity, path)
    print(json.dumps({'status': 'measured', 'label': result['label'], 'path': str(path),
                      'median_wall_ms': {key: value['synchronized_wall']['median_ms'] for key, value in result['modes'].items()}}))
    return result


if __name__ == '__main__':
    main()
