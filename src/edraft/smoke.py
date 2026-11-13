"""Download-free CPU integration gate exercising all mandatory decoding stages."""
from __future__ import annotations

import json
import time

import torch

from .runtime import ROOT, atomic_json, common_parser, parse_config, record_run


def run_smoke(args):
    from .adapter import DenseAdapter
    from .dflash_loop import generate_toy
    from .features import fit_hidden_projection, fit_token_features
    from .online import OnlineState, OnlineBigram
    from .sampler import SamplingWarper
    from .sparse import SignedHash
    from .toy import ToyTarget, ToyParallelDrafter

    if args.scale != 'smoke':
        raise ValueError('This entry point only runs built-in synthetic CPU models')
    torch.set_num_threads(1)
    record_run(args, 'smoke')
    target = ToyTarget(seed=args.seed)
    drafter = ToyParallelDrafter(target, seed=args.seed + 1)
    calibration = [[i % 8, (i + 3) % 8, (i * 3 + 1) % 8] for i in range(16)]
    projection = fit_hidden_projection([drafter.draft(p, 4).h_raw for p in calibration],
                                       target.hidden_size, p=6, seed=args.seed,
                                       prompt_ids=[f'synthetic-calibration-{i}' for i in range(16)])
    E, _ = fit_token_features(target.embedding, r=4, seed=args.seed,
                             calibration_token_ids=torch.tensor(sum(calibration, [])))
    adapter = DenseAdapter(E, p=6, D=4, sparse=SignedHash(256), M_sparse=4)
    prefix = [1, 4, 3, 2]
    outputs = []
    # Warmup is excluded from the measured synthetic throughput.
    generate_toy(target, drafter, prefix, 16, temperature=0)
    started = time.perf_counter()
    for T, warp in [(0., SamplingWarper()), (.7, SamplingWarper(top_k=3, top_p=.8)), (1., SamplingWarper())]:
        plain = generate_toy(target, drafter, prefix, 128, temperature=T, warper=warp,
                             generator=torch.Generator().manual_seed(args.seed))
        zero = generate_toy(target, drafter, prefix, 128, temperature=T, warper=warp,
                            generator=torch.Generator().manual_seed(args.seed), adapter=adapter,
                            projection=projection)
        assert torch.equal(plain.output_ids, zero.output_ids), 'zero adapter changed generated tokens'
        if T == 0:
            expected = target.generate(prefix, 128)
            assert plain.output_ids[0, len(prefix):].tolist() == expected, 'greedy target mismatch'
        outputs.extend([plain, zero])
    state = OnlineState(adapter, lr=.01, update_stride=1, lambda_l1_s=1e-4,
                        lambda_l1_b=1e-4, sparse_capacity=256)
    online = generate_toy(target, drafter, prefix, 256, temperature=1.,
                          generator=torch.Generator().manual_seed(args.seed + 5),
                          online_state=state, projection=projection)
    outputs.append(online)
    bigram = OnlineBigram(target.vocab_size)
    outputs.append(generate_toy(target, drafter, prefix, 128, temperature=.7,
                                warper=SamplingWarper(top_k=3, top_p=.8), bigram=bigram,
                                generator=torch.Generator().manual_seed(args.seed + 6)))
    elapsed = time.perf_counter() - started
    blocks = [block for result in outputs for block in result.blocks]
    residuals = sum(block.get('replacement_token') is not None for block in blocks)
    bonuses = sum(block.get('bonus_token') is not None for block in blocks)
    assert residuals > 0, 'smoke did not exercise rejection residuals'
    assert bonuses > 0, 'smoke did not exercise bonus tokens'
    assert state.u > 0 and state.db.norm() > 0, 'online update did not execute'
    count = sum(result.output_ids.shape[1] - len(prefix) for result in outputs)
    result = {'scale': 'smoke', 'data': 'synthetic', 'device': 'cpu', 'status': 'passed',
              'seed': args.seed, 'generated_tokens': count, 'elapsed_seconds': elapsed,
              'tokens_per_second': count / elapsed, 'rounds': len(blocks),
              'residual_corrections': residuals, 'bonus_tokens': bonuses,
              'zero_weight_token_parity': True, 'greedy_target_parity': True,
              'online': state.metrics(), 'bigram': bigram.metrics(),
              'note': 'CPU synthetic timing is not a real-model speed result.'}
    atomic_json(args.output / 'smoke.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    args = parse_config(common_parser(__doc__))
    run_smoke(args)


if __name__ == '__main__':
    main()
