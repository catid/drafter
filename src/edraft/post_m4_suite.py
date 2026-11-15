"""Reproducible post-Gate-E ablation runner (`python -m edraft.post_m4_suite`).

Executes three isolated, resumable post-M4 experiment suites without mutating
frozen M0-M4 artifacts:
  1. `fastpath_bench`:
     - `bench_adapter` microbenchmark + Round-3 compact proposal microbenchmarks
       (`OnlineBigram.merge_compact_candidates`, `precompute_sparse_pool` +
       `score_slot_compact_candidates`),
     - `bench_grouped` microbenchmark (`loop` vs `vectorized` vs `precollated`
       LSE-replay vs `fused_head_grouped`),
     - Matched back-to-back closed-loop comparison of pre-Round-3 eager full-V
       scoring vs Round-3 compact scoring vs Round-3 + `--compile` (`tok/s`,
       paired ratios, and phase-profile `ms`).
  2. `online_head_to_head`:
     - Matched closed-loop evaluation + isolated timing across
       `plain_dflash`, `online_bigram`, `online_bigram_count_gated` (`kappa=12`),
       `online_dbS_zero_init`, and `zero_offline_online_only`, reporting overall
       metrics, per-offset (`0-1K`..`16K+`) acceptance & `tok/s` slices, and
       paired head-to-head deltas.
  3. `m2_refit_calibration`:
     - Block-shuffled + weight-decayed + dev-beta-calibrated M2 refit
       (`--block_shuffle_window 32 --weight_decay 1e-2 --calibrate_beta`),
     - Scalar `0.20 * dense.pt` / `0.20 * dense_sparse.pt` calibration checkpoints
       via `scale_adapter_checkpoint`,
     - Held-out offline KL (`dev` & `test` `live_full` and `union_rest_grouped`)
       and closed-loop acceptance/throughput comparison across uncalibrated,
       dev-calibrated, and `0.20`-scaled checkpoints.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch

from .adapter import DenseAdapter, top_candidates
from .bench_adapter import assert_idle, assert_no_external_gpu_processes, measure
from .dflash_loop import generate_toy
from .diagnose import OUTPUT_LABELS
from .eval_closed_loop import (
    EvaluationRunner,
    evaluation_jobs,
    implementation_hash,
    sampling_conditions,
    select_prompts,
    smoke_manifest,
)
from .fastpath import FastPathConfig, FastPathRuntime
from .features import HiddenProjection, fit_hidden_projection
from .online import OnlineBigram
from .runtime import (
    ROOT,
    atomic_json,
    common_parser,
    fingerprint,
    git_commit,
    parse_config,
    preflight,
    record_run,
    resolved_config,
    source_provenance,
)
from .sampler import SamplingWarper, sample_from_top_scaled, sample_token_compact
from .sparse import SignedHash
from .toy import ToyParallelDrafter, ToyTarget
from .traces import read_shard
from .train import (
    epoch_references,
    evaluate as evaluate_offline,
    load_adapter_checkpoint,
    load_frozen_head,
    scale_adapter_checkpoint,
    smoke_bundle,
    trace_index,
)

DEFAULT_ONLINE_HEAD_TO_HEAD_ARMS = (
    'plain_dflash',
    'online_bigram',
    'online_bigram_count_gated',
    'online_dbS_zero_init',
    'zero_offline_online_only',
)

DEFAULT_FASTPATH_CLOSED_LOOP_ARMS = (
    'plain_dflash',
    'online_bigram',
    'offline_static',
    'offline_online_stride1',
    'online_dbS',
)


def _link_prereqs(src_root: Path, dst_root: Path, names: tuple[str, ...]) -> None:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    if src_root == dst_root:
        return
    for rel in names:
        src = src_root / rel
        dst = dst_root / rel
        if not src.exists() or dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def _cli_base_flags(args, output_dir: Path) -> list[str]:
    flags = [
        '--scale', args.scale,
        '--seed', str(args.seed),
        '--output', str(output_dir),
    ]
    if args.scale != 'smoke':
        flags.extend([
            '--device', str(args.device),
            '--target', str(args.target),
            '--drafter', str(args.drafter),
        ])
        if getattr(args, 'allow_download', False):
            flags.append('--allow_download')
        if getattr(args, 'allow_large_run', False):
            flags.append('--allow_large_run')
    return flags


def _measure_round3_compact_microbench(args, device: torch.device) -> dict:
    tiny = args.scale == 'smoke'
    V = 64 if tiny else 151936
    K = 4 if tiny else 15
    r = 8 if tiny else 64
    p = 8 if tiny else 256
    D = 4
    M = 8 if tiny else 256
    cap = 257 if tiny else 2**20
    warmups = args.bench_warmups
    repeats = args.bench_repeats

    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    E = torch.randn(V, r, device=device, generator=gen) / (r ** 0.5)
    H = torch.randn(K, p, device=device, generator=gen)
    l_unit = torch.log_softmax(torch.randn(K, V, device=device, generator=gen), dim=-1)
    sparse = SignedHash(cap, device=device)
    adapter = DenseAdapter(E, p, D, sparse=sparse, M_sparse=M)
    with torch.no_grad():
        adapter.b.copy_(torch.randn(V, device=device, generator=gen) * 0.01)
        sparse.values.copy_(torch.randn(cap, device=device, generator=gen) * 0.01)
    snap_zero_pair = adapter.snapshot()
    assert snap_zero_pair.pair_is_zero()

    with torch.no_grad():
        adapter.W.copy_(torch.randn(r, p, device=device, generator=gen) * 0.01)
        adapter.A.copy_(torch.randn(D, r, r, device=device, generator=gen) * 0.01)
    snap_full_pair = adapter.snapshot()

    prefix = [int(x) for x in torch.randint(V, (D,), generator=gen, device=device).cpu()]
    fixed_tokens = [int(x) for x in torch.randint(V, (K,), generator=gen, device=device).cpu()]
    top_k = min(20, max(1, V - 1))
    warper = SamplingWarper(top_k=top_k, top_p=0.8)
    temperature = 0.7

    bigram = OnlineBigram(
        vocab_size=V,
        alpha=1.0,
        lambda_tr=0.15,
        k_ret=min(32, V),
        count_gate_kappa=args.bigram_count_gate_kappa,
    )
    bigram.update(prefix, fixed_tokens * 3)

    prefix_t = torch.tensor(prefix, dtype=torch.long, device=device)
    ctx_buf = torch.empty(D + K, dtype=torch.long, device=device)
    offsets_all = torch.arange(1, D + 1, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed + 31)

    def bigram_round2_full_v():
        ctx_buf[:D] = prefix_t
        for i in range(K):
            sc = bigram.score_tensor(l_unit[i], ctx_buf[D + i - 1])
            tok_t, _, _ = sample_token_compact(sc, temperature, warper, generator=rng)
            ctx_buf[D + i] = tok_t
        return ctx_buf[D:]

    def bigram_round3_compact():
        ctx_buf[:D] = prefix_t
        st = bigram._sync_device_table(device, l_unit.dtype)
        _, base_ids = torch.topk(l_unit / temperature, top_k, dim=-1)
        base_vals = l_unit.gather(-1, base_ids)
        for i in range(K):
            sel_sc, sel_ids = bigram.merge_compact_candidates(
                l_unit[i], base_vals[i], base_ids[i], ctx_buf[D + i - 1],
                temperature=temperature, k_eff=top_k, state=st,
            )
            tok_t, _, _, _ = sample_from_top_scaled(sel_sc, sel_ids, warper, rng)
            ctx_buf[D + i] = tok_t
        return ctx_buf[D:]

    def sparse_round2_full_v(active_snap):
        ctx_buf[:D] = prefix_t
        U = active_snap.unary(H)
        for i in range(K):
            prev_ids = ctx_buf[D + i - offsets_all]
            sc = active_snap.score_slot_tensor(
                l_unit[i], U[i], offsets_all[:, None], prev_ids, contiguous_offsets=True
            )
            tok_t, _, _ = sample_token_compact(sc, temperature, warper, generator=rng)
            ctx_buf[D + i] = tok_t
        return ctx_buf[D:]

    def sparse_round3_compact(active_snap):
        ctx_buf[:D] = prefix_t
        U = active_snap.unary(H)
        pre = (
            active_snap.precompute_sparse_pool(l_unit, U, temperature=temperature, k_eff=top_k, beta=1.0)
            if active_snap.pair_is_zero()
            else None
        )
        for i in range(K):
            prev_ids = ctx_buf[D + i - offsets_all]
            pre_i = (pre[0][i], pre[1][i]) if pre is not None else None
            sel_sc, sel_ids = active_snap.score_slot_compact_candidates(
                l_unit[i], U[i], offsets_all[:, None], prev_ids,
                temperature=temperature, k_eff=top_k, beta=1.0,
                contiguous_offsets=True, precomputed_ids=pre_i,
            )
            if sel_sc is None:
                tok_t, _, _ = sample_token_compact(sel_ids, temperature, warper, generator=rng)
            else:
                tok_t, _, _, _ = sample_from_top_scaled(sel_sc, sel_ids, warper, rng)
            ctx_buf[D + i] = tok_t
        return ctx_buf[D:]

    t_bg_r2 = measure(bigram_round2_full_v, device, warmups, repeats, args.seed)
    t_bg_r3 = measure(bigram_round3_compact, device, warmups, repeats, args.seed)
    t_dbS_r2 = measure(lambda: sparse_round2_full_v(snap_zero_pair), device, warmups, repeats, args.seed)
    t_dbS_r3 = measure(lambda: sparse_round3_compact(snap_zero_pair), device, warmups, repeats, args.seed)
    t_full_r2 = measure(lambda: sparse_round2_full_v(snap_full_pair), device, warmups, repeats, args.seed)
    t_full_r3 = measure(lambda: sparse_round3_compact(snap_full_pair), device, warmups, repeats, args.seed)

    def _med(entry):
        return float(entry['synchronized_wall']['median_ms'])

    return {
        'dimensions': {'V': V, 'K': K, 'r': r, 'p': p, 'D': D, 'M_sparse': M, 'top_k': top_k},
        'count_gate_kappa': args.bigram_count_gate_kappa,
        'timings': {
            'bigram_round2_full_v': t_bg_r2,
            'bigram_round3_compact': t_bg_r3,
            'online_dbS_round2_full_v': t_dbS_r2,
            'online_dbS_round3_precomputed_pool': t_dbS_r3,
            'full_adapter_round2_full_v': t_full_r2,
            'full_adapter_round3_compact_candidates': t_full_r3,
        },
        'speedups': {
            'bigram_round3_over_round2': _med(t_bg_r2) / max(_med(t_bg_r3), 1e-12),
            'online_dbS_round3_over_round2': _med(t_dbS_r2) / max(_med(t_dbS_r3), 1e-12),
            'full_adapter_round3_over_round2': _med(t_full_r2) / max(_med(t_full_r3), 1e-12),
        },
    }


def _default_closed_loop_setting(arm: str, kappa: float) -> dict:
    if arm == 'plain_dflash':
        return {}
    if arm == 'online_bigram':
        return {'lambda_tr': 0.05, 'alpha': 0.5, 'k_ret': None}
    if arm == 'online_bigram_count_gated':
        return {'lambda_tr': 0.05, 'alpha': 0.5, 'k_ret': None, 'count_gate_kappa': kappa}
    if arm in ('offline_static', 'dense', 'dense_sparse', 'dense_full'):
        return {'beta': 1.0}
    return {
        'beta': 1.0,
        'lr': 1e-4,
        'update_stride': 1,
        'lambda_l2': 0.01,
        'lambda_l1_b': 1e-6,
        'lambda_l1_s': 1e-5,
        'weighting_mode': 'exponential',
    }


def _run_matched_closed_loop_fastpath_bench(args, artifacts_root: Path, bench_dir: Path) -> dict:
    ws = bench_dir / 'closed_loop_workspace'
    if args.scale != 'smoke':
        _link_prereqs(artifacts_root, ws, ('m2.json', 'm2', 'm3', 'm4', 'prompts.json', 'h_proj.pt', 'environment.json'))

    prompt_manifest = (
        smoke_manifest()
        if args.scale == 'smoke'
        else json.loads((ws / 'prompts.json').read_text())
    )
    all_test = select_prompts(prompt_manifest, 'test', args.scale)
    selected_prompts = []
    limit = args.timing_prompts
    for cond in sorted({p['condition'] for p in all_test}):
        c_prompts = [p for p in all_test if p['condition'] == cond]
        selected_prompts.extend(c_prompts if limit == 0 else c_prompts[:limit])

    m4_settings_path = ws / 'm4' / 'settings.json'
    frozen_selected = (
        json.loads(m4_settings_path.read_text()).get('selected', {})
        if m4_settings_path.is_file()
        else {}
    )
    arms = DEFAULT_FASTPATH_CLOSED_LOOP_ARMS
    arm_settings = {
        arm: frozen_selected.get(arm, _default_closed_loop_setting(arm, args.bigram_count_gate_kappa))
        for arm in arms
    }

    runner_ns = SimpleNamespace(
        scale=args.scale,
        stage='m4' if (args.scale == 'smoke' or (ws / 'm3' / 'summary.json').is_file()) else 'm3',
        mode='timing',
        seed=args.seed,
        device=args.device,
        target=args.target,
        drafter=args.drafter,
        allow_download=getattr(args, 'allow_download', False),
        allow_large_run=getattr(args, 'allow_large_run', False),
        output=ws,
        prompts_manifest=ws / 'prompts.json',
        dense_checkpoint=ws / 'm2' / 'dense.pt',
        sparse_checkpoint=ws / 'm2' / 'dense_sparse.pt',
        projection=ws / 'h_proj.pt',
        max_new_tokens=args.max_new_tokens,
        long_max_new_tokens=args.long_max_new_tokens,
        smoke_tokens=args.smoke_tokens,
        compile=False,
        cuda_graph=False,
        simulate_cpu_graph=False,
        bf16_gemm=False,
        compile_drafter=False,
        compile_online=False,
        disable_round3_compact=False,
        m5_oracle_checkpoint=None,
        checkpoint_scale=None,
        m5_triton=False,
        m5_torch_compile=False,
    )
    runner = EvaluationRunner(runner_ns)
    static_k = 4 if runner.tiny else (runner.backend.block_size - 1)
    compiled_fp = FastPathRuntime(FastPathConfig(compile=True), static_block_length=static_k)

    modes = (
        ('eager_pre_round3', True, None),
        ('round3_compact', False, None),
        ('round3_compile', False, compiled_fp),
    )

    jobs = []
    for prompt in selected_prompts:
        name, temperature, warp = sampling_conditions(prompt['condition'], primary_only=True)[0]
        for arm in arms:
            jobs.append({
                'prompt': prompt,
                'sampling_name': name,
                'temperature': temperature,
                'warper': warp,
                'seed': args.seed,
                'arm': arm,
            })

    def _invoke(mode_disable_r3: bool, mode_fp, job: dict, profile: bool) -> dict:
        runner.args.disable_round3_compact = mode_disable_r3
        runner.fastpath = mode_fp
        return runner.run(job, arm_settings[job['arm']], profile=profile)

    for cond in sorted({p['condition'] for p in selected_prompts}):
        w_prompt = next(p for p in selected_prompts if p['condition'] == cond)
        w_name, w_temp, w_warp = sampling_conditions(cond, primary_only=True)[0]
        for _ in range(args.timing_warmups):
            for _, dis_r3, fp_rt in modes:
                for arm in arms:
                    w_job = {
                        'prompt': w_prompt,
                        'sampling_name': w_name,
                        'temperature': w_temp,
                        'warper': w_warp,
                        'seed': args.seed,
                        'arm': arm,
                        'max_new_tokens_override': 64,
                    }
                    _invoke(dis_r3, fp_rt, w_job, profile=False)

    speed_rows = []
    profile_rows = []
    for j_idx, job in enumerate(jobs):
        for rep in range(args.timing_repeats):
            rot = (j_idx + rep) % len(modes)
            ordered_modes = modes[rot:] + modes[:rot]
            for mode_name, dis_r3, fp_rt in ordered_modes:
                assert_no_external_gpu_processes(runner.device)
                row = _invoke(dis_r3, fp_rt, job, profile=False)
                speed_rows.append({
                    'fastpath_mode': mode_name,
                    'arm': job['arm'],
                    'prompt_id': job['prompt']['prompt_id'],
                    'condition': job['prompt']['condition'],
                    'repeat': rep,
                    'generated_tokens': row['generated_tokens'],
                    'elapsed_seconds': row['elapsed_seconds'],
                    'tokens_per_second': row['tokens_per_second'],
                    'mean_accepted_length': row['mean_accepted_length'],
                })
        for mode_name, dis_r3, fp_rt in modes:
            assert_no_external_gpu_processes(runner.device)
            p_row = _invoke(dis_r3, fp_rt, job, profile=True)
            profile_rows.append({
                'fastpath_mode': mode_name,
                'arm': job['arm'],
                'prompt_id': job['prompt']['prompt_id'],
                'condition': job['prompt']['condition'],
                'phase_ms': p_row['phase_ms'],
                'generated_tokens': p_row['generated_tokens'],
            })

    runner.args.disable_round3_compact = False
    runner.fastpath = None

    summary_by_mode_arm = []
    for mode_name, _, _ in modes:
        for arm in arms:
            m_rows = [r for r in speed_rows if r['fastpath_mode'] == mode_name and r['arm'] == arm]
            p_rows = [r for r in profile_rows if r['fastpath_mode'] == mode_name and r['arm'] == arm]
            phase_keys = sorted({k for r in p_rows for k in (r.get('phase_ms') or {})})
            phase_means = {
                k: statistics.mean(float(r['phase_ms'][k]) for r in p_rows if k in r['phase_ms'])
                for k in phase_keys
            }
            summary_by_mode_arm.append({
                'fastpath_mode': mode_name,
                'arm': arm,
                'timed_runs': len(m_rows),
                'mean_tokens_per_second': statistics.mean(r['tokens_per_second'] for r in m_rows) if m_rows else None,
                'median_tokens_per_second': statistics.median(r['tokens_per_second'] for r in m_rows) if m_rows else None,
                'mean_accepted_length': statistics.mean(r['mean_accepted_length'] for r in m_rows) if m_rows else None,
                'mean_phase_ms': phase_means,
            })

    by_slot = {
        (r['fastpath_mode'], r['arm'], r['prompt_id'], r['repeat']): r['tokens_per_second']
        for r in speed_rows
    }
    paired_speedups = []
    for arm in arms:
        r3_over_eager = []
        comp_over_r3 = []
        comp_over_eager = []
        for (m_name, a_name, pid, rep), tps_eager in by_slot.items():
            if m_name != 'eager_pre_round3' or a_name != arm or tps_eager <= 0:
                continue
            tps_r3 = by_slot.get(('round3_compact', arm, pid, rep))
            tps_cmp = by_slot.get(('round3_compile', arm, pid, rep))
            if tps_r3 is not None:
                r3_over_eager.append(tps_r3 / tps_eager)
            if tps_cmp is not None and tps_r3 and tps_r3 > 0:
                comp_over_r3.append(tps_cmp / tps_r3)
            if tps_cmp is not None:
                comp_over_eager.append(tps_cmp / tps_eager)
        paired_speedups.append({
            'arm': arm,
            'round3_over_eager_mean_ratio': statistics.mean(r3_over_eager) if r3_over_eager else None,
            'compile_over_round3_mean_ratio': statistics.mean(comp_over_r3) if comp_over_r3 else None,
            'compile_over_eager_mean_ratio': statistics.mean(comp_over_eager) if comp_over_eager else None,
            'pairs': len(r3_over_eager),
        })

    return {
        'arms': list(arms),
        'modes': [m[0] for m in modes],
        'prompt_ids': [p['prompt_id'] for p in selected_prompts],
        'repeats': args.timing_repeats,
        'warmups': args.timing_warmups,
        'summary_by_mode_and_arm': summary_by_mode_arm,
        'paired_speedups': paired_speedups,
        'speed_rows': speed_rows,
        'profile_rows': profile_rows,
    }


def run_fastpath_bench(args, artifacts_root: Path, suite_root: Path) -> dict:
    from . import bench_adapter, bench_grouped

    device = torch.device(args.device)
    inventory = assert_idle(device)
    sub_dir = suite_root / 'fastpath_bench'
    sub_dir.mkdir(parents=True, exist_ok=True)
    report_path = suite_root / 'fastpath_bench.json'
    if args.resume and report_path.is_file():
        existing = json.loads(report_path.read_text())
        if existing.get('status') == 'complete' and existing.get('scale') == args.scale:
            return existing

    adapter_dir = sub_dir / 'adapter'
    grouped_dir = sub_dir / 'grouped'
    adapter_report = bench_adapter.main(
        _cli_base_flags(args, adapter_dir) + [
            '--warmups', str(args.bench_warmups),
            '--repeats', str(args.bench_repeats),
            '--compile',
        ]
    )
    grouped_report = bench_grouped.main(
        _cli_base_flags(args, grouped_dir) + [
            '--warmups', str(args.bench_warmups),
            '--repeats', str(args.bench_repeats),
            '--compile',
        ]
    )
    with torch.no_grad():
        round3_micro = _measure_round3_compact_microbench(args, device)
    closed_loop = _run_matched_closed_loop_fastpath_bench(args, artifacts_root, sub_dir)

    report = {
        'suite': 'fastpath_bench',
        'scale': args.scale,
        'status': 'complete',
        'device': str(device),
        'device_inventory': inventory,
        'adapter_microbenchmark': {
            'path': str(adapter_dir / f'adapter_benchmark_{args.scale}.json'),
            'median_wall_ms': {
                k: v['synchronized_wall']['median_ms']
                for k, v in adapter_report['modes'].items()
            },
        },
        'round3_compact_microbenchmark': round3_micro,
        'grouped_kl_microbenchmark': {
            'path': str(grouped_dir / f'grouped_benchmark_{args.scale}.json'),
            'parity_passed': grouped_report['parity']['passed'],
            'forward_backward_speedup': grouped_report['forward_backward_speedup'],
            'precollated_forward_backward_speedup': grouped_report['precollated_forward_backward_speedup'],
        },
        'closed_loop_matched_comparison': closed_loop,
    }
    atomic_json(report_path, report)
    return report


def run_online_head_to_head(args, artifacts_root: Path, suite_root: Path) -> dict:
    from . import eval_closed_loop

    report_path = suite_root / 'online_head_to_head.json'
    if args.resume and report_path.is_file():
        existing = json.loads(report_path.read_text())
        if existing.get('status') == 'complete' and existing.get('scale') == args.scale:
            return existing

    sub_ws = suite_root / 'online_head_to_head_workspace'
    if args.scale != 'smoke':
        _link_prereqs(
            artifacts_root,
            sub_ws,
            ('m2.json', 'm2', 'm3', 'm4', 'prompts.json', 'h_proj.pt', 'environment.json'),
        )

    arms = list(args.online_arms)
    cli = _cli_base_flags(args, sub_ws) + [
        '--stage', 'm5',
        '--mode', 'all',
        '--bigram_count_gate_kappa', str(args.bigram_count_gate_kappa),
        '--timing_repeats', str(args.timing_repeats),
        '--timing_warmups', str(args.timing_warmups),
        '--timing_prompts', str(args.timing_prompts),
        '--max_new_tokens', str(args.max_new_tokens),
        '--long_max_new_tokens', str(args.long_max_new_tokens),
        '--smoke_tokens', str(args.smoke_tokens),
        '--prompts_manifest', str(sub_ws / 'prompts.json'),
        '--dense_checkpoint', str(sub_ws / 'm2' / 'dense.pt'),
        '--sparse_checkpoint', str(sub_ws / 'm2' / 'dense_sparse.pt'),
        '--projection', str(sub_ws / 'h_proj.pt'),
        '--m5_arms', *arms,
    ]
    eval_closed_loop.main(cli)

    summary = json.loads((sub_ws / 'm5' / 'summary.json').read_text())
    timing = json.loads((sub_ws / 'm5' / 'timing.json').read_text())
    settings = json.loads((sub_ws / 'm5' / 'settings.json').read_text())

    report = {
        'suite': 'online_head_to_head',
        'scale': args.scale,
        'status': 'complete',
        'arms': arms,
        'bigram_count_gate_kappa': args.bigram_count_gate_kappa,
        'offset_labels': list(OUTPUT_LABELS),
        'selected_settings': settings.get('selected', {}),
        'evaluation_status': summary.get('status'),
        'timing_status': timing.get('status'),
        'overall_evaluation_groups': summary.get('groups', []),
        'isolated_timing_groups': timing.get('groups', []),
        'evaluation_offset_bin_slices': summary.get('offset_bin_slices', []),
        'isolated_timing_offset_bin_slices': timing.get('offset_bin_slices', []),
        'head_to_head_online_offset_comparison': summary.get('head_to_head_online_offset_comparison', []),
        'matched_support_comparisons': summary.get('matched_support_comparisons', []),
        'artifacts': {
            'summary_json': str((sub_ws / 'm5' / 'summary.json').resolve()),
            'timing_json': str((sub_ws / 'm5' / 'timing.json').resolve()),
            'settings_json': str((sub_ws / 'm5' / 'settings.json').resolve()),
        },
    }
    atomic_json(report_path, report)
    return report


def _ensure_smoke_baseline_m2(args, ws: Path) -> None:
    from . import train

    if not (ws / 'm2_smoke.json').is_file():
        train.main(_cli_base_flags(args, ws))


def _smoke_closed_loop_checkpoint_eval(
    args,
    dense_ckpt: Path,
    sparse_ckpt: Path,
) -> dict:
    device = torch.device('cpu')
    target = ToyTarget(vocab_size=8, hidden_size=12, seed=args.seed)
    drafter = ToyParallelDrafter(target, seed=args.seed + 1)
    calibration = [[i % 8, (i + 3) % 8, (i * 3 + 1) % 8] for i in range(64)]
    projection = fit_hidden_projection(
        [drafter.draft(x, 4).h_raw for x in calibration],
        12,
        6,
        seed=args.seed,
        prompt_ids=[f'synthetic-train-{i}' for i in range(64)],
    ).to(device)
    dense_adapter, _ = load_adapter_checkpoint(dense_ckpt, device)
    sparse_adapter, _ = load_adapter_checkpoint(sparse_ckpt, device)
    dense_adapter.eval()
    sparse_adapter.eval()

    prompts = select_prompts(smoke_manifest(), 'test', 'smoke')
    arms = (
        ('plain_dflash', None),
        ('dense', dense_adapter),
        ('dense_sparse', sparse_adapter),
    )
    by_arm = {name: [] for name, _ in arms}
    for p in prompts:
        prefix = [1, 2, 3, int(fingerprint(p['prompt_id'])[:4], 16) % 8]
        for s_name, temp, warp in sampling_conditions(p['condition'], primary_only=False):
            eff_warp = SamplingWarper(
                top_k=min(warp.top_k, 7) if warp.top_k > 0 else 0,
                top_p=warp.top_p,
                min_p=warp.min_p,
            )
            for arm_name, adapter in arms:
                gen = torch.Generator(device=device).manual_seed(args.seed)
                t0 = time.perf_counter()
                res = generate_toy(
                    target, drafter, prefix, args.smoke_tokens,
                    temperature=temp, warper=eff_warp, generator=gen,
                    adapter=adapter,
                    projection=projection if adapter is not None else None,
                    beta=1.0,
                )
                elapsed = max(time.perf_counter() - t0, 1e-12)
                acc = statistics.mean(b['accepted_length'] for b in res.blocks) if res.blocks else 0.0
                by_arm[arm_name].append({
                    'prompt_id': p['prompt_id'],
                    'sampling_name': s_name,
                    'mean_accepted_length': acc,
                    'tokens_per_second': res.num_output_tokens / elapsed,
                    'target_forwards_per_generated_token': len(res.blocks) / max(res.num_output_tokens, 1),
                })
    out = {}
    for arm_name, rows in by_arm.items():
        out[arm_name] = {
            'requests': len(rows),
            'mean_accepted_length': statistics.mean(r['mean_accepted_length'] for r in rows),
            'mean_tokens_per_second': statistics.mean(r['tokens_per_second'] for r in rows),
            'mean_target_forwards_per_token': statistics.mean(
                r['target_forwards_per_generated_token'] for r in rows
            ),
        }
    return out


def _prepare_offline_eval_context(args, ws: Path, device: torch.device):
    eval_ns = SimpleNamespace(
        scale=args.scale,
        seed=args.seed,
        output=ws,
        history_depth=4,
        batch_blocks=8,
        vocab_chunk=8192,
        bf16_gemm=False,
        objective='union_rest_grouped',
        allow_long_oracle_fit=False,
        allow_pure_t1_ablation=False,
    )
    if args.scale == 'smoke':
        trace_dir, _, head, bias, head_metadata = smoke_bundle(eval_ns)
        entries, _ = trace_index(trace_dir)
    else:
        audit = json.loads((ws / 'm1_audit.json').read_text())
        entries, _ = trace_index(ws / 'traces', accepted_paths=audit['shards'])
        first = read_shard(entries[0].path)['blocks'][0]
        head_metadata = first['draft_lm_head']
        head, bias = load_frozen_head(head_metadata, device)
    refs = {
        'dev': epoch_references(entries, 'dev'),
        'test': epoch_references(entries, 'test'),
    }
    return eval_ns, refs, head, bias, head_metadata


def _offline_kl_summary(eval_ctx, ckpt_path: Path, device: torch.device) -> dict:
    eval_ns, refs, head, bias, head_metadata = eval_ctx
    adapter, meta = load_adapter_checkpoint(ckpt_path, device)
    adapter.eval()
    dev_eval = evaluate_offline(adapter, refs['dev'], head, bias, eval_ns, head_metadata)
    test_eval = evaluate_offline(adapter, refs['test'], head, bias, eval_ns, head_metadata)
    return {
        'checkpoint': str(ckpt_path),
        'stage': meta.get('stage'),
        'checkpoint_scale': meta.get('checkpoint_scale', 1.0),
        'dev': dev_eval,
        'test': test_eval,
    }


def run_m2_refit_calibration(args, artifacts_root: Path, suite_root: Path) -> dict:
    from . import eval_closed_loop, train

    report_path = suite_root / 'm2_refit_calibration.json'
    if args.resume and report_path.is_file():
        existing = json.loads(report_path.read_text())
        if existing.get('status') == 'complete' and existing.get('scale') == args.scale:
            return existing

    ws = suite_root / 'm2_refit_workspace'
    if args.scale == 'smoke':
        _ensure_smoke_baseline_m2(args, ws)
    else:
        _link_prereqs(
            artifacts_root,
            ws,
            (
                'traces', 'm1_audit.json', 'h_proj.pt', 'token_features.pt',
                'm2.json', 'm2', 'm3', 'm4', 'prompts.json', 'environment.json',
            ),
        )

    fit_tag = args.refit_tag
    fit_subdir = f'{fit_tag}_smoke' if args.scale == 'smoke' else fit_tag
    base_subdir = 'm2_smoke' if args.scale == 'smoke' else 'm2'
    train_cli = _cli_base_flags(args, ws) + [
        '--m5_fit_tag', fit_tag,
        '--block_shuffle_window', str(args.block_shuffle_window),
        '--weight_decay', str(args.weight_decay),
        '--calibrate_beta',
    ]
    if args.scale != 'smoke':
        train_cli.extend([
            '--trace_dir', str(ws / 'traces'),
            '--m1_audit', str(ws / 'm1_audit.json'),
            '--features', str(ws / 'token_features.pt'),
            '--projection', str(ws / 'h_proj.pt'),
            '--prompts', str(ws / 'prompts.json'),
        ])
    refit_report = train.main(train_cli)
    if refit_report is None:
        refit_report = json.loads((ws / f'{fit_subdir}.json').read_text())

    refit_dir = ws / fit_subdir
    base_dir = ws / base_subdir
    scale_tag = f'{args.fixed_checkpoint_scale:.2f}'.replace('.', 'p')
    refit_dense_s = refit_dir / f'dense_scale{scale_tag}.pt'
    refit_sparse_s = refit_dir / f'dense_sparse_scale{scale_tag}.pt'
    scale_adapter_checkpoint(refit_dir / 'dense.pt', refit_dense_s, args.fixed_checkpoint_scale)
    scale_adapter_checkpoint(refit_dir / 'dense_sparse.pt', refit_sparse_s, args.fixed_checkpoint_scale)

    base_scaled_dir = suite_root / 'baseline_m2_scaled'
    base_dense_s = base_scaled_dir / f'dense_scale{scale_tag}.pt'
    base_sparse_s = base_scaled_dir / f'dense_sparse_scale{scale_tag}.pt'
    scale_adapter_checkpoint(base_dir / 'dense.pt', base_dense_s, args.fixed_checkpoint_scale)
    scale_adapter_checkpoint(base_dir / 'dense_sparse.pt', base_sparse_s, args.fixed_checkpoint_scale)

    variants = [
        ('baseline_m2_unscaled', base_dir / 'dense.pt', base_dir / 'dense_sparse.pt'),
        (f'baseline_m2_scale_{scale_tag}', base_dense_s, base_sparse_s),
        ('refit_shuf_wd_uncalibrated', refit_dir / 'dense.pt', refit_dir / 'dense_sparse.pt'),
        (
            'refit_shuf_wd_dev_calibrated',
            refit_dir / 'dense_calibrated.pt',
            refit_dir / 'dense_sparse_calibrated.pt',
        ),
        (f'refit_shuf_wd_scale_{scale_tag}', refit_dense_s, refit_sparse_s),
    ]

    device = torch.device(args.device)
    eval_ctx = _prepare_offline_eval_context(args, ws, device)
    variant_reports = []
    for v_name, d_ckpt, s_ckpt in variants:
        dense_kl = _offline_kl_summary(eval_ctx, d_ckpt, device)
        sparse_kl = _offline_kl_summary(eval_ctx, s_ckpt, device)
        if args.scale == 'smoke':
            cl_summary = _smoke_closed_loop_checkpoint_eval(args, d_ckpt, s_ckpt)
        else:
            v_ws = suite_root / 'closed_loop_variants' / v_name
            _link_prereqs(
                ws,
                v_ws,
                ('m2.json', 'm2', 'm3', 'm4', 'prompts.json', 'h_proj.pt', 'environment.json'),
            )
            eval_closed_loop.main(
                _cli_base_flags(args, v_ws) + [
                    '--stage', 'm5',
                    '--mode', 'all',
                    '--dense_checkpoint', str(d_ckpt),
                    '--sparse_checkpoint', str(s_ckpt),
                    '--projection', str(ws / 'h_proj.pt'),
                    '--prompts_manifest', str(ws / 'prompts.json'),
                    '--timing_repeats', str(args.timing_repeats),
                    '--timing_warmups', str(args.timing_warmups),
                    '--timing_prompts', str(args.timing_prompts),
                    '--max_new_tokens', str(args.max_new_tokens),
                    '--long_max_new_tokens', str(args.long_max_new_tokens),
                    '--m5_arms', 'plain_dflash', 'dense_full',
                ]
            )
            v_sum = json.loads((v_ws / 'm5' / 'summary.json').read_text())
            v_tim = json.loads((v_ws / 'm5' / 'timing.json').read_text())
            cl_summary = {
                'evaluation_groups': v_sum.get('groups', []),
                'isolated_timing_groups': v_tim.get('groups', []),
                'summary_json': str((v_ws / 'm5' / 'summary.json').resolve()),
                'timing_json': str((v_ws / 'm5' / 'timing.json').resolve()),
            }
        variant_reports.append({
            'variant': v_name,
            'dense_checkpoint': str(d_ckpt.resolve()),
            'sparse_checkpoint': str(s_ckpt.resolve()),
            'offline_dense_kl': dense_kl,
            'offline_dense_sparse_kl': sparse_kl,
            'closed_loop': cl_summary,
        })

    report = {
        'suite': 'm2_refit_calibration',
        'scale': args.scale,
        'status': 'complete',
        'refit_hyperparameters': {
            'm5_fit_tag': fit_tag,
            'block_shuffle_window': args.block_shuffle_window,
            'weight_decay': args.weight_decay,
            'calibrate_beta': True,
            'fixed_checkpoint_scale': args.fixed_checkpoint_scale,
        },
        'refit_report_path': str((ws / f'{fit_subdir}.json').resolve()),
        'beta_calibration': {
            stage: refit_report.get('stages', {}).get(stage, {}).get('beta_calibration')
            for stage in ('dense', 'dense_sparse')
            if refit_report.get('stages', {}).get(stage, {}).get('beta_calibration') is not None
        } or None,
        'variants': variant_reports,
    }
    atomic_json(report_path, report)
    return report


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument(
        '--suite',
        choices=['all', 'fastpath_bench', 'online_head_to_head', 'm2_refit_calibration'],
        default='all',
    )
    parser.add_argument(
        '--artifacts_root',
        type=Path,
        default=None,
        help='read-only frozen M0-M4 artifacts directory (defaults to --output)',
    )
    parser.add_argument('--no_resume', dest='resume', action='store_false', default=True)
    parser.add_argument('--bigram_count_gate_kappa', type=float, default=12.0)
    parser.add_argument('--online_arms', nargs='+', default=list(DEFAULT_ONLINE_HEAD_TO_HEAD_ARMS))
    parser.add_argument('--block_shuffle_window', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--fixed_checkpoint_scale', type=float, default=0.20)
    parser.add_argument('--refit_tag', default='post_m4_shuf32_wd1e2')
    parser.add_argument('--bench_warmups', type=int, default=None)
    parser.add_argument('--bench_repeats', type=int, default=None)
    parser.add_argument('--timing_repeats', type=int, default=2)
    parser.add_argument('--timing_warmups', type=int, default=1)
    parser.add_argument('--timing_prompts', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--long_max_new_tokens', type=int, default=24576)
    parser.add_argument('--smoke_tokens', type=int, default=32)
    args = parse_config(parser, argv)

    tiny = args.scale == 'smoke'
    if args.bench_warmups is None:
        args.bench_warmups = 2 if tiny else 5
    if args.bench_repeats is None:
        args.bench_repeats = 5 if tiny else 25
    if not math.isfinite(args.bigram_count_gate_kappa) or args.bigram_count_gate_kappa <= 0:
        parser.error('--bigram_count_gate_kappa must be finite and > 0')
    if args.block_shuffle_window < 1:
        parser.error('--block_shuffle_window must be >= 1')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight_decay must be finite and >= 0')
    if not math.isfinite(args.fixed_checkpoint_scale) or args.fixed_checkpoint_scale <= 0:
        parser.error('--fixed_checkpoint_scale must be finite and > 0')
    if args.timing_repeats < 2 or args.timing_warmups < 1 or args.timing_prompts < 0:
        parser.error('timing requires repeats>=2, warmups>=1, prompts>=0')
    if args.bench_repeats < 2 or args.bench_warmups < 1:
        parser.error('bench requires repeats>=2 and warmups>=1')

    artifacts_root = Path(args.artifacts_root) if args.artifacts_root is not None else Path(args.output)
    suite_root = Path(args.output) / 'post_m4_suite'
    suite_root.mkdir(parents=True, exist_ok=True)
    record = record_run(args, f'post_m4_suite-{args.suite}')
    if not tiny:
        preflight(args, 4096)

    torch.set_num_threads(1)
    selected_suites = (
        ['fastpath_bench', 'online_head_to_head', 'm2_refit_calibration']
        if args.suite == 'all'
        else [args.suite]
    )
    results = {}
    if 'fastpath_bench' in selected_suites:
        results['fastpath_bench'] = run_fastpath_bench(args, artifacts_root, suite_root)
    if 'online_head_to_head' in selected_suites:
        results['online_head_to_head'] = run_online_head_to_head(args, artifacts_root, suite_root)
    if 'm2_refit_calibration' in selected_suites:
        results['m2_refit_calibration'] = run_m2_refit_calibration(args, artifacts_root, suite_root)

    summary = {
        'status': 'complete' if all(r.get('status') == 'complete' for r in results.values()) else 'incomplete',
        'scale': args.scale,
        'requested_suite': args.suite,
        'executed_suites': selected_suites,
        'project_git_commit': git_commit(),
        'implementation_hashes': {
            'm3': implementation_hash('m3'),
            'm4': implementation_hash('m4'),
            'm5': implementation_hash('m5'),
        },
        'source_provenance': source_provenance(),
        'resolved_config': resolved_config(args),
        'run_record': record,
        'suites': results,
    }
    summary_path = suite_root / 'post_m4_suite_summary.json'
    atomic_json(summary_path, summary)
    print(json.dumps({
        'status': summary['status'],
        'scale': summary['scale'],
        'executed_suites': selected_suites,
        'summary_path': str(summary_path),
    }))
    return summary


if __name__ == '__main__':
    main()
