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
       and single-session matched-bundle closed-loop acceptance/throughput
       comparison across uncalibrated, dev-calibrated, and `0.20`-scaled
       `dense` and `dense_sparse` checkpoints at fixed `beta = 1.0`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import torch

from .adapter import DenseAdapter
from .bench_adapter import assert_idle, assert_no_external_gpu_processes, measure
from .diagnose import OUTPUT_LABELS
from .eval_closed_loop import (
    EvaluationRunner,
    implementation_hash,
    sampling_conditions,
    select_prompts,
    smoke_manifest,
)
from .fastpath import FastPathConfig, FastPathRuntime
from .features import fit_hidden_projection
from .online import OnlineBigram
from .runtime import (
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
from .traces import read_shard
from .train import (
    epoch_references,
    evaluate as evaluate_offline,
    file_digest,
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


def _update_latest_ws_symlink(link_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        link_path.unlink()
    if not link_path.exists():
        link_path.symlink_to(target_dir.name)


def _prereq_identity(artifacts_root: Path, scale: str) -> dict:
    if scale == 'smoke':
        return {'scale': 'smoke'}
    root_resolved = artifacts_root.resolve()
    files = {}
    for rel in (
        'm2.json',
        'm1_audit.json',
        f'diagnostic_{scale}.json',
        'prompts.json',
        'h_proj.pt',
        'token_features.pt',
        'm2/dense.pt',
        'm2/dense_sparse.pt',
        'm3/summary.json',
        'm4/summary.json',
        'm4/settings.json',
    ):
        p = root_resolved / rel
        if p.is_file():
            files[rel] = file_digest(p)
    return {'scale': scale, 'artifacts_root': str(root_resolved), 'files': files}


def _suite_config_fingerprint(suite_name: str, args, artifacts_root: Path) -> str:
    payload: dict = {
        'suite': suite_name,
        'scale': args.scale,
        'seed': int(args.seed),
        'prereqs': _prereq_identity(artifacts_root, args.scale),
    }
    if args.scale != 'smoke':
        payload['device'] = str(args.device)
        payload['target'] = str(args.target)
        payload['drafter'] = str(args.drafter)
    if suite_name == 'fastpath_bench':
        payload.update({
            'bench_warmups': int(args.bench_warmups),
            'bench_repeats': int(args.bench_repeats),
            'bigram_count_gate_kappa': float(args.bigram_count_gate_kappa),
            'timing_warmups': int(args.timing_warmups),
            'timing_repeats': int(args.timing_repeats),
            'timing_prompts': int(args.timing_prompts),
            'max_new_tokens': int(args.max_new_tokens),
            'long_max_new_tokens': int(args.long_max_new_tokens),
            'smoke_tokens': int(args.smoke_tokens),
        })
    elif suite_name == 'online_head_to_head':
        payload.update({
            'online_arms': list(args.online_arms),
            'bigram_count_gate_kappa': float(args.bigram_count_gate_kappa),
            'timing_warmups': int(args.timing_warmups),
            'timing_repeats': int(args.timing_repeats),
            'timing_prompts': int(args.timing_prompts),
            'max_new_tokens': int(args.max_new_tokens),
            'long_max_new_tokens': int(args.long_max_new_tokens),
            'smoke_tokens': int(args.smoke_tokens),
        })
    elif suite_name == 'm2_refit_calibration':
        payload.update({
            'refit_tag': str(args.refit_tag),
            'block_shuffle_window': int(args.block_shuffle_window),
            'weight_decay': float(args.weight_decay),
            'fixed_checkpoint_scale': float(args.fixed_checkpoint_scale),
            'timing_warmups': int(args.timing_warmups),
            'timing_repeats': int(args.timing_repeats),
            'timing_prompts': int(args.timing_prompts),
            'max_new_tokens': int(args.max_new_tokens),
            'long_max_new_tokens': int(args.long_max_new_tokens),
            'smoke_tokens': int(args.smoke_tokens),
        })
    else:
        raise ValueError(f'Unknown suite_name: {suite_name}')
    return fingerprint(payload)


def _can_resume_report(report_path: Path, args, expected_fingerprint: str) -> dict | None:
    if not (args.resume and report_path.is_file()):
        return None
    existing = json.loads(report_path.read_text())
    if (
        existing.get('status') == 'complete'
        and existing.get('scale') == args.scale
        and existing.get('config_fingerprint') == expected_fingerprint
    ):
        return existing
    return None


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


def _select_test_prompts(prompt_manifest: dict, scale: str, timing_prompts: int) -> tuple[list[dict], list[dict], bool]:
    all_test = select_prompts(prompt_manifest, 'test', scale)
    selected_prompts = []
    limit = int(timing_prompts)
    for cond in sorted({p['condition'] for p in all_test}):
        c_prompts = [p for p in all_test if p['condition'] == cond]
        selected_prompts.extend(c_prompts if limit == 0 else c_prompts[:limit])
    is_pilot = len(selected_prompts) < len(all_test)
    return all_test, selected_prompts, is_pilot


def _run_matched_closed_loop_fastpath_bench(args, artifacts_root: Path, bench_dir: Path) -> dict:
    ws = bench_dir / 'closed_loop_workspace'
    if args.scale != 'smoke':
        _link_prereqs(
            artifacts_root,
            ws,
            (
                'm2.json', 'm2', 'm3', 'm4', 'prompts.json', 'h_proj.pt',
                'environment.json', 'smoke.json', f'diagnostic_{args.scale}.json',
            ),
        )

    prompt_manifest = (
        smoke_manifest()
        if args.scale == 'smoke'
        else json.loads((ws / 'prompts.json').read_text())
    )
    all_test, selected_prompts, is_pilot = _select_test_prompts(prompt_manifest, args.scale, args.timing_prompts)

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
        'status': 'pilot_complete' if is_pilot else 'complete',
        'is_pilot_subset': is_pilot,
        'selected_prompt_count': len(selected_prompts),
        'total_test_prompt_count': len(all_test),
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

    cfg_fp = _suite_config_fingerprint('fastpath_bench', args, artifacts_root)
    sub_dir = suite_root / f'fastpath_bench_{cfg_fp[:12]}'
    report_path = suite_root / 'fastpath_bench.json'
    cached = _can_resume_report(report_path, args, cfg_fp)
    if cached is not None:
        if sub_dir.is_dir():
            _update_latest_ws_symlink(suite_root / 'fastpath_bench', sub_dir)
        return cached

    caller_threads = torch.get_num_threads()
    device = torch.device(args.device)
    inventory = assert_idle(device)
    _update_latest_ws_symlink(suite_root / 'fastpath_bench', sub_dir)
    adapter_dir = sub_dir / 'adapter'
    grouped_dir = sub_dir / 'grouped'
    if args.scale != 'smoke':
        _link_prereqs(artifacts_root, adapter_dir, ('smoke.json', 'environment.json'))
        _link_prereqs(artifacts_root, grouped_dir, ('smoke.json', 'environment.json'))

    try:
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
        if args.scale != 'smoke':
            torch.set_num_threads(caller_threads)
        closed_loop = _run_matched_closed_loop_fastpath_bench(args, artifacts_root, sub_dir)
    finally:
        torch.set_num_threads(caller_threads)

    report = {
        'suite': 'fastpath_bench',
        'scale': args.scale,
        'status': closed_loop['status'],
        'config_fingerprint': cfg_fp,
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

    cfg_fp = _suite_config_fingerprint('online_head_to_head', args, artifacts_root)
    sub_ws = suite_root / f'online_head_to_head_ws_{cfg_fp[:12]}'
    report_path = suite_root / 'online_head_to_head.json'
    cached = _can_resume_report(report_path, args, cfg_fp)
    if cached is not None:
        if sub_ws.is_dir():
            _update_latest_ws_symlink(suite_root / 'online_head_to_head_workspace', sub_ws)
        return cached

    _update_latest_ws_symlink(suite_root / 'online_head_to_head_workspace', sub_ws)
    if args.scale != 'smoke':
        _link_prereqs(
            artifacts_root,
            sub_ws,
            (
                'm2.json', 'm2', 'm3', 'm4', 'prompts.json', 'h_proj.pt',
                'environment.json', 'smoke.json', f'diagnostic_{args.scale}.json',
            ),
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

    missing_reqs = summary.get('missing_requests')
    no_missing = isinstance(missing_reqs, list) and len(missing_reqs) == 0
    if summary.get('status') == 'complete' and timing.get('status') == 'complete':
        suite_status = 'complete'
    elif (
        no_missing
        and summary.get('greedy_output_comparisons_complete', False)
        and timing.get('status') in ('complete', 'pilot_complete')
    ):
        suite_status = 'pilot_complete'
    else:
        suite_status = 'incomplete'

    report = {
        'suite': 'online_head_to_head',
        'scale': args.scale,
        'status': suite_status,
        'config_fingerprint': cfg_fp,
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


def _evaluate_refit_variants_closed_loop(
    args,
    ws: Path,
    variants: list[tuple[str, Path, Path]],
) -> tuple[str, dict[str, dict]]:
    """Evaluate `plain_dflash` + all 5 `(dense, dense_sparse)` variants in a single matched session at fixed `beta = 1.0`."""
    device = torch.device(args.device)
    if args.scale != 'smoke':
        assert_idle(device)

    base_subdir = 'm2_smoke' if args.scale == 'smoke' else 'm2'
    runner_ns = SimpleNamespace(
        scale=args.scale,
        stage='m3',
        mode='timing',
        seed=args.seed,
        device=args.device,
        target=args.target,
        drafter=args.drafter,
        allow_download=getattr(args, 'allow_download', False),
        allow_large_run=getattr(args, 'allow_large_run', False),
        output=ws,
        prompts_manifest=ws / 'prompts.json',
        dense_checkpoint=ws / base_subdir / 'dense.pt',
        sparse_checkpoint=ws / base_subdir / 'dense_sparse.pt',
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
    if runner.tiny:
        cal = [[i % 8, (i + 3) % 8, (i * 3 + 1) % 8] for i in range(64)]
        runner.projection = fit_hidden_projection(
            [runner.drafter.draft(x, 4).h_raw for x in cal],
            12,
            6,
            seed=args.seed,
            prompt_ids=[f'synthetic-train-{i}' for i in range(64)],
        ).to(runner.device)

    loaded_variants = []
    for v_name, d_ckpt, s_ckpt in variants:
        d_ad, d_meta = load_adapter_checkpoint(d_ckpt, runner.device)
        s_ad, s_meta = load_adapter_checkpoint(s_ckpt, runner.device)
        if d_meta.get('stage') != 'dense' or s_meta.get('stage') != 'dense_sparse':
            raise ValueError(f'Variant {v_name} checkpoints must have stages dense and dense_sparse')
        if not runner.tiny:
            for meta in (d_meta, s_meta):
                rep = meta.get('report', {})
                if rep.get('model_pair') != runner.model_pair or rep.get('model_revisions') != runner.model_revisions:
                    raise ValueError(f'Variant {v_name} checkpoint model pair/revisions mismatch backend')
                if rep.get('frozen_artifacts', {}).get('hidden_projection_sha256') != runner.checkpoint_hashes['projection']:
                    raise ValueError(f'Variant {v_name} checkpoint hidden projection mismatch')
        d_ad.eval()
        s_ad.eval()
        loaded_variants.append((v_name, d_ad, d_meta, s_ad, s_meta))

    prompt_manifest = (
        smoke_manifest()
        if args.scale == 'smoke'
        else json.loads((ws / 'prompts.json').read_text())
    )
    all_test, selected_prompts, is_pilot = _select_test_prompts(prompt_manifest, args.scale, args.timing_prompts)

    # 11-arm matched bundle at fixed beta=1.0 (preserving each checkpoint's baked-in scale without dev re-tuning)
    bundle_arms: list[tuple[str, str, DenseAdapter | None, float | None]] = [
        ('plain_dflash', 'plain_dflash', None, None),
    ]
    for v_name, d_ad, d_meta, s_ad, s_meta in loaded_variants:
        bundle_arms.append((f'{v_name}::dense', 'dense', d_ad, float(d_meta.get('checkpoint_scale', 1.0))))
        bundle_arms.append((f'{v_name}::dense_sparse', 'dense_sparse', s_ad, float(s_meta.get('checkpoint_scale', 1.0))))

    orig_dense, orig_sparse = runner.dense, runner.dense_sparse

    def _invoke_bundle_arm(arm_kind: str, adapter_obj: DenseAdapter | None, base_job: dict) -> dict:
        job = dict(base_job, arm=arm_kind, force_setting_beta=True)
        setting = {} if arm_kind == 'plain_dflash' else {'beta': 1.0}
        try:
            if arm_kind == 'dense':
                assert adapter_obj is not None
                runner.dense = adapter_obj
            elif arm_kind == 'dense_sparse':
                assert adapter_obj is not None
                runner.dense_sparse = adapter_obj
            return runner.run(job, setting, profile=False)
        finally:
            runner.dense = orig_dense
            runner.dense_sparse = orig_sparse

    warmup_cap = min(64, args.smoke_tokens) if runner.tiny else 64
    for cond in sorted({p['condition'] for p in selected_prompts}):
        w_prompt = next(p for p in selected_prompts if p['condition'] == cond)
        w_name, w_temp, w_warp = sampling_conditions(cond, primary_only=True)[0]
        base_w_job = {
            'prompt': w_prompt,
            'sampling_name': w_name,
            'temperature': w_temp,
            'warper': w_warp,
            'seed': args.seed,
            'max_new_tokens_override': warmup_cap,
        }
        for _ in range(args.timing_warmups):
            for _, arm_kind, ad_obj, _ in bundle_arms:
                _invoke_bundle_arm(arm_kind, ad_obj, base_w_job)

    rows_by_key: dict[str, list[dict]] = {key: [] for key, _, _, _ in bundle_arms}
    for p_idx, prompt in enumerate(selected_prompts):
        s_name, temperature, warp = sampling_conditions(prompt['condition'], primary_only=True)[0]
        base_job = {
            'prompt': prompt,
            'sampling_name': s_name,
            'temperature': temperature,
            'warper': warp,
            'seed': args.seed,
        }
        for rep in range(args.timing_repeats):
            rot = (p_idx + rep) % len(bundle_arms)
            ordered = bundle_arms[rot:] + bundle_arms[:rot]
            for arm_key, arm_kind, ad_obj, _ in ordered:
                assert_no_external_gpu_processes(runner.device)
                row = _invoke_bundle_arm(arm_kind, ad_obj, base_job)
                rows_by_key[arm_key].append({
                    'prompt_id': prompt['prompt_id'],
                    'condition': prompt['condition'],
                    'sampling_name': s_name,
                    'repeat': rep,
                    'effective_beta': float(row['effective_beta']),
                    'generated_tokens': int(row['generated_tokens']),
                    'elapsed_seconds': float(row['elapsed_seconds']),
                    'tokens_per_second': float(row['tokens_per_second']),
                    'mean_accepted_length': float(row['mean_accepted_length']),
                    'median_accepted_length': float(row['median_accepted_length']),
                    'target_forwards_per_generated_token': float(row['target_forwards_per_generated_token']),
                })

    plain_rows = rows_by_key['plain_dflash']
    plain_by_slot = {(r['prompt_id'], r['repeat']): r for r in plain_rows}

    def _summarize_arm_rows(rows: list[dict], *, ckpt_scale: float | None = None, paired_vs_plain: bool = False) -> dict:
        by_cond = {}
        for cond in sorted({r['condition'] for r in rows}):
            c_rows = [r for r in rows if r['condition'] == cond]
            by_cond[cond] = {
                'requests': len(c_rows),
                'mean_accepted_length': statistics.mean(r['mean_accepted_length'] for r in c_rows),
                'mean_tokens_per_second': statistics.mean(r['tokens_per_second'] for r in c_rows),
                'median_tokens_per_second': statistics.median(r['tokens_per_second'] for r in c_rows),
                'mean_target_forwards_per_token': statistics.mean(
                    r['target_forwards_per_generated_token'] for r in c_rows
                ),
            }
        entry = {
            'requests': len(rows),
            'mean_accepted_length': statistics.mean(r['mean_accepted_length'] for r in rows) if rows else 0.0,
            'median_accepted_length': statistics.median(r['mean_accepted_length'] for r in rows) if rows else 0.0,
            'mean_tokens_per_second': statistics.mean(r['tokens_per_second'] for r in rows) if rows else 0.0,
            'median_tokens_per_second': statistics.median(r['tokens_per_second'] for r in rows) if rows else 0.0,
            'mean_target_forwards_per_token': (
                statistics.mean(r['target_forwards_per_generated_token'] for r in rows) if rows else 0.0
            ),
            'by_condition': by_cond,
        }
        if ckpt_scale is not None:
            entry['checkpoint_scale'] = ckpt_scale
            entry['effective_beta'] = 1.0
        if paired_vs_plain:
            tps_ratios = []
            acc_deltas = []
            for r in rows:
                p_ref = plain_by_slot.get((r['prompt_id'], r['repeat']))
                if p_ref is not None:
                    acc_deltas.append(r['mean_accepted_length'] - p_ref['mean_accepted_length'])
                    if p_ref['tokens_per_second'] > 0:
                        tps_ratios.append(r['tokens_per_second'] / p_ref['tokens_per_second'])
            entry['paired_tps_ratio_vs_plain'] = statistics.mean(tps_ratios) if tps_ratios else None
            entry['paired_accepted_length_delta_vs_plain'] = statistics.mean(acc_deltas) if acc_deltas else None
        return entry

    plain_summary = _summarize_arm_rows(plain_rows, ckpt_scale=None, paired_vs_plain=False)
    cl_status = 'pilot_complete' if is_pilot else 'complete'
    out_by_variant: dict[str, dict] = {}
    for v_name, _, d_meta, _, s_meta in loaded_variants:
        d_scale = float(d_meta.get('checkpoint_scale', 1.0))
        s_scale = float(s_meta.get('checkpoint_scale', 1.0))
        out_by_variant[v_name] = {
            'status': cl_status,
            'is_pilot_subset': is_pilot,
            'selected_prompt_count': len(selected_prompts),
            'total_test_prompt_count': len(all_test),
            'repeats': args.timing_repeats,
            'warmups': args.timing_warmups,
            'evaluation_beta': 1.0,
            'plain_dflash': plain_summary,
            'dense': _summarize_arm_rows(
                rows_by_key[f'{v_name}::dense'], ckpt_scale=d_scale, paired_vs_plain=True
            ),
            'dense_sparse': _summarize_arm_rows(
                rows_by_key[f'{v_name}::dense_sparse'], ckpt_scale=s_scale, paired_vs_plain=True
            ),
        }
    return cl_status, out_by_variant


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
        head, bias = load_frozen_head(head_metadata, device, output_dir=ws)
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


def run_m2_refit_calibration(args, artifacts_root: Path, suite_root: Path, *, cpu_threads: int | None = None) -> dict:
    from . import train

    cfg_fp = _suite_config_fingerprint('m2_refit_calibration', args, artifacts_root)
    ws = suite_root / f'm2_refit_ws_{cfg_fp[:12]}'
    report_path = suite_root / 'm2_refit_calibration.json'
    cached = _can_resume_report(report_path, args, cfg_fp)
    if cached is not None:
        if ws.is_dir():
            _update_latest_ws_symlink(suite_root / 'm2_refit_workspace', ws)
        return cached

    if cpu_threads is not None and args.scale != 'smoke':
        torch.set_num_threads(cpu_threads)

    _update_latest_ws_symlink(suite_root / 'm2_refit_workspace', ws)
    if args.scale == 'smoke':
        _ensure_smoke_baseline_m2(args, ws)
    else:
        _link_prereqs(
            artifacts_root,
            ws,
            (
                'traces', 'm1_audit.json', f'diagnostic_{args.scale}.json',
                'h_proj.pt', 'token_features.pt',
                'm2.json', 'm2', 'm3', 'm4', 'prompts.json',
                'environment.json', 'smoke.json',
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
            '--m1_diagnostic', str(ws / f'diagnostic_{args.scale}.json'),
            '--features', str(ws / 'token_features.pt'),
            '--projection', str(ws / 'h_proj.pt'),
            '--prompts', str(ws / 'prompts.json'),
        ])
    refit_report = train.main(train_cli)
    if refit_report is None:
        refit_report = json.loads((ws / f'{fit_subdir}.json').read_text())

    if cpu_threads is not None and args.scale != 'smoke':
        torch.set_num_threads(cpu_threads)

    refit_dir = ws / fit_subdir
    base_dir = ws / base_subdir
    scale_tag = f'{args.fixed_checkpoint_scale:.2f}'.replace('.', 'p')
    refit_dense_s = refit_dir / f'dense_scale{scale_tag}.pt'
    refit_sparse_s = refit_dir / f'dense_sparse_scale{scale_tag}.pt'
    scale_adapter_checkpoint(refit_dir / 'dense.pt', refit_dense_s, args.fixed_checkpoint_scale)
    scale_adapter_checkpoint(refit_dir / 'dense_sparse.pt', refit_sparse_s, args.fixed_checkpoint_scale)

    base_scaled_dir = ws / 'baseline_m2_scaled'
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
    offline_by_variant = {}
    for v_name, d_ckpt, s_ckpt in variants:
        offline_by_variant[v_name] = (
            _offline_kl_summary(eval_ctx, d_ckpt, device),
            _offline_kl_summary(eval_ctx, s_ckpt, device),
        )

    cl_status, cl_by_variant = _evaluate_refit_variants_closed_loop(args, ws, variants)
    variant_reports = []
    for v_name, d_ckpt, s_ckpt in variants:
        dense_kl, sparse_kl = offline_by_variant[v_name]
        variant_reports.append({
            'variant': v_name,
            'dense_checkpoint': str(d_ckpt.resolve()),
            'sparse_checkpoint': str(s_ckpt.resolve()),
            'offline_dense_kl': dense_kl,
            'offline_dense_sparse_kl': sparse_kl,
            'closed_loop': cl_by_variant[v_name],
        })

    report = {
        'suite': 'm2_refit_calibration',
        'scale': args.scale,
        'status': cl_status,
        'config_fingerprint': cfg_fp,
        'refit_fit_gate_status': refit_report.get('status'),
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
    if not tiny:
        _link_prereqs(artifacts_root, Path(args.output), ('smoke.json', 'environment.json'))
        _link_prereqs(artifacts_root, suite_root, ('smoke.json', 'environment.json'))
    record = record_run(args, f'post_m4_suite-{args.suite}')
    if not tiny:
        preflight(SimpleNamespace(scale=args.scale, output=suite_root), 4096)

    orig_threads = torch.get_num_threads()
    try:
        if tiny:
            torch.set_num_threads(1)
        selected_suites = (
            ['fastpath_bench', 'online_head_to_head', 'm2_refit_calibration']
            if args.suite == 'all'
            else [args.suite]
        )
        results = {}
        if 'fastpath_bench' in selected_suites:
            results['fastpath_bench'] = run_fastpath_bench(args, artifacts_root, suite_root)
            if not tiny:
                torch.set_num_threads(orig_threads)
        if 'online_head_to_head' in selected_suites:
            results['online_head_to_head'] = run_online_head_to_head(args, artifacts_root, suite_root)
            if not tiny:
                torch.set_num_threads(orig_threads)
        if 'm2_refit_calibration' in selected_suites:
            results['m2_refit_calibration'] = run_m2_refit_calibration(
                args, artifacts_root, suite_root, cpu_threads=orig_threads,
            )
            if not tiny:
                torch.set_num_threads(orig_threads)
    finally:
        torch.set_num_threads(orig_threads)

    statuses = [r.get('status') for r in results.values()]
    if statuses and all(s == 'complete' for s in statuses):
        master_status = 'complete'
    elif statuses and all(s in ('complete', 'pilot_complete') for s in statuses):
        master_status = 'pilot_complete'
    else:
        master_status = 'incomplete'

    summary = {
        'status': master_status,
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
