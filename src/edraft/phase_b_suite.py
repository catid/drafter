"""Reproducible 8-GPU NUMA/core-pinned Phase-B ablation & full-scale runner (`python -m edraft.phase_b_suite`).

Executes all remaining `prompt.md` secondary `--scale reference` ablations plus
the complete `--scale full --allow_large_run` pipeline (`M1-pre -> M1a -> M1b ->
M2 -> M3 -> M4`) inside an isolated output tree (default `artifacts/full_scale`)
so all frozen `--scale reference` M0-M4 and Phase-A artifacts under `artifacts/`
remain 100% byte-immutable (verified by SHA256 before and after every suite):

  1. `bigram_eviction` (`--scale reference` secondary ablation, `prompt.md` §9 & §10 M3):
     - Core-pinned sharded closed-loop dev tuning, evaluation, sharded timing,
       and 1-GPU isolation calibration comparing `plain_dflash`, exact un-evicted
       `online_bigram` (`k_ret=None`), top-k evicted `online_bigram_kret256`
       (`k_ret=256`, configurable via `--bigram_k_ret`), `online_bigram_kret128`
       (`k_ret=128`), and `online_bigram_count_gated` (`kappa=12`),
     - Reports acceptance, throughput, per-offset (`0-1K`..`16K+`) slices, and
       per-request bigram memory/eviction telemetry (`exact_ledger_pairs`,
       `retained_prior_pairs`, `evicted_prior_pairs`, payload byte estimates).

  2. `full_vocab_lookahead` (`--scale reference` secondary ablation, `prompt.md` §7 & §10 M5):
     - Matched-support closed-loop comparison pairing `full_vocabulary` lookahead
       (`lookahead_static_full`, `lookahead_rerelax_full`) against causal
       full-vocabulary `dense_full`, alongside `candidate_masked` lookahead
       (`lookahead_static_candidate`, `lookahead_rerelax_candidate`) against
       causal `dense_candidate` with identical frozen `C0[j]` `beta_relax`.

  3. `pure_t1_fit` (`--scale reference` secondary ablation, `prompt.md` §10 M2):
     - 8-GPU DDP `--allow_pure_t1_ablation` offline fit on pure `T_collection=1.0`
       unwarped `standard_nonthinking` traces (`--block_shuffle_window 32
       --weight_decay 1e-2 --calibrate_beta`),
     - Cross-distribution held-out offline KL evaluation on BOTH `pure_T=1.0`
       (`pure_t1_eligible`) and primary (`primary_eligible`, including exact
       `live_full` KL/overlap/top-1) dev/test shards, plus core-pinned sharded
       closed-loop evaluation against `plain_dflash` and primary M2 variants.

  4. `full_scale` (`--scale full --allow_large_run`, `prompt.md` §0 & §10 M1-M4):
     - Isolated `artifacts/full_scale` bootstrap (`environment.json`, `smoke.json`,
       verified M0 gate provenance, and `prompts.json` capacity verification:
       `>=7,409` GSM8K train, `128` selected / `164` held-out HumanEval test,
       `32` long dev + `96` long test MATH-500),
     - 1-GPU M1-pre `calibrate` -> 8-GPU dynamic M1a `collect` (`2-4M` primary
       train tokens + `32` long dev + `96` long test + `128` std test + `T=1.0`
       diagnostic) -> M1a `audit` + 16-worker M1b `diagnose` -> 8-GPU DDP M2
       `train` -> 8-GPU sharded M3 (`tune`, `tune_summarize`, `evaluate`, 1-GPU
       `timing_calibration`, 8-GPU `timing`, `summarize`) -> 8-GPU sharded M4
       (all 9 full-scale M4 arms including `offline_online_stride5`,
       `zero_offline_online_only`, `online_rejection_relative`, `online_cutoff`).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import torch

from .adapter import DenseAdapter
from .bench_adapter import assert_idle, assert_no_external_gpu_processes
from .diagnose import OUTPUT_LABELS
from .eval_closed_loop import (
    EvaluationRunner,
    M3_ARMS,
    M4_ARMS,
    apply_gpu_cpu_affinity,
    arms_for,
    candidate_settings,
    checkpoint_hashes as eval_checkpoint_hashes,
    file_sha256,
    gpu_cpu_affinity,
    implementation_hash,
    sampling_conditions,
    select_prompts,
    smoke_manifest,
    validate_frozen_settings,
)
from .features import fit_hidden_projection
from .runtime import (
    ROOT,
    Manifest,
    atomic_json,
    common_parser,
    environment as write_environment,
    fingerprint,
    git_commit,
    parse_config,
    preflight,
    record_run,
    resolved_config,
    source_provenance,
)
from .sparse import ExactObservedPairs
from .traces import read_shard
from .train import (
    epoch_references,
    evaluate as evaluate_offline,
    load_adapter_checkpoint,
    load_frozen_head,
    primary_eligible,
    pure_t1_eligible,
    scale_adapter_checkpoint,
    smoke_bundle,
    trace_index,
)

DEFAULT_BIGRAM_EVICTION_ARMS = (
    'plain_dflash',
    'online_bigram',
    'online_bigram_kret256',
    'online_bigram_kret128',
    'online_bigram_count_gated',
)

DEFAULT_FULL_VOCAB_LOOKAHEAD_ARMS = (
    'plain_dflash',
    'dense_full',
    'lookahead_static_full',
    'lookahead_rerelax_full',
    'dense_candidate',
    'lookahead_static_candidate',
    'lookahead_rerelax_candidate',
)

REF_SECONDARY_SUITES = ('bigram_eviction', 'full_vocab_lookahead', 'pure_t1_fit')
ALL_SUITES = (*REF_SECONDARY_SUITES, 'full_scale')

UPSTREAM_REFERENCE_SENTINELS = (
    'milestones.json',
    'm0.json',
    'm0_numerics.json',
    'calibration.json',
    'h_proj.pt',
    'token_features.pt',
    'prompts.json',
    'm1_audit.json',
    'diagnostic_reference.json',
    'm2.json',
    'm2/dense.pt',
    'm2/dense_sparse.pt',
    'm3/settings.json',
    'm3/summary.json',
    'm3/timing.json',
    'm3/greedy_output_comparisons.json',
    'm4/settings.json',
    'm4/summary.json',
    'm4/timing.json',
    'm4/timing_isolation_calibration.json',
    'm4/feature_context.json',
    'm4/greedy_output_comparisons.json',
    'm3_validation_reference.json',
    'm4_validation_reference.json',
    'reference_continuation_state.json',
)


def _is_cuda_run(args) -> bool:
    return args.scale != 'smoke' and str(args.device).startswith('cuda')


def _resolve_gpus(args) -> list[int]:
    raw = getattr(args, 'gpus', None)
    if raw is not None:
        gpus = [int(tok.strip()) for tok in str(raw).split(',') if tok.strip()]
        if not gpus or len(gpus) != len(set(gpus)) or any(g < 0 for g in gpus):
            raise ValueError('--gpus requires distinct non-negative GPU indices')
        if len(gpus) > 8:
            raise ValueError('--gpus supports at most 8 workers (0..7)')
        return gpus
    if _is_cuda_run(args):
        return list(range(8))
    return [0]


def _default_stats_memory_limit(args, effective_scale: str) -> str:
    explicit = getattr(args, 'stats_memory_limit', None)
    if explicit:
        return str(explicit)
    return '512MB' if effective_scale == 'smoke' else '32GB'


def _snapshot_upstream_hashes(reference_artifacts: Path) -> dict[str, str]:
    ref_root = Path(reference_artifacts)
    hashes: dict[str, str] = {}
    if not ref_root.is_dir():
        return hashes
    for rel in UPSTREAM_REFERENCE_SENTINELS:
        p = ref_root / rel
        if p.is_file():
            hashes[rel] = file_sha256(p)
    return hashes


def _verify_upstream_immutability(reference_artifacts: Path, before_hashes: dict[str, str]) -> dict:
    after_hashes = _snapshot_upstream_hashes(reference_artifacts)
    modified = []
    for rel, sha_before in before_hashes.items():
        sha_after = after_hashes.get(rel)
        if sha_after != sha_before:
            modified.append({'file': rel, 'before': sha_before, 'after': sha_after})
    if modified:
        raise RuntimeError(f'Upstream reference artifacts were mutated during Phase B: {modified}')
    return {
        'verified_files': len(before_hashes),
        'status': 'intact',
        'reference_artifacts': str(Path(reference_artifacts).resolve()),
    }


def _link_prereqs(src_root: Path, dst_ws: Path, names: tuple[str, ...] | list[str]) -> None:
    dst_ws.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = (src_root / name).resolve()
        dst = dst_ws / name
        if not src.exists():
            continue
        if dst.is_symlink():
            if dst.resolve() == src:
                continue
            dst.unlink()
        elif dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def _update_latest_ws_symlink(link_path: Path, target_ws: Path) -> None:
    target_ws.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.is_file():
        if link_path.is_symlink() and link_path.resolve() == target_ws.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        return
    link_path.symlink_to(target_ws.name)


def _atomic_copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f'{dst.name}.{os.getpid()}.tmp')
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return file_sha256(dst)


def _worker_pythonpath() -> str:
    src_path = str((ROOT / 'src').resolve())
    existing = os.environ.get('PYTHONPATH', '')
    return f'{src_path}{os.pathsep}{existing}' if existing else src_path


def _spawn_parallel_workers(
    worker_specs: list[tuple[int, int, list[str]]],
    logdir: Path,
    label: str,
    *,
    scale: str,
    device_str: str,
    omp_threads: int = 1,
) -> dict:
    """Spawn rank-pinned subprocesses cleanly, reaping any remaining children on failure."""
    logdir.mkdir(parents=True, exist_ok=True)
    use_cuda = (scale != 'smoke' and str(device_str).startswith('cuda'))
    if use_cuda:
        from .launch import selected_gpus_idle
        selected_gpus_idle([gpu for _, gpu, _ in worker_specs])

    processes: list[subprocess.Popen] = []
    streams = []
    py_path = _worker_pythonpath()
    state = {
        'label': label,
        'scale': scale,
        'started_at_unix': time.time(),
        'workers': [],
    }
    state_path = logdir / f'{label}-processes.json'
    t0 = time.perf_counter()
    try:
        for rank, gpu, cmd in worker_specs:
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu) if use_cuda else '',
                RANK=str(rank),
                WORLD_SIZE=str(len(worker_specs)),
                PYTHONPATH=py_path,
                PYTHONDONTWRITEBYTECODE='1',
                HF_HUB_OFFLINE='1',
                HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS=str(omp_threads),
                NCCL_NVLS_ENABLE=os.environ.get('NCCL_NVLS_ENABLE', '0'),
            )
            for stale_key in ('LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(stale_key, None)
            log_path = logdir / f'{label}-rank{rank}.log'
            stream = log_path.open('a', buffering=1)
            streams.append(stream)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(proc)
            aff = gpu_cpu_affinity(gpu)
            state['workers'].append({
                'rank': rank,
                'physical_gpu': gpu,
                'numa_node': aff['numa_node'],
                'cpu_affinity': aff['cpu_affinity'],
                'pid': proc.pid,
                'command': cmd,
                'log': str(log_path),
                'status': 'running',
            })
        atomic_json(state_path, state)

        poll_sleep = 0.15 if scale == 'smoke' else 2.0
        while any(p.poll() is None for p in processes):
            for row, p in zip(state['workers'], processes):
                rc = p.poll()
                if rc is not None:
                    row.update({'status': 'complete' if rc == 0 else 'failed', 'returncode': rc})
            atomic_json(state_path, state)
            if any(p.returncode not in (None, 0) for p in processes):
                bad_logs = []
                for row, p in zip(state['workers'], processes):
                    if p.returncode not in (None, 0):
                        tail = Path(row['log']).read_text()[-1500:] if Path(row['log']).is_file() else ''
                        bad_logs.append(f"rank {row['rank']} (rc={p.returncode}):\n{tail}")
                raise RuntimeError(f'Parallel worker failed in {label}:\n' + '\n'.join(bad_logs))
            time.sleep(poll_sleep)

        wall_elapsed = time.perf_counter() - t0
        for row, p in zip(state['workers'], processes):
            row.update({'status': 'complete' if p.returncode == 0 else 'failed', 'returncode': p.returncode})
        state['completed_at_unix'] = time.time()
        state['wall_elapsed_seconds'] = wall_elapsed
        atomic_json(state_path, state)
        return state
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        for s in streams:
            s.close()


def _fastpath_config_and_flags(args) -> tuple[dict | None, list[str]]:
    from .fastpath import FastPathConfig
    want_fast_decode = bool(
        getattr(args, 'fast_decode', False) or os.environ.get('EDRAFT_FAST_DECODE', '0') == '1'
    )
    fp = FastPathConfig(
        compile=bool(getattr(args, 'compile', False)),
        cuda_graph=bool(getattr(args, 'cuda_graph', False)),
        bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
        compile_drafter=bool(getattr(args, 'compile_drafter', False)),
        compile_online=bool(getattr(args, 'compile_online', False)),
        analytic_online_grad=bool(getattr(args, 'analytic_online_grad', True)),
        fast_decode=want_fast_decode,
    )
    flags: list[str] = []
    if fp.compile:
        flags.append('--compile')
    if fp.cuda_graph:
        flags.append('--cuda_graph')
    if fp.bf16_gemm:
        flags.append('--bf16_gemm')
    if fp.compile_drafter:
        flags.append('--compile_drafter')
    if fp.compile_online:
        flags.append('--compile_online')
    if fp.fast_decode:
        flags.append('--fast_decode')
    if not fp.analytic_online_grad:
        flags.append('--no_analytic_online_grad')
    return (fp.to_dict() if fp.enabled else None), flags


def _cli_base_flags(
    args,
    output_dir: Path,
    *,
    scale_override: str | None = None,
    device_override: str | None = None,
) -> list[str]:
    eff_scale = scale_override or args.scale
    eff_dev = 'cpu' if eff_scale == 'smoke' else (device_override or str(args.device))
    flags = [
        '--scale', eff_scale,
        '--seed', str(args.seed),
        '--device', eff_dev,
        '--target', str(args.target),
        '--drafter', str(args.drafter),
        '--output', str(output_dir),
    ]
    if eff_scale == 'full' or (getattr(args, 'allow_large_run', False) and eff_scale != 'smoke'):
        flags.append('--allow_large_run')
    if getattr(args, 'allow_download', False) and eff_scale != 'smoke':
        flags.append('--allow_download')
    return flags


def _find_phase_a_calibrated_refit(reference_artifacts: Path) -> tuple[Path, Path] | None:
    """Locate Phase-A calibrated refit `(dense_calibrated.pt, dense_sparse_calibrated.pt)` if present."""
    refit_candidates = [
        reference_artifacts / 'post_m4_suite' / 'm2_refit_workspace' / 'post_m4_shuf32_wd1e2',
    ]
    post_m4_dir = reference_artifacts / 'post_m4_suite'
    if post_m4_dir.is_dir():
        for cand_ws in sorted(post_m4_dir.glob('m2_refit_ws_*')):
            refit_candidates.append(cand_ws / 'post_m4_shuf32_wd1e2')
    for cand in refit_candidates:
        d_cal = cand / 'dense_calibrated.pt'
        s_cal = cand / 'dense_sparse_calibrated.pt'
        if d_cal.is_file() and s_cal.is_file():
            return d_cal, s_cal
    return None


def _resolve_lookahead_checkpoints(args, reference_artifacts: Path, sub_ws: Path, eff_scale: str) -> dict:
    """Resolve calibrated or scaled offline checkpoints for `full_vocab_lookahead`."""
    if eff_scale == 'smoke':
        return {
            'source': 'smoke_synthetic',
            'dense_checkpoint': sub_ws / 'm2_smoke' / 'dense.pt',
            'sparse_checkpoint': sub_ws / 'm2_smoke' / 'dense_sparse.pt',
            'checkpoint_scale': None,
        }
    if getattr(args, 'lookahead_dense_checkpoint', None) and getattr(args, 'lookahead_sparse_checkpoint', None):
        d_p = Path(args.lookahead_dense_checkpoint).resolve()
        s_p = Path(args.lookahead_sparse_checkpoint).resolve()
        return {
            'source': 'explicit_cli_override',
            'dense_checkpoint': d_p,
            'sparse_checkpoint': s_p,
            'checkpoint_scale': getattr(args, 'lookahead_checkpoint_scale', None),
        }
    phase_a_pair = _find_phase_a_calibrated_refit(reference_artifacts)
    if phase_a_pair is not None:
        return {
            'source': 'phase_a_calibrated_refit',
            'dense_checkpoint': phase_a_pair[0],
            'sparse_checkpoint': phase_a_pair[1],
            'checkpoint_scale': getattr(args, 'lookahead_checkpoint_scale', None),
        }
    fallback_scale = (
        float(args.lookahead_checkpoint_scale)
        if getattr(args, 'lookahead_checkpoint_scale', None) is not None
        else float(args.fixed_checkpoint_scale)
    )
    return {
        'source': 'm2_scaled_fallback',
        'dense_checkpoint': sub_ws / 'm2' / 'dense.pt',
        'sparse_checkpoint': sub_ws / 'm2' / 'dense_sparse.pt',
        'checkpoint_scale': fallback_scale,
    }


def _prereq_identity(reference_artifacts: Path, eff_scale: str) -> dict:
    if eff_scale == 'smoke':
        return {'mode': 'smoke_synthetic'}
    files = {}
    for rel in (
        'prompts.json', 'h_proj.pt', 'token_features.pt',
        'm1_audit.json', f'diagnostic_{eff_scale}.json',
        'm2.json', 'm2/dense.pt', 'm2/dense_sparse.pt',
    ):
        p = reference_artifacts / rel
        files[rel] = file_sha256(p) if p.is_file() else None
    return files


def _suite_config_fingerprint(suite_name: str, args, reference_artifacts: Path) -> str:
    gpus = _resolve_gpus(args)
    if suite_name in REF_SECONDARY_SUITES:
        eff_scale = 'smoke' if args.scale == 'smoke' else 'reference'
    else:
        eff_scale = 'smoke' if args.scale == 'smoke' else 'full'

    payload: dict = {
        'schema_version': 1,
        'suite': suite_name,
        'effective_scale': eff_scale,
        'seed': int(args.seed),
        'device_kind': 'cuda' if (eff_scale != 'smoke' and str(args.device).startswith('cuda')) else 'cpu',
        'gpus': gpus,
        'target': str(args.target),
        'drafter': str(args.drafter),
        'max_new_tokens': int(args.max_new_tokens),
        'long_max_new_tokens': int(args.long_max_new_tokens),
        'smoke_tokens': int(args.smoke_tokens),
        'timing_repeats': int(args.timing_repeats),
        'timing_warmups': int(args.timing_warmups),
        'timing_prompts': int(args.timing_prompts),
    }
    fp_dict, _ = _fastpath_config_and_flags(args)
    if fp_dict is not None:
        payload['fastpath'] = fp_dict
    if suite_name in REF_SECONDARY_SUITES:
        payload['prereqs'] = _prereq_identity(reference_artifacts, eff_scale)
    if suite_name == 'bigram_eviction':
        payload.update({
            'arms': list(args.bigram_eviction_arms),
            'bigram_k_ret': int(args.bigram_k_ret),
            'bigram_count_gate_kappa': float(args.bigram_count_gate_kappa),
        })
    elif suite_name == 'full_vocab_lookahead':
        la_ckpts = _resolve_lookahead_checkpoints(args, reference_artifacts, reference_artifacts, eff_scale)
        payload.update({
            'arms': list(args.full_vocab_lookahead_arms),
            'checkpoint_source': la_ckpts['source'],
            'dense_sha256': file_sha256(la_ckpts['dense_checkpoint']) if eff_scale != 'smoke' and la_ckpts['dense_checkpoint'].is_file() else None,
            'sparse_sha256': file_sha256(la_ckpts['sparse_checkpoint']) if eff_scale != 'smoke' and la_ckpts['sparse_checkpoint'].is_file() else None,
            'checkpoint_scale': la_ckpts['checkpoint_scale'],
            'm5_beta_relax': float(args.m5_beta_relax),
            'm5_beta_candidates': [float(b) for b in args.m5_beta_candidates],
        })
    elif suite_name == 'pure_t1_fit':
        phase_a_pair = _find_phase_a_calibrated_refit(reference_artifacts) if eff_scale != 'smoke' else None
        payload.update({
            'pure_t1_tag': str(args.pure_t1_tag),
            'block_shuffle_window': int(args.block_shuffle_window),
            'weight_decay': float(args.weight_decay),
            'fixed_checkpoint_scale': float(args.fixed_checkpoint_scale),
            'phase_a_refit_sha256': (
                [file_sha256(phase_a_pair[0]), file_sha256(phase_a_pair[1])]
                if phase_a_pair is not None
                else None
            ),
        })
    elif suite_name == 'full_scale':
        payload.update({
            'train_tokens_per_temperature': int(args.train_tokens_per_temperature) if args.train_tokens_per_temperature is not None else None,
            'standard_max_new_tokens': int(args.standard_max_new_tokens),
            'diagnose_workers': int(args.diagnose_workers),
            'full_m2_calibrated_fit': bool(args.full_m2_calibrated_fit),
            'block_shuffle_window': int(args.block_shuffle_window) if args.full_m2_calibrated_fit else 0,
            'weight_decay': float(args.weight_decay) if args.full_m2_calibrated_fit else 1e-4,
        })
    return fingerprint(payload)


def _can_resume_report(report_path: Path, args, expected_fp: str) -> dict | None:
    if not args.resume or not report_path.is_file():
        return None
    try:
        existing = json.loads(report_path.read_text())
    except (OSError, ValueError):
        return None
    allowed_statuses = ('complete', 'pilot_complete') if int(args.timing_prompts) > 0 else ('complete',)
    if existing.get('config_fingerprint') == expected_fp and existing.get('status') in allowed_statuses:
        return existing
    return None


def _select_test_prompts(prompt_manifest: dict, scale: str, timing_prompts: int) -> tuple[list[dict], list[dict], bool]:
    prompts = select_prompts(prompt_manifest, 'test', scale)
    ordered = [
        p for cond in sorted({row['condition'] for row in prompts})
        for p in prompts if p['condition'] == cond
    ]
    if timing_prompts and timing_prompts > 0:
        selected = []
        for cond in sorted({row['condition'] for row in ordered}):
            cond_rows = [p for p in ordered if p['condition'] == cond]
            selected.extend(cond_rows[:timing_prompts])
        is_pilot = len(selected) < len(ordered)
        return ordered, selected, is_pilot
    return ordered, ordered, False


def _snake_rank(index: int, world_size: int) -> int:
    band, offset = divmod(int(index), int(world_size))
    return offset if (band % 2 == 0) else (int(world_size) - 1 - offset)


def _snake_shard_prompts(
    selected_prompts: list[dict],
    prompt_manifest: dict,
    eval_ws: Path,
    world_size: int,
) -> tuple[int, dict[int, list[dict]]]:
    eff_w = max(1, min(int(world_size), len(selected_prompts)))
    manifest_order = {p['prompt_id']: idx for idx, p in enumerate(prompt_manifest.get('prompts', []))}
    conds = sorted({p['condition'] for p in selected_prompts})
    ordered_by_cond = {
        cond: sorted(
            [p for p in selected_prompts if p['condition'] == cond],
            key=lambda p: (manifest_order.get(p['prompt_id'], 0), p['prompt_id']),
        )
        for cond in conds
    }
    prompt_to_rank: dict[str, int] = {}
    if all(len(v) >= eff_w for v in ordered_by_cond.values()):
        for cond in conds:
            for k, p in enumerate(ordered_by_cond[cond]):
                prompt_to_rank[p['prompt_id']] = _snake_rank(k, eff_w)
    else:
        flat = [p for cond in conds for p in ordered_by_cond[cond]]
        for k, p in enumerate(flat):
            prompt_to_rank[p['prompt_id']] = _snake_rank(k, eff_w)
    shards = {
        r: [p for p in selected_prompts if prompt_to_rank[p['prompt_id']] == r]
        for r in range(eff_w)
    }
    return eff_w, shards


def _stage_settings_already_valid(
    args,
    sub_ws: Path,
    stage: str,
    eff_scale: str,
    prompt_manifest: dict,
    *,
    dense_ckpt: Path,
    sparse_ckpt: Path,
    proj_ckpt: Path,
    m5_arms: list[str] | None = None,
    extra_flags: list[str] | None = None,
) -> bool:
    if not getattr(args, 'resume', True):
        return False
    settings_path = sub_ws / stage / 'settings.json'
    if not settings_path.is_file():
        return False
    try:
        existing = json.loads(settings_path.read_text())
        extra = list(extra_flags or [])
        kappa = None
        k_ret = None
        ckpt_scale = None
        m5_betas = None
        m5_br = None
        idx = 0
        while idx < len(extra):
            tok = extra[idx]
            if tok == '--bigram_count_gate_kappa' and idx + 1 < len(extra):
                kappa = float(extra[idx + 1])
                idx += 2
            elif tok == '--bigram_k_ret' and idx + 1 < len(extra):
                k_ret = int(extra[idx + 1])
                idx += 2
            elif tok == '--checkpoint_scale' and idx + 1 < len(extra):
                ckpt_scale = float(extra[idx + 1])
                idx += 2
            elif tok == '--m5_beta_candidates' and idx + 2 < len(extra):
                m5_betas = (float(extra[idx + 1]), float(extra[idx + 2]))
                idx += 3
            elif tok == '--m5_beta_relax' and idx + 1 < len(extra):
                m5_br = float(extra[idx + 1])
                idx += 2
            else:
                idx += 1
        if eff_scale == 'smoke':
            curr_hashes = dict(existing.get('checkpoint_hashes', {}), implementation=implementation_hash(stage))
            if stage == 'm5':
                if ckpt_scale is not None:
                    curr_hashes['checkpoint_scale'] = ckpt_scale
                else:
                    curr_hashes.pop('checkpoint_scale', None)
            model_pair = existing.get('model_pair')
        else:
            ns = SimpleNamespace(
                stage=stage,
                dense_checkpoint=dense_ckpt,
                sparse_checkpoint=sparse_ckpt,
                projection=proj_ckpt,
                m5_oracle_checkpoint=None,
                checkpoint_scale=ckpt_scale if stage == 'm5' else None,
            )
            curr_hashes = eval_checkpoint_hashes(ns)
            model_pair = [str(args.target), str(args.drafter)]
        caps = {
            'standard': int(args.max_new_tokens),
            'long': int(args.long_max_new_tokens),
            'smoke': int(args.smoke_tokens),
        }
        validate_frozen_settings(
            existing, stage, eff_scale, curr_hashes, prompt_manifest, caps,
            model_pair=model_pair, m5_arms=m5_arms if stage == 'm5' else None,
            include_long_oracle=False,
        )
        cands_map = existing.get('candidates', {})
        for arm in existing['selected']:
            expected_grid = candidate_settings(
                arm,
                bigram_count_gate_kappa=kappa if stage == 'm5' else None,
                bigram_k_ret=k_ret if stage == 'm5' else None,
                m5_beta_candidates=m5_betas if stage == 'm5' else None,
                m5_beta_relax=m5_br if stage == 'm5' else None,
            )
            if [r.get('settings') for r in cands_map.get(arm, [])] != expected_grid:
                return False
        return True
    except Exception:
        return False


def _execute_closed_loop_stage(
    args,
    sub_ws: Path,
    stage: str,
    eff_scale: str,
    gpus: list[int],
    *,
    dense_ckpt: Path | None = None,
    sparse_ckpt: Path | None = None,
    m5_arms: list[str] | None = None,
    extra_stage_flags: list[str] | None = None,
    run_1gpu_calibration: bool = True,
) -> tuple[dict, dict, dict]:
    """Run resumable `<stage>` `tune -> tune_summarize -> evaluate -> [timing_calibration] -> timing -> summarize`."""
    from . import eval_closed_loop

    use_cuda = (eff_scale != 'smoke' and str(args.device).startswith('cuda'))
    d_ckpt = dense_ckpt or (sub_ws / 'm2' / 'dense.pt')
    s_ckpt = sparse_ckpt or (sub_ws / 'm2' / 'dense_sparse.pt')
    p_ckpt = sub_ws / 'h_proj.pt'
    p_manifest_path = sub_ws / 'prompts.json'
    extra = list(extra_stage_flags or [])

    shared_flags = [
        '--stage', stage,
        '--timing_repeats', str(args.timing_repeats),
        '--timing_warmups', str(args.timing_warmups),
        '--timing_prompts', str(args.timing_prompts),
        '--max_new_tokens', str(args.max_new_tokens),
        '--long_max_new_tokens', str(args.long_max_new_tokens),
        '--smoke_tokens', str(args.smoke_tokens),
        '--prompts_manifest', str(p_manifest_path),
        '--dense_checkpoint', str(d_ckpt),
        '--sparse_checkpoint', str(s_ckpt),
        '--projection', str(p_ckpt),
    ]
    if stage == 'm5' and m5_arms:
        shared_flags.extend(['--m5_arms', *m5_arms])
    _, fp_flags = _fastpath_config_and_flags(args)
    shared_flags.extend(fp_flags)
    shared_flags.extend(extra)

    prompt_manifest = (
        smoke_manifest()
        if eff_scale == 'smoke'
        else json.loads(p_manifest_path.read_text())
    )

    if not use_cuda and len(gpus) == 1:
        if _stage_settings_already_valid(
            args, sub_ws, stage, eff_scale, prompt_manifest,
            dense_ckpt=d_ckpt, sparse_ckpt=s_ckpt, proj_ckpt=p_ckpt,
            m5_arms=m5_arms, extra_flags=extra,
        ):
            eval_closed_loop.main(_cli_base_flags(args, sub_ws, scale_override=eff_scale) + ['--mode', 'evaluate', *shared_flags])
            eval_closed_loop.main(_cli_base_flags(args, sub_ws, scale_override=eff_scale) + ['--mode', 'timing', *shared_flags])
        else:
            fd_flags = [] if args.resume else ['--fresh_session']
            eval_closed_loop.main(_cli_base_flags(args, sub_ws, scale_override=eff_scale) + ['--mode', 'all', *fd_flags, *shared_flags])
    else:
        worker_dev = 'cuda:0' if use_cuda else 'cpu'
        logdir = sub_ws / 'logs'
        sharded_tune_supported = (stage == 'm5' or eff_scale == 'full')
        if len(gpus) <= 1 or not sharded_tune_supported:
            tune_w = 1
            tune_gpus = gpus[:1]
        else:
            std_dev_n = int(getattr(args, 'standard_dev_prompts', 8))
            long_dev_n = int(getattr(args, 'long_dev_prompts', None) or (32 if eff_scale == 'full' else 12))
            dev_prompts = select_prompts(prompt_manifest, 'dev', eff_scale, std_dev_n, long_dev_n)
            tune_w = max(1, min(len(gpus), len(dev_prompts)))
            tune_gpus = gpus[:tune_w]

        # 1) Dev tune (+ tune_summarize when sharded)
        if not _stage_settings_already_valid(
            args, sub_ws, stage, eff_scale, prompt_manifest,
            dense_ckpt=d_ckpt, sparse_ckpt=s_ckpt, proj_ckpt=p_ckpt,
            m5_arms=m5_arms, extra_flags=extra,
        ):
            if tune_w == 1:
                tune_cmd = [
                    sys.executable, '-m', 'edraft.eval_closed_loop',
                    *_cli_base_flags(args, sub_ws, scale_override=eff_scale, device_override=worker_dev),
                    '--mode', 'tune',
                    '--rank', '0', '--world_size', '1',
                    *shared_flags,
                ]
                if not args.resume:
                    tune_cmd.append('--fresh_session')
                _spawn_parallel_workers(
                    [(0, tune_gpus[0], tune_cmd)], logdir, f'{stage}-tune',
                    scale=eff_scale, device_str=worker_dev, omp_threads=2,
                )
            else:
                tune_plan_path = sub_ws / stage / 'tune' / 'sharding_plan.json'
                tune_session = (
                    json.loads(tune_plan_path.read_text()).get('logical_session')
                    if (args.resume and tune_plan_path.is_file()
                        and json.loads(tune_plan_path.read_text()).get('world_size') == tune_w)
                    else uuid.uuid4().hex
                )
                tune_specs = []
                for r, gpu in enumerate(tune_gpus):
                    t_cmd = [
                        sys.executable, '-m', 'edraft.eval_closed_loop',
                        *_cli_base_flags(args, sub_ws, scale_override=eff_scale, device_override=worker_dev),
                        '--mode', 'tune',
                        '--rank', str(r), '--world_size', str(tune_w),
                        '--logical_session', tune_session,
                        *shared_flags,
                    ]
                    if not args.resume:
                        t_cmd.append('--fresh_session')
                    tune_specs.append((r, gpu, t_cmd))
                _spawn_parallel_workers(
                    tune_specs, logdir, f'{stage}-tune',
                    scale=eff_scale, device_str=worker_dev, omp_threads=2,
                )
                tune_sum_cmd = [
                    sys.executable, '-m', 'edraft.eval_closed_loop',
                    *_cli_base_flags(args, sub_ws, scale_override=eff_scale, device_override='cpu'),
                    '--mode', 'tune_summarize',
                    *shared_flags,
                ]
                _spawn_parallel_workers(
                    [(0, tune_gpus[0], tune_sum_cmd)], logdir, f'{stage}-tune-summarize',
                    scale='smoke', device_str='cpu', omp_threads=1,
                )

        # 2) Dynamic sharded evaluate across all W workers
        eval_specs = []
        for r, gpu in enumerate(gpus):
            ev_cmd = [
                sys.executable, '-m', 'edraft.eval_closed_loop',
                *_cli_base_flags(args, sub_ws, scale_override=eff_scale, device_override=worker_dev),
                '--mode', 'evaluate',
                '--schedule', 'dynamic',
                '--rank', str(r), '--world_size', str(len(gpus)),
                *shared_flags,
            ]
            eval_specs.append((r, gpu, ev_cmd))
        _spawn_parallel_workers(
            eval_specs, logdir, f'{stage}-evaluate',
            scale=eff_scale, device_str=worker_dev, omp_threads=2,
        )

        # 3) Optional 1-GPU cluster-isolated timing_calibration on GPU 0 (`CPUs 0..15`)
        _, selected_test, is_pilot = _select_test_prompts(prompt_manifest, eff_scale, int(args.timing_prompts))
        timing_w = max(1, min(len(gpus), len(selected_test)))
        timing_gpus = gpus[:timing_w]
        if run_1gpu_calibration and use_cuda and timing_w > 1 and gpus[0] == 0 and not is_pilot:
            cal_cmd = [
                sys.executable, '-m', 'edraft.eval_closed_loop',
                *_cli_base_flags(args, sub_ws, scale_override=eff_scale, device_override='cuda:0'),
                '--mode', 'timing_calibration',
                '--calibration_repeats', '2',
                *shared_flags,
            ]
            if not args.resume:
                cal_cmd.append('--fresh_session')
            _spawn_parallel_workers(
                [(0, 0, cal_cmd)], logdir, f'{stage}-timing-calibration',
                scale=eff_scale, device_str='cuda:0', omp_threads=1,
            )

        # 4) Token-balanced snake-sharded timing across `timing_w` workers
        plan_path = sub_ws / stage / 'timing' / 'sharding_plan.json'
        logical_session = (
            json.loads(plan_path.read_text()).get('logical_session')
            if (args.resume and timing_w > 1 and plan_path.is_file()
                and json.loads(plan_path.read_text()).get('world_size') == timing_w)
            else (uuid.uuid4().hex if timing_w > 1 else None)
        )
        timing_specs = []
        for r, gpu in enumerate(timing_gpus):
            tm_cmd = [
                sys.executable, '-m', 'edraft.eval_closed_loop',
                *_cli_base_flags(args, sub_ws, scale_override=eff_scale, device_override=worker_dev),
                '--mode', 'timing',
                '--rank', str(r), '--world_size', str(timing_w),
                *shared_flags,
            ]
            if logical_session is not None:
                tm_cmd.extend(['--logical_session', logical_session])
            if not args.resume:
                tm_cmd.append('--fresh_session')
            timing_specs.append((r, gpu, tm_cmd))
        _spawn_parallel_workers(
            timing_specs, logdir, f'{stage}-timing',
            scale=eff_scale, device_str=worker_dev, omp_threads=1,
        )

        # 5) Pure-CPU summarize (+ calibration reconciliation when calibration exists)
        sum_cmd = [
            sys.executable, '-m', 'edraft.eval_closed_loop',
            *_cli_base_flags(args, sub_ws, scale_override=eff_scale, device_override='cpu'),
            '--mode', 'summarize',
            *shared_flags,
        ]
        if (sub_ws / stage / 'timing_isolation_calibration.json').is_file() and timing_w > 1:
            sum_cmd.append('--reconcile_sharded_timing')
        _spawn_parallel_workers(
            [(0, gpus[0], sum_cmd)], logdir, f'{stage}-summarize',
            scale='smoke', device_str='cpu', omp_threads=1,
        )

    summary = json.loads((sub_ws / stage / 'summary.json').read_text())
    timing = json.loads((sub_ws / stage / 'timing.json').read_text())
    settings = json.loads((sub_ws / stage / 'settings.json').read_text())
    return summary, timing, settings


def _suite_completion_status(summary: dict, timing: dict) -> str:
    missing_reqs = summary.get('missing_requests')
    no_missing = isinstance(missing_reqs, list) and len(missing_reqs) == 0
    if summary.get('status') == 'complete' and timing.get('status') == 'complete':
        return 'complete'
    if (
        no_missing
        and summary.get('greedy_output_comparisons_complete', False)
        and timing.get('status') in ('complete', 'pilot_complete')
    ):
        return 'pilot_complete'
    return 'incomplete'


# ---------------------------------------------------------------------------
# Suite 1: `bigram_eviction` (`prompt.md` §9 & §10 M3)
# ---------------------------------------------------------------------------


def run_bigram_eviction(args, reference_artifacts: Path, ablations_root: Path) -> dict:
    eff_scale = 'smoke' if args.scale == 'smoke' else 'reference'
    cfg_fp = _suite_config_fingerprint('bigram_eviction', args, reference_artifacts)
    sub_ws = ablations_root / f'bigram_eviction_ws_{cfg_fp[:12]}'
    report_path = ablations_root / 'bigram_eviction.json'
    cached = _can_resume_report(report_path, args, cfg_fp)
    if cached is not None:
        if sub_ws.is_dir():
            _update_latest_ws_symlink(ablations_root / 'bigram_eviction_workspace', sub_ws)
        return cached

    gpus = _resolve_gpus(args)
    _update_latest_ws_symlink(ablations_root / 'bigram_eviction_workspace', sub_ws)
    if eff_scale != 'smoke':
        _link_prereqs(
            reference_artifacts,
            sub_ws,
            (
                'm2.json', 'm2', 'm3', 'm4', 'm1_audit.json',
                'prompts.json', 'h_proj.pt', 'environment.json',
                'smoke.json', f'diagnostic_{eff_scale}.json',
            ),
        )

    arms = list(args.bigram_eviction_arms)
    extra_flags = [
        '--bigram_k_ret', str(args.bigram_k_ret),
        '--bigram_count_gate_kappa', str(args.bigram_count_gate_kappa),
    ]
    summary, timing, settings = _execute_closed_loop_stage(
        args, sub_ws, 'm5', eff_scale, gpus,
        m5_arms=arms, extra_stage_flags=extra_flags,
    )
    cal_path = sub_ws / 'm5' / 'timing_isolation_calibration.json'
    cal_report = json.loads(cal_path.read_text()) if cal_path.is_file() else None
    suite_status = _suite_completion_status(summary, timing)

    report = {
        'suite': 'bigram_eviction',
        'scale': eff_scale,
        'status': suite_status,
        'config_fingerprint': cfg_fp,
        'gpus': gpus,
        'timing_world_size': timing.get('world_size', 1),
        'sharding_mode': timing.get('sharding_mode', 'single_gpu'),
        'arms': arms,
        'bigram_k_ret': int(args.bigram_k_ret),
        'bigram_count_gate_kappa': float(args.bigram_count_gate_kappa),
        'selected_settings': settings.get('selected', {}),
        'evaluation_status': summary.get('status'),
        'timing_status': timing.get('status'),
        'overall_evaluation_groups': summary.get('groups', []),
        'isolated_timing_groups': timing.get('groups', []),
        'bigram_memory_ablation': summary.get('bigram_memory_ablation', []),
        'matched_support_comparisons': summary.get('matched_support_comparisons', []),
        'evaluation_offset_bin_slices': summary.get('offset_bin_slices', []),
        'isolated_timing_offset_bin_slices': timing.get('offset_bin_slices', []),
        'head_to_head_online_offset_comparison': summary.get('head_to_head_online_offset_comparison', []),
        'timing_isolation_calibration': cal_report,
        'artifacts': {
            'summary_json': str((sub_ws / 'm5' / 'summary.json').resolve()),
            'timing_json': str((sub_ws / 'm5' / 'timing.json').resolve()),
            'settings_json': str((sub_ws / 'm5' / 'settings.json').resolve()),
            'timing_calibration_json': str(cal_path.resolve()) if cal_path.is_file() else None,
        },
    }
    atomic_json(report_path, report)
    return report


# ---------------------------------------------------------------------------
# Suite 2: `full_vocab_lookahead` (`prompt.md` §7 & §10 M5)
# ---------------------------------------------------------------------------


def run_full_vocab_lookahead(args, reference_artifacts: Path, ablations_root: Path) -> dict:
    eff_scale = 'smoke' if args.scale == 'smoke' else 'reference'
    cfg_fp = _suite_config_fingerprint('full_vocab_lookahead', args, reference_artifacts)
    sub_ws = ablations_root / f'full_vocab_lookahead_ws_{cfg_fp[:12]}'
    report_path = ablations_root / 'full_vocab_lookahead.json'
    cached = _can_resume_report(report_path, args, cfg_fp)
    if cached is not None:
        if sub_ws.is_dir():
            _update_latest_ws_symlink(ablations_root / 'full_vocab_lookahead_workspace', sub_ws)
        return cached

    gpus = _resolve_gpus(args)
    _update_latest_ws_symlink(ablations_root / 'full_vocab_lookahead_workspace', sub_ws)
    if eff_scale != 'smoke':
        _link_prereqs(
            reference_artifacts,
            sub_ws,
            (
                'm2.json', 'm2', 'm3', 'm4', 'm1_audit.json',
                'prompts.json', 'h_proj.pt', 'environment.json',
                'smoke.json', f'diagnostic_{eff_scale}.json',
            ),
        )

    la_ckpts = _resolve_lookahead_checkpoints(args, reference_artifacts, sub_ws, eff_scale)
    arms = list(args.full_vocab_lookahead_arms)
    b0, b1 = [float(x) for x in args.m5_beta_candidates]
    extra_flags = [
        '--m5_beta_candidates', str(b0), str(b1),
        '--m5_beta_relax', str(args.m5_beta_relax),
    ]
    if la_ckpts['checkpoint_scale'] is not None:
        extra_flags.extend(['--checkpoint_scale', str(la_ckpts['checkpoint_scale'])])

    summary, timing, settings = _execute_closed_loop_stage(
        args,
        sub_ws,
        'm5',
        eff_scale,
        gpus,
        dense_ckpt=la_ckpts['dense_checkpoint'] if eff_scale != 'smoke' else None,
        sparse_ckpt=la_ckpts['sparse_checkpoint'] if eff_scale != 'smoke' else None,
        m5_arms=arms,
        extra_stage_flags=extra_flags,
    )
    cal_path = sub_ws / 'm5' / 'timing_isolation_calibration.json'
    cal_report = json.loads(cal_path.read_text()) if cal_path.is_file() else None
    suite_status = _suite_completion_status(summary, timing)

    report = {
        'suite': 'full_vocab_lookahead',
        'scale': eff_scale,
        'status': suite_status,
        'config_fingerprint': cfg_fp,
        'gpus': gpus,
        'timing_world_size': timing.get('world_size', 1),
        'sharding_mode': timing.get('sharding_mode', 'single_gpu'),
        'arms': arms,
        'checkpoint_provenance': {
            'source': la_ckpts['source'],
            'dense_checkpoint': str(la_ckpts['dense_checkpoint']),
            'sparse_checkpoint': str(la_ckpts['sparse_checkpoint']),
            'checkpoint_scale': la_ckpts['checkpoint_scale'],
            'm5_beta_candidates': [b0, b1],
            'm5_beta_relax': float(args.m5_beta_relax),
        },
        'selected_settings': settings.get('selected', {}),
        'evaluation_status': summary.get('status'),
        'timing_status': timing.get('status'),
        'overall_evaluation_groups': summary.get('groups', []),
        'isolated_timing_groups': timing.get('groups', []),
        'phase_profiles': timing.get('phase_profiles', []),
        'matched_support_comparisons': summary.get('matched_support_comparisons', []),
        'evaluation_offset_bin_slices': summary.get('offset_bin_slices', []),
        'isolated_timing_offset_bin_slices': timing.get('offset_bin_slices', []),
        'timing_isolation_calibration': cal_report,
        'artifacts': {
            'summary_json': str((sub_ws / 'm5' / 'summary.json').resolve()),
            'timing_json': str((sub_ws / 'm5' / 'timing.json').resolve()),
            'settings_json': str((sub_ws / 'm5' / 'settings.json').resolve()),
            'timing_calibration_json': str(cal_path.resolve()) if cal_path.is_file() else None,
        },
    }
    atomic_json(report_path, report)
    return report


# ---------------------------------------------------------------------------
# Suite 3: `pure_t1_fit` (`prompt.md` §10 M2 `--allow_pure_t1_ablation`)
# ---------------------------------------------------------------------------


def _run_training_fit_isolated(
    args,
    ws: Path,
    eff_scale: str,
    gpus: list[int],
    fit_tag: str | None,
    extra_train_flags: list[str],
) -> dict:
    from . import launch, train

    use_cuda = (eff_scale != 'smoke' and str(args.device).startswith('cuda'))
    stats_mem = _default_stats_memory_limit(args, eff_scale)
    raw_nw = getattr(args, 'train_num_workers', None)
    eff_num_workers = (2 if use_cuda else 0) if raw_nw is None else int(raw_nw)
    eff_mb = getattr(args, 'microbatch_blocks', None)
    extra_list = list(extra_train_flags)
    base_name = fit_tag or 'm2'
    fit_subdir = f'{base_name}_smoke' if eff_scale == 'smoke' else base_name

    report_json = ws / f'{fit_subdir}.json'
    if (
        getattr(args, 'resume', True)
        and report_json.is_file()
        and (ws / fit_subdir / 'dense.pt').is_file()
        and (ws / fit_subdir / 'dense_sparse.pt').is_file()
    ):
        need_cal = ('--calibrate_beta' in extra_list or '--apply_calibrated_beta' in extra_list)
        cal_ok = (
            not need_cal
            or (
                (ws / fit_subdir / 'dense_calibrated.pt').is_file()
                and (ws / fit_subdir / 'dense_sparse_calibrated.pt').is_file()
            )
        )
        if cal_ok:
            try:
                rep_existing = json.loads(report_json.read_text())
                impl_map = rep_existing.get('implementation_sha256')
                src_dir = Path(train.__file__).resolve().parent
                impl_ok = (
                    isinstance(impl_map, dict)
                    and bool(impl_map)
                    and all(
                        (src_dir / fname).is_file() and file_sha256(src_dir / fname) == digest
                        for fname, digest in impl_map.items()
                    )
                )
                if (
                    rep_existing.get('status') == 'passed'
                    and rep_existing.get('scale') == eff_scale
                    and impl_ok
                ):
                    return rep_existing
            except (OSError, ValueError):
                pass

    if use_cuda:
        launch_src = Path(launch.__file__).read_text()
        ddp_cli = [
            '--scale', eff_scale,
            '--seed', str(args.seed),
            '--output', str(ws),
            '--target', str(args.target),
            '--drafter', str(args.drafter),
            '--task', 'train',
            '--gpus', ','.join(str(g) for g in gpus),
            '--m1_diagnostic', str(ws / f'diagnostic_{eff_scale}.json'),
            '--stats_memory_limit', stats_mem,
            '--pin_memory',
        ]
        if fit_tag is not None:
            ddp_cli.extend(['--m5_fit_tag', fit_tag])
        ddp_cli.extend(extra_list)
        if '--num_workers' not in extra_list and eff_num_workers > 0:
            ddp_cli.extend(['--num_workers', str(eff_num_workers)])
        if '--microbatch_blocks' not in extra_list and eff_mb is not None:
            ddp_cli.extend(['--microbatch_blocks', str(eff_mb)])
        if '--checkpoint_every' in launch_src and '--checkpoint_every' not in extra_list:
            ddp_cli.extend(['--checkpoint_every', '1000'])
        if eff_scale == 'full' or getattr(args, 'allow_large_run', False):
            ddp_cli.append('--allow_large_run')
        if getattr(args, 'allow_download', False):
            ddp_cli.append('--allow_download')
        if not args.resume:
            ddp_cli.append('--no_resume')
        launch.main(ddp_cli)
        return json.loads((ws / f'{fit_subdir}.json').read_text())

    train_cli = _cli_base_flags(args, ws, scale_override=eff_scale) + ['--stats_memory_limit', stats_mem]
    if fit_tag is not None:
        train_cli.extend(['--m5_fit_tag', fit_tag])
    train_cli.extend(extra_list)
    if '--num_workers' not in extra_list and eff_num_workers > 0:
        train_cli.extend(['--num_workers', str(eff_num_workers)])
    if '--microbatch_blocks' not in extra_list and eff_mb is not None:
        train_cli.extend(['--microbatch_blocks', str(eff_mb)])
    if eff_scale != 'smoke':
        train_cli.extend([
            '--trace_dir', str(ws / 'traces'),
            '--m1_audit', str(ws / 'm1_audit.json'),
            '--m1_diagnostic', str(ws / f'diagnostic_{eff_scale}.json'),
            '--features', str(ws / 'token_features.pt'),
            '--projection', str(ws / 'h_proj.pt'),
            '--prompts', str(ws / 'prompts.json'),
        ])
    if not args.resume:
        train_cli.append('--no_resume')
    rep = train.main(train_cli)
    if rep is None:
        rep = json.loads((ws / f'{fit_subdir}.json').read_text())
    return rep


def _prepare_offline_eval_context(
    ws: Path,
    eff_scale: str,
    seed: int,
    device: torch.device,
    *,
    regime: str = 'primary',
):
    allow_t1 = (regime == 'pure_t1')
    elig_fn = pure_t1_eligible if allow_t1 else primary_eligible
    idx_regime = 'pure_T1_calibration_ablation_m5' if allow_t1 else 'primary'
    eval_ns = SimpleNamespace(
        scale=eff_scale,
        seed=seed,
        output=ws,
        history_depth=4,
        batch_blocks=8,
        vocab_chunk=8192,
        bf16_gemm=False,
        objective='union_rest_grouped',
        allow_long_oracle_fit=False,
        allow_pure_t1_ablation=allow_t1,
    )
    if eff_scale == 'smoke':
        trace_dir, _, head, bias, head_metadata = smoke_bundle(eval_ns)
        entries, _ = trace_index(trace_dir, eligibility_fn=elig_fn, regime=idx_regime)
    else:
        audit = json.loads((ws / 'm1_audit.json').read_text())
        entries, _ = trace_index(
            ws / 'traces', accepted_paths=audit['shards'],
            eligibility_fn=elig_fn, regime=idx_regime,
        )
        first = read_shard(entries[0].path)['blocks'][0]
        head_metadata = first['draft_lm_head']
        head, bias = load_frozen_head(head_metadata, device, output_dir=ws)
    refs = {
        'dev': epoch_references(entries, 'dev'),
        'test': epoch_references(entries, 'test'),
    }
    return eval_ns, refs, head, bias, head_metadata, elig_fn


def _offline_kl_summary(eval_ctx, ckpt_path: Path, device: torch.device) -> dict:
    eval_ns, refs, head, bias, head_metadata, elig_fn = eval_ctx
    adapter, meta = load_adapter_checkpoint(ckpt_path, device)
    adapter.eval()
    dev_eval = evaluate_offline(adapter, refs['dev'], head, bias, eval_ns, head_metadata, eligibility_fn=elig_fn)
    test_eval = evaluate_offline(adapter, refs['test'], head, bias, eval_ns, head_metadata, eligibility_fn=elig_fn)
    rep_collision_stats = (
        meta.get('report', {})
        .get('stages', {})
        .get('dense_sparse', {})
        .get('sparse_collision_statistics', {})
    )
    collision_count = rep_collision_stats.get('collision_count')
    if collision_count is None and isinstance(adapter.sparse, ExactObservedPairs):
        collision_count = int(adapter.sparse.collision_report([])['collision_count'])
    sparse_backend = (
        meta.get('adapter_config', {}).get('sparse_backend')
        or meta.get('report', {}).get('sparse_backend')
        or meta.get('sparse_backend', 'signed_feature_hash')
    )
    return {
        'checkpoint': str(ckpt_path),
        'stage': meta.get('stage'),
        'checkpoint_scale': meta.get('checkpoint_scale', 1.0),
        'sparse_backend': sparse_backend,
        'collision_count': collision_count,
        'dev': dev_eval,
        'test': test_eval,
    }


def _worker_dual_offline_kl(args, spec_path: Path, out_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    ws = Path(spec['workspace'])
    eff_scale = str(spec['effective_scale'])
    rank_gpu = int(spec['physical_gpu'])
    use_cuda = (eff_scale != 'smoke' and str(args.device).startswith('cuda'))
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    apply_gpu_cpu_affinity(device, rank_gpu)
    if eff_scale != 'smoke':
        assert_idle(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    primary_ctx = _prepare_offline_eval_context(ws, eff_scale, int(args.seed), device, regime='primary')
    pure_t1_ctx = _prepare_offline_eval_context(ws, eff_scale, int(args.seed), device, regime='pure_t1')
    primary_results = {
        key: _offline_kl_summary(primary_ctx, Path(ckpt_str), device)
        for key, ckpt_str in spec['checkpoints'].items()
    }
    pure_t1_results = {
        key: _offline_kl_summary(pure_t1_ctx, Path(ckpt_str), device)
        for key, ckpt_str in spec['checkpoints'].items()
    }
    atomic_json(out_path, {
        'primary_heldout_kl': primary_results,
        'pure_t1_heldout_kl': pure_t1_results,
    })


def _evaluate_dual_offline_kl_batch(
    args,
    ws: Path,
    eff_scale: str,
    checkpoints_by_key: dict[str, Path],
    *,
    gpu_index: int = 0,
) -> tuple[dict[str, dict], dict[str, dict]]:
    use_cuda = (eff_scale != 'smoke' and str(args.device).startswith('cuda'))
    if use_cuda:
        kl_dir = ws / 'dual_offline_kl_eval'
        kl_dir.mkdir(parents=True, exist_ok=True)
        tag = fingerprint(sorted((k, str(v.resolve())) for k, v in checkpoints_by_key.items()))[:10]
        spec_path = kl_dir / f'spec-{tag}.json'
        out_path = kl_dir / f'result-{tag}.json'
        atomic_json(spec_path, {
            'physical_gpu': gpu_index,
            'effective_scale': eff_scale,
            'workspace': str(ws.resolve()),
            'checkpoints': {k: str(v.resolve()) for k, v in checkpoints_by_key.items()},
        })
        cmd = [
            sys.executable, '-m', 'edraft.phase_b_suite',
            *_cli_base_flags(args, ws, scale_override=eff_scale, device_override='cuda:0'),
            '--_worker_mode', 'dual_offline_kl',
            '--_worker_spec', str(spec_path),
            '--_worker_out', str(out_path),
        ]
        _spawn_parallel_workers(
            [(0, gpu_index, cmd)], kl_dir / 'logs', f'dual-offline-kl-{tag}',
            scale=eff_scale, device_str='cuda:0', omp_threads=2,
        )
        payload = json.loads(out_path.read_text())
        return payload['primary_heldout_kl'], payload['pure_t1_heldout_kl']

    device = torch.device('cpu')
    primary_ctx = _prepare_offline_eval_context(ws, eff_scale, int(args.seed), device, regime='primary')
    pure_t1_ctx = _prepare_offline_eval_context(ws, eff_scale, int(args.seed), device, regime='pure_t1')
    return (
        {k: _offline_kl_summary(primary_ctx, p, device) for k, p in checkpoints_by_key.items()},
        {k: _offline_kl_summary(pure_t1_ctx, p, device) for k, p in checkpoints_by_key.items()},
    )


def _is_rep_excluded(row: dict) -> bool:
    if not str(row.get('condition', '')).startswith('long_'):
        return False
    if 'repetition_dominated' in row:
        return bool(row['repetition_dominated'])
    return bool(row.get('generation', {}).get('repetition_dominated', False))


def _run_variant_bundle_on_prompts(
    args,
    ws: Path,
    eff_scale: str,
    variants: list[tuple[str, Path, Path]],
    selected_prompts: list[dict],
    *,
    rank_gpu: int = 0,
    world_size: int = 1,
    device_override: str | None = None,
) -> tuple[dict[str, list[dict]], list[tuple[str, float, float]], dict]:
    dev_str = 'cpu' if eff_scale == 'smoke' else (device_override or str(args.device))
    device = torch.device(dev_str)
    if eff_scale != 'smoke':
        assert_idle(device)

    base_subdir = 'm2_smoke' if eff_scale == 'smoke' else 'm2'
    runner_ns = SimpleNamespace(
        scale=eff_scale,
        stage='m3',
        mode='timing',
        rank=rank_gpu,
        world_size=world_size,
        seed=args.seed,
        device=str(device),
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
        compile=bool(getattr(args, 'compile', False)),
        cuda_graph=bool(getattr(args, 'cuda_graph', False)),
        simulate_cpu_graph=False,
        bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
        compile_drafter=bool(getattr(args, 'compile_drafter', False)),
        compile_online=bool(getattr(args, 'compile_online', False)),
        fast_decode=bool(getattr(args, 'fast_decode', False) or os.environ.get('EDRAFT_FAST_DECODE', '0') == '1'),
        analytic_online_grad=bool(getattr(args, 'analytic_online_grad', True)),
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
            12, 6, seed=args.seed,
            prompt_ids=[f'synthetic-train-{i}' for i in range(64)],
        ).to(runner.device)

    loaded_variants = []
    variant_scales: list[tuple[str, float, float]] = []
    for v_name, d_ckpt, s_ckpt in variants:
        d_ad, d_meta = load_adapter_checkpoint(d_ckpt, runner.device)
        s_ad, s_meta = load_adapter_checkpoint(s_ckpt, runner.device)
        if d_meta.get('stage') != 'dense' or s_meta.get('stage') != 'dense_sparse':
            raise ValueError(f'Variant {v_name} checkpoints must have stages dense and dense_sparse')
        d_ad.eval()
        s_ad.eval()
        d_sc = float(d_meta.get('checkpoint_scale', 1.0))
        s_sc = float(s_meta.get('checkpoint_scale', 1.0))
        loaded_variants.append((v_name, d_ad, d_sc, s_ad, s_sc))
        variant_scales.append((v_name, d_sc, s_sc))

    bundle_arms: list[tuple[str, str, DenseAdapter | None, float | None]] = [
        ('plain_dflash', 'plain_dflash', None, None),
    ]
    for v_name, d_ad, d_sc, s_ad, s_sc in loaded_variants:
        bundle_arms.append((f'{v_name}::dense', 'dense', d_ad, d_sc))
        bundle_arms.append((f'{v_name}::dense_sparse', 'dense_sparse', s_ad, s_sc))

    orig_dense, orig_sparse = runner.dense, runner.dense_sparse

    def _invoke(arm_kind: str, adapter_obj: DenseAdapter | None, base_job: dict) -> dict:
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
        base_w = {
            'prompt': w_prompt, 'sampling_name': w_name, 'temperature': w_temp,
            'warper': w_warp, 'seed': args.seed, 'max_new_tokens_override': warmup_cap,
        }
        for _ in range(args.timing_warmups):
            for _, arm_kind, ad_obj, _ in bundle_arms:
                _invoke(arm_kind, ad_obj, base_w)

    rows_by_key: dict[str, list[dict]] = {key: [] for key, _, _, _ in bundle_arms}
    for p_idx, prompt in enumerate(selected_prompts):
        # For standard_nonthinking in pure_T=1 calibration ablation, evaluate ALL 3 regimes
        # (`greedy T=0`, `recommended T=0.7`, and `pure_temperature_diagnostic T=1.0`)!
        cond_list = sampling_conditions(prompt['condition'], primary_only=prompt['condition'].startswith('long_'))
        for s_name, temperature, warp in cond_list:
            base_job = {
                'prompt': prompt, 'sampling_name': s_name,
                'temperature': temperature, 'warper': warp, 'seed': args.seed,
            }
            for rep in range(args.timing_repeats):
                rot = (p_idx + rep) % len(bundle_arms)
                ordered = bundle_arms[rot:] + bundle_arms[:rot]
                for arm_key, arm_kind, ad_obj, _ in ordered:
                    assert_no_external_gpu_processes(runner.device)
                    row = _invoke(arm_kind, ad_obj, base_job)
                    rep_dom = bool(row.get('generation', {}).get('repetition_dominated', False))
                    rows_by_key[arm_key].append({
                        'prompt_id': prompt['prompt_id'],
                        'condition': prompt['condition'],
                        'sampling_name': s_name,
                        'temperature': float(temperature),
                        'repeat': rep,
                        'repetition_dominated': rep_dom,
                        'generation': {'repetition_dominated': rep_dom},
                        'effective_beta': float(row['effective_beta']),
                        'generated_tokens': int(row['generated_tokens']),
                        'elapsed_seconds': float(row['elapsed_seconds']),
                        'tokens_per_second': float(row['tokens_per_second']),
                        'mean_accepted_length': float(row['mean_accepted_length']),
                        'median_accepted_length': float(row['median_accepted_length']),
                        'target_forwards_per_generated_token': float(row['target_forwards_per_generated_token']),
                    })

    affinity = getattr(runner, 'affinity_info', None) or gpu_cpu_affinity(rank_gpu)
    return rows_by_key, variant_scales, affinity


def _worker_variant_eval_shard(args, spec_path: Path, out_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    ws = Path(spec['workspace'])
    eff_scale = str(spec['effective_scale'])
    variants = [(vn, Path(dp), Path(sp)) for vn, dp, sp in spec['variants']]
    shard_prompts = spec['prompts']
    rank_gpu = int(spec['physical_gpu'])
    use_cuda = (eff_scale != 'smoke' and str(args.device).startswith('cuda'))
    worker_dev = 'cuda:0' if use_cuda else 'cpu'
    rows_by_key, variant_scales, affinity = _run_variant_bundle_on_prompts(
        args, ws, eff_scale, variants, shard_prompts,
        rank_gpu=rank_gpu, world_size=int(spec['world_size']), device_override=worker_dev,
    )
    atomic_json(out_path, {
        'rank': int(spec['rank']),
        'physical_gpu': affinity['physical_gpu'],
        'numa_node': affinity['numa_node'],
        'cpu_affinity': affinity['cpu_affinity'],
        'cpu_affinity_applied': bool(affinity.get('cpu_affinity_applied', False)),
        'variant_scales': variant_scales,
        'rows_by_key': rows_by_key,
    })


def _evaluate_variants_closed_loop(
    args,
    ws: Path,
    eff_scale: str,
    variants: list[tuple[str, Path, Path]],
    gpus: list[int],
) -> tuple[str, dict[str, dict]]:
    prompt_manifest = (
        smoke_manifest()
        if eff_scale == 'smoke'
        else json.loads((ws / 'prompts.json').read_text())
    )
    all_test, selected_prompts, is_pilot = _select_test_prompts(prompt_manifest, eff_scale, int(args.timing_prompts))
    use_cuda = (eff_scale != 'smoke' and str(args.device).startswith('cuda'))

    if not use_cuda and len(gpus) <= 1:
        rows_by_key, variant_scales, _ = _run_variant_bundle_on_prompts(
            args, ws, eff_scale, variants, selected_prompts,
            rank_gpu=gpus[0] if gpus else 0, world_size=1,
        )
    else:
        eff_w, shards = _snake_shard_prompts(selected_prompts, prompt_manifest, ws, len(gpus))
        eff_gpus = gpus[:eff_w]
        shard_dir = ws / 'pure_t1_closed_loop_shards'
        shard_dir.mkdir(parents=True, exist_ok=True)
        worker_specs = []
        out_paths = []
        serialized_variants = [(vn, str(dc.resolve()), str(sc.resolve())) for vn, dc, sc in variants]
        for r, gpu in enumerate(eff_gpus):
            spec_path = shard_dir / f'spec-rank{r}.json'
            out_path = shard_dir / f'result-rank{r}.json'
            atomic_json(spec_path, {
                'rank': r,
                'world_size': eff_w,
                'physical_gpu': gpu,
                'effective_scale': eff_scale,
                'workspace': str(ws.resolve()),
                'variants': serialized_variants,
                'prompts': shards[r],
            })
            cmd = [
                sys.executable, '-m', 'edraft.phase_b_suite',
                *_cli_base_flags(args, ws, scale_override=eff_scale, device_override='cuda:0' if use_cuda else 'cpu'),
                '--timing_warmups', str(args.timing_warmups),
                '--timing_repeats', str(args.timing_repeats),
                '--smoke_tokens', str(args.smoke_tokens),
                '--max_new_tokens', str(args.max_new_tokens),
                '--long_max_new_tokens', str(args.long_max_new_tokens),
                *_fastpath_config_and_flags(args)[1],
                '--_worker_mode', 'variant_eval_shard',
                '--_worker_spec', str(spec_path),
                '--_worker_out', str(out_path),
            ]
            worker_specs.append((r, gpu, cmd))
            out_paths.append(out_path)

        _spawn_parallel_workers(
            worker_specs, shard_dir / 'logs', 'pure-t1-eval',
            scale=eff_scale, device_str='cuda:0' if use_cuda else 'cpu', omp_threads=1,
        )
        shard_payloads = [json.loads(p.read_text()) for p in out_paths]
        variant_scales = [(vn, float(ds), float(ss)) for vn, ds, ss in shard_payloads[0]['variant_scales']]
        prompt_order = {p['prompt_id']: idx for idx, p in enumerate(selected_prompts)}
        rows_by_key = {}
        for key in shard_payloads[0]['rows_by_key']:
            merged = [rw for sp in shard_payloads for rw in sp['rows_by_key'][key]]
            merged.sort(key=lambda rw: (prompt_order.get(rw['prompt_id'], 0), rw['sampling_name'], int(rw['repeat'])))
            rows_by_key[key] = merged

    plain_rows = rows_by_key['plain_dflash']
    plain_by_slot = {(r['prompt_id'], r['sampling_name'], r['repeat']): r for r in plain_rows}

    def _summarize(rows: list[dict], *, ckpt_scale: float | None = None, paired: bool = False) -> dict:
        prim = [r for r in rows if not _is_rep_excluded(r)]
        by_regime = {}
        for reg_key in sorted({(r['condition'], r['sampling_name'], r['temperature']) for r in rows}):
            cond, s_name, temp = reg_key
            sub = [r for r in rows if (r['condition'], r['sampling_name'], r['temperature']) == reg_key]
            sub_p = [r for r in sub if not _is_rep_excluded(r)]
            by_regime[f'{cond}::{s_name}'] = {
                'condition': cond,
                'sampling_name': s_name,
                'temperature': temp,
                'requests': len(sub),
                'primary_requests': len(sub_p),
                'repetition_excluded': len(sub) - len(sub_p),
                'mean_accepted_length': statistics.mean(r['mean_accepted_length'] for r in sub_p) if sub_p else None,
                'mean_tokens_per_second': statistics.mean(r['tokens_per_second'] for r in sub_p) if sub_p else None,
            }
        entry = {
            'requests': len(rows),
            'primary_requests': len(prim),
            'repetition_excluded': len(rows) - len(prim),
            'mean_accepted_length': statistics.mean(r['mean_accepted_length'] for r in prim) if prim else None,
            'mean_tokens_per_second': statistics.mean(r['tokens_per_second'] for r in prim) if prim else None,
            'by_regime': by_regime,
        }
        if ckpt_scale is not None:
            entry['checkpoint_scale'] = ckpt_scale
            entry['effective_beta'] = 1.0
        if paired:
            acc_deltas, tps_ratios = [], []
            excl_pairs = 0
            for r in rows:
                p_ref = plain_by_slot.get((r['prompt_id'], r['sampling_name'], r['repeat']))
                if p_ref is not None:
                    if _is_rep_excluded(r) or _is_rep_excluded(p_ref):
                        excl_pairs += 1
                        continue
                    acc_deltas.append(r['mean_accepted_length'] - p_ref['mean_accepted_length'])
                    if p_ref['tokens_per_second'] > 0:
                        tps_ratios.append(r['tokens_per_second'] / p_ref['tokens_per_second'])
            entry['paired_primary_requests'] = len(acc_deltas)
            entry['repetition_excluded_comparison_pairs'] = excl_pairs
            entry['paired_accepted_length_delta_vs_plain'] = statistics.mean(acc_deltas) if acc_deltas else None
            entry['paired_tps_ratio_vs_plain'] = statistics.mean(tps_ratios) if tps_ratios else None
        return entry

    plain_summary = _summarize(plain_rows, ckpt_scale=None, paired=False)
    cl_status = 'pilot_complete' if is_pilot else 'complete'
    out_by_variant = {}
    for v_name, d_sc, s_sc in variant_scales:
        out_by_variant[v_name] = {
            'status': cl_status,
            'is_pilot_subset': is_pilot,
            'selected_prompt_count': len(selected_prompts),
            'total_test_prompt_count': len(all_test),
            'repeats': int(args.timing_repeats),
            'warmups': int(args.timing_warmups),
            'plain_dflash': plain_summary,
            'dense': _summarize(rows_by_key[f'{v_name}::dense'], ckpt_scale=d_sc, paired=True),
            'dense_sparse': _summarize(rows_by_key[f'{v_name}::dense_sparse'], ckpt_scale=s_sc, paired=True),
        }
    return cl_status, out_by_variant


def run_pure_t1_fit(args, reference_artifacts: Path, ablations_root: Path) -> dict:
    from . import train

    eff_scale = 'smoke' if args.scale == 'smoke' else 'reference'
    cfg_fp = _suite_config_fingerprint('pure_t1_fit', args, reference_artifacts)
    ws = ablations_root / f'pure_t1_ws_{cfg_fp[:12]}'
    report_path = ablations_root / 'pure_t1_fit.json'
    cached = _can_resume_report(report_path, args, cfg_fp)
    if cached is not None:
        if ws.is_dir():
            _update_latest_ws_symlink(ablations_root / 'pure_t1_workspace', ws)
        return cached

    gpus = _resolve_gpus(args)
    _update_latest_ws_symlink(ablations_root / 'pure_t1_workspace', ws)
    if eff_scale == 'smoke':
        if not (ws / 'm2_smoke.json').is_file():
            train.main(_cli_base_flags(args, ws, scale_override='smoke'))
    else:
        _link_prereqs(
            reference_artifacts,
            ws,
            (
                'traces', 'm1_audit.json', f'diagnostic_{eff_scale}.json',
                'h_proj.pt', 'token_features.pt',
                'm2.json', 'm2', 'm3', 'm4', 'prompts.json',
                'environment.json', 'smoke.json',
            ),
        )

    fit_tag = str(args.pure_t1_tag)
    fit_subdir = f'{fit_tag}_smoke' if eff_scale == 'smoke' else fit_tag
    base_subdir = 'm2_smoke' if eff_scale == 'smoke' else 'm2'
    t1_fit_report = _run_training_fit_isolated(
        args, ws, eff_scale, gpus, fit_tag,
        [
            '--allow_pure_t1_ablation',
            '--block_shuffle_window', str(args.block_shuffle_window),
            '--weight_decay', str(args.weight_decay),
            '--calibrate_beta',
        ],
    )

    t1_dir = ws / fit_subdir
    base_dir = ws / base_subdir
    scale_tag = f'{args.fixed_checkpoint_scale:.2f}'.replace('.', 'p')
    base_scaled_dir = ws / 'baseline_m2_scaled'
    base_dense_s = base_scaled_dir / f'dense_scale{scale_tag}.pt'
    base_sparse_s = base_scaled_dir / f'dense_sparse_scale{scale_tag}.pt'
    scale_adapter_checkpoint(base_dir / 'dense.pt', base_dense_s, args.fixed_checkpoint_scale)
    scale_adapter_checkpoint(base_dir / 'dense_sparse.pt', base_sparse_s, args.fixed_checkpoint_scale)

    variants: list[tuple[str, Path, Path]] = [
        ('baseline_m2_unscaled', base_dir / 'dense.pt', base_dir / 'dense_sparse.pt'),
        (f'baseline_m2_scale_{scale_tag}', base_dense_s, base_sparse_s),
        ('pure_t1_uncalibrated', t1_dir / 'dense.pt', t1_dir / 'dense_sparse.pt'),
        ('pure_t1_dev_calibrated', t1_dir / 'dense_calibrated.pt', t1_dir / 'dense_sparse_calibrated.pt'),
    ]
    phase_a_pair = _find_phase_a_calibrated_refit(reference_artifacts) if eff_scale != 'smoke' else None
    if phase_a_pair is not None:
        variants.append((
            'primary_refit_shuf32_calibrated',
            phase_a_pair[0],
            phase_a_pair[1],
        ))

    ckpt_batch = {}
    for v_name, d_ckpt, s_ckpt in variants:
        ckpt_batch[f'{v_name}::dense'] = d_ckpt
        ckpt_batch[f'{v_name}::dense_sparse'] = s_ckpt
    primary_kl, pure_t1_kl = _evaluate_dual_offline_kl_batch(
        args, ws, eff_scale, ckpt_batch, gpu_index=gpus[0],
    )
    cl_status, cl_by_variant = _evaluate_variants_closed_loop(args, ws, eff_scale, variants, gpus)

    variant_reports = []
    for v_name, d_ckpt, s_ckpt in variants:
        variant_reports.append({
            'variant': v_name,
            'dense_checkpoint': str(d_ckpt.resolve()),
            'sparse_checkpoint': str(s_ckpt.resolve()),
            'primary_heldout_dense_kl': primary_kl[f'{v_name}::dense'],
            'primary_heldout_dense_sparse_kl': primary_kl[f'{v_name}::dense_sparse'],
            'pure_t1_heldout_dense_kl': pure_t1_kl[f'{v_name}::dense'],
            'pure_t1_heldout_dense_sparse_kl': pure_t1_kl[f'{v_name}::dense_sparse'],
            'closed_loop': cl_by_variant[v_name],
        })

    report = {
        'suite': 'pure_t1_fit',
        'scale': eff_scale,
        'status': cl_status,
        'config_fingerprint': cfg_fp,
        'gpus': gpus,
        'training_distribution': t1_fit_report.get('training_distribution'),
        'ablation_warning': t1_fit_report.get('ablation_warning'),
        'fit_gate_status': t1_fit_report.get('status'),
        'fit_report_path': str((ws / f'{fit_subdir}.json').resolve()),
        'beta_calibration': {
            st: t1_fit_report.get('stages', {}).get(st, {}).get('beta_calibration')
            for st in ('dense', 'dense_sparse')
            if t1_fit_report.get('stages', {}).get(st, {}).get('beta_calibration') is not None
        } or None,
        'variants': variant_reports,
    }
    atomic_json(report_path, report)
    return report


# ---------------------------------------------------------------------------
# Suite 4: `full_scale` (`--scale full --allow_large_run`, M1-pre -> M4)
# ---------------------------------------------------------------------------


def verify_full_scale_prompt_capacity(prompt_manifest: dict, eff_scale: str) -> dict:
    """Verify prompt dataset capacities against `prompt.md` §10 M1a/M3/M4 requirements."""
    rows = prompt_manifest.get('prompts', [])
    ids = [r['prompt_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prompt IDs in manifest')

    std_train = [r for r in rows if r['condition'] == 'standard_nonthinking' and r['split'] == 'train']
    std_dev = [r for r in rows if r['condition'] == 'standard_nonthinking' and r['split'] == 'dev']
    std_test_all = [r for r in rows if r['condition'] == 'standard_nonthinking' and r['split'] == 'test']
    std_test_sel = [r for r in std_test_all if r.get('reference_selected', True)]
    long_dev = [r for r in rows if r['condition'].startswith('long_') and r['split'] == 'dev' and r.get('reference_selected', True)]
    long_test_sel = [r for r in rows if r['condition'].startswith('long_') and r['split'] == 'test' and r.get('reference_selected', True)]
    cal_ids = prompt_manifest.get('calibration_ids', [])

    if eff_scale == 'smoke':
        reqs = {
            'unique_prompt_ids': True,
            'standard_train_nonempty': len(std_train) >= 1,
            'standard_dev_nonempty': len(std_dev) >= 1,
            'standard_test_nonempty': len(std_test_sel) >= 1,
        }
    else:
        heldout_ds = prompt_manifest.get('complete_heldout_dataset')
        heldout_ok = bool(heldout_ds) and all(r['split'] == 'test' for r in rows if r.get('dataset') == heldout_ds)
        # At ~296 empirical tokens/prompt on GSM8K, >=6,800 prompts yields >2.0M tokens per temperature (>4.0M across T=0 & T=0.7).
        reqs = {
            'unique_prompt_ids': True,
            'calibration_prompts_ge_64': len(cal_ids) >= 64,
            'standard_train_prompts_sufficient_for_2M_tokens': len(std_train) >= 6800,
            'standard_dev_prompts_ge_64': len(std_dev) >= 64,
            'standard_test_selected_ge_128': len(std_test_sel) >= 128,
            'complete_heldout_dataset_test_only': heldout_ok,
            'long_dev_prompts_ge_32': len(long_dev) >= 32,
            'long_test_selected_ge_96': len(long_test_sel) >= 96,
        }

    if not all(reqs.values()):
        raise RuntimeError(f'Prompt manifest failed capacity verification for scale={eff_scale}: {reqs}')
    return {
        'scale': eff_scale,
        'status': 'passed',
        'counts': {
            'total_prompts': len(rows),
            'calibration_prompts': len(cal_ids),
            'standard_train_prompts': len(std_train),
            'standard_dev_prompts': len(std_dev),
            'standard_test_total_heldout': len(std_test_all),
            'standard_test_selected': len(std_test_sel),
            'long_dev_prompts': len(long_dev),
            'long_test_selected': len(long_test_sel),
        },
        'requirements': reqs,
    }


def _bootstrap_full_scale_workspace(args, reference_artifacts: Path, full_ws: Path, eff_scale: str, gpus: list[int]) -> dict:
    full_ws.mkdir(parents=True, exist_ok=True)
    logdir = full_ws / 'logs'

    # 1. Seed environment.json and smoke.json without touching reference_artifacts
    if eff_scale == 'smoke':
        from .smoke import run_smoke
        if not (full_ws / 'smoke.json').is_file():
            run_smoke(SimpleNamespace(scale='smoke', seed=args.seed, device='cpu', output=full_ws))
    else:
        ref_env = reference_artifacts / 'environment.json'
        if not (full_ws / 'environment.json').is_file() and ref_env.is_file():
            _atomic_copy_file(ref_env, full_ws / 'environment.json')
        write_environment(resolved_config(args), output_dir=full_ws)
        ref_smoke = reference_artifacts / 'smoke.json'
        if not (full_ws / 'smoke.json').is_file():
            if ref_smoke.is_file():
                _atomic_copy_file(ref_smoke, full_ws / 'smoke.json')
            else:
                smoke_cmd = [sys.executable, '-m', 'edraft.smoke', '--scale', 'smoke', '--seed', str(args.seed), '--output', str(full_ws)]
                _spawn_parallel_workers([(0, gpus[0], smoke_cmd)], logdir, 'full-smoke-preflight', scale='smoke', device_str='cpu', omp_threads=1)

        # 2. Seed verified M0 reproduction & numerics gate into full_ws (`collect.py:117` allows reference M0 for full scale)
        m0_prov = {}
        if not (full_ws / 'm0.json').is_file() or not (full_ws / 'm0_numerics.json').is_file():
            ref_m0 = reference_artifacts / 'm0.json'
            ref_num = reference_artifacts / 'm0_numerics.json'
            if not ref_m0.is_file() or not ref_num.is_file():
                raise RuntimeError(f'Missing passed reference m0.json / m0_numerics.json in {reference_artifacts}')
            m0_prov['m0_sha256'] = _atomic_copy_file(ref_m0, full_ws / 'm0.json')
            m0_prov['m0_numerics_sha256'] = _atomic_copy_file(ref_num, full_ws / 'm0_numerics.json')
            ref_m0_dir = reference_artifacts / 'm0'
            if ref_m0_dir.is_dir():
                (full_ws / 'm0').mkdir(parents=True, exist_ok=True)
                copied_rows = 0
                for row_p in sorted(ref_m0_dir.glob('*.json')):
                    _atomic_copy_file(row_p, full_ws / 'm0' / row_p.name)
                    copied_rows += 1
                m0_prov['m0_row_files_copied'] = copied_rows
            m0_prov['source_reference_artifacts'] = str(reference_artifacts.resolve())
            atomic_json(full_ws / 'm0_provenance.json', m0_prov)

    # 3. Build / verify prompt manifest
    prompts_file = full_ws / ('prompts-smoke.json' if eff_scale == 'smoke' else 'prompts.json')
    need_prompts = True
    if prompts_file.is_file() and args.resume:
        try:
            existing_pm = json.loads(prompts_file.read_text())
            if existing_pm.get('scale') == eff_scale and int(existing_pm.get('seed', -1)) == int(args.seed):
                need_prompts = False
        except (OSError, ValueError):
            need_prompts = True
    if need_prompts:
        pm_cmd = [
            sys.executable, '-m', 'edraft.prompts',
            *_cli_base_flags(args, full_ws, scale_override=eff_scale, device_override='cpu'),
        ]
        _spawn_parallel_workers(
            [(0, gpus[0], pm_cmd)], logdir, 'full-prompts',
            scale='smoke', device_str='cpu', omp_threads=1,
        )
    if eff_scale == 'smoke' and (full_ws / 'prompts-smoke.json').is_file() and not (full_ws / 'prompts.json').is_file():
        _atomic_copy_file(full_ws / 'prompts-smoke.json', full_ws / 'prompts.json')

    pm_data = json.loads((full_ws / 'prompts.json').read_text())
    return verify_full_scale_prompt_capacity(pm_data, eff_scale)


def run_full_scale_pipeline(args, reference_artifacts: Path, phase_b_root: Path) -> dict:
    """Execute resumable 8-GPU `--scale full --allow_large_run` pipeline (`M1-pre -> M1a -> M1b -> M2 -> M3 -> M4`)."""
    eff_scale = 'smoke' if args.scale == 'smoke' else 'full'
    if eff_scale == 'full' and not getattr(args, 'allow_large_run', False):
        raise ValueError('full_scale suite requires --scale full --allow_large_run (or --scale smoke)')

    cfg_fp = _suite_config_fingerprint('full_scale', args, reference_artifacts)
    full_ws = (phase_b_root / f'full_scale_smoke_ws_{cfg_fp[:12]}') if eff_scale == 'smoke' else phase_b_root
    report_path = phase_b_root / 'full_scale_pipeline.json'
    cached = _can_resume_report(report_path, args, cfg_fp)
    if cached is not None:
        return cached

    gpus = _resolve_gpus(args)
    use_cuda = (eff_scale != 'smoke' and str(args.device).startswith('cuda'))
    worker_dev = 'cuda:0' if use_cuda else 'cpu'
    logdir = full_ws / 'logs'

    # Stage 0: Bootstrap workspace + verify prompt capacity
    prompt_cap_report = _bootstrap_full_scale_workspace(args, reference_artifacts, full_ws, eff_scale, gpus)

    # Stage 1: M1-pre hidden projection & token feature calibration (rank 0 / GPU 0)
    cal_path = full_ws / 'calibration.json'
    cal_done = False
    if args.resume and cal_path.is_file() and (full_ws / 'h_proj.pt').is_file() and (full_ws / 'token_features.pt').is_file():
        try:
            cal_data = json.loads(cal_path.read_text())
            cal_done = (cal_data.get('status') == 'passed' and cal_data.get('scale') == eff_scale)
        except (OSError, ValueError):
            cal_done = False
    if not cal_done:
        cal_cmd = [
            sys.executable, '-m', 'edraft.collect', '--stage', 'calibrate',
            *_cli_base_flags(args, full_ws, scale_override=eff_scale, device_override=worker_dev),
        ]
        if getattr(args, 'fast_decode', False) or os.environ.get('EDRAFT_FAST_DECODE', '0') == '1':
            cal_cmd.append('--fast_decode')
        _spawn_parallel_workers(
            [(0, gpus[0], cal_cmd)], logdir, 'full-m1-calibrate',
            scale=eff_scale, device_str=worker_dev, omp_threads=2,
        )
    cal_summary = json.loads(cal_path.read_text())

    # Stage 2 & 3: 8-GPU Dynamic M1a Trace Collection + M1a Audit
    train_tok_per_temp = (
        int(args.train_tokens_per_temperature)
        if args.train_tokens_per_temperature is not None
        else (32 if eff_scale == 'smoke' else 1_000_000)
    )
    m1_audit_path = full_ws / 'm1_audit.json'
    m1_passed = False
    if args.resume and m1_audit_path.is_file():
        try:
            existing_audit = json.loads(m1_audit_path.read_text())
            m1_passed = (existing_audit.get('status') == 'passed' and existing_audit.get('scale') == eff_scale)
        except (OSError, ValueError):
            m1_passed = False

    if not m1_passed:
        collect_w = 1 if eff_scale == 'smoke' else len(gpus)
        collect_gpus = gpus[:collect_w]
        collect_specs = []
        for r, gpu in enumerate(collect_gpus):
            c_cmd = [
                sys.executable, '-m', 'edraft.collect', '--stage', 'collect',
                *_cli_base_flags(args, full_ws, scale_override=eff_scale, device_override=worker_dev),
                '--rank', str(r), '--world_size', str(collect_w),
                '--train_tokens_per_temperature', str(train_tok_per_temp),
                '--standard_max_new_tokens', str(args.standard_max_new_tokens),
                '--long_max_new_tokens', str(args.long_max_new_tokens),
            ]
            if collect_w > 1:
                c_cmd.extend(['--schedule', 'dynamic'])
            if getattr(args, 'fast_decode', False) or os.environ.get('EDRAFT_FAST_DECODE', '0') == '1':
                c_cmd.append('--fast_decode')
            collect_specs.append((r, gpu, c_cmd))
        _spawn_parallel_workers(
            collect_specs, logdir, 'full-m1a-collect',
            scale=eff_scale, device_str=worker_dev, omp_threads=2,
        )
        audit_cmd = [
            sys.executable, '-m', 'edraft.collect', '--stage', 'audit',
            *_cli_base_flags(args, full_ws, scale_override=eff_scale, device_override='cpu'),
            '--train_tokens_per_temperature', str(train_tok_per_temp),
        ]
        _spawn_parallel_workers(
            [(0, gpus[0], audit_cmd)], logdir, 'full-m1a-audit',
            scale='smoke', device_str='cpu', omp_threads=2,
        )
    m1_audit = json.loads(m1_audit_path.read_text())
    if m1_audit.get('status') != 'passed':
        raise RuntimeError(f"M1a full-scale audit did not pass: {m1_audit.get('requirements')}")

    # Stage 4: Multi-worker CPU M1b Diagnose (`diagnostic_<scale>.json`)
    diag_path = full_ws / f'diagnostic_{eff_scale}.json'
    diag_done = False
    if args.resume and diag_path.is_file():
        try:
            d_rep = json.loads(diag_path.read_text())
            if d_rep.get('status') == 'measured' and d_rep.get('scale') == eff_scale:
                prov = d_rep.get('audit_provenance')
                diag_done = (eff_scale == 'smoke') or (
                    isinstance(prov, dict) and prov.get('audit_sha256') == file_sha256(m1_audit_path)
                )
        except (OSError, ValueError):
            diag_done = False
    if not diag_done:
        diag_workers = 1 if eff_scale == 'smoke' else int(args.diagnose_workers)
        diag_cmd = [
            sys.executable, '-m', 'edraft.diagnose',
            *_cli_base_flags(args, full_ws, scale_override=eff_scale, device_override='cpu'),
            '--trace_dir', str(full_ws / 'traces'),
            '--m1_audit', str(m1_audit_path),
            '--workers', str(diag_workers),
        ]
        _spawn_parallel_workers(
            [(0, gpus[0], diag_cmd)], logdir, 'full-m1b-diagnose',
            scale='smoke', device_str='cpu', omp_threads=2,
        )
    diag_report = json.loads(diag_path.read_text())

    # Stage 5: 8-GPU DDP M2 Offline Training (`m2/dense.pt`, `m2/dense_sparse.pt`, `m2.json`)
    m2_json_name = 'm2_smoke.json' if eff_scale == 'smoke' else 'm2.json'
    m2_subdir = 'm2_smoke' if eff_scale == 'smoke' else 'm2'
    m2_done = False
    if (
        args.resume
        and (full_ws / m2_json_name).is_file()
        and (full_ws / m2_subdir / 'dense.pt').is_file()
        and (full_ws / m2_subdir / 'dense_sparse.pt').is_file()
    ):
        try:
            m2_existing = json.loads((full_ws / m2_json_name).read_text())
            m2_done = (m2_existing.get('status') == 'passed' and m2_existing.get('scale') == eff_scale)
        except (OSError, ValueError):
            m2_done = False
    if not m2_done:
        m2_extra: list[str] = []
        m2_tag: str | None = None
        if args.full_m2_calibrated_fit:
            m2_tag = 'm2'
            m2_extra = [
                '--block_shuffle_window', str(args.block_shuffle_window),
                '--weight_decay', str(args.weight_decay),
                '--apply_calibrated_beta',
            ]
        _run_training_fit_isolated(args, full_ws, eff_scale, gpus, m2_tag, m2_extra)
    m2_report = json.loads((full_ws / m2_json_name).read_text())

    # Stage 6: 8-GPU Sharded M3 Closed-Loop (`tune -> tune_summarize -> evaluate -> timing_calibration -> timing -> summarize`)
    m3_summary, m3_timing, m3_settings = _execute_closed_loop_stage(
        args, full_ws, 'm3', eff_scale, gpus,
        dense_ckpt=full_ws / m2_subdir / 'dense.pt',
        sparse_ckpt=full_ws / m2_subdir / 'dense_sparse.pt',
    )
    if int(args.timing_prompts) == 0 and m3_summary.get('status') != 'complete':
        raise RuntimeError(f"M3 full-scale evaluation incomplete: missing={len(m3_summary.get('missing_requests', []))}")

    # Stage 7: 8-GPU Sharded M4 Closed-Loop (all 9 full-scale M4 arms + M3 baseline reuse + 1-GPU calibration)
    m4_summary, m4_timing, m4_settings = _execute_closed_loop_stage(
        args, full_ws, 'm4', eff_scale, gpus,
        dense_ckpt=full_ws / m2_subdir / 'dense.pt',
        sparse_ckpt=full_ws / m2_subdir / 'dense_sparse.pt',
    )

    m3_status = _suite_completion_status(m3_summary, m3_timing)
    m4_status = _suite_completion_status(m4_summary, m4_timing)
    if m3_status == 'complete' and m4_status == 'complete':
        pipe_status = 'complete'
    elif m3_status in ('complete', 'pilot_complete') and m4_status in ('complete', 'pilot_complete'):
        pipe_status = 'pilot_complete'
    else:
        pipe_status = 'incomplete'

    expected_m4_arms = list(arms_for('m4', eff_scale))
    report = {
        'suite': 'full_scale',
        'scale': eff_scale,
        'status': pipe_status,
        'config_fingerprint': cfg_fp,
        'gpus': gpus,
        'workspace': str(full_ws.resolve()),
        'prompt_capacity': prompt_cap_report,
        'm1_calibration': {
            'status': cal_summary.get('status'),
            'calibration_prompts': cal_summary.get('calibration_prompts'),
            'artifact_sha256': cal_summary.get('artifact_sha256'),
        },
        'm1a_audit': {
            'status': m1_audit.get('status'),
            'floors': m1_audit.get('floors'),
            'token_counts': m1_audit.get('token_counts'),
            'prompt_counts': m1_audit.get('prompt_counts'),
            'requirements': m1_audit.get('requirements'),
        },
        'm1b_diagnostic': {
            'status': diag_report.get('status'),
            'groups_count': len(diag_report.get('groups', [])),
            'path': str(diag_path.resolve()),
        },
        'm2_offline_fit': {
            'status': m2_report.get('status'),
            'relative_dev_improvement': m2_report.get('relative_dev_improvement'),
            'checkpoints': m2_report.get('checkpoints'),
            'path': str((full_ws / m2_json_name).resolve()),
        },
        'm3_closed_loop': {
            'status': m3_status,
            'arms': list(M3_ARMS),
            'selected_settings': m3_settings.get('selected', {}),
            'tune_sharding_mode': m3_settings.get('sharding_mode', 'single_gpu'),
            'timing_sharding_mode': m3_timing.get('sharding_mode', 'single_gpu'),
            'expected_requests': m3_summary.get('expected_requests'),
            'completed_requests': m3_summary.get('completed_requests'),
            'summary_json': str((full_ws / 'm3' / 'summary.json').resolve()),
            'timing_json': str((full_ws / 'm3' / 'timing.json').resolve()),
        },
        'm4_closed_loop': {
            'status': m4_status,
            'arms': expected_m4_arms,
            'selected_settings': m4_settings.get('selected', {}),
            'tune_sharding_mode': m4_settings.get('sharding_mode', 'single_gpu'),
            'timing_sharding_mode': m4_timing.get('sharding_mode', 'single_gpu'),
            'baseline_reuse': m4_summary.get('baseline_reuse'),
            'expected_requests': m4_summary.get('expected_requests'),
            'completed_requests': m4_summary.get('completed_requests'),
            'summary_json': str((full_ws / 'm4' / 'summary.json').resolve()),
            'timing_json': str((full_ws / 'm4' / 'timing.json').resolve()),
        },
    }
    atomic_json(report_path, report)
    return report


def main(argv=None):
    parser = common_parser(__doc__)
    parser.set_defaults(output=ROOT / 'artifacts/full_scale')
    parser.add_argument(
        '--suite',
        choices=['all', 'ref_secondary_ablations', *ALL_SUITES],
        default='all',
    )
    parser.add_argument(
        '--gpus',
        default=None,
        help='comma-separated physical GPU indices (defaults to 0..7 on CUDA, 0 on CPU/smoke)',
    )
    parser.add_argument(
        '--reference_artifacts',
        type=Path,
        default=ROOT / 'artifacts',
        help='read-only frozen `--scale reference` artifacts directory (never written to)',
    )
    parser.add_argument('--no_resume', dest='resume', action='store_false', default=True)
    parser.add_argument('--bigram_k_ret', type=int, default=256)
    parser.add_argument('--bigram_count_gate_kappa', type=float, default=12.0)
    parser.add_argument('--bigram_eviction_arms', nargs='+', default=list(DEFAULT_BIGRAM_EVICTION_ARMS))
    parser.add_argument('--full_vocab_lookahead_arms', nargs='+', default=list(DEFAULT_FULL_VOCAB_LOOKAHEAD_ARMS))
    parser.add_argument('--lookahead_dense_checkpoint', type=Path, default=None)
    parser.add_argument('--lookahead_sparse_checkpoint', type=Path, default=None)
    parser.add_argument('--lookahead_checkpoint_scale', type=float, default=None)
    parser.add_argument('--m5_beta_candidates', type=float, nargs=2, default=[0.5, 1.0])
    parser.add_argument('--m5_beta_relax', type=float, default=1.0)
    parser.add_argument('--pure_t1_tag', default='m5_pure_t1_calibration')
    parser.add_argument('--block_shuffle_window', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--fixed_checkpoint_scale', type=float, default=0.20)
    parser.add_argument('--stats_memory_limit', default=None)
    parser.add_argument('--train_num_workers', type=int, default=None)
    parser.add_argument('--microbatch_blocks', type=int, default=None)
    parser.add_argument('--train_tokens_per_temperature', type=int, default=None)
    parser.add_argument('--standard_max_new_tokens', type=int, default=1024)
    parser.add_argument('--diagnose_workers', type=int, default=16)
    parser.add_argument('--full_m2_calibrated_fit', action='store_true', default=False)
    parser.add_argument('--timing_repeats', type=int, default=3)
    parser.add_argument('--timing_warmups', type=int, default=1)
    parser.add_argument('--timing_prompts', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--long_max_new_tokens', type=int, default=24576)
    parser.add_argument('--smoke_tokens', type=int, default=32)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--cuda_graph', action='store_true', default=False)
    parser.add_argument('--bf16_gemm', action='store_true', default=False)
    parser.add_argument('--compile_drafter', action='store_true', default=False)
    parser.add_argument('--compile_online', action='store_true', default=False)
    parser.add_argument('--fast_decode', action='store_true', default=False)
    parser.add_argument('--no_analytic_online_grad', action='store_false', dest='analytic_online_grad', default=True)
    parser.add_argument(
        '--_worker_mode',
        choices=['dual_offline_kl', 'variant_eval_shard'],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--_worker_spec', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--_worker_out', type=Path, default=None, help=argparse.SUPPRESS)
    args = parse_config(parser, argv)
    args.output = Path(args.output).resolve()
    args.reference_artifacts = Path(args.reference_artifacts).resolve()
    if args.lookahead_dense_checkpoint is not None:
        args.lookahead_dense_checkpoint = Path(args.lookahead_dense_checkpoint).resolve()
    if args.lookahead_sparse_checkpoint is not None:
        args.lookahead_sparse_checkpoint = Path(args.lookahead_sparse_checkpoint).resolve()

    if args._worker_mode is not None:
        torch.set_num_threads(1)
        if args._worker_spec is None or args._worker_out is None:
            parser.error('internal worker mode requires --_worker_spec and --_worker_out')
        spec_p = Path(args._worker_spec).resolve()
        out_p = Path(args._worker_out).resolve()
        if args._worker_mode == 'dual_offline_kl':
            _worker_dual_offline_kl(args, spec_p, out_p)
        elif args._worker_mode == 'variant_eval_shard':
            _worker_variant_eval_shard(args, spec_p, out_p)
        return None

    try:
        gpus = _resolve_gpus(args)
    except ValueError as err:
        parser.error(str(err))

    if args.bigram_k_ret < 1:
        parser.error('--bigram_k_ret must be >= 1')
    if not math.isfinite(args.bigram_count_gate_kappa) or args.bigram_count_gate_kappa <= 0:
        parser.error('--bigram_count_gate_kappa must be finite and > 0')
    if (args.lookahead_dense_checkpoint is None) != (args.lookahead_sparse_checkpoint is None):
        parser.error('--lookahead_dense_checkpoint and --lookahead_sparse_checkpoint must be provided together')
    if args.lookahead_checkpoint_scale is not None and (
        not math.isfinite(args.lookahead_checkpoint_scale) or args.lookahead_checkpoint_scale < 0
    ):
        parser.error('--lookahead_checkpoint_scale must be finite and >= 0')
    b0, b1 = args.m5_beta_candidates
    if not (math.isfinite(b0) and math.isfinite(b1)) or b0 < 0 or b1 < 0 or b0 == b1:
        parser.error('--m5_beta_candidates requires two distinct finite nonnegative values')
    if not math.isfinite(args.m5_beta_relax) or args.m5_beta_relax < 0:
        parser.error('--m5_beta_relax must be finite and >= 0')
    if args.block_shuffle_window < 1:
        parser.error('--block_shuffle_window must be >= 1')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight_decay must be finite and >= 0')
    if not math.isfinite(args.fixed_checkpoint_scale) or args.fixed_checkpoint_scale <= 0:
        parser.error('--fixed_checkpoint_scale must be finite and > 0')
    if args.diagnose_workers < 1:
        parser.error('--diagnose_workers must be >= 1')
    if args.timing_repeats < 2 or args.timing_warmups < 1 or args.timing_prompts < 0:
        parser.error('timing requires repeats>=2, warmups>=1, prompts>=0')

    if args.suite == 'all':
        selected_suites = list(ALL_SUITES)
    elif args.suite == 'ref_secondary_ablations':
        selected_suites = list(REF_SECONDARY_SUITES)
    else:
        selected_suites = [args.suite]

    if 'full_scale' in selected_suites and args.scale != 'smoke':
        if args.scale != 'full' or not args.allow_large_run:
            parser.error("running 'full_scale' or 'all' on real models requires --scale full --allow_large_run")

    phase_b_root = Path(args.output)
    reference_artifacts = Path(args.reference_artifacts)
    if (
        args.scale != 'smoke'
        and reference_artifacts.is_dir()
        and (reference_artifacts / 'm4' / 'summary.json').is_file()
        and phase_b_root.resolve() == reference_artifacts.resolve()
    ):
        parser.error(
            '--output cannot equal --reference_artifacts when frozen M0-M4 reference artifacts exist; '
            'use --output artifacts/full_scale'
        )

    before_hashes = _snapshot_upstream_hashes(reference_artifacts)
    phase_b_root.mkdir(parents=True, exist_ok=True)
    ablations_root = phase_b_root / 'reference_secondary_ablations'
    if any(s in REF_SECONDARY_SUITES for s in selected_suites):
        ablations_root.mkdir(parents=True, exist_ok=True)
        if args.scale != 'smoke':
            _link_prereqs(reference_artifacts, ablations_root, ('smoke.json', 'environment.json'))

    record = record_run(args, f'phase_b_suite-{args.suite}')
    if args.scale != 'smoke':
        if not (phase_b_root / 'smoke.json').is_file() and (reference_artifacts / 'smoke.json').is_file():
            _atomic_copy_file(reference_artifacts / 'smoke.json', phase_b_root / 'smoke.json')
        if (phase_b_root / 'smoke.json').is_file():
            preflight(SimpleNamespace(scale=args.scale, output=phase_b_root), 4096)

    use_cuda = _is_cuda_run(args)
    orig_threads = torch.get_num_threads()
    results = {}
    try:
        if args.scale == 'smoke':
            torch.set_num_threads(1)
        for suite_name in selected_suites:
            if suite_name == 'bigram_eviction':
                results['bigram_eviction'] = run_bigram_eviction(args, reference_artifacts, ablations_root)
            elif suite_name == 'full_vocab_lookahead':
                results['full_vocab_lookahead'] = run_full_vocab_lookahead(args, reference_artifacts, ablations_root)
            elif suite_name == 'pure_t1_fit':
                results['pure_t1_fit'] = run_pure_t1_fit(args, reference_artifacts, ablations_root)
            elif suite_name == 'full_scale':
                results['full_scale'] = run_full_scale_pipeline(args, reference_artifacts, phase_b_root)

            if use_cuda and torch.cuda.is_initialized():
                raise RuntimeError(
                    f'Parent phase_b_suite orchestrator initialized CUDA context during {suite_name}'
                )
            _verify_upstream_immutability(reference_artifacts, before_hashes)
            if args.scale != 'smoke':
                torch.set_num_threads(orig_threads)
    finally:
        torch.set_num_threads(orig_threads)

    immutability_report = _verify_upstream_immutability(reference_artifacts, before_hashes)
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
        'gpus': gpus,
        'requested_suite': args.suite,
        'executed_suites': selected_suites,
        'output_root': str(phase_b_root.resolve()),
        'upstream_reference_immutability': immutability_report,
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
    summary_path = phase_b_root / 'phase_b_summary.json'
    atomic_json(summary_path, summary)
    print(json.dumps({
        'status': summary['status'],
        'scale': summary['scale'],
        'gpus': gpus,
        'executed_suites': selected_suites,
        'summary_path': str(summary_path),
    }))
    return summary


if __name__ == '__main__':
    main()
