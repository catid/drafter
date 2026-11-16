"""Launch resumable collection, DDP training, acceptance evaluation, or sharded matched-bundle timing."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from .eval_closed_loop import gpu_cpu_affinity
from .runtime import ROOT, atomic_json, common_parser, parse_config, record_run


def _free_localhost_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def selected_gpus_idle(indices):
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True)
    uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in query.stdout.splitlines()}
    missing = set(indices) - uuids.keys()
    if missing:
        raise RuntimeError(f'Unknown GPU indices: {sorted(missing)}')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                '--format=csv,noheader'], capture_output=True, text=True, check=True)
    occupied = [line for line in processes.stdout.splitlines()
                if line.split(',')[0].strip() in {uuids[i] for i in indices}]
    if occupied:
        raise RuntimeError('Selected GPU has an existing compute process: ' + '; '.join(occupied))


def _append_m5_eval_flags(command: list[str], args) -> None:
    if args.m5_arms:
        command.extend(['--m5_arms', *args.m5_arms])
    if args.m5_oracle_checkpoint is not None:
        command.extend(['--m5_oracle_checkpoint', str(args.m5_oracle_checkpoint)])
    if args.sparse_overflow_policy is not None:
        command.extend(['--sparse_overflow_policy', args.sparse_overflow_policy])
    if args.m5_triton:
        command.append('--m5_triton')
    if args.m5_torch_compile:
        command.append('--m5_torch_compile')
    if args.m5_server_url is not None:
        command.extend(['--m5_server_url', args.m5_server_url])
    if args.m5_server_model is not None:
        command.extend(['--m5_server_model', args.m5_server_model])
    if args.m5_server_api_key is not None:
        command.extend(['--m5_server_api_key', args.m5_server_api_key])
    if args.m5_server_timeout is not None:
        command.extend(['--m5_server_timeout', str(args.m5_server_timeout)])
    if args.bigram_count_gate_kappa is not None:
        command.extend(['--bigram_count_gate_kappa', str(args.bigram_count_gate_kappa)])
    if args.checkpoint_scale is not None:
        command.extend(['--checkpoint_scale', str(args.checkpoint_scale)])
    if args.m5_beta_candidates is not None:
        command.extend(['--m5_beta_candidates', *(str(b) for b in args.m5_beta_candidates)])
    if args.m5_beta_relax is not None:
        command.extend(['--m5_beta_relax', str(args.m5_beta_relax)])


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument(
        '--task',
        choices=['collect', 'm3', 'm4', 'm5', 'train', 'm3_timing', 'm4_timing', 'm5_timing'],
        default='collect',
    )
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--schedule', choices=['modulo', 'dynamic'], default='modulo')
    parser.add_argument('--partitions', nargs='+', default=['train_primary', 'standard_eval', 'diagnostic', 'long'],
                        choices=['train_primary', 'standard_eval', 'diagnostic', 'long', 'long_oracle_train'])
    parser.add_argument('--train_tokens_per_temperature', type=int)
    parser.add_argument('--standard_max_new_tokens', type=int, default=1024)
    parser.add_argument('--long_max_new_tokens', type=int, default=24576)
    parser.add_argument('--evaluation_max_new_tokens', type=int, default=2048)
    parser.add_argument('--logical_session', type=str, default=None)
    parser.add_argument('--fresh_session', action='store_true')
    parser.add_argument('--timing_repeats', type=int, default=3)
    parser.add_argument('--timing_warmups', type=int, default=1)
    parser.add_argument('--timing_prompts', type=int, default=0)
    parser.add_argument('--batch_blocks', type=int, default=8)
    parser.add_argument('--microbatch_blocks', type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--cuda_graph', action='store_true', default=False)
    parser.add_argument('--bf16_gemm', action='store_true', default=False)
    parser.add_argument('--compile_drafter', action='store_true', default=False)
    parser.add_argument('--compile_online', action='store_true', default=False)
    parser.add_argument('--optimizer', choices=['adamw', 'muon', 'pacemuon', 'anchormuon', 'adam_atan2'], default='adamw')
    parser.add_argument('--lr_schedule', choices=['constant', 'wsd', 'cosine'], default='constant')
    parser.add_argument('--block_shuffle_window', type=int, default=0)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--bias_weight_decay', type=float)
    parser.add_argument('--sparse_learning_rate', type=float)
    parser.add_argument('--lambda_l1', type=float)
    parser.add_argument('--epochs_dense', type=int)
    parser.add_argument('--epochs_sparse', type=int)
    parser.add_argument('--ema_eval', action='store_true', default=None)
    parser.add_argument('--no_ema_eval', action='store_false', dest='ema_eval')
    parser.add_argument('--pace_pullback', type=float)
    parser.add_argument('--anchor_soda', type=float)
    parser.add_argument('--calibrate_beta', action='store_true', default=False)
    parser.add_argument('--apply_calibrated_beta', action='store_true', default=False)
    parser.add_argument('--calibrate_beta_grid', type=float, nargs='+', default=None)
    parser.add_argument('--sparse_backend', choices=['signed_feature_hash', 'exact_observed_pairs'], default=None)
    parser.add_argument('--objective', choices=['union_rest_grouped', 'two_sided_pseudolikelihood'], default=None)
    parser.add_argument('--allow_long_oracle_fit', action='store_true', default=False)
    parser.add_argument('--allow_pure_t1_ablation', action='store_true', default=False)
    parser.add_argument('--m5_fit_tag', default=None)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    parser.add_argument('--eval_mode', choices=['evaluate', 'beta_sweep', 'server_eval'], default='evaluate')
    parser.add_argument('--beta_grid', type=float, nargs='+', default=None)
    parser.add_argument('--beta_sweep_split', choices=['dev', 'test'], default=None)
    parser.add_argument('--beta_sweep_arms', nargs='+', default=None)
    parser.add_argument('--m5_arms', nargs='+', default=None)
    parser.add_argument('--m5_oracle_checkpoint', type=Path, default=None)
    parser.add_argument('--sparse_overflow_policy', choices=['freeze_new', 'error'], default=None)
    parser.add_argument('--m5_triton', action='store_true', default=False)
    parser.add_argument('--m5_torch_compile', action='store_true', default=False)
    parser.add_argument('--m5_server_url', default=None)
    parser.add_argument('--m5_server_model', default=None)
    parser.add_argument('--m5_server_api_key', default=None)
    parser.add_argument('--m5_server_timeout', type=float, default=None)
    parser.add_argument('--bigram_count_gate_kappa', type=float, default=None)
    parser.add_argument('--dense_checkpoint', type=Path, default=None)
    parser.add_argument('--sparse_checkpoint', type=Path, default=None)
    parser.add_argument('--checkpoint_scale', type=float, default=None)
    parser.add_argument('--m5_beta_candidates', type=float, nargs=2, default=None)
    parser.add_argument('--m5_beta_relax', type=float, default=None)
    args = parse_config(parser, argv)

    is_timing_task = args.task in ('m3_timing', 'm4_timing', 'm5_timing')
    if args.gpus is None:
        args.gpus = '0,1,2,3,4,5,6,7' if is_timing_task else '0,1,2,3,4,5,6'

    if args.bigram_count_gate_kappa is not None:
        if not math.isfinite(args.bigram_count_gate_kappa) or args.bigram_count_gate_kappa <= 0:
            parser.error('--bigram_count_gate_kappa must be finite and > 0')
    if args.m5_server_timeout is not None:
        if not math.isfinite(args.m5_server_timeout) or args.m5_server_timeout <= 0:
            parser.error('--m5_server_timeout must be finite and > 0')
    if args.checkpoint_scale is not None:
        if not math.isfinite(args.checkpoint_scale) or args.checkpoint_scale < 0:
            parser.error('--checkpoint_scale must be finite and >= 0')
    if args.m5_beta_candidates is not None:
        b0, b1 = args.m5_beta_candidates
        if not (math.isfinite(b0) and math.isfinite(b1)) or b0 < 0 or b1 < 0 or b0 == b1:
            parser.error('--m5_beta_candidates requires two distinct finite nonnegative values')
    if args.m5_beta_relax is not None:
        if not math.isfinite(args.m5_beta_relax) or args.m5_beta_relax < 0:
            parser.error('--m5_beta_relax must be finite and >= 0')
    if args.beta_grid is not None:
        if not args.beta_grid or any(not math.isfinite(b) or b < 0 for b in args.beta_grid):
            parser.error('--beta_grid values must be finite and >= 0')
    if args.calibrate_beta_grid is not None:
        if not args.calibrate_beta_grid or any(not math.isfinite(b) or b < 0 for b in args.calibrate_beta_grid):
            parser.error('--calibrate_beta_grid values must be finite and >= 0')
    if (args.task in ('collect', 'train') or is_timing_task) and (
        args.eval_mode != 'evaluate' or args.beta_grid is not None
        or args.beta_sweep_split is not None or args.beta_sweep_arms is not None
    ):
        parser.error(f'evaluation mode/beta_sweep options are invalid when --task {args.task}')
    m5_only_used = bool(
        args.eval_mode == 'server_eval'
        or args.m5_arms
        or args.m5_oracle_checkpoint is not None
        or (args.sparse_overflow_policy is not None and args.task != 'train')
        or args.m5_triton
        or args.m5_torch_compile
        or args.m5_server_url is not None
        or args.m5_server_model is not None
        or args.m5_server_api_key is not None
        or args.m5_server_timeout is not None
        or args.bigram_count_gate_kappa is not None
        or args.dense_checkpoint is not None
        or args.sparse_checkpoint is not None
        or args.checkpoint_scale is not None
        or args.m5_beta_candidates is not None
        or args.m5_beta_relax is not None
    )
    if args.task not in ('m5', 'm5_timing') and m5_only_used:
        parser.error('M5-specific options (--m5_*, --dense_checkpoint, --sparse_checkpoint, --checkpoint_scale, --sparse_overflow_policy, --bigram_count_gate_kappa, --eval_mode server_eval) require --task m5 or m5_timing')
    if args.eval_mode == 'server_eval' and not args.m5_server_url:
        parser.error('--eval_mode server_eval requires --m5_server_url')
    if args.train_tokens_per_temperature is None:
        args.train_tokens_per_temperature = 1000000 if args.scale == 'full' else 250000
    record_run(args, 'launch')
    if args.scale == 'smoke' and not is_timing_task:
        from .smoke import run_smoke
        return run_smoke(args)
    gpus = [int(g) for g in args.gpus.split(',') if g.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('--gpus requires distinct physical indices')
    if is_timing_task and len(gpus) > 8:
        raise ValueError('timing sharding supports 1..8 GPUs')
    if args.scale != 'smoke':
        selected_gpus_idle(gpus)

    logdir = args.output / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    processes, streams = [], []
    stage = args.task.removesuffix('_timing') if is_timing_task else args.task
    if args.task == 'collect':
        label, log_label = 'collection', 'collect'
    elif args.task == 'train':
        label, log_label = 'training', 'train'
    elif is_timing_task:
        label, log_label = f'{stage}-timing', f'{stage}-timing'
    else:
        label = f'{args.task}-evaluation'
        log_label = f'{args.task}-{args.eval_mode}'

    master_port = _free_localhost_port() if args.task == 'train' and len(gpus) > 1 else None
    state_path = args.output / f'{label}-processes.json'
    logical_session = None
    if is_timing_task:
        plan_path = args.output / stage / 'timing' / 'sharding_plan.json'
        if args.logical_session is not None:
            logical_session = args.logical_session
        elif not args.fresh_session and len(gpus) > 1 and plan_path.is_file():
            existing_plan = json.loads(plan_path.read_text())
            logical_session = (
                existing_plan.get('logical_session')
                if existing_plan.get('world_size') == len(gpus)
                else uuid.uuid4().hex
            )
        elif len(gpus) > 1:
            logical_session = uuid.uuid4().hex

    state = {
        'scale': args.scale,
        'task': args.task,
        'schedule': args.schedule,
        'launcher_pid': os.getpid(),
        'logical_session': logical_session,
        'started_at_unix': time.time(),
        'workers': [],
    }
    src_path = str((ROOT / 'src').resolve())
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    worker_pythonpath = f'{src_path}{os.pathsep}{existing_pythonpath}' if existing_pythonpath else src_path
    worker_device = 'cpu' if args.scale == 'smoke' else 'cuda:0'
    dense_ckpt = Path(args.dense_checkpoint) if args.dense_checkpoint is not None else (args.output / 'm2/dense.pt')
    sparse_ckpt = Path(args.sparse_checkpoint) if args.sparse_checkpoint is not None else (args.output / 'm2/dense_sparse.pt')

    try:
        for rank, gpu in enumerate(gpus):
            base = ['--scale', args.scale, '--rank', str(rank), '--world_size', str(len(gpus)), '--device', worker_device,
                    '--seed', str(args.seed), '--output', str(args.output), '--target', args.target,
                    '--drafter', args.drafter]
            if args.task == 'train':
                command = [sys.executable, '-m', 'edraft.train', *base,
                           '--trace_dir', str(args.output / 'traces'),
                           '--m1_audit', str(args.output / 'm1_audit.json'),
                           '--features', str(args.output / 'token_features.pt'),
                           '--projection', str(args.output / 'h_proj.pt'),
                           '--prompts', str(args.output / 'prompts.json'),
                           '--batch_blocks', str(args.batch_blocks)]
                if args.microbatch_blocks is not None:
                    command.extend(['--microbatch_blocks', str(args.microbatch_blocks)])
                if args.num_workers > 0:
                    command.extend(['--num_workers', str(args.num_workers)])
                if args.pin_memory:
                    command.append('--pin_memory')
                if args.bf16_gemm:
                    command.append('--bf16_gemm')
                if args.compile:
                    command.append('--compile')
                if args.optimizer != 'adamw':
                    command.extend(['--optimizer', args.optimizer])
                if args.lr_schedule != 'constant':
                    command.extend(['--lr_schedule', args.lr_schedule])
                if args.block_shuffle_window != 0:
                    command.extend(['--block_shuffle_window', str(args.block_shuffle_window)])
                for flag_name in (
                    'learning_rate', 'weight_decay', 'bias_weight_decay',
                    'sparse_learning_rate', 'lambda_l1', 'epochs_dense',
                    'epochs_sparse', 'pace_pullback', 'anchor_soda',
                    'sparse_backend', 'sparse_overflow_policy', 'objective',
                    'm5_fit_tag',
                ):
                    val = getattr(args, flag_name)
                    if val is not None:
                        command.extend([f'--{flag_name}', str(val)])
                if args.calibrate_beta:
                    command.append('--calibrate_beta')
                if args.apply_calibrated_beta:
                    command.append('--apply_calibrated_beta')
                if args.calibrate_beta_grid is not None:
                    command.extend(['--calibrate_beta_grid', *(str(b) for b in args.calibrate_beta_grid)])
                if args.allow_long_oracle_fit:
                    command.append('--allow_long_oracle_fit')
                if args.allow_pure_t1_ablation:
                    command.append('--allow_pure_t1_ablation')
                if args.ema_eval is True:
                    command.append('--ema_eval')
                elif args.ema_eval is False:
                    command.append('--no_ema_eval')
                if not args.resume:
                    command.append('--no_resume')
            elif args.task == 'collect':
                command = [sys.executable, '-m', 'edraft.collect', '--stage', 'collect', *base,
                           '--long_max_new_tokens', str(args.long_max_new_tokens),
                           '--train_tokens_per_temperature', str(args.train_tokens_per_temperature),
                           '--standard_max_new_tokens', str(args.standard_max_new_tokens), '--partitions', *args.partitions]
                if args.schedule == 'dynamic':
                    command.extend(['--schedule', 'dynamic'])
            elif is_timing_task:
                command = [sys.executable, '-m', 'edraft.eval_closed_loop', '--stage', stage,
                           '--mode', 'timing', *base,
                           '--long_max_new_tokens', str(args.long_max_new_tokens),
                           '--max_new_tokens', str(args.evaluation_max_new_tokens),
                           '--dense_checkpoint', str(dense_ckpt),
                           '--sparse_checkpoint', str(sparse_ckpt),
                           '--projection', str(args.output / 'h_proj.pt'),
                           '--prompts_manifest', str(args.output / 'prompts.json'),
                           '--timing_repeats', str(args.timing_repeats),
                           '--timing_warmups', str(args.timing_warmups),
                           '--timing_prompts', str(args.timing_prompts)]
                if logical_session is not None:
                    command.extend(['--logical_session', logical_session])
                if args.fresh_session:
                    command.append('--fresh_session')
                if args.compile:
                    command.append('--compile')
                if args.cuda_graph:
                    command.append('--cuda_graph')
                if args.bf16_gemm:
                    command.append('--bf16_gemm')
                if args.compile_drafter:
                    command.append('--compile_drafter')
                if args.compile_online:
                    command.append('--compile_online')
                if stage == 'm5':
                    _append_m5_eval_flags(command, args)
            else:
                command = [sys.executable, '-m', 'edraft.eval_closed_loop', '--stage', args.task,
                           '--mode', args.eval_mode, *base, '--long_max_new_tokens', str(args.long_max_new_tokens),
                           '--max_new_tokens', str(args.evaluation_max_new_tokens),
                           '--dense_checkpoint', str(dense_ckpt),
                           '--sparse_checkpoint', str(sparse_ckpt),
                           '--projection', str(args.output / 'h_proj.pt'),
                           '--prompts_manifest', str(args.output / 'prompts.json')]
                if args.schedule == 'dynamic':
                    command.extend(['--schedule', 'dynamic'])
                if args.compile:
                    command.append('--compile')
                if args.cuda_graph:
                    command.append('--cuda_graph')
                if args.bf16_gemm:
                    command.append('--bf16_gemm')
                if args.compile_drafter:
                    command.append('--compile_drafter')
                if args.compile_online:
                    command.append('--compile_online')
                if args.beta_grid is not None:
                    command.extend(['--beta_grid', *(str(b) for b in args.beta_grid)])
                if args.beta_sweep_split is not None:
                    command.extend(['--beta_sweep_split', args.beta_sweep_split])
                if args.beta_sweep_arms is not None:
                    command.extend(['--beta_sweep_arms', *args.beta_sweep_arms])
                if args.task == 'm5':
                    _append_m5_eval_flags(command, args)
            if args.allow_large_run:
                command.append('--allow_large_run')
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES='' if args.scale == 'smoke' else str(gpu),
                RANK=str(rank),
                WORLD_SIZE=str(len(gpus)),
                PYTHONPATH=worker_pythonpath,
                HF_HUB_OFFLINE='1',
                HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='1' if is_timing_task else '4',
            )
            if master_port is not None:
                env['MASTER_ADDR'] = '127.0.0.1'
                env['MASTER_PORT'] = str(master_port)
            log_path = logdir / f'{log_label}-rank{rank}.log'
            log = log_path.open('a', buffering=1)
            streams.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            affinity = gpu_cpu_affinity(gpu)
            state['workers'].append({
                'rank': rank,
                'physical_gpu': gpu,
                'numa_node': affinity['numa_node'],
                'cpu_affinity': affinity['cpu_affinity'],
                'pid': process.pid,
                'command': command,
                'log': str(log_path),
                'status': 'running',
            })
            atomic_json(state_path, state)
        print(json.dumps(state), flush=True)
        while any(p.poll() is None for p in processes):
            for row, process in zip(state['workers'], processes):
                code = process.poll()
                if code is not None:
                    row.update({'status': 'complete' if code == 0 else 'failed', 'returncode': code})
            atomic_json(state_path, state)
            failed = [p.returncode for p in processes if p.returncode not in (None, 0)]
            if failed:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                raise RuntimeError('Worker failed; inspect rank-local logs. Completed jobs remain resumable.')
            time.sleep(0.2 if args.scale == 'smoke' else 5)
        for row, process in zip(state['workers'], processes):
            row.update({'status': 'complete' if process.returncode == 0 else 'failed',
                        'returncode': process.returncode})
        if any(p.returncode for p in processes):
            state['completed_at_unix'] = time.time()
            atomic_json(state_path, state)
            raise RuntimeError('Run failed; inspect worker logs')
        if is_timing_task:
            summarize_cmd = [
                sys.executable, '-m', 'edraft.eval_closed_loop', '--stage', stage, '--mode', 'summarize',
                '--scale', args.scale, '--device', 'cpu', '--seed', str(args.seed),
                '--output', str(args.output), '--target', args.target, '--drafter', args.drafter,
                '--long_max_new_tokens', str(args.long_max_new_tokens),
                '--max_new_tokens', str(args.evaluation_max_new_tokens),
                '--dense_checkpoint', str(dense_ckpt),
                '--sparse_checkpoint', str(sparse_ckpt),
                '--projection', str(args.output / 'h_proj.pt'),
                '--prompts_manifest', str(args.output / 'prompts.json'),
                '--timing_repeats', str(args.timing_repeats),
                '--timing_warmups', str(args.timing_warmups),
                '--timing_prompts', str(args.timing_prompts),
            ]
            if args.compile:
                summarize_cmd.append('--compile')
            if args.cuda_graph:
                summarize_cmd.append('--cuda_graph')
            if args.bf16_gemm:
                summarize_cmd.append('--bf16_gemm')
            if args.compile_drafter:
                summarize_cmd.append('--compile_drafter')
            if args.compile_online:
                summarize_cmd.append('--compile_online')
            if stage == 'm5':
                _append_m5_eval_flags(summarize_cmd, args)
            if args.allow_large_run:
                summarize_cmd.append('--allow_large_run')
            sum_env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES='',
                PYTHONPATH=worker_pythonpath,
                HF_HUB_OFFLINE='1',
                HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='1',
            )
            sum_proc = subprocess.run(summarize_cmd, cwd=ROOT, env=sum_env, capture_output=True, text=True, check=True)
            state['summarize_stdout'] = sum_proc.stdout.strip()
        state['completed_at_unix'] = time.time()
        atomic_json(state_path, state)
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
