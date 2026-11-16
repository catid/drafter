"""Launch resumable collection, acceptance evaluation, or sharded matched-bundle timing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .eval_closed_loop import gpu_cpu_affinity
from .runtime import ROOT, atomic_json, common_parser, parse_config, record_run


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


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument('--task', choices=['collect', 'm3', 'm4', 'm3_timing', 'm4_timing'], default='collect')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--partitions', nargs='+', default=['train_primary', 'standard_eval', 'diagnostic', 'long'],
                        choices=['train_primary', 'standard_eval', 'diagnostic', 'long'])
    parser.add_argument('--train_tokens_per_temperature', type=int)
    parser.add_argument('--standard_max_new_tokens', type=int, default=1024)
    parser.add_argument('--long_max_new_tokens', type=int, default=24576)
    parser.add_argument('--evaluation_max_new_tokens', type=int, default=2048)
    parser.add_argument('--logical_session', type=str, default=None)
    parser.add_argument('--fresh_session', action='store_true')
    parser.add_argument('--timing_repeats', type=int, default=3)
    parser.add_argument('--timing_warmups', type=int, default=1)
    parser.add_argument('--timing_prompts', type=int, default=0)
    args = parse_config(parser, argv)
    is_timing_task = args.task in ('m3_timing', 'm4_timing')
    if args.gpus is None:
        args.gpus = '0,1,2,3,4,5,6,7' if is_timing_task else '0,1,2,3,4,5,6'
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
    label = 'collection' if args.task == 'collect' else (f'{stage}-timing' if is_timing_task else f'{args.task}-evaluation')
    log_label = 'collect' if args.task == 'collect' else (f'{stage}-timing' if is_timing_task else f'{args.task}-evaluate')
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
    state = {'scale': args.scale, 'task': args.task, 'launcher_pid': os.getpid(),
             'logical_session': logical_session, 'started_at_unix': time.time(), 'workers': []}
    worker_device = 'cpu' if args.scale == 'smoke' else 'cuda:0'
    try:
        for rank, gpu in enumerate(gpus):
            shared = ['--scale', args.scale, '--rank', str(rank), '--world_size', str(len(gpus)), '--device', worker_device,
                       '--seed', str(args.seed), '--output', str(args.output), '--target', args.target,
                       '--drafter', args.drafter, '--long_max_new_tokens', str(args.long_max_new_tokens)]
            if args.task == 'collect':
                command = [sys.executable, '-m', 'edraft.collect', '--stage', 'collect', *shared,
                           '--train_tokens_per_temperature', str(args.train_tokens_per_temperature),
                           '--standard_max_new_tokens', str(args.standard_max_new_tokens), '--partitions', *args.partitions]
            elif is_timing_task:
                command = [sys.executable, '-m', 'edraft.eval_closed_loop', '--stage', stage,
                           '--mode', 'timing', *shared, '--max_new_tokens', str(args.evaluation_max_new_tokens),
                           '--dense_checkpoint', str(args.output / 'm2/dense.pt'),
                           '--sparse_checkpoint', str(args.output / 'm2/dense_sparse.pt'),
                           '--projection', str(args.output / 'h_proj.pt'),
                           '--prompts_manifest', str(args.output / 'prompts.json'),
                           '--timing_repeats', str(args.timing_repeats),
                           '--timing_warmups', str(args.timing_warmups),
                           '--timing_prompts', str(args.timing_prompts)]
                if logical_session is not None:
                    command.extend(['--logical_session', logical_session])
                if args.fresh_session:
                    command.append('--fresh_session')
            else:
                command = [sys.executable, '-m', 'edraft.eval_closed_loop', '--stage', args.task,
                           '--mode', 'evaluate', *shared, '--max_new_tokens', str(args.evaluation_max_new_tokens),
                           '--dense_checkpoint', str(args.output / 'm2/dense.pt'),
                           '--sparse_checkpoint', str(args.output / 'm2/dense_sparse.pt'),
                           '--projection', str(args.output / 'h_proj.pt'),
                           '--prompts_manifest', str(args.output / 'prompts.json')]
            if args.allow_large_run:
                command.append('--allow_large_run')
            # Downloads must already be resolved once by the parent preparation step.
            env = dict(os.environ,
                       CUDA_VISIBLE_DEVICES='' if args.scale == 'smoke' else str(gpu),
                       RANK=str(rank), WORLD_SIZE=str(len(gpus)),
                       HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='1' if is_timing_task else '4')
            log_path = logdir / f'{log_label}-rank{rank}.log'
            log = log_path.open('a', buffering=1)
            streams.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            affinity = gpu_cpu_affinity(gpu)
            state['workers'].append({'rank': rank, 'physical_gpu': gpu, 'numa_node': affinity['numa_node'],
                                     'cpu_affinity': affinity['cpu_affinity'], 'pid': process.pid,
                                     'command': command, 'log': str(log_path),
                                     'status': 'running'})
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
                '--dense_checkpoint', str(args.output / 'm2/dense.pt'),
                '--sparse_checkpoint', str(args.output / 'm2/dense_sparse.pt'),
                '--projection', str(args.output / 'h_proj.pt'),
                '--prompts_manifest', str(args.output / 'prompts.json'),
                '--timing_repeats', str(args.timing_repeats),
                '--timing_warmups', str(args.timing_warmups),
                '--timing_prompts', str(args.timing_prompts),
            ]
            if args.allow_large_run:
                summarize_cmd.append('--allow_large_run')
            sum_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                           HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
            sum_proc = subprocess.run(summarize_cmd, cwd=ROOT, env=sum_env, capture_output=True, text=True, check=True)
            state['summarize_stdout'] = sum_proc.stdout.strip()
        state['completed_at_unix'] = time.time()
        atomic_json(state_path, state)
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
