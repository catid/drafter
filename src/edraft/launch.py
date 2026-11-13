"""Launch resumable collection or acceptance evaluation, one process per GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

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


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--task', choices=['collect', 'm3', 'm4'], default='collect')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--partitions', nargs='+', default=['train_primary', 'standard_eval', 'diagnostic', 'long'],
                        choices=['train_primary', 'standard_eval', 'diagnostic', 'long'])
    parser.add_argument('--train_tokens_per_temperature', type=int)
    parser.add_argument('--standard_max_new_tokens', type=int, default=1024)
    parser.add_argument('--long_max_new_tokens', type=int, default=24576)
    parser.add_argument('--evaluation_max_new_tokens', type=int, default=2048)
    args = parse_config(parser)
    if args.train_tokens_per_temperature is None:
        args.train_tokens_per_temperature = 1000000 if args.scale == 'full' else 250000
    record_run(args, 'launch')
    if args.scale == 'smoke':
        from .smoke import run_smoke
        return run_smoke(args)
    gpus = [int(g) for g in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('--gpus requires distinct physical indices')
    selected_gpus_idle(gpus)
    logdir = args.output / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    processes, streams = [], []
    label = 'collection' if args.task == 'collect' else args.task + '-evaluation'
    log_label = 'collect' if args.task == 'collect' else args.task + '-evaluate'
    state_path = args.output / f'{label}-processes.json'
    state = {'scale': args.scale, 'task': args.task, 'launcher_pid': os.getpid(),
             'started_at_unix': time.time(), 'workers': []}
    try:
        for rank, gpu in enumerate(gpus):
            shared = ['--scale', args.scale, '--rank', str(rank), '--world_size', str(len(gpus)), '--device', 'cuda:0',
                       '--seed', str(args.seed), '--output', str(args.output), '--target', args.target,
                       '--drafter', args.drafter, '--long_max_new_tokens', str(args.long_max_new_tokens)]
            if args.task == 'collect':
                command = [sys.executable, '-m', 'edraft.collect', '--stage', 'collect', *shared,
                           '--train_tokens_per_temperature', str(args.train_tokens_per_temperature),
                           '--standard_max_new_tokens', str(args.standard_max_new_tokens), '--partitions', *args.partitions]
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
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), RANK=str(rank), WORLD_SIZE=str(len(gpus)),
                       HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4')
            log_path = logdir / f'{log_label}-rank{rank}.log'
            log = log_path.open('a', buffering=1)
            streams.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            state['workers'].append({'rank': rank, 'physical_gpu': gpu, 'pid': process.pid,
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
            time.sleep(5)
        for row, process in zip(state['workers'], processes):
            row.update({'status': 'complete' if process.returncode == 0 else 'failed',
                        'returncode': process.returncode})
        state['completed_at_unix'] = time.time()
        atomic_json(state_path, state)
        if any(p.returncode for p in processes):
            raise RuntimeError('Run failed; inspect worker logs')
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
