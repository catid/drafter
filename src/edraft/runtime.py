"""Shared configuration, atomic run records, provenance, and resumable jobs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def locked_json(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(path.read_text()) if path.exists() else {}
        yield value
        atomic_json(path, value)


def git_commit(directory=ROOT):
    p = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else 'uncommitted'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_provenance():
    paths = sorted((ROOT / 'src/edraft').glob('*.py'))
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    status = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'], capture_output=True, text=True)
    return {'project_git_commit': git_commit(), 'git_dirty': bool(status.stdout.strip()),
            'source_tree_sha256': fingerprint(files), 'source_files_sha256': files}


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--scale', choices=['smoke', 'reference', 'full'], default='smoke')
    parser.add_argument('--allow_download', action='store_true')
    parser.add_argument('--allow_large_run', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target', default='Qwen/Qwen3-8B')
    parser.add_argument('--drafter', default='z-lab/Qwen3-8B-DFlash-b16')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    return parser


def parse_config(parser, argv=None):
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        config = json.loads(preliminary.config.read_text())
        unknown = set(config) - {a.dest for a in parser._actions}
        if unknown:
            parser.error(f'Unknown config keys: {sorted(unknown)}')
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    args.output = Path(args.output)
    if args.scale == 'full' and not args.allow_large_run:
        parser.error('Full scale requires --allow_large_run')
    if args.scale == 'smoke':
        args.device = 'cpu'
        if args.allow_download:
            parser.error('Smoke scale never downloads; omit --allow_download')
    return args


def resolved_config(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def record_run(args, name):
    config = resolved_config(args)
    record = {'scale': args.scale, 'seed': args.seed, 'config': config,
              **source_provenance(), 'started_at_unix': time.time(),
              'rank': int(os.environ.get('RANK', 0)), 'pid': os.getpid(),
              'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset (all visible)')}
    path = args.output / 'runs' / f'{name}-{fingerprint(config)[:16]}-rank{record["rank"]}.json'
    atomic_json(path, record)
    return record


def environment(config=None):
    import torch
    packages = {}
    for name in ['torch', 'transformers', 'accelerate', 'triton', 'datasets', 'dflash',
                 'duckdb', 'numpy', 'pyarrow', 'huggingface-hub']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # nvidia-smi inventory avoids creating a CUDA context on every visible GPU.
    # In particular, a collection worker must not reserve memory on other ranks.
    gpus = []
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,name,compute_cap,memory.free,memory.total',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if inventory.returncode == 0:
        for line in inventory.stdout.strip().splitlines():
            index, name, capability, free, total = [x.strip() for x in line.split(',')]
            gpus.append({'physical_index': int(index), 'name': name,
                         'compute_capability': [int(x) for x in capability.split('.')],
                         'free_bytes': int(free) * 2**20, 'total_bytes': int(total) * 2**20})
    with locked_json(ROOT / 'artifacts/environment.json') as env:
        env.update({'os': platform.platform(), 'python': platform.python_version(), 'packages': packages,
                    'cuda': torch.version.cuda, 'gpus': gpus,
                    'status': 'recorded', 'scale': (config or {}).get('scale', 'not_run'),
                    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'pytorch': torch.__version__, 'transformers': packages['transformers'],
                    'accelerate': packages['accelerate'], 'triton': packages['triton'],
                    'datasets': packages['datasets'],
                    'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset (all visible)'),
                    'project_git_commit': git_commit(), 'resolved_config': config,
                    'dflash': {'url': 'https://github.com/z-lab/dflash', 'commit': git_commit(ROOT / 'vendor/dflash')},
                    'free_disk_bytes': shutil.disk_usage(ROOT).free})
        identifiers = env.get('models', {})
        identifiers['revisions'] = {role: env.get('model_artifacts', {}).get(identifiers.get(role), {}).get('revision')
                                    for role in ['target', 'drafter']}
        env['models'] = identifiers
        env['tokenizer'] = {'identifier': identifiers.get('target'),
                            'revision': identifiers['revisions'].get('target')}
    return env


class Manifest:
    """One rank owns each file. Every entry is keyed by the full experiment identity."""
    REQUIRED = {'model_pair', 'prompt', 'condition', 'seed', 'temperature', 'verification_rule',
                'proposal_mode', 'checkpoint', 'adapter_version'}

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'jobs': {}}

    def key(self, identity):
        if not self.REQUIRED <= identity.keys():
            raise ValueError(f'Missing run identity: {self.REQUIRED - identity.keys()}')
        return fingerprint(identity)

    def completed(self, identity):
        item = self.data['jobs'].get(self.key(identity), {})
        return item.get('status') == 'complete' and Path(item.get('result_path', '')).is_file()

    def finish(self, identity, result_path):
        self.data['jobs'][self.key(identity)] = {'identity': identity, 'status': 'complete',
                                                'result_path': str(result_path)}
        atomic_json(self.path, self.data)


def preflight(args, expected_tokens):
    """Require current smoke evidence and record explicitly rough CPU-based estimate."""
    path = ROOT / 'artifacts/smoke.json'
    if not path.exists():
        raise RuntimeError('Run python -m edraft.smoke --scale smoke before a reference/full run')
    smoke = json.loads(path.read_text())
    if smoke.get('status') != 'passed':
        raise RuntimeError('Smoke gate has not passed')
    free = shutil.disk_usage(ROOT).free
    estimate = expected_tokens / smoke['tokens_per_second']
    record = {'scale': args.scale, 'expected_tokens': expected_tokens, 'smoke_tokens_per_second': smoke['tokens_per_second'],
              'cpu_smoke_extrapolation_seconds': estimate, 'free_disk_bytes': free,
              'caveat': 'CPU synthetic extrapolation is not a real-model throughput prediction; refine after pilot.'}
    with (ROOT / 'NOTES.md').open('a') as out:
        out.write('\nPre-run estimate: ' + json.dumps(record) + '\n')
    if free < 50 * 2**30:
        raise RuntimeError(f'Insufficient free space for reference artifacts: {free} bytes')
    return record
