"""The sole explicit opt-in path for fetching reference model/data artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .runtime import ROOT, common_parser, parse_config, locked_json, atomic_json, record_run, resolve_artifact_path

METADATA = ['README.md', 'config.json', 'generation_config.json', 'tokenizer_config.json',
            'tokenizer.json', 'special_tokens_map.json', 'chat_template.jinja']


def _write_env_path(read_env_path: Path, output_dir: Path | str | None) -> Path:
    if output_dir is None:
        return read_env_path
    out_p = Path(output_dir)
    if out_p.resolve() == (ROOT / 'artifacts').resolve():
        return ROOT / 'artifacts/environment.json'
    return out_p / 'environment.json'


def fetch_model(repo_id, allow_download=False, metadata_only=False, revision=None, *, output_dir=None):
    from huggingface_hub import HfApi, snapshot_download
    env_path = resolve_artifact_path('environment.json', output_dir, root=ROOT)
    prior = json.loads(env_path.read_text())
    pinned = prior.get('model_artifacts', {}).get(repo_id, {})
    revision = revision or pinned.get('revision')
    if not allow_download:
        try:
            return snapshot_download(repo_id, revision=revision, local_files_only=True,
                                     allow_patterns=METADATA if metadata_only else None)
        except Exception as error:
            raise RuntimeError(f'blocked: {repo_id} is not cached; explicit --allow_download required') from error
    info = HfApi().model_info(repo_id, revision=revision or 'main', files_metadata=True)
    sha = info.sha
    print(f'Authorized download: https://huggingface.co/{repo_id}/tree/{sha} metadata_only={metadata_only}', flush=True)
    started = time.time()
    path = Path(snapshot_download(repo_id, revision=sha, allow_patterns=METADATA if metadata_only else None))
    files = []
    for p in sorted(path.rglob('*')):
        if not p.is_file():
            continue
        relative = str(p.relative_to(path))
        sibling = next((x for x in info.siblings if x.rfilename == relative), None)
        lfs = sibling.lfs if sibling is not None else None
        checksum = getattr(lfs, 'sha256', None)
        if not checksum and p.stat().st_size < 100_000_000:
            checksum = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': checksum,
                      'url': f'https://huggingface.co/{repo_id}/resolve/{sha}/{relative}'})
    item = {'repo_id': repo_id, 'revision': sha, 'path': str(path), 'metadata_only': metadata_only,
            'url': f'https://huggingface.co/{repo_id}/tree/{sha}', 'files': files,
            'disk_usage_bytes': sum(x['bytes'] for x in files), 'seconds': time.time() - started}
    write_target = _write_env_path(env_path, output_dir)
    with locked_json(write_target) as env:
        if not env and prior:
            env.update(prior)
        env.setdefault('downloads', []).append(item)
        env.setdefault('model_artifacts', {})[repo_id] = item
    return str(path)


def fetch_dataset(repo_id, allow_download=False, revision=None, *, output_dir=None):
    """Pin benchmark source files; callers parse only local files afterward."""
    from huggingface_hub import HfApi, snapshot_download
    env_path = resolve_artifact_path('environment.json', output_dir, root=ROOT)
    prior = json.loads(env_path.read_text())
    pinned = prior.get('dataset_artifacts', {}).get(repo_id, {})
    revision = revision or pinned.get('revision')
    patterns = ['*.parquet', '*.json', '*.jsonl', 'README.md']
    if not allow_download:
        try:
            return snapshot_download(repo_id, repo_type='dataset', revision=revision,
                                     local_files_only=True, allow_patterns=patterns)
        except Exception as error:
            raise RuntimeError(f'blocked: dataset {repo_id} is not cached; --allow_download required') from error
    info = HfApi().dataset_info(repo_id, revision=revision or 'main', files_metadata=True)
    print(f'Authorized dataset download: https://huggingface.co/datasets/{repo_id}/tree/{info.sha}', flush=True)
    path = Path(snapshot_download(repo_id, repo_type='dataset', revision=info.sha, allow_patterns=patterns))
    files = []
    for p in sorted(path.rglob('*')):
        if p.is_file():
            relative = str(p.relative_to(path))
            files.append({'path': relative, 'bytes': p.stat().st_size,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                          'url': f'https://huggingface.co/datasets/{repo_id}/resolve/{info.sha}/{relative}'})
    item = {'repo_id': repo_id, 'repo_type': 'dataset', 'revision': info.sha, 'path': str(path),
            'disk_usage_bytes': sum(f['bytes'] for f in files), 'files': files}
    write_target = _write_env_path(env_path, output_dir)
    with locked_json(write_target) as env:
        if not env and prior:
            env.update(prior)
        env.setdefault('downloads', []).append(item)
        env.setdefault('dataset_artifacts', {})[repo_id] = item
    return str(path)


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--metadata_only', action='store_true')
    parser.add_argument('--datasets', nargs='*', default=[])
    args = parse_config(parser)
    record_run(args, 'download')
    if args.scale == 'smoke':
        print('Smoke uses built-in models; no downloads.')
        return
    if args.datasets:
        for dataset in args.datasets:
            fetch_dataset(dataset, args.allow_download, output_dir=args.output)
    else:
        for model in [args.target, args.drafter]:
            fetch_model(model, args.allow_download, args.metadata_only, output_dir=args.output)


if __name__ == '__main__':
    main()
