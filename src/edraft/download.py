"""The sole explicit opt-in path for fetching reference model/data artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .runtime import ROOT, common_parser, parse_config, locked_json, atomic_json, record_run

METADATA = ['README.md', 'config.json', 'generation_config.json', 'tokenizer_config.json',
            'tokenizer.json', 'special_tokens_map.json', 'chat_template.jinja']


def fetch_model(repo_id, allow_download=False, metadata_only=False, revision=None):
    from huggingface_hub import HfApi, snapshot_download
    env_path = ROOT / 'artifacts/environment.json'
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
    with locked_json(env_path) as env:
        env.setdefault('downloads', []).append(item)
        env.setdefault('model_artifacts', {})[repo_id] = item
    return str(path)


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--metadata_only', action='store_true')
    args = parse_config(parser)
    record_run(args, 'download')
    if args.scale == 'smoke':
        print('Smoke uses built-in models; no downloads.')
        return
    for model in [args.target, args.drafter]:
        fetch_model(model, args.allow_download, args.metadata_only)


if __name__ == '__main__':
    main()
