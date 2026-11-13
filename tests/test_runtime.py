import argparse
import json

import pytest

from edraft.runtime import Manifest, atomic_json, common_parser, parse_config


def test_config_cli_override_and_smoke_never_downloads(tmp_path):
    path = tmp_path / 'config.json'
    atomic_json(path, {'scale': 'reference', 'seed': 7})
    args = parse_config(common_parser('test'), ['--config', str(path), '--scale', 'smoke', '--seed', '9'])
    assert args.seed == 9 and args.scale == 'smoke' and args.device == 'cpu'
    with pytest.raises(SystemExit):
        parse_config(common_parser('test'), ['--scale', 'smoke', '--allow_download'])
    with pytest.raises(SystemExit):
        parse_config(common_parser('test'), ['--scale', 'full'])


def test_manifest_requires_artifact_and_full_identity(tmp_path):
    manifest = Manifest(tmp_path / 'manifest.json')
    identity = {k: k for k in manifest.REQUIRED}
    result = tmp_path / 'result.json'
    assert not manifest.completed(identity)
    atomic_json(result, {'scale': 'synthetic'})
    manifest.finish(identity, result)
    assert Manifest(manifest.path).completed(identity)
    assert not manifest.completed({**identity, 'seed': 'another seed'})
    result.unlink()
    assert not manifest.completed(identity)
    with pytest.raises(ValueError):
        manifest.key({'seed': 1})
