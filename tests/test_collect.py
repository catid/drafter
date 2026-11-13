from collections import Counter
import copy
import json
from pathlib import Path

import pytest
import torch

from edraft.collect import (CollectionBackend, audit, calibration, check_reference_gates, collect,
                            job_identity, load_calibration, load_prompt_manifest, parser, planned_jobs,
                            prepare_args, synthetic_manifest)
from edraft.runtime import atomic_json, parse_config
from edraft.traces import read_shard


def args_for(path, *extra):
    return prepare_args(parse_config(parser(), ["--scale", "smoke", "--output", str(path), *extra]))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("smoke collection attempted a model download")
    monkeypatch.setattr("edraft.download.fetch_model", forbidden)
    args = args_for(tmp_path)
    backend = CollectionBackend(args)
    prompts = load_prompt_manifest(args)
    cal = calibration(args, prompts, backend)
    summary = collect(args, prompts, backend)
    return args, prompts, backend, cal, summary


def test_smoke_calibration_collection_and_complete_audit(completed):
    args, prompts, backend, cal, summary = completed
    assert summary["status"] == "passed" and summary["scale"] == "smoke"
    assert summary["primary_tokens_by_temperature"] == {0.: 32, .7: 32}
    assert summary["live_full_blocks_by_split"]["dev"] == 2
    assert summary["live_full_blocks_by_split"]["test"] == 2
    assert cal["calibration_prompts"] == 8
    assert all(id_ in prompts["calibration_ids"] for id_ in cal["identity"]["prompt_ids"])
    features = torch.load(args.output / "token_features.pt", weights_only=True)
    assert features["E"].shape == (8, 4)
    assert features["metadata"]["calibration_split"] == "train"
    assert features["metadata"]["calibration_prompt_ids"] == prompts["calibration_ids"]
    report = audit(args)
    assert report["status"] == "passed", report["requirements"]
    assert report["token_counts"]["primary_train_total"] == 64
    assert report["token_counts"]["standard_nonthinking/train/T=1.0"] == 64
    assert report["token_counts"]["long_thinking/test/T=0.6"] == 48
    assert len(report["head_reconstruction"]) == 4
    assert all(row["status"] == "passed" for row in report["head_reconstruction"])
    assert (args.output / "m1_audit.json").is_file()
    payload = read_shard(report["shards"][0])
    assert payload["metadata"]["generation"]["complete"] is True
    assert "diagnostics" in payload["blocks"][0]
    identity = payload["metadata"]["identity"]
    assert identity["projection_sha256"] == cal["artifact_sha256"]["h_proj.pt"]
    assert identity["token_features_sha256"] == cal["artifact_sha256"]["token_features.pt"]
    assert identity["prompt_manifest_sha256"] == cal["identity"]["prompt_manifest_sha256"]


def test_resume_uses_actual_completed_tokens_and_does_not_rewrite(completed):
    args, prompts, backend, cal, summary = completed
    paths = list((args.output / "traces").rglob("*.pt"))
    mtimes = {path: path.stat().st_mtime_ns for path in paths}
    cal_mtime = (args.output / "h_proj.pt").stat().st_mtime_ns
    assert calibration(args, prompts, backend) == cal
    again = collect(args, prompts, backend)
    assert again["primary_tokens_by_temperature"] == summary["primary_tokens_by_temperature"]
    assert again["completed_generations"] == summary["completed_generations"]
    assert all(path.stat().st_mtime_ns == old for path, old in mtimes.items())
    assert (args.output / "h_proj.pt").stat().st_mtime_ns == cal_mtime
    args.train_tokens_per_temperature = 64
    expanded = collect(args, prompts, backend)
    assert expanded["primary_tokens_by_temperature"] == {0.: 64, .7: 64}
    assert expanded["completed_generations"] == summary["completed_generations"] + 2


def test_resume_rechecks_published_but_unchecked_heldout_shard(completed):
    args, prompts, backend, _, _ = completed
    path = args.output / "head_reconstruction-rank0.json"
    path.unlink()
    collect(args, prompts, backend)
    checks = json.loads(path.read_text())
    assert len(checks) == 4 and all(row["status"] == "passed" for row in checks)


def test_projection_frozen_checksum_and_configuration_enforced(completed):
    args, prompts, backend, _, _ = completed
    changed = copy.copy(args)
    changed.seed += 1
    with pytest.raises(RuntimeError, match="does not match"):
        load_calibration(changed, prompts, backend)
    path = args.output / "h_proj.pt"
    with path.open("ab") as out:
        out.write(b"modified")
    with pytest.raises(RuntimeError, match="artifact changed"):
        load_calibration(args, prompts, backend)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        audit(args)


def test_prompt_sharding_disjoint_and_preserves_identical_job_seeds(tmp_path):
    args = args_for(tmp_path)
    prompts = synthetic_manifest(args)
    all_jobs = list(planned_jobs(args, prompts))
    expected = {(j["partition"], j["row"]["prompt_id"], j["temperature"]): j["seed"] for j in all_jobs}
    seen = {}
    for rank in range(3):
        worker = copy.copy(args)
        worker.rank, worker.world_size = rank, 3
        for job in planned_jobs(worker, prompts):
            key = (job["partition"], job["row"]["prompt_id"], job["temperature"])
            assert key not in seen
            seen[key] = job["seed"]
    assert seen == expected
    assert all(j["row"]["split"] == "train" and j["temperature"] in {0., .7}
               for j in all_jobs if j["partition"] == "train_primary")
    assert all(j["temperature"] == 1 for j in all_jobs if j["partition"] == "diagnostic")
    assert all(j["temperature"] > 0 for j in all_jobs if j["partition"] == "long")


def test_nonselected_test_and_long_reserves_are_excluded(tmp_path):
    args = args_for(tmp_path)
    prompts = synthetic_manifest(args)
    prompts["prompts"].extend([
        {"prompt_id": "reserve-standard", "condition": "standard_nonthinking", "split": "test", "reference_selected": False},
        {"prompt_id": "reserve-long", "condition": "long_thinking", "split": "test", "reference_selected": False}])
    assert not any(j["row"]["prompt_id"].startswith("reserve") for j in planned_jobs(args, prompts))


def test_rank_quota_exhaustion_is_explicit_blocker(tmp_path):
    args = args_for(tmp_path, "--partitions", "train_primary", "--train_tokens_per_temperature", "10000")
    backend = CollectionBackend(args)
    prompts = synthetic_manifest(args)
    calibration(args, prompts, backend)
    with pytest.raises(RuntimeError, match="exhausted"):
        collect(args, prompts, backend)
    report = json.loads((tmp_path / "collection-rank0.json").read_text())
    assert report["status"] == "blocked"
    assert report["primary_tokens_by_temperature"] == {"0.0": 384, "0.7": 384}


def test_audit_excludes_superseded_and_duplicate_identities(completed):
    args, _, _, _, _ = completed
    manifest_path = args.output / "traces" / "manifest-rank0.json"
    manifest = json.loads(manifest_path.read_text())
    # A copied rank manifest simulates changing process count during resume.
    atomic_json(args.output / "traces" / "manifest-rank1.json", manifest)
    # A superseded job must not credit tokens, even if its result file is missing.
    old = copy.deepcopy(next(iter(manifest["jobs"].values())))
    old["identity"]["projection_sha256"] = "superseded"
    old["result_path"] = str(args.output / "missing-old-projection.pt")
    atomic_json(args.output / "traces" / "manifest-rank2.json", {"jobs": {"old": old}})
    report = audit(args)
    assert report["status"] == "passed"
    assert report["token_counts"]["primary_train_total"] == 64
    reasons = Counter(item["reason"] for item in report["excluded_shards"])
    assert reasons["duplicate_job_identity"] == len(manifest["jobs"])
    assert reasons["superseded_or_unselected_identity"] == 1


def test_reference_m0_gate_blocks_before_loading_models(tmp_path):
    args = args_for(tmp_path)
    args.scale = "reference"
    with pytest.raises(RuntimeError, match="M0 reproduction"):
        check_reference_gates(args)


def test_calibration_rejects_evaluation_leakage(tmp_path):
    args = args_for(tmp_path)
    manifest = synthetic_manifest(args)
    manifest["calibration_ids"][0] = "synthetic-test-0"
    # Reference path reads provided manifests; validation happens before model IO.
    args.scale = "reference"
    args.prompt_manifest = tmp_path / "prompts.json"
    atomic_json(args.prompt_manifest, manifest)
    with pytest.raises(ValueError, match="only designated"):
        load_prompt_manifest(args)
