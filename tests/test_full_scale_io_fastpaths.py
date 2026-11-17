from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

import edraft.collect as collect_mod
import edraft.diagnose as diagnose_mod
import edraft.traces as traces_mod
from edraft.collect import (
    CollectionBackend,
    _find_peer_completed_shard,
    _globally_completed_trace_map,
    _maybe_apply_cuda_cpu_affinity,
    audit,
    calibration,
    check_reference_gates,
    collect,
    load_prompt_manifest,
    parser as collect_parser,
    prepare_args,
)
from edraft.dflash_loop import generate_toy
from edraft.diagnose import accumulate_trace_shards, audited_trace_sources, live_block_diagnostics
from edraft.runtime import WorkQueue, atomic_json, fingerprint, parse_config
from edraft.sampler import SamplingWarper
from edraft.toy import ToyParallelDrafter, ToyTarget
from edraft.traces import TraceShardWriter, compact_block, read_shard, validate_block, write_shard


def _make_live_fixture(seed: int = 77):
    target = ToyTarget(vocab_size=16)
    drafter = ToyParallelDrafter(target)
    generated = generate_toy(
        target,
        drafter,
        [2, 5, 8, 1],
        8,
        block_length=4,
        temperature=0.7,
        warper=SamplingWarper(top_k=3, top_p=0.8),
        generator=torch.Generator().manual_seed(seed),
        retain_tensors=True,
    )
    metadata = {
        "prompt_id": f"fixture-{seed}",
        "split": "test",
        "condition": "standard_nonthinking",
        "seed": seed,
        "scale": "synthetic",
        "model_pair": ["toy-target-v1", "toy-drafter-v1"],
        "projection_metadata": {
            "artifact_id": "fixture-identity-projection",
            "frozen": True,
            "calibration_split": "train",
            "calibration_prompt_ids": ["fixture-train"],
        },
        "history_depth": 4,
    }
    return generated.blocks[0], metadata, drafter


def _write_valid_m0_bundle(root_dir: Path, *, m0_scale: str = "reference", mismatched_tokens: int = 0) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    m0_dir = root_dir / "m0"
    m0_dir.mkdir(parents=True, exist_ok=True)
    revs = {
        "Qwen/Qwen3-8B": "pinned-target-rev",
        "z-lab/Qwen3-8B-DFlash-b16": "pinned-drafter-rev",
    }
    atomic_json(
        root_dir / "environment.json",
        {
            "status": "measured",
            "model_artifacts": {k: {"revision": v} for k, v in revs.items()},
        },
    )
    atomic_json(
        root_dir / "m0.json",
        {
            "status": "passed",
            "scale": m0_scale,
            "num_distinct_prompts": 20,
            "comparisons": {"upstream_local": {"mismatched_tokens": mismatched_tokens}},
        },
    )
    atomic_json(
        root_dir / "m0_numerics.json",
        {
            "status": "passed",
            "head_reconstruction": {"passed": True},
            "findings": [],
            "model_revisions": revs,
        },
    )
    for idx in range(20):
        atomic_json(
            m0_dir / f"prompt{idx:02d}.json",
            {
                "identity": {"prompt": f"p{idx}"},
                "local_target": {"sequence_equal": True},
            },
        )


def test_compact_block_trust_l_unit_and_default_safety(monkeypatch):
    record, metadata, _ = _make_live_fixture()
    assert isinstance(record.get("l_unit"), torch.Tensor)

    ref = compact_block(record, metadata, M_draft=3, M_target=2, special_tokens=[0, 15], retain_live_full=True)

    log_softmax_calls = 0
    orig_log_softmax = torch.Tensor.log_softmax

    def counting_log_softmax(self, *args, **kwargs):
        nonlocal log_softmax_calls
        log_softmax_calls += 1
        return orig_log_softmax(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "log_softmax", counting_log_softmax)
    fast = compact_block(
        record,
        metadata,
        M_draft=3,
        M_target=2,
        special_tokens=[0, 15],
        retain_live_full=True,
        _trust_l_unit=True,
    )
    assert log_softmax_calls == 0

    for field in ("h_raw_bf16", "h", "target_rest_mass", "draft_log_normalizer", "target_log_normalizer"):
        assert torch.equal(fast[field], ref[field])
        assert fast[field].storage_offset() == 0
    for field in ("union_ids", "draft_l_unit_union", "target_probs_union"):
        assert all(torch.equal(a, b) for a, b in zip(fast[field], ref[field]))
    assert torch.equal(fast["live_full"]["draft_l_unit"], ref["live_full"]["draft_l_unit"])
    assert torch.equal(fast["live_full"]["target_probs"], ref["live_full"]["target_probs"])

    # Default `_trust_l_unit=False` never trusts stale `record["l_unit"]` when `draft_logits` is modified.
    scaled_logits = record["draft_logits"] * 1.7
    mutated = dict(record, draft_logits=scaled_logits)
    out_mutated = compact_block(mutated, metadata, M_draft=3, M_target=2, special_tokens=[0, 15])
    expected_lu = scaled_logits.float().log_softmax(-1)
    for i, ids in enumerate(out_mutated["union_ids"]):
        assert torch.equal(out_mutated["draft_l_unit_union"][i], expected_lu[i, ids])
    torch.testing.assert_close(out_mutated["draft_log_normalizer"], scaled_logits.float().logsumexp(-1))

    # `_trust_l_unit=True` still rejects non-finite `l_unit` and falls back when `l_unit` is None/mismatched.
    bad_lu = record["l_unit"].clone()
    bad_lu[0, 0] = -torch.inf
    with pytest.raises(ValueError, match="finite and unwarped"):
        compact_block(dict(record, l_unit=bad_lu), metadata, _trust_l_unit=True)

    fallback = compact_block(dict(record, l_unit=None), metadata, M_draft=3, M_target=2, _trust_l_unit=True)
    assert torch.equal(fallback["draft_log_normalizer"], ref["draft_log_normalizer"])


def test_hashing_writer_streaming_and_seek_fallback(tmp_path, monkeypatch):
    record, metadata, _ = _make_live_fixture()
    block = compact_block(record, metadata, M_draft=3, M_target=2)

    sha_calls = 0
    orig_sha = traces_mod._sha256

    def spy_sha(p):
        nonlocal sha_calls
        sha_calls += 1
        return orig_sha(p)

    monkeypatch.setattr(traces_mod, "_sha256", spy_sha)

    shard_a = tmp_path / "streamed.pt"
    summary_a = write_shard(shard_a, [block], {"rank": 0})
    assert sha_calls == 0
    disk_sha_a = hashlib.sha256(shard_a.read_bytes()).hexdigest()
    assert summary_a["sha256"] == disk_sha_a
    assert summary_a["disk_bytes"] == shard_a.stat().st_size

    # Force a seek during save to verify `_HashingWriter` falls back cleanly to `_sha256(path)`.
    orig_save = torch.save

    def seeking_save(obj, f, *args, **kwargs):
        orig_save(obj, f, *args, **kwargs)
        pos = f.tell()
        f.seek(pos)

    monkeypatch.setattr(torch, "save", seeking_save)
    shard_b = tmp_path / "seek_fallback.pt"
    summary_b = write_shard(shard_b, [block], {"rank": 0})
    assert sha_calls == 1
    assert summary_b["sha256"] == hashlib.sha256(shard_b.read_bytes()).hexdigest()


def test_write_and_read_shard_prevalidated_and_validate_flags(tmp_path, monkeypatch):
    record, metadata, _ = _make_live_fixture()
    val_calls = 0
    orig_validate = traces_mod.validate_block

    def spy_validate(blk, *args, **kwargs):
        nonlocal val_calls
        val_calls += 1
        return orig_validate(blk, *args, **kwargs)

    monkeypatch.setattr(traces_mod, "validate_block", spy_validate)
    block = compact_block(record, metadata, M_draft=3, M_target=2)
    assert val_calls == 1

    p_fast = tmp_path / "fast.pt"
    p_std = tmp_path / "std.pt"
    write_shard(p_fast, [block], {"rank": 0}, _prevalidated_cpu=True)
    assert val_calls == 1  # Not re-validated when `_prevalidated_cpu=True`!
    write_shard(p_std, [block], {"rank": 0}, _prevalidated_cpu=False)
    assert val_calls == 2

    sha_calls = 0
    orig_sha = traces_mod._sha256

    def spy_sha(p):
        nonlocal sha_calls
        sha_calls += 1
        return orig_sha(p)

    monkeypatch.setattr(traces_mod, "_sha256", spy_sha)

    loaded_fast = read_shard(p_fast, verify_checksum=False, validate=False)
    assert sha_calls == 0
    assert val_calls == 2
    loaded_std = read_shard(p_std)
    assert sha_calls == 1
    assert val_calls == 3
    for field in ("h_raw_bf16", "h", "target_rest_mass", "draft_log_normalizer", "target_log_normalizer"):
        assert torch.equal(loaded_fast["blocks"][0][field], loaded_std["blocks"][0][field])

    # Default `read_shard` still rejects corrupted bytes.
    raw = bytearray(p_fast.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    p_fast.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_shard(p_fast)


def test_check_reference_gates_isolated_output_and_strict_tmp_block(tmp_path, monkeypatch):
    monkeypatch.delenv("EDRAFT_ARTIFACTS_ROOT", raising=False)
    empty_out = tmp_path / "isolated_tmp"
    args_ref = prepare_args(parse_config(collect_parser(), ["--scale", "reference", "--output", str(empty_out)]))
    with pytest.raises(RuntimeError, match="M0 reproduction"):
        check_reference_gates(args_ref)

    m0_store = tmp_path / "shared_m0"
    _write_valid_m0_bundle(m0_store, m0_scale="reference", mismatched_tokens=0)
    monkeypatch.setenv("EDRAFT_ARTIFACTS_ROOT", str(m0_store))

    check_reference_gates(args_ref)
    args_full = prepare_args(
        parse_config(collect_parser(), ["--scale", "full", "--allow_large_run", "--output", str(empty_out)])
    )
    check_reference_gates(args_full)

    # `--scale full` accepts M0 at scale `full`, whereas `--scale reference` requires `reference`.
    _write_valid_m0_bundle(m0_store, m0_scale="full", mismatched_tokens=0)
    check_reference_gates(args_full)
    with pytest.raises(RuntimeError, match="token reproduction gate"):
        check_reference_gates(args_ref)

    # Local failing `m0.json` inside `args.output` is never bypassed by fallback.
    _write_valid_m0_bundle(m0_store, m0_scale="reference", mismatched_tokens=0)
    _write_valid_m0_bundle(empty_out, m0_scale="reference", mismatched_tokens=3)
    with pytest.raises(RuntimeError, match="token reproduction gate"):
        check_reference_gates(args_ref)


def _assert_nested_close(actual, expected, *, rel: float = 1e-12, abs_tol: float = 1e-12) -> None:
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=rel, abs=abs_tol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            _assert_nested_close(actual[k], expected[k], rel=rel, abs_tol=abs_tol)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_nested_close(a, b, rel=rel, abs_tol=abs_tol)
    else:
        assert actual == expected


def test_collect_data_label_by_scale(tmp_path, monkeypatch):
    m0_store = tmp_path / "m0_ok"
    _write_valid_m0_bundle(m0_store, m0_scale="reference", mismatched_tokens=0)
    monkeypatch.setenv("EDRAFT_ARTIFACTS_ROOT", str(m0_store))

    for scale, expected_data in (("smoke", "synthetic"), ("reference", "reference"), ("full", "full")):
        out_dir = tmp_path / f"run_{scale}"
        cal_prompts = 8 if scale == "smoke" else 64
        cli = [
            "--scale",
            scale,
            "--device",
            "cpu",
            "--output",
            str(out_dir),
            "--partitions",
            "train_primary",
            "--train_tokens_per_temperature",
            "16",
            "--standard_max_new_tokens",
            "16",
            "--calibration_max_new_tokens",
            "4",
            "--calibration_prompts",
            str(cal_prompts),
            "--projection_dim",
            "6",
            "--feature_rank",
            "4",
        ]
        if scale == "full":
            cli.append("--allow_large_run")
        args = prepare_args(parse_config(collect_parser(), cli))
        prompts = collect_mod.synthetic_manifest(args)
        if cal_prompts > len(prompts["calibration_ids"]):
            extra_rows = [
                {
                    "prompt_id": f"synthetic-train-{i}",
                    "dataset": "builtin_synthetic",
                    "split": "train",
                    "condition": "standard_nonthinking",
                    "reference_selected": True,
                    "text": f"Synthetic train {i}",
                }
                for i in range(12, cal_prompts)
            ]
            prompts["prompts"].extend(extra_rows)
            prompts["calibration_ids"] = [f"synthetic-train-{i}" for i in range(cal_prompts)]
        prompts["scale"] = scale
        atomic_json(args.prompt_manifest, prompts)
        smoke_args = copy.copy(args)
        smoke_args.scale = "smoke"
        backend = CollectionBackend(smoke_args)
        calibration(args, prompts, backend)
        summary = collect(args, prompts, backend)
        assert summary["status"] == "passed"
        sidecars = sorted((out_dir / "traces" / "rank0").glob("*.pt.json"))
        assert sidecars
        for sc_path in sidecars:
            gen = json.loads(sc_path.read_text())["generation"]
            assert gen["scale"] == scale
            assert gen["data"] == expected_data


def test_dynamic_queue_zero_redundant_sha256_and_peer_crash_recovery(tmp_path, monkeypatch):
    out = tmp_path / "dyn_collect"
    collect_mod.main(["--scale", "smoke", "--output", str(out), "--stage", "calibrate"])

    trace_sha_calls = 0
    orig_sha = collect_mod.sha256

    def spy_sha(path):
        nonlocal trace_sha_calls
        p = Path(path)
        if p.suffix == ".pt" and p.parent.name.startswith("rank"):
            trace_sha_calls += 1
        return orig_sha(path)

    monkeypatch.setattr(collect_mod, "sha256", spy_sha)

    # Rank 0 fresh run on `--schedule dynamic` claims all dynamic jobs with 0 trace-shard `sha256` calls!
    collect_mod.main(
        ["--scale", "smoke", "--output", str(out), "--stage", "collect", "--rank", "0", "--world_size", "2", "--schedule", "dynamic"]
    )
    assert trace_sha_calls == 0

    # Simulate peer crash window:
    # Remove `work_queue.json` and `head_reconstruction-rank0.json`, AND hide `manifest-rank0.json` only during
    # startup `_globally_completed_trace_map` so rank 1 claims rank 0's published shards via `_find_peer_completed_shard`.
    (out / "traces" / "work_queue.json").unlink()
    for hp in out.glob("head_reconstruction-rank*.json"):
        hp.unlink()

    orig_global_map = collect_mod._globally_completed_trace_map
    monkeypatch.setattr(collect_mod, "_globally_completed_trace_map", lambda *a, **kw: {})
    trace_sha_calls = 0
    collect_mod.main(
        ["--scale", "smoke", "--output", str(out), "--stage", "collect", "--rank", "1", "--world_size", "2", "--schedule", "dynamic"]
    )
    monkeypatch.setattr(collect_mod, "_globally_completed_trace_map", orig_global_map)

    rank0_manifest = json.loads((out / "traces" / "manifest-rank0.json").read_text())
    dynamic_peer_count = sum(
        1
        for item in rank0_manifest["jobs"].values()
        if json.loads(Path(item["result_path"]).with_suffix(".pt.json").read_text())["generation"]["partition"] != "train_primary"
    )
    # `_find_peer_completed_shard` hashed each peer dynamic shard AT MOST ONCE (`O(N)`, never `O(N^2)`!).
    assert trace_sha_calls == dynamic_peer_count

    trace_sha_calls = 0
    collect_mod.main(["--scale", "smoke", "--output", str(out), "--stage", "audit"])
    audit_rep = json.loads((out / "m1_audit.json").read_text())
    assert audit_rep["status"] == "passed"
    assert audit_rep["requirements"]["serialized_head_reconstruction"] is True


def test_global_trace_map_prefers_valid_duplicate_over_bit_flipped_peer(tmp_path):
    record, metadata, _ = _make_live_fixture()
    block = compact_block(record, metadata)
    identity = {
        "model_pair": metadata["model_pair"],
        "prompt": metadata["prompt_id"],
        "condition": metadata["condition"],
        "seed": metadata["seed"],
        "temperature": record["collection_temperature"],
        "verification_rule": record["verification_rule"],
        "proposal_mode": record["proposal_mode"],
        "checkpoint": "frozen-toy-head-v1",
        "adapter_version": record["adapter_version"],
        "sampling_warp_config": record["sampling_warp_config"],
    }
    traces_dir = tmp_path / "traces"
    w0 = TraceShardWriter(traces_dir, rank=0)
    w1 = TraceShardWriter(traces_dir, rank=1)
    p0 = w0.write(identity, [block])
    p1 = w1.write(identity, [block])

    # Flip 1 bit in-place in `p0` (preserving `disk_bytes`).
    raw0 = bytearray(p0.read_bytes())
    raw0[len(raw0) // 2] ^= 0x04
    p0.write_bytes(raw0)

    key = fingerprint(identity)
    done = _globally_completed_trace_map(traces_dir)
    assert done[key] == p1
    assert _find_peer_completed_shard(traces_dir, key) == p1


def test_parallel_audit_parity_and_in_place_corruption_detection(tmp_path):
    out = tmp_path / "audit_check"
    collect_mod.main(["--scale", "smoke", "--output", str(out), "--stage", "calibrate"])
    collect_mod.main(["--scale", "smoke", "--output", str(out), "--stage", "collect"])
    args = prepare_args(parse_config(collect_parser(), ["--scale", "smoke", "--output", str(out)]))
    rep = audit(args)
    assert rep["status"] == "passed"
    assert len(rep["shards"]) > 1

    # Flip 1 bit in-place (preserving `disk_bytes`) and confirm parallel `audit` blocks.
    victim = Path(rep["shards"][-1])
    raw = bytearray(victim.read_bytes())
    raw[len(raw) // 2] ^= 0x02
    victim.write_bytes(raw)
    with pytest.raises(RuntimeError, match="corrupted committed trace shard"):
        audit(args)


def test_cuda_cpu_affinity_helper_cpu_and_mocked_cuda(monkeypatch):
    cpu_args = prepare_args(parse_config(collect_parser(), ["--scale", "smoke", "--device", "cpu"]))
    assert _maybe_apply_cuda_cpu_affinity(cpu_args) is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cuda_args = prepare_args(
        parse_config(collect_parser(), ["--scale", "reference", "--device", "cuda:3", "--rank", "3", "--world_size", "8"])
    )
    info = _maybe_apply_cuda_cpu_affinity(cuda_args)
    assert info is not None
    assert info["cpu_affinity_applied"] is False
    assert info["physical_gpu"] == 3
    assert len(info["cpu_affinity"]) > 0


def test_diagnose_parallel_audit_sources_and_single_pass_main(tmp_path, monkeypatch):
    out = tmp_path / "m1_to_m1b"
    out.mkdir(parents=True, exist_ok=True)
    smoke_args = prepare_args(parse_config(collect_parser(), ["--scale", "smoke", "--output", str(out)]))
    prompts_manifest = collect_mod.synthetic_manifest(smoke_args)
    atomic_json(out / "prompts.json", prompts_manifest)

    collect_mod.main(["--scale", "smoke", "--output", str(out), "--stage", "calibrate"])
    collect_mod.main(["--scale", "smoke", "--output", str(out), "--stage", "collect"])
    collect_mod.main(["--scale", "smoke", "--output", str(out), "--stage", "audit"])

    # Promote the smoke M1 bundle labels to `reference` so `diagnose.main` exercises the full audited shard pipeline.
    audit_path = out / "m1_audit.json"
    audit_rep = json.loads(audit_path.read_text())
    audit_rep["scale"] = "reference"
    prompt_bytes_sha = hashlib.sha256((out / "prompts.json").read_bytes()).hexdigest()
    audit_rep["calibration"]["identity"]["prompt_manifest_sha256"] = prompt_bytes_sha
    updated_sources = []
    for src in audit_rep["source_shards"]:
        p = Path(src["path"])
        payload = read_shard(p)
        for blk in payload["blocks"]:
            blk["scale"] = "reference"
        payload["metadata"]["generation"]["scale"] = "reference"
        summary = write_shard(p, payload["blocks"], payload["metadata"])
        side = json.loads(p.with_suffix(".pt.json").read_text())
        side["identity"] = payload["metadata"]["identity"]
        atomic_json(p.with_suffix(".pt.json"), side)
        updated_sources.append(
            {
                "path": str(p.resolve()),
                "sha256": summary["sha256"],
                "identity_sha256": fingerprint(payload["metadata"]["identity"]),
            }
        )
    audit_rep["source_shards"] = updated_sources
    audit_rep["shards"] = [s["path"] for s in updated_sources]
    atomic_json(audit_path, audit_rep)

    accepted, provenance = audited_trace_sources(audit_path, "reference", out / "traces")
    assert len(accepted) == len(updated_sources) > 1
    assert len(provenance["source_set_sha256"]) == 64

    ref_acc = accumulate_trace_shards(accepted, "reference", workers=1, preverified_audit=False)
    ref_report = ref_acc.report()

    # Verify `diagnose.main` with `--workers 1` calls `traces._sha256` ZERO times (single-pass read).
    trace_sha_calls = 0
    orig_trace_sha = traces_mod._sha256

    def spy_trace_sha(p):
        nonlocal trace_sha_calls
        trace_sha_calls += 1
        return orig_trace_sha(p)

    monkeypatch.setattr(traces_mod, "_sha256", spy_trace_sha)
    out_w1 = tmp_path / "diag_w1"
    diagnose_mod.main(
        [
            "--scale",
            "reference",
            "--device",
            "cpu",
            "--output",
            str(out_w1),
            "--m1_audit",
            str(audit_path),
            "--trace_dir",
            str(out / "traces"),
            "--workers",
            "1",
        ]
    )
    assert trace_sha_calls == 0
    rep_w1 = json.loads((out_w1 / "diagnostic_reference.json").read_text())
    assert rep_w1["groups"] == ref_report["groups"]
    assert rep_w1["generation_lengths"] == ref_report["generation_lengths"]

    # Verify `diagnose.main` with `--workers` omitted (auto-scaling multi-process workers + `preverified_audit=True`).
    out_auto = tmp_path / "diag_auto"
    diagnose_mod.main(
        [
            "--scale",
            "reference",
            "--device",
            "cpu",
            "--output",
            str(out_auto),
            "--m1_audit",
            str(audit_path),
            "--trace_dir",
            str(out / "traces"),
        ]
    )
    rep_auto = json.loads((out_auto / "diagnostic_reference.json").read_text())
    _assert_nested_close(rep_auto["groups"], ref_report["groups"])
    _assert_nested_close(rep_auto["generation_lengths"], ref_report["generation_lengths"])

    # Tampering a shard + updating its `.pt.json` sidecar still fails parallel `audited_trace_sources`.
    tampered = Path(updated_sources[-1]["path"])
    raw = bytearray(tampered.read_bytes())
    raw[len(raw) // 2] ^= 0x08
    tampered.write_bytes(raw)
    sc = json.loads(tampered.with_suffix(".pt.json").read_text())
    sc["sha256"] = hashlib.sha256(raw).hexdigest()
    atomic_json(tampered.with_suffix(".pt.json"), sc)
    with pytest.raises(RuntimeError, match="Audited trace checksum changed"):
        audited_trace_sources(audit_path, "reference", out / "traces")


def test_io_bug_1_hashing_writer_typed_and_partial_writes():
    import array
    import io
    from edraft.traces import _HashingWriter

    class _PartialTypedRawWriter(io.RawIOBase):
        def __init__(self, *, max_chunk_bytes: int | None = None, return_none: bool = False):
            self.buf = bytearray()
            self.max_chunk_bytes = max_chunk_bytes
            self.return_none = return_none

        def writable(self) -> bool:
            return True

        def write(self, b) -> int | None:
            raw_bytes = memoryview(b).tobytes()
            if self.return_none:
                self.buf.extend(raw_bytes)
                return None
            take = len(raw_bytes) if self.max_chunk_bytes is None else min(len(raw_bytes), self.max_chunk_bytes)
            self.buf.extend(raw_bytes[:take])
            return take

    arr = array.array("I", [0x11223344, 0x55667788, 0x99AABBCC])
    mv_typed = memoryview(arr)
    assert len(mv_typed) == 3 and mv_typed.nbytes == 12 and mv_typed.itemsize == 4

    # 1A: `written is None` on typed memoryview records exact byte count (`12`, not item count `3`).
    raw_none = _PartialTypedRawWriter(return_none=True)
    hw_none = _HashingWriter(raw_none)
    ret = hw_none.write(mv_typed)
    assert ret == 12 and hw_none.bytes_written == 12
    assert hw_none.hexdigest == hashlib.sha256(raw_none.buf).hexdigest()

    # 1B: POSIX partial write (`6` of `12` bytes) hashes exact byte slice `[:6]`, not `[:6]` items!
    raw_partial = _PartialTypedRawWriter(max_chunk_bytes=6)
    hw_partial = _HashingWriter(raw_partial)
    byte_view = mv_typed.cast("B")
    n1 = hw_partial.write(mv_typed)
    n2 = hw_partial.write(byte_view[n1:])
    assert n1 == 6 and n2 == 6 and hw_partial.bytes_written == 12
    assert hw_partial.hexdigest == hashlib.sha256(raw_partial.buf).hexdigest()

    # 1C: Non-contiguous 2D tensor view also hashes exact serialized bytes.
    t2d = torch.arange(12, dtype=torch.int32).view(3, 4)[:, ::2].numpy()
    assert not memoryview(t2d).c_contiguous
    raw_nc = _PartialTypedRawWriter(max_chunk_bytes=5)
    hw_nc = _HashingWriter(raw_nc)
    nc_bytes = memoryview(t2d).tobytes()
    off = 0
    first = True
    while off < len(nc_bytes):
        wrote = hw_nc.write(memoryview(t2d) if first else memoryview(nc_bytes)[off:])
        first = False
        off += wrote
    assert hw_nc.bytes_written == len(nc_bytes)
    assert hw_nc.hexdigest == hashlib.sha256(raw_nc.buf).hexdigest()


def test_io_bug_2_m0_gate_atomic_bundle_no_split_brain(tmp_path, monkeypatch):
    ref_root = tmp_path / "ref_artifacts"
    _write_valid_m0_bundle(ref_root, m0_scale="reference", mismatched_tokens=0)
    # Inject an undocumented `sequence_equal == False` prompt in `ref_root/m0/prompt00.json`.
    atomic_json(
        ref_root / "m0" / "prompt00.json",
        {"identity": {"prompt": "undocumented-mismatch"}, "local_target": {"sequence_equal": False}},
    )
    monkeypatch.setenv("EDRAFT_ARTIFACTS_ROOT", str(ref_root))

    # Pre-existing empty `m0/` directory in isolated `--scale full` output MUST NOT split-brain bypass `ref_root/m0`!
    full_out = tmp_path / "full_scale_out"
    (full_out / "m0").mkdir(parents=True)
    args_split = prepare_args(
        parse_config(collect_parser(), ["--scale", "full", "--allow_large_run", "--output", str(full_out)])
    )
    with pytest.raises(RuntimeError, match="M0 numerics must cover every target-only mismatch"):
        check_reference_gates(args_split)

    # Local `m0.json` + `m0_numerics.json` with incomplete `m0/*.json` (< `num_distinct_prompts`) MUST block!
    partial_local = tmp_path / "partial_local_m0"
    _write_valid_m0_bundle(partial_local, m0_scale="reference", mismatched_tokens=0)
    for extra in sorted((partial_local / "m0").glob("*.json"))[5:]:
        extra.unlink()
    args_partial = prepare_args(
        parse_config(collect_parser(), ["--scale", "full", "--allow_large_run", "--output", str(partial_local)])
    )
    with pytest.raises(RuntimeError, match="token reproduction gate"):
        check_reference_gates(args_partial)


def test_fused_live_block_diagnostics_compact_and_buffered_sparse_recorder_parity(tmp_path):
    from edraft.diagnose import live_block_diagnostics
    from edraft.sparse import SignedHash
    from edraft.sparse_stats import SparseAccessRecorder
    from edraft.traces import compact_block, validate_block

    gen = torch.Generator().manual_seed(453)
    K, V, p_dim, hidden = 6, 128, 8, 16
    draft_logits = torch.randn((K, V), generator=gen, dtype=torch.bfloat16)
    target_logits = torch.randn((K, V), generator=gen, dtype=torch.bfloat16)
    l_unit = draft_logits.float().log_softmax(-1)
    h_raw_bf16 = torch.randn((K, hidden), generator=gen, dtype=torch.bfloat16)
    h = torch.randn((K, p_dim), generator=gen, dtype=torch.float32)
    meta = {
        "prompt_id": "p0", "split": "train", "condition": "standard_nonthinking",
        "seed": 2026, "scale": "full", "model_pair": ["a", "b"],
        "projection_metadata": {"dim": p_dim}, "history_depth": 4,
    }
    specials = [120, 121, 122]

    for T, warp in [
        (0.0, {"top_k": 0, "top_p": 1.0, "min_p": 0.0}),
        (0.7, {"top_k": 12, "top_p": 0.8, "min_p": 0.0}),
        (1.0, {"top_k": 0, "top_p": 1.0, "min_p": 0.0}),
    ]:
        rec = {
            "round_idx": 1,
            "draft_tokens": [int(i * 3 + 1) for i in range(K)],
            "draft_logits": draft_logits,
            "l_unit": l_unit,
            "target_logits": target_logits,
            "h_raw_bf16": h_raw_bf16,
            "h": h,
            "draft_lm_head": {
                "artifact_id": "x", "revision": "y", "dtype": "torch.bfloat16",
                "tied_weights": False, "bias": False, "shape": [V, hidden], "hook": "draft_lm_head",
            },
            "collection_temperature": T,
            "loss_temperature": 1.0,
            "verification_rule": "greedy" if T == 0 else "ratio",
            "sampling_warp_config": warp,
            "absolute_output_offset": 0,
            "input_context_length": 16,
            "verified_prefix_tail": [1, 2, 3, 4],
            "physical_to_logical": {"anchor_physical_index": 0, "logical_to_physical": list(range(1, K + 1))},
            "accepted_length": 2,
            "first_rejection_index": 2,
            "replacement_token": 17,
            "adapter_version": 0,
        }
        d_std = live_block_diagnostics(dict(rec), seed=99)
        c_std = compact_block(dict(rec, diagnostics=d_std), meta, special_tokens=specials, _trust_l_unit=True)
        d_fus, p_sh = live_block_diagnostics(dict(rec), seed=99, _return_shared=True)
        c_fus = compact_block(
            dict(rec, diagnostics=d_fus), meta, special_tokens=specials,
            _trust_l_unit=True, _precomputed_p_train=p_sh, _prevalidated_collect=True,
        )
        validate_block(c_fus)
        assert d_fus["target_rank"] == d_std["target_rank"]
        assert d_fus["sampled_target_rank"] == d_std["sampled_target_rank"]
        assert d_fus["sampled_target_rank_unwarped"] == d_std["sampled_target_rank_unwarped"]
        assert d_fus["first_rejection"] == d_std["first_rejection"]
        for ua, ub in zip(c_fus["union_ids"], c_std["union_ids"]):
            assert torch.equal(ua, ub)
        for la, lb in zip(c_fus["draft_l_unit_union"], c_std["draft_l_unit_union"]):
            assert torch.equal(la, lb)
        for pa, pb in zip(c_fus["target_probs_union"], c_std["target_probs_union"]):
            assert torch.equal(pa, pb)
        assert torch.equal(c_fus["target_rest_mass"], c_std["target_rest_mass"])

    # Buffered SparseAccessRecorder(flush_every=8) + mid-stream checkpoint flush matches flush_every=1 bit-for-bit
    batches = [torch.randint(0, 50_000, (512,), generator=gen, dtype=torch.int64) for _ in range(21)]
    sp_tbl = SignedHash(4096)
    r_ref = SparseAccessRecorder(tmp_path, "ref1", 128, flush_every=1)
    ch_ref = [r_ref.write_batch(bk, 0, i) for i, bk in enumerate(batches)]
    s_ref = r_ref.finalize(ch_ref, sp_tbl, memory_limit="256MB", benchmark_entries=1000, seed=2026)

    r_buf = SparseAccessRecorder(tmp_path, "buf8", 128, flush_every=8)
    ch_buf = []
    for i, bk in enumerate(batches):
        ch_buf.extend(r_buf.record_batch(bk, 0, i))
        if i == 10:  # simulate mid-epoch checkpoint_every flush
            ch_buf.extend(r_buf.flush())
    ch_buf.extend(r_buf.flush())
    assert len(ch_buf) == 4  # [0..7], [8..10], [11..18], [19..20]
    s_buf = r_buf.finalize(ch_buf, sp_tbl, memory_limit="256MB", benchmark_entries=1000, seed=2026)
    for k in (
        "unique_observed_keys", "observed_lookup_events", "occupied_buckets",
        "collision_count", "distinct_keys_in_collided_buckets", "frequency_weighted_collision_count",
    ):
        assert s_buf[k] == s_ref[k]

