from copy import deepcopy
import json

import pytest
import torch

from edraft.dflash_loop import generate_toy
from edraft.sampler import SamplingWarper
from edraft.toy import ToyParallelDrafter, ToyTarget
from edraft.traces import (TraceShardWriter, compact_block, read_shard,
                           reconstruction_diagnostic, validate_block, write_shard)


@pytest.fixture
def live_record():
    target = ToyTarget(vocab_size=16)
    drafter = ToyParallelDrafter(target)
    generated = generate_toy(target, drafter, [2, 5, 8, 1], 8, block_length=4,
                             temperature=.7, warper=SamplingWarper(top_k=3, top_p=.8),
                             generator=torch.Generator().manual_seed(50), retain_tensors=True)
    metadata = {"prompt_id": "fixture-test-1", "split": "test", "condition": "standard_nonthinking",
                "seed": 50, "scale": "synthetic", "model_pair": ["toy-target-v1", "toy-drafter-v1"],
                "projection_metadata": {"artifact_id": "fixture-identity-projection", "frozen": True,
                                        "calibration_split": "train", "calibration_prompt_ids": ["fixture-train"]},
                "history_depth": 4}
    return generated.blocks[0], metadata, drafter


def test_compact_union_keeps_unwarped_target_mass_and_exact_normalizers(live_record, monkeypatch):
    record, metadata, _ = live_record
    def forbidden(*args, **kwargs):
        raise AssertionError("trace training targets used an inference sampling helper")
    monkeypatch.setattr("edraft.sampler.probabilities", forbidden)
    block = compact_block(record, metadata, M_draft=2, M_target=1, special_tokens=[0, 15])
    P = record["target_logits"].float().softmax(-1)
    log_q0 = record["draft_logits"].float().log_softmax(-1)
    assert block["loss_temperature"] == block["beta_train"] == 1
    assert block["collection_temperature"] == .7
    assert block["target_distribution"] == "unwarped_full_softmax_unit_temperature"
    assert "live_full" not in block
    for i, ids in enumerate(block["union_ids"]):
        assert ids.device.type == "cpu" and ids.dtype == torch.int64
        assert {0, 15, record["draft_tokens"][i]} <= set(ids.tolist())
        assert set(record["draft_logits"][i].topk(2).indices.tolist()) <= set(ids.tolist())
        assert set(record["target_logits"][i].topk(1).indices.tolist()) <= set(ids.tolist())
        assert len(ids) == len(ids.unique())
        torch.testing.assert_close(block["target_probs_union"][i], P[i, ids], rtol=0, atol=0)
        torch.testing.assert_close(block["draft_l_unit_union"][i], log_q0[i, ids], rtol=0, atol=0)
        # A warped collection context must not turn training labels into the
        # top-3 sampling distribution or normalize the union to one.
        assert len(ids) < P.shape[1]
        assert block["target_probs_union"][i].sum() < 1
        assert block["target_rest_mass"][i] > 0
        total = block["target_probs_union"][i].double().sum() + block["target_rest_mass"][i]
        assert float(total) == pytest.approx(1, abs=2e-7)
    torch.testing.assert_close(block["draft_log_normalizer"], record["draft_logits"].float().logsumexp(-1))
    torch.testing.assert_close(block["target_log_normalizer"], record["target_logits"].float().logsumexp(-1))


def test_greedy_context_still_stores_full_soft_targets(live_record):
    record, metadata, _ = live_record
    record = dict(record, collection_temperature=0, verification_rule="greedy")
    block = compact_block(record, metadata, M_draft=2, M_target=1)
    assert block["loss_temperature"] == 1
    assert all(bool((row > 0).all() and (row < 1).all()) for row in block["target_probs_union"])


def test_atomic_shard_roundtrip_safe_load_and_live_full_gap_data(live_record, tmp_path):
    record, metadata, _ = live_record
    compact = compact_block(record, metadata, M_draft=2, M_target=1, retain_live_full=True)
    path = tmp_path / "rank0" / "trace.pt"
    summary = write_shard(path, [compact], {"rank": 0, "condition": metadata["condition"]})
    loaded = read_shard(path)
    block = loaded["blocks"][0]
    assert summary["blocks"] == 1
    assert summary["logical_slots"] == len(record["draft_tokens"])
    assert len(summary["sha256"]) == 64
    assert summary["disk_bytes"] == path.stat().st_size
    assert not list(path.parent.glob("*.tmp"))
    assert block["draft_lm_head"] == compact["draft_lm_head"]
    assert block["physical_to_logical"] == compact["physical_to_logical"]
    assert block["projection_metadata"] == metadata["projection_metadata"]
    for field in ["h_raw_bf16", "h", "target_rest_mass", "target_log_normalizer"]:
        assert torch.equal(block[field], compact[field])
        assert block[field].device.type == "cpu"
    for field in ["union_ids", "draft_l_unit_union", "target_probs_union"]:
        assert all(torch.equal(a, b) for a, b in zip(block[field], compact[field]))
    assert torch.equal(block["live_full"]["target_probs"], record["target_logits"].float().softmax(-1))
    # Serialization is detached from the active live snapshot / callback record.
    old = record["h_raw_bf16"].clone()
    compact["h_raw_bf16"].zero_()
    assert torch.equal(record["h_raw_bf16"], old)


def test_checksum_detects_truncated_or_corrupted_shard(live_record, tmp_path):
    record, metadata, _ = live_record
    path = tmp_path / "trace.pt"
    write_shard(path, [compact_block(record, metadata)], {})
    data = bytearray(path.read_bytes())
    data[len(data)//2] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_shard(path)


def _identity(record, metadata):
    return {"model_pair": metadata["model_pair"], "prompt": metadata["prompt_id"],
            "condition": metadata["condition"], "seed": metadata["seed"],
            "temperature": record["collection_temperature"], "verification_rule": record["verification_rule"],
            "proposal_mode": record["proposal_mode"], "checkpoint": "frozen-toy-head-v1",
            "adapter_version": record["adapter_version"], "sampling_warp_config": record["sampling_warp_config"]}


def test_rank_local_manifest_resume_full_identity_and_corruption_recovery(live_record, tmp_path):
    record, metadata, _ = live_record
    identity = _identity(record, metadata)
    block = compact_block(record, metadata)
    writer = TraceShardWriter(tmp_path, rank=2)
    assert not writer.completed(identity)
    path = writer.write(identity, [block], {"run": "synthetic-test"})
    assert writer.completed(identity)
    assert path.parent.name == "rank2"
    assert (tmp_path / "manifest-rank2.json").is_file()
    old_mtime = path.stat().st_mtime_ns
    assert writer.write(identity, [block]) == path
    assert path.stat().st_mtime_ns == old_mtime
    restarted = TraceShardWriter(tmp_path, rank=2)
    assert restarted.completed(identity)
    assert not restarted.completed(dict(identity, seed=identity["seed"]+1))
    assert not restarted.completed(dict(identity, checkpoint="different-checkpoint"))
    assert not TraceShardWriter(tmp_path, rank=3).completed(identity)
    path.write_bytes(b"interrupted/corrupted")
    assert not restarted.completed(identity)
    restarted.write(identity, [block])
    assert restarted.completed(identity)
    assert read_shard(path)["metadata"]["identity"] == identity


def test_manifest_refuses_wrong_block_identity(live_record, tmp_path):
    record, metadata, _ = live_record
    writer = TraceShardWriter(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        writer.write(dict(_identity(record, metadata), seed=1), [compact_block(record, metadata)])
    assert not list(tmp_path.glob("rank0/*.pt"))


@pytest.mark.parametrize("change,error", [({"loss_temperature": .7}, "loss_temperature"),
    ({"verification_rule": "independent_match"}, "ratio rule"),
    ({"verified_prefix_tail": [1]}, "cross-boundary"),
    ({"physical_to_logical": {"anchor_physical_index": 0, "logical_to_physical": [0, 1, 2, 3]}}, "logical slot"),
    ({"first_rejection_index": 99}, "accepted length")])
def test_rejects_invalid_canonical_fields(live_record, change, error):
    record, metadata, _ = live_record
    with pytest.raises(ValueError, match=error):
        compact_block(dict(record, **change), metadata)


def test_rejects_warped_logits_and_renormalized_union(live_record):
    record, metadata, _ = live_record
    wrong = dict(record, target_logits=record["target_logits"].clone())
    wrong["target_logits"][0, 0] = -torch.inf
    with pytest.raises(ValueError, match="finite and unwarped"):
        compact_block(wrong, metadata)
    compact = compact_block(record, metadata, M_draft=2, M_target=1)
    compact["target_probs_union"][0] /= compact["target_probs_union"][0].sum()
    with pytest.raises(ValueError, match="unrenormalized rest"):
        validate_block(compact)


def test_reconstruction_diagnostic_checks_full_vocabulary_and_reports_tolerances(live_record):
    record, metadata, drafter = live_record
    result = reconstruction_diagnostic(record, drafter.draft_lm_head, chunk_size=3,
                                       atol=.02, rtol=0, probability_atol=.003, probability_rtol=0)
    assert result["status"] == "passed"
    assert result["elements_checked"] == record["draft_logits"].numel()
    assert result["shape"] == list(record["draft_logits"].shape)
    assert result["log_probability_atol"] == .02
    assert result["probability_atol"] == .003
    assert result["accumulation_dtype"] == "torch.float32"
    assert result["head_metadata"] == record["draft_lm_head"]
    corrupted = dict(record, h_raw_bf16=torch.zeros_like(record["h_raw_bf16"]))
    failed = reconstruction_diagnostic(corrupted, drafter.draft_lm_head, atol=.02, rtol=0,
                                       probability_atol=.003, probability_rtol=0)
    assert failed["status"] == "failed"


def test_reconstruction_includes_bias_and_output_multiplier(live_record):
    record, _, drafter = live_record
    record = dict(record, draft_logits=record["draft_logits"] * 1.7,
                  draft_lm_head=dict(record["draft_lm_head"], output_multiplier=1.7))
    result = reconstruction_diagnostic(record, drafter.draft_lm_head, atol=.03, rtol=0)
    assert result["status"] == "passed"
    assert result["output_multiplier"] == 1.7


def test_schema_missing_provenance_rejected(live_record):
    record, metadata, _ = live_record
    with pytest.raises(ValueError, match="projection"):
        compact_block(record, {k:v for k,v in metadata.items() if k != "projection_metadata"})
    wrong_head = dict(record["draft_lm_head"], revision=None)
    with pytest.raises(ValueError, match="immutable artifact"):
        compact_block(dict(record, draft_lm_head=wrong_head), metadata)
