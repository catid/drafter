from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from edraft.dflash_loop import generate_toy
from edraft.sampler import SamplingWarper
from edraft.toy import ToyParallelDrafter, ToyTarget
from edraft.traces import TraceShardWriter, compact_block
from edraft.train import _setup_distributed, fit, load_adapter_checkpoint, trace_index


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_trace_bundle(root: Path) -> tuple:
    """Create deterministic synthetic shards where train block count % 4 == 1 (tests tail < world_size)."""
    torch.set_num_threads(1)
    torch.manual_seed(93)
    target = ToyTarget(vocab_size=11, hidden_size=6, seed=93)
    draft = ToyParallelDrafter(target, seed=94)
    writer = TraceShardWriter(root / "traces")
    # 3 train shards * 3 blocks = 9 train blocks -> with batch_blocks=4: [4, 4, 1] (tail=1 < world_size=2!)
    plan = [
        ("train", "p-train-0", 0.0, 9),
        ("train", "p-train-1", 0.7, 9),
        ("train", "p-train-2", 0.0, 9),
        ("dev", "p-dev-0", 0.0, 9),
        ("test", "p-test-0", 0.7, 9),
    ]
    for idx, (split, pid, temperature, max_new) in enumerate(plan):
        warper = SamplingWarper() if temperature == 0 else SamplingWarper(top_k=20, top_p=0.8)
        result = generate_toy(
            target, draft, [idx, 2, 4, 1], max_new, block_length=3,
            temperature=temperature, warper=warper, retain_tensors=True,
            generator=torch.Generator().manual_seed(93 + idx),
        )
        identity = {
            "model_pair": ["toy-target", "toy-draft"],
            "prompt": pid,
            "condition": "standard_nonthinking",
            "seed": 93 + idx,
            "temperature": temperature,
            "verification_rule": "greedy" if temperature == 0 else "ratio",
            "proposal_mode": "plain_dflash",
            "checkpoint": "synthetic-v1",
            "adapter_version": 0,
            "sampling_warp_config": warper.to_dict(),
        }
        metadata = {
            "prompt_id": pid,
            "split": split,
            "condition": "standard_nonthinking",
            "seed": 93 + idx,
            "scale": "synthetic",
            "model_pair": identity["model_pair"],
            "projection_metadata": {"method": "synthetic-identity", "frozen": True},
            "history_depth": 2,
        }
        blocks = [
            compact_block(record, metadata, M_draft=3, M_target=2, retain_live_full=True)
            for record in result.blocks[:3]
        ]
        writer.write(identity, blocks, {
            "generation": {
                "output_tokens": result.output_tokens,
                "realized_length": result.num_output_tokens,
                "condition": "standard_nonthinking",
                "split": split,
                "temperature": temperature,
                "sampling_warp_config": warper.to_dict(),
                "complete": True,
            }
        })
    entries, report = trace_index(root / "traces")
    return (
        root / "traces",
        entries,
        report,
        target.embedding[:, :4].detach().clone(),
        draft.draft_lm_head.weight.detach().clone(),
        None if draft.draft_lm_head.bias is None else draft.draft_lm_head.bias.detach().clone(),
        draft.head_metadata(),
    )


def _train_args(output: Path, *, rank: int = 0, world_size: int = 1,
                batch_blocks: int = 4, microbatch_blocks: int | None = None):
    return SimpleNamespace(
        output=output,
        scale="smoke",
        seed=101,
        history_depth=2,
        M_sparse=4,
        sparse_capacity=48,
        learning_rate=0.015,
        weight_decay=0.0002,
        bias_weight_decay=0.0001,
        sparse_learning_rate=0.025,
        lambda_l1=0.0002,
        batch_blocks=batch_blocks,
        microbatch_blocks=microbatch_blocks,
        vocab_chunk=4,
        epochs_dense=1,
        epochs_sparse=1,
        checkpoint_every=2,
        resume=True,
        rank=rank,
        world_size=world_size,
        dist_backend="gloo",
    )


def _ddp_worker(rank: int, world_size: int, port: int, trace_dir_str: str, output_str: str,
                E: torch.Tensor, head_w: torch.Tensor, head_b: torch.Tensor | None,
                head_meta: dict, microbatch_blocks: int | None, stop_after_steps: int | None):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(101)
        entries, report = trace_index(Path(trace_dir_str))
        args = _train_args(
            Path(output_str), rank=rank, world_size=world_size,
            batch_blocks=4, microbatch_blocks=microbatch_blocks,
        )
        fit(entries, report, E, head_w, head_b, head_meta, args, stop_after_steps=stop_after_steps)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("microbatch_blocks", [1, 3])
def test_microbatch_blocks_matches_full_batch_fit_and_sparse_collisions(tmp_path: Path, microbatch_blocks: int):
    _, entries, report, E, head_w, head_b, head_meta = _build_trace_bundle(tmp_path / "bundle")
    torch.manual_seed(101)
    ref_report = fit(entries, report, E, head_w, head_b, head_meta, _train_args(tmp_path / "ref", batch_blocks=4))

    torch.manual_seed(101)
    mb_report = fit(
        entries, report, E, head_w, head_b, head_meta,
        _train_args(tmp_path / f"mb_{microbatch_blocks}", batch_blocks=4, microbatch_blocks=microbatch_blocks),
    )

    for stage in ("dense", "dense_sparse"):
        ref_model, _ = load_adapter_checkpoint(Path(ref_report["checkpoints"][stage]))
        mb_model, _ = load_adapter_checkpoint(Path(mb_report["checkpoints"][stage]))
        for key, ref_val in ref_model.state_dict().items():
            torch.testing.assert_close(mb_model.state_dict()[key], ref_val, atol=2e-6, rtol=2e-5)

    ref_stats = ref_report["stages"]["dense_sparse"]["sparse_collision_statistics"]
    mb_stats = mb_report["stages"]["dense_sparse"]["sparse_collision_statistics"]
    for field in ("observed_lookup_events", "unique_observed_keys", "collision_count", "frequency_weighted_collision_count"):
        assert mb_stats[field] == ref_stats[field]


def test_ddp_gloo_multi_process_and_resume_match_single_process(tmp_path: Path):
    trace_dir, entries, report, E, head_w, head_b, head_meta = _build_trace_bundle(tmp_path / "bundle")

    # 1. Single-process reference fit
    torch.manual_seed(101)
    ref_report = fit(entries, report, E, head_w, head_b, head_meta, _train_args(tmp_path / "single", batch_blocks=4))
    ref_dense, _ = load_adapter_checkpoint(Path(ref_report["checkpoints"]["dense"]))
    ref_sparse, _ = load_adapter_checkpoint(Path(ref_report["checkpoints"]["dense_sparse"]))

    # 2. 2-rank DDP fit with microbatch_blocks=1 AND interrupted resume across dense -> sparse boundary
    ddp_out = tmp_path / "ddp_resumed"
    port1 = _free_port()
    mp.spawn(
        _ddp_worker,
        args=(2, port1, str(trace_dir), str(ddp_out), E, head_w, head_b, head_meta, 1, 2),
        nprocs=2,
        join=True,
    )
    assert (ddp_out / "m2_smoke" / "latest.pt").is_file()
    assert not (ddp_out / "m2_smoke.json").exists()

    port2 = _free_port()
    mp.spawn(
        _ddp_worker,
        args=(2, port2, str(trace_dir), str(ddp_out), E, head_w, head_b, head_meta, 1, None),
        nprocs=2,
        join=True,
    )
    ddp_report = json.loads((ddp_out / "m2_smoke.json").read_text())
    assert ddp_report["status"] == ref_report["status"]

    ddp_dense, _ = load_adapter_checkpoint(Path(ddp_report["checkpoints"]["dense"]))
    ddp_sparse, _ = load_adapter_checkpoint(Path(ddp_report["checkpoints"]["dense_sparse"]))
    for key, ref_val in ref_dense.state_dict().items():
        torch.testing.assert_close(ddp_dense.state_dict()[key], ref_val, atol=3e-6, rtol=3e-5)
    for key, ref_val in ref_sparse.state_dict().items():
        torch.testing.assert_close(ddp_sparse.state_dict()[key], ref_val, atol=3e-6, rtol=3e-5)

    for split in ("train", "dev", "test"):
        assert ddp_report["baseline"][split]["weighted_mean_kl"] == pytest.approx(
            ref_report["baseline"][split]["weighted_mean_kl"], rel=1e-6, abs=1e-6
        )
        assert ddp_report["stages"]["dense_sparse"][split]["weighted_mean_kl"] == pytest.approx(
            ref_report["stages"]["dense_sparse"][split]["weighted_mean_kl"], rel=1e-5, abs=1e-6
        )

    ref_stats = ref_report["stages"]["dense_sparse"]["sparse_collision_statistics"]
    ddp_stats = ddp_report["stages"]["dense_sparse"]["sparse_collision_statistics"]
    for field in ("observed_lookup_events", "unique_observed_keys", "collision_count", "frequency_weighted_collision_count"):
        assert ddp_stats[field] == ref_stats[field]


def test_launch_cli_supports_ddp_train_and_dynamic_schedules(tmp_path: Path, monkeypatch):
    import edraft.launch as launch_mod

    monkeypatch.setattr(launch_mod, "selected_gpus_idle", lambda indices: None)
    monkeypatch.setattr(launch_mod, "record_run", lambda args, name: {})

    spawned = []

    class DummyProcess:
        def __init__(self, command, cwd, env, stdout, stderr):
            self.pid = 10000 + len(spawned)
            self.returncode = 0
            spawned.append((command, env))

        def poll(self):
            return 0

    monkeypatch.setattr(launch_mod.subprocess, "Popen", DummyProcess)

    # 1. DDP train launch across 2 GPUs
    launch_mod.main([
        "--scale", "reference", "--task", "train", "--gpus", "2,5",
        "--output", str(tmp_path / "out_train"), "--batch_blocks", "16", "--microbatch_blocks", "4",
    ])
    assert len(spawned) == 2
    cmd0, env0 = spawned[0]
    cmd1, env1 = spawned[1]
    assert "edraft.train" in cmd0 and "--batch_blocks" in cmd0 and "16" in cmd0
    assert "--microbatch_blocks" in cmd0 and "4" in cmd0
    assert env0["CUDA_VISIBLE_DEVICES"] == "2" and env1["CUDA_VISIBLE_DEVICES"] == "5"
    assert env0["MASTER_ADDR"] == "127.0.0.1" and env0["MASTER_PORT"] == env1["MASTER_PORT"]

    # 2. Dynamic M4 evaluation with opt-in compile + cuda_graph
    spawned.clear()
    launch_mod.main([
        "--scale", "reference", "--task", "m4", "--gpus", "1,3", "--schedule", "dynamic",
        "--compile", "--cuda_graph", "--output", str(tmp_path / "out_m4"),
    ])
    assert len(spawned) == 2
    m4_cmd, _ = spawned[0]
    assert "edraft.eval_closed_loop" in m4_cmd
    assert "--schedule" in m4_cmd and "dynamic" in m4_cmd
    assert "--compile" in m4_cmd and "--cuda_graph" in m4_cmd


def test_setup_distributed_torchrun_local_rank_and_nccl_device_id(monkeypatch):
    from datetime import timedelta

    set_device_calls: list[torch.device] = []
    init_pg_calls: list[dict] = []
    exp_timeout = timedelta(minutes=60)

    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("RANK")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    monkeypatch.delenv("NCCL_NVLS_ENABLE")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: set_device_calls.append(dev))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: init_pg_calls.append(kwargs))

    # 1. torchrun multi-GPU visibility (device_count=8, LOCAL_RANK=3, default device 'cuda:0' or 'cuda')
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setenv("LOCAL_RANK", "3")
    for initial_dev in ("cuda:0", "cuda"):
        set_device_calls.clear()
        init_pg_calls.clear()
        args = SimpleNamespace(rank=3, world_size=8, device=initial_dev, dist_backend=None)
        assert _setup_distributed(args) is True
        assert args.device == "cuda:3"
        assert set_device_calls == [torch.device("cuda", 3)]
        assert init_pg_calls == [{
            "backend": "nccl", "rank": 3, "world_size": 8,
            "timeout": exp_timeout, "device_id": torch.device("cuda", 3),
        }]

    # 2. launch.py single-visible-GPU per rank (CUDA_VISIBLE_DEVICES='5' -> device_count=1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("LOCAL_RANK", "3")
    set_device_calls.clear()
    init_pg_calls.clear()
    args_launch = SimpleNamespace(rank=3, world_size=8, device="cuda:0", dist_backend=None)
    assert _setup_distributed(args_launch) is True
    assert args_launch.device == "cuda:0"
    assert set_device_calls == [torch.device("cuda", 0)]
    assert init_pg_calls == [{
        "backend": "nccl", "rank": 3, "world_size": 8,
        "timeout": exp_timeout, "device_id": torch.device("cuda", 0),
    }]

    # 3. Explicit non-zero --device cuda:5 is preserved even when LOCAL_RANK is set
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setenv("LOCAL_RANK", "2")
    set_device_calls.clear()
    init_pg_calls.clear()
    args_explicit = SimpleNamespace(rank=2, world_size=8, device="cuda:5", dist_backend=None)
    assert _setup_distributed(args_explicit) is True
    assert args_explicit.device == "cuda:5"
    assert set_device_calls == [torch.device("cuda", 5)]
    assert init_pg_calls == [{
        "backend": "nccl", "rank": 2, "world_size": 8,
        "timeout": exp_timeout, "device_id": torch.device("cuda", 5),
    }]

    # 4. Out-of-range LOCAL_RANK fails fast with ValueError
    monkeypatch.setenv("LOCAL_RANK", "9")
    args_bad = SimpleNamespace(rank=1, world_size=8, device="cuda:0", dist_backend=None)
    with pytest.raises(ValueError, match="LOCAL_RANK=9"):
        _setup_distributed(args_bad)

