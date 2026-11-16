from __future__ import annotations

from datetime import timedelta
import gc
import os
from pathlib import Path
import random
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from edraft import launch, post_m4_suite, train
from edraft.dflash_loop import generate_toy
from edraft.sampler import SamplingWarper
from edraft.sparse import ExactObservedPairs
from edraft.toy import ToyParallelDrafter, ToyTarget
from edraft.traces import TraceShardWriter, compact_block
from edraft.train import (
    _HEAD_CAST_CACHE,
    _SharedShardCache,
    _comm_device,
    _reconstruct_h_raw,
    fit,
    load_adapter_checkpoint,
    trace_index,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ref_reconstruct_h_raw(
    h_raw_bf16: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    vocab_chunk: int = 8192,
    output_multiplier: float = 1.0,
    bf16_gemm: bool = False,
) -> torch.Tensor:
    device = weight.device
    num_rows = int(h_raw_bf16.shape[0])
    vocab_size = int(weight.shape[0])
    if num_rows == 0:
        return torch.empty((0, vocab_size), device=device, dtype=torch.float32)
    logits = torch.empty((num_rows, vocab_size), device=device, dtype=torch.float32)
    mult = float(output_multiplier)
    if bf16_gemm:
        h_bf16 = h_raw_bf16.to(device=device, dtype=torch.bfloat16)
        for start in range(0, vocab_size, vocab_chunk):
            end = min(start + vocab_chunk, vocab_size)
            w_chunk = weight[start:end]
            w_bf16 = w_chunk if w_chunk.dtype == torch.bfloat16 else w_chunk.to(torch.bfloat16)
            out = F.linear(h_bf16, w_bf16).float()
            if bias is not None:
                out = out + bias[start:end].float()
            if mult != 1.0:
                out = out * mult
            logits[:, start:end] = out
    else:
        h_fp32 = h_raw_bf16.to(device=device, dtype=torch.float32)
        for start in range(0, vocab_size, vocab_chunk):
            end = min(start + vocab_chunk, vocab_size)
            b_chunk = None if bias is None else bias[start:end].float()
            logits[:, start:end] = F.linear(h_fp32, weight[start:end].float(), b_chunk) * mult
    return logits.log_softmax(-1)


def _build_multishard_trace_bundle(root: Path) -> tuple:
    """Create deterministic 6-train / 2-dev / 2-test shards with live_full retained."""
    torch.set_num_threads(1)
    torch.manual_seed(311)
    target = ToyTarget(vocab_size=13, hidden_size=8, seed=311)
    draft = ToyParallelDrafter(target, seed=312)
    writer = TraceShardWriter(root / "traces")
    plan = [
        ("train", "p-train-0", 0.0, 9),
        ("train", "p-train-1", 0.7, 9),
        ("train", "p-train-2", 0.0, 9),
        ("train", "p-train-3", 0.7, 9),
        ("train", "p-train-4", 0.0, 9),
        ("train", "p-train-5", 0.7, 9),
        ("dev", "p-dev-0", 0.0, 9),
        ("dev", "p-dev-1", 0.7, 9),
        ("test", "p-test-0", 0.0, 9),
        ("test", "p-test-1", 0.7, 9),
    ]
    for idx, (split, pid, temperature, max_new) in enumerate(plan):
        warper = SamplingWarper() if temperature == 0 else SamplingWarper(top_k=20, top_p=0.8)
        result = generate_toy(
            target,
            draft,
            [idx % 13, 3, 5, 1],
            max_new,
            block_length=3,
            temperature=temperature,
            warper=warper,
            retain_tensors=True,
            generator=torch.Generator().manual_seed(311 + idx),
        )
        identity = {
            "model_pair": ["toy-target", "toy-draft"],
            "prompt": pid,
            "condition": "standard_nonthinking",
            "seed": 311 + idx,
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
            "seed": 311 + idx,
            "scale": "synthetic",
            "model_pair": identity["model_pair"],
            "projection_metadata": {"method": "synthetic-identity", "frozen": True},
            "history_depth": 2,
        }
        blocks = [
            compact_block(record, metadata, M_draft=3, M_target=2, retain_live_full=True)
            for record in result.blocks[:3]
        ]
        writer.write(
            identity,
            blocks,
            {
                "generation": {
                    "output_tokens": result.output_tokens,
                    "realized_length": result.num_output_tokens,
                    "condition": "standard_nonthinking",
                    "split": split,
                    "temperature": temperature,
                    "sampling_warp_config": warper.to_dict(),
                    "complete": True,
                }
            },
        )
    entries, report = trace_index(root / "traces")
    return (
        root / "traces",
        entries,
        report,
        target.embedding[:, :4].detach().clone(),
        draft.draft_lm_head.weight.detach().clone().to(torch.bfloat16),
        None if draft.draft_lm_head.bias is None else draft.draft_lm_head.bias.detach().clone().to(torch.bfloat16),
        draft.head_metadata(),
    )


def _nccl_train_args(
    output: Path,
    *,
    device: str,
    rank: int = 0,
    world_size: int = 1,
    objective: str = "union_rest_grouped",
    sparse_backend: str = "signed_feature_hash",
    block_shuffle_window: int = 0,
    calibrate_beta: bool = False,
    apply_calibrated_beta: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    checkpoint_every: int = 2,
    m5_fit_tag: str = "nccl_fit",
) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        scale="smoke",
        seed=404,
        device=device,
        history_depth=2,
        M_sparse=4,
        sparse_capacity=64,
        sparse_backend=sparse_backend,
        sparse_overflow_policy="freeze_new",
        objective=objective,
        block_shuffle_window=block_shuffle_window,
        calibrate_beta=calibrate_beta,
        apply_calibrated_beta=apply_calibrated_beta,
        calibrate_beta_grid=[0.0, 0.5, 1.0],
        learning_rate=0.015,
        weight_decay=0.0002,
        bias_weight_decay=0.0001,
        sparse_learning_rate=0.025,
        lambda_l1=0.0002,
        batch_blocks=4,
        microbatch_blocks=1,
        vocab_chunk=5,
        epochs_dense=1,
        epochs_sparse=1,
        checkpoint_every=checkpoint_every,
        num_workers=num_workers,
        pin_memory=pin_memory,
        resume=True,
        rank=rank,
        world_size=world_size,
        dist_backend="nccl" if world_size > 1 else None,
        stats_memory_limit="256MB",
        stats_benchmark_entries=1024,
        m5_fit_tag=m5_fit_tag,
    )


def _nccl_suite_worker(
    rank: int,
    world_size: int,
    port: int,
    trace_dir_str: str,
    base_dir_str: str,
    E_cpu: torch.Tensor,
    head_w_cpu: torch.Tensor,
    head_b_cpu: torch.Tensor | None,
    head_meta: dict,
    jobs: list[tuple[str, dict, int | None, str | None]],
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    torch.set_num_threads(1)
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
        device_id=dev,
    )
    try:
        entries, idx_report = trace_index(Path(trace_dir_str))
        E = E_cpu.to(device=dev, dtype=torch.float32)
        head_w = head_w_cpu.to(device=dev)
        head_b = None if head_b_cpu is None else head_b_cpu.to(device=dev)
        base_dir = Path(base_dir_str)
        for subdir_name, cfg, stop_after_steps, r1_save_name in jobs:
            dist.barrier()
            torch.manual_seed(404)
            random.seed(404)
            torch.cuda.manual_seed_all(404)
            args = _nccl_train_args(
                base_dir / subdir_name,
                device=f"cuda:{rank}",
                rank=rank,
                world_size=world_size,
                **cfg,
            )
            rep = fit(entries, idx_report, E, head_w, head_b, head_meta, args, stop_after_steps=stop_after_steps)
            if rank == 1 and r1_save_name is not None:
                torch.save(rep, base_dir / r1_save_name)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_comm_device_never_returns_cpu_under_nccl(monkeypatch):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert _comm_device(torch.device("cpu")) == torch.device("cuda", 1)
    assert _comm_device("cpu") == torch.device("cuda", 1)
    assert _comm_device(torch.device("cuda")) == torch.device("cuda", 1)
    assert _comm_device(torch.device("cuda", 0)) == torch.device("cuda", 0)

    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    assert _comm_device(torch.device("cuda", 0)) == torch.device("cpu")


def test_shared_shard_cache_single_flight_and_error_recovery(tmp_path: Path):
    cache = _SharedShardCache(max_cached_shards=3)
    load_counts: dict[Path, int] = {}
    count_lock = threading.Lock()
    barrier = threading.Barrier(6)

    p0 = tmp_path / "shard_0.pt"

    def _slow_loader():
        with count_lock:
            load_counts[p0] = load_counts.get(p0, 0) + 1
        time.sleep(0.03)
        return {"blocks": [1, 2, 3]}

    results: list[dict] = []
    res_lock = threading.Lock()

    def _worker():
        barrier.wait()
        out = cache.get_or_load(p0, _slow_loader)
        with res_lock:
            results.append(out)

    threads = [threading.Thread(target=_worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert load_counts[p0] == 1
    assert len(results) == 6
    assert all(r is results[0] for r in results)

    # Error in loader must wake waiters and allow subsequent retry
    p_err = tmp_path / "shard_err.pt"
    attempts = 0

    def _flaky_loader():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient read error")
        return {"ok": True}

    with pytest.raises(RuntimeError, match="transient read error"):
        cache.get_or_load(p_err, _flaky_loader)
    assert cache.get_or_load(p_err, _flaky_loader) == {"ok": True}
    assert attempts == 2


def test_launch_cli_forwards_checkpoint_every_and_reaps_children(tmp_path: Path, monkeypatch):
    spawned: list[list[str]] = []
    terminated: list[int] = []

    class _DummyProc:
        def __init__(self, cmd, idx):
            self.cmd = cmd
            self.pid = 40000 + idx
            self._alive = idx == 1
            self.returncode = None if self._alive else 0

        def poll(self):
            self.returncode = None if self._alive else 0
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 1 if self._alive else 0
            return self.returncode

        def terminate(self):
            terminated.append(self.pid)
            self._alive = False
            self.returncode = -15

        def kill(self):
            self._alive = False
            self.returncode = -9

    def _fake_popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        idx = len(spawned)
        spawned.append(list(cmd))
        return _DummyProc(cmd, idx)

    monkeypatch.setattr(launch, "selected_gpus_idle", lambda indices: None)
    monkeypatch.setattr(launch, "record_run", lambda args, name: {})
    monkeypatch.setattr(launch.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit):
        launch.main(["--scale", "reference", "--task", "train", "--gpus", "0", "--checkpoint_every", "0", "--output", str(tmp_path)])

    # Simulate rank 0 failing while rank 1 is still alive -> worker loop + finally block must reap rank 1
    class _FailRank0Proc(_DummyProc):
        def poll(self):
            if self.pid == 40000:
                self.returncode = 2
                return 2
            self.returncode = None if self._alive else 0
            return self.returncode

    def _fail_popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        idx = len(spawned)
        spawned.append(list(cmd))
        return _FailRank0Proc(cmd, idx)

    spawned.clear()
    terminated.clear()
    monkeypatch.setattr(launch.subprocess, "Popen", _fail_popen)
    with pytest.raises(RuntimeError, match="Worker failed"):
        launch.main(
            [
                "--scale",
                "reference",
                "--task",
                "train",
                "--gpus",
                "0,1",
                "--checkpoint_every",
                "1000",
                "--output",
                str(tmp_path),
            ]
        )
    assert len(spawned) == 2
    for cmd in spawned:
        assert "--checkpoint_every" in cmd
        idx = cmd.index("--checkpoint_every")
        assert cmd[idx + 1] == "1000"
    assert terminated == [40001]


def test_offline_kl_summary_exact_observed_pairs_collision_report(tmp_path: Path):
    _, entries, idx_report, E, head_w, head_b, head_meta = _build_multishard_trace_bundle(tmp_path / "bundle")
    args = _nccl_train_args(
        tmp_path / "out",
        device="cpu",
        sparse_backend="exact_observed_pairs",
        num_workers=0,
        pin_memory=False,
        m5_fit_tag="exact_unit",
    )
    rep = fit(entries, idx_report, E, head_w.float(), None if head_b is None else head_b.float(), head_meta, args)
    ckpt_path = Path(rep["checkpoints"]["dense_sparse"])

    # Force collision_statistics out of checkpoint metadata so _offline_kl_summary exercises ExactObservedPairs fallback
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    raw_ckpt["report"]["stages"]["dense_sparse"].pop("sparse_collision_statistics", None)
    stripped_ckpt = tmp_path / "dense_sparse_stripped.pt"
    torch.save(raw_ckpt, stripped_ckpt)

    eval_ctx = (
        SimpleNamespace(batch_blocks=4, history_depth=2, vocab_chunk=5, bf16_gemm=False),
        {"dev": train.epoch_references(entries, "dev"), "test": train.epoch_references(entries, "test")},
        head_w.float(),
        None if head_b is None else head_b.float(),
        head_meta,
    )
    summary = post_m4_suite._offline_kl_summary(eval_ctx, stripped_ckpt, torch.device("cpu"))
    assert summary["sparse_backend"] == "exact_observed_pairs"
    assert summary["collision_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for _reconstruct_h_raw 0-ULP head cache test")
def test_reconstruct_h_raw_zero_ulp_parity_and_weakref_eviction():
    device = torch.device("cuda", 0)
    torch.manual_seed(77)
    V, D = 19000, 128
    w_bf16 = torch.randn(V, D, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(V, device=device, dtype=torch.bfloat16)

    for bf16_gemm in (False, True):
        for bias_in in (None, b_bf16, b_bf16.float()):
            for mult in (1.0, 0.75):
                for n in (0, 1, 2, 7, 15, 16, 17, 24, 32, 40, 65):
                    h_raw = torch.randn(n, D, device=device, dtype=torch.bfloat16)
                    ref = _ref_reconstruct_h_raw(
                        h_raw, w_bf16, bias_in, vocab_chunk=8192, output_multiplier=mult, bf16_gemm=bf16_gemm
                    )
                    got = _reconstruct_h_raw(
                        h_raw, w_bf16, bias_in, vocab_chunk=8192, output_multiplier=mult, bf16_gemm=bf16_gemm
                    )
                    assert torch.equal(ref, got), (
                        f"Non-0-ULP diff={(ref - got).abs().max().item()} for "
                        f"bf16_gemm={bf16_gemm}, bias={bias_in is not None}, mult={mult}, n={n}"
                    )

    # In-place mutation of weight invalidates cache via _version
    h_probe = torch.randn(8, D, device=device, dtype=torch.bfloat16)
    w_mut = torch.randn(512, D, device=device, dtype=torch.bfloat16)
    out1 = _reconstruct_h_raw(h_probe, w_mut, None, vocab_chunk=256)
    w_mut.mul_(-1.0)
    out2 = _reconstruct_h_raw(h_probe, w_mut, None, vocab_chunk=256)
    ref2 = _ref_reconstruct_h_raw(h_probe, w_mut, None, vocab_chunk=256)
    assert torch.equal(out2, ref2)
    assert not torch.equal(out1, out2)

    # Weakref finalizer evicts cached cast when weight tensor is garbage-collected
    w_ephemeral = torch.randn(256, D, device=device, dtype=torch.bfloat16)
    _ = _reconstruct_h_raw(h_probe, w_ephemeral, None, vocab_chunk=128)
    assert len(_HEAD_CAST_CACHE) >= 1
    eph_id = id(w_ephemeral)
    del w_ephemeral
    gc.collect()
    assert all(k[0] != eph_id for k in _HEAD_CAST_CACHE)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="Requires >= 2 CUDA GPUs with NCCL support",
)
def test_nccl_2gpu_end_to_end_all_objectives_and_backends(tmp_path: Path):
    trace_dir, entries, idx_report, E_cpu, head_w_cpu, head_b_cpu, head_meta = _build_multishard_trace_bundle(
        tmp_path / "bundle"
    )
    dev0 = torch.device("cuda", 0)
    E_g0 = E_cpu.to(device=dev0, dtype=torch.float32)
    head_w_g0 = head_w_cpu.to(device=dev0)
    head_b_g0 = None if head_b_cpu is None else head_b_cpu.to(device=dev0)

    # (a) union_rest_grouped + signed_feature_hash + block_shuffle_window=4 + calibrate_beta + apply_calibrated_beta
    cfg_a = dict(
        objective="union_rest_grouped",
        sparse_backend="signed_feature_hash",
        block_shuffle_window=4,
        calibrate_beta=True,
        apply_calibrated_beta=True,
        m5_fit_tag="fit_a",
    )
    torch.manual_seed(404)
    random.seed(404)
    torch.cuda.manual_seed_all(404)
    rep_1gpu_a = fit(
        entries,
        idx_report,
        E_g0,
        head_w_g0,
        head_b_g0,
        head_meta,
        _nccl_train_args(tmp_path / "1gpu_a", device="cuda:0", **cfg_a),
    )

    cfg_b = dict(
        objective="union_rest_grouped",
        sparse_backend="exact_observed_pairs",
        block_shuffle_window=0,
        calibrate_beta=False,
        apply_calibrated_beta=False,
        m5_fit_tag="fit_b",
    )
    cfg_c = dict(
        objective="two_sided_pseudolikelihood",
        sparse_backend="signed_feature_hash",
        block_shuffle_window=2,
        calibrate_beta=False,
        apply_calibrated_beta=False,
        m5_fit_tag="fit_c",
    )

    jobs = [
        ("2gpu_a", cfg_a, None, "rank1_rep_a.pt"),
        ("2gpu_b", cfg_b, None, "rank1_rep_b.pt"),
        ("2gpu_c_direct", cfg_c, None, None),
        ("2gpu_c_resumed", cfg_c, 7, None),
        ("2gpu_c_resumed", cfg_c, None, None),
    ]
    mp.spawn(
        _nccl_suite_worker,
        args=(
            2,
            _free_port(),
            str(trace_dir),
            str(tmp_path),
            E_cpu,
            head_w_cpu,
            head_b_cpu,
            head_meta,
            jobs,
        ),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    rep_2gpu_a = torch.load(tmp_path / "2gpu_a/fit_a_smoke/latest.pt", map_location="cpu", weights_only=True)["report"]
    rep_2gpu_a_rank1 = torch.load(tmp_path / "rank1_rep_a.pt", map_location="cpu", weights_only=False)
    col_2gpu_r0 = rep_2gpu_a["stages"]["dense_sparse"]["sparse_collision_statistics"]
    col_2gpu_r1 = rep_2gpu_a_rank1["stages"]["dense_sparse"]["sparse_collision_statistics"]
    col_1gpu = rep_1gpu_a["stages"]["dense_sparse"]["sparse_collision_statistics"]
    assert col_2gpu_r1 == col_2gpu_r0
    for field in ("observed_lookup_events", "unique_observed_keys", "collision_count", "frequency_weighted_collision_count"):
        assert col_2gpu_r0[field] == col_1gpu[field]
    for stage in ("dense", "dense_sparse"):
        assert (
            rep_2gpu_a["stages"][stage]["beta_calibration"]["optimal_beta"]
            == rep_1gpu_a["stages"][stage]["beta_calibration"]["optimal_beta"]
        )
        m1, _ = load_adapter_checkpoint(Path(rep_1gpu_a["checkpoints"][stage]))
        m2, _ = load_adapter_checkpoint(Path(rep_2gpu_a["checkpoints"][stage]))
        for k, v1 in m1.state_dict().items():
            torch.testing.assert_close(m2.state_dict()[k], v1, atol=2e-6, rtol=2e-5)

    rep_2gpu_b_rank1 = torch.load(tmp_path / "rank1_rep_b.pt", map_location="cpu", weights_only=False)
    col_b = rep_2gpu_b_rank1["stages"]["dense_sparse"]["sparse_collision_statistics"]
    assert col_b["representation"] == "exact_observed_pairs"
    assert col_b["collision_count"] == 0
    exact_ckpt = tmp_path / "2gpu_b/fit_b_smoke/dense_sparse.pt"
    m_exact, _ = load_adapter_checkpoint(exact_ckpt)
    assert isinstance(m_exact.sparse, ExactObservedPairs)
    assert m_exact.sparse.num_occupied > 0

    mc_dir, _ = load_adapter_checkpoint(tmp_path / "2gpu_c_direct/fit_c_smoke/dense_sparse.pt")
    mc_res, _ = load_adapter_checkpoint(tmp_path / "2gpu_c_resumed/fit_c_smoke/dense_sparse.pt")
    for k, v in mc_dir.state_dict().items():
        torch.testing.assert_close(mc_res.state_dict()[k], v, rtol=0, atol=0)
