from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from edraft import post_m4_suite


def test_can_reuse_completed_fit_and_inplace_symlink_workspace(tmp_path: Path, monkeypatch):
    # 1. Synthetic workspace with completed report + checkpoints -> reused in-place when cfg_fp changes
    suite_root = tmp_path / "suite"
    suite_root.mkdir(parents=True)
    old_ws = suite_root / "m2_refit_ws_oldfingerprint"
    fit_dir = old_ws / "post_m4_shuf32_wd1e2_smoke"
    fit_dir.mkdir(parents=True)
    for ck in ("dense.pt", "dense_sparse.pt", "dense_calibrated.pt", "dense_sparse_calibrated.pt"):
        (fit_dir / ck).write_bytes(b"ckpt")

    src_dir = Path(post_m4_suite.__file__).parent
    impl_hashes = {
        name: post_m4_suite.file_digest(src_dir / name)
        for name in ("train.py", "adapter.py", "features.py", "sparse.py", "sparse_stats.py", "traces.py")
    }
    fake_report = {
        "status": "passed",
        "scale": "smoke",
        "block_shuffle_window": 32,
        "optimizer": {"weight_decay_W_A": 0.01},
        "implementation_sha256": impl_hashes,
        "checkpoints": {
            "dense": str(fit_dir / "dense.pt"),
            "dense_sparse": str(fit_dir / "dense_sparse.pt"),
        },
    }
    (old_ws / "post_m4_shuf32_wd1e2_smoke.json").write_text(json.dumps(fake_report))
    (suite_root / "m2_refit_workspace").symlink_to(old_ws.name)

    args = SimpleNamespace(
        scale="smoke",
        resume=True,
        refit_tag="post_m4_shuf32_wd1e2",
        block_shuffle_window=32,
        weight_decay=0.01,
    )
    flags = ["--block_shuffle_window", "32", "--weight_decay", "0.01", "--calibrate_beta"]
    reused = post_m4_suite._can_reuse_completed_fit(args, old_ws, "post_m4_shuf32_wd1e2_smoke", flags)
    assert reused is not None
    assert reused["checkpoints"]["dense"] == str(fit_dir / "dense.pt")

    # _run_training_fit must fast-return without invoking launch.main or train.main
    monkeypatch.setattr("edraft.launch.main", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("launch called")))
    monkeypatch.setattr("edraft.train.main", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("train called")))
    got = post_m4_suite._run_training_fit(args, old_ws, [0], "post_m4_shuf32_wd1e2", flags)
    assert got == fake_report

    # Rejects mismatching weight_decay, block_shuffle_window, --no_resume, or tampered implementation_sha256
    assert post_m4_suite._can_reuse_completed_fit(
        args, old_ws, "post_m4_shuf32_wd1e2_smoke",
        ["--block_shuffle_window", "32", "--weight_decay", "0.05", "--calibrate_beta"],
    ) is None
    assert post_m4_suite._can_reuse_completed_fit(
        args, old_ws, "post_m4_shuf32_wd1e2_smoke",
        ["--block_shuffle_window", "16", "--weight_decay", "0.01", "--calibrate_beta"],
    ) is None
    args_no_resume = SimpleNamespace(**{**vars(args), "resume": False})
    assert post_m4_suite._can_reuse_completed_fit(args_no_resume, old_ws, "post_m4_shuf32_wd1e2_smoke", flags) is None

    # 2. Read-only check against live /home/catid/drafter completed Stage-1+2 refit workspace if present
    live_ws = Path("/home/catid/drafter/artifacts/post_m4_suite/m2_refit_ws_2c7b2ed9dfa2")
    if (live_ws / "post_m4_shuf32_wd1e2.json").is_file():
        live_rep_raw = json.loads((live_ws / "post_m4_shuf32_wd1e2.json").read_text())
        impl_matches_current_tree = all(
            (src_dir / fname).is_file() and post_m4_suite.file_digest(src_dir / fname) == sha
            for fname, sha in live_rep_raw.get("implementation_sha256", {}).items()
        )
        ref_args = SimpleNamespace(
            scale="reference",
            resume=True,
            refit_tag="post_m4_shuf32_wd1e2",
            block_shuffle_window=32,
            weight_decay=0.01,
        )
        live_reused = post_m4_suite._can_reuse_completed_fit(ref_args, live_ws, "post_m4_shuf32_wd1e2", flags)
        if impl_matches_current_tree:
            assert live_reused is not None
            assert live_reused["status"] == "passed"
        else:
            assert live_reused is None
            monkeypatch.setattr(
                post_m4_suite,
                "file_digest",
                lambda p: live_rep_raw["implementation_sha256"][Path(p).name],
            )
            live_reused_matched = post_m4_suite._can_reuse_completed_fit(
                ref_args, live_ws, "post_m4_shuf32_wd1e2", flags
            )
            assert live_reused_matched is not None
            assert live_reused_matched["status"] == "passed"


def test_worker_idle_check_precedes_cuda_set_device_ordering(tmp_path: Path, monkeypatch):
    events: list[str] = []

    monkeypatch.setattr(post_m4_suite, "_is_cuda_run", lambda _args: True)
    monkeypatch.setattr(post_m4_suite, "apply_gpu_cpu_affinity", lambda dev, gpu: events.append(f"affinity:{dev}:{gpu}") or {})
    monkeypatch.setattr(post_m4_suite, "assert_idle", lambda dev: events.append(f"assert_idle:{dev}") or {"ok": True})
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: events.append(f"set_device:{dev}"))
    monkeypatch.setattr(post_m4_suite, "_measure_round3_compact_microbench", lambda _a, _d: {"speedup": 2.0})
    monkeypatch.setattr(post_m4_suite, "_prepare_offline_eval_context", lambda _a, _w, _d: None)
    monkeypatch.setattr(post_m4_suite, "_offline_kl_summary", lambda _c, p, _d: {"ckpt": str(p)})

    spec_r3 = tmp_path / "r3_spec.json"
    out_r3 = tmp_path / "r3_out.json"
    spec_r3.write_text(json.dumps({"physical_gpu": 0}))
    ns = SimpleNamespace(scale="reference", device="cuda:0")

    post_m4_suite._worker_round3_micro(ns, spec_r3, out_r3)
    assert events == ["affinity:cuda:0:0", "assert_idle:cuda:0", "set_device:cuda:0"]

    events.clear()
    spec_kl = tmp_path / "kl_spec.json"
    out_kl = tmp_path / "kl_out.json"
    spec_kl.write_text(json.dumps({"physical_gpu": 0, "workspace": str(tmp_path), "checkpoints": {"d": str(tmp_path / "d.pt")}}))
    post_m4_suite._worker_offline_kl_batch(ns, spec_kl, out_kl)
    assert events == ["affinity:cuda:0:0", "assert_idle:cuda:0", "set_device:cuda:0"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA GPU required for subprocess assert_idle check")
def test_real_cuda_subprocess_round3_micro_and_offline_kl_pass_assert_idle(tmp_path: Path):
    src_root = str((Path(__file__).resolve().parents[1] / "src").resolve())
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        PYTHONPATH=src_root,
        OMP_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    # 1) Real subprocess _worker_round3_micro on B300 GPU 0 with scale=reference (assert_idle active!)
    spec_r3 = tmp_path / "spec_r3.json"
    out_r3 = tmp_path / "out_r3.json"
    spec_r3.write_text(json.dumps({"physical_gpu": 0}))
    cmd_r3 = [
        sys.executable,
        "-m",
        "edraft.post_m4_suite",
        "--scale",
        "reference",
        "--device",
        "cuda:0",
        "--output",
        str(tmp_path / "out"),
        "--bench_warmups",
        "1",
        "--bench_repeats",
        "2",
        "--_worker_mode",
        "round3_micro",
        "--_worker_spec",
        str(spec_r3),
        "--_worker_out",
        str(out_r3),
    ]
    proc = subprocess.run(cmd_r3, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"round3_micro failed: {proc.stdout}\n{proc.stderr}"
    assert out_r3.is_file()

    # 2) Real subprocess _worker_offline_kl_batch against 2 real shards + m2/dense.pt (read-only) if present
    art = Path("/home/catid/drafter/artifacts")
    if (art / "m2" / "dense.pt").is_file() and (art / "m1_audit.json").is_file():
        full_audit = json.loads((art / "m1_audit.json").read_text())
        picked_train, picked_dev, picked_test = None, None, None
        for s_str in full_audit["shards"]:
            sp = Path(s_str)
            side = json.loads(sp.with_suffix(".pt.json").read_text())
            gen = side.get("generation", {})
            if gen.get("condition") == "standard_nonthinking" and gen.get("temperature") in (0.0, 0.7):
                if gen.get("split") == "train" and picked_train is None:
                    picked_train = str(sp)
                elif gen.get("split") == "dev" and picked_dev is None:
                    picked_dev = str(sp)
                elif gen.get("split") == "test" and picked_test is None:
                    picked_test = str(sp)
            if picked_train and picked_dev and picked_test:
                break
        assert picked_train and picked_dev and picked_test
        mini_ws = tmp_path / "mini_kl_ws"
        mini_traces = mini_ws / "traces"
        mini_traces.mkdir(parents=True)
        local_shards = []
        for orig_s in (picked_train, picked_dev, picked_test):
            orig_p = Path(orig_s)
            dst_pt = mini_traces / orig_p.name
            dst_json = mini_traces / orig_p.with_suffix(".pt.json").name
            dst_pt.symlink_to(orig_p.resolve())
            dst_json.symlink_to(orig_p.with_suffix(".pt.json").resolve())
            local_shards.append(str(dst_pt))
        for name in ("environment.json", "smoke.json"):
            if (art / name).exists():
                (mini_ws / name).symlink_to((art / name).resolve())
        (mini_ws / "m1_audit.json").write_text(
            json.dumps({"status": "passed", "scale": "reference", "shards": local_shards})
        )
        spec_kl = tmp_path / "spec_kl.json"
        out_kl = tmp_path / "out_kl.json"
        spec_kl.write_text(
            json.dumps(
                {
                    "physical_gpu": 0,
                    "workspace": str(mini_ws),
                    "checkpoints": {"base_dense": str(art / "m2" / "dense.pt")},
                }
            )
        )
        cmd_kl = [
            sys.executable,
            "-m",
            "edraft.post_m4_suite",
            "--scale",
            "reference",
            "--device",
            "cuda:0",
            "--output",
            str(mini_ws),
            "--_worker_mode",
            "offline_kl_batch",
            "--_worker_spec",
            str(spec_kl),
            "--_worker_out",
            str(out_kl),
        ]
        proc_kl = subprocess.run(cmd_kl, env=env, capture_output=True, text=True, timeout=60)
        assert proc_kl.returncode == 0, f"offline_kl_batch failed: {proc_kl.stdout}\n{proc_kl.stderr}"
        payload = json.loads(out_kl.read_text())
        assert "base_dense" in payload["results"]
