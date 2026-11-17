#!/usr/bin/env python3
"""Detached SIGHUP-immune sequential runner for 8-GPU Phase-B ablations & `--scale full` pipeline.

Chains all 4 Phase-B suites plus the final cache-verified unified summary (`--suite all`):
  1. `bigram_eviction`      (`--scale reference --suite bigram_eviction --timing_repeats 2 --bf16_gemm --fast_decode`)
  2. `full_vocab_lookahead` (`--scale reference --suite full_vocab_lookahead --timing_repeats 2 --bf16_gemm --fast_decode`)
  3. `pure_t1_fit`          (`--scale reference --suite pure_t1_fit --timing_repeats 2 --bf16_gemm --fast_decode`)
  4. `full_scale`           (`--scale full --suite full_scale --allow_large_run --timing_repeats 2 --bf16_gemm --fast_decode`)
  5. `all`                  (`--scale full --suite all --allow_large_run --timing_repeats 2 --bf16_gemm --fast_decode`)

All outputs are isolated under `/home/catid/drafter/artifacts/full_scale` while reading
frozen reference prerequisites from `/home/catid/drafter/artifacts` (SHA256-verified
before and after every suite step).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

DEFAULT_REPO = Path("/home/catid/drafter")
DEFAULT_REF_ART = DEFAULT_REPO / "artifacts"
DEFAULT_OUT_ART = DEFAULT_REF_ART / "full_scale"
DEFAULT_PYTHON = DEFAULT_REPO / ".venv/bin/python"
DEFAULT_LOG_PATH = DEFAULT_REF_ART / "logs/phase_b_suite_driver.log"
DEFAULT_STATE_PATH = DEFAULT_REF_ART / "logs/phase_b_suite_state.json"
DEFAULT_LOCK_PATH = Path("/tmp/edraft_m3_validator/phase_b_suite_driver.lock")

PHASE_B_STEP_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("bigram_eviction", "reference", False),
    ("full_vocab_lookahead", "reference", False),
    ("pure_t1_fit", "reference", False),
    ("full_scale", "full", True),
    ("all", "full", True),
)

BASE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "NCCL_NVLS_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def proc_info(pid: int) -> dict | None:
    stat_p = Path(f"/proc/{pid}/stat")
    if not stat_p.exists():
        return None
    fields = stat_p.read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "state": fields[0],
        "process_group": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _parse_gpu_list(gpus_str: str) -> list[int]:
    gpus = [int(tok.strip()) for tok in str(gpus_str).split(",") if tok.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or any(g < 0 for g in gpus) or len(gpus) > 8:
        raise ValueError(f"Invalid --gpus specification: {gpus_str!r}")
    return gpus


def wait_selected_gpus_idle(gpus: list[int], timeout_sec: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_sec
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    uuids = {int(line.split(",")[0]): line.split(",")[1].strip() for line in query.splitlines() if line.strip()}
    missing = set(gpus) - uuids.keys()
    if missing:
        raise RuntimeError(f"Unknown GPU indices in --gpus: {sorted(missing)}")
    target_uuids = {uuids[i] for i in gpus}

    last_occupied: list[str] = []
    while time.monotonic() < deadline:
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
            text=True,
        ).splitlines()
        last_occupied = [
            line.strip()
            for line in apps
            if line.strip() and line.split(",")[0].strip() in target_uuids
        ]
        if not last_occupied:
            return
        time.sleep(1.0)
    raise RuntimeError(f"Selected GPUs {gpus} not idle before Phase-B suite launch: {'; '.join(last_occupied)}")


def build_steps(args) -> list[dict]:
    repo = Path(args.repo).resolve()
    # Preserve virtualenv symlink path (do NOT .resolve() .venv/bin/python or pyvenv.cfg is bypassed!)
    py_exe = Path(os.path.abspath(args.python))
    out_art = Path(args.output).resolve()
    ref_art = Path(args.reference_artifacts).resolve()
    gpus_list = _parse_gpu_list(args.gpus)
    gpus_csv = ",".join(str(g) for g in gpus_list)
    is_smoke = (args.scale_mode == "smoke")
    bf16_gemm = bool(getattr(args, "bf16_gemm", True))
    eff_timing_repeats = (
        int(args.smoke_timing_repeats)
        if is_smoke
        else int(getattr(args, "timing_repeats", 2))
    )
    if eff_timing_repeats < 2:
        raise ValueError(f"--timing_repeats must be >= 2 (got {eff_timing_repeats})")

    if not is_smoke and out_art == ref_art:
        raise ValueError("--output cannot equal --reference_artifacts on production Phase-B runs")

    steps: list[dict] = []
    for suite_name, prod_scale, needs_large_run in PHASE_B_STEP_SPECS:
        eff_scale = "smoke" if is_smoke else prod_scale
        eff_device = "cpu" if is_smoke else "cuda:0"
        argv = [
            str(py_exe),
            "-m",
            "edraft.phase_b_suite",
            "--scale",
            eff_scale,
            "--suite",
            suite_name,
            "--device",
            eff_device,
            "--gpus",
            gpus_csv,
            "--output",
            str(out_art),
            "--reference_artifacts",
            str(ref_art),
            "--timing_repeats",
            str(eff_timing_repeats),
        ]
        if not is_smoke and (needs_large_run or eff_scale == "full"):
            argv.append("--allow_large_run")
        if bf16_gemm:
            argv.append("--bf16_gemm")
        if args.fast_decode:
            argv.append("--fast_decode")
        if not args.resume:
            argv.append("--no_resume")
        if is_smoke:
            argv.extend([
                "--smoke_tokens",
                str(args.smoke_tokens),
                "--timing_warmups",
                "1",
                "--block_shuffle_window",
                "2",
            ])
        steps.append(
            {
                "key": f"suite_{suite_name}",
                "suite": suite_name,
                "scale": eff_scale,
                "log": str(out_art / f"logs/phase_b_{suite_name}.log"),
                "argv": argv,
            }
        )
    return steps


def run_driver(args) -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if args.scale_mode != "smoke":
        try:
            ncpu = os.cpu_count() or 256
            os.sched_setaffinity(0, set(range(ncpu)))
        except OSError:
            pass

    repo = Path(args.repo).resolve()
    out_art = Path(args.output).resolve()
    ref_art = Path(args.reference_artifacts).resolve()
    state_path = Path(args.state_path).resolve()
    gpus_list = _parse_gpu_list(args.gpus)
    is_smoke = (args.scale_mode == "smoke")
    steps = build_steps(args)

    active_child: list[subprocess.Popen | None] = [None]

    def _forward_term(signum: int, _frame) -> None:
        ch = active_child[0]
        if ch is not None and ch.poll() is None:
            try:
                os.killpg(ch.pid, signum)
            except ProcessLookupError:
                pass
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _forward_term)
    signal.signal(signal.SIGINT, _forward_term)

    state: dict = {
        "driver": "phase_b_suite_driver",
        "driver_pid": os.getpid(),
        "driver_process": proc_info(os.getpid()),
        "repo": str(repo),
        "output_root": str(out_art),
        "reference_artifacts": str(ref_art),
        "scale_mode": args.scale_mode,
        "gpus": gpus_list,
        "timing_repeats": int(args.smoke_timing_repeats) if is_smoke else int(getattr(args, "timing_repeats", 2)),
        "bf16_gemm": bool(getattr(args, "bf16_gemm", True)),
        "fast_decode": bool(args.fast_decode),
        "started_at_utc": now_utc(),
        "status": "running",
        "inflight": None,
        "completed_steps": [],
        "steps": steps,
    }
    atomic_json(state_path, state)

    def log_msg(msg: str) -> None:
        print(f"{now_utc()} [phase_b_suite_driver] {msg}", flush=True)

    src_path = str((repo / "src").resolve())
    existing_py = os.environ.get("PYTHONPATH", "")
    worker_pythonpath = f"{src_path}{os.pathsep}{existing_py}" if existing_py else src_path

    try:
        for step in steps:
            if not is_smoke:
                wait_selected_gpus_idle(gpus_list, timeout_sec=60.0)

            env = dict(os.environ, **BASE_ENV, PYTHONPATH=worker_pythonpath)
            for stale in (
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
                "CUDA_VISIBLE_DEVICES",
                "EDRAFT_ARTIFACTS_ROOT",
            ):
                env.pop(stale, None)
            if is_smoke:
                env["CUDA_VISIBLE_DEVICES"] = ""

            step_log = Path(step["log"])
            step_log.parent.mkdir(parents=True, exist_ok=True)
            log_msg(f"Starting {step['key']} (scale={step['scale']}): {' '.join(step['argv'])}")

            with step_log.open("ab", buffering=0) as sf:
                sf.write(f"\n{now_utc()} starting {step['key']}\n".encode())
                child = subprocess.Popen(
                    step["argv"],
                    cwd=str(repo),
                    env=env,
                    stdout=sf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active_child[0] = child
                inflight = {
                    "key": step["key"],
                    "suite": step["suite"],
                    "scale": step["scale"],
                    "pid": child.pid,
                    "process_group": child.pid,
                    "process": proc_info(child.pid),
                    "started_at_utc": now_utc(),
                    "log": str(step_log),
                    "argv": step["argv"],
                }
                state["inflight"] = inflight
                state["updated_at_utc"] = now_utc()
                atomic_json(state_path, state)

                poll_interval = 0.25 if is_smoke else 5.0
                while child.poll() is None:
                    time.sleep(poll_interval)
                    inflight["last_observed_at_utc"] = now_utc()
                    state["updated_at_utc"] = now_utc()
                    atomic_json(state_path, state)

                rc = child.wait()
                active_child[0] = None

            rec = dict(inflight, exit_code=rc, finished_at_utc=now_utc())
            state["inflight"] = None
            state["completed_steps"].append(rec)
            state["updated_at_utc"] = now_utc()
            atomic_json(state_path, state)

            if rc != 0:
                raise RuntimeError(f"{step['key']} exited with status {rc}; inspect {step_log}")
            log_msg(f"Finished {step['key']} cleanly with exit_code=0")

        state["status"] = "complete"
        state["completed_at_utc"] = now_utc()
        state["updated_at_utc"] = now_utc()
        atomic_json(state_path, state)
        log_msg("All 4 Phase-B suites + unified master summary completed cleanly with exit_code=0!")
        return 0
    except BaseException as exc:
        ch = active_child[0]
        if ch is not None and ch.poll() is None:
            try:
                os.killpg(ch.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = now_utc()
        state["updated_at_utc"] = now_utc()
        atomic_json(state_path, state)
        log_msg(f"FATAL: {state['error']}")
        return 1


def spawn_daemon(args) -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log_path = Path(args.log_path).resolve()
    state_path = Path(args.state_path).resolve()
    lock_path = Path(args.lock_path).resolve()
    # Validate step construction eagerly in foreground before daemonizing:
    _ = build_steps(args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_f = lock_path.open("a+b")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"ERROR: phase_b_suite_driver lock already held ({lock_path})!", file=sys.stderr)
        return 1
    os.set_inheritable(lock_f.fileno(), True)

    r_fd, w_fd = os.pipe()
    pid1 = os.fork()
    if pid1 > 0:
        os.close(w_fd)
        lock_f.close()
        os.waitpid(pid1, 0)
        grandchild_bytes = os.read(r_fd, 64).decode().strip()
        os.close(r_fd)
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "driver_pid": int(grandchild_bytes),
                    "log": str(log_path),
                    "state": str(state_path),
                }
            )
        )
        return 0

    os.close(r_fd)
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    pid2 = os.fork()
    if pid2 > 0:
        os.write(w_fd, str(pid2).encode())
        os.close(w_fd)
        os._exit(0)

    os.close(w_fd)
    devnull_fd = os.open(os.devnull, os.O_RDONLY)
    log_fd = os.open(str(log_path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    os.dup2(devnull_fd, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    if devnull_fd > 2:
        os.close(devnull_fd)
    if log_fd > 2:
        os.close(log_fd)

    rc = run_driver(args)
    os._exit(rc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true", help="Print resolved step plan JSON and exit")
    parser.add_argument("--spawn", action="store_true", help="Spawn detached SIGHUP-immune background driver")
    parser.add_argument("--foreground", action="store_true", help="Run driver synchronously in foreground")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_ART)
    parser.add_argument("--reference_artifacts", type=Path, default=DEFAULT_REF_ART)
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--state_path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--lock_path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--scale_mode",
        choices=["production", "smoke"],
        default="production",
        help="'production' runs reference secondary ablations -> full_scale; 'smoke' runs CPU-only smoke dry-test",
    )
    parser.add_argument(
        "--timing_repeats",
        type=int,
        default=2,
        help="Timed repeats per prompt/regime in production mode (must be >= 2; default 2)",
    )
    parser.add_argument(
        "--no_bf16_gemm",
        dest="bf16_gemm",
        action="store_false",
        default=True,
        help="Disable BF16 TensorCore adapter/online/lookahead GEMMs",
    )
    parser.add_argument("--no_fast_decode", dest="fast_decode", action="store_false", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--smoke_tokens", type=int, default=8)
    parser.add_argument("--smoke_timing_repeats", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = int(bool(args.dry_run)) + int(bool(args.spawn)) + int(bool(args.foreground))
    if modes != 1:
        parser.error("Specify exactly one of --dry_run, --spawn, or --foreground")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "steps": build_steps(args),
                    "log": str(Path(args.log_path).resolve()),
                    "state": str(Path(args.state_path).resolve()),
                    "lock": str(Path(args.lock_path).resolve()),
                },
                indent=2,
            )
        )
        return 0
    if args.foreground:
        return run_driver(args)
    return spawn_daemon(args)


if __name__ == "__main__":
    raise SystemExit(main())
