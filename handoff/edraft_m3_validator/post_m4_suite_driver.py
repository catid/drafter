#!/usr/bin/env python3
"""Detached SIGHUP-immune sequential runner for 8-GPU post_m4_suite (immediate 8-GPU saturation order)."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/catid/drafter")
ART = REPO / "artifacts"
PYTHON = REPO / ".venv/bin/python"
LOG_PATH = ART / "logs/post_m4_suite_driver.log"
STATE_PATH = ART / "logs/post_m4_suite_state.json"
LOCK_PATH = Path("/tmp/edraft_m3_validator/post_m4_suite_driver.lock")

SUITE_ORDER = (
    "m2_refit_calibration",
    "online_head_to_head",
    "m5_eval",
    "fastpath_bench",
    "all",
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


def wait_all_gpus_idle(timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    last_out = ""
    while time.monotonic() < deadline:
        last_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not last_out:
            return
        time.sleep(1.0)
    raise RuntimeError(f"GPUs 0..7 not idle before suite step launch: {last_out}")


def build_steps() -> list[dict]:
    suite_src = (REPO / "src/edraft/post_m4_suite.py").read_text()
    has_explicit_resume_flag = "'--resume'" in suite_src or '"--resume"' in suite_src
    steps = []
    for suite_name in SUITE_ORDER:
        argv = [
            str(PYTHON),
            "-m",
            "edraft.post_m4_suite",
            "--scale",
            "reference",
            "--suite",
            suite_name,
            "--device",
            "cuda:0",
            "--gpus",
            "0,1,2,3,4,5,6,7",
            "--allow_large_run",
            "--output",
            str(ART),
        ]
        if has_explicit_resume_flag:
            argv.append("--resume")
        steps.append(
            {
                "key": f"suite_{suite_name}",
                "suite": suite_name,
                "log": str(ART / f"logs/post_m4_{suite_name}.log"),
                "argv": argv,
            }
        )
    return steps


def run_driver() -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        ncpu = os.cpu_count() or 256
        os.sched_setaffinity(0, set(range(ncpu)))
    except OSError:
        pass

    steps = build_steps()
    state: dict = {
        "driver_pid": os.getpid(),
        "driver_process": proc_info(os.getpid()),
        "started_at_utc": now_utc(),
        "status": "running",
        "inflight": None,
        "completed_steps": [],
        "steps": steps,
    }
    atomic_json(STATE_PATH, state)

    def log_msg(msg: str) -> None:
        print(f"{now_utc()} [post_m4_suite_driver] {msg}", flush=True)

    try:
        for step in steps:
            wait_all_gpus_idle(timeout_sec=45.0)
            env = dict(os.environ, **BASE_ENV)
            for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"):
                env.pop(k, None)

            step_log = Path(step["log"])
            step_log.parent.mkdir(parents=True, exist_ok=True)
            log_msg(f"Starting {step['key']}: {' '.join(step['argv'])}")

            with step_log.open("ab", buffering=0) as sf:
                sf.write(f"\n{now_utc()} starting {step['key']}\n".encode())
                child = subprocess.Popen(
                    step["argv"],
                    cwd=str(REPO),
                    env=env,
                    stdout=sf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                inflight = {
                    "key": step["key"],
                    "suite": step["suite"],
                    "pid": child.pid,
                    "process_group": child.pid,
                    "process": proc_info(child.pid),
                    "started_at_utc": now_utc(),
                    "log": str(step_log),
                    "argv": step["argv"],
                }
                state["inflight"] = inflight
                state["updated_at_utc"] = now_utc()
                atomic_json(STATE_PATH, state)

                while child.poll() is None:
                    time.sleep(5.0)
                    inflight["last_observed_at_utc"] = now_utc()
                    state["updated_at_utc"] = now_utc()
                    atomic_json(STATE_PATH, state)

                rc = child.wait()

            rec = dict(inflight, exit_code=rc, finished_at_utc=now_utc())
            state["inflight"] = None
            state["completed_steps"].append(rec)
            state["updated_at_utc"] = now_utc()
            atomic_json(STATE_PATH, state)

            if rc != 0:
                raise RuntimeError(f"{step['key']} exited with status {rc}; inspect {step_log}")
            log_msg(f"Finished {step['key']} cleanly with exit_code=0")

        state["status"] = "complete"
        state["completed_at_utc"] = now_utc()
        state["updated_at_utc"] = now_utc()
        atomic_json(STATE_PATH, state)
        log_msg("All 4 post-M4 suites + unified master summary completed cleanly with exit_code=0!")
        return 0
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = now_utc()
        state["updated_at_utc"] = now_utc()
        atomic_json(STATE_PATH, state)
        log_msg(f"FATAL: {state['error']}")
        return 1


def spawn_daemon() -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    lock_f = LOCK_PATH.open("a+b")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: post_m4_suite_driver lock already held!", file=sys.stderr)
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
                    "log": str(LOG_PATH),
                    "state": str(STATE_PATH),
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
    log_fd = os.open(str(LOG_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    os.dup2(devnull_fd, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    if devnull_fd > 2:
        os.close(devnull_fd)
    if log_fd > 2:
        os.close(log_fd)

    rc = run_driver()
    os._exit(rc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"steps": build_steps(), "log": str(LOG_PATH), "state": str(STATE_PATH)}, indent=2))
        return 0
    if args.spawn:
        return spawn_daemon()
    parser.error("Specify --dry_run or --spawn")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
