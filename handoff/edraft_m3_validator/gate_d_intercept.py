#!/usr/bin/env python3
"""Gate D status monitor & safe post-m4_evaluate intercept/archive helper."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/catid/drafter")
ART = REPO / "artifacts"
ARCHIVE_ROOT = Path("/tmp/edraft_09a5469_source")
EXPECTED_HEAD = "09a5469c72c442408d1a85af7ccd58c231db9fb9"

sys.path.insert(0, str(ART / "m4_evidence_validator"))
import online_checks  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def proc_start_ticks(pid: int) -> int | None:
    p = Path(f"/proc/{pid}/stat")
    if not p.exists():
        return None
    fields = p.read_text().rsplit(")", 1)[1].split()
    return int(fields[19])


def pgrp_members(pgrp: int) -> list[int]:
    if not pgrp or pgrp <= 1:
        return []
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgrp:
                found.append(int(entry.name))
        except OSError:
            continue
    return sorted(found)


def check_bindings(state: dict) -> tuple[int, int, list]:
    fab = state["frozen_artifact_binding"]
    ok = 0
    bad = []
    for abs_path, meta in fab.items():
        p = Path(abs_path)
        exp = meta["sha256"] if isinstance(meta, dict) else meta
        got = digest(p) if p.exists() else "MISSING"
        if got == exp:
            ok += 1
        else:
            bad.append((abs_path, got, exp))
    return ok, len(fab), bad


def cmd_status() -> int:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = json.loads((ART / "reference_continuation_state.json").read_text())
    ok_b, tot_b, bad_b = check_bindings(state)
    drv = state.get("driver_process") or {}
    inf = state.get("inflight") or {}
    inf_proc = inf.get("process") or {}

    drv_ticks = proc_start_ticks(drv["pid"]) if drv.get("pid") else None
    inf_ticks = proc_start_ticks(inf_proc["pid"]) if inf_proc.get("pid") else None

    print(f"UTC: {now_utc}")
    print(f"STATE: status={state['status']} step={inf.get('step')} attempt={inf.get('attempt_id')}")
    print(
        f"DRIVER: pid={drv.get('pid')} alive={drv_ticks is not None} "
        f"ticks_match={drv_ticks == drv.get('start_ticks')}"
    )
    print(
        f"INFLIGHT: pid={inf_proc.get('pid')} alive={inf_ticks is not None} "
        f"ticks_match={inf_ticks == inf_proc.get('start_ticks')}"
    )
    print(f"FROZEN_BINDINGS: {ok_b}/{tot_b} bad={bad_b}")

    s = json.loads((ART / "m4/settings.json").read_text())
    frozen_fp = fingerprint(s)

    tot = reused = fresh = online_rows = online_chk = online_fail = 0
    rank_summary = []
    for r in range(8):
        mp = ART / f"m4/evaluate/manifest-rank{r}.json"
        rc = l_rc = 0
        last_age = None
        if mp.exists():
            m = json.loads(mp.read_text())
            mtimes = []
            for j in m.get("jobs", {}).values():
                ident = j["identity"]
                if j.get("status") == "complete" and ident.get("frozen_settings_hash") == frozen_fp:
                    tot += 1
                    rc += 1
                    if ident["condition"] == "long_thinking":
                        l_rc += 1
                    rp = Path(j["result_path"])
                    mtimes.append(rp.stat().st_mtime)
                    row = json.loads(rp.read_text())
                    if (row.get("baseline_reuse") or {}).get("status") == "reused":
                        reused += 1
                    else:
                        fresh += 1
                    if row["arm"] in online_checks.ONLINE_ARMS:
                        online_rows += 1
                        res = online_checks.check_online_row(
                            row,
                            expected_scale="reference",
                            expected_arm=row["arm"],
                            expected_setting=ident["setting"],
                            profile=False,
                        )
                        online_chk += res["checks"]
                        if res["failures"]:
                            online_fail += len(res["failures"])
            if mtimes:
                last_age = int(time.time() - max(mtimes))
        rank_summary.append(f"r{r}:{rc}/300(long={l_rc}/20,age={last_age}s)")

    print(
        f"M4_EVAL: {tot}/2400 ({tot/2400*100:.1f}%) | reused={reused}/960 | "
        f"fresh={fresh}/1440 | online={online_rows}/960 ({online_chk} checks, {online_fail} fail)"
    )
    print("RANKS:", " ".join(rank_summary))
    return 0


def make_readonly_archive() -> dict:
    if ARCHIVE_ROOT.exists():
        # Temporarily restore write perms if re-archiving
        for root, dirs, files in os.walk(ARCHIVE_ROOT, topdown=False):
            for name in files:
                os.chmod(Path(root) / name, 0o644)
            for name in dirs:
                os.chmod(Path(root) / name, 0o755)
        os.chmod(ARCHIVE_ROOT, 0o755)
        shutil.rmtree(ARCHIVE_ROOT)

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    for label, src_dir in [
        ("src", REPO / "src"),
        ("artifacts/m4_evidence_validator", ART / "m4_evidence_validator"),
        ("m4_evidence_validator", ART / "m4_evidence_validator"),
    ]:
        dst_dir = ARCHIVE_ROOT / label
        shutil.copytree(
            src_dir,
            dst_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for src_p in sorted(src_dir.rglob("*")):
            if src_p.is_file() and "__pycache__" not in src_p.parts and src_p.suffix != ".pyc":
                rel = src_p.relative_to(src_dir)
                dst_p = dst_dir / rel
                h_src = digest(src_p)
                h_dst = digest(dst_p)
                if h_src != h_dst:
                    raise RuntimeError(f"Archive hash mismatch on {rel}: {h_src} != {h_dst}")
                manifest[f"{label}/{rel.as_posix()}"] = h_src

    manifest_path = ARCHIVE_ROOT / "MANIFEST_SHA256.json"
    manifest_path.write_text(
        json.dumps(
            {
                "git_head": EXPECTED_HEAD,
                "archived_at_utc": datetime.now(timezone.utc).isoformat(),
                "file_count": len(manifest),
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    for root, dirs, files in os.walk(ARCHIVE_ROOT, topdown=False):
        for name in files:
            os.chmod(Path(root) / name, 0o444)
        for name in dirs:
            os.chmod(Path(root) / name, 0o555)
    os.chmod(ARCHIVE_ROOT, 0o555)
    return {
        "archive_root": str(ARCHIVE_ROOT),
        "file_count": len(manifest),
        "manifest_sha256": digest(manifest_path),
    }


def cmd_intercept() -> int:
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"Unexpected git HEAD: {head}")

    state = json.loads((ART / "reference_continuation_state.json").read_text())
    ok_b, tot_b, bad_b = check_bindings(state)
    if bad_b or ok_b != tot_b:
        raise RuntimeError(f"Frozen binding mismatch before intercept: {bad_b}")

    eval_attempts = [a for a in state.get("attempts", []) if a.get("step") == "m4_evaluate"]
    if not eval_attempts or eval_attempts[-1].get("status") != "complete" or eval_attempts[-1].get("exit_code") != 0:
        raise RuntimeError(f"m4_evaluate has NOT completed cleanly yet: {eval_attempts[-1] if eval_attempts else None}")

    inflight = state.get("inflight")
    if not inflight or inflight.get("step") != "m4_timing":
        raise RuntimeError(f"Refusing intercept: expected inflight.step=='m4_timing', got {inflight}")

    # Confirm Gate D artifacts exist and are complete BEFORE signaling!
    for req_rel in (
        "m4/settings.json",
        "m4/greedy_output_comparisons.json",
        "m4/summary.json",
        "m4/feature_context.json",
    ):
        req_p = ART / req_rel
        if not req_p.is_file():
            raise RuntimeError(f"Missing required Gate D artifact before intercept: {req_p}")

    greedy_doc = json.loads((ART / "m4/greedy_output_comparisons.json").read_text())
    summary_doc = json.loads((ART / "m4/summary.json").read_text())
    if greedy_doc.get("status") != "complete" or greedy_doc.get("completed_comparisons") != 640:
        raise RuntimeError("m4/greedy_output_comparisons.json is not 640/640 complete!")
    if summary_doc.get("completed_requests") != 2400 or summary_doc.get("missing_requests") != []:
        raise RuntimeError("m4/summary.json does not have 2400/2400 completed_requests!")

    preserved_hashes_before = {
        rel: digest(ART / rel)
        for rel in (
            "m4/settings.json",
            "m4/greedy_output_comparisons.json",
            "m4/summary.json",
            "m4/feature_context.json",
        )
    }

    # Verify child & driver PIDs and start_ticks
    drv = state["driver_process"]
    drv_pid = drv["pid"]
    drv_exp_ticks = drv["start_ticks"]

    child_proc = inflight["process"]
    child_pid = child_proc["pid"]
    child_pgrp = inflight["process_group"]
    child_exp_ticks = child_proc["start_ticks"]

    got_child_ticks = proc_start_ticks(child_pid)
    if got_child_ticks is not None:
        if got_child_ticks != child_exp_ticks:
            raise RuntimeError(f"Child PID {child_pid} start_ticks mismatch: {got_child_ticks} != {child_exp_ticks}")
        print(f"Sending SIGTERM to 1-GPU m4_timing process group {child_pgrp} (PID {child_pid})...")
        os.killpg(child_pgrp, signal.SIGTERM)

    # Wait up to 20s for child group AND driver to exit cleanly
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        c_alive = pgrp_members(child_pgrp)
        d_ticks = proc_start_ticks(drv_pid)
        if not c_alive and d_ticks is None:
            break
        time.sleep(0.5)

    # If driver is somehow still alive after child exited, verify start_ticks and SIGTERM driver
    d_ticks = proc_start_ticks(drv_pid)
    if d_ticks is not None:
        if d_ticks != drv_exp_ticks:
            raise RuntimeError(f"Driver PID {drv_pid} start_ticks mismatch: {d_ticks} != {drv_exp_ticks}")
        print(f"Sending SIGTERM to driver PID {drv_pid}...")
        os.kill(drv_pid, signal.SIGTERM)
        deadline2 = time.monotonic() + 15.0
        while time.monotonic() < deadline2 and proc_start_ticks(drv_pid) is not None:
            time.sleep(0.5)

    if pgrp_members(child_pgrp) or proc_start_ticks(drv_pid) is not None:
        raise RuntimeError(
            f"Processes still alive after SIGTERM: child_pgrp={pgrp_members(child_pgrp)}, drv={proc_start_ticks(drv_pid)}"
        )

    # Verify 0 GPU compute apps across all GPUs 0..7 (give kernel 2s to release GPU contexts)
    for _ in range(10):
        gpu_apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not gpu_apps:
            break
        time.sleep(1.0)
    if gpu_apps:
        raise RuntimeError(f"GPUs not idle after intercept: {gpu_apps}")

    # Remove ONLY partial 1-GPU m4/timing inflight artifacts
    removed = []
    timing_dir = ART / "m4/timing"
    if timing_dir.exists():
        shutil.rmtree(timing_dir)
        removed.append(str(timing_dir))
    timing_json = ART / "m4/timing.json"
    if timing_json.exists():
        timing_json.unlink()
        removed.append(str(timing_json))

    # Verify preserved artifacts are 100% byte-identical
    for rel, exp_h in preserved_hashes_before.items():
        got_h = digest(ART / rel)
        if got_h != exp_h:
            raise RuntimeError(f"Preserved artifact {rel} changed! {got_h} != {exp_h}")

    archive_info = make_readonly_archive()
    final_state = json.loads((ART / "reference_continuation_state.json").read_text())
    print(
        json.dumps(
            {
                "status": "gate_d_intercept_complete",
                "driver_final_status": final_state.get("status"),
                "driver_inflight": final_state.get("inflight"),
                "driver_surviving_owned_processes": final_state.get("surviving_owned_processes"),
                "gpus_idle": True,
                "removed_partial_timing_paths": removed,
                "preserved_artifact_sha256": preserved_hashes_before,
                "archive": archive_info,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("status", "intercept"), default="status")
    args = parser.parse_args()
    return cmd_status() if args.mode == "status" else cmd_intercept()


if __name__ == "__main__":
    raise SystemExit(main())
