#!/usr/bin/env python3
"""Read-only Socket-1 progress & online invariant check for active M4 stage."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/home/catid/drafter")
ART = REPO / "artifacts"
sys.path.insert(0, str(ART / "m4_evidence_validator"))
import online_checks  # noqa: E402


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    state = json.loads((ART / "reference_continuation_state.json").read_text())
    print(f"UTC: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"STATE_STATUS: {state['status']} | STAGE_GATES: {list(state.get('stage_gates', {}).keys())}")

    for label, proc in [
        ("driver", state.get("driver_process")),
        ("inflight", (state.get("inflight") or {}).get("process")),
    ]:
        if not proc:
            print(f"PROC_{label}: None")
            continue
        pid = proc["pid"]
        p = Path(f"/proc/{pid}")
        if not p.exists():
            print(f"PROC_{label}: pid={pid} MISSING")
            continue
        stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        sig_ign = [
            l for l in (p / "status").read_text().splitlines() if l.startswith("SigIgn:")
        ][0].split()[1]
        st_ticks = int(stat[19])
        print(
            f"PROC_{label}: pid={pid} state={stat[0]} start_ticks={st_ticks} "
            f"match={st_ticks == proc['start_ticks']} SigIgn=0x{sig_ign}"
        )

    fab = state["frozen_artifact_binding"]
    ok_bind = 0
    bad_bind = []
    for abs_path, meta in fab.items():
        p = Path(abs_path)
        exp = meta["sha256"] if isinstance(meta, dict) else meta
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
        if got == exp:
            ok_bind += 1
        else:
            bad_bind.append((abs_path, got, exp))
    print(f"FROZEN_BINDINGS: {ok_bind}/{len(fab)} bad={bad_bind}")

    inflight = state.get("inflight") or {}
    step = inflight.get("step")
    print(f"ACTIVE_STEP: {step} | attempt={inflight.get('attempt_id')} | started={inflight.get('started_at_utc')}")

    impl_id = "60da3e0c57fee2fbd9cd7ce756c66c56ba5642a63da27ef428b9b30538a459df"

    if (ART / "m4" / "tune" / "manifest-rank0.json").exists():
        man = json.loads((ART / "m4" / "tune" / "manifest-rank0.json").read_text())
        sess_files = sorted((ART / "m4" / "tune" / "sessions").glob("*.json"))
        sess = json.loads(sess_files[-1].read_text())
        sess_id = sess["logical_session"]
        dev_ids = sess["configuration"]["configuration"]["dev_ids"]

        jobs = []
        for jid, j in man["jobs"].items():
            ident = j.get("identity") or {}
            if (
                ident.get("timing_session") == sess_id
                and (ident.get("checkpoint") or {}).get("implementation") == impl_id
                and j.get("status") == "complete"
            ):
                jobs.append((jid, j))

        by_prompt = {pid: [] for pid in dev_ids}
        online_rows = 0
        online_checks_total = 0
        online_failures = []
        basic_checks = 0
        rep_flagged = []

        for jid, j in jobs:
            ident = j["identity"]
            rpath = Path(j["result_path"])
            raw = rpath.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == j["result_sha256"], f"SHA mismatch {rpath}"
            basic_checks += 1
            row = json.loads(raw)
            pid = row["prompt_id"]
            by_prompt.setdefault(pid, []).append(row)

            assert math.isfinite(row["tokens_per_second"]) and row["tokens_per_second"] > 0
            assert math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
            assert math.isfinite(row["mean_accepted_length"]) and row["mean_accepted_length"] >= 0
            assert row["generation"]["repetition_detector_version"] == 2
            basic_checks += 4
            if row["temperature"] > 0:
                assert row["verification_rule"] == "ratio"
                basic_checks += 1
            if row["generation"].get("repetition_dominated"):
                rep_flagged.append(
                    (pid, row["arm"], row.get("settings"), row["generated_tokens"])
                )

            if row["arm"] in online_checks.ONLINE_ARMS:
                online_rows += 1
                res = online_checks.check_online_row(
                    row,
                    expected_scale="reference",
                    expected_arm=row["arm"],
                    expected_setting=ident["setting"],
                    profile=bool(ident["phase_profile"]),
                )
                online_checks_total += res["checks"]
                if res["failures"]:
                    online_failures.append((pid, row["arm"], res["failures"][:5]))

        bundles_done = len(sess.get("bundles", []))
        std_done = sum(1 for pid in dev_ids[:8] if len(by_prompt[pid]) == 9)
        long_done = sum(1 for pid in dev_ids[8:] if len(by_prompt[pid]) == 9)
        print(
            f"M4_TUNE_PROGRESS: rows={len(jobs)}/180 ({len(jobs)/180*100:.1f}%), "
            f"bundles={bundles_done}/20 (std={std_done}/8, long={long_done}/12)"
        )
        print(
            f"M4_TUNE_VALIDATION: basic_checks={basic_checks} online_rows={online_rows} "
            f"online_checks={online_checks_total} online_failures={len(online_failures)} "
            f"rep_dominated={len(rep_flagged)} {rep_flagged}"
        )

        # Matched M4/M3 dev wall-clock ETA
        m3_man = json.loads((ART / "m3" / "tune" / "manifest-rank0.json").read_text())
        m3_wall = {pid: 0.0 for pid in dev_ids}
        for jid, j in m3_man["jobs"].items():
            ident = j.get("identity") or {}
            if (
                ident.get("timing_session") == "00df5018aa1946d6821bd38a71de77b8"
                and (ident.get("checkpoint") or {}).get("implementation") == impl_id
                and j.get("status") == "complete"
            ):
                r3 = json.loads(Path(j["result_path"]).read_bytes())
                m3_wall[r3["prompt_id"]] = m3_wall.get(r3["prompt_id"], 0.0) + r3["elapsed_seconds"]

        m4_matched_long = 0.0
        m3_matched_long = 0.0
        m3_rem_long = 0.0
        for idx, pid in enumerate(dev_ids):
            rlist = by_prompt[pid]
            w4 = sum(r["elapsed_seconds"] for r in rlist)
            w3 = m3_wall.get(pid, 0.0)
            if idx >= 8:
                if len(rlist) == 9:
                    m4_matched_long += w4
                    m3_matched_long += w3
                    print(f"  [DONE {idx+1:02d}/20] {pid:22s}: 9/9 rows | m4={w4:6.1f}s | m3={w3:6.1f}s | ratio={w4/w3:.3f}x")
                else:
                    frac_rem = (9 - len(rlist)) / 9.0
                    m3_rem_long += w3 * frac_rem
                    if len(rlist) > 0:
                        print(f"  [ACTV {idx+1:02d}/20] {pid:22s}: {len(rlist)}/9 rows | m4_partial={w4:6.1f}s | m3_full={w3:6.1f}s")
                    else:
                        print(f"  [PEND {idx+1:02d}/20] {pid:22s}: 0/9 rows | m3_full={w3:6.1f}s")

        ratio = m4_matched_long / m3_matched_long if m3_matched_long > 0 else 9.0 / 7.0
        # Account for elapsed time already spent on the currently in-flight row!
        latest_mtime = max(Path(j["result_path"]).stat().st_mtime for _, j in jobs) if jobs else now_utc.timestamp()
        inflight_row_elapsed = max(0.0, now_utc.timestamp() - latest_mtime)
        est_rem_sec = max(0.0, m3_rem_long * ratio - inflight_row_elapsed)
        eta_utc = now_utc + timedelta(seconds=est_rem_sec)
        print(
            f"LONG_RATIO: {ratio:.3f}x | inflight_row_elapsed={inflight_row_elapsed:.0f}s | "
            f"EST_REMAINING: {est_rem_sec/3600:.2f}h (ETA ~{eta_utc.strftime('%Y-%m-%d %H:%M UTC')})"
        )

        # Also summarize cumulative dev mean_accepted_length by candidate across completed bundles
        cand_stats = {}
        for pid in dev_ids:
            if len(by_prompt[pid]) == 9:
                cond = by_prompt[pid][0]["condition"]
                for r in by_prompt[pid]:
                    key = (cond, r["arm"], json.dumps(r["settings"], sort_keys=True))
                    cand_stats.setdefault(key, []).append(
                        (r["mean_accepted_length"], r["tokens_per_second"], r["generation"]["repetition_dominated"])
                    )
        print("COMPLETED_BUNDLE_SUMMARY (mean_acc, mean_tps, n_prompts, n_rep):")
        for (cond, arm, sjson), vals in sorted(cand_stats.items()):
            non_rep = [v for v in vals if not v[2]]
            m_acc = sum(v[0] for v in non_rep) / len(non_rep) if non_rep else float("nan")
            m_tps = sum(v[1] for v in non_rep) / len(non_rep) if non_rep else float("nan")
            n_rep = sum(1 for v in vals if v[2])
            print(f"  {cond:20s} | {arm:22s} | {sjson:95s} | acc={m_acc:.4f} | tps={m_tps:6.2f} | n={len(vals)} rep={n_rep}")


if __name__ == "__main__":
    main()
