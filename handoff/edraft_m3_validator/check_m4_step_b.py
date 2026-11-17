#!/usr/bin/env python3
"""Ultra-lightweight incremental read-only monitor for 8-GPU Step B m4_timing."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/home/catid/drafter")
ART = REPO / "artifacts"
CACHE_PATH = Path("/tmp/edraft_m3_validator/step_b_validated_cache.json")

sys.path.insert(0, str(ART / "m4_evidence_validator"))
import online_checks  # noqa: E402


def main() -> int:
    now_utc = datetime.now(timezone.utc)
    st = json.loads((ART / "logs/m4_sharded_timing_state.json").read_text())
    print(f"UTC: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')} | DRIVER_STATUS: {st['status']}")

    drv_pid = st["driver_pid"]
    drv_alive = Path(f"/proc/{drv_pid}/stat").exists()
    inf = st.get("inflight") or {}
    inf_pid = inf.get("pid")
    inf_alive = Path(f"/proc/{inf_pid}/stat").exists() if inf_pid else False
    print(f"DRIVER_PID: {drv_pid} alive={drv_alive} | STEP: {inf.get('key')} pid={inf_pid} alive={inf_alive}")

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {
        "validated_keys": {},
        "total_online_checks": 0,
        "total_online_rows": 0,
        "failures": [],
    }
    validated = cache["validated_keys"]

    new_rows = 0
    new_online_rows = 0
    new_online_checks = 0
    rank_strs = []
    tot_complete = 0
    tot_timed = 0
    tot_prof = 0
    by_cond = {"long_thinking": 0, "standard_nonthinking": 0}

    for r in range(8):
        mp = ART / f"m4/timing/manifest-rank{r}.json"
        rc = l_rc = s_rc = 0
        last_age = None
        if mp.exists():
            m = json.loads(mp.read_text())
            mtimes = []
            for key, j in m.get("jobs", {}).items():
                if j.get("status") != "complete":
                    continue
                ident = j["identity"]
                tot_complete += 1
                rc += 1
                if ident.get("phase_profile"):
                    tot_prof += 1
                else:
                    tot_timed += 1
                cond = ident["condition"]
                by_cond[cond] = by_cond.get(cond, 0) + 1
                if cond == "long_thinking":
                    l_rc += 1
                else:
                    s_rc += 1
                rp = Path(j["result_path"])
                mtimes.append(rp.stat().st_mtime)

                if key not in validated:
                    raw = rp.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != j.get("result_sha256"):
                        cache["failures"].append(f"{key}: sha256 mismatch")
                    row = json.loads(raw)
                    new_rows += 1
                    if row["arm"] in online_checks.ONLINE_ARMS:
                        res = online_checks.check_online_row(
                            row,
                            expected_scale="reference",
                            expected_arm=row["arm"],
                            expected_setting=ident["setting"],
                            profile=bool(ident.get("phase_profile")),
                        )
                        new_online_rows += 1
                        new_online_checks += res["checks"]
                        if res["failures"]:
                            cache["failures"].extend(f"{key}: {f}" for f in res["failures"][:3])
                    validated[key] = 1
            if mtimes:
                last_age = int(time.time() - max(mtimes))
        # Each rank has 4 long prompts * 4 passes * 5 arms = 80 long rows + 1120 std rows = 1200 rows
        rank_strs.append(f"r{r}:{rc}/1200(L={l_rc}/80,S={s_rc}/1120,age={last_age}s)")

    cache["total_online_rows"] += new_online_rows
    cache["total_online_checks"] += new_online_checks
    CACHE_PATH.write_text(json.dumps(cache))

    print(
        f"PROGRESS: {tot_complete}/9600 ({tot_complete/9600*100:.2f}%) "
        f"[timed={tot_timed}/7200, profile={tot_prof}/2400 | long={by_cond['long_thinking']}/640, std={by_cond['standard_nonthinking']}/8960]"
    )
    print(
        f"INCREMENTAL_AUDIT: +{new_rows} new rows (+{new_online_rows} online) | "
        f"cumulative_online_rows={cache['total_online_rows']}/3840 | "
        f"cumulative_online_checks={cache['total_online_checks']} | failures={len(cache['failures'])}"
    )
    print("RANKS:", " ".join(rank_strs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
