#!/usr/bin/env python3
"""Fast 0-ULP Batch-Parallel (`ThreadPoolExecutor(8)`) Progressive & Complete Auditor
for Suite 3 (`m5_eval` / `m5_eval_ws_9343c88a80db`).

Supports:
  1. Progressive mode (default when `m5_eval.json` is not yet written): audits completed
     Offline Fit A (`m5_two_sided_pl`), Offline Fit B (`m5_exact_pairs`), `offline_kl_eval`,
     upstream M0-M4 + Suites 1-2 immutability, and whatever `m5/tune` (`0..380`),
     `m5/settings.json`, `m5/evaluate` (`0..4,800`), `greedy_output_comparisons.json`,
     and `m5/timing` (`0..14,400`) rows currently exist on disk.
  2. Complete mode (automatic when `artifacts/post_m4_suite/m5_eval.json` exists, or
     enforced via `--require_complete`): additionally verifies `_suite_config_fingerprint`,
     runs full `PostM4SuiteAuditor.audit_m5_eval()` + `verify_immutability_at_end()`,
     enforces 100% `T=0` greedy bit-exactness + `r0/r1/evaluate` trajectory parity across
     all `14,400` timing rows, and cross-checks paired `plain_dflash` overall & `8K-16K`
     deep-horizon deltas across all `19,580` JSON files.

Strictly CPU-only (`CUDA_VISIBLE_DEVICES=''`) and 100% read-only on `/home/catid/drafter`.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CUDA_VISIBLE_DEVICES must be empty ('')"

REPO = Path("/home/catid/drafter")
ART = REPO / "artifacts"
SUITE_DIR = ART / "post_m4_suite"

for p in ("/tmp/edraft_reporting", "/tmp/edraft_m3_validator", str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _discover_suite3_workspace(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    sym = SUITE_DIR / "m5_eval_workspace"
    if sym.exists():
        return sym.resolve()
    cands = sorted(SUITE_DIR.glob("m5_eval_ws_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No m5_eval_ws_* workspace found in {SUITE_DIR}")
    return cands[-1]


class _LocalRowAuditor:
    """Thread-local collector binding `PostM4SuiteAuditor._audit_single_m5_row` without shared state."""

    def __init__(self, single_row_fn) -> None:
        self.checks = 0
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self._fn = single_row_fn

    def check(self, cond: bool, msg: str) -> None:
        self.checks += 1
        if not cond:
            self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def audit_row(
        self,
        *,
        label: str,
        row: dict[str, Any],
        expected_c0_beta_relax: float | None,
        is_profile: bool,
    ) -> None:
        self._fn(
            self,
            label=label,
            row=row,
            expected_c0_beta_relax=expected_c0_beta_relax,
            expected_bigram_kappa=None,
            is_profile=is_profile,
        )


def _extract_8k_16k_stats(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return `(mean_accepted_length_8k_16k, tok_per_sec_8k_16k)` for blocks starting in `[8192, 16384)`."""
    acc_sum = 0.0
    tok_sum = 0.0
    sec_sum = 0.0
    n_blk = 0
    for blk in row.get("blocks") or ():
        off = int(blk.get("absolute_output_offset", 0))
        if 8192 <= off < 16384:
            acc = float(blk.get("accepted_length", 0))
            acc_sum += acc
            # Committed tokens in block = accepted + 1 (replacement or bonus), clipped by generated_tokens
            gen_t = int(row.get("generated_tokens", 0))
            committed = min(int(acc) + 1, max(0, gen_t - off))
            tok_sum += float(committed)
            sec_sum += float(blk.get("round_elapsed_seconds", 0.0))
            n_blk += 1
    if n_blk == 0:
        return None, None
    tps_bin = (tok_sum / sec_sum) if sec_sum > 0.0 else None
    return acc_sum / n_blk, tps_bin


_WORKER_UNBOUND_FN = None


def _get_worker_unbound_fn():
    global _WORKER_UNBOUND_FN
    if _WORKER_UNBOUND_FN is None:
        import audit_post_m4_suite as apm
        _WORKER_UNBOUND_FN = apm.PostM4SuiteAuditor._audit_single_m5_row
    return _WORKER_UNBOUND_FN


def _audit_raw_row_chunk(
    args_tuple: tuple[
        str,  # stage_kind: "tune" | "evaluate" | "timing"
        list[tuple[int, str, str, str]],  # (rank, job_key, result_path, expected_sha256)
        set[str],  # expected_arms
        float | None,  # expected_c0_beta_relax
        str | None,  # expected_frozen_settings_hash
    ],
) -> tuple[int, list[str], list[dict[str, Any]]]:
    stage_kind, items, expected_arms, c0_br, exp_fhash = args_tuple
    loc = _LocalRowAuditor(_get_worker_unbound_fn())
    summaries: list[dict[str, Any]] = []

    for rk, jkey, rpath_s, exp_sha in items:
        rp = Path(rpath_s)
        if not rp.is_file():
            loc.check(False, f"{stage_kind}/rank{rk}: missing {rpath_s}")
            continue
        rb = rp.read_bytes()
        loc.check(sha256_bytes(rb) == exp_sha, f"{stage_kind}/rank{rk}: SHA256 mismatch on {rp.name}")
        row = json.loads(rb)

        arm = str(row.get("arm", ""))
        cond = str(row.get("condition", ""))
        split = str(row.get("split", ""))
        loc.check(arm in expected_arms, f"{stage_kind}/rank{rk}: unexpected arm {arm!r} in {rp.name}")
        exp_split = "dev" if stage_kind == "tune" else "test"
        loc.check(split == exp_split, f"{stage_kind}/rank{rk}: split={split!r} != {exp_split!r} in {rp.name}")

        if exp_fhash is not None and stage_kind in ("evaluate", "timing"):
            rec_fh = row.get("frozen_settings_hash") or (row.get("identity") or {}).get("frozen_settings_hash")
            if rec_fh is not None:
                loc.check(rec_fh == exp_fhash, f"{stage_kind}/rank{rk}: frozen_settings_hash mismatch in {rp.name}")

        gen_toks = int(row.get("generated_tokens", 0))
        out_toks = row.get("output_tokens") or []
        el_s = float(row.get("elapsed_seconds", 0.0))
        tps = float(row.get("tokens_per_second", 0.0))
        tau = float(row.get("mean_accepted_length", -1.0))
        loc.check(
            gen_toks >= 1
            and len(out_toks) == gen_toks
            and math.isfinite(el_s) and el_s > 0.0
            and math.isfinite(tps) and tps > 0.0
            and math.isfinite(tau) and tau >= 0.0,
            f"{stage_kind}/rank{rk}: invalid token/timing scalars in {rp.name}",
        )

        blks = row.get("blocks") or ()
        bad_blk = False
        for blk in blks:
            re_s = blk.get("round_elapsed_seconds")
            if re_s is None or not (math.isfinite(float(re_s)) and float(re_s) > 0.0):
                bad_blk = True
                break
        loc.check(not bad_blk, f"{stage_kind}/rank{rk}: non-positive round_elapsed_seconds in {rp.name}")

        is_prof = bool(row.get("phase_profile", False)) if stage_kind == "timing" else False
        if stage_kind == "timing" and is_prof and arm != "plain_dflash" and len(blks) > 0:
            pms = row.get("phase_ms") or {}
            if arm in ("dense_candidate", "lookahead_static_candidate", "lookahead_rerelax_candidate"):
                loc.check(float(pms.get("proposal_draw", 0.0)) > 0.0, f"timing/rank{rk}: proposal_draw <= 0 on {rp.name}")
            else:
                loc.check(float(pms.get("online_update", 0.0)) > 0.0, f"timing/rank{rk}: online_update <= 0 on {rp.name}")

        loc.audit_row(
            label=f"{stage_kind}/rank{rk}:{rp.name}",
            row=row,
            expected_c0_beta_relax=c0_br,
            is_profile=is_prof,
        )

        ws_k = json.dumps(row.get("sampling_warp_config") or {}, sort_keys=True)
        temp = float(row.get("temperature", 0.0))
        pid = str(row.get("prompt_id", ""))
        seed = int(row.get("seed", 0))
        rep_dom = bool((row.get("generation") or {}).get("repetition_dominated", False))
        tau_8k, tps_8k = _extract_8k_16k_stats(row) if cond.startswith("long_") else (None, None)

        setting_dict = (row.get("identity") or {}).get("setting") or {}
        setting_sub = (
            f"lr={setting_dict['lr']}"
            if "lr" in setting_dict
            else (f"beta={setting_dict['beta']}" if "beta" in setting_dict else "default")
        )

        summaries.append({
            "rank": rk,
            "job_key": jkey,
            "arm": arm,
            "condition": cond,
            "temperature": temp,
            "warp_key": ws_k,
            "prompt_id": pid,
            "seed": seed,
            "is_profile": is_prof,
            "repeat_base": str(row.get("repeat", "0")).split(":epoch=")[0],
            "setting_sub": setting_sub,
            "rep_dom": rep_dom,
            "tau": tau,
            "tps": tps,
            "elapsed_s": el_s,
            "tau_8k_16k": tau_8k,
            "tps_8k_16k": tps_8k,
            "tokens_arr": array("i", out_toks),
        })

    return loc.checks, loc.failures, summaries


def _audit_substage_dir(
    sub_dir: Path,
    stage_kind: str,
    expected_arms: set[str],
    c0_br: float | None,
    exp_fhash: str | None,
    unbound_row_fn: Any,
    workers: int,
    check_fn,
    *,
    pool: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    sp_path = sub_dir / "sharding_plan.json"
    if sp_path.is_file():
        sp = json.loads(sp_path.read_bytes())
        check_fn(
            sp.get("world_size") == 8 and sp.get("sharding_mode") == "sharded_8gpu",
            f"{stage_kind}: sharding_plan world_size/mode mismatch",
        )

    sess_dir = sub_dir / "sessions"
    total_bundles_done = 0
    work_queue_info = None
    wq_files = sorted(sub_dir.glob("work_queue-*.json"))
    if wq_files:
        wq = json.loads(wq_files[-1].read_bytes())
        okeys = wq.get("ordered_keys") or []
        comp = wq.get("completed") or []
        claimed = wq.get("claimed") or {}
        check_fn(len(okeys) == 96, f"{stage_kind}: work_queue ordered_keys={len(okeys)} != 96")
        # Verify all 32 `long` prompts precede `standard` prompts
        check_fn(all("long" in k for k in okeys[:32]), f"{stage_kind}: first 32 work_queue keys must be long prompts")
        active_rk_cpus = {}
        for pkey, cinfo in claimed.items():
            rk_c = int(cinfo["rank"])
            pid_c = int(cinfo["pid"])
            st_p = Path(f"/proc/{pid_c}/status")
            cpus_allow = None
            if st_p.exists():
                for line in st_p.read_text(errors="replace").splitlines():
                    if line.startswith("Cpus_allowed_list:"):
                        cpus_allow = line.split(":", 1)[1].strip()
                        break
            active_rk_cpus[rk_c] = {"pid": pid_c, "prompt": pkey, "cpus_allowed": cpus_allow}
        work_queue_info = {
            "total_bundles": len(okeys),
            "completed_bundles": len(comp),
            "claimed_bundles": len(claimed),
            "active_workers": active_rk_cpus,
        }
        total_bundles_done = len(comp)

    if sess_dir.is_dir():
        sfiles = sorted(sess_dir.glob("*-rank*.json"))
        by_rk = {}
        for sf in sfiles:
            sobj = json.loads(sf.read_bytes())
            by_rk[int(sobj["rank"])] = sobj
        if len(by_rk) == 8:
            gpus, cores = set(), set()
            for r_idx in range(8):
                so = by_rk[r_idx]
                check_fn(
                    so.get("physical_gpu") == r_idx and so.get("cpu_affinity_applied") is True,
                    f"{stage_kind}: rank {r_idx} session GPU/CPU affinity invalid",
                )
                gpus.add(so["physical_gpu"])
                cores.update(so.get("cpu_affinity", []))
                total_bundles_done += sum(1 for b in (so.get("bundles") or {}).values() if b.get("status") == "complete")
            check_fn(gpus == set(range(8)) and len(cores) >= 128, f"{stage_kind}: rank GPU/CPU disjointness failure")

    items: list[tuple[int, str, str, str]] = []
    per_rank_done: dict[int, int] = {}
    seen_jkeys: set[str] = set()
    for rk in range(8):
        mf = sub_dir / f"manifest-rank{rk}.json"
        if not mf.is_file():
            continue
        mobj = json.loads(mf.read_bytes())
        cnt = 0
        for jkey, ent in (mobj.get("jobs") or {}).items():
            if ent.get("status") != "complete":
                continue
            check_fn(jkey not in seen_jkeys, f"{stage_kind}: duplicate job key {jkey}")
            seen_jkeys.add(jkey)
            items.append((rk, jkey, str(ent["result_path"]), str(ent["result_sha256"])))
            cnt += 1
        per_rank_done[rk] = cnt

    if not items:
        return {"completed_rows": 0, "per_rank_done": per_rank_done, "bundles_done": total_bundles_done, "work_queue": work_queue_info, "rows": [], "row_checks": 0}

    chunk_size = max(1, (len(items) + workers * 4 - 1) // (workers * 4))
    chunks = [
        (stage_kind, items[i : i + chunk_size], expected_arms, c0_br, exp_fhash)
        for i in range(0, len(items), chunk_size)
    ]
    all_rows: list[dict[str, Any]] = []
    row_checks = 0
    row_failures: list[str] = []
    if pool is not None:
        for ch_c, ch_f, ch_r in pool.map(_audit_raw_row_chunk, chunks):
            row_checks += ch_c
            row_failures.extend(ch_f)
            all_rows.extend(ch_r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as own_pool:
            for ch_c, ch_f, ch_r in own_pool.map(_audit_raw_row_chunk, chunks):
                row_checks += ch_c
                row_failures.extend(ch_f)
                all_rows.extend(ch_r)

    check_fn(not row_failures, f"{stage_kind} row failures ({len(row_failures)}): {row_failures[:5]}")
    return {
        "completed_rows": len(all_rows),
        "per_rank_done": per_rank_done,
        "bundles_done": total_bundles_done,
        "work_queue": work_queue_info,
        "rows": all_rows,
        "row_checks": row_checks,
    }


def _compute_paired_vs_plain(rows: list[dict[str, Any]], *, is_timing: bool) -> list[dict[str, Any]]:
    """Deterministic paired deltas vs `plain_dflash` (overall & `8K-16K` horizon, non-rep pairs)."""
    # Key: (condition, temperature, warp_key, prompt_id, seed[, repeat_base])
    plain_map: dict[tuple, dict[str, Any]] = {}
    by_group: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if is_timing and r["is_profile"]:
            continue
        k = (
            (r["condition"], r["temperature"], r["warp_key"], r["prompt_id"], r["seed"], r["repeat_base"])
            if is_timing
            else (r["condition"], r["temperature"], r["warp_key"], r["prompt_id"], r["seed"])
        )
        if r["arm"] == "plain_dflash":
            plain_map[k] = r
        by_group[(r["condition"], r["temperature"], r["arm"])].append(r)

    out = []
    for (cond, temp, arm), rlist in sorted(by_group.items()):
        prim = [x for x in rlist if not x["rep_dom"]]
        dtau_all, tpsr_all, dtau_8k, tpsr_8k = [], [], [], []
        for x in prim:
            k = (
                (x["condition"], x["temperature"], x["warp_key"], x["prompt_id"], x["seed"], x["repeat_base"])
                if is_timing
                else (x["condition"], x["temperature"], x["warp_key"], x["prompt_id"], x["seed"])
            )
            pr = plain_map.get(k)
            if pr is None or pr["rep_dom"]:
                continue
            dtau_all.append(x["tau"] - pr["tau"])
            if pr["tps"] > 0:
                tpsr_all.append(x["tps"] / pr["tps"])
            if x["tau_8k_16k"] is not None and pr["tau_8k_16k"] is not None:
                dtau_8k.append(x["tau_8k_16k"] - pr["tau_8k_16k"])
            if x["tps_8k_16k"] is not None and pr["tps_8k_16k"] is not None and pr["tps_8k_16k"] > 0:
                tpsr_8k.append(x["tps_8k_16k"] / pr["tps_8k_16k"])
        out.append({
            "condition": cond,
            "temperature": temp,
            "arm": arm,
            "primary_n": len(prim),
            "mean_tau": round(statistics.mean(x["tau"] for x in prim), 4) if prim else None,
            "mean_tps": round(statistics.mean(x["tps"] for x in prim), 2) if prim else None,
            "paired_n": len(dtau_all),
            "paired_delta_tau": round(statistics.mean(dtau_all), 4) if dtau_all else None,
            "paired_tps_ratio": round(statistics.mean(tpsr_all), 4) if tpsr_all else None,
            "deep_8k_16k_paired_n": len(dtau_8k),
            "deep_8k_16k_delta_tau": round(statistics.mean(dtau_8k), 4) if dtau_8k else None,
            "deep_8k_16k_tps_ratio": round(statistics.mean(tpsr_8k), 4) if tpsr_8k else None,
        })
    return out


def _compute_pairwise_long_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt-seed-paired head-to-head comparisons on `long_thinking` rows (plus complete 10-arm bundle subset)."""
    by_slot_arm: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["condition"] == "long_thinking" and not r["is_profile"]:
            by_slot_arm[(r["prompt_id"], r["seed"])][r["arm"]] = r

    complete_10arm_slots = sorted(s for s, m in by_slot_arm.items() if len(m) == 10)
    pairs_spec = [
        ("lookahead_static_candidate", "dense_candidate"),
        ("lookahead_rerelax_candidate", "dense_candidate"),
        ("exact_observed_pairs_online", "signed_feature_hash_online"),
        ("exact_observed_pairs_online", "offline_online_stride1"),
        ("signed_feature_hash_online", "offline_online_stride1"),
        ("online_buffered_stride5", "offline_online_stride5"),
        ("online_buffered_stride5", "offline_online_stride1"),
        ("online_two_sided_pl", "plain_dflash"),
    ]

    def _pair_stats(slots_iter, a1: str, a2: str) -> dict[str, Any] | None:
        dtau, tpsr, dtau_8k, tpsr_8k = [], [], [], []
        bit_eq = 0
        for s in slots_iter:
            m = by_slot_arm[s]
            if a1 not in m or a2 not in m:
                continue
            r1, r2 = m[a1], m[a2]
            if r1["rep_dom"] or r2["rep_dom"]:
                continue
            if r1["tokens_arr"] == r2["tokens_arr"]:
                bit_eq += 1
            dtau.append(r1["tau"] - r2["tau"])
            if r2["tps"] > 0:
                tpsr.append(r1["tps"] / r2["tps"])
            if r1["tau_8k_16k"] is not None and r2["tau_8k_16k"] is not None:
                dtau_8k.append(r1["tau_8k_16k"] - r2["tau_8k_16k"])
            if r1["tps_8k_16k"] is not None and r2["tps_8k_16k"] is not None and r2["tps_8k_16k"] > 0:
                tpsr_8k.append(r1["tps_8k_16k"] / r2["tps_8k_16k"])
        if not dtau:
            return None
        return {
            "paired_n": len(dtau),
            "bit_exact_trajectories": bit_eq,
            "delta_tau": round(statistics.mean(dtau), 4),
            "tps_ratio": round(statistics.mean(tpsr), 4) if tpsr else None,
            "deep_8k_16k_paired_n": len(dtau_8k),
            "deep_8k_16k_delta_tau": round(statistics.mean(dtau_8k), 4) if dtau_8k else None,
            "deep_8k_16k_tps_ratio": round(statistics.mean(tpsr_8k), 4) if tpsr_8k else None,
        }

    all_shared = {}
    complete_bundle_pairs = {}
    all_slots = sorted(by_slot_arm.keys())
    for a1, a2 in pairs_spec:
        k = f"{a1}_vs_{a2}"
        st_all = _pair_stats(all_slots, a1, a2)
        if st_all is not None:
            all_shared[k] = st_all
        st_comp = _pair_stats(complete_10arm_slots, a1, a2)
        if st_comp is not None:
            complete_bundle_pairs[k] = st_comp

    # Also summarize all 10 arms strictly on the complete 10-arm bundles vs plain_dflash
    complete_bundle_vs_plain = {}
    if complete_10arm_slots:
        arms_sorted = sorted({a for s in complete_10arm_slots for a in by_slot_arm[s]})
        for arm in arms_sorted:
            st_p = _pair_stats(complete_10arm_slots, arm, "plain_dflash")
            taus = [by_slot_arm[s][arm]["tau"] for s in complete_10arm_slots if not by_slot_arm[s][arm]["rep_dom"]]
            taus_8k = [by_slot_arm[s][arm]["tau_8k_16k"] for s in complete_10arm_slots if not by_slot_arm[s][arm]["rep_dom"] and by_slot_arm[s][arm]["tau_8k_16k"] is not None]
            complete_bundle_vs_plain[arm] = {
                "mean_tau": round(statistics.mean(taus), 4) if taus else None,
                "mean_tau_8k_16k": round(statistics.mean(taus_8k), 4) if taus_8k else None,
                **(st_p or {}),
            }

    return {
        "complete_10arm_long_bundles": len(complete_10arm_slots),
        "complete_10arm_prompts": [s[0] for s in complete_10arm_slots],
        "complete_10arm_vs_plain": complete_bundle_vs_plain,
        "complete_10arm_head_to_head": complete_bundle_pairs,
        "all_shared_head_to_head": all_shared,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast 0-ULP Suite-3 m5_eval progressive & complete auditor")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--skip_full_auditor", action="store_true", default=False)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    assert not torch.cuda.is_available()
    import audit_post_m4_suite as apm
    from edraft import post_m4_suite
    from edraft.runtime import fingerprint
    import post_m4_suite_driver as pm4_drv

    t0 = time.monotonic()
    explicit_checks = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal explicit_checks
        if not cond:
            raise AssertionError(f"CHECK FAILED: {msg}")
        explicit_checks += 1

    ws = _discover_suite3_workspace(args.workspace)
    check(ws.is_dir(), f"missing Suite-3 workspace {ws}")
    exp_arms_list = list(post_m4_suite.DEFAULT_M5_EVAL_ARMS)
    exp_arms = set(exp_arms_list)

    # 1. Upstream M0-M4 + completed Suites 1 & 2 immutability
    aud = apm.PostM4SuiteAuditor(
        repo=REPO,
        artifacts=ART,
        scale="reference",
        allow_partial=not args.require_complete,
        require_complete=args.require_complete,
    )
    up_rep = aud.audit_upstream_m0_m4_immutability()
    s1_p = SUITE_DIR / "m2_refit_calibration.json"
    s2_p = SUITE_DIR / "online_head_to_head.json"
    sibling_shas = {}
    for sp in (s1_p, s2_p):
        if sp.is_file():
            sibling_shas[sp.name] = sha256_file(sp)
            check(json.loads(sp.read_bytes()).get("status") == "complete", f"{sp.name} status != complete")

    # 2. Offline Fits A (`m5_two_sided_pl`) & B (`m5_exact_pairs`) + `offline_kl_eval`
    pl_p = ws / "m5_two_sided_pl.json"
    ex_p = ws / "m5_exact_pairs.json"
    okl_files = sorted((ws / "offline_kl_eval").glob("result-*.json")) if (ws / "offline_kl_eval").is_dir() else []
    check(pl_p.is_file() and ex_p.is_file() and len(okl_files) == 1, "missing Suite-3 offline fit or offline_kl_eval JSONs")

    pl_rep = json.loads(pl_p.read_bytes())
    ex_rep = json.loads(ex_p.read_bytes())
    okl = json.loads(okl_files[0].read_bytes())["results"]
    ex_sc = ex_rep["stages"]["dense_sparse"]["sparse_collision_statistics"]
    check(
        pl_rep.get("status") == "passed"
        and pl_rep.get("objective") == "two_sided_pseudolikelihood"
        and ex_rep.get("status") == "passed"
        and ex_rep.get("sparse_backend") == "exact_observed_pairs"
        and type(ex_sc.get("collision_count")) is int and ex_sc["collision_count"] == 0
        and type(ex_sc.get("active_representation_collision_count")) is int and ex_sc["active_representation_collision_count"] == 0
        and type(ex_sc.get("frequency_weighted_collision_count")) is int and ex_sc["frequency_weighted_collision_count"] == 0
        and ex_sc.get("overflow_events") == 0
        and ex_sc.get("dropped_unique_keys") == 0
        and okl["exact_sparse"]["collision_count"] == 0,
        "Suite-3 offline fit / ExactObservedPairs invariant failure",
    )

    # 3. Progressive / Complete Closed-Loop M5 Workspace (`tune`, `settings.json`, `evaluate`, `timing`)
    m5_dir = ws / "m5"
    settings_p = m5_dir / "settings.json"
    c0_br: float | None = 1.0
    fhash: str | None = None
    settings_sha256: str | None = None
    selected_settings = None
    if settings_p.is_file():
        s_raw = settings_p.read_bytes()
        settings_sha256 = sha256_bytes(s_raw)
        settings = json.loads(s_raw)
        check(
            settings.get("frozen") is True
            and settings.get("stage") == "m5"
            and settings.get("scale") == "reference",
            f"settings.json invalid header: frozen={settings.get('frozen')}, stage={settings.get('stage')}",
        )
        fhash = fingerprint(settings)
        selected_settings = settings.get("selected", {})
        check(set(selected_settings.keys()) == exp_arms, f"settings.json arms mismatch: {set(selected_settings.keys())}")
        dc_sel = selected_settings.get("dense_candidate") or {}
        if dc_sel.get("beta_relax") is not None:
            c0_br = float(dc_sel["beta_relax"])

    unbound_row_fn = apm.PostM4SuiteAuditor._audit_single_m5_row

    with ProcessPoolExecutor(max_workers=args.workers) as shared_pool:
        tune_res = (
            _audit_substage_dir(m5_dir / "tune", "tune", exp_arms, 1.0, None, unbound_row_fn, args.workers, check, pool=shared_pool)
            if (m5_dir / "tune").is_dir()
            else {"completed_rows": 0, "per_rank_done": {}, "bundles_done": 0, "rows": [], "row_checks": 0}
        )
        eval_res = (
            _audit_substage_dir(m5_dir / "evaluate", "evaluate", exp_arms, c0_br, fhash, unbound_row_fn, args.workers, check, pool=shared_pool)
            if (m5_dir / "evaluate").is_dir()
            else {"completed_rows": 0, "per_rank_done": {}, "bundles_done": 0, "rows": [], "row_checks": 0}
        )
        timing_res = (
            _audit_substage_dir(m5_dir / "timing", "timing", exp_arms, c0_br, fhash, unbound_row_fn, args.workers, check, pool=shared_pool)
            if (m5_dir / "timing").is_dir()
            else {"completed_rows": 0, "per_rank_done": {}, "bundles_done": 0, "rows": [], "row_checks": 0}
        )

    explicit_checks += tune_res["row_checks"] + eval_res["row_checks"] + timing_res["row_checks"]

    # Greedy output comparisons check (when present)
    greedy_p = m5_dir / "greedy_output_comparisons.json"
    greedy_info = None
    if greedy_p.is_file():
        greedy = json.loads(greedy_p.read_bytes())
        check(
            greedy.get("status") == "complete"
            and greedy.get("missing_comparisons") == []
            and int(greedy.get("completed_comparisons", -1)) == int(greedy.get("expected_comparisons", -2)),
            "incomplete greedy_output_comparisons.json",
        )
        for gg in greedy.get("groups", []):
            check(int(gg.get("missing_requests", 1)) == 0 and int(gg.get("compared_requests", 0)) == 64, f"bad greedy group {gg}")
            if gg.get("reference") == "plain_dflash":
                check(
                    int(gg.get("mismatched_sequences", 1)) == 0 and int(gg.get("mismatched_tokens", 1)) == 0,
                    f"T=0 greedy mismatch vs plain_dflash: {gg}",
                )
        greedy_info = {
            "completed_comparisons": greedy["completed_comparisons"],
            "sha256": sha256_file(greedy_p),
        }

    # Trajectory parity (`timing == evaluate` & `r0 == r1` & `T=0` greedy bit-exactness)
    parity_info = None
    if eval_res["rows"] and timing_res["rows"]:
        eval_by_slot = {
            (r["condition"], r["temperature"], r["warp_key"], r["prompt_id"], r["seed"], r["arm"]): r["tokens_arr"]
            for r in eval_res["rows"]
        }
        exact_vs_eval = 0
        compared_vs_eval = 0
        t0_checked = 0
        by_slot_rep: dict[tuple, dict[str, array]] = defaultdict(dict)
        divergences = []
        for tr in timing_res["rows"]:
            slot = (tr["condition"], tr["temperature"], tr["warp_key"], tr["prompt_id"], tr["seed"], tr["arm"])
            if slot in eval_by_slot:
                compared_vs_eval += 1
                if tr["tokens_arr"] == eval_by_slot[slot]:
                    exact_vs_eval += 1
                else:
                    divergences.append((tr["condition"], tr["temperature"], tr["prompt_id"], tr["arm"], "profile" if tr["is_profile"] else f"r{tr['repeat_base']}"))
                    check(tr["temperature"] > 0.0, f"T=0 greedy trajectory diverged between timing and evaluate on {slot}!")
                if tr["temperature"] == 0.0:
                    t0_checked += 1
            if not tr["is_profile"]:
                by_slot_rep[slot][tr["repeat_base"]] = tr["tokens_arr"]
        r0_r1_pairs = sum(1 for rm in by_slot_rep.values() if "0" in rm and "1" in rm)
        r0_eq_r1 = sum(1 for rm in by_slot_rep.values() if "0" in rm and "1" in rm and rm["0"] == rm["1"])
        for slot, rm in by_slot_rep.items():
            if slot[1] == 0.0 and "0" in rm and "1" in rm:
                check(rm["0"] == rm["1"], f"T=0 greedy r0 != r1 on {slot}!")
        parity_info = {
            "timing_eq_evaluate": f"{exact_vs_eval}/{compared_vs_eval}",
            "r0_eq_r1": f"{r0_eq_r1}/{r0_r1_pairs}",
            "t0_greedy_timing_verified": t0_checked,
            "stochastic_divergences": divergences,
        }

    # 4. Complete Mode: `artifacts/post_m4_suite/m5_eval.json` + `_suite_config_fingerprint` + `PostM4SuiteAuditor`
    suite3_report_p = SUITE_DIR / "m5_eval.json"
    complete_mode_info = None
    if args.require_complete:
        check(suite3_report_p.is_file(), f"--require_complete set, missing {suite3_report_p}")

    if suite3_report_p.is_file():
        s3_raw = suite3_report_p.read_bytes()
        s3_sha = sha256_bytes(s3_raw)
        s3_rep = json.loads(s3_raw)

        # Independently recompute `_suite_config_fingerprint` from `post_m4_suite_driver.build_steps()`
        step3 = next(s for s in pm4_drv.build_steps() if s["suite"] == "m5_eval")
        p_fp = post_m4_suite.common_parser("fp3")
        p_fp.add_argument("--suite", default="m5_eval")
        p_fp.add_argument("--gpus", default=None)
        p_fp.add_argument("--artifacts_root", type=Path, default=None)
        p_fp.add_argument("--no_resume", dest="resume", action="store_false", default=True)
        p_fp.add_argument("--m5_eval_arms", nargs="+", default=exp_arms_list)
        p_fp.add_argument("--m5_exact_overflow_policy", default="freeze_new")
        p_fp.add_argument("--m5_exact_sparse_capacity", type=int, default=None)
        p_fp.add_argument("--stats_memory_limit", default=None)
        p_fp.add_argument("--train_num_workers", type=int, default=2)
        p_fp.add_argument("--microbatch_blocks", type=int, default=None)
        p_fp.add_argument("--timing_repeats", type=int, default=2)
        p_fp.add_argument("--timing_warmups", type=int, default=1)
        p_fp.add_argument("--timing_prompts", type=int, default=0)
        p_fp.add_argument("--max_new_tokens", type=int, default=2048)
        p_fp.add_argument("--long_max_new_tokens", type=int, default=24576)
        p_fp.add_argument("--smoke_tokens", type=int, default=32)
        args_s3 = post_m4_suite.parse_config(p_fp, step3["argv"][3:])
        exp_fp = post_m4_suite._suite_config_fingerprint("m5_eval", args_s3, ART)

        check(
            s3_rep.get("status") == "complete"
            and s3_rep.get("config_fingerprint") == exp_fp
            and tune_res["completed_rows"] == 380
            and eval_res["completed_rows"] == 4800
            and timing_res["completed_rows"] == 14400,
            f"m5_eval.json completeness/fingerprint mismatch: status={s3_rep.get('status')}, fp={s3_rep.get('config_fingerprint')} vs {exp_fp}",
        )

        full_auditor_ran = False
        if not args.skip_full_auditor:
            s3_aud_res = aud.audit_m5_eval()
            full_auditor_ran = True
            check(s3_aud_res.get("status") == "complete", f"PostM4SuiteAuditor status={s3_aud_res.get('status')}")

        complete_mode_info = {
            "m5_eval_json_sha256": s3_sha,
            "config_fingerprint": exp_fp,
            "full_post_m4_auditor_executed": full_auditor_ran,
        }

    aud.verify_immutability_at_end()
    check(not aud.failures, f"PostM4SuiteAuditor failures: {aud.failures[:5]}")

    # Check logs + git cleanliness
    for lp in (ws / "logs").glob("*"):
        if lp.is_file():
            ltxt = lp.read_text(errors="replace")
            check("Traceback" not in ltxt and "FloatingPointError" not in ltxt, f"error in {lp}")
    git_st = subprocess.check_output(["git", "-C", str(REPO), "status", "--short", "-uno"], text=True).strip()
    check(git_st == "", f"dirty repo: {git_st}")

    total_json_files_audited = (
        tune_res["completed_rows"]
        + eval_res["completed_rows"]
        + timing_res["completed_rows"]
        + len(aud.read_sha256)
        + 3  # pl_json + ex_json + offline_kl_eval
    )
    elapsed = time.monotonic() - t0
    files_per_sec = round(total_json_files_audited / max(elapsed, 1e-6), 1)

    eval_paired = _compute_paired_vs_plain(eval_res["rows"], is_timing=False) if eval_res["rows"] else []
    eval_long_pairwise = _compute_pairwise_long_eval(eval_res["rows"]) if eval_res["rows"] else None
    timing_paired = _compute_paired_vs_plain(timing_res["rows"], is_timing=True) if timing_res["rows"] else []

    out_payload = {
        "status": "PASSED",
        "mode": "complete" if complete_mode_info is not None else "progressive",
        "workspace": str(ws),
        "elapsed_sec": round(elapsed, 2),
        "files_audited": total_json_files_audited,
        "throughput_files_per_sec": files_per_sec,
        "explicit_checks": explicit_checks,
        "post_m4_auditor_checks": aud.checks,
        "combined_checks": explicit_checks + aud.checks,
        "upstream_m0_m4_intact": up_rep["checked_upstream_files"],
        "sibling_suites_verified": sibling_shas,
        "stage_progress": {
            "tune": {
                "completed_rows": f"{tune_res['completed_rows']}/380",
                "per_rank_done": tune_res["per_rank_done"],
                "bundles_done": tune_res["bundles_done"],
            },
            "settings_sha256": settings_sha256,
            "settings_frozen_hash": fhash,
            "matched_c0_beta_relax": c0_br,
            "evaluate": {
                "completed_rows": f"{eval_res['completed_rows']}/4800",
                "per_rank_done": eval_res["per_rank_done"],
                "bundles_done": eval_res["bundles_done"],
                "work_queue": eval_res.get("work_queue"),
            },
            "timing": {
                "completed_rows": f"{timing_res['completed_rows']}/14400",
                "per_rank_done": timing_res["per_rank_done"],
                "bundles_done": f"{timing_res['bundles_done']}/2880",
            },
            "greedy_comparisons": greedy_info,
        },
        "trajectory_parity": parity_info,
        "complete_mode": complete_mode_info,
        "evaluate_paired_vs_plain": eval_paired,
        "evaluate_long_pairwise": eval_long_pairwise,
        "timing_paired_vs_plain": timing_paired,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out_payload, indent=2))
    print(json.dumps(out_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
