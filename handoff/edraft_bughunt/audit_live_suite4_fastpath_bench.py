#!/usr/bin/env python3
"""Lightweight `<5s` Read-Only Progressive & Complete Auditor for Suite 4 (`fastpath_bench`)
and Step 5 (`post_m4_suite_summary.json`) via `PostM4SuiteAuditor` (`33ccf725...`).

Supports:
  1. Progressive mode (default while Step 4 is not started or in progress): audits upstream
     M0-M4 immutability, completed sibling suite JSONs (`m2_refit_calibration.json`,
     `online_head_to_head.json`, `m5_eval.json`), and whichever Step-4 sub-artifacts
     currently exist in `fastpath_bench_<fp12>/`:
       - `adapter/adapter_benchmark_<scale>.json` (incl. `bf16_precision_audit`),
       - `grouped/grouped_benchmark_<scale>.json` (`parity.passed == True`),
       - `round3_micro_result.json` (`round3_micro`, `round4_online_microbench`, `round5_lookahead_microbench`),
       - `closed_loop_shards/result-rank{0..7}.json` (`eager_pre_round3` / `round3_compact` / `round3_compile`,
         NUMA/16-core pinning, `independent_repetition_dominated`, `T=0` mode-to-mode bit-exactness),
       - `cluster_1gpu_calibration/result-cal-rank0.json`.
  2. Complete mode (automatic when `fastpath_bench.json` / `post_m4_suite_summary.json` exist,
     or enforced via `--require_complete`): verifies `_suite_config_fingerprint("fastpath_bench", ...)`,
     runs `PostM4SuiteAuditor.audit_fastpath_bench()` + `audit_master_summary(...)` +
     `verify_immutability_at_end()`.
  3. `--self_test`: Runs a self-cleaning `<3s` `/tmp/edraft_bughunt` sandbox test exercising
     progressive mode, complete Step-4 + Step-5 mode, and adversarial fault injection.

Strictly CPU-only (`CUDA_VISIBLE_DEVICES=''`) and 100% read-only on `/home/catid/drafter`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CUDA_VISIBLE_DEVICES must be empty ('')"

REPO = Path("/home/catid/drafter")

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


def _discover_fb_workspace(suite_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    sym = suite_root / "fastpath_bench"
    if sym.exists():
        return sym.resolve()
    cands = sorted(
        (p for p in suite_root.glob("fastpath_bench_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def run_suite4_and_step5_audit(
    *,
    repo: Path,
    artifacts: Path,
    scale: str = "reference",
    workspace: Path | None = None,
    require_complete: bool = False,
    check_git_clean: bool = True,
) -> dict[str, Any]:
    import torch
    assert not torch.cuda.is_available()
    import audit_post_m4_suite as apm
    from build_and_audit_reports import independent_repetition_dominated
    from edraft import post_m4_suite
    import post_m4_suite_driver as pm4_drv

    t0 = time.monotonic()
    explicit_checks = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal explicit_checks
        if not cond:
            raise AssertionError(f"CHECK FAILED: {msg}")
        explicit_checks += 1

    suite_root = artifacts / "post_m4_suite"
    aud = apm.PostM4SuiteAuditor(
        repo=repo,
        artifacts=artifacts,
        scale=scale,
        allow_partial=not require_complete,
        require_complete=require_complete,
    )
    up_rep = aud.audit_upstream_m0_m4_immutability()

    # Check sibling suites 1-3 SHA256 & status if present
    sibling_suites: dict[str, dict[str, Any]] = {}
    for s_name in ("m2_refit_calibration", "online_head_to_head", "m5_eval"):
        sp = suite_root / f"{s_name}.json"
        if sp.is_file():
            raw = sp.read_bytes()
            sobj = json.loads(raw)
            if require_complete:
                check(sobj.get("status") == "complete", f"{s_name}.json status != complete")
            sibling_suites[s_name] = {
                "status": sobj.get("status"),
                "config_fingerprint": sobj.get("config_fingerprint"),
                "sha256": sha256_bytes(raw),
            }
        elif require_complete:
            check(False, f"--require_complete: missing {sp}")

    fb_ws = _discover_fb_workspace(suite_root, workspace)
    prog_info: dict[str, Any] = {
        "workspace": str(fb_ws) if fb_ws is not None else None,
        "adapter_microbench_done": False,
        "grouped_microbench_done": False,
        "round3_microbench_done": False,
        "closed_loop_rank_shards_done": [],
        "calibration_1gpu_done": False,
    }

    exp_modes = ("eager_pre_round3", "round3_compact", "round3_compile")
    exp_arms = tuple(post_m4_suite.DEFAULT_FASTPATH_CLOSED_LOOP_ARMS)

    if fb_ws is not None and fb_ws.is_dir():
        # 1) `adapter/adapter_benchmark_<scale>.json`
        ad_p = fb_ws / "adapter" / f"adapter_benchmark_{scale}.json"
        if ad_p.is_file():
            ad_obj = json.loads(ad_p.read_bytes())
            modes_dict = ad_obj.get("modes") or {}
            check(bool(modes_dict), f"empty modes in {ad_p}")
            for mk, mv in modes_dict.items():
                med = float((mv.get("synchronized_wall") or {}).get("median_ms", 0.0))
                check(math.isfinite(med) and med > 0.0, f"invalid median_ms on adapter mode {mk}")
            if "bf16_precision_audit" in ad_obj:
                bpa = ad_obj["bf16_precision_audit"]
                check(isinstance(bpa, dict) and bool(bpa), "empty bf16_precision_audit")
            prog_info["adapter_microbench_done"] = True

        # 2) `grouped/grouped_benchmark_<scale>.json`
        gr_p = fb_ws / "grouped" / f"grouped_benchmark_{scale}.json"
        if gr_p.is_file():
            gr_obj = json.loads(gr_p.read_bytes())
            check((gr_obj.get("parity") or {}).get("passed") is True, "grouped_kl parity.passed != True")
            check(float(gr_obj.get("forward_backward_speedup", 0.0)) > 0.0, "grouped_kl speedup <= 0")
            prog_info["grouped_microbench_done"] = True

        # 3) `round3_micro_result.json`
        r3_p = fb_ws / "round3_micro_result.json"
        if r3_p.is_file():
            r3_obj = json.loads(r3_p.read_bytes())
            rm = r3_obj.get("round3_micro") or {}
            check(
                bool(rm)
                and bool(rm.get("round4_online_microbench"))
                and bool(rm.get("round5_lookahead_microbench")),
                "missing round3/round4/round5 microbench blocks in round3_micro_result.json",
            )
            prog_info["round3_microbench_done"] = True

        # 4) `closed_loop_shards/result-rank{0..7}.json`
        sh_dir = fb_ws / "closed_loop_shards"
        if sh_dir.is_dir():
            seen_gpus, seen_cores = set(), set()
            all_speed_rows = []
            for rk in range(8):
                rp = sh_dir / f"result-rank{rk}.json"
                if not rp.is_file():
                    continue
                sp = json.loads(rp.read_bytes())
                check(int(sp.get("rank", -1)) == rk, f"result-rank{rk}.json rank mismatch")
                if scale != "smoke":
                    pg = int(sp["physical_gpu"])
                    cores = list(sp.get("cpu_affinity") or [])
                    check(
                        sp.get("cpu_affinity_applied") is True
                        and len(cores) == 16
                        and int(sp.get("numa_node", -1)) == pg // 2
                        and not (seen_cores & set(cores))
                        and pg not in seen_gpus,
                        f"rank {rk} fastpath shard affinity failure",
                    )
                    seen_gpus.add(pg)
                    seen_cores.update(cores)
                for rw in (sp.get("speed_rows") or []) + (sp.get("profile_rows") or []):
                    check(
                        rw.get("fastpath_mode") in exp_modes
                        and rw.get("arm") in exp_arms
                        and float(rw.get("tokens_per_second", 0.0)) > 0.0
                        and float(rw.get("elapsed_seconds", 0.0)) > 0.0,
                        f"invalid fastpath row on rank {rk}",
                    )
                    if str(rw.get("condition", "")).startswith("long_") and "output_tokens" in rw:
                        rep_flag = bool((rw.get("generation") or {}).get("repetition_dominated", rw.get("repetition_dominated", False)))
                        check(
                            independent_repetition_dominated(rw["output_tokens"]) == rep_flag,
                            f"repetition_dominated mismatch on fastpath rank {rk} {rw.get('prompt_id')}",
                        )
                all_speed_rows.extend(sp.get("speed_rows") or [])
                prog_info["closed_loop_rank_shards_done"].append(rk)

            # Verify T=0 greedy output_tokens parity across `eager_pre_round3`, `round3_compact`, `round3_compile` when output_tokens present!
            by_t0_slot: dict[tuple, dict[str, list[int]]] = {}
            for rw in all_speed_rows:
                if float(rw.get("temperature", -1.0)) == 0.0 and "output_tokens" in rw:
                    by_t0_slot.setdefault((rw["arm"], rw["prompt_id"], rw["repeat"]), {})[rw["fastpath_mode"]] = rw["output_tokens"]
            for slot_k, m_toks in by_t0_slot.items():
                if "eager_pre_round3" in m_toks and "round3_compact" in m_toks:
                    check(m_toks["eager_pre_round3"] == m_toks["round3_compact"], f"T=0 greedy mismatch eager vs round3_compact on {slot_k}")
                if "round3_compact" in m_toks and "round3_compile" in m_toks:
                    check(m_toks["round3_compact"] == m_toks["round3_compile"], f"T=0 greedy mismatch round3_compact vs round3_compile on {slot_k}")

        # 5) `cluster_1gpu_calibration/result-cal-rank0.json`
        cal_p = fb_ws / "cluster_1gpu_calibration" / "result-cal-rank0.json"
        if cal_p.is_file():
            cal_obj = json.loads(cal_p.read_bytes())
            c_rows = cal_obj.get("rows") or []
            c_pids = cal_obj.get("selected_prompt_ids") or []
            check(len(c_rows) > 0 and len(c_pids) > 0, "empty cluster_1gpu_calibration payload")
            for rw in c_rows:
                check(float(rw.get("tokens_per_second", 0.0)) > 0.0, "non-positive 1gpu cal tps")
            prog_info["calibration_1gpu_done"] = True

        # Check logs in `fb_ws` for errors
        for ldir in (fb_ws / "logs", fb_ws / "closed_loop_shards" / "logs", fb_ws / "cluster_1gpu_calibration" / "logs"):
            if ldir.is_dir():
                for lp in ldir.iterdir():
                    if lp.is_file():
                        ltxt = lp.read_text(errors="replace")
                        check("Traceback" not in ltxt and "FloatingPointError" not in ltxt, f"error in {lp}")

    # Complete Step 4: `fastpath_bench.json`
    fb_json_p = suite_root / "fastpath_bench.json"
    fb_complete_info = None
    if require_complete:
        check(fb_json_p.is_file(), f"--require_complete: missing {fb_json_p}")

    suite_results_for_master = dict(sibling_suites)
    if fb_json_p.is_file():
        fb_raw = fb_json_p.read_bytes()
        fb_sha = sha256_bytes(fb_raw)
        fb_rep = json.loads(fb_raw)

        if scale == "reference" and artifacts == (REPO / "artifacts"):
            step4 = next(s for s in pm4_drv.build_steps() if s["suite"] == "fastpath_bench")
            p_fp = post_m4_suite.common_parser("fp4")
            p_fp.add_argument("--suite", default="fastpath_bench")
            p_fp.add_argument("--gpus", default=None)
            p_fp.add_argument("--artifacts_root", type=Path, default=None)
            p_fp.add_argument("--no_resume", dest="resume", action="store_false", default=True)
            p_fp.add_argument("--bench_warmups", type=int, default=3)
            p_fp.add_argument("--bench_repeats", type=int, default=10)
            p_fp.add_argument("--bigram_count_gate_kappa", type=float, default=12.0)
            p_fp.add_argument("--timing_repeats", type=int, default=2)
            p_fp.add_argument("--timing_warmups", type=int, default=1)
            p_fp.add_argument("--timing_prompts", type=int, default=0)
            p_fp.add_argument("--max_new_tokens", type=int, default=2048)
            p_fp.add_argument("--long_max_new_tokens", type=int, default=24576)
            p_fp.add_argument("--smoke_tokens", type=int, default=32)
            args_s4 = post_m4_suite.parse_config(p_fp, step4["argv"][3:])
            exp_fp4 = post_m4_suite._suite_config_fingerprint("fastpath_bench", args_s4, artifacts)
            check(fb_rep.get("config_fingerprint") == exp_fp4, f"fastpath_bench config_fingerprint mismatch: {fb_rep.get('config_fingerprint')} vs {exp_fp4}")

        fb_aud = aud.audit_fastpath_bench()
        suite_results_for_master["fastpath_bench"] = fb_aud
        fb_complete_info = {
            "sha256": fb_sha,
            "status": fb_aud.get("status"),
            "config_fingerprint": fb_aud.get("config_fingerprint"),
            "paired_speedups": fb_aud.get("closed_loop_paired_speedups"),
            "aggregate_cluster_tps": (fb_aud.get("cluster_sharded_throughput") or {}).get("aggregate_cluster_tokens_per_second"),
        }

    # Complete Step 5: `post_m4_suite_summary.json`
    master_p = suite_root / "post_m4_suite_summary.json"
    master_info = None
    if require_complete:
        check(master_p.is_file(), f"--require_complete: missing {master_p}")
    if master_p.is_file():
        m_raw = master_p.read_bytes()
        m_obj = json.loads(m_raw)
        # Only enforce complete 4-suite master summary once all 4 suites have landed or require_complete=True
        all_4_present = len(suite_results_for_master) == 4
        if all_4_present or require_complete:
            m_aud = aud.audit_master_summary(suite_results_for_master)
            check(
                m_aud.get("status") == "complete"
                and set(m_aud.get("executed_suites") or ()) == {"m2_refit_calibration", "online_head_to_head", "m5_eval", "fastpath_bench"},
                f"incomplete post_m4_suite_summary.json: {m_aud}",
            )
            master_info = {
                "sha256": sha256_bytes(m_raw),
                "status": m_aud.get("status"),
                "executed_suites": m_aud.get("executed_suites"),
                "project_git_commit": m_aud.get("project_git_commit"),
            }
        else:
            master_info = {
                "sha256": sha256_bytes(m_raw),
                "status": m_obj.get("status"),
                "executed_suites": m_obj.get("executed_suites"),
                "note": "partial master summary from earlier suite invocation; Step-5 `--suite all` pending",
            }

    aud.verify_immutability_at_end()
    check(not aud.failures, f"PostM4SuiteAuditor failures: {aud.failures[:5]}")

    if check_git_clean:
        git_st = subprocess.check_output(["git", "-C", str(repo), "status", "--short", "-uno"], text=True).strip()
        check(git_st == "", f"dirty repo: {git_st}")

    return {
        "status": "PASSED",
        "mode": "complete" if (fb_complete_info is not None and master_info is not None and master_info.get("status") == "complete" and len(suite_results_for_master) == 4) else "progressive",
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "explicit_checks": explicit_checks,
        "post_m4_auditor_checks": aud.checks,
        "combined_checks": explicit_checks + aud.checks,
        "upstream_m0_m4_intact": up_rep["checked_upstream_files"],
        "sibling_suites": sibling_suites,
        "step4_progressive": prog_info,
        "step4_complete": fb_complete_info,
        "step5_master_summary": master_info,
    }


def _run_self_test() -> dict[str, Any]:
    """Self-cleaning `<3s` sandbox test of progressive Step 4, complete Step 4+5, and fault injection."""
    sb = Path(tempfile.mkdtemp(prefix="s4_selftest_", dir="/tmp/edraft_bughunt"))
    try:
        art = sb / "artifacts"
        sroot = art / "post_m4_suite"
        ws = sroot / "fastpath_bench_abcdef012345"
        (ws / "adapter").mkdir(parents=True)
        (ws / "grouped").mkdir(parents=True)
        (ws / "closed_loop_shards").mkdir(parents=True)
        (ws / "cluster_1gpu_calibration").mkdir(parents=True)

        # Write partial progressive artifacts first
        (ws / "adapter" / "adapter_benchmark_smoke.json").write_text(json.dumps({
            "modes": {"eager": {"synchronized_wall": {"median_ms": 0.45}}},
            "bf16_precision_audit": {"max_abs_err": 1e-3},
        }))
        (ws / "grouped" / "grouped_benchmark_smoke.json").write_text(json.dumps({
            "parity": {"passed": True},
            "forward_backward_speedup": 3.2,
            "precollated_forward_backward_speedup": 4.8,
        }))
        (ws / "round3_micro_result.json").write_text(json.dumps({
            "inventory": {"device": "cpu"},
            "round3_micro": {
                "speedup": 2.1,
                "round4_online_microbench": {"speedup": 2.5},
                "round5_lookahead_microbench": {"speedup": 3.1},
            },
        }))
        prog_res = run_suite4_and_step5_audit(
            repo=REPO, artifacts=art, scale="smoke", require_complete=False, check_git_clean=False,
        )
        assert prog_res["status"] == "PASSED" and prog_res["step4_progressive"]["round3_microbench_done"] is True

        # Now populate full 2-GPU Step-4 + 4-suite Step-5 sandbox and run `require_complete=True`!
        arms = ["plain_dflash", "online_bigram", "offline_static", "offline_online_stride1", "online_dbS"]
        modes = ["eager_pre_round3", "round3_compact", "round3_compile"]
        speed_rows = []
        profile_rows = []
        for rk, pid in ((0, "p0"), (1, "p1")):
            rk_speed = []
            rk_prof = []
            for m_idx, mn in enumerate(modes):
                for arm in arms:
                    tps = 100.0 * (1.0 + 0.2 * m_idx)
                    rw = {
                        "rank": rk,
                        "fastpath_mode": mn,
                        "arm": arm,
                        "condition": "standard_nonthinking",
                        "temperature": 0.0,
                        "prompt_id": pid,
                        "repeat": 0,
                        "generated_tokens": 64,
                        "output_tokens": [1, 2, 3, 4],
                        "elapsed_seconds": 64.0 / tps,
                        "tokens_per_second": tps,
                        "mean_accepted_length": 4.5,
                        "generation": {"repetition_dominated": False},
                    }
                    prw = dict(rw, phase_ms={"proposal_draw": 0.5})
                    rk_speed.append(rw)
                    rk_prof.append(prw)
                    speed_rows.append(rw)
                    profile_rows.append(prw)
            (ws / "closed_loop_shards" / f"result-rank{rk}.json").write_text(json.dumps({
                "rank": rk, "physical_gpu": rk, "numa_node": 0,
                "cpu_affinity": list(range(16 * rk, 16 * rk + 16)),
                "cpu_affinity_applied": True,
                "speed_rows": rk_speed,
                "profile_rows": rk_prof,
            }))

        cal_rows = [
            {"arm": arm, "prompt_id": "p0", "tokens_per_second": 120.0, "generation": {"repetition_dominated": False}}
            for arm in arms
        ]
        (ws / "cluster_1gpu_calibration" / "result-cal-rank0.json").write_text(json.dumps({
            "selected_prompt_ids": ["p0"],
            "rows": cal_rows,
        }))

        # Build exact `summary_by_mode_and_arm`, `paired_speedups`, and `cluster_sharded_throughput`
        sum_by_ma = []
        for mn in modes:
            for arm in arms:
                mr = [r for r in speed_rows if r["fastpath_mode"] == mn and r["arm"] == arm]
                mean_tps = statistics.mean(r["tokens_per_second"] for r in mr)
                med_tps = statistics.median(r["tokens_per_second"] for r in mr)
                sum_by_ma.append({
                    "fastpath_mode": mn, "arm": arm, "timed_runs": 2, "primary_timed_runs": 2,
                    "repetition_excluded": 0,
                    "mean_tokens_per_second": mean_tps, "median_tokens_per_second": med_tps,
                    "mean_accepted_length": 4.5,
                    "all_runs_mean_tokens_per_second": mean_tps, "all_runs_mean_accepted_length": 4.5,
                    "by_condition": {
                        "standard_nonthinking": {
                            "timed_runs": 2, "primary_timed_runs": 2, "repetition_excluded": 0,
                            "mean_tokens_per_second": mean_tps, "median_tokens_per_second": med_tps,
                            "mean_accepted_length": 4.5,
                            "all_runs_mean_tokens_per_second": mean_tps, "all_runs_mean_accepted_length": 4.5,
                        }
                    },
                    "mean_phase_ms": {"proposal_draw": 0.5},
                })
        paired_sp = [
            {
                "arm": arm,
                "round3_over_eager_mean_ratio": 1.2,
                "compile_over_round3_mean_ratio": 1.4 / 1.2,
                "compile_over_eager_mean_ratio": 1.4,
                "pairs": 2,
                "repetition_excluded_pairs": 0,
                "by_condition": {
                    "standard_nonthinking": {
                        "round3_over_eager_mean_ratio": 1.2,
                        "compile_over_round3_mean_ratio": 1.4 / 1.2,
                        "compile_over_eager_mean_ratio": 1.4,
                        "pairs": 2,
                    }
                },
            }
            for arm in arms
        ]
        r3_rows = [r for r in speed_rows if r["fastpath_mode"] == "round3_compact"]
        max_sec = sum(r["elapsed_seconds"] for r in r3_rows if r["rank"] == 0)
        tot_tok = sum(r["generated_tokens"] for r in r3_rows)
        cst = {
            "status": "complete",
            "world_size": 2,
            "gpus": [0, 1],
            "rank_affinity": {
                "0": {"physical_gpu": 0, "numa_node": 0, "cpu_affinity": list(range(0, 16)), "cpu_affinity_applied": True},
                "1": {"physical_gpu": 1, "numa_node": 0, "cpu_affinity": list(range(16, 32)), "cpu_affinity_applied": True},
            },
            "calibration_1gpu_gpu": 0,
            "calibration_selected_prompt_ids": ["p0"],
            "max_rank_round3_measured_seconds": max_sec,
            "total_generated_tokens": tot_tok,
            "aggregate_cluster_tokens_per_second": tot_tok / max_sec,
            "by_arm": [
                {
                    "arm": arm,
                    "primary_timed_runs": 2,
                    "mean_per_rank_tokens_per_second": 120.0,
                    "sum_across_ranks_tokens_per_second": 240.0,
                    "calibration_1gpu_tokens_per_second": 120.0,
                    "calibration_sharded_per_gpu_tokens_per_second": 120.0,
                    "sharded_over_1gpu_per_gpu_ratio": 1.0,
                    "cluster_over_1gpu_throughput_ratio": 2.0,
                    "per_gpu_scaling_efficiency": 1.0,
                }
                for arm in arms
            ],
        }
        (sroot / "fastpath_bench.json").write_text(json.dumps({
            "suite": "fastpath_bench",
            "scale": "smoke",
            "status": "complete",
            "config_fingerprint": "abcdef0123456789",
            "device": "cpu",
            "gpus": [0, 1],
            "adapter_microbenchmark": {"median_wall_ms": {"eager": 0.45}},
            "grouped_kl_microbenchmark": {"parity_passed": True, "forward_backward_speedup": 3.2},
            "round3_compact_microbenchmark": {"speedup": 2.1},
            "round4_online_microbenchmark": {"speedup": 2.5},
            "round5_lookahead_microbenchmark": {"speedup": 3.1},
            "closed_loop_matched_comparison": {
                "status": "complete",
                "is_pilot_subset": False,
                "arms": arms,
                "modes": modes,
                "speed_rows": speed_rows,
                "profile_rows": profile_rows,
                "summary_by_mode_and_arm": sum_by_ma,
                "paired_speedups": paired_sp,
            },
            "cluster_sharded_throughput": cst,
        }))
        for sn, fp in (("m2_refit_calibration", "fp1"), ("online_head_to_head", "fp2"), ("m5_eval", "fp3")):
            (sroot / f"{sn}.json").write_text(json.dumps({"suite": sn, "scale": "smoke", "status": "complete", "config_fingerprint": fp}))
        (sroot / "post_m4_suite_summary.json").write_text(json.dumps({
            "status": "complete",
            "scale": "smoke",
            "executed_suites": ["m2_refit_calibration", "online_head_to_head", "m5_eval", "fastpath_bench"],
            "project_git_commit": "b92f0fbfc685ad92034e8da1f54072f7a18ae8be",
            "suites": {
                "m2_refit_calibration": {"config_fingerprint": "fp1"},
                "online_head_to_head": {"config_fingerprint": "fp2"},
                "m5_eval": {"config_fingerprint": "fp3"},
                "fastpath_bench": {"config_fingerprint": "abcdef0123456789"},
            },
        }))

        comp_res = run_suite4_and_step5_audit(
            repo=REPO, artifacts=art, scale="smoke", require_complete=True, check_git_clean=False,
        )
        assert comp_res["status"] == "PASSED" and comp_res["mode"] == "complete"

        # Fault injection: drift `paired_speedups[0]["round3_over_eager_mean_ratio"]` -> must fail!
        bad_fb = json.loads((sroot / "fastpath_bench.json").read_text())
        bad_fb["closed_loop_matched_comparison"]["paired_speedups"][0]["round3_over_eager_mean_ratio"] = 9.99
        (sroot / "fastpath_bench.json").write_text(json.dumps(bad_fb))
        caught = False
        try:
            run_suite4_and_step5_audit(
                repo=REPO, artifacts=art, scale="smoke", require_complete=True, check_git_clean=False,
            )
        except AssertionError:
            caught = True
        assert caught, "failed to catch drifted paired_speedups!"
        return {
            "self_test": "PASSED",
            "progressive_checks": prog_res["combined_checks"],
            "complete_checks": comp_res["combined_checks"],
            "fault_injection_caught": caught,
        }
    finally:
        shutil.rmtree(sb, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight <5s Suite-4 fastpath_bench & Step-5 auditor")
    parser.add_argument("--artifacts", type=Path, default=REPO / "artifacts")
    parser.add_argument("--scale", default="reference")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()

    out: dict[str, Any] = {}
    if args.self_test:
        out["self_test_result"] = _run_self_test()
    out["live_result"] = run_suite4_and_step5_audit(
        repo=REPO,
        artifacts=args.artifacts,
        scale=args.scale,
        workspace=args.workspace,
        require_complete=args.require_complete,
        check_git_clean=True,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
