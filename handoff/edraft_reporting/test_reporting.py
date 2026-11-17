"""Unit and fault-injection tests for `/tmp/edraft_reporting/build_and_audit_reports.py`."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, "/tmp/edraft_reporting")
import build_and_audit_reports as bar  # noqa: E402

REPO = Path("/home/catid/drafter")
SMOKE_FIXTURE = Path("/tmp/edraft_reporting/smoke_fixture")


def test_safety_guard_rejects_output_dir_inside_repo() -> None:
    with pytest.raises(ValueError, match="must NOT lie inside --repo"):
        bar.run_audit(
            [
                "--repo",
                str(REPO),
                "--artifacts",
                str(SMOKE_FIXTURE),
                "--scale",
                "smoke",
                "--output_dir",
                str(REPO / "artifacts" / "illegal_write"),
            ]
        )


def test_smoke_end_to_end_zero_drift_and_m4_validator(tmp_path: Path) -> None:
    assert SMOKE_FIXTURE.is_dir(), "smoke_fixture must be generated before running test suite"
    out_dir = tmp_path / "smoke_out"
    payload = bar.run_audit(
        [
            "--repo",
            str(REPO),
            "--artifacts",
            str(SMOKE_FIXTURE),
            "--scale",
            "smoke",
            "--require_complete",
            "--output_dir",
            str(out_dir),
        ]
    )
    assert payload["status"] == "passed", payload["failures"]
    assert payload["failures"] == []
    assert payload["m4_evidence_validator"] is not None
    assert payload["m4_evidence_validator"]["status"] == "passed"
    assert payload["m4_evidence_validator"]["checks"] > 20000
    assert payload["repo_conformance"]["section12_tests_verified"] == 25
    assert all(
        v["passed"] for v in payload["repo_conformance"]["readme_topics"].values()
    )

    md = (out_dir / "rendered_m3_m4_tables_smoke.md").read_text(encoding="utf-8")
    for expected_heading in (
        "## M3 Closed-Loop Results",
        "### M3 Development Selection (`tune`)",
        "### M3 Held-Out Test Acceptance & Realized Lengths (Primary)",
        "### M3 Acceptance Probability by Logical Block Position (`0..K-1`)",
        "### M3 Standard Greedy (`T_sample=0, beta=1`) Output Exactness & BF16 Tie Report",
        "### M3 Isolated Single-GPU Throughput, Latency & Peak Memory",
        "0.00 [disabled in M0-M4]",
        "## M4 Closed-Loop Results",
        "### M4 Online Parameter Norms, Exact-Zero Sparsity, Active Buckets & Update Bandwidth",
        "### M4 Linked M1b Plain-DFlash `h[i]` Covariate-Shift Context",
    ):
        assert expected_heading in md, f"missing heading/marker {expected_heading!r}"


def test_detects_summary_and_timing_metric_drift(tmp_path: Path) -> None:
    copied = tmp_path / "tampered_smoke"
    shutil.copytree(SMOKE_FIXTURE, copied)

    s_path = copied / "m3" / "summary.json"
    s_data = json.loads(s_path.read_text(encoding="utf-8"))
    s_data["groups"][0]["metrics"]["mean_accepted_length"]["mean"] += 1e-4
    s_path.write_text(json.dumps(s_data), encoding="utf-8")

    t_path = copied / "m4" / "timing.json"
    t_data = json.loads(t_path.read_text(encoding="utf-8"))
    t_data["groups"][0]["paired_throughput_ratio"]["mean"] += 1e-4
    t_path.write_text(json.dumps(t_data), encoding="utf-8")

    auditor = bar.PostGateEAuditor(
        repo=REPO, artifacts=copied, scale="smoke", allow_partial=False, require_complete=False
    )
    auditor.audit_stage("m3")
    auditor.audit_stage("m4")
    assert any("m3/summary.json groups drift" in f for f in auditor.failures)
    assert any("m4/timing.json groups drift" in f for f in auditor.failures)


def test_online_bigram_cuda_event_zero_detection_and_zero_block_exemption(tmp_path: Path) -> None:
    art = tmp_path / "art"
    rank0 = art / "m3" / "timing" / "rank0"
    rank0.mkdir(parents=True)

    # Case 1: blocks > 0 and online_update == 0.0 -> MUST fail!
    bad_row = {
        "arm": "online_bigram",
        "prompt_id": "p0",
        "condition": "long_thinking",
        "sampling_name": "thinking_t0.6_top_k20_top_p0.95",
        "phase_profile": True,
        "blocks": [{"accepted_length": 3, "draft_tokens": [1, 2, 3]}],
        "phase_ms": {"online_update": 0.0},
    }
    (rank0 / "bad.json").write_text(json.dumps(bad_row), encoding="utf-8")
    aud1 = bar.PostGateEAuditor(repo=REPO, artifacts=art, scale="reference", allow_partial=True)
    aud1.audit_online_bigram_cuda_events()
    assert any("online_bigram" in f and "online_update<=0" in f for f in aud1.failures)

    # Case 2: blocks == [] (hypothetical prefill-EOS) -> MUST pass (matches validate.py:296 `if row['blocks']`)!
    (rank0 / "bad.json").unlink()
    empty_blocks_row = {
        "arm": "online_bigram",
        "prompt_id": "p_empty",
        "condition": "long_thinking",
        "sampling_name": "thinking_t0.6_top_k20_top_p0.95",
        "phase_profile": True,
        "blocks": [],
        "phase_ms": {},
    }
    (rank0 / "empty.json").write_text(json.dumps(empty_blocks_row), encoding="utf-8")
    aud2 = bar.PostGateEAuditor(repo=REPO, artifacts=art, scale="reference", allow_partial=True)
    res2 = aud2.audit_online_bigram_cuda_events()
    assert aud2.failures == []
    assert res2["profile_timing_audit"]["m3"]["by_arm"]["online_bigram"]["zero_or_missing_with_blocks"] == 0


def test_sparse_16k_long_bin_marked_unavailable() -> None:
    def make_long_row(pid: str, reach_16k: bool) -> dict:
        blocks = [
            {
                "accepted_length": 4,
                "absolute_output_offset": 100,
                "input_context_length": 2048,
                "draft_tokens": [1] * 15,
            }
        ]
        if reach_16k:
            blocks.append(
                {
                    "accepted_length": 2,
                    "absolute_output_offset": 17000,
                    "input_context_length": 19048,
                    "draft_tokens": [2] * 15,
                }
            )
        return {
            "condition": "long_thinking",
            "temperature": 0.6,
            "sampling_warp_config": {"top_k": 20, "top_p": 0.95, "min_p": 0.0},
            "arm": "plain_dflash",
            "prompt_id": pid,
            "seed": 2026,
            "generation": {"repetition_dominated": False},
            "elapsed_seconds": 10.0,
            "tokens_per_second": 100.0,
            "mean_accepted_length": 3.0,
            "median_accepted_length": 3.0,
            "target_forwards_per_generated_token": 0.25,
            "generated_tokens": 17002 if reach_16k else 5000,
            "blocks": blocks,
            "phase_ms": {},
        }

    # 10 prompts total, but only 2 reach 16K+ (< 8 threshold)
    rows = [make_long_row(f"p{i}", reach_16k=(i < 2)) for i in range(10)]
    aud = bar.PostGateEAuditor(repo=REPO, artifacts=SMOKE_FIXTURE, scale="reference", allow_partial=True)
    groups = aud.recompute_group_rows(rows, isolated_timing=False, exclude_repetition=True)
    mock_payload = {
        "scale": "reference",
        "artifacts": "/mock",
        "stages": {
            "m3": {
                "stage": "m3",
                "frozen_settings_hash": "abc",
                "selected_settings": {"plain_dflash": {}},
                "dev_selection_table": [],
                "evaluated_rows": 10,
                "expected_evaluated_rows": 10,
                "test_acceptance_groups": groups,
                "long_all_rows_sensitivity": [],
                "sparse_long_bins": [],
                "greedy_comparisons": [],
                "timing": {"status": "not_complete"},
            },
            "m4": {"status": "not_started"},
        },
    }
    md = bar.render_markdown_tables(mock_payload)
    assert "**SPARSE (`n=2 < 8`; conclusion unavailable)**" in md
    assert md.index("`1024-4096`") < md.index("`16384-32768`")


def test_detects_online_dbS_nonzero_dW_and_doc_staleness(tmp_path: Path) -> None:
    copied = tmp_path / "tampered_dbs"
    shutil.copytree(SMOKE_FIXTURE, copied)

    # Mutate an online_dbS evaluation row so dW_norm > 0 and update its manifest sha256
    mf_path = copied / "m4" / "evaluate" / "manifest-rank0.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    mutated = False
    for entry in mf["jobs"].values():
        rp_orig = Path(entry["result_path"])
        rel = rp_orig.relative_to(SMOKE_FIXTURE)
        rp_new = copied / rel
        entry["result_path"] = str(rp_new)
        row = json.loads(rp_new.read_text(encoding="utf-8"))
        if row.get("baseline_reuse", {}).get("status") == "reused":
            src_rel = Path(row["baseline_reuse"]["source_result_path"]).relative_to(SMOKE_FIXTURE)
            row["baseline_reuse"]["source_result_path"] = str(copied / src_rel)
        if row["arm"] == "online_dbS" and not mutated:
            row["online"]["dW_norm"] = 0.05
            mutated = True
        rp_new.write_text(json.dumps(row), encoding="utf-8")
        entry["result_sha256"] = bar.sha256_file(rp_new)
    mf_path.write_text(json.dumps(mf), encoding="utf-8")
    assert mutated

    aud = bar.PostGateEAuditor(repo=REPO, artifacts=copied, scale="smoke", allow_partial=True)
    aud.audit_stage("m4")
    assert any("online_dbS must keep dW_norm==0" in f for f in aud.failures)

    # Verify finalized REPO has 0 staleness markers AND synthetic stale repo triggers pre-completion markers
    aud_clean = bar.PostGateEAuditor(
        repo=REPO, artifacts=SMOKE_FIXTURE, scale="reference", allow_partial=True, require_complete=True
    )
    aud_clean.m4_timing_world_size = 8
    aud_clean.audit_repo_conformance()
    assert aud_clean.failures == [], f"Finalized REPO had doc/conformance failures: {aud_clean.failures}"

    stale_repo = tmp_path / "stale_repo"
    stale_repo.mkdir()
    for rel in bar.REQUIRED_LAYOUT_PATHS:
        target = stale_repo / rel
        if "." in Path(rel).name:
            target.parent.mkdir(parents=True, exist_ok=True)
            src_real = REPO / rel
            if src_real.is_file():
                shutil.copy2(src_real, target)
            else:
                target.write_text("", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "src" / "edraft", stale_repo / "src" / "edraft", dirs_exist_ok=True)
    shutil.copytree(REPO / "configs", stale_repo / "configs", dirs_exist_ok=True)
    shutil.copytree(REPO / "artifacts" / "m4_evidence_validator", stale_repo / "artifacts" / "m4_evidence_validator", dirs_exist_ok=True)
    shutil.copytree(REPO / "tests", stale_repo / "tests", dirs_exist_ok=True)
    shutil.copy2(REPO / "reports" / "SUMMARY.md", stale_repo / "reports" / "SUMMARY.md")
    ms = json.loads((REPO / "artifacts" / "milestones.json").read_text(encoding="utf-8"))
    ms["M4"]["status"] = "not_run"
    (stale_repo / "artifacts").mkdir(parents=True, exist_ok=True)
    (stale_repo / "artifacts" / "milestones.json").write_text(json.dumps(ms), encoding="utf-8")
    (stale_repo / "README.md").write_text(
        (stale_repo / "README.md").read_text(encoding="utf-8") + "\nM4 is **not_run**.\n",
        encoding="utf-8",
    )
    aud_docs = bar.PostGateEAuditor(
        repo=stale_repo, artifacts=SMOKE_FIXTURE, scale="reference", allow_partial=True, require_complete=True
    )
    aud_docs.audit_repo_conformance()
    assert any("pre-completion markers" in f for f in aud_docs.failures)


def test_canonicalize_group_list_online_final_state_permutation_invariance() -> None:
    s1 = {"u": 71, "active_buckets": 65000, "state_bytes_estimate_total": 1000, "dS_norm": 0.0021, "db_norm": 0.0011, "last_full_loss": 8.5}
    s2 = {"u": 92, "active_buckets": 102000, "state_bytes_estimate_total": 2000, "dS_norm": 0.0030, "db_norm": 0.0031, "last_full_loss": 9.2}
    g_a = [{"arm": "offline_online_stride1", "excluded_prompt_ids": ["x"], "online_final_state": [s2, s1]}]
    g_b = [{"arm": "offline_online_stride1", "online_final_state": [dict(s1), dict(s2)]}]
    ok, diff = bar.close_nested(bar._canonicalize_group_list(g_a), bar._canonicalize_group_list(g_b))
    assert ok, diff
    g_bad = [{"arm": "offline_online_stride1", "online_final_state": [dict(s1, active_buckets=65001), dict(s2)]}]
    ok_bad, _ = bar.close_nested(bar._canonicalize_group_list(g_a), bar._canonicalize_group_list(g_bad))
    assert not ok_bad


def test_independent_repetition_detector_v2() -> None:
    assert not bar.independent_repetition_dominated(list(range(200)))
    assert not bar.independent_repetition_dominated([(i * 37 + i // 3) % 997 for i in range(400)])
    # Periodic 12-token loop spanning 300 tokens (>= max(256, 0.5*320))
    periodic = [99, 98, 97] + ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 26)
    assert bar.independent_repetition_dominated(periodic)


def _create_sharded_m4_fixture(dst: Path, *, world_size: int = 2) -> Path:
    """Convert a copy of SMOKE_FIXTURE's M4 timing into a multi-GPU core-pinned sharded + 1-GPU calibrated tree."""
    shutil.copytree(SMOKE_FIXTURE, dst)
    # Fix M4 evaluate result_path + baseline_reuse paths so dst is self-contained
    eval_mf_p = dst / "m4" / "evaluate" / "manifest-rank0.json"
    eval_mf = json.loads(eval_mf_p.read_text(encoding="utf-8"))
    for entry in eval_mf["jobs"].values():
        rel = Path(entry["result_path"]).relative_to(SMOKE_FIXTURE)
        rp_new = dst / rel
        entry["result_path"] = str(rp_new)
        row = json.loads(rp_new.read_text(encoding="utf-8"))
        if row.get("baseline_reuse", {}).get("status") == "reused":
            s_rel = Path(row["baseline_reuse"]["source_result_path"]).relative_to(SMOKE_FIXTURE)
            row["baseline_reuse"]["source_result_path"] = str(dst / s_rel)
        rp_new.write_text(json.dumps(row), encoding="utf-8")
        entry["result_sha256"] = bar.sha256_file(rp_new)
    eval_mf_p.write_text(json.dumps(eval_mf), encoding="utf-8")

    # Fix tune manifest paths
    tune_mf_p = dst / "m4" / "tune" / "manifest-rank0.json"
    tune_mf = json.loads(tune_mf_p.read_text(encoding="utf-8"))
    for entry in tune_mf["jobs"].values():
        rel = Path(entry["result_path"]).relative_to(SMOKE_FIXTURE)
        rp_new = dst / rel
        entry["result_path"] = str(rp_new)
        entry["result_sha256"] = bar.sha256_file(rp_new)
    tune_mf_p.write_text(json.dumps(tune_mf), encoding="utf-8")

    # Fix feature_context paths if present
    fc_p = dst / "m4" / "feature_context.json"
    if fc_p.is_file():
        fc = json.loads(fc_p.read_text(encoding="utf-8"))
        fc_p.write_text(json.dumps(fc), encoding="utf-8")

    # Shard M4 timing across `world_size` ranks (SMOKE_FIXTURE has 4 prompts -> supports world_size in {2, 4})
    orig_t = json.loads((SMOKE_FIXTURE / "m4" / "timing.json").read_text(encoding="utf-8"))
    orig_sess = json.loads(Path(orig_t["raw_evidence"]["session_path"]).read_text(encoding="utf-8"))
    prompts_sorted = sorted(orig_t["prompt_ids"])
    rank_prompts: dict[int, list[str]] = {r: [] for r in range(world_size)}
    for idx, pid in enumerate(prompts_sorted):
        rank_prompts[idx % world_size].append(pid)
    prompt_to_rank = {pid: r for r, pids in rank_prompts.items() for pid in pids}

    harness_sha = "a" * 64
    timing_dir = dst / "m4" / "timing"
    shutil.rmtree(timing_dir)
    (timing_dir / "sessions").mkdir(parents=True)
    for r in range(world_size):
        (timing_dir / f"rank{r}").mkdir(parents=True)

    rank_jobs: dict[int, dict[str, dict]] = {r: {} for r in range(world_size)}
    rank_bundles: dict[int, dict[str, dict]] = {r: {} for r in range(world_size)}

    # Also build 1-GPU calibration on 1 std + 1 long prompt (2 repeats)
    cal_selected = {"synthetic-test-standard_nonthinking-0", "synthetic-test-long_thinking-0"}
    cal_dir = dst / "m4" / "timing_calibration"
    (cal_dir / "rank0").mkdir(parents=True)
    (cal_dir / "sessions").mkdir(parents=True)
    cal_jobs: dict[str, dict] = {}
    cal_bundles: dict[str, dict] = {}

    for b_key, b_val in orig_sess["bundles"].items():
        if b_val.get("status") != "complete":
            continue
        first_orig = json.loads(Path(b_val["result_paths"][0]).read_text(encoding="utf-8"))
        rk = prompt_to_rank[first_orig["prompt_id"]]
        aff_cpus = list(range(16 * rk, 16 * rk + 16))
        new_paths = []
        cal_paths = []
        is_cal_candidate = (
            first_orig["prompt_id"] in cal_selected
            and float(first_orig["temperature"]) in (0.6, 0.7)
            and not first_orig.get("phase_profile")
            and str(first_orig["repeat"]).split(":epoch=")[0] in ("0", "1")
        )
        for p_str in b_val["result_paths"]:
            orig_rp = Path(p_str)
            row = json.loads(orig_rp.read_text(encoding="utf-8"))
            row["rank"] = rk
            row["physical_gpu"] = rk
            row["numa_node"] = rk // 2
            row["cpu_affinity"] = aff_cpus
            row["timing_harness_implementation_sha256"] = harness_sha
            row["sharding_mode"] = f"sharded_{world_size}gpu"
            key = bar.fingerprint(row["identity"])
            dst_rp = timing_dir / f"rank{rk}" / f"{key}.json"
            dst_rp.write_text(json.dumps(row), encoding="utf-8")
            rank_jobs[rk][key] = {
                "identity": row["identity"],
                "status": "complete",
                "result_path": str(dst_rp),
                "result_sha256": bar.sha256_file(dst_rp),
            }
            new_paths.append(str(dst_rp))

            if is_cal_candidate:
                cal_row = copy.deepcopy(row)
                cal_row["rank"] = 0
                cal_ident = dict(cal_row["identity"], timing_session="cal-session-1gpu")
                cal_row["identity"] = cal_ident
                cal_row["timing_session"] = "cal-session-1gpu"
                cal_row["physical_gpu"] = 0
                cal_row["numa_node"] = 0
                cal_row["cpu_affinity"] = list(range(0, 16))
                # 1-GPU isolated is 2% faster uniformly -> identical paired arm/plain ratio!
                cal_row["tokens_per_second"] = float(row["tokens_per_second"]) * 1.02
                cal_key = bar.fingerprint(cal_ident)
                cal_rp = cal_dir / "rank0" / f"{cal_key}.json"
                cal_rp.write_text(json.dumps(cal_row), encoding="utf-8")
                cal_jobs[cal_key] = {
                    "identity": cal_ident,
                    "status": "complete",
                    "result_path": str(cal_rp),
                    "result_sha256": bar.sha256_file(cal_rp),
                }
                cal_paths.append(str(cal_rp))

        rank_bundles[rk][b_key] = {"status": "complete", "epoch": 0, "result_paths": new_paths}
        if is_cal_candidate and cal_paths:
            cal_bundles[f"cal-{b_key}"] = {"status": "complete", "epoch": 0, "result_paths": cal_paths}

    manifest_paths = []
    manifest_shas = []
    session_paths = []
    session_shas = []
    for r in range(world_size):
        mp = timing_dir / f"manifest-rank{r}.json"
        mp.write_text(json.dumps({"jobs": rank_jobs[r]}), encoding="utf-8")
        manifest_paths.append(str(mp))
        manifest_shas.append(bar.sha256_file(mp))

        sp = timing_dir / "sessions" / f"logical-sharded-rank{r}.json"
        sp.write_text(
            json.dumps(
                {
                    "rank": r,
                    "world_size": world_size,
                    "physical_gpu": r,
                    "numa_node": r // 2,
                    "cpu_affinity": list(range(16 * r, 16 * r + 16)),
                    "bundles": rank_bundles[r],
                }
            ),
            encoding="utf-8",
        )
        session_paths.append(str(sp))
        session_shas.append(bar.sha256_file(sp))

    plan_path = timing_dir / "sharding_plan.json"
    plan_payload = {
        "world_size": world_size,
        "strategy": "token_balanced_snake",
        "ranks": {str(r): {"prompt_ids": rank_prompts[r]} for r in range(world_size)},
    }
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    orig_t["world_size"] = world_size
    orig_t["sharding_mode"] = f"sharded_{world_size}gpu"
    orig_t["timing_harness_implementation_sha256"] = harness_sha
    orig_t["raw_evidence"] = {
        "world_size": world_size,
        "manifest_paths": manifest_paths,
        "manifest_sha256s": manifest_shas,
        "session_paths": session_paths,
        "session_sha256s": session_shas,
        "sharding_plan_path": str(plan_path),
        "sharding_plan_sha256": bar.sha256_file(plan_path),
    }
    (dst / "m4" / "timing.json").write_text(json.dumps(orig_t), encoding="utf-8")

    cal_mf = cal_dir / "manifest-rank0.json"
    cal_mf.write_text(json.dumps({"jobs": cal_jobs}), encoding="utf-8")
    cal_sp = cal_dir / "sessions" / "cal-rank0.json"
    cal_sp.write_text(
        json.dumps(
            {
                "rank": 0,
                "world_size": 1,
                "physical_gpu": 0,
                "numa_node": 0,
                "cpu_affinity": list(range(0, 16)),
                "bundles": cal_bundles,
            }
        ),
        encoding="utf-8",
    )
    cal_report = {
        "status": "complete",
        "scale": "smoke",
        "stage": "m4",
        "raw_evidence": {
            "manifest_path": str(cal_mf),
            "manifest_sha256": bar.sha256_file(cal_mf),
            "session_path": str(cal_sp),
            "session_sha256": bar.sha256_file(cal_sp),
        },
        "groups": [],
    }
    (dst / "m4" / "timing_isolation_calibration.json").write_text(
        json.dumps(cal_report), encoding="utf-8"
    )
    return dst


def test_sharded_m4_timing_and_1gpu_calibration_happy_path(tmp_path: Path) -> None:
    fx = _create_sharded_m4_fixture(tmp_path / "sharded4", world_size=4)
    aud = bar.PostGateEAuditor(
        repo=REPO, artifacts=fx, scale="smoke", allow_partial=False, require_complete=True
    )
    m4_rep = aud.audit_stage("m4")
    assert aud.failures == [], aud.failures
    tim = m4_rep["timing"]
    assert tim["is_sharded"] is True
    assert tim["world_size"] == 4
    assert tim["timed_requests"] == 120
    assert tim["profile_requests"] == 40
    assert tim["bundles"] == 32
    iso = tim["isolation_calibration"]
    assert iso is not None and iso["status"] == "complete"
    assert iso["calibrated_timed_runs"] == 20  # 2 prompts * 2 repeats * 5 arms
    for cg in iso["groups"]:
        assert cg["runs_1gpu"] == 2 and cg["runs_8gpu"] == 3
        assert 0.90 < cg["tps_8gpu_over_1gpu_ratio"] < 1.05
        assert abs(cg["paired_ratio_diff_8gpu_minus_1gpu"]) < 0.05

    md = bar.render_markdown_tables({"scale": "smoke", "artifacts": str(fx), "stages": {"m3": {"status": "not_started"}, "m4": m4_rep}})
    assert "### M4 Core-Pinned 4-GPU Sharded Matched-Bundle Throughput, Latency & Peak Memory" in md
    assert "Methodology Disclosure (4-GPU NUMA/Core-Pinned Sharded Timing + 1-GPU Calibration):" in md
    assert "### M4 1-GPU Isolated vs 4-GPU Core-Pinned Timing Calibration (`timing_isolation_calibration.json`)" in md


def test_fault_injection_overlapping_cpu_affinity_across_ranks(tmp_path: Path) -> None:
    fx = _create_sharded_m4_fixture(tmp_path / "overlap_cpu", world_size=4)
    t_path = fx / "m4" / "timing.json"
    t_data = json.loads(t_path.read_text(encoding="utf-8"))
    # Corrupt rank 2 session & rows so cpu_affinity overlaps rank 1's [16..31]
    sp2 = Path(t_data["raw_evidence"]["session_paths"][2])
    s2 = json.loads(sp2.read_text(encoding="utf-8"))
    bad_cpus = list(range(24, 40))  # overlaps 24..31 AND violates [32..47]
    s2["cpu_affinity"] = bad_cpus
    mf2_p = Path(t_data["raw_evidence"]["manifest_paths"][2])
    mf2 = json.loads(mf2_p.read_text(encoding="utf-8"))
    for b in s2["bundles"].values():
        for rp_s in b["result_paths"]:
            rp = Path(rp_s)
            rw = json.loads(rp.read_text(encoding="utf-8"))
            rw["cpu_affinity"] = bad_cpus
            rp.write_text(json.dumps(rw), encoding="utf-8")
            mf2["jobs"][bar.fingerprint(rw["identity"])]["result_sha256"] = bar.sha256_file(rp)
    mf2_p.write_text(json.dumps(mf2), encoding="utf-8")
    sp2.write_text(json.dumps(s2), encoding="utf-8")
    t_data["raw_evidence"]["manifest_sha256s"][2] = bar.sha256_file(mf2_p)
    t_data["raw_evidence"]["session_sha256s"][2] = bar.sha256_file(sp2)
    t_path.write_text(json.dumps(t_data), encoding="utf-8")

    aud = bar.PostGateEAuditor(repo=REPO, artifacts=fx, scale="smoke", require_complete=True)
    aud.audit_stage("m4")
    assert any("overlapping cpu_affinity" in f or "violates 16-core NUMA map" in f for f in aud.failures)


def test_fault_injection_cross_rank_prompt_overlap_in_sharding_plan(tmp_path: Path) -> None:
    fx = _create_sharded_m4_fixture(tmp_path / "bad_plan", world_size=2)
    t_path = fx / "m4" / "timing.json"
    t_data = json.loads(t_path.read_text(encoding="utf-8"))
    plan_p = Path(t_data["raw_evidence"]["sharding_plan_path"])
    plan = json.loads(plan_p.read_text(encoding="utf-8"))
    # Duplicate rank 0's prompt into rank 1's sharding plan
    dup_pid = plan["ranks"]["0"]["prompt_ids"][0]
    plan["ranks"]["1"]["prompt_ids"].append(dup_pid)
    plan_p.write_text(json.dumps(plan), encoding="utf-8")
    t_data["raw_evidence"]["sharding_plan_sha256"] = bar.sha256_file(plan_p)
    t_path.write_text(json.dumps(t_data), encoding="utf-8")

    aud = bar.PostGateEAuditor(repo=REPO, artifacts=fx, scale="smoke", require_complete=True)
    aud.audit_stage("m4")
    assert any("duplicate/overlapping prompt assignments" in f for f in aud.failures)


def test_fault_injection_mismatched_timing_harness_sha256(tmp_path: Path) -> None:
    fx = _create_sharded_m4_fixture(tmp_path / "bad_harness_sha", world_size=2)
    t_path = fx / "m4" / "timing.json"
    t_data = json.loads(t_path.read_text(encoding="utf-8"))
    sp1 = Path(t_data["raw_evidence"]["session_paths"][1])
    s1 = json.loads(sp1.read_text(encoding="utf-8"))
    mf1_p = Path(t_data["raw_evidence"]["manifest_paths"][1])
    mf1 = json.loads(mf1_p.read_text(encoding="utf-8"))
    first_b = next(iter(s1["bundles"].values()))
    rp = Path(first_b["result_paths"][0])
    rw = json.loads(rp.read_text(encoding="utf-8"))
    rw["timing_harness_implementation_sha256"] = "b" * 64
    rp.write_text(json.dumps(rw), encoding="utf-8")
    mf1["jobs"][bar.fingerprint(rw["identity"])]["result_sha256"] = bar.sha256_file(rp)
    mf1_p.write_text(json.dumps(mf1), encoding="utf-8")
    t_data["raw_evidence"]["manifest_sha256s"][1] = bar.sha256_file(mf1_p)
    t_path.write_text(json.dumps(t_data), encoding="utf-8")

    aud = bar.PostGateEAuditor(repo=REPO, artifacts=fx, scale="smoke", require_complete=True)
    aud.audit_stage("m4")
    assert any("timing_harness_implementation_sha256" in f for f in aud.failures)


def test_fault_injection_incomplete_bundle_arms(tmp_path: Path) -> None:
    fx = _create_sharded_m4_fixture(tmp_path / "incomplete_bundle", world_size=2)
    t_path = fx / "m4" / "timing.json"
    t_data = json.loads(t_path.read_text(encoding="utf-8"))
    sp0 = Path(t_data["raw_evidence"]["session_paths"][0])
    s0 = json.loads(sp0.read_text(encoding="utf-8"))
    first_b = next(iter(s0["bundles"].values()))
    first_b["result_paths"] = first_b["result_paths"][:4]  # drop 5th arm from bundle!
    sp0.write_text(json.dumps(s0), encoding="utf-8")
    t_data["raw_evidence"]["session_sha256s"][0] = bar.sha256_file(sp0)
    t_path.write_text(json.dumps(t_data), encoding="utf-8")

    aud = bar.PostGateEAuditor(repo=REPO, artifacts=fx, scale="smoke", require_complete=True)
    aud.audit_stage("m4")
    assert any("expected 5 arms" in f for f in aud.failures)


def test_fault_injection_missing_and_drifted_1gpu_isolation_calibration(tmp_path: Path) -> None:
    fx = _create_sharded_m4_fixture(tmp_path / "cal_fault", world_size=2)
    cal_path = fx / "m4" / "timing_isolation_calibration.json"

    # Case 1: Tamper with a recorded comparison number inside timing_isolation_calibration.json
    cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
    cal_data["groups"] = [
        {
            "condition": "standard_nonthinking",
            "arm": "online_bigram",
            "isolated_1gpu_paired_ratio": 9.9999,  # bogus drifted ratio!
        }
    ]
    cal_path.write_text(json.dumps(cal_data), encoding="utf-8")
    aud1 = bar.PostGateEAuditor(repo=REPO, artifacts=fx, scale="smoke", require_complete=True)
    aud1.audit_stage("m4")
    assert any("timing_isolation_calibration.json" in f and "drifts from raw recomputation" in f for f in aud1.failures)

    # Case 2: Delete timing_isolation_calibration.json completely under require_complete=True
    cal_path.unlink()
    aud2 = bar.PostGateEAuditor(repo=REPO, artifacts=fx, scale="smoke", require_complete=True)
    aud2.audit_stage("m4")
    assert any("m4/timing_isolation_calibration.json is required" in f for f in aud2.failures)


def test_fault_injection_missing_8gpu_and_calibration_doc_disclosures(tmp_path: Path) -> None:
    # Build a minimal synthetic repo skeleton to test Post-Gate-E README.md & SUMMARY.md disclosure gate
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    for rel in bar.REQUIRED_LAYOUT_PATHS:
        target = fake_repo / rel
        if "." in Path(rel).name:
            target.parent.mkdir(parents=True, exist_ok=True)
            src_real = REPO / rel
            if src_real.is_file():
                shutil.copy2(src_real, target)
            else:
                target.write_text("", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
    # Copy tests/ so AST check of all 25 §12 tests passes
    shutil.copytree(REPO / "tests", fake_repo / "tests", dirs_exist_ok=True)

    # Clear pre-Gate-E staleness markers in milestones.json, README.md, and SUMMARY.md,
    # BUT omit the 8-GPU core-pinned + 1-GPU calibration disclosures first!
    (fake_repo / "artifacts" / "milestones.json").write_text(
        json.dumps({"M3": {"status": "passed"}, "M4": {"status": "passed"}}), encoding="utf-8"
    )
    clean_readme = (REPO / "README.md").read_text(encoding="utf-8")
    clean_summary = "# Final Summary\nAll M0-M4 reference gates complete.\n"
    for pat in (
        "Fresh M3 development selection is **running**",
        "Fresh M3 development selection is running",
        "M3 test evaluation and timing, and the M4 experiments, remain pending",
        "M3 test\nevaluation and timing, and the M4 experiments, remain pending",
    ):
        clean_readme = clean_readme.replace(pat, "M3 and M4 reference experiments are complete")
    (fake_repo / "README.md").write_text(clean_readme, encoding="utf-8")
    (fake_repo / "reports" / "SUMMARY.md").write_text(clean_summary, encoding="utf-8")

    aud_missing = bar.PostGateEAuditor(
        repo=fake_repo, artifacts=SMOKE_FIXTURE, scale="reference", require_complete=True
    )
    aud_missing.m4_timing_world_size = 8
    aud_missing.audit_repo_conformance()
    assert any("missing mandatory 8-GPU sharded M4 timing / 1-GPU calibration disclosures" in f for f in aud_missing.failures)

    # Now append explicit 8-GPU NUMA/core-pinned + 1-GPU timing_isolation_calibration disclosures to both!
    disclosure = (
        "\n\n### M4 Timing Methodology\n"
        "Reference `m4_timing` ran with 8-GPU (`world_size=8`) NUMA-local 16-core-pinned (`cpu_affinity`) "
        "matched all-arm bundles and was verified against a 60-run 1-GPU isolated calibration "
        "(`artifacts/m4/timing_isolation_calibration.json`).\n"
    )
    (fake_repo / "README.md").write_text(clean_readme + disclosure, encoding="utf-8")
    (fake_repo / "reports" / "SUMMARY.md").write_text(clean_summary + disclosure, encoding="utf-8")

    aud_ok = bar.PostGateEAuditor(
        repo=fake_repo, artifacts=SMOKE_FIXTURE, scale="reference", require_complete=True
    )
    aud_ok.m4_timing_world_size = 8
    aud_ok.audit_repo_conformance()
    assert aud_ok.failures == [], aud_ok.failures


def test_live_inflight_manifest_warn_only_under_allow_partial(tmp_path: Path) -> None:
    art = tmp_path / "live_art"
    ev = art / "m4" / "evaluate"
    ev.mkdir(parents=True)
    mf = ev / "manifest-rank0.json"
    row_file = ev / "rank0_job.json"
    mf.write_text('{"jobs": {"a": 1}}', encoding="utf-8")
    row_file.write_text('{"accepted": 3}', encoding="utf-8")

    # 1) Under allow_partial=True, require_complete=False: manifest advancing warns; non-manifest mutating FAILS!
    aud_live = bar.PostGateEAuditor(repo=REPO, artifacts=art, scale="reference", allow_partial=True, require_complete=False)
    aud_live.read_json_tracked(mf)
    aud_live.read_json_tracked(row_file)
    mf.write_text('{"jobs": {"a": 1, "b": 2}}', encoding="utf-8")
    aud_live.read_json_tracked(mf)
    assert aud_live.failures == []
    assert any("inflight manifest advanced" in w for w in aud_live.warnings)

    row_file.write_text('{"accepted": 999}', encoding="utf-8")
    aud_live.read_json_tracked(row_file)
    assert any("file mutated during audit" in f for f in aud_live.failures)

    # 2) Under require_complete=True (or allow_partial=False): even manifest mutating FAILS hard!
    aud_strict = bar.PostGateEAuditor(repo=REPO, artifacts=art, scale="reference", allow_partial=False, require_complete=True)
    aud_strict.read_json_tracked(mf)
    mf.write_text('{"jobs": {"a": 1, "b": 2, "c": 3}}', encoding="utf-8")
    aud_strict.read_json_tracked(mf)
    assert any("file mutated during audit" in f for f in aud_strict.failures)


def test_post_m4_triton_ops_and_09a5469_source_repo_compatibility(tmp_path: Path) -> None:
    """Verify post-M4 layout (`triton_ops.py`, `fastpath.py`, modified `online.py`) passes with `--source_repo` and fails if `online.py` imports `triton_ops`."""
    def _mk_repo(dst: Path) -> Path:
        dst.mkdir()
        for rel in bar.REQUIRED_LAYOUT_PATHS:
            target = dst / rel
            if "." in Path(rel).name:
                target.parent.mkdir(parents=True, exist_ok=True)
                src_real = REPO / rel
                if src_real.is_file():
                    shutil.copy2(src_real, target)
                else:
                    target.write_text("", encoding="utf-8")
            else:
                target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO / "src" / "edraft", dst / "src" / "edraft", dirs_exist_ok=True)
        shutil.copytree(REPO / "configs", dst / "configs", dirs_exist_ok=True)
        shutil.copytree(REPO / "artifacts" / "m4_evidence_validator", dst / "artifacts" / "m4_evidence_validator", dirs_exist_ok=True)
        shutil.copytree(REPO / "tests", dst / "tests", dirs_exist_ok=True)
        shutil.copy2(REPO / "reports" / "SUMMARY.md", dst / "reports" / "SUMMARY.md")
        shutil.copy2(REPO / "artifacts" / "milestones.json", dst / "artifacts" / "milestones.json")
        return dst

    fake_repo = _mk_repo(tmp_path / "post_m4_repo")
    src_snap = Path("/tmp/edraft_09a5469_source") if Path("/tmp/edraft_09a5469_source").is_dir() else _mk_repo(tmp_path / "frozen_09a_source")
    (fake_repo / "src/edraft/fastpath.py").write_text("# fastpath\n", encoding="utf-8")
    (fake_repo / "src/edraft/triton_ops.py").write_text("import triton\n", encoding="utf-8")
    (fake_repo / "src/edraft/online.py").write_text(
        (fake_repo / "src/edraft/online.py").read_text(encoding="utf-8") + "\n# round4 analytic\n",
        encoding="utf-8",
    )

    aud_ok = bar.PostGateEAuditor(
        repo=fake_repo,
        source_repo=src_snap,
        artifacts=SMOKE_FIXTURE,
        scale="smoke",
        allow_partial=True,
    )
    aud_ok.audit_repo_conformance()
    assert aud_ok.failures == [], f"Unexpected failures on valid post-M4 repo: {aud_ok.failures}"

    # Also test partial inflight sharded M4 timing + 1-GPU calibration BEFORE m4/timing.json exists!
    fx_partial = _create_sharded_m4_fixture(tmp_path / "partial_m4", world_size=2)
    (fx_partial / "m4" / "timing.json").unlink()  # simulate live Step B before timing.json is written!
    aud_partial = bar.PostGateEAuditor(
        repo=fake_repo,
        source_repo=src_snap,
        artifacts=fx_partial,
        scale="smoke",
        allow_partial=True,
    )
    m4_part = aud_partial.audit_stage("m4")
    assert aud_partial.failures == [], f"Unexpected failures on partial sharded M4: {aud_partial.failures}"
    assert m4_part["timing"]["status"] == "in_progress_partial"
    assert m4_part["timing"]["online_rows_verified"] > 0
    assert m4_part["timing"]["isolation_calibration"] is not None

    # Now inject illegal `import triton_ops` into M0-M4 core `online.py` -> MUST fail!
    (fake_repo / "src/edraft/online.py").write_text(
        (fake_repo / "src/edraft/online.py").read_text(encoding="utf-8") + "\nimport triton_ops\n",
        encoding="utf-8",
    )
    aud_bad = bar.PostGateEAuditor(
        repo=fake_repo,
        source_repo=src_snap,
        artifacts=SMOKE_FIXTURE,
        scale="smoke",
        allow_partial=True,
    )
    aud_bad.audit_repo_conformance()
    assert any("imports_m5_triton_ops" in f for f in aud_bad.failures)


