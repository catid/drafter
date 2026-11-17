"""Comprehensive unit & fault-injection tests for `/tmp/edraft_reporting/audit_post_m4_suite.py`."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

_THIS_DIR = Path(__file__).resolve().parent
REPO = Path("/home/catid/drafter")
for p in (str(_THIS_DIR), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import audit_post_m4_suite as apm  # noqa: E402
from edraft import post_m4_suite  # noqa: E402


@pytest.fixture(scope="module")
def real_smoke_post_m4_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Execute the real 2-worker `edraft.post_m4_suite --scale smoke --suite all --gpus 0,1` once on CPU."""
    out = tmp_path_factory.mktemp("post_m4_smoke_all")
    post_m4_suite.main(
        [
            "--scale",
            "smoke",
            "--suite",
            "all",
            "--gpus",
            "0,1",
            "--output",
            str(out),
            "--bench_warmups",
            "1",
            "--bench_repeats",
            "2",
            "--timing_warmups",
            "1",
            "--timing_repeats",
            "2",
            "--smoke_tokens",
            "24",
            "--online_arms",
            "plain_dflash",
            "online_bigram",
            "online_bigram_count_gated",
            "online_dbS_zero_init",
            "zero_offline_online_only",
            "--m5_eval_arms",
            "plain_dflash",
            "dense_candidate",
            "lookahead_static_candidate",
            "lookahead_rerelax_candidate",
            "online_two_sided_pl",
            "signed_feature_hash_online",
            "exact_observed_pairs_online",
            "offline_online_stride5",
            "online_buffered_stride5",
        ]
    )
    return out


def _copy_suite_tree(src_out: Path, dst_out: Path) -> Path:
    """Copy `post_m4_suite` output directory rewriting internal absolute JSON paths and `frozen_settings_hash`es."""
    shutil.copytree(src_out, dst_out, symlinks=True)
    old_prefix = str(src_out.resolve())
    new_prefix = str(dst_out.resolve())

    # Collect old settings.json hashes before path rewriting
    old_fh_by_rel: dict[Path, str] = {}
    for sp_old in src_out.rglob("settings.json"):
        if not sp_old.is_symlink():
            old_fh_by_rel[sp_old.relative_to(src_out)] = apm.fingerprint(
                json.loads(sp_old.read_text(encoding="utf-8"))
            )

    for jf in dst_out.rglob("*.json"):
        if jf.is_symlink():
            continue
        txt = jf.read_text(encoding="utf-8")
        if old_prefix in txt:
            jf.write_text(txt.replace(old_prefix, new_prefix), encoding="utf-8")

    # Map any changed frozen_settings_hash to the new fingerprint
    fh_replacements: list[tuple[str, str]] = []
    for rel_sp, old_fh in old_fh_by_rel.items():
        new_fh = apm.fingerprint(json.loads((dst_out / rel_sp).read_text(encoding="utf-8")))
        if new_fh != old_fh:
            fh_replacements.append((old_fh, new_fh))
    if fh_replacements:
        for jf in dst_out.rglob("*.json"):
            if jf.is_symlink():
                continue
            txt = jf.read_text(encoding="utf-8")
            orig_txt = txt
            for old_fh, new_fh in fh_replacements:
                txt = txt.replace(old_fh, new_fh)
            if txt != orig_txt:
                jf.write_text(txt, encoding="utf-8")

    # Refresh manifest & session result_sha256 entries after path/hash string replacement in row JSONs!
    for mf in dst_out.rglob("manifest*.json"):
        if mf.is_symlink():
            continue
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        changed = False
        for entry in mdata.get("jobs", {}).values():
            rp = Path(entry.get("result_path", ""))
            if rp.is_file() and "result_sha256" in entry:
                entry["result_sha256"] = apm.sha256_file(rp)
                changed = True
        if changed:
            mf.write_text(json.dumps(mdata), encoding="utf-8")
    for sf in dst_out.rglob("sessions/*.json"):
        if sf.is_symlink():
            continue
        sdata = json.loads(sf.read_text(encoding="utf-8"))
        changed = False
        for bval in sdata.get("bundles", {}).values():
            rshas = bval.get("result_sha256")
            if isinstance(rshas, dict):
                for rps in list(rshas.keys()):
                    rp = Path(rps)
                    if rp.is_file():
                        rshas[rps] = apm.sha256_file(rp)
                        changed = True
        if changed:
            sf.write_text(json.dumps(sdata), encoding="utf-8")
    return dst_out


def test_clean_smoke_all_4_suites_and_markdown_rendering(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "rendered"
    rc = apm.main(
        [
            "--scale",
            "smoke",
            "--repo",
            str(REPO),
            "--artifacts",
            str(real_smoke_post_m4_fixture),
            "--require_complete",
            "--output_dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    rep = json.loads((out_dir / "post_m4_audit_report_smoke.json").read_text(encoding="utf-8"))
    assert rep["status"] == "passed"
    assert rep["failures"] == []
    assert rep["warnings"] == []
    assert rep["checks"] > 200

    # Verify all 4 suites + master summary passed
    for s_name in apm.SUITE_NAMES:
        assert rep["suites"][s_name]["status"] == "complete"
    assert rep["master_summary"]["status"] == "complete"

    # Verify ExactObservedPairs collision_count == 0 and matched C0 beta_relax == 1.0
    assert rep["suites"]["m5_eval"]["offline_fits"]["exact_observed_pairs"]["collision_count"] == 0
    assert rep["suites"]["m5_eval"]["workspace_audit"]["matched_c0_beta_relax"] == 1.0

    # Verify sparse <8-prompt long offset bins are flagged `conclusion_available == False`
    sparse_bins = rep["suites"]["online_head_to_head"]["workspace_audit"]["sparse_long_offset_bins"]
    assert len(sparse_bins) > 0
    assert all(b["conclusion_available"] is False for b in sparse_bins)

    # Verify publication-ready Markdown contains all 4 sections and sparsity gate labels
    md = (out_dir / "rendered_post_m4_tables_smoke.md").read_text(encoding="utf-8")
    assert "## 1. M2 Offline Refit & Post-Hoc Calibration" in md
    assert "## 2. Online Zero-Init & Count-Gated Bigram Head-to-Head" in md
    assert "## 3. Optional M5 Extensions (`m5_eval`)" in md
    assert "## 4. Fastpath & Multi-GPU Scaling Benchmark (`fastpath_bench`)" in md
    assert "conclusion unavailable" in md
    assert "1-GPU Isolated vs `2-GPU` Core-Pinned Sharded Calibration" in md


def test_allow_partial_when_only_subset_of_suites_present(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    part = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "partial_ws")
    # Remove 3 suite JSONs + master summary to simulate live Step 1 (`m2_refit_calibration` just finished)
    for fn in ("online_head_to_head.json", "m5_eval.json", "fastpath_bench.json", "post_m4_suite_summary.json"):
        (part / "post_m4_suite" / fn).unlink()

    aud_part = apm.PostM4SuiteAuditor(
        repo=REPO, artifacts=part, scale="smoke", allow_partial=True, require_complete=False
    )
    rep_part = aud_part.run_all()
    assert rep_part["status"] == "passed", rep_part["failures"]
    assert len(rep_part["warnings"]) == 4

    # Under --require_complete, missing suites MUST fail!
    aud_strict = apm.PostM4SuiteAuditor(
        repo=REPO, artifacts=part, scale="smoke", allow_partial=False, require_complete=True
    )
    rep_strict = aud_strict.run_all()
    assert rep_strict["status"] == "failed"
    assert len(rep_strict["failures"]) >= 4


def test_fault_exact_observed_pairs_nonzero_collision_caught(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    bad = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "bad_exact")
    m5_p = bad / "post_m4_suite" / "m5_eval.json"
    m5 = json.loads(m5_p.read_text(encoding="utf-8"))
    # Mutate underlying exact-fit report collision_count to 2 while keeping top-level 0!
    ex_rep_p = Path(m5["offline_fits"]["exact_observed_pairs"]["report_path"])
    ex_fit = json.loads(ex_rep_p.read_text(encoding="utf-8"))
    ex_fit["stages"]["dense_sparse"]["sparse_collision_statistics"]["collision_count"] = 2
    ex_rep_p.write_text(json.dumps(ex_fit), encoding="utf-8")

    aud = apm.PostM4SuiteAuditor(repo=REPO, artifacts=bad, scale="smoke", require_complete=True)
    res = aud.run_all()
    assert res["status"] == "failed"
    assert any("m5_exact_pairs" in f and "collision_count" in f for f in res["failures"])

    # Also test top-level collision_count > 0
    m5["offline_fits"]["exact_observed_pairs"]["collision_count"] = 5
    m5_p.write_text(json.dumps(m5), encoding="utf-8")
    aud2 = apm.PostM4SuiteAuditor(repo=REPO, artifacts=bad, scale="smoke", require_complete=True)
    res2 = aud2.run_all()
    assert any("ExactObservedPairs top-level collision_count=5 != 0" in f for f in res2["failures"])


def test_fault_mismatched_c0_beta_relax_and_lookahead_telemetry_caught(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    bad = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "bad_c0")
    m5 = json.loads((bad / "post_m4_suite" / "m5_eval.json").read_text(encoding="utf-8"))
    settings_p = Path(m5["closed_loop"]["artifacts"]["settings_json"])
    settings = json.loads(settings_p.read_text(encoding="utf-8"))

    # Fault A: mismatch beta_relax between dense_candidate (1.0) and lookahead_rerelax_candidate (0.5)
    settings_bad = copy.deepcopy(settings)
    settings_bad["selected"]["lookahead_rerelax_candidate"]["beta_relax"] = 0.5
    settings_p.write_text(json.dumps(settings_bad), encoding="utf-8")

    aud = apm.PostM4SuiteAuditor(repo=REPO, artifacts=bad, scale="smoke", require_complete=True)
    res = aud.run_all()
    assert res["status"] == "failed"
    assert any("matched C0[j] violation" in f for f in res["failures"])

    # Restore settings.json, then Fault B: set lookahead_iterations = 0.0 on an active lookahead profile row!
    settings_p.write_text(json.dumps(settings), encoding="utf-8")
    timing_p = Path(m5["closed_loop"]["artifacts"]["timing_json"])
    timing = json.loads(timing_p.read_text(encoding="utf-8"))
    sess_paths = timing["raw_evidence"].get("session_paths") or [timing["raw_evidence"]["session_path"]]
    mutated_prof = False
    for sp_str in sess_paths:
        sp = Path(sp_str)
        sdata = json.loads(sp.read_text(encoding="utf-8"))
        for bval in sdata["bundles"].values():
            if bval.get("status") != "complete":
                continue
            for rps in bval["result_paths"]:
                rp = Path(rps)
                row = json.loads(rp.read_text(encoding="utf-8"))
                if (
                    row.get("phase_profile") is True
                    and row["arm"] == "lookahead_rerelax_candidate"
                    and len(row.get("blocks", [])) > 0
                ):
                    row["phase_ms"]["lookahead_iterations"] = 0.0
                    rp.write_text(json.dumps(row), encoding="utf-8")
                    new_sha = apm.sha256_file(rp)
                    if isinstance(bval.get("result_sha256"), dict):
                        bval["result_sha256"][rps] = new_sha
                    for tmf in sorted((timing_p.parent / "timing").glob("manifest-rank*.json")):
                        tmdata = json.loads(tmf.read_text(encoding="utf-8"))
                        for t_e in tmdata.get("jobs", {}).values():
                            if t_e.get("result_path") == rps:
                                t_e["result_sha256"] = new_sha
                        tmf.write_text(json.dumps(tmdata), encoding="utf-8")
                    mutated_prof = True
                    break
            if mutated_prof:
                break
        if mutated_prof:
            sp.write_text(json.dumps(sdata), encoding="utf-8")
            break
    assert mutated_prof

    aud2 = apm.PostM4SuiteAuditor(repo=REPO, artifacts=bad, scale="smoke", require_complete=True)
    res2 = aud2.run_all()
    assert res2["status"] == "failed"
    assert any("lookahead_iterations=0.0 <= 0" in f for f in res2["failures"])


def test_fault_long_repetition_dominated_misflag_and_inclusion_caught(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    bad = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "bad_rep")
    h2h = json.loads((bad / "post_m4_suite" / "online_head_to_head.json").read_text(encoding="utf-8"))
    summary_p = Path(h2h["artifacts"]["summary_json"])
    eval_dir = summary_p.parent / "evaluate"

    # Inject a 300-token exact 10-token cycle into a `long_*` evaluate row while keeping repetition_dominated=False
    mutated = False
    for mf in sorted(eval_dir.glob("manifest-rank*.json")):
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        for entry in mdata["jobs"].values():
            rp = Path(entry["result_path"])
            row = json.loads(rp.read_text(encoding="utf-8"))
            if str(row["condition"]).startswith("long_"):
                row["output_tokens"] = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 30)
                row["generation"]["repetition_dominated"] = False
                rp.write_text(json.dumps(row), encoding="utf-8")
                entry["result_sha256"] = apm.sha256_file(rp)
                mutated = True
                break
        if mutated:
            mf.write_text(json.dumps(mdata), encoding="utf-8")
            break
    assert mutated

    aud = apm.PostM4SuiteAuditor(repo=REPO, artifacts=bad, scale="smoke", require_complete=True)
    res = aud.run_all()
    assert res["status"] == "failed"
    assert any("repetition_dominated mismatch" in f for f in res["failures"])


def test_fault_fastpath_bench_calibration_and_offset_bin_ci_drift_caught(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    bad = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "bad_cal_ci")

    # Fault 1: Perturb 1-GPU vs sharded calibration ratio in `fastpath_bench.json`
    fb_p = bad / "post_m4_suite" / "fastpath_bench.json"
    fb = json.loads(fb_p.read_text(encoding="utf-8"))
    fb["cluster_sharded_throughput"]["by_arm"][0]["sharded_over_1gpu_per_gpu_ratio"] += 0.25
    fb_p.write_text(json.dumps(fb), encoding="utf-8")

    # Fault 2: Perturb offset-bin CI in `online_head_to_head.json`
    h2h_p = bad / "post_m4_suite" / "online_head_to_head.json"
    h2h = json.loads(h2h_p.read_text(encoding="utf-8"))
    h2h["evaluation_offset_bin_slices"][0]["bins"][0]["accepted_length"]["ci95"][0] += 0.5
    h2h_p.write_text(json.dumps(h2h), encoding="utf-8")

    # Fault 3: Mismatch config_fingerprint in `post_m4_suite_summary.json`
    ms_p = bad / "post_m4_suite" / "post_m4_suite_summary.json"
    ms = json.loads(ms_p.read_text(encoding="utf-8"))
    ms["suites"]["m5_eval"]["config_fingerprint"] = "deadbeef" * 8
    ms_p.write_text(json.dumps(ms), encoding="utf-8")

    aud = apm.PostM4SuiteAuditor(repo=REPO, artifacts=bad, scale="smoke", require_complete=True)
    res = aud.run_all()
    assert res["status"] == "failed"
    assert any("cluster_sharded_throughput.by_arm drift" in f for f in res["failures"])
    assert any("evaluation_offset_bin_slices drift" in f for f in res["failures"])
    assert any("config_fingerprint mismatch" in f for f in res["failures"])


def test_fault_upstream_m0_m4_artifact_mutation_caught(tmp_path: Path) -> None:
    fake_art = tmp_path / "fake_ref_art"
    (fake_art / "m4").mkdir(parents=True)
    m4_sum = fake_art / "m4" / "summary.json"
    m4_sum.write_text('{"status": "complete"}', encoding="utf-8")
    good_sha = apm.sha256_file(m4_sum)
    (fake_art / "milestones.json").write_text(
        json.dumps(
            {
                "M3": {"status": "passed", "artifact_sha256": {}},
                "M4": {"status": "passed", "artifact_sha256": {"summary": good_sha}},
            }
        ),
        encoding="utf-8",
    )
    # Tamper with m4/summary.json
    m4_sum.write_text('{"status": "tampered"}', encoding="utf-8")

    aud = apm.PostM4SuiteAuditor(
        repo=REPO, artifacts=fake_art, scale="reference", allow_partial=True
    )
    aud.audit_upstream_m0_m4_immutability()
    assert any("upstream m4/summary.json SHA256" in f for f in aud.failures)


def test_build_and_verify_implementation_audit_m0_m4_and_m5_upgrade(tmp_path: Path) -> None:
    import build_implementation_audit as bia

    # 1. Regenerate against real /home/catid/drafter + /home/catid/drafter/artifacts -> 0 errors!
    reg = bia.build_implementation_audit(REPO, REPO / "artifacts")
    errs = bia.verify_implementation_audit(reg, REPO, REPO / "artifacts")
    assert errs == [], errs
    assert reg["status"] == "reference_m0_m4_complete_verified"
    assert reg["reference_M1_results"]["M1b"]["diagnostic_groups"] == 128
    assert all(
        r["status"] == "implemented_verified" and r["remaining_work"] == []
        for r in reg["requirements"]
        if r["id"] != "optional_m5"
    )

    # 2. Simulate partial Phase A (`executed_suites=['m2_refit_calibration']`) -> must NOT prematurely upgrade to M0-M5!
    sim_art = tmp_path / "sim_art"
    (sim_art / "post_m4_suite").mkdir(parents=True)
    shutil.copy2(REPO / "artifacts" / "milestones.json", sim_art / "milestones.json")
    (sim_art / "post_m4_suite" / "post_m4_suite_summary.json").write_text(
        json.dumps({"status": "complete", "executed_suites": ["m2_refit_calibration"]}),
        encoding="utf-8",
    )
    reg_partial = bia.build_implementation_audit(
        REPO, sim_art, template_path=REPO / "artifacts" / "implementation_audit.json"
    )
    assert reg_partial["status"] == "reference_m0_m4_complete_verified"

    # 3. Simulate completed Phase A (all 4 suites in `executed_suites`) -> upgrades to M0-M5!
    (sim_art / "post_m4_suite" / "post_m4_suite_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "executed_suites": [
                    "m2_refit_calibration",
                    "online_head_to_head",
                    "m5_eval",
                    "fastpath_bench",
                ],
            }
        ),
        encoding="utf-8",
    )
    reg_m5 = bia.build_implementation_audit(
        REPO, sim_art, template_path=REPO / "artifacts" / "implementation_audit.json"
    )
    assert reg_m5["status"] == "reference_m0_m5_complete_verified"
    opt_m5 = next(r for r in reg_m5["requirements"] if r["id"] == "optional_m5")
    assert opt_m5["status"] == "implemented_verified"
    assert bia.verify_implementation_audit(reg_m5, REPO, sim_art) == []

    # 4. Fault injection: corrupt an AST line number and a source SHA256 -> caught!
    bad = copy.deepcopy(reg)
    bad["requirements"][0]["implementation_evidence"][2]["line"] += 999
    bad["source_tree_sha256"] = "0" * 64
    errs_bad = bia.verify_implementation_audit(bad, REPO, REPO / "artifacts")
    assert any("stale AST line" in e for e in errs_bad)
    assert any("source_tree_sha256" in e for e in errs_bad)

    # 5. Regression test for `IMPL-AUDIT-POST-COMMIT-HEAD-PARADOX-1`: committing tracked
    # `artifacts/implementation_audit.json` advances git HEAD by 1 doc-only commit without invalidating `check_git_head=True`,
    # whereas a subsequent `src/` commit strictly invalidates `project_git_commit`!
    import subprocess

    mini_repo = tmp_path / "mini_git_repo"
    subprocess.run(["git", "clone", "--shared", "-q", str(REPO), str(mini_repo)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=mini_repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=mini_repo, check=True)
    code_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=mini_repo, text=True).strip()

    ia_built = bia.build_implementation_audit(mini_repo, mini_repo / "artifacts")
    assert ia_built["project_git_commit"] == code_head
    (mini_repo / "artifacts/implementation_audit.json").write_text(json.dumps(ia_built, indent=2) + "\n", encoding="utf-8")
    assert bia.verify_implementation_audit(ia_built, mini_repo, mini_repo / "artifacts", check_git_head=True) == []

    # Commit `artifacts/implementation_audit.json` -> HEAD advances to `doc_head != code_head`, MUST still pass `check_git_head=True`!
    subprocess.run(["git", "add", "artifacts/implementation_audit.json"], cwd=mini_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "docs: record implementation_audit"], cwd=mini_repo, check=True)
    doc_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=mini_repo, text=True).strip()
    assert doc_head != code_head
    assert bia.verify_implementation_audit(ia_built, mini_repo, mini_repo / "artifacts", check_git_head=True) == []

    # Now touch a `src/` file and commit -> MUST fail `project_git_commit != git HEAD`!
    init_py = mini_repo / "src/edraft/__init__.py"
    init_py.write_text(init_py.read_text(encoding="utf-8") + "\n# code change\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/edraft/__init__.py"], cwd=mini_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "feat: code change"], cwd=mini_repo, check=True)
    errs_after_code = bia.verify_implementation_audit(ia_built, mini_repo, mini_repo / "artifacts", check_git_head=True)
    assert any("project_git_commit" in e for e in errs_after_code)


def test_audit_rep_bug_1_compacted_top_level_repetition_dominated_rows(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    """Verify compacted Suite 1 & Suite 4 rows (`row['repetition_dominated'] = True`, no `generation` dict) exclude cleanly."""
    from types import SimpleNamespace

    fx = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "compacted_rep")
    fb_p = fx / "post_m4_suite" / "fastpath_bench.json"
    fb = json.loads(fb_p.read_text(encoding="utf-8"))
    cl = fb["closed_loop_matched_comparison"]
    speed_rows = cl["speed_rows"]
    profile_rows = cl["profile_rows"]

    # Mark one `long_thinking` prompt's compacted speed_rows (`no 'generation' key`) as `repetition_dominated=True`
    target_pid = next(r["prompt_id"] for r in speed_rows if r["condition"].startswith("long_"))
    flipped = 0
    for r in speed_rows:
        assert "generation" not in r
        if r["prompt_id"] == target_pid and r["arm"] == "online_bigram" and r["repeat"] == 0:
            r["repetition_dominated"] = True
            r["tokens_per_second"] = 999999.0  # extreme value: would blow up primary mean if not excluded!
            flipped += 1
    assert flipped > 0

    # Re-summarize with real `post_m4_suite._summarize_fastpath_closed_loop_rows`
    ns = SimpleNamespace(timing_repeats=cl["repeats"], timing_warmups=cl["warmups"])
    resum = post_m4_suite._summarize_fastpath_closed_loop_rows(
        ns,
        [{"prompt_id": p} for p in cl["prompt_ids"]],
        [{"prompt_id": p} for p in cl["prompt_ids"]],
        cl["is_pilot_subset"],
        cl["arms"],
        speed_rows,
        profile_rows,
        world_size=cl["world_size"],
        gpus=cl["gpus"],
        rank_affinity=cl["rank_affinity"],
        max_rank_speed_seconds=cl["max_rank_speed_seconds"],
    )
    fb["closed_loop_matched_comparison"] = resum
    # Also update cluster_sharded_throughput to exclude that `round3_compact` row
    cal_file = next((fx / "post_m4_suite").glob("fastpath_bench_*/cluster_1gpu_calibration/result-cal-rank0.json"))
    cal_payload = json.loads(cal_file.read_text(encoding="utf-8"))
    cal_pids = set(cal_payload["selected_prompt_ids"])
    r3_sh = [r for r in speed_rows if r["fastpath_mode"] == "round3_compact"]
    eff_w = int(fb["cluster_sharded_throughput"]["world_size"])
    for ba in fb["cluster_sharded_throughput"]["by_arm"]:
        arm = ba["arm"]
        a_sh = [r for r in r3_sh if r["arm"] == arm and not post_m4_suite._is_repetition_excluded(r)]
        rk_means = [
            sum(rw["tokens_per_second"] for rw in a_sh if rw.get("rank", 0) == rk)
            / len([rw for rw in a_sh if rw.get("rank", 0) == rk])
            for rk in range(eff_w)
            if [rw for rw in a_sh if rw.get("rank", 0) == rk]
        ]
        a_c1 = [r for r in cal_payload["rows"] if r["arm"] == arm and not post_m4_suite._is_repetition_excluded(r)]
        a_cw = [r for r in a_sh if r["prompt_id"] in cal_pids]
        tps_1 = sum(r["tokens_per_second"] for r in a_c1) / len(a_c1)
        tps_w = sum(r["tokens_per_second"] for r in a_cw) / len(a_cw)
        ba["primary_timed_runs"] = len(a_sh)
        ba["mean_per_rank_tokens_per_second"] = sum(rk_means) / len(rk_means)
        ba["sum_across_ranks_tokens_per_second"] = sum(rk_means)
        ba["calibration_1gpu_tokens_per_second"] = tps_1
        ba["calibration_sharded_per_gpu_tokens_per_second"] = tps_w
        ba["sharded_over_1gpu_per_gpu_ratio"] = tps_w / tps_1
        ba["cluster_over_1gpu_throughput_ratio"] = eff_w * (tps_w / tps_1)
        ba["per_gpu_scaling_efficiency"] = tps_w / tps_1
    fb_p.write_text(json.dumps(fb), encoding="utf-8")

    aud = apm.PostM4SuiteAuditor(repo=REPO, artifacts=fx, scale="smoke", require_complete=True)
    res = aud.run_all()
    assert res["status"] == "passed", res["failures"]


def _build_reference_sandbox_with_post_m4(
    real_smoke_post_m4_fixture: Path, dst_root: Path
) -> Path:
    """Create a sandboxed copy of `/home/catid/drafter` in `tmp_path` with real M0-M4 symlinks + promoted `reference` `post_m4_suite`."""
    import os

    s_repo = dst_root / "sandbox_repo"
    s_repo.mkdir(parents=True, exist_ok=True)
    for item in REPO.iterdir():
        if item.name == "artifacts":
            continue
        if item.name in ("README.md", "NOTES.md", "progress.md", "AGENTS.md"):
            shutil.copy2(item, s_repo / item.name)
        elif item.name == "reports":
            (s_repo / "reports").mkdir(exist_ok=True)
            for r_item in item.iterdir():
                if r_item.name == "SUMMARY.md":
                    shutil.copy2(r_item, s_repo / "reports" / "SUMMARY.md")
                else:
                    os.symlink(r_item.resolve(), s_repo / "reports" / r_item.name)
        else:
            os.symlink(item.resolve(), s_repo / item.name)

    s_art = s_repo / "artifacts"
    s_art.mkdir(parents=True, exist_ok=True)
    for a_item in (REPO / "artifacts").iterdir():
        if a_item.name == "post_m4_suite":
            continue
        if a_item.name in ("milestones.json", "implementation_audit.json") and a_item.is_file():
            shutil.copy2(a_item, s_art / a_item.name)
        else:
            os.symlink(a_item.resolve(), s_art / a_item.name)

    # Copy smoke `post_m4_suite` into `s_art` and promote top-level suite/train scale tags to `"reference"`
    copied_out = _copy_suite_tree(real_smoke_post_m4_fixture, dst_root / "promoted_out")
    shutil.move(str(copied_out / "post_m4_suite"), str(s_art / "post_m4_suite"))
    old_p = str((copied_out / "post_m4_suite").resolve())
    new_p = str((s_art / "post_m4_suite").resolve())

    # Rewrite paths to `s_art / post_m4_suite` and refresh settings/manifest/session hashes
    old_fh_by_rel: dict[Path, str] = {}
    for sp_f in (s_art / "post_m4_suite").rglob("settings.json"):
        if not sp_f.is_symlink():
            old_fh_by_rel[sp_f.relative_to(s_art / "post_m4_suite")] = apm.fingerprint(
                json.loads(sp_f.read_text(encoding="utf-8"))
            )
    for jf in (s_art / "post_m4_suite").rglob("*.json"):
        if jf.is_symlink():
            continue
        txt = jf.read_text(encoding="utf-8")
        if old_p in txt:
            txt = txt.replace(old_p, new_p)
        data = json.loads(txt)
        if isinstance(data, dict) and data.get("scale") == "smoke":
            data["scale"] = "reference"
            txt = json.dumps(data)
        jf.write_text(txt, encoding="utf-8")

    fh_repl: list[tuple[str, str]] = []
    for rel_sp, old_fh in old_fh_by_rel.items():
        new_fh = apm.fingerprint(
            json.loads((s_art / "post_m4_suite" / rel_sp).read_text(encoding="utf-8"))
        )
        if new_fh != old_fh:
            fh_repl.append((old_fh, new_fh))
    if fh_repl:
        for jf in (s_art / "post_m4_suite").rglob("*.json"):
            if jf.is_symlink():
                continue
            txt = jf.read_text(encoding="utf-8")
            orig = txt
            for ofh, nfh in fh_repl:
                txt = txt.replace(ofh, nfh)
            if txt != orig:
                jf.write_text(txt, encoding="utf-8")

    for mf in (s_art / "post_m4_suite").rglob("manifest*.json"):
        if mf.is_symlink():
            continue
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        changed = False
        for entry in mdata.get("jobs", {}).values():
            rp = Path(entry.get("result_path", ""))
            if rp.is_file() and "result_sha256" in entry:
                entry["result_sha256"] = apm.sha256_file(rp)
                changed = True
        if changed:
            mf.write_text(json.dumps(mdata), encoding="utf-8")
    for sf in (s_art / "post_m4_suite").rglob("sessions/*.json"):
        if sf.is_symlink():
            continue
        sdata = json.loads(sf.read_text(encoding="utf-8"))
        changed = False
        for bval in sdata.get("bundles", {}).values():
            rshas = bval.get("result_sha256")
            if isinstance(rshas, dict):
                for rps in list(rshas.keys()):
                    rp = Path(rps)
                    if rp.is_file():
                        rshas[rps] = apm.sha256_file(rp)
                        changed = True
        if changed:
            sf.write_text(json.dumps(sdata), encoding="utf-8")

    # Sync nested `suites[name].config_fingerprint` in `post_m4_suite_summary.json` if needed
    ms_p = s_art / "post_m4_suite" / "post_m4_suite_summary.json"
    ms_d = json.loads(ms_p.read_text(encoding="utf-8"))
    for s_name in apm.SUITE_NAMES:
        sd = json.loads((s_art / "post_m4_suite" / f"{s_name}.json").read_text(encoding="utf-8"))
        if s_name in ms_d.get("suites", {}):
            ms_d["suites"][s_name]["config_fingerprint"] = sd.get("config_fingerprint")
            ms_d["suites"][s_name]["scale"] = "reference"
    ms_p.write_text(json.dumps(ms_d), encoding="utf-8")
    return s_repo


def test_update_docs_post_m4_end_to_end_idempotency_dryrun_and_pregate(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    import update_docs_post_m4 as ud

    s_repo = _build_reference_sandbox_with_post_m4(real_smoke_post_m4_fixture, tmp_path)
    target_relpaths = (
        "AGENTS.md",
        "README.md",
        "NOTES.md",
        "progress.md",
        "reports/SUMMARY.md",
        "artifacts/milestones.json",
        "artifacts/implementation_audit.json",
    )
    before_shas = {rel: apm.sha256_file(s_repo / rel) for rel in target_relpaths}

    # 1. `--dry_run` must pass all post-gates while leaving all 7 sandbox files 100% untouched!
    dry_res = ud.run_update_docs_post_m4(
        repo=s_repo,
        artifacts=s_repo / "artifacts",
        scale="reference",
        dry_run=True,
    )
    assert dry_res["status"] == "passed", dry_res["failures"]
    assert dry_res["m5_status"] == "passed"
    assert dry_res["implementation_audit_status"] == "reference_m0_m5_complete_verified"
    assert {rel: apm.sha256_file(s_repo / rel) for rel in target_relpaths} == before_shas

    # 2. Pre-gate rejection test: if `post_m4_suite_summary.json` is incomplete, 0 files are modified!
    ms_sum_p = s_repo / "artifacts" / "post_m4_suite" / "post_m4_suite_summary.json"
    orig_ms_sum_bytes = ms_sum_p.read_bytes()
    bad_sum = json.loads(orig_ms_sum_bytes)
    bad_sum["status"] = "incomplete"
    ms_sum_p.write_text(json.dumps(bad_sum), encoding="utf-8")
    fail_res = ud.run_update_docs_post_m4(
        repo=s_repo,
        artifacts=s_repo / "artifacts",
        scale="reference",
        dry_run=False,
    )
    assert fail_res["status"] == "failed_pre_gate"
    assert {rel: apm.sha256_file(s_repo / rel) for rel in target_relpaths} == before_shas
    # Restore exact bytes & mtime
    ms_sum_p.write_bytes(orig_ms_sum_bytes)

    # 3. Live sandbox run #1: updates all 7 files and passes BOTH `PostM4SuiteAuditor` & `PostGateEAuditor`!
    rc1 = ud.main(["--repo", str(s_repo), "--scale", "reference"])
    assert rc1 == 0
    run1_shas = {rel: apm.sha256_file(s_repo / rel) for rel in target_relpaths}
    for rel in target_relpaths:
        assert run1_shas[rel] != before_shas[rel], f"{rel} was not updated!"

    # Verify M0..M4 in milestones.json are 100% preserved and M5 == "passed"
    orig_ms = json.loads((REPO / "artifacts" / "milestones.json").read_text(encoding="utf-8"))
    new_ms = json.loads((s_repo / "artifacts" / "milestones.json").read_text(encoding="utf-8"))
    for k in ("M0", "M1-pre", "M1a", "M1b", "M2", "M3", "M4"):
        assert new_ms[k] == orig_ms[k]
    assert new_ms["M5"]["status"] == "passed"
    assert set(new_ms["M5"]["artifact_sha256"].keys()) == {
        "m2_refit_calibration",
        "online_head_to_head",
        "m5_eval",
        "fastpath_bench",
        "post_m4_suite_summary",
    }

    # Verify implementation_audit.json upgraded cleanly and tracks all 5 post_m4_suite JSONs
    ia = json.loads((s_repo / "artifacts" / "implementation_audit.json").read_text(encoding="utf-8"))
    assert ia["status"] == "reference_m0_m5_complete_verified"
    opt_m5 = next(r for r in ia["requirements"] if r["id"] == "optional_m5")
    assert opt_m5["status"] == "implemented_verified"
    assert any("Token Recycling" in c for c in ia["claims_not_established"])
    for rel in (
        "README.md",
        "NOTES.md",
        "progress.md",
        "reports/SUMMARY.md",
        "artifacts/milestones.json",
        "artifacts/post_m4_suite/post_m4_suite_summary.json",
    ):
        assert rel in ia["artifact_sha256_at_audit"]

    # Verify AGENTS.md has the standing 20-minute bug-hunt rotation bullet
    assert "Standing 20-Minute Bug-Hunt Rotation" in (s_repo / "AGENTS.md").read_text(encoding="utf-8")

    # 4. Live sandbox run #2: 100% byte-for-byte idempotency across ALL 7 files!
    rc2 = ud.main(["--repo", str(s_repo), "--scale", "reference"])
    assert rc2 == 0
    run2_shas = {rel: apm.sha256_file(s_repo / rel) for rel in target_relpaths}
    assert run2_shas == run1_shas

    # 5. Independent standalone CLI check with `audit_post_m4_suite.py --require_complete --check_implementation_audit`
    rc_aud = apm.main(
        [
            "--scale",
            "reference",
            "--repo",
            str(s_repo),
            "--artifacts",
            str(s_repo / "artifacts"),
            "--require_complete",
            "--check_implementation_audit",
            "--output_dir",
            str(tmp_path / "verify_aud_out"),
        ]
    )
    assert rc_aud == 0


def test_verifier_301_all_4_post_m4_audit_fault_gaps_caught(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    """Verify all 4 fault-injection gaps from `#301` are deterministically caught."""
    # Gap 1 (`POSTM4-AUDIT-TIMING-GROUPS-1`): forge `throughput_gain_supported_by_95_ci=True`
    # in `online_head_to_head.json` (`isolated_timing_groups`) and `phase_profiles` in `m5_eval`'s `timing.json`!
    g1 = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "gap1")
    h2h_p = g1 / "post_m4_suite" / "online_head_to_head.json"
    h2h = json.loads(h2h_p.read_text(encoding="utf-8"))
    for tg in h2h["isolated_timing_groups"]:
        if tg["arm"] != "plain_dflash":
            tg["throughput_gain_supported_by_95_ci"] = True
            tg["matched_arm_tokens_per_second"] = 99999.0
    h2h_p.write_text(json.dumps(h2h), encoding="utf-8")

    m5_p = g1 / "post_m4_suite" / "m5_eval.json"
    m5 = json.loads(m5_p.read_text(encoding="utf-8"))
    tm_p = Path(m5["closed_loop"]["artifacts"]["timing_json"])
    tm = json.loads(tm_p.read_text(encoding="utf-8"))
    tm["phase_profiles"][0]["phase_ms_median"]["lookahead_iterations"] = 99999.0
    tm_p.write_text(json.dumps(tm), encoding="utf-8")

    res_g1 = apm.PostM4SuiteAuditor(repo=REPO, artifacts=g1, scale="smoke", require_complete=True).run_all()
    assert res_g1["status"] == "failed"
    assert any("suite isolated_timing_groups drift" in f for f in res_g1["failures"])
    assert any("timing.phase_profiles drift" in f for f in res_g1["failures"])

    # Gap 2 (`POSTM4-AUDIT-ONLINE-ROW-GUARDS-2`): `row['online'] = None` on online arms,
    # `accumulate_stride=True` on `offline_online_stride5`, and `count_gate_kappa=999.0` on `online_bigram_count_gated`!
    g2 = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "gap2")
    h2h2 = json.loads((g2 / "post_m4_suite" / "online_head_to_head.json").read_text(encoding="utf-8"))
    h2h_eval_dir = Path(h2h2["artifacts"]["summary_json"]).parent / "evaluate"
    for mf in sorted(h2h_eval_dir.glob("manifest-rank*.json")):
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        for entry in mdata["jobs"].values():
            rp = Path(entry["result_path"])
            rw = json.loads(rp.read_text(encoding="utf-8"))
            if rw["arm"] == "online_bigram_count_gated":
                rw["bigram"]["count_gate_kappa"] = 999.0
                rp.write_text(json.dumps(rw), encoding="utf-8")
                entry["result_sha256"] = apm.sha256_file(rp)
                break
        mf.write_text(json.dumps(mdata), encoding="utf-8")

    m5_2 = json.loads((g2 / "post_m4_suite" / "m5_eval.json").read_text(encoding="utf-8"))
    m5_eval_dir = Path(m5_2["closed_loop"]["artifacts"]["summary_json"]).parent / "evaluate"
    mutated_arms: set[str] = set()
    for mf in sorted(m5_eval_dir.glob("manifest-rank*.json")):
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        for entry in mdata["jobs"].values():
            rp = Path(entry["result_path"])
            rw = json.loads(rp.read_text(encoding="utf-8"))
            if rw["arm"] in ("online_two_sided_pl", "exact_observed_pairs_online") and rw["arm"] not in mutated_arms:
                rw["online"] = None
                rp.write_text(json.dumps(rw), encoding="utf-8")
                entry["result_sha256"] = apm.sha256_file(rp)
                mutated_arms.add(rw["arm"])
            elif rw["arm"] == "offline_online_stride5" and rw["arm"] not in mutated_arms:
                rw["online"]["accumulate_stride"] = True
                rp.write_text(json.dumps(rw), encoding="utf-8")
                entry["result_sha256"] = apm.sha256_file(rp)
                mutated_arms.add(rw["arm"])
        mf.write_text(json.dumps(mdata), encoding="utf-8")

    res_g2 = apm.PostM4SuiteAuditor(repo=REPO, artifacts=g2, scale="smoke", require_complete=True).run_all()
    assert res_g2["status"] == "failed"
    assert any("count_gate_kappa=999.0" in f for f in res_g2["failures"])
    assert any("online_two_sided_pl" in f and "missing/invalid row['online']" in f for f in res_g2["failures"])
    assert any("exact_observed_pairs_online" in f and "missing/invalid row['online']" in f for f in res_g2["failures"])
    assert any("offline_online_stride5" in f and "unbuffered single-round" in f for f in res_g2["failures"])

    # Gap 3 (`POSTM4-AUDIT-M5-OFFLINE-GUARDS-3`): corrupt `m5_two_sided_pl.json`, `bool` `collision_count=False`,
    # `test_grouped_kl_delta_vs_signed_hash=None`, and drop `frequency_weighted_collision_rate`!
    g3 = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "gap3")
    m5_3_p = g3 / "post_m4_suite" / "m5_eval.json"
    m5_3 = json.loads(m5_3_p.read_text(encoding="utf-8"))
    pl_rep_p = Path(m5_3["offline_fits"]["two_sided_pseudolikelihood"]["report_path"])
    pl_fit = json.loads(pl_rep_p.read_text(encoding="utf-8"))
    pl_fit["status"] = "failed"
    pl_fit["objective"] = "union_rest_grouped"
    pl_rep_p.write_text(json.dumps(pl_fit), encoding="utf-8")

    ex_rep_p = Path(m5_3["offline_fits"]["exact_observed_pairs"]["report_path"])
    ex_fit = json.loads(ex_rep_p.read_text(encoding="utf-8"))
    ex_fit["stages"]["dense_sparse"]["sparse_collision_statistics"].pop("frequency_weighted_collision_rate", None)
    ex_rep_p.write_text(json.dumps(ex_fit), encoding="utf-8")

    m5_3["offline_fits"]["exact_observed_pairs"]["collision_count"] = False
    m5_3["offline_fits"]["exact_observed_pairs"]["test_grouped_kl_delta_vs_signed_hash"] = None
    m5_3_p.write_text(json.dumps(m5_3), encoding="utf-8")

    res_g3 = apm.PostM4SuiteAuditor(repo=REPO, artifacts=g3, scale="smoke", require_complete=True).run_all()
    assert res_g3["status"] == "failed"
    assert any("m5_two_sided_pl.json status='failed'" in f for f in res_g3["failures"])
    assert any("m5_two_sided_pl.json objective='union_rest_grouped'" in f for f in res_g3["failures"])
    assert any("top-level collision_count=False != 0" in f for f in res_g3["failures"])
    assert any("missing/invalid test_grouped_kl_delta_vs_signed_hash=None" in f for f in res_g3["failures"])
    assert any("frequency_weighted_collision_rate=None" in f for f in res_g3["failures"])

    # Gap 4 (`POSTM4-AUDIT-STAGE-JSON-CROSSCHECK-4`): wipe `m5/summary.json` & `m5/timing.json` fields!
    g4 = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "gap4")
    h2h_4 = json.loads((g4 / "post_m4_suite" / "online_head_to_head.json").read_text(encoding="utf-8"))
    h2h_sum4_p = Path(h2h_4["artifacts"]["summary_json"])
    h2h_sum4 = json.loads(h2h_sum4_p.read_text(encoding="utf-8"))
    h2h_sum4["head_to_head_online_offset_comparison"] = []
    h2h_sum4_p.write_text(json.dumps(h2h_sum4), encoding="utf-8")

    m5_4 = json.loads((g4 / "post_m4_suite" / "m5_eval.json").read_text(encoding="utf-8"))
    sum4_p = Path(m5_4["closed_loop"]["artifacts"]["summary_json"])
    sum4 = json.loads(sum4_p.read_text(encoding="utf-8"))
    sum4["matched_support_comparisons"] = []
    sum4["offset_bin_slices"] = []
    sum4_p.write_text(json.dumps(sum4), encoding="utf-8")

    tm4_p = Path(m5_4["closed_loop"]["artifacts"]["timing_json"])
    tm4 = json.loads(tm4_p.read_text(encoding="utf-8"))
    tm4["groups"] = []
    tm4["phase_profiles"] = []
    tm4["offset_bin_slices"] = []
    tm4_p.write_text(json.dumps(tm4), encoding="utf-8")

    res_g4 = apm.PostM4SuiteAuditor(repo=REPO, artifacts=g4, scale="smoke", require_complete=True).run_all()
    assert res_g4["status"] == "failed"
    assert any("summary.matched_support_comparisons drift" in f for f in res_g4["failures"])
    assert any("summary.offset_bin_slices drift" in f for f in res_g4["failures"])
    assert any("summary.head_to_head_online_offset_comparison drift" in f for f in res_g4["failures"])
    assert any("timing.groups drift" in f for f in res_g4["failures"])
    assert any("timing.phase_profiles drift" in f for f in res_g4["failures"])
    assert any("timing.offset_bin_slices drift" in f for f in res_g4["failures"])


def test_m5eval_audit_bug1_short_stride5_online_final_state_canonicalization() -> None:
    """Regression test for `M5EVAL-AUDIT-BUG-1`: `<5`-block `stride=5` requests tie on `(u=0, active_buckets=0, ...)`
    when dynamically scheduled across multi-rank workers (`manifest-rank*` order != `select_prompts` order)."""
    import build_and_audit_reports as bar

    base_zero = {
        "adapter_version": 0,
        "u": 0,
        "active_buckets": 0,
        "state_bytes_estimate_total": 0,
        "dS_norm": 0.0,
        "db_norm": 0.0,
        "last_full_loss": None,
        "accumulate_stride": True,
        "buffered_rounds_accumulated_total": 0,
    }
    s_r3 = dict(base_zero, last_round=3, buffered_pending_rounds=3)
    s_r4 = dict(base_zero, last_round=4, buffered_pending_rounds=4)
    s_eos = dict(base_zero, last_round=-1, buffered_pending_rounds=0)
    # Also verify arbitrary extra-field tie-breaking via canonical JSON fallback
    s_r3_alt = dict(s_r3, dense_optimizer="adamw")

    g_rank_order = [
        {
            "arm": "online_buffered_stride5",
            "excluded_prompt_ids": ["p_rep"],
            "online_final_state": [s_r4, s_r3_alt, s_eos, s_r3],
        }
    ]
    g_prompt_order = [
        {
            "arm": "online_buffered_stride5",
            "online_final_state": [dict(s_r3), dict(s_eos), dict(s_r3_alt), dict(s_r4)],
        }
    ]
    ok, diff = bar.close_nested(
        bar._canonicalize_group_list(g_rank_order),
        bar._canonicalize_group_list(g_prompt_order),
    )
    assert ok, f"M5EVAL-AUDIT-BUG-1 regression: {diff}"

    #Genuine mismatch (`last_round=2` vs `last_round=3`) must still fail `close_nested`
    g_mismatch = [
        {
            "arm": "online_buffered_stride5",
            "online_final_state": [dict(s_r3, last_round=2), dict(s_eos), dict(s_r3_alt), dict(s_r4)],
        }
    ]
    ok_bad, _ = bar.close_nested(
        bar._canonicalize_group_list(g_rank_order),
        bar._canonicalize_group_list(g_mismatch),
    )
    assert not ok_bad


def test_m5eval_audit_bug2_exact_observed_pairs_pre_proximal_overflow_invariants(
    real_smoke_post_m4_fixture: Path, tmp_path: Path
) -> None:
    """Regression test for `M5EVAL-AUDIT-BUG-2`: `LazyExactObservedPairs(overflow_policy='freeze_new')` can
    legitimately overflow during pre-proximal admission (`overflow_events >= 1, dropped_unique_keys >= overflow_events`)
    and then shrink to `peak_active < capacity` (even `peak_active == 0`) after `proximal_l1` zero reclamation."""
    import torch
    from edraft.adapter import DenseAdapter
    from edraft.online import OnlineState

    # 1. Reproduce with real `OnlineState` (`sparse_capacity=4` + strong `lambda_l1_s` shrinking admitted keys to 0)
    torch.manual_seed(0)
    adapter = DenseAdapter(torch.randn(32, 8), p=8, D=2, M_sparse=8)
    state = OnlineState(
        adapter,
        lr=1e-3,
        update_stride=1,
        enable_sparse=True,
        lambda_l2=0.0,
        lambda_l1_s=10.0,
        sparse_backend="exact_observed_pairs",
        sparse_capacity=4,
        sparse_overflow_policy="freeze_new",
    )
    assert state.dS is not None
    keys = torch.arange(48, dtype=torch.int64)
    grads = torch.full((48,), 0.05, dtype=torch.float32)  # |lr * grad| = 5e-5 <= lr * lambda_l1 = 1e-2 -> shrinks to 0
    state.dS.step(keys, grads, lr=1e-3, lambda_l1=10.0)
    metrics = state.metrics()
    assert metrics["sparse_capacity"] == 4
    assert metrics["sparse_overflow_events"] == 1
    assert metrics["sparse_dropped_unique_keys"] == 44
    assert metrics["sparse_peak_active"] == 0
    assert metrics["active_buckets"] == 0

    # 2. Inject those exact `OnlineState.metrics()` values into a smoke `exact_observed_pairs_online` row (+ sync group summary)
    fx_ok = _copy_suite_tree(real_smoke_post_m4_fixture, tmp_path / "ovf_valid")
    m5_p = fx_ok / "post_m4_suite" / "m5_eval.json"
    m5 = json.loads(m5_p.read_text(encoding="utf-8"))
    sum_p = Path(m5["closed_loop"]["artifacts"]["summary_json"])
    eval_dir = sum_p.parent / "evaluate"
    mutated = False
    for mf in sorted(eval_dir.glob("manifest-rank*.json")):
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        for entry in mdata["jobs"].values():
            rp = Path(entry["result_path"])
            rw = json.loads(rp.read_text(encoding="utf-8"))
            if rw["arm"] == "exact_observed_pairs_online" and not mutated:
                rw["online"].update(
                    {
                        "sparse_capacity": 4,
                        "sparse_overflow_events": 1,
                        "sparse_dropped_unique_keys": 44,
                        "sparse_peak_active": 0,
                        "active_buckets": 0,
                    }
                )
                rp.write_text(json.dumps(rw), encoding="utf-8")
                entry["result_sha256"] = apm.sha256_file(rp)
                mutated = True
                break
        mf.write_text(json.dumps(mdata), encoding="utf-8")
    assert mutated

    # Refresh `summary.json` and `m5_eval.json` `online_final_state` so raw recomputation matches
    import build_and_audit_reports as bar

    helper = bar.PostGateEAuditor(repo=REPO, artifacts=fx_ok, scale="smoke")
    eval_rows = []
    for mf in sorted(eval_dir.glob("manifest-rank*.json")):
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        for entry in mdata["jobs"].values():
            eval_rows.append(json.loads(Path(entry["result_path"]).read_text(encoding="utf-8")))
    recomp_groups = helper.recompute_group_rows(
        eval_rows, isolated_timing=False, exclude_repetition=True
    )
    clean_groups = [{k: v for k, v in g.items() if k != "excluded_prompt_ids"} for g in recomp_groups]
    sum_d = json.loads(sum_p.read_text(encoding="utf-8"))
    sum_d["groups"] = clean_groups
    sum_p.write_text(json.dumps(sum_d), encoding="utf-8")
    m5["closed_loop"]["overall_evaluation_groups"] = clean_groups
    m5_p.write_text(json.dumps(m5), encoding="utf-8")

    res_ok = apm.PostM4SuiteAuditor(repo=REPO, artifacts=fx_ok, scale="smoke", require_complete=True).run_all()
    assert res_ok["status"] == "passed", res_ok["failures"]

    # 3. Verify genuine invariant violations (`overflow_events=1, dropped_unique_keys=0` and `peak_active > capacity`) ARE caught!
    fx_bad = _copy_suite_tree(fx_ok, tmp_path / "ovf_invalid")
    m5_b_p = fx_bad / "post_m4_suite" / "m5_eval.json"
    m5_b = json.loads(m5_b_p.read_text(encoding="utf-8"))
    sum_b_p = Path(m5_b["closed_loop"]["artifacts"]["summary_json"])
    eval_b_dir = sum_b_p.parent / "evaluate"
    idx_mut = 0
    for mf in sorted(eval_b_dir.glob("manifest-rank*.json")):
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        for entry in mdata["jobs"].values():
            rp = Path(entry["result_path"])
            rw = json.loads(rp.read_text(encoding="utf-8"))
            if rw["arm"] == "exact_observed_pairs_online":
                if idx_mut == 0:
                    rw["online"]["sparse_overflow_events"] = 2
                    rw["online"]["sparse_dropped_unique_keys"] = 1  # dropped < overflow_events -> invalid!
                elif idx_mut == 1:
                    rw["online"]["sparse_capacity"] = 4
                    rw["online"]["sparse_peak_active"] = 5  # peak_active > capacity -> invalid!
                rp.write_text(json.dumps(rw), encoding="utf-8")
                entry["result_sha256"] = apm.sha256_file(rp)
                idx_mut += 1
        mf.write_text(json.dumps(mdata), encoding="utf-8")
    res_bad = apm.PostM4SuiteAuditor(repo=REPO, artifacts=fx_bad, scale="smoke", require_complete=True).run_all()
    assert res_bad["status"] == "failed"
    assert any("inconsistent sparse_overflow_events=2 vs sparse_dropped_unique_keys=1" in f for f in res_bad["failures"])
    assert any("invalid sparse_peak_active=5" in f for f in res_bad["failures"])


def test_allow_pilot_timing_prompts_subset_isolated_timing_complete_guard(tmp_path: Path) -> None:
    """Regression test for `#431` / `#432`: running `edraft.post_m4_suite --timing_prompts 1` sets
    `summary["isolated_timing_complete"] = False` and omits pilot timing from `matched_support_comparisons`
    and `head_to_head_online_offset_comparison`.
    - `allow_pilot=True, require_complete=True` must PASS with 0 drift.
    - `allow_pilot=False, require_complete=True` must FAIL (`isolated_timing_complete=False != True` + `pilot_complete`).
    """
    from edraft import post_m4_suite

    pilot_out = tmp_path / "pilot_pm4"
    post_m4_suite.main(
        [
            "--scale",
            "smoke",
            "--suite",
            "all",
            "--gpus",
            "0",
            "--output",
            str(pilot_out),
            "--bench_warmups",
            "1",
            "--bench_repeats",
            "2",
            "--timing_warmups",
            "1",
            "--timing_repeats",
            "2",
            "--timing_prompts",
            "1",
            "--smoke_tokens",
            "16",
            "--online_arms",
            "plain_dflash",
            "online_bigram",
            "online_bigram_count_gated",
            "online_dbS_zero_init",
            "zero_offline_online_only",
            "--m5_eval_arms",
            "plain_dflash",
            "dense_candidate",
            "lookahead_rerelax_candidate",
            "exact_observed_pairs_online",
            "online_buffered_stride5",
        ]
    )

    h2h_rep = json.loads((pilot_out / "post_m4_suite" / "online_head_to_head.json").read_text(encoding="utf-8"))
    h2h_sum = json.loads(Path(h2h_rep["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert h2h_rep["status"] == "pilot_complete"
    assert h2h_sum["isolated_timing_complete"] is False

    res_pilot = apm.PostM4SuiteAuditor(
        repo=REPO,
        artifacts=pilot_out,
        scale="smoke",
        require_complete=True,
        allow_pilot=True,
    ).run_all()
    assert res_pilot["status"] == "passed", res_pilot["failures"]
    assert res_pilot["failures"] == []

    res_strict = apm.PostM4SuiteAuditor(
        repo=REPO,
        artifacts=pilot_out,
        scale="smoke",
        require_complete=True,
        allow_pilot=False,
    ).run_all()
    assert res_strict["status"] == "failed"
    assert any("isolated_timing_complete=False != True" in f for f in res_strict["failures"])






