#!/usr/bin/env python3
"""Comprehensive end-to-end, chained Phase-A -> Phase-B, idempotency & link-safety tests for `update_docs_phase_b.py`."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys

import pytest

_THIS_DIR = Path(__file__).resolve().parent
REPO = Path("/home/catid/drafter")
IWT = Path(
    "/home/catid/.gemini/antigravity-cli/worktrees/"
    "cfcf3ba7-dc2c-4cac-926a-577986ac8615/swarm-integrator-970cc63a"
)
if (IWT / "src").is_dir():
    sys.path[:] = [p for p in sys.path if p != str(REPO / "src")]
    if str(IWT / "src") not in sys.path:
        sys.path.insert(0, str(IWT / "src"))
    if "edraft" in sys.modules and not str(getattr(sys.modules["edraft"], "__file__", "")).startswith(
        str(IWT / "src")
    ):
        for _mod_k in [k for k in list(sys.modules) if k == "edraft" or k.startswith("edraft.")]:
            sys.modules.pop(_mod_k, None)
for p in (str(_THIS_DIR), str(REPO / "src")):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.append(p)

import audit_post_m4_suite as apm  # noqa: E402
from edraft import phase_b_suite, post_m4_suite  # noqa: E402
from test_phase_b_reporting import _build_synthetic_full_scale_workspace  # noqa: E402
from test_post_m4_reporting import _copy_suite_tree  # noqa: E402
import update_docs_phase_b as upb  # noqa: E402
import update_docs_post_m4 as upm4  # noqa: E402


TARGET_RELPATHS = (
    "AGENTS.md",
    "README.md",
    "NOTES.md",
    "progress.md",
    "reports/SUMMARY.md",
    "artifacts/milestones.json",
    "artifacts/implementation_audit.json",
)


def _check_markdown_links_and_headings(s_repo: Path) -> None:
    """Verify zero broken relative links outside fenced code blocks and zero H1 `# ` inside `SUMMARY.md` table body."""
    link_re = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for rel in ("AGENTS.md", "README.md", "NOTES.md", "progress.md", "reports/SUMMARY.md"):
        fp = s_repo / rel
        lines = fp.read_text(encoding="utf-8").splitlines()
        in_fence = False
        for idx, ln in enumerate(lines, 1):
            if ln.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if rel == "reports/SUMMARY.md" and idx > 5 and ln.startswith("# "):
                raise AssertionError(f"Un-demoted H1 heading in {rel}:{idx}: {ln}")
            for raw_target in link_re.findall(ln):
                target = raw_target.split("#", 1)[0].strip()
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (fp.parent / target).resolve()
                assert resolved.exists(), f"Broken link in {rel}:{idx} -> {raw_target} ({resolved})"


def _create_phase_b_sandbox_repo(dst_root: Path) -> Path:
    """Create a sandbox copy of `swarm-integrator-970cc63a` (`077d4df3`) with real M0-M4 reference symlinks."""
    src_repo = IWT if IWT.is_dir() else REPO
    s_repo = dst_root / "sandbox_phase_b"
    s_repo.mkdir(parents=True, exist_ok=True)
    for item in src_repo.iterdir():
        if item.name == "artifacts":
            continue
        if item.name in ("README.md", "NOTES.md", "progress.md", "AGENTS.md"):
            # Copy current canonical docs from REPO (`main`)
            shutil.copy2(REPO / item.name, s_repo / item.name)
        elif item.name == "reports":
            (s_repo / "reports").mkdir(exist_ok=True)
            shutil.copy2(REPO / "reports" / "SUMMARY.md", s_repo / "reports" / "SUMMARY.md")
        else:
            os.symlink(item.resolve(), s_repo / item.name)

    s_art = s_repo / "artifacts"
    s_art.mkdir(parents=True, exist_ok=True)
    for a_item in (REPO / "artifacts").iterdir():
        if a_item.name in ("post_m4_suite", "full_scale"):
            continue
        if a_item.name in ("milestones.json", "implementation_audit.json") and a_item.is_file():
            shutil.copy2(a_item, s_art / a_item.name)
        else:
            os.symlink(a_item.resolve(), s_art / a_item.name)
    return s_repo


def test_chained_phase_a_then_phase_b_end_to_end_idempotency_and_links(tmp_path: Path) -> None:
    """Verify full sequential Phase-A (`update_docs_post_m4.py`) -> Phase-B (`update_docs_phase_b.py`) flow."""
    # 1. Generate real smoke `post_m4_suite` and promote scale tags to `reference`
    pm4_raw = tmp_path / "pm4_smoke"
    post_m4_suite.main(
        [
            "--scale",
            "smoke",
            "--suite",
            "all",
            "--gpus",
            "0,1",
            "--output",
            str(pm4_raw),
            "--bench_warmups",
            "1",
            "--bench_repeats",
            "2",
            "--timing_warmups",
            "1",
            "--timing_repeats",
            "2",
            "--smoke_tokens",
            "20",
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

    s_repo = _create_phase_b_sandbox_repo(tmp_path)
    s_art = s_repo / "artifacts"

    # Place promoted `post_m4_suite` into `s_art / post_m4_suite`
    copied_pm4 = _copy_suite_tree(pm4_raw, tmp_path / "pm4_copied")
    shutil.move(str(copied_pm4 / "post_m4_suite"), str(s_art / "post_m4_suite"))
    old_p = str((copied_pm4 / "post_m4_suite").resolve())
    new_p = str((s_art / "post_m4_suite").resolve())
    old_fh: dict[Path, str] = {}
    for sp_f in (s_art / "post_m4_suite").rglob("settings.json"):
        if not sp_f.is_symlink():
            old_fh[sp_f.relative_to(s_art / "post_m4_suite")] = apm.fingerprint(
                json.loads(sp_f.read_text(encoding="utf-8"))
            )
    for jf in (s_art / "post_m4_suite").rglob("*.json"):
        if jf.is_symlink():
            continue
        txt = jf.read_text(encoding="utf-8").replace(old_p, new_p)
        d = json.loads(txt)
        if isinstance(d, dict) and d.get("scale") == "smoke":
            d["scale"] = "reference"
            txt = json.dumps(d)
        jf.write_text(txt, encoding="utf-8")
    fh_repl = [
        (ofh, apm.fingerprint(json.loads((s_art / "post_m4_suite" / rel_sp).read_text(encoding="utf-8"))))
        for rel_sp, ofh in old_fh.items()
    ]
    fh_repl = [(a, b) for a, b in fh_repl if a != b]
    if fh_repl:
        for jf in (s_art / "post_m4_suite").rglob("*.json"):
            if jf.is_symlink():
                continue
            t0 = jf.read_text(encoding="utf-8")
            t1 = t0
            for a, b in fh_repl:
                t1 = t1.replace(a, b)
            if t1 != t0:
                jf.write_text(t1, encoding="utf-8")
    for mf in (s_art / "post_m4_suite").rglob("manifest*.json"):
        if not mf.is_symlink():
            md = json.loads(mf.read_text(encoding="utf-8"))
            for e in md.get("jobs", {}).values():
                rp = Path(e.get("result_path", ""))
                if rp.is_file() and "result_sha256" in e:
                    e["result_sha256"] = apm.sha256_file(rp)
            mf.write_text(json.dumps(md), encoding="utf-8")
    for sf in (s_art / "post_m4_suite").rglob("sessions/*.json"):
        if not sf.is_symlink():
            sd = json.loads(sf.read_text(encoding="utf-8"))
            for bv in sd.get("bundles", {}).values():
                rs = bv.get("result_sha256")
                if isinstance(rs, dict):
                    for k in list(rs.keys()):
                        if Path(k).is_file():
                            rs[k] = apm.sha256_file(Path(k))
            sf.write_text(json.dumps(sd), encoding="utf-8")
    ms_pm4 = s_art / "post_m4_suite" / "post_m4_suite_summary.json"
    ms_pm4_d = json.loads(ms_pm4.read_text(encoding="utf-8"))
    for sn in apm.SUITE_NAMES:
        ms_pm4_d["suites"][sn]["config_fingerprint"] = json.loads(
            (s_art / "post_m4_suite" / f"{sn}.json").read_text(encoding="utf-8")
        )["config_fingerprint"]
        ms_pm4_d["suites"][sn]["scale"] = "reference"
    ms_pm4.write_text(json.dumps(ms_pm4_d), encoding="utf-8")

    # Step A: Run Phase-A updater (`update_docs_post_m4.py`)!
    assert upm4.main(["--repo", str(s_repo), "--scale", "reference"]) == 0
    after_phase_a_shas = {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS}
    after_phase_a_ms = json.loads((s_art / "milestones.json").read_text(encoding="utf-8"))
    assert after_phase_a_ms["M5"]["status"] == "passed"

    # Step B: Execute real `edraft.phase_b_suite --scale smoke --suite all` into `s_art / full_scale`!
    pb_out = s_art / "full_scale"
    phase_b_suite.main(
        [
            "--scale",
            "smoke",
            "--suite",
            "all",
            "--gpus",
            "0,1",
            "--output",
            str(pb_out),
            "--reference_artifacts",
            str(s_art),
            "--timing_repeats",
            "2",
            "--timing_warmups",
            "1",
            "--smoke_tokens",
            "16",
            "--block_shuffle_window",
            "2",
        ]
    )

    # 1. `--dry_run` leaves all 7 Phase-A files 100% untouched while passing ALL 4 post-write gates!
    dry = upb.run_update_docs_phase_b(
        repo=s_repo,
        reference_artifacts=s_art,
        phase_b_root=pb_out,
        scale="smoke",
        suite="all",
        dry_run=True,
    )
    assert dry["status"] == "passed", dry["failures"]
    assert dry["post_m4_audit_checks"] > 200
    assert {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS} == after_phase_a_shas

    # 2a. Pre-gate rejection (`PHASEB-PARTIAL-SUITE-OVERCLAIM-3`): partial `--suite` must be rejected with 0 writes!
    for partial_suite in ("full_scale", "ref_secondary_ablations", "bigram_eviction"):
        assert upb.main(["--repo", str(s_repo), "--scale", "smoke", "--suite", partial_suite]) == 1
        assert {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS} == after_phase_a_shas

    # 2b. Pre-gate rejection (`PHASEB-MASTER-SUMMARY-SUITES-2`): Step-4 partial `phase_b_summary.json`
    # (`status="complete"`, `executed_suites=["full_scale"]`) MUST fail `--suite all`!
    pb_sum_p = pb_out / "phase_b_summary.json"
    orig_pb_bytes = pb_sum_p.read_bytes()
    step4_partial_pb = json.loads(orig_pb_bytes)
    step4_partial_pb["executed_suites"] = ["full_scale"]
    step4_partial_pb["suites"] = {"full_scale": step4_partial_pb["suites"]["full_scale"]}
    pb_sum_p.write_text(json.dumps(step4_partial_pb), encoding="utf-8")
    fail_step4 = upb.run_update_docs_phase_b(
        repo=s_repo,
        reference_artifacts=s_art,
        phase_b_root=pb_out,
        scale="smoke",
        suite="all",
        dry_run=False,
    )
    assert fail_step4["status"] == "failed_pre_gate"
    assert any("missing required suite" in f for f in fail_step4["failures"])
    assert {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS} == after_phase_a_shas

    # 2c. Pre-gate rejection: `status="incomplete"` on `phase_b_summary.json` aborts with 0 disk writes!
    bad_pb = json.loads(orig_pb_bytes)
    bad_pb["status"] = "incomplete"
    pb_sum_p.write_text(json.dumps(bad_pb), encoding="utf-8")
    fail = upb.run_update_docs_phase_b(
        repo=s_repo,
        reference_artifacts=s_art,
        phase_b_root=pb_out,
        scale="smoke",
        suite="all",
        dry_run=False,
    )
    assert fail["status"] == "failed_pre_gate"
    assert {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS} == after_phase_a_shas
    pb_sum_p.write_bytes(orig_pb_bytes)

    # 3. Live Phase-B Run #1: updates docs/milestones/implementation_audit while preserving Phase-A 100%!
    assert upb.main(["--repo", str(s_repo), "--scale", "smoke", "--suite", "all"]) == 0
    run1_shas = {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS}
    # AGENTS.md already had the rotation bullet from Phase A -> byte-identical!
    assert run1_shas["AGENTS.md"] == after_phase_a_shas["AGENTS.md"]
    for rel in TARGET_RELPATHS[1:]:
        assert run1_shas[rel] != after_phase_a_shas[rel], f"{rel} was not updated by Phase B!"

    # Verify M0..M4 AND M5 in milestones.json are 100% preserved and `phase_b_full_scale == "passed"`
    pb_ms = json.loads((s_art / "milestones.json").read_text(encoding="utf-8"))
    for k in ("M0", "M1-pre", "M1a", "M1b", "M2", "M3", "M4", "M5"):
        assert pb_ms[k] == after_phase_a_ms[k], f"{k} drifted during Phase-B doc update!"
    assert pb_ms["phase_b_full_scale"]["status"] == "passed"
    assert set(pb_ms["phase_b_full_scale"]["artifact_sha256"].keys()) == {
        "bigram_eviction",
        "full_vocab_lookahead",
        "pure_t1_fit",
        "full_scale_pipeline",
        "phase_b_summary",
    }

    # Verify BOTH Phase-A AND Phase-B delimited blocks exist cleanly in SUMMARY, README, NOTES, progress
    sum_txt = (s_repo / "reports" / "SUMMARY.md").read_text(encoding="utf-8")
    assert "<!-- BEGIN POST_M4_M5_SUMMARY -->" in sum_txt
    assert "<!-- BEGIN PHASE_B_FULL_SCALE_SUMMARY -->" in sum_txt
    _check_markdown_links_and_headings(s_repo)

    # Verify implementation_audit.json tracks BOTH `post_m4_suite` AND `full_scale` SHA256s!
    import build_implementation_audit as bia

    ia = json.loads((s_art / "implementation_audit.json").read_text(encoding="utf-8"))
    assert ia["status"] == "reference_m0_m5_complete_verified"
    for rel in bia.POST_M4_AUDIT_RELPATHS + bia.PHASE_B_AUDIT_RELPATHS:
        assert rel in ia["artifact_sha256_at_audit"], f"Missing {rel}"
    assert any(s.get("milestone") == "Phase-B" for s in ia["scheduled_reference_experiments"])

    # 3b. `PHASEB-IMPL-AUDIT-WIPE-1` regression proof:
    # Standalone `build_implementation_audit()` AND re-running `update_docs_post_m4.py` MUST preserve
    # all 5 `artifacts/full_scale/*.json` SHA256s + `milestone="Phase-B"`, and wiping either MUST fail `verify_implementation_audit`!
    standalone_ia = bia.build_implementation_audit(s_repo, s_art)
    assert bia.verify_implementation_audit(standalone_ia, s_repo, s_art) == []
    for rel in bia.PHASE_B_AUDIT_RELPATHS:
        assert rel in standalone_ia["artifact_sha256_at_audit"]
    assert any(s.get("milestone") == "Phase-B" for s in standalone_ia["scheduled_reference_experiments"])

    wiped_ia = json.loads(json.dumps(standalone_ia))
    wiped_ia["artifact_sha256_at_audit"].pop("artifacts/full_scale/phase_b_summary.json")
    wiped_ia["scheduled_reference_experiments"] = [
        s for s in wiped_ia["scheduled_reference_experiments"] if s.get("milestone") != "Phase-B"
    ]
    wipe_errs = bia.verify_implementation_audit(wiped_ia, s_repo, s_art)
    assert any("phase_b_summary.json" in e for e in wipe_errs)
    assert any("missing passed Phase-B entry" in e for e in wipe_errs)

    # Re-running `update_docs_post_m4.py` after Phase-B must ALSO preserve Phase-B in `milestones.json` & `implementation_audit.json`!
    assert upm4.main(["--repo", str(s_repo), "--scale", "reference"]) == 0
    ia_after_pm4 = json.loads((s_art / "implementation_audit.json").read_text(encoding="utf-8"))
    for rel in bia.PHASE_B_AUDIT_RELPATHS:
        assert rel in ia_after_pm4["artifact_sha256_at_audit"]
    assert any(s.get("milestone") == "Phase-B" for s in ia_after_pm4["scheduled_reference_experiments"])

    # 4. Live Phase-B Run #2 & #3: 100% byte-for-byte SHA256 idempotency across ALL 7 files!
    assert upb.main(["--repo", str(s_repo), "--scale", "smoke", "--suite", "all"]) == 0
    run2_shas = {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS}
    assert upb.main(["--repo", str(s_repo), "--scale", "smoke", "--suite", "all"]) == 0
    run3_shas = {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS}
    assert run3_shas == run2_shas


def _make_ref_m5_ablation_row(
    *,
    arm: str,
    prompt_id: str,
    split: str,
    condition: str,
    temperature: float,
    warp: dict,
    seed: int,
    setting: dict,
    tps: float,
    acc_len: float,
    phase_profile: bool = False,
) -> dict:
    import audit_phase_b_full_scale as apb
    import build_and_audit_reports as bar

    blocks = [
        {
            "round_idx": r_i + 1,
            "absolute_output_offset": r_i * 4,
            "input_context_length": 128,
            "draft_tokens": [1, 2, 3, 4],
            "accepted_length": 4 if r_i % 2 == 0 else 3,
            "committed_tokens": [1, 2, 3, 4],
        }
        for r_i in range(6)
    ]
    out_toks = [((i * 7 + seed) % 97) + 1 for i in range(24)]
    phase_ms = {
        ph: (0.45 if phase_profile and ph != "lookahead_iterations" else 0.0)
        for ph in bar.SECTION11_REQUIRED_PHASES
    }
    if phase_profile and arm in apb.LOOKAHEAD_ARMS:
        phase_ms["lookahead_iterations"] = 0.65
    row = {
        "scale": "reference",
        "label": "reference",
        "stage": "m5",
        "arm": arm,
        "prompt_id": prompt_id,
        "split": split,
        "condition": condition,
        "seed": seed,
        "sampling_name": (
            "thinking_recommended"
            if condition.startswith("long_")
            else (
                "greedy"
                if temperature == 0.0
                else ("nonthinking_recommended" if temperature == 0.7 else "pure_temperature_diagnostic")
            )
        ),
        "temperature": temperature,
        "sampling_warp_config": warp,
        "settings": setting,
        "effective_beta": float(setting.get("beta", 1.0)) if setting else 1.0,
        "verification_rule": "ratio" if temperature > 0.0 else "greedy",
        "input_metadata": {"input_context_length": 128},
        "generated_tokens": len(out_toks),
        "elapsed_seconds": len(out_toks) / tps,
        "generation_loop_elapsed_seconds": len(out_toks) / tps,
        "tokens_per_second": tps,
        "mean_accepted_length": acc_len,
        "median_accepted_length": 4.0,
        "target_forwards_per_generated_token": (len(blocks) + 1) / len(out_toks),
        "phase_ms": phase_ms,
        "blocks": blocks,
        "generation": {
            "complete": True,
            "eos_reached": True,
            "capped": False,
            "repetition_dominated": False,
        },
        "output_tokens": out_toks,
        "online": None,
        "online_invariants": None,
        "parameter_norms": {"W": 1.2, "A": 0.7, "b": 0.2} if arm != "plain_dflash" else {},
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "phase_profile": phase_profile,
    }
    if arm in ("online_bigram", "online_bigram_kret256", "online_bigram_kret128", "online_bigram_count_gated"):
        kr = None if arm in ("online_bigram", "online_bigram_count_gated") else (256 if "256" in arm else 128)
        ex_p = 20
        ret_p = 20 if kr is None else (16 if kr == 256 else 12)
        row["bigram"] = {
            "k_ret": kr,
            "count_gate_kappa": 12.0 if arm == "online_bigram_count_gated" else None,
            "ledger_retained_for_exact_counts": True,
            "counted_transitions": 23,
            "exact_ledger_pairs": ex_p,
            "retained_prior_pairs": ret_p,
            "evicted_prior_pairs": ex_p - ret_p,
            "exact_ledger_payload_bytes": ex_p * 12,
            "retained_prior_payload_bytes": ret_p * 12,
        }
    if arm == "dense_candidate":
        row["m5_telemetry"] = {"matched_c0_beta_relax": float(setting.get("beta_relax", 1.0))}
    elif arm in apb.LOOKAHEAD_ARMS:
        row["m5_telemetry"] = {
            "lookahead": {
                "mode": "static" if "static" in arm else "rerelax",
                "domain": "candidate_masked" if arm.endswith("_candidate") else "full_vocabulary",
                "T_relax": float(setting.get("T_relax", 1.0)),
                "beta_relax": float(setting.get("beta_relax", 1.0)),
                "R_init": 2,
            }
        }
    return row


def _build_synthetic_reference_m5_ablation(
    *,
    abl_root: Path,
    ref_art: Path,
    suite_name: str,
    arms: list[str],
    sel_settings: dict[str, dict],
    check_h2h: bool,
    check_bg_mem: bool,
) -> dict:
    import audit_phase_b_full_scale as apb
    import build_and_audit_reports as bar
    from test_phase_b_reporting import _write_json

    ws_dir = abl_root / f"{suite_name}_ws_refsynth"
    stg_dir = ws_dir / "m5"
    stg_dir.mkdir(parents=True, exist_ok=True)
    warp_pure = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "presence_penalty": 0.0}
    warp_nt = {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "presence_penalty": 0.0}
    warp_th = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0}

    dev_specs = [(f"std-dev-{i:02d}", "standard_nonthinking", 0.7, warp_nt) for i in range(8)] + [
        (f"long-dev-{i:02d}", "long_thinking", 0.6, warp_th) for i in range(12)
    ]
    tune_jobs, cands_map = {}, {}
    for a in arms:
        cfg = sel_settings[a]
        tps_a = 105.0 if a != "plain_dflash" else 100.0
        for pid, cond, temp, wp in dev_specs:
            drow = _make_ref_m5_ablation_row(
                arm=a, prompt_id=pid, split="dev", condition=cond,
                temperature=temp, warp=wp, seed=2026, setting=cfg, tps=tps_a, acc_len=4.1,
            )
            dp = stg_dir / "tune" / "rank0" / f"{pid}-{a}.json"
            dsha = _write_json(dp, drow)
            tune_jobs[f"{pid}::{a}"] = {
                "status": "complete",
                "identity": {"timing_session": f"sess_tune_{suite_name}"},
                "result_path": str(dp),
                "result_sha256": dsha,
            }
        cands_map[a] = [{"settings": cfg, "selection_score": tps_a / 100.0, "repetition_excluded_matched_pairs": 0}]
    _write_json(stg_dir / "tune" / "manifest-rank0.json", {"jobs": tune_jobs})

    settings_payload = {
        "frozen": True,
        "stage": "m5",
        "scale": "reference",
        "sharding_mode": "sharded_2gpu",
        "comparison_session": f"sess_tune_{suite_name}",
        "development_prompt_ids": [p for p, _, _, _ in dev_specs],
        "selected": sel_settings,
        "candidates": cands_map,
    }
    settings_p = stg_dir / "settings.json"
    _write_json(settings_p, settings_payload)
    frozen_hash = bar.fingerprint(settings_payload)

    test_specs = []
    for i in range(64):
        pid = f"std-test-{i:03d}"
        test_specs.append((pid, "standard_nonthinking", 0.0, warp_pure, 2026))
        for s_off in range(3):
            test_specs.append((pid, "standard_nonthinking", 0.7, warp_nt, 2026 + s_off))
            test_specs.append((pid, "standard_nonthinking", 1.0, warp_pure, 2026 + s_off))
    for i in range(32):
        test_specs.append((f"long-test-{i:03d}", "long_thinking", 0.6, warp_th, 2026))

    eval_jobs, eval_rows = {}, []
    for pid, cond, temp, wp, sd in test_specs:
        for a in arms:
            erow = _make_ref_m5_ablation_row(
                arm=a, prompt_id=pid, split="test", condition=cond,
                temperature=temp, warp=wp, seed=sd, setting=sel_settings[a],
                tps=112.0 + (4.0 if a != "plain_dflash" else 0.0),
                acc_len=4.0 + (0.25 if a != "plain_dflash" else 0.0),
            )
            rp = stg_dir / "evaluate" / "rank0" / f"{pid}-{temp}-{sd}-{a}.json"
            rsha = _write_json(rp, erow)
            eval_jobs[f"{pid}::{temp}::{sd}::{a}"] = {
                "status": "complete",
                "identity": {"frozen_settings_hash": frozen_hash},
                "result_path": str(rp),
                "result_sha256": rsha,
            }
            eval_rows.append(erow)
    _write_json(stg_dir / "evaluate" / "manifest-rank0.json", {"jobs": eval_jobs})

    timed_rows, prof_rows = [], []
    timing_jobs_by_rk = {0: {}, 1: {}}
    sess_paths = []
    rank_pids = {0: ["std-test-000", "long-test-000"], 1: ["std-test-001", "long-test-001"]}
    _write_json(
        stg_dir / "timing" / "sharding_plan.json",
        {"world_size": 2, "ranks": [{"rank": rk, "prompt_ids": pids} for rk, pids in rank_pids.items()]},
    )
    for rk, pids in rank_pids.items():
        bundles = {}
        for pid in pids:
            cond_t = "long_thinking" if pid.startswith("long-") else "standard_nonthinking"
            temp_t = 0.6 if cond_t.startswith("long_") else 0.7
            warp_t = warp_th if cond_t.startswith("long_") else warp_nt
            for b_idx, is_prof in ((0, False), (1, False), (2, True)):
                rpaths, rshas = [], {}
                for a in arms:
                    trow = _make_ref_m5_ablation_row(
                        arm=a, prompt_id=pid, split="test", condition=cond_t,
                        temperature=temp_t, warp=warp_t, seed=2026, setting=sel_settings[a],
                        tps=115.0 + (3.5 if a != "plain_dflash" else 0.0),
                        acc_len=4.15 if a != "plain_dflash" else 4.0,
                        phase_profile=is_prof,
                    )
                    trow["repeat"] = b_idx
                    tp = stg_dir / "timing" / f"rank{rk}" / f"{pid}-b{b_idx}-{a}.json"
                    tsha = _write_json(tp, trow)
                    rpaths.append(str(tp))
                    rshas[str(tp)] = tsha
                    timing_jobs_by_rk[rk][f"{pid}::b{b_idx}::{a}"] = {
                        "status": "complete",
                        "result_path": str(tp),
                        "result_sha256": tsha,
                    }
                    (prof_rows if is_prof else timed_rows).append(trow)
                bundles[f"{pid}::b{b_idx}"] = {"status": "complete", "result_paths": rpaths, "result_sha256": rshas}
        _write_json(stg_dir / "timing" / f"manifest-rank{rk}.json", {"jobs": timing_jobs_by_rk[rk]})
        sp = stg_dir / "timing" / "sessions" / f"sess_tm_{suite_name}-rank{rk}.json"
        _write_json(
            sp,
            {"rank": rk, "physical_gpu": rk, "numa_node": 0, "cpu_affinity": list(range(rk * 16, (rk + 1) * 16)), "bundles": bundles},
        )
        sess_paths.append(str(sp))

    src_repo = IWT if IWT.is_dir() else REPO
    helper = bar.PostGateEAuditor(repo=src_repo, artifacts=ws_dir, scale="reference", seed=2026)
    enrich = apb.PhaseBFullScaleAuditor(
        repo=src_repo, phase_b_root=abl_root.parent, reference_artifacts=ref_art, scale="reference", seed=2026
    )
    eval_groups = helper.recompute_group_rows(eval_rows, isolated_timing=False, exclude_repetition=True)
    timed_groups = enrich._enrich_timed_groups_with_paired_plain(
        label="synth",
        timed_rows=timed_rows,
        recomp_t_groups=helper.recompute_group_rows(timed_rows, isolated_timing=True, exclude_repetition=True),
    )
    prof_groups = helper.recompute_group_rows(prof_rows, isolated_timing=True, exclude_repetition=True)
    eval_slices = apb.recompute_offset_bin_slices(eval_rows, isolated_timing=False, seed=2026)
    timed_slices = apb.recompute_offset_bin_slices(timed_rows, isolated_timing=True, seed=2026)
    msc = apb.recompute_phase_b_matched_support_comparisons(eval_rows, timed_groups, seed=2026)
    bg_mem = apb.recompute_bigram_memory_ablation(eval_rows) if check_bg_mem else []
    h2h = apb.recompute_phase_b_head_to_head_offset_comparison(eval_rows, timed_rows, seed=2026) if check_h2h else []

    _write_json(
        stg_dir / "timing_isolation_calibration.json",
        {
            "status": "complete",
            "calibrated_timed_runs": 6 * 2 * len(arms),
            "reconciled_with_sharded_timing": True,
            "groups": [{"arm": "plain_dflash", "isolated_vs_sharded_ratio": 1.002}],
            "by_prompt": [{"prompt_id": "std-test-000", "arm": "plain_dflash"}],
        },
    )
    timing_p = stg_dir / "timing.json"
    _write_json(
        timing_p,
        {
            "scale": "reference",
            "stage": "m5",
            "status": "complete",
            "world_size": 2,
            "sharding_mode": "sharded_2gpu",
            "timed_requests": len(timed_rows),
            "groups": timed_groups,
            "phase_profiles": prof_groups,
            "raw_evidence": {
                "world_size": 2,
                "sharding_plan_path": str(stg_dir / "timing" / "sharding_plan.json"),
                "session_paths": sess_paths,
            },
        },
    )
    summary_p = stg_dir / "summary.json"
    sum_payload: dict = {
        "scale": "reference",
        "stage": "m5",
        "status": "complete",
        "isolated_timing_complete": True,
        "expected_requests": len(eval_rows),
        "completed_requests": len(eval_rows),
        "missing_requests": [],
        "groups": eval_groups,
    }
    if check_bg_mem:
        sum_payload["bigram_memory_ablation"] = bg_mem
    _write_json(summary_p, sum_payload)

    rep_payload: dict = {
        "suite": suite_name,
        "scale": "reference",
        "status": "complete",
        "config_fingerprint": f"fp_{suite_name}_ref",
        "arms": arms,
        "selected_settings": sel_settings,
        "artifacts": {"settings_json": str(settings_p), "summary_json": str(summary_p), "timing_json": str(timing_p)},
        "overall_evaluation_groups": eval_groups,
        "isolated_timing_groups": timed_groups,
        "evaluation_offset_bin_slices": eval_slices,
        "isolated_timing_offset_bin_slices": timed_slices,
        "matched_support_comparisons": msc,
    }
    if check_bg_mem:
        rep_payload["bigram_k_ret"] = 256
        rep_payload["bigram_count_gate_kappa"] = 12.0
        rep_payload["bigram_memory_ablation"] = bg_mem
    if check_h2h:
        rep_payload["head_to_head_online_offset_comparison"] = h2h
    _write_json(abl_root / f"{suite_name}.json", rep_payload)
    return rep_payload


def test_update_docs_phase_b_synthetic_true_full_scale_all_suites(tmp_path: Path) -> None:
    """Verify `update_docs_phase_b.py --scale full --suite all` on a true mixed-scale (`reference` ablations + `2.4M`-tok `full`) tree."""
    s_repo = _create_phase_b_sandbox_repo(tmp_path)
    s_art = s_repo / "artifacts"

    # Build synthetic `scale="full"` M1-M4 workspace
    synth_pb = _build_synthetic_full_scale_workspace(tmp_path / "synth_build", tmp_path / "dummy_ref")

    # Generate `pure_t1_fit` via fast smoke run into `synth_pb` and promote its scale tag to `"reference"`
    phase_b_suite.main(
        [
            "--scale",
            "smoke",
            "--suite",
            "pure_t1_fit",
            "--gpus",
            "0,1",
            "--output",
            str(synth_pb),
            "--reference_artifacts",
            str(s_art),
            "--timing_repeats",
            "2",
            "--timing_warmups",
            "1",
            "--smoke_tokens",
            "16",
            "--block_shuffle_window",
            "2",
        ]
    )
    abl_dir = synth_pb / "reference_secondary_ablations"
    pt1_p = abl_dir / "pure_t1_fit.json"
    pt1 = json.loads(pt1_p.read_text(encoding="utf-8"))
    pt1["scale"] = "reference"
    fit_p = Path(pt1["fit_report_path"])
    fit_d = json.loads(fit_p.read_text(encoding="utf-8"))
    fit_d["scale"] = "reference"
    fit_d.setdefault("trace_index", {}).setdefault("excluded_generated_tokens", {})["long_thinking"] = 1024
    fit_p.write_text(json.dumps(fit_d, indent=2), encoding="utf-8")
    pt1_p.write_text(json.dumps(pt1, indent=2), encoding="utf-8")

    # Build full 2,400-row `bigram_eviction` and 3,360-row `full_vocab_lookahead` reference ablations
    _build_synthetic_reference_m5_ablation(
        abl_root=abl_dir,
        ref_art=s_art,
        suite_name="bigram_eviction",
        arms=["plain_dflash", "online_bigram", "online_bigram_kret256", "online_bigram_kret128", "online_bigram_count_gated"],
        sel_settings={
            "plain_dflash": {},
            "online_bigram": {"alpha": 1.0, "k_ret": None, "lambda_tr": 0.2},
            "online_bigram_kret256": {"alpha": 1.0, "k_ret": 256, "lambda_tr": 0.2},
            "online_bigram_kret128": {"alpha": 1.0, "k_ret": 128, "lambda_tr": 0.2},
            "online_bigram_count_gated": {"alpha": 1.0, "k_ret": None, "lambda_tr": 0.2, "count_gate_kappa": 12.0},
        },
        check_h2h=True,
        check_bg_mem=True,
    )
    _build_synthetic_reference_m5_ablation(
        abl_root=abl_dir,
        ref_art=s_art,
        suite_name="full_vocab_lookahead",
        arms=[
            "plain_dflash",
            "dense_full",
            "lookahead_static_full",
            "lookahead_rerelax_full",
            "dense_candidate",
            "lookahead_static_candidate",
            "lookahead_rerelax_candidate",
        ],
        sel_settings={
            "plain_dflash": {},
            "dense_full": {"beta": 1.0},
            "lookahead_static_full": {"beta": 1.0, "beta_relax": 1.0, "T_relax": 1.0},
            "lookahead_rerelax_full": {"beta": 1.0, "beta_relax": 1.0, "T_relax": 1.0},
            "dense_candidate": {"beta": 1.0, "beta_relax": 1.0},
            "lookahead_static_candidate": {"beta": 1.0, "beta_relax": 1.0, "T_relax": 1.0},
            "lookahead_rerelax_candidate": {"beta": 1.0, "beta_relax": 1.0, "T_relax": 1.0},
        },
        check_h2h=False,
        check_bg_mem=False,
    )

    # Populate unified 4-suite `phase_b_summary.json` matching Step 5 (`--suite all`)
    pb_sum = {
        "status": "complete",
        "scale": "full",
        "executed_suites": ["bigram_eviction", "full_vocab_lookahead", "pure_t1_fit", "full_scale"],
        "upstream_reference_immutability": {"status": "intact", "verified_files": 23},
        "suites": {
            "full_scale": {"config_fingerprint": "fp_full_1234", "scale": "full"},
        },
    }
    for sn in ("bigram_eviction", "full_vocab_lookahead", "pure_t1_fit"):
        s_rep = json.loads((abl_dir / f"{sn}.json").read_text(encoding="utf-8"))
        pb_sum["suites"][sn] = {"config_fingerprint": s_rep["config_fingerprint"], "scale": "reference"}
    (synth_pb / "phase_b_summary.json").write_text(json.dumps(pb_sum), encoding="utf-8")

    os.symlink(synth_pb.resolve(), s_art / "full_scale")

    rc1 = upb.main(["--repo", str(s_repo), "--scale", "full", "--suite", "all"])
    assert rc1 == 0
    shas_1 = {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS}

    rc2 = upb.main(["--repo", str(s_repo), "--scale", "full", "--suite", "all"])
    assert rc2 == 0
    shas_2 = {rel: apm.sha256_file(s_repo / rel) for rel in TARGET_RELPATHS}
    assert shas_2 == shas_1
    _check_markdown_links_and_headings(s_repo)
