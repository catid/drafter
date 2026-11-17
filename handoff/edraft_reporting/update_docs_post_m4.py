#!/usr/bin/env python3
"""Zero-Idle-Gap Phase-A -> Phase-B Documentation & Milestone Updater (`update_docs_post_m4.py`).

Deterministically validates completed `artifacts/post_m4_suite/*.json` via `PostM4SuiteAuditor`
(`--require_complete`, `0` failures required) and atomically + idempotently updates:
  1. `AGENTS.md` (standing 20-minute parallel bug-hunt rotation rule under `# Working instructions`),
  2. `artifacts/milestones.json` (`M5 -> "passed"` with SHA256s + headline metrics, preserving `M0`..`M4` 100% intact),
  3. `reports/SUMMARY.md` (idempotent delimited `POST_M4_M5_SUMMARY` empirical synthesis + publication tables),
  4. `README.md` (idempotent delimited `POST_M4_M5_README` reproduction commands + headline findings),
  5. `NOTES.md` (idempotent delimited `POST_M4_M5_NOTES` operational & SHA256 verification log),
  6. `progress.md` (idempotent delimited `POST_M4_M5_PROGRESS` Phase-A completion handoff record),
  7. `artifacts/implementation_audit.json` (regenerated LAST via `build_implementation_audit` so all updated
     doc & milestone SHA256s are recorded and `status` upgrades to `"reference_m0_m5_complete_verified"`).

After writing (or staging in a temporary directory under `--dry_run`), re-verifies BOTH:
  - `PostM4SuiteAuditor(..., require_complete=True, check_implementation_audit=True)` (`0` failures), and
  - `PostGateEAuditor(..., require_complete=True).audit_repo_conformance()` (`0` failures, `0` stale markers).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any
import uuid

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from audit_post_m4_suite import (  # noqa: E402
    PostM4SuiteAuditor,
    SUITE_NAMES,
    _fmt_num,
    render_post_m4_markdown,
)
from build_and_audit_reports import (  # noqa: E402
    PostGateEAuditor,
    _cond_short,
    _fmt_ci,
    sha256_file,
)
from build_implementation_audit import (  # noqa: E402
    _git_head,
    build_implementation_audit,
    verify_implementation_audit,
)


POST_M4_SUITE_REL_JSONS: dict[str, str] = {
    "m2_refit_calibration": "post_m4_suite/m2_refit_calibration.json",
    "online_head_to_head": "post_m4_suite/online_head_to_head.json",
    "m5_eval": "post_m4_suite/m5_eval.json",
    "fastpath_bench": "post_m4_suite/fastpath_bench.json",
    "post_m4_suite_summary": "post_m4_suite/post_m4_suite_summary.json",
}

EXTERNAL_BASELINES_NOT_RUN: tuple[str, ...] = (
    "full tree-based Token Recycling",
    "Test-Time Speculation",
    "SharpDraft",
)

AGENTS_ROTATION_BULLET = (
    "- **Standing 20-Minute Bug-Hunt Rotation**: While any long-running GPU\n"
    "  training, evaluation, or benchmark job is active, run an independent\n"
    "  multi-angle parallel bug-hunt and artifact/manifest conformance check every\n"
    "  20 minutes across active driver processes, output shards, validators, and\n"
    "  staged next-phase code so defects are caught and fixed before GPU transitions."
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:10]}"
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _upsert_delimited_section(original_text: str, marker_name: str, section_body: str) -> str:
    begin_tag = f"<!-- BEGIN {marker_name} -->"
    end_tag = f"<!-- END {marker_name} -->"
    wrapped = f"{begin_tag}\n{section_body.rstrip()}\n{end_tag}\n"

    begin_idx = original_text.find(begin_tag)
    end_idx = original_text.find(end_tag)
    if begin_idx != -1 and end_idx != -1 and begin_idx < end_idx:
        after_end = end_idx + len(end_tag)
        if after_end < len(original_text) and original_text[after_end] == "\n":
            after_end += 1
        updated = original_text[:begin_idx] + wrapped + original_text[after_end:]
    elif begin_idx != -1 or end_idx != -1:
        raise ValueError(f"Malformed delimiter pair for {marker_name} in target document")
    else:
        updated = original_text.rstrip("\n") + "\n\n" + wrapped

    if updated.count(begin_tag) != 1 or updated.count(end_tag) != 1:
        raise AssertionError(f"Expected exactly 1 delimiter pair for {marker_name}")
    return updated


def update_agents_md_text(original_text: str) -> str:
    """Idempotently insert or update the standing 20-minute bug-hunt rotation bullet in `AGENTS.md`."""
    existing_pat = re.compile(
        r"- \*\*Standing 20-Minute Bug-Hunt Rotation\*\*:.*?(?=\n- \*\*|\n\nAfter each round|\Z)",
        flags=re.DOTALL,
    )
    if existing_pat.search(original_text):
        return existing_pat.sub(AGENTS_ROTATION_BULLET, original_text, count=1)

    anchor_pat = re.compile(
        r"(- \*\*Periodic Parallel Bug Checks\*\*:.*?)(\n\nAfter each round of work:)",
        flags=re.DOTALL,
    )
    m = anchor_pat.search(original_text)
    if m:
        return (
            original_text[: m.start()]
            + m.group(1)
            + "\n"
            + AGENTS_ROTATION_BULLET
            + m.group(2)
            + original_text[m.end() :]
        )
    return original_text.rstrip("\n") + "\n\n" + AGENTS_ROTATION_BULLET + "\n"


def _resolve_completed_utc(
    *,
    explicit_utc: str | None,
    old_milestones: dict[str, Any],
    suite_shas: dict[str, str],
    master_summary_path: Path,
    master_summary_json: dict[str, Any],
) -> str:
    if explicit_utc:
        return explicit_utc
    old_m5 = old_milestones.get("M5", {})
    if (
        isinstance(old_m5, dict)
        and old_m5.get("artifact_sha256") == suite_shas
        and isinstance(old_m5.get("completed_utc"), str)
        and old_m5["completed_utc"]
    ):
        return old_m5["completed_utc"]
    for key in ("completed_utc", "completed_at_utc", "generated_at_utc"):
        val = master_summary_json.get(key)
        if isinstance(val, str) and val:
            return val
    mtime = master_summary_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _extract_headline_metrics(audit_report: dict[str, Any]) -> dict[str, Any]:
    """Extract deterministic, machine-verifiable headline metrics across all 4 suites."""
    suites = audit_report["suites"]

    # 1. m2_refit_calibration
    s1 = suites["m2_refit_calibration"]
    s1_variants = []
    best_sparse_kl_var: dict[str, Any] | None = None
    for v in s1.get("variants", []):
        d_cl = v.get("closed_loop", {}).get("dense", {})
        s_cl = v.get("closed_loop", {}).get("dense_sparse", {})
        entry = {
            "variant": v["variant"],
            "dense_dev_kl": v.get("dense_dev_kl"),
            "dense_test_kl": v.get("dense_test_kl"),
            "sparse_dev_kl": v.get("sparse_dev_kl"),
            "sparse_test_kl": v.get("sparse_test_kl"),
            "closed_loop_dense_accepted_length": d_cl.get("mean_accepted_length"),
            "closed_loop_dense_delta_vs_plain": d_cl.get("paired_accepted_length_delta_vs_plain"),
            "closed_loop_dense_tps_ratio_vs_plain": d_cl.get("paired_tps_ratio_vs_plain"),
            "closed_loop_sparse_accepted_length": s_cl.get("mean_accepted_length"),
            "closed_loop_sparse_delta_vs_plain": s_cl.get("paired_accepted_length_delta_vs_plain"),
            "closed_loop_sparse_tps_ratio_vs_plain": s_cl.get("paired_tps_ratio_vs_plain"),
        }
        s1_variants.append(entry)
        if entry["sparse_test_kl"] is not None and (
            best_sparse_kl_var is None or entry["sparse_test_kl"] < best_sparse_kl_var["sparse_test_kl"]
        ):
            best_sparse_kl_var = entry

    bcal = s1.get("beta_calibration") or {}
    dense_beta_star = (bcal.get("dense") or {}).get(
        "optimal_beta", (bcal.get("dense") or {}).get("selected_beta")
    )
    sparse_beta_star = (bcal.get("dense_sparse") or {}).get(
        "optimal_beta", (bcal.get("dense_sparse") or {}).get("selected_beta")
    )

    # 2. online_head_to_head
    s2 = suites["online_head_to_head"]
    ws2 = s2.get("workspace_audit", {})
    t_map2 = {
        (
            g["condition"],
            float(g["temperature"]),
            json.dumps(g["sampling_warp_config"], sort_keys=True),
            g["arm"],
        ): g
        for g in ws2.get("timing_groups", [])
    }
    s2_groups = []
    for eg in ws2.get("eval_groups", []):
        k = (
            eg["condition"],
            float(eg["temperature"]),
            json.dumps(eg["sampling_warp_config"], sort_keys=True),
            eg["arm"],
        )
        tg = t_map2.get(k, {})
        s2_groups.append(
            {
                "condition": eg["condition"],
                "temperature": eg["temperature"],
                "arm": eg["arm"],
                "primary_requests": eg["primary_requests"],
                "mean_accepted_length": (eg.get("metrics", {}).get("mean_accepted_length") or {}).get("mean"),
                "sharded_tokens_per_second": (tg.get("metrics", {}).get("tokens_per_second") or {}).get("mean"),
                "paired_tps_ratio_vs_plain": (tg.get("paired_throughput_ratio") or {}).get("mean"),
                "throughput_gain_supported_by_95_ci": tg.get("throughput_gain_supported_by_95_ci"),
            }
        )

    # 3. m5_eval
    s3 = suites["m5_eval"]
    ws3 = s3.get("workspace_audit", {})
    off3 = s3.get("offline_fits", {})
    ex3 = off3.get("exact_observed_pairs", {})
    pl3 = off3.get("two_sided_pseudolikelihood", {})
    sh3 = off3.get("baseline_signed_hash", {})
    t_map3 = {
        (
            g["condition"],
            float(g["temperature"]),
            json.dumps(g["sampling_warp_config"], sort_keys=True),
            g["arm"],
        ): g
        for g in ws3.get("timing_groups", [])
    }
    p_map3 = {
        (
            g["condition"],
            float(g["temperature"]),
            json.dumps(g["sampling_warp_config"], sort_keys=True),
            g["arm"],
        ): g
        for g in ws3.get("phase_profiles", [])
    }
    s3_groups = []
    for eg in ws3.get("eval_groups", []):
        k = (
            eg["condition"],
            float(eg["temperature"]),
            json.dumps(eg["sampling_warp_config"], sort_keys=True),
            eg["arm"],
        )
        tg = t_map3.get(k, {})
        pg = p_map3.get(k, {})
        s3_groups.append(
            {
                "condition": eg["condition"],
                "temperature": eg["temperature"],
                "arm": eg["arm"],
                "primary_requests": eg["primary_requests"],
                "mean_accepted_length": (eg.get("metrics", {}).get("mean_accepted_length") or {}).get("mean"),
                "sharded_tokens_per_second": (tg.get("metrics", {}).get("tokens_per_second") or {}).get("mean"),
                "paired_tps_ratio_vs_plain": (tg.get("paired_throughput_ratio") or {}).get("mean"),
                "throughput_gain_supported_by_95_ci": tg.get("throughput_gain_supported_by_95_ci"),
                "lookahead_iterations_median_ms": (pg.get("phase_ms_median") or {}).get(
                    "lookahead_iterations", 0.0
                ),
            }
        )

    # 4. fastpath_bench
    s4 = suites["fastpath_bench"]
    cst = s4.get("cluster_sharded_throughput") or {}
    s4_speedups = [
        {
            "arm": ps["arm"],
            "pairs": ps["pairs"],
            "round3_over_eager_mean_ratio": ps.get("round3_over_eager_mean_ratio"),
            "compile_over_round3_mean_ratio": ps.get("compile_over_round3_mean_ratio"),
            "compile_over_eager_mean_ratio": ps.get("compile_over_eager_mean_ratio"),
        }
        for ps in s4.get("closed_loop_paired_speedups", [])
    ]

    any_tps_ci_gain = any(
        bool(g.get("throughput_gain_supported_by_95_ci"))
        for g in (s2_groups + s3_groups)
        if g["arm"] != "plain_dflash"
    )

    return {
        "m2_refit_calibration": {
            "dense_beta_star": dense_beta_star,
            "dense_sparse_beta_star": sparse_beta_star,
            "best_sparse_test_kl_variant": best_sparse_kl_var["variant"] if best_sparse_kl_var else None,
            "best_sparse_test_kl": best_sparse_kl_var["sparse_test_kl"] if best_sparse_kl_var else None,
            "variants": s1_variants,
        },
        "online_head_to_head": {
            "timing_world_size": s2.get("timing_world_size"),
            "sharding_mode": s2.get("sharding_mode"),
            "bigram_count_gate_kappa": s2.get("bigram_count_gate_kappa"),
            "sparse_long_offset_bin_count": len(ws2.get("sparse_long_offset_bins", [])),
            "groups": s2_groups,
        },
        "m5_eval": {
            "timing_world_size": s3.get("timing_world_size"),
            "sharding_mode": s3.get("sharding_mode"),
            "matched_c0_beta_relax": ws3.get("matched_c0_beta_relax"),
            "baseline_signed_hash_test_kl": (sh3.get("test") or {}).get("weighted_mean_kl"),
            "two_sided_pseudolikelihood_test_kl": (
                (pl3.get("held_out_union_rest_grouped_kl") or {}).get("test") or {}
            ).get("weighted_mean_kl"),
            "exact_observed_pairs_collision_count": ex3.get("collision_count"),
            "exact_observed_pairs_test_kl": (
                (ex3.get("held_out_union_rest_grouped_kl") or {}).get("test") or {}
            ).get("weighted_mean_kl"),
            "exact_vs_signed_hash_test_kl_delta": ex3.get("test_grouped_kl_delta_vs_signed_hash"),
            "sparse_long_offset_bin_count": len(ws3.get("sparse_long_offset_bins", [])),
            "groups": s3_groups,
        },
        "fastpath_bench": {
            "device": s4.get("device"),
            "gpus": s4.get("gpus"),
            "paired_speedups": s4_speedups,
            "cluster_world_size": cst.get("world_size"),
            "aggregate_cluster_tokens_per_second": cst.get("aggregate_cluster_tokens_per_second"),
            "cluster_by_arm": cst.get("by_arm", []),
        },
        "net_closed_loop_tps_gain_over_plain_95ci": any_tps_ci_gain,
    }


def _build_m5_result_sentence(hm: dict[str, Any], scale: str) -> str:
    s1 = hm["m2_refit_calibration"]
    s3 = hm["m5_eval"]
    s4 = hm["fastpath_bench"]
    best_v = s1.get("best_sparse_test_kl_variant") or "calibrated_refit"
    best_kl = _fmt_num(s1.get("best_sparse_test_kl"), 4)
    ex_coll = s3.get("exact_observed_pairs_collision_count")
    ex_delta = _fmt_num(s3.get("exact_vs_signed_hash_test_kl_delta"), 6)
    c0_br = s3.get("matched_c0_beta_relax")
    tps_stmt = (
        "at least one non-baseline arm achieved a 95%-CI-supported net tokens/sec gain over plain_dflash"
        if hm["net_closed_loop_tps_gain_over_plain_95ci"]
        else "no adapter/lookahead arm achieved a 95%-CI-supported net closed-loop tokens/sec gain over plain_dflash after proposal/update costs"
    )
    cl_tps = _fmt_num(s4.get("aggregate_cluster_tokens_per_second"), 2)
    return (
        f"post_m4_suite ({scale}) passed all four suites: block-shuffled + weight-decayed M2 refit & calibration "
        f"resolved M2 norm overshoot (best held-out test grouped KL {best_kl} via {best_v}); ExactObservedPairs "
        f"verified collision_count={ex_coll} (test KL delta {ex_delta} vs SignedHash); matched-C0[j] lookahead "
        f"(beta_relax={c0_br}), two-sided pseudolikelihood, buffered stride-5, and zero-init/count-gated online "
        f"ablations completed ({tps_stmt}); fastpath_bench measured Round-3 compact/compile speedups and "
        f"{s4.get('cluster_world_size')}-GPU core-pinned cluster throughput ({cl_tps} tok/s); <8-prompt long "
        f"offset bins remain marked SPARSE (conclusion unavailable), and external baselines "
        f"(full tree-based Token Recycling, Test-Time Speculation, SharpDraft) were not run"
    )


def build_updated_milestones(
    old_milestones: dict[str, Any],
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> dict[str, Any]:
    new_ms = copy.deepcopy(old_milestones)
    new_ms["M5"] = {
        "status": "passed",
        "optional": True,
        "current_phase": "complete",
        "scale": scale,
        "completed_utc": completed_utc,
        "project_git_commit": git_commit_sha,
        "suites_executed": list(SUITE_NAMES),
        "evidence": [f"artifacts/{rel}" for rel in POST_M4_SUITE_REL_JSONS.values()],
        "artifact_sha256": dict(suite_shas),
        "validation": {
            "status": audit_report["status"],
            "checks": audit_report["checks"],
            "audited_files": audit_report["audited_file_count"],
            "validator": "audit_post_m4_suite.py",
        },
        "headline_metrics": headline_metrics,
        "external_baselines_not_run": list(EXTERNAL_BASELINES_NOT_RUN),
        "result": _build_m5_result_sentence(headline_metrics, scale),
    }
    for k in old_milestones:
        if k != "M5":
            assert new_ms[k] == old_milestones[k], f"Upstream milestone {k} mutated!"
    return new_ms


def _demote_rendered_markdown_headings(md: str) -> str:
    """Shift `#` -> `###` and `##` -> `###` / `####` so rendered tables nest cleanly inside `reports/SUMMARY.md`."""
    out_lines = []
    for line in md.splitlines():
        if line.startswith("# Deterministically Recomputed"):
            continue
        if line.startswith("## "):
            out_lines.append("### " + line[3:])
        elif line.startswith("### "):
            out_lines.append("#### " + line[4:])
        else:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def render_summary_post_m4_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> str:
    s1 = headline_metrics["m2_refit_calibration"]
    s2 = headline_metrics["online_head_to_head"]
    s3 = headline_metrics["m5_eval"]
    s4 = headline_metrics["fastpath_bench"]

    tps_verdict = (
        "At least one evaluated arm achieved a 95%-CI-supported net `tokens/sec` gain over `plain_dflash`."
        if headline_metrics["net_closed_loop_tps_gain_over_plain_95ci"]
        else "Across closed-loop M5 and online head-to-head suites, **no adapter or lookahead arm established a 95%-CI-supported net end-to-end `tokens/sec` speedup over `plain_dflash`** once proposal, lookahead, or online update latency is included."
    )

    speedup_bits = ", ".join(
        f"`{x['arm']}` `{_fmt_num(x['round3_over_eager_mean_ratio'], 2)}x` compact / `{_fmt_num(x['compile_over_eager_mean_ratio'], 2)}x` compile"
        for x in s4.get("paired_speedups", [])
    )

    nested_tables = _demote_rendered_markdown_headings(render_post_m4_markdown(audit_report))

    lines = [
        "---",
        "",
        f"## Post-M4 Ablations, Fastpath Benchmarks & Optional M5 Extensions (`scale={scale}`)",
        "",
        f"Completed at `{completed_utc}` on commit `{git_commit_sha}` via `edraft.post_m4_suite` without modifying any frozen M0–M4 reference artifact (`{audit_report['upstream_m0_m4_immutability']['checked_upstream_files']}` upstream M0–M4 files SHA256-verified intact; `{audit_report['checks']}` `audit_post_m4_suite.py` checks across `{audit_report['audited_file_count']}` files with `0` failures).",
        "",
        "**Headline Post-M4 & M5 empirical findings:**",
        f"- **M2 offline refit & calibration (`m2_refit_calibration`):** block-shuffled (`window=32`) + weight-decayed (`1e-2`) refitting and dev `beta*` calibration (`dense beta*={s1['dense_beta_star']}`, `dense_sparse beta*={s1['dense_sparse_beta_star']}`) eliminated the undecayed M2 norm overshoot, reaching best held-out test grouped KL **`{_fmt_num(s1['best_sparse_test_kl'], 6)}`** (`{s1['best_sparse_test_kl_variant']}`).",
        f"- **Zero-init online & count-gated bigram head-to-head (`online_head_to_head`):** evaluated `plain_dflash`, `online_bigram`, `online_bigram_count_gated` (`kappa={s2['bigram_count_gate_kappa']}`), `online_dbS_zero_init`, and `zero_offline_online_only` under `{s2['timing_world_size']}-GPU` NUMA/core-pinned sharded timing (`{s2['sharding_mode']}`). All `{s2['sparse_long_offset_bin_count']}` `< 8`-prompt long-output offset bins are explicitly flagged **SPARSE / EMPTY (`conclusion unavailable`)**.",
        f"- **Optional M5 offline & closed-loop extensions (`m5_eval`):** capacity-matched `ExactObservedPairs` verified **`collision_count = {s3['exact_observed_pairs_collision_count']}`** (held-out test grouped KL `{_fmt_num(s3['exact_observed_pairs_test_kl'], 6)}`, `Δ = {_fmt_num(s3['exact_vs_signed_hash_test_kl_delta'], 6)}` vs `SignedHash` `{_fmt_num(s3['baseline_signed_hash_test_kl'], 6)}`); `two_sided_pseudolikelihood` reached test grouped KL `{_fmt_num(s3['two_sided_pseudolikelihood_test_kl'], 6)}`. Closed-loop candidate-masked lookahead (`lookahead_static_candidate`, `lookahead_rerelax_candidate`) shared exact frozen `C0[j]` (`beta_relax={s3['matched_c0_beta_relax']}`) with causal `dense_candidate` and recorded positive `lookahead_iterations` CUDA-event durations on every active profile row; `online_buffered_stride5` accumulated all intermediate rounds (`acc + pend == blocks`) against single-round `offline_online_stride5` and `offline_online_stride1`.",
        f"- **Systems fastpath & multi-GPU cluster scaling (`fastpath_bench`):** matched closed-loop speedups over `eager_pre_round3`: {speedup_bits or 'recorded in table below'}. `{s4.get('cluster_world_size')}-GPU` NUMA/16-core-pinned sharded execution reconciled against 1-GPU isolated calibration delivered **`{_fmt_num(s4.get('aggregate_cluster_tokens_per_second'), 2)} tok/s`** aggregate cluster throughput.",
        f"- **Net closed-loop speed & unrun external baselines:** {tps_verdict} **External baselines still not run:** {', '.join(EXTERNAL_BASELINES_NOT_RUN)} (`online_bigram` and `online_bigram_count_gated` remain smoothed committed-transition priors, not tree-based Token Recycling).",
        "",
        "**Checksummed `post_m4_suite` artifact manifest:**",
        f"- `artifacts/post_m4_suite/m2_refit_calibration.json`: `{suite_shas['m2_refit_calibration']}`",
        f"- `artifacts/post_m4_suite/online_head_to_head.json`: `{suite_shas['online_head_to_head']}`",
        f"- `artifacts/post_m4_suite/m5_eval.json`: `{suite_shas['m5_eval']}`",
        f"- `artifacts/post_m4_suite/fastpath_bench.json`: `{suite_shas['fastpath_bench']}`",
        f"- `artifacts/post_m4_suite/post_m4_suite_summary.json`: `{suite_shas['post_m4_suite_summary']}`",
        "",
        nested_tables,
    ]
    return "\n".join(lines)


def render_readme_post_m4_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> str:
    s1 = headline_metrics["m2_refit_calibration"]
    s2 = headline_metrics["online_head_to_head"]
    s3 = headline_metrics["m5_eval"]
    s4 = headline_metrics["fastpath_bench"]

    tps_clause = (
        "see [reports/SUMMARY.md](reports/SUMMARY.md) for per-condition 95% CIs showing where net throughput improved"
        if headline_metrics["net_closed_loop_tps_gain_over_plain_95ci"]
        else "no adapter or lookahead arm established a 95%-CI-supported net closed-loop `tokens/sec` speedup over `plain_dflash`"
    )

    lines = [
        "## Post-M4 ablations, fastpath benchmarks, and optional M5 execution",
        "",
        f"Optional milestone **M5** and the four isolated post-M4 suites (`m2_refit_calibration`, `online_head_to_head`, `m5_eval`, and `fastpath_bench`) have **passed** at `--scale {scale}` (`{completed_utc}`, commit `{git_commit_sha}`, `{audit_report['checks']}` independent `audit_post_m4_suite.py` checks with zero failures) while keeping all upstream M0–M4 reference artifacts immutable:",
        "",
        "```bash",
        "# CPU synthetic smoke check across all four post-M4 / M5 suites (no downloads):",
        "uv run --frozen python -m edraft.post_m4_suite --scale smoke --suite all --gpus 0,1 --output artifacts/smoke_post_m4",
        "",
        "# Full 8-GPU NUMA/16-core-pinned reference execution (parent process initializes 0 CUDA contexts):",
        "uv run --frozen python -m edraft.post_m4_suite --scale reference --suite all --gpus 0,1,2,3,4,5,6,7 --output artifacts",
        "```",
        "",
        f"- **Calibrated M2 refit (`m2_refit_calibration`):** block-shuffled (`window=32`), weight-decayed (`1e-2`) offline training with dev `beta*` calibration (`dense beta*={s1['dense_beta_star']}`, `dense_sparse beta*={s1['dense_sparse_beta_star']}`) resolved the undecayed M2 overshoot, achieving **`{_fmt_num(s1['best_sparse_test_kl'], 4)}`** held-out test grouped KL (`{s1['best_sparse_test_kl_variant']}`).",
        f"- **Online zero-init & count-gated bigram (`online_head_to_head`):** compared `zero_offline_online_only`, `online_dbS_zero_init`, `online_bigram`, and `online_bigram_count_gated` (`kappa={s2['bigram_count_gate_kappa']}`) against `plain_dflash` across output-offset bins (`0-1K`..`16K+`, with `< 8`-prompt bins marked `SPARSE` / `conclusion unavailable`).",
        f"- **Optional M5 extensions (`m5_eval`):** verified collision-free `ExactObservedPairs` (`collision_count = {s3['exact_observed_pairs_collision_count']}`, `Δ = {_fmt_num(s3['exact_vs_signed_hash_test_kl_delta'], 6)}` test grouped KL vs `SignedHash`), `two_sided_pseudolikelihood` (test grouped KL `{_fmt_num(s3['two_sided_pseudolikelihood_test_kl'], 6)}`), matched-`C0[j]` closed-loop lookahead (`beta_relax={s3['matched_c0_beta_relax']}`, positive `T_relax`, profiled `lookahead_iterations > 0 ms`), and multi-round `online_buffered_stride5` vs single-round `offline_online_stride5`; {tps_clause}.",
        f"- **Fastpath & cluster scaling (`fastpath_bench`):** measured Round-3 compact/compiled proposal & online-update speedups and `{s4.get('cluster_world_size')}-GPU` NUMA/core-pinned sharded scaling reconciled against 1-GPU isolated calibration (**`{_fmt_num(s4.get('aggregate_cluster_tokens_per_second'), 2)} tok/s`** aggregate cluster throughput). Full tree-based Token Recycling, Test-Time Speculation, and SharpDraft remain unrun external baselines. Full tables are in [SUMMARY](reports/SUMMARY.md) and [milestones.json](artifacts/milestones.json).",
    ]
    return "\n".join(lines)


def render_notes_post_m4_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> str:
    s1 = headline_metrics["m2_refit_calibration"]
    s2 = headline_metrics["online_head_to_head"]
    s3 = headline_metrics["m5_eval"]
    s4 = headline_metrics["fastpath_bench"]
    return "\n".join(
        [
            f"Post-M4 & Optional M5 suite completion and audit, {completed_utc} (scale={scale}, commit={git_commit_sha}):",
            f"- Independent validator `audit_post_m4_suite.py --scale {scale} --require_complete --check_implementation_audit` passed `{audit_report['checks']}` checks across `{audit_report['audited_file_count']}` files (`0` failures; `{audit_report['upstream_m0_m4_immutability']['checked_upstream_files']}` upstream M0–M4 reference files verified byte-for-byte intact).",
            f"- Artifact SHA256s: `m2_refit_calibration.json={suite_shas['m2_refit_calibration']}`, `online_head_to_head.json={suite_shas['online_head_to_head']}`, `m5_eval.json={suite_shas['m5_eval']}`, `fastpath_bench.json={suite_shas['fastpath_bench']}`, `post_m4_suite_summary.json={suite_shas['post_m4_suite_summary']}`.",
            f"- Headline empirical summary: `m2_refit_calibration` best test grouped KL `{_fmt_num(s1['best_sparse_test_kl'], 6)}` (`{s1['best_sparse_test_kl_variant']}`, `dense beta*={s1['dense_beta_star']}`, `dense_sparse beta*={s1['dense_sparse_beta_star']}`); `online_head_to_head` (`world_size={s2['timing_world_size']}`, `kappa={s2['bigram_count_gate_kappa']}`) flagged `{s2['sparse_long_offset_bin_count']}` `<8`-prompt long offset bins as sparse/empty; `m5_eval` (`world_size={s3['timing_world_size']}`, `matched_c0_beta_relax={s3['matched_c0_beta_relax']}`) verified `ExactObservedPairs` `collision_count={s3['exact_observed_pairs_collision_count']}` (`Δ={_fmt_num(s3['exact_vs_signed_hash_test_kl_delta'], 6)}` vs `SignedHash`) and `two_sided_pseudolikelihood` test KL `{_fmt_num(s3['two_sided_pseudolikelihood_test_kl'], 6)}`; `fastpath_bench` (`gpus={s4['gpus']}`) recorded `{_fmt_num(s4.get('aggregate_cluster_tokens_per_second'), 2)} tok/s` aggregate `{s4.get('cluster_world_size')}-GPU` core-pinned throughput reconciled against 1-GPU isolated calibration. External baselines not run: {', '.join(EXTERNAL_BASELINES_NOT_RUN)}.",
        ]
    )


def render_progress_post_m4_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    audit_report: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "---",
            "",
            f"## Phase-A Post-M4 & Optional M5 Completion Update (`{completed_utc}` — `scale={scale}`)",
            "",
            f"1. **All Four `post_m4_suite` Suites Completed (`commit {git_commit_sha}`):**",
            f"   - `artifacts/post_m4_suite/m2_refit_calibration.json` (`sha256: {suite_shas['m2_refit_calibration']}`)",
            f"   - `artifacts/post_m4_suite/online_head_to_head.json` (`sha256: {suite_shas['online_head_to_head']}`)",
            f"   - `artifacts/post_m4_suite/m5_eval.json` (`sha256: {suite_shas['m5_eval']}`)",
            f"   - `artifacts/post_m4_suite/fastpath_bench.json` (`sha256: {suite_shas['fastpath_bench']}`)",
            f"   - `artifacts/post_m4_suite/post_m4_suite_summary.json` (`sha256: {suite_shas['post_m4_suite_summary']}`)",
            f"2. **Zero-Drift Independent Verification Passed (`{audit_report['checks']}` Checks, `0` Failures):**",
            f"   - `audit_post_m4_suite.py --scale {scale} --require_complete --check_implementation_audit` verified all raw shards, session manifests, matched `C0[j]` lookahead invariants, `ExactObservedPairs` `collision_count == 0`, 1-GPU vs sharded calibration ratios, `<8`-prompt sparse-bin disclosures, and `{audit_report['upstream_m0_m4_immutability']['checked_upstream_files']}` frozen M0–M4 reference artifacts.",
            "3. **Milestones, Implementation Audit & Documentation Synchronized:**",
            "   - `artifacts/milestones.json` (`M5.status = \"passed\"`), `artifacts/implementation_audit.json` (`status = \"reference_m0_m5_complete_verified\"`, `optional_m5 = \"implemented_verified\"`), `reports/SUMMARY.md`, `README.md`, `NOTES.md`, and `AGENTS.md` updated atomically and re-verified.",
        ]
    )


def _stage_dry_run_sandbox(repo: Path, artifacts: Path, dst_root: Path) -> tuple[Path, Path]:
    """Create a lightweight symlinked sandbox of `repo` and `artifacts` for `--dry_run` full-gate verification."""
    s_repo = dst_root / "repo"
    s_repo.mkdir(parents=True, exist_ok=True)
    for item in repo.iterdir():
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

    s_art = s_repo / "artifacts" if artifacts.resolve() == (repo / "artifacts").resolve() else (dst_root / "artifacts")
    s_art.mkdir(parents=True, exist_ok=True)
    if (s_repo / "artifacts") != s_art and not (s_repo / "artifacts").exists():
        os.symlink(s_art, s_repo / "artifacts")

    for a_item in artifacts.iterdir():
        if a_item.name in ("milestones.json", "implementation_audit.json") and a_item.is_file():
            shutil.copy2(a_item, s_art / a_item.name)
        else:
            os.symlink(a_item.resolve(), s_art / a_item.name)
    for fallback_name in ("milestones.json", "implementation_audit.json"):
        if not (s_art / fallback_name).exists() and (repo / "artifacts" / fallback_name).is_file():
            shutil.copy2(repo / "artifacts" / fallback_name, s_art / fallback_name)
    return s_repo, s_art


def run_update_docs_post_m4(
    *,
    repo: Path,
    artifacts: Path,
    suite_root: Path | None = None,
    scale: str = "reference",
    source_repo: Path | None = None,
    completed_utc: str | None = None,
    dry_run: bool = False,
    skip_post_gate_e_audit: bool = False,
) -> dict[str, Any]:
    repo = repo.resolve()
    artifacts = artifacts.resolve()
    eff_suite_root = suite_root.resolve() if suite_root is not None else (artifacts / "post_m4_suite")

    # Step 0: Mandatory Pre-Write Gate — verify 100% completion & zero drift across all 4 suites!
    pre_auditor = PostM4SuiteAuditor(
        repo=repo,
        artifacts=artifacts,
        suite_root=eff_suite_root,
        scale=scale,
        allow_partial=False,
        require_complete=True,
        allow_pilot=False,
        check_implementation_audit=False,
    )
    audit_report = pre_auditor.run_all()
    if audit_report["status"] != "passed" or audit_report["failures"]:
        return {
            "status": "failed_pre_gate",
            "checks": audit_report["checks"],
            "failures": audit_report["failures"],
        }

    # Compute exact SHA256s of the 5 suite JSONs
    suite_shas: dict[str, str] = {}
    for key, rel in POST_M4_SUITE_REL_JSONS.items():
        sub_name = Path(rel).name
        fp = eff_suite_root / sub_name
        suite_shas[key] = sha256_file(fp)

    ms_in_path = (
        (artifacts / "milestones.json")
        if (artifacts / "milestones.json").is_file()
        else (repo / "artifacts" / "milestones.json")
    )
    old_milestones = json.loads(ms_in_path.read_text(encoding="utf-8"))
    master_path = eff_suite_root / "post_m4_suite_summary.json"
    master_json = json.loads(master_path.read_text(encoding="utf-8"))

    eff_utc = _resolve_completed_utc(
        explicit_utc=completed_utc,
        old_milestones=old_milestones,
        suite_shas=suite_shas,
        master_summary_path=master_path,
        master_summary_json=master_json,
    )
    git_sha = _git_head(repo)
    if git_sha == "unknown":
        git_sha = str(master_json.get("project_git_commit") or "unknown")

    headline_metrics = _extract_headline_metrics(audit_report)
    new_milestones = build_updated_milestones(
        old_milestones,
        scale=scale,
        completed_utc=eff_utc,
        git_commit_sha=git_sha,
        suite_shas=suite_shas,
        audit_report=audit_report,
        headline_metrics=headline_metrics,
    )

    new_agents_text = update_agents_md_text((repo / "AGENTS.md").read_text(encoding="utf-8"))
    new_summary_text = _upsert_delimited_section(
        (repo / "reports" / "SUMMARY.md").read_text(encoding="utf-8"),
        "POST_M4_M5_SUMMARY",
        render_summary_post_m4_section(
            scale=scale,
            completed_utc=eff_utc,
            git_commit_sha=git_sha,
            suite_shas=suite_shas,
            audit_report=audit_report,
            headline_metrics=headline_metrics,
        ),
    )
    new_readme_text = _upsert_delimited_section(
        (repo / "README.md").read_text(encoding="utf-8"),
        "POST_M4_M5_README",
        render_readme_post_m4_section(
            scale=scale,
            completed_utc=eff_utc,
            git_commit_sha=git_sha,
            audit_report=audit_report,
            headline_metrics=headline_metrics,
        ),
    )
    new_notes_text = _upsert_delimited_section(
        (repo / "NOTES.md").read_text(encoding="utf-8"),
        "POST_M4_M5_NOTES",
        render_notes_post_m4_section(
            scale=scale,
            completed_utc=eff_utc,
            git_commit_sha=git_sha,
            suite_shas=suite_shas,
            audit_report=audit_report,
            headline_metrics=headline_metrics,
        ),
    )
    new_progress_text = _upsert_delimited_section(
        (repo / "progress.md").read_text(encoding="utf-8"),
        "POST_M4_M5_PROGRESS",
        render_progress_post_m4_section(
            scale=scale,
            completed_utc=eff_utc,
            git_commit_sha=git_sha,
            suite_shas=suite_shas,
            audit_report=audit_report,
        ),
    )

    def _apply_and_verify(target_repo: Path, target_art: Path, target_suite_root: Path) -> dict[str, Any]:
        # Write 1..6 BEFORE regenerating implementation_audit.json (Step 7) so all doc SHA256s match!
        ms_target = (
            (target_art / "milestones.json")
            if (target_art / "milestones.json").exists() or target_art != (target_repo / "artifacts")
            else (target_repo / "artifacts" / "milestones.json")
        )
        _atomic_write_text(target_repo / "AGENTS.md", new_agents_text)
        _atomic_write_text(ms_target, json.dumps(new_milestones, indent=2) + "\n")
        if (target_repo / "artifacts" / "milestones.json").resolve() != ms_target.resolve() and (
            target_repo / "artifacts" / "milestones.json"
        ).is_file():
            _atomic_write_text(
                target_repo / "artifacts" / "milestones.json",
                json.dumps(new_milestones, indent=2) + "\n",
            )
        _atomic_write_text(target_repo / "reports" / "SUMMARY.md", new_summary_text)
        _atomic_write_text(target_repo / "README.md", new_readme_text)
        _atomic_write_text(target_repo / "NOTES.md", new_notes_text)
        _atomic_write_text(target_repo / "progress.md", new_progress_text)

        # Step 7: Regenerate & verify implementation_audit.json LAST!
        tpl_p = (
            (target_art / "implementation_audit.json")
            if (target_art / "implementation_audit.json").is_file()
            else (repo / "artifacts" / "implementation_audit.json")
        )
        impl_audit = build_implementation_audit(
            target_repo,
            target_art,
            template_path=tpl_p,
            audited_at_utc=eff_utc,
        )
        ia_target = (
            (target_art / "implementation_audit.json")
            if (target_art / "implementation_audit.json").exists()
            or target_art != (target_repo / "artifacts")
            else (target_repo / "artifacts" / "implementation_audit.json")
        )
        _atomic_write_text(ia_target, json.dumps(impl_audit, indent=2) + "\n")
        if (
            target_repo / "artifacts" / "implementation_audit.json"
        ).resolve() != ia_target.resolve() and (
            target_repo / "artifacts" / "implementation_audit.json"
        ).is_file():
            _atomic_write_text(
                target_repo / "artifacts" / "implementation_audit.json",
                json.dumps(impl_audit, indent=2) + "\n",
            )

        ia_errs = verify_implementation_audit(
            impl_audit,
            target_repo,
            target_art,
            check_git_head=(_git_head(target_repo) != "unknown"),
        )

        # Post-Write Gate A: Verify newly written `milestones.json` & `implementation_audit.json`
        # AND re-verify SHA256 immutability across ALL pre-audited `post_m4_suite` + M0-M4 files without
        # re-bootstrapping untouched raw rows a second time!
        post_aud = PostM4SuiteAuditor(
            repo=target_repo,
            artifacts=target_art,
            suite_root=target_suite_root,
            scale=scale,
            allow_partial=False,
            require_complete=True,
            allow_pilot=False,
            check_implementation_audit=True,
        )
        updated_meta_names = {"milestones.json", "implementation_audit.json"}
        for p_str, exp_sha in pre_auditor.read_sha256.items():
            if Path(p_str).name not in updated_meta_names:
                post_aud.read_sha256[p_str] = exp_sha
        post_aud.audit_upstream_m0_m4_immutability()
        post_aud.audit_implementation_audit_freshness()
        post_aud.verify_immutability_at_end()
        post_rep = {
            "status": "passed" if not post_aud.failures else "failed",
            "checks": pre_auditor.checks + post_aud.checks,
            "failures": list(post_aud.failures),
        }
        if _git_head(target_repo) == "unknown":
            # In a non-git temp dir (`--dry_run`), filter only git-HEAD check if any
            post_rep["failures"] = [
                f for f in post_rep["failures"] if "!= git HEAD" not in f
            ]
            post_rep["status"] = "passed" if not post_rep["failures"] else "failed"

        # Post-Write Gate B: PostGateEAuditor repo & doc conformance check!
        gate_e_failures: list[str] = []
        gate_e_checks = 0
        if not skip_post_gate_e_audit:
            eff_src = source_repo
            if eff_src is None and Path("/tmp/edraft_09a5469_source").is_dir():
                eff_src = Path("/tmp/edraft_09a5469_source")
            ge_aud = PostGateEAuditor(
                repo=target_repo,
                artifacts=target_art,
                scale=scale if scale in ("smoke", "reference", "full") else "reference",
                allow_partial=True,
                require_complete=(scale == "reference"),
                source_repo=eff_src,
            )
            ge_aud.m4_timing_world_size = int(
                (new_milestones.get("M4") or {}).get("timing_world_size", 1) or 1
            )
            ge_aud.audit_repo_conformance()
            gate_e_checks = ge_aud.checks
            gate_e_failures = list(ge_aud.failures)

        written_shas = {
            "AGENTS.md": sha256_file(target_repo / "AGENTS.md"),
            "README.md": sha256_file(target_repo / "README.md"),
            "NOTES.md": sha256_file(target_repo / "NOTES.md"),
            "progress.md": sha256_file(target_repo / "progress.md"),
            "reports/SUMMARY.md": sha256_file(target_repo / "reports" / "SUMMARY.md"),
            "artifacts/milestones.json": sha256_file(ms_target),
            "artifacts/implementation_audit.json": sha256_file(ia_target),
        }
        all_failures = list(ia_errs) + list(post_rep["failures"]) + gate_e_failures
        return {
            "status": "passed" if not all_failures else "failed_post_gate",
            "dry_run": dry_run,
            "scale": scale,
            "completed_utc": eff_utc,
            "project_git_commit": git_sha,
            "m5_status": new_milestones["M5"]["status"],
            "implementation_audit_status": impl_audit["status"],
            "post_m4_audit_checks": post_rep["checks"],
            "post_gate_e_doc_checks": gate_e_checks,
            "failures": all_failures,
            "updated_file_sha256": written_shas,
            "suite_artifact_sha256": suite_shas,
        }

    if dry_run:
        with tempfile.TemporaryDirectory(prefix="edraft_post_m4_docs_dryrun_") as td:
            s_repo, s_art = _stage_dry_run_sandbox(repo, artifacts, Path(td))
            s_suite = (
                eff_suite_root
                if eff_suite_root != (artifacts / "post_m4_suite")
                else (s_art / "post_m4_suite")
            )
            return _apply_and_verify(s_repo, s_art, s_suite)

    return _apply_and_verify(repo, artifacts, eff_suite_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/catid/drafter"))
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--suite_root", type=Path, default=None)
    parser.add_argument("--scale", choices=["smoke", "reference", "full"], default="reference")
    parser.add_argument("--source_repo", type=Path, default=None)
    parser.add_argument("--completed_utc", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_post_gate_e_audit", action="store_true")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    artifacts = args.artifacts.resolve() if args.artifacts is not None else (repo / "artifacts")
    res = run_update_docs_post_m4(
        repo=repo,
        artifacts=artifacts,
        suite_root=args.suite_root,
        scale=args.scale,
        source_repo=args.source_repo,
        completed_utc=args.completed_utc,
        dry_run=args.dry_run,
        skip_post_gate_e_audit=args.skip_post_gate_e_audit,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
