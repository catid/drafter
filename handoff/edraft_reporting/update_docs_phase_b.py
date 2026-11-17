#!/usr/bin/env python3
"""Zero-Idle-Gap Phase-B (`--scale full` + Secondary Reference Ablations) Doc & Milestone Updater.

Deterministically validates completed `artifacts/full_scale/` outputs via `PhaseBFullScaleAuditor`
(`--require_complete`, `0` failures required) while verifying 100% immutability of frozen M0–M4
(and Phase-A `post_m4_suite`) reference artifacts, and atomically + idempotently updates:
  1. `AGENTS.md` (ensuring the standing 20-minute bug-hunt rotation rule is present),
  2. `artifacts/milestones.json` (`phase_b_full_scale -> "passed"` with SHA256s + headline metrics,
     preserving `M0`..`M4` and `M5` 100% byte-for-byte in value),
  3. `reports/SUMMARY.md` (idempotent delimited `PHASE_B_FULL_SCALE_SUMMARY` empirical synthesis + tables),
  4. `README.md` (idempotent delimited `PHASE_B_FULL_SCALE_README` commands + headline findings),
  5. `NOTES.md` (idempotent delimited `PHASE_B_FULL_SCALE_NOTES` operational & SHA256 log),
  6. `progress.md` (idempotent delimited `PHASE_B_FULL_SCALE_PROGRESS` Phase-B completion record),
  7. `artifacts/implementation_audit.json` (regenerated LAST so all updated doc, milestone, M0–M5, and
     Phase-B top-level artifact SHA256s are recorded and verified).

Re-verifies ALL applicable post-write gates before returning `0`:
  - `PhaseBFullScaleAuditor(..., require_complete=True)` (`0` failures),
  - `verify_implementation_audit(...)` (`0` errors),
  - `PostM4SuiteAuditor(..., require_complete=True, check_implementation_audit=True)` when `post_m4_suite` is present (`0` failures), and
  - `PostGateEAuditor(...).audit_repo_conformance()` (`0` failures, `0` stale markers).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from audit_phase_b_full_scale import (  # noqa: E402
    ALL_PHASE_B_SUITES,
    PhaseBFullScaleAuditor,
    render_phase_b_markdown,
)
from audit_post_m4_suite import PostM4SuiteAuditor, _fmt_num  # noqa: E402
from build_and_audit_reports import PostGateEAuditor, sha256_file  # noqa: E402
from build_implementation_audit import (  # noqa: E402
    _git_head,
    build_implementation_audit,
    verify_implementation_audit,
)
from update_docs_post_m4 import (  # noqa: E402
    EXTERNAL_BASELINES_NOT_RUN,
    _atomic_write_text,
    _upsert_delimited_section,
    update_agents_md_text,
)


PHASE_B_SUITE_REL_JSONS: dict[str, str] = {
    "bigram_eviction": "reference_secondary_ablations/bigram_eviction.json",
    "full_vocab_lookahead": "reference_secondary_ablations/full_vocab_lookahead.json",
    "pure_t1_fit": "reference_secondary_ablations/pure_t1_fit.json",
    "full_scale_pipeline": "full_scale_pipeline.json",
    "phase_b_summary": "phase_b_summary.json",
}


def _resolve_phase_b_completed_utc(
    *,
    explicit_utc: str | None,
    old_milestones: dict[str, Any],
    suite_shas: dict[str, str],
    master_summary_path: Path,
    master_summary_json: dict[str, Any],
) -> str:
    if explicit_utc:
        return explicit_utc
    old_pb = old_milestones.get("phase_b_full_scale", {})
    if (
        isinstance(old_pb, dict)
        and old_pb.get("artifact_sha256") == suite_shas
        and isinstance(old_pb.get("completed_utc"), str)
        and old_pb["completed_utc"]
    ):
        return old_pb["completed_utc"]
    for key in ("completed_utc", "completed_at_utc", "generated_at_utc"):
        val = master_summary_json.get(key)
        if isinstance(val, str) and val:
            return val
    if master_summary_path.is_file():
        return datetime.fromtimestamp(master_summary_path.stat().st_mtime, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _demote_phase_b_markdown_headings(md: str) -> str:
    """Shift `#` / `##` / `###` so `render_phase_b_markdown` nests cleanly under `## Phase-B ...` in `SUMMARY.md`."""
    out_lines = []
    for line in md.splitlines():
        if line.startswith("# Phase-B Secondary"):
            continue
        if line.startswith("## "):
            out_lines.append("### " + line[3:])
        elif line.startswith("### "):
            out_lines.append("#### " + line[4:])
        else:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def _extract_phase_b_headline_metrics(audit_report: dict[str, Any]) -> dict[str, Any]:
    suites = audit_report.get("suites", {})

    # 1. bigram_eviction
    bg = suites.get("bigram_eviction", {})
    bg_ws = bg.get("workspace_audit", {})
    bg_mem = bg_ws.get("bigram_memory_ablation", [])

    # 2. full_vocab_lookahead
    la = suites.get("full_vocab_lookahead", {})
    la_ws = la.get("workspace_audit", {})
    la_msc = [
        {
            "matched_support_family": c["matched_support_family"],
            "condition": c["condition"],
            "temperature": c["temperature"],
            "arm": c["arm"],
            "reference_arm": c["reference_arm"],
            "mean_accepted_length_delta": (c.get("mean_accepted_length_delta") or {}).get("mean"),
            "isolated_timing_tps_ratio": (c.get("isolated_timing_tokens_per_second") or {}).get("ratio"),
        }
        for c in la_ws.get("matched_support_comparisons", [])
    ]

    # 3. pure_t1_fit
    pt1 = suites.get("pure_t1_fit", {})
    pt1_vars = []
    best_pt1_on_primary: dict[str, Any] | None = None
    for v in pt1.get("variants", []):
        ds_cl = (v.get("closed_loop") or {}).get("dense_sparse", {})
        item = {
            "variant": v["variant"],
            "primary_dense_test_kl": v.get("primary_dense_test_kl"),
            "primary_sparse_test_kl": v.get("primary_sparse_test_kl"),
            "pure_t1_dense_test_kl": v.get("pure_t1_dense_test_kl"),
            "pure_t1_sparse_test_kl": v.get("pure_t1_sparse_test_kl"),
            "closed_loop_sparse_delta_vs_plain": ds_cl.get("paired_accepted_length_delta_vs_plain"),
            "closed_loop_sparse_tps_ratio_vs_plain": ds_cl.get("paired_tps_ratio_vs_plain"),
        }
        pt1_vars.append(item)
        if (
            v["variant"].startswith("pure_t1_")
            and item["primary_sparse_test_kl"] is not None
            and (
                best_pt1_on_primary is None
                or item["primary_sparse_test_kl"] < best_pt1_on_primary["primary_sparse_test_kl"]
            )
        ):
            best_pt1_on_primary = item

    # 4. full_scale pipeline (M1-pre -> M4)
    fs = suites.get("full_scale", {})
    m1a = fs.get("m1a_audit", {})
    m2_fit = fs.get("m2_offline_fit", {})
    m3_cl = fs.get("m3_closed_loop", {})
    m4_cl = fs.get("m4_closed_loop", {})

    def _summarize_stage_groups(st_cl: dict[str, Any]) -> list[dict[str, Any]]:
        tmap = {
            (
                g["condition"],
                float(g["temperature"]),
                json.dumps(g["sampling_warp_config"], sort_keys=True),
                g["arm"],
            ): g
            for g in st_cl.get("timing_groups", [])
        }
        rows_out = []
        for eg in st_cl.get("eval_groups", []):
            k = (
                eg["condition"],
                float(eg["temperature"]),
                json.dumps(eg["sampling_warp_config"], sort_keys=True),
                eg["arm"],
            )
            tg = tmap.get(k, {})
            rows_out.append(
                {
                    "condition": eg["condition"],
                    "temperature": eg["temperature"],
                    "arm": eg["arm"],
                    "primary_requests": eg["primary_requests"],
                    "mean_accepted_length": (eg.get("metrics", {}).get("mean_accepted_length") or {}).get("mean"),
                    "matched_tokens_per_second": tg.get("matched_arm_tokens_per_second"),
                    "paired_tps_ratio_vs_plain": (tg.get("paired_throughput_ratio") or {}).get("mean"),
                    "throughput_gain_supported_by_95_ci": tg.get("throughput_gain_supported_by_95_ci"),
                }
            )
        return rows_out

    m3_groups = _summarize_stage_groups(m3_cl)
    m4_groups = _summarize_stage_groups(m4_cl)
    any_full_tps_gain = any(
        bool(g.get("throughput_gain_supported_by_95_ci"))
        for g in (m3_groups + m4_groups)
        if g["arm"] != "plain_dflash"
    )

    return {
        "bigram_eviction": {
            "status": bg.get("status"),
            "bigram_k_ret": bg.get("bigram_k_ret"),
            "bigram_count_gate_kappa": bg.get("bigram_count_gate_kappa"),
            "memory_ablation_groups": len(bg_mem),
            "sparse_long_offset_bin_count": len(bg_ws.get("sparse_long_offset_bins", [])),
        },
        "full_vocab_lookahead": {
            "status": la.get("status"),
            "matched_beta_relax": la_ws.get("matched_beta_relax"),
            "matched_support_comparisons": la_msc,
            "sparse_long_offset_bin_count": len(la_ws.get("sparse_long_offset_bins", [])),
        },
        "pure_t1_fit": {
            "status": pt1.get("status"),
            "training_distribution": pt1.get("training_distribution"),
            "ablation_warning": pt1.get("ablation_warning"),
            "best_pure_t1_variant_on_primary": (
                best_pt1_on_primary["variant"] if best_pt1_on_primary else None
            ),
            "best_pure_t1_primary_sparse_test_kl": (
                best_pt1_on_primary["primary_sparse_test_kl"] if best_pt1_on_primary else None
            ),
            "best_pure_t1_own_sparse_test_kl": (
                best_pt1_on_primary["pure_t1_sparse_test_kl"] if best_pt1_on_primary else None
            ),
            "variants": pt1_vars,
        },
        "full_scale": {
            "status": fs.get("status"),
            "primary_train_tokens": m1a.get("primary_train_total"),
            "primary_train_T0_tokens": m1a.get("primary_train_T0"),
            "primary_train_T07_tokens": m1a.get("primary_train_T07"),
            "long_condition": m1a.get("long_condition"),
            "long_dev_generations": m1a.get("long_dev_generations"),
            "long_test_generations": m1a.get("long_test_generations"),
            "m2_dense_test_kl": m2_fit.get("dense_test_kl"),
            "m2_sparse_test_kl": m2_fit.get("dense_sparse_test_kl"),
            "m3_groups": m3_groups,
            "m4_groups": m4_groups,
            "m3_sparse_long_offset_bin_count": len(m3_cl.get("sparse_long_offset_bins", [])),
            "m4_sparse_long_offset_bin_count": len(m4_cl.get("sparse_long_offset_bins", [])),
            "net_closed_loop_tps_gain_over_plain_95ci": any_full_tps_gain,
        },
    }


def _build_phase_b_result_sentence(hm: dict[str, Any], scale: str) -> str:
    bg = hm["bigram_eviction"]
    la = hm["full_vocab_lookahead"]
    pt1 = hm["pure_t1_fit"]
    fs = hm["full_scale"]
    tps_stmt = (
        "at least one full-scale arm achieved a 95%-CI-supported net tokens/sec gain over plain_dflash"
        if fs["net_closed_loop_tps_gain_over_plain_95ci"]
        else "no full-scale adapter arm achieved a 95%-CI-supported net tokens/sec gain over plain_dflash after proposal/update costs"
    )
    tok_tot = fs.get("primary_train_tokens")
    tok_str = f"{int(tok_tot):,}" if isinstance(tok_tot, int) else "N/A"
    return (
        f"Phase-B ({scale}) passed all executed suites with 0 upstream reference mutations: "
        f"bigram_eviction verified k_ret={bg.get('bigram_k_ret')} / k_ret=128 / count-gated kappa={bg.get('bigram_count_gate_kappa')} "
        f"prior footprint vs exact ledger; full_vocab_lookahead paired full_vocabulary and frozen-C0[j] "
        f"(beta_relax={la.get('matched_beta_relax')}) lookahead strictly against matched causal baselines; "
        f"pure_t1_fit ({pt1.get('training_distribution')}) reached primary held-out sparse test KL "
        f"{_fmt_num(pt1.get('best_pure_t1_primary_sparse_test_kl'), 4)} and pure-T=1 test KL "
        f"{_fmt_num(pt1.get('best_pure_t1_own_sparse_test_kl'), 4)}; full_scale M1-M4 collected {tok_str} "
        f"primary train tokens, {fs.get('long_dev_generations')} long dev + {fs.get('long_test_generations')} "
        f"long test generations, and evaluated 4 M3 + {len({g['arm'] for g in fs['m4_groups']})} M4 arms "
        f"with 3-seed prompt-aggregated bootstrap CIs ({tps_stmt}); <8-prompt long offset bins remain marked "
        f"SPARSE/EMPTY (conclusion unavailable), and external baselines "
        f"(full tree-based Token Recycling, Test-Time Speculation, SharpDraft) were not run"
    )


def build_updated_milestones_phase_b(
    old_milestones: dict[str, Any],
    *,
    scale: str,
    selected_suites: list[str],
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    evidence_paths: list[str],
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> dict[str, Any]:
    new_ms = copy.deepcopy(old_milestones)
    new_ms["phase_b_full_scale"] = {
        "status": "passed",
        "optional": True,
        "current_phase": "complete",
        "scale": scale,
        "suites_executed": list(selected_suites),
        "completed_utc": completed_utc,
        "project_git_commit": git_commit_sha,
        "evidence": list(evidence_paths),
        "artifact_sha256": dict(suite_shas),
        "validation": {
            "status": audit_report["status"],
            "checks": audit_report["checks"],
            "audited_files": audit_report["audited_files_count"],
            "validator": "audit_phase_b_full_scale.py",
        },
        "headline_metrics": headline_metrics,
        "external_baselines_not_run": list(EXTERNAL_BASELINES_NOT_RUN),
        "result": _build_phase_b_result_sentence(headline_metrics, scale),
    }
    for k in old_milestones:
        if k != "phase_b_full_scale":
            assert new_ms[k] == old_milestones[k], f"Upstream milestone {k} mutated!"
    return new_ms


def render_summary_phase_b_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> str:
    bg = headline_metrics["bigram_eviction"]
    la = headline_metrics["full_vocab_lookahead"]
    pt1 = headline_metrics["pure_t1_fit"]
    fs = headline_metrics["full_scale"]

    tps_verdict = (
        "At least one `--scale full` arm achieved a 95%-CI-supported net `tokens/sec` gain over `plain_dflash`."
        if fs["net_closed_loop_tps_gain_over_plain_95ci"]
        else "Across `--scale full` M3 (4 arms) and M4 (all 9 full-scale arms), **no adapter arm established a 95%-CI-supported net end-to-end `tokens/sec` speedup over `plain_dflash`** after proposal and online update costs."
    )
    tok_tot = fs.get("primary_train_tokens")
    tok_str = f"{int(tok_tot):,}" if isinstance(tok_tot, int) else "N/A"
    nested_md = _demote_phase_b_markdown_headings(render_phase_b_markdown(audit_report))

    sha_lines = [
        f"- `artifacts/full_scale/{PHASE_B_SUITE_REL_JSONS[k]}`: `{v}`"
        for k, v in suite_shas.items()
        if k in PHASE_B_SUITE_REL_JSONS
    ]

    lines = [
        "---",
        "",
        f"## Phase-B Secondary Reference Ablations & Full-Scale (`--scale {scale}`) Study (`edraft.phase_b_suite`)",
        "",
        f"Completed at `{completed_utc}` on commit `{git_commit_sha}` inside isolated `artifacts/full_scale/` while keeping all `{audit_report['upstream_reference_immutability']['checked_upstream_files']}` frozen reference M0–M4/M5 artifacts byte-for-byte intact (`{audit_report['checks']:,}` `audit_phase_b_full_scale.py` checks across `{audit_report['audited_files_count']:,}` files with `0` failures).",
        "",
        "**Headline Phase-B & `--scale full` empirical findings:**",
        f"- **`OnlineBigram` top-`k_ret` eviction & count-gating (`bigram_eviction`):** compared exact un-evicted `online_bigram` (`k_ret=None`), `online_bigram_kret256` (`k_ret={bg.get('bigram_k_ret')}`), `online_bigram_kret128` (`k_ret=128`), and `online_bigram_count_gated` (`kappa={bg.get('bigram_count_gate_kappa')}`), quantifying exact ledger vs retained prior payload bytes across `{bg['memory_ablation_groups']}` condition/arm groups.",
        f"- **Full-vocabulary vs candidate-masked lookahead (`full_vocab_lookahead`):** evaluated `lookahead_static_full` and `lookahead_rerelax_full` strictly against matched-support causal `dense_full` alongside `lookahead_static_candidate` and `lookahead_rerelax_candidate` against matched `C0[j]` `dense_candidate` (`beta_relax={la.get('matched_beta_relax')}`, positive `T_relax > 0`, profiled `lookahead_iterations > 0 ms`).",
        f"- **Pure `T_collection=1.0` offline calibration ablation (`pure_t1_fit`):** trained exclusively on `pure_T1_calibration_ablation_m5` shards (`0` `T=0.0`/`T=0.7` or `long_*` leakage), achieving `{_fmt_num(pt1.get('best_pure_t1_primary_sparse_test_kl'), 4)}` primary held-out `dense_sparse` test KL and `{_fmt_num(pt1.get('best_pure_t1_own_sparse_test_kl'), 4)}` pure-`T=1` held-out test KL (`{pt1.get('best_pure_t1_variant_on_primary')}`).",
        f"- **Full-scale (`--scale full --allow_large_run`) M1–M4 pipeline (`full_scale`):** collected `{tok_str}` primary train tokens (`T=0.0`: `{fs.get('primary_train_T0_tokens')}`, `T=0.7`: `{fs.get('primary_train_T07_tokens')}`) plus `{fs.get('long_dev_generations')}` long dev and `{fs.get('long_test_generations')}` long test generations (`{fs.get('long_condition')}`), fitted full-scale M2 adapters, and evaluated all 4 M3 arms and all `{len({g['arm'] for g in fs['m4_groups']})}` M4 arms (`offline_online_stride5`, `zero_offline_online_only`, `online_rejection_relative`, `online_cutoff` in addition to the 5 reference arms) with 3-seed prompt-aggregated bootstrap 95% CIs and reconciled 1-GPU vs 8-GPU NUMA/16-core-pinned timing. {tps_verdict}",
        f"- **Sparsity gates & external baselines:** any `< 8`-prompt long-output offset bin (`16K+`) is explicitly marked **SPARSE / EMPTY (`conclusion unavailable`)**. **External baselines not run:** {', '.join(EXTERNAL_BASELINES_NOT_RUN)}.",
        "",
        "**Checksummed Phase-B artifact manifest:**",
        *sha_lines,
        "",
        nested_md,
    ]
    return "\n".join(lines)


def render_readme_phase_b_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> str:
    bg = headline_metrics["bigram_eviction"]
    la = headline_metrics["full_vocab_lookahead"]
    pt1 = headline_metrics["pure_t1_fit"]
    fs = headline_metrics["full_scale"]
    tok_tot = fs.get("primary_train_tokens")
    tok_str = f"{int(tok_tot):,}" if isinstance(tok_tot, int) else "N/A"
    tps_clause = (
        "see [reports/SUMMARY.md](reports/SUMMARY.md) for per-condition 95% CIs"
        if fs["net_closed_loop_tps_gain_over_plain_95ci"]
        else "no adapter arm established a 95%-CI-supported net `tokens/sec` gain over `plain_dflash`"
    )
    return "\n".join(
        [
            "## Phase-B secondary reference ablations and full-scale (`--scale full`) execution",
            "",
            f"All three secondary reference ablations (`bigram_eviction`, `full_vocab_lookahead`, `pure_t1_fit`) and the isolated `--scale {scale}` M1–M4 pipeline (`artifacts/full_scale/`) have **passed** (`{completed_utc}`, commit `{git_commit_sha}`, `{audit_report['checks']:,}` independent `audit_phase_b_full_scale.py` checks with zero failures and zero mutations to `artifacts/`):",
            "",
            "```bash",
            "# CPU synthetic smoke check across all Phase-B suites (no downloads):",
            "uv run --frozen python -m edraft.phase_b_suite --scale smoke --suite all --gpus 0,1 --output artifacts/smoke_phase_b",
            "",
            "# Full 8-GPU NUMA/16-core-pinned secondary ablations + --scale full pipeline:",
            "uv run --frozen python -m edraft.phase_b_suite --scale full --suite all --allow_large_run --timing_repeats 2 --bf16_gemm --fast_decode --gpus 0,1,2,3,4,5,6,7 --output artifacts/full_scale --reference_artifacts artifacts",
            "```",
            "",
            f"- **Secondary reference ablations:** verified `OnlineBigram` top-`k_ret` eviction (`k_ret={bg.get('bigram_k_ret')}`, `128`, vs exact `None` and count-gated `kappa={bg.get('bigram_count_gate_kappa')}`), matched-support `full_vocabulary` vs `frozen_C0` (`beta_relax={la.get('matched_beta_relax')}`) lookahead, and pure `T_collection=1.0` offline fitting (`{_fmt_num(pt1.get('best_pure_t1_primary_sparse_test_kl'), 4)}` primary test KL, `{_fmt_num(pt1.get('best_pure_t1_own_sparse_test_kl'), 4)}` pure-`T=1` test KL).",
            f"- **Full-scale M1–M4 (`artifacts/full_scale/`):** collected `{tok_str}` primary train tokens and `{fs.get('long_dev_generations')}` dev / `{fs.get('long_test_generations')}` test long generations, fitted full-scale M2 checkpoints, and evaluated 4 M3 + `{len({g['arm'] for g in fs['m4_groups']})}` M4 arms (`stride=5`, `zero_offline_online_only`, `rejection_relative`, `cutoff`, plus reference arms) with 3-seed prompt-aggregated bootstrap 95% CIs and reconciled 1-GPU vs 8-GPU core-pinned timing ({tps_clause}). `< 8`-prompt long offset bins remain marked `SPARSE` (`conclusion unavailable`), and full tree-based Token Recycling, Test-Time Speculation, and SharpDraft remain unrun external baselines.",
        ]
    )


def render_notes_phase_b_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    audit_report: dict[str, Any],
    headline_metrics: dict[str, Any],
) -> str:
    fs = headline_metrics["full_scale"]
    pt1 = headline_metrics["pure_t1_fit"]
    sha_inline = ", ".join(f"`{k}={v}`" for k, v in suite_shas.items())
    return "\n".join(
        [
            f"Phase-B secondary ablations & full-scale (`scale={scale}`) completion and audit, {completed_utc} (commit={git_commit_sha}):",
            f"- Independent validator `audit_phase_b_full_scale.py --scale {scale} --suite all --require_complete` passed `{audit_report['checks']:,}` checks across `{audit_report['audited_files_count']:,}` files (`0` failures; `{audit_report['upstream_reference_immutability']['checked_upstream_files']}` frozen reference M0–M4/M5 files verified intact).",
            f"- Phase-B artifact SHA256s: {sha_inline}.",
            f"- Headline summary: `bigram_eviction` (`k_ret=None, 256, 128`, `kappa={headline_metrics['bigram_eviction'].get('bigram_count_gate_kappa')}`), `full_vocab_lookahead` (`beta_relax={headline_metrics['full_vocab_lookahead'].get('matched_beta_relax')}`), `pure_t1_fit` (`{_fmt_num(pt1.get('best_pure_t1_primary_sparse_test_kl'), 6)}` primary sparse test KL / `{_fmt_num(pt1.get('best_pure_t1_own_sparse_test_kl'), 6)}` pure-`T=1` sparse test KL), and `full_scale` (`primary_train_tokens={fs.get('primary_train_tokens')}`, `long_dev={fs.get('long_dev_generations')}`, `long_test={fs.get('long_test_generations')}`, `net_tps_95ci_gain={fs['net_closed_loop_tps_gain_over_plain_95ci']}`). External baselines not run: {', '.join(EXTERNAL_BASELINES_NOT_RUN)}.",
        ]
    )


def render_progress_phase_b_section(
    *,
    scale: str,
    completed_utc: str,
    git_commit_sha: str,
    suite_shas: dict[str, str],
    audit_report: dict[str, Any],
) -> str:
    sha_bullets = [
        f"   - `artifacts/full_scale/{PHASE_B_SUITE_REL_JSONS[k]}` (`sha256: {v}`)"
        for k, v in suite_shas.items()
        if k in PHASE_B_SUITE_REL_JSONS
    ]
    return "\n".join(
        [
            "---",
            "",
            f"## Phase-B Secondary Ablations & Full-Scale (`scale={scale}`) Completion Update (`{completed_utc}`)",
            "",
            f"1. **All Phase-B Suites Completed in Isolated `artifacts/full_scale/` (`commit {git_commit_sha}`):**",
            *sha_bullets,
            f"2. **Zero-Drift Independent Verification Passed (`{audit_report['checks']:,}` Checks, `0` Failures):**",
            f"   - `audit_phase_b_full_scale.py --scale {scale} --suite all --require_complete` verified `bigram_eviction`, `full_vocab_lookahead`, `pure_t1_fit`, and `full_scale` (M1-pre -> M4) raw shards, 3-seed prompt-aggregated bootstrap CIs, 1-GPU vs 8-GPU timing reconciliation, `<8`-prompt sparse-bin flags, and `{audit_report['upstream_reference_immutability']['checked_upstream_files']}` frozen reference artifacts.",
            "3. **Milestones, Implementation Audit & Documentation Synchronized:**",
            "   - `artifacts/milestones.json` (`phase_b_full_scale.status = \"passed\"`), `artifacts/implementation_audit.json`, `reports/SUMMARY.md`, `README.md`, `NOTES.md`, and `AGENTS.md` updated atomically and re-verified.",
        ]
    )


def _stage_phase_b_dry_run_sandbox(
    repo: Path, reference_artifacts: Path, phase_b_root: Path, dst_root: Path
) -> tuple[Path, Path, Path]:
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

    s_ref_art = (
        s_repo / "artifacts"
        if reference_artifacts.resolve() == (repo / "artifacts").resolve()
        else (dst_root / "ref_artifacts")
    )
    s_ref_art.mkdir(parents=True, exist_ok=True)
    if (s_repo / "artifacts") != s_ref_art and not (s_repo / "artifacts").exists():
        os.symlink(s_ref_art, s_repo / "artifacts")

    for a_item in reference_artifacts.iterdir():
        if a_item.name in ("milestones.json", "implementation_audit.json") and a_item.is_file():
            shutil.copy2(a_item, s_ref_art / a_item.name)
        else:
            os.symlink(a_item.resolve(), s_ref_art / a_item.name)
    for fallback_name in ("milestones.json", "implementation_audit.json"):
        if not (s_ref_art / fallback_name).exists() and (repo / "artifacts" / fallback_name).is_file():
            shutil.copy2(repo / "artifacts" / fallback_name, s_ref_art / fallback_name)

    if phase_b_root.resolve() == (reference_artifacts / "full_scale").resolve():
        s_pb = s_ref_art / "full_scale"
    else:
        s_pb = phase_b_root
    return s_repo, s_ref_art, s_pb


def run_update_docs_phase_b(
    *,
    repo: Path,
    reference_artifacts: Path,
    phase_b_root: Path,
    scale: str = "full",
    suite: str = "all",
    source_repo: Path | None = None,
    completed_utc: str | None = None,
    dry_run: bool = False,
    skip_post_gate_e_audit: bool = False,
) -> dict[str, Any]:
    repo = repo.resolve()
    reference_artifacts = reference_artifacts.resolve()
    phase_b_root = phase_b_root.resolve()

    if suite != "all":
        return {
            "status": "failed_pre_gate",
            "checks": 0,
            "failures": [
                f"update_docs_phase_b requires suite=='all' (all 4 Phase-B suites verified complete) "
                f"before marking phase_b_full_scale 'passed', got suite={suite!r}"
            ],
        }

    # Step 0: Mandatory Pre-Write Gate — verify 100% completion & zero drift across all 4 Phase-B suites!
    pre_auditor = PhaseBFullScaleAuditor(
        repo=repo,
        phase_b_root=phase_b_root,
        reference_artifacts=reference_artifacts,
        scale=scale,
        suite="all",
        allow_partial=False,
        require_complete=True,
        allow_pilot=False,
    )
    audit_report = pre_auditor.run_all()
    if audit_report["status"] != "passed" or audit_report["failures"]:
        return {
            "status": "failed_pre_gate",
            "checks": audit_report["checks"],
            "failures": audit_report["failures"],
        }

    suite_shas: dict[str, str] = {}
    evidence_paths: list[str] = []
    missing_rel: list[str] = []
    for key, rel in PHASE_B_SUITE_REL_JSONS.items():
        fp = phase_b_root / rel
        if not fp.is_file():
            missing_rel.append(rel)
        else:
            suite_shas[key] = sha256_file(fp)
            evidence_paths.append(f"artifacts/full_scale/{rel}")
    if missing_rel:
        return {
            "status": "failed_pre_gate",
            "checks": audit_report["checks"],
            "failures": [f"missing required Phase-B suite report(s): {missing_rel}"],
        }

    ms_in_path = (
        (reference_artifacts / "milestones.json")
        if (reference_artifacts / "milestones.json").is_file()
        else (repo / "artifacts" / "milestones.json")
    )
    old_milestones = json.loads(ms_in_path.read_text(encoding="utf-8"))
    master_path = phase_b_root / "phase_b_summary.json"
    master_json = json.loads(master_path.read_text(encoding="utf-8")) if master_path.is_file() else {}

    eff_utc = _resolve_phase_b_completed_utc(
        explicit_utc=completed_utc,
        old_milestones=old_milestones,
        suite_shas=suite_shas,
        master_summary_path=master_path,
        master_summary_json=master_json,
    )
    git_sha = _git_head(repo)
    if git_sha == "unknown":
        git_sha = str(master_json.get("project_git_commit") or "unknown")

    headline_metrics = _extract_phase_b_headline_metrics(audit_report)
    new_milestones = build_updated_milestones_phase_b(
        old_milestones,
        scale=scale,
        selected_suites=list(audit_report.get("selected_suites") or ALL_PHASE_B_SUITES),
        completed_utc=eff_utc,
        git_commit_sha=git_sha,
        suite_shas=suite_shas,
        evidence_paths=evidence_paths,
        audit_report=audit_report,
        headline_metrics=headline_metrics,
    )

    new_agents_text = update_agents_md_text((repo / "AGENTS.md").read_text(encoding="utf-8"))
    new_summary_text = _upsert_delimited_section(
        (repo / "reports" / "SUMMARY.md").read_text(encoding="utf-8"),
        "PHASE_B_FULL_SCALE_SUMMARY",
        render_summary_phase_b_section(
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
        "PHASE_B_FULL_SCALE_README",
        render_readme_phase_b_section(
            scale=scale,
            completed_utc=eff_utc,
            git_commit_sha=git_sha,
            audit_report=audit_report,
            headline_metrics=headline_metrics,
        ),
    )
    new_notes_text = _upsert_delimited_section(
        (repo / "NOTES.md").read_text(encoding="utf-8"),
        "PHASE_B_FULL_SCALE_NOTES",
        render_notes_phase_b_section(
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
        "PHASE_B_FULL_SCALE_PROGRESS",
        render_progress_phase_b_section(
            scale=scale,
            completed_utc=eff_utc,
            git_commit_sha=git_sha,
            suite_shas=suite_shas,
            audit_report=audit_report,
        ),
    )

    def _apply_and_verify(
        target_repo: Path, target_ref_art: Path, target_pb_root: Path
    ) -> dict[str, Any]:
        # Write 1..6 BEFORE regenerating implementation_audit.json (Step 7)!
        ms_target = (
            (target_ref_art / "milestones.json")
            if (target_ref_art / "milestones.json").exists()
            or target_ref_art != (target_repo / "artifacts")
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

        # Step 7: Regenerate implementation_audit.json LAST (natively tracking Phase-B SHA256s & milestone)!
        if target_pb_root.resolve() != (target_ref_art / "full_scale").resolve() and not (
            target_ref_art / "full_scale"
        ).exists():
            os.symlink(target_pb_root, target_ref_art / "full_scale")
        tpl_p = (
            (target_ref_art / "implementation_audit.json")
            if (target_ref_art / "implementation_audit.json").is_file()
            else (repo / "artifacts" / "implementation_audit.json")
        )
        impl_audit = build_implementation_audit(
            target_repo,
            target_ref_art,
            template_path=tpl_p,
            audited_at_utc=eff_utc,
        )

        ia_target = (
            (target_ref_art / "implementation_audit.json")
            if (target_ref_art / "implementation_audit.json").exists()
            or target_ref_art != (target_repo / "artifacts")
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
            target_ref_art,
            check_git_head=(_git_head(target_repo) != "unknown"),
        )

        # Post-Write Gate 1: Verify newly written `milestones.json` against frozen M0-M5 reference files
        # AND re-verify SHA256 immutability across ALL pre-audited Phase-B + reference files (`70,000+` files!)
        # without re-bootstrapping untouched raw rows a second time!
        pb_post_aud = PhaseBFullScaleAuditor(
            repo=target_repo,
            phase_b_root=target_pb_root,
            reference_artifacts=target_ref_art,
            scale=scale,
            suite=suite,
            allow_partial=False,
            require_complete=True,
            allow_pilot=False,
        )
        updated_meta_names = {"milestones.json", "implementation_audit.json"}
        for p_str, exp_sha in pre_auditor.read_sha256.items():
            if Path(p_str).name not in updated_meta_names:
                pb_post_aud.read_sha256[p_str] = exp_sha
        pb_post_aud.audit_upstream_reference_immutability()
        pb_post_aud.verify_immutability_at_end()
        pb_post = {
            "status": "passed" if not pb_post_aud.failures else "failed",
            "checks": pre_auditor.checks + pb_post_aud.checks,
            "failures": list(pb_post_aud.failures),
        }

        # Post-Write Gate 2: PostM4SuiteAuditor (--require_complete --check_implementation_audit) if all 4 Phase-A suites present
        post_m4_failures: list[str] = []
        post_m4_checks = 0
        pm4_sum_p = target_ref_art / "post_m4_suite" / "post_m4_suite_summary.json"
        if pm4_sum_p.is_file():
            pm4_sum_data = json.loads(pm4_sum_p.read_text(encoding="utf-8"))
            if (
                pm4_sum_data.get("status") == "complete"
                and set(pm4_sum_data.get("executed_suites") or [])
                >= {"m2_refit_calibration", "online_head_to_head", "m5_eval", "fastpath_bench"}
            ):
                pm4_scale = str(pm4_sum_data.get("scale") or "reference")
                pm4_rep = PostM4SuiteAuditor(
                    repo=target_repo,
                    artifacts=target_ref_art,
                    scale=pm4_scale,
                    allow_partial=False,
                    require_complete=True,
                    allow_pilot=False,
                    check_implementation_audit=True,
                ).run_all()
                if _git_head(target_repo) == "unknown":
                    pm4_rep["failures"] = [
                        f for f in pm4_rep["failures"] if "!= git HEAD" not in f
                    ]
                    pm4_rep["status"] = "passed" if not pm4_rep["failures"] else "failed"
                post_m4_checks = pm4_rep["checks"]
                post_m4_failures = list(pm4_rep["failures"])

        # Post-Write Gate 3: PostGateEAuditor repo & doc conformance
        gate_e_failures: list[str] = []
        gate_e_checks = 0
        if not skip_post_gate_e_audit:
            eff_src = source_repo
            if eff_src is None and Path("/tmp/edraft_09a5469_source").is_dir():
                eff_src = Path("/tmp/edraft_09a5469_source")
            has_ref_m4 = (target_ref_art / "m4" / "summary.json").is_file()
            ge_aud = PostGateEAuditor(
                repo=target_repo,
                artifacts=target_ref_art,
                scale="reference" if has_ref_m4 else "smoke",
                allow_partial=True,
                require_complete=has_ref_m4,
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
        all_failures = (
            list(ia_errs)
            + list(pb_post["failures"])
            + post_m4_failures
            + gate_e_failures
        )
        return {
            "status": "passed" if not all_failures else "failed_post_gate",
            "dry_run": dry_run,
            "scale": scale,
            "suite": suite,
            "completed_utc": eff_utc,
            "project_git_commit": git_sha,
            "phase_b_status": new_milestones["phase_b_full_scale"]["status"],
            "implementation_audit_status": impl_audit["status"],
            "phase_b_audit_checks": pb_post["checks"],
            "post_m4_audit_checks": post_m4_checks,
            "post_gate_e_doc_checks": gate_e_checks,
            "failures": all_failures,
            "updated_file_sha256": written_shas,
            "phase_b_artifact_sha256": suite_shas,
        }

    if dry_run:
        with tempfile.TemporaryDirectory(prefix="edraft_phase_b_docs_dryrun_") as td:
            s_repo, s_ref_art, s_pb = _stage_phase_b_dry_run_sandbox(
                repo, reference_artifacts, phase_b_root, Path(td)
            )
            return _apply_and_verify(s_repo, s_ref_art, s_pb)

    return _apply_and_verify(repo, reference_artifacts, phase_b_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/catid/drafter"))
    parser.add_argument("--reference_artifacts", type=Path, default=None)
    parser.add_argument("--phase_b_root", type=Path, default=None)
    parser.add_argument("--scale", choices=["smoke", "reference", "full"], default="full")
    parser.add_argument(
        "--suite",
        choices=["all", "ref_secondary_ablations", *ALL_PHASE_B_SUITES],
        default="all",
    )
    parser.add_argument("--source_repo", type=Path, default=None)
    parser.add_argument("--completed_utc", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_post_gate_e_audit", action="store_true")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    ref_art = (
        args.reference_artifacts.resolve()
        if args.reference_artifacts is not None
        else (repo / "artifacts")
    )
    pb_root = (
        args.phase_b_root.resolve()
        if args.phase_b_root is not None
        else (ref_art / "full_scale")
    )
    res = run_update_docs_phase_b(
        repo=repo,
        reference_artifacts=ref_art,
        phase_b_root=pb_root,
        scale=args.scale,
        suite=args.suite,
        source_repo=args.source_repo,
        completed_utc=args.completed_utc,
        dry_run=args.dry_run,
        skip_post_gate_e_audit=args.skip_post_gate_e_audit,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
