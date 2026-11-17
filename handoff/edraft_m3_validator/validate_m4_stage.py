#!/usr/bin/env python3
"""Read-only stage-scoped M4 validator wrapper (Gate C: tune, Gate D: evaluate).

Reuses the exact Audit logic from artifacts/m4_evidence_validator/validate.py
without modifying a single byte inside /home/catid/drafter.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = Path("/home/catid/drafter")
sys.path.insert(0, str(REPO_DEFAULT / "artifacts" / "m4_evidence_validator"))

import validate as m4v  # noqa: E402


def run_scoped_audit(audit: m4v.Audit, scope: str) -> dict:
    audit.prepare()
    _, session = audit.session("tune", audit.settings["comparison_session"])
    config = session["configuration"]["configuration"]
    audit.require(
        config
        == {
            "prompt_manifest_hash": m4v.fingerprint(audit.manifest),
            "dev_ids": audit.settings["development_prompt_ids"],
            "candidates": {a: m4v.candidates(a) for a in m4v.ARMS},
        },
        "tune configuration mismatch",
    )
    entries = audit.manifests("tune")
    expected = set()
    for i, p in enumerate(audit.prompts["dev"]):
        sample = m4v.conditions(p["condition"], primary=True)[0]
        pairs = [
            (p, a, setting, sample, audit.args.seed)
            for a in m4v.ARMS
            for setting in m4v.candidates(a)
        ]
        offset = i % len(pairs)
        expected.add(
            audit.bundle("tune", session, entries, pairs[offset:] + pairs[:offset], "dev", False)
        )
    audit.require(set(session["bundles"]) == expected, "tune bundle population differs")
    audit.check_selection()

    req_tune = 36 if audit.args.scale == "smoke" else 180
    req_tune_online = 16 if audit.args.scale == "smoke" else 80
    audit.require(audit.counts["tune"] == req_tune, "tune: row count differs from full schedule")
    audit.require(
        audit.counts["tune_online"] == req_tune_online,
        "tune: online count differs from full schedule",
    )

    extra_summary: dict = {
        "frozen_settings_hash": audit.frozen,
        "selected_settings": audit.settings["selected"],
        "candidates": audit.settings["candidates"],
    }

    if scope == "evaluate":
        entries = audit.manifests("evaluate")
        expected_eval = set()
        reused_count = 0
        fresh_count = 0
        for p, sample, seed in audit.test_cases(audit.prompts["test"]):
            for arm in m4v.ARMS:
                key = audit.row(
                    audit.identity(
                        p, arm, audit.settings["selected"][arm], sample, seed, "evaluate"
                    ),
                    entries,
                    "evaluate",
                )
                expected_eval.add(key)
                rpath = Path(entries[key]["result_path"]).resolve()
                rjson = json.loads(rpath.read_text())
                if (rjson.get("baseline_reuse") or {}).get("status") == "reused":
                    reused_count += 1
                else:
                    fresh_count += 1
        selected = {
            k
            for k, e in entries.items()
            if e["identity"].get("frozen_settings_hash") == audit.frozen
        }
        audit.require(selected == expected_eval, "evaluation population differs for frozen settings")

        n = 40 if audit.args.scale == "smoke" else 2400
        audit.require(audit.counts["evaluate"] == n, "evaluate: row count differs")
        audit.require(
            audit.counts["evaluate_online"] == n * 2 // 5,
            "evaluate: online row count differs",
        )
        if audit.args.scale == "reference":
            audit.require(
                reused_count == 960 and fresh_count == 1440,
                f"evaluate: expected 960 reused M3 baseline + 1440 fresh M4 neural rows, got {reused_count}/{fresh_count}",
            )
            greedy_cmp_path = audit.stage / "greedy_output_comparisons.json"
            greedy_cmp = audit.artifact(greedy_cmp_path)
            comps = greedy_cmp.get("comparisons") or []
            audit.require(
                greedy_cmp.get("status") == "complete"
                and greedy_cmp.get("expected_prompts") == 64
                and greedy_cmp.get("expected_comparisons") == 640
                and greedy_cmp.get("completed_comparisons") == 640
                and greedy_cmp.get("missing_comparisons") == []
                and len(greedy_cmp.get("groups") or []) == 10
                and len(comps) == 640,
                f"unexpected greedy_output_comparisons state: {greedy_cmp.get('status')} / {len(comps)}",
            )
            extra_summary["greedy_comparisons_count"] = len(comps)
            extra_summary["greedy_groups"] = greedy_cmp.get("groups")

            summary_path = audit.stage / "summary.json"
            if summary_path.exists():
                summary_doc = audit.artifact(summary_path)
                audit.require(
                    summary_doc.get("stage") == "m4"
                    and summary_doc.get("scale") == "reference"
                    and summary_doc.get("expected_requests") == 2400
                    and summary_doc.get("completed_requests") == 2400
                    and summary_doc.get("missing_requests") == []
                    and summary_doc.get("greedy_output_comparisons_complete") is True
                    and summary_doc.get("feature_context_complete") is True,
                    "m4/summary.json pre-timing evaluation fields incomplete",
                )
                extra_summary["summary_pre_timing_verified"] = True
            feat_path = audit.stage / "feature_context.json"
            if feat_path.exists():
                feat_doc = audit.artifact(feat_path)
                audit.require(
                    feat_doc.get("status") == "complete" and feat_doc.get("label") == "reference",
                    "m4/feature_context.json not complete at reference scale",
                )
                extra_summary["feature_context_verified"] = True
        extra_summary["evaluate_reused_rows"] = reused_count
        extra_summary["evaluate_fresh_rows"] = fresh_count

    for path, expected_hash in audit.files.items():
        audit.require(m4v.digest(path) == expected_hash, f"artifact changed during audit: {path}")
    return extra_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("smoke", "reference"), default="reference")
    parser.add_argument("--scope", choices=("tune", "evaluate"), required=True)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--artifacts", type=Path, default=REPO_DEFAULT / "artifacts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = m4v.Audit(args)
    started = datetime.now(timezone.utc).isoformat()
    extra_summary = {}
    try:
        extra_summary = run_scoped_audit(audit, args.scope)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        audit.failures.append(f"{type(error).__name__}: {error}")

    commit = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "status": "passed" if not audit.failures else "failed",
        "scope": args.scope,
        "scale": args.scale,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": commit,
        "checks": audit.checks,
        "counts": dict(audit.counts),
        "failures": audit.failures,
        "extra_summary": extra_summary,
        "input_artifact_sha256": audit.files,
        "limitations": m4v.LIMITATIONS,
    }
    m4v.atomic_json(args.output, result)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "scope", "scale", "checks", "counts", "failures")}
        )
    )
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
