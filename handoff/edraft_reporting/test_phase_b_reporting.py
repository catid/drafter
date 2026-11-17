#!/usr/bin/env python3
"""Comprehensive end-to-end & fault-injection test suite for `audit_phase_b_full_scale.py`."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

IWT = Path(
    "/home/catid/.gemini/antigravity-cli/worktrees/"
    "cfcf3ba7-dc2c-4cac-926a-577986ac8615/swarm-integrator-970cc63a"
)
if (IWT / "src").is_dir():
    sys.path[:] = [p for p in sys.path if p != str(Path("/home/catid/drafter/src"))]
    if str(IWT / "src") not in sys.path:
        sys.path.insert(0, str(IWT / "src"))
    if "edraft" in sys.modules and not str(getattr(sys.modules["edraft"], "__file__", "")).startswith(
        str(IWT / "src")
    ):
        for _mod_k in [k for k in list(sys.modules) if k == "edraft" or k.startswith("edraft.")]:
            sys.modules.pop(_mod_k, None)

import audit_phase_b_full_scale as apb  # noqa: E402
import build_and_audit_reports as bar  # noqa: E402
import build_implementation_audit as bia  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for deterministic synthetic `scale="full"` Phase-B workspace
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return bar.sha256_bytes(raw)


def _make_ClosedLoopRow(
    *,
    stage: str,
    arm: str,
    prompt_id: str,
    condition: str,
    temperature: float,
    warp: dict[str, Any],
    seed: int,
    setting: dict[str, Any],
    tps: float = 120.0,
    acc_len: float = 4.2,
    phase_profile: bool = False,
    reused_from: tuple[Path, str] | None = None,
) -> dict[str, Any]:
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
    online = None
    online_inv = None
    pnorms: dict[str, float] = {}
    if arm in ("dense", "dense_sparse", "offline_static"):
        pnorms = {"W": 1.5, "A": 0.8, "b": 0.25}
    elif arm in bar.ONLINE_ARMS:
        if arm == "zero_offline_online_only":
            pnorms = {"W": 0.0, "A": 0.0, "b": 0.0}
        else:
            pnorms = {"W": 1.5, "A": 0.8, "b": 0.25}
        stride = 5 if arm == "offline_online_stride5" else 1
        u_steps = len(blocks) // stride
        is_dbs = arm == "online_dbS"
        online = {
            "u": u_steps,
            "dW_norm": 0.0 if is_dbs else 0.05,
            "dA_norm": 0.0 if is_dbs else 0.03,
            "db_norm": 0.02,
            "dS_norm": 0.04,
            "db_exact_zero_fraction": 0.99,
            "active_buckets": 12,
            "active_buckets_max_over_updates": 12,
            "sparse_capacity": 1048576,
            "state_bytes_estimate_total": 4096,
        }
        online_inv = {
            "request_reset": True,
            "version_only_between_rounds": True,
            "stride": stride,
            "executed_updates": u_steps,
            "single_round_no_buffering": True,
        }

    phase_ms: dict[str, float] | None = {
        ph: (0.5 if phase_profile else 0.0) for ph in bar.SECTION11_REQUIRED_PHASES
    }
    row: dict[str, Any] = {
        "scale": "full",
        "label": "full",
        "stage": stage,
        "arm": arm,
        "prompt_id": prompt_id,
        "split": "test",
        "condition": condition,
        "seed": seed,
        "sampling_name": (
            "thinking_recommended"
            if condition.startswith("long_")
            else ("greedy" if temperature == 0.0 else ("nonthinking_recommended" if temperature == 0.7 else "pure_temperature_diagnostic"))
        ),
        "temperature": temperature,
        "sampling_warp_config": warp,
        "settings": setting,
        "effective_beta": float(setting.get("beta", 1.0)) if setting else 1.0,
        "verification_rule": "ratio" if temperature > 0.0 else "greedy",
        "input_metadata": {"input_context_length": 128},
        "generated_tokens": len(out_toks),
        "elapsed_seconds": None if reused_from else (len(out_toks) / tps),
        "generation_loop_elapsed_seconds": None if reused_from else (len(out_toks) / tps),
        "tokens_per_second": None if reused_from else tps,
        "mean_accepted_length": acc_len,
        "median_accepted_length": 4.0,
        "target_forwards_per_generated_token": (len(blocks) + 1) / len(out_toks),
        "phase_ms": None if reused_from else phase_ms,
        "blocks": blocks,
        "generation": {
            "complete": True,
            "eos_reached": True,
            "capped": False,
            "repetition_dominated": False,
        },
        "output_tokens": out_toks,
        "online": online,
        "online_invariants": online_inv,
        "parameter_norms": pnorms,
        "peak_allocated_bytes": None if reused_from else 1024,
        "peak_reserved_bytes": None if reused_from else 2048,
        "phase_profile": phase_profile,
    }
    if arm == "online_bigram":
        row["bigram"] = {
            "k_ret": None,
            "count_gate_kappa": None,
            "ledger_retained_for_exact_counts": True,
            "counted_transitions": 23,
            "exact_ledger_pairs": 18,
            "retained_prior_pairs": 18,
        }
    if reused_from is not None:
        src_p, src_sha = reused_from
        row["baseline_reuse"] = {
            "status": "reused",
            "source_stage": "m3",
            "source_result_path": str(src_p),
            "source_result_sha256": src_sha,
        }
    return row


def _build_synthetic_full_scale_workspace(root: Path, ref_dir: Path) -> Path:
    """Construct a complete, mathematically consistent `scale='full'` Phase-B workspace."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    _write_json(ref_dir / "milestones.json", {"M0": {"status": "passed"}, "M4": {"status": "passed", "artifact_sha256": {}}})
    _write_json(ref_dir / "m4" / "summary.json", {"status": "complete", "scale": "reference"})
    _write_json(ref_dir / "smoke.json", {"status": "passed"})
    _write_json(ref_dir / "m0.json", {"status": "passed", "scale": "reference"})
    _write_json(ref_dir / "m0_numerics.json", {"status": "passed", "head_reconstruction": {"passed": True}})

    pb_root = root / "full_scale_out"
    pb_root.mkdir(parents=True, exist_ok=True)
    _write_json(pb_root / "smoke.json", {"status": "passed"})
    _write_json(pb_root / "m0.json", {"status": "passed", "scale": "reference"})
    _write_json(pb_root / "m0_numerics.json", {"status": "passed", "head_reconstruction": {"passed": True}})

    # 1. Prompts manifest (`6800` std train, `64` std dev, `128` std test, `32` long dev, `96` long test)
    prompt_rows = []
    for i in range(6800):
        prompt_rows.append({
            "prompt_id": f"std-train-{i:04d}",
            "dataset": "gsm8k",
            "split": "train",
            "condition": "standard_nonthinking",
            "reference_selected": True,
            "text": f"Train {i}",
        })
    for i in range(64):
        prompt_rows.append({
            "prompt_id": f"std-dev-{i:03d}",
            "dataset": "gsm8k",
            "split": "dev",
            "condition": "standard_nonthinking",
            "reference_selected": True,
            "text": f"Dev {i}",
        })
    for i in range(128):
        prompt_rows.append({
            "prompt_id": f"std-test-{i:03d}",
            "dataset": "humaneval",
            "split": "test",
            "condition": "standard_nonthinking",
            "reference_selected": True,
            "text": f"Test {i}",
        })
    for i in range(32):
        prompt_rows.append({
            "prompt_id": f"long-dev-{i:03d}",
            "dataset": "math500",
            "split": "dev",
            "condition": "long_thinking",
            "reference_selected": True,
            "text": f"Long dev {i}",
        })
    for i in range(96):
        prompt_rows.append({
            "prompt_id": f"long-test-{i:03d}",
            "dataset": "math500",
            "split": "test",
            "condition": "long_thinking",
            "reference_selected": True,
            "text": f"Long test {i}",
        })
    cal_ids = [f"std-train-{i:04d}" for i in range(64)]
    _write_json(
        pb_root / "prompts.json",
        {
            "scale": "full",
            "seed": 2026,
            "complete_heldout_dataset": "humaneval",
            "calibration_ids": cal_ids,
            "prompts": prompt_rows,
        },
    )

    # 2. Calibration (`h_proj.pt`, `token_features.pt`, `calibration.json`)
    torch.save({"proj": torch.zeros(4, 4)}, pb_root / "h_proj.pt")
    torch.save({"E": torch.zeros(16, 4)}, pb_root / "token_features.pt")
    cal_sha = _write_json(
        pb_root / "calibration.json",
        {
            "scale": "full",
            "status": "passed",
            "calibration_prompts": 64,
            "hidden_vectors": 1024,
            "identity": {"prompt_ids": cal_ids},
            "artifact_sha256": {
                "h_proj.pt": bar.sha256_file(pb_root / "h_proj.pt"),
                "token_features.pt": bar.sha256_file(pb_root / "token_features.pt"),
            },
        },
    )

    # 3. M1a audit (`2,400,000` primary train tokens = `1.2M` T=0 + `1.2M` T=0.7)
    dummy_shard = pb_root / "traces" / "shard0.pt"
    dummy_shard.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"blocks": []}, dummy_shard)
    d_sha = bar.sha256_file(dummy_shard)
    _write_json(
        dummy_shard.with_suffix(".pt.json"),
        {"sha256": d_sha, "disk_bytes": dummy_shard.stat().st_size},
    )

    warp_nt = {"min_p": 0.0, "top_k": 20, "top_p": 0.8}
    warp_pure = {"min_p": 0.0, "top_k": 0, "top_p": 1.0}
    warp_th = {"min_p": 0.0, "top_k": 20, "top_p": 0.95}
    gens = [
        {
            "prompt_id": "std-train-0000",
            "condition": "standard_nonthinking",
            "split": "train",
            "partition": "train_primary",
            "temperature": 0.0,
            "sampling_warp_config": warp_pure,
            "realized_length": 1_200_000,
            "complete": True,
            "scale": "full",
            "data": "full",
            "live_full_blocks": 0,
        },
        {
            "prompt_id": "std-train-0001",
            "condition": "standard_nonthinking",
            "split": "train",
            "partition": "train_primary",
            "temperature": 0.7,
            "sampling_warp_config": warp_nt,
            "realized_length": 1_200_000,
            "complete": True,
            "scale": "full",
            "data": "full",
            "live_full_blocks": 0,
        },
        {
            "prompt_id": "std-train-0002",
            "condition": "standard_nonthinking",
            "split": "train",
            "partition": "diagnostic",
            "temperature": 1.0,
            "sampling_warp_config": warp_pure,
            "realized_length": 50_000,
            "complete": True,
            "scale": "full",
            "data": "full",
            "live_full_blocks": 0,
        },
    ]
    for i in range(64):
        for tv, wp in ((0.0, warp_pure), (0.7, warp_nt)):
            gens.append({
                "prompt_id": f"std-dev-{i:03d}",
                "condition": "standard_nonthinking",
                "split": "dev",
                "partition": "standard_eval",
                "temperature": tv,
                "sampling_warp_config": wp,
                "realized_length": 128,
                "complete": True,
                "scale": "full",
                "data": "full",
                "live_full_blocks": 1 if (i == 0 and tv == 0.0) else 0,
            })
    for i in range(128):
        for tv, wp in ((0.0, warp_pure), (0.7, warp_nt), (1.0, warp_pure)):
            gens.append({
                "prompt_id": f"std-test-{i:03d}",
                "condition": "standard_nonthinking",
                "split": "test",
                "partition": "standard_eval" if tv < 1.0 else "diagnostic",
                "temperature": tv,
                "sampling_warp_config": wp,
                "realized_length": 128,
                "complete": True,
                "scale": "full",
                "data": "full",
                "live_full_blocks": 1 if (i == 0 and tv == 0.0) else 0,
            })
    for i in range(32):
        gens.append({
            "prompt_id": f"long-dev-{i:03d}",
            "condition": "long_thinking",
            "split": "dev",
            "partition": "long",
            "temperature": 0.6,
            "sampling_warp_config": warp_th,
            "realized_length": 1024,
            "complete": True,
            "scale": "full",
            "data": "full",
            "live_full_blocks": 0,
        })
    for i in range(96):
        gens.append({
            "prompt_id": f"long-test-{i:03d}",
            "condition": "long_thinking",
            "split": "test",
            "partition": "long",
            "temperature": 0.6,
            "sampling_warp_config": warp_th,
            "realized_length": 1024,
            "complete": True,
            "scale": "full",
            "data": "full",
            "live_full_blocks": 0,
        })

    tok_counts: dict[str, int] = {}
    pm_sets: dict[str, set[str]] = {}
    for g in gens:
        cat = f"{g['condition']}/{g['split']}/T={float(g['temperature'])}"
        tok_counts[cat] = tok_counts.get(cat, 0) + int(g["realized_length"])
        pm_sets.setdefault(cat, set()).add(g["prompt_id"])
    tok_counts["primary_train_total"] = 2_400_000
    tok_counts["primary_train_T=0.0"] = 1_200_000
    tok_counts["primary_train_T=0.7"] = 1_200_000

    m1_sha = _write_json(
        pb_root / "m1_audit.json",
        {
            "scale": "full",
            "status": "passed",
            "long_condition": "long_thinking",
            "requirements": {
                "primary_train_floor": True,
                "primary_train_ceiling": True,
                "each_primary_temperature_floor": True,
                "long_dev_generations": True,
                "long_test_generations": True,
            },
            "floors": {"train_tokens": 2_000_000, "max_train_tokens": 4_000_000},
            "token_counts": tok_counts,
            "prompt_counts": {k: len(v) for k, v in pm_sets.items()},
            "shards": [str(dummy_shard)],
            "source_shards": [{"path": str(dummy_shard.resolve()), "sha256": d_sha, "identity_sha256": "x" * 64}],
            "head_reconstruction": [
                {"shard": str(dummy_shard.resolve()), "round_idx": 0, "status": "passed"},
                {"shard": str(dummy_shard.resolve()), "round_idx": 1, "status": "passed"},
            ],
            "generations": gens,
        },
    )

    # 4. M1b diagnostic
    diag_sha = _write_json(
        pb_root / "diagnostic_full.json",
        {
            "scale": "full",
            "status": "measured",
            "audit_provenance": {"audit_sha256": m1_sha},
            "groups": [{"condition": "standard_nonthinking", "temperature": 0.7}],
            "feature_shift": {"long_thinking": {"variance_ratio": 1.15}},
        },
    )

    # 5. M2 primary offline fit
    (pb_root / "m2").mkdir(parents=True, exist_ok=True)
    torch.save({"stage": "dense"}, pb_root / "m2" / "dense.pt")
    torch.save({"stage": "dense_sparse"}, pb_root / "m2" / "dense_sparse.pt")
    _write_json(
        pb_root / "m2.json",
        {
            "scale": "full",
            "status": "passed",
            "training_distribution": "standard_nonthinking, greedy or model-card T=.7 topk20 topp.8 minp0 contexts; unwarped T_train=beta_train=1",
            "relative_dev_improvement": 0.085,
            "under_one_percent_stop_tuning": False,
            "trace_index": {
                "primary": {
                    "train": {"generated_output_tokens": 2_400_000, "prompts": 6800, "shards": 13600, "blocks": 300000},
                    "dev": {"generated_output_tokens": 40_000, "prompts": 64, "shards": 128, "blocks": 6000},
                    "test": {"generated_output_tokens": 80_000, "prompts": 128, "shards": 256, "blocks": 12000},
                },
                "primary_train_tokens_by_temperature": {"0.0": 1_200_000, "0.7": 1_200_000},
                "excluded_generated_tokens": {
                    "train/standard_nonthinking/T=1.0": 50_000,
                    "dev/long_thinking/T=0.6": 32_768,
                    "test/long_thinking/T=0.6": 98_304,
                },
                "source_fingerprint": "f" * 64,
            },
            "stages": {
                "dense": {
                    "dev": {"weighted_mean_kl": 0.42, "live_full_diagnostic": {"logical_slots": 64}},
                    "test": {"weighted_mean_kl": 0.43, "live_full_diagnostic": {"logical_slots": 64}},
                },
                "dense_sparse": {
                    "dev": {"weighted_mean_kl": 0.39, "live_full_diagnostic": {"logical_slots": 64}},
                    "test": {"weighted_mean_kl": 0.40, "live_full_diagnostic": {"logical_slots": 64}},
                    "parameters": {
                        "W": {"norm": 1.5},
                        "A": {"norm": 0.8},
                        "b": {"norm": 0.25},
                    },
                    "sparse_collision_statistics": {"collision_count": 120, "unique_observed_keys": 50000},
                },
            },
        },
    )

    # 6 & 7. Build M3 (4 arms x 1088 rows) and M4 (9 full-scale arms x 1088 rows, reusing plain_dflash & online_bigram)
    helper = bar.PostGateEAuditor(repo=IWT, artifacts=pb_root, scale="full", seed=2026)
    m3_eval_index: dict[tuple[str, str, float, int], tuple[Path, str, dict[str, Any]]] = {}

    for stage, arms in (("m3", bar.M3_ARMS), ("m4", apb.M4_ARMS_FULL)):
        stg_dir = pb_root / stage
        sel_settings: dict[str, dict[str, Any]] = {}
        cands_map: dict[str, list[dict[str, Any]]] = {}
        for a in arms:
            if a == "plain_dflash":
                sel = {}
            elif a == "online_bigram":
                sel = {"lambda_tr": 0.2, "alpha": 1.0, "k_ret": None}
            elif a in ("dense", "dense_sparse", "offline_static"):
                sel = {"beta": 0.5}
            else:
                stride = 5 if a == "offline_online_stride5" else 1
                wm = (
                    "rejection_relative"
                    if a == "online_rejection_relative"
                    else ("cutoff" if a == "online_cutoff" else "exponential")
                )
                sel = {
                    "beta": 1.0,
                    "lr": 1e-4,
                    "update_stride": stride,
                    "lambda_l2": 0.01,
                    "lambda_l1_b": 1e-6,
                    "lambda_l1_s": 1e-5,
                    "weighting_mode": wm,
                }
            sel_settings[a] = sel
            cands_map[a] = [
                {
                    "settings": sel,
                    "selection_score": 1.0 if a == "plain_dflash" else 1.05,
                    "repetition_excluded_matched_pairs": 0,
                }
            ]

        dev_pids = [f"std-dev-{i:03d}" for i in range(8)] + [f"long-dev-{i:03d}" for i in range(32)]
        # Write sharded dev tune (2 ranks)
        tune_jobs_r0, tune_jobs_r1 = {}, {}
        for idx_p, dpid in enumerate(dev_pids):
            rk = idx_p % 2
            cond_d = "long_thinking" if dpid.startswith("long-") else "standard_nonthinking"
            temp_d = 0.6 if cond_d.startswith("long_") else 0.7
            warp_d = warp_th if cond_d.startswith("long_") else warp_nt
            for a in arms:
                d_row = _make_ClosedLoopRow(
                    stage=stage,
                    arm=a,
                    prompt_id=dpid,
                    condition=cond_d,
                    temperature=temp_d,
                    warp=warp_d,
                    seed=2026,
                    setting=sel_settings[a],
                    tps=100.0 if a == "plain_dflash" else 105.0,
                )
                d_row["split"] = "dev"
                rp = stg_dir / "tune" / f"rank{rk}" / f"{dpid}-{a}.json"
                rsha = _write_json(rp, d_row)
                entry = {
                    "status": "complete",
                    "identity": {"timing_session": f"sess_tune_{stage}-rank{rk}"},
                    "result_path": str(rp),
                    "result_sha256": rsha,
                }
                (tune_jobs_r0 if rk == 0 else tune_jobs_r1)[f"{dpid}::{a}"] = entry

        _write_json(
            stg_dir / "tune" / "sharding_plan.json",
            {"stage": stage, "scale": "full", "world_size": 2, "logical_session": f"sess_tune_{stage}"},
        )
        for rk, cpus in ((0, list(range(0, 16))), (1, list(range(16, 32)))):
            _write_json(
                stg_dir / "tune" / "sessions" / f"sess_tune_{stage}-rank{rk}.json",
                {"rank": rk, "physical_gpu": rk, "numa_node": 0, "cpu_affinity": cpus},
            )
        _write_json(stg_dir / "tune" / "manifest-rank0.json", {"jobs": tune_jobs_r0})
        _write_json(stg_dir / "tune" / "manifest-rank1.json", {"jobs": tune_jobs_r1})

        settings_payload = {
            "frozen": True,
            "stage": stage,
            "scale": "full",
            "sharding_mode": "sharded_2gpu",
            "comparison_session": f"sess_tune_{stage}",
            "development_prompt_ids": dev_pids,
            "selected": sel_settings,
            "candidates": cands_map,
        }
        _write_json(stg_dir / "settings.json", settings_payload)
        frozen_hash = bar.fingerprint(settings_payload)

        # Write 1,088 evaluate rows per arm (`128*(1+3+3) + 64*3`)
        eval_jobs = {}
        eval_rows = []
        test_specs: list[tuple[str, str, float, dict[str, Any], int]] = []
        for i in range(128):
            pid = f"std-test-{i:03d}"
            test_specs.append((pid, "standard_nonthinking", 0.0, warp_pure, 2026))
            for s_off in range(3):
                test_specs.append((pid, "standard_nonthinking", 0.7, warp_nt, 2026 + s_off))
                test_specs.append((pid, "standard_nonthinking", 1.0, warp_pure, 2026 + s_off))
        for i in range(64):
            pid = f"long-test-{i:03d}"
            for s_off in range(3):
                test_specs.append((pid, "long_thinking", 0.6, warp_th, 2026 + s_off))

        assert len(test_specs) == 1088
        for pid, cond, temp, wp, sd in test_specs:
            for a in arms:
                reused_info = None
                if stage == "m4" and a in ("plain_dflash", "online_bigram"):
                    src_p, src_sha, _ = m3_eval_index[(a, pid, temp, sd)]
                    reused_info = (src_p, src_sha)
                erow = _make_ClosedLoopRow(
                    stage=stage,
                    arm=a,
                    prompt_id=pid,
                    condition=cond,
                    temperature=temp,
                    warp=wp,
                    seed=sd,
                    setting=sel_settings[a],
                    tps=110.0 + (5.0 if a != "plain_dflash" else 0.0),
                    acc_len=4.0 + (0.2 if a != "plain_dflash" else 0.0),
                    reused_from=reused_info,
                )
                rp = stg_dir / "evaluate" / "rank0" / f"{pid}-{temp}-{sd}-{a}.json"
                rsha = _write_json(rp, erow)
                if stage == "m3" and a in ("plain_dflash", "online_bigram"):
                    m3_eval_index[(a, pid, temp, sd)] = (rp, rsha, erow)
                eval_jobs[f"{pid}::{temp}::{sd}::{a}"] = {
                    "status": "complete",
                    "identity": {"frozen_settings_hash": frozen_hash},
                    "result_path": str(rp),
                    "result_sha256": rsha,
                }
                eval_rows.append(erow)

        _write_json(stg_dir / "evaluate" / "manifest-rank0.json", {"jobs": eval_jobs})
        eval_groups = helper.recompute_group_rows(eval_rows, isolated_timing=False, exclude_repetition=True)

        # Write sharded timing (2 ranks, 2 prompts each x 2 repeats + 1 profile bundle)
        timed_rows, prof_rows = [], []
        timing_jobs_by_rk: dict[int, dict[str, Any]] = {0: {}, 1: {}}
        sess_paths = []
        rank_pids = {0: ["std-test-000", "long-test-000"], 1: ["std-test-001", "long-test-001"]}
        _write_json(
            stg_dir / "timing" / "sharding_plan.json",
            {
                "world_size": 2,
                "ranks": [
                    {"rank": 0, "prompt_ids": rank_pids[0]},
                    {"rank": 1, "prompt_ids": rank_pids[1]},
                ],
            },
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
                        trow = _make_ClosedLoopRow(
                            stage=stage,
                            arm=a,
                            prompt_id=pid,
                            condition=cond_t,
                            temperature=temp_t,
                            warp=warp_t,
                            seed=2026,
                            setting=sel_settings[a],
                            tps=115.0 + (4.0 if a != "plain_dflash" else 0.0),
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
                        if is_prof:
                            prof_rows.append(trow)
                        else:
                            timed_rows.append(trow)
                    bundles[f"{pid}::b{b_idx}"] = {
                        "status": "complete",
                        "result_paths": rpaths,
                        "result_sha256": rshas,
                    }
            _write_json(stg_dir / "timing" / f"manifest-rank{rk}.json", {"jobs": timing_jobs_by_rk[rk]})
            sp = stg_dir / "timing" / "sessions" / f"sess_tm_{stage}-rank{rk}.json"
            _write_json(
                sp,
                {
                    "rank": rk,
                    "physical_gpu": rk,
                    "numa_node": 0,
                    "cpu_affinity": list(range(rk * 16, (rk + 1) * 16)),
                    "bundles": bundles,
                },
            )
            sess_paths.append(str(sp))

        enrich_helper = apb.PhaseBFullScaleAuditor(
            repo=IWT, phase_b_root=pb_root, reference_artifacts=ref_dir, scale="full"
        )
        timed_groups = enrich_helper._enrich_timed_groups_with_paired_plain(
            label="build",
            timed_rows=timed_rows,
            recomp_t_groups=helper.recompute_group_rows(
                timed_rows, isolated_timing=True, exclude_repetition=True
            ),
        )
        prof_groups = helper.recompute_group_rows(prof_rows, isolated_timing=True, exclude_repetition=True)
        _write_json(
            stg_dir / "timing_isolation_calibration.json",
            {
                "status": "complete",
                "calibrated_timed_runs": 6 * 2 * len(arms),
                "reconciled_with_sharded_timing": True,
                "groups": [{"arm": "plain_dflash", "isolated_vs_sharded_ratio": 1.004}],
                "by_prompt": [{"prompt_id": "std-test-000", "arm": "plain_dflash"}],
            },
        )
        _write_json(
            stg_dir / "timing.json",
            {
                "scale": "full",
                "stage": stage,
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
        _write_json(stg_dir / "greedy_output_comparisons.json", {"status": "complete", "groups": []})
        if stage == "m4":
            _write_json(
                stg_dir / "feature_context.json",
                {
                    "status": "complete",
                    "feature_group_count": 2,
                    "diagnostic_path": str(pb_root / "diagnostic_full.json"),
                    "diagnostic_sha256": diag_sha,
                    "audit_path": str(pb_root / "m1_audit.json"),
                    "audit_sha256": m1_sha,
                    "calibration_path": str(pb_root / "calibration.json"),
                    "calibration_sha256": cal_sha,
                },
            )
        _write_json(
            stg_dir / "summary.json",
            {
                "scale": "full",
                "stage": stage,
                "status": "complete",
                "isolated_timing_complete": True,
                "greedy_output_comparisons_complete": True,
                "expected_requests": len(eval_rows),
                "completed_requests": len(eval_rows),
                "missing_requests": [],
                "groups": eval_groups,
                "baseline_reuse": {
                    "reused_requests": (2 * 1088) if stage == "m4" else 0,
                    "fresh_baseline_requests": 0,
                },
            },
        )

    _write_json(
        pb_root / "full_scale_pipeline.json",
        {
            "suite": "full_scale",
            "scale": "full",
            "status": "complete",
            "config_fingerprint": "fp_full_1234",
            "workspace": str(pb_root),
        },
    )
    _write_json(
        pb_root / "phase_b_summary.json",
        {
            "status": "complete",
            "scale": "full",
            "executed_suites": ["full_scale"],
            "upstream_reference_immutability": {"status": "intact", "verified_files": 5},
            "suites": {"full_scale": {"config_fingerprint": "fp_full_1234"}},
        },
    )
    return pb_root


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_real_phase_b_suite_smoke_all_and_strict_auditor(tmp_path: Path):
    """Run real `edraft.phase_b_suite --scale smoke --suite all` and verify strict 0-drift audit + markdown."""
    from edraft import phase_b_suite

    ref_dir = tmp_path / "frozen_ref"
    ref_dir.mkdir(parents=True)
    (ref_dir / "m4").mkdir(parents=True)
    _write_json(ref_dir / "milestones.json", {"M4": {"status": "passed", "artifact_sha256": {}}})
    _write_json(ref_dir / "m4" / "summary.json", {"status": "complete", "scale": "reference"})

    pb_out = tmp_path / "phase_b_smoke_out"
    phase_b_suite.main(
        [
            "--scale",
            "smoke",
            "--suite",
            "all",
            "--output",
            str(pb_out),
            "--reference_artifacts",
            str(ref_dir),
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

    json_out = tmp_path / "audit_smoke.json"
    md_out = tmp_path / "tables_smoke.md"
    rc = apb.main(
        [
            "--repo",
            str(IWT),
            "--phase_b_root",
            str(pb_out),
            "--reference_artifacts",
            str(ref_dir),
            "--scale",
            "smoke",
            "--suite",
            "all",
            "--require_complete",
            "--json_out",
            str(json_out),
            "--markdown_out",
            str(md_out),
        ]
    )
    assert rc == 0
    rep = json.loads(json_out.read_text())
    assert rep["status"] == "passed"
    assert rep["failures"] == []
    assert rep["checks"] >= 800
    md_txt = md_out.read_text()
    assert "OnlineBigram" in md_txt
    assert "Full-Vocabulary vs Candidate-Masked Lookahead" in md_txt
    assert "Pure `T_collection=1.0` Offline Fit" in md_txt
    assert "Full-Scale" in md_txt

    # Fault injections on secondary ablations (`bigram_eviction`, `full_vocab_lookahead`, `pure_t1_fit`):
    # (a) Corrupt `bigram_memory_ablation` evicted_prior_pairs_mean -> caught!
    bg_path = pb_out / "reference_secondary_ablations" / "bigram_eviction.json"
    bg_orig = bg_path.read_bytes()
    bg_mut = json.loads(bg_orig)
    bg_mut["bigram_memory_ablation"][0]["evicted_prior_pairs_mean"] += 5.0
    _write_json(bg_path, bg_mut)
    aud_bg = apb.PhaseBFullScaleAuditor(
        repo=IWT, phase_b_root=pb_out, reference_artifacts=ref_dir,
        scale="smoke", suite="bigram_eviction", require_complete=True,
    )
    rep_bg = aud_bg.run_all()
    assert rep_bg["status"] == "failed"
    assert any("bigram_memory_ablation drift" in f for f in rep_bg["failures"])
    bg_path.write_bytes(bg_orig)

    # (b) Mismatched `beta_relax` in `full_vocab_lookahead` `settings.json` -> caught!
    la_rep = json.loads((pb_out / "reference_secondary_ablations" / "full_vocab_lookahead.json").read_text())
    la_set_p = Path(la_rep["artifacts"]["settings_json"])
    la_set_orig = la_set_p.read_bytes()
    la_set_mut = json.loads(la_set_orig)
    la_set_mut["selected"]["lookahead_static_full"]["beta_relax"] = 2.5
    _write_json(la_set_p, la_set_mut)
    aud_la = apb.PhaseBFullScaleAuditor(
        repo=IWT, phase_b_root=pb_out, reference_artifacts=ref_dir,
        scale="smoke", suite="full_vocab_lookahead", require_complete=True,
    )
    rep_la = aud_la.run_all()
    assert rep_la["status"] == "failed"
    assert any("beta_relax differs" in f for f in rep_la["failures"])
    la_set_p.write_bytes(la_set_orig)

    # (c) Strip `ablation_warning` in `pure_t1_fit.json` -> caught!
    pt1_path = pb_out / "reference_secondary_ablations" / "pure_t1_fit.json"
    pt1_orig = pt1_path.read_bytes()
    pt1_mut = json.loads(pt1_orig)
    pt1_mut["ablation_warning"] = ""
    _write_json(pt1_path, pt1_mut)
    aud_pt1 = apb.PhaseBFullScaleAuditor(
        repo=IWT, phase_b_root=pb_out, reference_artifacts=ref_dir,
        scale="smoke", suite="pure_t1_fit", require_complete=True,
    )
    rep_pt1 = aud_pt1.run_all()
    assert rep_pt1["status"] == "failed"
    assert any("ablation_warning" in f for f in rep_pt1["failures"])
    pt1_path.write_bytes(pt1_orig)


def test_full_scale_validator_and_all_12_fault_injections(tmp_path: Path):
    """Verify full-scale `[2M, 4M]` tokens, `128x3` std + `64x3` long, 9-arm M4, and 12 adversarial faults."""
    ref_dir = tmp_path / "ref_artifacts"
    pb_root = _build_synthetic_full_scale_workspace(tmp_path, ref_dir)

    # 0. Valid full-scale workspace passes with 0 failures!
    aud0 = apb.PhaseBFullScaleAuditor(
        repo=IWT,
        phase_b_root=pb_root,
        reference_artifacts=ref_dir,
        scale="full",
        suite="full_scale",
        require_complete=True,
    )
    rep0 = aud0.run_all()
    assert rep0["status"] == "passed", rep0["failures"]
    assert rep0["checks"] > 20_000
    # Sparse 16K+ long offset bin (`0 < 8` prompts reaching 16K+) is explicitly flagged!
    m4_sparse = rep0["suites"]["full_scale"]["m4_closed_loop"]["sparse_long_offset_bins"]
    assert any(b["bin"] == "16K+" and b["conclusion_available"] is False for b in m4_sparse)

    def _run_fault(mutator, touched_relpaths: list[str] | None = None) -> list[str]:
        saved_pb: dict[Path, bytes] = {}
        saved_ref: dict[Path, bytes] = {}
        created_dirs: list[Path] = []
        for rel in (touched_relpaths or []):
            p = pb_root / rel
            if p.is_file():
                saved_pb[p] = p.read_bytes()
        for rp in ref_dir.rglob("*.json"):
            saved_ref[rp] = rp.read_bytes()
        try:
            mutator(pb_root, ref_dir, created_dirs)
            aud = apb.PhaseBFullScaleAuditor(
                repo=IWT,
                phase_b_root=pb_root,
                reference_artifacts=ref_dir,
                scale="full",
                suite="full_scale",
                require_complete=True,
            )
            res = aud.run_all()
            assert res["status"] == "failed"
            return res["failures"]
        finally:
            for p, data in saved_pb.items():
                p.write_bytes(data)
            for p, data in saved_ref.items():
                p.write_bytes(data)
            for d in created_dirs:
                if d.exists():
                    shutil.rmtree(d)

    # Fault 1: Primary train tokens < 2,000,000 (`1,900,000`)
    def _f1(w: Path, _r: Path, _cd: list[Path]):
        m1 = json.loads((w / "m1_audit.json").read_text())
        m1["generations"][0]["realized_length"] = 700_000
        m1["token_counts"]["standard_nonthinking/train/T=0.0"] = 700_000
        m1["token_counts"]["primary_train_T=0.0"] = 700_000
        m1["token_counts"]["primary_train_total"] = 1_900_000
        _write_json(w / "m1_audit.json", m1)

    assert any("outside mandatory [2,000,000, 4,000,000]" in f for f in _run_fault(_f1, ["m1_audit.json"]))

    # Fault 2: Primary train tokens > 4,000,000 (`4,200,000`)
    def _f2(w: Path, _r: Path, _cd: list[Path]):
        m1 = json.loads((w / "m1_audit.json").read_text())
        m1["generations"][0]["realized_length"] = 3_000_000
        m1["token_counts"]["standard_nonthinking/train/T=0.0"] = 3_000_000
        m1["token_counts"]["primary_train_T=0.0"] = 3_000_000
        m1["token_counts"]["primary_train_total"] = 4_200_000
        _write_json(w / "m1_audit.json", m1)

    assert any("outside mandatory [2,000,000, 4,000,000]" in f for f in _run_fault(_f2, ["m1_audit.json"]))

    # Fault 3: Unbalanced primary temperature (< 1M on T=0.0 even when total == 2.4M)
    def _f3(w: Path, _r: Path, _cd: list[Path]):
        m1 = json.loads((w / "m1_audit.json").read_text())
        m1["generations"][0]["realized_length"] = 900_000
        m1["generations"][1]["realized_length"] = 1_500_000
        m1["token_counts"]["standard_nonthinking/train/T=0.0"] = 900_000
        m1["token_counts"]["standard_nonthinking/train/T=0.7"] = 1_500_000
        m1["token_counts"]["primary_train_T=0.0"] = 900_000
        m1["token_counts"]["primary_train_T=0.7"] = 1_500_000
        _write_json(w / "m1_audit.json", m1)

    assert any("each primary temperature requires >=1,000,000" in f for f in _run_fault(_f3, ["m1_audit.json"]))

    # Fault 4 (`PHASEB-AUDIT-BUG-3`): Pure T=1.0 or long_* tokens leaked into primary M2 `trace_index`
    def _f4(w: Path, _r: Path, _cd: list[Path]):
        m2 = json.loads((w / "m2.json").read_text())
        m2["trace_index"]["primary"]["train"]["generated_output_tokens"] = 2_450_000
        _write_json(w / "m2.json", m2)

    assert any("leaked into primary M2 fit" in f for f in _run_fault(_f4, ["m2.json"]))

    # Fault 4b (`PHASEB-AUDIT-BUG-1`): Fewer than 40 `development_prompt_ids` in `settings.json` -> caught!
    def _f4b(w: Path, _r: Path, _cd: list[Path]):
        s3 = json.loads((w / "m3" / "settings.json").read_text())
        s3["development_prompt_ids"] = s3["development_prompt_ids"][:20]
        _write_json(w / "m3" / "settings.json", s3)

    assert any("requires >=40 dev prompts" in f for f in _run_fault(_f4b, ["m3/settings.json"]))

    # Fault 5: Insufficient long_dev generations (< 32) in m1_audit.json
    def _f5(w: Path, _r: Path, _cd: list[Path]):
        m1 = json.loads((w / "m1_audit.json").read_text())
        m1["generations"] = [g for g in m1["generations"] if g["prompt_id"] != "long-dev-031"]
        m1["token_counts"]["long_thinking/dev/T=0.6"] -= 1024
        m1["prompt_counts"]["long_thinking/dev/T=0.6"] = 31
        _write_json(w / "m1_audit.json", m1)

    assert any("long dev generations=31 < 32" in f for f in _run_fault(_f5, ["m1_audit.json"]))

    # Fault 6: Missing full-scale M4 arm (`online_cutoff` omitted from settings.json)
    def _f6(w: Path, _r: Path, _cd: list[Path]):
        s4 = json.loads((w / "m4" / "settings.json").read_text())
        s4["selected"].pop("online_cutoff")
        _write_json(w / "m4" / "settings.json", s4)

    assert any("selected arms" in f for f in _run_fault(_f6, ["m4/settings.json"]))

    # Fault 7: online_dbS nonzero dW norm -> caught!
    def _f7(w: Path, _r: Path, _cd: list[Path]):
        rp = w / "m4" / "evaluate" / "rank0" / "long-test-000-0.6-2026-online_dbS.json"
        rw = json.loads(rp.read_text())
        rw["online"]["dW_norm"] = 0.01
        new_sha = _write_json(rp, rw)
        mf = json.loads((w / "m4" / "evaluate" / "manifest-rank0.json").read_text())
        mf["jobs"]["long-test-000::0.6::2026::online_dbS"]["result_sha256"] = new_sha
        _write_json(w / "m4" / "evaluate" / "manifest-rank0.json", mf)

    assert any(
        "online_dbS must keep dW==0" in f
        for f in _run_fault(
            _f7,
            [
                "m4/evaluate/rank0/long-test-000-0.6-2026-online_dbS.json",
                "m4/evaluate/manifest-rank0.json",
            ],
        )
    )

    # Fault 7b: Long test prompt has duplicate seed instead of 3 distinct stochastic seeds -> caught!
    def _f7b(w: Path, _r: Path, _cd: list[Path]):
        rp = w / "m4" / "evaluate" / "rank0" / "long-test-000-0.6-2028-offline_static.json"
        rw = json.loads(rp.read_text())
        rw["seed"] = 2026
        new_sha = _write_json(rp, rw)
        mf = json.loads((w / "m4" / "evaluate" / "manifest-rank0.json").read_text())
        mf["jobs"]["long-test-000::0.6::2028::offline_static"]["result_sha256"] = new_sha
        _write_json(w / "m4" / "evaluate" / "manifest-rank0.json", mf)

    assert any(
        "expected >=64 long test prompts x 3 stochastic seeds" in f
        for f in _run_fault(
            _f7b,
            [
                "m4/evaluate/rank0/long-test-000-0.6-2028-offline_static.json",
                "m4/evaluate/manifest-rank0.json",
            ],
        )
    )

    # Fault 8: zero_offline_online_only nonzero offline parameter_norms -> caught!
    def _f8(w: Path, _r: Path, _cd: list[Path]):
        rp = w / "m4" / "evaluate" / "rank0" / "long-test-000-0.6-2026-zero_offline_online_only.json"
        rw = json.loads(rp.read_text())
        rw["parameter_norms"]["W"] = 0.5
        new_sha = _write_json(rp, rw)
        mf = json.loads((w / "m4" / "evaluate" / "manifest-rank0.json").read_text())
        mf["jobs"]["long-test-000::0.6::2026::zero_offline_online_only"]["result_sha256"] = new_sha
        _write_json(w / "m4" / "evaluate" / "manifest-rank0.json", mf)

    assert any(
        "zero_offline_online_only must have exact 0.0" in f
        for f in _run_fault(
            _f8,
            [
                "m4/evaluate/rank0/long-test-000-0.6-2026-zero_offline_online_only.json",
                "m4/evaluate/manifest-rank0.json",
            ],
        )
    )

    # Fault 9: offline_online_stride5 buffered update (`single_round_no_buffering=False`) -> caught!
    def _f9(w: Path, _r: Path, _cd: list[Path]):
        rp = w / "m4" / "evaluate" / "rank0" / "long-test-000-0.6-2026-offline_online_stride5.json"
        rw = json.loads(rp.read_text())
        rw["online_invariants"]["single_round_no_buffering"] = False
        new_sha = _write_json(rp, rw)
        mf = json.loads((w / "m4" / "evaluate" / "manifest-rank0.json").read_text())
        mf["jobs"]["long-test-000::0.6::2026::offline_online_stride5"]["result_sha256"] = new_sha
        _write_json(w / "m4" / "evaluate" / "manifest-rank0.json", mf)

    assert any(
        "single-round unbuffered updates" in f
        for f in _run_fault(
            _f9,
            [
                "m4/evaluate/rank0/long-test-000-0.6-2026-offline_online_stride5.json",
                "m4/evaluate/manifest-rank0.json",
            ],
        )
    )

    # Fault 10: M4 timing row illegally reusing M3 baseline log instead of re-measuring -> caught!
    def _f10(w: Path, _r: Path, _cd: list[Path]):
        tp = w / "m4" / "timing" / "rank0" / "std-test-000-b0-plain_dflash.json"
        tr = json.loads(tp.read_text())
        src_p = w / "m3" / "evaluate" / "rank0" / "std-test-000-0.7-2026-plain_dflash.json"
        tr["baseline_reuse"] = {
            "status": "reused",
            "source_stage": "m3",
            "source_result_path": str(src_p),
            "source_result_sha256": bar.sha256_file(src_p),
        }
        for pf in ("elapsed_seconds", "generation_loop_elapsed_seconds", "tokens_per_second", "phase_ms", "peak_allocated_bytes", "peak_reserved_bytes"):
            tr[pf] = None
        new_sha = _write_json(tp, tr)
        tmf = json.loads((w / "m4" / "timing" / "manifest-rank0.json").read_text())
        tmf["jobs"]["std-test-000::b0::plain_dflash"]["result_sha256"] = new_sha
        _write_json(w / "m4" / "timing" / "manifest-rank0.json", tmf)
        sp = w / "m4" / "timing" / "sessions" / "sess_tm_m4-rank0.json"
        sdata = json.loads(sp.read_text())
        sdata["bundles"]["std-test-000::b0"]["result_sha256"][str(tp)] = new_sha
        _write_json(sp, sdata)

    assert any(
        "illegally reused M3 log" in f
        for f in _run_fault(
            _f10,
            [
                "m4/timing/rank0/std-test-000-b0-plain_dflash.json",
                "m4/timing/manifest-rank0.json",
                "m4/timing/sessions/sess_tm_m4-rank0.json",
            ],
        )
    )

    # Fault 11: Upstream reference `milestones.json` SHA256 mismatch -> caught!
    def _f11(_w: Path, r: Path, _cd: list[Path]):
        _write_json(
            r / "milestones.json",
            {"M4": {"status": "passed", "artifact_sha256": {"summary": "0" * 64}}},
        )

    assert any("upstream reference m4/summary.json SHA256" in f for f in _run_fault(_f11))

    # Fault 13 (`IO-BUG-2`): Split-brain empty `m0/` directory in full_ws -> caught!
    def _f13(w: Path, _r: Path, cd: list[Path]):
        empty_m0 = w / "m0"
        empty_m0.mkdir(parents=True, exist_ok=True)
        cd.append(empty_m0)

    assert any("split-brain/incomplete m0/ dir" in f for f in _run_fault(_f13))

    # Fault 14 (`EVAL-BUG-3` / `PHASEB-AUDIT-BUG-2`): Unreconciled `timing_isolation_calibration.json` on sharded M4 -> caught under `require_complete=True`, allowed under `allow_partial=True`!
    def _f14(w: Path, _r: Path, _cd: list[Path]):
        cal4 = json.loads((w / "m4" / "timing_isolation_calibration.json").read_text())
        cal4["status"] = "calibration_only_unreconciled"
        _write_json(w / "m4" / "timing_isolation_calibration.json", cal4)

    assert any(
        "unreconciled FastPath calibration" in f
        for f in _run_fault(_f14, ["m4/timing_isolation_calibration.json"])
    )

    # Under allow_partial=True (require_complete=False), `calibration_only_unreconciled` is accepted cleanly!
    cal4_p = pb_root / "m4" / "timing_isolation_calibration.json"
    cal4_orig = cal4_p.read_bytes()
    try:
        cal4_tmp = json.loads(cal4_orig)
        cal4_tmp["status"] = "calibration_only_unreconciled"
        _write_json(cal4_p, cal4_tmp)
        aud_part = apb.PhaseBFullScaleAuditor(
            repo=IWT, phase_b_root=pb_root, reference_artifacts=ref_dir,
            scale="full", suite="full_scale", allow_partial=True, require_complete=False,
        )
        rep_part = aud_part.run_all()
        assert rep_part["status"] == "passed", rep_part["failures"]
    finally:
        cal4_p.write_bytes(cal4_orig)

    # Fault 14b (`PHASEB-AUDIT-BUG-2`): `status=='complete'` with `reconciled_with_sharded_timing=False` -> caught!
    def _f14b(w: Path, _r: Path, _cd: list[Path]):
        cal4 = json.loads((w / "m4" / "timing_isolation_calibration.json").read_text())
        cal4["reconciled_with_sharded_timing"] = False
        _write_json(w / "m4" / "timing_isolation_calibration.json", cal4)

    assert any(
        "missing sharded_vs_isolated_1gpu reconciliation payload" in f
        for f in _run_fault(_f14b, ["m4/timing_isolation_calibration.json"])
    )

    # Fault 12: phase_b_root colliding with frozen reference_artifacts -> caught!
    aud_collide = apb.PhaseBFullScaleAuditor(
        repo=IWT,
        phase_b_root=ref_dir,
        reference_artifacts=ref_dir,
        scale="full",
        suite="full_scale",
        allow_partial=True,
    )
    rep_collide = aud_collide.run_all()
    assert rep_collide["status"] == "failed"
    assert any("must NOT collide with frozen reference_artifacts" in f for f in rep_collide["failures"])


def test_phase_b_branch_0afc795_ast_and_upstream_reference_conformance():
    """Verify `feat/phase-b-full-scale` (`0afc795d`) passes AST audit & preserves all 27 live reference files."""
    if not IWT.is_dir():
        pytest.skip("Integrator Phase-B worktree not present")
    live_art = Path("/home/catid/drafter/artifacts")
    ia = bia.build_implementation_audit(IWT, live_art)
    errs = bia.verify_implementation_audit(ia, IWT, live_art)
    assert errs == [], errs
    assert len(ia["source_files_sha256"]) >= 64  # includes phase_b_suite.py + new Phase-B test files!

    aud = apb.PhaseBFullScaleAuditor(
        repo=IWT,
        phase_b_root=Path("/tmp/edraft_reporting/nonexistent_phase_b_partial"),
        reference_artifacts=live_art,
        scale="full",
        suite="all",
        allow_partial=True,
    )
    rep = aud.run_all()
    assert rep["status"] == "passed", rep["failures"]
    assert rep["upstream_reference_immutability"]["checked_upstream_files"] >= 27

    # Directly verify PHASEB-AUDIT-BUG-1..3 against real `/home/catid/drafter/artifacts` schemas (`reference`)!
    aud_real_schema = apb.PhaseBFullScaleAuditor(
        repo=IWT,
        phase_b_root=Path("/tmp/edraft_reporting/nonexistent_phase_b_partial"),
        reference_artifacts=live_art,
        scale="reference",
        suite="ref_secondary_ablations",
        require_complete=True,
    )
    s3_real = json.loads((live_art / "m3" / "settings.json").read_text())
    aud_real_schema._audit_dev_tune_selection(
        label="real_m3_tune",
        stage_dir=live_art / "m3",
        settings=s3_real,
        arms=bar.M3_ARMS,
        eff_scale="reference",
    )
    t4_real = json.loads((live_art / "m4" / "timing.json").read_text())
    aud_real_schema._audit_optional_timing_calibration(
        label="real_m4",
        stage_dir=live_art / "m4",
        timing=t4_real,
        eff_scale="reference",
        expected_arms=apb.M4_ARMS_REF,
        timed_rows=[],
    )
    aud_real_schema._audit_full_scale_m2(live_art, "reference")
    assert aud_real_schema.failures == [], aud_real_schema.failures


def test_phaseb_audit_allrep_3seed_guard_full_scale(tmp_path: Path):
    """Regression test for `PHASEB-AUDIT-ALLREP-3SEED-GUARD-1` (`#369`): an uncalibrated M4 arm
    hitting 100% (`192/192`) `repetition_dominated=True` on `long_thinking` (`primary_requests == 0`)
    at `scale='full'` must pass `PhaseBFullScaleAuditor` cleanly."""
    ref_dir = tmp_path / "ref_art"
    ref_dir.mkdir(parents=True)
    _write_json(ref_dir / "milestones.json", {"M4": {"status": "passed", "artifact_sha256": {}}})
    pb_root = _build_synthetic_full_scale_workspace(tmp_path / "pb_full", ref_dir)
    ws = pb_root

    # Mark all `online_dbS` `long_thinking` evaluate rows as `repetition_dominated=True` (`192/192` excluded!)
    m4_eval_dir = ws / "m4" / "evaluate"
    eval_rows = []
    for mf in sorted(m4_eval_dir.glob("manifest-rank*.json")):
        mdata = json.loads(mf.read_text(encoding="utf-8"))
        for entry in mdata["jobs"].values():
            rp = Path(entry["result_path"])
            rw = json.loads(rp.read_text(encoding="utf-8"))
            if rw["arm"] == "online_dbS" and rw["condition"] == "long_thinking":
                rw["generation"]["repetition_dominated"] = True
                # Matching periodic token tail so independent_repetition_dominated agrees!
                rw["output_tokens"] = ([11, 12, 13, 14, 15, 16, 17, 18] * 40)[:300]
                rw["generated_tokens"] = len(rw["output_tokens"])
                _write_json(rp, rw)
                entry["result_sha256"] = apb.sha256_file(rp)
            eval_rows.append(rw)
        _write_json(mf, mdata)

    helper = bar.PostGateEAuditor(repo=IWT, artifacts=ws, scale="full", seed=2026)
    recomp_groups = [
        {k: v for k, v in g.items() if k != "excluded_prompt_ids"}
        for g in helper.recompute_group_rows(eval_rows, isolated_timing=False, exclude_repetition=True)
    ]
    assert any(g["arm"] == "online_dbS" and g["condition"] == "long_thinking" and g["primary_requests"] == 0 for g in recomp_groups)
    sum4_p = ws / "m4" / "summary.json"
    sum4 = json.loads(sum4_p.read_text(encoding="utf-8"))
    sum4["groups"] = recomp_groups
    _write_json(sum4_p, sum4)

    aud = apb.PhaseBFullScaleAuditor(
        repo=IWT,
        phase_b_root=pb_root,
        reference_artifacts=ref_dir,
        scale="full",
        suite="full_scale",
        require_complete=True,
    )
    res = aud.run_all()
    assert res["status"] == "passed", res["failures"]


def test_phaseb_allow_pilot_timing_prompts_subset_isolated_timing_complete_guard(tmp_path: Path):
    """Regression test for `#431` / `#432` (+ `PHASEB-PILOT-FULLSCALE-SUMMARY-STATUS-2`):
    `edraft.phase_b_suite --scale smoke --suite all --timing_prompts 1` sets `isolated_timing_complete=False`
    across `bigram_eviction`, `full_vocab_lookahead`, and `full_scale` (`m3`/`m4`).
    - `allow_pilot=True, require_complete=True` must PASS across all 4 suites (`0` failures).
    - `allow_pilot=False, require_complete=True` must FAIL (`isolated_timing_complete=False != True` + `pilot_complete`).
    """
    from edraft import phase_b_suite

    ref_dir = tmp_path / "frozen_ref_pilot"
    ref_dir.mkdir(parents=True)
    (ref_dir / "m4").mkdir(parents=True)
    _write_json(ref_dir / "milestones.json", {"M4": {"status": "passed", "artifact_sha256": {}}})
    _write_json(ref_dir / "m4" / "summary.json", {"status": "complete", "scale": "reference"})

    pb_pilot = tmp_path / "phase_b_pilot_out"
    phase_b_suite.main(
        [
            "--scale",
            "smoke",
            "--suite",
            "all",
            "--gpus",
            "0",
            "--output",
            str(pb_pilot),
            "--reference_artifacts",
            str(ref_dir),
            "--timing_repeats",
            "2",
            "--timing_warmups",
            "1",
            "--timing_prompts",
            "1",
            "--smoke_tokens",
            "16",
            "--block_shuffle_window",
            "2",
        ]
    )

    bg_rep = json.loads((pb_pilot / "reference_secondary_ablations" / "bigram_eviction.json").read_text(encoding="utf-8"))
    bg_sum = json.loads(Path(bg_rep["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
    assert bg_rep["status"] == "pilot_complete"
    assert bg_sum["isolated_timing_complete"] is False

    res_pilot = apb.PhaseBFullScaleAuditor(
        repo=IWT,
        phase_b_root=pb_pilot,
        reference_artifacts=ref_dir,
        scale="smoke",
        suite="all",
        require_complete=True,
        allow_pilot=True,
    ).run_all()
    assert res_pilot["status"] == "passed", res_pilot["failures"]
    assert res_pilot["failures"] == []

    res_strict = apb.PhaseBFullScaleAuditor(
        repo=IWT,
        phase_b_root=pb_pilot,
        reference_artifacts=ref_dir,
        scale="smoke",
        suite="all",
        require_complete=True,
        allow_pilot=False,
    ).run_all()
    assert res_strict["status"] == "failed"
    assert any("isolated_timing_complete=False != True" in f for f in res_strict["failures"])



