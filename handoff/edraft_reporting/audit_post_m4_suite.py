#!/usr/bin/env python3
"""Deterministic Zero-Drift Post-M4 & M5 Suite Auditor and Markdown Table Renderer.

Audits all four `post_m4_suite` experiment suites (`m2_refit_calibration`,
`online_head_to_head`, `m5_eval`, `fastpath_bench`) plus the master cached
summary (`post_m4_suite_summary.json`) and raw rank/stage workspaces without
mutating any repository or artifact file.

Supports:
  - `--allow_partial`: progressive read-only audit while `post_m4_suite` runs,
    verifying every completed suite + in-progress workspace invariants +
    100% immutability of frozen upstream M0-M4 artifacts, and rendering
    publication-ready Markdown tables for completed suites.
  - `--require_complete`: strict completion gate requiring all 4 suite JSONs +
    `post_m4_suite_summary.json` to exist with `status == 'complete'` (unless
    `--allow_pilot` is set), matching `config_fingerprint`s, and 0 failures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

# Reuse verified bootstrap, repetition-v2, and nested-comparison primitives from `build_and_audit_reports.py`.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_and_audit_reports import (  # noqa: E402
    OUTPUT_LABELS,
    SPARSE_LONG_BIN_MIN_PROMPTS,
    _canonicalize_group_list,
    _cond_short,
    _fmt_ci,
    close_nested,
    fingerprint,
    independent_prompt_bootstrap,
    independent_repetition_dominated,
    output_bin_label as independent_output_bin,
    sha256_bytes,
    sha256_file,
)

SUITE_NAMES: tuple[str, ...] = (
    "m2_refit_calibration",
    "online_head_to_head",
    "m5_eval",
    "fastpath_bench",
)

M5_MATCHED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("lookahead_static_candidate", "dense_candidate", "frozen_C0_candidate_masked"),
    ("lookahead_rerelax_candidate", "dense_candidate", "frozen_C0_candidate_masked"),
    ("lookahead_static_full", "dense_full", "full_vocabulary"),
    ("lookahead_rerelax_full", "dense_full", "full_vocabulary"),
    ("exact_observed_pairs_online", "signed_feature_hash_online", "online_sparse_representation"),
    ("online_buffered_stride5", "offline_online_stride5", "online_stride5_update_mode"),
    ("online_bigram_count_gated", "online_bigram", "online_bigram_count_gate"),
    ("online_dbS_zero_init", "online_bigram_count_gated", "online_zero_init_dbS_vs_count_gated_bigram"),
    ("zero_offline_online_only", "offline_online_stride1", "online_zero_vs_checkpoint_offline_init"),
    ("online_dbS_zero_init", "online_dbS", "online_dbS_zero_vs_checkpoint_offline_init"),
)

M5_ONLINE_HEAD_TO_HEAD_PAIRS: tuple[tuple[str, str], ...] = (
    ("online_bigram", "plain_dflash"),
    ("online_bigram_count_gated", "plain_dflash"),
    ("offline_online_stride1", "plain_dflash"),
    ("zero_offline_online_only", "plain_dflash"),
    ("online_dbS", "plain_dflash"),
    ("online_dbS_zero_init", "plain_dflash"),
    ("online_bigram_count_gated", "online_bigram"),
    ("zero_offline_online_only", "online_bigram"),
    ("zero_offline_online_only", "online_bigram_count_gated"),
    ("zero_offline_online_only", "offline_online_stride1"),
    ("zero_offline_online_only", "online_dbS_zero_init"),
    ("online_dbS_zero_init", "online_bigram"),
    ("online_dbS_zero_init", "online_bigram_count_gated"),
    ("online_dbS", "online_bigram_count_gated"),
)

MATCHED_C0_CANDIDATE_ARMS: tuple[str, ...] = (
    "dense_candidate",
    "lookahead_static_candidate",
    "lookahead_rerelax_candidate",
)

LOOKAHEAD_ARMS: tuple[str, ...] = (
    "lookahead_static_candidate",
    "lookahead_rerelax_candidate",
    "lookahead_static_full",
    "lookahead_rerelax_full",
)

UPSTREAM_M0_M4_IMMUTABLE_RELPATHS: tuple[str, ...] = (
    "smoke.json",
    "m0.json",
    "m0_numerics.json",
    "calibration.json",
    "h_proj.pt",
    "token_features.pt",
    "prompts.json",
    "m1_audit.json",
    "diagnostic_reference.json",
    "m2.json",
    "m2/dense.pt",
    "m2/dense_sparse.pt",
    "m3/settings.json",
    "m3/summary.json",
    "m3/timing.json",
    "m3/greedy_output_comparisons.json",
    "m4/settings.json",
    "m4/summary.json",
    "m4/timing.json",
    "m4/timing_isolation_calibration.json",
    "m4/feature_context.json",
    "m4/greedy_output_comparisons.json",
)


def _row_rep_flag(row: dict[str, Any]) -> bool:
    if "repetition_dominated" in row:
        return bool(row["repetition_dominated"])
    return bool(row.get("generation", {}).get("repetition_dominated", False))


def _is_rep_excluded(row: dict[str, Any]) -> bool:
    return bool(str(row.get("condition", "")).startswith("long_") and _row_rep_flag(row))


def _block_committed_count(block: dict[str, Any]) -> int:
    if block.get("committed_tokens") is not None:
        return len(block["committed_tokens"])
    return int(block["accepted_length"]) + (
        1
        if block.get("replacement_token") is not None or block.get("bonus_token") is not None
        else 0
    )


def _all_offset_bin_stats(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute per-output-bin stats across all 5 `OUTPUT_LABELS` in a single fast pass over `row['blocks']`."""
    blocks = row.get("blocks") or []
    acc_lists: dict[str, list[float]] = {lbl: [] for lbl in OUTPUT_LABELS}
    yes_map: dict[str, int] = {lbl: 0 for lbl in OUTPUT_LABELS}
    tot_map: dict[str, int] = {lbl: 0 for lbl in OUTPUT_LABELS}
    committed_map: dict[str, int] = {lbl: 0 for lbl in OUTPUT_LABELS}
    elapsed_lists: dict[str, list[float]] = {lbl: [] for lbl in OUTPUT_LABELS}
    is_prof = bool(row.get("phase_profile"))

    for b in blocks:
        accepted = int(b["accepted_length"])
        base_off = int(b["absolute_output_offset"])
        b_lbl = independent_output_bin(base_off)
        acc_lists[b_lbl].append(float(accepted))
        r_el = b.get("round_elapsed_seconds")
        if r_el is not None:
            committed_map[b_lbl] += _block_committed_count(b)
            elapsed_lists[b_lbl].append(float(r_el))
        n_draft = len(b["draft_tokens"])
        if n_draft > 0:
            end_lbl = independent_output_bin(base_off + n_draft - 1)
            if b_lbl == end_lbl:
                yes_map[b_lbl] += min(max(accepted, 0), n_draft)
                tot_map[b_lbl] += n_draft
            else:
                for i in range(n_draft):
                    s_lbl = independent_output_bin(base_off + i)
                    yes_map[s_lbl] += int(i < accepted)
                    tot_map[s_lbl] += 1

    out: dict[str, dict[str, Any]] = {}
    for lbl in OUTPUT_LABELS:
        a_lst = acc_lists[lbl]
        e_lst = elapsed_lists[lbl]
        tot = tot_map[lbl]
        yes = yes_map[lbl]
        comm = committed_map[lbl]
        el_sum = sum(e_lst)
        out[lbl] = {
            "reached_bin": bool(a_lst) or tot > 0,
            "blocks_in_bin": len(a_lst),
            "slots_in_bin": tot,
            "committed_tokens_in_bin": comm,
            "round_elapsed_seconds_in_bin": el_sum if e_lst else None,
            "accepted_length": statistics.mean(a_lst) if a_lst else None,
            "slot_acceptance_rate": (yes / tot) if tot > 0 else None,
            "tokens_per_second": (comm / el_sum) if (comm > 0 and el_sum > 0.0 and not is_prof) else None,
        }
    return out


def _request_offset_bin_stats(row: dict[str, Any], bin_label: str) -> dict[str, Any]:
    return _all_offset_bin_stats(row)[bin_label]


def recompute_offset_bin_slices(
    rows: list[dict[str, Any]], *, isolated_timing: bool = False, seed: int = 2026
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    row_bin_cache: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("phase_profile"):
            continue
        warp = json.dumps(row["sampling_warp_config"], sort_keys=True)
        grouped[(row["condition"], float(row["temperature"]), warp, row["arm"])].append(row)
    slices: list[dict[str, Any]] = []
    for (condition, temperature, warp, arm), group in sorted(grouped.items()):
        primary = [r for r in group if not _is_rep_excluded(r)]
        for r in primary:
            rid = id(r)
            if rid not in row_bin_cache:
                row_bin_cache[rid] = _all_offset_bin_stats(r)
        bins_out = []
        for bin_label in OUTPUT_LABELS:
            acc_obs, slot_obs, tps_obs = [], [], []
            total_blocks = 0
            total_committed = 0
            reaching_requests = 0
            for r in primary:
                st = row_bin_cache[id(r)][bin_label]
                if st["reached_bin"]:
                    reaching_requests += 1
                total_blocks += st["blocks_in_bin"]
                total_committed += st["committed_tokens_in_bin"]
                if st["accepted_length"] is not None:
                    acc_obs.append(
                        {
                            "prompt_id": r["prompt_id"],
                            "seed": r["seed"],
                            "accepted_length": st["accepted_length"],
                        }
                    )
                if st["slot_acceptance_rate"] is not None:
                    slot_obs.append(
                        {
                            "prompt_id": r["prompt_id"],
                            "seed": r["seed"],
                            "slot_acceptance_rate": st["slot_acceptance_rate"],
                        }
                    )
                if st["tokens_per_second"] is not None:
                    tps_obs.append(
                        {
                            "prompt_id": r["prompt_id"],
                            "seed": r["seed"],
                            "tokens_per_second": st["tokens_per_second"],
                        }
                    )
            bins_out.append(
                {
                    "bin": bin_label,
                    "empty_bin": reaching_requests == 0,
                    "requests_reaching_bin": reaching_requests,
                    "primary_requests": len(primary),
                    "blocks_in_bin": total_blocks,
                    "committed_tokens_in_bin": total_committed,
                    "accepted_length": independent_prompt_bootstrap(
                        acc_obs, "accepted_length", seed=seed
                    )
                    if acc_obs
                    else None,
                    "slot_acceptance_rate": independent_prompt_bootstrap(
                        slot_obs, "slot_acceptance_rate", seed=seed
                    )
                    if slot_obs
                    else None,
                    "tokens_per_second": independent_prompt_bootstrap(
                        tps_obs, "tokens_per_second", seed=seed
                    )
                    if tps_obs
                    else None,
                    "timing_source": (
                        "isolated_timing_rounds" if isolated_timing else "observational_eval_rounds"
                    ),
                }
            )
        slices.append(
            {
                "condition": condition,
                "temperature": temperature,
                "sampling_warp_config": json.loads(warp),
                "arm": arm,
                "primary_requests": len(primary),
                "repetition_excluded": len(group) - len(primary),
                "timing_isolated": isolated_timing,
                "bins": bins_out,
            }
        )
    return slices


def recompute_matched_support_comparisons(
    rows: list[dict[str, Any]],
    timed_groups_list: list[dict[str, Any]] | None = None,
    *,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    by_slot: dict[tuple[str, float, str, str, int, str], dict[str, Any]] = {}
    regimes: set[tuple[str, float, str]] = set()
    for row in rows:
        warp = json.dumps(row["sampling_warp_config"], sort_keys=True)
        regimes.add((row["condition"], float(row["temperature"]), warp))
        by_slot[
            (
                row["condition"],
                float(row["temperature"]),
                warp,
                row["prompt_id"],
                int(row["seed"]),
                row["arm"],
            )
        ] = row
    timed_groups: dict[tuple[str, float, str, str], dict[str, Any]] = {}
    for g in timed_groups_list or []:
        w = json.dumps(g["sampling_warp_config"], sort_keys=True)
        timed_groups[(g["condition"], float(g["temperature"]), w, g["arm"])] = g

    comparisons: list[dict[str, Any]] = []
    for arm, ref_arm, support_family in M5_MATCHED_PAIRS:
        for condition, temperature, warp in sorted(regimes):
            acc_deltas, fwd_deltas = [], []
            excluded = 0
            for (c, t, w, pid, sd, a), row in by_slot.items():
                if a != arm or (c, t, w) != (condition, temperature, warp):
                    continue
                ref_row = by_slot.get((c, t, w, pid, sd, ref_arm))
                if ref_row is None:
                    continue
                if condition.startswith("long_") and (
                    row["generation"]["repetition_dominated"]
                    or ref_row["generation"]["repetition_dominated"]
                ):
                    excluded += 1
                    continue
                acc_deltas.append(
                    {
                        "prompt_id": pid,
                        "seed": sd,
                        "delta": float(row["mean_accepted_length"])
                        - float(ref_row["mean_accepted_length"]),
                    }
                )
                fwd_deltas.append(
                    {
                        "prompt_id": pid,
                        "seed": sd,
                        "delta": float(row["target_forwards_per_generated_token"])
                        - float(ref_row["target_forwards_per_generated_token"]),
                    }
                )
            if not acc_deltas and excluded == 0:
                continue
            tg_arm = timed_groups.get((condition, temperature, warp, arm))
            tg_ref = timed_groups.get((condition, temperature, warp, ref_arm))
            tps_arm = tg_arm.get("matched_arm_tokens_per_second") if tg_arm else None
            tps_ref = tg_ref.get("matched_arm_tokens_per_second") if tg_ref else None
            comparisons.append(
                {
                    "arm": arm,
                    "reference_arm": ref_arm,
                    "matched_support_family": support_family,
                    "condition": condition,
                    "temperature": temperature,
                    "sampling_warp_config": json.loads(warp),
                    "matched_pairs": len(acc_deltas),
                    "repetition_excluded_pairs": excluded,
                    "mean_accepted_length_delta": independent_prompt_bootstrap(
                        acc_deltas, "delta", seed=seed
                    )
                    if acc_deltas
                    else None,
                    "target_forwards_per_generated_token_delta": independent_prompt_bootstrap(
                        fwd_deltas, "delta", seed=seed
                    )
                    if fwd_deltas
                    else None,
                    "isolated_timing_tokens_per_second": {
                        "arm": tps_arm,
                        "reference_arm": tps_ref,
                        "ratio": (tps_arm / tps_ref) if (tps_arm and tps_ref) else None,
                    }
                    if (tg_arm is not None and tg_ref is not None)
                    else None,
                }
            )
    return comparisons


def recompute_head_to_head_online_offset_comparison(
    eval_rows: list[dict[str, Any]],
    isolated_timing_rows: list[dict[str, Any]] | None = None,
    *,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    eval_by_slot: dict[tuple[str, float, str, str, int, str], dict[str, Any]] = {}
    regimes: set[tuple[str, float, str]] = set()
    active_arms: set[str] = set()
    for row in eval_rows:
        warp = json.dumps(row["sampling_warp_config"], sort_keys=True)
        regimes.add((row["condition"], float(row["temperature"]), warp))
        active_arms.add(row["arm"])
        eval_by_slot[
            (
                row["condition"],
                float(row["temperature"]),
                warp,
                row["prompt_id"],
                int(row["seed"]),
                row["arm"],
            )
        ] = row
    timed_by_slot: dict[tuple[str, float, str, str, int, Any, str], dict[str, Any]] = {}
    for row in isolated_timing_rows or []:
        if row.get("phase_profile"):
            continue
        warp = json.dumps(row["sampling_warp_config"], sort_keys=True)
        timed_by_slot[
            (
                row["condition"],
                float(row["temperature"]),
                warp,
                row["prompt_id"],
                int(row["seed"]),
                row.get("repeat"),
                row["arm"],
            )
        ] = row

    row_bin_cache: dict[int, dict[str, dict[str, Any]]] = {}

    def _get_bin_stats(r: dict[str, Any], lbl: str) -> dict[str, Any]:
        rid = id(r)
        cached = row_bin_cache.get(rid)
        if cached is None:
            cached = _all_offset_bin_stats(r)
            row_bin_cache[rid] = cached
        return cached[lbl]

    comparisons: list[dict[str, Any]] = []
    for arm, ref_arm in M5_ONLINE_HEAD_TO_HEAD_PAIRS:
        if arm not in active_arms or ref_arm not in active_arms:
            continue
        for condition, temperature, warp in sorted(regimes):
            paired_eval = []
            excluded_eval = 0
            for (c, t, w, pid, sd, a), row in eval_by_slot.items():
                if a != arm or (c, t, w) != (condition, temperature, warp):
                    continue
                ref_row = eval_by_slot.get((c, t, w, pid, sd, ref_arm))
                if ref_row is None:
                    continue
                if condition.startswith("long_") and (
                    row["generation"]["repetition_dominated"]
                    or ref_row["generation"]["repetition_dominated"]
                ):
                    excluded_eval += 1
                    continue
                paired_eval.append((pid, sd, row, ref_row))
            if not paired_eval and excluded_eval == 0:
                continue
            paired_timed = []
            for (c, t, w, pid, sd, rep, a), row in timed_by_slot.items():
                if a != arm or (c, t, w) != (condition, temperature, warp):
                    continue
                ref_row = timed_by_slot.get((c, t, w, pid, sd, rep, ref_arm))
                if ref_row is None:
                    continue
                if condition.startswith("long_") and (
                    row["generation"]["repetition_dominated"]
                    or ref_row["generation"]["repetition_dominated"]
                ):
                    continue
                paired_timed.append((pid, sd, row, ref_row))

            bin_reports = []
            for bin_label in OUTPUT_LABELS:
                acc_deltas, slot_deltas, obs_tps_ratios = [], [], []
                arm_reach, ref_reach = 0, 0
                for pid, sd, row, ref_row in paired_eval:
                    st_arm = _get_bin_stats(row, bin_label)
                    st_ref = _get_bin_stats(ref_row, bin_label)
                    arm_reach += int(st_arm["reached_bin"])
                    ref_reach += int(st_ref["reached_bin"])
                    if st_arm["accepted_length"] is not None and st_ref["accepted_length"] is not None:
                        acc_deltas.append(
                            {
                                "prompt_id": pid,
                                "seed": sd,
                                "delta": st_arm["accepted_length"] - st_ref["accepted_length"],
                            }
                        )
                    if (
                        st_arm["slot_acceptance_rate"] is not None
                        and st_ref["slot_acceptance_rate"] is not None
                    ):
                        slot_deltas.append(
                            {
                                "prompt_id": pid,
                                "seed": sd,
                                "delta": st_arm["slot_acceptance_rate"]
                                - st_ref["slot_acceptance_rate"],
                            }
                        )
                    if st_arm["tokens_per_second"] and st_ref["tokens_per_second"]:
                        obs_tps_ratios.append(
                            {
                                "prompt_id": pid,
                                "seed": sd,
                                "ratio": st_arm["tokens_per_second"] / st_ref["tokens_per_second"],
                            }
                        )
                iso_tps_ratios, iso_tps_deltas, iso_arm_tps, iso_ref_tps = [], [], [], []
                for pid, sd, t_row, t_ref in paired_timed:
                    st_arm = _get_bin_stats(t_row, bin_label)
                    st_ref = _get_bin_stats(t_ref, bin_label)
                    if st_arm["tokens_per_second"] and st_ref["tokens_per_second"]:
                        iso_arm_tps.append(
                            {"prompt_id": pid, "seed": sd, "tps": st_arm["tokens_per_second"]}
                        )
                        iso_ref_tps.append(
                            {"prompt_id": pid, "seed": sd, "tps": st_ref["tokens_per_second"]}
                        )
                        iso_tps_ratios.append(
                            {
                                "prompt_id": pid,
                                "seed": sd,
                                "ratio": st_arm["tokens_per_second"] / st_ref["tokens_per_second"],
                            }
                        )
                        iso_tps_deltas.append(
                            {
                                "prompt_id": pid,
                                "seed": sd,
                                "delta": st_arm["tokens_per_second"] - st_ref["tokens_per_second"],
                            }
                        )
                bin_reports.append(
                    {
                        "bin": bin_label,
                        "empty_bin": len(acc_deltas) == 0 and len(slot_deltas) == 0,
                        "arm_requests_reaching_bin": arm_reach,
                        "reference_requests_reaching_bin": ref_reach,
                        "paired_requests_in_bin": len(acc_deltas),
                        "accepted_length_delta": independent_prompt_bootstrap(
                            acc_deltas, "delta", seed=seed
                        )
                        if acc_deltas
                        else None,
                        "slot_acceptance_rate_delta": independent_prompt_bootstrap(
                            slot_deltas, "delta", seed=seed
                        )
                        if slot_deltas
                        else None,
                        "observational_tokens_per_second_ratio": independent_prompt_bootstrap(
                            obs_tps_ratios, "ratio", seed=seed
                        )
                        if obs_tps_ratios
                        else None,
                        "isolated_timing_tokens_per_second": {
                            "arm": independent_prompt_bootstrap(iso_arm_tps, "tps", seed=seed),
                            "reference_arm": independent_prompt_bootstrap(
                                iso_ref_tps, "tps", seed=seed
                            ),
                            "paired_ratio": independent_prompt_bootstrap(
                                iso_tps_ratios, "ratio", seed=seed
                            ),
                            "paired_delta": independent_prompt_bootstrap(
                                iso_tps_deltas, "delta", seed=seed
                            ),
                        }
                        if iso_tps_ratios
                        else None,
                    }
                )
            comparisons.append(
                {
                    "arm": arm,
                    "reference_arm": ref_arm,
                    "condition": condition,
                    "temperature": temperature,
                    "sampling_warp_config": json.loads(warp),
                    "matched_pairs": len(paired_eval),
                    "repetition_excluded_pairs": excluded_eval,
                    "bins": bin_reports,
                }
            )
    return comparisons


class PostM4SuiteAuditor:
    """Read-only zero-drift auditor for all 4 `post_m4_suite` suites + upstream M0-M4 immutability."""

    def __init__(
        self,
        *,
        repo: Path,
        artifacts: Path,
        suite_root: Path | None = None,
        scale: str = "reference",
        seed: int = 2026,
        allow_partial: bool = False,
        require_complete: bool = False,
        allow_pilot: bool = False,
        check_implementation_audit: bool = False,
    ) -> None:
        self.repo = repo.resolve()
        self.artifacts = artifacts.resolve()
        self.suite_root = (
            suite_root.resolve() if suite_root is not None else (self.artifacts / "post_m4_suite")
        )
        self.scale = scale
        self.seed = seed
        self.allow_partial = allow_partial
        self.require_complete = require_complete
        self.allow_pilot = allow_pilot
        self.check_implementation_audit = check_implementation_audit
        self.checks = 0
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.read_sha256: dict[str, str] = {}

    def check(self, cond: bool, message: str) -> bool:
        self.checks += 1
        if not cond:
            self.failures.append(message)
        return bool(cond)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def _is_live_inflight_file(self, path: Path) -> bool:
        name = path.name
        if name.startswith("manifest") or name.endswith("-processes.json"):
            return True
        if "sessions" in path.parts:
            return True
        return False

    def read_bytes_tracked(self, path: Path) -> bytes:
        p = Path(path).resolve()
        data = p.read_bytes()
        digest = sha256_bytes(data)
        prev = self.read_sha256.get(str(p))
        if prev is not None and prev != digest:
            if self.allow_partial and not self.require_complete and self._is_live_inflight_file(p):
                self.warn(f"inflight file advanced during live --allow_partial audit: {p}")
            else:
                self.check(False, f"file mutated during audit: {p}")
        else:
            self.check(True, f"file immutable during audit: {p}")
        self.read_sha256[str(p)] = digest
        return data

    def read_json_tracked(self, path: Path) -> Any:
        return json.loads(self.read_bytes_tracked(path))

    def read_json_batch_tracked(self, paths: list[Path]) -> list[Any]:
        """Read, SHA256, and JSON-parse `paths` in parallel (`max_workers=8`), then record tracking checks in exact input order."""
        if len(paths) < 16:
            return [self.read_json_tracked(p) for p in paths]
        from concurrent.futures import ThreadPoolExecutor

        def _load_one(raw_p: Path) -> tuple[Path, str, Any]:
            rp = Path(raw_p).resolve()
            data = rp.read_bytes()
            return (rp, sha256_bytes(data), json.loads(data))

        with ThreadPoolExecutor(max_workers=8) as ex:
            loaded = list(ex.map(_load_one, paths))
        out: list[Any] = []
        for p, digest, obj in loaded:
            prev = self.read_sha256.get(str(p))
            if prev is not None and prev != digest:
                if self.allow_partial and not self.require_complete and self._is_live_inflight_file(p):
                    self.warn(f"inflight file advanced during live --allow_partial audit: {p}")
                else:
                    self.check(False, f"file mutated during audit: {p}")
            else:
                self.check(True, f"file immutable during audit: {p}")
            self.read_sha256[str(p)] = digest
            out.append(obj)
        return out

    def verify_immutability_at_end(self) -> None:
        items = list(self.read_sha256.items())
        if len(items) >= 32:
            from concurrent.futures import ThreadPoolExecutor

            def _hash_existing(path_str: str) -> str | None:
                p = Path(path_str)
                return sha256_file(p) if p.is_file() else None

            with ThreadPoolExecutor(max_workers=8) as ex:
                cur_shas = list(ex.map(_hash_existing, [k for k, _ in items]))
        else:
            cur_shas = [sha256_file(Path(k)) if Path(k).is_file() else None for k, _ in items]

        for (path_str, expected_sha), cur in zip(items, cur_shas):
            p = Path(path_str)
            if cur is None:
                self.check(False, f"tracked file disappeared during audit: {p}")
                continue
            if cur != expected_sha:
                if (
                    self.allow_partial
                    and not self.require_complete
                    and self._is_live_inflight_file(p)
                ):
                    self.warn(f"inflight file advanced before audit end: {p}")
                else:
                    self.check(False, f"file mutated before audit end: {p}")
            else:
                self.check(True, f"file unchanged at audit end: {p}")

    # ------------------------------------------------------------------
    # 0. Upstream M0-M4 Immutability Verification
    # ------------------------------------------------------------------

    def audit_upstream_m0_m4_immutability(self) -> dict[str, Any]:
        """Verify that `post_m4_suite` has not mutated any upstream M0-M4 reference artifact."""
        checked_files: dict[str, str] = {}
        ms_path = self.artifacts / "milestones.json"
        if not ms_path.is_file() and (self.repo / "artifacts" / "milestones.json").is_file():
            ms_path = self.repo / "artifacts" / "milestones.json"

        milestones = self.read_json_tracked(ms_path) if ms_path.is_file() else None
        if milestones is not None and self.scale == "reference":
            for stg_key, rel_map in (
                (
                    "M3",
                    {
                        "settings": "m3/settings.json",
                        "summary": "m3/summary.json",
                        "timing": "m3/timing.json",
                        "greedy_output_comparisons": "m3/greedy_output_comparisons.json",
                        "validation_reference": "m3_validation_reference.json",
                    },
                ),
                (
                    "M4",
                    {
                        "settings": "m4/settings.json",
                        "summary": "m4/summary.json",
                        "timing": "m4/timing.json",
                        "timing_isolation_calibration": "m4/timing_isolation_calibration.json",
                        "feature_context": "m4/feature_context.json",
                        "greedy_output_comparisons": "m4/greedy_output_comparisons.json",
                        "sharding_plan": "m4/timing/sharding_plan.json",
                        "validation_reference": "m4_validation_reference.json",
                    },
                ),
            ):
                rec_shas = milestones.get(stg_key, {}).get("artifact_sha256", {})
                for sha_key, rel_p in rel_map.items():
                    exp = rec_shas.get(sha_key)
                    fp = self.artifacts / rel_p
                    if exp and fp.is_file():
                        act = sha256_file(fp)
                        self.read_sha256[str(fp.resolve())] = act
                        checked_files[rel_p] = act
                        self.check(
                            act == exp,
                            f"upstream {rel_p} SHA256 {act} != milestones.json {stg_key}.{sha_key} {exp}",
                        )

        cont_path = self.artifacts / "reference_continuation_state.json"
        if cont_path.is_file() and self.scale == "reference":
            cont = json.loads(cont_path.read_text(encoding="utf-8"))
            for bound_path_str, exp_sha in sorted(
                cont.get("frozen_artifact_binding", {}).items()
            ):
                bp = Path(bound_path_str)
                if "artifacts/" in bound_path_str and "m4_evidence_validator" not in bound_path_str:
                    sub = bound_path_str.split("artifacts/", 1)[1]
                    cand = self.artifacts / sub
                    if cand.is_file():
                        act = sha256_file(cand)
                        self.read_sha256[str(cand.resolve())] = act
                        checked_files[sub] = act
                        self.check(
                            act == exp_sha,
                            f"upstream frozen M0-M2 artifact {sub} mutated: {act} != {exp_sha}",
                        )

        for rel in UPSTREAM_M0_M4_IMMUTABLE_RELPATHS:
            fp = self.artifacts / rel
            if fp.is_file() and rel not in checked_files:
                act = sha256_file(fp)
                self.read_sha256[str(fp.resolve())] = act
                checked_files[rel] = act

        return {
            "checked_upstream_files": len(checked_files),
            "sha256s": checked_files,
        }

    # ------------------------------------------------------------------
    # 1. Suite: `m2_refit_calibration`
    # ------------------------------------------------------------------

    def audit_m2_refit_calibration(self) -> dict[str, Any]:
        report_path = self.suite_root / "m2_refit_calibration.json"
        ws_link = self.suite_root / "m2_refit_workspace"
        if not report_path.is_file():
            msg = "m2_refit_calibration.json not yet present"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                inflight_info: dict[str, Any] = {"status": "in_progress_or_not_started"}
                if ws_link.exists():
                    ws_real = ws_link.resolve()
                    inflight_info["workspace"] = str(ws_real)
                    inflight_info["workspace_entries"] = sorted(p.name for p in ws_real.iterdir())
                return inflight_info
            self.check(False, msg)
            return {"status": "missing"}

        rep = self.read_json_tracked(report_path)
        self.check(rep.get("suite") == "m2_refit_calibration", "m2_refit: wrong suite tag")
        self.check(rep.get("scale") == self.scale, f"m2_refit: scale {rep.get('scale')} != {self.scale}")
        allowed_statuses = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                rep.get("status") in allowed_statuses,
                f"m2_refit: status {rep.get('status')!r} not in {allowed_statuses}",
            )
        self.check(
            rep.get("refit_fit_gate_status") in ("passed", "complete"),
            f"m2_refit: refit_fit_gate_status={rep.get('refit_fit_gate_status')!r}",
        )

        refit_rep_p = Path(rep.get("refit_report_path", ""))
        self.check(refit_rep_p.is_file(), f"m2_refit: missing refit_report_path {refit_rep_p}")
        if refit_rep_p.is_file():
            fit_json = self.read_json_tracked(refit_rep_p)
            self.check(
                fit_json.get("scale") == self.scale,
                f"m2_refit: underlying train report scale mismatch",
            )

        beta_cal = rep.get("beta_calibration") or {}
        self.check(
            "dense" in beta_cal and "dense_sparse" in beta_cal,
            f"m2_refit: beta_calibration missing dense/dense_sparse keys: {list(beta_cal.keys())}",
        )
        for stg_name, binfo in beta_cal.items():
            if isinstance(binfo, dict):
                sel_b = binfo.get("optimal_beta", binfo.get("selected_beta", binfo.get("beta_star")))
                self.check(
                    isinstance(sel_b, (int, float)) and math.isfinite(float(sel_b)) and float(sel_b) >= 0.0,
                    f"m2_refit: beta_calibration.{stg_name} invalid optimal_beta={sel_b!r}",
                )
                cal_kl = binfo.get("calibrated_kl")
                uncal_kl = binfo.get("uncalibrated_beta1_kl")
                base0_kl = binfo.get("baseline_beta0_kl")
                if all(isinstance(x, (int, float)) for x in (cal_kl, uncal_kl, base0_kl)):
                    self.check(
                        float(cal_kl) <= float(uncal_kl) + 1e-11
                        and float(cal_kl) <= float(base0_kl) + 1e-11,
                        f"m2_refit: beta_calibration.{stg_name} calibrated_kl={cal_kl} > min(beta1={uncal_kl}, beta0={base0_kl})",
                    )

        variants = rep.get("variants", [])
        self.check(len(variants) >= 5, f"m2_refit: expected >=5 variants, got {len(variants)}")

        # Locate raw closed-loop shards if multi-GPU sharded evaluation ran
        ws_dir = refit_rep_p.parent if refit_rep_p.is_file() else (ws_link.resolve() if ws_link.exists() else None)
        shard_rows_by_key: dict[str, list[dict[str, Any]]] | None = None
        if ws_dir is not None and (ws_dir / "refit_closed_loop_shards").is_dir():
            shard_files = sorted((ws_dir / "refit_closed_loop_shards").glob("result-rank*.json"))
            if shard_files:
                shard_payloads = [self.read_json_tracked(sf) for sf in shard_files]
                shard_rows_by_key = defaultdict(list)
                for sp in shard_payloads:
                    for k, rlist in sp.get("rows_by_key", {}).items():
                        for rw in rlist:
                            if rw["condition"].startswith("long_") and "output_tokens" in rw:
                                self.check(
                                    independent_repetition_dominated(rw["output_tokens"])
                                    == _row_rep_flag(rw),
                                    f"m2_refit: repetition_dominated mismatch on {rw['prompt_id']} ({k})",
                                )
                            shard_rows_by_key[k].append(rw)

        variant_summaries = []
        plain_by_slot: dict[tuple[str, Any], dict[str, Any]] = {}
        if shard_rows_by_key and "plain_dflash" in shard_rows_by_key:
            plain_by_slot = {
                (r["prompt_id"], r["repeat"]): r for r in shard_rows_by_key["plain_dflash"]
            }

        for v in variants:
            vname = v["variant"]
            d_ckpt = Path(v["dense_checkpoint"])
            s_ckpt = Path(v["sparse_checkpoint"])
            self.check(d_ckpt.is_file(), f"m2_refit/{vname}: missing dense_checkpoint {d_ckpt}")
            self.check(s_ckpt.is_file(), f"m2_refit/{vname}: missing sparse_checkpoint {s_ckpt}")
            for kl_key in ("offline_dense_kl", "offline_dense_sparse_kl"):
                kl_obj = v.get(kl_key, {})
                for split_k in ("dev", "test"):
                    wkl = kl_obj.get(split_k, {}).get("weighted_mean_kl")
                    self.check(
                        isinstance(wkl, (int, float)) and math.isfinite(float(wkl)) and float(wkl) >= 0.0,
                        f"m2_refit/{vname}.{kl_key}.{split_k}.weighted_mean_kl invalid: {wkl!r}",
                    )
                if kl_obj.get("sparse_backend") == "exact_observed_pairs":
                    self.check(
                        kl_obj.get("collision_count") == 0,
                        f"m2_refit/{vname}.{kl_key}: ExactObservedPairs collision_count={kl_obj.get('collision_count')} != 0",
                    )

            cl = v.get("closed_loop", {})
            if self.require_complete and not self.allow_pilot:
                self.check(
                    cl.get("is_pilot_subset") is False,
                    f"m2_refit/{vname}: closed_loop is a pilot subset",
                )
            # Recompute from raw shards when available!
            if shard_rows_by_key is not None:
                for sub_arm, row_key, paired in (
                    ("plain_dflash", "plain_dflash", False),
                    ("dense", f"{vname}::dense", True),
                    ("dense_sparse", f"{vname}::dense_sparse", True),
                ):
                    raw_r = shard_rows_by_key.get(row_key, [])
                    prim_r = [r for r in raw_r if not _is_rep_excluded(r)]
                    rec_sub = cl.get(sub_arm, {})
                    self.check(
                        rec_sub.get("requests") == len(raw_r)
                        and rec_sub.get("primary_requests") == len(prim_r)
                        and rec_sub.get("repetition_excluded") == len(raw_r) - len(prim_r),
                        f"m2_refit/{vname}.{sub_arm}: request/repetition_excluded counts drift from raw shards",
                    )
                    def _is_fin_num(v: Any) -> bool:
                        return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))

                    if prim_r:
                        exp_acc = statistics.mean(r["mean_accepted_length"] for r in prim_r)
                        exp_tps = statistics.mean(r["tokens_per_second"] for r in prim_r)
                        rec_acc_v = rec_sub.get("mean_accepted_length")
                        rec_tps_v = rec_sub.get("mean_tokens_per_second")
                        self.check(
                            _is_fin_num(rec_acc_v) and math.isclose(exp_acc, float(rec_acc_v), rel_tol=1e-10, abs_tol=1e-10),
                            f"m2_refit/{vname}.{sub_arm}: mean_accepted_length {rec_acc_v!r} != raw {exp_acc}",
                        )
                        self.check(
                            _is_fin_num(rec_tps_v) and math.isclose(exp_tps, float(rec_tps_v), rel_tol=1e-10, abs_tol=1e-10),
                            f"m2_refit/{vname}.{sub_arm}: mean_tokens_per_second {rec_tps_v!r} != raw {exp_tps}",
                        )
                    rec_bc = rec_sub.get("by_condition")
                    exp_conds = {r["condition"] for r in raw_r}
                    self.check(
                        isinstance(rec_bc, dict) and set(rec_bc.keys()) == exp_conds,
                        f"m2_refit/{vname}.{sub_arm}.by_condition: keys {sorted(rec_bc.keys()) if isinstance(rec_bc, dict) else rec_bc!r} != {sorted(exp_conds)}",
                    )
                    rec_bc = rec_bc if isinstance(rec_bc, dict) else {}
                    for cond_k in sorted(exp_conds):
                        c_all = [r for r in raw_r if r["condition"] == cond_k]
                        c_prim = [r for r in c_all if not _is_rep_excluded(r)]
                        cb = rec_bc.get(cond_k) or {}
                        self.check(
                            cb.get("requests") == len(c_all)
                            and cb.get("primary_requests") == len(c_prim)
                            and cb.get("repetition_excluded") == len(c_all) - len(c_prim),
                            f"m2_refit/{vname}.{sub_arm}.by_condition.{cond_k}: counts drift from raw shards",
                        )
                        if c_prim:
                            cb_acc = cb.get("mean_accepted_length")
                            cb_tps = cb.get("mean_tokens_per_second")
                            self.check(
                                _is_fin_num(cb_acc)
                                and math.isclose(
                                    statistics.mean(r["mean_accepted_length"] for r in c_prim),
                                    float(cb_acc),
                                    rel_tol=1e-10,
                                    abs_tol=1e-10,
                                ),
                                f"m2_refit/{vname}.{sub_arm}.by_condition.{cond_k}: mean_accepted_length drift ({cb_acc!r})",
                            )
                            self.check(
                                _is_fin_num(cb_tps)
                                and math.isclose(
                                    statistics.mean(r["tokens_per_second"] for r in c_prim),
                                    float(cb_tps),
                                    rel_tol=1e-10,
                                    abs_tol=1e-10,
                                ),
                                f"m2_refit/{vname}.{sub_arm}.by_condition.{cond_k}: mean_tokens_per_second drift ({cb_tps!r})",
                            )
                    if paired and plain_by_slot:
                        acc_d, tps_r, excl_p = [], [], 0
                        for r in raw_r:
                            pref = plain_by_slot.get((r["prompt_id"], r["repeat"]))
                            if pref is None:
                                continue
                            if _is_rep_excluded(r) or _is_rep_excluded(pref):
                                excl_p += 1
                                continue
                            acc_d.append(r["mean_accepted_length"] - pref["mean_accepted_length"])
                            if pref["tokens_per_second"] > 0:
                                tps_r.append(r["tokens_per_second"] / pref["tokens_per_second"])
                        self.check(
                            excl_p == rec_sub.get("repetition_excluded_comparison_pairs"),
                            f"m2_refit/{vname}.{sub_arm}: repetition_excluded_comparison_pairs mismatch",
                        )
                        if acc_d:
                            rec_pd = rec_sub.get("paired_accepted_length_delta_vs_plain")
                            self.check(
                                _is_fin_num(rec_pd)
                                and math.isclose(
                                    statistics.mean(acc_d),
                                    float(rec_pd),
                                    rel_tol=1e-10,
                                    abs_tol=1e-10,
                                ),
                                f"m2_refit/{vname}.{sub_arm}: paired_accepted_length_delta_vs_plain mismatch ({rec_pd!r})",
                            )
                        if tps_r:
                            rec_pr = rec_sub.get("paired_tps_ratio_vs_plain")
                            self.check(
                                _is_fin_num(rec_pr)
                                and math.isclose(
                                    statistics.mean(tps_r),
                                    float(rec_pr),
                                    rel_tol=1e-10,
                                    abs_tol=1e-10,
                                ),
                                f"m2_refit/{vname}.{sub_arm}: paired_tps_ratio_vs_plain mismatch ({rec_pr!r})",
                            )

            variant_summaries.append(
                {
                    "variant": vname,
                    "dense_dev_kl": v["offline_dense_kl"]["dev"]["weighted_mean_kl"],
                    "dense_test_kl": v["offline_dense_kl"]["test"]["weighted_mean_kl"],
                    "sparse_dev_kl": v["offline_dense_sparse_kl"]["dev"]["weighted_mean_kl"],
                    "sparse_test_kl": v["offline_dense_sparse_kl"]["test"]["weighted_mean_kl"],
                    "closed_loop": cl,
                }
            )

        return {
            "status": rep.get("status"),
            "config_fingerprint": rep.get("config_fingerprint"),
            "refit_hyperparameters": rep.get("refit_hyperparameters"),
            "beta_calibration": beta_cal,
            "variants": variant_summaries,
        }

    # ------------------------------------------------------------------
    # Shared M5 Stage Workspace Auditor (`online_head_to_head` & `m5_eval`)
    # ------------------------------------------------------------------

    def _enrich_timed_groups_with_paired_plain(
        self,
        *,
        label: str,
        timed_rows: list[dict[str, Any]],
        recomp_t_groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        paired_plain = {
            (
                r["condition"],
                float(r["temperature"]),
                json.dumps(r["sampling_warp_config"], sort_keys=True),
                r["prompt_id"],
                r["seed"],
                str(r["repeat"]).split(":epoch=")[0],
            ): r
            for r in timed_rows
            if r["arm"] == "plain_dflash"
        }
        for g in recomp_t_groups:
            gk = (
                g["condition"],
                float(g["temperature"]),
                json.dumps(g["sampling_warp_config"], sort_keys=True),
            )
            paired, m_arm, m_plain = [], [], []
            excl_pairs = 0
            for r in timed_rows:
                rk = (
                    r["condition"],
                    float(r["temperature"]),
                    json.dumps(r["sampling_warp_config"], sort_keys=True),
                )
                if r["arm"] != g["arm"] or rk != gk:
                    continue
                rep_norm = str(r["repeat"]).split(":epoch=")[0]
                b_key_lookup = (*rk, r["prompt_id"], r["seed"], rep_norm)
                self.check(
                    b_key_lookup in paired_plain,
                    f"{label}/timing: missing matched plain_dflash row for {b_key_lookup}",
                )
                if b_key_lookup not in paired_plain:
                    continue
                b_row = paired_plain[b_key_lookup]
                if _is_rep_excluded(r) or _is_rep_excluded(b_row):
                    excl_pairs += 1
                    continue
                m_arm.append(r)
                m_plain.append(b_row)
                paired.append(
                    {
                        "prompt_id": r["prompt_id"],
                        "seed": r["seed"],
                        "ratio": float(r["tokens_per_second"]) / float(b_row["tokens_per_second"]),
                    }
                )
            ratio_boot = independent_prompt_bootstrap(paired, "ratio", seed=self.seed)
            m_arm_mean = independent_prompt_bootstrap(m_arm, "tokens_per_second", seed=self.seed)["mean"]
            m_plain_mean = independent_prompt_bootstrap(
                m_plain, "tokens_per_second", seed=self.seed
            )["mean"]
            g["matched_timing_requests"] = len(m_arm)
            g["repetition_excluded_comparison_pairs"] = excl_pairs
            g["matched_arm_tokens_per_second"] = m_arm_mean
            g["matched_plain_tokens_per_second"] = m_plain_mean
            g["throughput_relative_to_plain"] = (
                m_arm_mean / m_plain_mean if m_plain_mean else None
            )
            g["throughput_improved_after_all_proposal_and_online_costs"] = (
                m_arm_mean > m_plain_mean if m_plain_mean else None
            )
            g["paired_throughput_ratio"] = ratio_boot
            g["throughput_gain_supported_by_95_ci"] = (
                ratio_boot["ci95"][0] > 1.0 if ratio_boot["ci95"] else None
            )
        return recomp_t_groups

    def _audit_m5_stage_workspace(
        self,
        *,
        label: str,
        suite_report: dict[str, Any],
        artifacts_block: dict[str, str],
        expected_arms: list[str],
        check_head_to_head_offsets: bool,
        expected_bigram_kappa: float | None = None,
    ) -> dict[str, Any]:
        from build_and_audit_reports import PostGateEAuditor

        settings_p = Path(artifacts_block["settings_json"])
        summary_p = Path(artifacts_block["summary_json"])
        timing_p = Path(artifacts_block["timing_json"])
        for p_item in (settings_p, summary_p, timing_p):
            self.check(p_item.is_file(), f"{label}: missing stage artifact {p_item}")
        if not (settings_p.is_file() and summary_p.is_file() and timing_p.is_file()):
            return {"status": "missing_stage_artifacts"}

        settings = self.read_json_tracked(settings_p)
        summary = self.read_json_tracked(summary_p)
        timing = self.read_json_tracked(timing_p)
        stage_dir = summary_p.parent
        frozen_hash = fingerprint(settings)

        self.check(
            settings.get("frozen") is True and settings.get("stage") == "m5",
            f"{label}: m5/settings.json frozen/stage invalid",
        )
        self.check(
            set(settings.get("selected", {}).keys()) == set(expected_arms),
            f"{label}: selected arms {sorted(settings.get('selected', {}).keys())} != {sorted(expected_arms)}",
        )

        # Matched C0[j] & beta_relax check across dense_candidate / lookahead_*_candidate!
        present_c0_arms = [a for a in MATCHED_C0_CANDIDATE_ARMS if a in expected_arms]
        expected_c0_beta_relax: float | None = None
        if present_c0_arms:
            br_values = []
            for a in present_c0_arms:
                sel_cfg = settings["selected"][a]
                self.check(
                    "beta_relax" in sel_cfg,
                    f"{label}: matched-C0 arm {a} missing explicit beta_relax in settings.json",
                )
                br = sel_cfg.get("beta_relax", 1.0)
                self.check(
                    not isinstance(br, bool)
                    and isinstance(br, (int, float))
                    and math.isfinite(float(br))
                    and float(br) >= 0.0,
                    f"{label}: invalid beta_relax={br!r} on {a}",
                )
                br_values.append(float(br))
            self.check(
                len(set(br_values)) == 1,
                f"{label}: matched C0[j] violation — beta_relax differs across {present_c0_arms}: {br_values}",
            )
            expected_c0_beta_relax = br_values[0]

        for la_arm in (a for a in LOOKAHEAD_ARMS if a in expected_arms):
            la_cfg = settings["selected"][la_arm]
            t_rel = la_cfg.get("T_relax")
            self.check(
                isinstance(t_rel, (int, float)) and math.isfinite(float(t_rel)) and float(t_rel) > 0.0,
                f"{label}: {la_arm} requires strictly positive T_relax > 0, got {t_rel!r}",
            )

        # Load and audit all raw evaluate rows
        eval_rows: list[dict[str, Any]] = []
        seen_eval_keys: set[str] = set()
        eval_manifest_entries: list[tuple[str, dict[str, Any], Path]] = []
        for mf in sorted((stage_dir / "evaluate").glob("manifest-rank*.json")):
            mdata = self.read_json_tracked(mf)
            for key, entry in mdata.get("jobs", {}).items():
                if entry.get("status") != "complete":
                    continue
                if entry.get("identity", {}).get("frozen_settings_hash") != frozen_hash:
                    continue
                self.check(key not in seen_eval_keys, f"{label}/evaluate: duplicate job key {key}")
                seen_eval_keys.add(key)
                eval_manifest_entries.append((key, entry, Path(entry["result_path"])))
        loaded_eval_objs = self.read_json_batch_tracked([rp for _, _, rp in eval_manifest_entries])
        for (_key, entry, rp), row in zip(eval_manifest_entries, loaded_eval_objs):
            self.check(
                self.read_sha256[str(rp.resolve())] == entry.get("result_sha256"),
                f"{label}/evaluate: SHA256 mismatch on {rp}",
            )
            self._audit_single_m5_row(
                label=f"{label}/evaluate",
                row=row,
                expected_c0_beta_relax=expected_c0_beta_relax,
                expected_bigram_kappa=expected_bigram_kappa,
                is_profile=False,
            )
            eval_rows.append(row)

        self.check(
            len(eval_rows) == int(summary.get("completed_requests", -1)),
            f"{label}: loaded {len(eval_rows)} eval rows != summary.completed_requests {summary.get('completed_requests')}",
        )
        if self.require_complete:
            self.check(
                summary.get("missing_requests") == [],
                f"{label}: missing_requests non-empty: {summary.get('missing_requests')}",
            )
            if not self.allow_pilot:
                self.check(
                    summary.get("isolated_timing_complete") is True,
                    f"{label}: summary.isolated_timing_complete={summary.get('isolated_timing_complete')!r} != True",
                )

        # Load and audit all raw timing & phase_profile rows (verifying manifest SHA256s + session bundles)
        manifest_sha_by_path: dict[str, str] = {}
        for tmf in sorted((stage_dir / "timing").glob("manifest-rank*.json")):
            tmdata = self.read_json_tracked(tmf)
            for t_entry in tmdata.get("jobs", {}).values():
                if t_entry.get("status") == "complete" and t_entry.get("result_path") and t_entry.get("result_sha256"):
                    manifest_sha_by_path[str(Path(t_entry["result_path"]).resolve())] = t_entry["result_sha256"]

        timed_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        raw_ev = timing.get("raw_evidence", {})
        sess_paths = (
            [Path(p) for p in raw_ev["session_paths"]]
            if raw_ev.get("session_paths")
            else ([Path(raw_ev["session_path"])] if raw_ev.get("session_path") else [])
        )
        self.check(bool(sess_paths), f"{label}/timing: no session_paths in timing.json")
        sess_objs = [self.read_json_tracked(sp) for sp in sess_paths]
        all_bundle_rpaths: list[Path] = []
        for sess in sess_objs:
            for bkey, bval in sess.get("bundles", {}).items():
                if bval.get("status") != "complete":
                    continue
                all_bundle_rpaths.extend(Path(p_str) for p_str in bval.get("result_paths", []))
        loaded_bundle_rows = iter(self.read_json_batch_tracked(all_bundle_rpaths))
        for sess in sess_objs:
            for bkey, bval in sess.get("bundles", {}).items():
                if bval.get("status") != "complete":
                    continue
                rpaths = bval.get("result_paths", [])
                rshas = bval.get("result_sha256", {})
                self.check(
                    len(rpaths) == len(expected_arms),
                    f"{label}/timing: bundle {bkey} has {len(rpaths)} rows != {len(expected_arms)} arms",
                )
                bundle_prof_flags: set[bool] = set()
                for p_str in rpaths:
                    rp = Path(p_str)
                    row = next(loaded_bundle_rows)
                    rp_res = str(rp.resolve())
                    exp_sha = rshas.get(p_str) or manifest_sha_by_path.get(rp_res)
                    if exp_sha:
                        self.check(
                            self.read_sha256[rp_res] == exp_sha,
                            f"{label}/timing: SHA256 mismatch on {rp}",
                        )
                    is_prof = bool(row.get("phase_profile", False))
                    bundle_prof_flags.add(is_prof)
                    self._audit_single_m5_row(
                        label=f"{label}/timing",
                        row=row,
                        expected_c0_beta_relax=expected_c0_beta_relax,
                        expected_bigram_kappa=expected_bigram_kappa,
                        is_profile=is_prof,
                    )
                    if is_prof:
                        profile_rows.append(row)
                    else:
                        timed_rows.append(row)
                self.check(
                    len(bundle_prof_flags) == 1,
                    f"{label}/timing: bundle {bkey} mixes phase_profile and timed speed rows",
                )

        self.check(
            len(timed_rows) == int(timing.get("timed_requests", -1)),
            f"{label}/timing: loaded {len(timed_rows)} timed_rows != timing.timed_requests {timing.get('timed_requests')}",
        )

        # Independently recompute eval groups, timed groups (with paired plain ratios), and phase profiles!
        helper = PostGateEAuditor(repo=self.repo, artifacts=self.artifacts, scale=self.scale, seed=self.seed)
        recomp_eval_groups = helper.recompute_group_rows(
            eval_rows, isolated_timing=False, exclude_repetition=True
        )
        ok_eg, diff_eg = close_nested(
            _canonicalize_group_list(recomp_eval_groups),
            _canonicalize_group_list(summary.get("groups", [])),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_eg, f"{label}: summary.groups drift from raw recomputation: {diff_eg}")

        rec_suite_eg = (
            suite_report.get("overall_evaluation_groups")
            if "overall_evaluation_groups" in suite_report
            else suite_report.get("closed_loop", {}).get("overall_evaluation_groups", [])
        )
        ok_seg, diff_seg = close_nested(
            _canonicalize_group_list(recomp_eval_groups),
            _canonicalize_group_list(rec_suite_eg),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_seg, f"{label}: suite overall_evaluation_groups drift: {diff_seg}")

        recomp_timed_groups = self._enrich_timed_groups_with_paired_plain(
            label=label,
            timed_rows=timed_rows,
            recomp_t_groups=helper.recompute_group_rows(
                timed_rows, isolated_timing=True, exclude_repetition=True
            ),
        )
        ok_tg, diff_tg = close_nested(
            _canonicalize_group_list(recomp_timed_groups),
            _canonicalize_group_list(timing.get("groups", [])),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_tg, f"{label}: timing.groups drift from raw recomputation: {diff_tg}")

        rec_timed_groups = (
            suite_report.get("isolated_timing_groups")
            if "isolated_timing_groups" in suite_report
            else suite_report.get("closed_loop", {}).get("isolated_timing_groups", [])
        )
        ok_stg, diff_stg = close_nested(
            _canonicalize_group_list(recomp_timed_groups),
            _canonicalize_group_list(rec_timed_groups),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_stg, f"{label}: suite isolated_timing_groups drift from raw recomputation: {diff_stg}")

        recomp_phase_profiles = helper.recompute_group_rows(
            profile_rows, isolated_timing=True, exclude_repetition=True
        )
        ok_pp, diff_pp = close_nested(
            _canonicalize_group_list(recomp_phase_profiles),
            _canonicalize_group_list(timing.get("phase_profiles", [])),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_pp, f"{label}: timing.phase_profiles drift from raw recomputation: {diff_pp}")
        if "phase_profiles" in suite_report or "phase_profiles" in suite_report.get("closed_loop", {}):
            rec_suite_pp = (
                suite_report["phase_profiles"]
                if "phase_profiles" in suite_report
                else suite_report["closed_loop"]["phase_profiles"]
            )
            ok_spp, diff_spp = close_nested(
                _canonicalize_group_list(recomp_phase_profiles),
                _canonicalize_group_list(rec_suite_pp),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            self.check(ok_spp, f"{label}: suite phase_profiles drift: {diff_spp}")

        # Recompute offset_bin_slices (eval + isolated timing) and cross-check BOTH stage JSONs & suite_report!
        recomp_eval_slices = recompute_offset_bin_slices(
            eval_rows, isolated_timing=False, seed=self.seed
        )
        ok_sum_es, diff_sum_es = close_nested(
            recomp_eval_slices, summary.get("offset_bin_slices", []), rel_tol=1e-10, abs_tol=1e-10
        )
        self.check(ok_sum_es, f"{label}: summary.offset_bin_slices drift from raw rows: {diff_sum_es}")

        rec_eval_slices = (
            suite_report.get("evaluation_offset_bin_slices")
            if "evaluation_offset_bin_slices" in suite_report
            else suite_report.get("closed_loop", {}).get("evaluation_offset_bin_slices", [])
        )
        ok_es, diff_es = close_nested(
            recomp_eval_slices, rec_eval_slices, rel_tol=1e-10, abs_tol=1e-10
        )
        self.check(ok_es, f"{label}: evaluation_offset_bin_slices drift from raw rows: {diff_es}")

        recomp_timed_slices = recompute_offset_bin_slices(
            timed_rows, isolated_timing=True, seed=self.seed
        )
        ok_tm_ts, diff_tm_ts = close_nested(
            recomp_timed_slices, timing.get("offset_bin_slices", []), rel_tol=1e-10, abs_tol=1e-10
        )
        self.check(ok_tm_ts, f"{label}: timing.offset_bin_slices drift from raw rows: {diff_tm_ts}")

        rec_timed_slices = (
            suite_report.get("isolated_timing_offset_bin_slices")
            if "isolated_timing_offset_bin_slices" in suite_report
            else suite_report.get("closed_loop", {}).get("isolated_timing_offset_bin_slices", [])
        )
        ok_ts, diff_ts = close_nested(
            recomp_timed_slices, rec_timed_slices, rel_tol=1e-10, abs_tol=1e-10
        )
        self.check(ok_ts, f"{label}: isolated_timing_offset_bin_slices drift from raw rows: {diff_ts}")

        # Recompute matched_support_comparisons from raw recomputed timed groups & cross-check summary + suite_report!
        # Under `--allow_pilot` (`--timing_prompts > 0`), `eval_closed_loop.py` records
        # `summary["isolated_timing_complete"] = False` and omits pilot timing from matched-support / head-to-head
        # throughput comparisons (`timed=None`, `isolated_rows=[]`).
        pilot_incomplete_timing = (
            bool(self.allow_pilot or (self.allow_partial and not self.require_complete))
            and not bool(summary.get("isolated_timing_complete"))
        )
        recomp_msc = recompute_matched_support_comparisons(
            eval_rows,
            None if pilot_incomplete_timing else recomp_timed_groups,
            seed=self.seed,
        )
        ok_sum_msc, diff_sum_msc = close_nested(
            recomp_msc, summary.get("matched_support_comparisons", []), rel_tol=1e-10, abs_tol=1e-10
        )
        self.check(
            ok_sum_msc, f"{label}: summary.matched_support_comparisons drift from raw rows: {diff_sum_msc}"
        )

        rec_msc = (
            suite_report.get("matched_support_comparisons")
            if "matched_support_comparisons" in suite_report
            else suite_report.get("closed_loop", {}).get("matched_support_comparisons", [])
        )
        ok_msc, diff_msc = close_nested(recomp_msc, rec_msc, rel_tol=1e-10, abs_tol=1e-10)
        self.check(ok_msc, f"{label}: matched_support_comparisons drift from raw rows: {diff_msc}")

        recomp_h2h = recompute_head_to_head_online_offset_comparison(
            eval_rows,
            None if pilot_incomplete_timing else timed_rows,
            seed=self.seed,
        )
        ok_sum_h2h, diff_sum_h2h = close_nested(
            recomp_h2h,
            summary.get("head_to_head_online_offset_comparison", []),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(
            ok_sum_h2h,
            f"{label}: summary.head_to_head_online_offset_comparison drift from raw rows: {diff_sum_h2h}",
        )
        if check_head_to_head_offsets:
            rec_h2h = suite_report.get("head_to_head_online_offset_comparison", [])
            ok_h2h, diff_h2h = close_nested(recomp_h2h, rec_h2h, rel_tol=1e-10, abs_tol=1e-10)
            self.check(
                ok_h2h,
                f"{label}: head_to_head_online_offset_comparison drift from raw rows: {diff_h2h}",
            )

        # Build sparse bin flags (`< 8` prompts on `long_*` -> conclusion unavailable)
        sparse_offset_bins = []
        for sl in recomp_eval_slices:
            if not str(sl["condition"]).startswith("long_"):
                continue
            for b in sl["bins"]:
                n_reach = int(b["requests_reaching_bin"])
                if n_reach < SPARSE_LONG_BIN_MIN_PROMPTS:
                    sparse_offset_bins.append(
                        {
                            "condition": sl["condition"],
                            "temperature": sl["temperature"],
                            "arm": sl["arm"],
                            "bin": b["bin"],
                            "requests_reaching_bin": n_reach,
                            "gate_status": (
                                "EMPTY (conclusion unavailable)"
                                if n_reach == 0
                                else f"SPARSE (n={n_reach} < {SPARSE_LONG_BIN_MIN_PROMPTS}; conclusion unavailable)"
                            ),
                            "conclusion_available": False,
                        }
                    )

        return {
            "frozen_settings_hash": frozen_hash,
            "selected_settings": settings.get("selected", {}),
            "matched_c0_beta_relax": expected_c0_beta_relax,
            "evaluated_rows": len(eval_rows),
            "timed_rows": len(timed_rows),
            "profile_rows": len(profile_rows),
            "eval_groups": recomp_eval_groups,
            "timing_groups": recomp_timed_groups,
            "phase_profiles": recomp_phase_profiles,
            "eval_offset_slices": recomp_eval_slices,
            "timing_offset_slices": recomp_timed_slices,
            "matched_support_comparisons": recomp_msc,
            "head_to_head_offset_comparisons": (
                recomp_h2h if check_head_to_head_offsets else []
            ),
            "sparse_long_offset_bins": sparse_offset_bins,
        }

    def _audit_single_m5_row(
        self,
        *,
        label: str,
        row: dict[str, Any],
        expected_c0_beta_relax: float | None,
        expected_bigram_kappa: float | None = None,
        is_profile: bool,
    ) -> None:
        arm = row["arm"]
        pid = row["prompt_id"]
        # 1. Exact-cycle repetition detector v2 check on every `long_*` output
        if str(row.get("condition", "")).startswith("long_"):
            rep_recomp = independent_repetition_dominated(row.get("output_tokens", []))
            self.check(
                rep_recomp == bool(row.get("generation", {}).get("repetition_dominated", False)),
                f"{label}: repetition_dominated mismatch on {pid} ({arm})",
            )

        # 2. Matched C0[j] & lookahead telemetry invariants
        tel = row.get("m5_telemetry") or {}
        if arm == "dense_candidate" and expected_c0_beta_relax is not None:
            rec_br = tel.get("matched_c0_beta_relax")
            self.check(
                rec_br is not None
                and math.isclose(float(rec_br), expected_c0_beta_relax, rel_tol=1e-12, abs_tol=1e-12),
                f"{label}: dense_candidate row {pid} matched_c0_beta_relax={rec_br!r} != {expected_c0_beta_relax}",
            )
        elif arm in LOOKAHEAD_ARMS:
            la = tel.get("lookahead") or {}
            self.check(bool(la), f"{label}: lookahead arm {arm} ({pid}) missing m5_telemetry.lookahead")
            if la:
                self.check(
                    float(la.get("T_relax", 0.0)) > 0.0,
                    f"{label}: {arm} ({pid}) T_relax={la.get('T_relax')} must be > 0",
                )
                if arm in MATCHED_C0_CANDIDATE_ARMS and expected_c0_beta_relax is not None:
                    self.check(
                        la.get("domain") == "candidate_masked"
                        and math.isclose(
                            float(la.get("beta_relax", -1.0)),
                            expected_c0_beta_relax,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ),
                        f"{label}: {arm} ({pid}) lookahead domain/beta_relax mismatch: {la}",
                    )
            if is_profile and len(row.get("blocks") or []) > 0:
                la_ms = float((row.get("phase_ms") or {}).get("lookahead_iterations", 0.0))
                self.check(
                    la_ms > 0.0,
                    f"{label}: active lookahead profile row {arm} ({pid}) has lookahead_iterations={la_ms} <= 0",
                )

        # 3. Bigram invariants (`online_bigram`, `online_bigram_count_gated`)
        if arm in ("online_bigram", "online_bigram_count_gated"):
            bg = row.get("bigram")
            self.check(
                isinstance(bg, dict),
                f"{label}: bigram arm {arm} ({pid}) missing/invalid row['bigram']",
            )
            if isinstance(bg, dict):
                if arm == "online_bigram":
                    self.check(
                        "count_gate_kappa" not in bg,
                        f"{label}: online_bigram ({pid}) unexpectedly recorded count_gate_kappa={bg.get('count_gate_kappa')!r}",
                    )
                else:
                    rec_kp = bg.get("count_gate_kappa")
                    self.check(
                        not isinstance(rec_kp, bool)
                        and isinstance(rec_kp, (int, float))
                        and math.isfinite(float(rec_kp))
                        and float(rec_kp) > 0.0,
                        f"{label}: online_bigram_count_gated ({pid}) invalid count_gate_kappa={rec_kp!r}",
                    )
                    if expected_bigram_kappa is not None and isinstance(rec_kp, (int, float)):
                        self.check(
                            math.isclose(
                                float(rec_kp), float(expected_bigram_kappa), rel_tol=1e-12, abs_tol=1e-12
                            ),
                            f"{label}: online_bigram_count_gated ({pid}) count_gate_kappa={rec_kp} != {expected_bigram_kappa}",
                        )

        # 4. Online adapter arm invariants (strictly requiring `isinstance(onl, dict)`!)
        m5_online_adapter_arms = (
            "online_dbS",
            "online_dbS_zero_init",
            "zero_offline_online_only",
            "offline_online_stride1",
            "offline_online_stride5",
            "online_buffered_stride5",
            "online_two_sided_pl",
            "signed_feature_hash_online",
            "exact_observed_pairs_online",
            "online_rejection_relative",
            "online_cutoff",
        )
        onl = row.get("online")
        inv = row.get("online_invariants")
        if arm in m5_online_adapter_arms:
            self.check(
                isinstance(onl, dict),
                f"{label}: online arm {arm} ({pid}) missing/invalid row['online']={onl!r}",
            )
            self.check(
                isinstance(inv, dict)
                and inv.get("request_reset") is True
                and inv.get("version_only_between_rounds") is True,
                f"{label}: online arm {arm} ({pid}) missing/invalid online_invariants={inv!r}",
            )
        if arm in ("online_dbS", "online_dbS_zero_init") and isinstance(onl, dict):
            self.check(
                float(onl.get("dW_norm", 1.0)) == 0.0 and float(onl.get("dA_norm", 1.0)) == 0.0,
                f"{label}: {arm} ({pid}) violated dW_norm==0 / dA_norm==0",
            )
        if arm in ("online_dbS_zero_init", "zero_offline_online_only"):
            pnorms = row.get("parameter_norms")
            self.check(
                isinstance(pnorms, dict)
                and all(float(pnorms.get(k, 1.0)) == 0.0 for k in ("W", "A", "b")),
                f"{label}: zero-init arm {arm} ({pid}) has non-zero offline parameter_norms={pnorms}",
            )
        if arm == "exact_observed_pairs_online" and isinstance(onl, dict):
            self.check(
                onl.get("sparse_backend") == "exact_observed_pairs"
                and onl.get("sparse_snapshot_strategy") == "compact_active_exact_pair_copy",
                f"{label}: exact_observed_pairs_online ({pid}) wrong sparse_backend metadata: {onl.get('sparse_backend')}",
            )
            cap = onl.get("sparse_capacity")
            ovf = onl.get("sparse_overflow_events")
            dropped = onl.get("sparse_dropped_unique_keys")
            peak_act = onl.get("sparse_peak_active")
            act_b = onl.get("active_buckets")
            self.check(
                type(cap) is int and cap > 0,
                f"{label}: exact_observed_pairs_online ({pid}) invalid sparse_capacity={cap!r}",
            )
            self.check(
                type(ovf) is int and ovf >= 0,
                f"{label}: exact_observed_pairs_online ({pid}) invalid sparse_overflow_events={ovf!r}",
            )
            self.check(
                type(dropped) is int and dropped >= 0,
                f"{label}: exact_observed_pairs_online ({pid}) invalid sparse_dropped_unique_keys={dropped!r}",
            )
            self.check(
                type(peak_act) is int
                and type(cap) is int
                and 0 <= peak_act <= cap
                and type(act_b) is int
                and 0 <= act_b <= peak_act,
                f"{label}: exact_observed_pairs_online ({pid}) invalid sparse_peak_active={peak_act!r} vs active_buckets={act_b!r}, capacity={cap!r}",
            )
            self.check(
                type(ovf) is int
                and type(dropped) is int
                and ((ovf == 0) == (dropped == 0))
                and dropped >= ovf,
                f"{label}: exact_observed_pairs_online ({pid}) inconsistent sparse_overflow_events={ovf!r} vs sparse_dropped_unique_keys={dropped!r}",
            )
        if arm == "online_two_sided_pl" and isinstance(onl, dict):
            self.check(
                onl.get("loss_mode") == "two_sided_pseudolikelihood",
                f"{label}: online_two_sided_pl ({pid}) loss_mode={onl.get('loss_mode')!r}",
            )
        if arm in ("offline_online_stride5", "online_buffered_stride5") and isinstance(onl, dict):
            n_blk = len(row.get("blocks") or [])
            self.check(
                int(onl.get("u", -1)) == n_blk // 5 and int(onl.get("adapter_version", -1)) == n_blk // 5,
                f"{label}: {arm} ({pid}) stride-5 update count u={onl.get('u')} != {n_blk // 5}",
            )
            if arm == "offline_online_stride5":
                self.check(
                    onl.get("accumulate_stride", False) is False
                    and isinstance(inv, dict)
                    and inv.get("single_round_no_buffering") is True
                    and "buffered_stride_accumulation" not in inv,
                    f"{label}: offline_online_stride5 ({pid}) violated unbuffered single-round stride-5 invariant",
                )
            elif arm == "online_buffered_stride5":
                acc_tot = int(onl.get("buffered_rounds_accumulated_total", -1))
                pend = int(onl.get("buffered_pending_rounds", -1))
                self.check(
                    onl.get("accumulate_stride") is True
                    and isinstance(inv, dict)
                    and inv.get("single_round_no_buffering") is False
                    and inv.get("buffered_stride_accumulation") is True
                    and acc_tot == (n_blk // 5) * 5
                    and pend == (n_blk % 5)
                    and (acc_tot + pend) == n_blk,
                    f"{label}: online_buffered_stride5 ({pid}) accumulate_stride/buffered_rounds mismatch: acc={acc_tot}, pend={pend}, n_blk={n_blk}",
                )

    # ------------------------------------------------------------------
    # 2. Suite: `online_head_to_head`
    # ------------------------------------------------------------------

    def audit_online_head_to_head(self) -> dict[str, Any]:
        report_path = self.suite_root / "online_head_to_head.json"
        if not report_path.is_file():
            msg = "online_head_to_head.json not yet present"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "not_started_or_in_progress"}
            self.check(False, msg)
            return {"status": "missing"}

        rep = self.read_json_tracked(report_path)
        self.check(rep.get("suite") == "online_head_to_head", "online_h2h: wrong suite tag")
        self.check(rep.get("scale") == self.scale, "online_h2h: scale mismatch")
        allowed_statuses = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                rep.get("status") in allowed_statuses,
                f"online_h2h: status {rep.get('status')!r} not in {allowed_statuses}",
            )
        kappa = rep.get("bigram_count_gate_kappa")
        kappa_ok = (
            not isinstance(kappa, bool)
            and isinstance(kappa, (int, float))
            and math.isfinite(float(kappa))
            and float(kappa) > 0.0
        )
        self.check(kappa_ok, f"online_h2h: invalid bigram_count_gate_kappa={kappa!r}")

        ws_audit = self._audit_m5_stage_workspace(
            label="online_head_to_head",
            suite_report=rep,
            artifacts_block=rep.get("artifacts", {}),
            expected_arms=list(rep.get("arms", [])),
            check_head_to_head_offsets=True,
            expected_bigram_kappa=float(kappa) if kappa_ok else None,
        )
        return {
            "status": rep.get("status"),
            "config_fingerprint": rep.get("config_fingerprint"),
            "timing_world_size": rep.get("timing_world_size"),
            "sharding_mode": rep.get("sharding_mode"),
            "bigram_count_gate_kappa": kappa,
            "arms": rep.get("arms", []),
            "workspace_audit": ws_audit,
        }

    # ------------------------------------------------------------------
    # 3. Suite: `m5_eval`
    # ------------------------------------------------------------------

    def audit_m5_eval(self) -> dict[str, Any]:
        report_path = self.suite_root / "m5_eval.json"
        if not report_path.is_file():
            msg = "m5_eval.json not yet present"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "not_started_or_in_progress"}
            self.check(False, msg)
            return {"status": "missing"}

        rep = self.read_json_tracked(report_path)
        self.check(rep.get("suite") == "m5_eval", "m5_eval: wrong suite tag")
        self.check(rep.get("scale") == self.scale, "m5_eval: scale mismatch")
        allowed_statuses = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                rep.get("status") in allowed_statuses,
                f"m5_eval: status {rep.get('status')!r} not in {allowed_statuses}",
            )

        def _is_nonneg_num(val: Any) -> bool:
            return (
                not isinstance(val, bool)
                and isinstance(val, (int, float))
                and math.isfinite(float(val))
                and float(val) >= 0.0
            )

        def _is_finite_num(val: Any) -> bool:
            return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(float(val))

        # 1. Offline fits: `two_sided_pseudolikelihood` and `exact_observed_pairs` (`collision_count == 0`!)
        off = rep.get("offline_fits", {})
        base_sh = off.get("baseline_signed_hash", {})
        pl = off.get("two_sided_pseudolikelihood", {})
        exact = off.get("exact_observed_pairs", {})

        for sp_k in ("dev", "test"):
            v_sh = (base_sh.get(sp_k) or {}).get("weighted_mean_kl")
            self.check(
                _is_nonneg_num(v_sh),
                f"m5_eval: baseline_signed_hash {sp_k} weighted_mean_kl invalid: {v_sh!r}",
            )

        self.check(
            pl.get("fit_status") in ("passed", "complete"),
            f"m5_eval: two_sided_pseudolikelihood fit_status={pl.get('fit_status')!r}",
        )
        pl_rep_p = Path(pl.get("report_path", ""))
        self.check(pl_rep_p.is_file(), f"m5_eval: missing two_sided report {pl_rep_p}")
        if pl_rep_p.is_file():
            pl_fit_json = self.read_json_tracked(pl_rep_p)
            self.check(
                pl_fit_json.get("status") in ("passed", "complete"),
                f"m5_eval: underlying m5_two_sided_pl.json status={pl_fit_json.get('status')!r}",
            )
            self.check(
                pl_fit_json.get("scale") == self.scale,
                f"m5_eval: underlying m5_two_sided_pl.json scale={pl_fit_json.get('scale')!r} != {self.scale!r}",
            )
            self.check(
                pl_fit_json.get("objective") == "two_sided_pseudolikelihood",
                f"m5_eval: underlying m5_two_sided_pl.json objective={pl_fit_json.get('objective')!r} != 'two_sided_pseudolikelihood'",
            )
            self.check(
                "verified stream only" in str(pl_fit_json.get("post_rejection_canvas", "")),
                f"m5_eval: underlying m5_two_sided_pl.json invalid post_rejection_canvas={pl_fit_json.get('post_rejection_canvas')!r}",
            )
        for sp_k in ("dev", "test"):
            v = (pl.get("held_out_union_rest_grouped_kl", {}).get(sp_k) or {}).get("weighted_mean_kl")
            self.check(
                _is_nonneg_num(v),
                f"m5_eval: two_sided {sp_k} weighted_mean_kl invalid: {v!r}",
            )

        # Strict ExactObservedPairs collision-freedom & KL checks!
        self.check(
            exact.get("fit_status") in ("passed", "complete"),
            f"m5_eval: exact_observed_pairs fit_status={exact.get('fit_status')!r}",
        )
        top_cc = exact.get("collision_count")
        self.check(
            type(top_cc) is int and top_cc == 0,
            f"m5_eval: ExactObservedPairs top-level collision_count={top_cc!r} != 0 (int)",
        )
        ex_kl_block = exact.get("held_out_union_rest_grouped_kl") or {}
        kl_cc = ex_kl_block.get("collision_count")
        self.check(
            type(kl_cc) is int and kl_cc == 0,
            f"m5_eval: ExactObservedPairs held_out_union_rest_grouped_kl.collision_count={kl_cc!r} != 0 (int)",
        )
        self.check(
            ex_kl_block.get("sparse_backend") == "exact_observed_pairs",
            f"m5_eval: ExactObservedPairs held_out_union_rest_grouped_kl.sparse_backend={ex_kl_block.get('sparse_backend')!r}",
        )
        for sp_k in ("dev", "test"):
            v_ex = (ex_kl_block.get(sp_k) or {}).get("weighted_mean_kl")
            self.check(
                _is_nonneg_num(v_ex),
                f"m5_eval: exact_observed_pairs {sp_k} weighted_mean_kl invalid: {v_ex!r}",
            )
        exact_rep_p = Path(exact.get("report_path", ""))
        self.check(exact_rep_p.is_file(), f"m5_eval: missing exact_observed_pairs report {exact_rep_p}")
        if exact_rep_p.is_file():
            ex_fit_json = self.read_json_tracked(exact_rep_p)
            self.check(
                ex_fit_json.get("status") in ("passed", "complete"),
                f"m5_eval: underlying m5_exact_pairs.json status={ex_fit_json.get('status')!r}",
            )
            self.check(
                ex_fit_json.get("scale") == self.scale,
                f"m5_eval: underlying m5_exact_pairs.json scale={ex_fit_json.get('scale')!r} != {self.scale!r}",
            )
            self.check(
                ex_fit_json.get("sparse_backend") == "exact_observed_pairs",
                f"m5_eval: underlying m5_exact_pairs.json sparse_backend={ex_fit_json.get('sparse_backend')!r}",
            )
            sp_stats = (
                ex_fit_json.get("stages", {})
                .get("dense_sparse", {})
                .get("sparse_collision_statistics", {})
            )
            self.check(
                sp_stats.get("status") == "measured",
                f"m5_eval: ExactObservedPairs sparse_collision_statistics.status={sp_stats.get('status')!r} != 'measured'",
            )
            self.check(
                type(sp_stats.get("collision_count")) is int and sp_stats.get("collision_count") == 0,
                f"m5_eval: underlying m5_exact_pairs.json collision_count={sp_stats.get('collision_count')!r} != int(0)",
            )
            self.check(
                sp_stats.get("representation") == "exact_observed_pairs",
                f"m5_eval: ExactObservedPairs representation={sp_stats.get('representation')!r} != 'exact_observed_pairs'",
            )
            self.check(
                type(sp_stats.get("active_representation_collision_count")) is int
                and sp_stats.get("active_representation_collision_count") == 0,
                f"m5_eval: ExactObservedPairs active_representation_collision_count={sp_stats.get('active_representation_collision_count')!r} != int(0)",
            )
            self.check(
                type(sp_stats.get("frequency_weighted_collision_count")) is int
                and sp_stats.get("frequency_weighted_collision_count") == 0,
                f"m5_eval: ExactObservedPairs frequency_weighted_collision_count={sp_stats.get('frequency_weighted_collision_count')!r} != int(0)",
            )
            self.check(
                _is_nonneg_num(sp_stats.get("collision_rate_per_nonempty_bucket"))
                and float(sp_stats["collision_rate_per_nonempty_bucket"]) == 0.0,
                f"m5_eval: ExactObservedPairs collision_rate_per_nonempty_bucket={sp_stats.get('collision_rate_per_nonempty_bucket')!r} != 0.0",
            )
            self.check(
                _is_nonneg_num(sp_stats.get("frequency_weighted_collision_rate"))
                and float(sp_stats["frequency_weighted_collision_rate"]) == 0.0,
                f"m5_eval: ExactObservedPairs frequency_weighted_collision_rate={sp_stats.get('frequency_weighted_collision_rate')!r} != 0.0",
            )

        # Verify delta algebra vs baseline_signed_hash unconditionally!
        ex_test_kl = (ex_kl_block.get("test") or {}).get("weighted_mean_kl")
        sh_test_kl = (base_sh.get("test") or {}).get("weighted_mean_kl")
        rec_delta = exact.get("test_grouped_kl_delta_vs_signed_hash")
        self.check(
            _is_finite_num(rec_delta),
            f"m5_eval: missing/invalid test_grouped_kl_delta_vs_signed_hash={rec_delta!r}",
        )
        if _is_nonneg_num(ex_test_kl) and _is_nonneg_num(sh_test_kl) and _is_finite_num(rec_delta):
            self.check(
                math.isclose(
                    float(ex_test_kl) - float(sh_test_kl),
                    float(rec_delta),
                    rel_tol=1e-11,
                    abs_tol=1e-11,
                ),
                f"m5_eval: test_grouped_kl_delta_vs_signed_hash={rec_delta} != {float(ex_test_kl) - float(sh_test_kl)}",
            )

        # 2. Closed-loop M5 workspace audit
        cl = rep.get("closed_loop", {})
        ws_audit = self._audit_m5_stage_workspace(
            label="m5_eval",
            suite_report=rep,
            artifacts_block=cl.get("artifacts", {}),
            expected_arms=list(rep.get("arms", [])),
            check_head_to_head_offsets=False,
        )
        return {
            "status": rep.get("status"),
            "config_fingerprint": rep.get("config_fingerprint"),
            "timing_world_size": rep.get("timing_world_size"),
            "sharding_mode": rep.get("sharding_mode"),
            "arms": rep.get("arms", []),
            "offline_fits": off,
            "workspace_audit": ws_audit,
        }

    # ------------------------------------------------------------------
    # 4. Suite: `fastpath_bench`
    # ------------------------------------------------------------------

    def audit_fastpath_bench(self) -> dict[str, Any]:
        report_path = self.suite_root / "fastpath_bench.json"
        if not report_path.is_file():
            msg = "fastpath_bench.json not yet present"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "not_started_or_in_progress"}
            self.check(False, msg)
            return {"status": "missing"}

        rep = self.read_json_tracked(report_path)
        self.check(rep.get("suite") == "fastpath_bench", "fastpath_bench: wrong suite tag")
        self.check(rep.get("scale") == self.scale, "fastpath_bench: scale mismatch")
        allowed_statuses = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                rep.get("status") in allowed_statuses,
                f"fastpath_bench: status {rep.get('status')!r} not in {allowed_statuses}",
            )

        # Microbenchmarks check
        gr = rep.get("grouped_kl_microbenchmark", {})
        self.check(gr.get("parity_passed") is True, "fastpath_bench: grouped_kl parity_passed != True")
        self.check(
            bool(rep.get("round3_compact_microbenchmark")),
            "fastpath_bench: missing round3_compact_microbenchmark",
        )
        self.check(
            bool(rep.get("round4_online_microbenchmark")),
            "fastpath_bench: missing round4_online_microbenchmark (stale schema < 3)",
        )
        self.check(
            bool(rep.get("round5_lookahead_microbenchmark")),
            "fastpath_bench: missing round5_lookahead_microbenchmark (stale schema < 3)",
        )

        # Closed-loop matched comparison (`eager_pre_round3` vs `round3_compact` vs `round3_compile`)
        cl = rep.get("closed_loop_matched_comparison", {})
        if self.require_complete and not self.allow_pilot:
            self.check(
                cl.get("is_pilot_subset") is False,
                "fastpath_bench: closed_loop_matched_comparison is a pilot subset",
            )
        speed_rows = cl.get("speed_rows", [])
        profile_rows = cl.get("profile_rows", [])
        arms = cl.get("arms", [])
        modes = cl.get("modes", [])
        self.check(
            tuple(modes) == ("eager_pre_round3", "round3_compact", "round3_compile"),
            f"fastpath_bench: unexpected modes {modes}",
        )
        for rw in speed_rows + profile_rows:
            if str(rw.get("condition", "")).startswith("long_") and "output_tokens" in rw:
                self.check(
                    independent_repetition_dominated(rw["output_tokens"])
                    == _row_rep_flag(rw),
                    f"fastpath_bench: repetition_dominated mismatch on {rw['prompt_id']}",
                )

        # Recompute summary_by_mode_and_arm & paired_speedups from raw speed_rows!
        recomp_by_ma = []
        for m_name in modes:
            for arm in arms:
                m_rows = [r for r in speed_rows if r["fastpath_mode"] == m_name and r["arm"] == arm]
                m_prim = [r for r in m_rows if not _is_rep_excluded(r)]
                p_rows = [r for r in profile_rows if r["fastpath_mode"] == m_name and r["arm"] == arm]
                p_prim = [r for r in p_rows if not _is_rep_excluded(r)]
                ph_keys = sorted({k for r in p_prim for k in (r.get("phase_ms") or {})})
                ph_means = {
                    k: statistics.mean(float(r["phase_ms"][k]) for r in p_prim if k in r["phase_ms"])
                    for k in ph_keys
                }
                by_cond = {}
                for cond in sorted({r["condition"] for r in m_rows}):
                    c_all = [r for r in m_rows if r["condition"] == cond]
                    c_pr = [r for r in c_all if not _is_rep_excluded(r)]
                    by_cond[cond] = {
                        "timed_runs": len(c_all),
                        "primary_timed_runs": len(c_pr),
                        "repetition_excluded": len(c_all) - len(c_pr),
                        "mean_tokens_per_second": (
                            statistics.mean(r["tokens_per_second"] for r in c_pr) if c_pr else None
                        ),
                        "median_tokens_per_second": (
                            statistics.median(r["tokens_per_second"] for r in c_pr) if c_pr else None
                        ),
                        "mean_accepted_length": (
                            statistics.mean(r["mean_accepted_length"] for r in c_pr)
                            if c_pr
                            else None
                        ),
                        "all_runs_mean_tokens_per_second": (
                            statistics.mean(r["tokens_per_second"] for r in c_all)
                            if c_all
                            else None
                        ),
                        "all_runs_mean_accepted_length": (
                            statistics.mean(r["mean_accepted_length"] for r in c_all)
                            if c_all
                            else None
                        ),
                    }
                recomp_by_ma.append(
                    {
                        "fastpath_mode": m_name,
                        "arm": arm,
                        "timed_runs": len(m_rows),
                        "primary_timed_runs": len(m_prim),
                        "repetition_excluded": len(m_rows) - len(m_prim),
                        "mean_tokens_per_second": (
                            statistics.mean(r["tokens_per_second"] for r in m_prim)
                            if m_prim
                            else None
                        ),
                        "median_tokens_per_second": (
                            statistics.median(r["tokens_per_second"] for r in m_prim)
                            if m_prim
                            else None
                        ),
                        "mean_accepted_length": (
                            statistics.mean(r["mean_accepted_length"] for r in m_prim)
                            if m_prim
                            else None
                        ),
                        "all_runs_mean_tokens_per_second": (
                            statistics.mean(r["tokens_per_second"] for r in m_rows)
                            if m_rows
                            else None
                        ),
                        "all_runs_mean_accepted_length": (
                            statistics.mean(r["mean_accepted_length"] for r in m_rows)
                            if m_rows
                            else None
                        ),
                        "by_condition": by_cond,
                        "mean_phase_ms": ph_means,
                    }
                )
        ok_ma, diff_ma = close_nested(
            recomp_by_ma, cl.get("summary_by_mode_and_arm", []), rel_tol=1e-10, abs_tol=1e-10
        )
        self.check(
            ok_ma, f"fastpath_bench: summary_by_mode_and_arm drift from speed_rows: {diff_ma}"
        )

        by_slot = {
            (r["fastpath_mode"], r["arm"], r["prompt_id"], r["repeat"]): r for r in speed_rows
        }
        recomp_ps = []
        for arm in arms:
            r3_e, c_r3, c_e = [], [], []
            excl_p = 0
            by_c: dict[str, dict[str, list[float]]] = {}
            for (mn, an, pid, rep_idx), r_eg in by_slot.items():
                tps_eg = r_eg["tokens_per_second"]
                if mn != "eager_pre_round3" or an != arm or tps_eg <= 0:
                    continue
                r_r3 = by_slot.get(("round3_compact", arm, pid, rep_idx))
                r_cm = by_slot.get(("round3_compile", arm, pid, rep_idx))
                cb = by_c.setdefault(
                    r_eg["condition"],
                    {"r3_over_eager": [], "comp_over_r3": [], "comp_over_eager": []},
                )
                if any(x is not None and _is_rep_excluded(x) for x in (r_eg, r_r3, r_cm)):
                    excl_p += 1
                    continue
                if r_r3 is not None:
                    v1 = r_r3["tokens_per_second"] / tps_eg
                    r3_e.append(v1)
                    cb["r3_over_eager"].append(v1)
                if r_cm is not None and r_r3 is not None and r_r3["tokens_per_second"] > 0:
                    v2 = r_cm["tokens_per_second"] / r_r3["tokens_per_second"]
                    c_r3.append(v2)
                    cb["comp_over_r3"].append(v2)
                if r_cm is not None:
                    v3 = r_cm["tokens_per_second"] / tps_eg
                    c_e.append(v3)
                    cb["comp_over_eager"].append(v3)
            recomp_ps.append(
                {
                    "arm": arm,
                    "round3_over_eager_mean_ratio": statistics.mean(r3_e) if r3_e else None,
                    "compile_over_round3_mean_ratio": statistics.mean(c_r3) if c_r3 else None,
                    "compile_over_eager_mean_ratio": statistics.mean(c_e) if c_e else None,
                    "pairs": len(r3_e),
                    "repetition_excluded_pairs": excl_p,
                    "by_condition": {
                        c: {
                            "round3_over_eager_mean_ratio": (
                                statistics.mean(b["r3_over_eager"]) if b["r3_over_eager"] else None
                            ),
                            "compile_over_round3_mean_ratio": (
                                statistics.mean(b["comp_over_r3"]) if b["comp_over_r3"] else None
                            ),
                            "compile_over_eager_mean_ratio": (
                                statistics.mean(b["comp_over_eager"])
                                if b["comp_over_eager"]
                                else None
                            ),
                            "pairs": len(b["r3_over_eager"]),
                        }
                        for c, b in sorted(by_c.items())
                    },
                }
            )
        ok_ps, diff_ps = close_nested(
            recomp_ps, cl.get("paired_speedups", []), rel_tol=1e-10, abs_tol=1e-10
        )
        self.check(ok_ps, f"fastpath_bench: paired_speedups drift from speed_rows: {diff_ps}")

        # 1-GPU isolated vs W-GPU sharded calibration check!
        gpus = rep.get("gpus", [])
        cst = rep.get("cluster_sharded_throughput")
        if len(gpus) > 1:
            self.check(
                isinstance(cst, dict),
                f"fastpath_bench: gpus={gpus} (W>1) requires cluster_sharded_throughput calibration",
            )
        if isinstance(cst, dict):
            eff_w = int(cst.get("world_size", 1))
            r_aff = cst.get("rank_affinity") or cl.get("rank_affinity") or {}
            self.check(
                len(r_aff) == eff_w,
                f"fastpath_bench: rank_affinity count {len(r_aff)} != world_size {eff_w}",
            )
            seen_cores: set[int] = set()
            seen_gpus: set[int] = set()
            is_cuda = str(rep.get("device", "")).startswith("cuda")
            for rk_s, aff in sorted(r_aff.items()):
                pg = int(aff["physical_gpu"])
                seen_gpus.add(pg)
                cores = list(aff.get("cpu_affinity") or [])
                if is_cuda and self.scale != "smoke":
                    self.check(
                        aff.get("cpu_affinity_applied") is True and len(cores) == 16,
                        f"fastpath_bench: rank {rk_s} missing 16-core pinning: {aff}",
                    )
                    self.check(
                        int(aff.get("numa_node", -1)) == pg // 2,
                        f"fastpath_bench: rank {rk_s} GPU {pg} NUMA {aff.get('numa_node')} != {pg // 2}",
                    )
                    self.check(
                        not (seen_cores & set(cores)),
                        f"fastpath_bench: overlapping CPU cores on rank {rk_s}: {cores}",
                    )
                seen_cores.update(cores)
            self.check(
                len(seen_gpus) == eff_w,
                f"fastpath_bench: non-distinct physical_gpus across ranks: {seen_gpus}",
            )

            # Locate raw 1-GPU calibration file `cluster_1gpu_calibration/result-cal-rank0.json`
            fb_ws = self.suite_root / f"fastpath_bench_{str(rep.get('config_fingerprint', ''))[:12]}"
            if not fb_ws.is_dir() and (self.suite_root / "fastpath_bench").exists():
                fb_ws = (self.suite_root / "fastpath_bench").resolve()
            cal_file = fb_ws / "cluster_1gpu_calibration" / "result-cal-rank0.json"
            self.check(
                cal_file.is_file(),
                f"fastpath_bench: missing raw 1-GPU calibration file {cal_file}",
            )
            if cal_file.is_file():
                cal_payload = self.read_json_tracked(cal_file)
                cal_rows = cal_payload.get("rows", [])
                cal_pids = set(cal_payload.get("selected_prompt_ids", []))
                self.check(
                    cal_payload.get("selected_prompt_ids") == cst.get("calibration_selected_prompt_ids"),
                    "fastpath_bench: calibration_selected_prompt_ids mismatch vs raw 1-GPU payload",
                )
                r3_sharded = [r for r in speed_rows if r["fastpath_mode"] == "round3_compact"]
                r3_wall_per_rank = [
                    sum(float(r["elapsed_seconds"]) for r in r3_sharded if r.get("rank", 0) == rk)
                    for rk in range(eff_w)
                ]
                max_r3_sec = max(r3_wall_per_rank, default=0.0)
                tot_tok = sum(int(r["generated_tokens"]) for r in r3_sharded)
                exp_cluster_tps = tot_tok / max(max_r3_sec, 1e-9)
                self.check(
                    math.isclose(
                        exp_cluster_tps,
                        float(cst["aggregate_cluster_tokens_per_second"]),
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    ),
                    f"fastpath_bench: aggregate_cluster_tokens_per_second {cst['aggregate_cluster_tokens_per_second']} != raw {exp_cluster_tps}",
                )
                recomp_by_arm = []
                for arm in arms:
                    a_sh = [r for r in r3_sharded if r["arm"] == arm and not _is_rep_excluded(r)]
                    rk_means = []
                    for rk in range(eff_w):
                        rk_r = [rw for rw in a_sh if rw.get("rank", 0) == rk]
                        if rk_r:
                            rk_means.append(statistics.mean(rw["tokens_per_second"] for rw in rk_r))
                    sum_rk = sum(rk_means) if rk_means else None
                    a_c1 = [r for r in cal_rows if r["arm"] == arm and not _is_rep_excluded(r)]
                    a_cw = [r for r in a_sh if r["prompt_id"] in cal_pids]
                    tps_1 = statistics.mean(r["tokens_per_second"] for r in a_c1) if a_c1 else None
                    tps_w = statistics.mean(r["tokens_per_second"] for r in a_cw) if a_cw else None
                    pg_ratio = (tps_w / tps_1) if (tps_w is not None and tps_1) else None
                    recomp_by_arm.append(
                        {
                            "arm": arm,
                            "primary_timed_runs": len(a_sh),
                            "mean_per_rank_tokens_per_second": (
                                statistics.mean(rk_means) if rk_means else None
                            ),
                            "sum_across_ranks_tokens_per_second": sum_rk,
                            "calibration_1gpu_tokens_per_second": tps_1,
                            "calibration_sharded_per_gpu_tokens_per_second": tps_w,
                            "sharded_over_1gpu_per_gpu_ratio": pg_ratio,
                            "cluster_over_1gpu_throughput_ratio": (
                                eff_w * pg_ratio if pg_ratio is not None else None
                            ),
                            "per_gpu_scaling_efficiency": pg_ratio,
                        }
                    )
                ok_ca, diff_ca = close_nested(
                    recomp_by_arm, cst.get("by_arm", []), rel_tol=1e-10, abs_tol=1e-10
                )
                self.check(
                    ok_ca,
                    f"fastpath_bench: cluster_sharded_throughput.by_arm drift from raw 1-GPU/8-GPU calibration: {diff_ca}",
                )

        return {
            "status": rep.get("status"),
            "config_fingerprint": rep.get("config_fingerprint"),
            "device": rep.get("device"),
            "gpus": gpus,
            "adapter_microbenchmark": rep.get("adapter_microbenchmark"),
            "grouped_kl_microbenchmark": gr,
            "round3_compact_microbenchmark": rep.get("round3_compact_microbenchmark"),
            "round4_online_microbenchmark": rep.get("round4_online_microbenchmark"),
            "round5_lookahead_microbenchmark": rep.get("round5_lookahead_microbenchmark"),
            "closed_loop_summary_by_mode_and_arm": recomp_by_ma,
            "closed_loop_paired_speedups": recomp_ps,
            "cluster_sharded_throughput": cst,
        }

    # ------------------------------------------------------------------
    # 5. Master Summary (`post_m4_suite_summary.json`)
    # ------------------------------------------------------------------

    def audit_master_summary(self, suite_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        master_path = self.suite_root / "post_m4_suite_summary.json"
        if not master_path.is_file():
            msg = "post_m4_suite_summary.json not yet present"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "not_started_or_in_progress"}
            self.check(False, msg)
            return {"status": "missing"}

        master = self.read_json_tracked(master_path)
        allowed_statuses = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                master.get("status") in allowed_statuses,
                f"post_m4_suite_summary: status {master.get('status')!r} not in {allowed_statuses}",
            )
            self.check(
                set(master.get("executed_suites", [])) == set(SUITE_NAMES),
                f"post_m4_suite_summary: executed_suites {master.get('executed_suites')} != all 4 suites",
            )

        ms_suites = master.get("suites", {})
        for s_name, s_audit in suite_results.items():
            if s_name in ms_suites and s_audit.get("config_fingerprint"):
                self.check(
                    ms_suites[s_name].get("config_fingerprint") == s_audit["config_fingerprint"],
                    f"post_m4_suite_summary: {s_name} config_fingerprint mismatch vs standalone {s_name}.json",
                )
        return {
            "status": master.get("status"),
            "executed_suites": master.get("executed_suites"),
            "project_git_commit": master.get("project_git_commit"),
            "implementation_hashes": master.get("implementation_hashes"),
        }

    def audit_implementation_audit_freshness(self) -> dict[str, Any]:
        """Verify `artifacts/implementation_audit.json` freshness when `--check_implementation_audit` is enabled."""
        from build_implementation_audit import verify_implementation_audit

        ia_p = self.artifacts / "implementation_audit.json"
        if (
            not ia_p.is_file()
            and self.check_implementation_audit
            and (self.repo / "artifacts" / "implementation_audit.json").is_file()
        ):
            ia_p = self.repo / "artifacts" / "implementation_audit.json"
        if not ia_p.is_file():
            if self.check_implementation_audit:
                self.check(False, f"missing implementation_audit.json at {ia_p}")
            return {"present": False}
        ia_data = self.read_json_tracked(ia_p)
        errs = verify_implementation_audit(ia_data, self.repo, self.artifacts, check_git_head=True)
        if self.check_implementation_audit:
            self.check(
                not errs,
                f"implementation_audit.json stale/mismatched ({len(errs)} errors): {errs[:5]}",
            )
        elif errs:
            self.warn(
                f"implementation_audit.json currently has {len(errs)} pre-regeneration staleness items (run build_implementation_audit.py before final commit)"
            )
        return {
            "present": True,
            "path": str(ia_p),
            "status": ia_data.get("status"),
            "project_git_commit": ia_data.get("project_git_commit"),
            "staleness_error_count": len(errs),
        }

    def run_all(self) -> dict[str, Any]:
        upstream = self.audit_upstream_m0_m4_immutability()
        impl_audit = self.audit_implementation_audit_freshness()
        suites = {
            "m2_refit_calibration": self.audit_m2_refit_calibration(),
            "online_head_to_head": self.audit_online_head_to_head(),
            "m5_eval": self.audit_m5_eval(),
            "fastpath_bench": self.audit_fastpath_bench(),
        }
        master = self.audit_master_summary(suites)
        self.verify_immutability_at_end()
        return {
            "status": "passed" if not self.failures else "failed",
            "scale": self.scale,
            "repo": str(self.repo),
            "artifacts": str(self.artifacts),
            "suite_root": str(self.suite_root),
            "allow_partial": self.allow_partial,
            "require_complete": self.require_complete,
            "checks": self.checks,
            "failures": self.failures,
            "warnings": self.warnings,
            "audited_file_count": len(self.read_sha256),
            "upstream_m0_m4_immutability": upstream,
            "implementation_audit_freshness": impl_audit,
            "suites": suites,
            "master_summary": master,
        }


# ---------------------------------------------------------------------------
# Publication-Ready Markdown Renderer (`reports/SUMMARY.md` & `README.md`)
# ---------------------------------------------------------------------------


def _fmt_num(val: float | None, digits: int = 4) -> str:
    if val is None:
        return "—"
    return f"{float(val):.{digits}f}"


def render_post_m4_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Deterministically Recomputed Post-M4 & M5 Evidence Tables (`scale={report['scale']}`)",
        "",
        "> Generated strictly from verified `post_m4_suite` raw artifacts and sidecars (`0` manual transcription).",
        "",
    ]
    suites = report["suites"]

    # 1. Suite: `m2_refit_calibration`
    s1 = suites.get("m2_refit_calibration", {})
    lines.append("## 1. M2 Offline Refit & Post-Hoc Calibration (`m2_refit_calibration`)")
    lines.append("")
    if not s1.get("variants"):
        lines.append(f"- Status: `{s1.get('status', 'pending')}`")
        lines.append("")
    else:
        bcal = s1.get("beta_calibration") or {}
        if bcal:
            d_b = bcal.get("dense", {}).get("optimal_beta", bcal.get("dense", {}).get("selected_beta"))
            s_b = bcal.get("dense_sparse", {}).get("optimal_beta", bcal.get("dense_sparse", {}).get("selected_beta"))
            lines.append(
                f"- Status: **`{s1.get('status')}`** | Dev-calibrated `beta*`: dense `{d_b}`, dense+sparse `{s_b}`"
            )
            lines.append("")
        lines.extend(
            [
                "| Variant | Dense Dev KL | Dense Test KL | Sparse Dev KL | Sparse Test KL | CL Dense Acc (`Δ vs plain`) | CL Sparse Acc (`Δ vs plain`) | CL Sparse `tok/s` (`ratio`) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for v in s1["variants"]:
            cl = v["closed_loop"]
            d_cl = cl.get("dense", {})
            s_cl = cl.get("dense_sparse", {})
            lines.append(
                f"| `{v['variant']}` | {_fmt_num(v['dense_dev_kl'], 4)} | {_fmt_num(v['dense_test_kl'], 4)} "
                f"| {_fmt_num(v['sparse_dev_kl'], 4)} | {_fmt_num(v['sparse_test_kl'], 4)} "
                f"| {_fmt_num(d_cl.get('mean_accepted_length'), 3)} (`{_fmt_num(d_cl.get('paired_accepted_length_delta_vs_plain'), 3)}`) "
                f"| {_fmt_num(s_cl.get('mean_accepted_length'), 3)} (`{_fmt_num(s_cl.get('paired_accepted_length_delta_vs_plain'), 3)}`) "
                f"| {_fmt_num(s_cl.get('mean_tokens_per_second'), 2)} (`{_fmt_num(s_cl.get('paired_tps_ratio_vs_plain'), 3)}x`) |"
            )
        lines.append("")
        p_bc = (s1["variants"][0]["closed_loop"].get("plain_dflash") or {}).get("by_condition") or {}
        if "standard_nonthinking" in p_bc and "long_thinking" in p_bc:
            p_std = p_bc["standard_nonthinking"]
            p_lng = p_bc["long_thinking"]
            lines.extend(
                [
                    f"### Suite 1 Closed-Loop Breakdown by Condition (`plain_dflash`: `std = {_fmt_num(p_std.get('mean_accepted_length'), 3)}` / `{_fmt_num(p_std.get('mean_tokens_per_second'), 2)} tok/s`; `long_thinking = {_fmt_num(p_lng.get('mean_accepted_length'), 3)}` / `{_fmt_num(p_lng.get('mean_tokens_per_second'), 2)} tok/s`)",
                    "",
                    "| Variant | `std` Dense Acc (`tok/s`) | `std` Sparse Acc (`tok/s`) | `long_thinking` Dense Acc (`tok/s`) | `long_thinking` Sparse Acc (`tok/s`) |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for v in s1["variants"]:
                d_bc = (v["closed_loop"].get("dense") or {}).get("by_condition") or {}
                s_bc = (v["closed_loop"].get("dense_sparse") or {}).get("by_condition") or {}
                ds = d_bc.get("standard_nonthinking") or {}
                ss = s_bc.get("standard_nonthinking") or {}
                dl = d_bc.get("long_thinking") or {}
                sl = s_bc.get("long_thinking") or {}
                lines.append(
                    f"| `{v['variant']}` "
                    f"| {_fmt_num(ds.get('mean_accepted_length'), 3)} (`{_fmt_num(ds.get('mean_tokens_per_second'), 2)}`) "
                    f"| {_fmt_num(ss.get('mean_accepted_length'), 3)} (`{_fmt_num(ss.get('mean_tokens_per_second'), 2)}`) "
                    f"| {_fmt_num(dl.get('mean_accepted_length'), 3)} (`{_fmt_num(dl.get('mean_tokens_per_second'), 2)}`) "
                    f"| {_fmt_num(sl.get('mean_accepted_length'), 3)} (`{_fmt_num(sl.get('mean_tokens_per_second'), 2)}`) |"
                )
            lines.append("")

    # 2. Suite: `online_head_to_head`
    s2 = suites.get("online_head_to_head", {})
    lines.append("## 2. Online Zero-Init & Count-Gated Bigram Head-to-Head (`online_head_to_head`)")
    lines.append("")
    ws2 = s2.get("workspace_audit", {})
    if not ws2.get("eval_groups"):
        lines.append(f"- Status: `{s2.get('status', 'pending')}`")
        lines.append("")
    else:
        lines.append(
            f"- Status: **`{s2.get('status')}`** | `world_size={s2.get('timing_world_size')}` (`{s2.get('sharding_mode')}`) | `bigram_count_gate_kappa={s2.get('bigram_count_gate_kappa')}`"
        )
        lines.append("")
        t_map2 = {
            (
                g["condition"],
                float(g["temperature"]),
                json.dumps(g["sampling_warp_config"], sort_keys=True),
                g["arm"],
            ): g
            for g in ws2.get("timing_groups", [])
        }
        lines.extend(
            [
                "### Overall Held-Out Acceptance & Sharded Matched-Bundle Throughput",
                "",
                "| Condition | Arm | Primary / Total | Mean accepted len [95% CI] | Target fwds/tok [95% CI] | Sharded `tok/s` [95% CI] | Paired ratio vs `plain` [95% CI] |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for eg in ws2["eval_groups"]:
            k = (
                eg["condition"],
                float(eg["temperature"]),
                json.dumps(eg["sampling_warp_config"], sort_keys=True),
                eg["arm"],
            )
            tg = t_map2.get(k, {})
            cond_lbl = _cond_short(eg["condition"], eg["temperature"], eg["sampling_warp_config"])
            lines.append(
                f"| {cond_lbl} | `{eg['arm']}` | {eg['primary_requests']}/{eg['requests']} "
                f"| {_fmt_ci(eg['metrics']['mean_accepted_length'], 3)} "
                f"| {_fmt_ci(eg['metrics']['target_forwards_per_generated_token'], 4)} "
                f"| {_fmt_ci(tg.get('metrics', {}).get('tokens_per_second'), 2)} "
                f"| {_fmt_ci(tg.get('paired_throughput_ratio'), 4)} |"
            )
        lines.append("")
        _append_long_offset_slice_table(
            lines,
            title="### Long-Output Offset-Bin Trajectory (`0-1K`..`16K+`, Sparsity Gate Enforced)",
            eval_slices=ws2.get("eval_offset_slices", []),
            timing_slices=ws2.get("timing_offset_slices", []),
        )
        long_h2h = [
            c
            for c in ws2.get("head_to_head_offset_comparisons", [])
            if str(c.get("condition", "")).startswith("long_")
            and (c.get("arm"), c.get("reference_arm"))
            in (
                ("zero_offline_online_only", "plain_dflash"),
                ("online_dbS_zero_init", "plain_dflash"),
                ("online_bigram_count_gated", "plain_dflash"),
                ("online_bigram_count_gated", "online_bigram"),
                ("zero_offline_online_only", "online_dbS_zero_init"),
            )
        ]
        if long_h2h:
            lines.extend(
                [
                    "### Paired Long-Context Head-to-Head Horizon Crossover (`0-1K`..`16K+`, Sparsity Gate Enforced)",
                    "",
                    "| Pair (`arm` vs `ref`) | Offset Bin | Paired Reqs (`n`) | Paired `ΔE[τ]` [95% CI] | Paired Sharded `tok/s` Ratio [95% CI] | Gate Status |",
                    "| --- | --- | ---: | ---: | ---: | --- |",
                ]
            )
            for comp in long_h2h:
                pair_lbl = f"`{comp['arm']}` vs `{comp['reference_arm']}`"
                for b in comp.get("bins", []):
                    n_p = int(b.get("paired_requests_in_bin", 0))
                    iso_block = b.get("isolated_timing_tokens_per_second") or {}
                    p_ratio = iso_block.get("paired_ratio")
                    n_iso = int(p_ratio.get("n_prompts", n_p)) if isinstance(p_ratio, dict) else n_p
                    if n_p == 0:
                        g_stat = "**EMPTY (conclusion unavailable)**"
                    elif n_p < SPARSE_LONG_BIN_MIN_PROMPTS:
                        g_stat = f"**SPARSE (`n={n_p} < {SPARSE_LONG_BIN_MIN_PROMPTS}`; conclusion unavailable)**"
                    else:
                        g_stat = f"usable (`n_tps={n_iso}`)"
                    lines.append(
                        f"| {pair_lbl} | `{b['bin']}` | {n_p} "
                        f"| {_fmt_ci(b.get('accepted_length_delta'), 4)} "
                        f"| {_fmt_ci(p_ratio, 4)} | {g_stat} |"
                    )
            lines.append("")

    # 3. Suite: `m5_eval`
    s3 = suites.get("m5_eval", {})
    lines.append("## 3. Optional M5 Extensions (`m5_eval`)")
    lines.append("")
    ws3 = s3.get("workspace_audit", {})
    if not ws3.get("eval_groups"):
        lines.append(f"- Status: `{s3.get('status', 'pending')}`")
        lines.append("")
    else:
        off3 = s3.get("offline_fits", {})
        sh3 = off3.get("baseline_signed_hash", {})
        pl3 = off3.get("two_sided_pseudolikelihood", {})
        ex3 = off3.get("exact_observed_pairs", {})
        ex_kl = ex3.get("held_out_union_rest_grouped_kl", {})
        pl_kl = pl3.get("held_out_union_rest_grouped_kl", {})
        lines.extend(
            [
                f"- Status: **`{s3.get('status')}`** | `world_size={s3.get('timing_world_size')}` (`{s3.get('sharding_mode')}`) | Matched `C0[j]` `beta_relax = {ws3.get('matched_c0_beta_relax')}` | `ExactObservedPairs` `collision_count = {ex3.get('collision_count')}`",
                "",
                "### M5 Offline Objectives & Collision-Free Exact Sparse Table vs Signed Hash",
                "",
                "| Offline Checkpoint | Sparse Backend | Collisions | Dev Grouped KL | Test Grouped KL | `Δ` Test KL vs SignedHash |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
                f"| `baseline_signed_hash` | `signed_feature_hash` | {sh3.get('collision_count', '—')} | {_fmt_num(sh3.get('dev', {}).get('weighted_mean_kl'), 6)} | {_fmt_num(sh3.get('test', {}).get('weighted_mean_kl'), 6)} | 0.000000 |",
                f"| `two_sided_pseudolikelihood` | `{pl_kl.get('sparse_backend', 'signed_feature_hash')}` | {pl_kl.get('collision_count', '—')} | {_fmt_num(pl_kl.get('dev', {}).get('weighted_mean_kl'), 6)} | {_fmt_num(pl_kl.get('test', {}).get('weighted_mean_kl'), 6)} | — |",
                f"| `exact_observed_pairs` (`{ex3.get('overflow_policy')}`) | `exact_observed_pairs` | **{ex3.get('collision_count')}** | {_fmt_num(ex_kl.get('dev', {}).get('weighted_mean_kl'), 6)} | {_fmt_num(ex_kl.get('test', {}).get('weighted_mean_kl'), 6)} | {_fmt_num(ex3.get('test_grouped_kl_delta_vs_signed_hash'), 6)} |",
                "",
                "### M5 Closed-Loop Acceptance, Throughput & Lookahead Phase Duration",
                "",
                "| Condition | Arm | Primary / Total | Mean accepted len [95% CI] | Sharded `tok/s` [95% CI] | Paired ratio vs `plain` [95% CI] | `lookahead_iterations` med ms |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
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
        for eg in ws3["eval_groups"]:
            k = (
                eg["condition"],
                float(eg["temperature"]),
                json.dumps(eg["sampling_warp_config"], sort_keys=True),
                eg["arm"],
            )
            tg = t_map3.get(k, {})
            pg = p_map3.get(k, {})
            la_ms = (pg.get("phase_ms_median") or {}).get("lookahead_iterations", 0.0)
            cond_lbl = _cond_short(eg["condition"], eg["temperature"], eg["sampling_warp_config"])
            lines.append(
                f"| {cond_lbl} | `{eg['arm']}` | {eg['primary_requests']}/{eg['requests']} "
                f"| {_fmt_ci(eg['metrics']['mean_accepted_length'], 3)} "
                f"| {_fmt_ci(tg.get('metrics', {}).get('tokens_per_second'), 2)} "
                f"| {_fmt_ci(tg.get('paired_throughput_ratio'), 4)} "
                f"| {_fmt_num(la_ms, 2)} |"
            )
        lines.append("")
        if ws3.get("matched_support_comparisons"):
            lines.extend(
                [
                    "### M5 Matched-Support Paired Comparisons (`C0[j]`, Exact vs SignedHash, Buffered Stride-5)",
                    "",
                    "| Condition | Arm vs Reference | Support Family | Pairs (`rep_excl`) | `Δ` Accepted Len [95% CI] | Timing `tok/s` Ratio |",
                    "| --- | --- | --- | ---: | ---: | ---: |",
                ]
            )
            for mc in ws3["matched_support_comparisons"]:
                cond_lbl = _cond_short(
                    mc["condition"], mc["temperature"], mc["sampling_warp_config"]
                )
                tr = (mc.get("isolated_timing_tokens_per_second") or {}).get("ratio")
                lines.append(
                    f"| {cond_lbl} | `{mc['arm']}` vs `{mc['reference_arm']}` | `{mc['matched_support_family']}` "
                    f"| {mc['matched_pairs']} ({mc['repetition_excluded_pairs']}) "
                    f"| {_fmt_ci(mc.get('mean_accepted_length_delta'), 4)} "
                    f"| {_fmt_num(tr, 4)} |"
                )
            lines.append("")
        _append_long_offset_slice_table(
            lines,
            title="### M5 Long-Output Offset-Bin Trajectory (`0-1K`..`16K+`, Sparsity Gate Enforced)",
            eval_slices=ws3.get("eval_offset_slices", []),
            timing_slices=ws3.get("timing_offset_slices", []),
        )

    # 4. Suite: `fastpath_bench`
    s4 = suites.get("fastpath_bench", {})
    lines.append("## 4. Fastpath & Multi-GPU Scaling Benchmark (`fastpath_bench`)")
    lines.append("")
    if not s4.get("closed_loop_paired_speedups"):
        lines.append(f"- Status: `{s4.get('status', 'pending')}`")
        lines.append("")
    else:
        lines.extend(
            [
                f"- Status: **`{s4.get('status')}`** | Device: `{s4.get('device')}` | GPUs: `{s4.get('gpus')}`",
                "",
                "### Closed-Loop Fastpath Speedups (`eager_pre_round3` vs `round3_compact` vs `round3_compile`)",
                "",
                "| Arm | Matched Pairs (`rep_excl`) | Eager `tok/s` | Round-3 Compact `tok/s` (`speedup`) | Round-3 + Compile `tok/s` (`speedup vs eager`) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        ma_map = {
            (x["fastpath_mode"], x["arm"]): x
            for x in s4.get("closed_loop_summary_by_mode_and_arm", [])
        }
        for ps in s4["closed_loop_paired_speedups"]:
            arm = ps["arm"]
            eg_tps = ma_map.get(("eager_pre_round3", arm), {}).get("mean_tokens_per_second")
            r3_tps = ma_map.get(("round3_compact", arm), {}).get("mean_tokens_per_second")
            cm_tps = ma_map.get(("round3_compile", arm), {}).get("mean_tokens_per_second")
            lines.append(
                f"| `{arm}` | {ps['pairs']} ({ps['repetition_excluded_pairs']}) "
                f"| {_fmt_num(eg_tps, 2)} "
                f"| {_fmt_num(r3_tps, 2)} (**`{_fmt_num(ps.get('round3_over_eager_mean_ratio'), 3)}x`**) "
                f"| {_fmt_num(cm_tps, 2)} (**`{_fmt_num(ps.get('compile_over_eager_mean_ratio'), 3)}x`**) |"
            )
        lines.append("")

        cst = s4.get("cluster_sharded_throughput")
        if isinstance(cst, dict) and cst.get("by_arm"):
            lines.extend(
                [
                    f"### 1-GPU Isolated vs `{cst.get('world_size')}-GPU` Core-Pinned Sharded Calibration (`round3_compact`)",
                    "",
                    f"- Aggregate `{cst.get('world_size')}-GPU` cluster throughput across `{cst.get('total_generated_tokens')}` tokens: **`{_fmt_num(cst.get('aggregate_cluster_tokens_per_second'), 2)} tok/s`**",
                    "",
                    "| Arm | 1-GPU Cal `tok/s` | Sharded/GPU Cal `tok/s` | Per-GPU Scaling (`sharded / 1-GPU`) | Sum Across Ranks `tok/s` | Est Cluster Speedup vs 1-GPU |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for ba in cst["by_arm"]:
                lines.append(
                    f"| `{ba['arm']}` | {_fmt_num(ba.get('calibration_1gpu_tokens_per_second'), 2)} "
                    f"| {_fmt_num(ba.get('calibration_sharded_per_gpu_tokens_per_second'), 2)} "
                    f"| {_fmt_num(ba.get('sharded_over_1gpu_per_gpu_ratio'), 4)} "
                    f"| {_fmt_num(ba.get('sum_across_ranks_tokens_per_second'), 2)} "
                    f"| `{_fmt_num(ba.get('cluster_over_1gpu_throughput_ratio'), 2)}x` |"
                )
            lines.append("")

    return "\n".join(lines)


def _append_long_offset_slice_table(
    lines: list[str],
    *,
    title: str,
    eval_slices: list[dict[str, Any]],
    timing_slices: list[dict[str, Any]],
) -> None:
    long_eval = [s for s in eval_slices if str(s.get("condition", "")).startswith("long_")]
    if not long_eval:
        return
    t_bin_map: dict[tuple[str, float, str, str], dict[str, Any]] = {}
    for ts in timing_slices:
        if not str(ts.get("condition", "")).startswith("long_"):
            continue
        for tb in ts.get("bins", []):
            t_bin_map[(ts["condition"], float(ts["temperature"]), ts["arm"], tb["bin"])] = tb

    lines.extend(
        [
            title,
            "",
            "| Arm | Offset Bin | Reaching Reqs (`n`) | Accepted Len [95% CI] | Sharded Round `tok/s` [95% CI] | Gate Status |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for sl in long_eval:
        for b in sl.get("bins", []):
            n_r = int(b["requests_reaching_bin"])
            tb = t_bin_map.get((sl["condition"], float(sl["temperature"]), sl["arm"], b["bin"]), {})
            if n_r == 0:
                gate = "**EMPTY (conclusion unavailable)**"
            elif n_r < SPARSE_LONG_BIN_MIN_PROMPTS:
                gate = f"**SPARSE (`n={n_r} < {SPARSE_LONG_BIN_MIN_PROMPTS}`; conclusion unavailable)**"
            else:
                gate = "usable"
            lines.append(
                f"| `{sl['arm']}` | `{b['bin']}` | {n_r} "
                f"| {_fmt_ci(b.get('accepted_length'), 3)} "
                f"| {_fmt_ci(tb.get('tokens_per_second'), 2)} "
                f"| {gate} |"
            )
    lines.append("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["smoke", "reference", "full"], default="reference")
    parser.add_argument("--repo", type=Path, default=Path("/home/catid/drafter"))
    parser.add_argument("--artifacts", type=Path, default=Path("/home/catid/drafter/artifacts"))
    parser.add_argument("--suite_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("/tmp/edraft_reporting/post_m4_out"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--allow_pilot", action="store_true")
    parser.add_argument("--check_implementation_audit", action="store_true")
    args = parser.parse_args(argv)

    auditor = PostM4SuiteAuditor(
        repo=args.repo,
        artifacts=args.artifacts,
        suite_root=args.suite_root,
        scale=args.scale,
        seed=args.seed,
        allow_partial=args.allow_partial,
        require_complete=args.require_complete,
        allow_pilot=args.allow_pilot,
        check_implementation_audit=args.check_implementation_audit,
    )
    report = auditor.run_all()
    md = render_post_m4_markdown(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rep_path = args.output_dir / f"post_m4_audit_report_{args.scale}.json"
    md_path = args.output_dir / f"rendered_post_m4_tables_{args.scale}.md"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    summary_out = {
        "status": report["status"],
        "scale": report["scale"],
        "checks": report["checks"],
        "failures": report["failures"],
        "warnings_count": len(report["warnings"]),
        "audited_file_count": report["audited_file_count"],
        "report_json": str(rep_path),
        "tables_md": str(md_path),
    }
    print(json.dumps(summary_out))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
