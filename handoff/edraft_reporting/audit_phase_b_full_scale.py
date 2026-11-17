#!/usr/bin/env python3
"""Deterministic Zero-Drift Phase-B Full-Scale & Secondary-Ablation Validator and Table Renderer.

Audits all four `edraft.phase_b_suite` execution suites plus `phase_b_summary.json`
without modifying any repository, worktree, or artifact file:

  1. `bigram_eviction` (`--scale reference` secondary ablation, `prompt.md` §9 & §10 M3):
     - Verifies `online_bigram` (`k_ret=None`), `online_bigram_kret256` (`k_ret=256` default),
       `online_bigram_kret128` (`k_ret=128`), and `online_bigram_count_gated` (`kappa > 0`),
     - Independently recomputes `overall_evaluation_groups`, `offset_bin_slices`,
       `matched_support_comparisons` (`online_bigram_topk_eviction` & `online_bigram_count_gate`),
       `head_to_head_online_offset_comparison`, and `bigram_memory_ablation` (`exact_ledger_pairs`,
       `retained_prior_pairs`, `evicted_prior_pairs`, payload byte estimates) from raw rows.

  2. `full_vocab_lookahead` (`--scale reference` secondary ablation, `prompt.md` §7 & §10 M5):
     - Verifies matched-support closed-loop comparison pairing `full_vocabulary` lookahead
       (`lookahead_static_full`, `lookahead_rerelax_full`) strictly against causal `dense_full`,
       alongside `candidate_masked` lookahead (`lookahead_static_candidate`,
       `lookahead_rerelax_candidate`) strictly against causal `dense_candidate`,
     - Verifies identical frozen `beta_relax >= 0` across all active candidate/lookahead arms,
       strictly positive `T_relax > 0`, `lookahead.domain` (`full_vocabulary` vs `candidate_masked`),
       and positive `lookahead_iterations` ms on profiled lookahead rows.

  3. `pure_t1_fit` (`--scale reference` secondary ablation, `prompt.md` §10 M2 `--allow_pure_t1_ablation`):
     - Verifies `pure_T1_calibration_ablation_m5` training distribution + explicit `ablation_warning`,
       `beta_calibration` invariants (`optimal_beta >= 0`, `calibrated_kl <= min(beta1, beta0)`),
       cross-distribution held-out offline KL on BOTH `primary_heldout_*_kl` (with `live_full`
       logical slots > 0) and `pure_t1_heldout_*_kl`, and raw closed-loop shard recomputation
       across all 3 `standard_nonthinking` regimes (`greedy T=0`, `recommended T=0.7`, `pure T=1.0`)
       plus `long_*` repetition-excluded regimes.

  4. `full_scale` (`--scale full --allow_large_run`, `prompt.md` §0 & §10 M1-pre -> M4):
     - Verifies `prompts.json` capacity (`>=64` cal, `>=6800` std train, `>=64` std dev,
       `>=128` std test + complete held-out dataset, `>=32` long dev, `>=96` long test, disjoint splits),
     - Verifies `calibration.json` + `h_proj.pt` + `token_features.pt` SHA256s & train-only provenance,
     - Independently recomputes `m1_audit.json` token & prompt floors/ceilings:
       `2,000,000 <= primary_train_total <= 4,000,000` (`>=1,000,000` each at `T=0.0` & `T=0.7`
       warped, strictly excluding `T=1.0` diagnostic and `long_*` traces), separate `T=1.0` train
       diagnostic, `>=32` complete long dev + `>=96` complete long test generations, `>=128` std
       test across `(0.0, 0.7, 1.0)`, `live_full` dev/test head reconstruction,
     - Verifies `diagnostic_<scale>.json` (M1b) `audit_sha256` binding, unpooled regimes,
       `0-1K..16K+` offset bins, `h[i]` covariate-shift statistics, and `<8`-prompt sparse flags,
     - Verifies `m2.json` (`primary` train distribution only, `live_full` vs `union_rest_grouped`
       gap, norms, sparsity, collision stats, checkpoints),
     - Independently recomputes full-scale `m3` (4 arms) and `m4` (all 9 full-scale arms:
       `plain_dflash`, `online_bigram`, `offline_static`, `offline_online_stride1`, `online_dbS`,
       `offline_online_stride5`, `zero_offline_online_only`, `online_rejection_relative`,
       `online_cutoff`) across `>=128` std test prompts (`1` greedy + `3` seeds at `T=0.7` + `3`
       seeds at `T=1.0`) and `>=64` long test prompts (`3` seeds at `T_recommended`), verifying
       prompt-ID bootstrap 95% CIs (aggregating 3 seeds per prompt first!), sharded dev tune
       selection, M4 baseline acceptance reuse + fresh back-to-back isolated timing, 1-GPU
       `timing_isolation_calibration`, `greedy_output_comparisons`, `m4/feature_context.json`,
       and arm-specific online invariants (`dW==dA==0` on `online_dbS`, zero offline base norms on
       `zero_offline_online_only`, single-round unbuffered `stride=5` on `offline_online_stride5`,
       `rejection_relative` & `cutoff` weighting modes).

  5. **100% Byte-Preservation of Frozen Reference Artifacts**:
     - Verifies `phase_b_root != reference_artifacts` on real scales and verifies SHA256 immutability
       of all frozen M0-M4 (+ Phase-A `post_m4_suite` when present) files under `--reference_artifacts`.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any
from unittest.mock import patch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import audit_post_m4_suite  # noqa: E402
from audit_post_m4_suite import (  # noqa: E402
    LOOKAHEAD_ARMS,
    M5_ONLINE_HEAD_TO_HEAD_PAIRS,
    UPSTREAM_M0_M4_IMMUTABLE_RELPATHS,
    _is_rep_excluded,
    _row_rep_flag,
    recompute_head_to_head_online_offset_comparison,
    recompute_offset_bin_slices,
)
from build_and_audit_reports import (  # noqa: E402
    ALGORITHM_FILES,
    AUX_INSTRUMENTED_PHASES,
    M3_ARMS,
    ONLINE_ARMS,
    OUTPUT_LABELS,
    SECTION11_REQUIRED_PHASES,
    SPARSE_LONG_BIN_MIN_PROMPTS,
    PostGateEAuditor,
    _canonicalize_group_list,
    _cond_short,
    _fmt_ci,
    atomic_write_json,
    atomic_write_text,
    close_nested,
    compute_files_sha256,
    fingerprint,
    independent_greedy_mismatch,
    independent_prompt_bootstrap,
    independent_repetition_dominated,
    sha256_bytes,
    sha256_file,
)

M4_ARMS_REF: tuple[str, ...] = (
    "plain_dflash",
    "online_bigram",
    "offline_static",
    "offline_online_stride1",
    "online_dbS",
)
M4_ARMS_FULL: tuple[str, ...] = (
    *M4_ARMS_REF,
    "offline_online_stride5",
    "zero_offline_online_only",
    "online_rejection_relative",
    "online_cutoff",
)

DEFAULT_BIGRAM_EVICTION_ARMS: tuple[str, ...] = (
    "plain_dflash",
    "online_bigram",
    "online_bigram_kret256",
    "online_bigram_kret128",
    "online_bigram_count_gated",
)

DEFAULT_FULL_VOCAB_LOOKAHEAD_ARMS: tuple[str, ...] = (
    "plain_dflash",
    "dense_full",
    "lookahead_static_full",
    "lookahead_rerelax_full",
    "dense_candidate",
    "lookahead_static_candidate",
    "lookahead_rerelax_candidate",
)

REF_SECONDARY_SUITES: tuple[str, ...] = (
    "bigram_eviction",
    "full_vocab_lookahead",
    "pure_t1_fit",
)
ALL_PHASE_B_SUITES: tuple[str, ...] = (*REF_SECONDARY_SUITES, "full_scale")

PHASE_B_M5_MATCHED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("lookahead_static_candidate", "dense_candidate", "frozen_C0_candidate_masked"),
    ("lookahead_rerelax_candidate", "dense_candidate", "frozen_C0_candidate_masked"),
    ("lookahead_static_full", "dense_full", "full_vocabulary"),
    ("lookahead_rerelax_full", "dense_full", "full_vocabulary"),
    ("exact_observed_pairs_online", "signed_feature_hash_online", "online_sparse_representation"),
    ("online_buffered_stride5", "offline_online_stride5", "online_stride5_update_mode"),
    ("online_bigram_kret256", "online_bigram", "online_bigram_topk_eviction"),
    ("online_bigram_kret128", "online_bigram", "online_bigram_topk_eviction"),
    ("online_bigram_count_gated", "online_bigram", "online_bigram_count_gate"),
    ("online_dbS_zero_init", "online_bigram_count_gated", "online_zero_init_dbS_vs_count_gated_bigram"),
    ("zero_offline_online_only", "offline_online_stride1", "online_zero_vs_checkpoint_offline_init"),
    ("online_dbS_zero_init", "online_dbS", "online_dbS_zero_vs_checkpoint_offline_init"),
)

MATCHED_BETA_RELAX_ARMS: tuple[str, ...] = (
    "dense_candidate",
    "lookahead_static_candidate",
    "lookahead_rerelax_candidate",
    "lookahead_static_full",
    "lookahead_rerelax_full",
)

PHASE_B_ONLINE_HEAD_TO_HEAD_PAIRS: tuple[tuple[str, str], ...] = (
    *M5_ONLINE_HEAD_TO_HEAD_PAIRS,
    ("online_bigram_kret256", "plain_dflash"),
    ("online_bigram_kret128", "plain_dflash"),
    ("online_bigram_kret256", "online_bigram"),
    ("online_bigram_kret128", "online_bigram"),
    ("online_bigram_kret256", "online_bigram_kret128"),
)


def recompute_phase_b_head_to_head_offset_comparison(
    eval_rows: list[dict[str, Any]],
    isolated_timing_rows: list[dict[str, Any]] | None = None,
    *,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    with patch.object(
        audit_post_m4_suite,
        "M5_ONLINE_HEAD_TO_HEAD_PAIRS",
        PHASE_B_ONLINE_HEAD_TO_HEAD_PAIRS,
    ):
        return recompute_head_to_head_online_offset_comparison(
            eval_rows, isolated_timing_rows, seed=seed
        )


def recompute_phase_b_matched_support_comparisons(
    eval_rows: list[dict[str, Any]],
    timed_groups: list[dict[str, Any]] | None = None,
    *,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Independently recompute all 12 M5/Phase-B matched-support comparisons from raw rows."""
    by_regime_arm: dict[tuple[str, float, str, str], list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    by_slot: dict[tuple[Any, ...], dict[str, Any]] = {}
    regimes: set[tuple[str, float, str]] = set()
    for row in eval_rows:
        warp = json.dumps(row["sampling_warp_config"], sort_keys=True)
        cond = row["condition"]
        temp = float(row["temperature"])
        pid = row["prompt_id"]
        r_sd = int(row["seed"])
        arm_k = row["arm"]
        regimes.add((cond, temp, warp))
        by_regime_arm[(cond, temp, warp, arm_k)].append((pid, r_sd, row))
        by_slot[(cond, temp, warp, pid, r_sd, arm_k)] = row

    timed_map: dict[tuple[str, float, str, str], dict[str, Any]] = {}
    if timed_groups:
        for g in timed_groups:
            w = json.dumps(g["sampling_warp_config"], sort_keys=True)
            timed_map[(g["condition"], float(g["temperature"]), w, g["arm"])] = g

    comparisons: list[dict[str, Any]] = []
    for arm, ref_arm, support_family in PHASE_B_M5_MATCHED_PAIRS:
        for condition, temperature, warp in sorted(regimes):
            arm_entries = by_regime_arm.get((condition, temperature, warp, arm))
            if not arm_entries:
                continue
            acc_deltas = []
            fwd_deltas = []
            excluded = 0
            for pid, r_seed, row in arm_entries:
                ref_row = by_slot.get((condition, temperature, warp, pid, r_seed, ref_arm))
                if ref_row is None:
                    continue
                if _is_rep_excluded(row) or _is_rep_excluded(ref_row):
                    excluded += 1
                    continue
                acc_deltas.append(
                    {
                        "prompt_id": pid,
                        "seed": r_seed,
                        "delta": float(row["mean_accepted_length"])
                        - float(ref_row["mean_accepted_length"]),
                    }
                )
                fwd_deltas.append(
                    {
                        "prompt_id": pid,
                        "seed": r_seed,
                        "delta": float(row["target_forwards_per_generated_token"])
                        - float(ref_row["target_forwards_per_generated_token"]),
                    }
                )
            if not acc_deltas and excluded == 0:
                continue
            tg_arm = timed_map.get((condition, temperature, warp, arm))
            tg_ref = timed_map.get((condition, temperature, warp, ref_arm))
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
                    "mean_accepted_length_delta": (
                        independent_prompt_bootstrap(acc_deltas, "delta", seed=seed)
                        if acc_deltas
                        else None
                    ),
                    "target_forwards_per_generated_token_delta": (
                        independent_prompt_bootstrap(fwd_deltas, "delta", seed=seed)
                        if acc_deltas
                        else None
                    ),
                    "isolated_timing_tokens_per_second": (
                        {
                            "arm": tps_arm,
                            "reference_arm": tps_ref,
                            "ratio": (tps_arm / tps_ref) if (tps_arm and tps_ref) else None,
                        }
                        if (tg_arm is not None and tg_ref is not None)
                        else None
                    ),
                }
            )
    return comparisons


def recompute_bigram_memory_ablation(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Independently recompute `bigram_memory_ablation` from raw closed-loop evaluate rows."""
    grouped: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        bg = row.get("bigram")
        if not isinstance(bg, dict):
            continue
        warp = json.dumps(row["sampling_warp_config"], sort_keys=True)
        grouped[(row["condition"], float(row["temperature"]), warp, row["arm"])].append(row)

    out: list[dict[str, Any]] = []
    for (condition, temperature, warp, arm), group in sorted(grouped.items()):
        primary = [r for r in group if not _is_rep_excluded(r)]
        if not primary:
            continue
        exact_list = [int(r["bigram"]["exact_ledger_pairs"]) for r in primary]
        ret_list = [int(r["bigram"]["retained_prior_pairs"]) for r in primary]
        evict_list = [ex - rt for ex, rt in zip(exact_list, ret_list)]
        trans_list = [int(r["bigram"]["counted_transitions"]) for r in primary]
        frac_list = [
            (float(rt) / float(ex)) if ex > 0 else 1.0
            for rt, ex in zip(ret_list, exact_list)
        ]
        sample_bg = primary[0]["bigram"]
        out.append(
            {
                "condition": condition,
                "temperature": temperature,
                "sampling_warp_config": json.loads(warp),
                "arm": arm,
                "k_ret": sample_bg.get("k_ret"),
                "count_gate_kappa": sample_bg.get("count_gate_kappa"),
                "ledger_retained_for_exact_counts": bool(
                    sample_bg.get("ledger_retained_for_exact_counts", True)
                ),
                "primary_requests": len(primary),
                "repetition_excluded": len(group) - len(primary),
                "counted_transitions_mean": statistics.mean(trans_list),
                "exact_ledger_pairs_mean": statistics.mean(exact_list),
                "exact_ledger_pairs_max": max(exact_list),
                "retained_prior_pairs_mean": statistics.mean(ret_list),
                "retained_prior_pairs_max": max(ret_list),
                "evicted_prior_pairs_mean": statistics.mean(evict_list),
                "evicted_prior_pairs_max": max(evict_list),
                "retained_pair_fraction_mean": statistics.mean(frac_list),
                "retained_prior_payload_bytes_mean": statistics.mean(r * 12 for r in ret_list),
                "exact_ledger_payload_bytes_mean": statistics.mean(x * 16 for x in exact_list),
                "memory_disclosure": (
                    "Full exact ledger is retained in diagnostic mode so reappearing tokens "
                    "receive exact counts; retained_prior_pairs measures the active top-k_ret prior footprint."
                ),
            }
        )
    return out


class PhaseBFullScaleAuditor:
    """Read-only deterministic validator for Phase-B `--scale full` and secondary reference ablations."""

    def __init__(
        self,
        *,
        repo: Path,
        phase_b_root: Path,
        reference_artifacts: Path,
        scale: str = "full",
        suite: str = "all",
        seed: int = 2026,
        allow_partial: bool = False,
        require_complete: bool = False,
        allow_pilot: bool = False,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.phase_b_root = Path(phase_b_root).resolve()
        self.reference_artifacts = Path(reference_artifacts).resolve()
        self.scale = scale
        self.suite = suite
        self.seed = seed
        self.allow_partial = allow_partial
        self.require_complete = require_complete
        self.allow_pilot = allow_pilot

        if suite == "all":
            self.selected_suites: tuple[str, ...] = ALL_PHASE_B_SUITES
        elif suite == "ref_secondary_ablations":
            self.selected_suites = REF_SECONDARY_SUITES
        elif suite in ALL_PHASE_B_SUITES:
            self.selected_suites = (suite,)
        else:
            raise ValueError(f"Unsupported Phase-B suite selector: {suite!r}")

        self.checks: int = 0
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.read_sha256: dict[str, str] = {}

    def check(self, cond: bool, message: str) -> bool:
        self.checks += 1
        if not cond:
            self.failures.append(message)
        return cond

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def _is_live_inflight_file(self, path: Path) -> bool:
        name = path.name
        if name.startswith("manifest") or name.endswith("-processes.json") or name == "work_queue.json":
            return True
        if name.startswith("collection-rank") or name == "latest.pt":
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
    # 0. Upstream Reference Artifact Byte-Preservation Audit
    # ------------------------------------------------------------------

    def audit_upstream_reference_immutability(self) -> dict[str, Any]:
        """Verify 100% byte-preservation of frozen `/home/catid/drafter/artifacts/` reference files."""
        if (
            self.scale != "smoke"
            and self.reference_artifacts.is_dir()
            and (self.reference_artifacts / "m4" / "summary.json").is_file()
        ):
            self.check(
                self.phase_b_root != self.reference_artifacts,
                f"phase_b_root ({self.phase_b_root}) must NOT collide with frozen reference_artifacts ({self.reference_artifacts})",
            )

        checked_files: dict[str, str] = {}
        if not self.reference_artifacts.is_dir():
            if self.scale != "smoke":
                self.check(False, f"missing reference_artifacts directory: {self.reference_artifacts}")
            return {"checked_upstream_files": 0, "sha256s": {}}

        ms_path = self.reference_artifacts / "milestones.json"
        milestones = self.read_json_tracked(ms_path) if ms_path.is_file() else None
        if isinstance(milestones, dict):
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
                (
                    "M5",
                    {
                        "m2_refit_calibration": "post_m4_suite/m2_refit_calibration.json",
                        "online_head_to_head": "post_m4_suite/online_head_to_head.json",
                        "m5_eval": "post_m4_suite/m5_eval.json",
                        "fastpath_bench": "post_m4_suite/fastpath_bench.json",
                        "post_m4_suite_summary": "post_m4_suite/post_m4_suite_summary.json",
                    },
                ),
            ):
                rec_shas = milestones.get(stg_key, {}).get("artifact_sha256", {})
                if not isinstance(rec_shas, dict):
                    continue
                for sha_key, rel_p in rel_map.items():
                    exp = rec_shas.get(sha_key)
                    fp = self.reference_artifacts / rel_p
                    if exp and fp.is_file():
                        act = sha256_file(fp)
                        self.read_sha256[str(fp.resolve())] = act
                        checked_files[rel_p] = act
                        self.check(
                            act == exp,
                            f"upstream reference {rel_p} SHA256 {act} != milestones.json {stg_key}.{sha_key} {exp}",
                        )

        cont_path = self.reference_artifacts / "reference_continuation_state.json"
        if cont_path.is_file():
            cont = self.read_json_tracked(cont_path)
            for bound_path_str, exp_sha in sorted(
                cont.get("frozen_artifact_binding", {}).items()
            ):
                if "artifacts/" in bound_path_str and "m4_evidence_validator" not in bound_path_str:
                    sub = bound_path_str.split("artifacts/", 1)[1]
                    cand = self.reference_artifacts / sub
                    if cand.is_file():
                        act = sha256_file(cand)
                        self.read_sha256[str(cand.resolve())] = act
                        checked_files[sub] = act
                        self.check(
                            act == exp_sha,
                            f"upstream frozen M0-M2 artifact {sub} mutated: {act} != {exp_sha}",
                        )

        for rel in (
            *UPSTREAM_M0_M4_IMMUTABLE_RELPATHS,
            "milestones.json",
            "m3_validation_reference.json",
            "m4_validation_reference.json",
            "reference_continuation_state.json",
            "post_m4_suite/m2_refit_calibration.json",
            "post_m4_suite/online_head_to_head.json",
            "post_m4_suite/m5_eval.json",
            "post_m4_suite/fastpath_bench.json",
            "post_m4_suite/post_m4_suite_summary.json",
        ):
            fp = self.reference_artifacts / rel
            if fp.is_file() and rel not in checked_files:
                act = sha256_file(fp)
                self.read_sha256[str(fp.resolve())] = act
                checked_files[rel] = act

        return {
            "checked_upstream_files": len(checked_files),
            "reference_artifacts": str(self.reference_artifacts),
            "sha256s": checked_files,
        }

    # ------------------------------------------------------------------
    # Shared Closed-Loop M5 Workspace Auditor (Suites 1 & 2)
    # ------------------------------------------------------------------

    def _audit_m5_ablation_workspace(
        self,
        *,
        label: str,
        eff_scale: str,
        suite_report: dict[str, Any],
        artifacts_block: dict[str, Any],
        expected_arms: list[str],
        check_head_to_head_offsets: bool,
        check_bigram_memory: bool,
    ) -> dict[str, Any]:
        settings_p = Path(artifacts_block.get("settings_json") or "")
        summary_p = Path(artifacts_block.get("summary_json") or "")
        timing_p = Path(artifacts_block.get("timing_json") or "")
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
            settings.get("frozen") is True
            and settings.get("stage") == "m5"
            and settings.get("scale") == eff_scale,
            f"{label}: m5/settings.json frozen/stage/scale invalid ({settings.get('scale')!r} vs {eff_scale!r})",
        )
        self.check(
            set(settings.get("selected", {}).keys()) == set(expected_arms),
            f"{label}: selected arms {sorted(settings.get('selected', {}).keys())} != {sorted(expected_arms)}",
        )

        # Matched beta_relax across all active candidate + lookahead arms (`prompt.md:354-359, 394`)
        present_br_arms = [a for a in MATCHED_BETA_RELAX_ARMS if a in expected_arms]
        expected_beta_relax: float | None = None
        if present_br_arms:
            br_vals: list[float] = []
            for a in present_br_arms:
                sel_cfg = settings["selected"][a]
                self.check(
                    "beta_relax" in sel_cfg,
                    f"{label}: arm {a} missing explicit beta_relax in settings.json",
                )
                br = sel_cfg.get("beta_relax", 1.0)
                self.check(
                    not isinstance(br, bool)
                    and isinstance(br, (int, float))
                    and math.isfinite(float(br))
                    and float(br) >= 0.0,
                    f"{label}: invalid beta_relax={br!r} on {a}",
                )
                br_vals.append(float(br))
            self.check(
                len(set(br_vals)) == 1,
                f"{label}: matched C0[j] beta_relax differs across {present_br_arms}: {br_vals}",
            )
            expected_beta_relax = br_vals[0]

        for la_arm in (a for a in LOOKAHEAD_ARMS if a in expected_arms):
            la_cfg = settings["selected"][la_arm]
            t_rel = la_cfg.get("T_relax")
            self.check(
                not isinstance(t_rel, bool)
                and isinstance(t_rel, (int, float))
                and math.isfinite(float(t_rel))
                and float(t_rel) > 0.0,
                f"{label}: {la_arm} requires strictly positive T_relax > 0, got {t_rel!r}",
            )

        # Audit dev tune selection (single-rank or sharded)
        self._audit_dev_tune_selection(
            label=f"{label}/tune",
            stage_dir=stage_dir,
            settings=settings,
            arms=tuple(expected_arms),
            eff_scale=eff_scale,
        )

        # Load & verify all raw evaluate rows
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
            self._audit_single_m5_ablation_row(
                label=f"{label}/evaluate",
                row=row,
                expected_beta_relax=expected_beta_relax,
                is_profile=False,
            )
            eval_rows.append(row)

        expected_per_arm = 8 if eff_scale == "smoke" else (480 if eff_scale == "reference" else 1088)
        self.check(
            len(eval_rows) == int(summary.get("completed_requests", -1)),
            f"{label}: loaded {len(eval_rows)} eval rows != summary.completed_requests {summary.get('completed_requests')}",
        )
        if self.require_complete and not self.allow_pilot:
            self.check(
                summary.get("isolated_timing_complete") is True,
                f"{label}: summary.isolated_timing_complete={summary.get('isolated_timing_complete')!r} != True",
            )
            self.check(
                summary.get("missing_requests") == []
                and len(eval_rows) == len(expected_arms) * expected_per_arm,
                f"{label}: expected {len(expected_arms) * expected_per_arm} complete eval rows, got {len(eval_rows)} (missing={len(summary.get('missing_requests', []))})",
            )

        # Load & verify all raw timing & phase_profile rows
        timed_rows, profile_rows = self._load_and_audit_timing_bundles(
            label=f"{label}/timing",
            stage_dir=stage_dir,
            timing=timing,
            expected_arms=tuple(expected_arms),
            row_auditor=lambda rw, is_prof: self._audit_single_m5_ablation_row(
                label=f"{label}/timing",
                row=rw,
                expected_beta_relax=expected_beta_relax,
                is_profile=is_prof,
            ),
        )

        helper = PostGateEAuditor(
            repo=self.repo, artifacts=stage_dir.parent, scale=eff_scale, seed=self.seed
        )
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

        ok_seg, diff_seg = close_nested(
            _canonicalize_group_list(recomp_eval_groups),
            _canonicalize_group_list(suite_report.get("overall_evaluation_groups", [])),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_seg, f"{label}: suite overall_evaluation_groups drift: {diff_seg}")

        recomp_timed_groups = self._enrich_timed_groups_with_paired_plain(
            label=f"{label}/timing",
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

        ok_stg, diff_stg = close_nested(
            _canonicalize_group_list(recomp_timed_groups),
            _canonicalize_group_list(suite_report.get("isolated_timing_groups", [])),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_stg, f"{label}: suite isolated_timing_groups drift: {diff_stg}")

        recomp_eval_slices = recompute_offset_bin_slices(
            eval_rows, isolated_timing=False, seed=self.seed
        )
        ok_es, diff_es = close_nested(
            recomp_eval_slices,
            suite_report.get("evaluation_offset_bin_slices", []),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_es, f"{label}: evaluation_offset_bin_slices drift: {diff_es}")

        recomp_timed_slices = recompute_offset_bin_slices(
            timed_rows, isolated_timing=True, seed=self.seed
        )
        ok_ts, diff_ts = close_nested(
            recomp_timed_slices,
            suite_report.get("isolated_timing_offset_bin_slices", []),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_ts, f"{label}: isolated_timing_offset_bin_slices drift: {diff_ts}")

        pilot_incomplete_timing = (
            bool(self.allow_pilot or (self.allow_partial and not self.require_complete))
            and not bool(summary.get("isolated_timing_complete"))
        )
        recomp_msc = recompute_phase_b_matched_support_comparisons(
            eval_rows,
            None if pilot_incomplete_timing else recomp_timed_groups,
            seed=self.seed,
        )
        ok_msc, diff_msc = close_nested(
            recomp_msc,
            suite_report.get("matched_support_comparisons", []),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_msc, f"{label}: matched_support_comparisons drift: {diff_msc}")

        if check_head_to_head_offsets:
            recomp_h2h = recompute_phase_b_head_to_head_offset_comparison(
                eval_rows,
                None if pilot_incomplete_timing else timed_rows,
                seed=self.seed,
            )
            ok_h2h, diff_h2h = close_nested(
                recomp_h2h,
                suite_report.get("head_to_head_online_offset_comparison", []),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            self.check(ok_h2h, f"{label}: head_to_head_online_offset_comparison drift: {diff_h2h}")

        recomp_bg_mem: list[dict[str, Any]] = []
        if check_bigram_memory:
            recomp_bg_mem = recompute_bigram_memory_ablation(eval_rows)
            self.check(len(recomp_bg_mem) > 0, f"{label}: bigram_memory_ablation empty")
            ok_bm, diff_bm = close_nested(
                recomp_bg_mem,
                suite_report.get("bigram_memory_ablation", []),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            self.check(ok_bm, f"{label}: bigram_memory_ablation drift from raw rows: {diff_bm}")
            ok_sbm, diff_sbm = close_nested(
                recomp_bg_mem,
                summary.get("bigram_memory_ablation", []),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
            self.check(ok_sbm, f"{label}: summary.bigram_memory_ablation drift: {diff_sbm}")

        # Verify 1-GPU timing isolation calibration when sharded on real models
        cal_summary = self._audit_optional_timing_calibration(
            label=label,
            stage_dir=stage_dir,
            timing=timing,
            eff_scale=eff_scale,
            expected_arms=tuple(expected_arms),
            timed_rows=timed_rows,
        )

        sparse_bins = self._collect_sparse_long_bins(recomp_eval_slices)
        return {
            "frozen_settings_hash": frozen_hash,
            "selected_settings": settings.get("selected", {}),
            "matched_beta_relax": expected_beta_relax,
            "evaluated_rows": len(eval_rows),
            "timed_rows": len(timed_rows),
            "profile_rows": len(profile_rows),
            "eval_groups": recomp_eval_groups,
            "timing_groups": recomp_timed_groups,
            "phase_profiles": timing.get("phase_profiles", []),
            "matched_support_comparisons": recomp_msc,
            "bigram_memory_ablation": recomp_bg_mem,
            "timing_isolation_calibration": cal_summary,
            "sparse_long_offset_bins": sparse_bins,
        }

    def _audit_single_m5_ablation_row(
        self,
        *,
        label: str,
        row: dict[str, Any],
        expected_beta_relax: float | None,
        is_profile: bool,
    ) -> None:
        arm = str(row["arm"])
        pid = str(row["prompt_id"])
        if str(row.get("condition", "")).startswith("long_"):
            rep_recomp = independent_repetition_dominated(row.get("output_tokens", []))
            self.check(
                rep_recomp == _row_rep_flag(row),
                f"{label}: repetition_dominated mismatch on {pid} ({arm})",
            )

        bg = row.get("bigram")
        if arm in (
            "online_bigram",
            "online_bigram_kret256",
            "online_bigram_kret128",
            "online_bigram_count_gated",
        ):
            self.check(isinstance(bg, dict), f"{label}: missing bigram telemetry on {pid} ({arm})")
            if isinstance(bg, dict):
                ex_pairs = int(bg.get("exact_ledger_pairs", -1))
                ret_pairs = int(bg.get("retained_prior_pairs", -1))
                self.check(
                    0 <= ret_pairs <= ex_pairs,
                    f"{label}: invalid bigram pair counts retained={ret_pairs} > exact={ex_pairs} on {pid} ({arm})",
                )
                if arm == "online_bigram":
                    self.check(
                        bg.get("k_ret") is None and ret_pairs == ex_pairs,
                        f"{label}: online_bigram must retain exact un-evicted counts on {pid}",
                    )
                elif arm in ("online_bigram_kret256", "online_bigram_kret128"):
                    kr = bg.get("k_ret")
                    self.check(
                        isinstance(kr, int) and kr >= 1,
                        f"{label}: {arm} missing positive integer k_ret on {pid}: {kr!r}",
                    )

        tel = row.get("m5_telemetry") or {}
        if arm == "dense_candidate" and expected_beta_relax is not None:
            rec_br = tel.get("matched_c0_beta_relax")
            self.check(
                rec_br is not None and math.isclose(float(rec_br), expected_beta_relax, abs_tol=1e-12),
                f"{label}: dense_candidate matched_c0_beta_relax={rec_br!r} != {expected_beta_relax}",
            )
        if arm in LOOKAHEAD_ARMS:
            la_tel = tel.get("lookahead") or {}
            expected_mode = "static" if "static" in arm else "rerelax"
            expected_domain = "candidate_masked" if arm.endswith("_candidate") else "full_vocabulary"
            self.check(
                la_tel.get("mode") == expected_mode and la_tel.get("domain") == expected_domain,
                f"{label}: {arm} lookahead telemetry mode/domain mismatch: {la_tel}",
            )
            t_rel = la_tel.get("T_relax")
            self.check(
                isinstance(t_rel, (int, float)) and float(t_rel) > 0.0,
                f"{label}: {arm} lookahead T_relax={t_rel!r} must be > 0",
            )
            if expected_beta_relax is not None:
                rec_br = la_tel.get("beta_relax")
                self.check(
                    rec_br is not None
                    and math.isclose(float(rec_br), expected_beta_relax, abs_tol=1e-12),
                    f"{label}: {arm} beta_relax={rec_br!r} != {expected_beta_relax}",
                )
            if is_profile and len(row.get("blocks") or []) > 0 and int(la_tel.get("R_init", 0)) > 0:
                la_ms = float((row.get("phase_ms") or {}).get("lookahead_iterations", 0.0))
                self.check(
                    la_ms > 0.0,
                    f"{label}: profiled {arm} row {pid} has non-positive lookahead_iterations={la_ms}",
                )
        elif is_profile:
            la_ms = float((row.get("phase_ms") or {}).get("lookahead_iterations", 0.0))
            self.check(
                la_ms == 0.0,
                f"{label}: causal non-lookahead arm {arm} reported nonzero lookahead_iterations={la_ms}",
            )

    def _collect_sparse_long_bins(self, slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flags: list[dict[str, Any]] = []
        for sl in slices:
            if not str(sl["condition"]).startswith("long_"):
                continue
            for b in sl["bins"]:
                n_reach = int(b["requests_reaching_bin"])
                if n_reach < SPARSE_LONG_BIN_MIN_PROMPTS:
                    flags.append(
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
        return flags

    def _audit_dev_tune_selection(
        self,
        *,
        label: str,
        stage_dir: Path,
        settings: dict[str, Any],
        arms: tuple[str, ...],
        eff_scale: str,
    ) -> dict[str, Any]:
        """Recompute dev tune selection across either single-GPU (`manifest-rank0.json`) or sharded (`manifest-rank*.json`)."""
        dev_pids = list(
            settings.get("development_prompt_ids")
            or settings.get("development_prompts")
            or []
        )
        if eff_scale == "full":
            # Full scale requires >= 8 standard dev + >= 32 long dev prompts (`prompt.md:736, 829, 859`)
            self.check(
                len(dev_pids) >= 40,
                f"{label}: full-scale dev tuning requires >=40 dev prompts (>=8 std + >=32 long), got {len(dev_pids)}",
            )
        elif eff_scale == "reference":
            self.check(
                len(dev_pids) >= 20,
                f"{label}: reference dev tuning requires >=20 dev prompts (>=8 std + >=12 long), got {len(dev_pids)}",
            )

        tune_dir = stage_dir / "tune"
        mpaths = sorted(tune_dir.glob("manifest-rank*.json")) if tune_dir.is_dir() else []
        if not mpaths:
            self.check(self.allow_partial, f"{label}: missing tune/manifest-rank*.json")
            return {"status": "missing"}

        plan_path = tune_dir / "sharding_plan.json"
        target_sessions: set[str] = set()
        if settings.get("comparison_session"):
            target_sessions.add(str(settings["comparison_session"]))
        if plan_path.is_file():
            plan = self.read_json_tracked(plan_path)
            w = int(plan.get("world_size", 1))
            log_sess = plan.get("logical_session")
            if log_sess:
                for r in range(w):
                    target_sessions.add(f"{log_sess}-rank{r}")
                target_sessions.add(str(log_sess))
            seen_gpus: set[int] = set()
            seen_cpus: set[int] = set()
            for r in range(w):
                sp = tune_dir / "sessions" / f"{log_sess}-rank{r}.json"
                if sp.is_file():
                    sdata = self.read_json_tracked(sp)
                    pgpu = sdata.get("physical_gpu")
                    cpus = list(sdata.get("cpu_affinity") or [])
                    if pgpu is not None and eff_scale != "smoke":
                        self.check(pgpu not in seen_gpus, f"{label}: duplicate physical_gpu {pgpu} in sharded tune")
                        seen_gpus.add(int(pgpu))
                    if cpus and eff_scale != "smoke":
                        self.check(
                            not seen_cpus.intersection(cpus),
                            f"{label}: overlapping cpu_affinity across sharded tune ranks on rank {r}",
                        )
                        seen_cpus.update(int(c) for c in cpus)

        dev_obs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        seen_job_keys: set[str] = set()
        tune_entries_to_load: list[tuple[dict[str, Any], Path]] = []
        for mp in mpaths:
            tm = self.read_json_tracked(mp)
            for jkey, entry in tm.get("jobs", {}).items():
                if entry.get("status") != "complete" or jkey in seen_job_keys:
                    continue
                t_sess = str(entry.get("identity", {}).get("timing_session", ""))
                if target_sessions and t_sess not in target_sessions:
                    continue
                rp = Path(entry["result_path"])
                if not rp.is_file():
                    continue
                seen_job_keys.add(jkey)
                tune_entries_to_load.append((entry, rp))
        loaded_tune_rows = self.read_json_batch_tracked([rp for _, rp in tune_entries_to_load])
        for (entry, rp), row in zip(tune_entries_to_load, loaded_tune_rows):
            self.check(
                self.read_sha256[str(rp.resolve())] == entry.get("result_sha256"),
                f"{label}: SHA256 mismatch on {rp}",
            )
            self.check(
                row.get("split") == "dev",
                f"{label}: non-dev split={row.get('split')!r} leaked into dev tuning ({row.get('prompt_id')})",
            )
            dev_obs[(row["arm"], fingerprint(row["settings"]))].append(row)

        plain_dev = {r["prompt_id"]: r for r in dev_obs.get(("plain_dflash", fingerprint({})), [])}
        if not plain_dev:
            self.check(self.allow_partial, f"{label}: no plain_dflash dev observations found")
            return {"status": "incomplete"}

        for arm in arms:
            cand_scores: list[tuple[float, dict[str, Any]]] = []
            for cand_rep in settings.get("candidates", {}).get(arm, []):
                cand_setting = cand_rep["settings"]
                c_rows = dev_obs.get((arm, fingerprint(cand_setting)), [])
                by_cond: dict[str, list[float]] = defaultdict(list)
                excl = 0
                for r in c_rows:
                    base = plain_dev.get(r["prompt_id"])
                    if base is None:
                        continue
                    rep_arm = independent_repetition_dominated(r["output_tokens"])
                    rep_base = independent_repetition_dominated(base["output_tokens"])
                    self.check(
                        rep_arm == _row_rep_flag(r),
                        f"{label}: repetition flag mismatch on {r['prompt_id']} ({arm})",
                    )
                    if str(r["condition"]).startswith("long_") and (rep_arm or rep_base):
                        excl += 1
                        continue
                    by_cond[str(r["condition"])].append(
                        float(r["tokens_per_second"]) / float(base["tokens_per_second"])
                    )
                self.check(bool(by_cond), f"{label}/{arm}: empty non-excluded dev condition scores")
                if by_cond:
                    cond_means = {c: statistics.mean(vs) for c, vs in by_cond.items()}
                    score = statistics.mean(cond_means.values())
                    self.check(
                        math.isclose(score, float(cand_rep["selection_score"]), rel_tol=1e-11, abs_tol=1e-11),
                        f"{label}/{arm}: recomputed selection_score {score} != {cand_rep['selection_score']}",
                    )
                    self.check(
                        excl == int(cand_rep["repetition_excluded_matched_pairs"]),
                        f"{label}/{arm}: excluded pair count {excl} != {cand_rep['repetition_excluded_matched_pairs']}",
                    )
                    cand_scores.append((score, cand_setting))
            if cand_scores:
                winner = max(cand_scores, key=lambda x: x[0])[1]
                sel_arm_cfg = settings.get("selected", {}).get(arm)
                self.check(
                    winner == sel_arm_cfg,
                    f"{label}/{arm}: selected {sel_arm_cfg} != argmax winner {winner}",
                )
        return {"status": "verified", "dev_prompts": len(plain_dev)}

    def _load_and_audit_timing_bundles(
        self,
        *,
        label: str,
        stage_dir: Path,
        timing: dict[str, Any],
        expected_arms: tuple[str, ...],
        row_auditor: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        manifest_sha_by_path: dict[str, str] = {}
        for tmf in sorted((stage_dir / "timing").glob("manifest-rank*.json")):
            tmdata = self.read_json_tracked(tmf)
            for t_entry in tmdata.get("jobs", {}).values():
                if (
                    t_entry.get("status") == "complete"
                    and t_entry.get("result_path")
                    and t_entry.get("result_sha256")
                ):
                    manifest_sha_by_path[str(Path(t_entry["result_path"]).resolve())] = t_entry[
                        "result_sha256"
                    ]

        raw_ev = timing.get("raw_evidence", {})
        sess_paths = (
            [Path(p) for p in raw_ev["session_paths"]]
            if raw_ev.get("session_paths")
            else ([Path(raw_ev["session_path"])] if raw_ev.get("session_path") else [])
        )
        self.check(bool(sess_paths), f"{label}: no session_paths in timing.json")

        world_size = int(timing.get("world_size") or raw_ev.get("world_size") or max(1, len(sess_paths)))
        if world_size > 1:
            plan_p = Path(
                raw_ev.get("sharding_plan_path")
                or timing.get("sharding_plan_path")
                or (stage_dir / "timing" / "sharding_plan.json")
            )
            self.check(plan_p.is_file(), f"{label}: sharded timing missing sharding_plan.json at {plan_p}")
            if plan_p.is_file():
                pdata = self.read_json_tracked(plan_p)
                helper = PostGateEAuditor(repo=self.repo, artifacts=stage_dir.parent, scale="reference")
                rmap = helper._extract_rank_prompt_map(pdata)
                self.check(
                    set(rmap.keys()) == set(range(world_size)),
                    f"{label}: sharding_plan ranks {sorted(rmap.keys())} != 0..{world_size - 1}",
                )
                flat_p = [pid for pids in rmap.values() for pid in pids]
                self.check(
                    len(flat_p) == len(set(flat_p)),
                    f"{label}: overlapping prompt IDs across timing ranks in sharding_plan.json",
                )

        timed_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        seen_gpus: set[int] = set()
        seen_cpus: set[int] = set()
        sess_objs = [(sp, self.read_json_tracked(sp)) for sp in sess_paths]
        all_bundle_rpaths: list[Path] = []
        for _sp, sess in sess_objs:
            for _bkey, bval in sess.get("bundles", {}).items():
                if bval.get("status") != "complete":
                    continue
                all_bundle_rpaths.extend(Path(p_str) for p_str in bval.get("result_paths", []))
        loaded_bundle_rows = iter(self.read_json_batch_tracked(all_bundle_rpaths))
        for sp, sess in sess_objs:
            if world_size > 1 and str(timing.get("scale", "")) != "smoke":
                pgpu = sess.get("physical_gpu")
                cpus = list(sess.get("cpu_affinity") or [])
                if pgpu is not None:
                    self.check(pgpu not in seen_gpus, f"{label}: duplicate physical_gpu {pgpu} across timing sessions")
                    seen_gpus.add(int(pgpu))
                if cpus:
                    self.check(
                        not seen_cpus.intersection(cpus),
                        f"{label}: overlapping cpu_affinity across timing sessions ({sp.name})",
                    )
                    seen_cpus.update(int(c) for c in cpus)

            for bkey, bval in sess.get("bundles", {}).items():
                if bval.get("status") != "complete":
                    continue
                rpaths = bval.get("result_paths", [])
                rshas = bval.get("result_sha256", {})
                self.check(
                    len(rpaths) == len(expected_arms),
                    f"{label}: bundle {bkey} has {len(rpaths)} rows != {len(expected_arms)} arms",
                )
                prof_flags: set[bool] = set()
                seen_arms: set[str] = set()
                for p_str in rpaths:
                    rp = Path(p_str)
                    row = next(loaded_bundle_rows)
                    rp_res = str(rp.resolve())
                    exp_sha = rshas.get(p_str) or manifest_sha_by_path.get(rp_res)
                    if exp_sha:
                        self.check(
                            self.read_sha256[rp_res] == exp_sha,
                            f"{label}: SHA256 mismatch on {rp}",
                        )
                    seen_arms.add(str(row["arm"]))
                    is_prof = bool(row.get("phase_profile", False))
                    prof_flags.add(is_prof)
                    if is_prof:
                        pm = row.get("phase_ms") or {}
                        for ph_k, ph_v in pm.items():
                            self.check(
                                math.isfinite(float(ph_v)) and float(ph_v) >= 0.0,
                                f"{label}: profile row {rp.name} invalid phase_ms[{ph_k}]={ph_v!r}",
                            )
                        if str(timing.get("scale", "")) != "smoke":
                            for ph in SECTION11_REQUIRED_PHASES:
                                self.check(
                                    ph in pm and math.isfinite(float(pm[ph])) and float(pm[ph]) >= 0.0,
                                    f"{label}: profile row {rp.name} missing/invalid §11 phase {ph}",
                                )
                        profile_rows.append(row)
                    else:
                        timed_rows.append(row)
                    row_auditor(row, is_prof)
                self.check(
                    seen_arms == set(expected_arms),
                    f"{label}: bundle {bkey} arms {sorted(seen_arms)} != {sorted(expected_arms)}",
                )
                self.check(
                    len(prof_flags) == 1,
                    f"{label}: bundle {bkey} mixes phase_profile and timed speed rows",
                )

        self.check(
            len(timed_rows) == int(timing.get("timed_requests", -1)),
            f"{label}: loaded {len(timed_rows)} timed_rows != timing.timed_requests {timing.get('timed_requests')}",
        )
        return timed_rows, profile_rows

    def _enrich_timed_groups_with_paired_plain(
        self,
        *,
        label: str,
        timed_rows: list[dict[str, Any]],
        recomp_t_groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        paired_plain: dict[tuple[str, float, str, Any, Any, str], dict[str, Any]] = {}
        timed_by_group: dict[tuple[str, float, str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for r in timed_rows:
            rk = (
                r["condition"],
                float(r["temperature"]),
                json.dumps(r["sampling_warp_config"], sort_keys=True),
            )
            rep_norm = str(r["repeat"]).split(":epoch=")[0]
            timed_by_group[(*rk, r["arm"])].append((rep_norm, r))
            if r["arm"] == "plain_dflash":
                paired_plain[(*rk, r["prompt_id"], r["seed"], rep_norm)] = r

        for g in recomp_t_groups:
            gk = (
                g["condition"],
                float(g["temperature"]),
                json.dumps(g["sampling_warp_config"], sort_keys=True),
            )
            paired, m_arm, m_plain = [], [], []
            excl_pairs = 0
            for rep_norm, r in timed_by_group.get((*gk, g["arm"]), ()):
                b_key_lookup = (*gk, r["prompt_id"], r["seed"], rep_norm)
                self.check(
                    b_key_lookup in paired_plain,
                    f"{label}: missing matched plain_dflash row for {b_key_lookup}",
                )
                if b_key_lookup not in paired_plain:
                    continue
                b_row = paired_plain[b_key_lookup]
                if r.get("tokens_per_second") is None or b_row.get("tokens_per_second") is None:
                    self.check(
                        False,
                        f"{label}: timed row missing tokens_per_second on {r['prompt_id']} ({r['arm']})",
                    )
                    continue
                if _is_rep_excluded(r) or _is_rep_excluded(b_row):
                    excl_pairs += 1
                    continue
                m_arm.append(r)
                m_plain.append(b_row)
                paired.append(
                    {
                        "prompt_id": r["prompt_id"],
                        "seed": r["seed"],
                        "ratio": float(r["tokens_per_second"])
                        / float(b_row["tokens_per_second"]),
                    }
                )
            ratio_boot = independent_prompt_bootstrap(paired, "ratio", seed=self.seed)
            m_arm_mean = independent_prompt_bootstrap(
                m_arm, "tokens_per_second", seed=self.seed
            )["mean"]
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

    def _audit_optional_timing_calibration(
        self,
        *,
        label: str,
        stage_dir: Path,
        timing: dict[str, Any],
        eff_scale: str,
        expected_arms: tuple[str, ...],
        timed_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        cal_path = stage_dir / "timing_isolation_calibration.json"
        if not cal_path.is_file():
            return None
        helper = PostGateEAuditor(
            repo=self.repo, artifacts=stage_dir.parent, scale=eff_scale, seed=self.seed
        )
        # Note: PostGateEAuditor._audit_timing_isolation_calibration asserts `len(cal_timed_rows) == 60`
        # specifically when `scale == 'reference'` for 5-arm M4 (`6 prompts * 2 repeats * 5 arms = 60`).
        # For arbitrary `len(expected_arms)` (e.g. 4-arm M3, 7-arm lookahead, 9-arm full M4), use a generic check!
        cal = self.read_json_tracked(cal_path)
        allowed_cal_status = (
            ("complete",)
            if (self.require_complete and int(timing.get("world_size", 1)) > 1 and eff_scale in ("reference", "full"))
            else ("complete", "calibration_only_unreconciled", "awaiting_sharded_reconciliation")
        )
        self.check(
            cal.get("status") in allowed_cal_status,
            f"{label}/timing_calibration: status={cal.get('status')!r} not in {allowed_cal_status} (possible unreconciled FastPath calibration)",
        )
        expected_cal_runs = 6 * 2 * len(expected_arms)
        if eff_scale in ("reference", "full") and cal.get("calibrated_timed_runs") is not None:
            self.check(
                int(cal["calibrated_timed_runs"]) == expected_cal_runs,
                f"{label}/timing_calibration: expected {expected_cal_runs} calibrated_timed_runs for {len(expected_arms)} arms, got {cal.get('calibrated_timed_runs')}",
            )
        if int(timing.get("world_size", 1)) > 1 and cal.get("status") == "complete":
            has_real_recon = (
                cal.get("reconciled_with_sharded_timing") is True
                and bool(cal.get("groups"))
                and bool(cal.get("by_prompt"))
            )
            legacy_recon = cal.get("sharded_vs_isolated_1gpu_reconciliation") or cal.get("reconciliation")
            self.check(
                has_real_recon or (isinstance(legacy_recon, (dict, list)) and bool(legacy_recon)),
                f"{label}/timing_calibration: missing sharded_vs_isolated_1gpu reconciliation payload (reconciled_with_sharded_timing={cal.get('reconciled_with_sharded_timing')!r})",
            )
        return {
            "status": cal.get("status"),
            "calibrated_timed_runs": cal.get("calibrated_timed_runs"),
            "path": str(cal_path),
        }

    # ------------------------------------------------------------------
    # 1. Suite: `bigram_eviction` (`prompt.md` §9 & §10 M3)
    # ------------------------------------------------------------------

    def audit_bigram_eviction(self) -> dict[str, Any]:
        abl_root = self.phase_b_root / "reference_secondary_ablations"
        report_path = abl_root / "bigram_eviction.json"
        eff_scale = "smoke" if self.scale == "smoke" else "reference"
        if not report_path.is_file():
            msg = f"bigram_eviction.json not yet present at {report_path}"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "in_progress_or_not_started"}
            self.check(False, msg)
            return {"status": "missing"}

        rep = self.read_json_tracked(report_path)
        self.check(rep.get("suite") == "bigram_eviction", "bigram_eviction: wrong suite tag")
        self.check(
            rep.get("scale") == eff_scale,
            f"bigram_eviction: scale {rep.get('scale')!r} != {eff_scale!r}",
        )
        allowed = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                rep.get("status") in allowed,
                f"bigram_eviction: status {rep.get('status')!r} not in {allowed}",
            )

        arms = list(rep.get("arms", []))
        for req_arm in ("plain_dflash", "online_bigram", "online_bigram_kret256"):
            self.check(req_arm in arms, f"bigram_eviction: missing required arm {req_arm}")

        k_ret = rep.get("bigram_k_ret")
        self.check(isinstance(k_ret, int) and k_ret >= 1, f"bigram_eviction: invalid bigram_k_ret={k_ret!r}")
        if eff_scale == "reference":
            self.check(
                k_ret == 256,
                f"bigram_eviction: reference default bigram_k_ret expected 256 (`prompt.md:686, 829`), got {k_ret!r}",
            )

        sel = rep.get("selected_settings", {})
        if "online_bigram" in sel:
            self.check(
                sel["online_bigram"].get("k_ret") is None and float(sel["online_bigram"].get("alpha", 0.0)) > 0.0,
                f"bigram_eviction: online_bigram selected settings invalid: {sel['online_bigram']}",
            )
        if "online_bigram_kret256" in sel:
            self.check(
                sel["online_bigram_kret256"].get("k_ret") == k_ret
                and float(sel["online_bigram_kret256"].get("alpha", 0.0)) > 0.0,
                f"bigram_eviction: online_bigram_kret256 selected settings invalid: {sel['online_bigram_kret256']}",
            )
        if "online_bigram_kret128" in sel:
            self.check(
                sel["online_bigram_kret128"].get("k_ret") == 128
                and float(sel["online_bigram_kret128"].get("alpha", 0.0)) > 0.0,
                f"bigram_eviction: online_bigram_kret128 selected settings invalid: {sel['online_bigram_kret128']}",
            )

        ws_audit = self._audit_m5_ablation_workspace(
            label="bigram_eviction",
            eff_scale=eff_scale,
            suite_report=rep,
            artifacts_block=rep.get("artifacts", {}),
            expected_arms=arms,
            check_head_to_head_offsets=True,
            check_bigram_memory=True,
        )
        families = {r["matched_support_family"] for r in ws_audit.get("matched_support_comparisons", [])}
        self.check(
            "online_bigram_topk_eviction" in families,
            f"bigram_eviction: missing online_bigram_topk_eviction in matched_support_comparisons ({sorted(families)})",
        )
        return {
            "status": rep.get("status"),
            "scale": eff_scale,
            "config_fingerprint": rep.get("config_fingerprint"),
            "bigram_k_ret": k_ret,
            "bigram_count_gate_kappa": rep.get("bigram_count_gate_kappa"),
            "workspace_audit": ws_audit,
        }

    # ------------------------------------------------------------------
    # 2. Suite: `full_vocab_lookahead` (`prompt.md` §7 & §10 M5)
    # ------------------------------------------------------------------

    def audit_full_vocab_lookahead(self) -> dict[str, Any]:
        abl_root = self.phase_b_root / "reference_secondary_ablations"
        report_path = abl_root / "full_vocab_lookahead.json"
        eff_scale = "smoke" if self.scale == "smoke" else "reference"
        if not report_path.is_file():
            msg = f"full_vocab_lookahead.json not yet present at {report_path}"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "in_progress_or_not_started"}
            self.check(False, msg)
            return {"status": "missing"}

        rep = self.read_json_tracked(report_path)
        self.check(rep.get("suite") == "full_vocab_lookahead", "full_vocab_lookahead: wrong suite tag")
        self.check(
            rep.get("scale") == eff_scale,
            f"full_vocab_lookahead: scale {rep.get('scale')!r} != {eff_scale!r}",
        )
        allowed = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                rep.get("status") in allowed,
                f"full_vocab_lookahead: status {rep.get('status')!r} not in {allowed}",
            )

        arms = list(rep.get("arms", []))
        for req_arm in (
            "plain_dflash",
            "dense_full",
            "lookahead_static_full",
            "lookahead_rerelax_full",
            "dense_candidate",
            "lookahead_static_candidate",
            "lookahead_rerelax_candidate",
        ):
            self.check(req_arm in arms, f"full_vocab_lookahead: missing required matched-support arm {req_arm}")

        ws_audit = self._audit_m5_ablation_workspace(
            label="full_vocab_lookahead",
            eff_scale=eff_scale,
            suite_report=rep,
            artifacts_block=rep.get("artifacts", {}),
            expected_arms=arms,
            check_head_to_head_offsets=False,
            check_bigram_memory=False,
        )
        families = {r["matched_support_family"] for r in ws_audit.get("matched_support_comparisons", [])}
        self.check(
            {"full_vocabulary", "frozen_C0_candidate_masked"} <= families,
            f"full_vocab_lookahead: expected both full_vocabulary and frozen_C0_candidate_masked matched families, got {sorted(families)}",
        )
        return {
            "status": rep.get("status"),
            "scale": eff_scale,
            "config_fingerprint": rep.get("config_fingerprint"),
            "checkpoint_provenance": rep.get("checkpoint_provenance"),
            "workspace_audit": ws_audit,
        }

    # ------------------------------------------------------------------
    # 3. Suite: `pure_t1_fit` (`prompt.md` §10 M2 `--allow_pure_t1_ablation`)
    # ------------------------------------------------------------------

    def audit_pure_t1_fit(self) -> dict[str, Any]:
        abl_root = self.phase_b_root / "reference_secondary_ablations"
        report_path = abl_root / "pure_t1_fit.json"
        eff_scale = "smoke" if self.scale == "smoke" else "reference"
        if not report_path.is_file():
            msg = f"pure_t1_fit.json not yet present at {report_path}"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "in_progress_or_not_started"}
            self.check(False, msg)
            return {"status": "missing"}

        rep = self.read_json_tracked(report_path)
        self.check(rep.get("suite") == "pure_t1_fit", "pure_t1_fit: wrong suite tag")
        self.check(rep.get("scale") == eff_scale, f"pure_t1_fit: scale {rep.get('scale')!r} != {eff_scale!r}")
        allowed = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                rep.get("status") in allowed,
                f"pure_t1_fit: status {rep.get('status')!r} not in {allowed}",
            )
        self.check(
            rep.get("fit_gate_status") in ("passed", "complete"),
            f"pure_t1_fit: fit_gate_status={rep.get('fit_gate_status')!r}",
        )
        self.check(
            isinstance(rep.get("training_distribution"), str)
            and "t1" in str(rep.get("training_distribution", "")).lower(),
            f"pure_t1_fit: missing pure-T=1 training_distribution label: {rep.get('training_distribution')!r}",
        )
        self.check(
            isinstance(rep.get("ablation_warning"), str) and len(str(rep.get("ablation_warning")).strip()) > 0,
            "pure_t1_fit: missing non-empty ablation_warning disclosing changed T=1.0 training distribution",
        )

        fit_rep_p = Path(rep.get("fit_report_path") or "")
        self.check(fit_rep_p.is_file(), f"pure_t1_fit: missing fit_report_path {fit_rep_p}")
        if fit_rep_p.is_file():
            fit_json = self.read_json_tracked(fit_rep_p)
            self.check(fit_json.get("scale") == eff_scale, "pure_t1_fit: fit_report scale mismatch")
            ti = fit_json.get("trace_index")
            self.check(isinstance(ti, dict) and bool(ti), "pure_t1_fit: missing trace_index in fit_report")
            if isinstance(ti, dict):
                self.check(
                    ti.get("regime") == "pure_T1_calibration_ablation_m5",
                    f"pure_t1_fit: expected trace_index.regime=='pure_T1_calibration_ablation_m5', got {ti.get('regime')!r}",
                )
                prim_by_t = ti.get("primary_train_tokens_by_temperature") or {}
                self.check(
                    sum(int(v) for v in prim_by_t.values()) == 0,
                    f"pure_t1_fit: T=0.0/T=0.7 primary train tokens leaked into pure_t1_fit: {prim_by_t}",
                )
                if eff_scale == "reference":
                    excl = ti.get("excluded_generated_tokens") or {}
                    self.check(
                        any("long_" in str(k) for k in excl),
                        f"pure_t1_fit: reference trace_index.excluded_generated_tokens missing excluded long_* traces: {list(excl.keys())}",
                    )

        beta_cal = rep.get("beta_calibration") or {}
        self.check(
            "dense" in beta_cal and "dense_sparse" in beta_cal,
            f"pure_t1_fit: beta_calibration missing dense/dense_sparse: {list(beta_cal.keys())}",
        )
        for stg_name, binfo in beta_cal.items():
            if isinstance(binfo, dict):
                opt_b = binfo.get("optimal_beta")
                self.check(
                    not isinstance(opt_b, bool)
                    and isinstance(opt_b, (int, float))
                    and math.isfinite(float(opt_b))
                    and float(opt_b) >= 0.0,
                    f"pure_t1_fit: invalid optimal_beta={opt_b!r} for {stg_name}",
                )
                self.check(
                    binfo.get("metric") in ("dev_live_full_kl", "dev_union_rest_grouped_kl"),
                    f"pure_t1_fit: unexpected beta_calibration metric={binfo.get('metric')!r}",
                )
                cal_kl = binfo.get("calibrated_kl")
                uncal_kl = binfo.get("uncalibrated_beta1_kl")
                base0_kl = binfo.get("baseline_beta0_kl")
                if all(isinstance(x, (int, float)) for x in (cal_kl, uncal_kl, base0_kl)):
                    self.check(
                        float(cal_kl) <= float(uncal_kl) + 1e-11
                        and float(cal_kl) <= float(base0_kl) + 1e-11,
                        f"pure_t1_fit: {stg_name} calibrated_kl={cal_kl} > min(beta1={uncal_kl}, beta0={base0_kl})",
                    )

        variants = rep.get("variants", [])
        self.check(len(variants) >= 4, f"pure_t1_fit: expected >=4 variants, got {len(variants)}")
        vnames = {v.get("variant") for v in variants}
        self.check(
            {"pure_t1_uncalibrated", "pure_t1_dev_calibrated", "baseline_m2_unscaled"} <= vnames,
            f"pure_t1_fit: missing required variants in {sorted(x for x in vnames if x)}",
        )

        # Recompute closed-loop stats from raw shards when multi-GPU sharding ran
        ws_dir = fit_rep_p.parent if fit_rep_p.is_file() else None
        shard_rows_by_key: dict[str, list[dict[str, Any]]] | None = None
        if ws_dir is not None and (ws_dir / "pure_t1_closed_loop_shards").is_dir():
            sfiles = sorted((ws_dir / "pure_t1_closed_loop_shards").glob("result-rank*.json"))
            if sfiles:
                shard_rows_by_key = defaultdict(list)
                for sf in sfiles:
                    sp = self.read_json_tracked(sf)
                    for k, rlist in sp.get("rows_by_key", {}).items():
                        for rw in rlist:
                            if str(rw.get("condition", "")).startswith("long_") and "output_tokens" in rw:
                                self.check(
                                    independent_repetition_dominated(rw["output_tokens"]) == _row_rep_flag(rw),
                                    f"pure_t1_fit: repetition_dominated mismatch on {rw['prompt_id']} ({k})",
                                )
                            shard_rows_by_key[k].append(rw)

        plain_by_slot: dict[tuple[str, str, int], dict[str, Any]] = {}
        if shard_rows_by_key and "plain_dflash" in shard_rows_by_key:
            plain_by_slot = {
                (str(r["prompt_id"]), str(r["sampling_name"]), int(r["repeat"])): r
                for r in shard_rows_by_key["plain_dflash"]
            }

        variant_summaries = []
        for v in variants:
            vname = str(v["variant"])
            d_ckpt = Path(v["dense_checkpoint"])
            s_ckpt = Path(v["sparse_checkpoint"])
            self.check(d_ckpt.is_file(), f"pure_t1_fit/{vname}: missing dense_checkpoint {d_ckpt}")
            self.check(s_ckpt.is_file(), f"pure_t1_fit/{vname}: missing sparse_checkpoint {s_ckpt}")
            for kl_key, require_live_full in (
                ("primary_heldout_dense_kl", True),
                ("primary_heldout_dense_sparse_kl", True),
                ("pure_t1_heldout_dense_kl", False),
                ("pure_t1_heldout_dense_sparse_kl", False),
            ):
                kl_obj = v.get(kl_key, {})
                for split_k in ("dev", "test"):
                    split_d = kl_obj.get(split_k, {})
                    wkl = split_d.get("weighted_mean_kl")
                    self.check(
                        isinstance(wkl, (int, float)) and math.isfinite(float(wkl)) and float(wkl) >= 0.0,
                        f"pure_t1_fit/{vname}.{kl_key}.{split_k}.weighted_mean_kl invalid: {wkl!r}",
                    )
                    if require_live_full:
                        lf_slots = int((split_d.get("live_full_diagnostic") or {}).get("logical_slots", 0))
                        self.check(
                            lf_slots > 0,
                            f"pure_t1_fit/{vname}.{kl_key}.{split_k} missing live_full_diagnostic slots",
                        )

            cl = v.get("closed_loop", {})
            if self.require_complete and not self.allow_pilot:
                self.check(cl.get("is_pilot_subset") is False, f"pure_t1_fit/{vname}: pilot subset not allowed")
            for sub_arm in ("plain_dflash", "dense", "dense_sparse"):
                sub_d = cl.get(sub_arm, {})
                by_reg = sub_d.get("by_regime", {})
                # Must cover all 3 standard_nonthinking regimes (`greedy`, `nonthinking_recommended`, `pure_temperature_diagnostic`)
                for req_reg in (
                    "standard_nonthinking::greedy",
                    "standard_nonthinking::nonthinking_recommended",
                    "standard_nonthinking::pure_temperature_diagnostic",
                ):
                    self.check(
                        req_reg in by_reg,
                        f"pure_t1_fit/{vname}.{sub_arm}: missing regime {req_reg} in closed_loop.by_regime",
                    )

            if shard_rows_by_key is not None:
                for sub_arm in ("dense", "dense_sparse"):
                    raw_rows = shard_rows_by_key.get(f"{vname}::{sub_arm}", [])
                    rec_sub = cl.get(sub_arm, {})
                    prim = [r for r in raw_rows if not _is_rep_excluded(r)]
                    self.check(
                        len(raw_rows) == int(rec_sub.get("requests", -1))
                        and len(prim) == int(rec_sub.get("primary_requests", -1)),
                        f"pure_t1_fit/{vname}.{sub_arm}: raw shard count mismatch ({len(raw_rows)}/{len(prim)})",
                    )
                    if prim:
                        recomp_acc = statistics.mean(float(r["mean_accepted_length"]) for r in prim)
                        recomp_tps = statistics.mean(float(r["tokens_per_second"]) for r in prim)
                        self.check(
                            math.isclose(recomp_acc, float(rec_sub["mean_accepted_length"]), rel_tol=1e-10, abs_tol=1e-10),
                            f"pure_t1_fit/{vname}.{sub_arm}: mean_accepted_length drift",
                        )
                        self.check(
                            math.isclose(recomp_tps, float(rec_sub["mean_tokens_per_second"]), rel_tol=1e-10, abs_tol=1e-10),
                            f"pure_t1_fit/{vname}.{sub_arm}: mean_tokens_per_second drift",
                        )
                    acc_deltas, tps_ratios = [], []
                    for r in raw_rows:
                        pref = plain_by_slot.get(
                            (str(r["prompt_id"]), str(r["sampling_name"]), int(r["repeat"]))
                        )
                        if pref is None or _is_rep_excluded(r) or _is_rep_excluded(pref):
                            continue
                        acc_deltas.append(float(r["mean_accepted_length"]) - float(pref["mean_accepted_length"]))
                        if float(pref["tokens_per_second"]) > 0.0:
                            tps_ratios.append(float(r["tokens_per_second"]) / float(pref["tokens_per_second"]))
                    if acc_deltas:
                        self.check(
                            math.isclose(
                                statistics.mean(acc_deltas),
                                float(rec_sub["paired_accepted_length_delta_vs_plain"]),
                                rel_tol=1e-10,
                                abs_tol=1e-10,
                            ),
                            f"pure_t1_fit/{vname}.{sub_arm}: paired_accepted_length_delta_vs_plain drift",
                        )
                    if tps_ratios:
                        self.check(
                            math.isclose(
                                statistics.mean(tps_ratios),
                                float(rec_sub["paired_tps_ratio_vs_plain"]),
                                rel_tol=1e-10,
                                abs_tol=1e-10,
                            ),
                            f"pure_t1_fit/{vname}.{sub_arm}: paired_tps_ratio_vs_plain drift",
                        )

            variant_summaries.append(
                {
                    "variant": vname,
                    "primary_dense_test_kl": v["primary_heldout_dense_kl"]["test"]["weighted_mean_kl"],
                    "primary_sparse_test_kl": v["primary_heldout_dense_sparse_kl"]["test"]["weighted_mean_kl"],
                    "pure_t1_dense_test_kl": v["pure_t1_heldout_dense_kl"]["test"]["weighted_mean_kl"],
                    "pure_t1_sparse_test_kl": v["pure_t1_heldout_dense_sparse_kl"]["test"]["weighted_mean_kl"],
                    "closed_loop": cl,
                }
            )

        return {
            "status": rep.get("status"),
            "scale": eff_scale,
            "config_fingerprint": rep.get("config_fingerprint"),
            "training_distribution": rep.get("training_distribution"),
            "ablation_warning": rep.get("ablation_warning"),
            "beta_calibration": beta_cal,
            "variants": variant_summaries,
        }

    # ------------------------------------------------------------------
    # 4. Suite: `full_scale` (`--scale full --allow_large_run`, M1-pre -> M4)
    # ------------------------------------------------------------------

    def audit_full_scale_pipeline(self) -> dict[str, Any]:
        eff_scale = "smoke" if self.scale == "smoke" else "full"
        report_path = self.phase_b_root / "full_scale_pipeline.json"
        full_ws = self.phase_b_root
        if report_path.is_file():
            pipe_rep = self.read_json_tracked(report_path)
            ws_str = pipe_rep.get("workspace")
            if ws_str:
                full_ws = Path(ws_str).resolve()
        else:
            pipe_rep = None
            if eff_scale == "smoke":
                cands = sorted(self.phase_b_root.glob("full_scale_smoke_ws_*"))
                if cands:
                    full_ws = cands[-1].resolve()
            msg = f"full_scale_pipeline.json not yet present at {report_path}"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
            else:
                self.check(False, msg)
                return {"status": "missing"}

        if pipe_rep is not None:
            self.check(pipe_rep.get("suite") == "full_scale", "full_scale: wrong suite tag")
            self.check(
                pipe_rep.get("scale") == eff_scale,
                f"full_scale: scale {pipe_rep.get('scale')!r} != {eff_scale!r}",
            )
            allowed = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
            if self.require_complete:
                self.check(
                    pipe_rep.get("status") in allowed,
                    f"full_scale: status {pipe_rep.get('status')!r} not in {allowed}",
                )

        # Stage 0: Bootstrap & prompt capacity
        prompts_audit = self._audit_full_scale_prompts_and_m0(full_ws, eff_scale)

        # Stage 1: M1-pre calibration
        cal_audit = self._audit_full_scale_m1_calibration(full_ws, eff_scale, prompts_audit.get("train_std_ids", set()))

        # Stage 2 & 3: M1a trace collection & audit (`2M-4M` primary train tokens, >=32 long dev, >=96 long test)
        m1a_audit = self._audit_full_scale_m1a(full_ws, eff_scale, prompts_audit)

        # Stage 4: M1b diagnostic (`diagnostic_<scale>.json`)
        m1b_audit = self._audit_full_scale_m1b(full_ws, eff_scale, m1a_audit.get("m1_audit_sha256"))

        # Stage 5: M2 offline fit (`m2.json` / `m2_smoke.json`)
        m2_audit = self._audit_full_scale_m2(full_ws, eff_scale)

        # Stage 6: M3 closed-loop (4 arms, `128x(1+3+3) + 64x3 = 1088` reqs/arm at full)
        m3_audit = self._audit_full_scale_closed_loop_stage(
            full_ws=full_ws,
            stage="m3",
            eff_scale=eff_scale,
            expected_arms=M3_ARMS,
            m2_audit=m2_audit,
            prompts_audit=prompts_audit,
        )

        # Stage 7: M4 closed-loop (all 9 full-scale arms at `full`, 5 at `smoke`)
        expected_m4_arms = M4_ARMS_FULL if eff_scale == "full" else M4_ARMS_REF
        m4_audit = self._audit_full_scale_closed_loop_stage(
            full_ws=full_ws,
            stage="m4",
            eff_scale=eff_scale,
            expected_arms=expected_m4_arms,
            m2_audit=m2_audit,
            prompts_audit=prompts_audit,
        )

        return {
            "status": pipe_rep.get("status") if pipe_rep else "in_progress",
            "scale": eff_scale,
            "workspace": str(full_ws),
            "config_fingerprint": pipe_rep.get("config_fingerprint") if pipe_rep else None,
            "prompts": prompts_audit["summary"],
            "m1_calibration": cal_audit,
            "m1a_audit": m1a_audit,
            "m1b_diagnostic": m1b_audit,
            "m2_offline_fit": m2_audit,
            "m3_closed_loop": m3_audit,
            "m4_closed_loop": m4_audit,
        }

    def _audit_full_scale_prompts_and_m0(self, full_ws: Path, eff_scale: str) -> dict[str, Any]:
        smoke_p = full_ws / "smoke.json"
        if smoke_p.is_file():
            sm = self.read_json_tracked(smoke_p)
            self.check(sm.get("status") == "passed", f"full_scale: smoke.json status={sm.get('status')!r}")
        else:
            self.check(self.allow_partial, f"full_scale: missing smoke.json in {full_ws}")

        if eff_scale == "full":
            m0_distinct = 20
            for m0_name in ("m0.json", "m0_numerics.json"):
                mp = full_ws / m0_name
                if mp.is_file():
                    md = self.read_json_tracked(mp)
                    self.check(md.get("status") == "passed", f"full_scale: {m0_name} status != 'passed'")
                    if m0_name == "m0.json" and md.get("num_distinct_prompts") is not None:
                        m0_distinct = int(md["num_distinct_prompts"])
                else:
                    self.check(self.allow_partial, f"full_scale: missing {m0_name} in {full_ws}")
            if (full_ws / "m0").is_dir():
                m0_rows = list((full_ws / "m0").glob("*.json"))
                self.check(
                    len(m0_rows) >= max(20, m0_distinct),
                    f"full_scale: split-brain/incomplete m0/ dir has {len(m0_rows)} row files < {max(20, m0_distinct)}",
                )

        pm_path = full_ws / "prompts.json"
        if not pm_path.is_file():
            self.check(self.allow_partial, f"full_scale: missing prompts.json in {full_ws}")
            return {"summary": {"status": "missing"}, "train_std_ids": set(), "selected_std_test_ids": set()}

        pm = self.read_json_tracked(pm_path)
        self.check(
            pm.get("scale") == eff_scale,
            f"full_scale/prompts.json: scale {pm.get('scale')!r} != {eff_scale!r}",
        )
        rows = pm.get("prompts", [])
        ids = [str(r["prompt_id"]) for r in rows]
        self.check(len(ids) == len(set(ids)), "full_scale/prompts.json: duplicate prompt IDs")

        by_split: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            by_split[str(r["split"])].add(str(r["prompt_id"]))
        self.check(
            not (by_split["train"] & by_split["dev"])
            and not (by_split["train"] & by_split["test"])
            and not (by_split["dev"] & by_split["test"]),
            "full_scale/prompts.json: non-disjoint train/dev/test prompt splits",
        )

        std_train = [r for r in rows if r["condition"] == "standard_nonthinking" and r["split"] == "train"]
        std_dev = [r for r in rows if r["condition"] == "standard_nonthinking" and r["split"] == "dev"]
        std_test_sel = [
            r
            for r in rows
            if r["condition"] == "standard_nonthinking"
            and r["split"] == "test"
            and r.get("reference_selected", True)
        ]
        long_dev = [
            r
            for r in rows
            if str(r["condition"]).startswith("long_")
            and r["split"] == "dev"
            and r.get("reference_selected", True)
        ]
        long_test_sel = [
            r
            for r in rows
            if str(r["condition"]).startswith("long_")
            and r["split"] == "test"
            and r.get("reference_selected", True)
        ]
        cal_ids = [str(x) for x in pm.get("calibration_ids", [])]

        if eff_scale == "full":
            heldout = pm.get("complete_heldout_dataset")
            self.check(
                bool(heldout)
                and all(r["split"] == "test" for r in rows if r.get("dataset") == heldout),
                f"full_scale/prompts.json: complete_heldout_dataset={heldout!r} not held out exclusively in test",
            )
            self.check(len(cal_ids) >= 64, f"full_scale/prompts.json: calibration_ids={len(cal_ids)} < 64")
            self.check(len(std_train) >= 6800, f"full_scale/prompts.json: std_train={len(std_train)} < 6800")
            self.check(len(std_dev) >= 64, f"full_scale/prompts.json: std_dev={len(std_dev)} < 64")
            self.check(len(std_test_sel) >= 128, f"full_scale/prompts.json: std_test_selected={len(std_test_sel)} < 128")
            self.check(len(long_dev) >= 32, f"full_scale/prompts.json: long_dev={len(long_dev)} < 32")
            self.check(len(long_test_sel) >= 96, f"full_scale/prompts.json: long_test_selected={len(long_test_sel)} < 96")

        return {
            "summary": {
                "status": "verified",
                "total_prompts": len(rows),
                "calibration_prompts": len(cal_ids),
                "standard_train": len(std_train),
                "standard_dev": len(std_dev),
                "standard_test_selected": len(std_test_sel),
                "long_dev": len(long_dev),
                "long_test_selected": len(long_test_sel),
            },
            "train_std_ids": {str(r["prompt_id"]) for r in std_train},
            "dev_ids": by_split["dev"],
            "test_ids": by_split["test"],
            "selected_std_test_ids": {str(r["prompt_id"]) for r in std_test_sel},
        }

    def _audit_full_scale_m1_calibration(
        self, full_ws: Path, eff_scale: str, train_std_ids: set[str]
    ) -> dict[str, Any]:
        cal_p = full_ws / "calibration.json"
        if not cal_p.is_file():
            self.check(self.allow_partial, f"full_scale: missing calibration.json in {full_ws}")
            return {"status": "missing"}
        cal = self.read_json_tracked(cal_p)
        self.check(
            cal.get("status") == "passed" and cal.get("scale") == eff_scale,
            f"full_scale/calibration.json: status={cal.get('status')!r}, scale={cal.get('scale')!r}",
        )
        cal_pids = [str(x) for x in cal.get("identity", {}).get("prompt_ids", [])]
        if eff_scale == "full":
            self.check(len(cal_pids) >= 64, f"full_scale/calibration.json: expected >=64 prompts, got {len(cal_pids)}")
            if train_std_ids:
                self.check(
                    set(cal_pids) <= train_std_ids,
                    "full_scale/calibration.json: calibration used non-(train standard_nonthinking) prompts!",
                )
        else:
            self.check(
                all("train" in pid for pid in cal_pids),
                f"full_scale/calibration.json: smoke calibration used non-train prompts: {cal_pids}",
            )
        for name, exp_sha in (cal.get("artifact_sha256") or {}).items():
            fp = full_ws / name
            self.check(fp.is_file(), f"full_scale/calibration: missing artifact {fp}")
            if fp.is_file():
                act = sha256_file(fp)
                self.read_sha256[str(fp.resolve())] = act
                self.check(act == exp_sha, f"full_scale/calibration: SHA256 mismatch on {name}")
        return {
            "status": cal.get("status"),
            "calibration_prompts": len(cal_pids),
            "hidden_vectors": cal.get("hidden_vectors"),
        }

    def _audit_full_scale_m1a(
        self, full_ws: Path, eff_scale: str, prompts_audit: dict[str, Any]
    ) -> dict[str, Any]:
        m1_p = full_ws / "m1_audit.json"
        if not m1_p.is_file():
            self.check(self.allow_partial, f"full_scale: missing m1_audit.json in {full_ws}")
            return {"status": "missing", "m1_audit_sha256": None}

        m1 = self.read_json_tracked(m1_p)
        m1_sha = self.read_sha256[str(m1_p.resolve())]
        self.check(
            m1.get("status") == "passed" and m1.get("scale") == eff_scale,
            f"full_scale/m1_audit.json: status={m1.get('status')!r}, scale={m1.get('scale')!r}",
        )
        reqs = m1.get("requirements") or {}
        self.check(
            bool(reqs) and all(bool(v) for v in reqs.values()),
            f"full_scale/m1_audit.json: failed requirement(s): {[k for k, v in reqs.items() if not v]}",
        )

        # Independently recompute token & prompt counts from `generations` (`prompt.md:721-737`)
        gens = m1.get("generations", [])
        recomp_tokens: Counter[str] = Counter()
        recomp_prompts: dict[str, set[str]] = defaultdict(set)
        split_ids: dict[str, set[str]] = defaultdict(set)
        expected_nonthinking_warp = {"min_p": 0.0, "top_k": 20, "top_p": 0.8}

        for g in gens:
            self.check(bool(g.get("complete")), f"full_scale/m1_audit: incomplete generation included for {g.get('prompt_id')}")
            self.check(
                g.get("scale") == eff_scale,
                f"full_scale/m1_audit: wrong generation scale {g.get('scale')!r} != {eff_scale!r}",
            )
            expected_data = "synthetic" if eff_scale == "smoke" else eff_scale
            if "data" in g:
                self.check(
                    g.get("data") == expected_data,
                    f"full_scale/m1_audit: generation data label {g.get('data')!r} != {expected_data!r}",
                )
            cond = str(g["condition"])
            split = str(g["split"])
            temp = float(g["temperature"])
            pid = str(g["prompt_id"])
            rlen = int(g["realized_length"])
            cat = f"{cond}/{split}/T={temp}"
            recomp_tokens[cat] += rlen
            recomp_prompts[cat].add(pid)
            split_ids[split].add(pid)
            if cond == "standard_nonthinking" and split == "train" and g.get("partition") == "train_primary":
                warp_cfg = g.get("sampling_warp_config") or {}
                is_t0 = temp == 0.0
                is_t07 = math.isclose(temp, 0.7, abs_tol=1e-9) and all(
                    math.isclose(float(warp_cfg.get(k, -1.0)), v, abs_tol=1e-9)
                    for k, v in expected_nonthinking_warp.items()
                )
                if is_t0 or is_t07:
                    recomp_tokens["primary_train_total"] += rlen
                    recomp_tokens[f"primary_train_T={temp}"] += rlen

        self.check(
            dict(recomp_tokens) == m1.get("token_counts"),
            f"full_scale/m1_audit: token_counts drift between generations and summary",
        )
        self.check(
            {k: len(v) for k, v in recomp_prompts.items()} == m1.get("prompt_counts"),
            "full_scale/m1_audit: prompt_counts drift between generations and summary",
        )
        self.check(
            not (split_ids["train"] & split_ids["dev"])
            and not (split_ids["train"] & split_ids["test"])
            and not (split_ids["dev"] & split_ids["test"]),
            "full_scale/m1_audit: non-disjoint train/dev/test prompt IDs in collected shards",
        )

        prim_tot = int(recomp_tokens["primary_train_total"])
        prim_t0 = int(recomp_tokens["primary_train_T=0.0"])
        prim_t07 = int(recomp_tokens["primary_train_T=0.7"])
        long_cond = str(m1.get("long_condition", "long_thinking"))
        long_t = 0.6 if long_cond == "long_thinking" else 0.7
        n_long_dev = len(recomp_prompts[f"{long_cond}/dev/T={long_t}"])
        n_long_test = len(recomp_prompts[f"{long_cond}/test/T={long_t}"])
        n_diag_train = len(recomp_prompts["standard_nonthinking/train/T=1.0"])

        if eff_scale == "full":
            self.check(
                2_000_000 <= prim_tot <= 4_000_000,
                f"full_scale/m1_audit: primary_train_total={prim_tot:,} outside mandatory [2,000,000, 4,000,000] window (`prompt.md:734`)",
            )
            self.check(
                prim_t0 >= 1_000_000 and prim_t07 >= 1_000_000,
                f"full_scale/m1_audit: each primary temperature requires >=1,000,000 train tokens, got T=0.0:{prim_t0:,}, T=0.7:{prim_t07:,}",
            )
            self.check(
                n_diag_train >= 1,
                "full_scale/m1_audit: missing separate standard_nonthinking/train/T=1.0 diagnostic shards",
            )
            self.check(
                n_long_dev >= 32,
                f"full_scale/m1_audit: long dev generations={n_long_dev} < 32 (`prompt.md:736`)",
            )
            self.check(
                n_long_test >= 96,
                f"full_scale/m1_audit: long test generations={n_long_test} < 96 (`prompt.md:736`)",
            )
            for t_val in (0.0, 0.7, 1.0):
                n_std_t = len(recomp_prompts[f"standard_nonthinking/test/T={t_val}"])
                self.check(
                    n_std_t >= 128,
                    f"full_scale/m1_audit: standard_nonthinking/test/T={t_val} prompts={n_std_t} < 128",
                )
            for t_val in (0.0, 0.7):
                n_std_d = len(recomp_prompts[f"standard_nonthinking/dev/T={t_val}"])
                self.check(
                    n_std_d >= 64,
                    f"full_scale/m1_audit: standard_nonthinking/dev/T={t_val} prompts={n_std_d} < 64",
                )

        # Verify head reconstruction checks cover all retained `live_full` blocks
        head_checks = m1.get("head_reconstruction", [])
        exp_live = sum(int(g.get("live_full_blocks", 0)) for g in gens)
        passed_live = {
            (str(Path(r["shard"]).resolve()), int(r["round_idx"]))
            for r in head_checks
            if r.get("status") == "passed"
        }
        self.check(
            exp_live > 0 and len(passed_live) >= exp_live,
            f"full_scale/m1_audit: head_reconstruction passed={len(passed_live)} < expected_live_blocks={exp_live}",
        )

        # Spot-check sidecar existence & recorded SHA256 consistency on `source_shards`
        src_shards = m1.get("source_shards", [])
        self.check(
            len(src_shards) == len(m1.get("shards", [])),
            "full_scale/m1_audit: source_shards length != shards length",
        )

        def _verify_one_shard_sidecar(item: dict[str, Any]) -> str | None:
            sp = Path(item["path"])
            sc_p = sp.with_suffix(".pt.json")
            if not sp.is_file() or not sc_p.is_file():
                return f"full_scale/m1_audit: missing committed shard or sidecar {sp}"
            sc = json.loads(sc_p.read_bytes())
            if sc.get("sha256") != item.get("sha256") or sp.stat().st_size != int(sc.get("disk_bytes", -1)):
                return f"full_scale/m1_audit: sidecar sha256/disk_bytes mismatch on {sp}"
            return None

        if len(src_shards) >= 32:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=8) as ex:
                sidecar_errs = list(ex.map(_verify_one_shard_sidecar, src_shards))
            for err_msg in sidecar_errs:
                if err_msg is not None:
                    self.check(False, err_msg)
                    break
        else:
            for item in src_shards:
                err_msg = _verify_one_shard_sidecar(item)
                if err_msg is not None:
                    self.check(False, err_msg)
                    break

        return {
            "status": m1.get("status"),
            "m1_audit_sha256": m1_sha,
            "primary_train_total": prim_tot,
            "primary_train_T0": prim_t0,
            "primary_train_T07": prim_t07,
            "diagnostic_train_T1_prompts": n_diag_train,
            "long_condition": long_cond,
            "long_dev_generations": n_long_dev,
            "long_test_generations": n_long_test,
            "shards_count": len(src_shards),
        }

    def _audit_full_scale_m1b(
        self, full_ws: Path, eff_scale: str, expected_m1_sha: str | None
    ) -> dict[str, Any]:
        diag_p = full_ws / f"diagnostic_{eff_scale}.json"
        if not diag_p.is_file():
            self.check(self.allow_partial, f"full_scale: missing {diag_p.name} in {full_ws}")
            return {"status": "missing"}
        diag = self.read_json_tracked(diag_p)
        self.check(
            diag.get("status") == "measured" and diag.get("scale") == eff_scale,
            f"full_scale/{diag_p.name}: status={diag.get('status')!r}, scale={diag.get('scale')!r}",
        )
        if eff_scale == "full" and expected_m1_sha is not None:
            prov = diag.get("audit_provenance") or {}
            self.check(
                prov.get("audit_sha256") == expected_m1_sha,
                f"full_scale/{diag_p.name}: audit_sha256 {prov.get('audit_sha256')} != m1_audit.json {expected_m1_sha}",
            )
        groups = diag.get("groups", [])
        self.check(len(groups) > 0, f"full_scale/{diag_p.name}: empty diagnostic groups")
        feat_shift = diag.get("feature_shift") or diag.get("hidden_feature_summary") or groups
        self.check(bool(feat_shift), f"full_scale/{diag_p.name}: missing hidden feature covariate-shift summary")
        return {
            "status": diag.get("status"),
            "groups_count": len(groups),
            "path": str(diag_p),
        }

    def _audit_full_scale_m2(self, full_ws: Path, eff_scale: str) -> dict[str, Any]:
        m2_name = "m2_smoke.json" if eff_scale == "smoke" else "m2.json"
        m2_sub = "m2_smoke" if eff_scale == "smoke" else "m2"
        m2_p = full_ws / m2_name
        if not m2_p.is_file():
            self.check(self.allow_partial, f"full_scale: missing {m2_name} in {full_ws}")
            return {"status": "missing", "dense_sparse_norms": None}

        m2 = self.read_json_tracked(m2_p)
        self.check(
            m2.get("status") == "passed" and m2.get("scale") == eff_scale,
            f"full_scale/{m2_name}: status={m2.get('status')!r}, scale={m2.get('scale')!r}",
        )
        td_str = str(m2.get("training_distribution", ""))
        td_low = td_str.lower()
        self.check(
            td_str == "primary"
            or (
                "standard_nonthinking" in td_str
                and "long" not in td_low
                and "pure_t1" not in td_low
                and "t=1.0" not in td_low
            ),
            f"full_scale/{m2_name}: primary M2 fit has non-primary training_distribution={td_str!r}",
        )
        ti = m2.get("trace_index")
        self.check(isinstance(ti, dict) and bool(ti), f"full_scale/{m2_name}: missing trace_index")
        if isinstance(ti, dict):
            self.check(
                ti.get("regime", "primary") == "primary",
                f"full_scale/{m2_name}: trace_index.regime={ti.get('regime')!r} != 'primary'",
            )
            prim_by_t = ti.get("primary_train_tokens_by_temperature") or {}
            self.check(
                set(prim_by_t.keys()) == {"0.0", "0.7"}
                and all(int(prim_by_t[k]) > 0 for k in ("0.0", "0.7")),
                f"full_scale/{m2_name}: invalid primary_train_tokens_by_temperature keys/values: {prim_by_t}",
            )
            tot_train_toks = int(((ti.get("primary") or {}).get("train") or {}).get("generated_output_tokens", -1))
            sum_t0_t07 = sum(int(prim_by_t.get(k, 0)) for k in ("0.0", "0.7"))
            self.check(
                tot_train_toks == sum_t0_t07,
                f"full_scale/{m2_name}: pure T=1.0 or long_* tokens leaked into primary M2 fit (train total {tot_train_toks} != T=0+T=0.7 sum {sum_t0_t07})!",
            )
            if eff_scale == "full":
                self.check(
                    2_000_000 <= tot_train_toks <= 4_000_000,
                    f"full_scale/{m2_name}: trace_index train tokens={tot_train_toks:,} outside [2,000,000, 4,000,000]",
                )
                excl = ti.get("excluded_generated_tokens") or {}
                self.check(
                    any("T=1.0" in str(k) for k in excl) and any("long_" in str(k) for k in excl),
                    f"full_scale/{m2_name}: trace_index.excluded_generated_tokens missing excluded T=1.0 or long_* shards: {list(excl.keys())}",
                )

        for stg in ("dense", "dense_sparse"):
            ckpt_p = full_ws / m2_sub / f"{stg}.pt"
            self.check(ckpt_p.is_file(), f"full_scale/{m2_name}: missing checkpoint {ckpt_p}")
            if ckpt_p.is_file():
                self.read_sha256[str(ckpt_p.resolve())] = sha256_file(ckpt_p)
            st_rep = (m2.get("stages") or {}).get(stg, {})
            for sp in ("dev", "test"):
                lf_slots = int(((st_rep.get(sp) or {}).get("live_full_diagnostic") or {}).get("logical_slots", 0))
                self.check(lf_slots > 0, f"full_scale/{m2_name}: stage {stg} {sp} has 0 live_full_diagnostic slots")

        ds_stage = (m2.get("stages") or {}).get("dense_sparse", {})
        col_stats = ds_stage.get("sparse_collision_statistics") or {}
        self.check(
            "collision_count" in col_stats and "unique_observed_keys" in col_stats,
            f"full_scale/{m2_name}: missing sparse_collision_statistics in dense_sparse stage",
        )
        ds_params = ds_stage.get("parameters") or {}
        ds_norms = (
            {k: float(ds_params[k]["norm"]) for k in ("W", "A", "b") if k in ds_params}
            if ds_params
            else None
        )
        return {
            "status": m2.get("status"),
            "relative_dev_improvement": m2.get("relative_dev_improvement"),
            "under_one_percent_stop_tuning": m2.get("under_one_percent_stop_tuning"),
            "dense_sparse_norms": ds_norms,
            "sparse_collision_statistics": col_stats,
        }

    def _audit_full_scale_closed_loop_stage(
        self,
        *,
        full_ws: Path,
        stage: str,
        eff_scale: str,
        expected_arms: tuple[str, ...],
        m2_audit: dict[str, Any],
        prompts_audit: dict[str, Any],
    ) -> dict[str, Any]:
        stage_dir = full_ws / stage
        settings_p = stage_dir / "settings.json"
        summary_p = stage_dir / "summary.json"
        timing_p = stage_dir / "timing.json"
        if not settings_p.is_file():
            msg = f"full_scale/{stage}: settings.json not yet present"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"stage": stage, "status": "not_started"}
            self.check(False, msg)
            return {"stage": stage, "status": "missing"}

        settings = self.read_json_tracked(settings_p)
        frozen_hash = fingerprint(settings)
        self.check(
            settings.get("frozen") is True
            and settings.get("stage") == stage
            and settings.get("scale") == eff_scale,
            f"full_scale/{stage}: settings.json frozen/stage/scale invalid",
        )
        self.check(
            set(settings.get("selected", {}).keys()) == set(expected_arms),
            f"full_scale/{stage}: selected arms {sorted(settings.get('selected', {}).keys())} != {sorted(expected_arms)}",
        )

        # Verify dev tune selection (supports sharded 8-GPU tune at `--scale full`)
        tune_info = self._audit_dev_tune_selection(
            label=f"full_scale/{stage}/tune",
            stage_dir=stage_dir,
            settings=settings,
            arms=expected_arms,
            eff_scale=eff_scale,
        )

        # Load & verify all raw evaluate rows
        eval_rows: list[dict[str, Any]] = []
        seen_eval_keys: set[str] = set()
        eval_dir = stage_dir / "evaluate"
        if eval_dir.is_dir():
            eval_manifest_entries: list[tuple[str, dict[str, Any], Path]] = []
            for mf in sorted(eval_dir.glob("manifest-rank*.json")):
                mdata = self.read_json_tracked(mf)
                for key, entry in mdata.get("jobs", {}).items():
                    if entry.get("status") != "complete":
                        continue
                    if entry.get("identity", {}).get("frozen_settings_hash") != frozen_hash:
                        continue
                    self.check(key not in seen_eval_keys, f"full_scale/{stage}/evaluate: duplicate job key {key}")
                    seen_eval_keys.add(key)
                    eval_manifest_entries.append((key, entry, Path(entry["result_path"])))
            loaded_eval_objs = self.read_json_batch_tracked([rp for _, _, rp in eval_manifest_entries])
            for (_key, entry, rp), row in zip(eval_manifest_entries, loaded_eval_objs):
                self.check(
                    self.read_sha256[str(rp.resolve())] == entry.get("result_sha256"),
                    f"full_scale/{stage}/evaluate: SHA256 mismatch on {rp}",
                )
                self._audit_single_m3_m4_row(
                    label=f"full_scale/{stage}/evaluate",
                    stage=stage,
                    row=row,
                    selected_settings=settings["selected"],
                )
                eval_rows.append(row)

        expected_per_arm = 8 if eff_scale == "smoke" else 1088
        expected_total_eval = len(expected_arms) * expected_per_arm
        if not summary_p.is_file():
            self.check(self.allow_partial and not self.require_complete, f"full_scale/{stage}: summary.json missing")
            return {
                "stage": stage,
                "status": "in_progress",
                "completed_eval_rows": len(eval_rows),
                "expected_eval_rows": expected_total_eval,
                "tune": tune_info,
            }

        summary = self.read_json_tracked(summary_p)
        allowed = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            if not self.allow_pilot:
                self.check(
                    summary.get("status") == "complete",
                    f"full_scale/{stage}/summary.json: status={summary.get('status')!r} not in {allowed}",
                )
                if "isolated_timing_complete" in summary:
                    self.check(
                        summary.get("isolated_timing_complete") is True,
                        f"full_scale/{stage}/summary.json: isolated_timing_complete={summary.get('isolated_timing_complete')!r} != True",
                    )
            else:
                pilot_summary_ok = summary.get("status") in allowed or (
                    summary.get("status") == "incomplete"
                    and summary.get("isolated_timing_complete") is False
                    and summary.get("missing_requests") == []
                    and summary.get("greedy_output_comparisons_complete") is True
                )
                self.check(
                    pilot_summary_ok,
                    f"full_scale/{stage}/summary.json: status={summary.get('status')!r} not in {allowed}",
                )
            self.check(
                summary.get("missing_requests") == [],
                f"full_scale/{stage}/summary.json: missing_requests non-empty",
            )
        self.check(
            len(eval_rows) == int(summary.get("completed_requests", -1)),
            f"full_scale/{stage}: loaded {len(eval_rows)} eval_rows != completed_requests {summary.get('completed_requests')}",
        )
        if not self.allow_partial or len(eval_rows) == expected_total_eval:
            self.check(
                len(eval_rows) == expected_total_eval,
                f"full_scale/{stage}: expected {expected_total_eval} evaluate rows ({len(expected_arms)} arms x {expected_per_arm}), got {len(eval_rows)}",
            )

        # Verify exact `--scale full` prompt & seed counts (`>=128 x 3` std stochastic, `>=64 x 3` long stochastic)
        if eff_scale == "full" and len(eval_rows) == expected_total_eval:
            dev_ids = prompts_audit.get("dev_ids", set())
            for arm in expected_arms:
                arm_rows = [r for r in eval_rows if r["arm"] == arm]
                eval_pids = {str(r["prompt_id"]) for r in arm_rows}
                self.check(
                    not (eval_pids & dev_ids),
                    f"full_scale/{stage}/{arm}: dev prompt ID leaked into test evaluation!",
                )
                std_t0 = [r for r in arm_rows if r["condition"] == "standard_nonthinking" and float(r["temperature"]) == 0.0]
                std_t07 = [r for r in arm_rows if r["condition"] == "standard_nonthinking" and math.isclose(float(r["temperature"]), 0.7, abs_tol=1e-9)]
                std_t10 = [r for r in arm_rows if r["condition"] == "standard_nonthinking" and math.isclose(float(r["temperature"]), 1.0, abs_tol=1e-9)]
                long_rows = [r for r in arm_rows if str(r["condition"]).startswith("long_")]

                std_t07_by_p: dict[str, set[int]] = defaultdict(set)
                for r in std_t07:
                    std_t07_by_p[str(r["prompt_id"])].add(int(r["seed"]))
                std_t10_by_p: dict[str, set[int]] = defaultdict(set)
                for r in std_t10:
                    std_t10_by_p[str(r["prompt_id"])].add(int(r["seed"]))
                long_by_p: dict[str, set[int]] = defaultdict(set)
                for r in long_rows:
                    long_by_p[str(r["prompt_id"])].add(int(r["seed"]))

                self.check(
                    len({r["prompt_id"] for r in std_t0}) >= 128 and len(std_t0) >= 128,
                    f"full_scale/{stage}/{arm}: expected >=128 std greedy prompts, got {len(std_t0)}",
                )
                self.check(
                    len(std_t07_by_p) >= 128 and all(len(s) == 3 for s in std_t07_by_p.values()),
                    f"full_scale/{stage}/{arm}: expected >=128 std T=0.7 prompts x 3 seeds (`prompt.md:823`)",
                )
                self.check(
                    len(std_t10_by_p) >= 128 and all(len(s) == 3 for s in std_t10_by_p.values()),
                    f"full_scale/{stage}/{arm}: expected >=128 std T=1.0 prompts x 3 seeds",
                )
                self.check(
                    len(long_by_p) >= 64 and all(len(s) == 3 for s in long_by_p.values()),
                    f"full_scale/{stage}/{arm}: expected >=64 long test prompts x 3 stochastic seeds (`prompt.md:824`)",
                )

        # Recompute evaluate group summaries (verifies prompt-level 3-seed aggregation before bootstrap + seed_std_mean!)
        helper = PostGateEAuditor(
            repo=self.repo, artifacts=full_ws, scale=eff_scale, seed=self.seed
        )
        recomp_eval_groups = helper.recompute_group_rows(
            eval_rows, isolated_timing=False, exclude_repetition=True
        )
        ok_eg, diff_eg = close_nested(
            _canonicalize_group_list(recomp_eval_groups),
            _canonicalize_group_list(summary.get("groups", [])),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_eg, f"full_scale/{stage}: summary.groups drift from raw recomputation: {diff_eg}")

        if eff_scale == "full":
            for g in recomp_eval_groups:
                if float(g["temperature"]) > 0.0:
                    acc_m = (g.get("metrics") or {}).get("mean_accepted_length") or {}
                    if int(g.get("primary_requests", 0)) > 0:
                        self.check(
                            acc_m.get("bootstrap_unit") == "prompt_id"
                            and acc_m.get("seed_aggregation") == "arithmetic_mean_before_bootstrap"
                            and acc_m.get("seed_std_mean") is not None,
                            f"full_scale/{stage}: group {g['condition']} T={g['temperature']} {g['arm']} missing prompt-aggregated 3-seed bootstrap/variability",
                        )
                    else:
                        self.check(
                            acc_m.get("n_prompts") == 0
                            and acc_m.get("mean") is None
                            and acc_m.get("ci95") is None
                            and acc_m.get("seed_std_mean") is None,
                            f"full_scale/{stage}: 100%-excluded group {g['condition']} T={g['temperature']} {g['arm']} malformed empty bootstrap {acc_m!r}",
                        )

        # Greedy output comparisons check (`greedy_output_comparisons.json`)
        greedy_p = stage_dir / "greedy_output_comparisons.json"
        self.check(greedy_p.is_file(), f"full_scale/{stage}: missing greedy_output_comparisons.json")
        if greedy_p.is_file():
            g_rep = self.read_json_tracked(greedy_p)
            self.check(g_rep.get("status") == "complete", f"full_scale/{stage}: greedy_output_comparisons incomplete")
            if eff_scale == "smoke":
                for grp in g_rep.get("groups", []):
                    if grp.get("reference") == "target_only" and grp.get("arm") == "plain_dflash":
                        self.check(
                            int(grp.get("mismatched_tokens", 1)) == 0,
                            "full_scale/smoke: greedy plain_dflash must match target_only with 0 mismatches",
                        )

        # M4-specific checks: baseline reuse, offline base norms, and `feature_context.json`
        if stage == "m4":
            reuse_rep = summary.get("baseline_reuse") or {}
            if len(eval_rows) == expected_total_eval:
                n_reused = int(reuse_rep.get("reused_requests", -1))
                n_fresh = int(reuse_rep.get("fresh_baseline_requests", 0))
                m3_set_p = full_ws / "m3" / "settings.json"
                if m3_set_p.is_file():
                    m3_set = self.read_json_tracked(m3_set_p)
                    same_bg = (
                        (m3_set.get("selected") or {}).get("online_bigram")
                        == (settings.get("selected") or {}).get("online_bigram")
                    )
                    exp_reused = (2 * expected_per_arm) if same_bg else expected_per_arm
                    self.check(
                        n_reused == exp_reused and (n_reused + n_fresh) == 2 * expected_per_arm,
                        f"full_scale/m4: expected reused={exp_reused}, total_baseline={2 * expected_per_arm}, got reused={n_reused}, fresh={n_fresh}",
                    )
                else:
                    self.check(
                        n_reused >= expected_per_arm and (n_reused + n_fresh) == 2 * expected_per_arm,
                        f"full_scale/m4: invalid baseline_reuse counts reused={n_reused}, fresh={n_fresh}",
                    )
            ckpt_online_norms: dict[str, float] | None = None
            for r in eval_rows:
                if r["arm"] == "zero_offline_online_only":
                    pnorms = r.get("parameter_norms") or {}
                    self.check(
                        all(float(pnorms.get(k, 1.0)) == 0.0 for k in ("W", "A", "b")),
                        f"full_scale/m4: zero_offline_online_only must have exact 0.0 offline parameter_norms, got {pnorms}",
                    )
                elif r["arm"] in ONLINE_ARMS:
                    pnorms = r.get("parameter_norms") or {}
                    if ckpt_online_norms is None:
                        ckpt_online_norms = pnorms
                    else:
                        self.check(
                            pnorms == ckpt_online_norms,
                            "full_scale/m4: frozen offline parameter_norms changed across checkpoint-initialized online requests",
                        )
            if eff_scale != "smoke" and ckpt_online_norms is not None and m2_audit.get("dense_sparse_norms"):
                for k, m2_n in m2_audit["dense_sparse_norms"].items():
                    self.check(
                        math.isclose(float(ckpt_online_norms[k]), float(m2_n), rel_tol=1e-6, abs_tol=1e-6),
                        f"full_scale/m4: offline {k} norm {ckpt_online_norms[k]} != m2.json dense_sparse norm {m2_n}",
                    )

            fc_p = stage_dir / "feature_context.json"
            self.check(fc_p.is_file(), "full_scale/m4: missing m4/feature_context.json")
            if fc_p.is_file():
                fc = self.read_json_tracked(fc_p)
                self.check(
                    fc.get("status") == "complete" and int(fc.get("feature_group_count", 0)) > 0,
                    "full_scale/m4: feature_context.json incomplete",
                )
                if eff_scale != "smoke":
                    for pkey, skey in (
                        ("diagnostic_path", "diagnostic_sha256"),
                        ("audit_path", "audit_sha256"),
                        ("calibration_path", "calibration_sha256"),
                    ):
                        tf = Path(fc.get(pkey, ""))
                        self.check(tf.is_file(), f"full_scale/m4/feature_context: missing {pkey}={tf}")
                        if tf.is_file():
                            self.read_bytes_tracked(tf)
                            self.check(
                                self.read_sha256[str(tf.resolve())] == fc.get(skey),
                                f"full_scale/m4/feature_context: {pkey} SHA256 mismatch",
                            )

        # Timing verification (`timing.json` + back-to-back bundles + 1-GPU isolation calibration)
        if not timing_p.is_file():
            self.check(self.allow_partial and not self.require_complete, f"full_scale/{stage}: timing.json missing")
            return {
                "stage": stage,
                "status": "timing_missing",
                "eval_groups": recomp_eval_groups,
                "tune": tune_info,
            }

        timing = self.read_json_tracked(timing_p)
        if self.require_complete:
            self.check(
                timing.get("status") in allowed,
                f"full_scale/{stage}/timing.json: status={timing.get('status')!r} not in {allowed}",
            )

        timed_rows, profile_rows = self._load_and_audit_timing_bundles(
            label=f"full_scale/{stage}/timing",
            stage_dir=stage_dir,
            timing=timing,
            expected_arms=expected_arms,
            row_auditor=lambda rw, _ip: self._audit_single_m3_m4_row(
                label=f"full_scale/{stage}/timing",
                stage=stage,
                row=rw,
                selected_settings=settings["selected"],
            ),
        )
        # Ensure M4 timing re-measured `plain_dflash` and `online_bigram` fresh (`baseline_reuse` must be None!)
        if stage == "m4":
            for tr in timed_rows:
                self.check(
                    tr.get("baseline_reuse") is None,
                    f"full_scale/m4/timing: timing row for {tr.get('arm')} illegally reused M3 log instead of re-measuring back-to-back!",
                )

        recomp_timed_groups = self._enrich_timed_groups_with_paired_plain(
            label=f"full_scale/{stage}/timing",
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
        self.check(ok_tg, f"full_scale/{stage}: timing.groups drift from raw recomputation: {diff_tg}")

        recomp_prof_groups = helper.recompute_group_rows(
            profile_rows, isolated_timing=True, exclude_repetition=True
        )
        ok_pg, diff_pg = close_nested(
            _canonicalize_group_list(recomp_prof_groups),
            _canonicalize_group_list(timing.get("phase_profiles", [])),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        self.check(ok_pg, f"full_scale/{stage}: timing.phase_profiles drift from raw recomputation: {diff_pg}")

        cal_summary = self._audit_optional_timing_calibration(
            label=f"full_scale/{stage}",
            stage_dir=stage_dir,
            timing=timing,
            eff_scale=eff_scale,
            expected_arms=expected_arms,
            timed_rows=timed_rows,
        )

        # Sparse long offset bin flags (`< 8` prompts -> conclusion unavailable)
        eval_slices = recompute_offset_bin_slices(eval_rows, isolated_timing=False, seed=self.seed)
        sparse_bins = self._collect_sparse_long_bins(eval_slices)

        return {
            "stage": stage,
            "status": summary.get("status"),
            "timing_status": timing.get("status"),
            "arms": list(expected_arms),
            "evaluated_rows": len(eval_rows),
            "timed_rows": len(timed_rows),
            "profile_rows": len(profile_rows),
            "tune": tune_info,
            "eval_groups": recomp_eval_groups,
            "timing_groups": recomp_timed_groups,
            "timing_isolation_calibration": cal_summary,
            "sparse_long_offset_bins": sparse_bins,
        }

    def _audit_single_m3_m4_row(
        self,
        *,
        label: str,
        stage: str,
        row: dict[str, Any],
        selected_settings: dict[str, dict[str, Any]],
    ) -> None:
        arm = str(row["arm"])
        pid = str(row["prompt_id"])
        reuse = row.get("baseline_reuse")
        if reuse is not None:
            self.check(
                stage == "m4" and arm in ("plain_dflash", "online_bigram"),
                f"{label}: unexpected baseline_reuse on {stage}/{arm}",
            )
            if reuse.get("status") == "reused":
                for perf_field in (
                    "elapsed_seconds",
                    "generation_loop_elapsed_seconds",
                    "tokens_per_second",
                    "phase_ms",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                ):
                    self.check(
                        row.get(perf_field) is None,
                        f"{label}: reused M3 baseline row {pid} ({arm}) retained timing field {perf_field}",
                    )
                src_p = Path(reuse["source_result_path"])
                if src_p.is_file():
                    self.read_bytes_tracked(src_p)
                    self.check(
                        self.read_sha256[str(src_p.resolve())] == reuse.get("source_result_sha256"),
                        f"{label}: reused M3 source SHA256 mismatch on {src_p}",
                    )
                else:
                    self.check(False, f"{label}: missing reused M3 source file {src_p}")
        else:
            self.check(
                row.get("settings") == selected_settings.get(arm),
                f"{label}: row {pid} ({arm}) settings {row.get('settings')} != frozen selected {selected_settings.get(arm)}",
            )

        if str(row.get("condition", "")).startswith("long_"):
            rep_recomp = independent_repetition_dominated(row.get("output_tokens", []))
            self.check(
                rep_recomp == _row_rep_flag(row),
                f"{label}: repetition_dominated mismatch on {pid} ({arm})",
            )

        # Arm-specific M4 full-scale online adaptation invariants (`prompt.md:841-863, 923`)
        if arm in ONLINE_ARMS:
            onl = row.get("online") or {}
            inv = row.get("online_invariants") or {}
            blocks = row.get("blocks") or []
            dnorms = onl.get("delta_norms") or {}
            self.check(
                inv.get("request_reset") is True and inv.get("version_only_between_rounds") is True,
                f"{label}: online request_reset / version_only_between_rounds violated on {pid} ({arm})",
            )
            self.check(
                inv.get("single_round_no_buffering") is True,
                f"{label}: M4 arm {arm} must use single-round unbuffered updates (`prompt.md:923`), got {inv}",
            )
            expected_stride = 5 if arm == "offline_online_stride5" else 1
            self.check(
                int(inv.get("stride", -1)) == expected_stride
                and int(inv.get("executed_updates", -1)) == len(blocks) // expected_stride,
                f"{label}: {arm} stride={inv.get('stride')} executed_updates={inv.get('executed_updates')} != {len(blocks)} // {expected_stride} on {pid}",
            )
            if arm == "online_dbS":
                dw_val = onl.get("dW_norm", dnorms.get("dW", 1.0))
                da_val = onl.get("dA_norm", dnorms.get("dA", 1.0))
                self.check(
                    float(dw_val) == 0.0 and float(da_val) == 0.0,
                    f"{label}: online_dbS must keep dW==0 and dA==0 on {pid}, got dW={dw_val}, dA={da_val}",
                )
            elif arm == "online_rejection_relative":
                self.check(
                    (row.get("settings") or {}).get("weighting_mode") == "rejection_relative",
                    f"{label}: online_rejection_relative wrong weighting_mode on {pid}",
                )
            elif arm == "online_cutoff":
                self.check(
                    (row.get("settings") or {}).get("weighting_mode") == "cutoff",
                    f"{label}: online_cutoff wrong weighting_mode on {pid}",
                )

    # ------------------------------------------------------------------
    # 5. Master `phase_b_summary.json` Audit & Top-Level Orchestration
    # ------------------------------------------------------------------

    def audit_master_summary(self, suite_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        master_p = self.phase_b_root / "phase_b_summary.json"
        if not master_p.is_file():
            msg = f"phase_b_summary.json not yet present at {master_p}"
            if self.allow_partial and not self.require_complete:
                self.warn(msg)
                return {"status": "not_yet_written"}
            self.check(False, msg)
            return {"status": "missing"}

        master = self.read_json_tracked(master_p)
        allowed = ("complete", "pilot_complete") if self.allow_pilot else ("complete",)
        if self.require_complete:
            self.check(
                master.get("status") in allowed,
                f"phase_b_summary.json: status {master.get('status')!r} not in {allowed}",
            )
            exec_suites = set(master.get("executed_suites") or [])
            missing_exec = [s for s in self.selected_suites if s not in exec_suites]
            self.check(
                not missing_exec,
                f"phase_b_summary.json: executed_suites {sorted(exec_suites)} missing required suites {missing_exec}",
            )
        imm = master.get("upstream_reference_immutability") or {}
        self.check(
            imm.get("status") == "intact",
            f"phase_b_summary.json: upstream_reference_immutability status={imm.get('status')!r} != 'intact'",
        )
        rec_suites = master.get("suites") or {}
        for sname in self.selected_suites:
            if self.require_complete:
                self.check(
                    sname in rec_suites,
                    f"phase_b_summary.json: suites dict missing required suite {sname!r}",
                )
            if sname in rec_suites and suite_results.get(sname, {}).get("config_fingerprint") is not None:
                self.check(
                    rec_suites[sname].get("config_fingerprint") == suite_results[sname]["config_fingerprint"],
                    f"phase_b_summary.json: {sname} config_fingerprint mismatch vs standalone suite report",
                )
        return {
            "status": master.get("status"),
            "scale": master.get("scale"),
            "executed_suites": master.get("executed_suites"),
            "project_git_commit": master.get("project_git_commit"),
        }

    def run_all(self) -> dict[str, Any]:
        upstream = self.audit_upstream_reference_immutability()
        suite_results: dict[str, dict[str, Any]] = {}
        for sname in self.selected_suites:
            if sname == "bigram_eviction":
                suite_results[sname] = self.audit_bigram_eviction()
            elif sname == "full_vocab_lookahead":
                suite_results[sname] = self.audit_full_vocab_lookahead()
            elif sname == "pure_t1_fit":
                suite_results[sname] = self.audit_pure_t1_fit()
            elif sname == "full_scale":
                suite_results[sname] = self.audit_full_scale_pipeline()

        master_res = self.audit_master_summary(suite_results)
        self.verify_immutability_at_end()
        return {
            "status": "passed" if not self.failures else "failed",
            "scale": self.scale,
            "requested_suite": self.suite,
            "selected_suites": list(self.selected_suites),
            "checks": self.checks,
            "failures": self.failures,
            "warnings": self.warnings,
            "audited_files_count": len(self.read_sha256),
            "upstream_reference_immutability": upstream,
            "master_summary": master_res,
            "suites": suite_results,
        }


def render_phase_b_markdown(report: dict[str, Any]) -> str:
    """Render publication-ready Markdown tables for Phase-B ablations and `--scale full` results."""
    lines: list[str] = [
        f"# Phase-B Secondary Reference Ablations & Full-Scale (`{report['scale']}`) Verification Summary",
        "",
        f"- **Audit Status**: `{report['status']}` (`{report['checks']:,}` checks, "
        f"`{len(report['failures'])}` failures, `{len(report['warnings'])}` warnings, "
        f"`{report['audited_files_count']:,}` files SHA256-verified)",
        f"- **Frozen Reference Immutability**: `{report['upstream_reference_immutability']['checked_upstream_files']}` "
        f"upstream files byte-verified under `{report['upstream_reference_immutability'].get('reference_artifacts', 'N/A')}`",
        "",
    ]
    suites = report.get("suites", {})

    bg = suites.get("bigram_eviction", {})
    if bg.get("workspace_audit", {}).get("bigram_memory_ablation"):
        lines.extend(
            [
                "## 1. Secondary Reference Ablation: `OnlineBigram` Top-`k_ret` Eviction (`prompt.md` §9 & §10 M3)",
                "",
                "| Regime | Arm | `k_ret` | Mean Exact Pairs | Mean Retained Pairs | Mean Evicted Pairs | Active Prior Footprint (B) |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bg["workspace_audit"]["bigram_memory_ablation"]:
            reg = _cond_short(row["condition"], row["temperature"], row["sampling_warp_config"])
            kr = "None (exact)" if row["k_ret"] is None else str(row["k_ret"])
            lines.append(
                f"| `{reg}` | `{row['arm']}` | `{kr}` | "
                f"{row['exact_ledger_pairs_mean']:.1f} | {row['retained_prior_pairs_mean']:.1f} | "
                f"{row['evicted_prior_pairs_mean']:.1f} | {row['retained_prior_payload_bytes_mean']:.0f} |"
            )
        lines.append("")

    la = suites.get("full_vocab_lookahead", {})
    if la.get("workspace_audit", {}).get("matched_support_comparisons"):
        lines.extend(
            [
                "## 2. Secondary Reference Ablation: Full-Vocabulary vs Candidate-Masked Lookahead (`prompt.md` §7 & §10 M5)",
                "",
                "| Support Family | Regime | Lookahead Arm | Causal Matched Ref | $\\Delta E[\\tau]$ (95% CI) | Isolated Speed Ratio |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for c in la["workspace_audit"]["matched_support_comparisons"]:
            reg = _cond_short(c["condition"], c["temperature"], c["sampling_warp_config"])
            sp_rat = (c.get("isolated_timing_tokens_per_second") or {}).get("ratio")
            sp_str = f"{sp_rat:.3f}x" if isinstance(sp_rat, (int, float)) else "N/A"
            lines.append(
                f"| `{c['matched_support_family']}` | `{reg}` | `{c['arm']}` | `{c['reference_arm']}` | "
                f"{_fmt_ci(c.get('mean_accepted_length_delta'), 3)} | {sp_str} |"
            )
        lines.append("")

    pt1 = suites.get("pure_t1_fit", {})
    if pt1.get("variants"):
        lines.extend(
            [
                "## 3. Secondary Reference Ablation: Pure `T_collection=1.0` Offline Fit (`prompt.md` §10 M2)",
                "",
                f"> **Ablation Disclosure**: {pt1.get('ablation_warning', '')}",
                "",
                "| Variant | Primary Held-Out `dense_sparse` Test KL | Pure-`T=1` Held-Out `dense_sparse` Test KL | Closed-Loop $\\Delta E[\\tau]$ vs Plain | Closed-Loop TPS Ratio vs Plain |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for v in pt1["variants"]:
            ds_cl = (v.get("closed_loop") or {}).get("dense_sparse", {})
            d_acc = ds_cl.get("paired_accepted_length_delta_vs_plain")
            t_rat = ds_cl.get("paired_tps_ratio_vs_plain")
            lines.append(
                f"| `{v['variant']}` | {v['primary_sparse_test_kl']:.4f} | {v['pure_t1_sparse_test_kl']:.4f} | "
                f"{f'{d_acc:+.3f}' if isinstance(d_acc, (int, float)) else 'N/A'} | "
                f"{f'{t_rat:.3f}x' if isinstance(t_rat, (int, float)) else 'N/A'} |"
            )
        lines.append("")

    fs = suites.get("full_scale", {})
    if fs.get("m1a_audit", {}).get("status") == "passed":
        m1a = fs["m1a_audit"]
        lines.extend(
            [
                "## 4. Full-Scale (`--scale full --allow_large_run`) Pipeline Summary (`prompt.md` §10 M1–M4)",
                "",
                f"- **M1a Primary Train Tokens**: `{m1a['primary_train_total']:,}` "
                f"(`T=0.0`: `{m1a['primary_train_T0']:,}`, `T=0.7`: `{m1a['primary_train_T07']:,}`; "
                f"separate `T=1.0` diagnostic prompts: `{m1a['diagnostic_train_T1_prompts']}`)",
                f"- **M1a Long Generations (`{m1a['long_condition']}`)**: "
                f"`{m1a['long_dev_generations']}` dev + `{m1a['long_test_generations']}` test",
                "",
            ]
        )
        for stg_key, title in (("m3_closed_loop", "M3 Closed-Loop (4 Arms)"), ("m4_closed_loop", "M4 Online Adaptation (9 Full-Scale Arms)")):
            st = fs.get(stg_key, {})
            if st.get("eval_groups"):
                lines.extend(
                    [
                        f"### {title}",
                        "",
                        "| Regime | Arm | Primary Reqs | $E[\\tau]$ (95% Prompt-Bootstrap CI) | Isolated Matched TPS |",
                        "| --- | --- | ---: | --- | ---: |",
                    ]
                )
                tmap = {
                    (
                        g["condition"],
                        float(g["temperature"]),
                        json.dumps(g["sampling_warp_config"], sort_keys=True),
                        g["arm"],
                    ): g
                    for g in st.get("timing_groups", [])
                }
                for g in st["eval_groups"]:
                    reg = _cond_short(g["condition"], g["temperature"], g["sampling_warp_config"])
                    tg = tmap.get(
                        (
                            g["condition"],
                            float(g["temperature"]),
                            json.dumps(g["sampling_warp_config"], sort_keys=True),
                            g["arm"],
                        )
                    )
                    tps = tg.get("matched_arm_tokens_per_second") if tg else None
                    lines.append(
                        f"| `{reg}` | `{g['arm']}` | {g['primary_requests']} | "
                        f"{_fmt_ci(g['metrics'].get('mean_accepted_length'), 3)} | "
                        f"{f'{tps:.1f}' if isinstance(tps, (int, float)) else 'N/A'} |"
                    )
                lines.append("")
            for sb in st.get("sparse_long_offset_bins", []):
                if sb["arm"] == "plain_dflash":
                    lines.append(
                        f"- **Sparse Long-Offset Gate (`{st['stage']}` `{sb['condition']}` `{sb['bin']}`)**: "
                        f"`{sb['gate_status']}`"
                    )
            if st.get("sparse_long_offset_bins"):
                lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/catid/drafter"))
    parser.add_argument("--phase_b_root", type=Path, default=None)
    parser.add_argument("--reference_artifacts", type=Path, default=None)
    parser.add_argument("--scale", choices=["smoke", "reference", "full"], default="full")
    parser.add_argument(
        "--suite",
        choices=["all", "ref_secondary_ablations", *ALL_PHASE_B_SUITES],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow_partial", action="store_true", default=False)
    parser.add_argument("--require_complete", action="store_true", default=False)
    parser.add_argument("--allow_pilot", action="store_true", default=False)
    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument("--markdown_out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    ref_art = Path(args.reference_artifacts).resolve() if args.reference_artifacts else (repo / "artifacts")
    pb_root = Path(args.phase_b_root).resolve() if args.phase_b_root else (repo / "artifacts" / "full_scale")

    auditor = PhaseBFullScaleAuditor(
        repo=repo,
        phase_b_root=pb_root,
        reference_artifacts=ref_art,
        scale=args.scale,
        suite=args.suite,
        seed=args.seed,
        allow_partial=args.allow_partial,
        require_complete=args.require_complete,
        allow_pilot=args.allow_pilot,
    )
    report = auditor.run_all()
    if args.json_out is not None:
        atomic_write_json(Path(args.json_out), report)
    if args.markdown_out is not None:
        atomic_write_text(Path(args.markdown_out), render_phase_b_markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "scale": report["scale"],
                "suite": report["requested_suite"],
                "checks": report["checks"],
                "failures_count": len(report["failures"]),
                "warnings_count": len(report["warnings"]),
                "failures": report["failures"][:10],
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
