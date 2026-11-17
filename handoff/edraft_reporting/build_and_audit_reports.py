#!/usr/bin/env python3
"""Deterministic Post-Gate-E Verification, Independent Raw-Row Recomputer,
Markdown Table Renderer, and Full `prompt.md` + `AGENTS.md` Conformance Checker.

Strictly read-only toward the repository and input artifacts:
- Never creates, modifies, or deletes any file inside `--repo`.
- Recomputes all M3 and M4 metrics, prompt-ID bootstrap 95% CIs, seed-to-seed
  variability, block-position curves, output-offset and input-context bins
  (marking `n_prompts < 8` long bins as `SPARSE / CONCLUSION UNAVAILABLE`),
  greedy `T=0` comparisons, isolated single-GPU timing + paired ratios + all
  7 `prompt.md` §11 phases + `lookahead_iterations = 0.0 ms [disabled in M0-M4]`,
  M4 online norms/sparsity/active-buckets/bandwidth, and M1b `h[i]` covariate
  shift directly from raw request JSONs and sidecars, verifying zero drift
  against `settings.json`, `summary.json`, `timing.json`, and `m2.json`.
- Cross-checks `artifacts/m4_evidence_validator/validate.py` (`lines 296-303`)
  for the `online_bigram` zero-kernel CUDA-event `online_update` edge case.
- Verifies all 10 required `README.md` topics (`prompt.md:956-967`), all 25
  required `tests/` (`prompt.md:895-924`), `prompt.md` §13 repository layout,
  zero custom CUDA/Triton kernels in M0-M4 (`prompt.md:891`), and `AGENTS.md`
  completion/honesty gates (including the 36 frozen continuation bindings).
"""
from __future__ import annotations

import argparse
import ast
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch

M3_ARMS = ("plain_dflash", "online_bigram", "dense", "dense_sparse")
M4_ARMS = (
    "plain_dflash",
    "online_bigram",
    "offline_static",
    "offline_online_stride1",
    "online_dbS",
)
ONLINE_ARMS = frozenset(
    {
        "offline_online_stride1",
        "online_dbS",
        "offline_online_stride5",
        "zero_offline_online_only",
        "online_rejection_relative",
        "online_cutoff",
    }
)
OUTPUT_EDGES = (1024, 4096, 8192, 16384)
OUTPUT_LABELS = ("0-1K", "1K-4K", "4K-8K", "8K-16K", "16K+")
DEFAULT_CONTEXT_EDGES = (1024, 4096, 8192, 16384, 32768)
SECTION11_REQUIRED_PHASES = (
    "target_feature_extraction",
    "dflash_draft",
    "dense_adapter_score",
    "sparse_lookup",
    "sequential_sampling",
    "target_verification",
    "online_update",
)
AUX_INSTRUMENTED_PHASES = (
    "unit_log_probability",
    "feature_projection",
    "dense_pair_score",
    "bigram_score",
)
SPARSE_LONG_BIN_MIN_PROMPTS = 8

ALGORITHM_FILES = (
    "adapter.py",
    "online.py",
    "features.py",
    "dflash_loop.py",
    "sampler.py",
    "verify.py",
    "backend.py",
    "sparse.py",
    "toy.py",
    "diagnose.py",
)

IMPLEMENTATION_FILES = (
    *ALGORITHM_FILES,
    "eval_closed_loop.py",
)

HARNESS_SHARDING_AUTHORIZED_FILES = frozenset(
    {
        "eval_closed_loop.py",
        "launch.py",
        "runtime.py",
    }
)

REQUIRED_LAYOUT_PATHS = (
    "pyproject.toml",
    "README.md",
    "NOTES.md",
    "artifacts",
    "reports",
    "configs",
    "src/edraft/toy.py",
    "src/edraft/features.py",
    "src/edraft/adapter.py",
    "src/edraft/sparse.py",
    "src/edraft/dflash_loop.py",
    "src/edraft/sampler.py",
    "src/edraft/verify.py",
    "src/edraft/traces.py",
    "src/edraft/collect.py",
    "src/edraft/diagnose.py",
    "src/edraft/train.py",
    "src/edraft/eval_closed_loop.py",
    "src/edraft/online.py",
    "src/edraft/launch.py",
    "tests",
)

# Exact 1-to-1 mapping of all 25 required tests in prompt.md Section 12 (lines 899-923).
SECTION12_REQUIRED_TESTS: tuple[tuple[int, str, tuple[tuple[str, str], ...]], ...] = (
    (
        1,
        "zero-weight adapter exactly matches plain DFlash",
        (
            ("tests/test_adapter.py", "test_zero_correction_and_snapshot_ownership"),
            ("tests/test_backend.py", "test_real_loop_zero_adapter_exact"),
        ),
    ),
    (
        2,
        "synthetic CPU smoke path completes without downloads",
        (
            ("tests/test_runtime.py", "test_config_cli_override_and_smoke_never_downloads"),
            ("tests/test_toy.py", "test_synthetic_constructors_deterministic_and_no_downloads"),
            ("tests/test_backend.py", "test_cpu_smoke_round_update_and_version_isolation"),
        ),
    ),
    (
        3,
        "causal T_sample=0 never divides by zero, calls sampling softmax, or calls residual sampling",
        (
            ("tests/test_sampler_verify.py", "test_greedy_never_calls_sampling_softmax_or_residual"),
            ("tests/test_sampler_verify.py", "test_candidate_masking_and_lowest_vocabulary_tie"),
        ),
    ),
    (
        4,
        "lookahead T_sample=0 uses positive T_relax, respects C0 vs full-V domain, keeps C0 immutable, deterministic",
        (
            ("tests/test_adapter.py", "test_greedy_lookahead_positive_relax_immutable_candidates"),
        ),
    ),
    (
        5,
        "W and A[d-1] gradients match autograd on unwarped full-V L_full and P_train/q_train invariant to warpers",
        (
            ("tests/test_adapter.py", "test_full_analytic_W_A_and_one_descent_step"),
            ("tests/test_adapter.py", "test_training_distributions_invariant_to_inference_warp"),
            ("tests/test_train.py", "test_training_never_calls_inference_warpers"),
        ),
    ),
    (
        6,
        "one toy descent step lowers loss",
        (
            ("tests/test_adapter.py", "test_full_analytic_W_A_and_one_descent_step"),
            ("tests/test_adapter.py", "test_controlled_shared_free_energy_small_step_descent_with_line_search"),
        ),
    ),
    (
        7,
        "cross-boundary prefix tokens reach slot zero in causal, online bigram, and lookahead without negative-index wrap",
        (
            ("tests/test_adapter.py", "test_history_never_wraps_and_no_offset_zero"),
            ("tests/test_adapter.py", "test_causal_prefix_and_A_offset"),
            ("tests/test_online.py", "test_bigram_missing_prefix_skips_first_transition"),
            ("tests/test_adapter.py", "test_lookahead_shared_energy_partial_derivatives"),
        ),
    ),
    (
        8,
        "no self-interaction or offset indexing error and backward lookahead messages zero when i + d >= K",
        (
            ("tests/test_adapter.py", "test_history_never_wraps_and_no_offset_zero"),
            ("tests/test_adapter.py", "test_lookahead_backward_transpose_and_boundaries"),
        ),
    ),
    (
        9,
        "backward bilinear message uses A[d-1].T",
        (
            ("tests/test_adapter.py", "test_lookahead_backward_transpose_and_boundaries"),
        ),
    ),
    (
        10,
        "backward and forward hashed lookahead messages match exact partial derivatives of single masked shared energy",
        (
            ("tests/test_adapter.py", "test_lookahead_shared_energy_partial_derivatives"),
        ),
    ),
    (
        11,
        "exact toy sampling with V<=8 and K<=4",
        (
            ("tests/test_sampler_verify.py", "test_exhaustive_per_prefix_ratio_identity"),
            ("tests/test_sampler_verify.py", "test_exhaustive_multiround_sequence_and_bonus_exactness"),
        ),
    ),
    (
        12,
        "toy exactness covers target mass outside C0 (-inf mask before warper), residual, bonus, ratio rule, top_k=3/top_p=0.8/min_p",
        (
            ("tests/test_lookahead_positive.py", "test_positive_lookahead_masks_before_warper_and_preserves_outside_residual"),
            ("tests/test_sampler_verify.py", "test_forced_residual_outside_proposal_support_and_bonus"),
            ("tests/test_sampler_verify.py", "test_warper_applies_after_correction_and_matches_pinned_dflash_formula"),
            ("tests/test_sampler_verify.py", "test_min_p_and_dtype"),
        ),
    ),
    (
        13,
        "tolerances derive from sample size",
        (
            ("tests/test_sampler_verify.py", "test_actual_verifier_sampling_sample_size_tolerance"),
        ),
    ),
    (
        14,
        "greedy toy output equals target-only greedy output",
        (
            ("tests/test_toy.py", "test_greedy_speculative_output_equals_target_only"),
            ("tests/test_eval_closed_loop.py", "test_all_smoke_arms_run_exact_greedy_and_state_isolation"),
        ),
    ),
    (
        15,
        "real-model greedy mismatch is reported against target-only and plain DFlash",
        (
            ("tests/test_greedy_output_report.py", "test_paired_report_detects_adapter_mismatch_against_both_references"),
            ("tests/test_greedy_output_report.py", "test_numerical_mismatch_is_reported_without_failing_completed_coverage"),
        ),
    ),
    (
        16,
        "trace shards round-trip",
        (
            ("tests/test_traces.py", "test_atomic_shard_roundtrip_safe_load_and_live_full_gap_data"),
        ),
    ),
    (
        17,
        "hash and sign functions are deterministic",
        (
            ("tests/test_sparse.py", "test_hash_deterministic_signed_and_collision_report"),
        ),
    ),
    (
        18,
        "L1 shrinkage preserves exact zeros, pure read-time evaluation never mutates, lazy cum_thresh matches eager over 10+ updates (stride 1 & 5, const & scheduled LR)",
        (
            ("tests/test_sparse.py", "test_proximal_preserves_exact_zero"),
            ("tests/test_sparse.py", "test_lazy_matches_eager_15_updates_and_pure_reads"),
            ("tests/test_online.py", "test_reported_sparse_norm_uses_lazy_effective_values_without_mutation"),
        ),
    ),
    (
        19,
        "snapshot updates and sequential clamping of support[i] never mutate frozen ProposalSnapshot or C0[j]",
        (
            ("tests/test_adapter.py", "test_zero_correction_and_snapshot_ownership"),
            ("tests/test_adapter.py", "test_greedy_lookahead_positive_relax_immutable_candidates"),
            ("tests/test_online.py", "test_online_single_round_stride_snapshot_and_request_isolation"),
        ),
    ),
    (
        20,
        "adapter_version changes only between rounds",
        (
            ("tests/test_online.py", "test_online_single_round_stride_snapshot_and_request_isolation"),
            ("tests/test_backend.py", "test_cpu_smoke_round_update_and_version_isolation"),
        ),
    ),
    (
        21,
        "online request state is isolated",
        (
            ("tests/test_online.py", "test_online_single_round_stride_snapshot_and_request_isolation"),
            ("tests/test_eval_closed_loop.py", "test_all_smoke_arms_run_exact_greedy_and_state_isolation"),
        ),
    ),
    (
        22,
        "target-cache rollback preserves target-only behavior",
        (
            ("tests/test_toy.py", "test_cache_partial_acceptance_rollback_preserves_target"),
            ("tests/test_backend.py", "test_actual_target_cache_partial_rollback"),
        ),
    ),
    (
        23,
        "unwarped unit-temperature log-probabilities reconstructed via draft_lm_head(h_raw_bf16) match live DFlash l_unit within bf16 tolerance on synthetic and real hooks",
        (
            ("tests/test_toy.py", "test_bf16_hidden_reconstruction_tolerance_from_rounding_bound"),
            ("tests/test_backend.py", "test_head_hook_reconstructs_live_installed_classes"),
            ("tests/test_traces.py", "test_reconstruction_diagnostic_checks_full_vocabulary_and_reports_tolerances"),
        ),
    ),
    (
        24,
        "online bigram baseline rejects alpha<=0, handles missing history, normalized over V with and without top-k_ret eviction, N_total=0 or lambda_tr=0 bit-exact with plain DFlash",
        (
            ("tests/test_online.py", "test_bigram_rejects_bad_alpha"),
            ("tests/test_online.py", "test_bigram_normalized_evicted_prior_and_empty_equivalence"),
            ("tests/test_online.py", "test_bigram_missing_prefix_skips_first_transition"),
            ("tests/test_backend.py", "test_zero_strength_bigram_bit_exact_and_initial_transition"),
        ),
    ),
    (
        25,
        "M4 online L_full update executes single-round updates and increments u only on active stride boundaries without buffering skipped rounds",
        (
            ("tests/test_online.py", "test_online_single_round_stride_snapshot_and_request_isolation"),
            ("tests/test_online.py", "test_online_bias_only_and_inference_mode_context"),
        ),
    ),
)

# All 10 required README explanations in prompt.md lines 956-967.
README_REQUIRED_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("proposal_equations", (r"U_all\s*=\s*\(H\s*@\s*W\.T\)\s*@\s*E\.T\s*\+\s*b", r"beta_sample")),
    ("temperatures_T_sample_T_relax_T_train", (r"T_sample", r"T_relax", r"T_train\s*=\s*1")),
    ("full_vocabulary_dense_scoring", (r"full-vocabulary", r"M_sparse")),
    ("exact_ratio_residual_correction", (r"min\(1,\s*P\(y\)/q\(y\)\)", r"normalize\(max\(P-q,\s*0\)\)")),
    ("union_rest_grouped_and_h_raw_bf16", (r"union_rest_grouped", r"h_raw_bf16")),
    ("hashed_pair_collision_and_lazy_cum_thresh", (r"collision", r"cum_thresh")),
    ("cross_boundary_context_and_missing_history", (r"i-d\s*<\s*0", r"prefix\[len\(prefix\)\+i-d\]", r"Missing history is skipped")),
    ("snapshot_version_C0_support_and_stride", (r"C0", r"support", r"clamped", r"stride")),
    ("dflash_and_backend_limitations", (r"DFlash2DraftModel", r"BF16", r"Token Recycling", r"Test-Time Speculation", r"SharpDraft")),
    ("exact_smoke_and_reference_commands", (r"python -m edraft\.smoke --scale smoke", r"python -m edraft\.eval_closed_loop --scale reference --stage m4")),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compute_files_sha256(edraft_src_dir: Path, file_names: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for name in file_names:
        h.update(name.encode("utf-8"))
        h.update((edraft_src_dir / name).read_bytes())
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def output_bin_label(offset: int) -> str:
    if offset < 0:
        raise ValueError("negative output offset")
    return OUTPUT_LABELS[bisect.bisect_right(OUTPUT_EDGES, offset)]


def context_bin_label(length: int, edges: tuple[int, ...] = DEFAULT_CONTEXT_EDGES) -> str:
    if length < 0:
        raise ValueError("negative context length")
    idx = bisect.bisect_right(edges, length)
    lo = 0 if idx == 0 else edges[idx - 1]
    hi = edges[idx] if idx < len(edges) else None
    return f"{lo}-{hi}" if hi is not None else f"{lo}+"


_REP_DET_CACHE: dict[tuple[int, bytes], bool] = {}


def independent_repetition_dominated(tokens: list[int]) -> bool:
    """Independent v2 periodic + n-gram dominance check (`prompt.md:777`)."""
    n_tok = len(tokens)
    if n_tok < 256:
        return False
    arr = np.asarray(tokens, dtype=np.int64)
    ck = (n_tok, hashlib.blake2b(arr.tobytes(), digest_size=16).digest())
    cached = _REP_DET_CACHE.get(ck)
    if cached is not None:
        return cached
    best_span = 0
    for period in range(1, n_tok // 4 + 1):
        eq = arr[period:] == arr[:-period]
        if int(np.count_nonzero(eq)) < 3 * period:
            continue
        bounds = np.concatenate(([-1], np.flatnonzero(~eq), [len(eq)]))
        max_match = int((np.diff(bounds) - 1).max())
        span = max_match + period
        if max_match >= 3 * period and span > best_span:
            best_span = span
    if best_span >= max(256, 0.5 * n_tok):
        if len(_REP_DET_CACHE) < 8192:
            _REP_DET_CACHE[ck] = True
        return True
    c8 = Counter(tuple(tokens[i : i + 8]) for i in range(n_tok - 7))
    w8 = n_tok - 7
    rep8 = 1.0 - len(c8) / w8 if w8 else 0.0
    c16 = Counter(tuple(tokens[i : i + 16]) for i in range(n_tok - 15))
    dom16 = min(1.0, max(c16.values(), default=0) * 16 / n_tok)
    res = bool(rep8 >= 0.8 and dom16 >= 0.5)
    if len(_REP_DET_CACHE) < 8192:
        _REP_DET_CACHE[ck] = res
    return res


_BOOT_IDX_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}
_QUANT_IDX_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}
_Q95_TENSOR = torch.tensor([0.025, 0.975], dtype=torch.float64)


def independent_prompt_bootstrap(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    seed: int = 2026,
    iterations: int = 2000,
) -> dict[str, Any]:
    """Aggregate multiple seeds within each prompt before prompt-ID resampling (`prompt.md:835`)."""
    by_seed: dict[tuple[str, Any], list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(value_key)
        if val is None or not math.isfinite(float(val)):
            continue
        by_seed[(str(row["prompt_id"]), row.get("seed", 0))].append(float(val))
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for (prompt_id, _run_seed), vals in by_seed.items():
        by_prompt[prompt_id].append(statistics.mean(vals))
    if not by_prompt:
        return {"n_prompts": 0, "mean": None, "ci95": None, "seed_std_mean": None}
    sorted_items = sorted(by_prompt.items())
    prompt_means = torch.tensor(
        [statistics.mean(vals) for _, vals in sorted_items], dtype=torch.float64
    )
    n_pm = len(prompt_means)
    bkey = (n_pm, seed, iterations)
    idx_all = _BOOT_IDX_CACHE.get(bkey)
    if idx_all is None:
        gen = torch.Generator().manual_seed(seed)
        chunks = []
        for start in range(0, iterations, 128):
            b = min(128, iterations - start)
            chunks.append(torch.randint(n_pm, (b, n_pm), generator=gen))
        idx_all = torch.cat(chunks, dim=0)
        if len(_BOOT_IDX_CACHE) < 512:
            _BOOT_IDX_CACHE[bkey] = idx_all
    ci95 = torch.quantile(prompt_means[idx_all].mean(-1), _Q95_TENSOR).tolist()
    seed_std_mean = statistics.mean(
        statistics.pstdev(vals) if len(vals) > 1 else 0.0 for _, vals in sorted_items
    )
    return {
        "n_prompts": len(by_prompt),
        "mean": float(prompt_means.mean()),
        "ci95": ci95,
        "seed_std_mean": seed_std_mean,
        "bootstrap_unit": "prompt_id",
        "seed_aggregation": "arithmetic_mean_before_bootstrap",
    }


def independent_prompt_quantile_ci(
    rows: list[dict[str, Any]],
    value_key: str,
    quantile: float,
    *,
    seed: int = 2026,
    iterations: int = 2000,
) -> dict[str, Any]:
    by_seed: dict[tuple[str, Any], list[float]] = defaultdict(list)
    for row in rows:
        by_seed[(str(row["prompt_id"]), row["seed"])].append(float(row[value_key]))
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for (prompt_id, _), vals in by_seed.items():
        by_prompt[prompt_id].append(statistics.mean(vals))
    if not by_prompt:
        return {"estimate": None, "ci95": None, "n_prompts": 0}
    values = torch.tensor(
        [statistics.mean(v) for _, v in sorted(by_prompt.items())], dtype=torch.float64
    )
    n_val = len(values)
    qkey = (n_val, seed, iterations)
    idx = _QUANT_IDX_CACHE.get(qkey)
    if idx is None:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randint(n_val, (iterations, n_val), generator=gen)
        if len(_QUANT_IDX_CACHE) < 512:
            _QUANT_IDX_CACHE[qkey] = idx
    ests = torch.quantile(values[idx], quantile, dim=-1)
    return {
        "estimate": float(torch.quantile(values, quantile)),
        "ci95": torch.quantile(ests, _Q95_TENSOR).tolist(),
        "n_prompts": len(values),
        "unit": "prompt_id",
        "seed_aggregation": "mean before prompt resampling",
    }


def independent_greedy_mismatch(output: list[int], reference: list[int]) -> dict[str, Any]:
    count = max(len(output), len(reference))
    diffs = [
        i
        for i in range(count)
        if i >= len(output) or i >= len(reference) or output[i] != reference[i]
    ]
    return {
        "sequence_equal": not diffs,
        "mismatched_tokens": len(diffs),
        "compared_tokens": count,
        "token_mismatch_rate": len(diffs) / count if count else 0.0,
        "first_mismatch": diffs[0] if diffs else None,
        "output_length": len(output),
        "reference_length": len(reference),
        "length_difference": len(output) - len(reference),
    }


def close_nested(a: Any, b: Any, *, rel_tol: float = 1e-11, abs_tol: float = 1e-11) -> tuple[bool, str]:
    """Deep exact/float-tight structural comparison against recorded summary artifacts."""
    if type(a) is bool or type(b) is bool:
        return (a is b, f"bool mismatch: {a!r} vs {b!r}")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return (False, f"NaN encountered: {a!r} vs {b!r}")
        ok = math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        return (ok, f"float mismatch: {a!r} vs {b!r}")
    if type(a) is not type(b):
        return (False, f"type mismatch: {type(a).__name__} ({a!r}) vs {type(b).__name__} ({b!r})")
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return (
                False,
                f"dict keys differ: only_left={sorted(set(a) - set(b))}, only_right={sorted(set(b) - set(a))}",
            )
        for k in sorted(a.keys()):
            ok, msg = close_nested(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol)
            if not ok:
                return (False, f".{k}: {msg}")
        return (True, "")
    if isinstance(a, list):
        if len(a) != len(b):
            return (False, f"list len {len(a)} != {len(b)}")
        for idx, (va, vb) in enumerate(zip(a, b)):
            ok, msg = close_nested(va, vb, rel_tol=rel_tol, abs_tol=abs_tol)
            if not ok:
                return (False, f"[{idx}]: {msg}")
        return (True, "")
    return (a == b, f"value mismatch: {a!r} vs {b!r}")


def _canonicalize_group_list(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip auditor-only keys and sort unordered multi-rank `online_final_state` lists canonically."""
    out: list[dict[str, Any]] = []
    for g in groups:
        d = {k: v for k, v in g.items() if k != "excluded_prompt_ids"}
        ofs = d.get("online_final_state")
        if isinstance(ofs, list) and ofs and isinstance(ofs[0], dict):
            d["online_final_state"] = sorted(
                ofs,
                key=lambda s: (
                    int(s.get("u") or 0),
                    int(s.get("last_round") or 0),
                    int(s.get("buffered_pending_rounds") or 0),
                    int(s.get("buffered_rounds_accumulated_total") or 0),
                    int(s.get("active_buckets") or 0),
                    int(s.get("state_bytes_estimate_total") or 0),
                    round(float(s.get("dS_norm") or 0.0), 12),
                    round(float(s.get("db_norm") or 0.0), 12),
                    s.get("last_full_loss") is not None,
                    round(float(s.get("last_full_loss") or 0.0), 9),
                    json.dumps(s, sort_keys=True),
                ),
            )
        out.append(d)
    return out


class PostGateEAuditor:
    def __init__(
        self,
        *,
        repo: Path,
        artifacts: Path,
        scale: str,
        seed: int = 2026,
        allow_partial: bool = False,
        require_complete: bool = False,
        source_repo: Path | None = None,
    ) -> None:
        self.repo = repo.resolve()
        self.source_repo = source_repo.resolve() if source_repo is not None else None
        self.artifacts = artifacts.resolve()
        self.scale = scale
        self.seed = seed
        self.allow_partial = allow_partial
        self.require_complete = require_complete
        self.checks = 0
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.read_sha256: dict[str, str] = {}
        self.m4_timing_world_size: int = 1
        self._git_09a5469_cache: dict[str, bytes | None] = {}
        self._git_09a5469_checked: bool = False
        self._git_09a5469_available: bool = False

    def _is_post_m4_repo(self) -> bool:
        return (
            (self.repo / "src/edraft/post_m4_suite.py").is_file()
            or (self.repo / "src/edraft/fastpath.py").is_file()
        )

    def _read_09a5469_repo_file(self, rel_path: str) -> bytes | None:
        """Read frozen M0-M4 `09a5469` file bytes from `--source_repo`, local git history, or `--repo`."""
        rel_norm = rel_path.lstrip("/")
        if self.source_repo is not None:
            cand = self.source_repo / rel_norm
            if cand.is_file():
                return self.read_bytes_tracked(cand)
        if rel_norm in self._git_09a5469_cache:
            return self._git_09a5469_cache[rel_norm]
        if self._is_post_m4_repo() and (self.repo / ".git").exists():
            if not self._git_09a5469_checked:
                self._git_09a5469_checked = True
                chk = subprocess.run(
                    ["git", "-C", str(self.repo), "cat-file", "-e", "09a5469c72c442408d1a85af7ccd58c231db9fb9^{commit}"],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    capture_output=True,
                    check=False,
                )
                self._git_09a5469_available = chk.returncode == 0
            if self._git_09a5469_available:
                proc = subprocess.run(
                    ["git", "-C", str(self.repo), "show", f"09a5469c72c442408d1a85af7ccd58c231db9fb9:{rel_norm}"],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    capture_output=True,
                    check=False,
                )
                blob = proc.stdout if proc.returncode == 0 else None
                self._git_09a5469_cache[rel_norm] = blob
                if blob is not None:
                    return blob
        cand_repo = self.repo / rel_norm
        if cand_repo.is_file():
            return self.read_bytes_tracked(cand_repo)
        return None

    def _compute_09a5469_alg_sha256(self) -> tuple[str | None, list[str]]:
        missing: list[str] = []
        h = hashlib.sha256()
        for fn in ALGORITHM_FILES:
            blob = self._read_09a5469_repo_file(f"src/edraft/{fn}")
            if blob is None:
                missing.append(fn)
            else:
                h.update(fn.encode("utf-8"))
                h.update(blob)
        return (h.hexdigest() if not missing else None), missing

    def check(self, cond: bool, message: str) -> bool:
        self.checks += 1
        if not cond:
            self.failures.append(message)
        return bool(cond)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def _is_live_inflight_manifest(self, path: Path) -> bool:
        try:
            rel = Path(path).resolve().relative_to(self.artifacts)
        except ValueError:
            return False
        if len(rel.parts) < 3 or rel.parts[0] not in ("m3", "m4") or not rel.name.endswith(".json"):
            return False
        if rel.name.startswith("manifest"):
            return True
        if "sessions" in rel.parts:
            return True
        return False

    def read_bytes_tracked(self, path: Path) -> bytes:
        p = Path(path).resolve()
        data = p.read_bytes()
        digest = sha256_bytes(data)
        prev = self.read_sha256.get(str(p))
        if prev is not None and prev != digest:
            if self.allow_partial and not self.require_complete and self._is_live_inflight_manifest(p):
                self.warn(f"inflight manifest advanced during live --allow_partial audit: {p}")
            else:
                self.check(False, f"file mutated during audit: {p}")
        else:
            self.check(True, f"file immutable during audit: {p}")
        self.read_sha256[str(p)] = digest
        return data

    def read_json_tracked(self, path: Path) -> Any:
        return json.loads(self.read_bytes_tracked(path))

    def recompute_group_rows(
        self, rows: list[dict[str, Any]], *, isolated_timing: bool = False, exclude_repetition: bool = True
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                row["condition"],
                float(row["temperature"]),
                json.dumps(row["sampling_warp_config"], sort_keys=True),
                row["arm"],
            )
            groups[key].append(row)
        out = []
        for (condition, temperature, warp_json, arm), group in sorted(groups.items()):
            if exclude_repetition:
                primary = [
                    r
                    for r in group
                    if not (
                        condition.startswith("long_") and r["generation"]["repetition_dominated"]
                    )
                ]
            else:
                primary = list(group)
            excluded_prompt_ids = sorted(
                {
                    r["prompt_id"]
                    for r in group
                    if condition.startswith("long_") and r["generation"]["repetition_dominated"]
                }
            )
            timed_primary = [r for r in primary if r.get("elapsed_seconds") is not None]
            metrics: dict[str, Any] = {
                m: independent_prompt_bootstrap(primary, m, seed=self.seed)
                for m in (
                    "mean_accepted_length",
                    "median_accepted_length",
                    "target_forwards_per_generated_token",
                )
            }
            if any(r.get("estimated_online_state_bandwidth_GBps") is not None for r in primary):
                metrics["estimated_online_state_bandwidth_GBps"] = independent_prompt_bootstrap(
                    primary, "estimated_online_state_bandwidth_GBps", seed=self.seed
                )
            if isolated_timing:
                metrics["tokens_per_second"] = independent_prompt_bootstrap(
                    [r for r in primary if r.get("tokens_per_second") is not None],
                    "tokens_per_second",
                    seed=self.seed,
                )
                metrics["elapsed_seconds"] = independent_prompt_bootstrap(
                    timed_primary, "elapsed_seconds", seed=self.seed
                )

            bins: dict[tuple[str, str], dict[tuple[str, Any], list[int]]] = defaultdict(
                lambda: defaultdict(lambda: [0, 0])
            )
            accepted_in_bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            by_pos: dict[int, dict[tuple[str, Any], list[int]]] = defaultdict(
                lambda: defaultdict(lambda: [0, 0])
            )
            for r in primary:
                ps_key = (r["prompt_id"], r["seed"])
                row_blocks = r["blocks"]
                pos_yes_row: dict[int, int] = defaultdict(int)
                pos_tot_row: dict[int, int] = defaultdict(int)
                for block in row_blocks:
                    acc = int(block["accepted_length"])
                    base_off = int(block["absolute_output_offset"])
                    ctx_len = int(block["input_context_length"])
                    out_lbl = output_bin_label(base_off)
                    ctx_lbl = context_bin_label(ctx_len)
                    entry_rec = {"prompt_id": ps_key[0], "seed": ps_key[1], "accepted_length": acc}
                    accepted_in_bins[("output_offset", out_lbl)].append(entry_rec)
                    accepted_in_bins[("input_context_length", ctx_lbl)].append(entry_rec)
                    n_draft = len(block["draft_tokens"])
                    if n_draft == 0:
                        continue
                    acc_clamped = min(max(acc, 0), n_draft)
                    if out_lbl == output_bin_label(base_off + n_draft - 1):
                        c_o = bins[("output_offset", out_lbl)][ps_key]
                        c_o[0] += acc_clamped
                        c_o[1] += n_draft
                    else:
                        for i in range(n_draft):
                            c_o = bins[("output_offset", output_bin_label(base_off + i))][ps_key]
                            c_o[0] += int(i < acc)
                            c_o[1] += 1
                    if ctx_lbl == context_bin_label(ctx_len + n_draft - 1):
                        c_c = bins[("input_context_length", ctx_lbl)][ps_key]
                        c_c[0] += acc_clamped
                        c_c[1] += n_draft
                    else:
                        for i in range(n_draft):
                            c_c = bins[("input_context_length", context_bin_label(ctx_len + i))][ps_key]
                            c_c[0] += int(i < acc)
                            c_c[1] += 1
                    for i in range(n_draft):
                        if i < acc:
                            pos_yes_row[i] += 1
                        pos_tot_row[i] += 1
                for i, tot_i in pos_tot_row.items():
                    cp = by_pos[i][ps_key]
                    cp[0] += pos_yes_row[i]
                    cp[1] += tot_i

            def rate_boot(data: dict[tuple[str, Any], list[int]]) -> dict[str, Any]:
                obs = [
                    {"prompt_id": p, "seed": s, "rate": yes / tot}
                    for (p, s), (yes, tot) in data.items()
                ]
                return independent_prompt_bootstrap(obs, "rate", seed=self.seed)

            pooled_lengths = [int(b["accepted_length"]) for r in primary for b in r["blocks"]]
            latency = torch.tensor(
                [float(r["elapsed_seconds"]) for r in timed_primary], dtype=torch.float64
            )
            phases: dict[str, list[float]] = defaultdict(list)
            for r in primary:
                for ph, val in (r.get("phase_ms") or {}).items():
                    phases[ph].append(float(val))

            acceptance_bins = []
            for (axis, label), data in sorted(bins.items()):
                rb = rate_boot(data)
                ab = independent_prompt_bootstrap(
                    accepted_in_bins[(axis, label)], "accepted_length", seed=self.seed
                )
                acceptance_bins.append(
                    {
                        "axis": axis,
                        "bin": label,
                        **rb,
                        "accepted_length": ab,
                    }
                )

            out.append(
                {
                    "condition": condition,
                    "temperature": temperature,
                    "sampling_warp_config": json.loads(warp_json),
                    "arm": arm,
                    "requests": len(group),
                    "primary_requests": len(primary),
                    "repetition_excluded": (len(group) - len(primary)) if exclude_repetition else 0,
                    "excluded_prompt_ids": excluded_prompt_ids if exclude_repetition else [],
                    "reused_acceptance_output_requests": sum(
                        r.get("baseline_reuse", {}).get("status") == "reused" for r in group
                    ),
                    "requests_with_latency": len(timed_primary),
                    "requests_without_latency": len(primary) - len(timed_primary),
                    "metrics": metrics,
                    "pooled_block_accepted_length_mean": (
                        statistics.mean(pooled_lengths) if pooled_lengths else None
                    ),
                    "pooled_block_accepted_length_median": (
                        statistics.median(pooled_lengths) if pooled_lengths else None
                    ),
                    "metric_aggregation": (
                        "metrics are prompt-weighted after seed aggregation; "
                        "median_accepted_length aggregates request medians; "
                        "pooled block statistics are separate"
                    ),
                    "acceptance_by_block_position": {
                        str(i): rate_boot(data) for i, data in sorted(by_pos.items())
                    },
                    "acceptance_bins": acceptance_bins,
                    "request_latency_p50_seconds": (
                        float(torch.quantile(latency, 0.5)) if len(latency) else None
                    ),
                    "request_latency_p95_seconds": (
                        float(torch.quantile(latency, 0.95)) if len(latency) else None
                    ),
                    "request_latency_p50_prompt_bootstrap": (
                        independent_prompt_quantile_ci(
                            timed_primary, "elapsed_seconds", 0.5, seed=self.seed
                        )
                        if isolated_timing
                        else None
                    ),
                    "request_latency_p95_prompt_bootstrap": (
                        independent_prompt_quantile_ci(
                            timed_primary, "elapsed_seconds", 0.95, seed=self.seed
                        )
                        if isolated_timing
                        else None
                    ),
                    "phase_ms_median": {
                        ph: statistics.median(vals) for ph, vals in phases.items()
                    },
                    "realized_output_length_mean": (
                        statistics.mean(r["generated_tokens"] for r in primary) if primary else None
                    ),
                    "realized_generations_above_8K": sum(
                        r["generated_tokens"] > 8192 for r in primary
                    ),
                    "realized_generations_above_16K": sum(
                        r["generated_tokens"] > 16384 for r in primary
                    ),
                    "peak_allocated_bytes": max(
                        (
                            r["peak_allocated_bytes"]
                            for r in group
                            if r.get("peak_allocated_bytes") is not None
                        ),
                        default=None,
                    ),
                    "peak_reserved_bytes": max(
                        (
                            r["peak_reserved_bytes"]
                            for r in group
                            if r.get("peak_reserved_bytes") is not None
                        ),
                        default=None,
                    ),
                    "online_final_state": [
                        r["online"] for r in primary if r.get("online") is not None
                    ],
                }
            )
        return out

    def audit_online_bigram_cuda_events(self) -> dict[str, Any]:
        """Item 1 of `#51`: cross-check `validate.py` re `online_bigram` `online_update` across all timing ranks."""
        validator_path = self.repo / "artifacts/m4_evidence_validator/validate.py"
        lines = self.read_bytes_tracked(validator_path).decode("utf-8").splitlines()
        anchor_idx = next(
            (i for i, ln in enumerate(lines) if "required_phases.append('online_update')" in ln),
            None,
        )
        if anchor_idx is not None:
            lo = max(0, anchor_idx - 5)
            hi = min(len(lines), anchor_idx + 8)
            snippet = "\n".join(lines[lo:hi])
        else:
            snippet = "\n".join(lines[292:304])
        self.check(
            "identity['proposal_mode'] in ('online_bigram', *ONLINE)" in snippet
            and "required_phases.append('online_update')" in snippet
            and "value > 0" in snippet,
            "validate.py changed unexpectedly from audited online_update > 0 rule",
        )

        stage_stats: dict[str, Any] = {}
        for stage in ("m3", "m4"):
            timing_dir = self.artifacts / stage / "timing"
            rank_dirs = sorted(
                d for d in timing_dir.glob("rank*") if d.is_dir()
            ) if timing_dir.is_dir() else []
            if not rank_dirs:
                stage_stats[stage] = {"present": False, "profile_rows": 0}
                continue
            by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r_dir in rank_dirs:
                for row_path in sorted(r_dir.glob("*.json")):
                    try:
                        row = self.read_json_tracked(row_path)
                    except Exception:
                        continue
                    if not row.get("phase_profile"):
                        continue
                    blocks = len(row.get("blocks", []))
                    upd_ms = (row.get("phase_ms") or {}).get("online_update")
                    by_arm[row["arm"]].append(
                        {
                            "prompt_id": row["prompt_id"],
                            "condition": row["condition"],
                            "sampling_name": row["sampling_name"],
                            "blocks": blocks,
                            "online_update_ms": upd_ms,
                            "us_per_block": (upd_ms * 1000.0 / blocks)
                            if (upd_ms is not None and blocks > 0)
                            else None,
                        }
                    )
            arm_summaries = {}
            for arm, items in sorted(by_arm.items()):
                nonempty = [x for x in items if x["blocks"] > 0]
                zero_or_missing = [
                    x
                    for x in nonempty
                    if x["online_update_ms"] is None or float(x["online_update_ms"]) <= 0.0
                ]
                if arm in ("online_bigram", *ONLINE_ARMS):
                    self.check(
                        len(zero_or_missing) == 0,
                        f"{stage}/{arm}: {len(zero_or_missing)} profile rows with blocks>0 have online_update<=0 (would fail validate.py:302)",
                    )
                vals = [float(x["online_update_ms"]) for x in nonempty if x["online_update_ms"] is not None]
                us_vals = [float(x["us_per_block"]) for x in nonempty if x["us_per_block"] is not None]
                arm_summaries[arm] = {
                    "profile_rows": len(items),
                    "nonempty_block_rows": len(nonempty),
                    "min_blocks": min((x["blocks"] for x in items), default=None),
                    "max_blocks": max((x["blocks"] for x in items), default=None),
                    "online_update_ms_min": min(vals) if vals else None,
                    "online_update_ms_median": statistics.median(vals) if vals else None,
                    "online_update_ms_max": max(vals) if vals else None,
                    "us_per_block_median": statistics.median(us_vals) if us_vals else None,
                    "zero_or_missing_with_blocks": len(zero_or_missing),
                }
            stage_stats[stage] = {
                "present": True,
                "total_profile_rows": sum(len(v) for v in by_arm.values()),
                "by_arm": arm_summaries,
            }

        # Also check minimum block counts across completed evaluate manifests so we know
        # whether any 0-block or 1-block prompt exists in the frozen test population.
        eval_block_floor: dict[str, Any] = {}
        for stage in ("m3", "m4"):
            eval_dir = self.artifacts / stage / "evaluate"
            if not eval_dir.is_dir():
                continue
            floors: dict[str, int] = {}
            zero_blocks = 0
            total = 0
            for mf in sorted(eval_dir.glob("manifest-rank*.json")):
                jobs = self.read_json_tracked(mf).get("jobs", {})
                for entry in jobs.values():
                    if entry.get("status") != "complete":
                        continue
                    rp = Path(entry["result_path"])
                    if not rp.is_file():
                        continue
                    row = self.read_json_tracked(rp)
                    nb = len(row.get("blocks", []))
                    total += 1
                    if nb == 0:
                        zero_blocks += 1
                    floors[row["arm"]] = min(floors.get(row["arm"], 10**9), nb)
            eval_block_floor[stage] = {
                "evaluated_rows": total,
                "zero_block_rows": zero_blocks,
                "min_blocks_by_arm": floors,
            }

        return {
            "validator_rule_lines_293_303": snippet,
            "mechanism": (
                "verify_block() synchronizes the CUDA stream via .tolist()/.item() every round; "
                "dflash_loop._finish_round wraps both online_state.update and bigram.update in "
                "timer.phase('online_update'). Because the CUDA stream is idle after verify_block + "
                "tiny extract_context_feature, CUDA events a.record()/b.record() bracket host CPU "
                "execution of bigram.update (~15-50 us/round) and record strictly positive elapsed ms "
                "whenever blocks > 0 (and validate.py:296 already guards `if row['blocks']`)."
            ),
            "profile_timing_audit": stage_stats,
            "evaluate_block_floors": eval_block_floor,
        }

    def audit_stage(self, stage: str) -> dict[str, Any]:
        stage_dir = self.artifacts / stage
        arms = M3_ARMS if stage == "m3" else M4_ARMS
        settings_path = stage_dir / "settings.json"
        if not settings_path.is_file():
            msg = f"{stage}: settings.json not yet present"
            if self.allow_partial:
                self.warn(msg)
                return {"stage": stage, "status": "not_started"}
            self.check(False, msg)
            return {"stage": stage, "status": "missing"}

        settings = self.read_json_tracked(settings_path)
        self.check(
            settings.get("frozen") is True
            and settings.get("stage") == stage
            and settings.get("scale") == self.scale,
            f"{stage}: settings.json frozen/stage/scale invalid",
        )
        self.check(set(settings.get("selected", {})) == set(arms), f"{stage}: selected arms mismatch")
        frozen_hash = fingerprint(settings)

        # 1. Recompute dev tuning selection from raw tune rows.
        tune_manifest_path = stage_dir / "tune" / "manifest-rank0.json"
        dev_table: list[dict[str, Any]] = []
        if tune_manifest_path.is_file():
            tune_manifest = self.read_json_tracked(tune_manifest_path)
            dev_obs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for key, entry in tune_manifest["jobs"].items():
                if entry.get("status") != "complete":
                    continue
                row_path = Path(entry["result_path"])
                row = self.read_json_tracked(row_path)
                self.check(
                    self.read_sha256[str(row_path.resolve())] == entry.get("result_sha256"),
                    f"{stage}/tune: checksum mismatch on {row_path}",
                )
                if row.get("identity", {}).get("timing_session") == settings["comparison_session"]:
                    dev_obs[(row["arm"], fingerprint(row["settings"]))].append(row)
            plain_dev = {
                r["prompt_id"]: r for r in dev_obs[("plain_dflash", fingerprint({}))]
            }
            for arm in arms:
                cand_scores = []
                for cand_rep in settings["candidates"][arm]:
                    cand_setting = cand_rep["settings"]
                    c_rows = dev_obs[(arm, fingerprint(cand_setting))]
                    by_cond: dict[str, list[float]] = defaultdict(list)
                    excl = 0
                    for r in c_rows:
                        base = plain_dev[r["prompt_id"]]
                        rep_arm = independent_repetition_dominated(r["output_tokens"])
                        rep_base = independent_repetition_dominated(base["output_tokens"])
                        self.check(
                            rep_arm == bool(r["generation"]["repetition_dominated"]),
                            f"{stage}/tune: repetition flag mismatch on {r['prompt_id']} ({arm})",
                        )
                        if r["condition"].startswith("long_") and (rep_arm or rep_base):
                            excl += 1
                            continue
                        by_cond[r["condition"]].append(
                            float(r["tokens_per_second"]) / float(base["tokens_per_second"])
                        )
                    cond_means = {c: statistics.mean(vs) for c, vs in by_cond.items()}
                    score = statistics.mean(cond_means.values())
                    self.check(
                        math.isclose(score, cand_rep["selection_score"], rel_tol=1e-12, abs_tol=1e-12),
                        f"{stage}/tune/{arm}: recomputed selection_score {score} != {cand_rep['selection_score']}",
                    )
                    self.check(
                        excl == cand_rep["repetition_excluded_matched_pairs"],
                        f"{stage}/tune/{arm}: excluded pair count mismatch",
                    )
                    cand_scores.append((score, cand_setting))
                    dev_table.append(
                        {
                            "arm": arm,
                            "settings": cand_setting,
                            "selected": cand_setting == settings["selected"][arm],
                            "selection_score": score,
                            "condition_normalized_throughput": cond_means,
                            "dev_requests": len(c_rows),
                            "repetition_excluded_matched_pairs": excl,
                            "generated_tokens": sum(r["generated_tokens"] for r in c_rows),
                            "wall_seconds": sum(r["elapsed_seconds"] for r in c_rows),
                        }
                    )
                winner = max(cand_scores, key=lambda x: x[0])[1]
                self.check(
                    winner == settings["selected"][arm],
                    f"{stage}/tune/{arm}: selected {settings['selected'][arm]} != argmax winner {winner}",
                )
        else:
            self.check(self.allow_partial, f"{stage}: missing tune/manifest-rank0.json")

        # 2. Recompute closed-loop evaluate metrics, secondary all-rows sensitivity, greedy report, and online state.
        eval_dir = stage_dir / "evaluate"
        eval_rows: list[dict[str, Any]] = []
        if eval_dir.is_dir():
            seen_eval_keys: set[str] = set()
            for mf in sorted(eval_dir.glob("manifest-rank*.json")):
                mdata = self.read_json_tracked(mf)
                for key, entry in mdata.get("jobs", {}).items():
                    if entry.get("status") != "complete":
                        continue
                    ident = entry.get("identity", {})
                    if ident.get("frozen_settings_hash") != frozen_hash:
                        continue
                    self.check(key not in seen_eval_keys, f"{stage}/evaluate: duplicate key {key}")
                    seen_eval_keys.add(key)
                    rp = Path(entry["result_path"])
                    row = self.read_json_tracked(rp)
                    self.check(
                        self.read_sha256[str(rp.resolve())] == entry.get("result_sha256"),
                        f"{stage}/evaluate: checksum mismatch on {rp}",
                    )
                    # Independently verify repetition detector v2 on every long output!
                    if row["condition"].startswith("long_"):
                        rep_recomp = independent_repetition_dominated(row["output_tokens"])
                        self.check(
                            rep_recomp == bool(row["generation"]["repetition_dominated"]),
                            f"{stage}/evaluate: repetition v2 mismatch on {row['prompt_id']} ({row['arm']})",
                        )
                    # Verify greedy effective_beta == 1.0 rule (`prompt.md:831`)
                    if float(row["temperature"]) == 0.0:
                        self.check(
                            float(row["effective_beta"]) == 1.0,
                            f"{stage}/evaluate: greedy effective_beta != 1.0 on {row['prompt_id']}",
                        )
                    # Verify M3->M4 baseline reuse invariants (`prompt.md:827`)
                    reuse = row.get("baseline_reuse", {})
                    if reuse.get("status") == "reused":
                        self.check(
                            stage == "m4" and row["arm"] in ("plain_dflash", "online_bigram"),
                            f"{stage}/evaluate: illegal baseline reuse on {row['arm']}",
                        )
                        for perf_field in (
                            "elapsed_seconds",
                            "generation_loop_elapsed_seconds",
                            "tokens_per_second",
                            "phase_ms",
                            "estimated_online_state_bandwidth_GBps",
                            "peak_allocated_bytes",
                            "peak_reserved_bytes",
                        ):
                            self.check(
                                row.get(perf_field) is None,
                                f"{stage}/evaluate: reused row retained {perf_field}",
                            )
                        src_path = Path(reuse["source_result_path"])
                        self.read_bytes_tracked(src_path)
                        self.check(
                            self.read_sha256[str(src_path.resolve())] == reuse["source_result_sha256"],
                            f"{stage}/evaluate: reused M3 source SHA256 mismatch",
                        )
                    eval_rows.append(row)

        expected_eval = (len(arms) * 8) if self.scale == "smoke" else (len(arms) * 480)
        if eval_rows:
            if not self.allow_partial or len(eval_rows) == expected_eval:
                self.check(
                    len(eval_rows) == expected_eval,
                    f"{stage}/evaluate: expected {expected_eval} rows, got {len(eval_rows)}",
                )

        recomputed_eval_groups = self.recompute_group_rows(
            eval_rows, isolated_timing=False, exclude_repetition=True
        )
        recomputed_eval_all_rows_sensitivity = [
            g
            for g in self.recompute_group_rows(
                eval_rows, isolated_timing=False, exclude_repetition=False
            )
            if g["condition"].startswith("long_")
        ]

        # Annotate sparse long bins (`n_prompts < 8`, especially `16K+`) explicitly (`prompt.md:779`).
        sparse_bin_flags: list[dict[str, Any]] = []
        for g in recomputed_eval_groups:
            if not g["condition"].startswith("long_"):
                continue
            observed_out_bins = {
                b["bin"]: b["n_prompts"] for b in g["acceptance_bins"] if b["axis"] == "output_offset"
            }
            for ob in OUTPUT_LABELS:
                n_p = observed_out_bins.get(ob, 0)
                if n_p < SPARSE_LONG_BIN_MIN_PROMPTS:
                    sparse_bin_flags.append(
                        {
                            "condition": g["condition"],
                            "temperature": g["temperature"],
                            "arm": g["arm"],
                            "axis": "output_offset",
                            "bin": ob,
                            "n_prompts": n_p,
                            "status": "EMPTY" if n_p == 0 else "SPARSE_UNAVAILABLE",
                            "long_context_conclusion_available": False,
                        }
                    )
            for b in sorted(
                (x for x in g["acceptance_bins"] if x["axis"] == "input_context_length"),
                key=lambda x: int(x["bin"].split("-")[0].rstrip("+")),
            ):
                if b["n_prompts"] < SPARSE_LONG_BIN_MIN_PROMPTS:
                    sparse_bin_flags.append(
                        {
                            "condition": g["condition"],
                            "temperature": g["temperature"],
                            "arm": g["arm"],
                            "axis": "input_context_length",
                            "bin": b["bin"],
                            "n_prompts": b["n_prompts"],
                            "status": "SPARSE_UNAVAILABLE",
                            "long_context_conclusion_available": False,
                        }
                    )

        # Greedy comparisons from raw output_tokens (`prompt.md:913`).
        greedy_rows = [
            r
            for r in eval_rows
            if float(r["temperature"]) == 0.0 and r["condition"] == "standard_nonthinking"
        ]
        greedy_by_key = {(r["prompt_id"], r["seed"], r["arm"]): r for r in greedy_rows}
        greedy_summary_groups: list[dict[str, Any]] = []
        if greedy_rows:
            prompts_seeds = sorted({(r["prompt_id"], r["seed"]) for r in greedy_rows})
            for arm in sorted(arms):
                for ref_kind in ("plain_dflash", "target_only"):
                    comps = []
                    for pid, sd in prompts_seeds:
                        row = greedy_by_key.get((pid, sd, arm))
                        if row is None:
                            continue
                        if ref_kind == "plain_dflash":
                            ref_row = greedy_by_key.get((pid, sd, "plain_dflash"))
                            if ref_row is None:
                                continue
                            ref_tokens = ref_row["output_tokens"]
                        else:
                            ident = {
                                k: v
                                for k, v in row["identity"].items()
                                if k not in ("stage", "frozen_settings_hash")
                            }
                            ident.update(
                                mode="greedy_target_only",
                                proposal_mode="target_only",
                                setting={},
                                input_metadata=row["input_metadata"],
                                timing_session=None,
                                repeat=None,
                                phase_profile=False,
                            )
                            t_path = (
                                self.artifacts / "greedy_target_only" / f"{fingerprint(ident)}.json"
                            )
                            if not t_path.is_file():
                                self.check(False, f"{stage}/greedy: missing target_only {t_path}")
                                continue
                            ref_tokens = self.read_json_tracked(t_path)["output_tokens"]
                        comps.append(independent_greedy_mismatch(row["output_tokens"], ref_tokens))
                    tot_tok = sum(c["compared_tokens"] for c in comps)
                    mis_tok = sum(c["mismatched_tokens"] for c in comps)
                    mis_seq = sum(not c["sequence_equal"] for c in comps)
                    if self.scale == "smoke" and ref_kind == "target_only" and arm == "plain_dflash":
                        self.check(mis_seq == 0 and mis_tok == 0, "smoke greedy toy must equal target_only")
                    greedy_summary_groups.append(
                        {
                            "arm": arm,
                            "reference": ref_kind,
                            "compared_requests": len(comps),
                            "mismatched_sequences": mis_seq,
                            "sequence_mismatch_rate": mis_seq / len(comps) if comps else None,
                            "mismatched_tokens": mis_tok,
                            "compared_tokens": tot_tok,
                            "token_mismatch_rate": mis_tok / tot_tok if tot_tok else None,
                            "length_mismatched_sequences": sum(
                                c["length_difference"] != 0 for c in comps
                            ),
                        }
                    )

        # Cross-check against recorded `summary.json` and `greedy_output_comparisons.json` when present.
        summary_path = stage_dir / "summary.json"
        recorded_summary = None
        if summary_path.is_file():
            recorded_summary = self.read_json_tracked(summary_path)
            if self.require_complete:
                self.check(
                    recorded_summary.get("status") == "complete",
                    f"{stage}/summary.json status is {recorded_summary.get('status')!r}, expected 'complete'",
                )
            # Strip our extra `excluded_prompt_ids` diagnostic key and canonicalize multi-rank `online_final_state` order.
            ok, diff = close_nested(
                _canonicalize_group_list(recomputed_eval_groups),
                _canonicalize_group_list(recorded_summary.get("groups", [])),
            )
            self.check(ok, f"{stage}/summary.json groups drift from raw recomputation: {diff}")

            greedy_path = stage_dir / "greedy_output_comparisons.json"
            if greedy_path.is_file():
                rec_greedy = self.read_json_tracked(greedy_path)
                for rg, cg in zip(rec_greedy.get("groups", []), greedy_summary_groups):
                    for k in cg:
                        ok_g, diff_g = close_nested(cg[k], rg.get(k))
                        self.check(ok_g, f"{stage}/greedy group {cg['arm']}->{cg['reference']} .{k}: {diff_g}")

        # 3. Recompute isolated 1-GPU or 8-GPU core-pinned sharded timing + paired ratios + 7+1 phases + 1-GPU calibration.
        timing_path = stage_dir / "timing.json"
        timing_summary: dict[str, Any] = {"status": "not_complete"}
        if timing_path.is_file():
            rec_timing = self.read_json_tracked(timing_path)
            if self.require_complete:
                self.check(
                    rec_timing.get("status") == "complete"
                    and rec_timing.get("full_reference_prompt_coverage") is True,
                    f"{stage}/timing.json incomplete",
                )
            timing_summary = self._audit_recorded_timing(
                stage=stage,
                stage_dir=stage_dir,
                arms=arms,
                settings=settings,
                rec_timing=rec_timing,
            )
            if stage == "m4":
                self.m4_timing_world_size = int(timing_summary.get("world_size", 1))
        else:
            # Partial live timing + calibration inspection when `--allow_partial` is enabled.
            if not self.allow_partial:
                self.check(False, f"{stage}: timing.json missing")
                timing_summary = {"status": "missing_timing_json", "partial_rank_files": 0}
            else:
                timing_summary = self._audit_partial_inflight_timing(
                    stage=stage,
                    stage_dir=stage_dir,
                    arms=arms,
                    settings=settings,
                )
                if stage == "m4":
                    self.m4_timing_world_size = int(timing_summary.get("world_size", 1))

        # 4. M4 Online state, norms, sparsity, active buckets, bandwidth, and M1b feature context.
        online_summary: list[dict[str, Any]] = []
        feature_context_summary: dict[str, Any] | None = None
        if stage == "m4" and eval_rows:
            offline_norms_seen: dict[str, float] | None = None
            for g in recomputed_eval_groups:
                if g["arm"] not in ONLINE_ARMS:
                    continue
                states = g["online_final_state"]
                if not states:
                    continue
                if g["arm"] == "online_dbS":
                    self.check(
                        all(s["dW_norm"] == 0.0 and s["dA_norm"] == 0.0 for s in states),
                        "m4/online_dbS must keep dW_norm==0 and dA_norm==0",
                    )
                online_summary.append(
                    {
                        "condition": g["condition"],
                        "temperature": g["temperature"],
                        "arm": g["arm"],
                        "primary_requests": len(states),
                        "executed_updates_mean": statistics.mean(s["u"] for s in states),
                        "dW_norm_mean": statistics.mean(s["dW_norm"] for s in states),
                        "dA_norm_mean": statistics.mean(s["dA_norm"] for s in states),
                        "db_norm_mean": statistics.mean(s["db_norm"] for s in states),
                        "dS_norm_mean": statistics.mean(s["dS_norm"] for s in states),
                        "db_exact_zero_fraction_mean": statistics.mean(
                            s["db_exact_zero_fraction"] for s in states
                        ),
                        "active_buckets_mean": statistics.mean(s["active_buckets"] for s in states),
                        "active_buckets_max": max(
                            s["active_buckets_max_over_updates"] for s in states
                        ),
                        "sparse_capacity": states[0]["sparse_capacity"],
                    }
                )
            # Check constant offline parameter norms across all online rows + match against m2.json if reference.
            for r in eval_rows:
                if r["arm"] in ONLINE_ARMS:
                    norms = r["parameter_norms"]
                    if offline_norms_seen is None:
                        offline_norms_seen = norms
                    else:
                        self.check(
                            norms == offline_norms_seen,
                            "m4: frozen offline parameter_norms changed across online requests",
                        )
            if self.scale == "reference" and offline_norms_seen is not None:
                m2_path = self.artifacts / "m2.json"
                if m2_path.is_file():
                    m2 = self.read_json_tracked(m2_path)
                    m2_params = m2["stages"]["dense_sparse"]["parameters"]
                    for pname in ("W", "A", "b"):
                        self.check(
                            math.isclose(
                                offline_norms_seen[pname],
                                m2_params[pname]["norm"],
                                rel_tol=1e-6,
                                abs_tol=1e-6,
                            ),
                            f"m4 offline {pname} norm != m2.json dense_sparse norm",
                        )

            # Bandwidth from timing profile pass (where `online_update` CUDA events were recorded).
            if timing_summary.get("groups") and timing_path.is_file():
                rec_timing = self.read_json_tracked(timing_path)
                bw_by_group = {}
                for pg in rec_timing.get("phase_profiles", []):
                    if pg["arm"] in ONLINE_ARMS:
                        bw = pg["metrics"].get("estimated_online_state_bandwidth_GBps")
                        bw_by_group[(pg["condition"], pg["temperature"], pg["arm"])] = bw
                for item in online_summary:
                    item["profile_estimated_state_bandwidth_GBps"] = bw_by_group.get(
                        (item["condition"], item["temperature"], item["arm"])
                    )

            fc_path = stage_dir / "feature_context.json"
            if fc_path.is_file():
                fc = self.read_json_tracked(fc_path)
                self.check(
                    fc.get("status") == "complete" and fc.get("feature_group_count", 0) > 0,
                    "m4/feature_context.json incomplete",
                )
                if self.scale == "reference":
                    for pkey, skey in (
                        ("diagnostic_path", "diagnostic_sha256"),
                        ("audit_path", "audit_sha256"),
                        ("calibration_path", "calibration_sha256"),
                    ):
                        target_file = Path(fc[pkey])
                        self.read_bytes_tracked(target_file)
                        self.check(
                            self.read_sha256[str(target_file.resolve())] == fc[skey],
                            f"m4/feature_context {pkey} SHA256 mismatch",
                        )
                feature_context_summary = {
                    "status": fc.get("status"),
                    "source": fc.get("source"),
                    "interpretation": fc.get("interpretation"),
                    "feature_group_count": fc.get("feature_group_count"),
                    "condition_bin_coverage": fc.get("condition_bin_coverage"),
                    "all_offsets_primary_groups": [
                        {
                            "condition": grp["condition"],
                            "temperature": grp["collection_temperature"],
                            "bin": grp["bin"],
                            "logical_slots": grp["logical_slots"],
                            "coordinate_mean_norm": grp["features"]["coordinate_mean_norm"],
                            "variance_ratio": grp["features"][
                                "variance_ratio_to_standard_calibration"
                            ],
                            "l2_median": grp["features"]["l2_norm_quantiles"]["p50"],
                            "l2_p95": grp["features"]["l2_norm_quantiles"]["p95"],
                            "outlier_slot_rate": grp["features"]["outlier_slot_rate"],
                        }
                        for grp in fc.get("groups", [])
                        if grp.get("inclusion") == "primary"
                    ],
                }
            elif not self.allow_partial:
                self.check(False, "m4/feature_context.json missing")

        return {
            "stage": stage,
            "frozen_settings_hash": frozen_hash,
            "selected_settings": settings["selected"],
            "dev_selection_table": dev_table,
            "evaluated_rows": len(eval_rows),
            "expected_evaluated_rows": expected_eval,
            "test_acceptance_groups": recomputed_eval_groups,
            "long_all_rows_sensitivity": recomputed_eval_all_rows_sensitivity,
            "sparse_long_bins": sparse_bin_flags,
            "greedy_comparisons": greedy_summary_groups,
            "timing": timing_summary,
            "online_state_summary": online_summary,
            "feature_context_summary": feature_context_summary,
        }

    @staticmethod
    def _extract_rank_prompt_map(plan: dict[str, Any]) -> dict[int, list[str]]:
        """Normalize sharding_plan.json rank->prompt_ids across list/dict/per-condition layouts."""
        rank_map: dict[int, list[str]] = defaultdict(list)

        def _ingest_entry(rk: Any, val: Any) -> None:
            r_int = int(rk)
            if isinstance(val, list):
                for item in val:
                    pid = item["prompt_id"] if isinstance(item, dict) else str(item)
                    if pid not in rank_map[r_int]:
                        rank_map[r_int].append(pid)
            elif isinstance(val, dict):
                pids = val.get("prompt_ids") or val.get("prompts") or []
                for item in pids:
                    pid = item["prompt_id"] if isinstance(item, dict) else str(item)
                    if pid not in rank_map[r_int]:
                        rank_map[r_int].append(pid)
                for sub_v in val.values():
                    if isinstance(sub_v, dict):
                        for sub_k, sub_list in sub_v.items():
                            if str(sub_k).isdigit() and isinstance(sub_list, list):
                                _ingest_entry(int(sub_k), sub_list)

        for key in ("ranks", "shards", "assignments", "by_rank"):
            container = plan.get(key)
            if isinstance(container, dict):
                for rk, val in container.items():
                    if str(rk).isdigit():
                        _ingest_entry(int(rk), val)
            elif isinstance(container, list):
                for idx, entry in enumerate(container):
                    if isinstance(entry, dict):
                        rk = entry.get("rank", idx)
                        _ingest_entry(rk, entry)
                    elif isinstance(entry, list):
                        _ingest_entry(idx, entry)
            if rank_map:
                return dict(rank_map)

        by_cond = plan.get("by_condition") or plan.get("conditions")
        if isinstance(by_cond, dict):
            for cond_val in by_cond.values():
                if isinstance(cond_val, dict):
                    r_sub = cond_val.get("ranks") if isinstance(cond_val.get("ranks"), (dict, list)) else cond_val
                    if isinstance(r_sub, dict):
                        for rk, val in r_sub.items():
                            if str(rk).isdigit():
                                _ingest_entry(int(rk), val)
                    elif isinstance(r_sub, list):
                        for idx, entry in enumerate(r_sub):
                            rk = entry.get("rank", idx) if isinstance(entry, dict) else idx
                            _ingest_entry(rk, entry)
        return dict(rank_map)

    @staticmethod
    def _extract_affinity_info(obj: dict[str, Any]) -> tuple[int | None, int | None, list[int] | None]:
        """Extract (physical_gpu, numa_node, cpu_affinity) from a row or session payload."""
        containers = [obj]
        for sub_key in ("affinity", "hardware", "configuration", "execution", "isolation"):
            sub = obj.get(sub_key)
            if isinstance(sub, dict):
                containers.append(sub)
                inner = sub.get("configuration")
                if isinstance(inner, dict):
                    containers.append(inner)
        phys_gpu: int | None = None
        numa_node: int | None = None
        cpu_aff: list[int] | None = None
        for c in containers:
            if phys_gpu is None and c.get("physical_gpu") is not None:
                phys_gpu = int(c["physical_gpu"])
            if numa_node is None and c.get("numa_node") is not None:
                numa_node = int(c["numa_node"])
            if cpu_aff is None and c.get("cpu_affinity") is not None:
                cpu_aff = [int(x) for x in c["cpu_affinity"]]
        return phys_gpu, numa_node, cpu_aff

    def _resolve_evidence_file_list(
        self,
        raw_ev: dict[str, Any],
        *,
        single_path_key: str,
        single_sha_key: str,
        plural_keys: tuple[str, ...],
        plural_sha_key: str,
        fallback_glob_dir: Path,
        fallback_pattern: str,
        label: str,
    ) -> list[Path]:
        paths_and_shas: list[tuple[Path, str | None]] = []
        for pk in plural_keys:
            val = raw_ev.get(pk)
            if not val:
                continue
            sha_container = raw_ev.get(plural_sha_key)
            if isinstance(val, list):
                for idx, item in enumerate(val):
                    if isinstance(item, dict):
                        p = Path(item.get("path") or item.get("manifest_path") or item.get("session_path"))
                        s = item.get("sha256")
                    else:
                        p = Path(item)
                        s = (
                            sha_container[idx]
                            if isinstance(sha_container, list) and idx < len(sha_container)
                            else (sha_container.get(str(p)) or sha_container.get(str(idx)))
                            if isinstance(sha_container, dict)
                            else None
                        )
                    paths_and_shas.append((p, s))
            elif isinstance(val, dict):
                for rk in sorted(val.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                    item = val[rk]
                    if isinstance(item, dict):
                        p = Path(item.get("path") or item.get("manifest_path") or item.get("session_path"))
                        s = item.get("sha256")
                    else:
                        p = Path(item)
                        s = sha_container.get(rk) if isinstance(sha_container, dict) else None
                    paths_and_shas.append((p, s))
            if paths_and_shas:
                break

        if not paths_and_shas and raw_ev.get(single_path_key):
            paths_and_shas.append((Path(raw_ev[single_path_key]), raw_ev.get(single_sha_key)))

        if not paths_and_shas and fallback_glob_dir.is_dir():
            for p in sorted(fallback_glob_dir.glob(fallback_pattern)):
                paths_and_shas.append((p, None))

        resolved: list[Path] = []
        for p, expected_sha in paths_and_shas:
            self.check(p.is_file(), f"{label}: missing raw evidence file {p}")
            if p.is_file():
                self.read_bytes_tracked(p)
                if expected_sha:
                    self.check(
                        self.read_sha256[str(p.resolve())] == expected_sha,
                        f"{label}: SHA256 mismatch on {p}",
                    )
                resolved.append(p)
        return resolved

    def _audit_recorded_timing(
        self,
        *,
        stage: str,
        stage_dir: Path,
        arms: tuple[str, ...],
        settings: dict[str, Any],
        rec_timing: dict[str, Any],
    ) -> dict[str, Any]:
        raw_ev = rec_timing.get("raw_evidence", {})
        manifest_paths = self._resolve_evidence_file_list(
            raw_ev,
            single_path_key="manifest_path",
            single_sha_key="manifest_sha256",
            plural_keys=("manifest_paths", "manifests", "rank_manifests"),
            plural_sha_key="manifest_sha256s",
            fallback_glob_dir=stage_dir / "timing",
            fallback_pattern="manifest-rank*.json",
            label=f"{stage}/timing/manifests",
        )
        session_paths = self._resolve_evidence_file_list(
            raw_ev,
            single_path_key="session_path",
            single_sha_key="session_sha256",
            plural_keys=("session_paths", "sessions", "rank_sessions"),
            plural_sha_key="session_sha256s",
            fallback_glob_dir=stage_dir / "timing" / "sessions",
            fallback_pattern="*.json",
            label=f"{stage}/timing/sessions",
        )

        merged_jobs: dict[str, dict[str, Any]] = {}
        for mp in manifest_paths:
            mdata = self.read_json_tracked(mp)
            for key, entry in mdata.get("jobs", {}).items():
                self.check(
                    key not in merged_jobs,
                    f"{stage}/timing: duplicate job key {key} across timing rank manifests",
                )
                merged_jobs[key] = entry

        world_size = int(
            rec_timing.get("world_size")
            or raw_ev.get("world_size")
            or max(1, len(session_paths))
        )
        sharding_mode = str(
            rec_timing.get("sharding_mode")
            or ("sharded_8gpu" if world_size > 1 else "single_gpu")
        )
        is_sharded = world_size > 1 or sharding_mode not in ("single_gpu", "unsharded", "none")
        if is_sharded:
            self.check(
                len(manifest_paths) == world_size and len(session_paths) == world_size,
                f"{stage}/timing: sharded world_size={world_size} mismatch vs manifests={len(manifest_paths)}, sessions={len(session_paths)}",
            )

        # Sharding plan verification when world_size > 1 or sharding_plan.json exists.
        plan_path_raw = raw_ev.get("sharding_plan_path") or rec_timing.get("sharding_plan_path")
        plan_path = Path(plan_path_raw) if plan_path_raw else (stage_dir / "timing" / "sharding_plan.json")
        rank_prompt_sets: dict[int, set[str]] = {}
        sharding_plan_summary: dict[str, Any] | None = None
        if is_sharded or plan_path.is_file():
            self.check(plan_path.is_file(), f"{stage}/timing: missing sharding_plan.json at {plan_path}")
            if plan_path.is_file():
                plan_data = self.read_json_tracked(plan_path)
                exp_plan_sha = raw_ev.get("sharding_plan_sha256") or rec_timing.get("sharding_plan_sha256")
                if exp_plan_sha:
                    self.check(
                        self.read_sha256[str(plan_path.resolve())] == exp_plan_sha,
                        f"{stage}/timing: sharding_plan.json SHA256 mismatch",
                    )
                rank_lists = self._extract_rank_prompt_map(plan_data)
                self.check(
                    set(rank_lists.keys()) == set(range(world_size)),
                    f"{stage}/timing: sharding_plan ranks {sorted(rank_lists.keys())} != 0..{world_size - 1}",
                )
                all_assigned: list[str] = []
                for rk, pids in sorted(rank_lists.items()):
                    self.check(len(pids) > 0, f"{stage}/timing: rank {rk} assigned 0 prompts in sharding_plan")
                    rank_prompt_sets[rk] = set(pids)
                    all_assigned.extend(pids)
                self.check(
                    len(all_assigned) == len(set(all_assigned)),
                    f"{stage}/timing: duplicate/overlapping prompt assignments across ranks in sharding_plan.json",
                )
                if rec_timing.get("prompt_ids"):
                    self.check(
                        set(all_assigned) == set(rec_timing["prompt_ids"]),
                        f"{stage}/timing: sharding_plan union ({len(set(all_assigned))}) != timing.json prompt_ids ({len(set(rec_timing['prompt_ids']))})",
                    )
                sharding_plan_summary = {
                    "path": str(plan_path),
                    "world_size": world_size,
                    "prompts_per_rank": {str(rk): len(pids) for rk, pids in sorted(rank_lists.items())},
                    "total_unique_prompts": len(set(all_assigned)),
                }

        # Honest provenance split check on sharded runs (`10/10` ALGORITHM_FILES + `timing_harness_implementation_sha256`).
        expected_harness_sha = rec_timing.get("timing_harness_implementation_sha256")
        if is_sharded:
            self.check(
                isinstance(expected_harness_sha, str) and len(expected_harness_sha) == 64,
                f"{stage}/timing: sharded timing.json missing valid 64-hex timing_harness_implementation_sha256",
            )
            alg_sha_09a, missing_alg = self._compute_09a5469_alg_sha256()
            self.check(not missing_alg, f"{stage}/timing: missing ALGORITHM_FILES in 09a5469 source: {missing_alg}")
            if alg_sha_09a is not None and rec_timing.get("algorithm_implementation_sha256"):
                self.check(
                    rec_timing["algorithm_implementation_sha256"] == alg_sha_09a,
                    f"{stage}/timing: algorithm_implementation_sha256 drifted from 09a5469 ALGORITHM_FILES",
                )
            state_path = self.artifacts / "reference_continuation_state.json"
            if not state_path.is_file():
                state_path = self.repo / "artifacts/reference_continuation_state.json"
            if state_path.is_file() and self.scale == "reference":
                fb = json.loads(state_path.read_text(encoding="utf-8")).get("frozen_artifact_binding", {})
                for fn in ALGORITHM_FILES:
                    matching_vals = [
                        v for k, v in fb.items() if Path(k).name == fn and Path(k).parent.name == "edraft"
                    ]
                    blob = self._read_09a5469_repo_file(f"src/edraft/{fn}")
                    if matching_vals and blob is not None:
                        actual_alg_sha = sha256_bytes(blob)
                        self.check(
                            actual_alg_sha == matching_vals[0],
                            f"{stage}/timing: core algorithm file {fn} drifted from frozen 09a5469 SHA256",
                        )

        timed_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        seen_bundles: set[str] = set()
        rank_affinity_summary: dict[int, dict[str, Any]] = {}

        for s_idx, sp in enumerate(session_paths):
            sess = self.read_json_tracked(sp)
            # Infer rank index from session payload or `-rank{r}.json` filename.
            m_rank = re.search(r"rank(\d+)\.json$", sp.name)
            rank_id = int(
                sess.get("rank")
                if sess.get("rank") is not None
                else (m_rank.group(1) if m_rank else s_idx)
            )
            s_gpu, s_numa, s_cpus = self._extract_affinity_info(sess)
            row_gpus: set[int] = set()
            row_numas: set[int] = set()
            row_cpus_set: set[tuple[int, ...]] = set()

            for b_key, b_val in sess.get("bundles", {}).items():
                if b_val.get("status") != "complete":
                    continue
                self.check(
                    b_key not in seen_bundles,
                    f"{stage}/timing: duplicate bundle key {b_key} across rank sessions",
                )
                seen_bundles.add(b_key)
                b_paths = b_val.get("result_paths", [])
                self.check(
                    len(b_paths) == len(arms),
                    f"{stage}/timing: bundle {b_key} on rank {rank_id} has {len(b_paths)} paths, expected {len(arms)} arms",
                )
                b_rows: list[dict[str, Any]] = []
                for p_str in b_paths:
                    rp = Path(p_str)
                    row = self.read_json_tracked(rp)
                    key = fingerprint(row["identity"])
                    self.check(
                        key in merged_jobs
                        and merged_jobs[key]["result_sha256"] == self.read_sha256[str(rp.resolve())],
                        f"{stage}/timing: checksum or manifest entry mismatch on {rp}",
                    )
                    if rank_id in rank_prompt_sets:
                        self.check(
                            row["prompt_id"] in rank_prompt_sets[rank_id],
                            f"{stage}/timing: row prompt {row['prompt_id']} in rank {rank_id} violates sharding_plan.json",
                        )
                    if is_sharded:
                        row_h_sha = row.get("timing_harness_implementation_sha256") or row.get(
                            "checkpoint_hashes", {}
                        ).get("timing_harness_implementation")
                        self.check(
                            row_h_sha == expected_harness_sha,
                            f"{stage}/timing: row {rp.name} timing_harness_implementation_sha256 ({row_h_sha}) != timing.json ({expected_harness_sha})",
                        )
                        self.check(
                            row.get("checkpoint_hashes", {}).get("implementation")
                            == settings.get("checkpoint_hashes", {}).get("implementation"),
                            f"{stage}/timing: row {rp.name} checkpoint_hashes.implementation drifted from frozen settings.json",
                        )
                    r_gpu, r_numa, r_cpus = self._extract_affinity_info(row)
                    if r_gpu is not None:
                        row_gpus.add(r_gpu)
                    if r_numa is not None:
                        row_numas.add(r_numa)
                    if r_cpus is not None:
                        row_cpus_set.add(tuple(sorted(r_cpus)))
                    b_rows.append(row)
                    (profile_rows if row.get("phase_profile") else timed_rows).append(row)

                if b_rows:
                    self.check(
                        {r["arm"] for r in b_rows} == set(arms),
                        f"{stage}/timing: bundle {b_key} arms {sorted({r['arm'] for r in b_rows})} != expected {sorted(arms)}",
                    )
                    slot_sigs = {
                        (
                            r["prompt_id"],
                            r["condition"],
                            r["sampling_name"],
                            r["seed"],
                            str(r.get("repeat", "")).split(":epoch=")[0],
                            bool(r.get("phase_profile")),
                        )
                        for r in b_rows
                    }
                    self.check(
                        len(slot_sigs) == 1,
                        f"{stage}/timing: bundle {b_key} mixes mismatched prompt/seed/repeat/profile slots: {slot_sigs}",
                    )

            eff_gpu = s_gpu if s_gpu is not None else (next(iter(row_gpus)) if len(row_gpus) == 1 else None)
            eff_numa = s_numa if s_numa is not None else (next(iter(row_numas)) if len(row_numas) == 1 else None)
            eff_cpus = (
                sorted(s_cpus)
                if s_cpus is not None
                else (list(next(iter(row_cpus_set))) if len(row_cpus_set) == 1 else None)
            )
            if is_sharded:
                self.check(
                    len(row_gpus) <= 1 and len(row_cpus_set) <= 1,
                    f"{stage}/timing: rank {rank_id} rows have inconsistent GPU/CPU affinity metadata",
                )
                self.check(
                    eff_gpu is not None and eff_numa is not None and bool(eff_cpus),
                    f"{stage}/timing: rank {rank_id} missing physical_gpu/numa_node/cpu_affinity metadata",
                )
                if eff_gpu is not None and eff_cpus is not None and len(eff_cpus) == 16:
                    self.check(
                        eff_cpus == list(range(16 * eff_gpu, 16 * eff_gpu + 16))
                        and eff_numa == (eff_gpu // 2),
                        f"{stage}/timing: rank {rank_id} GPU {eff_gpu} affinity {eff_cpus} (numa={eff_numa}) violates 16-core NUMA map [16g..16g+15], g//2",
                    )
            if eff_gpu is not None or eff_cpus is not None:
                rank_affinity_summary[rank_id] = {
                    "physical_gpu": eff_gpu,
                    "numa_node": eff_numa,
                    "cpu_affinity": eff_cpus,
                }

        if is_sharded and rank_affinity_summary:
            gpus_seen = [v["physical_gpu"] for v in rank_affinity_summary.values() if v["physical_gpu"] is not None]
            self.check(
                len(gpus_seen) == len(set(gpus_seen)) == world_size,
                f"{stage}/timing: physical_gpu assignments across ranks are not distinct: {gpus_seen}",
            )
            union_cpus: set[int] = set()
            total_cpu_count = 0
            for rk, info in sorted(rank_affinity_summary.items()):
                cpus = info.get("cpu_affinity") or []
                overlap = union_cpus.intersection(cpus)
                self.check(
                    not overlap,
                    f"{stage}/timing: overlapping cpu_affinity cores {sorted(overlap)} on rank {rk}",
                )
                union_cpus.update(cpus)
                total_cpu_count += len(cpus)
            self.check(
                len(union_cpus) == total_cpu_count,
                f"{stage}/timing: cpu_affinity sets across ranks are not pairwise disjoint",
            )

        recomp_t_groups = self.recompute_group_rows(
            timed_rows, isolated_timing=True, exclude_repetition=True
        )
        recomp_p_groups = self.recompute_group_rows(
            profile_rows, isolated_timing=True, exclude_repetition=True
        )

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
                    f"{stage}/timing: missing matched plain_dflash row for {b_key_lookup}",
                )
                if b_key_lookup not in paired_plain:
                    continue
                b_row = paired_plain[b_key_lookup]
                if r["condition"].startswith("long_") and (
                    r["generation"]["repetition_dominated"]
                    or b_row["generation"]["repetition_dominated"]
                ):
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

        ok_t, diff_t = close_nested(
            _canonicalize_group_list(recomp_t_groups),
            _canonicalize_group_list(rec_timing.get("groups", [])),
        )
        self.check(ok_t, f"{stage}/timing.json groups drift from raw recomputation: {diff_t}")
        ok_p, diff_p = close_nested(
            _canonicalize_group_list(recomp_p_groups),
            _canonicalize_group_list(rec_timing.get("phase_profiles", [])),
        )
        self.check(
            ok_p, f"{stage}/timing.json phase_profiles drift from raw recomputation: {diff_p}"
        )

        phase_table = []
        for pg in recomp_p_groups:
            med = pg["phase_ms_median"]
            phase_table.append(
                {
                    "condition": pg["condition"],
                    "temperature": pg["temperature"],
                    "arm": pg["arm"],
                    "primary_requests": pg["primary_requests"],
                    "section11_phases_ms_median": {
                        ph: med.get(ph, 0.0) for ph in SECTION11_REQUIRED_PHASES
                    },
                    "lookahead_iterations_ms": 0.0,
                    "lookahead_status": "0.0 ms [disabled in M0-M4]",
                    "auxiliary_subphases_ms_median": {
                        ph: med[ph] for ph in AUX_INSTRUMENTED_PHASES if ph in med
                    },
                }
            )

        isolation_calibration = None
        if stage == "m4":
            isolation_calibration = self._audit_timing_isolation_calibration(
                stage_dir=stage_dir,
                arms=arms,
                sharded_timed_rows=timed_rows,
                is_sharded=is_sharded,
            )

        return {
            "status": rec_timing.get("status"),
            "world_size": world_size,
            "sharding_mode": sharding_mode,
            "is_sharded": is_sharded,
            "timing_harness_implementation_sha256": expected_harness_sha,
            "sharding_plan": sharding_plan_summary,
            "rank_affinity": {str(k): v for k, v in sorted(rank_affinity_summary.items())},
            "timed_requests": len(timed_rows),
            "profile_requests": len(profile_rows),
            "bundles": len(seen_bundles),
            "groups": recomp_t_groups,
            "phase_table": phase_table,
            "isolation_calibration": isolation_calibration,
        }

    def _audit_partial_inflight_timing(
        self,
        *,
        stage: str,
        stage_dir: Path,
        arms: tuple[str, ...],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Audit live in-progress sharded `m4/timing` + `timing_isolation_calibration.json` before `timing.json` exists."""
        timing_dir = stage_dir / "timing"
        rank_dirs = sorted(d for d in timing_dir.glob("rank*") if d.is_dir()) if timing_dir.is_dir() else []
        plan_path = timing_dir / "sharding_plan.json"
        is_sharded = plan_path.is_file() or len(rank_dirs) > 1
        world_size = 8 if (is_sharded and self.scale == "reference") else max(1, len(rank_dirs))

        rank_prompt_sets: dict[int, set[str]] = {}
        sharding_plan_summary: dict[str, Any] | None = None
        if plan_path.is_file():
            plan_data = self.read_json_tracked(plan_path)
            world_size = int(plan_data.get("world_size", world_size))
            rank_lists = self._extract_rank_prompt_map(plan_data)
            self.check(
                set(rank_lists.keys()) == set(range(world_size)),
                f"{stage}/timing(partial): sharding_plan ranks {sorted(rank_lists.keys())} != 0..{world_size - 1}",
            )
            all_assigned: list[str] = []
            for rk, pids in sorted(rank_lists.items()):
                self.check(len(pids) > 0, f"{stage}/timing(partial): rank {rk} assigned 0 prompts")
                rank_prompt_sets[rk] = set(pids)
                all_assigned.extend(pids)
            self.check(
                len(all_assigned) == len(set(all_assigned)),
                f"{stage}/timing(partial): overlapping prompt assignments in sharding_plan.json",
            )
            sharding_plan_summary = {
                "path": str(plan_path),
                "world_size": world_size,
                "prompts_per_rank": {str(rk): len(pids) for rk, pids in sorted(rank_lists.items())},
                "total_unique_prompts": len(set(all_assigned)),
            }

        alg_sha_09a, missing_alg = self._compute_09a5469_alg_sha256()
        self.check(not missing_alg, f"{stage}/timing(partial): missing ALGORITHM_FILES in 09a5469 source: {missing_alg}")

        # Load online_checks dynamically from repo validator if stage == 'm4'
        online_checks_mod = None
        if stage == "m4":
            oc_path = self.repo / "artifacts/m4_evidence_validator/online_checks.py"
            if oc_path.is_file():
                import importlib.util
                spec = importlib.util.spec_from_file_location("_edraft_online_checks_live", oc_path)
                if spec and spec.loader:
                    online_checks_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(online_checks_mod)

        timed_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        rank_affinity_summary: dict[int, dict[str, Any]] = {}
        seen_harness_shas: set[str] = set()
        online_row_checks = 0
        online_rows_verified = 0

        for r_dir in rank_dirs:
            m_rk = re.search(r"rank(\d+)$", r_dir.name)
            rk_id = int(m_rk.group(1)) if m_rk else 0
            row_gpus: set[int] = set()
            row_numas: set[int] = set()
            row_cpus_set: set[tuple[int, ...]] = set()
            for rp in sorted(r_dir.glob("*.json")):
                try:
                    row = self.read_json_tracked(rp)
                except Exception:
                    continue
                self.check(row.get("rank") == rk_id, f"{stage}/timing(partial): row {rp.name} rank={row.get('rank')} != {rk_id}")
                if rk_id in rank_prompt_sets:
                    self.check(
                        row.get("prompt_id") in rank_prompt_sets[rk_id],
                        f"{stage}/timing(partial): row {rp.name} prompt {row.get('prompt_id')} violates sharding_plan rank {rk_id}",
                    )
                h_sha = row.get("timing_harness_implementation_sha256")
                if h_sha is not None:
                    seen_harness_shas.add(str(h_sha))
                if alg_sha_09a is not None and row.get("algorithm_implementation_sha256") is not None:
                    self.check(
                        row.get("algorithm_implementation_sha256") == alg_sha_09a,
                        f"{stage}/timing(partial): row {rp.name} algorithm_implementation_sha256 drifted from 09a5469",
                    )
                r_gpu, r_numa, r_cpus = self._extract_affinity_info(row)
                if r_gpu is not None:
                    row_gpus.add(r_gpu)
                if r_numa is not None:
                    row_numas.add(r_numa)
                if r_cpus is not None:
                    row_cpus_set.add(tuple(sorted(r_cpus)))
                if online_checks_mod is not None and row.get("arm") in ONLINE_ARMS:
                    oc_res = online_checks_mod.check_online_row(
                        row,
                        expected_scale=self.scale,
                        expected_arm=row["arm"],
                        expected_setting=settings["selected"][row["arm"]],
                        profile=bool(row.get("phase_profile")),
                    )
                    online_row_checks += int(oc_res["checks"])
                    online_rows_verified += 1
                    self.check(
                        not oc_res["failures"],
                        f"{stage}/timing(partial): online_checks failure on {rp.name}: {oc_res['failures']}",
                    )
                if row.get("phase_profile"):
                    profile_rows.append(row)
                else:
                    self.check(not row.get("phase_ms"), f"{stage}/timing(partial): timed row {rp.name} has non-empty phase_ms")
                    timed_rows.append(row)

            if row_gpus or row_cpus_set:
                self.check(
                    len(row_gpus) == 1 and len(row_cpus_set) == 1,
                    f"{stage}/timing(partial): rank {rk_id} inconsistent GPU/CPU affinity across rows",
                )
                eff_gpu = next(iter(row_gpus)) if len(row_gpus) == 1 else None
                eff_numa = next(iter(row_numas)) if len(row_numas) == 1 else None
                eff_cpus = list(next(iter(row_cpus_set))) if len(row_cpus_set) == 1 else None
                if eff_gpu is not None and eff_cpus is not None and len(eff_cpus) == 16:
                    self.check(
                        eff_cpus == list(range(16 * eff_gpu, 16 * eff_gpu + 16)) and eff_numa == (eff_gpu // 2),
                        f"{stage}/timing(partial): rank {rk_id} affinity {eff_cpus} violates [16g..16g+15], g//2",
                    )
                rank_affinity_summary[rk_id] = {
                    "physical_gpu": eff_gpu,
                    "numa_node": eff_numa,
                    "cpu_affinity": eff_cpus,
                }

        if is_sharded and seen_harness_shas:
            self.check(
                len(seen_harness_shas) == 1 and all(len(s) == 64 for s in seen_harness_shas),
                f"{stage}/timing(partial): inconsistent timing_harness_implementation_sha256 across rows: {seen_harness_shas}",
            )
        if is_sharded and len(rank_affinity_summary) > 1:
            gpus_seen = [v["physical_gpu"] for v in rank_affinity_summary.values() if v["physical_gpu"] is not None]
            self.check(len(gpus_seen) == len(set(gpus_seen)), f"{stage}/timing(partial): duplicate physical_gpu across ranks: {gpus_seen}")
            union_cpus: set[int] = set()
            tot_cpus = 0
            for rk, info in sorted(rank_affinity_summary.items()):
                cpus = info.get("cpu_affinity") or []
                self.check(not union_cpus.intersection(cpus), f"{stage}/timing(partial): overlapping CPU cores on rank {rk}")
                union_cpus.update(cpus)
                tot_cpus += len(cpus)
            self.check(len(union_cpus) == tot_cpus, f"{stage}/timing(partial): non-disjoint CPU affinity sets")

        n_partial = len(timed_rows) + len(profile_rows)
        self.warn(
            f"{stage}: timing.json not yet written (live timing rows audited={n_partial}: timed={len(timed_rows)}, profile={len(profile_rows)}, online_rows={online_rows_verified})"
        )

        isolation_calibration = None
        if stage == "m4":
            isolation_calibration = self._audit_timing_isolation_calibration(
                stage_dir=stage_dir,
                arms=arms,
                sharded_timed_rows=timed_rows,
                is_sharded=is_sharded,
                allow_partial_8gpu=True,
            )

        return {
            "status": "in_progress_partial",
            "world_size": world_size,
            "sharding_mode": f"sharded_{world_size}gpu" if is_sharded else "single_gpu",
            "is_sharded": is_sharded,
            "timing_harness_implementation_sha256": next(iter(seen_harness_shas)) if len(seen_harness_shas) == 1 else None,
            "sharding_plan": sharding_plan_summary,
            "rank_affinity": {str(k): v for k, v in sorted(rank_affinity_summary.items())},
            "partial_rank_files": n_partial,
            "timed_requests": len(timed_rows),
            "profile_requests": len(profile_rows),
            "online_rows_verified": online_rows_verified,
            "online_row_checks": online_row_checks,
            "isolation_calibration": isolation_calibration,
        }

    def _audit_timing_isolation_calibration(
        self,
        *,
        stage_dir: Path,
        arms: tuple[str, ...],
        sharded_timed_rows: list[dict[str, Any]],
        is_sharded: bool,
        allow_partial_8gpu: bool = False,
    ) -> dict[str, Any] | None:
        cal_path = stage_dir / "timing_isolation_calibration.json"
        if not cal_path.is_file():
            if is_sharded and self.require_complete:
                self.check(
                    False,
                    "m4/timing_isolation_calibration.json is required when M4 timing uses multi-GPU sharding",
                )
            elif is_sharded and self.allow_partial:
                self.warn("m4/timing_isolation_calibration.json not yet written")
            return None

        cal = self.read_json_tracked(cal_path)
        if self.require_complete:
            self.check(
                cal.get("status") == "complete",
                f"m4/timing_isolation_calibration.json status={cal.get('status')!r}, expected 'complete'",
            )

        raw_ev = cal.get("raw_evidence", {})
        cal_dir = stage_dir / "timing_calibration"
        cal_manifests = self._resolve_evidence_file_list(
            raw_ev,
            single_path_key="manifest_path",
            single_sha_key="manifest_sha256",
            plural_keys=("manifest_paths", "manifests"),
            plural_sha_key="manifest_sha256s",
            fallback_glob_dir=cal_dir,
            fallback_pattern="manifest-rank*.json",
            label="m4/timing_calibration/manifest",
        )
        cal_sessions = self._resolve_evidence_file_list(
            raw_ev,
            single_path_key="session_path",
            single_sha_key="session_sha256",
            plural_keys=("session_paths", "sessions"),
            plural_sha_key="session_sha256s",
            fallback_glob_dir=cal_dir / "sessions",
            fallback_pattern="*.json",
            label="m4/timing_calibration/session",
        )

        cal_jobs: dict[str, dict[str, Any]] = {}
        for mp in cal_manifests:
            mdata = self.read_json_tracked(mp)
            cal_jobs.update(mdata.get("jobs", {}))

        cal_timed_rows: list[dict[str, Any]] = []
        if cal_sessions:
            for sp in cal_sessions:
                sess = self.read_json_tracked(sp)
                for b_key, b_val in sess.get("bundles", {}).items():
                    if b_val.get("status") != "complete":
                        continue
                    b_paths = b_val.get("result_paths", [])
                    self.check(
                        len(b_paths) == len(arms),
                        f"m4/timing_calibration: bundle {b_key} has {len(b_paths)} paths, expected {len(arms)}",
                    )
                    for p_str in b_paths:
                        rp = Path(p_str)
                        row = self.read_json_tracked(rp)
                        key = fingerprint(row["identity"])
                        if cal_jobs:
                            self.check(
                                key in cal_jobs
                                and cal_jobs[key]["result_sha256"]
                                == self.read_sha256[str(rp.resolve())],
                                f"m4/timing_calibration: checksum mismatch on {rp}",
                            )
                        if not row.get("phase_profile"):
                            cal_timed_rows.append(row)
        elif cal_jobs:
            for key, entry in cal_jobs.items():
                if entry.get("status") != "complete":
                    continue
                rp = Path(entry["result_path"])
                row = self.read_json_tracked(rp)
                self.check(
                    entry["result_sha256"] == self.read_sha256[str(rp.resolve())],
                    f"m4/timing_calibration: checksum mismatch on {rp}",
                )
                if not row.get("phase_profile"):
                    cal_timed_rows.append(row)
        elif (cal_dir / "rank0").is_dir():
            for rp in sorted((cal_dir / "rank0").glob("*.json")):
                row = self.read_json_tracked(rp)
                if not row.get("phase_profile"):
                    cal_timed_rows.append(row)

        self.check(len(cal_timed_rows) > 0, "m4/timing_calibration: 0 timed calibration rows found")
        if not cal_timed_rows:
            return {"status": cal.get("status"), "calibrated_timed_runs": 0, "groups": []}

        # Verify 1-GPU isolation metadata & reference prompt/repeat counts (`4 std + 2 long x 2 repeats x 5 arms = 60`).
        std_pids = sorted({r["prompt_id"] for r in cal_timed_rows if r["condition"] == "standard_nonthinking"})
        long_pids = sorted({r["prompt_id"] for r in cal_timed_rows if r["condition"].startswith("long_")})
        for r in cal_timed_rows:
            r_gpu, r_numa, r_cpus = self._extract_affinity_info(r)
            if r_gpu is not None:
                self.check(r_gpu == 0, f"m4/timing_calibration: expected isolated physical_gpu==0, got {r_gpu}")
            if r_cpus is not None and len(r_cpus) == 16:
                self.check(
                    sorted(r_cpus) == list(range(0, 16)) and (r_numa in (None, 0)),
                    f"m4/timing_calibration: expected CPUs 0..15 on NUMA 0, got {r_cpus} (numa={r_numa})",
                )
        if self.scale == "reference" and cal.get("status") in ("complete", "awaiting_sharded_reconciliation"):
            self.check(
                len(std_pids) == 4 and len(long_pids) == 2 and len(cal_timed_rows) == 60,
                f"m4/timing_calibration: expected 4 std + 2 long prompts (60 runs), got {len(std_pids)} std + {len(long_pids)} long ({len(cal_timed_rows)} runs)",
            )
            psel = cal.get("prompt_selection")
            if isinstance(psel, dict) and psel.get("long_selection_source") is not None:
                self.check(
                    psel.get("long_selection_source") == "evaluate_shortest_non_repetition_dominated",
                    f"m4/timing_calibration: expected evaluate_shortest_non_repetition_dominated, got {psel.get('long_selection_source')!r}",
                )
                if any(
                    r["generation"]["repetition_dominated"]
                    for r in cal_timed_rows
                    if r["condition"].startswith("long_")
                ):
                    self.warn(
                        "m4/timing_calibration: non-rep-dominated evaluate prompt produced a repetition_dominated repeat during calibration"
                    )
            else:
                self.check(
                    not any(
                        r["generation"]["repetition_dominated"]
                        for r in cal_timed_rows
                        if r["condition"].startswith("long_")
                    ),
                    "m4/timing_calibration: selected long_thinking calibration prompt has repetition_dominated=True",
                )

        # Independently recompute 1-GPU isolated vs 8-GPU sharded tok/s and arm/plain_dflash ratios on the exact same prompts.
        cal_slots = {
            (r["condition"], float(r["temperature"]), json.dumps(r["sampling_warp_config"], sort_keys=True), r["prompt_id"], r["seed"])
            for r in cal_timed_rows
        }
        sharded_sub = [
            r
            for r in sharded_timed_rows
            if (
                r["condition"],
                float(r["temperature"]),
                json.dumps(r["sampling_warp_config"], sort_keys=True),
                r["prompt_id"],
                r["seed"],
            )
            in cal_slots
        ]

        def _prompt_mean_tps(rows_subset: list[dict[str, Any]]) -> float | None:
            by_p: dict[str, list[float]] = defaultdict(list)
            for rw in rows_subset:
                by_p[rw["prompt_id"]].append(float(rw["tokens_per_second"]))
            if not by_p:
                return None
            return statistics.mean(statistics.mean(vs) for vs in by_p.values())

        def _prompt_mean_paired_ratio(rows_subset: list[dict[str, Any]], target_arm: str) -> float | None:
            plain_map = {
                (rw["prompt_id"], rw["seed"], str(rw["repeat"]).split(":epoch=")[0]): float(rw["tokens_per_second"])
                for rw in rows_subset
                if rw["arm"] == "plain_dflash"
            }
            by_p: dict[str, list[float]] = defaultdict(list)
            for rw in rows_subset:
                if rw["arm"] != target_arm:
                    continue
                pk = (rw["prompt_id"], rw["seed"], str(rw["repeat"]).split(":epoch=")[0])
                if pk in plain_map and plain_map[pk] > 0:
                    by_p[rw["prompt_id"]].append(float(rw["tokens_per_second"]) / plain_map[pk])
            if not by_p:
                return None
            return statistics.mean(statistics.mean(vs) for vs in by_p.values())

        cal_groups: list[dict[str, Any]] = []
        cond_keys = sorted(
            {
                (r["condition"], float(r["temperature"]), json.dumps(r["sampling_warp_config"], sort_keys=True))
                for r in cal_timed_rows
            }
        )
        for cond, temp, warp_json in cond_keys:
            c_1gpu = [
                r
                for r in cal_timed_rows
                if r["condition"] == cond
                and float(r["temperature"]) == temp
                and json.dumps(r["sampling_warp_config"], sort_keys=True) == warp_json
            ]
            c_8gpu = [
                r
                for r in sharded_sub
                if r["condition"] == cond
                and float(r["temperature"]) == temp
                and json.dumps(r["sampling_warp_config"], sort_keys=True) == warp_json
            ]
            if not allow_partial_8gpu:
                self.check(
                    len(c_8gpu) > 0,
                    f"m4/timing_calibration: no matching sharded_timed_rows found for condition {cond} (T={temp})",
                )
            matched_pids_cond = sorted(
                {r["prompt_id"] for r in c_1gpu}.intersection({r["prompt_id"] for r in c_8gpu})
            )
            for arm in arms:
                a_1_all = [r for r in c_1gpu if r["arm"] == arm]
                a_8_all = [r for r in c_8gpu if r["arm"] == arm]
                p_1 = sorted({r["prompt_id"] for r in a_1_all})
                p_8 = sorted({r["prompt_id"] for r in a_8_all})
                if not allow_partial_8gpu:
                    self.check(
                        p_1 == p_8 and len(p_1) > 0,
                        f"m4/timing_calibration: prompt mismatch between 1-GPU ({p_1}) and 8-GPU ({p_8}) for {cond}/{arm}",
                    )
                    eff_pids = p_1
                    a_1_matched, a_8_matched = a_1_all, a_8_all
                    c_1_matched, c_8_matched = c_1gpu, c_8gpu
                else:
                    self.check(
                        set(p_8).issubset(set(p_1)),
                        f"m4/timing_calibration(partial): unexpected 8-GPU prompt {p_8} not in 1-GPU {p_1}",
                    )
                    eff_pids = sorted(set(p_1).intersection(set(p_8)))
                    a_1_matched = [r for r in a_1_all if r["prompt_id"] in eff_pids]
                    a_8_matched = [r for r in a_8_all if r["prompt_id"] in eff_pids]
                    c_1_matched = [r for r in c_1gpu if r["prompt_id"] in matched_pids_cond]
                    c_8_matched = [r for r in c_8gpu if r["prompt_id"] in matched_pids_cond]
                tps_1 = _prompt_mean_tps(a_1_all)
                pr_1 = _prompt_mean_paired_ratio(c_1gpu, arm)
                tps_1_m = _prompt_mean_tps(a_1_matched) if a_1_matched else None
                tps_8_m = _prompt_mean_tps(a_8_matched) if a_8_matched else None
                pr_1_m = _prompt_mean_paired_ratio(c_1_matched, arm) if c_1_matched else None
                pr_8_m = _prompt_mean_paired_ratio(c_8_matched, arm) if c_8_matched else None
                tps_8 = tps_8_m if not allow_partial_8gpu else (tps_8_m if len(eff_pids) == len(p_1) else None)
                pr_8 = pr_8_m if not allow_partial_8gpu else (pr_8_m if len(eff_pids) == len(p_1) else None)
                scale_8_over_1 = (tps_8_m / tps_1_m) if (tps_1_m and tps_8_m is not None) else None
                pr_diff = (pr_8_m - pr_1_m) if (pr_1_m is not None and pr_8_m is not None) else None
                cal_groups.append(
                    {
                        "condition": cond,
                        "temperature": temp,
                        "sampling_warp_config": json.loads(warp_json),
                        "arm": arm,
                        "n_prompts": len(p_1),
                        "matched_8gpu_prompts": len(eff_pids),
                        "matched_8gpu_prompt_ids": eff_pids,
                        "runs_1gpu": len(a_1_all),
                        "runs_8gpu": len(a_8_matched),
                        "tps_1gpu_mean": tps_1,
                        "tps_8gpu_mean": tps_8,
                        "matched_tps_1gpu_mean": tps_1_m,
                        "matched_tps_8gpu_mean": tps_8_m,
                        "tps_8gpu_over_1gpu_ratio": scale_8_over_1,
                        "tps_rel_diff_pct": (100.0 * (scale_8_over_1 - 1.0)) if scale_8_over_1 is not None else None,
                        "paired_ratio_vs_plain_1gpu": pr_1,
                        "paired_ratio_vs_plain_8gpu": pr_8,
                        "matched_paired_ratio_vs_plain_1gpu": pr_1_m,
                        "matched_paired_ratio_vs_plain_8gpu": pr_8_m,
                        "paired_ratio_diff_8gpu_minus_1gpu": pr_diff,
                    }
                )

        # Cross-check against any recorded comparison entries inside `timing_isolation_calibration.json`.
        recomp_map = {(g["condition"], g["arm"]): g for g in cal_groups}
        for list_key in ("groups", "comparisons", "arm_comparisons", "by_condition_and_arm"):
            rec_list = cal.get(list_key)
            if not isinstance(rec_list, list):
                continue
            for item in rec_list:
                if not isinstance(item, dict) or "condition" not in item or "arm" not in item:
                    continue
                ref_g = recomp_map.get((item["condition"], item["arm"]))
                if ref_g is None:
                    continue
                for rec_k, our_k in (
                    ("tps_1gpu_mean", "tps_1gpu_mean"),
                    ("isolated_1gpu_tokens_per_second", "tps_1gpu_mean"),
                    ("one_gpu_tokens_per_second", "tps_1gpu_mean"),
                    ("tps_8gpu_mean", "tps_8gpu_mean"),
                    ("sharded_8gpu_tokens_per_second", "tps_8gpu_mean"),
                    ("eight_gpu_tokens_per_second", "tps_8gpu_mean"),
                    ("paired_ratio_vs_plain_1gpu", "paired_ratio_vs_plain_1gpu"),
                    ("isolated_1gpu_paired_ratio", "paired_ratio_vs_plain_1gpu"),
                    ("one_gpu_paired_ratio", "paired_ratio_vs_plain_1gpu"),
                    ("paired_ratio_vs_plain_8gpu", "paired_ratio_vs_plain_8gpu"),
                    ("sharded_8gpu_paired_ratio", "paired_ratio_vs_plain_8gpu"),
                    ("eight_gpu_paired_ratio", "paired_ratio_vs_plain_8gpu"),
                ):
                    if item.get(rec_k) is not None and ref_g.get(our_k) is not None:
                        self.check(
                            math.isclose(float(item[rec_k]), float(ref_g[our_k]), rel_tol=1e-9, abs_tol=1e-9),
                            f"m4/timing_isolation_calibration.json {item['condition']}/{item['arm']}.{rec_k}={item[rec_k]} drifts from raw recomputation {ref_g[our_k]}",
                        )

        return {
            "status": cal.get("status"),
            "path": str(cal_path),
            "standard_prompts": std_pids,
            "long_prompts": long_pids,
            "calibrated_timed_runs": len(cal_timed_rows),
            "groups": cal_groups,
        }

    def audit_repo_conformance(self) -> dict[str, Any]:
        """Static checks for `prompt.md` §§11-13, README 10 topics, 25 tests, and `AGENTS.md`."""
        # 1. Section 13 layout.
        missing_layout = [
            rel for rel in REQUIRED_LAYOUT_PATHS if not (self.repo / rel).exists()
        ]
        self.check(not missing_layout, f"missing Section 13 paths: {missing_layout}")

        # 2. Zero custom CUDA/Triton kernels in M0-M4 (`prompt.md:891`); optional M5-only `triton_ops.py`
        # is permitted on post-M4 branches provided none of the 10 M0-M4 `ALGORITHM_FILES` import it.
        forbidden_hits = []
        for py in sorted((self.repo / "src/edraft").glob("*.py")):
            src = self.read_bytes_tracked(py).decode("utf-8")
            if py.name == "triton_ops.py":
                continue
            for pat in (r"\bimport\s+triton\b", r"\bfrom\s+triton\b", r"load_inline\s*\(", r"CUDAExtension\b"):
                if re.search(pat, src):
                    forbidden_hits.append(f"{py.name}:{pat}")
            if py.name in ALGORITHM_FILES and re.search(r"\btriton_ops\b", src):
                forbidden_hits.append(f"{py.name}:imports_m5_triton_ops")
        if self.source_repo is not None and (self.source_repo / "src/edraft").is_dir():
            for py in sorted((self.source_repo / "src/edraft").glob("*.py")):
                src = self.read_bytes_tracked(py).decode("utf-8")
                for pat in (r"\bimport\s+triton\b", r"\bfrom\s+triton\b", r"load_inline\s*\(", r"CUDAExtension\b"):
                    if re.search(pat, src):
                        forbidden_hits.append(f"source_repo/{py.name}:{pat}")
        self.check(not forbidden_hits, f"custom CUDA/Triton kernel usage in M0-M4 src/edraft: {forbidden_hits}")

        # 3. All 25 required tests in `prompt.md` Section 12 verified via AST.
        ast_cache: dict[str, dict[str, ast.FunctionDef]] = {}
        test_mapping_report = []
        for num, spec, targets in SECTION12_REQUIRED_TESTS:
            verified_fns = []
            for rel_file, fn_name in targets:
                if rel_file not in ast_cache:
                    t_path = self.repo / rel_file
                    self.check(t_path.is_file(), f"missing test file {rel_file}")
                    tree = ast.parse(self.read_bytes_tracked(t_path).decode("utf-8"))
                    ast_cache[rel_file] = {
                        n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)
                    }
                fn_node = ast_cache[rel_file].get(fn_name)
                self.check(fn_node is not None, f"Section 12 test #{num} missing {rel_file}::{fn_name}")
                if fn_node is not None:
                    # Reject empty `pass` / `...` stubs (`AGENTS.md` anti-placeholder rule).
                    non_doc = [
                        stmt
                        for stmt in fn_node.body
                        if not (
                            isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Constant)
                            and isinstance(stmt.value.value, str)
                        )
                    ]
                    is_stub = len(non_doc) == 0 or (
                        len(non_doc) == 1 and isinstance(non_doc[0], ast.Pass)
                    )
                    self.check(not is_stub, f"Section 12 test #{num} {rel_file}::{fn_name} is a stub")
                    verified_fns.append(f"{rel_file}::{fn_name}")
            test_mapping_report.append(
                {"test_number": num, "requirement": spec, "verified_functions": verified_fns}
            )

        # 4. All 10 required README topics (`prompt.md:956-967`).
        readme_text = self.read_bytes_tracked(self.repo / "README.md").decode("utf-8")
        readme_topics_report = {}
        for topic, patterns in README_REQUIRED_TOPICS:
            missing_pats = [p for p in patterns if not re.search(p, readme_text)]
            self.check(not missing_pats, f"README.md missing {topic} patterns: {missing_pats}")
            readme_topics_report[topic] = {"passed": not missing_pats, "checked_patterns": len(patterns)}

        # 5. Frozen 36-binding integrity check on `/home/catid/drafter` (`progress.md`).
        # Note: `reference_continuation_state.json` itself updates its heartbeat timestamp every
        # 5s while the live driver runs, so we read it without adding `state_path` to
        # `self.read_sha256`, while tracking and re-verifying bound immutable files!
        bindings_report: dict[str, Any] = {"present": False}
        state_path = self.artifacts / "reference_continuation_state.json"
        if not state_path.is_file():
            state_path = self.repo / "artifacts/reference_continuation_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            frozen_bindings = state.get("frozen_artifact_binding", {})
            mismatches = []
            immutable_mismatches = []
            sharded_harness_modified = []
            post_m4_verified_via_09a5469 = []
            post_m4_mode = self._is_post_m4_repo() or (self.source_repo is not None and self.repo != self.source_repo)
            for rel_or_abs, expected_sha in sorted(frozen_bindings.items()):
                raw_p = Path(rel_or_abs)
                rel_repo: str | None = None
                if str(raw_p).startswith("/home/catid/drafter/artifacts/m4_evidence_validator/"):
                    rel_repo = str(raw_p.relative_to("/home/catid/drafter"))
                    fpath = self.repo / rel_repo
                elif str(raw_p).startswith("/home/catid/drafter/artifacts/"):
                    sub = raw_p.relative_to("/home/catid/drafter/artifacts")
                    cand_art = self.artifacts / sub
                    fpath = cand_art if cand_art.is_file() else (
                        (self.repo / "artifacts" / sub) if (self.repo / "artifacts" / sub).is_file() else raw_p
                    )
                elif str(raw_p).startswith("/home/catid/drafter/"):
                    rel_repo = str(raw_p.relative_to("/home/catid/drafter"))
                    fpath = self.repo / rel_repo
                elif not raw_p.is_absolute():
                    if raw_p.parts and raw_p.parts[0] == "artifacts" and len(raw_p.parts) >= 2 and raw_p.parts[1] != "m4_evidence_validator":
                        sub = Path(*raw_p.parts[1:])
                        cand_art = self.artifacts / sub
                        fpath = cand_art if cand_art.is_file() else (self.repo / raw_p)
                    else:
                        rel_repo = str(raw_p)
                        fpath = self.repo / raw_p
                else:
                    fpath = raw_p
                actual_sha = sha256_file(fpath)
                self.read_sha256[str(fpath.resolve())] = actual_sha
                if actual_sha != expected_sha:
                    entry = {"path": str(fpath), "expected": expected_sha, "actual": actual_sha}
                    mismatches.append(entry)
                    is_auth_harness = (
                        fpath.parent.name == "edraft"
                        and fpath.name in HARNESS_SHARDING_AUTHORIZED_FILES
                    )
                    if is_auth_harness and (
                        self.m4_timing_world_size > 1
                        or (self.artifacts / "m4" / "summary.json").is_file()
                    ):
                        sharded_harness_modified.append(fpath.name)
                    elif post_m4_mode and rel_repo is not None:
                        blob_09a = self._read_09a5469_repo_file(rel_repo)
                        if blob_09a is not None and sha256_bytes(blob_09a) == expected_sha:
                            post_m4_verified_via_09a5469.append(rel_repo)
                        else:
                            immutable_mismatches.append(entry)
                    else:
                        immutable_mismatches.append(entry)
            # Upstream M0-M2/prompt/trace artifacts (11) must NEVER drift; M0-M4 09a5469 code/config
            # bindings must match either directly or via verified 09a5469 snapshot/git commit on post-M4 branches.
            if not self.require_complete:
                self.check(
                    len(frozen_bindings) == 36 and not immutable_mismatches,
                    f"frozen continuation immutable/algorithm bindings modified prematurely (count={len(frozen_bindings)}): {immutable_mismatches}",
                )
            bindings_report = {
                "present": True,
                "total_bindings": len(frozen_bindings),
                "mismatches": mismatches,
                "immutable_mismatches": immutable_mismatches,
                "authorized_sharded_harness_updates": sharded_harness_modified,
                "post_m4_verified_via_09a5469": post_m4_verified_via_09a5469,
            }

        # 6. Post-Gate-E documentation freshness & 8-GPU sharding + 1-GPU calibration disclosure (`AGENTS.md` + `#119`).
        summary_md_text = self.read_bytes_tracked(self.repo / "reports/SUMMARY.md").decode("utf-8")
        ms_path = self.repo / "artifacts/milestones.json"
        if not ms_path.is_file():
            ms_path = self.artifacts / "milestones.json"
        milestones_json = self.read_json_tracked(ms_path)
        stale_markers = []
        for doc_name, txt in (("README.md", readme_text), ("reports/SUMMARY.md", summary_md_text)):
            for pat in (
                r"Fresh M3 development selection is \*?\*?running",
                r"M4 is \*?\*?not_run",
                r"M3 test\s+evaluation and timing, and the M4 experiments, remain pending",
                r"Closed-loop quality, online adaptation, and end-to-end speed conclusions remain pending",
            ):
                if re.search(pat, txt):
                    stale_markers.append(f"{doc_name}: matches stale pre-Gate-E pattern {pat!r}")
        if milestones_json.get("M3", {}).get("status") != "passed":
            stale_markers.append(f"artifacts/milestones.json: M3.status={milestones_json.get('M3', {}).get('status')!r}")
        if milestones_json.get("M4", {}).get("status") != "passed":
            stale_markers.append(f"artifacts/milestones.json: M4.status={milestones_json.get('M4', {}).get('status')!r}")

        sharded_disclosure_missing = []
        if self.m4_timing_world_size > 1:
            for doc_name, txt in (("README.md", readme_text), ("reports/SUMMARY.md", summary_md_text)):
                if not re.search(r"8-GPU|world_size\s*=\s*8", txt, re.IGNORECASE):
                    sharded_disclosure_missing.append(f"{doc_name}: missing 8-GPU sharded M4 timing disclosure")
                if not re.search(r"core-pin|cpu_affinity|NUMA", txt, re.IGNORECASE):
                    sharded_disclosure_missing.append(f"{doc_name}: missing NUMA/16-core pinning disclosure")
                if not re.search(r"timing_isolation_calibration|1-GPU\s+isolated\s+calibration|1-GPU\s+calibration", txt, re.IGNORECASE):
                    sharded_disclosure_missing.append(f"{doc_name}: missing 1-GPU isolated timing calibration disclosure")

        if self.require_complete and self.scale == "reference":
            self.check(
                not stale_markers,
                f"Post-Gate-E documentation/milestone status still has pre-completion markers: {stale_markers}",
            )
            self.check(
                not sharded_disclosure_missing,
                f"Post-Gate-E documentation missing mandatory 8-GPU sharded M4 timing / 1-GPU calibration disclosures: {sharded_disclosure_missing}",
            )

        return {
            "layout_files_checked": len(REQUIRED_LAYOUT_PATHS),
            "custom_kernels_detected": forbidden_hits,
            "section12_tests_verified": len(test_mapping_report),
            "section12_mapping": test_mapping_report,
            "readme_topics": readme_topics_report,
            "frozen_36_bindings": bindings_report,
            "post_gate_e_doc_staleness_markers": stale_markers,
            "sharded_m4_disclosure_missing": sharded_disclosure_missing,
        }


def _fmt_ci(stat: dict[str, Any] | None, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    mean = stat["mean"]
    ci = stat.get("ci95")
    if not ci:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _fmt_q_ci(val: float | None, qboot: dict[str, Any] | None, digits: int = 3) -> str:
    if val is None:
        return "—"
    if not qboot or not qboot.get("ci95"):
        return f"{val:.{digits}f}"
    ci = qboot["ci95"]
    return f"{val:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _cond_short(cond: str, temp: float, warp: dict[str, Any]) -> str:
    if cond == "long_thinking":
        return f"Long thinking (T={temp:g}, top-k={warp['top_k']}, top-p={warp['top_p']:g})"
    if temp == 0:
        return "Standard non-thinking greedy (T=0)"
    if warp.get("top_k", 0) == 0 and warp.get("top_p", 1.0) == 1.0:
        return f"Standard pure-T={temp:g} diagnostic"
    return f"Standard recommended (T={temp:g}, top-k={warp['top_k']}, top-p={warp['top_p']:g})"


def render_markdown_tables(audit_payload: dict[str, Any]) -> str:
    """Deterministic zero-drift Markdown table generator for `reports/SUMMARY.md`."""
    lines: list[str] = []
    scale = audit_payload["scale"]
    lines.append(f"# Deterministically Recomputed M3 & M4 Evidence Tables (`scale={scale}`)")
    lines.append("")
    lines.append(
        "> Generated strictly from verified raw request JSONs and sidecars with zero manual transcription."
    )
    lines.append("")

    for stage_key in ("m3", "m4"):
        st = audit_payload["stages"].get(stage_key, {})
        title = stage_key.upper()
        lines.append(f"## {title} Closed-Loop Results")
        lines.append("")
        if st.get("status") in ("not_started", "missing"):
            lines.append(f"*{title} has not started yet in `{ audit_payload['artifacts'] }`.*")
            lines.append("")
            continue

        lines.append(f"- Frozen settings SHA256: `{st['frozen_settings_hash']}`")
        lines.append(
            f"- Selected settings: `{json.dumps(st['selected_settings'], sort_keys=True)}`"
        )
        lines.append(
            f"- Completed evaluation requests: **{st['evaluated_rows']} / {st['expected_evaluated_rows']}**"
        )
        lines.append("")

        # Dev selection table
        if st.get("dev_selection_table"):
            lines.append(f"### {title} Development Selection (`tune`)")
            lines.append("")
            lines.append(
                "| Arm | Candidate setting | Selected | Selection score | Std norm tput | Long norm tput | Dev reqs | Rep-excl pairs |"
            )
            lines.append("| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: |")
            for r in st["dev_selection_table"]:
                cn = r["condition_normalized_throughput"]
                lines.append(
                    f"| `{r['arm']}` | `{json.dumps(r['settings'], sort_keys=True)}` | "
                    f"{'yes' if r['selected'] else 'no'} | {r['selection_score']:.4f} | "
                    f"{cn.get('standard_nonthinking', float('nan')):.4f} | "
                    f"{cn.get('long_thinking', float('nan')):.4f} | "
                    f"{r['dev_requests']} | {r['repetition_excluded_matched_pairs']} |"
                )
            lines.append("")

        # Test acceptance & output table
        if st.get("test_acceptance_groups"):
            lines.append(f"### {title} Held-Out Test Acceptance & Realized Lengths (Primary)")
            lines.append("")
            lines.append(
                "| Condition | Arm | Primary / Total (Rep-excl) | Mean accepted len [95% CI] | Seed std | Req median accepted len [95% CI] | Target fwds/tok [95% CI] | Mean out len (>8K / >16K) |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for g in st["test_acceptance_groups"]:
                m = g["metrics"]
                c_label = _cond_short(g["condition"], g["temperature"], g["sampling_warp_config"])
                seed_std = m["mean_accepted_length"].get("seed_std_mean")
                seed_str = f"{seed_std:.4f}" if seed_std is not None else "—"
                out_mean = g["realized_output_length_mean"]
                out_str = (
                    f"{out_mean:.1f} ({g['realized_generations_above_8K']} / {g['realized_generations_above_16K']})"
                    if out_mean is not None
                    else "—"
                )
                excl_note = (
                    f" ({g['repetition_excluded']}: {', '.join(g['excluded_prompt_ids'])})"
                    if g["repetition_excluded"]
                    else " (0)"
                )
                lines.append(
                    f"| {c_label} | `{g['arm']}` | {g['primary_requests']}/{g['requests']}{excl_note} | "
                    f"{_fmt_ci(m['mean_accepted_length'], 3)} | {seed_str} | "
                    f"{_fmt_ci(m['median_accepted_length'], 3)} | "
                    f"{_fmt_ci(m['target_forwards_per_generated_token'], 4)} | {out_str} |"
                )
            lines.append("")

        # Secondary loop-included sensitivity for long_thinking when any exclusions exist
        if any(g["repetition_excluded"] > 0 for g in st.get("test_acceptance_groups", [])):
            lines.append(
                f"### {title} Secondary Sensitivity: All `long_thinking` Test Generations (Including Repetition-Dominated Loops)"
            )
            lines.append("")
            lines.append(
                "| Arm | All requests | All-rows mean accepted len [95% CI] | Target fwds/tok [95% CI] | Mean out len (>8K / >16K) |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for g in st["long_all_rows_sensitivity"]:
                m = g["metrics"]
                lines.append(
                    f"| `{g['arm']}` | {g['requests']} | {_fmt_ci(m['mean_accepted_length'], 3)} | "
                    f"{_fmt_ci(m['target_forwards_per_generated_token'], 4)} | "
                    f"{g['realized_output_length_mean']:.1f} ({g['realized_generations_above_8K']} / {g['realized_generations_above_16K']}) |"
                )
            lines.append("")

        # Block-position acceptance table (positions 0, 1, 2, 4, 7, 10, 14 or all available)
        if st.get("test_acceptance_groups"):
            lines.append(f"### {title} Acceptance Probability by Logical Block Position (`0..K-1`)")
            lines.append("")
            sample_pos = sorted(
                int(k) for k in st["test_acceptance_groups"][0]["acceptance_by_block_position"]
            )
            hdr = " | ".join(f"pos {p}" for p in sample_pos)
            sep = " | ".join("---:" for _ in sample_pos)
            lines.append(f"| Condition | Arm | {hdr} |")
            lines.append(f"| --- | --- | {sep} |")
            for g in st["test_acceptance_groups"]:
                c_label = _cond_short(g["condition"], g["temperature"], g["sampling_warp_config"])
                pos_vals = " | ".join(
                    f"{g['acceptance_by_block_position'][str(p)]['mean']:.3f}"
                    if str(p) in g["acceptance_by_block_position"]
                    else "—"
                    for p in sample_pos
                )
                lines.append(f"| {c_label} | `{g['arm']}` | {pos_vals} |")
            lines.append("")

        # Output-offset & context-length bins with explicit 16K+ sparse unavailability flag
        if st.get("test_acceptance_groups"):
            lines.append(
                f"### {title} Acceptance by Output-Token Offset & Input-Context-Length Bins (`16K+` Sparsity Gate Enforced)"
            )
            lines.append("")
            lines.append(
                "| Condition | Arm | Axis | Bin | Prompts (`n`) | Slot accept prob [95% CI] | Block accepted len [95% CI] | Gate status |"
            )
            lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
            for g in st["test_acceptance_groups"]:
                c_label = _cond_short(g["condition"], g["temperature"], g["sampling_warp_config"])
                observed = {(b["axis"], b["bin"]): b for b in g["acceptance_bins"]}
                for ob in OUTPUT_LABELS:
                    b = observed.get(("output_offset", ob))
                    if b is None:
                        if g["condition"].startswith("long_"):
                            lines.append(
                                f"| {c_label} | `{g['arm']}` | `output_offset` | `{ob}` | 0 | — | — | **EMPTY (conclusion unavailable)** |"
                            )
                        continue
                    gate = (
                        f"**SPARSE (`n={b['n_prompts']} < {SPARSE_LONG_BIN_MIN_PROMPTS}`; conclusion unavailable)**"
                        if (
                            g["condition"].startswith("long_")
                            and b["n_prompts"] < SPARSE_LONG_BIN_MIN_PROMPTS
                        )
                        else "usable"
                    )
                    lines.append(
                        f"| {c_label} | `{g['arm']}` | `output_offset` | `{ob}` | {b['n_prompts']} | "
                        f"{_fmt_ci(b, 3)} | {_fmt_ci(b['accepted_length'], 3)} | {gate} |"
                    )
                ctx_bins = sorted(
                    (b for b in g["acceptance_bins"] if b["axis"] == "input_context_length"),
                    key=lambda b: int(b["bin"].split("-")[0].rstrip("+")),
                )
                for b in ctx_bins:
                    gate = (
                        f"**SPARSE (`n={b['n_prompts']} < {SPARSE_LONG_BIN_MIN_PROMPTS}`; conclusion unavailable)**"
                        if (
                            g["condition"].startswith("long_")
                            and b["n_prompts"] < SPARSE_LONG_BIN_MIN_PROMPTS
                        )
                        else "usable"
                    )
                    lines.append(
                        f"| {c_label} | `{g['arm']}` | `input_context_length` | `{b['bin']}` | {b['n_prompts']} | "
                        f"{_fmt_ci(b, 3)} | {_fmt_ci(b['accepted_length'], 3)} | {gate} |"
                    )
            lines.append("")

        # Greedy output comparisons
        if st.get("greedy_comparisons"):
            lines.append(f"### {title} Standard Greedy (`T_sample=0, beta=1`) Output Exactness & BF16 Tie Report")
            lines.append("")
            lines.append(
                "| Arm | Reference | Compared prompts | Mismatched seqs (rate) | Mismatched tokens / compared (rate) | Length-diff seqs |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
            for gr in st["greedy_comparisons"]:
                lines.append(
                    f"| `{gr['arm']}` | `{gr['reference']}` | {gr['compared_requests']} | "
                    f"{gr['mismatched_sequences']} ({gr['sequence_mismatch_rate']:.3f}) | "
                    f"{gr['mismatched_tokens']} / {gr['compared_tokens']} ({gr['token_mismatch_rate']:.4f}) | "
                    f"{gr['length_mismatched_sequences']} |"
                )
            lines.append("")

        # Isolated 1-GPU or 8-GPU core-pinned sharded timing + phases + 1-GPU isolation calibration
        tim = st.get("timing", {})
        if tim.get("groups"):
            if tim.get("is_sharded") or int(tim.get("world_size", 1)) > 1:
                w_sz = int(tim.get("world_size", 8))
                lines.append(
                    f"### {title} Core-Pinned {w_sz}-GPU Sharded Matched-Bundle Throughput, Latency & Peak Memory"
                )
                lines.append("")
                lines.append(
                    f"> **Methodology Disclosure ({w_sz}-GPU NUMA/Core-Pinned Sharded Timing + 1-GPU Calibration):** "
                    f"Executed across `world_size={w_sz}` GPUs (`sharding_mode={tim.get('sharding_mode')}`) with "
                    f"disjoint NUMA-local 16-physical-core pinning (`GPU g -> CPUs [16g..16g+15]`, NUMA node `g // 2`), "
                    f"deterministic token-balanced snake sharding (`sharding_plan.json`), per-GPU all-arm warmup, and "
                    f"rotated back-to-back {len(st.get('selected_settings', {}))}-arm `comparison_bundle`s "
                    f"(`{tim.get('timed_requests')}` timed + `{tim.get('profile_requests')}` separate CUDA-event profile runs "
                    f"across `{tim.get('bundles')}` bundles). All 10 core `ALGORITHM_FILES` are 100% byte-identical to frozen "
                    f"`09a5469`, with live `timing_harness_implementation_sha256={tim.get('timing_harness_implementation_sha256')}` "
                    f"recorded alongside the 1-GPU isolated calibration check below."
                )
                lines.append("")
            else:
                lines.append(f"### {title} Isolated Single-GPU Throughput, Latency & Peak Memory")
                lines.append("")
            lines.append(
                "| Condition | Arm | Tokens/sec [95% CI] | Paired ratio vs plain [95% CI] | Net gain? (95% CI > 1?) | Latency p50 s [95% CI] | Latency p95 s [95% CI] | Peak alloc / res GiB |"
            )
            lines.append("| --- | --- | ---: | ---: | :---: | ---: | ---: | ---: |")
            for g in tim["groups"]:
                c_label = _cond_short(g["condition"], g["temperature"], g["sampling_warp_config"])
                tps = _fmt_ci(g["metrics"]["tokens_per_second"], 2)
                pr = _fmt_ci(g["paired_throughput_ratio"], 4)
                gain = (
                    f"{'yes' if g['throughput_improved_after_all_proposal_and_online_costs'] else 'no'} "
                    f"({'yes' if g['throughput_gain_supported_by_95_ci'] else 'no'})"
                )
                p50 = _fmt_q_ci(
                    g["request_latency_p50_seconds"],
                    g["request_latency_p50_prompt_bootstrap"],
                    3,
                )
                p95 = _fmt_q_ci(
                    g["request_latency_p95_seconds"],
                    g["request_latency_p95_prompt_bootstrap"],
                    3,
                )
                mem = (
                    f"{g['peak_allocated_bytes'] / 2**30:.2f} / {g['peak_reserved_bytes'] / 2**30:.2f}"
                    if g.get("peak_allocated_bytes") is not None
                    else "CPU (n/a)"
                )
                lines.append(
                    f"| {c_label} | `{g['arm']}` | {tps} | {pr} | {gain} | {p50} | {p95} | {mem} |"
                )
            lines.append("")

            iso_cal = tim.get("isolation_calibration")
            if iso_cal and iso_cal.get("groups"):
                lines.append(
                    f"### {title} 1-GPU Isolated vs {tim.get('world_size', 8)}-GPU Core-Pinned Timing Calibration (`timing_isolation_calibration.json`)"
                )
                lines.append("")
                lines.append(
                    f"- Calibrated on isolated `GPU 0` (`CPUs 0..15`, GPUs `0..7` verified idle around every bundle) across "
                    f"`{len(iso_cal.get('standard_prompts', []))}` `standard_nonthinking` prompts "
                    f"(`{', '.join(iso_cal.get('standard_prompts', []))}`) and "
                    f"`{len(iso_cal.get('long_prompts', []))}` non-repetition-dominated `long_thinking` prompts "
                    f"(`{', '.join(iso_cal.get('long_prompts', []))}`), `{iso_cal.get('calibrated_timed_runs')}` timed 1-GPU runs total."
                )
                lines.append("")
                lines.append(
                    "| Condition | Arm | Prompts (`n`) | 1-GPU isolated `tok/s` | 8-GPU sharded `tok/s` | `8-GPU / 1-GPU` scale (`% diff`) | 1-GPU `arm/plain` ratio | 8-GPU `arm/plain` ratio | Paired ratio diff (`8-GPU - 1-GPU`) |"
                )
                lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                for cg in iso_cal["groups"]:
                    c_label = _cond_short(cg["condition"], cg["temperature"], cg["sampling_warp_config"])
                    t1 = f"{cg['tps_1gpu_mean']:.2f}" if cg["tps_1gpu_mean"] is not None else "—"
                    t8 = f"{cg['tps_8gpu_mean']:.2f}" if cg["tps_8gpu_mean"] is not None else "—"
                    sc_str = (
                        f"{cg['tps_8gpu_over_1gpu_ratio']:.4f} ({cg['tps_rel_diff_pct']:+.2f}%)"
                        if cg["tps_8gpu_over_1gpu_ratio"] is not None
                        else "—"
                    )
                    r1 = f"{cg['paired_ratio_vs_plain_1gpu']:.4f}" if cg["paired_ratio_vs_plain_1gpu"] is not None else "—"
                    r8 = f"{cg['paired_ratio_vs_plain_8gpu']:.4f}" if cg["paired_ratio_vs_plain_8gpu"] is not None else "—"
                    rd = (
                        f"{cg['paired_ratio_diff_8gpu_minus_1gpu']:+.4f}"
                        if cg["paired_ratio_diff_8gpu_minus_1gpu"] is not None
                        else "—"
                    )
                    lines.append(
                        f"| {c_label} | `{cg['arm']}` | {cg['n_prompts']} | {t1} | {t8} | {sc_str} | {r1} | {r8} | {rd} |"
                    )
                lines.append("")

            lines.append(
                f"### {title} Separate Phase-Profile Median Durations (ms/request; `lookahead_iterations = 0.0 ms [disabled in M0-M4]`)"
            )
            lines.append("")
            lines.append(
                "| Condition | Arm | Target feat | DFlash draft | Dense score | Sparse lookup | Seq sample | Target verify | Online update* | Lookahead iters |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for pt in tim["phase_table"]:
                s11 = pt["section11_phases_ms_median"]
                lines.append(
                    f"| {pt['condition']} (T={pt['temperature']:g}) | `{pt['arm']}` | "
                    f"{s11['target_feature_extraction']:.2f} | {s11['dflash_draft']:.2f} | "
                    f"{s11['dense_adapter_score']:.2f} | {s11['sparse_lookup']:.2f} | "
                    f"{s11['sequential_sampling']:.2f} | {s11['target_verification']:.2f} | "
                    f"{s11['online_update']:.2f} | 0.00 [disabled in M0-M4] |"
                )
            lines.append("")
            lines.append(
                "*Note on `online_update`: `dflash_loop._finish_round` brackets between-round updates with CUDA events on an already-synchronized stream; for `plain_dflash`/static arms this records empty event-pair overhead (~7–23 us/round), and for `online_bigram` it records host Python table updates.*"
            )
            lines.append("")
        else:
            lines.append(f"*{title} isolated timing status: `{tim.get('status')}`.*")
            lines.append("")

        # M4 online norms/sparsity/bandwidth + M1b feature context
        if stage_key == "m4" and st.get("online_state_summary"):
            lines.append("### M4 Online Parameter Norms, Exact-Zero Sparsity, Active Buckets & Update Bandwidth")
            lines.append("")
            lines.append(
                "| Condition | Arm | Mean updates `u` | `||dW||_2` | `||dA||_2` | `||db||_2` | `||dS||_2` | `db` exact-zero frac | Active buckets mean (max / cap) | Est state BW GB/s [95% CI] |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for osum in st["online_state_summary"]:
                bw_str = _fmt_ci(osum.get("profile_estimated_state_bandwidth_GBps"), 3)
                lines.append(
                    f"| {osum['condition']} (T={osum['temperature']:g}) | `{osum['arm']}` | "
                    f"{osum['executed_updates_mean']:.1f} | {osum['dW_norm_mean']:.5f} | "
                    f"{osum['dA_norm_mean']:.5f} | {osum['db_norm_mean']:.5f} | "
                    f"{osum['dS_norm_mean']:.5f} | {osum['db_exact_zero_fraction_mean']:.4f} | "
                    f"{osum['active_buckets_mean']:.1f} ({osum['active_buckets_max']} / {osum['sparse_capacity']}) | "
                    f"{bw_str} |"
                )
            lines.append("")

        if stage_key == "m4" and st.get("feature_context_summary"):
            fc = st["feature_context_summary"]
            lines.append("### M4 Linked M1b Plain-DFlash `h[i]` Covariate-Shift Context")
            lines.append("")
            lines.append(f"- Provenance: {fc['source']} ({fc['interpretation']})")
            lines.append("")
            lines.append(
                "| Condition | Temp | Output bin | Logical slots | Coord mean norm | Variance / cal | L2 median (p95) | Outlier slot rate |"
            )
            lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
            for fg in fc["all_offsets_primary_groups"]:
                lines.append(
                    f"| `{fg['condition']}` | {fg['temperature']:g} | `{fg['bin']}` | {fg['logical_slots']} | "
                    f"{fg['coordinate_mean_norm']:.3f} | {fg['variance_ratio']:.3f} | "
                    f"{fg['l2_median']:.3f} ({fg['l2_p95']:.3f}) | {fg['outlier_slot_rate'] * 100:.3f}% |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def run_audit(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/catid/drafter"))
    parser.add_argument(
        "--source_repo",
        type=Path,
        default=None,
        help="Optional frozen 09a5469 source repo snapshot for M3/M4 implementation hash verification",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("/home/catid/drafter/artifacts"),
        help="Artifacts root (e.g. artifacts/repetition_detector_review/integration_final for smoke)",
    )
    parser.add_argument("--scale", choices=("smoke", "reference"), default="reference")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Permit in-progress M3 timing / unstarted M4 while live continuation runs",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Post-Gate-E strict mode: require M3+M4 complete, m4 validator passed, and docs updated",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory outside --repo (never writes inside --repo)",
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    out_dir = args.output_dir.resolve()
    if out_dir == repo or repo in out_dir.parents:
        raise ValueError(f"Safety guard: --output_dir ({out_dir}) must NOT lie inside --repo ({repo})")

    started_utc = datetime.now(timezone.utc).isoformat()
    auditor = PostGateEAuditor(
        repo=repo,
        source_repo=args.source_repo,
        artifacts=args.artifacts,
        scale=args.scale,
        seed=args.seed,
        allow_partial=args.allow_partial,
        require_complete=args.require_complete,
    )

    bigram_event_audit = auditor.audit_online_bigram_cuda_events()
    m3_report = auditor.audit_stage("m3")
    m4_report = auditor.audit_stage("m4")
    conformance = auditor.audit_repo_conformance()

    # Run `artifacts/m4_evidence_validator/validate.py` automatically into `out_dir` whenever M4 timing.json exists!
    m4_validator_result = None
    if (args.artifacts.resolve() / "m4" / "timing.json").is_file():
        val_out = out_dir / f"m4_validator_{args.scale}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        val_script = repo / "artifacts/m4_evidence_validator/validate.py"
        cmd = [
            sys.executable,
            str(val_script),
            "--scale",
            args.scale,
            "--repo",
            str(repo),
            "--artifacts",
            str(args.artifacts.resolve()),
            "--seed",
            str(args.seed),
            "--output",
            str(val_out),
        ]
        if args.source_repo is not None and "--source_repo" in val_script.read_text(encoding="utf-8"):
            cmd.extend(["--source_repo", str(args.source_repo.resolve())])
        proc = subprocess.run(
            cmd,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        auditor.check(proc.returncode == 0, f"m4_evidence_validator exited {proc.returncode}: {proc.stderr}")
        if val_out.is_file():
            m4_validator_result = json.loads(val_out.read_text(encoding="utf-8"))
            auditor.check(
                m4_validator_result.get("status") == "passed"
                and not m4_validator_result.get("failures"),
                f"m4_evidence_validator reported failures: {m4_validator_result.get('failures')}",
            )
    elif args.require_complete:
        auditor.check(False, "require_complete set but m4/timing.json is missing")

    # Re-verify every read file's SHA256 at the end of the run to prove 0 mutations occurred.
    for path_str, expected_digest in list(auditor.read_sha256.items()):
        p_check = Path(path_str)
        actual_digest = sha256_file(p_check)
        if actual_digest != expected_digest:
            if args.allow_partial and not args.require_complete and auditor._is_live_inflight_manifest(p_check):
                auditor.warn(f"inflight manifest advanced during live --allow_partial audit: {path_str}")
            else:
                auditor.check(False, f"input file changed during audit: {path_str}")
        else:
            auditor.check(True, f"input file unchanged during audit: {path_str}")

    payload = {
        "status": "passed" if not auditor.failures else "failed",
        "scale": args.scale,
        "repo": str(repo),
        "artifacts": str(args.artifacts.resolve()),
        "allow_partial": args.allow_partial,
        "require_complete": args.require_complete,
        "started_at_utc": started_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": auditor.checks,
        "failures": auditor.failures,
        "warnings": auditor.warnings,
        "online_bigram_cuda_event_audit": bigram_event_audit,
        "m4_evidence_validator": (
            {
                "status": m4_validator_result["status"],
                "checks": m4_validator_result["checks"],
                "counts": m4_validator_result["counts"],
                "failures": m4_validator_result["failures"],
            }
            if m4_validator_result is not None
            else None
        ),
        "stages": {"m3": m3_report, "m4": m4_report},
        "repo_conformance": conformance,
        "audited_file_count": len(auditor.read_sha256),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = out_dir / f"audit_report_{args.scale}.json"
    tables_md_path = out_dir / f"rendered_m3_m4_tables_{args.scale}.md"
    atomic_write_json(report_json_path, payload)
    atomic_write_text(tables_md_path, render_markdown_tables(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scale": payload["scale"],
                "checks": payload["checks"],
                "failures": payload["failures"],
                "warnings_count": len(payload["warnings"]),
                "audited_file_count": payload["audited_file_count"],
                "report_json": str(report_json_path),
                "tables_md": str(tables_md_path),
            }
        )
    )
    return payload


if __name__ == "__main__":
    res = run_audit()
    sys.exit(0 if res["status"] == "passed" else 1)
