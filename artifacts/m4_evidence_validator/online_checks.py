"""Independent checks of recorded M4 online rows; no model or project imports.

These checks validate JSON evidence, not the underlying tensor execution. The
caller owns schedule coverage, frozen settings, identities, and file checksums.
"""
from __future__ import annotations

import math


EVIDENCE_LIMITS = (
    "Version counters do not prove tensor snapshot immutability.",
    "Request-reset and no-buffering flags are declarations; initial tensors and "
    "intermediate gradients are not serialized.",
    "Final norms do not independently prove unchanged offline tensors, correct "
    "gradients, or exact generation distributions.",
    "State bytes and bandwidth are algorithmic estimates, not hardware traffic.",
)
ONLINE_ARMS = frozenset({
    "offline_online_stride1", "online_dbS", "offline_online_stride5",
    "zero_offline_online_only", "online_rejection_relative", "online_cutoff",
})
BYTE_OPERATIONS = frozenset({
    "dense_activation_clone", "dense_optimizer", "dense_snapshot",
    "sparse_update", "sparse_snapshot",
})


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def check_online_row(row, *, expected_scale, expected_arm, expected_setting, profile):
    """Return check counts and failures without mutating any supplied object.

    Inputs are decoded JSON objects. ``update_count`` is derived independently
    from the number of blocks and the externally supplied expected stride.
    Malformed JSON field types produce failures, rather than exceptions.
    """
    result = {"checks": 0, "failures": [], "block_count": None,
              "update_count": None, "evidence_limits": list(EVIDENCE_LIMITS)}

    def check(condition, label):
        result["checks"] += 1
        if not condition:
            result["failures"].append(label)
        return bool(condition)

    def object_field(container, key, label=None):
        value = container.get(key)
        return value if check(isinstance(value, dict), label or key + ": object required") else {}

    def int_field(container, key, label, minimum=0):
        value = container.get(key)
        valid = _integer(value) and value >= minimum
        return value if check(valid, label + ": integer >= " + str(minimum) + " required") else None

    if not check(isinstance(row, dict), "row: object required"):
        return result
    valid_scale = check(isinstance(expected_scale, str) and expected_scale in ("smoke", "reference", "full"),
                        "expected_scale: unsupported")
    check(isinstance(expected_arm, str) and expected_arm in ONLINE_ARMS, "expected_arm: online arm required")
    check(isinstance(profile, bool), "profile argument: boolean required")
    check(row.get("scale") == expected_scale, "scale differs from expected")
    check(row.get("stage") == "m4", "stage must be m4")
    check(row.get("arm") == expected_arm, "arm differs from expected")
    check(isinstance(row.get("phase_profile"), bool) and row["phase_profile"] == profile,
          "phase_profile differs from expected")
    check("baseline_reuse" not in row, "online row cannot reuse baseline evidence")
    setting = object_field(row, "settings")
    expected = expected_setting if check(isinstance(expected_setting, dict), "expected_setting: object required") else {}
    check(setting == expected, "settings differ from expected")
    # Frozen evaluator settings are flat JSON scalars. Python equality alone
    # would silently equate bool/int/float values such as True, 1, and 1.0.
    for key, value in expected.items():
        check(type(setting.get(key)) is type(value),
              "settings." + str(key) + ": JSON type differs from expected")
    stride = int_field(expected, "update_stride", "expected_setting.update_stride", minimum=1)

    blocks = row.get("blocks")
    valid_blocks = check(isinstance(blocks, list), "blocks: array required")
    if valid_blocks:
        result["block_count"] = len(blocks)
        if stride is not None:
            result["update_count"] = len(blocks) // stride
        for index, block in enumerate(blocks, 1):
            label = "blocks[" + str(index - 1) + "]"
            if not check(isinstance(block, dict), label + ": object required"):
                continue
            round_idx = int_field(block, "round_idx", label + ".round_idx", minimum=1)
            before = int_field(block, "adapter_version", label + ".adapter_version")
            after = int_field(block, "next_adapter_version", label + ".next_adapter_version")
            check(round_idx == index, label + ": rounds must be contiguous and one-based")
            if stride is not None:
                check(before == (index - 1) // stride, label + ": incorrect pre-update version")
                check(after == index // stride, label + ": incorrect post-update version")
            check(block.get("proposal_mode") == "dense_full", label + ": expected dense_full proposal")
            check(_finite(block.get("loss_temperature")) and block["loss_temperature"] == 1,
                  label + ": loss_temperature must be 1")

    online = object_field(row, "online")
    invariants = object_field(row, "online_invariants")
    norms = object_field(row, "parameter_norms")
    for key in ("last_full_loss", "active_buckets_min_over_updates"):
        check(key in online, "online." + key + ": explicit field required even when null")
    check("estimated_online_state_bandwidth_GBps" in row,
          "estimated_online_state_bandwidth_GBps: explicit field required even when null")
    updates = result["update_count"]
    for key in ("u", "adapter_version"):
        value = int_field(online, key, "online." + key)
        if updates is not None:
            check(value == updates, "online." + key + " differs from derived update count")
    last_round = int_field(online, "last_round", "online.last_round")
    if valid_blocks:
        check(last_round == len(blocks), "online.last_round differs from block count")
    inv_stride = int_field(invariants, "stride", "online_invariants.stride", minimum=1)
    inv_updates = int_field(invariants, "executed_updates", "online_invariants.executed_updates")
    if stride is not None:
        check(inv_stride == stride, "online_invariants.stride differs from expected")
    if updates is not None:
        check(inv_updates == updates, "online_invariants.executed_updates differs from derived count")
    for key in ("request_reset", "version_only_between_rounds", "single_round_no_buffering"):
        check(invariants.get(key) is True, "online_invariants." + key + ": true declaration required")
    check(online.get("loss_mode") == "live_full", "online.loss_mode must be live_full")
    for key in ("T_train", "beta_train"):
        check(_finite(online.get(key)) and online[key] == 1, "online." + key + " must be 1")
    for key in ("dW_norm", "dA_norm", "db_norm", "dS_norm"):
        check(_finite(online.get(key)) and online[key] >= 0, "online." + key + ": finite nonnegative norm required")
    for key in ("W", "A", "b"):
        check(_finite(norms.get(key)) and norms[key] >= 0, "parameter_norms." + key + ": finite nonnegative norm required")
    if expected_arm == "online_dbS":
        check(online.get("dW_norm") == 0 and online.get("dA_norm") == 0,
              "online_dbS must leave dW and dA exactly zero")
    zero_fraction = online.get("db_exact_zero_fraction")
    check(_finite(zero_fraction) and 0 <= zero_fraction <= 1, "db_exact_zero_fraction outside [0,1]")

    capacity = int_field(online, "sparse_capacity", "online.sparse_capacity", minimum=1)
    if valid_scale:
        check(capacity == (257 if expected_scale == "smoke" else 2**20), "online sparse capacity differs from run policy")
    active = int_field(online, "active_buckets", "online.active_buckets")
    maximum = int_field(online, "active_buckets_max_over_updates", "online.active_buckets_max_over_updates")
    if capacity is not None and active is not None:
        check(active <= capacity, "active_buckets exceeds capacity")
    if active == 0:
        check(online.get("dS_norm") == 0, "zero active buckets must have zero effective dS norm")
    if updates == 0:
        check(online.get("active_buckets_min_over_updates") is None, "no updates: minimum active count must be null")
        check(active == 0 and maximum == 0, "no updates: active counts must be zero")
        check(online.get("last_full_loss") is None, "no updates: last_full_loss must be null")
        check(all(online.get(key) == 0 for key in ("dW_norm", "dA_norm", "db_norm", "dS_norm")),
              "no updates: all delta norms must be zero")
        check(zero_fraction == 1, "no updates: db must be entirely zero")
    elif updates is not None:
        minimum = int_field(online, "active_buckets_min_over_updates", "online.active_buckets_min_over_updates")
        if all(value is not None for value in (minimum, active, maximum, capacity)):
            check(minimum <= active <= maximum <= capacity, "active-count extrema inconsistent")
        check(_finite(online.get("last_full_loss")), "updated request: finite last_full_loss required")

    byte_values = {}
    for key in ("state_bytes_estimate_total", "sparse_update_bytes_estimate", "sparse_snapshot_bytes", "dense_snapshot_bytes"):
        byte_values[key] = int_field(online, key, "online." + key)
    operations = object_field(online, "state_bytes_estimate_by_operation")
    check(set(operations) == BYTE_OPERATIONS, "state-byte operation keys differ from schema")
    operation_values = [int_field(operations, key, "state_bytes_estimate_by_operation." + key)
                        for key in sorted(BYTE_OPERATIONS)]
    total = byte_values["state_bytes_estimate_total"]
    if all(value is not None for value in operation_values):
        check(total == sum(operation_values), "state-byte total differs from operation sum")
    if active is not None and updates is not None:
        check(byte_values["sparse_snapshot_bytes"] == 16 * active + 4 * (updates + 1),
              "sparse snapshot-byte algebra mismatch")
    dense_bytes = byte_values["dense_snapshot_bytes"]
    if dense_bytes is not None:
        check(dense_bytes > 0 and dense_bytes % 4 == 0, "dense snapshot bytes must describe nonempty FP32 state")
        if updates is not None:
            check(operations.get("dense_activation_clone") == 2 * dense_bytes * updates,
                  "dense activation-clone byte algebra mismatch")
            check(operations.get("dense_snapshot") == 3 * dense_bytes * updates,
                  "dense snapshot cumulative byte algebra mismatch")
    if updates == 0:
        check(total == 0 and byte_values["sparse_update_bytes_estimate"] == 0,
              "no updates: update-state byte estimates must be zero")
    elif updates is not None and all(value is not None for value in
                                    (active, capacity, byte_values["sparse_update_bytes_estimate"])):
        dirty_bytes = byte_values["sparse_update_bytes_estimate"] - 12 * active
        check(0 <= dirty_bytes <= 16 * capacity and dirty_bytes % 16 == 0,
              "sparse update bytes must represent deduplicated dirty buckets plus active-list traffic")
    check(online.get("sparse_snapshot_strategy") == "compact_active_bucket_copy", "unexpected sparse snapshot strategy")
    check(online.get("sparse_update_strategy") == "dirty_lazy_catchup_and_active_list_compaction",
          "unexpected sparse update strategy")
    for key in ("dS_norm_definition", "state_bytes_definition"):
        check(isinstance(online.get(key), str) and bool(online[key].strip()), "online." + key + ": definition required")
    check(isinstance(row.get("online_bandwidth_definition"), str) and bool(row["online_bandwidth_definition"].strip()),
          "online_bandwidth_definition: definition required")

    phases = object_field(row, "phase_ms")
    for name, value in phases.items():
        check(isinstance(name, str) and _finite(value) and value >= 0,
              "phase_ms: each phase must have a finite nonnegative duration")
    if profile is True:
        required_phases = set()
        if valid_blocks and blocks:
            required_phases.update(("dflash_draft", "sequential_sampling", "target_verification"))
            # Optional zero-offline runs can remain identically zero, e.g. when
            # only one round executes or lr=0; their scoring fast path is valid.
            if expected_arm != "zero_offline_online_only":
                required_phases.update(("dense_adapter_score", "sparse_lookup"))
        if expected_scale in ("reference", "full"):
            required_phases.add("target_feature_extraction")
        for name in sorted(required_phases):
            duration = phases.get(name)
            check(_finite(duration) and duration > 0, "profile requires positive " + name + " ms")
    update_ms = phases.get("online_update", 0)
    if profile is True and updates is not None and updates > 0:
        check(_finite(update_ms) and update_ms > 0, "profile with executed updates requires positive online_update ms")
    if profile is False:
        check(not phases, "uninstrumented row must not contain phase durations")
    bandwidth = row.get("estimated_online_state_bandwidth_GBps")
    if _finite(update_ms) and update_ms > 0 and total is not None:
        try:
            expected_bandwidth = total / (update_ms * 1e6)
        except (OverflowError, ZeroDivisionError):
            expected_bandwidth = math.nan
        check(_finite(bandwidth) and _finite(expected_bandwidth)
              and math.isclose(bandwidth, expected_bandwidth, rel_tol=1e-9, abs_tol=1e-12),
              "online state bandwidth differs from bytes/update duration")
    else:
        check(bandwidth is None, "absent or zero update duration requires null bandwidth")
    return result
