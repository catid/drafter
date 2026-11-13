"""Compact, unwarped block traces and atomic, rank-local resumable shards.

These helpers do not collect traces or bypass milestone gates. The caller must
freeze the train-only hidden projection before supplying projected features.
Only tensors and primitive containers are serialized; loading uses weights_only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from .runtime import Manifest, atomic_json, git_commit


SCHEMA_VERSION = 1
_META_REQUIRED = {"prompt_id", "split", "condition", "seed", "scale", "model_pair", "projection_metadata"}
_RECORD_REQUIRED = {"draft_tokens", "draft_logits", "target_logits", "h_raw_bf16", "h",
                    "draft_lm_head", "collection_temperature", "loss_temperature",
                    "verification_rule", "sampling_warp_config", "absolute_output_offset",
                    "input_context_length", "verified_prefix_tail", "physical_to_logical",
                    "accepted_length", "first_rejection_index", "replacement_token", "adapter_version"}
_COPY_FIELDS = _RECORD_REQUIRED - {"draft_logits", "target_logits"}
_OPTIONAL_FIELDS = {"round_idx", "bonus_token", "decisions", "proposal_token_probs", "target_token_probs",
                    "proposal_mode", "off_policy_teacher_positions", "candidate_ids", "committed_tokens",
                    "next_adapter_version", "target_cache_length_after_rollback", "diagnostics"}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("trace dictionaries require string keys")
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_tree(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported trace value type: {type(value).__name__}")


def _missing(value, names, label):
    if absent := names - value.keys():
        raise ValueError(f"missing {label} fields: {sorted(absent)}")


def compact_block(record: Mapping, metadata: Mapping, M_draft: int = 256, M_target: int = 64,
                  special_tokens: Sequence[int] = (), retain_live_full: bool = False) -> dict:
    """Build a canonical CPU trace from one live loop callback.

    Stored target probabilities are the unwarped T_train=1 full softmax values
    gathered on the union, with their original mass retained. The rest category
    is 1-sum(P_union), never a renormalization of the union. ``live_full`` is an
    explicit selected-shard diagnostic for the grouped-versus-full KL gap.
    """
    _missing(record, _RECORD_REQUIRED, "live record")
    _missing(metadata, _META_REQUIRED, "metadata")
    if M_draft < 1 or M_target < 1:
        raise ValueError("union top counts must be positive")
    if float(record["loss_temperature"]) != 1:
        raise ValueError("trace loss_temperature must be unwarped unit temperature 1")
    if any(key in record and record[key] != value for key, value in metadata.items()):
        raise ValueError("metadata must not override live block fields")
    draft_logits = record["draft_logits"].detach().float()
    target_logits = record["target_logits"].detach().float()
    if draft_logits.ndim != 2 or draft_logits.shape != target_logits.shape:
        raise ValueError("draft and target logits must share shape [K,V]")
    if not bool(torch.isfinite(draft_logits).all() and torch.isfinite(target_logits).all()):
        raise ValueError("training logits must be finite and unwarped; truncation masks are forbidden")
    K, V = draft_logits.shape
    if K < 1 or len(record["draft_tokens"]) != K:
        raise ValueError("one realized draft token is required per logical slot")
    specials = sorted(set(int(token) for token in special_tokens))
    if any(token < 0 or token >= V for token in specials):
        raise ValueError("special token outside vocabulary")
    draft_log_normalizer = torch.logsumexp(draft_logits, dim=-1)
    target_log_normalizer = torch.logsumexp(target_logits, dim=-1)
    # Match the live full training distribution; never use inference warpers.
    l_unit = draft_logits.log_softmax(-1)
    P_train = target_logits.softmax(-1)
    top_draft = draft_logits.topk(min(M_draft, V), dim=-1).indices
    top_target = target_logits.topk(min(M_target, V), dim=-1).indices
    special_tensor = torch.tensor(specials, dtype=torch.long, device=draft_logits.device)
    union_ids, draft_union, target_union, rest = [], [], [], []
    for i, token in enumerate(record["draft_tokens"]):
        token = int(token)
        if not 0 <= token < V:
            raise ValueError("draft token outside vocabulary")
        ids = torch.cat([top_draft[i], top_target[i], special_tensor,
                         torch.tensor([token], dtype=torch.long, device=draft_logits.device)]).unique(sorted=True)
        p = P_train[i, ids]
        union_ids.append(ids.cpu())
        draft_union.append(l_unit[i, ids].cpu())
        target_union.append(p.cpu())
        # FP64 sum avoids losing a small tail while accumulating compact FP32
        # entries. Tiny negative rounding errors are clipped and validated.
        # A full-vocabulary union has no tail category. FP32 normalization
        # roundoff must not invent positive target mass on an empty q support.
        rest.append(torch.tensor(0., dtype=torch.float32) if len(ids) == V
                    else (1 - p.double().sum()).clamp_min(0).float().cpu())
    result = _cpu_tree(dict(metadata))
    result.update({key: _cpu_tree(record[key]) for key in _COPY_FIELDS})
    result.update({key: _cpu_tree(record[key]) for key in _OPTIONAL_FIELDS if key in record})
    result.update({"schema_version": SCHEMA_VERSION, "vocab_size": V, "block_length": K,
                   "M_draft": M_draft, "M_target": M_target, "special_tokens": specials,
                   "union_ids": union_ids, "draft_l_unit_union": draft_union,
                   "target_probs_union": target_union, "target_rest_mass": torch.stack(rest),
                   "draft_log_normalizer": draft_log_normalizer.cpu(),
                   "target_log_normalizer": target_log_normalizer.cpu(),
                   "target_distribution": "unwarped_full_softmax_unit_temperature",
                   "loss_mode": "union_rest_grouped", "beta_train": 1.,
                   "project_git_commit": metadata.get("project_git_commit", git_commit())})
    result["h_raw_bf16"] = result["h_raw_bf16"].to(torch.bfloat16)
    result["h"] = result["h"].float()
    if retain_live_full:
        result["live_full"] = {"draft_l_unit": l_unit.cpu(), "target_probs": P_train.cpu(),
                               "purpose": "heldout_grouped_gap_and_reconstruction_diagnostic"}
    validate_block(result)
    return result


def validate_block(block: Mapping, mass_atol: float = 2e-6) -> None:
    _missing(block, _META_REQUIRED | _COPY_FIELDS | {"schema_version", "vocab_size", "block_length",
             "union_ids", "draft_l_unit_union", "target_probs_union", "target_rest_mass",
             "draft_log_normalizer", "target_log_normalizer", "target_distribution", "beta_train"}, "trace")
    if block["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported trace schema version")
    if block["split"] not in {"train", "dev", "test"}:
        raise ValueError("trace split must be train, dev, or test")
    if block["scale"] not in {"synthetic", "smoke", "reference", "full"}:
        raise ValueError("trace scale must label the actual data")
    if block["condition"] not in {"standard_nonthinking", "long_nonthinking", "long_thinking"}:
        raise ValueError("unknown generation condition")
    if block["loss_temperature"] != 1 or block["beta_train"] != 1:
        raise ValueError("trace training requires T_train=beta_train=1")
    if block["target_distribution"] != "unwarped_full_softmax_unit_temperature":
        raise ValueError("trace targets must be unwarped full unit-temperature distributions")
    T = block["collection_temperature"]
    if not math.isfinite(T) or T < 0:
        raise ValueError("invalid collection temperature")
    if block["verification_rule"] != ("greedy" if T == 0 else "ratio"):
        raise ValueError("collection requires greedy argmax or the stochastic ratio rule")
    if not isinstance(block["projection_metadata"], Mapping) or not block["projection_metadata"]:
        raise ValueError("frozen projection provenance is required")
    head = block["draft_lm_head"]
    _missing(head, {"artifact_id", "revision", "dtype", "bias", "tied_weights", "hook"}, "head metadata")
    if not head["artifact_id"] or not head["revision"]:
        raise ValueError("draft output head must have immutable artifact and revision references")
    if not ("shape" in head or "weight_shape" in head):
        raise ValueError("draft head shape/output projection metadata is required")
    if head.get("final_logit_softcapping"):
        raise ValueError("nonlinear softcapping cannot reconstruct a linear draft head")
    K, V = block["block_length"], block["vocab_size"]
    if not 0 <= block["accepted_length"] <= K or block["first_rejection_index"] != block["accepted_length"]:
        raise ValueError("invalid logical accepted length/first rejection index")
    if len(block["draft_tokens"]) != K:
        raise ValueError("invalid draft token count")
    mapping = block["physical_to_logical"]
    if mapping.get("anchor_physical_index") != 0 or mapping.get("logical_to_physical") != list(range(1, K+1)):
        raise ValueError("logical slot zero must map to physical slot one after the clean anchor")
    for name in ("h", "h_raw_bf16"):
        value = block[name]
        if value.ndim != 2 or value.shape[0] != K or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid {name} tensor")
    if block["h_raw_bf16"].dtype != torch.bfloat16:
        raise ValueError("h_raw must be stored in bf16")
    D = int(block.get("history_depth", 4))
    if len(block["verified_prefix_tail"]) < min(D, block["input_context_length"]):
        raise ValueError("verified prefix tail omits required available cross-boundary history")
    for name in ("union_ids", "draft_l_unit_union", "target_probs_union", "target_rest_mass",
                 "draft_log_normalizer", "target_log_normalizer"):
        if len(block[name]) != K:
            raise ValueError(f"{name} must have K rows")
    for i, ids in enumerate(block["union_ids"]):
        lp, P, rest = block["draft_l_unit_union"][i], block["target_probs_union"][i], block["target_rest_mass"][i]
        if ids.ndim != 1 or ids.dtype != torch.int64 or not len(ids):
            raise ValueError("union IDs must be nonempty int64 vectors")
        if ids.min() < 0 or ids.max() >= V or not torch.equal(ids, ids.unique(sorted=True)):
            raise ValueError("union IDs must be unique, sorted, and in vocabulary")
        if lp.shape != ids.shape or P.shape != ids.shape:
            raise ValueError("union probability shapes do not match IDs")
        if not bool(torch.isfinite(lp).all() and torch.isfinite(P).all() and torch.isfinite(rest)):
            raise ValueError("nonfinite compact trace probabilities")
        if bool((lp > mass_atol).any() or (P < 0).any() or (P > 1).any() or rest < 0 or rest > 1):
            raise ValueError("invalid probability or log-probability range")
        if abs(float(P.double().sum() + rest.double()) - 1) > mass_atol:
            raise ValueError("target union probabilities and unrenormalized rest must sum to one")
        if not bool((ids == int(block["draft_tokens"][i])).any()):
            raise ValueError("union omits the realized draft token")
        if any(not bool((ids == int(token)).any()) for token in block.get("special_tokens", [])):
            raise ValueError("union omits a requested special token")
    if "live_full" in block:
        live = block["live_full"]
        if live["target_probs"].shape != (K, V) or live["draft_l_unit"].shape != (K, V):
            raise ValueError("live_full diagnostic tensors must cover the full vocabulary")
        for i, ids in enumerate(block["union_ids"]):
            torch.testing.assert_close(live["target_probs"][i, ids], block["target_probs_union"][i], rtol=0, atol=0)
            torch.testing.assert_close(live["draft_l_unit"][i, ids], block["draft_l_unit_union"][i], rtol=0, atol=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(path: str | Path, blocks: Sequence[Mapping], metadata: Mapping) -> dict:
    """Write all bytes before publishing the shard and its checksum sidecar."""
    path = Path(path)
    if not blocks:
        raise ValueError("cannot write an empty trace shard")
    for block in blocks:
        validate_block(block)
    payload = {"schema_version": SCHEMA_VERSION, "metadata": _cpu_tree(metadata),
               "blocks": _cpu_tree(blocks)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    summary = {"schema_version": SCHEMA_VERSION, "path": str(path.resolve()),
               "sha256": _sha256(path), "disk_bytes": path.stat().st_size,
               "blocks": len(blocks), "logical_slots": sum(b["block_length"] for b in blocks)}
    for key in ("generation", "identity", "rank"):
        if key in metadata:
            summary[key] = metadata[key]
    atomic_json(path.with_suffix(path.suffix + ".json"), summary)
    return summary


def read_shard(path: str | Path, verify_checksum: bool = True) -> dict:
    path = Path(path)
    if verify_checksum:
        sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if path.stat().st_size != sidecar["disk_bytes"] or _sha256(path) != sidecar["sha256"]:
            raise ValueError(f"trace shard checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION or not payload.get("blocks"):
        raise ValueError("invalid or empty trace shard")
    for block in payload["blocks"]:
        validate_block(block)
    return payload


class TraceShardWriter:
    """One rank writes its own manifest; interrupted/unpublished jobs can be retried."""

    def __init__(self, directory: str | Path, rank: int = 0):
        if rank < 0:
            raise ValueError("rank must be nonnegative")
        self.directory = Path(directory).resolve()
        self.rank = rank
        self.manifest = Manifest(self.directory / f"manifest-rank{rank}.json")

    def path(self, identity: Mapping) -> Path:
        return self.directory / f"rank{self.rank}" / f"{self.manifest.key(identity)}.pt"

    def completed(self, identity: Mapping) -> bool:
        if not self.manifest.completed(identity):
            return False
        path = self.path(identity)
        sidecar_path = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not sidecar_path.is_file():
            return False
        try:
            sidecar = json.loads(sidecar_path.read_text())
            return path.stat().st_size == sidecar["disk_bytes"] and _sha256(path) == sidecar["sha256"]
        except (OSError, ValueError, KeyError):
            return False

    def write(self, identity: Mapping, blocks: Sequence[Mapping], metadata: Mapping | None = None) -> Path:
        path = self.path(identity)
        if self.completed(identity):
            return path
        for block in blocks:
            comparisons = {"prompt": "prompt_id", "condition": "condition", "seed": "seed",
                           "temperature": "collection_temperature", "verification_rule": "verification_rule",
                           "adapter_version": "adapter_version", "model_pair": "model_pair"}
            for identity_key, block_key in comparisons.items():
                if identity[identity_key] != block[block_key]:
                    raise ValueError(f"shard identity mismatch for {identity_key}")
        header = dict(metadata or {})
        header.update({"identity": dict(identity), "rank": self.rank, "project_git_commit": git_commit()})
        write_shard(path, blocks, header)
        self.manifest.finish(dict(identity), path)
        return path


@torch.no_grad()
def reconstruction_diagnostic(record: Mapping, head, *, bias: torch.Tensor | None = None,
                              output_multiplier: float | None = None, atol: float = 0.15,
                              rtol: float = 0.01, probability_atol: float = 0.01,
                              probability_rtol: float = 0.05, chunk_size: int = 8192) -> dict:
    """Compare stored bf16 h_raw + exact frozen linear head against live logits.

    FP32 matmul accumulation is performed in vocabulary chunks to avoid cloning
    a full large head in FP32. Tolerances are explicit report fields and are
    evaluated on all K*V entries, with no inference-temperature/warper applied.
    """
    if min(atol, rtol, probability_atol, probability_rtol) < 0 or chunk_size < 1:
        raise ValueError("invalid reconstruction tolerance or chunk size")
    weight = head.weight if hasattr(head, "weight") else head
    bias = getattr(head, "bias", bias) if hasattr(head, "weight") else bias
    metadata = record.get("draft_lm_head", {})
    if metadata.get("final_logit_softcapping"):
        raise ValueError("linear reconstruction does not support nonlinear softcapping")
    multiplier = metadata.get("output_multiplier", 1.) if output_multiplier is None else output_multiplier
    h = record["h_raw_bf16"].to(device=weight.device, dtype=torch.float32)
    reconstructed = torch.cat([
        F.linear(h, weight[start:start+chunk_size].float(),
                 None if bias is None else bias[start:start+chunk_size].float()) * multiplier
        for start in range(0, weight.shape[0], chunk_size)
    ], dim=-1)
    live_is_log_probability = "draft_logits" not in record
    source = record["live_full"]["draft_l_unit"] if live_is_log_probability else record["draft_logits"]
    live = source.detach().to(device=weight.device, dtype=torch.float32)
    if reconstructed.shape != live.shape:
        raise ValueError("head and live logits disagree in shape")
    reconstructed_lp, live_lp = reconstructed.log_softmax(-1), live.log_softmax(-1)
    reconstructed_P, live_P = reconstructed_lp.exp(), live_lp.exp()
    lp_pass = bool(torch.allclose(reconstructed_lp, live_lp, atol=atol, rtol=rtol))
    p_pass = bool(torch.allclose(reconstructed_P, live_P, atol=probability_atol, rtol=probability_rtol))
    return {"status": "passed" if lp_pass and p_pass else "failed", "shape": list(live.shape),
            "elements_checked": live.numel(), "h_storage_dtype": str(record["h_raw_bf16"].dtype),
            "head_dtype": str(weight.dtype), "accumulation_dtype": "torch.float32",
            "output_multiplier": multiplier, "log_probability_atol": atol, "log_probability_rtol": rtol,
            "probability_atol": probability_atol, "probability_rtol": probability_rtol,
            "live_comparison": "stored_live_unit_logprob" if live_is_log_probability else "live_logits",
            "max_abs_logit_error": None if live_is_log_probability else float((reconstructed-live).abs().max()),
            "max_abs_l_unit_error": float((reconstructed_lp-live_lp).abs().max()),
            "max_abs_probability_error": float((reconstructed_P-live_P).abs().max()),
            "mean_abs_probability_error": float((reconstructed_P-live_P).abs().mean()),
            "log_probability_passed": lp_pass, "probability_passed": p_pass,
            "live_argmax_agreement": float((reconstructed.argmax(-1) == live.argmax(-1)).float().mean()),
            "head_metadata": _cpu_tree(metadata)}
