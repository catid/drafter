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

_CACHED_ROOT_GIT_COMMIT: str | None = None
_ORIG_GIT_COMMIT = git_commit
_SPECIALS_TENSOR_CACHE: dict[tuple[torch.device, tuple[int, ...]], torch.Tensor] = {}


def _default_git_commit() -> str:
    global _CACHED_ROOT_GIT_COMMIT
    if git_commit is not _ORIG_GIT_COMMIT:
        return git_commit()
    if _CACHED_ROOT_GIT_COMMIT is None:
        _CACHED_ROOT_GIT_COMMIT = git_commit()
    return _CACHED_ROOT_GIT_COMMIT


def _all_finite(x: torch.Tensor) -> bool:
    if x.numel() == 0:
        return True
    if bool(torch.isfinite(x.sum(dtype=torch.float32))):
        return True
    return bool(torch.isfinite(x).all())


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


def _all_tensors_finite(tensors: Sequence[torch.Tensor]) -> bool:
    non_empty = [t for t in tensors if t.numel() > 0]
    if not non_empty:
        return True
    combined = non_empty[0].sum(dtype=torch.float32)
    for t in non_empty[1:]:
        combined = combined + t.sum(dtype=torch.float32)
    if bool(torch.isfinite(combined)):
        return True
    return all(_all_finite(t) for t in non_empty)


def compact_block(record: Mapping, metadata: Mapping, M_draft: int = 256, M_target: int = 64,
                  special_tokens: Sequence[int] = (), retain_live_full: bool = False,
                  *, _trust_l_unit: bool = False,
                  _precomputed_p_train: torch.Tensor | None = None) -> dict:
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
    l_in = record.get("l_unit") if _trust_l_unit else None
    reused_lu = (
        l_in.detach().float()
        if isinstance(l_in, torch.Tensor) and l_in.shape == draft_logits.shape and l_in.device == draft_logits.device
        else None
    )
    K, V = draft_logits.shape
    if K < 1 or len(record["draft_tokens"]) != K:
        raise ValueError("one realized draft token is required per logical slot")
    specials = sorted(set(int(token) for token in special_tokens))
    if any(token < 0 or token >= V for token in specials):
        raise ValueError("special token outside vocabulary")
    tokens_int = [int(token) for token in record["draft_tokens"]]
    if any(token < 0 or token >= V for token in tokens_int):
        raise ValueError("draft token outside vocabulary")
    # Match the live full training distribution; never use inference warpers.
    l_unit = draft_logits.log_softmax(-1) if reused_lu is None else reused_lu
    if (
        _precomputed_p_train is not None
        and _precomputed_p_train.shape == target_logits.shape
        and _precomputed_p_train.device == target_logits.device
    ):
        P_train = _precomputed_p_train
    else:
        P_train = target_logits.softmax(-1)
    top_draft_vals, top_draft = draft_logits.topk(min(M_draft, V), dim=-1)
    top_target_vals, top_target = target_logits.topk(min(M_target, V), dim=-1)
    draft_log_normalizer = top_draft_vals[:, 0] - l_unit.gather(-1, top_draft[:, :1]).squeeze(-1)
    target_log_normalizer = top_target_vals[:, 0] - P_train.gather(-1, top_target[:, :1]).squeeze(-1).log()
    tok_dev = torch.tensor(tokens_int, dtype=torch.long, device=draft_logits.device)[:, None]
    if specials:
        sp_key = (draft_logits.device, tuple(specials))
        sp_dev = _SPECIALS_TENSOR_CACHE.get(sp_key)
        if sp_dev is None:
            sp_dev = torch.tensor(specials, dtype=torch.long, device=draft_logits.device)
            _SPECIALS_TENSOR_CACHE[sp_key] = sp_dev
        cand_parts = [top_draft, top_target, tok_dev, sp_dev.unsqueeze(0).expand(K, -1)]
    else:
        cand_parts = [top_draft, top_target, tok_dev]
    cands_sorted = torch.cat(cand_parts, dim=-1).sort(dim=-1).values
    C = cands_sorted.shape[1]
    fin_ok = torch.isfinite(draft_logits).all() & torch.isfinite(target_logits).all()
    if reused_lu is not None:
        fin_ok = fin_ok & torch.isfinite(reused_lu).all()
    float_pack_cpu = torch.cat(
        (
            l_unit.gather(-1, cands_sorted),
            P_train.gather(-1, cands_sorted),
            draft_log_normalizer[:, None],
            target_log_normalizer[:, None],
            fin_ok.to(dtype=draft_log_normalizer.dtype).reshape(1, 1).expand(K, 1),
        ),
        dim=-1,
    ).cpu()
    if not bool(torch.isfinite(float_pack_cpu).all()) or float(float_pack_cpu[0, 2 * C + 2].item()) != 1.0:
        raise ValueError("training logits must be finite and unwarped; truncation masks are forbidden")
    l_gath_cpu = float_pack_cpu[:, :C]
    p_gath_cpu = float_pack_cpu[:, C:2 * C]
    draft_lse_cpu = float_pack_cpu[:, 2 * C].clone()
    target_lse_cpu = float_pack_cpu[:, 2 * C + 1].clone()
    cands_cpu = cands_sorted.cpu()
    uniq_mask = torch.ones_like(cands_cpu, dtype=torch.bool)
    if cands_cpu.shape[1] > 1:
        uniq_mask[:, 1:] = cands_cpu[:, 1:] != cands_cpu[:, :-1]
    union_ids, draft_union, target_union, rest = [], [], [], []
    for i in range(K):
        row_mask = uniq_mask[i]
        ids = cands_cpu[i][row_mask]
        lp = l_gath_cpu[i][row_mask]
        p = p_gath_cpu[i][row_mask]
        union_ids.append(ids)
        draft_union.append(lp)
        target_union.append(p)
        # FP64 sum avoids losing a small tail while accumulating compact FP32
        # entries. Tiny negative rounding errors are clipped and validated.
        # A full-vocabulary union has no tail category. FP32 normalization
        # roundoff must not invent positive target mass on an empty q support.
        rest.append(torch.tensor(0., dtype=torch.float32) if len(ids) == V
                    else (1 - p.double().sum()).clamp_min(0).float())
    proj_commit = metadata["project_git_commit"] if "project_git_commit" in metadata else _default_git_commit()
    result = _cpu_tree(dict(metadata))
    result.update({key: _cpu_tree(record[key]) for key in _COPY_FIELDS})
    result.update({key: _cpu_tree(record[key]) for key in _OPTIONAL_FIELDS if key in record})
    result.update({"schema_version": SCHEMA_VERSION, "vocab_size": V, "block_length": K,
                   "M_draft": M_draft, "M_target": M_target, "special_tokens": specials,
                   "union_ids": union_ids, "draft_l_unit_union": draft_union,
                   "target_probs_union": target_union, "target_rest_mass": torch.stack(rest),
                   "draft_log_normalizer": draft_lse_cpu,
                   "target_log_normalizer": target_lse_cpu,
                   "target_distribution": "unwarped_full_softmax_unit_temperature",
                   "loss_mode": "union_rest_grouped", "beta_train": 1.,
                   "project_git_commit": proj_commit})
    result["h_raw_bf16"] = result["h_raw_bf16"].to(torch.bfloat16)
    result["h"] = result["h"].float()
    if retain_live_full:
        result["live_full"] = {"draft_l_unit": l_unit.detach().cpu().clone(),
                               "target_probs": P_train.detach().cpu().clone(),
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
        if value.ndim != 2 or value.shape[0] != K or not _all_finite(value):
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
    union_ids_list = block["union_ids"]
    draft_lp_list = block["draft_l_unit_union"]
    target_p_list = block["target_probs_union"]
    lengths_list = []
    for ids in union_ids_list:
        if ids.ndim != 1 or ids.dtype != torch.int64 or not len(ids):
            raise ValueError("union IDs must be nonempty int64 vectors")
        lengths_list.append(len(ids))
    all_ids = torch.cat(union_ids_list)
    dev = all_ids.device
    lengths_t = torch.tensor(lengths_list, dtype=torch.long, device=dev)
    ends = lengths_t.cumsum(0)
    starts = ends - lengths_t
    if all_ids.numel() > K:
        same_row = torch.ones(all_ids.numel() - 1, dtype=torch.bool, device=dev)
        if K > 1:
            same_row[ends[:-1] - 1] = False
        mono_ok = bool((all_ids[1:][same_row] > all_ids[:-1][same_row]).all())
    else:
        mono_ok = True
    if not mono_ok or int(all_ids[starts].min()) < 0 or int(all_ids[ends - 1].max()) >= V:
        raise ValueError("union IDs must be unique, sorted, and in vocabulary")
    for i, ids in enumerate(union_ids_list):
        if draft_lp_list[i].shape != ids.shape or target_p_list[i].shape != ids.shape:
            raise ValueError("union probability shapes do not match IDs")
    all_lp = torch.cat(draft_lp_list)
    all_P = torch.cat(target_p_list)
    rest_raw = block["target_rest_mass"]
    rest_t = rest_raw if isinstance(rest_raw, torch.Tensor) else torch.stack([torch.as_tensor(x, device=dev) for x in rest_raw])
    if not (_all_finite(all_lp) and _all_finite(all_P) and _all_finite(rest_t)):
        raise ValueError("nonfinite compact trace probabilities")
    if bool((all_lp > mass_atol).any() or (all_P < 0).any() or (all_P > 1).any()
            or (rest_t < 0).any() or (rest_t > 1).any()):
        raise ValueError("invalid probability or log-probability range")
    p_sums = torch.segment_reduce(all_P.double(), "sum", lengths=lengths_t)
    if bool(((p_sums + rest_t.double() - 1.0).abs() > mass_atol).any()):
        raise ValueError("target union probabilities and unrenormalized rest must sum to one")
    row_ids = torch.repeat_interleave(torch.arange(K, dtype=torch.long, device=dev), lengths_t)
    global_keys = row_ids * V + all_ids
    draft_toks = [int(t) for t in block["draft_tokens"]]
    if any(t < 0 or t >= V for t in draft_toks):
        raise ValueError("union omits the realized draft token")
    draft_queries = torch.arange(K, dtype=torch.long, device=dev) * V + torch.tensor(draft_toks, dtype=torch.long, device=dev)
    draft_pos = torch.searchsorted(global_keys, draft_queries).clamp_max(global_keys.numel() - 1)
    if not bool((global_keys[draft_pos] == draft_queries).all()):
        raise ValueError("union omits the realized draft token")
    specials = [int(token) for token in block.get("special_tokens", [])]
    if specials:
        if any(t < 0 or t >= V for t in specials):
            raise ValueError("union omits a requested special token")
        spec_t = torch.tensor(specials, dtype=torch.long, device=dev)
        spec_queries = (torch.arange(K, dtype=torch.long, device=dev)[:, None] * V + spec_t[None, :]).reshape(-1)
        spec_pos = torch.searchsorted(global_keys, spec_queries).clamp_max(global_keys.numel() - 1)
        if not bool((global_keys[spec_pos] == spec_queries).all()):
            raise ValueError("union omits a requested special token")
    if "live_full" in block:
        live = block["live_full"]
        if live["target_probs"].shape != (K, V) or live["draft_l_unit"].shape != (K, V):
            raise ValueError("live_full diagnostic tensors must cover the full vocabulary")
        torch.testing.assert_close(live["target_probs"][row_ids, all_ids], all_P, rtol=0, atol=0)
        torch.testing.assert_close(live["draft_l_unit"][row_ids, all_ids], all_lp, rtol=0, atol=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _HashingWriter:
    """Stream wrapper computing SHA256 in-flight during sequential `torch.save` writes."""

    __slots__ = ("_raw", "_digest", "bytes_written", "seeked")

    def __init__(self, raw):
        self._raw = raw
        self._digest = hashlib.sha256()
        self.bytes_written = 0
        self.seeked = False

    def write(self, data) -> int:
        mv = memoryview(data)
        byte_mv = mv.cast("B") if mv.c_contiguous else memoryview(mv.tobytes())
        written = self._raw.write(byte_mv)
        if written is None:
            written = len(byte_mv)
        if written < 0 or written > len(byte_mv):
            self.seeked = True
        elif not self.seeked and written > 0:
            self._digest.update(byte_mv[:written])
        self.bytes_written += max(0, written)
        return written

    def flush(self) -> None:
        self._raw.flush()

    def fileno(self) -> int:
        return self._raw.fileno()

    def tell(self) -> int:
        return self._raw.tell()

    def seek(self, *args):
        self.seeked = True
        return self._raw.seek(*args)

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def write_shard(
    path: str | Path,
    blocks: Sequence[Mapping],
    metadata: Mapping,
    *,
    _prevalidated_cpu: bool = False,
) -> dict:
    """Write all bytes before publishing the shard and its checksum sidecar."""
    path = Path(path)
    if not blocks:
        raise ValueError("cannot write an empty trace shard")
    if _prevalidated_cpu:
        blocks_payload = list(blocks)
    else:
        for block in blocks:
            validate_block(block)
        blocks_payload = _cpu_tree(blocks)
    payload = {"schema_version": SCHEMA_VERSION, "metadata": _cpu_tree(metadata),
               "blocks": blocks_payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    in_flight_sha: str | None = None
    in_flight_size: int | None = None
    try:
        with os.fdopen(fd, "wb") as stream:
            hashing_stream = _HashingWriter(stream)
            torch.save(payload, hashing_stream)
            hashing_stream.flush()
            os.fsync(hashing_stream.fileno())
            if not hashing_stream.seeked:
                in_flight_sha = hashing_stream.hexdigest
                in_flight_size = hashing_stream.bytes_written
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    disk_bytes = path.stat().st_size
    digest_hex = (
        in_flight_sha
        if (in_flight_sha is not None and in_flight_size == disk_bytes)
        else _sha256(path)
    )
    summary = {"schema_version": SCHEMA_VERSION, "path": str(path.resolve()),
               "sha256": digest_hex, "disk_bytes": disk_bytes,
               "blocks": len(blocks), "logical_slots": sum(b["block_length"] for b in blocks)}
    for key in ("generation", "identity", "rank"):
        if key in metadata:
            summary[key] = metadata[key]
    atomic_json(path.with_suffix(path.suffix + ".json"), summary)
    return summary


def read_shard(path: str | Path, verify_checksum: bool = True, *, validate: bool = True) -> dict:
    path = Path(path)
    if verify_checksum:
        sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if path.stat().st_size != sidecar["disk_bytes"] or _sha256(path) != sidecar["sha256"]:
            raise ValueError(f"trace shard checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION or not payload.get("blocks"):
        raise ValueError("invalid or empty trace shard")
    if validate:
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

    def write(
        self,
        identity: Mapping,
        blocks: Sequence[Mapping],
        metadata: Mapping | None = None,
        *,
        _prevalidated_cpu: bool = False,
    ) -> Path:
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
        header.update({"identity": dict(identity), "rank": self.rank, "project_git_commit": _default_git_commit()})
        write_shard(path, blocks, header, _prevalidated_cpu=_prevalidated_cpu)
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
