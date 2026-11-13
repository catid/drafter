"""M1-pre calibration and resumable M1a plain-DFlash trace collection."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from .features import HiddenProjection, fit_hidden_projection, fit_token_features
from .runtime import (ROOT, Manifest, atomic_json, common_parser, fingerprint, git_commit,
                      parse_config, preflight, record_run, resolved_config)
from .sampler import SamplingWarper
from .traces import TraceShardWriter, compact_block, read_shard, reconstruction_diagnostic


PARTITIONS = ("train_primary", "standard_eval", "diagnostic", "long")
STANDARD_ARMS = (("greedy", 0., SamplingWarper()),
                 ("recommended", .7, SamplingWarper(top_k=20, top_p=.8)),
                 ("pure_temperature_diagnostic", 1., SamplingWarper()))


def parser():
    p = common_parser(__doc__)
    p.add_argument("--stage", choices=["calibrate", "collect", "audit"], default="collect")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--partitions", nargs="+", choices=PARTITIONS, default=list(PARTITIONS))
    p.add_argument("--prompt_manifest", type=Path)
    p.add_argument("--train_tokens_per_temperature", type=int)
    p.add_argument("--standard_max_new_tokens", type=int, default=1024)
    p.add_argument("--long_max_new_tokens", type=int, default=24576)
    p.add_argument("--calibration_max_new_tokens", type=int, default=128)
    p.add_argument("--calibration_prompts", type=int, default=64)
    p.add_argument("--projection_dim", type=int, default=256)
    p.add_argument("--feature_rank", type=int, default=64)
    p.add_argument("--embedding_source", choices=["target_input", "target_output"], default="target_input")
    p.add_argument("--diagnostic_train_prompts", type=int, default=64)
    p.add_argument("--live_full_blocks", type=int, default=2)
    return p


def prepare_args(args):
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    for field in ("standard_max_new_tokens", "long_max_new_tokens", "calibration_max_new_tokens",
                  "calibration_prompts", "projection_dim", "feature_rank"):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")
    if args.live_full_blocks < 0 or args.diagnostic_train_prompts < 0:
        raise ValueError("diagnostic counts must be nonnegative")
    if args.train_tokens_per_temperature is None:
        args.train_tokens_per_temperature = {"smoke": 32, "reference": 250000, "full": 1000000}[args.scale]
    if args.train_tokens_per_temperature < 1:
        raise ValueError("train_tokens_per_temperature must be positive")
    if args.scale != "smoke" and args.calibration_prompts < 64:
        raise ValueError("reference calibration requires at least 64 train prompts")
    if args.scale == "smoke":
        # Tiny dimensions and generation caps are labelled synthetic, never used
        # to claim reference acceptance or satisfy the 0.5M-token floor.
        args.projection_dim = min(args.projection_dim, 6)
        args.feature_rank = min(args.feature_rank, 4)
        args.calibration_prompts = min(args.calibration_prompts, 8)
        args.calibration_max_new_tokens = min(args.calibration_max_new_tokens, 16)
        args.standard_max_new_tokens = min(args.standard_max_new_tokens, 32)
        args.long_max_new_tokens = min(args.long_max_new_tokens, 48)
        args.diagnostic_train_prompts = min(args.diagnostic_train_prompts, 2)
    args.prompt_manifest = Path(args.prompt_manifest) if args.prompt_manifest else args.output / "prompts.json"
    os.environ["RANK"] = str(args.rank)
    return args


def atomic_torch(path, payload):
    path = Path(path)
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_reference_gates(args):
    """Require actual complete M0 evidence, not just the presence of status files."""
    if args.scale == "smoke":
        return
    try:
        m0 = json.loads((args.output / "m0.json").read_text())
        numerics = json.loads((args.output / "m0_numerics.json").read_text())
    except FileNotFoundError as error:
        raise RuntimeError("blocked: M0 reproduction and numerical investigation must finish before M1") from error
    if (m0.get("scale") != "reference" or m0.get("status") != "passed"
            or m0.get("num_distinct_prompts", 0) < 20
            or m0["comparisons"]["upstream_local"]["mismatched_tokens"] != 0):
        raise RuntimeError("blocked: complete real M0 token reproduction gate has not passed")
    mismatch_ids = set()
    for path in (args.output / "m0").glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("local_target", {}).get("sequence_equal") is False:
            mismatch_ids.add(row["identity"]["prompt"])
    examined = {row["prompt_id"] for row in numerics.get("findings", [])}
    if (numerics.get("status") != "passed" or not numerics.get("head_reconstruction", {}).get("passed")
            or not mismatch_ids <= examined):
        raise RuntimeError("blocked: M0 numerics must cover every target-only mismatch and the real hidden/head hook")
    env = json.loads((ROOT / "artifacts/environment.json").read_text())
    for model in [args.target, args.drafter]:
        revision = env.get("model_artifacts", {}).get(model, {}).get("revision")
        if revision is None or numerics.get("model_revisions", {}).get(model) != revision:
            raise RuntimeError("blocked: M0 numerical evidence does not match the selected pinned model pair")


def synthetic_manifest(args):
    rows = []
    for split, count in [("train", 12), ("dev", 2), ("test", 2)]:
        for i in range(count):
            rows.append({"prompt_id": f"synthetic-{split}-{i}", "dataset": "builtin_synthetic",
                         "split": split, "condition": "standard_nonthinking", "reference_selected": True,
                         "text": f"Synthetic {split} {i}"})
    for split in ["dev", "test"]:
        rows.append({"prompt_id": f"synthetic-long-{split}", "dataset": "builtin_synthetic_long",
                     "split": split, "condition": "long_thinking", "reference_selected": True,
                     "text": f"Synthetic long {split}", "drafter_distribution": "synthetic_shift"})
    return {"scale": "smoke", "seed": args.seed, "prompts": rows,
            "calibration_ids": [row["prompt_id"] for row in rows if row["split"] == "train"][:args.calibration_prompts]}


def load_prompt_manifest(args):
    result = synthetic_manifest(args) if args.scale == "smoke" else json.loads(args.prompt_manifest.read_text())
    rows = result["prompts"]
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be globally unique across train/dev/test and conditions")
    by_id = {row["prompt_id"]: row for row in rows}
    if len(result["calibration_ids"]) < args.calibration_prompts:
        raise ValueError("manifest has too few designated calibration prompts")
    for id_ in result["calibration_ids"][:args.calibration_prompts]:
        row = by_id[id_]
        if row["split"] != "train" or row["condition"] != "standard_nonthinking":
            raise ValueError("calibration must use only designated standard_nonthinking train prompts")
    return result


class CollectionBackend:
    def __init__(self, args):
        self.args = args
        if args.scale == "smoke":
            from .toy import ToyTarget, ToyParallelDrafter
            self.target = ToyTarget(seed=args.seed)
            self.drafter = ToyParallelDrafter(self.target, seed=args.seed+1)
            self.model_pair = ["builtin/ToyTarget-v1", "builtin/ToyParallelDrafter-v1"]
            self.revisions = ["synthetic-v1", "synthetic-v1"]
            self.hidden_size = self.target.hidden_size
            self.vocab_size = self.target.vocab_size
            self.head = self.drafter.draft_lm_head
            self.special_tokens = []
            self.real = None
        else:
            from .backend import ReferenceBackend
            from .download import fetch_model
            target_path = fetch_model(args.target, args.allow_download)
            draft_path = fetch_model(args.drafter, args.allow_download)
            env = json.loads((ROOT / "artifacts/environment.json").read_text())
            self.model_pair = [args.target, args.drafter]
            self.revisions = [env["model_artifacts"][model]["revision"] for model in self.model_pair]
            self.real = ReferenceBackend.from_pretrained(target_path, draft_path, device=args.device,
                             target_revision=self.revisions[0], drafter_revision=self.revisions[1])
            self.target, self.drafter = self.real.target, self.real.drafter
            self.hidden_size = self.target.config.hidden_size
            self.vocab_size = self.target.config.vocab_size
            self.head = self.real.draft_lm_head
            self.special_tokens = self.real.tokenizer.all_special_ids

    def token_ids(self, row):
        if self.real:
            return self.real.tokenize(row["text"], enable_thinking=row["condition"] == "long_thinking")
        digest = hashlib.sha256(row["prompt_id"].encode()).digest()
        return torch.tensor([[int(x) % self.vocab_size for x in digest[:4]]])

    def generate(self, row, max_new_tokens, temperature, warper, seed, projection=None, on_block=None):
        from .dflash_loop import generate_reference, generate_toy
        ids = self.token_ids(row)
        generator = torch.Generator(device=ids.device).manual_seed(seed)
        if self.real:
            return generate_reference(self.real, ids, max_new_tokens, temperature=temperature,
                                      warper=warper, generator=generator, projection=projection, on_block=on_block)
        return generate_toy(self.target, self.drafter, ids[0].tolist(), max_new_tokens,
                            temperature=temperature, warper=warper, generator=generator,
                            projection=projection, on_block=on_block)

    def embedding(self, source):
        if source == "target_output":
            return self.head.weight
        return self.target.get_input_embeddings().weight if self.real else self.target.embedding


def _calibration_identity(args, prompts, backend):
    return {"model_pair": backend.model_pair, "checkpoint": backend.revisions, "seed": args.seed,
            "prompt_ids": prompts["calibration_ids"][:args.calibration_prompts],
            "projection_dim": args.projection_dim, "feature_rank": args.feature_rank,
            "embedding_source": args.embedding_source, "max_new_tokens": args.calibration_max_new_tokens,
            "scale": args.scale, "calibration_temperatures": [0., .7],
            "prompt_manifest_sha256": fingerprint(prompts) if args.scale == "smoke" else sha256(args.prompt_manifest),
            "calibration_warper_positive": asdict(STANDARD_ARMS[1][2])}


def calibration(args, prompts=None, backend=None):
    if args.rank != 0:
        raise ValueError("calibration has one owner: rank zero")
    check_reference_gates(args)
    prompts = load_prompt_manifest(args) if prompts is None else prompts
    backend = CollectionBackend(args) if backend is None else backend
    identity = _calibration_identity(args, prompts, backend)
    summary_path = args.output / "calibration.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        if prior.get("identity") == identity and prior.get("status") == "passed":
            if all((args.output / name).is_file() and sha256(args.output / name) == digest
                   for name, digest in prior["artifact_sha256"].items()):
                return prior
    by_id = {row["prompt_id"]: row for row in prompts["prompts"]}
    warmup_dir = args.output / "calibration_warmup"
    manifest = Manifest(warmup_dir / "manifest-rank0.json")
    sample_paths, token_ids = [], []
    for index, id_ in enumerate(identity["prompt_ids"]):
        row = by_id[id_]
        _, temperature, warper = STANDARD_ARMS[index % 2]
        job = {"model_pair": backend.model_pair, "prompt": id_, "condition": row["condition"],
               "seed": args.seed+index, "temperature": temperature,
               "verification_rule": "greedy" if temperature == 0 else "ratio",
               "proposal_mode": "plain_dflash_calibration", "checkpoint": backend.revisions,
               "adapter_version": 0, "calibration_identity": identity,
               "sampling_warp_config": asdict(warper)}
        path = warmup_dir / f"{manifest.key(job)}.pt"
        if not manifest.completed(job):
            h_batches = []
            def capture(block):
                h_batches.append(block["h_raw"].detach().cpu())
            result = backend.generate(row, args.calibration_max_new_tokens, temperature, warper,
                                      args.seed+index, on_block=capture)
            if not h_batches:
                raise RuntimeError(f"blocked: calibration prompt {id_} produced no DFlash hidden features")
            payload = {"identity": job, "h_raw": torch.cat(h_batches),
                       "token_ids": torch.tensor(backend.token_ids(row)[0].tolist() + result.output_tokens),
                       "generated_tokens": result.num_output_tokens}
            atomic_torch(path, payload)
            manifest.finish(job, path)
        saved = torch.load(path, weights_only=True, map_location="cpu")
        token_ids.append(saved["token_ids"])
        sample_paths.append(path)
        print(f"calibration {index+1}/{len(identity['prompt_ids'])}: {id_}", flush=True)
    batches = (torch.load(path, weights_only=True, map_location="cpu")["h_raw"] for path in sample_paths)
    projection = fit_hidden_projection(batches, backend.hidden_size, args.projection_dim, args.seed,
                                       prompt_ids=identity["prompt_ids"])
    projection.metadata.update({"frozen": True, "model_pair": backend.model_pair,
                                "model_revisions": backend.revisions, "warmup_identity": identity})
    projection.save(args.output / "h_proj.pt")
    E, feature_metadata = fit_token_features(backend.embedding(args.embedding_source), args.feature_rank,
                                            args.seed, calibration_token_ids=torch.cat(token_ids))
    feature_metadata.update({"embedding_source": args.embedding_source, "model_pair": backend.model_pair,
                             "model_revisions": backend.revisions, "calibration_prompt_ids": identity["prompt_ids"],
                             "calibration_split": "train", "calibration_condition": "standard_nonthinking",
                             "frozen": True, "scale": args.scale})
    atomic_torch(args.output / "token_features.pt", {"E": E.cpu(), "metadata": feature_metadata})
    result = {"scale": args.scale, "status": "passed", "identity": identity,
              "calibration_prompts": len(identity["prompt_ids"]), "hidden_vectors": projection.metadata["sample_count"],
              "artifact_sha256": {name: sha256(args.output / name) for name in ["h_proj.pt", "token_features.pt"]},
              "projection_metadata": {"artifact_id": str((args.output / "h_proj.pt").resolve()),
                                      **projection.metadata},
              "project_git_commit": git_commit(), "resolved_config": resolved_config(args)}
    result["projection_metadata"]["sha256"] = result["artifact_sha256"]["h_proj.pt"]
    atomic_json(summary_path, result)
    return result


def load_calibration(args, prompts, backend):
    path = args.output / "calibration.json"
    if not path.exists():
        raise RuntimeError("blocked: run --stage calibrate before writing M1a trace shards")
    calibration_record = json.loads(path.read_text())
    if calibration_record.get("status") != "passed" or calibration_record.get("identity") != _calibration_identity(args, prompts, backend):
        raise RuntimeError("blocked: frozen calibration does not match model pair, prompts, seed, or projection configuration")
    for name, digest in calibration_record["artifact_sha256"].items():
        if sha256(args.output / name) != digest:
            raise RuntimeError(f"blocked: frozen calibration artifact changed: {name}")
    loaded = HiddenProjection.load(args.output / "h_proj.pt")
    # Keep fixed projection tensors resident; no CPU-to-GPU matrix copy per block.
    projection = HiddenProjection(loaded.matrix.to(args.device), loaded.mean.to(args.device),
                                  loaded.std.to(args.device), loaded.metadata)
    return projection, calibration_record


def planned_jobs(args, prompts):
    """Shard each category's prompt list; never split a single generation across ranks."""
    rows = prompts["prompts"]
    standard = [row for row in rows if row["condition"] == "standard_nonthinking"]
    train = [row for row in standard if row["split"] == "train"]
    eval_rows = [row for row in standard if row["split"] in {"dev", "test"}
                 and row.get("reference_selected", True)]
    groups = []
    if "train_primary" in args.partitions:
        groups.append(("train_primary", train, STANDARD_ARMS[:2], args.standard_max_new_tokens))
    if "standard_eval" in args.partitions:
        groups.append(("standard_eval", eval_rows, STANDARD_ARMS[:2], args.standard_max_new_tokens))
    if "diagnostic" in args.partitions:
        groups.append(("diagnostic", train[:args.diagnostic_train_prompts] + eval_rows,
                       STANDARD_ARMS[2:], args.standard_max_new_tokens))
    if "long" in args.partitions:
        long_rows = [row for row in rows if row["condition"] in {"long_thinking", "long_nonthinking"}
                     and row.get("reference_selected", True)]
        for condition in ["long_thinking", "long_nonthinking"]:
            selected = [row for row in long_rows if row["condition"] == condition]
            arm = ("recommended", .6, SamplingWarper(top_k=20, top_p=.95)) if condition == "long_thinking" else STANDARD_ARMS[1]
            groups.append(("long", selected, [arm], args.long_max_new_tokens))
    for partition, selected, arms, max_new_tokens in groups:
        for index, row in enumerate(selected):
            if index % args.world_size != args.rank:
                continue
            for arm, temperature, warper in arms:
                # Stable across process count/partitions and distinct from diagnostics RNG.
                seed = args.seed + int(fingerprint({"prompt_id": row["prompt_id"]})[:8], 16) % 1_000_000
                yield {"partition": partition, "row": row, "arm": arm, "temperature": temperature,
                       "warper": warper, "seed": seed, "max_new_tokens": max_new_tokens}


def job_identity(args, job, backend, calibration_record):
    return {"model_pair": backend.model_pair, "prompt": job["row"]["prompt_id"],
            "condition": job["row"]["condition"], "seed": job["seed"], "temperature": job["temperature"],
            "verification_rule": "greedy" if job["temperature"] == 0 else "ratio", "proposal_mode": "plain_dflash",
            "checkpoint": backend.revisions, "adapter_version": 0, "sampling_warp_config": asdict(job["warper"]),
            "projection_sha256": calibration_record["artifact_sha256"]["h_proj.pt"],
            "token_features_sha256": calibration_record["artifact_sha256"]["token_features.pt"],
            "prompt_manifest_sha256": calibration_record["identity"]["prompt_manifest_sha256"],
            "trace_format": "compact_v1_per_split_live_full", "live_full_blocks_per_split": args.live_full_blocks,
            "scale": args.scale, "max_new_tokens": job["max_new_tokens"], "partition": job["partition"],
            "split": job["row"]["split"], "prompt_text_sha256": hashlib.sha256(job["row"]["text"].encode()).hexdigest()}


def collect(args, prompts=None, backend=None):
    check_reference_gates(args)
    prompts = load_prompt_manifest(args) if prompts is None else prompts
    backend = CollectionBackend(args) if backend is None else backend
    if args.scale == "smoke" and not (args.output / "calibration.json").exists():
        calibration(args, prompts, backend)
    projection, calibration_record = load_calibration(args, prompts, backend)
    from .diagnose import live_block_diagnostics, generation_diagnostics
    writer = TraceShardWriter(args.output / "traces", args.rank)
    quota = math.ceil(args.train_tokens_per_temperature / args.world_size)
    primary_tokens = Counter()
    rows, live_blocks_written = [], Counter()
    diagnostics_path = args.output / f"head_reconstruction-rank{args.rank}.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else []
    jobs = list(planned_jobs(args, prompts))
    # Read completed jobs first so an interrupted rank stops at its real emitted
    # budget instead of recollecting or crediting expected/max-length tokens.
    completed = {}
    for job in jobs:
        identity = job_identity(args, job, backend, calibration_record)
        if writer.completed(identity):
            sidecar = json.loads(writer.path(identity).with_suffix(".pt.json").read_text())
            generation = sidecar["generation"]
            completed[writer.manifest.key(identity)] = generation
            if job["partition"] == "train_primary":
                primary_tokens[job["temperature"]] += generation["realized_length"]
            live_blocks_written[generation["split"]] += generation.get("live_full_blocks", 0)
            # A crash after atomic shard publication but before the diagnostic
            # record must resume the check, not silently credit an unchecked shard.
            if generation.get("live_full_blocks", 0):
                restored = read_shard(writer.path(identity))
                for block in restored["blocks"]:
                    if "live_full" not in block:
                        continue
                    path_text = str(writer.path(identity))
                    previous = [check for check in diagnostics if check["shard"] == path_text
                                and check["round_idx"] == block["round_idx"] and check["status"] == "passed"]
                    if previous:
                        continue
                    check = reconstruction_diagnostic(block, backend.head)
                    diagnostics.append({"shard": path_text, "prompt_id": block["prompt_id"],
                                        "round_idx": block["round_idx"], **check})
                    atomic_json(diagnostics_path, diagnostics)
                    if check["status"] != "passed":
                        raise RuntimeError("blocked: resumed heldout shard failed head reconstruction")
    for job in jobs:
        identity = job_identity(args, job, backend, calibration_record)
        key = writer.manifest.key(identity)
        if key in completed:
            rows.append(completed[key])
            continue
        if job["partition"] == "train_primary" and primary_tokens[job["temperature"]] >= quota:
            continue
        row = job["row"]
        metadata = {"prompt_id": row["prompt_id"], "split": row["split"], "condition": row["condition"],
                    "seed": job["seed"], "scale": args.scale, "model_pair": backend.model_pair,
                    "projection_metadata": calibration_record["projection_metadata"], "history_depth": 4,
                    "dataset": row["dataset"], "dataset_revision": prompts.get("dataset_revisions", {}).get(row["dataset"]),
                    "drafter_distribution": row.get("drafter_distribution", "in_distribution_nonthinking"),
                    "collection_arm": job["arm"], "partition": job["partition"]}
        blocks, live_count = [], 0
        def capture(block):
            nonlocal live_count
            block["diagnostics"] = live_block_diagnostics(block, seed=job["seed"] + 10_000_000 + block["round_idx"])
            retain = (row["condition"] == "standard_nonthinking" and row["split"] in {"dev", "test"}
                      and live_blocks_written[row["split"]] < args.live_full_blocks)
            blocks.append(compact_block(block, metadata, special_tokens=backend.special_tokens, retain_live_full=retain))
            live_blocks_written[row["split"]] += int(retain)
            live_count += int(retain)
        generated = backend.generate(row, job["max_new_tokens"], job["temperature"], job["warper"],
                                     job["seed"], projection=projection, on_block=capture)
        if not blocks:
            raise RuntimeError(f"blocked: prompt {row['prompt_id']} completed without a DFlash verification block")
        tokens = generated.output_tokens
        eos = bool(backend.real and tokens and tokens[-1] in backend.real.stop_token_ids)
        generation = {**generation_diagnostics(tokens), "prompt_id": row["prompt_id"], "split": row["split"],
                      "condition": row["condition"], "seed": job["seed"], "temperature": job["temperature"],
                      "sampling_warp_config": asdict(job["warper"]), "partition": job["partition"],
                      "output_tokens": tokens, "realized_length": len(tokens), "eos_reached": eos,
                      "capped": len(tokens) >= job["max_new_tokens"] and not eos, "complete": True,
                      "max_new_tokens": job["max_new_tokens"], "live_full_blocks": live_count,
                      "elapsed_seconds": generated.elapsed_seconds, "tokens_per_second": generated.tokens_per_second,
                      "scale": args.scale, "data": "synthetic" if args.scale == "smoke" else "reference",
                      "long_output_policy": "natural EOS or cap; no forced min_new_tokens or padding"}
        path = writer.write(identity, blocks, {"generation": generation, "resolved_config": resolved_config(args)})
        if live_count:
            restored = read_shard(path)
            for block in restored["blocks"]:
                if "live_full" not in block:
                    continue
                check = reconstruction_diagnostic(block, backend.head)
                diagnostics.append({"shard": str(path), "prompt_id": row["prompt_id"],
                                    "round_idx": block["round_idx"], **check})
                if check["status"] != "passed":
                    atomic_json(diagnostics_path, diagnostics)
                    raise RuntimeError("blocked: serialized heldout shard failed draft head reconstruction tolerance")
            atomic_json(diagnostics_path, diagnostics)
        rows.append(generation)
        if job["partition"] == "train_primary":
            primary_tokens[job["temperature"]] += len(tokens)
        print(f"rank {args.rank} {job['partition']} {row['prompt_id']} T={job['temperature']}: "
              f"{len(tokens)} tokens; primary={dict(primary_tokens)}", flush=True)
        # Rank progress is atomic and authoritative for completed output counts.
        atomic_json(args.output / f"collection-rank{args.rank}.json", {
            "scale": args.scale, "status": "running", "rank": args.rank, "world_size": args.world_size,
            "primary_tokens_by_temperature": dict(primary_tokens), "quota_per_temperature": quota,
            "completed_generations": len(rows), "partitions": args.partitions})
    missing = [T for T in [0., .7] if primary_tokens[T] < quota] if "train_primary" in args.partitions else []
    summary = {"scale": args.scale, "status": "blocked" if missing else "passed", "rank": args.rank,
               "world_size": args.world_size, "partitions": args.partitions,
               "primary_tokens_by_temperature": dict(primary_tokens), "quota_per_temperature": quota,
               "completed_generations": len(rows), "live_full_blocks_by_split": dict(live_blocks_written),
               "blocker": f"all assigned train prompts exhausted before token quota at T={missing}" if missing else None,
               "resolved_config": resolved_config(args), "project_git_commit": git_commit()}
    atomic_json(args.output / f"collection-rank{args.rank}.json", summary)
    if missing:
        raise RuntimeError(f"blocked: {summary['blocker']}")
    return summary


def audit(args):
    """Aggregate actual committed outputs; never count diagnostic/long tokens as training."""
    check_reference_gates(args)
    prompt_manifest = load_prompt_manifest(args)
    calibration_record = json.loads((args.output / "calibration.json").read_text())
    frozen = calibration_record["identity"]
    backend_identity = SimpleNamespace(model_pair=frozen["model_pair"], revisions=frozen["checkpoint"])
    if frozen != _calibration_identity(args, prompt_manifest, backend_identity):
        raise RuntimeError("blocked: audit configuration/manifest differs from frozen calibration")
    for name, digest in calibration_record["artifact_sha256"].items():
        if sha256(args.output / name) != digest:
            raise RuntimeError(f"blocked: calibration artifact checksum mismatch: {name}")
    # Rank assignment is deliberately absent from job identity. Build the full
    # expected set once to exclude superseded configurations and duplicate jobs
    # left by a run resumed with a different process count.
    all_args = copy.copy(args)
    all_args.rank, all_args.world_size, all_args.partitions = 0, 1, list(PARTITIONS)
    expected = {fingerprint(job_identity(args, job, backend_identity, calibration_record)):
                job_identity(args, job, backend_identity, calibration_record)
                for job in planned_jobs(all_args, prompt_manifest)}
    totals = Counter()
    prompts = defaultdict(set)
    generations, shards, head_checks, excluded = [], [], [], []
    seen = {}
    for manifest_path in sorted((args.output / "traces").glob("manifest-rank*.json")):
        manifest = json.loads(manifest_path.read_text())
        for key, job in manifest.get("jobs", {}).items():
            if job.get("status") != "complete":
                continue
            canonical_key = fingerprint(job["identity"])
            if canonical_key not in expected:
                excluded.append({"path": job["result_path"], "reason": "superseded_or_unselected_identity"})
                continue
            path = Path(job["result_path"])
            sidecar_path = path.with_suffix(".pt.json")
            if not path.is_file() or not sidecar_path.is_file():
                raise RuntimeError(f"blocked: missing committed trace shard {path}")
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar["disk_bytes"] != path.stat().st_size or sidecar["sha256"] != sha256(path):
                raise RuntimeError(f"blocked: corrupted committed trace shard {path}")
            gen = sidecar["generation"]
            if gen["scale"] != args.scale:
                continue
            if not gen["complete"]:
                continue
            if sidecar.get("identity") != job["identity"]:
                raise RuntimeError(f"blocked: shard/manifest identity mismatch: {path}")
            output_hash = fingerprint(gen["output_tokens"])
            if canonical_key in seen:
                if seen[canonical_key] != output_hash:
                    raise RuntimeError(f"blocked: duplicate job identities have different generated outputs: {path}")
                excluded.append({"path": str(path), "reason": "duplicate_job_identity"})
                continue
            seen[canonical_key] = output_hash
            condition, split, T = gen["condition"], gen["split"], gen["temperature"]
            category = f"{condition}/{split}/T={T}"
            totals[category] += gen["realized_length"]
            prompts[category].add(gen["prompt_id"])
            if condition == "standard_nonthinking" and split == "train" and gen["partition"] == "train_primary":
                if T == 0 or (T == .7 and gen["sampling_warp_config"] == asdict(STANDARD_ARMS[1][2])):
                    totals["primary_train_total"] += gen["realized_length"]
                    totals[f"primary_train_T={T}"] += gen["realized_length"]
            generations.append({k: v for k, v in gen.items() if k != "output_tokens"})
            shards.append(str(path))
    for path in args.output.glob("head_reconstruction-rank*.json"):
        head_checks.extend(row for row in json.loads(path.read_text()) if row["shard"] in shards)
    floors = {"train_tokens": 64 if args.scale == "smoke" else 500000 if args.scale == "reference" else 2000000,
              "standard_test_prompts": 2 if args.scale == "smoke" else 64,
              "long_dev_prompts": 1 if args.scale == "smoke" else 12 if args.scale == "reference" else 32,
              "long_test_prompts": 1 if args.scale == "smoke" else 32 if args.scale == "reference" else 96}
    requirements = {"primary_train_floor": totals["primary_train_total"] >= floors["train_tokens"],
                    "each_primary_temperature_floor": all(totals[f"primary_train_T={T}"] >= floors["train_tokens"]//2 for T in [0., .7]),
                    "standard_test_each_temperature": all(len(prompts[f"standard_nonthinking/test/T={T}"]) >= floors["standard_test_prompts"] for T in [0., .7, 1.]),
                    "standard_dev_present": all(len(prompts[f"standard_nonthinking/dev/T={T}"]) >= (2 if args.scale == "smoke" else 64) for T in [0., .7]),
                    "diagnostic_train_separate": bool(prompts["standard_nonthinking/train/T=1.0"]),
                    "serialized_head_reconstruction": bool(head_checks) and all(row["status"] == "passed" for row in head_checks)}
    long_condition = "long_thinking" if any(g["condition"] == "long_thinking" for g in generations) else "long_nonthinking"
    long_T = .6 if long_condition == "long_thinking" else .7
    requirements["long_dev_generations"] = len(prompts[f"{long_condition}/dev/T={long_T}"]) >= floors["long_dev_prompts"]
    requirements["long_test_generations"] = len(prompts[f"{long_condition}/test/T={long_T}"]) >= floors["long_test_prompts"]
    split_ids = defaultdict(set)
    for generation in generations:
        split_ids[generation["split"]].add(generation["prompt_id"])
    requirements["disjoint_prompt_splits"] = not (split_ids["train"] & split_ids["dev"] or split_ids["train"] & split_ids["test"] or split_ids["dev"] & split_ids["test"])
    selected_test = {row["prompt_id"] for row in prompt_manifest["prompts"]
                     if row["condition"] == "standard_nonthinking" and row["split"] == "test"
                     and row.get("reference_selected", True)}
    requirements["selected_standard_test_ids"] = all(selected_test <= prompts[f"standard_nonthinking/test/T={T}"] for T in [0., .7, 1.])
    heldout = prompt_manifest.get("complete_heldout_dataset")
    requirements["complete_dataset_held_out"] = args.scale == "smoke" or (bool(heldout) and all(
        row["split"] == "test" for row in prompt_manifest["prompts"] if row["dataset"] == heldout))
    requirements["live_full_dev_and_test"] = all(any(g["split"] == split and g.get("live_full_blocks", 0) > 0 for g in generations) for split in ["dev", "test"])
    report = {"scale": args.scale, "status": "passed" if all(requirements.values()) else "incomplete",
              "requirements": requirements, "floors": floors, "token_counts": dict(totals),
              "prompt_counts": {k: len(v) for k, v in prompts.items()}, "shards": shards,
              "generations": generations, "long_condition": long_condition, "head_reconstruction": head_checks,
              "excluded_shards": excluded, "calibration": calibration_record,
              "project_git_commit": git_commit()}
    atomic_json(args.output / "m1_audit.json", report)
    return report


def main(argv=None):
    args = prepare_args(parse_config(parser(), argv))
    torch.set_num_threads(1)
    record_run(args, f"collect-{args.stage}")
    if args.stage == "audit":
        report = audit(args)
    else:
        check_reference_gates(args)
        if args.scale != "smoke":
            expected = (args.calibration_prompts * args.calibration_max_new_tokens if args.stage == "calibrate"
                        else 2 * args.train_tokens_per_temperature / args.world_size)
            preflight(args, math.ceil(expected))
        report = calibration(args) if args.stage == "calibrate" else collect(args)
    print(json.dumps({k:v for k,v in report.items() if k not in {"generations", "shards", "head_reconstruction"}}, indent=2))
    return report


if __name__ == "__main__":
    main()
