"""M2 streamed, resumable offline fitting on unwarped compact DFlash traces.

The only primary fitting contexts are standard non-thinking greedy and the
inspected model-card warped T=.7 arm. Long conditions and pure-T=1 diagnostics
never enter the primary optimizer. Real fitting requires the M1 audit gate.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import threading
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .adapter import (DenseAdapter, PrecollatedGroupedTargets, collate_grouped_targets,
                      loss_weights, next_token, prev_token, two_sided_pseudolikelihood_loss,
                      union_rest_grouped, union_rest_grouped_precollated)
from .fastpath import _fresh_compile
from .features import HiddenProjection
from .optimizers import AdamATan2, AnchorMuon, PaceMuon, compute_lr_multiplier
from .runtime import (ROOT, atomic_json, common_parser, fingerprint, git_commit, parse_config,
                      preflight, record_run, resolve_artifact_path, resolved_config)
from .sparse import ExactObservedPairs, SignedHash, hash_keys, pack_exact_keys
from .sparse_stats import SparseAccessRecorder, pack_keys
from .traces import read_shard

TRANSIENT_CONFIG_KEYS = {
    'resume', 'rank', 'world_size', 'dist_backend', 'microbatch_blocks',
    'num_workers', 'pin_memory', 'compile', 'stats_memory_limit', 'stats_benchmark_entries',
}
OPT_IMPL_DEFAULT = {
    'bf16_gemm': False,
    'optimizer': 'adamw',
    'lr_schedule': 'constant',
    'ema_eval': None,
    'pace_pullback': 0.05,
    'anchor_soda': 0.05,
    'block_shuffle_window': 0,
}
DEFAULT_OPTIONAL_CONFIG = {
    **OPT_IMPL_DEFAULT,
    'sparse_backend': 'signed_feature_hash',
    'sparse_overflow_policy': 'freeze_new',
    'objective': 'union_rest_grouped',
    'allow_long_oracle_fit': False,
    'min_long_oracle_train_tokens': 50_000,
    'allow_pure_t1_ablation': False,
    'max_primary_train_tokens': 4_000_000,
    'calibrate_beta': False,
    'apply_calibrated_beta': False,
    'calibrate_beta_grid': None,
    'scale_checkpoint_from': None,
    'scale_checkpoint_to': None,
    'checkpoint_scale': None,
    'm5_fit_tag': None,
}


def _comm_device(device: torch.device) -> torch.device:
    if dist.is_available() and dist.is_initialized() and dist.get_backend() == 'nccl':
        return device
    return torch.device('cpu')


def _all_gather_1d(tensor: torch.Tensor, world_size: int, device: torch.device) -> list[torch.Tensor]:
    comm_dev = _comm_device(device)
    local = tensor.to(comm_dev).reshape(-1)
    local_len = torch.tensor([local.numel()], dtype=torch.int64, device=comm_dev)
    lengths = [torch.zeros(1, dtype=torch.int64, device=comm_dev) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    counts = [int(x.item()) for x in lengths]
    max_len = max(counts, default=0)
    if max_len == 0:
        return [torch.empty(0, dtype=tensor.dtype, device=device) for _ in range(world_size)]
    padded = torch.zeros(max_len, dtype=local.dtype, device=comm_dev)
    if local.numel():
        padded[:local.numel()] = local
    gathered = [torch.zeros(max_len, dtype=local.dtype, device=comm_dev) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return [buf[:n].to(device) for buf, n in zip(gathered, counts)]


class _CompactTrainSparse:
    """Differentiate only unique touched buckets per microbatch instead of 2**24 floats."""
    def __init__(self, base: SignedHash):
        self.base = base
        self.capacity = base.capacity
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def lookup(self, d, previous, current):
        buckets, signs = hash_keys(d, previous, current, self.capacity, self.base.values.device)
        unique, inverse = torch.unique(buckets.reshape(-1), sorted=True, return_inverse=True)
        touched = self.base.values.index_select(0, unique).detach().requires_grad_(True)
        if unique.numel():
            self.calls.append((unique, touched))
        return touched[inverse].reshape(buckets.shape) * signs

    def grad_pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (u, t.grad.detach() if t.grad is not None else torch.zeros_like(t))
            for u, t in self.calls
        ]


class _CompactTrainExactSparse:
    """Differentiate only unique touched exact `(d, prev, curr)` keys per microbatch."""
    def __init__(self, base: ExactObservedPairs):
        self.base = base
        self.raw_calls: list[torch.Tensor] = []
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def lookup(self, d, previous, current):
        packed = pack_exact_keys(d, previous, current, self.base.vocab_size, device=self.base.values.device)
        flat = packed.reshape(-1)
        if flat.numel():
            self.raw_calls.append(flat.detach())
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        touched = self.base.lookup_packed(unique, insert_missing=False).detach().requires_grad_(True)
        if unique.numel():
            self.calls.append((unique, touched))
        return touched[inverse].reshape(packed.shape)

    def grad_pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (u, t.grad.detach() if t.grad is not None else torch.zeros_like(t))
            for u, t in self.calls
        ]


def _reduce_sparse_pairs(pairs: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device):
    if not pairs:
        return (torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.float32, device=device))
    if len(pairs) == 1:
        return pairs[0][0].to(device=device, dtype=torch.int64), pairs[0][1].to(device=device, dtype=torch.float32)
    all_idx = torch.cat([idx.to(device=device, dtype=torch.int64).reshape(-1) for idx, _ in pairs])
    all_grad = torch.cat([grad.to(device=device, dtype=torch.float32).reshape(-1) for _, grad in pairs])
    if all_idx.numel() == 0:
        return all_idx, all_grad
    unique, inverse = torch.unique(all_idx, sorted=True, return_inverse=True)
    summed = torch.zeros(len(unique), dtype=torch.float32, device=device)
    summed.scatter_add_(0, inverse, all_grad)
    return unique, summed


def primary_eligible(record, *, split=None):
    if split is not None and record.get('split') != split:
        return False
    if record.get('condition') != 'standard_nonthinking':
        return False
    temperature = record.get('collection_temperature', record.get('temperature'))
    if temperature == 0:
        return True
    warp = record.get('sampling_warp_config', {})
    return (temperature is not None and abs(temperature - .7) < 1e-10
            and warp.get('top_k') == 20 and warp.get('top_p') == .8 and warp.get('min_p', 0.) == 0.)


def long_oracle_eligible(record, *, split=None):
    if split is not None and record.get('split') != split:
        return False
    if record.get('condition') not in {'long_thinking', 'long_nonthinking'}:
        return False
    if record.get('repetition_dominated', False):
        return False
    temperature = record.get('collection_temperature', record.get('temperature'))
    warp = record.get('sampling_warp_config', {})
    if record.get('condition') == 'long_thinking':
        return (temperature is not None and abs(temperature - .6) < 1e-10
                and warp.get('top_k') == 20 and warp.get('top_p') == .95 and warp.get('min_p', 0.) == 0.)
    return (temperature is not None and abs(temperature - .7) < 1e-10
            and warp.get('top_k') == 20 and warp.get('top_p') == .8 and warp.get('min_p', 0.) == 0.)


def pure_t1_eligible(record, *, split=None):
    if split is not None and record.get('split') != split:
        return False
    if record.get('condition') != 'standard_nonthinking':
        return False
    temperature = record.get('collection_temperature', record.get('temperature'))
    warp = record.get('sampling_warp_config', {})
    return (temperature is not None and abs(temperature - 1.) < 1e-10
            and warp.get('top_k', 0) in (0, None)
            and warp.get('top_p', 1.) == 1. and warp.get('min_p', 0.) == 0.)


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    split: str
    prompt_id: str
    output_tokens: int
    blocks: int
    sha256: str
    identity: dict


def trace_index(directory, accepted_paths=None, *, eligibility_fn=primary_eligible, regime='primary'):
    """Index immutable per-generation sidecars without holding traces in RAM."""
    entries, excluded, seen, prompt_splits = [], {}, set(), {}
    paths = sorted(Path(p) for p in accepted_paths) if accepted_paths is not None else sorted(Path(directory).glob('rank*/*.pt'))
    for path in paths:
        side = json.loads(path.with_suffix('.pt.json').read_text())
        generation = side.get('generation')
        identity = side.get('identity')
        if not generation or not identity:
            raise ValueError(f'Training needs per-generation sidecar metadata: {path}')
        if not generation.get('complete'):
            raise ValueError(f'Incomplete trace generation cannot be fitted: {path}')
        key = fingerprint(identity)
        if key in seen:
            raise ValueError('Duplicate trace identity across ranks')
        seen.add(key)
        prompt = identity['prompt']
        split = generation['split']
        if identity.get('split', split) != split or generation.get('condition') != identity['condition']:
            raise ValueError('Generation and resumable identity disagree on split or condition')
        if split not in {'train', 'dev', 'test'}:
            raise ValueError('Unknown trace split')
        if prompt in prompt_splits and prompt_splits[prompt] != split:
            raise ValueError(f'Prompt leaks across disjoint splits: {prompt}')
        prompt_splits[prompt] = split
        candidate = {**generation, 'condition': identity['condition'],
                     'temperature': identity['temperature'],
                     'sampling_warp_config': identity['sampling_warp_config']}
        if not eligibility_fn(candidate):
            reason = f"{split}/{identity['condition']}/T={identity['temperature']}"
            excluded[reason] = excluded.get(reason, 0) + generation_length(generation)
            continue
        if identity.get('adapter_version') != 0 or identity.get('proposal_mode') != 'plain_dflash':
            raise ValueError('M2 requires plain DFlash traces collected before fitting any adapter')
        entries.append(ShardInfo(path, split, prompt, generation_length(generation),
                                 int(side['blocks']), side['sha256'], identity))
    if not entries:
        if regime == 'long_oracle_m5':
            raise RuntimeError('blocked: insufficient long-condition train/dev/test traces for M5 long oracle ablation')
        raise ValueError(f'No eligible {regime} traces')
    summaries = {split: {'generated_output_tokens': sum(x.output_tokens for x in entries if x.split == split),
                        'prompts': len({x.prompt_id for x in entries if x.split == split}),
                        'shards': sum(x.split == split for x in entries),
                        'blocks': sum(x.blocks for x in entries if x.split == split)}
                 for split in ['train', 'dev', 'test']}
    if any(summaries[s]['blocks'] == 0 for s in summaries):
        if regime == 'long_oracle_m5':
            raise RuntimeError('blocked: M5 long oracle ablation requires disjoint nonempty long train, dev, and test traces')
        raise ValueError('Primary train, development, and test traces are all required')
    report = {'primary': summaries, 'excluded_generated_tokens': excluded,
              'primary_train_tokens_by_temperature': {str(T): sum(e.output_tokens for e in entries if e.split == 'train' and e.identity['temperature'] == T) for T in [0., .7]},
              'source_fingerprint': fingerprint([{'identity': e.identity, 'sha256': e.sha256} for e in entries])}
    if regime != 'primary':
        report['regime'] = regime
    return entries, report


def generation_length(generation):
    values = generation['output_tokens']
    count = len(values) if isinstance(values, list) else int(values)
    if generation.get('realized_length', count) != count:
        raise ValueError('Generation output tokens disagree with realized_length')
    return count


def validate_reference_gate(args, index_report, entries, projection, token_metadata):
    if args.scale == 'smoke':
        return
    audit = json.loads(Path(args.m1_audit).read_text())
    if audit.get('status') != 'passed' or audit.get('scale') != args.scale:
        raise RuntimeError('blocked: M1 collection/diagnostic audit must pass before M2')
    floor = 2_000_000 if args.scale == 'full' else 500_000
    if not getattr(args, 'allow_long_oracle_fit', False) and not getattr(args, 'allow_pure_t1_ablation', False):
        if index_report['primary']['train']['generated_output_tokens'] < floor:
            raise RuntimeError(f'blocked: primary standard train traces have fewer than {floor} generated tokens')
        if args.scale == 'full':
            max_tokens = getattr(args, 'max_primary_train_tokens', None) or 4_000_000
            if index_report['primary']['train']['generated_output_tokens'] > max_tokens:
                raise RuntimeError(f'blocked: full-scale primary train traces exceed {max_tokens} token ceiling')
        if any(index_report['primary_train_tokens_by_temperature'][str(T)] < floor // 2 for T in [0., .7]):
            raise RuntimeError('blocked: each primary collection temperature must supply half the training floor')
        test_floor = 128 if args.scale == 'full' else 64
        if index_report['primary']['test']['prompts'] < test_floor:
            raise RuntimeError(f'blocked: fewer than {test_floor} standard test prompts')
    elif getattr(args, 'allow_long_oracle_fit', False):
        min_long = getattr(args, 'min_long_oracle_train_tokens', None) or 50_000
        if index_report['primary']['train']['generated_output_tokens'] < min_long:
            raise RuntimeError(f'blocked: M5 long oracle train traces have fewer than {min_long} tokens')
    manifest = json.loads(Path(args.prompts).read_text())
    prompts = {p['prompt_id']: p for p in manifest['prompts']}
    for entry in entries:
        if entry.prompt_id not in prompts or prompts[entry.prompt_id]['split'] != entry.split:
            raise ValueError('Trace prompt is missing or has inconsistent split in frozen prompt manifest')
    heldout = manifest['complete_heldout_dataset']
    if any(prompts[e.prompt_id]['dataset'] == heldout for e in entries if e.split != 'test'):
        raise ValueError('The completely held-out dataset leaked into train or development')
    calibration = set(projection.metadata['prompt_ids'])
    if len(calibration) < 64 or any(prompts[p]['split'] != 'train' or
                                   prompts[p]['condition'] != 'standard_nonthinking' for p in calibration):
        raise ValueError('Hidden projection calibration must use at least 64 standard train prompts')
    if set(token_metadata['calibration_prompt_ids']) != calibration:
        raise ValueError('Token and hidden feature statistics must use the same calibration split')
    if projection.metadata['condition'] != 'standard_nonthinking' or projection.metadata['split'] != 'train':
        raise ValueError('Invalid frozen projection calibration provenance')
    projection_hash, feature_hash = file_digest(args.projection), file_digest(args.features)
    for entry in entries:
        if (entry.identity.get('projection_sha256') != projection_hash or
                entry.identity.get('token_features_sha256') != feature_hash):
            raise ValueError('Trace identity differs from frozen hidden/token feature artifacts')
        if entry.identity['model_pair'] != [args.target, args.drafter]:
            raise ValueError('Trace model pair differs from the configured reference pair')
    diagnostic = json.loads(Path(args.m1_diagnostic).read_text())
    if diagnostic.get('status') not in {'measured', 'passed'} or diagnostic.get('scale') != args.scale:
        raise RuntimeError('blocked: M1b diagnostic report must be measured at the requested scale')
    diagnosed = {str(Path(x['path']).resolve()): x['sha256'] for x in diagnostic['source_shards']}
    if any(diagnosed.get(str(e.path.resolve())) != e.sha256 for e in entries):
        raise RuntimeError('blocked: M1b diagnostics do not cover the current eligible trace shards')


def load_frozen_head(metadata, device, *, output_dir=None):
    """Load exactly the referenced output projection, without the target model."""
    from safetensors import safe_open

    location = Path(metadata['artifact_id'])
    if not location.is_dir():
        env = json.loads(resolve_artifact_path('environment.json', output_dir, root=ROOT).read_text())
        info = env['model_artifacts'].get(metadata['artifact_id'])
        if not info or info['revision'] != metadata['revision']:
            raise ValueError('Frozen head artifact/revision is not locally available')
        location = Path(info['path'])
    if location.name != metadata['revision']:
        raise ValueError('Frozen head snapshot directory does not match its recorded immutable revision')
    if metadata.get('final_logit_softcapping'):
        raise ValueError('The required frozen head must be linear')
    config = json.loads((location / 'config.json').read_text())
    key = 'model.embed_tokens.weight' if metadata['tied_weights'] else 'lm_head.weight'
    if bool(config.get('tie_word_embeddings', False)) != bool(metadata['tied_weights']):
        raise ValueError('Frozen head tied-weight metadata disagrees with target configuration')
    index_path = location / 'model.safetensors.index.json'
    mapping = json.loads(index_path.read_text())['weight_map'] if index_path.exists() else None
    weight_path = location / (mapping[key] if mapping else 'model.safetensors')
    with safe_open(weight_path, framework='pt', device='cpu') as source:
        weight = source.get_tensor(key).to(device).detach()
    bias = None
    if metadata['bias']:
        bias_key = 'lm_head.bias'
        bias_path = location / (mapping[bias_key] if mapping else 'model.safetensors')
        with safe_open(bias_path, framework='pt', device='cpu') as source:
            bias = source.get_tensor(bias_key).to(device).detach()
    if list(weight.shape) != metadata.get('shape', metadata.get('weight_shape')):
        raise ValueError('Loaded head shape differs from stored trace metadata')
    if str(weight.dtype) != metadata['dtype']:
        raise ValueError('Loaded head dtype differs from stored trace metadata')
    return weight, bias


def epoch_references(entries, split, seed=None, *, block_shuffle_window: int = 0):
    win = int(block_shuffle_window)
    if win < 0:
        raise ValueError('block_shuffle_window must be nonnegative')
    chosen = [x for x in entries if x.split == split]
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(chosen)
    if win <= 0 or rng is None:
        return [(shard, i) for shard in chosen for i in range(shard.blocks)]
    refs = []
    for start in range(0, len(chosen), win):
        window_refs = [(shard, i) for shard in chosen[start:start + win] for i in range(shard.blocks)]
        rng.shuffle(window_refs)
        refs.extend(window_refs)
    return refs


@dataclass
class _PreparedMicrobatch:
    h_raw_bf16: torch.Tensor
    H: torch.Tensor
    valid_rows: torch.Tensor
    valid_cols: torch.Tensor
    prev_ids: torch.Tensor
    targets: PrecollatedGroupedTargets

    def pin_memory(self) -> '_PreparedMicrobatch':
        return _PreparedMicrobatch(
            h_raw_bf16=self.h_raw_bf16.pin_memory(),
            H=self.H.pin_memory(),
            valid_rows=self.valid_rows.pin_memory(),
            valid_cols=self.valid_cols.pin_memory(),
            prev_ids=self.prev_ids.pin_memory(),
            targets=self.targets.pin_memory(),
        )

    def to(self, device: torch.device, *, non_blocking: bool = False) -> '_PreparedMicrobatch':
        return _PreparedMicrobatch(
            h_raw_bf16=self.h_raw_bf16.to(device=device, non_blocking=non_blocking),
            H=self.H.to(device=device, dtype=torch.float32, non_blocking=non_blocking),
            valid_rows=self.valid_rows.to(device=device, dtype=torch.long, non_blocking=non_blocking),
            valid_cols=self.valid_cols.to(device=device, dtype=torch.long, non_blocking=non_blocking),
            prev_ids=self.prev_ids.to(device=device, dtype=torch.long, non_blocking=non_blocking),
            targets=self.targets.to(device=device, non_blocking=non_blocking),
        )


@dataclass
class _PreparedBatch:
    batch_index: int
    weight_sum: torch.Tensor
    microbatches: list[_PreparedMicrobatch]

    def pin_memory(self) -> '_PreparedBatch':
        return _PreparedBatch(
            batch_index=self.batch_index,
            weight_sum=self.weight_sum.pin_memory(),
            microbatches=[mb.pin_memory() for mb in self.microbatches],
        )


def _prepare_microbatch(blocks: list[dict], history_depth: int, vocab_size: int) -> _PreparedMicrobatch:
    h_raw_bf16 = torch.cat([b['h_raw_bf16'] for b in blocks]).contiguous()
    H = torch.cat([b['h'] for b in blocks]).to(dtype=torch.float32).contiguous()
    predecessors = []
    ids, probabilities, rest, weights = [], [], [], []
    cpu = torch.device('cpu')
    for block in blocks:
        prefix = block['verified_prefix_tail']
        draft_tokens = block['draft_tokens']
        K = block['block_length']
        for i in range(K):
            predecessors.append([prev_token(prefix, draft_tokens, i, d) for d in range(1, history_depth + 1)])
        ids.extend(block['union_ids'])
        probabilities.extend(block['target_probs_union'])
        rest.extend(block['target_rest_mass'])
        weights.append(loss_weights(K, cpu))
    previous = torch.tensor([[x if x is not None else -1 for x in row] for row in predecessors], dtype=torch.long)
    valid_rows, valid_cols = (previous >= 0).nonzero(as_tuple=True)
    prev_ids = previous[valid_rows, valid_cols].contiguous() if valid_rows.numel() else torch.empty(0, dtype=torch.long)
    targets = collate_grouped_targets(ids, probabilities, rest, torch.cat(weights), vocab_size=vocab_size, device=cpu)
    return _PreparedMicrobatch(
        h_raw_bf16=h_raw_bf16,
        H=H,
        valid_rows=valid_rows.contiguous(),
        valid_cols=valid_cols.contiguous(),
        prev_ids=prev_ids,
        targets=targets,
    )


class _ShardBatchDataset(Dataset):
    """Worker-safe dataset reading at most `max_cached_shards` verified compact shards at a time."""

    def __init__(self, references, batch_blocks: int, *, start_batch: int = 0, head_metadata=None,
                 rank: int = 0, world_size: int = 1, microbatch_blocks: int | None = None,
                 history_depth: int = 4, vocab_size: int, max_cached_shards: int = 1,
                 eligibility_fn=primary_eligible):
        self.references = references
        self.batch_blocks = int(batch_blocks)
        self.start_batch = int(start_batch)
        self.total_batches = (len(references) + self.batch_blocks - 1) // self.batch_blocks
        self.head_metadata = head_metadata
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.microbatch_blocks = int(microbatch_blocks) if microbatch_blocks is not None else None
        self.history_depth = int(history_depth)
        self.vocab_size = int(vocab_size)
        self.max_cached_shards = max(1, int(max_cached_shards))
        self.eligibility_fn = eligibility_fn
        self._tls = threading.local()

    def _get_cache(self) -> OrderedDict[Path, dict]:
        cache = getattr(self._tls, 'cache', None)
        if cache is None:
            cache = OrderedDict()
            self._tls.cache = cache
        return cache

    def __len__(self) -> int:
        return max(0, self.total_batches - self.start_batch)

    def _load_block(self, shard: ShardInfo, index: int) -> dict:
        cache = self._get_cache()
        payload = cache.get(shard.path)
        if payload is not None:
            cache.move_to_end(shard.path)
        else:
            if json.loads(shard.path.with_suffix('.pt.json').read_text())['sha256'] != shard.sha256:
                raise ValueError('An indexed shard changed after the frozen training manifest was built')
            payload = read_shard(shard.path)
            if len(payload['blocks']) != shard.blocks:
                raise ValueError('Shard block count differs from immutable sidecar')
            cache[shard.path] = payload
            while len(cache) > self.max_cached_shards:
                cache.popitem(last=False)
        block = payload['blocks'][index]
        if block['split'] != shard.split or block['prompt_id'] != shard.prompt_id or not self.eligibility_fn(block):
            raise ValueError('Block metadata differs from eligible shard identity')
        if self.head_metadata is not None and block['draft_lm_head'] != self.head_metadata:
            raise ValueError('Mixed frozen head artifacts are forbidden in one fit')
        return block

    def __getitem__(self, item: int) -> _PreparedBatch:
        batch_index = self.start_batch + int(item)
        selected = self.references[batch_index * self.batch_blocks:(batch_index + 1) * self.batch_blocks]
        if self.world_size > 1:
            selected = selected[self.rank::self.world_size]
        local_blocks = [self._load_block(shard, idx) for shard, idx in selected]
        if not local_blocks:
            return _PreparedBatch(batch_index=batch_index, weight_sum=torch.zeros((), dtype=torch.float32), microbatches=[])
        mb_step = self.microbatch_blocks if self.microbatch_blocks is not None else max(1, len(local_blocks))
        microbatches = [_prepare_microbatch(local_blocks[s:s + mb_step], self.history_depth, self.vocab_size)
                        for s in range(0, len(local_blocks), mb_step)]
        weight_sum = torch.stack([mb.targets.weights.sum() for mb in microbatches]).sum()
        return _PreparedBatch(batch_index=batch_index, weight_sum=weight_sum, microbatches=microbatches)


def _threaded_prefetch_iter(dataset, num_workers: int, *, pin_fn=None):
    total = len(dataset)
    if total == 0:
        return
    eff_workers = int(num_workers)
    if eff_workers <= 0:
        for i in range(total):
            item = dataset[i]
            yield pin_fn(item) if pin_fn is not None else item
        return
    window = max(2, 2 * eff_workers)
    ex = ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix='edraft-shard')
    pending: deque = deque()
    next_idx = 0
    try:
        while next_idx < min(total, window):
            pending.append(ex.submit(dataset.__getitem__, next_idx))
            next_idx += 1
        while pending:
            item = pending.popleft().result()
            if next_idx < total:
                pending.append(ex.submit(dataset.__getitem__, next_idx))
                next_idx += 1
            yield pin_fn(item) if pin_fn is not None else item
    finally:
        for fut in pending:
            fut.cancel()
        pending.clear()
        ex.shutdown(wait=True, cancel_futures=True)


def iter_prepared_batches(references, batch_blocks: int, *, start_batch: int = 0, head_metadata=None,
                          rank: int = 0, world_size: int = 1, microbatch_blocks: int | None = None,
                          history_depth: int = 4, vocab_size: int, num_workers: int = 0,
                          pin_memory: bool = False, loader_seed: int = 0, max_cached_shards: int = 1,
                          eligibility_fn=primary_eligible):
    del loader_seed  # Deterministic sequential order with zero global/worker RNG draws.
    dataset = _ShardBatchDataset(
        references, batch_blocks, start_batch=start_batch, head_metadata=head_metadata,
        rank=rank, world_size=world_size, microbatch_blocks=microbatch_blocks,
        history_depth=history_depth, vocab_size=vocab_size, max_cached_shards=max_cached_shards,
        eligibility_fn=eligibility_fn,
    )
    use_pin = bool(pin_memory and torch.cuda.is_available())
    pin_fn = (lambda item: item.pin_memory()) if use_pin else None
    yield from _threaded_prefetch_iter(dataset, int(num_workers), pin_fn=pin_fn)


def _block_own_committed(block) -> list[int]:
    if block.get('committed_tokens') is not None:
        return [int(x) for x in block['committed_tokens']]
    acc = int(block['accepted_length'])
    K = int(block['block_length'])
    own = [int(x) for x in block['draft_tokens'][:acc]]
    if acc < K and block.get('replacement_token') is not None:
        own.append(int(block['replacement_token']))
    elif acc == K and block.get('bonus_token') is not None:
        own.append(int(block['bonus_token']))
    return own


def _annotate_verified_stream(payload_blocks, history_depth: int, *, sidecar_generation: dict | None = None):
    gen_tokens = sidecar_generation.get('output_tokens') if isinstance(sidecar_generation, dict) else None
    realized_len = (
        len(gen_tokens) if isinstance(gen_tokens, list)
        else int(sidecar_generation.get('realized_length', gen_tokens))
        if isinstance(gen_tokens, int) and isinstance(sidecar_generation, dict)
        else None
    )
    own_lists = []
    for idx, block in enumerate(payload_blocks):
        if block.get('committed_tokens') is not None:
            own_lists.append([int(x) for x in block['committed_tokens']])
        elif isinstance(gen_tokens, list):
            start = int(block['absolute_output_offset'])
            end = int(payload_blocks[idx + 1]['absolute_output_offset']) if idx + 1 < len(payload_blocks) else len(gen_tokens)
            own_lists.append([int(x) for x in gen_tokens[start:max(start, end)]])
        else:
            raw_own = _block_own_committed(block)
            if realized_len is not None:
                rem = max(0, realized_len - int(block['absolute_output_offset']))
                raw_own = raw_own[:rem]
            own_lists.append(raw_own)
    for idx, block in enumerate(payload_blocks):
        own = own_lists[idx]
        future: list[int] = []
        j = idx + 1
        while j < len(own_lists) and len(future) < history_depth:
            need = history_depth - len(future)
            future.extend(own_lists[j][:need])
            j += 1
        block['_verified_committed_window'] = own + future
        block['_verified_eval_slots'] = min(int(block['block_length']), len(own))


class _RawShardBatchDataset(Dataset):
    """Parallel-prefetchable dataset yielding `(batch_index, list[block_dict])` for `two_sided_pseudolikelihood`."""

    def __init__(self, references, batch_blocks: int, *, start_batch: int = 0, head_metadata=None,
                 rank: int = 0, world_size: int = 1, eligibility_fn=primary_eligible,
                 history_depth: int = 4, max_cached_shards: int = 1):
        self.references = list(references)
        self.batch_blocks = int(batch_blocks)
        self.start_batch = int(start_batch)
        self.total_batches = (len(self.references) + self.batch_blocks - 1) // self.batch_blocks
        self.head_metadata = head_metadata
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.eligibility_fn = eligibility_fn
        self.history_depth = int(history_depth)
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._tls = threading.local()

    def _get_cache(self) -> OrderedDict[Path, dict]:
        cache = getattr(self._tls, 'cache', None)
        if cache is None:
            cache = OrderedDict()
            self._tls.cache = cache
        return cache

    def __len__(self) -> int:
        return max(0, self.total_batches - self.start_batch)

    def __getitem__(self, item: int) -> tuple[int, list[dict]]:
        cache = self._get_cache()
        batch_index = self.start_batch + int(item)
        selected = self.references[batch_index * self.batch_blocks:(batch_index + 1) * self.batch_blocks]
        if self.world_size > 1:
            selected = selected[self.rank::self.world_size]
        result = []
        for shard, index in selected:
            if shard.path in cache:
                cache.move_to_end(shard.path)
                payload = cache[shard.path]
            else:
                sidecar = json.loads(shard.path.with_suffix('.pt.json').read_text())
                if sidecar['sha256'] != shard.sha256:
                    raise ValueError('An indexed shard changed after the frozen training manifest was built')
                payload = read_shard(shard.path)
                if len(payload['blocks']) != shard.blocks:
                    raise ValueError('Shard block count differs from immutable sidecar')
                _annotate_verified_stream(payload['blocks'], self.history_depth, sidecar_generation=sidecar.get('generation'))
                cache[shard.path] = payload
                while len(cache) > self.max_cached_shards:
                    cache.popitem(last=False)
            block = payload['blocks'][index]
            if block['split'] != shard.split or block['prompt_id'] != shard.prompt_id or not self.eligibility_fn(block):
                raise ValueError('Block metadata differs from eligible shard identity')
            if self.head_metadata is not None and block['draft_lm_head'] != self.head_metadata:
                raise ValueError('Mixed frozen head artifacts are forbidden in one fit')
            result.append(block)
        return batch_index, result


def batches(references, batch_blocks, *, start_batch=0, head_metadata=None,
            rank=0, world_size=1, eligibility_fn=primary_eligible,
            history_depth: int = 4, shard_cache_size: int = 1,
            num_workers: int = 0, loader_seed: int = 0):
    """Bounded compact-shard LRU cache (`1` shard by default; `W` under multi-shard block shuffle)."""
    del loader_seed
    dataset = _RawShardBatchDataset(
        references, batch_blocks, start_batch=start_batch, head_metadata=head_metadata,
        rank=rank, world_size=world_size, eligibility_fn=eligibility_fn,
        history_depth=history_depth, max_cached_shards=shard_cache_size,
    )
    yield from _threaded_prefetch_iter(dataset, int(num_workers))


@torch.no_grad()
def _reconstruct_h_raw(h_raw_bf16: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                       *, vocab_chunk: int = 8192, output_multiplier: float = 1., bf16_gemm: bool = False,
                       non_blocking: bool = False) -> torch.Tensor:
    device = weight.device
    num_rows = int(h_raw_bf16.shape[0])
    vocab_size = int(weight.shape[0])
    if num_rows == 0:
        return torch.empty((0, vocab_size), device=device, dtype=torch.float32)
    logits = torch.empty((num_rows, vocab_size), device=device, dtype=torch.float32)
    mult = float(output_multiplier)
    if bf16_gemm:
        h_bf16 = h_raw_bf16.to(device=device, dtype=torch.bfloat16, non_blocking=non_blocking)
        for start in range(0, vocab_size, vocab_chunk):
            end = min(start + vocab_chunk, vocab_size)
            w_chunk = weight[start:end]
            w_bf16 = w_chunk if w_chunk.dtype == torch.bfloat16 else w_chunk.to(torch.bfloat16)
            out = F.linear(h_bf16, w_bf16).float()
            if bias is not None:
                out = out + bias[start:end].float()
            if mult != 1.0:
                out = out * mult
            logits[:, start:end] = out
    else:
        h_fp32 = h_raw_bf16.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        for start in range(0, vocab_size, vocab_chunk):
            end = min(start + vocab_chunk, vocab_size)
            b_chunk = None if bias is None else bias[start:end].float()
            logits[:, start:end] = F.linear(h_fp32, weight[start:end].float(), b_chunk) * mult
    return logits.log_softmax(-1)


@torch.no_grad()
def reconstruct_batch(blocks, weight, bias=None, *, vocab_chunk=8192, output_multiplier=1., bf16_gemm=False):
    if not blocks:
        return torch.empty((0, len(weight)), device=weight.device, dtype=torch.float32)
    h_raw_bf16 = torch.cat([b['h_raw_bf16'] for b in blocks])
    return _reconstruct_h_raw(h_raw_bf16, weight, bias, vocab_chunk=vocab_chunk,
                              output_multiplier=output_multiplier, bf16_gemm=bf16_gemm)


def _dense_correction_tensor(W: torch.Tensor, A: torch.Tensor, b: torch.Tensor, E: torch.Tensor,
                             E_bf16: torch.Tensor | None, H: torch.Tensor, valid_rows: torch.Tensor,
                             valid_cols: torch.Tensor, prev_ids: torch.Tensor, *, bf16_gemm: bool = False) -> torch.Tensor:
    if bf16_gemm:
        assert E_bf16 is not None
        proj_bf16 = (H.to(torch.bfloat16) @ W.to(torch.bfloat16).T)
        unary = (proj_bf16 @ E_bf16.T).float() + b.float()
        if valid_rows.numel():
            E_prev = E_bf16.index_select(0, prev_ids)
            A_sel = A.to(torch.bfloat16).index_select(0, valid_cols)
            parts = torch.bmm(E_prev.unsqueeze(1), A_sel.transpose(1, 2)).squeeze(1).float()
            pair = torch.zeros((H.shape[0], E.shape[1]), device=H.device, dtype=torch.float32).index_add(0, valid_rows, parts)
            return unary + (pair.to(torch.bfloat16) @ E_bf16.T).float()
        return unary
    unary = (H.float() @ W.float().T) @ E.float().T + b.float()
    if valid_rows.numel():
        E_prev = E.index_select(0, prev_ids).float()
        parts = torch.bmm(E_prev.unsqueeze(1), A.index_select(0, valid_cols).transpose(1, 2)).squeeze(1)
        pair = torch.zeros((H.shape[0], E.shape[1]), device=H.device, dtype=torch.float32).index_add(0, valid_rows, parts)
        return unary + pair @ E.float().T
    return unary


def _corrected_prepared_scores(adapter: DenseAdapter, H: torch.Tensor, valid_rows: torch.Tensor,
                               valid_cols: torch.Tensor, prev_ids: torch.Tensor, l_unit: torch.Tensor,
                               *, return_sparse_keys: bool = False, sparse_override=None,
                               bf16_gemm: bool = False, dense_fn=None, beta: float = 1.0):
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError('beta must be a finite nonnegative scalar')
    beta_f = float(beta)
    E_bf16 = adapter._token_features_bf16() if bf16_gemm else None
    fn = _dense_correction_tensor if dense_fn is None else dense_fn
    dense = fn(adapter.W, adapter.A, adapter.b, adapter.E, E_bf16, H, valid_rows, valid_cols, prev_ids,
               bf16_gemm=bf16_gemm)
    scores = (l_unit - dense) if beta_f == 1.0 else (l_unit - beta_f * dense)
    sparse_table = sparse_override if sparse_override is not None else adapter.sparse
    observed_keys = None
    if sparse_table is not None and valid_rows.numel():
        candidates = scores.topk(min(adapter.M_sparse, scores.shape[-1]), dim=-1).indices
        d_pair = (valid_cols + 1)[:, None]
        prev_pair = prev_ids[:, None]
        cands_pair = candidates.index_select(0, valid_rows)
        corrections = sparse_table.lookup(d_pair, prev_pair, cands_pair)
        values = torch.zeros(len(scores), candidates.shape[1], device=scores.device, dtype=torch.float32).index_add(
            0, valid_rows, corrections)
        scaled_vals = -values if beta_f == 1.0 else (-beta_f * values)
        scores = scores.scatter_add(1, candidates, scaled_vals)
        if return_sparse_keys:
            observed_keys = pack_keys(d_pair, prev_pair, cands_pair, adapter.E.shape[0]).reshape(-1)
    if return_sparse_keys:
        if observed_keys is None:
            observed_keys = torch.empty(0, dtype=torch.long, device=scores.device)
        return scores, observed_keys
    return scores


def corrected_batch_scores(adapter, blocks, l_unit, *, return_sparse_keys=False,
                           sparse_override=None, bf16_gemm=False, beta: float = 1.0):
    """Batched teacher-forced dense features; sampled canvas IDs stay constants."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError('beta must be a finite nonnegative scalar')
    H = torch.cat([b['h'] for b in blocks]).to(adapter.W.device, dtype=torch.float32)
    predecessors = []
    for block in blocks:
        prefix = block['verified_prefix_tail']
        for i in range(block['block_length']):
            predecessors.append([prev_token(prefix, block['draft_tokens'], i, d)
                                 for d in range(1, len(adapter.A) + 1)])
    previous = torch.tensor([[x if x is not None else -1 for x in row] for row in predecessors],
                            device=adapter.W.device, dtype=torch.long)
    # Skip missing predecessors entirely, including sparse lookup below.
    valid_rows, valid_cols = (previous >= 0).nonzero(as_tuple=True)
    prev_ids = previous[valid_rows, valid_cols] if valid_rows.numel() else torch.empty(0, device=H.device, dtype=torch.long)
    return _corrected_prepared_scores(
        adapter, H, valid_rows, valid_cols, prev_ids, l_unit,
        return_sparse_keys=return_sparse_keys, sparse_override=sparse_override,
        bf16_gemm=bf16_gemm, beta=beta,
    )


def batch_targets(blocks, device):
    ids, probabilities, rest, weights = [], [], [], []
    for block in blocks:
        ids.extend(x.to(device) for x in block['union_ids'])
        probabilities.extend(x.to(device) for x in block['target_probs_union'])
        rest.extend(x.to(device) for x in block['target_rest_mass'])
        weights.append(loss_weights(block['block_length'], device))
    return ids, probabilities, rest, torch.cat(weights)


def two_sided_batch_scores_and_targets(adapter, blocks, l_unit_full, *, return_sparse_keys=False,
                                       sparse_override=None, bf16_gemm: bool = False):
    """Batched two-sided verified-stream scores on committed on-policy slots only."""
    device = adapter.W.device
    D = len(adapter.A)
    h_rows, l_rows, prev_rows, succ_rows, target_ids, weight_rows = [], [], [], [], [], []
    cursor = 0
    for block in blocks:
        K = int(block['block_length'])
        committed = block['_verified_committed_window'] if '_verified_committed_window' in block else _block_own_committed(block)
        n_eval = int(block['_verified_eval_slots']) if '_verified_eval_slots' in block else min(K, len(committed))
        block_l = l_unit_full[cursor:cursor + K]
        cursor += K
        if n_eval <= 0:
            continue
        prefix = block['verified_prefix_tail']
        h_rows.append(block['h'][:n_eval])
        l_rows.append(block_l[:n_eval])
        for i in range(n_eval):
            prev_rows.append([prev_token(prefix, committed, i, d) for d in range(1, D + 1)])
            succ_rows.append([next_token(committed, i, d) for d in range(1, D + 1)])
            target_ids.append(int(committed[i]))
        weight_rows.append(loss_weights(n_eval, device))
    if not h_rows:
        empty_s = torch.empty((0, adapter.E.shape[0]), dtype=torch.float32, device=device)
        empty_t = torch.empty(0, dtype=torch.long, device=device)
        empty_w = torch.empty(0, dtype=torch.float32, device=device)
        if return_sparse_keys:
            return empty_s, empty_t, empty_w, torch.empty(0, dtype=torch.long, device=device)
        return empty_s, empty_t, empty_w
    H = torch.cat(h_rows).to(device, dtype=torch.float32)
    l_unit = torch.cat(l_rows)
    previous = torch.tensor([[x if x is not None else -1 for x in row] for row in prev_rows],
                            device=device, dtype=torch.long)
    successor = torch.tensor([[x if x is not None else -1 for x in row] for row in succ_rows],
                             device=device, dtype=torch.long)
    pair = torch.zeros(len(H), adapter.E.shape[1], device=device, dtype=torch.float32)
    if bf16_gemm:
        E_bf16 = adapter._token_features_bf16()
        for d in range(D):
            rows_f = (previous[:, d] >= 0).nonzero(as_tuple=True)[0]
            if len(rows_f):
                part_f = (E_bf16.index_select(0, previous[rows_f, d]) @ adapter.A[d].to(torch.bfloat16).T).float()
                pair = pair.index_add(0, rows_f, part_f)
            rows_b = (successor[:, d] >= 0).nonzero(as_tuple=True)[0]
            if len(rows_b):
                # (A[d].T @ E[succ])^T = E[succ] @ A[d]
                part_b = (E_bf16.index_select(0, successor[rows_b, d]) @ adapter.A[d].to(torch.bfloat16)).float()
                pair = pair.index_add(0, rows_b, part_b)
        dense = adapter.unary(H, bf16_gemm=True) + (pair.to(torch.bfloat16) @ E_bf16.T).float()
    else:
        for d in range(D):
            rows_f = (previous[:, d] >= 0).nonzero(as_tuple=True)[0]
            if len(rows_f):
                part_f = adapter.E[previous[rows_f, d]].float() @ adapter.A[d].T
                pair = pair.index_add(0, rows_f, part_f)
            rows_b = (successor[:, d] >= 0).nonzero(as_tuple=True)[0]
            if len(rows_b):
                # (A[d].T @ E[succ])^T = E[succ] @ A[d]
                part_b = adapter.E[successor[rows_b, d]].float() @ adapter.A[d]
                pair = pair.index_add(0, rows_b, part_b)
        dense = adapter.unary(H, bf16_gemm=False) + pair @ adapter.E.float().T
    scores = l_unit - dense
    observed_keys = []
    sparse_table = sparse_override if sparse_override is not None else adapter.sparse
    if sparse_table is not None:
        candidates = scores.topk(min(adapter.M_sparse, scores.shape[-1]), dim=-1).indices
        values = torch.zeros_like(candidates, dtype=torch.float32)
        V = adapter.E.shape[0]
        for d in range(D):
            rows_f = (previous[:, d] >= 0).nonzero(as_tuple=True)[0]
            if len(rows_f):
                corr_f = sparse_table.lookup(d + 1, previous[rows_f, d, None], candidates[rows_f])
                values = values.index_add(0, rows_f, corr_f)
                if return_sparse_keys:
                    observed_keys.append(pack_keys(d + 1, previous[rows_f, d, None], candidates[rows_f], V).reshape(-1))
            rows_b = (successor[:, d] >= 0).nonzero(as_tuple=True)[0]
            if len(rows_b):
                corr_b = sparse_table.lookup(d + 1, candidates[rows_b], successor[rows_b, d, None])
                values = values.index_add(0, rows_b, corr_b)
                if return_sparse_keys:
                    observed_keys.append(pack_keys(d + 1, candidates[rows_b], successor[rows_b, d, None], V).reshape(-1))
        scores = scores - torch.zeros_like(scores).scatter(1, candidates, values)
    targets = torch.tensor(target_ids, device=device, dtype=torch.long)
    weights = torch.cat(weight_rows)
    if return_sparse_keys:
        packed = torch.cat(observed_keys) if observed_keys else torch.empty(0, dtype=torch.long, device=device)
        return scores, targets, weights, packed
    return scores, targets, weights


def batch_loss(adapter, blocks, l_unit, *, collect_sparse_keys=False, sparse_override=None,
               bf16_gemm=False, objective='union_rest_grouped'):
    if objective == 'two_sided_pseudolikelihood':
        out = two_sided_batch_scores_and_targets(adapter, blocks, l_unit,
                                                 return_sparse_keys=collect_sparse_keys,
                                                 sparse_override=sparse_override,
                                                 bf16_gemm=bf16_gemm)
        scores, targets, weights = out[:3]
        if len(targets) == 0:
            zero_loss = 0.0 * adapter.b.sum()
            result = (zero_loss, torch.zeros((), dtype=torch.float32, device=adapter.b.device), scores)
            return (*result, out[3]) if collect_sparse_keys else result
        loss = two_sided_pseudolikelihood_loss(scores, targets, weights)
        result = (loss, weights.sum(), scores)
        return (*result, out[3]) if collect_sparse_keys else result
    corrected = corrected_batch_scores(adapter, blocks, l_unit, return_sparse_keys=collect_sparse_keys,
                                       sparse_override=sparse_override, bf16_gemm=bf16_gemm)
    scores = corrected[0] if collect_sparse_keys else corrected
    ids, probabilities, rest, weights = batch_targets(blocks, scores.device)
    result = (union_rest_grouped(scores, ids, probabilities, rest, weights), weights.sum(), scores)
    return (*result, corrected[1]) if collect_sparse_keys else result


@torch.no_grad()
def evaluate(adapter, references, head, bias, args, head_metadata, *, rank=0, world_size=1,
             use_dist=False, eligibility_fn=primary_eligible):
    was_training = adapter.training
    adapter.eval()
    try:
        bf16_gemm = bool(getattr(args, 'bf16_gemm', False))
        objective = getattr(args, 'objective', 'union_rest_grouped')
        sums = dict(loss=0., weight=0., slots=0, union_overlap_lower=0., grouped_overlap_upper=0.,
                    full_overlap=0., full_top1=0., full_slots=0, full_loss=0., full_grouped_loss=0.,
                    full_weight=0., pl_loss=0., pl_weight=0., pl_slots=0, pl_top1=0)
        for _, blocks in batches(references, args.batch_blocks, head_metadata=head_metadata,
                                 rank=rank, world_size=world_size, eligibility_fn=eligibility_fn,
                                 history_depth=args.history_depth):
            if not blocks:
                continue
            l_unit = reconstruct_batch(blocks, head, bias, vocab_chunk=args.vocab_chunk,
                                       output_multiplier=head_metadata.get('output_multiplier', 1.),
                                       bf16_gemm=bf16_gemm)
            loss, weight, scores = batch_loss(adapter, blocks, l_unit, bf16_gemm=bf16_gemm,
                                              objective='union_rest_grouped')
            q_log = scores.log_softmax(-1)
            q = q_log.exp()
            sums['loss'] += float(loss)
            sums['weight'] += float(weight)
            if objective == 'two_sided_pseudolikelihood':
                pl_scores, pl_targets, pl_weights = two_sided_batch_scores_and_targets(
                    adapter, blocks, l_unit, bf16_gemm=bf16_gemm,
                )
                if len(pl_targets) > 0:
                    pl_val = two_sided_pseudolikelihood_loss(pl_scores, pl_targets, pl_weights)
                    sums['pl_loss'] += float(pl_val)
                    sums['pl_weight'] += float(pl_weights.sum())
                    sums['pl_slots'] += len(pl_targets)
                    sums['pl_top1'] += int((pl_scores.argmax(-1) == pl_targets).sum())
            cursor = 0
            for block in blocks:
                K = block['block_length']
                rows = q[cursor:cursor + K]
                log_rows = q_log[cursor:cursor + K]
                ids, probabilities, rest, weights = batch_targets([block], q.device)
                for i in range(K):
                    p = probabilities[i]
                    qi = rows[i, ids[i]]
                    lower = float(torch.minimum(p, qi).sum())
                    upper = lower + min(float(rest[i]), max(0., 1 - float(qi.double().sum())))
                    sums['union_overlap_lower'] += lower
                    sums['grouped_overlap_upper'] += upper
                if 'live_full' in block:
                    P = block['live_full']['target_probs'].to(q.device)
                    full_kl_row = (torch.xlogy(P, P) - P * log_rows).sum(-1)
                    grouped = union_rest_grouped(scores[cursor:cursor + K], ids, probabilities, rest, weights)
                    sums['full_loss'] += float((full_kl_row * weights).sum())
                    sums['full_grouped_loss'] += float(grouped)
                    sums['full_weight'] += float(weights.sum())
                    sums['full_overlap'] += float(torch.minimum(P, rows).sum())
                    sums['full_top1'] += int((P.argmax(-1) == rows.argmax(-1)).sum())
                    sums['full_slots'] += K
                sums['slots'] += K
                cursor += K
        if use_dist and world_size > 1:
            keys = ('loss', 'weight', 'slots', 'union_overlap_lower', 'grouped_overlap_upper',
                    'full_overlap', 'full_top1', 'full_slots', 'full_loss', 'full_grouped_loss',
                    'full_weight', 'pl_loss', 'pl_weight', 'pl_slots', 'pl_top1')
            comm_dev = _comm_device(adapter.E.device)
            vec = torch.tensor([float(sums[k]) for k in keys], dtype=torch.float64, device=comm_dev)
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            reduced = vec.cpu().tolist()
            for k, val in zip(keys, reduced):
                sums[k] = int(round(val)) if k in ('slots', 'full_slots', 'full_top1', 'pl_slots', 'pl_top1') else float(val)
        full = sums['full_slots']
        grouped_mean = sums['loss'] / sums['weight']
        primary_metric = (sums['pl_loss'] / sums['pl_weight']) if objective == 'two_sided_pseudolikelihood' else grouped_mean
        result = {'loss_mode': objective, 'weighted_mean_kl': primary_metric,
                  'weighted_kl_sum': sums['pl_loss'] if objective == 'two_sided_pseudolikelihood' else sums['loss'],
                  'weight_sum': sums['pl_weight'] if objective == 'two_sided_pseudolikelihood' else sums['weight'],
                  'logical_slots': sums['slots'],
                  'union_overlap_lower_bound': sums['union_overlap_lower'] / sums['slots'],
                  'grouped_overlap_upper_bound': sums['grouped_overlap_upper'] / sums['slots'],
                  'live_full_diagnostic': {'logical_slots': full,
                      'mean_overlap': sums['full_overlap'] / full if full else None,
                      'top1_agreement': sums['full_top1'] / full if full else None,
                      'weighted_mean_full_kl': sums['full_loss'] / sums['full_weight'] if full else None,
                      'weighted_mean_grouped_kl': sums['full_grouped_loss'] / sums['full_weight'] if full else None,
                      'full_minus_grouped_kl': (sums['full_loss'] - sums['full_grouped_loss']) / sums['full_weight'] if full else None,
                      'proposal_source': 'same reconstructed full-vocabulary q for both objectives'},
                  'T_train': 1., 'beta_train': 1., 'warpers_used_for_training': False}
        if objective == 'two_sided_pseudolikelihood':
            result['two_sided_pseudolikelihood'] = {
                'weighted_mean_nll': sums['pl_loss'] / sums['pl_weight'],
                'verified_slots': sums['pl_slots'],
                'verified_top1_accuracy': sums['pl_top1'] / sums['pl_slots'] if sums['pl_slots'] else None,
                'causal_union_rest_grouped_kl': grouped_mean,
            }
        return result
    finally:
        adapter.train(was_training)


DEFAULT_BETA_CALIBRATION_GRID = (
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.85, 1.0, 1.25, 1.50
)


@torch.no_grad()
def scale_adapter_inplace(adapter, scale: float):
    """Scale linear energy parameters `(W, A, b, sparse.values)` by `scale` while keeping frozen `E` untouched."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError('checkpoint scale must be a finite nonnegative scalar')
    s = float(scale)
    adapter.W.mul_(s)
    adapter.A.mul_(s)
    adapter.b.mul_(s)
    if adapter.sparse is not None:
        adapter.sparse.values.mul_(s)
    return adapter


def scale_adapter_checkpoint(
    src_path: Path | str,
    dst_path: Path | str,
    scale: float,
    *,
    extra_report: dict | None = None,
    preserve_fit_identity: bool = False,
) -> dict:
    """Create a scaled copy of a saved adapter checkpoint (`scale * (W, A, b, sparse.values)`)."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError('checkpoint scale must be a finite nonnegative scalar')
    src = Path(src_path)
    dst = Path(dst_path)
    if src.resolve() == dst.resolve():
        raise ValueError('scale_adapter_checkpoint requires distinct source and destination paths')
    checkpoint = torch.load(src, map_location='cpu', weights_only=True)
    s = float(scale)
    state = {k: v.clone() for k, v in checkpoint['adapter_state'].items()}
    for key in ('W', 'A', 'b', 'sparse.values'):
        if key in state:
            state[key] = state[key] * s
    checkpoint['adapter_state'] = state
    prev_scale = float(checkpoint.get('checkpoint_scale', 1.0))
    eff_scale = prev_scale * s
    non_unit = bool(checkpoint.get('non_unit_scaled', False) or s != 1.0)
    checkpoint['checkpoint_scale'] = eff_scale
    checkpoint['non_unit_scaled'] = non_unit
    base_identity = checkpoint.get('base_identity', checkpoint.get('identity'))
    checkpoint['base_identity'] = base_identity
    if not preserve_fit_identity:
        if eff_scale != 1.0 or non_unit:
            checkpoint['identity'] = fingerprint({
                'base_identity': base_identity,
                'checkpoint_scale': eff_scale,
                'non_unit_scaled': non_unit,
            })
        else:
            checkpoint['identity'] = base_identity
    checkpoint['adapter_version'] = fingerprint({
        'base_version': checkpoint.get('adapter_version'),
        'checkpoint_scale': eff_scale,
    })
    if extra_report:
        rep = dict(checkpoint.get('report', {}))
        rep.update(extra_report)
        checkpoint['report'] = rep
    atomic_torch_save(dst, checkpoint)
    return {
        'source': str(src),
        'destination': str(dst),
        'scale': s,
        'effective_checkpoint_scale': eff_scale,
        'identity': checkpoint.get('identity'),
        'sha256': file_digest(dst),
    }


@torch.no_grad()
def calibrate_beta_on_dev_live_full(adapter, references_dev, head, bias, args, head_metadata,
                                    *, rank=0, world_size=1, use_dist=False,
                                    eligibility_fn=primary_eligible, beta_grid=None) -> dict:
    """Post-fit scalar `beta` calibration minimizing exact `L_full` on held-out dev `live_full` blocks."""
    raw_grid = DEFAULT_BETA_CALIBRATION_GRID if beta_grid is None else tuple(beta_grid)
    if not raw_grid:
        raise ValueError('calibrate_beta_grid must be non-empty')
    betas = []
    for b in raw_grid:
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(float(b)) or float(b) < 0.0:
            raise ValueError('calibrate_beta_grid values must be finite nonnegative scalars')
        bf = float(b)
        if bf not in betas:
            betas.append(bf)
    for required in (0.0, 1.0):
        if required not in betas:
            betas.append(required)
    betas = sorted(betas)

    was_training = adapter.training
    adapter.eval()
    try:
        bf16_gemm = bool(getattr(args, 'bf16_gemm', False))
        full_sums = {b: 0.0 for b in betas}
        grouped_sums = {b: 0.0 for b in betas}
        full_weight = 0.0
        grouped_weight = 0.0
        full_slots = 0
        total_slots = 0

        for _, blocks in batches(references_dev, args.batch_blocks, head_metadata=head_metadata,
                                 rank=rank, world_size=world_size,
                                 eligibility_fn=eligibility_fn, history_depth=args.history_depth):
            if not blocks:
                continue
            l_unit = reconstruct_batch(blocks, head, bias, vocab_chunk=args.vocab_chunk,
                                       output_multiplier=head_metadata.get('output_multiplier', 1.),
                                       bf16_gemm=bf16_gemm)
            H = torch.cat([blk['h'] for blk in blocks]).to(adapter.W.device, dtype=torch.float32)
            predecessors = []
            for blk in blocks:
                prefix = blk['verified_prefix_tail']
                for i in range(blk['block_length']):
                    predecessors.append([prev_token(prefix, blk['draft_tokens'], i, d)
                                         for d in range(1, len(adapter.A) + 1)])
            previous = torch.tensor([[x if x is not None else -1 for x in row] for row in predecessors],
                                    device=adapter.W.device, dtype=torch.long)
            valid_rows, valid_cols = (previous >= 0).nonzero(as_tuple=True)
            prev_ids = (
                previous[valid_rows, valid_cols]
                if valid_rows.numel()
                else torch.empty(0, device=H.device, dtype=torch.long)
            )
            E_bf16 = adapter._token_features_bf16() if bf16_gemm else None
            dense = _dense_correction_tensor(
                adapter.W, adapter.A, adapter.b, adapter.E, E_bf16,
                H, valid_rows, valid_cols, prev_ids, bf16_gemm=bf16_gemm,
            )

            ids_all, probs_all, rest_all, weights_all = batch_targets(blocks, l_unit.device)
            grouped_weight += float(weights_all.sum())
            total_slots += int(len(H))

            live_slices = []
            cursor = 0
            for blk in blocks:
                K = int(blk['block_length'])
                if 'live_full' in blk:
                    P = blk['live_full']['target_probs'].to(l_unit.device)
                    w_blk = loss_weights(K, l_unit.device)
                    ent_blk = torch.xlogy(P, P).sum(-1)
                    live_slices.append((cursor, cursor + K, P, ent_blk, w_blk))
                    full_weight += float(w_blk.sum())
                    full_slots += K
                cursor += K

            for b in betas:
                if b == 0.0:
                    scores = l_unit
                else:
                    scores = (l_unit - dense) if b == 1.0 else (l_unit - b * dense)
                    if adapter.sparse is not None and valid_rows.numel():
                        cands = scores.topk(min(adapter.M_sparse, scores.shape[-1]), dim=-1).indices
                        d_pair = (valid_cols + 1)[:, None]
                        prev_pair = prev_ids[:, None]
                        cands_pair = cands.index_select(0, valid_rows)
                        corrections = adapter.sparse.lookup(d_pair, prev_pair, cands_pair)
                        vals = torch.zeros(len(scores), cands.shape[1], device=scores.device, dtype=torch.float32).index_add(
                            0, valid_rows, corrections,
                        )
                        scaled_vals = -vals if b == 1.0 else (-b * vals)
                        scores = scores.scatter_add(1, cands, scaled_vals)
                grouped_sums[b] += float(union_rest_grouped(scores, ids_all, probs_all, rest_all, weights_all))
                if live_slices:
                    log_q = scores.log_softmax(-1)
                    for start_i, end_i, P, ent_blk, w_blk in live_slices:
                        kl_row = ent_blk - (P * log_q[start_i:end_i]).sum(-1)
                        full_sums[b] += float((kl_row * w_blk).sum())

        if use_dist and world_size > 1:
            comm_dev = _comm_device(adapter.E.device)
            flat_vec = [float(full_weight), float(grouped_weight), float(full_slots), float(total_slots)]
            flat_vec.extend(float(full_sums[b]) for b in betas)
            flat_vec.extend(float(grouped_sums[b]) for b in betas)
            vec_t = torch.tensor(flat_vec, dtype=torch.float64, device=comm_dev)
            dist.all_reduce(vec_t, op=dist.ReduceOp.SUM)
            red = vec_t.cpu().tolist()
            full_weight = float(red[0])
            grouped_weight = float(red[1])
            full_slots = int(round(red[2]))
            total_slots = int(round(red[3]))
            nb = len(betas)
            for idx_b, b in enumerate(betas):
                full_sums[b] = float(red[4 + idx_b])
                grouped_sums[b] = float(red[4 + nb + idx_b])

        use_live_full = full_slots > 0 and full_weight > 0.0
        denom = full_weight if use_live_full else max(grouped_weight, 1e-12)
        active_sums = full_sums if use_live_full else grouped_sums
        curve = [{'beta': b, 'kl': active_sums[b] / denom,
                  'dev_live_full_kl': (full_sums[b] / full_weight) if use_live_full else None,
                  'dev_union_rest_grouped_kl': grouped_sums[b] / max(grouped_weight, 1e-12)}
                 for b in betas]
        best_row = min(curve, key=lambda r: (r['kl'], r['beta']))
        base0 = active_sums[0.0] / denom
        uncal1 = active_sums[1.0] / denom
        return {
            'metric': 'dev_live_full_kl' if use_live_full else 'dev_union_rest_grouped_kl',
            'dev_live_full_logical_slots': full_slots,
            'dev_total_logical_slots': total_slots,
            'optimal_beta': best_row['beta'],
            'baseline_beta0_kl': base0,
            'uncalibrated_beta1_kl': uncal1,
            'calibrated_kl': best_row['kl'],
            'relative_gain_over_beta0': (base0 - best_row['kl']) / max(base0, 1e-12),
            'relative_gain_over_beta1': (uncal1 - best_row['kl']) / max(uncal1, 1e-12),
            'grid_curve': curve,
        }
    finally:
        adapter.train(was_training)


def optimizer_for(adapter, args, stage):
    adapter.b.requires_grad_(stage == 'dense_sparse')
    groups = [{'params': [adapter.W, adapter.A], 'weight_decay': args.weight_decay}]
    if stage == 'dense_sparse':
        groups.append({'params': [adapter.b], 'weight_decay': args.bias_weight_decay})
    opt_name = str(getattr(args, 'optimizer', 'adamw'))
    ema_cfg = getattr(args, 'ema_eval', None)
    ema_active = bool(opt_name == 'pacemuon' if ema_cfg is None else ema_cfg)
    if opt_name == 'adamw':
        dense = torch.optim.AdamW(groups, lr=args.learning_rate)
    elif opt_name == 'adam_atan2':
        dense = AdamATan2(groups, lr=args.learning_rate)
    elif opt_name == 'muon':
        dense = PaceMuon(groups, lr=args.learning_rate, fallback_mode='adamw',
                         ema_enabled=ema_active, pullback_c=0.0)
    elif opt_name == 'pacemuon':
        dense = PaceMuon(groups, lr=args.learning_rate, ema_enabled=True,
                         pullback_c=float(getattr(args, 'pace_pullback', 0.05)))
    elif opt_name == 'anchormuon':
        dense = AnchorMuon(groups, lr=args.learning_rate,
                           soda_lambda_scale=float(getattr(args, 'anchor_soda', 0.05)),
                           ema_enabled=ema_active)
    else:
        raise ValueError(f'Unknown dense optimizer: {opt_name}')
    sparse = torch.optim.SGD([adapter.sparse.values], lr=args.sparse_learning_rate) if adapter.sparse is not None else None
    return dense, sparse


@torch.no_grad()
def parameter_report(adapter):
    result = {name: {'norm': float(value.norm()), 'exact_zero_fraction': float((value == 0).float().mean()),
                     'shape': list(value.shape), 'dtype': str(value.dtype)}
              for name, value in [('W', adapter.W), ('A', adapter.A), ('b', adapter.b)]}
    if adapter.sparse is not None:
        s_rep = {'norm': float(adapter.sparse.values.norm()),
                 'exact_zero_fraction': float((adapter.sparse.values == 0).float().mean()),
                 'nonzero_buckets': int(adapter.sparse.values.count_nonzero()),
                 'capacity': adapter.sparse.capacity, 'memory_bytes': adapter.sparse.values.numel() * 4,
                 'optimizer': 'SGD followed by explicit proximal L1'}
        if isinstance(adapter.sparse, ExactObservedPairs):
            s_rep.update({
                'representation': 'exact_observed_pairs',
                'occupied_entries': adapter.sparse.num_occupied,
                'peak_occupied_entries': adapter.sparse.peak_occupied,
                'overflow_events': adapter.sparse.overflow_events,
                'dropped_unique_keys': adapter.sparse.dropped_unique_keys,
                'memory_bytes': (adapter.sparse.values.numel() * 4
                                 + adapter.sparse.key_table.numel() * 8
                                 + adapter.sparse.slot_table.numel() * 8),
            })
        result['S'] = s_rep
    return result


def checkpoint_payload(adapter, args, identity, stage, epoch, next_batch, dense_optimizer, sparse_optimizer, report):
    adapter_cfg = {'p': adapter.W.shape[1], 'r': adapter.E.shape[1], 'D': len(adapter.A),
                   'M_sparse': adapter.M_sparse,
                   'sparse_capacity': adapter.sparse.capacity if adapter.sparse is not None else None}
    sparse_backend = getattr(args, 'sparse_backend', 'signed_feature_hash')
    if sparse_backend != 'signed_feature_hash':
        adapter_cfg['sparse_backend'] = sparse_backend
        adapter_cfg['sparse_overflow_policy'] = getattr(args, 'sparse_overflow_policy', 'freeze_new')
    return {'schema_version': 1, 'scale': args.scale, 'identity': identity,
            'stage': stage, 'epoch': epoch, 'next_batch': next_batch,
            'adapter_state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            'adapter_config': adapter_cfg,
            'dense_optimizer': dense_optimizer.state_dict() if dense_optimizer is not None else None,
            'sparse_optimizer': sparse_optimizer.state_dict() if sparse_optimizer is not None else None,
            'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state(adapter.E.device) if adapter.E.is_cuda else None,
            'resolved_config': resolved_config(args), 'project_git_commit': git_commit(), 'report': report,
            'adapter_version': fingerprint({'identity': identity, 'stage': stage, 'epoch': epoch, 'next_batch': next_batch})}


def load_adapter_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['adapter_config']
    sparse_backend = config.get('sparse_backend', 'signed_feature_hash')
    if config['sparse_capacity'] is None:
        sparse = None
    elif sparse_backend == 'exact_observed_pairs':
        sparse = ExactObservedPairs(
            config['sparse_capacity'],
            vocab_size=checkpoint['adapter_state']['E'].shape[0],
            overflow_policy=config.get('sparse_overflow_policy', 'freeze_new'),
            device=device,
        )
    else:
        sparse = SignedHash(config['sparse_capacity'], device=device)
    adapter = DenseAdapter(checkpoint['adapter_state']['E'].to(device), config['p'], config['D'], sparse, config['M_sparse'])
    adapter.load_state_dict(checkpoint['adapter_state'])
    adapter.eval()
    return adapter, {key: value for key, value in checkpoint.items() if key not in
                     {'adapter_state', 'dense_optimizer', 'sparse_optimizer', 'torch_rng', 'python_rng', 'cuda_rng'}}


def _resolve_fit_regime(args):
    allow_long = bool(getattr(args, 'allow_long_oracle_fit', False))
    allow_t1 = bool(getattr(args, 'allow_pure_t1_ablation', False))
    if allow_long and allow_t1:
        raise ValueError('Cannot combine --allow_long_oracle_fit and --allow_pure_t1_ablation in one fit')
    sparse_backend = getattr(args, 'sparse_backend', 'signed_feature_hash')
    objective = getattr(args, 'objective', 'union_rest_grouped')
    block_shuffle = int(getattr(args, 'block_shuffle_window', 0) or 0)
    if block_shuffle < 0:
        raise ValueError('block_shuffle_window must be nonnegative')
    calibrate_beta = bool(getattr(args, 'calibrate_beta', False) or getattr(args, 'apply_calibrated_beta', False))
    custom_tag = getattr(args, 'm5_fit_tag', None)
    m5_aware_ns = any(hasattr(args, k) for k in ('sparse_backend', 'objective', 'm5_fit_tag', 'calibrate_beta'))
    if allow_long:
        regime = 'long_oracle_m5'
        eligibility_fn = long_oracle_eligible
        base = custom_tag or 'm5_long_oracle'
        subdir = f'{base}_smoke' if args.scale == 'smoke' else base
        training_dist = 'long_oracle_m5'
        scope = 'M5 long-condition oracle ablation shard sets; separate from primary M2 adaptation'
        warning = 'M5 ablation only: fitted on long-condition traces; never reported as primary adaptation'
    elif allow_t1:
        regime = 'pure_T1_calibration_ablation_m5'
        eligibility_fn = pure_t1_eligible
        base = custom_tag or 'm5_pure_t1_calibration'
        subdir = f'{base}_smoke' if args.scale == 'smoke' else base
        training_dist = 'pure_T1_calibration_ablation_m5'
        scope = 'M5 pure-T=1.0 unwarped calibration ablation shard sets; separate from primary M2 adaptation'
        warning = 'M5 calibration ablation only: fitted on pure T=1.0 unwarped traces; never reported as primary adaptation'
    else:
        regime = 'primary'
        eligibility_fn = primary_eligible
        training_dist = 'standard_nonthinking, greedy or model-card T=.7 topk20 topp.8 minp0 contexts; unwarped T_train=beta_train=1'
        scope = 'complete eligible train/dev/test shard sets; no long/pure-temperature fitting'
        warning = None
        if (custom_tag or sparse_backend != 'signed_feature_hash'
                or objective != 'union_rest_grouped' or calibrate_beta
                or (block_shuffle > 0 and m5_aware_ns)):
            shuf_suffix = f'_shuf{block_shuffle}' if block_shuffle > 0 else ''
            cal_suffix = '_calbeta' if calibrate_beta else ''
            base = custom_tag or f'm5_{objective}_{sparse_backend}{shuf_suffix}{cal_suffix}'
            subdir = f'{base}_smoke' if args.scale == 'smoke' else base
        else:
            subdir = 'm2_smoke' if args.scale == 'smoke' else 'm2'
    return regime, eligibility_fn, subdir, f'{subdir}.json', training_dist, scope, warning


def _apply_sparse_step(adapter, stage_sparse, local_sparse_pairs, local_raw_exact_keys,
                       *, use_dist: bool, world_size: int, device: torch.device,
                       sparse_lr: float, lambda_l1: float):
    rank_idx, rank_grads = _reduce_sparse_pairs(local_sparse_pairs, device)
    if use_dist:
        gathered_b = _all_gather_1d(rank_idx, world_size, device)
        gathered_g = _all_gather_1d(rank_grads, world_size, device)
        global_idx, global_grads = _reduce_sparse_pairs(list(zip(gathered_b, gathered_g)), device)
    else:
        global_idx, global_grads = rank_idx, rank_grads
    with torch.no_grad():
        if isinstance(stage_sparse, ExactObservedPairs):
            n_occ = stage_sparse.num_occupied
            if n_occ > 0 and global_idx.numel():
                active_keys = stage_sparse.key_table[:n_occ]
                active_slots = stage_sparse.slot_table[:n_occ]
                pos = torch.searchsorted(active_keys, global_idx)
                safe = pos.clamp_max(n_occ - 1)
                present = (pos < n_occ) & (active_keys[safe] == global_idx)
                if bool(present.any()):
                    stage_sparse.values.index_add_(
                        0, active_slots[safe[present]], global_grads[present], alpha=-sparse_lr
                    )
        else:
            if global_idx.numel():
                stage_sparse.values.index_add_(0, global_idx, global_grads, alpha=-sparse_lr)
        stage_sparse.proximal_step(sparse_lr, lambda_l1)


def _insert_exact_sparse_batch_keys(stage_sparse: ExactObservedPairs, local_raw_exact_keys: list[torch.Tensor],
                                    *, use_dist: bool, world_size: int, device: torch.device):
    rank_raw = (torch.cat(local_raw_exact_keys) if local_raw_exact_keys
                else torch.empty(0, dtype=torch.int64, device=device))
    if use_dist:
        gathered_raw = _all_gather_1d(rank_raw, world_size, device)
        global_raw = torch.cat(gathered_raw) if gathered_raw else torch.empty(0, dtype=torch.int64, device=device)
    else:
        global_raw = rank_raw
    if global_raw.numel():
        stage_sparse._insert_missing(global_raw)


def fit(entries, index_report, E, head, bias, head_metadata, args, *, stop_after_steps=None):
    """Two predeclared fit phases. Optional stop hook tests exact resumption."""
    rank = int(getattr(args, 'rank', 0))
    world_size = int(getattr(args, 'world_size', 1))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('rank must satisfy 0 <= rank < world_size')
    microbatch_blocks = getattr(args, 'microbatch_blocks', None)
    if microbatch_blocks is not None and int(microbatch_blocks) < 1:
        raise ValueError('microbatch_blocks must be positive when specified')
    use_dist = world_size > 1 and dist.is_available() and dist.is_initialized()
    if world_size > 1 and not use_dist:
        raise RuntimeError('world_size > 1 requires an initialized torch.distributed process group')
    is_main = rank == 0

    regime, eligibility_fn, subdir, json_name, training_dist, scope_text, ablation_warning = _resolve_fit_regime(args)
    sparse_backend = getattr(args, 'sparse_backend', 'signed_feature_hash')
    sparse_overflow_policy = getattr(args, 'sparse_overflow_policy', 'freeze_new')
    objective = getattr(args, 'objective', 'union_rest_grouped')
    block_shuffle = int(getattr(args, 'block_shuffle_window', 0) or 0)
    if block_shuffle < 0:
        raise ValueError('block_shuffle_window must be nonnegative')
    stats_memory_limit = getattr(args, 'stats_memory_limit', None) or (
        '32GB' if args.scale in ('reference', 'full') else '512MB'
    )

    outdir = args.output / subdir
    if is_main:
        outdir.mkdir(parents=True, exist_ok=True)
    if use_dist:
        dist.barrier()
    latest = outdir / 'latest.pt'
    config = resolved_config(args)
    non_default_optional = any(
        k in config and config[k] != default_val for k, default_val in OPT_IMPL_DEFAULT.items()
    )
    impl_files = ['train.py', 'adapter.py', 'features.py', 'sparse.py', 'sparse_stats.py', 'traces.py']
    if non_default_optional:
        impl_files.append('optimizers.py')
    implementation = {name: file_digest(Path(__file__).with_name(name)) for name in impl_files}
    identity_config = {
        k: v for k, v in config.items()
        if k not in TRANSIENT_CONFIG_KEYS and (k not in DEFAULT_OPTIONAL_CONFIG or v != DEFAULT_OPTIONAL_CONFIG[k])
    }
    identity = fingerprint({'config': identity_config, 'sources': index_report['source_fingerprint'], 'head': head_metadata,
                            'implementation_sha256': implementation,
                            'E_sha256': hashlib.sha256(E.detach().cpu().numpy().tobytes()).hexdigest()})
    stats_dir = args.output / ('m2_sparse_stats' if subdir in {'m2', 'm2_smoke'} else f'{subdir}_sparse_stats')
    stats_recorder = SparseAccessRecorder(stats_dir, identity, len(E)) if is_main else None
    references = {split: epoch_references(entries, split) for split in ['train', 'dev', 'test']}
    first = read_shard(entries[0].path)['blocks'][0]
    p = first['h'].shape[1]
    adapter = DenseAdapter(E, p, args.history_depth, M_sparse=args.M_sparse)
    opt_name = str(getattr(args, 'optimizer', 'adamw'))
    lr_schedule = str(getattr(args, 'lr_schedule', 'constant'))
    ema_cfg = getattr(args, 'ema_eval', None)
    ema_eval_active = bool(opt_name == 'pacemuon' if ema_cfg is None else ema_cfg)
    bf16_gemm = bool(getattr(args, 'bf16_gemm', False))
    num_workers = int(getattr(args, 'num_workers', 0))
    pin_memory = bool(getattr(args, 'pin_memory', False))
    use_compile = bool(getattr(args, 'compile', False))
    block_shuffle_window = max(0, block_shuffle)
    max_cached_shards = max(1, block_shuffle_window)
    dense_fn = _fresh_compile(_dense_correction_tensor, mode='default', dynamic=True) if use_compile else _dense_correction_tensor
    grouped_kl_fn = _fresh_compile(union_rest_grouped_precollated, mode='default', dynamic=True) if use_compile else union_rest_grouped_precollated
    dense_label = {'adamw': 'AdamW', 'adam_atan2': 'AdamATan2', 'muon': 'Muon',
                   'pacemuon': 'PaceMuon', 'anchormuon': 'AnchorMuon'}[opt_name]
    optimizer_report = {'dense': dense_label, 'learning_rate': args.learning_rate, 'weight_decay_W_A': args.weight_decay,
                        'weight_decay_b': args.bias_weight_decay, 'sparse': 'SGD+proximal_L1',
                        'sparse_learning_rate': args.sparse_learning_rate, 'lambda_l1': args.lambda_l1}
    if lr_schedule != 'constant':
        optimizer_report['lr_schedule'] = lr_schedule
    if ema_eval_active:
        optimizer_report['ema_eval'] = True
    if block_shuffle_window > 0:
        optimizer_report['block_shuffle_window'] = block_shuffle_window
    post_rejection_desc = ('committed verified stream only (no post-rejection canvas slots)'
                           if objective == 'two_sided_pseudolikelihood'
                           else 'off-policy teacher data retained with exp(-i/5) weights')
    report = {'scale': args.scale, 'data': 'synthetic' if args.scale == 'smoke' else args.scale,
              'identity': identity, 'trace_index': index_report, 'head_metadata': head_metadata,
              'model_pair': entries[0].identity['model_pair'],
              'model_revisions': entries[0].identity['checkpoint'],
              'implementation_sha256': implementation,
              'frozen_artifacts': {'hidden_projection_sha256': entries[0].identity.get('projection_sha256'),
                                   'token_features_sha256': entries[0].identity.get('token_features_sha256')},
              'optimizer': optimizer_report,
              'training_distribution': training_dist,
              'post_rejection_canvas': post_rejection_desc,
              'fixed_required_phases': ['dense', 'dense_sparse'], 'extra_tuning_or_sweeps': False, 'stages': {}}
    if ablation_warning is not None:
        report['ablation_warning'] = ablation_warning
    if objective != 'union_rest_grouped':
        report['objective'] = objective
    if sparse_backend != 'signed_feature_hash':
        report['sparse_backend'] = sparse_backend
        report['sparse_overflow_policy'] = sparse_overflow_policy
    if block_shuffle > 0:
        report['block_shuffle_window'] = block_shuffle
    resumed = None
    if latest.exists() and args.resume:
        resumed = torch.load(latest, map_location='cpu', weights_only=True)
        if resumed['identity'] != identity:
            raise ValueError('Checkpoint identity differs from config, traces, or frozen head/features; use a new output directory')
        report = resumed['report']
        adapter, _ = load_adapter_checkpoint(latest, E.device)
        torch.set_rng_state(resumed['torch_rng'])
        random.setstate(resumed['python_rng'])
        if E.is_cuda:
            torch.cuda.set_rng_state(resumed['cuda_rng'], E.device)
        if use_dist:
            dist.barrier()
        if resumed['stage'] == 'complete':
            if is_main:
                atomic_json(args.output / json_name, report)
            return report
    if not report.get('baseline'):
        report['baseline'] = {split: evaluate(adapter, refs, head, bias, args, head_metadata,
                                              rank=rank, world_size=world_size, use_dist=use_dist,
                                              eligibility_fn=eligibility_fn)
                              for split, refs in references.items()}
    if regime == 'primary' and any(report['baseline'][s]['live_full_diagnostic']['logical_slots'] == 0 for s in ['dev', 'test']):
        raise RuntimeError('Held-out dev/test live_full shards are required before fitting to report exact metrics')
    steps_this_run = 0
    begun = time.perf_counter()
    stages = [('dense', args.epochs_dense), ('dense_sparse', args.epochs_sparse)]
    for stage_index, (stage, epochs) in enumerate(stages):
        if resumed is not None and stages.index(next(x for x in stages if x[0] == resumed['stage'])) > stage_index:
            continue
        if stage == 'dense_sparse' and adapter.sparse is None:
            if sparse_backend == 'exact_observed_pairs':
                adapter.sparse = ExactObservedPairs(
                    args.sparse_capacity, vocab_size=len(E),
                    overflow_policy=sparse_overflow_policy, device=E.device,
                )
            else:
                adapter.sparse = SignedHash(args.sparse_capacity, device=E.device)
        adapter.train()
        dense_optimizer, sparse_optimizer = optimizer_for(adapter, args, stage)
        dense_params = [adapter.W, adapter.A] + ([adapter.b] if stage == 'dense_sparse' else [])
        start_epoch, start_batch = 0, 0
        if resumed is not None and resumed['stage'] == stage:
            dense_optimizer.load_state_dict(resumed['dense_optimizer'])
            if sparse_optimizer is not None:
                sparse_optimizer.load_state_dict(resumed['sparse_optimizer'])
            start_epoch, start_batch = resumed['epoch'], resumed['next_batch']
        if lr_schedule != 'constant':
            for group in dense_optimizer.param_groups:
                group.setdefault('base_lr', float(group['lr']))
        batches_per_epoch = max(1, (len(references['train']) + args.batch_blocks - 1) // args.batch_blocks)
        total_stage_steps = max(1, epochs * batches_per_epoch)
        non_blocking_h2d = bool(pin_memory and E.is_cuda)
        for epoch in range(start_epoch, epochs):
            adapter.train()
            epoch_seed = args.seed + 1000 * stage_index + epoch
            train_refs = epoch_references(entries, 'train', seed=epoch_seed,
                                          block_shuffle_window=block_shuffle_window)
            out_mult = head_metadata.get('output_multiplier', 1.)
            if objective == 'union_rest_grouped':
                batch_iter = iter_prepared_batches(
                    train_refs, args.batch_blocks, start_batch=start_batch, head_metadata=head_metadata,
                    rank=rank, world_size=world_size, microbatch_blocks=microbatch_blocks,
                    history_depth=args.history_depth, vocab_size=len(E), num_workers=num_workers,
                    pin_memory=pin_memory, loader_seed=epoch_seed + rank,
                    max_cached_shards=max_cached_shards, eligibility_fn=eligibility_fn,
                )
                for prepared in batch_iter:
                    batch_index = prepared.batch_index
                    for param in dense_params:
                        param.grad = torch.zeros_like(param)
                    local_weight = prepared.weight_sum.to(device=E.device, dtype=torch.float32, non_blocking=non_blocking_h2d)
                    if use_dist:
                        comm_dev = _comm_device(E.device)
                        w_comm = local_weight.to(comm_dev)
                        dist.all_reduce(w_comm, op=dist.ReduceOp.SUM)
                        global_weight = w_comm.to(E.device)
                    else:
                        global_weight = local_weight

                    local_loss_sum = torch.zeros((), device=E.device, dtype=torch.float32)
                    local_sparse_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
                    local_raw_exact_keys: list[torch.Tensor] = []
                    local_observed_keys: list[torch.Tensor] = []
                    is_exact_sparse = stage == 'dense_sparse' and isinstance(adapter.sparse, ExactObservedPairs)
                    if float(global_weight) > 0.0:
                        for mb in prepared.microbatches:
                            mb_dev = mb.to(E.device, non_blocking=non_blocking_h2d)
                            l_unit = _reconstruct_h_raw(mb_dev.h_raw_bf16, head, bias, vocab_chunk=args.vocab_chunk,
                                                        output_multiplier=out_mult, bf16_gemm=bf16_gemm)
                            if stage == 'dense_sparse':
                                compact_sparse = (_CompactTrainExactSparse(adapter.sparse)
                                                  if is_exact_sparse else _CompactTrainSparse(adapter.sparse))
                            else:
                                compact_sparse = None
                            corrected = _corrected_prepared_scores(
                                adapter, mb_dev.H, mb_dev.valid_rows, mb_dev.valid_cols, mb_dev.prev_ids, l_unit,
                                return_sparse_keys=stage == 'dense_sparse', sparse_override=compact_sparse,
                                bf16_gemm=bf16_gemm, dense_fn=dense_fn,
                            )
                            mb_scores = corrected[0] if stage == 'dense_sparse' else corrected
                            mb_loss = grouped_kl_fn(mb_scores, mb_dev.targets)
                            local_loss_sum = local_loss_sum + mb_loss.detach()
                            (mb_loss / global_weight).backward()
                            if stage == 'dense_sparse':
                                if is_exact_sparse:
                                    local_raw_exact_keys.extend(compact_sparse.raw_calls)
                                local_sparse_pairs.extend(compact_sparse.grad_pairs())
                                local_observed_keys.append(corrected[1])

                        if use_dist:
                            comm_dev = _comm_device(E.device)
                            loss_comm = local_loss_sum.to(comm_dev)
                            dist.all_reduce(loss_comm, op=dist.ReduceOp.SUM)
                            global_loss = loss_comm.to(E.device)
                        else:
                            global_loss = local_loss_sum
                        if not bool(torch.isfinite(global_loss / global_weight)):
                            raise FloatingPointError('Nonfinite training loss')

                        if is_exact_sparse:
                            _insert_exact_sparse_batch_keys(
                                adapter.sparse, local_raw_exact_keys,
                                use_dist=use_dist, world_size=world_size, device=E.device,
                            )

                        if use_dist:
                            comm_dev = _comm_device(E.device)
                            flat_grads = torch.cat([param.grad.reshape(-1) for param in dense_params]).to(comm_dev)
                            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                            flat_grads = flat_grads.to(E.device)
                            offset = 0
                            for param in dense_params:
                                numel = param.numel()
                                param.grad.copy_(flat_grads[offset:offset + numel].view_as(param))
                                offset += numel
                        if lr_schedule != 'constant':
                            step_one_based = epoch * batches_per_epoch + batch_index + 1
                            lr_mult = compute_lr_multiplier(step_one_based, total_stage_steps, schedule=lr_schedule)
                            for group in dense_optimizer.param_groups:
                                group['lr'] = float(group['base_lr']) * lr_mult
                        dense_optimizer.step()

                        if sparse_optimizer is not None:
                            _apply_sparse_step(
                                adapter, adapter.sparse, local_sparse_pairs, local_raw_exact_keys,
                                use_dist=use_dist, world_size=world_size, device=E.device,
                                sparse_lr=args.sparse_learning_rate, lambda_l1=args.lambda_l1,
                            )
                            rank_keys = (torch.cat(local_observed_keys) if local_observed_keys
                                         else torch.empty(0, dtype=torch.long, device=E.device))
                            if use_dist:
                                gathered_keys = _all_gather_1d(rank_keys, world_size, torch.device('cpu'))
                                global_keys = torch.cat(gathered_keys) if gathered_keys else torch.empty(0, dtype=torch.long)
                            else:
                                global_keys = rank_keys
                            if is_main:
                                chunk = stats_recorder.write_batch(global_keys, epoch, batch_index)
                                report.setdefault('sparse_statistics_chunks', []).append(chunk)

                    steps_this_run += 1
                    if steps_this_run % args.checkpoint_every == 0 or stop_after_steps == steps_this_run:
                        if is_main:
                            atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, stage, epoch, batch_index + 1,
                                                                         dense_optimizer, sparse_optimizer, report))
                        if use_dist:
                            dist.barrier()
                    if stop_after_steps == steps_this_run:
                        return {'status': 'interrupted_for_resume_test', 'checkpoint': str(latest)}
            else:
                for batch_index, blocks in batches(train_refs, args.batch_blocks, start_batch=start_batch,
                                                   head_metadata=head_metadata, rank=rank, world_size=world_size,
                                                   eligibility_fn=eligibility_fn, history_depth=args.history_depth,
                                                   shard_cache_size=max_cached_shards, num_workers=num_workers,
                                                   loader_seed=epoch_seed + rank):
                    for param in dense_params:
                        param.grad = torch.zeros_like(param)
                    local_weight_val = 0.0
                    for b in blocks:
                        k_b = int(b['block_length'])
                        comm_b = b['_verified_committed_window'] if '_verified_committed_window' in b else _block_own_committed(b)
                        n_ev = int(b['_verified_eval_slots']) if '_verified_eval_slots' in b else min(k_b, len(comm_b))
                        if n_ev > 0:
                            local_weight_val += float(loss_weights(n_ev, torch.device('cpu')).sum())
                    local_weight = torch.tensor(local_weight_val, dtype=torch.float32, device=E.device)
                    if use_dist:
                        comm_dev = _comm_device(E.device)
                        w_comm = local_weight.to(comm_dev)
                        dist.all_reduce(w_comm, op=dist.ReduceOp.SUM)
                        global_weight = w_comm.to(E.device)
                    else:
                        global_weight = local_weight

                    local_loss_sum = torch.zeros((), device=E.device, dtype=torch.float32)
                    local_sparse_pairs = []
                    local_raw_exact_keys = []
                    local_observed_keys = []
                    is_exact_sparse = stage == 'dense_sparse' and isinstance(adapter.sparse, ExactObservedPairs)
                    if float(global_weight) > 0.0:
                        mb_step = len(blocks) if microbatch_blocks is None or not blocks else max(1, int(microbatch_blocks))
                        for mb_start in range(0, len(blocks), max(1, mb_step)):
                            mb_blocks = blocks[mb_start:mb_start + mb_step]
                            if not mb_blocks:
                                continue
                            l_unit = reconstruct_batch(mb_blocks, head, bias, vocab_chunk=args.vocab_chunk,
                                                       output_multiplier=out_mult, bf16_gemm=bf16_gemm)
                            if stage == 'dense_sparse':
                                compact_sparse = (_CompactTrainExactSparse(adapter.sparse)
                                                  if is_exact_sparse else _CompactTrainSparse(adapter.sparse))
                            else:
                                compact_sparse = None
                            computed = two_sided_batch_scores_and_targets(
                                adapter, mb_blocks, l_unit, return_sparse_keys=stage == 'dense_sparse',
                                sparse_override=compact_sparse, bf16_gemm=bf16_gemm,
                            )
                            scores, targets, weights = computed[:3]
                            if len(targets) > 0:
                                mb_loss = two_sided_pseudolikelihood_loss(scores, targets, weights)
                                local_loss_sum = local_loss_sum + mb_loss.detach()
                                (mb_loss / global_weight).backward()
                            if stage == 'dense_sparse':
                                if is_exact_sparse:
                                    local_raw_exact_keys.extend(compact_sparse.raw_calls)
                                local_sparse_pairs.extend(compact_sparse.grad_pairs())
                                local_observed_keys.append(computed[3])

                        if use_dist:
                            comm_dev = _comm_device(E.device)
                            loss_comm = local_loss_sum.to(comm_dev)
                            dist.all_reduce(loss_comm, op=dist.ReduceOp.SUM)
                            global_loss = loss_comm.to(E.device)
                        else:
                            global_loss = local_loss_sum
                        if not bool(torch.isfinite(global_loss / global_weight)):
                            raise FloatingPointError('Nonfinite training loss')

                        if is_exact_sparse:
                            _insert_exact_sparse_batch_keys(
                                adapter.sparse, local_raw_exact_keys,
                                use_dist=use_dist, world_size=world_size, device=E.device,
                            )

                        if use_dist:
                            comm_dev = _comm_device(E.device)
                            flat_grads = torch.cat([param.grad.reshape(-1) for param in dense_params]).to(comm_dev)
                            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                            flat_grads = flat_grads.to(E.device)
                            offset = 0
                            for param in dense_params:
                                numel = param.numel()
                                param.grad.copy_(flat_grads[offset:offset + numel].view_as(param))
                                offset += numel
                        if lr_schedule != 'constant':
                            step_one_based = epoch * batches_per_epoch + batch_index + 1
                            lr_mult = compute_lr_multiplier(step_one_based, total_stage_steps, schedule=lr_schedule)
                            for group in dense_optimizer.param_groups:
                                group['lr'] = float(group['base_lr']) * lr_mult
                        dense_optimizer.step()
                        if sparse_optimizer is not None:
                            _apply_sparse_step(
                                adapter, adapter.sparse, local_sparse_pairs, local_raw_exact_keys,
                                use_dist=use_dist, world_size=world_size, device=E.device,
                                sparse_lr=args.sparse_learning_rate, lambda_l1=args.lambda_l1,
                            )
                            rank_keys = (torch.cat(local_observed_keys) if local_observed_keys
                                         else torch.empty(0, dtype=torch.long, device=E.device))
                            if use_dist:
                                gathered_keys = _all_gather_1d(rank_keys, world_size, torch.device('cpu'))
                                global_keys = torch.cat(gathered_keys) if gathered_keys else torch.empty(0, dtype=torch.long)
                            else:
                                global_keys = rank_keys
                            if is_main:
                                chunk = stats_recorder.write_batch(global_keys, epoch, batch_index)
                                report.setdefault('sparse_statistics_chunks', []).append(chunk)

                    steps_this_run += 1
                    if steps_this_run % args.checkpoint_every == 0 or stop_after_steps == steps_this_run:
                        if is_main:
                            atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, stage, epoch, batch_index + 1,
                                                                         dense_optimizer, sparse_optimizer, report))
                        if use_dist:
                            dist.barrier()
                    if stop_after_steps == steps_this_run:
                        return {'status': 'interrupted_for_resume_test', 'checkpoint': str(latest)}

            start_batch = 0
            if ema_eval_active and hasattr(dense_optimizer, 'use_ema_weights'):
                with dense_optimizer.use_ema_weights():
                    dev = evaluate(adapter, references['dev'], head, bias, args, head_metadata,
                                   rank=rank, world_size=world_size, use_dist=use_dist,
                                   eligibility_fn=eligibility_fn)
            else:
                dev = evaluate(adapter, references['dev'], head, bias, args, head_metadata,
                               rank=rank, world_size=world_size, use_dist=use_dist,
                               eligibility_fn=eligibility_fn)
            base = report['baseline']['dev']['weighted_mean_kl']
            gain = (base - dev['weighted_mean_kl']) / max(base, 1e-12)
            report.setdefault('epoch_development', []).append({'stage': stage, 'epoch': epoch, 'dev': dev,
                                                               'relative_improvement_over_zero': gain})
            if is_main:
                atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, stage, epoch + 1, 0,
                                                             dense_optimizer, sparse_optimizer, report))
            if gain < .01:
                report.setdefault('stopped_additional_epochs', []).append(stage)
                if is_main:
                    atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, stage, epochs, 0,
                                                                 dense_optimizer, sparse_optimizer, report))
                if use_dist:
                    dist.barrier()
                break
            if use_dist:
                dist.barrier()
        if ema_eval_active and hasattr(dense_optimizer, 'commit_ema'):
            dense_optimizer.commit_ema()
        adapter.eval()
        # Test is read only after each predeclared phase is finished; its metrics
        # never choose parameters, epochs, learning rates, or checkpoints.
        report['stages'][stage] = {split: evaluate(adapter, refs, head, bias, args, head_metadata,
                                                   rank=rank, world_size=world_size, use_dist=use_dist,
                                                   eligibility_fn=eligibility_fn)
                                    for split, refs in references.items()}
        report['stages'][stage]['parameters'] = parameter_report(adapter)
        if stage == 'dense_sparse':
            if is_main:
                collision_stats = stats_recorder.finalize(
                    report.get('sparse_statistics_chunks', []), adapter.sparse,
                    memory_limit=stats_memory_limit,
                    benchmark_entries=getattr(args, 'stats_benchmark_entries', 1_000_000), seed=args.seed)
                if isinstance(adapter.sparse, ExactObservedPairs) and isinstance(collision_stats, dict) and collision_stats.get('status') == 'measured':
                    cf_bench = collision_stats.get('lookup_benchmark')
                    if isinstance(cf_bench, dict):
                        cf_bench = dict(cf_bench, benchmarked_representation='counterfactual_signed_feature_hash')
                    collision_stats = dict(
                        collision_stats,
                        representation='exact_observed_pairs',
                        counterfactual_signed_hash_collision_count=collision_stats['collision_count'],
                        counterfactual_signed_hash_frequency_weighted_collisions=collision_stats['frequency_weighted_collision_count'],
                        collision_count=0,
                        frequency_weighted_collision_count=0,
                        collision_rate_per_nonempty_bucket=0.0,
                        frequency_weighted_collision_rate=0.0,
                        active_representation_collision_count=0,
                        occupied_exact_entries=adapter.sparse.num_occupied,
                        peak_occupied_exact_entries=adapter.sparse.peak_occupied,
                        overflow_events=adapter.sparse.overflow_events,
                        dropped_unique_keys=adapter.sparse.dropped_unique_keys,
                        lookup_benchmark=cf_bench,
                    )
                report['stages'][stage]['sparse_collision_statistics'] = collision_stats
            if use_dist:
                obj_box = [report['stages'][stage].get('sparse_collision_statistics') if is_main else None]
                dist.broadcast_object_list(obj_box, src=0)
                report['stages'][stage]['sparse_collision_statistics'] = obj_box[0]

        do_cal_beta = bool(getattr(args, 'calibrate_beta', False) or getattr(args, 'apply_calibrated_beta', False))
        apply_cal_beta = bool(getattr(args, 'apply_calibrated_beta', False))
        if do_cal_beta:
            cal = calibrate_beta_on_dev_live_full(
                adapter, references['dev'], head, bias, args, head_metadata,
                rank=rank, world_size=world_size, use_dist=use_dist,
                eligibility_fn=eligibility_fn,
                beta_grid=getattr(args, 'calibrate_beta_grid', None),
            )
            cal['applied_to_parameters'] = apply_cal_beta
            report['stages'][stage]['beta_calibration'] = cal
        stage_payload = checkpoint_payload(adapter, args, identity, stage, epochs, 0,
                                           dense_optimizer, sparse_optimizer, report) if is_main else None
        if apply_cal_beta:
            unscaled_path = outdir / f'{stage}_unscaled.pt'
            cal_path = outdir / f'{stage}_calibrated.pt'
            if is_main:
                atomic_torch_save(unscaled_path, stage_payload)
                scale_adapter_checkpoint(
                    unscaled_path, cal_path, cal['optimal_beta'],
                    extra_report={'calibrated_stage': stage, 'applied_beta': cal['optimal_beta']},
                    preserve_fit_identity=True,
                )
                scale_adapter_checkpoint(
                    unscaled_path, outdir / f'{stage}.pt', cal['optimal_beta'],
                    extra_report={'calibrated_stage': stage, 'applied_beta': cal['optimal_beta']},
                    preserve_fit_identity=True,
                )
            if use_dist:
                dist.barrier()
            cal_adapter, _ = load_adapter_checkpoint(outdir / f'{stage}.pt', E.device)
            report['stages'][stage]['calibrated_eval'] = {
                split: evaluate(cal_adapter, refs, head, bias, args, head_metadata,
                                rank=rank, world_size=world_size, use_dist=use_dist,
                                eligibility_fn=eligibility_fn)
                for split, refs in references.items()
            }
            if is_main:
                # Re-save stage checkpoint so its embedded report includes calibrated_eval.
                stage_payload_updated = checkpoint_payload(adapter, args, identity, stage, epochs, 0,
                                                           dense_optimizer, sparse_optimizer, report)
                atomic_torch_save(unscaled_path, stage_payload_updated)
                scale_adapter_checkpoint(
                    unscaled_path, cal_path, cal['optimal_beta'],
                    extra_report={'calibrated_stage': stage, 'applied_beta': cal['optimal_beta']},
                    preserve_fit_identity=True,
                )
                scale_adapter_checkpoint(
                    unscaled_path, outdir / f'{stage}.pt', cal['optimal_beta'],
                    extra_report={'calibrated_stage': stage, 'applied_beta': cal['optimal_beta']},
                    preserve_fit_identity=True,
                )
        else:
            if is_main:
                atomic_torch_save(outdir / f'{stage}.pt', stage_payload)
                if do_cal_beta:
                    scale_adapter_checkpoint(
                        outdir / f'{stage}.pt', outdir / f'{stage}_calibrated.pt', cal['optimal_beta'],
                        extra_report={'calibrated_stage': stage, 'applied_beta': cal['optimal_beta']},
                        preserve_fit_identity=True,
                    )
        if use_dist:
            dist.barrier()
        resumed = None
    do_cal_beta = bool(getattr(args, 'calibrate_beta', False) or getattr(args, 'apply_calibrated_beta', False))
    apply_cal_beta = bool(getattr(args, 'apply_calibrated_beta', False))
    if apply_cal_beta and 'calibrated_eval' in report['stages']['dense_sparse']:
        dev_after = report['stages']['dense_sparse']['calibrated_eval']['dev']['weighted_mean_kl']
    else:
        dev_after = report['stages']['dense_sparse']['dev']['weighted_mean_kl']
    final_gain = ((report['baseline']['dev']['weighted_mean_kl'] - dev_after)
                  / max(report['baseline']['dev']['weighted_mean_kl'], 1e-12))
    ckpt_map = {'dense': str(outdir / 'dense.pt'), 'dense_sparse': str(outdir / 'dense_sparse.pt')}
    if do_cal_beta:
        ckpt_map['dense_calibrated'] = str(outdir / 'dense_calibrated.pt')
        ckpt_map['dense_sparse_calibrated'] = str(outdir / 'dense_sparse_calibrated.pt')
    if apply_cal_beta:
        ckpt_map['dense_unscaled'] = str(outdir / 'dense_unscaled.pt')
        ckpt_map['dense_sparse_unscaled'] = str(outdir / 'dense_sparse_unscaled.pt')
    report.update({'status': 'passed', 'relative_dev_improvement': final_gain,
                   'under_one_percent_stop_tuning': final_gain < .01,
                   'fit_elapsed_seconds_this_invocation': time.perf_counter() - begun,
                   'checkpoints': ckpt_map,
                   'checkpoint_selection': 'last completed epoch of each predeclared phase; test never selects',
                   'scope': scope_text})
    if regime == 'primary' and any(report['stages']['dense_sparse'][s]['live_full_diagnostic']['logical_slots'] == 0 for s in ['dev', 'test']):
        raise RuntimeError('Held-out live_full shards are required to report grouped loss gap and exact overlap')
    if is_main:
        atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, 'complete', 0, 0, None, None, report))
        atomic_json(args.output / json_name, report)
    if use_dist:
        dist.barrier()
    return report


def smoke_bundle(args):
    """Create a small independent synthetic train/dev/test bundle without I/O downloads."""
    from .dflash_loop import generate_toy
    from .features import fit_hidden_projection, fit_token_features
    from .sampler import SamplingWarper
    from .toy import ToyParallelDrafter, ToyTarget
    from .traces import TraceShardWriter, compact_block

    target = ToyTarget(vocab_size=8, hidden_size=12, seed=args.seed)
    draft = ToyParallelDrafter(target, seed=args.seed + 1)
    calibration = [[i % 8, (i + 3) % 8, (i * 3 + 1) % 8] for i in range(64)]
    projection = fit_hidden_projection([draft.draft(x, 4).h_raw for x in calibration], 12, 6,
                                       seed=args.seed, prompt_ids=[f'synthetic-train-{i}' for i in range(64)])
    E, _ = fit_token_features(target.embedding, 4, args.seed, torch.tensor(sum(calibration, [])))
    allow_long = bool(getattr(args, 'allow_long_oracle_fit', False))
    allow_t1 = bool(getattr(args, 'allow_pure_t1_ablation', False))
    trace_subdir = ('m2_smoke_long_traces' if allow_long
                    else 'm2_smoke_t1_traces' if allow_t1
                    else 'm2_smoke_traces')
    writer = TraceShardWriter(args.output / trace_subdir)
    if allow_long:
        condition = 'long_thinking'
        temp_arms = [(.6, SamplingWarper(top_k=20, top_p=.95))]
        partition_name = 'long_oracle_train'
    elif allow_t1:
        condition = 'standard_nonthinking'
        temp_arms = [(1., SamplingWarper())]
        partition_name = 'diagnostic'
    else:
        condition = 'standard_nonthinking'
        temp_arms = [(0., SamplingWarper()), (.7, SamplingWarper(top_k=20, top_p=.8))]
        partition_name = 'primary'
    for split, count in [('train', 8), ('dev', 2), ('test', 2)]:
        for i in range(count):
            prompt = f'synthetic-{split}-{i}'
            prefix = [(i + len(split)) % 8, (3 * i + 1) % 8, (i + 4) % 8, i % 8]
            for temperature, warper in temp_arms:
                identity = {'model_pair': ['builtin-target', 'builtin-drafter'], 'prompt': prompt,
                            'condition': condition, 'seed': args.seed + i,
                            'temperature': temperature, 'verification_rule': 'greedy' if temperature == 0 else 'ratio',
                            'proposal_mode': 'plain_dflash', 'checkpoint': 'synthetic-v1', 'adapter_version': 0,
                            'sampling_warp_config': warper.to_dict()}
                if writer.completed(identity):
                    continue
                result = generate_toy(target, draft, prefix, 32, temperature=temperature, warper=warper,
                                      projection=projection, retain_tensors=True,
                                      generator=torch.Generator().manual_seed(args.seed + i))
                metadata = {'prompt_id': prompt, 'split': split, 'condition': condition,
                            'seed': args.seed + i, 'scale': 'smoke', 'model_pair': identity['model_pair'],
                            'projection_metadata': projection.metadata, 'history_depth': args.history_depth}
                blocks = [compact_block(record, metadata, M_draft=2, M_target=1,
                                         retain_live_full=split != 'train' or j == 0)
                          for j, record in enumerate(result.blocks)]
                writer.write(identity, blocks, {'generation': {'output_tokens': result.num_output_tokens,
                    'realized_length': result.num_output_tokens, 'condition': condition, 'split': split,
                    'seed': args.seed + i, 'temperature': temperature, 'sampling_warp_config': warper.to_dict(),
                    'partition': partition_name, 'complete': True, 'capped': True, 'eos_reached': False}})
    return args.output / trace_subdir, E, draft.draft_lm_head.weight, draft.draft_lm_head.bias, draft.head_metadata()


def _setup_distributed(args) -> bool:
    """Bind per-rank CUDA device under torchrun/launch and initialize process group if needed."""
    os.environ['RANK'] = str(args.rank)
    cuda_dev = None
    if str(args.device).startswith('cuda') and torch.cuda.is_available():
        if str(args.device) == 'cuda':
            args.device = 'cuda:0'
        if (args.world_size > 1 and 'LOCAL_RANK' in os.environ
                and str(args.device) == 'cuda:0' and torch.cuda.device_count() > 1):
            local_rank = int(os.environ['LOCAL_RANK'])
            if not 0 <= local_rank < torch.cuda.device_count():
                raise ValueError(f'LOCAL_RANK={local_rank} out of range for {torch.cuda.device_count()} visible CUDA devices')
            args.device = f'cuda:{local_rank}'
        cuda_dev = torch.device(str(args.device))
        if cuda_dev.index is None:
            cuda_dev = torch.device('cuda', 0)
        torch.cuda.set_device(cuda_dev)
        from .eval_closed_loop import apply_gpu_cpu_affinity
        apply_gpu_cpu_affinity(cuda_dev, args.rank)
    if args.world_size > 1 and not dist.is_initialized():
        backend = args.dist_backend or ('nccl' if str(args.device).startswith('cuda') else 'gloo')
        if backend == 'nccl':
            os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
        init_kwargs = {'backend': backend, 'rank': args.rank, 'world_size': args.world_size}
        if backend == 'nccl' and cuda_dev is not None:
            init_kwargs['device_id'] = cuda_dev
        dist.init_process_group(**init_kwargs)
        return True
    return False


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument('--trace_dir', type=Path, default=ROOT / 'artifacts/traces')
    parser.add_argument('--m1_audit', type=Path, default=ROOT / 'artifacts/m1_audit.json')
    parser.add_argument('--m1_diagnostic', type=Path)
    parser.add_argument('--features', type=Path, default=ROOT / 'artifacts/token_features.pt')
    parser.add_argument('--projection', type=Path, default=ROOT / 'artifacts/h_proj.pt')
    parser.add_argument('--prompts', type=Path, default=ROOT / 'artifacts/prompts.json')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--bias_weight_decay', type=float, default=0.)
    parser.add_argument('--sparse_learning_rate', type=float, default=.01)
    parser.add_argument('--lambda_l1', type=float, default=1e-5)
    parser.add_argument('--sparse_capacity', type=int, default=2**24)
    parser.add_argument('--M_sparse', type=int, default=256)
    parser.add_argument('--history_depth', type=int, default=4)
    parser.add_argument('--batch_blocks', type=int, default=8)
    parser.add_argument('--microbatch_blocks', type=int)
    parser.add_argument('--rank', type=int, default=int(os.environ.get('RANK', 0)))
    parser.add_argument('--world_size', type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    parser.add_argument('--dist_backend', choices=['nccl', 'gloo'])
    parser.add_argument('--vocab_chunk', type=int, default=8192)
    parser.add_argument('--epochs_dense', type=int, default=1)
    parser.add_argument('--epochs_sparse', type=int, default=1)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true', default=False)
    parser.add_argument('--bf16_gemm', action='store_true', default=False)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--optimizer', choices=['adamw', 'muon', 'pacemuon', 'anchormuon', 'adam_atan2'], default='adamw')
    parser.add_argument('--lr_schedule', choices=['constant', 'wsd', 'cosine'], default='constant')
    parser.add_argument('--ema_eval', action='store_true', default=None)
    parser.add_argument('--no_ema_eval', action='store_false', dest='ema_eval')
    parser.add_argument('--pace_pullback', type=float, default=0.05)
    parser.add_argument('--anchor_soda', type=float, default=0.05)
    parser.add_argument('--stats_memory_limit', default=None)
    parser.add_argument('--stats_benchmark_entries', type=int, default=1_000_000)
    parser.add_argument('--sparse_backend', choices=['signed_feature_hash', 'exact_observed_pairs'],
                        default='signed_feature_hash')
    parser.add_argument('--sparse_overflow_policy', choices=['freeze_new', 'error'], default='freeze_new')
    parser.add_argument('--objective', choices=['union_rest_grouped', 'two_sided_pseudolikelihood'],
                        default='union_rest_grouped')
    parser.add_argument('--allow_long_oracle_fit', action='store_true', default=False,
                        help='M5 ablation only: fit separate long-condition oracle adapter without overwriting m2/')
    parser.add_argument('--min_long_oracle_train_tokens', type=int, default=50_000)
    parser.add_argument('--allow_pure_t1_ablation', action='store_true', default=False,
                        help='M5 calibration ablation: fit separate pure-T=1.0 adapter without overwriting m2/')
    parser.add_argument('--max_primary_train_tokens', type=int, default=4_000_000)
    parser.add_argument('--block_shuffle_window', type=int, default=0,
                        help='Multi-shard i.i.d. block shuffle buffer size in shards (0 preserves shard-sequential default)')
    parser.add_argument('--calibrate_beta', action='store_true', default=False,
                        help='M5 post-fit scalar beta calibration on held-out dev live_full KL; writes *_calibrated.pt')
    parser.add_argument('--apply_calibrated_beta', action='store_true', default=False,
                        help='M5 post-fit scalar beta calibration applied directly into saved dense.pt/dense_sparse.pt')
    parser.add_argument('--calibrate_beta_grid', type=float, nargs='+', default=None,
                        help='Optional nonnegative scalar beta grid for --calibrate_beta')
    parser.add_argument('--scale_checkpoint_from', type=Path, default=None,
                        help='Standalone M5 utility: source adapter checkpoint to scale by --checkpoint_scale')
    parser.add_argument('--scale_checkpoint_to', type=Path, default=None,
                        help='Standalone M5 utility: destination path for scaled adapter checkpoint')
    parser.add_argument('--checkpoint_scale', type=float, default=None,
                        help='Standalone M5 utility: finite nonnegative scalar factor for --scale_checkpoint_from')
    parser.add_argument('--m5_fit_tag', default=None)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    args = parse_config(parser, argv)
    if (args.scale_checkpoint_from is not None or args.scale_checkpoint_to is not None
            or args.checkpoint_scale is not None):
        if args.scale_checkpoint_from is None or args.scale_checkpoint_to is None or args.checkpoint_scale is None:
            parser.error('--scale_checkpoint_from, --scale_checkpoint_to, and --checkpoint_scale must be provided together')
        if not math.isfinite(args.checkpoint_scale) or args.checkpoint_scale < 0.0:
            parser.error('--checkpoint_scale must be finite and >= 0')
        if args.rank == 0:
            record_run(args, 'scale-checkpoint')
        scaled_meta = scale_adapter_checkpoint(args.scale_checkpoint_from, args.scale_checkpoint_to, args.checkpoint_scale)
        if args.rank == 0:
            print(json.dumps({'status': 'scaled', **scaled_meta}, indent=2))
        return
    if args.calibrate_beta_grid is not None:
        if not args.calibrate_beta_grid or any(not math.isfinite(b) or b < 0.0 for b in args.calibrate_beta_grid):
            parser.error('--calibrate_beta_grid values must be finite and >= 0')
    if args.stats_memory_limit is None:
        args.stats_memory_limit = '32GB' if args.scale in ('reference', 'full') else '512MB'
    if args.m1_diagnostic is None:
        args.m1_diagnostic = args.output / f'diagnostic_{args.scale}.json'
    if min(args.batch_blocks, args.vocab_chunk, args.epochs_dense, args.epochs_sparse, args.checkpoint_every) < 1:
        raise ValueError('Training counts must be positive')
    if args.block_shuffle_window < 0:
        raise ValueError('block_shuffle_window must be nonnegative')
    if args.num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    if args.microbatch_blocks is not None and args.microbatch_blocks < 1:
        raise ValueError('microbatch_blocks must be positive')
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError('rank must satisfy 0 <= rank < world_size')
    if min(args.learning_rate, args.sparse_learning_rate) <= 0 or min(args.weight_decay, args.bias_weight_decay, args.lambda_l1) < 0:
        raise ValueError('Learning rates must be positive and regularization nonnegative')
    regime, eligibility_fn, *_ = _resolve_fit_regime(args)
    initialized_dist = _setup_distributed(args)
    try:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if args.scale == 'smoke':
            torch.set_num_threads(1)
            args.sparse_capacity = min(args.sparse_capacity, 256)
            args.M_sparse = min(args.M_sparse, 4)
            if args.rank == 0:
                args.trace_dir, E, head, bias, head_metadata = smoke_bundle(args)
            if args.world_size > 1:
                dist.barrier()
            if args.rank != 0:
                args.trace_dir, E, head, bias, head_metadata = smoke_bundle(args)
            entries, index_report = trace_index(args.trace_dir, eligibility_fn=eligibility_fn, regime=regime)
        else:
            if args.rank == 0:
                preflight(args, 2_000_000 if args.scale == 'full' else 500_000)
            if args.world_size > 1:
                dist.barrier()
            audit = json.loads(Path(args.m1_audit).read_text())
            if audit.get('status') != 'passed':
                raise RuntimeError('blocked: M1 audit has not passed')
            entries, index_report = trace_index(args.trace_dir, accepted_paths=audit['shards'],
                                                eligibility_fn=eligibility_fn, regime=regime)
            feature = torch.load(args.features, map_location='cpu', weights_only=True)
            projection = HiddenProjection.load(args.projection)
            validate_reference_gate(args, index_report, entries, projection, feature['metadata'])
            E = feature['E'].to(args.device, dtype=torch.float32)
            first = read_shard(entries[0].path)['blocks'][0]
            head_metadata = first['draft_lm_head']
            head, bias = load_frozen_head(head_metadata, args.device, output_dir=args.output)
        if args.rank == 0:
            record_run(args, 'train' if regime == 'primary' else f'train-{regime}')
        report = fit(entries, index_report, E, head, bias, head_metadata, args)
        if args.rank == 0:
            print(json.dumps({'scale': args.scale, 'status': report['status'],
                              'relative_dev_improvement': report['relative_dev_improvement'],
                              'under_one_percent_stop_tuning': report['under_one_percent_stop_tuning'],
                              'checkpoints': report['checkpoints']}, indent=2))
        return report
    finally:
        if initialized_dist and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
