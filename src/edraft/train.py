"""M2 streamed, resumable offline fitting on unwarped compact DFlash traces.

The only primary fitting contexts are standard non-thinking greedy and the
inspected model-card warped T=.7 arm. Long conditions and pure-T=1 diagnostics
never enter the primary optimizer. Real fitting requires the M1 audit gate.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .adapter import (DenseAdapter, PrecollatedGroupedTargets, collate_grouped_targets,
                      loss_weights, prev_token, union_rest_grouped, union_rest_grouped_precollated)
from .fastpath import _fresh_compile
from .features import HiddenProjection
from .optimizers import AdamATan2, AnchorMuon, PaceMuon, compute_lr_multiplier
from .runtime import ROOT, atomic_json, common_parser, fingerprint, git_commit, parse_config, preflight, record_run, resolved_config
from .sparse import SignedHash, hash_keys
from .sparse_stats import SparseAccessRecorder, pack_keys
from .traces import read_shard

TRANSIENT_CONFIG_KEYS = {
    'resume', 'rank', 'world_size', 'dist_backend', 'microbatch_blocks',
    'num_workers', 'pin_memory', 'compile',
}
DEFAULT_OPTIONAL_CONFIG = {
    'bf16_gemm': False,
    'optimizer': 'adamw',
    'lr_schedule': 'constant',
    'ema_eval': None,
    'pace_pullback': 0.05,
    'anchor_soda': 0.05,
    'block_shuffle_window': 0,
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
        self.unique_buckets: torch.Tensor | None = None
        self.touched_values: torch.Tensor | None = None

    def lookup(self, d, previous, current):
        buckets, signs = hash_keys(d, previous, current, self.capacity, self.base.values.device)
        unique, inverse = torch.unique(buckets.reshape(-1), sorted=True, return_inverse=True)
        touched = self.base.values.index_select(0, unique).detach().requires_grad_(True)
        self.unique_buckets = unique
        self.touched_values = touched
        return touched[inverse].reshape(buckets.shape) * signs


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


def trace_index(directory, accepted_paths=None):
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
        if not primary_eligible(candidate):
            reason = f"{split}/{identity['condition']}/T={identity['temperature']}"
            excluded[reason] = excluded.get(reason, 0) + generation_length(generation)
            continue
        if identity.get('adapter_version') != 0 or identity.get('proposal_mode') != 'plain_dflash':
            raise ValueError('M2 requires plain DFlash traces collected before fitting any adapter')
        entries.append(ShardInfo(path, split, prompt, generation_length(generation),
                                 int(side['blocks']), side['sha256'], identity))
    if not entries:
        raise ValueError('No eligible primary traces')
    summaries = {split: {'generated_output_tokens': sum(x.output_tokens for x in entries if x.split == split),
                        'prompts': len({x.prompt_id for x in entries if x.split == split}),
                        'shards': sum(x.split == split for x in entries),
                        'blocks': sum(x.blocks for x in entries if x.split == split)}
                 for split in ['train', 'dev', 'test']}
    if any(summaries[s]['blocks'] == 0 for s in summaries):
        raise ValueError('Primary train, development, and test traces are all required')
    return entries, {'primary': summaries, 'excluded_generated_tokens': excluded,
                     'primary_train_tokens_by_temperature': {str(T): sum(e.output_tokens for e in entries if e.split == 'train' and e.identity['temperature'] == T) for T in [0., .7]},
                     'source_fingerprint': fingerprint([{'identity': e.identity, 'sha256': e.sha256} for e in entries])}


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
    if index_report['primary']['train']['generated_output_tokens'] < floor:
        raise RuntimeError(f'blocked: primary standard train traces have fewer than {floor} generated tokens')
    if any(index_report['primary_train_tokens_by_temperature'][str(T)] < floor // 2 for T in [0., .7]):
        raise RuntimeError('blocked: each primary collection temperature must supply half the training floor')
    if index_report['primary']['test']['prompts'] < 64:
        raise RuntimeError('blocked: fewer than 64 standard test prompts')
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


def load_frozen_head(metadata, device):
    """Load exactly the referenced output projection, without the target model."""
    from safetensors import safe_open

    location = Path(metadata['artifact_id'])
    if not location.is_dir():
        env = json.loads((ROOT / 'artifacts/environment.json').read_text())
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
    chosen = [x for x in entries if x.split == split]
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(chosen)
    win = int(block_shuffle_window)
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
                 history_depth: int = 4, vocab_size: int, max_cached_shards: int = 1):
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
        self._cache: OrderedDict[Path, dict] = OrderedDict()

    def __len__(self) -> int:
        return max(0, self.total_batches - self.start_batch)

    def _load_block(self, shard: ShardInfo, index: int) -> dict:
        payload = self._cache.get(shard.path)
        if payload is not None:
            self._cache.move_to_end(shard.path)
        else:
            if json.loads(shard.path.with_suffix('.pt.json').read_text())['sha256'] != shard.sha256:
                raise ValueError('An indexed shard changed after the frozen training manifest was built')
            payload = read_shard(shard.path)
            if len(payload['blocks']) != shard.blocks:
                raise ValueError('Shard block count differs from immutable sidecar')
            self._cache[shard.path] = payload
            while len(self._cache) > self.max_cached_shards:
                self._cache.popitem(last=False)
        block = payload['blocks'][index]
        if block['split'] != shard.split or block['prompt_id'] != shard.prompt_id or not primary_eligible(block):
            raise ValueError('Block metadata differs from eligible primary shard identity')
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


def iter_prepared_batches(references, batch_blocks: int, *, start_batch: int = 0, head_metadata=None,
                          rank: int = 0, world_size: int = 1, microbatch_blocks: int | None = None,
                          history_depth: int = 4, vocab_size: int, num_workers: int = 0,
                          pin_memory: bool = False, loader_seed: int = 0, max_cached_shards: int = 1):
    dataset = _ShardBatchDataset(
        references, batch_blocks, start_batch=start_batch, head_metadata=head_metadata,
        rank=rank, world_size=world_size, microbatch_blocks=microbatch_blocks,
        history_depth=history_depth, vocab_size=vocab_size, max_cached_shards=max_cached_shards,
    )
    use_pin = bool(pin_memory and torch.cuda.is_available())
    if int(num_workers) <= 0:
        for i in range(len(dataset)):
            item = dataset[i]
            yield item.pin_memory() if use_pin else item
        return
    gen = torch.Generator()
    gen.manual_seed(int(loader_seed))
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=int(num_workers),
        pin_memory=use_pin,
        prefetch_factor=2,
        persistent_workers=False,
        multiprocessing_context='spawn',
        generator=gen,
    )
    yield from loader


def batches(references, batch_blocks, *, start_batch=0, head_metadata=None, rank=0, world_size=1):
    """Only one compact shard and one block batch are resident at a time."""
    cached_path, payload = None, None
    for batch_index in range(start_batch, (len(references) + batch_blocks - 1) // batch_blocks):
        selected = references[batch_index * batch_blocks:(batch_index + 1) * batch_blocks]
        if world_size > 1:
            selected = selected[rank::world_size]
        result = []
        for shard, index in selected:
            if shard.path != cached_path:
                if json.loads(shard.path.with_suffix('.pt.json').read_text())['sha256'] != shard.sha256:
                    raise ValueError('An indexed shard changed after the frozen training manifest was built')
                payload = read_shard(shard.path)
                cached_path = shard.path
                if len(payload['blocks']) != shard.blocks:
                    raise ValueError('Shard block count differs from immutable sidecar')
            block = payload['blocks'][index]
            if block['split'] != shard.split or block['prompt_id'] != shard.prompt_id or not primary_eligible(block):
                raise ValueError('Block metadata differs from eligible primary shard identity')
            if head_metadata is not None and block['draft_lm_head'] != head_metadata:
                raise ValueError('Mixed frozen head artifacts are forbidden in one fit')
            result.append(block)
        yield batch_index, result


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
                               bf16_gemm: bool = False, dense_fn=None):
    E_bf16 = adapter._token_features_bf16() if bf16_gemm else None
    fn = _dense_correction_tensor if dense_fn is None else dense_fn
    dense = fn(adapter.W, adapter.A, adapter.b, adapter.E, E_bf16, H, valid_rows, valid_cols, prev_ids,
               bf16_gemm=bf16_gemm)
    scores = l_unit - dense
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
        scores = scores.scatter_add(1, candidates, -values)
        if return_sparse_keys:
            observed_keys = pack_keys(d_pair, prev_pair, cands_pair, adapter.E.shape[0]).reshape(-1)
    if return_sparse_keys:
        if observed_keys is None:
            observed_keys = torch.empty(0, dtype=torch.long, device=scores.device)
        return scores, observed_keys
    return scores


def corrected_batch_scores(adapter, blocks, l_unit, *, return_sparse_keys=False, sparse_override=None, bf16_gemm=False):
    """Batched teacher-forced dense features; sampled canvas IDs stay constants."""
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
        return_sparse_keys=return_sparse_keys, sparse_override=sparse_override, bf16_gemm=bf16_gemm,
    )


def batch_targets(blocks, device):
    ids, probabilities, rest, weights = [], [], [], []
    for block in blocks:
        ids.extend(x.to(device) for x in block['union_ids'])
        probabilities.extend(x.to(device) for x in block['target_probs_union'])
        rest.extend(x.to(device) for x in block['target_rest_mass'])
        weights.append(loss_weights(block['block_length'], device))
    return ids, probabilities, rest, torch.cat(weights)


def batch_loss(adapter, blocks, l_unit, *, collect_sparse_keys=False, sparse_override=None, bf16_gemm=False):
    corrected = corrected_batch_scores(adapter, blocks, l_unit, return_sparse_keys=collect_sparse_keys,
                                       sparse_override=sparse_override, bf16_gemm=bf16_gemm)
    scores = corrected[0] if collect_sparse_keys else corrected
    ids, probabilities, rest, weights = batch_targets(blocks, scores.device)
    result = (union_rest_grouped(scores, ids, probabilities, rest, weights), weights.sum(), scores)
    return (*result, corrected[1]) if collect_sparse_keys else result


@torch.no_grad()
def evaluate(adapter, references, head, bias, args, head_metadata, *, rank=0, world_size=1, use_dist=False):
    bf16_gemm = bool(getattr(args, 'bf16_gemm', False))
    sums = dict(loss=0., weight=0., slots=0, union_overlap_lower=0., grouped_overlap_upper=0.,
                full_overlap=0., full_top1=0., full_slots=0, full_loss=0., full_grouped_loss=0., full_weight=0.)
    for _, blocks in batches(references, args.batch_blocks, head_metadata=head_metadata,
                             rank=rank, world_size=world_size):
        if not blocks:
            continue
        l_unit = reconstruct_batch(blocks, head, bias, vocab_chunk=args.vocab_chunk,
                                   output_multiplier=head_metadata.get('output_multiplier', 1.),
                                   bf16_gemm=bf16_gemm)
        loss, weight, scores = batch_loss(adapter, blocks, l_unit, bf16_gemm=bf16_gemm)
        q_log = scores.log_softmax(-1)
        q = q_log.exp()
        sums['loss'] += float(loss)
        sums['weight'] += float(weight)
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
                full_kl = (torch.xlogy(P, P) - P * log_rows).sum(-1)
                grouped = union_rest_grouped(scores[cursor:cursor + K], ids, probabilities, rest, weights)
                sums['full_loss'] += float((full_kl * weights).sum())
                sums['full_grouped_loss'] += float(grouped)
                sums['full_weight'] += float(weights.sum())
                sums['full_overlap'] += float(torch.minimum(P, rows).sum())
                sums['full_top1'] += int((P.argmax(-1) == rows.argmax(-1)).sum())
                sums['full_slots'] += K
            sums['slots'] += K
            cursor += K
    if use_dist and world_size > 1:
        keys = ('loss', 'weight', 'slots', 'union_overlap_lower', 'grouped_overlap_upper',
                'full_overlap', 'full_top1', 'full_slots', 'full_loss', 'full_grouped_loss', 'full_weight')
        comm_dev = _comm_device(adapter.E.device)
        vec = torch.tensor([float(sums[k]) for k in keys], dtype=torch.float64, device=comm_dev)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        reduced = vec.cpu().tolist()
        for k, val in zip(keys, reduced):
            sums[k] = int(round(val)) if k in ('slots', 'full_slots', 'full_top1') else float(val)
    full = sums['full_slots']
    return {'loss_mode': 'union_rest_grouped', 'weighted_mean_kl': sums['loss'] / sums['weight'],
            'weighted_kl_sum': sums['loss'], 'weight_sum': sums['weight'], 'logical_slots': sums['slots'],
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
        result['S'] = {'norm': float(adapter.sparse.values.norm()),
                       'exact_zero_fraction': float((adapter.sparse.values == 0).float().mean()),
                       'nonzero_buckets': int(adapter.sparse.values.count_nonzero()),
                       'capacity': adapter.sparse.capacity, 'memory_bytes': adapter.sparse.values.numel() * 4,
                       'optimizer': 'SGD followed by explicit proximal L1'}
    return result


def checkpoint_payload(adapter, args, identity, stage, epoch, next_batch, dense_optimizer, sparse_optimizer, report):
    return {'schema_version': 1, 'scale': args.scale, 'identity': identity,
            'stage': stage, 'epoch': epoch, 'next_batch': next_batch,
            'adapter_state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            'adapter_config': {'p': adapter.W.shape[1], 'r': adapter.E.shape[1], 'D': len(adapter.A),
                               'M_sparse': adapter.M_sparse,
                               'sparse_capacity': adapter.sparse.capacity if adapter.sparse is not None else None},
            'dense_optimizer': dense_optimizer.state_dict() if dense_optimizer is not None else None,
            'sparse_optimizer': sparse_optimizer.state_dict() if sparse_optimizer is not None else None,
            'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state(adapter.E.device) if adapter.E.is_cuda else None,
            'resolved_config': resolved_config(args), 'project_git_commit': git_commit(), 'report': report,
            'adapter_version': fingerprint({'identity': identity, 'stage': stage, 'epoch': epoch, 'next_batch': next_batch})}


def load_adapter_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['adapter_config']
    sparse = SignedHash(config['sparse_capacity'], device=device) if config['sparse_capacity'] is not None else None
    adapter = DenseAdapter(checkpoint['adapter_state']['E'].to(device), config['p'], config['D'], sparse, config['M_sparse'])
    adapter.load_state_dict(checkpoint['adapter_state'])
    adapter.eval()
    return adapter, {key: value for key, value in checkpoint.items() if key not in
                     {'adapter_state', 'dense_optimizer', 'sparse_optimizer', 'torch_rng', 'python_rng', 'cuda_rng'}}


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

    outdir = args.output / ('m2_smoke' if args.scale == 'smoke' else 'm2')
    if is_main:
        outdir.mkdir(parents=True, exist_ok=True)
    if use_dist:
        dist.barrier()
    latest = outdir / 'latest.pt'
    config = resolved_config(args)
    non_default_optional = any(
        k in config and config[k] != default_val for k, default_val in DEFAULT_OPTIONAL_CONFIG.items()
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
    stats_recorder = SparseAccessRecorder(args.output / 'm2_sparse_stats', identity, len(E)) if is_main else None
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
    block_shuffle_window = max(0, int(getattr(args, 'block_shuffle_window', 0)))
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
    report = {'scale': args.scale, 'data': 'synthetic' if args.scale == 'smoke' else args.scale,
              'identity': identity, 'trace_index': index_report, 'head_metadata': head_metadata,
              'model_pair': entries[0].identity['model_pair'],
              'model_revisions': entries[0].identity['checkpoint'],
              'implementation_sha256': implementation,
              'frozen_artifacts': {'hidden_projection_sha256': entries[0].identity.get('projection_sha256'),
                                   'token_features_sha256': entries[0].identity.get('token_features_sha256')},
              'optimizer': optimizer_report,
              'training_distribution': 'standard_nonthinking, greedy or model-card T=.7 topk20 topp.8 minp0 contexts; unwarped T_train=beta_train=1',
              'post_rejection_canvas': 'off-policy teacher data retained with exp(-i/5) weights',
              'fixed_required_phases': ['dense', 'dense_sparse'], 'extra_tuning_or_sweeps': False, 'stages': {}}
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
                atomic_json(args.output / ('m2_smoke.json' if args.scale == 'smoke' else 'm2.json'), report)
            return report
    if not report.get('baseline'):
        report['baseline'] = {split: evaluate(adapter, refs, head, bias, args, head_metadata,
                                              rank=rank, world_size=world_size, use_dist=use_dist)
                              for split, refs in references.items()}
    if any(report['baseline'][s]['live_full_diagnostic']['logical_slots'] == 0 for s in ['dev', 'test']):
        raise RuntimeError('Held-out dev/test live_full shards are required before fitting to report exact metrics')
    steps_this_run = 0
    begun = time.perf_counter()
    stages = [('dense', args.epochs_dense), ('dense_sparse', args.epochs_sparse)]
    for stage_index, (stage, epochs) in enumerate(stages):
        if resumed is not None and stages.index(next(x for x in stages if x[0] == resumed['stage'])) > stage_index:
            continue
        if stage == 'dense_sparse' and adapter.sparse is None:
            adapter.sparse = SignedHash(args.sparse_capacity, device=E.device)
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
            epoch_seed = args.seed + 1000 * stage_index + epoch
            train_refs = epoch_references(entries, 'train', seed=epoch_seed,
                                          block_shuffle_window=block_shuffle_window)
            for prepared in iter_prepared_batches(
                train_refs, args.batch_blocks, start_batch=start_batch, head_metadata=head_metadata,
                rank=rank, world_size=world_size, microbatch_blocks=microbatch_blocks,
                history_depth=args.history_depth, vocab_size=len(E), num_workers=num_workers,
                pin_memory=pin_memory, loader_seed=epoch_seed + rank,
                max_cached_shards=max_cached_shards,
            ):
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
                local_observed_keys: list[torch.Tensor] = []
                out_mult = head_metadata.get('output_multiplier', 1.)
                for mb in prepared.microbatches:
                    mb_dev = mb.to(E.device, non_blocking=non_blocking_h2d)
                    l_unit = _reconstruct_h_raw(mb_dev.h_raw_bf16, head, bias, vocab_chunk=args.vocab_chunk,
                                                output_multiplier=out_mult, bf16_gemm=bf16_gemm)
                    compact_sparse = _CompactTrainSparse(adapter.sparse) if stage == 'dense_sparse' else None
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
                        if compact_sparse.unique_buckets is not None and compact_sparse.unique_buckets.numel():
                            grad = compact_sparse.touched_values.grad
                            local_sparse_pairs.append((
                                compact_sparse.unique_buckets,
                                grad.detach() if grad is not None else torch.zeros_like(compact_sparse.touched_values),
                            ))
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
                    rank_buckets, rank_grads = _reduce_sparse_pairs(local_sparse_pairs, E.device)
                    if use_dist:
                        gathered_b = _all_gather_1d(rank_buckets, world_size, E.device)
                        gathered_g = _all_gather_1d(rank_grads, world_size, E.device)
                        global_buckets, global_grads = _reduce_sparse_pairs(
                            list(zip(gathered_b, gathered_g)), E.device)
                    else:
                        global_buckets, global_grads = rank_buckets, rank_grads
                    with torch.no_grad():
                        if global_buckets.numel():
                            adapter.sparse.values.index_add_(
                                0, global_buckets, global_grads, alpha=-args.sparse_learning_rate)
                        adapter.sparse.proximal_step(args.sparse_learning_rate, args.lambda_l1)

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
                                   rank=rank, world_size=world_size, use_dist=use_dist)
            else:
                dev = evaluate(adapter, references['dev'], head, bias, args, head_metadata,
                               rank=rank, world_size=world_size, use_dist=use_dist)
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
        # Test is read only after each predeclared phase is finished; its metrics
        # never choose parameters, epochs, learning rates, or checkpoints.
        report['stages'][stage] = {split: evaluate(adapter, refs, head, bias, args, head_metadata,
                                                   rank=rank, world_size=world_size, use_dist=use_dist)
                                    for split, refs in references.items()}
        report['stages'][stage]['parameters'] = parameter_report(adapter)
        if stage == 'dense_sparse' and is_main:
            report['stages'][stage]['sparse_collision_statistics'] = stats_recorder.finalize(
                report.get('sparse_statistics_chunks', []), adapter.sparse,
                memory_limit=getattr(args, 'stats_memory_limit', '512MB'),
                benchmark_entries=getattr(args, 'stats_benchmark_entries', 1_000_000), seed=args.seed)
        if is_main:
            atomic_torch_save(outdir / f'{stage}.pt', checkpoint_payload(adapter, args, identity, stage, epochs, 0,
                                                                        dense_optimizer, sparse_optimizer, report))
        if use_dist:
            dist.barrier()
        resumed = None
    final_gain = ((report['baseline']['dev']['weighted_mean_kl'] - report['stages']['dense_sparse']['dev']['weighted_mean_kl'])
                  / max(report['baseline']['dev']['weighted_mean_kl'], 1e-12))
    report.update({'status': 'passed', 'relative_dev_improvement': final_gain,
                   'under_one_percent_stop_tuning': final_gain < .01,
                   'fit_elapsed_seconds_this_invocation': time.perf_counter() - begun,
                   'checkpoints': {'dense': str(outdir / 'dense.pt'), 'dense_sparse': str(outdir / 'dense_sparse.pt')},
                   'checkpoint_selection': 'last completed epoch of each predeclared phase; test never selects',
                   'scope': 'complete eligible train/dev/test shard sets; no long/pure-temperature fitting'})
    if any(report['stages']['dense_sparse'][s]['live_full_diagnostic']['logical_slots'] == 0 for s in ['dev', 'test']):
        raise RuntimeError('Held-out live_full shards are required to report grouped loss gap and exact overlap')
    if is_main:
        atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, 'complete', 0, 0, None, None, report))
        atomic_json(args.output / ('m2_smoke.json' if args.scale == 'smoke' else 'm2.json'), report)
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
    writer = TraceShardWriter(args.output / 'm2_smoke_traces')
    for split, count in [('train', 8), ('dev', 2), ('test', 2)]:
        for i in range(count):
            prompt = f'synthetic-{split}-{i}'
            prefix = [(i + len(split)) % 8, (3 * i + 1) % 8, (i + 4) % 8, i % 8]
            for temperature in [0., .7]:
                warper = SamplingWarper() if temperature == 0 else SamplingWarper(top_k=20, top_p=.8)
                identity = {'model_pair': ['builtin-target', 'builtin-drafter'], 'prompt': prompt,
                            'condition': 'standard_nonthinking', 'seed': args.seed + i,
                            'temperature': temperature, 'verification_rule': 'greedy' if temperature == 0 else 'ratio',
                            'proposal_mode': 'plain_dflash', 'checkpoint': 'synthetic-v1', 'adapter_version': 0,
                            'sampling_warp_config': warper.to_dict()}
                if writer.completed(identity):
                    continue
                result = generate_toy(target, draft, prefix, 32, temperature=temperature, warper=warper,
                                      projection=projection, retain_tensors=True,
                                      generator=torch.Generator().manual_seed(args.seed + i))
                metadata = {'prompt_id': prompt, 'split': split, 'condition': 'standard_nonthinking',
                            'seed': args.seed + i, 'scale': 'smoke', 'model_pair': identity['model_pair'],
                            'projection_metadata': projection.metadata, 'history_depth': args.history_depth}
                blocks = [compact_block(record, metadata, M_draft=2, M_target=1,
                                         retain_live_full=split != 'train' or j == 0)
                          for j, record in enumerate(result.blocks)]
                writer.write(identity, blocks, {'generation': {'output_tokens': result.num_output_tokens,
                    'realized_length': result.num_output_tokens, 'condition': 'standard_nonthinking', 'split': split,
                    'seed': args.seed + i, 'temperature': temperature, 'sampling_warp_config': warper.to_dict(),
                    'partition': 'primary', 'complete': True, 'capped': True, 'eos_reached': False}})
    return args.output / 'm2_smoke_traces', E, draft.draft_lm_head.weight, draft.draft_lm_head.bias, draft.head_metadata()


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
    if args.world_size > 1 and not dist.is_initialized():
        backend = args.dist_backend or ('nccl' if str(args.device).startswith('cuda') else 'gloo')
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
    parser.add_argument('--block_shuffle_window', type=int, default=0)
    parser.add_argument('--stats_memory_limit', default='512MB')
    parser.add_argument('--stats_benchmark_entries', type=int, default=1_000_000)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    args = parse_config(parser, argv)
    if args.m1_diagnostic is None:
        args.m1_diagnostic = args.output / f'diagnostic_{args.scale}.json'
    if min(args.batch_blocks, args.vocab_chunk, args.epochs_dense, args.epochs_sparse, args.checkpoint_every) < 1:
        raise ValueError('Training counts must be positive')
    if args.num_workers < 0:
        raise ValueError('num_workers must be nonnegative')
    if args.microbatch_blocks is not None and args.microbatch_blocks < 1:
        raise ValueError('microbatch_blocks must be positive')
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError('rank must satisfy 0 <= rank < world_size')
    if min(args.learning_rate, args.sparse_learning_rate) <= 0 or min(args.weight_decay, args.bias_weight_decay, args.lambda_l1) < 0:
        raise ValueError('Learning rates must be positive and regularization nonnegative')
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
            entries, index_report = trace_index(args.trace_dir)
        else:
            if args.rank == 0:
                preflight(args, 500_000)
            if args.world_size > 1:
                dist.barrier()
            audit = json.loads(Path(args.m1_audit).read_text())
            if audit.get('status') != 'passed':
                raise RuntimeError('blocked: M1 audit has not passed')
            entries, index_report = trace_index(args.trace_dir, accepted_paths=audit['shards'])
            feature = torch.load(args.features, map_location='cpu', weights_only=True)
            projection = HiddenProjection.load(args.projection)
            validate_reference_gate(args, index_report, entries, projection, feature['metadata'])
            E = feature['E'].to(args.device, dtype=torch.float32)
            first = read_shard(entries[0].path)['blocks'][0]
            head_metadata = first['draft_lm_head']
            head, bias = load_frozen_head(head_metadata, args.device)
        record_run(args, 'train')
        report = fit(entries, index_report, E, head, bias, head_metadata, args)
        if args.rank == 0:
            print(json.dumps({'scale': args.scale, 'status': report['status'],
                              'relative_dev_improvement': report['relative_dev_improvement'],
                              'under_one_percent_stop_tuning': report['under_one_percent_stop_tuning'],
                              'checkpoints': report['checkpoints']}, indent=2))
    finally:
        if initialized_dist and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
