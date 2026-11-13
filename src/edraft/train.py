"""M2 streamed, resumable offline fitting on unwarped compact DFlash traces.

The only primary fitting contexts are standard non-thinking greedy and the
inspected model-card warped T=.7 arm. Long conditions and pure-T=1 diagnostics
never enter the primary optimizer. Real fitting requires the M1 audit gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn import functional as F

from .adapter import DenseAdapter, loss_weights, prev_token, union_rest_grouped
from .features import HiddenProjection
from .runtime import ROOT, atomic_json, common_parser, fingerprint, git_commit, parse_config, preflight, record_run, resolved_config
from .sparse import SignedHash
from .sparse_stats import SparseAccessRecorder, pack_keys
from .traces import read_shard


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


def epoch_references(entries, split, seed=None):
    chosen = [x for x in entries if x.split == split]
    if seed is not None:
        random.Random(seed).shuffle(chosen)
    return [(shard, i) for shard in chosen for i in range(shard.blocks)]


def batches(references, batch_blocks, *, start_batch=0, head_metadata=None):
    """Only one compact shard and one block batch are resident at a time."""
    cached_path, payload = None, None
    for batch_index in range(start_batch, (len(references) + batch_blocks - 1) // batch_blocks):
        selected = references[batch_index * batch_blocks:(batch_index + 1) * batch_blocks]
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
def reconstruct_batch(blocks, weight, bias=None, *, vocab_chunk=8192, output_multiplier=1.):
    h = torch.cat([b['h_raw_bf16'] for b in blocks]).to(weight.device, dtype=torch.float32)
    logits = torch.cat([F.linear(h, weight[start:start + vocab_chunk].float(),
                                None if bias is None else bias[start:start + vocab_chunk].float()) * output_multiplier
                        for start in range(0, len(weight), vocab_chunk)], dim=-1)
    return logits.log_softmax(-1)


def corrected_batch_scores(adapter, blocks, l_unit, *, return_sparse_keys=False):
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
    pair = torch.zeros(len(H), adapter.E.shape[1], device=H.device, dtype=torch.float32)
    for d in range(len(adapter.A)):
        valid = previous[:, d] >= 0
        # Skip missing predecessors entirely, including sparse lookup below.
        rows = valid.nonzero(as_tuple=True)[0]
        if len(rows):
            part = adapter.E[previous[rows, d]].float() @ adapter.A[d].T
            pair = pair.index_add(0, rows, part)
    dense = adapter.unary(H) + pair @ adapter.E.float().T
    scores = l_unit - dense
    observed_keys = []
    if adapter.sparse is not None:
        candidates = scores.topk(min(adapter.M_sparse, scores.shape[-1]), dim=-1).indices
        values = torch.zeros_like(candidates, dtype=torch.float32)
        for d in range(len(adapter.A)):
            rows = (previous[:, d] >= 0).nonzero(as_tuple=True)[0]
            if len(rows):
                corrections = adapter.sparse.lookup(d + 1, previous[rows, d, None], candidates[rows])
                values = values.index_add(0, rows, corrections)
                if return_sparse_keys:
                    observed_keys.append(pack_keys(d + 1, previous[rows, d, None], candidates[rows],
                                                   adapter.E.shape[0]).reshape(-1))
        scores = scores - torch.zeros_like(scores).scatter(1, candidates, values)
    if return_sparse_keys:
        return scores, torch.cat(observed_keys) if observed_keys else torch.empty(0, dtype=torch.long, device=scores.device)
    return scores


def batch_targets(blocks, device):
    ids, probabilities, rest, weights = [], [], [], []
    for block in blocks:
        ids.extend(x.to(device) for x in block['union_ids'])
        probabilities.extend(x.to(device) for x in block['target_probs_union'])
        rest.extend(x.to(device) for x in block['target_rest_mass'])
        weights.append(loss_weights(block['block_length'], device))
    return ids, probabilities, rest, torch.cat(weights)


def batch_loss(adapter, blocks, l_unit, *, collect_sparse_keys=False):
    corrected = corrected_batch_scores(adapter, blocks, l_unit, return_sparse_keys=collect_sparse_keys)
    scores = corrected[0] if collect_sparse_keys else corrected
    ids, probabilities, rest, weights = batch_targets(blocks, scores.device)
    result = (union_rest_grouped(scores, ids, probabilities, rest, weights), weights.sum(), scores)
    return (*result, corrected[1]) if collect_sparse_keys else result


@torch.no_grad()
def evaluate(adapter, references, head, bias, args, head_metadata):
    sums = dict(loss=0., weight=0., slots=0, union_overlap_lower=0., grouped_overlap_upper=0.,
                full_overlap=0., full_top1=0., full_slots=0, full_loss=0., full_grouped_loss=0., full_weight=0.)
    for _, blocks in batches(references, args.batch_blocks, head_metadata=head_metadata):
        l_unit = reconstruct_batch(blocks, head, bias, vocab_chunk=args.vocab_chunk,
                                   output_multiplier=head_metadata.get('output_multiplier', 1.))
        loss, weight, scores = batch_loss(adapter, blocks, l_unit)
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
    dense = torch.optim.AdamW(groups, lr=args.learning_rate)
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
    outdir = args.output / ('m2_smoke' if args.scale == 'smoke' else 'm2')
    outdir.mkdir(parents=True, exist_ok=True)
    latest = outdir / 'latest.pt'
    config = resolved_config(args)
    implementation = {name: file_digest(Path(__file__).with_name(name)) for name in
                      ['train.py', 'adapter.py', 'features.py', 'sparse.py', 'sparse_stats.py', 'traces.py']}
    identity_config = {k: v for k, v in config.items() if k != 'resume'}
    identity = fingerprint({'config': identity_config, 'sources': index_report['source_fingerprint'], 'head': head_metadata,
                            'implementation_sha256': implementation,
                            'E_sha256': hashlib.sha256(E.detach().cpu().numpy().tobytes()).hexdigest()})
    stats_recorder = SparseAccessRecorder(args.output / 'm2_sparse_stats', identity, len(E))
    references = {split: epoch_references(entries, split) for split in ['train', 'dev', 'test']}
    first = read_shard(entries[0].path)['blocks'][0]
    p = first['h'].shape[1]
    adapter = DenseAdapter(E, p, args.history_depth, M_sparse=args.M_sparse)
    report = {'scale': args.scale, 'data': 'synthetic' if args.scale == 'smoke' else args.scale,
              'identity': identity, 'trace_index': index_report, 'head_metadata': head_metadata,
              'model_pair': entries[0].identity['model_pair'],
              'model_revisions': entries[0].identity['checkpoint'],
              'implementation_sha256': implementation,
              'frozen_artifacts': {'hidden_projection_sha256': entries[0].identity.get('projection_sha256'),
                                   'token_features_sha256': entries[0].identity.get('token_features_sha256')},
              'optimizer': {'dense': 'AdamW', 'learning_rate': args.learning_rate, 'weight_decay_W_A': args.weight_decay,
                            'weight_decay_b': args.bias_weight_decay, 'sparse': 'SGD+proximal_L1',
                            'sparse_learning_rate': args.sparse_learning_rate, 'lambda_l1': args.lambda_l1},
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
        if resumed['stage'] == 'complete':
            atomic_json(args.output / ('m2_smoke.json' if args.scale == 'smoke' else 'm2.json'), report)
            return report
    if not report.get('baseline'):
        report['baseline'] = {split: evaluate(adapter, refs, head, bias, args, head_metadata) for split, refs in references.items()}
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
        start_epoch, start_batch = 0, 0
        if resumed is not None and resumed['stage'] == stage:
            dense_optimizer.load_state_dict(resumed['dense_optimizer'])
            if sparse_optimizer is not None:
                sparse_optimizer.load_state_dict(resumed['sparse_optimizer'])
            start_epoch, start_batch = resumed['epoch'], resumed['next_batch']
        for epoch in range(start_epoch, epochs):
            train_refs = epoch_references(entries, 'train', seed=args.seed + 1000 * stage_index + epoch)
            for batch_index, blocks in batches(train_refs, args.batch_blocks, start_batch=start_batch,
                                                head_metadata=head_metadata):
                l_unit = reconstruct_batch(blocks, head, bias, vocab_chunk=args.vocab_chunk,
                                           output_multiplier=head_metadata.get('output_multiplier', 1.))
                dense_optimizer.zero_grad(set_to_none=True)
                if sparse_optimizer is not None:
                    sparse_optimizer.zero_grad(set_to_none=True)
                computed = batch_loss(adapter, blocks, l_unit, collect_sparse_keys=stage == 'dense_sparse')
                loss, weight = computed[:2]
                normalized = loss / weight
                if not bool(torch.isfinite(normalized)):
                    raise FloatingPointError('Nonfinite training loss')
                normalized.backward()
                dense_optimizer.step()
                if sparse_optimizer is not None:
                    sparse_optimizer.step()
                    adapter.sparse.proximal_step(args.sparse_learning_rate, args.lambda_l1)
                    chunk = stats_recorder.write_batch(computed[3], epoch, batch_index)
                    report.setdefault('sparse_statistics_chunks', []).append(chunk)
                steps_this_run += 1
                if steps_this_run % args.checkpoint_every == 0 or stop_after_steps == steps_this_run:
                    atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, stage, epoch, batch_index + 1,
                                                                 dense_optimizer, sparse_optimizer, report))
                if stop_after_steps == steps_this_run:
                    return {'status': 'interrupted_for_resume_test', 'checkpoint': str(latest)}
            start_batch = 0
            dev = evaluate(adapter, references['dev'], head, bias, args, head_metadata)
            base = report['baseline']['dev']['weighted_mean_kl']
            gain = (base - dev['weighted_mean_kl']) / max(base, 1e-12)
            report.setdefault('epoch_development', []).append({'stage': stage, 'epoch': epoch, 'dev': dev,
                                                               'relative_improvement_over_zero': gain})
            atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, stage, epoch + 1, 0,
                                                         dense_optimizer, sparse_optimizer, report))
            if gain < .01:
                report.setdefault('stopped_additional_epochs', []).append(stage)
                atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, stage, epochs, 0,
                                                             dense_optimizer, sparse_optimizer, report))
                break
        # Test is read only after each predeclared phase is finished; its metrics
        # never choose parameters, epochs, learning rates, or checkpoints.
        report['stages'][stage] = {split: evaluate(adapter, refs, head, bias, args, head_metadata)
                                    for split, refs in references.items()}
        report['stages'][stage]['parameters'] = parameter_report(adapter)
        if stage == 'dense_sparse':
            report['stages'][stage]['sparse_collision_statistics'] = stats_recorder.finalize(
                report['sparse_statistics_chunks'], adapter.sparse,
                memory_limit=getattr(args, 'stats_memory_limit', '512MB'),
                benchmark_entries=getattr(args, 'stats_benchmark_entries', 1_000_000), seed=args.seed)
        atomic_torch_save(outdir / f'{stage}.pt', checkpoint_payload(adapter, args, identity, stage, epochs, 0,
                                                                    dense_optimizer, sparse_optimizer, report))
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
    atomic_torch_save(latest, checkpoint_payload(adapter, args, identity, 'complete', 0, 0, None, None, report))
    atomic_json(args.output / ('m2_smoke.json' if args.scale == 'smoke' else 'm2.json'), report)
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


def main():
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
    parser.add_argument('--vocab_chunk', type=int, default=8192)
    parser.add_argument('--epochs_dense', type=int, default=1)
    parser.add_argument('--epochs_sparse', type=int, default=1)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--stats_memory_limit', default='512MB')
    parser.add_argument('--stats_benchmark_entries', type=int, default=1_000_000)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_false', dest='resume')
    args = parse_config(parser)
    if args.m1_diagnostic is None:
        args.m1_diagnostic = args.output / f'diagnostic_{args.scale}.json'
    if min(args.batch_blocks, args.vocab_chunk, args.epochs_dense, args.epochs_sparse, args.checkpoint_every) < 1:
        raise ValueError('Training counts must be positive')
    if min(args.learning_rate, args.sparse_learning_rate) <= 0 or min(args.weight_decay, args.bias_weight_decay, args.lambda_l1) < 0:
        raise ValueError('Learning rates must be positive and regularization nonnegative')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.scale == 'smoke':
        torch.set_num_threads(1)
        args.sparse_capacity = min(args.sparse_capacity, 256)
        args.M_sparse = min(args.M_sparse, 4)
        args.trace_dir, E, head, bias, head_metadata = smoke_bundle(args)
        entries, index_report = trace_index(args.trace_dir)
    else:
        preflight(args, 500_000)
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
    print(json.dumps({'scale': args.scale, 'status': report['status'],
                      'relative_dev_improvement': report['relative_dev_improvement'],
                      'under_one_percent_stop_tuning': report['under_one_percent_stop_tuning'],
                      'checkpoints': report['checkpoints']}, indent=2))


if __name__ == '__main__':
    main()
