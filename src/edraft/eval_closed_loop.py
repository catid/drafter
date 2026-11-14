"""M3/M4 dev-only selection, frozen evaluation, and isolated matched-arm timing."""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid
import torch

from .adapter import DenseAdapter
from .diagnose import (prompt_bootstrap, output_bin, context_bin, generation_diagnostics,
                       DEFAULT_CONTEXT_EDGES)
from .features import HiddenProjection
from .online import OnlineBigram, OnlineState
from .runtime import (ROOT, Manifest, atomic_json, common_parser, fingerprint, parse_config,
                      preflight, record_run, resolved_config)
from .sampler import SamplingWarper

M3_ARMS = ('plain_dflash', 'online_bigram', 'dense', 'dense_sparse')
M4_ARMS = ('plain_dflash', 'online_bigram', 'offline_static', 'offline_online_stride1', 'online_dbS')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash():
    paths = [Path(__file__).parent / name for name in ('adapter.py', 'online.py', 'features.py', 'dflash_loop.py',
             'sampler.py', 'verify.py', 'backend.py', 'sparse.py', 'toy.py', 'diagnose.py', 'eval_closed_loop.py',
             'fastpath.py', 'optimizers.py')]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def adapter_tensor_hash(adapter):
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def arms_for(stage, scale='reference'):
    if stage == 'm3':
        return M3_ARMS
    if stage != 'm4':
        raise ValueError('stage must be m3 or m4')
    return M4_ARMS + (('offline_online_stride5', 'zero_offline_online_only', 'online_rejection_relative', 'online_cutoff') if scale == 'full' else ())


def sampling_conditions(condition, primary_only=False):
    if condition == 'long_thinking':
        return [('thinking_recommended', .6, SamplingWarper(top_k=20, top_p=.95))]
    if condition not in ('standard_nonthinking', 'long_nonthinking'):
        raise ValueError('unsupported prompt condition')
    primary = ('nonthinking_recommended', .7, SamplingWarper(top_k=20, top_p=.8))
    return [primary] if primary_only else [('greedy', 0., SamplingWarper()), primary,
                                          ('pure_temperature_diagnostic', 1., SamplingWarper())]


def select_prompts(manifest, split, scale, standard_dev_prompts=8, long_dev_prompts=12):
    all_rows = manifest['prompts']
    ids = [row['prompt_id'] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError('prompt IDs must be disjoint across all splits')
    rows = [row for row in all_rows if row['split'] == split]
    standard = [row for row in rows if row['condition'] == 'standard_nonthinking']
    long = [row for row in rows if row['condition'].startswith('long_') and row.get('reference_selected', True)]
    if len({row['condition'] for row in long}) > 1:
        raise ValueError('this evaluation expects one explicitly authorized long condition; run extra conditions separately')
    if scale == 'smoke':
        return standard[:2] + long[:2]
    if split == 'test':
        standard = [row for row in standard if row.get('reference_selected', True)]
        n_standard, n_long = (128, 64) if scale == 'full' else (64, 32)
    elif split == 'dev':
        n_standard, n_long = standard_dev_prompts, long_dev_prompts
        if n_long < (32 if scale == 'full' else 12):
            raise ValueError('long dev tuning requires >=12 prompts at reference or >=32 at full scale')
    else:
        raise ValueError('evaluation never uses train prompts')
    if len(standard) < n_standard or len(long) < n_long:
        raise ValueError(f'insufficient {split} prompts: need {n_standard} standard and {n_long} long')
    return standard[:n_standard] + long[:n_long]


def evaluation_jobs(prompts, stage, scale, seed=2026):
    jobs = []
    for prompt in prompts:
        for name, temperature, warp in sampling_conditions(prompt['condition']):
            seeds = 1 if temperature == 0 else (3 if prompt['condition'] == 'standard_nonthinking' or scale == 'full' else 1)
            if scale == 'smoke':
                seeds = 1
            for index in range(seeds):
                for arm in arms_for(stage, scale):
                    jobs.append({'prompt': prompt, 'arm': arm, 'sampling_name': name,
                                 'temperature': temperature, 'warper': warp, 'seed': seed + index})
    return jobs


def candidate_settings(arm):
    if arm == 'plain_dflash':
        return [{}]
    if arm == 'online_bigram':
        return [{'lambda_tr': .05, 'alpha': .5, 'k_ret': None}, {'lambda_tr': .2, 'alpha': 1., 'k_ret': None}]
    if arm in ('dense', 'dense_sparse', 'offline_static'):
        return [{'beta': .5}, {'beta': 1.}]
    stride = 5 if arm == 'offline_online_stride5' else 1
    weighting = 'rejection_relative' if arm == 'online_rejection_relative' else 'cutoff' if arm == 'online_cutoff' else 'exponential'
    return [{'beta': 1., 'lr': lr, 'update_stride': stride, 'lambda_l2': l2,
             'lambda_l1_b': 1e-6, 'lambda_l1_s': 1e-5, 'weighting_mode': weighting}
            for lr, l2 in ((1e-4, .01), (1e-3, .1))]


def validate_frozen_settings(settings, stage, scale, checkpoint_hashes, prompt_manifest, generation_caps=None, model_pair=None, model_revisions=None, generation_environment=None):
    if not settings.get('frozen') or settings.get('stage') != stage or settings.get('scale') != scale:
        raise ValueError('test evaluation requires frozen settings for this stage and scale')
    if settings['checkpoint_hashes'] != checkpoint_hashes:
        raise ValueError('checkpoint/projection changed since dev tuning; freeze new settings')
    if settings.get('prompt_manifest_hash') != fingerprint(prompt_manifest):
        raise ValueError('prompt manifest changed since development selection')
    if generation_caps is not None and settings.get('generation_caps') != generation_caps:
        raise ValueError('generation caps changed since development selection')
    if model_pair is not None and settings.get('model_pair') != model_pair:
        raise ValueError('model pair changed since development selection')
    if model_revisions is not None and settings.get('model_revisions') != model_revisions:
        raise ValueError('model revisions changed since development selection')
    if generation_environment is not None and settings.get('generation_environment') != generation_environment:
        raise ValueError('tokenizer/model generation environment changed since development selection')
    dev_ids = set(settings['development_prompt_ids'])
    by_id = {p['prompt_id']: p for p in prompt_manifest['prompts']}
    if not dev_ids or any(i not in by_id or by_id[i]['split'] != 'dev' for i in dev_ids):
        raise ValueError('frozen hyperparameters include a non-development prompt')
    long_ids = [i for i in dev_ids if by_id[i]['condition'].startswith('long_')]
    minimum_long = 1 if scale == 'smoke' else 32 if scale == 'full' else 12
    if len(long_ids) < minimum_long:
        raise ValueError('frozen settings lack the required long development prompts')
    if set(settings['selected']) != set(arms_for(stage, scale)):
        raise ValueError('frozen settings omit a required arm')


class EvaluationRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.tiny = args.scale == 'smoke'
        if self.tiny:
            from .toy import ToyTarget, ToyParallelDrafter
            from .features import fit_hidden_projection, fit_token_features
            from .sparse import SignedHash
            self.target = ToyTarget(seed=args.seed)
            self.drafter = ToyParallelDrafter(self.target, seed=args.seed + 1)
            calibration = [[i % 8, (i + 1) % 8, (i + 3) % 8] for i in range(16)]
            self.projection = fit_hidden_projection([self.drafter.draft(p, 4).h_raw for p in calibration], self.target.hidden_size, p=6, seed=args.seed)
            E, _ = fit_token_features(self.target.embedding, 4, args.seed, torch.tensor(sum(calibration, [])))
            self.dense = DenseAdapter(E, 6, 4)
            self.dense_sparse = DenseAdapter(E, 6, 4, sparse=SignedHash(257), M_sparse=4)
            # Small genuine fit on synthetic training prompts, disjoint from dev/test IDs.
            optimizer = torch.optim.AdamW([{'params': [self.dense.W, self.dense.A], 'weight_decay': .01},
                                           {'params': [self.dense.b], 'weight_decay': 0.}], lr=.01)
            for prefix in calibration[:4]:
                draft = self.drafter.draft(prefix, 4)
                tokens = draft.logits.argmax(-1).tolist()
                target = self.target.verify_logits(prefix, tokens)[:4]
                optimizer.zero_grad()
                loss = self.dense.full_loss(draft.logits.log_softmax(-1), target, self.projection(draft.h_raw), prefix, tokens)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                self.dense_sparse.W.copy_(self.dense.W)
                self.dense_sparse.A.copy_(self.dense.A)
                self.dense_sparse.b.copy_(self.dense.b)
            self.backend = None
            self.checkpoint_hashes = {'dense': adapter_tensor_hash(self.dense), 'dense_sparse': adapter_tensor_hash(self.dense_sparse),
                                      'projection': 'synthetic-calibration-seed-' + str(args.seed), 'implementation': implementation_hash()}
            self.model_pair = ['builtin_toy_target', 'builtin_parallel_drafter']
            self.model_revisions = ['synthetic-v1', 'synthetic-v1']
        else:
            from .train import load_adapter_checkpoint
            from .backend import ReferenceBackend
            from .download import fetch_model
            m2 = json.loads((args.output / 'm2.json').read_text())
            if m2.get('status') != 'passed' or m2.get('scale', args.scale) != args.scale:
                raise RuntimeError('M2 reference acceptance gate must pass before M3/M4')
            if args.stage == 'm4':
                prior = json.loads((args.output / 'm3' / 'summary.json').read_text())
                if prior.get('status') != 'complete' or prior.get('scale') != args.scale:
                    raise RuntimeError('M3 must complete before M4')
            self.checkpoint_hashes = checkpoint_hashes(args)
            self.dense, self.dense_metadata = load_adapter_checkpoint(args.dense_checkpoint, args.device)
            self.dense_sparse, self.sparse_metadata = load_adapter_checkpoint(args.sparse_checkpoint, args.device)
            self.projection = HiddenProjection.load(args.projection).to(args.device)
            if self.dense.W.shape[1] != self.projection.matrix.shape[0]:
                raise ValueError('checkpoint and frozen hidden projection dimensions differ')
            target = fetch_model(args.target, args.allow_download)
            drafter = fetch_model(args.drafter, args.allow_download)
            env = json.loads((ROOT / 'artifacts/environment.json').read_text())
            self.model_revisions = [env['model_artifacts'][name]['revision'] for name in (args.target, args.drafter)]
            for label, metadata in [('dense', self.dense_metadata), ('dense_sparse', self.sparse_metadata)]:
                report = metadata.get('report', {})
                if metadata.get('identity') != m2.get('identity') or metadata.get('stage') != label:
                    raise ValueError('selected checkpoint is not the corresponding stage of the accepted M2 fit')
                if report.get('model_pair') != [args.target, args.drafter] or report.get('model_revisions') != self.model_revisions:
                    raise ValueError('checkpoint training models/revisions do not match evaluation backend')
                if report.get('frozen_artifacts', {}).get('hidden_projection_sha256') != self.checkpoint_hashes['projection']:
                    raise ValueError('checkpoint was fitted with a different hidden projection')
            self.backend = ReferenceBackend.from_pretrained(target, drafter, device=args.device,
                target_revision=self.model_revisions[0], drafter_revision=self.model_revisions[1])
            self.model_pair = [args.target, args.drafter]
        self.dense.eval()
        self.dense_sparse.eval()
        from .fastpath import FastPathConfig, FastPathRuntime
        fp_cfg = FastPathConfig(
            compile=bool(getattr(args, 'compile', False)),
            cuda_graph=bool(getattr(args, 'cuda_graph', False)),
            simulate_cpu_graph=bool(getattr(args, 'simulate_cpu_graph', False)),
            bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
            compile_drafter=bool(getattr(args, 'compile_drafter', False)),
            compile_online=bool(getattr(args, 'compile_online', False)),
        )
        static_k = 4 if self.tiny else (self.backend.block_size - 1)
        self.fastpath = FastPathRuntime(fp_cfg, static_block_length=static_k) if fp_cfg.enabled else None
        if self.tiny:
            self.generation_environment = {'target_state_sha256': adapter_tensor_hash(self.target),
                'tokenizer': {'kind': 'synthetic prompt-ID prefix', 'vocabulary_size': self.target.embedding.shape[0]},
                'stop_token_ids': [], 'context_limit': None, 'torch_version': torch.__version__}
        else:
            tokenizer = self.backend.tokenizer
            self.generation_environment = {
                'tokenizer': {'model': self.model_pair[0], 'revision': self.model_revisions[0],
                    'class': type(tokenizer).__module__ + '.' + type(tokenizer).__qualname__,
                    'vocabulary_sha256': fingerprint(tokenizer.get_vocab()),
                    'backend_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                    'chat_template_sha256': fingerprint(tokenizer.chat_template),
                    'special_tokens_sha256': fingerprint(tokenizer.special_tokens_map)},
                'target_config_sha256': fingerprint(self.backend.target.config.to_dict()),
                'target_generation_config_sha256': fingerprint(self.backend.target.generation_config.to_dict()),
                'stop_token_ids': self.backend.stop_token_ids,
                'context_limit': self.backend.target.config.max_position_embeddings,
                'dtype': str(self.backend.target.dtype), 'torch_version': torch.__version__}

    def prepare_inputs(self, prompt, max_new_tokens_override=None):
        cap = self.args.smoke_tokens if self.tiny else self.args.long_max_new_tokens if prompt['condition'].startswith('long_') else self.args.max_new_tokens
        if max_new_tokens_override is not None:
            cap = min(cap, max_new_tokens_override)
        if self.tiny:
            inputs = [1, 2, 3, int(hashlib.sha256(prompt['prompt_id'].encode()).hexdigest()[:4], 16) % 8]
            input_tokens, effective_cap = inputs, cap
        else:
            inputs = self.backend.tokenize(prompt['text'], enable_thinking=prompt['condition'] == 'long_thinking')
            input_tokens = inputs[0].tolist()
            effective_cap = min(cap, self.backend.target.config.max_position_embeddings - len(input_tokens))
        if effective_cap < 1:
            raise ValueError('prompt exhausts model context')
        metadata = {'input_token_ids_sha256': fingerprint(input_tokens), 'input_token_count': len(input_tokens),
                    'enable_thinking': prompt['condition'] == 'long_thinking',
                    'requested_generation_cap': cap, 'effective_generation_cap': effective_cap,
                    'generation_environment_sha256': fingerprint(self.generation_environment)}
        return inputs, effective_cap, metadata

    def run_target_only(self, prompt, prepared):
        inputs, cap, metadata = prepared
        if self.tiny:
            output = self.target.generate(inputs, cap)
        else:
            result = self.backend.target_only_generate(inputs, cap)
            output = result[0, inputs.shape[1]:].tolist()
        eos = bool(output and output[-1] in self.generation_environment['stop_token_ids'])
        return {'scale': self.args.scale, 'label': 'synthetic' if self.tiny else self.args.scale,
                'origin_stage': self.args.stage, 'prompt_id': prompt['prompt_id'], 'input_metadata': metadata,
                'output_tokens': output, 'generated_tokens': len(output),
                'generation': {'complete': True, 'eos_reached': eos, 'capped': len(output) == cap},
                'timing_scope': 'output exactness diagnostic only; no throughput or latency claim'}

    def run(self, job, setting, profile=False):
        from .dflash_loop import generate_reference, generate_toy
        args, arm, prompt = self.args, job['arm'], job['prompt']
        temperature, warper = job['temperature'], job['warper']
        # Greedy beta=1 is a fixed required arm, independent of stochastic dev beta.
        beta = 1. if temperature == 0 else setting.get('beta', 1.)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        request_started = time.perf_counter()
        adapter, online, bigram = None, None, None
        if arm == 'online_bigram':
            bigram = OnlineBigram(self.dense.E.shape[0], **setting)
        elif arm == 'dense':
            adapter = self.dense
        elif arm in ('dense_sparse', 'offline_static'):
            adapter = self.dense_sparse
        elif arm != 'plain_dflash':
            offline = self.dense_sparse
            if arm == 'zero_offline_online_only':
                offline = DenseAdapter(self.dense.E, self.dense.W.shape[1], len(self.dense.A), M_sparse=self.dense.M_sparse)
            kwargs = {k: v for k, v in setting.items() if k != 'beta'}
            online = OnlineState(offline, enable_sparse=True, sparse_capacity=257 if self.tiny else 2**20,
                                 bias_sparse_only=arm == 'online_dbS', fastpath=self.fastpath, **kwargs)
        inputs, max_tokens, input_metadata = self.prepare_inputs(prompt, job.get('max_new_tokens_override'))
        generator = torch.Generator(device=self.device).manual_seed(job['seed'])
        if self.tiny:
            prefix = inputs
            result = generate_toy(self.target, self.drafter, prefix, max_tokens, temperature=temperature, warper=warper,
                generator=generator, adapter=adapter, online_state=online, projection=self.projection if adapter is not None or online is not None else None,
                bigram=bigram, beta=beta, measure_phases=profile, fastpath=self.fastpath)
            eos = False
        else:
            result = generate_reference(self.backend, inputs, max_tokens, temperature=temperature, warper=warper,
                generator=generator, adapter=adapter, online_state=online, projection=self.projection if adapter is not None or online is not None else None,
                bigram=bigram, beta=beta, measure_phases=profile, fastpath=self.fastpath)
            eos = bool(result.output_tokens and result.output_tokens[-1] in self.backend.stop_token_ids)
        request_elapsed = time.perf_counter() - request_started
        if self.tiny and temperature == 0:
            expected = self.target.generate(prefix, result.num_output_tokens)
            if result.output_tokens != expected:
                raise RuntimeError('synthetic greedy exactness failed')
        online_invariants = None
        if online is not None:
            for block in result.blocks:
                expected_before = (block['round_idx'] - 1) // online.update_stride
                expected_after = block['round_idx'] // online.update_stride
                if block['adapter_version'] != expected_before or block['next_adapter_version'] != expected_after:
                    raise RuntimeError('online version changed outside a stride boundary')
            online_invariants = {'request_reset': True, 'version_only_between_rounds': True, 'stride': online.update_stride,
                                 'executed_updates': online.u, 'single_round_no_buffering': True}
        lengths = [block['accepted_length'] for block in result.blocks]
        generation = generation_diagnostics(result.output_tokens)
        generation.update({'eos_reached': eos, 'capped': result.num_output_tokens == max_tokens, 'complete': True})
        online_metrics = online.metrics() if online else None
        update_ms = result.phase_ms.get('online_update', 0.)
        state_bandwidth = online_metrics['state_bytes_estimate_total'] / (update_ms * 1e6) if online_metrics is not None and update_ms > 0 else None
        return {'scale': args.scale, 'label': 'synthetic' if self.tiny else args.scale,
            'stage': args.stage, 'arm': arm, 'prompt_id': prompt['prompt_id'], 'split': prompt['split'],
            'condition': prompt['condition'], 'seed': job['seed'], 'sampling_name': job['sampling_name'],
            'temperature': temperature, 'sampling_warp_config': asdict(warper), 'settings': setting, 'effective_beta': beta,
            'verification_rule': 'ratio' if temperature > 0 else 'greedy',
            'model_pair': self.model_pair, 'model_revisions': self.model_revisions,
            'input_metadata': input_metadata,
            'checkpoint_hashes': self.checkpoint_hashes,
            'offline_fit_provenance': 'four built-in synthetic calibration-train prefixes' if self.tiny else 'gated M2 checkpoints', 'generated_tokens': result.num_output_tokens,
            'elapsed_seconds': request_elapsed, 'generation_loop_elapsed_seconds': result.elapsed_seconds,
            'tokens_per_second': result.num_output_tokens / request_elapsed,
            'latency_scope': 'request adapter initialization + tokenizer + generation; excludes reporting/serialization',
            'mean_accepted_length': statistics.mean(lengths) if lengths else 0.,
            'median_accepted_length': statistics.median(lengths) if lengths else 0.,
            'target_forwards_per_generated_token': (len(result.blocks) + (0 if self.tiny else 1)) / max(1, result.num_output_tokens),
            'phase_ms': result.phase_ms, 'blocks': result.blocks, 'generation': generation,
            'output_tokens': result.output_tokens, 'online': online_metrics,
            'estimated_online_state_bandwidth_GBps': state_bandwidth,
            'online_bandwidth_definition': 'algorithmic state bytes divided by measured full online_update phase duration, including live-full backprop and snapshots; not hardware HBM bandwidth',
            'bigram': bigram.metrics() if bigram else None, 'online_invariants': online_invariants,
            'parameter_norms': {'W': float((adapter or self.dense_sparse).W.detach().norm()),
                                'A': float((adapter or self.dense_sparse).A.detach().norm()),
                                'b': float((adapter or self.dense_sparse).b.detach().norm())} if adapter is not None or online is not None else {},
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None,
            'phase_profile': profile}


def checkpoint_hashes(args):
    return {'dense': file_sha256(args.dense_checkpoint), 'dense_sparse': file_sha256(args.sparse_checkpoint),
            'projection': file_sha256(args.projection), 'implementation': implementation_hash()}


def run_identity(runner, job, setting, mode, frozen_hash='', session=None, repeat=None, profile=False):
    cap = runner.args.smoke_tokens if runner.tiny else runner.args.long_max_new_tokens if job['prompt']['condition'].startswith('long_') else runner.args.max_new_tokens
    return {'model_pair': runner.model_pair, 'prompt': job['prompt']['prompt_id'], 'scale': runner.args.scale,
            'prompt_payload_hash': fingerprint(job['prompt']),
            'condition': job['prompt']['condition'], 'seed': job['seed'], 'temperature': job['temperature'],
            'sampling_name': job['sampling_name'],
            'verification_rule': 'ratio' if job['temperature'] > 0 else 'greedy',
            'proposal_mode': job['arm'], 'checkpoint': runner.checkpoint_hashes, 'adapter_version': 0,
            'model_revisions': runner.model_revisions, 'stage': runner.args.stage, 'mode': mode,
            'sampling_warp_config': asdict(job['warper']), 'setting': setting, 'frozen_settings_hash': frozen_hash,
            'generation_cap': min(cap, job.get('max_new_tokens_override', cap)),
            'generation_caps': generation_caps(runner.args),
            'timing_session': session, 'repeat': repeat, 'phase_profile': profile}


BASELINE_REUSE_ARMS = ('plain_dflash', 'online_bigram')
BASELINE_REUSE_EXCLUDED_IDENTITY_FIELDS = ('stage', 'frozen_settings_hash')
REUSED_UNAVAILABLE_FIELDS = ('elapsed_seconds', 'generation_loop_elapsed_seconds', 'tokens_per_second',
                            'phase_ms', 'estimated_online_state_bandwidth_GBps',
                            'peak_allocated_bytes', 'peak_reserved_bytes')


def baseline_generation_identity(identity):
    """Only stage and the all-arm selection record may differ between M3/M4."""
    return {key: value for key, value in identity.items() if key not in BASELINE_REUSE_EXCLUDED_IDENTITY_FIELDS}


def m3_baseline_sources(runner):
    """Index metadata once; read only the exact matching request's output file."""
    if hasattr(runner, '_m3_baseline_sources'):
        return runner._m3_baseline_sources
    directory = runner.args.output / 'm3'
    index = {'candidates': defaultdict(list), 'settings': None, 'reason': None}
    try:
        settings_path = directory / 'settings.json'
        settings_bytes = settings_path.read_bytes()
        settings = json.loads(settings_bytes)
        if not settings.get('frozen') or settings.get('stage') != 'm3' or settings.get('scale') != runner.args.scale:
            raise ValueError('M3 source settings are not frozen at the matching scale')
        index.update(settings=settings, settings_hash=fingerprint(settings), settings_path=str(settings_path.resolve()),
                     settings_sha256=hashlib.sha256(settings_bytes).hexdigest())
        for manifest_path in sorted((directory / 'evaluate').glob('manifest-rank*.json')):
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            for key, entry in manifest['jobs'].items():
                identity = entry.get('identity', {})
                if (entry.get('status') == 'complete' and identity.get('stage') == 'm3'
                        and identity.get('mode') == 'evaluate' and identity.get('proposal_mode') in BASELINE_REUSE_ARMS
                        and key == fingerprint(identity)):
                    index['candidates'][(identity.get('prompt'), identity['proposal_mode'])].append(
                        dict(entry, source_manifest_path=str(manifest_path.resolve()),
                             source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()))
    except (OSError, ValueError, KeyError, TypeError) as error:
        index['reason'] = f'M3 source evidence unavailable: {error}'
    runner._m3_baseline_sources = index
    return index


def reuse_m3_baseline(runner, identity):
    """Return copied acceptance/output evidence, never tune or timing work."""
    if (identity['stage'] != 'm4' or identity['mode'] != 'evaluate'
            or identity['proposal_mode'] not in BASELINE_REUSE_ARMS or identity['phase_profile']
            or identity['timing_session'] is not None or identity['repeat'] is not None):
        return None, None
    sources = m3_baseline_sources(runner)
    decision = {'status': 'fresh', 'source_stage': 'm3',
                'scope': 'acceptance and output metrics only'}
    if sources['reason']:
        return None, dict(decision, reason=sources['reason'])
    candidates = sources['candidates'].get((identity['prompt'], identity['proposal_mode']), [])
    if not candidates:
        return None, dict(decision, reason='no completed M3 baseline request for this prompt and arm')
    expected = baseline_generation_identity(identity)
    differences = []
    for entry in candidates:
        source = baseline_generation_identity(entry['identity'])
        differences.append((entry, sorted(key for key in expected.keys() | source.keys()
                                         if key not in expected or key not in source or expected[key] != source[key])))
    matches = [entry for entry, fields in differences if not fields]
    if not matches:
        fields = min((fields for _, fields in differences), key=lambda fields: (len(fields), fields))
        return None, dict(decision, reason='M3 generation identity differs; use the frozen M4 settings', mismatched_fields=fields)
    # A stable choice avoids silently preferring one rank's differing evidence.
    if len({entry.get('result_sha256') for entry in matches}) != 1:
        return None, dict(decision, reason='conflicting completed M3 baseline sources')
    entry = matches[0]
    source_identity = entry['identity']
    try:
        settings = sources['settings']
        # The all-arm frozen hashes differ across stages, so independently bind
        # the full tokenizer/model environment and its per-request input digest.
        if (source_identity['frozen_settings_hash'] != sources['settings_hash']
                or settings['selected'][identity['proposal_mode']] != identity['setting']
                or settings['checkpoint_hashes'] != identity['checkpoint']
                or settings['model_pair'] != identity['model_pair']
                or settings['model_revisions'] != identity['model_revisions']
                or settings.get('generation_environment') != getattr(runner, 'generation_environment', None)
                or settings['generation_caps'] != identity['generation_caps']):
            raise ValueError('M3 request does not match its frozen source settings')
        path = Path(entry['result_path'])
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if not entry.get('result_sha256') or digest != entry['result_sha256']:
            raise ValueError('M3 source result checksum missing or changed')
        row = json.loads(payload)
        if row['identity'] != source_identity or row.get('phase_profile') or not row['generation']['complete']:
            raise ValueError('M3 source result is not a complete unprofiled evaluation request')
        row_fields = {'stage': 'stage', 'scale': 'scale', 'arm': 'proposal_mode', 'prompt_id': 'prompt',
                      'condition': 'condition', 'seed': 'seed', 'temperature': 'temperature',
                      'sampling_name': 'sampling_name', 'sampling_warp_config': 'sampling_warp_config',
                      'settings': 'setting', 'verification_rule': 'verification_rule',
                      'model_pair': 'model_pair', 'model_revisions': 'model_revisions', 'checkpoint_hashes': 'checkpoint'}
        if any(row[field] != source_identity[key] for field, key in row_fields.items()):
            raise ValueError('M3 source row metadata differs from its generation identity')
        if row['generated_tokens'] != len(row['output_tokens']):
            raise ValueError('M3 source output length differs from its token count')
        if row.get('input_metadata', {}).get('generation_environment_sha256') != fingerprint(getattr(runner, 'generation_environment', None)):
            raise ValueError('M3 source tokenizer/model generation environment differs')
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, dict(decision, reason=f'invalid M3 source evidence: {error}')
    provenance = dict(decision, status='reused', reason='exact generation identity and verified source checksums',
                      source_result_path=str(path.resolve()), source_result_sha256=digest,
                      source_manifest_path=entry['source_manifest_path'], source_manifest_sha256=entry['source_manifest_sha256'],
                      source_settings_path=sources['settings_path'], source_settings_sha256=sources['settings_sha256'],
                      source_identity_hash=fingerprint(source_identity), source_frozen_settings_hash=sources['settings_hash'],
                      excluded_identity_fields=list(BASELINE_REUSE_EXCLUDED_IDENTITY_FIELDS),
                      unavailable_measurements=list(REUSED_UNAVAILABLE_FIELDS))
    row = copy.deepcopy(row)
    row.update({field: None for field in REUSED_UNAVAILABLE_FIELDS})
    row.update(stage='m4', latency_scope='unavailable: reused M3 acceptance/output evidence; M4 isolated timing is measured separately')
    return row, provenance


def execute_job(runner, job, setting, mode, frozen_hash='', session=None, repeat=None, profile=False):
    args = runner.args
    directory = args.output / args.stage / mode
    manifest = Manifest(directory / f'manifest-rank{args.rank}.json')
    identity = run_identity(runner, job, setting, mode, frozen_hash, session, repeat, profile)
    path = directory / f'rank{args.rank}' / f'{manifest.key(identity)}.json'
    if manifest.completed(identity):
        entry = manifest.data['jobs'][manifest.key(identity)]
        if entry.get('result_sha256') and entry['result_sha256'] != file_sha256(path):
            raise ValueError('completed evaluation result checksum changed')
        return json.loads(path.read_text())
    row, reuse = reuse_m3_baseline(runner, identity)
    if row is None:
        row = runner.run(job, setting, profile)
    if reuse is not None:
        row['baseline_reuse'] = reuse
    row.update({'identity': identity, 'timing_session': session, 'repeat': repeat,
                'timing_isolated': mode == 'timing', 'rank': args.rank})
    atomic_json(path, row)
    manifest.data['jobs'][manifest.key(identity)] = {'identity': identity, 'status': 'complete',
                                                   'result_path': str(path), 'result_sha256': file_sha256(path)}
    atomic_json(manifest.path, manifest.data)
    return row


def generation_caps(args):
    return {'standard': args.max_new_tokens, 'long': args.long_max_new_tokens, 'smoke': args.smoke_tokens}


def comparison_session(runner, mode, configuration):
    args = runner.args
    if getattr(runner, 'fastpath', None) is not None:
        configuration = dict(configuration, fastpath=runner.fastpath.config.to_dict())
    config = {'configuration': configuration, 'checkpoint_hashes': runner.checkpoint_hashes,
              'model_pair': runner.model_pair, 'model_revisions': runner.model_revisions,
              'generation_environment': getattr(runner, 'generation_environment', None),
              'generation_caps': generation_caps(args), 'stage': args.stage, 'scale': args.scale,
              'device': str(runner.device), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'seed': args.seed}
    path = args.output / args.stage / mode / 'sessions' / f'{fingerprint(config)}.json'
    if path.exists() and not getattr(args, 'fresh_session', False):
        state = json.loads(path.read_text())
    else:
        state = {'logical_session': uuid.uuid4().hex, 'configuration': config, 'created_at_unix': time.time(), 'epochs': [], 'bundles': {}}
    epoch = uuid.uuid4().hex
    state['epochs'].append({'epoch': epoch, 'started_at_unix': time.time(), 'device': str(runner.device)})
    atomic_json(path, state)
    return path, state, epoch


def comparison_bundle_key(jobs_and_settings, repeat, profile):
    return fingerprint({'jobs': [(job['prompt'], job['arm'], job['sampling_name'], job['seed'], setting)
                                 for job, setting in jobs_and_settings], 'repeat': repeat, 'profile': profile})


def read_comparison_bundle(runner, mode, jobs_and_settings, frozen_hash, session_state, repeat, profile=False, manifest=None):
    """Return only a complete, checksummed, exactly matched all-arm bundle.

    Missing files are an interrupted bundle: callers may rerun every arm together.
    Contradictory identities or checksums fail explicitly rather than accepting or
    partially repairing measurements whose provenance is uncertain.
    """
    bundle_key = comparison_bundle_key(jobs_and_settings, repeat, profile)
    prior = session_state['bundles'].get(bundle_key)
    if prior is None or prior.get('status') != 'complete':
        return None
    paths = prior.get('result_paths', [])
    if len(paths) != len(jobs_and_settings) or len(set(paths)) != len(paths):
        raise ValueError('completed comparison bundle has incorrect result coverage')
    epoch = prior.get('epoch')
    if not epoch or epoch not in {item['epoch'] for item in session_state['epochs']}:
        raise ValueError('completed comparison bundle has an unknown epoch')
    if not all(Path(path).is_file() for path in paths):
        return None
    directory = runner.args.output / runner.args.stage / mode
    if manifest is None:
        manifest = Manifest(directory / 'manifest-rank0.json').data
    rows = []
    for (job, setting), path in zip(jobs_and_settings, paths):
        identity = run_identity(runner, job, setting, mode, frozen_hash,
            session=session_state['logical_session'], repeat=f'{repeat}:epoch={epoch}', profile=profile)
        expected_path = directory / 'rank0' / f'{fingerprint(identity)}.json'
        if Path(path).resolve() != expected_path.resolve():
            raise ValueError('completed comparison bundle result path does not match expected identity')
        entry = manifest['jobs'].get(fingerprint(identity))
        if entry is None or entry.get('status') != 'complete':
            return None
        if entry.get('identity') != identity or Path(entry.get('result_path', '')).resolve() != expected_path.resolve():
            raise ValueError('completed comparison manifest identity/path mismatch')
        payload = Path(path).read_bytes()
        if not entry.get('result_sha256') or hashlib.sha256(payload).hexdigest() != entry['result_sha256']:
            raise ValueError('completed comparison result checksum missing or changed')
        row = json.loads(payload)
        if row.get('identity') != identity:
            raise ValueError('completed comparison result identity mismatch')
        if (row.get('generation', {}).get('complete') is not True
                or row.get('phase_profile') is not profile or row.get('timing_isolated') is not (mode == 'timing')
                or row.get('timing_session') != identity['timing_session'] or row.get('repeat') != identity['repeat']):
            raise ValueError('completed comparison result is incomplete or has mismatched timing/profile metadata')
        for key, expected in [('arm', job['arm']), ('prompt_id', job['prompt']['prompt_id']),
                              ('condition', job['prompt']['condition']), ('seed', job['seed']),
                              ('temperature', job['temperature']), ('sampling_name', job['sampling_name']),
                              ('sampling_warp_config', asdict(job['warper'])), ('settings', setting)]:
            if row.get(key) != expected:
                raise ValueError(f'completed comparison result {key} mismatch')
        rows.append(row)
    return rows


def comparison_bundle(runner, mode, jobs_and_settings, frozen_hash, session_path, session_state, epoch, repeat, profile=False):
    bundle_key = comparison_bundle_key(jobs_and_settings, repeat, profile)
    prior = read_comparison_bundle(runner, mode, jobs_and_settings, frozen_hash, session_state, repeat, profile)
    if prior is not None:
        return prior
    # Never reuse only some arms from an interrupted comparison. Every arm is
    # remeasured back-to-back in this invocation's epoch before publishing it.
    rows, paths = [], []
    repeat_identity = f'{repeat}:epoch={epoch}'
    from .bench_adapter import assert_no_external_gpu_processes
    for job, setting in jobs_and_settings:
        assert_no_external_gpu_processes(runner.device)
        row = execute_job(runner, job, setting, mode, frozen_hash, session=session_state['logical_session'],
                          repeat=repeat_identity, profile=profile)
        rows.append(row)
        paths.append(str(runner.args.output / runner.args.stage / mode / f'rank{runner.args.rank}' / f"{fingerprint(row['identity'])}.json"))
    assert_no_external_gpu_processes(runner.device)
    session_state['bundles'][bundle_key] = {'status': 'complete', 'epoch': epoch, 'result_paths': paths}
    atomic_json(session_path, session_state)
    return rows


def warm_comparisons(runner, prompts, selected_settings):
    for condition in sorted({p['condition'] for p in prompts}):
        prompt = next(p for p in prompts if p['condition'] == condition)
        name, temperature, warp = sampling_conditions(condition, primary_only=True)[0]
        for arm, settings in selected_settings.items():
            for setting in settings:
                job = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature, 'warper': warp,
                       'seed': runner.args.seed, 'arm': arm, 'max_new_tokens_override': 64}
                runner.run(job, setting, profile=False)


def tune(runner, manifest):
    args = runner.args
    if args.rank != 0 or args.world_size != 1:
        raise ValueError('dev selection is a single-process matched timing session')
    prompts = select_prompts(manifest, 'dev', args.scale, args.standard_dev_prompts, args.long_dev_prompts)
    session_path, session_state, epoch = comparison_session(runner, 'tune', {'prompt_manifest_hash': fingerprint(manifest),
        'dev_ids': [p['prompt_id'] for p in prompts], 'candidates': {a: candidate_settings(a) for a in arms_for(args.stage, args.scale)}})
    warm_comparisons(runner, prompts, {a: candidate_settings(a) for a in arms_for(args.stage, args.scale)})
    observations = defaultdict(lambda: defaultdict(list))
    dev_rows = []
    for prompt_index, prompt in enumerate(prompts):
        name, temperature, warp = sampling_conditions(prompt['condition'], primary_only=True)[0]
        base = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature, 'warper': warp, 'seed': args.seed}
        jobs = [(arm, setting) for arm in arms_for(args.stage, args.scale) for setting in candidate_settings(arm)]
        # Rotate arm order across prompts to reduce systematic thermal/order bias.
        offset = prompt_index % len(jobs)
        comparisons = [(dict(base, arm=arm), setting) for arm, setting in jobs[offset:] + jobs[:offset]]
        rows = comparison_bundle(runner, 'tune', comparisons, '', session_path, session_state, epoch, 'dev')
        for row in rows:
            dev_rows.append(row)
            observations[row['arm']][fingerprint(row['settings'])].append(row)
    plain_by_prompt = {row['prompt_id']: row for row in observations['plain_dflash'][fingerprint({})]}
    selected, candidate_reports = {}, {}
    for arm in arms_for(args.stage, args.scale):
        reports = []
        for setting in candidate_settings(arm):
            rows = observations[arm][fingerprint(setting)]
            # Each condition receives equal weight; long completions do not dilute standard tuning.
            by_condition = defaultdict(list)
            excluded = 0
            for row in rows:
                plain = plain_by_prompt[row['prompt_id']]
                if row['condition'].startswith('long_') and (row['generation']['repetition_dominated'] or plain['generation']['repetition_dominated']):
                    excluded += 1
                    continue
                by_condition[row['condition']].append(row['tokens_per_second'] / plain['tokens_per_second'])
            score = statistics.mean(statistics.mean(values) for values in by_condition.values())
            reports.append({'settings': setting, 'selection_score': score, 'condition_normalized_throughput': {k: statistics.mean(v) for k, v in by_condition.items()},
                            'dev_requests': len(rows), 'repetition_excluded_matched_pairs': excluded,
                            'long_condition_data_sparsity': not any(k.startswith('long_') and v for k, v in by_condition.items()), 'generated_tokens': sum(row['generated_tokens'] for row in rows),
                            'wall_seconds': sum(row['elapsed_seconds'] for row in rows)})
        chosen = max(reports, key=lambda row: row['selection_score'])
        selected[arm] = chosen['settings']
        candidate_reports[arm] = reports
    frozen = {'schema_version': 1, 'frozen': True, 'scale': args.scale, 'stage': args.stage,
              'development_prompt_ids': [p['prompt_id'] for p in prompts],
              'prompt_manifest_hash': fingerprint(manifest), 'generation_caps': generation_caps(args),
              'comparison_session': session_state['logical_session'], 'warmup_policy': 'all candidates on each condition before measured dev requests',
              'checkpoint_hashes': runner.checkpoint_hashes,
              'model_pair': runner.model_pair, 'model_revisions': runner.model_revisions, 'selected': selected,
              'generation_environment': runner.generation_environment,
              'candidates': candidate_reports, 'selection_metric': 'mean across conditions of prompt throughput relative to same-prompt plain DFlash',
              'budget': 'two configurations per tuned arm, identical dev prompts, seeds and token caps; plain DFlash has no tuned parameters',
              'settings_seed': args.seed, 'created_at_unix': session_state['created_at_unix']}
    atomic_json(args.output / args.stage / 'settings.json', frozen)
    return frozen


def prompt_quantile_ci(rows, value_key, quantile, seed=2026, iterations=2000):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[(row['prompt_id'], row['seed'])].append(float(row[value_key]))
    by_prompt = defaultdict(list)
    for (prompt, _), values in by_seed.items():
        by_prompt[prompt].append(statistics.mean(values))
    if not by_prompt:
        return {'estimate': None, 'ci95': None, 'n_prompts': 0}
    values = torch.tensor([statistics.mean(v) for _, v in sorted(by_prompt.items())], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(values), (iterations, len(values)), generator=generator)
    estimates = torch.quantile(values[indices], quantile, dim=-1)
    return {'estimate': float(torch.quantile(values, quantile)),
            'ci95': torch.quantile(estimates, torch.tensor([.025, .975], dtype=torch.float64)).tolist(),
            'n_prompts': len(values), 'unit': 'prompt_id', 'seed_aggregation': 'mean before prompt resampling'}


def summarize_rows(rows, isolated_timing=False):
    groups = defaultdict(list)
    for row in rows:
        key = (row['condition'], row['temperature'], json.dumps(row['sampling_warp_config'], sort_keys=True), row['arm'])
        groups[key].append(row)
    reports = []
    for (condition, temperature, warp, arm), group in sorted(groups.items()):
        primary = [row for row in group if not (condition.startswith('long_') and row['generation']['repetition_dominated'])]
        timed_primary = [row for row in primary if row.get('elapsed_seconds') is not None]
        metrics = {metric: prompt_bootstrap(primary, metric) for metric in ('mean_accepted_length', 'median_accepted_length', 'target_forwards_per_generated_token')}
        if any(row.get('estimated_online_state_bandwidth_GBps') is not None for row in primary):
            metrics['estimated_online_state_bandwidth_GBps'] = prompt_bootstrap(primary, 'estimated_online_state_bandwidth_GBps')
        if isolated_timing:
            metrics['tokens_per_second'] = prompt_bootstrap([row for row in primary if row.get('tokens_per_second') is not None], 'tokens_per_second')
            metrics['elapsed_seconds'] = prompt_bootstrap(timed_primary, 'elapsed_seconds')
        bins = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        accepted_in_bins = defaultdict(list)
        by_position = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for row in primary:
            for block in row['blocks']:
                accepted = block['accepted_length']
                for axis, label in [('output_offset', output_bin(block['absolute_output_offset'])),
                                    ('input_context_length', context_bin(block['input_context_length']))]:
                    accepted_in_bins[(axis, label)].append({'prompt_id': row['prompt_id'], 'seed': row['seed'], 'accepted_length': accepted})
                for i in range(len(block['draft_tokens'])):
                    for axis, label in [('output_offset', output_bin(block['absolute_output_offset'] + i)),
                                        ('input_context_length', context_bin(block['input_context_length'] + i))]:
                        counts = bins[(axis, label)][(row['prompt_id'], row['seed'])]
                        counts[0] += int(i < accepted)
                        counts[1] += 1
                    counts = by_position[i][(row['prompt_id'], row['seed'])]
                    counts[0] += int(i < accepted)
                    counts[1] += 1
        def rates(data):
            return prompt_bootstrap([{'prompt_id': p, 'seed': seed, 'rate': yes / total} for (p, seed), (yes, total) in data.items()], 'rate')
        pooled_lengths = [block['accepted_length'] for row in primary for block in row['blocks']]
        latency = torch.tensor([row['elapsed_seconds'] for row in timed_primary], dtype=torch.float64)
        phases = defaultdict(list)
        for row in primary:
            for phase, value in (row.get('phase_ms') or {}).items():
                phases[phase].append(value)
        reports.append({'condition': condition, 'temperature': temperature, 'sampling_warp_config': json.loads(warp), 'arm': arm,
            'requests': len(group), 'primary_requests': len(primary), 'repetition_excluded': len(group) - len(primary),
            'reused_acceptance_output_requests': sum(row.get('baseline_reuse', {}).get('status') == 'reused' for row in group),
            'requests_with_latency': len(timed_primary), 'requests_without_latency': len(primary) - len(timed_primary),
            'metrics': metrics,
            'pooled_block_accepted_length_mean': statistics.mean(pooled_lengths) if pooled_lengths else None,
            'pooled_block_accepted_length_median': statistics.median(pooled_lengths) if pooled_lengths else None,
            'metric_aggregation': 'metrics are prompt-weighted after seed aggregation; median_accepted_length aggregates request medians; pooled block statistics are separate',
            'acceptance_by_block_position': {str(i): rates(data) for i, data in sorted(by_position.items())},
            'acceptance_bins': [{'axis': axis, 'bin': label, **rates(data), 'accepted_length': prompt_bootstrap(accepted_in_bins[(axis, label)], 'accepted_length')} for (axis, label), data in sorted(bins.items())],
            'request_latency_p50_seconds': float(torch.quantile(latency, .5)) if len(latency) else None,
            'request_latency_p95_seconds': float(torch.quantile(latency, .95)) if len(latency) else None,
            'request_latency_p50_prompt_bootstrap': prompt_quantile_ci(timed_primary, 'elapsed_seconds', .5) if isolated_timing else None,
            'request_latency_p95_prompt_bootstrap': prompt_quantile_ci(timed_primary, 'elapsed_seconds', .95) if isolated_timing else None,
            'phase_ms_median': {phase: statistics.median(values) for phase, values in phases.items()},
            'realized_output_length_mean': statistics.mean(row['generated_tokens'] for row in primary) if primary else None,
            'realized_generations_above_8K': sum(row['generated_tokens'] > 8192 for row in primary),
            'realized_generations_above_16K': sum(row['generated_tokens'] > 16384 for row in primary),
            'peak_allocated_bytes': max((row['peak_allocated_bytes'] for row in group if row.get('peak_allocated_bytes') is not None), default=None),
            'peak_reserved_bytes': max((row['peak_reserved_bytes'] for row in group if row.get('peak_reserved_bytes') is not None), default=None),
            'online_final_state': [row['online'] for row in primary if row['online'] is not None]})
    return reports


def baseline_reuse_report(rows):
    decisions = [row['baseline_reuse'] for row in rows if row.get('baseline_reuse')]
    reasons = defaultdict(int)
    for decision in decisions:
        if decision['status'] != 'reused':
            reason = decision['reason']
            if decision.get('mismatched_fields'):
                reason += ': ' + ', '.join(decision['mismatched_fields'])
            reasons[reason] += 1
    return {'policy': 'M4 baseline acceptance/output only; exact M3 generation identities and source checksums; tune/timing/profile always execute',
            'reused_requests': sum(decision['status'] == 'reused' for decision in decisions),
            'fresh_baseline_requests': sum(decision['status'] != 'reused' for decision in decisions),
            'fresh_reasons': dict(reasons)}


def target_only_identity(evaluation_identity, input_metadata):
    """Conservatively retain all generation provenance, including unused draft/fit hashes.

    Only stage, frozen arm selection, and the adapter proposal/setting differ.
    This diagnostic has no timing session and is shared by M3 and M4.
    """
    identity = baseline_generation_identity(evaluation_identity)
    identity.update(mode='greedy_target_only', proposal_mode='target_only', setting={},
                    input_metadata=input_metadata, timing_session=None, repeat=None, phase_profile=False)
    return identity


def target_only_paths(output, identity):
    directory = Path(output) / 'greedy_target_only'
    key = fingerprint(identity)
    return directory / f'{key}.json', directory / f'{key}.manifest.json'


def read_target_only(output, identity):
    path, manifest_path = target_only_paths(output, identity)
    try:
        entry = Manifest(manifest_path).data['jobs'][fingerprint(identity)]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if (entry['status'] != 'complete' or entry['identity'] != identity
                or entry.get('result_sha256') != digest):
            raise ValueError('target-only diagnostic identity/checksum changed')
        row = json.loads(payload)
        if (row['identity'] != identity or row['input_metadata'] != identity['input_metadata']
                or not row['generation']['complete'] or row['generated_tokens'] != len(row['output_tokens'])):
            raise ValueError('target-only diagnostic output metadata is inconsistent')
        return row, {'path': str(path.resolve()), 'sha256': digest,
                     'manifest_path': str(manifest_path.resolve()), 'manifest_sha256': file_sha256(manifest_path)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, {'reason': f'missing or invalid matching target-only diagnostic: {error}'}


def execute_target_only(runner, job):
    if job['temperature'] != 0 or job['arm'] != 'plain_dflash' or job['prompt']['condition'] != 'standard_nonthinking':
        raise ValueError('target-only diagnostic is restricted to standard greedy test prompts')
    prepared = runner.prepare_inputs(job['prompt'])
    identity = target_only_identity(run_identity(runner, job, {}, 'evaluate'), prepared[2])
    previous, source = read_target_only(runner.args.output, identity)
    if previous is not None:
        return dict(source, status='reused', prompt_id=job['prompt']['prompt_id'], origin_stage=previous['origin_stage'])
    row = runner.run_target_only(job['prompt'], prepared)
    row['identity'] = identity
    path, manifest_path = target_only_paths(runner.args.output, identity)
    atomic_json(path, row)
    manifest = Manifest(manifest_path)
    manifest.data['jobs'][manifest.key(identity)] = {'identity': identity, 'status': 'complete',
        'result_path': str(path), 'result_sha256': file_sha256(path)}
    atomic_json(manifest_path, manifest.data)
    return {'status': 'generated', 'prompt_id': job['prompt']['prompt_id'], 'origin_stage': runner.args.stage,
            'path': str(path.resolve()), 'sha256': file_sha256(path), 'reason': source['reason']}


def greedy_token_mismatch(output, reference):
    count = max(len(output), len(reference))
    positions = [i for i in range(count) if i >= len(output) or i >= len(reference) or output[i] != reference[i]]
    return {'sequence_equal': not positions, 'mismatched_tokens': len(positions), 'compared_tokens': count,
            'token_mismatch_rate': len(positions) / count if count else 0.,
            'first_mismatch': positions[0] if positions else None,
            'output_length': len(output), 'reference_length': len(reference),
            'length_difference': len(output) - len(reference)}


def greedy_output_report(rows, jobs, output):
    expected = [job for job in jobs if job['temperature'] == 0 and job['prompt']['condition'] == 'standard_nonthinking']
    by_key = {(row['prompt_id'], row['seed'], row['arm']): row for row in rows
              if row['temperature'] == 0 and row['condition'] == 'standard_nonthinking'}
    comparisons, missing = [], []
    targets = {}
    for job in expected:
        key = (job['prompt']['prompt_id'], job['seed'])
        row = by_key.get((*key, job['arm']))
        plain = by_key.get((*key, 'plain_dflash'))
        for reference in ('plain_dflash', 'target_only'):
            label = {'prompt_id': key[0], 'seed': key[1], 'arm': job['arm'], 'reference': reference}
            if row is None:
                missing.append(dict(label, reason='missing greedy arm output'))
                continue
            if not row.get('input_metadata'):
                missing.append(dict(label, reason='missing tokenized input/cap provenance'))
                continue
            if reference == 'plain_dflash':
                if plain is None:
                    missing.append(dict(label, reason='missing paired plain-DFlash greedy output'))
                    continue
                def comparison_identity(value):
                    return {k: v for k, v in baseline_generation_identity(value['identity']).items()
                            if k not in ('proposal_mode', 'setting')}
                if (comparison_identity(row) != comparison_identity(plain)
                        or row['input_metadata'] != plain.get('input_metadata')):
                    missing.append(dict(label, reason='plain-DFlash generation/input identities do not match'))
                    continue
                reference_row, provenance = plain, {'identity_sha256': fingerprint(plain['identity'])}
            else:
                identity = target_only_identity(row['identity'], row['input_metadata'])
                target_key = fingerprint(identity)
                if target_key not in targets:
                    targets[target_key] = read_target_only(output, identity)
                reference_row, provenance = targets[target_key]
                if reference_row is None:
                    missing.append(dict(label, reason=provenance['reason']))
                    continue
            comparisons.append(dict(label, **greedy_token_mismatch(row['output_tokens'], reference_row['output_tokens']),
                                    output_generation=row['generation'], reference_generation=reference_row['generation'],
                                    reference_provenance=provenance))
    groups = []
    for arm in sorted({job['arm'] for job in expected}):
        expected_requests = sum(job['arm'] == arm for job in expected)
        for reference in ('plain_dflash', 'target_only'):
            matched = [row for row in comparisons if row['arm'] == arm and row['reference'] == reference]
            compared_tokens = sum(row['compared_tokens'] for row in matched)
            mismatched_tokens = sum(row['mismatched_tokens'] for row in matched)
            groups.append({'arm': arm, 'reference': reference, 'expected_requests': expected_requests,
                'compared_requests': len(matched), 'missing_requests': expected_requests - len(matched),
                'mismatched_sequences': sum(not row['sequence_equal'] for row in matched),
                'sequence_mismatch_rate': sum(not row['sequence_equal'] for row in matched) / len(matched) if matched else None,
                'mismatched_tokens': mismatched_tokens, 'compared_tokens': compared_tokens,
                'token_mismatch_rate': mismatched_tokens / compared_tokens if compared_tokens else None,
                'length_mismatched_sequences': sum(row['length_difference'] != 0 for row in matched)})
    return {'status': 'complete' if expected and not missing else 'incomplete',
            'scope': 'all standard nonthinking greedy test prompts; outputs only, no timing comparisons',
            'expected_prompts': len({job['prompt']['prompt_id'] for job in expected}),
            'expected_comparisons': len(expected) * 2, 'completed_comparisons': len(comparisons),
            'missing_comparisons': missing, 'groups': groups, 'comparisons': comparisons,
            'numerical_mismatches_observed': any(not row['sequence_equal'] for row in comparisons),
            'interpretation': 'Coverage gates completeness. Real-model BF16/tie-breaking mismatches are reported, not automatically failed; toy greedy equality and ratio-verifier exactness are separate guarantees.'}


def evaluate(runner, prompt_manifest, settings):
    args = runner.args
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    jobs = evaluation_jobs(prompts, args.stage, args.scale, args.seed)
    frozen_hash = fingerprint(settings)
    schedule = getattr(args, 'schedule', 'modulo')
    rows, target_diagnostics = [], []
    if schedule == 'dynamic' and args.world_size > 1:
        from .runtime import WorkQueue
        directory = args.output / args.stage / 'evaluate'
        directory.mkdir(parents=True, exist_ok=True)
        def _scan_done_across_ranks():
            done = {}
            for mpath in sorted(directory.glob('manifest-rank*.json')):
                try:
                    entries = Manifest(mpath).data.get('jobs', {})
                except (OSError, ValueError):
                    continue
                for key, entry in entries.items():
                    rpath = Path(entry.get('result_path', ''))
                    if entry.get('status') == 'complete' and rpath.is_file():
                        sha = entry.get('result_sha256')
                        if not sha or sha == file_sha256(rpath):
                            done[key] = str(rpath)
            return done

        done_across_ranks = _scan_done_across_ranks()
        jobs_by_prompt = defaultdict(list)
        for job in jobs:
            jobs_by_prompt[job['prompt']['prompt_id']].append(job)
        # Schedule long conditions first so 24K-token generations never straggle at the tail.
        ordered_prompts = sorted(prompts, key=lambda p: (0 if p['condition'].startswith('long_') else 1))
        ordered_ids = [p['prompt_id'] for p in ordered_prompts]
        precompleted = set()
        for pid in ordered_ids:
            pjobs = jobs_by_prompt[pid]
            if all(fingerprint(run_identity(runner, j, settings['selected'][j['arm']], 'evaluate', frozen_hash)) in done_across_ranks
                   for j in pjobs):
                td_jobs = [j for j in pjobs if j['temperature'] == 0 and j['arm'] == 'plain_dflash'
                           and j['prompt']['condition'] == 'standard_nonthinking']
                if all(read_target_only(args.output, target_only_identity(
                        run_identity(runner, j, {}, 'evaluate'), runner.prepare_inputs(j['prompt'])[2]))[0] is not None
                       for j in td_jobs):
                    precompleted.add(pid)
        queue = WorkQueue(directory / f'work_queue-{frozen_hash[:16]}.json', ordered_ids, precompleted=precompleted)
        while (claimed_id := queue.claim(args.rank)) is not None:
            done_across_ranks = _scan_done_across_ranks()
            for job in jobs_by_prompt[claimed_id]:
                identity = run_identity(runner, job, settings['selected'][job['arm']], 'evaluate', frozen_hash)
                ident_key = fingerprint(identity)
                # Recheck cross-rank manifest in case a previous worker finished some arms before crashing.
                existing_path = done_across_ranks.get(ident_key)
                if existing_path and Path(existing_path).is_file():
                    rows.append(json.loads(Path(existing_path).read_text()))
                else:
                    rows.append(execute_job(runner, job, settings['selected'][job['arm']], 'evaluate', frozen_hash))
                if (job['temperature'] == 0 and job['arm'] == 'plain_dflash'
                        and job['prompt']['condition'] == 'standard_nonthinking'):
                    target_diagnostics.append(execute_target_only(runner, job))
            queue.complete(claimed_id)
    else:
        prompt_rank = {p['prompt_id']: i % args.world_size for i, p in enumerate(prompts)}
        for job in jobs:
            if prompt_rank[job['prompt']['prompt_id']] != args.rank:
                continue
            rows.append(execute_job(runner, job, settings['selected'][job['arm']], 'evaluate', frozen_hash))
        target_diagnostics = [execute_target_only(runner, job) for job in jobs
                              if job['temperature'] == 0 and job['arm'] == 'plain_dflash'
                              and job['prompt']['condition'] == 'standard_nonthinking'
                              and prompt_rank[job['prompt']['prompt_id']] == args.rank]
    report = {'scale': args.scale, 'stage': args.stage, 'status': 'rank_complete', 'rank': args.rank,
              'world_size': args.world_size, 'rows': len(rows), 'frozen_settings_hash': frozen_hash,
              'baseline_reuse': baseline_reuse_report(rows),
              'greedy_target_only_diagnostics': target_diagnostics,
              'groups': summarize_rows(rows), 'timing_warning': 'Sharded evaluation timing is observational only; isolated timing session determines speed conclusions.'}
    atomic_json(args.output / args.stage / f'evaluation-rank{args.rank}.json', report)
    return report


def timing_schedule(jobs, settings, repeats):
    groups = defaultdict(list)
    for job in jobs:
        groups[(job['prompt']['prompt_id'], job['sampling_name'], job['seed'])].append(job)
    for group_index, same_prompt_jobs in enumerate(groups.values()):
        for repeat in range(repeats):
            offset = (group_index + repeat) % len(same_prompt_jobs)
            yield repeat, False, [(job, settings['selected'][job['arm']])
                                  for job in same_prompt_jobs[offset:] + same_prompt_jobs[:offset]]
        yield 'phase_profile', True, [(job, settings['selected'][job['arm']]) for job in same_prompt_jobs]


def audit_timing_evidence(runner, prompt_manifest, settings, report):
    """CPU-only completion gate over every expected raw timing/profile result."""
    try:
        args = runner.args
        frozen_hash = fingerprint(settings)
        if (report is None or report.get('status') != 'complete'
                or report.get('full_reference_prompt_coverage') is not True
                or report.get('frozen_settings_hash') != frozen_hash):
            raise ValueError('complete full-population timing report for current settings is missing')
        if report.get('stage') != args.stage or report.get('scale') != args.scale:
            raise ValueError('timing report stage/scale mismatch')
        prompts = select_prompts(prompt_manifest, 'test', args.scale)
        selected = [prompt for condition in sorted({p['condition'] for p in prompts})
                    for prompt in prompts if prompt['condition'] == condition]
        if (report['prompt_ids'] != [p['prompt_id'] for p in selected]
                or report['expected_full_prompt_count'] != len(prompts)):
            raise ValueError('timing report prompt coverage mismatch')
        repeats, warmups = report['repeats'], report['warmups_per_arm_condition']
        if type(repeats) is not int or repeats < 2 or type(warmups) is not int or warmups < 1:
            raise ValueError('timing evidence lacks required repeats/warmups')
        evidence = report['raw_evidence']
        session_path = Path(evidence['session_path'])
        session_bytes = session_path.read_bytes()
        if hashlib.sha256(session_bytes).hexdigest() != evidence['session_sha256']:
            raise ValueError('timing comparison session checksum changed')
        session = json.loads(session_bytes)
        config = session['configuration']
        expected_path = args.output / args.stage / 'timing' / 'sessions' / f'{fingerprint(config)}.json'
        if session_path.resolve() != expected_path.resolve() or session['logical_session'] != report['timing_session']:
            raise ValueError('timing session path/identity mismatch')
        expected_inner = {'frozen_settings_hash': frozen_hash,
            'prompt_manifest_hash': fingerprint(prompt_manifest), 'selected_prompts': report['prompt_ids'],
            'repeats': repeats, 'warmups': warmups}
        report_fastpath = None
        if 'fastpath' in report:
            raw_fp = report['fastpath']
            req_fp_keys = {'compile', 'cuda_graph', 'compile_mode'}
            opt_fp_keys = {'bf16_gemm', 'compile_drafter', 'compile_online'}
            if (not isinstance(raw_fp, dict)
                    or not req_fp_keys <= set(raw_fp.keys()) <= (req_fp_keys | opt_fp_keys)
                    or type(raw_fp['compile']) is not bool
                    or type(raw_fp['cuda_graph']) is not bool
                    or type(raw_fp['compile_mode']) is not str
                    or any(type(raw_fp[k]) is not bool for k in opt_fp_keys if k in raw_fp)):
                raise ValueError('invalid fastpath metadata in timing report')
            report_fastpath = raw_fp
        if hasattr(runner, 'fastpath'):
            expected_fp = runner.fastpath.config.to_dict() if runner.fastpath is not None else None
            if report_fastpath != expected_fp:
                raise ValueError('timing session configuration mismatch')
            if expected_fp is not None:
                expected_inner['fastpath'] = expected_fp
        elif any(getattr(args, k, False) for k in ('compile', 'cuda_graph', 'bf16_gemm', 'compile_drafter', 'compile_online')):
            from .fastpath import FastPathConfig
            cli_fp = FastPathConfig(
                compile=bool(getattr(args, 'compile', False)),
                cuda_graph=bool(getattr(args, 'cuda_graph', False)),
                bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
                compile_drafter=bool(getattr(args, 'compile_drafter', False)),
                compile_online=bool(getattr(args, 'compile_online', False)),
            ).to_dict()
            if report_fastpath != cli_fp:
                raise ValueError('timing session configuration mismatch')
            expected_inner['fastpath'] = cli_fp
        elif report_fastpath is not None:
            expected_inner['fastpath'] = report_fastpath
        expected_config = {'configuration': expected_inner, 'checkpoint_hashes': runner.checkpoint_hashes,
            'model_pair': runner.model_pair, 'model_revisions': runner.model_revisions,
            'generation_environment': getattr(runner, 'generation_environment', None),
            'generation_caps': generation_caps(args), 'stage': args.stage, 'scale': args.scale,
            'device': report['device'], 'seed': args.seed}
        if any(config.get(key) != value for key, value in expected_config.items()):
            raise ValueError('timing session configuration mismatch')
        manifest_path = args.output / args.stage / 'timing' / 'manifest-rank0.json'
        if Path(evidence['manifest_path']).resolve() != manifest_path.resolve():
            raise ValueError('timing manifest path mismatch')
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != evidence['manifest_sha256']:
            raise ValueError('timing manifest checksum changed')
        manifest = json.loads(manifest_bytes)
        jobs = evaluation_jobs(selected, args.stage, args.scale, args.seed)
        counts = {'timing_requests': 0, 'profile_requests': 0, 'bundles': 0}
        for repeat, profile, comparisons in timing_schedule(jobs, settings, repeats):
            rows = read_comparison_bundle(runner, 'timing', comparisons, frozen_hash, session,
                                          repeat, profile, manifest=manifest)
            if rows is None:
                raise ValueError('required all-arm timing/profile bundle is incomplete')
            counts['profile_requests' if profile else 'timing_requests'] += len(rows)
            counts['bundles'] += 1
        if (counts['timing_requests'] != len(jobs) * repeats or counts['profile_requests'] != len(jobs)
                or report['timed_requests'] != counts['timing_requests']
                or any(evidence.get(key) != value for key, value in counts.items())):
            raise ValueError('raw timing/profile evidence count mismatch')
        return {'status': 'complete', **counts, 'session_sha256': evidence['session_sha256'],
                'manifest_sha256': evidence['manifest_sha256']}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'status': 'incomplete', 'reason': str(error)}


def timing(runner, prompt_manifest, settings):
    args = runner.args
    if args.world_size != 1 or args.rank != 0:
        raise ValueError('latency/throughput timing requires one process on one idle GPU')
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    selected = []
    for condition in sorted({p['condition'] for p in prompts}):
        condition_prompts = [p for p in prompts if p['condition'] == condition]
        selected.extend(condition_prompts if args.timing_prompts == 0 else condition_prompts[:args.timing_prompts])
    jobs = evaluation_jobs(selected, args.stage, args.scale, args.seed)
    session_path, session_state, epoch = comparison_session(runner, 'timing', {'frozen_settings_hash': fingerprint(settings),
        'prompt_manifest_hash': fingerprint(prompt_manifest), 'selected_prompts': [p['prompt_id'] for p in selected],
        'repeats': args.timing_repeats, 'warmups': args.timing_warmups})
    session = session_state['logical_session']
    frozen_hash = fingerprint(settings)
    # All arms are warmed up with fresh request state before any measured runs.
    for condition in sorted({p['condition'] for p in selected}):
        prompt = next(p for p in selected if p['condition'] == condition)
        name, temperature, warp = sampling_conditions(condition, primary_only=True)[0]
        for _ in range(args.timing_warmups):
            for arm in arms_for(args.stage, args.scale):
                job = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature,
                       'warper': warp, 'seed': args.seed, 'arm': arm, 'max_new_tokens_override': 64}
                runner.run(job, settings['selected'][arm], profile=False)
    rows, profiles = [], []
    bundle_count = 0
    # Phase events add instrumentation overhead; their separate pass is required
    # evidence but is never charged as proposal computation in the speed pass.
    for repeat, profile, comparisons in timing_schedule(jobs, settings, args.timing_repeats):
        results = comparison_bundle(runner, 'timing', comparisons, frozen_hash,
                                    session_path, session_state, epoch, repeat, profile=profile)
        (profiles if profile else rows).extend(results)
        bundle_count += 1
    manifest_path = args.output / args.stage / 'timing' / 'manifest-rank0.json'
    full_coverage = len(selected) == len(prompts)
    report = {'scale': args.scale, 'stage': args.stage, 'status': 'complete' if full_coverage else 'pilot_complete',
              'full_reference_prompt_coverage': full_coverage, 'expected_full_prompt_count': len(prompts), 'timing_session': session,
              'frozen_settings_hash': frozen_hash, 'restart_epochs': session_state['epochs'],
              'resume_policy': 'reuse complete all-arm prompt/seed/repeat bundles only; rerun every arm of interrupted bundles in the new epoch',
              'device': str(runner.device), 'isolation': 'one process; idle-device check before model load',
              'warmups_per_arm_condition': args.timing_warmups, 'repeats': args.timing_repeats,
              'prompt_ids': [p['prompt_id'] for p in selected], 'timed_requests': len(rows),
              'raw_evidence': {'session_path': str(session_path.resolve()), 'session_sha256': file_sha256(session_path),
                               'manifest_path': str(manifest_path.resolve()), 'manifest_sha256': file_sha256(manifest_path),
                               'timing_requests': len(rows), 'profile_requests': len(profiles), 'bundles': bundle_count},
              'groups': summarize_rows(rows, isolated_timing=True), 'phase_profiles': summarize_rows(profiles, isolated_timing=True),
              'instrumentation_policy': 'Uninstrumented synchronized request wall time for speed; separate same-arm/same-prompt CUDA-event profile runs for phases.',
              'timing_scope': 'all required held-out test prompts and condition-specific seed counts' if full_coverage else 'pilot matched subset; cannot satisfy milestone throughput gate',
              'baseline_policy': 'all baselines remeasured in this same session; no M3/M4 throughput reuse'}
    if getattr(runner, 'fastpath', None) is not None:
        report['fastpath'] = runner.fastpath.config.to_dict()
    plain = {(g['condition'], g['temperature'], json.dumps(g['sampling_warp_config'], sort_keys=True)): g['metrics']['tokens_per_second']['mean']
             for g in report['groups'] if g['arm'] == 'plain_dflash'}
    paired_plain = {(row['condition'], row['temperature'], json.dumps(row['sampling_warp_config'], sort_keys=True), row['prompt_id'], row['seed'], row['repeat']): row
                    for row in rows if row['arm'] == 'plain_dflash'}
    for group in report['groups']:
        key = (group['condition'], group['temperature'], json.dumps(group['sampling_warp_config'], sort_keys=True))
        mean = group['metrics']['tokens_per_second']['mean']
        baseline = plain.get(key)
        group['throughput_relative_to_plain'] = mean / baseline if mean is not None and baseline else None
        group['throughput_improved_after_all_proposal_and_online_costs'] = mean > baseline if mean is not None and baseline else None
        paired, matched_arm, matched_plain = [], [], []
        excluded_pairs = 0
        for row in rows:
            row_key = (row['condition'], row['temperature'], json.dumps(row['sampling_warp_config'], sort_keys=True))
            if row['arm'] != group['arm'] or row_key != key:
                continue
            baseline_row = paired_plain[(*row_key, row['prompt_id'], row['seed'], row['repeat'])]
            if row['condition'].startswith('long_') and (row['generation']['repetition_dominated'] or baseline_row['generation']['repetition_dominated']):
                excluded_pairs += 1
                continue
            matched_arm.append(row)
            matched_plain.append(baseline_row)
            paired.append({'prompt_id': row['prompt_id'], 'seed': row['seed'], 'ratio': row['tokens_per_second'] / baseline_row['tokens_per_second']})
        ratio = prompt_bootstrap(paired, 'ratio')
        matched_arm_mean = prompt_bootstrap(matched_arm, 'tokens_per_second')['mean']
        matched_plain_mean = prompt_bootstrap(matched_plain, 'tokens_per_second')['mean']
        group['matched_timing_requests'] = len(matched_arm)
        group['repetition_excluded_comparison_pairs'] = excluded_pairs
        group['matched_arm_tokens_per_second'] = matched_arm_mean
        group['matched_plain_tokens_per_second'] = matched_plain_mean
        group['throughput_relative_to_plain'] = matched_arm_mean / matched_plain_mean if matched_plain_mean else None
        group['throughput_improved_after_all_proposal_and_online_costs'] = matched_arm_mean > matched_plain_mean if matched_plain_mean else None
        group['paired_throughput_ratio'] = ratio
        group['throughput_gain_supported_by_95_ci'] = ratio['ci95'][0] > 1 if ratio['ci95'] else None
    atomic_json(args.output / args.stage / 'timing.json', report)
    return report


FEATURE_FIELDS = {'coordinate_mean_norm', 'variance_ratio_to_standard_calibration',
                  'l2_norm_quantiles', 'outlier_slot_rate', 'outlier_coordinate_rate',
                  'outlier_threshold_abs_coordinate', 'calibration_reference'}


def _feature_key(condition, temperature, warper):
    return condition, float(temperature), json.dumps(warper, sort_keys=True)


def _required_feature_keys(prompts):
    return {_feature_key(prompt['condition'], temperature, asdict(warper))
            for prompt in prompts for _, temperature, warper in sampling_conditions(prompt['condition'])}


def extract_feature_context_groups(diagnostic, prompts):
    """Keep pre-adapter M1 features separate, including documented empty bins."""
    from .diagnose import OUTPUT_LABELS
    required = _required_feature_keys(prompts)
    selected, found = [], set()
    for group in diagnostic.get('groups', []):
        key = _feature_key(group['condition'], group['collection_temperature'], group['sampling_warp_config'])
        if key not in required or group.get('split') != 'test' or group.get('axis') != 'output_offset' or group.get('data_policy') != 'on_policy':
            continue
        if group.get('inclusion') not in ('primary', 'repetition_excluded') or group.get('bin') not in OUTPUT_LABELS:
            raise ValueError('M1b feature group has unknown inclusion or output bin')
        features = group.get('features', {})
        if not FEATURE_FIELDS <= features.keys() or group.get('logical_slots', 0) <= 0:
            raise ValueError('M1b feature group lacks required statistics or observed slots')
        numeric = [features[name] for name in FEATURE_FIELDS - {'l2_norm_quantiles', 'calibration_reference'}]
        quantiles = features['l2_norm_quantiles']
        if not isinstance(quantiles, dict) or not {'p0', 'p25', 'p50', 'p75', 'p95', 'p99', 'p100'} <= quantiles.keys():
            raise ValueError('M1b feature norm quantiles are incomplete')
        if any(not math.isfinite(float(value)) for value in numeric + list(quantiles.values())):
            raise ValueError('M1b feature statistics contain nonfinite values')
        if not all(0 <= features[name] <= 1 for name in ('outlier_slot_rate', 'outlier_coordinate_rate')):
            raise ValueError('M1b feature outlier rate is invalid')
        found.add(key)
        selected.append({name: group[name] for name in ('condition', 'collection_temperature', 'sampling_warp_config',
                         'split', 'axis', 'bin', 'data_policy', 'inclusion', 'logical_slots', 'features')})
    if missing := required - found:
        raise ValueError(f'M1b feature statistics missing for required condition/temperature/warper: {sorted(missing)}')
    coverage = []
    for condition, temperature, warp in sorted(required):
        groups = [group for group in selected if _feature_key(group['condition'], group['collection_temperature'], group['sampling_warp_config']) == (condition, temperature, warp)]
        observed = {group['bin'] for group in groups}
        primary = {group['bin'] for group in groups if group['inclusion'] == 'primary'}
        coverage.append({'condition': condition, 'collection_temperature': temperature, 'sampling_warp_config': json.loads(warp),
                         'observed_bins': [label for label in OUTPUT_LABELS if label in observed],
                         'empty_bins': [label for label in OUTPUT_LABELS if label not in observed],
                         'primary_empty_bins': [label for label in OUTPUT_LABELS if label not in primary],
                         'logical_slots': sum(group['logical_slots'] for group in groups),
                         'feature_group_count': len(groups)})
    return selected, coverage


def audited_m4_feature_context(args, prompt_manifest, settings, prompts):
    """Bind the M1b report to its accepted sources and the evaluation calibration.

    M1b verified complete shard bytes; this linkage validates the report, accepted
    source-set identities/checksums, audit, and current frozen calibration files.
    It does not substitute compact statistics from updated M4 arms.
    """
    path = Path(getattr(args, 'feature_diagnostic', None) or args.output / f'diagnostic_{args.scale}.json')
    diagnostic = json.loads(path.read_text())
    if diagnostic.get('status') != 'measured' or diagnostic.get('scale') != args.scale or diagnostic.get('label') != args.scale:
        raise ValueError('M4 requires a measured M1b diagnostic at the evaluation scale')
    provenance = diagnostic.get('audit_provenance')
    if not provenance:
        raise ValueError('M1b diagnostic has no accepted-audit provenance')
    audit_path = Path(provenance['audit_path'])
    if file_sha256(audit_path) != provenance['audit_sha256']:
        raise ValueError('M1 audit changed after the M1b diagnostic')
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'passed' or audit.get('scale') != args.scale:
        raise ValueError('M1 audit is not passed at the evaluation scale')
    sources = audit['source_shards']
    if fingerprint(sources) != provenance['source_set_sha256']:
        raise ValueError('M1b source identity set differs from the accepted audit')
    def source_map(rows):
        result = {str(Path(row['path']).resolve()): row['sha256'] for row in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate M1b source path')
        return result
    if source_map(diagnostic['source_shards']) != source_map(sources):
        raise ValueError('M1b diagnostic source checksums differ from the accepted audit')
    calibration = audit['calibration']
    calibration_path = audit_path.parent / 'calibration.json'
    if fingerprint(json.loads(calibration_path.read_text())) != fingerprint(calibration):
        raise ValueError('calibration record changed after the M1 audit')
    if calibration['identity']['model_pair'] != settings['model_pair'] or calibration['identity']['checkpoint'] != settings['model_revisions']:
        raise ValueError('M1 feature model identities differ from M4 evaluation')
    artifact_hashes = calibration['artifact_sha256']
    if provenance['frozen_artifact_sha256'] != artifact_hashes or artifact_hashes['h_proj.pt'] != settings['checkpoint_hashes']['projection']:
        raise ValueError('M1 feature projection hashes differ from M4 calibration')
    for name, digest in artifact_hashes.items():
        if file_sha256(audit_path.parent / name) != digest:
            raise ValueError(f'frozen feature calibration artifact changed: {name}')
    manifest_path = Path(calibration['resolved_config']['prompt_manifest'])
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash != calibration['identity']['prompt_manifest_sha256'] or manifest_hash != provenance['prompt_manifest_sha256']:
        raise ValueError('M1 feature prompt manifest checksum changed')
    if fingerprint(json.loads(manifest_path.read_text())) != fingerprint(prompt_manifest) or fingerprint(prompt_manifest) != settings['prompt_manifest_hash']:
        raise ValueError('M1 feature prompts differ from frozen M4 evaluation prompts')
    observed = {(generation['prompt_id'], *_feature_key(generation['condition'], generation['temperature'], generation['sampling_warp_config']))
                for generation in audit['generations'] if generation['split'] == 'test' and generation.get('complete', False)}
    expected = {(prompt['prompt_id'], *_feature_key(prompt['condition'], temperature, asdict(warp)))
                for prompt in prompts for _, temperature, warp in sampling_conditions(prompt['condition'])}
    if expected - observed:
        raise ValueError('audited M1 features do not cover the required M4 test prompt/condition identities')
    groups, coverage = extract_feature_context_groups(diagnostic, prompts)
    return {'status': 'complete', 'label': args.scale, 'source': 'plain DFlash M1b before offline fit and online adaptation',
            'interpretation': 'Feature covariate shift is context from the plain-DFlash M1 trace distribution, not measurements of updated M4 arms.',
            'diagnostic_path': str(path.resolve()), 'diagnostic_sha256': file_sha256(path),
            'audit_path': str(audit_path.resolve()), 'audit_sha256': provenance['audit_sha256'],
            'calibration_path': str(calibration_path.resolve()), 'calibration_sha256': file_sha256(calibration_path),
            'calibration_identity': calibration['identity'], 'frozen_artifact_sha256': artifact_hashes,
            'prompt_manifest_sha256': manifest_hash, 'source_set_sha256': provenance['source_set_sha256'],
            'source_shard_count': len(sources), 'matched_test_prompt_conditions': len(expected),
            'feature_group_count': len(groups), 'groups': groups, 'condition_bin_coverage': coverage}


def synthetic_m4_feature_context(args, prompt_manifest, settings, prompts):
    """Recompute plain-toy context separately from the measured adapter arms."""
    from .diagnose import DiagnosticAccumulator, live_block_diagnostics
    from .dflash_loop import generate_toy
    if torch.device(args.device).type != 'cpu':
        raise ValueError('synthetic feature context must remain CPU-only')
    runner = EvaluationRunner(args)
    if runner.checkpoint_hashes != settings['checkpoint_hashes']:
        raise ValueError('synthetic feature calibration does not match frozen evaluation state')
    accumulator = DiagnosticAccumulator(seed=args.seed)
    for prompt_index, prompt in enumerate(prompts):
        prefix = [1, 2, 3, int(hashlib.sha256(prompt['prompt_id'].encode()).hexdigest()[:4], 16) % 8]
        for name, temperature, warp in sampling_conditions(prompt['condition']):
            records = []
            def callback(record):
                record['diagnostics'] = live_block_diagnostics(record, seed=args.seed + 10000 * prompt_index + record['round_idx'])
                record.update({'prompt_id': prompt['prompt_id'], 'seed': args.seed, 'condition': prompt['condition'], 'split': 'test'})
                records.append(record)
            result = generate_toy(runner.target, runner.drafter, prefix, args.smoke_tokens,
                temperature=temperature, warper=warp, generator=torch.Generator().manual_seed(args.seed),
                projection=runner.projection, on_block=callback)
            generation = generation_diagnostics(result.output_tokens)
            for record in records:
                accumulator.add_block(record, repetition_dominated=generation['repetition_dominated'])
    diagnostic = accumulator.report()
    groups, coverage = extract_feature_context_groups(diagnostic, prompts)
    digest = hashlib.sha256()
    for tensor in (runner.projection.matrix, runner.projection.mean, runner.projection.std):
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return {'status': 'complete', 'label': 'synthetic', 'source': 'plain-toy feature context recomputed separately on smoke test prompts',
            'interpretation': 'Synthetic reporting exercise; not evidence of Qwen feature shift and not features from updated adapter arms.',
            'checkpoint_hashes': runner.checkpoint_hashes, 'prompt_manifest_hash': fingerprint(prompt_manifest),
            'calibration_projection_tensor_sha256': digest.hexdigest(), 'calibration_metadata': runner.projection.metadata,
            'feature_group_count': len(groups), 'groups': groups, 'condition_bin_coverage': coverage}


def m4_feature_context(args, prompt_manifest, settings, prompts):
    try:
        context = synthetic_m4_feature_context(args, prompt_manifest, settings, prompts) if args.scale == 'smoke' else audited_m4_feature_context(args, prompt_manifest, settings, prompts)
        artifact = args.output / 'm4' / 'feature_context.json'
        atomic_json(artifact, context)
        return dict(context, artifact_path=str(artifact.resolve()), artifact_sha256=file_sha256(artifact))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {'status': 'incomplete', 'reason': str(error),
                'source': 'required plain-DFlash M1b feature context', 'groups': [], 'feature_group_count': 0}


def aggregate(args, prompt_manifest, settings):
    from types import SimpleNamespace
    runner = SimpleNamespace(args=args, tiny=args.scale == 'smoke', checkpoint_hashes=settings['checkpoint_hashes'],
                             model_pair=settings['model_pair'], model_revisions=settings['model_revisions'],
                             generation_environment=settings.get('generation_environment'))
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    jobs = evaluation_jobs(prompts, args.stage, args.scale, args.seed)
    frozen_hash = fingerprint(settings)
    directory = args.output / args.stage / 'evaluate'
    available = {}
    for manifest_path in directory.glob('manifest-rank*.json'):
        manifest = Manifest(manifest_path)
        available.update(manifest.data['jobs'])
    rows, missing = [], []
    for job in jobs:
        identity = run_identity(runner, job, settings['selected'][job['arm']], 'evaluate', frozen_hash)
        entry = available.get(fingerprint(identity))
        if entry is None or entry['status'] != 'complete' or not Path(entry['result_path']).is_file():
            missing.append({'prompt_id': job['prompt']['prompt_id'], 'arm': job['arm'], 'temperature': job['temperature'], 'seed': job['seed']})
            continue
        row = json.loads(Path(entry['result_path']).read_text())
        if entry.get('result_sha256') and file_sha256(entry['result_path']) != entry['result_sha256']:
            raise ValueError('evaluation result checksum changed')
        if row['identity'] != identity:
            raise ValueError('evaluation result identity mismatch')
        rows.append(row)
    timing_path = args.output / args.stage / 'timing.json'
    timed = json.loads(timing_path.read_text()) if timing_path.exists() else None
    timing_evidence = audit_timing_evidence(runner, prompt_manifest, settings, timed)
    timing_valid = timing_evidence['status'] == 'complete'
    feature_context = m4_feature_context(args, prompt_manifest, settings, prompts) if args.stage == 'm4' else None
    feature_context_valid = feature_context is None or feature_context['status'] == 'complete'
    greedy_report = greedy_output_report(rows, jobs, args.output)
    greedy_path = args.output / args.stage / 'greedy_output_comparisons.json'
    atomic_json(greedy_path, greedy_report)
    greedy_complete = greedy_report['status'] == 'complete'
    status = 'complete' if not missing and timing_valid and feature_context_valid and greedy_complete else 'incomplete'
    report = {'scale': args.scale, 'label': 'synthetic' if args.scale == 'smoke' else args.scale,
              'stage': args.stage, 'status': status, 'expected_requests': len(jobs), 'completed_requests': len(rows),
              'missing_requests': missing, 'settings': settings, 'groups': summarize_rows(rows),
              'baseline_reuse': baseline_reuse_report(rows),
              'greedy_output_comparisons_complete': greedy_complete,
              'greedy_output_comparisons': dict(greedy_report, artifact_path=str(greedy_path.resolve()), artifact_sha256=file_sha256(greedy_path)),
              'feature_context_complete': feature_context_valid if args.stage == 'm4' else None,
              'feature_context': feature_context,
              'isolated_timing_complete': timing_valid, 'timing_report': str(timing_path) if timing_valid else None,
              'timing_evidence': timing_evidence,
              'external_baselines_not_run': ['full tree-based Token Recycling', 'Test-Time Speculation', 'SharpDraft'],
              'baseline_naming': 'online_bigram is the normalized committed-transition baseline, not full Token Recycling',
              'long_condition': sorted({p['condition'] for p in prompts if p['condition'].startswith('long_')}),
              'thinking_drafter_distribution': 'out_of_distribution' if any(p['condition'] == 'long_thinking' for p in prompts) else 'long_thinking unavailable or not selected; see prompt manifest'}
    atomic_json(args.output / args.stage / 'summary.json', report)
    return report


def smoke_manifest():
    return {'scale': 'smoke', 'prompts': [{'prompt_id': f'synthetic-{split}-{condition}-{i}', 'split': split,
        'condition': condition, 'text': f'Synthetic {i}', 'dataset': 'builtin', 'reference_selected': True}
        for split in ('dev', 'test') for condition in ('standard_nonthinking', 'long_thinking') for i in range(2)]}


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument('--stage', choices=['m3', 'm4'], default='m3')
    parser.add_argument('--mode', choices=['tune', 'evaluate', 'timing', 'summarize', 'all'], default='all')
    parser.add_argument('--fresh_session', action='store_true', help='start new dev/timing measurements instead of resuming complete comparison bundles')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--schedule', choices=['modulo', 'dynamic'], default='modulo')
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--cuda_graph', action='store_true', default=False)
    parser.add_argument('--bf16_gemm', action='store_true', default=False)
    parser.add_argument('--compile_drafter', action='store_true', default=False)
    parser.add_argument('--compile_online', action='store_true', default=False)
    parser.add_argument('--prompts_manifest', type=Path, default=ROOT / 'artifacts/prompts.json')
    parser.add_argument('--dense_checkpoint', type=Path, default=ROOT / 'artifacts/m2/dense.pt')
    parser.add_argument('--sparse_checkpoint', type=Path, default=ROOT / 'artifacts/m2/dense_sparse.pt')
    parser.add_argument('--projection', type=Path, default=ROOT / 'artifacts/h_proj.pt')
    parser.add_argument('--feature_diagnostic', type=Path, help='M4 plain-DFlash M1b report; default artifacts/diagnostic_<scale>.json')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--long_max_new_tokens', type=int, default=24576)
    parser.add_argument('--smoke_tokens', type=int, default=32)
    parser.add_argument('--standard_dev_prompts', type=int, default=8)
    parser.add_argument('--long_dev_prompts', type=int, default=12)
    parser.add_argument('--timing_prompts', type=int, default=0, help='0 times every required test prompt; positive values are incomplete pilot subsets')
    parser.add_argument('--timing_repeats', type=int, default=3)
    parser.add_argument('--timing_warmups', type=int, default=1)
    args = parse_config(parser, argv)
    for name in ('prompts_manifest', 'dense_checkpoint', 'sparse_checkpoint', 'projection'):
        setattr(args, name, Path(getattr(args, name)))
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error('rank must lie in 0..world_size-1')
    if args.timing_repeats < 2 or args.timing_warmups < 1 or args.timing_prompts < 0:
        parser.error('timing requires repeats>=2, warmups>=1 and prompts>=0 (0=all)')
    if min(args.max_new_tokens, args.long_max_new_tokens, args.smoke_tokens) < 1:
        parser.error('generation caps must be positive')
    if args.mode == 'all' and args.world_size != 1:
        parser.error('shard only --mode evaluate; tune/timing run once on an idle GPU')
    record_run(args, f'{args.stage}-{args.mode}')
    torch.set_num_threads(1)
    prompt_manifest = smoke_manifest() if args.scale == 'smoke' else json.loads(args.prompts_manifest.read_text())
    if args.scale != 'smoke':
        if args.mode != 'summarize':
            preflight(args, (args.standard_dev_prompts * args.max_new_tokens + args.long_dev_prompts * args.long_max_new_tokens) if args.mode == 'tune' else 64 * args.max_new_tokens + 32 * args.long_max_new_tokens)
        if args.mode in ('tune', 'timing', 'all'):
            from .bench_adapter import assert_idle
            assert_idle(torch.device(args.device))
    settings_path = args.output / args.stage / 'settings.json'
    runner = None if args.mode == 'summarize' else EvaluationRunner(args)
    if args.mode in ('tune', 'all'):
        settings = tune(runner, prompt_manifest)
    else:
        settings = json.loads(settings_path.read_text())
    current_hashes = (runner.checkpoint_hashes if runner is not None else dict(settings['checkpoint_hashes'], implementation=implementation_hash())) if args.scale == 'smoke' else checkpoint_hashes(args)
    validate_frozen_settings(settings, args.stage, args.scale, current_hashes, prompt_manifest,
        generation_caps(args), runner.model_pair if runner is not None else [args.target, args.drafter] if args.scale != 'smoke' else settings['model_pair'],
        runner.model_revisions if runner is not None else None,
        runner.generation_environment if runner is not None else None)
    if args.mode in ('evaluate', 'all'):
        evaluate(runner, prompt_manifest, settings)
    if args.mode in ('timing', 'all'):
        timing(runner, prompt_manifest, settings)
    if args.mode in ('evaluate', 'timing', 'summarize', 'all'):
        result = aggregate(args, prompt_manifest, settings)
    else:
        result = {'stage': args.stage, 'scale': args.scale, 'status': 'settings_frozen'}
    print(json.dumps({'stage': args.stage, 'scale': args.scale, 'mode': args.mode, 'status': result['status']}))


if __name__ == '__main__':
    main()
