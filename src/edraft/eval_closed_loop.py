"""M3/M4 dev-only selection, frozen evaluation, and isolated matched-arm timing."""
from __future__ import annotations

from collections import defaultdict
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
    paths = [Path(__file__).parent / name for name in ('adapter.py', 'online.py', 'features.py', 'dflash_loop.py', 'sampler.py', 'verify.py', 'backend.py')]
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


def validate_frozen_settings(settings, stage, scale, checkpoint_hashes, prompt_manifest, generation_caps=None, model_pair=None, model_revisions=None):
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
                                 bias_sparse_only=arm == 'online_dbS', **kwargs)
        max_tokens = args.smoke_tokens if self.tiny else args.long_max_new_tokens if prompt['condition'].startswith('long_') else args.max_new_tokens
        max_tokens = min(max_tokens, job.get('max_new_tokens_override', max_tokens))
        generator = torch.Generator(device=self.device).manual_seed(job['seed'])
        if self.tiny:
            prefix = [1, 2, 3, int(hashlib.sha256(prompt['prompt_id'].encode()).hexdigest()[:4], 16) % 8]
            result = generate_toy(self.target, self.drafter, prefix, max_tokens, temperature=temperature, warper=warper,
                generator=generator, adapter=adapter, online_state=online, projection=self.projection if adapter is not None or online is not None else None,
                bigram=bigram, beta=beta, measure_phases=profile)
            eos = False
        else:
            inputs = self.backend.tokenize(prompt['text'], enable_thinking=prompt['condition'] == 'long_thinking')
            max_tokens = min(max_tokens, self.backend.target.config.max_position_embeddings - inputs.shape[1])
            if max_tokens < 1:
                raise ValueError('prompt exhausts model context')
            result = generate_reference(self.backend, inputs, max_tokens, temperature=temperature, warper=warper,
                generator=generator, adapter=adapter, online_state=online, projection=self.projection if adapter is not None or online is not None else None,
                bigram=bigram, beta=beta, measure_phases=profile)
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


def run_identity(runner, job, setting, mode, frozen_hash='', session=None, repeat=None):
    return {'model_pair': runner.model_pair, 'prompt': job['prompt']['prompt_id'],
            'prompt_payload_hash': fingerprint(job['prompt']),
            'condition': job['prompt']['condition'], 'seed': job['seed'], 'temperature': job['temperature'],
            'verification_rule': 'ratio' if job['temperature'] > 0 else 'greedy',
            'proposal_mode': job['arm'], 'checkpoint': runner.checkpoint_hashes, 'adapter_version': 0,
            'model_revisions': runner.model_revisions, 'stage': runner.args.stage, 'mode': mode,
            'sampling_warp_config': asdict(job['warper']), 'setting': setting, 'frozen_settings_hash': frozen_hash,
            'generation_cap': runner.args.smoke_tokens if runner.tiny else runner.args.long_max_new_tokens if job['prompt']['condition'].startswith('long_') else runner.args.max_new_tokens,
            'timing_session': session, 'repeat': repeat}


def execute_job(runner, job, setting, mode, frozen_hash='', session=None, repeat=None, profile=False):
    args = runner.args
    directory = args.output / args.stage / mode
    manifest = Manifest(directory / f'manifest-rank{args.rank}.json')
    identity = run_identity(runner, job, setting, mode, frozen_hash, session, repeat)
    path = directory / f'rank{args.rank}' / f'{manifest.key(identity)}.json'
    if manifest.completed(identity):
        return json.loads(path.read_text())
    row = runner.run(job, setting, profile)
    row.update({'identity': identity, 'timing_session': session, 'repeat': repeat,
                'timing_isolated': mode == 'timing', 'rank': args.rank})
    atomic_json(path, row)
    manifest.finish(identity, path)
    return row


def generation_caps(args):
    return {'standard': args.max_new_tokens, 'long': args.long_max_new_tokens, 'smoke': args.smoke_tokens}


def comparison_session(runner, mode, configuration):
    args = runner.args
    config = {'configuration': configuration, 'checkpoint_hashes': runner.checkpoint_hashes,
              'model_pair': runner.model_pair, 'model_revisions': runner.model_revisions,
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


def comparison_bundle(runner, mode, jobs_and_settings, frozen_hash, session_path, session_state, epoch, repeat, profile=False):
    bundle_key = fingerprint({'jobs': [(job['prompt'], job['arm'], job['sampling_name'], job['seed'], setting) for job, setting in jobs_and_settings],
                              'repeat': repeat, 'profile': profile})
    prior = session_state['bundles'].get(bundle_key)
    if prior is not None and prior['status'] == 'complete' and all(Path(p).is_file() for p in prior['result_paths']):
        return [json.loads(Path(path).read_text()) for path in prior['result_paths']]
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
        metrics = {metric: prompt_bootstrap(primary, metric) for metric in ('mean_accepted_length', 'median_accepted_length', 'target_forwards_per_generated_token')}
        if any(row.get('estimated_online_state_bandwidth_GBps') is not None for row in primary):
            metrics['estimated_online_state_bandwidth_GBps'] = prompt_bootstrap(primary, 'estimated_online_state_bandwidth_GBps')
        if isolated_timing:
            metrics['tokens_per_second'] = prompt_bootstrap(primary, 'tokens_per_second')
            metrics['elapsed_seconds'] = prompt_bootstrap(primary, 'elapsed_seconds')
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
        latency = torch.tensor([row['elapsed_seconds'] for row in primary], dtype=torch.float64)
        phases = defaultdict(list)
        for row in primary:
            for phase, value in row.get('phase_ms', {}).items():
                phases[phase].append(value)
        reports.append({'condition': condition, 'temperature': temperature, 'sampling_warp_config': json.loads(warp), 'arm': arm,
            'requests': len(group), 'primary_requests': len(primary), 'repetition_excluded': len(group) - len(primary),
            'metrics': metrics,
            'pooled_block_accepted_length_mean': statistics.mean(pooled_lengths) if pooled_lengths else None,
            'pooled_block_accepted_length_median': statistics.median(pooled_lengths) if pooled_lengths else None,
            'metric_aggregation': 'metrics are prompt-weighted after seed aggregation; median_accepted_length aggregates request medians; pooled block statistics are separate',
            'acceptance_by_block_position': {str(i): rates(data) for i, data in sorted(by_position.items())},
            'acceptance_bins': [{'axis': axis, 'bin': label, **rates(data), 'accepted_length': prompt_bootstrap(accepted_in_bins[(axis, label)], 'accepted_length')} for (axis, label), data in sorted(bins.items())],
            'request_latency_p50_seconds': float(torch.quantile(latency, .5)) if len(latency) else None,
            'request_latency_p95_seconds': float(torch.quantile(latency, .95)) if len(latency) else None,
            'request_latency_p50_prompt_bootstrap': prompt_quantile_ci(primary, 'elapsed_seconds', .5) if isolated_timing else None,
            'request_latency_p95_prompt_bootstrap': prompt_quantile_ci(primary, 'elapsed_seconds', .95) if isolated_timing else None,
            'phase_ms_median': {phase: statistics.median(values) for phase, values in phases.items()},
            'realized_output_length_mean': statistics.mean(row['generated_tokens'] for row in primary) if primary else None,
            'realized_generations_above_8K': sum(row['generated_tokens'] > 8192 for row in primary),
            'realized_generations_above_16K': sum(row['generated_tokens'] > 16384 for row in primary),
            'peak_allocated_bytes': max((row['peak_allocated_bytes'] or 0 for row in group), default=0),
            'peak_reserved_bytes': max((row['peak_reserved_bytes'] or 0 for row in group), default=0),
            'online_final_state': [row['online'] for row in primary if row['online'] is not None]})
    return reports


def evaluate(runner, prompt_manifest, settings):
    args = runner.args
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    jobs = evaluation_jobs(prompts, args.stage, args.scale, args.seed)
    prompt_rank = {p['prompt_id']: i % args.world_size for i, p in enumerate(prompts)}
    frozen_hash = fingerprint(settings)
    rows = []
    for job in jobs:
        if prompt_rank[job['prompt']['prompt_id']] != args.rank:
            continue
        rows.append(execute_job(runner, job, settings['selected'][job['arm']], 'evaluate', frozen_hash))
    report = {'scale': args.scale, 'stage': args.stage, 'status': 'rank_complete', 'rank': args.rank,
              'world_size': args.world_size, 'rows': len(rows), 'frozen_settings_hash': frozen_hash,
              'groups': summarize_rows(rows), 'timing_warning': 'Sharded evaluation timing is observational only; isolated timing session determines speed conclusions.'}
    atomic_json(args.output / args.stage / f'evaluation-rank{args.rank}.json', report)
    return report


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
    groups = defaultdict(list)
    for job in jobs:
        groups[(job['prompt']['prompt_id'], job['sampling_name'], job['seed'])].append(job)
    rows, profiles = [], []
    for group_index, same_prompt_jobs in enumerate(groups.values()):
        for repeat in range(args.timing_repeats):
            offset = (group_index + repeat) % len(same_prompt_jobs)
            comparisons = [(job, settings['selected'][job['arm']]) for job in same_prompt_jobs[offset:] + same_prompt_jobs[:offset]]
            rows.extend(comparison_bundle(runner, 'timing', comparisons, frozen_hash, session_path, session_state, epoch, repeat, profile=False))
        # Phase events add CPU instrumentation overhead. Profile separately so
        # phase-event creation is not silently charged as adapter computation.
        comparisons = [(job, settings['selected'][job['arm']]) for job in same_prompt_jobs]
        profiles.extend(comparison_bundle(runner, 'timing', comparisons, frozen_hash, session_path, session_state, epoch, 'phase_profile', profile=True))
    full_coverage = len(selected) == len(prompts)
    report = {'scale': args.scale, 'stage': args.stage, 'status': 'complete' if full_coverage else 'pilot_complete',
              'full_reference_prompt_coverage': full_coverage, 'expected_full_prompt_count': len(prompts), 'timing_session': session,
              'frozen_settings_hash': frozen_hash, 'restart_epochs': session_state['epochs'],
              'resume_policy': 'reuse complete all-arm prompt/seed/repeat bundles only; rerun every arm of interrupted bundles in the new epoch',
              'device': str(runner.device), 'isolation': 'one process; idle-device check before model load',
              'warmups_per_arm_condition': args.timing_warmups, 'repeats': args.timing_repeats,
              'prompt_ids': [p['prompt_id'] for p in selected], 'timed_requests': len(rows),
              'groups': summarize_rows(rows, isolated_timing=True), 'phase_profiles': summarize_rows(profiles, isolated_timing=True),
              'instrumentation_policy': 'Uninstrumented synchronized request wall time for speed; separate same-arm/same-prompt CUDA-event profile runs for phases.',
              'timing_scope': 'all required held-out test prompts and condition-specific seed counts' if full_coverage else 'pilot matched subset; cannot satisfy milestone throughput gate',
              'baseline_policy': 'all baselines remeasured in this same session; no M3/M4 throughput reuse'}
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


def aggregate(args, prompt_manifest, settings):
    from types import SimpleNamespace
    runner = SimpleNamespace(args=args, tiny=args.scale == 'smoke', checkpoint_hashes=settings['checkpoint_hashes'],
                             model_pair=settings['model_pair'], model_revisions=settings['model_revisions'])
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
        if row['identity'] != identity:
            raise ValueError('evaluation result identity mismatch')
        rows.append(row)
    timing_path = args.output / args.stage / 'timing.json'
    timed = json.loads(timing_path.read_text()) if timing_path.exists() else None
    timing_valid = timed is not None and timed.get('status') == 'complete' and timed.get('full_reference_prompt_coverage', False) and timed.get('frozen_settings_hash') == frozen_hash
    status = 'complete' if not missing and timing_valid else 'incomplete'
    report = {'scale': args.scale, 'label': 'synthetic' if args.scale == 'smoke' else args.scale,
              'stage': args.stage, 'status': status, 'expected_requests': len(jobs), 'completed_requests': len(rows),
              'missing_requests': missing, 'settings': settings, 'groups': summarize_rows(rows),
              'isolated_timing_complete': timing_valid, 'timing_report': str(timing_path) if timing_valid else None,
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
    parser.add_argument('--prompts_manifest', type=Path, default=ROOT / 'artifacts/prompts.json')
    parser.add_argument('--dense_checkpoint', type=Path, default=ROOT / 'artifacts/m2/dense.pt')
    parser.add_argument('--sparse_checkpoint', type=Path, default=ROOT / 'artifacts/m2/dense_sparse.pt')
    parser.add_argument('--projection', type=Path, default=ROOT / 'artifacts/h_proj.pt')
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
        runner.model_revisions if runner is not None else None)
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
