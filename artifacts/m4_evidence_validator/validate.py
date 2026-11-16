"""Read-only M4 artifact audit for the frozen mandatory reference plan.

Uses only the standard library, streams raw rows, and never loads a model.
Smoke validates the same artifact schema with the built-in four-prompt fixture.
This supplements the experiment gates; it is not evidence of model exactness.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

from online_checks import check_online_row

ARMS = ('plain_dflash', 'online_bigram', 'offline_static',
        'offline_online_stride1', 'online_dbS')
ONLINE = ARMS[-2:]
ALGORITHM_FILES = ('adapter.py', 'online.py', 'features.py', 'dflash_loop.py',
    'sampler.py', 'verify.py', 'backend.py', 'sparse.py', 'toy.py', 'diagnose.py')
IMPLEMENTATION_FILES = (*ALGORITHM_FILES, 'eval_closed_loop.py', 'fastpath.py', 'optimizers.py')
FROZEN_09A5469_ALGORITHM_FILES = {
    'adapter.py': 'b8960babc3f3b126a11e1f990ef144b2a9be6331a181bdbeea3da2f4c0c5b9d5',
    'online.py': 'ba781eb2f0a5c06b29e9101001724c1045b6bdc845b13400e5305fc603fe8b22',
    'features.py': 'adfbfa21ab6622f730df33513142c65a7d3c5adc6079ddb7f444d6471e5a6bff',
    'dflash_loop.py': '06001ce2fd529475e1c31e3aa2bfae69d203e70d72ab2a09b41ad794d8fa60d2',
    'sampler.py': '78f42af34ae1c1fe181273331c121d0fa7f17f8b62d5c9616c47de517892e38f',
    'verify.py': '0f664b84a4dbe963432386b5d8732aff112271ef341422eda254da08f39f3fea',
    'backend.py': '7f9b59880136f5de0ff63ce6b13e8351d81f3f2b15eaeac0a8a365f6a7a5ee78',
    'sparse.py': '66beee3396ade7a6104729086a6a68c0180b5dd2e89909448cf3430344747864',
    'toy.py': '5b7c325ede7455d219ba9fd93eb086daf656695608eece0129269fca1ee9494e',
    'diagnose.py': 'bf10bb17b3a26872591407f634f1948a492badf3731e2ad4e92312bb828584ff',
}
FROZEN_09A5469_ALGORITHM_HASH = '54b714207e175169a0971a4d7b7f0820cb69f5fa8ac50d799c2eef397db7dd8f'
FROZEN_09A5469_IMPLEMENTATION_HASH = '60da3e0c57fee2fbd9cd7ce756c66c56ba5642a63da27ef428b9b30538a459df'
CAPS = {'standard': 2048, 'long': 24576, 'smoke': 32}
LIMITATIONS = [
    'Artifact consistency and recorded measurements only; no model/loss recomputation.',
    'Initial zero deltas, no gradient buffering, and tensor snapshot immutability remain source/test-supported, not directly observed by final JSON flags.',
    'No independent proof of GPU isolation or device clock stability from these artifacts.',
    'Smoke uses synthetic fixtures and does not complete reference M4.',
]


def _valid_fastpath_dict(fp):
    req = {'compile', 'cuda_graph', 'compile_mode'}
    opt = {'bf16_gemm', 'compile_drafter', 'compile_online', 'analytic_online_grad'}
    return (isinstance(fp, dict)
            and req <= set(fp.keys()) <= (req | opt)
            and type(fp['compile']) is bool
            and type(fp['cuda_graph']) is bool
            and type(fp['compile_mode']) is str
            and all(type(fp[k]) is bool for k in opt if k in fp))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def candidates(arm):
    if arm == 'plain_dflash':
        return [{}]
    if arm == 'online_bigram':
        return [{'lambda_tr': .05, 'alpha': .5, 'k_ret': None},
                {'lambda_tr': .2, 'alpha': 1., 'k_ret': None}]
    if arm == 'offline_static':
        return [{'beta': .5}, {'beta': 1.}]
    return [{'beta': 1., 'lr': lr, 'update_stride': 1, 'lambda_l2': l2,
             'lambda_l1_b': 1e-6, 'lambda_l1_s': 1e-5, 'weighting_mode': 'exponential'}
            for lr, l2 in ((1e-4, .01), (1e-3, .1))]


def conditions(condition, primary=False):
    if condition == 'long_thinking':
        return [('thinking_recommended', .6, {'top_k': 20, 'top_p': .95, 'min_p': 0.})]
    if condition != 'standard_nonthinking':
        raise ValueError('this audit requires the inspected, supported long_thinking reference condition')
    recommended = ('nonthinking_recommended', .7, {'top_k': 20, 'top_p': .8, 'min_p': 0.})
    unwarped = {'top_k': 0, 'top_p': 1., 'min_p': 0.}
    return [recommended] if primary else [('greedy', 0., unwarped), recommended,
                                         ('pure_temperature_diagnostic', 1., unwarped)]


def smoke_manifest():
    return {'scale': 'smoke', 'prompts': [
        {'prompt_id': f'synthetic-{split}-{condition}-{i}', 'split': split,
         'condition': condition, 'text': f'Synthetic {i}', 'dataset': 'builtin',
         'reference_selected': True}
        for split in ('dev', 'test')
        for condition in ('standard_nonthinking', 'long_thinking') for i in range(2)]}


def _snake_rank(index, world_size):
    band, offset = divmod(int(index), int(world_size))
    return offset if (band % 2 == 0) else (int(world_size) - 1 - offset)


class Audit:
    def __init__(self, args):
        self.args = args
        self.root = args.artifacts.resolve()
        self.stage = self.root / 'm4'
        self.checks = 0
        self.failures = []
        self.counts = Counter()
        self.files = {}
        self.entry_ranks = {}
        self.inputs = {}
        self.offline_norms = None
        self.dev_rows = defaultdict(list)
        self.eval_token_sums = defaultdict(int)
        self.eval_long_rep = defaultdict(bool)
        self.timed_speed_rows = []
        self.cal_speed_rows = []
        for path in Path(__file__).parent.glob('*.py'):
            self.files[str(path.resolve())] = digest(path)

    def require(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def artifact(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        value = hashlib.sha256(data).hexdigest()
        self.require(str(path) not in self.files or self.files[str(path)] == value,
                     f'artifact changed during audit: {path}')
        self.files[str(path)] = value
        return json.loads(data)

    def _hash_tree(self, repo_root, allow_missing_opt_files=False):
        impl_h = hashlib.sha256()
        alg_h = hashlib.sha256()
        alg_files = {}
        optional_opt_files = {'fastpath.py', 'optimizers.py'}
        for name in IMPLEMENTATION_FILES:
            path = (Path(repo_root) / 'src/edraft' / name).resolve()
            if allow_missing_opt_files and name in optional_opt_files and not path.exists():
                continue
            data = path.read_bytes()
            file_h = hashlib.sha256(data).hexdigest()
            self.files[str(path)] = file_h
            impl_h.update(name.encode())
            impl_h.update(data)
            if name in ALGORITHM_FILES:
                alg_h.update(name.encode())
                alg_h.update(data)
                alg_files[name] = file_h
        return impl_h.hexdigest(), alg_h.hexdigest(), alg_files

    def prepare(self):
        self.require(self.args.scale in ('smoke', 'reference'), 'full scale is outside this mandatory reference-plan audit')
        self.settings = self.artifact(self.stage / 'settings.json')
        s = self.settings
        self.require(s.get('frozen') is True and s.get('stage') == 'm4' and s.get('scale') == self.args.scale,
                     'frozen M4 settings stage/scale mismatch')
        self.require(s['settings_seed'] == self.args.seed, 'settings seed mismatch')
        self.require(s['generation_caps'] == CAPS, 'generation caps differ from reviewed continuation plan')
        self.require(set(s['selected']) == set(ARMS), 'missing or additional M4 reference arm')
        for arm in ARMS:
            self.require([r['settings'] for r in s['candidates'][arm]] == candidates(arm), f'{arm}: tuning grid differs')
            self.require(s['selected'][arm] in candidates(arm), f'{arm}: selected setting was not a development candidate')
        self.frozen = fingerprint(s)
        self.manifest = smoke_manifest() if self.args.scale == 'smoke' else self.artifact(self.root / 'prompts.json')
        self.require(fingerprint(self.manifest) == s['prompt_manifest_hash'], 'prompt manifest changed since tuning')
        prompts = self.manifest['prompts']
        self.by_id = {p['prompt_id']: p for p in prompts}
        self.require(len(self.by_id) == len(prompts), 'prompt IDs overlap across splits')
        self.prompts = {}
        for split in ('dev', 'test'):
            rows = [p for p in prompts if p['split'] == split]
            standard = [p for p in rows if p['condition'] == 'standard_nonthinking'
                        and (split == 'dev' or p.get('reference_selected', True))]
            long = [p for p in rows if p['condition'].startswith('long_') and p.get('reference_selected', True)]
            ns, nl = (2, 2) if self.args.scale == 'smoke' else ((8, 12) if split == 'dev' else (64, 32))
            self.require(len(standard) >= ns and len(long) >= nl, f'insufficient {split} prompts')
            self.require({p['condition'] for p in long} == {'long_thinking'}, 'authorized long condition changed')
            self.prompts[split] = standard[:ns] + long[:nl]
        self.require(s['development_prompt_ids'] == [p['prompt_id'] for p in self.prompts['dev']], 'development selection differs')
        self.repo_impl_hash, self.repo_alg_hash, self.repo_alg_files = self._hash_tree(self.args.repo, allow_missing_opt_files=False)
        src_repo_arg = getattr(self.args, 'source_repo', None)
        source_repo = (src_repo_arg or self.args.repo).resolve()
        self.require((source_repo / 'src/edraft/eval_closed_loop.py').is_file(), f'invalid source_repo: {source_repo}')
        self.src_impl_hash, self.src_alg_hash, self.src_alg_files = self._hash_tree(
            source_repo, allow_missing_opt_files=(src_repo_arg is not None)
        )
        self.valid_harness_hashes = {self.repo_impl_hash, self.src_impl_hash}
        self.valid_alg_hashes = {self.repo_alg_hash, self.src_alg_hash}
        recorded_impl = s['checkpoint_hashes']['implementation']
        exact_source_match = (self.src_impl_hash == recorded_impl)
        frozen_alg_match = (
            recorded_impl == FROZEN_09A5469_IMPLEMENTATION_HASH
            and self.src_alg_hash == FROZEN_09A5469_ALGORITHM_HASH
            and self.src_alg_files == FROZEN_09A5469_ALGORITHM_FILES
            and (
                src_repo_arg is not None
                or (
                    self.repo_alg_hash == FROZEN_09A5469_ALGORITHM_HASH
                    and self.repo_alg_files == FROZEN_09A5469_ALGORITHM_FILES
                )
            )
        )
        self.require(
            exact_source_match or frozen_alg_match,
            'current evaluator source differs from frozen source and 10/10 09a5469 algorithm lock',
        )
        if self.args.scale == 'reference':
            environment = self.artifact(self.root / 'environment.json')
            recorded_models = environment['models']
            self.require(s['model_pair'] == [recorded_models['target'], recorded_models['drafter']] and
                         s['model_revisions'] == [recorded_models['revisions']['target'], recorded_models['revisions']['drafter']],
                         'model revisions differ from recorded reference environment')
            for key, rel in [('dense', 'm2/dense.pt'), ('dense_sparse', 'm2/dense_sparse.pt'), ('projection', 'h_proj.pt')]:
                path = self.root / rel
                self.files[str(path)] = digest(path)
                self.require(self.files[str(path)] == s['checkpoint_hashes'][key], f'{key} checkpoint mismatch')
            self.require(s['model_pair'] == ['Qwen/Qwen3-8B', 'z-lab/Qwen3-8B-DFlash-b16'], 'reference model pair changed')

    def identity(self, prompt, arm, setting, sample, seed, mode, session=None, repeat=None, profile=False):
        name, temperature, warper = sample
        s = self.settings
        return {'model_pair': s['model_pair'], 'prompt': prompt['prompt_id'], 'scale': self.args.scale,
            'prompt_payload_hash': fingerprint(prompt), 'condition': prompt['condition'], 'seed': seed,
            'temperature': temperature, 'sampling_name': name, 'verification_rule': 'ratio' if temperature > 0 else 'greedy',
            'proposal_mode': arm, 'checkpoint': s['checkpoint_hashes'], 'adapter_version': 0,
            'model_revisions': s['model_revisions'], 'stage': 'm4', 'mode': mode,
            'sampling_warp_config': warper, 'setting': setting, 'frozen_settings_hash': '' if mode == 'tune' else self.frozen,
            'generation_cap': 32 if self.args.scale == 'smoke' else 24576 if prompt['condition'].startswith('long_') else 2048,
            'generation_caps': CAPS, 'timing_session': session, 'repeat': repeat, 'phase_profile': profile}

    def manifests(self, mode):
        entries = {}
        paths = sorted((self.stage / mode).glob('manifest-rank*.json'))
        self.require(bool(paths), f'{mode}: no raw manifests')
        for path in paths:
            rank = int(path.stem.removeprefix('manifest-rank'))
            for key, entry in self.artifact(path)['jobs'].items():
                self.require(key not in entries, f'{mode}: duplicate identity across rank manifests: {key}')
                entries[key] = entry
                self.entry_ranks[(mode, key)] = rank
        return entries

    def row(self, identity, entries, category, expected_path=None, expected_rank=None,
            expected_harness_sha=None, expected_affinity=None):
        key = fingerprint(identity)
        self.require(key in entries, f'{category}: missing expected request {key}')
        entry = entries[key]
        self.require(entry['status'] == 'complete' and fingerprint(entry['identity']) == key, f'{key}: manifest identity/status')
        path = Path(entry['result_path']).resolve()
        self.require(path.stem == key and path.parent.parent == (self.stage / identity['mode']).resolve(), f'{key}: raw path mismatch')
        if expected_path is not None:
            self.require(path == Path(expected_path).resolve(), f'{key}: bundle path mismatch')
        row = self.artifact(path)
        self.require(self.files[str(path)] == entry.get('result_sha256'), f'{key}: checksum missing or changed')
        self.require(fingerprint(row['identity']) == key, f'{key}: raw identity mismatch')
        manifest_rank = self.entry_ranks[(identity['mode'], key)]
        if expected_rank is not None:
            rank = expected_rank
        elif identity['mode'] == 'evaluate' and self.args.scale == 'reference':
            rank = manifest_rank
        else:
            rank = 0
        self.require(type(row['rank']) is int and 0 <= row['rank'] < 64 and row['rank'] == rank
                     and path.parent.name == f'rank{rank}' and manifest_rank == rank,
                     f'{key}: rank/path/manifest mismatch')
        if expected_harness_sha is not None:
            self.require(row.get('timing_harness_implementation_sha256') == expected_harness_sha,
                         f'{key}: timing_harness_implementation_sha256 mismatch')
            self.require(row.get('algorithm_implementation_sha256') in self.valid_alg_hashes,
                         f'{key}: algorithm_implementation_sha256 mismatch')
        if expected_affinity is not None:
            self.require(
                row.get('physical_gpu') == expected_affinity['physical_gpu']
                and row.get('numa_node') == expected_affinity['numa_node']
                and row.get('cpu_affinity') == expected_affinity['cpu_affinity'],
                f'{key}: row GPU/NUMA/CPU affinity mismatch',
            )
        fields = {'scale': 'scale', 'stage': 'stage', 'arm': 'proposal_mode', 'prompt_id': 'prompt',
            'condition': 'condition', 'seed': 'seed', 'temperature': 'temperature', 'sampling_name': 'sampling_name',
            'sampling_warp_config': 'sampling_warp_config', 'settings': 'setting', 'verification_rule': 'verification_rule',
            'model_pair': 'model_pair', 'model_revisions': 'model_revisions', 'checkpoint_hashes': 'checkpoint',
            'phase_profile': 'phase_profile', 'timing_session': 'timing_session', 'repeat': 'repeat'}
        for field, source in fields.items():
            self.require(fingerprint(row.get(field)) == fingerprint(identity[source]), f'{key}: row {field} mismatch')
        self.require(row['split'] == ('dev' if category == 'tune' else 'test'), f'{key}: split mismatch')
        self.require(row['label'] == ('synthetic' if self.args.scale == 'smoke' else 'reference'), f'{key}: label mismatch')
        self.require(row['generation'].get('complete') is True, f'{key}: generation incomplete')
        self.require(row['timing_isolated'] is (identity['mode'] in ('timing', 'timing_calibration')), f'{key}: timing isolation label')
        self.require(type(row['generated_tokens']) is int and row['generated_tokens'] == len(row['output_tokens']), f'{key}: output count')
        self.require(row['effective_beta'] == (1. if identity['temperature'] == 0 else identity['setting'].get('beta', 1.)), f'{key}: effective beta')
        meta = row['input_metadata']
        self.require(meta['generation_environment_sha256'] == fingerprint(self.settings['generation_environment']), f'{key}: environment binding')
        self.require(type(meta['input_token_count']) is int and meta['input_token_count'] > 0 and
                     meta['enable_thinking'] is (identity['condition'] == 'long_thinking'), f'{key}: input count/thinking flag')
        input_hash = meta['input_token_ids_sha256']
        self.require(isinstance(input_hash, str) and len(input_hash) == 64 and
                     all(c in '0123456789abcdef' for c in input_hash), f'{key}: malformed input-token hash')
        cap = identity['generation_cap']
        context = self.settings['generation_environment']['context_limit']
        effective = cap if context is None else min(cap, context - meta['input_token_count'])
        self.require(meta['requested_generation_cap'] == cap and meta['effective_generation_cap'] == effective and effective >= 1,
                     f'{key}: input/context generation cap')
        self.require(1 <= row['generated_tokens'] <= effective, f'{key}: emitted count outside valid range')
        tokens = row['output_tokens']
        self.require(all(type(token) is int and token >= 0 for token in tokens), f'{key}: output token type/range')
        stops = set(self.settings['generation_environment']['stop_token_ids'])
        eos = tokens[-1] in stops
        self.require(row['generation'].get('capped') is (len(tokens) == effective) and
                     row['generation'].get('eos_reached') is eos and (eos or len(tokens) == effective),
                     f'{key}: termination/cap flags contradict output')
        self.require(not any(token in stops for token in tokens[:-1]), f'{key}: tokens after EOS')
        prompt = identity['prompt']
        self.require(prompt not in self.inputs or self.inputs[prompt] == meta, f'{key}: input metadata differs across requests')
        self.inputs[prompt] = meta
        reuse = row.get('baseline_reuse', {})
        reused = reuse.get('status') == 'reused'
        self.require(not reused or (identity['mode'] == 'evaluate' and identity['proposal_mode'] in ARMS[:2]), f'{key}: forbidden reuse')
        if reused:
            sources = {}
            for label in ('result', 'manifest', 'settings'):
                source_path = Path(reuse[f'source_{label}_path']).resolve()
                sources[label] = self.artifact(source_path)
                self.require(self.files[str(source_path)] == reuse[f'source_{label}_sha256'], f'{key}: reused M3 {label} changed')
            source, source_settings = sources['result'], sources['settings']
            src_rank = source.get('rank')
            self.require(type(src_rank) is int and 0 <= src_rank < 64, f'{key}: invalid M3 source rank')
            expected_sources = {'settings': self.root / 'm3/settings.json',
                'manifest': self.root / f'm3/evaluate/manifest-rank{src_rank}.json'}
            for label, expected_source in expected_sources.items():
                self.require(Path(reuse[f'source_{label}_path']).resolve() == expected_source.resolve(), f'{key}: M3 source {label} path')
            source_identity = source['identity']
            source_key = fingerprint(source_identity)
            self.require(Path(reuse['source_result_path']).resolve() == (self.root / f'm3/evaluate/rank{src_rank}/{source_key}.json').resolve(),
                         f'{key}: M3 raw source path')
            self.require(source_identity['stage'] == 'm3' and source['stage'] == 'm3' and
                         source_identity['frozen_settings_hash'] == fingerprint(source_settings) == reuse['source_frozen_settings_hash']
                         and source_key == reuse['source_identity_hash'], f'{key}: source stage/frozen identity binding')
            source_entry = sources['manifest']['jobs'][source_key]
            self.require(source_entry['identity'] == source_identity and source_entry['status'] == 'complete' and
                         Path(source_entry['result_path']).resolve() == Path(reuse['source_result_path']).resolve() and
                         source_entry['result_sha256'] == reuse['source_result_sha256'], f'{key}: source manifest binding')
            self.require(source_settings['frozen'] is True and source_settings['stage'] == 'm3' and
                         source_settings['scale'] == self.args.scale and
                         source_settings['selected'][identity['proposal_mode']] == identity['setting'], f'{key}: source settings binding')
            for field in ('model_pair', 'model_revisions', 'generation_caps', 'generation_environment', 'checkpoint_hashes'):
                self.require(source_settings[field] == self.settings[field], f'{key}: source settings {field}')
            self.require({k: v for k, v in source['identity'].items() if k not in ('stage', 'frozen_settings_hash')} ==
                         {k: v for k, v in identity.items() if k not in ('stage', 'frozen_settings_hash')}, f'{key}: reuse generation identity')
            for field in ('output_tokens', 'blocks', 'generation', 'input_metadata', 'generated_tokens',
                          'mean_accepted_length', 'median_accepted_length', 'target_forwards_per_generated_token', 'bigram'):
                self.require(row[field] == source[field], f'{key}: reused {field} changed')
            for field in ('elapsed_seconds', 'generation_loop_elapsed_seconds', 'tokens_per_second', 'phase_ms',
                          'estimated_online_state_bandwidth_GBps', 'peak_allocated_bytes', 'peak_reserved_bytes'):
                self.require(row[field] is None, f'{key}: reused performance field retained')
        else:
            seconds, speed = row['elapsed_seconds'], row['tokens_per_second']
            self.require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0, f'{key}: request time')
            self.require(type(speed) in (int, float) and math.isfinite(speed) and
                         math.isclose(speed, row['generated_tokens'] / seconds, rel_tol=1e-10, abs_tol=1e-12), f'{key}: throughput algebra')
            allocated, reserved = row['peak_allocated_bytes'], row['peak_reserved_bytes']
            self.require((type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved)
                         if self.args.scale == 'reference' else allocated is None and reserved is None,
                         f'{key}: peak memory measurements')
        if identity['phase_profile']:
            phases = row['phase_ms']
            self.require(isinstance(phases, dict) and all(type(v) in (float, int) and math.isfinite(v) and v >= 0
                         for v in phases.values()), f'{key}: phase timing values')
            required_phases = ['dflash_draft', 'sequential_sampling', 'target_verification'] if row['blocks'] else []
            if row['blocks'] and identity['proposal_mode'] not in ARMS[:2]:
                required_phases.extend(['dense_adapter_score', 'sparse_lookup'])
            if row['blocks'] and identity['proposal_mode'] in ('online_bigram', *ONLINE):
                required_phases.append('online_update')
            if self.args.scale == 'reference':
                required_phases.append('target_feature_extraction')
            for phase in required_phases:
                value = phases.get(phase)
                self.require(type(value) in (float, int) and math.isfinite(value) and value > 0,
                             f'{key}: required measured phase {phase}')
        self.counts[category] += 1
        if category == 'tune':
            self.dev_rows[(identity['proposal_mode'], fingerprint(identity['setting']))].append({
                k: row[k] for k in ('prompt_id', 'condition', 'generation', 'tokens_per_second', 'generated_tokens', 'elapsed_seconds')})
        elif category == 'evaluate':
            self.eval_token_sums[identity['prompt']] += row['generated_tokens']
            if identity['condition'].startswith('long_') and row['generation'].get('repetition_dominated'):
                self.eval_long_rep[identity['prompt']] = True
        elif category == 'timing':
            self.timed_speed_rows.append({
                k: row[k] for k in ('prompt_id', 'condition', 'temperature', 'sampling_warp_config',
                                    'seed', 'repeat', 'arm', 'tokens_per_second', 'generation')
            })
        elif category == 'timing_calibration':
            self.cal_speed_rows.append({
                k: row[k] for k in ('prompt_id', 'condition', 'temperature', 'sampling_warp_config',
                                    'seed', 'repeat', 'arm', 'tokens_per_second', 'generation')
            })
        if identity['proposal_mode'] in ONLINE:
            result = check_online_row(row, expected_scale=self.args.scale, expected_arm=identity['proposal_mode'],
                                      expected_setting=identity['setting'], profile=identity['phase_profile'])
            self.checks += result['checks']
            self.failures.extend(f'{key}: {message}' for message in result['failures'])
            self.counts[category + '_online'] += 1
            self.counts['online_blocks'] += result['block_count'] or 0
            self.counts['online_updates'] += result['update_count'] or 0
            norms = row['parameter_norms']
            self.require(self.offline_norms is None or norms == self.offline_norms, f'{key}: offline norms differ across online requests')
            self.offline_norms = norms
            for block in row['blocks']:
                self.require(block['collection_temperature'] == identity['temperature'] and
                             block['verification_rule'] == identity['verification_rule'] and
                             block['sampling_warp_config'] == identity['sampling_warp_config'], f'{key}: block sampling settings')
        return key

    def check_selection(self):
        plain = {r['prompt_id']: r for r in self.dev_rows[('plain_dflash', fingerprint({}))]}
        for arm in ARMS:
            calculated = []
            for setting, reported in zip(candidates(arm), self.settings['candidates'][arm]):
                rows = self.dev_rows[(arm, fingerprint(setting))]
                grouped = defaultdict(list)
                excluded = 0
                for row in rows:
                    baseline = plain[row['prompt_id']]
                    if row['condition'].startswith('long_') and (row['generation']['repetition_dominated'] or baseline['generation']['repetition_dominated']):
                        excluded += 1
                        continue
                    grouped[row['condition']].append(row['tokens_per_second'] / baseline['tokens_per_second'])
                scores = {c: statistics.mean(values) for c, values in grouped.items()}
                score = statistics.mean(scores.values())
                self.require(math.isclose(score, reported['selection_score'], rel_tol=1e-12, abs_tol=1e-12), f'{arm}: dev selection score')
                self.require(set(scores) == set(reported['condition_normalized_throughput']) and all(
                    math.isclose(v, reported['condition_normalized_throughput'][c], rel_tol=1e-12, abs_tol=1e-12)
                    for c, v in scores.items()), f'{arm}: dev condition scores')
                self.require(reported['dev_requests'] == len(rows) and reported['repetition_excluded_matched_pairs'] == excluded
                    and reported['generated_tokens'] == sum(r['generated_tokens'] for r in rows), f'{arm}: dev coverage totals')
                self.require(math.isclose(reported['wall_seconds'], sum(r['elapsed_seconds'] for r in rows), rel_tol=1e-12), f'{arm}: dev wall total')
                calculated.append((score, setting))
            winner = max(calculated, key=lambda item: item[0])[1]
            self.require(fingerprint(winner) == fingerprint(self.settings['selected'][arm]), f'{arm}: selected candidate differs from dev-only winner')

    def session(self, mode, logical):
        found = []
        for path in (self.stage / mode / 'sessions').glob('*.json'):
            value = read(path)
            if value.get('logical_session') == logical:
                found.append((path, value))
        self.require(len(found) == 1, f'{mode}: expected exactly one selected logical session')
        path, session = found[0]
        session = self.artifact(path)
        config = session['configuration']
        self.require(path.stem == fingerprint(config), f'{mode}: configuration-hash filename mismatch')
        for key, value in {'checkpoint_hashes': self.settings['checkpoint_hashes'], 'model_pair': self.settings['model_pair'],
            'model_revisions': self.settings['model_revisions'], 'generation_environment': self.settings['generation_environment'],
            'generation_caps': CAPS, 'stage': 'm4', 'scale': self.args.scale, 'seed': self.args.seed}.items():
            self.require(config.get(key) == value, f'{mode}: session configuration {key}')
        if self.args.scale == 'reference':
            vis = config.get('visible_devices')
            self.require(config['device'] == 'cuda:0' and isinstance(vis, str) and vis.strip().isdigit(),
                         f'{mode}: selected physical device differs from isolated single-GPU plan')
        return path, session

    def bundle(self, mode, session, entries, comparisons, repeat, profile,
               expected_rank=None, expected_harness_sha=None, expected_affinity=None):
        key = fingerprint({'jobs': [(p, a, sample[0], seed, setting) for p, a, setting, sample, seed in comparisons],
                           'repeat': repeat, 'profile': profile})
        b = session['bundles'].get(key)
        self.require(b is not None and b['status'] == 'complete', f'{mode}: missing completed all-arm bundle {key}')
        self.require(b['epoch'] in {e['epoch'] for e in session['epochs']}, f'{mode}: unknown bundle epoch')
        self.require(len(b['result_paths']) == len(comparisons) and len(set(b['result_paths'])) == len(comparisons), f'{mode}: bundle size/duplicate')
        category = 'profile' if profile else mode
        for (p, arm, setting, sample, seed), path in zip(comparisons, b['result_paths']):
            identity = self.identity(p, arm, setting, sample, seed, mode, session['logical_session'],
                                     f'{repeat}:epoch={b["epoch"]}', profile)
            self.row(identity, entries, category, path, expected_rank=expected_rank,
                     expected_harness_sha=expected_harness_sha, expected_affinity=expected_affinity)
        return key

    def _expected_sharding(self, timing_prompts, world_size):
        manifest_order = {p['prompt_id']: idx for idx, p in enumerate(self.manifest['prompts'])}
        conds = sorted({p['condition'] for p in timing_prompts})
        ordered_by_cond = {
            cond: sorted(
                [p for p in timing_prompts if p['condition'] == cond],
                key=lambda p: (-self.eval_token_sums[p['prompt_id']], manifest_order[p['prompt_id']], p['prompt_id']),
            )
            for cond in conds
        }
        prompt_to_rank = {}
        if all(len(v) >= world_size for v in ordered_by_cond.values()):
            for cond in conds:
                for k, p in enumerate(ordered_by_cond[cond]):
                    prompt_to_rank[p['prompt_id']] = _snake_rank(k, world_size)
        else:
            flat = [p for cond in conds for p in ordered_by_cond[cond]]
            for k, p in enumerate(flat):
                prompt_to_rank[p['prompt_id']] = _snake_rank(k, world_size)
        ranks = {str(r): [p['prompt_id'] for p in timing_prompts if prompt_to_rank[p['prompt_id']] == r]
                 for r in range(world_size)}
        return prompt_to_rank, ranks

    def _audit_calibration(self):
        cal_path = self.stage / 'timing_isolation_calibration.json'
        self.require(cal_path.is_file(), 'sharded timing requires m4/timing_isolation_calibration.json')
        cal = self.artifact(cal_path)
        self.require(
            cal.get('status') == 'complete'
            and cal.get('reconciled_with_sharded_timing') is True
            and cal.get('scale') == self.args.scale
            and cal.get('stage') == 'm4'
            and cal.get('frozen_settings_hash') == self.frozen,
            'timing_isolation_calibration status/binding mismatch',
        )
        self.require(
            cal.get('timing_harness_implementation_sha256') in self.valid_harness_hashes
            and cal.get('algorithm_implementation_sha256') in self.valid_alg_hashes,
            'timing_isolation_calibration provenance hash mismatch',
        )
        repeats = cal.get('repeats')
        warmups = cal.get('warmups_per_arm_condition')
        self.require(repeats == 2 and warmups == 1, 'timing_isolation_calibration repeats/warmups differ from 2/1')

        standard = [p for p in self.prompts['test'] if p['condition'] == 'standard_nonthinking'][:min(4, 2 if self.args.scale == 'smoke' else 4)]
        long_all = [p for p in self.prompts['test'] if p['condition'].startswith('long_')]
        n_long = min(2, len(long_all))
        non_rep = [(self.eval_token_sums[p['prompt_id']], idx, p)
                   for idx, p in enumerate(long_all) if not self.eval_long_rep[p['prompt_id']]]
        non_rep.sort(key=lambda item: (item[0], item[1], item[2]['prompt_id']))
        selected_long = [item[2] for item in non_rep[:n_long]]
        if len(selected_long) < n_long:
            self.require(self.args.scale == 'smoke', 'reference calibration requires >=2 non-rep-dominated long_thinking prompts')
            for idx, p in enumerate(long_all):
                if len(selected_long) >= n_long:
                    break
                if p['prompt_id'] not in {x['prompt_id'] for x in selected_long}:
                    selected_long.append(p)
        expected_cal_prompts = sorted(standard + selected_long, key=lambda p: p['condition'])
        expected_ids = [p['prompt_id'] for p in expected_cal_prompts]
        self.require(cal.get('selected_prompt_ids') == expected_ids, 'timing_isolation_calibration prompt selection differs')

        path, session = self.session('timing_calibration', cal['calibration_session'])
        self.require(
            session['configuration']['configuration'] == {
                'frozen_settings_hash': self.frozen,
                'prompt_manifest_hash': fingerprint(self.manifest),
                'selected_prompts': expected_ids,
                'repeats': 2,
                'warmups': 1,
            },
            'timing_calibration session configuration mismatch',
        )
        affinity = {
            'physical_gpu': session.get('physical_gpu'),
            'numa_node': session.get('numa_node'),
            'cpu_affinity': session.get('cpu_affinity'),
        }
        self.require(affinity['physical_gpu'] == 0 and affinity['numa_node'] == 0 and bool(affinity['cpu_affinity']),
                     'timing_calibration must bind to physical_gpu=0 and numa_node=0')
        if len(affinity['cpu_affinity']) == 16:
            self.require(affinity['cpu_affinity'] == list(range(0, 16)), 'timing_calibration cpu_affinity must be [0..15]')
        if self.args.scale == 'reference':
            self.require(
                cal.get('all_gpus_idle_verified') is True
                and session.get('cpu_affinity_applied') is True
                and affinity['cpu_affinity'] == list(range(0, 16)),
                'reference timing_calibration lacks verified cluster-idle or applied [0..15] affinity',
            )
        evidence = cal['raw_evidence']
        self.require(Path(evidence['session_path']).resolve() == path.resolve() and digest(path) == evidence['session_sha256'],
                     'timing_calibration session path/checksum mismatch')
        manifest_path = self.stage / 'timing_calibration/manifest-rank0.json'
        self.require(Path(evidence['manifest_path']).resolve() == manifest_path.resolve() and digest(manifest_path) == evidence['manifest_sha256'],
                     'timing_calibration manifest path/checksum mismatch')
        entries = self.manifests('timing_calibration')
        expected_bundles = set()
        cal_expected_harness = cal.get('timing_harness_implementation_sha256')
        for i, p in enumerate(expected_cal_prompts):
            sample = conditions(p['condition'], primary=True)[0]
            pairs = [(p, a, self.settings['selected'][a], sample, self.args.seed) for a in ARMS]
            for repeat in range(2):
                offset = (i + repeat) % len(pairs)
                expected_bundles.add(
                    self.bundle('timing_calibration', session, entries, pairs[offset:] + pairs[:offset],
                                repeat, False, expected_rank=0, expected_harness_sha=cal_expected_harness,
                                expected_affinity=affinity)
                )
        self.require(set(session['bundles']) == expected_bundles, 'timing_calibration bundle population differs')
        expected_runs = len(expected_cal_prompts) * 2 * len(ARMS)
        self.require(
            self.counts['timing_calibration'] == expected_runs
            and self.counts['timing_calibration_online'] == expected_runs * 2 // 5
            and cal.get('calibrated_timed_runs') == expected_runs
            and evidence.get('timing_requests') == expected_runs
            and evidence.get('bundles') == len(expected_bundles),
            'timing_calibration row/bundle count mismatch',
        )

        # Independently recompute 1-GPU isolated vs sharded timing comparisons on matched prompt slots.
        cal_slots = {
            (r['condition'], float(r['temperature']), json.dumps(r['sampling_warp_config'], sort_keys=True),
             r['prompt_id'], r['seed'])
            for r in self.cal_speed_rows
        }
        sharded_sub = [
            r for r in self.timed_speed_rows
            if (r['condition'], float(r['temperature']), json.dumps(r['sampling_warp_config'], sort_keys=True),
                r['prompt_id'], r['seed']) in cal_slots
        ]

        def _prompt_mean_tps(subset):
            by_p = defaultdict(list)
            for rw in subset:
                by_p[rw['prompt_id']].append(float(rw['tokens_per_second']))
            return statistics.mean(statistics.mean(vs) for vs in by_p.values())

        def _prompt_mean_paired_ratio(subset, target_arm):
            plain_map = {
                (rw['prompt_id'], rw['seed'], str(rw['repeat']).split(':epoch=')[0]): float(rw['tokens_per_second'])
                for rw in subset if rw['arm'] == 'plain_dflash'
            }
            by_p = defaultdict(list)
            for rw in subset:
                if rw['arm'] != target_arm:
                    continue
                pk = (rw['prompt_id'], rw['seed'], str(rw['repeat']).split(':epoch=')[0])
                by_p[rw['prompt_id']].append(float(rw['tokens_per_second']) / plain_map[pk])
            return statistics.mean(statistics.mean(vs) for vs in by_p.values())

        rec_groups = {(g['condition'], g['arm']): g for g in cal.get('groups', [])}
        cond_keys = sorted({
            (r['condition'], float(r['temperature']), json.dumps(r['sampling_warp_config'], sort_keys=True))
            for r in self.cal_speed_rows
        })
        self.require(len(rec_groups) == len(cond_keys) * len(ARMS), 'timing_isolation_calibration group count mismatch')
        for cond, temp, warp_json in cond_keys:
            c_1 = [r for r in self.cal_speed_rows if r['condition'] == cond and float(r['temperature']) == temp
                   and json.dumps(r['sampling_warp_config'], sort_keys=True) == warp_json]
            c_8 = [r for r in sharded_sub if r['condition'] == cond and float(r['temperature']) == temp
                   and json.dumps(r['sampling_warp_config'], sort_keys=True) == warp_json]
            for arm in ARMS:
                g = rec_groups.get((cond, arm))
                self.require(g is not None, f'missing calibration group ({cond}, {arm})')
                a_1 = [r for r in c_1 if r['arm'] == arm]
                a_8 = [r for r in c_8 if r['arm'] == arm]
                tps_1 = _prompt_mean_tps(a_1)
                tps_8 = _prompt_mean_tps(a_8)
                pr_1 = _prompt_mean_paired_ratio(c_1, arm)
                pr_8 = _prompt_mean_paired_ratio(c_8, arm)
                self.require(
                    math.isclose(g['tps_1gpu_mean'], tps_1, rel_tol=1e-9, abs_tol=1e-9)
                    and math.isclose(g['tps_8gpu_mean'], tps_8, rel_tol=1e-9, abs_tol=1e-9)
                    and math.isclose(g['tps_8gpu_over_1gpu_ratio'], tps_8 / tps_1, rel_tol=1e-9, abs_tol=1e-9)
                    and math.isclose(g['paired_ratio_vs_plain_1gpu'], pr_1, rel_tol=1e-9, abs_tol=1e-9)
                    and math.isclose(g['paired_ratio_vs_plain_8gpu'], pr_8, rel_tol=1e-9, abs_tol=1e-9)
                    and math.isclose(g['paired_ratio_diff_8gpu_minus_1gpu'], pr_8 - pr_1, rel_tol=1e-9, abs_tol=1e-9),
                    f'calibration group ({cond}, {arm}) algebra drift from raw rows',
                )

    def run(self):
        self.prepare()
        path, session = self.session('tune', self.settings['comparison_session'])
        config = dict(session['configuration']['configuration'])
        tune_fp = config.pop('fastpath', None)
        self.require(tune_fp is None or _valid_fastpath_dict(tune_fp), 'invalid tune fastpath metadata')
        self.require(config == {'prompt_manifest_hash': fingerprint(self.manifest),
            'dev_ids': self.settings['development_prompt_ids'], 'candidates': {a: candidates(a) for a in ARMS}}, 'tune configuration mismatch')
        entries = self.manifests('tune')
        expected = set()
        for i, p in enumerate(self.prompts['dev']):
            sample = conditions(p['condition'], primary=True)[0]
            pairs = [(p, a, setting, sample, self.args.seed) for a in ARMS for setting in candidates(a)]
            offset = i % len(pairs)
            expected.add(self.bundle('tune', session, entries, pairs[offset:] + pairs[:offset], 'dev', False))
        self.require(set(session['bundles']) == expected, 'tune bundle population differs')
        self.check_selection()
        entries = self.manifests('evaluate')
        expected = set()
        for p, sample, seed in self.test_cases(self.prompts['test']):
            for arm in ARMS:
                expected.add(self.row(self.identity(p, arm, self.settings['selected'][arm], sample, seed, 'evaluate'), entries, 'evaluate'))
        selected = {k for k, e in entries.items() if e['identity'].get('frozen_settings_hash') == self.frozen}
        self.require(selected == expected, 'evaluation population differs for frozen settings')

        report = self.artifact(self.stage / 'timing.json')
        self.require(report.get('status') == 'complete' and report['repeats'] == 3 and report['warmups_per_arm_condition'] == 1,
                     'timing report incomplete or repetition/warmup plan changed')
        self.require(report['scale'] == self.args.scale and report['stage'] == 'm4' and report['frozen_settings_hash'] == self.frozen,
                     'timing report binding mismatch')
        timing_prompts = sorted(self.prompts['test'], key=lambda p: p['condition'])
        ids = [p['prompt_id'] for p in timing_prompts]
        self.require(report['prompt_ids'] == ids and report['expected_full_prompt_count'] == len(ids) and
                     report['full_reference_prompt_coverage'] is True, 'timing report prompt population')
        report_fp = report.get('fastpath')
        self.require(report_fp is None or _valid_fastpath_dict(report_fp), 'invalid timing fastpath metadata')
        evidence = report['raw_evidence']
        world_size = report.get('world_size', 1)
        self.require(type(world_size) is int and 1 <= world_size <= 8, 'invalid timing world_size')
        entries = self.manifests('timing')
        expected = set()

        if world_size == 1:
            path, session = self.session('timing', report['timing_session'])
            expected_timing_cfg = {'frozen_settings_hash': self.frozen,
                'prompt_manifest_hash': fingerprint(self.manifest), 'selected_prompts': ids, 'repeats': 3, 'warmups': 1}
            if report_fp is not None:
                expected_timing_cfg['fastpath'] = report_fp
            self.require(session['configuration']['configuration'] == expected_timing_cfg, 'timing configuration mismatch')
            self.require(Path(evidence['session_path']).resolve() == path.resolve() and digest(path) == evidence['session_sha256'], 'timing session checksum/path')
            manifest_path = self.stage / 'timing/manifest-rank0.json'
            self.require(Path(evidence['manifest_path']).resolve() == manifest_path.resolve() and digest(manifest_path) == evidence['manifest_sha256'], 'timing manifest checksum/path')
            expected_harness = report.get('timing_harness_implementation_sha256')
            if expected_harness is not None:
                self.require(expected_harness in self.valid_harness_hashes, 'timing_harness_implementation_sha256 mismatch')
            for i, (p, sample, seed) in enumerate(self.test_cases(timing_prompts)):
                pairs = [(p, a, self.settings['selected'][a], sample, seed) for a in ARMS]
                for repeat in range(3):
                    offset = (i + repeat) % len(pairs)
                    expected.add(self.bundle('timing', session, entries, pairs[offset:] + pairs[:offset],
                                             repeat, False, expected_rank=0, expected_harness_sha=expected_harness))
                expected.add(self.bundle('timing', session, entries, pairs, 'phase_profile', True,
                                         expected_rank=0, expected_harness_sha=expected_harness))
            self.require(set(session['bundles']) == expected, 'timing bundle population differs')
        else:
            if self.args.scale == 'reference':
                self.require(world_size == 8, 'reference sharded M4 timing requires world_size=8')
            self.require(report.get('sharding_mode') == f'sharded_{world_size}gpu', 'timing sharding_mode mismatch')
            expected_harness = report.get('timing_harness_implementation_sha256')
            self.require(
                expected_harness in self.valid_harness_hashes
                and report.get('algorithm_implementation_sha256') in self.valid_alg_hashes,
                'sharded timing report implementation/algorithm hash mismatch',
            )
            plan_path = self.stage / 'timing/sharding_plan.json'
            self.require(
                Path(report.get('sharding_plan_path', '')).resolve() == plan_path.resolve()
                and Path(evidence.get('sharding_plan_path', '')).resolve() == plan_path.resolve(),
                'sharding_plan.json path mismatch',
            )
            plan = self.artifact(plan_path)
            plan_sha = self.files[str(plan_path.resolve())]
            self.require(
                plan_sha == report.get('sharding_plan_sha256') == evidence.get('sharding_plan_sha256'),
                'sharding_plan.json checksum mismatch',
            )
            expected_prompt_to_rank, expected_ranks = self._expected_sharding(timing_prompts, world_size)
            self.require(
                plan.get('logical_session') == report['timing_session']
                and plan.get('stage') == 'm4'
                and plan.get('scale') == self.args.scale
                and plan.get('world_size') == world_size
                and plan.get('frozen_settings_hash') == self.frozen
                and plan.get('prompt_manifest_hash') == fingerprint(self.manifest)
                and plan.get('selected_prompt_ids') == ids
                and plan.get('repeats') == 3
                and plan.get('warmups') == 1
                and plan.get('weight_source') == 'evaluate_realized_tokens'
                and plan.get('prompt_token_sums') == dict(self.eval_token_sums)
                and plan.get('prompt_to_rank') == expected_prompt_to_rank
                and plan.get('ranks') == expected_ranks,
                'sharding_plan.json differs from deterministic token-balanced snake assignment',
            )
            session_paths = evidence.get('session_paths', [])
            session_shas = evidence.get('session_sha256s', [])
            manifest_paths = evidence.get('manifest_paths', [])
            manifest_shas = evidence.get('manifest_sha256s', [])
            self.require(
                len(session_paths) == len(session_shas) == len(manifest_paths) == len(manifest_shas) == world_size,
                'sharded timing evidence file list counts differ from world_size',
            )
            actual_manifest_files = sorted((self.stage / 'timing').glob('manifest-rank*.json'))
            self.require(len(actual_manifest_files) == world_size, 'unexpected extra timing manifest-rank*.json files')
            matching_sessions = [
                p for p in (self.stage / 'timing/sessions').glob('*.json')
                if read(p).get('logical_session') == report['timing_session']
            ]
            self.require(len(matching_sessions) == world_size, 'unexpected session count for selected logical_session')

            rank_sessions = {}
            rank_affinities = {}
            seen_gpus, seen_cpus = set(), set()
            for r in range(world_size):
                exp_sp = (self.stage / 'timing/sessions' / f"{report['timing_session']}-rank{r}.json").resolve()
                self.require(Path(session_paths[r]).resolve() == exp_sp, f'rank {r}: session path mismatch')
                sess = self.artifact(exp_sp)
                self.require(self.files[str(exp_sp)] == session_shas[r], f'rank {r}: session checksum mismatch')
                self.require(
                    sess.get('logical_session') == report['timing_session']
                    and sess.get('rank') == r
                    and sess.get('world_size') == world_size
                    and sess.get('timing_harness_implementation_sha256') == expected_harness
                    and sess.get('algorithm_implementation_sha256') == report.get('algorithm_implementation_sha256'),
                    f'rank {r}: session metadata/hash mismatch',
                )
                cfg = sess['configuration']
                exp_inner = {
                    'frozen_settings_hash': self.frozen,
                    'prompt_manifest_hash': fingerprint(self.manifest),
                    'selected_prompts': ids,
                    'rank_prompts': expected_ranks[str(r)],
                    'sharding_plan_sha256': plan_sha,
                    'repeats': 3,
                    'warmups': 1,
                }
                if report_fp is not None:
                    exp_inner['fastpath'] = report_fp
                self.require(
                    cfg.get('configuration') == exp_inner,
                    f'rank {r}: inner timing configuration mismatch',
                )
                for key, value in {
                    'checkpoint_hashes': self.settings['checkpoint_hashes'],
                    'model_pair': self.settings['model_pair'],
                    'model_revisions': self.settings['model_revisions'],
                    'generation_environment': self.settings['generation_environment'],
                    'generation_caps': CAPS,
                    'stage': 'm4',
                    'scale': self.args.scale,
                    'rank': r,
                    'world_size': world_size,
                    'seed': self.args.seed,
                }.items():
                    self.require(cfg.get(key) == value, f'rank {r}: session configuration {key}')
                phys_gpu = sess.get('physical_gpu')
                numa_node = sess.get('numa_node')
                cpus = sess.get('cpu_affinity')
                self.require(
                    type(phys_gpu) is int and type(numa_node) is int
                    and isinstance(cpus, list) and bool(cpus) and all(type(c) is int for c in cpus),
                    f'rank {r}: malformed GPU/NUMA/CPU affinity metadata',
                )
                self.require(phys_gpu not in seen_gpus, f'rank {r}: duplicate physical_gpu {phys_gpu}')
                self.require(not seen_cpus.intersection(cpus), f'rank {r}: overlapping cpu_affinity')
                seen_gpus.add(phys_gpu)
                seen_cpus.update(cpus)
                if len(cpus) == 16:
                    self.require(cpus == list(range(16 * phys_gpu, 16 * phys_gpu + 16)) and numa_node == phys_gpu // 2,
                                 f'rank {r}: violates 16-core NUMA map [16g..16g+15], g//2')
                if self.args.scale == 'reference':
                    self.require(
                        cfg['device'] == 'cuda:0' and cfg['visible_devices'] == str(r)
                        and phys_gpu == r and numa_node == r // 2
                        and cpus == list(range(16 * r, 16 * r + 16))
                        and sess.get('cpu_affinity_applied') is True,
                        f'rank {r}: reference device/affinity differs from 8-GPU NUMA plan',
                    )
                exp_mp = (self.stage / 'timing' / f'manifest-rank{r}.json').resolve()
                self.require(Path(manifest_paths[r]).resolve() == exp_mp and self.files[str(exp_mp)] == manifest_shas[r],
                             f'rank {r}: manifest path/checksum mismatch')
                rank_sessions[r] = sess
                rank_affinities[r] = {'physical_gpu': phys_gpu, 'numa_node': numa_node, 'cpu_affinity': cpus}

            expected_by_rank = defaultdict(set)
            for i, (p, sample, seed) in enumerate(self.test_cases(timing_prompts)):
                r = expected_prompt_to_rank[p['prompt_id']]
                pairs = [(p, a, self.settings['selected'][a], sample, seed) for a in ARMS]
                for repeat in range(3):
                    offset = (i + repeat) % len(pairs)
                    bkey = self.bundle('timing', rank_sessions[r], entries, pairs[offset:] + pairs[:offset],
                                       repeat, False, expected_rank=r, expected_harness_sha=expected_harness,
                                       expected_affinity=rank_affinities[r])
                    expected.add(bkey)
                    expected_by_rank[r].add(bkey)
                bkey_p = self.bundle('timing', rank_sessions[r], entries, pairs, 'phase_profile', True,
                                     expected_rank=r, expected_harness_sha=expected_harness,
                                     expected_affinity=rank_affinities[r])
                expected.add(bkey_p)
                expected_by_rank[r].add(bkey_p)
            for r in range(world_size):
                self.require(set(rank_sessions[r]['bundles']) == expected_by_rank[r],
                             f'rank {r}: timing bundle population differs')

        n = 40 if self.args.scale == 'smoke' else 2400
        required = {'tune': 36 if self.args.scale == 'smoke' else 180, 'evaluate': n, 'timing': n * 3, 'profile': n}
        for category, count in required.items():
            self.require(self.counts[category] == count, f'{category}: row count differs from full schedule')
            online_count = (16 if self.args.scale == 'smoke' else 80) if category == 'tune' else count * 2 // 5
            self.require(self.counts[category + '_online'] == online_count, f'{category}: online count differs')
        self.require(report['timed_requests'] == n * 3 and evidence['timing_requests'] == n * 3 and
                     evidence['profile_requests'] == n and evidence['bundles'] == len(expected), 'timing evidence count mismatch')
        if world_size > 1 or (self.stage / 'timing_isolation_calibration.json').is_file():
            self._audit_calibration()
        for path, expected_hash in self.files.items():
            self.require(digest(path) == expected_hash, f'artifact changed during audit: {path}')

    def test_cases(self, prompts):
        for p in prompts:
            for sample in conditions(p['condition']):
                count = 1 if self.args.scale == 'smoke' or sample[1] == 0 or p['condition'] == 'long_thinking' else 3
                for i in range(count):
                    yield p, sample, self.args.seed + i


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config', type=Path)
    known, _ = preliminary.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument('--scale', choices=('smoke', 'reference', 'full'), default='reference')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--source_repo', type=Path, default=None,
                        help='optional recorded source checkout root for frozen implementation hash verification (defaults to --repo)')
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    if known.config:
        values = read(known.config)
        if set(values) - {'scale', 'repo', 'source_repo', 'artifacts', 'seed'}:
            parser.error('unsupported configuration key')
        parser.set_defaults(**{
            k: Path(v) if k in ('repo', 'source_repo', 'artifacts') and v is not None else v
            for k, v in values.items()
        })
    args = parser.parse_args(argv)
    audit = Audit(args)
    started = datetime.now(timezone.utc).isoformat()
    try:
        audit.run()
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        audit.failures.append(f'{type(error).__name__}: {error}')
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(args.repo), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_commit = commit
    if args.source_repo is not None:
        try:
            source_commit = subprocess.check_output(
                ['git', '-C', str(args.source_repo), 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = None
    result = {'status': 'passed' if not audit.failures else 'failed', 'scale': args.scale,
        'label': 'synthetic' if args.scale == 'smoke' else args.scale, 'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'project_commit': commit,
        'source_commit': source_commit,
        'resolved_config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
        'checks': audit.checks, 'counts': dict(audit.counts), 'failures': audit.failures,
        'input_artifact_sha256': audit.files, 'limitations': LIMITATIONS}
    if args.output.resolve() in {Path(p).resolve() for p in audit.files}:
        raise ValueError('output must not overwrite an audited input artifact')
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'scale', 'checks', 'counts', 'failures')}))
    return 0 if not audit.failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
