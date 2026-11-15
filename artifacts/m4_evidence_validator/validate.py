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
IMPLEMENTATION_FILES = ('adapter.py', 'online.py', 'features.py', 'dflash_loop.py',
    'sampler.py', 'verify.py', 'backend.py', 'sparse.py', 'toy.py', 'diagnose.py',
    'eval_closed_loop.py', 'fastpath.py', 'optimizers.py')
CAPS = {'standard': 2048, 'long': 24576, 'smoke': 32}
LIMITATIONS = [
    'Artifact consistency and recorded measurements only; no model/loss recomputation.',
    'Initial zero deltas, no gradient buffering, and tensor snapshot immutability remain source/test-supported, not directly observed by final JSON flags.',
    'No independent proof of GPU isolation or device clock stability from these artifacts.',
    'Smoke uses synthetic fixtures and does not complete reference M4.',
]


def _valid_fastpath_dict(fp):
    req = {'compile', 'cuda_graph', 'compile_mode'}
    opt = {'bf16_gemm', 'compile_drafter', 'compile_online'}
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
        source_root = (self.args.source_repo or self.args.repo).resolve()
        self.require((source_root / 'src/edraft/eval_closed_loop.py').is_file(),
                     f'invalid source_repo: {source_root}')
        optional_opt_files = {'fastpath.py', 'optimizers.py'}
        h = hashlib.sha256()
        for name in IMPLEMENTATION_FILES:
            path = source_root / 'src/edraft' / name
            if self.args.source_repo is not None and name in optional_opt_files and not path.exists():
                continue
            data = path.read_bytes()
            h.update(name.encode()); h.update(data)
            self.files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        self.require(h.hexdigest() == s['checkpoint_hashes']['implementation'], 'current evaluator source differs from frozen source')
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

    def row(self, identity, entries, category, expected_path=None):
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
        rank = (manifest_rank if identity['mode'] == 'evaluate' and self.args.scale == 'reference' else 0)
        self.require(type(row['rank']) is int and 0 <= row['rank'] < 64 and row['rank'] == rank
                     and path.parent.name == f'rank{rank}' and manifest_rank == rank,
                     f'{key}: rank/path/manifest mismatch')
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
        self.require(row['timing_isolated'] is (identity['mode'] == 'timing'), f'{key}: timing isolation label')
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

    def bundle(self, mode, session, entries, comparisons, repeat, profile):
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
            self.row(identity, entries, category, path)
        return key

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
        path, session = self.session('timing', report['timing_session'])
        timing_prompts = sorted(self.prompts['test'], key=lambda p: p['condition'])
        ids = [p['prompt_id'] for p in timing_prompts]
        self.require(report['prompt_ids'] == ids and report['expected_full_prompt_count'] == len(ids) and
                     report['full_reference_prompt_coverage'] is True, 'timing report prompt population')
        report_fp = report.get('fastpath')
        self.require(report_fp is None or _valid_fastpath_dict(report_fp), 'invalid timing fastpath metadata')
        expected_timing_cfg = {'frozen_settings_hash': self.frozen,
            'prompt_manifest_hash': fingerprint(self.manifest), 'selected_prompts': ids, 'repeats': 3, 'warmups': 1}
        if report_fp is not None:
            expected_timing_cfg['fastpath'] = report_fp
        self.require(session['configuration']['configuration'] == expected_timing_cfg, 'timing configuration mismatch')
        evidence = report['raw_evidence']
        self.require(Path(evidence['session_path']).resolve() == path.resolve() and digest(path) == evidence['session_sha256'], 'timing session checksum/path')
        manifest_path = self.stage / 'timing/manifest-rank0.json'
        self.require(Path(evidence['manifest_path']).resolve() == manifest_path.resolve() and digest(manifest_path) == evidence['manifest_sha256'], 'timing manifest checksum/path')
        entries = self.manifests('timing')
        expected = set()
        for i, (p, sample, seed) in enumerate(self.test_cases(timing_prompts)):
            pairs = [(p, a, self.settings['selected'][a], sample, seed) for a in ARMS]
            for repeat in range(3):
                offset = (i + repeat) % len(pairs)
                expected.add(self.bundle('timing', session, entries, pairs[offset:] + pairs[:offset], repeat, False))
            expected.add(self.bundle('timing', session, entries, pairs, 'phase_profile', True))
        self.require(set(session['bundles']) == expected, 'timing bundle population differs')
        n = 40 if self.args.scale == 'smoke' else 2400
        required = {'tune': 36 if self.args.scale == 'smoke' else 180, 'evaluate': n, 'timing': n * 3, 'profile': n}
        for category, count in required.items():
            self.require(self.counts[category] == count, f'{category}: row count differs from full schedule')
            online_count = (16 if self.args.scale == 'smoke' else 80) if category == 'tune' else count * 2 // 5
            self.require(self.counts[category + '_online'] == online_count, f'{category}: online count differs')
        self.require(report['timed_requests'] == n * 3 and evidence['timing_requests'] == n * 3 and
                     evidence['profile_requests'] == n and evidence['bundles'] == len(expected), 'timing evidence count mismatch')
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
