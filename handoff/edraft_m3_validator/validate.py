"""Read-only M3 artifact audit for the frozen mandatory reference plan.

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

ARMS = ('plain_dflash', 'online_bigram', 'dense', 'dense_sparse')
ADAPTER_ARMS = ('dense', 'dense_sparse')
IMPLEMENTATION_FILES = ('adapter.py', 'online.py', 'features.py', 'dflash_loop.py',
    'sampler.py', 'verify.py', 'backend.py', 'sparse.py', 'toy.py', 'diagnose.py',
    'eval_closed_loop.py')
CAPS = {'standard': 2048, 'long': 24576, 'smoke': 32}
LIMITATIONS = [
    'Artifact consistency and recorded measurements only; no model/loss recomputation.',
    'Full-vocabulary scoring, ratio residual sampling, and frozen adapter immutability remain source/test-supported beyond recorded block metadata.',
    'No independent proof of GPU isolation or device clock stability from these artifacts.',
    'Smoke uses synthetic fixtures and does not complete reference M3.',
]


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


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
    if arm in ADAPTER_ARMS:
        return [{'beta': .5}, {'beta': 1.}]
    raise ValueError(f'unsupported M3 arm: {arm}')


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


def greedy_token_mismatch(output, reference):
    count = max(len(output), len(reference))
    positions = [i for i in range(count) if i >= len(output) or i >= len(reference) or output[i] != reference[i]]
    return {
        'sequence_equal': not positions,
        'mismatched_tokens': len(positions),
        'compared_tokens': count,
        'token_mismatch_rate': len(positions) / count if count else 0.,
        'first_mismatch': positions[0] if positions else None,
        'output_length': len(output),
        'reference_length': len(reference),
        'length_difference': len(output) - len(reference),
    }


class Audit:
    def __init__(self, args):
        self.args = args
        self.root = args.artifacts.resolve()
        self.stage = self.root / 'm3'
        self.checks = 0
        self.failures = []
        self.counts = Counter()
        self.files = {}
        self.entry_ranks = {}
        self.inputs = {}
        self.arm_norms = {}
        self.dev_rows = defaultdict(list)
        self.greedy_eval_outputs = {}
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
        self.require(s.get('frozen') is True and s.get('stage') == 'm3' and s.get('scale') == self.args.scale,
                     'frozen M3 settings stage/scale mismatch')
        self.require(s['settings_seed'] == self.args.seed, 'settings seed mismatch')
        self.require(s['generation_caps'] == CAPS, 'generation caps differ from reviewed continuation plan')
        self.require(set(s['selected']) == set(ARMS), 'missing or additional M3 reference arm')
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
        h = hashlib.sha256()
        for name in IMPLEMENTATION_FILES:
            path = self.args.repo / 'src/edraft' / name
            data = path.read_bytes()
            h.update(name.encode())
            h.update(data)
            self.files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        src_repo = getattr(self.args, 'source_repo', None)
        if src_repo is None and self.args.repo.resolve() == Path('/home/catid/drafter').resolve():
            snap_default = Path('/tmp/edraft_09a5469_source')
            if h.hexdigest() != s['checkpoint_hashes']['implementation'] and (snap_default / 'src/edraft/eval_closed_loop.py').is_file():
                src_repo = snap_default
        if src_repo is not None:
            src_repo = Path(src_repo).resolve()
            h_src = hashlib.sha256()
            for name in IMPLEMENTATION_FILES:
                sp = src_repo / 'src/edraft' / name
                sdata = sp.read_bytes()
                h_src.update(name.encode())
                h_src.update(sdata)
                self.files[str(sp)] = hashlib.sha256(sdata).hexdigest()
                if name != 'eval_closed_loop.py':
                    live_p = self.args.repo / 'src/edraft' / name
                    self.require(live_p.read_bytes() == sdata, f'algorithm file {name} drifted from 09a5469 source_repo')
            self.require(h_src.hexdigest() == s['checkpoint_hashes']['implementation'], 'source_repo differs from frozen M3 source')
        else:
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
        return {
            'model_pair': s['model_pair'],
            'prompt': prompt['prompt_id'],
            'scale': self.args.scale,
            'prompt_payload_hash': fingerprint(prompt),
            'condition': prompt['condition'],
            'seed': seed,
            'temperature': temperature,
            'sampling_name': name,
            'verification_rule': 'ratio' if temperature > 0 else 'greedy',
            'proposal_mode': arm,
            'checkpoint': s['checkpoint_hashes'],
            'adapter_version': 0,
            'model_revisions': s['model_revisions'],
            'stage': 'm3',
            'mode': mode,
            'sampling_warp_config': warper,
            'setting': setting,
            'frozen_settings_hash': '' if mode == 'tune' else self.frozen,
            'generation_cap': 32 if self.args.scale == 'smoke' else 24576 if prompt['condition'].startswith('long_') else 2048,
            'generation_caps': CAPS,
            'timing_session': session,
            'repeat': repeat,
            'phase_profile': profile,
        }

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

    def check_static_and_block_invariants(self, key, row, identity):
        arm = identity['proposal_mode']
        # M3 must never carry neural online adaptation state or baseline reuse
        self.require('baseline_reuse' not in row, f'{key}: M3 rows cannot reuse baseline evidence')
        self.require(row.get('online') is None and row.get('online_invariants') is None
                     and row.get('estimated_online_state_bandwidth_GBps') is None,
                     f'{key}: M3 static row must have null neural online fields')
        gen = row['generation']
        self.require(gen.get('repetition_detector_version') == 2 and _integer(gen.get('repetition_detector_version')),
                     f'{key}: repetition_detector_version must be integer 2')
        self.require(isinstance(gen.get('repetition_dominated'), bool), f'{key}: repetition_dominated must be bool')
        self.require(gen.get('realized_length') == row['generated_tokens'], f'{key}: realized_length mismatch')
        # Bigram vs non-bigram metadata
        if arm == 'online_bigram':
            bg = row.get('bigram')
            self.require(isinstance(bg, dict) and bg.get('k_ret') is None
                         and bg.get('ledger_retained_for_exact_counts') is True
                         and _integer(bg.get('counted_transitions')) and bg['counted_transitions'] >= 0
                         and _integer(bg.get('exact_ledger_pairs')) and bg['exact_ledger_pairs'] >= 0
                         and bg.get('retained_prior_pairs') == bg.get('exact_ledger_pairs'),
                         f'{key}: invalid online_bigram metadata')
        else:
            self.require(row.get('bigram') is None, f'{key}: unexpected bigram metadata on {arm}')
        # Parameter norms
        norms = row.get('parameter_norms')
        if arm in ADAPTER_ARMS:
            self.require(isinstance(norms, dict) and set(norms) == {'W', 'A', 'b'}
                         and all(_finite(norms[k]) and norms[k] >= 0 for k in ('W', 'A', 'b')),
                         f'{key}: invalid adapter parameter_norms')
            if self.args.scale == 'reference' and arm == 'dense':
                self.require(norms['b'] == 0.0, f'{key}: reference dense checkpoint must keep b == 0')
            self.require(arm not in self.arm_norms or self.arm_norms[arm] == norms,
                         f'{key}: offline parameter_norms changed across {arm} requests')
            self.arm_norms[arm] = norms
        else:
            self.require(norms == {}, f'{key}: baseline arm {arm} must have empty parameter_norms')

        # Per-block & stream/acceptance arithmetic
        blocks = row.get('blocks')
        self.require(isinstance(blocks, list), f'{key}: blocks must be a list')
        self.counts['blocks'] += len(blocks)
        expected_proposal = 'dense_full' if arm in ADAPTER_ARMS else 'plain_dflash'
        max_K = 4 if self.args.scale == 'smoke' else 15
        effective_cap = row['input_metadata']['effective_generation_cap']
        stops = set(self.settings['generation_environment']['stop_token_ids'])
        expected_offset = 0 if self.args.scale == 'smoke' else 1
        acc_lengths = []
        committed = []
        for idx, b in enumerate(blocks, 1):
            self.require(isinstance(b, dict), f'{key}: block {idx} must be dict')
            self.require(_integer(b.get('round_idx')) and b['round_idx'] == idx, f'{key}: non-contiguous round_idx')
            self.require(_integer(b.get('absolute_output_offset')) and b['absolute_output_offset'] == expected_offset,
                         f'{key}: absolute_output_offset mismatch at block {idx}')
            self.require(b.get('adapter_version') == 0 and b.get('next_adapter_version') == 0,
                         f'{key}: M3 block adapter_version must remain 0')
            self.require(b.get('proposal_mode') == expected_proposal,
                         f'{key}: block proposal_mode {b.get("proposal_mode")} != {expected_proposal}')
            self.require(b.get('verification_rule') == identity['verification_rule']
                         and b.get('collection_temperature') == identity['temperature']
                         and b.get('sampling_warp_config') == identity['sampling_warp_config'],
                         f'{key}: block sampling/verification metadata mismatch')
            self.require(_finite(b.get('loss_temperature')) and b['loss_temperature'] == 1,
                         f'{key}: block loss_temperature must be 1')
            expected_K = min(max_K, effective_cap - expected_offset)
            draft_tokens = b.get('draft_tokens')
            self.require(isinstance(draft_tokens, list) and 1 <= expected_K <= max_K
                         and len(draft_tokens) == expected_K
                         and all(_integer(t) and t >= 0 for t in draft_tokens),
                         f'{key}: unexpected draft_tokens length/type at block {idx}')
            self.require(b.get('physical_to_logical') == {
                'anchor_physical_index': 0, 'logical_to_physical': list(range(1, expected_K + 1))},
                f'{key}: physical_to_logical mapping mismatch at block {idx}')
            al = b.get('accepted_length')
            self.require(_integer(al) and 0 <= al <= expected_K, f'{key}: invalid accepted_length')
            decs = b.get('decisions')
            expected_decs = ([True] * al + [False]) if al < expected_K else ([True] * expected_K)
            self.require(decs == expected_decs and all(isinstance(d, bool) for d in decs),
                         f'{key}: decisions disagree with accepted_length')
            if al < expected_K:
                self.require(b.get('first_rejection_index') == al and b.get('bonus_token') is None
                             and _integer(b.get('replacement_token')), f'{key}: invalid rejection metadata')
                tail_tok = b['replacement_token']
            else:
                self.require(b.get('first_rejection_index') == expected_K and b.get('replacement_token') is None
                             and _integer(b.get('bonus_token')), f'{key}: invalid full-accept bonus metadata')
                tail_tok = b['bonus_token']
            raw_emitted = (draft_tokens[:al] + [tail_tok])[:effective_cap - expected_offset]
            if self.args.scale != 'smoke':
                for stop_idx, item in enumerate(raw_emitted):
                    if item in stops:
                        raw_emitted = raw_emitted[:stop_idx + 1]
                        break
            block_committed = b.get('committed_tokens')
            self.require(block_committed == raw_emitted, f'{key}: block committed_tokens mismatch verified emission')
            if identity['temperature'] > 0:
                p_probs = b.get('proposal_token_probs', [])
                t_probs = b.get('target_token_probs', [])
                self.require(len(p_probs) == expected_K and len(t_probs) == expected_K
                             and all(_finite(v) and -1e-6 <= v <= 1.0 + 1e-6 for v in p_probs + t_probs),
                             f'{key}: invalid stochastic block probabilities')
            else:
                self.require(b.get('proposal_token_probs') is None and b.get('target_token_probs') is None,
                             f'{key}: greedy block must have null proposal/target probabilities')
            acc_lengths.append(al)
            committed.extend(block_committed)
            expected_offset += len(block_committed)

        n_tok = row['generated_tokens']
        if self.args.scale == 'smoke':
            self.require(committed == row['output_tokens'], f'{key}: smoke committed stream mismatch')
            expected_tf = len(blocks) / max(1, n_tok)
        else:
            if blocks:
                prefill_tok = blocks[0]['verified_prefix_tail'][-1]
                self.require([prefill_tok] + committed == row['output_tokens'],
                             f'{key}: reference committed stream + prefill mismatch')
            else:
                self.require(n_tok == 1, f'{key}: empty blocks on reference requires single prefill token')
            expected_tf = (len(blocks) + 1) / max(1, n_tok)

        expected_mean_al = statistics.mean(acc_lengths) if acc_lengths else 0.
        expected_med_al = statistics.median(acc_lengths) if acc_lengths else 0.
        self.require(_finite(row.get('mean_accepted_length'))
                     and math.isclose(row['mean_accepted_length'], expected_mean_al, rel_tol=1e-12, abs_tol=1e-12),
                     f'{key}: mean_accepted_length mismatch')
        self.require(_finite(row.get('median_accepted_length'))
                     and math.isclose(row['median_accepted_length'], expected_med_al, rel_tol=1e-12, abs_tol=1e-12),
                     f'{key}: median_accepted_length mismatch')
        self.require(_finite(row.get('target_forwards_per_generated_token'))
                     and math.isclose(row['target_forwards_per_generated_token'], expected_tf, rel_tol=1e-12, abs_tol=1e-12),
                     f'{key}: target_forwards_per_generated_token mismatch')

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
        rank = ([p['prompt_id'] for p in self.prompts['test']].index(identity['prompt']) % 8
                if identity['mode'] == 'evaluate' and self.args.scale == 'reference' else 0)
        self.require(type(row['rank']) is int and row['rank'] == rank and path.parent.name == f'rank{rank}'
                     and self.entry_ranks[(identity['mode'], key)] == rank, f'{key}: rank/path/manifest mismatch')
        fields = {
            'scale': 'scale', 'stage': 'stage', 'arm': 'proposal_mode', 'prompt_id': 'prompt',
            'condition': 'condition', 'seed': 'seed', 'temperature': 'temperature', 'sampling_name': 'sampling_name',
            'sampling_warp_config': 'sampling_warp_config', 'settings': 'setting', 'verification_rule': 'verification_rule',
            'model_pair': 'model_pair', 'model_revisions': 'model_revisions', 'checkpoint_hashes': 'checkpoint',
            'phase_profile': 'phase_profile', 'timing_session': 'timing_session', 'repeat': 'repeat',
        }
        for field, source in fields.items():
            self.require(fingerprint(row.get(field)) == fingerprint(identity[source]), f'{key}: row {field} mismatch')
        self.require(isinstance(row.get('phase_profile'), bool) and row['phase_profile'] is identity['phase_profile'],
                     f'{key}: phase_profile must be boolean')
        for sk, sv in identity['setting'].items():
            self.require(type(row['settings'].get(sk)) is type(sv), f'{key}: settings.{sk} JSON type mismatch')
        self.require(row['split'] == ('dev' if category == 'tune' else 'test'), f'{key}: split mismatch')
        self.require(row['label'] == ('synthetic' if self.args.scale == 'smoke' else 'reference'), f'{key}: label mismatch')
        self.require(row['generation'].get('complete') is True, f'{key}: generation incomplete')
        self.require(isinstance(row.get('timing_isolated'), bool) and row['timing_isolated'] is (identity['mode'] == 'timing'),
                     f'{key}: timing isolation label')
        self.require(type(row['generated_tokens']) is int and row['generated_tokens'] == len(row['output_tokens']), f'{key}: output count')
        self.require(row['effective_beta'] == (1. if identity['temperature'] == 0 else identity['setting'].get('beta', 1.)), f'{key}: effective beta')
        meta = row['input_metadata']
        self.require(meta['generation_environment_sha256'] == fingerprint(self.settings['generation_environment']), f'{key}: environment binding')
        self.require(type(meta['input_token_count']) is int and meta['input_token_count'] > 0 and
                     isinstance(meta.get('enable_thinking'), bool) and
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
        self.require(isinstance(row['generation'].get('capped'), bool) and
                     isinstance(row['generation'].get('eos_reached'), bool) and
                     row['generation'].get('capped') is (len(tokens) == effective) and
                     row['generation'].get('eos_reached') is eos and
                     (self.args.scale == 'smoke' or eos or len(tokens) == effective),
                     f'{key}: termination/cap flags contradict output')
        self.require(not any(token in stops for token in tokens[:-1]), f'{key}: tokens after EOS')
        prompt = identity['prompt']
        self.require(prompt not in self.inputs or self.inputs[prompt] == meta, f'{key}: input metadata differs across requests')
        self.inputs[prompt] = meta

        seconds, speed = row['elapsed_seconds'], row['tokens_per_second']
        self.require(_finite(seconds) and seconds > 0, f'{key}: request time')
        self.require(_finite(row.get('generation_loop_elapsed_seconds')) and row['generation_loop_elapsed_seconds'] > 0,
                     f'{key}: generation loop time')
        self.require(_finite(speed) and math.isclose(speed, row['generated_tokens'] / seconds, rel_tol=1e-10, abs_tol=1e-12),
                     f'{key}: throughput algebra')
        allocated, reserved = row['peak_allocated_bytes'], row['peak_reserved_bytes']
        self.require((type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved)
                     if self.args.scale == 'reference' else allocated is None and reserved is None,
                     f'{key}: peak memory measurements')

        phases = row['phase_ms']
        if identity['phase_profile']:
            self.require(isinstance(phases, dict) and all(_finite(v) and v >= 0 for v in phases.values()),
                         f'{key}: phase timing values')
            strictly_positive_phases = ['dflash_draft', 'sequential_sampling', 'target_verification'] if row['blocks'] else []
            if row['blocks'] and identity['proposal_mode'] in ADAPTER_ARMS:
                strictly_positive_phases.append('dense_adapter_score')
            if row['blocks'] and identity['proposal_mode'] == 'dense_sparse':
                strictly_positive_phases.append('sparse_lookup')
            if self.args.scale == 'reference':
                strictly_positive_phases.append('target_feature_extraction')
            for phase in strictly_positive_phases:
                value = phases.get(phase)
                self.require(_finite(value) and value > 0, f'{key}: required measured phase {phase}')
            if row['blocks'] and identity['proposal_mode'] == 'online_bigram':
                # OnlineBigram.score/update are pure CPU dict operations; on CUDA PhaseTimer
                # records GPU stream events around 0-kernel intervals which can measure 0.0 ms.
                for phase in ('bigram_score', 'online_update'):
                    value = phases.get(phase)
                    if self.args.scale == 'smoke':
                        self.require(_finite(value) and value > 0, f'{key}: required smoke CPU phase {phase}')
                    else:
                        self.require(phase in phases and _finite(value) and value >= 0,
                                     f'{key}: required online_bigram phase {phase}')
        else:
            self.require(phases == {}, f'{key}: uninstrumented row must not contain phase durations')

        self.check_static_and_block_invariants(key, row, identity)
        self.counts[category] += 1
        if category == 'tune':
            self.dev_rows[(identity['proposal_mode'], fingerprint(identity['setting']))].append({
                k: row[k] for k in ('prompt_id', 'condition', 'generation', 'tokens_per_second', 'generated_tokens', 'elapsed_seconds')})
        if category == 'evaluate' and identity['temperature'] == 0 and identity['condition'] == 'standard_nonthinking':
            self.greedy_eval_outputs[(identity['prompt'], identity['seed'], identity['proposal_mode'])] = (
                row['output_tokens'], row['input_metadata'], identity)
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

    def check_greedy_and_summary(self, n_eval):
        # Verify rank evaluation summaries
        expected_ranks = range(1) if self.args.scale == 'smoke' else range(8)
        rank_row_sum = 0
        for rank in expected_ranks:
            rank_report = self.artifact(self.stage / f'evaluation-rank{rank}.json')
            self.require(rank_report.get('status') == 'rank_complete' and rank_report.get('stage') == 'm3'
                         and rank_report.get('scale') == self.args.scale and rank_report.get('rank') == rank
                         and rank_report.get('frozen_settings_hash') == self.frozen,
                         f'evaluation-rank{rank}: invalid status or settings hash')
            self.require(rank_report.get('baseline_reuse', {}).get('reused_requests') == 0,
                         f'evaluation-rank{rank}: M3 cannot reuse baselines')
            rank_row_sum += rank_report.get('rows', 0)
        self.require(rank_row_sum == n_eval, 'summed evaluation-rank row counts differ from expected total')

        # Verify shared target_only greedy artifacts and greedy_output_comparisons.json
        std_test = [p for p in self.prompts['test'] if p['condition'] == 'standard_nonthinking']
        target_outputs = {}
        for p in std_test:
            plain_out, input_meta, plain_ident = self.greedy_eval_outputs[(p['prompt_id'], self.args.seed, 'plain_dflash')]
            t_ident = {k: v for k, v in plain_ident.items() if k not in ('stage', 'frozen_settings_hash')}
            t_ident.update(mode='greedy_target_only', proposal_mode='target_only', setting={},
                           input_metadata=input_meta, timing_session=None, repeat=None, phase_profile=False)
            t_key = fingerprint(t_ident)
            t_path = self.root / 'greedy_target_only' / f'{t_key}.json'
            t_manifest_path = self.root / 'greedy_target_only' / f'{t_key}.manifest.json'
            t_manifest = self.artifact(t_manifest_path)
            t_entry = t_manifest.get('jobs', {}).get(t_key)
            self.require(t_entry is not None and t_entry.get('status') == 'complete'
                         and t_entry.get('identity') == t_ident, f'greedy_target_only manifest mismatch: {p["prompt_id"]}')
            t_row = self.artifact(t_path)
            self.require(self.files[str(t_path.resolve())] == t_entry.get('result_sha256'),
                         f'greedy_target_only checksum mismatch: {p["prompt_id"]}')
            self.require(t_row.get('identity') == t_ident and t_row.get('input_metadata') == input_meta
                         and t_row.get('generation', {}).get('complete') is True
                         and t_row.get('generated_tokens') == len(t_row.get('output_tokens', [])),
                         f'greedy_target_only row metadata mismatch: {p["prompt_id"]}')
            target_outputs[(p['prompt_id'], self.args.seed)] = t_row['output_tokens']
            self.counts['greedy_target_only'] += 1

        goc_path = self.stage / 'greedy_output_comparisons.json'
        goc = self.artifact(goc_path)
        expected_comp_count = len(std_test) * len(ARMS) * 2
        self.require(goc.get('status') == 'complete' and goc.get('expected_prompts') == len(std_test)
                     and goc.get('expected_comparisons') == expected_comp_count
                     and goc.get('completed_comparisons') == expected_comp_count
                     and goc.get('missing_comparisons') == [],
                     'greedy_output_comparisons incomplete or count mismatch')
        comp_by_key = {(c['prompt_id'], c['seed'], c['arm'], c['reference']): c for c in goc.get('comparisons', [])}
        self.require(len(comp_by_key) == expected_comp_count, 'duplicate or missing greedy comparison entries')
        for p in std_test:
            plain_out = self.greedy_eval_outputs[(p['prompt_id'], self.args.seed, 'plain_dflash')][0]
            targ_out = target_outputs[(p['prompt_id'], self.args.seed)]
            for arm in ARMS:
                arm_out = self.greedy_eval_outputs[(p['prompt_id'], self.args.seed, arm)][0]
                for ref_name, ref_out in (('plain_dflash', plain_out), ('target_only', targ_out)):
                    recorded = comp_by_key[(p['prompt_id'], self.args.seed, arm, ref_name)]
                    expected_mm = greedy_token_mismatch(arm_out, ref_out)
                    for mk, mv in expected_mm.items():
                        if isinstance(mv, float):
                            self.require(math.isclose(recorded[mk], mv, rel_tol=1e-12, abs_tol=1e-12),
                                         f'greedy mismatch metric {mk} differs')
                        else:
                            self.require(recorded[mk] == mv, f'greedy mismatch field {mk} differs')

        summary = self.artifact(self.stage / 'summary.json')
        self.require(summary.get('status') == 'complete' and summary.get('scale') == self.args.scale
                     and summary.get('stage') == 'm3' and summary.get('settings') == self.settings,
                     'M3 summary incomplete or settings mismatch')
        self.require(summary.get('expected_requests') == n_eval and summary.get('completed_requests') == n_eval
                     and summary.get('missing_requests') == [] and summary.get('isolated_timing_complete') is True
                     and summary.get('greedy_output_comparisons_complete') is True,
                     'M3 summary coverage flags incomplete')
        self.require(summary.get('feature_context') is None and summary.get('feature_context_complete') is None,
                     'M3 summary must not populate M4 feature_context')
        self.require(summary.get('baseline_reuse', {}).get('reused_requests') == 0,
                     'M3 summary reports nonzero baseline reuse')
        self.require(summary.get('greedy_output_comparisons', {}).get('artifact_sha256') == self.files[str(goc_path.resolve())],
                     'M3 summary greedy_output_comparisons sha256 mismatch')

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
        for key, value in {
            'checkpoint_hashes': self.settings['checkpoint_hashes'],
            'model_pair': self.settings['model_pair'],
            'model_revisions': self.settings['model_revisions'],
            'generation_environment': self.settings['generation_environment'],
            'generation_caps': CAPS,
            'stage': 'm3',
            'scale': self.args.scale,
            'seed': self.args.seed,
        }.items():
            self.require(config.get(key) == value, f'{mode}: session configuration {key}')
        if self.args.scale == 'reference':
            self.require(config['device'] == 'cuda:0' and config['visible_devices'] == '0',
                         f'{mode}: selected physical device differs from plan')
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
        scope = getattr(self.args, 'scope', 'full')
        self.require(scope in ('full', 'tune'), f'unsupported audit scope: {scope}')
        self.prepare()
        path, session = self.session('tune', self.settings['comparison_session'])
        config = session['configuration']['configuration']
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
        expected_tune = 28 if self.args.scale == 'smoke' else 140
        if scope == 'tune':
            self.require(self.counts['tune'] == expected_tune,
                         f'tune: row count {self.counts["tune"]} differs from full schedule {expected_tune}')
            for path, expected_hash in self.files.items():
                self.require(digest(path) == expected_hash, f'artifact changed during audit: {path}')
            return
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
        self.require(report['scale'] == self.args.scale and report['stage'] == 'm3' and report['frozen_settings_hash'] == self.frozen,
                     'timing report binding mismatch')
        path, session = self.session('timing', report['timing_session'])
        timing_prompts = sorted(self.prompts['test'], key=lambda p: p['condition'])
        ids = [p['prompt_id'] for p in timing_prompts]
        self.require(report['prompt_ids'] == ids and report['expected_full_prompt_count'] == len(ids) and
                     report['full_reference_prompt_coverage'] is True, 'timing report prompt population')
        self.require(session['configuration']['configuration'] == {'frozen_settings_hash': self.frozen,
            'prompt_manifest_hash': fingerprint(self.manifest), 'selected_prompts': ids, 'repeats': 3, 'warmups': 1}, 'timing configuration mismatch')
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
        n = 32 if self.args.scale == 'smoke' else 1920
        required = {
            'tune': 28 if self.args.scale == 'smoke' else 140,
            'evaluate': n,
            'timing': n * 3,
            'profile': n,
            'greedy_target_only': 2 if self.args.scale == 'smoke' else 64,
        }
        self.check_greedy_and_summary(n)
        for category, count in required.items():
            self.require(self.counts[category] == count, f'{category}: row count {self.counts[category]} differs from full schedule {count}')
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
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
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
    parser.add_argument('--scope', choices=('full', 'tune'), default='full')
    parser.add_argument('--repo', type=Path, default=Path('/home/catid/drafter'))
    parser.add_argument('--source_repo', type=Path, default=None)
    parser.add_argument('--artifacts', type=Path, default=Path('/home/catid/drafter/artifacts'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    if known.config:
        values = read(known.config)
        if set(values) - {'scale', 'scope', 'repo', 'artifacts', 'seed'}:
            parser.error('unsupported configuration key')
        parser.set_defaults(**{k: Path(v) if k in ('repo', 'artifacts') else v for k, v in values.items()})
    args = parser.parse_args(argv)
    audit = Audit(args)
    started = datetime.now(timezone.utc).isoformat()
    try:
        audit.run()
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        audit.failures.append(f'{type(error).__name__}: {error}')
    try:
        commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    result = {
        'status': 'passed' if not audit.failures else 'failed',
        'scale': args.scale,
        'label': 'synthetic' if args.scale == 'smoke' else args.scale,
        'started_at_utc': started,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'project_commit': commit,
        'resolved_config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
        'checks': audit.checks,
        'counts': dict(audit.counts),
        'failures': audit.failures,
        'input_artifact_sha256': audit.files,
        'limitations': LIMITATIONS,
    }
    if args.output.resolve() in {Path(p).resolve() for p in audit.files}:
        raise ValueError('output must not overwrite an audited input artifact')
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'scale', 'checks', 'counts', 'failures')}))
    return 0 if not audit.failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
