"""Exact live headroom diagnostics and separately grouped trace reports."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import bisect
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import numpy as np
import torch

from .runtime import common_parser, parse_config, record_run, atomic_json, Manifest, fingerprint
from .sampler import SamplingWarper, compact_probabilities, probabilities

TOP_M = (1, 4, 16, 64, 256)
OUTPUT_EDGES = (1024, 4096, 8192, 16384)
OUTPUT_LABELS = ('0-1K', '1K-4K', '4K-8K', '8K-16K', '16K+')
DEFAULT_CONTEXT_EDGES = (1024, 4096, 8192, 16384, 32768)
_VOCAB_IDS_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _cached_vocab_ids(V: int, device: torch.device) -> torch.Tensor:
    key = (device, int(V))
    cached = _VOCAB_IDS_CACHE.get(key)
    if cached is None:
        cached = torch.arange(V, device=device, dtype=torch.long)
        _VOCAB_IDS_CACHE[key] = cached
    return cached


def token_rank(scores, token_ids, *, _vocab_ids=None):
    """One-based full-vocabulary rank with deterministic lowest-ID tie break."""
    selected = scores.gather(-1, token_ids[:, None])
    if not scores.is_cuda:
        gt = (scores > selected).sum(-1)
        eq = (scores == selected).sum(-1)
        if not bool((eq > 1).any()):
            return 1 + gt
        ids = torch.arange(scores.shape[-1], device=scores.device) if _vocab_ids is None else _vocab_ids
        return 1 + gt + ((scores == selected) & (ids[None, :] < token_ids[:, None])).sum(-1)
    ids = torch.arange(scores.shape[-1], device=scores.device) if _vocab_ids is None else _vocab_ids
    return 1 + ((scores > selected) | ((scores == selected) & (ids[None, :] < token_ids[:, None]))).sum(-1)


def multi_token_rank(scores: torch.Tensor, token_ids_2d: torch.Tensor, *, _vocab_ids=None) -> torch.Tensor:
    """One-based full-vocabulary ranks for multiple query columns [K, Q] in one sweep."""
    if not scores.is_cuda:
        return torch.stack(
            [token_rank(scores, token_ids_2d[:, c], _vocab_ids=_vocab_ids) for c in range(token_ids_2d.shape[1])],
            dim=-1,
        )
    ids = torch.arange(scores.shape[-1], device=scores.device) if _vocab_ids is None else _vocab_ids
    selected = scores.gather(-1, token_ids_2d)
    s_exp = scores[:, None, :]
    sel_exp = selected[:, :, None]
    return 1 + ((s_exp > sel_exp) | ((s_exp == sel_exp) & (ids[None, None, :] < token_ids_2d[:, :, None]))).sum(-1)


@torch.no_grad()
def live_block_diagnostics(record: dict, seed: int, *, _return_shared: bool = False):
    """Compute before compacting; independent RNG never changes generation draws."""
    target = record['target_logits'].detach().float()
    l_unit = record.get('l_unit')
    if l_unit is None:
        l_unit = record['draft_logits'].detach().float().log_softmax(-1)
    else:
        l_unit = l_unit.detach().float()
    K, V = l_unit.shape
    if target.shape != l_unit.shape:
        raise ValueError('live diagnostics require full K-by-V target and draft tensors')
    T = float(record['collection_temperature'])
    warper = SamplingWarper(**record['sampling_warp_config'])
    p_train = target.softmax(-1) if _return_shared else target.log_softmax(-1).exp()
    q_train = l_unit.exp()
    target_ids = target.argmax(-1)
    max_ids = torch.topk(l_unit, min(max(TOP_M), V), dim=-1).indices

    use_compact_pq = (T > 0 and 0 < warper.top_k < V)
    if T > 0:
        if use_compact_pq:
            p_sub, p_idx = compact_probabilities(target, T, warper)
            q_sub, q_idx = compact_probabilities(l_unit, T, warper)
            P = torch.zeros((K, V), dtype=p_sub.dtype, device=target.device).scatter_(-1, p_idx, p_sub)
        else:
            P, q = probabilities(target, T, warper), probabilities(l_unit, T, warper)
            p_sub = p_idx = q_sub = q_idx = None
    else:
        P = torch.zeros_like(target).scatter_(-1, target_ids[:, None], 1.)
        q = p_sub = p_idx = q_sub = q_idx = None

    generator = torch.Generator(device=target.device).manual_seed(seed)
    sampled_ids = torch.multinomial(P, 1, generator=generator)[:, 0]
    sampled_train_ids = torch.multinomial(p_train, 1, generator=generator)[:, 0]
    vocab_ids = _cached_vocab_ids(V, l_unit.device) if l_unit.is_cuda else None

    first = int(record['first_rejection_index'])
    replacement = record.get('replacement_token') if first < K else None
    if replacement is not None:
        repl_col = target_ids.clone()
        repl_col[first] = int(replacement)
        ranks_all = multi_token_rank(
            l_unit,
            torch.stack((target_ids, sampled_ids, sampled_train_ids, repl_col), dim=-1),
            _vocab_ids=vocab_ids,
        )
        ranks_3 = ranks_all[:, :3]
        replacement_rank_dev = ranks_all[first:first + 1, 3]
    else:
        ranks_3 = multi_token_rank(
            l_unit, torch.stack((target_ids, sampled_ids, sampled_train_ids), dim=-1), _vocab_ids=vocab_ids
        )
        replacement_rank_dev = None

    if T == 0:
        same_top1 = (target_ids == max_ids[:, 0])
        zero_mass = (~same_top1).to(target.dtype)
        overlap_sampling = same_top1.to(target.dtype)
        ce = torch.where(
            same_top1,
            torch.zeros_like(zero_mass),
            torch.full_like(zero_mass, -math.log(torch.finfo(target.dtype).tiny)),
        )
    elif use_compact_pq:
        eq = (p_idx[:, :, None] == q_idx[:, None, :])
        q_at_p = (eq.to(q_sub.dtype) * q_sub[:, None, :]).sum(-1)
        zero_mass = (p_sub * (q_at_p == 0)).sum(-1)
        ce = -(p_sub * q_at_p.clamp_min(torch.finfo(q_sub.dtype).tiny).log()).sum(-1)
        overlap_sampling = torch.minimum(p_sub, q_at_p).sum(-1)
    else:
        zero_mass = (P * (q == 0)).sum(-1)
        ce = -(P * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum(-1)
        overlap_sampling = torch.minimum(P, q).sum(-1)

    overlap_unwarped = torch.minimum(p_train, q_train).sum(-1)
    ce_unwarped = -(p_train * l_unit).sum(-1)
    mass = P.gather(-1, max_ids).cumsum(-1)
    mass_train = p_train.gather(-1, max_ids).cumsum(-1)
    m_idx = torch.tensor([min(m, V) - 1 for m in TOP_M], device=l_unit.device, dtype=torch.long)
    mass_m = mass.index_select(-1, m_idx)
    mass_train_m = mass_train.index_select(-1, m_idx)

    # Consolidate all device-to-host transfers into at most 2 CPU copies.
    float_bundle_cpu = torch.cat(
        (
            overlap_sampling[:, None],
            overlap_unwarped[:, None],
            ce_unwarped[:, None],
            ce[:, None],
            zero_mass[:, None],
            mass_m,
            mass_train_m,
        ),
        dim=-1,
    ).cpu()
    if replacement_rank_dev is not None:
        int_bundle_cpu = torch.cat((ranks_3.reshape(-1), replacement_rank_dev)).cpu()
        ranks_cpu = int_bundle_cpu[:-1].view(K, 3)
        replacement_rank = int(int_bundle_cpu[-1])
    else:
        ranks_cpu = ranks_3.cpu()
        replacement_rank = None

    ranks_col0 = ranks_cpu[:, 0]
    ranks_col1 = ranks_cpu[:, 1]
    ranks_col2 = ranks_cpu[:, 2]
    m_thresh = torch.tensor(TOP_M, dtype=ranks_cpu.dtype)
    rec_top1_cpu = (ranks_col0[:, None] <= m_thresh).float()
    rec_samp_cpu = (ranks_col1[:, None] <= m_thresh).float()

    m_metrics = {}
    for idx_m, m in enumerate(TOP_M):
        m_metrics[str(m)] = {
            'target_top1_recall': rec_top1_cpu[:, idx_m].tolist(),
            'sampled_target_recall': rec_samp_cpu[:, idx_m].tolist(),
            'target_mass_sampling': float_bundle_cpu[:, 5 + idx_m].tolist(),
            'target_mass_unwarped': float_bundle_cpu[:, 5 + len(TOP_M) + idx_m].tolist(),
        }

    rejection = None
    if first < K:
        target_rank = int(ranks_col0[first])
        rejection = {'slot': first, 'target_top1_rank': target_rank, 'replacement_token_rank': replacement_rank,
                     'recoverability_target_top1': {str(m): target_rank <= m for m in TOP_M},
                     'recoverability_replacement': {str(m): replacement_rank <= m if replacement_rank is not None else None for m in TOP_M}}
    zero_mass_list = float_bundle_cpu[:, 4].tolist()
    infinite_list = [z > 0.0 for z in zero_mass_list]
    ce_list = float_bundle_cpu[:, 3].tolist()
    out_diag = {'definition': 'plain_DFlash_full_vocabulary_live_diagnostics', 'diagnostic_rng_seed': seed,
            'sampled_target_distribution': 'condition_sampling_distribution',
            'target_rank': ranks_col0.tolist(), 'sampled_target_rank': ranks_col1.tolist(),
            'sampled_target_rank_unwarped': ranks_col2.tolist(),
            'overlap_sampling': float_bundle_cpu[:, 0].tolist(),
            'overlap_unwarped': float_bundle_cpu[:, 1].tolist(),
            'cross_entropy_unwarped': float_bundle_cpu[:, 2].tolist(),
            'sampling_cross_entropy': [None if bad else value for bad, value in zip(infinite_list, ce_list)],
            'sampling_cross_entropy_infinite': infinite_list,
            'target_mass_on_zero_proposal': zero_mass_list,
            'top_m': m_metrics, 'first_rejection': rejection,
            'off_policy_teacher_positions': list(range(first + 1, K))}
    if _return_shared:
        return out_diag, p_train
    return out_diag


def _longest_periodic_run(tokens):
    """Find exact runs of at least four cycles, with no fixed period cutoff.

    A qualifying period cannot exceed N/4. For each period, equality between
    shifted token arrays gives every possible run, including runs inside a
    nonrepeating prefix/suffix. The count check only skips impossible periods.
    This uses O(N) working memory and O(N**2) vectorized comparisons; generation
    caps bound N, and this diagnostic runs outside request timing.
    """
    values = np.asarray(tokens, dtype=np.int64)
    best_length = 0
    best = {'start': None, 'end': None, 'period': None, 'complete_cycles': 0}
    for period in range(1, len(values) // 4 + 1):
        equal = values[period:] == values[:-period]
        if np.count_nonzero(equal) < 3 * period:
            continue
        boundaries = np.concatenate(([-1], np.flatnonzero(~equal), [len(equal)]))
        lengths = np.diff(boundaries) - 1
        index = int(lengths.argmax())
        matching_length = int(lengths[index])
        span = matching_length + period
        if matching_length < 3 * period or span <= best_length:
            continue
        start = int(boundaries[index]) + 1
        best_length = span
        best = {'start': start, 'end': start + span, 'period': period,
                'complete_cycles': span // period}
        if best_length == len(values):
            break
    return best_length, best


def generation_diagnostics(output_tokens) -> dict:
    """Report realized lengths and explicit repetition heuristics, without padding."""
    tokens = [int(token) for token in output_tokens]
    length = len(tokens)
    repeated = {}
    dominant = {}
    for n in (4, 8, 16):
        counts = Counter(tuple(tokens[i:i + n]) for i in range(max(0, length - n + 1)))
        windows = max(0, length - n + 1)
        repeated[str(n)] = 1 - len(counts) / windows if windows else 0.
        dominant[str(n)] = min(1., max(counts.values(), default=0) * n / max(length, 1))
    # Exact periodic spans distinguish loops from changing repeated boilerplate.
    longest, periodic_run = _longest_periodic_run(tokens)
    repetition = length >= 256 and (longest >= max(256, .5 * length) or
                                    (repeated['8'] >= .8 and dominant['16'] >= .5))
    return {'realized_length': length, 'repeated_ngram_fraction': repeated,
            'dominant_ngram_token_coverage': dominant, 'longest_periodic_run_tokens': longest,
            'periodic_run': periodic_run, 'repetition_detector_version': 2,
            'repetition_dominated': repetition,
            'repetition_rule': 'length>=256 and (exact repeated run with >=4 cycles, any period, spans >=max(256,50% length) or repeated8>=80% and dominant16 coverage>=50%)',
            'output_bins_reached': [label for edge, label in zip((0, *OUTPUT_EDGES), OUTPUT_LABELS) if length > edge]}


def recompute_generation_diagnostics(generation):
    """Refresh derived flags from stored tokens without changing trace evidence."""
    if 'output_tokens' not in generation:
        raise ValueError('Stored output_tokens are required to recompute generation diagnostics')
    diagnostics = generation_diagnostics(generation['output_tokens'])
    if generation.get('realized_length', diagnostics['realized_length']) != diagnostics['realized_length']:
        raise ValueError('Stored realized_length disagrees with output_tokens')
    previous = generation.get('diagnostic_recomputation', {})
    stored_version = previous.get('stored_repetition_detector_version',
                                  generation.get('repetition_detector_version', 'legacy_period_le_64'))
    stored_rule = previous.get('stored_repetition_rule', generation.get('repetition_rule'))
    stored_flag = previous.get('stored_repetition_dominated', generation.get('repetition_dominated'))
    provenance = {'basis': 'stored_output_tokens',
                  'stored_repetition_detector_version': stored_version,
                  'stored_repetition_rule': stored_rule,
                  'stored_repetition_dominated': stored_flag,
                  'classification_changed': stored_flag != diagnostics['repetition_dominated']}
    return {**generation, **diagnostics, 'diagnostic_recomputation': provenance}


def bootstrap_prompt_means(observations, seed=2026, samples=2000):
    """Aggregate seed results within each prompt before prompt-ID bootstrap."""
    grouped = defaultdict(list)
    for row in observations:
        grouped[str(row['prompt_id'])].append(float(row['value']))
    if not grouped:
        return {'n_prompts': 0, 'mean': None, 'ci95': None, 'seed_std_mean': None}
    means = torch.tensor([statistics.mean(values) for _, values in sorted(grouped.items())], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    # Chunk bootstrap indices to bound memory on large prompt manifests.
    estimates = []
    for start in range(0, samples, 128):
        indices = torch.randint(len(means), (min(128, samples - start), len(means)), generator=generator)
        estimates.append(means[indices].mean(-1))
    quantiles = torch.quantile(torch.cat(estimates), torch.tensor([.025, .975], dtype=torch.float64))
    return {'n_prompts': len(grouped), 'mean': float(means.mean()), 'ci95': quantiles.tolist(),
            'seed_std_mean': statistics.mean(statistics.pstdev(values) if len(values) > 1 else 0. for values in grouped.values()),
            'bootstrap_unit': 'prompt_id', 'seed_aggregation': 'arithmetic_mean_before_bootstrap'}


def prompt_bootstrap(rows, value_key, prompt_key='prompt_id', seed=2026, iterations=2000):
    """Public M3/M4 helper: aggregate observations per seed, then per prompt."""
    by_seed = defaultdict(list)
    for row in rows:
        value = row[value_key]
        if value is None or not math.isfinite(float(value)):
            continue
        by_seed[(str(row[prompt_key]), row.get('seed', 0))].append(float(value))
    observations = [{'prompt_id': prompt, 'seed': run_seed, 'value': statistics.mean(values)}
                    for (prompt, run_seed), values in by_seed.items()]
    return bootstrap_prompt_means(observations, seed=seed, samples=iterations)


def output_bin(offset):
    if offset < 0:
        raise ValueError('output offset must be nonnegative')
    return OUTPUT_LABELS[bisect.bisect_right(OUTPUT_EDGES, offset)]


def context_bin(length, edges=DEFAULT_CONTEXT_EDGES):
    if length < 0:
        raise ValueError('context length must be nonnegative')
    return _context_label(length, edges)


def _context_label(length, edges):
    i = bisect.bisect_right(edges, length)
    lower = 0 if i == 0 else edges[i - 1]
    upper = edges[i] if i < len(edges) else None
    return f'{lower}-{upper}' if upper is not None else f'{lower}+'


class DiagnosticAccumulator:
    """Streaming, condition-separated metrics and projected-feature moments."""
    def __init__(self, context_edges=DEFAULT_CONTEXT_EDGES, outlier_threshold=5., seed=2026):
        self.context_edges = tuple(context_edges)
        if tuple(sorted(set(self.context_edges))) != self.context_edges:
            raise ValueError('context edges must be unique and increasing')
        self._context_labels = tuple(
            (f'{0 if j == 0 else self.context_edges[j - 1]}-{self.context_edges[j]}'
             if j < len(self.context_edges)
             else f'{0 if j == 0 else self.context_edges[j - 1]}+')
            for j in range(len(self.context_edges) + 1)
        )
        self._output_edges_np = np.asarray(OUTPUT_EDGES, dtype=np.int64)
        self._context_edges_np = np.asarray(self.context_edges, dtype=np.int64)
        self.outlier_threshold = outlier_threshold
        self.seed = seed
        self.groups = {}
        self.generations = []

    def add_generation(self, generation):
        self.generations.append(dict(generation))

    def _accumulate_prepared_block(self, block, repetition_dominated: bool, h: torch.Tensor,
                                   h_squared: torch.Tensor, norms: list[float], outlier_counts: list[int],
                                   out_bins: list[str], ctx_bins: list[str], warp_json: str) -> None:
        diag = block['diagnostics']
        K = len(block['draft_tokens'])
        first = int(block['first_rejection_index'])
        condition = block['condition']
        include_primary = not (condition.startswith('long_') and repetition_dominated)
        inclusion = 'primary' if include_primary else 'repetition_excluded'
        arm = (condition, float(block['collection_temperature']), warp_json, block['split'])
        members = defaultdict(list)
        for i in range(K):
            policy = 'off_policy_teacher' if i > first else 'on_policy'
            members[(*arm, 'all', 'all', policy, inclusion)].append(i)
            members[(*arm, 'output_offset', out_bins[i], policy, inclusion)].append(i)
            members[(*arm, 'input_context_length', ctx_bins[i], policy, inclusion)].append(i)
        metric_values = [(metric, [None if value is None else float(value) for value in diag[metric]])
                         for metric in ('target_rank', 'sampled_target_rank', 'sampled_target_rank_unwarped',
                                        'overlap_sampling', 'overlap_unwarped', 'cross_entropy_unwarped',
                                        'sampling_cross_entropy', 'sampling_cross_entropy_infinite',
                                        'target_mass_on_zero_proposal')]
        metric_values.extend((f'{metric}_at_{m}', [float(value) for value in values])
                             for m, stats in diag['top_m'].items() for metric, values in stats.items())
        prompt_seed = (block['prompt_id'], block['seed'])
        moments = {}
        for key, indices in members.items():
            start, stop = indices[0], indices[-1] + 1
            group = self.groups.get(key)
            if group is None:
                group = {'rows': defaultdict(lambda: defaultdict(lambda: [0., 0])), 'n': 0,
                         'h_sum': torch.zeros(h.shape[-1], dtype=torch.float64),
                         'h_sq_sum': torch.zeros(h.shape[-1], dtype=torch.float64), 'h_norms': [],
                         'outlier_slots': 0, 'outlier_coordinates': 0,
                         'accepted_length': [], 'first_rejection': [],
                         'acceptance_by_block_position': defaultdict(list)}
                self.groups[key] = group
            if (start, stop) not in moments:
                moments[start, stop] = (h[start:stop].sum(0), h_squared[start:stop].sum(0))
            h_sum, h_sq_sum = moments[start, stop]
            group['n'] += stop - start
            group['h_sum'] += h_sum
            group['h_sq_sum'] += h_sq_sum
            group['h_norms'].extend(norms[start:stop])
            group['outlier_slots'] += sum(count > 0 for count in outlier_counts[start:stop])
            group['outlier_coordinates'] += sum(outlier_counts[start:stop])
            for metric, values in metric_values:
                selected = values[start:stop]
                if any(value is not None for value in selected):
                    accumulator = group['rows'][metric][prompt_seed]
                    total, count = accumulator
                    # Keep scalar addition order for exact per-prompt metrics
                    # and unchanged prompt-bootstrap inputs. Only feature
                    # moment reductions are batched, in float64.
                    for value in selected:
                        if value is not None:
                            total += value
                            count += 1
                    accumulator[:] = total, count
            for i in indices:
                group['acceptance_by_block_position'][str(i)].append(float(i < first))
            # Record a round once in every bin it actually starts in, and its
            # first rejection only in the bins containing that logical slot.
            if start == 0:
                group['accepted_length'].append(float(block['accepted_length']))
            if start <= first < stop and diag['first_rejection'] is not None:
                group['first_rejection'].append(diag['first_rejection'])

    def add_block(self, block, repetition_dominated=False):
        if 'diagnostics' not in block:
            raise ValueError('exact live diagnostics are missing; compact union alone cannot recover full ranks or overlap')
        h = block['h'].double().cpu()
        K = len(block['draft_tokens'])
        out_bins = [OUTPUT_LABELS[bisect.bisect_right(OUTPUT_EDGES, block['absolute_output_offset'] + i)] for i in range(K)]
        ctx_bins = [self._context_labels[bisect.bisect_right(self.context_edges, block['input_context_length'] + i)] for i in range(K)]
        warp_json = json.dumps(block['sampling_warp_config'], sort_keys=True)
        self._accumulate_prepared_block(
            block, repetition_dominated, h, h.square(), h.norm(dim=-1).tolist(),
            (h.abs() > self.outlier_threshold).sum(-1).tolist(), out_bins, ctx_bins, warp_json,
        )

    def add_blocks(self, blocks, repetition_dominated=False):
        """Shard-batched diagnostic accumulation across multiple blocks."""
        if not blocks:
            return
        for block in blocks:
            if 'diagnostics' not in block:
                raise ValueError('exact live diagnostics are missing; compact union alone cannot recover full ranks or overlap')
        H_all = torch.cat([b['h'].double().cpu() for b in blocks], dim=0)
        H_sq_all = H_all.square()
        all_norms = H_all.norm(dim=-1).tolist()
        all_outliers = (H_all.abs() > self.outlier_threshold).sum(-1).tolist()
        offsets_list = []
        contexts_list = []
        lengths = []
        for b in blocks:
            K = len(b['draft_tokens'])
            lengths.append(K)
            off0 = int(b['absolute_output_offset'])
            ctx0 = int(b['input_context_length'])
            offsets_list.extend(range(off0, off0 + K))
            contexts_list.extend(range(ctx0, ctx0 + K))
        offsets_np = np.asarray(offsets_list, dtype=np.int64)
        contexts_np = np.asarray(contexts_list, dtype=np.int64)
        out_idx_all = np.searchsorted(self._output_edges_np, offsets_np, side='right').tolist()
        ctx_idx_all = np.searchsorted(self._context_edges_np, contexts_np, side='right').tolist()
        ctx_labels = self._context_labels
        cursor = 0
        for block, K in zip(blocks, lengths):
            end = cursor + K
            warp_json = json.dumps(block['sampling_warp_config'], sort_keys=True)
            out_bins = [OUTPUT_LABELS[j] for j in out_idx_all[cursor:end]]
            ctx_bins = [ctx_labels[j] for j in ctx_idx_all[cursor:end]]
            self._accumulate_prepared_block(
                block, repetition_dominated, H_all[cursor:end], H_sq_all[cursor:end],
                all_norms[cursor:end], all_outliers[cursor:end], out_bins, ctx_bins, warp_json,
            )
            cursor = end

    def report(self):
        results = []
        for key, group in sorted(self.groups.items()):
            condition, temperature, warp, split, axis, bin_name, policy, inclusion = key
            n = group['n']
            mean = group['h_sum'] / n
            variance = (group['h_sq_sum'] / n - mean.square()).clamp_min(0)
            norms = torch.tensor(group['h_norms'], dtype=torch.float64)
            metrics = {}
            for metric, values in group['rows'].items():
                # First aggregate positions within prompt+seed, then seeds per prompt.
                observations = [{'prompt_id': p, 'seed': s, 'value': total / count} for (p, s), (total, count) in values.items()]
                metrics[metric] = bootstrap_prompt_means(observations, self.seed)
            results.append({'condition': condition, 'collection_temperature': temperature, 'sampling_warp_config': json.loads(warp),
                'split': split, 'axis': axis, 'bin': bin_name, 'data_policy': policy, 'inclusion': inclusion,
                'logical_slots': n, 'metrics': metrics,
                'accepted_length_mean': statistics.mean(group['accepted_length']) if group['accepted_length'] else None,
                'acceptance_by_block_position': {i: statistics.mean(values) for i, values in group['acceptance_by_block_position'].items()},
                'first_rejection_count': len(group['first_rejection']),
                'first_rejection_target_rank_mean': statistics.mean(x['target_top1_rank'] for x in group['first_rejection']) if group['first_rejection'] else None,
                'first_rejection_replacement_rank_mean': statistics.mean(x['replacement_token_rank'] for x in group['first_rejection'] if x['replacement_token_rank'] is not None) if any(x['replacement_token_rank'] is not None for x in group['first_rejection']) else None,
                'recoverability_target_top1': {str(m): statistics.mean(float(x['recoverability_target_top1'][str(m)]) for x in group['first_rejection']) if group['first_rejection'] else None for m in TOP_M},
                'recoverability_replacement': {str(m): statistics.mean(float(x['recoverability_replacement'][str(m)]) for x in group['first_rejection'] if x['recoverability_replacement'][str(m)] is not None) if any(x['recoverability_replacement'][str(m)] is not None for x in group['first_rejection']) else None for m in TOP_M},
                'features': {'coordinate_mean_norm': float(mean.norm()),
                             'variance_ratio_to_standard_calibration': float(variance.mean()),
                             'calibration_reference': 'standardized training-calibration per-coordinate population variance = 1',
                             'l2_norm_quantiles': dict(zip(['p0', 'p25', 'p50', 'p75', 'p95', 'p99', 'p100'], torch.quantile(norms, torch.tensor([0, .25, .5, .75, .95, .99, 1], dtype=torch.float64)).tolist())),
                             'outlier_threshold_abs_coordinate': self.outlier_threshold,
                             'outlier_slot_rate': group['outlier_slots'] / n,
                             'outlier_coordinate_rate': group['outlier_coordinates'] / (n * len(mean))}})
        generation_groups = defaultdict(list)
        for generation in self.generations:
            key = (generation.get('condition', 'unknown'), generation.get('temperature', generation.get('collection_temperature')), generation.get('split', 'unknown'), json.dumps(generation.get('sampling_warp_config', {}), sort_keys=True))
            generation_groups[key].append(generation)
        lengths = []
        for key, generations in sorted(generation_groups.items()):
            condition, temperature, split, warp = key
            primary = [g for g in generations if not (condition.startswith('long_') and g.get('repetition_dominated', False))]
            realized = [int(g['realized_length']) for g in primary]
            counts = {label: sum(length > edge for length in realized) for edge, label in zip((0, *OUTPUT_EDGES), OUTPUT_LABELS)}
            lengths.append({'condition': condition, 'collection_temperature': temperature, 'split': split, 'sampling_warp_config': json.loads(warp),
                'generations': len(generations), 'primary_generations': len(primary),
                'repetition_excluded': len(generations) - len(primary),
                'realized_length_mean': statistics.mean(realized) if realized else None,
                'realized_length_median': statistics.median(realized) if realized else None,
                'natural_eos_count': sum(bool(g.get('eos_reached')) for g in generations),
                'capped_count': sum(bool(g.get('capped')) for g in generations),
                'primary_generations_reaching_bins': counts,
                'empty_bins': [label for label, count in counts.items() if count == 0],
                'minimum_long_generations_per_bin': 8,
                'greedy_long_thinking_diagnostic_only': condition == 'long_thinking' and temperature == 0,
                'long_context_conclusions_supported': counts['8K-16K'] >= 8 and not (condition == 'long_thinking' and temperature == 0),
                'data_sparsity': any(counts[label] < 8 for label in ('8K-16K', '16K+')) if condition.startswith('long_') else None})
        return {'groups': results, 'generation_lengths': lengths, 'context_bin_edges': self.context_edges,
                'output_bins': OUTPUT_LABELS, 'outlier_threshold': self.outlier_threshold,
                'cross_entropy_note': 'Sampling CE is +infinity when target has positive mass on zero-q support; represented by None plus a separate infinite fraction. Unwarped CE remains finite.',
                'off_policy_note': 'Post-rejection draft-canvas conditionals are reported separately from the committed-prefix policy.'}


def smoke_diagnostic(args):
    from .smoke import run_smoke
    from .toy import ToyTarget, ToyParallelDrafter
    from .dflash_loop import generate_toy
    # Preserve the integration contract for every smoke entry point.
    run_smoke(args)
    target = ToyTarget(seed=args.seed)
    drafter = ToyParallelDrafter(target, seed=args.seed + 1)
    accumulator = DiagnosticAccumulator(seed=args.seed)
    for index, T in enumerate((0., .7, 1.)):
        prompt_id = f'synthetic-diagnostic-{index}'
        warp = SamplingWarper(top_k=3, top_p=.8) if T == .7 else SamplingWarper()
        def callback(record):
            record['diagnostics'] = live_block_diagnostics(record, args.seed + 1000 + record['round_idx'])
            record.update({'prompt_id': prompt_id, 'seed': args.seed, 'condition': 'standard_nonthinking', 'split': 'test'})
            accumulator.add_block(record)
        result = generate_toy(target, drafter, [1, 2, 3, 4], 64, temperature=T, warper=warp,
                              generator=torch.Generator().manual_seed(args.seed), on_block=callback)
        generation = generation_diagnostics(result.output_ids[0, 4:].tolist())
        generation.update({'prompt_id': prompt_id, 'condition': 'standard_nonthinking', 'split': 'test',
                           'temperature': T, 'seed': args.seed, 'sampling_warp_config': warp.to_dict(),
                           'eos_reached': False, 'capped': True, 'complete': True})
        accumulator.add_generation(generation)
    return accumulator.report()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audited_trace_sources(audit_path, scale, trace_dir=None):
    """Consume the exact audited shard set, never a directory-wide glob.

    The audit's frozen checksums prevent a sidecar and shard changed together
    after acceptance from entering diagnostics. Duplicate/superseded jobs that
    remain on disk are deliberately outside this source set.
    """
    audit_path = Path(audit_path)
    report = json.loads(audit_path.read_text())
    if report.get('status') != 'passed' or report.get('scale') != scale:
        raise RuntimeError('M1b requires a passed M1 audit at the requested scale')
    sources = report.get('source_shards')
    if not sources:
        raise RuntimeError('M1 audit lacks frozen source checksums; rerun collect --stage audit')
    paths = [str(Path(row['path']).resolve()) for row in sources]
    if len(paths) != len(set(paths)) or set(paths) != {str(Path(p).resolve()) for p in report['shards']}:
        raise RuntimeError('M1 audit has duplicate or inconsistent accepted source paths')
    seen_identities = set()
    resolved_paths: list[Path] = []
    for source, path_text in zip(sources, paths):
        path = Path(path_text)
        if trace_dir is not None and not path.is_relative_to(Path(trace_dir).resolve()):
            raise RuntimeError('Audited source lies outside the selected trace directory')
        sidecar = json.loads(path.with_suffix('.pt.json').read_text())
        identity_hash = fingerprint(sidecar['identity'])
        if identity_hash in seen_identities or identity_hash != source['identity_sha256']:
            raise RuntimeError('Audited source identity changed or is duplicated')
        seen_identities.add(identity_hash)
        if sidecar['sha256'] != source['sha256'] or not path.is_file():
            raise RuntimeError(f'Audited trace checksum changed: {path}')
        resolved_paths.append(path)
    if len(resolved_paths) > 1:
        hash_workers = min(16, len(resolved_paths), max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=hash_workers) as pool:
            actual_shas = list(pool.map(_file_sha256, resolved_paths))
    else:
        actual_shas = [_file_sha256(p) for p in resolved_paths]
    for source, path, actual_sha in zip(sources, resolved_paths, actual_shas):
        if actual_sha != source['sha256']:
            raise RuntimeError(f'Audited trace checksum changed: {path}')
    calibration = report['calibration']
    for name, expected in calibration['artifact_sha256'].items():
        if _file_sha256(audit_path.parent / name) != expected:
            raise RuntimeError(f'Frozen calibration artifact changed after audit: {name}')
    prompt_manifest = Path(calibration['resolved_config']['prompt_manifest'])
    if _file_sha256(prompt_manifest) != calibration['identity']['prompt_manifest_sha256']:
        raise RuntimeError('Prompt manifest changed after the accepted M1 audit')
    return sources, {'audit_path': str(audit_path.resolve()), 'audit_sha256': _file_sha256(audit_path),
                     'source_set_sha256': fingerprint(sources),
                     'frozen_artifact_sha256': calibration['artifact_sha256'],
                     'prompt_manifest_sha256': calibration['identity']['prompt_manifest_sha256']}


def _accumulate_source(accumulator, source, scale, *, preverified_audit: bool = False):
    from .traces import read_shard
    path = Path(source['path'])
    sidecar = json.loads(path.with_suffix('.pt.json').read_text())
    if sidecar['sha256'] != source['sha256']:
        raise RuntimeError(f'Audited trace checksum changed: {path}')
    if preverified_audit:
        shard = read_shard(path, verify_checksum=False, validate=False)
    else:
        shard = read_shard(path)  # Retain full checksum and canonical-schema checks.
    generation = shard['metadata'].get('generation')
    if generation is None:
        raise RuntimeError(f'{path}: generation metadata missing; realized length/repetition cannot be inferred from a cap')
    if not generation.get('complete', False):
        raise RuntimeError(f'{path}: generation is incomplete')
    generation = recompute_generation_diagnostics(generation)
    accumulator.add_generation(generation)
    for block in shard['blocks']:
        if block['scale'] != scale:
            raise RuntimeError(f'{path}: trace scale does not match requested scale')
    accumulator.add_blocks(shard['blocks'], repetition_dominated=generation['repetition_dominated'])


def _diagnostic_worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)


def _diagnostic_shard_worker(source, scale, context_edges, outlier_threshold, seed, preverified_audit: bool = False):
    accumulator = DiagnosticAccumulator(context_edges, outlier_threshold, seed)
    if preverified_audit:
        _accumulate_source(accumulator, source, scale, preverified_audit=True)
    else:
        _accumulate_source(accumulator, source, scale)
    # Avoid lambda defaultdicts and torch shared-memory/file-descriptor traffic.
    groups = {}
    for key, group in accumulator.groups.items():
        groups[key] = {**group, 'rows': {name: dict(rows) for name, rows in group['rows'].items()},
                       'acceptance_by_block_position': dict(group['acceptance_by_block_position']),
                       'h_sum': group['h_sum'].tolist(), 'h_sq_sum': group['h_sq_sum'].tolist()}
    return {'source': source, 'groups': groups, 'generations': accumulator.generations}


def _merge_shard_result(accumulator, result, source, scale):
    if result['source'] != source:
        raise RuntimeError('Diagnostic worker returned a different audited source')
    # Normally one complete generation owns each prompt/seed/condition shard.
    # For overlapping input shards, replay the original blocks before merging
    # anything: adding pre-summed metric totals would change floating-add order.
    for key, group in result['groups'].items():
        previous = accumulator.groups.get(key)
        if previous is not None and any(
                prompt_seed in previous['rows'].get(metric, {})
                for metric, rows in group['rows'].items() for prompt_seed in rows):
            _accumulate_source(accumulator, source, scale)
            return
    accumulator.generations.extend(result['generations'])
    for key, incoming in result['groups'].items():
        incoming['h_sum'] = torch.tensor(incoming['h_sum'], dtype=torch.float64)
        incoming['h_sq_sum'] = torch.tensor(incoming['h_sq_sum'], dtype=torch.float64)
        group = accumulator.groups.get(key)
        if group is None:
            incoming['rows'] = defaultdict(lambda: defaultdict(lambda: [0., 0]),
                {metric: defaultdict(lambda: [0., 0], rows) for metric, rows in incoming['rows'].items()})
            incoming['acceptance_by_block_position'] = defaultdict(list, incoming['acceptance_by_block_position'])
            accumulator.groups[key] = incoming
            continue
        for field in ('n', 'h_sum', 'h_sq_sum', 'outlier_slots', 'outlier_coordinates'):
            group[field] += incoming[field]
        for field in ('h_norms', 'accepted_length', 'first_rejection'):
            group[field].extend(incoming[field])
        for metric, rows in incoming['rows'].items():
            group['rows'][metric].update(rows)
        for position, values in incoming['acceptance_by_block_position'].items():
            group['acceptance_by_block_position'][position].extend(values)


def accumulate_trace_shards(sources, scale, context_edges=DEFAULT_CONTEXT_EDGES,
                            outlier_threshold=5., seed=2026, workers=1, *, preverified_audit: bool = False):
    """Load only caller-selected sources, with at most `workers` pending shards.

    Merge in source order and bootstrap once after all shards are merged. The
    bounded submission window also bounds completed-but-not-yet-merged results.
    """
    if workers < 1:
        raise ValueError('diagnostic workers must be positive')
    accumulator = DiagnosticAccumulator(context_edges, outlier_threshold, seed)
    if workers == 1:
        for source in sources:
            if preverified_audit:
                _accumulate_source(accumulator, source, scale, preverified_audit=True)
            else:
                _accumulate_source(accumulator, source, scale)
        return accumulator
    remaining = iter(sources)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_diagnostic_worker_init) as pool:
        pending = deque()
        def submit_next():
            source = next(remaining, None)
            if source is not None:
                if preverified_audit:
                    future = pool.submit(_diagnostic_shard_worker, source, scale, context_edges, outlier_threshold, seed, True)
                else:
                    future = pool.submit(_diagnostic_shard_worker, source, scale, context_edges, outlier_threshold, seed)
                pending.append((source, future))
        for _ in range(workers):
            submit_next()
        while pending:
            source, future = pending.popleft()
            result = future.result()  # Worker failures propagate; no partial report is published.
            _merge_shard_result(accumulator, result, source, scale)
            del result, future
            submit_next()
    return accumulator


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument('--trace_dir', type=Path)
    parser.add_argument('--m1_audit', type=Path)
    parser.add_argument('--context_edges', type=int, nargs='+', default=list(DEFAULT_CONTEXT_EDGES))
    parser.add_argument('--outlier_threshold', type=float, default=5.)
    parser.add_argument('--workers', type=int, default=None)
    args = parse_config(parser, argv)
    auto_workers = args.workers is None
    if auto_workers:
        args.workers = 1 if args.scale == 'smoke' else min(16, max(1, os.cpu_count() or 1))
    if args.workers < 1:
        parser.error('--workers must be positive')
    record = record_run(args, 'diagnose')
    torch.set_num_threads(1)
    if args.scale == 'smoke':
        report = smoke_diagnostic(args)
        sources = []
        audit_provenance = None
    else:
        from .runtime import preflight
        if args.trace_dir is None:
            parser.error('Reference/full diagnosis requires --trace_dir from the completed M1a collection')
        accepted, audit_provenance = audited_trace_sources(
            args.m1_audit or args.output / 'm1_audit.json', args.scale, args.trace_dir)
        preflight(args, 0)
        run_workers = min(args.workers, max(1, len(accepted))) if auto_workers else args.workers
        accumulator = accumulate_trace_shards(accepted, args.scale, args.context_edges,
                                             args.outlier_threshold, args.seed, run_workers,
                                             preverified_audit=True)
        sources = [{'path': source['path'], 'sha256': source['sha256']} for source in accepted]
        report = accumulator.report()
    report.update({'label': 'synthetic' if args.scale == 'smoke' else args.scale, 'scale': args.scale,
                   'status': 'measured', 'run': record, 'source_shards': sources,
                   'audit_provenance': audit_provenance})
    path = args.output / f'diagnostic_{args.scale}.json'
    atomic_json(path, report)
    identity = {'model_pair': [args.target, args.drafter], 'prompt': fingerprint(sources),
                'condition': 'separate_conditions_in_report', 'seed': args.seed, 'temperature': 'separate_temperatures',
                'verification_rule': 'ratio_stochastic_greedy_argmax', 'proposal_mode': 'plain_dflash',
                'checkpoint': fingerprint(record['config']), 'adapter_version': 0}
    Manifest(args.output / 'diagnostic_manifest.json').finish(identity, path)
    print(json.dumps({'status': 'measured', 'path': str(path), 'groups': len(report['groups']), 'scale': args.scale}))


if __name__ == '__main__':
    main()
