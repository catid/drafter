"""M3/M4/M5 dev-only selection, frozen evaluation, and isolated matched-arm timing."""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid
import torch

from .adapter import DenseAdapter, LookaheadConfig
from .diagnose import (prompt_bootstrap, output_bin, context_bin, generation_diagnostics,
                       DEFAULT_CONTEXT_EDGES, OUTPUT_LABELS)
from .features import HiddenProjection
from .online import OnlineBigram, OnlineState
from .runtime import (ROOT, Manifest, atomic_json, call_with_output_dir, common_parser, fingerprint, parse_config,
                      preflight, record_run, resolve_artifact_path, resolved_config)
from .sampler import SamplingWarper

M3_ARMS = ('plain_dflash', 'online_bigram', 'dense', 'dense_sparse')
M4_ARMS = ('plain_dflash', 'online_bigram', 'offline_static', 'offline_online_stride1', 'online_dbS')
M5_ARMS = (
    'plain_dflash',
    'online_bigram',
    'online_bigram_count_gated',
    'offline_online_stride1',
    'zero_offline_online_only',
    'online_dbS',
    'online_dbS_zero_init',
    'dense_full',
    'dense_candidate',
    'lookahead_static_candidate',
    'lookahead_rerelax_candidate',
    'lookahead_static_full',
    'lookahead_rerelax_full',
    'signed_feature_hash_online',
    'exact_observed_pairs_online',
    'offline_online_stride5',
    'online_buffered_stride5',
    'online_two_sided_pl',
    'token_recycling_tree',
    'tts',
    'sharp_draft',
    'adaptive_block_length',
)
M5_EXTRA_ARMS = (
    'online_bigram_kret256',
    'online_bigram_kret128',
)

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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


ALGORITHM_FILES = (
    'adapter.py', 'online.py', 'features.py', 'dflash_loop.py',
    'sampler.py', 'verify.py', 'backend.py', 'sparse.py', 'toy.py', 'diagnose.py',
)
IMPLEMENTATION_FILES = (
    *ALGORITHM_FILES, 'eval_closed_loop.py', 'fastpath.py', 'optimizers.py',
)


def algorithm_file_hashes(source_dir=None):
    base = Path(__file__).parent if source_dir is None else Path(source_dir)
    return {name: file_sha256(base / name) for name in ALGORITHM_FILES}


def algorithm_hash(source_dir=None):
    base = Path(__file__).parent if source_dir is None else Path(source_dir)
    digest = hashlib.sha256()
    for name in ALGORITHM_FILES:
        path = base / name
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def implementation_hash(stage: str = 'm3', source_dir=None):
    base = Path(__file__).parent if source_dir is None else Path(source_dir)
    names = IMPLEMENTATION_FILES
    if stage == 'm5':
        names = names + ('m5_baselines.py', 'triton_ops.py')
    optional_when_external = {'fastpath.py', 'optimizers.py', 'm5_baselines.py', 'triton_ops.py'}
    digest = hashlib.sha256()
    for name in names:
        path = base / name
        if source_dir is not None and name in optional_when_external and not path.exists():
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def gpu_cpu_affinity(gpu: int, total_cpus: int | None = None) -> dict:
    """Deterministic NUMA-local 16-physical-core pinning on 128-core 8-GPU hosts."""
    if int(gpu) < 0:
        raise ValueError('gpu index must be non-negative')
    gpu = int(gpu)
    n_cpus = (os.cpu_count() or 1) if total_cpus is None else int(total_cpus)
    if n_cpus < 1:
        raise ValueError('total_cpus must be positive')
    if n_cpus >= 128 and gpu < 8:
        cpu_affinity = list(range(16 * gpu, 16 * gpu + 16))
        numa_node = gpu // 2
    else:
        width = max(1, min(16, n_cpus // 8)) if n_cpus >= 8 else 1
        start = (gpu * width) % n_cpus
        cpu_affinity = sorted({(start + i) % n_cpus for i in range(width)})
        numa_node = (gpu // 2) if n_cpus >= 8 else 0
    return {'physical_gpu': gpu, 'numa_node': numa_node, 'cpu_affinity': cpu_affinity}


def resolve_physical_gpu(device: torch.device, fallback_rank: int = 0) -> int | None:
    if device.type != 'cuda':
        return None
    logical_index = device.index or 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.strip():
        tokens = [token.strip() for token in visible.split(',') if token.strip()]
        if logical_index >= len(tokens):
            raise ValueError(f'device index {logical_index} out of range for CUDA_VISIBLE_DEVICES={visible!r}')
        selector = tokens[logical_index]
        if selector.isdigit():
            return int(selector)
        rows = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True
        ).splitlines()
        for row in rows:
            parts = [part.strip() for part in row.split(',')]
            if len(parts) == 2 and parts[1] == selector:
                return int(parts[0])
        raise RuntimeError(f'Unable to resolve physical GPU index for selector {selector!r}')
    return int(logical_index if device.index is not None else fallback_rank)


def apply_gpu_cpu_affinity(device: torch.device, rank: int = 0) -> dict:
    """Pin CUDA workers to NUMA-local physical cores; record metadata without mutating CPU test affinity."""
    if device.type == 'cuda':
        phys_gpu = resolve_physical_gpu(device, fallback_rank=rank)
        info = gpu_cpu_affinity(phys_gpu)
        applied = False
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, set(info['cpu_affinity']))
            actual = sorted(os.sched_getaffinity(0))
            if actual != info['cpu_affinity']:
                raise RuntimeError(f'CPU affinity mismatch after sched_setaffinity: expected {info["cpu_affinity"]}, got {actual}')
            applied = True
        return {**info, 'cpu_affinity_applied': applied}
    info = gpu_cpu_affinity(rank)
    return {**info, 'cpu_affinity_applied': False}


def assert_all_gpus_idle(device: torch.device) -> dict | None:
    """Verify every GPU on the host is completely idle before isolated 1-GPU calibration model load."""
    if device.type != 'cuda':
        return None
    phys_gpu = resolve_physical_gpu(device, fallback_rank=0)
    if phys_gpu != 0:
        raise RuntimeError(f'1-GPU isolated calibration must run on physical GPU 0, got GPU {phys_gpu}')
    query = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,name,compute_cap',
         '--format=csv,noheader,nounits'],
        text=True,
    ).splitlines()
    if not query:
        raise RuntimeError('nvidia-smi returned no GPUs during cluster isolation check')
    gpus = []
    for line in query:
        idx, uuid_str, mem, util, name, cap = [part.strip() for part in line.split(',')]
        mem_i, util_i = int(mem), int(util)
        if mem_i > 128 or util_i > 2:
            raise RuntimeError(f'Cluster GPU {idx} is not idle before 1-GPU calibration: {line}')
        gpus.append({'index': int(idx), 'uuid': uuid_str, 'initial_used_MiB': mem_i,
                     'initial_utilization_percent': util_i, 'name': name, 'compute_capability': cap})
    procs = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True
    ).splitlines()
    external = []
    for line in procs:
        fields = [field.strip() for field in line.split(',') if field.strip()]
        if len(fields) == 2 and int(fields[1]) != os.getpid():
            external.append(f'{fields[0]}:pid={fields[1]}')
    if external:
        raise RuntimeError('Cluster has external GPU compute processes before calibration: ' + ', '.join(external))
    return {'gpu_count': len(gpus), 'all_gpus_idle_before_load': True, 'gpus': gpus}


def assert_no_external_cluster_gpu_processes(device: torch.device) -> None:
    """Verify GPUs 1..N-1 remain completely idle and GPU 0 has only this PID around each calibration bundle."""
    if device.type != 'cuda':
        return
    phys_gpu = resolve_physical_gpu(device, fallback_rank=0)
    query = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True,
    ).splitlines()
    for line in query:
        idx_s, _, mem_s, util_s = [part.strip() for part in line.split(',')]
        idx, mem_i, util_i = int(idx_s), int(mem_s), int(util_s)
        if idx != phys_gpu and (mem_i > 128 or util_i > 2):
            raise RuntimeError(f'Sibling GPU {idx} became active during 1-GPU isolated calibration: {line}')
    procs = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True
    ).splitlines()
    external = []
    for line in procs:
        fields = [field.strip() for field in line.split(',') if field.strip()]
        if len(fields) == 2 and int(fields[1]) != os.getpid():
            external.append(f'{fields[0]}:pid={fields[1]}')
    if external:
        raise RuntimeError('External GPU compute processes appeared on cluster during calibration: ' + ', '.join(external))


def adapter_tensor_hash(adapter):
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def arms_for(stage, scale='reference', *, m5_arms=None, include_long_oracle=False):
    if stage == 'm3':
        return M3_ARMS
    if stage == 'm4':
        return M4_ARMS + (('offline_online_stride5', 'zero_offline_online_only',
                           'online_rejection_relative', 'online_cutoff') if scale == 'full' else ())
    if stage != 'm5':
        raise ValueError('stage must be m3, m4, or m5')
    default_m5 = M5_ARMS + (('long_oracle_offline',) if include_long_oracle else ())
    if not m5_arms:
        return default_m5
    available = default_m5 + M5_EXTRA_ARMS
    requested = tuple(dict.fromkeys(m5_arms))
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f'unknown M5 arm(s): {sorted(unknown)}')
    if 'plain_dflash' not in requested:
        requested = ('plain_dflash',) + requested
    return requested


def _arms_from_args(args):
    return arms_for(
        args.stage,
        args.scale,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )


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


def evaluation_jobs(prompts, stage, scale, seed=2026, *, m5_arms=None, include_long_oracle=False):
    jobs = []
    active_arms = arms_for(stage, scale, m5_arms=m5_arms, include_long_oracle=include_long_oracle)
    for prompt in prompts:
        for name, temperature, warp in sampling_conditions(prompt['condition']):
            seeds = 1 if temperature == 0 else (3 if prompt['condition'] == 'standard_nonthinking' or scale == 'full' else 1)
            if scale == 'smoke':
                seeds = 1
            for index in range(seeds):
                for arm in active_arms:
                    jobs.append({'prompt': prompt, 'arm': arm, 'sampling_name': name,
                                 'temperature': temperature, 'warper': warp, 'seed': seed + index})
    return jobs


def candidate_settings(
    arm,
    *,
    bigram_count_gate_kappa=None,
    bigram_k_ret=None,
    m5_beta_candidates=None,
    m5_beta_relax=None,
):
    if arm == 'plain_dflash':
        return [{}]
    if arm == 'online_bigram':
        return [{'lambda_tr': .05, 'alpha': .5, 'k_ret': None}, {'lambda_tr': .2, 'alpha': 1., 'k_ret': None}]
    if arm == 'online_bigram_kret256':
        kr = 256 if bigram_k_ret is None else int(bigram_k_ret)
        if kr < 1:
            raise ValueError('bigram_k_ret must be a positive integer')
        return [{'lambda_tr': .05, 'alpha': .5, 'k_ret': kr}, {'lambda_tr': .2, 'alpha': 1., 'k_ret': kr}]
    if arm == 'online_bigram_kret128':
        return [{'lambda_tr': .05, 'alpha': .5, 'k_ret': 128}, {'lambda_tr': .2, 'alpha': 1., 'k_ret': 128}]
    if arm == 'online_bigram_count_gated':
        if bigram_count_gate_kappa is not None:
            kappa = float(bigram_count_gate_kappa)
            return [{'lambda_tr': .15, 'alpha': 1., 'k_ret': None, 'count_gate_kappa': kappa},
                    {'lambda_tr': .25, 'alpha': 1., 'k_ret': None, 'count_gate_kappa': kappa}]
        return [{'lambda_tr': .15, 'alpha': 1., 'k_ret': None, 'count_gate_kappa': 8.},
                {'lambda_tr': .25, 'alpha': 1., 'k_ret': None, 'count_gate_kappa': 16.}]
    if arm in ('dense', 'dense_sparse', 'offline_static'):
        return [{'beta': .5}, {'beta': 1.}]
    b0, b1 = ((float(m5_beta_candidates[0]), float(m5_beta_candidates[1]))
              if m5_beta_candidates is not None else (.5, 1.))
    br = 1. if m5_beta_relax is None else float(m5_beta_relax)
    if arm in ('dense_full', 'long_oracle_offline'):
        return [{'beta': b0}, {'beta': b1}]
    if arm == 'dense_candidate':
        return [{'beta': b0, 'beta_relax': br}, {'beta': b1, 'beta_relax': br}]
    if arm in ('lookahead_static_candidate', 'lookahead_rerelax_candidate',
               'lookahead_static_full', 'lookahead_rerelax_full'):
        return [{'beta': b0, 'beta_relax': br, 'R_init': 2, 'R_clamp': 1, 'eta': .5, 'T_relax': 1.},
                {'beta': b1, 'beta_relax': br, 'R_init': 2, 'R_clamp': 1, 'eta': .5, 'T_relax': 1.}]
    if arm == 'online_buffered_stride5':
        return [{'beta': 1., 'lr': lr, 'update_stride': 5, 'lambda_l2': l2,
                 'lambda_l1_b': 1e-6, 'lambda_l1_s': 1e-5, 'weighting_mode': 'exponential',
                 'buffer_reduction': 'mean'}
                for lr, l2 in ((1e-4, .01), (1e-3, .1))]
    if arm == 'online_two_sided_pl':
        return [{'beta': 1., 'lr': lr, 'update_stride': 1, 'lambda_l2': l2,
                 'lambda_l1_b': 1e-6, 'lambda_l1_s': 1e-5}
                for lr, l2 in ((1e-4, .01), (1e-3, .1))]
    if arm == 'token_recycling_tree':
        return [{'max_successors': 4, 'branch_widths': [2, 2, 1, 1]},
                {'max_successors': 8, 'branch_widths': [3, 2, 1, 1]}]
    if arm == 'tts':
        return [{'lr': 5e-3, 'update_stride': 1, 'lambda_l2': .01, 'lambda_l1_b': 1e-5},
                {'lr': 2e-2, 'update_stride': 1, 'lambda_l2': .05, 'lambda_l1_b': 1e-5}]
    if arm == 'sharp_draft':
        return [{'gamma_max': 1.25, 'gamma_min': .95, 'entropy_pivot': .35},
                {'gamma_max': 1.45, 'gamma_min': .9, 'entropy_pivot': .4}]
    if arm == 'adaptive_block_length':
        return [{'beta': 1., 'min_block_length': 2, 'policy': 'hybrid',
                 'ema_decay': .7, 'ema_multiplier': 1.5, 'ema_margin': 1., 'min_cumulative_confidence': .08},
                {'beta': 1., 'min_block_length': 1, 'policy': 'hybrid',
                 'ema_decay': .7, 'ema_multiplier': 1.25, 'ema_margin': 1., 'min_cumulative_confidence': .15}]
    stride = 5 if arm == 'offline_online_stride5' else 1
    weighting = 'rejection_relative' if arm == 'online_rejection_relative' else 'cutoff' if arm == 'online_cutoff' else 'exponential'
    return [{'beta': 1., 'lr': lr, 'update_stride': stride, 'lambda_l2': l2,
             'lambda_l1_b': 1e-6, 'lambda_l1_s': 1e-5, 'weighting_mode': weighting}
            for lr, l2 in ((1e-4, .01), (1e-3, .1))]


def validate_checkpoint_hashes(expected_hashes, current_hashes):
    expected_non_impl = {k: v for k, v in expected_hashes.items() if k != 'implementation'}
    current_non_impl = {k: v for k, v in current_hashes.items() if k != 'implementation'}
    if expected_non_impl != current_non_impl:
        raise ValueError('checkpoint/projection changed since dev tuning; freeze new settings')
    expected_impl = expected_hashes.get('implementation')
    current_impl = current_hashes.get('implementation')
    if expected_impl == current_impl:
        return
    if (expected_impl == FROZEN_09A5469_IMPLEMENTATION_HASH
            and algorithm_hash() == FROZEN_09A5469_ALGORITHM_HASH
            and algorithm_file_hashes() == FROZEN_09A5469_ALGORITHM_FILES):
        return
    raise ValueError('checkpoint/projection changed since dev tuning; freeze new settings')


def validate_frozen_settings(settings, stage, scale, checkpoint_hashes, prompt_manifest, generation_caps=None,
                             model_pair=None, model_revisions=None, generation_environment=None, *,
                             m5_arms=None, include_long_oracle=False):
    if not settings.get('frozen') or settings.get('stage') != stage or settings.get('scale') != scale:
        raise ValueError('test evaluation requires frozen settings for this stage and scale')
    validate_checkpoint_hashes(settings['checkpoint_hashes'], checkpoint_hashes)
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
    expected_arms = arms_for(stage, scale, m5_arms=m5_arms, include_long_oracle=include_long_oracle)
    if set(settings['selected']) != set(expected_arms):
        raise ValueError('frozen settings omit a required arm')
    if stage == 'm5':
        matched_c0_arms = [
            a for a in (
                'dense_candidate', 'lookahead_static_candidate', 'lookahead_rerelax_candidate',
                'lookahead_static_full', 'lookahead_rerelax_full',
            )
            if a in settings['selected']
        ]
        if matched_c0_arms:
            br_vals = []
            for a in matched_c0_arms:
                raw_br = settings['selected'][a].get('beta_relax', 1.)
                if (isinstance(raw_br, bool) or not isinstance(raw_br, (int, float))
                        or not math.isfinite(float(raw_br)) or float(raw_br) < 0.):
                    raise ValueError(f'frozen setting beta_relax for {a} must be finite and >= 0')
                br_vals.append(float(raw_br))
            if len(set(br_vals)) > 1:
                raise ValueError('matched-support candidate arms must share identical beta_relax so frozen C0[j] matches')


def _cli_or_env_fast_decode(args) -> bool:
    return bool(getattr(args, 'fast_decode', False) or os.environ.get('EDRAFT_FAST_DECODE', '0') == '1')


class EvaluationRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.affinity_info = apply_gpu_cpu_affinity(self.device, getattr(args, 'rank', 0))
        self.tiny = args.scale == 'smoke'
        self.long_oracle = None
        self.triton_gate_report = None
        self.torch_compile_gate_report = None
        self.m5_active_kernels = {}
        self.is_m5_custom_checkpoint = False
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
                                      'projection': 'synthetic-calibration-seed-' + str(args.seed),
                                      'implementation': implementation_hash(args.stage)}
            if args.stage == 'm5' and getattr(args, 'm5_oracle_checkpoint', None):
                from .train import load_adapter_checkpoint
                self.long_oracle, oracle_meta = load_adapter_checkpoint(args.m5_oracle_checkpoint, args.device)
                if oracle_meta.get('report', {}).get('training_distribution') != 'long_oracle_m5':
                    raise ValueError('m5_oracle_checkpoint must have training_distribution=="long_oracle_m5"')
                self.checkpoint_hashes['long_oracle'] = file_sha256(args.m5_oracle_checkpoint)
            self.model_pair = ['builtin_toy_target', 'builtin_parallel_drafter']
            self.model_revisions = ['synthetic-v1', 'synthetic-v1']
        else:
            from .train import load_adapter_checkpoint
            from .backend import ReferenceBackend
            from .download import fetch_model
            m2 = json.loads((args.output / 'm2.json').read_text())
            if m2.get('status') != 'passed' or m2.get('scale', args.scale) != args.scale:
                raise RuntimeError('M2 reference acceptance gate must pass before M3/M4/M5')
            if args.stage == 'm4':
                prior = json.loads((args.output / 'm3' / 'summary.json').read_text())
                if prior.get('status') != 'complete' or prior.get('scale') != args.scale:
                    raise RuntimeError('M3 must complete before M4')
            elif args.stage == 'm5':
                for prior_stage in ('m3', 'm4'):
                    prior = json.loads((args.output / prior_stage / 'summary.json').read_text())
                    if prior.get('status') != 'complete' or prior.get('scale') != args.scale:
                        raise RuntimeError(f'{prior_stage.upper()} must complete before M5')
            self.checkpoint_hashes = checkpoint_hashes(args)
            self.dense, self.dense_metadata = load_adapter_checkpoint(args.dense_checkpoint, args.device)
            self.dense_sparse, self.sparse_metadata = load_adapter_checkpoint(args.sparse_checkpoint, args.device)
            self.projection = HiddenProjection.load(args.projection).to(args.device)
            if self.dense.W.shape[1] != self.projection.matrix.shape[0]:
                raise ValueError('checkpoint and frozen hidden projection dimensions differ')
            target = call_with_output_dir(fetch_model, args.target, args.allow_download, output_dir=args.output)
            drafter = call_with_output_dir(fetch_model, args.drafter, args.allow_download, output_dir=args.output)
            env = json.loads(resolve_artifact_path('environment.json', args.output, root=ROOT).read_text())
            self.model_revisions = [env['model_artifacts'][name]['revision'] for name in (args.target, args.drafter)]
            for label, metadata in [('dense', self.dense_metadata), ('dense_sparse', self.sparse_metadata)]:
                report = metadata.get('report', {})
                if args.stage == 'm5':
                    if metadata.get('stage') != label:
                        raise ValueError('selected M5 checkpoint is not the corresponding fit stage')
                elif metadata.get('identity') != m2.get('identity') or metadata.get('stage') != label:
                    raise ValueError('selected checkpoint is not the corresponding stage of the accepted M2 fit')
                if report.get('model_pair') != [args.target, args.drafter] or report.get('model_revisions') != self.model_revisions:
                    raise ValueError('checkpoint training models/revisions do not match evaluation backend')
                if report.get('frozen_artifacts', {}).get('hidden_projection_sha256') != self.checkpoint_hashes['projection']:
                    raise ValueError('checkpoint was fitted with a different hidden projection')
            if args.stage == 'm5':
                if self.dense_metadata.get('identity') != self.sparse_metadata.get('identity'):
                    raise ValueError('M5 dense and dense_sparse checkpoints must originate from the same fit identity')
                dense_cal = bool(self.dense_metadata.get('report', {}).get('calibrated_stage'))
                sparse_cal = bool(self.sparse_metadata.get('report', {}).get('calibrated_stage'))
                if dense_cal != sparse_cal:
                    raise ValueError('M5 dense and dense_sparse checkpoints cannot mix calibrated and uncalibrated variants')
                self.is_m5_custom_checkpoint = (
                    self.dense_metadata.get('identity') != m2.get('identity')
                    or bool(self.dense_metadata.get('non_unit_scaled', False))
                    or float(self.dense_metadata.get('checkpoint_scale', 1.0)) != 1.0
                )
            if args.stage == 'm5' and getattr(args, 'm5_oracle_checkpoint', None):
                self.long_oracle, oracle_meta = load_adapter_checkpoint(args.m5_oracle_checkpoint, args.device)
                oracle_report = oracle_meta.get('report', {})
                if oracle_report.get('training_distribution') != 'long_oracle_m5':
                    raise ValueError('m5_oracle_checkpoint must have training_distribution=="long_oracle_m5"')
                if oracle_report.get('model_pair') != [args.target, args.drafter] or oracle_report.get('model_revisions') != self.model_revisions:
                    raise ValueError('long_oracle checkpoint model pair/revisions do not match evaluation backend')
            self.backend = ReferenceBackend.from_pretrained(
                target, drafter, device=args.device,
                target_revision=self.model_revisions[0], drafter_revision=self.model_revisions[1],
                fast_decode=True if _cli_or_env_fast_decode(args) else None,
            )
            self.model_pair = [args.target, args.drafter]
        if args.stage == 'm5' and getattr(args, 'checkpoint_scale', None) is not None:
            from .train import scale_adapter_inplace
            scale_val = float(args.checkpoint_scale)
            scale_adapter_inplace(self.dense, scale_val)
            scale_adapter_inplace(self.dense_sparse, scale_val)
            if scale_val != 1.0:
                self.is_m5_custom_checkpoint = True
            if self.tiny:
                self.checkpoint_hashes['dense'] = adapter_tensor_hash(self.dense)
                self.checkpoint_hashes['dense_sparse'] = adapter_tensor_hash(self.dense_sparse)
                self.checkpoint_hashes['checkpoint_scale'] = scale_val
        self.dense.eval()
        self.dense_sparse.eval()
        if self.long_oracle is not None:
            self.long_oracle.eval()
        from .fastpath import FastPathConfig, FastPathRuntime
        fp_cfg = FastPathConfig(
            compile=bool(getattr(args, 'compile', False)),
            cuda_graph=bool(getattr(args, 'cuda_graph', False)),
            simulate_cpu_graph=bool(getattr(args, 'simulate_cpu_graph', False)),
            bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
            compile_drafter=bool(getattr(args, 'compile_drafter', False)),
            compile_online=bool(getattr(args, 'compile_online', False)),
            analytic_online_grad=bool(getattr(args, 'analytic_online_grad', True)),
            fast_decode=_cli_or_env_fast_decode(args),
        )
        static_k = 4 if self.tiny else (self.backend.block_size - 1)
        self.fastpath = FastPathRuntime(fp_cfg, static_block_length=static_k) if fp_cfg.enabled else None
        if args.stage == 'm5' and getattr(args, 'm5_triton', False):
            from .triton_ops import build_gated_m5_kernels
            triton_fns, reports = build_gated_m5_kernels(
                device=self.device,
                vocab_size=self.dense.E.shape[0],
                sparse_capacity=257 if self.tiny else 2**20,
                M_sparse=self.dense.M_sparse,
                D=len(self.dense.A),
                K=4 if self.tiny else (self.backend.block_size - 1),
                return_callables=True,
            )
            self.triton_gate_report = {k: v.to_dict() for k, v in reports.items()}
            if triton_fns.get('signed_hash_lookup') is not None:
                self.m5_active_kernels['signed_hash_lookup'] = triton_fns['signed_hash_lookup']
                if self.dense_sparse.sparse is not None and hasattr(self.dense_sparse.sparse, 'lookup_fn'):
                    self.dense_sparse.sparse.lookup_fn = triton_fns['signed_hash_lookup']
                if self.long_oracle is not None and self.long_oracle.sparse is not None and hasattr(self.long_oracle.sparse, 'lookup_fn'):
                    self.long_oracle.sparse.lookup_fn = triton_fns['signed_hash_lookup']
            if triton_fns.get('rowwise_full_kl') is not None:
                self.m5_active_kernels['rowwise_full_kl'] = triton_fns['rowwise_full_kl']
        if args.stage == 'm5' and getattr(args, 'm5_torch_compile', False):
            from .triton_ops import build_gated_torch_compile
            tc_fns, tc_reports = build_gated_torch_compile(
                device=self.device,
                vocab_size=self.dense.E.shape[0],
                r=self.dense.E.shape[1],
                p=self.dense.W.shape[1],
                K=4 if self.tiny else (self.backend.block_size - 1),
                bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
                return_callables=True,
            )
            self.torch_compile_gate_report = {k: v.to_dict() for k, v in tc_reports.items()}
            if tc_fns.get('torch_compile_block_unary') is not None:
                unary_fn = tc_fns['torch_compile_block_unary']
                self.m5_active_kernels['torch_compile_block_unary'] = unary_fn
                self.dense.unary_fn = unary_fn
                self.dense_sparse.unary_fn = unary_fn
                if self.long_oracle is not None:
                    self.long_oracle.unary_fn = unary_fn
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
        # Greedy beta=1 is a fixed required arm, independent of stochastic dev beta
        # (unless an explicit beta_sweep overrides greedy_allow_beta).
        beta = (setting.get('beta', 1.)
                if (temperature > 0 or job.get('force_setting_beta', False))
                else 1.)
        beta_relax_val = float(setting.get('beta_relax', 1.)) if 'beta_relax' in setting else None
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        request_started = time.perf_counter()
        adapter, online, bigram = None, None, None
        proposal_mode = 'dense_full'
        lookahead = None
        proposal_controller = None
        adaptive_controller = None
        recycling_state = None
        inputs, max_tokens, input_metadata = self.prepare_inputs(prompt, job.get('max_new_tokens_override'))
        generator = torch.Generator(device=self.device).manual_seed(job['seed'])
        record_round_timing = args.stage == 'm5'

        if arm == 'token_recycling_tree':
            from .m5_baselines import generate_token_recycling_reference, generate_token_recycling_toy
            max_succ = int(setting.get('max_successors', 4 if self.tiny else 8))
            b_widths = tuple(int(w) for w in setting.get('branch_widths', [2, 2, 1, 1] if self.tiny else [4, 2, 1, 1]))
            if self.tiny:
                prefix = inputs
                result, recycling_state = generate_token_recycling_toy(
                    self.target, self.drafter, prefix, max_tokens,
                    temperature=temperature, warper=warper, generator=generator,
                    max_successors=max_succ, branch_widths=b_widths, measure_phases=profile,
                    record_round_timing=record_round_timing,
                )
                eos = False
            else:
                result, recycling_state = generate_token_recycling_reference(
                    self.backend, inputs, max_tokens,
                    temperature=temperature, warper=warper, generator=generator,
                    max_successors=max_succ, branch_widths=b_widths, measure_phases=profile,
                    record_round_timing=record_round_timing,
                )
                eos = bool(result.output_tokens and result.output_tokens[-1] in self.backend.stop_token_ids)
        else:
            if arm in ('online_bigram', 'online_bigram_kret256', 'online_bigram_kret128', 'online_bigram_count_gated'):
                bigram = OnlineBigram(self.dense.E.shape[0], **setting)
            elif arm == 'dense':
                adapter = self.dense
            elif arm in ('dense_sparse', 'offline_static', 'dense_full'):
                adapter = self.dense_sparse
            elif arm == 'long_oracle_offline':
                if self.long_oracle is None:
                    raise RuntimeError('long_oracle_offline requires --m5_oracle_checkpoint')
                adapter = self.long_oracle
            elif arm == 'dense_candidate':
                adapter = self.dense_sparse
                proposal_mode = 'dense_candidate'
                if beta_relax_val is None:
                    beta_relax_val = 1.0
            elif arm in ('lookahead_static_candidate', 'lookahead_rerelax_candidate',
                         'lookahead_static_full', 'lookahead_rerelax_full'):
                adapter = self.dense_sparse
                la_mode = 'static' if 'static' in arm else 'rerelax'
                la_domain = 'candidate_masked' if arm.endswith('_candidate') else 'full_vocabulary'
                proposal_mode = 'dense_candidate' if la_domain == 'candidate_masked' else 'dense_full'
                if beta_relax_val is None:
                    beta_relax_val = 1.0
                lookahead = LookaheadConfig(
                    mode=la_mode,
                    domain=la_domain,
                    R_init=int(setting.get('R_init', 2)),
                    R_clamp=int(setting.get('R_clamp', 1)),
                    eta=float(setting.get('eta', 0.5)),
                    T_relax=float(setting.get('T_relax', 1.0)),
                    beta_relax=beta_relax_val,
                )
            elif arm == 'tts':
                from .m5_baselines import TTSState
                proposal_controller = TTSState(
                    self.dense.E.shape[0], self.dense.W.shape[1],
                    max_slots=4 if self.tiny else 32, device=self.device, **setting,
                )
            elif arm == 'sharp_draft':
                from .m5_baselines import SharpDraftProposal
                proposal_controller = SharpDraftProposal(**setting)
            elif arm == 'adaptive_block_length':
                from .m5_baselines import AdaptiveBlockController
                adapter = self.dense_sparse
                max_blk = 4 if self.tiny else (self.backend.block_size - 1)
                ctrl_kwargs = {k: v for k, v in setting.items() if k != 'beta'}
                adaptive_controller = AdaptiveBlockController(max_block_length=max_blk, **ctrl_kwargs)
            elif arm != 'plain_dflash':
                offline = self.dense_sparse
                if arm in ('zero_offline_online_only', 'online_dbS_zero_init'):
                    offline = DenseAdapter(
                        self.dense.E, self.dense.W.shape[1], len(self.dense.A),
                        M_sparse=self.dense.M_sparse,
                        unary_fn=self.m5_active_kernels.get('torch_compile_block_unary') if args.stage == 'm5' else None,
                    )
                kwargs = {k: v for k, v in setting.items() if k not in ('beta', 'beta_relax')}
                if arm == 'exact_observed_pairs_online':
                    kwargs['sparse_backend'] = 'exact_observed_pairs'
                    kwargs.setdefault('sparse_overflow_policy', getattr(args, 'sparse_overflow_policy', 'freeze_new'))
                elif arm == 'signed_feature_hash_online':
                    kwargs['sparse_backend'] = 'signed_feature_hash'
                elif arm == 'online_buffered_stride5':
                    kwargs['accumulate_stride'] = True
                    kwargs.setdefault('update_stride', 5)
                elif arm == 'online_two_sided_pl':
                    kwargs['loss_mode'] = 'two_sided_pseudolikelihood'
                if args.stage == 'm5' and self.m5_active_kernels.get('rowwise_full_kl') is not None:
                    kwargs['kl_loss_fn'] = self.m5_active_kernels['rowwise_full_kl']
                enable_r3 = not bool(getattr(args, 'disable_round3_compact', False))
                analytic_grad = enable_r3 and (
                    self.fastpath.config.analytic_online_grad
                    if self.fastpath is not None
                    else bool(getattr(args, 'analytic_online_grad', True))
                )
                kwargs.setdefault('analytic_online_grad', analytic_grad)
                online = OnlineState(offline, enable_sparse=True, sparse_capacity=257 if self.tiny else 2**20,
                                     bias_sparse_only=arm in ('online_dbS', 'online_dbS_zero_init'),
                                     fastpath=self.fastpath, **kwargs)

            need_projection = (adapter is not None or online is not None
                               or (proposal_controller is not None and getattr(proposal_controller, 'needs_hidden', False)))
            proj = self.projection if need_projection else None
            enable_r3 = not bool(getattr(args, 'disable_round3_compact', False))
            if self.tiny:
                prefix = inputs
                result = generate_toy(self.target, self.drafter, prefix, max_tokens, temperature=temperature, warper=warper,
                    generator=generator, adapter=adapter, online_state=online, projection=proj,
                    bigram=bigram, beta=beta, beta_relax=beta_relax_val, measure_phases=profile,
                    fastpath=self.fastpath, proposal_mode=proposal_mode, lookahead=lookahead,
                    proposal_controller=proposal_controller, adaptive_controller=adaptive_controller,
                    record_round_timing=record_round_timing, enable_round3_compact=enable_r3)
                eos = False
            else:
                result = generate_reference(self.backend, inputs, max_tokens, temperature=temperature, warper=warper,
                    generator=generator, adapter=adapter, online_state=online, projection=proj,
                    bigram=bigram, beta=beta, beta_relax=beta_relax_val, measure_phases=profile,
                    fastpath=self.fastpath, proposal_mode=proposal_mode, lookahead=lookahead,
                    proposal_controller=proposal_controller, adaptive_controller=adaptive_controller,
                    record_round_timing=record_round_timing, enable_round3_compact=enable_r3)
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
                                 'executed_updates': online.u,
                                 'single_round_no_buffering': not online.accumulate_stride}
            if online.accumulate_stride:
                online_invariants['buffered_stride_accumulation'] = True
        lengths = [block['accepted_length'] for block in result.blocks]
        generation = generation_diagnostics(result.output_tokens)
        generation.update({'eos_reached': eos, 'capped': result.num_output_tokens == max_tokens, 'complete': True})
        online_metrics = online.metrics() if online else None
        update_ms = result.phase_ms.get('online_update', 0.)
        state_bandwidth = online_metrics['state_bytes_estimate_total'] / (update_ms * 1e6) if online_metrics is not None and update_ms > 0 else None
        base_adapter = adapter if adapter is not None else (online.offline if online is not None else None)
        row = {'scale': args.scale, 'label': 'synthetic' if self.tiny else args.scale,
            'stage': args.stage, 'arm': arm, 'prompt_id': prompt['prompt_id'], 'split': prompt['split'],
            'condition': prompt['condition'], 'seed': job['seed'], 'sampling_name': job['sampling_name'],
            'temperature': temperature, 'sampling_warp_config': asdict(warper), 'settings': setting, 'effective_beta': beta,
            'verification_rule': 'ratio' if temperature > 0 else 'greedy',
            'model_pair': self.model_pair, 'model_revisions': self.model_revisions,
            'input_metadata': input_metadata,
            'checkpoint_hashes': self.checkpoint_hashes,
            'offline_fit_provenance': ('M5 long-condition oracle checkpoint' if arm == 'long_oracle_offline'
                                       else 'four built-in synthetic calibration-train prefixes' if self.tiny
                                       else 'M5 custom/calibrated fit checkpoints' if self.is_m5_custom_checkpoint
                                       else 'gated M2 checkpoints'),
            'generated_tokens': result.num_output_tokens,
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
            'parameter_norms': {'W': float(base_adapter.W.detach().norm()),
                                'A': float(base_adapter.A.detach().norm()),
                                'b': float(base_adapter.b.detach().norm())} if base_adapter is not None else {},
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None,
            'phase_profile': profile}
        if args.stage == 'm5':
            m5_telemetry = {'active_dispatched_kernels': sorted(self.m5_active_kernels.keys())}
            if lookahead is not None:
                m5_telemetry['lookahead'] = asdict(lookahead)
            elif beta_relax_val is not None and proposal_mode == 'dense_candidate':
                m5_telemetry['matched_c0_beta_relax'] = beta_relax_val
            if recycling_state is not None:
                m5_telemetry['token_recycling'] = recycling_state.metrics()
            if proposal_controller is not None and hasattr(proposal_controller, 'metrics'):
                m5_telemetry[arm] = proposal_controller.metrics()
            if adaptive_controller is not None:
                m5_telemetry['adaptive_block'] = adaptive_controller.metrics()
            if self.triton_gate_report is not None:
                m5_telemetry['triton_gate'] = self.triton_gate_report
            if self.torch_compile_gate_report is not None:
                m5_telemetry['torch_compile_gate'] = self.torch_compile_gate_report
            row['m5_telemetry'] = m5_telemetry
        return row


def checkpoint_hashes(args):
    hashes = {'dense': file_sha256(args.dense_checkpoint), 'dense_sparse': file_sha256(args.sparse_checkpoint),
              'projection': file_sha256(args.projection), 'implementation': implementation_hash(args.stage)}
    if args.stage == 'm5' and getattr(args, 'm5_oracle_checkpoint', None):
        hashes['long_oracle'] = file_sha256(args.m5_oracle_checkpoint)
    if args.stage == 'm5' and getattr(args, 'checkpoint_scale', None) is not None:
        hashes['checkpoint_scale'] = float(args.checkpoint_scale)
    return hashes


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
                'timing_isolated': mode in ('timing', 'timing_calibration'), 'rank': args.rank})
    if mode in ('timing', 'timing_calibration'):
        ws = getattr(args, 'world_size', 1)
        rk = getattr(args, 'rank', 0)
        affinity = getattr(runner, 'affinity_info', None) or apply_gpu_cpu_affinity(torch.device('cpu'), rk)
        sharding_mode = (
            'single_gpu_calibration' if mode == 'timing_calibration'
            else (f'sharded_{ws}gpu' if ws > 1 else 'single_gpu')
        )
        row.update({
            'world_size': ws,
            'sharding_mode': sharding_mode,
            'physical_gpu': affinity['physical_gpu'],
            'numa_node': affinity['numa_node'],
            'cpu_affinity': affinity['cpu_affinity'],
            'cpu_affinity_applied': affinity['cpu_affinity_applied'],
            'timing_harness_implementation_sha256': implementation_hash(args.stage),
            'algorithm_implementation_sha256': algorithm_hash(),
        })
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
        logical = getattr(args, 'logical_session', None) or uuid.uuid4().hex
        state = {'logical_session': logical, 'configuration': config, 'created_at_unix': time.time(), 'epochs': [], 'bundles': {}}
    if mode == 'tune' and args.stage == 'm5':
        if state.get('triton_gate_report') is None and getattr(runner, 'triton_gate_report', None) is not None:
            state['triton_gate_report'] = runner.triton_gate_report
        if state.get('torch_compile_gate_report') is None and getattr(runner, 'torch_compile_gate_report', None) is not None:
            state['torch_compile_gate_report'] = runner.torch_compile_gate_report
    rk = getattr(args, 'rank', 0)
    ws = getattr(args, 'world_size', 1)
    affinity = getattr(runner, 'affinity_info', None) or apply_gpu_cpu_affinity(torch.device('cpu'), rk)
    if mode in ('timing', 'timing_calibration'):
        state.update({
            'rank': rk,
            'world_size': ws,
            'sharding_mode': 'single_gpu_calibration' if mode == 'timing_calibration' else 'single_gpu',
            'physical_gpu': affinity['physical_gpu'],
            'numa_node': affinity['numa_node'],
            'cpu_affinity': affinity['cpu_affinity'],
            'cpu_affinity_applied': affinity['cpu_affinity_applied'],
            'timing_harness_implementation_sha256': implementation_hash(args.stage),
            'algorithm_implementation_sha256': algorithm_hash(),
        })
    epoch = uuid.uuid4().hex
    state['epochs'].append({'epoch': epoch, 'started_at_unix': time.time(), 'device': str(runner.device),
                            'physical_gpu': affinity['physical_gpu'], 'numa_node': affinity['numa_node'],
                            'cpu_affinity': affinity['cpu_affinity']})
    atomic_json(path, state)
    return path, state, epoch


def sharded_timing_session(runner, logical_session, configuration, plan_path, plan_sha256):
    args = runner.args
    if getattr(runner, 'fastpath', None) is not None:
        configuration = dict(configuration, fastpath=runner.fastpath.config.to_dict())
    affinity = getattr(runner, 'affinity_info', None) or apply_gpu_cpu_affinity(torch.device('cpu'), args.rank)
    config = {
        'configuration': configuration,
        'checkpoint_hashes': runner.checkpoint_hashes,
        'model_pair': runner.model_pair,
        'model_revisions': runner.model_revisions,
        'generation_environment': getattr(runner, 'generation_environment', None),
        'generation_caps': generation_caps(args),
        'stage': args.stage,
        'scale': args.scale,
        'device': str(runner.device),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'rank': args.rank,
        'world_size': args.world_size,
        'physical_gpu': affinity['physical_gpu'],
        'numa_node': affinity['numa_node'],
        'cpu_affinity': affinity['cpu_affinity'],
        'seed': args.seed,
    }
    path = args.output / args.stage / 'timing' / 'sessions' / f'{logical_session}-rank{args.rank}.json'
    if path.exists() and not getattr(args, 'fresh_session', False):
        state = json.loads(path.read_text())
        if state.get('logical_session') != logical_session or state.get('rank') != args.rank or state.get('world_size') != args.world_size:
            raise ValueError('existing sharded timing session identity/rank/world_size mismatch')
        prior_cfg = state.get('configuration', {})
        for key in ('configuration', 'checkpoint_hashes', 'model_pair', 'model_revisions',
                    'generation_environment', 'generation_caps', 'stage', 'scale', 'rank', 'world_size', 'seed'):
            if prior_cfg.get(key) != config.get(key):
                raise ValueError(f'existing sharded timing session configuration mismatch on {key}')
    else:
        state = {
            'logical_session': logical_session,
            'rank': args.rank,
            'world_size': args.world_size,
            'sharding_mode': f'sharded_{args.world_size}gpu',
            'sharding_plan_path': str(Path(plan_path).resolve()),
            'sharding_plan_sha256': plan_sha256,
            'configuration': config,
            'created_at_unix': time.time(),
            'epochs': [],
            'bundles': {},
        }
    state.update({
        'physical_gpu': affinity['physical_gpu'],
        'numa_node': affinity['numa_node'],
        'cpu_affinity': affinity['cpu_affinity'],
        'cpu_affinity_applied': affinity['cpu_affinity_applied'],
        'timing_harness_implementation_sha256': implementation_hash(args.stage),
        'algorithm_implementation_sha256': algorithm_hash(),
    })
    epoch = uuid.uuid4().hex
    state['epochs'].append({'epoch': epoch, 'started_at_unix': time.time(), 'device': str(runner.device),
                            'physical_gpu': affinity['physical_gpu'], 'numa_node': affinity['numa_node'],
                            'cpu_affinity': affinity['cpu_affinity']})
    atomic_json(path, state)
    return path, state, epoch


def comparison_bundle_key(jobs_and_settings, repeat, profile):
    return fingerprint({'jobs': [(job['prompt'], job['arm'], job['sampling_name'], job['seed'], setting)
                                 for job, setting in jobs_and_settings], 'repeat': repeat, 'profile': profile})


def read_comparison_bundle(runner, mode, jobs_and_settings, frozen_hash, session_state, repeat, profile=False, manifest=None, rank=None):
    """Return only a complete, checksummed, exactly matched all-arm bundle.

    Missing files are an interrupted bundle: callers may rerun every arm together.
    Contradictory identities or checksums fail explicitly rather than accepting or
    partially repairing measurements whose provenance is uncertain.
    """
    effective_rank = runner.args.rank if rank is None else int(rank)
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
        manifest = Manifest(directory / f'manifest-rank{effective_rank}.json').data
    rows = []
    for (job, setting), path in zip(jobs_and_settings, paths):
        identity = run_identity(runner, job, setting, mode, frozen_hash,
            session=session_state['logical_session'], repeat=f'{repeat}:epoch={epoch}', profile=profile)
        expected_path = directory / f'rank{effective_rank}' / f'{fingerprint(identity)}.json'
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
        if row.get('rank') != effective_rank:
            raise ValueError('completed comparison result rank mismatch')
        if (row.get('generation', {}).get('complete') is not True
                or row.get('phase_profile') is not profile
                or row.get('timing_isolated') is not (mode in ('timing', 'timing_calibration'))
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
    check_isolation = assert_no_external_cluster_gpu_processes if mode == 'timing_calibration' else assert_no_external_gpu_processes
    for job, setting in jobs_and_settings:
        check_isolation(runner.device)
        row = execute_job(runner, job, setting, mode, frozen_hash, session=session_state['logical_session'],
                          repeat=repeat_identity, profile=profile)
        rows.append(row)
        paths.append(str(runner.args.output / runner.args.stage / mode / f'rank{runner.args.rank}' / f"{fingerprint(row['identity'])}.json"))
    check_isolation(runner.device)
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


def _select_from_dev_observations(active_arms, arm_candidates, observations):
    plain_by_prompt = {row['prompt_id']: row for row in observations['plain_dflash'][fingerprint({})]}
    selected, candidate_reports = {}, {}
    for arm in active_arms:
        reports = []
        for setting in arm_candidates[arm]:
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
            nonempty_conds = [values for values in by_condition.values() if values]
            score = statistics.mean(statistics.mean(values) for values in nonempty_conds) if nonempty_conds else 0.0
            reports.append({'settings': setting, 'selection_score': score, 'condition_normalized_throughput': {k: statistics.mean(v) for k, v in by_condition.items()},
                            'dev_requests': len(rows), 'repetition_excluded_matched_pairs': excluded,
                            'long_condition_data_sparsity': not any(k.startswith('long_') and v for k, v in by_condition.items()), 'generated_tokens': sum(row['generated_tokens'] for row in rows),
                            'wall_seconds': sum(row['elapsed_seconds'] for row in rows)})
        chosen = max(reports, key=lambda row: row['selection_score'])
        selected[arm] = chosen['settings']
        candidate_reports[arm] = reports
    return selected, candidate_reports


def compute_tune_sharding_plan(args, prompt_manifest, prompts, arm_candidates):
    """Condition-stratified token-balanced snake assignment of dev prompts across `1..8` sharded tune ranks."""
    token_sums = defaultdict(int)
    weight_source = 'prompt_manifest_order_fallback'
    has_weights = False
    target_pids = {p['prompt_id'] for p in prompts}
    for prior_stage in ('m4', 'm3'):
        if prior_stage == args.stage:
            continue
        prior_tune_dir = args.output / prior_stage / 'tune'
        mpaths = sorted(prior_tune_dir.glob('manifest-rank*.json')) if prior_tune_dir.is_dir() else []
        if not mpaths:
            continue
        stage_sums = defaultdict(int)
        seen_pids = set()
        seen_job_keys = set()
        for mpath in mpaths:
            for jkey, entry in Manifest(mpath).data.get('jobs', {}).items():
                if entry.get('status') != 'complete' or jkey in seen_job_keys:
                    continue
                rpath = Path(entry.get('result_path', ''))
                if not rpath.is_file() and rpath.name:
                    for sub in sorted(prior_tune_dir.glob('rank*')):
                        alt = sub / rpath.name
                        if alt.is_file():
                            rpath = alt
                            break
                if not rpath.is_file():
                    continue
                row = json.loads(rpath.read_text())
                pid = row.get('prompt_id')
                if pid in target_pids:
                    seen_job_keys.add(jkey)
                    stage_sums[pid] += int(row.get('generated_tokens', 0))
                    seen_pids.add(pid)
        if target_pids and target_pids <= seen_pids:
            token_sums = stage_sums
            weight_source = f'{prior_stage}_tune_realized_tokens'
            has_weights = True
            break
    if not has_weights:
        m1_audit_path = args.output / 'm1_audit.json'
        if m1_audit_path.is_file():
            try:
                m1_data = json.loads(m1_audit_path.read_text())
                m1_sums = defaultdict(int)
                seen_m1 = set()
                for gen in m1_data.get('generations', []):
                    pid = gen.get('prompt_id')
                    if pid in target_pids and gen.get('realized_length') is not None:
                        m1_sums[pid] += int(gen['realized_length'])
                        seen_m1.add(pid)
                if target_pids and target_pids <= seen_m1:
                    token_sums = m1_sums
                    weight_source = 'm1_audit_realized_tokens'
                    has_weights = True
            except (OSError, ValueError, TypeError):
                pass

    manifest_order = {p['prompt_id']: idx for idx, p in enumerate(prompt_manifest['prompts'])}
    conditions_sorted = sorted({p['condition'] for p in prompts})
    ordered_by_condition = {}
    if has_weights:
        prompt_token_sums = {p['prompt_id']: int(token_sums[p['prompt_id']]) for p in prompts}
        for cond in conditions_sorted:
            cond_prompts = [p for p in prompts if p['condition'] == cond]
            ordered_by_condition[cond] = sorted(
                cond_prompts,
                key=lambda p: (-prompt_token_sums[p['prompt_id']], manifest_order[p['prompt_id']], p['prompt_id']),
            )
    else:
        prompt_token_sums = {p['prompt_id']: None for p in prompts}
        for cond in conditions_sorted:
            cond_prompts = [p for p in prompts if p['condition'] == cond]
            ordered_by_condition[cond] = sorted(cond_prompts, key=lambda p: (manifest_order[p['prompt_id']], p['prompt_id']))

    prompt_to_rank = {}
    if all(len(cond_prompts) >= args.world_size for cond_prompts in ordered_by_condition.values()):
        for cond in conditions_sorted:
            for k, p in enumerate(ordered_by_condition[cond]):
                prompt_to_rank[p['prompt_id']] = _snake_rank(k, args.world_size)
    else:
        flat = [p for cond in conditions_sorted for p in ordered_by_condition[cond]]
        for k, p in enumerate(flat):
            prompt_to_rank[p['prompt_id']] = _snake_rank(k, args.world_size)

    rank_prompts = {
        r: [p['prompt_id'] for p in prompts if prompt_to_rank[p['prompt_id']] == r]
        for r in range(args.world_size)
    }
    if any(not pids for pids in rank_prompts.values()):
        raise ValueError('M5 tune sharding plan left at least one rank with zero dev prompts')
    return {
        'schema_version': 1,
        'stage': args.stage,
        'scale': args.scale,
        'world_size': args.world_size,
        'sharding_mode': f'sharded_{args.world_size}gpu',
        'candidates_hash': fingerprint(arm_candidates),
        'prompt_manifest_hash': fingerprint(prompt_manifest),
        'selected_prompt_ids': [p['prompt_id'] for p in prompts],
        'seed': args.seed,
        'weight_source': weight_source,
        'prompt_token_sums': prompt_token_sums,
        'prompt_to_rank': prompt_to_rank,
        'ranks': {str(r): rank_prompts[r] for r in range(args.world_size)},
        'by_rank': {
            str(r): {
                'rank': r,
                'prompt_ids': rank_prompts[r],
                'prompt_count': len(rank_prompts[r]),
                'realized_prior_tune_tokens': (
                    sum(prompt_token_sums[pid] for pid in rank_prompts[r]) if has_weights else None
                ),
            }
            for r in range(args.world_size)
        },
        'by_condition': {
            cond: {
                'ordered_prompt_ids': [p['prompt_id'] for p in ordered_by_condition[cond]],
                'ranks': {
                    str(r): [p['prompt_id'] for p in ordered_by_condition[cond] if prompt_to_rank[p['prompt_id']] == r]
                    for r in range(args.world_size)
                },
            }
            for cond in conditions_sorted
        },
    }


def ensure_tune_sharding_plan(args, prompt_manifest, prompts, arm_candidates):
    """Atomically persist or validate `m5/tune/sharding_plan.json` across concurrent rank startups."""
    tune_dir = args.output / args.stage / 'tune'
    tune_dir.mkdir(parents=True, exist_ok=True)
    plan_path = tune_dir / 'sharding_plan.json'
    lock_path = tune_dir / '.sharding_plan.lock'
    computed = compute_tune_sharding_plan(args, prompt_manifest, prompts, arm_candidates)
    compare_keys = (
        'stage', 'scale', 'world_size', 'sharding_mode', 'candidates_hash',
        'prompt_manifest_hash', 'selected_prompt_ids', 'seed', 'weight_source',
        'prompt_token_sums', 'prompt_to_rank', 'ranks',
    )
    with lock_path.open('a+') as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if plan_path.is_file() and not getattr(args, 'fresh_session', False):
            existing = json.loads(plan_path.read_text())
            for key in compare_keys:
                if existing.get(key) != computed.get(key):
                    raise ValueError(f'existing tune sharding_plan.json mismatch on {key}; pass --fresh_session to rebuild')
            requested_session = getattr(args, 'logical_session', None)
            if requested_session is not None and existing.get('logical_session') != requested_session:
                raise ValueError('existing tune sharding_plan.json has a different logical_session; pass --fresh_session or omit --logical_session to resume')
            return plan_path, existing, file_sha256(plan_path)
        logical_session = getattr(args, 'logical_session', None) or uuid.uuid4().hex
        plan = {'logical_session': logical_session, **computed}
        atomic_json(plan_path, plan)
        return plan_path, plan, file_sha256(plan_path)


def sharded_tune_session(runner, logical_session, configuration, plan_path, plan_sha256):
    args = runner.args
    if getattr(runner, 'fastpath', None) is not None:
        configuration = dict(configuration, fastpath=runner.fastpath.config.to_dict())
    affinity = getattr(runner, 'affinity_info', None) or apply_gpu_cpu_affinity(torch.device('cpu'), args.rank)
    config = {
        'configuration': configuration,
        'checkpoint_hashes': runner.checkpoint_hashes,
        'model_pair': runner.model_pair,
        'model_revisions': runner.model_revisions,
        'generation_environment': getattr(runner, 'generation_environment', None),
        'generation_caps': generation_caps(args),
        'stage': args.stage,
        'scale': args.scale,
        'device': str(runner.device),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'rank': args.rank,
        'world_size': args.world_size,
        'physical_gpu': affinity['physical_gpu'],
        'numa_node': affinity['numa_node'],
        'cpu_affinity': affinity['cpu_affinity'],
        'seed': args.seed,
    }
    path = args.output / args.stage / 'tune' / 'sessions' / f'{logical_session}-rank{args.rank}.json'
    if path.exists() and not getattr(args, 'fresh_session', False):
        state = json.loads(path.read_text())
        if state.get('logical_session') != logical_session or state.get('rank') != args.rank or state.get('world_size') != args.world_size:
            raise ValueError('existing sharded tune session identity/rank/world_size mismatch')
        prior_cfg = state.get('configuration', {})
        for key in ('configuration', 'checkpoint_hashes', 'model_pair', 'model_revisions',
                    'generation_environment', 'generation_caps', 'stage', 'scale', 'rank', 'world_size', 'seed'):
            if prior_cfg.get(key) != config.get(key):
                raise ValueError(f'existing sharded tune session configuration mismatch on {key}')
    else:
        state = {
            'logical_session': logical_session,
            'rank': args.rank,
            'world_size': args.world_size,
            'sharding_mode': f'sharded_{args.world_size}gpu',
            'sharding_plan_path': str(Path(plan_path).resolve()),
            'sharding_plan_sha256': plan_sha256,
            'configuration': config,
            'created_at_unix': time.time(),
            'epochs': [],
            'bundles': {},
        }
    state.update({
        'physical_gpu': affinity['physical_gpu'],
        'numa_node': affinity['numa_node'],
        'cpu_affinity': affinity['cpu_affinity'],
        'cpu_affinity_applied': affinity['cpu_affinity_applied'],
        'timing_harness_implementation_sha256': implementation_hash(args.stage),
        'algorithm_implementation_sha256': algorithm_hash(),
    })
    if state.get('triton_gate_report') is None and getattr(runner, 'triton_gate_report', None) is not None:
        state['triton_gate_report'] = runner.triton_gate_report
    if state.get('torch_compile_gate_report') is None and getattr(runner, 'torch_compile_gate_report', None) is not None:
        state['torch_compile_gate_report'] = runner.torch_compile_gate_report
    epoch = uuid.uuid4().hex
    state['epochs'].append({'epoch': epoch, 'started_at_unix': time.time(), 'device': str(runner.device),
                            'physical_gpu': affinity['physical_gpu'], 'numa_node': affinity['numa_node'],
                            'cpu_affinity': affinity['cpu_affinity']})
    atomic_json(path, state)
    return path, state, epoch


def summarize_sharded_tune(args, manifest):
    """Pure-CPU aggregation of sharded dev comparison bundles into `<stage>/settings.json`."""
    from types import SimpleNamespace
    if not (args.stage == 'm5' or args.scale == 'full'):
        raise ValueError('summarize_sharded_tune is only supported for stage m5 or scale full')
    tune_dir = args.output / args.stage / 'tune'
    plan_path = tune_dir / 'sharding_plan.json'
    settings_path = args.output / args.stage / 'settings.json'
    if not plan_path.is_file():
        if settings_path.is_file():
            return json.loads(settings_path.read_text())
        raise FileNotFoundError(f'missing {args.stage} tune sharding_plan.json: {plan_path}')
    plan = json.loads(plan_path.read_text())
    if plan.get('stage') != args.stage or plan.get('scale') != args.scale:
        raise ValueError('tune sharding_plan.json stage/scale mismatch')
    world_size = int(plan['world_size'])
    if world_size < 2 or world_size > 8:
        raise ValueError('tune sharding_plan.json world_size must lie in 2..8')
    active_arms = _arms_from_args(args)
    kappa = getattr(args, 'bigram_count_gate_kappa', None) if args.stage == 'm5' else None
    k_ret = getattr(args, 'bigram_k_ret', None) if args.stage == 'm5' else None
    m5_betas = getattr(args, 'm5_beta_candidates', None) if args.stage == 'm5' else None
    m5_br = getattr(args, 'm5_beta_relax', None) if args.stage == 'm5' else None
    arm_candidates = {
        a: candidate_settings(
            a,
            bigram_count_gate_kappa=kappa,
            bigram_k_ret=k_ret,
            m5_beta_candidates=m5_betas,
            m5_beta_relax=m5_br,
        )
        for a in active_arms
    }
    if plan.get('candidates_hash') != fingerprint(arm_candidates):
        raise ValueError('tune sharding_plan.json candidates_hash differs from requested arms/candidates')
    prompts = select_prompts(manifest, 'dev', args.scale, args.standard_dev_prompts, args.long_dev_prompts)
    if [p['prompt_id'] for p in prompts] != plan.get('selected_prompt_ids'):
        raise ValueError('tune sharding_plan.json selected_prompt_ids mismatch')
    logical_session = plan['logical_session']
    session_paths = [tune_dir / 'sessions' / f'{logical_session}-rank{r}.json' for r in range(world_size)]
    manifest_paths = [tune_dir / f'manifest-rank{r}.json' for r in range(world_size)]
    if not all(p.is_file() for p in session_paths + manifest_paths):
        raise RuntimeError(f'sharded {args.stage} tune is incomplete; missing rank session or manifest files')
    rank_sessions = [json.loads(p.read_text()) for p in session_paths]
    rank_manifests = [json.loads(p.read_text()) for p in manifest_paths]

    cfg0 = rank_sessions[0]['configuration']
    fp0 = cfg0.get('configuration', {}).get('fastpath')
    if (
        any(getattr(args, k, False) for k in ('compile', 'cuda_graph', 'bf16_gemm', 'compile_drafter', 'compile_online'))
        or _cli_or_env_fast_decode(args)
        or not bool(getattr(args, 'analytic_online_grad', True))
    ):
        from .fastpath import FastPathConfig
        cli_fp = FastPathConfig(
            compile=bool(getattr(args, 'compile', False)),
            cuda_graph=bool(getattr(args, 'cuda_graph', False)),
            bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
            compile_drafter=bool(getattr(args, 'compile_drafter', False)),
            compile_online=bool(getattr(args, 'compile_online', False)),
            analytic_online_grad=bool(getattr(args, 'analytic_online_grad', True)),
            fast_decode=_cli_or_env_fast_decode(args),
        ).to_dict()
        if fp0 != cli_fp:
            raise ValueError('sharded tune rank 0 fastpath configuration differs from CLI fastpath flags')
    rank_affinity = {}
    seen_gpus, seen_cpus = set(), set()
    for r, sess in enumerate(rank_sessions):
        rcfg = sess.get('configuration', {})
        for key in ('checkpoint_hashes', 'model_pair', 'model_revisions', 'generation_environment',
                    'generation_caps', 'stage', 'scale', 'seed'):
            if rcfg.get(key) != cfg0.get(key):
                raise ValueError(f'sharded tune rank {r} configuration mismatch on {key}')
        if rcfg.get('configuration', {}).get('fastpath') != fp0:
            raise ValueError(f'sharded tune rank {r} fastpath configuration mismatch')
        phys_gpu = sess.get('physical_gpu')
        numa_node = sess.get('numa_node')
        cpus = list(sess.get('cpu_affinity') or [])
        if phys_gpu is None or numa_node is None or not cpus:
            raise ValueError(f'sharded tune session rank {r} lacks GPU/NUMA/CPU affinity metadata')
        if phys_gpu in seen_gpus:
            raise ValueError(f'duplicate physical_gpu {phys_gpu} across sharded tune ranks')
        if seen_cpus.intersection(cpus):
            raise ValueError(f'overlapping cpu_affinity across sharded tune ranks on rank {r}')
        seen_gpus.add(phys_gpu)
        seen_cpus.update(cpus)
        rank_affinity[str(r)] = {
            'physical_gpu': phys_gpu,
            'numa_node': numa_node,
            'cpu_affinity': cpus,
            'cpu_affinity_applied': bool(sess.get('cpu_affinity_applied', False)),
        }

    stub_args = copy.copy(args)
    stub_args.world_size = world_size
    stub = SimpleNamespace(
        args=stub_args, tiny=args.scale == 'smoke',
        checkpoint_hashes=cfg0['checkpoint_hashes'],
        model_pair=cfg0['model_pair'],
        model_revisions=cfg0['model_revisions'],
        generation_environment=cfg0.get('generation_environment'),
    )
    prompt_to_rank = {pid: int(rk) for pid, rk in plan['prompt_to_rank'].items()}
    observations = defaultdict(lambda: defaultdict(list))
    for prompt_index, prompt in enumerate(prompts):
        r = prompt_to_rank[prompt['prompt_id']]
        name, temperature, warp = sampling_conditions(prompt['condition'], primary_only=True)[0]
        base = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature, 'warper': warp, 'seed': args.seed}
        jobs = [(arm, setting) for arm in active_arms for setting in arm_candidates[arm]]
        offset = prompt_index % len(jobs)
        comparisons = [(dict(base, arm=arm), setting) for arm, setting in jobs[offset:] + jobs[:offset]]
        rows = read_comparison_bundle(
            stub, 'tune', comparisons, '', rank_sessions[r], 'dev', False,
            manifest=rank_manifests[r], rank=r,
        )
        if rows is None:
            raise RuntimeError(f'sharded {args.stage} tune is incomplete; missing completed dev bundle for {prompt["prompt_id"]} on rank {r}')
        for row in rows:
            observations[row['arm']][fingerprint(row['settings'])].append(row)

    selected, candidate_reports = _select_from_dev_observations(active_arms, arm_candidates, observations)
    frozen = {
        'schema_version': 1, 'frozen': True, 'scale': args.scale, 'stage': args.stage,
        'development_prompt_ids': [p['prompt_id'] for p in prompts],
        'prompt_manifest_hash': fingerprint(manifest), 'generation_caps': generation_caps(args),
        'comparison_session': logical_session,
        'sharding_mode': f'sharded_{world_size}gpu',
        'world_size': world_size,
        'sharding_plan_path': str(plan_path.resolve()),
        'sharding_plan_sha256': file_sha256(plan_path),
        'rank_affinity': rank_affinity,
        'warmup_policy': 'all candidates on each rank condition before measured dev requests',
        'checkpoint_hashes': cfg0['checkpoint_hashes'],
        'model_pair': cfg0['model_pair'], 'model_revisions': cfg0['model_revisions'],
        'selected': selected,
        'generation_environment': cfg0.get('generation_environment'),
        'candidates': candidate_reports,
        'selection_metric': 'mean across conditions of prompt throughput relative to same-prompt plain DFlash',
        'budget': 'two configurations per tuned arm, identical dev prompts, seeds and token caps; plain DFlash has no tuned parameters',
        'settings_seed': args.seed,
        'created_at_unix': min(s['created_at_unix'] for s in rank_sessions),
    }
    triton_rep = rank_sessions[0].get('triton_gate_report') if args.stage == 'm5' else None
    tc_rep = rank_sessions[0].get('torch_compile_gate_report') if args.stage == 'm5' else None
    if triton_rep is not None:
        frozen['triton_gate_report'] = triton_rep
    if tc_rep is not None:
        frozen['torch_compile_gate_report'] = tc_rep
    if settings_path.is_file() and not getattr(args, 'fresh_session', False):
        existing = json.loads(settings_path.read_text())
        stable_keys = (
            'frozen', 'scale', 'stage', 'development_prompt_ids', 'prompt_manifest_hash',
            'generation_caps', 'comparison_session', 'checkpoint_hashes', 'model_pair',
            'model_revisions', 'selected', 'candidates', 'settings_seed',
        )
        if all(existing.get(k) == frozen.get(k) for k in stable_keys):
            return existing
    atomic_json(settings_path, frozen)
    return frozen


def tune(runner, manifest):
    args = runner.args
    sharded_tune_ok = (args.stage == 'm5' or args.scale == 'full')
    if not sharded_tune_ok:
        if args.rank != 0 or args.world_size != 1:
            raise ValueError('dev selection is a single-process matched timing session')
    else:
        if args.world_size < 1 or args.world_size > 8 or not (0 <= args.rank < args.world_size):
            raise ValueError('sharded dev selection requires world_size in 1..8 and 0 <= rank < world_size')
    active_arms = _arms_from_args(args)
    kappa = getattr(args, 'bigram_count_gate_kappa', None) if args.stage == 'm5' else None
    k_ret = getattr(args, 'bigram_k_ret', None) if args.stage == 'm5' else None
    m5_betas = getattr(args, 'm5_beta_candidates', None) if args.stage == 'm5' else None
    m5_br = getattr(args, 'm5_beta_relax', None) if args.stage == 'm5' else None
    arm_candidates = {
        a: candidate_settings(
            a,
            bigram_count_gate_kappa=kappa,
            bigram_k_ret=k_ret,
            m5_beta_candidates=m5_betas,
            m5_beta_relax=m5_br,
        )
        for a in active_arms
    }
    prompts = select_prompts(manifest, 'dev', args.scale, args.standard_dev_prompts, args.long_dev_prompts)

    if sharded_tune_ok and args.world_size > 1:
        plan_path, plan, plan_sha256 = ensure_tune_sharding_plan(args, manifest, prompts, arm_candidates)
        logical_session = plan['logical_session']
        rank_pids = set(plan['ranks'][str(args.rank)])
        rank_prompts = [p for p in prompts if p['prompt_id'] in rank_pids]
        session_path, session_state, epoch = sharded_tune_session(
            runner, logical_session,
            {
                'prompt_manifest_hash': fingerprint(manifest),
                'dev_ids': [p['prompt_id'] for p in prompts],
                'rank_dev_ids': [p['prompt_id'] for p in rank_prompts],
                'candidates': arm_candidates,
            },
            plan_path, plan_sha256,
        )
        rank_bundle_specs = []
        need_warmup = False
        for prompt_index, prompt in enumerate(prompts):
            if prompt['prompt_id'] not in rank_pids:
                continue
            name, temperature, warp = sampling_conditions(prompt['condition'], primary_only=True)[0]
            base = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature, 'warper': warp, 'seed': args.seed}
            jobs = [(arm, setting) for arm in active_arms for setting in arm_candidates[arm]]
            offset = prompt_index % len(jobs)
            comparisons = [(dict(base, arm=arm), setting) for arm, setting in jobs[offset:] + jobs[:offset]]
            prior = read_comparison_bundle(runner, 'tune', comparisons, '', session_state, 'dev')
            if prior is None:
                need_warmup = True
            rank_bundle_specs.append((comparisons, prior))
        if need_warmup:
            warm_comparisons(runner, rank_prompts, arm_candidates)
        for comparisons, prior in rank_bundle_specs:
            if prior is None:
                comparison_bundle(runner, 'tune', comparisons, '', session_path, session_state, epoch, 'dev')
        return {
            'stage': args.stage, 'scale': args.scale, 'mode': 'tune',
            'rank': args.rank, 'world_size': args.world_size,
            'logical_session': logical_session, 'bundles': len(rank_bundle_specs),
            'status': 'tune_rank_complete',
        }

    session_path, session_state, epoch = comparison_session(runner, 'tune', {'prompt_manifest_hash': fingerprint(manifest),
        'dev_ids': [p['prompt_id'] for p in prompts], 'candidates': arm_candidates})
    bundle_specs = []
    need_warmup = False
    for prompt_index, prompt in enumerate(prompts):
        name, temperature, warp = sampling_conditions(prompt['condition'], primary_only=True)[0]
        base = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature, 'warper': warp, 'seed': args.seed}
        jobs = [(arm, setting) for arm in active_arms for setting in arm_candidates[arm]]
        # Rotate arm order across prompts to reduce systematic thermal/order bias.
        offset = prompt_index % len(jobs)
        comparisons = [(dict(base, arm=arm), setting) for arm, setting in jobs[offset:] + jobs[:offset]]
        prior = read_comparison_bundle(runner, 'tune', comparisons, '', session_state, 'dev')
        if prior is None:
            need_warmup = True
        bundle_specs.append((comparisons, prior))
    if need_warmup:
        warm_comparisons(runner, prompts, arm_candidates)
    observations = defaultdict(lambda: defaultdict(list))
    dev_rows = []
    for comparisons, prior in bundle_specs:
        rows = prior if prior is not None else comparison_bundle(runner, 'tune', comparisons, '', session_path, session_state, epoch, 'dev')
        for row in rows:
            dev_rows.append(row)
            observations[row['arm']][fingerprint(row['settings'])].append(row)
    selected, candidate_reports = _select_from_dev_observations(active_arms, arm_candidates, observations)
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
    triton_rep = session_state.get('triton_gate_report') if args.stage == 'm5' else None
    if triton_rep is None and args.stage == 'm5':
        triton_rep = getattr(runner, 'triton_gate_report', None)
    tc_rep = session_state.get('torch_compile_gate_report') if args.stage == 'm5' else None
    if tc_rep is None and args.stage == 'm5':
        tc_rep = getattr(runner, 'torch_compile_gate_report', None)
    if triton_rep is not None:
        frozen['triton_gate_report'] = triton_rep
    if tc_rep is not None:
        frozen['torch_compile_gate_report'] = tc_rep
    settings_path = args.output / args.stage / 'settings.json'
    if settings_path.is_file() and not getattr(args, 'fresh_session', False):
        existing = json.loads(settings_path.read_text())
        stable_keys = (
            'frozen', 'scale', 'stage', 'development_prompt_ids', 'prompt_manifest_hash',
            'generation_caps', 'comparison_session', 'checkpoint_hashes', 'model_pair',
            'model_revisions', 'selected', 'candidates', 'settings_seed',
        )
        if all(existing.get(k) == frozen.get(k) for k in stable_keys):
            return existing
    atomic_json(settings_path, frozen)
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
    jobs = evaluation_jobs(
        prompts, args.stage, args.scale, args.seed,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )
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


def _snake_rank(index: int, world_size: int) -> int:
    band, offset = divmod(int(index), int(world_size))
    return offset if (band % 2 == 0) else (int(world_size) - 1 - offset)


def compute_timing_sharding_plan(args, prompt_manifest, settings, selected):
    """Deterministically shard timing prompts across ranks via condition-stratified token-balanced snake order."""
    from types import SimpleNamespace
    if not 1 <= args.world_size <= 8:
        raise ValueError('timing sharding world_size must lie in 1..8')
    if args.world_size > len(selected):
        raise ValueError(f'timing world_size ({args.world_size}) cannot exceed selected prompt count ({len(selected)})')
    frozen_hash = fingerprint(settings)
    stub = SimpleNamespace(
        args=args, tiny=args.scale == 'smoke', checkpoint_hashes=settings['checkpoint_hashes'],
        model_pair=settings['model_pair'], model_revisions=settings['model_revisions'],
        generation_environment=settings.get('generation_environment'),
    )
    eval_dir = args.output / args.stage / 'evaluate'
    available = {}
    if eval_dir.is_dir():
        for manifest_path in sorted(eval_dir.glob('manifest-rank*.json')):
            available.update(Manifest(manifest_path).data.get('jobs', {}))
    eval_jobs = evaluation_jobs(
        selected, args.stage, args.scale, args.seed,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )
    token_sums = defaultdict(int)
    eval_complete = bool(eval_jobs)
    for job in eval_jobs:
        identity = run_identity(stub, job, settings['selected'][job['arm']], 'evaluate', frozen_hash)
        entry = available.get(fingerprint(identity))
        if entry is None or entry.get('status') != 'complete':
            eval_complete = False
            break
        rpath = Path(entry.get('result_path', ''))
        if not rpath.is_file():
            eval_complete = False
            break
        payload = rpath.read_bytes()
        if entry.get('result_sha256') and hashlib.sha256(payload).hexdigest() != entry['result_sha256']:
            raise ValueError(f'evaluate result checksum changed while building timing sharding plan: {rpath}')
        row = json.loads(payload)
        if row.get('identity') != identity:
            raise ValueError(f'evaluate result identity mismatch while building timing sharding plan: {rpath}')
        token_sums[job['prompt']['prompt_id']] += int(row['generated_tokens'])

    manifest_order = {p['prompt_id']: idx for idx, p in enumerate(prompt_manifest['prompts'])}
    conditions_sorted = sorted({p['condition'] for p in selected})
    ordered_by_condition = {}
    if eval_complete:
        weight_source = 'evaluate_realized_tokens'
        prompt_token_sums = {p['prompt_id']: int(token_sums[p['prompt_id']]) for p in selected}
        for cond in conditions_sorted:
            cond_prompts = [p for p in selected if p['condition'] == cond]
            ordered_by_condition[cond] = sorted(
                cond_prompts,
                key=lambda p: (-prompt_token_sums[p['prompt_id']], manifest_order[p['prompt_id']], p['prompt_id']),
            )
    else:
        weight_source = 'prompt_manifest_order_fallback'
        prompt_token_sums = {p['prompt_id']: None for p in selected}
        for cond in conditions_sorted:
            cond_prompts = [p for p in selected if p['condition'] == cond]
            ordered_by_condition[cond] = sorted(cond_prompts, key=lambda p: (manifest_order[p['prompt_id']], p['prompt_id']))

    prompt_to_rank = {}
    if all(len(cond_prompts) >= args.world_size for cond_prompts in ordered_by_condition.values()):
        for cond in conditions_sorted:
            for k, p in enumerate(ordered_by_condition[cond]):
                prompt_to_rank[p['prompt_id']] = _snake_rank(k, args.world_size)
    else:
        flat = [p for cond in conditions_sorted for p in ordered_by_condition[cond]]
        for k, p in enumerate(flat):
            prompt_to_rank[p['prompt_id']] = _snake_rank(k, args.world_size)

    rank_prompts = {
        r: [p['prompt_id'] for p in selected if prompt_to_rank[p['prompt_id']] == r]
        for r in range(args.world_size)
    }
    if any(not pids for pids in rank_prompts.values()):
        raise ValueError('timing sharding plan left at least one rank with zero prompts')
    return {
        'schema_version': 1,
        'stage': args.stage,
        'scale': args.scale,
        'world_size': args.world_size,
        'sharding_mode': f'sharded_{args.world_size}gpu',
        'frozen_settings_hash': frozen_hash,
        'prompt_manifest_hash': fingerprint(prompt_manifest),
        'selected_prompt_ids': [p['prompt_id'] for p in selected],
        'repeats': args.timing_repeats,
        'warmups': args.timing_warmups,
        'seed': args.seed,
        'weight_source': weight_source,
        'prompt_token_sums': prompt_token_sums,
        'prompt_to_rank': prompt_to_rank,
        'ranks': {str(r): rank_prompts[r] for r in range(args.world_size)},
        'by_rank': {
            str(r): {
                'rank': r,
                'prompt_ids': rank_prompts[r],
                'prompt_count': len(rank_prompts[r]),
                'realized_evaluate_tokens': (
                    sum(prompt_token_sums[pid] for pid in rank_prompts[r]) if eval_complete else None
                ),
            }
            for r in range(args.world_size)
        },
        'by_condition': {
            cond: {
                'ordered_prompt_ids': [p['prompt_id'] for p in ordered_by_condition[cond]],
                'ranks': {
                    str(r): [p['prompt_id'] for p in ordered_by_condition[cond] if prompt_to_rank[p['prompt_id']] == r]
                    for r in range(args.world_size)
                },
            }
            for cond in conditions_sorted
        },
    }


def ensure_timing_sharding_plan(args, prompt_manifest, settings, selected):
    """Atomically persist or validate `sharding_plan.json` across concurrent rank startups."""
    timing_dir = args.output / args.stage / 'timing'
    timing_dir.mkdir(parents=True, exist_ok=True)
    plan_path = timing_dir / 'sharding_plan.json'
    lock_path = timing_dir / '.sharding_plan.lock'
    computed = compute_timing_sharding_plan(args, prompt_manifest, settings, selected)
    compare_keys = (
        'stage', 'scale', 'world_size', 'sharding_mode', 'frozen_settings_hash',
        'prompt_manifest_hash', 'selected_prompt_ids', 'repeats', 'warmups', 'seed',
        'weight_source', 'prompt_token_sums', 'prompt_to_rank', 'ranks',
    )
    with lock_path.open('a+') as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if plan_path.is_file() and not getattr(args, 'fresh_session', False):
            existing = json.loads(plan_path.read_text())
            for key in compare_keys:
                if existing.get(key) != computed.get(key):
                    raise ValueError(f'existing sharding_plan.json mismatch on {key}; pass --fresh_session to rebuild')
            requested_session = getattr(args, 'logical_session', None)
            if requested_session is not None and existing.get('logical_session') != requested_session:
                raise ValueError('existing sharding_plan.json has a different logical_session; pass --fresh_session or omit --logical_session to resume')
            return plan_path, existing, file_sha256(plan_path)
        logical_session = getattr(args, 'logical_session', None) or uuid.uuid4().hex
        plan = {'logical_session': logical_session, **computed}
        atomic_json(plan_path, plan)
        return plan_path, plan, file_sha256(plan_path)


def _finalize_timing_report(args, prompts, selected, frozen_hash, session, rows, profiles,
                            raw_evidence, *, world_size, device_str, isolation_str,
                            restart_epochs=None, restart_epochs_by_rank=None,
                            resume_policy=None, rank_affinity=None,
                            sharding_plan_path=None, sharding_plan_sha256=None,
                            repeats=None, warmups=None, harness_sha=None, alg_sha=None,
                            fastpath_dict=None):
    full_coverage = len(selected) == len(prompts)
    eff_repeats = args.timing_repeats if repeats is None else repeats
    eff_warmups = args.timing_warmups if warmups is None else warmups
    eff_harness_sha = implementation_hash(args.stage) if harness_sha is None else harness_sha
    eff_alg_sha = algorithm_hash() if alg_sha is None else alg_sha
    sharding_mode = 'single_gpu' if world_size == 1 else f'sharded_{world_size}gpu'
    report = {
        'scale': args.scale,
        'stage': args.stage,
        'status': 'complete' if full_coverage else 'pilot_complete',
        'full_reference_prompt_coverage': full_coverage,
        'expected_full_prompt_count': len(prompts),
        'timing_session': session,
        'frozen_settings_hash': frozen_hash,
        'world_size': world_size,
        'sharding_mode': sharding_mode,
        'timing_harness_implementation_sha256': eff_harness_sha,
        'algorithm_implementation_sha256': eff_alg_sha,
        'algorithm_files_sha256': algorithm_file_hashes(),
        'restart_epochs': restart_epochs if world_size == 1 else restart_epochs_by_rank,
        'resume_policy': resume_policy or 'reuse complete all-arm prompt/seed/repeat bundles only; rerun every arm of interrupted bundles in the new epoch',
        'device': device_str,
        'isolation': isolation_str,
        'rank_affinity': rank_affinity,
        'sharding_plan_path': str(Path(sharding_plan_path).resolve()) if sharding_plan_path else None,
        'sharding_plan_sha256': sharding_plan_sha256,
        'warmups_per_arm_condition': eff_warmups,
        'repeats': eff_repeats,
        'prompt_ids': [p['prompt_id'] for p in selected],
        'timed_requests': len(rows),
        'raw_evidence': raw_evidence,
        'groups': summarize_rows(rows, isolated_timing=True),
        'phase_profiles': summarize_rows(profiles, isolated_timing=True),
        'instrumentation_policy': 'Uninstrumented synchronized request wall time for speed; separate same-arm/same-prompt CUDA-event profile runs for phases.',
        'timing_scope': 'all required held-out test prompts and condition-specific seed counts' if full_coverage else 'pilot matched subset; cannot satisfy milestone throughput gate',
        'baseline_policy': 'all baselines remeasured in this same session; no M3/M4 throughput reuse',
    }
    if fastpath_dict is not None:
        report['fastpath'] = fastpath_dict
    plain = {(g['condition'], g['temperature'], json.dumps(g['sampling_warp_config'], sort_keys=True)): g['metrics']['tokens_per_second']['mean']
             for g in report['groups'] if g['arm'] == 'plain_dflash'}
    paired_plain = {
        (row['condition'], row['temperature'], json.dumps(row['sampling_warp_config'], sort_keys=True),
         row['prompt_id'], row['seed'], str(row['repeat']).split(':epoch=')[0]): row
        for row in rows if row['arm'] == 'plain_dflash'
    }
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
            rep_base = str(row['repeat']).split(':epoch=')[0]
            baseline_row = paired_plain[(*row_key, row['prompt_id'], row['seed'], rep_base)]
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
    if args.stage == 'm5':
        report['offset_bin_slices'] = summarize_offset_bin_slices(rows, isolated_timing=True)
    cal_path = args.output / args.stage / 'timing_isolation_calibration.json'
    if cal_path.is_file():
        cal_data = json.loads(cal_path.read_text())
        report['isolation_calibration'] = {
            'path': str(cal_path.resolve()),
            'sha256': file_sha256(cal_path),
            'status': cal_data.get('status'),
            'calibrated_timed_runs': cal_data.get('calibrated_timed_runs'),
        }
    return report


def summarize_sharded_timing(args, prompt_manifest, settings):
    """Aggregate rank 0..W-1 matched timing bundles into `timing.json` once all ranks finish."""
    from types import SimpleNamespace
    plan_path = args.output / args.stage / 'timing' / 'sharding_plan.json'
    if not plan_path.is_file():
        return None
    plan = json.loads(plan_path.read_text())
    frozen_hash = fingerprint(settings)
    if (plan.get('stage') != args.stage or plan.get('scale') != args.scale
            or plan.get('frozen_settings_hash') != frozen_hash):
        return None
    world_size = int(plan['world_size'])
    if world_size < 2:
        return None
    logical_session = plan['logical_session']
    repeats = int(plan['repeats'])
    warmups = int(plan['warmups'])
    timing_dir = args.output / args.stage / 'timing'
    session_paths = [timing_dir / 'sessions' / f'{logical_session}-rank{r}.json' for r in range(world_size)]
    manifest_paths = [timing_dir / f'manifest-rank{r}.json' for r in range(world_size)]
    if not all(p.is_file() for p in session_paths + manifest_paths):
        return None
    rank_sessions = [json.loads(p.read_text()) for p in session_paths]
    rank_manifests = [json.loads(p.read_text()) for p in manifest_paths]

    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    by_id = {p['prompt_id']: p for p in prompts}
    selected = [by_id[pid] for pid in plan['selected_prompt_ids'] if pid in by_id]
    if [p['prompt_id'] for p in selected] != plan['selected_prompt_ids']:
        raise ValueError('sharding_plan.json selected_prompt_ids contain unknown test prompts')

    stub_args = copy.copy(args)
    stub_args.world_size = world_size
    stub = SimpleNamespace(
        args=stub_args, tiny=args.scale == 'smoke', checkpoint_hashes=settings['checkpoint_hashes'],
        model_pair=settings['model_pair'], model_revisions=settings['model_revisions'],
        generation_environment=settings.get('generation_environment'),
    )
    jobs = evaluation_jobs(
        selected, args.stage, args.scale, args.seed,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )
    rows, profiles = [], []
    bundle_count = 0
    prompt_to_rank = {pid: int(rk) for pid, rk in plan['prompt_to_rank'].items()}
    for repeat, profile, comparisons in timing_schedule(jobs, settings, repeats):
        prompt_id = comparisons[0][0]['prompt']['prompt_id']
        r = prompt_to_rank[prompt_id]
        bundle_rows = read_comparison_bundle(
            stub, 'timing', comparisons, frozen_hash, rank_sessions[r],
            repeat, profile, manifest=rank_manifests[r], rank=r,
        )
        if bundle_rows is None:
            return None
        (profiles if profile else rows).extend(bundle_rows)
        bundle_count += 1

    rank_affinity = {}
    seen_gpus, seen_cpus = set(), set()
    harness_shas = set()
    alg_shas = set()
    for r, sess in enumerate(rank_sessions):
        phys_gpu = sess.get('physical_gpu')
        numa_node = sess.get('numa_node')
        cpus = list(sess.get('cpu_affinity') or [])
        if phys_gpu is None or numa_node is None or not cpus:
            raise ValueError(f'sharded timing session rank {r} lacks GPU/NUMA/CPU affinity metadata')
        if phys_gpu in seen_gpus:
            raise ValueError(f'duplicate physical_gpu {phys_gpu} across sharded timing ranks')
        if seen_cpus.intersection(cpus):
            raise ValueError(f'overlapping cpu_affinity across sharded timing ranks on rank {r}')
        seen_gpus.add(phys_gpu)
        seen_cpus.update(cpus)
        rank_affinity[str(r)] = {
            'physical_gpu': phys_gpu,
            'numa_node': numa_node,
            'cpu_affinity': cpus,
            'cpu_affinity_applied': bool(sess.get('cpu_affinity_applied', False)),
        }
        if sess.get('timing_harness_implementation_sha256'):
            harness_shas.add(sess['timing_harness_implementation_sha256'])
        if sess.get('algorithm_implementation_sha256'):
            alg_shas.add(sess['algorithm_implementation_sha256'])
    if len(harness_shas) > 1 or len(alg_shas) > 1:
        raise ValueError('inconsistent harness/algorithm implementation SHA256 across sharded timing ranks')
    plan_sha256 = file_sha256(plan_path)
    raw_evidence = {
        'world_size': world_size,
        'sharding_plan_path': str(plan_path.resolve()),
        'sharding_plan_sha256': plan_sha256,
        'session_paths': [str(p.resolve()) for p in session_paths],
        'session_sha256s': [file_sha256(p) for p in session_paths],
        'manifest_paths': [str(p.resolve()) for p in manifest_paths],
        'manifest_sha256s': [file_sha256(p) for p in manifest_paths],
        'timing_requests': len(rows),
        'profile_requests': len(profiles),
        'bundles': bundle_count,
    }
    device_str = rank_sessions[0].get('configuration', {}).get('device', str(args.device))
    isolation_str = (
        f'{world_size} NUMA/core-pinned processes (16 physical cores/GPU); '
        'per-GPU idle check before model load and external-process check around every matched all-arm bundle; '
        'paired with 1-GPU cluster-isolated calibration'
    )
    fastpath_dict = rank_sessions[0].get('configuration', {}).get('configuration', {}).get('fastpath')
    report = _finalize_timing_report(
        stub_args, prompts, selected, frozen_hash, logical_session, rows, profiles, raw_evidence,
        world_size=world_size, device_str=device_str, isolation_str=isolation_str,
        restart_epochs_by_rank={str(r): rank_sessions[r]['epochs'] for r in range(world_size)},
        resume_policy='reuse complete all-arm prompt/seed/repeat bundles per rank session; rerun every arm of interrupted bundles in the new epoch',
        rank_affinity=rank_affinity, sharding_plan_path=plan_path, sharding_plan_sha256=plan_sha256,
        repeats=repeats, warmups=warmups,
        harness_sha=next(iter(harness_shas), implementation_hash(args.stage)),
        alg_sha=next(iter(alg_shas), algorithm_hash()),
        fastpath_dict=fastpath_dict,
    )
    atomic_json(args.output / args.stage / 'timing.json', report)
    return report


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
        world_size = int(report.get('world_size', evidence.get('world_size', 1)))
        report_fastpath = None
        if 'fastpath' in report:
            raw_fp = report['fastpath']
            req_fp_keys = {'compile', 'cuda_graph', 'compile_mode'}
            opt_fp_keys = {'bf16_gemm', 'compile_drafter', 'compile_online', 'analytic_online_grad', 'fast_decode'}
            if (not isinstance(raw_fp, dict)
                    or not req_fp_keys <= set(raw_fp.keys()) <= (req_fp_keys | opt_fp_keys)
                    or type(raw_fp['compile']) is not bool
                    or type(raw_fp['cuda_graph']) is not bool
                    or type(raw_fp['compile_mode']) is not str
                    or any(type(raw_fp[k]) is not bool for k in opt_fp_keys if k in raw_fp)):
                raise ValueError('invalid fastpath metadata in timing report')
            report_fastpath = raw_fp
        expected_fp_opt = None
        if hasattr(runner, 'fastpath'):
            expected_fp = runner.fastpath.config.to_dict() if runner.fastpath is not None else None
            if report_fastpath != expected_fp:
                raise ValueError('timing session configuration mismatch')
            expected_fp_opt = expected_fp
        elif (
            any(getattr(args, k, False) for k in ('compile', 'cuda_graph', 'bf16_gemm', 'compile_drafter', 'compile_online'))
            or _cli_or_env_fast_decode(args)
            or not bool(getattr(args, 'analytic_online_grad', True))
        ):
            from .fastpath import FastPathConfig
            cli_fp = FastPathConfig(
                compile=bool(getattr(args, 'compile', False)),
                cuda_graph=bool(getattr(args, 'cuda_graph', False)),
                bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
                compile_drafter=bool(getattr(args, 'compile_drafter', False)),
                compile_online=bool(getattr(args, 'compile_online', False)),
                analytic_online_grad=bool(getattr(args, 'analytic_online_grad', True)),
                fast_decode=_cli_or_env_fast_decode(args),
            ).to_dict()
            if report_fastpath != cli_fp:
                raise ValueError('timing session configuration mismatch')
            expected_fp_opt = cli_fp
        elif report_fastpath is not None:
            expected_fp_opt = report_fastpath
        jobs = evaluation_jobs(
            selected, args.stage, args.scale, args.seed,
            m5_arms=getattr(args, 'm5_arms', None),
            include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
        )
        counts = {'timing_requests': 0, 'profile_requests': 0, 'bundles': 0}

        if world_size == 1:
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
            if expected_fp_opt is not None:
                expected_inner['fastpath'] = expected_fp_opt
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
            for repeat, profile, comparisons in timing_schedule(jobs, settings, repeats):
                rows = read_comparison_bundle(runner, 'timing', comparisons, frozen_hash, session,
                                              repeat, profile, manifest=manifest, rank=0)
                if rows is None:
                    raise ValueError('required all-arm timing/profile bundle is incomplete')
                counts['profile_requests' if profile else 'timing_requests'] += len(rows)
                counts['bundles'] += 1
            if (counts['timing_requests'] != len(jobs) * repeats or counts['profile_requests'] != len(jobs)
                    or report['timed_requests'] != counts['timing_requests']
                    or any(evidence.get(key) != value for key, value in counts.items())):
                raise ValueError('raw timing/profile evidence count mismatch')
            return {'status': 'complete', **counts, 'world_size': 1,
                    'session_sha256': evidence['session_sha256'], 'manifest_sha256': evidence['manifest_sha256']}

        if not 2 <= world_size <= 8:
            raise ValueError(f'unsupported sharded timing world_size={world_size}')
        if report.get('sharding_mode') != f'sharded_{world_size}gpu':
            raise ValueError('sharded timing report sharding_mode mismatch')
        harness_sha = report.get('timing_harness_implementation_sha256')
        if not isinstance(harness_sha, str) or len(harness_sha) != 64:
            raise ValueError('sharded timing report lacks valid timing_harness_implementation_sha256')
        timing_dir = args.output / args.stage / 'timing'
        plan_path = timing_dir / 'sharding_plan.json'
        if Path(evidence.get('sharding_plan_path', '')).resolve() != plan_path.resolve():
            raise ValueError('sharding_plan_path mismatch')
        plan_bytes = plan_path.read_bytes()
        plan_sha = hashlib.sha256(plan_bytes).hexdigest()
        if plan_sha != evidence.get('sharding_plan_sha256') or plan_sha != report.get('sharding_plan_sha256'):
            raise ValueError('sharding_plan.json checksum changed')
        plan = json.loads(plan_bytes)
        plan_args = copy.copy(args)
        plan_args.world_size = world_size
        plan_args.timing_repeats = repeats
        plan_args.timing_warmups = warmups
        plan_args.timing_prompts = 0
        expected_plan = compute_timing_sharding_plan(plan_args, prompt_manifest, settings, selected)
        for key in ('stage', 'scale', 'world_size', 'sharding_mode', 'frozen_settings_hash',
                    'prompt_manifest_hash', 'selected_prompt_ids', 'repeats', 'warmups', 'seed',
                    'weight_source', 'prompt_token_sums', 'prompt_to_rank', 'ranks'):
            if plan.get(key) != expected_plan.get(key):
                raise ValueError(f'sharding_plan.json {key} does not match deterministic sharding rule')
        if plan.get('logical_session') != report['timing_session']:
            raise ValueError('sharding_plan.json logical_session mismatch')

        session_paths = evidence.get('session_paths', [])
        session_shas = evidence.get('session_sha256s', [])
        manifest_paths = evidence.get('manifest_paths', [])
        manifest_shas = evidence.get('manifest_sha256s', [])
        if (len(session_paths) != world_size or len(session_shas) != world_size
                or len(manifest_paths) != world_size or len(manifest_shas) != world_size):
            raise ValueError('sharded timing evidence file list length mismatch')

        rank_sessions, rank_manifests = [], []
        seen_gpus, seen_cpus = set(), set()
        for r in range(world_size):
            sp = Path(session_paths[r])
            exp_sp = timing_dir / 'sessions' / f"{report['timing_session']}-rank{r}.json"
            if sp.resolve() != exp_sp.resolve():
                raise ValueError(f'timing session path mismatch on rank {r}')
            sb = sp.read_bytes()
            if hashlib.sha256(sb).hexdigest() != session_shas[r]:
                raise ValueError(f'timing session checksum changed on rank {r}')
            sess = json.loads(sb)
            if sess.get('logical_session') != report['timing_session'] or sess.get('rank') != r or sess.get('world_size') != world_size:
                raise ValueError(f'timing session header mismatch on rank {r}')
            cfg = sess.get('configuration', {})
            exp_inner = {
                'frozen_settings_hash': frozen_hash,
                'prompt_manifest_hash': fingerprint(prompt_manifest),
                'selected_prompts': report['prompt_ids'],
                'rank_prompts': plan['ranks'][str(r)],
                'sharding_plan_sha256': plan_sha,
                'repeats': repeats,
                'warmups': warmups,
            }
            if expected_fp_opt is not None:
                exp_inner['fastpath'] = expected_fp_opt
            expected_cfg = {
                'configuration': exp_inner,
                'checkpoint_hashes': runner.checkpoint_hashes,
                'model_pair': runner.model_pair,
                'model_revisions': runner.model_revisions,
                'generation_environment': getattr(runner, 'generation_environment', None),
                'generation_caps': generation_caps(args),
                'stage': args.stage,
                'scale': args.scale,
                'rank': r,
                'world_size': world_size,
                'seed': args.seed,
            }
            if any(cfg.get(k) != v for k, v in expected_cfg.items()):
                raise ValueError(f'timing session configuration mismatch on rank {r}')
            phys_gpu = sess.get('physical_gpu')
            numa_node = sess.get('numa_node')
            cpus = list(sess.get('cpu_affinity') or [])
            if phys_gpu is None or numa_node is None or not cpus:
                raise ValueError(f'rank {r} missing physical_gpu/numa_node/cpu_affinity')
            if phys_gpu in seen_gpus or seen_cpus.intersection(cpus):
                raise ValueError(f'non-disjoint GPU/CPU affinity on rank {r}')
            seen_gpus.add(phys_gpu)
            seen_cpus.update(cpus)
            mp = Path(manifest_paths[r])
            exp_mp = timing_dir / f'manifest-rank{r}.json'
            if mp.resolve() != exp_mp.resolve():
                raise ValueError(f'timing manifest path mismatch on rank {r}')
            mb = mp.read_bytes()
            if hashlib.sha256(mb).hexdigest() != manifest_shas[r]:
                raise ValueError(f'timing manifest checksum changed on rank {r}')
            rank_sessions.append(sess)
            rank_manifests.append(json.loads(mb))

        prompt_to_rank = {pid: int(rk) for pid, rk in plan['prompt_to_rank'].items()}
        for repeat, profile, comparisons in timing_schedule(jobs, settings, repeats):
            pid = comparisons[0][0]['prompt']['prompt_id']
            r = prompt_to_rank[pid]
            rows = read_comparison_bundle(runner, 'timing', comparisons, frozen_hash, rank_sessions[r],
                                          repeat, profile, manifest=rank_manifests[r], rank=r)
            if rows is None:
                raise ValueError(f'required all-arm timing/profile bundle is incomplete on rank {r}')
            for row in rows:
                if row.get('timing_harness_implementation_sha256') != harness_sha:
                    raise ValueError('row timing_harness_implementation_sha256 mismatch')
            counts['profile_requests' if profile else 'timing_requests'] += len(rows)
            counts['bundles'] += 1
        if (counts['timing_requests'] != len(jobs) * repeats or counts['profile_requests'] != len(jobs)
                or report['timed_requests'] != counts['timing_requests']
                or any(evidence.get(key) != value for key, value in counts.items())):
            raise ValueError('raw timing/profile evidence count mismatch')
        return {'status': 'complete', **counts, 'world_size': world_size,
                'sharding_plan_sha256': plan_sha, 'session_sha256s': session_shas,
                'manifest_sha256s': manifest_shas}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'status': 'incomplete', 'reason': str(error)}


def timing(runner, prompt_manifest, settings):
    args = runner.args
    if not 1 <= args.world_size <= 8 or not 0 <= args.rank < args.world_size:
        raise ValueError('timing requires rank in 0..world_size-1 with world_size in 1..8')
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    selected = []
    for condition in sorted({p['condition'] for p in prompts}):
        condition_prompts = [p for p in prompts if p['condition'] == condition]
        selected.extend(condition_prompts if args.timing_prompts == 0 else condition_prompts[:args.timing_prompts])
    jobs = evaluation_jobs(
        selected, args.stage, args.scale, args.seed,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )
    frozen_hash = fingerprint(settings)
    active_arms = _arms_from_args(args)
    fastpath_dict = runner.fastpath.config.to_dict() if getattr(runner, 'fastpath', None) is not None else None

    if args.world_size == 1:
        session_path, session_state, epoch = comparison_session(runner, 'timing', {
            'frozen_settings_hash': frozen_hash,
            'prompt_manifest_hash': fingerprint(prompt_manifest),
            'selected_prompts': [p['prompt_id'] for p in selected],
            'repeats': args.timing_repeats,
            'warmups': args.timing_warmups,
        })
        session = session_state['logical_session']
        for condition in sorted({p['condition'] for p in selected}):
            prompt = next(p for p in selected if p['condition'] == condition)
            name, temperature, warp = sampling_conditions(condition, primary_only=True)[0]
            for _ in range(args.timing_warmups):
                for arm in active_arms:
                    job = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature,
                           'warper': warp, 'seed': args.seed, 'arm': arm, 'max_new_tokens_override': 64}
                    runner.run(job, settings['selected'][arm], profile=False)
        session_state['warmup_completed'] = True
        atomic_json(session_path, session_state)
        rows, profiles = [], []
        bundle_count = 0
        for repeat, profile, comparisons in timing_schedule(jobs, settings, args.timing_repeats):
            results = comparison_bundle(runner, 'timing', comparisons, frozen_hash,
                                        session_path, session_state, epoch, repeat, profile=profile)
            (profiles if profile else rows).extend(results)
            bundle_count += 1
        manifest_path = args.output / args.stage / 'timing' / 'manifest-rank0.json'
        affinity = getattr(runner, 'affinity_info', None) or apply_gpu_cpu_affinity(torch.device('cpu'), 0)
        raw_evidence = {
            'session_path': str(session_path.resolve()),
            'session_sha256': file_sha256(session_path),
            'manifest_path': str(manifest_path.resolve()),
            'manifest_sha256': file_sha256(manifest_path),
            'timing_requests': len(rows),
            'profile_requests': len(profiles),
            'bundles': bundle_count,
            'world_size': 1,
        }
        report = _finalize_timing_report(
            args, prompts, selected, frozen_hash, session, rows, profiles, raw_evidence,
            world_size=1, device_str=str(runner.device),
            isolation_str='one process; idle-device check before model load',
            restart_epochs=session_state['epochs'],
            rank_affinity={'0': affinity},
            fastpath_dict=fastpath_dict,
        )
        atomic_json(args.output / args.stage / 'timing.json', report)
        return report

    plan_path, plan, plan_sha256 = ensure_timing_sharding_plan(args, prompt_manifest, settings, selected)
    logical_session = plan['logical_session']
    rank_prompt_ids = list(plan['ranks'][str(args.rank)])
    rank_prompt_set = set(rank_prompt_ids)
    rank_prompts = [p for p in selected if p['prompt_id'] in rank_prompt_set]
    session_path, session_state, epoch = sharded_timing_session(
        runner, logical_session,
        {
            'frozen_settings_hash': frozen_hash,
            'prompt_manifest_hash': fingerprint(prompt_manifest),
            'selected_prompts': [p['prompt_id'] for p in selected],
            'rank_prompts': rank_prompt_ids,
            'sharding_plan_sha256': plan_sha256,
            'repeats': args.timing_repeats,
            'warmups': args.timing_warmups,
        },
        plan_path, plan_sha256,
    )
    # Warm up all arms on every condition assigned to this rank before timed measurements.
    for condition in sorted({p['condition'] for p in rank_prompts}):
        prompt = next(p for p in rank_prompts if p['condition'] == condition)
        name, temperature, warp = sampling_conditions(condition, primary_only=True)[0]
        for _ in range(args.timing_warmups):
            for arm in active_arms:
                job = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature,
                       'warper': warp, 'seed': args.seed, 'arm': arm, 'max_new_tokens_override': 64}
                runner.run(job, settings['selected'][arm], profile=False)
    session_state['warmup_completed'] = True
    atomic_json(session_path, session_state)
    rows, profiles = [], []
    bundle_count = 0
    for repeat, profile, comparisons in timing_schedule(jobs, settings, args.timing_repeats):
        if comparisons[0][0]['prompt']['prompt_id'] not in rank_prompt_set:
            continue
        results = comparison_bundle(runner, 'timing', comparisons, frozen_hash,
                                    session_path, session_state, epoch, repeat, profile=profile)
        (profiles if profile else rows).extend(results)
        bundle_count += 1
    manifest_path = args.output / args.stage / 'timing' / f'manifest-rank{args.rank}.json'
    full_coverage = len(selected) == len(prompts)
    rank_report = {
        'scale': args.scale,
        'stage': args.stage,
        'status': 'rank_complete' if full_coverage else 'pilot_rank_complete',
        'rank': args.rank,
        'world_size': args.world_size,
        'sharding_mode': f'sharded_{args.world_size}gpu',
        'timing_session': logical_session,
        'frozen_settings_hash': frozen_hash,
        'timing_harness_implementation_sha256': implementation_hash(args.stage),
        'algorithm_implementation_sha256': algorithm_hash(),
        'physical_gpu': runner.affinity_info['physical_gpu'],
        'numa_node': runner.affinity_info['numa_node'],
        'cpu_affinity': runner.affinity_info['cpu_affinity'],
        'cpu_affinity_applied': runner.affinity_info['cpu_affinity_applied'],
        'rank_prompt_ids': rank_prompt_ids,
        'timed_requests': len(rows),
        'profile_requests': len(profiles),
        'bundles': bundle_count,
        'sharding_plan_path': str(plan_path.resolve()),
        'sharding_plan_sha256': plan_sha256,
        'session_path': str(session_path.resolve()),
        'session_sha256': file_sha256(session_path),
        'manifest_path': str(manifest_path.resolve()),
        'manifest_sha256': file_sha256(manifest_path),
    }
    atomic_json(args.output / args.stage / 'timing' / f'timing-rank{args.rank}.json', rank_report)
    return rank_report


def select_calibration_prompts(args, prompt_manifest, settings):
    """Select 4 deterministic standard test prompts + 2 shortest non-rep-dominated long test prompts from evaluate."""
    from types import SimpleNamespace
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    standard = [p for p in prompts if p['condition'] == 'standard_nonthinking']
    long_prompts = [p for p in prompts if p['condition'].startswith('long_')]
    n_std = min(4, len(standard))
    n_long = min(2, len(long_prompts))
    selected_std = standard[:n_std]

    frozen_hash = fingerprint(settings)
    stub = SimpleNamespace(
        args=args, tiny=args.scale == 'smoke', checkpoint_hashes=settings['checkpoint_hashes'],
        model_pair=settings['model_pair'], model_revisions=settings['model_revisions'],
        generation_environment=settings.get('generation_environment'),
    )
    eval_dir = args.output / args.stage / 'evaluate'
    available = {}
    if eval_dir.is_dir():
        for manifest_path in sorted(eval_dir.glob('manifest-rank*.json')):
            available.update(Manifest(manifest_path).data.get('jobs', {}))

    non_rep_candidates = []
    fallback_candidates = []
    long_token_sums = {}
    for idx, p in enumerate(long_prompts):
        p_jobs = evaluation_jobs(
            [p], args.stage, args.scale, args.seed,
            m5_arms=getattr(args, 'm5_arms', None),
            include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
        )
        p_rows = []
        complete = True
        for job in p_jobs:
            identity = run_identity(stub, job, settings['selected'][job['arm']], 'evaluate', frozen_hash)
            entry = available.get(fingerprint(identity))
            if entry is None or entry.get('status') != 'complete' or not Path(entry.get('result_path', '')).is_file():
                complete = False
                break
            p_rows.append(json.loads(Path(entry['result_path']).read_text()))
        if complete and p_rows:
            tok_sum = sum(int(r['generated_tokens']) for r in p_rows)
            any_rep = any(bool(r['generation']['repetition_dominated']) for r in p_rows)
            long_token_sums[p['prompt_id']] = tok_sum
            if not any_rep:
                non_rep_candidates.append((tok_sum, idx, p))
            else:
                fallback_candidates.append((1, tok_sum, idx, p))
        else:
            fallback_candidates.append((2, 0, idx, p))

    non_rep_candidates.sort(key=lambda item: (item[0], item[1], item[2]['prompt_id']))
    selected_long = [item[2] for item in non_rep_candidates[:n_long]]
    long_source = 'evaluate_shortest_non_repetition_dominated'
    if len(selected_long) < n_long:
        if args.scale == 'reference' and available:
            raise ValueError('insufficient non-repetition-dominated long_thinking prompts in evaluate for calibration')
        long_source = 'manifest_order_fallback'
        fallback_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]['prompt_id']))
        for _, _, _, p in fallback_candidates:
            if len(selected_long) >= n_long:
                break
            if p['prompt_id'] not in {x['prompt_id'] for x in selected_long}:
                selected_long.append(p)

    selected = sorted(selected_std + selected_long, key=lambda p: p['condition'])
    return selected, {
        'standard_prompt_ids': [p['prompt_id'] for p in selected_std],
        'long_prompt_ids': [p['prompt_id'] for p in selected_long],
        'long_selection_source': long_source,
        'long_evaluate_token_sums': {p['prompt_id']: long_token_sums.get(p['prompt_id']) for p in selected_long},
    }


def calibration_jobs(prompts, stage, scale, seed=2026, *, m5_arms=None, include_long_oracle=False):
    jobs = []
    active_arms = arms_for(stage, scale, m5_arms=m5_arms, include_long_oracle=include_long_oracle)
    for prompt in prompts:
        name, temperature, warp = sampling_conditions(prompt['condition'], primary_only=True)[0]
        for arm in active_arms:
            jobs.append({'prompt': prompt, 'arm': arm, 'sampling_name': name,
                         'temperature': temperature, 'warper': warp, 'seed': seed})
    return jobs


def _load_sharded_or_single_timed_rows(args, settings):
    timing_dir = args.output / args.stage / 'timing'
    timing_report_path = args.output / args.stage / 'timing.json'
    if not timing_report_path.is_file():
        return []
    rec_timing = json.loads(timing_report_path.read_text())
    if rec_timing.get('frozen_settings_hash') != fingerprint(settings):
        return []
    raw_ev = rec_timing.get('raw_evidence', {})
    session_paths = (
        [Path(p) for p in raw_ev['session_paths']]
        if raw_ev.get('session_paths')
        else ([Path(raw_ev['session_path'])] if raw_ev.get('session_path') else sorted((timing_dir / 'sessions').glob('*.json')))
    )
    rows = []
    for sp in session_paths:
        if not sp.is_file():
            continue
        sess = json.loads(sp.read_text())
        for b_val in sess.get('bundles', {}).values():
            if b_val.get('status') != 'complete':
                continue
            for p_str in b_val.get('result_paths', []):
                rp = Path(p_str)
                if rp.is_file():
                    row = json.loads(rp.read_text())
                    if not row.get('phase_profile'):
                        rows.append(row)
    return rows


def reconcile_timing_isolation_calibration(
    args, prompt_manifest, settings, *,
    session_path=None, session_state=None, selected=None, selection_meta=None,
    cal_rows=None, cluster_idle_inventory=None, require_sharded=False,
):
    """Compare 1-GPU cluster-isolated calibration runs against 8-GPU sharded timing on matched prompts."""
    from types import SimpleNamespace
    frozen_hash = fingerprint(settings)
    cal_repeats = getattr(args, 'calibration_repeats', 2)
    cal_out_existing = args.output / args.stage / 'timing_isolation_calibration.json'
    if cal_rows is None and cal_out_existing.is_file():
        cal_repeats = int(json.loads(cal_out_existing.read_text()).get('repeats', cal_repeats))
    if selected is None or selection_meta is None:
        selected, selection_meta = select_calibration_prompts(args, prompt_manifest, settings)
    cal_dir = args.output / args.stage / 'timing_calibration'
    manifest_path = cal_dir / 'manifest-rank0.json'
    if cal_rows is None:
        if not manifest_path.is_file() or not (cal_dir / 'sessions').is_dir():
            if require_sharded:
                raise ValueError('timing_calibration raw artifacts are missing')
            return None
        stub_args = copy.copy(args)
        stub_args.rank = 0
        stub_args.world_size = 1
        stub = SimpleNamespace(
            args=stub_args, tiny=args.scale == 'smoke', checkpoint_hashes=settings['checkpoint_hashes'],
            model_pair=settings['model_pair'], model_revisions=settings['model_revisions'],
            generation_environment=settings.get('generation_environment'),
        )
        manifest_data = json.loads(manifest_path.read_bytes())
        timing_report_path_pre = args.output / args.stage / 'timing.json'
        report_fastpath = (
            json.loads(timing_report_path_pre.read_text()).get('fastpath')
            if timing_report_path_pre.is_file() else None
        )
        existing_cal_fastpath = (
            json.loads(cal_out_existing.read_text()).get('fastpath')
            if cal_out_existing.is_file() else None
        )
        expected_fp_opt = None
        if (
            any(getattr(args, k, False) for k in ('compile', 'cuda_graph', 'bf16_gemm', 'compile_drafter', 'compile_online'))
            or _cli_or_env_fast_decode(args)
            or not bool(getattr(args, 'analytic_online_grad', True))
        ):
            from .fastpath import FastPathConfig
            cli_fp = FastPathConfig(
                compile=bool(getattr(args, 'compile', False)),
                cuda_graph=bool(getattr(args, 'cuda_graph', False)),
                bf16_gemm=bool(getattr(args, 'bf16_gemm', False)),
                compile_drafter=bool(getattr(args, 'compile_drafter', False)),
                compile_online=bool(getattr(args, 'compile_online', False)),
                analytic_online_grad=bool(getattr(args, 'analytic_online_grad', True)),
                fast_decode=_cli_or_env_fast_decode(args),
            ).to_dict()
            if report_fastpath is not None and report_fastpath != cli_fp:
                if require_sharded:
                    raise ValueError('timing_calibration fastpath configuration differs from timing.json')
                return None
            expected_fp_opt = cli_fp
        elif report_fastpath is not None:
            expected_fp_opt = report_fastpath
        elif existing_cal_fastpath is not None:
            expected_fp_opt = existing_cal_fastpath
        expected_cfg = {
            'frozen_settings_hash': frozen_hash,
            'prompt_manifest_hash': fingerprint(prompt_manifest),
            'selected_prompts': [p['prompt_id'] for p in selected],
            'repeats': cal_repeats,
            'warmups': args.timing_warmups,
        }
        if expected_fp_opt is not None:
            expected_cfg['fastpath'] = expected_fp_opt
        expected_outer = {
            'checkpoint_hashes': settings['checkpoint_hashes'],
            'model_pair': settings['model_pair'],
            'model_revisions': settings['model_revisions'],
            'generation_environment': settings.get('generation_environment'),
            'generation_caps': generation_caps(args),
            'stage': args.stage,
            'scale': args.scale,
            'seed': args.seed,
        }
        matched_sessions = []
        for sp in sorted((cal_dir / 'sessions').glob('*.json')):
            sess = json.loads(sp.read_text())
            outer = sess.get('configuration', {})
            if outer.get('configuration') == expected_cfg and all(outer.get(k) == v for k, v in expected_outer.items()):
                matched_sessions.append((sp, sess))
        if len(matched_sessions) != 1:
            if require_sharded:
                raise ValueError('expected exactly one matching timing_calibration session')
            return None
        session_path, session_state = matched_sessions[0]
        cal_rows = []
        jobs = calibration_jobs(
            selected, args.stage, args.scale, args.seed,
            m5_arms=getattr(args, 'm5_arms', None),
            include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
        )
        groups_by_prompt = defaultdict(list)
        for job in jobs:
            groups_by_prompt[(job['prompt']['prompt_id'], job['sampling_name'], job['seed'])].append(job)
        for group_index, same_prompt_jobs in enumerate(groups_by_prompt.values()):
            for repeat in range(cal_repeats):
                offset = (group_index + repeat) % len(same_prompt_jobs)
                comparisons = [(job, settings['selected'][job['arm']])
                               for job in same_prompt_jobs[offset:] + same_prompt_jobs[:offset]]
                b_rows = read_comparison_bundle(
                    stub, 'timing_calibration', comparisons, frozen_hash, session_state,
                    repeat, profile=False, manifest=manifest_data, rank=0,
                )
                if b_rows is None:
                    if require_sharded:
                        raise ValueError('timing_calibration session has incomplete bundles')
                    return None
                cal_rows.extend(b_rows)

    sharded_rows = _load_sharded_or_single_timed_rows(args, settings)
    if require_sharded and not sharded_rows:
        raise ValueError('cannot reconcile timing_isolation_calibration: timing.json / raw sharded timing rows are missing')

    cal_slots = {
        (r['condition'], float(r['temperature']), json.dumps(r['sampling_warp_config'], sort_keys=True),
         r['prompt_id'], r['seed'])
        for r in cal_rows
    }
    sharded_sub = [
        r for r in sharded_rows
        if (r['condition'], float(r['temperature']), json.dumps(r['sampling_warp_config'], sort_keys=True),
            r['prompt_id'], r['seed']) in cal_slots
    ]

    def _prompt_mean_tps(subset):
        by_p = defaultdict(list)
        for rw in subset:
            by_p[rw['prompt_id']].append(float(rw['tokens_per_second']))
        return statistics.mean(statistics.mean(vs) for vs in by_p.values()) if by_p else None

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
            if pk in plain_map and plain_map[pk] > 0:
                by_p[rw['prompt_id']].append(float(rw['tokens_per_second']) / plain_map[pk])
        return statistics.mean(statistics.mean(vs) for vs in by_p.values()) if by_p else None

    arms = _arms_from_args(args)
    cond_keys = sorted({
        (r['condition'], float(r['temperature']), json.dumps(r['sampling_warp_config'], sort_keys=True))
        for r in cal_rows
    })
    groups = []
    by_prompt = []
    reconciled = bool(sharded_sub)
    for cond, temp, warp_json in cond_keys:
        c_1gpu = [r for r in cal_rows if r['condition'] == cond and float(r['temperature']) == temp
                  and json.dumps(r['sampling_warp_config'], sort_keys=True) == warp_json]
        c_8gpu = [r for r in sharded_sub if r['condition'] == cond and float(r['temperature']) == temp
                  and json.dumps(r['sampling_warp_config'], sort_keys=True) == warp_json]
        if not c_8gpu:
            reconciled = False
        for arm in arms:
            a_1 = [r for r in c_1gpu if r['arm'] == arm]
            a_8 = [r for r in c_8gpu if r['arm'] == arm]
            p_1 = sorted({r['prompt_id'] for r in a_1})
            p_8 = sorted({r['prompt_id'] for r in a_8})
            if c_8gpu and p_1 != p_8:
                reconciled = False
            tps_1 = _prompt_mean_tps(a_1)
            tps_8 = _prompt_mean_tps(a_8) if a_8 else None
            pr_1 = _prompt_mean_paired_ratio(c_1gpu, arm)
            pr_8 = _prompt_mean_paired_ratio(c_8gpu, arm) if c_8gpu else None
            scale_8_over_1 = (tps_8 / tps_1) if (tps_1 and tps_8 is not None) else None
            pr_diff = (pr_8 - pr_1) if (pr_1 is not None and pr_8 is not None) else None
            groups.append({
                'condition': cond,
                'temperature': temp,
                'sampling_warp_config': json.loads(warp_json),
                'arm': arm,
                'n_prompts': len(p_1),
                'runs_1gpu': len(a_1),
                'runs_8gpu': len(a_8),
                'tps_1gpu_mean': tps_1,
                'tps_8gpu_mean': tps_8,
                'tps_8gpu_over_1gpu_ratio': scale_8_over_1,
                'tps_rel_diff_pct': (100.0 * (scale_8_over_1 - 1.0)) if scale_8_over_1 is not None else None,
                'paired_ratio_vs_plain_1gpu': pr_1,
                'paired_ratio_vs_plain_8gpu': pr_8,
                'paired_ratio_diff_8gpu_minus_1gpu': pr_diff,
            })
            for pid in p_1:
                p_a1 = [r for r in a_1 if r['prompt_id'] == pid]
                p_a8 = [r for r in a_8 if r['prompt_id'] == pid]
                p_tps1 = _prompt_mean_tps(p_a1)
                p_tps8 = _prompt_mean_tps(p_a8) if p_a8 else None
                by_prompt.append({
                    'prompt_id': pid,
                    'condition': cond,
                    'arm': arm,
                    'tps_1gpu_mean': p_tps1,
                    'tps_8gpu_mean': p_tps8,
                    'tps_8gpu_over_1gpu_ratio': (p_tps8 / p_tps1) if (p_tps1 and p_tps8 is not None) else None,
                })

    if require_sharded and not reconciled:
        raise ValueError('sharded timing rows did not cover all calibration prompt/condition slots')

    timing_report_path = args.output / args.stage / 'timing.json'
    sharded_world_size = (
        json.loads(timing_report_path.read_text()).get('world_size')
        if timing_report_path.is_file() else None
    )
    affinity0 = session_state.get('cpu_affinity') if session_state else gpu_cpu_affinity(0)['cpu_affinity']
    payload = {
        'schema_version': 1,
        'scale': args.scale,
        'stage': args.stage,
        'status': 'complete' if reconciled else 'calibration_only_unreconciled',
        'reconciled_with_sharded_timing': reconciled,
        'sharded_world_size': sharded_world_size,
        'frozen_settings_hash': frozen_hash,
        'timing_harness_implementation_sha256': implementation_hash(args.stage),
        'algorithm_implementation_sha256': algorithm_hash(),
        'calibration_session': session_state['logical_session'],
        'physical_gpu': session_state.get('physical_gpu', 0),
        'numa_node': session_state.get('numa_node', 0),
        'cpu_affinity': affinity0,
        'cpu_affinity_applied': bool(session_state.get('cpu_affinity_applied', False)),
        'all_gpus_idle_verified': bool(cluster_idle_inventory is not None or session_state.get('all_gpus_idle_verified', False)),
        'peer_gpus_idle_verified': bool(cluster_idle_inventory is not None or session_state.get('all_gpus_idle_verified', False)),
        'warmup_completed': bool(session_state.get('warmup_completed', False)),
        'cluster_idle_inventory': cluster_idle_inventory or session_state.get('cluster_idle_inventory'),
        'repeats': cal_repeats,
        'warmups_per_arm_condition': args.timing_warmups,
        'selected_prompt_ids': [p['prompt_id'] for p in selected],
        'prompt_selection': selection_meta,
        'calibrated_timed_runs': len(cal_rows),
        'bundles': len(cal_rows) // len(arms),
        'raw_evidence': {
            'session_path': str(Path(session_path).resolve()),
            'session_sha256': file_sha256(session_path),
            'manifest_path': str(manifest_path.resolve()),
            'manifest_sha256': file_sha256(manifest_path),
            'timing_requests': len(cal_rows),
            'bundles': len(cal_rows) // len(arms),
        },
        'groups': groups,
        'by_prompt': by_prompt,
    }
    sess_fp = session_state.get('configuration', {}).get('configuration', {}).get('fastpath') if session_state else None
    if sess_fp is not None:
        payload['fastpath'] = sess_fp
    cal_out = args.output / args.stage / 'timing_isolation_calibration.json'
    atomic_json(cal_out, payload)
    return payload


def timing_calibration(runner, prompt_manifest, settings, cluster_idle_inventory=None):
    args = runner.args
    if args.world_size != 1 or args.rank != 0:
        raise ValueError('1-GPU isolated calibration requires world_size=1 and rank=0 on physical GPU 0')
    cal_repeats = getattr(args, 'calibration_repeats', 2)
    selected, selection_meta = select_calibration_prompts(args, prompt_manifest, settings)
    jobs = calibration_jobs(
        selected, args.stage, args.scale, args.seed,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )
    active_arms = _arms_from_args(args)
    frozen_hash = fingerprint(settings)
    session_path, session_state, epoch = comparison_session(runner, 'timing_calibration', {
        'frozen_settings_hash': frozen_hash,
        'prompt_manifest_hash': fingerprint(prompt_manifest),
        'selected_prompts': [p['prompt_id'] for p in selected],
        'repeats': cal_repeats,
        'warmups': args.timing_warmups,
    })
    if cluster_idle_inventory is not None:
        session_state['all_gpus_idle_verified'] = True
        session_state['cluster_idle_inventory'] = cluster_idle_inventory
        atomic_json(session_path, session_state)
    groups_by_prompt = defaultdict(list)
    for job in jobs:
        groups_by_prompt[(job['prompt']['prompt_id'], job['sampling_name'], job['seed'])].append(job)
    cal_bundle_specs = []
    need_warmup = False
    for group_index, same_prompt_jobs in enumerate(groups_by_prompt.values()):
        for repeat in range(cal_repeats):
            offset = (group_index + repeat) % len(same_prompt_jobs)
            comparisons = [(job, settings['selected'][job['arm']])
                           for job in same_prompt_jobs[offset:] + same_prompt_jobs[:offset]]
            prior = read_comparison_bundle(runner, 'timing_calibration', comparisons, frozen_hash, session_state, repeat, False)
            if prior is None:
                need_warmup = True
            cal_bundle_specs.append((repeat, comparisons, prior))
    if need_warmup:
        for condition in sorted({p['condition'] for p in selected}):
            prompt = next(p for p in selected if p['condition'] == condition)
            name, temperature, warp = sampling_conditions(condition, primary_only=True)[0]
            for _ in range(args.timing_warmups):
                for arm in active_arms:
                    job = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature,
                           'warper': warp, 'seed': args.seed, 'arm': arm, 'max_new_tokens_override': 64}
                    runner.run(job, settings['selected'][arm], profile=False)
        session_state['warmup_completed'] = True
        atomic_json(session_path, session_state)
    rows = []
    for repeat, comparisons, prior in cal_bundle_specs:
        results = (
            prior if prior is not None else comparison_bundle(
                runner, 'timing_calibration', comparisons, frozen_hash,
                session_path, session_state, epoch, repeat, profile=False,
            )
        )
        rows.extend(results)
    return reconcile_timing_isolation_calibration(
        args, prompt_manifest, settings,
        session_path=session_path, session_state=session_state,
        selected=selected, selection_meta=selection_meta, cal_rows=rows,
        cluster_idle_inventory=cluster_idle_inventory,
        require_sharded=getattr(args, 'reconcile_sharded_timing', False),
    )


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
    validate_checkpoint_hashes(settings['checkpoint_hashes'], runner.checkpoint_hashes)
    runner.checkpoint_hashes = dict(settings['checkpoint_hashes'])
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


M5_MATCHED_PAIRS = (
    ('lookahead_static_candidate', 'dense_candidate', 'frozen_C0_candidate_masked'),
    ('lookahead_rerelax_candidate', 'dense_candidate', 'frozen_C0_candidate_masked'),
    ('lookahead_static_full', 'dense_full', 'full_vocabulary'),
    ('lookahead_rerelax_full', 'dense_full', 'full_vocabulary'),
    ('exact_observed_pairs_online', 'signed_feature_hash_online', 'online_sparse_representation'),
    ('online_buffered_stride5', 'offline_online_stride5', 'online_stride5_update_mode'),
    ('online_bigram_kret256', 'online_bigram', 'online_bigram_topk_eviction'),
    ('online_bigram_kret128', 'online_bigram', 'online_bigram_topk_eviction'),
    ('online_bigram_count_gated', 'online_bigram', 'online_bigram_count_gate'),
    ('online_dbS_zero_init', 'online_bigram_count_gated', 'online_zero_init_dbS_vs_count_gated_bigram'),
    ('zero_offline_online_only', 'offline_online_stride1', 'online_zero_vs_checkpoint_offline_init'),
    ('online_dbS_zero_init', 'online_dbS', 'online_dbS_zero_vs_checkpoint_offline_init'),
)

M5_ONLINE_HEAD_TO_HEAD_PAIRS = (
    ('online_bigram', 'plain_dflash'),
    ('online_bigram_count_gated', 'plain_dflash'),
    ('offline_online_stride1', 'plain_dflash'),
    ('zero_offline_online_only', 'plain_dflash'),
    ('online_dbS', 'plain_dflash'),
    ('online_dbS_zero_init', 'plain_dflash'),
    ('online_bigram_count_gated', 'online_bigram'),
    ('zero_offline_online_only', 'online_bigram'),
    ('zero_offline_online_only', 'online_bigram_count_gated'),
    ('zero_offline_online_only', 'offline_online_stride1'),
    ('zero_offline_online_only', 'online_dbS_zero_init'),
    ('online_dbS_zero_init', 'online_bigram'),
    ('online_dbS_zero_init', 'online_bigram_count_gated'),
    ('online_dbS', 'online_bigram_count_gated'),
)
M5_EXTRA_ONLINE_HEAD_TO_HEAD_PAIRS = (
    ('online_bigram_kret256', 'plain_dflash'),
    ('online_bigram_kret128', 'plain_dflash'),
    ('online_bigram_kret256', 'online_bigram'),
    ('online_bigram_kret128', 'online_bigram'),
    ('online_bigram_kret256', 'online_bigram_kret128'),
)


def _block_committed_count(block: dict) -> int:
    if block.get('committed_tokens') is not None:
        return len(block['committed_tokens'])
    return int(block['accepted_length']) + (
        1 if block.get('replacement_token') is not None or block.get('bonus_token') is not None else 0
    )


def _request_offset_bin_stats(row: dict, bin_label: str) -> dict:
    blocks = row.get('blocks') or []
    bin_blocks = [b for b in blocks if output_bin(int(b['absolute_output_offset'])) == bin_label]
    acc_len = statistics.mean(float(b['accepted_length']) for b in bin_blocks) if bin_blocks else None
    yes, total = 0, 0
    for b in blocks:
        accepted = int(b['accepted_length'])
        base_off = int(b['absolute_output_offset'])
        for i in range(len(b['draft_tokens'])):
            if output_bin(base_off + i) == bin_label:
                yes += int(i < accepted)
                total += 1
    slot_rate = (yes / total) if total > 0 else None
    timed_blocks = [b for b in bin_blocks if b.get('round_elapsed_seconds') is not None]
    committed = sum(_block_committed_count(b) for b in timed_blocks)
    elapsed = sum(float(b['round_elapsed_seconds']) for b in timed_blocks)
    tps = (committed / elapsed) if (committed > 0 and elapsed > 0.0 and not row.get('phase_profile')) else None
    return {
        'reached_bin': bool(bin_blocks) or total > 0,
        'blocks_in_bin': len(bin_blocks),
        'slots_in_bin': total,
        'committed_tokens_in_bin': committed,
        'round_elapsed_seconds_in_bin': elapsed if timed_blocks else None,
        'accepted_length': acc_len,
        'slot_acceptance_rate': slot_rate,
        'tokens_per_second': tps,
    }


def summarize_offset_bin_slices(rows, *, isolated_timing: bool = False):
    """Slice acceptance and per-round wall-clock tok/s across OUTPUT_LABELS (`0-1K`..`16K+`)."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get('phase_profile'):
            continue
        warp = json.dumps(row['sampling_warp_config'], sort_keys=True)
        grouped[(row['condition'], row['temperature'], warp, row['arm'])].append(row)
    slices = []
    for (condition, temperature, warp, arm), group in sorted(grouped.items()):
        primary = [
            r for r in group
            if not (r['condition'].startswith('long_') and r['generation']['repetition_dominated'])
        ]
        bins_out = []
        for bin_label in OUTPUT_LABELS:
            acc_obs, slot_obs, tps_obs = [], [], []
            total_blocks = 0
            total_committed = 0
            reaching_requests = 0
            for r in primary:
                st = _request_offset_bin_stats(r, bin_label)
                if st['reached_bin']:
                    reaching_requests += 1
                total_blocks += st['blocks_in_bin']
                total_committed += st['committed_tokens_in_bin']
                if st['accepted_length'] is not None:
                    acc_obs.append({'prompt_id': r['prompt_id'], 'seed': r['seed'], 'accepted_length': st['accepted_length']})
                if st['slot_acceptance_rate'] is not None:
                    slot_obs.append({'prompt_id': r['prompt_id'], 'seed': r['seed'], 'slot_acceptance_rate': st['slot_acceptance_rate']})
                if st['tokens_per_second'] is not None:
                    tps_obs.append({'prompt_id': r['prompt_id'], 'seed': r['seed'], 'tokens_per_second': st['tokens_per_second']})
            bins_out.append({
                'bin': bin_label,
                'empty_bin': reaching_requests == 0,
                'requests_reaching_bin': reaching_requests,
                'primary_requests': len(primary),
                'blocks_in_bin': total_blocks,
                'committed_tokens_in_bin': total_committed,
                'accepted_length': prompt_bootstrap(acc_obs, 'accepted_length') if acc_obs else None,
                'slot_acceptance_rate': prompt_bootstrap(slot_obs, 'slot_acceptance_rate') if slot_obs else None,
                'tokens_per_second': prompt_bootstrap(tps_obs, 'tokens_per_second') if tps_obs else None,
                'timing_source': 'isolated_timing_rounds' if isolated_timing else 'observational_eval_rounds',
            })
        slices.append({
            'condition': condition,
            'temperature': temperature,
            'sampling_warp_config': json.loads(warp),
            'arm': arm,
            'primary_requests': len(primary),
            'repetition_excluded': len(group) - len(primary),
            'timing_isolated': isolated_timing,
            'bins': bins_out,
        })
    return slices


def load_isolated_timing_rows(args):
    """Load non-profile completed rows from the stage timing sessions/manifests across all ranks."""
    timing_dir = args.output / args.stage / 'timing'
    timing_report_path = args.output / args.stage / 'timing.json'
    try:
        if timing_report_path.is_file():
            raw_ev = json.loads(timing_report_path.read_text()).get('raw_evidence', {})
            session_paths = (
                [Path(p) for p in raw_ev['session_paths']]
                if raw_ev.get('session_paths')
                else ([Path(raw_ev['session_path'])] if raw_ev.get('session_path') else [])
            )
            if session_paths and all(sp.is_file() for sp in session_paths):
                out = []
                for sp in session_paths:
                    sess = json.loads(sp.read_text())
                    for b_val in sess.get('bundles', {}).values():
                        if b_val.get('status') != 'complete':
                            continue
                        for p_str in b_val.get('result_paths', []):
                            rp = Path(p_str)
                            if rp.is_file():
                                row = json.loads(rp.read_text())
                                if not row.get('phase_profile'):
                                    out.append(row)
                return out
        manifest_paths = sorted(timing_dir.glob('manifest-rank*.json'))
        if not manifest_paths:
            return []
        out = []
        for mp in manifest_paths:
            for entry in Manifest(mp).data.get('jobs', {}).values():
                identity = entry.get('identity', {})
                if entry.get('status') != 'complete' or identity.get('phase_profile'):
                    continue
                rpath = Path(entry.get('result_path', ''))
                if rpath.is_file():
                    out.append(json.loads(rpath.read_text()))
        return out
    except (OSError, ValueError, KeyError, TypeError):
        return []


def compute_head_to_head_online_offset_comparison(eval_rows, isolated_timing_rows=None):
    """Paired prompt-bootstrap M5 comparison by output-offset bin (`0-1K`, `1K-4K`, `4K-8K`, `8K-16K`, `16K+`)."""
    eval_by_slot = {}
    regimes = set()
    active_arms = set()
    for row in eval_rows:
        warp = json.dumps(row['sampling_warp_config'], sort_keys=True)
        regimes.add((row['condition'], row['temperature'], warp))
        active_arms.add(row['arm'])
        eval_by_slot[(row['condition'], row['temperature'], warp, row['prompt_id'], row['seed'], row['arm'])] = row
    timed_by_slot = {}
    for row in (isolated_timing_rows or []):
        if row.get('phase_profile'):
            continue
        warp = json.dumps(row['sampling_warp_config'], sort_keys=True)
        timed_by_slot[(row['condition'], row['temperature'], warp, row['prompt_id'],
                       row['seed'], row.get('repeat'), row['arm'])] = row
    comparisons = []
    for arm, ref_arm in M5_ONLINE_HEAD_TO_HEAD_PAIRS + M5_EXTRA_ONLINE_HEAD_TO_HEAD_PAIRS:
        if arm not in active_arms or ref_arm not in active_arms:
            continue
        for condition, temperature, warp in sorted(regimes):
            paired_eval = []
            excluded_eval = 0
            for (c, t, w, pid, seed, a), row in eval_by_slot.items():
                if a != arm or (c, t, w) != (condition, temperature, warp):
                    continue
                ref_row = eval_by_slot.get((c, t, w, pid, seed, ref_arm))
                if ref_row is None:
                    continue
                if condition.startswith('long_') and (
                    row['generation']['repetition_dominated'] or ref_row['generation']['repetition_dominated']
                ):
                    excluded_eval += 1
                    continue
                paired_eval.append((pid, seed, row, ref_row))
            if not paired_eval and excluded_eval == 0:
                continue
            paired_timed = []
            for (c, t, w, pid, seed, rep, a), row in timed_by_slot.items():
                if a != arm or (c, t, w) != (condition, temperature, warp):
                    continue
                ref_row = timed_by_slot.get((c, t, w, pid, seed, rep, ref_arm))
                if ref_row is None:
                    continue
                if condition.startswith('long_') and (
                    row['generation']['repetition_dominated'] or ref_row['generation']['repetition_dominated']
                ):
                    continue
                paired_timed.append((pid, seed, row, ref_row))
            bin_reports = []
            for bin_label in OUTPUT_LABELS:
                acc_deltas, slot_deltas, obs_tps_ratios = [], [], []
                arm_reach, ref_reach = 0, 0
                for pid, seed, row, ref_row in paired_eval:
                    st_arm = _request_offset_bin_stats(row, bin_label)
                    st_ref = _request_offset_bin_stats(ref_row, bin_label)
                    arm_reach += int(st_arm['reached_bin'])
                    ref_reach += int(st_ref['reached_bin'])
                    if st_arm['accepted_length'] is not None and st_ref['accepted_length'] is not None:
                        acc_deltas.append({'prompt_id': pid, 'seed': seed,
                                           'delta': st_arm['accepted_length'] - st_ref['accepted_length']})
                    if st_arm['slot_acceptance_rate'] is not None and st_ref['slot_acceptance_rate'] is not None:
                        slot_deltas.append({'prompt_id': pid, 'seed': seed,
                                            'delta': st_arm['slot_acceptance_rate'] - st_ref['slot_acceptance_rate']})
                    if st_arm['tokens_per_second'] and st_ref['tokens_per_second']:
                        obs_tps_ratios.append({'prompt_id': pid, 'seed': seed,
                                               'ratio': st_arm['tokens_per_second'] / st_ref['tokens_per_second']})
                iso_tps_ratios, iso_tps_deltas, iso_arm_tps, iso_ref_tps = [], [], [], []
                for pid, seed, t_row, t_ref in paired_timed:
                    st_arm = _request_offset_bin_stats(t_row, bin_label)
                    st_ref = _request_offset_bin_stats(t_ref, bin_label)
                    if st_arm['tokens_per_second'] and st_ref['tokens_per_second']:
                        iso_arm_tps.append({'prompt_id': pid, 'seed': seed, 'tps': st_arm['tokens_per_second']})
                        iso_ref_tps.append({'prompt_id': pid, 'seed': seed, 'tps': st_ref['tokens_per_second']})
                        iso_tps_ratios.append({'prompt_id': pid, 'seed': seed,
                                               'ratio': st_arm['tokens_per_second'] / st_ref['tokens_per_second']})
                        iso_tps_deltas.append({'prompt_id': pid, 'seed': seed,
                                               'delta': st_arm['tokens_per_second'] - st_ref['tokens_per_second']})
                bin_reports.append({
                    'bin': bin_label,
                    'empty_bin': len(acc_deltas) == 0 and len(slot_deltas) == 0,
                    'arm_requests_reaching_bin': arm_reach,
                    'reference_requests_reaching_bin': ref_reach,
                    'paired_requests_in_bin': len(acc_deltas),
                    'accepted_length_delta': prompt_bootstrap(acc_deltas, 'delta') if acc_deltas else None,
                    'slot_acceptance_rate_delta': prompt_bootstrap(slot_deltas, 'delta') if slot_deltas else None,
                    'observational_tokens_per_second_ratio': prompt_bootstrap(obs_tps_ratios, 'ratio') if obs_tps_ratios else None,
                    'isolated_timing_tokens_per_second': {
                        'arm': prompt_bootstrap(iso_arm_tps, 'tps'),
                        'reference_arm': prompt_bootstrap(iso_ref_tps, 'tps'),
                        'paired_ratio': prompt_bootstrap(iso_tps_ratios, 'ratio'),
                        'paired_delta': prompt_bootstrap(iso_tps_deltas, 'delta'),
                    } if iso_tps_ratios else None,
                })
            comparisons.append({
                'arm': arm,
                'reference_arm': ref_arm,
                'condition': condition,
                'temperature': temperature,
                'sampling_warp_config': json.loads(warp),
                'matched_pairs': len(paired_eval),
                'repetition_excluded_pairs': excluded_eval,
                'bins': bin_reports,
            })
    return comparisons


def compute_matched_support_comparisons(rows, timed_report=None):
    """Compute paired prompt-bootstrap M5 comparisons under matched support/representation."""
    by_slot = {}
    regimes = set()
    for row in rows:
        warp = json.dumps(row['sampling_warp_config'], sort_keys=True)
        regimes.add((row['condition'], row['temperature'], warp))
        by_slot[(row['condition'], row['temperature'], warp, row['prompt_id'], row['seed'], row['arm'])] = row
    timed_groups = {}
    if timed_report and timed_report.get('groups'):
        for g in timed_report['groups']:
            w = json.dumps(g['sampling_warp_config'], sort_keys=True)
            timed_groups[(g['condition'], g['temperature'], w, g['arm'])] = g
    comparisons = []
    for arm, ref_arm, support_family in M5_MATCHED_PAIRS:
        for condition, temperature, warp in sorted(regimes):
            acc_deltas, fwd_deltas = [], []
            excluded = 0
            for (c, t, w, pid, seed, a), row in by_slot.items():
                if a != arm or (c, t, w) != (condition, temperature, warp):
                    continue
                ref_row = by_slot.get((c, t, w, pid, seed, ref_arm))
                if ref_row is None:
                    continue
                if condition.startswith('long_') and (
                    row['generation']['repetition_dominated'] or ref_row['generation']['repetition_dominated']
                ):
                    excluded += 1
                    continue
                acc_deltas.append({'prompt_id': pid, 'seed': seed,
                                   'delta': row['mean_accepted_length'] - ref_row['mean_accepted_length']})
                fwd_deltas.append({'prompt_id': pid, 'seed': seed,
                                   'delta': row['target_forwards_per_generated_token'] - ref_row['target_forwards_per_generated_token']})
            if not acc_deltas and excluded == 0:
                continue
            tg_arm = timed_groups.get((condition, temperature, warp, arm))
            tg_ref = timed_groups.get((condition, temperature, warp, ref_arm))
            tps_arm = tg_arm.get('matched_arm_tokens_per_second') if tg_arm else None
            tps_ref = tg_ref.get('matched_arm_tokens_per_second') if tg_ref else None
            comparisons.append({
                'arm': arm,
                'reference_arm': ref_arm,
                'matched_support_family': support_family,
                'condition': condition,
                'temperature': temperature,
                'sampling_warp_config': json.loads(warp),
                'matched_pairs': len(acc_deltas),
                'repetition_excluded_pairs': excluded,
                'mean_accepted_length_delta': prompt_bootstrap(acc_deltas, 'delta') if acc_deltas else None,
                'target_forwards_per_generated_token_delta': prompt_bootstrap(fwd_deltas, 'delta') if fwd_deltas else None,
                'isolated_timing_tokens_per_second': {
                    'arm': tps_arm,
                    'reference_arm': tps_ref,
                    'ratio': (tps_arm / tps_ref) if (tps_arm and tps_ref) else None,
                } if (tg_arm is not None and tg_ref is not None) else None,
            })
    return comparisons


def beta_sweep(runner, prompt_manifest, settings):
    """Resumable inference beta sweep across configured arms (`prompt.md:831`)."""
    args = runner.args
    split = args.beta_sweep_split
    prompts = select_prompts(prompt_manifest, split, args.scale, args.standard_dev_prompts, args.long_dev_prompts)
    prompt_rank = {p['prompt_id']: i % args.world_size for i, p in enumerate(prompts)}
    active_arms = _arms_from_args(args)
    if args.beta_sweep_arms:
        unknown = set(args.beta_sweep_arms) - set(active_arms)
        if unknown:
            raise ValueError(f'unknown beta_sweep arm(s): {sorted(unknown)}')
        sweep_arms = tuple(a for a in args.beta_sweep_arms if a != 'plain_dflash')
    else:
        sweep_arms = tuple(
            a for a in active_arms
            if a != 'plain_dflash' and (
                'beta' in settings['selected'].get(a, {})
                or a in ('dense', 'dense_sparse', 'offline_static', 'dense_full', 'dense_candidate')
            )
        )
    betas = [float(b) for b in args.beta_grid]
    if not betas or any(not math.isfinite(b) or b < 0 for b in betas):
        raise ValueError('beta_grid must contain finite nonnegative values')
    frozen_hash = fingerprint(settings)
    rows = []
    for prompt in prompts:
        if prompt_rank[prompt['prompt_id']] != args.rank:
            continue
        cond_list = sampling_conditions(prompt['condition'], primary_only=(split == 'dev'))
        for name, temperature, warp in cond_list:
            seeds = 1 if (temperature == 0 or args.scale == 'smoke' or split == 'dev') else (
                3 if prompt['condition'] == 'standard_nonthinking' or args.scale == 'full' else 1
            )
            for s_idx in range(seeds):
                base_job = {'prompt': prompt, 'sampling_name': name, 'temperature': temperature,
                            'warper': warp, 'seed': args.seed + s_idx}
                plain_row = execute_job(runner, dict(base_job, arm='plain_dflash'), {}, 'beta_sweep', frozen_hash)
                rows.append(dict(plain_row, swept_beta=0.0))
                for arm in sweep_arms:
                    base_setting = dict(settings['selected'][arm])
                    for beta_val in betas:
                        setting = dict(base_setting, beta=beta_val)
                        job = dict(base_job, arm=arm, force_setting_beta=True)
                        row = execute_job(runner, job, setting, 'beta_sweep', frozen_hash)
                        rows.append(dict(row, swept_beta=beta_val))
    by_curve = defaultdict(list)
    for row in rows:
        if row['condition'].startswith('long_') and row['generation']['repetition_dominated']:
            continue
        key = (row['arm'], row['swept_beta'], row['condition'], row['temperature'],
               json.dumps(row['sampling_warp_config'], sort_keys=True))
        by_curve[key].append(row)
    curves = []
    for (arm, beta_val, condition, temperature, warp), group in sorted(by_curve.items()):
        curves.append({
            'arm': arm,
            'beta': beta_val,
            'condition': condition,
            'temperature': temperature,
            'sampling_warp_config': json.loads(warp),
            'requests': len(group),
            'mean_accepted_length': prompt_bootstrap(group, 'mean_accepted_length'),
            'target_forwards_per_generated_token': prompt_bootstrap(group, 'target_forwards_per_generated_token'),
            'tokens_per_second_observational': prompt_bootstrap(group, 'tokens_per_second'),
        })
    report = {
        'scale': args.scale,
        'stage': args.stage,
        'mode': 'beta_sweep',
        'status': 'complete' if args.world_size == 1 else 'rank_complete',
        'split': split,
        'rank': args.rank,
        'world_size': args.world_size,
        'beta_grid': betas,
        'swept_arms': list(sweep_arms),
        'frozen_settings_hash': frozen_hash,
        'requests': len(rows),
        'curves': curves,
    }
    out_name = 'beta_sweep.json' if args.world_size == 1 else f'beta_sweep-rank{args.rank}.json'
    atomic_json(args.output / args.stage / out_name, report)
    return report


def server_eval(args, prompt_manifest):
    """Resumable M5 OpenAI-compatible serving benchmark (`Qwen35ServerClient`)."""
    if args.stage != 'm5':
        raise ValueError('--mode server_eval is restricted to --stage m5')
    if not getattr(args, 'm5_server_url', None):
        raise ValueError('--mode server_eval requires --m5_server_url')
    from .m5_baselines import Qwen35ServerClient
    client = Qwen35ServerClient(
        args.m5_server_url,
        model=args.m5_server_model,
        api_key=args.m5_server_api_key,
        timeout_seconds=args.m5_server_timeout,
    )
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    prompt_rank = {p['prompt_id']: i % args.world_size for i, p in enumerate(prompts)}
    directory = args.output / 'm5' / 'server_eval'
    manifest = Manifest(directory / f'manifest-rank{args.rank}.json')
    rows = []
    for prompt in prompts:
        if prompt_rank[prompt['prompt_id']] != args.rank:
            continue
        cap = args.smoke_tokens if args.scale == 'smoke' else (
            args.long_max_new_tokens if prompt['condition'].startswith('long_') else args.max_new_tokens
        )
        for name, temperature, warp in sampling_conditions(prompt['condition']):
            seeds = 1 if (temperature == 0 or args.scale == 'smoke') else (
                3 if prompt['condition'] == 'standard_nonthinking' or args.scale == 'full' else 1
            )
            for s_idx in range(seeds):
                seed_val = args.seed + s_idx
                identity = {
                    'stage': 'm5', 'mode': 'server_eval', 'scale': args.scale,
                    'model_pair': [client.model, 'server_managed_drafter'],
                    'endpoint': client.base_url, 'server_model': client.model,
                    'prompt': prompt['prompt_id'],
                    'prompt_id': prompt['prompt_id'], 'prompt_payload_hash': fingerprint(prompt),
                    'condition': prompt['condition'], 'sampling_name': name,
                    'temperature': temperature, 'sampling_warp_config': asdict(warp),
                    'verification_rule': 'greedy' if temperature == 0 else 'ratio',
                    'proposal_mode': 'openai_server_dflash',
                    'checkpoint': client.base_url,
                    'adapter_version': 0,
                    'max_tokens': cap, 'seed': seed_val,
                }
                key = manifest.key(identity)
                path = directory / f'rank{args.rank}' / f'{key}.json'
                if manifest.completed(identity):
                    entry = manifest.data['jobs'][key]
                    if entry.get('result_sha256') and entry['result_sha256'] != file_sha256(path):
                        raise ValueError('completed server_eval result checksum changed')
                    rows.append(json.loads(path.read_text()))
                    continue
                resp = client.chat_completion(
                    prompt['text'],
                    temperature=temperature,
                    top_p=warp.top_p,
                    top_k=-1 if warp.top_k is None else warp.top_k,
                    max_tokens=cap,
                    seed=seed_val,
                    enable_thinking=(prompt['condition'] == 'long_thinking'),
                )
                row = {
                    'identity': identity,
                    'prompt_id': prompt['prompt_id'],
                    'condition': prompt['condition'],
                    'sampling_name': name,
                    'temperature': temperature,
                    'sampling_warp_config': asdict(warp),
                    'seed': seed_val,
                    **resp,
                }
                atomic_json(path, row)
                manifest.data['jobs'][key] = {
                    'identity': identity, 'status': 'complete',
                    'result_path': str(path), 'result_sha256': file_sha256(path),
                }
                atomic_json(manifest.path, manifest.data)
                rows.append(row)
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[(row['condition'], row['temperature'], json.dumps(row['sampling_warp_config'], sort_keys=True))].append(row)
    groups = []
    for (condition, temperature, warp), group in sorted(by_cond.items()):
        tps_rows = [{'prompt_id': r['prompt_id'], 'seed': r['seed'], 'tokens_per_second': r['tokens_per_second']} for r in group]
        logical_acc = [
            {'prompt_id': r['prompt_id'], 'seed': r['seed'],
             'logical_mean_accepted_length': r['server_meta_info']['logical_mean_accepted_length']}
            for r in group if r.get('server_meta_info', {}).get('logical_mean_accepted_length') is not None
        ]
        groups.append({
            'condition': condition,
            'temperature': temperature,
            'sampling_warp_config': json.loads(warp),
            'requests': len(group),
            'tokens_per_second': prompt_bootstrap(tps_rows, 'tokens_per_second'),
            'logical_mean_accepted_length': prompt_bootstrap(logical_acc, 'logical_mean_accepted_length') if logical_acc else None,
        })
    report = {
        'scale': args.scale,
        'stage': 'm5',
        'mode': 'server_eval',
        'status': 'complete' if args.world_size == 1 else 'rank_complete',
        'endpoint': client.base_url,
        'server_model': client.model,
        'rank': args.rank,
        'world_size': args.world_size,
        'completed_requests': len(rows),
        'groups': groups,
    }
    out_name = 'server_eval.json' if args.world_size == 1 else f'server_eval-rank{args.rank}.json'
    atomic_json(args.output / 'm5' / out_name, report)
    return report


def aggregate(args, prompt_manifest, settings):
    from types import SimpleNamespace
    runner = SimpleNamespace(args=args, tiny=args.scale == 'smoke', checkpoint_hashes=settings['checkpoint_hashes'],
                             model_pair=settings['model_pair'], model_revisions=settings['model_revisions'],
                             generation_environment=settings.get('generation_environment'))
    prompts = select_prompts(prompt_manifest, 'test', args.scale)
    jobs = evaluation_jobs(
        prompts, args.stage, args.scale, args.seed,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )
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
    sharded_plan_path = args.output / args.stage / 'timing' / 'sharding_plan.json'
    if sharded_plan_path.is_file():
        summarize_sharded_timing(args, prompt_manifest, settings)
    cal_path = args.output / args.stage / 'timing_isolation_calibration.json'
    timing_path = args.output / args.stage / 'timing.json'
    if getattr(args, 'reconcile_sharded_timing', False) or (cal_path.is_file() and timing_path.is_file()):
        reconcile_timing_isolation_calibration(
            args, prompt_manifest, settings,
            require_sharded=getattr(args, 'reconcile_sharded_timing', False),
        )
        if sharded_plan_path.is_file():
            summarize_sharded_timing(args, prompt_manifest, settings)
    timed = json.loads(timing_path.read_text()) if timing_path.exists() else None
    if timed is not None and cal_path.is_file() and timed.get('isolation_calibration', {}).get('sha256') != file_sha256(cal_path):
        cal_data = json.loads(cal_path.read_text())
        timed['isolation_calibration'] = {
            'path': str(cal_path.resolve()),
            'sha256': file_sha256(cal_path),
            'status': cal_data.get('status'),
            'calibrated_timed_runs': cal_data.get('calibrated_timed_runs'),
        }
        atomic_json(timing_path, timed)
    timing_evidence = audit_timing_evidence(runner, prompt_manifest, settings, timed)
    timing_valid = timing_evidence['status'] == 'complete'
    feature_context = m4_feature_context(args, prompt_manifest, settings, prompts) if args.stage == 'm4' else None
    feature_context_valid = feature_context is None or feature_context['status'] == 'complete'
    greedy_report = greedy_output_report(rows, jobs, args.output)
    greedy_path = args.output / args.stage / 'greedy_output_comparisons.json'
    atomic_json(greedy_path, greedy_report)
    greedy_complete = greedy_report['status'] == 'complete'
    status = 'complete' if not missing and timing_valid and feature_context_valid and greedy_complete else 'incomplete'
    if args.stage == 'm5':
        active_set = set(_arms_from_args(args))
        ext_map = [
            ('token_recycling_tree', 'full tree-based Token Recycling'),
            ('tts', 'Test-Time Speculation'),
            ('sharp_draft', 'SharpDraft'),
        ]
        external_not_run = [label for arm_key, label in ext_map if arm_key not in active_set]
    else:
        external_not_run = ['full tree-based Token Recycling', 'Test-Time Speculation', 'SharpDraft']
    isolation_calibration = None
    if cal_path.is_file():
        cal_data = json.loads(cal_path.read_text())
        isolation_calibration = {
            'path': str(cal_path.resolve()),
            'sha256': file_sha256(cal_path),
            'status': cal_data.get('status'),
            'calibrated_timed_runs': cal_data.get('calibrated_timed_runs'),
        }
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
              'timing_isolation_calibration': isolation_calibration,
              'external_baselines_not_run': external_not_run,
              'baseline_naming': 'online_bigram is the normalized committed-transition baseline, not full Token Recycling',
              'long_condition': sorted({p['condition'] for p in prompts if p['condition'].startswith('long_')}),
              'thinking_drafter_distribution': 'out_of_distribution' if any(p['condition'] == 'long_thinking' for p in prompts) else 'long_thinking unavailable or not selected; see prompt manifest'}
    if args.stage == 'm5':
        report['matched_support_comparisons'] = compute_matched_support_comparisons(
            rows, timed if timing_valid else None
        )
        report['offset_bin_slices'] = summarize_offset_bin_slices(rows, isolated_timing=False)
        isolated_rows = load_isolated_timing_rows(args) if timing_valid else []
        report['head_to_head_online_offset_comparison'] = compute_head_to_head_online_offset_comparison(
            rows, isolated_rows
        )
        bigram_mem = summarize_bigram_memory(rows)
        if bigram_mem:
            report['bigram_memory_ablation'] = bigram_mem
        if settings.get('triton_gate_report') is not None:
            report['triton_gate_report'] = settings['triton_gate_report']
        if settings.get('torch_compile_gate_report') is not None:
            report['torch_compile_gate_report'] = settings['torch_compile_gate_report']
    atomic_json(args.output / args.stage / 'summary.json', report)
    return report


def summarize_bigram_memory(rows) -> list[dict]:
    """Summarize per-request OnlineBigram exact ledger vs top-k_ret retained-prior footprint."""
    grouped = defaultdict(list)
    for row in rows:
        bg = row.get('bigram')
        if not isinstance(bg, dict):
            continue
        warp = json.dumps(row['sampling_warp_config'], sort_keys=True)
        grouped[(row['condition'], row['temperature'], warp, row['arm'])].append(row)
    summaries = []
    for (condition, temperature, warp, arm), group in sorted(grouped.items()):
        primary = [
            r for r in group
            if not (condition.startswith('long_') and r['generation']['repetition_dominated'])
        ]
        if not primary:
            continue
        exact_list = [int(r['bigram']['exact_ledger_pairs']) for r in primary]
        ret_list = [int(r['bigram']['retained_prior_pairs']) for r in primary]
        evict_list = [int(r['bigram']['exact_ledger_pairs']) - int(r['bigram']['retained_prior_pairs']) for r in primary]
        trans_list = [int(r['bigram']['counted_transitions']) for r in primary]
        frac_list = [
            (float(ret) / float(ex)) if ex > 0 else 1.0
            for ret, ex in zip(ret_list, exact_list)
        ]
        sample_bg = primary[0]['bigram']
        summaries.append({
            'condition': condition,
            'temperature': temperature,
            'sampling_warp_config': json.loads(warp),
            'arm': arm,
            'k_ret': sample_bg.get('k_ret'),
            'count_gate_kappa': sample_bg.get('count_gate_kappa'),
            'ledger_retained_for_exact_counts': bool(sample_bg.get('ledger_retained_for_exact_counts', True)),
            'primary_requests': len(primary),
            'repetition_excluded': len(group) - len(primary),
            'counted_transitions_mean': statistics.mean(trans_list),
            'exact_ledger_pairs_mean': statistics.mean(exact_list),
            'exact_ledger_pairs_max': max(exact_list),
            'retained_prior_pairs_mean': statistics.mean(ret_list),
            'retained_prior_pairs_max': max(ret_list),
            'evicted_prior_pairs_mean': statistics.mean(evict_list),
            'evicted_prior_pairs_max': max(evict_list),
            'retained_pair_fraction_mean': statistics.mean(frac_list),
            'retained_prior_payload_bytes_mean': statistics.mean(r * 12 for r in ret_list),
            'exact_ledger_payload_bytes_mean': statistics.mean(x * 16 for x in exact_list),
            'memory_disclosure': 'Full exact ledger is retained in diagnostic mode so reappearing tokens receive exact counts; retained_prior_pairs measures the active top-k_ret prior footprint.',
        })
    return summaries


def smoke_manifest():
    return {'scale': 'smoke', 'prompts': [{'prompt_id': f'synthetic-{split}-{condition}-{i}', 'split': split,
        'condition': condition, 'text': f'Synthetic {i}', 'dataset': 'builtin', 'reference_selected': True}
        for split in ('dev', 'test') for condition in ('standard_nonthinking', 'long_thinking') for i in range(2)]}


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument('--stage', choices=['m3', 'm4', 'm5'], default='m3')
    parser.add_argument('--mode', choices=['tune', 'tune_summarize', 'evaluate', 'timing', 'timing_calibration', 'summarize', 'all',
                                           'beta_sweep', 'server_eval'], default='all')
    parser.add_argument('--fresh_session', action='store_true', help='start new dev/timing measurements instead of resuming complete comparison bundles')
    parser.add_argument('--logical_session', type=str, default=None, help='shared logical session identifier across sharded timing ranks')
    parser.add_argument('--reconcile_sharded_timing', action='store_true', help='reconcile 1-GPU isolated calibration against sharded M4 timing')
    parser.add_argument('--calibration_repeats', type=int, default=2)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--schedule', choices=['modulo', 'dynamic'], default='modulo')
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--cuda_graph', action='store_true', default=False)
    parser.add_argument('--bf16_gemm', action='store_true', default=False)
    parser.add_argument('--compile_drafter', action='store_true', default=False)
    parser.add_argument('--compile_online', action='store_true', default=False)
    parser.add_argument('--fast_decode', action='store_true', default=False)
    parser.add_argument('--no_analytic_online_grad', action='store_false', dest='analytic_online_grad', default=True)
    parser.add_argument('--prompts_manifest', type=Path, default=ROOT / 'artifacts/prompts.json')
    parser.add_argument('--dense_checkpoint', type=Path, default=ROOT / 'artifacts/m2/dense.pt')
    parser.add_argument('--sparse_checkpoint', type=Path, default=ROOT / 'artifacts/m2/dense_sparse.pt')
    parser.add_argument('--projection', type=Path, default=ROOT / 'artifacts/h_proj.pt')
    parser.add_argument('--feature_diagnostic', type=Path, help='M4 plain-DFlash M1b report; default artifacts/diagnostic_<scale>.json')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--long_max_new_tokens', type=int, default=24576)
    parser.add_argument('--smoke_tokens', type=int, default=32)
    parser.add_argument('--standard_dev_prompts', type=int, default=8)
    parser.add_argument('--long_dev_prompts', type=int, default=None)
    parser.add_argument('--timing_prompts', type=int, default=0, help='0 times every required test prompt; positive values are incomplete pilot subsets')
    parser.add_argument('--timing_repeats', type=int, default=3)
    parser.add_argument('--timing_warmups', type=int, default=1)
    parser.add_argument('--beta_grid', type=float, nargs='+', default=[0., .25, .5, .75, 1., 1.25, 1.5])
    parser.add_argument('--beta_sweep_split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--beta_sweep_arms', nargs='+', default=None)
    parser.add_argument('--m5_arms', nargs='+', default=None)
    parser.add_argument('--m5_oracle_checkpoint', type=Path, default=None)
    parser.add_argument('--sparse_overflow_policy', choices=['freeze_new', 'error'], default='freeze_new')
    parser.add_argument('--m5_triton', action='store_true', default=False)
    parser.add_argument('--m5_torch_compile', action='store_true', default=False)
    parser.add_argument('--m5_server_url', default=None)
    parser.add_argument('--m5_server_model', default='default')
    parser.add_argument('--m5_server_api_key', default='EMPTY')
    parser.add_argument('--m5_server_timeout', type=float, default=600.)
    parser.add_argument('--bigram_count_gate_kappa', type=float, default=None,
                        help='M5 count-confidence gate kappa > 0 for online_bigram_count_gated')
    parser.add_argument('--bigram_k_ret', type=int, default=None,
                        help='M5 top-k_ret eviction retention count >= 1 for online_bigram_kret256 (default 256)')
    parser.add_argument('--checkpoint_scale', type=float, default=None,
                        help='M5 scalar multiplier applied to loaded dense/dense_sparse parameters (e.g. 0.20)')
    parser.add_argument('--m5_beta_candidates', type=float, nargs=2, default=None,
                        help='M5 two-point dev tuning beta candidates for offline/lookahead arms')
    parser.add_argument('--m5_beta_relax', type=float, default=None,
                        help='M5 shared beta_relax for matched-support C0[j] candidate and lookahead arms')
    args = parse_config(parser, argv)
    if args.long_dev_prompts is None:
        args.long_dev_prompts = 32 if args.scale == 'full' else 12
    for name in ('prompts_manifest', 'dense_checkpoint', 'sparse_checkpoint', 'projection'):
        setattr(args, name, Path(getattr(args, name)))
    if args.m5_oracle_checkpoint is not None:
        args.m5_oracle_checkpoint = Path(args.m5_oracle_checkpoint)
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error('rank must lie in 0..world_size-1')
    if args.timing_repeats < 2 or args.timing_warmups < 1 or args.timing_prompts < 0 or args.calibration_repeats < 2:
        parser.error('timing requires repeats>=2, calibration_repeats>=2, warmups>=1 and prompts>=0 (0=all)')
    if min(args.max_new_tokens, args.long_max_new_tokens, args.smoke_tokens) < 1:
        parser.error('generation caps must be positive')
    if args.mode in ('timing_calibration', 'all', 'tune_summarize') and (args.world_size != 1 or args.rank != 0):
        parser.error('timing_calibration, tune_summarize, and all require world_size=1 and rank=0')
    sharded_tune_ok = (args.stage == 'm5' or args.scale == 'full')
    if args.mode == 'tune':
        if not sharded_tune_ok and (args.world_size != 1 or args.rank != 0):
            parser.error('M3/M4 reference/smoke tune requires world_size=1 and rank=0')
        if sharded_tune_ok and args.world_size > 8:
            parser.error('sharded tune world_size must lie in 1..8')
    if args.mode == 'timing' and args.world_size > 8:
        parser.error('timing world_size must lie in 1..8')
    if args.bigram_count_gate_kappa is not None:
        if not math.isfinite(args.bigram_count_gate_kappa) or args.bigram_count_gate_kappa <= 0:
            parser.error('--bigram_count_gate_kappa must be finite and > 0')
    if args.bigram_k_ret is not None and args.bigram_k_ret < 1:
        parser.error('--bigram_k_ret must be an integer >= 1')
    if args.checkpoint_scale is not None:
        if not math.isfinite(args.checkpoint_scale) or args.checkpoint_scale < 0:
            parser.error('--checkpoint_scale must be finite and >= 0')
    if args.m5_beta_candidates is not None:
        b0, b1 = args.m5_beta_candidates
        if not (math.isfinite(b0) and math.isfinite(b1)) or b0 < 0 or b1 < 0 or b0 == b1:
            parser.error('--m5_beta_candidates requires two distinct finite nonnegative values')
    if args.m5_beta_relax is not None:
        if not math.isfinite(args.m5_beta_relax) or args.m5_beta_relax < 0:
            parser.error('--m5_beta_relax must be finite and >= 0')
    if args.stage != 'm5' and (args.m5_arms or args.m5_oracle_checkpoint or args.m5_triton
                               or args.m5_torch_compile
                               or args.mode == 'server_eval'
                               or (args.mode == 'tune_summarize' and not sharded_tune_ok)
                               or args.bigram_count_gate_kappa is not None
                               or args.bigram_k_ret is not None
                               or args.checkpoint_scale is not None
                               or args.m5_beta_candidates is not None
                               or args.m5_beta_relax is not None):
        parser.error('M5-specific options (--m5_*, --checkpoint_scale, --bigram_count_gate_kappa, --bigram_k_ret, --mode server_eval) require --stage m5 (and --mode tune_summarize requires --stage m5 or --scale full)')
    record_run(args, f'{args.stage}-{args.mode}')
    torch.set_num_threads(1)
    prompt_manifest = smoke_manifest() if args.scale == 'smoke' else json.loads(args.prompts_manifest.read_text())
    if args.mode == 'server_eval':
        if args.scale != 'smoke':
            n_std_eval, n_long_eval = (128, 64) if args.scale == 'full' else (64, 32)
            preflight(args, n_std_eval * args.max_new_tokens + n_long_eval * args.long_max_new_tokens)
        result = server_eval(args, prompt_manifest)
        print(json.dumps({'stage': args.stage, 'scale': args.scale, 'mode': args.mode, 'status': result['status']}))
        return
    cluster_idle_inventory = None
    if args.scale != 'smoke':
        if args.mode not in ('summarize', 'tune_summarize'):
            n_std_eval, n_long_eval = (128, 64) if args.scale == 'full' else (64, 32)
            expected_tokens = (
                args.standard_dev_prompts * args.max_new_tokens + args.long_dev_prompts * args.long_max_new_tokens
                if args.mode == 'tune'
                else n_std_eval * args.max_new_tokens + n_long_eval * args.long_max_new_tokens
            )
            preflight(args, expected_tokens)
        if args.mode == 'timing_calibration':
            if not os.environ.get('CUDA_VISIBLE_DEVICES') and str(args.device).startswith('cuda'):
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            cluster_idle_inventory = assert_all_gpus_idle(torch.device(args.device))
        elif args.mode in ('tune', 'timing', 'all'):
            from .bench_adapter import assert_idle
            assert_idle(torch.device(args.device))
    settings_path = args.output / args.stage / 'settings.json'
    runner = None if args.mode in ('summarize', 'tune_summarize') else EvaluationRunner(args)
    if args.mode == 'tune_summarize':
        settings = summarize_sharded_tune(args, prompt_manifest)
    elif args.mode in ('tune', 'all'):
        tune_out = tune(runner, prompt_manifest)
        if args.mode == 'tune' and args.world_size > 1:
            print(json.dumps({
                'stage': args.stage, 'scale': args.scale, 'mode': args.mode,
                'rank': args.rank, 'world_size': args.world_size, 'status': tune_out['status'],
            }))
            return
        settings = tune_out
    else:
        settings = json.loads(settings_path.read_text())
    if args.scale == 'smoke':
        if runner is not None:
            current_hashes = runner.checkpoint_hashes
        else:
            current_hashes = dict(settings['checkpoint_hashes'], implementation=implementation_hash(args.stage))
            if args.stage == 'm5':
                if args.checkpoint_scale is not None:
                    current_hashes['checkpoint_scale'] = float(args.checkpoint_scale)
                else:
                    current_hashes.pop('checkpoint_scale', None)
    else:
        current_hashes = checkpoint_hashes(args)
    validate_frozen_settings(
        settings, args.stage, args.scale, current_hashes, prompt_manifest,
        generation_caps(args),
        runner.model_pair if runner is not None else [args.target, args.drafter] if args.scale != 'smoke' else settings['model_pair'],
        runner.model_revisions if runner is not None else None,
        runner.generation_environment if runner is not None else None,
        m5_arms=getattr(args, 'm5_arms', None),
        include_long_oracle=bool(getattr(args, 'm5_oracle_checkpoint', None)),
    )
    if args.mode not in ('tune', 'all') and runner is not None:
        runner.checkpoint_hashes = dict(settings['checkpoint_hashes'])
    if args.mode == 'beta_sweep':
        result = beta_sweep(runner, prompt_manifest, settings)
        print(json.dumps({'stage': args.stage, 'scale': args.scale, 'mode': args.mode, 'status': result['status']}))
        return
    if args.mode in ('evaluate', 'all'):
        evaluate(runner, prompt_manifest, settings)
    timing_res = None
    if args.mode in ('timing', 'all'):
        timing_res = timing(runner, prompt_manifest, settings)
    if args.mode == 'timing_calibration':
        cal_res = timing_calibration(runner, prompt_manifest, settings, cluster_idle_inventory=cluster_idle_inventory)
        if (args.output / args.stage / 'timing.json').is_file():
            aggregate(args, prompt_manifest, settings)
        result = cal_res
    elif args.mode == 'timing' and args.world_size > 1:
        result = timing_res
    elif args.mode in ('evaluate', 'timing', 'summarize', 'all'):
        result = aggregate(args, prompt_manifest, settings)
    else:
        result = {'stage': args.stage, 'scale': args.scale, 'status': 'settings_frozen'}
    print(json.dumps({'stage': args.stage, 'scale': args.scale, 'mode': args.mode, 'status': result['status']}))


if __name__ == '__main__':
    main()
