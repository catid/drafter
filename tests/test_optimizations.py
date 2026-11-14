from __future__ import annotations

import copy
import importlib.util
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from edraft.adapter import (DenseAdapter, _pair_message, block_correction, block_unary, full_kl,
                            loss_weights, prev_token, slot_correction, top_candidates)
from edraft.collect import audit as collect_audit, calibration as collect_calibration, collect as collect_traces, prepare_args as prepare_collect_args, parser as collect_parser
from edraft.dflash_loop import generate_toy
from edraft.diagnose import _longest_periodic_run
from edraft.eval_closed_loop import (EvaluationRunner, aggregate as eval_aggregate, arms_for,
                                     audit_timing_evidence, candidate_settings, evaluate as run_evaluate,
                                     main as eval_main, sampling_conditions, smoke_manifest,
                                     timing as run_timing, tune as run_tune)
from edraft.fastpath import FastPathConfig, FastPathRuntime
from edraft.online import OnlineBigram, OnlineState, _TrainingSparse
from edraft.runtime import WorkQueue, parse_config
from edraft.sampler import SamplingWarper, sample_token, sample_token_tensor
from edraft.sparse import LazySignedHash, SignedHash, SumSparse, hash_keys, proximal_l1
from edraft.toy import ToyParallelDrafter, ToyTarget


def _load_online_validator():
    validator_path = Path(__file__).resolve().parents[1] / "artifacts" / "m4_evidence_validator" / "online_checks.py"
    spec = importlib.util.spec_from_file_location("m4_online_checks", validator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _reference_online_update_step(state: OnlineState, round_idx: int, l_unit: torch.Tensor,
                                  target_logits: torch.Tensor, H: torch.Tensor,
                                  prefix: list[int], tokens: list[int], first_rejection: int | None = None):
    """Verbatim pre-optimization 09a5469 2-pass per-slot Python loop oracle."""
    assert round_idx > state.last_round
    state.last_round = round_idx
    if round_idx % state.update_stride:
        return False
    weights = loss_weights(len(l_unit), l_unit.device, state.weighting_mode, state.gamma, first_rejection)
    lr = float(state.lr(state.u + 1) if callable(state.lr) else state.lr)
    l_unit, target_logits, H = (x.detach().float().clone() for x in (l_unit, target_logits, H))
    if weights is not None:
        weights = weights.detach().clone()
    dW = state.dW.detach().clone().requires_grad_(not state.bias_sparse_only)
    dA = state.dA.detach().clone().requires_grad_(not state.bias_sparse_only)
    db = state.db.detach().clone().requires_grad_(True)
    W, A, b = state.offline.W + dW, state.offline.A + dA, state.offline.b + db
    E = state.offline.E
    U = block_unary(E, W, b, H)
    sparse = state.offline.sparse
    dirty = sparse_values = None
    if state.dS is not None:
        accesses = []
        for i in range(len(H)):
            dense = slot_correction(E, A, U[i], l_unit[i], prefix, tokens, i)
            candidates = top_candidates(l_unit[i] - dense, state.offline.M_sparse)
            for d in range(1, len(A) + 1):
                previous = prev_token(prefix, tokens, i, d)
                if previous is not None:
                    accesses.append(hash_keys(d, previous, candidates, state.dS.capacity, E.device)[0])
        dirty = torch.unique(torch.cat(accesses), sorted=True) if accesses else torch.empty(0, device=E.device, dtype=torch.long)
        sparse_values = state.dS.read(dirty).detach().clone().requires_grad_(True)
        sparse = _TrainingSparse(state.offline.sparse, dirty, sparse_values, state.dS.capacity)
    correction = torch.stack([slot_correction(E, A, U[i], l_unit[i], prefix, tokens, i,
                                              sparse, state.offline.M_sparse) for i in range(len(H))])
    loss = full_kl(l_unit - correction, target_logits, weights)
    loss.backward()
    with torch.no_grad():
        shrink = 1 + lr * state.lambda_l2
        if not state.bias_sparse_only:
            state.dW = (dW - lr * (dW.grad if dW.grad is not None else torch.zeros_like(dW))) / shrink
            state.dA = (dA - lr * (dA.grad if dA.grad is not None else torch.zeros_like(dA))) / shrink
        state.db = proximal_l1((db - lr * db.grad) / shrink, lr * state.lambda_l1_b)
        if state.dS is not None:
            gradient = sparse_values.grad if sparse_values.grad is not None else torch.zeros_like(sparse_values)
            state.dS.step(dirty, gradient, lr, state.lambda_l1_s)
        state.u += 1
        state.version += 1
        state.last_loss = float(loss.detach())
        state._snapshot = state._make_snapshot()
        dense_bytes = sum(x.numel() * x.element_size() for x in (state.dW, state.dA, state.db))
        optimized_bytes = state.db.numel() * state.db.element_size() if state.bias_sparse_only else dense_bytes
        state.state_bytes_estimate["dense_activation_clone"] += 2 * dense_bytes
        state.state_bytes_estimate["dense_optimizer"] += 5 * optimized_bytes + 2 * state.db.numel() * state.db.element_size()
        state.state_bytes_estimate["dense_snapshot"] += 3 * dense_bytes
        if state.dS is not None:
            active = len(state.dS.active)
            snapshot_bytes = active * 16 + (state.dS.u + 1) * 4
            state.state_bytes_estimate["sparse_update"] += state.dS.last_update_bytes
            state.state_bytes_estimate["sparse_snapshot"] += 2 * snapshot_bytes
            state.active_bucket_min = active if state.active_bucket_min is None else min(state.active_bucket_min, active)
            state.active_bucket_max = max(state.active_bucket_max, active)
    return True


@pytest.mark.parametrize("offline_has_sparse", [False, True])
@pytest.mark.parametrize("bias_sparse_only", [False, True])
@pytest.mark.parametrize("stride,weighting_mode", [(1, "exponential"), (5, "rejection_relative"), (1, "cutoff")])
def test_vectorized_online_update_matches_two_pass_loop_oracle(
    offline_has_sparse: bool, bias_sparse_only: bool, stride: int, weighting_mode: str
):
    gen = torch.Generator().manual_seed(1234)
    V, r, p, D, K = 29, 6, 8, 4, 5
    E = torch.randn(V, r, generator=gen) / math.sqrt(r)
    offline_sparse = SignedHash(97) if offline_has_sparse else None
    with torch.no_grad():
        if offline_sparse is not None:
            offline_sparse.values.copy_(torch.randn(97, generator=gen) * 0.02)
    adapter = DenseAdapter(E, p=p, D=D, sparse=offline_sparse, M_sparse=7)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(r, p, generator=gen) * 0.02)
        adapter.A.copy_(torch.randn(D, r, r, generator=gen) * 0.02)
        adapter.b.copy_(torch.randn(V, generator=gen) * 0.01)

    lr_fn = lambda u: 0.03 / math.sqrt(u)
    kwargs = dict(
        lr=lr_fn,
        update_stride=stride,
        lambda_l2=0.01,
        lambda_l1_b=0.002,
        lambda_l1_s=0.003,
        sparse_capacity=101,
        enable_sparse=True,
        bias_sparse_only=bias_sparse_only,
        weighting_mode=weighting_mode,
        gamma=4.0,
    )
    fast_state = OnlineState(adapter, **kwargs)
    ref_state = OnlineState(adapter, **kwargs)

    prefixes = [[], [2], [1, 4, 2, 0, 7]]
    for round_idx in range(1, 11):
        k_cur = K if round_idx < 10 else 3  # Exercise variable terminal K as well
        first_rej = min(2, k_cur)
        l_unit = torch.randn(k_cur, V, generator=gen).log_softmax(-1)
        target_logits = torch.randn(k_cur, V, generator=gen)
        H = torch.randn(k_cur, p, generator=gen)
        prefix = prefixes[(round_idx - 1) % len(prefixes)]
        tokens = [int(x) for x in torch.randint(V, (k_cur,), generator=gen).tolist()]

        c_fast = fast_state.update(round_idx, l_unit, target_logits, H, prefix, tokens, first_rejection=first_rej)
        c_ref = _reference_online_update_step(ref_state, round_idx, l_unit, target_logits, H, prefix, tokens, first_rejection=first_rej)
        assert c_fast == c_ref
        assert fast_state.u == ref_state.u
        assert fast_state.version == ref_state.version
        torch.testing.assert_close(fast_state.dW, ref_state.dW, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fast_state.dA, ref_state.dA, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fast_state.db, ref_state.db, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            fast_state.dS.read(torch.arange(101)),
            ref_state.dS.read(torch.arange(101)),
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.equal(fast_state.dS.active, ref_state.dS.active)
        assert fast_state.state_bytes_estimate == ref_state.state_bytes_estimate
        assert fast_state.active_bucket_min == ref_state.active_bucket_min
        assert fast_state.active_bucket_max == ref_state.active_bucket_max
        if fast_state.last_loss is None:
            assert ref_state.last_loss is None
        else:
            assert fast_state.last_loss == pytest.approx(ref_state.last_loss, rel=1e-5, abs=1e-6)


def test_online_rows_pass_m4_evidence_validator_eager_and_simulated_graph():
    validator = _load_online_validator()
    eager_args = SimpleNamespace(scale="smoke", seed=53, device="cpu", stage="m4", smoke_tokens=16)
    eager_runner = EvaluationRunner(eager_args)

    graph_args = SimpleNamespace(scale="smoke", seed=53, device="cpu", stage="m4", smoke_tokens=16)
    graph_runner = EvaluationRunner(graph_args)
    graph_runner.fastpath = FastPathRuntime(FastPathConfig(simulate_cpu_graph=True), static_block_length=4)

    prompt = smoke_manifest()["prompts"][-1]
    name, T, warp = sampling_conditions("long_thinking")[0]
    for arm in ("offline_online_stride1", "online_dbS", "offline_online_stride5", "zero_offline_online_only"):
        setting = candidate_settings(arm)[0]
        job = dict(prompt=prompt, arm=arm, sampling_name=name, temperature=T, warper=warp, seed=53)
        eager_row = eager_runner.run(job, setting, profile=True)
        graph_row = graph_runner.run(job, setting, profile=True)

        for row in (eager_row, graph_row):
            decoded = json.loads(json.dumps(row))
            checked = validator.check_online_row(
                decoded, expected_scale="smoke", expected_arm=arm, expected_setting=setting, profile=True
            )
            assert checked["failures"] == [], f"{arm} failed validator: {checked['failures']}"

        assert eager_row["output_tokens"] == graph_row["output_tokens"]
        assert eager_row["online"]["u"] == graph_row["online"]["u"]
        assert eager_row["online"]["db_norm"] == pytest.approx(graph_row["online"]["db_norm"], abs=1e-6)
        assert eager_row["online"]["dS_norm"] == pytest.approx(graph_row["online"]["dS_norm"], abs=1e-6)


@pytest.mark.parametrize("k_ret", [None, 3])
def test_online_bigram_cached_score_matches_dense_oracle_and_preserves_bypass(k_ret):
    V = 19
    alpha = 0.4
    lambda_tr = 0.25
    bigram = OnlineBigram(V, alpha=alpha, lambda_tr=lambda_tr, k_ret=k_ret)
    gen = torch.Generator().manual_seed(91)
    l_unit = torch.randn(V, generator=gen).log_softmax(-1)

    assert bigram.score(l_unit, None) is l_unit
    assert bigram.score(l_unit, 5) is l_unit  # N_total == 0 bypass

    stream = [5, 2, 5, 2, 5, 7, 5, 1, 5, 9, 5, 7, 5, 11, 5, 3, 2, 4]
    for idx in range(1, len(stream)):
        bigram.update([stream[idx - 1]], [stream[idx]])
        for prev in (5, 2, 17):
            counts = bigram.counts.get(prev)
            if not counts:
                assert bigram.score(l_unit, prev) is l_unit
                continue
            # Dense uncached oracle from prompt.md formula
            total = sum(counts.values())
            retained = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
            if k_ret is not None:
                retained = retained[:k_ret]
            ret_dict = dict(retained)
            evicted = total - sum(ret_dict.values())
            absent = V - len(ret_dict)
            expected_probs = torch.tensor(
                [((ret_dict[v] if v in ret_dict else evicted / absent) + alpha) / (total + alpha * V) for v in range(V)],
                dtype=torch.float32,
            )
            assert abs(float(expected_probs.sum()) - 1.0) <= 1e-6
            prior = bigram.prior(prev)
            assert abs(float(prior.exp().sum()) - 1.0) <= 1e-6
            torch.testing.assert_close(prior, expected_probs.log(), atol=1e-6, rtol=1e-6)
            # Call score twice to hit both cache miss and cache hit paths
            s1 = bigram.score(l_unit, prev)
            s2 = bigram.score(l_unit, prev)
            torch.testing.assert_close(s1, l_unit + lambda_tr * expected_probs.log(), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(s2, s1, atol=0, rtol=0)

    bigram.lambda_tr = 0.0
    assert bigram.score(l_unit, 5) is l_unit


@pytest.mark.parametrize("proposal_mode", ["dense_full", "dense_candidate"])
@pytest.mark.parametrize("prefix", [[], [3], [3, 1, 4, 2, 6]])
def test_block_correction_and_score_slot_tensor_match_per_slot_reference(proposal_mode: str, prefix: list[int]):
    gen = torch.Generator().manual_seed(77)
    V, r, p, D, K, M = 23, 5, 7, 4, 6, 8
    E = torch.randn(V, r, generator=gen) / math.sqrt(r)
    offline_sparse = SignedHash(89)
    with torch.no_grad():
        offline_sparse.values.copy_(torch.randn(89, generator=gen) * 0.03)
    lazy = LazySignedHash(53)
    lazy.step(torch.tensor([3, 11, 27]), torch.tensor([0.2, -0.4, 0.1]), lr=0.5, lambda_l1=0.01)
    combined_sparse = SumSparse((offline_sparse, lazy.snapshot()))

    adapter = DenseAdapter(E, p=p, D=D, sparse=offline_sparse, M_sparse=M)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(r, p, generator=gen) * 0.03)
        adapter.A.copy_(torch.randn(D, r, r, generator=gen) * 0.03)
        adapter.b.copy_(torch.randn(V, generator=gen) * 0.02)
    l_unit = torch.randn(K, V, generator=gen).log_softmax(-1)
    H = torch.randn(K, p, generator=gen)
    tokens = [int(x) for x in torch.randint(V, (K,), generator=gen).tolist()]

    U = block_unary(E, adapter.W, adapter.b, H)
    batched_corr = block_correction(E, adapter.A, U, l_unit, prefix, tokens, sparse=combined_sparse, M_sparse=M)
    expected_slots = [
        slot_correction(E, adapter.A, U[i], l_unit[i], prefix, tokens, i, sparse=combined_sparse, M_sparse=M)
        for i in range(K)
    ]
    torch.testing.assert_close(batched_corr, torch.stack(expected_slots), atol=1e-6, rtol=1e-6)

    snap = adapter.snapshot()
    if proposal_mode == "dense_candidate":
        C0 = top_candidates(l_unit - snap.unary(H), M)
        snap = snap.with_candidates(C0)
        unary_cand = snap.unary(H, proposal_mode="dense_candidate")
        with pytest.raises(ValueError, match="dense_full"):
            snap.score_slot_tensor(
                l_unit[0], unary_cand[0], torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long), proposal_mode="dense_candidate",
            )
        return

    for temperature, warper in [(0.0, SamplingWarper()), (0.7, SamplingWarper(top_k=5, top_p=0.8, min_p=0.05))]:
        # Force an exact tie on slot 0 in greedy mode (after top-M sparse lookup!) to verify lowest-ID tie-break
        l_test = l_unit.clone()
        if temperature == 0.0:
            u0 = snap.unary(H)[0]
            l_test[0, 2] += 10.0
            l_test[0, 9] += 10.0
            s_boosted = snap.score_slot(l_test[0], u0, prefix, [], 0, proposal_mode=proposal_mode)
            l_test[0, 9] += s_boosted[2] - s_boosted[9]
            s_tied = snap.score_slot(l_test[0], u0, prefix, [], 0, proposal_mode=proposal_mode)
            assert float(s_tied[2]) == float(s_tied[9])

        unary = snap.unary(H, proposal_mode=proposal_mode)
        rng_ref = torch.Generator().manual_seed(314)
        ref_tokens, ref_probs = [], []
        for i in range(K):
            sc = snap.score_slot(l_test[i], unary[i], prefix, ref_tokens, i, proposal_mode=proposal_mode)
            tok, prb = sample_token(sc, temperature, warper, generator=rng_ref)
            ref_tokens.append(tok)
            ref_probs.append(prb)
        if temperature == 0.0:
            assert ref_tokens[0] == 2  # Lowest ID wins exact tie between 2 and 9

        rng_fast = torch.Generator().manual_seed(314)
        ctx_buf = torch.empty(len(prefix) + K, dtype=torch.long)
        if prefix:
            ctx_buf[:len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        fast_probs = []
        for i in range(K):
            avail = min(D, len(prefix) + i)
            offs = torch.arange(1, avail + 1, dtype=torch.long)
            prev_ids = ctx_buf[len(prefix) + i - offs]
            sc_fast = snap.score_slot_tensor(
                l_test[i], unary[i], offs, prev_ids, proposal_mode=proposal_mode, contiguous_offsets=True,
            )
            sc_default = snap.score_slot_tensor(
                l_test[i], unary[i], offs, prev_ids, proposal_mode=proposal_mode, contiguous_offsets=False,
            )
            torch.testing.assert_close(sc_fast, sc_default, atol=1e-6, rtol=1e-6)
            tok_t, prb_f = sample_token_tensor(sc_fast, temperature, warper, generator=rng_fast)
            ctx_buf[len(prefix) + i] = tok_t
            fast_probs.append(prb_f)
        assert ctx_buf[len(prefix):].tolist() == ref_tokens
        for pf, pr in zip(fast_probs, ref_probs):
            if pr is None:
                assert pf is None
            else:
                torch.testing.assert_close(pf, pr, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cfg", [
    FastPathConfig(simulate_cpu_graph=True),
    FastPathConfig(compile=True),
    FastPathConfig(compile=True, simulate_cpu_graph=True),
])
def test_fastpath_static_graph_replay_fallback_cloning_and_rng_safety(cfg: FastPathConfig):
    gen = torch.Generator().manual_seed(2026)
    V, r, p, K = 17, 4, 6, 4
    E = torch.randn(V, r, generator=gen)
    W = torch.randn(r, p, generator=gen)
    b = torch.randn(V, generator=gen)
    snap_W_orig, snap_b_orig = W.clone(), b.clone()

    fp = FastPathRuntime(cfg, static_block_length=K)
    for _ in range(3):
        H = torch.randn(K, p, generator=gen)
        out = fp.unary(E, W, b, H)
        torch.testing.assert_close(out, block_unary(E, W, b, H), atol=1e-5, rtol=1e-5)
        # Mutate returned buffer aggressively to ensure static replay buffer is never aliased
        out.fill_(9999.0)
    # Terminal cap-trimmed block with K_eff = 2 < K = 4 must fall back to eager cleanly
    H_short = torch.randn(2, p, generator=gen)
    out_short = fp.unary(E, W, b, H_short)
    torch.testing.assert_close(out_short, block_unary(E, W, b, H_short), atol=1e-5, rtol=1e-5)
    assert torch.equal(W, snap_W_orig) and torch.equal(b, snap_b_orig)

    if cfg.simulate_cpu_graph:
        stats = fp.stats()
        assert stats["unary_graph_captures"] == 1
        assert stats["unary_graph_replays"] == 2
        assert stats["unary_eager_fallbacks"] == 1

    # RNG safety: repeated draws on identical scores MUST produce varying stochastic samples matching eager
    scores = torch.randn(V, generator=gen)
    warper = SamplingWarper(top_k=6, top_p=0.85)
    rng_eager = torch.Generator().manual_seed(999)
    rng_fast = torch.Generator().manual_seed(999)
    eager_draws = [sample_token(scores, 0.8, warper, generator=rng_eager)[0] for _ in range(24)]
    fast_draws = [int(sample_token_tensor(scores, 0.8, warper, generator=rng_fast, probs_fn=fp.probabilities)[0]) for _ in range(24)]
    assert fast_draws == eager_draws
    assert len(set(fast_draws)) > 1  # Never stuck replaying a frozen sample!

    # End-to-end generate_toy with both multi-round static K=4 (max_new=10) and cap-trimmed K=3<4 (max_new=3)
    target = ToyTarget(vocab_size=V, hidden_size=p, seed=11)
    drafter = ToyParallelDrafter(target, seed=12)
    adapter = DenseAdapter(target.embedding[:, :r], p=p, D=2, sparse=SignedHash(64), M_sparse=4)
    with torch.no_grad():
        adapter.W.normal_(0, 0.02)
        adapter.A.normal_(0, 0.02)
        adapter.b.normal_(0, 0.01)
    for max_new in (10, 3):
        for T in (0.0, 0.7):
            res_eager = generate_toy(
                target, drafter, [1, 2, 3], max_new, block_length=4, temperature=T,
                warper=warper if T > 0 else SamplingWarper(), adapter=adapter,
                generator=torch.Generator().manual_seed(42),
            )
            fp_loop = FastPathRuntime(cfg, static_block_length=4)
            res_fast = generate_toy(
                target, drafter, [1, 2, 3], max_new, block_length=4, temperature=T,
                warper=warper if T > 0 else SamplingWarper(), adapter=adapter,
                generator=torch.Generator().manual_seed(42), fastpath=fp_loop,
            )
            assert res_fast.output_tokens == res_eager.output_tokens
            if cfg.simulate_cpu_graph:
                if max_new == 3:
                    assert fp_loop.stats()["unary_eager_fallbacks"] >= 1
                else:
                    assert fp_loop.stats()["unary_graph_captures"] == 1
                    assert fp_loop.stats()["unary_graph_replays"] >= 1


def test_work_queue_reclaims_dead_pid_and_dynamic_sharding_matches_modulo(tmp_path: Path):
    q_path = tmp_path / "queue.json"
    q = WorkQueue(q_path, ["long-0", "long-1", "std-0"], precompleted={"std-0"})
    first = q.claim(rank=0, pid=999_999_999)  # Non-existent PID
    assert first == "long-0"
    # Dead PID claim must be reclaimed immediately by next live worker
    reclaimed = q.claim(rank=1)
    assert reclaimed == "long-0"
    second = q.claim(rank=0)
    assert second == "long-1"
    assert q.claim(rank=1) is None
    q.complete("long-0")
    q.complete("long-1")
    q2 = WorkQueue(q_path, ["long-0", "long-1", "std-0"])
    assert q2.claim(rank=0) is None

    # Compare dynamic vs modulo 2-rank evaluation in eval_closed_loop
    p_manifest = smoke_manifest()
    for sched in ("modulo", "dynamic"):
        out = tmp_path / f"eval_{sched}"
        base = dict(
            output=out, stage="m3", scale="smoke", seed=19, device="cpu",
            max_new_tokens=16, long_max_new_tokens=20, smoke_tokens=16,
            standard_dev_prompts=2, long_dev_prompts=2, fresh_session=False, schedule=sched,
            compile=False, cuda_graph=False, target="t", drafter="d",
            dense_checkpoint=None, sparse_checkpoint=None, projection=None, prompts_manifest=None,
            feature_diagnostic=None,
        )
        settings = run_tune(EvaluationRunner(SimpleNamespace(**base, mode="tune", rank=0, world_size=1)), p_manifest)
        for rank in (0, 1):
            run_evaluate(EvaluationRunner(SimpleNamespace(**base, mode="evaluate", rank=rank, world_size=2)), p_manifest, settings)
        eval_aggregate(SimpleNamespace(**base, mode="summarize", rank=0, world_size=1), p_manifest, settings)
    mod_summary = json.loads((tmp_path / "eval_modulo" / "m3" / "summary.json").read_text())
    dyn_summary = json.loads((tmp_path / "eval_dynamic" / "m3" / "summary.json").read_text())
    assert mod_summary["missing_requests"] == [] and dyn_summary["missing_requests"] == []
    assert mod_summary["completed_requests"] == dyn_summary["completed_requests"]
    wall_fields = {"request_latency_p50_seconds", "request_latency_p95_seconds", "phase_ms_median"}
    strip_wall = lambda groups: [{k: v for k, v in g.items() if k not in wall_fields} for g in groups]
    assert strip_wall(mod_summary["groups"]) == strip_wall(dyn_summary["groups"])
    # Verify WorkQueue put long_thinking keys ahead of standard_nonthinking keys and completed all prompts
    wq_files = list((tmp_path / "eval_dynamic" / "m3" / "evaluate").glob("work_queue-*.json"))
    assert len(wq_files) == 1
    wq_state = json.loads(wq_files[0].read_text())
    assert len(wq_state["completed"]) == len(wq_state["ordered_keys"])
    assert all("long_thinking" in k for k in wq_state["ordered_keys"][:2])

    # Compare dynamic vs modulo 2-rank trace collection + audit in collect.py
    for sched in ("modulo", "dynamic"):
        c_out = tmp_path / f"collect_{sched}"
        cal_args = prepare_collect_args(parse_config(collect_parser(), [
            "--scale", "smoke", "--stage", "calibrate", "--output", str(c_out), "--seed", "23",
        ]))
        collect_calibration(cal_args)
        for rank in (0, 1):
            col_args = prepare_collect_args(parse_config(collect_parser(), [
                "--scale", "smoke", "--stage", "collect", "--output", str(c_out), "--seed", "23",
                "--rank", str(rank), "--world_size", "2", "--schedule", sched,
            ]))
            collect_traces(col_args)
        aud_args = prepare_collect_args(parse_config(collect_parser(), [
            "--scale", "smoke", "--stage", "audit", "--output", str(c_out), "--seed", "23",
        ]))
        report = collect_audit(aud_args)
        assert report["status"] == "passed", f"{sched} audit failed: {report['requirements']}"

    mod_audit = json.loads((tmp_path / "collect_modulo" / "m1_audit.json").read_text())
    dyn_audit = json.loads((tmp_path / "collect_dynamic" / "m1_audit.json").read_text())
    assert mod_audit["prompt_counts"] == dyn_audit["prompt_counts"]


def test_longest_periodic_run_early_exit_matches_reference():
    import numpy as np

    def unoptimized_oracle(tokens):
        values = np.asarray(tokens, dtype=np.int64)
        best_length = 0
        best = {"start": None, "end": None, "period": None, "complete_cycles": 0}
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
            best = {"start": start, "end": start + span, "period": period, "complete_cycles": span // period}
        return best_length, best

    cases = [
        [],
        [4],
        [1, 2] * 8,
        [7] * 16,
        [9, 3, 4, 3, 4, 3, 4, 3, 4, 8, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3] * 4 + [9] + [5] * 12,
    ]
    for seq in cases:
        assert _longest_periodic_run(seq) == unoptimized_oracle(seq)


def test_bench_adapter_includes_online_and_zero_sync_modes():
    from edraft.bench_adapter import benchmark

    args = SimpleNamespace(
        device="cpu", seed=13, scale="smoke", vocab_size=31, block_length=4,
        rank=8, projection_dim=8, history_depth=4, candidates=8, hash_capacity=257,
        warmups=1, repeats=2, compile=True, cuda_graph=False,
    )
    with torch.no_grad():
        report = benchmark(args)
    assert "online_update_step" in report["phases"]
    assert "online_bigram_score" in report["phases"]
    assert "online_update_step_fastpath" in report["phases"]
    assert "batched_full_zero_sync" in report["modes"]
    assert "batched_full_fastpath" in report["modes"]


def test_score_slot_tensor_non_contiguous_reordered_offsets_and_validation():
    gen = torch.Generator().manual_seed(404)
    V, r, p, D, M = 31, 6, 8, 4, 9
    E = torch.randn(V, r, generator=gen) / math.sqrt(r)
    offline_sparse = SignedHash(73)
    with torch.no_grad():
        offline_sparse.values.copy_(torch.randn(73, generator=gen) * 0.25)
    lazy = LazySignedHash(67)
    lazy.step(torch.tensor([1, 7, 19, 41]), torch.tensor([0.35, -0.5, 0.2, -0.3]), lr=0.4, lambda_l1=0.01)
    combined_sparse = SumSparse((offline_sparse, lazy.snapshot()))

    adapter = DenseAdapter(E, p=p, D=D, sparse=offline_sparse, M_sparse=M)
    with torch.no_grad():
        adapter.W.copy_(torch.randn(r, p, generator=gen) * 0.2)
        adapter.A.copy_(torch.randn(D, r, r, generator=gen) * 0.35)
        adapter.b.copy_(torch.randn(V, generator=gen) * 0.1)
    base_snap = adapter.snapshot()
    snap = type(base_snap)(
        base_snap.E, base_snap.W, base_snap.A, base_snap.b, base_snap.version,
        combined_sparse, base_snap.M_sparse, None, False,
    )
    l_unit_i = torch.randn(V, generator=gen).log_softmax(-1)
    U_i = snap.unary(torch.randn(1, p, generator=gen))[0]

    predecessor_cases = [
        [],
        [(3, 14)],
        [(2, 7), (4, 19)],
        [(2, 11), (1, 5)],
        [(4, 3), (2, 22), (1, 9)],
    ]
    for beta in (1.0, 0.65, 0.0):
        for preds in predecessor_cases:
            offs_1d = torch.tensor([d for d, _ in preds], dtype=torch.long)
            prev_1d = torch.tensor([tok for _, tok in preds], dtype=torch.long)
            if beta == 0.0:
                expected = l_unit_i
            else:
                if preds:
                    msg = _pair_message(E, snap.A, preds)
                    dense = U_i + E.to(snap.A.dtype) @ msg
                    cands = top_candidates(l_unit_i - beta * dense, M)
                    s_vals = combined_sparse.lookup(offs_1d[:, None], prev_1d[:, None], cands[None, :]).sum(0)
                    corr = dense.scatter_add(0, cands, s_vals.to(dense.dtype))
                else:
                    corr = U_i
                expected = l_unit_i - beta * corr

            got_1d = snap.score_slot_tensor(l_unit_i, U_i, offs_1d, prev_1d, beta=beta)
            got_2d = snap.score_slot_tensor(l_unit_i, U_i, offs_1d[:, None], prev_1d, beta=beta)
            torch.testing.assert_close(got_1d, expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(got_2d, expected, atol=1e-6, rtol=1e-6)

        # Validation checks must trigger even when beta == 0.0
        with pytest.raises(ValueError, match="matching 1D"):
            snap.score_slot_tensor(l_unit_i, U_i, torch.tensor([1, 2]), torch.tensor([5]), beta=beta)
        with pytest.raises(ValueError, match="matching 1D"):
            snap.score_slot_tensor(l_unit_i, U_i, torch.tensor([[1, 2], [3, 4]]), torch.tensor([5, 6, 7, 8]), beta=beta)
        with pytest.raises(ValueError, match="matching 1D"):
            snap.score_slot_tensor(l_unit_i, U_i, torch.tensor(1), torch.tensor([5]), beta=beta)
        with pytest.raises(ValueError, match="matching 1D"):
            snap.score_slot_tensor(l_unit_i, U_i, torch.tensor([1, 2]), torch.tensor([[5], [6]]), beta=beta)
        with pytest.raises(ValueError, match="exceeds adapter history depth"):
            snap.score_slot_tensor(
                l_unit_i, U_i, torch.arange(1, D + 2), torch.zeros(D + 1, dtype=torch.long),
                beta=beta, contiguous_offsets=True,
            )


def test_standalone_summarize_succeeds_after_compiled_timing(tmp_path: Path):
    p_manifest = smoke_manifest()
    out_compiled = tmp_path / "compiled_m3"
    base_kwargs = dict(
        output=out_compiled, stage="m3", scale="smoke", seed=29, device="cpu",
        max_new_tokens=12, long_max_new_tokens=16, smoke_tokens=12,
        standard_dev_prompts=2, long_dev_prompts=2, timing_prompts=0,
        timing_repeats=2, timing_warmups=1, fresh_session=False, schedule="modulo",
        target="t", drafter="d", dense_checkpoint=None, sparse_checkpoint=None,
        projection=None, prompts_manifest=None, feature_diagnostic=None,
    )
    settings = run_tune(
        EvaluationRunner(SimpleNamespace(**base_kwargs, mode="tune", rank=0, world_size=1, compile=True, cuda_graph=False)),
        p_manifest,
    )
    run_evaluate(
        EvaluationRunner(SimpleNamespace(**base_kwargs, mode="evaluate", rank=0, world_size=1, compile=True, cuda_graph=False)),
        p_manifest, settings,
    )
    timing_report = run_timing(
        EvaluationRunner(SimpleNamespace(**base_kwargs, mode="timing", rank=0, world_size=1, compile=True, cuda_graph=False)),
        p_manifest, settings,
    )
    assert timing_report["fastpath"] == {"compile": True, "cuda_graph": False, "compile_mode": "default"}

    # Standalone --mode summarize with default flags (compile=False, cuda_graph=False) must succeed!
    summary = eval_aggregate(
        SimpleNamespace(**base_kwargs, mode="summarize", rank=0, world_size=1, compile=False, cuda_graph=False),
        p_manifest, settings,
    )
    assert summary["status"] == "complete"
    assert summary["isolated_timing_complete"] is True
    assert summary["timing_evidence"]["status"] == "complete"

    # Also test CLI main(["--mode", "summarize", ...]) with zero fastpath flags
    eval_main([
        "--scale", "smoke", "--stage", "m3", "--mode", "summarize",
        "--output", str(out_compiled), "--seed", "29",
        "--max_new_tokens", "12", "--long_max_new_tokens", "16", "--smoke_tokens", "12",
    ])
    cli_summary = json.loads((out_compiled / "m3" / "summary.json").read_text())
    assert cli_summary["status"] == "complete"
    assert cli_summary["isolated_timing_complete"] is True

    # Passing a contradicting explicit CLI flag (--cuda_graph) to summarize must fail cleanly
    bad_summary = eval_aggregate(
        SimpleNamespace(**base_kwargs, mode="summarize", rank=0, world_size=1, compile=False, cuda_graph=True),
        p_manifest, settings,
    )
    assert bad_summary["status"] == "incomplete"
    assert "configuration mismatch" in bad_summary["timing_evidence"]["reason"]

    # Auditing with a live eager EvaluationRunner(compile=False) must also reject the compiled timing report
    eager_runner = EvaluationRunner(SimpleNamespace(**base_kwargs, mode="timing", rank=0, world_size=1, compile=False, cuda_graph=False))
    live_mismatch = audit_timing_evidence(eager_runner, p_manifest, settings, timing_report)
    assert live_mismatch["status"] == "incomplete"
    assert "configuration mismatch" in live_mismatch["reason"]

    # Baseline timing (compile=False, cuda_graph=False) omits 'fastpath' key completely
    out_eager = tmp_path / "eager_m3"
    eager_kwargs = dict(base_kwargs, output=out_eager)
    eager_settings = run_tune(
        EvaluationRunner(SimpleNamespace(**eager_kwargs, mode="tune", rank=0, world_size=1, compile=False, cuda_graph=False)),
        p_manifest,
    )
    eager_timing = run_timing(
        EvaluationRunner(SimpleNamespace(**eager_kwargs, mode="timing", rank=0, world_size=1, compile=False, cuda_graph=False)),
        p_manifest, eager_settings,
    )
    assert "fastpath" not in eager_timing


def test_fastpath_compile_zero_dynamo_recompile_limit_across_full_pipeline(tmp_path: Path):
    recompile_messages: list[str] = []

    class _RecompileWatcher(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "recompile_limit" in msg:
                recompile_messages.append(msg)

    watcher = _RecompileWatcher()
    dynamo_logger = logging.getLogger("torch._dynamo")
    dynamo_logger.addHandler(watcher)
    try:
        torch._dynamo.reset()
        gen = torch.Generator().manual_seed(5150)
        V, r, p, D, K = 29, 6, 8, 3, 4
        E = torch.randn(V, r, generator=gen) / math.sqrt(r)
        adapter = DenseAdapter(E, p=p, D=D, sparse=SignedHash(64), M_sparse=6)
        with torch.no_grad():
            adapter.W.normal_(0, 0.02)
            adapter.A.normal_(0, 0.02)
            adapter.b.normal_(0, 0.01)

        warpers = [
            SamplingWarper(),
            SamplingWarper(top_k=12, top_p=0.95),
            SamplingWarper(top_k=8, top_p=0.8, min_p=0.05),
            SamplingWarper(top_k=0, top_p=1.0, min_p=0.02),
        ]
        for cfg in (FastPathConfig(compile=True), FastPathConfig(compile=True, simulate_cpu_graph=True)):
            fp = FastPathRuntime(cfg, static_block_length=K)
            # Mix grad_mode True and False, 1D + static 2D (K, K+1) + cap-trimmed 2D (1, 2, 3), 4 temps, 4 warpers
            for use_no_grad in (False, True):
                ctx = torch.no_grad() if use_no_grad else torch.enable_grad()
                with ctx:
                    for shape in ((V,), (K, V), (K + 1, V), (1, V), (2, V), (3, V)):
                        x = torch.randn(*shape, generator=gen)
                        for T in (0.6, 0.7, 1.0, 0.85):
                            for w in warpers:
                                got = fp.probabilities(x, T, w)
                                assert got.shape == shape
                                assert abs(float(got.sum(-1).mean()) - 1.0) <= 1e-5
                    # Static K=4 and cap-trimmed K=2 unary (both inference no-grad and requires_grad=True)
                    H_static = torch.randn(K, p, generator=gen)
                    H_trim = torch.randn(2, p, generator=gen)
                    W_det, b_det = adapter.W.detach(), adapter.b.detach()
                    torch.testing.assert_close(fp.unary(E, W_det, b_det, H_static), block_unary(E, W_det, b_det, H_static), atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(fp.unary(E, W_det, b_det, H_trim), block_unary(E, W_det, b_det, H_trim), atol=1e-5, rtol=1e-5)
                    if not use_no_grad:
                        u_grad = fp.unary(E, adapter.W, adapter.b, H_static)
                        assert u_grad.requires_grad
                        torch.testing.assert_close(u_grad, block_unary(E, adapter.W, adapter.b, H_static), atol=1e-5, rtol=1e-5)
            # OnlineState update with static K=4 and cap-trimmed K=2
            online = OnlineState(adapter, lr=0.01, lambda_l2=1e-3, lambda_l1_b=1e-4, lambda_l1_s=1e-4, sparse_capacity=64, enable_sparse=True, fastpath=fp)
            for r_idx, k_cur in ((1, K), (2, 2), (3, K), (4, 1)):
                online.update(
                    r_idx,
                    torch.randn(k_cur, V, generator=gen).log_softmax(-1),
                    torch.randn(k_cur, V, generator=gen),
                    torch.randn(k_cur, p, generator=gen),
                    [1, 2],
                    [int(t) for t in torch.randint(V, (k_cur,), generator=gen).tolist()],
                )

        # Also exercise full tune + evaluate + timing with smoke_tokens=15 (forces cap-trimmed terminal blocks < K=4)
        p_manifest = smoke_manifest()
        out = tmp_path / "m3_compile_no_bailout"
        base = dict(
            output=out, stage="m3", scale="smoke", seed=61, device="cpu",
            max_new_tokens=15, long_max_new_tokens=15, smoke_tokens=15,
            standard_dev_prompts=1, long_dev_prompts=1, timing_prompts=1,
            timing_repeats=2, timing_warmups=1, fresh_session=False, schedule="modulo",
            compile=True, cuda_graph=False, target="t", drafter="d",
            dense_checkpoint=None, sparse_checkpoint=None, projection=None,
            prompts_manifest=None, feature_diagnostic=None,
        )
        settings = run_tune(EvaluationRunner(SimpleNamespace(**base, mode="tune", rank=0, world_size=1)), p_manifest)
        run_evaluate(EvaluationRunner(SimpleNamespace(**base, mode="evaluate", rank=0, world_size=1)), p_manifest, settings)
        run_timing(EvaluationRunner(SimpleNamespace(**base, mode="timing", rank=0, world_size=1)), p_manifest, settings)

        assert recompile_messages == [], f"Unexpected Dynamo recompile_limit bailouts: {recompile_messages}"
    finally:
        dynamo_logger.removeHandler(watcher)


