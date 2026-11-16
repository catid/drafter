"""Comprehensive Round-5 optimization parity, zero-sync, sharded M5 tune, and parallel shard loading tests."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from edraft import eval_closed_loop, launch, post_m4_suite, train
from edraft.adapter import (
    DenseAdapter,
    LookaheadConfig,
    damped_lookahead_step,
    initialize_lookahead,
    lookahead_energy,
    lookahead_fields,
)
from edraft.dflash_loop import PhaseTimer, _sample_proposal, generate_toy
from edraft.features import fit_hidden_projection, fit_token_features
from edraft.online import OnlineState
from edraft.runtime import Manifest, atomic_json, fingerprint
from edraft.sampler import SamplingWarper, probabilities
from edraft.sparse import ExactObservedPairs, LazySignedHash, SignedHash, SumSparse
from edraft.toy import ToyParallelDrafter, ToyTarget


# ---------------------------------------------------------------------------
# 1. Opt 1: Batched Precomputed Candidate-Masked Lookahead
# ---------------------------------------------------------------------------


def _make_adapter_with_sparse(
    gen: torch.Generator,
    *,
    V: int = 64,
    r: int = 8,
    p: int = 8,
    D: int = 4,
    M: int = 8,
    sparse_kind: str = "SignedHash",
):
    E = torch.randn(V, r, generator=gen) * 0.15
    if sparse_kind == "none":
        sp = None
    elif sparse_kind == "SignedHash":
        sp = SignedHash(128)
        with torch.no_grad():
            sp.values.copy_(torch.randn(128, generator=gen) * 0.08)
    elif sparse_kind == "ExactObservedPairs":
        sp = ExactObservedPairs(capacity=256, vocab_size=V)
        sp.train()
        d_seed = torch.randint(1, D + 1, (48,), generator=gen)
        prev_seed = torch.randint(0, V, (48,), generator=gen)
        curr_seed = torch.randint(0, V, (48,), generator=gen)
        _ = sp.lookup(d_seed, prev_seed, curr_seed)
        with torch.no_grad():
            n_occ = sp.num_occupied
            sp.values.data[:n_occ] = torch.randn(n_occ, generator=gen) * 0.08
        sp.eval()
    elif sparse_kind == "SumSparse":
        off = SignedHash(128)
        with torch.no_grad():
            off.values.copy_(torch.randn(128, generator=gen) * 0.06)
        base = DenseAdapter(E, p=p, D=D, sparse=off, M_sparse=M)
        with torch.no_grad():
            base.W.copy_(torch.randn(r, p, generator=gen) * 0.1)
            base.A.copy_(torch.randn(D, r, r, generator=gen) * 0.1)
            base.b.copy_(torch.randn(V, generator=gen) * 0.05)
        st = OnlineState(base, lr=0.05, sparse_capacity=128, enable_sparse=True)
        st.update(
            1,
            torch.randn(4, V, generator=gen).log_softmax(-1),
            torch.randn(4, V, generator=gen),
            torch.randn(4, p, generator=gen),
            [1, 2, 3],
            [4, 5, 6, 7],
        )
        return base, st.snapshot()
    else:
        raise ValueError(f"Unknown sparse_kind: {sparse_kind}")

    adap = DenseAdapter(E, p=p, D=D, sparse=sp, M_sparse=M)
    with torch.no_grad():
        adap.W.copy_(torch.randn(r, p, generator=gen) * 0.1)
        adap.A.copy_(torch.randn(D, r, r, generator=gen) * 0.1)
        adap.b.copy_(torch.randn(V, generator=gen) * 0.05)
    return adap, adap.snapshot()


@pytest.mark.parametrize("sparse_kind", ["none", "SignedHash", "ExactObservedPairs", "SumSparse"])
@pytest.mark.parametrize("mode", ["static", "rerelax"])
@pytest.mark.parametrize("K,D,M,pref_len", [(1, 2, 4, 0), (4, 4, 8, 2), (8, 4, 12, 5)])
def test_round5_lookahead_belief_field_score_parity_all_sparse_backends(
    sparse_kind: str, mode: str, K: int, D: int, M: int, pref_len: int
):
    gen = torch.Generator().manual_seed(4100 + K * 17 + pref_len * 3)
    V, r, p = 48, 6, 8
    _, snap = _make_adapter_with_sparse(gen, V=V, r=r, p=p, D=D, M=M, sparse_kind=sparse_kind)
    l_unit = torch.randn(K, V, generator=gen).log_softmax(-1)
    H = torch.randn(K, p, generator=gen)
    prefix = [int(x) for x in torch.randint(0, V, (pref_len,), generator=gen).tolist()]

    # Count sparse.lookup calls on fast vs reference path!
    fast_calls = 0
    if snap.sparse is not None:
        sp_cls = type(snap.sparse)
        unbound_lookup = sp_cls.lookup

        def _counting_lookup(self_sp, *a, **kw):
            nonlocal fast_calls
            if self_sp is snap.sparse:
                fast_calls += 1
            return unbound_lookup(self_sp, *a, **kw)

        with mock.patch.object(sp_cls, "lookup", autospec=True, side_effect=_counting_lookup):
            st_fast = initialize_lookahead(
                snap, l_unit, H, prefix, domain="candidate_masked",
                T_relax=0.85, beta_relax=0.9, use_fast_candidate=True,
            )
            assert st_fast._fast is not None
            for _ in range(2):
                damped_lookahead_step(st_fast, eta=0.55)
            for i in range(K):
                cand_sc = st_fast.candidate_slot_scores(i, beta=0.75)
                chosen_tok = int(st_fast.snapshot.C0[i][int(torch.argmax(cand_sc).item())].item())
                st_fast.clamp(i, chosen_tok)
                if mode == "rerelax" and i + 1 < K:
                    damped_lookahead_step(st_fast, eta=0.55)
        # At most 1 prefix lookup + 1 intra-block [P, M, M] lookup across the ENTIRE block!
        expected_max_calls = (1 if pref_len > 0 else 0) + (1 if K > 1 else 0)
        assert fast_calls == expected_max_calls
    else:
        st_fast = initialize_lookahead(
            snap, l_unit, H, prefix, domain="candidate_masked",
            T_relax=0.85, beta_relax=0.9, use_fast_candidate=True,
        )
        assert st_fast._fast is not None
        for _ in range(2):
            damped_lookahead_step(st_fast, eta=0.55)
        for i in range(K):
            cand_sc = st_fast.candidate_slot_scores(i, beta=0.75)
            chosen_tok = int(st_fast.snapshot.C0[i][int(torch.argmax(cand_sc).item())].item())
            st_fast.clamp(i, chosen_tok)
            if mode == "rerelax" and i + 1 < K:
                damped_lookahead_step(st_fast, eta=0.55)

    # Exact step-by-step parity check (unclamped + sequentially clamped) against use_fast_candidate=False:
    st_a = initialize_lookahead(
        snap, l_unit, H, prefix, domain="candidate_masked",
        T_relax=0.85, beta_relax=0.9, use_fast_candidate=True,
    )
    st_b = initialize_lookahead(
        snap, l_unit, H, prefix, domain="candidate_masked",
        T_relax=0.85, beta_relax=0.9, use_fast_candidate=False,
    )
    for _ in range(2):
        damped_lookahead_step(st_a, eta=0.55)
        damped_lookahead_step(st_b, eta=0.55)

    for i in range(K):
        torch.testing.assert_close(
            st_a.candidate_slot_scores(i, beta=0.75),
            st_b.candidate_slot_scores(i, beta=0.75),
            atol=1e-6, rtol=1e-6,
        )
        choice_tok = int(st_b.snapshot.C0[i][i % M].item())
        st_a.clamp(i, choice_tok)
        st_b.clamp(i, choice_tok)
        if mode == "rerelax" and i + 1 < K:
            damped_lookahead_step(st_a, eta=0.55)
            damped_lookahead_step(st_b, eta=0.55)

    # Full [K, V] g (including outside C0!), r, mu, fields, and energy must all match < 1e-6!
    torch.testing.assert_close(st_a.g, st_b.g, atol=1e-6, rtol=1e-6)
    for r_a, r_b in zip(st_a.r, st_b.r):
        torch.testing.assert_close(r_a, r_b, atol=1e-6, rtol=1e-6)
    for m_a, m_b in zip(st_a.mu, st_b.mu):
        torch.testing.assert_close(m_a, m_b, atol=1e-6, rtol=1e-6)
    f_a, b_a = lookahead_fields(st_a)
    f_b, b_b = lookahead_fields(st_b)
    torch.testing.assert_close(f_a, f_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b_a, b_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(lookahead_energy(st_a), lookahead_energy(st_b), atol=1e-6, rtol=1e-6)


def test_round5_lookahead_mutation_and_outside_c0_clamp_safety():
    gen = torch.Generator().manual_seed(9001)
    V, r, p, D, K, M = 32, 6, 8, 3, 4, 5
    _, snap = _make_adapter_with_sparse(gen, V=V, r=r, p=p, D=D, M=M, sparse_kind="SignedHash")
    l_unit = torch.randn(K, V, generator=gen).log_softmax(-1)
    H = torch.randn(K, p, generator=gen)
    prefix = [2, 5, 9]

    # 1. Clamping a token OUTSIDE C0[0] cleanly abandons _fast and matches reference!
    st_fast = initialize_lookahead(snap, l_unit, H, prefix, use_fast_candidate=True)
    st_ref = initialize_lookahead(snap, l_unit, H, prefix, use_fast_candidate=False)
    damped_lookahead_step(st_fast, eta=0.5)
    damped_lookahead_step(st_ref, eta=0.5)
    c0_set = set(st_fast.snapshot.C0[0].tolist())
    outside_tok = next(t for t in range(V) if t not in c0_set)
    st_fast.clamp(0, outside_tok)
    st_ref.clamp(0, outside_tok)
    assert st_fast._fast is None
    damped_lookahead_step(st_fast, eta=0.5)
    damped_lookahead_step(st_ref, eta=0.5)
    torch.testing.assert_close(st_fast.g, st_ref.g, atol=1e-6, rtol=1e-6)
    for ra, rb in zip(st_fast.r, st_ref.r):
        torch.testing.assert_close(ra, rb, atol=1e-6, rtol=1e-6)

    # 2. In-place mutation of `state.r[0]` after reading `state.r` is detected by `_check_fast_integrity`!
    st_mut = initialize_lookahead(snap, l_unit, H, prefix, use_fast_candidate=True)
    st_mut_ref = initialize_lookahead(snap, l_unit, H, prefix, use_fast_candidate=False)
    r_list = st_mut.r
    r_list[0].fill_(1.0 / M)
    st_mut.mu[0].copy_(r_list[0] @ snap.E[st_mut.support[0]])
    st_mut_ref.r[0].fill_(1.0 / M)
    st_mut_ref.mu[0].copy_(st_mut_ref.r[0] @ snap.E[st_mut_ref.support[0]])
    damped_lookahead_step(st_mut, eta=0.4)
    damped_lookahead_step(st_mut_ref, eta=0.4)
    assert st_mut._fast is None
    torch.testing.assert_close(st_mut.g, st_mut_ref.g, atol=1e-6, rtol=1e-6)

    # 3. Ragged C0 and duplicate-ID C0 gracefully use general path with exact parity!
    ragged_snap = snap.with_candidates((torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9]), torch.tensor([0, 1])))
    st_rag = initialize_lookahead(ragged_snap, l_unit, H, prefix, use_fast_candidate=True)
    assert st_rag._fast is None
    dup_snap = snap.with_candidates(tuple(torch.tensor([1, 1, 2, 3]) for _ in range(K)))
    st_dup = initialize_lookahead(dup_snap, l_unit, H, prefix, use_fast_candidate=True)
    assert st_dup._fast is None


@pytest.mark.parametrize("mode", ["static", "rerelax"])
@pytest.mark.parametrize("top_k", [0, 2, 6])
def test_round5_lookahead_closed_loop_proposal_and_zero_host_sync(mode: str, top_k: int):
    gen = torch.Generator().manual_seed(5500 + top_k)
    V, r, p, D, K, M = 40, 6, 8, 4, 5, 8
    adap, snap = _make_adapter_with_sparse(gen, V=V, r=r, p=p, D=D, M=M, sparse_kind="SumSparse")
    draft_logits = torch.randn(K, V, generator=gen) * 1.3
    H = torch.randn(K, p, generator=gen)
    prefix = [3, 7, 11, 2]
    la_cfg = LookaheadConfig(mode=mode, domain="candidate_masked", R_init=2, R_clamp=1, eta=0.5, T_relax=0.9, beta_relax=0.8)
    timer = PhaseTimer("cpu", False)

    # 1. Greedy T=0 exact token-for-token parity + zero host syncs
    def _ban(name):
        def _err(*_a, **_kw):
            raise AssertionError(f"Host sync {name} triggered inside Round-5 candidate lookahead!")
        return _err

    with (
        mock.patch.object(torch.Tensor, "item", _ban("item")),
        mock.patch.object(torch.Tensor, "tolist", _ban("tolist")),
        mock.patch.object(torch.Tensor, "__bool__", _ban("__bool__")),
        mock.patch.object(torch.Tensor, "__int__", _ban("__int__")),
        mock.patch.object(torch.Tensor, "__float__", _ban("__float__")),
        mock.patch.object(torch, "nonzero", _ban("nonzero")),
        mock.patch.object(torch, "unique", _ban("unique")),
    ):
        tok_g_fast, q_g_fast, _, _ = _sample_proposal(
            draft_logits, H, prefix, temperature=0.0, warper=SamplingWarper(),
            generator=torch.Generator().manual_seed(12), snapshot=snap, projection=None,
            beta=0.8, bigram=None, timer=timer, need_training_tensors=False,
            proposal_mode="dense_candidate", lookahead=la_cfg,
            _return_token_tensor=True, _allow_compact_q=True, enable_round3_compact=True,
        )
    assert q_g_fast is None
    tok_g_ref, _, _, _ = _sample_proposal(
        draft_logits, H, prefix, temperature=0.0, warper=SamplingWarper(),
        generator=torch.Generator().manual_seed(12), snapshot=snap, projection=None,
        beta=0.8, bigram=None, timer=timer, need_training_tensors=False,
        proposal_mode="dense_candidate", lookahead=la_cfg,
        _return_token_tensor=False, _allow_compact_q=False, enable_round3_compact=False,
    )
    assert tok_g_fast.tolist() == tok_g_ref

    # 2. Stochastic T=0.7 zero host syncs + conditional distribution parity at realized trajectory
    warper = SamplingWarper(top_k=top_k, top_p=0.85, min_p=0.05)
    rng_fast = torch.Generator().manual_seed(99)
    with (
        mock.patch.object(torch.Tensor, "item", _ban("item")),
        mock.patch.object(torch.Tensor, "tolist", _ban("tolist")),
        mock.patch.object(torch.Tensor, "__bool__", _ban("__bool__")),
        mock.patch.object(torch.Tensor, "__int__", _ban("__int__")),
        mock.patch.object(torch.Tensor, "__float__", _ban("__float__")),
        mock.patch.object(torch, "nonzero", _ban("nonzero")),
        mock.patch.object(torch, "unique", _ban("unique")),
    ):
        tok_s_fast, q_s_fast, l_u, _ = _sample_proposal(
            draft_logits, H, prefix, temperature=0.7, warper=warper,
            generator=rng_fast, snapshot=snap, projection=None,
            beta=0.8, bigram=None, timer=timer, need_training_tensors=True,
            proposal_mode="dense_candidate", lookahead=la_cfg,
            _return_token_tensor=True, _allow_compact_q=True, enable_round3_compact=True,
        )

    q_dense = q_s_fast.to_dense()
    realized_toks = tok_s_fast.tolist()
    st_oracle = initialize_lookahead(
        snap, l_u, H, prefix, domain="candidate_masked",
        T_relax=la_cfg.T_relax, beta_relax=la_cfg.beta_relax, use_fast_candidate=False,
    )
    for _ in range(la_cfg.R_init):
        damped_lookahead_step(st_oracle, eta=la_cfg.eta)
    for i in range(K):
        ids_i = st_oracle.snapshot.C0[i]
        sc_i = st_oracle.l_unit[i] - 0.8 * st_oracle.g[i]
        masked_i = torch.full_like(sc_i, -torch.inf).scatter(0, ids_i, sc_i[ids_i])
        q_oracle_i = probabilities(masked_i, 0.7, warper)
        torch.testing.assert_close(q_dense[i], q_oracle_i, atol=1e-6, rtol=1e-6)
        st_oracle.clamp(i, realized_toks[i])
        if mode == "rerelax" and i + 1 < K:
            damped_lookahead_step(st_oracle, eta=la_cfg.eta)


# ---------------------------------------------------------------------------
# 2. Opt 2: Core-Pinned Snake-Sharded `--stage m5 --mode tune` + `tune_summarize`
# ---------------------------------------------------------------------------


def test_round5_m3_m4_tune_still_rejects_world_size_gt_1(tmp_path):
    for stage in ("m3", "m4"):
        with pytest.raises(SystemExit):
            eval_closed_loop.main([
                "--scale", "smoke", "--device", "cpu",
                "--stage", stage, "--mode", "tune",
                "--rank", "0", "--world_size", "2",
                "--output", str(tmp_path / f"bad_{stage}"),
            ])
        with pytest.raises(SystemExit):
            eval_closed_loop.main([
                "--scale", "smoke", "--device", "cpu",
                "--stage", stage, "--mode", "tune_summarize",
                "--output", str(tmp_path / f"bad_sum_{stage}"),
            ])


def test_round5_m5_sharded_tune_prior_token_weights_and_exact_selection(tmp_path):
    # 1. Verify compute_tune_sharding_plan uses realized dev prompt tokens from m4/tune/manifest-rank0.json!
    prompts = [
        {"prompt_id": "dev_std_0", "split": "dev", "condition": "standard_nonthinking"},
        {"prompt_id": "dev_std_1", "split": "dev", "condition": "standard_nonthinking"},
        {"prompt_id": "dev_std_2", "split": "dev", "condition": "standard_nonthinking"},
        {"prompt_id": "dev_std_3", "split": "dev", "condition": "standard_nonthinking"},
        {"prompt_id": "dev_long_0", "split": "dev", "condition": "long_thinking", "reference_selected": True},
        {"prompt_id": "dev_long_1", "split": "dev", "condition": "long_thinking", "reference_selected": True},
    ]
    manifest = {"prompts": prompts}
    m4_tune_dir = tmp_path / "ws_weights" / "m4" / "tune" / "rank0"
    m4_tune_dir.mkdir(parents=True)
    tok_map = {
        "dev_std_0": 10, "dev_std_1": 90, "dev_std_2": 60, "dev_std_3": 30,
        "dev_long_0": 200, "dev_long_1": 800,
    }
    jobs_dict = {}
    for idx, p in enumerate(prompts):
        rpath = m4_tune_dir / f"r{idx}.json"
        atomic_json(rpath, {"prompt_id": p["prompt_id"], "generated_tokens": tok_map[p["prompt_id"]]})
        jobs_dict[f"k{idx}"] = {"status": "complete", "result_path": str(rpath)}
    atomic_json(tmp_path / "ws_weights" / "m4" / "tune" / "manifest-rank0.json", {"jobs": jobs_dict})

    ns = SimpleNamespace(
        output=tmp_path / "ws_weights", stage="m5", scale="smoke", world_size=2, seed=2026,
    )
    cands = {"plain_dflash": [{}], "dense_candidate": [{"beta": 0.5, "beta_relax": 1.0}, {"beta": 1.0, "beta_relax": 1.0}]}
    plan = eval_closed_loop.compute_tune_sharding_plan(ns, manifest, prompts, cands)
    assert plan["weight_source"] == "m4_tune_realized_tokens"
    # Descending std order: dev_std_1 (90), dev_std_2 (60), dev_std_3 (30), dev_std_0 (10)
    # Snake W=2: rank 0 -> [dev_std_1, dev_std_0], rank 1 -> [dev_std_2, dev_std_3]
    assert set(plan["by_condition"]["standard_nonthinking"]["ranks"]["0"]) == {"dev_std_1", "dev_std_0"}
    assert set(plan["by_condition"]["standard_nonthinking"]["ranks"]["1"]) == {"dev_std_2", "dev_std_3"}
    assert plan["by_condition"]["long_thinking"]["ranks"]["0"] == ["dev_long_1"]
    assert plan["by_condition"]["long_thinking"]["ranks"]["1"] == ["dev_long_0"]

    # 2. End-to-end smoke M5 sharded tune (W=2) -> tune_summarize -> evaluate -> timing -> summarize!
    ws = tmp_path / "m5_sharded_ws"
    m5_arms = ["plain_dflash", "dense_candidate", "lookahead_static_candidate", "lookahead_rerelax_candidate"]
    sess_id = "m5tune2gpusess"
    for rk in (0, 1):
        eval_closed_loop.main([
            "--scale", "smoke", "--device", "cpu",
            "--stage", "m5", "--mode", "tune",
            "--rank", str(rk), "--world_size", "2",
            "--logical_session", sess_id,
            "--m5_arms", *m5_arms,
            "--output", str(ws),
        ])

    # Verify incomplete tune_summarize fails cleanly if a bundle file is missing, then resumes!
    sess_r1_path = ws / "m5" / "tune" / "sessions" / f"{sess_id}-rank1.json"
    sess_r1 = json.loads(sess_r1_path.read_text())
    first_bkey = next(iter(sess_r1["bundles"]))
    victim_file = Path(sess_r1["bundles"][first_bkey]["result_paths"][-1])
    victim_file.unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        eval_closed_loop.main([
            "--scale", "smoke", "--device", "cpu",
            "--stage", "m5", "--mode", "tune_summarize",
            "--m5_arms", *m5_arms,
            "--output", str(ws),
        ])

    # Re-running rank 1 resumes completed bundles and re-executes only the interrupted bundle!
    eval_closed_loop.main([
        "--scale", "smoke", "--device", "cpu",
        "--stage", "m5", "--mode", "tune",
        "--rank", "1", "--world_size", "2",
        "--logical_session", sess_id,
        "--m5_arms", *m5_arms,
        "--output", str(ws),
    ])
    eval_closed_loop.main([
        "--scale", "smoke", "--device", "cpu",
        "--stage", "m5", "--mode", "tune_summarize",
        "--m5_arms", *m5_arms,
        "--output", str(ws),
    ])
    settings = json.loads((ws / "m5" / "settings.json").read_text())
    assert settings["frozen"] is True
    assert settings["world_size"] == 2
    assert settings["sharding_mode"] == "sharded_2gpu"
    assert set(settings["selected"].keys()) == set(m5_arms)

    # Verify downstream evaluate + sharded timing + summarize succeed against sharded settings.json!
    for rk in (0, 1):
        eval_closed_loop.main([
            "--scale", "smoke", "--device", "cpu",
            "--stage", "m5", "--mode", "evaluate",
            "--schedule", "dynamic",
            "--rank", str(rk), "--world_size", "2",
            "--m5_arms", *m5_arms,
            "--output", str(ws),
        ])
    for rk in (0, 1):
        eval_closed_loop.main([
            "--scale", "smoke", "--device", "cpu",
            "--stage", "m5", "--mode", "timing",
            "--rank", str(rk), "--world_size", "2",
            "--logical_session", "m5timingsess",
            "--timing_repeats", "2",
            "--m5_arms", *m5_arms,
            "--output", str(ws),
        ])
    eval_closed_loop.main([
        "--scale", "smoke", "--device", "cpu",
        "--stage", "m5", "--mode", "summarize",
        "--timing_repeats", "2",
        "--m5_arms", *m5_arms,
        "--output", str(ws),
    ])
    summary = json.loads((ws / "m5" / "summary.json").read_text())
    assert summary["status"] == "complete"

    # 3. Also test post_m4_suite._execute_m5_stage_pipeline with 2 workers (`gpus=[0, 1]`) end-to-end!
    pipe_ws = tmp_path / "pipe_2gpu_ws"
    pipe_args = SimpleNamespace(
        scale="smoke", seed=2026, device="cpu", resume=True,
        timing_repeats=2, timing_warmups=1, timing_prompts=0,
        max_new_tokens=32, long_max_new_tokens=32, smoke_tokens=16,
    )
    p_sum, p_tim, p_set = post_m4_suite._execute_m5_stage_pipeline(
        pipe_args, pipe_ws, ["plain_dflash", "lookahead_rerelax_candidate"], [0, 1],
    )
    assert p_sum["status"] == "complete"
    assert p_tim["world_size"] == 2
    assert p_set["world_size"] == 2
    assert p_set["sharding_mode"] == "sharded_2gpu"


# ---------------------------------------------------------------------------
# 3. Opt 3: Parallel Shard Loading in train.py, launch.py, and post_m4_suite.py
# ---------------------------------------------------------------------------


def test_round5_parallel_shard_loading_and_transient_identity(tmp_path, monkeypatch):
    assert "num_workers" in train.TRANSIENT_CONFIG_KEYS
    assert "microbatch_blocks" in train.TRANSIENT_CONFIG_KEYS

    # 1. Verify num_workers=0 vs num_workers=2 preserves exact fit identity AND loss/parameter parity
    #    on BOTH `union_rest_grouped` AND `two_sided_pseudolikelihood`!
    for objective in ("union_rest_grouped", "two_sided_pseudolikelihood"):
        same_ws = tmp_path / f"fit_{objective}_same"
        rep0 = train.main([
            "--scale", "smoke", "--output", str(same_ws),
            "--objective", objective, "--batch_blocks", "4", "--num_workers", "0",
        ])
        rep2 = train.main([
            "--scale", "smoke", "--output", str(same_ws), "--no_resume",
            "--objective", objective, "--batch_blocks", "4", "--num_workers", "2", "--microbatch_blocks", "2",
        ])
        assert rep0["identity"] == rep2["identity"]
        for stg in ("dense", "dense_sparse"):
            assert math.isclose(
                rep0["stages"][stg]["test"]["weighted_mean_kl"],
                rep2["stages"][stg]["test"]["weighted_mean_kl"],
                rel_tol=1e-5, abs_tol=1e-6,
            )

    # 2. Verify post_m4_suite._run_training_fit passes `--num_workers 2` on CUDA runs!
    captured = []

    def _fake_launch(cli):
        captured.append(list(cli))
        out_p = Path(cli[cli.index("--output") + 1])
        tag = cli[cli.index("--m5_fit_tag") + 1]
        (out_p / f"{tag}.json").write_text(json.dumps({"status": "passed"}))

    monkeypatch.setattr(launch, "main", _fake_launch)
    ws_cuda = tmp_path / "cuda_fit_ws"
    ws_cuda.mkdir()
    args_cuda = SimpleNamespace(
        scale="reference", seed=2026, device="cuda:0",
        target="Qwen/Qwen3-8B", drafter="z-lab/Qwen3-8B-DFlash-b16",
        resume=True, allow_large_run=True, stats_memory_limit="32GB",
        train_num_workers=None, microbatch_blocks=4,
    )
    res = post_m4_suite._run_training_fit(args_cuda, ws_cuda, [0, 1, 2, 3], "r5_test_fit", ["--bf16_gemm"])
    assert res["status"] == "passed"
    assert len(captured) == 1
    cli = captured[0]
    assert cli[cli.index("--num_workers") + 1] == "2"
    assert cli[cli.index("--microbatch_blocks") + 1] == "4"


# ---------------------------------------------------------------------------
# 4. Round-5 Lookahead Microbenchmark Wiring
# ---------------------------------------------------------------------------


def test_round5_fastpath_bench_includes_lookahead_microbenchmark():
    args = SimpleNamespace(
        scale="smoke", seed=2026, bench_warmups=1, bench_repeats=2, bigram_count_gate_kappa=12.0,
    )
    r5 = post_m4_suite._measure_round5_lookahead_microbench(args, torch.device("cpu"))
    assert r5["dimensions"]["R_init"] == 2
    assert r5["dimensions"]["R_clamp"] == 1
    assert r5["speedups"]["rerelax_iterations_round5_over_pre_round5"] > 0.0
    assert r5["speedups"]["rerelax_proposal_round5_over_pre_round5"] > 0.0

    # Also measure realistic high-vocab [K=15, V=32768, r=64, M=256] speedup on CPU!
    gen = torch.Generator().manual_seed(777)
    V, K, r, p, D, M = 32768, 15, 64, 64, 4, 256
    E = torch.randn(V, r, generator=gen) * 0.05
    sp = SignedHash(2**16)
    with torch.no_grad():
        sp.values.normal_(0, 0.02)
    adap = DenseAdapter(E, p=p, D=D, sparse=sp, M_sparse=M)
    with torch.no_grad():
        adap.W.normal_(0, 0.05)
        adap.A.normal_(0, 0.05)
    snap = adap.snapshot()
    l_u = torch.randn(K, V, generator=gen).log_softmax(-1)
    H = torch.randn(K, p, generator=gen)
    U_all = snap.unary(H)

    import time
    t0 = time.perf_counter()
    st_pre = initialize_lookahead(snap, l_u, H, [1, 2, 3, 4], U_precomputed=U_all, use_fast_candidate=False)
    for _ in range(2):
        damped_lookahead_step(st_pre, eta=0.5)
    for idx in range(K - 1):
        st_pre.clamp(idx, int(st_pre.snapshot.C0[idx][0].item()))
        damped_lookahead_step(st_pre, eta=0.5)
    pre_ms = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    st_r5 = initialize_lookahead(snap, l_u, H, [1, 2, 3, 4], U_precomputed=U_all, use_fast_candidate=True)
    for _ in range(2):
        damped_lookahead_step(st_r5, eta=0.5)
    for idx in range(K - 1):
        st_r5._clamp_orig_pos(idx, st_r5._fast.c0_mat[idx, 0], torch.zeros((), dtype=torch.long))
        damped_lookahead_step(st_r5, eta=0.5)
    r5_ms = (time.perf_counter() - t1) * 1e3

    assert pre_ms > r5_ms * 2.0, f"Expected >2x speedup at V=32768, got {pre_ms:.2f}ms vs {r5_ms:.2f}ms"


# ---------------------------------------------------------------------------
# 5. Regression tests for R5-BUG-1, R5-BUG-2, and R5-BUG-3
# ---------------------------------------------------------------------------


def test_r5_bug_1_m5_resume_preserves_frozen_settings_hash_and_skips_completed_warmup(tmp_path):
    ws = tmp_path / "m5_resume_ws"
    arms = ["plain_dflash", "lookahead_rerelax_candidate"]
    args = SimpleNamespace(
        scale="smoke", seed=2026, device="cpu", resume=True,
        timing_repeats=2, timing_warmups=1, timing_prompts=0,
        max_new_tokens=32, long_max_new_tokens=32, smoke_tokens=16,
    )
    post_m4_suite._execute_m5_stage_pipeline(
        args, ws, arms, [0, 1], extra_m5_flags=["--m5_torch_compile"],
    )
    settings_before = json.loads((ws / "m5/settings.json").read_text())
    hash_before = fingerprint(settings_before)

    # Second call with `resume=True` must NOT mutate `settings.json` / `frozen_settings_hash` or crash `m5-timing`!
    spawned_labels: list[str] = []
    orig_spawn = post_m4_suite._spawn_parallel_workers

    def _spy_spawn(worker_specs, logdir, label, **kw):
        spawned_labels.append(label)
        return orig_spawn(worker_specs, logdir, label, **kw)

    with mock.patch.object(post_m4_suite, "_spawn_parallel_workers", side_effect=_spy_spawn):
        post_m4_suite._execute_m5_stage_pipeline(
            args, ws, arms, [0, 1], extra_m5_flags=["--m5_torch_compile"],
        )
    assert "m5-tune" not in spawned_labels
    assert "m5-tune-summarize" not in spawned_labels
    settings_after = json.loads((ws / "m5/settings.json").read_text())
    assert fingerprint(settings_after) == hash_before

    # Direct `eval_closed_loop.main --mode tune` on already-complete rank bundles must skip `warm_comparisons`
    # AND `tune_summarize` must keep `settings.json` bit-identical!
    warm_calls = 0
    orig_warm = eval_closed_loop.warm_comparisons

    def _count_warm(*a, **kw):
        nonlocal warm_calls
        warm_calls += 1
        return orig_warm(*a, **kw)

    sess = json.loads((ws / "m5/tune/sharding_plan.json").read_text())["logical_session"]
    with mock.patch.object(eval_closed_loop, "warm_comparisons", side_effect=_count_warm):
        eval_closed_loop.main([
            "--scale", "smoke", "--device", "cpu",
            "--stage", "m5", "--mode", "tune",
            "--rank", "0", "--world_size", "2",
            "--logical_session", sess,
            "--m5_arms", *arms, "--m5_torch_compile",
            "--max_new_tokens", "32", "--long_max_new_tokens", "32", "--smoke_tokens", "16",
            "--output", str(ws),
        ])
    assert warm_calls == 0
    eval_closed_loop.main([
        "--scale", "smoke", "--device", "cpu",
        "--stage", "m5", "--mode", "tune_summarize",
        "--m5_arms", *arms, "--m5_torch_compile",
        "--max_new_tokens", "32", "--long_max_new_tokens", "32", "--smoke_tokens", "16",
        "--output", str(ws),
    ])
    assert fingerprint(json.loads((ws / "m5/settings.json").read_text())) == hash_before


def test_r5_bug_2_support_and_clamped_list_mutations_invalidate_fast_cache():
    gen = torch.Generator().manual_seed(77)
    V, r, p, D, K, M = 32, 6, 8, 3, 4, 5
    _, snap = _make_adapter_with_sparse(gen, V=V, r=r, p=p, D=D, M=M, sparse_kind="SignedHash")
    l_u = torch.randn(K, V, generator=gen).log_softmax(-1)
    H = torch.randn(K, p, generator=gen)
    prefix = [2, 5, 9]
    new_sup = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    # 2a: list-element replacement `state.support[0] = new_sup`
    st_fast = initialize_lookahead(snap, l_u, H, prefix, use_fast_candidate=True)
    st_ref = initialize_lookahead(snap, l_u, H, prefix, use_fast_candidate=False)
    st_fast.support[0] = new_sup
    st_ref.support[0] = new_sup.clone()
    damped_lookahead_step(st_fast, eta=0.5)
    damped_lookahead_step(st_ref, eta=0.5)
    assert st_fast._fast is None
    torch.testing.assert_close(st_fast.g, st_ref.g, atol=1e-6, rtol=1e-6)

    # 2b: in-place tensor mutation `state.support[0].copy_(new_sup)`
    st_fast2 = initialize_lookahead(snap, l_u, H, prefix, use_fast_candidate=True)
    st_ref2 = initialize_lookahead(snap, l_u, H, prefix, use_fast_candidate=False)
    st_fast2.support[0].copy_(new_sup)
    st_ref2.support[0].copy_(new_sup)
    damped_lookahead_step(st_fast2, eta=0.5)
    damped_lookahead_step(st_ref2, eta=0.5)
    assert st_fast2._fast is None
    torch.testing.assert_close(st_fast2.g, st_ref2.g, atol=1e-6, rtol=1e-6)

    # 2c: direct list-element clamp `state.clamped[0] = tok0`
    st_fast3 = initialize_lookahead(snap, l_u, H, prefix, use_fast_candidate=True)
    st_ref3 = initialize_lookahead(snap, l_u, H, prefix, use_fast_candidate=False)
    tok0 = int(st_ref3.support[0][0].item())
    st_fast3.clamped[0] = tok0
    st_ref3.clamped[0] = tok0
    damped_lookahead_step(st_fast3, eta=0.5)
    damped_lookahead_step(st_ref3, eta=0.5)
    assert st_fast3._fast is None
    for ra, rb in zip(st_fast3.r, st_ref3.r):
        torch.testing.assert_close(ra, rb, atol=1e-6, rtol=1e-6)


def test_r5_bug_3_dataloader_spawn_early_break_clean_shutdown(tmp_path):
    from edraft.runtime import common_parser, parse_config

    ws = tmp_path / "early_break_ws"
    parser = common_parser("test")
    parser.add_argument("--history_depth", type=int, default=4)
    parser.add_argument("--M_sparse", type=int, default=4)
    args = parse_config(parser, ["--scale", "smoke", "--output", str(ws)])
    trace_dir, E, head, bias, head_meta = train.smoke_bundle(args)
    entries, _ = train.trace_index(trace_dir)
    refs = train.epoch_references(entries, "train")

    # Early break from both `iter_prepared_batches` and `batches` with `num_workers=2` must shut down cleanly!
    gen1 = train.iter_prepared_batches(
        refs, 2, head_metadata=head_meta, vocab_size=len(E), num_workers=2, loader_seed=11,
    )
    first_item = next(gen1)
    assert first_item.batch_index == 0
    gen1.close()

    gen2 = train.batches(refs, 2, head_metadata=head_meta, num_workers=2, loader_seed=11)
    b_idx, blk_list = next(gen2)
    assert b_idx == 0 and len(blk_list) > 0
    gen2.close()
