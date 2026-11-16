from __future__ import annotations

from pathlib import Path
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from dflash.model import DFlashDraftModel

from edraft.adapter import DenseAdapter, LookaheadConfig
from edraft.backend import (
    ReferenceBackend,
    crop_cache,
    extract_context_feature,
    make_cache,
    register_edraft_sdpa_clr,
)
from edraft.dflash_loop import generate_reference
from edraft.fastpath import FastPathConfig, FastPathRuntime
from edraft.features import fit_hidden_projection
from edraft.m5_baselines import AdaptiveBlockController
from edraft.online import OnlineBigram, OnlineState
from edraft.sampler import SamplingWarper
from edraft.sparse import SignedHash
from edraft.verify import CompactProposal, verify_block


def _make_mini_qwen3_dflash_pair(
    *,
    seed: int = 123,
    target_layers: int = 4,
    draft_layers: int = 2,
    target_layer_ids: list[int] | None = None,
    fast_decode: bool = False,
) -> ReferenceBackend:
    torch.manual_seed(seed)
    layer_ids = [0, 2, target_layers - 1] if target_layer_ids is None else list(target_layer_ids)
    t_cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=target_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        eos_token_id=62,
        attn_implementation="sdpa",
    )
    d_cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=draft_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
        attn_implementation="sdpa",
    )
    d_cfg.num_target_layers = target_layers
    d_cfg.dflash_config = {
        "block_size": 5,
        "mask_token_id": 63,
        "target_layer_ids": layer_ids,
        "input_embedding_scale": 1.0,
        "output_multiplier": 1.0,
    }
    target = Qwen3ForCausalLM(t_cfg).to(dtype=torch.float32).eval()
    drafter = DFlashDraftModel(d_cfg).to(dtype=torch.float32).eval()
    return ReferenceBackend(
        target=target,
        drafter=drafter,
        tokenizer=None,
        target_id="mini-qwen3",
        drafter_id="mini-dflash",
        target_revision="r0",
        drafter_revision="r0",
        fast_decode=fast_decode,
    )


def test_preallocated_cache_bshd_contiguity_dynamic_growth_and_crop():
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    # Start with small initial cap=64 and grow dynamically beyond 64 -> 150 tokens
    pre_cache = make_cache(cfg, max_cache_len=64, preallocate=True)
    dyn_cache = make_cache(cfg)

    torch.manual_seed(7)
    for step_len in (20, 35, 45, 50):
        k_new = torch.randn(1, 2, step_len, 8)
        v_new = torch.randn(1, 2, step_len, 8)
        for lid in range(3):
            kp, vp = pre_cache.update(k_new, v_new, lid)
            kd, vd = dyn_cache.update(k_new, v_new, lid)
            assert torch.equal(kp, kd)
            assert torch.equal(vp, vd)
            # BSHD physical storage ensures transpose(1, 2) is 100% contiguous for FlashAttention-2
            assert kp.stride(-1) == 1
            assert kp.transpose(1, 2).is_contiguous()
            assert vp.transpose(1, 2).is_contiguous()

    assert pre_cache.get_seq_length() == 150
    crop_cache(pre_cache, 93)
    crop_cache(dyn_cache, 93)
    assert pre_cache.get_seq_length() == 93
    for lid in range(3):
        assert torch.equal(pre_cache.layers[lid].keys, dyn_cache.layers[lid].keys)
        assert torch.equal(pre_cache.layers[lid].values, dyn_cache.layers[lid].values)

    # Zero-remove crop_cache and positive legacy crop
    crop_cache(pre_cache, 93)
    assert pre_cache.get_seq_length() == 93
    pre_cache.crop(40)
    dyn_cache.crop(40)
    assert pre_cache.get_seq_length() == 40
    for lid in range(3):
        assert torch.equal(pre_cache.layers[lid].keys, dyn_cache.layers[lid].keys)

    # Reset preserves underlying buffer capacity while zeroing sequence length
    cap_before = pre_cache.layers[0]._capacity
    pre_cache.reset()
    assert pre_cache.get_seq_length() == 0
    assert pre_cache.layers[0]._capacity == cap_before


def test_edraft_sdpa_clr_eliminates_repeat_kv_and_preserves_zero_ulp_parity(monkeypatch):
    import transformers.integrations.sdpa_attention as sdpa_mod

    b_ref = _make_mini_qwen3_dflash_pair(seed=41, fast_decode=False)
    b_fast = _make_mini_qwen3_dflash_pair(seed=41, fast_decode=True)

    # target.config.to_dict() must remain identical so target_config_sha256 never drifts
    assert b_ref.target.config.to_dict() == b_fast.target.config.to_dict()

    repeat_calls = {"ref": 0, "fast": 0}
    orig_repeat_kv = sdpa_mod.repeat_kv

    prompt = torch.tensor([[1, 5, 9, 12, 3, 7]], dtype=torch.long)
    spec_canvas = torch.tensor([[7, 14, 2, 19, 8]], dtype=torch.long)
    pos_prefill = torch.arange(6)[None]
    pos_spec = torch.arange(5, 10)[None]

    c_ref = make_cache(b_ref.target.config)
    monkeypatch.setattr(
        sdpa_mod,
        "repeat_kv",
        lambda x, n: (repeat_calls.__setitem__("ref", repeat_calls["ref"] + 1) or orig_repeat_kv(x, n)),
    )
    out_ref_pre = b_ref.target(prompt, position_ids=pos_prefill, past_key_values=c_ref, use_cache=True, logits_to_keep=1)
    crop_cache(c_ref, 5)
    out_ref_spec = b_ref.target(spec_canvas, position_ids=pos_spec, past_key_values=c_ref, use_cache=True)

    c_fast, _, pooled = b_fast.acquire_decode_caches(64)
    try:
        monkeypatch.setattr(
            sdpa_mod,
            "repeat_kv",
            lambda x, n: (repeat_calls.__setitem__("fast", repeat_calls["fast"] + 1) or orig_repeat_kv(x, n)),
        )
        out_fast_pre = b_fast.target(
            prompt, position_ids=pos_prefill, past_key_values=c_fast, use_cache=True, logits_to_keep=1
        )
        crop_cache(c_fast, 5)
        out_fast_spec = b_fast.target(spec_canvas, position_ids=pos_spec, past_key_values=c_fast, use_cache=True)
    finally:
        b_fast.release_decode_caches(pooled)

    # Stock SDPA invokes repeat_kv twice per layer (K and V) during q_len=5 speculative verify; edraft_sdpa_clr invokes 0!
    assert repeat_calls["ref"] == 2 * b_ref.target.config.num_hidden_layers
    assert repeat_calls["fast"] == 0
    assert torch.equal(out_ref_pre.logits, out_fast_pre.logits)
    assert torch.allclose(out_ref_spec.logits, out_fast_spec.logits, atol=1e-6, rtol=1e-6)

    # Fallback check when explicit 2D attention_mask with padding is supplied
    padded_ids = torch.tensor([[0, 0, 4, 8, 11], [2, 6, 9, 1, 3]], dtype=torch.long)
    pad_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long)
    r_pad = b_ref.target(padded_ids, attention_mask=pad_mask, use_cache=False).logits
    f_pad = b_fast.target(padded_ids, attention_mask=pad_mask, use_cache=False).logits
    assert torch.allclose(r_pad, f_pad, atol=1e-6, rtol=1e-6)


def test_target_feature_tap_matches_hidden_states_including_final_norm_layer():
    # Include -1 (embed), interior layer 1, and final layer 3 (post-norm tie_last_hidden_states=True)
    b_ref = _make_mini_qwen3_dflash_pair(seed=88, target_layers=4, target_layer_ids=[-1, 1, 3], fast_decode=False)
    b_fast = _make_mini_qwen3_dflash_pair(seed=88, target_layers=4, target_layer_ids=[-1, 1, 3], fast_decode=True)

    ids = torch.tensor([[3, 11, 22, 7, 19, 5]], dtype=torch.long)
    pos = torch.arange(6)[None]
    c_ref = make_cache(b_ref.target.config)
    c_fast, _, pooled = b_fast.acquire_decode_caches(32)
    try:
        out_ref = b_ref.target(ids, position_ids=pos, past_key_values=c_ref, use_cache=True, output_hidden_states=True)
        feat_ref_full = extract_context_feature(out_ref.hidden_states, b_ref.drafter.target_layer_ids)
        feat_ref_sub = extract_context_feature(out_ref.hidden_states, b_ref.drafter.target_layer_ids, max_tokens=3)

        with b_fast.capture_target_features() as tap:
            out_fast = b_fast.target(
                ids, position_ids=pos, past_key_values=c_fast, use_cache=True, output_hidden_states=False
            )
            assert out_fast.hidden_states is None
            feat_fast_sub = tap.collect(max_tokens=3)

        with b_fast.capture_target_features() as tap2:
            _ = b_fast.target(ids, position_ids=pos, past_key_values=make_cache(b_fast.target.config), use_cache=True)
            feat_fast_full = tap2.collect()

        assert torch.equal(feat_ref_full, feat_fast_full)
        assert torch.equal(feat_ref_sub, feat_fast_sub)
        assert feat_fast_sub.is_contiguous()
    finally:
        b_fast.release_decode_caches(pooled)


def test_generate_reference_and_target_only_fast_decode_exact_parity_all_modes(monkeypatch):
    b_ref = _make_mini_qwen3_dflash_pair(seed=105, fast_decode=False)
    b_fast = _make_mini_qwen3_dflash_pair(seed=105, fast_decode=True)

    # Check rollback helper works under both modes
    probe_pre = torch.tensor([[2, 4, 6, 8]], dtype=torch.long)
    probe_prop = torch.tensor([[10, 12, 14, 16]], dtype=torch.long)
    rb_ref = b_ref.check_cache_rollback(probe_pre, probe_prop, 2, 19)
    rb_fast = b_fast.check_cache_rollback(probe_pre, probe_prop, 2, 19)
    assert rb_ref["argmax_equal"] and rb_fast["argmax_equal"]

    # 1. target_only_generate parity (greedy + stochastic + max_new_tokens=0 + stop token)
    prompt = torch.tensor([[1, 9, 17, 25, 33]], dtype=torch.long)
    assert torch.equal(b_ref.target_only_generate(prompt, 0), b_fast.target_only_generate(prompt, 0))
    t_only_ref = b_ref.target_only_generate(prompt, 24, temperature=0.0)
    t_only_fast = b_fast.target_only_generate(prompt, 24, temperature=0.0)
    assert torch.equal(t_only_ref, t_only_fast)

    warper = SamplingWarper(top_k=8, top_p=0.85)
    t_st_ref = b_ref.target_only_generate(
        prompt, 20, temperature=0.7, warper=warper, generator=torch.Generator().manual_seed(55)
    )
    t_st_fast = b_fast.target_only_generate(
        prompt, 20, temperature=0.7, warper=warper, generator=torch.Generator().manual_seed(55)
    )
    assert torch.equal(t_st_ref, t_st_fast)

    # 2. generate_reference across plain_dflash, bigram, static dense_sparse, online_state, lookahead, adaptive
    torch.manual_seed(77)
    E = torch.randn(64, 8)
    proj = fit_hidden_projection([torch.randn(16, 32)], 32, 12, seed=77, prompt_ids=["p0"])
    sparse_ref = SignedHash(128)
    with torch.no_grad():
        sparse_ref.values.normal_(0.0, 0.05)
    adapter_ref = DenseAdapter(E.clone(), 12, 3, sparse=sparse_ref, M_sparse=8)
    with torch.no_grad():
        adapter_ref.W.normal_(0.0, 0.05)
        adapter_ref.A.normal_(0.0, 0.05)
        adapter_ref.b.normal_(0.0, 0.02)

    sparse_fast = SignedHash(128)
    with torch.no_grad():
        sparse_fast.values.copy_(sparse_ref.values)
    adapter_fast = DenseAdapter(E.clone(), 12, 3, sparse=sparse_fast, M_sparse=8)
    adapter_fast.load_state_dict(adapter_ref.state_dict())

    for temp in (0.0, 0.7):
        w = SamplingWarper() if temp == 0.0 else warper
        # Plain DFlash (adapter=None, online_state=None)
        r0 = generate_reference(
            b_ref, prompt, 26, temperature=temp, warper=w, generator=torch.Generator().manual_seed(901),
            measure_phases=True,
        )
        f0 = generate_reference(
            b_fast, prompt, 26, temperature=temp, warper=w, generator=torch.Generator().manual_seed(901),
            measure_phases=True,
        )
        assert torch.equal(r0.output_ids, f0.output_ids)
        assert len(r0.blocks) == len(f0.blocks)
        if temp == 0.0:
            assert torch.equal(f0.output_ids, b_fast.target_only_generate(prompt, 26, temperature=0.0))

        # OnlineState + OnlineBigram (multiple sequential requests testing pooled cache reset!)
        st_ref = OnlineState(adapter_ref, enable_sparse=True, sparse_capacity=128)
        st_fast = OnlineState(adapter_fast, enable_sparse=True, sparse_capacity=128)
        bg_ref = OnlineBigram(64, count_gate_kappa=4.0, k_ret=16)
        bg_fast = OnlineBigram(64, count_gate_kappa=4.0, k_ret=16)
        r_on = generate_reference(
            b_ref, prompt, 28, temperature=temp, warper=w, generator=torch.Generator().manual_seed(902),
            online_state=st_ref, projection=proj, bigram=bg_ref,
        )
        f_on = generate_reference(
            b_fast, prompt, 28, temperature=temp, warper=w, generator=torch.Generator().manual_seed(902),
            online_state=st_fast, projection=proj, bigram=bg_fast,
        )
        assert torch.equal(r_on.output_ids, f_on.output_ids)
        assert [b["accepted_length"] for b in r_on.blocks] == [b["accepted_length"] for b in f_on.blocks]

        # Lookahead rerelax + AdaptiveBlockController
        la_cfg = LookaheadConfig(mode="rerelax", domain="candidate_masked", R_init=2, R_clamp=1)
        ac_ref = AdaptiveBlockController()
        ac_fast = AdaptiveBlockController()
        r_la = generate_reference(
            b_ref, prompt, 22, temperature=temp, warper=w, generator=torch.Generator().manual_seed(903),
            adapter=adapter_ref, projection=proj, lookahead=la_cfg,
            adaptive_controller=ac_ref,
        )
        f_la = generate_reference(
            b_fast, prompt, 22, temperature=temp, warper=w, generator=torch.Generator().manual_seed(903),
            adapter=adapter_fast, projection=proj, lookahead=la_cfg,
            adaptive_controller=ac_fast,
        )
        assert torch.equal(r_la.output_ids, f_la.output_ids)

    # 3. Dynamic toggle via FastPathRuntime(FastPathConfig(fast_decode=True)) on a non-fast ReferenceBackend
    fp_rt = FastPathRuntime(FastPathConfig(fast_decode=True), static_block_length=4)
    assert fp_rt.enabled is True
    assert fp_rt.config.to_dict()["fast_decode"] is True
    assert b_ref.fast_decode is False
    via_fp = generate_reference(
        b_ref, prompt, 20, temperature=0.7, warper=warper, generator=torch.Generator().manual_seed(999),
        fastpath=fp_rt,
    )
    assert b_ref.fast_decode is False
    direct_fast = generate_reference(
        b_fast, prompt, 20, temperature=0.7, warper=warper, generator=torch.Generator().manual_seed(999),
    )
    assert torch.equal(via_fp.output_ids, direct_fast.output_ids)

    # 4. EDRAFT_FAST_DECODE=1 environment auto-enable
    monkeypatch.setenv("EDRAFT_FAST_DECODE", "1")
    b_env = _make_mini_qwen3_dflash_pair(seed=105, fast_decode=None)
    assert b_env.fast_decode is True


def test_verify_block_tokens_out_and_out_of_vocab_guards():
    torch.manual_seed(19)
    V = 32
    draft_t = torch.tensor([3, 7, 11, 2], dtype=torch.long)
    t_logits = torch.randn(5, V)
    out_tokens: list[int] = []

    res_g = verify_block(draft_t, None, t_logits, 0.0, _tokens_out=out_tokens)
    assert out_tokens == [3, 7, 11, 2]
    assert res_g.verification_rule == "greedy"

    slot_idx = torch.tensor([[3, 1], [7, 2], [11, 4], [2, 5]], dtype=torch.long)
    slot_p = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], dtype=torch.float32)
    comp_q = CompactProposal(slot_idx, slot_p, slot_p[:, 0], V)
    out_tokens_s: list[int] = []
    res_s = verify_block(
        draft_t, comp_q, t_logits, 0.7, SamplingWarper(top_k=4, top_p=0.9),
        generator=torch.Generator().manual_seed(12), _tokens_out=out_tokens_s,
    )
    assert out_tokens_s == [3, 7, 11, 2]
    assert res_s.verification_rule == "ratio"

    with pytest.raises(ValueError, match="draft token outside vocabulary"):
        verify_block(torch.tensor([3, 99, 1, 2], dtype=torch.long), None, t_logits, 0.0)
    with pytest.raises(ValueError, match="draft token outside vocabulary"):
        verify_block(torch.tensor([-1, 4, 1, 2], dtype=torch.long), comp_q, t_logits, 0.7)


def test_sdpa_clr_static_cache_rollback_probe_pool_grow_base_attn_and_tap_cleanup(monkeypatch):
    from transformers.cache_utils import StaticCache
    import edraft.backend as backend_mod

    b_ref = _make_mini_qwen3_dflash_pair(seed=211, fast_decode=False)
    b_fast = _make_mini_qwen3_dflash_pair(seed=211, fast_decode=True)

    # 1. SDPA-CLR-BUG-1: StaticCache right-padded prefill (q_offset == 0, 1 < q_len < kv_len)
    # Poison trailing KV slots [5..31] to prove q_offset == 0 never uses causal_lower_right(5, 32).
    sc_ref = StaticCache(config=b_ref.target.config, max_cache_len=32)
    sc_fast = StaticCache(config=b_fast.target.config, max_cache_len=32)
    prefill_ids = torch.tensor([[2, 7, 13, 19, 23]], dtype=torch.long)
    cache_pos = torch.arange(5, dtype=torch.long)
    out_sc_ref = b_ref.target(prefill_ids, past_key_values=sc_ref, cache_position=cache_pos, use_cache=True).logits
    out_sc_fast = b_fast.target(prefill_ids, past_key_values=sc_fast, cache_position=cache_pos, use_cache=True).logits
    assert torch.allclose(out_sc_ref, out_sc_fast, atol=1e-6, rtol=1e-6)

    # Re-run on StaticCache with non-zero garbage in trailing KV positions [5..31]
    for lid in range(len(sc_ref.layers)):
        torch.manual_seed(500 + lid)
        noise_k = torch.randn_like(sc_ref.layers[lid].keys[:, :, 5:, :])
        noise_v = torch.randn_like(sc_ref.layers[lid].values[:, :, 5:, :])
        sc_ref.layers[lid].keys[:, :, 5:, :].copy_(noise_k)
        sc_ref.layers[lid].values[:, :, 5:, :].copy_(noise_v)
        sc_fast.layers[lid].keys[:, :, 5:, :].copy_(noise_k)
        sc_fast.layers[lid].values[:, :, 5:, :].copy_(noise_v)
    out_poison_ref = b_ref.target(prefill_ids, past_key_values=sc_ref, cache_position=cache_pos, use_cache=True).logits
    out_poison_fast = b_fast.target(
        prefill_ids, past_key_values=sc_fast, cache_position=cache_pos, use_cache=True
    ).logits
    assert torch.equal(out_poison_ref, out_poison_fast)

    # 2. ROLLBACK-PROBE-BUG-2: check_cache_rollback must test PreallocatedDynamicLayer when fast_decode=True
    seen_preallocate: list[bool] = []
    orig_make_cache = backend_mod.make_cache

    def _spy_make_cache(config, max_cache_len=None, *, preallocate=False):
        seen_preallocate.append(bool(preallocate))
        return orig_make_cache(config, max_cache_len=max_cache_len, preallocate=preallocate)

    monkeypatch.setattr(backend_mod, "make_cache", _spy_make_cache)
    rb0 = b_ref.check_cache_rollback(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]]), 2, 9)
    rb1 = b_fast.check_cache_rollback(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]]), 2, 9)
    monkeypatch.setattr(backend_mod, "make_cache", orig_make_cache)
    assert seen_preallocate == [False, True]
    assert rb0["argmax_equal"] and rb1["argmax_equal"]

    # 3. Short-to-long pooled cache direct growth (0 mid-decode doublings) + 40960 pool cap
    c_t, c_d, pooled = b_fast.acquire_decode_caches(128)
    try:
        assert pooled is True
        _ = b_fast.target(prefill_ids, past_key_values=c_t, use_cache=True, logits_to_keep=1)
        assert c_t.layers[0]._capacity == 128
    finally:
        b_fast.release_decode_caches(pooled)

    c_t_long, _, pooled_long = b_fast.acquire_decode_caches(40960)
    try:
        assert pooled_long is True
        assert c_t_long is c_t
        assert c_t_long.get_seq_length() == 0
        assert c_t_long.layers[0]._capacity == 40960
        assert c_t_long.layers[0]._k_buf.shape[1] == 40960
    finally:
        b_fast.release_decode_caches(pooled_long)

    _, _, pooled_over = b_fast.acquire_decode_caches(40961)
    assert pooled_over is False

    # 4. _base_attn_impl preserved across enable_fast_decode(True -> False)
    b_eager = _make_mini_qwen3_dflash_pair(seed=211, fast_decode=False)
    backend_mod._set_model_attn_implementation(b_eager.target, "eager")
    b_eager_wrapped = ReferenceBackend(
        target=b_eager.target,
        drafter=b_eager.drafter,
        tokenizer=None,
        target_id="mini-qwen3",
        drafter_id="mini-dflash",
        fast_decode=False,
    )
    assert b_eager_wrapped._base_attn_impl == "eager"
    b_eager_wrapped.enable_fast_decode(True)
    assert b_eager_wrapped.target.config._attn_implementation == "edraft_sdpa_clr"
    b_eager_wrapped.enable_fast_decode(False)
    assert b_eager_wrapped.target.config._attn_implementation == "eager"

    # 5. _feature_tap.clear() runs in generate_reference finally block on both normal exit and exception
    res_ok = generate_reference(b_fast, prefill_ids, 12, temperature=0.0)
    assert res_ok.output_ids.shape[1] > prefill_ids.shape[1]
    assert b_fast._feature_tap._captured == {}
    assert b_fast._pooled_in_use is False

    def _boom_on_block(_block_info):
        raise RuntimeError("simulated callback failure")

    with pytest.raises(RuntimeError, match="simulated callback failure"):
        generate_reference(b_fast, prefill_ids, 12, temperature=0.0, on_block=_boom_on_block)
    assert b_fast._feature_tap._captured == {}
    assert b_fast._pooled_in_use is False

