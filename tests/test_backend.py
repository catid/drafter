"""Model-class/cache integration, using tiny random Qwen3 weights without downloads."""
import pytest
import torch

from edraft.backend import ReferenceBackend, crop_cache, make_cache
from edraft.dflash_loop import generate_reference, generate_toy
from edraft.sampler import SamplingWarper


@pytest.fixture(scope="module")
def backend():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from dflash.model import DFlashDraftModel

    torch.set_num_threads(1)
    torch.manual_seed(42)
    common = dict(vocab_size=17, hidden_size=24, intermediate_size=32, num_attention_heads=3,
                  num_key_value_heads=1, head_dim=8, max_position_embeddings=256,
                  attention_dropout=0., bos_token_id=1, eos_token_id=None)
    target_config = Qwen3Config(num_hidden_layers=3, **common)
    target_config._attn_implementation = "sdpa"
    draft_config = Qwen3Config(num_hidden_layers=1, **common)
    draft_config._attn_implementation = "sdpa"
    draft_config.num_target_layers = 3
    draft_config.block_size = 4
    draft_config.dflash_config = {"target_layer_ids": [1], "mask_token_id": 16}
    return ReferenceBackend(Qwen3ForCausalLM(target_config), DFlashDraftModel(draft_config))


@pytest.mark.parametrize("temperature,warper", [(0., SamplingWarper()), (.7, SamplingWarper(3, .8)), (1., SamplingWarper())])
def test_installed_qwen_dflash_upstream_local_seeded_parity(backend, temperature, warper):
    # This is synthetic integration evidence, not the 20-real-prompt M0 gate.
    for prompt in [[1, 3, 4], [2], [1, 5, 7, 2, 9]]:
        ids = torch.tensor([prompt])
        torch.manual_seed(923)
        upstream = backend.upstream_generate(ids, 25, temperature=temperature,
                                             top_k=warper.top_k, top_p=warper.top_p, stop_token_ids=[])
        torch.manual_seed(923)
        local = generate_reference(backend, ids, 25, temperature=temperature, warper=warper, stop_token_ids=[])
        assert torch.equal(upstream, local.output_ids)
        assert local.num_output_tokens == 25
        if temperature == 0:
            target = backend.target_only_generate(ids, 25, stop_token_ids=[])
            assert torch.equal(target, local.output_ids)
        assert all(b["target_cache_length_after_rollback"] >= len(prompt) for b in local.blocks)


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_actual_target_cache_partial_rollback(backend, accepted):
    evidence = backend.check_cache_rollback(torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]]), accepted, 8)
    assert evidence["cache_length"] == evidence["expected_length"]
    assert evidence["argmax_equal"]
    assert evidence["max_abs_logit_error"] < 1e-5


def test_head_hook_reconstructs_live_installed_classes(backend):
    live = []
    generate_reference(backend, torch.tensor([[1, 2]]), 12, on_block=live.append, stop_token_ids=[])
    assert live
    for block in live:
        reconstructed = torch.nn.functional.linear(block["h_raw_bf16"].float(), backend.draft_lm_head.weight.float())
        live_logp = block["draft_logits"].float().log_softmax(-1)
        assert torch.allclose(reconstructed.log_softmax(-1), live_logp, atol=.01, rtol=.002)
        assert block["physical_to_logical"]["anchor_physical_index"] == 0
        assert block["physical_to_logical"]["logical_to_physical"] == list(range(1, len(block["draft_tokens"]) + 1))
        assert len(block["h_raw"]) == len(block["draft_tokens"])


def test_stopping_caps_and_first_token(backend):
    ids = torch.tensor([[1, 2]])
    first = backend.target_only_generate(ids, 1, stop_token_ids=[])[0, -1].item()
    stopped = generate_reference(backend, ids, 20, stop_token_ids=[first])
    assert stopped.num_output_tokens == 1
    assert stopped.blocks == []
    for cap in [0, 1, 2, 3, 7]:
        result = generate_reference(backend, ids, cap, stop_token_ids=[])
        assert result.num_output_tokens == cap


def test_real_loop_zero_adapter_exact(backend):
    from edraft.adapter import DenseAdapter

    adapter = DenseAdapter(torch.randn(17, 3), p=24, D=2)
    ids = torch.tensor([[1, 2, 4]])
    for temperature in [0., .7, 1.]:
        torch.manual_seed(82)
        plain = generate_reference(backend, ids, 20, temperature=temperature, stop_token_ids=[])
        torch.manual_seed(82)
        zero = generate_reference(backend, ids, 20, temperature=temperature, adapter=adapter, stop_token_ids=[])
        assert torch.equal(plain.output_ids, zero.output_ids)


def test_cpu_smoke_round_update_and_version_isolation():
    from edraft.adapter import DenseAdapter
    from edraft.online import OnlineState
    from edraft.toy import ToyTarget, ToyParallelDrafter

    target = ToyTarget()
    drafter = ToyParallelDrafter(target)
    adapter = DenseAdapter(target.embedding[:, :3], p=target.hidden_size, D=2)
    state = OnlineState(adapter, lr=.01, update_stride=1)
    frozen = []
    result = generate_toy(target, drafter, [1, 3], 40, temperature=.7, online_state=state,
                          generator=torch.Generator().manual_seed(7), retain_tensors=True,
                          on_block=lambda record: frozen.append(record["adapter_version"]))
    assert result.num_output_tokens == 40
    assert state.u == len(result.blocks)
    assert frozen == list(range(len(result.blocks)))
    assert any(b["replacement_token"] is not None for b in result.blocks)
    assert all(b["next_adapter_version"] == b["adapter_version"] + 1 for b in result.blocks)


def test_zero_strength_bigram_bit_exact_and_initial_transition(backend):
    from edraft.online import OnlineBigram

    ids = torch.tensor([[1, 2, 4]])
    for temperature in [0., .7, 1.]:
        torch.manual_seed(11)
        plain = generate_reference(backend, ids, 20, temperature=temperature, stop_token_ids=[])
        bigram = OnlineBigram(17, lambda_tr=0)
        torch.manual_seed(11)
        baseline = generate_reference(backend, ids, 20, temperature=temperature,
                                      bigram=bigram, stop_token_ids=[])
        assert torch.equal(plain.output_ids, baseline.output_ids)
        assert bigram.metrics()["counted_transitions"] == 20
        assert bigram.counts[4][baseline.output_tokens[0]] > 0


@pytest.mark.parametrize('arm', ['plain', 'adapter', 'bigram', 'online'])
def test_bounded_history_preserves_long_prefix_tokens_and_real_context_lengths(backend, arm):
    from edraft.adapter import DenseAdapter
    from edraft.online import OnlineState, OnlineBigram
    torch.manual_seed(818)
    adapter = DenseAdapter(torch.randn(17, 3), p=24, D=7)
    with torch.no_grad():
        adapter.W.normal_(std=.01)
        adapter.A.normal_(std=.01)
    ids = torch.tensor([[1, 2, 4, 7, 10] * 12])
    def options():
        if arm == 'adapter':
            return {'adapter': adapter}
        if arm == 'bigram':
            return {'bigram': OnlineBigram(17, lambda_tr=.1)}
        if arm == 'online':
            return {'online_state': OnlineState(adapter, lr=.001)}
        return {}
    torch.manual_seed(19)
    bounded = generate_reference(backend, ids, 24, temperature=.7, stop_token_ids=[], **options())
    torch.manual_seed(19)
    retained = generate_reference(backend, ids, 24, temperature=.7, stop_token_ids=[], retain_tensors=True, **options())
    assert torch.equal(bounded.output_ids, retained.output_ids)
    assert len(bounded.blocks) == len(retained.blocks)
    for short, full in zip(bounded.blocks, retained.blocks):
        assert short['input_context_length'] == ids.shape[1] + short['absolute_output_offset']
        assert short['input_context_length'] == len(full['prefix'])
        assert short['verified_prefix_tail'] == full['verified_prefix_tail']
        assert len(short['verified_prefix_tail']) <= (7 if arm in ('adapter', 'online') else 4)
