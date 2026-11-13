import pytest
import torch
from torch.nn import functional as F

from edraft.sampler import SamplingWarper, sample_token
from edraft.toy import ToyCache, ToyParallelDrafter, ToyTarget
from edraft.verify import verify_block


def test_toy_parallel_features_are_fixed_and_head_hook_is_exact():
    target = ToyTarget()
    drafter = ToyParallelDrafter(target)
    block = drafter.draft([1, 4, 3], 4)
    assert block.h_raw.shape == (4, target.hidden_size)
    assert block.logits.shape == (4, target.vocab_size)
    assert block.physical_to_logical == (1, 2, 3, 4)
    assert not any(p.requires_grad for p in drafter.parameters())
    torch.testing.assert_close(drafter.draft_lm_head(block.h_raw), block.logits, rtol=0, atol=0)
    old_h = block.h_raw.clone()
    for row in block.logits:
        sample_token(row, 1)
    assert torch.equal(old_h, block.h_raw)
    assert torch.equal(block.h_raw, drafter.draft([1, 4, 3], 4).h_raw)


def test_bf16_hidden_reconstruction_tolerance_from_rounding_bound():
    target = ToyTarget()
    drafter = ToyParallelDrafter(target)
    block = drafter.draft([7, 3, 0, 1], 4)
    restored_h = block.h_raw.to(torch.bfloat16).float()
    reconstructed = F.linear(restored_h, drafter.draft_lm_head.weight.float(),
                             drafter.draft_lm_head.bias.float())
    # Elementwise bf16 roundoff propagated through the *frozen actual head*.
    error_bound = (restored_h - block.h_raw).abs() @ drafter.draft_lm_head.weight.abs().T
    assert ((reconstructed - block.logits).abs() <= error_bound + 5e-7).all()
    # log_softmax is at most 2-Lipschitz in the infinity norm.
    tolerance = 2 * error_bound.max().item() + 1e-6
    torch.testing.assert_close(reconstructed.log_softmax(-1), block.logits.log_softmax(-1),
                               atol=tolerance, rtol=0)


def test_target_verification_rows_use_realized_draft_prefix():
    target = ToyTarget(dtype=torch.float64)
    prefix, draft = [2, 7], [1, 5, 3, 0]
    logits = target.verify_logits(prefix, draft)
    for i in range(len(draft) + 1):
        assert torch.equal(logits[i], target.logits(prefix + draft[:i]))


def test_cache_partial_acceptance_rollback_preserves_target():
    target = ToyTarget()
    prefix = [6, 1, 2]
    first = int(target.logits(prefix).argmax())
    second = int(target.logits(prefix + [first]).argmax())
    draft = [first, (second + 1) % target.vocab_size, 7, 3]
    cache = ToyCache(target, prefix)
    target_logits = cache.append(draft)
    result = verify_block(draft, None, target_logits, 0)
    assert result.accepted_length == 1
    cache.crop(len(prefix) + result.accepted_length)
    cache.append([result.replacement_token])
    expected = prefix + result.emitted_tokens
    assert cache.tokens == expected
    assert torch.equal(cache.next_logits(), target.logits(expected))
    assert target.generate(cache.tokens, 12) == target.generate(expected, 12)


@pytest.mark.parametrize("prefix", [[], [0], [3, 2, 6]])
def test_greedy_speculative_output_equals_target_only(prefix):
    target = ToyTarget()
    drafter = ToyParallelDrafter(target)
    generated = []
    while len(generated) < 40:
        context = prefix + generated
        block = drafter.draft(context, 4)
        draft = [sample_token(row, 0)[0] for row in block.logits]
        result = verify_block(draft, None, target.verify_logits(context, draft), 0)
        generated.extend(result.emitted_tokens)
    assert generated[:40] == target.generate(prefix, 40)


def test_synthetic_constructors_deterministic_and_no_downloads(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic code attempted network access")
    monkeypatch.setattr(socket, "socket", forbidden)
    first = ToyParallelDrafter(ToyTarget(seed=11), seed=12).draft([1, 2], 4)
    second = ToyParallelDrafter(ToyTarget(seed=11), seed=12).draft([1, 2], 4)
    assert first.logits.device.type == "cpu"
    assert torch.equal(first.h_raw, second.h_raw)
    assert torch.equal(first.logits, second.logits)
