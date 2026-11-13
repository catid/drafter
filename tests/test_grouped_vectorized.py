import pytest
import torch

from edraft.bench_grouped import grouped_loop_reference, grouped_vectorized_candidate


@pytest.mark.parametrize('seed', [2, 91, 611])
@pytest.mark.parametrize('extreme', [False, True])
def test_ragged_grouped_loss_and_every_score_gradient_match_loop(seed, extreme):
    generator = torch.Generator().manual_seed(seed)
    scores = (torch.randn(6, 19, generator=generator) * (500 if extreme else 2)).requires_grad_(True)
    P = torch.randn(6, 19, generator=generator).softmax(-1)
    ids = [torch.randperm(19, generator=generator)[:n].sort().values for n in [0, 1, 3, 7, 18, 19]]
    target = [P[i, selected] for i, selected in enumerate(ids)]
    rest = torch.stack([1 - row.sum() if len(row) < 19 else row.new_tensor(1e-8) for row in target])
    weights = torch.tensor([1., .8, .6, .4, .2, 0.])
    reference = grouped_loop_reference(scores, ids, target, rest, weights)
    refgrad = torch.autograd.grad(reference, scores)[0]
    result = grouped_vectorized_candidate(scores, ids, target, rest, weights)
    gradient = torch.autograd.grad(result, scores)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(result, reference, atol=3e-5, rtol=2e-6)
    torch.testing.assert_close(gradient, refgrad, atol=2e-6, rtol=2e-5)


def test_full_union_gradient_is_full_kl_and_tiny_rest_cannot_create_nan():
    scores = torch.tensor([[3., 2., 1.], [-4., 5., 0.]], requires_grad=True)
    P = torch.tensor([[.1, .3, .6], [.3, .2, .5]])
    ids = [torch.arange(3), torch.arange(3)]
    weights = torch.tensor([1., .7])
    result = grouped_vectorized_candidate(scores, ids, list(P), torch.tensor([0., 1e-8]), weights)
    gradient = torch.autograd.grad(result, scores)[0]
    expected = (scores.softmax(-1) - P) * weights[:, None]
    torch.testing.assert_close(gradient, expected, atol=2e-7, rtol=2e-6)
    assert torch.isfinite(result) and torch.isfinite(gradient).all()
    with pytest.raises(ValueError, match='nonzero target rest'):
        grouped_vectorized_candidate(scores, ids, list(P), torch.tensor([0., .01]))


def test_nonzero_extremely_small_proposal_tail_keeps_exact_gradient():
    scores = torch.tensor([[1000., 999., -1000., -999.]], requires_grad=True)
    ids, P, rest = [torch.tensor([0, 1])], [torch.tensor([.4, .4])], [torch.tensor(.2)]
    result = grouped_vectorized_candidate(scores, ids, P, rest)
    gradient = torch.autograd.grad(result, scores)[0]
    reference = grouped_loop_reference(scores, ids, P, rest)
    refgrad = torch.autograd.grad(reference, scores)[0]
    assert torch.isfinite(result) and torch.isfinite(gradient).all()
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    torch.testing.assert_close(gradient, refgrad, atol=2e-7, rtol=2e-6)
    # Clamping 1-sum(q_union) to epsilon would erase this tail gradient.
    assert gradient[0, 2] < 0 and gradient[0, 3] < 0


def test_target_distributions_and_input_tensors_are_immutable():
    scores = torch.randn(2, 7, requires_grad=True)
    ids = [torch.tensor([1, 5]), torch.tensor([2, 4, 6])]
    targets = [torch.tensor([.2, .3], requires_grad=True), torch.tensor([.2, .1, .4], requires_grad=True)]
    rest = torch.tensor([.5, .3], requires_grad=True)
    original = [scores.detach().clone(), *(x.clone() for x in ids), *(x.detach().clone() for x in targets), rest.detach().clone()]
    grouped_vectorized_candidate(scores, ids, targets, rest).backward()
    assert all(x.grad is None for x in targets) and rest.grad is None
    current = [scores, *ids, *targets, rest]
    assert all(torch.equal(a, b) for a, b in zip(original, current))
    with pytest.raises(ValueError, match='one scalar per score row'):
        grouped_vectorized_candidate(scores, ids, targets, torch.tensor([.5]))
