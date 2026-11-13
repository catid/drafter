import pytest
import torch
from edraft.adapter import DenseAdapter
from edraft.online import OnlineState, OnlineBigram


def inputs(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(3, 7, generator=generator).log_softmax(-1),
            torch.randn(3, 7, generator=generator), torch.randn(3, 4, generator=generator),
            [1, 2], [3, 4, 1])


@pytest.mark.parametrize('stride', [1, 5])
def test_online_single_round_stride_snapshot_and_request_isolation(stride):
    torch.manual_seed(7)
    adapter = DenseAdapter(torch.randn(7, 3), p=4, D=2, M_sparse=3)
    state = OnlineState(adapter, update_stride=stride, enable_sparse=True, sparse_capacity=67,
                        lambda_l1_s=.001, lambda_l1_b=.001, lambda_l2=.02)
    isolated = OnlineState(adapter)
    initial = state.snapshot()
    initial_W, initial_b = initial.W.clone(), initial.b.clone()
    control = OnlineState(adapter, update_stride=1, enable_sparse=True, sparse_capacity=67,
                          lambda_l1_s=.001, lambda_l1_b=.001, lambda_l2=.02)
    firing = 0
    for round_idx in range(1, 3 * stride + 1):
        args = inputs(round_idx)
        changed = state.update(round_idx, *args)
        assert changed == (round_idx % stride == 0)
        if changed:
            firing += 1
            control.update(firing, *args)
        assert state.u == firing and state.version == firing
        # Skipped blocks have no effect, even though their inputs are different.
        assert torch.equal(state.db, control.db)
        assert torch.equal(state.dA, control.dA)
        assert torch.equal(state.dS.read(torch.arange(67)), control.dS.read(torch.arange(67)))
    assert torch.equal(initial.W, initial_W) and torch.equal(initial.b, initial_b)
    assert isolated.version == 0 and isolated.u == 0
    assert torch.count_nonzero(isolated.db) == 0
    assert torch.count_nonzero(adapter.W) == 0
    assert state.metrics()['loss_mode'] == 'live_full'


def test_online_bias_only_and_inference_mode_context():
    torch.manual_seed(2)
    adapter = DenseAdapter(torch.randn(7, 3), p=4, D=2)
    state = OnlineState(adapter, bias_sparse_only=True)
    with torch.inference_mode():
        args = inputs(3)
        assert state.update(1, *args)
    assert state.db.norm() > 0
    assert state.dW.norm() == 0 and state.dA.norm() == 0


@pytest.mark.parametrize('alpha', [0., -1.])
def test_bigram_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        OnlineBigram(7, alpha=alpha)


@pytest.mark.parametrize('k_ret', [None, 2])
def test_bigram_normalized_evicted_prior_and_empty_equivalence(k_ret):
    baseline = OnlineBigram(7, alpha=.3, lambda_tr=.5, k_ret=k_ret)
    l_unit = torch.tensor([0., 1., 1., -2., -3., -1., -1.])
    assert baseline.score(l_unit, None) is l_unit
    assert baseline.score(l_unit, 4) is l_unit
    for token in [0, 1, 1, 2, 2, 2, 3, 4, 5, 6]:
        baseline.update([4], [token])
    prior = baseline.prior(4)
    assert abs(float(prior.exp().sum()) - 1) <= 1e-6
    if k_ret == 2:
        # Counts 3 and 2 retained; all other five IDs share 5 evicted counts.
        expected = torch.tensor([1., 2., 3., 1., 1., 1., 1.])
        torch.testing.assert_close(prior.exp(), (expected + .3) / 12.1)
    baseline.lambda_tr = 0
    assert baseline.score(l_unit, 4) is l_unit
    assert baseline.score(l_unit, 4).argmax() == 1
    other = OnlineBigram(7)
    assert not other.counts


def test_bigram_missing_prefix_skips_first_transition():
    baseline = OnlineBigram(5)
    baseline.update([], [1, 2, 3])
    assert set(baseline.counts) == {1, 2}
    assert baseline.counts[1][2] == 1
