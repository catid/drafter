"""Exercise actual lookahead sampling with target mass outside frozen support."""

import math

import pytest
import torch

from edraft import sampler
from edraft.adapter import DenseAdapter, initialize_lookahead
from edraft.sampler import SamplingWarper, probabilities
from edraft.verify import residual_distribution, verify_block


@pytest.mark.parametrize('min_p', [0., .3])
def test_positive_lookahead_masks_before_warper_and_preserves_outside_residual(monkeypatch, min_p):
    adapter = DenseAdapter(torch.eye(8), p=1, D=1, M_sparse=4).double()
    with torch.no_grad():
        # Candidate corrected scores are token1=4, token6=3, token4=2,
        # token2=1, despite a different ordering in the uncorrected logits.
        adapter.b[1] = -3.5
        adapter.b[2] = 2.
        adapter.b[6] = -2.
    candidates = [torch.tensor([6, 2, 4, 1])] * 2
    snapshot = adapter.snapshot(version=7, C0=candidates)
    original_candidates = [ids.clone() for ids in snapshot.C0]
    original_parameters = [value.clone() for value in (snapshot.W, snapshot.A, snapshot.b)]
    # The three globally highest scores are outside C0. Warping before
    # masking would discard every allowed token and fail this test.
    logits = torch.tensor([[20., .5, 3., 19., 2., 18., 1., -10.]] * 2,
                          dtype=torch.float64)
    l_unit = logits.log_softmax(-1)
    original_l_unit = l_unit.clone()
    state = initialize_lookahead(snapshot, l_unit, torch.zeros(2, 1, dtype=torch.float64),
                                 prefix=[3], domain='candidate_masked', T_relax=.9)
    outside = torch.ones(8, dtype=torch.bool)
    outside[snapshot.C0[0]] = False
    warper = SamplingWarper(top_k=3, top_p=.8, min_p=min_p)
    temperature = .7
    seen_scores = []
    original_sample = sampler.sample_token

    def inspect_input(scores, T_sample, sampling_warper, **kwargs):
        seen_scores.append(scores.clone())
        assert scores.shape == (8,)
        assert torch.isneginf(scores[outside]).all()
        torch.testing.assert_close(scores[snapshot.C0[0]],
                                   (l_unit[0] - state.g[0])[snapshot.C0[0]])
        assert T_sample == temperature and sampling_warper == warper
        return original_sample(scores, T_sample, sampling_warper, **kwargs)

    monkeypatch.setattr(sampler, 'sample_token', inspect_input)
    token, q = state.sample_slot(0, temperature, warper=warper,
                                 generator=torch.Generator().manual_seed(2026))
    assert len(seen_scores) == 1
    # top-k first keeps scores4,3,2; top-p then keeps4,3. Their score gap
    # is1, giving this independent two-token categorical reference.
    expected = torch.zeros(8, dtype=torch.float64)
    expected[1] = 1 / (1 + math.exp(-1 / temperature))
    expected[6] = 1 - expected[1]
    if min_p:
        expected.zero_()
        expected[1] = 1.
    torch.testing.assert_close(q, expected, rtol=1e-14, atol=1e-14)
    assert q[outside].eq(0).all()
    assert q.sum().item() == pytest.approx(1., abs=1e-14)
    assert token in snapshot.C0[0].tolist()
    assert state.clamped == [token, None]
    assert state.support[0].tolist() == [token]
    assert state.r[0].tolist() == [1.]
    assert torch.equal(state.mu[0], snapshot.E[token])
    assert torch.equal(state.support[1], original_candidates[1])
    assert all(torch.equal(a, b) for a, b in zip(snapshot.C0, original_candidates))
    assert all(torch.equal(a, b) for a, b in zip((snapshot.W, snapshot.A, snapshot.b), original_parameters))
    assert snapshot.version == 7 and torch.equal(l_unit, original_l_unit)

    # The same target warper retains token0 outside C0 and token1 inside.
    # Positive target mass outside the proposal must survive the residual.
    target_logits = torch.tensor([3.7, 3.5, -9., 2.8, -7., 1., -6., -5.],
                                 dtype=torch.float64)
    P = probabilities(target_logits, temperature, warper)
    assert 0 < P[outside].sum() < 1
    residual = residual_distribution(P, q)
    assert residual[outside].sum().item() == pytest.approx(1., abs=1e-14)
    accepted_mass = torch.minimum(P, q)
    emitted = accepted_mass + (1 - accepted_mass.sum()) * residual
    torch.testing.assert_close(emitted, P, rtol=1e-14, atol=1e-14)
    # Seed0's first FP64 acceptance draw exceeds every possible sampled
    # token's acceptance ratio here, making the actual rejection deterministic.
    result = verify_block([token], q[None, :], torch.stack([target_logits, target_logits]),
                          temperature, warper, generator=torch.Generator().manual_seed(0))
    assert result.accepted_length == 0 and result.first_rejection == 0
    assert result.decisions == [False] and result.verification_rule == 'ratio'
    assert result.replacement_token == 0 and result.emitted_tokens == [0]
    assert result.bonus_token is None
    assert all(torch.equal(a, b) for a, b in zip(snapshot.C0, original_candidates))
