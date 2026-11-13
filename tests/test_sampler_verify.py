from collections import defaultdict
from functools import lru_cache
from itertools import product
import math

import pytest
import torch

from edraft import sampler, verify
from edraft.sampler import SamplingWarper, apply_warper, mask_scores, probabilities, sample_token
from edraft.verify import residual_distribution, verify_block


@pytest.mark.parametrize("kwargs", [{"top_k": -1}, {"top_k": 2.5}, {"top_p": 0},
                                    {"top_p": 1.1}, {"min_p": -0.1}, {"min_p": 1.1}])
def test_invalid_warper(kwargs):
    with pytest.raises(ValueError):
        SamplingWarper(**kwargs)


@pytest.mark.parametrize("temperature", [-1, float("inf"), float("nan")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        sample_token(torch.ones(4), temperature)


def test_greedy_never_calls_sampling_softmax_or_residual(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("greedy entered a stochastic operation")
    monkeypatch.setattr(torch, "softmax", forbidden)
    monkeypatch.setattr(torch, "multinomial", forbidden)
    monkeypatch.setattr(torch, "rand", forbidden)
    monkeypatch.setattr(verify, "residual_distribution", forbidden)
    scores = torch.tensor([3.0, 3.0, 1.0, 3.0])
    token, q = sample_token(scores, 0, SamplingWarper(top_k=1, top_p=0.1))
    assert token == 0 and q is None
    result = verify_block([0, 3], None, torch.tensor([[3., 3., 1., 3.],
                                                    [0., 1., 2., 3.],
                                                    [0., 7., 7., 0.]]), 0)
    assert result.accepted_length == 2
    assert result.first_rejection == 2
    assert result.emitted_tokens == [0, 3, 1]
    result = verify_block([3, 3], None, torch.tensor([[3., 3., 1., 3.],
                                                    [0., 1., 2., 3.],
                                                    [0., 7., 7., 0.]]), 0)
    assert result.emitted_tokens == [0] and result.first_rejection == 0


def test_candidate_masking_and_lowest_vocabulary_tie():
    scores = torch.tensor([3., 8., 8., 9., 0.], dtype=torch.float64)
    C0 = torch.tensor([2, 1])  # Candidate order is deliberately not token-ID order.
    original = C0.clone()
    masked = mask_scores(scores, C0)
    assert sample_token(masked, 0)[0] == 1
    q = probabilities(masked, 0.7, SamplingWarper(top_k=3, top_p=0.8, min_p=0.1))
    assert q[[0, 3, 4]].eq(0).all()
    assert q.sum().item() == pytest.approx(1, abs=1e-15)
    assert torch.equal(C0, original)
    assert scores[3] == 9  # Neither input was modified.


def test_warper_applies_after_correction_and_matches_pinned_dflash_formula():
    raw = torch.tensor([[8., 5., 4., 3., 0.], [3., 1., 4., 2., 0.]], dtype=torch.float64)
    correction = torch.tensor([[10., 10., 0., -3., -6.], [0., 0., 0., 0., 0.]], dtype=torch.float64)
    scores = raw - correction
    warper = SamplingWarper(top_k=3, top_p=0.8)
    q = probabilities(scores, 0.7, warper)
    selected, ids = (scores / 0.7).topk(3, dim=-1)
    expected_small = selected.softmax(-1)
    ordered, order = expected_small.sort(descending=True)
    ordered *= ordered.cumsum(-1) - ordered < 0.8
    expected_small = torch.zeros_like(expected_small).scatter(-1, order, ordered)
    expected_small /= expected_small.sum(-1, keepdim=True)
    expected = torch.zeros_like(raw).scatter(-1, ids, expected_small)
    assert torch.equal(q, expected)
    assert q[0, 4] > 0  # Initially lowest raw logit can be retained after correction.
    masked = apply_warper(scores / 0.7, warper).softmax(-1)
    torch.testing.assert_close(q, masked, atol=1e-14, rtol=1e-14)


def test_min_p_and_dtype():
    q = probabilities(torch.log(torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)),
                      1, SamplingWarper(min_p=0.2))
    torch.testing.assert_close(q, torch.tensor([7/9, 2/9, 0], dtype=torch.float64))
    assert probabilities(torch.ones(3, dtype=torch.bfloat16), 1).dtype == torch.float32
    assert probabilities(torch.ones(3, dtype=torch.float64), 1).dtype == torch.float64
    with pytest.raises(ValueError):
        probabilities(torch.ones(3), 0)


def test_forced_residual_outside_proposal_support_and_bonus():
    q = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
    target = torch.tensor([[-torch.inf, -torch.inf, 0., -torch.inf],
                           [0., -torch.inf, -torch.inf, -torch.inf]], dtype=torch.float64)
    result = verify_block([0], q, target, 1, SamplingWarper(top_k=3, top_p=0.8))
    assert result.emitted_tokens == [2]
    assert result.replacement_token == 2
    assert result.first_rejection == 0
    assert result.decisions == [False]
    assert result.proposal_token_probs == [1.]
    assert result.target_token_probs == [0.]
    target[0] = torch.tensor([0., -torch.inf, -torch.inf, -torch.inf])
    target[1] = torch.tensor([-torch.inf, -torch.inf, -torch.inf, 0.])
    result = verify_block([0], q, target, 1)
    assert result.emitted_tokens == [0, 3]
    assert result.first_rejection == 1 and result.bonus_token == 3
    assert result.replacement_token is None


def test_rejects_nonratio_rule_and_impossible_proposal():
    with pytest.raises(ValueError, match="requires verification_rule"):
        verify_block([0], torch.tensor([[1., 0.]]), torch.ones(2, 2), 1,
                     verification_rule="independent_match")
    with pytest.raises(ValueError, match="positive finite"):
        verify_block([0], torch.tensor([[0., 1.]]), torch.ones(2, 2), 1)
    with pytest.raises(RuntimeError, match="no positive"):
        residual_distribution(torch.tensor([.5, .5]), torch.tensor([.5, .5]))


@pytest.mark.parametrize("warper", [SamplingWarper(), SamplingWarper(top_k=3, top_p=0.8),
                                    SamplingWarper(top_k=3, top_p=0.8, min_p=0.15)])
@pytest.mark.parametrize("K", [1, 2, 4])
def test_exhaustive_per_prefix_ratio_identity(warper, K):
    """Enumerate every realized prefix, including target mass outside frozen C0.

    This is exact enumeration, so tolerance comes from FP64 arithmetic rather
    than a sample count. Every prefix through K-1 and every vocabulary outcome
    is checked. Causal corrections use the realized predecessor.
    """
    V = 4
    base = torch.tensor([2., 1.5, 0.5, -0.5], dtype=torch.float64)
    for n in range(K):
        for prefix in product(range(V), repeat=n):
            previous = prefix[-1] if prefix else 3  # cross-boundary verified anchor
            correction = torch.tensor([0.1 * previous, -0.3 * previous, 0.7, 0.2],
                                      dtype=torch.float64)
            q = probabilities(mask_scores(base - correction, torch.tensor([0, 1])), 0.7, warper)
            target_scores = torch.tensor([0.3, -0.2, 1.8, 0.4], dtype=torch.float64).roll(previous)
            P = probabilities(target_scores, 0.7, warper)
            accepted_mass = torch.minimum(P, q)
            rejection_mass = 1 - accepted_mass.sum()
            emitted = accepted_mass.clone()
            if rejection_mass > 1e-14:
                emitted += rejection_mass * residual_distribution(P, q)
            torch.testing.assert_close(emitted, P, atol=4e-15, rtol=4e-15)


def test_exhaustive_multiround_sequence_and_bonus_exactness():
    """Enumerate all drafts/acceptances/residual/bonus branches for a 3-token stream.

    The target is prefix dependent and proposal slots depend on realized IDs.
    Multiple rounds are required on rejection. This checks joint distributions,
    not only unigram marginals or independently sampled match acceptance.
    """
    V, K, N = 3, 2, 3
    warper = SamplingWarper(top_k=3, top_p=0.8)

    def target(prefix):
        last = prefix[-1] if prefix else 2
        return probabilities(torch.tensor([0.3, 0.0, -0.4], dtype=torch.float64).roll(last), 1, warper)

    def proposal(prefix, realized, i):
        last = realized[-1] if realized else (prefix[-1] if prefix else 2)
        scores = torch.tensor([0.1, -0.3, 0.2], dtype=torch.float64).roll(last + i)
        return probabilities(mask_scores(scores, torch.tensor([0, 2])), 1, warper)

    @lru_cache(None)
    def round_branches(prefix):
        emitted = defaultdict(float)
        for draft in product(range(V), repeat=K):
            qs = [proposal(prefix, draft[:i], i) for i in range(K)]
            draft_mass = math.prod(float(qs[i][draft[i]]) for i in range(K))
            if draft_mass == 0:
                continue
            surviving = draft_mass
            for i, token in enumerate(draft):
                P = target(prefix + draft[:i])
                alpha = min(1., float(P[token] / qs[i][token]))
                if alpha < 1:
                    residual = residual_distribution(P, qs[i])
                    for replacement in range(V):
                        emitted[draft[:i] + (replacement,)] += surviving * (1-alpha) * float(residual[replacement])
                surviving *= alpha
            bonus_P = target(prefix + draft)
            for bonus in range(V):
                emitted[draft + (bonus,)] += surviving * float(bonus_P[bonus])
        assert sum(emitted.values()) == pytest.approx(1, abs=2e-14)
        return tuple(emitted.items())

    actual = defaultdict(float)
    def extend(prefix=(), mass=1.):
        if len(prefix) >= N:
            actual[prefix[:N]] += mass
            return
        for emitted, probability in round_branches(prefix):
            if probability > 0:
                extend(prefix + emitted, mass * probability)
    extend()
    for sequence in product(range(V), repeat=N):
        expected = math.prod(float(target(sequence[:i])[sequence[i]]) for i in range(N))
        assert actual[sequence] == pytest.approx(expected, abs=3e-14)
    # Recover each conditional distribution from the enumerated joint law.
    for n in range(N):
        for prefix in product(range(V), repeat=n):
            denominator = sum(p for s, p in actual.items() if s[:n] == prefix)
            if denominator == 0:
                continue
            conditional = torch.tensor([sum(p for s, p in actual.items()
                                             if s[:n+1] == prefix + (a,)) / denominator
                                        for a in range(V)], dtype=torch.float64)
            torch.testing.assert_close(conditional, target(prefix), atol=3e-14, rtol=3e-14)


def test_actual_verifier_sampling_sample_size_tolerance():
    # A union-bound Hoeffding tolerance: with probability >=1-delta all V
    # frequencies are within epsilon. The fixed seed makes the test repeatable.
    V, K, samples, delta = 4, 4, 5000, 1e-7
    epsilon = math.sqrt(math.log(2 * V / delta) / (2 * samples))
    target_logits = torch.tensor([[0.4, 0.2, -0.2, 1.0]] * (K+1), dtype=torch.float64)
    warper = SamplingWarper(top_k=3, top_p=0.8, min_p=0.05)
    q = probabilities(torch.tensor([0.6, -0.1, -torch.inf, -torch.inf], dtype=torch.float64), 1, warper)
    target_P = probabilities(target_logits[0], 1, warper)
    generator = torch.Generator().manual_seed(1776)
    counts = torch.zeros(V, dtype=torch.float64)
    rejections = bonuses = 0
    for _ in range(samples):
        draft = torch.multinomial(q.expand(K, -1), 1, generator=generator).flatten()
        result = verify_block(draft, q.expand(K, -1), target_logits, 1, warper, generator)
        counts[result.emitted_tokens[0]] += 1
        rejections += result.replacement_token is not None
        bonuses += result.bonus_token is not None
    assert (counts / samples - target_P).abs().max() <= epsilon
    assert rejections > 0 and bonuses > 0


def test_vector_acceptance_rng_matches_upstream_draw_order():
    K, V = 4, 4
    q = torch.tensor([[0.7, 0.1, 0.1, 0.1]] * K)
    logits = torch.tensor([[0., 1., 2., 3.]] * (K+1))
    P = probabilities(logits, 1)
    ours_rng = torch.Generator().manual_seed(934)
    reference_rng = torch.Generator().manual_seed(934)
    result = verify_block([0] * K, q, logits, 1, generator=ours_rng)
    uniforms = torch.rand(K, generator=reference_rng)
    accepted = int((uniforms * q[:, 0] < P[:K, 0]).int().cumprod(0).sum())
    distribution = P[K] if accepted == K else residual_distribution(P[accepted], q[accepted])
    extra = int(torch.multinomial(distribution, 1, generator=reference_rng))
    assert result.emitted_tokens == [0] * accepted + [extra]
    assert torch.equal(ours_rng.get_state(), reference_rng.get_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature,warper", [(1., SamplingWarper()),
    (.7, SamplingWarper(top_k=20, top_p=.8)), (.6, SamplingWarper(top_k=20, top_p=.95))])
def test_pinned_upstream_sampling_probabilities_bit_identical(dtype, temperature, warper):
    # Importing pinned source requires no checkpoint/network access. This test
    # guards the upstream convention that the hard M0 seeded gate depends on.
    from dflash.model import _sampling_probs
    generator = torch.Generator().manual_seed(125)
    logits = torch.randn(1, 16, 100, generator=generator, dtype=dtype)
    upstream = _sampling_probs(logits, temperature, warper.top_p, warper.top_k)
    assert torch.equal(probabilities(logits, temperature, warper), upstream)


@pytest.mark.parametrize("temperature,warper", [(1., SamplingWarper()),
                                                (.7, SamplingWarper(top_k=20, top_p=.8))])
def test_pinned_upstream_batch_rng_and_verification_bit_identical(temperature, warper):
    from dflash.model import _sampling_probs, _sample_probs, _rejection_sample
    with torch.random.fork_rng(devices=[]):
        for seed in range(8):
            torch.manual_seed(seed)
            draft_logits = torch.randn(1, 4, 100, dtype=torch.bfloat16)
            target_logits = torch.randn(1, 5, 100, dtype=torch.bfloat16)
            q = _sampling_probs(draft_logits, temperature, warper.top_p, warper.top_k)
            P = _sampling_probs(target_logits, temperature, warper.top_p, warper.top_k)
            torch.manual_seed(seed)
            draft = _sample_probs(q)
            accepted, extra = _rejection_sample(draft, P, q)
            upstream_state = torch.get_rng_state()
            torch.manual_seed(seed)
            local_draft = torch.multinomial(probabilities(draft_logits[0], temperature, warper), 1)[:, 0]
            result = verify_block(local_draft, q[0], target_logits[0], temperature, warper)
            assert torch.equal(draft[0], local_draft)
            assert result.emitted_tokens == draft[0, :accepted].tolist() + [int(extra)]
            assert torch.equal(upstream_state, torch.get_rng_state())
