import dataclasses
import pytest
import torch
from edraft.adapter import (DenseAdapter, prev_token, loss_weights, initialize_lookahead,
    lookahead_fields, lookahead_energy, damped_lookahead_step, union_rest_grouped)
from edraft.features import fit_hidden_projection, fit_token_features, HiddenProjection
from edraft.sparse import SignedHash


def fixture_adapter(sparse=False):
    gen = torch.Generator().manual_seed(71)
    E = torch.randn(7, 3, generator=gen)
    adapter = DenseAdapter(E, p=4, D=3, sparse=SignedHash(257) if sparse else None, M_sparse=3)
    H = torch.randn(4, 4, generator=gen)
    logits = torch.randn(4, 7, generator=gen)
    target = torch.randn(4, 7, generator=gen)
    return adapter, H, logits.log_softmax(-1), target


def test_zero_correction_and_snapshot_ownership():
    adapter, H, l_unit, _ = fixture_adapter()
    snap = adapter.snapshot(C0=[torch.tensor([1, 3])] * 4)
    U = snap.unary(H)
    for i in range(4):
        assert torch.equal(snap.score_slot(l_unit[i], U[i], [2], [4] * i, i), l_unit[i])
    with torch.no_grad():
        adapter.W.fill_(10)
        adapter.A.fill_(10)
        adapter.b.fill_(10)
    assert snap.is_zero
    assert torch.count_nonzero(snap.W) == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.version = 9


def test_history_never_wraps_and_no_offset_zero():
    assert prev_token([4, 6], [1, 2, 3], 0, 1) == 6
    assert prev_token([4, 6], [1, 2, 3], 0, 2) == 4
    assert prev_token([4, 6], [1, 2, 3], 0, 3) is None
    assert prev_token([], [1, 2, 3], 1, 2) is None
    assert prev_token([], [1, 2, 3], 1, 1) == 1
    with pytest.raises(ValueError):
        prev_token([1], [2], 0, 0)


def test_causal_prefix_and_A_offset():
    adapter = DenseAdapter(torch.eye(3), p=1, D=2)
    with torch.no_grad():
        adapter.A[0, 1, 2] = 2
        adapter.A[1, 0, 1] = 3
    H = torch.zeros(2, 1)
    l_unit = torch.zeros(2, 3)
    corr = adapter.correction(l_unit, H, [1, 2], [0, 1])
    assert torch.equal(corr[0], torch.tensor([3., 2., 0.]))
    assert torch.equal(adapter.correction(l_unit, H, [], [0, 1])[0], torch.zeros(3))


def test_full_analytic_W_A_and_one_descent_step():
    adapter, H, l_unit, target = fixture_adapter()
    torch.manual_seed(3)
    with torch.no_grad():
        adapter.W.normal_(std=.08)
        adapter.A.normal_(std=.08)
    prefix, tokens = [2, 1], [6, 4, 3, 5]
    weights = loss_weights(4)
    P, q = adapter.training_distributions(l_unit, target, H, prefix, tokens)
    G = (q.detach() - P) @ adapter.E
    W_expected = -(G * weights[:, None]).T @ H
    A_expected = torch.zeros_like(adapter.A)
    for i in range(4):
        for d in range(1, 4):
            previous = prev_token(prefix, tokens, i, d)
            if previous is not None:
                A_expected[d - 1] -= weights[i] * torch.outer(G[i], adapter.E[previous])
    loss = adapter.full_loss(l_unit, target, H, prefix, tokens)
    loss.backward()
    torch.testing.assert_close(adapter.W.grad, W_expected, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(adapter.A.grad, A_expected, rtol=2e-5, atol=1e-6)
    before = float(loss.detach())
    with torch.no_grad():
        for p in adapter.parameters():
            p -= .005 * p.grad
    assert adapter.full_loss(l_unit, target, H, prefix, tokens) < before


def test_training_distributions_invariant_to_inference_warp():
    from edraft.sampler import SamplingWarper, probabilities
    adapter, H, l_unit, target = fixture_adapter()
    args = (l_unit, target, H, [2], [1, 2, 3, 4])
    before = adapter.training_distributions(*args)
    for warp in (SamplingWarper(), SamplingWarper(top_k=3, top_p=.8, min_p=.1)):
        probabilities(target[0], .7, warp)
        probabilities(l_unit[0], .7, warp)
        after = adapter.training_distributions(*args)
        assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_grouped_loss_partition_gradient_and_gap():
    torch.manual_seed(11)
    scores = torch.randn(2, 7, requires_grad=True)
    p = torch.softmax(torch.randn(2, 7), -1)
    ids = [torch.tensor([0, 1, 4]), torch.tensor([2, 3, 5])]
    pu = [p[i, u] for i, u in enumerate(ids)]
    rest = torch.stack([1 - x.sum() for x in pu])
    weights = torch.ones(2)
    loss = union_rest_grouped(scores, ids, pu, rest, weights)
    q = scores.softmax(-1)
    reference = 0
    for i, u in enumerate(ids):
        qr = 1 - q[i, u].sum()
        reference = reference + (pu[i] * (pu[i].log() - q[i, u].log())).sum() + rest[i] * (rest[i].log() - qr.log())
    torch.testing.assert_close(loss, reference)
    g1, = torch.autograd.grad(loss, scores, retain_graph=True)
    g2, = torch.autograd.grad(reference, scores)
    torch.testing.assert_close(g1, g2, atol=1e-6, rtol=1e-5)
    full = (p * (p.log() - q.log())).sum()
    assert full >= loss - 1e-6


@pytest.mark.parametrize('domain', ['candidate_masked', 'full_vocabulary'])
def test_lookahead_shared_energy_partial_derivatives(domain):
    adapter, H, l_unit, _ = fixture_adapter(sparse=True)
    torch.manual_seed(21)
    with torch.no_grad():
        adapter.W.normal_(std=.1)
        adapter.A.normal_(std=.1)
        adapter.sparse.values.normal_(std=.2)
    state = initialize_lookahead(adapter.snapshot(), l_unit, H, [5, 2], domain=domain)
    state.r = [r.detach().clone().requires_grad_() for r in state.r]
    state.mu = [r @ (state.snapshot.E[ids] if domain == 'candidate_masked' else state.snapshot.E)
                for r, ids in zip(state.r, state.support)]
    grads = torch.autograd.grad(lookahead_energy(state), state.r)
    forward, backward = lookahead_fields(state)
    fields = state.U_all + forward + backward
    for i, gradient in enumerate(grads):
        expected = fields[i, state.support[i]] if domain == 'candidate_masked' else fields[i]
        torch.testing.assert_close(gradient, expected, rtol=3e-5, atol=2e-6)
    # Sparse mass outside support is deliberately nonzero in full mode and is
    # never renormalized into support; this detects the common wrong derivative.
    if domain == 'full_vocabulary':
        assert all(float(state.raw_support_mass(i).sum().detach()) < 1 for i in range(4))


def test_lookahead_backward_transpose_and_boundaries():
    E = torch.eye(3)
    adapter = DenseAdapter(E, 1, D=1, M_sparse=3)
    with torch.no_grad():
        adapter.A[0] = torch.tensor([[0., 1., 2.], [3., 0., 4.], [5., 6., 0.]])
    state = initialize_lookahead(adapter.snapshot(), torch.zeros(2, 3), torch.zeros(2, 1), [1], domain='full_vocabulary')
    state.clamp(1, 2)
    forward, backward = lookahead_fields(state)
    torch.testing.assert_close(backward[0], E @ (adapter.A[0].T @ E[2]))
    assert torch.equal(backward[1], torch.zeros(3))
    torch.testing.assert_close(forward[0], E @ (adapter.A[0] @ E[1]))
    empty = initialize_lookahead(adapter.snapshot(), torch.zeros(2, 3), torch.zeros(2, 1), [], domain='full_vocabulary')
    assert torch.equal(lookahead_fields(empty)[0][0], torch.zeros(3))


@pytest.mark.parametrize('domain', ['candidate_masked', 'full_vocabulary'])
def test_greedy_lookahead_positive_relax_immutable_candidates(domain):
    adapter = DenseAdapter(torch.eye(4), 1, D=1, M_sparse=2)
    C0 = [torch.tensor([3, 2]), torch.tensor([1, 3])]
    snap = adapter.snapshot(C0=C0)
    l_unit = torch.tensor([[10., 0., 2., 2.], [8., 4., 1., 4.]])
    original = [c.clone() for c in snap.C0]
    outputs = []
    for repeat in range(2):
        state = initialize_lookahead(snap, l_unit, torch.zeros(2, 1), [], domain=domain, T_relax=.8)
        damped_lookahead_step(state)
        y0, p0 = state.sample_slot(0, 0.)
        assert p0 is None
        assert y0 == (2 if domain == 'candidate_masked' else 0)
        assert state.clamped[0] == y0
        assert state.support[0].tolist() == [y0]
        damped_lookahead_step(state)
        y1, _ = state.sample_slot(1, 0.)
        outputs.append((y0, y1))
    assert outputs[0] == outputs[1]
    assert all(torch.equal(a, b) for a, b in zip(original, snap.C0))
    with pytest.raises(ValueError):
        initialize_lookahead(snap, l_unit, torch.zeros(2, 1), [], T_relax=0)


def test_fixed_projection_calibration_roundtrip(tmp_path):
    torch.manual_seed(10)
    hidden = torch.randn(101, 8)
    proj = fit_hidden_projection(hidden.split(17), d_h=8, p=4, prompt_ids=['train-1'])
    z = proj(hidden)
    torch.testing.assert_close(z.mean(0), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(z.var(0, unbiased=False), torch.ones(4), atol=1e-6, rtol=0)
    path = tmp_path / 'h_proj.pt'
    proj.save(path)
    assert torch.equal(HiddenProjection.load(path)(hidden), z)
    E, metadata = fit_token_features(torch.randn(20, 8), r=3, calibration_token_ids=torch.arange(10))
    torch.testing.assert_close(E[:10].mean(0), torch.zeros(3), atol=1e-6, rtol=0)
    assert metadata['calibration_token_count'] == 10


def test_true_candidate_dense_scores_match_full_on_same_support():
    adapter, H, l_unit, _ = fixture_adapter()
    with torch.no_grad():
        adapter.W.normal_(std=.1)
        adapter.A.normal_(std=.1)
        adapter.b.normal_(std=.1)
    snap = adapter.snapshot(C0=[torch.tensor([1, 4, 6])] * len(H))
    full_U = snap.unary(H)
    candidate_U = snap.unary(H, proposal_mode='dense_candidate')
    assert all(len(row) == 3 for row in candidate_U)
    for i in range(len(H)):
        full = snap.score_slot(l_unit[i], full_U[i], [3, 1], [2] * i, i)
        candidate = snap.score_slot(l_unit[i], candidate_U[i], [3, 1], [2] * i, i, proposal_mode='dense_candidate')
        ids = snap.C0[i]
        torch.testing.assert_close(candidate[ids], full[ids], rtol=1e-5, atol=1e-6)
        outside = torch.ones(7, dtype=torch.bool)
        outside[ids] = False
        assert torch.isneginf(candidate[outside]).all()


def test_full_vocabulary_candidate_unary_handles_permuted_all_token_ids():
    adapter, H, l_unit, _ = fixture_adapter()
    with torch.no_grad():
        adapter.W.normal_(std=.1)
    ids = torch.tensor([5, 1, 3, 6, 0, 4, 2])
    snap = adapter.snapshot(C0=[ids] * len(H))
    selected_U = snap.unary(H, proposal_mode='dense_candidate')
    full_U = snap.unary(H)
    for i in range(len(H)):
        candidate = snap.score_slot(l_unit[i], selected_U[i], [], [1] * i, i, proposal_mode='dense_candidate')
        full = snap.score_slot(l_unit[i], full_U[i], [], [1] * i, i)
        torch.testing.assert_close(candidate, full)


def test_controlled_shared_free_energy_small_step_descent_with_line_search():
    adapter, H, l_unit, _ = fixture_adapter(sparse=True)
    with torch.no_grad():
        adapter.A.normal_(std=.03)
        adapter.W.normal_(std=.03)
        adapter.sparse.values.normal_(std=.03)
    state = initialize_lookahead(adapter.snapshot(), l_unit, H, [2, 4], domain='candidate_masked')
    # Perturb beliefs away from their unary optimum while keeping interior mass.
    state.r = [torch.full_like(r, 1 / len(r)) for r in state.r]
    state.mu = [r @ state.snapshot.E[ids] for r, ids in zip(state.r, state.support)]
    def free_energy():
        return lookahead_energy(state) + sum((r * (r.log() - l_unit[i, ids])).sum()
            for i, (r, ids) in enumerate(zip(state.r, state.support)))
    initial = free_energy().detach()
    original = [r.clone() for r in state.r]
    forward, backward = lookahead_fields(state)
    gradient = [state.U_all[i, ids] + forward[i, ids] + backward[i, ids] + r.log() + 1 - l_unit[i, ids]
                for i, (r, ids) in enumerate(zip(state.r, state.support))]
    tangent = [g - g.mean() for g in gradient]
    accepted = False
    for step in [1., .5, .25, .125, .0625, .03125, .015625, .0078125]:
        candidate = [r - step * g for r, g in zip(original, tangent)]
        if any(bool((r <= 0).any()) for r in candidate):
            continue
        state.r = candidate
        state.mu = [r @ state.snapshot.E[ids] for r, ids in zip(state.r, state.support)]
        if free_energy() < initial:
            accepted = True
            break
    assert accepted


def test_grouped_full_union_has_no_numerical_tail():
    logits = torch.tensor([[1., 2., 3.]], requires_grad=True)
    p = logits.detach().softmax(-1)[0]
    loss = union_rest_grouped(logits, [torch.arange(3)], [p], torch.tensor([1e-8]))
    assert abs(float(loss.detach())) < 1e-6
    with pytest.raises(ValueError, match='full-vocabulary union'):
        union_rest_grouped(logits, [torch.arange(3)], [p], torch.tensor([.1]))
