import pytest
import torch
from edraft.sparse import hash_keys, proximal_l1, SignedHash, LazySignedHash


def test_hash_deterministic_signed_and_collision_report():
    ids = torch.arange(100)
    a = hash_keys(2, ids, ids + 1, 17)
    b = hash_keys(2, ids, ids + 1, 17)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert set(a[1].tolist()) == {-1., 1.}
    report = SignedHash(17).collision_report([(1, i, i + 1) for i in range(100)])
    assert report['collision_count'] >= 83
    assert report['unique_observed_keys'] == 100


def test_proximal_preserves_exact_zero():
    x = torch.tensor([-2., -.05, 0., .05, 2.])
    assert torch.equal(proximal_l1(x, .1), torch.tensor([-1.9, 0., 0., 0., 1.9]))


@pytest.mark.parametrize('stride', [1, 5])
@pytest.mark.parametrize('scheduled', [False, True])
def test_lazy_matches_eager_15_updates_and_pure_reads(stride, scheduled):
    state = LazySignedHash(43)
    eager = torch.zeros(43)
    gen = torch.Generator().manual_seed(39)
    old_snapshots = []
    u = 0
    for round_idx in range(1, 15 * stride + 1):
        if round_idx % stride:
            assert state.u == u
            continue
        indices = torch.randint(43, (11,), generator=gen)
        grad = torch.randn(11, generator=gen)
        lr = .2 / (1 + u / 5) if scheduled else .2
        full_grad = torch.zeros(43).scatter_add_(0, indices, grad)
        eager = proximal_l1(eager - lr * full_grad, lr * .07)
        snapshot = state.snapshot()
        before = snapshot.read(torch.arange(43)).clone()
        before_values, before_last = state.values.clone(), state.last_u.clone()
        state.read(indices)
        assert torch.equal(before_values, state.values)
        assert torch.equal(before_last, state.last_u)
        state.step(indices, grad, lr, .07)
        u += 1
        assert state.u == u
        torch.testing.assert_close(state.read(torch.arange(43)), eager, atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(snapshot.read(torch.arange(43)), before, atol=0, rtol=0)
        old_snapshots.append((snapshot, before))
    for snapshot, before in old_snapshots:
        assert torch.equal(snapshot.read(torch.arange(43)), before)
    assert state.last_u.dtype == torch.int32
    assert state.values.numel() == 43


def test_online_snapshot_never_clones_capacity_sized_values():
    state = LazySignedHash(2**20)
    state.step(torch.tensor([1, 5]), torch.tensor([1., -1.]), .1, .01)
    snapshot = state.snapshot()
    assert snapshot.values.numel() == 2
    assert snapshot.snapshot_bytes < 1024
