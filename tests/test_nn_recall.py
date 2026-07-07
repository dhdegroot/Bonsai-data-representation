"""
Reference tests for getFracCorrectNNs, which scores how well an approximate
nearest-neighbour search (nns) recovers a reference/brute-force neighbour set
(true_nns). Both arguments are (cells x k) index matrices, matching the
documented return shape of getApproxNNs/getNNssklearn. Only exercised today
from the `if __name__ == '__main__'` benchmarking block at the bottom of
bonsai_approxNN.py (used to compare candidate NN backends), not from the main
tree-building pipeline.
"""
import numpy as np
import pytest

from bonsai.bonsai_approxNN import getFracCorrectNNs


def random_neighbour_matrix(rng, n_cells, k, pool_size):
    return np.array([rng.permutation(pool_size)[:k] for _ in range(n_cells)])


def manual_recall(nns, true_nns):
    total = sum(len(set(nns[r]) & set(true_nns[r])) for r in range(nns.shape[0]))
    return total / np.prod(nns.shape)


def test_identical_arrays_give_perfect_score():
    rng = np.random.default_rng(0)
    true_nns = random_neighbour_matrix(rng, n_cells=6, k=4, pool_size=20)

    assert getFracCorrectNNs(true_nns.copy(), true_nns) == pytest.approx(1.0)


def test_completely_disjoint_neighbours_give_zero():
    n_cells, k = 5, 4
    true_nns = np.arange(n_cells * k).reshape(n_cells, k)
    pred_nns = np.arange(n_cells * k, 2 * n_cells * k).reshape(n_cells, k)

    assert getFracCorrectNNs(pred_nns, true_nns) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(6))
def test_matches_manual_set_based_recall(seed):
    rng = np.random.default_rng(seed)
    n_cells, k = 8, 5
    true_nns = random_neighbour_matrix(rng, n_cells, k, pool_size=30)
    pred_nns = true_nns.copy()
    # Corrupt a random subset of predictions so the score is neither 0 nor 1.
    for row in rng.choice(n_cells, size=n_cells // 2, replace=False):
        n_swap = rng.integers(1, k + 1)
        pred_nns[row, :n_swap] = rng.permutation(100)[:n_swap] + 1000  # guaranteed not in true_nns's pool

    assert getFracCorrectNNs(pred_nns, true_nns) == pytest.approx(manual_recall(pred_nns, true_nns))


@pytest.mark.parametrize("seed", range(5))
def test_is_symmetric_in_its_two_arguments(seed):
    # |set(a) ∩ set(b)| is symmetric, and both arrays have the same shape, so
    # swapping which one is "true" and which is "predicted" doesn't change
    # the score -- this function is really just a shared intersection-size
    # metric, not a directional recall/precision distinction.
    rng = np.random.default_rng(seed)
    n_cells, k = 6, 4
    a = random_neighbour_matrix(rng, n_cells, k, pool_size=20)
    b = random_neighbour_matrix(rng, n_cells, k, pool_size=20)

    assert getFracCorrectNNs(a, b) == pytest.approx(getFracCorrectNNs(b, a))
