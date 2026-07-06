"""
Reference tests for get_pdist_inds, which converts a pair of point indices
(i, j) into the corresponding index in a scipy.spatial.distance.pdist-style
condensed distance array (upper triangle, row-major, no diagonal). Used in
Tree.reorderChildrenRoot (bonsai_treeHelpers.py) to look up precomputed
pairwise distances between children when picking an ordering/MST.

NOTE: there is no check that i != j. Calling get_pdist_inds(shape, i, i)
does not raise -- it silently falls through to the "i >= j" branch and
returns a plausible-looking index that collides with some *other*, unrelated
valid pair's index (e.g. get_pdist_inds(5, 2, 2) == get_pdist_inds(5, 1, 4)
== 6). Current call sites happen to never pass i == j, but we test with 
test_diagonal_call_silently_collides_with_an_unrelated_pair.
"""
import numpy as np
import pytest
from scipy.spatial import distance

from bonsai.bonsai_helpers import get_pdist_inds


@pytest.mark.parametrize("n", [3, 4, 5, 8, 12])
def test_matches_scipy_pdist_ordering(n):
    rng = np.random.default_rng(n)
    points = rng.normal(size=(n, 4))
    condensed = distance.pdist(points)

    for i in range(n):
        for j in range(i + 1, n):
            idx = get_pdist_inds(n, i, j)
            assert condensed[idx] == pytest.approx(np.linalg.norm(points[i] - points[j]))


@pytest.mark.parametrize("n", [3, 4, 5, 8])
def test_is_symmetric_in_i_and_j(n):
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            assert get_pdist_inds(n, i, j) == get_pdist_inds(n, j, i)


@pytest.mark.parametrize("n", [3, 4, 5, 8, 12])
def test_covers_every_condensed_index_exactly_once(n):
    seen = sorted(get_pdist_inds(n, i, j) for i in range(n) for j in range(i + 1, n))
    expected_len = n * (n - 1) // 2
    assert seen == list(range(expected_len))


def test_diagonal_call_silently_collides_with_an_unrelated_pair():
    n = 5
    diagonal_idx = get_pdist_inds(n, 2, 2)
    colliding_pair_idx = get_pdist_inds(n, 1, 4)
    assert diagonal_idx == colliding_pair_idx
