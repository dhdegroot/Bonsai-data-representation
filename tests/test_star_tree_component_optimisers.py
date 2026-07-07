"""
Standalone tests for the lower-level building blocks that optimiseTStar and
calcSingleDLogL are built from. These were previously only exercised
indirectly (as dependencies inside those higher-level functions' tests);
here each is isolated and checked against an analytic or independently
computed reference.

    findOptimalTSingleCell / derLogLikStar
        One EM step of optimiseTStar: given a star's current centre xr_g and
        total precision W_g (which already includes this leaf's own current
        contribution), find this single leaf's optimal diffusion time.

    optimiseT3LeafStar / optimiseT3LeafStarSequential
        The two specialised 3-leaf optimisers used inside calcSingleDLogL.
        optimiseT3LeafStar jointly optimises all 3 branch lengths;
        optimiseT3LeafStarSequential fixes the two-leaf branch length first
        (via getOptTime2LeafTree) and only searches the split + ancestor
        edge. Both should be consistent with the general n-leaf optimiseTStar
        specialised to n=3, and (per the gotcha already pinned down in
        test_pairwise_merge_dlogl.py) the sequential path is a lower bound on
        the direct one, not an equivalent alternative.
"""
import numpy as np
import pytest

import bonsai.bonsai_globals as bs_glob
from bonsai.bonsai_treeHelpers import (
    derLogLikStar,
    findOptimalTSingleCell,
    getOptTime2LeafTree,
    optimiseT3LeafStar,
    optimiseT3LeafStarSequential,
    optimiseTStar,
)


@pytest.mark.parametrize("seed", range(6))
def test_findOptimalTSingleCell_derivative_vanishes_at_the_optimum(seed):
    rng = np.random.default_rng(seed)
    n_genes = 6
    ltqs_g = rng.normal(size=n_genes)
    xr_g = rng.normal(size=n_genes)
    ltqsVars_g = rng.uniform(0.2, 1.5, n_genes)
    W_g = rng.uniform(1.0, 5.0, n_genes)

    t_opt = findOptimalTSingleCell(ltqs_g, ltqsVars_g, xr_g, 1.0, W_g)

    sq_dists_g = (xr_g - ltqs_g) ** 2
    assert derLogLikStar(t_opt, ltqsVars_g, sq_dists_g, W_g) == pytest.approx(0.0, abs=1e-3)


def test_findOptimalTSingleCell_matches_analytic_solution_for_identical_genes():
    # For identical genes (same var, same squared distance to the star
    # centre), the root of derLogLikStar reduces to a single-gene closed
    # form: t = sqDist - var + 1/W.
    n_genes, var, sqdist, W = 6, 0.4, 5.0, 3.0
    ltqs_g = np.zeros(n_genes)
    xr_g = np.full(n_genes, np.sqrt(sqdist))
    ltqsVars_g = np.full(n_genes, var)
    W_g = np.full(n_genes, W)

    t_opt = findOptimalTSingleCell(ltqs_g, ltqsVars_g, xr_g, 1.0, W_g)

    assert t_opt == pytest.approx(sqdist - var + 1 / W, abs=1e-3)


def test_findOptimalTSingleCell_clamps_to_zero_when_already_well_explained():
    t_opt = findOptimalTSingleCell(np.zeros(3), np.full(3, 0.5), np.zeros(3), 1.0, np.full(3, 10.0))
    assert t_opt == 0


@pytest.fixture(autouse=True)
def _set_nGenes():
    old = bs_glob.nGenes
    bs_glob.nGenes = 5
    yield
    bs_glob.nGenes = old


@pytest.mark.parametrize("seed", range(6))
def test_optimiseT3LeafStar_matches_general_optimiseTStar_for_3_leaves(seed):
    rng = np.random.default_rng(seed)
    n_genes = 5
    ltqs_gi = rng.normal(size=(n_genes, 3))
    ltqsVars_gi = rng.uniform(0.2, 1.5, (n_genes, 3))

    loglik_3leaf, times_3leaf, success = optimiseT3LeafStar(ltqs_gi, ltqsVars_gi, np.array([0.5, 0.5, 0.5]))
    t_star, loglik_star, _, _ = optimiseTStar(ltqs_gi, ltqsVars_gi)

    assert success
    assert loglik_3leaf == pytest.approx(loglik_star, abs=1e-3)
    assert sorted(times_3leaf) == pytest.approx(sorted(t_star), abs=1e-2)


def test_optimiseT3LeafStar_identical_leaves_give_zero_times():
    n_genes = 5
    ltqs = np.zeros((n_genes, 3))
    ltqsVars = np.full((n_genes, 3), 0.5)

    loglik, times, success = optimiseT3LeafStar(ltqs, ltqsVars, np.array([0.5, 0.5, 0.5]))

    assert success
    assert times == pytest.approx(np.zeros(3), abs=1e-3)


@pytest.mark.parametrize("seed", range(6))
def test_sequential_t12_matches_two_leaf_optimiser(seed):
    rng = np.random.default_rng(seed)
    n_genes = 5
    ltqs_gi = rng.normal(size=(n_genes, 3))
    ltqsVars_gi = rng.uniform(0.2, 1.5, (n_genes, 3))

    _, _, t12Opt, success = optimiseT3LeafStarSequential(ltqs_gi, ltqsVars_gi, np.array([0.5, 0.5]), tol=1e-8)
    expected_t12, expected_converged = getOptTime2LeafTree(ltqs_gi[:, 0], ltqsVars_gi[:, 0],
                                                             ltqs_gi[:, 1], ltqsVars_gi[:, 1], tol=1e-8)

    assert success and expected_converged
    assert t12Opt == pytest.approx(expected_t12)


@pytest.mark.parametrize("seed", range(6))
def test_sequential_path_is_a_lower_bound_of_the_direct_optimiser(seed):
    # Same behavioural note as calcSingleDLogL (which is built on exactly
    # these two functions): fixing t12 up front restricts the search space,
    # so optimiseT3LeafStarSequential's log-likelihood should never exceed
    # optimiseT3LeafStar's, and in general falls strictly below it.
    rng = np.random.default_rng(seed)
    n_genes = 5
    ltqs_gi = rng.normal(size=(n_genes, 3))
    ltqsVars_gi = rng.uniform(0.2, 1.5, (n_genes, 3))
    t0_i = np.array([0.5, 0.5, 0.5])

    loglik_direct, _, success_direct = optimiseT3LeafStar(ltqs_gi, ltqsVars_gi, t0_i)
    loglik_seq, _, _, success_seq = optimiseT3LeafStarSequential(ltqs_gi, ltqsVars_gi, t0_i[1:], tol=1e-8)

    assert success_direct and success_seq
    assert loglik_direct >= loglik_seq - 1e-4
