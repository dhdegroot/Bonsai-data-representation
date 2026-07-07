"""
Reference tests for optimising all of a star tree's diffusion times at once:

    optimiseTStar          -- EM-like iterative optimiser (with a gradient-based fallback)
    logLGradStarTreeLogT   -- log-likelihood + gradient in log(t) space (the fallback objective)
    getLoglikAndGradStarTree -- the underlying log-likelihood + gradient (used everywhere above)

optimiseTStar first tries an EM-style scheme (alternately recomputing the
star's centre xr_g and each child's own optimal diffusion time via
findOptimalTSingleCell). If that fails to converge it falls back to jointly
optimising all times at once in log-space via logLGradStarTreeLogT. Both
should reach the same log-likelihood, and that log-likelihood should match
getLoglikAndGradStarTree computed independently from the returned times.
"""
import numpy as np
import pytest

import bonsai.bonsai_globals as bs_glob
from bonsai.bonsai_treeHelpers import getLoglikAndGradStarTree, logLGradStarTreeLogT, optimiseTStar


@pytest.fixture(autouse=True)
def _set_nGenes():
    # optimiseTStar's verbose branch divides by bs_glob.nGenes; harmless when
    # verbose=False (the default), but set it anyway so the fixture matches
    # what production code guarantees at this point in the pipeline.
    old = bs_glob.nGenes
    bs_glob.nGenes = None
    yield
    bs_glob.nGenes = old


def test_identical_children_optimize_to_zero_time():
    n_genes, n_children = 5, 4
    ltqs = np.zeros((n_genes, n_children))
    ltqsVars = np.full((n_genes, n_children), 0.5)

    t_i, loglik, W_g, xr_g = optimiseTStar(ltqs, ltqsVars)

    assert t_i == pytest.approx(np.zeros(n_children), abs=1e-6)
    assert xr_g == pytest.approx(np.zeros(n_genes), abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_optimum_satisfies_first_order_condition(seed):
    # At an interior optimum (t_i > 0), the gradient of the log-likelihood
    # w.r.t. log(t_i) must vanish. At the boundary (t_i == 0, which is a
    # legal constrained optimum here) the gradient need not be zero, so only
    # interior times are checked.
    rng = np.random.default_rng(seed)
    n_genes, n_children = 6, 4
    ltqs = rng.normal(size=(n_genes, n_children))
    ltqsVars = rng.uniform(0.2, 1.5, (n_genes, n_children))

    t_i, loglik, W_g, xr_g = optimiseTStar(ltqs, ltqsVars)

    neg_loglik_at_opt, neg_grad_at_opt = logLGradStarTreeLogT(np.log(np.maximum(t_i, 1e-8)), ltqsVars, ltqs)
    assert -neg_loglik_at_opt == pytest.approx(loglik, abs=1e-4)

    interior = t_i > 1e-6
    assert neg_grad_at_opt[interior] == pytest.approx(np.zeros(interior.sum()), abs=1e-3)


@pytest.mark.parametrize("seed", range(6))
def test_returned_loglik_matches_independent_recomputation(seed):
    rng = np.random.default_rng(seed)
    n_genes, n_children = 6, 4
    ltqs = rng.normal(size=(n_genes, n_children))
    ltqsVars = rng.uniform(0.2, 1.5, (n_genes, n_children))

    t_i, loglik, W_g, xr_g = optimiseTStar(ltqs, ltqsVars)

    loglik_direct = getLoglikAndGradStarTree(ltqs_gi=ltqs, xr_g=xr_g, wbar_gi=1 / (ltqsVars + t_i), W_g=W_g)
    assert loglik == pytest.approx(loglik_direct, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_getLoglikAndGradStarTree_gradient_matches_finite_difference(seed):
    rng = np.random.default_rng(seed)
    n_genes, n_children = 5, 3
    ltqs = rng.normal(size=(n_genes, n_children))
    ltqsVars = rng.uniform(0.2, 1.5, (n_genes, n_children))
    t_i = rng.uniform(0.1, 2.0, n_children)

    _, grad = getLoglikAndGradStarTree(ltqs_gi=ltqs, ltqsVars_gi=ltqsVars, t_i=t_i, returnGrad=True)

    dx = 1e-6
    numeric_grad = np.zeros(n_children)
    for i in range(n_children):
        t_plus, t_minus = t_i.copy(), t_i.copy()
        t_plus[i] += dx
        t_minus[i] -= dx
        loglik_plus = getLoglikAndGradStarTree(ltqs_gi=ltqs, ltqsVars_gi=ltqsVars, t_i=t_plus)
        loglik_minus = getLoglikAndGradStarTree(ltqs_gi=ltqs, ltqsVars_gi=ltqsVars, t_i=t_minus)
        numeric_grad[i] = (loglik_plus - loglik_minus) / (2 * dx)

    assert grad == pytest.approx(numeric_grad, abs=1e-4)
