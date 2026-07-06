"""
Reference tests for the two-leaf branch-length optimisation used when Bonsai
decides how much diffusion time separates a candidate pair of nodes:

    der2LeafTree       -- derivative of the two-leaf log-likelihood w.r.t. t12
    getOptTime2LeafTree -- finds the t12 that maximises that log-likelihood
    getLoglik2LeafTree  -- the log-likelihood itself, given a fixed t12

For two leaves with per-gene measurements (ltqs) and measurement variances
(ltqsVars) separated by a branch of total diffusion time t12, the model is
Gaussian: totalVar_g = ltqsVars1_g + ltqsVars2_g + t12, and the log-likelihood
(up to an additive constant) is -sum(log(totalVar_g) + sqDist_g / totalVar_g).
"""
import numpy as np
import pytest

from bonsai.bonsai_treeHelpers import der2LeafTree, getLoglik2LeafTree, getOptTime2LeafTree


def test_identical_leaves_give_zero_optimal_time():
    ltqs = np.array([1.0, 2.0, -3.0])
    var = np.array([0.5, 0.2, 1.0])

    t_opt, converged = getOptTime2LeafTree(ltqs, var, ltqs.copy(), var.copy())

    assert converged
    assert t_opt == 0


def test_analytic_optimum_for_equal_genes():
    # When every gene has the same squared distance d and the same summed
    # variance v, the root of der2LeafTree collapses to the single-gene
    # closed form: der = (-1 + d / (t + v)) / (t + v) = 0  =>  t = d - v.
    n_genes = 8
    d, v = 9.0, 1.5
    ltqs1 = np.zeros(n_genes)
    ltqs2 = np.full(n_genes, np.sqrt(d))
    var_each = np.full(n_genes, v / 2)  # split evenly so ltqsVars1 + ltqsVars2 == v

    t_opt, converged = getOptTime2LeafTree(ltqs1, var_each, ltqs2, var_each)

    assert converged
    assert t_opt == pytest.approx(d - v, abs=1e-4)


def test_optimum_clamps_to_zero_when_variance_exceeds_distance():
    # If the measurement noise alone can explain the observed distance, the
    # derivative at t=0 is already <= 0, so the constrained optimum is 0.
    n_genes = 4
    ltqs1 = np.zeros(n_genes)
    ltqs2 = np.full(n_genes, 0.1)
    var_each = np.full(n_genes, 5.0)

    t_opt, converged = getOptTime2LeafTree(ltqs1, var_each, ltqs2, var_each)

    assert converged
    assert t_opt == 0
    assert der2LeafTree(0.0, var_each + var_each, (ltqs1 - ltqs2) ** 2) <= 0


@pytest.mark.parametrize("seed", range(5))
def test_der2LeafTree_matches_finite_difference_of_loglik(seed):
    # der2LeafTree is meant to be exactly d(loglik)/d(t12). Cross-check the
    # analytic derivative against a central finite difference at several
    # random, non-optimal points so a future port can be checked the same way.
    rng = np.random.default_rng(seed)
    n_genes = 10
    ltqs1 = rng.normal(size=n_genes)
    ltqs2 = rng.normal(size=n_genes)
    var1 = rng.uniform(0.1, 2.0, n_genes)
    var2 = rng.uniform(0.1, 2.0, n_genes)
    summed_vars = var1 + var2
    sq_dists = (ltqs1 - ltqs2) ** 2
    t12 = rng.uniform(0.1, 5.0)

    analytic = der2LeafTree(t12, summed_vars, sq_dists)

    dx = 1e-6
    loglik_plus = getLoglik2LeafTree(ltqs1, var1, ltqs2, var2, t12 + dx)
    loglik_minus = getLoglik2LeafTree(ltqs1, var1, ltqs2, var2, t12 - dx)
    numeric = (loglik_plus - loglik_minus) / (2 * dx)

    assert analytic == pytest.approx(numeric, abs=1e-4)


@pytest.mark.parametrize("seed", range(5))
def test_getOptTime2LeafTree_is_a_local_maximum_of_the_loglik(seed):
    rng = np.random.default_rng(seed)
    n_genes = 10
    ltqs1 = rng.normal(size=n_genes)
    ltqs2 = rng.normal(size=n_genes)
    var1 = rng.uniform(0.1, 2.0, n_genes)
    var2 = rng.uniform(0.1, 2.0, n_genes)

    t_opt, converged = getOptTime2LeafTree(ltqs1, var1, ltqs2, var2)
    assert converged

    loglik_opt = getLoglik2LeafTree(ltqs1, var1, ltqs2, var2, t_opt)
    eps = 1e-3
    loglik_plus = getLoglik2LeafTree(ltqs1, var1, ltqs2, var2, t_opt + eps)
    # Only probe below the optimum when it is interior (t_opt can be clamped to 0).
    if t_opt > eps:
        loglik_minus = getLoglik2LeafTree(ltqs1, var1, ltqs2, var2, t_opt - eps)
        assert loglik_opt >= loglik_minus
    assert loglik_opt >= loglik_plus
