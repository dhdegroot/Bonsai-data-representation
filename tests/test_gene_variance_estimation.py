"""
Reference tests for estimating a gene's "true" biological variance from
repeated noisy measurements (each with its own known measurement-error
variance) -- a per-cell-type/per-batch mean estimate plus its variance,
regressed onto a shared underlying variance v.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from bonsai.bonsai_dataprocessing import (
    infer_true_var_log,
    loglik_given_true_var_log,
    neg_loglik_grad_using_measurements_and_errors,
)


@pytest.mark.parametrize("seed", range(6))
def test_gradient_matches_finite_difference(seed):
    rng = np.random.default_rng(seed)
    n = 12
    measurements = rng.normal(size=n)
    variances = rng.uniform(0.1, 1.0, n)
    logv = rng.uniform(-1.0, 1.0)

    _, grad = neg_loglik_grad_using_measurements_and_errors(logv, measurements, variances)

    dx = 1e-6
    f_plus, _ = neg_loglik_grad_using_measurements_and_errors(logv + dx, measurements, variances)
    f_minus, _ = neg_loglik_grad_using_measurements_and_errors(logv - dx, measurements, variances)
    numeric_grad = (f_plus - f_minus) / (2 * dx)

    assert grad == pytest.approx(numeric_grad, abs=1e-4)


def test_mean_ml_is_the_precision_weighted_average():
    rng = np.random.default_rng(1)
    n = 10
    measurements = rng.normal(size=n)
    variances = rng.uniform(0.1, 1.0, n)
    logv = 0.4

    _, _, mean_ml = neg_loglik_grad_using_measurements_and_errors(logv, measurements, variances,
                                                                   return_mean_ML=True)

    v = np.exp(logv)
    precision = 1 / (v + variances)
    expected = np.sum(precision * measurements) / np.sum(precision)
    assert mean_ml == pytest.approx(expected)


def test_mean_ml_reduces_to_plain_mean_when_variances_are_equal():
    rng = np.random.default_rng(2)
    n = 8
    measurements = rng.normal(size=n)
    variances = np.full(n, 0.3)

    _, _, mean_ml = neg_loglik_grad_using_measurements_and_errors(0.0, measurements, variances,
                                                                   return_mean_ML=True)

    assert mean_ml == pytest.approx(np.mean(measurements))


@pytest.mark.parametrize("seed", range(5))
def test_dead_reference_differs_from_live_by_the_documented_missing_term(seed):
    rng = np.random.default_rng(seed)
    n = 10
    measurements = rng.normal(size=n)
    variances = rng.uniform(0.1, 1.0, n)
    logv = rng.uniform(-1.0, 1.0)

    neg_loglik, _ = neg_loglik_grad_using_measurements_and_errors(logv, measurements, variances)
    dead_val = loglik_given_true_var_log(logv, measurements, variances)

    total_precision = np.sum(1 / (np.exp(logv) + variances))
    assert dead_val == pytest.approx(neg_loglik + 0.5 * np.log(total_precision))


def test_mle_of_v_differs_between_the_two_objectives():
    # Same data, same statistical target (the shared variance v), two
    # implementations -> materially different point estimates. This is the
    # behavioural consequence of the missing term above, demonstrated on a
    # single fixed, seeded dataset so the discrepancy size stays reproducible.
    rng = np.random.default_rng(0)
    n = 20
    true_mu, true_v = 2.0, 1.5
    variances = rng.uniform(0.1, 1.0, n)
    measurements = true_mu + rng.normal(scale=np.sqrt(true_v + variances))

    opt_live = minimize(neg_loglik_grad_using_measurements_and_errors, x0=0.0, jac=True,
                         args=(measurements, variances, False))
    v_live = np.exp(opt_live.x[0])
    v_dead = infer_true_var_log(measurements[None, :], variances[None, :])[0]

    assert v_live != pytest.approx(v_dead, rel=0.05)


def test_infer_true_var_log_matches_grid_search_per_gene():
    rng = np.random.default_rng(3)
    n_cells = 15
    # Two independent "genes" with different true variances.
    measurements = rng.normal(size=(2, n_cells))
    variances = rng.uniform(0.1, 1.0, (2, n_cells))

    fitted = infer_true_var_log(measurements, variances)

    logv_grid = np.linspace(-8, 8, 4001)
    for gene_ind in range(2):
        grid_vals = [loglik_given_true_var_log(lv, measurements[gene_ind], variances[gene_ind])
                     for lv in logv_grid]
        expected_v = np.exp(logv_grid[np.argmin(grid_vals)])
        assert fitted[gene_ind] == pytest.approx(expected_v, rel=1e-2)

    # The two genes were fit independently, so their estimates need not match.
    assert fitted[0] != pytest.approx(fitted[1], rel=1e-2)
