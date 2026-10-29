"""
downstream_analyses/average_over_groups.py (area D): the group mean and group variance of a feature given
per-object estimates with error bars (README "Optional downstream analyses", "Averaging over groups").

The model, from the README and the function: object i of a group has estimate x_i with error variance v_i, and the
true values scatter around the group mean mu with variance tau^2, so x_i ~ N(mu, v_i + tau^2). tau is the
maximum-likelihood value with mu profiled out, mu the inverse-variance weighted mean at that tau and the reported
error on mu sqrt(1 / sum_i 1 / (v_i + tau^2)). The reference below fits the same model with scipy.stats.norm and a
bounded scalar minimizer, independently of fminbound.
"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from downstream_analyses.average_over_groups import find_tau, log_likelihood, run_averaging


def reference_fit(x, v):
	"""(tau, mu, error on mu) maximizing prod_i N(x_i; mu, v_i + tau^2) over mu and tau >= 0."""

	def profile_nll(tau):
		weights = 1.0 / (v + tau ** 2)
		mu = np.sum(weights * x) / np.sum(weights)
		return -np.sum(stats.norm.logpdf(x, loc=mu, scale=np.sqrt(v + tau ** 2)))

	grid = np.linspace(0.0, 20.0, 801)
	start = grid[int(np.argmin([profile_nll(tau) for tau in grid]))]
	low, high = max(0.0, start - 0.03), start + 0.03
	tau = optimize.minimize_scalar(profile_nll, bounds=(low, high), method='bounded',
	                               options={'xatol': 1e-10}).x
	if profile_nll(0.0) <= profile_nll(tau):
		tau = 0.0
	weights = 1.0 / (v + tau ** 2)
	return tau, np.sum(weights * x) / np.sum(weights), math.sqrt(1.0 / np.sum(weights))


def test_log_likelihood_is_the_profiled_gaussian_negative_log_likelihood():
	"""log_likelihood(t) = -sum_i log N(x_i; mu_hat(t), v_i + t^2) - n/2 log(2 pi), with mu_hat the weighted mean."""
	rng = np.random.default_rng(0)
	x = rng.normal(1.0, 2.0, 12)
	v = rng.uniform(0.1, 1.0, 12)
	for t in (0.0, 0.3, 1.7, 5.0):
		weights = 1.0 / (v + t * t)
		mu = np.sum(weights * x) / np.sum(weights)
		want = -np.sum(stats.norm.logpdf(x, loc=mu, scale=np.sqrt(v + t * t))) - 0.5 * len(x) * math.log(2 * math.pi)
		assert log_likelihood(t, x, v) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize('spread', [3.0, 0.8])
def test_find_tau_is_the_maximum_likelihood_group_variance(spread):
	"""A group whose spread is larger than its error bars: tau is the maximum-likelihood scatter (fminbound's
	xtol is 1e-6, so tau agrees to 1e-5), mu the weighted mean and the error on mu the inverse-variance formula."""
	rng = np.random.default_rng(1)
	v = rng.uniform(0.05, 0.4, 25)
	x = 2.0 + rng.normal(0.0, spread, 25) + rng.normal(0.0, np.sqrt(v))
	tau, mu, sig = find_tau(x, v)
	want_tau, want_mu, want_sig = reference_fit(x, v)
	assert want_tau > 0.3
	assert tau == pytest.approx(want_tau, abs=1e-5)
	# mu and sig follow from tau in closed form; the tau tolerance carries over at first order.
	weights = 1.0 / (v + tau ** 2)
	assert mu == pytest.approx(np.sum(weights * x) / np.sum(weights), rel=1e-12)
	assert sig == pytest.approx(math.sqrt(1.0 / np.sum(weights)), rel=1e-12)
	assert mu == pytest.approx(want_mu, abs=1e-4)
	assert sig == pytest.approx(want_sig, rel=1e-4)


def test_find_tau_without_scatter_gives_the_weighted_mean():
	"""Estimates that agree within their error bars: tau goes to zero and mu is the plain inverse-variance
	weighted mean with error sqrt(1 / sum 1 / v_i)."""
	x = np.array([1.00, 1.02, 0.99, 1.01, 1.00])
	v = np.array([0.5, 0.4, 0.6, 0.5, 0.45])
	tau, mu, sig = find_tau(x, v)
	assert reference_fit(x, v)[0] == 0.0
	assert tau == pytest.approx(0.0, abs=1e-4)
	assert mu == pytest.approx(np.sum(x / v) / np.sum(1 / v), rel=1e-9)
	assert sig == pytest.approx(math.sqrt(1 / np.sum(1 / v)), rel=1e-8)


def test_run_averaging_tables():
	"""run_averaging fits every (group, feature): tables indexed by the sorted group names with the feature names
	as columns; significance per feature is sqrt(mean over groups of (mu / error)^2), sorted from high to low. The
	input error bars are standard deviations (squared inside)."""
	rng = np.random.default_rng(2)
	n_cells, n_features = 30, 3
	activities = rng.normal(0.0, 1.0, (n_cells, n_features))
	activities[:12, 1] += 3.0
	deltas = rng.uniform(0.2, 0.6, (n_cells, n_features))
	groups = {'g2': list(range(0, 12)), 'g1': list(range(12, 25)), 'g0': [25, 26, 27, 28, 29]}
	features = ['f0', 'f1', 'f2']
	avg, err, significance = run_averaging(activities, deltas, groups, features)
	assert list(avg.index) == ['g0', 'g1', 'g2'] and list(err.index) == ['g0', 'g1', 'g2']
	assert list(avg.columns) == features and list(err.columns) == features
	for group, cells in groups.items():
		for col, feature in enumerate(features):
			_, want_mu, want_sig = reference_fit(activities[cells, col], deltas[cells, col] ** 2)
			assert avg.loc[group, feature] == pytest.approx(want_mu, abs=1e-4)
			assert err.loc[group, feature] == pytest.approx(want_sig, rel=1e-4)
	want = np.sqrt(((avg / err) ** 2).mean(axis=0))
	assert list(significance.index) == list(want.sort_values(ascending=False).index)
	assert significance.index[0] == 'f1'
	np.testing.assert_allclose(significance.values, want.sort_values(ascending=False).values, rtol=1e-12)
	assert isinstance(significance, pd.Series)
