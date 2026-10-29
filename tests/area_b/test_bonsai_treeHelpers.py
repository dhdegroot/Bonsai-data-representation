"""
Tree and TreeNode of bonsai/bonsai_treeHelpers.py and its module-level functions (area B).

Expected values come from the dense multivariate-Gaussian reference likelihood and posteriors of
tests/bonsai_testref.py (log-likelihoods in Bonsai's units: twice the natural log-likelihood plus
(C - 1) G log 2 pi), from central differences of that reference, from scipy optimisations of it, from
brute-force enumeration of topologies and from small trees worked by hand.
"""
import itertools
import json
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

import bonsai.bonsai_globals as bs_glob
import bonsai.bonsai_treeHelpers as th
from bonsai.bonsai_treeHelpers import Tree, TreeNode

from area_b.helpers_b import (REPO_ROOT, bonsai_nodes, build, central_difference, edges_of, leaf_data,
                              leaf_pairs_distances, matrices, optimise_lengths, reference_of, star_of, to_node,
                              tree_data, unrooted_topologies, with_lengths, SimpleSCData)
from bonsai_testdata import balanced_newick, caterpillar_newick
from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testref import build_bonsai_tree, reference_bonsai_loglik, reference_posteriors, star_tree
from bonsai_testrun import bonsai_env, mpiexec_path, python_has_mpi4py
from bonsai_testtrees import Node, clades, parse_newick, path_length, rf_distance, splits, write_newick

SMALL_TREE = '(((A:0.3,B:0.7)i:0.4,C:0.2)j:0.1,(D:0.5,E:0.05,F:0.2)k:0.3,G:0.4)r;'


def _leaf_data(tree, n_genes, seed):
	rng = np.random.default_rng(seed)
	means = {name: rng.normal(0.0, 1.0, n_genes) for name in tree.leaf_names()}
	variances = {name: rng.uniform(0.01, 0.5, n_genes) for name in tree.leaf_names()}
	return means, variances


def _star_inputs(n_cells, n_genes, seed):
	rng = np.random.default_rng(seed)
	ltqs = rng.normal(size=(n_genes, n_cells))
	ltqs_vars = rng.uniform(0.05, 0.5, (n_genes, n_cells))
	times = rng.uniform(0.1, 1.0, n_cells)
	names = ['L%d' % ind for ind in range(n_cells)]
	means = {name: ltqs[:, ind] for ind, name in enumerate(names)}
	variances = {name: ltqs_vars[:, ind] for ind, name in enumerate(names)}
	return ltqs, ltqs_vars, times, names, means, variances


def _star_ref(names, times, means, variances):
	return reference_bonsai_loglik(star_tree(names, times), means, variances)


@pytest.fixture
def genes_and_cells():
	"""Set bonsai_globals.nGenes / nCells for a test (restored by bonsai_globals_reset)."""

	def setter(n_genes, n_cells=None):
		bs_glob.nGenes = n_genes
		bs_glob.nCells = n_cells

	return setter


# ============================================================ log-likelihood of a tree

@pytest.mark.parametrize('mem_friendly', [True, False])
def test_calcLogLComplete_matches_the_dense_gaussian_reference(mem_friendly):
	"""A 7-leaf tree with a polytomy: Bonsai's log-likelihood equals the dense multivariate-normal likelihood of
	the leaves with a flat root prior, in Bonsai's units (twice the log-likelihood plus (C-1) G log 2 pi)."""
	tree = parse_newick(SMALL_TREE)
	means, variances = _leaf_data(tree, n_genes=4, seed=11)
	bonsai_tree = build_bonsai_tree(tree, means, variances)
	got = bonsai_tree.calcLogLComplete(mem_friendly=mem_friendly)
	assert got == pytest.approx(reference_bonsai_loglik(tree, means, variances), rel=1e-9)
	assert bonsai_tree.loglik == got


def test_calcLogLComplete_adds_loglikVarCorr():
	tree = parse_newick(SMALL_TREE)
	means, variances = _leaf_data(tree, n_genes=3, seed=12)
	correction = -6 * math.log(2.5)
	got = build_bonsai_tree(tree, means, variances).calcLogLComplete(mem_friendly=True, loglikVarCorr=correction)
	assert got == pytest.approx(reference_bonsai_loglik(tree, means, variances, loglik_var_corr=correction), rel=1e-9)


def test_calcLogLComplete_with_and_without_recalc():
	"""recalc=False on a tree whose messages are up to date gives the same log-likelihood; after a branch length
	changes, recalc=True gives the reference on the new lengths. (What recalc=False returns on stale messages is
	left open: a rewrite may recompute or cache.)"""
	tree = parse_newick(SMALL_TREE)
	means, variances = _leaf_data(tree, n_genes=3, seed=13)
	bonsai_tree = build_bonsai_tree(tree, means, variances)
	first = bonsai_tree.calcLogLComplete()
	assert bonsai_tree.calcLogLComplete(recalc=False) == pytest.approx(first, rel=1e-12)
	bonsai_tree.root.childNodes[0].tParent += 1.0
	assert bonsai_tree.calcLogLComplete(recalc=True) == pytest.approx(
		reference_of(bonsai_tree, means, variances), rel=1e-9)


def test_getLtqsComplete_sets_every_node_to_its_downstream_message():
	"""ltqs and 1/W at an internal node are the mean and variance of the node's position given only the cells below
	it (flat prior above): the root of the subtree's posterior in the reference."""
	tree = parse_newick(SMALL_TREE)
	means, variances = _leaf_data(tree, n_genes=3, seed=14)
	bonsai_tree = build_bonsai_tree(tree, means, variances)
	bonsai_tree.root.getLtqsComplete(mem_friendly=True)
	for node in bonsai_nodes(bonsai_tree.root):
		if node.isLeaf:
			continue
		subtree = to_node(node)
		post_means, post_vars = reference_posteriors(subtree, means, variances)
		np.testing.assert_allclose(node.ltqs, post_means[node.nodeId], rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(node.getLtqsVars(), post_vars[node.nodeId], rtol=1e-9)


# ===================================================== star trees, two- and three-leaf trees

@pytest.mark.parametrize('seed', range(3))
def test_getLoglikAndGradStarTree_array_input_matches_reference_and_its_derivative(seed):
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(5, 4, seed)
	loglik, grad = th.getLoglikAndGradStarTree(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times, returnGrad=True)
	assert loglik == pytest.approx(_star_ref(names, times, means, variances), rel=1e-10)
	expected = central_difference(lambda t: _star_ref(names, t, means, variances), times)
	np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-7)
	assert th.getLoglikAndGradStarTree(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times) == pytest.approx(loglik)


@pytest.mark.parametrize('mem_friendly', [True, False])
def test_getLoglikAndGradStarTree_child_nodes_input_gives_the_same_numbers(mem_friendly, genes_and_cells):
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(4, 3, seed=5)
	genes_and_cells(3)
	children = [TreeNode(nodeInd=ind, ltqs=ltqs[:, ind], ltqsVars=ltqs_vars[:, ind], tParent=times[ind], isLeaf=True)
	            for ind in range(4)]
	expected_loglik, expected_grad = th.getLoglikAndGradStarTree(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times,
	                                                             returnGrad=True)
	xr_g, W_g = th.findNodeLtqsGivenLeafs(childNodes=children)
	loglik, grad = th.getLoglikAndGradStarTree(childNodes=children, xr_g=xr_g, W_g=W_g, returnGrad=True,
	                                           mem_friendly=mem_friendly)
	assert loglik == pytest.approx(expected_loglik, rel=1e-12)
	np.testing.assert_allclose(grad, expected_grad, rtol=1e-10)
	loglik = th.getLoglikAndGradStarTree(childNodes=children, mem_friendly=True)
	assert loglik == pytest.approx(expected_loglik, rel=1e-12)


def test_star_wrappers_are_the_negated_loglik_and_chain_rule_gradients():
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(3, 4, seed=7)

	def ref(t):
		return _star_ref(names, t, means, variances)

	value, grad = th.getLogLikAndGradStarTreeWrapper(np.log(times), ltqs_vars, ltqs)
	assert value == pytest.approx(-ref(times), rel=1e-10)
	np.testing.assert_allclose(grad, central_difference(lambda lt: -ref(np.exp(lt)), np.log(times)), rtol=1e-5)
	value, grad = th.logLGradStarTreeLogT(np.log(times), ltqs_vars, ltqs)
	assert value == pytest.approx(-ref(times), rel=1e-10)
	np.testing.assert_allclose(grad, central_difference(lambda lt: -ref(np.exp(lt)), np.log(times)), rtol=1e-5)
	value, grad = th.getLogLikAndGradStarTreeWrapperOld(times, ltqs_vars, ltqs)
	assert value == pytest.approx(-ref(times), rel=1e-10)
	np.testing.assert_allclose(grad, central_difference(lambda t: -ref(t), times), rtol=1e-5)

	# Sequential parametrisation: (t_1a, log t_ar) with t_1a + t_2a = t12 held fixed.
	t12 = times[0] + times[1]
	params = np.array([times[0], np.log(times[2])])

	def seq(p):
		return -ref(np.array([p[0], t12 - p[0], np.exp(p[1])]))

	value, grad = th.getLogLikAndGradStarTreeSequentialWrapper(params, t12, ltqs_vars, ltqs)
	assert value == pytest.approx(seq(params), rel=1e-10)
	np.testing.assert_allclose(grad, central_difference(seq, params), rtol=1e-5)
	params_old = np.array([times[0], times[2]])

	def seq_old(p):
		return -ref(np.array([p[0], t12 - p[0], p[1]]))

	value, grad = th.getLogLikAndGradStarTreeSequentialWrapperOld(params_old, t12, ltqs_vars, ltqs)
	assert value == pytest.approx(seq_old(params_old), rel=1e-10)
	np.testing.assert_allclose(grad, central_difference(seq_old, params_old), rtol=1e-5)


def test_derLogLikStar_is_the_star_gradient_with_the_centre_held():
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(4, 5, seed=9)
	xr_g, W_g = th.findNodeLtqsGivenLeafs(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times)
	expected = central_difference(lambda t: _star_ref(names, t, means, variances), times)
	for ind in range(4):
		sq_dists = (xr_g - ltqs[:, ind]) ** 2
		assert th.derLogLikStar(times[ind], ltqs_vars[:, ind], sq_dists, W_g) == pytest.approx(expected[ind], rel=1e-5)


@pytest.mark.parametrize('seed', range(3))
def test_findNodeLtqsGivenLeafs_is_the_posterior_of_the_star_centre(seed, genes_and_cells):
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(4, 3, seed)
	post_means, post_vars = reference_posteriors(star_tree(names, times), means, variances)
	xr_g, W_g, wbar_gi, w_over_w = th.findNodeLtqsGivenLeafs(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times,
	                                                          return_wbar_gi=True, return_wOverW_gi=True)
	np.testing.assert_allclose(xr_g, post_means['root'], rtol=1e-10)
	np.testing.assert_allclose(1 / W_g, post_vars['root'], rtol=1e-10)
	np.testing.assert_allclose(wbar_gi, 1 / (ltqs_vars + times), rtol=1e-12)
	np.testing.assert_allclose(w_over_w.sum(axis=1), 1.0, rtol=1e-12)
	genes_and_cells(3)
	children = [TreeNode(nodeInd=ind, ltqs=ltqs[:, ind], ltqsVars=ltqs_vars[:, ind], tParent=times[ind], isLeaf=True)
	            for ind in range(4)]
	xr_nodes, W_nodes = th.findNodeLtqsGivenLeafs(childNodes=children)
	np.testing.assert_allclose(xr_nodes, xr_g, rtol=1e-12)
	np.testing.assert_allclose(W_nodes, W_g, rtol=1e-12)
	for gene in range(3):
		xr, W, wbar_i, w_over_w_i = th.findNodeLtqsGivenLeafsSingleGene(ltqs[gene], ltqs_vars[gene], times,
		                                                                return_wbar_i=True, return_wOverW_i=True)
		assert xr == pytest.approx(xr_g[gene], rel=1e-12) and W == pytest.approx(W_g[gene], rel=1e-12)
		np.testing.assert_allclose(wbar_i, wbar_gi[gene], rtol=1e-12)
		np.testing.assert_allclose(w_over_w_i, w_over_w[gene], rtol=1e-12)
		assert th.findNodeLtqsGivenLeafsSingleGene(ltqs[gene], ltqs_vars[gene], times)[0] == pytest.approx(xr)


def test_two_leaf_loglik_and_derivative_match_the_reference():
	rng = np.random.default_rng(3)
	x1, x2 = rng.normal(size=5), rng.normal(size=5)
	v1, v2 = rng.uniform(0.05, 0.3, 5), rng.uniform(0.05, 0.3, 5)
	means, variances = {'A': x1, 'B': x2}, {'A': v1, 'B': v2}

	def ref(t):
		return reference_bonsai_loglik(parse_newick('(A:%r,B:0.0)r;' % float(t)), means, variances)

	assert th.getLoglik2LeafTree(x1, v1, x2, v2, 0.0) == pytest.approx(ref(0.0), rel=1e-10)
	for t in (0.05, 0.4, 3.0):
		assert th.getLoglik2LeafTree(x1, v1, x2, v2, t) == pytest.approx(ref(t), rel=1e-10)
		assert th.der2LeafTree(t, v1 + v2, (x1 - x2) ** 2) == pytest.approx(
			central_difference(lambda s: ref(s[0]), np.array([t]))[0], rel=1e-5)


@pytest.mark.parametrize('seed', range(4))
def test_getOptTime2LeafTree_maximises_the_two_leaf_reference(seed):
	rng = np.random.default_rng(seed)
	x1, x2 = rng.normal(size=6), rng.normal(size=6) * 2
	v1, v2 = rng.uniform(0.05, 0.3, 6), rng.uniform(0.05, 0.3, 6)
	means, variances = {'A': x1, 'B': x2}, {'A': v1, 'B': v2}
	t_opt, converged = th.getOptTime2LeafTree(x1, v1, x2, v2, tol=1e-12)
	assert converged
	best = minimize_scalar(lambda t: -reference_bonsai_loglik(parse_newick('(A:%r,B:0.0)r;' % float(t)), means,
	                                                          variances),
	                       bounds=(0.0, 100.0), method='bounded', options={'xatol': 1e-10})
	assert t_opt == pytest.approx(best.x, rel=1e-5, abs=1e-7)


def test_getOptTime2LeafTree_gives_up_beyond_a_million():
	"""A distance that needs t > 1e6 is reported as not converged; get_dlogl_spr_move relies on the flag."""
	x1, x2, v = np.zeros(2), np.full(2, 1e4), np.full(2, 0.1)
	assert th.getOptTime2LeafTree(x1, v, x2, v) == (None, False)


@pytest.mark.parametrize('seed', range(3))
def test_optimiseTStar_returns_a_stationary_point_of_the_star_reference(seed):
	"""The EM scheme stops at a local maximum of the star likelihood (the star likelihood can have several, seed 2
	has two): the reference gradient vanishes on every positive branch length and points below zero at a zero one.
	The returned log-likelihood, centre and precision are the reference's at the returned times."""
	ltqs, ltqs_vars, _, names, means, variances = _star_inputs(5, 6, seed)
	ltqs[:, 0] = ltqs[:, 1]  # two identical children: their branch lengths can go to zero
	means['L0'] = ltqs[:, 0]
	t_i, loglik, W_g, xr_g = th.optimiseTStar(ltqs, ltqs_vars)
	assert loglik == pytest.approx(_star_ref(names, t_i, means, variances), rel=1e-10)
	assert loglik >= _star_ref(names, np.ones(5), means, variances)
	assert np.all(t_i >= 0)
	# EM stops when no time moves by more than 1e-4, so the gradient is small, not zero.
	for ind in range(5):
		step = np.zeros(5)
		step[ind] = 1e-7
		if t_i[ind] > 0:
			slope = (_star_ref(names, t_i + step, means, variances) -
			         _star_ref(names, t_i - step, means, variances)) / 2e-7
			assert slope == pytest.approx(0.0, abs=2e-2)
		else:
			slope = (_star_ref(names, t_i + step, means, variances) - loglik) / 1e-7
			assert slope < 2e-2
	post_means, post_vars = reference_posteriors(star_tree(names, t_i), means, variances)
	np.testing.assert_allclose(xr_g, post_means['root'], rtol=1e-9)
	np.testing.assert_allclose(W_g, 1 / post_vars['root'], rtol=1e-9)


def test_optimiseTStar_finds_the_maximum_of_a_well_separated_star():
	ltqs, ltqs_vars, _, names, means, variances = _star_inputs(5, 8, seed=0)
	t_i, loglik, _, _ = th.optimiseTStar(ltqs, ltqs_vars)
	best, lengths = optimise_lengths(star_tree(names, np.ones(5)), means, variances)
	assert loglik == pytest.approx(best, abs=1e-5 * abs(best))
	np.testing.assert_allclose(t_i, lengths, atol=2e-3)


def test_optimiseTStar_falls_back_to_unit_times_when_nothing_converges(monkeypatch):
	"""When neither the EM scheme nor the gradient fallback converges, unit times and -1e9 are returned."""

	class Failed:
		success = False
		x = np.zeros(3)

	monkeypatch.setattr(th, 'findOptimalTSingleCell', lambda *args: 1e9)
	monkeypatch.setattr(th, 'minimize', lambda *args, **kwargs: Failed())
	ltqs, ltqs_vars, _, _, _, _ = _star_inputs(3, 2, seed=1)
	t_i, loglik, W_g, xr_g = th.optimiseTStar(ltqs, ltqs_vars)
	np.testing.assert_array_equal(t_i, np.ones(3))
	assert loglik == -1e9
	expected_xr, expected_W = th.findNodeLtqsGivenLeafs(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=np.ones(3))
	np.testing.assert_allclose(xr_g, expected_xr)
	np.testing.assert_allclose(W_g, expected_W)


def test_findOptimalTSingleCell_brackets_a_far_optimum():
	"""A leaf far from the centre: the bracket is widened by factors of 10 until the derivative changes sign."""
	ltqs_g = np.zeros(3)
	xr_g = np.full(3, 30.0)
	t = th.findOptimalTSingleCell(ltqs_g, np.full(3, 0.1), xr_g, 0.01, np.full(3, 1e6))
	assert th.derLogLikStar(t, np.full(3, 0.1), (xr_g - ltqs_g) ** 2, np.full(3, 1e6)) == pytest.approx(0, abs=1e-6)
	assert t == pytest.approx(900.0 - 0.1, rel=1e-3)


@pytest.mark.parametrize('seed', range(3))
def test_optimiseT3LeafStar_reaches_the_three_leaf_reference_maximum(seed):
	ltqs, ltqs_vars, _, names, means, variances = _star_inputs(3, 5, seed + 10)
	loglik, times, success = th.optimiseT3LeafStar(ltqs, ltqs_vars, np.array([0.5, 0.5, 0.5]))
	assert success
	assert loglik == pytest.approx(_star_ref(names, times, means, variances), rel=1e-8)
	best, _ = optimise_lengths(star_tree(names, np.ones(3)), means, variances)
	# optimiseT3LeafStar uses scipy's default tolerances.
	assert loglik == pytest.approx(best, abs=1e-4)


@pytest.mark.parametrize('seed', range(3))
def test_optimiseT3LeafStarSequential_fixes_t12_and_maximises_the_rest(seed):
	ltqs, ltqs_vars, _, names, means, variances = _star_inputs(3, 5, seed + 20)
	loglik, times, t12, success = th.optimiseT3LeafStarSequential(ltqs, ltqs_vars, np.array([0.5, 0.5]), tol=1e-10)
	assert success
	two_leaf = minimize_scalar(lambda t: -th.getLoglik2LeafTree(ltqs[:, 0], ltqs_vars[:, 0], ltqs[:, 1],
	                                                            ltqs_vars[:, 1], t),
	                           bounds=(0, 50), method='bounded', options={'xatol': 1e-11})
	assert t12 == pytest.approx(two_leaf.x, rel=1e-5, abs=1e-8)
	full = np.array([times[0], t12 - times[0], times[1]])
	assert loglik == pytest.approx(_star_ref(names, full, means, variances), rel=1e-9)

	def constrained(p):
		return -_star_ref(names, np.array([p[0], t12 - p[0], p[1]]), means, variances)

	best = min((minimize(constrained, [frac * t12, start], method='L-BFGS-B',
	                     bounds=[(0, t12), (1e-9, 100)]) for frac in (0.2, 0.5, 0.8) for start in (0.1, 1.0)),
	           key=lambda res: res.fun)
	assert loglik == pytest.approx(-best.fun, abs=1e-5 * abs(best.fun) + 1e-6)


# ================================================================= as-if-root coordinates

def test_getLtqsAsIfRoot_gives_each_nodes_full_posterior():
	tree = parse_newick('((A:0.3,B:0.6)n:0.45,(C:0.2,D:0.1)m:0.3,E:0.8)r;')
	means, variances = _leaf_data(tree, 4, seed=4)
	bonsai_tree = build(write_newick(tree), means, variances)
	post_means, post_vars = reference_posteriors(tree, means, variances)
	root = bonsai_tree.root
	for child in root.childNodes:
		xr, W = th.getLtqsAsIfRoot(child.ltqs, child.getW(), child.tParent, root.ltqs, root.getW())
		np.testing.assert_allclose(xr, post_means[child.nodeId], rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(1 / W, post_vars[child.nodeId], rtol=1e-9)
	ltqs, W_children, times = (np.array([c.ltqs for c in root.childNodes]).T,
	                           np.array([c.getW() for c in root.childNodes]).T,
	                           np.array([c.tParent for c in root.childNodes]))
	xr_all, W_all = th.getLtqsAsIfRoot_vectorized(ltqs, W_children, times, root.ltqs, root.getW())
	for ind, child in enumerate(root.childNodes):
		np.testing.assert_allclose(xr_all[:, ind], post_means[child.nodeId], rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(1 / W_all[:, ind], post_vars[child.nodeId], rtol=1e-9)


def test_getLtqsAsIfRoot_of_an_only_child_is_its_own_message():
	"""When the parent's information comes from this node alone, the flat prior above adds nothing."""
	node_ltqs, node_W = np.array([0.3, -1.0]), np.array([4.0, 2.0])
	parent_W = 1 / (0.5 + 1 / node_W)
	parent_ltqs = node_ltqs.copy()
	xr, W = th.getLtqsAsIfRoot(node_ltqs, node_W, 0.5, parent_ltqs, parent_W)
	np.testing.assert_allclose(xr, node_ltqs, rtol=1e-12)
	np.testing.assert_allclose(W, node_W, rtol=1e-12)
	assert xr is not node_ltqs


def test_getAIRootInfo_and_getAIRootUpstream_give_every_nodes_posterior():
	tree = parse_newick(SMALL_TREE)
	means, variances = _leaf_data(tree, 3, seed=8)
	bonsai_tree = build(SMALL_TREE, means, variances)
	post_means, post_vars = reference_posteriors(tree, means, variances)
	post_means['root'], post_vars['root'] = post_means['r'], post_vars['r']
	bonsai_tree.root.getAIRootInfo(None, None)
	for node in bonsai_nodes(bonsai_tree.root):
		np.testing.assert_allclose(node.ltqsAIRoot, post_means[node.nodeId], rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(node.getLtqsVars(AIRoot=True), post_vars[node.nodeId], rtol=1e-9)
	bonsai_tree.root.clear_AIRoot()
	assert all(node.ltqsAIRoot is None and node.getW(AIRoot=True) is None for node in bonsai_nodes(bonsai_tree.root))
	leaf = [node for node in bonsai_nodes(bonsai_tree.root) if node.nodeId == 'B'][0]
	leaf.getAIRootUpstream(as_if_root_version=3)
	np.testing.assert_allclose(leaf.ltqsAIRoot, post_means['B'], rtol=1e-9, atol=1e-12)
	np.testing.assert_allclose(1 / leaf.getW(AIRoot=True), post_vars['B'], rtol=1e-9)
	# Without ltqs at the root, getAIRootInfo computes them first.
	fresh = build_bonsai_tree(tree, means, variances)
	fresh.root.getAIRootInfo(None, None)
	np.testing.assert_allclose(fresh.root.ltqsAIRoot, post_means['r'], rtol=1e-9)


def test_getLtqsAfterChildUpdate_matches_the_recomputed_root_message():
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(4, 3, seed=2)
	xr_g, W_g = th.findNodeLtqsGivenLeafs(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times)
	new_ltqs, new_vars = np.array([0.5, -0.2, 2.0]), np.array([0.3, 0.2, 0.1])
	got_xr, got_W = th.getLtqsAfterChildUpdate(xr_g, W_g, times[1], ltqs[:, 1], ltqs_vars[:, 1], new_ltqs, new_vars)
	means['L1'], variances['L1'] = new_ltqs, new_vars
	post_means, post_vars = reference_posteriors(star_tree(names, times), means, variances)
	np.testing.assert_allclose(got_xr, post_means['root'], rtol=1e-9)
	np.testing.assert_allclose(1 / got_W, post_vars['root'], rtol=1e-9)


# ================================================================== pair merges

def _merge_star_pair(names, times, pair, opt_times):
	i, j = pair
	merged = Node(None, opt_times[2], [Node(names[i], opt_times[0]), Node(names[j], opt_times[1])])
	return Node('root', None, [merged] + [Node(name, times[k]) for k, name in enumerate(names) if k not in pair])


def _pair_inputs(ltqs, ltqs_vars, times, i, j):
	xr_g, W_g = th.findNodeLtqsGivenLeafs(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times)
	wbar = 1 / (ltqs_vars + times)
	root_minus_first_W = W_g - wbar[:, i]
	root_minus_first_ltqs = xr_g * W_g - wbar[:, i] * ltqs[:, i]
	return (xr_g, W_g, ltqs[:, i], ltqs_vars[:, i], wbar[:, i], times[i], root_minus_first_W, root_minus_first_ltqs,
	        ltqs[:, j], ltqs_vars[:, j], wbar[:, j], times[j])


@pytest.mark.parametrize('sequential', [True, False])
@pytest.mark.parametrize('seed', range(2))
def test_calcSingleDLogL_is_the_change_in_tree_loglik_at_the_returned_times(sequential, seed):
	"""dLogL of merging two children of a star equals the reference log-likelihood of the merged tree (other edges
	unchanged, the three new edges at the returned times) minus that of the star; the joint optimiser reaches the
	maximum over the three new edges, and the sequential one stays at or below it."""
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(5, 6, seed + 30)
	ltqs[:, 1] = ltqs[:, 0] + 0.1 * np.random.default_rng(seed).normal(size=6)
	means['L1'] = ltqs[:, 1]
	args = _pair_inputs(ltqs, ltqs_vars, times, 0, 1)
	dlogl, opt_times, ltqs_gi, ltqs_vars_gi, wbar_gi = th.calcSingleDLogL(*args, sequential=sequential, tol=1e-8,
	                                                                     returnAll=True)
	star = _star_ref(names, times, means, variances)
	merged = _merge_star_pair(names, times, (0, 1), opt_times)
	assert dlogl == pytest.approx(reference_bonsai_loglik(merged, means, variances) - star, rel=1e-8, abs=1e-9)
	assert dlogl > 0
	best, _ = optimise_lengths(merged, means, variances,
	                           fixed={ind + 3: times[ind + 2] for ind in range(3)})
	# best comes from L-BFGS-B on the reference, converged to about 1e-5.
	if sequential:
		assert dlogl <= best - star + 1e-4
	else:
		assert dlogl == pytest.approx(best - star, abs=1e-4)
	assert th.calcSingleDLogL(*args, sequential=sequential, tol=1e-8, onlyTimes=True) == pytest.approx(
		list(opt_times))
	np.testing.assert_allclose(ltqs_gi[:, :2], ltqs[:, :2])
	np.testing.assert_allclose(wbar_gi[:, 2], 1 / ltqs_vars_gi[:, 2])


def test_calcSingleDLogL_reports_minus_infinity_when_the_optimiser_fails(monkeypatch):
	ltqs, ltqs_vars, times, _, _, _ = _star_inputs(4, 3, seed=1)
	monkeypatch.setattr(th, 'optimiseT3LeafStar', lambda *args, **kwargs: (0.0, np.zeros(3), False))
	monkeypatch.setattr(th, 'optimiseT3LeafStarSequential', lambda *args, **kwargs: (None, None, None, False))
	dlogl, opt_times = th.calcSingleDLogL(*_pair_inputs(ltqs, ltqs_vars, times, 0, 1), sequential=True)
	assert dlogl == -np.inf
	assert opt_times == pytest.approx(th.calcTInit(times[0], times[1], False))


def test_getOptTimesSingleDLogLWrapper_returns_the_times_calcSingleDLogL_uses(genes_and_cells):
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(4, 3, seed=3)
	genes_and_cells(3)
	children = [TreeNode(nodeInd=ind, ltqs=ltqs[:, ind], ltqsVars=ltqs_vars[:, ind], tParent=times[ind], isLeaf=True)
	            for ind in range(4)]
	xr_g, W_g = th.findNodeLtqsGivenLeafs(ltqs_gi=ltqs, ltqsVars_gi=ltqs_vars, t_i=times)
	for sequential in (True, False):
		got = th.getOptTimesSingleDLogLWrapper(xr_g, W_g, children[2], children[3], sequential=sequential, tol=1e-9)
		_, expected = th.calcSingleDLogL(*_pair_inputs(ltqs, ltqs_vars, times, 2, 3), sequential=sequential, tol=1e-9)
		assert list(got) == pytest.approx(list(expected), rel=1e-12)


@pytest.mark.parametrize('use_children', [False, True])
def test_estimateDerBasedDLogLUB_returns_the_root_derivatives_of_dLogL(use_children, genes_and_cells):
	"""The two numbers that scale the upper bound are the norms of d dLogL / d x_r divided by sqrt(W_r) and of
	d dLogL / d W_r times W_r, at the given new times (central differences of the three-leaf reference)."""
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(5, 6, seed=3)
	genes_and_cells(6)
	args = _pair_inputs(ltqs, ltqs_vars, times, 0, 1)
	dlogl, opt_times, ltqs_gi, ltqs_vars_gi, wir_gi = th.calcSingleDLogL(*args, sequential=False, returnAll=True)
	xr_g, W_g = args[0], args[1]
	wbar = 1 / (ltqs_vars + times)

	def dlogl_at(xr, W):
		WR = W - wbar[:, 0] - wbar[:, 1]
		xR = (xr * W - wbar[:, 0] * ltqs[:, 0] - wbar[:, 1] * ltqs[:, 1]) / WR
		named = {'a': ltqs[:, 0], 'b': ltqs[:, 1], 'R': xR}
		var_named = {'a': ltqs_vars[:, 0], 'b': ltqs_vars[:, 1], 'R': 1 / WR}
		new = reference_bonsai_loglik(star_tree(['a', 'b', 'R'], opt_times), named, var_named)
		old = reference_bonsai_loglik(star_tree(['a', 'b', 'R'], [times[0], times[1], 0.0]), named, var_named)
		return new - old

	assert dlogl_at(xr_g, W_g) == pytest.approx(dlogl, rel=1e-8)
	root_info = {'pos': xr_g, 'prec': W_g, 'nChild': 5}
	if use_children:
		children = [TreeNode(nodeInd=ind, ltqs=ltqs[:, ind], ltqsVars=ltqs_vars[:, ind], tParent=times[ind],
		                     isLeaf=True) for ind in range(2)]
		got_dlogl, per_eps_x, per_eps_w = th.estimateDerBasedDLogLUB(root_info, np.array(opt_times),
		                                                            child1=children[0], child2=children[1],
		                                                            sequential=False)
		# The times are re-optimised from opt_times with scipy's default tolerance.
		assert got_dlogl == pytest.approx(dlogl, rel=1e-4)
	else:
		got_dlogl, per_eps_x, per_eps_w = th.estimateDerBasedDLogLUB(root_info, opt_times, ltqs_gi=ltqs_gi,
		                                                            ltqsVars_gi=ltqs_vars_gi, wir_gi=wir_gi)
		assert got_dlogl is None
	grad_x = central_difference(lambda xr: dlogl_at(xr, W_g), xr_g)
	grad_w = central_difference(lambda W: dlogl_at(xr_g, W), W_g)
	tolerance = 1e-3 if use_children else 1e-5
	assert per_eps_x == pytest.approx(np.linalg.norm(grad_x / np.sqrt(W_g)), rel=tolerance)
	assert per_eps_w == pytest.approx(np.linalg.norm(grad_w * W_g), rel=tolerance)


# ===================================================== per-gene variance scaling (lambda)

def _scaled(means, variances, lam):
	return ({k: v / np.sqrt(lam) for k, v in means.items()}, {k: v / lam for k, v in variances.items()})


def test_logLGradStarTreeLogLambda_is_the_rescaled_star_reference():
	ltqs, ltqs_vars, times, names, means, variances = _star_inputs(4, 3, seed=4)
	log_lambda = np.array([0.3, -0.5, 1.1])

	def objective(log_lam):
		scaled = _scaled(means, variances, np.exp(log_lam))
		return -(-4 * np.sum(log_lam) + _star_ref(names, times, *scaled))

	value, grad = th.logLGradStarTreeLogLambda(log_lambda, times, ltqs, ltqs_vars, 4)
	assert value == pytest.approx(objective(log_lambda), rel=1e-10)
	np.testing.assert_allclose(grad, central_difference(objective, log_lambda), rtol=1e-5)
	for gene in range(3):
		single_means = {k: v[[gene]] for k, v in means.items()}
		single_vars = {k: v[[gene]] for k, v in variances.items()}

		def single(log_lam):
			scaled = _scaled(single_means, single_vars, np.exp(log_lam[0]))
			return -(-4 * log_lam[0] + _star_ref(names, times, *scaled))

		value, grad = th.logLGradStarTreeLogLambdaSingleGene(log_lambda[[gene]], times, ltqs[gene], ltqs_vars[gene], 4)
		assert value == pytest.approx(single(log_lambda[[gene]]), rel=1e-10)
		assert grad == pytest.approx(central_difference(single, log_lambda[[gene]])[0], rel=1e-5)


def test_tree_lambda_objective_value_matches_the_rescaled_reference(genes_and_cells):
	"""Tree.logLGradCompleteLambda: minus (-C sum_g log lambda_g + the tree log-likelihood of the data divided by
	sqrt(lambda_g), variances by lambda_g), for all genes (geneInd < 0) or one gene."""
	tree = parse_newick('((A:0.3,B:0.6)n:0.45,(C:0.2,D:0.1)m:0.3,E:0.8)r;')
	means, variances = _leaf_data(tree, 3, seed=21)
	bonsai_tree = build(write_newick(tree), means, variances)
	genes_and_cells(3, 5)
	log_lambda = np.array([0.2, -0.4, 0.7])
	value, _ = bonsai_tree.logLGradCompleteLambda(log_lambda, -1, 5)
	expected = -(-5 * np.sum(log_lambda) + reference_bonsai_loglik(tree, *_scaled(means, variances,
	                                                                                np.exp(log_lambda))))
	assert value == pytest.approx(expected, rel=1e-10)
	for gene in range(3):
		single = ({k: v[[gene]] for k, v in means.items()}, {k: v[[gene]] for k, v in variances.items()})
		value, _ = bonsai_tree.logLGradCompleteLambda(log_lambda[[gene]], gene, 5)
		expected = -(-5 * log_lambda[gene] + reference_bonsai_loglik(tree, *_scaled(*single, np.exp(log_lambda[gene]))))
		assert value == pytest.approx(expected, rel=1e-10)


def test_tree_lambda_gradient_and_optGeneVars_on_a_star(genes_and_cells):
	"""On a star the gradient of logLGradCompleteLambda is the derivative of its value, and optGeneVars (all genes
	at once) moves every gene's variance to the maximum of the rescaled likelihood and rescales the leaves.
	Deeper trees and the per-gene mode are left out: on a deeper tree the returned gradient is wrong, and the
	per-gene mode raises ValueError with numpy 2.4."""
	names = ['A', 'B', 'C', 'D', 'E']
	times = [0.3, 0.6, 0.45, 0.2, 0.8]
	means, variances = leaf_data(names, 3, seed=22)
	tree = star_tree(names, times)
	genes_and_cells(3, 5)
	log_lambda = np.array([0.2, -0.4, 0.7])

	def objective(log_lam):
		return -(-5 * np.sum(log_lam) + reference_bonsai_loglik(tree, *_scaled(means, variances, np.exp(log_lam))))

	value, grad = star_of(names, times, means, variances).logLGradCompleteLambda(log_lambda, -1, 5)
	np.testing.assert_allclose(grad, central_difference(objective, log_lambda), rtol=1e-6)
	for gene in range(3):
		single = ({k: v[[gene]] for k, v in means.items()}, {k: v[[gene]] for k, v in variances.items()})

		def objective_one(log_lam):
			return -(-5 * log_lam[0] + reference_bonsai_loglik(tree, *_scaled(*single, np.exp(log_lam[0]))))

		_, grad = star_of(names, times, means, variances).logLGradCompleteLambda(log_lambda[[gene]], gene, 5)
		assert grad == pytest.approx(central_difference(objective_one, log_lambda[[gene]])[0], rel=1e-6)

	gene_vars = np.array([1.0, 2.0, 0.5])
	bonsai_tree = star_of(names, times, means, variances)
	new_vars = bonsai_tree.optGeneVars(gene_vars, genesSimultaneously=True, singleProcess=True)
	lam = new_vars / gene_vars
	for gene in range(3):
		best = minimize_scalar(lambda x: objective(np.where(np.arange(3) == gene, x, 0.0)),
		                       bounds=(-10, 10), method='bounded', options={'xatol': 1e-10})
		assert np.log(lam[gene]) == pytest.approx(best.x, abs=1e-3)
	leaf_a = [node for node in bonsai_nodes(bonsai_tree.root) if node.nodeId == 'A'][0]
	np.testing.assert_allclose(leaf_a.ltqs, means['A'] / np.sqrt(lam), rtol=1e-12)
	np.testing.assert_allclose(leaf_a.getLtqsVars(), variances['A'] / lam, rtol=1e-12)
	assert bonsai_tree.root.prefactor == pytest.approx(reference_bonsai_loglik(tree, *_scaled(means, variances, lam)),
	                                                   rel=1e-9)


def test_assignVs_rescales_every_leaf_and_nothing_else():
	tree = parse_newick('((A:0.3,B:0.6)n:0.45,C:0.2)r;')
	means, variances = _leaf_data(tree, 2, seed=23)
	bonsai_tree = build(write_newick(tree), means, variances)
	internal = [node for node in bonsai_nodes(bonsai_tree.root) if node.nodeId == 'n'][0]
	internal_ltqs = internal.ltqs.copy()
	bonsai_tree.root.assignVs(np.array([4.0, 0.25]))
	for node in bonsai_nodes(bonsai_tree.root):
		if node.isLeaf:
			np.testing.assert_allclose(node.ltqs, means[node.nodeId] / np.array([2.0, 0.5]))
			np.testing.assert_allclose(node.getLtqsVars(), variances[node.nodeId] / np.array([4.0, 0.25]))
	np.testing.assert_array_equal(internal.ltqs, internal_ltqs)


# ============================================================ TreeNode structure

def _hand_tree():
	"""((A:1,B:2)x:3,C:4,(D:0.5,E:1.5,F:2.5)y:0.25)root with cells A..F = nodeInd 0..5, x = 6, y = 7."""
	tree = parse_newick('((A:1.0,B:2.0)x:3.0,C:4.0,(D:0.5,E:1.5,F:2.5)y:0.25)r;')
	means, variances = leaf_data(tree.leaf_names(), 2, seed=0)
	return tree, build_bonsai_tree(tree, means, variances), means, variances


def _by_id(bonsai_tree):
	return {node.nodeId: node for node in bonsai_nodes(bonsai_tree.root)}


def test_TreeNode_init_and_repr():
	node = TreeNode(nodeInd=4, ltqs=np.array([1.0]), ltqsVars=np.array([0.5]), tParent=0.3, nodeId='c4', isCell=True)
	assert node.childNodes == [] and node.parentNode is None and node.isLeaf is False and node.isRoot is False
	assert node.isCell is True and node.vert_ind is None and node.tParent == 0.3
	np.testing.assert_array_equal(node.getLtqsVars(), [0.5])
	np.testing.assert_array_equal(node.getW(), [2.0])
	text = repr(node)
	assert isinstance(text, str) and "'c4'" in text
	assert isinstance(repr(Tree()), str)


def test_Tree_init_buildTree_and_initialize_star_tree(genes_and_cells):
	tree = Tree()
	assert tree.root.isRoot and tree.root.nodeInd == -1 and tree.root.nodeId == 'root' and tree.root.childNodes == []
	assert bs_glob.nNodes == 1
	ltqs, ltqs_vars, _, names, means, variances = _star_inputs(4, 3, seed=40)
	genes_and_cells(3, 4)
	tree.buildTree(ltqs, ltqs_vars, names)
	assert [child.nodeId for child in tree.root.childNodes] == names
	assert [child.nodeInd for child in tree.root.childNodes] == [0, 1, 2, 3]
	assert all(child.isLeaf and child.isCell and child.tParent == 1 for child in tree.root.childNodes)
	assert bs_glob.nNodes == tree.nNodes == 5 and bs_glob.max_node_ind == tree.max_node_ind == 3

	class Metadata:
		cellIds = names
		loglikVarCorr = -2.0

	star = Tree()
	star.initialize_star_tree(ltqs, ltqs_vars, Metadata(), opt_times=True, verbose=True)
	t_star, loglik, W_g, xr_g = th.optimiseTStar(ltqs, ltqs_vars)
	np.testing.assert_allclose([child.tParent for child in star.root.childNodes], t_star, rtol=1e-12)
	np.testing.assert_allclose(star.root.ltqs, xr_g, rtol=1e-12)
	assert star.calcLogLComplete() == pytest.approx(_star_ref(names, t_star, means, variances), rel=1e-10)
	unit = Tree()
	unit.initialize_star_tree(ltqs, ltqs_vars, Metadata(), opt_times=False)
	assert [child.tParent for child in unit.root.childNodes] == [1.0] * 4


def test_to_newick_round_trips_names_lengths_and_structure(tmp_path):
	tree, bonsai_tree, _, _ = _hand_tree()
	text = bonsai_tree.to_newick(use_ids=True, results_path=str(tmp_path / 'tree.nwk'))
	assert (tmp_path / 'tree.nwk').read_text() == text
	parsed = parse_newick(text)
	assert parsed.name == 'root' and parsed.length is None
	assert clades(parsed, include_root=True) == clades(tree, include_root=True)
	for name in tree.leaf_names() + ['x', 'y']:
		assert parsed.find(name).length == tree.find(name).length
	# Nodes without an ID get one, unique and stable.
	_by_id(bonsai_tree)['x'].nodeId = None
	text = bonsai_tree.to_newick()
	parsed = parse_newick(text)
	names = [node.name for node in parsed.iter_nodes()]
	assert None not in names and len(names) == len(set(names))
	assert bonsai_tree.to_newick() == text


def test_get_max_node_ind_count_leafs_and_node_counts():
	_, bonsai_tree, _, _ = _hand_tree()
	assert bonsai_tree.root.get_max_node_ind(-np.inf) == 7
	nodes = _by_id(bonsai_tree)
	assert nodes['root'].countDSLeafs(0) == 6 and nodes['y'].countDSLeafs(0) == 3 and nodes['A'].countDSLeafs(0) == 1
	assert bonsai_tree.root.get_ds_node_counts() == 9
	assert [nodes[k].n_ds_nodes for k in ('x', 'y', 'C', 'A')] == [3, 4, 1, 1]
	assert bonsai_tree.root.get_ds_info_for_ladderize() == 6
	assert [nodes[k].dsLeafs for k in ('x', 'y', 'C')] == [2, 3, 1]
	nodes['x'].add_n_nodes_upstream(5)
	assert nodes['x'].n_ds_nodes == 8 and nodes['root'].n_ds_nodes == 14 and nodes['y'].n_ds_nodes == 4


def test_get_edge_dataframe_lists_every_edge():
	tree, bonsai_tree, _, _ = _hand_tree()
	frame = bonsai_tree.get_edge_dataframe()
	assert list(frame.columns) == ['source', 'target', 'dist']
	expected = {(parent.name, child.name, child.length) for parent in tree.iter_nodes() for child in parent.children}
	expected = {(('root' if s == 'r' else s), t, d) for s, t, d in expected}
	assert set(map(tuple, frame.itertuples(index=False))) == expected
	vert = {name: ind for ind, name in enumerate(['root', 'x', 'A', 'B', 'C', 'y', 'D', 'E', 'F'])}
	frame = bonsai_tree.get_edge_dataframe(nodeIdToVertInd=vert)
	assert list(frame.columns) == ['source', 'source_ind', 'target', 'target_ind', 'dist']
	for row in frame.itertuples(index=False):
		assert vert[row.source] == row.source_ind and vert[row.target] == row.target_ind


def test_copy_tree_topology_is_an_independent_topology_copy():
	tree, bonsai_tree, _, _ = _hand_tree()
	bonsai_tree.nNodes, bonsai_tree.max_node_ind = 9, 7
	copy = bonsai_tree.copy_tree_topology()
	assert copy.nNodes == 9 and copy.max_node_ind == 7
	copied = to_node(copy.root)
	assert write_newick(copied) == write_newick(to_node(bonsai_tree.root))
	assert all(node.ltqs is None and node.getLtqsVars() is None for node in bonsai_nodes(copy.root))
	assert [node.nodeInd for node in bonsai_nodes(copy.root)] == [node.nodeInd for node in bonsai_nodes(bonsai_tree.root)]
	assert [node.isCell for node in bonsai_nodes(copy.root)] == [node.isCell for node in bonsai_nodes(bonsai_tree.root)]
	copy.root.childNodes[0].tParent = 99.0
	assert bonsai_tree.root.childNodes[0].tParent == 3.0


def test_getTotalTopology_identifies_unrooted_topologies():
	"""The same unrooted tree gives the same topology object whatever the rooting and child order; an NNI away
	gives a different one. Cells are compared by ID (nodeInd follows the newick's leaf order)."""
	means, variances = leaf_data(list('ABCDE'), 2, seed=1)

	def topo(newick, **kwargs):
		return build_bonsai_tree(parse_newick(newick), means, variances).root.getTotalTopology(**kwargs)

	base, center = topo('((A:1,B:1)x:1,C:1,(D:1,E:1)y:1)r;', use_cell_ids=True, center_leaf_id='A')
	assert center == 'A'
	assert topo('((E:2,D:1)y:1,(B:1,A:3)x:1,C:1)r;', use_cell_ids=True, center_leaf_id='A')[0] == base
	assert topo('(A:1,B:1,(C:1,(D:1,E:1)y:1)z:1)r;', use_cell_ids=True, center_leaf_id='A')[0] == base
	assert topo('((A:1,C:1)x:1,B:1,(D:1,E:1)y:1)r;', use_cell_ids=True, center_leaf_id='A')[0] != base
	assert topo('((A:1,B:1)x:1,C:1,(D:1,E:1)y:1)r;', use_cell_ids=True, center_leaf_id='D')[0] == \
		topo('(D:1,E:1,((A:1,B:1)x:1,C:1)z:1)r;', use_cell_ids=True, center_leaf_id='D')[0]
	# By nodeInd (cells numbered in leaf order, the same in these two strings).
	by_ind, center_ind = topo('((A:1,B:1)x:1,C:1,(D:1,E:1)y:1)r;')
	assert center_ind == 0
	assert topo('(A:1,B:1,(C:1,(D:1,E:1)y:1)z:1)r;', centerLeafInd=0)[0] == by_ind
	assert topo('(A:1,B:1,(C:1,(D:1,E:1)y:1)z:1)r;', centerLeafInd=2)[1] == 2
	timed, _ = topo('((A:1,B:1)x:1,C:1,(D:1,E:1)y:1)r;', with_times=True, use_cell_ids=True, center_leaf_id='A')
	assert timed == topo('((E:1,D:1)y:1,(B:1,A:1)x:1,C:1)r;', with_times=True, use_cell_ids=True,
	                     center_leaf_id='A')[0]
	assert timed != topo('((A:1,B:1)x:1,C:1,(D:1,E:2)y:1)r;', with_times=True, use_cell_ids=True,
	                     center_leaf_id='A')[0]


def test_getNodeList_getNodeDict_and_breadth_first_list():
	_, bonsai_tree, _, _ = _hand_tree()
	pre_order = ['root', 'x', 'A', 'B', 'C', 'y', 'D', 'E', 'F']
	assert [node.nodeId for node in bonsai_tree.root.getNodeList([])] == pre_order
	assert [node.nodeId for node in bonsai_tree.root.getNodeList([], returnLeafs=False)] == ['root', 'x', 'y']
	assert [node.nodeId for node in bonsai_tree.root.getNodeList([], returnRoot=False, returnLeafs=False)] == ['x', 'y']
	assert list(bonsai_tree.root.getNodeDict({})) == pre_order
	assert list(bonsai_tree.root.getNodeDict({}, returnLeafs=False, returnRoot=False)) == ['x', 'y']
	assert [node.nodeId for node in bonsai_tree.getNodeListBF()] == ['root', 'x', 'C', 'y', 'A', 'B', 'D', 'E', 'F']
	for nth, name in enumerate(pre_order):
		bonsai_tree.root.get_ds_node_counts()
		assert bonsai_tree.root.select_nth_node_df(nth, 0).nodeId == name


def test_renumber_verts_and_vert_index_lookup():
	_, bonsai_tree, _, _ = _hand_tree()
	pre_order = ['root', 'x', 'A', 'B', 'C', 'y', 'D', 'E', 'F']
	old = {node.nodeId: node.nodeInd for node in bonsai_nodes(bonsai_tree.root)}
	vert_to_node, count = bonsai_tree.root.renumber_verts({}, 0)
	assert count == 9 and [vert_to_node[ind].nodeId for ind in range(9)] == pre_order
	assert {node.nodeId: node.nodeInd for node in bonsai_nodes(bonsai_tree.root)} == old
	assert {ind: node.nodeId for ind, node in bonsai_tree.get_vert_ind_to_node_DF().items()} == dict(enumerate(pre_order))
	# update=True reflects the vertex indices the nodes hold now (what update=False returns after a change is left
	# open: a rewrite may cache or recompute).
	_by_id(bonsai_tree)['C'].vert_ind = 42
	assert bonsai_tree.get_vert_ind_to_node_DF(update=True)[42].nodeId == 'C'
	vert_to_node, count, mapping = bonsai_tree.root.renumber_verts({}, 0, include_nodeInd=True, old_ind_to_new_ind={})
	assert mapping[42] == 4 and count == 9
	assert [node.nodeInd for node in bonsai_nodes(bonsai_tree.root)] == list(range(9))


def test_renumberNodes_counts_nodes_and_renames_internal_ones(genes_and_cells):
	_, bonsai_tree, _, _ = _hand_tree()
	genes_and_cells(2, 6)
	bs_glob.nNodes = None
	bonsai_tree.root.renumberNodes(change_node_inds=False)
	assert bs_glob.nNodes == 9
	assert [node.nodeInd for node in bonsai_nodes(bonsai_tree.root)] == [-1, 6, 0, 1, 2, 7, 3, 4, 5]
	bs_glob.max_node_ind = 5
	bonsai_tree.root.renumberNodes(change_node_inds=True)
	assert [(node.nodeInd, node.nodeId) for node in bonsai_nodes(bonsai_tree.root) if not node.isLeaf] == \
		[(-1, 'root'), (6, 'internal_6'), (7, 'internal_7')]
	assert bs_glob.max_node_ind == 7 and bs_glob.nNodes == 9
	assert [node.nodeId for node in bonsai_nodes(bonsai_tree.root) if node.isLeaf] == list('ABCDEF')
	# Nodes without an ID are named after their index.
	_by_id(bonsai_tree)['internal_7'].nodeId = None
	bonsai_tree.root.renumberNodes(change_node_inds=False)
	assert 'internal_7' in _by_id(bonsai_tree)


def test_gatherInfoDepthFirst_and_version_reset():
	_, bonsai_tree, means, _ = _hand_tree()
	assert bonsai_tree.root.gatherInfoDepthFirst([]) == [3, 2, 0, 0, 0, 3, 0, 0, 0]
	info = bonsai_tree.root.gatherInfoDepthFirstGeneral([])
	assert len(info) == 6
	for got, name in zip(info, 'ABCDEF'):
		np.testing.assert_array_equal(got, means[name][:2])
	# reset_version_numbers sets the named attribute to -1 on every node (an attribute of the test's own here).
	for node in bonsai_nodes(bonsai_tree.root):
		node.test_version = 5
	bonsai_tree.root.reset_version_numbers('test_version')
	assert all(node.test_version == -1 for node in bonsai_nodes(bonsai_tree.root))


def test_ltqs_variance_and_precision_storage():
	"""A node's variances and precisions are each other's reciprocals whichever one was set, with or without
	mem_friendly and for the as-if-root coordinates. Which of the two a node keeps in memory is not pinned. The
	divisions here are by powers of two or give the correctly rounded 0.1, so the values are exact."""
	node = TreeNode(ltqsVars=np.array([0.5, 4.0]))
	np.testing.assert_array_equal(node.getW(), [2.0, 0.25])
	node.setLtqsVarsOrW(W_g=np.array([10.0, 1.0]))
	np.testing.assert_array_equal(node.getLtqsVars(), [0.1, 1.0])
	np.testing.assert_array_equal(node.getW(), [10.0, 1.0])
	node.setLtqsVarsOrW(W_g=np.array([10.0, 1.0]))
	np.testing.assert_array_equal(node.getLtqsVars(mem_friendly=True), [0.1, 1.0])
	np.testing.assert_allclose(node.getW(mem_friendly=True), [10.0, 1.0], rtol=1e-15)
	assert node.getLtqsVars() == pytest.approx([0.1, 1.0])
	node.setLtqsVarsOrW(ltqsVars=np.array([2.0]), AIRoot=True)
	np.testing.assert_array_equal(node.getW(AIRoot=True), [0.5])
	node.setLtqsVarsOrW(W_g=np.array([8.0]), AIRoot=True)
	np.testing.assert_array_equal(node.getLtqsVars(AIRoot=True), [0.125])
	node.setLtqsVarsOrW(ltqsVars=np.array([0.125]), AIRoot=True)
	np.testing.assert_array_equal(node.getW(AIRoot=True, mem_friendly=True), [8.0])
	np.testing.assert_array_equal(node.getLtqsVars(AIRoot=True, mem_friendly=True), [0.125])
	empty = TreeNode()
	assert empty.getW() is None and empty.getLtqsVars() is None and empty.getW(AIRoot=True) is None


def test_keep_one_ltqsvars_or_W_and_clear_memory():
	"""Freeing memory keeps every node's variances and precisions (reciprocals of each other) and the tree's
	log-likelihood. What is freed is not pinned: a rewrite may keep both or neither."""
	tree = parse_newick('((A:0.3,B:0.6)n:0.45,C:0.2)r;')
	means, variances = leaf_data(tree.leaf_names(), 2, seed=2)
	bonsai_tree = build(write_newick(tree), means, variances)
	expected = {node.nodeId: node.getLtqsVars().copy() for node in bonsai_nodes(bonsai_tree.root)}

	def check():
		for node in bonsai_nodes(bonsai_tree.root):
			np.testing.assert_allclose(node.getLtqsVars(), expected[node.nodeId], rtol=1e-14)
			np.testing.assert_allclose(node.getW(), 1 / expected[node.nodeId], rtol=1e-14)

	for node in bonsai_nodes(bonsai_tree.root):
		node.getW()
		node.getLtqsVars()
	bonsai_tree.root.keep_one_ltqsvars_or_W(keep_ltqsvars=True)
	check()
	bonsai_tree.root.keep_one_ltqsvars_or_W(keep_ltqsvars=False)
	check()
	bonsai_tree.root.getAIRootInfo(None, None)
	bonsai_tree.root.clear_memory(mem_friendly=True)
	check()
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(tree, means, variances), rel=1e-10)


def test_times_info_and_child_arrays(genes_and_cells):
	_, bonsai_tree, means, variances = _hand_tree()
	genes_and_cells(2, 6)
	times = bonsai_tree.root.getTs({})
	assert times == {6: 3.0, 0: 1.0, 1: 2.0, 2: 4.0, 7: 0.25, 3: 0.5, 4: 1.5, 5: 2.5}
	bonsai_tree.root.assignTs({key: value * 2 for key, value in times.items()})
	assert bonsai_tree.root.getTs({}) == {key: value * 2 for key, value in times.items()}
	assert bonsai_tree.root.tParent is None
	leaf = _by_id(bonsai_tree)['B']
	t, wbar, ltqs, ltqs_vars, ind = leaf.getInfo()
	assert t == 4.0 and ind == 1
	np.testing.assert_allclose(wbar, 1 / (4.0 + variances['B']))
	np.testing.assert_array_equal(ltqs, means['B'])
	node_y = _by_id(bonsai_tree)['y']
	ltqs, ltqs_vars, t = node_y.getInfoChildren()
	np.testing.assert_array_equal(ltqs, np.array([means[k] for k in 'DEF']).T)
	np.testing.assert_array_equal(ltqs_vars, np.array([variances[k] for k in 'DEF']).T)
	np.testing.assert_array_equal(t, [1.0, 3.0, 5.0])
	for node in bonsai_nodes(bonsai_tree.root):
		node.dLoglikdtParent = node.nodeInd * 10.0
	grads = bonsai_tree.root.getGrads({})
	assert grads == {key: key * 10.0 for key in times}


def test_getLtqsNoneOnly_setLtqsUpstream_and_getLtqsUponMerge():
	tree = parse_newick(SMALL_TREE)
	means, variances = _leaf_data(tree, 3, seed=31)
	bonsai_tree = build(SMALL_TREE, means, variances)
	expected = {node.nodeId: (node.ltqs.copy(), node.getW().copy()) for node in bonsai_nodes(bonsai_tree.root)}
	nodes = _by_id(bonsai_tree)
	for name in ('i', 'j', 'root'):
		nodes[name].ltqs = None
	bonsai_tree.root.getLtqsNoneOnly()
	for name in ('i', 'j', 'root'):
		np.testing.assert_allclose(nodes[name].ltqs, expected[name][0], rtol=1e-12)
		np.testing.assert_allclose(nodes[name].getW(), expected[name][1], rtol=1e-12)
	# Change cell A's data, recompute its parent by hand and propagate upwards.
	new_means = dict(means)
	new_means['A'] = means['A'] + 1.0
	nodes['A'].ltqs = new_means['A']
	nodes['i'].getLtqsUponMerge()
	nodes['i'].setLtqsUpstream()
	post_means, post_vars = reference_posteriors(tree, new_means, variances)
	np.testing.assert_allclose(nodes['root'].ltqs, post_means['r'], rtol=1e-9)
	np.testing.assert_allclose(nodes['root'].getLtqsVars(), post_vars['r'], rtol=1e-9)
	nodes['root'].setLtqsUpstream()  # the root has nothing above it


def test_getChildMergers_lists_merges_in_post_order():
	tree, bonsai_tree, _, _ = _hand_tree()
	# Every child's subtree first, then the node's own merges: its first child's representative cell with each
	# further child's, the first pair at both children's times, later ones at time 0 for the representative.
	assert bonsai_tree.getMergeList() == [(0, 1, 1.0, 2.0), (3, 4, 0.5, 1.5), (3, 5, 0.0, 2.5), (0, 2, 3.0, 4.0),
	                                      (0, 3, 0.0, 0.25)]
	childless = TreeNode(nodeInd=9, isLeaf=False)
	assert childless.getChildMergers([]) == 9 and childless.isLeaf


def test_getEdgesComplete_walks_every_edge_from_any_node():
	tree, bonsai_tree, _, _ = _hand_tree()
	bonsai_tree.root.storeParent()
	ind = {node.nodeId: node.nodeInd for node in bonsai_nodes(bonsai_tree.root)}
	expected = {frozenset((ind[('root' if p.name == 'r' else p.name)], ind[c.name])): c.length
	            for p in tree.iter_nodes() for c in p.children}
	for start in ('root', 'A', 'y'):
		edges = _by_id(bonsai_tree)[start].getEdgesComplete([])
		assert len(edges) == 8
		assert {frozenset(edge[:2]): edge[2] for edge in edges} == expected
		for a, b, length, b_is_leaf in edges:
			assert b_is_leaf == (b in range(6))
	edges = _by_id(bonsai_tree)['x'].getEdgesComplete([], indMap={k: k + 100 for k in ind.values()})
	assert {frozenset(edge[:2]) for edge in edges} == {frozenset(k + 100 for k in e) for e in expected}


def test_storeParent_sets_every_parent_pointer():
	_, bonsai_tree, _, _ = _hand_tree()
	for node in bonsai_nodes(bonsai_tree.root):
		node.parentNode = None
	bonsai_tree.root.storeParent()
	for node in bonsai_nodes(bonsai_tree.root):
		for child in node.childNodes:
			assert child.parentNode is node


def test_addClosenessNNN_adds_inverse_distances_to_internal_neighbours():
	"""From node x, every internal non-root node d edges away (moving away from x) gains 1/(d+1)."""
	means, variances = leaf_data(list('ABCDEFG'), 1, seed=0)
	bonsai_tree = build_bonsai_tree(parse_newick('(((A:1,B:1)u:1,C:1)x:1,((D:1,E:1)w:1,F:1)v:1,G:1)r;'), means,
	                                variances)
	bonsai_tree.root.storeParent()
	nodes = _by_id(bonsai_tree)
	for node in bonsai_nodes(bonsai_tree.root):
		node.cumClosenessNNN = 0.0
	nodes['x'].addClosenessNNN(dist=0)
	assert nodes['u'].cumClosenessNNN == pytest.approx(1 / 1)
	assert nodes['root'].cumClosenessNNN == 0.0
	assert nodes['x'].cumClosenessNNN == 0.0
	# The root is not counted and not crossed.
	assert nodes['v'].cumClosenessNNN == 0.0 and nodes['w'].cumClosenessNNN == 0.0
	nodes['w'].addClosenessNNN(dist=0)
	assert nodes['v'].cumClosenessNNN == pytest.approx(1.0)
	assert nodes['u'].cumClosenessNNN == pytest.approx(1.0)
	nodes['root'].addClosenessNNN(dist=0)
	assert nodes['x'].cumClosenessNNN == pytest.approx(1.0) and nodes['v'].cumClosenessNNN == pytest.approx(2.0)
	assert nodes['u'].cumClosenessNNN == pytest.approx(1.5) and nodes['w'].cumClosenessNNN == pytest.approx(0.5)


# ================================================== cleanup, rooting and child order

def _assert_same_unrooted_tree(tree_a, tree_b, rel=1e-12):
	"""Same leaves, same splits (zero-length edges collapsed) and the same path length between every leaf pair."""
	assert sorted(tree_a.leaf_names()) == sorted(tree_b.leaf_names())
	assert rf_distance(tree_a, tree_b) == 0
	got, expected = leaf_pairs_distances(tree_a), leaf_pairs_distances(tree_b)
	for pair, length in expected.items():
		assert got[pair] == pytest.approx(length, rel=rel, abs=1e-12)


def test_deleteParentsWithOneChild_removes_degree_two_nodes(genes_and_cells):
	newick = '((A:1.0,(B:2.0)u:0.5)v:1.0,(C:1.0)w:2.0,(D:1.0,E:0.5)x:0.25)r;'
	tree = parse_newick(newick)
	names = ['A', 'B', 'C', 'D', 'E']
	means, variances = leaf_data(names, 2, seed=3)
	bonsai_tree = build_bonsai_tree(tree, means, variances)
	# An internal node left without children (not a cell) is dropped as well.
	empty = TreeNode(nodeInd=99, childNodes=[], isLeaf=False, isCell=False, tParent=1.0, nodeId='empty',
	                 parentNode=bonsai_tree.root)
	bonsai_tree.root.childNodes.append(empty)
	genes_and_cells(2, 5)
	bonsai_tree.root.deleteParentsWithOneChild(recalc_ltqs=True)
	result = to_node(bonsai_tree.root)
	assert sorted(result.leaf_names()) == names
	assert all(len(node.children) != 1 for node in result.iter_nodes())
	expected = parse_newick('((A:1.0,B:2.5)v:1.0,C:3.0,(D:1.0,E:0.5)x:0.25)r;')
	_assert_same_unrooted_tree(result, expected)
	assert bs_glob.nNodes == 8
	assert [node.vert_ind for node in bonsai_nodes(bonsai_tree.root)] == list(range(8))
	assert bonsai_tree.root.prefactor == pytest.approx(reference_bonsai_loglik(expected, means, variances), rel=1e-10)


def test_deleteParentsWithOneChild_collapses_a_single_child_root_and_keeps_cells(genes_and_cells):
	means, variances = leaf_data(['A', 'B', 'C'], 2, seed=4)
	bonsai_tree = build_bonsai_tree(parse_newick('(((A:1.0,B:2.0)x:3.0)y:0.5)r;'), means, variances)
	genes_and_cells(2, 3)
	bonsai_tree.root.deleteParentsWithOneChild(recalc_ltqs=False)
	assert write_newick(to_node(bonsai_tree.root)) == '(A:4.5,B:5.5)root;'
	assert bonsai_tree.root.childNodes[0].parentNode is bonsai_tree.root
	# A cell with a single child stays: it holds a measurement.
	cell_tree = build_bonsai_tree(parse_newick('((A:1.0)C:2.0,B:1.0,D:1.0)r;'), *leaf_data(list('ABCD'), 2, seed=5))
	cell = _by_id(cell_tree)['C']
	cell.isCell = True
	cell_tree.root.deleteParentsWithOneChild(recalc_ltqs=False)
	assert 'C' in _by_id(cell_tree) and [child.nodeId for child in cell.childNodes] == ['A']


def test_mergeZeroTimeChilds_collapses_internal_zero_edges_only(genes_and_cells):
	newick = '(((A:1.0,B:0.5)x:0.0,C:1.0)y:0.0,(D:1.0,E:0.0)z:2.0,F:0.0)r;'
	tree = parse_newick(newick)
	means, variances = leaf_data(tree.leaf_names(), 3, seed=6)
	bonsai_tree = build_bonsai_tree(tree, means, variances)
	genes_and_cells(3, 6)
	before = bonsai_tree.calcLogLComplete()
	bonsai_tree.root.mergeZeroTimeChilds()
	result = to_node(bonsai_tree.root)
	assert 'x' not in result.leaf_names() and all(node.name not in ('x', 'y') for node in result.iter_nodes())
	assert sorted(child.name for child in result.children) == ['A', 'B', 'C', 'F', 'z']
	assert sorted(result.find('z').leaf_names()) == ['D', 'E'] and result.find('E').length == 0.0
	assert clades(result) == clades(tree)
	assert all(child.parentNode is bonsai_tree.root for child in bonsai_tree.root.childNodes)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(before, rel=1e-12)
	# A cell sitting at an internal position on a zero-length edge is kept.
	cell_tree = build_bonsai_tree(parse_newick('((A:1.0,B:1.0)M:0.0,C:1.0,D:1.0)r;'), *leaf_data(list('ABCD'), 2, 7))
	_by_id(cell_tree)['M'].isCell = True
	cell_tree.root.mergeZeroTimeChilds()
	assert 'M' in _by_id(cell_tree)


def _tree_with_lengths(seed, newick='(((A:1,B:1)u:1,C:1)x:1,((D:1,E:1)w:1,F:1)v:1,(G:1,H:1)s:1)r;'):
	tree = parse_newick(newick)
	rng = np.random.default_rng(seed)
	tree = with_lengths(tree, rng.uniform(0.1, 2.0, len(edges_of(tree))))
	means, variances = leaf_data(tree.leaf_names(), 3, seed=seed)
	return tree, build_bonsai_tree(tree, means, variances), means, variances


@pytest.mark.parametrize('new_root', ['u', 'w', 's', 'x'])
def test_reset_root_reroots_without_changing_the_unrooted_tree(new_root, genes_and_cells):
	tree, bonsai_tree, means, variances = _tree_with_lengths(seed=1)
	genes_and_cells(3, 8)
	before = bonsai_tree.calcLogLComplete()
	bonsai_tree.root.renumber_verts({}, 0)
	bonsai_tree.reset_root(new_root_ind=_by_id(bonsai_tree)[new_root].vert_ind)
	root = bonsai_tree.root
	assert root.nodeId == new_root and root.isRoot and root.parentNode is None and root.tParent is None
	assert sum(node.isRoot for node in bonsai_nodes(root)) == 1
	for node in bonsai_nodes(root):
		for child in node.childNodes:
			assert child.parentNode is node
	_assert_same_unrooted_tree(to_node(root), tree)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(before, rel=1e-9)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_of(bonsai_tree, means, variances), rel=1e-9)


@pytest.mark.parametrize('seed', range(3))
def test_set_midpoint_root_halves_the_longest_internal_path(seed, genes_and_cells):
	"""get_longest_path_ds ignores the edges to cells, so the root goes to the middle of the longest path between
	two parents of cells."""
	tree, bonsai_tree, means, variances = _tree_with_lengths(seed + 10)
	genes_and_cells(3, 8)
	before = bonsai_tree.calcLogLComplete()
	parent_of = tree.parents()
	best = max(((path_length(tree, a, b) - tree.find(a).length - tree.find(b).length, a, b)
	            for a, b in itertools.combinations(tree.leaf_names(), 2)))
	bonsai_tree.root.renumber_verts({}, 0)  # set_midpoint_root looks nodes up by vert_ind
	bonsai_tree.set_midpoint_root()
	result = to_node(bonsai_tree.root)
	assert result.name == 'midpoint_root' and len(result.children) == 2
	_assert_same_unrooted_tree(result, tree)
	length, a, b = best
	assert path_length(result, 'midpoint_root', a) - tree.find(a).length == pytest.approx(length / 2, rel=1e-9)
	assert path_length(result, 'midpoint_root', b) - tree.find(b).length == pytest.approx(length / 2, rel=1e-9)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(before, rel=1e-9)
	assert [node.vert_ind for node in bonsai_nodes(bonsai_tree.root)] == list(range(bonsai_tree.nNodes))


def _cross_pair_distance(tree):
	"""For every edge, the summed path length over leaf pairs it separates."""
	names = tree.leaf_names()
	distances = leaf_pairs_distances(tree)
	out = {}
	for node in edges_of(tree):
		below = set(node.leaf_names())
		out[node.name] = sum(distances[frozenset((a, b))] for a in below for b in names if b not in below)
	return out


@pytest.mark.parametrize('seed', range(3))
def test_set_mindist_root_roots_at_the_middle_of_the_edge_that_best_splits_the_cells(seed, genes_and_cells):
	"""The edge whose removal leaves the smallest summed within-cluster distance separates the largest summed
	cross-pair distance; the new root halves it."""
	tree, bonsai_tree, means, variances = _tree_with_lengths(seed + 20)
	genes_and_cells(3, 8)
	before = bonsai_tree.calcLogLComplete()
	cross = _cross_pair_distance(tree)
	best_edge = max(cross, key=cross.get)
	centers = bonsai_tree.get_cluster_centers(cell_ids=None, n_clusters=2)
	parent = tree.parents()[id(tree.find(best_edge))]
	assert centers == ['root' if parent is tree else parent.name]
	assert bonsai_tree.set_mindist_root(cell_ids=tree.leaf_names())
	result = to_node(bonsai_tree.root)
	assert result.name == 'mindist_root' and len(result.children) == 2
	_assert_same_unrooted_tree(result, tree)
	sides = [frozenset(child.leaf_names()) for child in result.children]
	assert frozenset(tree.find(best_edge).leaf_names()) in sides
	assert result.children[0].length == pytest.approx(tree.find(best_edge).length / 2, rel=1e-12)
	assert result.children[1].length == pytest.approx(tree.find(best_edge).length / 2, rel=1e-12)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(before, rel=1e-9)


def test_remove_two_child_root_moves_the_root_to_an_internal_node(genes_and_cells):
	tree, bonsai_tree, means, variances = _tree_with_lengths(seed=30)
	genes_and_cells(3, 8)
	before = bonsai_tree.calcLogLComplete()
	bonsai_tree.set_mindist_root(cell_ids=tree.leaf_names())
	bonsai_tree.remove_two_child_root()
	result = to_node(bonsai_tree.root)
	assert len(result.children) >= 3
	assert all(len(node.children) != 1 for node in result.iter_nodes())
	assert 'mindist_root' not in [node.name for node in result.iter_nodes()]
	_assert_same_unrooted_tree(result, tree)
	assert bs_glob.nNodes == bonsai_tree.nNodes == len(list(result.iter_nodes()))
	assert bonsai_tree.calcLogLComplete() == pytest.approx(before, rel=1e-9)
	# A root with three or more children is left alone.
	unchanged = write_newick(result)
	bonsai_tree.remove_two_child_root()
	assert write_newick(to_node(bonsai_tree.root)) == unchanged


def test_remove_two_child_root_searches_below_leaf_children(genes_and_cells):
	"""Root (A, x) with x = (B, (C, D)): the first child with two children is found breadth-first."""
	means, variances = leaf_data(list('ABCD'), 2, seed=8)
	bonsai_tree = build_bonsai_tree(parse_newick('(A:1.0,(B:1.0,(C:1.0,D:2.0)y:0.5)x:0.25)r;'), means, variances)
	genes_and_cells(2, 4)
	bonsai_tree.remove_two_child_root()
	result = to_node(bonsai_tree.root)
	assert len(result.children) == 3
	_assert_same_unrooted_tree(result, parse_newick('(A:1.25,B:1.0,(C:1.0,D:2.0)y:0.5)r;'))


def _cyclic_cost(order, points):
	return sum(math.log(1e-6 + np.linalg.norm(points[order[k]] - points[order[k - 1]])) for k in range(len(order)))


def _path_cost(order, points, left, right):
	sequence = [left] + [points[ind] for ind in order] + [right]
	return sum(math.log(1e-9 + np.linalg.norm(sequence[k] - sequence[k - 1])) for k in range(1, len(sequence)))


def test_reorderChildrenRoot_minimises_the_summed_log_distance_around_the_root(genes_and_cells):
	rng = np.random.default_rng(2)
	names = list('ABCDEF')
	means, variances = {name: rng.normal(size=3) for name in names}, {name: np.ones(3) for name in names}
	bonsai_tree = star_of(names, [1.0] * 6, means, variances)
	genes_and_cells(3, 6)
	bonsai_tree.root.reorderChildrenRoot(maxChild=8)
	order = [names.index(child.nodeId) for child in bonsai_tree.root.childNodes]
	assert sorted(order) == list(range(6))
	points = [means[name] for name in names]
	best = min(_cyclic_cost((0,) + perm, points) for perm in itertools.permutations(range(1, 6)))
	assert _cyclic_cost(order, points) == pytest.approx(best, rel=1e-12)
	# More children than maxChild: a greedy path through all of them.
	bonsai_tree.root.reorderChildrenRoot(maxChild=4)
	assert sorted(child.nodeId for child in bonsai_tree.root.childNodes) == names


def test_reorderChildren_minimises_the_path_between_the_neighbours(genes_and_cells):
	rng = np.random.default_rng(3)
	names = list('ABCDE')
	means, variances = {name: rng.normal(size=3) for name in names}, {name: np.ones(3) for name in names}
	bonsai_tree = star_of(names, [1.0] * 5, means, variances)
	genes_and_cells(3, 5)
	left, right = TreeNode(ltqs=rng.normal(size=3)), TreeNode(ltqs=rng.normal(size=3))
	bonsai_tree.root.reorderChildren(left, right, maxChild=8)
	order = [names.index(child.nodeId) for child in bonsai_tree.root.childNodes]
	points = [means[name] for name in names]
	best = min(_path_cost(perm, points, left.ltqs, right.ltqs) for perm in itertools.permutations(range(5)))
	assert _path_cost(order, points, left.ltqs, right.ltqs) == pytest.approx(best, rel=1e-12)
	bonsai_tree.root.reorderChildren(left, right, maxChild=3)
	assert sorted(child.nodeId for child in bonsai_tree.root.childNodes) == names


def test_ladderize_in_main_orders_children_by_cell_count():
	means, variances = leaf_data(list('ABCDEFGH'), 2, seed=9)
	bonsai_tree = build_bonsai_tree(parse_newick('(((A:1,B:1)u:1,C:1,(D:1,E:1,F:1)t:1)x:1,G:1,H:1)r;'), means,
	                                variances)
	bonsai_tree.root.ladderize_in_main()
	for node in bonsai_nodes(bonsai_tree.root):
		counts = [len(to_node(child).leaves()) for child in node.childNodes]
		assert counts == sorted(counts)
	assert [child.nodeId for child in _by_id(bonsai_tree)['x'].childNodes] == ['C', 'u', 't']
	assert bonsai_tree.root.childNodes[-1].nodeId == 'x'


# ========================================================= files and vertex numbering

def test_get_tree_info_numbers_vertices_depth_first_and_names_internal_nodes():
	tree, bonsai_tree, _, _ = _hand_tree()
	_by_id(bonsai_tree)['x'].nodeId = None
	_by_id(bonsai_tree)['y'].nodeId = 'internal_77'
	bonsai_tree.root.renumber_verts({}, 0)
	edges, dists, vert_info, counter = bonsai_tree.root.get_tree_info([], [], {}, 0)
	assert edges == [[0, 1], [1, 2], [1, 3], [0, 4], [0, 5], [5, 6], [5, 7], [5, 8]]
	assert dists == [3.0, 1.0, 2.0, 4.0, 0.25, 0.5, 1.5, 2.5]
	assert vert_info == {0: (-1, 'root'), 1: (6, 'internal_0'), 2: (0, 'A'), 3: (1, 'B'), 4: (2, 'C'),
	                     5: (7, 'internal_1'), 6: (3, 'D'), 7: (4, 'E'), 8: (5, 'F')}
	assert counter == 2


def test_compile_tree_from_scData_tree_and_getEdgeDistVertNamesFromNode():
	_, bonsai_tree, _, _ = _hand_tree()
	_by_id(bonsai_tree)['x'].nodeId = None
	bonsai_tree.root.nodeId = None
	edges, dists, names, starry, node_ind_to_node = bonsai_tree.compile_tree_from_scData_tree()
	assert edges == [[-1, 6], [6, 0], [6, 1], [-1, 2], [-1, 7], [7, 3], [7, 4], [7, 5]]
	assert dists == [3.0, 1.0, 2.0, 4.0, 0.25, 0.5, 1.5, 2.5]
	assert names == {-1: 'root', 6: 'internal_0', 0: 'A', 1: 'B', 2: 'C', 7: 'y', 3: 'D', 4: 'E', 5: 'F'}
	assert starry is False
	assert {ind: node.nodeId for ind, node in node_ind_to_node.items()} == names
	star = star_of(list('ABCD'), [1.0] * 4, *leaf_data(list('ABCD'), 1, 0))
	assert star.compile_tree_from_scData_tree()[3] is True


def _posteriors_by_vert(folder, prefix):
	return np.load(os.path.join(folder, prefix + 'ltqs_vertByGene.npy')), \
		np.load(os.path.join(folder, prefix + 'ltqsVars_vertByGene.npy'))


@pytest.mark.parametrize('mem_friendly', [False, True])
@pytest.mark.parametrize('scaling', ['geneVariances', 'scalar'])
def test_store_tree_info_and_coords_writes_the_documented_files(tmp_path, mem_friendly, scaling, genes_and_cells):
	"""edgeInfo.txt (parent vertInd, child vertInd, length as %.8e), vertInfo.txt (vertInd, nodeInd, vertName), the
	node coordinates by vertInd, and the posterior coordinates rescaled back by the gene variances (or by their
	mean, when the data were rescaled by one number)."""
	tree, bonsai_tree, means, variances = _hand_tree()
	genes_and_cells(2, 6)
	bs_glob.mem_friendly = mem_friendly
	bonsai_tree.nNodes = 9
	bonsai_tree.root.getLtqsComplete()
	gene_vars = np.array([2.0, 0.5])
	diffusion = 'geneVariances' if scaling == 'geneVariances' else 1.7
	edges, dists, vert_info = bonsai_tree.store_tree_info_and_coords(str(tmp_path), coords_folder=str(tmp_path))
	lines = (tmp_path / 'edgeInfo.txt').read_text().splitlines()
	assert lines[0] == '0\t1\t3.00000000e+00' and lines[4] == '0\t5\t2.50000000e-01' and len(lines) == 8
	vert_lines = (tmp_path / 'vertInfo.txt').read_text().splitlines()
	assert vert_lines[0] == 'vertInd\tnodeInd\tvertName'
	assert vert_lines[1:] == ['0\t-1\troot', '1\t6\tx', '2\t0\tA', '3\t1\tB', '4\t2\tC', '5\t7\ty', '6\t3\tD',
	                          '7\t4\tE', '8\t5\tF']
	ltqs, ltqs_vars = _posteriors_by_vert(tmp_path, '')
	order = ['root', 'x', 'A', 'B', 'C', 'y', 'D', 'E', 'F']
	nodes = _by_id(bonsai_tree)
	for vert, name in enumerate(order):
		np.testing.assert_allclose(ltqs[vert], nodes[name].ltqs, rtol=1e-15)
		np.testing.assert_allclose(ltqs_vars[vert], nodes[name].getLtqsVars(), rtol=1e-15)
	bonsai_tree.store_tree_info_and_coords(str(tmp_path), coords_folder=str(tmp_path), store_posterior_ltqs=True,
	                                       geneDiffusionScaling=diffusion, variances=gene_vars)
	post_ltqs, post_vars = _posteriors_by_vert(tmp_path, 'posterior_')
	ref_means, ref_vars = reference_posteriors(tree, means, variances)
	ref_means['root'], ref_vars['root'] = ref_means['r'], ref_vars['r']
	factor = gene_vars if scaling == 'geneVariances' else 1.7
	for vert, name in enumerate(order):
		np.testing.assert_allclose(post_ltqs[vert], ref_means[name] * np.sqrt(factor), rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(post_vars[vert], ref_vars[name] * factor, rtol=1e-9)


@pytest.mark.parametrize('mem_friendly_version', [False, True])
def test_getEdgeVertInfo_numbers_vertices_in_edge_order_and_writes_coordinates(tmp_path, mem_friendly_version,
                                                                               genes_and_cells):
	tree, bonsai_tree, means, variances = _hand_tree()
	genes_and_cells(2, 6)
	bonsai_tree.nNodes = 9
	bonsai_tree.root.getLtqsComplete()
	function = bonsai_tree.getEdgeVertInfo_memfriendly if mem_friendly_version else bonsai_tree.getEdgeVertInfo
	edges, dists, vert_info = function(coords_folder=None)
	assert edges == [[0, 1], [1, 2], [1, 3], [0, 4], [0, 5], [5, 6], [5, 7], [5, 8]]
	assert [vert_info[ind][1] for ind in range(9)] == ['root', 'x', 'A', 'B', 'C', 'y', 'D', 'E', 'F']
	assert dists == [3.0, 1.0, 2.0, 4.0, 0.25, 0.5, 1.5, 2.5]
	_, bonsai_tree, _, _ = _hand_tree()
	bonsai_tree.nNodes = 9
	bonsai_tree.root.getLtqsComplete()
	edges, _, vert_info = function(coords_folder=str(tmp_path))
	ltqs, ltqs_vars = _posteriors_by_vert(tmp_path, '')
	nodes = _by_id(bonsai_tree)
	for vert in range(9):
		np.testing.assert_allclose(ltqs[vert], nodes[vert_info[vert][1]].ltqs, rtol=1e-15)
	if not mem_friendly_version:
		_, bonsai_tree, _, _ = _hand_tree()
		bonsai_tree.root.getLtqsComplete()
		gene_vars = np.array([2.0, 0.5])
		bonsai_tree.getEdgeVertInfo(coords_folder=str(tmp_path), store_posterior_ltqs=True,
		                            geneDiffusionScaling='geneVariances', variances=gene_vars)
		post_ltqs, post_vars = _posteriors_by_vert(tmp_path, 'posterior_')
		ref_means, ref_vars = reference_posteriors(tree, means, variances)
		np.testing.assert_allclose(post_ltqs[2], ref_means['A'] * np.sqrt(gene_vars), rtol=1e-9)
		np.testing.assert_allclose(post_vars[5], ref_vars['y'] * gene_vars, rtol=1e-9)


# ===================================================== branch lengths of a fixed topology

THREE_ROOT = '((A:0.5,B:0.1)u:0.2,(C:0.3,D:0.6)w:0.4,(E:0.1,F:0.3)v:0.2)r;'


def _three_root_data(seed, n_genes=8, noise=0.2):
	truth = parse_newick(THREE_ROOT)
	means, variances = tree_data(truth, n_genes, seed, noise=noise)
	return truth, means, variances


def test_getLikelihoodGradGivenTimes_matches_the_reference_and_its_derivatives(genes_and_cells):
	truth, means, variances = _three_root_data(0)
	bonsai_tree = build(THREE_ROOT, means, variances)
	genes_and_cells(8, 6)
	node_inds = [node.nodeInd for node in bonsai_nodes(bonsai_tree.root)[1:]]
	rng = np.random.default_rng(1)
	times = rng.uniform(0.1, 1.0, len(node_inds))

	def ref(t):
		lengths = dict(zip(node_inds, t))
		return reference_bonsai_loglik(with_lengths(truth, [lengths[ind] for ind in node_inds]), means, variances)

	for mem_friendly in (False, True):
		loglik, grad = bonsai_tree.getLikelihoodGradGivenTimes(times, node_inds, mem_friendly=mem_friendly)
		assert loglik == pytest.approx(ref(times), rel=1e-10)
		np.testing.assert_allclose(grad, central_difference(ref, times), rtol=1e-5, atol=1e-8)
	value, grad = bonsai_tree.getNegLikelihoodGradGivenLogTimesWrapper(np.log(times), node_inds)
	assert value == pytest.approx(-ref(times), rel=1e-10)
	np.testing.assert_allclose(grad, central_difference(lambda lt: -ref(np.exp(lt)), np.log(times)), rtol=1e-5,
	                           atol=1e-8)
	value, grad = bonsai_tree.getNegLikelihoodGradGivenLogTimesWrapper_single_scalar(0.3, np.log(times), node_inds)
	assert value == pytest.approx(-ref(np.exp(0.3) * times), rel=1e-10)
	assert grad == pytest.approx(central_difference(lambda s: -ref(np.exp(s[0]) * times), np.array([0.3]))[0],
	                             rel=1e-5)


@pytest.mark.parametrize('seed', range(2))
def test_optTimes_reaches_the_reference_maximum(seed, genes_and_cells):
	truth, means, variances = _three_root_data(seed, n_genes=10, noise=0.1)
	bonsai_tree = build(write_newick(with_lengths(truth, [1.0] * 9)), means, variances)
	genes_and_cells(10, 6)
	before = bonsai_tree.calcLogLComplete()
	times, n_iterations = bonsai_tree.optTimes(singleProcess=True, return_diagnostics=True)
	assert isinstance(n_iterations, int) and n_iterations > 0
	assert set(times) == {node.nodeInd for node in bonsai_nodes(bonsai_tree.root)[1:]}
	assert all(node.tParent == times[node.nodeInd] for node in bonsai_nodes(bonsai_tree.root)[1:])
	loglik = bonsai_tree.calcLogLComplete()
	assert loglik == pytest.approx(reference_of(bonsai_tree, means, variances), rel=1e-10)
	assert bonsai_tree.root.prefactor == pytest.approx(loglik)
	best, lengths = optimise_lengths(truth, means, variances)
	assert loglik >= before
	assert loglik == pytest.approx(best, abs=1e-4)
	got = [node.tParent for node in bonsai_nodes(bonsai_tree.root)[1:]]
	np.testing.assert_allclose(got, lengths, atol=5e-3)
	assert isinstance(bonsai_tree.optTimes(singleProcess=True), dict)


def test_optTimes_sets_vanishing_branch_lengths_to_exactly_zero(genes_and_cells):
	"""Two identical cells: the optimum puts them at their parent, and lengths under 2e-6 are stored as 0."""
	names = ['A', 'B', 'C', 'D']
	means, variances = leaf_data(names, 6, seed=3)
	means['B'] = means['A'].copy()
	variances['B'] = variances['A'].copy()
	bonsai_tree = build('((A:0.5,B:0.5)u:0.5,C:1.0,D:1.0)r;', means, variances)
	genes_and_cells(6, 4)
	times = bonsai_tree.optTimes(singleProcess=True)
	by_name = {node.nodeId: node.tParent for node in bonsai_nodes(bonsai_tree.root)[1:]}
	assert by_name['A'] == 0.0 and by_name['B'] == 0.0
	assert all(value == 0.0 or value >= 2e-6 for value in times.values())


def test_optTimes_single_scalar_rescales_all_lengths_by_the_best_factor(genes_and_cells):
	truth, means, variances = _three_root_data(4, n_genes=6, noise=0.1)
	start = with_lengths(truth, np.linspace(0.2, 2.0, 9))
	bonsai_tree = build(write_newick(start), means, variances)
	genes_and_cells(6, 6)
	bonsai_tree.optTimes_single_scalar(singleProcess=True, tol=1e-10)
	t0 = np.linspace(0.2, 2.0, 9)
	best = minimize_scalar(lambda s: -reference_bonsai_loglik(with_lengths(truth, np.exp(s) * t0), means, variances),
	                       bounds=(-5, 5), method='bounded', options={'xatol': 1e-10})
	got = np.array([node.tParent for node in bonsai_nodes(bonsai_tree.root)[1:]])
	np.testing.assert_allclose(got, np.exp(best.x) * t0, rtol=1e-4)


def test_test_for_zero_times_keeps_the_cutoff_with_the_highest_likelihood(genes_and_cells):
	"""Branch lengths under 1e-6, 1e-5, 1e-4 or 1e-3 are set to zero if that raises the likelihood most; the
	expected choice is computed on the reference."""
	names = list('ABCDE')
	means, variances = leaf_data(names, 4, seed=5, var_range=(0.001, 0.01))
	means['B'] = means['A'] + 1e-4
	# D is a copy of C, so its edge of 5e-4 is better at zero: the largest cutoff wins.
	means['D'], variances['D'] = means['C'].copy(), variances['C'].copy()
	lengths = [5e-7, 3e-6, 4e-5, 5e-4, 0.4, 0.8, 0.7]
	tree = with_lengths(parse_newick('((A:1,B:1)u:1,(C:1,D:1)w:1,E:1)r;'), [lengths[ind] for ind in (5, 0, 1, 6, 2,
	                                                                                                 3, 4)])
	bonsai_tree = build(write_newick(tree), means, variances)
	genes_and_cells(4, 5)
	edge_lengths = np.array([node.length for node in edges_of(tree)])
	best_cutoff, best_loglik = None, -np.inf
	for cutoff in (None, 1e-6, 1e-5, 1e-4, 1e-3):
		candidate = edge_lengths.copy()
		if cutoff is not None:
			candidate[candidate < cutoff] = 0.0
		loglik = reference_bonsai_loglik(with_lengths(tree, candidate), means, variances)
		if loglik > best_loglik:
			best_cutoff, best_loglik = cutoff, loglik
	assert best_cutoff == 1e-3
	bonsai_tree.test_for_zero_times(singleProcess=True)
	expected = edge_lengths.copy()
	if best_cutoff is not None:
		expected[expected < best_cutoff] = 0.0
	np.testing.assert_array_equal([node.tParent for node in bonsai_nodes(bonsai_tree.root)[1:]], expected)
	assert bonsai_tree.root.prefactor == pytest.approx(best_loglik, rel=1e-10)


def test_optimiseConnected_optimises_the_root_edges(genes_and_cells):
	import bonsai.mpi_wrapper as mpi_wrapper
	ltqs, ltqs_vars, _, names, means, variances = _star_inputs(5, 6, seed=0)
	bonsai_tree = star_of(names, [1.0] * 5, means, variances)
	genes_and_cells(6, 5)
	done, W_g, xr_g = bonsai_tree.root.optimiseConnected(mpi_wrapper.get_mpi_info(singleProcess=True),
	                                                     singleProcess=True)
	t_star, _, expected_W, expected_xr = th.optimiseTStar(ltqs, ltqs_vars)
	assert done is True
	np.testing.assert_allclose([child.tParent for child in bonsai_tree.root.childNodes], t_star, rtol=1e-12)
	np.testing.assert_allclose(W_g, expected_W, rtol=1e-12)
	np.testing.assert_allclose(bonsai_tree.root.ltqs, xr_g, rtol=1e-12)
	assert bonsai_tree.root.prefactor == pytest.approx(_star_ref(names, t_star, means, variances), rel=1e-10)


# ============================================================ greedy merging

def _greedy_star(truth, n_genes, seed, noise=0.1):
	import bonsai.mpi_wrapper as mpi_wrapper
	means, variances = tree_data(truth, n_genes, seed, noise=noise)
	names = truth.leaf_names()
	bonsai_tree = star_of(names, [1.0] * len(names), means, variances)
	bs_glob.nGenes, bs_glob.nCells = n_genes, len(names)
	bonsai_tree.root.optimiseConnected(mpi_wrapper.get_mpi_info(singleProcess=True), singleProcess=True)
	return bonsai_tree, means, variances


def _clade_lengths(node_tree):
	return {frozenset(node.leaf_names()): node.length for node in node_tree.iter_nodes() if node is not node_tree}


BALANCED8 = balanced_newick(3, internal_length=1.0, leaf_length=0.1)


@pytest.mark.parametrize('seed', range(3))
def test_mergeChildrenUB_turns_a_star_into_the_true_binary_tree(seed):
	truth = parse_newick(BALANCED8)
	bonsai_tree, means, variances = _greedy_star(truth, 20, seed)
	star_loglik = bonsai_tree.calcLogLComplete()
	changed = bonsai_tree.root.mergeChildrenUB(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), sequential=True,
	                                           ellipsoidSize=None, singleProcess=True)
	assert changed is True
	result = to_node(bonsai_tree.root)
	assert rf_distance(result, truth) == 0
	assert len(result.children) == 3 and all(len(node.children) in (0, 2) for node in edges_of(result))
	loglik = bonsai_tree.calcLogLComplete()
	assert loglik == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-10)
	assert loglik > star_loglik
	# The ltqs the merge left at every node are the downstream messages a full recomputation gives.
	kept = {node.nodeInd: (node.ltqs.copy(), node.getLtqsVars().copy()) for node in bonsai_nodes(bonsai_tree.root)}
	bonsai_tree.root.getLtqsComplete()
	for node in bonsai_nodes(bonsai_tree.root):
		np.testing.assert_allclose(kept[node.nodeInd][0], node.ltqs, rtol=1e-8, atol=1e-10)
		np.testing.assert_allclose(kept[node.nodeInd][1], node.getLtqsVars(), rtol=1e-8)
	new_inds = sorted(node.nodeInd for node in bonsai_nodes(bonsai_tree.root) if not node.isLeaf and not node.isRoot)
	assert new_inds == list(range(8, 8 + len(new_inds))) and bs_glob.max_node_ind == new_inds[-1]


@pytest.mark.parametrize('seed', range(3))
def test_mergeChildrenUB_with_upper_bounds_gives_the_exhaustive_result(seed):
	"""The upper bounds only decide which pairs need not be evaluated (YAML comment of UB_ellipsoid_size: "does not
	affect the final result"): on these 8-cell sets the tree is the one an exhaustive search over all pairs builds.
	On larger sets Bonsai's greedy trees with and without the upper bounds do differ, contrary to that comment."""
	truth = parse_newick(BALANCED8)
	exhaustive, means, variances = _greedy_star(truth, 20, seed, noise=0.1)
	exhaustive.root.mergeChildrenUB(exhaustive.root.ltqs, exhaustive.root.getW(), ellipsoidSize=None,
	                                singleProcess=True)
	for ellipsoid in (1.0, 0.2):
		bounded, _, _ = _greedy_star(truth, 20, seed, noise=0.1)
		bounded.root.mergeChildrenUB(bounded.root.ltqs, bounded.root.getW(), ellipsoidSize=ellipsoid, nChildUB=2,
		                             singleProcess=True)
		got, expected = _clade_lengths(to_node(bounded.root)), _clade_lengths(to_node(exhaustive.root))
		assert set(got) == set(expected)
		for clade, length in expected.items():
			assert got[clade] == pytest.approx(length, rel=1e-9, abs=1e-12)


# The greedy trees of 32 cells (balanced tree, 15 genes, noise 0.15) with upper bounds refreshed down to two
# children, and with one nearest neighbour per child as candidates. Pinned from Bonsai at b16ef6f: on these sets
# the upper bounds and the neighbour lists do change the tree, although the YAML comment says they do not,
# so any change to how they are computed or refreshed shows up here.
PINNED_GREEDY = {
	'upper_bounds': (1, dict(ellipsoidSize=1.0, nChildUB=2),
	                 '(((C00010:0.12488711964610888,C00011:0.026947156026923075):0.22310646991702487,'
	                 '((C00100:0.09887743281366514,C00101:0.03334766072542404):0.7566681510878017,'
	                 '(C00110:0.07456552472178028,'
	                 'C00111:0.13874410197351267):0.5255501976397556):0.46698848906893375):0.30527430136411343,'
	                 '(((C10000:0.06971206816662485,(C10001:0.07682827818430016,(C10010:0.0,'
	                 'C10011:0.17036993437890924):1.1686765046339629):0.06309153399709966):0.34545797222784946,'
	                 '((C10110:0.0,C10111:0.30275719793178296):0.36672714319780414,(C10100:0.2361066426114625,'
	                 'C10101:0.08791998035581236):0.7258441591654823):0.6059191125901804):1.2682834566833618,'
	                 '((C11110:0.03269647948080668,C11111:0.3041724384624209):0.39389054775446264,'
	                 '((C11100:0.2699472925879203,C11101:0.0):0.46748034229108865,((C11010:0.07300182840655216,'
	                 'C11011:0.0036414619565225265):0.48750693147376484,(C11000:0.06025158059055962,'
	                 'C11001:0.20888924909405715):0.36704892370181436):0.2616879764044108):0.27519421503423):0.28433641017352507):0.5721383462526013,'
	                 '((C01011:0.0,(C01010:0.15989277876298877,(C01000:0.10349483126108942,'
	                 'C01001:0.0):0.5729418568381975):0.01603027285909803):0.5136820164547435,((C01111:0.0,'
	                 '(C01110:0.06410409179922281,(C01100:0.0,'
	                 'C01101:0.17072963127880392):0.5941153715952667):0.02623933012477055):0.6677764884633451,'
	                 '(C00000:0.22267406421081745,'
	                 'C00001:0.0):0.3351164687930094):0.20643764909918327):0.10709222816505057)root;',
	                 -124.70983656981839),
	'one_neighbour': (0, dict(ellipsoidSize=1.0, nChildNN=4, kNN=1),
	                  '(((C11110:0.06556315416912319,C11111:0.0):0.44223235032733804,(C11100:0.08286006623202598,'
	                  'C11101:0.0):0.5712771613816459):0.9413109600084888,((((C10100:0.19989164136361018,'
	                  'C10101:0.0):0.3436954356883694,(C10110:0.30420099137431444,'
	                  'C10111:0.02633660174387803):0.5718931274338075):0.7111833706495028,((C10010:0.027618290488174532,'
	                  'C10011:0.07303499364903178):0.7087912648029763,(C10000:0.02607054321047223,'
	                  'C10001:0.10654928535077873):0.20617150195966194):0.2350095183643165):0.5520736698554715,'
	                  '((C11000:0.0,C11001:0.06224082856192766):0.42004971451518036,(C11010:0.06543667868714595,'
	                  'C11011:0.051557412334007416):0.7284924809458976):0.27738539079363467):0.10521389777653599,'
	                  '(((C01100:0.08300695522394086,C01101:0.13618207539944047):0.26338910476398025,'
	                  '((C01110:0.011360857140528952,C01111:0.23398994760530717):0.5631973231566616,'
	                  '((C01010:0.11503556692133568,C01011:0.029376597740177396):0.1679762261878646,'
	                  '(C01000:0.3118139849763114,'
	                  'C01001:0.0):0.2553647265204486):1.1139028985353137):0.009765596272021592):0.6910097264177383,'
	                  '((C00110:0.0,C00111:0.12294062453594318):0.3677462078906615,((C00100:0.07935835757660699,'
	                  'C00101:0.261062857674936):0.4220831955713366,((C00000:0.0872898502802363,'
	                  'C00001:0.1001348995889348):0.28422584099563897,(C00010:0.31364742648557187,'
	                  'C00011:0.0):0.41595944869296897):0.34725778216934927):0.21965848880990957):0.3490381729700137):0.3816639300686359)root;',
	                  -102.79970429474412),
}


@pytest.mark.parametrize('case', sorted(PINNED_GREEDY))
def test_mergeChildrenUB_pins_the_current_greedy_tree(case):
	"""Pins the current output: rooted clades (zero-length edges kept), branch lengths and log-likelihood of the
	greedy tree with upper bounds or with nearest-neighbour candidates."""
	seed, options, newick, loglik = PINNED_GREEDY[case]
	truth = parse_newick(balanced_newick(5, internal_length=0.5, leaf_length=0.1))
	bonsai_tree, means, variances = _greedy_star(truth, 15, seed, noise=0.15)
	bonsai_tree.root.mergeChildrenUB(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), singleProcess=True, **options)
	result, expected = to_node(bonsai_tree.root), parse_newick(newick)
	assert clades(result, min_length=-1) == clades(expected, min_length=-1)
	got_lengths = _clade_lengths(result)
	for clade, length in _clade_lengths(expected).items():
		assert got_lengths[clade] == pytest.approx(length, rel=1e-9, abs=1e-12)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(loglik, rel=1e-9)


def test_mergeChildrenUB_with_nearest_neighbour_candidates(tmp_path):
	"""use_knn: only nearest-neighbour pairs are candidates while more than nChildNN children are left; on clearly
	separated cells the true tree is still found."""
	truth = parse_newick(balanced_newick(4, internal_length=1.0, leaf_length=0.1))
	bonsai_tree, means, variances = _greedy_star(truth, 20, seed=1)
	bonsai_tree.root.mergeChildrenUB(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), ellipsoidSize=1.0, nChildNN=4,
	                                 kNN=3, nChildUB=4, singleProcess=True, outputFolder=str(tmp_path),
	                                 n_mem_friendly=4)
	result = to_node(bonsai_tree.root)
	assert rf_distance(result, truth) == 0
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-10)
	# Temporary trees and the folder for exchanging data between processes are removed again.
	assert os.listdir(str(tmp_path)) == []


def test_mergeChildrenUB_joint_time_optimisation_and_random_merges():
	truth = parse_newick(BALANCED8)
	bonsai_tree, means, variances = _greedy_star(truth, 20, seed=2)
	bonsai_tree.root.mergeChildrenUB(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), sequential=False,
	                                 ellipsoidSize=1.0, singleProcess=True)
	result = to_node(bonsai_tree.root)
	assert rf_distance(result, truth) == 0
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-10)
	np.random.seed(5)
	random_tree, means, variances = _greedy_star(truth, 20, seed=2)
	star_loglik = random_tree.calcLogLComplete()
	random_tree.root.mergeChildrenUB(random_tree.root.ltqs, random_tree.root.getW(), random=True, singleProcess=True)
	result = to_node(random_tree.root)
	assert sorted(result.leaf_names()) == sorted(truth.leaf_names())
	assert len(result.children) <= 3
	loglik = random_tree.calcLogLComplete()
	assert loglik == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-10)
	assert loglik > star_loglik


def test_mergeChildrenUB_leaves_a_root_with_three_children_alone():
	means, variances = leaf_data(list('ABC'), 3, seed=0)
	bonsai_tree = star_of(list('ABC'), [0.5, 0.5, 0.5], means, variances)
	bs_glob.nGenes, bs_glob.nCells = 3, 3
	before = write_newick(to_node(bonsai_tree.root))
	assert bonsai_tree.root.mergeChildrenUB(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), singleProcess=True) is False
	assert write_newick(to_node(bonsai_tree.root)) == before


def test_mergeChildrenRecursive_resolves_polytomies_below_the_root():
	truth = parse_newick('(((A:0.1,B:0.1)p:1.0,(C:0.1,D:0.1)q:1.0)x:1.0,((E:0.1,F:0.1)s:1.0,G:0.1)y:1.0,H:1.5)r;')
	means, variances = tree_data(truth, 20, seed=3, noise=0.1)
	bonsai_tree = build('((A:1.1,B:1.1,C:1.1,D:1.1)x:1.0,(E:1.1,F:1.1,G:0.1)y:1.0,H:1.5)r;', means, variances)
	bs_glob.nGenes, bs_glob.nCells = 20, 8
	bonsai_tree.optTimes(singleProcess=True)
	before = bonsai_tree.calcLogLComplete()
	bonsai_tree.root.storeParent()
	bonsai_tree.root.getAIRootInfo(None, None)
	merged, ltqs, ltqs_vars, _, _ = bonsai_tree.root.mergeChildrenRecursive(
		bonsai_tree.root.ltqs, bonsai_tree.root.getW(), ellipsoidSize=1.0, single_process=True, tree=bonsai_tree)
	assert merged is True
	result = to_node(bonsai_tree.root)
	assert rf_distance(result, truth) == 0
	after = bonsai_tree.calcLogLComplete()
	assert after > before
	assert after == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-10)
	# A binary tree has nothing left to merge.
	again, _, _, _, _ = bonsai_tree.root.mergeChildrenRecursive(bonsai_tree.root.ltqs, bonsai_tree.root.getW(),
	                                                            single_process=True)
	assert again is False
	assert write_newick(to_node(bonsai_tree.root)) == write_newick(result)


def test_mergeNodes_adds_a_new_ancestor_and_updates_the_root(genes_and_cells):
	names = list('ABCDE')
	means, variances = leaf_data(names, 3, seed=6)
	bonsai_tree = star_of(names, [0.4, 0.5, 0.6, 0.7, 0.8], means, variances)
	genes_and_cells(3, 5)
	bs_glob.max_node_ind = 4
	index = {child.nodeInd: ind for ind, child in enumerate(bonsai_tree.root.childNodes)}
	xr, W, new_node, deleted = bonsai_tree.root.mergeNodes(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), 0, 1,
	                                                       {0: 0.15, 1: 0.25, 5: 0.35}, index, 1.0,
	                                                       runConfigs={'useNN': False})
	assert new_node.nodeInd == 5 and bs_glob.max_node_ind == 5 and sorted(deleted) == [0, 1]
	result = to_node(bonsai_tree.root)
	expected = parse_newick('((A:0.15,B:0.25):0.35,C:0.6,D:0.7,E:0.8)root;')
	assert clades(result) == clades(expected)
	assert _clade_lengths(result) == _clade_lengths(expected)
	post_means, post_vars = reference_posteriors(result, means, variances)
	np.testing.assert_allclose(xr, post_means['root'], rtol=1e-9)
	np.testing.assert_allclose(1 / W, post_vars['root'], rtol=1e-9)
	np.testing.assert_allclose(new_node.getLtqsVars(), 1 / (1 / (0.15 + variances['A']) + 1 / (0.25 + variances['B'])))


def test_mergeNodes_adds_a_child_below_an_ancestor_at_zero_distance(genes_and_cells):
	"""When the optimal time from the root-side node to the new ancestor is zero (and that node is not a cell), the
	other node becomes its child instead of getting a new ancestor; the moved node may then merge further down."""
	means, variances = leaf_data(list('ABCDE'), 3, seed=7)
	bonsai_tree = build('((A:0.3,B:0.4)p:0.5,C:0.6,D:0.7,E:0.8)r;', means, variances)
	genes_and_cells(3, 5)
	bs_glob.max_node_ind = 5
	p = _by_id(bonsai_tree)['p']
	index = {child.nodeInd: ind for ind, child in enumerate(bonsai_tree.root.childNodes)}
	xr, W, new_node, deleted = bonsai_tree.root.mergeNodes(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), p.nodeInd, 2,
	                                                       {p.nodeInd: 0.0, 2: 0.25, 6: 0.45}, index, 1.0,
	                                                       mergeDownstream=False, runConfigs={'useNN': False})
	assert new_node is p and bs_glob.max_node_ind == 5 and deleted == [2]
	result = to_node(bonsai_tree.root)
	expected = parse_newick('((A:0.3,B:0.4,C:0.25)p:0.45,D:0.7,E:0.8)root;')
	assert _clade_lengths(result) == _clade_lengths(expected)
	post_means, _ = reference_posteriors(result, means, variances)
	np.testing.assert_allclose(xr, post_means['root'], rtol=1e-9)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(expected, means, variances),
	                                                       rel=1e-10)


def test_mergeNodes_child_merge_continues_downstream(genes_and_cells):
	"""After C is placed under p, mergeDownstreamChildrenSingle lets C pair with A inside p when that raises the
	likelihood (C is a near copy of A)."""
	means, variances = leaf_data(list('ABCDE'), 6, seed=8, var_range=(0.01, 0.02))
	means['C'] = means['A'] + 0.01
	bonsai_tree = build('((A:0.3,B:0.4)p:0.5,C:0.6,D:0.7,E:0.8)r;', means, variances)
	genes_and_cells(6, 5)
	bs_glob.max_node_ind = 5
	p = _by_id(bonsai_tree)['p']
	bonsai_tree.root.storeParent()
	index = {child.nodeInd: ind for ind, child in enumerate(bonsai_tree.root.childNodes)}
	bonsai_tree.root.mergeNodes(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), p.nodeInd, 2,
	                            {p.nodeInd: 0.0, 2: 0.25, 6: 0.45}, index, 1.0, mergeDownstream=True,
	                            runConfigs={'useNN': False, 'outputFolder': None})
	result = to_node(bonsai_tree.root)
	assert frozenset('AC') in clades(result) and frozenset('ABC') in clades(result)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(result, means, variances),
	                                                       rel=1e-10)


def test_resetChildTimes_sets_two_branch_lengths_and_updates_the_root(genes_and_cells):
	names = list('ABCD')
	means, variances = leaf_data(names, 3, seed=9)
	bonsai_tree = star_of(names, [0.4, 0.5, 0.6, 0.7], means, variances)
	genes_and_cells(3, 4)
	index = {child.nodeInd: ind for ind, child in enumerate(bonsai_tree.root.childNodes)}
	xr, W = bonsai_tree.root.resetChildTimes(bonsai_tree.root.ltqs, bonsai_tree.root.getW(), 1, 3, {1: 0.05, 3: 2.0},
	                                         index)
	assert [child.tParent for child in bonsai_tree.root.childNodes] == [0.4, 0.05, 0.6, 2.0]
	post_means, post_vars = reference_posteriors(star_tree(names, [0.4, 0.05, 0.6, 2.0]), means, variances)
	np.testing.assert_allclose(xr, post_means['root'], rtol=1e-9)
	np.testing.assert_allclose(1 / W, post_vars['root'], rtol=1e-9)


def test_getOptimalDLogLPairInfoNew_picks_the_best_pair_or_a_time_optimisation(genes_and_cells):
	names = list('ABCDE')
	means, variances = leaf_data(names, 3, seed=10)
	bonsai_tree = star_of(names, [0.4, 0.5, 0.6, 0.7, 0.8], means, variances)
	genes_and_cells(3, 5)
	bs_glob.max_node_ind = 4
	root = bonsai_tree.root
	index = {child.nodeInd: ind for ind, child in enumerate(root.childNodes)}
	info = {'doTimeOpt': False, 'timeOptRoundsAgo': 0}
	ind1, ind2, times, info, best = root.getOptimalDLogLPairInfoNew(
		root.ltqs, root.getW(), [(0, 1, 2.0), (2, 4, 3.5)], [(-1, -1, 1e-9)], index, info, verbose=False)
	assert (ind1, ind2, best) == (2, 4, 3.5) and info == {'doTimeOpt': False, 'timeOptRoundsAgo': 1}
	expected = th.getOptTimesSingleDLogLWrapper(root.ltqs, root.getW(), root.childNodes[2], root.childNodes[4],
	                                            tol=1e-9)
	assert set(times) == {2, 4, 5}
	assert [times[2], times[4], times[5]] == pytest.approx(list(expected), rel=1e-6)
	# A better merge whose ancestor would sit on the root: re-optimise the root's edges instead.
	ind1, ind2, times, info, _ = root.getOptimalDLogLPairInfoNew(
		root.ltqs, root.getW(), [(2, 4, 3.5)], [(0, 1, 5.0)], index, info, verbose=False)
	assert (ind1, ind2, times) == (None, None, None) and info == {'doTimeOpt': True, 'timeOptRoundsAgo': 0}
	# Not right after a time optimisation, and nothing that raises the likelihood: no merge.
	ind1, _, _, info, _ = root.getOptimalDLogLPairInfoNew(root.ltqs, root.getW(), [(-1, -1, 1e-9)], [(0, 1, 5.0)],
	                                                      index, info, verbose=False)
	assert ind1 is None and info['doTimeOpt'] is False
	# Random merges pick among the pairs that raise the likelihood, weighted by exp(dLogL).
	np.random.seed(0)
	picks = {root.getOptimalDLogLPairInfoNew(root.ltqs, root.getW(), [(0, 1, 1.0)], [(-1, -1, 1e-9)], index,
	                                         {'doTimeOpt': False, 'timeOptRoundsAgo': 1}, random=True,
	                                         dLogLDict={(0, 1): 40.0, (2, 3): -3.0, (1, 4): 1.0})[:2]
	         for _ in range(5)}
	assert picks == {(0, 1)}


def test_getNNPairs_pairs_every_child_with_its_nearest_neighbours(genes_and_cells):
	"""Candidate pairs are the k nearest neighbours of each child of the root, by Euclidean distance between the
	normalised posterior positions (relative to the root's) of the children, computed here from the reference."""
	from scipy.sparse import lil_array
	rng = np.random.default_rng(11)
	names = ['L%d' % ind for ind in range(9)]
	means, variances = {n: rng.normal(size=4) for n in names}, {n: rng.uniform(0.05, 0.2, 4) for n in names}
	times = rng.uniform(0.2, 1.0, 9)
	bonsai_tree = star_of(names, times, means, variances)
	genes_and_cells(4, 9)
	root = bonsai_tree.root
	post_means, _ = reference_posteriors(star_tree(names, times), means, variances)
	points = np.array([post_means[name] - post_means['root'] for name in names])
	points /= np.linalg.norm(points, axis=1)[:, None]
	distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
	expected = set()
	for ind in range(9):
		for other in np.argsort(distances[ind])[:3]:
			if other != ind:
				expected.add(tuple(sorted((ind, int(other)))))
	nn_info = {'NNcounter': 0, 'conn_mat': lil_array((18, 18), dtype=bool), 'subtracted_mean': None}
	pairs, nn_info = root.getNNPairs(root.ltqs, root.getLtqsVars(), nn_info, kNN=2)
	assert set(pairs) == expected
	got_posterior = root.get_posterior_info_children(root.ltqs, root.getW())
	np.testing.assert_allclose(got_posterior, np.array([post_means[name] for name in names]).T, rtol=1e-9)
	# Children closer to the root than minimum_time are moved out to it first.
	tiny = star_of(names, np.full(9, 1e-7), means, variances)
	clamped = tiny.root.get_posterior_info_children(tiny.root.ltqs, tiny.root.getW(), minimum_time=1e-4)
	reference = star_of(names, np.full(9, 1e-4), means, variances)
	expected_clamped = th.getLtqsAsIfRoot_vectorized(
		np.array([c.ltqs for c in reference.root.childNodes]).T, np.array([c.getW() for c in reference.root.childNodes]).T,
		np.full(9, 1e-4), tiny.root.ltqs, tiny.root.getW())[0]
	np.testing.assert_allclose(clamped, expected_clamped, rtol=1e-12)


# ============================================================ subtree prune and regraft

SPR_TRUTH = '(((C0:0.1,C1:0.1)a:1.0,(C2:0.1,C3:0.1)b:1.0)x:0.5,((C4:0.1,C5:0.1)c:1.0,(C6:0.1,C7:0.1)d:1.0)y:0.5,' \
            'C8:1.0)r;'
# C0 misplaced next to C7.
SPR_WRONG = '((C1:1.2,(C2:0.1,C3:0.1)b:1.0)x:0.5,((C4:0.1,C5:0.1)c:1.0,(C6:0.1,(C7:0.1,C0:1.5)e:0.1)d:1.0)y:0.5,' \
            'C8:1.0)r;'


def _spr_setup(newick=SPR_WRONG, seed=0, n_genes=20):
	truth = parse_newick(SPR_TRUTH)
	means, variances = tree_data(truth, n_genes, seed, noise=0.05)
	bonsai_tree = build_bonsai_tree(parse_newick(newick), means, variances)
	bs_glob.nGenes, bs_glob.nCells = n_genes, 9
	bonsai_tree.optTimes(singleProcess=True)
	bonsai_tree.root.storeParent()
	bonsai_tree.root.get_ds_node_counts()
	bonsai_tree.nNodes = bs_glob.nNodes = len(bonsai_nodes(bonsai_tree.root))
	return truth, bonsai_tree, means, variances


def _without(node_tree, name):
	"""A copy of a Node tree without the named leaf (its parent keeps the remaining children)."""
	copy = node_tree.copy()
	parent = copy.parents()[id(copy.find(name))]
	parent.children = [child for child in parent.children if child.name != name]
	return copy


def _attach(node_tree, target, name, length):
	"""A copy of a Node tree with a new leaf under target; under a leaf target, through a zero-length twin parent."""
	copy = node_tree.copy()
	node = copy.find(target)
	if node.is_leaf:
		parent = copy.parents()[id(node)]
		twin = Node('twin', node.length, [Node(node.name, 0.0), Node(name, length)])
		parent.children = [twin if child is node else child for child in parent.children]
	else:
		node.children.append(Node(name, length))
	return copy


def _best_attachment(base, name, means, variances):
	"""Over every node of base, the reference log-likelihood after attaching the leaf at its best branch length."""
	out = {}
	for node in base.iter_nodes():
		res = minimize_scalar(lambda t: -reference_bonsai_loglik(_attach(base, node.name, name, t), means, variances),
		                      bounds=(0.0, 20.0), method='bounded', options={'xatol': 1e-9})
		out[node.name] = -res.fun
	return out


def test_detach_and_add_subtree_keep_every_message_up_to_date():
	truth, bonsai_tree, means, variances = _spr_setup()
	bonsai_tree.root.get_ds_node_counts()
	nodes = _by_id(bonsai_tree)
	full = to_node(bonsai_tree.root)
	leaf = nodes['C0']
	parent = leaf.parentNode
	root_count = bonsai_tree.root.n_ds_nodes
	ind, orig_t, ltqs, ltqs_vars = parent.detach_subtree(leaf)
	assert ind == leaf.nodeInd and orig_t == full.find('C0').length and leaf.parentNode is None
	np.testing.assert_array_equal(ltqs, means['C0'])
	assert bonsai_tree.root.n_ds_nodes == root_count - 1
	reduced = _without(full, 'C0')
	post_means, post_vars = reference_posteriors(to_node(bonsai_tree.root), means, variances)
	for node in bonsai_nodes(bonsai_tree.root):
		if node.isLeaf:
			continue
		sub_means, sub_vars = reference_posteriors(to_node(node), means, variances)
		np.testing.assert_allclose(node.ltqs, sub_means[node.nodeId], rtol=1e-8, atol=1e-10)
		np.testing.assert_allclose(node.getLtqsVars(), sub_vars[node.nodeId], rtol=1e-8)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(reduced, means, variances), rel=1e-9)
	target = nodes['b']
	target.add_subtree(leaf, 0.3, ltqs, ltqs_vars)
	assert leaf.parentNode is target and leaf.tParent == 0.3 and bonsai_tree.root.n_ds_nodes == root_count
	expected = _attach(reduced, 'b', 'C0', 0.3)
	sub_means, _ = reference_posteriors(to_node(bonsai_tree.root), means, variances)
	np.testing.assert_allclose(bonsai_tree.root.ltqs, reference_posteriors(expected, means, variances)[0]['root'],
	                           rtol=1e-8)
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(expected, means, variances),
	                                                       rel=1e-9)


def test_get_dlogl_spr_move_is_the_gain_of_attaching_at_the_best_length():
	truth, bonsai_tree, means, variances = _spr_setup()
	nodes = _by_id(bonsai_tree)
	full = to_node(bonsai_tree.root)
	leaf = nodes['C0']
	leaf.parentNode.detach_subtree(leaf)
	reduced = _without(full, 'C0')
	without = reference_bonsai_loglik(reduced, means, variances)
	best = _best_attachment(reduced, 'C0', means, variances)
	for name in ('c', 'b', 'x', 'C3', 'root'):
		dlogl, opt_t = nodes[name].get_dlogl_spr_move(means['C0'], variances['C0'], as_if_root_version=1,
		                                              spr_target_version=1)
		attached = reference_bonsai_loglik(_attach(reduced, name, 'C0', opt_t), means, variances)
		assert dlogl == pytest.approx(attached - without, rel=1e-9)
		assert attached == pytest.approx(best[name], abs=1e-6)
	# Asking again at the same versions gives the same answer (whether it is cached is not pinned).
	again, again_t = nodes['c'].get_dlogl_spr_move(means['C0'], variances['C0'], as_if_root_version=1,
	                                               spr_target_version=1)
	first, first_t = nodes['c'].get_dlogl_spr_move(means['C0'], variances['C0'], as_if_root_version=1,
	                                               spr_target_version=None)
	assert again == pytest.approx(first, rel=1e-12) and again_t == pytest.approx(first_t, rel=1e-12)


def test_do_spr_search_with_a_wide_tolerance_finds_the_best_attachment_point():
	truth, bonsai_tree, means, variances = _spr_setup()
	nodes = _by_id(bonsai_tree)
	full = to_node(bonsai_tree.root)
	leaf = nodes['C0']
	leaf.parentNode.detach_subtree(leaf)
	reduced = _without(full, 'C0')
	best = _best_attachment(reduced, 'C0', means, variances)
	best_name = max(best, key=best.get)
	for start in ('d', 'C8'):
		opt_node, opt_dlogl, opt_t, count = nodes[start].do_spr_search(means['C0'], variances['C0'], prev_dlogl=-np.inf,
		                                                               opt_dlogl=-np.inf, as_if_root_version=1,
		                                                               spr_target_version=None, search_tol=1e9)
		assert opt_node.nodeId == best_name
		assert count == len(list(reduced.iter_nodes()))
		assert opt_dlogl + reference_bonsai_loglik(reduced, means, variances) == pytest.approx(best[best_name],
		                                                                                         abs=1e-6)


def test_get_twin_parent_inserts_a_copy_of_a_leaf_above_it(genes_and_cells):
	truth, bonsai_tree, means, variances = _spr_setup()
	bonsai_tree.root.get_ds_node_counts()
	nodes = _by_id(bonsai_tree)
	leaf = nodes['C3']
	old_parent, old_t = leaf.parentNode, leaf.tParent
	bs_glob.max_node_ind, bs_glob.nNodes = 30, 17
	root_count = bonsai_tree.root.n_ds_nodes
	twin = leaf.get_twin_parent()
	assert twin.nodeInd == 31 and bs_glob.max_node_ind == 31 and bs_glob.nNodes == 18
	assert twin.nodeId == 'internal_31' and twin.childNodes == [leaf] and leaf.parentNode is twin
	assert leaf.tParent == 0.0 and twin.tParent == old_t and twin.parentNode is old_parent
	assert twin in old_parent.childNodes and leaf not in old_parent.childNodes
	np.testing.assert_array_equal(twin.ltqs, leaf.ltqs)
	assert bonsai_tree.root.n_ds_nodes == root_count + 1


def test_do_spr_moves_moves_a_misplaced_cell_to_its_best_place():
	"""One candidate, every node as target: the cell ends up where the reference says attaching it is best."""
	truth, bonsai_tree, means, variances = _spr_setup()
	full = to_node(bonsai_tree.root)
	before = bonsai_tree.calcLogLComplete()
	best = _best_attachment(_without(full, 'C0'), 'C0', means, variances)
	cell = _by_id(bonsai_tree)['C0']
	successes, gain, _ = bonsai_tree.do_spr_moves(select_cand='list', cand_list=[cell.nodeInd], select_target='all',
	                                              do_postprocessing=False)
	assert successes == 1
	after = bonsai_tree.calcLogLComplete()
	assert after == pytest.approx(reference_bonsai_loglik(to_node(bonsai_tree.root), means, variances), rel=1e-9)
	assert after == pytest.approx(max(best.values()), abs=1e-6)
	assert gain == pytest.approx(after - before, rel=1e-6)
	result = to_node(bonsai_tree.root)
	assert frozenset(['C0', 'C1']) in clades(result, min_length=1e-9)


def test_do_spr_moves_only_scan_reports_gains_without_moving():
	truth, bonsai_tree, means, variances = _spr_setup()
	full = to_node(bonsai_tree.root)
	before = reference_bonsai_loglik(full, means, variances)
	best = _best_attachment(_without(full, 'C0'), 'C0', means, variances)
	cell = _by_id(bonsai_tree)['C0']
	others = [node.nodeInd for node in bonsai_nodes(bonsai_tree.root) if node.nodeId in ('C3', 'C5')]
	found = bonsai_tree.do_spr_moves(select_cand='list', cand_list=[cell.nodeInd] + others, select_target='all',
	                                 only_scan=True)
	assert write_newick(to_node(bonsai_tree.root)) == write_newick(full)
	assert [ind for _, ind in found] == [cell.nodeInd]
	assert -found[0][0] == pytest.approx(max(best.values()) - before, abs=1e-6)


@pytest.mark.parametrize('cand, target', [('long_branches_first', 'root'), ('long_branches_first', 'cluster_centers'),
                                          ('random', 'random'), ('long_branches_first', 'all')])
def test_do_spr_moves_never_lowers_the_likelihood(cand, target):
	truth, bonsai_tree, means, variances = _spr_setup()
	np.random.seed(1)
	before = bonsai_tree.calcLogLComplete()
	successes, gain, _ = bonsai_tree.do_spr_moves(select_cand=cand, select_target=target, do_postprocessing=True,
	                                              min_branch_length=1e-2)
	after = bonsai_tree.calcLogLComplete()
	result = to_node(bonsai_tree.root)
	assert sorted(result.leaf_names()) == sorted(truth.leaf_names())
	assert after == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-9)
	assert successes >= 1 and after > before + 1.0
	# The reported gain counts the moves; resolving polytomies after a move can only add to it.
	assert after - before >= gain - 1e-6
	assert bonsai_tree.nNodes == bs_glob.nNodes


def test_spr_moves_then_postprocessing_recover_the_true_tree():
	truth, bonsai_tree, means, variances = _spr_setup()
	before = bonsai_tree.calcLogLComplete()
	bonsai_tree.do_spr_moves(select_cand='long_branches_first', select_target='all', min_branch_length=1e-2)
	bonsai_tree.do_spr_postprocessing()
	result = to_node(bonsai_tree.root)
	assert rf_distance(result, truth) == 0
	after = bonsai_tree.calcLogLComplete()
	assert after > before
	assert after == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-9)


def test_prepare_tree_spr_moves_and_postprocessing_options():
	truth, bonsai_tree, means, variances = _spr_setup()
	bonsai_tree.set_mindist_root(cell_ids=truth.leaf_names())
	before = bonsai_tree.calcLogLComplete()
	loglik = bonsai_tree.prepare_tree_spr_moves()
	assert loglik == pytest.approx(before, rel=1e-9)
	assert len(bonsai_tree.root.childNodes) >= 3
	bonsai_tree.set_mindist_root(cell_ids=truth.leaf_names())
	bonsai_tree.do_spr_postprocessing(only_cleanup=True)
	assert len(bonsai_tree.root.childNodes) >= 3
	assert bonsai_tree.calcLogLComplete() == pytest.approx(before, rel=1e-9)
	unchanged = clades(to_node(bonsai_tree.root))
	bonsai_tree.do_spr_postprocessing(only_time_opt=True)
	assert clades(to_node(bonsai_tree.root)) == unchanged
	# Lengths are optimised with tol=1e-3, and lengths under 2e-6 are set to zero.
	assert bonsai_tree.calcLogLComplete() >= before - 1e-3


def test_select_spr_candidate_and_targets():
	truth, bonsai_tree, means, variances = _spr_setup()
	bonsai_tree.root.get_ds_node_counts()
	nodes = _by_id(bonsai_tree)
	np.random.seed(3)
	for _ in range(30):
		candidate, old_parent = bonsai_tree.select_spr_candidate(select_cand='random')
		assert not candidate.isRoot and old_parent is candidate.parentNode and len(old_parent.childNodes) >= 2
	cands = [nodes['C0'], bonsai_tree.root, nodes['x']]
	assert len(bonsai_tree.root.childNodes) == 3
	assert bonsai_tree.select_spr_candidate('list', cands_list=cands, n_moves=0) == (nodes['C0'], nodes['C0'].parentNode)
	assert bonsai_tree.select_spr_candidate('list', cands_list=cands, n_moves=1) == (None, None)
	assert bonsai_tree.select_spr_candidate('list', cands_list=cands, n_moves=2) == (nodes['x'], bonsai_tree.root)
	assert bonsai_tree.select_spr_targets('root') == [bonsai_tree.root]
	assert bonsai_tree.select_spr_targets('all') == bonsai_tree.root.getNodeList([])
	for _ in range(30):
		[target] = bonsai_tree.select_spr_targets('random', old_parent=nodes['c'])
		assert target is not nodes['c']
	detached = TreeNode(nodeInd=99)
	targets = bonsai_tree.select_spr_targets('cluster_centers', old_parent=nodes['c'],
	                                         cluster_centers=[nodes['b'], detached])
	assert targets == [nodes['c'], nodes['b']]
	assert bonsai_tree.select_spr_targets('cluster_centers', old_parent=None, cluster_centers=[nodes['b'], detached],
	                                      cluster_centers_guaranteed=True) == [nodes['b'], detached]


@pytest.mark.parametrize('target', ['cluster_centers', 'root'])
def test_add_cells_places_new_cells_where_they_belong(target):
	"""Start from the true tree on six cells, add C4 and C6 (with C8 in the tree): the result is the true tree."""
	truth = parse_newick(SPR_TRUTH)
	means, variances = tree_data(truth, 20, seed=4, noise=0.05)
	keep = ['C0', 'C1', 'C2', 'C3', 'C5', 'C7', 'C8']
	start = truth.copy()
	for name in ('C4', 'C6'):
		start = _without(start, name)
	bonsai_tree = build_bonsai_tree(start, {k: means[k] for k in keep}, {k: variances[k] for k in keep})
	bs_glob.nGenes, bs_glob.nCells = 20, 7
	bonsai_tree.root.storeParent()
	bonsai_tree.root.deleteParentsWithOneChild()
	bonsai_tree.optTimes(singleProcess=True)
	bonsai_tree.nNodes = bs_glob.nNodes
	scdata = SimpleSCData(keep)
	bonsai_tree.add_cells(np.array([means['C4'], means['C6']]), np.array([variances['C4'], variances['C6']]),
	                      ['C4', 'C6'], select_target=target, scdata=scdata, n_centers=2)
	result = to_node(bonsai_tree.root)
	assert sorted(result.leaf_names()) == sorted(truth.leaf_names())
	assert rf_distance(result, truth, min_length=1e-9) == 0
	assert scdata.metadata.cellIds == keep + ['C4', 'C6'] and scdata.metadata.nCells == 9 and bs_glob.nCells == 9
	assert bonsai_tree.calcLogLComplete() == pytest.approx(reference_bonsai_loglik(result, means, variances),
	                                                       rel=1e-9)


# ===================================================== next-nearest-neighbour interchanges

# One interchange away from SPR_TRUTH: C1 and b = (C2, C3) swapped across the edge above a.
NNI_WRONG = '(((C0:0.1,(C2:0.1,C3:0.1)b:1.0)a:1.0,C1:0.1)x:0.5,((C4:0.1,C5:0.1)c:1.0,(C6:0.1,C7:0.1)d:1.0)y:0.5,' \
            'C8:1.0)r;'


def test_getOptEdgeList_and_processOptReconfig_fix_a_wrong_interchange():
	"""Around the edge above node a, the four subtrees (C0, b, C1 and the rest) are re-merged; the change in the
	small tree's likelihood is the change of the whole tree's, and the result is the true tree."""
	truth, bonsai_tree, means, variances = _spr_setup(newick=NNI_WRONG)
	bonsai_tree.root.storeParent()
	nodes = _by_id(bonsai_tree)
	before = bonsai_tree.calcLogLComplete()
	gain, changed = bonsai_tree.getOptEdgeList(nodes['a'], None, finalOptTimes=False, returnEdgelist=False,
	                                           mem_friendly=True)
	assert changed is True and gain > 1.0
	nodes_list = bonsai_tree.root.getNodeList([])
	node_ind_to_node = {node.nodeInd: node for node in nodes_list}
	candidates = [node for node in nodes_list if not node.isLeaf and not node.isRoot]
	for node in nodes_list:
		node.cumClosenessNNN = 0.0
	gain, edges, most_upstream = bonsai_tree.getOptEdgeList(nodes['a'], None, finalOptTimes=True, returnEdgelist=True,
	                                                        mem_friendly=True)
	bonsai_tree.processOptReconfig(edges, node_ind_to_node, nodes_list, candidates, most_upstream)
	result = to_node(bonsai_tree.root)
	assert rf_distance(result, truth) == 0
	after = bonsai_tree.calcLogLComplete()
	assert after == pytest.approx(before + gain, rel=1e-9)
	assert after == pytest.approx(reference_bonsai_loglik(result, means, variances), rel=1e-9)
	assert bonsai_tree.nNodes == bs_glob.nNodes == len(list(result.iter_nodes()))


def test_getOptEdgeList_on_the_true_tree_changes_nothing():
	truth, bonsai_tree, means, variances = _spr_setup(newick=SPR_TRUTH)
	bonsai_tree.root.storeParent()
	gain, changed = bonsai_tree.getOptEdgeList(_by_id(bonsai_tree)['a'], None, finalOptTimes=True, mem_friendly=True)
	assert changed is False and gain == pytest.approx(0.0, abs=1e-3)


def test_buildTreeNNN_summarises_the_neighbourhood_of_an_edge():
	"""The small tree has the upper node as root, the lower node with its children, the upper node's other
	children, and the rest of the tree as one leaf carrying its message; its likelihood differs from the full
	tree's by a constant that does not depend on how these pieces are joined."""
	truth, bonsai_tree, means, variances = _spr_setup(newick=SPR_TRUTH)
	bonsai_tree.root.storeParent()
	bs_glob.max_node_ind = 20
	nodes = _by_id(bonsai_tree)
	ok, small, us_node, ds_copy, most_us, index_map, info = bonsai_tree.buildTreeNNN(nodes['a'],
	                                                                                 returnMostUSInfo=True)
	assert ok and us_node is nodes['x']
	assert small.root.nodeInd == nodes['x'].nodeInd and ds_copy.nodeInd == nodes['a'].nodeInd
	assert sorted(child.nodeInd for child in ds_copy.childNodes) == sorted(c.nodeInd for c in nodes['a'].childNodes)
	assert sorted(child.nodeInd for child in small.root.childNodes) == sorted(
		[nodes['a'].nodeInd, nodes['b'].nodeInd, bonsai_tree.root.nodeInd])
	assert info == {'mostUSNodeInd': bonsai_tree.root.nodeInd,
	                'childrenNotInvolved': [nodes['y'].nodeInd, nodes['C8'].nodeInd]}
	assert most_us is small.root.childNodes[-1]
	full = bonsai_tree.calcLogLComplete()
	small_loglik = small.calcLogLComplete()
	# Shorten the edge above a in both trees: the two likelihoods move by the same amount.
	nodes['a'].tParent *= 0.5
	ds_copy.tParent *= 0.5
	assert bonsai_tree.calcLogLComplete() - full == pytest.approx(small.calcLogLComplete() - small_loglik, rel=1e-8)
	# Too many neighbours (more than 30, or 50 with random=True): skipped.
	wide = star_of(['L%d' % ind for ind in range(33)], [1.0] * 33, *leaf_data(['L%d' % ind for ind in range(33)], 2, 0))
	bs_glob.nGenes = 2
	inner = wide.root.childNodes[0]
	assert wide.buildTreeNNN(inner)[0] is False


# ================================================= parallel merge rounds under mpiexec

# Run as a script under mpiexec: every rank enters mergeChildrenUB, rank 0 with the star tree. nNewPairsPar=0 and
# nDonePairsPar=0 make every round a parallel one; n_mem_friendly=0 sends the children's coordinates through
# memory-mapped files, a large value through MPI broadcasts; nChildUB=2 keeps the upper bounds in use (and their
# exchange between ranks) down to the last rounds.
PARALLEL_MERGE_SCRIPT = textwrap.dedent("""
	import json, logging, os, sys
	repo, ellipsoid, n_mem, out = sys.argv[1:5]
	sys.path.insert(0, repo)
	sys.path.insert(1, os.path.join(repo, 'tests'))
	import bonsai.mpi_wrapper as mpi_wrapper
	mpi_wrapper.mpi_init()
	import bonsai.bonsai_globals as bs_glob
	from bonsai.bonsai_treeHelpers import Tree
	from area_b.helpers_b import parallel_merge_star, to_node
	from bonsai_testtrees import write_newick
	logging.getLogger('myapp').setLevel(logging.ERROR)
	rank = mpi_wrapper.get_process_rank()
	if rank == 0:
		tree = parallel_merge_star()
		xr, W = tree.root.ltqs, tree.root.getW()
	else:
		parallel_merge_star()
		tree, xr, W = Tree(), None, None
	os.makedirs(out, exist_ok=True)
	tree.root.mergeChildrenUB(xr, W, sequential=True, ellipsoidSize=None if ellipsoid == 'None' else float(ellipsoid),
	                          nNewPairsPar=0, nDonePairsPar=0, n_mem_friendly=int(n_mem), outputFolder=out, nChildUB=2)
	if rank == 0:
		left = sorted(os.listdir(out))
		with open(os.path.join(out, 'result.json'), 'w') as handle:
			json.dump({'newick': write_newick(to_node(tree.root), length_format=lambda x: repr(float(x))),
			           'loglik': float(tree.calcLogLComplete()), 'left': left}, handle)
""")

PARALLEL_CASES = [(2, 'None', 0), (2, '1.0', 1000), (4, '1.0', 0)]


def _parallel_merge_job(pool, workdir):
	script = os.path.join(workdir, 'parallel_merge.py')
	with open(script, 'w') as handle:
		handle.write(PARALLEL_MERGE_SCRIPT)
	results = {}
	# The two 2-rank runs together, then the 4-rank run: at most four processes at a time.
	for batch in (PARALLEL_CASES[:2], PARALLEL_CASES[2:]):
		procs = {}
		for ranks, ellipsoid, n_mem in batch:
			out = os.path.join(workdir, 'out_%s_%s_%s' % (ranks, ellipsoid, n_mem))
			cmd = [mpiexec_path(), '--bind-to', 'none', '-n', str(ranks), sys.executable, '-m', 'mpi4py', script,
			       REPO_ROOT, ellipsoid, str(n_mem), out]
			procs[(ranks, ellipsoid, n_mem)] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			                                                     text=True, env=bonsai_env()), out)
		for key, (proc, out) in procs.items():
			_, stderr = proc.communicate(timeout=200)
			result_path = os.path.join(out, 'result.json')
			results[key] = (proc.returncode, stderr[-3000:],
			                open(result_path).read() if os.path.exists(result_path) else None)
	return results


b_parallel_merge_job = bonsai_job_fixture('b_parallel_merge_job', _parallel_merge_job, slots=4, mpi=True)


@pytest.mark.mpi
@pytest.mark.parametrize('ranks, ellipsoid, n_mem', PARALLEL_CASES)
def test_parallel_merge_rounds_give_the_serial_tree(ranks, ellipsoid, n_mem, b_parallel_merge_job, tmp_path):
	"""Merge rounds spread over 2 or 4 ranks (coordinates sent by broadcast or through memory-mapped files, with
	and without upper bounds) build the same tree as one process, and leave no exchange files behind."""
	import json
	from area_b.helpers_b import parallel_merge_star
	returncode, stderr, result = b_parallel_merge_job[(ranks, ellipsoid, n_mem)]
	assert returncode == 0 and result is not None, stderr
	parallel = json.loads(result)
	serial = parallel_merge_star()
	serial.root.mergeChildrenUB(serial.root.ltqs, serial.root.getW(), sequential=True,
	                            ellipsoidSize=None if ellipsoid == 'None' else float(ellipsoid), singleProcess=True,
	                            nChildUB=2)
	serial_tree, parallel_tree = to_node(serial.root), parse_newick(parallel['newick'])
	assert clades(parallel_tree, min_length=-1) == clades(serial_tree, min_length=-1)
	expected = _clade_lengths(serial_tree)
	for clade, length in _clade_lengths(parallel_tree).items():
		assert length == pytest.approx(expected[clade], rel=1e-9, abs=1e-12)
	assert parallel['loglik'] == pytest.approx(serial.calcLogLComplete(), rel=1e-9)
	assert parallel['left'] == []
	assert rf_distance(serial_tree, parse_newick(BALANCED8)) == 0


# ===================================== greedy merging at scale, with bonsai_main.py's settings
#
# Every rank enters mergeChildrenUB as bonsai_main.py's greedy step calls it at the README's recommended settings
# (UB_ellipsoid_size 1.0, use_knn 10, so nChildNN 50; SEQUENTIAL; the default nChildUB, nNewPairsPar,
# nDonePairsPar and n_mem_friendly). With 200 cells the first rounds have more than 1000 new candidate pairs, so on 4
# ranks they run in parallel; with 520 cells the first round also sends the coordinates through memory-mapped files
# (more than n_mem_friendly = 500 children).
LARGE_MERGE_SCRIPT = textwrap.dedent("""
	import json, logging, os, sys
	repo, key, out = sys.argv[1:4]
	sys.path.insert(0, repo)
	sys.path.insert(1, os.path.join(repo, 'tests'))
	import bonsai.mpi_wrapper as mpi_wrapper
	mpi_wrapper.mpi_init()
	from bonsai.bonsai_treeHelpers import Tree
	from area_b.helpers_b import large_greedy_star, to_node
	from bonsai_testtrees import write_newick
	logging.getLogger('myapp').setLevel(logging.ERROR)
	rank = mpi_wrapper.get_process_rank()
	tree, _ = large_greedy_star(key, build=(rank == 0))
	if rank == 0:
		xr, W = tree.root.ltqs, tree.root.getW()
	else:
		tree, xr, W = Tree(), None, None
	os.makedirs(out, exist_ok=True)
	tree.root.mergeChildrenUB(xr, W, sequential=True, ellipsoidSize=1.0, nChildNN=50, kNN=10, mergeDownstream=True,
	                          tree=tree if rank == 0 else None, outputFolder=out)
	if rank == 0:
		left = sorted(os.listdir(out))
		with open(os.path.join(out, 'result.json'), 'w') as handle:
			json.dump({'newick': write_newick(to_node(tree.root), length_format=lambda x: repr(float(x))),
			           'loglik': float(tree.calcLogLComplete()), 'left': left,
			           'ranks': mpi_wrapper.get_process_size()}, handle)
""")


def _large_greedy_job(key, ranks):
	"""A bonsai_job_fixture function: LARGE_MERGE_SCRIPT on large_greedy_specs()[key], serially or under mpiexec."""

	def job(pool, workdir):
		script = os.path.join(workdir, 'large_merge.py')
		with open(script, 'w') as handle:
			handle.write(LARGE_MERGE_SCRIPT)
		out = os.path.join(workdir, 'out')
		cmd = [sys.executable, script, REPO_ROOT, key, out]
		if ranks:
			cmd = [mpiexec_path(), '--bind-to', 'none', '-n', str(ranks), sys.executable, '-m', 'mpi4py'] + cmd[1:]
		proc = subprocess.run(cmd, capture_output=True, text=True, env=bonsai_env(), timeout=280)
		with open(os.path.join(workdir, 'large_merge.log'), 'w') as handle:
			handle.write('# %s\n# exit %d\n%s%s' % (' '.join(cmd), proc.returncode, proc.stdout, proc.stderr))
		result_path = os.path.join(out, 'result.json')
		result = json.load(open(result_path)) if os.path.exists(result_path) else None
		return proc.returncode, (proc.stdout + proc.stderr)[-3000:], result

	job.__name__ = 'large_greedy_job_%s_%s' % (key, ranks)
	return job


b_greedy200_serial = bonsai_job_fixture('b_greedy200_serial', _large_greedy_job('200', None), slots=1)
b_greedy200_mpi4 = bonsai_job_fixture('b_greedy200_mpi4', _large_greedy_job('200', 4), slots=4, mpi=True)
b_greedy520_mpi4 = bonsai_job_fixture('b_greedy520_mpi4', _large_greedy_job('520', 4), slots=4, mpi=True)


def _large_result(job, ranks):
	returncode, log, result = job
	assert returncode == 0 and result is not None, log
	assert result['ranks'] == ranks and result['left'] == [], result['left']
	return parse_newick(result['newick']), result['loglik']


def _assert_every_true_split(tree, sim):
	"""Every split of the generator's tree is a split of the greedy tree (which may have more: the cells of a clade
	sit on a star in the truth)."""
	assert sorted(tree.leaf_names()) == sorted(sim.cell_ids)
	missing = [len(split) for split in sim.truth_splits() if split not in splits(tree)]
	assert missing == [], 'true splits missing from the greedy tree, by size: %s' % missing


def _assert_same_tree(got, expected, got_loglik, expected_loglik):
	assert clades(got, min_length=-1) == clades(expected, min_length=-1)
	lengths = _clade_lengths(expected)
	for clade, length in _clade_lengths(got).items():
		assert length == pytest.approx(lengths[clade], rel=1e-9, abs=1e-12)
	assert got_loglik == pytest.approx(expected_loglik, rel=1e-9)


def test_greedy_merging_of_200_cells_finds_every_true_split(b_greedy200_serial):
	"""Eight clear clades of 18-32 cells, 60 genes: greedy merging with the upper bounds and nearest-neighbour
	candidates at the README's settings recovers every split of the generator's tree (the eight clades and the
	groupings above them), and its log-likelihood is the dense reference's on the tree it built."""
	from area_b.helpers_b import large_greedy_star
	tree, loglik = _large_result(b_greedy200_serial, 1)
	_, sim = large_greedy_star('200', build=False)
	_assert_every_true_split(tree, sim)
	means, variances = sim.bonsai_units()
	assert loglik == pytest.approx(reference_bonsai_loglik(tree, means, variances), rel=1e-9)


@pytest.mark.mpi
def test_greedy_merging_of_200_cells_on_4_ranks_builds_the_serial_tree(b_greedy200_serial, b_greedy200_mpi4):
	"""The same merges spread over 4 ranks (parallel rounds while more than 1000 new pairs are evaluated) build the
	serial tree: same rooted clades with zero-length edges kept, same branch lengths and log-likelihood (Bonsai
	at b16ef6f gives identical lengths here)."""
	serial, serial_loglik = _large_result(b_greedy200_serial, 1)
	parallel, parallel_loglik = _large_result(b_greedy200_mpi4, 4)
	_assert_same_tree(parallel, serial, parallel_loglik, serial_loglik)


@pytest.mark.mpi
def test_greedy_merging_of_520_cells_through_memory_mapped_files(b_greedy520_mpi4):
	"""520 cells on 4 ranks, more than the 500 children above which a merge round sends the children's coordinates
	through memory-mapped files at the default setting: every true split is found, the log-likelihood is the dense
	reference's on the tree built, and the exchange files are removed again. (Not compared with a serial run: on
	these cells Bonsai at b16ef6f builds the same unrooted tree serially, rooted differently, with a log-likelihood
	4e-7 relative apart.)"""
	from area_b.helpers_b import large_greedy_star
	tree, loglik = _large_result(b_greedy520_mpi4, 4)
	_, sim = large_greedy_star('520', build=False)
	_assert_every_true_split(tree, sim)
	means, variances = sim.bonsai_units()
	assert loglik == pytest.approx(reference_bonsai_loglik(tree, means, variances), rel=1e-9)


# ======================================================== posterior with the prior back in

@pytest.mark.parametrize('scaling', ['geneVariances', 0.8])
def test_lik_to_post_coords_old_reintroduces_the_gene_prior(scaling):
	"""Data rescaled by the diffusion scaling d (per gene: d = v_g): with the prior delta ~ N(0, v_g) on the
	deviation from the gene mean put back, the posterior mean of delta is (x - mu) v / (v + s^2) in original
	units; the function returns it divided by sqrt(d)."""
	rng = np.random.default_rng(0)
	gene_means, gene_vars = rng.normal(size=3), rng.uniform(0.5, 2.0, 3)
	bs_glob.geneMeans, bs_glob.geneVariances, bs_glob.geneDiffusionScaling = gene_means, gene_vars, scaling
	d = gene_vars if scaling == 'geneVariances' else scaling
	ltqs, ltqs_vars = rng.normal(size=(3, 4)), rng.uniform(0.05, 0.5, (3, 4))
	x = ltqs * np.sqrt(np.reshape(d, (-1, 1)) if scaling == 'geneVariances' else d)
	s2 = ltqs_vars * (np.reshape(d, (-1, 1)) if scaling == 'geneVariances' else d)
	delta_post = (x - gene_means[:, None]) * gene_vars[:, None] / (gene_vars[:, None] + s2)
	expected = delta_post / np.sqrt(np.reshape(d, (-1, 1)) if scaling == 'geneVariances' else d)
	np.testing.assert_allclose(th.lik_to_post_coords_old(ltqs, ltqs_vars), expected, rtol=1e-12)
	np.testing.assert_allclose(th.lik_to_post_coords_old(ltqs[:, 1], ltqs_vars[:, 1]), expected[:, 1], rtol=1e-12)
