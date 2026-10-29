"""
downstream_analyses/calc_marker_genes_helpers.py (area D): marker-gene scores between two groups of leaves.

The score of a gene, from the code comments and the docstrings of the error-bar variants, is the probability that
a random object from group 1 has a higher value than a random object from group 2: the fraction of (i, j) pairs
with x_i > x_j for the rank-based scores, and the average of Phi((m_i - m_j) / sqrt(v_i + v_j)) for the scores
with error bars. The references below compute these pair by pair. (The comment in calc_marker_genes_single says
"lower"; the code, the docstrings of the error-bar variants and Bonsai-scout's use agree on "higher".)

When a group has fewer than min_marker_genes genes past the cutoff, Bonsai fills it with an unspecified
selection from the wrong end of the scores; the tests use either fewer genes than
2 * min_marker_genes (every gene is returned) or enough genes past both cutoffs.
"""
import itertools
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import downstream_analyses.calc_marker_genes_helpers as cmg
from downstream_analyses.get_cluster_helpers import Cluster_Tree
from bonsai_testdata import DatasetSpec, balanced_newick
from bonsai_testfixtures import DATASET_NAME, bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env, read_edge_info, read_vert_info, run_bonsai, write_config
from bonsai_testtrees import parse_newick

HELPERS = os.path.join(REPO_ROOT, 'downstream_analyses', 'calc_marker_genes_helpers.py')


def pair_fraction(values, group1, group2):
	"""Fraction of pairs (i in group1, j in group2) with values[i] > values[j]."""
	return np.mean([values[i] > values[j] for i in group1 for j in group2])


def pair_probability(means, variances, group1, group2, weights=None):
	"""Average over pairs of P(X_i > X_j) for independent Gaussians, optionally weighted by n_i * n_j."""
	total, norm = 0.0, 0.0
	for i in group1:
		for j in group2:
			weight = 1.0 if weights is None else weights[i] * weights[j]
			total += weight * stats.norm.cdf((means[i] - means[j]) / math.sqrt(variances[i] + variances[j]))
			norm += weight
	return total / norm


def ranks(values):
	return np.argsort(np.argsort(values, axis=1), axis=1)


@pytest.fixture
def leaf_data():
	rng = np.random.default_rng(3)
	return rng.normal(0.0, 1.0, (6, 12))  # 6 genes x 12 leaves, no ties


def test_single_scores_every_gene_by_pair_counting(leaf_data):
	"""calc_marker_genes_single: for a group of leaves against all others, every gene (fewer genes than
	2 * min_marker_genes) gets the fraction of pairs where the group's leaf is higher; gene_subset restricts the
	genes and keeps their IDs."""
	group1 = [1, 4, 5, 9]
	group2 = [ind for ind in range(12) if ind not in group1]
	gene_ids = ['g%d' % ind for ind in range(6)]
	got = cmg.calc_marker_genes_single(group1, 12, gene_ids, ranks(leaf_data), min_marker_genes=10)
	assert list(got) == gene_ids
	for gene, score in enumerate(got.values()):
		assert score == pytest.approx(pair_fraction(leaf_data[gene], group1, group2), rel=1e-12)
	subset = np.array([4, 1])
	got = cmg.calc_marker_genes_single(group1, 12, gene_ids, ranks(leaf_data), gene_subset=subset,
	                                   min_marker_genes=10)
	assert got == pytest.approx({'g4': pair_fraction(leaf_data[4], group1, group2),
	                             'g1': pair_fraction(leaf_data[1], group1, group2)}, rel=1e-12)


def test_double_scores_two_groups_by_pair_counting(leaf_data):
	"""calc_marker_genes_double compares two groups that need not cover all leaves (ranks are recomputed within
	their union)."""
	group1, group2 = [0, 3, 7], [2, 5, 6, 11]
	gene_ids = ['g%d' % ind for ind in range(6)]
	got = cmg.calc_marker_genes_double(group1, group2, 12, gene_ids, ranks(leaf_data), min_marker_genes=10)
	for gene, score in enumerate(got.values()):
		assert score == pytest.approx(pair_fraction(leaf_data[gene], group1, group2), rel=1e-12)
	got = cmg.calc_marker_genes_double(group1, group2, 12, gene_ids, ranks(leaf_data), gene_subset=np.array([5]),
	                                   min_marker_genes=10)
	assert got == pytest.approx({'g5': pair_fraction(leaf_data[5], group1, group2)}, rel=1e-12)


def test_selection_past_both_cutoffs():
	"""With at least min_marker_genes genes above the cutoff and below 1 - cutoff, exactly those genes are kept."""
	rng = np.random.default_rng(4)
	data = rng.normal(0.0, 1.0, (30, 20))
	group1 = list(range(8))
	data[:5, group1] += 10.0   # five genes clearly higher in group 1
	data[5:11, group1] -= 10.0  # six genes clearly lower
	gene_ids = ['g%02d' % ind for ind in range(30)]
	for func, args in ((cmg.calc_marker_genes_single, (group1, 20)),
	                   (cmg.calc_marker_genes_double, (group1, list(range(8, 20)), 20))):
		got = func(*args, gene_ids, ranks(data), min_marker_genes=3, marker_cutoff=0.9)
		assert sorted(got) == gene_ids[:11]
		assert all(got[gene] == 1.0 for gene in gene_ids[:5]) and all(got[gene] == 0.0 for gene in gene_ids[5:11])


def test_calc_marker_genes_scores_every_edge():
	"""calc_marker_genes: for every edge (parent vertex, child vertex) the leaves below the child against all other
	leaves; store_ds_info lists every vertex's leaves in depth-first order."""
	newick = '(((A:1,B:1)x:1,C:1)y:1,(D:1,(E:1,F:1)z:1)w:1)r;'
	tree = Cluster_Tree()
	tree.from_newick_string(newick)
	leaves = parse_newick(newick).leaf_names()
	rng = np.random.default_rng(5)
	values = {leaf: rng.normal(0.0, 1.0, 4) for leaf in leaves}
	for node in tree.vert_ind_to_node.values():
		if node.isLeaf:
			node.ltqs = values[node.nodeId]
	gene_ids = ['a', 'b', 'c', 'd']
	result = cmg.calc_marker_genes(tree, gene_ids, min_marker_genes=10, verbose=False)
	matrix = np.array([values[leaf] for leaf in leaves]).T
	expected_edges = set()
	for node in tree.vert_ind_to_node.values():
		if node.parentNode is None:
			continue
		below = [leaves.index(name) for name in parse_newick(newick).find(node.nodeId).leaf_names()]
		others = [ind for ind in range(6) if ind not in below]
		edge = (node.parentNode.vert_ind, node.vert_ind)
		expected_edges.add(edge)
		for gene_ind, gene in enumerate(gene_ids):
			assert result[edge][gene] == pytest.approx(pair_fraction(matrix[gene_ind], below, others), rel=1e-12)
	assert set(result) == expected_edges
	vert_ind_to_node, leaf_ltqs, n_leafs = cmg.store_ds_info(tree.root, {}, [], 0)
	assert n_leafs == 6 and len(vert_ind_to_node) == tree.nNodes
	by_id = {node.nodeId: node for node in tree.vert_ind_to_node.values()}
	assert by_id['w'].ds_leaf_inds == [3, 4, 5] and by_id['r'].ds_leaf_inds == list(range(6))


@pytest.mark.parametrize('weighted', [False, True])
def test_error_bar_scores_are_gaussian_pair_probabilities(weighted):
	"""calc_marker_genes_error_bars: the (cell-count weighted) average of Phi((m_i - m_j) / sqrt(v_i + v_j));
	features without IDs are called Feature_<index>."""
	rng = np.random.default_rng(6)
	means = rng.normal(0.0, 1.0, (5, 9))
	variances = rng.uniform(0.05, 0.5, (5, 9))
	n_cells = rng.integers(1, 5, 9) if weighted else None
	group1, group2 = [0, 2, 3, 8], [1, 4, 5, 6]
	got = cmg.calc_marker_genes_error_bars(group1, group2, means, variances, gene_ids=None, min_marker_genes=10,
	                                       n_cells_per_object=n_cells)
	assert list(got) == ['Feature_%d' % ind for ind in range(5)]
	for gene in range(5):
		want = pair_probability(means[gene], variances[gene], group1, group2, weights=n_cells)
		assert got['Feature_%d' % gene] == pytest.approx(want, rel=1e-12)


def _mixture_integral(means, variances, group1, group2, low, high):
	"""Integral over [low, high] of the group-1 mixture density times the group-2 mixture CDF, averaged."""
	def integrand(x):
		pdf = np.mean(stats.norm.pdf(x, loc=means[group1], scale=np.sqrt(variances[group1])))
		cdf = np.mean(stats.norm.cdf(x, loc=means[group2], scale=np.sqrt(variances[group2])))
		return pdf * cdf
	return integrate.quad(integrand, low, high, limit=200, epsabs=1e-12)[0]


def test_grid_approximations_of_the_pair_probability():
	"""calc_marker_genes_error_bars_approx integrates group 1's density against group 2's CDF on a 1000-point grid
	from the lowest mean - 2 sd to the highest mean + 2 sd of both groups; _approx2 integrates on a grid uniform in
	group 1's sorted means, between group 1's own mean -/+ 2 sd. Checked against adaptive quadrature over the same
	ranges (to the grid error), and against the exact pair probability within the probability mass outside the
	ranges."""
	rng = np.random.default_rng(7)
	means = rng.normal(0.0, 1.0, (4, 30))
	variances = rng.uniform(0.2, 0.6, (4, 30))
	group1, group2 = list(range(0, 14)), list(range(14, 30))
	gene_ids = ['w', 'x', 'y', 'z']
	approx = cmg.calc_marker_genes_error_bars_approx(group1, group2, means, variances, gene_ids=gene_ids,
	                                                 min_marker_genes=10)
	approx2 = cmg.calc_marker_genes_error_bars_approx2(group1, group2, means, variances, gene_ids=gene_ids,
	                                                   min_marker_genes=10)
	approx2_fine = cmg.calc_marker_genes_error_bars_approx2(group1, group2, means, variances, gene_ids=gene_ids,
	                                                        min_marker_genes=10, n_points_total=2000)
	for gene, name in enumerate(gene_ids):
		m, v = means[gene], variances[gene]
		sd = np.sqrt(v)
		exact = pair_probability(m, v, group1, group2)
		low = min((m - 2 * sd)[group1].min(), (m - 2 * sd)[group2].min())
		high = max((m + 2 * sd)[group1].max(), (m + 2 * sd)[group2].max())
		# 1000-point rectangle rule: error below 2e-3 on these smooth integrands.
		assert approx[name] == pytest.approx(_mixture_integral(m, v, group1, group2, low, high), abs=2e-3)
		low1, high1 = (m - 2 * sd)[group1].min(), (m + 2 * sd)[group1].max()
		within = _mixture_integral(m, v, group1, group2, low1, high1)
		assert approx2_fine[name] == pytest.approx(within, abs=1e-4)
		# 100 grid points: the trapezoid error on these data stays below 5e-3.
		assert approx2[name] == pytest.approx(within, abs=5e-3)
		assert approx[name] == pytest.approx(exact, abs=0.05)
		assert approx2[name] == pytest.approx(exact, abs=0.05)
	# Duplicated objects in a group (several cells at one vertex) are counted twice.
	dup = cmg.calc_marker_genes_error_bars_approx2([0, 0, 1], [2, 3], means, variances, gene_ids=None,
	                                               min_marker_genes=10, n_points_total=2000)
	m, v = means[0], variances[0]
	sd = np.sqrt(v)
	want = _mixture_integral(m, v, [0, 0, 1], [2, 3], (m - 2 * sd)[[0, 1]].min(), (m + 2 * sd)[[0, 1]].max())
	assert dup['Feature_0'] == pytest.approx(want, abs=1e-4)


def test_store_load_and_get_marker_genes(tmp_path):
	"""store_marker_genes writes a CSV with columns marker_genes, marker_scores, edge_id (one row per gene per
	edge); load_marker_genes reads back the same dictionary; get_marker_genes loads an existing file unless
	recalc is set, calculates and stores otherwise, and returns None when it can do neither."""
	markers = {(0, 1): {'g1': 0.25, 'g2': 0.875}, (0, 2): {'g2': 0.1, 'g3': 1.0 / 3.0}}
	path = str(tmp_path / 'marker_genes.csv')
	cmg.store_marker_genes(path, markers)
	table = pd.read_csv(path)
	assert list(table.columns) == ['marker_genes', 'marker_scores', 'edge_id']
	assert list(table['edge_id']) == ['(0, 1)', '(0, 1)', '(0, 2)', '(0, 2)']
	assert cmg.load_marker_genes(path) == markers
	assert cmg.get_marker_genes(path) == markers
	tree = Cluster_Tree()
	tree.from_newick_string('((A:1,B:1)x:1,C:1)r;')
	for node, values in zip([tree.vert_ind_to_node[ind] for ind in (2, 3, 4)], ([1.0, 0.0], [2.0, 3.0], [0.5, 1.0])):
		node.ltqs = np.array(values)
	recalculated = cmg.get_marker_genes(path, store=True, recalc=True, tree=tree, gene_ids=['p', 'q'], verbose=False)
	assert recalculated[(1, 2)] == {'p': 0.5, 'q': 0.0}
	assert recalculated[(0, 4)] == {'p': 0.0, 'q': 0.5}
	assert cmg.load_marker_genes(path) == recalculated
	assert cmg.get_marker_genes(str(tmp_path / 'absent.csv')) is None


# ------------------------------------------------------------------------------------------- the script

MARKER_DATA = DatasetSpec(newick=balanced_newick(3, internal_length=0.6, leaf_length=0.1), n_genes=8, seed=9,
                          layout='features')


def _bonsai_then_markers(pool, workdir):
	data_folder, sim = pool.dataset(MARKER_DATA)
	results = os.path.join(workdir, 'results')
	config = write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder, results,
	                      input_is_sanity_output=False, filenames_data='features.txt,standard_deviations.txt',
	                      nnn_n_randommoves=20, nnn_n_randomtrees=2)
	run = run_bonsai(config).check()
	tree_folder = run.final_folder()
	proc = subprocess.run([sys.executable, HELPERS, '--path_to_code', REPO_ROOT, '--dataset', DATASET_NAME,
	                       '--data_folder', data_folder, '--results_folder', results, '--tree_folder', tree_folder,
	                       '--filenames_data', 'features.txt,standard_deviations.txt', '--verbose', 'False'],
	                      capture_output=True, text=True, env=bonsai_env(), cwd=workdir, timeout=240)
	return run, sim, proc, results, tree_folder


d_marker_script = bonsai_job_fixture('d_marker_script', _bonsai_then_markers)


def test_script_writes_marker_genes_for_every_edge(d_marker_script):
	"""The script loads the tree of --tree_folder with the features of --data_folder/--filenames_data (taken as
	they are: not Sanity output, no rescaling, no z-score filter) and writes <results_folder>/marker_genes.csv with,
	for every edge of the tree, every gene's score of the cells below the edge against all other cells."""
	run, sim, proc, results, tree_folder = d_marker_script
	assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
	markers = cmg.load_marker_genes(os.path.join(results, 'marker_genes.csv'))
	names = {vert: name for vert, _, name in read_vert_info(tree_folder)}
	children = {}
	for parent, child, _ in read_edge_info(tree_folder):
		children.setdefault(parent, []).append(child)

	def cells_below(vert):
		out = [names[vert]] if names[vert] in sim.cell_ids else []
		for child in children.get(vert, []):
			out += cells_below(child)
		return out

	edges = [(parent, child) for parent, child, _ in read_edge_info(tree_folder)]
	assert set(markers) == set(edges)
	cell_column = {cell: ind for ind, cell in enumerate(sim.cell_ids)}
	for parent, child in edges:
		below = [cell_column[cell] for cell in cells_below(child)]
		others = [ind for ind in range(sim.n_cells) if ind not in below]
		assert sorted(markers[(parent, child)]) == sorted(sim.gene_ids)
		for gene_ind, gene in enumerate(sim.gene_ids):
			want = pair_fraction(sim.measured[gene_ind], below, others)
			assert markers[(parent, child)][gene] == pytest.approx(want, rel=1e-12, abs=1e-15)
