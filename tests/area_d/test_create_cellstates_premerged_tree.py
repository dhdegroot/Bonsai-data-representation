"""
optional_preprocessing/create_cellstates_premerged_tree.py (area D): builds Bonsai's star tree, hangs the cells of
every cellstate with more than one cell under a new internal node, leaves one-cell cellstates on the root, and
optionally optimizes all branch lengths (help text), then stores the tree like any Bonsai tree folder.

Checks, all derived without Bonsai's code: the rooted clades are exactly the cellstates of more than one cell; with
--opt_times False every branch keeps its initial length 1.0; with the default (True) the stored lengths maximize the
dense reference likelihood (bonsai_testref) over all branch lengths of that topology, as found by scipy; the stored
node coordinates (ltqs_vertByGene.npy) are the cells' data and, at internal nodes, the Gaussian combination of the
children's coordinates passed through their branches.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

from bonsai_testdata import DatasetSpec, MULTIFURCATING_NEWICK
from bonsai_testfixtures import DATASET_NAME, bonsai_job_fixture
from bonsai_testref import reference_loglik
from bonsai_testrun import (REPO_ROOT, bonsai_env, read_edge_info, read_metadata_json, read_newick_file,
                            read_vert_info, tree_from_folder, write_config)
from bonsai_testtrees import clades

SCRIPT = os.path.join(REPO_ROOT, 'optional_preprocessing', 'create_cellstates_premerged_tree.py')
SPEC = DatasetSpec(newick=MULTIFURCATING_NEWICK, n_genes=20, seed=3, layout='sanity2')
# One cellstate per cell, in cellID.txt order (M0..M8). 'x' cuts across the true clades on purpose: the grouping
# must come from this file, not from the data. 'q' and 'z' have one cell each.
CELLSTATES = ['x', 'x', 'm', 'm', 'm', 'm', 'q', 'x', 'z']
EXPECTED_CLADES = {frozenset({'M0', 'M1', 'M7'}), frozenset({'M2', 'M3', 'M4', 'M5'})}


def _premerge(opt_times):
	def job(pool, workdir):
		data_folder, sim = pool.dataset(SPEC)
		config = write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder,
		                      os.path.join(workdir, 'results'), input_is_sanity_output=True, zscore_cutoff=-1)
		cellstates = os.path.join(workdir, 'optimized_clusters.txt')
		with open(cellstates, 'w') as handle:
			handle.write(''.join('%s\n' % label for label in CELLSTATES))
		premerged = os.path.join(workdir, 'premerged')
		args = ['--config_filepath', config, '--premerged_folder', premerged, '--cellstates_file', cellstates]
		if opt_times is not None:
			args += ['--opt_times', str(opt_times), '--verbose', 'False']
		proc = subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True, env=bonsai_env(),
		                      cwd=workdir, timeout=120)
		return proc, sim, premerged
	return job


# Options left out run at their defaults (--opt_times True, --verbose True).
d_opt_premerged_default = bonsai_job_fixture('d_opt_premerged_default', _premerge(None))
d_opt_premerged_no_opt = bonsai_job_fixture('d_opt_premerged_no_opt', _premerge(False))


def _ok(result):
	proc, sim, premerged = result
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	return sim, premerged


@pytest.mark.parametrize('which', ['d_opt_premerged_default', 'd_opt_premerged_no_opt'])
def test_cellstates_of_several_cells_become_clades(which, request):
	sim, premerged = _ok(request.getfixturevalue(which))
	tree = read_newick_file(premerged)
	assert sorted(tree.leaf_names()) == sorted(sim.cell_ids)
	assert clades(tree, min_length=-1) == EXPECTED_CLADES
	# Every cell is a leaf at depth one or two: a child of the root or of its cellstate's node.
	assert {child.name for child in tree.children if child.is_leaf} == {'M6', 'M8'}
	assert len(tree.children) == 4
	# edgeInfo.txt/vertInfo.txt describe the same rooted tree as tree.nwk.
	assert clades(tree_from_folder(premerged), min_length=-1, include_root=True) == \
		clades(tree, min_length=-1, include_root=True)
	metadata = read_metadata_json(premerged)
	assert metadata['cellIds'] == sim.cell_ids
	assert metadata['geneIds'] == sim.gene_ids
	assert metadata['nCells'] == sim.n_cells and metadata['nGenes'] == sim.n_genes


def test_without_time_optimization_every_branch_is_one(d_opt_premerged_no_opt):
	sim, premerged = _ok(d_opt_premerged_no_opt)
	lengths = [length for _, _, length in read_edge_info(premerged)]
	assert len(lengths) == sim.n_cells + len(EXPECTED_CLADES)
	assert lengths == [1.0] * len(lengths)


def test_default_optimizes_branch_lengths_to_the_reference_optimum(d_opt_premerged_default):
	"""
	The stored lengths (tree.nwk, full precision) are compared with a scipy L-BFGS-B maximization of the dense
	reference likelihood started from them. Bonsai's optimizer stops at its own tolerance: on this input (Bonsai
	b16ef6f, 2026-10-27) its log-likelihood was 1.8e-6 below scipy's and its lengths within 5.2e-5 of scipy's, while
	a 10 % change of any one length (all are above 0.03) cost at least 4.8e-3, so the tolerances below separate
	converged from wrong lengths with a margin of about ten on each side.
	"""
	sim, premerged = _ok(d_opt_premerged_default)
	means, variances = sim.bonsai_units()
	tree = read_newick_file(premerged)
	edges = [node for node in tree.iter_nodes() if node is not tree]
	stored = np.array([node.length for node in edges])

	def negative_loglik(lengths):
		for node, length in zip(edges, lengths):
			node.length = length
		return -reference_loglik(tree, means, variances)

	at_stored = negative_loglik(stored)
	best = minimize(negative_loglik, stored, method='L-BFGS-B', bounds=[(0.0, None)] * len(stored))
	assert at_stored - best.fun < 1e-4
	np.testing.assert_allclose(stored, best.x, rtol=0, atol=5e-4)
	# Not the initial star-tree lengths: the optimization did something.
	assert np.max(np.abs(stored - 1.0)) > 0.5


@pytest.mark.parametrize('which', ['d_opt_premerged_default', 'd_opt_premerged_no_opt'])
def test_stored_coordinates_are_the_downstream_gaussian_messages(which, request):
	"""
	ltqs_vertByGene.npy / ltqsVars_vertByGene.npy: rows by vertInd. A cell's row is its data in Bonsai's units
	(divided by sqrt(v_g) and v_g); an internal node's row is the product of its children's Gaussians, each widened
	by its branch length: 1/w = sum_c 1/(w_c + t_c), m = w sum_c m_c/(w_c + t_c).
	"""
	sim, premerged = _ok(request.getfixturevalue(which))
	means, variances = sim.bonsai_units()
	coords = np.load(os.path.join(premerged, 'ltqs_vertByGene.npy'))
	coord_vars = np.load(os.path.join(premerged, 'ltqsVars_vertByGene.npy'))
	rows = {name: vert for vert, _, name in read_vert_info(premerged)}
	assert coords.shape == coord_vars.shape == (len(rows), sim.n_genes)
	tree = read_newick_file(premerged)

	def message(node):
		if node.is_leaf:
			return means[node.name], variances[node.name]
		precision, weighted = 0.0, 0.0
		for child in node.children:
			mean, var = message(child)
			precision = precision + 1.0 / (var + child.length)
			weighted = weighted + mean / (var + child.length)
		return weighted / precision, 1.0 / precision

	for node in tree.iter_nodes():
		mean, var = message(node)
		np.testing.assert_allclose(coords[rows[node.name]], mean, rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(coord_vars[rows[node.name]], var, rtol=1e-9, atol=1e-15)
