"""
Helpers for the area C (pipeline) tests: writing a tree in Bonsai's tree-folder
format, reading a final folder into a comparable state, and reference
computations of the numbers metadata_bonsai.txt reports (the likelihood of a
tree with all branch lengths, or one common scale factor, optimised).

The reference optimisers use the dense model of tests/bonsai_testref.py and an
analytic gradient derived from it, not Bonsai's code.
"""
import json
import math
import os
import shutil

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from bonsai_testref import reference_bonsai_loglik, shared_path_matrix
from bonsai_testrun import (read_metadata_bonsai, read_metadata_json, read_newick_file, posteriors_by_name,
                            read_vert_info, stage_folder)
from bonsai_testtrees import Node, clades, splits

# Every file of a final folder (README, "The Bonsai results").
FINAL_FILES = {'edgeInfo.txt', 'vertInfo.txt', 'tree.nwk', 'metadata.json', 'metadata_bonsai.txt',
               'posterior_ltqs_vertByGene.npy', 'posterior_ltqsVars_vertByGene.npy'}
# The intermediate stages, in the order bonsai_main.py runs them.
STAGES = ('preprocessed', 'greedy', 'redo_starry', 'opt_times', 'spr_moves', 'nnn_reorder', 'reordered_edges')
METADATA_COLUMNS = ['nGenes', 'nCells', 'reconsTime', 'zscoreCutoff', 'coresUsed', 'finalLoglik', 'trueLoglik',
                    'fullLoglik', 'remainingRootChildren']


# ------------------------------------------------------------ tree folders

def write_tree_folder(tree, folder, cell_ids, metadata=None):
	"""
	Write a Node tree as a Bonsai tree folder (README, "The Bonsai results"):
	edgeInfo.txt (parentVert, childVert, length; parents before children),
	vertInfo.txt (vertInd, nodeInd, vertName) and, when given, metadata.json.
	Cells get nodeInd = their index in cell_ids, the root -1 and internal nodes
	nCells, nCells + 1, ... Lengths are written in full precision.
	"""
	os.makedirs(folder, exist_ok=True)
	cell_index = {cell: ind for ind, cell in enumerate(cell_ids)}
	vert_of = {}
	rows = []
	next_internal = len(cell_ids)
	for node in tree.iter_nodes():
		vert = len(rows)
		vert_of[id(node)] = vert
		if node is tree:
			rows.append((vert, -1, node.name or 'root'))
		elif node.is_leaf:
			rows.append((vert, cell_index[node.name], node.name))
		else:
			rows.append((vert, next_internal, node.name or 'internal_%d' % (next_internal - len(cell_ids))))
			next_internal += 1
	with open(os.path.join(folder, 'edgeInfo.txt'), 'w') as handle:
		for node in tree.iter_nodes():
			for child in node.children:
				handle.write('%d\t%d\t%r\n' % (vert_of[id(node)], vert_of[id(child)], float(child.length)))
	with open(os.path.join(folder, 'vertInfo.txt'), 'w') as handle:
		handle.write('vertInd\tnodeInd\tvertName\n')
		for row in rows:
			handle.write('%d\t%d\t%s\n' % row)
	if metadata is not None:
		with open(os.path.join(folder, 'metadata.json'), 'w') as handle:
			json.dump(metadata, handle, indent=4)
	return folder


def group_tree(groups, inner_length=0.5, leaf_length=0.3):
	"""A root with one polytomy per group of cell names: ((a,b,c),(d,e,f),...)."""
	return Node('root', None, [Node(None, inner_length, [Node(cell, leaf_length) for cell in group])
	                           for group in groups])


def star(cells, length=0.5):
	return Node('root', None, [Node(cell, length) for cell in cells])


# --------------------------------------------------------- final folders

def final_state(folder):
	"""
	What a final folder says about the reconstruction: the tree (from tree.nwk,
	full precision), its lengths keyed by the leaf set below each edge, the
	posteriors keyed by vertex name and metadata_bonsai.txt.
	"""
	tree = read_newick_file(folder)
	lengths = {frozenset(node.leaf_names()): node.length for node in tree.iter_nodes() if node.length is not None}
	means, variances = posteriors_by_name(folder)
	return dict(tree=tree, lengths=lengths, means=means, variances=variances, meta=read_metadata_bonsai(folder))


def assert_same_reconstruction(state, expected, rtol=1e-9, atol=1e-12, posteriors=True):
	"""
	Same rooted tree (every clade, the root included) and the same branch
	lengths, final log-likelihood and posteriors to the given tolerances.
	Edges are matched by the set of cells below them, so internal node names
	and child order may differ.
	"""
	assert clades(state['tree'], include_root=True, min_length=-1) == \
		clades(expected['tree'], include_root=True, min_length=-1)
	assert set(state['lengths']) == set(expected['lengths'])
	for clade, length in expected['lengths'].items():
		assert state['lengths'][clade] == _approx(length, rtol, atol), sorted(clade)
	assert float(state['meta']['finalLoglik']) == _approx(float(expected['meta']['finalLoglik']), rtol, 0.0)
	if posteriors:
		by_clade = _posteriors_by_clade(state)
		for clade, (mean, var) in _posteriors_by_clade(expected).items():
			np.testing.assert_allclose(by_clade[clade][0], mean, rtol=rtol, atol=atol)
			np.testing.assert_allclose(by_clade[clade][1], var, rtol=rtol, atol=atol)


def _approx(value, rtol, atol):
	import pytest
	return pytest.approx(value, rel=rtol, abs=atol)


def _posteriors_by_clade(state):
	"""Posteriors keyed by the leaf set below each vertex (the root by the full leaf set)."""
	out = {}
	for node in state['tree'].iter_nodes():
		out[frozenset(node.leaf_names()) if not node.is_leaf else node.name] = (state['means'][node.name],
		                                                                     state['variances'][node.name])
	return out


def rename_leaves(tree, mapping):
	tree = tree.copy()
	for node in tree.iter_nodes():
		if node.is_leaf:
			node.name = mapping[node.name]
	return tree


# --------------------------------------------------- reference optimisers

def _leaf_data(tree, means, variances):
	leaves = tree.leaves()
	x = np.array([np.asarray(means[leaf.name], dtype=float) for leaf in leaves])
	v = np.array([np.asarray(variances[leaf.name], dtype=float) for leaf in leaves])
	return leaves, x, v


def _edges_and_masks(tree):
	"""Every non-root edge of the tree and, for each, the indicator over leaves below it."""
	leaves = tree.leaves()
	index = {id(leaf): ind for ind, leaf in enumerate(leaves)}
	edges = [node for node in tree.iter_nodes() if node is not tree]
	masks = np.zeros((len(edges), len(leaves)))
	for row, node in enumerate(edges):
		for leaf in node.leaves():
			masks[row, index[id(leaf)]] = 1.0
	return edges, masks


def loglik_and_gradient(lengths, masks, x, v):
	"""
	The flat-root-prior log-likelihood of the leaves (natural units, summed over
	genes) as a function of the edge lengths, and its gradient. With
	Sigma = sum_e t_e u_e u_e' + diag(v_g) and P = Sigma^-1 - Sigma^-1 1 1' Sigma^-1 / (1' Sigma^-1 1),
	d log p / d Sigma = (P x x' P - P) / 2, so d log p / d t_e = u_e' (P x x' P - P) u_e / 2.
	"""
	n_cells, n_genes = x.shape
	shared = (masks * lengths[:, None]).T @ masks
	ones = np.ones(n_cells)
	total = 0.0
	grad = np.zeros(len(lengths))
	for g in range(n_genes):
		sigma = shared + np.diag(v[:, g])
		chol = np.linalg.cholesky(sigma)
		inv = np.linalg.inv(sigma)
		inv_ones = inv @ ones
		a = ones @ inv_ones
		proj = inv - np.outer(inv_ones, inv_ones) / a
		px = proj @ x[:, g]
		total += -0.5 * (n_cells - 1) * math.log(2 * math.pi) - np.sum(np.log(np.diag(chol))) - 0.5 * math.log(a) \
			- 0.5 * x[:, g] @ px
		middle = np.outer(px, px) - proj
		grad += 0.5 * np.einsum('ei,ij,ej->e', masks, middle, masks)
	return total, grad


def optimise_all_lengths(tree, means, variances, start=None):
	"""
	Maximise the dense log-likelihood over every edge length (>= 0) of the
	tree's topology. Returns (natural log-likelihood, optimal lengths in edge order).
	"""
	leaves, x, v = _leaf_data(tree, means, variances)
	edges, masks = _edges_and_masks(tree)
	t0 = np.array([max(node.length, 1e-4) for node in edges]) if start is None else np.asarray(start, dtype=float)

	def negative(t):
		value, grad = loglik_and_gradient(t, masks, x, v)
		return -value, -grad

	result = minimize(negative, t0, jac=True, method='L-BFGS-B', bounds=[(0.0, None)] * len(t0),
	                  options=dict(ftol=1e-15, gtol=1e-10, maxiter=10000, maxcor=30))
	return -result.fun, result.x


def optimise_common_scale(tree, means, variances):
	"""
	Maximise the dense log-likelihood over one factor that multiplies every
	edge length, the lengths first clipped to [1e-4, 1e4]. Returns the natural
	log-likelihood at the optimum.
	"""
	leaves, x, v = _leaf_data(tree, means, variances)
	edges, masks = _edges_and_masks(tree)
	t0 = np.clip([node.length for node in edges], 1e-4, 1e4)

	def negative(log_scale):
		return -loglik_and_gradient(t0 * math.exp(log_scale), masks, x, v)[0]

	result = minimize_scalar(negative, bounds=(-13.8, 13.8), method='bounded', options=dict(xatol=1e-12))
	return -result.fun


def to_bonsai_units(loglik, n_cells, n_genes):
	return 2.0 * loglik + (n_cells - 1) * n_genes * math.log(2 * math.pi)


# ------------------------------------------------------- running scripts

# Written to a sitecustomize.py that only the backbone tests put on PYTHONPATH.
ARGPARSE_SHIM = '''"""
Python 3.14's argparse checks every help string when an argument is added.
backbone_based_bonsai.py and bonsai_add_cells.py have the help text
"(so .1 means 10% growth)", whose lone '%' fails that check, so on Python 3.14
both scripts stop at start-up with "ValueError: badly formed help string"
(a bug in both scripts). Python 3.13 and older only
format help strings for --help. This turns the check off, as it is on those
versions, and changes nothing else; any other sitecustomize still runs after it.
"""
import argparse
import importlib.machinery
import importlib.util
import os
import sys

if hasattr(argparse._ActionsContainer, '_check_help'):
	argparse._ActionsContainer._check_help = lambda self, action: None

_here = os.path.dirname(os.path.abspath(__file__))
for _entry in sys.path:
	if os.path.abspath(_entry or os.curdir) == _here:
		continue
	_spec = importlib.machinery.PathFinder.find_spec('sitecustomize', [_entry or os.curdir])
	if _spec is not None and _spec.origin and os.path.dirname(os.path.abspath(_spec.origin)) != _here:
		_module = importlib.util.module_from_spec(_spec)
		_spec.loader.exec_module(_module)
		break
'''


def argparse_shim_env(folder):
	"""Environment additions that put the argparse shim (ARGPARSE_SHIM) before the checkout on PYTHONPATH."""
	from bonsai_testrun import REPO_ROOT
	os.makedirs(folder, exist_ok=True)
	with open(os.path.join(folder, 'sitecustomize.py'), 'w') as handle:
		handle.write(ARGPARSE_SHIM)
	return {'PYTHONPATH': folder + os.pathsep + REPO_ROOT}


def run_script(script, args, log_path, env=None, timeout=240):
	"""
	Run a script of the checkout (path relative to the repository root) as a
	subprocess with the harness environment; write the command, exit code and
	output to log_path. Returns (returncode, output).
	"""
	import signal
	import subprocess
	import sys
	import time
	from bonsai_testrun import REPO_ROOT, TESTS_DIR, bonsai_env
	cmd = [sys.executable, os.path.join(REPO_ROOT, script)] + [str(arg) for arg in args]
	start = time.time()
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=bonsai_env(env),
	                        cwd=TESTS_DIR, start_new_session=True)
	try:
		stdout, stderr = proc.communicate(timeout=timeout)
	except subprocess.TimeoutExpired:
		if hasattr(os, 'killpg'):
			os.killpg(proc.pid, signal.SIGKILL)
		else:
			proc.kill()
		stdout, stderr = proc.communicate()
	with open(log_path, 'w') as handle:
		handle.write('# %s\n# exit %s, %.1f s\n' % (' '.join(cmd), proc.returncode, time.time() - start))
		handle.write(stdout + stderr)
	return proc.returncode, stdout + stderr


# ------------------------------------------------------------ result files

def copy_results(source, target):
	"""Copy a results folder. metadata.json files keep pointing at the source's processed data."""
	shutil.copytree(source, target)
	return target


def remove_stages_after(results_folder, stage, zscore_cutoff=1.0):
	"""Delete the folders of every stage after `stage`, and the final folder, so pickup restarts there."""
	order = list(STAGES) + ['final']
	for later in order[order.index(stage) + 1:]:
		path = os.path.join(results_folder, stage_folder(later, zscore_cutoff))
		if os.path.isdir(path):
			shutil.rmtree(path)
