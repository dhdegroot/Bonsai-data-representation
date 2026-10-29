"""
Helpers shared by the area B tests (tree algorithms): converting Bonsai's Tree
to the harness's Node tree, random leaf data, finite differences, brute-force
topologies and branch-length optima computed on the dense reference likelihood
of tests/bonsai_testref.py, and an in-process SCData for the driver functions
of bonsai_dataprocessing.py.
"""
import itertools
import math
import os

import numpy as np
from scipy.optimize import minimize

from bonsai_testref import build_bonsai_tree, reference_bonsai_loglik, star_tree
from bonsai_testtrees import Node, parse_newick, path_length

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ data

def leaf_data(names, n_genes, seed, scale=1.0, var_range=(0.01, 0.5)):
	"""Random means and variances per leaf (dictionaries from name to a vector over genes)."""
	rng = np.random.default_rng(seed)
	means = {name: scale * rng.normal(0.0, 1.0, n_genes) for name in names}
	variances = {name: rng.uniform(var_range[0], var_range[1], n_genes) for name in names}
	return means, variances


def tree_data(tree, n_genes, seed, noise=0.05):
	"""
	Means that diffuse along a Node tree (Brownian motion at unit rate) plus
	Gaussian noise of variance noise**2, which is also the reported variance.
	"""
	rng = np.random.default_rng(seed)
	position = {id(tree): np.zeros(n_genes)}
	for node in tree.iter_nodes():
		for child in node.children:
			position[id(child)] = position[id(node)] + rng.normal(0.0, math.sqrt(child.length), n_genes)
	means, variances = {}, {}
	for leaf in tree.leaves():
		means[leaf.name] = position[id(leaf)] + rng.normal(0.0, noise, n_genes)
		variances[leaf.name] = np.full(n_genes, noise ** 2)
	return means, variances


def matrices(names, means, variances):
	"""Genes x cells matrices in the order of names."""
	return (np.array([means[name] for name in names], dtype=float).T,
	        np.array([variances[name] for name in names], dtype=float).T)


# ------------------------------------------------------------ conversion

def to_node(treenode, is_root=True):
	"""The harness Node tree of a Bonsai TreeNode (names are nodeIds, lengths are tParent)."""
	return Node(treenode.nodeId, None if is_root else treenode.tParent,
	            [to_node(child, False) for child in treenode.childNodes])


def bonsai_nodes(treenode):
	"""Every TreeNode below (and including) treenode, pre-order."""
	out = [treenode]
	for child in treenode.childNodes:
		out.extend(bonsai_nodes(child))
	return out


def reference_of(bonsai_tree, means, variances, loglik_var_corr=None):
	"""The dense reference log-likelihood (Bonsai's units) of a Bonsai Tree's current topology and lengths."""
	return reference_bonsai_loglik(to_node(bonsai_tree.root), means, variances, loglik_var_corr=loglik_var_corr)


def build(newick, means, variances):
	"""A Bonsai Tree for a newick string and leaf data, with ltqs computed at every node."""
	tree = build_bonsai_tree(parse_newick(newick), means, variances)
	tree.root.getLtqsComplete(mem_friendly=False)
	return tree


def leaf_pairs_distances(tree):
	"""Path length between every pair of leaves of a Node tree, keyed by the frozenset of the two names."""
	names = tree.leaf_names()
	return {frozenset(pair): path_length(tree, *pair) for pair in itertools.combinations(names, 2)}


# ------------------------------------------------------- numerical helpers

def central_difference(fun, x, step=1e-6):
	"""Gradient of a scalar function by central differences, with a step relative to |x| (at least step)."""
	x = np.asarray(x, dtype=float)
	grad = np.zeros_like(x)
	for ind in range(x.size):
		h = step * max(1.0, abs(x.flat[ind]))
		up, down = x.copy(), x.copy()
		up.flat[ind] += h
		down.flat[ind] -= h
		grad.flat[ind] = (fun(up) - fun(down)) / (2 * h)
	return grad


def edges_of(tree):
	"""The non-root nodes of a Node tree, pre-order: one per edge."""
	return [node for node in tree.iter_nodes() if node is not tree]


def with_lengths(tree, lengths):
	"""A copy of a Node tree with new edge lengths (pre-order over edges_of)."""
	out = tree.copy()
	for node, length in zip(edges_of(out), lengths):
		node.length = float(length)
	return out


def optimise_lengths(tree, means, variances, starts=((0.5,), (0.05,), (2.0,)), fixed=None):
	"""
	Maximise the dense reference log-likelihood over the edge lengths of a Node
	tree (L-BFGS-B on log lengths, several constant starting points, lengths in
	[1e-7, 1e3]). fixed: dictionary from edge index to a fixed length.
	Returns (best Bonsai-units log-likelihood, lengths).
	"""
	edges = edges_of(tree)
	fixed = fixed or {}
	free = [ind for ind in range(len(edges)) if ind not in fixed]

	def full(logt):
		lengths = np.zeros(len(edges))
		for ind, value in fixed.items():
			lengths[ind] = value
		lengths[free] = np.exp(logt)
		return lengths

	def objective(logt):
		return -reference_bonsai_loglik(with_lengths(tree, full(logt)), means, variances)

	best = (-np.inf, None)
	for start in starts:
		x0 = np.log(np.full(len(free), start[0]))
		res = minimize(objective, x0, method='L-BFGS-B', bounds=[(math.log(1e-7), math.log(1e3))] * len(free),
		               options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 2000})
		if -res.fun > best[0]:
			best = (-res.fun, full(res.x))
	return best


def unrooted_topologies(names):
	"""
	Every unrooted binary topology on the given leaf names (at least 3), each a
	Node tree rooted at a node of degree three, without edge lengths. 4 leaves
	give 3 topologies, 5 leaves give 15.
	"""
	first = Node('r', None, [Node(name, 1.0) for name in names[:3]])
	trees = [first]
	for name in names[3:]:
		grown = []
		for tree in trees:
			for edge_ind in range(len(edges_of(tree))):
				new = tree.copy()
				target = edges_of(new)[edge_ind]
				parent = new.parents()[id(target)]
				position = [id(child) for child in parent.children].index(id(target))
				parent.children[position] = Node(None, 1.0, [target, Node(name, 1.0)])
				grown.append(new)
		trees = grown
	return trees


def star_of(names, lengths, means, variances):
	"""A Bonsai Tree that is a star over the given leaves, with ltqs computed."""
	tree = build_bonsai_tree(star_tree(names, lengths), means, variances)
	tree.root.getLtqsComplete(mem_friendly=False)
	return tree


# ------------------------------------------------------- in-process SCData

class SimpleMetadata:
	"""What Tree.add_cells reads from scdata.metadata."""

	def __init__(self, cell_ids):
		self.cellIds = list(cell_ids)
		self.nCells = len(cell_ids)


class SimpleSCData:
	def __init__(self, cell_ids):
		self.metadata = SimpleMetadata(cell_ids)


def clade_newick(sizes, internal_length=1.5, leaf_length=0.1):
	"""
	len(sizes) clades (a power of two), clade k a star of sizes[k] cells K<k>c<j> on edges of leaf_length, the
	clades joined in a balanced binary tree with internal_length on every edge above a clade's ancestor.
	"""
	level = ['(' + ','.join('K%dc%02d:%r' % (k, j, leaf_length) for j in range(size)) + ')'
	         for k, size in enumerate(sizes)]
	while len(level) > 1:
		level = ['(%s:%r,%s:%r)' % (level[i], internal_length, level[i + 1], internal_length)
		         for i in range(0, len(level), 2)]
	return level[0] + ';'


def large_greedy_specs():
	"""The generator's inputs of the greedy-merging runs at scale: eight clear clades of 200 cells in all (above
	use_knn's 50-children threshold and, on 4 ranks, the 1000 new pairs that make a merge round parallel) and of
	520 cells (also above the 500 children from which the coordinates go through memory-mapped files)."""
	from bonsai_testdata import DatasetSpec
	return {'200': DatasetSpec(newick=clade_newick((20, 22, 24, 26, 28, 18, 30, 32)), n_genes=60, noise_sd=0.1, seed=0),
	        '520': DatasetSpec(newick=clade_newick((60, 62, 64, 66, 68, 58, 70, 72)), n_genes=60, noise_sd=0.1, seed=0)}


def large_greedy_star(key, build=True):
	"""
	Simulate large_greedy_specs()[key]; set bonsai_globals.nGenes and nCells; with build, also return the star
	tree over all cells with its branch lengths optimised (as bonsai_main.py starts the greedy step).
	Returns (tree or None, sim).
	"""
	import bonsai.bonsai_globals as bs_glob
	import bonsai.mpi_wrapper as mpi_wrapper
	from bonsai_testdata import simulate
	spec = large_greedy_specs()[key]
	sim = simulate(spec)
	bs_glob.nGenes, bs_glob.nCells = spec.n_genes, sim.n_cells
	if not build:
		return None, sim
	means, variances = sim.bonsai_units()
	tree = star_of(sim.cell_ids, [1.0] * sim.n_cells, means, variances)
	tree.root.optimiseConnected(mpi_wrapper.get_mpi_info(singleProcess=True), singleProcess=True)
	return tree, sim


def parallel_merge_star():
	"""The star tree the parallel-merge test hands to mergeChildrenUB (8 cells, 20 genes), times optimised;
	sets bonsai_globals.nGenes and nCells."""
	import bonsai.bonsai_globals as bs_glob
	import bonsai.mpi_wrapper as mpi_wrapper
	from bonsai_testdata import balanced_newick
	truth = parse_newick(balanced_newick(3, internal_length=1.0, leaf_length=0.1))
	means, variances = tree_data(truth, 20, 0, noise=0.1)
	names = truth.leaf_names()
	tree = star_of(names, [1.0] * len(names), means, variances)
	bs_glob.nGenes, bs_glob.nCells = 20, len(names)
	tree.root.optimiseConnected(mpi_wrapper.get_mpi_info(singleProcess=True), singleProcess=True)
	return tree
