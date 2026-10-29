"""
The tree utilities of bonsai/bonsai_helpers.py (area B): condensed-distance indices, newick and tree-object
conversion, shortest distances on a tree, the bookkeeping of candidate pairs and their log-likelihood gains, the
initial branch lengths of a merge, Kruskal's algorithm in OwnGraph and the cell-type annotation of a tree.
The rest of bonsai_helpers.py belongs to area A (tests/area_a/test_bonsai_helpers.py).
"""
import itertools
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

import bonsai.bonsai_helpers as helpers
from bonsai.bonsai_treeHelpers import TreeNode

from bonsai_testtrees import Node, parse_newick, path_length


@pytest.mark.parametrize('n', [2, 3, 7])
def test_get_pdist_inds_indexes_the_condensed_distance_vector(n):
	square = np.arange(n * n, dtype=float).reshape(n, n)
	square = square + square.T
	np.fill_diagonal(square, 0.0)
	condensed = squareform(square)
	for i, j in itertools.permutations(range(n), 2):
		assert condensed[helpers.get_pdist_inds(n, i, j)] == square[i, j]


class FakeSkbioNode:
	"""The part of skbio.TreeNode that compile_tree_from_nj_treenode uses: name, length, parent, preorder()."""

	def __init__(self, name, length, parent=None):
		self.name, self.length, self.parent, self.children = name, length, parent, []

	@classmethod
	def from_node(cls, node, parent=None):
		out = cls(node.name, node.length, parent)
		out.children = [cls.from_node(child, out) for child in node.children]
		return out

	def is_root(self):
		return self.parent is None

	def preorder(self):
		yield self
		for child in self.children:
			yield from child.preorder()


NJ_TREE = '((A:1.0,B:2.0):0.5,C:3.0,(D:0.25,E:0.75)y:1.5);'


def _expected_nj_lists():
	# Pre-order: root, internal, A, B, C, y, D, E; unnamed nodes become internal_<k> in that order.
	names = ['internal_0', 'internal_1', 'A', 'B', 'C', 'y', 'D', 'E']
	edges = [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (5, 6), (5, 7)]
	dists = [0.5, 1.0, 2.0, 3.0, 1.5, 0.25, 0.75]
	return edges, dists, names


def test_compile_tree_from_nj_treenode_lists_edges_in_preorder():
	edges, dists, names = helpers.compile_tree_from_nj_treenode(FakeSkbioNode.from_node(parse_newick(NJ_TREE)))
	assert (edges, dists, names) == _expected_nj_lists()


def test_compile_tree_from_newick_reads_the_file_with_skbio(tmp_path, monkeypatch):
	"""skbio is not among Bonsai's requirements; a stand-in module checks how the file is read."""
	calls = []

	class FakeTreeNode:
		@staticmethod
		def read(filename, convert_underscores=True):
			calls.append((filename, convert_underscores))
			with open(filename) as handle:
				return FakeSkbioNode.from_node(parse_newick(handle.read()))

	monkeypatch.setitem(sys.modules, 'skbio', types.SimpleNamespace(TreeNode=FakeTreeNode))
	path = tmp_path / 'tree.nwk'
	path.write_text(NJ_TREE)
	assert helpers.compile_tree_from_newick(str(path)) == _expected_nj_lists()
	assert calls == [(str(path), False)]


class FakeGraph:
	"""What get_shortest_distances_on_tree uses of an igraph Graph: get_edge_dataframe()."""

	def __init__(self, edges, weights):
		self.frame = pd.DataFrame({'source': [e[0] for e in edges], 'target': [e[1] for e in edges],
		                           'weight': weights, 'name': ['e%d' % k for k in range(len(edges))]})

	def get_edge_dataframe(self):
		return self.frame


def test_get_shortest_distances_on_tree_gives_condensed_path_lengths():
	tree = parse_newick('((A:1.0,B:2.0)x:0.5,C:3.0,(D:0.25,E:0.75)y:1.5)r;')
	order = ['r', 'x', 'A', 'B', 'C', 'y', 'D', 'E']
	index = {name: ind for ind, name in enumerate(order)}
	edges, weights = [], []
	for node in tree.iter_nodes():
		for child in node.children:
			edges.append((index[node.name], index[child.name]))
			weights.append(child.length)
	leaves = [index[name] for name in 'ABCDE']
	got = helpers.get_shortest_distances_on_tree(FakeGraph(edges, weights), indices=leaves)
	expected = [path_length(tree, a, b) for a, b in itertools.combinations('ABCDE', 2)]
	np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_pair_bookkeeping():
	info = {(1, 4): 0.5, (2, 3): 1.5}
	assert helpers.getNodePairInfo(info, (1, 4)) == 0.5
	assert helpers.getNodePairInfo(info, (4, 1)) is None
	merged = helpers.communicateNodePairInfo(dict(info), {(5, 6): 2.5}, mpi_rank=0)
	assert merged == {(1, 4): 0.5, (2, 3): 1.5, (5, 6): 2.5}
	assert helpers.communicateNodePairInfo(dict(info), {(5, 6): 2.5}, mpi_rank=1, allRanksGetInfo=False) is None
	cleaned = helpers.cleanUpPairDict([3, 7], {(1, 4): 0.5, (2, 3): 1.5, (3, 9): 0.1, (5, 6): 2.5})
	assert cleaned == {(1, 4): 0.5, (5, 6): 2.5}
	assert all(isinstance(key, tuple) for key in cleaned)


def test_getAllDLogLs_concatenates_the_rows_of_the_upper_triangle():
	"""Row i holds the gains of pairs (i, j) for j > i; the result is the condensed (pdist-ordered) vector."""
	n = 4
	gains = {(i, j): 10.0 * i + j for i, j in itertools.combinations(range(n), 2)}
	rows = {i: np.array([gains[(i, j)] for j in range(i + 1, n)]) for i in range(n - 1)}
	got = helpers.getAllDLogLs(rows, mpi_rank=0, nChildren=n)
	for (i, j), value in gains.items():
		assert got[helpers.get_pdist_inds(n, i, j)] == value
	assert helpers.getAllDLogLs(rows, mpi_rank=1, nChildren=n) is None


def test_calcTInit_starts_the_ancestor_at_half_the_shorter_branch():
	assert helpers.calcTInit(0.6, 1.0, True) == pytest.approx([1.0, 0.3])
	assert helpers.calcTInit(0.6, 1.0, False) == pytest.approx([0.3, 0.7, 0.3])


def _random_graph(seed, n):
	rng = np.random.default_rng(seed)
	weights = np.triu(rng.uniform(0.1, 10.0, (n, n)), 1)
	return weights


@pytest.mark.parametrize('seed', range(3))
def test_kruskal_algo_finds_a_minimum_spanning_tree(seed):
	n = 7
	weights = _random_graph(seed, n)
	graph = helpers.OwnGraph(n)
	for i, j in itertools.combinations(range(n), 2):
		graph.add_edge(i, j, weights[i, j])
	result = graph.kruskal_algo(verbose=False)
	assert len(result) == n - 1
	assert sum(edge[2] for edge in result) == pytest.approx(minimum_spanning_tree(weights).sum(), rel=1e-12)
	parent, rank = list(range(n)), [0] * n
	for u, v, _ in result:
		assert graph.find(parent, u) != graph.find(parent, v)
		graph.apply_union(parent, rank, graph.find(parent, u), graph.find(parent, v))
	assert len({graph.find(parent, node) for node in range(n)}) == 1


def test_union_find_merges_by_rank():
	graph = helpers.OwnGraph(4)
	parent, rank = [0, 1, 2, 3], [0, 0, 0, 0]
	graph.apply_union(parent, rank, 0, 1)
	assert parent == [0, 0, 2, 3] and rank == [1, 0, 0, 0]
	graph.apply_union(parent, rank, 2, graph.find(parent, 1))
	assert parent == [0, 0, 0, 3] and rank == [1, 0, 0, 0]
	assert graph.find(parent, 2) == 0 and graph.find(parent, 3) == 3


@pytest.mark.parametrize('seed', range(3))
def test_kruskal_algo_maxdegree_two_builds_the_greedy_path(seed):
	"""Edges in order of weight, kept when they join two components and neither end already has two edges: the
	result is a path through every vertex."""
	n = 7
	weights = _random_graph(seed, n)
	graph = helpers.OwnGraph(n)
	for i, j in itertools.combinations(range(n), 2):
		graph.add_edge(i, j, weights[i, j])
	result = graph.kruskal_algo_maxdegree_two(verbose=False)
	component, degree, expected = list(range(n)), [0] * n, []

	def root(i):
		while component[i] != i:
			i = component[i]
		return i

	for w, i, j in sorted((weights[i, j], i, j) for i, j in itertools.combinations(range(n), 2)):
		if root(i) != root(j) and degree[i] < 2 and degree[j] < 2:
			expected.append([i, j])
			component[root(i)] = root(j)
			degree[i] += 1
			degree[j] += 1
		if len(expected) == n - 1:
			break
	assert result == expected
	counts = np.bincount(np.array(result).ravel(), minlength=n)
	assert sorted(counts) == [1, 1] + [2] * (n - 2)


def test_add_celltype_info_to_tree_labels_every_node():
	root = TreeNode(nodeId='root', childNodes=[TreeNode(nodeId='A'), TreeNode(nodeId='x', childNodes=[
		TreeNode(nodeId='B'), TreeNode(nodeId='C')])])
	helpers.add_celltype_info_to_tree(root, {'A': 'neuron', 'C': 'glia'})
	labels = {}

	def collect(node):
		labels[node.nodeId] = node.celltype
		for child in node.childNodes:
			collect(child)

	collect(root)
	assert labels == {'root': 'unknown', 'A': 'neuron', 'x': 'unknown', 'B': 'unknown', 'C': 'glia'}
