"""
downstream_analyses/get_clusters_max_diameter.py (area D).

get_min_pdists_clustering_from_nwk_str runs inside every `bonsai_main.py --step all`: Tree.set_mindist_root
(n_clusters=2) picks the edge Bonsai roots its tree on, and Tree.get_cluster_centers uses the cut edges of a
larger clustering. The expected clusterings here come from a brute-force reference that scores every edge from
explicit pairwise path lengths (written from the harness newick parser, not from Cluster_Tree's recursions), and the
annotation-guided clustering is checked against sklearn's normalized mutual information and a brute-force search
over single cuts.

The edge score Bonsai maximises is not written down anywhere in the README or docstrings; reading the code it is,
for an edge of length t above a subtree holding k of the n cells of the current cluster,
	score = sum over the k*(n-k) separated cell pairs of their path length + k*(n-k)*t
(the second term is the edge's "footfall"; footfall=True scores the footfall alone). The reference computes the
first term pair by pair, so it does not share Bonsai's ds_dists/us_dists bookkeeping.
"""
import itertools
import math
import os
import random

import numpy as np
import pytest
from sklearn.metrics import normalized_mutual_info_score

import downstream_analyses.get_clusters_max_diameter as gcmd
from downstream_analyses.get_cluster_helpers import Cluster_Tree
from bonsai_testdata import DatasetSpec, balanced_newick, caterpillar_newick
from bonsai_testfixtures import bonsai_run_fixture, run_spec
from bonsai_testrun import read_newick_file
from bonsai_testtrees import Node, parse_newick, path_length, write_newick

# ----------------------------------------------------------------------------------------------- reference


def random_newick(seed, n_leaves, polytomy_prob=0.0, zero_leaf_prob=0.0, grid=None):
	"""A random rooted tree with named nodes (cells C<i>, internal nodes n<k>, root 'root'), written the way Bonsai
	writes trees (every node named, lengths as str(float)). Lengths are uniform in [0.05, 1.5], or drawn from the
	values in `grid` (which makes equal and nearly equal edge scores common)."""
	rng = np.random.default_rng(seed)

	def draw():
		if grid is not None:
			return float(grid[int(rng.integers(len(grid)))])
		return float(np.round(rng.uniform(0.05, 1.5), 6))

	nodes = []
	for ind in range(n_leaves):
		length = 0.0 if rng.uniform() < zero_leaf_prob else draw()
		nodes.append(Node('C%d' % ind, length))
	counter = 0
	while len(nodes) > 1:
		size = 3 if (len(nodes) >= 3 and rng.uniform() < polytomy_prob) else 2
		picks = sorted(rng.choice(len(nodes), size, replace=False), reverse=True)
		children = [nodes.pop(int(pick)) for pick in picks]
		nodes.append(Node('n%d' % counter, draw(), children))
		counter += 1
	root = nodes[0]
	root.name, root.length = 'root', None
	return write_newick(root)


class _Forest:
	"""The clusters as a list of rooted subtrees of one parsed tree, cut the way the clustering cuts it."""

	def __init__(self, newick, n_cells=None):
		self.root = parse_newick(newick)
		self.is_leaf = {id(node): node.is_leaf for node in self.root.iter_nodes()}
		self.weight = {}
		self.cell_node = {}
		for node in self.root.iter_nodes():
			if n_cells is not None and node.name in n_cells:
				self.weight[id(node)] = n_cells[node.name]
				self.cell_node[id(node)] = min(n_cells[node.name], 1)
			else:
				self.weight[id(node)] = 1 if self.is_leaf[id(node)] else 0
				self.cell_node[id(node)] = 1 if self.is_leaf[id(node)] else 0
		self.trees = [self.root]

	def edge_scores(self, tree, footfall):
		"""(score, node) for every edge of `tree` in pre-order, scored by brute force over node pairs."""
		nodes = list(tree.iter_nodes())
		total = sum(self.weight[id(node)] for node in nodes)
		out = []
		for node in nodes:
			if node is tree:
				continue
			below = list(node.iter_nodes())
			below_ids = set(id(x) for x in below)
			k = sum(self.weight[id(x)] for x in below)
			score = k * (total - k) * node.length
			if not footfall:
				for a in below:
					if self.weight[id(a)] == 0:
						continue
					for b in nodes:
						if id(b) in below_ids or self.weight[id(b)] == 0:
							continue
						score += self.weight[id(a)] * self.weight[id(b)] * path_length(tree, a.name, b.name)
			out.append((score, node))
		return out

	def best_cut(self, footfall, single_cell_trees_score_zero=True):
		candidates = []
		for tree_ind, tree in enumerate(self.trees):
			n_cell_nodes = sum(self.cell_node[id(node)] for node in tree.iter_nodes())
			if single_cell_trees_score_zero and n_cell_nodes == 1:
				candidates.append((0.0, tree_ind, -1, None))
				continue
			for edge_ind, (score, node) in enumerate(self.edge_scores(tree, footfall)):
				candidates.append((score, tree_ind, edge_ind, node))
		if not candidates:
			return None
		scores = np.array([candidate[0] for candidate in candidates])
		best = int(np.argmax(scores))
		others = np.delete(scores, best)
		if len(others) and scores[best] > 1e-9:
			# A near-tie would make the test depend on rounding: pick seeds and lengths without one.
			assert scores[best] - others.max() > 1e-7 * max(1.0, abs(scores[best])), 'ambiguous test tree'
		return candidates[best]

	def cut(self, tree_ind, node):
		parent = self.trees[tree_ind].parents()[id(node)]
		parent.children = [child for child in parent.children if child is not node]
		self.trees.append(node)
		return parent

	def clusters(self, cell_ids=None):
		out = []
		for tree in self.trees:
			if cell_ids is None:
				out.append([node.name for node in tree.iter_nodes() if self.is_leaf[id(node)]])
			else:
				out.append([node.name for node in tree.iter_nodes() if node.name in cell_ids])
		return out


def reference_min_pdists(newick, n_clusters, cell_ids=None, n_cells=None, footfall=False):
	"""(all_clusterings, cut_edges) as get_min_pdists_clustering_from_nwk_str documents them in its return values:
	one entry 'annot_bnsi_cluster_n<k>' per number of clusters k, each a list of clusters (node IDs in pre-order),
	and the cut edges as (downstream ID, upstream ID)."""
	forest = _Forest(newick, n_cells)
	n_clusters = min(n_clusters, sum(forest.cell_node.values()))
	all_clusterings, edges = {}, []
	while len(forest.trees) < n_clusters:
		score, tree_ind, _, node = forest.best_cut(footfall)
		if score < 1e-9:
			break
		parent = forest.cut(tree_ind, node)
		edges.append((node.name, parent.name))
		all_clusterings['annot_bnsi_cluster_n%d' % len(forest.trees)] = forest.clusters(cell_ids)
	return all_clusterings, edges


def reference_footfall_deprecated(newick, n_clusters, footfall):
	"""The deprecated clusterings: the same greedy cuts over all edges of all clusters, final clusters only."""
	forest = _Forest(newick)
	edges, splits = [], {}
	while len(forest.trees) < n_clusters:
		_, tree_ind, _, node = forest.best_cut(footfall, single_cell_trees_score_zero=False)
		max_tree = forest.trees[tree_ind]
		parent = forest.cut(tree_ind, node)
		edges.append((node.name, parent.name))
		splits[(node.name, parent.name)] = [
			[x.name for x in max_tree.iter_nodes() if forest.is_leaf[id(x)]],
			[x.name for x in node.iter_nodes() if forest.is_leaf[id(x)]]]
	return forest.clusters(), edges, splits


MIXED_TREES = [
	random_newick(seed=3, n_leaves=6),
	random_newick(seed=5, n_leaves=9, polytomy_prob=0.4),
	random_newick(seed=8, n_leaves=11, polytomy_prob=0.3, zero_leaf_prob=0.2),
	# a ladder (caterpillar) with distinct lengths
	'((((((C0:0.21,C1:0.17)i0:0.33,C2:0.24)i1:0.41,C3:0.19)i2:0.29,C4:0.27)i3:0.37,C5:0.22)i4:0.31,C6:0.26)root;',
]


# ------------------------------------------------------------------------------------ min-pdists clustering

@pytest.mark.parametrize('newick', MIXED_TREES)
@pytest.mark.parametrize('footfall', [False, True])
def test_min_pdists_clusterings_match_brute_force(newick, footfall):
	"""Every intermediate clustering and every cut edge, for up to 6 clusters, equal the brute-force greedy cuts."""
	got_clusterings, got_edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=6, verbose=False,
	                                                                          footfall=footfall)
	want_clusterings, want_edges = reference_min_pdists(newick, 6, footfall=footfall)
	assert got_edges == want_edges
	assert list(got_clusterings) == list(want_clusterings)
	assert got_clusterings == want_clusterings


@pytest.mark.parametrize('seed', range(6))
def test_two_cluster_edge_matches_brute_force(seed):
	"""The n_clusters=2 call Tree.set_mindist_root makes, on random 5-12 leaf trees with polytomies and cells on
	zero-length edges: the one cut edge is the brute-force maximum."""
	newick = random_newick(seed=100 + seed, n_leaves=5 + seed, polytomy_prob=0.3, zero_leaf_prob=0.15)
	parsed = parse_newick(newick)
	cell_ids = parsed.leaf_names()
	_, edges = gcmd.get_min_pdists_clustering_from_nwk_str(tree_nwk_str=newick, n_clusters=2, cell_ids=cell_ids,
	                                                       get_cell_ids_all_splits=False, node_id_to_n_cells=None,
	                                                       verbose=False)
	assert edges == reference_min_pdists(newick, 2)[1]


def _node_distances(root):
	"""Path lengths between every pair of nodes of a parsed tree: ({id(node): row}, matrix)."""
	nodes = list(root.iter_nodes())
	row = {id(node): ind for ind, node in enumerate(nodes)}
	neighbours = [[] for _ in nodes]
	for node in nodes:
		for child in node.children:
			neighbours[row[id(node)]].append((row[id(child)], child.length))
			neighbours[row[id(child)]].append((row[id(node)], child.length))
	dist = np.zeros((len(nodes), len(nodes)))
	for start in range(len(nodes)):
		seen, stack = {start}, [(start, 0.0)]
		while stack:
			ind, d = stack.pop()
			dist[start, ind] = d
			for other, length in neighbours[ind]:
				if other not in seen:
					seen.add(other)
					stack.append((other, d + length))
	return row, dist


@pytest.mark.parametrize('seed, n_leaves, footfall, grid', [
	(301, 80, False, None), (302, 80, True, None), (303, 150, False, None), (415, 100, False, None),
	(304, 120, False, (0.1, 0.2, 0.3)), (305, 120, True, (0.1, 0.2, 0.3))])
def test_min_pdists_cuts_on_larger_trees_are_brute_force_optimal(seed, n_leaves, footfall, grid):
	"""Random trees of 80 to 150 cells (polytomies, cells on zero-length edges). On seed 415 the best and second-best
	cut of one step differ by 2e-5 (relative); with three branch lengths only (the last two cases) equal and nearly
	equal edge scores occur. The clustering is replayed cut by cut: before each cut, the edge it cuts scores within
	1e-9 (relative) of the best edge of every current cluster, scored by brute force over cell pairs, and the
	reported clustering is the pieces left by the cuts so far. Either edge of a near-tie is accepted, so the test
	does not depend on the order in which the distances are summed."""
	n_clusters = 12
	newick = random_newick(seed=seed, n_leaves=n_leaves, polytomy_prob=0.3, zero_leaf_prob=0.1, grid=grid)
	got, edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=n_clusters, verbose=False,
	                                                       footfall=footfall)
	assert list(got) == ['annot_bnsi_cluster_n%d' % k for k in range(2, n_clusters + 1)]
	assert len(edges) == n_clusters - 1
	forest = _Forest(newick)
	row, dist = _node_distances(forest.root)
	for step, (child_name, parent_name) in enumerate(edges):
		best, chosen = -np.inf, None
		for tree_ind, tree in enumerate(forest.trees):
			nodes = list(tree.iter_nodes())
			cells = [node for node in nodes if forest.is_leaf[id(node)]]
			if len(cells) < 2:
				continue
			cell_rows = np.array([row[id(cell)] for cell in cells])
			for node in nodes[1:]:
				below = set(id(x) for x in node.iter_nodes())
				mask = np.array([id(cell) in below for cell in cells])
				k = int(mask.sum())
				score = k * (len(cells) - k) * node.length
				if not footfall:
					score += dist[np.ix_(cell_rows[mask], cell_rows[~mask])].sum()
				best = max(best, score)
				if node.name == child_name:
					chosen = (score, tree_ind, node)
		assert chosen is not None, 'cut edge %r is not an edge of a current cluster' % child_name
		score, tree_ind, node = chosen
		assert forest.trees[tree_ind].parents()[id(node)].name == parent_name
		assert score >= best - 1e-9 * abs(best), (step, child_name, score, best)
		forest.cut(tree_ind, node)
		assert got['annot_bnsi_cluster_n%d' % (step + 2)] == forest.clusters()


def test_equal_scores_cut_the_edge_met_first():
	"""Exact ties are common on Bonsai's own trees: a tree rooted by set_mindist_root has a root with two equal
	half-edges, and both halves separate the same cells. With exactly equal scores (dyadic lengths, so every
	summation order gives the same number) the edge met first in pre-order is cut, which decides where Bonsai puts
	the new root."""
	for newick, want in (('((A:0.25,B:0.75)p:0.5,(C:0.5,D:0.5)c:0.5)mindist_root;', ('p', 'mindist_root')),
	                     ('((C:0.5,D:0.5)c:0.5,(A:0.25,B:0.75)p:0.5)mindist_root;', ('c', 'mindist_root'))):
		forest = _Forest(newick)
		scores = forest.edge_scores(forest.root, footfall=False)
		assert scores[0][0] == max(score for score, _ in scores)
		assert [node.name for score, node in scores if score == scores[0][0]] == [want[0], [
			name for name in ('p', 'c') if name != want[0]][0]]
		_, edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=2, verbose=False)
		assert edges == [want]


def test_cell_ids_choose_the_nodes_reported_not_the_cut():
	"""cell_ids lists the nodes that appear in the clusters (internal nodes included); the cuts still weigh the
	leaves only, so the edges are the same as without cell_ids."""
	newick = '(((A:0.2,B:0.3)x:0.9,(C:0.25,D:0.1)y:0.7)u:0.4,(E:0.6,F:0.15)z:1.3)root;'
	cell_ids = ['A', 'x', 'C', 'D', 'E', 'F', 'z']
	got, edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=4, cell_ids=cell_ids, verbose=False)
	plain, plain_edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=4, verbose=False)
	want, want_edges = reference_min_pdists(newick, 4, cell_ids=set(cell_ids))
	assert edges == plain_edges == want_edges
	assert got == want
	for clustering in got.values():
		assert sorted(itertools.chain(*clustering)) == sorted(cell_ids)


def test_cell_counts_weight_the_score():
	"""node_id_to_n_cells (Bonsai-scout passes the number of cells at each vertex) weighs every pair by the
	product of the counts; a vertex with zero cells is not a cell node. Different weights move the cut."""
	newick = '(((A:0.5,B:0.45)x:0.6,C:1.0)u:0.8,(D:0.4,E:0.55)z:0.7)root;'
	counts = {'A': 6, 'B': 1, 'C': 1, 'D': 1, 'E': 2, 'x': 0, 'u': 3, 'z': 0, 'root': 0}
	got, edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=4, node_id_to_n_cells=counts,
	                                                       verbose=False)
	want, want_edges = reference_min_pdists(newick, 4, n_cells=counts)
	assert (got, edges) == (want, want_edges)
	unweighted_edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=4, verbose=False)[1]
	assert edges != unweighted_edges


def test_number_of_clusters_is_capped_by_the_cells():
	"""Asking for more clusters than cells gives one clustering per number of clusters up to one cell each."""
	newick = random_newick(seed=21, n_leaves=5)
	got, edges = gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=50, verbose=False)
	assert list(got) == ['annot_bnsi_cluster_n%d' % k for k in range(2, 6)]
	assert len(edges) == 4
	assert sorted(map(tuple, got['annot_bnsi_cluster_n5'])) == sorted((name,) for name in 'C0 C1 C2 C3 C4'.split())
	assert (got, edges) == reference_min_pdists(newick, 50)


def test_tree_without_length_cannot_be_cut():
	"""When no edge separates cells by a positive distance, nothing is cut and both outputs are empty."""
	got, edges = gcmd.get_min_pdists_clustering_from_nwk_str('((A:0.0,B:0.0)x:0.0,C:0.0)root;', n_clusters=3,
	                                                       verbose=False)
	assert got == {} and edges == []


def test_unnamed_internal_nodes_get_internal_ids():
	"""A newick string without internal names: the edges and clusters name them internal_<k> in the order their
	closing parentheses appear."""
	newick = '(((A:0.2,B:0.3):0.9,(C:0.25,D:0.1):0.7):0.4,(E:0.6,F:0.15):1.3);'
	named = '(((A:0.2,B:0.3)internal_0:0.9,(C:0.25,D:0.1)internal_1:0.7)internal_2:0.4,(E:0.6,F:0.15)internal_3:1.3)' \
	        'internal_4;'
	assert gcmd.get_min_pdists_clustering_from_nwk_str(newick, n_clusters=4, verbose=False) == \
		reference_min_pdists(named, 4)


def test_clustering_on_a_cluster_tree_object():
	"""get_min_pdists_clustering takes a Cluster_Tree directly, also one whose vertices were not numbered yet."""
	newick = MIXED_TREES[1]
	tree = Cluster_Tree()
	tree.from_newick(nwk_str=newick)
	assert tree.vert_ind_to_node is None
	got = gcmd.get_min_pdists_clustering(tree, 5, verbose=False)
	assert got == reference_min_pdists(newick, 5)


# ----------------------------------------------------------------------- the edge Bonsai roots its trees on

MINDIST_DATA = DatasetSpec(newick=caterpillar_newick(10, internal_length=0.3, leaf_length=0.15), n_genes=40, seed=31,
                           layout='sanity2')
d_mindist_run = bonsai_run_fixture('d_mindist_run', run_spec(MINDIST_DATA, label='d_mindist', nnn_n_randommoves=20,
                                                             nnn_n_randomtrees=2))


def test_final_bonsai_tree_is_rooted_on_the_min_pdists_edge(d_mindist_run):
	"""On a whole --step all run, Bonsai roots the final tree in the middle of the edge the min-pdists clustering
	cuts first: join the root's two edges into one, and the brute-force best edge of that unrooted tree is the joined
	edge, split in two equal halves."""
	d_mindist_run.check()
	tree = read_newick_file(d_mindist_run.final_folder())
	assert len(tree.children) == 2
	left, right = tree.children
	assert left.length == pytest.approx(right.length, rel=1e-12)
	# Hang `right` below `left` on the joined edge, so `left` becomes the root of the same unrooted tree.
	joined = Node(right.name, left.length + right.length, right.children)
	rerooted = Node(left.name, None, left.children + [joined])
	if not left.children:
		pytest.skip('the root has a cell as one child; the joined edge cannot be re-hung')
	forest = _Forest(write_newick(rerooted))
	_, _, _, node = forest.best_cut(footfall=False)
	assert node.name == right.name


# ------------------------------------------------------------------------------------- deprecated variants

@pytest.mark.parametrize('newick', MIXED_TREES[:3])
def test_deprecated_min_pdists_clustering_matches_brute_force(newick):
	"""The deprecated min-pdists clustering returns only the final clustering, plus the cell IDs on both sides of
	every split when asked."""
	parsed_leaves = parse_newick(newick).leaf_names()
	clusters, edges = gcmd.get_min_pdists_clustering_from_nwk_str_deprecated(newick, 4, verbose=False)
	want_clusters, want_edges, want_splits = reference_footfall_deprecated(newick, 4, footfall=False)
	assert (clusters, edges) == (want_clusters, want_edges)
	clusters, edges, splits = gcmd.get_min_pdists_clustering_from_nwk_str_deprecated(
		newick, 4, get_cell_ids_all_splits=True, verbose=False)
	assert splits == want_splits
	# With cell_ids, only those cells are reported.
	keep = parsed_leaves[::2]
	clusters, _ = gcmd.get_min_pdists_clustering_from_nwk_str_deprecated(newick, 4, cell_ids=keep, verbose=False)
	assert clusters == [[name for name in cluster if name in keep] for cluster in want_clusters]


@pytest.mark.parametrize('newick', MIXED_TREES[:3])
def test_footfall_clustering_matches_brute_force(newick):
	"""Footfall clustering cuts the edge with the largest k*(n-k)*t."""
	clusters, edges = gcmd.get_footfall_clustering_from_nwk_str_deprecated(newick, 4)
	want_clusters, want_edges, want_splits = reference_footfall_deprecated(newick, 4, footfall=True)
	assert (clusters, edges) == (want_clusters, want_edges)
	clusters, edges, splits = gcmd.get_footfall_clustering_from_nwk_str_deprecated(newick, 4,
	                                                                                get_cell_ids_all_splits=True)
	assert splits == want_splits
	tree = Cluster_Tree()
	tree.from_newick_string(newick)
	keep = parse_newick(newick).leaf_names()[1::2]
	clusters, _ = gcmd.get_footfall_clustering(tree, 4, cell_ids=keep)
	assert clusters == [[name for name in cluster if name in keep] for cluster in want_clusters]


# --------------------------------------------------------------------------------- max-diameter clustering

def _component_diameter(tree, names):
	names = list(names)
	if len(names) < 2:
		return 0.0
	return max(path_length(tree, a, b) for a, b in itertools.combinations(names, 2))


def _min_clusters_brute_force(tree, threshold):
	"""Fewest connected pieces (cutting edges) whose leaf sets all have a diameter below the threshold."""
	edges = [node for node in tree.iter_nodes() if node is not tree]
	best = None
	for n_cuts in range(len(edges) + 1):
		for cut in itertools.combinations(range(len(edges)), n_cuts):
			cut_ids = set(id(edges[ind]) for ind in cut)
			pieces = []
			for start in [tree] + [edges[ind] for ind in cut]:
				stack, leaves = [start], []
				while stack:
					node = stack.pop()
					if node.is_leaf:
						leaves.append(node.name)
					stack.extend(child for child in node.children if id(child) not in cut_ids)
				pieces.append(leaves)
			if all(_component_diameter(tree, leaves) < threshold for leaves in pieces if leaves):
				return n_cuts + 1
	return best


@pytest.mark.parametrize('seed,threshold', [(41, 1.5), (42, 2.0), (43, 2.5), (44, 3.0)])
def test_max_diameter_clusters_partition_the_cells_below_the_threshold(seed, threshold):
	"""Max-diameter clustering: the clusters partition the leaves, every cluster is a connected piece of the tree
	whose largest leaf-to-leaf distance is below the threshold, and (checked by enumerating every set of cut edges on
	these 7-leaf binary trees) no partition into fewer such pieces exists."""
	newick = random_newick(seed=seed, n_leaves=7)
	tree = parse_newick(newick)
	clusters = gcmd.get_max_diam_clustering_from_nwk_str(newick, threshold)
	assert sorted(itertools.chain(*clusters)) == sorted(tree.leaf_names())
	for cluster in clusters:
		assert _component_diameter(tree, cluster) < threshold
	assert len(clusters) == _min_clusters_brute_force(tree, threshold)


def test_max_diameter_hand_example():
	"""((A:1,B:1)x:1,(C:3,D:0.5)y:1)r with threshold 3: at y, C-D = 3.5 is too far, C is cut off; at x, A-B = 2
	stays; at r, A..D = 1+1+1+0.5 = 3.5 is too far, so the x side (longest reach 2 against 1.5) is cut off."""
	clusters = gcmd.get_max_diam_clustering_from_nwk_str('((A:1,B:1)x:1,(C:3,D:0.5)y:1)r;', 3.0)
	assert clusters == [['C'], ['A', 'B'], ['D']]
	tree = Cluster_Tree()
	tree.from_newick_string('((A:1,B:1)x:1,(C:3,D:0.5)y:1)r;')
	assert gcmd.get_max_diam_clustering(tree, 3.0, cell_ids=['A', 'C', 'x']) == [['C'], ['x', 'A'], []]


# ------------------------------------------------------------------------------ annotation-based clustering
#
# Bonsai skips a candidate cut whose joint-entropy change is exactly 0.0, which is also what a cut that splits no
# label within its cluster gives when the rounding happens to cancel, so it can skip a cut that separates labels
# cleanly. The tests below therefore check properties that hold whether or not such cuts are taken:
# the clusters are the pieces left by the reported cut edges, the reported value is their normalized mutual
# information (NMI), and no single cut that splits some label inside its cluster (those are never skipped), and no
# merge back over a cut edge, improves the NMI beyond the tolerances.

def _nmi(clusters, annotation, node_of=None):
	node_of = node_of or {cell: cell for cell in annotation}
	cluster_of = {}
	for ind, cluster in enumerate(clusters):
		for node in cluster:
			cluster_of[node] = ind
	cells = [cell for cell in annotation if node_of[cell] in cluster_of]
	return normalized_mutual_info_score([annotation[cell] for cell in cells],
	                                    [cluster_of[node_of[cell]] for cell in cells], average_method='geometric')


def _pieces(newick, cut_edges, keep=None):
	"""The parsed tree, the ids of the nodes below cut edges, and the leaf sets (or `keep` node sets) of the
	connected pieces left after removing the cut edges, keyed by the id of the node each piece hangs from."""
	tree = parse_newick(newick)
	parents = tree.parents()
	cut_ids = set()
	for child_name, parent_name in cut_edges:
		child = tree.find(child_name)
		assert parents[id(child)].name == parent_name, 'reported cut edge is not an edge of the tree'
		cut_ids.add(id(child))
	pieces = {}
	for start in [tree] + [node for node in tree.iter_nodes() if id(node) in cut_ids]:
		stack, members = [start], []
		while stack:
			node = stack.pop()
			if (keep is None and node.is_leaf) or (keep is not None and node.name in keep):
				members.append(node.name)
			stack.extend(child for child in node.children if id(child) not in cut_ids)
		pieces[id(start)] = frozenset(members)
	return tree, cut_ids, pieces


def _splits_a_label(part, cluster, annotation):
	labels_in = set(annotation[cell] for cell in part if cell in annotation)
	labels_out = set(annotation[cell] for cell in cluster - part if cell in annotation)
	return bool(labels_in & labels_out)


def _check_greedy_optimum(newick, annotation, clusters, cut_edges, nmi, cutting_tol=1e-4):
	tree, cut_ids, pieces = _pieces(newick, cut_edges)
	cluster_sets = [frozenset(cluster) for cluster in clusters]
	assert set(cluster_sets) == set(piece for piece in pieces.values() if piece)
	assert sorted(itertools.chain(*clusters)) == sorted(tree.leaf_names())
	assert nmi == pytest.approx(_nmi(clusters, annotation), abs=1e-12)
	parents = tree.parents()
	for node in tree.iter_nodes():
		if node is tree:
			continue
		if id(node) in cut_ids:
			# Merging the piece below this cut edge back into the piece its parent belongs to.
			top = parents[id(node)]
			while id(top) not in cut_ids and top is not tree:
				top = parents[id(top)]
			merged_piece = pieces[id(node)] | pieces[id(top)]
			merged = [c for c in cluster_sets if c not in (pieces[id(node)], pieces[id(top)])] + [merged_piece]
			assert _nmi(merged, annotation) <= nmi - 1e-5 + 1e-12
			continue
		below = frozenset(node.leaf_names())
		for cluster in cluster_sets:
			part = cluster & below
			if part and part != cluster and _splits_a_label(part, cluster, annotation):
				split = [c for c in cluster_sets if c != cluster] + [part, cluster - part]
				assert _nmi(split, annotation) <= nmi + cutting_tol + 1e-12


ANNOT_NEWICK = ('((((A1:0.2,A2:0.3)a:0.5,(A3:0.1,B1:0.4)ab:0.6)u:0.3,((B2:0.2,B3:0.25)b:0.7,B4:0.3)bb:0.4)v:0.2,'
                '((C1:0.3,C2:0.1)c:0.8,(C3:0.2,(A4:0.1,C4:0.3)ac:0.2)cc:0.4)w:0.5)root;')
ANNOTATION = {'A1': 'a', 'A2': 'a', 'A3': 'a', 'A4': 'a', 'B1': 'b', 'B2': 'b', 'B3': 'b', 'B4': 'b',
              'C1': 'c', 'C2': 'c', 'C3': 'c', 'C4': 'c'}


@pytest.mark.parametrize('random_sampling', [False, True])
def test_annotation_clustering_reports_the_nmi_of_a_local_optimum(random_sampling):
	"""Greedy (and annealed) annotation-guided clustering: the clusters are the pieces between the reported cut
	edges, the returned value is their NMI (geometric-mean normalization, as sklearn computes it), no label-splitting
	single cut raises it by more than cutting_tol, and no merge over a cut edge keeps it within 1e-5."""
	clusters, cut_edges, nmi = gcmd.get_annotation_based_clustering_from_nwk_str(
		ANNOT_NEWICK, ANNOTATION, verbose=False, random_sampling=random_sampling)
	assert len(clusters) > 1
	_check_greedy_optimum(ANNOT_NEWICK, ANNOTATION, clusters, cut_edges, nmi)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_annotation_clustering_on_random_trees(seed):
	"""The same properties on random 14-leaf trees with three noisy labels."""
	newick = random_newick(seed=200 + seed, n_leaves=14, polytomy_prob=0.2)
	rng = np.random.default_rng(seed)
	labels = {'C%d' % ind: 'abc'[int(rng.integers(3))] for ind in range(14)}
	clusters, cut_edges, nmi = gcmd.get_annotation_based_clustering_from_nwk_str(newick, labels, verbose=False)
	_check_greedy_optimum(newick, labels, clusters, cut_edges, nmi)


def test_first_cut_is_the_best_label_splitting_cut():
	"""With max_moves=0 the clustering makes one move; its cut is at least as good as every single cut that splits
	some label (the cut with the highest NMI, unless a label-preserving cut is better still)."""
	clusters, cut_edges, _ = gcmd.get_annotation_based_clustering_from_nwk_str(ANNOT_NEWICK, ANNOTATION, verbose=False,
	                                                                          max_moves=0)
	assert len(clusters) == 2 and len(cut_edges) == 1
	tree = parse_newick(ANNOT_NEWICK)
	everyone = frozenset(tree.leaf_names())
	assert set(frozenset(c) for c in clusters) == set(_pieces(ANNOT_NEWICK, cut_edges)[2].values())
	got = _nmi(clusters, ANNOTATION)
	for node in tree.iter_nodes():
		below = frozenset(node.leaf_names())
		if node is not tree and _splits_a_label(below, everyone, ANNOTATION):
			assert _nmi([below, everyone - below], ANNOTATION) <= got + 1e-12


def test_random_sampling_annotation_clustering_is_seeded_and_tracked(tmp_path):
	"""The annealing variant: two calls give the same result (fixed seed), and the tracking file has one row per
	move with the documented columns, ending at the returned clustering and NMI."""
	tracking = str(tmp_path / 'stats.tsv')
	first = gcmd.get_annotation_based_clustering_from_nwk_str(ANNOT_NEWICK, ANNOTATION, verbose=False,
	                                                         random_sampling=True, tracking_path=tracking)
	second = gcmd.get_annotation_based_clustering_from_nwk_str(ANNOT_NEWICK, ANNOTATION, verbose=False,
	                                                          random_sampling=True)
	assert first[0] == second[0] and first[1] == second[1] and first[2] == second[2]
	clusters, cut_edges, nmi = first
	import pandas as pd
	stats = pd.read_csv(tracking, sep='\t')
	assert list(stats.columns) == ['n_moves', 'norm_mut_info', 'n_clusters', 'random_sampling', 'greedy',
	                               'optimality_decrease']
	assert (np.diff(stats['n_moves']) > 0).all()
	assert stats['n_clusters'].iloc[-1] == len(clusters)
	assert stats['norm_mut_info'].iloc[-1] == pytest.approx(nmi, abs=1e-12)


def test_annotation_clustering_trivial_annotations_return_one_cluster():
	"""One label for every cell, or a different label for every cell: one cluster, no cuts, NMI reported as 1."""
	leaves = parse_newick(ANNOT_NEWICK).leaf_names()
	for labels in ({cell: 'same' for cell in leaves}, {cell: cell for cell in leaves}):
		clusters, cut_edges, nmi = gcmd.get_annotation_based_clustering_from_nwk_str(ANNOT_NEWICK, labels,
		                                                                             verbose=False)
		assert clusters == [leaves] and cut_edges == [] and nmi == 1.0


def test_annotation_clustering_partial_annotation_and_cell_to_node_mapping():
	"""Cells mapped onto tree nodes (several cells per node, a cell at an internal node) through
	cell_id_to_node_id; node_ids_to_clst picks which nodes are reported; a node without annotated cells is still
	clustered; a cell whose node is not in the tree is skipped. The NMI counts each annotated cell at its node."""
	newick = '(((n1:0.3,n2:0.2)p:0.9,(n3:0.1,n4:0.5)q:0.8)u:0.3,(n5:0.4,n6:0.3)r:1.1)root;'
	node_of = {'c1': 'n1', 'c2': 'n1', 'c3': 'n2', 'c4': 'p', 'c5': 'n3', 'c6': 'n4', 'c7': 'n4', 'c8': 'n5',
	           'c9': 'n2', 'c10': 'missing', 'c11': 'n3'}
	labels = {'c1': 'x', 'c2': 'x', 'c3': 'x', 'c4': 'y', 'c5': 'y', 'c6': 'y', 'c7': 'x', 'c8': 'z', 'c9': 'y',
	          'c10': 'z', 'c11': 'z'}
	reported = ['n1', 'n2', 'p', 'n3', 'n4', 'n5', 'n6']
	clusters, cut_edges, nmi = gcmd.get_annotation_based_clustering_from_nwk_str(
		newick, labels, cell_id_to_node_id=node_of, node_ids_to_clst=reported, verbose=False, random_sampling=False)
	assert len(clusters) > 1
	assert sorted(itertools.chain(*clusters)) == sorted(reported)
	assert set(frozenset(c) for c in clusters) == \
		set(piece for piece in _pieces(newick, cut_edges, keep=set(reported))[2].values() if piece)
	annotated = {cell: label for cell, label in labels.items() if node_of[cell] != 'missing'}
	assert nmi == pytest.approx(_nmi(clusters, annotated, node_of=node_of), abs=1e-12)


def test_annotation_clustering_cutting_tol_and_max_moves():
	"""A cutting_tol above 1 forbids every cut; max_moves=0 allows exactly one move."""
	clusters, cut_edges, nmi = gcmd.get_annotation_based_clustering_from_nwk_str(
		ANNOT_NEWICK, ANNOTATION, verbose=False, cutting_tol=2.0)
	assert len(clusters) == 1 and cut_edges == [] and nmi == 0.0
	clusters, cut_edges, _ = gcmd.get_annotation_based_clustering_from_nwk_str(
		ANNOT_NEWICK, ANNOTATION, verbose=False, max_moves=0)
	assert len(clusters) == 2 and len(cut_edges) == 1


def test_prohibit_small_clusters_keeps_clusters_above_the_lower_bound():
	"""prohibit_small_clsts: a cluster may not be split off when it holds fewer annotated cells than
	min(average label size / 10, cells / 200), unless it captures more than 30% of some label. Two stray 'b' cells
	at different places inside a clade of 150 'a' cells (302 cells, so the bound is 1.51 cells): unconstrained
	greedy cutting isolates a stray cell, the constrained clustering keeps every cluster at two or more cells."""
	a_cells = ['a%d' % ind for ind in range(150)]
	b_cells = ['b%d' % ind for ind in range(150)]
	rng = np.random.default_rng(7)

	def comb(names, prefix):
		out = '%s:0.1' % names[0]
		for ind, name in enumerate(names[1:]):
			out = '(%s,%s:%r)%s%d' % (out, name, float(np.round(rng.uniform(0.05, 0.2), 4)), prefix, ind)
			if ind < len(names) - 2:
				out += ':0.01'
		return out

	newick = '(((%s:0.01,bx:0.05)A0:0.3,(%s:0.01,by:0.07)A1:0.2)A:1.0,%s:1.0)root;' % (
		comb(a_cells[:75], 'ia'), comb(a_cells[75:], 'ja'), comb(b_cells, 'ib'))
	labels = {cell: 'a' for cell in a_cells}
	labels.update({cell: 'b' for cell in b_cells})
	labels['bx'] = labels['by'] = 'b'
	free = gcmd.get_annotation_based_clustering_from_nwk_str(newick, labels, verbose=False)[0]
	held = gcmd.get_annotation_based_clustering_from_nwk_str(newick, labels, verbose=False,
	                                                        prohibit_small_clsts=True)[0]
	assert ['bx'] in free or ['by'] in free
	assert all(len(cluster) >= 2 for cluster in held)
	assert sorted(itertools.chain(*held)) == sorted(labels)


def test_entropy_bookkeeping_helpers():
	"""get_new_mut_info is (H_ent_diff + dC - dJ) / sqrt(H_annot * (H_clade + dC)), zero without a positive
	normalization; test_move_acceptance accepts with probability exp(-beta * (1 - new / orig))."""
	assert gcmd.get_new_mut_info(1.0, 4.0, 2.0, 1.0, 0.5) == pytest.approx(1.5 / math.sqrt(4.0 * 3.0), rel=1e-15)
	assert gcmd.get_new_mut_info(1.0, 4.0, 0.0, 0.0, 0.5) == 0.0
	beta, orig, new = 2.0, 0.8, 0.6
	prob = math.exp(-beta * (1 - new / orig))
	random.seed(12)
	draws = [random.random() for _ in range(200)]
	random.seed(12)
	accepted = [gcmd.test_move_acceptance(beta, orig, new) for _ in range(200)]
	assert accepted == [prob > draw for draw in draws]


def test_merge_move_entropy_change():
	"""get_ent_change_merge_move: the clade and joint entropy (times the number of cells) a merge of a cut subtree
	back into the cluster of its old parent would remove, from n log n terms."""
	tree = Cluster_Tree()
	tree.from_newick_string('((A:1,B:1)x:1,(C:1,D:1)y:1)r;')
	nodes = {node.nodeId: node for node in tree.vert_ind_to_node.values()}
	x, r = nodes['x'], nodes['r']
	x.ds_n_annots, x.ds_annot_counts = 3, np.array([2, 1])
	r.ds_n_annots, r.ds_annot_counts = 5, np.array([1, 4])

	def nlogn(values):
		return sum(v * math.log(v) for v in np.atleast_1d(values) if v > 0)

	x.clade_ent_contrib_ds, x.annot_ent_contrib_ds = nlogn(3), nlogn([2, 1])
	r.clade_ent_contrib_ds, r.annot_ent_contrib_ds = nlogn(5), nlogn([1, 4])
	x.old_parent_node = nodes['y']
	nodes['y'].belongs_to_tree = tree
	tree.root = r
	gcmd.get_ent_change_merge_move(x, n_labels=2)
	assert x.revert_clade_ent_change == pytest.approx(nlogn(3) + nlogn(5) - nlogn(8), rel=1e-12)
	assert x.revert_annot_ent_change == pytest.approx(nlogn([2, 1]) + nlogn([1, 4]) - nlogn([3, 5]), rel=1e-12)


# ---------------------------------------------------------------------------------------- small utilities

def test_get_cluster_assignments_table():
	"""One column per clustering, one row per ID (sorted), entries 'cl_<cluster index>'; a node that stands for
	several cell(state)s gives each of them the node's cluster; IDs missing from a clustering are NaN."""
	clusterings = {'two': [['b', 'n1'], ['a']], 'three': [['b'], ['n1'], ['a', 'c']]}
	df = gcmd.get_cluster_assignments(clusterings, node_ids_multiple_cs_ids={'n1': ['cs1', 'cs2']})
	assert list(df.columns) == ['two', 'three']
	assert list(df.index) == ['a', 'b', 'c', 'cs1', 'cs2']
	assert df.loc['a', 'two'] == 'cl_1' and df.loc['b', 'two'] == 'cl_0' and df.loc['cs1', 'two'] == 'cl_0'
	assert df.loc['cs2', 'three'] == 'cl_1' and df.loc['c', 'three'] == 'cl_2'
	assert isinstance(df.loc['c', 'two'], float) and math.isnan(df.loc['c', 'two'])
	plain = gcmd.get_cluster_assignments({'k': [['x'], ['y']]})
	assert plain.to_dict() == {'k': {'x': 'cl_0', 'y': 'cl_1'}}


def test_get_clustering_lists():
	"""Leaves (or the requested node IDs) of each cluster tree, in pre-order."""
	trees = []
	for newick in ('((A:1,B:1)x:1,C:1)r;', '(D:1,E:1)y;'):
		tree = Cluster_Tree()
		tree.from_newick_string(newick)
		trees.append(tree)
	assert gcmd.get_clustering_lists(trees, None) == [['A', 'B', 'C'], ['D', 'E']]
	assert gcmd.get_clustering_lists(trees, {'x', 'C', 'y', 'E'}) == [['x', 'C'], ['y', 'E']]
