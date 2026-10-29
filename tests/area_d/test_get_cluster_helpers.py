"""
downstream_analyses/get_cluster_helpers.py (area D): Cluster_Tree and Cluster_TreeNode, the slim tree the
clusterings (and through them Tree.set_mindist_root in every Bonsai run) parse Bonsai's newick strings into.
Expected values come from the harness newick parser and path lengths (tests/bonsai_testtrees.py), the edgeInfo /
vertInfo formats of README "The Bonsai results", and n log n entropy terms worked out by hand.
"""
import itertools
import math

import numpy as np
import pytest

from downstream_analyses.get_cluster_helpers import (Cluster_Tree, Cluster_TreeNode, entropy, entropy_non_norm,
                                                     getEdgeDistVertNamesFromNode)
from bonsai_testrun import read_edge_info, read_vert_info, tree_from_folder
from bonsai_testtrees import clades, parse_newick, path_length, rf_distance, splits, write_newick

TREE = '(((A:0.3,B:0.7)i:0.4,C:0.2)j:0.1,(D:0.5,E:0.05,F:0.2)k:0.3,G:0.4)r;'
BINARY = '(((A:0.3,B:0.7)i:0.4,C:0.2)j:0.1,((D:0.5,E:0.05)l:0.6,F:0.2)k:0.3)r;'


def cluster_tree(newick):
	tree = Cluster_Tree()
	tree.from_newick_string(newick)
	return tree


def as_node_tree(cluster_node):
	"""Convert a Cluster_TreeNode subtree to the harness Node representation."""
	from bonsai_testtrees import Node
	return Node(cluster_node.nodeId, cluster_node.tParent, [as_node_tree(child) for child in cluster_node.childNodes])


def leaf_distances(node_tree, names=None):
	names = node_tree.leaf_names() if names is None else names
	return {(a, b): path_length(node_tree, a, b) for a, b in itertools.combinations(sorted(names), 2)}


# --------------------------------------------------------------------------------------------- parsing

@pytest.mark.parametrize('newick', [TREE, BINARY, '((A:1e-05,B:0.0)x:2.5,C:3)root;'])
def test_from_newick_string_reproduces_the_tree(newick):
	"""Names, lengths, child order and leaf flags equal the harness parse; vertices are numbered in pre-order
	(depth-first, children in written order) and every child knows its parent."""
	tree = cluster_tree(newick)
	want = parse_newick(newick)
	got = as_node_tree(tree.root)
	assert write_newick(got) == write_newick(want)
	want_order = [node.name for node in want.iter_nodes()]
	assert [tree.vert_ind_to_node[ind].nodeId for ind in range(len(want_order))] == want_order
	assert tree.nNodes == len(want_order)
	for node in tree.vert_ind_to_node.values():
		assert node.isLeaf == (len(node.childNodes) == 0)
		for child in node.childNodes:
			assert child.parentNode is node
	assert tree.root.isRoot and tree.root.parentNode is None


def test_unnamed_nodes_get_internal_ids(tmp_path):
	"""Unnamed internal nodes are named internal_<k> in the order their closing parentheses appear; the unnamed
	root gets the next number. from_newick_file reads the first line of a file and gives the same tree."""
	newick = '((A:1,B:2):3,(C:4,(D:5,E:6):7):8);'
	tree = cluster_tree(newick)
	assert write_newick(as_node_tree(tree.root)) == \
		'((A:1.0,B:2.0)internal_0:3.0,(C:4.0,(D:5.0,E:6.0)internal_1:7.0)internal_2:8.0)internal_3;'
	path = tmp_path / 'tree.nwk'
	path.write_text(newick + '\nsecond line is ignored\n')
	from_file = Cluster_Tree()
	from_file.from_newick_file(str(path))
	assert write_newick(as_node_tree(from_file.root)) == write_newick(as_node_tree(tree.root))
	assert sorted(from_file.vert_ind_to_node) == list(range(from_file.nNodes))


def test_node_id_to_vert_ind_is_applied_then_renumbered():
	"""from_newick alone keeps the given vertex indices (no renumbering); from_newick_string renumbers in
	pre-order afterwards."""
	tree = Cluster_Tree()
	tree.from_newick('((A:1,B:2)x:3,C:4)r;', node_id_to_vert_ind={'A': 10, 'x': 11, 'C': 12})
	nodes = {node.nodeId: node for node in [tree.root] + tree.root.childNodes + tree.root.childNodes[0].childNodes}
	assert (nodes['A'].vert_ind, nodes['x'].vert_ind, nodes['C'].vert_ind, nodes['B'].vert_ind) == (10, 11, 12, None)
	assert tree.vert_ind_to_node is None
	tree = Cluster_Tree()
	tree.from_newick_string('((A:1,B:2)x:3,C:4)r;', node_id_to_vert_ind={'A': 10})
	assert [tree.vert_ind_to_node[ind].nodeId for ind in range(5)] == ['r', 'x', 'A', 'B', 'C']


def test_to_newick_round_trip(tmp_path):
	"""to_newick writes '(children)name:length' with the root's name and ';', as Bonsai's Tree.to_newick does, and
	optionally writes the string to a file."""
	tree = cluster_tree(TREE)
	path = tmp_path / 'out.nwk'
	text = tree.to_newick(use_ids=True, results_path=str(path))
	assert text == TREE
	assert path.read_text() == text


def test_copy_is_independent():
	"""copy(minimal_copy=True) copies names, lengths, leaf flags and parent links; changing the copy leaves the
	original alone. A full copy also carries the layout attributes."""
	tree = cluster_tree(TREE)
	twin = tree.copy(minimal_copy=True)
	assert write_newick(as_node_tree(twin.root)) == write_newick(as_node_tree(tree.root))
	for child in twin.root.childNodes:
		assert child.parentNode is twin.root
	twin.root.childNodes[0].tParent = 99.0
	assert tree.root.childNodes[0].tParent == 0.1
	tree.coords = np.arange(2 * tree.nNodes, dtype=float).reshape(-1, 2)
	for node in tree.vert_ind_to_node.values():
		node.coords = tree.coords[node.vert_ind].copy()
		node.dsNodes, node.usNodes = [node.vert_ind], []
		node.angle, node.thetaParent = 10.0 * node.vert_ind, 5.0
	full = tree.copy()
	np.testing.assert_array_equal(full.coords, tree.coords)
	full.coords[0, 0] = -1
	assert tree.coords[0, 0] == 0
	assert full.root.childNodes[1].angle == tree.root.childNodes[1].angle
	np.testing.assert_array_equal(full.root.childNodes[1].coords, tree.root.childNodes[1].coords)


# ------------------------------------------------------------------------------------ files and edge lists

def test_storeTreeInFolder_writes_bonsai_tree_files(tmp_path):
	"""edgeInfo.txt ('parent<TAB>child<TAB>%.8e'), vertInfo.txt ('vertInd<TAB>nodeInd<TAB>vertName' header), and
	tree_test.nwk; the tree read back has the same topology and lengths (to the %.8e rounding); vertices are
	numbered in order of first appearance in the edge list, nodeInd is the Cluster_Tree vertex index, and internal
	nodes without a name (or named internal_*) are renamed internal_<k> in depth-first order."""
	newick = '(((A:0.3,B:0.7):0.4,C:0.2)internal_7:0.1,(D:0.5,E:0.05,F:0.2)k:0.3,G:0.4)r;'
	tree = cluster_tree(newick)
	folder = tmp_path / 'tree'
	tree.storeTreeInFolder(str(folder))
	edges = read_edge_info(str(folder))
	verts = read_vert_info(str(folder))
	back = tree_from_folder(str(folder))
	assert write_newick(back, length_format=lambda x: '%.8e' % x) == write_newick(
		parse_newick('(((A:0.3,B:0.7)internal_1:0.4,C:0.2)internal_0:0.1,(D:0.5,E:0.05,F:0.2)k:0.3,G:0.4)r;'),
		length_format=lambda x: '%.8e' % x)
	order = []
	for parent, child, _ in edges:
		for vert in (parent, child):
			if vert not in order:
				order.append(vert)
	assert order == list(range(len(verts)))
	names_by_vert = {vert: name for vert, _, name in verts}
	node_inds = {name: node_ind for _, node_ind, name in verts}
	for name, node_ind in node_inds.items():
		assert tree.vert_ind_to_node[node_ind].nodeId == name
	assert names_by_vert[0] == 'r'
	assert parse_newick((folder / 'tree_test.nwk').read_text()) is not None


def test_getEdgeVertInfo_and_compile_tree():
	"""Edges in depth-first order as [parent, child] with child lengths; a root without an ID becomes 'root';
	starryYN flags a root with more than three children."""
	tree = cluster_tree('((A:1,B:2)x:3,C:4,D:5,E:6)r;')
	edge_list, dist_list, names, starry, node_ind_to_node = tree.compile_tree_from_scData_tree()
	assert edge_list == [[0, 1], [1, 2], [1, 3], [0, 4], [0, 5], [0, 6]]
	assert dist_list == [3.0, 1.0, 2.0, 4.0, 5.0, 6.0]
	assert names == {0: 'r', 1: 'x', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E'}
	assert starry is True and sorted(node_ind_to_node) == list(range(7))
	tree = cluster_tree('((A:1,B:2):3,C:4);')
	tree.root.nodeId = None
	edges, dists, vert_info = tree.getEdgeVertInfo()
	assert vert_info[0] == (0, 'root')
	assert vert_info[1] == (1, 'internal_0')
	assert [vert_info[child][1] for _, child in edges] == ['internal_0', 'A', 'B', 'C']
	out = getEdgeDistVertNamesFromNode(tree.root, [], [], {}, 5, {})
	assert out[3] == 6


# ---------------------------------------------------------------------------------------- re-rooting

@pytest.mark.parametrize('new_root', ['i', 'k', 'B', 'G'])
def test_reset_root_keeps_the_unrooted_tree(new_root):
	"""Re-rooting at any vertex (a leaf included) keeps the distance between every pair of vertices and, at an
	internal vertex, every split; the new root has no parent and no length."""
	tree = cluster_tree(TREE)
	before = as_node_tree(tree.root)
	ind = next(ind for ind, node in tree.vert_ind_to_node.items() if node.nodeId == new_root)
	tree.reset_root(ind)
	after = as_node_tree(tree.root)
	assert tree.root.nodeId == new_root and tree.root.parentNode is None and tree.root.tParent is None
	assert tree.root.isRoot
	names = [node.name for node in before.iter_nodes()]
	assert leaf_distances(after, names) == pytest.approx(leaf_distances(before, names), rel=1e-12)
	if new_root not in ('B', 'G'):
		assert rf_distance(before, after) == 0


def _pendantless_diameter(node_tree):
	"""Longest path between two vertices when every edge to a leaf counts as zero."""
	zeroed = node_tree.copy()
	for node in zeroed.iter_nodes():
		if node.is_leaf:
			node.length = 0.0
	names = [node.name for node in zeroed.iter_nodes()]
	return max(path_length(zeroed, a, b) for a, b in itertools.combinations(names, 2)), zeroed


def test_set_midpoint_root_halves_the_longest_internal_path():
	"""set_midpoint_root roots the tree on a new vertex 'midpoint_root' on the longest path between vertices with
	edges to leaves counted as zero, at equal distance (half that path) from both of its ends; the unrooted tree is
	otherwise unchanged."""
	newick = '((((A:0.1,B:0.2)a:1.0,C:0.3)b:0.5,D:0.2)c:0.4,((E:0.1,F:0.1)d:2.5,G:0.3)e:0.2)r;'
	tree = cluster_tree(newick)
	before = as_node_tree(tree.root)
	diameter, zeroed = _pendantless_diameter(before)
	tree.set_midpoint_root()
	after = as_node_tree(tree.root)
	assert tree.root.nodeId == 'midpoint_root'
	assert leaf_distances(after) == pytest.approx(leaf_distances(before), rel=1e-12)
	# Hand check: the longest internal path is a - b - c - r - e - d: 1.0 + 0.5 + 0.4 + 0.2 + 2.5 = 4.6, so the
	# midpoint is 2.3 from both a and d, on the edge e - d, 0.2 below e.
	assert diameter == pytest.approx(4.6, rel=1e-12)
	for leaf in after.iter_nodes():
		if leaf.is_leaf:
			leaf.length = 0.0
	assert path_length(after, 'midpoint_root', 'a') == pytest.approx(2.3, rel=1e-12)
	assert path_length(after, 'midpoint_root', 'd') == pytest.approx(2.3, rel=1e-12)
	assert sorted(child.nodeId for child in tree.root.childNodes) == ['d', 'e']


# --------------------------------------------------------------------------------- traversals and sums

def test_traversal_orders():
	"""Post-orders (children before parents, in written order), the internal-only post-order, downstream leaves
	in depth-first order, and the upstream complement."""
	tree = cluster_tree(BINARY)
	name = {ind: node.nodeId for ind, node in tree.vert_ind_to_node.items()}
	post = ['A', 'B', 'i', 'C', 'j', 'D', 'E', 'l', 'F', 'k', 'r']
	assert [name[ind] for ind in tree.root.getPostOrder()] == post
	assert [name[ind] for ind in tree.root.getPostOrder_v2()] == post
	assert [name[ind] for ind in tree.root.getPostOrder_only_internalNodes()] == ['i', 'j', 'l', 'k', 'r']
	polytomy = cluster_tree(TREE)
	pname = {ind: node.nodeId for ind, node in polytomy.vert_ind_to_node.items()}
	assert [pname[ind] for ind in polytomy.root.getPostOrder_v2()] == ['A', 'B', 'i', 'C', 'j', 'D', 'E', 'F', 'k',
	                                                                   'G', 'r']
	j = polytomy.root.childNodes[0]
	n_leafs, ds = j.getDsLeafs_DForder()
	assert n_leafs == 3 and [pname[ind] for ind in ds] == ['j', 'i', 'A', 'B', 'C']
	n_leafs, ds, us = j.getDsLeafs_and_UsLeafs_DForder(polytomy.nNodes)
	assert n_leafs == 3 and [pname[ind] for ind in ds] == ['j', 'i', 'A', 'B', 'C']
	assert sorted(pname[ind] for ind in us) == ['D', 'E', 'F', 'G', 'k', 'r']
	assert polytomy.root.get_ds_and_parent_info() == 7
	assert {node.nodeId: node.ds_leafs for node in polytomy.vert_ind_to_node.values()}['k'] == 3


def test_min_pdists_info_sums_distances_to_every_vertex():
	"""get_min_pdists_info: for every vertex, ds_dists is the summed distance to the leaves below it and us_dists
	the summed distance to all other leaves (brute force over leaf pairs); n_leafs and n_cell_nodes count leaves,
	or the given cell counts per vertex."""
	tree = cluster_tree(TREE)
	node_tree = parse_newick(TREE)
	assert tree.get_min_pdists_info() == 7 and tree.n_cell_nodes == 7
	for node in tree.vert_ind_to_node.values():
		below = set(parse_newick(TREE).find(node.nodeId).leaf_names())
		want_ds = sum(path_length(node_tree, node.nodeId, leaf) for leaf in below if leaf != node.nodeId)
		want_us = sum(path_length(node_tree, node.nodeId, leaf) for leaf in node_tree.leaf_names() if leaf not in below)
		assert node.ds_dists == pytest.approx(want_ds, rel=1e-12, abs=1e-15)
		assert node.us_dists == pytest.approx(want_us, rel=1e-12, abs=1e-15)
	weighted = cluster_tree(TREE)
	counts = {'A': 3, 'B': 0, 'j': 2, 'G': 5}
	weighted.root.add_info_to_nodes(node_id_to_info=counts, info_key='n_cells')
	assert weighted.get_min_pdists_info() == 3 + 2 + 1 + 3 + 5
	assert weighted.n_cell_nodes == 1 + 1 + 1 + 3 + 1
	j = weighted.root.childNodes[0]
	assert j.ds_leafs == 3 + 2 + 1 and j.ds_dists == pytest.approx(3 * 0.7 + 0.2, rel=1e-12)


def test_longest_paths():
	"""find_longest_path_between_two_leafs gives the longest and the shortest root-to-leaf path (that is what
	Bonsai-scout uses it for); get_longest_path_ds the longest downward path and the longest path through each
	vertex, counting edges to leaves as zero."""
	tree = cluster_tree(TREE)
	node_tree = parse_newick(TREE)
	leaf_depths = [node_tree.depths()[id(leaf)] for leaf in node_tree.leaves()]
	longest, shortest = tree.root.find_longest_path_between_two_leafs()
	assert longest == pytest.approx(max(leaf_depths), rel=1e-12)
	assert shortest == pytest.approx(min(leaf_depths), rel=1e-12)
	tree.root.get_longest_path_ds()
	by_id = {node.nodeId: node for node in tree.vert_ind_to_node.values()}
	assert by_id['j'].max_dist_ds == pytest.approx(0.4, rel=1e-12)
	assert by_id['r'].max_dist_ds == pytest.approx(0.5, rel=1e-12)
	assert by_id['r'].max_summed_dist_ds == pytest.approx(0.5 + 0.3, rel=1e-12)
	assert by_id['k'].max_dist_ds == 0 and by_id['A'].max_summed_dist_ds == 0


def test_cut_and_cluster_ids():
	"""cut returns the leaves (or the given cell IDs) below a vertex that were not cut off before, in depth-first
	order, and marks the subtree cut; assign_cluster_id_to_all_downstream_nodes labels unlabeled vertices below a
	vertex."""
	tree = cluster_tree(TREE)
	by_id = {node.nodeId: node for node in tree.vert_ind_to_node.values()}
	assert tree.cut(by_id['i']) == ['A', 'B']
	assert tree.cut(by_id['j']) == ['C']
	assert tree.cut(by_id['r'], cell_ids=['j', 'D', 'G']) == ['D', 'G']
	assert all(node.is_deleted for node in tree.vert_ind_to_node.values())
	tree = cluster_tree(TREE)
	by_id = {node.nodeId: node for node in tree.vert_ind_to_node.values()}
	tree.assign_cluster_id_to_all_downstream_nodes(by_id['k'], 2)
	tree.assign_cluster_id_to_all_downstream_nodes(by_id['r'], 1)
	assert {name: node.cluster_idx for name, node in by_id.items()} == {
		'r': 1, 'j': 1, 'i': 1, 'A': 1, 'B': 1, 'C': 1, 'k': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 1}


def test_print_node_info(capsys):
	"""print_all_nodes_info prints one line per vertex (index, ID, edge length, parent index, leaf flag,
	cluster index); print_all_leafs_info one line per leaf."""
	tree = cluster_tree('((A:1.5,B:2)x:3,C:4)r;')
	tree.print_all_nodes_info()
	lines = capsys.readouterr().out.splitlines()
	assert len(lines) == 5
	assert lines[0] == 'vert_idx: 0 - nodeId: r - edge_length: None - parent node idx : None - isLeaf False ' \
	                   '-cluster_idx: 0'
	assert lines[2] == 'vert_idx: 2 - nodeId: A - edge_length: 1.5 - parent node idx : 1 - isLeaf True -cluster_idx: 0'
	tree.print_all_leafs_info()
	lines = capsys.readouterr().out.splitlines()
	assert [line.split(' ')[1] for line in lines] == ['2', '3', '4']
	assert 'vert_ind: 2\nnodeId: A' in repr(tree.vert_ind_to_node[2])


def test_vertex_maps_and_coordinates():
	"""get_vert_ind_to_node_DF maps vertex indices to nodes (rebuilt on update=True); set_coords_at_node gives each
	node its row of the coordinate array; renumber_verts numbers in pre-order from a start count."""
	tree = Cluster_Tree()
	tree.from_newick('((A:1,B:2)x:3,C:4)r;')
	mapping, count = tree.root.renumber_verts(vertIndToNode={}, vert_count=5)
	assert count == 10 and [mapping[ind].nodeId for ind in range(5, 10)] == ['r', 'x', 'A', 'B', 'C']
	tree.root.storeParent()
	assert tree.get_vert_ind_to_node_DF() is not None
	assert sorted(tree.get_vert_ind_to_node_DF(update=True)) == list(range(5, 10))
	tree = cluster_tree('((A:1,B:2)x:3,C:4)r;')
	tree.coords = np.arange(10, dtype=float).reshape(5, 2)
	tree.set_coords_at_node()
	for ind, node in tree.vert_ind_to_node.items():
		np.testing.assert_array_equal(node.coords, tree.coords[ind])
	tree.root.add_info_to_nodes(info_key='flag', const_val=7)
	assert all(node.flag == 7 for node in tree.vert_ind_to_node.values())
	tree.root.add_info_to_nodes(node_id_to_info={'A': 'a'}, info_key='label')
	assert {node.nodeId: node.label for node in tree.vert_ind_to_node.values()} == {
		'r': None, 'x': None, 'A': 'a', 'B': None, 'C': None}
	tree.root.add_info_to_nodes(info_key='nothing')
	assert not hasattr(tree.root, 'nothing')


# ------------------------------------------------------------------------------- entropy bookkeeping

def nlogn(values):
	return sum(v * math.log(v) for v in np.atleast_1d(values) if v > 0)


def test_entropy_non_norm():
	"""-sum n_r log n_r without normalization; zero counts contribute nothing; a scalar with num_annots=1."""
	assert entropy_non_norm(np.array([0.0, 2.0, 3.0])) == pytest.approx(-(2 * math.log(2) + 3 * math.log(3)))
	assert entropy_non_norm(np.array([0.2, 0.8]), num_annots=2) == pytest.approx(
		-(0.2 * math.log(0.2) + 0.8 * math.log(0.8)), rel=1e-15)
	assert entropy_non_norm(5, num_annots=1) == pytest.approx(-5 * math.log(5), rel=1e-15)
	assert entropy_non_norm(0, num_annots=1) == 0.0
	assert entropy(np.array([0.5, 0.5])) is None  # the function has only a docstring


def test_annotation_counts_and_entropy_changes():
	"""On a tree with labelled leaves: get_ds_annot_info sums the label counts below every vertex;
	get_ent_contribs stores n log n of the cells below and above (clade) and the same per label (annotation);
	get_ent_changes gives, for every edge, how much N times the clade entropy and N times the joint entropy grow when
	the edge is cut; subtract_or_add_ds_info_us removes a cut subtree's counts from every ancestor (and adds them
	back); set_include_as_cand drops edges that would split off fewer annotated cells than the bound."""
	tree = cluster_tree(TREE)
	labels = {'A': 0, 'B': 1, 'C': 0, 'D': 1, 'E': 1, 'F': 2, 'G': 0}
	for node in tree.vert_ind_to_node.values():
		node.own_annot_counts = np.zeros(3, dtype=int)
		node.own_n_annots = 0
		if node.nodeId in labels:
			node.own_annot_counts[labels[node.nodeId]] = 1
			node.own_n_annots = 1
	tree.root.get_ds_annot_info()
	by_id = {node.nodeId: node for node in tree.vert_ind_to_node.values()}
	assert list(by_id['j'].ds_annot_counts) == [2, 1, 0] and by_id['j'].ds_n_annots == 3
	assert list(by_id['r'].ds_annot_counts) == [3, 3, 1]
	total = by_id['r'].ds_annot_counts.copy()
	tree.root.get_ent_contribs(total_n=7, total_annot_counts=total, num_annots=3)
	tree.get_ent_changes()
	j = by_id['j']
	assert j.clade_ent_contrib_ds == pytest.approx(nlogn(3)) and j.clade_ent_contrib_us == pytest.approx(nlogn(4))
	assert j.annot_ent_contrib_us == pytest.approx(nlogn([1, 2, 1]))
	assert j.clade_ent_change == pytest.approx(nlogn(7) - nlogn(3) - nlogn(4), rel=1e-12)
	assert j.annot_ent_change == pytest.approx(nlogn([3, 3, 1]) - nlogn([2, 1, 0]) - nlogn([1, 2, 1]), rel=1e-12)
	# The NMI identity: N * H(clade) after the cut equals the change computed above (clade entropy was 0).
	assert j.clade_ent_change == pytest.approx(-7 * sum(p * math.log(p) for p in (3 / 7, 4 / 7)), rel=1e-12)
	by_id['j'].subtract_or_add_ds_info_us(by_id['i'], add=False)
	assert list(by_id['j'].ds_annot_counts) == [1, 0, 0] and list(by_id['r'].ds_annot_counts) == [2, 2, 1]
	assert by_id['j'].ds_n_annots == 1 and by_id['r'].ds_n_annots == 5 and list(by_id['i'].ds_annot_counts) == [1, 1, 0]
	tree.root.get_ds_annot_info()
	by_id['r'].subtract_or_add_ds_info_us(by_id['k'], add=True)
	assert list(by_id['r'].ds_annot_counts) == [3, 5, 2]
	tree.root.get_ds_annot_info()
	total = tree.root.ds_annot_counts.copy()
	tree.root.set_include_as_cand(total_n_annots=7, total_annot_counts=total, clst_size_lb=2,
	                              clst_cat_capture_lbs=total * 0.025, clst_cat_capture_ubs=total * 10)
	assert {name: node.include_as_cand for name, node in by_id.items() if name in labels} == {name: False for name in labels}
	assert by_id['i'].include_as_cand and by_id['k'].include_as_cand and by_id['j'].include_as_cand
