"""
bonsai_scout/my_tree_layout.py (area D): the tree layouts Bonsai-scout draws (equal angle, equal daylight, the
ladderized dendrogram). Runs in the main environment (numpy and pandas only).

The references are written from the layout definitions, not from the code:
- equal angle (Felsenstein, "Inferring Phylogenies", ch. 34): every vertex owns a wedge of directions, the root
  all 360 degrees starting at 180 degrees (the direction to the root's notional parent); a vertex splits its wedge
  among its children in order, in proportion to their numbers of leaves, and each child sits in the middle of its
  share at its branch length from the vertex;
- ladderized: children ordered by increasing number of leaves (the order my_tree_layout imposes, see the comment
  "Ladderize all branches");
- dendrogram: x is the root-to-vertex path length scaled to [-0.95, 0.95], leaves evenly spaced in y from -0.95 to
  0.95 in ladderized depth-first order, an internal vertex at the mean y of its children;
- equal daylight: from every internal vertex, the vertices of each subtree hanging off it form one contiguous block
  of directions (subtrees do not overlap).
Zero-length edges to leaves are avoided where equal daylight runs: with a leaf on top of its parent the equal-angle
start already overlaps, and equal daylight then raises UnboundLocalError.
"""
import itertools
import math
import os
import types

import numpy as np
import pandas as pd
import pytest

from bonsai_scout import my_tree_layout as mtl
from downstream_analyses.get_cluster_helpers import Cluster_Tree
from bonsai_testtrees import Node, parse_newick, path_length, write_newick

TREES = [
	'(((C2:0.678034,C1:0.894135)n1:1.059513,C4:0.214824)n3:0.898486,((C5:0.799273,C3:0.281621)n0:1.436588,'
	'C0:0.393375)n2:0.482682)root;',
	'((A:0.3,(B:0.2,C:0.4,D:0.25)x:0.7)y:0.5,(E:0.6,F:0.15)z:0.9,G:0.45)root;',
	'(((((A:0.21,B:0.17)i0:0.33,C:0.24)i1:0.41,D:0.19)i2:0.29,E:0.27)i3:0.37,(F:0.22,G:0.26)i4:0.31)root;',
]


def scdata_for(newick):
	"""A stand-in for the scData object: my_tree_layout reads .tree (root, nNodes, vert_ind_to_node) and .nVerts."""
	tree = Cluster_Tree()
	tree.from_newick_string(newick)
	return types.SimpleNamespace(tree=tree, nVerts=tree.nNodes)


def vert_of(scdata):
	return {node.nodeId: ind for ind, node in scdata.tree.vert_ind_to_node.items()}


def ladderize(node):
	node.children = sorted(node.children, key=lambda child: len(child.leaves()))
	for child in node.children:
		ladderize(child)
	return node


def reference_equal_angle(newick, weights=None):
	"""Coordinates by vertex name; weights scale a leaf's share of the wedges (1 by default)."""
	tree = ladderize(parse_newick(newick))
	weight = {}

	def count(node):
		weight[id(node)] = (weights or {}).get(node.name, 1.0) if node.is_leaf else sum(count(c) for c in node.children)
		return weight[id(node)]

	count(tree)
	coords = {tree.name: np.zeros(2)}

	def place(node, low, high):
		start = low
		for child in node.children:
			share = (high - low) * weight[id(child)] / weight[id(node)]
			direction = math.radians(start + share / 2)
			coords[child.name] = coords[node.name] + child.length * np.array([math.cos(direction), math.sin(direction)])
			place(child, start, start + share)
			start += share

	place(tree, 180.0, 540.0)
	return coords


def reference_dendrogram(newick, flips=(), ladderized=True):
	tree = ladderize(parse_newick(newick)) if ladderized else parse_newick(newick)
	for name in flips:
		node = tree.find(name)
		node.children = node.children[::-1]
	depth = tree.depths()
	leaves = tree.leaves()
	y = {leaf.name: value for leaf, value in zip(leaves, np.linspace(-0.95, 0.95, len(leaves)))}

	def mean_y(node):
		if node.is_leaf:
			return y[node.name]
		y[node.name] = np.mean([mean_y(child) for child in node.children])
		return y[node.name]

	mean_y(tree)
	max_depth = max(depth.values())
	return {node.name: np.array([depth[id(node)] / max_depth * 1.9 - 0.95, y[node.name]]) for node in tree.iter_nodes()}


def as_named(coords, scdata):
	return {name: coords[ind] for name, ind in vert_of(scdata).items()}


@pytest.mark.parametrize('newick', TREES)
def test_my_tree_layout_returns_and_stores_three_layouts(newick, tmp_path):
	"""my_tree_layout(calc=True, return_all=True): equal-angle, equal-daylight and ladderized-dendrogram
	coordinates per vertex index, the dendrogram picked as the layout to show, and the same numbers in a CSV with
	columns <layout>_x, <layout>_y and the vertex IDs (in vertex order) as index."""
	scdata = scdata_for(newick)
	path = str(tmp_path / 'layout.csv')
	coords, picked = mtl.my_tree_layout(scdata, calc=True, filepath=path, eq_daylight=True, return_all=True)
	assert picked == 'ly_dendrogram_ladderized'
	assert list(coords) == ['ly_eq_angle', 'ly_eq_daylight', 'ly_dendrogram_ladderized']
	got = as_named(coords['ly_eq_angle'], scdata)
	for name, want in reference_equal_angle(newick).items():
		np.testing.assert_allclose(got[name], want, rtol=1e-9, atol=1e-12, err_msg=name)
	got = as_named(coords['ly_dendrogram_ladderized'], scdata)
	for name, want in reference_dendrogram(newick).items():
		np.testing.assert_allclose(got[name], want, rtol=1e-12, atol=1e-12, err_msg=name)
	table = pd.read_csv(path, index_col=0)
	assert list(table.columns) == ['ly_eq_angle_x', 'ly_eq_angle_y', 'ly_eq_daylight_x', 'ly_eq_daylight_y',
	                               'ly_dendrogram_ladderized_x', 'ly_dendrogram_ladderized_y']
	assert list(table.index) == [scdata.tree.vert_ind_to_node[ind].nodeId for ind in range(scdata.nVerts)]
	for ly_type in coords:
		np.testing.assert_allclose(table[[ly_type + '_x', ly_type + '_y']].values, coords[ly_type], rtol=1e-12)
	without = mtl.my_tree_layout(scdata_for(newick), calc=True, filepath=path, eq_daylight=False, return_all=True)
	assert list(without[0]) == ['ly_eq_angle', 'ly_dendrogram_ladderized']


def _edge_lengths_kept(coords, scdata):
	for node in scdata.tree.vert_ind_to_node.values():
		for child in node.childNodes:
			assert np.linalg.norm(coords[child.vert_ind] - coords[node.vert_ind]) == pytest.approx(child.tParent,
			                                                                                      rel=1e-9)


def _subtrees_are_contiguous(coords, scdata):
	"""From every internal vertex, sort all other vertices by direction: each subtree hanging off the vertex (one
	per child, plus the rest of the tree) must occupy one contiguous circular block."""
	tree = parse_newick(write_newick(_as_node(scdata.tree.root)))
	vert = vert_of(scdata)
	parents = tree.parents()
	for node in tree.iter_nodes():
		if node.is_leaf:
			continue
		groups = {}
		for child_ind, child in enumerate(node.children):
			for member in child.iter_nodes():
				groups[member.name] = child_ind
		others = [x.name for x in tree.iter_nodes() if x is not node and x.name not in groups]
		for name in others:
			groups[name] = -1
		origin = coords[vert[node.name]]
		angles = {name: math.atan2(*(coords[vert[name]] - origin)[::-1]) for name in groups}
		order = [groups[name] for name in sorted(groups, key=angles.get)]
		changes = sum(1 for a, b in zip(order, order[1:] + order[:1]) if a != b)
		assert changes == len(set(order)) or len(set(order)) == 1, node.name


def _as_node(cluster_node):
	return Node(cluster_node.nodeId, cluster_node.tParent, [_as_node(child) for child in cluster_node.childNodes])


@pytest.mark.parametrize('newick', TREES)
def test_layouts_keep_branch_lengths_and_daylight_keeps_subtrees_apart(newick, tmp_path):
	"""Both radial layouts draw every edge at its branch length; equal daylight leaves every subtree in its own
	block of directions as seen from each internal vertex."""
	scdata = scdata_for(newick)
	coords, _ = mtl.my_tree_layout(scdata, calc=True, filepath=str(tmp_path / 'ly.csv'), eq_daylight=True,
	                               return_all=True)
	_edge_lengths_kept(coords['ly_eq_angle'], scdata)
	_edge_lengths_kept(coords['ly_eq_daylight'], scdata)
	_subtrees_are_contiguous(coords['ly_eq_angle'], scdata)
	_subtrees_are_contiguous(coords['ly_eq_daylight'], scdata)


def test_getLayout_with_flipped_branches():
	"""getLayout copies the scData tree into a ladderized Layout_Tree; flipped_node_ids reverses the order of the
	children of the listed vertex in the dendrogram (the next of its orderings, for two children)."""
	newick = TREES[0]
	scdata = scdata_for(newick)
	ly_tree, coords = mtl.getLayout(scdata.tree, eq_daylight=False, dendrogram=True, verbose=False,
	                                all_coords_dict={'start': None}, flipped_node_ids=['n2'])
	got = as_named(coords['ly_dendrogram_ladderized'], scdata)
	for name, want in reference_dendrogram(newick, flips=['n2']).items():
		np.testing.assert_allclose(got[name], want, rtol=1e-12, atol=1e-12, err_msg=name)
	assert ly_tree.nNodes == scdata.nVerts


def test_dendrogram_in_vertex_order():
	"""get_dendrogram(ladderized=False) spaces the leaves in the tree's own depth-first order, within the given
	x and y limits."""
	newick = TREES[1]
	tree = layout_tree(newick)
	tree.get_dendrogram(ladderized=False)
	vert = {node.nodeId: ind for ind, node in tree.get_vert_ind_to_node_DF().items()}
	for name, want in reference_dendrogram(newick, ladderized=False).items():
		np.testing.assert_allclose(tree.coords[vert[name]], want, rtol=1e-12, atol=1e-12, err_msg=name)
	tree.get_dendrogram(ladderized=False, xlims=(0.0, 1.0), ylims=(-1.0, 1.0))
	assert tree.coords[:, 0].min() == pytest.approx(0.0, abs=1e-15) and tree.coords[:, 0].max() == pytest.approx(1.0)
	assert sorted(tree.coords[[vert[n] for n in 'ABCDEFG'], 1])[0] == -1.0


def test_get_ly_tree_from_scdata_tree_ladderizes_a_copy():
	"""The Layout_Tree copy keeps vertex indices, IDs and lengths, orders children by increasing number of leaves,
	and leaves the source tree as it was."""
	newick = TREES[1]
	scdata = scdata_for(newick)
	before = write_newick(_as_node(scdata.tree.root))
	ly_tree = mtl.get_ly_tree_from_scdata_tree(scdata.tree)
	assert write_newick(_as_node(scdata.tree.root)) == before
	got = write_newick(_layout_as_node(ly_tree.root))
	assert got == write_newick(ladderize(parse_newick(newick)))
	for node in ly_tree.get_vert_ind_to_node_DF().values():
		assert node.vert_ind == vert_of(scdata)[node.nodeId]


def _layout_as_node(node):
	return Node(node.nodeId, node.tParent, [_layout_as_node(child) for child in node.childNodes])


def layout_tree(newick):
	tree = mtl.Layout_Tree()
	tree.from_newick(newick)
	tree.root.renumber_verts(vertIndToNode={}, vert_count=0)
	tree.root.storeParent()
	return tree


def test_from_newick_and_edge_dict():
	"""Layout_Tree.from_newick parses names, lengths and child order (vertex indices only from
	node_id_to_vert_ind); get_edge_dict lists source, target and length depth-first, with vertex indices when a
	mapping is given."""
	tree = mtl.Layout_Tree()
	tree.from_newick('((A:1.5,B:2)x:3,C:4)r;', node_id_to_vert_ind={'A': 7})
	assert write_newick(_layout_as_node(tree.root)) == '((A:1.5,B:2.0)x:3.0,C:4.0)r;'
	assert tree.nNodes == 5 and tree.root.childNodes[0].childNodes[0].vert_ind == 7
	assert tree.root.isRoot and not tree.root.isLeaf and tree.root.childNodes[1].isLeaf
	assert tree.get_edge_dict() == {'source': ['r', 'x', 'x', 'r'], 'target': ['x', 'A', 'B', 'C'],
	                                'dist': [3.0, 1.5, 2.0, 4.0]}
	mapping = {'r': 0, 'x': 1, 'A': 2, 'B': 3, 'C': 4}
	edges = tree.get_edge_dict(nodeIdToVertInd=mapping)
	assert edges['source_ind'] == [0, 1, 1, 0] and edges['target_ind'] == [1, 2, 3, 4]


def test_reconstructTreeFromEdgeVertInfo(tmp_path):
	"""Reads a Bonsai tree folder (edgeInfo.txt, vertInfo.txt): the Layout_Tree has the folder's topology and
	lengths, vertex indices from vertInfo's nodeInd column, and the mappings and lists are returned too."""
	(tmp_path / 'edgeInfo.txt').write_text('0\t1\t3.0e+00\n1\t2\t1.5e+00\n1\t3\t2.0e+00\n0\t4\t4.0e+00\n')
	(tmp_path / 'vertInfo.txt').write_text('vertInd\tnodeInd\tvertName\n0\t10\tr\n1\t11\tx\n2\t0\tA\n3\t1\tB\n'
	                                       '4\t2\tC\n')
	vert_to_node, vert_to_node_ind, edges, dists, tree = mtl.reconstructTreeFromEdgeVertInfo(str(tmp_path))
	assert write_newick(_layout_as_node(tree.root)) == '((A:1.5,B:2.0)x:3.0,C:4.0)r;'
	assert edges == [(0, 1), (1, 2), (1, 3), (0, 4)] and dists == [3.0, 1.5, 2.0, 4.0]
	assert vert_to_node_ind == {0: 10, 1: 11, 2: 0, 3: 1, 4: 2}
	assert {vert: node.vert_ind for vert, node in vert_to_node.items()} == {0: 10, 1: 11, 2: 0, 3: 1, 4: 2}
	assert tree.nNodes == 5


def test_equal_angle_angles_round_trip_and_theta_parent():
	"""get_angles_from_coords recovers each child's angle (relative to the direction to its parent) from the
	coordinates equalAngle produced; set_thetaParent recomputes the direction to the parent from those angles."""
	tree = layout_tree(TREES[1])
	tree.equalAngle()
	nodes = tree.get_vert_ind_to_node_DF()
	angles = {ind: node.angle for ind, node in nodes.items() if not node.isRoot}
	thetas = {ind: node.thetaParent for ind, node in nodes.items()}
	tree.set_coords_at_node()
	for node in nodes.values():
		node.angle, node.thetaParent = None, None
	tree.get_angles_from_coords()
	for ind, angle in angles.items():
		assert (nodes[ind].angle - angle + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)
	tree.root.thetaParent = 180
	tree.root.set_thetaParent()
	for ind, theta in thetas.items():
		assert (nodes[ind].thetaParent - theta + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)


def test_increase_equal_angle():
	"""increaseEqualAngle multiplies the weight of the leaves in the chosen subtree (the child of the ancestor
	vert_inds[-2] that contains vert_inds[-1]) and redraws equal angle with weighted wedges, returning the origin
	that keeps the ancestor at the same spot on screen. (resetEqualAngle puts the leaf weights back to 1 but keeps
	the internal vertices' weighted counts, so it does not restore the layout.)"""
	newick = '((A:0.3,B:0.7)x:0.4,(C:0.5,D:0.2,E:0.6)y:0.3)root;'
	tree = layout_tree(newick)
	tree.equalAngle()
	tree.set_coords_at_node()
	vert = {node.nodeId: ind for ind, node in tree.get_vert_ind_to_node_DF().items()}
	origin = np.array([0.25, -0.5])
	ancestor_before = tree.root.coords.copy()
	new_origin = tree.increaseEqualAngle([vert['root'], vert['A']], 3.0, origin)
	want = reference_equal_angle(newick, weights={'A': 3.0, 'B': 3.0})
	for name, ind in vert.items():
		np.testing.assert_allclose(tree.coords[ind], want[name], atol=1e-12, err_msg=name)
	np.testing.assert_allclose(new_origin, tree.root.coords - (ancestor_before - origin), atol=1e-15)
	tree.resetEqualAngle()
	assert all(node.dsLeafs_weighted == 1 for node in tree.get_vert_ind_to_node_DF().values() if node.isLeaf)


def _gaps_at(tree, node):
	"""Angular gaps between the blocks of directions of the subtrees around `node`."""
	coords = tree.coords
	blocks = [list(child.dsNodes) for child in node.childNodes] + ([list(node.usNodes)] if not node.isRoot else [])
	origin = coords[node.vert_ind]
	spans = []
	for block in blocks:
		degrees = np.sort(np.degrees(np.arctan2(*(coords[block] - origin).T[::-1])) % 360)
		gaps = np.diff(np.append(degrees, degrees[0] + 360))
		widest = int(np.argmax(gaps))
		start = degrees[(widest + 1) % len(degrees)]
		spans.append((start, (degrees[widest] - start) % 360))
	spans.sort()
	return [((spans[(k + 1) % len(spans)][0] - (spans[k][0] + spans[k][1])) % 360) for k in range(len(spans))]


def test_equal_daylight_at_one_vertex():
	"""equalDaylightSome with stepsize 1 rotates the subtrees around the given vertex until the daylight (the empty
	angle between neighbouring subtrees) is the same on every side."""
	tree = layout_tree(TREES[1])
	tree.equalAngle()
	x = next(node for node in tree.get_vert_ind_to_node_DF().values() if node.nodeId == 'y')
	before = _gaps_at(tree, x)
	tree.equalDaylightSome(vert_inds=[x.vert_ind], stepsize=1)
	after = _gaps_at(tree, x)
	assert max(before) - min(before) > 1.0
	assert after == pytest.approx([np.mean(after)] * len(after), abs=1e-9)


def test_reset_root_and_vertex_lists():
	"""reset_root keeps the distance between every pair of vertices; breadth-first and depth-first vertex lists;
	is_vert_downstream; get_dsNodes."""
	newick = TREES[1]
	tree = layout_tree(newick)
	before = parse_newick(newick)
	names = [node.name for node in before.iter_nodes()]
	vert = {node.nodeId: ind for ind, node in tree.get_vert_ind_to_node_DF().items()}
	assert list(tree.getNodeListBF()) == ['root', 'y', 'z', 'G', 'A', 'x', 'E', 'F', 'B', 'C', 'D']
	assert [tree.getNodeListBFVertInd()[ind].nodeId for ind in tree.getNodeListBFVertInd()] == list(
		tree.getNodeListBF())
	y = tree.root.childNodes[0]
	assert y.is_vert_downstream(vert['C']) and not y.is_vert_downstream(vert['E'])
	assert sorted(y.get_dsNodes()) == sorted(vert[name] for name in ('y', 'A', 'x', 'B', 'C', 'D'))
	tree.reset_root(vert['x'])
	after = _layout_as_node(tree.root)
	assert tree.root.nodeId == 'x' and tree.root.parentNode is None and tree.root.tParent is None
	for a, b in itertools.combinations(names, 2):
		assert path_length(after, a, b) == pytest.approx(path_length(before, a, b), rel=1e-12)


def test_copy_and_ladderized_listing():
	"""Layout_Tree.copy duplicates coordinates and per-vertex layout data independently; the ladderized vertex
	listing visits children by increasing number of leaves, and a flipped vertex in the other order."""
	tree = layout_tree(TREES[1])
	tree.equalAngle()
	tree.set_coords_at_node()
	twin = tree.copy()
	np.testing.assert_array_equal(twin.coords, tree.coords)
	twin.coords[1, 0] += 1.0
	assert twin.coords[1, 0] != tree.coords[1, 0]
	assert twin.root.childNodes[0].dsLeafs == tree.root.childNodes[0].dsLeafs
	listing = tree.root.get_vert_ind_to_node_ladderized({})
	names = [node.nodeId for node in listing.values()]
	assert names == [node.name for node in ladderize(parse_newick(TREES[1])).iter_nodes()]
	flipped = [node.nodeId for node in tree.root.get_vert_ind_to_node_ladderized({}, flipped_node_ids=['z']).values()]
	assert flipped.index('F') < flipped.index('E')
	tree.root.rearrange_branches_node(flipped_node_ids=['x', 'x'], nNodes=tree.nNodes)
	x = next(node for node in tree.get_vert_ind_to_node_DF().values() if node.nodeId == 'x')
	# three children, flipped twice: the third ordering of the ladderized order
	ladder = sorted(range(3), key=lambda k: 0)
	assert [child.nodeId for child in x.childNodes] == [['B', 'C', 'D'][k] for k in list(
		itertools.permutations(ladder))[2]]


def test_legacy_daylight_angles_fill_the_circle():
	"""setequalDaylightAngles gives every subtree its angular extent plus an equal share of the remaining daylight;
	at the root (no parent side) the shares add up to 360 degrees."""
	tree = layout_tree(TREES[2])
	tree.equalAngle()
	tree.root.setequalDaylightAngles(tree.coords)
	assert sum(child.angle for child in tree.root.childNodes) == pytest.approx(360.0, rel=1e-12)
