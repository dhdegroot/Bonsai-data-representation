"""
bonsai_scout/bonsai_scout_preprocess.py (area D): the script that turns Bonsai's results into
bonsai_vis_data.hdf and bonsai_vis_settings.json for Bonsai-scout. Needs the Bonsai-scout environment
(requirements-dev_bonsai_scout.txt); skipped elsewhere.

Expected values come from Bonsai's own final_bonsai_* files read with the harness readers, from the
dense-Gaussian reference posteriors (tests/bonsai_testref.py) on Bonsai's final tree, from the generator's
truth and from the annotation files the tests write. The runs are the `scout_rich` job in helpers_d.py:
Bonsai once, then the script on copies of the results folder with different options.
"""
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.scout
h5py = pytest.importorskip('h5py')
pytest.importorskip('tables')
pytest.importorskip('natsort')

import pandas as pd  # noqa: E402

from bonsai_testfixtures import DATASET_NAME  # noqa: E402
from bonsai_testref import reference_posteriors  # noqa: E402
from bonsai_testrun import posteriors_by_name, read_metadata_json, tree_from_folder  # noqa: E402
from bonsai_testtrees import parse_newick  # noqa: E402
from area_d.helpers_d import (NEW_GENE_IDS, clade_of, contracted_edges, contracted_vertices, extra_depth,  # noqa: E402,F401
                              extra_features, is_cell_name, scout_preprocessed, scout_rich)


def test_preprocess_writes_the_visualization_data(scout_preprocessed):
	run, sim, proc = scout_preprocessed
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	assert os.path.exists(run.path('bonsai_vis_data.hdf'))
	with open(run.path('bonsai_vis_settings.json')) as handle:
		assert isinstance(json.load(handle), dict)
	final = run.final_folder()
	metadata = read_metadata_json(final)
	posterior_means, _ = posteriors_by_name(final)
	with h5py.File(run.path('bonsai_vis_data.hdf'), 'r') as hdf:
		normalized = hdf['data/normalized']
		gene_ids = json.loads(normalized.attrs['gene_ids'])
		node_ids = json.loads(normalized.attrs['node_ids'])
		means = normalized['means'][()]
	# The genes Bonsai kept, and for every cell the posterior Bonsai wrote in final_bonsai_*.
	assert gene_ids == metadata['geneIds']
	assert means.shape == (len(gene_ids), len(node_ids))
	assert set(sim.cell_ids) <= set(node_ids)
	for cell in sim.cell_ids:
		np.testing.assert_allclose(means[:, node_ids.index(cell)], posterior_means[cell], rtol=1e-12)


# ------------------------------------------------------------------ helpers

def _metadata(path):
	with h5py.File(path, 'r') as hdf:
		return json.loads(hdf.attrs['metadata_json']), json.loads(hdf.attrs['feature_paths'])


def _expected_vertices(tree):
	"""
	The vertices Bonsai-scout should keep after merging zero-length edges, as {expected node ID: (rep node,
	cell names)}. A vertex that holds cells is named after them, joined by '_' in pre-order; one without
	cells keeps the name of the node that represents it.
	"""
	_, members = contracted_vertices(tree)
	expected = {}
	for node in tree.iter_nodes():
		if id(node) not in members:
			continue
		cells = [member.name for member in members[id(node)] if is_cell_name(member.name)]
		name = '_'.join(cells) if cells else node.name
		expected[name] = (node, cells)
	return expected


def _ready(scout_rich, variant):
	scout_rich.check(variant)
	return scout_rich


def _cell_to_vertex(metadata, cell_info):
	node_ids = metadata['nodeIds']
	return {cell: node_ids[vert] for cell, vert in zip(cell_info.index, cell_info['cell_ind_to_vert_ind'])}


# --------------------------------------------------------- default options

def test_cells_at_zero_distance_share_one_vertex(scout_rich):
	"""Every vertex of the stored tree is a node of Bonsai's final tree with its zero-length edges contracted;
	the cells on those edges are stored at that vertex, and the vertex is named after them."""
	run = _ready(scout_rich, 'default')
	tree = tree_from_folder(run.final_folder)
	expected = _expected_vertices(tree)
	assert any(len(cells) > 1 for _, cells in expected.values()), 'the dataset must have duplicated cells'
	metadata, _ = _metadata(run.hdf_path('default'))
	node_ids = metadata['nodeIds']
	assert sorted(node_ids) == sorted(expected)
	cell_info = pd.read_hdf(run.hdf_path('default'), key='cell_info/cell_info_dict')
	assert list(cell_info.index) == run.sim.cell_ids
	cell_to_vertex = _cell_to_vertex(metadata, cell_info)
	for name, (_, cells) in expected.items():
		assert sorted(cell for cell, vert in cell_to_vertex.items() if vert == name) == sorted(cells)
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		n_cells_per_vert = hdf['vert_info/n_cells_per_vert'][()]
		vert_to_cells = json.loads(hdf['vert_info'].attrs['vert_ind_to_cell_inds_json'])
		single = set(hdf['cell_info/single_at_vert'][()].tolist())
		multi = set(hdf['cell_info/multi_at_vert'][()].tolist())
		np.testing.assert_array_equal(hdf['tree_info/n_cells_per_vert'][()], n_cells_per_vert)
		cell_inds = hdf['tree_info/cell_inds'][()].tolist()
		multi_cell_inds = hdf['tree_info/multi_cell_inds'][()].tolist()
		int_inds = hdf['tree_info/int_inds'][()].tolist()
		n_leafs = int(hdf['tree_info'].attrs['n_leafs'])
	counts = [len(expected[name][1]) for name in node_ids]
	np.testing.assert_array_equal(n_cells_per_vert, counts)
	for vert, name in enumerate(node_ids):
		assert sorted(run.sim.cell_ids[ind] for ind in vert_to_cells[str(vert)]) == sorted(expected[name][1])
	multi_cells = {ind for ind, cell in enumerate(run.sim.cell_ids) if counts[node_ids.index(cell_to_vertex[cell])] > 1}
	assert multi == multi_cells and single == set(range(run.sim.n_cells)) - multi_cells
	assert cell_inds == [vert for vert in range(len(node_ids)) if counts[vert] > 0]
	assert multi_cell_inds == [vert for vert in range(len(node_ids)) if counts[vert] > 1]
	# Internal vertices are the ones with children in the contracted tree.
	rep_to_name = {id(node): name for name, (node, _) in expected.items()}
	parents = {rep_to_name[id(parent)] for parent, _, _ in contracted_edges(tree)}
	assert sorted(node_ids[vert] for vert in int_inds) == sorted(parents)
	assert n_leafs == len(node_ids) - len(int_inds)


def test_edges_and_newick_are_the_contracted_tree(scout_rich):
	"""edge_df and the stored newick string hold the contracted tree with Bonsai's branch lengths (as
	edgeInfo.txt stores them); vertex indices follow the order of nodeIds."""
	run = _ready(scout_rich, 'default')
	tree = tree_from_folder(run.final_folder)
	expected = _expected_vertices(tree)
	rep_to_name = {id(node): name for name, (node, _) in expected.items()}
	rep, _ = contracted_vertices(tree)
	expected_edges = sorted((rep_to_name[id(par)], rep_to_name[id(rep[id(child)])], length)
	                        for par, child, length in contracted_edges(tree))
	metadata, _ = _metadata(run.hdf_path('default'))
	node_ids = metadata['nodeIds']
	edge_df = pd.read_hdf(run.hdf_path('default'), key='tree_info/edge_df')
	assert list(edge_df.columns) == ['source', 'source_ind', 'target', 'target_ind', 'dist']
	assert [node_ids[ind] for ind in edge_df['source_ind']] == list(edge_df['source'])
	assert [node_ids[ind] for ind in edge_df['target_ind']] == list(edge_df['target'])
	got = sorted(zip(edge_df['source'], edge_df['target'], edge_df['dist']))
	assert [(s, t) for s, t, _ in got] == [(s, t) for s, t, _ in expected_edges]
	np.testing.assert_allclose([d for _, _, d in got], [d for _, _, d in expected_edges], rtol=1e-12)
	assert node_ids[edge_df['source_ind'][0]] == tree.name
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		nwk = parse_newick(hdf['tree_info'].attrs['nwk_str'])
	nwk_edges = sorted((parent.name, child.name, child.length) for parent in nwk.iter_nodes()
	                   for child in parent.children)
	assert [(s, t) for s, t, _ in nwk_edges] == [(s, t) for s, t, _ in expected_edges]
	np.testing.assert_allclose([d for _, _, d in nwk_edges], [d for _, _, d in expected_edges], rtol=1e-12)


def test_posteriors_are_the_reference_posteriors_on_bonsais_tree(scout_rich):
	"""data/normalized holds, per vertex, the posterior mean and variance of every kept gene in the data's own
	units: the dense-Gaussian posterior on Bonsai's final tree in Bonsai's rescaled units, times sqrt(v_g) and
	v_g. zscores, variances and the no-variation list follow from those."""
	run = _ready(scout_rich, 'default')
	sim = run.sim
	tree = run.final_tree()
	expected = _expected_vertices(tree)
	final_metadata = read_metadata_json(run.final_folder)
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		normalized = hdf['data/normalized']
		gene_ids = json.loads(normalized.attrs['gene_ids'])
		node_ids = json.loads(normalized.attrs['node_ids'])
		means, variances = normalized['means'][()], normalized['vars'][()]
		zscores, gene_vars = normalized['zscores'][()], normalized['variances'][()]
		no_variation = normalized['no_variation_features'][()]
		cells_wo_nan, ranks = normalized['cells_wo_nan'][()], normalized['ranks_per_gene'][()]
	assert gene_ids == final_metadata['geneIds']
	assert set(gene_ids) < set(sim.gene_ids)
	genes = np.array([sim.gene_ids.index(gene) for gene in gene_ids])
	scale = np.sqrt(sim.gene_variances[genes])
	ref_means, ref_vars = reference_posteriors(tree, *sim.bonsai_units(genes))
	assert means.shape == variances.shape == (len(genes), len(node_ids))
	for vert, name in enumerate(node_ids):
		node = expected[name][0]
		np.testing.assert_allclose(means[:, vert], ref_means[node.name] * scale, rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(variances[:, vert], ref_vars[node.name] * scale ** 2, rtol=1e-9)
	np.testing.assert_allclose(gene_vars, sim.gene_variances[genes], rtol=1e-12)
	centred = means - means.mean(axis=1, keepdims=True)
	np.testing.assert_allclose(zscores, np.sqrt(np.mean(centred ** 2 / variances, axis=1)), rtol=1e-12)
	assert no_variation.tolist() == [g for g in range(len(genes)) if means[g].min() == means[g].max()]
	# Ranks over cells: every gene ranks all cells 0..n-1, consistently with the cells' values.
	metadata, _ = _metadata(run.hdf_path('default'))
	cell_info = pd.read_hdf(run.hdf_path('default'), key='cell_info/cell_info_dict')
	cell_values = means[:, np.array(cell_info['cell_ind_to_vert_ind'])]
	assert cells_wo_nan.tolist() == list(range(sim.n_cells))
	assert ranks.shape == (len(genes), sim.n_cells)
	for g in range(len(genes)):
		assert sorted(ranks[g]) == list(range(sim.n_cells))
		order = np.argsort(ranks[g])
		assert np.all(np.diff(cell_values[g, order]) >= 0)


def test_layout_coordinates_respect_branch_lengths(scout_rich):
	"""The three stored layouts put every vertex at its tree position: in the equal-angle and equal-daylight
	layouts each edge is drawn with its branch length, from the root at the origin; the ladderized dendrogram
	spans x in [-0.95, 0.95] in proportion to depth, spreads the leaves evenly over y in [-0.95, 0.95] and puts
	each internal vertex at the mean height of its children. Edge coordinates are drawn from the vertices."""
	run = _ready(scout_rich, 'default')
	edge_df = pd.read_hdf(run.hdf_path('default'), key='tree_info/edge_df')
	src, tgt, dist = (np.array(edge_df[col]) for col in ('source_ind', 'target_ind', 'dist'))
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		node_coords = {key: hdf['layout_coords/node_coords'][key][()] for key in hdf['layout_coords/node_coords']}
		edge_coords = {key: hdf['layout_coords/edge_coords'][key][()] for key in hdf['layout_coords/edge_coords']}
	assert sorted(node_coords) == sorted(edge_coords) == ['ly_dendrogram_ladderized', 'ly_eq_angle',
	                                                       'ly_eq_daylight']
	n_verts = len(_metadata(run.hdf_path('default'))[0]['nodeIds'])
	root = src[0]
	for ly_type in ('ly_eq_angle', 'ly_eq_daylight'):
		coords = node_coords[ly_type]
		assert coords.shape == (n_verts, 2)
		np.testing.assert_allclose(np.linalg.norm(coords[tgt] - coords[src], axis=1), dist, rtol=1e-9)
		np.testing.assert_allclose(coords[root], [0.0, 0.0], atol=1e-15)
		np.testing.assert_allclose(edge_coords[ly_type], np.linspace(coords[src], coords[tgt], 40, axis=1),
		                           rtol=1e-12, atol=1e-15)
	coords = node_coords['ly_dendrogram_ladderized']
	depth = np.zeros(n_verts)
	for s, t, d in zip(src, tgt, dist):
		depth[t] = depth[s] + d
	np.testing.assert_allclose(coords[:, 0], -0.95 + 1.9 * depth / depth.max(), rtol=1e-12, atol=1e-12)
	leaves = sorted(set(range(n_verts)) - set(src.tolist()))
	np.testing.assert_allclose(np.sort(coords[leaves, 1]), np.linspace(-0.95, 0.95, len(leaves)), atol=1e-12)
	for parent in set(src.tolist()):
		np.testing.assert_allclose(coords[parent, 1], coords[tgt[src == parent], 1].mean(), atol=1e-12)
	dendro_edges = edge_coords['ly_dendrogram_ladderized']
	assert dendro_edges.shape == (len(src), 3, 2)
	np.testing.assert_allclose(dendro_edges[:, 0], coords[src], atol=1e-15)
	np.testing.assert_allclose(dendro_edges[:, 2], coords[tgt], atol=1e-15)
	np.testing.assert_allclose(dendro_edges[:, 1], np.stack([coords[src, 0], coords[tgt, 1]], axis=1), atol=1e-15)


def test_annotation_files_become_cellstate_annotations_and_data_matrices(scout_rich):
	"""Without a cell-to-cellstate file every Bonsai cell is its own cellstate, so annotation.csv goes to the
	cellstate table: categorical columns as 'annot_<name>' (missing values as 'NaN'), numerical columns as
	'annot_num_<name>' plus 'annot_categorized_<name>' when they have fewer than 20 values. A 'mat_' file is a
	features x cells data matrix."""
	run = _ready(scout_rich, 'default')
	sim = run.sim
	cs_info = pd.read_hdf(run.hdf_path('default'), key='cs_info/cs_info_dict')
	cell_info = pd.read_hdf(run.hdf_path('default'), key='cell_info/cell_info_dict')
	assert list(cell_info.columns) == ['annot_default', 'cell_ind_to_vert_ind']
	assert set(cell_info['annot_default']) == {'default'}
	assert list(cs_info.columns) == ['annot_celltype', 'annot_partial', 'annot_num_depth', 'annot_categorized_depth',
	                                 'cs_ind_to_vert_ind', 'n_cells_per_cs']
	assert list(cs_info.index) == sim.cell_ids
	assert [str(value) for value in cs_info['annot_celltype']] == [clade_of(cell) for cell in sim.cell_ids]
	assert [str(value) for value in cs_info['annot_partial']] == [
		(('a' if ind % 2 else 'b') if ind < 6 else 'NaN') for ind in range(sim.n_cells)]
	np.testing.assert_allclose(cs_info['annot_num_depth'], [extra_depth(ind) for ind in range(sim.n_cells)])
	assert list(cs_info['annot_categorized_depth']) == ['c_%r' % extra_depth(ind) for ind in range(sim.n_cells)]
	assert list(cs_info['n_cells_per_cs']) == [1] * sim.n_cells
	assert list(cs_info['cs_ind_to_vert_ind']) == list(cell_info['cell_ind_to_vert_ind'])

	metadata, feature_paths = _metadata(run.hdf_path('default'))
	assert feature_paths == ['data/normalized', 'data/extra']
	values = np.array([extra_features(ind) for ind in range(sim.n_cells)]).T
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		extra = hdf['data/extra']
		assert json.loads(extra.attrs['gene_ids']) == ['f1', 'f2', 'f3']
		assert json.loads(extra.attrs['node_ids']) == sim.cell_ids
		assert extra.attrs['cell_or_cs'] == 'cell'
		np.testing.assert_allclose(extra['means'][()], values, rtol=1e-15)
		np.testing.assert_allclose(extra['variances'][()], values.var(axis=1), rtol=1e-12)
		assert extra['no_variation_features'][()].tolist() == [2]
		assert extra['cells_wo_nan'][()].tolist() == list(range(sim.n_cells))
		ranks = extra['ranks_per_gene'][()]
	for f in (0, 1):
		np.testing.assert_array_equal(ranks[f], np.argsort(np.argsort(values[f])))
	assert metadata['cellIds'] == metadata['csIds'] == sim.cell_ids
	assert metadata['nCells'] == metadata['nCss'] == sim.n_cells
	assert metadata['dataset'] == DATASET_NAME
	assert metadata['cell_ind_to_cs_ind'] == {str(ind): ind for ind in range(sim.n_cells)}
	assert metadata['cs_ind_to_cell_inds'] == {str(ind): [ind] for ind in range(sim.n_cells)}


def _vertex_cells(run, variant):
	"""{vertex index: sorted cellstate IDs} and the weight (number of cells) of each vertex."""
	with h5py.File(run.hdf_path(variant), 'r') as hdf:
		vert_to_cs = json.loads(hdf['vert_info'].attrs['vert_ind_to_cs_inds_json'])
		n_cells = hdf['vert_info/n_cells_per_vert'][()]
	metadata, _ = _metadata(run.hdf_path(variant))
	return {int(v): sorted(metadata['csIds'][ind] for ind in inds) for v, inds in vert_to_cs.items()}, n_cells


def _clusters(column):
	"""{cluster label (the 'cl_<i>' part): set of row IDs}."""
	out = {}
	for row, name in column.items():
		out.setdefault(name.split(',')[0], set()).add(row)
	return out


def _check_min_distance_clusterings(run, variant):
	cluster_info = pd.read_hdf(run.hdf_path(variant), key='cs_info/cluster_info_dict')
	metadata, _ = _metadata(run.hdf_path(variant))
	vert_cs, n_cells = _vertex_cells(run, variant)
	n_cell_verts = int(np.sum(n_cells > 0))
	names = ['annot_bnsi_cluster_n%d' % k for k in range(2, n_cell_verts + 1)]
	assert [col for col in cluster_info.columns if col.startswith('annot_bnsi_cluster_n')] == names
	assert list(cluster_info.index) == metadata['csIds']
	previous = None
	for k, col in zip(range(2, n_cell_verts + 1), names):
		clusters = _clusters(cluster_info[col])
		assert sorted(clusters) == sorted('cl_%d' % ind for ind in range(k))
		for row, name in cluster_info[col].items():
			label = name.split(',')[0]
			assert name == '%s, Size=%d' % (label, len(clusters[label]))
		for cs_ids in vert_cs.values():
			assert len({cluster_info[col][cs] for cs in cs_ids}) <= 1
		if previous is not None:
			# One cluster of the previous clustering splits; the new part is cl_<k-1>.
			new = clusters['cl_%d' % (k - 1)]
			assert len({previous[cs] for cs in new}) == 1
			for label, members in clusters.items():
				if label != 'cl_%d' % (k - 1):
					assert {previous[cs] for cs in members} == {label}
		previous = {cs: name.split(',')[0] for cs, name in cluster_info[col].items()}
	return cluster_info


def test_min_distance_clusterings_split_one_cluster_at_a_time(scout_rich):
	"""cs_info/cluster_info_dict holds the minimal-distance clusterings for 2 up to the number of vertices with
	cells: each one splits one cluster of the previous, cells at one vertex stay together, and every cluster is
	named 'cl_<i>, Size=<number of cellstates>'. The first cut is the edge with the largest summed distance
	over cell pairs it separates plus the cell pairs times its length, every vertex weighted by its cells."""
	run = _ready(scout_rich, 'default')
	cluster_info = _check_min_distance_clusterings(run, 'default')
	from bonsai_testtrees import path_length
	tree = tree_from_folder(run.final_folder)
	expected = _expected_vertices(tree)
	weights = {name: len(cells) for name, (_, cells) in expected.items() if cells}
	total = sum(weights.values())
	scores = []
	rep, _ = contracted_vertices(tree)
	rep_to_name = {id(node): name for name, (node, _) in expected.items()}
	for parent, child, length in contracted_edges(tree):
		below = {rep_to_name[id(rep[id(node)])] for node in child.iter_nodes()} & set(weights)
		above = set(weights) - below
		w_below = sum(weights[name] for name in below)
		cross = sum(weights[a] * weights[b] * path_length(tree, expected[a][0].name, expected[b][0].name)
		            for a in below for b in above)
		cells_below = frozenset(cell for name in below for cell in expected[name][1])
		partition = frozenset([cells_below, frozenset(run.sim.cell_ids) - cells_below])
		scores.append((cross + w_below * (total - w_below) * length, partition))
	# The two edges at Bonsai's root give the same partition; compare partitions, not edges.
	best = {}
	for score, partition in scores:
		best[partition] = max(score, best.get(partition, -np.inf))
	ranked = sorted(best.items(), key=lambda item: -item[1])
	assert ranked[0][1] > ranked[1][1] * (1 + 1e-9), 'the best first cut must be unique for this test'
	got = frozenset(frozenset(members) for members in _clusters(cluster_info['annot_bnsi_cluster_n2']).values())
	assert got == ranked[0][0]


def _check_annotation_name(name, members, annotation):
	parts = name.split(', ')
	assert parts[1] == 'Size=%d' % len(members)
	entries = parts[2:]
	assert 1 <= len(entries) <= 2
	percs = []
	labels = [str(annotation.get(cell, 'nan')) for cell in members]
	for entry in entries:
		perc, label = entry.split('%=')
		assert int(perc) == round(100 * labels.count(label) / len(members))
		percs.append(int(perc))
	assert percs == sorted(percs, reverse=True)
	assert percs[0] == max(round(100 * labels.count(label) / len(members)) for label in set(labels))


def test_annotation_guided_clustering_recovers_the_clade_annotation(scout_rich):
	"""The annotation-guided clustering of 'celltype' (the two clades of the true tree) cuts the tree into
	exactly the two clades; it is stored as 'annot_bnsi_cluster_celltype' with names giving the size and the
	top annotations with their percentages. Annotations with one label ('default') or a different label for
	every cell ('categorized_depth') give no cut and are not stored."""
	run = _ready(scout_rich, 'default')
	cluster_info = pd.read_hdf(run.hdf_path('default'), key='cs_info/cluster_info_dict')
	assert [col for col in cluster_info.columns if not col.startswith('annot_bnsi_cluster_n')] == [
		'annot_bnsi_cluster_celltype', 'annot_bnsi_cluster_partial']
	clusters = _clusters(cluster_info['annot_bnsi_cluster_celltype'])
	sim = run.sim
	assert sorted(map(sorted, clusters.values())) == sorted(
		[sorted(c for c in sim.cell_ids if clade_of(c) == 'left'), sorted(c for c in sim.cell_ids if clade_of(c) == 'right')])
	for cell, name in cluster_info['annot_bnsi_cluster_celltype'].items():
		assert name == '%s, Size=%d, 100%%=%s' % (name.split(',')[0], len(clusters[name.split(',')[0]]),
		                                          clade_of(cell))
	partial = {cell: ('a' if ind % 2 else 'b') for ind, cell in enumerate(sim.cell_ids) if ind < 6}
	partial_clusters = _clusters(cluster_info['annot_bnsi_cluster_partial'])
	assert set().union(*partial_clusters.values()) == set(sim.cell_ids)
	for cell, name in cluster_info['annot_bnsi_cluster_partial'].items():
		_check_annotation_name(name, partial_clusters[name.split(',')[0]], partial)


def test_settings_file_holds_the_initial_view(scout_rich):
	"""bonsai_vis_settings.json: the ladderized dendrogram (drawn flat, untransformed), coloured by the
	'celltype' annotation (preferred by name) and sized by the number of cells per vertex. Categorical colours
	come from Bonsai-scout's eight-colour palette in natural-sort order, 'NaN' is gray; the numerical
	annotation gets the YlOrRd colormap between its 1% and 99% quantiles."""
	from matplotlib import colors as mcolors
	from matplotlib import pyplot as plt
	run = _ready(scout_rich, 'default')
	with open(run.settings_path('default')) as handle:
		settings = json.load(handle)
	assert settings['ly_type'] == 'ly_dendrogram_ladderized'
	transf = settings['transf_info']
	assert (transf['geometry'], transf['no_transform'], transf['origin_style']) == ('flat', True, 'root')
	assert settings['node_style']['annot_info']['info_key'] == 'annot_celltype'
	assert settings['node_style']['annot_info']['label'] == 'Celltype'
	assert settings['node_style']['size_annot_info']['info_key'] == 'cell_number'
	assert settings['node_style']['feature_path'] == 'data/normalized'
	palette = ['#AFD2E9', '#AF601A', '#FCD0A1', '#0B5345', '#48C9B0', '#FE776D', '#229954', '#A690A4']
	infos = settings['celltype_info']['annot_infos']
	celltype = infos['annot_celltype']
	assert celltype['cats'] == ['left', 'right'] and celltype['color_type'] == 'categorical'
	assert celltype['info_object'] == 'cs_info_dict' and celltype['annot_type'] == 'cellstates'
	for cat, hex_color in zip(['left', 'right'], palette):
		np.testing.assert_allclose(celltype['annot_to_color'][cat], mcolors.to_rgba(hex_color))
	partial = infos['annot_partial']
	assert partial['cats'] == ['NaN', 'a', 'b']
	np.testing.assert_allclose(partial['annot_to_color']['NaN'], plt.get_cmap('gray')(0.75))
	np.testing.assert_allclose(partial['annot_to_color']['a'], mcolors.to_rgba(palette[0]))
	np.testing.assert_allclose(partial['annot_to_color']['b'], mcolors.to_rgba(palette[1]))
	depth = infos['annot_num_depth']
	values = np.array([extra_depth(ind) for ind in range(run.sim.n_cells)])
	assert depth['color_type'] == 'sequential' and depth['label'] == 'Depth' and depth['cats'] is None
	assert depth['cbar_info']['cmap'] == 'YlOrRd'
	np.testing.assert_allclose([depth['cbar_info']['vmin'], depth['cbar_info']['vmax']],
	                           np.quantile(values, [0.01, 0.99]), rtol=1e-12)
	n2 = infos['annot_bnsi_cluster_n2']
	assert n2['hidden'] is True and n2['info_object'] == 'cluster_info_dict'
	annot_clst = infos['annot_bnsi_cluster_celltype']
	assert annot_clst['cats'] == sorted(annot_clst['cats'], key=lambda cat: -int(cat.split('Size=')[1].split(',')[0]))
	assert annot_clst['cat_counts_dict'] == {cat: int(cat.split('Size=')[1].split(',')[0]) for cat in annot_clst['cats']}
	vert = settings['verttype_info']
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		n_cells_per_vert = hdf['vert_info/n_cells_per_vert'][()]
	assert list(vert['new_cell_info_dict']['cell_number']) == n_cells_per_vert.tolist()
	assert list(vert['new_cell_info_dict']['binarized_cell_number']) == np.minimum(n_cells_per_vert, 1).tolist()
	assert vert['annot_infos']['cell_number']['cbar_info']['vmax'] == max(int(n_cells_per_vert.max()), 2)


# ------------------------------------------------------- other options

def test_take_all_genes_gives_every_gene_in_data_units(scout_rich):
	"""--take_all_genes True reads every gene back (also the ones under the zscore cutoff) and undoes Bonsai's
	rescaling by the gene variance; --config_filepath points at a YAML file outside the results folder;
	--annotation_path may name a single file; --perform_annot_guided_clustering False stores only the
	minimal-distance clusterings."""
	run = _ready(scout_rich, 'all_genes')
	sim = run.sim
	tree = tree_from_folder(run.final_folder)
	expected = _expected_vertices(tree)
	metadata, feature_paths = _metadata(run.hdf_path('all_genes'))
	assert feature_paths == ['data/normalized']
	assert metadata['geneIds'] == sim.gene_ids
	with h5py.File(run.hdf_path('all_genes'), 'r') as hdf:
		normalized = hdf['data/normalized']
		assert json.loads(normalized.attrs['gene_ids']) == sim.gene_ids
		node_ids = json.loads(normalized.attrs['node_ids'])
		means, variances = normalized['means'][()], normalized['vars'][()]
	genes = np.arange(sim.n_genes)
	scale = np.sqrt(sim.gene_variances)
	# The tree is read back from edgeInfo.txt, so the reference uses the tree with those (rounded) lengths.
	ref_means, ref_vars = reference_posteriors(tree, *sim.bonsai_units(genes))
	for vert, name in enumerate(node_ids):
		node = expected[name][0]
		np.testing.assert_allclose(means[:, vert], ref_means[node.name] * scale, rtol=1e-9, atol=1e-12)
		np.testing.assert_allclose(variances[:, vert], ref_vars[node.name] * scale ** 2, rtol=1e-9)
	cluster_info = pd.read_hdf(run.hdf_path('all_genes'), key='cs_info/cluster_info_dict')
	assert all(col.startswith('annot_bnsi_cluster_n') for col in cluster_info.columns)
	cs_info = pd.read_hdf(run.hdf_path('all_genes'), key='cs_info/cs_info_dict')
	assert 'annot_celltype' in cs_info.columns
	assert not os.path.exists(os.path.join(run.folder('all_genes'), 'used_run_configs.yaml'))


def test_cellstate_mapping_and_no_genes(scout_rich):
	"""--cell_id_to_cs_id_file maps the real cells onto Bonsai's cells (cellstates): the stored cell IDs are the
	real cells, each vertex counts its real cells, and a per-cell annotation file goes to the cell table.
	--take_no_genes True stores one dummy feature. --new_gene_ids_file renames the genes in Bonsai's
	metadata.json (a repeated new name gets '_promoter2') after keeping a time-stamped copy."""
	run = _ready(scout_rich, 'cellstates')
	sim = run.sim
	rows = run.mapping
	metadata, feature_paths = _metadata(run.hdf_path('cellstates'))
	assert metadata['cellIds'] == [cell for cell, _ in rows]
	assert metadata['csIds'] == sim.cell_ids
	assert metadata['nCells'] == len(rows) and metadata['nCss'] == sim.n_cells
	assert metadata['cell_ind_to_cs_ind'] == {str(ind): sim.cell_ids.index(cs) for ind, (_, cs) in enumerate(rows)}
	assert metadata['cs_ind_to_cell_inds'] == {
		str(cs_ind): [ind for ind, (_, cs) in enumerate(rows) if cs == cs_id] for cs_ind, cs_id in enumerate(sim.cell_ids)}
	renamed = dict(NEW_GENE_IDS)
	final_metadata = read_metadata_json(run.final_folder)
	new_names = [renamed.get(gene, gene) for gene in final_metadata['geneIds']]
	seen = {}
	for ind, name in enumerate(new_names):
		seen[name] = seen.get(name, 0) + 1
		if seen[name] > 1:
			new_names[ind] = '%s_promoter%d' % (name, seen[name])
	assert 'NEWA_promoter2' in new_names
	assert metadata['geneIds'] == new_names
	tree_folder = os.path.join(run.folder('cellstates'), os.path.basename(run.final_folder))
	with open(os.path.join(tree_folder, 'metadata.json')) as handle:
		assert json.load(handle)['geneIds'] == new_names
	backups = [name for name in os.listdir(tree_folder) if name.startswith('metadata_') and name.endswith('.json')]
	assert len(backups) == 1
	with open(os.path.join(tree_folder, backups[0])) as handle:
		assert json.load(handle)['geneIds'] == final_metadata['geneIds']

	assert feature_paths == ['data/dummy_gene']
	with h5py.File(run.hdf_path('cellstates'), 'r') as hdf:
		assert list(hdf['data']) == ['dummy_gene']
		dummy = hdf['data/dummy_gene']
		n_verts = len(metadata['nodeIds'])
		np.testing.assert_allclose(dummy['means'][()], np.linspace(0.9, 1.1, n_verts)[None, :], rtol=1e-15)
		assert json.loads(dummy.attrs['gene_ids']) == ['dummy_gene']
		np.testing.assert_array_equal(dummy['variances'][()], [1.0])
		n_cells_per_vert = hdf['vert_info/n_cells_per_vert'][()]
		n_css_per_vert = hdf['vert_info/n_css_per_vert'][()]
		single = hdf['cell_info/single_at_vert'][()].tolist()
		multi_cs = hdf['cs_info/multi_cs_at_vert'][()].tolist()
		multi_cell_inds = hdf['tree_info/multi_cell_inds'][()].tolist()
		multi_cs_inds = hdf['tree_info/multi_cs_inds'][()].tolist()

	tree = tree_from_folder(run.final_folder)
	expected = _expected_vertices(tree)
	per_cs = {cs: sum(1 for _, c in rows if c == cs) for cs in sim.cell_ids}
	cs_per_vert = [expected[name][1] for name in metadata['nodeIds']]
	np.testing.assert_array_equal(n_css_per_vert, [len(cs) for cs in cs_per_vert])
	np.testing.assert_array_equal(n_cells_per_vert, [sum(per_cs[cs] for cs in css) for css in cs_per_vert])
	assert multi_cell_inds == [v for v, n in enumerate(n_cells_per_vert) if n > 1]
	assert multi_cs_inds == [v for v, n in enumerate(n_css_per_vert) if n > 1]
	assert sorted(single) == [ind for ind, (_, cs) in enumerate(rows)
	                          if n_cells_per_vert[metadata['nodeIds'].index(
		                          [name for name, (_, css) in expected.items() if cs in css][0])] == 1]
	assert sorted(multi_cs) == sorted(sim.cell_ids.index(cs) for css in cs_per_vert if len(css) > 1 for cs in css)

	cell_info = pd.read_hdf(run.hdf_path('cellstates'), key='cell_info/cell_info_dict')
	cs_info = pd.read_hdf(run.hdf_path('cellstates'), key='cs_info/cs_info_dict')
	assert list(cell_info.index) == [cell for cell, _ in rows]
	assert [str(v) for v in cell_info['annot_celltype']] == [clade_of(cs) for _, cs in rows]
	np.testing.assert_allclose(cell_info['annot_num_score'], [0.5 * ind for ind in range(len(rows))])
	vertex_of = {cs: metadata['nodeIds'].index(name) for name, (_, css) in expected.items() for cs in css}
	assert list(cell_info['cell_ind_to_vert_ind']) == [vertex_of[cs] for _, cs in rows]
	assert list(cs_info['n_cells_per_cs']) == [per_cs[cs] for cs in sim.cell_ids]
	assert list(cs_info['cs_ind_to_vert_ind']) == [vertex_of[cs] for cs in sim.cell_ids]
	_check_min_distance_clusterings(run, 'cellstates')
	cluster_info = pd.read_hdf(run.hdf_path('cellstates'), key='cs_info/cluster_info_dict')
	# The annotation-guided clustering of a per-cell annotation counts real cells in its names.
	cell_annot = {cell: clade_of(cs) for cell, cs in rows}
	cs_to_cells = {cs: [cell for cell, c in rows if c == cs] for cs in sim.cell_ids}
	clusters = _clusters(cluster_info['annot_bnsi_cluster_celltype'])
	for cs, name in cluster_info['annot_bnsi_cluster_celltype'].items():
		members = [cell for c in clusters[name.split(',')[0]] for cell in cs_to_cells[c]]
		_check_annotation_name(name, members, cell_annot)


def test_results_folder_without_yaml_is_read_as_a_tree_folder(scout_rich):
	"""With no YAML file in --results_folder, the folder itself is read as Bonsai's tree folder: the output lands
	there and holds the same tree, cells and posteriors as a run on the full results folder."""
	run = _ready(scout_rich, 'tree_only')
	run.check('default')
	meta_tree, paths_tree = _metadata(run.hdf_path('tree_only'))
	meta_default, _ = _metadata(run.hdf_path('default'))
	assert paths_tree == ['data/normalized']
	for key in ('nodeIds', 'cellIds', 'geneIds', 'nCells'):
		assert meta_tree[key] == meta_default[key]
	with h5py.File(run.hdf_path('tree_only'), 'r') as a, h5py.File(run.hdf_path('default'), 'r') as b:
		np.testing.assert_allclose(a['data/normalized/means'][()], b['data/normalized/means'][()], rtol=1e-12)
		np.testing.assert_allclose(a['data/normalized/vars'][()], b['data/normalized/vars'][()], rtol=1e-12)
		assert a['tree_info'].attrs['nwk_str'] == b['tree_info'].attrs['nwk_str']
	assert os.path.exists(run.settings_path('tree_only'))
