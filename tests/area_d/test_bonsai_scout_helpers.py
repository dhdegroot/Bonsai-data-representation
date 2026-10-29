"""
bonsai_scout/bonsai_scout_helpers.py (area D): the figure, settings and metadata objects behind
Bonsai-scout, the Poincare-disk view, colours, clustering names and marker-gene glue. Needs the
Bonsai-scout environment (h5py, tables, natsort); skipped elsewhere.

Expected values are derived here: the Poincare map and its inverse from their formulas, edge coordinates
from vertex coordinates, colours from the named matplotlib colormaps and Bonsai-scout's eight-colour palette,
subsets of cells from the tree stored in the HDF file, marker scores by brute force over cell pairs. The
in-process figures are built from the `scout_rich` (cells merged at zero distance) and `scout_preprocessed`
(no merged cells) runs in helpers_d.py.
"""
import colorsys
import copy
import json
import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.scout
h5py = pytest.importorskip('h5py')
pytest.importorskip('tables')
pytest.importorskip('natsort')

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from bonsai_scout import bonsai_scout_helpers as bsh  # noqa: E402
from area_d.helpers_d import (MARKER_GROUPS, cell_parity, clade_of, extra_depth, extra_features,  # noqa: E402,F401
                              gaussian_marker_score, pairwise_higher_fraction, scout_preprocessed, scout_rich,
                              two_sigma_truncation)

PALETTE = ['#AFD2E9', '#AF601A', '#FCD0A1', '#0B5345', '#48C9B0', '#FE776D', '#229954', '#A690A4']
GRAY_75 = plt.get_cmap('gray')(0.75)
MASKED = tuple(list(plt.get_cmap('gray')(0.95))[:3] + [0.25])


def _poincare(coords, origin, zoom):
	"""z (x - o) / (1 + sqrt(1 + |z (x - o)|^2)), the map of the plane into the unit (Poincare) disk."""
	shifted = zoom * (np.asarray(coords, dtype=float) - origin)
	radius_sq = np.sum(shifted ** 2, axis=-1, keepdims=True)
	return shifted / (1 + np.sqrt(1 + radius_sq))


class _Geometry:
	"""The two attributes of Transf_info the zoom and centroid helpers read."""

	def __init__(self, origin, geometry='hyperbolic'):
		self.origin = np.asarray(origin, dtype=float)
		self.geometry = geometry


# ------------------------------------------------------------ pure helpers

def test_poincare_transform_matches_its_formula_and_inverts():
	rng = np.random.default_rng(0)
	coords = rng.normal(0, 2, (50, 2))
	origin, zoom = np.array([0.3, -0.7]), 1.7
	got = bsh.transform_coords_poincare(coords, origin=origin, zoom=zoom)
	np.testing.assert_allclose(got, _poincare(coords, origin, zoom), rtol=1e-13)
	assert np.all(np.linalg.norm(got, axis=1) < 1)
	radii = bsh.transform_coords_poincare(coords, origin=origin, zoom=zoom, only_radii=True)
	np.testing.assert_allclose(radii, np.linalg.norm(got, axis=1), rtol=1e-13)
	np.testing.assert_allclose(bsh.invert_poincare_transform(got, origin, zoom), coords, rtol=1e-10, atol=1e-12)
	# Edges are arrays of points (edges x points x 2); the map works on the last axis.
	edges = rng.normal(0, 1, (4, 5, 2))
	np.testing.assert_allclose(bsh.transform_coords_poincare(edges, origin=origin, zoom=zoom),
	                           _poincare(edges, origin, zoom), rtol=1e-13)
	# Flat geometry: shift and scale, and draw anything outside [-1, 1] on the frame at +-1.02.
	flat = bsh.transform_coords_poincare(coords, origin=origin, zoom=0.5, transform=False)
	expected = 0.5 * (coords - origin)
	expected[expected > 1] = 1.02
	expected[expected < -1] = -1.02
	np.testing.assert_allclose(flat, expected, rtol=1e-15)
	np.testing.assert_allclose(bsh.transform_coords_poincare(coords, origin=origin, zoom=0.5, transform=False,
	                                                         only_radii=True), np.linalg.norm(expected, axis=1))
	np.testing.assert_allclose(bsh.invert_poincare_transform(0.5 * (coords - origin), origin, 0.5, transform=False),
	                           coords, rtol=1e-13)


def test_zoom_centroid_and_optimal_view():
	"""get_zoom_given_origin puts 80% of the points within radius 0.8 of the disk; get_centroid_poincare
	minimizes the mean disk radius, or returns the Euclidean centroid when the optimizer reports failure;
	get_opt_zoom_origin_poincare returns a zoom and origin that satisfy both."""
	rng = np.random.default_rng(2)
	coords = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(3, 0.5, (10, 2))])
	n_within = math.ceil(0.8 * len(coords))
	geometry = _Geometry([0.5, 0.5])
	zoom = bsh.get_zoom_given_origin(coords, geometry)
	radii = np.sort(np.linalg.norm(_poincare(coords, geometry.origin, zoom), axis=1))
	assert radii[n_within - 1] == pytest.approx(0.8, abs=1e-9)

	def mean_radius(origin, zoom):
		return np.mean(np.linalg.norm(_poincare(coords, origin, zoom), axis=1))

	def check_centroid(origin, zoom, tol):
		euclidean = coords.mean(axis=0)
		assert mean_radius(origin, zoom) <= mean_radius(euclidean, zoom) + 1e-12
		if np.allclose(origin, euclidean, rtol=0, atol=1e-12):
			return
		for step in ([1e-3, 0], [-1e-3, 0], [0, 1e-3], [0, -1e-3]):
			assert mean_radius(origin + step, zoom) >= mean_radius(origin, zoom) - tol

	centroid = bsh.get_centroid_poincare(coords, geometry, zoom=zoom)
	assert not np.allclose(centroid, coords.mean(axis=0))
	check_centroid(centroid, zoom, 1e-9)
	opt_geometry = _Geometry([0.0, 0.0])
	opt_zoom, opt_origin = bsh.get_opt_zoom_origin_poincare(coords, opt_geometry)
	opt_geometry.origin = opt_origin
	assert bsh.get_zoom_given_origin(coords, opt_geometry) == pytest.approx(opt_zoom, rel=1e-4)
	check_centroid(opt_origin, opt_zoom, 1e-7)


def test_transf_info_zoom_origin_and_dict():
	coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [1.0, -1.0], [3.0, 3.0]])
	edges = {'source_ind': [0, 1, 1, 1], 'target_ind': [1, 2, 3, 4]}
	info = bsh.Transf_info(geometry='hyperbolic', zoom=2.0, origin=[1, 2])
	assert info.zoom_per_geometry == {'hyperbolic': 2.0, 'flat': None}
	assert info.ax_lims == [-1.05, 1.05, -1.05, 1.05]
	np.testing.assert_array_equal(info.origin, [1, 2])
	info.set_zoom(multiply_by=3)
	assert info.get_zoom() == 6.0
	info.set_zoom(zoom=0.5)
	assert info.get_zoom() == 0.5
	info.set_geometry('flat')
	assert info.get_zoom() is None
	info.set_zoom(multiply_by=3)
	assert info.get_zoom() is None
	assert info.to_dict()['origin'] == [1, 2]
	# Origin at the root (the source of the first edge) or at the vertex with the most edges, then the zoom.
	root = bsh.Transf_info(geometry='hyperbolic', origin_style='root')
	root.set_geometry('hyperbolic', coords, edges)
	np.testing.assert_array_equal(root.origin, coords[0])
	assert root.get_zoom() == pytest.approx(bsh.get_zoom_given_origin(coords, _Geometry(coords[0])), rel=1e-12)
	hub = bsh.Transf_info(geometry='hyperbolic', origin_style='most_connected')
	hub.set_geometry('hyperbolic', coords, edges)
	np.testing.assert_array_equal(hub.origin, coords[1])
	other = bsh.Transf_info(geometry='hyperbolic', origin_style='centroid')
	other.set_geometry('hyperbolic', coords, edges)
	zoom, origin = bsh.get_opt_zoom_origin_poincare(coords, _Geometry([0, 0]))
	np.testing.assert_allclose(other.origin, origin, rtol=1e-9)
	assert other.get_zoom() == pytest.approx(zoom, rel=1e-9)


def test_edge_coordinates_from_vertex_coordinates():
	coords = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
	edges = {'source_ind': [0, 0], 'target_ind': [1, 2]}
	got = bsh.get_edge_coords(edges, {'ly_eq_angle': coords, 'ly_dendrogram_ladderized': coords})
	assert got['ly_eq_angle'].shape == (2, 40, 2)
	for e, (s, t) in enumerate(zip(edges['source_ind'], edges['target_ind'])):
		for k in range(40):
			np.testing.assert_allclose(got['ly_eq_angle'][e, k], coords[s] + k / 39 * (coords[t] - coords[s]),
			                           atol=1e-15)
		np.testing.assert_array_equal(got['ly_dendrogram_ladderized'][e],
		                              [coords[s], [coords[s, 0], coords[t, 1]], coords[t]])
	# Very large trees get straight two-point edges.
	big = bsh.get_edge_coords(edges, {'ly_eq_daylight': coords}, nVerts=20000)
	np.testing.assert_array_equal(big['ly_eq_daylight'], np.stack([coords[[0, 0]], coords[[1, 2]]], axis=1))


def test_category_colours():
	def rgba(cmap, n):
		return [tuple(cmap(ind)) for ind in range(n)]

	assert rgba(bsh.get_celltype_colors_new(3), 8) == [mcolors.to_rgba(c) for c in PALETTE]
	assert rgba(bsh.get_celltype_colors_new(9), 10) == rgba(plt.get_cmap('tab10'), 10)
	assert rgba(bsh.get_celltype_colors_new(3, colortype='tab10'), 10) == rgba(plt.get_cmap('tab10'), 10)
	assert rgba(bsh.get_celltype_colors_new(15), 20) == rgba(plt.get_cmap('tab20'), 20)
	assert rgba(bsh.get_celltype_colors_new(3, colortype='tab20'), 20) == rgba(plt.get_cmap('tab20'), 20)
	forty = np.vstack([plt.cm.tab20b(np.linspace(0, 1, 20)), plt.cm.tab20c(np.linspace(0, 1, 20))])
	np.testing.assert_allclose(np.array(rgba(bsh.get_celltype_colors_new(25), 40)), forty)
	_, many = bsh.categorical_cmap(10, 5, cmap='tab10')
	np.testing.assert_allclose(np.array(rgba(bsh.get_celltype_colors_new(45), 50))[:, :3], many)
	off_on = bsh.get_celltype_colors_new(2, colortype='offOn')
	assert tuple(off_on(0)) == tuple(plt.get_cmap('tab10')(1)) and tuple(off_on(1)) == GRAY_75
	assert bsh.get_celltype_colors_new(colortype='gradient_colormap', gradientType='viridis').name == 'viridis'
	np.testing.assert_allclose(np.array(rgba(bsh.get_celltype_colors_new(12, colortype='tab10'), 12)),
	                           plt.get_cmap('hsv')(np.linspace(0, 1, 12)))


def test_categorical_cmap_lightens_each_base_colour():
	cmap, cols = bsh.categorical_cmap(3, 4, cmap='tab10')
	assert cols.shape == (12, 3)
	for i in range(3):
		h, s, v = colorsys.rgb_to_hsv(*plt.get_cmap('tab10')(i)[:3])
		for j, (s_j, v_j) in enumerate(zip(np.linspace(s, 0.25, 4), np.linspace(v, 1, 4))):
			np.testing.assert_allclose(cols[4 * i + j], colorsys.hsv_to_rgb(h, s_j, v_j), atol=1e-12)
	np.testing.assert_allclose(np.array([cmap(ind)[:3] for ind in range(12)]), cols)
	_, continuous = bsh.categorical_cmap(2, 1, cmap='viridis', continuous=True)
	np.testing.assert_allclose(continuous, plt.get_cmap('viridis')(np.linspace(0, 1, 2))[:, :3], atol=1e-12)
	with pytest.raises(ValueError):
		bsh.categorical_cmap(11, 2, cmap='tab10')


def test_small_helpers():
	assert bsh.color_code_to_color([0, 1], color_codes=['#aa0000ff', '#00bb00ff']) == [
		'background-color:#aa0000ff;width:2em', 'background-color:#00bb00ff;width:2em']
	assert bsh._json_safe({'a': np.arange(3), 'b': (np.float64(1.5), [np.int64(2)]), 'c': 'x'}) == {
		'a': [0, 1, 2], 'b': [1.5, [2]], 'c': 'x'}
	assert type(bsh._json_safe(np.int64(4))) is int
	assert bsh.Edge_info('thin').style_type == 'thin' and bsh.Bg_info().style_type == 'default'
	fig = bsh.get_placeholder_fig()
	ax = fig.axes[0]
	assert ax.get_xlim() == (-1, 1) and ax.get_ylim() == (-1, 1)
	assert [text.get_text() for text in ax.texts] == [
		"Bonsai-scout is loading the tree representation of your data.\n\n"
		"For large datasets, this can take a while. Please be patient."]
	plt.close(fig)


def test_cluster_names_give_size_and_top_annotations():
	clusters = pd.Series({'c1': 'cl_0', 'c2': 'cl_0', 'c3': 'cl_0', 'c4': 'cl_1', 'c5': 'cl_2'})
	names = bsh.rename_clusters_fast(clusters)
	assert dict(names) == {'cl_0': 'cl_0, Size=3', 'cl_1': 'cl_1, Size=1', 'cl_2': 'cl_2, Size=1'}
	annot = pd.Series({'c1': 'T', 'c2': 'T', 'c3': 'B', 'c4': 'B', 'c5': 'NK'})
	names = bsh.rename_clusters_fast(clusters, annot)
	assert dict(names) == {'cl_0': 'cl_0, Size=3, 67%=T, 33%=B', 'cl_1': 'cl_1, Size=1, 100%=B',
	                       'cl_2': 'cl_2, Size=1, 100%=NK'}
	cl_df = pd.DataFrame({'k2': clusters, 'k3': pd.Series({'c1': 'cl_0', 'c2': 'cl_1', 'c3': 'cl_1', 'c4': 'cl_1',
	                                                       'c5': 'cl_1'})})
	renamed = bsh.process_annot_based_clsts(cl_df)
	assert list(renamed.index) == list(cl_df.index) and list(renamed.columns) == ['k2', 'k3']
	assert list(renamed['k3']) == ['cl_0, Size=1'] + ['cl_1, Size=4'] * 4
	renamed = bsh.process_annot_based_clsts(cl_df[['k2']], cell2annot=dict(annot))
	assert list(renamed['k2']) == ['cl_0, Size=3, 67%=T, 33%=B'] * 3 + ['cl_1, Size=1, 100%=B',
	                                                                   'cl_2, Size=1, 100%=NK']
	# Clusters of cellstates, sizes and percentages counted over the cells of each cellstate.
	cs_df = pd.DataFrame({'k': pd.Series({'s1': 'cl_0', 's2': 'cl_1'})})
	cell2cs = {'a': 's1', 'b': 's1', 'c': 's1', 'd': 's2'}
	renamed = bsh.process_annot_based_clsts(cs_df, cell2annot={'a': 'T', 'b': 'T', 'c': 'B', 'd': 'B'},
	                                        cell2cs=cell2cs)
	assert list(renamed.index) == ['s1', 's2']
	assert list(renamed['k']) == ['cl_0, Size=3, 67%=T, 33%=B', 'cl_1, Size=1, 100%=B']


def test_annotation_info_and_feature_annotation():
	info = bsh.Annotation_info(cats=['x', 'y'], annot_to_color={'x': [1, 0, 0, 1], 'y': [0, 1, 0, 1]}, label='L',
	                           cbar_info={'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'log': False},
	                           annot_type='cells', color_type='categorical', info_object='cell_info_dict',
	                           info_key='annot_l', hidden=True, cat_counts_dict={'x': 2, 'y': 1})
	assert info.annot_to_color == {'x': (1, 0, 0, 1), 'y': (0, 1, 0, 1)}
	assert info.cbar_info['cmap'] is plt.get_cmap('viridis') or info.cbar_info['cmap'].name == 'viridis'
	as_dict = info.to_dict()
	assert as_dict['cbar_info'] == {'cmap': 'viridis', 'vmin': 0, 'vmax': 1, 'log': False}
	stored = json.loads(json.dumps(bsh._json_safe(as_dict)))
	again = bsh.Annotation_info(**copy.deepcopy(stored))
	assert json.loads(json.dumps(bsh._json_safe(again.to_dict()))) == stored
	feature = bsh.get_feature_annot_info(7, 'GeneX', 'data/normalized', cells_or_verts='verts')
	assert (feature.label, feature.info_key, feature.info_object, feature.annot_type, feature.color_type) == (
		'GeneX', 'Feature_7', 'data/normalized', 'verts', 'sequential')
	assert feature.cats is None and feature.annot_to_color is None
	assert feature.cbar_info['cmap'].name == 'YlOrRd' and feature.cbar_info['vmin'] is None
	assert bsh.get_feature_annot_info(1, 'g', 'data/x', gradient_type='Blues').cbar_info['cmap'].name == 'Blues'


def test_celltype_info_builds_annotations_from_info_dicts():
	"""Keys starting with 'annot_num_' are numerical (YlOrRd between the 1% and 99% quantiles), other 'annot_'
	keys categorical (natural-sort order, palette colours, 'NaN' gray); keys from the cluster table are hidden,
	and an annotation-guided clustering orders its clusters by size. Other keys are skipped."""
	values = [3.0, 1.0, np.nan, 7.0, 2.0]
	cell_info = {'annot_type': ['b10', 'b2', 'NaN', 'a', 'b2'], 'annot_num_depth': values, 'cell_ind_to_vert_ind': [0] * 5}
	cs_info = {'annot_batch': ['x', 'y', 'x', 'x', 'x']}
	cluster_info = {'annot_bnsi_cluster_n2': ['cl_1', 'cl_0', 'cl_0', 'cl_1', 'cl_1'],
	                'annot_bnsi_cluster_type': ['p', 'q', 'q', 'q', 'p']}
	info = bsh.Celltype_info(cell_info_dict=cell_info, cs_info_dict=cs_info, cluster_info_dict=cluster_info)
	assert info.annot_alts == ['annot_type', 'annot_num_depth', 'annot_batch', 'annot_bnsi_cluster_n2',
	                           'annot_bnsi_cluster_type']
	typ = info.annot_infos['annot_type']
	assert typ.cats == ['NaN', 'a', 'b2', 'b10'] and typ.label == 'Type' and typ.color_type == 'categorical'
	assert (typ.annot_type, typ.info_object, typ.hidden) == ('cells', 'cell_info_dict', False)
	assert typ.annot_to_color == {'NaN': GRAY_75, 'a': mcolors.to_rgba(PALETTE[0]), 'b2': mcolors.to_rgba(PALETTE[1]),
	                              'b10': mcolors.to_rgba(PALETTE[2])}
	depth = info.annot_infos['annot_num_depth']
	assert depth.label == 'Depth' and depth.cats is None and depth.color_type == 'sequential'
	assert depth.cbar_info['cmap'].name == 'YlOrRd'
	np.testing.assert_allclose([depth.cbar_info['vmin'], depth.cbar_info['vmax']],
	                           np.quantile([3.0, 1.0, 7.0, 2.0], [0.01, 0.99]), rtol=1e-12)
	batch = info.annot_infos['annot_batch']
	assert (batch.annot_type, batch.info_object, batch.cats) == ('cellstates', 'cs_info_dict', ['x', 'y'])
	n2 = info.annot_infos['annot_bnsi_cluster_n2']
	assert (n2.hidden, n2.info_object, n2.cats) == (True, 'cluster_info_dict', ['cl_0', 'cl_1'])
	guided = info.annot_infos['annot_bnsi_cluster_type']
	assert guided.cats == ['q', 'p'] and guided.cat_counts_dict == {'q': 3, 'p': 2}
	# Round trip through a dictionary, as the settings file stores it.
	stored = json.loads(json.dumps(bsh._json_safe(info.to_dict())))
	again = bsh.Celltype_info(**copy.deepcopy(stored))
	assert json.loads(json.dumps(bsh._json_safe(again.to_dict()))) == stored


def _celltype_info(annotations):
	"""A Celltype_info from {key: number of categories (None for numerical)}."""
	infos = {}
	for key, n_cats in annotations.items():
		cats = None if n_cats is None else ['c%d' % ind for ind in range(n_cats)]
		infos[key] = {'cats': cats, 'label': key, 'cbar_info': {'cmap': None}, 'info_key': key}
	return bsh.Celltype_info(annot_alts=list(annotations), annot_infos=infos)


def test_initial_annotation_prefers_cell_type_names_then_about_ten_categories():
	assert _celltype_info({'annot_default': 1}).get_init_annot() == 'annot_default'
	base = {'annot_default': 1, 'annot_bnsi_cluster_n2': 2, 'annot_cluster_n3': 3, 'annot_few': 3, 'annot_many': 12,
	        'annot_num_x': None}
	assert _celltype_info(base).get_init_annot() == 'annot_many'
	assert _celltype_info(dict(base, annot_celltype_v2=30)).get_init_annot() == 'annot_celltype_v2'
	assert _celltype_info(dict(base, annot_celltype_v2=30, annot_CellTypeName=40)).get_init_annot() == \
		'annot_CellTypeName'
	assert _celltype_info({'annot_default': 1, 'annot_num_x': None}).get_init_annot() == 'annot_default'
	assert _celltype_info({'annot_bnsi_cluster_n2': 2, 'annot_num_x': None}).get_init_annot() == \
		'annot_bnsi_cluster_n2'


def test_verttype_info_adds_cell_number_annotations():
	n_cells = np.array([0, 1, 2, 0, 3])
	vert_info = {'n_cells_per_vert': n_cells, 'annot_side': ['l', 'r', 'NaN', 'l', 'r'],
	             'annot_num_w': [0.5, 1.5, 2.5, np.nan, 4.0]}
	info = bsh.Verttype_info(vert_info_dict=vert_info)
	assert info.annot_alts == ['annot_side', 'annot_num_w', 'cell_number', 'binarized_cell_number']
	side = info.annot_infos['annot_side']
	assert side.cats == ['NaN', 'l', 'r'] and side.annot_type == 'verts' and side.info_object == 'vert_info_dict'
	assert side.annot_to_color == {'NaN': GRAY_75, 'l': mcolors.to_rgba(PALETTE[0]), 'r': mcolors.to_rgba(PALETTE[1])}
	w = info.annot_infos['annot_num_w']
	np.testing.assert_allclose([w.cbar_info['vmin'], w.cbar_info['vmax']],
	                           np.quantile([0.5, 1.5, 2.5, 4.0], [0.01, 0.99]), rtol=1e-12)
	np.testing.assert_array_equal(info.new_cell_info_dict['cell_number'], n_cells)
	np.testing.assert_array_equal(info.new_cell_info_dict['binarized_cell_number'], [0, 1, 1, 0, 1])
	number = info.annot_infos['cell_number']
	assert (number.label, number.color_type, number.info_object) == ('Number of cells', 'sequential',
	                                                                  'new_cell_info_dict')
	assert (number.cbar_info['vmin'], number.cbar_info['vmax']) == (1, 3)
	assert info.annot_infos['binarized_cell_number'].label == 'Default sizing'
	stored = info.to_dict()
	assert stored['new_cell_info_dict']['cell_number'] == (0, 1, 2, 0, 3)
	assert stored['new_cell_info_dict']['binarized_cell_number'] == (0, 1, 1, 0, 1)
	again = bsh.Verttype_info(**copy.deepcopy(json.loads(json.dumps(bsh._json_safe(stored)))))
	np.testing.assert_array_equal(again.new_cell_info_dict['cell_number'], n_cells)
	assert again.annot_infos['annot_side'].annot_to_color == side.annot_to_color


def test_marker_genes_stop_gives_an_empty_table():
	table = bsh.update_marker_genes_df('stop', None)
	assert list(table.columns) == ['marker_genes', 'marker_scores'] and len(table) == 0


# ------------------------------------------------------ figures from HDF

@pytest.fixture
def figure_factory():
	"""Builds Bonvis_figure objects from a preprocessed results folder and closes them afterwards."""
	opened = []

	def make(folder, create=True):
		path = os.path.join(folder, 'bonsai_vis_data.hdf')
		metadata = bsh.Bonvis_metadata(path)
		settings = bsh.Bonvis_settings(load_settings_path=os.path.join(folder, 'bonsai_vis_settings.json'))
		hdf = h5py.File(path, 'r')
		opened.append(hdf)
		fig = bsh.Bonvis_figure(hdf, metadata, bonvis_data_path=path, bonvis_settings=settings)
		if create:
			fig.create_figure(figsize=(3, 3))
		return fig

	yield make
	for hdf in opened:
		hdf.close()
	plt.close('all')


def _rich(scout_rich):
	scout_rich.check('default')
	return scout_rich.folder('default')


def _hdf(folder):
	return h5py.File(os.path.join(folder, 'bonsai_vis_data.hdf'), 'r')


def _edges(folder):
	edge_df = pd.read_hdf(os.path.join(folder, 'bonsai_vis_data.hdf'), key='tree_info/edge_df')
	return np.array(edge_df['source_ind']), np.array(edge_df['target_ind']), np.array(edge_df['dist'])


def _below(folder, vert):
	"""Vertex indices in the subtree below vert (vert included), from the stored edges."""
	src, tgt, _ = _edges(folder)
	out, stack = [], [vert]
	while stack:
		v = stack.pop()
		out.append(v)
		stack.extend(tgt[src == v].tolist())
	return sorted(out)


def test_bonvis_metadata_reads_the_hdf(scout_rich):
	folder = _rich(scout_rich)
	path = os.path.join(folder, 'bonsai_vis_data.hdf')
	metadata = bsh.Bonvis_metadata(path)
	with _hdf(folder) as hdf:
		raw = json.loads(hdf.attrs['metadata_json'])
		assert metadata.feature_paths == json.loads(hdf.attrs['feature_paths'])
		n_cells_per_vert = hdf['vert_info/n_cells_per_vert'][()]
		assert metadata.tree_info['nwk_str'] == hdf['tree_info'].attrs['nwk_str']
		np.testing.assert_array_equal(metadata.tree_info['int_inds'], hdf['tree_info/int_inds'][()])
		np.testing.assert_array_equal(metadata.cs_info['multi_cs_at_vert'], hdf['cs_info/multi_cs_at_vert'][()])
	assert (metadata.node_ids, metadata.gene_ids, metadata.cell_ids, metadata.cs_ids) == (
		raw['nodeIds'], raw['geneIds'], raw['cellIds'], raw['csIds'])
	assert (metadata.n_nodes, metadata.n_genes, metadata.n_cells, metadata.n_Css) == (
		len(raw['nodeIds']), len(raw['geneIds']), raw['nCells'], raw['nCss'])
	np.testing.assert_array_equal(metadata.cell_info['int_vert_inds'], np.where(n_cells_per_vert == 0)[0])
	np.testing.assert_array_equal(metadata.cell_info['non_int_vert_inds'], np.where(n_cells_per_vert > 0)[0])
	assert metadata.tree_info['node_id_to_n_cells'] == dict(zip(raw['nodeIds'], n_cells_per_vert.tolist()))
	edge_df = pd.read_hdf(path, key='tree_info/edge_df')
	assert metadata.tree_info['edge_dict'] == edge_df.to_dict(orient='list')
	assert metadata.cs_info['cs_info_dict'] == pd.read_hdf(path, key='cs_info/cs_info_dict').to_dict(orient='list')
	assert metadata.clusters_cut_edges == {}


def test_settings_constructor_and_json_round_trip(scout_rich, tmp_path):
	"""A fresh Bonvis_settings from the metadata equals the stored settings file, and to_json/from_json keeps
	everything. set_annot and set_size_style accept known labels and fall back for unknown ones."""
	folder = _rich(scout_rich)
	metadata = bsh.Bonvis_metadata(os.path.join(folder, 'bonsai_vis_data.hdf'))
	fresh = bsh.Bonvis_settings(bonvis_metadata=metadata, geometry='hyperbolic', origin_style='root',
	                            ly_type='ly_dendrogram_ladderized')
	with open(os.path.join(folder, 'bonsai_vis_settings.json')) as handle:
		stored = json.load(handle)
	assert json.loads(json.dumps(bsh._json_safe(fresh.to_dict()))) == stored
	fresh.to_json(str(tmp_path / 'settings.json'))
	again = bsh.Bonvis_settings(load_settings_path=str(tmp_path / 'settings.json'))
	assert json.loads(json.dumps(bsh._json_safe(again.to_dict()))) == stored
	again.set_ly_type('ly_eq_angle')
	assert again.ly_type == 'ly_eq_angle' and again.transf_info.no_transform is False
	again.set_ly_type('ly_dendrogram')
	assert again.transf_info.no_transform is True and again.transf_info.geometry == 'flat'
	again.set_annot(annot_label='annot_partial')
	assert again.node_style['annot_info'].info_key == 'annot_partial'
	again.set_annot(annot_label='cell_number')
	assert again.node_style['annot_info'].info_key == 'cell_number'
	again.set_annot(annot_label='no_such_annotation')
	assert again.node_style['annot_info'].info_key == again.celltype_info.annot_alts[0]
	again.set_size_style(annot_label='annot_num_depth')
	assert again.node_style['size_annot_info'].info_key == 'annot_num_depth'
	again.set_size_style(annot_label='annot_celltype')  # categorical: ignored
	assert again.node_style['size_annot_info'].info_key == 'annot_num_depth'
	again.set_size_style(annot_label='binarized_cell_number')
	assert again.node_style['size_annot_info'].info_key == 'binarized_cell_number'
	again.set_size_style(annot_label='no_such_annotation')
	assert again.node_style['size_annot_info'].info_key == 'cell_number'
	# Loading a settings file whose colouring is a Bonsai clustering switches to the initial annotation.
	stored['node_style']['annot_info'] = stored['celltype_info']['annot_infos']['annot_bnsi_cluster_n3']
	with open(tmp_path / 'cluster.json', 'w') as handle:
		json.dump(stored, handle)
	loaded = bsh.Bonvis_settings(load_settings_path=str(tmp_path / 'cluster.json'))
	assert loaded.node_style['annot_info'].info_key == 'annot_celltype'


def test_figure_nodes_edges_and_sizes(scout_rich, figure_factory):
	"""The default view (ladderized dendrogram, untransformed, coloured by celltype per cellstate, sized by the
	number of cells): vertices at the stored coordinates, edges as stored, vertices without cells small and
	gray, single cellstates coloured by their annotation, vertices with several cellstates drawn as wedges."""
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	with _hdf(folder) as hdf:
		coords = hdf['layout_coords/node_coords/ly_dendrogram_ladderized'][()]
		edge_coords = hdf['layout_coords/edge_coords/ly_dendrogram_ladderized'][()]
		n_cells = hdf['vert_info/n_cells_per_vert'][()]
		single_cs = hdf['cs_info/single_cs_at_vert'][()]
		vert_to_cs = json.loads(hdf['vert_info'].attrs['vert_ind_to_cs_inds_json'])
	cs_info = pd.read_hdf(os.path.join(folder, 'bonsai_vis_data.hdf'), key='cs_info/cs_info_dict')
	np.testing.assert_array_equal(fig.coords_info.transf_node_coords_nx, coords)
	np.testing.assert_array_equal(np.array(fig.edge_obj['coll'].get_segments()), edge_coords)
	internal = np.where(n_cells == 0)[0]
	np.testing.assert_array_equal(fig.int_obj['vert_inds'], internal)
	np.testing.assert_array_equal(fig.int_obj['coll'].get_offsets(), coords[internal])
	assert np.all(fig.int_obj['coll'].get_facecolors() == np.array(GRAY_75))
	# Sizes: 0.005 without cells; with cells 0.015 * (1 + sqrt((n - min) / (max - min)) * 0.5) (max below 8).
	expected_size = np.where(n_cells == 0, 0.005,
	                         0.015 * (1 + np.sqrt(np.clip(n_cells - 1, 0, None) / (n_cells.max() - 1)) * 0.5))
	np.testing.assert_allclose(fig.vert_to_size, expected_size, rtol=1e-12)
	np.testing.assert_allclose(fig.int_obj['coll'].get_widths(), expected_size[internal], rtol=1e-12)
	cs_vert = np.array(cs_info['cs_ind_to_vert_ind'])
	np.testing.assert_array_equal(fig.single_obj['obj_inds'], single_cs)
	np.testing.assert_array_equal(fig.single_obj['coll'].get_offsets(), coords[cs_vert[single_cs]])
	colours = {'left': mcolors.to_rgba(PALETTE[0]), 'right': mcolors.to_rgba(PALETTE[1])}
	np.testing.assert_allclose(fig.single_obj['coll'].get_facecolors(),
	                           [colours[str(cs_info['annot_celltype'].iloc[cs])] for cs in single_cs])
	multi = sorted(int(v) for v, css in vert_to_cs.items() if len(css) > 1)
	assert multi and fig.multi_obj['vert_inds_per_wedge'] == multi
	for wedge, vert in zip(fig.multi_obj['wedges'], multi):
		assert (wedge.theta1, wedge.theta2) == (0, 360) and wedge.r == pytest.approx(expected_size[vert] / 2)
	# Colouring by 'Partial' splits each multi-cellstate vertex into wedges per category.
	fig.update_figure(node_style='Partial', verbose=False)
	partial = [str(value) for value in cs_info['annot_partial']]
	wedges = list(zip(fig.multi_obj['vert_inds_per_wedge'], fig.multi_obj['wedges']))
	for vert in multi:
		cats, counts = np.unique([partial[cs] for cs in vert_to_cs[str(vert)]], return_counts=True)
		angles = np.concatenate([[0], np.cumsum(360 * counts / counts.sum())])
		got = [(w.theta1, w.theta2) for v, w in wedges if v == vert]
		np.testing.assert_allclose(np.array(got), np.stack([angles[:-1], angles[1:]], axis=1))
	assert any(len(np.unique([partial[cs] for cs in vert_to_cs[str(v)]])) > 1 for v in multi)


def test_figure_of_real_cells_mapped_to_cellstates(scout_rich, figure_factory):
	"""With a cell-to-cellstate mapping the initial colouring is a per-cell annotation: cells alone at a vertex
	are discs in their category's colour, and a vertex with several cells is split into wedges in proportion to
	the cells of each category; sizes count the real cells."""
	scout_rich.check('cellstates')
	folder = scout_rich.folder('cellstates')
	fig = figure_factory(folder)
	annot = fig.bonvis_settings.node_style['annot_info']
	assert (annot.info_key, annot.info_object, annot.annot_type) == ('annot_celltype', 'cell_info_dict', 'cells')
	cell_info = pd.read_hdf(os.path.join(folder, 'bonsai_vis_data.hdf'), key='cell_info/cell_info_dict')
	with _hdf(folder) as hdf:
		single = hdf['cell_info/single_at_vert'][()]
		n_cells = hdf['vert_info/n_cells_per_vert'][()]
		vert_to_cells = json.loads(hdf['vert_info'].attrs['vert_ind_to_cell_inds_json'])
	celltype = [str(value) for value in cell_info['annot_celltype']]
	colours = {'left': mcolors.to_rgba(PALETTE[0]), 'right': mcolors.to_rgba(PALETTE[1])}
	np.testing.assert_array_equal(fig.single_obj['obj_inds'], single)
	np.testing.assert_allclose(fig.single_obj['coll'].get_facecolors(), [colours[celltype[c]] for c in single])
	multi = sorted(int(v) for v, cells in vert_to_cells.items() if len(cells) > 1)
	assert fig.multi_obj['vert_inds_per_wedge'] == [v for v in multi for _ in
	                                                range(len(set(celltype[c] for c in vert_to_cells[str(v)])))]
	for vert, wedge in zip(fig.multi_obj['vert_inds_per_wedge'], fig.multi_obj['wedges']):
		assert wedge.theta2 - wedge.theta1 == pytest.approx(360.0)
		assert wedge.r == pytest.approx(fig.vert_to_size[vert] / 2)
	# Colouring by a per-cell category that splits a vertex unevenly: wedges in proportion to the cells.
	fig.update_figure(node_style='Parity', verbose=False)
	parity = [cell_parity(cell) for cell in cell_info.index]
	wedges = list(zip(fig.multi_obj['vert_inds_per_wedge'], fig.multi_obj['wedges']))
	uneven = 0
	for vert in multi:
		cats, counts = np.unique([parity[c] for c in vert_to_cells[str(vert)]], return_counts=True)
		uneven += len(set(counts)) > 1
		angles = np.concatenate([[0], np.cumsum(360 * counts / counts.sum())])
		got = [(w.theta1, w.theta2) for v, w in wedges if v == vert]
		np.testing.assert_allclose(np.array(got), np.stack([angles[:-1], angles[1:]], axis=1))
	assert uneven > 0
	# Fewer than 8 cells at the fullest vertex: radii from 0.015 up to 1.5 times that.
	low, high = n_cells[n_cells > 0].min(), n_cells.max()
	assert high < 8 and high > low
	scaled = np.sqrt(np.clip(n_cells - low, 0, None) / (high - low))
	np.testing.assert_allclose(fig.vert_to_size, np.where(n_cells == 0, 0.005, 0.015 * (1 + scaled * 0.5)), rtol=1e-12)


def test_hyperbolic_view_zoom_origin_and_axis_limits(scout_rich, figure_factory):
	"""Switching to the equal-angle layout in hyperbolic geometry centres the view on the root, picks the zoom
	that puts 80% of the vertices within radius 0.8 and maps every vertex and edge point through the Poincare
	map; zooming multiplies the zoom; a double click moves the clicked point to the centre; new axis limits get
	a 7.5% margin and stay within +-1.05; the background lines follow the axis limits."""
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	with _hdf(folder) as hdf:
		coords = hdf['layout_coords/node_coords/ly_eq_angle'][()]
		edge_coords = hdf['layout_coords/edge_coords/ly_eq_angle'][()]
	src, _, _ = _edges(folder)
	fig.update_figure(ly_type='ly_eq_angle', geometry='hyperbolic', verbose=False)
	info = fig.bonvis_settings.transf_info
	assert (fig.bonvis_settings.ly_type, info.geometry, info.no_transform) == ('ly_eq_angle', 'hyperbolic', False)
	np.testing.assert_array_equal(info.origin, coords[src[0]])
	zoom = info.get_zoom()
	radii = np.sort(np.linalg.norm(_poincare(coords, info.origin, zoom), axis=1))
	assert radii[math.ceil(0.8 * len(coords)) - 1] == pytest.approx(0.8, abs=1e-9)
	np.testing.assert_allclose(fig.coords_info.transf_node_coords_nx, _poincare(coords, info.origin, zoom), rtol=1e-12)
	np.testing.assert_allclose(np.array(fig.edge_obj['coll'].get_segments()), _poincare(edge_coords, info.origin, zoom),
	                           rtol=1e-12, atol=1e-15)
	# Background: five horizontal and five vertical lines on each side, at multiples of the distance d that
	# the map takes to radius 0.4, each drawn with 20 points and mapped with the current zoom.
	d = 0.4 * 2 / (1 - 0.4 ** 2) / zoom
	assert np.linalg.norm(_poincare([0, d], np.zeros(2), zoom)) == pytest.approx(0.4, rel=1e-12)
	offsets = np.hstack([-d * np.arange(1, 6), d * np.arange(1, 6)])
	span = np.linspace(-5 * d, 5 * d, 20)
	lines = [np.stack([span, np.full(20, y)], axis=1) for y in offsets]
	lines += [np.stack([np.full(20, x), span], axis=1) for x in offsets]
	first = fig.coords_info.transf_bg_edge_coords_eix
	np.testing.assert_allclose(first, _poincare(np.array(lines), np.zeros(2), zoom), rtol=1e-12, atol=1e-15)
	fig.update_figure(zoom=2, verbose=False)
	assert info.get_zoom() == pytest.approx(2 * zoom, rel=1e-15)
	np.testing.assert_allclose(fig.coords_info.transf_node_coords_nx, _poincare(coords, info.origin, 2 * zoom),
	                           rtol=1e-12)
	click = np.array([0.3, -0.2])
	old_origin = info.origin.copy()
	fig.update_figure(origin=click, verbose=False)
	expected_origin = click * 2 / (1 - np.sum(click ** 2)) / (2 * zoom) + old_origin
	np.testing.assert_allclose(np.ravel(info.origin), expected_origin, rtol=1e-12)
	fig.update_figure(ax_lims=[-0.5, 0.5, -0.2, 0.4], verbose=False)
	np.testing.assert_allclose(info.ax_lims, [-0.575, 0.575, -0.245, 0.445], rtol=1e-12)
	fig.update_figure(ax_lims=[-1.0, 1.0, -1.0, 1.0], verbose=False)
	np.testing.assert_allclose(info.ax_lims, [-1.05, 1.05, -1.05, 1.05])
	fig.update_figure(ax_lims=[-0.5, 0.5, -0.2, 0.4], verbose=False)
	fig.update_figure(zoom_ax_lims=0.5, verbose=False)
	np.testing.assert_allclose(info.ax_lims, [-0.2875, 0.2875, -0.0725, 0.2725], rtol=1e-12)
	xmin, xmax, ymin, ymax = info.ax_lims
	bg = fig.coords_info.data_bg_edge_coords_eix
	np.testing.assert_allclose(bg[..., 0], (first[..., 0] - xmin) / (xmax - xmin), atol=1e-12)
	np.testing.assert_allclose(bg[..., 1], (first[..., 1] - ymin) / (ymax - ymin), atol=1e-12)


def test_colours_by_numerical_annotation_and_by_gene(scout_rich, figure_factory):
	"""find_annot looks a label up among the annotations (case-insensitive), then among the genes. Numerical
	values are coloured with YlOrRd between vmin and vmax; a gene's vmin and vmax are its 3% and 97% quantiles
	over the vertices, and a gene colours every vertex."""
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	settings = fig.bonvis_settings
	assert bsh.find_annot('celltype', settings, fig.bonvis_data, fig.bonvis_metadata).info_key == 'annot_celltype'
	assert bsh.find_annot('NUMBER OF CELLS', settings, fig.bonvis_data, fig.bonvis_metadata).info_key == 'cell_number'
	assert bsh.find_annot('no_such_label', settings, fig.bonvis_data, fig.bonvis_metadata) is None
	cs_info = pd.read_hdf(os.path.join(folder, 'bonsai_vis_data.hdf'), key='cs_info/cs_info_dict')
	fig.update_figure(node_style='Depth', verbose=False)
	depth = fig.bonvis_settings.node_style['annot_info']
	assert depth.info_key == 'annot_num_depth'
	values = np.array(cs_info['annot_num_depth'])
	cmap = plt.get_cmap('YlOrRd')
	expected = cmap((values - depth.cbar_info['vmin']) / (depth.cbar_info['vmax'] - depth.cbar_info['vmin']))
	single = fig.single_obj['obj_inds']
	np.testing.assert_allclose(fig.single_obj['coll'].get_facecolors(), expected[single], rtol=1e-12)
	celltypes = fig.get_cell_to_celltype()
	np.testing.assert_allclose(celltypes, values)
	gene = fig.bonvis_metadata.gene_ids[3]
	fig.update_figure(node_style=gene.lower(), verbose=False)
	annot = fig.bonvis_settings.node_style['annot_info']
	assert (annot.info_key, annot.annot_type, annot.info_object) == ('Feature_3', 'verts', 'data/normalized')
	with _hdf(folder) as hdf:
		gene_values = hdf['data/normalized/means'][3]
	vmin, vmax = np.quantile(gene_values, [0.03, 0.97])
	assert (annot.cbar_info['vmin'], annot.cbar_info['vmax']) == pytest.approx((vmin, vmax), rel=1e-12)
	colours = cmap((gene_values - vmin) / (vmax - vmin))
	np.testing.assert_allclose(fig.int_obj['coll'].get_facecolors(), colours[fig.int_obj['vert_inds']], rtol=1e-12)
	np.testing.assert_allclose(fig.single_obj['coll'].get_facecolors(), colours[fig.single_obj['vert_inds']],
	                           rtol=1e-12)
	assert fig.multi_obj is None
	# Vertex colours make internal vertices data-sized (0.010).
	assert np.all(fig.vert_to_size[fig.int_obj['vert_inds']] == 0.010)


def test_sizes_from_a_numerical_annotation_and_scaling(scout_rich, figure_factory):
	"""Sizing by a numerical cellstate annotation: 0.015 * (1 + sqrt((x - min) / (max - min)) * 2.5) at the
	cellstate's vertex. Scaling nodes and edges multiplies radii, line widths and edge widths."""
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	cs_info = pd.read_hdf(os.path.join(folder, 'bonsai_vis_data.hdf'), key='cs_info/cs_info_dict')
	fig.update_figure(size_style='Depth', verbose=False)
	values = np.array(cs_info['annot_num_depth'])
	cs_vert = np.array(cs_info['cs_ind_to_vert_ind'])
	single = fig.single_obj['obj_inds']
	expected = 0.015 * (1 + np.sqrt((values - values.min()) / (values.max() - values.min())) * 2.5)
	np.testing.assert_allclose(fig.vert_to_size[cs_vert[single]], expected[single], rtol=1e-12)
	np.testing.assert_allclose(fig.vert_to_size[fig.int_obj['vert_inds']], 0.005)
	before = fig.vert_to_size.copy()
	lw = fig.bonvis_settings.node_style['lw_cell']
	fig.update_figure(scale_nodes=2, verbose=False)
	np.testing.assert_allclose(fig.vert_to_size, 2 * before, rtol=1e-12)
	assert fig.bonvis_settings.node_style['lw_cell'] == pytest.approx(2 * lw)
	fig.update_figure(scale_node_edges=0.5, verbose=False)
	assert fig.bonvis_settings.node_style['lw_cell'] == pytest.approx(lw)
	fig.update_figure(scale_edges=2, verbose=False)
	assert fig.bonvis_settings.edge_style['linewidth'] == pytest.approx(0.5)
	np.testing.assert_allclose(fig.edge_obj['coll'].get_linewidths(), [0.5])


def test_small_categories_share_one_colour(scout_rich, figure_factory):
	"""set_small_type_cutoff gives every category with fewer cells than the cutoff the first palette colour
	and the others the next colours, in category order."""
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	info = fig.bonvis_settings.celltype_info.annot_infos['annot_celltype']
	fig.set_small_type_cutoff(info, 5)
	assert info.small_type_cutoff == 5 and info.cat_counts_dict == {'left': 6, 'right': 4}
	assert info.annot_to_color == {'right': tuple(mcolors.to_rgba(PALETTE[0])),
	                               'left': tuple(mcolors.to_rgba(PALETTE[1]))}
	depth = fig.bonvis_settings.celltype_info.annot_infos['annot_categorized_depth']
	fig.set_small_type_cutoff(depth, 2)
	assert set(depth.annot_to_color.values()) == {tuple(mcolors.to_rgba(PALETTE[0]))}
	guided = fig.bonvis_settings.celltype_info.annot_infos['annot_bnsi_cluster_celltype']
	fig.set_small_type_cutoff(guided, 5)
	big = [cat for cat, count in guided.cat_counts_dict.items() if count >= 5]
	assert len(big) == 1 and guided.annot_to_color[big[0]] == tuple(mcolors.to_rgba(PALETTE[1]))


def test_pick_node_finds_the_nearest_vertices(scout_rich, figure_factory):
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	coords = fig.coords_info.node_coords_nx
	int_inds = fig.bonvis_metadata.tree_info['int_inds']
	cell_inds = fig.bonvis_metadata.tree_info['cell_inds']
	for point in (coords[5] + [0.01, -0.005], np.array([0.1, 0.2]), coords[int(cell_inds[0])]):
		dist = np.linalg.norm(coords - point, axis=1)
		picked = fig.pick_node(np.array(point, dtype=float))
		assert picked['int_inds'] == int_inds[np.argmin(dist[int_inds])]
		assert picked['cell_inds'] == cell_inds[np.argmin(dist[cell_inds])]
		assert picked['all'] == np.argmin(dist)
	# In the hyperbolic view the clicked point is first mapped back to layout coordinates.
	fig.update_figure(ly_type='ly_eq_angle', geometry='hyperbolic', verbose=False)
	shown = fig.coords_info.transf_node_coords_nx
	assert fig.pick_node(shown[7] + 1e-6)['all'] == 7


def test_layout_tweaks_keep_branch_lengths(scout_rich, figure_factory):
	"""Tweaking the equal-angle layout (widening a subtree) and the equal-daylight layout, flipping branches and
	re-rooting the dendrogram all redraw the tree with its branch lengths; the widened subtree's ancestor stays
	where it was on screen, and a re-rooted dendrogram starts at x = -0.95 from the new root."""
	folder = _rich(scout_rich)
	src, tgt, dist = _edges(folder)

	def check_lengths(coords):
		np.testing.assert_allclose(np.linalg.norm(coords[tgt] - coords[src], axis=1), dist, rtol=1e-9)

	fig = figure_factory(folder)
	fig.update_figure(ly_type='ly_eq_angle', geometry='hyperbolic', verbose=False)
	before = fig.coords_info.transf_node_coords_nx.copy()
	ancestor, inside = int(src[0]), int(tgt[0])
	fig.update_figure(tweak_inds=(ancestor, inside), multip_angle=4, verbose=False)
	check_lengths(fig.coords_info.node_coords_nx)
	assert not np.allclose(fig.coords_info.transf_node_coords_nx, before)
	np.testing.assert_allclose(fig.coords_info.transf_node_coords_nx[ancestor], before[ancestor], atol=1e-12)
	fig.update_figure(new_flip_id=True, flipped_node_ids=[fig.bonvis_metadata.node_ids[inside]], verbose=False)
	check_lengths(fig.coords_info.node_coords_nx)

	fig = figure_factory(folder)
	fig.update_figure(ly_type='ly_eq_daylight', verbose=False)
	internal = int(src[1])
	fig.update_figure(tweak_inds=internal, verbose=False)
	check_lengths(fig.coords_info.node_coords_nx)

	fig = figure_factory(folder)
	node_ids = fig.bonvis_metadata.node_ids
	children = tgt[src == src[0]]
	leaves_of = {child: [v for v in _below(folder, child) if v not in src] for child in children}
	x, y = fig.coords_info.node_coords_nx[:, 0].copy(), fig.coords_info.node_coords_nx[:, 1].copy()
	order = [np.mean(y[leaves_of[child]]) for child in children]
	fig.update_figure(new_flip_id=True, flipped_node_ids=[node_ids[src[0]]], verbose=False)
	flipped_y = fig.coords_info.node_coords_nx[:, 1]
	flipped = [np.mean(flipped_y[leaves_of[child]]) for child in children]
	# Flipping swaps the vertical order of the root's two subtrees and keeps depths and leaf heights.
	assert np.sign(order[0] - order[1]) == -np.sign(flipped[0] - flipped[1])
	np.testing.assert_allclose(fig.coords_info.node_coords_nx[:, 0], x, atol=1e-15)
	leaves = [v for v in range(len(node_ids)) if v not in src]
	np.testing.assert_allclose(np.sort(flipped_y[leaves]), np.sort(y[leaves]), atol=1e-15)
	new_root = int(tgt[-1])
	fig.update_figure(tweak_inds=new_root, verbose=False)
	coords = fig.coords_info.node_coords_nx
	assert coords[new_root, 0] == pytest.approx(-0.95)
	edge_dict = fig.bonvis_settings.upd_edge_coords['ly_dendrogram_ladderized']
	depth = np.full(len(node_ids), np.nan)
	depth[new_root] = 0.0
	adjacency = {}
	for s, t, d in zip(src, tgt, dist):
		adjacency.setdefault(s, []).append((t, d))
		adjacency.setdefault(t, []).append((s, d))
	stack = [new_root]
	while stack:
		v = stack.pop()
		for w, d in adjacency[v]:
			if np.isnan(depth[w]):
				depth[w] = depth[v] + d
				stack.append(w)
	np.testing.assert_allclose(coords[:, 0], -0.95 + 1.9 * depth / depth.max(), atol=1e-12)
	assert edge_dict.shape == (len(src), 3, 2)


def test_subsets_masks_and_masked_colours(scout_rich, figure_factory):
	"""Selections of cells: a subtree (the side of an edge), a category of an annotation, or vertices with a
	given number of cells. Masking everything outside a selection colours those cellstates light gray."""
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	metadata = fig.bonvis_metadata
	src, tgt, _ = _edges(folder)
	vert_to_cells = metadata.vert_info['vert_ind_to_cell_inds']
	ancestor, child = int(src[0]), int(tgt[0])
	below = _below(folder, child)
	cells_below = sorted(c for v in below for c in vert_to_cells[str(v)])
	got, kind = fig.get_cell_inds_in_subset({'type': 'subtree', 'info': [ancestor, child]})
	assert kind == 'cells' and sorted(got.tolist()) == cells_below
	got, kind = fig.get_cell_inds_in_subset({'type': 'subtree', 'info': [ancestor, child]}, return_vert_inds=True)
	assert kind == 'verts' and sorted(got.tolist()) == below
	# The other side: the ancestor seen from inside the subtree.
	got, _ = fig.get_cell_inds_in_subset({'type': 'subtree', 'info': [child, ancestor]})
	assert sorted(got.tolist()) == sorted(set(range(metadata.n_cells)) - set(cells_below))
	left = [ind for ind, cell in enumerate(metadata.cell_ids) if clade_of(cell) == 'left']
	got, kind = fig.get_cell_inds_in_subset({'type': 'annot', 'info': 'left'}, curr_categorical_annot='Celltype')
	assert kind == 'cells' and sorted(got) == left
	n_cells = metadata.vert_info['n_cells_per_vert']
	got, kind = fig.get_cell_inds_in_subset({'type': 'annot', 'info': 2}, curr_categorical_annot='Number of cells',
	                                        return_vert_inds='flexible')
	assert kind == 'verts' and got.tolist() == np.where(n_cells == 2)[0].tolist()
	assert fig.get_cell_inds_in_subset({'type': None}) == (None, 'cells')
	assert fig.get_cell_inds_in_subset({'type': 'unknown'}) == (None, 'cells')

	fig.set_mask_for_subset([{'mask_is_on': True, 'type': 'annot', 'info': 'left'}],
	                        curr_categorical_annot='Celltype')
	style = fig.bonvis_settings.node_style
	right = [ind for ind in range(metadata.n_cells) if ind not in left]
	assert style['use_mask'] is True and style['verts_masked'] is None and style['cells_masked'] == right
	celltypes, colours = fig.get_color_info(style['annot_info'])
	for cs in range(metadata.n_Css):
		if cs in right:
			assert tuple(colours[cs]) == pytest.approx(MASKED) and celltypes[cs] == 'nan'
		else:
			assert tuple(colours[cs]) == pytest.approx(mcolors.to_rgba(PALETTE[0])) and celltypes[cs] == 'left'
	fig.set_mask_for_subset([{'mask_is_on': False}])
	assert (style['use_mask'], style['verts_masked'], style['cells_masked']) == (False, None, None)


def test_marker_genes_from_the_figure(scout_rich, figure_factory):
	"""preprocess_marker_genes turns two selections into the arguments of the marker-gene calculation (cells
	mapped to their vertices for the posterior means); update_marker_genes_df computes the table. Scores with
	error bars lie within the integration bound of the exact pairwise probability; scores from ranks are the
	exact fraction of cell pairs."""
	folder = _rich(scout_rich)
	path = os.path.join(folder, 'bonsai_vis_data.hdf')
	fig = figure_factory(folder)
	metadata = fig.bonvis_metadata
	src, tgt, _ = _edges(folder)
	ancestor, child = int(src[0]), int(tgt[0])
	group1, _ = fig.get_cell_inds_in_subset({'type': 'subtree', 'info': [ancestor, child]})
	group1 = np.sort(group1)
	group2 = np.setdiff1d(np.arange(metadata.n_cells), group1)
	run_with_vars, args, time_estimate, info = fig.preprocess_marker_genes(path, {'type': 'subtree',
	                                                                              'info': [ancestor, child]})
	assert run_with_vars == 'yes'
	cell_vert = np.array(metadata.cell_info['cell_info_dict']['cell_ind_to_vert_ind'])
	np.testing.assert_array_equal(args[0], np.sort(cell_vert[group1]))
	np.testing.assert_array_equal(args[1], np.sort(cell_vert[group2]))
	assert args[2:6] == (path, 'data/normalized/means', 'data/normalized/vars', metadata.gene_ids) and args[6] == 300
	assert time_estimate == pytest.approx(5e-8 * metadata.n_cells * metadata.n_genes * 300)
	assert info == {'feature_path': 'data/normalized',
	                'cell_ids_group1': [metadata.cell_ids[ind] for ind in group1],
	                'cell_ids_group2': [metadata.cell_ids[ind] for ind in group2]}
	table = bsh.update_marker_genes_df(run_with_vars, args)
	with _hdf(folder) as hdf:
		means, variances = hdf['data/normalized/means'][()], hdf['data/normalized/vars'][()]
	exact = gaussian_marker_score(means[:, args[0]], variances[:, args[0]], means[:, args[1]], variances[:, args[1]])
	bound = two_sigma_truncation(means[:, args[0]], variances[:, args[0]])
	for gene, score in zip(table['marker_genes'], table['marker_scores']):
		g = metadata.gene_ids.index(gene)
		assert exact[g] - bound[g] - 2e-4 <= score <= exact[g] + 2e-4
	extremeness = np.minimum(table['marker_scores'].values, 1 - table['marker_scores'].values)
	assert np.all(np.diff(extremeness) >= 0)

	# Without error bars (the mat_extra data matrix): ranks, two groups and one group against the rest.
	fig.bonvis_settings.node_style['feature_path'] = 'data/extra'
	run_with_vars, args, time_estimate, info = fig.preprocess_marker_genes(path, {'type': 'subtree',
	                                                                              'info': [ancestor, child]})
	assert run_with_vars == 'no' and time_estimate is None
	assert args[0] == path and args[4] == ['f1', 'f2', 'f3'] and args[5] == 'data/extra/ranks_per_gene'
	np.testing.assert_array_equal(args[6], [0, 1])
	values = np.array([extra_features(ind) for ind in range(metadata.n_cells)]).T
	table = bsh.update_marker_genes_df('no', args)
	expected = pairwise_higher_fraction(values[:, group1], values[:, group2])
	assert dict(zip(table['marker_genes'], table['marker_scores'])) == pytest.approx(
		{'f1': expected[0], 'f2': expected[1]}, abs=1e-12)
	single = args[:2] + (None,) + args[3:]
	table = bsh.update_marker_genes_df('no', single)
	assert dict(zip(table['marker_genes'], table['marker_scores'])) == pytest.approx(
		{'f1': expected[0], 'f2': expected[1]}, abs=1e-12)
	run_with_vars, args, _, _ = fig.preprocess_marker_genes(path, {'type': None})
	assert (run_with_vars, args) == ('stop', None) and len(fig.marker_genes_df) == 0


def test_legend_for_categories_and_colour_bar(scout_rich, figure_factory):
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	fig.create_legend()
	assert fig.fig_cbar is None
	styler = fig.fig_leg_df
	assert list(styler.data['Celltype']) == ['left', 'right']
	html = styler.to_html()
	for hex_color in PALETTE[:2]:
		assert mcolors.to_hex(hex_color, keep_alpha=True) in html
	fig.update_figure(node_style='Depth', verbose=False)
	fig.create_legend()
	assert fig.fig_leg_df is None
	cbar_ax = fig.fig_cbar.axes[0]
	info = fig.bonvis_settings.node_style['annot_info']
	ticks = cbar_ax.get_yticks()
	labels = [label.get_text() for label in cbar_ax.get_yticklabels()]
	assert labels == ['{:.2e}'.format(t * (info.cbar_info['vmax'] - info.cbar_info['vmin']) + info.cbar_info['vmin'])
	                  for t in ticks]


def test_create_save_remove_and_reset(scout_rich, figure_factory, tmp_path):
	folder = _rich(scout_rich)
	fig = figure_factory(folder)
	assert fig.ax.get_xlim() == (-1.05, 1.05) and fig.ax.get_ylim() == (-1.05, 1.05)
	assert fig.is_present == {'nodes': True, 'edges': True, 'bg': 'flat'}
	assert len(fig.ax.collections) == 4 and len(fig.ax.patches) == 2
	fig.save_fig(str(tmp_path / 'tree.png'), format='png')
	with open(tmp_path / 'tree.png', 'rb') as handle:
		assert handle.read(8) == b'\x89PNG\r\n\x1a\n'
	fig.remove_artists()
	assert len(fig.ax.collections) == 0 and len(fig.ax.patches) == 0
	assert fig.is_present == {'nodes': False, 'edges': False, 'bg': None}
	fig.bonvis_settings.node_style['radius_cell'] = 1.0
	reset = fig.reset_figure(os.path.join(folder, 'bonsai_vis_settings.json'))
	assert isinstance(reset, bsh.Bonvis_figure) and reset.bonvis_settings.node_style['radius_cell'] == 0.015


def test_update_clusters_matches_the_stored_clustering(scout_preprocessed, figure_factory):
	"""On a tree without merged cells, update_clusters (minimal-distance) reproduces the clustering the
	preprocessing stored for the same number of clusters, coloured with tab20."""
	run, sim, proc = scout_preprocessed
	assert proc.returncode == 0, proc.stderr[-3000:]
	fig = figure_factory(run.results_folder)
	cluster_info = pd.read_hdf(run.path('bonsai_vis_data.hdf'), key='cs_info/cluster_info_dict')
	fig.update_clusters(cluster_param=3)
	annot = fig.bonvis_settings.node_style['annot_info']
	assert (annot.info_key, annot.label, annot.annot_type) == ('annot_cl_n=3', 'Bnsi_cluster_n=3', 'cellstates')
	assert annot.cats == ['cl_0', 'cl_1', 'cl_2']
	assert annot.annot_to_color == {cat: tuple(plt.get_cmap('tab20')(ind)) for ind, cat in enumerate(annot.cats)}
	assert 'annot_cl_n=3' in fig.bonvis_settings.celltype_info.annot_alts
	assert len(fig.bonvis_metadata.clusters_cut_edges['Bnsi_cluster_n=3']) == 2
	new = fig.bonvis_metadata.cs_info['cs_info_dict']['annot_cl_n=3']
	stored = [name.split(',')[0] for name in cluster_info['annot_bnsi_cluster_n3']]
	assert list(cluster_info.index) == fig.bonvis_metadata.cs_ids
	assert new == stored
