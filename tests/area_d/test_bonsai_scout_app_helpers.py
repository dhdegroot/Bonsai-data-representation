"""
bonsai_scout/bonsai_scout_app_helpers.py (area D): the per-session object of the Bonsai-scout app
(BonvisObject), its cache (BonvisObjects), the settings snapshot, the feature table and a small UI piece.
Needs the Bonsai-scout environment (h5py, tables, natsort, shiny); skipped elsewhere.

The app's server function (bonsai_scout_app.py) is exercised by the playwright tests in bonsai_scout/tests,
which stay outside testpaths. Expected values here come from the preprocessed HDF file of the `scout_rich`
run (helpers_d.py) and from the tree stored in it.
"""
import json
import os
import re

import numpy as np
import pytest

pytestmark = pytest.mark.scout
h5py = pytest.importorskip('h5py')
pytest.importorskip('tables')
pytest.importorskip('natsort')
pytest.importorskip('shiny')

import pandas as pd  # noqa: E402

from bonsai_scout import bonsai_scout_app_helpers as app_helpers  # noqa: E402
from bonsai_scout.bonsai_scout_helpers import Bonvis_settings, _json_safe  # noqa: E402
from bonsai_testtrees import parse_newick  # noqa: E402
from area_d.helpers_d import scout_preprocessed, scout_rich  # noqa: E402,F401


@pytest.fixture
def scout_env(scout_rich, monkeypatch):
	"""BONSAI_DATA_PATH and BONSAI_SETTINGS_PATH pointing at the default preprocessed run, as
	run_bonsai_scout_app.py sets them."""
	scout_rich.check('default')
	monkeypatch.setenv('BONSAI_DATA_PATH', scout_rich.hdf_path('default'))
	monkeypatch.setenv('BONSAI_SETTINGS_PATH', scout_rich.settings_path('default'))
	yield scout_rich
	import matplotlib.pyplot as plt
	plt.close('all')


def test_store_current_settings_writes_a_time_stamped_copy(scout_rich, tmp_path):
	scout_rich.check('default')
	settings = Bonvis_settings(load_settings_path=scout_rich.settings_path('default'))
	app_helpers.store_current_settings(settings, str(tmp_path / 'bonsai_vis_settings.json'))
	written = os.listdir(tmp_path)
	assert len(written) == 1 and re.fullmatch(r'bonsai_vis_settings_\d{6}_\d{6}\.json', written[0])
	with open(tmp_path / written[0]) as handle:
		assert json.load(handle) == json.loads(json.dumps(_json_safe(settings.to_dict())))


def test_small_cluster_grouping_ui():
	html = str(app_helpers.small_cluster_grouping_ui())
	slider = re.search(r'<input [^>]*id="small_cluster_cutoff_general"[^>]*>', html).group(0)
	for attribute in ('data-min="1"', 'data-max="20"', 'data-from="1"', 'data-step="1"'):
		assert attribute in slider
	assert 'Group clusters smaller than:' in html
	button = re.search(r'<button [^>]*id="go_group_small_clusters"[^>]*>', html).group(0)
	assert 'btn-success' in button and 'action-button' in button
	assert 'Group small clusters!' in html


def test_feature_info_display(tmp_path):
	"""The feature table lists the IDs with their z-scores (else their variances), largest first, rounded to two
	decimals; values all below 0.01 are scaled by a power of ten that the header names."""
	path = tmp_path / 'features.hdf'
	with h5py.File(path, 'w') as hdf:
		for name, key, values in (('z', 'zscores', [0.5, 2.0, 1.256]), ('ztiny', 'zscores', [0.001, 0.004]),
		                          ('v', 'variances', [3.0, 1.0]), ('vtiny', 'variances', [0.0002, 0.00051]),
		                          ('plain', None, [0, 0])):
			group = hdf.create_group('data/' + name)
			group.attrs['gene_ids'] = json.dumps(['g%d' % ind for ind in range(len(values))])
			if key is not None:
				group.create_dataset(key, data=np.array(values))
	with h5py.File(path, 'r') as hdf:
		assert app_helpers.get_feature_info_display(hdf, None) is None
		assert app_helpers.get_feature_info_display(hdf, '') is None
		table = app_helpers.get_feature_info_display(hdf, 'data/z')
		assert list(table.columns) == ['ids', 'zscores']
		assert table.values.tolist() == [['g1', 2.0], ['g2', 1.26], ['g0', 0.5]]
		table = app_helpers.get_feature_info_display(hdf, 'data/ztiny')
		assert list(table.columns) == ['ids', 'zscores (*10^(-3)']
		assert table.values.tolist() == [['g1', 4.0], ['g0', 1.0]]
		table = app_helpers.get_feature_info_display(hdf, 'data/v')
		assert table.values.tolist() == [['g0', 3.0], ['g1', 1.0]] and list(table.columns) == ['ids', 'variances']
		table = app_helpers.get_feature_info_display(hdf, 'data/vtiny')
		assert list(table.columns) == ['ids', 'variances (*10^(-4)']
		assert table.values.tolist() == [['g1', 5.1], ['g0', 2.0]]
		table = app_helpers.get_feature_info_display(hdf, 'data/plain')
		assert table.values.tolist() == [['g0'], ['g1']]


def test_bonvis_object_holds_the_session_state(scout_env):
	"""BonvisObject loads the data and settings that BONSAI_DATA_PATH and BONSAI_SETTINGS_PATH point at and
	derives what the app's menus show: annotation labels, the annotations with a guided clustering, the
	numerical annotations usable for sizes (not per cell or cellstate when several share a vertex), layout
	names, feature groups, the initial view, the largest stored number of minimal-distance clusters and the
	longest and shortest root-to-leaf path."""
	run = scout_env
	obj = app_helpers.BonvisObject(('user', ''))
	assert obj.shiny_paths == {'data_path': run.hdf_path('default'), 'settings_path': run.settings_path('default')}
	assert obj.annotation_dict == {label: label for label in ('Default', 'Celltype', 'Partial', 'Depth',
	                                                           'Categorized_depth', 'Number of cells',
	                                                           'Default sizing')}
	assert obj.clustering_annotation_dict == {'Celltype': 'Celltype', 'Partial': 'Partial'}
	assert obj.size_annotation_dict == {'Number of cells': 'Number of cells', 'Default sizing': 'Default sizing'}
	assert obj.layout_types_dict == {'ly_dendrogram_ladderized': 'Dendrogram (ladderized)',
	                                 'ly_eq_angle': 'Equal angle', 'ly_eq_daylight': 'Equal daylight'}
	assert obj.feature_dict == {'data/normalized': 'normalized', 'data/extra': 'extra'}
	assert (obj.init_geometry, obj.init_layout, obj.init_node_style, obj.init_size_style) == (
		'flat', 'ly_dendrogram_ladderized', 'Celltype', 'Number of cells')
	assert (obj.init_selected_annot, obj.init_switch_mask, obj.init_options_accordion) == (
		'no_subset', False, 'Annotation')
	assert obj.init_annotation_cats == ['left', 'right'] and obj.init_categorical_annot == 'Celltype'
	assert obj.init_feature_path == 'data/normalized'
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		zscores = hdf['data/normalized/zscores'][()]
		gene_ids = json.loads(hdf['data/normalized'].attrs['gene_ids'])
		nwk = parse_newick(hdf['tree_info'].attrs['nwk_str'])
	order = np.argsort(-zscores, kind='stable')
	assert list(obj.feature_display['ids']) == [gene_ids[ind] for ind in order]
	np.testing.assert_allclose(obj.feature_display['zscores'], np.round(zscores[order], 2))
	cluster_info = pd.read_hdf(run.hdf_path('default'), key='cs_info/cluster_info_dict')
	assert obj.max_n_clusters == max(int(col[len('annot_bnsi_cluster_n'):]) for col in cluster_info.columns
	                                 if re.fullmatch(r'annot_bnsi_cluster_n\d+', col))
	depths = nwk.depths()
	leaf_depths = [depths[id(leaf)] for leaf in nwk.leaves()]
	assert obj.longest_path_from_root_to_leaf == pytest.approx(max(leaf_depths), rel=1e-12)
	assert obj.shortest_path_from_root_to_leaf == pytest.approx(min(leaf_depths), rel=1e-12)
	assert obj.is_big_dataset is False and obj.num_clusters is None and obj.flip_ids == []
	assert obj.orig_geometry == {'ly_dendrogram_ladderized': 'flat'}
	assert set(obj.click_counters.values()) == {0}
	assert obj.bonvis_fig.bonvis_settings is obj.bonvis_settings


def test_bonvis_objects_cache_one_object_per_key(scout_env, monkeypatch):
	objects = app_helpers.BonvisObjects()
	first = objects[('user', '')]
	assert objects[('user', '')] is first and list(objects.keys()) == [('user', '')]
	del objects[('user', '')]
	del objects[('missing', '')]
	assert list(objects.keys()) == []
	assert objects[('user', '')] is not first
	monkeypatch.delenv('BONSAI_DATA_PATH')
	with pytest.raises(BaseException, match='Could not find bonvis data'):
		app_helpers.BonvisObject(('user', ''))


def test_size_menu_lists_numerical_annotations_when_no_vertex_is_shared(scout_preprocessed, monkeypatch):
	"""Without merged cells, a numerical annotation can size the vertices; categorical ones never can."""
	run, sim, proc = scout_preprocessed
	assert proc.returncode == 0, proc.stderr[-3000:]
	monkeypatch.setenv('BONSAI_DATA_PATH', run.path('bonsai_vis_data.hdf'))
	monkeypatch.setenv('BONSAI_SETTINGS_PATH', run.path('bonsai_vis_settings.json'))
	obj = app_helpers.BonvisObject(('pilot', ''))
	assert obj.size_annotation_dict == {label: label for label in ('Score', 'Number of cells', 'Default sizing')}
	assert {'Half', 'Categorized_score'} <= set(obj.annotation_dict)
	import matplotlib.pyplot as plt
	plt.close('all')
