"""
downstream_analyses/clustering_to_maximize_NMI.py (area D): annotation-guided and minimal-distance clusterings of a
tree compared by their normalized mutual information (NMI) with an annotation. NMI is computed here from its
definition, I(clusters; labels) / sqrt(H(clusters) H(labels)) with natural logarithms, without sklearn; the
clusterings themselves are tested in test_get_clusters_max_diameter.py.
"""
import collections
import math
import os
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

import downstream_analyses.clustering_to_maximize_NMI as nmi_mod
from downstream_analyses.get_clusters_max_diameter import (get_annotation_based_clustering_from_nwk_str,
                                                           get_min_pdists_clustering_from_nwk_str)
from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env
from bonsai_testtrees import parse_newick

SCRIPT = os.path.join(REPO_ROOT, 'downstream_analyses', 'clustering_to_maximize_NMI.py')

NEWICK = ('((((A1:0.2,A2:0.3)a:0.5,(A3:0.1,B1:0.4)ab:0.6)u:0.3,((B2:0.2,B3:0.25)b:0.7,B4:0.3)bb:0.4)v:0.2,'
          '((C1:0.3,C2:0.1)c:0.8,(C3:0.2,(A4:0.1,C4:0.3)ac:0.2)cc:0.4)w:0.5)root;')
CELLS = parse_newick(NEWICK).leaf_names()
LABELS = {'A1': 'a', 'A2': 'a', 'A3': 'a', 'A4': 'a', 'B1': 'b', 'B2': 'b', 'B3': 'b', 'B4': 'b',
          'C1': 'c', 'C2': 'c', 'C3': 'c', 'C4': 'c'}


def nmi(labels_true, labels_pred):
	"""Normalized mutual information with the geometric-mean normalization, from counts."""
	n = len(labels_true)
	joint = collections.Counter(zip(labels_true, labels_pred))
	count_true, count_pred = collections.Counter(labels_true), collections.Counter(labels_pred)
	h_true = -sum(c / n * math.log(c / n) for c in count_true.values())
	h_pred = -sum(c / n * math.log(c / n) for c in count_pred.values())
	mutual = sum(c / n * math.log(c * n / (count_true[t] * count_pred[p])) for (t, p), c in joint.items())
	if h_true == 0 or h_pred == 0:
		return 1.0 if h_true == h_pred else 0.0
	return mutual / math.sqrt(h_true * h_pred)


def column_nmi(series, labels):
	cells = list(series.index)
	return nmi([labels[cell] for cell in cells], list(series.values))


def test_calculate_NMI_for_clustering_df():
	"""One NMI per column of a cells x clusterings table; a table whose cells do not match the annotation is
	refused."""
	df = pd.DataFrame({'k2': ['x', 'x', 'y', 'y', 'y', 'x'], 'k3': ['p', 'q', 'q', 'r', 'r', 'p']},
	                  index=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
	labels = {'c1': 'a', 'c2': 'a', 'c3': 'b', 'c4': 'b', 'c5': 'c', 'c6': 'a'}
	got = nmi_mod.calculate_NMI_for_clustering_df(df, labels)
	assert got == pytest.approx([column_nmi(df['k2'], labels), column_nmi(df['k3'], labels)], rel=1e-12)
	with pytest.raises(ValueError, match='do not match'):
		nmi_mod.calculate_NMI_for_clustering_df(df, {'c1': 'a'})


def test_condition_label():
	assert nmi_mod.condition_label({'random_sampling': True, 'greedy': True}) == 'First greedy'
	assert nmi_mod.condition_label({'random_sampling': True, 'greedy': False, 'optimality_decrease': 0.99}) == \
		'Random, allowed NMI-decrease=1.00%'
	assert nmi_mod.condition_label({'random_sampling': False}) == 'Final greedy'


def test_annotbased_clustering_wrapper(tmp_path):
	"""The wrapper names its column 'annot_based_clst_of_<annotation_id>', assigns cells 'cl_<k>', reports the
	clustering's NMI under that name, maps greedy=True to plain greedy cuts and writes the tracking statistics to
	<tracking_folder>/<annotation_id>_annotbased_clustering_stats.tsv."""
	for greedy in (True, False):
		cl_df, scores, tracking = nmi_mod.annotbased_clustering_wrapper(
			NEWICK, LABELS, CELLS, greedy=greedy, tracking_folder=str(tmp_path), annotation_id='celltype',
			verbose=False)
		assert list(cl_df.columns) == ['annot_based_clst_of_celltype']
		assert sorted(cl_df.index) == sorted(CELLS)
		assert list(scores) == ['annot_based_clst_of_celltype']
		assert scores['annot_based_clst_of_celltype'] == pytest.approx(
			column_nmi(cl_df['annot_based_clst_of_celltype'], LABELS), rel=1e-12)
		assert tracking == os.path.join(str(tmp_path), 'celltype_annotbased_clustering_stats.tsv')
		assert os.path.exists(tracking)
		clusters = get_annotation_based_clustering_from_nwk_str(NEWICK, LABELS, verbose=False,
		                                                         random_sampling=not greedy)[0]
		want = {cell: 'cl_%d' % ind for ind, cluster in enumerate(clusters) for cell in cluster}
		assert cl_df['annot_based_clst_of_celltype'].to_dict() == want


def test_mindist_clustering_wrapper_picks_the_best_number_of_clusters(tmp_path):
	"""The minimal-distance wrapper makes the min-pdists clusterings for 2..cells clusters, writes their NMIs to
	<tracking_folder>/<annotation_id>_mindist_clustering_stats.tsv (columns n_clusters, NMI) and returns the
	clustering with the highest NMI."""
	cl_df, scores, tracking = nmi_mod.mindist_clustering_wrapper(NEWICK, LABELS, CELLS, tracking_folder=str(tmp_path),
	                                                            annotation_id='celltype', verbose=False)
	clusterings, _ = get_min_pdists_clustering_from_nwk_str(NEWICK, n_clusters=100, cell_ids=CELLS, verbose=False)
	nmis = {}
	for name, clusters in clusterings.items():
		pred = {cell: ind for ind, cluster in enumerate(clusters) for cell in cluster}
		nmis[name] = nmi([LABELS[cell] for cell in CELLS], [pred[cell] for cell in CELLS])
	stats = pd.read_csv(tracking, sep='\t')
	assert tracking == os.path.join(str(tmp_path), 'celltype_mindist_clustering_stats.tsv')
	assert list(stats.columns) == ['n_clusters', 'NMI']
	assert list(stats['n_clusters']) == list(range(2, len(CELLS) + 1))
	np.testing.assert_allclose(stats['NMI'], [nmis['annot_bnsi_cluster_n%d' % k] for k in stats['n_clusters']],
	                           rtol=1e-12)
	best = max(nmis, key=nmis.get)
	assert list(cl_df.columns) == [best]
	assert scores == pytest.approx({'min_dist_clst_of_celltype': nmis[best]}, rel=1e-12)
	assert scores['min_dist_clst_of_celltype'] == pytest.approx(column_nmi(cl_df[best], LABELS), rel=1e-12)


def test_do_annotbased_and_mindist_clustering_writes_both(tmp_path):
	"""Both clusterings side by side in clustering_results.tsv, their NMIs in
	normalized_mutual_information_scores.tsv, and the two statistics figures when a tracking folder is given."""
	results, tracking = tmp_path / 'results', tmp_path / 'tracking'
	results.mkdir()
	tracking.mkdir()
	cl_df, scores = nmi_mod.do_annotbased_and_mindist_clustering(NEWICK, LABELS, CELLS, results_folder=str(results),
	                                                            greedy=True, tracking_folder=str(tracking),
	                                                            annotation_id='celltype', verbose=False)
	table = pd.read_csv(results / 'clustering_results.tsv', sep='\t', index_col=0)
	assert list(table.columns) == list(cl_df.columns)
	assert table.columns[0] == 'annot_based_clst_of_celltype'
	written = pd.read_csv(results / 'normalized_mutual_information_scores.tsv', sep='\t')
	assert list(written.columns) == ['clustering_method', 'max NMI', 'times']
	assert list(written['clustering_method']) == ['annot_based_clst_of_celltype', 'min_dist_clst_of_celltype']
	for row, column in enumerate(table.columns):
		assert written['max NMI'][row] == pytest.approx(column_nmi(table[column], LABELS), rel=1e-12)
	assert (written['times'] >= 0).all()
	assert list(scores['clustering_method']) == list(written['clustering_method'])
	for name in ('annotbased_clustering_stats.png', 'mindist_clustering_stats.png'):
		with open(tracking / name, 'rb') as handle:
			assert handle.read(8) == b'\x89PNG\r\n\x1a\n'


def test_statistics_figures_plot_the_tracking_files(tmp_path):
	"""The annotation-based figure has the NMI (top) and number of clusters (bottom) against n_moves, one line per
	phase of the annealing; the min-dist figure has NMI against the number of clusters, titled by the file name."""
	stats = pd.DataFrame({'n_moves': [0, 1, 2, 3, 4], 'norm_mut_info': [0.2, 0.4, 0.5, 0.55, 0.6],
	                      'n_clusters': [2, 3, 4, 4, 5], 'random_sampling': [True, True, True, False, False],
	                      'greedy': [True, False, False, None, None],
	                      'optimality_decrease': [0.95, 0.99, 0.99, None, None]})
	path = tmp_path / 'annot_stats.tsv'
	stats.to_csv(path, sep='\t', index=False)
	nmi_mod.make_annotbased_stats_figure(str(path), str(tmp_path / 'annot.png'))
	fig = matplotlib.pyplot.gcf()
	top, bottom = fig.axes
	lines = {line.get_label(): (list(line.get_xdata()), list(line.get_ydata())) for line in top.get_lines()}
	assert lines == {'First greedy': ([0], [0.2]), 'Random, allowed NMI-decrease=1.00%': ([1, 2], [0.4, 0.5]),
	                 'Final greedy': ([3, 4], [0.55, 0.6])}
	assert {line.get_label(): list(line.get_ydata()) for line in bottom.get_lines()} == {
		'First greedy': [2], 'Random, allowed NMI-decrease=1.00%': [3, 4], 'Final greedy': [4, 5]}
	assert top.get_title() == 'annot_stats.tsv'
	assert (tmp_path / 'annot.png').exists()
	matplotlib.pyplot.close('all')
	path = tmp_path / 'mindist_stats.tsv'
	pd.DataFrame({'n_clusters': [2, 3, 4], 'NMI': [0.1, 0.3, 0.2]}).to_csv(path, sep='\t', index=False)
	nmi_mod.make_mindist_stats_figure(str(path), str(tmp_path / 'mindist.png'))
	ax = matplotlib.pyplot.gcf().axes[0]
	(line,) = ax.get_lines()
	assert list(line.get_xdata()) == [2, 3, 4] and list(line.get_ydata()) == [0.1, 0.3, 0.2]
	assert ax.get_title() == 'mindist_stats.tsv'
	assert (tmp_path / 'mindist.png').exists()
	matplotlib.pyplot.close('all')


# ------------------------------------------------------------------------------------------ the script

def _write_inputs(workdir, sep):
	with open(os.path.join(workdir, 'tree.nwk'), 'w') as handle:
		handle.write(NEWICK + '\n')
	with open(os.path.join(workdir, 'cells.txt'), 'w') as handle:
		handle.write('\n'.join(CELLS) + '\n')
	ext = 'tsv' if sep == '\t' else 'csv'
	path = os.path.join(workdir, 'annotation.' + ext)
	rng = np.random.default_rng(0)
	with open(path, 'w') as handle:
		handle.write(sep.join(['cell_id', 'other', 'celltype']) + '\n')
		for cell in CELLS:
			handle.write(sep.join([cell, 'o%d' % rng.integers(3), LABELS[cell]]) + '\n')
	return path


def _script(workdir, sep, extra):
	annotation = _write_inputs(workdir, sep)
	results = os.path.join(workdir, 'results')
	proc = subprocess.run([sys.executable, SCRIPT, '--nwk_file', os.path.join(workdir, 'tree.nwk'),
	                       '--annotation_file', annotation, '--annotation_id', 'celltype',
	                       '--cell_names_file', os.path.join(workdir, 'cells.txt'), '--results_folder', results] + extra,
	                      capture_output=True, text=True, env=bonsai_env(), cwd=workdir, timeout=240)
	return proc, results


d_nmi_script_greedy = bonsai_job_fixture('d_nmi_script_greedy', lambda pool, workdir: _script(
	workdir, ',', ['--greedy', 'True', '--cutting_tol', '2.0', '--verbose', 'False']))
d_nmi_script_annealed = bonsai_job_fixture('d_nmi_script_annealed', lambda pool, workdir: _script(
	workdir, '\t', ['--greedy', 'False', '--prohibit_small_clsts', 'True']))


def _check_script(proc, results, greedy, prohibit, cutting_tol):
	assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
	assert os.path.isdir(os.path.join(results, 'clustering_stats'))
	table = pd.read_csv(os.path.join(results, 'clustering_results.tsv'), sep='\t', index_col=0)
	scores = pd.read_csv(os.path.join(results, 'normalized_mutual_information_scores.tsv'), sep='\t')
	assert list(scores['clustering_method']) == ['annot_based_clst_of_celltype', 'min_dist_clst_of_celltype']
	for row, column in enumerate(table.columns):
		assert scores['max NMI'][row] == pytest.approx(column_nmi(table[column], LABELS), rel=1e-12)
	clusters = get_annotation_based_clustering_from_nwk_str(NEWICK, LABELS, verbose=False, random_sampling=not greedy,
	                                                         prohibit_small_clsts=prohibit, cutting_tol=cutting_tol)[0]
	want = {cell: 'cl_%d' % ind for ind, cluster in enumerate(clusters) for cell in cluster}
	assert table['annot_based_clst_of_celltype'].to_dict() == want
	return table, scores


def test_script_greedy_with_csv_annotation(d_nmi_script_greedy):
	"""--greedy True and --cutting_tol 2.0 with a .csv annotation: the annotation-guided column is the plain greedy
	clustering of the --annotation_id column; no cut can raise the NMI by 2, so it is one cluster with NMI 0."""
	table, scores = _check_script(*d_nmi_script_greedy, greedy=True, prohibit=False, cutting_tol=2.0)
	assert set(table['annot_based_clst_of_celltype']) == {'cl_0'}
	assert scores['max NMI'][0] == 0.0


def test_script_annealed_with_tsv_annotation(d_nmi_script_annealed):
	"""A .tsv annotation with --greedy False (the annealing scheme) and --prohibit_small_clsts True: the
	annotation-guided column is that clustering, and both NMIs in the scores file are those of the written
	clusterings."""
	table, scores = _check_script(*d_nmi_script_annealed, greedy=False, prohibit=True, cutting_tol=1e-4)
	assert len(set(table['annot_based_clst_of_celltype'])) > 1
