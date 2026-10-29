"""
Shared fixtures and reference computations for area D's Bonsai-scout tests
(tests/area_d/test_bonsai_scout_*.py, test_calc_marker_genes.py).

One session job, `scout_rich`, runs Bonsai once on a small synthetic dataset with
duplicated cells (so that some cells sit on zero-length edges and Bonsai-scout
merges them into one vertex), then runs bonsai_scout_preprocess.py on copies of
the results folder with different options, and calc_marker_genes.py on the
default copy. The variants run concurrently inside the job.

Nothing here imports h5py, tables or shiny at module level: test modules skip
themselves first (pytest.importorskip) and only then import this module.
"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from bonsai_testdata import DatasetSpec, balanced_newick
from bonsai_testfixtures import DATASET_NAME, bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env, read_newick_file, run_bonsai, stage_folder, write_config

PREPROCESS = os.path.join(REPO_ROOT, 'bonsai_scout', 'bonsai_scout_preprocess.py')
CALC_MARKER_GENES = os.path.join(REPO_ROOT, 'downstream_analyses', 'calc_marker_genes.py')

# Eight leaves in two clades of four (names C0.. and C1..); the first two leaves (C000, C001) get an exact
# twin on a zero-length edge, so Bonsai's final tree has cells at distance zero from each other.
RICH_DATA = DatasetSpec(newick=balanced_newick(3, internal_length=0.5, leaf_length=0.1), n_genes=24, n_noise_genes=6,
                        seed=7, n_duplicates=2, layout='sanity2')

# Values of the data matrix annotation file mat_extra.csv: f1 and f2 have no ties, f3 is constant.
EXTRA_F2_ORDER = (7, 2, 9, 0, 5, 3, 8, 1, 6, 4)


def run_script(script, args, timeout=240, cwd=None):
	"""Run one of Bonsai's Python scripts from this checkout as a subprocess; returns the CompletedProcess."""
	return subprocess.run([sys.executable, script] + [str(arg) for arg in args], capture_output=True, text=True,
	                      env=bonsai_env(), timeout=timeout, cwd=cwd)


def clade_of(cell):
	return 'left' if cell.startswith('C0') else 'right'


def write_annotations(sim, folder):
	"""
	annotation.csv: a categorical column (the root clade), a categorical column with missing values and a
	numerical column; mat_extra.csv: a cells x features data matrix (Bonsai-scout's 'mat_' convention).
	"""
	os.makedirs(folder, exist_ok=True)
	with open(os.path.join(folder, 'annotation.csv'), 'w') as handle:
		handle.write(',celltype,partial,depth\n')
		for ind, cell in enumerate(sim.cell_ids):
			partial = ('a' if ind % 2 else 'b') if ind < 6 else ''
			handle.write('%s,%s,%s,%r\n' % (cell, clade_of(cell), partial, extra_depth(ind)))
	with open(os.path.join(folder, 'mat_extra.csv'), 'w') as handle:
		handle.write(',f1,f2,f3\n')
		for ind, cell in enumerate(sim.cell_ids):
			f1, f2, f3 = extra_features(ind)
			handle.write('%s,%r,%r,%r\n' % (cell, f1, f2, f3))
	return folder


def extra_depth(ind):
	return round(0.37 * ind, 2)


def extra_features(ind):
	return 0.25 * ind - 1.0, 1.5 * EXTRA_F2_ORDER[ind], 5.0


def cellstate_mapping(sim):
	"""Real cells per Bonsai cell (here: cellstate): 1, 2 or 3 cells named cell<cs>_<k>, in cell order."""
	rows = []
	for ind, cs in enumerate(sim.cell_ids):
		for k in range(1 + ind % 3):
			rows.append(('cell%s_%d' % (cs, k), cs))
	return rows


def cell_parity(cell):
	"""'even' or 'odd' by the cell's number within its cellstate (cell<cs>_<k>)."""
	return 'odd' if int(cell.rsplit('_', 1)[1]) % 2 else 'even'


def write_cellstate_inputs(sim, mapping_path, annotation_folder):
	rows = cellstate_mapping(sim)
	with open(mapping_path, 'w') as handle:
		for cell, cs in rows:
			handle.write('%s,%s\n' % (cell, cs))
	os.makedirs(annotation_folder, exist_ok=True)
	with open(os.path.join(annotation_folder, 'cells.csv'), 'w') as handle:
		handle.write(',celltype,score,parity\n')
		for ind, (cell, cs) in enumerate(rows):
			handle.write('%s,%s,%r,%s\n' % (cell, clade_of(cs), 0.5 * ind, cell_parity(cell)))
	return rows


# Gene-ID mapping for --new_gene_ids_file: two old IDs map to the same new name.
NEW_GENE_IDS = (('G000', 'NEWA'), ('G001', 'NEWA'), ('G002', 'NEWB'))


@dataclass
class ScoutRun:
	"""What the scout_rich job produced. variants maps a name to (folder, CompletedProcess)."""
	sim: object
	data_folder: str
	bonsai_results: str
	annotation_folder: str
	variants: dict = field(default_factory=dict)
	markers: dict = field(default_factory=dict)
	mapping: list = field(default_factory=list)

	def folder(self, variant):
		return self.variants[variant][0]

	def hdf_path(self, variant):
		return os.path.join(self.folder(variant), 'bonsai_vis_data.hdf')

	def settings_path(self, variant):
		return os.path.join(self.folder(variant), 'bonsai_vis_settings.json')

	def check(self, variant):
		folder, proc = self.variants[variant]
		assert proc.returncode == 0, '%s failed:\n%s\n%s' % (variant, proc.stdout[-3000:], proc.stderr[-3000:])
		return folder

	@property
	def final_folder(self):
		return os.path.join(self.bonsai_results, stage_folder('final'))

	def final_tree(self):
		return read_newick_file(self.final_folder)


def _copy_results(src, dst):
	shutil.copytree(src, dst)
	return dst


def _scout_rich_job(pool, workdir):
	data_folder, sim = pool.dataset(RICH_DATA)
	results = os.path.join(workdir, 'bonsai_results')
	config = write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder, results,
	                      nnn_n_randommoves=20, nnn_n_randomtrees=2, input_is_sanity_output=True)
	run_bonsai(config).check()
	annotation = write_annotations(sim, os.path.join(workdir, 'annotation'))
	run = ScoutRun(sim=sim, data_folder=data_folder, bonsai_results=results, annotation_folder=annotation)

	gene_map = os.path.join(workdir, 'new_gene_ids.csv')
	with open(gene_map, 'w') as handle:
		for old, new in NEW_GENE_IDS:
			handle.write('%s,%s\n' % (old, new))
	cs_map = os.path.join(workdir, 'cell_to_cs.csv')
	run.mapping = write_cellstate_inputs(sim, cs_map, os.path.join(workdir, 'cell_annotation'))

	variants = {}
	folder = _copy_results(results, os.path.join(workdir, 'default'))
	variants['default'] = (folder, ['--results_folder', folder, '--annotation_path', annotation])
	# The YAML file is moved out of the results folder, so the run only works if --config_filepath is used.
	folder = _copy_results(results, os.path.join(workdir, 'all_genes'))
	moved_config = os.path.join(workdir, 'all_genes_run_configs.yaml')
	shutil.move(os.path.join(folder, 'used_run_configs.yaml'), moved_config)
	variants['all_genes'] = (folder, ['--results_folder', folder, '--take_all_genes', 'True',
	                                  '--perform_annot_guided_clustering', 'False',
	                                  '--annotation_path', os.path.join(annotation, 'annotation.csv'),
	                                  '--config_filepath', moved_config])
	folder = _copy_results(results, os.path.join(workdir, 'cellstates'))
	variants['cellstates'] = (folder, ['--results_folder', folder, '--cell_id_to_cs_id_file', cs_map,
	                                   '--take_no_genes', 'True', '--new_gene_ids_file', gene_map,
	                                   '--annotation_path', os.path.join(workdir, 'cell_annotation'),
	                                   '--config_filepath', ''])
	# No YAML file in the results folder: the script treats the folder as a tree folder.
	folder = _copy_results(os.path.join(results, stage_folder('final')), os.path.join(workdir, 'tree_only'))
	variants['tree_only'] = (folder, ['--results_folder', folder])

	workers = max(1, min(3, pool.slots))
	with ThreadPoolExecutor(max_workers=workers) as executor:
		futures = {name: executor.submit(run_script, PREPROCESS, args) for name, (folder, args) in variants.items()}
		run.variants = {name: (variants[name][0], future.result()) for name, future in futures.items()}

	if run.variants['default'][1].returncode == 0:
		run.markers = _marker_runs(run, workdir, workers)
	return run


# Marker-gene groups for calc_marker_genes.py, on the posterior means and variances (data/normalized, one
# column per vertex), so only cells that are alone at their vertex are named (calc_marker_genes.py crashes on
# cells that share a vertex). The second run swaps the groups and writes to the default output path.
MARKER_GROUPS = {
	'vars': ('data/normalized', ['C100', 'C101', 'C110'], ['C010', 'C011', 'C111']),
	'vars_swapped': ('data/normalized', ['C010', 'C011', 'C111'], ['C100', 'C101', 'C110']),
}


def _marker_runs(run, workdir, workers):
	results_folder = run.folder('default')
	jobs = {}
	for name, (feature_path, group1, group2) in MARKER_GROUPS.items():
		json_path = os.path.join(workdir, 'marker_groups_%s.json' % name)
		with open(json_path, 'w') as handle:
			json.dump({'feature_path': feature_path, 'cell_ids_group1': group1, 'cell_ids_group2': group2}, handle)
		args = ['--marker_groups_json', json_path, '--results_folder', results_folder]
		if name == 'vars_swapped':
			# Default output location: <results_folder>/output_marker_script.tsv
			output = os.path.join(results_folder, 'output_marker_script.tsv')
			args += ['--settings_filename', 'bonsai_vis_settings.json']
		else:
			output = os.path.join(workdir, 'markers_%s.tsv' % name)
			args += ['--marker_output_file', output]
		jobs[name] = (output, args)
	with ThreadPoolExecutor(max_workers=workers) as executor:
		futures = {name: executor.submit(run_script, CALC_MARKER_GENES, args) for name, (output, args) in jobs.items()}
		return {name: (jobs[name][0], future.result()) for name, future in futures.items()}


scout_rich = bonsai_job_fixture('scout_rich', _scout_rich_job, slots=3)


# The pilot run: eight cells without duplicates, one categorical and one numerical annotation, default options.
SCOUT_DATA = DatasetSpec(newick=balanced_newick(3, internal_length=0.5, leaf_length=0.1), n_genes=30,
                         n_noise_genes=5, seed=4, layout='sanity2')


def _write_pilot_annotation(sim, folder):
	"""One categorical column (the clade of the root's first split) and one numerical column."""
	os.makedirs(folder, exist_ok=True)
	with open(os.path.join(folder, 'annotation.csv'), 'w') as handle:
		handle.write(',half,score\n')
		for ind, cell in enumerate(sim.cell_ids):
			handle.write('%s,%s,%r\n' % (cell, 'left' if cell.startswith('C0') else 'right', ind / 10))
	return folder


def _bonsai_then_preprocess(pool, workdir):
	data_folder, sim = pool.dataset(SCOUT_DATA)
	results = os.path.join(workdir, 'results')
	config = write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder, results,
	                      nnn_n_randommoves=20, nnn_n_randomtrees=2)
	run = run_bonsai(config).check()
	annotation = _write_pilot_annotation(sim, os.path.join(workdir, 'annotation'))
	proc = run_script(PREPROCESS, ['--results_folder', results, '--annotation_path', annotation,
	                               '--take_all_genes', 'False', '--config_filepath', ''])
	return run, sim, proc


scout_preprocessed = bonsai_job_fixture('scout_preprocessed', _bonsai_then_preprocess)


# ---------------------------------------------------------------- references

def contracted_vertices(tree):
	"""
	Bonsai-scout merges every node whose edge to its parent has length exactly zero into that parent. For a
	harness Node tree, return (rep, members): rep maps id(node) to the node that represents its merged vertex
	(the nearest ancestor-or-self reached by a non-zero edge, or the root), members maps id(rep) to the list
	of nodes in that vertex.
	"""
	parent = tree.parents()
	rep, members = {}, {}
	for node in tree.iter_nodes():
		up = node
		while parent[id(up)] is not None and up.length == 0.0:
			up = parent[id(up)]
		rep[id(node)] = up
		members.setdefault(id(up), []).append(node)
	return rep, members


def is_cell_name(name):
	return (name is not None) and (not name.startswith('internal_')) and name != 'root'


def contracted_edges(tree):
	"""Edges (parent rep, child rep, length) of the contracted tree, as a list in no particular order."""
	rep, _ = contracted_vertices(tree)
	parent = tree.parents()
	edges = []
	for node in tree.iter_nodes():
		par = parent[id(node)]
		if par is None or node.length == 0.0:
			continue
		edges.append((rep[id(par)], node, node.length))
	return edges


def tree_distances(tree):
	"""All pairwise path lengths between named nodes of a harness Node tree, as {(name_a, name_b): length}."""
	from bonsai_testtrees import path_length
	names = [node.name for node in tree.iter_nodes()]
	return {(a, b): path_length(tree, a, b) for a in names for b in names}


def gaussian_marker_score(means1, vars1, means2, vars2):
	"""
	Probability that a random cell of group 1 has a higher value than a random cell of group 2 when every
	cell's value is Gaussian with the given mean and variance: the mean over all pairs of
	Phi((m1 - m2) / sqrt(v1 + v2)). Arrays are genes x cells.
	"""
	from scipy.stats import norm
	diff = means1[:, :, None] - means2[:, None, :]
	scale = np.sqrt(vars1[:, :, None] + vars2[:, None, :])
	return norm.cdf(diff / scale).mean(axis=(1, 2))


def two_sigma_truncation(means1, vars1):
	"""
	calc_marker_genes_error_bars_approx2 integrates over [min(m1 - 2 s1), max(m1 + 2 s1)] of group 1 only; the
	probability mass of group 1 outside that range (per gene) bounds how far its score can fall below the exact
	probability.
	"""
	from scipy.stats import norm
	sd = np.sqrt(vars1)
	lower = np.min(means1 - 2 * sd, axis=1, keepdims=True)
	upper = np.max(means1 + 2 * sd, axis=1, keepdims=True)
	return np.mean(norm.cdf((lower - means1) / sd) + norm.sf((upper - means1) / sd), axis=1)


def pairwise_higher_fraction(values1, values2):
	"""Fraction of pairs (i in group 1, j in group 2) with values1[i] > values2[j]; arrays are genes x cells."""
	return (values1[:, :, None] > values2[:, None, :]).mean(axis=(1, 2))
