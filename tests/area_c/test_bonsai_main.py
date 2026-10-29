"""
End-to-end runs of bonsai/bonsai_main.py (area C): every input layout, --step, --pickup_intermediate and the
checkpoints it reads, every YAML option's effect on a run, serial against mpiexec, without mpi4py, and the files a
run writes.

Expected values come from the generator's recorded truth (topology, measurements, gene variances), the dense
reference likelihood and posteriors of tests/bonsai_testref.py and the optimisers in area_c/helpers_c.py, the
README's file formats, and pairs of runs that must agree because the documentation says an option does not change
the result. Runs are session fixtures and start together on the harness pool.
"""
import csv
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from bonsai_testdata import DatasetSpec, balanced_newick, caterpillar_newick, write_features, simulate
from bonsai_testfixtures import DATASET_NAME, EXAMPLE, bonsai_job_fixture, bonsai_run_fixture, run_spec
from bonsai_testref import reference_bonsai_loglik, reference_posteriors
from bonsai_testrun import (EXAMPLE_DATA, REPO_ROOT, posteriors_by_name, processed_data_folder, read_config,
                            read_edge_info, read_metadata_bonsai, read_metadata_json, read_newick_file,
                            read_vert_info, run_bonsai, stage_folder, tree_from_folder, write_config)
from bonsai_testtrees import clades, path_length, rf_distance, splits

from area_c.helpers_c import (FINAL_FILES, METADATA_COLUMNS, STAGES, assert_same_reconstruction, copy_results,
                              final_state, group_tree, optimise_all_lengths, optimise_common_scale,
                              remove_stages_after, rename_leaves, run_script, star, to_bonsai_units,
                              write_tree_folder)

# Fewer NNI moves than the defaults (1000 moves x 10 trees) keep each run to a few seconds. nnnReorderRandom raises
# nnn_n_randomtrees to the rank count when it is smaller, and the random trees' seeds depend on that number; with
# nnn_n_randomtrees >= ranks the work is the same at every rank count, so the serial runs use 4 random trees too.
BASE = dict(nnn_n_randommoves=20, nnn_n_randomtrees=4)
FAST = dict(nnn_n_randommoves=20, nnn_n_randomtrees=2)
FEATURES_OPTIONS = dict(input_is_sanity_output=False, filenames_data='features.txt,standard_deviations.txt')

# 16 cells on a balanced tree, 60 signal genes and 20 noise genes, Sanity 2.0 output made with -v_m MAP.
SANITY2 = DatasetSpec(newick=balanced_newick(4, internal_length=0.5, leaf_length=0.1), n_genes=60,
                      n_noise_genes=20, noise_sd=0.1, seed=0, layout='sanity2')
# 8 cells on a ladder, non-Sanity input (features.txt, standard_deviations.txt).
FEATURES = DatasetSpec(newick=caterpillar_newick(8, internal_length=0.5, leaf_length=0.1), n_genes=60,
                       noise_sd=0.1, seed=1, layout='features')
# SANITY2 with exact twins of the first two cells (C0000d, C0001d).
DUPLICATES = replace(SANITY2, n_duplicates=2)


def _kept_genes(sim, folder):
	"""Indices (into the generator's genes) of the genes a final folder kept, in Bonsai's order."""
	return [sim.gene_ids.index(gene) for gene in read_metadata_json(folder)['geneIds']]


def _assert_loglik_is_the_reference(folder, means, variances, rtol=1e-9):
	"""finalLoglik (metadata_bonsai.txt, no loglikVarCorr) is the dense likelihood of tree.nwk in Bonsai's units."""
	tree = read_newick_file(folder)
	expected = reference_bonsai_loglik(tree, means, variances)
	assert float(read_metadata_bonsai(folder)['finalLoglik']) == pytest.approx(expected, rel=rtol)


def _assert_posteriors_are_the_reference(folder, means, variances, scale, rtol=1e-8):
	"""
	The posterior files are the reference posteriors of tree.nwk (Bonsai's rescaled units) multiplied back by
	sqrt(scale) (means) and scale (variances), scale being the per-gene variance or their mean.
	"""
	tree = read_newick_file(folder)
	ref_means, ref_vars = reference_posteriors(tree, means, variances)
	files = posteriors_by_name(folder)
	assert set(files[0]) == set(ref_means)
	for name in ref_means:
		np.testing.assert_allclose(files[0][name], ref_means[name] * np.sqrt(scale), rtol=rtol, atol=1e-10)
		np.testing.assert_allclose(files[1][name], ref_vars[name] * scale, rtol=rtol, atol=1e-14)


def _snapshot(folder):
	"""Every file below folder with its size, modification time and contents."""
	out = {}
	for dirpath, _, filenames in os.walk(folder):
		for filename in filenames:
			path = os.path.join(dirpath, filename)
			stat = os.stat(path)
			with open(path, 'rb') as handle:
				out[os.path.relpath(path, folder)] = (stat.st_size, stat.st_mtime_ns, handle.read())
	return out


def _tree_folders(results_folder):
	return sorted(name for name in os.listdir(results_folder)
	              if name.startswith(('preprocessed', 'intermediate_bonsai', 'final_bonsai')))


def _cells_by_bit(cells, position):
	"""The two halves of the 16 cells C<b0><b1><b2><b3> split by bit `position`."""
	return [[cell for cell in cells if cell[1 + position] == '0'], [cell for cell in cells if cell[1 + position] == '1']]


def _has_split(tree, group):
	everything = frozenset(tree.leaf_names())
	found = splits(tree, min_length=-1)
	return frozenset(group) in found or (everything - frozenset(group)) in found


# =================================================================== runs
#
# The pool starts fixtures in the order the tests first ask for them, and a fixture that needs several slots
# waits until that many are free at once, so the MPI runs are declared and used first; every other job takes
# one slot and runs its scenarios one after another.

sanity2_mpi2_run = bonsai_run_fixture('sanity2_mpi2_run', run_spec(SANITY2, label='sanity2_mpi2', mpi_ranks=2, **BASE))
sanity2_mpi4_run = bonsai_run_fixture('sanity2_mpi4_run', run_spec(SANITY2, label='sanity2_mpi4', mpi_ranks=4, **BASE))
c_mpi4_one_tree_run = bonsai_run_fixture('c_mpi4_one_tree_run', run_spec(
	SANITY2, label='mpi4_one_tree', mpi_ranks=4, nnn_n_randommoves=20, nnn_n_randomtrees=1))
c_features_mpi2_run = bonsai_run_fixture('c_features_mpi2_run', run_spec(
	FEATURES, label='features_mpi2', mpi_ranks=2, **FEATURES_OPTIONS, **FAST))
sanity2_run = bonsai_run_fixture('sanity2_run', run_spec(SANITY2, label='sanity2', **BASE))
features_run = bonsai_run_fixture('features_run', run_spec(FEATURES, label='features', **FEATURES_OPTIONS, **FAST))
example_run = bonsai_run_fixture('example_run', run_spec(EXAMPLE, label='example', **FAST))
c_mle_preprocess_run = bonsai_run_fixture('c_mle_preprocess_run', run_spec(
	replace(SANITY2, sanity_method='MLE'), steps=('preprocess',), label='mle', **BASE))
c_eap_preprocess_run = bonsai_run_fixture('c_eap_preprocess_run', run_spec(
	replace(SANITY2, sanity_method='EAP'), steps=('preprocess',), label='eap', **BASE))
c_legacy_run = bonsai_run_fixture('c_legacy_run', run_spec(
	replace(SANITY2, layout='sanity_legacy'), label='legacy', **BASE))
c_marg_run = bonsai_run_fixture('c_marg_run', run_spec(replace(SANITY2, sanity_method='MARG'), label='marg', **BASE))
c_old_sanity_run = bonsai_run_fixture('c_old_sanity_run', run_spec(
	replace(SANITY2, sanity_version='1.9.0'), label='old_sanity', **BASE))
c_features_nostd_run = bonsai_run_fixture('c_features_nostd_run', run_spec(
	replace(FEATURES, with_stds=False), label='features_nostd', input_is_sanity_output=False,
	filenames_data='features.txt', **FAST))
c_features_missing_std_run = bonsai_run_fixture('c_features_missing_std_run', run_spec(
	replace(FEATURES, with_stds=False), steps=('preprocess',), label='features_missing_std', **FEATURES_OPTIONS,
	**FAST))
c_stepwise_run = bonsai_run_fixture('c_stepwise_run', run_spec(
	SANITY2, steps=('preprocess', 'core_calc', 'metadata'), label='stepwise', **BASE))
c_core_calc_alone_run = bonsai_run_fixture('c_core_calc_alone_run', run_spec(
	SANITY2, steps=('core_calc',), label='core_calc_alone', **BASE))
c_metadata_alone_run = bonsai_run_fixture('c_metadata_alone_run', run_spec(
	SANITY2, steps=('metadata',), label='metadata_alone', **BASE))
c_ub_off_run = bonsai_run_fixture('c_ub_off_run', run_spec(SANITY2, label='ub_off', UB_ellipsoid_size=-1.0, **BASE))
c_knn_all_run = bonsai_run_fixture('c_knn_all_run', run_spec(SANITY2, label='knn_all', use_knn=-1, **BASE))
c_spr_long_branch_run = bonsai_run_fixture('c_spr_long_branch_run', run_spec(
	SANITY2, label='spr_lbs', spr_strategy='long_branch_sorted', **BASE))
c_spr_super_sure_run = bonsai_run_fixture('c_spr_super_sure_run', run_spec(
	SANITY2, label='spr_ss', spr_strategy='super_sure', **BASE))
c_no_rescale_run = bonsai_run_fixture('c_no_rescale_run', run_spec(
	SANITY2, label='no_rescale', rescale_by_var=False, **BASE))
c_all_genes_run = bonsai_run_fixture('c_all_genes_run', run_spec(
	SANITY2, label='all_genes', zscore_cutoff=-1.0, **BASE))
c_duplicates_run = bonsai_run_fixture('c_duplicates_run', run_spec(DUPLICATES, label='duplicates', **BASE))
c_zero_moves_no_trees_run = bonsai_run_fixture('c_zero_moves_no_trees_run', run_spec(
	SANITY2, label='zero_moves_0', nnn_n_randommoves=0, nnn_n_randomtrees=0))
c_zero_moves_three_trees_run = bonsai_run_fixture('c_zero_moves_three_trees_run', run_spec(
	SANITY2, label='zero_moves_3', nnn_n_randommoves=0, nnn_n_randomtrees=3))


def _base_config(workdir, data_folder, **options):
	values = dict(BASE, input_is_sanity_output=True)
	values.update(options)
	os.makedirs(workdir, exist_ok=True)
	return write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder,
	                    os.path.join(workdir, 'results'), **values)


def _relative_to_repository(path):
	"""The path relative to the repository root, or the absolute path when there is none (another drive)."""
	try:
		return os.path.relpath(path, REPO_ROOT)
	except ValueError:
		return path


# An mpi4py package that fails to import, put first on PYTHONPATH: Bonsai then sees no mpi4py.
MPI4PY_BLOCKER = 'raise ImportError("mpi4py hidden by the area C tests")\n'


def _no_effect_job(pool, workdir):
	"""
	SANITY2 with BASE options plus everything that must leave the reconstruction unchanged: verbose False,
	UB_ellipsoid_size 3, every deprecated skip_* key set to true, a 'step: preprocess' key with no --step on the
	command line, data_folder, results_folder and --config_filepath relative to the repository root,
	--store_all_nwk_folder, --print_annotations, and mpi4py hidden.
	"""
	data_folder, sim = pool.dataset(SANITY2)
	results = os.path.join(workdir, 'results')
	config = os.path.join(workdir, 'config.yaml')
	write_config(config, DATASET_NAME, _relative_to_repository(data_folder), _relative_to_repository(results),
	             input_is_sanity_output=True, verbose=False, UB_ellipsoid_size=3.0, **BASE)
	with open(config, 'a') as handle:
		for key in ('skip_greedy_merging', 'skip_redo_starry', 'skip_opt_times', 'skip_spr_moves',
		            'skip_nnn_reordering', 'skip_reorder_edges'):
			handle.write('%s: true\n' % key)
		handle.write('step: preprocess\n')
	blocker = os.path.join(workdir, 'blocker')
	os.makedirs(os.path.join(blocker, 'mpi4py'), exist_ok=True)
	with open(os.path.join(blocker, 'mpi4py', '__init__.py'), 'w') as handle:
		handle.write(MPI4PY_BLOCKER)
	nwk_folder = os.path.join(workdir, 'all_nwk')
	returncode, log = run_script('bonsai/bonsai_main.py', [
		'--config_filepath', _relative_to_repository(config), '--store_all_nwk_folder', nwk_folder,
		'--print_annotations', os.path.join(EXAMPLE_DATA, 'annotation')], os.path.join(workdir, 'bonsai.log'),
		env={'PYTHONPATH': blocker + os.pathsep + REPO_ROOT})
	return dict(returncode=returncode, log=log, results=results, data_folder=data_folder, nwk_folder=nwk_folder,
	            sim=sim)


c_no_effect_job = bonsai_job_fixture('c_no_effect_job', _no_effect_job)


def _renamed_features_job(pool, workdir):
	"""FEATURES written to my_means.tsv and my_sds.tsv, without cellID.txt and geneID.txt."""
	sim = simulate(FEATURES)
	data_folder = os.path.join(workdir, 'data')
	write_features(sim, data_folder, with_stds=True, with_ids=False, filenames=('my_means.tsv', 'my_sds.tsv'))
	config = write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder,
	                      os.path.join(workdir, 'results'), input_is_sanity_output=False,
	                      filenames_data='my_means.tsv,my_sds.tsv', **FAST)
	return run_bonsai(config)


c_renamed_features_job = bonsai_job_fixture('c_renamed_features_job', _renamed_features_job)


def _true_tree_job(pool, workdir):
	"""SANITY2 with the generator's tree written to <data_folder>/true_tree as edgeInfo.txt and vertInfo.txt."""
	source, sim = pool.dataset(SANITY2)
	data_folder = os.path.join(workdir, 'data')
	shutil.copytree(source, data_folder)
	write_tree_folder(sim.tree, os.path.join(data_folder, 'true_tree'), sim.cell_ids)
	run = run_bonsai(_base_config(workdir, data_folder))
	run.sim = sim
	return run


c_true_tree_job = bonsai_job_fixture('c_true_tree_job', _true_tree_job)


# ---------------------------------------------------- pickup and checkpoints
#
# Each job makes its own SANITY2 results folder (a full run, or --step preprocess only) with a cheaper NNI step
# (PICKUP_BASE), then runs scenarios on copies of it. A scenario returns its run, its results folder and a
# snapshot of that folder taken just before the run. Pickup from the nnnReorder and reorderedEdges folders under
# --step all is not tested: Bonsai then finishes from the preprocessed star tree and writes it as the final
# tree.

PICKUP_BASE = dict(nnn_n_randommoves=10, nnn_n_randomtrees=1)
PICKUP_STAGES = ('preprocessed', 'greedy', 'redo_starry', 'opt_times', 'spr_moves')
# (YAML pickup_intermediate, --pickup_intermediate on the command line or None when absent, whether Bonsai picks up)
PICKUP_SWITCHES = ((False, True, True), (True, False, True), (True, None, True), (False, False, False),
                   (False, None, False))


def _from_stage(stage):
	return lambda context, results: remove_stages_after(results, stage)


def _greedy_checkpoints(context, results):
	"""Checkpoints of the greedy step: greedy_tree_1 halves the cells by their third bit, greedy_tree_3 by their
	second; neither is a split of the true tree."""
	cells, metadata = context['cells'], context['prep_metadata']
	remove_stages_after(results, 'preprocessed')
	tmp_trees = os.path.join(results, stage_folder('greedy'), 'tmp_trees')
	write_tree_folder(group_tree(_cells_by_bit(cells, 2)), os.path.join(tmp_trees, 'greedy_tree_1'), cells, metadata)
	write_tree_folder(group_tree(_cells_by_bit(cells, 1)), os.path.join(tmp_trees, 'greedy_tree_3'), cells, metadata)


def _nni_checkpoints(context, results):
	"""Checkpoints of the NNI step: nnn_tree_2 is the sprMoves tree, nnn_tree_7 the star tree."""
	cells, metadata = context['cells'], context['prep_metadata']
	remove_stages_after(results, 'spr_moves')
	tmp = os.path.join(results, 'intermediate_trees')
	write_tree_folder(read_newick_file(os.path.join(results, stage_folder('spr_moves'))),
	                  os.path.join(tmp, 'nnn_tree_2'), cells, metadata)
	write_tree_folder(star(cells), os.path.join(tmp, 'nnn_tree_7'), cells, metadata)


def _mark_preprocessed(context, results):
	with open(os.path.join(results, stage_folder('preprocessed'), 'tree.nwk'), 'w') as handle:
		handle.write('sentinel\n')


def _start_tree(context, results):
	"""A tmp_folder tree of two polytomies of 8 cells, halved by their second bit (not a split of the true tree)."""
	write_tree_folder(group_tree(_cells_by_bit(context['cells'], 1)), context['start_folder'], context['cells'],
	                  context['prep_metadata'])


# name: (prepare(context, results), keyword arguments of the scenario run)
SCENARIOS = dict(
	[('from_' + stage, (_from_stage(stage), {})) for stage in PICKUP_STAGES] +
	[('final_present', (None, {})),
	 ('nni_checkpoints', (_nni_checkpoints, {})),
	 ('greedy_checkpoints', (_greedy_checkpoints, {})),
	 ('tmp_folder', (_start_tree, dict(pickup=None, copy=False, tmp_folder=True)))] +
	[('switch_%s_%s' % (yaml, cli), (_mark_preprocessed, dict(step='preprocess', pickup=cli, yaml_pickup=yaml)))
	 for yaml, cli, _ in PICKUP_SWITCHES])


def _pickup_job(names, base_step):
	def job(pool, workdir):
		data_folder, sim = pool.dataset(SANITY2)
		base = run_bonsai(_base_config(os.path.join(workdir, 'base'), data_folder, **PICKUP_BASE), step=base_step)
		base.check()
		base.sim = sim
		prep_metadata = read_metadata_json(os.path.join(base.results_folder, stage_folder('preprocessed')))
		context = dict(base=base, sim=sim, cells=prep_metadata['cellIds'], prep_metadata=prep_metadata,
		               start_folder=os.path.join(workdir, 'start_trees', 'my_start'), scenarios={})
		for name in names:
			prepare, options = SCENARIOS[name]
			options = dict(options)
			step, pickup = options.pop('step', 'all'), options.pop('pickup', True)
			yaml_pickup, copy = options.pop('yaml_pickup', False), options.pop('copy', True)
			if options.pop('tmp_folder', False):
				options['tmp_folder'] = context['start_folder']
			folder = os.path.join(workdir, name)
			config = _base_config(folder, data_folder, pickup_intermediate=yaml_pickup, **PICKUP_BASE, **options)
			results = os.path.join(folder, 'results')
			if copy:
				copy_results(base.results_folder, results)
			if prepare is not None:
				prepare(context, results)
			before = _snapshot(results) if os.path.isdir(results) else {}
			run = run_bonsai(config, step=step, pickup_intermediate=pickup)
			context['scenarios'][name] = dict(run=run, results=results, before=before)
		return context
	return job


c_pickup_a_job = bonsai_job_fixture('c_pickup_a_job', _pickup_job(('from_preprocessed', 'from_greedy'), 'all'))
c_pickup_b_job = bonsai_job_fixture('c_pickup_b_job', _pickup_job(('from_redo_starry', 'from_opt_times'), 'all'))
c_pickup_c_job = bonsai_job_fixture('c_pickup_c_job', _pickup_job(
	('from_spr_moves', 'final_present', 'nni_checkpoints'), 'all'))
c_pickup_d_job = bonsai_job_fixture('c_pickup_d_job', _pickup_job(('greedy_checkpoints', 'tmp_folder'), 'preprocess'))
c_pickup_e_job = bonsai_job_fixture('c_pickup_e_job', _pickup_job(
	tuple('switch_%s_%s' % (yaml, cli) for yaml, cli, _ in PICKUP_SWITCHES), 'preprocess'))
STAGE_JOB = dict(preprocessed='c_pickup_a_job', greedy='c_pickup_a_job', redo_starry='c_pickup_b_job',
                 opt_times='c_pickup_b_job', spr_moves='c_pickup_c_job')


# ===================================================================== MPI

def _assert_mpi_gives_the_serial_result(mpi_run, serial_run, ranks):
	mpi_run.check()
	assert_same_reconstruction(final_state(mpi_run.final_folder()), final_state(serial_run.final_folder()))
	assert int(read_metadata_bonsai(mpi_run.final_folder())['coresUsed']) == ranks
	assert int(read_metadata_bonsai(serial_run.final_folder())['coresUsed']) == 1
	assert set(os.listdir(mpi_run.results_folder)) == set(os.listdir(serial_run.results_folder))


@pytest.mark.mpi
def test_mpi_two_ranks_give_the_serial_result(sanity2_mpi2_run, sanity2_run):
	"""On this 16-cell dataset, with nnn_n_randomtrees >= ranks, Bonsai reaches the same tree under
	mpiexec -n 2 and -n 4 as in one process (byte-identical files on Linux; compared here with tolerances).
	On the 64-cell example it does not: see tests/README.md, "MPI rank count"."""
	_assert_mpi_gives_the_serial_result(sanity2_mpi2_run, sanity2_run, 2)


@pytest.mark.mpi
def test_mpi_four_ranks_give_the_serial_result(sanity2_mpi4_run, sanity2_run):
	_assert_mpi_gives_the_serial_result(sanity2_mpi4_run, sanity2_run, 4)


@pytest.mark.mpi
def test_mpi_features_input_gives_the_serial_result(c_features_mpi2_run, features_run):
	"""Non-Sanity input read by two ranks (genes split between them, gene variances inferred on each) gives the
	serial reconstruction."""
	c_features_mpi2_run.check()
	assert_same_reconstruction(final_state(c_features_mpi2_run.final_folder()), final_state(features_run.final_folder()))
	assert int(read_metadata_bonsai(c_features_mpi2_run.final_folder())['coresUsed']) == 2


@pytest.mark.mpi
def test_mpi_with_fewer_random_trees_than_ranks(c_mpi4_one_tree_run):
	"""nnn_n_randomtrees 1 under mpiexec -n 4 (the YAML comment suggests one random tree per core; fewer is
	allowed): the run finishes, recovers the true tree and reports the reference likelihood and posteriors of the
	tree it writes. Its result is not compared with other rank counts: Bonsai then makes one random tree per
	rank, with seeds that depend on the rank count (tests/README.md, "MPI rank count")."""
	run = c_mpi4_one_tree_run
	run.check()
	sim, final = run.sim, run.final_folder()
	assert set(os.listdir(final)) == FINAL_FILES
	assert int(read_metadata_bonsai(final)['coresUsed']) == 4
	assert rf_distance(read_newick_file(final), sim.tree) == 0
	genes = _kept_genes(sim, final)
	means, variances = sim.bonsai_units(genes)
	_assert_loglik_is_the_reference(final, means, variances)
	_assert_posteriors_are_the_reference(final, means, variances, sim.gene_variances[genes])


def test_without_mpi4py_bonsai_runs_as_one_process(c_no_effect_job):
	"""README: 'Even with a failed installation of mpi4py, Bonsai will run, but it will only make use of 1
	computing core.' The run with mpi4py hidden reports one core and gives the serial reconstruction
	(test_options_without_effect_leave_the_result_unchanged). Its warning message is not checked (log text)."""
	assert c_no_effect_job['returncode'] == 0, c_no_effect_job['log'][-3000:]
	final = os.path.join(c_no_effect_job['results'], stage_folder('final'))
	assert int(read_metadata_bonsai(final)['coresUsed']) == 1


# ====================================================== --pickup_intermediate

@pytest.mark.parametrize('stage', PICKUP_STAGES)
def test_pickup_from_each_intermediate_folder_reaches_the_same_final_folder(stage, c_pickup_a_job, c_pickup_b_job,
                                                                           c_pickup_c_job):
	"""README: with --pickup_intermediate True Bonsai 'will automatically find the latest intermediate result and
	re-start its computation from there'. The folders of the stages already done stay as they were. The picked-up
	tree is read back from edgeInfo.txt (9 significant digits) where the uninterrupted run kept it in memory, so
	the results agree to 1e-7 (relative, or 1e-8 absolute near zero; 3e-9 seen)."""
	job = dict(c_pickup_a_job=c_pickup_a_job, c_pickup_b_job=c_pickup_b_job, c_pickup_c_job=c_pickup_c_job)[
		STAGE_JOB[stage]]
	base = job['base']
	scenario = job['scenarios']['from_' + stage]
	scenario['run'].check()
	assert_same_reconstruction(final_state(os.path.join(scenario['results'], stage_folder('final'))),
	                           final_state(base.final_folder()), rtol=1e-7, atol=1e-8)
	after = _snapshot(scenario['results'])
	done = [stage_folder(name) for name in STAGES[:STAGES.index(stage) + 1]] + [processed_data_folder()]
	for path, entry in scenario['before'].items():
		if path.split(os.sep)[0] in done:
			assert after[path] == entry, path
	assert _tree_folders(scenario['results']) == _tree_folders(base.results_folder)


def test_pickup_with_the_final_folder_present_changes_nothing(c_pickup_c_job):
	scenario = c_pickup_c_job['scenarios']['final_present']
	scenario['run'].check()
	assert _snapshot(scenario['results']) == scenario['before']


@pytest.mark.parametrize('yaml_value, cli_value, picks_up', PICKUP_SWITCHES)
def test_pickup_switch_on_the_command_line_and_in_the_yaml(yaml_value, cli_value, picks_up, c_pickup_e_job):
	"""--pickup_intermediate help: 'If this argument is True, it over-rules the same argument in bonsai_config.yaml,
	and the other way around': a True in either place picks up. --step preprocess on a finished results folder whose
	preprocessed tree.nwk was overwritten: picking up loads the folder and leaves it alone, starting over rewrites it."""
	scenario = c_pickup_e_job['scenarios']['switch_%s_%s' % (yaml_value, cli_value)]
	scenario['run'].check()
	assert read_config(scenario['run'].config)['pickup_intermediate'] is yaml_value
	with open(os.path.join(scenario['results'], stage_folder('preprocessed'), 'tree.nwk')) as handle:
		text = handle.read()
	if picks_up:
		assert text == 'sentinel\n'
		assert _snapshot(scenario['results']) == scenario['before']
	else:
		assert text != 'sentinel\n'
		assert rf_distance(read_newick_file(os.path.join(scenario['results'], stage_folder('preprocessed'))),
		                   star(c_pickup_e_job['cells'])) == 0


def test_greedy_checkpoint_in_tmp_trees_is_picked_up(c_pickup_d_job):
	"""With only the preprocessed folder left, the greedy step resumes from the highest-numbered greedy_tree_<n> in
	intermediate_bonsai_*/tmp_trees. Greedy merging only joins children of one node, so the top split of
	greedy_tree_3 (cells halved by their second bit, not a split of the true tree) survives into the greedy and
	redoStarry folders, and that of the lower-numbered greedy_tree_1 (third bit) does not appear. The checkpoints
	are removed afterwards."""
	scenario = c_pickup_d_job['scenarios']['greedy_checkpoints']
	scenario['run'].check()
	cells = c_pickup_d_job['cells']
	for stage in ('greedy', 'redo_starry'):
		tree = read_newick_file(os.path.join(scenario['results'], stage_folder(stage)))
		assert sorted(tree.leaf_names()) == sorted(cells)
		assert _has_split(tree, _cells_by_bit(cells, 1)[0]), stage
		assert not _has_split(tree, _cells_by_bit(cells, 2)[0]), stage
	assert not os.path.exists(os.path.join(scenario['results'], stage_folder('greedy'), 'tmp_trees'))
	assert os.path.isdir(os.path.join(scenario['results'], stage_folder('final')))


def test_nni_checkpoint_in_intermediate_trees_is_picked_up(c_pickup_c_job):
	"""With the sprMoves folder the latest, the NNI step resumes from the highest-numbered nnn_tree_<n> in
	intermediate_trees/ instead of the sprMoves tree. nnn_tree_7 is a star, which has no internal edge to
	interchange around, so the nnnReorder and final trees have no split; nnn_tree_2 (the resolved sprMoves tree)
	is not used. intermediate_trees/ is removed afterwards."""
	scenario = c_pickup_c_job['scenarios']['nni_checkpoints']
	scenario['run'].check()
	for stage in ('nnn_reorder', 'final'):
		tree = read_newick_file(os.path.join(scenario['results'], stage_folder(stage)))
		assert sorted(tree.leaf_names()) == sorted(c_pickup_c_job['cells'])
		assert splits(tree, min_length=-1) == set(), stage
	assert not os.path.exists(os.path.join(scenario['results'], 'intermediate_trees'))
	spr_before = {path: entry for path, entry in scenario['before'].items()
	              if path.split(os.sep)[0] == stage_folder('spr_moves')}
	after = _snapshot(scenario['results'])
	assert all(after[path] == entry for path, entry in spr_before.items())


def test_tmp_folder_starts_from_the_given_tree(c_pickup_d_job):
	"""YAML tmp_folder: 'pointing to a tree-folder from which Bonsai reconstruction will start'. The given tree
	(two polytomies of 8 cells that are not a split of the true tree) is stored as the preprocessed result, greedy
	merging keeps its top split, and every folder of the run carries the suffix _tmpStart<folder name>."""
	scenario = c_pickup_d_job['scenarios']['tmp_folder']
	scenario['run'].check()
	results = scenario['results']
	suffix = '_tmpStart' + os.path.basename(c_pickup_d_job['start_folder'])
	expected_folders = sorted(stage_folder(stage) + suffix for stage in STAGES + ('final',))
	assert _tree_folders(results) == expected_folders
	given = group_tree(_cells_by_bit(c_pickup_d_job['cells'], 1))
	stored = read_newick_file(os.path.join(results, stage_folder('preprocessed') + suffix))
	assert clades(stored, include_root=True, min_length=-1) == clades(given, include_root=True, min_length=-1)
	assert {frozenset(n.leaf_names()): n.length for n in stored.iter_nodes() if n.length is not None} == \
		pytest.approx({frozenset(n.leaf_names()): n.length for n in given.iter_nodes() if n.length is not None},
		              rel=1e-12)
	for stage in ('greedy', 'redo_starry'):
		tree = read_newick_file(os.path.join(results, stage_folder(stage) + suffix))
		assert _has_split(tree, _cells_by_bit(c_pickup_d_job['cells'], 1)[0]), stage
	final = os.path.join(results, stage_folder('final') + suffix)
	assert set(os.listdir(final)) == FINAL_FILES
	sim = c_pickup_d_job['sim']
	means, variances = sim.bonsai_units(_kept_genes(sim, final))
	_assert_loglik_is_the_reference(final, means, variances)


# ============================================================ input layouts

def _example_annotation_groups():
	"""Groups of cells that share a 'Second_Cousins', 'Cousins' or 'Brothers' label in the example's annotation."""
	groups = {}
	with open(os.path.join(EXAMPLE_DATA, 'annotation', 'celltype_annotation.csv')) as handle:
		reader = csv.reader(handle)
		header = next(reader)
		for row in reader:
			for column, label in zip(header[1:], row[1:]):
				groups.setdefault((column, label), set()).add(row[0])
	return [frozenset(group) for group in groups.values()]


def test_bundled_example_runs_end_to_end(example_run):
	"""The 64-cell example of the README (pre-2.0 Sanity layout) runs through --step all, and every clade its
	annotation describes (brothers, cousins, second cousins of a 6-generation binary tree) is a split of the
	reconstructed tree."""
	example_run.check()
	final = example_run.final_folder()
	assert set(os.listdir(final)) == FINAL_FILES
	tree = read_newick_file(final)
	cell_ids = [line.strip() for line in open(os.path.join(EXAMPLE_DATA, 'cellID.txt'))]
	assert sorted(tree.leaf_names()) == sorted(cell_ids)
	everything = frozenset(cell_ids)
	found = splits(tree)
	groups = _example_annotation_groups()
	assert len(groups) == 32 + 16 + 8
	for group in groups:
		assert group in found or (everything - group) in found, sorted(group)
	means, variances = np.load(os.path.join(final, 'posterior_ltqs_vertByGene.npy')), \
		np.load(os.path.join(final, 'posterior_ltqsVars_vertByGene.npy'))
	n_verts = len(read_vert_info(final))
	assert means.shape == variances.shape == (n_verts, read_metadata_json(final)['nGenes'])
	assert np.all(np.isfinite(means)) and np.all(variances > 0)


def test_sanity2_step_all_recovers_the_true_topology(sanity2_run):
	sanity2_run.check()
	final = sanity2_run.final_folder()
	assert set(os.listdir(final)) == FINAL_FILES
	assert rf_distance(read_newick_file(final), sanity2_run.sim.tree) == 0
	assert rf_distance(tree_from_folder(final), sanity2_run.sim.tree) == 0


def test_sanity2_noise_genes_are_filtered_by_zscore(sanity2_run):
	"""The generator's noise genes sit far below the default cutoff of 1 and its signal genes far above."""
	sim = sanity2_run.sim
	metadata = read_metadata_json(sanity2_run.final_folder())
	assert metadata['geneIds'] == [sim.gene_ids[ind] for ind in sorted(sim.signal_genes)]
	assert metadata['nCells'] == sim.n_cells and metadata['cellIds'] == sim.cell_ids
	np.testing.assert_allclose(metadata['geneVariances'], sim.gene_variances[sorted(sim.signal_genes)], rtol=1e-12)


def test_sanity2_final_loglik_is_the_reference_likelihood_of_the_final_tree(sanity2_run):
	"""Sanity's prior N(0, v_g) divided out and every gene rescaled by v_g (variance.txt): the likelihood is that
	of the generator's measurements divided by sqrt(v_g)."""
	sim, final = sanity2_run.sim, sanity2_run.final_folder()
	means, variances = sim.bonsai_units(_kept_genes(sim, final))
	_assert_loglik_is_the_reference(final, means, variances)
	metadata = read_metadata_json(final)
	assert metadata['loglikVarCorr'] == pytest.approx(
		-(sim.n_cells - 1) * np.sum(np.log(sim.gene_variances[_kept_genes(sim, final)])), rel=1e-12)
	assert metadata['geneDiffusionScaling'] == 'geneVariances'


def test_sanity2_posteriors_are_the_reference_posteriors_rescaled_by_the_gene_variances(sanity2_run):
	sim, final = sanity2_run.sim, sanity2_run.final_folder()
	genes = _kept_genes(sim, final)
	means, variances = sim.bonsai_units(genes)
	_assert_posteriors_are_the_reference(final, means, variances, sim.gene_variances[genes])


def test_preprocessed_folder_holds_the_star_tree_with_optimal_branch_lengths(sanity2_run):
	"""Reconstruction starts from the star tree: every cell a child of the root, with the branch lengths that
	maximise the likelihood (the reference optimum to 1e-8 relative: Bonsai's optimiser stops at a tolerance;
	2e-10 seen)."""
	sim = sanity2_run.sim
	folder = sanity2_run.path(stage_folder('preprocessed'))
	tree = read_newick_file(folder)
	assert sorted(child.name for child in tree.children) == sorted(sim.cell_ids)
	assert all(child.is_leaf for child in tree.children)
	genes = _kept_genes(sim, folder)
	means, variances = sim.bonsai_units(genes)
	optimum = to_bonsai_units(optimise_all_lengths(tree, means, variances)[0], sim.n_cells, len(genes))
	at_bonsai = reference_bonsai_loglik(tree, means, variances)
	assert at_bonsai <= optimum + 1e-9 * abs(optimum)
	assert at_bonsai == pytest.approx(optimum, rel=1e-8)


def _features_units(run, folder):
	"""Likelihood means and variances of a features run, rescaled by the variances Bonsai inferred."""
	sim = run.sim
	genes = _kept_genes(sim, folder)
	scale = np.asarray(read_metadata_json(folder)['geneVariances'])
	means = {cell: sim.measured[genes, ind] / np.sqrt(scale) for ind, cell in enumerate(sim.cell_ids)}
	variances = {cell: sim.measured_sd[genes, ind] ** 2 / scale for ind, cell in enumerate(sim.cell_ids)}
	return means, variances


@pytest.mark.parametrize('name', ['sanity2', 'features', 'all_genes'])
def test_final_branch_lengths_maximise_the_likelihood(name, sanity2_run, features_run, c_all_genes_run):
	"""The core steps end by optimising every branch length and trying zero for the short ones, so the final lengths
	sit at the likelihood maximum of the final topology: the reference optimum, searched from the final lengths, is
	above finalLoglik by at most 1e-6 relative. Bonsai's optimiser stops once an iteration gains less than its
	tolerance (tol=1e-6); 2.4e-7 seen. Bonsai does not reach this on every run: on some inputs a length stays
	near its 1e-4 floor, short of the maximum."""
	run = dict(sanity2=sanity2_run, features=features_run, all_genes=c_all_genes_run)[name]
	run.check()
	folder = run.final_folder(-1.0 if name == 'all_genes' else 1.0)
	if name == 'features':
		means, variances = _features_units(run, folder)
	else:
		means, variances = run.sim.bonsai_units(_kept_genes(run.sim, folder))
	tree = read_newick_file(folder)
	n_genes = len(next(iter(means.values())))
	start = [max(node.length, 1e-4) for node in tree.iter_nodes() if node is not tree]
	optimum = to_bonsai_units(optimise_all_lengths(tree, means, variances, start=start)[0], len(tree.leaves()), n_genes)
	final = float(read_metadata_bonsai(folder)['finalLoglik'])
	assert final <= optimum + 1e-9 * abs(optimum)
	assert final >= optimum - 1e-6 * abs(optimum)


def test_newick_and_edge_vert_files_describe_the_same_tree(sanity2_run):
	final = sanity2_run.final_folder()
	from_newick = read_newick_file(final)
	from_files = tree_from_folder(final)
	assert clades(from_newick, include_root=True) == clades(from_files, include_root=True)
	# edgeInfo.txt rounds lengths to 9 significant digits (%.8e); tree.nwk keeps them in full.
	lengths_nwk = {node.name: node.length for node in from_newick.iter_nodes() if node.length is not None}
	for node in from_files.iter_nodes():
		if node.length is not None:
			assert node.length == pytest.approx(lengths_nwk[node.name], rel=1e-8, abs=1e-300)
	vert_names = [name for _, _, name in read_vert_info(final)]
	assert len(vert_names) == len(set(vert_names)) == len(read_edge_info(final)) + 1


@pytest.mark.parametrize('method', ['MLE', 'EAP'])
def test_sanity_mle_and_eap_output_is_preprocessed_like_map_output(method, c_mle_preprocess_run,
                                                                   c_eap_preprocess_run, sanity2_run):
	"""The method header of sanity_command.txt only decides whether Sanity output is accepted: --step preprocess on
	MLE or EAP output with the same numbers writes the preprocessed tree and processed data of the MAP run, which is
	all the later steps read, and nothing else: no intermediate or final folder."""
	run = c_mle_preprocess_run if method == 'MLE' else c_eap_preprocess_run
	run.check()
	names = set(os.listdir(run.results_folder))
	assert names == {stage_folder('preprocessed'), processed_data_folder(), 'used_run_configs.yaml'}
	for folder, files in ((stage_folder('preprocessed'), ('ltqs_vertByGene.npy', 'ltqsVars_vertByGene.npy')),
	                      (processed_data_folder(), ('delta.npy', 'delta_vars.npy'))):
		for filename in files:
			np.testing.assert_allclose(np.load(run.path(folder, filename)), np.load(sanity2_run.path(folder, filename)),
			                           rtol=1e-12, atol=0)
	assert read_metadata_json(run.path(processed_data_folder()))['geneIds'] == \
		read_metadata_json(sanity2_run.path(processed_data_folder()))['geneIds']
	tree = read_newick_file(run.path(stage_folder('preprocessed')))
	map_tree = read_newick_file(sanity2_run.path(stage_folder('preprocessed')))
	assert len(tree.children) == 16 and all(child.is_leaf for child in tree.children)
	assert {c.name: c.length for c in tree.children} == pytest.approx({c.name: c.length for c in map_tree.children},
	                                                                   rel=1e-12)


def test_pre2_sanity_layout_gives_the_sanity2_result(c_legacy_run, sanity2_run):
	"""The same numbers in the pre-2.0 '_vmax' files (no sanity_command.txt) give the same reconstruction."""
	c_legacy_run.check()
	assert_same_reconstruction(final_state(c_legacy_run.final_folder()), final_state(sanity2_run.final_folder()))


def _assert_rejected(run):
	"""An input error: no preprocessed, processed-data, intermediate or final folder is written."""
	assert not run.timed_out
	if os.path.isdir(run.results_folder):
		assert _tree_folders(run.results_folder) == []
		assert not os.path.exists(run.path(processed_data_folder()))


def test_sanity_marg_output_is_rejected(c_marg_run):
	"""README: MARG output 'cannot be used as Bonsai-input ... Bonsai recognises such output and stops with an
	explanatory message'."""
	_assert_rejected(c_marg_run)
	assert 'MARG' in c_marg_run.log and 'MAP' in c_marg_run.log


def test_sanity_older_than_2_with_a_command_file_is_rejected(c_old_sanity_run):
	"""A sanity_command.txt naming version 1.9.0 stops the run; the message names the version found and the minimum."""
	_assert_rejected(c_old_sanity_run)
	assert '1.9.0' in c_old_sanity_run.log and '2.0' in c_old_sanity_run.log


def test_features_input_recovers_the_true_topology(features_run):
	features_run.check()
	final = features_run.final_folder()
	assert set(os.listdir(final)) == FINAL_FILES
	assert rf_distance(read_newick_file(final), features_run.sim.tree) == 0


def test_features_input_loglik_and_posteriors_are_the_reference(features_run):
	"""Non-Sanity input is used as the likelihood as given (no prior divided out), each feature rescaled by the
	variance Bonsai infers for it (metadata.json geneVariances)."""
	final = features_run.final_folder()
	scale = np.asarray(read_metadata_json(final)['geneVariances'])
	means, variances = _features_units(features_run, final)
	_assert_loglik_is_the_reference(final, means, variances)
	_assert_posteriors_are_the_reference(final, means, variances, scale)


def test_features_without_standard_deviations_assume_tiny_error_bars(c_features_nostd_run):
	"""README: 'If standard_deviations.txt is not provided, Bonsai will assume that the uncertainty on the feature
	values is very small.' The tree is found, and every cell's posterior sits on its measurement with a variance far
	below the feature's spread."""
	c_features_nostd_run.check()
	sim, final = c_features_nostd_run.sim, c_features_nostd_run.final_folder()
	assert rf_distance(read_newick_file(final), sim.tree) == 0
	means, variances = posteriors_by_name(final)
	genes = _kept_genes(sim, final)
	spread = np.var(sim.measured[genes], axis=1)
	for ind, cell in enumerate(sim.cell_ids):
		np.testing.assert_allclose(means[cell], sim.measured[genes, ind], rtol=0, atol=1e-4 * np.sqrt(spread).max())
		assert np.all(variances[cell] < 1e-4 * spread)


def test_features_with_a_missing_standard_deviations_file_are_read_like_no_file(c_features_missing_std_run,
                                                                               c_features_nostd_run):
	c_features_missing_std_run.check()
	for filename in ('delta.npy', 'delta_vars.npy'):
		np.testing.assert_allclose(np.load(c_features_missing_std_run.path(processed_data_folder(), filename)),
		                           np.load(c_features_nostd_run.path(processed_data_folder(), filename)), rtol=1e-12)


def test_features_without_ids_in_renamed_files_give_the_features_result(c_renamed_features_job, features_run):
	"""filenames_data names the two input files; without cellID.txt the cells are called Cell_<column index>. The
	reconstruction is that of the same numbers in features.txt and standard_deviations.txt with IDs."""
	c_renamed_features_job.check()
	sim = features_run.sim
	state = final_state(c_renamed_features_job.final_folder())
	metadata = read_metadata_json(c_renamed_features_job.final_folder())
	assert metadata['cellIds'] == ['Cell_%d' % ind for ind in range(sim.n_cells)]
	assert metadata['nGenes'] == read_metadata_json(features_run.final_folder())['nGenes']
	expected = final_state(features_run.final_folder())
	mapping = {cell: 'Cell_%d' % ind for ind, cell in enumerate(sim.cell_ids)}
	expected['tree'] = rename_leaves(expected['tree'], mapping)
	expected['lengths'] = {frozenset(mapping[cell] for cell in clade): length
	                       for clade, length in expected['lengths'].items()}
	expected['means'] = {mapping.get(name, name): value for name, value in expected['means'].items()}
	expected['variances'] = {mapping.get(name, name): value for name, value in expected['variances'].items()}
	assert_same_reconstruction(state, expected)


def test_filenames_data_is_ignored_for_sanity_input(sanity2_run):
	"""YAML comment of input_is_sanity_output: 'When this flag is True, filenames_data is ignored'. The SANITY2 run's
	config names features.txt and standard_deviations.txt, which its data folder does not have."""
	config = read_config(sanity2_run.config)
	assert config['input_is_sanity_output'] is True
	for filename in config['filenames_data'].split(','):
		assert not os.path.exists(os.path.join(sanity2_run.data_folder, filename))
	sanity2_run.check()


# ===================================================================== --step

def test_stepwise_run_gives_the_step_all_result(c_stepwise_run, sanity2_run):
	"""preprocess, then core_calc, then metadata (README, Example 2) end in the --step all result. The metadata
	step reads the last tree back from edgeInfo.txt, which keeps 9 significant digits (%.8e), so lengths and
	posteriors agree to 1e-7 (relative, or 1e-8 absolute near zero; 1.3e-8 relative seen) instead of to rounding."""
	c_stepwise_run.check()
	assert len(c_stepwise_run.history) == 3
	assert_same_reconstruction(final_state(c_stepwise_run.final_folder()), final_state(sanity2_run.final_folder()),
	                           rtol=1e-7, atol=1e-8)
	assert set(os.listdir(c_stepwise_run.results_folder)) == set(os.listdir(sanity2_run.results_folder))


def test_core_calc_without_a_preprocessed_folder_writes_no_tree(c_core_calc_alone_run):
	"""--step core_calc continues from the preprocess step; without its folder there is nothing to continue from."""
	assert c_core_calc_alone_run.returncode != 0 or 'rror' in c_core_calc_alone_run.log
	assert _tree_folders(c_core_calc_alone_run.results_folder) == []


def test_metadata_without_intermediate_folders_writes_no_final_folder(c_metadata_alone_run):
	assert c_metadata_alone_run.returncode != 0 or 'rror' in c_metadata_alone_run.log
	assert _tree_folders(c_metadata_alone_run.results_folder) == []


def test_missing_config_file_stops_without_output(tmp_path):
	"""--config_filepath pointing nowhere: Bonsai says where it looked and stops before reading anything."""
	missing = str(tmp_path / 'absent.yaml')
	returncode, log = run_script('bonsai/bonsai_main.py', ['--config_filepath', missing], str(tmp_path / 'bonsai.log'))
	assert missing in log
	assert 'Traceback' not in log
	assert os.listdir(tmp_path) == ['bonsai.log']


# ================================================== YAML options, one by one

def test_options_without_effect_leave_the_result_unchanged(c_no_effect_job, sanity2_run):
	"""verbose False, UB_ellipsoid_size 3 (YAML comment: 'does not affect the final result'), the deprecated skip_*
	keys (Run_Configs: 'no longer have any effect'), a YAML 'step' key (--step decides, default all), paths relative
	to the repository root (README: 'either relative to the "Bonsai-data-representation"-directory, or absolute'),
	--store_all_nwk_folder, --print_annotations and no mpi4py: the same reconstruction as SANITY2 with absolute
	paths and default settings."""
	job = c_no_effect_job
	assert job['returncode'] == 0, job['log'][-3000:]
	final = os.path.join(job['results'], stage_folder('final'))
	assert_same_reconstruction(final_state(final), final_state(sanity2_run.final_folder()))
	assert set(os.listdir(job['results'])) == set(os.listdir(sanity2_run.results_folder))
	metadata = read_metadata_json(final)
	assert os.path.samefile(metadata['pathToOrigData'], job['data_folder'])
	assert os.path.samefile(metadata['results_folder'], job['results'])
	assert metadata['dataset'] == DATASET_NAME
	# --store_all_nwk_folder makes <folder>/<dataset>.
	assert os.path.isdir(os.path.join(job['nwk_folder'], DATASET_NAME))


def test_ub_ellipsoid_size_does_not_change_the_result(c_ub_off_run, sanity2_run):
	"""YAML comment: 'This is a purely computational optimization that does not affect the final result.' Off (-1)
	gives the reconstruction of the default (1); 3 is part of test_options_without_effect_leave_the_result_unchanged."""
	c_ub_off_run.check()
	assert_same_reconstruction(final_state(c_ub_off_run.final_folder()), final_state(sanity2_run.final_folder()))


def test_use_knn_all_pairs(c_knn_all_run, sanity2_run):
	"""use_knn -1 considers all pairs of leaves for merging. It recovers the true tree, with the reference likelihood.
	The comparison with use_knn 10 pins the current output: on this dataset both reach the same reconstruction."""
	c_knn_all_run.check()
	sim, final = c_knn_all_run.sim, c_knn_all_run.final_folder()
	assert rf_distance(read_newick_file(final), sim.tree) == 0
	means, variances = sim.bonsai_units(_kept_genes(sim, final))
	_assert_loglik_is_the_reference(final, means, variances)
	assert_same_reconstruction(final_state(final), final_state(sanity2_run.final_folder()))


@pytest.mark.parametrize('strategy', ['long_branch_sorted', 'super_sure'])
def test_other_spr_strategies(strategy, c_spr_long_branch_run, c_spr_super_sure_run):
	"""spr_strategy long_branch_sorted and super_sure (YAML comment: 'Current options are large_tree,
	long_branch_sorted, super_sure') recover the true tree, and report the reference likelihood of the tree they
	write."""
	run = c_spr_long_branch_run if strategy == 'long_branch_sorted' else c_spr_super_sure_run
	run.check()
	sim, final = run.sim, run.final_folder()
	assert read_config(run.config)['spr_strategy'] == strategy
	assert rf_distance(read_newick_file(final), sim.tree) == 0
	means, variances = sim.bonsai_units(_kept_genes(sim, final))
	_assert_loglik_is_the_reference(final, means, variances)


def test_rescale_by_var_false_rescales_by_the_mean_variance(c_no_rescale_run):
	"""rescale_by_var False: features are not rescaled per gene; Bonsai divides all of them by the mean gene variance
	instead (metadata.json geneDiffusionScaling) and multiplies the posteriors back by it. The generator diffuses
	each gene at a rate proportional to its variance, so this model is misspecified for the data and the tree is
	not checked against the truth (it misses one split), only against the reference likelihood and posteriors."""
	c_no_rescale_run.check()
	sim, final = c_no_rescale_run.sim, c_no_rescale_run.final_folder()
	genes = _kept_genes(sim, final)
	mean_variance = float(np.mean(sim.gene_variances[genes]))
	metadata = read_metadata_json(final)
	assert metadata['geneDiffusionScaling'] == pytest.approx(mean_variance, rel=1e-12)
	assert metadata['loglikVarCorr'] == pytest.approx(-(sim.n_cells - 1) * len(genes) * math.log(mean_variance),
	                                                  rel=1e-12)
	means = {cell: sim.measured[genes, ind] / math.sqrt(mean_variance) for ind, cell in enumerate(sim.cell_ids)}
	variances = {cell: sim.measured_sd[genes, ind] ** 2 / mean_variance for ind, cell in enumerate(sim.cell_ids)}
	_assert_loglik_is_the_reference(final, means, variances)
	_assert_posteriors_are_the_reference(final, means, variances, np.full(len(genes), mean_variance))


def test_negative_zscore_cutoff_keeps_every_gene(c_all_genes_run):
	"""zscore_cutoff < 0: 'Negative means: keep all.' Every gene, noise genes included, enters the likelihood, and the
	folder names carry no _zscore tag."""
	c_all_genes_run.check()
	sim = c_all_genes_run.sim
	results = c_all_genes_run.results_folder
	assert set(_tree_folders(results)) == {stage_folder(stage, -1.0) for stage in STAGES + ('final',)}
	assert processed_data_folder(-1.0) in os.listdir(results)
	final = os.path.join(results, stage_folder('final', -1.0))
	metadata = read_metadata_json(final)
	assert metadata['geneIds'] == sim.gene_ids
	assert read_metadata_bonsai(final)['zscoreCutoff'] == '-1.0'
	means, variances = sim.bonsai_units()
	_assert_loglik_is_the_reference(final, means, variances)
	assert rf_distance(read_newick_file(final), sim.tree) == 0


def test_random_trees_do_not_matter_without_random_moves(c_zero_moves_no_trees_run, c_zero_moves_three_trees_run):
	"""nnn_n_randomtrees random trees are made by nnn_n_randommoves random interchanges each; with zero moves every
	random tree is the starting tree, so 0 and 3 random trees give the same reconstruction."""
	c_zero_moves_no_trees_run.check()
	c_zero_moves_three_trees_run.check()
	assert_same_reconstruction(final_state(c_zero_moves_three_trees_run.final_folder()),
	                           final_state(c_zero_moves_no_trees_run.final_folder()))
	sim, final = c_zero_moves_no_trees_run.sim, c_zero_moves_no_trees_run.final_folder()
	assert rf_distance(read_newick_file(final), sim.tree) == 0
	means, variances = sim.bonsai_units(_kept_genes(sim, final))
	_assert_loglik_is_the_reference(final, means, variances)


def test_used_run_configs_yaml_records_the_config_and_start_time(sanity2_run):
	"""README: the results folder holds 'a copy of the configuration YAML-file with an added timestamp'."""
	given = read_config(sanity2_run.config)
	used_path = sanity2_run.path('used_run_configs.yaml')
	used = read_config(used_path)
	for key, value in given.items():
		if key != 'start_time':
			assert used[key] == value, key
	assert given['start_time'] is None
	started = datetime.strptime(used['start_time'], '%d.%m.%y_%H:%M:%S')
	written = datetime.fromtimestamp(os.stat(used_path).st_mtime)
	assert 0 <= (written - started).total_seconds() < sanity2_run.seconds + 2


# ========================================================= files and metadata

def test_step_all_writes_every_intermediate_folder_and_cleans_up(sanity2_run):
	"""The folders --pickup_intermediate looks for, in the order the README lists the steps; the processed data and
	the config copy; nothing else. The intermediate tree folders keep no per-vertex data (*_vertByGene.npy) and the
	scratch folders random_trees, intermediate_trees and spr_intermediates are gone."""
	names = set(os.listdir(sanity2_run.results_folder))
	expected = {stage_folder(stage) for stage in STAGES + ('final',)} | {processed_data_folder(),
	                                                                    'used_run_configs.yaml'}
	assert names == expected
	for stage in STAGES[1:]:
		files = os.listdir(sanity2_run.path(stage_folder(stage)))
		assert {'edgeInfo.txt', 'vertInfo.txt', 'metadata.json', 'tree.nwk'} <= set(files), stage
		assert not [name for name in files if name.endswith('_vertByGene.npy')], stage
	for scratch in ('random_trees', 'intermediate_trees', 'spr_intermediates'):
		assert not os.path.exists(sanity2_run.path(scratch))


@pytest.mark.parametrize('source', ['tree.nwk', 'edgeInfo.txt'])
def test_final_tree_is_ladderized(source, features_run, c_duplicates_run):
	"""The last core step reorders the children of every node (bonsai_main.py: 'Swap children order ... to improve
	visual appearance') by ladderizing: children with fewer leaves below them come first. Pins the current
	output: the order among children with equally many leaves is not checked, so the trees used are the ladder
	(FEATURES) and the balanced tree with twins, whose nodes have children of unequal sizes."""
	for run in (features_run, c_duplicates_run):
		final = run.final_folder()
		tree = read_newick_file(final) if source == 'tree.nwk' else tree_from_folder(final)
		for node in tree.iter_nodes():
			sizes = [len(child.leaf_names()) for child in node.children]
			assert sizes == sorted(sizes), (node.name, sizes)


def test_metadata_bonsai_columns(sanity2_run):
	final = sanity2_run.final_folder()
	with open(os.path.join(final, 'metadata_bonsai.txt')) as handle:
		rows = list(csv.reader(handle))
	assert rows[0] == [''] + METADATA_COLUMNS
	assert len(rows) == 2 and rows[1][0] == 'metadata'
	values = dict(zip(METADATA_COLUMNS, rows[1][1:]))
	sim = sanity2_run.sim
	assert int(values['nGenes']) == len(sim.signal_genes) == read_metadata_json(final)['nGenes']
	assert int(values['nCells']) == sim.n_cells
	assert float(values['zscoreCutoff']) == 1.0
	assert int(values['coresUsed']) == 1
	assert 0 < float(values['reconsTime']) < sanity2_run.seconds
	assert values['trueLoglik'] == '' and values['fullLoglik'] == ''
	assert int(values['remainingRootChildren']) == len(read_newick_file(final).children)


def test_duplicated_cells_end_on_zero_length_edges(c_duplicates_run):
	"""Exact twins have nothing between them: each pair ends on zero-length edges below a shared node, at path length
	zero. The likelihood is the reference one with those zero-length edges."""
	c_duplicates_run.check()
	sim, final = c_duplicates_run.sim, c_duplicates_run.final_folder()
	tree = read_newick_file(final)
	assert sorted(tree.leaf_names()) == sorted(sim.cell_ids)
	for twin, original in sim.duplicate_of.items():
		assert path_length(tree, twin, original) == pytest.approx(0.0, abs=1e-12)
	means, variances = sim.bonsai_units(_kept_genes(sim, final))
	_assert_loglik_is_the_reference(final, means, variances)
	assert rf_distance(tree, sim.tree, min_length=0.0) == 0


def test_true_tree_folder_adds_true_and_full_logliks(c_true_tree_job, sanity2_run):
	"""With <data_folder>/true_tree, metadata_bonsai.txt reports on all genes (noise genes too), without
	loglikVarCorr: trueLoglik, the true tree with one factor on all its branch lengths optimised, and fullLoglik,
	the reconstructed tree with every branch length optimised. The reconstruction itself does not change.
	fullLoglik comes from an optimiser that stops at a relative tolerance and sets lengths under 2e-6 to zero;
	it may fall short of the reference optimum by 1e-5 relative (6e-7 seen) but not exceed it."""
	c_true_tree_job.check()
	sim, final = c_true_tree_job.sim, c_true_tree_job.final_folder()
	assert_same_reconstruction(final_state(final), final_state(sanity2_run.final_folder()))
	metadata = read_metadata_bonsai(final)
	means, variances = sim.bonsai_units()
	n_genes = sim.n_genes
	true_expected = to_bonsai_units(optimise_common_scale(sim.tree, means, variances), sim.n_cells, n_genes)
	assert float(metadata['trueLoglik']) == pytest.approx(true_expected, rel=1e-9)
	full_expected = to_bonsai_units(optimise_all_lengths(read_newick_file(final), means, variances)[0], sim.n_cells,
	                                n_genes)
	full = float(metadata['fullLoglik'])
	assert full <= full_expected + 1e-9 * abs(full_expected)
	assert full >= full_expected - 1e-5 * abs(full_expected)
	assert processed_data_folder(-1.0) in os.listdir(c_true_tree_job.results_folder)
