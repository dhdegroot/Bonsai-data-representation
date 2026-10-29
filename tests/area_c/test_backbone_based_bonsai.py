"""
backbone_based_bonsai/backbone_based_bonsai.py (area C): the README's "Backbone-based Bonsai" - a Bonsai tree on
n_initial_cells cells, grown by growth_factor_guide with bonsai_add_cells.py and refined by bonsai_main.py until
every cell is on it - with --return_commands True (the commands written to iterative_build_commands.txt) and
False (everything run on one CPU).

The subsets come from numpy's legacy global generator seeded with --seed, so the tests re-derive them (and the
seed handed to bonsai_add_cells.py) independently. On Python 3.14 the script cannot start (a lone '%' in a help
string, which Python 3.14's argparse rejects); its runs here put the argparse shim of
area_c/helpers_c.py on PYTHONPATH, which only turns that check off.
"""
import csv
import os

import numpy as np
import pytest

from bonsai_testdata import DatasetSpec, balanced_newick
from bonsai_testfixtures import DATASET_NAME, bonsai_job_fixture
from bonsai_testref import reference_bonsai_loglik
from bonsai_testrun import (read_config, read_metadata_bonsai, read_metadata_json, read_newick_file, stage_folder,
                            write_config, processed_data_folder)
from bonsai_testtrees import rf_distance

from area_c.helpers_c import FINAL_FILES, argparse_shim_env, run_script

SANITY2 = DatasetSpec(newick=balanced_newick(4, internal_length=0.5, leaf_length=0.1), n_genes=60,
                      n_noise_genes=20, noise_sd=0.1, seed=0, layout='sanity2')
FAST = dict(nnn_n_randommoves=20, nnn_n_randomtrees=2)
SCRIPT = 'backbone_based_bonsai/backbone_based_bonsai.py'
TMP_START = '_tmpStartnon_refined_tree'


def expected_subsets(n_cells, n_initial, growth, seed):
	"""
	The subsets the README describes (n_initial cells, grown by a factor growth until every cell is in), drawn
	as the script draws them: nested, from the largest down, with numpy's legacy generator seeded by seed;
	then one random integer below 1e6 per growth step, the seed handed to bonsai_add_cells.py.
	"""
	sizes = [n_initial]
	while sizes[-1] < n_cells:
		sizes.append(int(min(growth * sizes[-1], n_cells)))
	state = np.random.RandomState(seed)
	subsets = [None] * len(sizes)
	subsets[-1] = np.arange(n_cells)
	for ind in range(len(sizes) - 2, -1, -1):
		subsets[ind] = state.choice(subsets[ind + 1], size=sizes[ind], replace=False)
	add_seeds = [state.randint(1e6) for _ in range(len(sizes) - 1)]
	return sizes, subsets, add_seeds


def _backbone_job(name, *extra, return_commands=True, n_initial=8, growth=2):
	def job(pool, workdir):
		data_folder, sim = pool.dataset(SANITY2)
		config = write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder,
		                      os.path.join(workdir, 'results'), input_is_sanity_output=True, **FAST)
		args = ['--config_filepath', config, '--n_initial_cells', n_initial, '--growth_factor_guide', growth,
		        '--return_commands', return_commands] + [str(arg) for arg in extra]
		returncode, log = run_script(SCRIPT, args, os.path.join(workdir, 'backbone.log'),
		                             env=argparse_shim_env(os.path.join(workdir, 'shim')))
		return dict(returncode=returncode, log=log, config=config, results=os.path.join(workdir, 'results'),
		            data_folder=data_folder, sim=sim)
	return bonsai_job_fixture(name, job)


c_backbone_commands_job = _backbone_job('c_backbone_commands_job')
c_backbone_commands_options_job = _backbone_job(
	'c_backbone_commands_options_job', '--seed', 7, '--select_target', 'root', '--search_tol', 3.5,
	'--growth_before_cleanup', 0.25, '--pickup_intermediate', 'True', '--iterative_cell_lists', 'not_a_file.tsv',
	growth=1.5)
c_backbone_too_large_start_job = _backbone_job('c_backbone_too_large_start_job', n_initial=16)
c_backbone_full_job = _backbone_job('c_backbone_full_job', return_commands=False)


def _read_ids(path):
	with open(path) as handle:
		return [line.strip() for line in handle if line.strip()]


def _commands(job):
	with open(os.path.join(job['results'], 'iterative_build_commands.txt')) as handle:
		return [line.split() for line in handle.read().splitlines()]


def _option(command, name):
	return command[command.index(name) + 1]


def _check_layout(job, n_initial, growth, seed):
	"""Subset folders, starting_cell_ids.txt and iteration_subset_annotation.tsv against the re-derived subsets."""
	sim = job['sim']
	sizes, subsets, add_seeds = expected_subsets(sim.n_cells, n_initial, growth, seed)
	results = job['results']
	subset_folders = [os.path.join(results, 'subset_%d' % ind) for ind in range(len(sizes) - 1)] + [results]
	for folder, subset in zip(subset_folders, subsets):
		assert _read_ids(os.path.join(folder, 'non_refined_tree', 'starting_cell_ids.txt')) == \
			[sim.cell_ids[ind] for ind in subset]
	with open(os.path.join(results, 'iteration_subset_annotation.tsv')) as handle:
		rows = list(csv.reader(handle, delimiter='\t'))
	assert rows[0] == ['', 'iteration_subset']
	first_subset = {}
	for ind, subset in enumerate(subsets):
		for cell in subset:
			first_subset.setdefault(sim.cell_ids[cell], 'subset_%d' % ind)
	assert {row[0]: row[1] for row in rows[1:]} == first_subset
	assert [row[0] for row in rows[1:]] == sim.cell_ids
	return sizes, subsets, add_seeds, subset_folders


def _check_config(path, dataset, data_folder, results_folder, tmp_folder, main_config):
	"""get_command_config_file: the run options come from the main YAML file, the paths are the subset's own."""
	config = read_config(path)
	assert config['dataset'] == dataset
	assert os.path.abspath(config['data_folder']) == os.path.abspath(data_folder)
	assert os.path.abspath(config['results_folder']) == os.path.abspath(results_folder)
	assert config['pickup_intermediate'] is True
	if tmp_folder is None:
		assert config['tmp_folder'] in ('', None)
	else:
		assert os.path.abspath(config['tmp_folder']) == os.path.abspath(tmp_folder)
	for key in ('input_is_sanity_output', 'zscore_cutoff', 'UB_ellipsoid_size', 'nnn_n_randommoves',
	            'nnn_n_randomtrees', 'use_knn', 'rescale_by_var'):
		assert config[key] == main_config[key], key


def _check_commands(job, n_initial, growth, seed, select_target='cluster_centers', search_tol=2.0,
                    growth_before_cleanup=0.5, pickup='False'):
	sizes, subsets, add_seeds, folders = _check_layout(job, n_initial, growth, seed)
	main_config = read_config(job['config'])
	commands = _commands(job)
	assert len(commands) == 2 * len(sizes)
	first = commands[0]
	assert first[0] == 'backbone_based_bonsai/bonsai_iterative_first_split.py'
	assert _option(first, '--config_filepath') == job['config']
	assert _option(first, '--subset_results_folder').split(',') == folders[:-1]
	for ind, folder in enumerate(folders):
		main = commands[1 + 2 * ind]
		assert main[:1] == ['bonsai/bonsai_main.py']
		assert _option(main, '--step') == 'all' and _option(main, '--pickup_intermediate') == 'True'
		config_path = _option(main, '--config_filepath')
		assert config_path == os.path.join(folder, 'iterative_configs.yaml')
		dataset = DATASET_NAME if ind == len(folders) - 1 else os.path.join(DATASET_NAME, 'subset_%d' % ind)
		_check_config(config_path, dataset, job['data_folder'] if ind == len(folders) - 1 else folder, folder,
		              os.path.join(folder, 'non_refined_tree'), main_config)
		if ind == len(folders) - 1:
			break
		add = commands[2 + 2 * ind]
		assert add[0] == 'backbone_based_bonsai/bonsai_add_cells.py'
		target = os.path.join(folders[ind + 1], 'non_refined_tree')
		assert _option(add, '--config_filepath') == os.path.join(target, 'iterative_configs.yaml')
		_check_config(os.path.join(target, 'iterative_configs.yaml'), dataset, folder, target, None, main_config)
		assert _option(add, '--guide_tree_folder') == os.path.join(folder, stage_folder('final') + TMP_START)
		assert _option(add, '--cells_to_be_added') == os.path.join(target, 'starting_cell_ids.txt')
		assert _option(add, '--preprocessed_data_folder') == os.path.join(folders[ind + 1], processed_data_folder())
		assert _option(add, '--select_target') == select_target
		assert float(_option(add, '--search_tol')) == search_tol
		assert float(_option(add, '--growth_before_cleanup')) == growth_before_cleanup
		assert int(_option(add, '--seed')) == add_seeds[ind]
		assert _option(add, '--pickup_intermediate') == pickup


def test_return_commands_writes_the_documented_commands(c_backbone_commands_job):
	"""README: with --return_commands True 'a file will be created in the Bonsai-results folder that is called
	iterative_build_commands.txt. In this file, each line contains a run command'. 16 cells, 8 to start, growth 2:
	the first split, Bonsai on the 8 cells, adding the other 8, Bonsai on all 16. Nothing is computed."""
	job = c_backbone_commands_job
	assert job['returncode'] == 0, job['log'][-3000:]
	_check_commands(job, n_initial=8, growth=2, seed=1231)
	assert not [name for name in os.listdir(job['results']) if name.startswith(('final_bonsai', 'preprocessed'))]


def test_return_commands_passes_the_options_on(c_backbone_commands_options_job):
	"""growth_factor_guide 1.5 from 8 of 16 cells gives three subsets (8, 12, 16) and six commands; --seed,
	--select_target, --search_tol, --growth_before_cleanup and --pickup_intermediate reach bonsai_add_cells.py.
	--iterative_cell_lists is '[NOT YET IMPLEMENTED]' (its help) and changes nothing."""
	job = c_backbone_commands_options_job
	assert job['returncode'] == 0, job['log'][-3000:]
	_check_commands(job, n_initial=8, growth=1.5, seed=7, select_target='root', search_tol=3.5,
	                growth_before_cleanup=0.25, pickup='True')


def test_a_start_as_large_as_the_dataset_is_refused(c_backbone_too_large_start_job):
	"""n_initial_cells >= the number of cells: the script says to run normal Bonsai and stops without commands."""
	job = c_backbone_too_large_start_job
	assert 'normal' in job['log']
	assert not os.path.exists(os.path.join(job['results'], 'iterative_build_commands.txt'))
	assert not os.path.exists(os.path.join(job['results'], 'subset_0'))


def test_backbone_run_on_one_cpu_reaches_a_tree_of_every_cell(c_backbone_full_job):
	"""--return_commands False runs the four steps itself. The subset tree holds exactly the 8 starting cells, the
	final tree (in final_bonsai_zscore1.0_tmpStartnon_refined_tree, as the commands name it) every cell, with the
	true topology and the reference likelihood."""
	job = c_backbone_full_job
	assert job['returncode'] == 0, job['log'][-3000:]
	sim = job['sim']
	sizes, subsets, _, folders = _check_layout(job, 8, 2, 1231)
	subset_final = os.path.join(folders[0], stage_folder('final') + TMP_START)
	assert sorted(read_newick_file(subset_final).leaf_names()) == sorted(sim.cell_ids[ind] for ind in subsets[0])
	final = os.path.join(job['results'], stage_folder('final') + TMP_START)
	assert set(os.listdir(final)) == FINAL_FILES
	tree = read_newick_file(final)
	assert sorted(tree.leaf_names()) == sorted(sim.cell_ids)
	assert rf_distance(tree, sim.tree) == 0
	genes = [sim.gene_ids.index(gene) for gene in read_metadata_json(final)['geneIds']]
	means, variances = sim.bonsai_units(genes)
	assert float(read_metadata_bonsai(final)['finalLoglik']) == pytest.approx(
		reference_bonsai_loglik(tree, means, variances), rel=1e-9)
	grown = os.path.join(job['results'], 'non_refined_tree')
	assert sorted(read_newick_file(grown).leaf_names()) == sorted(sim.cell_ids)
