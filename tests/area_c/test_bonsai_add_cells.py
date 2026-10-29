"""
backbone_based_bonsai/bonsai_add_cells.py (area C): placing cells that are in the data but not on a guide tree.

The data are SANITY2 with exact twins of two cells (C0000d, C0001d); the guide tree is the generator's tree
without the twins. An exact twin's best place is on its original, so after adding it the two are at path length
zero; with fewer added cells than trigger a clean-up (growth_before_cleanup), the rest of the guide tree stays as
it was, here the true topology. metadata.json reports the likelihood of the tree written (with loglikVarCorr),
checked against the dense reference.

On Python 3.14 the script cannot start (a lone '%' in a help string, which Python 3.14's argparse rejects); its
runs here put the argparse shim of area_c/helpers_c.py on PYTHONPATH, which only turns that check off.
"""
import os
from dataclasses import replace

import pytest

from bonsai_testdata import DatasetSpec, balanced_newick
from bonsai_testfixtures import DATASET_NAME, bonsai_job_fixture
from bonsai_testref import reference_bonsai_loglik
from bonsai_testrun import read_metadata_json, read_newick_file, run_bonsai, stage_folder, write_config, \
	processed_data_folder
from bonsai_testtrees import path_length, rf_distance, splits

from area_c.helpers_c import argparse_shim_env, run_script, write_tree_folder

DUPLICATES = DatasetSpec(newick=balanced_newick(4, internal_length=0.5, leaf_length=0.1), n_genes=60,
                         n_noise_genes=20, noise_sd=0.1, seed=0, layout='sanity2', n_duplicates=2)
SCRIPT = 'backbone_based_bonsai/bonsai_add_cells.py'


def _guide_tree(sim):
	"""The generator's tree with each twin pair (a parent of two zero-length leaves) back to one leaf."""
	tree = sim.tree.copy()
	for node in tree.iter_nodes():
		twins = [child.name for child in node.children if child.name in sim.duplicate_of]
		if twins:
			node.name = sim.duplicate_of[twins[0]]
			node.children = []
	return tree


def _swap(tree, first, second):
	tree = tree.copy()
	for node in tree.iter_nodes():
		if node.name == first:
			node.name = second
		elif node.name == second:
			node.name = first
	return tree


# name: (extra arguments, prepare checkpoints, give --guide_tree_folder, give --preprocessed_data_folder)
SCENARIOS = {
	'default': ([], False, True, True),
	'root': (['--select_target', 'root'], False, True, True),
	'all_targets': (['--select_target', 'all', '--search_tol', 0.1], False, True, True),
	'from_data_folder': ([], False, True, False),
	'early_cleanup': (['--growth_before_cleanup', 0.05], False, True, True),
	'pickup': (['--pickup_intermediate', 'True'], True, True, True),
	'no_pickup': (['--pickup_intermediate', 'False'], True, True, True),
	'no_guide': ([], False, False, True),
}


def _add_cells_job(names):
	"""
	--step preprocess of bonsai_main.py on all 18 cells gives the processed data; the guide tree is written as a
	tree folder of the 16 originals. Then bonsai_add_cells.py runs in each named scenario, one after another.
	"""
	def job(pool, workdir):
		data_folder, sim = pool.dataset(DUPLICATES)
		prep_config = write_config(os.path.join(workdir, 'prep.yaml'), DATASET_NAME, data_folder,
		                           os.path.join(workdir, 'prep'), input_is_sanity_output=True)
		run_bonsai(prep_config, step='preprocess').check()
		processed = os.path.join(workdir, 'prep', processed_data_folder())
		originals = [cell for cell in sim.cell_ids if cell not in sim.duplicate_of]
		metadata = dict(read_metadata_json(os.path.join(workdir, 'prep', stage_folder('preprocessed'))))
		metadata.update(cellIds=originals, nCells=len(originals))
		guide = os.path.join(workdir, 'guide')
		write_tree_folder(_guide_tree(sim), guide, originals, metadata)
		cells_file = os.path.join(workdir, 'cells.txt')
		with open(cells_file, 'w') as handle:
			handle.write('\n'.join(sim.cell_ids) + '\n')
		shim = argparse_shim_env(os.path.join(workdir, 'shim'))
		scenarios = {}
		for name in names:
			extra, checkpoints, with_guide, with_preprocessed = SCENARIOS[name]
			folder = os.path.join(workdir, name)
			results = os.path.join(folder, 'results')
			config = write_config(os.path.join(folder, 'config.yaml'), DATASET_NAME, data_folder, results,
			                      input_is_sanity_output=True)
			if checkpoints:
				# A checkpoint with C0000 and C1111 swapped (numbered 5) and the plain guide tree (numbered 2).
				write_tree_folder(_swap(_guide_tree(sim), 'C0000', 'C1111'), os.path.join(results, 'tmp_added', 'added_5'),
				                  originals, metadata)
				write_tree_folder(_guide_tree(sim), os.path.join(results, 'tmp_added', 'added_2'), originals, metadata)
			args = ['--config_filepath', config, '--cells_to_be_added', cells_file, '--seed', 3] + list(extra)
			if with_guide:
				args += ['--guide_tree_folder', guide]
			if with_preprocessed:
				args += ['--preprocessed_data_folder', processed]
			returncode, log = run_script(SCRIPT, args, os.path.join(folder, 'add_cells.log'), env=shim)
			scenarios[name] = dict(returncode=returncode, log=log, results=results)
		return dict(sim=sim, scenarios=scenarios)
	return job


c_add_cells_targets_job = bonsai_job_fixture('c_add_cells_targets_job', _add_cells_job(
	('default', 'root', 'all_targets', 'from_data_folder')))
c_add_cells_options_job = bonsai_job_fixture('c_add_cells_options_job', _add_cells_job(
	('early_cleanup', 'pickup', 'no_pickup', 'no_guide')))


def _result(job, name):
	scenario = job['scenarios'][name]
	assert scenario['returncode'] == 0, scenario['log'][-3000:]
	tree = read_newick_file(scenario['results'])
	metadata = read_metadata_json(scenario['results'])
	return tree, metadata


def _assert_reference_loglik(sim, tree, metadata):
	genes = [sim.gene_ids.index(gene) for gene in metadata['geneIds']]
	means, variances = sim.bonsai_units(genes)
	assert metadata['loglik'] == pytest.approx(
		reference_bonsai_loglik(tree, means, variances, loglik_var_corr=metadata['loglikVarCorr']), rel=1e-9)


@pytest.mark.parametrize('name', ['default', 'root', 'all_targets', 'from_data_folder', 'no_pickup'])
def test_twins_are_added_onto_their_originals(name, c_add_cells_targets_job, c_add_cells_options_job):
	"""Every --select_target (cluster_centers, root, all), a small --search_tol, the data read from the YAML data
	folder instead of --preprocessed_data_folder, and --pickup_intermediate False (checkpoints present but
	ignored): each twin ends at path length zero from its original and the true topology is kept."""
	job = c_add_cells_options_job if name == 'no_pickup' else c_add_cells_targets_job
	sim = job['sim']
	tree, metadata = _result(job, name)
	assert sorted(tree.leaf_names()) == sorted(sim.cell_ids)
	assert metadata['cellIds'] == sim.cell_ids and metadata['nCells'] == len(sim.cell_ids)
	for twin, original in sim.duplicate_of.items():
		assert path_length(tree, twin, original) == pytest.approx(0.0, abs=1e-12)
	assert rf_distance(tree, sim.tree) == 0
	_assert_reference_loglik(sim, tree, metadata)


def test_an_early_cleanup_reoptimises_the_tree(c_add_cells_options_job, c_add_cells_targets_job):
	"""growth_before_cleanup 0.05 of 16 cells re-optimises the tree after the first added cell: the true topology
	stays, the twins stay together (within 1e-3), and the branch lengths, which start at the generator's values,
	move to a higher likelihood than without the clean-up."""
	sim = c_add_cells_options_job['sim']
	tree, metadata = _result(c_add_cells_options_job, 'early_cleanup')
	_, plain = _result(c_add_cells_targets_job, 'default')
	assert sorted(tree.leaf_names()) == sorted(sim.cell_ids)
	for twin, original in sim.duplicate_of.items():
		assert path_length(tree, twin, original) < 1e-3
	assert rf_distance(tree, sim.tree) == 0
	_assert_reference_loglik(sim, tree, metadata)
	assert metadata['loglik'] > plain['loglik']


def test_pickup_starts_from_the_latest_checkpoint(c_add_cells_options_job):
	"""With --pickup_intermediate True the highest-numbered tmp_added/added_<n> replaces the guide tree: added_5
	has C0000 and C1111 swapped, so C1111 sits next to C0001 in the result, and C0000's twin, placed by its data,
	is far from C0000."""
	sim = c_add_cells_options_job['sim']
	tree, metadata = _result(c_add_cells_options_job, 'pickup')
	assert sorted(tree.leaf_names()) == sorted(sim.cell_ids)
	assert any({'C0001', 'C1111'} <= split and len(split) <= 4 for split in splits(tree, min_length=-1))
	assert path_length(tree, 'C0000d', 'C0000') > 1.0
	_assert_reference_loglik(sim, tree, metadata)


def test_without_a_guide_tree_nothing_is_done(c_add_cells_options_job):
	scenario = c_add_cells_options_job['scenarios']['no_guide']
	assert 'guide_tree_folder' in scenario['log']
	assert not os.path.exists(os.path.join(scenario['results'], 'tree.nwk'))
