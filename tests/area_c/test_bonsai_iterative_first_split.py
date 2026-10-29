"""
backbone_based_bonsai/bonsai_iterative_first_split.py (area C), the first step of backbone-based Bonsai: read and
filter the data of all cells once, store each subset's columns in its results folder, and make the star tree
of the first subset.

Expected values: the generator's measurements in Bonsai's rescaled units (Sanity prior divided out, divided by
sqrt(v_g)), the cell lists written to starting_cell_ids.txt, and the star tree's branch lengths at the maximum
of the dense reference likelihood.
"""
import os

import numpy as np
import pytest

from bonsai_testdata import DatasetSpec, balanced_newick
from bonsai_testfixtures import DATASET_NAME, bonsai_job_fixture
from bonsai_testref import reference_bonsai_loglik
from bonsai_testrun import read_metadata_json, read_newick_file, write_config, processed_data_folder

from area_c.helpers_c import optimise_all_lengths, run_script, to_bonsai_units

SANITY2 = DatasetSpec(newick=balanced_newick(4, internal_length=0.5, leaf_length=0.1), n_genes=60,
                      n_noise_genes=20, noise_sd=0.1, seed=0, layout='sanity2')
SCRIPT = 'backbone_based_bonsai/bonsai_iterative_first_split.py'


def _first_split_job(pool, workdir):
	data_folder, sim = pool.dataset(SANITY2)
	results = os.path.join(workdir, 'results')
	config = write_config(os.path.join(workdir, 'config.yaml'), DATASET_NAME, data_folder, results,
	                      input_is_sanity_output=True)
	subsets = {'first': sim.cell_ids[::3], 'second': list(reversed(sim.cell_ids[1::2]))}
	folders = []
	for name, cells in subsets.items():
		folder = os.path.join(results, name)
		os.makedirs(os.path.join(folder, 'non_refined_tree'))
		with open(os.path.join(folder, 'non_refined_tree', 'starting_cell_ids.txt'), 'w') as handle:
			handle.write('\n'.join(cells) + '\n')
		folders.append(folder)
	returncode, log = run_script(SCRIPT, ['--config_filepath', config, '--subset_results_folders', ','.join(folders)],
	                             os.path.join(workdir, 'first_split.log'))
	return dict(returncode=returncode, log=log, sim=sim, subsets=subsets, folders=dict(zip(subsets, folders)))


c_first_split_job = bonsai_job_fixture('c_first_split_job', _first_split_job)


def _check(job):
	assert job['returncode'] == 0, job['log'][-3000:]
	return job['sim']


@pytest.mark.parametrize('name', ['first', 'second'])
def test_each_subset_gets_its_columns_of_the_processed_data(name, c_first_split_job):
	"""The genes are chosen on all cells (the signal genes), and each subset folder holds its cells' columns, in
	the order of its starting_cell_ids.txt, in Bonsai's rescaled units."""
	sim = _check(c_first_split_job)
	cells = c_first_split_job['subsets'][name]
	folder = os.path.join(c_first_split_job['folders'][name], processed_data_folder())
	metadata = read_metadata_json(folder)
	assert metadata['cellIds'] == cells and metadata['nCells'] == len(cells)
	genes = sorted(sim.signal_genes)
	assert metadata['geneIds'] == [sim.gene_ids[ind] for ind in genes]
	means, variances = sim.bonsai_units(genes)
	np.testing.assert_allclose(np.load(os.path.join(folder, 'delta.npy')), np.array([means[c] for c in cells]).T,
	                           rtol=1e-9, atol=1e-12)
	np.testing.assert_allclose(np.load(os.path.join(folder, 'delta_vars.npy')),
	                           np.array([variances[c] for c in cells]).T, rtol=1e-9)


def test_only_the_first_subset_gets_a_star_tree_with_optimal_branch_lengths(c_first_split_job):
	"""The first subset's non_refined_tree is the star tree of its cells, every cell a child of the root, with
	branch lengths that maximise the likelihood (the reference optimum to 1e-8 relative: Bonsai's optimiser stops
	at a tolerance; 3e-10 seen). The second subset gets no tree."""
	sim = _check(c_first_split_job)
	cells = c_first_split_job['subsets']['first']
	folder = os.path.join(c_first_split_job['folders']['first'], 'non_refined_tree')
	tree = read_newick_file(folder)
	assert sorted(child.name for child in tree.children) == sorted(cells)
	assert all(child.is_leaf for child in tree.children)
	metadata = read_metadata_json(folder)
	assert metadata['cellIds'] == cells and metadata['nCells'] == len(cells)
	genes = [sim.gene_ids.index(gene) for gene in metadata['geneIds']]
	means, variances = sim.bonsai_units(genes)
	means = {cell: means[cell] for cell in cells}
	variances = {cell: variances[cell] for cell in cells}
	optimum = to_bonsai_units(optimise_all_lengths(tree, means, variances)[0], len(cells), len(genes))
	at_bonsai = reference_bonsai_loglik(tree, means, variances)
	assert at_bonsai <= optimum + 1e-9 * abs(optimum)
	assert at_bonsai == pytest.approx(optimum, rel=1e-8)
	second = os.path.join(c_first_split_job['folders']['second'], 'non_refined_tree')
	assert os.listdir(second) == ['starting_cell_ids.txt']
