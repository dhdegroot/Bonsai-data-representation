"""
optional_preprocessing/batch_correction/merge_batches.py (area D): reads the Sanity output of every batch (written by
sanity_batch.py), divides Sanity's prior out (bonsai_dataprocessing.read_and_filter), centers every gene within each
batch, pads genes a batch has no counts for (mean 0, error bar 1e3), stacks the batches, adds the gene means of the
'total_counts_per_batch' run, and writes features.txt, standard_deviations.txt, cellID.txt, geneID.txt and a Bonsai
YAML (all 6 options).

Expected values come from the harness generator: for a Sanity 2.0 folder written from measurements m_gc with error
bars s_gc, dividing out the prior gives back m_gc and s_gc exactly, so
	features[g, c] = m_gc - mean_{c' in batch(c)} m_gc' + mu_total_g,   standard_deviations[g, c] = s_gc
for genes in the batch, and mu_total_g and 1e3 for genes it lacks. Each batch also gets its own offset on the
measurements and Sanity gene means (a batch effect), which the centering must remove.
"""
import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

from bonsai_testdata import DatasetSpec, balanced_newick, simulate, write_sanity
from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env, mpiexec_path, read_config

SCRIPT = os.path.join(REPO_ROOT, 'optional_preprocessing', 'batch_correction', 'merge_batches.py')
SPEC = DatasetSpec(newick=balanced_newick(3), n_genes=10, seed=21, layout='sanity2')
BATCHES = {'b1': [0, 1, 2, 3], 'b0': [4, 5, 6, 7]}  # cell indices; sorted batch order is b0, b1
MISSING = {'b0': [1, 4, 7], 'b1': []}  # genes a batch has no counts for
SHIFT = {'b0': 0.7, 'b1': -0.4}


def _total_gene_means(n_genes):
	return np.linspace(-2.5, -1.0, n_genes)


def _write_inputs(folder):
	"""Per-batch Sanity 2.0 folders, the total_counts_per_batch folder and the batch annotation."""
	sim = simulate(SPEC)
	sanity_out = os.path.join(folder, 'sanity_out')
	for batch, cells in BATCHES.items():
		genes = [g for g in range(sim.n_genes) if g not in MISSING[batch]]
		part = dataclasses.replace(
			sim, cell_ids=[sim.cell_ids[c] for c in cells], gene_ids=[sim.gene_ids[g] for g in genes],
			gene_means=sim.gene_means[genes] + SHIFT[batch], gene_variances=sim.gene_variances[genes],
			measured=sim.measured[np.ix_(genes, cells)] + SHIFT[batch],
			measured_sd=sim.measured_sd[np.ix_(genes, cells)])
		write_sanity(part, os.path.join(sanity_out, batch))
	n_batches = len(BATCHES)
	mu_total = _total_gene_means(sim.n_genes)
	total = dataclasses.replace(
		sim, cell_ids=sorted(BATCHES), gene_means=mu_total,
		measured=mu_total[:, None] + np.linspace(-0.3, 0.3, n_batches)[None, :],
		measured_sd=np.full((sim.n_genes, n_batches), 0.2))
	write_sanity(total, os.path.join(sanity_out, 'total_counts_per_batch'))
	annotation = os.path.join(folder, 'batches.tsv')
	with open(annotation, 'w') as handle:
		for batch in ('b1', 'b0'):
			handle.write(''.join('%s\t%s\n' % (sim.cell_ids[c], batch) for c in BATCHES[batch]))
	return sim, sanity_out, annotation


def _merge(variant):
	def job(pool, workdir):
		sim, sanity_out, annotation = _write_inputs(workdir)
		results = os.path.join(workdir, 'bonsai_results')
		args = ['--sanity_output_folder', sanity_out, '--batch_annotation_file', annotation,
		        '--bonsai_results_folder', results]
		if variant == 'defaults':
			bonsai_input = sanity_out  # --bonsai_input_folder left out: the output goes to the Sanity folder
		else:
			bonsai_input = os.path.join(workdir, 'bonsai_input')
			args += ['--dataset', 'my_batches', '--bonsai_input_folder', bonsai_input, '--verbose', 'True']
		cmd = [sys.executable, SCRIPT] + args
		if variant.startswith('mpi'):
			cmd = [mpiexec_path(), '--bind-to', 'none', '-n', variant[3:], sys.executable, '-m', 'mpi4py',
			       SCRIPT] + args
		proc = subprocess.run(cmd, capture_output=True, text=True, env=bonsai_env(), cwd=workdir, timeout=180)
		return proc, sim, bonsai_input, results
	return job


d_opt_merge_batches = bonsai_job_fixture('d_opt_merge_batches', _merge('explicit'))
d_opt_merge_batches_defaults = bonsai_job_fixture('d_opt_merge_batches_defaults', _merge('defaults'))
# 2 ranks split the 10 genes evenly, 4 ranks unevenly.
d_opt_merge_batches_mpi2 = bonsai_job_fixture('d_opt_merge_batches_mpi2', _merge('mpi2'), slots=2, mpi=True)
d_opt_merge_batches_mpi4 = bonsai_job_fixture('d_opt_merge_batches_mpi4', _merge('mpi4'), slots=4, mpi=True)


def _expected(sim):
	"""features and standard deviations (genes x cells, cells in sorted batch order) and the cell IDs."""
	mu_total = _total_gene_means(sim.n_genes)
	features, stds, cell_ids = [], [], []
	for batch in sorted(BATCHES):
		cells = BATCHES[batch]
		measured = sim.measured[:, cells] + SHIFT[batch]
		block = measured - measured.mean(axis=1, keepdims=True) + mu_total[:, None]
		block_sd = sim.measured_sd[:, cells].copy()
		block[MISSING[batch], :] = mu_total[MISSING[batch], None]
		block_sd[MISSING[batch], :] = 1e3
		features.append(block)
		stds.append(block_sd)
		cell_ids += [sim.cell_ids[c] for c in cells]
	return np.hstack(features), np.hstack(stds), cell_ids


def _read_lines(path):
	with open(path) as handle:
		return handle.read().splitlines()


def _check_data(proc, sim, bonsai_input):
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	features, stds, cell_ids = _expected(sim)
	assert _read_lines(os.path.join(bonsai_input, 'cellID.txt')) == cell_ids
	assert _read_lines(os.path.join(bonsai_input, 'geneID.txt')) == sim.gene_ids
	# The files hold 17 significant digits; the prior division costs a few ulps.
	np.testing.assert_allclose(np.loadtxt(os.path.join(bonsai_input, 'features.txt')), features, rtol=1e-9, atol=1e-12)
	np.testing.assert_allclose(np.loadtxt(os.path.join(bonsai_input, 'standard_deviations.txt')), stds, rtol=1e-9)


def _check_config(results, bonsai_input, dataset):
	config = read_config(os.path.join(results, 'bonsai_config.yaml'))
	assert config['dataset'] == dataset
	assert os.path.abspath(config['data_folder']) == os.path.abspath(bonsai_input)
	assert os.path.abspath(config['results_folder']) == os.path.abspath(results)
	assert config['filenames_data'] == 'features.txt,standard_deviations.txt'
	assert config['input_is_sanity_output'] is False
	assert config['rescale_by_var'] is True
	assert config['pickup_intermediate'] is True
	assert float(config['zscore_cutoff']) == 1.0
	assert float(config['UB_ellipsoid_size']) == 1.0
	assert int(config['nnn_n_randommoves']) == 1000
	assert int(config['nnn_n_randomtrees']) == 10
	assert int(config['use_knn']) == 10


def test_batches_are_centered_padded_and_stacked(d_opt_merge_batches):
	proc, sim, bonsai_input, results = d_opt_merge_batches
	_check_data(proc, sim, bonsai_input)


def test_writes_a_bonsai_config_for_the_merged_data(d_opt_merge_batches):
	proc, sim, bonsai_input, results = d_opt_merge_batches
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	_check_config(results, bonsai_input, 'my_batches')


def test_defaults_write_into_the_sanity_folder(d_opt_merge_batches_defaults):
	"""
	Without --bonsai_input_folder the data go to --sanity_output_folder; --dataset defaults to
	batch_corrected_dataset.
	"""
	proc, sim, bonsai_input, results = d_opt_merge_batches_defaults
	_check_data(proc, sim, bonsai_input)
	_check_config(results, bonsai_input, 'batch_corrected_dataset')


def _check_against_serial(mpi_result, serial_result):
	"""read_and_filter splits the genes over the ranks; the merged data must not depend on that."""
	proc, sim, bonsai_input, results = mpi_result
	_check_data(proc, sim, bonsai_input)
	_check_config(results, bonsai_input, 'my_batches')
	for name in ('features.txt', 'standard_deviations.txt'):
		np.testing.assert_allclose(np.loadtxt(os.path.join(bonsai_input, name)),
		                           np.loadtxt(os.path.join(serial_result[2], name)), rtol=1e-12, atol=1e-14)


@pytest.mark.mpi
def test_two_ranks_give_the_serial_result(d_opt_merge_batches_mpi2, d_opt_merge_batches):
	_check_against_serial(d_opt_merge_batches_mpi2, d_opt_merge_batches)


@pytest.mark.mpi
def test_four_ranks_give_the_serial_result(d_opt_merge_batches_mpi4, d_opt_merge_batches):
	_check_against_serial(d_opt_merge_batches_mpi4, d_opt_merge_batches)
