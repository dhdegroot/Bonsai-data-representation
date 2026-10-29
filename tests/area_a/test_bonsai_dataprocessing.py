"""
The data side of bonsai/bonsai_dataprocessing.py (area A): reading Sanity 2.0+, pre-2.0 Sanity and features
input, dividing out Sanity's prior, the zscore filter, gene variances, the per-gene rescaling, Metadata,
SCData's readers and paths, storing and loading tree folders, posteriors, annotations and getMetadata.
The tree algorithms of this module are area B's (tests/area_b/test_bonsai_dataprocessing.py).

Expected values come from the generator's truth (bonsai_testdata.py), the dense Gaussian reference
(bonsai_testref.py) and short derivations in the docstrings. Everything runs in-process, without MPI.
"""
import csv
import json
import math
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

import bonsai.bonsai_globals as bs_glob
import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import (Metadata, OriginalData, SCData, addDataToTree, correct_means_stds,
                                          get_bonsai_euclidean_distances, get_bonsai_posteriors, get_cell_info_tree,
                                          getMetadata, initializeSCData, loadReconstructedTreeAndData,
                                          logLGradDepOnV, loglik_given_true_var_log,
                                          neg_loglik_grad_using_measurements_and_errors, read_and_filter,
                                          reconstructTreeFromEdgeVertInfo, storeData)
from bonsai_testdata import DatasetSpec, balanced_newick, caterpillar_newick, make_dataset, simulate
from bonsai_testref import build_bonsai_tree, reference_bonsai_loglik, reference_posteriors
from bonsai_testrun import (posteriors_by_name, read_edge_info, read_newick_file, read_vert_info,
                            tree_from_folder)
from bonsai_testtrees import Node, clades, parse_newick, rf_distance

from area_a.helpers_a import (SERIAL, SMALL, args_for, cell_data, cwd_in_tmp, dataset, init_scdata,  # noqa: F401
                              make_args, ml_feature_variance, named_truth, scdata_with_tree, small_run_spec,
                              write_tree_files)
from bonsai_testfixtures import bonsai_run_fixture
from bonsai_testrun import read_config, read_metadata_bonsai, read_metadata_json, stage_folder

RTOL = 1e-9


@pytest.fixture(autouse=True)
def _no_mpi(cwd_in_tmp):
	"""Every test here runs as a single process from an empty working directory."""
	assert mpi_wrapper.comm is None
	yield


def _signal(sim):
	return np.sort(sim.signal_genes)


def _noise(sim):
	return np.setdiff1d(np.arange(sim.n_genes), sim.signal_genes)


# ------------------------------------------------------------------ read_and_filter, Sanity input

def test_small_spec_is_unambiguous_under_both_zscore_definitions():
	"""The zscore tests below rely on SMALL's genes being on the same side of the cutoff 1 under the documented
	statistic (mean subtracted) and under the one Bonsai computes (mean not subtracted, compared with the squared
	cutoff)."""
	sim = simulate(SMALL)
	delta, d_delta = sim.sanity_posteriors()
	documented = np.mean((delta - delta.mean(axis=1, keepdims=True)) ** 2 / d_delta ** 2, axis=1)
	code = np.mean(delta ** 2 / d_delta ** 2, axis=1)
	for z in (documented, code):
		assert z[_signal(sim)].min() > 4.0 and z[_noise(sim)].max() < 0.5


@pytest.mark.parametrize('layout,method', [('sanity2', 'MAP'), ('sanity2', 'MLE'), ('sanity2', 'EAP'),
                                           ('sanity_legacy', 'MAP')])
def test_sanity_input_divides_out_the_prior_and_keeps_the_signal_genes(tmp_path, layout, method):
	"""Sanity reports the posterior under its N(0, v_g) prior; Bonsai divides that prior out and adds back the
	gene mean mu_g, which gives the measurement likelihood N(m_gc, s_g^2) the generator drew (Bonsai SI). Both
	Sanity layouts give the same result. Noise genes fall under the default cutoff."""
	folder, sim = dataset(tmp_path, replace(SMALL, layout=layout, sanity_method=method))
	out = read_and_filter(folder, None, None, True, 1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	ltqs, ltqs_vars, gene_vars, n_cells, n_genes, genes_to_keep, stds_found, n_genes_orig = out
	keep = _signal(sim)
	np.testing.assert_array_equal(genes_to_keep, keep)
	assert (n_cells, n_genes, n_genes_orig, stds_found) == (8, len(keep), sim.n_genes, True)
	np.testing.assert_allclose(ltqs, sim.measured[keep], rtol=RTOL)
	np.testing.assert_allclose(ltqs_vars, sim.measured_sd[keep] ** 2, rtol=RTOL)
	np.testing.assert_allclose(gene_vars, sim.gene_variances[keep], rtol=1e-15)


def test_negative_zscore_cutoff_keeps_every_gene(tmp_path):
	"""YAML comment of zscore_cutoff: 'Negative means: keep all'."""
	folder, sim = dataset(tmp_path)
	out = read_and_filter(folder, None, None, True, -1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	np.testing.assert_array_equal(out[5], np.arange(sim.n_genes))
	np.testing.assert_allclose(out[0], sim.measured, rtol=RTOL)
	np.testing.assert_allclose(out[1], sim.measured_sd ** 2, rtol=RTOL)


def test_gene_whose_posterior_is_wider_than_its_prior_is_dropped(tmp_path):
	"""A posterior variance d_delta^2 at or above the prior variance v_g has no likelihood to divide out
	(1/s^2 = 1/d_delta^2 - 1/v_g <= 0), so the gene cannot be used; the other genes are unaffected."""
	folder, sim = dataset(tmp_path)
	_, d_delta = sim.sanity_posteriors()
	variances = sim.gene_variances.copy()
	bad = int(_signal(sim)[3])
	variances[bad] = 0.5 * float(np.min(d_delta[bad] ** 2))
	with open(os.path.join(folder, 'variance.txt'), 'w') as handle:
		handle.write(''.join('%.17g\n' % value for value in variances))
	out = read_and_filter(folder, None, None, True, -1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	expected = np.setdiff1d(np.arange(sim.n_genes), [bad])
	np.testing.assert_array_equal(out[5], expected)
	np.testing.assert_allclose(out[0], sim.measured[expected], rtol=RTOL)


def test_gene_whose_posterior_is_barely_narrower_than_its_prior_is_kept(tmp_path):
	"""A prior variance v_g only 1e-3 above the widest posterior variance d_delta^2 still leaves a likelihood to
	divide out: the gene is kept, with 1/s^2 = 1/d_delta^2 - 1/v_g and m = mu_g + delta s^2 / d_delta^2 (Bonsai
	drops a gene only when v_g - d_delta^2 falls under 1e-4 for some cell)."""
	folder, sim = dataset(tmp_path)
	delta, d_delta = sim.sanity_posteriors()
	variances = sim.gene_variances.copy()
	tight = int(_signal(sim)[2])
	variances[tight] = float(np.max(d_delta[tight] ** 2)) + 1e-3
	with open(os.path.join(folder, 'variance.txt'), 'w') as handle:
		handle.write(''.join('%.17g\n' % value for value in variances))
	out = read_and_filter(folder, None, None, True, -1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	np.testing.assert_array_equal(out[5], np.arange(sim.n_genes))
	post_var = d_delta[tight] ** 2
	s2 = 1.0 / (1.0 / post_var - 1.0 / variances[tight])
	np.testing.assert_allclose(out[1][tight], s2, rtol=1e-9)
	np.testing.assert_allclose(out[0][tight], sim.gene_means[tight] + delta[tight] * s2 / post_var, rtol=1e-9)
	np.testing.assert_allclose(out[0][_signal(sim)[0]], sim.measured[_signal(sim)[0]], rtol=RTOL)


def test_sanity_gene_variance_file_of_the_wrong_length_stops(tmp_path):
	folder, sim = dataset(tmp_path)
	with open(os.path.join(folder, 'variance.txt'), 'w') as handle:
		handle.write(''.join('%.17g\n' % value for value in sim.gene_variances[:-1]))
	with pytest.raises(SystemExit):
		read_and_filter(folder, None, None, True, 1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))


def test_marg_sanity_output_is_rejected_before_reading(tmp_path, caplog):
	folder, _ = dataset(tmp_path, replace(SMALL, sanity_method='MARG'))
	with pytest.raises(SystemExit):
		read_and_filter(folder, None, None, True, 1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	assert 'MARG' in caplog.text


# ------------------------------------------------------------------ read_and_filter, features input

FEATURES = replace(SMALL, layout='features')


def test_features_with_standard_deviations_are_the_likelihood_as_given(tmp_path):
	"""features.txt and standard_deviations.txt are the likelihood means and error-bars (README, 'Running Bonsai
	on other data-types'); they are kept as they are. The gene variance is the maximum-likelihood v_g of
	m_gc ~ N(mu_g, v_g + s_gc^2), computed here independently."""
	folder, sim = dataset(tmp_path, FEATURES)
	out = read_and_filter(folder, 'features.txt', 'standard_deviations.txt', False, 1.0, SERIAL,
	                      tmp_folder=str(tmp_path / 'tmp'))
	ltqs, ltqs_vars, gene_vars, n_cells, n_genes, genes_to_keep, stds_found, n_genes_orig = out
	keep = _signal(sim)
	np.testing.assert_array_equal(genes_to_keep, keep)
	assert (n_cells, n_genes, n_genes_orig, stds_found) == (8, len(keep), sim.n_genes, True)
	np.testing.assert_array_equal(ltqs, sim.measured[keep])
	np.testing.assert_allclose(ltqs_vars, sim.measured_sd[keep] ** 2, rtol=1e-15)
	expected = [ml_feature_variance(sim.measured[g], sim.measured_sd[g] ** 2)[0] for g in keep]
	# Bonsai's BFGS stops at a gradient of 1e-5 in log v, which leaves v accurate to about 1e-6 relative here.
	np.testing.assert_allclose(gene_vars, expected, rtol=1e-5)


def test_features_signal_to_noise_filter_drops_the_noise_genes(tmp_path):
	"""The noise genes' scatter equals their error-bars (no diffusion), the signal genes vary a hundred times
	more than their error-bars: they are dropped at the default cutoff (previous test) and kept at a negative one
	('Negative means: keep all')."""
	folder, sim = dataset(tmp_path, FEATURES)
	for g in _noise(sim):
		v, mu = ml_feature_variance(sim.measured[g], sim.measured_sd[g] ** 2)
		e = sim.measured_sd[g] ** 2
		assert np.mean((sim.measured[g] - sim.measured[g].mean()) ** 2 / e) < 2.0
		assert np.mean(v / (v + e) * (sim.measured[g] - mu) ** 2 / e) < 1.0
	out = read_and_filter(folder, 'features.txt', 'standard_deviations.txt', False, -1.0, SERIAL,
	                      tmp_folder=str(tmp_path / 'tmp'))
	np.testing.assert_array_equal(out[5], np.arange(sim.n_genes))
	np.testing.assert_array_equal(out[0], sim.measured)


def test_features_signal_to_noise_ignores_a_constant_offset(tmp_path):
	"""The documented z_g subtracts the feature's mean (YAML comment and README of zscore_cutoff), so adding the
	same constant to a feature in every cell changes neither which features pass the cutoff nor their inferred
	variances; the values themselves are kept as given."""
	folder, sim = dataset(tmp_path, FEATURES)
	# Large enough that the uncentred statistic would keep every noise feature.
	shifted = sim.measured + 1e4
	with open(os.path.join(folder, 'features.txt'), 'w') as handle:
		for row in shifted:
			handle.write('\t'.join('%.17g' % value for value in row) + '\n')
	out = read_and_filter(folder, 'features.txt', 'standard_deviations.txt', False, 1.0, SERIAL,
	                      tmp_folder=str(tmp_path / 'tmp'))
	keep = _signal(sim)
	np.testing.assert_array_equal(out[5], keep)
	np.testing.assert_array_equal(out[0], shifted[keep])
	expected = [ml_feature_variance(shifted[g], sim.measured_sd[g] ** 2)[0] for g in keep]
	np.testing.assert_allclose(out[2], expected, rtol=1e-5)


@pytest.mark.parametrize('stds', [None, 'missing_file.txt'])
def test_features_without_standard_deviations_get_a_tiny_error_bar_and_no_filter(tmp_path, stds):
	"""README: 'If standard_deviations.txt is not provided, Bonsai will assume that the uncertainty on the feature
	values is very small.' With vanishing error-bars every gene's signal-to-noise is unbounded, so none is
	filtered, except a constant feature, which carries no signal (and cannot be rescaled by its variance of 0).
	The gene variance is then the maximum-likelihood variance of the values, np.var. Pins the error-bar value
	(variance 1e-6)."""
	folder, sim = dataset(tmp_path, FEATURES)
	measured = sim.measured.copy()
	measured[2] = 1.5
	with open(os.path.join(folder, 'features.txt'), 'w') as handle:
		for row in measured:
			handle.write('\t'.join('%.17g' % value for value in row) + '\n')
	out = read_and_filter(folder, 'features.txt', stds, False, 1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	ltqs, ltqs_vars, gene_vars, n_cells, n_genes, genes_to_keep, stds_found, n_genes_orig = out
	keep = np.setdiff1d(np.arange(sim.n_genes), [2])
	np.testing.assert_array_equal(genes_to_keep, keep)
	assert stds_found is False and n_genes == len(keep) and n_genes_orig == sim.n_genes
	np.testing.assert_array_equal(ltqs, measured[keep])
	np.testing.assert_array_equal(ltqs_vars, np.full(ltqs.shape, 1e-6))
	np.testing.assert_allclose(gene_vars, np.var(measured[keep], axis=1), rtol=1e-12)


def test_missing_features_file_stops(tmp_path):
	folder, _ = dataset(tmp_path, FEATURES)
	with pytest.raises(SystemExit):
		read_and_filter(folder, 'no_such_file.txt', 'standard_deviations.txt', False, 1.0, SERIAL,
		                tmp_folder=str(tmp_path / 'tmp'))


@pytest.mark.parametrize('n_signal', [0, 1])
def test_fewer_than_two_genes_passing_stops_with_an_error(tmp_path, n_signal):
	"""A tree needs at least two features; with none or one passing the cutoff Bonsai exits with status 1."""
	spec = replace(SMALL, n_genes=n_signal, n_noise_genes=4)
	folder, _ = dataset(tmp_path, spec)
	with pytest.raises(SystemExit) as excinfo:
		read_and_filter(folder, None, None, True, 1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	assert excinfo.value.code not in (None, 0)


# ------------------------------------------------------------------ SCData and initializeSCData

_init = init_scdata


@pytest.mark.parametrize('rescale_by_var', [True, False])
def test_initialize_scdata_rescales_and_stores_the_processed_data(tmp_path, rescale_by_var):
	"""rescale_by_var (YAML comment): divide every gene by the square root of its inferred variance, so a
	gene's diffusion is measured in units of its own variance; the log-likelihood correction is the Jacobian
	of that rescaling, -(C-1) sum_g log v_g in Bonsai's units (twice the log-likelihood). Without it Bonsai
	scales all genes alike, by the mean variance v_bar, with Jacobian -(C-1) G log v_bar. The processed data
	are stored in the results folder, under the name metadata.processedDatafolder records."""
	scdata, sim, args = _init(tmp_path, rescale_by_var=rescale_by_var)
	keep = _signal(sim)
	v = sim.gene_variances[keep]
	scale = v if rescale_by_var else np.full(len(keep), v.mean())
	md = scdata.metadata
	assert (md.nCells, md.nGenes, md.dataset) == (8, len(keep), 'bonsai_test')
	assert md.cellIds == sim.cell_ids and md.geneIds == [sim.gene_ids[g] for g in keep]
	assert md.pathToOrigData == os.path.abspath(args.data_folder)
	assert md.results_folder == os.path.abspath(args.results_folder)
	assert md.processedDatafolder == os.path.join(md.results_folder, 'zscorefiltered_1.000_and_processed')
	np.testing.assert_allclose(md.geneVariances, v, rtol=1e-15)
	if rescale_by_var:
		assert md.geneDiffusionScaling == 'geneVariances'
		assert md.loglikVarCorr == pytest.approx(-(8 - 1) * np.sum(np.log(v)), rel=1e-12)
	else:
		assert md.geneDiffusionScaling == pytest.approx(v.mean(), rel=1e-15)
		assert md.loglikVarCorr == pytest.approx(-(8 - 1) * len(keep) * math.log(v.mean()), rel=1e-12)
	expected_means = sim.measured[keep] / np.sqrt(scale)[:, None]
	expected_vars = sim.measured_sd[keep] ** 2 / scale[:, None]
	np.testing.assert_allclose(scdata.originalData.ltqs, expected_means, rtol=RTOL)
	np.testing.assert_allclose(scdata.originalData.ltqsVars, expected_vars, rtol=RTOL)
	np.testing.assert_allclose(np.load(os.path.join(md.processedDatafolder, 'delta.npy')), expected_means, rtol=RTOL)
	np.testing.assert_allclose(np.load(os.path.join(md.processedDatafolder, 'delta_vars.npy')), expected_vars,
	                           rtol=RTOL)
	with open(os.path.join(md.processedDatafolder, 'metadata.json')) as handle:
		stored = json.load(handle)
	assert stored['cellIds'] == md.cellIds and stored['geneIds'] == md.geneIds and stored['nGenes'] == md.nGenes
	assert stored['loglikVarCorr'] == pytest.approx(md.loglikVarCorr, rel=1e-15)
	assert (bs_glob.nCells, bs_glob.nGenes) == (8, len(keep))


def test_processed_data_folder_name_carries_the_cutoff(tmp_path):
	scdata, sim, _ = _init(tmp_path, zscore_cutoff=-1.0)
	assert os.path.basename(scdata.metadata.processedDatafolder) == 'zscorefiltered_-1.000_and_processed'
	assert scdata.metadata.nGenes == sim.n_genes


@pytest.mark.parametrize('minimal', [False, True])
def test_initialize_scdata_on_all_ranks_of_a_single_process(tmp_path, minimal):
	"""allRanks=True (what bonsai_main.py uses) on one process gives the same data as allRanks=False and sets the
	globals the tree code reads."""
	folder, sim = dataset(tmp_path)
	args = args_for(SMALL, folder, tmp_path / 'results')
	scdata = initializeSCData(args, createStarTree=True, getOrigData=False, allRanks=True,
	                          otherRanksMinimalInfo=minimal)
	assert scdata.metadata.cellIds == sim.cell_ids
	assert scdata.metadata.geneIds == [sim.gene_ids[g] for g in _signal(sim)]
	assert (bs_glob.nCells, bs_glob.nGenes) == (8, len(sim.signal_genes))
	assert [child.nodeId for child in scdata.tree.root.childNodes] == sim.cell_ids


def test_initialize_scdata_all_genes_ignores_the_cutoff(tmp_path):
	"""allGenes=True (used for the 'fullLoglik' of getMetadata) reads every gene whatever zscore_cutoff says."""
	folder, sim = dataset(tmp_path)
	args = args_for(SMALL, folder, tmp_path / 'results', zscore_cutoff=1.0)
	scdata = initializeSCData(args, createStarTree=False, allGenes=True, getOrigData=True, allRanks=False)
	assert scdata.metadata.geneIds == sim.gene_ids
	np.testing.assert_allclose(scdata.originalData.ltqs, sim.measured / np.sqrt(sim.gene_variances)[:, None],
	                           rtol=RTOL)


def test_star_tree_holds_every_cell_with_its_rescaled_data(tmp_path):
	"""createStarTree=True: a root with one leaf per cell, each leaf carrying that cell's rescaled likelihood, and
	the tree's log-likelihood at the branch lengths it was given equals the dense reference on that star."""
	scdata, sim, _ = _init(tmp_path, create_star_tree=True)
	keep = _signal(sim)
	means, variances = sim.bonsai_units(keep)
	root = scdata.tree.root
	assert [child.nodeId for child in root.childNodes] == sim.cell_ids
	lengths = {}
	for child in root.childNodes:
		assert child.isLeaf
		np.testing.assert_allclose(child.ltqs, means[child.nodeId], rtol=RTOL)
		np.testing.assert_allclose(child.getLtqsVars(), variances[child.nodeId], rtol=RTOL)
		lengths[child.nodeId] = child.tParent
	star = Node('root', None, [Node(name, lengths[name]) for name in sim.cell_ids])
	got = scdata.tree.calcLogLComplete(mem_friendly=True, loglikVarCorr=scdata.metadata.loglikVarCorr)
	assert got == pytest.approx(reference_bonsai_loglik(star, means, variances,
	                                                     loglik_var_corr=scdata.metadata.loglikVarCorr), rel=RTOL)


def test_missing_dataset_name_stops(tmp_path):
	with pytest.raises(SystemExit):
		SCData(dataset=None, pathToOrigData=str(tmp_path), results_folder=str(tmp_path / 'r'))


@pytest.mark.parametrize('char', [':', ',', '(', ')', ';', ' '])
def test_cell_ids_with_newick_characters_are_rejected(tmp_path, char, caplog):
	"""Cell IDs end up in tree.nwk; IDs with ': , ( ) ;' or spaces cannot be written or read back."""
	folder, sim = dataset(tmp_path)
	ids = list(sim.cell_ids)
	ids[3] = 'bad' + char + 'id'
	with open(os.path.join(folder, 'cellID.txt'), 'w') as handle:
		handle.write(''.join(cell + '\n' for cell in ids))
	with pytest.raises(SystemExit):
		_init_from(folder, tmp_path)
	assert 'bad' + char + 'id' in caplog.text


def _init_from(folder, tmp_path, spec=SMALL, **overrides):
	args = args_for(spec, folder, tmp_path / 'results', **overrides)
	return initializeSCData(args, createStarTree=False, getOrigData=True, allRanks=False)


def test_cell_ids_with_other_punctuation_are_kept(tmp_path):
	folder, sim = dataset(tmp_path)
	ids = ['cell-%d_x.%d' % (ind, ind) for ind in range(sim.n_cells)]
	with open(os.path.join(folder, 'cellID.txt'), 'w') as handle:
		handle.write(''.join(cell + '\n' for cell in ids))
	assert _init_from(folder, tmp_path).metadata.cellIds == ids


@pytest.mark.parametrize('problem', ['too_few', 'duplicate'])
def test_cell_ids_must_match_the_columns_one_to_one(tmp_path, problem):
	folder, sim = dataset(tmp_path)
	ids = list(sim.cell_ids)
	if problem == 'too_few':
		ids = ids[:-1]
	else:
		ids[1] = ids[0]
	with open(os.path.join(folder, 'cellID.txt'), 'w') as handle:
		handle.write(''.join(cell + '\n' for cell in ids))
	with pytest.raises(SystemExit):
		_init_from(folder, tmp_path)


def test_gene_ids_must_match_the_rows(tmp_path):
	folder, sim = dataset(tmp_path)
	with open(os.path.join(folder, 'geneID.txt'), 'w') as handle:
		handle.write(''.join(gene + '\n' for gene in sim.gene_ids[:-1]))
	with pytest.raises(SystemExit):
		_init_from(folder, tmp_path)


def test_missing_id_files_give_generic_names(tmp_path):
	"""Without cellID.txt and geneID.txt, objects are 'Cell_<column>' and features 'Gene_<row>' (every gene kept
	here, so the generic names follow the rows of the input)."""
	spec = replace(SMALL, with_ids=False, n_noise_genes=0)
	folder, sim = dataset(tmp_path, spec)
	scdata = _init_from(folder, tmp_path, spec)
	assert scdata.metadata.cellIds == ['Cell_%d' % ind for ind in range(sim.n_cells)]
	assert scdata.metadata.geneIds == ['Gene_%d' % ind for ind in range(sim.n_genes)]


def test_gene_ids_from_a_log_transcription_quotients_table(tmp_path):
	"""Without geneID.txt, gene IDs are read from the first column of log_transcription_quotients.txt below its
	header line, and filtered like the data."""
	spec = replace(SMALL, with_ids=False)
	folder, sim = dataset(tmp_path, spec)
	names = ['gene_%s' % gene for gene in sim.gene_ids]
	with open(os.path.join(folder, 'log_transcription_quotients.txt'), 'w') as handle:
		handle.write('GeneID\t' + '\t'.join('c%d' % ind for ind in range(sim.n_cells)) + '\n')
		for name in names:
			handle.write(name + '\t' + '\t'.join('0.0' for _ in range(sim.n_cells)) + '\n')
	scdata = _init_from(folder, tmp_path, spec)
	assert scdata.metadata.geneIds == [names[g] for g in _signal(sim)]


def test_read_in_data_no_ltqs_reads_only_the_ids(tmp_path):
	folder, sim = dataset(tmp_path)
	scdata = SCData(onlyObject=True, dataset='bonsai_test', results_folder=str(tmp_path / 'results'))
	scdata.metadata.pathToOrigData = folder
	scdata.read_in_data_no_ltqs()
	assert scdata.metadata.cellIds == sim.cell_ids and scdata.metadata.geneIds == sim.gene_ids
	assert (scdata.metadata.nCells, scdata.metadata.nGenes) == (sim.n_cells, sim.n_genes)


def test_data_and_result_paths_resolve_against_the_given_folders(tmp_path):
	"""data_path and result_path join onto the absolute data and results folders (README: paths absolute or
	relative to the repository root, which is the working directory of bonsai_main.py)."""
	scdata = SCData(onlyObject=True, dataset='bonsai_test', results_folder='rel_results')
	scdata.metadata.pathToOrigData = 'rel_data'
	cwd = os.getcwd()
	assert scdata.result_path() == os.path.join(cwd, 'rel_results')
	assert scdata.result_path('final_bonsai') == os.path.join(cwd, 'rel_results', 'final_bonsai')
	assert scdata.data_path() == os.path.join(cwd, 'rel_data')
	assert scdata.data_path('cellID.txt') == os.path.join(cwd, 'rel_data', 'cellID.txt')
	assert os.path.isdir(os.path.join(cwd, 'rel_results'))


def test_scdata_and_original_data_repr_name_their_fields():
	data = OriginalData()
	data.ltqs = np.array([[1.5]])
	text = repr(data)
	for field in ('umiCounts', 'ltqs', 'ltqsVars', 'geneVariances', 'priorVariances'):
		assert field + ' = ' in text
	assert '1.5' in text
	scdata = SCData.__new__(SCData)
	scdata.metadata = Metadata()
	scdata.metadata.dataset = 'some_name'
	text = repr(scdata)
	assert text.startswith('SCData(') and 'some_name' in text and 'tree = None' in text


# ------------------------------------------------------------------ gene variances and the prior

def test_get_gene_variances_is_the_total_variance_of_the_posteriors():
	"""With error-bars: Var[x] = Var_c[E(x | data)] + E_c[Var(x | data)] (law of total variance); without: the
	variance of the values. An existing estimate is left alone."""
	rng = np.random.default_rng(3)
	data = OriginalData()
	data.ltqs = rng.normal(size=(4, 9))
	data.ltqsVars = rng.uniform(0.1, 0.5, size=(4, 9))
	scdata = SCData.__new__(SCData)
	scdata.getGeneVariances(data, ltqStdsFound=True)
	expected = [np.mean((row - row.mean()) ** 2) + np.mean(var) for row, var in zip(data.ltqs, data.ltqsVars)]
	np.testing.assert_allclose(data.geneVariances, expected, rtol=1e-12)
	data.geneVariances = None
	scdata.getGeneVariances(data, ltqStdsFound=False)
	np.testing.assert_allclose(data.geneVariances, [np.mean((row - row.mean()) ** 2) for row in data.ltqs],
	                           rtol=1e-12)
	data.geneVariances = np.array([7.0])
	scdata.getGeneVariances(data, ltqStdsFound=True)
	np.testing.assert_array_equal(data.geneVariances, [7.0])
	empty = OriginalData()
	scdata.getGeneVariances(empty)
	assert empty.geneVariances is None


def test_correct_means_stds_divides_out_the_prior():
	"""correct_means_stds turns Sanity's posterior (delta, d_delta^2) under N(0, v) into the likelihood
	(m - mu, s^2) the generator drew, and drops genes whose posterior is not narrower than the prior."""
	sim = simulate(SMALL)
	delta, d_delta = sim.sanity_posteriors()
	prior = sim.gene_variances.copy()
	prior[5] = 0.9 * float(np.min(d_delta[5] ** 2))
	data = OriginalData()
	data.ltqs, data.ltqsVars = delta.copy(), d_delta ** 2
	data.geneVariances, data.priorVariances = prior.copy(), prior.copy()
	uncorrected, genes_to_keep, problematic = correct_means_stds(data, prior)
	keep = np.setdiff1d(np.arange(sim.n_genes), [5])
	np.testing.assert_array_equal(genes_to_keep, keep)
	assert problematic is None
	np.testing.assert_allclose(data.ltqs, (sim.measured - sim.gene_means[:, None])[keep], rtol=1e-9, atol=1e-12)
	np.testing.assert_allclose(data.ltqsVars, sim.measured_sd[keep] ** 2, rtol=1e-9)
	np.testing.assert_array_equal(data.geneVariances, prior[keep])
	np.testing.assert_array_equal(uncorrected.ltqs, delta[keep])
	np.testing.assert_array_equal(uncorrected.ltqsVars, d_delta[keep] ** 2)


def test_correct_means_stds_names_problem_cells_when_no_gene_survives():
	data = OriginalData()
	data.ltqs = np.ones((2, 3))
	data.ltqsVars = np.array([[0.1, 2.0, 0.1], [0.1, 2.0, 2.0]])
	_, genes_to_keep, problematic = correct_means_stds(data, np.array([1.0, 1.0]))
	assert len(genes_to_keep) == 0
	assert list(problematic[:2]) == [1, 2]


def test_neg_loglik_is_the_gaussian_likelihood_at_the_weighted_mean():
	"""neg_loglik_grad_using_measurements_and_errors returns -log L(v) + const, with L the likelihood of
	m_c ~ N(mu_ML, v + e_c), compared here with scipy.stats.norm."""
	rng = np.random.default_rng(4)
	m, e = rng.normal(size=11), rng.uniform(0.1, 0.9, size=11)
	for log_v in (-1.0, 0.3):
		value, _, mu = neg_loglik_grad_using_measurements_and_errors(log_v, m, e, return_mean_ML=True)
		direct = np.sum(norm.logpdf(m, loc=mu, scale=np.sqrt(math.exp(log_v) + e)))
		assert -value == pytest.approx(direct + 0.5 * len(m) * math.log(2 * math.pi), rel=1e-12)


def test_loglik_given_true_var_log_is_minus_the_restricted_likelihood():
	"""loglik_given_true_var_log = -log of the likelihood with mu integrated out under a flat prior, up to a
	constant: integrate prod_c N(m_c; mu, v + e_c) over mu numerically and compare differences between two v."""
	rng = np.random.default_rng(6)
	m, e = rng.normal(1.0, 1.0, size=7), rng.uniform(0.1, 0.5, size=7)

	def restricted(log_v):
		total = math.exp(log_v) + e
		value, _ = integrate.quad(lambda mu: math.exp(np.sum(norm.logpdf(m, loc=mu, scale=np.sqrt(total))) + 10.0),
		                          -20, 20, epsabs=0, epsrel=1e-12, limit=200)
		return math.log(value) - 10.0 + 0.5 * len(m) * math.log(2 * math.pi) - 0.5 * math.log(2 * math.pi)

	diff_code = loglik_given_true_var_log(0.4, m, e) - loglik_given_true_var_log(-0.7, m, e)
	assert diff_code == pytest.approx(-(restricted(0.4) - restricted(-0.7)), rel=1e-8)


def test_logLGradDepOnV_gradient_matches_finite_differences():
	rng = np.random.default_rng(8)
	ltqs_sq, ltqs_vars = rng.uniform(0.1, 1.0, 6) ** 2, rng.uniform(0.01, 0.2, 6)
	for v in (0.5, 1.7):
		value, grad = logLGradDepOnV(v, 6, ltqs_sq, ltqs_vars)
		h = 1e-6
		numeric = (logLGradDepOnV(v + h, 6, ltqs_sq, ltqs_vars)[0] - logLGradDepOnV(v - h, 6, ltqs_sq, ltqs_vars)[0]) / (2 * h)
		assert grad == pytest.approx(numeric, rel=1e-6)
		expected = 2 * 6 * math.log(v) + np.sum(np.log(1 / (v - ltqs_vars)) + ltqs_sq / (v - ltqs_vars))
		assert value == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------------ Metadata

def _metadata():
	md = Metadata()
	md.pathToOrigData, md.dataset, md.nCells, md.nGenes = '/data/x', 'ds', 3, 2
	md.cellIds, md.geneIds = ['a', 'b', 'c'], ['g1', 'g2']
	md.loglik, md.loglikVarCorr = np.float64(-12.25), np.float64(3.5)
	md.geneVariances = np.array([0.5, 2.0])
	md.geneDiffusionScaling = 'geneVariances'
	md.processedDatafolder, md.results_folder = '/res/proc', '/res'
	return md


FIELDS = ('pathToOrigData', 'dataset', 'nCells', 'nGenes', 'cellIds', 'geneIds', 'loglik', 'loglikVarCorr',
          'geneVariances', 'geneDiffusionScaling', 'processedDatafolder', 'results_folder')


def test_metadata_json_round_trip(tmp_path):
	"""metadata.json holds every Metadata field as plain JSON (arrays as lists, numpy scalars as numbers) and
	reads back to the same values."""
	md = _metadata()
	path = str(tmp_path / 'metadata.json')
	md.to_json(path)
	with open(path) as handle:
		raw = json.load(handle)
	assert set(raw) == set(FIELDS)
	assert raw['geneVariances'] == [0.5, 2.0] and raw['loglik'] == -12.25 and raw['cellIds'] == ['a', 'b', 'c']
	back = Metadata(json_filepath=path)
	for field in FIELDS:
		value = getattr(back, field)
		if field == 'geneVariances':
			assert isinstance(value, np.ndarray)
			np.testing.assert_array_equal(value, md.geneVariances)
		else:
			assert value == getattr(md, field)
	assert isinstance(back.loglik, np.float64)
	assert back.to_dict() == md.to_dict()


def test_metadata_to_dict_converts_numpy_scalars():
	md = Metadata()
	md.nCells, md.loglik = np.int64(4), np.float64(1.25)
	out = md.to_dict()
	assert type(out['nCells']) is int and type(out['loglik']) is float
	assert out['geneVariances'] is None


def test_metadata_copy_takes_non_none_fields_and_copies_lists(tmp_path):
	md = _metadata()
	copy = Metadata(curr_metadata=md)
	assert copy.cellIds == md.cellIds and copy.cellIds is not md.cellIds
	assert copy.geneVariances is not md.geneVariances
	copy.cellIds.append('d')
	assert md.cellIds == ['a', 'b', 'c']
	# From a file, overridden by the non-None fields of the current metadata.
	path = str(tmp_path / 'm.json')
	md.to_json(path)
	current = Metadata()
	current.dataset = 'other'
	merged = Metadata(json_filepath=path, curr_metadata=current)
	assert merged.dataset == 'other' and merged.cellIds == ['a', 'b', 'c']


def test_metadata_with_inconsistent_cell_count_stops(tmp_path):
	md = _metadata()
	md.nCells = 4
	path = str(tmp_path / 'm.json')
	md.to_json(path)
	with pytest.raises(SystemExit):
		Metadata(json_filepath=path)


def test_metadata_take_subset_genes():
	md = _metadata()
	md.take_subset_genes([1])
	assert md.nGenes == 1 and md.geneIds == ['g2']
	np.testing.assert_array_equal(md.geneVariances, [2.0])
	with pytest.raises(SystemExit):
		md.take_subset_genes([0, 1])


def test_metadata_repr_lists_every_field():
	text = repr(_metadata())
	assert text.startswith('Metadata(')
	for field in FIELDS:
		assert '%s = ' % field in text
	assert "'g2'" in text and '/res/proc' in text


def test_store_data_writes_the_processed_folder(tmp_path):
	md = _metadata()
	md.processedDatafolder = str(tmp_path / 'proc' / 'nested')
	ltqs, ltqs_vars = np.arange(6.0).reshape(2, 3), np.full((2, 3), 0.25)
	storeData(md, ltqs, ltqs_vars)
	np.testing.assert_array_equal(np.load(os.path.join(md.processedDatafolder, 'delta.npy')), ltqs)
	np.testing.assert_array_equal(np.load(os.path.join(md.processedDatafolder, 'delta_vars.npy')), ltqs_vars)
	assert Metadata(json_filepath=os.path.join(md.processedDatafolder, 'metadata.json')).to_dict() == md.to_dict()


# ------------------------------------------------------------------ storing and loading tree folders

_named_truth = named_truth
_cell_data = cell_data
_scdata_with_tree = scdata_with_tree


def _subtree(tree, name):
	return tree.find(name).copy()


def _names_by_clade(stored, tree):
	"""Map every vertex name of a stored tree to the name of the node of `tree` below which the same cells
	hang. Bonsai renames internal nodes internal_<k> when it stores a tree; cells keep their IDs."""
	ours = {frozenset(node.leaf_names()): node.name for node in tree.iter_nodes()}
	return {node.name: ours[frozenset(node.leaf_names())] for node in stored.iter_nodes()}


def _gene_scale(scdata):
	md = scdata.metadata
	return md.geneVariances if md.geneDiffusionScaling == 'geneVariances' else np.full(md.nGenes,
	                                                                                    md.geneDiffusionScaling)


@pytest.mark.parametrize('rescale_by_var,mem_friendly', [(True, False), (False, False), (True, True)])
def test_store_tree_in_folder_writes_the_tree_and_the_posteriors(tmp_path, rescale_by_var, mem_friendly):
	"""The final-folder files of the README: edgeInfo.txt and vertInfo.txt describe the tree (lengths %.8e),
	tree.nwk the same tree at full precision, metadata.json the metadata, and the posterior files every vertex's
	posterior mean and variance given all cells, in the input's units (the per-gene rescaling undone). The
	posteriors equal the dense Gaussian reference."""
	scdata, sim, args, tree, means, variances = _scdata_with_tree(tmp_path, rescale_by_var=rescale_by_var)
	bs_glob.mem_friendly = mem_friendly
	folder = str(tmp_path / 'stored')
	scdata.storeTreeInFolder(folder, with_coords=True, store_posterior_ltqs=True)
	assert sorted(os.listdir(folder)) == ['edgeInfo.txt', 'metadata.json', 'posterior_ltqsVars_vertByGene.npy',
	                                      'posterior_ltqs_vertByGene.npy', 'tree.nwk', 'vertInfo.txt']
	from_files, from_nwk = tree_from_folder(folder), read_newick_file(folder)
	for stored in (from_files, from_nwk):
		assert clades(stored, include_root=True) == clades(tree, include_root=True)
		ours = _names_by_clade(stored, tree)
		for node in stored.iter_nodes():
			if node is not stored:
				expected = tree.find(ours[node.name]).length
				assert node.length == pytest.approx(expected, rel=1e-8 if stored is from_files else 1e-15)
	verts = read_vert_info(folder)
	assert [vert for vert, _, _ in verts] == list(range(len(verts)))
	assert verts[0][2] == 'root' and len({name for _, _, name in verts}) == len(verts)
	assert Metadata(json_filepath=os.path.join(folder, 'metadata.json')).to_dict() == \
		json.loads(json.dumps(scdata.metadata.to_dict()))
	post_means, post_vars = reference_posteriors(tree, means, variances)
	scale = _gene_scale(scdata)
	got_means, got_vars = posteriors_by_name(folder)
	ours = _names_by_clade(from_files, tree)
	assert sorted(ours.values()) == sorted(post_means)
	for name, our_name in ours.items():
		np.testing.assert_allclose(got_means[name], post_means[our_name] * np.sqrt(scale), rtol=RTOL, atol=1e-12)
		np.testing.assert_allclose(got_vars[name], post_vars[our_name] * scale, rtol=RTOL)


def test_store_tree_with_coordinates_writes_each_vertex_given_its_subtree(tmp_path):
	"""with_coords without posteriors: ltqs_vertByGene.npy holds, for every vertex, its position given only the
	cells below it (a leaf: its own likelihood; the root: everything), in Bonsai's rescaled units. nwk=False
	leaves out tree.nwk."""
	scdata, sim, args, tree, means, variances = _scdata_with_tree(tmp_path)
	folder = str(tmp_path / 'stored')
	scdata.storeTreeInFolder(folder, with_coords=True, nwk=False)
	assert sorted(os.listdir(folder)) == ['edgeInfo.txt', 'ltqsVars_vertByGene.npy', 'ltqs_vertByGene.npy',
	                                      'metadata.json', 'vertInfo.txt']
	ltqs = np.load(os.path.join(folder, 'ltqs_vertByGene.npy'))
	ltqs_vars = np.load(os.path.join(folder, 'ltqsVars_vertByGene.npy'))
	ours = _names_by_clade(tree_from_folder(folder), tree)
	for vert, _, name in read_vert_info(folder):
		node = tree.find(ours[name])
		if node.is_leaf:
			expected_mean, expected_var = means[name], variances[name]
		else:
			sub_means, sub_vars = reference_posteriors(_subtree(tree, node.name), means, variances)
			expected_mean, expected_var = sub_means[node.name], sub_vars[node.name]
		np.testing.assert_allclose(ltqs[vert], expected_mean, rtol=RTOL)
		np.testing.assert_allclose(ltqs_vars[vert], expected_var, rtol=RTOL)


def test_store_tree_without_coordinates_writes_only_the_tree(tmp_path):
	scdata, *_ = _scdata_with_tree(tmp_path)
	folder = str(tmp_path / 'stored')
	scdata.storeTreeInFolder(folder)
	assert sorted(os.listdir(folder)) == ['edgeInfo.txt', 'metadata.json', 'tree.nwk', 'vertInfo.txt']


def test_stored_tree_drops_zero_length_internal_edges_but_keeps_cells(tmp_path):
	"""An ancestor at zero distance from its parent is the same position: storing merges it into the parent (a
	polytomy), which leaves the likelihood unchanged. A cell on a zero-length edge stays."""
	newick = '(((C000:0.1,C001:0.0)a:0.0,C010:0.2)b:0.5,((C011:0.1,C100:0.1)c:0.3,(C101:0.1,C110:0.1,C111:0.2)d:0.2)e:0.5)root;'
	scdata, sim, args, tree, means, variances = _scdata_with_tree(tmp_path, newick=newick)
	folder = str(tmp_path / 'stored')
	scdata.storeTreeInFolder(folder)
	stored = tree_from_folder(folder)
	expected = parse_newick('((C000:0.1,C001:0.0,C010:0.2)b:0.5,((C011:0.1,C100:0.1)c:0.3,'
	                        '(C101:0.1,C110:0.1,C111:0.2)d:0.2)e:0.5)root;')
	assert clades(stored, min_length=-1, include_root=True) == clades(expected, min_length=-1, include_root=True)
	assert stored.find('C001').length == 0.0
	assert reference_bonsai_loglik(stored, means, variances) == pytest.approx(
		reference_bonsai_loglik(tree, means, variances), rel=1e-12)


@pytest.fixture
def stored_folder(tmp_path):
	"""The true tree of SMALL stored as bonsai_main.py stores its final tree (coordinates and posteriors)."""
	scdata, sim, args, tree, means, variances = _scdata_with_tree(tmp_path)
	folder = scdata.result_path('final_test')
	scdata.storeTreeInFolder(folder, with_coords=True, store_posterior_ltqs=True)
	return dict(folder=folder, scdata=scdata, sim=sim, args=args, tree=tree, means=means, variances=variances)


def _expected_loglik(folder, means, variances, var_corr=None):
	"""The dense reference on the tree as written to edgeInfo.txt (lengths rounded to %.8e)."""
	return reference_bonsai_loglik(tree_from_folder(folder), means, variances, loglik_var_corr=var_corr)


@pytest.mark.parametrize('source', ['processed_data', 'reprocessed', 'args_as_dict'])
def test_load_reconstructed_tree_and_data(stored_folder, source):
	"""Loading a stored tree folder gives back the tree of edgeInfo.txt/vertInfo.txt with every cell's data on its
	leaf, whether the data come from the processed-data folder metadata.json names or are read and processed again
	from the input; the log-likelihood (calc_loglik) is the dense reference plus loglikVarCorr."""
	s = stored_folder
	for name in ('posterior_ltqs_vertByGene.npy', 'posterior_ltqsVars_vertByGene.npy'):
		os.remove(os.path.join(s['folder'], name))
	args = s['args']
	if source == 'args_as_dict':
		args = {key: value for key, value in vars(args).items()}
	scdata, vert_to_id = loadReconstructedTreeAndData(args, 'final_test', reprocess_data=(source == 'reprocessed'),
	                                                  all_ranks=False, get_cell_info=True, rel_to_results=True,
	                                                  calc_loglik=True)
	verts = read_vert_info(s['folder'])
	assert vert_to_id == {vert: name for vert, _, name in verts}
	var_corr = s['scdata'].metadata.loglikVarCorr
	assert scdata.metadata.loglik == pytest.approx(_expected_loglik(s['folder'], s['means'], s['variances'], var_corr),
	                                               rel=RTOL)
	assert scdata.metadata.cellIds == s['sim'].cell_ids
	leaves = {node.nodeId: node for node in scdata.tree.root.getNodeList([], returnRoot=False) if node.isLeaf}
	assert set(leaves) == set(s['sim'].cell_ids)
	for cell, node in leaves.items():
		assert node.isCell
		np.testing.assert_allclose(node.ltqs, s['means'][cell], rtol=RTOL)
	assert bs_glob.nNodes == len(verts)


def test_load_reconstructed_tree_reads_stored_coordinates(stored_folder):
	"""A folder with ltqs_vertByGene.npy (as the intermediate folders have during a run) is loaded from those
	coordinates, giving the same log-likelihood."""
	s = stored_folder
	folder = s['scdata'].result_path('coords_test')
	s['scdata'].storeTreeInFolder(folder, with_coords=True)
	os.remove(os.path.join(s['scdata'].metadata.processedDatafolder, 'delta.npy'))
	scdata = loadReconstructedTreeAndData(s['args'], folder, all_ranks=False, calc_loglik=True)
	var_corr = s['scdata'].metadata.loglikVarCorr
	assert scdata.metadata.loglik == pytest.approx(_expected_loglik(folder, s['means'], s['variances'], var_corr),
	                                               rel=RTOL)


def test_load_reconstructed_tree_with_posteriors(stored_folder):
	"""get_posterior_ltqs reads the stored posterior files onto the vertices when they are there, and otherwise
	computes the posteriors (in rescaled units) from the data."""
	s = stored_folder
	scdata = loadReconstructedTreeAndData(s['args'], s['folder'], all_ranks=False, get_posterior_ltqs=True)
	stored_means, _ = posteriors_by_name(s['folder'])
	for node in scdata.tree.root.getNodeList([], returnRoot=True):
		np.testing.assert_array_equal(node.ltqsAIRoot, stored_means[node.nodeId])
	for name in ('posterior_ltqs_vertByGene.npy', 'posterior_ltqsVars_vertByGene.npy'):
		os.remove(os.path.join(s['folder'], name))
	scdata = loadReconstructedTreeAndData(s['args'], s['folder'], all_ranks=False, get_posterior_ltqs=True)
	ref_means, ref_vars = reference_posteriors(tree_from_folder(s['folder']), s['means'], s['variances'])
	for node in scdata.tree.root.getNodeList([], returnRoot=True):
		np.testing.assert_allclose(node.ltqsAIRoot, ref_means[node.nodeId], rtol=RTOL)
		np.testing.assert_allclose(node.getLtqsVars(AIRoot=True), ref_vars[node.nodeId], rtol=RTOL)


def test_reconstruct_tree_from_edge_vert_info_accepts_either_edge_orientation(tmp_path):
	"""The root is the parent of the first edge; later edges may list the already-known vertex second."""
	folder = tmp_path / 'tree'
	folder.mkdir()
	(folder / 'edgeInfo.txt').write_text('0\t1\t1.0e-01\n0\t2\t5.0e-01\n3\t2\t2.5e-01\n2\t4\t0.0e+00\n')
	(folder / 'vertInfo.txt').write_text('vertInd\tnodeInd\tvertName\n0\t-1\troot\n1\t0\tA\n2\t5\tx\n'
	                                     '3\t1\tB\n4\t2\tC\n')
	scdata = SCData.__new__(SCData)
	vert_to_node, vert_to_ind, vert_to_id, edges, dists = reconstructTreeFromEdgeVertInfo(scdata, str(folder))
	root = scdata.tree.root
	assert root.nodeId == 'root' and root.isRoot and [c.nodeId for c in root.childNodes] == ['A', 'x']
	x = root.childNodes[1]
	assert [c.nodeId for c in x.childNodes] == ['B', 'C'] and not x.isLeaf
	assert [c.tParent for c in x.childNodes] == [0.25, 0.0] and x.tParent == 0.5
	assert {vert: node.nodeId for vert, node in vert_to_node.items()} == vert_to_id
	assert all(node.vert_ind == vert for vert, node in vert_to_node.items())
	assert scdata.tree.nNodes == bs_glob.nNodes == 5
	assert scdata.tree.max_node_ind == bs_glob.max_node_ind == 5
	assert dists == [0.1, 0.5, 0.25, 0.0]


def test_get_cell_info_tree_maps_cells_to_vertices(stored_folder):
	s = stored_folder
	scdata, vert_to_id = loadReconstructedTreeAndData(s['args'], s['folder'], all_ranks=False, get_cell_info=True)
	id_to_vert = {name: vert for vert, name in vert_to_id.items()}
	assert scdata.nVerts == len(vert_to_id)
	assert scdata.vertNames == ['vert_%d' % vert for vert in range(scdata.nVerts)]
	for ind, cell in enumerate(s['sim'].cell_ids):
		assert scdata.cellIndToVertInd[ind] == id_to_vert[cell]
		assert scdata.cellsToVerts[cell] == 'vert_%d' % id_to_vert[cell]
		assert scdata.vertIndToCellInds[id_to_vert[cell]] == [ind]
	assert scdata.nCellsPerVert.sum() == s['sim'].n_cells
	assert sorted(np.nonzero(scdata.nCellsPerVert)[0]) == sorted(id_to_vert[c] for c in s['sim'].cell_ids)
	scdata.metadata.cellIds = scdata.metadata.cellIds[:-1] + ['not_in_tree']
	vert_to_node = {node.vert_ind: node for node in scdata.tree.root.getNodeList([], returnRoot=True)}
	with pytest.raises(SystemExit):
		get_cell_info_tree(scdata, vert_to_node)


def test_add_data_to_tree_puts_each_cells_data_on_its_leaf(stored_folder):
	s = stored_folder
	scdata = SCData.__new__(SCData)
	scdata.metadata = Metadata(curr_metadata=s['scdata'].metadata)
	vert_to_node = reconstructTreeFromEdgeVertInfo(scdata, s['folder'])[0]
	data = OriginalData()
	data.ltqs = np.arange(24.0).reshape(3, 8)
	data.ltqsVars = data.ltqs + 0.5
	scdata.originalData = data
	addDataToTree(scdata, vert_to_node)
	for node in vert_to_node.values():
		if node.nodeId in s['sim'].cell_ids:
			ind = s['sim'].cell_ids.index(node.nodeId)
			assert node.isCell
			np.testing.assert_array_equal(node.ltqs, data.ltqs[:, ind])
			np.testing.assert_array_equal(node.getLtqsVars(), data.ltqsVars[:, ind])
		else:
			assert node.ltqs is None


def test_cleanup_node_inds_numbers_cells_by_their_position_in_cell_ids(tmp_path):
	"""Cells get the node index of their position in metadata.cellIds; the other vertices get distinct indices
	after them, and the global maximum index and cell count follow."""
	scdata, sim, args, tree, means, variances = _scdata_with_tree(tmp_path)
	nodes = scdata.tree.root.getNodeList([], returnRoot=True)
	cells = [node for node in nodes if node.nodeId in sim.cell_ids]
	for offset, node in enumerate(cells):
		node.nodeInd, node.isCell = 100 + offset, False
	scdata.cleanup_node_inds()
	for node in cells:
		assert node.nodeInd == sim.cell_ids.index(node.nodeId) and node.isCell
	others = [node.nodeInd for node in nodes if node.nodeId not in sim.cell_ids and not node.isRoot]
	assert len(set(others)) == len(others) == 6 and min(others) >= sim.n_cells
	assert bs_glob.max_node_ind == max(node.nodeInd for node in nodes)
	assert scdata.metadata.nCells == bs_glob.nCells == sim.n_cells


def test_get_metadata_reports_the_final_log_likelihood(stored_folder):
	"""metadata_bonsai.txt's row: finalLoglik is the dense reference log-likelihood of the tree without
	loglikVarCorr; trueLoglik and fullLoglik are NaN without a 'true_tree' folder next to the data."""
	s = stored_folder
	scdata = loadReconstructedTreeAndData(s['args'], s['folder'], all_ranks=False)
	frame = getMetadata(s['args'], scdata, 'final_test', 12.5)
	assert list(frame.index) == ['metadata']
	assert list(frame.columns) == ['nGenes', 'nCells', 'reconsTime', 'zscoreCutoff', 'coresUsed', 'finalLoglik',
	                               'trueLoglik', 'fullLoglik', 'remainingRootChildren']
	row = frame.loc['metadata']
	assert (row['nGenes'], row['nCells'], row['coresUsed'], row['remainingRootChildren']) == \
		(len(s['sim'].signal_genes), 8, 1, 2)
	assert row['reconsTime'] == 12.5 and row['zscoreCutoff'] == 1.0
	assert row['finalLoglik'] == pytest.approx(_expected_loglik(s['folder'], s['means'], s['variances']), rel=RTOL)
	assert math.isnan(row['trueLoglik']) and math.isnan(row['fullLoglik'])


def _branch_optimum(tree, means, variances):
	"""max over all branch lengths >= 0 of the dense reference, fixed topology (L-BFGS-B on the lengths)."""
	edges = [node for node in tree.iter_nodes() if node is not tree]
	start = np.array([node.length for node in edges])

	def neg(lengths):
		for node, length in zip(edges, lengths):
			node.length = float(length)
		return -reference_bonsai_loglik(tree, means, variances)

	res = minimize(neg, start, method='L-BFGS-B', bounds=[(0.0, None)] * len(edges),
	               options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 2000})
	return -neg(res.x)


def test_get_metadata_with_a_true_tree_reports_true_and_full_log_likelihoods(stored_folder):
	"""When the data folder holds a 'true_tree' folder (simulated data), getMetadata also reports, on all genes:
	trueLoglik, the true topology with its branch lengths scaled by the one best factor, and fullLoglik, the
	reconstructed topology with every branch length optimised. Both are computed here with the dense reference."""
	s = stored_folder
	sim = s['sim']
	truth = _named_truth(sim)
	write_tree_files(os.path.join(s['args'].data_folder, 'true_tree'), truth)
	# The reconstructed tree: the true topology at the wrong branch lengths.
	scdata = loadReconstructedTreeAndData(s['args'], s['folder'], all_ranks=False)
	for node in scdata.tree.root.getNodeList([], returnRoot=False):
		node.tParent = 0.3 if node.isLeaf else 0.2
	scdata.storeTreeInFolder(scdata.result_path('reconstructed'))
	frame = getMetadata(s['args'], scdata, 'reconstructed', 1.0)
	all_means, all_vars = sim.bonsai_units()

	def scaled(log_s):
		tree = truth.copy()
		for node in tree.iter_nodes():
			if node.length is not None:
				node.length *= math.exp(log_s)
		return -reference_bonsai_loglik(tree, all_means, all_vars)

	true_loglik = -minimize_scalar(scaled, bounds=(-5, 5), method='bounded', options={'xatol': 1e-10}).fun
	full_loglik = _branch_optimum(tree_from_folder(scdata.result_path('reconstructed')), all_means, all_vars)
	row = frame.loc['metadata']
	# One scalar: Bonsai's optimum agrees to 3e-14 relative. All branch lengths: Bonsai's optimiser stops 6e-8
	# relative below the optimum on this tree; it can never be above it.
	assert row['trueLoglik'] == pytest.approx(true_loglik, rel=1e-9)
	assert row['fullLoglik'] == pytest.approx(full_loglik, rel=1e-6)
	assert row['fullLoglik'] <= full_loglik + 1e-9 * abs(full_loglik)


# ------------------------------------------------------------------ posteriors read back

def test_get_bonsai_posteriors_returns_genes_by_requested_vertex(stored_folder, caplog):
	"""get_bonsai_posteriors gives the posterior means and variances as genes x vertices, in the order of the IDs
	asked for; an unknown ID is reported and gets a column of zeros."""
	s = stored_folder
	means, variances = posteriors_by_name(s['folder'])
	ids = ['C101', 'root', 'C000', 'internal_2', 'no_such_vertex']
	got_means, got_vars = get_bonsai_posteriors(s['folder'], ids)
	assert got_means.shape == got_vars.shape == (len(s['sim'].signal_genes), len(ids))
	for col, name in enumerate(ids[:-1]):
		np.testing.assert_array_equal(got_means[:, col], means[name])
		np.testing.assert_array_equal(got_vars[:, col], variances[name])
	assert not got_means[:, -1].any() and not got_vars[:, -1].any()
	assert 'ERROR' in caplog.text


def test_get_bonsai_euclidean_distances_are_mean_squared_differences(stored_folder):
	s = stored_folder
	means, _ = posteriors_by_name(s['folder'])
	cells = s['sim'].cell_ids
	got = get_bonsai_euclidean_distances(s['folder'], cells)
	expected = [np.mean((means[a] - means[b]) ** 2) for i, a in enumerate(cells) for b in cells[i + 1:]]
	np.testing.assert_allclose(got, expected, rtol=1e-12)


# ------------------------------------------------------------------ annotations and UMI counts

def _annotation_scdata(cell_ids, n_genes=3):
	scdata = SCData(onlyObject=True, dataset='bonsai_test', results_folder='results')
	scdata.metadata.cellIds = list(cell_ids)
	scdata.metadata.nCells = len(cell_ids)
	scdata.metadata.nGenes = n_genes
	return scdata


CELLS = ['c0', 'c1', 'c2', 'c3', 'c4']


def _write(path, text):
	os.makedirs(os.path.dirname(str(path)), exist_ok=True)
	with open(str(path), 'w') as handle:
		handle.write(text)


def test_get_annotations_orders_rows_by_cell_and_types_columns(tmp_path):
	"""Annotation CSVs (README, 'Preprocessing for the visualization'): a cell-ID column and one column per
	annotation; numerical columns become 'annot_num_<name>', the others categorical 'annot_<name>' with
	missing values as 'NaN'; rows follow metadata.cellIds whatever their order in the file. A file whose name
	starts with 'mat_' is a feature matrix, returned transposed (features x cells)."""
	folder = tmp_path / 'annotation'
	_write(folder / 'cells.csv', 'cellID,experimenter,measurement\n'
	       'c3,T,0.5\nc0,A,0.7\nc4,,1.5\nc1,T,0.25\nc2,A,2\n')
	_write(folder / 'mat_counts.csv', 'cellID,geneA,geneB\nc1,1,10\nc0,0,9\nc2,2,8\nc3,3,7\nc4,4,6\n')
	_write(folder / 'ignored.txt', 'not an annotation')
	frame, matrices = _annotation_scdata(CELLS).get_annotations(str(folder))
	assert list(frame.index) == CELLS
	assert list(frame.columns) == ['annot_experimenter', 'annot_num_measurement']
	assert list(frame['annot_experimenter'].astype(str)) == ['A', 'T', 'A', 'T', 'NaN']
	assert isinstance(frame['annot_experimenter'].dtype, pd.CategoricalDtype)
	np.testing.assert_array_equal(frame['annot_num_measurement'].values, [0.7, 0.25, 2.0, 0.5, 1.5])
	assert list(matrices) == ['counts']
	assert list(matrices['counts'].columns) == CELLS and list(matrices['counts'].index) == ['geneA', 'geneB']
	np.testing.assert_array_equal(matrices['counts'].values, [[0, 1, 2, 3, 4], [9, 10, 8, 7, 6]])


def test_get_annotations_without_a_header_line(tmp_path):
	"""A file with one row per cell and no header (so one row fewer once the first is taken as a header) is
	re-read without one, with generated column names."""
	path = tmp_path / 'annotation' / 'plain.csv'
	_write(path, ''.join('%s,%s\n' % (cell, 'x' if ind % 2 else 'y') for ind, cell in enumerate(CELLS)))
	frame, _ = _annotation_scdata(CELLS).get_annotations(str(path))
	assert list(frame.columns) == ['annot_Annotation 0_0']
	assert list(frame.iloc[:, 0].astype(str)) == ['y', 'x', 'y', 'x', 'y']


def test_get_annotations_default_when_there_is_none(tmp_path):
	frame, matrices = _annotation_scdata(CELLS).get_annotations(str(tmp_path / 'absent'))
	assert matrices == {} and list(frame.index) == CELLS and list(frame.columns) == ['annot_default']
	assert set(frame['annot_default'].astype(str)) == {'default'}


def test_get_annotations_with_cs_cells_cellstates_and_derived_columns(tmp_path):
	"""get_annotations_with_cs (used by bonsai_scout_preprocess.py) sorts each file to cells or to cellstates by
	its number of rows, reads .csv and .tsv, drops all-empty columns, makes column names identifiers, keeps a
	user column starting with 'num_' apart from the numerical ones, and adds a categorical copy
	('annot_categorized_<name>', values 'c_<value>') of each numerical column with fewer than 20 values."""
	folder = tmp_path / 'annotation'
	_write(folder / 'cells.tsv', 'cellID\tcell type\tnum_x\tscore-1.a\tempty\n'
	       'c4\tB\tp\t3\t\nc0\tA\tq\t1\t\nc1\tA\tq\t2\t\nc2\tB\tp\t2\t\nc3\tB\tq\t1\t\n')
	_write(folder / 'states.csv', 'csID,group\ns1,G1\ns0,G0\ns2,G1\n')
	scdata = _annotation_scdata(CELLS)
	scdata.metadata.csIds = ['s0', 's1', 's2']
	scdata.metadata.nCss = 3
	cell_frame, cs_frame, matrices = scdata.get_annotations_with_cs(str(folder))
	assert matrices == {}
	assert list(cell_frame.index) == CELLS
	assert list(cell_frame.columns) == ['annot_cell_type', 'annot_usernum_x', 'annot_num_score_1_a',
	                                    'annot_categorized_score_1_a']
	assert list(cell_frame['annot_cell_type'].astype(str)) == ['A', 'A', 'B', 'B', 'B']
	np.testing.assert_array_equal(cell_frame['annot_num_score_1_a'].values, [1.0, 2.0, 2.0, 1.0, 3.0])
	assert list(cell_frame['annot_categorized_score_1_a']) == ['c_1.0', 'c_2.0', 'c_2.0', 'c_1.0', 'c_3.0']
	assert list(cs_frame.index) == ['s0', 's1', 's2'] and list(cs_frame.columns) == ['annot_group']
	assert list(cs_frame['annot_group'].astype(str)) == ['G0', 'G1', 'G1']


def test_get_annotations_with_cs_without_cellstates(tmp_path):
	"""Without cellstates bonsai_scout_preprocess.py sets the cellstates to the cells; a file with a row per cell
	then lands in the cellstate frame, the cell frame gets the default annotation. A file covering only some cells
	is kept, with missing values for the others."""
	folder = tmp_path / 'annotation'
	_write(folder / 'cells.csv', 'cellID,kind\nc2,u\nc0,v\nc1,u\nc4,w\nc3,v\n')
	_write(folder / 'partial.csv', 'cellID,batch\nc1,b1\nc3,b2\n')
	scdata = _annotation_scdata(CELLS)
	scdata.metadata.csIds = list(CELLS)
	scdata.metadata.nCss = len(CELLS)
	cell_frame, cs_frame, _ = scdata.get_annotations_with_cs(str(folder))
	assert list(cell_frame.columns) == ['annot_default']
	assert list(cs_frame.index) == CELLS
	assert list(cs_frame['annot_kind'].astype(str)) == ['v', 'u', 'u', 'v', 'w']
	assert list(cs_frame['annot_batch'].astype(str)) == ['NaN', 'b1', 'NaN', 'b2', 'NaN']


def test_get_celltypes_minim_reads_one_annotation_file(tmp_path):
	path = tmp_path / 'annotation.csv'
	_write(path, 'cellID,label,value\nc2,x,1.5\nc0,y,2.5\nc1,x,0.5\nc4,y,1\nc3,x,3\n')
	frame = _annotation_scdata(CELLS).get_celltypes_minim(str(path))
	assert list(frame.columns) == ['annot_label', 'annot_num_value']
	assert sorted(frame.index) == CELLS
	assert {cell: str(frame.loc[cell, 'annot_label']) for cell in CELLS} == \
		{'c0': 'y', 'c1': 'x', 'c2': 'x', 'c3': 'x', 'c4': 'y'}
	assert {cell: frame.loc[cell, 'annot_num_value'] for cell in CELLS} == \
		{'c0': 2.5, 'c1': 0.5, 'c2': 1.5, 'c3': 3.0, 'c4': 1.0}
	default = _annotation_scdata(CELLS).get_celltypes_minim(str(tmp_path / 'absent.csv'))
	assert list(default.index) == CELLS and list(default.columns) == ['annot_default']


def test_get_celltype_annotations_labels_each_vertex(stored_folder, tmp_path):
	"""Celltype*.txt files in the data folder hold one label per cell (in cellID order). Every vertex gets the
	label of its cells (several types joined by '++', most frequent first), 'internal_node' without cells; the
	table is written per vertex name, one column per file."""
	s = stored_folder
	cells = s['sim'].cell_ids
	data = s['args'].data_folder
	_write(os.path.join(data, 'Celltype_a.txt'), ''.join('%s\n' % ('T%d' % (ind % 3)) for ind in range(len(cells))))
	_write(os.path.join(data, 'Celltype_b.txt'), ''.join('%s\n' % ('left' if ind < 4 else 'right')
	                                                     for ind in range(len(cells))))
	scdata, vert_to_id = loadReconstructedTreeAndData(s['args'], s['folder'], all_ranks=False, get_cell_info=True)
	scdata.metadata.pathToOrigData = data
	# Put cells 0, 3 and 6 (types T0, T0, T0) and 1 (T1) together on the root to see a mixed vertex.
	root_vert = [vert for vert, name in vert_to_id.items() if name == 'root'][0]
	for cell in (0, 1, 3, 6):
		scdata.vertIndToCellInds[scdata.cellIndToVertInd[cell]].remove(cell)
	scdata.vertIndToCellInds[root_vert] = [0, 1, 3, 6]
	out = str(tmp_path / 'celltypes.csv')
	scdata.get_celltype_annotations(celltype_metadata_path=out, vertIndToNodeId=vert_to_id)
	table = pd.read_csv(out, index_col=0)
	assert table.index.name == 'node_id' and list(table.columns) == ['Celltype_a', 'Celltype_b']
	assert list(table.index) == [vert_to_id[vert] for vert in range(len(vert_to_id))]
	assert table.loc['root', 'Celltype_a'] == 'T0++T1'
	assert table.loc['root', 'Celltype_b'] == 'left++right'
	for ind in (2, 4, 5, 7):
		assert table.loc[cells[ind], 'Celltype_a'] == 'T%d' % (ind % 3)
		assert table.loc[cells[ind], 'Celltype_b'] == ('left' if ind < 4 else 'right')
	for ind in (0, 1, 3, 6):
		assert table.loc[cells[ind], 'Celltype_a'] == 'internal_node'
	assert (table.loc[[name for name in table.index if name.startswith('internal_')]] == 'internal_node').all().all()


def test_read_umi_counts_drops_all_zero_genes_to_match(tmp_path):
	scdata = _annotation_scdata(CELLS, n_genes=2)
	scdata.metadata.pathToOrigData = str(tmp_path)
	_write(tmp_path / 'Gene_table.txt', 'gene\t' + '\t'.join(CELLS) + '\n'
	       'g1\t1\t0\t2\t0\t5\ng0\t0\t0\t0\t0\t0\ng2\t3\t1\t0\t0\t1\n')
	scdata.read_umi_counts()
	np.testing.assert_array_equal(scdata.umiCounts, [[1, 0, 2, 0, 5], [3, 1, 0, 0, 1]])
	scdata.metadata.nGenes = 1
	with pytest.raises(SystemExit):
		scdata.read_umi_counts()
	scdata.metadata.pathToOrigData = str(tmp_path / 'absent')
	scdata.read_umi_counts()
	assert scdata.umiCounts is None


# ------------------------------------------------------------------ the files of a full run

area_a_small_run = bonsai_run_fixture('area_a_small_run', small_run_spec())


def test_full_run_writes_the_results_the_readme_describes(area_a_small_run):
	"""README, 'The Bonsai results': a final_bonsai_ folder with tree.nwk, edgeInfo.txt, vertInfo.txt,
	metadata.json and the two posterior files; intermediate_bonsai_ folders without their '_vertByGene.npy'
	files; the configuration file."""
	run = area_a_small_run.check()
	final = run.final_folder()
	assert sorted(os.listdir(final)) == ['edgeInfo.txt', 'metadata.json', 'metadata_bonsai.txt',
	                                     'posterior_ltqsVars_vertByGene.npy', 'posterior_ltqs_vertByGene.npy',
	                                     'tree.nwk', 'vertInfo.txt']
	intermediates = [name for name in os.listdir(run.results_folder) if name.startswith('intermediate_bonsai_')]
	assert stage_folder('reordered_edges') in intermediates
	for name in intermediates:
		assert not [f for f in os.listdir(run.path(name)) if f.endswith('_vertByGene.npy')], name
		assert {'edgeInfo.txt', 'vertInfo.txt', 'metadata.json'} <= set(os.listdir(run.path(name)))


def test_full_run_metadata_json(area_a_small_run):
	"""metadata.json of the final folder: the cells and the genes that passed the cutoff, their variances, the
	rescaling and its log-likelihood correction, and absolute paths to the input, the results and the processed
	data; its loglik is the final log-likelihood with that correction included."""
	run = area_a_small_run.check()
	sim = run.sim
	keep = _signal(sim)
	md = read_metadata_json(run.final_folder())
	assert set(md) == set(FIELDS)
	assert md['cellIds'] == sim.cell_ids and md['nCells'] == sim.n_cells
	assert md['geneIds'] == [sim.gene_ids[g] for g in keep] and md['nGenes'] == len(keep)
	np.testing.assert_allclose(md['geneVariances'], sim.gene_variances[keep], rtol=1e-15)
	assert md['geneDiffusionScaling'] == 'geneVariances' and md['dataset'] == 'bonsai_test'
	assert md['loglikVarCorr'] == pytest.approx(-(sim.n_cells - 1) * np.sum(np.log(sim.gene_variances[keep])),
	                                            rel=1e-12)
	assert md['pathToOrigData'] == os.path.abspath(run.data_folder)
	assert md['results_folder'] == run.results_folder
	assert md['processedDatafolder'] == run.path('zscorefiltered_1.000_and_processed')
	final_loglik = float(read_metadata_bonsai(run.final_folder())['finalLoglik'])
	assert md['loglik'] - md['loglikVarCorr'] == pytest.approx(final_loglik, rel=1e-9)


def test_full_run_log_likelihood_and_posteriors_match_the_reference(area_a_small_run):
	"""metadata_bonsai.txt and the posterior files on the final tree (tree.nwk, full precision), against the dense
	Gaussian reference on the generator's measurements of the kept genes."""
	run = area_a_small_run.check()
	final = run.final_folder()
	keep = _signal(run.sim)
	means, variances = run.sim.bonsai_units(keep)
	tree = read_newick_file(final)
	row = read_metadata_bonsai(final)
	assert list(row) == ['nGenes', 'nCells', 'reconsTime', 'zscoreCutoff', 'coresUsed', 'finalLoglik', 'trueLoglik',
	                     'fullLoglik', 'remainingRootChildren']
	assert (int(row['nGenes']), int(row['nCells']), float(row['zscoreCutoff']), int(row['coresUsed'])) == \
		(len(keep), run.sim.n_cells, 1.0, 1)
	assert float(row['reconsTime']) > 0 and row['trueLoglik'] == '' and row['fullLoglik'] == ''
	assert int(row['remainingRootChildren']) == len(tree.children)
	assert float(row['finalLoglik']) == pytest.approx(reference_bonsai_loglik(tree, means, variances), rel=RTOL)
	ref_means, ref_vars = reference_posteriors(tree, means, variances)
	got_means, got_vars = posteriors_by_name(final)
	scale = run.sim.gene_variances[keep]
	assert set(got_means) == set(ref_means)
	for name in ref_means:
		np.testing.assert_allclose(got_means[name], ref_means[name] * np.sqrt(scale), rtol=RTOL)
		np.testing.assert_allclose(got_vars[name], ref_vars[name] * scale, rtol=RTOL)


@pytest.mark.parametrize('reprocess', [False, True])
def test_full_run_final_folder_loads_back(area_a_small_run, reprocess):
	"""The final folder loads back in-process (as bonsai_scout_preprocess.py does) with the log-likelihood the run
	reported, whether the data come from the processed-data folder or are read again from the input."""
	run = area_a_small_run.check()
	args = make_args(run.data_folder, run.results_folder, input_is_sanity_output=True,
	                 **{key: value for key, value in read_config(run.config).items()
	                    if key in ('zscore_cutoff', 'rescale_by_var')})
	scdata = loadReconstructedTreeAndData(args, os.path.basename(run.final_folder()), reprocess_data=reprocess,
	                                      all_ranks=False, rel_to_results=True, calc_loglik=True)
	md = read_metadata_json(run.final_folder())
	final_loglik = float(read_metadata_bonsai(run.final_folder())['finalLoglik'])
	# The loaded tree has edgeInfo.txt's lengths (%.8e), so agreement is to about 1e-8.
	assert scdata.metadata.loglik - md['loglikVarCorr'] == pytest.approx(final_loglik, rel=1e-7)
	assert rf_distance(tree_from_folder(run.final_folder()), read_newick_file(run.final_folder())) == 0


# ------------------------------------------------------------------ read_and_filter over MPI ranks

from area_a.helpers_a import MPI_JOBS  # noqa: E402
from bonsai_testfixtures import bonsai_job_fixture  # noqa: E402

area_a_mpi2 = bonsai_job_fixture('area_a_mpi2', MPI_JOBS[2], slots=2, mpi=True)
area_a_mpi4 = bonsai_job_fixture('area_a_mpi4', MPI_JOBS[4], slots=4, mpi=True)


@pytest.mark.mpi
@pytest.mark.parametrize('ranks', [2, 4])
@pytest.mark.parametrize('layout', ['sanity', 'features'])
def test_read_and_filter_over_mpi_ranks_matches_one_process(area_a_mpi2, area_a_mpi4, tmp_path, ranks, layout):
	"""Under 'mpiexec -n 2' and '-n 4' every rank reads its share of the genes; rank 0 ends up with what one
	process reads. (Both runs are fixture arguments so that they start with the session's other runs.)"""
	job = {2: area_a_mpi2, 4: area_a_mpi4}[ranks]
	assert job['returncode'] == 0, job['log'][-4000:]
	got = np.load(os.path.join(job['out'], 'result.npz'))
	if layout == 'sanity':
		folder, _ = dataset(tmp_path)
		serial = read_and_filter(folder, None, None, True, 1.0, SERIAL, tmp_folder=str(tmp_path / 'tmp'))
	else:
		folder, _ = dataset(tmp_path, FEATURES)
		serial = read_and_filter(folder, 'features.txt', 'standard_deviations.txt', False, 1.0, SERIAL,
		                         tmp_folder=str(tmp_path / 'tmp'))
	names = ('ltqs', 'ltqs_vars', 'gene_vars', 'n_cells', 'n_genes', 'genes_to_keep', 'stds_found', 'n_genes_orig')
	for name, expected in zip(names, serial):
		value = got[layout + '_' + name]
		if name in ('ltqs', 'ltqs_vars', 'gene_vars'):
			np.testing.assert_allclose(value, expected, rtol=1e-12, err_msg=name)
		else:
			np.testing.assert_array_equal(value, expected, err_msg=name)
