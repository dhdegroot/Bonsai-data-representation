"""
The non-tree utilities of bonsai/bonsai_helpers.py (area A): Run_Configs, the results-folder names and the
helpers that find, store and clean them up, MPI start-up and broadcasting helpers, task splitting, small file
readers and writers, the Sanity version helpers, recursion limits, and the Poincare-disk helpers of the tree
visualisation. The tree utilities of this module are area B's.

All in-process and single-process; MPI communicators are replaced by helpers_a.FakeComm where a test needs
another rank.
"""
import csv
import logging
import math
import os
import re
import stat
import sys
import time
import tracemalloc
from argparse import ArgumentTypeError, Namespace

import numpy as np
import pytest
from matplotlib.colors import ListedColormap
from ruamel.yaml import YAML
from scipy.optimize import minimize

import bonsai.bonsai_globals as bs_glob
import bonsai.bonsai_helpers as helpers
import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_helpers import (InteractionEvent, RecursionLimit, Run_Configs, SANITY_FILENAMES,
                                   SANITY_LEGACY_FILENAMES, broadcastRecursiveStruct, checkGrad,
                                   clean_up_redundant_data_files, communicateGlobalVars, convert_dict_to_named_tuple,
                                   create_celltypefile_from_cellstates, empty_folder, find_latest_tree_folder,
                                   find_latest_tree_folder_new, get_celltype_colors, get_centroid_poincare,
                                   get_latest_intermediate, get_opt_zoom_origin_poincare, get_radial_poincare,
                                   get_sanity_binary_version, get_zoom_given_origin, getMyTaskNumbers,
                                   getOutputFolder, invert_coords_poincare, is_numeric, look_for_output_folder,
                                   mp_print, parse_sanity_command_file, print_memory, printTiming,
                                   read_ids, readListFromTxt, recursionWrap, remove_folder, remove_tree_folders,
                                   removePickledTrees, resolve_sanity_filepaths, set_recursion_limits, startMPI,
                                   store_scdata_and_communicate_path, str2bool, time_format,
                                   transform_coords_poincare, write_ids, writeVectToEndCsv)
from bonsai_testfixtures import bonsai_run_fixture
from bonsai_testrun import read_config, stage_folder

from area_a.helpers_a import (FakeComm, cwd_in_tmp, make_args, scdata_with_tree,  # noqa: F401
                              small_run_spec)

STAGES = ('preprocessed', 'greedy', 'redo_starry', 'opt_times', 'spr_moves', 'nnn_reorder', 'reordered_edges',
          'final')
YAML_KEYS = ('dataset', 'data_folder', 'filenames_data', 'results_folder', 'verbose', 'input_is_sanity_output',
             'zscore_cutoff', 'rescale_by_var', 'nnn_n_randommoves', 'nnn_n_randomtrees', 'use_knn',
             'UB_ellipsoid_size', 'spr_strategy', 'pickup_intermediate', 'tmp_folder')


@pytest.fixture(autouse=True)
def _single_process(cwd_in_tmp):
	assert mpi_wrapper.comm is None
	yield


# ------------------------------------------------------------------ Run_Configs

def _write_yaml(path, values):
	yaml = YAML()
	with open(str(path), 'w') as handle:
		yaml.dump(dict(values), handle)
	return str(path)


FULL_CONFIG = dict(start_time=None, dataset='ds', data_folder='/in', filenames_data='f.txt,s.txt',
                   results_folder='/out', verbose=False, input_is_sanity_output=False, zscore_cutoff=0.5,
                   rescale_by_var=False, nnn_n_randommoves=7, nnn_n_randomtrees=3, use_knn=-1,
                   UB_ellipsoid_size=-1.0, spr_strategy='super_sure', pickup_intermediate=True, tmp_folder='/tmp_tree')


def test_run_configs_reads_every_key_of_the_yaml(tmp_path):
	config = Run_Configs(_write_yaml(tmp_path / 'c.yaml', FULL_CONFIG))
	for key in YAML_KEYS:
		assert getattr(config, key) == FULL_CONFIG[key], key
	assert config.step == 'all'
	assert re.fullmatch(r'\d\d\.\d\d\.\d\d_\d\d:\d\d:\d\d', config.config_yaml['start_time'])


def test_run_configs_step_argument_wins(tmp_path):
	config = Run_Configs(_write_yaml(tmp_path / 'c.yaml', dict(FULL_CONFIG, step='core_calc')), step='metadata')
	assert config.step == 'metadata'
	assert Run_Configs(_write_yaml(tmp_path / 'd.yaml', dict(FULL_CONFIG, step='core_calc'))).step == 'core_calc'


def test_run_configs_defaults_for_missing_and_empty_keys(tmp_path):
	"""Keys absent from the YAML, or present without a value, take Run_Configs' defaults; for these keys they
	agree with the defaults the template's comments state. (tmp_folder, filenames_data, data_folder and
	results_folder do not: Run_Configs gives tmp_folder, data_folder and results_folder None where the comments
	state '', and filenames_data 'delta.txt,d_delta.txt' where they state features.txt,standard_deviations.txt.)"""
	config = Run_Configs(_write_yaml(tmp_path / 'c.yaml', dict(dataset='ds', data_folder='/in',
	                                                          results_folder='/out', zscore_cutoff=None)))
	expected = dict(verbose=True, input_is_sanity_output=True, zscore_cutoff=1.0, rescale_by_var=True,
	                nnn_n_randommoves=1000, nnn_n_randomtrees=10, use_knn=10, UB_ellipsoid_size=1.0,
	                spr_strategy='large_tree', pickup_intermediate=False, step='all')
	for key, value in expected.items():
		assert getattr(config, key) == value, key


@pytest.mark.parametrize('key', ['skip_greedy_merging', 'skip_redo_starry', 'skip_opt_times', 'skip_spr_moves',
                                 'skip_nnn_reordering', 'skip_reorder_edges'])
def test_run_configs_ignores_deprecated_skip_keys(tmp_path, key, caplog):
	"""create_config_file.py: the skip_* arguments are deprecated and have no effect any more."""
	config = Run_Configs(_write_yaml(tmp_path / 'c.yaml', dict(FULL_CONFIG, **{key: True})))
	assert getattr(config, key) is False
	assert key in caplog.text


def test_run_configs_copies_named_arguments(tmp_path):
	args = Namespace(use_knn=3, not_a_config='x')
	config = Run_Configs(_write_yaml(tmp_path / 'c.yaml', FULL_CONFIG), args=args,
	                     args_to_copy=['use_knn', 'not_a_config', 'missing'])
	assert config.use_knn == 3 and config.not_a_config == 'x' and not hasattr(config, 'missing')


def test_run_configs_without_a_yaml_file_stops(tmp_path):
	with pytest.raises(SystemExit):
		Run_Configs(str(tmp_path / 'absent.yaml'))
	with pytest.raises(SystemExit):
		Run_Configs(None)


def test_run_configs_empty_has_the_defaults():
	config = Run_Configs(create_empty_configs=True)
	assert (config.step, config.dataset, config.zscore_cutoff, config.use_knn, config.spr_strategy) == \
		('all', 'new_dataset', 1.0, 10, 'large_tree')


def test_store_yaml_writes_the_configuration_with_its_start_time(tmp_path):
	config = Run_Configs(_write_yaml(tmp_path / 'c.yaml', FULL_CONFIG))
	out = tmp_path / 'stored.yaml'
	config.store_yaml(str(out))
	stored = read_config(out)
	assert stored['start_time'] == config.config_yaml['start_time']
	for key in YAML_KEYS:
		assert stored[key] == FULL_CONFIG[key], key
	config.store_yaml(None)


area_a_small_run = bonsai_run_fixture('area_a_small_run', small_run_spec())


def test_full_run_keeps_the_configuration_and_cleans_intermediate_data(area_a_small_run):
	"""README: the results folder holds 'a copy of the configuration YAML-file with an added timestamp', and
	the intermediate_bonsai_ folders lose their '_vertByGene.npy' files at the end."""
	run = area_a_small_run.check()
	used = read_config(run.path('used_run_configs.yaml'))
	given = read_config(run.config)
	assert set(used) == set(given)
	for key in YAML_KEYS:
		assert used[key] == given[key], key
	assert re.fullmatch(r'\d\d\.\d\d\.\d\d_\d\d:\d\d:\d\d', used['start_time'])
	with open(run.path('used_run_configs.yaml')) as handle:
		assert '# zscore_cutoff [float, default=1]' in handle.read()


# ------------------------------------------------------------------ str2bool, small converters

@pytest.mark.parametrize('text,value', [('yes', True), ('TRUE', True), ('t', True), ('Y', True), ('1', True),
                                        ('no', False), ('False', False), ('f', False), ('N', False), ('0', False)])
def test_str2bool(text, value):
	assert str2bool(text) is value


def test_str2bool_rejects_other_text():
	with pytest.raises(ArgumentTypeError):
		str2bool('maybe')


def test_convert_dict_to_named_tuple():
	nt = convert_dict_to_named_tuple({'b': 2, 'a': 1})
	assert nt._fields == ('a', 'b') and (nt.a, nt.b) == (1, 2)


@pytest.mark.parametrize('value,expected', [('1', True), ('-2.5e3', True), ('nan', True), (' 4 ', True),
                                            (3, True), ('x1', False), ('', False), ('1,5', False)])
def test_is_numeric(value, expected):
	assert is_numeric(value) is expected


def test_time_format():
	assert re.fullmatch(r'\d\d:\d\d:\d\d\|> ', time_format())


# ------------------------------------------------------------------ mp_print and MPI start-up

def test_mp_print_logs_on_rank_zero_at_the_requested_level(caplog):
	caplog.set_level(logging.DEBUG, logger='myapp')
	mp_print('plain', 1, 2.5)
	mp_print('bad', ERROR=True)
	mp_print('careful', WARNING=True)
	mp_print('detail', DEBUG=True)
	mp_print('everyone', ALL_RANKS=True)
	mp_print('rank zero only', ONLY_RANK=0)
	mp_print('rank one only', ONLY_RANK=1)
	records = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == 'myapp']
	assert records == [('INFO', 'plain 1 2.5'), ('ERROR', 'bad'), ('WARNING', 'careful'), ('DEBUG', 'detail'),
	                   ('INFO', 'Process 0: everyone'), ('INFO', 'Process 0: rank zero only')]


def test_mp_print_on_another_rank_is_silent_unless_asked(monkeypatch, caplog):
	monkeypatch.setattr(mpi_wrapper, 'comm', FakeComm(rank=2, size=4))
	mp_print('plain')
	mp_print('everyone', ALL_RANKS=True)
	mp_print('mine', ONLY_RANK=2)
	assert [r.getMessage() for r in caplog.records if r.name == 'myapp'] == ['Process 2: everyone',
	                                                                           'Process 2: mine']


def test_start_mpi_without_mpi4py_is_one_process(monkeypatch):
	monkeypatch.setitem(sys.modules, 'mpi4py', None)
	monkeypatch.setattr(sys, 'platform', 'linux')
	assert startMPI(verbose=False) == (0, 1)
	assert mpi_wrapper.comm is None


def test_start_mpi_reports_rank_and_size(monkeypatch, caplog):
	monkeypatch.setattr(mpi_wrapper, 'mpi_init', lambda: setattr(mpi_wrapper, 'comm', FakeComm(rank=3, size=4)))
	assert startMPI(verbose=True) == (3, 4)
	assert 'Process 3' in caplog.text


class _RecordingRoot(FakeComm):
	"""Rank 0 of two: every broadcast keeps its own data and is recorded."""

	def bcast(self, data, root=0):
		self.broadcast.append(data)
		return data


class _ReplayingRank(FakeComm):
	"""Rank 1 of two: the i-th broadcast returns what rank 0 sent in its i-th broadcast."""

	def __init__(self, sent):
		super().__init__(rank=1, size=2)
		self.sent = list(sent)

	def bcast(self, data, root=0):
		return self.sent.pop(0)


def test_communicate_global_vars_takes_every_global_from_the_root(monkeypatch):
	"""Every name in bonsai_globals.globalVars ends up on the other ranks with rank 0's value, whatever the number
	and packing of the broadcasts (rank 0's broadcasts are recorded and replayed on rank 1)."""
	root_values = {name: ('root', ind) for ind, name in enumerate(sorted(bs_glob.globalVars))}
	for name, value in root_values.items():
		setattr(bs_glob, name, value)
	root = _RecordingRoot(rank=0, size=2)
	monkeypatch.setattr(mpi_wrapper, 'comm', root)
	communicateGlobalVars()
	assert {name: getattr(bs_glob, name) for name in root_values} == root_values
	for name in root_values:
		setattr(bs_glob, name, ('rank 1', name))
	monkeypatch.setattr(mpi_wrapper, 'comm', _ReplayingRank(root.broadcast))
	communicateGlobalVars()
	assert {name: getattr(bs_glob, name) for name in root_values} == root_values
	monkeypatch.setattr(mpi_wrapper, 'comm', None)
	bs_glob.nCells = 5
	communicateGlobalVars()
	assert bs_glob.nCells == 5


def test_broadcast_recursive_struct_retries_with_a_raised_recursion_limit(monkeypatch):
	"""Pickling a deep tree for a broadcast can hit the recursion limit; the broadcast is retried under
	RecursionLimit, and the limit is restored afterwards."""

	class DeepComm(FakeComm):
		def bcast(self, data, root=0):
			if sys.getrecursionlimit() < 10 ** 6:
				raise RecursionError('too deep')
			return ('from_root', data)

	limit = sys.getrecursionlimit()
	monkeypatch.setattr(mpi_wrapper, 'comm', DeepComm())
	assert broadcastRecursiveStruct({'tree': 1}) == ('from_root', {'tree': 1})
	assert sys.getrecursionlimit() == limit
	monkeypatch.setattr(mpi_wrapper, 'comm', None)
	payload = [1, 2]
	assert broadcastRecursiveStruct(payload) is payload


@pytest.mark.parametrize('n_tasks,size', [(10, 3), (16, 4), (5, 7), (0, 2), (1, 1)])
def test_get_my_task_numbers_splits_tasks_into_contiguous_balanced_blocks(n_tasks, size):
	"""Every task goes to exactly one rank, in rank order, and block sizes differ by at most one; skippingSteps
	deals tasks round-robin instead."""
	blocks = [getMyTaskNumbers(n_tasks, size, rank) for rank in range(size)]
	assert np.concatenate(blocks).tolist() == list(range(n_tasks))
	sizes = [len(block) for block in blocks]
	assert max(sizes) - min(sizes) <= 1
	for rank in range(size):
		assert getMyTaskNumbers(n_tasks, size, rank, skippingSteps=True).tolist() == list(range(rank, n_tasks, size))


def test_print_timing_returns_the_current_time(monkeypatch, caplog):
	monkeypatch.setattr(time, 'time', lambda: 105.5)
	assert printTiming('the step', 100.0) == 105.5
	assert 'the step' in caplog.text and '5.5' in caplog.text


# ------------------------------------------------------------------ results folders

def test_output_folder_names_of_the_stages():
	"""The folder names --pickup_intermediate looks for (README, 'The Bonsai results'), as listed in the harness
	(bonsai_testrun.stage_folder)."""
	assert getOutputFolder(zscore_cutoff=1.0, greedy=False, redo_starry=False, opt_times=False) == \
		stage_folder('preprocessed')
	assert getOutputFolder(zscore_cutoff=1.0, redo_starry=False, opt_times=False) == stage_folder('greedy')
	assert getOutputFolder(zscore_cutoff=1.0, opt_times=False) == stage_folder('redo_starry')
	assert getOutputFolder(zscore_cutoff=1.0) == stage_folder('opt_times')
	assert getOutputFolder(zscore_cutoff=1.0, spr_moves=True) == stage_folder('spr_moves')
	assert getOutputFolder(zscore_cutoff=1.0, spr_moves=True, nnn_reorder=True) == stage_folder('nnn_reorder')
	assert getOutputFolder(zscore_cutoff=1.0, spr_moves=True, nnn_reorder=True, reorderedEdges=True) == \
		stage_folder('reordered_edges')
	assert getOutputFolder(zscore_cutoff=1.0, final=True, spr_moves=True, reorderedEdges=True) == \
		stage_folder('final')
	assert getOutputFolder(zscore_cutoff=2.5, final=True) == 'final_bonsai_zscore2.5'
	assert getOutputFolder(zscore_cutoff=-1, final=True) == 'final_bonsai'
	assert getOutputFolder(zscore_cutoff=0, greedy=False, redo_starry=False, opt_times=False) == 'preprocessed'


def test_output_folder_names_with_a_start_tree():
	assert getOutputFolder(zscore_cutoff=1.0, final=True, tmp_file='mytree') == 'final_bonsai_zscore1.0_tmpStartmytree'
	assert getOutputFolder(zscore_cutoff=1.0, greedy=False, redo_starry=False, opt_times=False, tmp_file='t') == \
		'preprocessed_zscore1.0_tmpStartt'


def test_output_folder_all_possibilities_are_the_intermediate_folders():
	assert getOutputFolder(zscore_cutoff=1.0, get_all_possibilities=True) == [
		stage_folder(stage) for stage in ('greedy', 'redo_starry', 'opt_times', 'spr_moves', 'nnn_reorder',
		                                  'reordered_edges')]
	assert getOutputFolder(zscore_cutoff=1.0, tmp_file='x', get_all_possibilities=True)[-1] == \
		stage_folder('reordered_edges') + '_tmpStartx'


def _make_stage_folders(results, stages, zscore_cutoff=1.0, with_vert_info=True):
	for stage in stages:
		folder = results / stage_folder(stage, zscore_cutoff)
		folder.mkdir(parents=True, exist_ok=True)
		if with_vert_info:
			(folder / 'vertInfo.txt').write_text('vertInd\tnodeInd\tvertName\n')


SKIPS = ('skip_greedy_merging', 'skip_redo_starry', 'skip_opt_times', 'skip_spr_moves', 'skip_nnn_reordering',
         'skip_reorder_edges')


@pytest.mark.parametrize('done', range(len(STAGES)))
def test_find_latest_tree_folder_new_picks_the_furthest_stage(tmp_path, done):
	"""--pickup_intermediate resumes from the furthest stage whose folder holds a vertInfo.txt, and skips the
	stages already done."""
	results = tmp_path / 'results'
	_make_stage_folders(results, STAGES[:done + 1])
	_make_stage_folders(results, STAGES[done + 1:], with_vert_info=False)
	args = make_args(tmp_path / 'data', results, zscore_cutoff=1.0)
	assert find_latest_tree_folder_new(args, str(results), set_skip_args=True) == stage_folder(STAGES[done])
	# Stages after 'preprocessed' map onto the skip flags in order; the final folder skips them all.
	n_skipped = {'preprocessed': 0, 'greedy': 1, 'redo_starry': 2, 'opt_times': 3, 'spr_moves': 4,
	             'nnn_reorder': 5, 'reordered_edges': 6, 'final': 6}[STAGES[done]]
	assert [getattr(args, flag) for flag in SKIPS] == [True] * n_skipped + [False] * (6 - n_skipped)


def test_find_latest_tree_folder_new_without_results_and_not_final(tmp_path):
	results = tmp_path / 'results'
	results.mkdir()
	args = make_args(tmp_path / 'data', results)
	assert find_latest_tree_folder_new(args, str(results)) == str(results)
	_make_stage_folders(results, ['opt_times', 'final'])
	assert find_latest_tree_folder_new(args, str(results), not_final=True) == stage_folder('opt_times')
	assert find_latest_tree_folder_new(args, str(results)) == stage_folder('final')
	args.zscore_cutoff = -1.0
	assert find_latest_tree_folder_new(args, str(results)) == str(results)


def test_find_latest_tree_folder_prefers_final_then_the_longest_intermediate_name(tmp_path):
	results = tmp_path / 'results'
	results.mkdir()
	assert find_latest_tree_folder(str(results)) == str(results)
	_make_stage_folders(results, ['preprocessed', 'greedy', 'opt_times', 'spr_moves'])
	assert find_latest_tree_folder(str(results)) == stage_folder('spr_moves')
	_make_stage_folders(results, ['final'])
	assert find_latest_tree_folder(str(results)) == stage_folder('final')
	assert find_latest_tree_folder(str(results), not_final=True) == stage_folder('spr_moves')


class _Results:
	def __init__(self, root):
		self.root = str(root)

	def result_path(self, name=''):
		return os.path.join(self.root, name)


def test_clean_up_redundant_data_files_empties_only_intermediate_coordinates(tmp_path):
	results = tmp_path / 'results'
	_make_stage_folders(results, STAGES)
	for stage in STAGES:
		for name in ('ltqs_vertByGene.npy', 'posterior_ltqs_vertByGene.npy', 'edgeInfo.txt'):
			(results / stage_folder(stage) / name).write_text('x')
	clean_up_redundant_data_files(_Results(results), make_args(tmp_path, results), verbose=False)
	for stage in STAGES:
		left = sorted(os.listdir(results / stage_folder(stage)))
		if stage in ('preprocessed', 'final'):
			assert left == ['edgeInfo.txt', 'ltqs_vertByGene.npy', 'posterior_ltqs_vertByGene.npy', 'vertInfo.txt']
		else:
			assert left == ['edgeInfo.txt', 'vertInfo.txt'], stage


def test_get_latest_intermediate_takes_the_highest_index():
	files = ['greedy_3', 'greedy_12.dat', 'other_40', 'greedy_x', 'greedy_7']
	assert get_latest_intermediate(files) == ('other_40', 40)
	assert get_latest_intermediate(files, base='greedy') == ('greedy_12.dat', 12)
	assert get_latest_intermediate([]) == (None, None)
	assert get_latest_intermediate(['a_b', 'c']) == (None, None)


def _touch(folder, names):
	folder.mkdir(parents=True, exist_ok=True)
	for name in names:
		(folder / name).write_text('x')


def test_remove_tree_folders(tmp_path):
	folder = tmp_path / 'intermediate_trees'
	_touch(folder, ['nnn_1', 'nnn_2', 'nnn_3', 'random_3', 'nnn_x.txt'])
	(folder / 'nnn_sub').mkdir()
	remove_tree_folders(str(folder), base='nnn', notRemove=3)
	assert sorted(os.listdir(folder)) == ['nnn_2', 'nnn_3', 'random_3']
	remove_tree_folders(str(folder), removeDir=True, base='random')
	assert sorted(os.listdir(folder)) == ['nnn_2', 'nnn_3']
	remove_tree_folders(str(folder), removeDir=True)
	assert not folder.exists()
	remove_tree_folders(str(folder), removeDir=True)


def test_remove_pickled_trees(tmp_path):
	folder = tmp_path / 'pickles'
	_touch(folder, ['tree_1.dat', 'tree_2.dat', 'tree_5.dat', 'other_9.dat'])
	removePickledTrees(str(folder), notRemove=5, base='tree')
	assert sorted(os.listdir(folder)) == ['other_9.dat', 'tree_5.dat']
	removePickledTrees(str(folder), removeDir=True)
	assert not folder.exists()
	removePickledTrees(str(folder), removeDir=True)


def test_remove_and_empty_folder(tmp_path):
	folder = tmp_path / 'f'
	_touch(folder, ['a', 'b'])
	empty_folder(str(folder))
	assert folder.exists() and os.listdir(folder) == []
	remove_folder(str(folder))
	assert not folder.exists()
	remove_folder(str(folder))
	empty_folder(str(folder))


def test_store_scdata_and_look_for_output_folder(tmp_path):
	"""The SPR step stores its state in a new folder under <results>/spr_intermediates whose name starts with the
	prefix (tree, metadata and vertex coordinates, no tree.nwk) so that other ranks, or a resumed run, can load
	it; look_for_output_folder finds the most recently written one with that prefix."""
	scdata, sim, args, tree, means, variances = scdata_with_tree(tmp_path)
	results = scdata.result_path()
	assert look_for_output_folder('spr_intermediates', general_results_folder=results) == (False, None)
	first = store_scdata_and_communicate_path(scdata, 'spr_intermediates', prefix='after_sprmoveset1')
	second = store_scdata_and_communicate_path(scdata, 'spr_intermediates', general_results_folder=results)
	assert os.path.basename(first).startswith('after_sprmoveset1')
	for path in (first, second):
		assert os.path.dirname(path) == os.path.join(results, 'spr_intermediates')
		assert sorted(os.listdir(path)) == ['edgeInfo.txt', 'ltqsVars_vertByGene.npy', 'ltqs_vertByGene.npy',
		                                    'metadata.json', 'vertInfo.txt']
	os.utime(first, (1e9, 1e9))
	os.utime(second, (2e9, 2e9))
	found, path = look_for_output_folder('spr_intermediates', general_results_folder=results)
	assert found and str(path) == second
	found, path = look_for_output_folder(os.path.join(results, 'spr_intermediates'), prefix='after_sprmoveset1')
	assert found and str(path) == first
	assert look_for_output_folder('spr_intermediates', general_results_folder=results, prefix='zzz') == (False, None)
	from bonsai.bonsai_dataprocessing import loadReconstructedTreeAndData
	from bonsai_testref import reference_bonsai_loglik
	from bonsai_testrun import tree_from_folder
	loaded = loadReconstructedTreeAndData(args, first, all_ranks=False, calc_loglik=True)
	assert loaded.metadata.loglik == pytest.approx(
		reference_bonsai_loglik(tree_from_folder(first), means, variances,
		                        loglik_var_corr=scdata.metadata.loglikVarCorr), rel=1e-9)


# ------------------------------------------------------------------ small file readers and writers

def test_write_vect_to_end_csv_appends_rows(tmp_path):
	path = str(tmp_path / 'trace.txt')
	writeVectToEndCsv(path, np.array([1.0, 2.5e-3]), init=True)
	writeVectToEndCsv(path, [3.0])
	writeVectToEndCsv(path, [])
	with open(path) as handle:
		rows = [line.rstrip('\n').split('\t') for line in handle]
	assert [[float(x) for x in row] for row in rows] == [[1.0, 2.5e-3], [3.0]]
	assert rows[0][0] == '1.00000000e+00'
	writeVectToEndCsv(path, [4.0], init=True)
	with open(path) as handle:
		assert handle.read().split() == ['4.00000000e+00']


def test_write_vect_to_end_csv_only_on_rank_zero(tmp_path, monkeypatch):
	monkeypatch.setattr(mpi_wrapper, 'comm', FakeComm(rank=1, size=2))
	writeVectToEndCsv(str(tmp_path / 'x.txt'), [1.0], init=True)
	assert not (tmp_path / 'x.txt').exists()


def test_ids_and_lists_read_the_first_column(tmp_path):
	path = str(tmp_path / 'ids.txt')
	write_ids(path, ['a', 'b c', 'd'])
	assert read_ids(path) == ['a', 'b c', 'd']
	(tmp_path / 'table.tsv').write_text('x\t1\ny\t2\n')
	assert read_ids(str(tmp_path / 'table.tsv')) == ['x', 'y']
	assert readListFromTxt(str(tmp_path / 'table.tsv')) == ['x', 'y']
	assert read_ids(str(tmp_path / 'absent.txt')) == []
	assert readListFromTxt(str(tmp_path / 'absent.txt')) is None


def test_create_celltypefile_from_cellstates(tmp_path):
	states = tmp_path / 'states.txt'
	states.write_text('3\n1\n3\n2\n3\n1\n')
	out = tmp_path / 'celltypes.txt'
	out.write_text('stale\n')
	create_celltypefile_from_cellstates(str(states), str(out))
	assert out.read_text().split('\n')[:-1] == ['CS_3', 'CS_1', 'CS_3', 'CS_2', 'CS_3', 'CS_1']
	create_celltypefile_from_cellstates(str(states), str(out), multiplicityCutoff=2)
	assert out.read_text().split('\n')[:-1] == ['CS_3', 'cs with <= 2 cells', 'CS_3', 'cs with <= 2 cells', 'CS_3',
	                                            'cs with <= 2 cells']


# ------------------------------------------------------------------ Sanity versions

def test_parse_sanity_command_file_upper_cases_the_method(tmp_path):
	path = tmp_path / 'sanity_command.txt'
	path.write_text('#Sanity version:2.3\n#   method :  eap\nSanity -e 1\n')
	assert parse_sanity_command_file(str(path)) == ((2, 3, 0), 'EAP')


def test_resolve_sanity_filepaths_returns_full_paths(tmp_path):
	for name in SANITY_LEGACY_FILENAMES.values():
		(tmp_path / name).write_text('')
	paths = resolve_sanity_filepaths(str(tmp_path))
	assert paths == {key: os.path.join(str(tmp_path), name) for key, name in SANITY_LEGACY_FILENAMES.items()}


def test_new_sanity_output_with_a_missing_file_is_rejected(tmp_path, caplog):
	(tmp_path / 'sanity_command.txt').write_text('# Sanity version: 2.0.0\n# Method: MAP\n')
	for name in list(SANITY_FILENAMES.values())[:-1]:
		(tmp_path / name).write_text('')
	with pytest.raises(SystemExit):
		resolve_sanity_filepaths(str(tmp_path))
	assert list(SANITY_FILENAMES.values())[-1] in caplog.text and '-e 1' in caplog.text


def _fake_binary(folder, output):
	path = folder / 'Sanity'
	path.write_text('#!/bin/sh\nprintf "%s"\n' % output)
	path.chmod(path.stat().st_mode | stat.S_IXUSR)
	return str(path)


@pytest.mark.skipif(sys.platform.startswith('win32'), reason='the fake Sanity binary is a shell script')
@pytest.mark.parametrize('output,version', [('Sanity version 2.0.0\\n', (2, 0, 0)), ('1.1\\n', (1, 1, 0)),
                                            ('', None), ('Sanity version unknown\\n', None)])
def test_get_sanity_binary_version(tmp_path, output, version):
	"""Sanity before 2.0 answers '-v' with the bare number, from 2.0 with 'Sanity version 2.0.0'."""
	assert get_sanity_binary_version(_fake_binary(tmp_path, output)) == version


def test_get_sanity_binary_version_of_a_missing_binary(tmp_path):
	assert get_sanity_binary_version(str(tmp_path / 'no_sanity')) is None


# ------------------------------------------------------------------ recursion limits

def test_recursion_limit_context_raises_and_restores_the_limit():
	limit = sys.getrecursionlimit()
	with RecursionLimit():
		assert sys.getrecursionlimit() == 10 ** 6
	assert sys.getrecursionlimit() == limit


def test_recursion_wrap_runs_deep_recursion():
	def depth(n):
		return 0 if n == 0 else 1 + depth(n - 1)

	limit = sys.getrecursionlimit()
	assert recursionWrap(depth, limit + 500) == limit + 500
	assert sys.getrecursionlimit() == limit


def test_set_recursion_limits_never_lowers_the_limit():
	limit = sys.getrecursionlimit()
	set_recursion_limits(10)
	assert sys.getrecursionlimit() == limit
	set_recursion_limits(limit + 1234)
	assert sys.getrecursionlimit() == limit + 1234


# ------------------------------------------------------------------ memory report

def test_print_memory_reports_traced_memory_and_resets_the_peak(caplog):
	tracemalloc.start()
	try:
		block = bytearray(3_000_000)
		del block
		print_memory('here')
		_, peak_after = tracemalloc.get_traced_memory()
	finally:
		tracemalloc.stop()
	# The message names the place and reports the peak in MB; its wording is not pinned.
	line = [record.getMessage() for record in caplog.records if 'here' in record.getMessage()][-1]
	assert max(float(x) for x in re.findall(r'\d+\.?\d*(?:e-?\d+)?', line.split('here', 1)[1])) >= 3.0
	assert peak_after < 3_000_000


# ------------------------------------------------------------------ interaction and gradient helpers

def test_interaction_event_calls_its_function():
	seen = []
	InteractionEvent(func=lambda a, b: seen.append((a, b)), kwargs={'a': 1, 'b': 2}).execute()
	InteractionEvent(func=lambda: seen.append('no args')).execute()
	assert seen == [(1, 2), 'no args']
	assert InteractionEvent().function is None


def test_check_grad_measures_the_gradient_error():
	"""checkGrad returns analytic minus central-difference gradient (and its relative size) per dimension."""

	def fun_grad(x, scale):
		return scale * np.sum(x ** 3), scale * 3 * x ** 2

	def wrong(x, scale):
		value, grad = fun_grad(x, scale)
		return value, grad + np.array([0.0, 0.5, 0.0])

	x = np.array([0.5, -1.0, 2.0])
	dev, rel = checkGrad(fun_grad, x, (2.0,))
	np.testing.assert_allclose(dev, 0.0, atol=1e-6)
	dev, rel = checkGrad(wrong, x, (2.0,))
	np.testing.assert_allclose(dev, [0.0, 0.5, 0.0], atol=1e-6)
	assert rel[1] == pytest.approx(0.5 / 6.5, rel=1e-6)
	dev, _ = checkGrad(wrong, x, (2.0,), dims=[1])
	assert dev.shape == (1,) and dev[0] == pytest.approx(0.5, abs=1e-6)


# ------------------------------------------------------------------ colours

def test_celltype_colors_choose_a_palette_by_number_of_types():
	assert get_celltype_colors(3).N == 8
	four = get_celltype_colors(4)
	assert four.N == 4 and isinstance(four, ListedColormap)
	assert get_celltype_colors(9).name == 'tab10'
	assert get_celltype_colors(15).name == 'tab20'
	off_on = get_celltype_colors(2, colortype='offOn')
	assert off_on.N == 2 and off_on(1)[:3] == pytest.approx(helpers.grey_cmap(0.5)[:3])
	gradient = get_celltype_colors(25)
	assert gradient.N == 25
	hsv = helpers.plt.get_cmap('hsv')
	assert gradient(0) == pytest.approx(hsv(0.0)) and gradient(24) == pytest.approx(hsv(1.0))


# ------------------------------------------------------------------ Poincare disk

def _poincare_radius(r):
	"""The disk radius of a point at Euclidean distance r: the root of rho / (1 - rho^2) = r in [0, 1)."""
	return 0.0 if r == 0 else (math.sqrt(4 * r * r + 1) - 1) / (2 * r)


def test_transform_coords_poincare_maps_the_plane_into_the_unit_disk():
	rng = np.random.default_rng(1)
	coords = rng.normal(0, 3, size=(20, 2))
	coords[0] = [1.0, 2.0]
	origin, zoom = np.array([1.0, 2.0]), 0.7
	out = transform_coords_poincare(coords, origin=origin, zoom=zoom)
	for point, mapped in zip(coords, out):
		r = np.linalg.norm(point - origin)
		assert np.linalg.norm(mapped) == pytest.approx(_poincare_radius(zoom * r), rel=1e-12, abs=1e-15)
		if r > 0:
			np.testing.assert_allclose(mapped / np.linalg.norm(mapped), (point - origin) / r, rtol=1e-12)
	np.testing.assert_array_equal(out[0], [0.0, 0.0])
	np.testing.assert_allclose(get_radial_poincare(coords, origin=origin, zoom=zoom), np.linalg.norm(out, axis=1),
	                           rtol=1e-12)
	np.testing.assert_allclose(invert_coords_poincare(out[1:], origin=origin, zoom=zoom), coords[1:], rtol=1e-9)


def test_poincare_without_transform_scales_and_clips():
	coords = np.array([[0.3, 0.4], [3.0, 4.0], [0.0, 0.0]])
	out = transform_coords_poincare(coords, zoom=1.0, no_transform=True)
	np.testing.assert_allclose(out, [[0.3, 0.4], [0.66, 0.88], [0.0, 0.0]], rtol=1e-12)
	np.testing.assert_allclose(invert_coords_poincare(out[:1], zoom=2.0, origin=np.array([1.0, 0.0]),
	                                                  no_transform=True), [[1.15, 0.2]], rtol=1e-12)


def test_zoom_puts_the_requested_fraction_of_points_within_the_radius():
	"""get_zoom_given_origin: the zoom at which the ceil(frac * n)-th nearest point sits at within_radius on the
	disk, i.e. zoom * d_k = w / (1 - w^2)."""
	rng = np.random.default_rng(2)
	coords = rng.normal(0, 2, size=(25, 2))
	origin = np.array([0.2, -0.1])
	zoom = get_zoom_given_origin(origin, coords, frac_within=0.8, within_radius=0.8)
	d_k = np.sort(np.linalg.norm(coords - origin, axis=1))[math.ceil(0.8 * 25) - 1]
	assert zoom == pytest.approx(0.8 / (1 - 0.64) / d_k, rel=1e-6)


def test_poincare_centroid_and_optimal_zoom_of_a_symmetric_cloud():
	"""Points symmetric under x -> -x and y -> -y: the disk centroid (minimum mean disk radius) is the origin,
	and the optimal zoom is the closed form of the previous test at that origin."""
	base = np.array([[1.0, 0.5], [2.0, 1.5], [0.5, 3.0], [4.0, 0.25]])
	coords = np.concatenate([base, base * [-1, 1], base * [1, -1], -base])
	centroid = get_centroid_poincare(coords + [0.3, 0.2], zoom=0.5)
	np.testing.assert_allclose(centroid, [0.3, 0.2], atol=1e-5)
	zoom, origin = get_opt_zoom_origin_poincare(coords, frac_within=0.5, within_radius=0.6)
	np.testing.assert_allclose(origin, [0.0, 0.0], atol=1e-5)
	d_k = np.sort(np.linalg.norm(coords, axis=1))[math.ceil(0.5 * len(coords)) - 1]
	assert zoom == pytest.approx(0.6 / (1 - 0.36) / d_k, rel=1e-5)
