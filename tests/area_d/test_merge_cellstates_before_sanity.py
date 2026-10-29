"""
optional_preprocessing/merge_cellstates_before_sanity.py (area D): merge_clusters_acc_cellstates, which sums the
UMI counts of all cells in a cellstate into one "super-cell", and the script around it (all 7 options).

Expected values are worked out by hand on a 3-gene, 7-cell count table. The label the function gives to small
cellstates differs between the cellstate and the cell annotation (`cs<=N` against `cs<N`) although the docstring
says they are the same, so the tests check only what both share. The script's .mtx input only works for an
absolute path without dots; that test skips when pytest's temporary folder has a dot in it.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix

from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env
from optional_preprocessing.merge_cellstates_before_sanity import merge_clusters_acc_cellstates

SCRIPT = os.path.join(REPO_ROOT, 'optional_preprocessing', 'merge_cellstates_before_sanity.py')

CELLS = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6']
GENES = ['gA', 'gB', 'gC']
CLUSTERS = np.array([2, 0, 2, 1, 0, 2, 3])
COUNTS = np.array([[1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 0, 1, 0, 1, 0],
                   [10, 0, 0, 0, 20, 0, 30]])
# Summed by hand per cellstate (sorted cluster labels 0, 1, 2, 3): cluster 0 = c1+c4, 1 = c3, 2 = c0+c2+c5, 3 = c6.
SUMMED_ALL = np.array([[7, 4, 10, 7],
                       [1, 1, 1, 0],
                       [20, 0, 10, 30]])
SMALL_LABEL = re.compile(r'^cs<=?(\d+)$')


def _is_small_label(label, cutoff):
	match = SMALL_LABEL.match(label)
	return match is not None and int(match.group(1)) == cutoff


def test_no_cutoffs_sums_every_cellstate():
	summed, counts, cs_annot, cell_annot, cs_ids, cell_id_to_cs_id, cell_ids = merge_clusters_acc_cellstates(
		CLUSTERS, COUNTS, CELLS, cutoff_annot=0, cutoff_filter=0, verbose=False)
	np.testing.assert_array_equal(summed, SUMMED_ALL)
	np.testing.assert_array_equal(counts, [2, 1, 3, 1])
	assert cs_ids == ['cs_0', 'cs_1', 'cs_2', 'cs_3']
	assert cs_annot == cs_ids
	assert cell_ids == CELLS
	expected = {'c0': 'cs_2', 'c1': 'cs_0', 'c2': 'cs_2', 'c3': 'cs_1', 'c4': 'cs_0', 'c5': 'cs_2', 'c6': 'cs_3'}
	assert cell_id_to_cs_id == expected
	assert cell_annot == [expected[cell] for cell in CELLS]
	# Every UMI ends up in exactly one super-cell.
	np.testing.assert_array_equal(summed.sum(axis=1), COUNTS.sum(axis=1))


def test_filter_and_annotation_cutoffs():
	"""cutoff_filter=1 drops the one-cell cellstates; cutoff_annot=2 marks the two-cell one as small."""
	summed, counts, cs_annot, cell_annot, cs_ids, cell_id_to_cs_id, cell_ids = merge_clusters_acc_cellstates(
		CLUSTERS, COUNTS, CELLS, cutoff_annot=2, cutoff_filter=1, verbose=False)
	# Kept: cluster 0 (c1, c4) as cs_0 and cluster 2 (c0, c2, c5) as cs_1; cells keep their input order.
	np.testing.assert_array_equal(summed, [[7, 10], [1, 1], [20, 10]])
	np.testing.assert_array_equal(counts, [2, 3])
	assert cs_ids == ['cs_0', 'cs_1']
	assert cell_ids == ['c0', 'c1', 'c2', 'c4', 'c5']
	assert cell_id_to_cs_id == {'c0': 'cs_1', 'c1': 'cs_0', 'c2': 'cs_1', 'c4': 'cs_0', 'c5': 'cs_1'}
	assert cs_annot[1] == 'cs_1'
	assert _is_small_label(cs_annot[0], 2)
	assert [cell_annot[ind] for ind in (0, 2, 4)] == ['cs_1'] * 3
	assert _is_small_label(cell_annot[1], 2) and cell_annot[1] == cell_annot[3]


def test_string_cluster_labels_are_sorted():
	labels = np.array(['b', 'a', 'b', 'a'])
	counts = np.array([[1, 2, 3, 4]])
	summed, n_cells, _, _, cs_ids, cell_id_to_cs_id, _ = merge_clusters_acc_cellstates(
		labels, counts, ['w', 'x', 'y', 'z'], verbose=False)
	np.testing.assert_array_equal(summed, [[6, 4]])
	np.testing.assert_array_equal(n_cells, [2, 2])
	assert cell_id_to_cs_id == {'w': 'cs_1', 'x': 'cs_0', 'y': 'cs_1', 'z': 'cs_0'}


def test_filter_that_removes_everything_exits():
	with pytest.raises(SystemExit):
		merge_clusters_acc_cellstates(CLUSTERS, COUNTS, CELLS, cutoff_filter=3, verbose=False)


# ---------------------------------------------------------------------------------------------- the script

def _run(args, cwd):
	return subprocess.run([sys.executable, SCRIPT] + [str(arg) for arg in args], capture_output=True, text=True,
	                      env=bonsai_env(), cwd=cwd, timeout=120)


def _write_inputs(folder, with_cellid_file=True, cellid_override=None):
	cs_folder = os.path.join(folder, 'cellstates_out')
	raw_folder = os.path.join(folder, 'raw')
	os.makedirs(cs_folder)
	os.makedirs(raw_folder)
	with open(os.path.join(cs_folder, 'optimized_clusters.txt'), 'w') as handle:
		handle.write(''.join('%d\n' % label for label in CLUSTERS))
	if with_cellid_file:
		with open(os.path.join(cs_folder, 'CellID.txt'), 'w') as handle:
			handle.write(''.join('%s\n' % cell for cell in (cellid_override or CELLS)))
	table = pd.DataFrame(COUNTS, index=GENES, columns=CELLS)
	table.to_csv(os.path.join(raw_folder, 'Gene_table.tsv'), sep='\t', index_label='GeneID')
	mmwrite(os.path.join(raw_folder, 'counts.mtx'), csr_matrix(COUNTS))
	with open(os.path.join(raw_folder, 'genes.tsv'), 'w') as handle:
		handle.write(''.join('%s\textra\n' % gene for gene in GENES))
	with open(os.path.join(raw_folder, 'cells.tsv'), 'w') as handle:
		handle.write(''.join('%s\n' % cell for cell in CELLS))
	return cs_folder, raw_folder


def _merge_script_runs(pool, workdir):
	out = {}
	# 1. Tab-separated counts, explicit output folder and cutoffs, a matching CellID.txt.
	folder = os.path.join(workdir, 'tsv')
	cs_folder, raw_folder = _write_inputs(folder)
	results = os.path.join(folder, 'merged')
	out['tsv'] = (_run(['--folder_cellstates_output', cs_folder, '--file_raw_umi_counts',
	                    os.path.join(raw_folder, 'Gene_table.tsv'), '--folder_clustered_umi_counts', results,
	                    '--cutoff_annot', 2, '--cutoff_filter', 1], cwd=folder), raw_folder, results)
	# 2. Matrix-market counts with the ID files, default output folder and cutoffs, no CellID.txt.
	folder = os.path.join(workdir, 'mtx')
	cs_folder, raw_folder = _write_inputs(folder, with_cellid_file=False)
	count_path = os.path.join(raw_folder, 'counts.mtx')
	if '.' in os.path.dirname(os.path.abspath(count_path)):
		out['mtx'] = None
	else:
		out['mtx'] = (_run(['--folder_cellstates_output', cs_folder, '--file_raw_umi_counts', count_path,
		                    '--file_cell_ids', os.path.join(raw_folder, 'cells.tsv'),
		                    '--file_gene_ids', os.path.join(raw_folder, 'genes.tsv')], cwd=folder),
		              raw_folder, os.path.join(raw_folder, 'cs_merged'))
	# 3. CellID.txt of the cellstates run disagrees with the count table: the script refuses.
	folder = os.path.join(workdir, 'mismatch')
	cs_folder, raw_folder = _write_inputs(folder, cellid_override=CELLS[::-1])
	results = os.path.join(folder, 'merged')
	out['mismatch'] = (_run(['--folder_cellstates_output', cs_folder, '--file_raw_umi_counts',
	                         os.path.join(raw_folder, 'Gene_table.tsv'), '--folder_clustered_umi_counts', results],
	                        cwd=folder), raw_folder, results)
	return out


d_opt_merge_cellstates = bonsai_job_fixture('d_opt_merge_cellstates', _merge_script_runs)


def _read_lines(path):
	with open(path) as handle:
		return handle.read().splitlines()


def _check_outputs(proc, raw_folder, results, summed, counts, cs_ids, cell_to_cs, cell_annot_named, cutoff_annot):
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	np.testing.assert_array_equal(mmread(os.path.join(results, 'prom_cs_expr_matrix.mtx')).toarray(), summed)
	assert _read_lines(os.path.join(results, 'cellID.txt')) == cs_ids
	assert _read_lines(os.path.join(results, 'geneID.txt')) == GENES
	assert _read_lines(os.path.join(results, 'orig_CellID.txt')) == CELLS
	assert [int(value) for value in _read_lines(os.path.join(results, 'cs_clusters.txt'))] == list(CLUSTERS)
	mapping = pd.read_csv(os.path.join(results, 'cell_id_to_cs_id.csv'), header=None, index_col=0)
	assert mapping.iloc[:, 0].to_dict() == cell_to_cs
	cs_annotation = pd.read_csv(os.path.join(results, 'annotation', 'cs_annotation.csv'), index_col=0)
	assert list(cs_annotation.index) == cs_ids
	assert list(cs_annotation.columns) == ['cellstates', 'cells_in_cellstate']
	assert list(cs_annotation['cells_in_cellstate']) == counts
	for cs_id, label, count in zip(cs_ids, cs_annotation['cellstates'], counts):
		assert label == cs_id if count > cutoff_annot else _is_small_label(label, cutoff_annot)
	# The per-cell annotation goes next to the raw counts.
	cell_annotation = pd.read_csv(os.path.join(raw_folder, 'annotation', 'cs_annotation.csv'), index_col=0)
	assert list(cell_annotation.index) == list(cell_to_cs)
	assert list(cell_annotation.columns) == ['cellstates']
	for cell, label in cell_annotation['cellstates'].items():
		if cell in cell_annot_named:
			assert label == cell_annot_named[cell]
		else:
			assert _is_small_label(label, cutoff_annot)


def test_script_with_tsv_counts_and_cutoffs(d_opt_merge_cellstates):
	proc, raw_folder, results = d_opt_merge_cellstates['tsv']
	_check_outputs(proc, raw_folder, results, summed=[[7, 10], [1, 1], [20, 10]], counts=[2, 3],
	               cs_ids=['cs_0', 'cs_1'],
	               cell_to_cs={'c0': 'cs_1', 'c1': 'cs_0', 'c2': 'cs_1', 'c4': 'cs_0', 'c5': 'cs_1'},
	               cell_annot_named={'c0': 'cs_1', 'c2': 'cs_1', 'c5': 'cs_1'}, cutoff_annot=2)


def test_script_with_mtx_counts_and_defaults(d_opt_merge_cellstates):
	"""Default cutoffs (0) keep every cellstate; the default output folder is <count folder>/cs_merged."""
	if d_opt_merge_cellstates['mtx'] is None:
		pytest.skip('the .mtx branch needs a dot-free absolute path')
	proc, raw_folder, results = d_opt_merge_cellstates['mtx']
	expected = {'c0': 'cs_2', 'c1': 'cs_0', 'c2': 'cs_2', 'c3': 'cs_1', 'c4': 'cs_0', 'c5': 'cs_2', 'c6': 'cs_3'}
	_check_outputs(proc, raw_folder, results, summed=SUMMED_ALL, counts=[2, 1, 3, 1],
	               cs_ids=['cs_0', 'cs_1', 'cs_2', 'cs_3'], cell_to_cs=expected, cell_annot_named=expected,
	               cutoff_annot=0)


def test_script_refuses_mismatched_cell_ids(d_opt_merge_cellstates):
	proc, raw_folder, results = d_opt_merge_cellstates['mismatch']
	assert 'Cell IDs do not match' in proc.stdout + proc.stderr
	assert not os.path.exists(os.path.join(results, 'prom_cs_expr_matrix.mtx'))
	assert not os.path.exists(os.path.join(raw_folder, 'annotation'))
