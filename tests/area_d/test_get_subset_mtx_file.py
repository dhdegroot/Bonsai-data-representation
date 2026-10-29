"""
optional_preprocessing/get_subset_mtx_file.py (area D): selects the cells of chosen samples and the genes of one
gene type from a Matrix Market count file and writes them as a gene table with cell and cell-type annotations.
All 7 options, with inputs small enough to select by hand. The input files are tab-separated with a header row and
the script reads fixed columns: cell ID / sample from columns 1 / 3 of the cell annotation, cell barcode / cell type
from columns 1 / 9 of the cell-type annotation, gene ID / gene type from columns 1 / 7 of the gene annotation.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env

SCRIPT = os.path.join(REPO_ROOT, 'optional_preprocessing', 'get_subset_mtx_file.py')

CELLS = ['run_4_AAA', 'run_4_CCC', 'run_4_GGG', 'run_4_TTT', 'run_4_ACG']
SAMPLES = ['embryo_6', 'embryo_1', 'embryo_3', 'embryo_6', 'embryo_2']
CELLTYPES = {'AAA': 'neuron', 'CCC': 'muscle', 'GGG': 'glia', 'TTT': 'neuron', 'ACG': 'blood'}
GENES = ['G0', 'G1', 'G2', 'G3', 'G4']
GENE_TYPES = ['protein_coding', 'lncRNA', 'protein_coding', 'protein_coding', 'lncRNA']
# counts[g, c] = 10 g + c + 1, with a few zeros so that the matrix is sparse.
COUNTS = (10 * np.arange(5)[:, None] + np.arange(5)[None, :] + 1) * (np.arange(25).reshape(5, 5) % 4 != 1)

DEFAULT_NAMES = ('gene_count.run_4.mtx', 'cell_annotation.run_4.csv', 'gene_annotation.csv',
                 'GSE186069_cell_annotate.csv')


def _write_tsv(path, header, rows):
	with open(path, 'w') as handle:
		handle.write('\t'.join(header) + '\n')
		for row in rows:
			handle.write('\t'.join(row) + '\n')


def _write_inputs(folder, names):
	os.makedirs(folder)
	counts_name, cell_name, gene_name, celltype_name = names
	mmwrite(os.path.join(folder, counts_name), coo_matrix(COUNTS))
	_write_tsv(os.path.join(folder, cell_name), ['cell', 'batch', 'sample'],
	           [[cell, 'b', sample] for cell, sample in zip(CELLS, SAMPLES)])
	_write_tsv(os.path.join(folder, gene_name), ['gene_id', 'c1', 'c2', 'c3', 'c4', 'c5', 'gene_type'],
	           [[gene, '.', '.', '.', '.', '.', kind] for gene, kind in zip(GENES, GENE_TYPES)])
	# The cell-type table lists barcodes without the 'run_4_' prefix, plus one barcode of another run.
	rows = [[barcode] + ['x'] * 7 + [celltype] for barcode, celltype in CELLTYPES.items()]
	rows.append(['TTTT'] + ['x'] * 7 + ['unused'])
	_write_tsv(os.path.join(folder, celltype_name), ['sample'] + ['c%d' % ind for ind in range(7)] + ['type'],
	           rows)


def _run(args, cwd):
	return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True, env=bonsai_env(),
	                      cwd=cwd, timeout=120)


def _subset_runs(pool, workdir):
	defaults = os.path.join(workdir, 'defaults')
	_write_inputs(defaults, DEFAULT_NAMES)
	explicit = os.path.join(workdir, 'explicit')
	names = ('counts.mtx', 'cells.tsv', 'genes.tsv', 'celltypes.tsv')
	_write_inputs(explicit, names)
	return {
		# Every option at its default: the folder is '.', so run inside it.
		'defaults': (_run([], cwd=defaults), os.path.join(defaults, 'embryo_6,embryo_3')),
		'explicit': (_run(['--folder_raw_umi_counts', explicit, '--file_raw_umi_counts', names[0],
		                   '--file_cell_annotation', names[1], '--file_gene_annotation', names[2],
		                   '--file_celltype_annotation', names[3], '--desired_cell_annotation', 'embryo_2,embryo_1',
		                   '--desired_gene_annotation', 'lncRNA'], cwd=workdir),
		             os.path.join(explicit, 'embryo_2,embryo_1')),
	}


d_opt_subset_mtx = bonsai_job_fixture('d_opt_subset_mtx', _subset_runs)


def _read_lines(path):
	with open(path) as handle:
		return handle.read().splitlines()


def _check(proc, out_folder, cell_inds, gene_inds):
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	table = pd.read_csv(os.path.join(out_folder, 'Gene_table.txt'), sep='\t')
	assert list(table.columns) == ['GeneID'] + [CELLS[ind] for ind in cell_inds]
	assert list(table['GeneID']) == [GENES[ind] for ind in gene_inds]
	np.testing.assert_array_equal(table.iloc[:, 1:].values, COUNTS[np.ix_(gene_inds, cell_inds)])
	assert _read_lines(os.path.join(out_folder, 'cellID.txt')) == [CELLS[ind] for ind in cell_inds]
	assert _read_lines(os.path.join(out_folder, 'Celltype.txt')) == [CELLTYPES[CELLS[ind][6:]] for ind in cell_inds]
	assert _read_lines(os.path.join(out_folder, 'Celltype_sample.txt')) == [SAMPLES[ind] for ind in cell_inds]


def test_defaults_select_embryo_6_and_3_protein_coding(d_opt_subset_mtx):
	proc, out_folder = d_opt_subset_mtx['defaults']
	# Cells in file order whose sample is embryo_6 or embryo_3; genes of type protein_coding.
	_check(proc, out_folder, cell_inds=[0, 2, 3], gene_inds=[0, 2, 3])


def test_explicit_files_samples_and_gene_type(d_opt_subset_mtx):
	proc, out_folder = d_opt_subset_mtx['explicit']
	# File order, not the order of the requested samples.
	_check(proc, out_folder, cell_inds=[1, 4], gene_inds=[1, 4])
