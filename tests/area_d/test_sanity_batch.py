"""
optional_preprocessing/batch_correction/sanity_batch.py (area D): splits a count table by batch, writes one
Matrix Market folder per batch plus one of the counts summed per batch ('total_counts_per_batch'), and runs Sanity on
each (all 8 options).

Sanity itself is replaced by a stub executable that answers the version query the way
bonsai_helpers.get_sanity_binary_version reads it ('Sanity version 2.0.0', or just '1.1' before 2.0) and records
the arguments of every run; --conda_env is checked with a stub 'conda' on PATH that records its arguments and runs
the rest of the command. Both stubs are scripts with a '#!' line, which Windows cannot execute, so the tests skip
there. Expected splits and sums are worked out by hand.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import psutil
import pytest
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix

from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env

pytestmark = pytest.mark.skipif(sys.platform.startswith('win32'),
                                reason="the Sanity and conda stubs are '#!' scripts, which Windows cannot execute")

SCRIPT = os.path.join(REPO_ROOT, 'optional_preprocessing', 'batch_correction', 'sanity_batch.py')

GENES = ['G0', 'G1', 'G2', 'G3']
CELLS = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
BATCH_OF = {'c0': 'b2', 'c1': 'a1', 'c2': 'b2', 'c3': 'a1', 'c4': 'a1', 'c5': 'b2'}
COUNTS = np.array([[1, 0, 2, 3, 0, 4],
                   [0, 5, 0, 0, 6, 0],
                   [7, 8, 9, 10, 11, 12],
                   [0, 0, 0, 0, 0, 1]])
# By hand: batches sorted (a1, b2); a1 = c1, c3, c4 and b2 = c0, c2, c5 in count-table order.
BATCH_CELLS = {'a1': ['c1', 'c3', 'c4'], 'b2': ['c0', 'c2', 'c5']}
BATCH_COUNTS = {'a1': COUNTS[:, [1, 3, 4]], 'b2': COUNTS[:, [0, 2, 5]]}
TOTALS = np.array([[3, 7], [11, 0], [29, 28], [0, 1]])
ALL_BATCHES = ['a1', 'b2', 'total_counts_per_batch']

SANITY_STUB = '''#!{python}
import json, os, sys
args = sys.argv[1:]
if args == ['-v']:
	print({answer!r})
	sys.exit(0)
out = args[args.index('-d') + 1]
with open(os.path.join(out, 'stub_argv.json'), 'w') as handle:
	json.dump(args, handle)
print('stub sanity ran on ' + args[args.index('-f') + 1])
sys.exit({exit_code})
'''

CONDA_STUB = '''#!{python}
import json, os, sys
with open({log!r}, 'a') as handle:
	handle.write(json.dumps(sys.argv[1:]) + '\\n')
args = sys.argv[1:]
command = args[args.index('--no-capture-output') + 1:]
os.execv(command[0], command)
'''


def _executable(path, text):
	with open(path, 'w') as handle:
		handle.write(text)
	os.chmod(path, 0o755)
	return path


def _write_inputs(folder):
	raw = os.path.join(folder, 'raw')
	os.makedirs(raw)
	pd.DataFrame(COUNTS, index=GENES, columns=CELLS).to_csv(os.path.join(raw, 'counts.tsv'), sep='\t',
	                                                        index_label='GeneID')
	mmwrite(os.path.join(raw, 'counts.mtx'), csr_matrix(COUNTS))
	with open(os.path.join(raw, 'genes.txt'), 'w') as handle:
		handle.write(''.join(gene + '\n' for gene in GENES))
	with open(os.path.join(raw, 'cells.txt'), 'w') as handle:
		handle.write(''.join(cell + '\n' for cell in CELLS))
	annotation = os.path.join(folder, 'batches.tsv')
	with open(annotation, 'w') as handle:
		handle.write(''.join('%s\t%s\n' % (cell, BATCH_OF[cell]) for cell in CELLS))
	return raw, annotation


def _sanity_batch(variant):
	def job(pool, workdir):
		raw, annotation = _write_inputs(workdir)
		results = os.path.join(workdir, 'sanity_results')
		answer = '1.1' if variant == 'mtx_legacy' else 'Sanity version 2.0.0'
		stub = _executable(os.path.join(workdir, 'Sanity'), SANITY_STUB.format(
			python=sys.executable, answer=answer, exit_code=3 if variant == 'failing' else 0))
		env = bonsai_env()
		env.pop('SLURM_CPUS_PER_TASK', None)
		args = ['--batch_annotation_file', annotation, '--results_folder', results, '--sanity_binary_path', stub]
		conda_log = None
		if variant == 'tsv_conda':
			bin_folder = os.path.join(workdir, 'bin')
			os.makedirs(bin_folder)
			conda_log = os.path.join(workdir, 'conda_calls.jsonl')
			_executable(os.path.join(bin_folder, 'conda'), CONDA_STUB.format(python=sys.executable, log=conda_log))
			env['PATH'] = bin_folder + os.pathsep + env.get('PATH', '')
			env['SLURM_CPUS_PER_TASK'] = '3'
			args += ['--count_file', os.path.join(raw, 'counts.tsv'), '--conda_env', 'sanity_env', '--verbose', 'True']
		elif variant == 'mtx_legacy':
			if '.' in os.path.dirname(os.path.abspath(raw)):
				return None
			env['SLURM_CPUS_PER_TASK'] = '2'
			args += ['--count_file', os.path.join(raw, 'counts.mtx'), '--gene_names_file',
			         os.path.join(raw, 'genes.txt'), '--cell_names_file', os.path.join(raw, 'cells.txt')]
		else:
			args += ['--count_file', os.path.join(raw, 'counts.tsv')]
		proc = subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True, env=env, cwd=workdir,
		                      timeout=120)
		return proc, raw, results, stub, conda_log
	return job


d_opt_sanity_batch_tsv = bonsai_job_fixture('d_opt_sanity_batch_tsv', _sanity_batch('tsv_conda'))
d_opt_sanity_batch_mtx = bonsai_job_fixture('d_opt_sanity_batch_mtx', _sanity_batch('mtx_legacy'))
d_opt_sanity_batch_failing = bonsai_job_fixture('d_opt_sanity_batch_failing', _sanity_batch('failing'))


def _read_lines(path):
	with open(path) as handle:
		return handle.read().splitlines()


def _check_split(raw):
	split = os.path.join(raw, 'batch_corrected')
	for batch in ('a1', 'b2'):
		folder = os.path.join(split, batch)
		np.testing.assert_array_equal(mmread(os.path.join(folder, 'prom_expr_matrix.mtx')).toarray(),
		                              BATCH_COUNTS[batch])
		assert _read_lines(os.path.join(folder, 'accepted_barcodes.tsv')) == BATCH_CELLS[batch]
		assert _read_lines(os.path.join(folder, 'prom_expr_promoters.tsv')) == GENES
	total = os.path.join(split, 'total_counts_per_batch')
	np.testing.assert_array_equal(mmread(os.path.join(total, 'prom_expr_matrix.mtx')).toarray(), TOTALS)
	assert _read_lines(os.path.join(total, 'accepted_barcodes.tsv')) == ['a1', 'b2']
	assert _read_lines(os.path.join(total, 'prom_expr_promoters.tsv')) == GENES
	# The batch annotation is copied next to the batch folders.
	assert _read_lines(os.path.join(split, 'batches.tsv')) == ['%s\t%s' % (cell, BATCH_OF[cell]) for cell in CELLS]


def _expected_sanity_args(raw, results, batch, n_cores, method_args):
	folder = os.path.join(raw, 'batch_corrected', batch)
	return ['-f', os.path.join(folder, 'prom_expr_matrix.mtx'),
	        '-mtx_genes', os.path.join(folder, 'prom_expr_promoters.tsv'),
	        '-mtx_cells', os.path.join(folder, 'accepted_barcodes.tsv'), '-d', os.path.join(results, batch),
	        '-n', str(n_cores), '-e', '1'] + method_args


def _check_sanity_runs(raw, results, n_cores, method_args):
	for batch in ALL_BATCHES:
		with open(os.path.join(results, batch, 'stub_argv.json')) as handle:
			assert json.load(handle) == _expected_sanity_args(raw, results, batch, n_cores, method_args)
		# Sanity's output is logged in the batch's output folder.
		log = _read_lines(os.path.join(results, batch, 'sanity_log.txt'))
		assert log == ['stub sanity ran on ' + os.path.join(raw, 'batch_corrected', batch, 'prom_expr_matrix.mtx')]


def test_tsv_counts_are_split_by_batch(d_opt_sanity_batch_tsv):
	proc, raw, results, stub, conda_log = d_opt_sanity_batch_tsv
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	_check_split(raw)


def test_sanity_2_runs_once_per_batch_through_conda(d_opt_sanity_batch_tsv):
	"""Sanity 2.0 gets '-v_m MAP'; SLURM_CPUS_PER_TASK sets '-n'; --conda_env wraps every Sanity call."""
	proc, raw, results, stub, conda_log = d_opt_sanity_batch_tsv
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	_check_sanity_runs(raw, results, 3, ['-v_m', 'MAP'])
	calls = [json.loads(line) for line in _read_lines(conda_log)]
	assert calls == [['run', '-n', 'sanity_env', '--no-capture-output', stub] +
	                 _expected_sanity_args(raw, results, batch, 3, ['-v_m', 'MAP']) for batch in ALL_BATCHES]


def test_mtx_counts_and_legacy_sanity(d_opt_sanity_batch_mtx):
	"""A Sanity older than 2.0 (answering '1.1') gets '-max_v only_max_output'; no conda means a direct call."""
	if d_opt_sanity_batch_mtx is None:
		pytest.skip('the .mtx branch needs a dot-free path')
	proc, raw, results, stub, _ = d_opt_sanity_batch_mtx
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	_check_split(raw)
	_check_sanity_runs(raw, results, 2, ['-max_v', 'only_max_output'])


def test_failing_sanity_stops_the_script(d_opt_sanity_batch_failing):
	"""Without SLURM, '-n' is the number of physical cores; a Sanity run that fails raises, and no later run starts."""
	proc, raw, results, stub, _ = d_opt_sanity_batch_failing
	assert proc.returncode != 0
	assert 'CalledProcessError' in proc.stderr
	with open(os.path.join(results, 'a1', 'stub_argv.json')) as handle:
		assert json.load(handle) == _expected_sanity_args(raw, results, 'a1', psutil.cpu_count(logical=False),
		                                                  ['-v_m', 'MAP'])
	assert not os.path.exists(os.path.join(results, 'b2', 'stub_argv.json'))
