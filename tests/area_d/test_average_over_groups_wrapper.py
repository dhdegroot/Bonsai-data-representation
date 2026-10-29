"""
downstream_analyses/average_over_groups_wrapper.py (area D): the command-line wrapper around run_averaging
(README "Averaging over groups"). Its options (-x, -ue, -names, -group_file, -col_idx, -output_folder,
-read_ids_from_file) are checked by running the script on small hand-made files; the group means are checked against
a direct maximum-likelihood fit of x_i ~ N(mu, v_i + tau^2) (scipy.stats.norm with a bounded minimizer).
"""
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env

WRAPPER = os.path.join(REPO_ROOT, 'downstream_analyses', 'average_over_groups_wrapper.py')


def reference_fit(x, v):
	def profile_nll(tau):
		weights = 1.0 / (v + tau ** 2)
		mu = np.sum(weights * x) / np.sum(weights)
		return -np.sum(stats.norm.logpdf(x, loc=mu, scale=np.sqrt(v + tau ** 2)))

	grid = np.linspace(0.0, 20.0, 801)
	start = grid[int(np.argmin([profile_nll(tau) for tau in grid]))]
	tau = optimize.minimize_scalar(profile_nll, bounds=(max(0.0, start - 0.03), start + 0.03), method='bounded',
	                               options={'xatol': 1e-10}).x
	if profile_nll(0.0) <= profile_nll(tau):
		tau = 0.0
	weights = 1.0 / (v + tau ** 2)
	return np.sum(weights * x) / np.sum(weights), math.sqrt(1.0 / np.sum(weights))


def _data():
	"""4 genes x 10 cells, the group file's second column holds the groups, two cells are 'NULL'."""
	rng = np.random.default_rng(5)
	deltas = rng.normal(0.0, 1.0, (4, 10))
	deltas[2, :5] += 2.5
	d_deltas = rng.uniform(0.2, 0.5, (4, 10))
	genes = ['geneA', 'geneB', 'geneC', 'geneD']
	cells = ['cell%d' % ind for ind in range(10)]
	groups = ['early', 'early', 'early', 'NULL', 'early', 'late', 'late', 'NULL', 'late', 'late']
	return deltas, d_deltas, genes, cells, groups


def _run(args, cwd):
	return subprocess.run([sys.executable, WRAPPER] + args, capture_output=True, text=True, env=bonsai_env(),
	                      cwd=cwd, timeout=120)


def _check_outputs(folder, deltas, d_deltas, genes, groups):
	avg = pd.read_csv(os.path.join(folder, 'avg_activities.csv'), index_col=0)
	err = pd.read_csv(os.path.join(folder, 'avg_deltas.csv'), index_col=0)
	significance = pd.read_csv(os.path.join(folder, 'significance.csv'), index_col=0).iloc[:, 0]
	assert list(avg.index) == ['early', 'late'] and list(avg.columns) == genes
	assert list(err.index) == ['early', 'late'] and list(err.columns) == genes
	for group in ('early', 'late'):
		cells = [ind for ind, name in enumerate(groups) if name == group]
		for row, gene in enumerate(genes):
			want_mu, want_sig = reference_fit(deltas[row, cells], d_deltas[row, cells] ** 2)
			assert avg.loc[group, gene] == pytest.approx(want_mu, abs=1e-4)
			assert err.loc[group, gene] == pytest.approx(want_sig, rel=1e-4)
	want = np.sqrt(((avg / err) ** 2).mean(axis=0)).sort_values(ascending=False)
	assert list(significance.index) == list(want.index)
	np.testing.assert_allclose(significance.values, want.values, rtol=1e-9)


def _plain_matrices(pool, workdir):
	deltas, d_deltas, genes, cells, groups = _data()
	np.savetxt(os.path.join(workdir, 'delta.txt'), deltas, delimiter='\t')
	np.savetxt(os.path.join(workdir, 'd_delta.txt'), d_deltas, delimiter='\t')
	with open(os.path.join(workdir, 'geneID.txt'), 'w') as handle:
		handle.write('\n'.join(genes) + '\n')
	with open(os.path.join(workdir, 'groups.tsv'), 'w') as handle:
		handle.write(''.join('%s\tbatch%d\t%s\n' % (cell, ind % 2, group)
		                     for ind, (cell, group) in enumerate(zip(cells, groups))))
	out = os.path.join(workdir, 'out')
	os.makedirs(out)
	proc = _run(['-x', os.path.join(workdir, 'delta.txt'), '-ue', os.path.join(workdir, 'd_delta.txt'),
	             '-names', os.path.join(workdir, 'geneID.txt'), '-group_file', os.path.join(workdir, 'groups.tsv'),
	             '-col_idx', '2', '-output_folder', out], cwd=workdir)
	return proc, out


def _ids_from_file(pool, workdir):
	deltas, d_deltas, genes, cells, groups = _data()
	pd.DataFrame(deltas, index=genes, columns=cells).to_csv(os.path.join(workdir, 'delta.txt'), sep='\t')
	pd.DataFrame(d_deltas, index=genes, columns=cells).to_csv(os.path.join(workdir, 'd_delta.txt'), sep='\t')
	with open(os.path.join(workdir, 'groups.tsv'), 'w') as handle:
		handle.write(''.join('%s\n' % group for group in groups))
	proc = _run(['-group_file', os.path.join(workdir, 'groups.tsv'), '-col_idx', '0', '-read_ids_from_file', 'yes'],
	            cwd=workdir)
	return proc, workdir


d_avg_wrapper_plain = bonsai_job_fixture('d_avg_wrapper_plain', _plain_matrices)
d_avg_wrapper_ids = bonsai_job_fixture('d_avg_wrapper_ids', _ids_from_file)


def test_wrapper_on_plain_matrices(d_avg_wrapper_plain):
	"""Default mode: -x and -ue are bare genes x cells matrices, -names one gene ID per line; the group of each cell
	(line i of the tab-separated group file is cell i) is in column -col_idx; 'NULL' cells are left out; the three
	CSV files land in -output_folder."""
	proc, out = d_avg_wrapper_plain
	assert proc.returncode == 0, proc.stderr[-3000:]
	deltas, d_deltas, genes, _, groups = _data()
	_check_outputs(out, deltas, d_deltas, genes, groups)


def test_wrapper_reads_ids_from_file_and_uses_default_paths(d_avg_wrapper_ids):
	"""-read_ids_from_file True: -x and -ue carry a header row and the gene IDs as index (-names is not read). With
	no -x, -ue, -names or -output_folder the defaults ./delta.txt, ./d_delta.txt and '.' are used."""
	proc, out = d_avg_wrapper_ids
	assert proc.returncode == 0, proc.stderr[-3000:]
	deltas, d_deltas, genes, _, groups = _data()
	_check_outputs(out, deltas, d_deltas, genes, groups)


def test_str2bool():
	"""Accepted spellings, case-insensitive; anything else stops with 'Boolean value expected.'."""
	from downstream_analyses.average_over_groups_wrapper import str2bool
	for value in ('yes', 'True', 't', 'Y', '1'):
		assert str2bool(value) is True
	for value in ('no', 'FALSE', 'f', 'n', '0'):
		assert str2bool(value) is False
	with pytest.raises(SystemExit, match='Boolean value expected'):
		str2bool('maybe')
