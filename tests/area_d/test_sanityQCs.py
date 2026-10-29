"""
optional_preprocessing/sanityQCs.py (area D): quality-control figures of a Sanity output folder (its one option,
--sanity_folder). The script only draws, so the test runs it through a small wrapper that records what each saved
figure was drawn from (the values passed to hist and scatter, title, axis labels and scales), and compares that with
the files the harness wrote: the gene means (mu.txt), gene variances (variance.txt) and the per-gene signal-to-noise
z_g = 1/C sum_c (delta_gc - mean_c delta_gc)^2 / d_delta_gc^2 as README defines it for --zscore_cutoff.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from bonsai_testdata import DatasetSpec, balanced_newick
from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env

SCRIPT = os.path.join(REPO_ROOT, 'optional_preprocessing', 'sanityQCs.py')
# Noise genes give z_g well below 1 and signal genes well above it, so the figure's count is not trivial.
SPEC = DatasetSpec(newick=balanced_newick(3), n_genes=12, n_noise_genes=8, seed=11, layout='sanity2')
FIGURES = ('mean_LTQ_histogram.png', 'var_histogram.png', 'scatter_mean_var.png', 'zscores_histogram.png')
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

WRAPPER = r'''
import json, os, runpy, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.axes, matplotlib.figure
record_path, script = sys.argv[1], sys.argv[2]
calls, saved = {}, []
orig_hist, orig_scatter = matplotlib.axes.Axes.hist, matplotlib.axes.Axes.scatter
orig_savefig = matplotlib.figure.Figure.savefig
def hist(self, x, *args, **kwargs):
	calls.setdefault(id(self), []).append(['hist', np.ravel(x).tolist(), kwargs.get('cumulative', False)])
	return orig_hist(self, x, *args, **kwargs)
def scatter(self, x, y, *args, **kwargs):
	calls.setdefault(id(self), []).append(['scatter', np.ravel(x).tolist(), np.ravel(y).tolist()])
	return orig_scatter(self, x, y, *args, **kwargs)
def savefig(self, fname, *args, **kwargs):
	ax = self.axes[0]
	saved.append({'file': os.path.basename(fname), 'title': ax.get_title(), 'xlabel': ax.get_xlabel(),
	              'ylabel': ax.get_ylabel(), 'yscale': ax.get_yscale(), 'calls': calls.get(id(ax), [])})
	return orig_savefig(self, fname, *args, **kwargs)
matplotlib.axes.Axes.hist, matplotlib.axes.Axes.scatter, matplotlib.figure.Figure.savefig = hist, scatter, savefig
sys.argv = [script] + sys.argv[3:]
try:
	runpy.run_path(script, run_name='__main__')
finally:
	with open(record_path, 'w') as handle:
		json.dump(saved, handle)
'''


def _qc_run(pool, workdir):
	data_folder, sim = pool.dataset(SPEC)
	# Work on a copy: the script writes QCfigs/ into the Sanity folder, which other tests may share.
	sanity_folder = os.path.join(workdir, 'sanity')
	os.makedirs(sanity_folder)
	for name in os.listdir(data_folder):
		with open(os.path.join(data_folder, name), 'rb') as src, open(os.path.join(sanity_folder, name), 'wb') as dst:
			dst.write(src.read())
	record = os.path.join(workdir, 'record.json')
	proc = subprocess.run([sys.executable, '-c', WRAPPER, record, SCRIPT, '--sanity_folder', sanity_folder],
	                      capture_output=True, text=True, env=bonsai_env(), cwd=workdir, timeout=120)
	return proc, sim, sanity_folder, record


d_opt_sanity_qcs = bonsai_job_fixture('d_opt_sanity_qcs', _qc_run)


@pytest.fixture(scope='module')
def qc(d_opt_sanity_qcs):
	proc, sim, sanity_folder, record = d_opt_sanity_qcs
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	with open(record) as handle:
		saved = {entry['file']: entry for entry in json.load(handle)}
	return sim, sanity_folder, saved


def test_four_png_figures_in_qcfigs(qc):
	_, sanity_folder, saved = qc
	assert set(saved) == set(FIGURES)
	for name in FIGURES:
		path = os.path.join(sanity_folder, 'QCfigs', name)
		with open(path, 'rb') as handle:
			assert handle.read(8) == PNG_SIGNATURE, name


def test_gene_mean_and_variance_figures_show_the_sanity_estimates(qc):
	sim, _, saved = qc
	means = saved['mean_LTQ_histogram.png']
	assert means['calls'][0][0] == 'hist'
	np.testing.assert_allclose(means['calls'][0][1], sim.gene_means, rtol=1e-12)
	assert means['xlabel'] == 'Mean LTQ per gene'
	variances = saved['var_histogram.png']
	np.testing.assert_allclose(variances['calls'][0][1], sim.gene_variances, rtol=1e-12)
	assert variances['yscale'] == 'log'
	scatter = saved['scatter_mean_var.png']
	assert scatter['calls'][0][0] == 'scatter'
	np.testing.assert_allclose(scatter['calls'][0][1], sim.gene_means, rtol=1e-12)
	np.testing.assert_allclose(scatter['calls'][0][2], sim.gene_variances, rtol=1e-12)
	assert scatter['yscale'] == 'log'


def test_zscore_figure_counts_genes_above_one(qc):
	sim, _, saved = qc
	delta, d_delta = sim.sanity_posteriors()
	z_g = np.mean((delta - delta.mean(axis=1, keepdims=True)) ** 2 / d_delta ** 2, axis=1)
	n_above = int(np.sum(z_g > 1))
	assert 0 < n_above < sim.n_genes
	figure = saved['zscores_histogram.png']
	kind, values, cumulative = figure['calls'][0]
	assert kind == 'hist' and cumulative == -1
	# The axis label writes z_g itself; the values drawn are z_g or its square root (the count above 1 is the same).
	values = np.asarray(values)
	assert np.allclose(values, z_g, rtol=1e-9) or np.allclose(values, np.sqrt(z_g), rtol=1e-9)
	assert '{} genes with zscore > 1'.format(n_above) in figure['title']
