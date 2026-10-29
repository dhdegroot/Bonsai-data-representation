"""
downstream_analyses/calc_marker_genes.py (area D): marker features for two groups of cells named in a
.json file, computed from Bonsai-scout's bonsai_vis_data.hdf. Needs the Bonsai-scout environment (h5py);
skipped elsewhere.

The marker score of a feature is the probability that a random cell of group 1 has a higher value than a
random cell of group 2 (calc_marker_genes_helpers docstrings). With error bars (posterior means and
variances) the exact value is the mean over all cell pairs of Phi((m1 - m2) / sqrt(v1 + v2)), computed here
by brute force. The script integrates numerically over group 1's values within two standard deviations, so
its score lies between the exact value minus group 1's mass outside that range and the exact value.
The runs are part of the `scout_rich` job in helpers_d.py. The branch without error bars crashes on every
feature and is not run.
"""
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.scout
h5py = pytest.importorskip('h5py')
pytest.importorskip('tables')
pytest.importorskip('natsort')

import pandas as pd  # noqa: E402

from area_d.helpers_d import (MARKER_GROUPS, gaussian_marker_score, scout_rich,  # noqa: E402,F401
                              two_sigma_truncation)

# Grid error of the script's 300-point integration, measured against adaptive quadrature of the same
# truncated integral on this dataset: below 6e-5.
GRID_ERROR = 2e-4


def _marker_output(run, name):
	output, proc = run.markers[name]
	assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
	table = pd.read_csv(output, sep='\t', index_col=0)
	assert list(table.columns) == ['marker_genes', 'marker_scores']
	return table


def _expected(run, name):
	feature_path, group1, group2 = MARKER_GROUPS[name]
	with h5py.File(run.hdf_path('default'), 'r') as hdf:
		group = hdf[feature_path]
		means, variances = group['means'][()], group['vars'][()]
		gene_ids, node_ids = json.loads(group.attrs['gene_ids']), json.loads(group.attrs['node_ids'])
	cols1 = [node_ids.index(cell) for cell in group1]
	cols2 = [node_ids.index(cell) for cell in group2]
	exact = gaussian_marker_score(means[:, cols1], variances[:, cols1], means[:, cols2], variances[:, cols2])
	return gene_ids, exact, two_sigma_truncation(means[:, cols1], variances[:, cols1])


def _check(run, name):
	gene_ids, exact, truncated = _expected(run, name)
	table = _marker_output(run, name)
	got = dict(zip(table['marker_genes'], table['marker_scores']))
	# With 24 genes and the script's cutoff of 0.5, at least 10 genes on each side mean every gene is reported.
	# (The groups are chosen so that this holds: with fewer than 10 on one side the selection falls back to a
	# branch that picks the wrong end of the scores.)
	scores = np.array(list(got.values()))
	assert np.sum(scores > 0.5) >= 10 and np.sum(scores < 0.5) >= 10
	assert sorted(got) == sorted(gene_ids)
	got = np.array([got[gene] for gene in gene_ids])
	assert np.all(got <= exact + GRID_ERROR)
	assert np.all(got >= exact - truncated - GRID_ERROR)
	# Rows are sorted by how far the score is from 0.5, most extreme first.
	ordered = table['marker_scores'].values
	assert np.all(np.diff(np.minimum(ordered, 1 - ordered)) >= 0)
	return exact


def test_marker_scores_with_error_bars(scout_rich):
	"""data/normalized has posterior variances, so the scores account for the error bars; the table goes to
	--marker_output_file."""
	scout_rich.check('default')
	exact = _check(scout_rich, 'vars')
	assert exact.max() - exact.min() > 0.5


def test_swapped_groups_and_default_output_path(scout_rich):
	"""Swapping the groups turns every exact score s into 1 - s; without --marker_output_file the table is
	written to <results_folder>/output_marker_script.tsv."""
	scout_rich.check('default')
	output, _ = scout_rich.markers['vars_swapped']
	assert output == os.path.join(scout_rich.folder('default'), 'output_marker_script.tsv')
	exact = _check(scout_rich, 'vars_swapped')
	_, exact_forward, _ = _expected(scout_rich, 'vars')
	np.testing.assert_allclose(exact, 1 - exact_forward, atol=1e-12)
