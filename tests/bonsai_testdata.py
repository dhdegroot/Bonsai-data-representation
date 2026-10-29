"""
Deterministic synthetic input for Bonsai.

A known tree (a newick string with branch lengths) is simulated with Brownian
diffusion of every feature from the root, at unit rate in Bonsai's rescaled
units, and the leaves become cells. The measurements are then written in the
layouts Bonsai reads:

- Sanity 2.0 and newer: delta.txt, d_delta.txt, mu.txt, variance.txt and a
  sanity_command.txt whose header names the version and the method;
- Sanity older than 2.0: delta_vmax.txt, d_delta_vmax.txt, mu_vmax.txt and
  variance_vmax.txt, without sanity_command.txt;
- other data: features.txt and, optionally, standard_deviations.txt.

cellID.txt and geneID.txt are optional in every layout.

The model, per gene g with prior variance v_g and baseline mu_g:
	y(child) = y(parent) + N(0, t)                 node positions, rescaled units
	x_gc     = mu_g + sqrt(v_g) * y_c              true log transcription quotient
	m_gc     = x_gc + N(0, s_g^2), s_g = noise_sd * sqrt(v_g)   the measurement
so the likelihood Bonsai should reconstruct has mean m_gc and variance s_g^2,
which is noise_sd^2 after Bonsai rescales by v_g. Sanity reports the posterior
under the prior delta ~ N(0, v_g):
	d_delta^2 = v_g s_g^2 / (v_g + s_g^2),  delta = d_delta^2 (m_gc - mu_g) / s_g^2
and Bonsai divides that prior out again to get back m_gc and s_g^2.

Noise genes do not diffuse (x_gc = mu_g) and have a small Sanity prior
variance, so their signal-to-noise ratio stays under the default cutoff of 1.
"""
import json
import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from bonsai_testtrees import Node, parse_newick, write_newick, clades, splits

SANITY_HEADER = ('# Timestamp: 2026-01-01 00:00:00\n'
                 '# Sanity version: {version}\n'
                 '# Method: {method}\n'
                 'Sanity -f counts.tsv -d sanity_out -e 1 -v_m {method}\n')


# ---------------------------------------------------------------- tree presets

def balanced_newick(depth, internal_length=0.5, leaf_length=0.1, prefix='C'):
	"""A balanced binary tree with 2**depth leaves named <prefix><path in binary>."""

	def build(path, level):
		if level == depth:
			return '%s%s:%r' % (prefix, path, leaf_length)
		inner = build(path + '0', level + 1) + ',' + build(path + '1', level + 1)
		return '(%s)' % inner + ('' if level == 0 else ':%r' % internal_length)

	return build('', 0) + ';'


def caterpillar_newick(n_leaves, internal_length=0.5, leaf_length=0.1, prefix='C'):
	"""A fully unbalanced (ladder) binary tree: (((C0,C1),C2),C3)..."""
	out = '%s0:%r,%s1:%r' % (prefix, leaf_length, prefix, leaf_length)
	out = '(' + out + ')'
	for ind in range(2, n_leaves):
		out = '(%s:%r,%s%d:%r)' % (out, internal_length, prefix, ind, leaf_length)
	return out + ';'


# A small multifurcating tree: two polytomies of three and four leaves under a
# binary split, plus a cherry.
MULTIFURCATING_NEWICK = ('((M0:0.1,M1:0.1,M2:0.1):0.6,((M3:0.1,M4:0.1,M5:0.1,M6:0.1):0.6,'
                         '(M7:0.1,M8:0.1):0.6):0.6);')


@dataclass(frozen=True)
class DatasetSpec:
	"""Everything that determines a synthetic dataset. Hashable, so it can key a cache."""
	newick: str = balanced_newick(3)
	n_genes: int = 60
	n_noise_genes: int = 0
	noise_sd: float = 0.1
	seed: int = 0
	n_duplicates: int = 0
	layout: str = 'sanity2'  # 'sanity2', 'sanity_legacy' or 'features'
	sanity_method: str = 'MAP'
	sanity_version: str = '2.0.0'
	with_ids: bool = True
	with_stds: bool = True
	gene_variance_range: tuple = (0.5, 2.0)
	gene_mean_range: tuple = (-3.0, -1.0)
	noise_gene_variance: float = 0.5
	noise_gene_sd: float = 1.0


@dataclass
class SimData:
	"""The simulated truth and measurements. Matrices are genes x cells (Bonsai's file layout)."""
	spec: DatasetSpec
	tree: Node
	truth_newick: str
	cell_ids: list
	gene_ids: list
	signal_genes: np.ndarray
	gene_variances: np.ndarray
	gene_means: np.ndarray
	positions: dict
	measured: np.ndarray
	measured_sd: np.ndarray
	duplicate_of: dict = field(default_factory=dict)

	@property
	def n_cells(self):
		return len(self.cell_ids)

	@property
	def n_genes(self):
		return len(self.gene_ids)

	def sanity_posteriors(self):
		"""The (delta, d_delta) Sanity would report under its prior N(0, v_g) on delta."""
		v = self.gene_variances[:, None]
		s2 = self.measured_sd ** 2
		post_var = v * s2 / (v + s2)
		delta = post_var * (self.measured - self.gene_means[:, None]) / s2
		return delta, np.sqrt(post_var)

	def bonsai_units(self, genes=None):
		"""
		The likelihood means and variances in Bonsai's rescaled units (divided by
		sqrt(v_g) and v_g) for the given gene indices, as dictionaries from cell
		ID to a vector over genes. Feed these to the reference likelihood.
		"""
		genes = np.arange(self.n_genes) if genes is None else np.asarray(genes)
		scale = np.sqrt(self.gene_variances[genes])[:, None]
		means = self.measured[genes] / scale
		variances = (self.measured_sd[genes] / scale) ** 2
		return ({cell: means[:, ind] for ind, cell in enumerate(self.cell_ids)},
		        {cell: variances[:, ind] for ind, cell in enumerate(self.cell_ids)})

	def truth_clades(self, min_length=0.0):
		return clades(self.tree, min_length=min_length)

	def truth_splits(self, min_length=0.0):
		return splits(self.tree, min_length=min_length)


def simulate(spec):
	"""Simulate a dataset from a DatasetSpec. The same spec gives the same numbers."""
	rng = np.random.default_rng(spec.seed)
	tree = parse_newick(spec.newick)
	for node in tree.iter_nodes():
		if node is not tree and node.length is None:
			raise ValueError('every non-root edge needs a length')
	leaves = tree.leaves()
	names = [leaf.name for leaf in leaves]
	if len(set(names)) != len(names) or any(name is None for name in names):
		raise ValueError('leaf names must be present and unique')
	for name in names:
		if name.startswith('internal_') or any(c in name for c in ':,();\t '):
			raise ValueError('leaf name %r collides with Bonsai naming or newick syntax' % name)

	# Duplicated cells: the first n_duplicates leaves get a twin with identical measurements.
	duplicate_of = {}
	for leaf in leaves[:spec.n_duplicates]:
		twin_name = leaf.name + 'd'
		duplicate_of[twin_name] = leaf.name
		original = Node(leaf.name, 0.0)
		twin = Node(twin_name, 0.0)
		leaf.name = None
		leaf.children = [original, twin]
	if spec.n_duplicates:
		counter = 0
		for node in tree.iter_nodes():
			if node.name is None and node is not tree:
				node.name = 'dup_anc%d' % counter
				counter += 1
	truth_newick = write_newick(tree)

	n_signal = spec.n_genes
	n_total = spec.n_genes + spec.n_noise_genes
	gene_ids = ['G%03d' % ind for ind in range(n_total)]
	signal = np.zeros(n_total, dtype=bool)
	signal[:n_signal] = True
	order = rng.permutation(n_total)
	signal = signal[order]

	lo, hi = spec.gene_variance_range
	gene_variances = np.where(signal, rng.uniform(lo, hi, n_total), spec.noise_gene_variance)
	lo, hi = spec.gene_mean_range
	gene_means = rng.uniform(lo, hi, n_total)

	# Brownian motion from the root in rescaled units; noise genes stay at zero.
	positions = {id(tree): np.zeros(n_total)}
	for node in tree.iter_nodes():
		for child in node.children:
			step = rng.normal(0.0, math.sqrt(child.length), n_total) if child.length > 0 else np.zeros(n_total)
			positions[id(child)] = positions[id(node)] + np.where(signal, step, 0.0)

	cells = [leaf for leaf in tree.leaves()]
	cell_ids = [leaf.name for leaf in cells]
	scale = np.sqrt(gene_variances)
	sd = np.where(signal, spec.noise_sd * scale, spec.noise_gene_sd)
	measured = np.zeros((n_total, len(cells)))
	measured_sd = np.repeat(sd[:, None], len(cells), axis=1)
	by_name = {}
	for ind, leaf in enumerate(cells):
		if leaf.name in duplicate_of:
			continue
		truth = gene_means + scale * positions[id(leaf)]
		measured[:, ind] = truth + rng.normal(0.0, 1.0, n_total) * sd
		by_name[leaf.name] = ind
	for ind, leaf in enumerate(cells):
		if leaf.name in duplicate_of:
			measured[:, ind] = measured[:, by_name[duplicate_of[leaf.name]]]

	named_positions = {}
	for node in tree.iter_nodes():
		if node.name is not None:
			named_positions[node.name] = positions[id(node)]

	return SimData(spec=spec, tree=tree, truth_newick=truth_newick, cell_ids=cell_ids, gene_ids=gene_ids,
	               signal_genes=np.where(signal)[0], gene_variances=gene_variances, gene_means=gene_means,
	               positions=named_positions, measured=measured, measured_sd=measured_sd,
	               duplicate_of=duplicate_of)


# ------------------------------------------------------------------- writers

def _write_matrix(path, matrix):
	with open(path, 'w') as handle:
		for row in matrix:
			handle.write('\t'.join('%.17g' % value for value in row) + '\n')


def _write_vector(path, vector):
	with open(path, 'w') as handle:
		for value in vector:
			handle.write('%.17g\n' % value)


def _write_ids(path, ids):
	with open(path, 'w') as handle:
		for value in ids:
			handle.write('%s\n' % value)


def write_sanity(sim, folder, legacy=False, method='MAP', version='2.0.0', with_ids=True):
	"""Write Sanity output; legacy=True gives the pre-2.0 '_vmax' layout without sanity_command.txt."""
	os.makedirs(folder, exist_ok=True)
	delta, d_delta = sim.sanity_posteriors()
	suffix = '_vmax' if legacy else ''
	_write_matrix(os.path.join(folder, 'delta%s.txt' % suffix), delta)
	_write_matrix(os.path.join(folder, 'd_delta%s.txt' % suffix), d_delta)
	_write_vector(os.path.join(folder, 'mu%s.txt' % suffix), sim.gene_means)
	_write_vector(os.path.join(folder, 'variance%s.txt' % suffix), sim.gene_variances)
	if not legacy:
		with open(os.path.join(folder, 'sanity_command.txt'), 'w') as handle:
			handle.write(SANITY_HEADER.format(version=version, method=method))
	if with_ids:
		_write_ids(os.path.join(folder, 'cellID.txt'), sim.cell_ids)
		_write_ids(os.path.join(folder, 'geneID.txt'), sim.gene_ids)
	return folder


def write_features(sim, folder, with_stds=True, with_ids=True,
                   filenames=('features.txt', 'standard_deviations.txt')):
	"""Write non-Sanity input: the likelihood means and, optionally, their standard deviations."""
	os.makedirs(folder, exist_ok=True)
	_write_matrix(os.path.join(folder, filenames[0]), sim.measured)
	if with_stds:
		_write_matrix(os.path.join(folder, filenames[1]), sim.measured_sd)
	if with_ids:
		_write_ids(os.path.join(folder, 'cellID.txt'), sim.cell_ids)
		_write_ids(os.path.join(folder, 'geneID.txt'), sim.gene_ids)
	return folder


def write_truth(sim, folder):
	"""Record the truth next to the data: truth.json with the newick, clades, cell and gene IDs."""
	truth = {
		'newick': sim.truth_newick,
		'cell_ids': sim.cell_ids,
		'gene_ids': sim.gene_ids,
		'signal_genes': [sim.gene_ids[ind] for ind in sim.signal_genes],
		'clades': sorted(sorted(clade) for clade in sim.truth_clades()),
		'duplicate_of': sim.duplicate_of,
		'spec': {key: (list(value) if isinstance(value, tuple) else value) for key, value in asdict(sim.spec).items()},
	}
	with open(os.path.join(folder, 'truth.json'), 'w') as handle:
		json.dump(truth, handle, indent=1)


def make_dataset(spec, folder):
	"""Simulate the spec and write it to folder in the spec's layout. Returns the SimData."""
	sim = simulate(spec)
	if spec.layout == 'sanity2':
		write_sanity(sim, folder, legacy=False, method=spec.sanity_method, version=spec.sanity_version,
		             with_ids=spec.with_ids)
	elif spec.layout == 'sanity_legacy':
		write_sanity(sim, folder, legacy=True, with_ids=spec.with_ids)
	elif spec.layout == 'features':
		write_features(sim, folder, with_stds=spec.with_stds, with_ids=spec.with_ids)
	else:
		raise ValueError('unknown layout %r' % spec.layout)
	write_truth(sim, folder)
	return sim


def input_is_sanity_output(spec):
	return spec.layout in ('sanity2', 'sanity_legacy')
