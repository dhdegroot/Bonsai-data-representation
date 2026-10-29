"""
Tests of the test harness itself: the newick parser, the topology comparison,
the synthetic data generator and the reference likelihood, each against cases
worked out by hand.
"""
import math
import os

import numpy as np
import pytest

from bonsai_testtrees import parse_newick, write_newick, clades, splits, rf_distance, path_length, Node
from bonsai_testdata import (DatasetSpec, simulate, make_dataset, write_sanity, write_features, balanced_newick,
                             caterpillar_newick, MULTIFURCATING_NEWICK)
from bonsai_testref import (reference_loglik, reference_loglik_per_gene, reference_posteriors, to_bonsai_units,
                            shared_path_matrix)
from bonsai_testrun import write_config, read_config, stage_folder


# ------------------------------------------------------------------ newick

def test_parse_newick_reads_names_lengths_and_structure():
	tree = parse_newick('((A:1,B:2.5)i1:0.5,C:3e-05)root;')
	assert tree.name == 'root' and tree.length is None
	assert [child.name for child in tree.children] == ['i1', 'C']
	assert tree.children[0].length == 0.5
	assert [(leaf.name, leaf.length) for leaf in tree.leaves()] == [('A', 1.0), ('B', 2.5), ('C', 3e-05)]


def test_parse_newick_allows_missing_names_and_whitespace():
	tree = parse_newick(' ( A : 0.0 , ( B:1 , C:2 ) : 0.25 ) ;')
	assert tree.name is None
	assert tree.children[1].name is None and tree.children[1].length == 0.25
	assert tree.leaf_names() == ['A', 'B', 'C']


@pytest.mark.parametrize('text', ['(A:1,B:2)', '(A:1,B:2;', '(A:1,B:2));', '(A:1,B:2)r;x'])
def test_parse_newick_rejects_malformed_strings(text):
	with pytest.raises((ValueError, IndexError)):
		parse_newick(text)


def test_write_newick_round_trips_bonsai_style_strings():
	text = '((Cell1:0.15223332644265247,Cell2:0.0)internal_4:1e-05,Cell3:0.3)root;'
	assert write_newick(parse_newick(text)) == text


def test_write_newick_writes_numpy_lengths_as_plain_floats():
	tree = Node('root', None, [Node('A', np.float64(0.1)), Node('B', np.float32(0.5)), Node('C', np.int64(2))])
	text = write_newick(tree)
	assert text == '(A:0.1,B:0.5,C:2.0)root;'
	assert [leaf.length for leaf in parse_newick(text).leaves()] == [0.1, 0.5, 2.0]


def test_path_length_sums_edges_through_the_common_ancestor():
	tree = parse_newick('((A:1,B:2)i:0.5,C:3)r;')
	assert path_length(tree, 'A', 'B') == 3.0
	assert path_length(tree, 'A', 'C') == 4.5
	assert path_length(tree, 'i', 'A') == 1.0


# ------------------------------------------------------- clades, splits, RF

def test_clades_are_leaf_sets_below_internal_nodes():
	tree = parse_newick('((A:1,B:1)x:1,(C:1,D:1)y:1,E:1)r;')
	assert clades(tree) == {frozenset('AB'), frozenset('CD')}
	assert clades(tree, include_root=True) == {frozenset('AB'), frozenset('CD'), frozenset('ABCDE')}


def test_rf_distance_of_two_different_quartets_is_two():
	# ((A,B),(C,D)) has the split AB|CD, ((A,C),(B,D)) has AC|BD: one split each, none shared.
	assert rf_distance(parse_newick('((A:1,B:1):1,(C:1,D:1):1);'), parse_newick('((A:1,C:1):1,(B:1,D:1):1);')) == 2


def test_rf_distance_ignores_rooting_and_child_order():
	rooted_mid = parse_newick('((A:1,B:1):1,(C:1,D:1):1);')
	rooted_at_a = parse_newick('(A:1,(B:1,(D:1,C:1):1):1);')
	assert rf_distance(rooted_mid, rooted_at_a) == 0
	assert splits(rooted_mid) == {frozenset('CD')}


def test_rf_distance_on_five_leaves():
	# Splits: AB|CDE and DE|ABC versus AC|BDE and DE|ABC.
	tree1 = parse_newick('((A:1,B:1):1,C:1,(D:1,E:1):1);')
	tree2 = parse_newick('((A:1,C:1):1,B:1,(D:1,E:1):1);')
	assert splits(tree1) == {frozenset('CDE'), frozenset('DE')}
	assert rf_distance(tree1, tree2) == 2


def test_zero_length_edges_are_collapsed_unless_asked_otherwise():
	resolved = parse_newick('((A:1,(B:1,C:1):0.0):1,D:1,E:1);')
	short = parse_newick('((A:1,(B:1,C:1):1e-9):1,D:1,E:1);')
	star_like = parse_newick('((A:1,B:1,C:1):1,D:1,E:1);')
	assert rf_distance(resolved, star_like) == 0
	assert rf_distance(resolved, star_like, min_length=-1.0) == 1
	assert rf_distance(short, star_like) == 1
	assert rf_distance(short, star_like, min_length=1e-6) == 0


def test_rf_distance_requires_equal_leaf_sets():
	with pytest.raises(ValueError):
		rf_distance(parse_newick('(A:1,B:1);'), parse_newick('(A:1,C:1);'))


# ------------------------------------------------------------- generator

def test_tree_presets_have_the_requested_shape():
	balanced = parse_newick(balanced_newick(3, internal_length=0.5, leaf_length=0.1))
	assert len(balanced.leaves()) == 8
	distances = sorted({round(path_length(balanced, 'C000', leaf), 9) for leaf in balanced.leaf_names()[1:]})
	assert distances == [0.2, 1.2, 2.2]
	assert len(splits(balanced)) == 5
	ladder = parse_newick(caterpillar_newick(6))
	assert len(ladder.leaves()) == 6 and len(splits(ladder)) == 3
	assert sorted(len(c) for c in clades(parse_newick(MULTIFURCATING_NEWICK))) == [2, 3, 4, 6]


def test_simulation_truth_is_self_consistent():
	spec = DatasetSpec(newick=balanced_newick(3), n_genes=20, n_noise_genes=5, n_duplicates=2, seed=3)
	sim = simulate(spec)
	truth = parse_newick(sim.truth_newick)
	assert truth.leaf_names() == sim.cell_ids
	assert clades(truth) == sim.truth_clades()
	assert sim.n_cells == 10 and sim.n_genes == 25 and len(sim.signal_genes) == 20
	assert sim.measured.shape == sim.measured_sd.shape == (25, 10)
	# Twins sit on zero-length edges under a shared ancestor and copy the original's measurements.
	for twin, original in sim.duplicate_of.items():
		assert frozenset([twin, original]) in sim.truth_clades()
		np.testing.assert_array_equal(sim.measured[:, sim.cell_ids.index(twin)],
		                              sim.measured[:, sim.cell_ids.index(original)])
		assert truth.find(twin).length == 0.0
	# Noise genes do not diffuse: every node keeps the root's position on them.
	noise = np.setdiff1d(np.arange(25), sim.signal_genes)
	assert all(np.all(position[noise] == 0.0) for position in sim.positions.values())
	# The same spec gives the same numbers.
	np.testing.assert_array_equal(simulate(spec).measured, sim.measured)


def test_measurement_noise_matches_the_spec():
	spec = DatasetSpec(newick=balanced_newick(5), n_genes=400, noise_sd=0.2, seed=7)
	sim = simulate(spec)
	residual = (sim.measured - sim.gene_means[:, None]) / np.sqrt(sim.gene_variances)[:, None]
	residual -= np.array([sim.positions[cell] for cell in sim.cell_ids]).T
	assert residual.std() == pytest.approx(0.2, rel=0.05)
	np.testing.assert_allclose(sim.measured_sd / np.sqrt(sim.gene_variances)[:, None], 0.2)


def _read_matrix(path):
	return np.loadtxt(path, delimiter='\t', ndmin=2)


def test_sanity_output_divides_back_to_the_likelihood(tmp_path):
	"""Undo Sanity's N(0, v) prior by hand, as the Bonsai SI describes, and get the measurements back."""
	sim = simulate(DatasetSpec(newick=balanced_newick(2), n_genes=6, n_noise_genes=2, seed=1))
	write_sanity(sim, tmp_path)
	delta = _read_matrix(tmp_path / 'delta.txt')
	d_delta = _read_matrix(tmp_path / 'd_delta.txt')
	mu = np.loadtxt(tmp_path / 'mu.txt')
	v = np.loadtxt(tmp_path / 'variance.txt')[:, None]
	post_var = d_delta ** 2
	lik_var = 1.0 / (1.0 / post_var - 1.0 / v)
	lik_mean = mu[:, None] + delta / post_var * lik_var
	np.testing.assert_allclose(lik_mean, sim.measured, rtol=1e-10)
	np.testing.assert_allclose(lik_var, sim.measured_sd ** 2, rtol=1e-10)
	assert (tmp_path / 'cellID.txt').read_text().split() == sim.cell_ids
	assert (tmp_path / 'geneID.txt').read_text().split() == sim.gene_ids
	header = (tmp_path / 'sanity_command.txt').read_text().splitlines()
	assert header[1] == '# Sanity version: 2.0.0' and header[2] == '# Method: MAP' and '-e 1 -v_m MAP' in header[3]


def test_legacy_sanity_layout_has_vmax_files_and_no_command_file(tmp_path):
	sim = simulate(DatasetSpec(newick=balanced_newick(2), n_genes=4))
	write_sanity(sim, tmp_path, legacy=True)
	names = sorted(os.listdir(tmp_path))
	assert names == ['cellID.txt', 'd_delta_vmax.txt', 'delta_vmax.txt', 'geneID.txt', 'mu_vmax.txt',
	                 'variance_vmax.txt']


def test_features_layout(tmp_path):
	sim = simulate(DatasetSpec(newick=balanced_newick(2), n_genes=4))
	write_features(sim, tmp_path, with_stds=False, with_ids=False)
	assert sorted(os.listdir(tmp_path)) == ['features.txt']
	np.testing.assert_allclose(_read_matrix(tmp_path / 'features.txt'), sim.measured, rtol=1e-15)


def test_noise_genes_fall_below_the_default_zscore_cutoff():
	"""Signal-to-noise per gene as the YAML comment of zscore_cutoff defines it, and as Bonsai's Sanity reader
	computes it (without subtracting the mean): signal genes far above 1, noise genes below."""
	sim = simulate(DatasetSpec(newick=balanced_newick(4), n_genes=60, n_noise_genes=20, seed=0))
	delta, d_delta = sim.sanity_posteriors()
	z_documented = np.mean((delta - delta.mean(axis=1, keepdims=True)) ** 2 / d_delta ** 2, axis=1)
	z_code = np.mean(delta ** 2 / d_delta ** 2, axis=1)
	noise = np.setdiff1d(np.arange(sim.n_genes), sim.signal_genes)
	for z in (z_documented, z_code):
		assert z[sim.signal_genes].min() > 4.0
		assert z[noise].max() < 1.0


def test_make_dataset_records_the_truth(tmp_path):
	import json
	sim = make_dataset(DatasetSpec(newick=MULTIFURCATING_NEWICK, n_genes=5), tmp_path)
	truth = json.loads((tmp_path / 'truth.json').read_text())
	assert truth['newick'] == sim.truth_newick
	assert sorted(map(frozenset, truth['clades']), key=sorted) == sorted(sim.truth_clades(), key=sorted)


# ------------------------------------------------------ reference likelihood

def test_reference_loglik_two_leaves_by_hand():
	"""Two leaves under the root: x_A - x_B ~ N(0, s) with s = t_A + v_A + t_B + v_B; the root drops out."""
	tree = parse_newick('(A:0.3,B:0.7)r;')
	x_a, x_b, v_a, v_b = 1.2, -0.4, 0.05, 0.2
	s = 0.3 + v_a + 0.7 + v_b
	expected = -0.5 * math.log(2 * math.pi * s) - (x_a - x_b) ** 2 / (2 * s)
	got = reference_loglik(tree, {'A': [x_a], 'B': [x_b]}, {'A': [v_a], 'B': [v_b]})
	assert got == pytest.approx(expected, rel=1e-12)
	# Bonsai's units: twice the log-likelihood plus (C-1) log(2 pi); for two leaves -log(s) - d^2/s.
	assert to_bonsai_units(got, 2, 1) == pytest.approx(-math.log(s) - (x_a - x_b) ** 2 / s, rel=1e-12)


def test_reference_loglik_three_leaves_by_hand():
	"""((A,B)i,C): the contrasts (x_A - x_C, x_B - x_C) are bivariate normal; with a flat root prior their
	density is the likelihood (unit Jacobian)."""
	tree = parse_newick('((A:0.3,B:0.6)i:0.4,C:0.2)r;')
	x = {'A': 0.7, 'B': 1.5, 'C': -0.2}
	v = {'A': 0.1, 'B': 0.05, 'C': 0.3}
	s_a, s_b, s_c, t_i = 0.3 + v['A'], 0.6 + v['B'], 0.2 + v['C'], 0.4
	cov = np.array([[s_a + t_i + s_c, t_i + s_c], [t_i + s_c, s_b + t_i + s_c]])
	d = np.array([x['A'] - x['C'], x['B'] - x['C']])
	expected = -math.log(2 * math.pi) - 0.5 * math.log(np.linalg.det(cov)) - 0.5 * d @ np.linalg.solve(cov, d)
	got = reference_loglik(tree, {k: [val] for k, val in x.items()}, {k: [val] for k, val in v.items()})
	assert got == pytest.approx(expected, rel=1e-12)


def test_reference_loglik_is_invariant_to_rerooting_and_shifts():
	"""Moving the root along the tree or shifting every measurement by a constant leaves the likelihood alone."""
	means = {'A': [0.1, 2.0], 'B': [0.5, 1.0], 'C': [-0.3, 0.2], 'D': [1.1, -1.0]}
	variances = {'A': [0.1, 0.2], 'B': [0.3, 0.1], 'C': [0.05, 0.4], 'D': [0.2, 0.2]}
	rooted = parse_newick('((A:0.3,B:0.2)x:0.25,(C:0.1,D:0.4)y:0.15)r;')
	rerooted = parse_newick('(A:0.3,B:0.2,(C:0.1,D:0.4)y:0.4)x;')
	base = reference_loglik_per_gene(rooted, means, variances)
	np.testing.assert_allclose(reference_loglik_per_gene(rerooted, means, variances), base, rtol=1e-12)
	shifted = {k: [val[0] + 5.0, val[1] - 3.0] for k, val in means.items()}
	np.testing.assert_allclose(reference_loglik_per_gene(rooted, shifted, variances), base, rtol=1e-10)


def test_reference_posteriors_two_leaves_by_hand():
	tree = parse_newick('(A:0.3,B:0.7)r;')
	x_a, x_b, v_a, v_b = 1.2, -0.4, 0.05, 0.2
	means, variances = reference_posteriors(tree, {'A': [x_a], 'B': [x_b]}, {'A': [v_a], 'B': [v_b]})
	# Leaf A: its own measurement combined with B's, carried over the path t_A + t_B (+ v_B).
	far = 0.3 + 0.7 + v_b
	precision = 1 / v_a + 1 / far
	assert variances['A'][0] == pytest.approx(1 / precision, rel=1e-12)
	assert means['A'][0] == pytest.approx((x_a / v_a + x_b / far) / precision, rel=1e-12)
	# Root: both leaves carried over their own edges.
	w_a, w_b = 1 / (0.3 + v_a), 1 / (0.7 + v_b)
	assert variances['r'][0] == pytest.approx(1 / (w_a + w_b), rel=1e-12)
	assert means['r'][0] == pytest.approx((w_a * x_a + w_b * x_b) / (w_a + w_b), rel=1e-12)


def test_shared_path_matrix():
	tree = parse_newick('((A:1,B:2)i:0.5,C:3)r;')
	leaves = tree.leaves()
	np.testing.assert_allclose(shared_path_matrix(tree, leaves), [[1.5, 0.5, 0], [0.5, 2.5, 0], [0, 0, 3]])


# ------------------------------------------------------------------ config

def test_write_config_goes_through_create_config_file(tmp_path):
	path = write_config(tmp_path / 'c.yaml', 'bonsai_test', str(tmp_path / 'data'), str(tmp_path / 'res'),
	                    zscore_cutoff=-1.0, use_knn=5, rescale_by_var=False)
	config = read_config(path)
	assert config['dataset'] == 'bonsai_test'
	assert config['zscore_cutoff'] == -1.0 and config['use_knn'] == 5 and config['rescale_by_var'] is False
	assert config['results_folder'] == str(tmp_path / 'res')
	with pytest.raises(KeyError):
		write_config(tmp_path / 'd.yaml', 'x', 'y', 'z', not_an_option=1)


def test_stage_folder_names():
	assert stage_folder('final') == 'final_bonsai_zscore1.0'
	assert stage_folder('final', -1) == 'final_bonsai'
	assert stage_folder('preprocessed', 0.5) == 'preprocessed_zscore0.5'
	assert stage_folder('reordered_edges') == \
		'intermediate_bonsai_zscore1.0_redoStarry_optTimes_sprMoves_nnnReorder_reorderedEdges'


# ------------------------------------------------------------------ run pool

def test_pool_gives_slots_first_come_first_served():
	"""A job that needs two of two slots, asking while one is taken, gets them before a one-slot job that asked
	after it, although one slot was free for the later job the whole time."""
	import threading
	import time
	from bonsai_testfixtures import BonsaiPool

	pool = BonsaiPool(tmp_path_factory=None, slots=2)
	try:
		order = []
		assert pool._acquire(1) == 1

		def job(name, need):
			pool._acquire(need)
			order.append(name)
			pool._release(need)

		def wait_for_tickets(count):
			deadline = time.time() + 10
			while pool._next_ticket < count and time.time() < deadline:
				time.sleep(0.01)
			assert pool._next_ticket == count

		two = threading.Thread(target=job, args=('two slots', 2))
		two.start()
		wait_for_tickets(2)
		one = threading.Thread(target=job, args=('one slot', 1))
		one.start()
		wait_for_tickets(3)
		time.sleep(0.2)
		assert order == []
		pool._release(1)
		two.join(10)
		one.join(10)
		assert order == ['two slots', 'one slot']
	finally:
		pool.shutdown()
