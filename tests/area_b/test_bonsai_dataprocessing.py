"""
The tree side of bonsai/bonsai_dataprocessing.py (area B): the SPR and NNI drivers that bonsai_main.py calls
(do_spr_moveset, do_spr_moves_with_postprocessing, extensive_spr_postprocessing, nnnReorderRandom, nnnReorder)
and the tree methods of SCData. The data side of the module belongs to area A
(tests/area_a/test_bonsai_dataprocessing.py).

The drivers run in-process on a 9-cell synthetic set: initializeSCData reads it, the cells' data are put on a
deliberately wrong tree, and the tree is stored the way bonsai_main.py stores it before each step. Expected values
come from the dense reference likelihood and the generator's true tree.
"""
import json
import logging
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import bonsai.bonsai_globals as bs_glob
import bonsai.bonsai_dataprocessing as dp
from bonsai.bonsai_helpers import Run_Configs, store_scdata_and_communicate_path

from area_b.helpers_b import REPO_ROOT, bonsai_nodes, to_node
from bonsai_testdata import DatasetSpec, make_dataset
from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testref import build_bonsai_tree, reference_bonsai_loglik
from bonsai_testrun import bonsai_env, mpiexec_path
from bonsai_testtrees import clades, parse_newick, rf_distance, splits, write_newick

TRUTH = '(((C0:0.1,C1:0.1)a:1.0,(C2:0.1,C3:0.1)b:1.0)x:1.5,((C4:0.1,C5:0.1)c:1.0,(C6:0.1,C7:0.1)d:1.0)y:1.5,C8:1.0);'
# C0 moved next to C7: one prune and regraft away from TRUTH.
SPR_WRONG = '((C1:1.2,(C2:0.1,C3:0.1)b:1.0)x:0.5,((C4:0.1,C5:0.1)c:1.0,(C6:0.1,(C7:0.1,C0:1.5)e:0.1)d:1.0)y:0.5,' \
            'C8:1.0)r;'
# C1 and b swapped across the edge above a: one nearest-neighbour interchange away from TRUTH.
NNI_WRONG = '(((C0:0.1,(C2:0.1,C3:0.1)b:1.0)a:1.0,C1:0.1)x:0.5,((C4:0.1,C5:0.1)c:1.0,(C6:0.1,C7:0.1)d:1.0)y:0.5,' \
            'C8:1.0)r;'
# TRUTH with the four cells under x in one polytomy.
POLYTOMY = '((C0:1.1,C1:1.1,C2:1.1,C3:1.1)x:0.5,((C4:0.1,C5:0.1)c:1.0,(C6:0.1,C7:0.1)d:1.0)y:0.5,C8:1.0)r;'


@pytest.fixture(autouse=True)
def _quiet_bonsai_logger():
	logger = logging.getLogger('myapp')
	level = logger.level
	logger.setLevel(logging.ERROR)
	yield
	logger.setLevel(level)


class Setup:
	"""args, the SCData with a given tree, the cells' data in Bonsai's units and the truth."""

	def __init__(self, args, scdata, means, variances, truth):
		self.args, self.scdata, self.means, self.variances, self.truth = args, scdata, means, variances, truth

	def reference(self, tree):
		return reference_bonsai_loglik(tree, self.means, self.variances)


@pytest.fixture
def make_setup(tmp_path, monkeypatch):
	"""
	Build a Setup whose tree is the given newick over the 9 cells, branch lengths optimised. The working directory
	moves into tmp_path: SCData(onlyObject=True) creates '../data/<dataset>' next to it (tests/README.md, "Known
	bugs").
	"""
	cwd = tmp_path / 'work' / 'cwd'
	cwd.mkdir(parents=True)
	monkeypatch.chdir(cwd)
	sim = make_dataset(DatasetSpec(newick=TRUTH, n_genes=30, seed=0, noise_sd=0.05), str(tmp_path / 'data'))
	args = Run_Configs(create_empty_configs=True)
	args.dataset, args.data_folder, args.results_folder = 'bonsai_test', str(tmp_path / 'data'), str(tmp_path / 'res')
	args.verbose = False

	def make(newick):
		scdata = dp.initializeSCData(args, createStarTree=True, getOrigData=False, otherRanksMinimalInfo=True,
		                             verbose=False)
		means = {child.nodeId: child.ltqs for child in scdata.tree.root.childNodes}
		variances = {child.nodeId: child.getLtqsVars() for child in scdata.tree.root.childNodes}
		scdata.tree = build_bonsai_tree(parse_newick(newick), means, variances)
		scdata.tree.root.storeParent()
		scdata.tree.root.deleteParentsWithOneChild()
		scdata.tree.optTimes(singleProcess=True)
		scdata.tree.nNodes = bs_glob.nNodes
		return Setup(args, scdata, means, variances, sim.tree)

	return make


def _stored(setup):
	return store_scdata_and_communicate_path(setup.scdata, specific_results_folder='spr_intermediates',
	                                         general_results_folder=setup.args.results_folder)


def _check_consistent(setup, scdata):
	tree = to_node(scdata.tree.root)
	assert sorted(tree.leaf_names()) == sorted(setup.truth.leaf_names())
	loglik = scdata.tree.calcLogLComplete()
	assert loglik == pytest.approx(setup.reference(tree), rel=1e-9)
	return tree, loglik


# ================================================================= SPR drivers

@pytest.mark.parametrize('strategy', ['large_tree', 'long_branch_sorted', 'super_sure'])
def test_do_spr_moveset_moves_the_misplaced_cell_back(strategy, make_setup):
	"""Every strategy of the spr_strategy option (YAML comment: 'large_tree', 'long_branch_sorted', 'super_sure')
	brings C0 back next to C1 and raises the likelihood; the first moveset is stored for --pickup_intermediate."""
	setup = make_setup(SPR_WRONG)
	start = setup.scdata.tree.calcLogLComplete()
	result = dp.do_spr_moveset(_stored(setup), setup.args, strategy=strategy)
	tree, loglik = _check_consistent(setup, result)
	assert loglik > start + 10.0
	assert frozenset(['C0', 'C1']) in clades(tree, min_length=1e-9)
	stored = os.listdir(os.path.join(setup.args.results_folder, 'spr_intermediates'))
	assert any(name.startswith('after_sprmoveset1_') for name in stored)
	assert any(name.startswith('after_sprmoveset2_') for name in stored) == (strategy == 'super_sure')


def test_do_spr_moveset_picks_up_the_first_moveset(make_setup):
	"""With pickup_intermediate, the stored result of the first moveset is the starting point: the run gives the
	same tree as the uninterrupted one."""
	setup = make_setup(SPR_WRONG)
	path = _stored(setup)
	full = dp.do_spr_moveset(path, setup.args, strategy='long_branch_sorted')
	expected = write_newick(to_node(full.tree.root))
	picked = dp.do_spr_moveset(path, setup.args, strategy='long_branch_sorted', pickup_intermediate=True)
	assert write_newick(to_node(picked.tree.root)) == expected
	assert picked.tree.calcLogLComplete() == pytest.approx(full.tree.calcLogLComplete(), rel=1e-12)


@pytest.mark.parametrize('large_tree', [False, True])
def test_do_spr_moves_with_postprocessing(large_tree, make_setup):
	setup = make_setup(SPR_WRONG)
	start = setup.scdata.tree.calcLogLComplete()
	result = dp.do_spr_moves_with_postprocessing(_stored(setup), setup.args, select_cand='long_branches_first',
	                                             select_target='all', large_tree=large_tree)
	tree, loglik = _check_consistent(setup, result)
	assert loglik > start + 10.0
	assert frozenset(['C0', 'C1']) in clades(tree, min_length=1e-9)


def test_extensive_spr_postprocessing_resolves_a_polytomy(make_setup):
	setup = make_setup(POLYTOMY)
	start = setup.scdata.tree.calcLogLComplete()
	result = dp.extensive_spr_postprocessing(setup.args, setup.scdata)
	tree, loglik = _check_consistent(setup, result)
	assert loglik > start
	assert rf_distance(tree, setup.truth) == 0
	assert result.tree.nNodes == bs_glob.nNodes == len(list(tree.iter_nodes()))


# ================================================================= NNI drivers

def test_nnnReorderRandom_then_nnnReorder_fix_a_wrong_interchange(make_setup):
	setup = make_setup(NNI_WRONG)
	start = setup.scdata.tree.calcLogLComplete(loglikVarCorr=setup.scdata.metadata.loglikVarCorr)
	setup.scdata.storeTreeInFolder(setup.scdata.result_path('before_nnn'), with_coords=True)
	results = setup.args.results_folder
	stored_ind, tmp_folder = dp.nnnReorderRandom(setup.args, 'before_nnn', randomMoves=5, randomTries=2,
	                                             resultsFolder=results)
	assert stored_ind == 0 and tmp_folder == os.path.join(results, 'intermediate_trees')
	assert os.listdir(tmp_folder) == ['nnn_tree_0'] and not os.path.exists(os.path.join(results, 'random_trees'))
	picked = dp.loadReconstructedTreeAndData(setup.args, os.path.join(tmp_folder, 'nnn_tree_0'))
	assert picked.tree.calcLogLComplete(loglikVarCorr=picked.metadata.loglikVarCorr) >= start - 1e-9
	result = dp.nnnReorder(setup.args, tmp_folder, stored_ind, maxMoves=100)
	tree, loglik = _check_consistent(setup, result)
	assert rf_distance(tree, setup.truth) == 0
	assert loglik > setup.reference(to_node(picked.tree.root))
	assert not os.path.exists(tmp_folder)


def test_nnnReorderRandom_without_random_moves_keeps_the_tree(make_setup):
	setup = make_setup(NNI_WRONG)
	expected = clades(to_node(setup.scdata.tree.root))
	setup.scdata.storeTreeInFolder(setup.scdata.result_path('before_nnn'), with_coords=True)
	stored_ind, tmp_folder = dp.nnnReorderRandom(setup.args, 'before_nnn', randomMoves=0,
	                                             resultsFolder=setup.args.results_folder)
	kept = dp.loadReconstructedTreeAndData(setup.args, os.path.join(tmp_folder, 'nnn_tree_%d' % stored_ind))
	assert clades(to_node(kept.tree.root)) == expected


def test_nnnReorderRandom_picks_up_the_latest_stored_tree(make_setup):
	"""With pickup_intermediate, the highest-numbered nnn_tree_<k> in intermediate_trees is the starting point and
	no random moves are made."""
	setup = make_setup(NNI_WRONG)
	results = setup.args.results_folder
	setup.scdata.storeTreeInFolder(setup.scdata.result_path('before_nnn'), with_coords=True)
	truth_setup = make_setup('(' + TRUTH[1:-2] + ')r;')
	truth_setup.scdata.storeTreeInFolder(os.path.join(results, 'intermediate_trees', 'nnn_tree_3'), with_coords=True)
	setup.args.pickup_intermediate = True
	stored_ind, tmp_folder = dp.nnnReorderRandom(setup.args, 'before_nnn', randomMoves=5, randomTries=2,
	                                             resultsFolder=results)
	assert stored_ind == 3
	loaded = dp.loadReconstructedTreeAndData(setup.args, os.path.join(tmp_folder, 'nnn_tree_3'))
	assert rf_distance(to_node(loaded.tree.root), setup.truth) == 0


# ================================================================= SCData tree methods

def _bare_scdata(cell_ids):
	scdata = dp.SCData.__new__(dp.SCData)
	scdata.metadata = dp.Metadata()
	scdata.metadata.cellIds = list(cell_ids)
	scdata.metadata.nCells = len(cell_ids)
	return scdata


def _tree_from_lists(edges, dists, names):
	from bonsai_testtrees import Node
	nodes = {}
	for (parent, child), dist in zip(edges, dists):
		nodes.setdefault(parent, Node(names[parent], None))
		nodes[child] = Node(names[child], dist)
		nodes[parent].children.append(nodes[child])
	return nodes[edges[0][0]]


@pytest.mark.parametrize('newick', ['((A:1.0,B:2.0)x:3.0,C:4.0,(D:0.5,E:1.5,F:2.5)y:0.25)r;',
                                    '(((A:1.0,B:2.0)x:3.0,C:4.0)z:0.5,(D:0.5,E:1.5)y:0.25,F:2.0)r;'])
def test_compile_tree_from_mergers_rebuilds_the_tree_of_a_merge_list(newick):
	"""Tree.getMergeList and SCData.compile_tree_from_mergers are inverses: same leaves, same splits once the
	zero-length edges that stand for polytomies are collapsed, and the same distance between every pair of cells."""
	from area_b.helpers_b import leaf_data, leaf_pairs_distances
	tree = parse_newick(newick)
	names = tree.leaf_names()
	bonsai_tree = build_bonsai_tree(tree, *leaf_data(names, 1, seed=0))
	scdata = _bare_scdata(names)
	scdata.mergers = bonsai_tree.getMergeList()
	edges, dists, vert_names, starry = scdata.compile_tree_from_mergers()
	assert starry is False
	rebuilt = _tree_from_lists(edges, dists, vert_names)
	assert sorted(rebuilt.leaf_names()) == sorted(names)
	assert splits(rebuilt) == splits(tree)
	expected = leaf_pairs_distances(tree)
	for pair, length in leaf_pairs_distances(rebuilt).items():
		assert length == pytest.approx(expected[pair], rel=1e-12)
	assert all(name.startswith('internal_') for name in vert_names if name not in names)


def test_compile_tree_from_mergers_of_an_unfinished_merge_starts_from_a_star():
	scdata = _bare_scdata(['A', 'B', 'C', 'D'])
	edges, dists, names, starry = scdata.compile_tree_from_mergers(mergers=[(0, 1, 0.5, 0.25)])
	assert starry is True
	rebuilt = _tree_from_lists(edges, dists, names)
	assert write_newick(rebuilt) == '((A:0.5,B:0.25)internal_1:1.0,C:1.0,D:1.0)internal_0;'


def test_communicate_tree_topology_in_one_process_keeps_the_tree():
	scdata = _bare_scdata(['A', 'B', 'C'])
	from area_b.helpers_b import leaf_data
	scdata.tree = build_bonsai_tree(parse_newick('(A:1,B:2,C:3)r;'), *leaf_data(['A', 'B', 'C'], 1, 0))
	tree = scdata.tree
	scdata.communicate_tree_topology()
	assert scdata.tree is tree


TOPOLOGY_SCRIPT = textwrap.dedent("""
	import json, os, sys
	repo, out = sys.argv[1:3]
	sys.path.insert(0, repo)
	sys.path.insert(1, os.path.join(repo, 'tests'))
	import bonsai.mpi_wrapper as mpi_wrapper
	mpi_wrapper.mpi_init()
	import bonsai.bonsai_dataprocessing as dp
	from area_b.helpers_b import leaf_data, to_node
	from bonsai_testref import build_bonsai_tree
	from bonsai_testtrees import parse_newick, write_newick
	scdata = dp.SCData.__new__(dp.SCData)
	rank = mpi_wrapper.get_process_rank()
	if rank == 0:
		names = ['A', 'B', 'C', 'D']
		scdata.tree = build_bonsai_tree(parse_newick('((A:1,B:2)x:0.5,C:3,D:4)r;'), *leaf_data(names, 2, 0))
	else:
		scdata.tree = None
	scdata.communicate_tree_topology()
	node = to_node(scdata.tree.root)
	parents_ok = all(child.parentNode is parent for parent in [scdata.tree.root] + scdata.tree.root.childNodes
	                 for child in parent.childNodes)
	with open(os.path.join(out, 'rank%d.json' % rank), 'w') as handle:
		json.dump({'newick': write_newick(node), 'has_data': scdata.tree.root.childNodes[1].ltqs is not None,
		           'parents_ok': parents_ok}, handle)
""")


def _topology_job(pool, workdir):
	script = os.path.join(workdir, 'topology.py')
	with open(script, 'w') as handle:
		handle.write(TOPOLOGY_SCRIPT)
	cmd = [mpiexec_path(), '--bind-to', 'none', '-n', '2', sys.executable, '-m', 'mpi4py', script, REPO_ROOT, workdir]
	proc = subprocess.run(cmd, capture_output=True, text=True, env=bonsai_env(), timeout=200)
	out = {}
	for rank in (0, 1):
		path = os.path.join(workdir, 'rank%d.json' % rank)
		out[rank] = json.load(open(path)) if os.path.exists(path) else None
	return proc.returncode, proc.stderr[-3000:], out


b_topology_job = bonsai_job_fixture('b_topology_job', _topology_job, slots=2, mpi=True)


@pytest.mark.mpi
def test_communicate_tree_topology_sends_the_topology_to_every_rank(b_topology_job):
	returncode, stderr, out = b_topology_job
	assert returncode == 0 and out[0] is not None and out[1] is not None, stderr
	assert out[1]['newick'] == out[0]['newick'] == '((A:1.0,B:2.0)x:0.5,C:3.0,D:4.0)root;'
	# Rank 0 keeps its tree with the data; what else the broadcast carries is not pinned.
	assert out[0]['has_data'] is True
	assert out[1]['parents_ok'] is True


def _clustered_scdata():
	"""Six cells on four vertices (cells 0, 1 on vertex 0; 2 on 1; 3, 4, 5 on 3; vertex 2 has no cell)."""
	scdata = _bare_scdata(['c%d' % ind for ind in range(6)])
	scdata.nVerts = 4
	scdata.cellIndToVertInd = {0: 0, 1: 0, 2: 1, 3: 3, 4: 3, 5: 3}
	scdata.nCellsPerVert = np.array([2.0, 1.0, 0.0, 3.0])
	bs_glob.nGenes = 2
	return scdata


def test_merge_clusters_acc_cellstates_sums_counts_and_averages_positions():
	scdata = _clustered_scdata()
	scdata.umiCounts = np.array([[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]])
	scdata.originalData = dp.OriginalData()
	scdata.originalData.ltqs = np.array([[1.0, 3.0, 5.0, 0.0, 3.0, 6.0], [2.0, 2.0, -1.0, 1.0, 1.0, 4.0]])
	scdata.merge_clusters_acc_cellstates()
	np.testing.assert_array_equal(scdata.vertUMIs, [[3, 3, 0, 15], [30, 30, 0, 150]])
	np.testing.assert_allclose(scdata.vertLtqs, [[2.0, 5.0, 0.0, 3.0], [2.0, -1.0, 0.0, 2.0]])
	empty = _clustered_scdata()
	empty.umiCounts, empty.originalData = None, None
	empty.merge_clusters_acc_cellstates()
	assert empty.vertUMIs is None and empty.vertLtqs is None


def test_set_node_edge_style_sizes_nodes_by_cell_count(monkeypatch):
	"""Vertices are drawn with size 4 + 20 sqrt(number of cells); vertices without cells are grey. The function
	imports bonsai_helpers as a top-level module, as it is when bonsai/bonsai_main.py runs as a script."""
	monkeypatch.syspath_prepend(os.path.join(REPO_ROOT, 'bonsai'))
	import bonsai_helpers as top_level_helpers
	scdata = _clustered_scdata()
	node_style, edge_style = scdata.set_node_edge_style()
	style = node_style.item()
	np.testing.assert_allclose(style['s'], [4 + 20 * np.sqrt(2), 4 + 20.0, 4.0, 4 + 20 * np.sqrt(3)])
	assert style['color'][2] == top_level_helpers.grey and style['edgecolor'][2] == top_level_helpers.grey
	assert style['color'][0] == top_level_helpers.custom_colors_rgba[0]
	assert edge_style == {'color': top_level_helpers.blackish, 'linewidth': 0.375}
	scdata.vertUMIs = np.array([[1, 10, 1, 100], [0, 0, 0, 0]])
	style = scdata.set_node_edge_style(style_type='umi_counts')[0].item()
	expected = top_level_helpers.gradient_colors((np.log([1, 10, 1, 100]) - 0) / np.log(100))
	assert style['color'][0] == tuple(expected[0]) and style['color'][1] == tuple(expected[1])
	assert style['color'][3] == tuple(expected[3]) and style['color'][2] == top_level_helpers.grey


def test_nnnReorderRandom_keeps_the_original_tree_when_random_trees_are_worse(make_setup):
	"""From the true tree with optimised branch lengths, random interchanges only lower the likelihood: the original
	tree is the one stored for the greedy phase."""
	setup = make_setup('(' + TRUTH[1:-2] + ')r;')
	start = setup.scdata.tree.calcLogLComplete(loglikVarCorr=setup.scdata.metadata.loglikVarCorr)
	expected = clades(to_node(setup.scdata.tree.root), min_length=1e-9)
	setup.scdata.storeTreeInFolder(setup.scdata.result_path('before_nnn'), with_coords=True)
	stored_ind, tmp_folder = dp.nnnReorderRandom(setup.args, 'before_nnn', randomMoves=5, randomTries=2,
	                                             resultsFolder=setup.args.results_folder)
	kept = dp.loadReconstructedTreeAndData(setup.args, os.path.join(tmp_folder, 'nnn_tree_%d' % stored_ind))
	assert clades(to_node(kept.tree.root), min_length=1e-9) == expected
	assert kept.tree.calcLogLComplete(loglikVarCorr=kept.metadata.loglikVarCorr) == pytest.approx(start, rel=1e-9)
