"""
Shared helpers for the area A tests (data in and out): small synthetic inputs,
run configurations built in-process, and an independent maximum-likelihood
estimate of a feature's variance.
"""
import math
import os
from collections import namedtuple

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from bonsai_testdata import DatasetSpec, balanced_newick, make_dataset

MpiInfo = namedtuple('MpiInfo', ['rank', 'size'])
SERIAL = MpiInfo(0, 1)

# Eight cells on a balanced tree, twelve diffusing genes and four noise genes. The noise genes sit far under the
# default zscore cutoff and the signal genes far above it under every definition of the statistic (checked in
# test_small_spec_is_unambiguous_under_both_zscore_definitions).
SMALL = DatasetSpec(newick=balanced_newick(3, internal_length=0.5, leaf_length=0.1), n_genes=12, n_noise_genes=4,
                    noise_sd=0.1, seed=5, layout='sanity2')


@pytest.fixture
def cwd_in_tmp(tmp_path, monkeypatch):
	"""
	Run the test from an empty folder inside tmp_path. SCData(onlyObject=True) creates '../data/<dataset>' relative
	to the working directory, a folder Bonsai never uses; this keeps that folder inside tmp_path.
	"""
	cwd = tmp_path / 'cwd'
	cwd.mkdir()
	monkeypatch.chdir(cwd)
	return cwd


def dataset(tmp_path, spec=SMALL, name='data'):
	"""Write spec's input to tmp_path/name; returns (folder, sim)."""
	folder = os.path.join(str(tmp_path), name)
	return folder, make_dataset(spec, folder)


def make_args(data_folder, results_folder, **overrides):
	"""A Run_Configs with its built-in defaults and the given paths, as bonsai_main.py passes it around."""
	from bonsai.bonsai_helpers import Run_Configs
	args = Run_Configs(create_empty_configs=True)
	args.dataset = 'bonsai_test'
	args.data_folder = str(data_folder)
	args.results_folder = str(results_folder)
	args.tmp_folder = ''
	for key, value in overrides.items():
		setattr(args, key, value)
	return args


def args_for(spec, data_folder, results_folder, **overrides):
	"""make_args with input_is_sanity_output and filenames_data set for the spec's layout."""
	if spec.layout == 'features':
		overrides.setdefault('input_is_sanity_output', False)
		overrides.setdefault('filenames_data', 'features.txt,standard_deviations.txt')
	else:
		overrides.setdefault('input_is_sanity_output', True)
	return make_args(data_folder, results_folder, **overrides)


def ml_feature_variance(measurements, error_vars):
	"""
	Maximum-likelihood variance v of a feature measured with known error variances, under
	m_c ~ N(mu, v + e_c) with mu profiled out (the precision-weighted mean). Written from the model, not from
	Bonsai's code: a bounded scalar search on log v of the profile log-likelihood.
	"""
	m = np.asarray(measurements, dtype=float)
	e = np.asarray(error_vars, dtype=float)

	def neg_profile(log_v):
		total = math.exp(log_v) + e
		w = 1.0 / total
		mu = np.sum(w * m) / np.sum(w)
		return 0.5 * np.sum(np.log(total) + (m - mu) ** 2 / total)

	res = minimize_scalar(neg_profile, bounds=(-30.0, 10.0), method='bounded', options={'xatol': 1e-12})
	v = math.exp(res.x)
	w = 1.0 / (v + e)
	return v, float(np.sum(w * m) / np.sum(w))


def write_tree_files(folder, tree, names_for_internal='internal_%d'):
	"""
	Write a harness Node tree as edgeInfo.txt and vertInfo.txt in the README's format (vertices numbered
	depth-first from the root at 0, edges parent<TAB>child<TAB>length). Unnamed internal nodes get
	names_for_internal % counter. Returns the list of vertex names by vertInd.
	"""
	os.makedirs(folder, exist_ok=True)
	names = []
	edges = []
	counter = [0]

	def visit(node, parent_vert):
		vert = len(names)
		name = node.name
		if name is None:
			name = 'root' if parent_vert is None else names_for_internal % counter[0]
			if parent_vert is not None:
				counter[0] += 1
		names.append(name)
		if parent_vert is not None:
			edges.append((parent_vert, vert, node.length))
		for child in node.children:
			visit(child, vert)

	visit(tree, None)
	with open(os.path.join(folder, 'edgeInfo.txt'), 'w') as handle:
		for parent, child, length in edges:
			handle.write('%d\t%d\t%.8e\n' % (parent, child, length))
	with open(os.path.join(folder, 'vertInfo.txt'), 'w') as handle:
		handle.write('vertInd\tnodeInd\tvertName\n')
		for vert, name in enumerate(names):
			handle.write('%d\t%d\t%s\n' % (vert, vert, name))
	return names


def small_run_spec():
	"""The one end-to-end run of area A: SMALL through 'bonsai_main.py --step all' with few NNI moves."""
	from bonsai_testfixtures import run_spec
	return run_spec(SMALL, label='area_a_small', nnn_n_randommoves=20, nnn_n_randomtrees=2)


def named_truth(sim):
	"""The generator's tree with every internal node named anc_<k> and the root named 'root'."""
	from bonsai_testtrees import parse_newick
	tree = parse_newick(sim.truth_newick)
	counter = 0
	for node in tree.iter_nodes():
		if node is tree:
			node.name = 'root'
		elif not node.is_leaf:
			node.name = 'anc_%d' % counter
			counter += 1
	return tree


def init_scdata(tmp_path, spec=SMALL, create_star_tree=False, **overrides):
	"""Write spec's input and read it with initializeSCData on one process; returns (scdata, sim, args)."""
	from bonsai.bonsai_dataprocessing import initializeSCData
	folder, sim = dataset(tmp_path, spec)
	args = args_for(spec, folder, os.path.join(str(tmp_path), 'results'), **overrides)
	scdata = initializeSCData(args, createStarTree=create_star_tree, getOrigData=True, allRanks=False)
	return scdata, sim, args


def cell_data(scdata):
	"""The rescaled leaf likelihoods SCData holds, as dictionaries from cell ID to a vector over genes."""
	ids = scdata.metadata.cellIds
	return ({cell: scdata.originalData.ltqs[:, ind] for ind, cell in enumerate(ids)},
	        {cell: scdata.originalData.ltqsVars[:, ind] for ind, cell in enumerate(ids)})


def scdata_with_tree(tmp_path, newick=None, **overrides):
	"""
	SMALL read through initializeSCData, with the true tree (or the tree of `newick`) built on its rescaled
	data. Returns (scdata, sim, args, tree, means, variances), tree being the harness Node tree.
	"""
	from bonsai_testref import build_bonsai_tree
	from bonsai_testtrees import parse_newick
	scdata, sim, args = init_scdata(tmp_path, **overrides)
	tree = named_truth(sim) if newick is None else parse_newick(newick)
	means, variances = cell_data(scdata)
	scdata.tree = build_bonsai_tree(tree, means, variances)
	return scdata, sim, args, tree, means, variances


class FakeComm:
	"""
	Stands in for mpi4py's COMM_WORLD of one rank of a larger run: its rank and size, and a bcast that marks what
	passed through it as coming from the root, so a test can see that a value was broadcast and the received
	value used.
	"""

	def __init__(self, rank=0, size=1):
		self.rank, self.size = rank, size
		self.broadcast = []

	def Get_rank(self):
		return self.rank

	def Get_size(self):
		return self.size

	def bcast(self, data, root=0):
		self.broadcast.append(data)
		return ('from_root', data)

	def barrier(self):
		return None


# ------------------------------------------------------------------ one script run under mpiexec

MPI_SCRIPT = r'''
"""Area A checks under real MPI ranks: mpi_wrapper's collectives, the globals broadcast, and read_and_filter
spread over the ranks. Rank 0 writes everything it learnt to <out>/result.npz and <out>/result.json."""
import json
import os
import sys

import numpy as np

repo, out, sanity_folder, features_folder = sys.argv[1:5]
sys.path.insert(0, repo)
import bonsai.mpi_wrapper as mw
mw.mpi_init()
import bonsai.bonsai_globals as bs_glob
from bonsai.bonsai_helpers import communicateGlobalVars, broadcastRecursiveStruct
from bonsai.bonsai_dataprocessing import read_and_filter

rank, size = mw.get_mpi_info()
checks = {'rank_size': [rank, size], 'first': mw.is_first_process()}
checks['allgather'] = mw.world_allgather(rank * 10)
checks['gather'] = mw.gather({'r': rank})
checks['bcast'] = mw.bcast('from %d' % rank)
doubles = np.arange(4) * 1.5 if rank == 0 else np.zeros(4)
mw.Bcast(doubles, type='double')
ints = np.array([7, 8, 9], dtype='i') if rank == 0 else np.zeros(3, dtype='i')
mw.Bcast(ints, type='int')
other = np.array([2.5, -1.0]) if rank == 0 else np.zeros(2)
mw.Bcast(other, type='raw')
rows = np.full((rank + 1, 3), float(rank)) + np.arange(3)
gathered = mw.GatherNpUnknownSize(rows)
if rank == 0:
	bs_glob.nCells, bs_glob.nGenes, bs_glob.nNodes, bs_glob.max_node_ind, bs_glob.mem_friendly = 11, 22, 33, 44, True
communicateGlobalVars()
deep = None
if rank == 0:
	for _ in range(4000):
		deep = [deep]
deep = broadcastRecursiveStruct(deep)
depth = 0
while deep is not None:
	deep, depth = deep[0], depth + 1
per_rank = {'doubles': doubles.tolist(), 'ints': ints.tolist(), 'other': other.tolist(),
            'globals': [bs_glob.nCells, bs_glob.nGenes, bs_glob.nNodes, bs_glob.max_node_ind, bs_glob.mem_friendly],
            'depth': depth}
all_ranks = mw.gather(per_rank)
info = mw.get_mpi_info()
sanity = read_and_filter(sanity_folder, None, None, True, 1.0, info, tmp_folder=os.path.join(out, 'tmp_sanity'))
features = read_and_filter(features_folder, 'features.txt', 'standard_deviations.txt', False, 1.0, info,
                           tmp_folder=os.path.join(out, 'tmp_features'))
mw.barrier()
if rank == 0:
	checks['per_rank'] = all_ranks
	with open(os.path.join(out, 'result.json'), 'w') as handle:
		json.dump(checks, handle)
	arrays = {'gathered': gathered}
	for name, result in (('sanity', sanity), ('features', features)):
		for key, value in zip(('ltqs', 'ltqs_vars', 'gene_vars', 'n_cells', 'n_genes', 'genes_to_keep',
		                       'stds_found', 'n_genes_orig'), result):
			arrays[name + '_' + key] = np.asarray(value)
	np.savez(os.path.join(out, 'result.npz'), **arrays)
else:
	assert all(value is None for value in sanity + features)
'''


def mpi_job(ranks):
	"""A bonsai_job_fixture function: run MPI_SCRIPT under mpiexec -n <ranks> on SMALL in both layouts."""

	def job(pool, workdir):
		import subprocess
		import sys
		from dataclasses import replace
		from bonsai_testrun import REPO_ROOT, bonsai_env, mpiexec_path
		sanity_folder, _ = pool.dataset(SMALL)
		features_folder, _ = pool.dataset(replace(SMALL, layout='features'))
		script = os.path.join(workdir, 'area_a_mpi.py')
		with open(script, 'w') as handle:
			handle.write(MPI_SCRIPT)
		cmd = [mpiexec_path(), '--bind-to', 'none', '-n', str(ranks), sys.executable, '-m', 'mpi4py', script,
		       REPO_ROOT, workdir, sanity_folder, features_folder]
		proc = subprocess.run(cmd, capture_output=True, text=True, env=bonsai_env(), timeout=240)
		with open(os.path.join(workdir, 'area_a_mpi.log'), 'w') as handle:
			handle.write('# %s\n# exit %d\n%s%s' % (' '.join(cmd), proc.returncode, proc.stdout, proc.stderr))
		return dict(returncode=proc.returncode, log=proc.stdout + proc.stderr, out=workdir, ranks=ranks)

	job.__name__ = 'area_a_mpi_job_%d' % ranks
	return job


MPI_JOBS = {ranks: mpi_job(ranks) for ranks in (2, 4)}
