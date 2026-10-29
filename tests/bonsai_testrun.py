"""
Running Bonsai's command-line scripts from the checkout under test, and reading
what they write.

Every script is run as a subprocess: bonsai_main.py and create_config_file.py
change the working directory and do their work at import time, and Bonsai keeps
module-level state in bonsai/bonsai_globals.py.
"""
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from bonsai_testtrees import Node, parse_newick

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
BONSAI_MAIN = os.path.join(REPO_ROOT, 'bonsai', 'bonsai_main.py')
CREATE_CONFIG = os.path.join(REPO_ROOT, 'bonsai', 'create_config_file.py')
EXAMPLE_DATA = os.path.join(REPO_ROOT, 'examples', 'example_data', 'simulated_binary_6_gens_samplingNoise')

# One thread per process: numpy's BLAS would otherwise start a thread per core.
THREAD_PINS = {
	'OMP_NUM_THREADS': '1',
	'OPENBLAS_NUM_THREADS': '1',
	'MKL_NUM_THREADS': '1',
	'NUMEXPR_NUM_THREADS': '1',
	'VECLIB_MAXIMUM_THREADS': '1',
	'NUMBA_NUM_THREADS': '1',
}
# Open MPI's hwloc probes GPUs through CUDA, NVML and X11 when MPI starts, which
# takes 15-25 s per process on a GPU server. Bonsai never uses a GPU.
HWLOC_NO_GPU = '-gl,-cuda,-nvml,-opencl,-rsmi,-levelzero'

# The YAML keys create_config_file.py takes as command-line options (all keys of
# bonsai/config_template_do_not_change/config_template.yaml except start_time).
CONFIG_OPTIONS = ('dataset', 'data_folder', 'filenames_data', 'results_folder', 'verbose', 'input_is_sanity_output',
                  'zscore_cutoff', 'rescale_by_var', 'nnn_n_randommoves', 'nnn_n_randomtrees', 'use_knn',
                  'UB_ellipsoid_size', 'spr_strategy', 'pickup_intermediate', 'tmp_folder')

# Folder names Bonsai writes after each step (README "The Bonsai results"; what
# --pickup_intermediate looks for). Pinned outputs, so they are written out here
# instead of being taken from bonsai_helpers.getOutputFolder.
STAGE_SUFFIXES = (
	('preprocessed', None),
	('greedy', ''),
	('redo_starry', '_redoStarry'),
	('opt_times', '_redoStarry_optTimes'),
	('spr_moves', '_redoStarry_optTimes_sprMoves'),
	('nnn_reorder', '_redoStarry_optTimes_sprMoves_nnnReorder'),
	('reordered_edges', '_redoStarry_optTimes_sprMoves_nnnReorder_reorderedEdges'),
	('final', None),
)


def zscore_tag(zscore_cutoff):
	return '_zscore' + str(float(zscore_cutoff)) if zscore_cutoff > 0 else ''


def stage_folder(stage, zscore_cutoff=1.0):
	"""The results subfolder Bonsai writes after a stage, e.g. stage_folder('final') == 'final_bonsai_zscore1.0'."""
	tag = zscore_tag(zscore_cutoff)
	if stage == 'preprocessed':
		return 'preprocessed' + tag
	if stage == 'final':
		return 'final_bonsai' + tag
	suffix = dict(STAGE_SUFFIXES)[stage]
	return 'intermediate_bonsai' + tag + suffix


def processed_data_folder(zscore_cutoff=1.0):
	return 'zscorefiltered_%.3f_and_processed' % zscore_cutoff


def bonsai_env(extra=None, pythonpath=REPO_ROOT):
	"""Environment for a Bonsai subprocess: thread pins, headless matplotlib, imports from this checkout."""
	env = dict(os.environ)
	env.update(THREAD_PINS)
	env['PYTHONPATH'] = pythonpath
	env['MPLBACKEND'] = 'Agg'
	env.setdefault('HWLOC_COMPONENTS', HWLOC_NO_GPU)
	if extra:
		env.update(extra)
	return env


def mpiexec_path():
	return shutil.which('mpiexec')


def python_has_mpi4py(python=None):
	python = python or sys.executable
	proc = subprocess.run([python, '-c', 'import mpi4py'], capture_output=True, env=bonsai_env())
	return proc.returncode == 0


def _yaml_value(value):
	if isinstance(value, bool):
		return 'True' if value else 'False'
	return str(value)


def write_config(yaml_path, dataset, data_folder, results_folder, python=None, **options):
	"""
	Write a Bonsai YAML config through bonsai/create_config_file.py, so the file
	is what a user gets. Paths should be absolute: the script resolves relative
	ones against the repository root. Returns yaml_path.
	"""
	values = dict(options)
	values.update(dataset=dataset, data_folder=data_folder, results_folder=results_folder)
	cmd = [python or sys.executable, CREATE_CONFIG, '--new_yaml_path', str(yaml_path)]
	for key, value in values.items():
		if key not in CONFIG_OPTIONS:
			raise KeyError('create_config_file.py has no option %r' % key)
		cmd += ['--' + key, _yaml_value(value)]
	proc = subprocess.run(cmd, capture_output=True, text=True, env=bonsai_env(), timeout=120)
	if proc.returncode != 0 or not os.path.exists(yaml_path):
		raise RuntimeError('create_config_file.py failed (%d):\n%s\n%s' % (proc.returncode, proc.stdout, proc.stderr))
	return str(yaml_path)


def read_config(yaml_path):
	from ruamel.yaml import YAML
	with open(yaml_path) as handle:
		return dict(YAML(typ='safe').load(handle))


@dataclass
class BonsaiRun:
	"""The outcome of one bonsai_main.py invocation (or of several, for a stepwise run)."""
	cmd: list
	returncode: int
	stdout: str
	stderr: str
	results_folder: str
	config: str
	seconds: float
	timed_out: bool = False
	history: list = field(default_factory=list)
	# Set by the run pool (bonsai_testfixtures.py): the simulated truth and where the input was written.
	sim: object = None
	data_folder: str = None
	spec: object = None

	@property
	def log(self):
		return self.stdout + self.stderr

	def check(self):
		"""Raise with the end of the log unless every invocation exited 0."""
		for run in self.history or [self]:
			if run.returncode != 0 or run.timed_out:
				raise AssertionError('bonsai_main.py failed (exit %s%s): %s\n--- log tail ---\n%s' % (
					run.returncode, ', timed out' if run.timed_out else '', ' '.join(map(str, run.cmd)),
					run.log[-4000:]))
		return self

	def path(self, *parts):
		return os.path.join(self.results_folder, *parts)

	def final_folder(self, zscore_cutoff=1.0):
		return self.path(stage_folder('final', zscore_cutoff))


def bonsai_command(config, step='all', mpi_ranks=None, pickup_intermediate=None, extra_args=(), python=None):
	python = python or sys.executable
	args = [BONSAI_MAIN, '--config_filepath', str(config), '--step', step]
	if pickup_intermediate is not None:
		args += ['--pickup_intermediate', _yaml_value(pickup_intermediate)]
	args += list(extra_args)
	if mpi_ranks is None:
		return [python] + args
	mpiexec = mpiexec_path()
	if mpiexec is None:
		raise RuntimeError('mpiexec not found')
	# --bind-to none: Open MPI otherwise pins every job's ranks to the same first cores.
	return [mpiexec, '--bind-to', 'none', '-n', str(mpi_ranks), python, '-m', 'mpi4py'] + args


def run_bonsai(config, step='all', mpi_ranks=None, timeout=240, pickup_intermediate=None, extra_args=(),
               python=None, env=None, log_path=None):
	"""
	Run bonsai/bonsai_main.py of this checkout once and return a BonsaiRun.
	mpi_ranks=None runs one plain process; an integer runs it under
	'mpiexec -n <mpi_ranks> python -m mpi4py' as the README describes.
	A run that exceeds timeout is killed with its whole process group.
	The command, exit code and output are also written to log_path
	(default: bonsai_<step>.log next to the config file).
	"""
	cmd = bonsai_command(config, step=step, mpi_ranks=mpi_ranks, pickup_intermediate=pickup_intermediate,
	                     extra_args=extra_args, python=python)
	start = time.time()
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
	                        env=bonsai_env(env), cwd=TESTS_DIR, start_new_session=True)
	timed_out = False
	try:
		stdout, stderr = proc.communicate(timeout=timeout)
	except subprocess.TimeoutExpired:
		timed_out = True
		if hasattr(os, 'killpg'):
			os.killpg(proc.pid, signal.SIGKILL)
		else:
			proc.kill()
		stdout, stderr = proc.communicate()
	results_folder = read_config(config).get('results_folder')
	log_path = log_path or os.path.join(os.path.dirname(os.path.abspath(str(config))), 'bonsai_%s.log' % step)
	with open(log_path, 'w') as handle:
		handle.write('# %s\n# exit %s%s, %.1f s\n' % (' '.join(map(str, cmd)), proc.returncode,
		                                              ' (timed out)' if timed_out else '', time.time() - start))
		handle.write(stdout + stderr)
	return BonsaiRun(cmd=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr,
	                 results_folder=os.path.abspath(results_folder) if results_folder else None,
	                 config=str(config), seconds=time.time() - start, timed_out=timed_out)


def run_bonsai_steps(config, steps=('preprocess', 'core_calc', 'metadata'), **kwargs):
	"""Run several --step invocations in order; stops at the first failure. Returns a BonsaiRun with history."""
	history = []
	for step in steps:
		run = run_bonsai(config, step=step, **kwargs)
		history.append(run)
		if run.returncode != 0 or run.timed_out:
			break
	last = history[-1]
	return BonsaiRun(cmd=[run.cmd for run in history], returncode=last.returncode,
	                 stdout=''.join(run.stdout for run in history), stderr=''.join(run.stderr for run in history),
	                 results_folder=last.results_folder, config=last.config,
	                 seconds=sum(run.seconds for run in history), timed_out=last.timed_out, history=history)


# ------------------------------------------------------------------- readers

EDGE_LINE = re.compile(r'^(\d+)\t(\d+)\t(-?\d\.\d{8}e[+-]\d{2,3})$')


def read_edge_info(folder):
	"""edgeInfo.txt: one edge per line, 'parentVert<TAB>childVert<TAB>length' with the length as %.8e."""
	edges = []
	with open(os.path.join(folder, 'edgeInfo.txt')) as handle:
		for line in handle.read().splitlines():
			match = EDGE_LINE.match(line)
			if match is None:
				raise ValueError('edgeInfo.txt line does not match the format: %r' % line)
			edges.append((int(match.group(1)), int(match.group(2)), float(match.group(3))))
	return edges


def read_vert_info(folder):
	"""vertInfo.txt: header 'vertInd<TAB>nodeInd<TAB>vertName', then one row per vertex."""
	with open(os.path.join(folder, 'vertInfo.txt')) as handle:
		lines = handle.read().splitlines()
	if lines[0] != 'vertInd\tnodeInd\tvertName':
		raise ValueError('unexpected vertInfo.txt header %r' % lines[0])
	rows = []
	for line in lines[1:]:
		parts = line.split('\t')
		if len(parts) != 3:
			raise ValueError('vertInfo.txt line does not have three columns: %r' % line)
		rows.append((int(parts[0]), int(parts[1]), parts[2]))
	return rows


def tree_from_folder(folder):
	"""Build a Node tree from edgeInfo.txt and vertInfo.txt; the root is the first edge's parent."""
	edges = read_edge_info(folder)
	names = {vert: name for vert, _, name in read_vert_info(folder)}
	nodes = {}
	root_vert = edges[0][0]
	nodes[root_vert] = Node(names[root_vert], None)
	for parent, child, length in edges:
		if parent not in nodes:
			raise ValueError('edge %d->%d appears before its parent is reached' % (parent, child))
		nodes[child] = Node(names[child], length)
		nodes[parent].children.append(nodes[child])
	return nodes[root_vert]


def read_newick_file(folder, filename='tree.nwk'):
	with open(os.path.join(folder, filename)) as handle:
		return parse_newick(handle.read())


def read_metadata_json(folder):
	with open(os.path.join(folder, 'metadata.json')) as handle:
		return json.load(handle)


def read_posteriors(folder):
	"""(means, variances), each vertices x genes, rows ordered by vertInd."""
	means = np.load(os.path.join(folder, 'posterior_ltqs_vertByGene.npy'))
	variances = np.load(os.path.join(folder, 'posterior_ltqsVars_vertByGene.npy'))
	return means, variances


def read_metadata_bonsai(folder):
	"""metadata_bonsai.txt (a one-row CSV written by pandas) as a dictionary of strings."""
	with open(os.path.join(folder, 'metadata_bonsai.txt')) as handle:
		rows = list(csv.reader(handle))
	header, values = rows[0], rows[1]
	return dict(zip(header[1:], values[1:]))


def posteriors_by_name(folder):
	"""Posterior means and variances keyed by vertName."""
	means, variances = read_posteriors(folder)
	names = {vert: name for vert, _, name in read_vert_info(folder)}
	return ({names[vert]: means[vert] for vert in names}, {names[vert]: variances[vert] for vert in names})
