"""
pytest fixtures and hooks shared by the whole suite; tests/conftest.py imports
everything here.

End-to-end runs are declared as session fixtures with bonsai_run_fixture() (or
bonsai_job_fixture() for a custom sequence of commands). When the first test
starts, every such fixture that a selected test asks for is submitted to a pool
of worker threads, each driving one Bonsai subprocess, so the runs overlap
instead of queueing behind each other. The pool's size follows the free cores
(at most half of them, at most 8; override with BONSAI_TEST_WORKERS) and an
MPI run takes one slot per rank.
"""
import hashlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pytest

from bonsai_testdata import DatasetSpec, make_dataset, input_is_sanity_output
from bonsai_testrun import (REPO_ROOT, EXAMPLE_DATA, write_config, run_bonsai, run_bonsai_steps, mpiexec_path,
                            python_has_mpi4py)

__all__ = ['EXAMPLE', 'DATASET_NAME', 'RunSpec', 'run_spec', 'BonsaiPool', 'bonsai_run_fixture',
           'bonsai_job_fixture', 'bonsai_pool', 'bonsai_globals_reset', 'requires_mpi', 'pool_slots',
           'pytest_collection_finish', '_bonsai_prefetch', '_bonsai_data_dir_cleanup']

EXAMPLE = 'example'  # the bundled 64-cell example, pre-2.0 Sanity layout, read in place
DATASET_NAME = 'bonsai_test'  # every config uses this dataset name (see tests/README.md, "Side effects")
PACKAGES = ('bonsai', 'backbone_based_bonsai', 'bonsai_scout', 'downstream_analyses', 'optional_preprocessing')


@dataclass(frozen=True)
class RunSpec:
	"""One end-to-end Bonsai run: input, YAML options, which --step invocations, how many MPI ranks."""
	dataset: object = DatasetSpec()
	options: tuple = ()
	steps: tuple = ('all',)
	mpi_ranks: object = None
	timeout: float = 240
	python: object = None
	label: str = ''

	def option(self, key, default=None):
		return dict(self.options).get(key, default)


def run_spec(dataset=DatasetSpec(), steps=('all',), mpi_ranks=None, timeout=240, python=None, label='', **options):
	"""Build a RunSpec; keyword arguments are YAML options passed to create_config_file.py."""
	return RunSpec(dataset=dataset, options=tuple(sorted(options.items())), steps=tuple(steps),
	               mpi_ranks=mpi_ranks, timeout=timeout, python=python, label=label)


def pool_slots():
	if os.environ.get('BONSAI_TEST_WORKERS'):
		return max(1, int(os.environ['BONSAI_TEST_WORKERS']))
	try:
		free = os.cpu_count() - os.getloadavg()[0]
	except (AttributeError, OSError):
		free = os.cpu_count() or 2
	return max(1, min(8, int(free // 2)))


def _short_hash(obj):
	return hashlib.sha1(repr(obj).encode()).hexdigest()[:10]


class BonsaiPool:
	"""Runs Bonsai jobs on worker threads; results are memoized by spec or by job name."""

	def __init__(self, tmp_path_factory, slots):
		self._factory = tmp_path_factory
		self._base = None
		self.slots = slots
		self._executor = ThreadPoolExecutor(max_workers=slots, thread_name_prefix='bonsai-run')
		self._lock = threading.Lock()
		self._cond = threading.Condition()
		self._free = slots
		self._next_ticket = 0
		self._serving = 0
		self._futures = {}
		self._datasets = {}
		self._dataset_locks = {}

	@property
	def base(self):
		with self._lock:
			if self._base is None:
				self._base = str(self._factory.mktemp('bonsai_runs', numbered=True))
			return self._base

	def _acquire(self, need):
		"""
		Take `need` slots, first come first served: a job waits until every job that asked before it has its
		slots, so a job of several slots (an MPI run) is not passed over by one-slot jobs taking each slot that
		frees up.
		"""
		need = min(need, self.slots)
		with self._cond:
			ticket = self._next_ticket
			self._next_ticket += 1
			while self._serving != ticket or self._free < need:
				self._cond.wait()
			self._free -= need
			self._serving += 1
			self._cond.notify_all()
		return need

	def _release(self, need):
		with self._cond:
			self._free += need
			self._cond.notify_all()

	def dataset(self, spec):
		"""(folder, sim) for a DatasetSpec, written once per session; (EXAMPLE_DATA, None) for EXAMPLE."""
		if spec == EXAMPLE:
			return EXAMPLE_DATA, None
		with self._lock:
			lock = self._dataset_locks.setdefault(spec, threading.Lock())
		with lock:
			if spec not in self._datasets:
				folder = os.path.join(self.base, 'data_' + _short_hash(spec))
				sim = make_dataset(spec, folder)
				self._datasets[spec] = (folder, sim)
			return self._datasets[spec]

	def submit(self, spec):
		with self._lock:
			if spec not in self._futures:
				self._futures[spec] = self._executor.submit(self._run_spec, spec)
			return self._futures[spec]

	def result(self, spec):
		return self.submit(spec).result()

	def submit_job(self, name, fn, slots=1):
		key = ('job', name)
		with self._lock:
			if key not in self._futures:
				self._futures[key] = self._executor.submit(self._run_job, name, fn, slots)
			return self._futures[key]

	def job_result(self, name, fn, slots=1):
		return self.submit_job(name, fn, slots).result()

	def _run_job(self, name, fn, slots):
		workdir = os.path.join(self.base, 'job_' + name)
		os.makedirs(workdir, exist_ok=True)
		need = self._acquire(slots)
		try:
			return fn(self, workdir)
		finally:
			self._release(need)

	def _run_spec(self, spec):
		data_folder, sim = self.dataset(spec.dataset)
		workdir = os.path.join(self.base, (spec.label + '_' if spec.label else '') + 'run_' + _short_hash(spec))
		os.makedirs(workdir, exist_ok=True)
		options = dict(spec.options)
		if 'input_is_sanity_output' not in options:
			options['input_is_sanity_output'] = (spec.dataset == EXAMPLE) or input_is_sanity_output(spec.dataset)
		config = write_config(os.path.join(workdir, 'config.yaml'), options.pop('dataset', DATASET_NAME),
		                      data_folder, os.path.join(workdir, 'results'), python=spec.python, **options)
		need = self._acquire(spec.mpi_ranks or 1)
		try:
			if spec.steps == ('all',):
				run = run_bonsai(config, mpi_ranks=spec.mpi_ranks, timeout=spec.timeout, python=spec.python)
			else:
				run = run_bonsai_steps(config, steps=spec.steps, mpi_ranks=spec.mpi_ranks, timeout=spec.timeout,
				                       python=spec.python)
		finally:
			self._release(need)
		run.sim, run.data_folder, run.spec = sim, data_folder, spec
		return run

	def shutdown(self):
		self._executor.shutdown(wait=True, cancel_futures=True)


_REGISTRY = {}


def _register(name, entry):
	if name in _REGISTRY and _REGISTRY[name] != entry:
		raise ValueError('two different Bonsai run fixtures are both named %r' % name)
	_REGISTRY[name] = entry


def bonsai_run_fixture(name, spec):
	"""
	Declare a session fixture `name` whose value is the BonsaiRun of `spec`.
	Assign it to a module-level variable of the same name in the test module:
		final_run = bonsai_run_fixture('final_run', run_spec(DatasetSpec(...), nnn_n_randommoves=20))
	Fixture names are global across the suite; reuse a name only for the same spec.
	"""
	_register(name, ('run', spec))

	@pytest.fixture(scope='session', name=name)
	def _fixture(bonsai_pool):
		if spec.mpi_ranks:
			requires_mpi(spec.python)
		return bonsai_pool.result(spec)

	return _fixture


def bonsai_job_fixture(name, fn, slots=1, mpi=False):
	"""
	Declare a session fixture `name` whose value is fn(pool, workdir), run on the
	pool. fn may call pool.dataset(spec), write_config and run_bonsai directly; it
	must not wait on pool.result() of another spec (that can deadlock the pool).
	slots: how many cores the job keeps busy at once (the MPI rank count).
	"""
	_register(name, ('job', fn, slots, mpi))

	@pytest.fixture(scope='session', name=name)
	def _fixture(bonsai_pool):
		if mpi:
			requires_mpi()
		return bonsai_pool.job_result(name, fn, slots)

	return _fixture


def requires_mpi(python=None):
	"""Skip the calling test unless mpiexec and mpi4py (in the Python that runs Bonsai) are both present."""
	if sys.platform.startswith('win32'):
		pytest.skip('Bonsai does not use mpi4py on Windows')
	if mpiexec_path() is None:
		pytest.skip('mpiexec not found')
	if not python_has_mpi4py(python):
		pytest.skip('mpi4py is not installed')


@pytest.fixture(scope='session')
def bonsai_pool(tmp_path_factory):
	# tmp_path_factory.getbasetemp() is not thread-safe: the first call creates pytest's base temporary folder.
	# Make it, and the pool's own folder, here in the main thread before any worker starts, so a worker's mktemp
	# and a test's tmp_path never race to create it.
	tmp_path_factory.getbasetemp()
	pool = BonsaiPool(tmp_path_factory, pool_slots())
	pool.base
	yield pool
	pool.shutdown()


@pytest.fixture(scope='session', autouse=True)
def _bonsai_prefetch(request, bonsai_pool, _bonsai_data_dir_cleanup):
	"""
	Start every registered run that a selected test needs, so they run concurrently: the run fixtures a test
	names as arguments (or through usefixtures), and those whose names are parameter values of the test, for a
	test that takes its run with request.getfixturevalue(name) from @pytest.mark.parametrize('name', [...]).
	"""
	mpi_ok = None
	for item in request.session.items:
		params = getattr(getattr(item, 'callspec', None), 'params', {})
		names = list(getattr(item, 'fixturenames', ())) + [value for value in params.values() if isinstance(value, str)]
		for name in names:
			entry = _REGISTRY.get(name)
			if entry is None:
				continue
			needs_mpi = (entry[0] == 'run' and entry[1].mpi_ranks) or (entry[0] == 'job' and entry[3])
			if needs_mpi:
				if mpi_ok is None:
					mpi_ok = (not sys.platform.startswith('win32')) and mpiexec_path() is not None and python_has_mpi4py()
				if not mpi_ok:
					continue
			if entry[0] == 'run':
				bonsai_pool.submit(entry[1])
			else:
				bonsai_pool.submit_job(name, entry[1], entry[2])
	yield


@pytest.fixture(scope='session', autouse=True)
def _bonsai_data_dir_cleanup():
	"""
	bonsai_main.py creates an empty '<repository>/../data/<dataset>' folder on
	every run (tests/README.md, "Known bugs"). Remove it at the end if this
	session created it and it is still empty.
	"""
	data_root = os.path.join(os.path.dirname(REPO_ROOT), 'data')
	target = os.path.join(data_root, DATASET_NAME)
	existed = (os.path.exists(data_root), os.path.exists(target))
	yield
	for path, was_there in ((target, existed[1]), (data_root, existed[0])):
		if not was_there:
			try:
				os.rmdir(path)
			except OSError:
				pass


@pytest.fixture(autouse=True)
def bonsai_globals_reset():
	"""
	Snapshot and restore the process-wide state Bonsai's in-process code changes:
	bonsai_globals, mpi_wrapper.comm, numpy's global random state and the
	recursion limit. Autouse, so every test starts from the same state.
	"""
	import bonsai.bonsai_globals as bs_glob
	import bonsai.mpi_wrapper as mpi_wrapper
	saved_globals = {key: value for key, value in vars(bs_glob).items() if not key.startswith('__')}
	saved_comm = mpi_wrapper.comm
	saved_random = np.random.get_state()
	saved_limit = sys.getrecursionlimit()
	yield
	for key in [key for key in vars(bs_glob) if not key.startswith('__')]:
		if key not in saved_globals:
			delattr(bs_glob, key)
	for key, value in saved_globals.items():
		setattr(bs_glob, key, value)
	mpi_wrapper.comm = saved_comm
	np.random.set_state(saved_random)
	sys.setrecursionlimit(saved_limit)


def _inside(path, root):
	path, root = os.path.realpath(path), os.path.realpath(root)
	return os.path.commonpath([path, root]) == root


def pytest_collection_finish(session):
	"""Every Bonsai module the tests imported must come from the checkout under test."""
	stray = []
	for name, module in list(sys.modules.items()):
		if name.split('.')[0] not in PACKAGES:
			continue
		# Namespace packages (bonsai/ has no __init__.py) have no __file__; their modules do.
		path = getattr(module, '__file__', None)
		if path and not _inside(path, REPO_ROOT):
			stray.append('%s: %s' % (name, path))
	if stray:
		raise pytest.UsageError('Bonsai modules imported from outside the checkout %s:\n%s' % (
			REPO_ROOT, '\n'.join(stray)))
