"""
bonsai/mpi_wrapper.py (area A): without mpi4py every collective degrades to the one-process answer; under
'mpiexec -n 2' and '-n 4' (marked mpi) the wrappers move the data as MPI would. The read_and_filter results of
the same MPI runs are checked in test_bonsai_dataprocessing.py.
"""
import json
import os
import sys

import numpy as np
import pytest

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai_testfixtures import bonsai_job_fixture

from area_a.helpers_a import MPI_JOBS


@pytest.fixture
def without_mpi4py(monkeypatch):
	"""Make 'from mpi4py import MPI' raise ImportError, as on a machine without mpi4py."""
	monkeypatch.setitem(sys.modules, 'mpi4py', None)
	monkeypatch.setattr(mpi_wrapper, 'comm', None)
	monkeypatch.setattr(sys, 'platform', 'linux')


def test_without_mpi4py_bonsai_runs_as_one_process(without_mpi4py, capsys):
	mpi_wrapper.mpi_init()
	assert mpi_wrapper.comm is None
	assert 'mpi4py-package is not installed' in capsys.readouterr().out
	assert mpi_wrapper.get_mpi_info() == (0, 1)
	assert mpi_wrapper.get_process_rank() == 0 and mpi_wrapper.get_process_size() == 1
	assert mpi_wrapper.is_first_process()
	data = {'a': np.arange(3)}
	assert mpi_wrapper.bcast(data) is data
	assert mpi_wrapper.world_allgather(data) == [data]
	assert mpi_wrapper.gather(data) == [data]
	assert mpi_wrapper.barrier() is None


def test_get_mpi_info_single_process_ignores_the_communicator():
	assert mpi_wrapper.get_mpi_info(singleProcess=True) == (0, 1)


def test_on_windows_mpi_is_skipped(monkeypatch, capsys):
	"""README: mpi4py does not work properly on Windows; Bonsai then runs on one core."""
	monkeypatch.setattr(mpi_wrapper, 'comm', None)
	monkeypatch.setattr(sys, 'platform', 'win32')
	mpi_wrapper.mpi_init()
	assert mpi_wrapper.comm is None and mpi_wrapper.get_mpi_info() == (0, 1)
	assert 'Windows' in capsys.readouterr().out


def test_buffer_collectives_without_a_communicator(without_mpi4py):
	"""Bcast of a typed buffer leaves it (the root's data) in place; Gather returns the one rank's data."""
	data = np.array([1.5, 2.5])
	assert mpi_wrapper.Bcast(data, type='double') is None and data.tolist() == [1.5, 2.5]
	ints = np.array([3, 4], dtype='i')
	assert mpi_wrapper.Bcast(ints, type='int') is None and ints.tolist() == [3, 4]
	assert mpi_wrapper.Bcast(data, type='raw') is data
	assert mpi_wrapper.Gather(data) == [data]


area_a_mpi2 = bonsai_job_fixture('area_a_mpi2', MPI_JOBS[2], slots=2, mpi=True)
area_a_mpi4 = bonsai_job_fixture('area_a_mpi4', MPI_JOBS[4], slots=4, mpi=True)


def _mpi_result(jobs, ranks):
	job = jobs[ranks]
	assert job['returncode'] == 0, job['log'][-4000:]
	with open(os.path.join(job['out'], 'result.json')) as handle:
		checks = json.load(handle)
	return checks, np.load(os.path.join(job['out'], 'result.npz'))


@pytest.mark.mpi
@pytest.mark.parametrize('ranks', [2, 4])
def test_collectives_under_mpiexec(area_a_mpi2, area_a_mpi4, ranks):
	checks, arrays = _mpi_result({2: area_a_mpi2, 4: area_a_mpi4}, ranks)
	assert checks['rank_size'] == [0, ranks] and checks['first'] is True
	assert checks['allgather'] == [10 * rank for rank in range(ranks)]
	assert checks['gather'] == [{'r': rank} for rank in range(ranks)]
	assert checks['bcast'] == 'from 0'
	for rank, seen in enumerate(checks['per_rank']):
		assert seen['doubles'] == [0.0, 1.5, 3.0, 4.5], rank
		assert seen['ints'] == [7, 8, 9], rank
		assert seen['other'] == [2.5, -1.0], rank
	# GatherNpUnknownSize: rank r sends r + 1 rows; the root stacks them in rank order.
	expected = np.vstack([np.full((rank + 1, 3), float(rank)) + np.arange(3) for rank in range(ranks)])
	np.testing.assert_array_equal(arrays['gathered'], expected)


@pytest.mark.mpi
@pytest.mark.parametrize('ranks', [2, 4])
def test_globals_and_deep_structures_reach_every_rank(area_a_mpi2, area_a_mpi4, ranks):
	"""communicateGlobalVars gives every rank the root's bonsai_globals; broadcastRecursiveStruct delivers a
	structure nested deeper than the default recursion limit."""
	checks, _ = _mpi_result({2: area_a_mpi2, 4: area_a_mpi4}, ranks)
	for seen in checks['per_rank']:
		assert seen['globals'] == [11, 22, 33, 44, True]
		assert seen['depth'] == 4000
