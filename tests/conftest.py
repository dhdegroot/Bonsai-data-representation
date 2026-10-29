import os
import sys

# The bonsai/ and downstream_analyses/ packages are plain namespace packages
# (no __init__.py) importable only when the repo root is on sys.path. Make
# that true regardless of the current working directory pytest is invoked
# from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The shared harness lives next to this file (see tests/README.md). Test files
# import it as top-level modules, so put tests/ on sys.path as well.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(1, TESTS_DIR)

import pytest  # noqa: E402

pytest.register_assert_rewrite('bonsai_testtrees', 'bonsai_testdata', 'bonsai_testref', 'bonsai_testrun',
                               'bonsai_testfixtures')

# Open MPI's hwloc probes GPUs at MPI start-up, 15-25 s per process on a GPU
# server; in-process tests that import mpi4py get the same fix as subprocesses.
os.environ.setdefault('HWLOC_COMPONENTS', '-gl,-cuda,-nvml,-opencl,-rsmi,-levelzero')
os.environ.setdefault('MPLBACKEND', 'Agg')

import bonsai.bonsai_globals  # noqa: E402

# Tests import Bonsai from the checkout they live in, never from an installed copy.
assert os.path.commonpath([os.path.realpath(bonsai.bonsai_globals.__file__), os.path.realpath(REPO_ROOT)]) == \
    os.path.realpath(REPO_ROOT), 'bonsai imported from %s, not from %s' % (bonsai.bonsai_globals.__file__, REPO_ROOT)

from bonsai_testfixtures import *  # noqa: E402,F401,F403  (fixtures and hooks)
