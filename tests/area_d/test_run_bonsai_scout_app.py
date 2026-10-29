"""
bonsai_scout/run_bonsai_scout_app.py (area D): the launcher that points the Shiny app at a results folder
(README, Bonsai-scout) and starts `shiny run bonsai_scout/app.py`. A stand-in `shiny` executable on PATH records
the command, working directory and environment, so no server starts. POSIX shell stand-in: skipped on Windows.
"""
import os
import re
import stat
import subprocess
import sys

import pytest

from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env

SCRIPT = os.path.join(REPO_ROOT, 'bonsai_scout', 'run_bonsai_scout_app.py')
STUB = """#!/bin/sh
{ pwd; printf 'DATA=%s\\n' "$BONSAI_DATA_PATH"; printf 'SETTINGS=%s\\n' "$BONSAI_SETTINGS_PATH"; printf '%s\\n' "$@"; } \\
  > "$SHINY_STUB_LOG"
"""


def _launch(workdir, args, make_files):
	stub_dir = os.path.join(workdir, 'bin')
	os.makedirs(stub_dir, exist_ok=True)
	stub = os.path.join(stub_dir, 'shiny')
	with open(stub, 'w') as handle:
		handle.write(STUB)
	os.chmod(stub, os.stat(stub).st_mode | stat.S_IXUSR)
	results = os.path.join(workdir, 'results')
	os.makedirs(results, exist_ok=True)
	if make_files:
		for name in ('bonsai_vis_data.hdf', 'my_settings.json'):
			open(os.path.join(results, name), 'w').close()
	log = os.path.join(workdir, 'shiny_stub.log')
	env = bonsai_env({'PATH': stub_dir + os.pathsep + os.environ.get('PATH', ''), 'SHINY_STUB_LOG': log})
	proc = subprocess.run([sys.executable, SCRIPT, '--results_folder', results] + args, capture_output=True,
	                      text=True, env=env, cwd=workdir, timeout=120)
	with open(log) as handle:
		lines = handle.read().splitlines()
	return proc, results, lines


d_scout_launch_port = bonsai_job_fixture('d_scout_launch_port', lambda pool, workdir: _launch(
	workdir, ['--settings_filename', 'my_settings.json', '--port', '4321'], make_files=True))
d_scout_launch_default = bonsai_job_fixture('d_scout_launch_default', lambda pool, workdir: _launch(
	workdir, [], make_files=False))

pytestmark = pytest.mark.skipif(sys.platform.startswith('win32'), reason='the shiny stand-in is a POSIX shell script')


def test_launcher_passes_paths_and_port(d_scout_launch_port):
	"""BONSAI_DATA_PATH is <results_folder>/bonsai_vis_data.hdf, BONSAI_SETTINGS_PATH
	<results_folder>/<settings_filename> (absolute), shiny runs bonsai_scout/app.py from the repository root on the
	given port on all interfaces, and the URL printed names that port."""
	proc, results, lines = d_scout_launch_port
	assert proc.returncode == 0, proc.stderr[-2000:]
	assert os.path.realpath(lines[0]) == os.path.realpath(REPO_ROOT)
	assert lines[1] == 'DATA=' + os.path.abspath(os.path.join(results, 'bonsai_vis_data.hdf'))
	assert lines[2] == 'SETTINGS=' + os.path.abspath(os.path.join(results, 'my_settings.json'))
	assert lines[3:] == ['run', 'bonsai_scout/app.py', '--port=4321', '--host=0.0.0.0']
	assert 'http://localhost:4321' in proc.stdout
	assert 'were not found' not in proc.stdout


def test_launcher_defaults(d_scout_launch_default):
	"""Without --port a random port in 1025-9999 is used (and printed); the settings file defaults to
	bonsai_vis_settings.json; missing preprocessed files are reported before starting."""
	proc, results, lines = d_scout_launch_default
	assert proc.returncode == 0, proc.stderr[-2000:]
	assert lines[2] == 'SETTINGS=' + os.path.abspath(os.path.join(results, 'bonsai_vis_settings.json'))
	port = int(re.fullmatch(r'--port=(\d+)', lines[5]).group(1))
	assert 1025 <= port <= 9999
	assert 'http://localhost:%d' % port in proc.stdout
	assert "Preprocessed results for Bonsai vis were not found" in proc.stdout
	assert 'Settings file for Bonsai vis were not found' in proc.stdout
