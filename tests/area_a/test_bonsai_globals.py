"""
bonsai/bonsai_globals.py (area A): the module-level state Bonsai's tree code reads, its initial values, and the
set of names communicateGlobalVars broadcasts. Checked in a fresh interpreter, since every in-process test
changes and restores these values (the bonsai_globals_reset fixture).
"""
import json
import subprocess
import sys

from bonsai_testrun import bonsai_env

import bonsai.bonsai_globals as bs_glob


def test_initial_values_in_a_fresh_interpreter():
	code = ('import json, bonsai.bonsai_globals as g; print(json.dumps({"vars": sorted(g.globalVars), '
	        '"values": [g.nCells, g.nGenes, g.nNodes, g.max_node_ind, g.mem_friendly]}))')
	proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=bonsai_env(), timeout=60)
	assert proc.returncode == 0, proc.stderr
	state = json.loads(proc.stdout)
	assert state['vars'] == ['max_node_ind', 'mem_friendly', 'nCells', 'nGenes', 'nNodes']
	assert state['values'] == [None, None, None, 0, False]


def test_every_broadcast_name_exists_in_the_module():
	for name in bs_glob.globalVars:
		assert hasattr(bs_glob, name), name
