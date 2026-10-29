"""
bonsai/create_config_file.py (area A): the YAML file it writes from the template, every option's key, value
and type, the comments the template carries, and that Run_Configs reads every key back. The script works at
import and changes directory, so it only runs as a subprocess; paths given to it are absolute.
"""
import os
import subprocess
import sys

import pytest
from ruamel.yaml import YAML

from bonsai_testrun import CREATE_CONFIG, REPO_ROOT, bonsai_env, read_config

TEMPLATE = os.path.join(REPO_ROOT, 'bonsai', 'config_template_do_not_change', 'config_template.yaml')

# The template's values (README: create_config_file.py copies the template and overrides the keys given).
DEFAULTS = {'start_time': None, 'dataset': 'new_dataset', 'data_folder': 'data/new_dataset',
            'filenames_data': 'features.txt,standard_deviations.txt', 'results_folder': 'results/new_dataset',
            'verbose': True, 'input_is_sanity_output': True, 'zscore_cutoff': 1.0, 'rescale_by_var': True,
            'nnn_n_randommoves': 1000, 'nnn_n_randomtrees': 10, 'use_knn': 10, 'UB_ellipsoid_size': 1.0,
            'spr_strategy': 'large_tree', 'pickup_intermediate': False, 'tmp_folder': ''}

# Every option at a value other than its default; booleans in the spellings str2bool accepts.
OPTIONS = {'dataset': ('my_set', 'my_set'), 'data_folder': ('/abs/in data', '/abs/in data'),
           'results_folder': ('/abs/out', '/abs/out'), 'filenames_data': ('a.txt,b.txt', 'a.txt,b.txt'),
           'input_is_sanity_output': ('F', False), 'verbose': ('no', False), 'zscore_cutoff': ('-0.5', -0.5),
           'rescale_by_var': ('0', False), 'use_knn': ('-1', -1), 'nnn_n_randommoves': ('25', 25),
           'nnn_n_randomtrees': ('3', 3), 'UB_ellipsoid_size': ('2.5', 2.5), 'spr_strategy': ('super_sure', 'super_sure'),
           'pickup_intermediate': ('Yes', True), 'tmp_folder': ('/abs/tree', '/abs/tree')}


@pytest.fixture(scope='module')
def configs(tmp_path_factory):
	"""Run the script four ways at once: defaults, every option set, deprecated options, a bad boolean."""
	base = tmp_path_factory.mktemp('create_config')
	jobs = {
		'defaults': [str(base / 'defaults.yaml')],
		'options': [str(base / 'nested' / 'deeper' / 'options.yaml')],
		'deprecated': [str(base / 'deprecated.yaml'), '--skip_greedy_merging', 'True', '--skip_opt_times', 'True',
		               '--skip_redo_starry', 'false', '--skip_nnn_reordering', 'y'],
		'bad_bool': [str(base / 'bad.yaml'), '--verbose', 'maybe'],
	}
	for key, (text, _) in OPTIONS.items():
		jobs['options'] += ['--' + key, text]
	procs = {}
	for name, (path, *rest) in jobs.items():
		procs[name] = (path, subprocess.Popen([sys.executable, CREATE_CONFIG, '--new_yaml_path', path] + rest,
		                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
		                                      env=bonsai_env()))
	out = {}
	for name, (path, proc) in procs.items():
		stdout, stderr = proc.communicate(timeout=120)
		out[name] = dict(path=path, returncode=proc.returncode, stdout=stdout, stderr=stderr)
	return out


def test_defaults_are_the_template(configs):
	run = configs['defaults']
	assert run['returncode'] == 0, run['stderr']
	assert read_config(run['path']) == DEFAULTS
	assert list(read_config(run['path'])) == list(DEFAULTS)


def test_the_written_file_keeps_the_template_comments(configs):
	"""README: 'In the YAML-file, one can find extensive explanations of the different run configurations.'"""
	with open(TEMPLATE) as handle:
		template_comments = [line for line in handle.read().splitlines() if line.lstrip().startswith('#')]
	with open(configs['options']['path']) as handle:
		written = handle.read().splitlines()
	assert [line for line in written if line.lstrip().startswith('#')] == template_comments


def test_every_option_is_written_with_its_type(configs):
	run = configs['options']
	assert run['returncode'] == 0, run['stderr']
	written = read_config(run['path'])
	assert set(written) == set(DEFAULTS)
	assert written['start_time'] is None
	for key, (_, value) in OPTIONS.items():
		assert written[key] == value and type(written[key]) is type(value), key


def test_run_configs_reads_back_every_key_the_script_writes(configs):
	from bonsai.bonsai_helpers import Run_Configs
	config = Run_Configs(configs['options']['path'])
	for key, (_, value) in OPTIONS.items():
		assert getattr(config, key) == value, key
	defaults = Run_Configs(configs['defaults']['path'])
	for key, value in DEFAULTS.items():
		if key != 'start_time':
			assert getattr(defaults, key) == value, key


def test_deprecated_options_are_accepted_and_not_written(configs):
	"""The skip_* options are deprecated: accepted, warned about, and without effect on the file."""
	run = configs['deprecated']
	assert run['returncode'] == 0, run['stderr']
	assert 'WARNING' in run['stdout']
	assert read_config(run['path']) == DEFAULTS


def test_a_value_that_is_not_a_boolean_is_refused(configs):
	run = configs['bad_bool']
	assert run['returncode'] == 2
	assert 'Boolean value expected' in run['stderr']
	assert not os.path.exists(run['path'])
