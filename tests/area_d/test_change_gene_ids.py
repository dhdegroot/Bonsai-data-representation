"""
bonsai_scout/change_gene_ids.py (area D): replace the gene IDs in a Bonsai metadata.json by new names from a
mapping file, keeping a time-stamped copy of the old file. Expected results are written out by hand from the
function's comments ("Make new gene names unique") and the script's help texts.
"""
import glob
import json
import os
import re
import subprocess
import sys

import pytest

from bonsai_scout.change_gene_ids import change_json_file
from bonsai_testfixtures import bonsai_job_fixture
from bonsai_testrun import REPO_ROOT, bonsai_env

SCRIPT = os.path.join(REPO_ROOT, 'bonsai_scout', 'change_gene_ids.py')
METADATA = {'geneIds': ['ENSG1', 'ENSG2', 'ENSG3', 'ENSG4', 'ENSG5'], 'cellIds': ['c1', 'c2'], 'nGenes': 5,
            'loglik': -12.5}


def _write_metadata(folder):
	path = os.path.join(folder, 'metadata.json')
	with open(path, 'w') as handle:
		json.dump(METADATA, handle)
	return path


def _backups(folder):
	return [path for path in glob.glob(os.path.join(folder, 'metadata_*.json'))
	        if re.fullmatch(r'metadata_\d{6}_\d{6}\.json', os.path.basename(path))]


@pytest.mark.parametrize('ext,sep', [('.csv', ','), ('.tsv', '\t')])
def test_change_json_file_without_header(tmp_path, ext, sep):
	"""A two-column mapping without header (old ID, new ID): mapped genes get the new name, a repeated new name
	gets _promoter2, _promoter3, ... from its second occurrence on, unmapped genes keep their ID, the other keys of
	metadata.json are untouched, and the old file is kept as metadata_<yymmdd>_<HHMMSS>.json."""
	path = _write_metadata(str(tmp_path))
	mapping = tmp_path / ('mapping' + ext)
	mapping.write_text(''.join(sep.join(pair) + '\n' for pair in (
		('ENSG1', 'GATA1'), ('ENSG2', 'TAL1'), ('ENSG4', 'TAL1'), ('ENSG5', 'TAL1'), ('ENSG9', 'ABSENT'))))
	change_json_file(new_gene_ids_file=str(mapping), path_to_json=path)
	with open(path) as handle:
		new = json.load(handle)
	assert new['geneIds'] == ['GATA1', 'TAL1', 'ENSG3', 'TAL1_promoter2', 'TAL1_promoter3']
	assert {key: value for key, value in new.items() if key != 'geneIds'} == {
		key: value for key, value in METADATA.items() if key != 'geneIds'}
	(backup,) = _backups(str(tmp_path))
	with open(backup) as handle:
		assert json.load(handle) == METADATA


def test_change_json_file_with_named_columns(tmp_path):
	"""With column names, the mapping file has a header and the named columns are used, wherever they are."""
	path = _write_metadata(str(tmp_path))
	mapping = tmp_path / 'mapping.csv'
	mapping.write_text('symbol,biotype,ensembl\nGATA1,pc,ENSG1\nMYB,pc,ENSG3\n')
	change_json_file(new_gene_ids_file=str(mapping), path_to_json=path, old_geneIDs_col='ensembl',
	                 new_geneIDs_col='symbol')
	with open(path) as handle:
		assert json.load(handle)['geneIds'] == ['GATA1', 'ENSG2', 'MYB', 'ENSG4', 'ENSG5']


def _script(pool, workdir):
	path = _write_metadata(workdir)
	with open(os.path.join(workdir, 'map.tsv'), 'w') as handle:
		handle.write('old\tnew\nENSG2\tKLF1\nENSG5\tKLF1\n')
	proc = subprocess.run([sys.executable, SCRIPT, '--path_to_json', path, '--mapping_file',
	                       os.path.join(workdir, 'map.tsv'), '--geneIDs_col', 'old', '--new_geneIDs_col', 'new'],
	                      capture_output=True, text=True, env=bonsai_env(), cwd=workdir, timeout=120)
	return proc, path


d_change_gene_ids_script = bonsai_job_fixture('d_change_gene_ids_script', _script)


def test_script_options(d_change_gene_ids_script):
	"""--path_to_json, --mapping_file, --geneIDs_col and --new_geneIDs_col reach change_json_file."""
	proc, path = d_change_gene_ids_script
	assert proc.returncode == 0, proc.stderr[-2000:]
	with open(path) as handle:
		assert json.load(handle)['geneIds'] == ['ENSG1', 'KLF1', 'ENSG3', 'ENSG4', 'KLF1_promoter2']
	assert len(_backups(os.path.dirname(path))) == 1
