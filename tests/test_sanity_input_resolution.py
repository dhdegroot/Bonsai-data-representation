"""
Tests for working out which files Bonsai should read from a Sanity output-folder.

Sanity older than 2.0 had to be run with '-max_v', and wrote the estimates at the maximum-likelihood gene-variance
into files with a '_vmax'-suffix. Sanity 2.0 replaced that argument by '-v_m', which picks the method used to estimate
the gene-variances, and writes the plain filenames whichever method was picked. It also started writing a
'sanity_command.txt'-file, which records the version and the method, and is what Bonsai uses to tell the two apart.
Sanity 3.0 names every file after what it holds. Bonsai then reads the measurement LTQs, out of which Sanity has
already divided its prior, as NPY-files ('--bonsai-output-mode') or as text-files.
"""
import pytest

from bonsai.bonsai_helpers import (
    SANITY_FILENAMES,
    SANITY_LEGACY_FILENAMES,
    SANITY_MEASUREMENT_FILENAMES,
    SANITY_MEASUREMENT_NPY_FILENAMES,
    parse_sanity_command_file,
    parse_sanity_command_headers,
    parse_version_string,
    resolve_sanity_filepaths,
    resolve_sanity_input,
)

COMMAND_TEMPLATE = ("# Timestamp: 2026-08-07 10:42:37\n"
                    "# Sanity version: {version}\n"
                    "# Method: {method}\n"
                    "../bin/Sanity -f ../tests/count_table.tsv -e 1 -d ../results/test_example -v_m {method}\n")


def make_sanity_folder(folder, filenames, command_contents=None):
    """
    Builds a folder that looks like Sanity output, with empty stubs for the given filenames.
    :param folder: Folder to fill, as a pathlib.Path.
    :param filenames: Iterable of filenames to create.
    :param command_contents: Contents of the 'sanity_command.txt'-file, or None to leave that file out.
    :return: The folder, as a string.
    """
    for filename in filenames:
        (folder / filename).write_text('')
    if command_contents is not None:
        (folder / 'sanity_command.txt').write_text(command_contents)
    return str(folder)


def test_parse_version_string_pads_missing_components():
    assert parse_version_string('2') == (2, 0, 0)
    assert parse_version_string('2.0') == (2, 0, 0)
    assert parse_version_string('2.0.0') == (2, 0, 0)
    assert parse_version_string(' 1.1 ') == (1, 1, 0)
    assert parse_version_string('2.1.0-beta') == (2, 1, 0)
    assert parse_version_string('unknown') is None


def test_parse_sanity_command_file_reads_headers(tmp_path):
    command_path = tmp_path / 'sanity_command.txt'
    command_path.write_text(COMMAND_TEMPLATE.format(version='2.0.0', method='MLE'))
    assert parse_sanity_command_file(str(command_path)) == ((2, 0, 0), 'MLE')


def test_parse_sanity_command_file_ignores_command_line(tmp_path):
    """The Sanity-command itself may well mention '# Method:'-like text; only the header-lines count."""
    command_path = tmp_path / 'sanity_command.txt'
    command_path.write_text("# Sanity version: 2.0\n"
                            "Sanity -f counts.tsv -d out\n"
                            "# Method: MARG\n")
    assert parse_sanity_command_file(str(command_path)) == ((2, 0, 0), None)


def test_legacy_sanity_output_resolves_to_vmax_files(tmp_path):
    """Without a 'sanity_command.txt', we expect the '_vmax'-files of Sanity older than 2.0."""
    data_folder = make_sanity_folder(tmp_path, SANITY_LEGACY_FILENAMES.values())
    filepaths = resolve_sanity_filepaths(data_folder)
    assert {key: filepath.rsplit('/', 1)[-1] for key, filepath in filepaths.items()} == SANITY_LEGACY_FILENAMES


def test_legacy_sanity_output_without_vmax_files_is_rejected(tmp_path, caplog):
    """Sanity older than 2.0 run without '-max_v' leaves only the plain filenames, which Bonsai cannot use."""
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values())
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert '-max_v only_max_output' in caplog.text


def test_empty_folder_is_rejected(tmp_path, caplog):
    data_folder = make_sanity_folder(tmp_path, [])
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert '--input_is_sanity_output' in caplog.text


@pytest.mark.parametrize('version', ['2.0', '2.0.0', '2.1.3'])
@pytest.mark.parametrize('method', ['MLE', 'MAP', 'EAP'])
def test_new_sanity_output_resolves_to_plain_files(tmp_path, version, method):
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values(),
                                     COMMAND_TEMPLATE.format(version=version, method=method))
    filepaths = resolve_sanity_filepaths(data_folder)
    assert {key: filepath.rsplit('/', 1)[-1] for key, filepath in filepaths.items()} == SANITY_FILENAMES


def test_marginalising_method_is_rejected(tmp_path, caplog):
    """The 'MARG'-method marginalises over the gene-variance, so its output is not a likelihood Bonsai can use."""
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values(),
                                     COMMAND_TEMPLATE.format(version='2.0.0', method='MARG'))
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert '-v_m MAP' in caplog.text


def test_unknown_method_is_rejected(tmp_path, caplog):
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values(),
                                     COMMAND_TEMPLATE.format(version='2.0.0', method='SOMETHING_NEW'))
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert 'SOMETHING_NEW' in caplog.text


def test_too_old_version_is_rejected(tmp_path, caplog):
    """A 'sanity_command.txt' claiming a version older than 2.0 should not be trusted to hold the new filenames."""
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values(),
                                     COMMAND_TEMPLATE.format(version='1.9', method='MLE'))
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert '1.9' in caplog.text


def test_unreadable_version_is_rejected(tmp_path, caplog):
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values(),
                                     "# Timestamp: 2026-08-07 10:42:37\n# Method: MLE\n")
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert 'Sanity version' in caplog.text


def test_missing_method_is_rejected(tmp_path, caplog):
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values(),
                                     "# Timestamp: 2026-08-07 10:42:37\n# Sanity version: 2.0.0\n")
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert 'Method' in caplog.text


def test_new_sanity_output_without_extended_output_is_rejected(tmp_path, caplog):
    """Without '-e 1', Sanity 2.0 only writes the log-transcription-quotients, not the files Bonsai needs."""
    data_folder = make_sanity_folder(tmp_path, ['delta.txt'],
                                     COMMAND_TEMPLATE.format(version='2.0.0', method='MAP'))
    with pytest.raises(SystemExit):
        resolve_sanity_filepaths(data_folder)
    assert '-e 1' in caplog.text


BONSAI_MODE_COMMAND = ("# Timestamp: 2026-11-16 12:00:00\n"
                       "# Sanity version: 3.0.0\n"
                       "# Method: EAP\n"
                       "# LTQ estimate: measured\n"
                       "# Output mode: bonsai\n"
                       "Sanity -f counts.tsv -d sanity_output --bonsai-output-mode\n")


def basenames(filepaths):
    return {key: filepath.rsplit('/', 1)[-1] for key, filepath in filepaths.items()}


def test_parse_sanity_command_headers_reads_ltq_estimate_and_output_mode(tmp_path):
    command_path = tmp_path / 'sanity_command.txt'
    command_path.write_text(BONSAI_MODE_COMMAND)
    assert parse_sanity_command_headers(str(command_path)) == {
        'version': (3, 0, 0), 'method': 'EAP', 'ltq_estimate': 'measured', 'output_mode': 'bonsai'}
    # The older function still gives just the version and the method
    assert parse_sanity_command_file(str(command_path)) == ((3, 0, 0), 'EAP')


def test_older_sanity_output_is_posterior_text(tmp_path):
    """Before Sanity 3.0 the files hold posterior deviations from the gene-mean, and Bonsai divides out the prior."""
    data_folder = make_sanity_folder(tmp_path, SANITY_FILENAMES.values(),
                                     COMMAND_TEMPLATE.format(version='2.0.0', method='MAP'))
    sanity_input = resolve_sanity_input(data_folder)
    assert not sanity_input['measurement_ltqs']
    assert not sanity_input['numpy_input']
    assert basenames(sanity_input['filepaths']) == SANITY_FILENAMES


def test_bonsai_output_mode_resolves_to_measurement_npy_files(tmp_path):
    data_folder = make_sanity_folder(tmp_path, SANITY_MEASUREMENT_NPY_FILENAMES.values(), BONSAI_MODE_COMMAND)
    sanity_input = resolve_sanity_input(data_folder)
    assert sanity_input['measurement_ltqs']
    assert sanity_input['numpy_input']
    assert basenames(sanity_input['filepaths']) == SANITY_MEASUREMENT_NPY_FILENAMES
    assert basenames(resolve_sanity_filepaths(data_folder)) == SANITY_MEASUREMENT_NPY_FILENAMES


@pytest.mark.parametrize('ltq_estimate', ['measured', 'both'])
@pytest.mark.parametrize('method', ['MLE', 'MAP', 'EAP'])
def test_text_run_with_measurement_ltqs_resolves_to_measurement_text_files(tmp_path, ltq_estimate, method):
    """Without '--bonsai-output-mode', '--ltq-estimate measured' or 'both' writes the same files as text."""
    data_folder = make_sanity_folder(
        tmp_path, SANITY_MEASUREMENT_FILENAMES.values(),
        "# Sanity version: 3.0.0\n# Method: {}\n# LTQ estimate: {}\n".format(method, ltq_estimate))
    sanity_input = resolve_sanity_input(data_folder)
    assert sanity_input['measurement_ltqs']
    assert not sanity_input['numpy_input']
    assert basenames(sanity_input['filepaths']) == SANITY_MEASUREMENT_FILENAMES


def test_npy_measurement_files_are_found_without_the_output_mode_line(tmp_path):
    """A run with '-npy --ltq-estimate measured' writes the NPY-files without recording the Bonsai output-mode."""
    data_folder = make_sanity_folder(tmp_path, SANITY_MEASUREMENT_NPY_FILENAMES.values(),
                                     "# Sanity version: 3.0.0\n# Method: EAP\n# LTQ estimate: measured\n")
    assert resolve_sanity_input(data_folder)['numpy_input']


def test_posterior_only_run_is_rejected_with_the_remedy(tmp_path, caplog):
    """The prior cannot be divided out of the posterior estimates of Sanity 3.0 cell by cell, so we do not read them."""
    data_folder = make_sanity_folder(tmp_path, ['ltqs_posterior.txt', 'ltqs_posterior_error_bars.txt',
                                                'gene_means.txt', 'gene_variances.txt'],
                                     "# Sanity version: 3.0.0\n# Method: MAP\n# LTQ estimate: posterior\n")
    with pytest.raises(SystemExit):
        resolve_sanity_input(data_folder)
    assert '--bonsai-output-mode' in caplog.text


def test_missing_ltq_estimate_line_is_rejected(tmp_path, caplog):
    data_folder = make_sanity_folder(tmp_path, SANITY_MEASUREMENT_NPY_FILENAMES.values(),
                                     "# Sanity version: 3.0.0\n# Method: EAP\n# Output mode: bonsai\n")
    with pytest.raises(SystemExit):
        resolve_sanity_input(data_folder)
    assert 'LTQ estimate' in caplog.text


def test_missing_measurement_files_are_rejected(tmp_path, caplog):
    data_folder = make_sanity_folder(tmp_path, ['gene_means.npy', 'gene_variances.npy'], BONSAI_MODE_COMMAND)
    with pytest.raises(SystemExit):
        resolve_sanity_input(data_folder)
    assert 'ltqs_measurement_approx.npy' in caplog.text
    assert '--bonsai-output-mode' in caplog.text


def test_marginalising_method_is_rejected_for_measurement_ltqs_too(tmp_path, caplog):
    data_folder = make_sanity_folder(tmp_path, SANITY_MEASUREMENT_NPY_FILENAMES.values(),
                                     BONSAI_MODE_COMMAND.replace('EAP', 'MARG'))
    with pytest.raises(SystemExit):
        resolve_sanity_input(data_folder)
    assert 'MARG' in caplog.text
