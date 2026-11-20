"""
Tests for reading the output of Sanity 3.0 and newer.

From version 3.0 onwards, Bonsai reads Sanity's measurement LTQs: the posterior with Sanity's prior divided out, which
Sanity does exactly, once, for all cells of a gene together. Bonsai therefore uses these values and error-bars as they
are: nothing is divided out, no gene-mean is added, and every cell carries information. For older Sanity-output, Bonsai
still divides out the prior itself.
"""
import numpy as np
import pytest

import bonsai.mpi_wrapper as mpi_wrapper
from bonsai.bonsai_dataprocessing import read_and_filter

# Four genes and four cells, with values that are exactly representable in float32 and in a text-file, such that the
# NPY-files and the text-files give bit-identical results.
MEANS = np.array([[-8., -8., -8., -8.],
                  [-7., -6., -5., -4.],
                  [-9., -9., -9., -9.],
                  [-6., -5., -6., -5.]])
STDS = np.array([[0.5, 0.5, 0.5, 0.5],
                 [0.25, 0.25, 0.25, 0.25],
                 [8., 8., 8., 8.],
                 [0.25, 0.25, 0.25, 0.25]])
GENE_MEANS = np.array([-8., -5.5, -9., -5.5])
GENE_VARIANCES = np.array([2., 2., 0.001, 0.5])
GENE_IDS = ['g0', 'g1', 'g2', 'g3']
CELL_IDS = ['cell_a', 'cell_b', 'cell_c', 'cell_d']


def write_ids(folder):
    (folder / 'geneID.txt').write_text('\n'.join(GENE_IDS) + '\n')
    (folder / 'cellID.txt').write_text('\n'.join(CELL_IDS) + '\n')


def write_npy_folder(folder):
    """Writes what Sanity 3.0 writes with '--bonsai-output-mode': float32 matrices and float64 vectors of shape (G, 1)."""
    write_ids(folder)
    (folder / 'sanity_command.txt').write_text("# Sanity version: 3.0.0\n# Method: EAP\n# LTQ estimate: measured\n"
                                               "# Output mode: bonsai\nSanity --bonsai-output-mode\n")
    np.save(folder / 'ltqs_measurement_approx.npy', MEANS.astype(np.float32))
    np.save(folder / 'ltqs_measurement_approx_error_bars.npy', STDS.astype(np.float32))
    np.save(folder / 'gene_means.npy', GENE_MEANS[:, np.newaxis])
    np.save(folder / 'gene_variances.npy', GENE_VARIANCES[:, np.newaxis])
    return str(folder)


def write_text_folder(folder):
    """Writes what Sanity 3.0 writes with '--ltq-estimate both': tables with a header-line and the gene-ID per row."""
    write_ids(folder)
    (folder / 'sanity_command.txt').write_text("# Sanity version: 3.0.0\n# Method: EAP\n# LTQ estimate: both\n"
                                               "Sanity --ltq-estimate both\n")

    def table(matrix):
        lines = ['\t'.join(['GeneID'] + CELL_IDS)]
        for gene_id, row in zip(GENE_IDS, matrix):
            lines.append('\t'.join([gene_id] + ['%.6f' % value for value in row]))
        return '\n'.join(lines) + '\n'

    (folder / 'ltqs_measurement_approx.txt').write_text(table(MEANS))
    (folder / 'ltqs_measurement_approx_error_bars.txt').write_text(table(STDS))
    (folder / 'gene_means.txt').write_text(''.join('%.6f\n' % value for value in GENE_MEANS))
    (folder / 'gene_variances.txt').write_text(''.join('%.6f\n' % value for value in GENE_VARIANCES))
    return str(folder)


def read(data_folder, zscore_cutoff, tmp_path):
    return read_and_filter(data_folder, meansfile=None, stdsfile=None, sanityOutput=True, zscoreCutoff=zscore_cutoff,
                           mpiInfo=mpi_wrapper.get_mpi_info(singleProcess=True), tmp_folder=str(tmp_path / 'tmp'))


@pytest.mark.parametrize('write_folder', [write_npy_folder, write_text_folder])
def test_measurement_ltqs_are_used_as_they_are(tmp_path, write_folder):
    data_folder = write_folder(tmp_path)
    ltqs, ltqs_vars, gene_vars, n_cells, n_genes, genes_to_keep, stds_found, n_genes_orig = read(data_folder, 1.0,
                                                                                               tmp_path)
    assert stds_found
    assert n_genes_orig == 4
    assert n_cells == 4
    # The signal-to-noise ratio is mean_c [v / (v + s_c^2)] (x_c - mu)^2 / s_c^2, with Sanity's own mu and v.
    # g0: all cells at the gene-mean, so zero. g1: (2 / 2.0625) * 16 * 1.25 = 19.4. g2: all cells at the gene-mean.
    # g3: (0.5 / 0.5625) * 16 * 0.25 = 3.56. So at a cutoff of 1, g1 and g3 are kept.
    np.testing.assert_array_equal(genes_to_keep, [1, 3])
    assert n_genes == 2
    # Nothing is divided out and no gene-mean is added: we get the measurement LTQs and their variances back.
    np.testing.assert_array_equal(ltqs, MEANS[[1, 3], :])
    np.testing.assert_array_equal(ltqs_vars, STDS[[1, 3], :] ** 2)
    # The gene-variances are the ones Sanity inferred.
    np.testing.assert_array_equal(gene_vars, GENE_VARIANCES[[1, 3]])


def test_npy_and_text_input_give_the_same_result(tmp_path):
    (tmp_path / 'npy').mkdir()
    (tmp_path / 'text').mkdir()
    from_npy = read(write_npy_folder(tmp_path / 'npy'), 1.0, tmp_path)
    from_text = read(write_text_folder(tmp_path / 'text'), 1.0, tmp_path)
    for value_npy, value_text in zip(from_npy, from_text):
        np.testing.assert_array_equal(value_npy, value_text)


def test_zscore_cutoff_selects_on_the_measurement_signal_to_noise(tmp_path):
    data_folder = write_npy_folder(tmp_path)
    # g3 has a squared signal-to-noise of 3.56, so it falls out at a cutoff of 2, which would leave only g1.
    # Bonsai refuses to continue with a single gene.
    with pytest.raises(SystemExit):
        read(data_folder, 2.0, tmp_path)
    _, _, _, _, n_genes, genes_to_keep, _, _ = read(data_folder, 1.8, tmp_path)
    np.testing.assert_array_equal(genes_to_keep, [1, 3])
    assert n_genes == 2


def test_error_bars_wider_than_the_prior_do_not_remove_a_gene(tmp_path):
    """
    For older Sanity-output, a cell with a posterior error-bar as wide as the prior makes Bonsai drop the gene, since
    the prior cannot be divided out. Measurement error-bars are often much wider than the prior, and that is fine.
    """
    data_folder = write_npy_folder(tmp_path)
    stds = STDS.astype(np.float32)
    stds[1, 0] = 64.  # Much wider than the prior of g1, which has standard deviation sqrt(2).
    np.save(tmp_path / 'ltqs_measurement_approx_error_bars.npy', stds)
    ltqs, ltqs_vars, _, _, _, genes_to_keep, _, _ = read(data_folder, 1.0, tmp_path)
    np.testing.assert_array_equal(genes_to_keep, [1, 3])
    assert ltqs[0, 0] == MEANS[1, 0]
    assert ltqs_vars[0, 0] == 64. ** 2


def test_older_sanity_output_still_gets_the_prior_divided_out(tmp_path):
    """Sanity 2 reports posterior deviations from the gene-mean; the prior is divided out and the mean added back."""
    write_ids(tmp_path)
    (tmp_path / 'sanity_command.txt').write_text("# Sanity version: 2.0.0\n# Method: MAP\nSanity -e 1 -v_m MAP\n")
    deltas = np.array([[1., -1., 1., -1.], [2., 2., -2., -2.], [0.5, 0.5, -0.5, -0.5]])
    stds = np.full((3, 4), 0.5)
    gene_means = np.array([-8., -6., -7.])
    gene_variances = np.array([2., 1., 4.])
    np.savetxt(tmp_path / 'delta.txt', deltas, delimiter='\t')
    np.savetxt(tmp_path / 'd_delta.txt', stds, delimiter='\t')
    np.savetxt(tmp_path / 'mu.txt', gene_means)
    np.savetxt(tmp_path / 'variance.txt', gene_variances)
    (tmp_path / 'geneID.txt').write_text('g0\ng1\ng2\n')
    ltqs, ltqs_vars, gene_vars, _, _, genes_to_keep, _, _ = read(str(tmp_path), 0.5, tmp_path)
    np.testing.assert_array_equal(genes_to_keep, [0, 1, 2])
    factor = gene_variances[:, np.newaxis] / (gene_variances[:, np.newaxis] - stds ** 2)
    np.testing.assert_allclose(ltqs, deltas * factor + gene_means[:, np.newaxis], rtol=1e-12)
    np.testing.assert_allclose(ltqs_vars, stds ** 2 * factor, rtol=1e-12)
    np.testing.assert_array_equal(gene_vars, gene_variances)
