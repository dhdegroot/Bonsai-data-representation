from pathlib import Path
import os
import sys
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path, such that the Bonsai-helpers can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from bonsai.bonsai_helpers import resolve_sanity_input, read_sanity_gene_vector

parser = ArgumentParser(
    description='')
# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--sanity_folder', type=str,
                    default='/Users/Daan/Documents/postdoc/waddington-code-github/python_waddington_code/data/Zeisel',
                    help='Path to folder where Sanity results can be found.')

args = parser.parse_args()
figsPath = os.path.join(args.sanity_folder, 'QCfigs')
Path(figsPath).mkdir(parents=True, exist_ok=True)

# Which files Sanity has written, and what they hold, depends on its version. We let Bonsai work that out.
sanity_input = resolve_sanity_input(args.sanity_folder)
sanity_filepaths = sanity_input['filepaths']

# Get mean LTQ for each gene
mu_g = read_sanity_gene_vector(sanity_filepaths['gene_means'])

# Plot histogram of mean LTQs
fig, ax = plt.subplots()
ax.hist(mu_g, bins=100)
ax.set_xlabel("Mean LTQ per gene")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of inferred mean log-transcription-quotients (LTQs)")
fig.savefig(os.path.join(figsPath, 'mean_LTQ_histogram.png'))

# Get variance for each gene
var_g = read_sanity_gene_vector(sanity_filepaths['gene_variances'])

# Plot histogram for variances
fig, ax = plt.subplots()
ax.hist(var_g, bins=100)
ax.set_xlabel("Variance per gene")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of inferred variances per gene (v_g)")
ax.set_yscale('log')
fig.savefig(os.path.join(figsPath, 'var_histogram.png'))

# Plot scatter of mean against variance
fig, ax = plt.subplots()
ax.scatter(mu_g, var_g, s=0.1, alpha=0.5)
ax.set_xlabel("Mean LTQ per gene")
ax.set_ylabel("Variance in LTQs per gene")
ax.set_title("Scatter of mean log-transcription-quotient (LTQ) against variance in LTQ")
ax.set_yscale('log')
fig.savefig(os.path.join(figsPath, 'scatter_mean_var.png'))

# Get the signal-to-noise ratio per gene, as Bonsai will calculate it for its zscore-cutoff
if sanity_input['measurement_ltqs']:
    # From Sanity 3.0 onwards we get the measurement LTQs with their error-bars, out of which Sanity's prior is already
    # divided. These are absolute LTQs, and the signal-to-noise ratio accounts for the inferred gene-variance.
    if sanity_input['numpy_input']:
        ltqs = np.load(sanity_filepaths['means'], allow_pickle=False).astype(float)
        d_ltqs = np.load(sanity_filepaths['stds'], allow_pickle=False).astype(float)
    else:
        # The text-tables have a header-line with cell-IDs, and start every row with the gene-ID
        n_columns = len(open(sanity_filepaths['means'], 'r').readline().split('\t'))
        ltqs = np.loadtxt(sanity_filepaths['means'], delimiter='\t', skiprows=1, usecols=range(1, n_columns), ndmin=2)
        d_ltqs = np.loadtxt(sanity_filepaths['stds'], delimiter='\t', skiprows=1, usecols=range(1, n_columns), ndmin=2)
    variances = np.maximum(d_ltqs ** 2, 1e-12)
    zscores = np.sqrt(np.mean((var_g[:, np.newaxis] / (var_g[:, np.newaxis] + variances)) * (
            ltqs - mu_g[:, np.newaxis]) ** 2 / variances, axis=1))
    zscore_label = "Zscore per gene: <(v / (v + std^2)) ((ltq - mu)/std)^2> "
else:
    # Before Sanity 3.0 we get the posterior log-fold changes with respect to the gene-mean, with their error-bars
    deltas = np.loadtxt(sanity_filepaths['means'], ndmin=2)
    d_deltas = np.loadtxt(sanity_filepaths['stds'], ndmin=2)
    zscores = np.sqrt(np.mean(((deltas - np.mean(deltas, axis=1)[:, np.newaxis]) ** 2) / (d_deltas ** 2), axis=1))
    zscore_label = "Zscore per gene: <((ltq - <ltq>)/std)^2> "
# Plot histogram for variances
fig, ax = plt.subplots()
ax.hist(zscores, bins=200, cumulative=-1, histtype='step')
ax.set_xlabel(zscore_label)
ax.set_ylabel("Frequency")
ax.set_title("Histogram of zscores per gene, {} genes with zscore > 1".format(np.sum(zscores > 1)))
ax.set_yscale('log')
# ax.set_xscale('log')
plt.show()
fig.savefig(os.path.join(figsPath, 'zscores_histogram.png'))
