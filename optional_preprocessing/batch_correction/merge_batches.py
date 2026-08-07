from argparse import ArgumentParser
import os
import sys
import csv
import numpy as np
import subprocess
from pathlib import Path

import logging
FORMAT = '%(asctime)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT,
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.WARNING)   # silence all libraries

# Create your app logger
logger = logging.getLogger("myapp")
logger.setLevel(log_level)

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from bonsai.bonsai_helpers import mp_print, str2bool, startMPI, read_ids, write_ids, resolve_sanity_filepaths

parser = ArgumentParser(description='Takes Sanity-output that was run on seperate batches, then merges the batches.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--dataset', type=str, default='batch_corrected_dataset',
                    help='Identifier of the dataset.')
parser.add_argument('--sanity_output_folder', type=str, default=None,
                    help='Absolute path to where the results from Sanity-batch run were stored. This folder should'
                         'contain sub-folders named after the batch-IDs')
parser.add_argument('--bonsai_input_folder', type=str, default=None,
                    help='Absolute path to where batch-corrected Bonsai-input should be stored.')
parser.add_argument('--batch_annotation_file', type=str, default=None,
                    help='Absolute path to tsv-file with two columns. Column 1 contains all cell-IDs, column2 contains'
                         'an identifier of the batch that cell belongs to.')
parser.add_argument('--bonsai_results_folder', type=str, default='batch_corrected_dataset',
                    help='Absolute path to where Bonsai-output should be stored. (This is only used for creating the '
                         'bonsai-config file. Bonsai is not run yet.)')
parser.add_argument('--verbose', type=str2bool, default=False,
                    help='--verbose False only shows essential print messages (default: True)')

args = parser.parse_args()

from bonsai.bonsai_dataprocessing import read_and_filter
from optional_preprocessing.batch_correction.batch_helpers import get_batch_annotation
import bonsai.mpi_wrapper as mpi_wrapper

mpi_rank, mpi_size = startMPI(args.verbose)
mpi_info = mpi_wrapper.get_mpi_info()

mp_print(args)

FORMAT = '%(asctime)s %(name)s %(funcName)s %(message)s'
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.getLogger().setLevel(log_level)

if args.sanity_output_folder is None:
    exit("Argument 'sanity_output_folder' is empty, so I don't know where to read the Sanity-results.")
if args.bonsai_input_folder is None:
    mp_print("Argument 'bonsai_input_folder' is empty, results will be written in the 'sanity_output_folder'.",
             WARNING=True)
    args.bonsai_input_folder = args.sanity_output_folder
if args.bonsai_results_folder is None:
    mp_print("Argument 'bonsai_results_folder' is empty, results will be written in the 'bonsai_input_folder'.",
             WARNING=True)
    args.bonsai_results_folder = args.bonsai_input_folder

"""---------------------------For each Sanity result, divide out prior independently.---------------------------"""

batch_ids, cell_ind_to_batch_ind, cell_id_to_batch_id, batch_counts = get_batch_annotation(args.batch_annotation_file)
n_batches = len(batch_ids)

# First read in the necessary output from the Sanity-run on the batch-summed counts
# We use these to get mean expression levels per gene and the total set of gene-IDs that have some counts somewhere
sanity_folder = os.path.join(args.sanity_output_folder, 'total_counts_per_batch')
all_gene_ids = read_ids(os.path.join(sanity_folder, 'geneID.txt'))
all_gene_ids_to_ind = {gene_id: ind for ind, gene_id in enumerate(all_gene_ids)}
n_genes_all = len(all_gene_ids)

# Read in mean-per-gene. Which filename Sanity used depends on its version, so we let Bonsai work that out.
gene_means = []
with open(resolve_sanity_filepaths(sanity_folder)['gene_means'], 'r') as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        gene_means.append(float(row[0]))
gene_means = np.array(gene_means)

cell_ids_all = []
deltas_all = []
d_deltas_all = []
for batch_id in batch_ids:
    # For each batch, read in the sanity-output, divide out the prior (done in read_and_filter)
    sanity_folder = os.path.join(args.sanity_output_folder, batch_id)
    tmp_folder = os.path.join(sanity_folder, 'processed_data_communication')

    # Read in the Sanity-output
    mp_print("Getting Sanity-output for {}".format(batch_id), DEBUG=True)
    # Since sanityOutput is True, read_and_filter works out the filenames itself and ignores meansfile and stdsfile.
    ltqs, ltqs_vars, gene_vars, n_cells, n_genes, genes_to_keep, \
    ltq_stds_found, n_genes_orig = read_and_filter(sanity_folder, meansfile=None,
                                                   stdsfile=None, sanityOutput=True,
                                                   zscoreCutoff=-1, mpiInfo=mpi_info, tmp_folder=tmp_folder,
                                                   verbose=args.verbose, all_genes=True)

    if mpi_rank != 0:
        mp_print("Done reading input, proceeding to next batch.", ALL_RANKS=True, DEBUG=True)
        continue

    # Center the data, such that each gene has mean zero
    ltqs -= np.mean(ltqs, axis=1, keepdims=True)

    # Get the cell- and gene-IDs
    gene_ids = read_ids(os.path.join(sanity_folder, 'geneID.txt'))
    cell_ids = read_ids(os.path.join(sanity_folder, 'cellID.txt'))
    cell_ids_all += cell_ids
    n_cells = len(cell_ids)

    if genes_to_keep is not None:
        gene_ids = list(np.array(gene_ids)[genes_to_keep])

    # Check which genes are missing. These genes had no counts at all in this batch. We will pad the Sanity-output
    # with values for these genes: the log-fold changes will all be 0, the error-bars will be large: 1e3. This will have
    # the effect that the Bonsai-likelihood will ignore the information of this gene for this cell
    gene_inds = np.array([all_gene_ids_to_ind[gene_id] for gene_id in gene_ids])

    # Initialize all genes with zero-mean, 1000 error-bar
    deltas_batch = np.zeros((n_genes_all, n_cells))
    d_deltas_batch = 1e3 * np.ones((n_genes_all, n_cells))

    # Overwrite the ones for which we have information
    deltas_batch[gene_inds, :] = ltqs
    d_deltas_batch[gene_inds, :] = np.sqrt(ltqs_vars)

    deltas_all.append(deltas_batch)
    d_deltas_all.append(d_deltas_batch)

"""---------------------------Compile the Sanity results into one feature-matrix.---------------------------"""

if mpi_rank != 0:
    mp_print("My job here is done.", ALL_RANKS=True, DEBUG=True)
    exit()

# Concatenate the ltq-matrices
mp_print("Stacking batch-corrected data.", DEBUG=True)
ltqs_all = np.hstack(deltas_all)
d_ltqs_all = np.hstack(d_deltas_all)
mp_print("Done stacking batch-corrected data.", DEBUG=True)

# Add the mean-per-gene
ltqs_all += gene_means[:, None]

# Store the results in a folder ready for being run by Sanity
Path(args.bonsai_input_folder).mkdir(parents=True, exist_ok=True)
# Write gene and cell names
write_ids(os.path.join(args.bonsai_input_folder, 'geneID.txt'), all_gene_ids)
write_ids(os.path.join(args.bonsai_input_folder, 'cellID.txt'), cell_ids_all)
# Write ltqs and corresponding stds describing the likelihood function
mp_print("Saving batch-corrected data.", DEBUG=True)
np.savetxt(os.path.join(args.bonsai_input_folder, 'features.txt'), ltqs_all, delimiter='\t')
np.savetxt(os.path.join(args.bonsai_input_folder, 'standard_deviations.txt'), d_ltqs_all, delimiter='\t')

"""---------------------------Create bonsai-YAML-config-file for the Bonsai run.---------------------------"""

Path(args.bonsai_results_folder).mkdir(parents=True, exist_ok=True)
mp_print("Done saving batch-corrected data, now creating config-file for the Bonsai-run.", DEBUG=True)
create_config_command = [sys.executable,
                         os.path.join(parent_dir, 'bonsai/create_config_file.py'),
                         '--new_yaml_path',
                         os.path.join(args.bonsai_results_folder, 'bonsai_config.yaml'),
                         '--dataset',
                         args.dataset,
                         '--data_folder',
                         args.bonsai_input_folder,
                         '--filenames_data',
                         'features.txt,standard_deviations.txt',
                         '--results_folder',
                         args.bonsai_results_folder,
                         '--input_is_sanity_output',
                         'False',
                         '--zscore_cutoff',
                         str(1.0),
                         '--UB_ellipsoid_size',
                         str(1.0),
                         '--nnn_n_randommoves',
                         str(1000),
                         '--nnn_n_randomtrees',
                         str(10),
                         '--pickup_intermediate',
                         'True',
                         '--use_knn',
                         str(10),
                         '--rescale_by_var',
                         'True']

output1 = subprocess.run(create_config_command, stdout=subprocess.PIPE, text=True)
mp_print(output1.stdout)
mp_print(output1.stderr)
