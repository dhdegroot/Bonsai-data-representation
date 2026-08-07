from argparse import ArgumentParser
import os
import sys
import pandas as pd
import csv
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from pathlib import Path
import psutil
import subprocess
import shutil

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
from bonsai.bonsai_helpers import mp_print, str2bool, get_sanity_binary_version, SANITY_MINIMUM_VERSION

parser = ArgumentParser(description='Splits a dataset into batches, and starts Sanity runs separately per batch.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--count_file', type=str, default=None,
                    help='Absolute path to file where counts can be found, should be either a .tsv- or a .mtx-file.')
parser.add_argument('--batch_annotation_file', type=str, default=None,
                    help='Absolute path to tsv-file with two columns. Column 1 contains all cell-IDs, column2 contains'
                         'an identifier of the batch that cell belongs to.')
parser.add_argument('--gene_names_file', type=str, default=None,
                    help='(OPTIONAL): Absolute path to file where gene-IDs are stored (one per line).')
parser.add_argument('--cell_names_file', type=str, default=None,
                    help='(OPTIONAL): Absolute path to file where cell-IDs are stored (one per line).')
parser.add_argument('--results_folder', type=str, default=None,
                    help='Absolute path to where the final results should be stored.')
parser.add_argument('--sanity_binary_path', type=str, default=None,
                    help='Absolute path to compiled Sanity binary')
parser.add_argument('--conda_env', type=str, default=None,
                    help='(OPTIONAL) Name of Sanity conda-environment')
parser.add_argument('--verbose', type=str2bool, default=False,
                    help='--verbose False only shows essential print messages (default: True)')

args = parser.parse_args()
mp_print(args)

SKIP_READING = False
SKIP_SANITY = False

from optional_preprocessing.batch_correction.batch_helpers import get_batch_annotation
from bonsai.bonsai_helpers import read_ids, write_ids

FORMAT = '%(asctime)s %(name)s %(funcName)s %(message)s'
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.getLogger().setLevel(log_level)

"""---------------------------Read in the raw counts.---------------------------"""
logger.info("Reading in data")
input_folder = os.path.dirname(os.path.abspath(args.count_file))

batch_ids, cell_ind_to_batch_ind, cell_id_to_batch_id, batch_counts = get_batch_annotation(args.batch_annotation_file)
n_batches = len(batch_ids)

if SKIP_READING:
    batch_ids.append('total_counts_per_batch')
else:
    if args.count_file.split('.')[1] == 'mtx':
        logger.debug("Reading in raw data from .mtx-file")
        M = mmread(args.count_file)
        logger.debug("Done reading in raw data from .mtx-file")
        umi_counts = M.toarray()
        # Read in promoter names
        gene_ids = read_ids(args.gene_names_file)
        # Read in cell barcodes as in mtx-file
        cell_ids = read_ids(args.cell_names_file)

    else:
        logger.debug("Reading in raw data from .txt-file")
        tmp = pd.read_csv(args.count_file, sep='\t', index_col=0)
        logger.debug("Done reading in raw data from .txt-file")
        cell_ids = list(tmp.columns)
        gene_ids = list(tmp.index)
        umi_counts = tmp.values

        # sparse_umis = csr_matrix(umi_counts)
        # mmwrite(os.path.join(input_folder, 'prom_expr_matrix.mtx'), sparse_umis)
        #
        # print("Writing cell IDs to file:")
        # with open(os.path.join(input_folder, 'cellID.txt'), 'w') as f:
        #     for ID in cell_ids:
        #         f.write("%s\n" % ID)
        #
        # print("Writing gene IDs to file:")
        # with open(os.path.join(input_folder, 'geneID.txt'), 'w') as f:
        #     for ID in gene_ids:
        #         f.write("%s\n" % ID)

    n_genes, n_cells = umi_counts.shape

    """---------------------------Split up the counts according to batch-annotation.---------------------------"""

    logger.info("Splitting data in batches")
    batch_cell_inds = {}
    batch_cell_ids = {}
    for ind_batch, batch_id in enumerate(batch_ids):
        n_cells_batch = batch_counts[ind_batch]
        batch_cell_inds[batch_id] = []
        batch_cell_ids[batch_id] = []

    for cell_ind, cell_id in enumerate(cell_ids):
        batch_id = cell_id_to_batch_id[cell_id]
        batch_cell_inds[batch_id].append(cell_ind)
        batch_cell_ids[batch_id].append(cell_id)

    logger.debug("Splitting the counts into batches.")
    batch_counts = {}
    total_counts_per_batch = np.zeros((n_genes, n_batches))
    for batch_ind, batch_id in enumerate(batch_ids):
        batch_counts[batch_id] = umi_counts[:, np.array(batch_cell_inds[batch_id])]
        total_counts_per_batch[:, batch_ind] = np.sum(batch_counts[batch_id], axis=1)

    logger.debug("Done splitting the counts into batches.")

    # Also create a subfolder in which we have all counts per batch added up
    batch_ids.append('total_counts_per_batch')
    batch_counts['total_counts_per_batch'] = total_counts_per_batch
    batch_cell_ids['total_counts_per_batch'] = batch_ids[:-1]

    # Store the data per batch in subfolders
    logger.debug("Storing the batch-counts.")
    for batch_id in batch_ids:
        Path(os.path.join(input_folder, 'batch_corrected', batch_id)).mkdir(parents=True, exist_ok=True)
        sparse_umis = csr_matrix(batch_counts[batch_id])
        mmwrite(os.path.join(input_folder, 'batch_corrected', batch_id, 'prom_expr_matrix.mtx'), sparse_umis)

        write_ids(os.path.join(input_folder, 'batch_corrected', batch_id, 'accepted_barcodes.tsv'),
                  batch_cell_ids[batch_id])
        write_ids(os.path.join(input_folder, 'batch_corrected', batch_id, 'prom_expr_promoters.tsv'), gene_ids)

    logger.debug("Done storing the batch-counts.")

    # Also copy the batch-annotation down to this downstream folder
    dst = os.path.join(os.path.join(input_folder, 'batch_corrected'), os.path.basename(args.batch_annotation_file))
    shutil.copy2(args.batch_annotation_file, dst)

    # Path(os.path.join(input_folder, 'batch_corrected', 'total_counts_per_batch')).mkdir(parents=True, exist_ok=True)
    # sparse_umis = csr_matrix(total_counts_per_batch)
    # mmwrite(os.path.join(input_folder, 'batch_corrected', 'total_counts_per_batch', 'prom_expr_matrix.mtx'), sparse_umis)
    # with open(os.path.join(input_folder, 'batch_corrected', 'total_counts_per_batch', 'accepted_barcodes.tsv'), 'w') as f:
    #     for ID in batch_ids:
    #         f.write("%s\n" % ID)
    # with open(os.path.join(input_folder, 'batch_corrected', 'total_counts_per_batch',
    #                        'prom_expr_promoters.tsv'), 'w') as f:
    #     for ID in gene_ids:
    #         f.write("%s\n" % ID)

"""---------------------------Run Sanity on different batches separately.---------------------------"""
if not SKIP_SANITY:
    logger.info("Running Sanity on batches")

    if args.conda_env is not None:
        cmd_root = ['conda', 'run', '-n', args.conda_env, "--no-capture-output"]
    else:
        cmd_root = []

    # Use SLURM allocation if available; otherwise get physical cores
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=False)))

    # Sanity 2.0 replaced the '-max_v'-argument by '-v_m', and rejects arguments it does not recognise, so we ask the
    # binary for its version once and pick the arguments that go with it.
    sanity_version = get_sanity_binary_version(args.sanity_binary_path)
    if sanity_version is None:
        logger.warning("Could not determine the version of the Sanity-binary at {}. Assuming it is Sanity 2.0 or "
                       "newer.".format(args.sanity_binary_path))
    if (sanity_version is None) or (sanity_version >= SANITY_MINIMUM_VERSION):
        method_args = ['-v_m', 'MAP']
    else:
        method_args = ['-max_v', 'only_max_output']
    logger.info("Running Sanity with the arguments {}.".format(" ".join(method_args)))

    for batch_id in batch_ids:
        batch_folder = os.path.join(input_folder, 'batch_corrected', batch_id)
        matrix_file = os.path.join(batch_folder, 'prom_expr_matrix.mtx')
        genes_file = os.path.join(batch_folder, 'prom_expr_promoters.tsv')
        cells_file = os.path.join(batch_folder, 'accepted_barcodes.tsv')

        sanity_output_folder = os.path.join(args.results_folder, batch_id)
        logfile = os.path.join(sanity_output_folder, 'sanity_log.txt')
        Path(sanity_output_folder).mkdir(parents=True, exist_ok=True)
        cmd = cmd_root + [args.sanity_binary_path,
                          "-f", matrix_file,
                          '-mtx_genes', genes_file,
                          '-mtx_cells', cells_file,
                          '-d', sanity_output_folder,
                          '-n', str(total_cpus),
                          '-e', '1'] + method_args

        logger.info("Starting Sanity on batch {}.".format(batch_id))
        logger.debug(" ".join(cmd))
        logger.info("Logging Sanity-output in {}\n".format(logfile))

        with open(logfile, "w") as logf:
            # Start the process
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )

            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                logger.debug(line.rstrip("\n"))

            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
        # subprocess.run(cmd, check=True)
