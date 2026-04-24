from argparse import ArgumentParser
import pandas as pd
import os
import numpy as np
import shutil
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
import csv
from pathlib import Path
from collections import defaultdict
from time import time

import logging

FORMAT = '%(asctime)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
log_level = logging.INFO
logging.basicConfig(format=FORMAT,
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.WARNING)  # silence all libraries

# Create your app logger
logger = logging.getLogger("myapp")
logger.setLevel(log_level)


def merge_clusters_acc_cellstates(unfiltered_clusters, unfiltered_umi_counts, unfiltered_cell_ids,
                                  cutoff_annot=0, cutoff_filter=0, verbose=True):
    """
    This function takes cellstates clusters and a umiCounts matrix, and the corresponding cell_ids.
    It finds the unique cellstates clusters, and then selects the corresponding columns from the UMI-count
    matrix, and adds them together in a new "summed" UMI-counts cluster.
    It returns
    summed_counts_clst: new UMI-matrix
    counts: number of cells per cellstate
    cs_annot: Annotation that can be used for the cellstates (indicating the cellstate name,
    and cs<{small_cs_annot_cutoff} for smaller cellstates
    cell_annot: Annotation that can be used for the cells (indicating the same as above, but now per cell)
    cs_ids: The new ids of the cellstates
    cell_id_to_cs_id: Mapping from each cell-ID to their cs-ID
    """
    unfiltered_clsts, unfiltered_cell_ind_to_cs_ind, unfiltered_counts = np.unique(unfiltered_clusters,
                                                                                   return_inverse=True,
                                                                                   return_counts=True)

    unfiltered_n_clsts = len(unfiltered_clsts)
    # Filter out cells/cellstates that are part of a cellstate with <= than cutoff_filter cells
    clsts = []
    counts = []
    selected_cs_inds = []
    csind_to_selected_csind = {}
    for ind, cs_count in enumerate(unfiltered_counts):
        if cs_count > cutoff_filter:
            clsts.append(unfiltered_clsts[ind])
            counts.append(unfiltered_counts[ind])
            csind_to_selected_csind[ind] = len(selected_cs_inds)
            selected_cs_inds.append(ind)
    clsts = np.array(clsts)
    counts = np.array(counts)
    selected_cs_inds = set(selected_cs_inds)

    cell_ind_to_cs_ind = []
    clusters = []
    selected_cell_inds = []
    for c_ind, cs_ind in enumerate(unfiltered_cell_ind_to_cs_ind):
        if cs_ind in selected_cs_inds:
            cell_ind_to_cs_ind.append(csind_to_selected_csind[cs_ind])
            selected_cell_inds.append(c_ind)
            clusters.append(unfiltered_clusters[c_ind])
    selected_cell_inds = np.array(selected_cell_inds)
    cell_ind_to_cs_ind = np.array(cell_ind_to_cs_ind)
    clusters = np.array(clusters)
    n_clst = len(clsts)

    if not len(selected_cell_inds):
        logger.error("No cellstates made the filter-cutoff. Exiting.")
        exit()

    # Select only the counts for the selected cells
    umi_counts = unfiltered_umi_counts[:, selected_cell_inds]
    cell_ids = [unfiltered_cell_ids[c_ind] for c_ind in selected_cell_inds]

    logger.info("Selected {}/{} cellstates because they had more than {} cells. "
                "Total number of selected cells: {}/{}.".format(n_clst, unfiltered_n_clsts, cutoff_filter,
                                                                len(cell_ids), len(unfiltered_cell_ids)))

    cs_ids = ['cs_' + str(ind) for ind in range(n_clst)]
    cs_annot = []
    for ind, cs_count in enumerate(counts):
        if cs_count > cutoff_annot:
            cs_annot.append('cs_{}'.format(ind))
        else:
            cs_annot.append('cs<={}'.format(cutoff_annot))

    cell_id_to_cs_id = {}
    cell_annot = []
    for cell_ind, cs_ind in enumerate(cell_ind_to_cs_ind):
        cell_id_to_cs_id[cell_ids[cell_ind]] = cs_ids[cs_ind]
        if counts[cs_ind] > cutoff_annot:
            cell_annot.append('cs_{}'.format(cs_ind))
        else:
            cell_annot.append('cs<{}'.format(cutoff_annot))

    start = time()
    # Create a dictionary going from cluster-label to cell-indices
    cluster_to_indices = defaultdict(list)
    for ind, cat in enumerate(clusters):
        cluster_to_indices[cat].append(ind)
    cluster_to_indices = dict(cluster_to_indices)
    summed_counts_clst = np.zeros((umi_counts.shape[0], n_clst), dtype=int)
    for i, clst_name in enumerate(clsts):
        if (i % 10) == 0:
            logger.debug("Merging cellstate {}".format(i))
        if len(cluster_to_indices[clst_name]):
            summed_counts_clst[:, i] = np.sum(umi_counts[:, cluster_to_indices[clst_name]], axis=1)
        else:
            logger.warning("How can this mask be empty?")
    print("Merging cellstates took {} seconds.".format(time() - start))

    return summed_counts_clst, counts, cs_annot, cell_annot, cs_ids, cell_id_to_cs_id, cell_ids


if __name__ == '__main__':
    """Parse input arguments"""
    parser = ArgumentParser(
        description='Sums UMI-counts for all cells in same cellstate. Stores resulting "super-cells" as Sanity input.')
    parser.add_argument('--folder_cellstates_output', type=str, default='.',
                        help='Path to the folder where the cellstates output can be found')
    parser.add_argument('--file_raw_umi_counts', type=str, default='',
                        help='Path to the folder where UMI-counts can be found')
    parser.add_argument('--file_cell_ids', type=str, default=None,
                        help='Path to file where cell-ids can be found')
    parser.add_argument('--file_gene_ids', type=str, default=None,
                        help='Path to file where gene-ids can be found')
    parser.add_argument('--folder_clustered_umi_counts', type=str, default=None,
                        help='Folder where clustered umi counts should be stored.')
    parser.add_argument('--cutoff_annot', type=int, default=0,
                        help='Determines size of cellstates that get annotated as "small".')
    parser.add_argument('--cutoff_filter', type=int, default=0,
                        help='Determines size of cellstates that get thrown out.')

    args = parser.parse_args()

    AS_MTX = True

    results_folder = args.folder_clustered_umi_counts
    input_folder = os.path.dirname(os.path.abspath(args.file_raw_umi_counts))
    if results_folder is None:
        results_folder = os.path.join(input_folder, 'cs_merged')
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    clustering = pd.read_csv(os.path.join(args.folder_cellstates_output, 'optimized_clusters.txt'), sep='\t',
                             header=None).values.astype(dtype=int).flatten()

    if args.file_raw_umi_counts.split('.')[1] == 'mtx':
        M = mmread(os.path.join(args.file_raw_umi_counts, args.file_raw_umi_counts))
        unfiltered_umi_counts = M.toarray().astype(dtype=int)
        # Read in promoter names
        gene_ids = []
        with open(os.path.join(args.file_gene_ids), 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                gene_ids.append(row[0])

        # Read in cell barcodes as in mtx-file
        unfiltered_cell_ids = []
        with open(args.file_cell_ids, 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                unfiltered_cell_ids.append(row[0])
    else:
        tmp = pd.read_csv(os.path.join(args.folder_raw_umi_counts, args.file_raw_umi_counts), sep='\t', index_col=0)
        unfiltered_cell_ids = list(tmp.columns)
        gene_ids = list(tmp.index)
        unfiltered_umi_counts = tmp.values.astype(dtype='int')

    if os.path.exists(os.path.join(args.folder_cellstates_output, 'CellID.txt')):
        cell_ids_cellstates = pd.read_csv(os.path.join(args.folder_cellstates_output, 'CellID.txt'), sep='\t',
                                          header=None).values.flatten()
        logger.debug("First 10 cell ids, raw input: {}".format(unfiltered_cell_ids[:10]))
        logger.debug("First 10 cell ids, cellstates input: {}".format(cell_ids_cellstates[:10]))
    else:
        cell_ids_cellstates = unfiltered_cell_ids

    for ind, cell_ID in enumerate(unfiltered_cell_ids):
        if cell_ID != cell_ids_cellstates[ind]:
            print("Cell IDs do not match between cellstates-output and raw data. Quitting conversion.")
            exit()

    summed_counts_clst, counts, cs_annot, cell_annot, \
        cs_ids, cell_id_to_cs_id, cell_ids = merge_clusters_acc_cellstates(
        clustering, unfiltered_umi_counts, unfiltered_cell_ids=cell_ids_cellstates, cutoff_annot=args.cutoff_annot,
        cutoff_filter=args.cutoff_filter
    )
    n_cellstates = summed_counts_clst.shape[1]

    if not AS_MTX:
        new_umi_counts_df = pd.DataFrame(summed_counts_clst, columns=cs_ids, index=gene_ids)
        new_umi_counts_df.to_csv(os.path.join(results_folder, 'Gene_table.txt'), index=True, sep='\t',
                                 index_label="GeneID")
    else:
        sparse_umis = csr_matrix(summed_counts_clst)
        mmwrite(os.path.join(results_folder, 'prom_cs_expr_matrix.mtx'), sparse_umis)
    shutil.copyfile(os.path.join(args.folder_cellstates_output, 'optimized_clusters.txt'),
                    os.path.join(results_folder, 'cs_clusters.txt'))
    with open(os.path.join(results_folder, 'orig_CellID.txt'), 'w') as f:
        for ID in cell_ids_cellstates:
            f.write("%s\n" % ID)

    # Store which cell-id was stored to which cs_id
    cell_id_to_cs_id_df = pd.DataFrame.from_dict(cell_id_to_cs_id, orient='index')
    cell_id_to_cs_id_df.to_csv(os.path.join(results_folder, 'cell_id_to_cs_id.csv'), header=None)

    print("Writing cell IDs to file:")
    with open(os.path.join(results_folder, 'cellID.txt'), 'w') as f:
        for ID in cs_ids:
            f.write("%s\n" % ID)

    print("Writing gene IDs to file:")
    with open(os.path.join(results_folder, 'geneID.txt'), 'w') as f:
        for ID in gene_ids:
            f.write("%s\n" % ID)

    print("Writing cellstates-annotation to file:")
    cs_annotation_dict = {}
    cs_annotation_dict['cellstates'] = cs_annot
    cs_annotation_dict['cells_in_cellstate'] = counts
    annotation_df = pd.DataFrame(cs_annotation_dict, index=cs_ids)
    Path(os.path.join(results_folder, 'annotation')).mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(os.path.join(results_folder, 'annotation', 'cs_annotation.csv'))

    cell_annotation_dict = {}
    cell_annotation_dict['cellstates'] = cell_annot
    annotation_df = pd.DataFrame(cell_annotation_dict, index=cell_ids)
    Path(os.path.join(input_folder, 'annotation')).mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(os.path.join(input_folder, 'annotation', 'cs_annotation.csv'))
