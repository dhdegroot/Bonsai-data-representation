from argparse import ArgumentParser
import pandas as pd
import os
import numpy as np
import shutil
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
import csv
from pathlib import Path

import logging

FORMAT = '%(asctime)s %(name)s %(funcName)s %(message)s'
log_level = logging.WARNING
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT, datefmt='%m-%d %H:%M:%S',
                    level=log_level)


def merge_clusters_acc_cellstates(clusters, umiCounts, cell_ids, verbose=True):
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
    clsts, cell_ind_to_cs_ind, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    n_clst = len(clsts)

    cs_ids = ['cs_' + str(ind) for ind in range(n_clst)]

    cs_annot = []
    for ind, cs_count in enumerate(counts):
        if cs_count > CUTOFF:
            cs_annot.append('cs_{}'.format(ind))
        else:
            cs_annot.append('cs<{}'.format(CUTOFF))

    cell_id_to_cs_id = {}
    cell_annot = []
    for cell_ind, cs_ind in enumerate(cell_ind_to_cs_ind):
        cell_id_to_cs_id[cell_ids[cell_ind]] = cs_ids[cs_ind]
        if counts[cs_ind] > CUTOFF:
            cell_annot.append('cs_{}'.format(cs_ind))
        else:
            cell_annot.append('cs<{}'.format(CUTOFF))

    summed_counts_clst = np.zeros((umiCounts.shape[0], n_clst), dtype=int)
    for i, clstName in enumerate(clsts):
        if (i % 10) == 0:
            logging.debug("Merging cellstate {}".format(i))
        clusterMask = clusters == clstName
        if np.sum(clusterMask) > 0:
            summed_counts_clst[:, i] = np.sum(umiCounts[:, clusterMask], axis=1)
        else:
            logging.warning("How can this mask be empty?")
    return summed_counts_clst, counts, cs_annot, cell_annot, cs_ids, cell_id_to_cs_id


if __name__ == '__main__':
    """Parse input arguments"""
    parser = ArgumentParser(
        description='Sums UMI-counts for all cells in same cellstate. Stores resulting "super-cells" as Sanity input.')
    parser.add_argument('--folder_cellstates_output', type=str, default='.',
                        help='Path to the folder where the cellstates output can be found')
    parser.add_argument('--file_raw_umi_counts', type=str, default='',
                        help='Path to the file where UMI-counts can be found')
    parser.add_argument('--file_cell_ids', type=str, default=None,
                        help='Path to file where cell-ids can be found')
    parser.add_argument('--file_gene_ids', type=str, default=None,
                        help='Path to file where gene-ids can be found')
    parser.add_argument('--folder_clustered_umi_counts', type=str, default=None,
                        help='Folder where clustered umi counts should be stored.')
    parser.add_argument('--cutoff', type=int, default=5,
                        help='Determines size of cellstates that get annotated as "small".')

    args = parser.parse_args()

    AS_MTX = True

    CUTOFF = args.cutoff
    results_folder = args.folder_clustered_umi_counts
    input_folder = os.path.dirname(os.path.abspath(args.file_raw_umi_counts))
    if results_folder is None:
        results_folder = os.path.join(input_folder, 'cs_merged')
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    clustering = pd.read_csv(os.path.join(args.folder_cellstates_output, 'optimized_clusters.txt'), sep='\t',
                             header=None).values.astype(dtype=int).flatten()
    cell_ids_cellstates = pd.read_csv(os.path.join(args.folder_cellstates_output, 'CellID.txt'), sep='\t',
                                      header=None).values.flatten()

    if args.file_raw_umi_counts.split('.')[1] == 'mtx':
        M = mmread(args.file_raw_umi_counts)
        umi_counts = M.toarray().astype(dtype=int)
        # Read in promoter names
        gene_ids = []
        with open(os.path.join(args.file_gene_ids), 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                gene_ids.append(row[0])

        # Read in cell barcodes as in mtx-file
        cell_ids = []
        with open(args.file_cell_ids, 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                cell_ids.append(row[0])
    else:
        tmp = pd.read_csv(args.file_raw_umi_counts, sep='\t', index_col=0)
        cell_ids = list(tmp.columns)
        gene_ids = list(tmp.index)
        umi_counts = tmp.values.astype(dtype='int')

    logging.debug("First 10 cell ids, raw input: {}".format(cell_ids[:10]))
    logging.debug("First 10 cell ids, cellstates input: {}".format(cell_ids_cellstates[:10]))

    for ind, cell_ID in enumerate(cell_ids):
        if cell_ID != cell_ids_cellstates[ind]:
            print("Cell IDs do not match between cellstates-output and raw data. Quitting conversion.")
            exit()

    summed_counts_clst, counts, cs_annot, cell_annot, cs_ids, cell_id_to_cs_id = merge_clusters_acc_cellstates(
        clustering, umi_counts, cell_ids=cell_ids_cellstates)
    n_cellstates = summed_counts_clst.shape[1]

    if not AS_MTX:
        new_umi_counts_df = pd.DataFrame(summed_counts_clst, columns=cs_ids, index=gene_ids)
        new_umi_counts_df.to_csv(os.path.join(results_folder, 'Gene_table.txt'), index=True, sep='\t', index_label="GeneID")
    else:
        sparse_umis = csr_matrix(summed_counts_clst)
        mmwrite(os.path.join(results_folder, 'prom_cs_expr_matrix.mtx'), sparse_umis)
    shutil.copyfile(os.path.join(args.folder_cellstates_output, 'optimized_clusters.txt'),
                    os.path.join(results_folder, 'cs_clusters.txt'))
    shutil.copyfile(os.path.join(args.folder_cellstates_output, 'CellID.txt'),
                    os.path.join(results_folder, 'orig_CellID.txt'))

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
