from argparse import ArgumentParser
import os
import sys
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm


def store_ds_info(tree_node, vert_ind_to_node, leaf_ltqs, leaf_ind):
    if tree_node.isLeaf:
        leaf_ltqs.append(tree_node.ltqs)
        tree_node.ds_leaf_inds = [leaf_ind]
        leaf_ind += 1
    else:
        tree_node.ds_leaf_inds = []
    vert_ind_to_node[tree_node.vert_ind] = tree_node
    for child in tree_node.childNodes:
        vert_ind_to_node, leaf_ltqs, leaf_ind = store_ds_info(child, vert_ind_to_node, leaf_ltqs, leaf_ind)
        tree_node.ds_leaf_inds += child.ds_leaf_inds
    return vert_ind_to_node, leaf_ltqs, leaf_ind


def calc_marker_genes(tree, gene_ids, min_marker_genes=10, marker_cutoff=.9, verbose=True):
    if tree.root.childNodes[0].parentNode is None:
        tree.root.storeParent()

    # Create array of ltqs on leafs
    vert_ind_to_node, leaf_ltqs, n_leafs = store_ds_info(tree.root, vert_ind_to_node={}, leaf_ltqs=[], leaf_ind=0)
    leaf_ltqs = np.array(leaf_ltqs).T

    # Argsort for all genes
    leaf_ranks_per_gene = np.argsort(np.argsort(leaf_ltqs, axis=1), axis=1)

    marker_genes_dict = {}
    for vert_ind, node in vert_ind_to_node.items():
        if verbose and (vert_ind % 1000 == 0):
            print("Finding marker genes for node {}".format(vert_ind))
        if node.isRoot:
            continue
        marker_genes = calc_marker_genes_single(node.ds_leaf_inds, n_leafs, gene_ids, leaf_ranks_per_gene,
                                                min_marker_genes=min_marker_genes, marker_cutoff=marker_cutoff)
        marker_genes_dict[(node.parentNode.vert_ind, vert_ind)] = marker_genes

    return marker_genes_dict


def calc_marker_genes_single(ds_leaf_inds, n_leafs, gene_ids, leaf_ranks_per_gene, gene_subset=None,
                             min_marker_genes=10, marker_cutoff=.9):
    """

    :param ds_leaf_inds:
    :param n_leafs:
    :param gene_ids:
    :param leaf_ranks_per_gene: This should be a matrix with dimensions n_genes x n_leafs giving for each gene the rank
    that the leaf would get if we would sort the leafs by that gene.
    Can be obtained using: np.argsort(np.argsort(leaf_ltqs, axis=1), axis=1)
    :return:
    """
    # for each node go over all genes and calculate the marker-score
    # As marker-score for a gene g to distinguish clusters C1,C2, we use the probability that an arbitrary cell
    # from C1 has lower gene expression for g than an arbitrary cell from C2:
    # 1/(|C1||C2|) sum_{i\in C1, j\in C2} \Theta(ltqs_gj - ltqs_gi)
    # = (C1 + 2*C2 - 1)/(2*C2) - 1/(C1 * C2) * sum_{i\in C1} rank(i)
    # where Theta is the Heaviside function that is 1 when ltqs_gj > ltqs_gi and 0 otherwise
    card_C1 = len(ds_leaf_inds)
    card_C2 = n_leafs - card_C1
    if gene_subset is not None:
        summed_ranks = np.sum(leaf_ranks_per_gene[:, ds_leaf_inds][gene_subset,:], axis=1)
    else:
        summed_ranks = np.sum(leaf_ranks_per_gene[:, ds_leaf_inds], axis=1)
    # marker_scores = 1 + (card_C1 - 1) / (2 * card_C2) - summed_ranks / (card_C1 * card_C2)
    marker_scores = - (card_C1 - 1) / (2 * card_C2) + summed_ranks / (card_C1 * card_C2)

    if len(marker_scores) < 2 * min_marker_genes:
        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in range(len(marker_scores))}
    else:
        top_high_genes = list(np.where(marker_scores > marker_cutoff)[0])
        if len(top_high_genes) < min_marker_genes:
            top_high_genes = list(np.argpartition(marker_scores, min_marker_genes)[:min_marker_genes - 1])

        top_low_genes = list(np.where(marker_scores < (1 - marker_cutoff))[0])
        if len(top_low_genes) < min_marker_genes:
            top_low_genes = list(np.argpartition(-marker_scores, min_marker_genes)[:min_marker_genes - 1])

        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}

    # top_high_genes = list(np.argpartition(marker_scores, n_marker_genes)[:n_marker_genes-1])
    # top_low_genes = list(np.argpartition(-marker_scores, n_marker_genes)[:n_marker_genes-1])
    # marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}
    if gene_subset is not None:
        marker_genes = {gene_subset[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    marker_genes = {gene_ids[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    return marker_genes


def calc_marker_genes_double(ds_leaf_inds_1, ds_leaf_inds_2, n_leafs, gene_ids, leaf_ranks_per_gene, gene_subset=None,
                             min_marker_genes=10, marker_cutoff=0.9):
    """

    :param ds_leaf_inds:
    :param n_leafs:
    :param gene_ids:
    :param leaf_ranks_per_gene: This should be a matrix with dimensions n_genes x n_leafs giving for each gene the rank
    that the leaf would get if we would sort the leafs by that gene.
    Can be obtained using: np.argsort(np.argsort(leaf_ltqs, axis=1), axis=1)
    :return:
    """
    # for each node go over all genes and calculate the marker-score
    # As marker-score for a gene g to distinguish clusters C1,C2, we use the probability that an arbitrary cell
    # from C1 has lower gene expression for g than an arbitrary cell from C2:
    # 1/(|C1||C2|) sum_{i\in C1, j\in C2} \Theta(ltqs_gj - ltqs_gi)
    # = (C1 + 2*C2 - 1)/(2*C2) - 1/(C1 * C2) * sum_{i\in C1} rank(i)
    # where Theta is the Heaviside function that is 1 when ltqs_gj > ltqs_gi and 0 otherwise
    card_C1 = len(ds_leaf_inds_1)
    card_C2 = len(ds_leaf_inds_2)

    # Adjust leaf_ranks_per_gene to take only leafs that are in subset 1 or 2
    leaf_ranks_subset = np.concatenate((leaf_ranks_per_gene[:, ds_leaf_inds_1], leaf_ranks_per_gene[:, ds_leaf_inds_2]), axis=1)
    ranks_per_gene_subset = np.argsort(np.argsort(leaf_ranks_subset, axis=1), axis=1)

    if gene_subset is not None:
        summed_ranks = np.sum(ranks_per_gene_subset[:, :card_C1][gene_subset, :], axis=1)
    else:
        summed_ranks = np.sum(ranks_per_gene_subset[:, :card_C1], axis=1)

    # marker_scores1 = 1 + (card_C1 - 1) / (2 * card_C2) - summed_ranks / (card_C1 * card_C2)
    marker_scores = - (card_C1 - 1) / (2 * card_C2) + summed_ranks / (card_C1 * card_C2)
    # print(marker_scores1[:10] + marker_scores2[:10])
    if len(marker_scores) < 2 * min_marker_genes:
        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in range(len(marker_scores))}
    else:
        top_high_genes = list(np.where(marker_scores > marker_cutoff)[0])
        if len(top_high_genes) < min_marker_genes:
            top_high_genes = list(np.argpartition(marker_scores, min_marker_genes)[:min_marker_genes - 1])

        top_low_genes = list(np.where(marker_scores < (1 - marker_cutoff))[0])
        if len(top_low_genes) < min_marker_genes:
            top_low_genes = list(np.argpartition(-marker_scores, min_marker_genes)[:min_marker_genes - 1])

        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}

    # top_high_genes = list(np.argpartition(marker_scores, n_marker_genes)[:n_marker_genes-1])
    # top_low_genes = list(np.argpartition(-marker_scores, n_marker_genes)[:n_marker_genes-1])
    # marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}
    if gene_subset is not None:
        marker_genes = {gene_subset[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    marker_genes = {gene_ids[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    return marker_genes


def calc_marker_genes_error_bars(indices1, indices2, means, vars, gene_ids=None, min_marker_genes=10,
                                 marker_cutoff=.9, n_cells_per_object=None):
    a_mu = means.T
    a_var = vars.T

    # Slice relevant portions of matrices
    a1_mu = a_mu[indices1, :]
    a2_mu = a_mu[indices2, :]

    a1_var = a_var[indices1, :]
    a2_var = a_var[indices2, :]

    if n_cells_per_object is not None:
        ncells1 = n_cells_per_object[indices1]
        ncells2 = n_cells_per_object[indices2]
    else:
        ncells1 = np.ones(len(indices1))
        ncells2 = np.ones(len(indices2))

    num_cells1, num_genes = a1_mu.shape
    num_cells2 = a2_mu.shape[0]

    # Initialize total weighted sum and total weight
    weighted_sum = np.zeros(num_genes)
    total_weight = 0.0

    # Process in chunks to reduce memory usage
    print_i = 10
    for i in range(num_cells1):
        if i == print_i:
            print("Calculating marker genes is at step {} out of {}".format(i, num_cells1))
            print_i *= 2
        diff = a1_mu[i, :] - a2_mu  # Compute diff for current row
        sig_sum_sq = a1_var[i, :] + a2_var  # Compute variance sum for current row

        z_matrix = diff / np.sqrt(sig_sum_sq)  # Compute z-scores
        prob_matrix = ndtr(z_matrix)  # Compute probabilities

        # Compute weight for current row
        weight_row = ncells1[i] * ncells2

        # Accumulate weighted sum and total weight
        weighted_sum += np.sum(prob_matrix * weight_row[:, None], axis=0)
        total_weight += np.sum(weight_row)

    # Normalize scores
    marker_scores = weighted_sum / total_weight

    if len(marker_scores) < 2 * min_marker_genes:
        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in range(len(marker_scores))}
    else:
        top_high_genes = list(np.where(marker_scores > marker_cutoff)[0])
        if len(top_high_genes) < min_marker_genes:
            top_high_genes = list(np.argpartition(marker_scores, min_marker_genes)[:min_marker_genes - 1])

        top_low_genes = list(np.where(marker_scores < (1 - marker_cutoff))[0])
        if len(top_low_genes) < min_marker_genes:
            top_low_genes = list(np.argpartition(-marker_scores, min_marker_genes)[:min_marker_genes - 1])

        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}

    # top_high_genes = list(np.argpartition(marker_scores, n_marker_genes)[:n_marker_genes-1])
    # top_low_genes = list(np.argpartition(-marker_scores, n_marker_genes)[:n_marker_genes-1])
    # marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}

    # top_high_genes = list(np.argpartition(marker_scores, n_marker_genes)[:n_marker_genes - 1])
    # top_low_genes = list(np.argpartition(-marker_scores, n_marker_genes)[:n_marker_genes - 1])
    # marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}
    # if gene_subset is not None:
    #     marker_genes = {gene_subset[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    if gene_ids is not None:
        marker_genes = {gene_ids[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    else:
        marker_genes = {'Feature_{}'.format(gene_ind): marker_score for gene_ind, marker_score in marker_genes.items()}
    return marker_genes


def calc_marker_genes_error_bars_approx(indices1, indices2, means, vars, gene_ids=None, min_marker_genes=10,
                                        marker_cutoff=.9, n_cells_per_object=None):
    """
    Gets marker genes. Probability per gene that the gene is higher expressed in c_1 than in c_2, when we take a random
    cell from C_1, and a random cell from C_2.
    :param indices1:
    :param indices2:
    :param means:
    :param vars:
    :param gene_ids:
    :param n_cells_per_object:
    :return:
    """
    # TODO: Account for n_cells_per_object

    means_g1 = means[:, indices1]  # shape: (n_genes, n_cells1)
    std_g1 = np.sqrt(vars[:, indices1])
    means_g2 = means[:, indices2]  # shape: (n_genes, n_cells2)
    std_g2 = np.sqrt(vars[:, indices2])
    num_cells1 = len(indices1)
    num_cells2 = len(indices2)

    n_genes = means.shape[0]
    marker_scores = np.zeros(n_genes)

    print_i = 100
    for gene in range(n_genes):
        if gene == print_i:
            print("Calculating marker genes is at gene {} out of {}".format(gene, n_genes))
            print_i *= 2
        m1 = means_g1[gene]  # shape: (n_cells1,)
        s1 = std_g1[gene]
        m2 = means_g2[gene]  # shape: (n_cells2,)
        s2 = std_g2[gene]

        lb = min(np.min(m1 - 2 * s1), np.min(m2 - 2 * s2))
        ub = max(np.max(m1 + 2 * s1), np.max(m2 + 2 * s2))
        grid = np.linspace(lb, ub, 1000)
        dx = grid[1] - grid[0]

        # Vectorized across grid and cells
        # Shape: (1000, n_cells1)
        pdfs_g1 = norm.pdf(grid[:, None], loc=m1[None, :], scale=s1[None, :])
        pdf_g1 = pdfs_g1.sum(axis=1)

        cdfs_g2 = norm.cdf(grid[:, None], loc=m2[None, :], scale=s2[None, :])
        cdf_g2 = cdfs_g2.sum(axis=1)

        marker_scores[gene] = np.dot(pdf_g1, cdf_g2) * dx / (num_cells1 * num_cells2)

    if len(marker_scores) < 2 * min_marker_genes:
        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in range(len(marker_scores))}
    else:
        top_high_genes = list(np.where(marker_scores > marker_cutoff)[0])
        if len(top_high_genes) < min_marker_genes:
            top_high_genes = list(np.argpartition(marker_scores, min_marker_genes)[:min_marker_genes - 1])

        top_low_genes = list(np.where(marker_scores < (1 - marker_cutoff))[0])
        if len(top_low_genes) < min_marker_genes:
            top_low_genes = list(np.argpartition(-marker_scores, min_marker_genes)[:min_marker_genes - 1])

        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}

    # top_high_genes = list(np.argpartition(marker_scores, n_marker_genes)[:n_marker_genes - 1])
    # top_low_genes = list(np.argpartition(-marker_scores, n_marker_genes)[:n_marker_genes - 1])
    # marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}
    # if gene_subset is not None:
    #     marker_genes = {gene_subset[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    if gene_ids is not None:
        marker_genes = {gene_ids[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    else:
        marker_genes = {'Feature_{}'.format(gene_ind): marker_score for gene_ind, marker_score in marker_genes.items()}
    return marker_genes


def calc_marker_genes_error_bars_approx2(indices1, indices2, means, vars, gene_ids=None, min_marker_genes=10,
                                         marker_cutoff=.9, n_cells_per_object=None, n_points_total=None):
    """
    Gets marker genes. Probability per gene that the gene is higher expressed in c_1 than in c_2, when we take a random
    cell from C_1, and a random cell from C_2.
    :param indices1:
    :param indices2:
    :param means:
    :param vars:
    :param gene_ids:
    :param n_cells_per_object:
    :return:
    """
    # TODO: Account for n_cells_per_object
    if n_points_total is None:
        n_points_total = 100

    try:
        means_g1 = means[:, indices1]
        std_g1 = np.sqrt(vars[:, indices1])
        means_g2 = means[:, indices2]
        std_g2 = np.sqrt(vars[:, indices2])
    except TypeError:  # Can occur when there is duplicates in the indices-list (in case of cells mapping to same vert)
        means_g1 = means[:][:, indices1]
        std_g1 = np.sqrt(vars[:][:, indices1])
        means_g2 = means[:][:, indices2]
        std_g2 = np.sqrt(vars[:][:, indices2])
    num_genes, num_cells1 = means_g1.shape
    _, num_cells2 = means_g2.shape

    marker_scores = np.zeros(means_g1.shape[0])

    lbs = np.min(means_g1 - 2 * std_g1, axis=1)
    ubs = np.max(means_g1 + 2 * std_g1, axis=1)

    # Sort g1
    argsort_g1 = np.argsort(means_g1, axis=1)
    row_idx = np.arange(num_genes)[:, None]
    means_g1 = means_g1[row_idx, argsort_g1]
    std_g1 = std_g1[row_idx, argsort_g1]

    n_points_center = int(.9 * n_points_total)
    n_points_sides = int(.05 * n_points_total)

    prod_denom = 2 * num_cells1 * num_cells2

    print_i = 100
    n_genes = means_g1.shape[0]
    for gene in range(n_genes):
        if gene == print_i:
            print("Calculating marker genes is at gene {} out of {}".format(gene, n_genes))
            print_i *= 2
        lb = lbs[gene]
        ub = ubs[gene]

        m1 = means_g1[gene]
        s1 = std_g1[gene]
        m2 = means_g2[gene]
        s2 = std_g2[gene]

        # Choose an optimal grid by taking it uniform in the cumulative distribution of group 1
        grid = np.interp(np.linspace(0,num_cells1-1,n_points_center), np.arange(num_cells1), m1)
        left_grid = np.linspace(lb, m1[0], n_points_sides + 1)[:-1]
        right_grid = np.linspace(m1[-1], ub, n_points_sides + 1)[1:]
        grid = np.concatenate((left_grid, grid, right_grid))
        # dx = np.diff(grid)

        cdf_g1 = np.sum(norm.cdf(grid[:, None], loc=m1[None, :], scale=s1[None, :]), axis=1)
        cdf_g2 = np.sum(norm.cdf(grid[:, None], loc=m2[None, :], scale=s2[None, :]), axis=1)

        # cdf_g1_old = np.zeros_like(grid)
        # cdf_g2_old = np.zeros_like(grid)
        # for ind, val in enumerate(grid):
        #     cdf_g1_old[ind] = np.sum(norm.cdf(val, loc=m1, scale=s1))
        #     cdf_g2_old[ind] = np.sum(norm.cdf(val, loc=m2, scale=s2))

        # Get the marker scores by multiplying the increases in the CDF of group 1 (to approx. PDF) and CDF of group 2
        # Then sum over that
        marker_scores[gene] = np.dot(np.diff(cdf_g1), cdf_g2[:-1] + cdf_g2[1:]) / prod_denom

        # tested = (np.sum(pdf_g1) * dx)/num_cells1
        # marker_scores[gene] = np.dot(pdf_g1, cdf_g2) * dx / (num_cells1 * num_cells2)

    if len(marker_scores) < 2 * min_marker_genes:
        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in range(len(marker_scores))}
    else:
        top_high_genes = list(np.where(marker_scores > marker_cutoff)[0])
        if len(top_high_genes) < min_marker_genes:
            top_high_genes = list(np.argpartition(marker_scores, min_marker_genes)[:min_marker_genes - 1])

        top_low_genes = list(np.where(marker_scores < (1 - marker_cutoff))[0])
        if len(top_low_genes) < min_marker_genes:
            top_low_genes = list(np.argpartition(-marker_scores, min_marker_genes)[:min_marker_genes - 1])

        marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}

    # if n_marker_genes is not None:
    #     top_high_genes = list(np.argpartition(marker_scores, n_marker_genes)[:n_marker_genes - 1])
    #     top_low_genes = list(np.argpartition(-marker_scores, n_marker_genes)[:n_marker_genes - 1])
    #     marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in top_high_genes + top_low_genes}
    # else:
    #     marker_genes = {gene_ind: marker_scores[gene_ind] for gene_ind in range(len(marker_scores))}

    # if gene_subset is not None:
    #     marker_genes = {gene_subset[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    if gene_ids is not None:
        marker_genes = {gene_ids[gene_ind]: marker_score for gene_ind, marker_score in marker_genes.items()}
    else:
        marker_genes = {'Feature_{}'.format(gene_ind): marker_score for gene_ind, marker_score in marker_genes.items()}
    return marker_genes


def store_marker_genes(filepath, marker_genes_dict):
    import pandas as pd
    single_dfs = []
    n_markers = None
    for edge_id, marker_genes in marker_genes_dict.items():
        single_df = pd.DataFrame.from_dict(marker_genes, orient='index', columns=['marker_scores'])
        single_df.reset_index(inplace=True)
        if n_markers is None:
            n_markers = single_df.shape[0]
        single_df['edge_id'] = [edge_id] * n_markers
        single_dfs.append(single_df)
    marker_genes_df = pd.concat(single_dfs, ignore_index=True)
    marker_genes_df.rename({'edge_id': 'edge_id', 'marker_scores': 'marker_scores', 'index': 'marker_genes'}, axis=1,
                           inplace=True)
    marker_genes_df.to_csv(filepath, index=False)


def load_marker_genes(filepath):
    import pandas as pd
    marker_genes_df = pd.read_csv(filepath, header=0, index_col=None)
    edge_ids = np.unique(marker_genes_df['edge_id'].values)
    marker_genes_dict = {}
    for edge_id in edge_ids:
        single_df = marker_genes_df[marker_genes_df['edge_id'] == edge_id]
        marker_genes_dict[eval(edge_id)] = {row[1]['marker_genes']: row[1]['marker_scores'] for row in
                                            single_df.iterrows()}
    return marker_genes_dict


def get_marker_genes(filepath=None, store=True, recalc=False, tree=None, gene_ids=None, min_marker_genes=10,
                     marker_cutoff=.9, verbose=True):
    if (filepath is not None) and (os.path.exists(filepath)) and not recalc:
        marker_genes_dict = load_marker_genes(filepath)
    elif (tree is not None) and (gene_ids is not None):
        marker_genes_dict = calc_marker_genes(tree=tree, gene_ids=gene_ids, min_marker_genes=min_marker_genes,
                                              marker_cutoff=marker_cutoff, verbose=verbose)
        if store:
            store_marker_genes(filepath, marker_genes_dict)
    else:
        print("Cannot load marker genes, and also don't have enough information to calculate marker genes.")
        return None
    return marker_genes_dict


if __name__ == '__main__':
    from argparse import ArgumentTypeError
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')

    parser = ArgumentParser(
        description='Starts from a reconstructed tree output by Bonsai and creates a data-object necessary for further '
                    'visualization and usage in the Bonsai-Shiny app.')

    parser.add_argument('--path_to_code', type=str, default='..',
                        help='Absolute path to folder where code can be found, should point to "bonsai-development".')

    parser.add_argument('--dataset', type=str, default='',
                        help='Name of dataset. This will determine name of results-folder where information is stored.')
    parser.add_argument('--data_folder', type=str, default='data/Zeisel',
                        help='path to folder where input data can be found. This folder should contain a file with '
                             'means and standard-deviations in files "delta.txt" and "d_delta.txt" unless argument '
                             'filenames_data changes this behaviour.')
    # Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity
    # or not
    parser.add_argument('--results_folder', type=str, default=None,
                        help='path to folder where results will be stored.')

    parser.add_argument('--tree_folder', type=str, default='',
                        help='Path to folder that determines tree topology. Should contain edgeInfo and vertInfo')

    parser.add_argument('--filenames_data', type=str, default='delta.txt,d_delta.txt',
                        help='Filenames of input-files for means and standard deviations separated by a comma. '
                             'These files should have different cells in the columns, and '
                             'features (like gene expression quotients) as rows.')
    # Arguments that determine running configurations of bonsai. How much is printed, which steps are run?
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='--verbose False only shows essential print messages (default: True)')

    args = parser.parse_args()
    print(args)
    os.chdir(args.path_to_code)
    from bonsai.bonsai_dataprocessing import loadReconstructedTreeAndData

    # args.very_small_errorbars = False
    args.input_is_sanity_output = False
    args.rescale_by_var = False
    args.zscore_cutoff = -1

    """In the following lines the tree is read in from the output generated by Bonsai"""
    # We use the tree-object that is also used for the Bonsai-reconstruction since it can nicely work with coordinates
    scData, _ = loadReconstructedTreeAndData(args, args.tree_folder, reprocess_data=True,
                                             get_cell_info=True, all_ranks=False,
                                             corrected_data=False, all_genes=False, get_data=True,
                                             rel_to_results=False, no_data_needed=True, calc_loglik=False)

    """Below you can do your postprocessing, change the tree, if you want."""
    # The following sets for every node on the tree, how many marker genes will be identified that are high expressed
    # on the downstream leafs, and as many marker genes for all other leafs
    min_marker_genes = 5
    get_marker_genes(os.path.join(scData.result_path(), 'marker_genes.csv'), store=True, recalc=False, tree=scData.tree,
                     gene_ids=scData.metadata.geneIds, min_marker_genes=5, marker_cutoff=.9)

    # marker_genes_dict = calc_marker_genes(scData.tree, scData.metadata.geneIds, n_marker_genes=n_marker_genes)
    #
    # store_marker_genes(os.path.join(scData.result_path(), 'marker_genes.csv'), marker_genes_dict)
    # marker_genes_dict2 = load_marker_genes(os.path.join(scData.result_path(), 'marker_genes.csv'))

"""If you want to store a tree that you have altered, you use the following command"""
# scData.storeTreeInFolder(new_tree_folder, with_coords=False, verbose=False, cleanup_tree=True)
