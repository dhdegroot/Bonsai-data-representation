from argparse import ArgumentParser
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import time

import logging

FORMAT = '%(asctime)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT,
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.WARNING)  # silence all libraries

# Create your app logger
logger = logging.getLogger("myapp")
logger.setLevel(log_level)

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from bonsai.bonsai_helpers import mp_print, str2bool, read_ids
from downstream_analyses.get_clusters_max_diameter import get_annotation_based_clustering_from_nwk_str, \
    get_cluster_assignments, get_min_pdists_clustering_from_nwk_str


def calculate_NMI_for_clustering_df(cl_df, annot_dict):
    """

    :param cl_df: Pandas dataframe with as index the cell-IDs, as columns different clusterings, as entries
    the cluster-id for each cell in that clustering
    :param annot_dict: Annotation per cell-ID to which we compare the clusterings using NMI
    :return nmis: List of NMIs for each clustering in the input-dataframe
    """
    # Keep only cells present in both
    common_cells = cl_df.index.intersection(annot_dict.keys())

    if len(common_cells) != cl_df.shape[0]:
        raise ValueError("Apparently, annotated cells and cells in clustering do not match.")

    # Ground-truth labels (same for all clusterings)
    labels_true = [annot_dict[cell] for cell in common_cells]

    nmis = []

    for col in cl_df.columns:
        labels_pred = cl_df.loc[common_cells, col].values
        nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
        nmis.append(nmi)

    return nmis


def condition_label(row):
    if row["random_sampling"]:
        if row["greedy"]:
            return "First greedy"
        else:
            return "Random, allowed NMI-decrease={:.2f}%".format(100 * (1 - row['optimality_decrease']))
    else:
        return "Final greedy"


def make_annotbased_stats_figure(tracking_path, figure_path):
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(nrows=2, sharex=True)
    tracking_path = Path(tracking_path)
    df = pd.read_csv(tracking_path, sep="\t", header=0)

    # Assign condition
    df["condition"] = df.apply(condition_label, axis=1)
    conditions = df["condition"].unique()

    color_map = {
        cond: cmap(i % cmap.N)
        for i, cond in enumerate(conditions)
    }
    linewidth = 2
    alpha = 1

    for cond, g in df.groupby("condition"):
        axs[0].plot(g["n_moves"], g["norm_mut_info"], linestyle="-", linewidth=linewidth, alpha=alpha,
                    color=color_map[cond], label=cond)

    # Plot number of clusters
    for cond, g in df.groupby("condition"):
        axs[1].plot(g["n_moves"], g["n_clusters"], linestyle="--", color=color_map[cond], linewidth=linewidth,
                    alpha=alpha, label=cond)

    axs[0].set_xlabel("n_moves")
    axs[0].set_ylabel("Normalized mutual information")
    axs[1].set_ylabel("Number of clusters")
    axs[1].set_xlabel("n_moves")

    axs[0].set_title(tracking_path.name)
    axs[1].legend(loc='upper left')

    plt.savefig(figure_path, dpi=300)


def make_mindist_stats_figure(tracking_path, figure_path):
    fig, ax = plt.subplots()
    stats = pd.read_csv(tracking_path, delimiter='\t', index_col=None)
    ax.plot(stats['n_clusters'], stats['NMI'])
    ax.set_ylabel("Normalized mutual information")
    ax.set_xlabel("Number of minimal-distance clusters")
    ax.set_title(Path(tracking_path).name)
    plt.savefig(figure_path, dpi=300)


def annotbased_clustering_wrapper(nwk_str, annotation_dict, cell_ids,
                                  greedy=False, tracking_folder=None, annotation_id='author_provided_annotation',
                                  verbose=True, prohibit_small_clsts=False, cutting_tol=1e-4):
    tracking_path_annotbased = None
    if tracking_folder is not None:
        tracking_path_annotbased = os.path.join(tracking_folder,
                                                annotation_id + '_annotbased_clustering_stats.tsv')
    clustering_name = 'annot_based_clst_of_' + annotation_id
    clusters, _, mut_info = get_annotation_based_clustering_from_nwk_str(tree_nwk_str=nwk_str,
                                                                         annotation_dict=annotation_dict,
                                                                         verbose=verbose,
                                                                         random_sampling=not greedy,
                                                                         node_ids_to_clst=cell_ids,
                                                                         tracking_path=tracking_path_annotbased,
                                                                         prohibit_small_clsts=prohibit_small_clsts,
                                                                         cutting_tol=cutting_tol)
    mut_info_dict = {clustering_name: mut_info}
    cl_df_annot = get_cluster_assignments(all_clusterings={clustering_name: clusters}, node_ids_multiple_cs_ids=None)
    return cl_df_annot, mut_info_dict, tracking_path_annotbased


def mindist_clustering_wrapper(nwk_str, annotation_dict, cell_ids, tracking_folder=None,
                               annotation_id='author_provided_annotation', verbose=True):
    clustering_name = 'min_dist_clst_of_' + annotation_id
    tracking_path_mindist = None
    if tracking_folder is not None:
        tracking_path_mindist = os.path.join(tracking_folder,
                                             annotation_id + '_mindist_clustering_stats.tsv')

    # Start with first doing 100 clusters. Then calculate NMI for all, if peak is not attained, do 10 times as much
    peak_NMI_not_reached = True
    n_clusters = 100
    while peak_NMI_not_reached:
        min_dist_clusterings, _ = get_min_pdists_clustering_from_nwk_str(tree_nwk_str=nwk_str, n_clusters=n_clusters,
                                                                         cell_ids=cell_ids, node_id_to_n_cells=None,
                                                                         footfall=False, verbose=verbose)
        cl_df_mindist = get_cluster_assignments(all_clusterings=min_dist_clusterings, node_ids_multiple_cs_ids=None)
        mut_infos_list = calculate_NMI_for_clustering_df(cl_df_mindist, annotation_dict)

        n_clusters_made = len(mut_infos_list)
        best_clustering_ind = np.argmax(mut_infos_list)
        if best_clustering_ind <= (n_clusters_made - 10):
            # Best NMI was likely reached
            peak_NMI_not_reached = False
        else:
            # Best NMI may not be reached yet
            if n_clusters_made != n_clusters:
                # In this case, we cannot make more clusters, so we did reach the best NMI
                peak_NMI_not_reached = False
            n_clusters *= 10

    stats = pd.DataFrame(data={'n_clusters': np.arange(2, len(mut_infos_list) + 2), 'NMI': np.array(mut_infos_list)})
    stats.to_csv(tracking_path_mindist, index=False, header=True, sep='\t')
    mut_info = mut_infos_list[best_clustering_ind]
    mut_info_dict = {clustering_name: mut_info}
    cl_df_mindist = cl_df_mindist.iloc[:, [best_clustering_ind]]
    return cl_df_mindist, mut_info_dict, tracking_path_mindist


def do_annotbased_and_mindist_clustering(nwk_str, annotation_dict, cell_ids, results_folder=None,
                                         greedy=False, tracking_folder=None, make_plots=True,
                                         annotation_id='author_provided_annotation', verbose=True,
                                         prohibit_small_clsts=False, cutting_tol=1e-4):
    """
    Takes a tree, defined through a newick-string, and clusters subtrees in two ways:
    - such that the normalized mutual information with a given annotation is optimized
    - such that summed pairwise distances within the clusters are minimal
    We compare the normalized mutual information of these clusterings.
    :param nwk_str:
    :param annotation_dict: Dictionary from cell-ID to annotation
    :param cell_ids:
    :param results_folder: This is where clustering results and resulting NMIs are stored.
    :param greedy: Default is False, if True it is faster but leads to somewhat worse results.
    :param tracking_folder: If provided, we store some stats of the clustering
    :param make_plots: If tracking_folder is provided, we make some figures of the stats.
    :param annotation_id: Which column name from the annotation file did we pick?
    :param verbose:
    :return:
    """
    all_cl_dfs = []
    # norm_mut_infos = {}
    timings = []
    dataset_ids = []
    NMIs = []
    """Do annotation-based clustering"""
    start = time.time()
    cl_df_annot, mut_info_dict, trackingpath_annotbased = annotbased_clustering_wrapper(nwk_str, annotation_dict,
                                                                                        cell_ids,
                                                                                        greedy=greedy,
                                                                                        tracking_folder=tracking_folder,
                                                                                        annotation_id=annotation_id,
                                                                                        verbose=verbose,
                                                                                        prohibit_small_clsts=prohibit_small_clsts,
                                                                                        cutting_tol=cutting_tol)
    timings.append(time.time() - start)
    print(time.time() - start)
    all_cl_dfs.append(cl_df_annot)
    dataset_id = list(mut_info_dict.keys())[0]
    dataset_ids.append(dataset_id)
    NMIs.append(mut_info_dict[dataset_id])
    # norm_mut_infos.update(mut_info_dict)

    """Do minimal-distance based clustering"""
    start = time.time()
    cl_df_mindist, mut_info_dict, trackingpath_mindist = mindist_clustering_wrapper(nwk_str, annotation_dict, cell_ids,
                                                                                    tracking_folder=tracking_folder,
                                                                                    annotation_id=annotation_id,
                                                                                    verbose=verbose)
    timings.append(time.time() - start)
    print(time.time() - start)
    all_cl_dfs.append(cl_df_mindist)
    dataset_id = list(mut_info_dict.keys())[0]
    dataset_ids.append(dataset_id)
    NMIs.append(mut_info_dict[dataset_id])

    """Concatenate the results"""
    cl_df = pd.concat(all_cl_dfs, axis=1)

    """Store the final results"""
    norm_mut_infos_df = pd.DataFrame(
        data={'clustering_method': dataset_ids, 'max NMI': NMIs, 'times': timings})
    if results_folder is not None:
        cl_df.to_csv(os.path.join(results_folder, 'clustering_results.tsv'), sep='\t', index=True, header=True)
        # test_nmis = calculate_NMI_for_clustering_df(cl_df, annotation_dict)

        norm_mut_infos_df.to_csv(os.path.join(results_folder, 'normalized_mutual_information_scores.tsv'), sep='\t',
                                 index=False, header=True)

    if make_plots and (tracking_folder is not None):
        """Plot some stats (if available)"""
        if os.path.exists(trackingpath_annotbased):
            stats_fig_path = os.path.join(tracking_folder, 'annotbased_clustering_stats.png')
            make_annotbased_stats_figure(tracking_path=trackingpath_annotbased, figure_path=stats_fig_path)

        if os.path.exists(trackingpath_mindist):
            stats_fig_path = os.path.join(tracking_folder, 'mindist_clustering_stats.png')
            make_mindist_stats_figure(tracking_path=trackingpath_mindist, figure_path=stats_fig_path)

    return cl_df, norm_mut_infos_df


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Splits a dataset into batches, and starts Sanity runs separately per batch.')
    parser.add_argument('--nwk_file', type=str, default=None,
                        help='Absolute path to file Newick-file describing tree can be found.')
    parser.add_argument('--annotation_file', type=str, default=None,
                        help='Absolute path to tsv-file with a header and more or equal to two columns. '
                             'Header is just "cell_id annotation_type1 annotation_type2"'
                             'Column 1 contains cell-IDs, '
                             'Columns 2 to N contain an identifier of the annotation that cell has.')
    parser.add_argument('--annotation_id', type=str, default=None,
                        help='Identifier matching one of the header-entries in --annotation_file. This annotation'
                             'will be used to optimize the clustering.')
    parser.add_argument('--cell_names_file', type=str, default=None,
                        help='Absolute path to file where all cell-IDs are stored (one per line).')
    parser.add_argument('--results_folder', type=str, default=None,
                        help='Absolute path to where the clustering results should be stored.')
    parser.add_argument('--prohibit_small_clsts', type=str2bool, default=False,
                        help='Boolean determining whether we prohibit the making of many small clusters.')
    parser.add_argument('--cutting_tol', type=float, default=1e-4,
                        help='Determines how large the increase in NMI should be before a cluster is split into two.')
    parser.add_argument('--greedy', type=str2bool, default=False,
                        help='If False (default), we do an MCMC-like scheme to optimize clustering. If not, we just '
                             'greedily cut subtrees from the tree until no move improves the score.')
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='--verbose False only shows essential print messages (default: True)')

    args = parser.parse_args()
    mp_print(args)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    """Make results folder"""
    Path(args.results_folder).mkdir(parents=True, exist_ok=True)
    tracking_folder = os.path.join(args.results_folder, 'clustering_stats')
    Path(tracking_folder).mkdir(parents=True, exist_ok=True)

    """Read in the newick string"""
    with open(args.nwk_file, "r") as f:
        nwk_str = f.readline()

    """Read in the cell-Ids that should be clustered."""
    cell_ids = read_ids(args.cell_names_file)

    """Read in the annotation and create an annotation_dict"""
    if args.annotation_file.endswith('.tsv'):
        sep = '\t'
    elif args.annotation_file.endswith('.csv'):
        sep = ','
    else:
        exit("Can't read this annotation-file, it's neither .tsv nor .csv.")
    annotation_df = pd.read_csv(args.annotation_file, header=0, index_col=0, sep=sep)
    annotation_dict = {}
    for annot_row in annotation_df.itertuples():
        annotation_dict[annot_row.Index] = getattr(annot_row, args.annotation_id)
    del annotation_df

    cl_df_concat, NMI_df = do_annotbased_and_mindist_clustering(nwk_str, annotation_dict, cell_ids,
                                                                  results_folder=args.results_folder,
                                                                  greedy=args.greedy, tracking_folder=None,
                                                                  make_plots=True, verbose=True,
                                                                  annotation_id=args.annotation_id,
                                                                  prohibit_small_clsts=args.prohibit_small_clsts,
                                                                  cutting_tol=args.cutting_tol)
    print(cl_df_concat)
    print(NMI_df)
