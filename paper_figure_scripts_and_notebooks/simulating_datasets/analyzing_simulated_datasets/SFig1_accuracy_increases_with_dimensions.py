import os, sys, re
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from scipy.spatial import distance
import numpy as np
import csv
import matplotlib.pyplot as plt
from natsort import natsorted
# from bonsai.bonsai_dataprocessing import get_bonsai_euclidean_distances

import logging

FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
logging.basicConfig(format=FORMAT, datefmt='%H:%M:%S',
                    level=log_level)

plt.set_loglevel(level='warning')
logging.getLogger("umap").disabled = True
logging.getLogger('numba').setLevel(logging.WARNING)

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai_scout.bonsai_scout_helpers import get_celltype_colors_new
from bonsai.bonsai_helpers import str2bool, find_latest_tree_folder
from knn_recall_helpers import get_pdists_on_tree, Dataset, do_pca, fit_umap, fit_phate, compare_pdists_to_truth, \
    fit_DTNE

parser = ArgumentParser(
    description='Runs Bonsai on several simulated datasets.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_folder', type=str, default='data/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where simulated trees can be found.")
parser.add_argument('--results_folder', type=str, default='results/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where reconstructed trees can be found.")
parser.add_argument('--num_dims', type=str, default="100",
                    help="Number of dimensions in which we sample the cells.")
parser.add_argument('--n_sampled_clsts', type=str, default="20",
                    help="Number of clusters in star-tree.")
parser.add_argument('--n_cells_per_clst', type=str, default="1,2,5,10,20,50",
                    help="Number of cells per cluster.")
parser.add_argument('--random_times', type=str2bool, default=True,
                    help="Determine if branch lengths of true tree should be sampled uniform in a logspace between"
                         "0.1 and 10, instead of keeping them all at 1.")
parser.add_argument('--sample_umi_counts', type=str2bool, default=False,
                    help="Determines if we want to ensure the tqs add up to 1. Doesn't have to when we don't sample "
                         "counts but rather give the true data to Bonsai.")
parser.add_argument('--add_noise', type=str2bool, default=True,
                    help="Determines if we want to ensure the tqs add up to 1. Doesn't have to when we don't sample "
                         "counts but rather give the true data to Bonsai.")
parser.add_argument('--seed', type=int, default=1231,
                    help="Sets the random seed.")
parser.add_argument('--noise_var', type=float, default=5.0,
                    help="Determines the variance of the Gaussian noise from which we sample cells around cluster"
                         "Typical variance from cluster-centers to mean is 1.0.")
parser.add_argument('--recalculate', type=str2bool, default=False,
                    help="Determines whether we recalculate PCAs and UMAPs.")
parser.add_argument('--n_pcs_umap', type=int, default=100,
                    help="Number of PCA components to project to before UMAP.")
parser.add_argument('--n_pcs_phate', type=int, default=100,
                    help="Number of PCA components to project to before PHATE.")
parser.add_argument('--n_pcs_dtne', type=int, default=100,
                    help="Number of PCA components to project to before DTNE.")

args = parser.parse_args()
print(args)

"""--------------------Layout settings----------------------"""
SMALL_SIZE = 9
MEDIUM_SIZE = 11
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

"""Constants"""
RECALCULATE = True
DO_OTHER_TOOLS = True
PCA_COMPS = [2]
if args.n_pcs_umap > 0:
    PCA_COMPS.append(args.n_pcs_umap)
if args.n_pcs_phate > 0:
    PCA_COMPS.append(args.n_pcs_phate)
if args.n_pcs_dtne > 0:
    PCA_COMPS.append(args.n_pcs_dtne)
PCA_COMPS = np.unique(PCA_COMPS)

seed = args.seed
num_dims_list = [int(num_dim) for num_dim in args.num_dims.split(',')]
n_cells_per_clst = 1
n_clsts = int(args.n_sampled_clsts)
n_cells = n_clsts * n_cells_per_clst

ADD_NOISE = False
noise_var = None
CELL_DEPENDENT = None

if ADD_NOISE:
    if not CELL_DEPENDENT:
        add_noise = '_add_noise_{}'.format(int(noise_var))
    else:
        add_noise = '_add_noise_{}_celldependent'.format(int(noise_var))
else:
    add_noise = ''

avg_rel_diffs = []
methods = ['bonsai', 'pca', 'umap', 'phate', 'DTNE']
figs_dict = {}
axs_dict = {}
fig, axs = plt.subplots(nrows=len(methods), ncols=len(num_dims_list), figsize=(14, 12))

base_folder = os.path.join(
    'paper_figure_scripts_and_notebooks/simulating_datasets/analyzing_simulated_datasets/results', args.input_folder)

for ind_dim, num_dims in enumerate(num_dims_list):

    datadir = "simulate_equidistant_{}_clsts_{}_cells_{}_dims".format(n_clsts, n_cells, num_dims)
    if args.random_times:
        datadir += '_random_times'
    if not args.sample_umi_counts:
        datadir += '_no_umi_counts'
    if args.add_noise:
        datadir += '_add_noise_{}'.format(int(noise_var))
    datadir += '_seed_{}'.format(seed)
    dataset = os.path.join(args.input_folder, datadir)
    data_path = os.path.abspath(os.path.join(args.input_folder, datadir))
    results_path = os.path.abspath(os.path.join(args.results_folder, datadir))

    subset_cells = np.arange(0, n_cells, n_cells_per_clst, dtype=int)
    args.input_simulated_dataset = data_path
    args.bonsai_results = os.path.join(results_path, find_latest_tree_folder(results_folder=results_path))
    args.output_folder = os.path.join(
        'paper_figure_scripts_and_notebooks/simulating_datasets/analyzing_simulated_datasets/results', dataset)
    print(args)
    Path(os.path.join(args.output_folder, 'intermediate_files')).mkdir(parents=True, exist_ok=True)

    try:
        umi_counts_df = pd.read_csv(os.path.join(args.input_simulated_dataset, 'Gene_table.txt'), header=0,
                                    index_col=0,
                                    sep='\t')
        cell_ids = list(umi_counts_df.columns)
    except FileNotFoundError:
        cell_ids = []
        with open(os.path.join(args.input_simulated_dataset, 'cellID.txt'), 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                cell_ids.append(row[0])

    # Calculate pairwise distances for ground truth
    delta_gc_true = pd.read_csv(os.path.join(args.input_simulated_dataset, 'delta_true.txt'), header=None,
                                index_col=None,
                                sep='\t')
    true_dists = distance.pdist(delta_gc_true.T, metric='sqeuclidean')/num_dims

    if DO_OTHER_TOOLS:
        # Perform PCA-projection.
        all_pca_files = [os.path.basename(filepath) for filepath in
                         Path(os.path.join(args.output_folder, 'intermediate_files')).glob('pca_*.npy')]
        all_pca_files = [filepath for filepath in all_pca_files if 'pdists' not in filepath]

        pca_projected = do_pca(delta_gc_true, n_comps_list=PCA_COMPS)

        if 'umap' in methods:
            # Perform UMAP.
            umap_projected = {}
            if args.n_pcs_umap < 0:
                preprocessed = delta_gc_true
            else:
                preprocessed = pca_projected[args.n_pcs_umap]
            umap_projected[args.n_pcs_umap] = fit_umap(preprocessed, random_state=None, n_neighbors=15, min_dist=0.1,
                                               n_components=2,
                                               metric='euclidean',
                                               make_plot=False, title='')

        if 'phate' in methods:
            # Perform PHATE.
            phate_projected = {}
            if args.n_pcs_phate < 0:
                preprocessed = delta_gc_true
            else:
                preprocessed = pca_projected[args.n_pcs_phate]
            phate_projected[args.n_pcs_phate] = fit_phate(preprocessed)

        if 'DTNE' in methods:
            # Perform DTNE.
            DTNE_projected = {}
            if args.n_pcs_dtne < 0:
                preprocessed = delta_gc_true
            else:
                preprocessed = pca_projected[args.n_pcs_dtne]
            DTNE_projected[args.n_pcs_dtne] = fit_DTNE(preprocessed)

    # Calculate pairwise distances for Bonsai.
    bonsai_dists = get_pdists_on_tree(os.path.join(args.bonsai_results, 'tree.nwk'), cell_ids)

    if DO_OTHER_TOOLS:
        # Calculate pairwise distances for 2D-PCA, UMAP
        if 'pca' in methods:
            for n_comps, pca_proj in pca_projected.items():
                if n_comps != 2:
                    continue
                pca_dists = distance.pdist(pca_proj.T, metric='sqeuclidean') / 2

        if 'umap' in methods:
            umap_proj = umap_projected[args.n_pcs_umap]
            umap_dists = distance.pdist(umap_proj.T, metric='sqeuclidean') / 2

        if 'phate' in methods:
            phate_proj = phate_projected[args.n_pcs_phate]
            phate_dists = distance.pdist(phate_proj.T, metric='sqeuclidean') / 2

        if 'DTNE' in methods:
            DTNE_proj = DTNE_projected[args.n_pcs_dtne]
            DTNE_dists = distance.pdist(DTNE_proj.T, metric='sqeuclidean') / 2

    ONE_PCAs = [False]

    datasets = []

    dist_dict = {'delta_true': true_dists, 'bonsai': bonsai_dists, 'pca': pca_dists,
                 'umap_{}'.format(args.n_pcs_umap): umap_dists, 'phate_{}'.format(args.n_pcs_phate): phate_dists,
                 'DTNE_{}'.format(args.n_pcs_dtne): DTNE_dists}
    for data_type, distances in dist_dict.items():
        data_id = data_type + ' {}_dims'.format(num_dims)
        datasets.append(
            Dataset(distances=distances, data_type=data_type, data_id=data_id,
                    color_types=['sanity', 'bonsai', 'pca', 'umap', 'phate', 'DTNE']))
        # data_families=['bonsai', 'logp1', 'pca', 'umap']))

    # Compare pairwise distances
    args.simulation_id = ''
    if len(args.simulation_id):
        title_id = ' with {}'.format(args.simulation_id)
    else:
        title_id = ''
    title_id += ' and {}'.format(num_dims)
    for ind in range(len(ONE_PCAs)):
        ONE_PCA = ONE_PCAs[ind]
        # png_filename = 'pdists_numdims_{}'.format(num_dims)
        # if ONE_PCA:
        #     png_filename += '_one_pca'
        # png_filename += '.png'
        dataset_subset = []
        for dataset in datasets:
            if ONE_PCA and (re.match(".*10.*", dataset.data_type) or re.match(".*100.*", dataset.data_type)):
                continue
            dataset_subset.append(dataset)
        data_types_unordered = [ds.data_type.split('_')[0] for ds in dataset_subset]
        index_map = {val: idx for idx, val in enumerate(methods)}
        index_map['delta'] = len(index_map)
        indices = [index_map[item] for item in data_types_unordered]
        dataset_subset_ordered = [None] * (len(methods) + 1)
        for ind_ds, ds in enumerate(dataset_subset):
            dataset_subset_ordered[indices[ind_ds]] = ds
        # dataset_subset_ordered = [dataset_subset[ind] for dataset in dataset_subset]
        dataset_subset = dataset_subset_ordered

        n_datasets = len(dataset_subset) - 1  # Minus 1 to subtract for true dataset
        n_rows = int(np.ceil(np.sqrt(n_datasets)))
        n_cols = int(np.ceil(n_datasets / n_rows))
        # fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 6))
        # true_dataset_ind = [ind for ind, dataset in enumerate(dataset_subset) if dataset.data_type == 'delta_true'][0]
        # for ind, dataset in enumerate(dataset_subset):
        #     if ind == true_dataset_ind:
        #         continue
        # method_type = dataset.data_type.split('_')[0]
        # axs = axs_dict[:, ind_dim]
        axs_col = axs[:, ind_dim]
        # ax = axs.flatten()[ind_dim]
        # avg_rel_diff_list = compare_pdists_to_truth([dataset_subset[true_dataset_ind], dataset], make_fig=True, axs=ax,
        #                                        title="kNN-recall on binary tree{}".format(title_id))
        avg_rel_diff_list = compare_pdists_to_truth(dataset_subset, make_fig=True, axs=axs_col, loglog_corr=False,
                                                    YLABEL=ind_dim == 0, first_title='{}-dim. data'.format(num_dims))
        # avg_rel_diffs.append(avg_rel_diff_list[0])
    plt.tight_layout()

    if num_dims == 2:
        delta_gc = pd.read_csv(os.path.join(args.input_simulated_dataset, 'delta_true.txt'), header=None,
                               index_col=None, sep='\t').values

        true_dists = distance.squareform(distance.pdist(delta_gc.T, metric='sqeuclidean') / num_dims)
        faraway_points = np.where(np.sum(true_dists > 10, axis=1) > 13)[0]
        from matplotlib import colormaps

        colors = np.array([colormaps.get_cmap('gray')(0.75)[:3]] * delta_gc.shape[1])
        colors_special = get_celltype_colors_new(len(faraway_points), colortype=None).colors
        colors[faraway_points, :] = np.array(colors_special)[:len(faraway_points), :3]

        fig3, ax3 = plt.subplots(ncols=len(methods))
        ax3[0].scatter(delta_gc[0, :], delta_gc[1, :], c=colors)
        ax3[0].set_title("Original 2-dimensional data")
        ax3[1].scatter(pca_projected[2][0, :], pca_projected[2][1, :],
                       c=colors)
        ax3[1].set_title("PCA-projected data")
        ax3[2].scatter(phate_projected[args.n_pcs_phate][0, :], phate_projected[args.n_pcs_phate][1, :],
                       c=colors)
        ax3[2].set_title("PHATE-embedded data")
        ax3[3].scatter(umap_projected[args.n_pcs_umap][0, :], umap_projected[args.n_pcs_umap][1, :],
                       c=colors)
        ax3[3].set_title("UMAP-embedded data")
        ax3[4].scatter(DTNE_projected[args.n_pcs_dtne][0, :], DTNE_projected[args.n_pcs_dtne][1, :],
                       c=colors)
        ax3[4].set_title("DTNE-embedded data")
        plt.tight_layout()

fig.savefig(os.path.join(base_folder, "SI_tree_better_at_high_dims.png"), dpi=300)
fig.savefig(os.path.join(base_folder, "SI_tree_better_at_high_dims.svg"))

print("Stored the png-figure at {}".format(os.path.join(base_folder, "SI_tree_better_at_high_dims.png")))

# fig, ax2 = plt.subplots()
# ax2.plot(np.array(num_dims_list), np.array(avg_rel_diffs))
# ax2.set_xlabel('Number of dimensions')
# ax2.set_ylabel('<|(Bonsai-dist - true-dist)/true_dist|>')

plt.show()
