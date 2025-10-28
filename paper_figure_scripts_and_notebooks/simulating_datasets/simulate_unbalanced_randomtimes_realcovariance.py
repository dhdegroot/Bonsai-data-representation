import numpy as np
import os
import sys
import h5py
from pathlib import Path
import pandas as pd
from scipy.special import logsumexp
from argparse import ArgumentParser
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

from bonsai.bonsai_treeHelpers import Tree, TreeNode
from bonsai.bonsai_dataprocessing import SCData
from bonsai.bonsai_helpers import str2bool
import bonsai.bonsai_globals as bs_glob

parser = ArgumentParser(
    description='Simulates a binary tree in a lower-dimensional space.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--results_folder', type=str, default='data/simulated_datasets',
                    help="Relative path from bonsai_development to base-folder where simulated tree needs to be stored.")
parser.add_argument('--num_gens', type=int, default=10,
                    help="Number of generations in binary tree.")
parser.add_argument('--data_path', type=str, default='data/Baron_subset_1024cells',
                    help='Path to real dataset, used for getting realistic covariance matrix. Should contain subdir'
                         'with Sanity-output')
parser.add_argument('--random_times', type=str2bool, default=False,
                    help='Determines whether we simulate using random branch lengths, or all equal to 1.')
parser.add_argument('--unbalanced', type=str2bool, default=False,
                    help='Determines whether we simulate an unbalanced tree, instead of a perfect binary tree.')
parser.add_argument('--realcovariance', type=str2bool, default=False,
                    help='Determines whether we reconstruct a realistic covariance structure')
parser.add_argument('--multiply_cell_counts', type=float, default=-1,
                    help='To create a dataset with more or less noise, we can multiply the cell counts by a number'
                         'smaller or larger than 1, respectively.')

args = parser.parse_args()
print(args)

seed = 2462
np.random.seed(seed)

# Set number of generations and diffusion times
N_GENERATIONS = args.num_gens
RANDOM_TIMES = args.random_times
UNBALANCED = args.unbalanced
REAL_COVARIANCE = args.realcovariance
n_cells_simu = 2 ** N_GENERATIONS

if REAL_COVARIANCE:
    args.sanity_output_path = os.path.join(args.data_path, "Sanity")
    # Read in Sanity-output on real dataset
    umi_counts = pd.read_csv(os.path.join(args.data_path, 'Gene_table.txt'), header=0,
                            index_col=0, sep='\t').values.astype(dtype=float)
    N_c = np.sum(umi_counts, axis=0)
    N_g = np.sum(umi_counts, axis=1)
    nGenes_orig, nCells_orig = umi_counts.shape

    # Take a subset of the cells such that we have exactly 2 ** N_GENERATIONS cells
    subset_cells = np.random.choice(nCells_orig, 2 ** N_GENERATIONS, replace=False)

    mean_ltqs = pd.read_csv(os.path.join(args.sanity_output_path, 'mu.txt'), header=None,
                            index_col=None, sep='\t').values.astype(dtype=float).flatten()
    real_deltas = pd.read_csv(os.path.join(args.sanity_output_path, 'delta.txt'), header=None,
                              index_col=None, sep='\t').values.astype(dtype=float)[:, subset_cells]
    real_d_deltas = pd.read_csv(os.path.join(args.sanity_output_path, 'd_delta.txt'), header=None,
                                index_col=None, sep='\t').values.astype(dtype=float)[:, subset_cells]
    real_variances = pd.read_csv(os.path.join(args.sanity_output_path, 'variance.txt'), header=None,
                                 index_col=None, sep='\t').values.astype(dtype=float).flatten()
    gene_ids = []

    nGenes_orig, nCells_orig = real_deltas.shape

    # We sample the ltqs from the posterior distribution on the ltqs to get a realistic dataset, taking only the means
    # would underestimate the variance in the dataset
    deviations = np.random.normal(0, 1, size=real_deltas.shape) * real_d_deltas
    real_ltqs = mean_ltqs[:, None] + real_deltas + deviations

    # Center the matrix and take SVD
    mean_ltqs = real_ltqs.mean(axis=1)
    real_ltqs_cd = real_ltqs - mean_ltqs[:, None]
    U_gk, S_k, Vh_kc = np.linalg.svd(real_ltqs_cd, full_matrices=False)
    n_genes_simu = len(S_k)
    gene_variances = np.ones(n_genes_simu)
else:
    # baron_hdf = h5py.File(args.input_dataset, 'r')
    baron_hdf = h5py.File('examples/example_data/baron.hdf', 'r')
    N_c = baron_hdf['N_c'][:]
    nGenes_orig = baron_hdf.attrs['nGenes']
    nCells_orig = baron_hdf.attrs['nCells']
    N_g = baron_hdf['N_g'][:]
    baron_hdf.close()

    n_genes_simu = nGenes_orig

if RANDOM_TIMES:
    lbTime = 0.5
    ubTime = 2
    randTimes = np.exp(np.random.uniform(np.log(lbTime), np.log(ubTime), size=2 ** (N_GENERATIONS + 1)))
    randTimes /= np.mean(randTimes)

# Randomly sample which cell gets what total umi count
np.random.shuffle(N_c)
if n_cells_simu > nCells_orig:
    N_c = np.tile(N_c, int(np.ceil(n_cells_simu / nCells_orig)))

if args.multiply_cell_counts > 0:
    N_c = np.ceil(N_c * args.multiply_cell_counts).astype(dtype=int)

if not REAL_COVARIANCE:
    # Draw gene variances from an exponential with mean 2
    gene_variances = np.random.exponential(2, nGenes_orig)
    # Take the mean ltqs such that they are sampled from the real data, and that the sum of tqs will in expectation be 1
    mean_ltqs = np.log(N_g / np.sum(N_g)) - .5 * gene_variances

# Initialise tree
tree = Tree()
tree.root.ltqs = np.zeros(n_genes_simu)
tree.nNodes = 1
tree.nLeafs = 0

cellIds = [None] * n_cells_simu
delta_gc = np.zeros((n_genes_simu, n_cells_simu))
if UNBALANCED:
    tree.root.barcode = ''
    all_leaf_nodes = [tree.root]


def addChildCells_unbalanced(node_counter, all_leaf_nodes, variances=None, n_genes=None):
    if n_genes is None:
        n_genes = nGenes_orig
    if variances is None:
        variances = np.ones(n_genes)
    while len(all_leaf_nodes) < n_cells_simu:
        # Pick a random leaf to add two children to
        treeNode_ind = np.random.randint(len(all_leaf_nodes))
        treeNode = all_leaf_nodes[treeNode_ind]
        del all_leaf_nodes[treeNode_ind]
        # treeNode = np.random.choice(all_leaf_nodes, size=1)

        # new_seed = np.random.randint(1e8)
        treeNode.childNodes = []
        for ind in range(2):
            node_counter += 1
            if not RANDOM_TIMES:
                diffTime = 1
            else:
                diffTime = randTimes[node_counter]
            newLtqs = treeNode.ltqs + np.random.normal(0, np.sqrt(diffTime * variances), n_genes)
            newNode = TreeNode(isLeaf=True, ltqs=newLtqs, childNodes=[], nodeInd=node_counter)
            newNode.tParent = diffTime
            newNode.barcode = treeNode.barcode + str(ind)
            treeNode.childNodes.append(newNode)
            tree.nNodes += 1
            all_leaf_nodes.append(newNode)
        treeNode.isLeaf = False

    for ind, leaf in enumerate(all_leaf_nodes):
        delta_gc[:, ind] = leaf.ltqs
        cellIds[ind] = 'Cell{}'.format(leaf.barcode)
        leaf.nodeId = cellIds[ind]
    return node_counter


def addChildCells(treeNode, counter, barcode, node_counter, variances, n_genes):
    counter += 1
    treeNode.childNodes = []
    for ind in range(2):
        node_counter += 1
        if not RANDOM_TIMES:
            diffTime = 1
        else:
            diffTime = randTimes[node_counter]
        newLtqs = treeNode.ltqs + np.random.normal(0, np.sqrt(diffTime * variances), n_genes)
        newNode = TreeNode(isLeaf=True, ltqs=newLtqs, childNodes=[], nodeInd=node_counter)
        newNode.tParent = diffTime
        treeNode.childNodes.append(newNode)
        tree.nNodes += 1
        if counter < N_GENERATIONS:  # Add more cells
            node_counter = addChildCells(newNode, counter, barcode + str(ind), node_counter, variances, n_genes=n_genes)
        else:  # Store information on these "observed cells"
            delta_gc[:, tree.nLeafs] = newLtqs
            cellIds[tree.nLeafs] = "Cell" + barcode + str(ind)
            newNode.nodeId = cellIds[tree.nLeafs]
            tree.nLeafs += 1
    treeNode.isLeaf = False
    return node_counter


if UNBALANCED:
    node_counter = addChildCells_unbalanced(node_counter=1, all_leaf_nodes=all_leaf_nodes, variances=gene_variances,
                                            n_genes=n_genes_simu)
else:
    node_counter = addChildCells(tree.root, counter=0, barcode="", node_counter=-1, variances=gene_variances,
                                 n_genes=n_genes_simu)

# Renumber the vertices to make everything consistent with depth first search
vertIndToNode, tree.nNodes = tree.root.renumber_verts(vertIndToNode={}, vert_count=0)

# Now center the deltas of the cells such that the average is 0
delta_mean = np.mean(delta_gc, axis=1)
delta_gc -= delta_mean[:, None]

if REAL_COVARIANCE:
    # Normalize the simulated cell-components of the SVD (S_k * Vh_kc) such that they have the same variance as in the
    # real dataset
    delta_gc = delta_gc / np.std(delta_gc, axis=1)[:, np.newaxis] * (S_k / np.sqrt(nCells_orig))[:, np.newaxis]
    delta_gc_genespace = np.matmul(U_gk, delta_gc)

    # Add average LTQ for each gene
    ltqs_gc = delta_gc_genespace + mean_ltqs[:, None]
else:
    # Normalise such that each gene has the prescribed variance
    factor = np.sqrt(gene_variances / np.var(delta_gc, axis=1))
    delta_gc *= factor[:, None]

    # Add average LTQ for each gene
    ltqs_gc = mean_ltqs[:, None] + delta_gc

n_genes_final = ltqs_gc.shape[0]

# Normalise such that each cell's TQs add up to 1
log_tqs = logsumexp(ltqs_gc, axis=0)
ltqs_gc = ltqs_gc - log_tqs

# Get true means and vars
true_means_g = np.mean(ltqs_gc, axis=1)
true_vars_g = np.var(ltqs_gc, axis=1)
delta_gc = ltqs_gc - true_means_g[:, None]

"""---------Now store the simulated dataset somewhere---------"""

datadir = "simulated_binary_" + str(N_GENERATIONS) + "_gens_samplingnoise"
if RANDOM_TIMES:
    datadir += '_randomtimes'
if UNBALANCED:
    datadir += '_unbalanced'
if REAL_COVARIANCE:
    datadir += '_realcovariance'
if args.multiply_cell_counts > 0:
    datadir += '_countstimes{}'.format(args.multiply_cell_counts)

datadir += '_seed_{}'.format(seed)

geneID = ['Gene_' + str(ind) for ind in range(n_genes_final)]

data_path = os.path.abspath(os.path.join(args.results_folder, datadir))
Path(data_path).mkdir(parents=True, exist_ok=True)

scData = SCData(onlyObject=True)
scData.tree = tree
scData.metadata.cellIds = cellIds
scData.metadata.nCells = len(cellIds)
bs_glob.nCells = scData.metadata.nCells
scData.metadata.geneVariances = true_vars_g
scData.metadata.geneIds = geneID

# Store ground truth tree with everything necessary to visualize it
scData.tree.set_midpoint_root()
scData.storeTreeInFolder(os.path.join(data_path, "true_tree", "final_bonsai"))
# Make fake yaml-file, necessary for visualizing dataset
scicore_data_path = os.path.join('/scicore/home/nimwegen/degroo0000/sc_datasets/simulated_datasets_copy', datadir)
subprocess.run(
    ['python3', 'bonsai/create_config_file.py', '--new_yaml_path', os.path.join(data_path, "true_tree", 'used_configs.yaml'),
     '--dataset', os.path.join('simulated_datasets', datadir), '--data_folder', scicore_data_path, '--results_folder',
     os.path.join(scicore_data_path, "true_tree"), '--input_is_sanity_output', 'True', '--zscore_cutoff', '-1',
     '--UB_ellipsoid_size', '1.0', '--use_knn', '10', '--filenames_data', 'delta_true.txt,d_delta.txt'])

print("Writing true deltas to file.")
delta_df = pd.DataFrame(delta_gc, columns=cellIds, index=geneID)
delta_df.to_csv(os.path.join(data_path, 'delta_true.txt'), sep='\t', header=False, index=False)

print("Writing true variances to file:")
with open(os.path.join(data_path, 'variance_true.txt'), 'w') as f:
    for var in true_vars_g:
        f.write("%s\n" % var)

print("Writing cell IDs to file:")
with open(os.path.join(data_path, 'cellID.txt'), 'w') as f:
    for ID in cellIds:
        f.write("%s\n" % ID)
print("Writing gene IDs to file:")
with open(os.path.join(data_path, 'geneID.txt'), 'w') as f:
    for ID in geneID:
        f.write("%s\n" % ID)

print("Sampling UMI counts:")
umi_counts = np.zeros((n_genes_final, n_cells_simu))

for cell_ind in range(n_cells_simu):
    if cell_ind % 100 == 0:
        print("Sampling counts for cell %d." % cell_ind)
    umi_counts[:, cell_ind] = np.random.multinomial(N_c[cell_ind], np.exp(ltqs_gc[:, cell_ind]))

print("Writing UMI counts to file:")
umi_df = pd.DataFrame(umi_counts, columns=cellIds, index=geneID)
umi_df.to_csv(os.path.join(data_path, 'Gene_table.txt'), sep='\t', index_label="GeneID")

print("Writing celltypes to file:")
annotation_dict = {}
for gen in range(N_GENERATIONS - 1):
    # with open(os.path.join(data_path, 'Celltype' + str(gen) + '.txt'), 'w') as f:
    celltype_list = ["Type" + ID.split('Cell')[1][:gen + 1] for ID in cellIds]
    annotation_dict['Celltype{}'.format(gen)] = celltype_list

for gen in range(N_GENERATIONS - 1):
    celltype_list = ["Type" + ID.split('Cell')[1][:gen + 1] for ID in cellIds]
    unq_cts, cnts = np.unique(celltype_list, return_counts=True)
    if len(unq_cts) > 20:
        top_ct_inds = np.argsort(-cnts)
        cts_to_new_cts = {ct: (ct if ind in top_ct_inds[:19] else 'rest') for ind, ct in enumerate(unq_cts)}
        celltype_list = [cts_to_new_cts[ct] for ct in celltype_list]
        annotation_dict['Celltype{}_largest_cts'.format(gen)] = celltype_list
# Add total UMI-counts per cell as annotation
total_counts_c = list(np.sum(umi_counts, axis=0))
annotation_dict['total_count'] = total_counts_c

annotation_df = pd.DataFrame(annotation_dict, index=cellIds)
Path(os.path.join(data_path, 'annotation')).mkdir(parents=True, exist_ok=True)
annotation_df.to_csv(os.path.join(data_path, 'annotation', 'lineage_annotation.csv'))

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
N_g_new = np.mean(umi_counts, axis=1)
ax.hist(N_g/nCells_orig, cumulative=-1,histtype='step', bins=1000, align='mid', log=True, color='blue', label='Original', density=True)
ax.hist(N_g_new, cumulative=-1,histtype='step', bins=1000, align='mid', log=True, color='orange', label='Simulated', density=True)
ax.set_xscale('log')
ax.set_xlabel('Mean counts per gene')
ax.set_ylabel('1-CDF')
ax.legend()
plt.savefig(os.path.join(data_path, 'Mean counts per gene distribution.png'))

fig, ax = plt.subplots()
N_c_new = np.sum(umi_counts, axis=0)
ax.hist(N_c, cumulative=-1,histtype='step', bins=1000, align='mid', log=True, color='blue', label='Original', density=True)
ax.hist(N_c_new, cumulative=-1,histtype='step', bins=1000, align='mid', log=True, color='orange', label='Simulated', density=True)
ax.set_xscale('log')
ax.set_xlabel('Total counts per cell')
ax.set_ylabel('1-CDF')
ax.legend()
plt.savefig(os.path.join(data_path, 'Counts per cell distribution.png'))

if REAL_COVARIANCE:
    pca_variances = S_k ** 2 / n_cells_simu
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(pca_variances)), np.cumsum(pca_variances) / np.sum(pca_variances))
    ax2.set_xlabel('Number of PCA components')
    ax2.set_ylabel('Cumulative fraction of explained variance')
    plt.savefig(os.path.join(data_path, 'PCA explained variance distribution.png'))

fig3, ax3 = plt.subplots()
if REAL_COVARIANCE:
    ax3.hist(real_variances, bins=100, cumulative=-1, histtype='step', label='reported by Sanity')
ax3.hist(np.var(ltqs_gc, axis=1), bins=100, cumulative=-1, label='simulated')
ax3.set_xlabel('Gene variances')
ax3.legend()
plt.savefig(os.path.join(data_path, 'Sampled variances.png'))
