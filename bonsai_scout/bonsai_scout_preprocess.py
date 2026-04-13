from argparse import ArgumentParser
import os, sys, re
from scipy.sparse import csr_matrix
import numpy as np
import h5py
import json
import pandas as pd
from argparse import ArgumentTypeError
from pathlib import Path

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

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory of this script-file to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

if sys.platform == "win32":
    import types
    m = types.ModuleType("resource")
    m.RLIMIT_AS = 0
    m.RUSAGE_SELF = 0
    m.getrusage = lambda x: types.SimpleNamespace(
        ru_maxrss=0, ru_utime=0, ru_stime=0
    )
    m.setrlimit = lambda x, y: None
    sys.modules["resource"] = m

from bonsai.bonsai_helpers import str2bool, Run_Configs, find_latest_tree_folder, mp_print
from downstream_analyses.get_clusters_max_diameter import get_min_pdists_clustering_from_nwk_str, \
    get_cluster_assignments, get_annotation_based_clustering_from_nwk_str

parser = ArgumentParser(
    description='Starts from a reconstructed tree output by Bonsai and creates a data-object necessary for further '
                'visualization and usage in the Bonsai-Shiny app.')

parser.add_argument('--results_folder', type=str, default='',
                    help='Path to folder where all Bonsai results were stored. (results_folder argument in '
                         'Bonsai configuration-file.'
                         'This folder should contain a copy of that configuration YAML-file, and a folder with tree '
                         'reconstructed tree information. This script will first look for "final_bonsai_..." then for '
                         'longest "intermediate_bonsai_", such that results are visualized that are furthest along.')

parser.add_argument('--annotation_path', type=str, default=None,
                    help='Path to csv-file (absolute or rel-to "bonsai-development) with as first column the original '
                         'cellIDs (as provided in cellID.txt). Next'
                         'columns can contain different cell-annotations. Header gives labels to the different '
                         'annotations.')

parser.add_argument('--config_filepath', type=str, default=None,
                    help='Only give this argument if you want to deviate from the "run_configs_....yaml"-file that is '
                         'stored in the results_folder. If so, give the absolute (or relative to "bonsai-development") '
                         'path to the YAML-file that contains all arguments that were used to run Bonsai.')

parser.add_argument('--take_all_genes', type=str2bool, default=False,
                    help='Whether all gene expression values are loaded or only the genes on which the tree '
                         'reconstruction was based. Loading all gene expression values will make this preprocessing'
                         'a lot slower for large datasets, and will require much more memory.')

parser.add_argument('--take_no_genes', type=str2bool, default=False,
                    help='For very large datasets, it can be too memory-intensive to load all gene expresion values'
                         'into memory. This mode allows to explore the tree, but prohibits plotting gene expression'
                         'or finding marker genes.')

parser.add_argument('--cell_id_to_cs_id_file', type=str, default='',
                    help='Should point towards csv-file with two columns: 1) cell-IDs, 2) the corresponding '
                         'cellstates-id'
                         'Note that this is only necessary if cells that were used as input for Bonsai were in fact '
                         'cellstates. In the visualization, the cellstates-clustering will be used to show number of'
                         'cells in cellstate and to translate cell-annotation to cellstate-coloring.')

parser.add_argument('--new_gene_ids_file', type=str, default='',
                    help='Should point towards a csv- or tsv-file with the first column containing the old gene-IDs'
                         '(as used in Bonsai) and the second column containing the new gene-IDs.')

parser.add_argument('--perform_annot_guided_clustering', type=str2bool, default=True,
                    help='Determines whether we use categorical annotations to create a tree-consistent clustering of'
                         'the cells that best aligns with that annotation.'
                         'Since this clustering can be somewhat slow when many categorical annotations are provided,'
                         'one can turn it off here.')

# parser.add_argument('--use_log1p_dists', type=str2bool, default=False,
#                     help='Can plot log(1+dist)-distances to make tree more easily viewable')

# parser.add_argument('--redo_layout', type=str2bool, default=True,
#                     help='Determines whether we recalculate tree layout')

args = parser.parse_args()

# Find correct YAML-file
config_filepath = None
if (args.config_filepath is not None) and len(args.config_filepath):
    config_filepath = args.config_filepath
else:
    for file in os.listdir(args.results_folder):
        if re.search(".*config.*.yaml$", file) is not None:
            config_filepath = file
            break

if config_filepath is None:
    mp_print("Couldn't find configurations file (YAML-file) that was used for running Bonsai.", ERROR=True)
    mp_print("For now, we're trying to interpret --results_folder as a folder with tree-information, rather than the"
             "main results folder of Bonsai. This may break. To be sure, --results_folder should point to "
             "all Bonsai-results, and --config_filepath can be used to point to the used Bonsai-config file.")
    tree_folder = args.results_folder
    run_configs = Run_Configs(create_empty_configs=True)
    run_configs.results_folder = args.results_folder
else:
    config_filepath = os.path.abspath(os.path.join(args.results_folder, config_filepath))

    run_configs = Run_Configs(config_filepath)
    run_configs.results_folder = args.results_folder

    # Find latest tree-folder
    rel_tree_folder = find_latest_tree_folder(run_configs.results_folder)
    tree_folder = os.path.abspath(os.path.join(args.results_folder, rel_tree_folder))

from bonsai.bonsai_dataprocessing import loadReconstructedTreeAndData
from bonsai_scout.bonsai_scout_helpers import get_edge_coords, Bonvis_figure, Bonvis_settings, merge_cells_at_zero_dist, \
    Bonvis_metadata, Celltype_info, process_annot_based_clsts
from bonsai_scout.my_tree_layout import my_tree_layout
from bonsai_scout.change_gene_ids import change_json_file

if len(args.new_gene_ids_file):
    change_json_file(new_gene_ids_file=args.new_gene_ids_file, path_to_json=os.path.join(tree_folder, 'metadata.json'))

if args.take_all_genes:
    reprocess_data = True
    all_genes = True
else:
    reprocess_data = False
    all_genes = False

get_data = not args.take_no_genes
scData, _ = loadReconstructedTreeAndData(run_configs, tree_folder, reprocess_data=reprocess_data,
                                         get_cell_info=True, all_ranks=False, all_genes=all_genes, get_data=get_data,
                                         rel_to_results=False, no_data_needed=True, calc_loglik=False,
                                         get_posterior_ltqs=True)

# We read in cellstates-information if it is available. This is only necessary if cellstates-output was run to group
# cells, and these groups were used as input to Bonsai.
# To make this consistent. We'll in all cases assume that Bonsai got cellstates as input. If this was not the case, this
# is captured by making the cell->cellstates the identity-mapping
if len(args.cell_id_to_cs_id_file) > 0:
    try:
        ext = os.path.splitext(args.cell_id_to_cs_id_file)[1]
        if ext == '.tsv':
            delim = '\t'
        elif ext == '.csv':
            delim = ','
        else:
            delim = '\t'
        cell_id_to_cs_id = pd.read_csv(args.cell_id_to_cs_id_file, sep=delim,
                                       header=None, index_col=0)
        scData.metadata.csIds = scData.metadata.cellIds.copy()
        scData.metadata.cellIds = list(cell_id_to_cs_id.index)
        scData.metadata.cell_id_to_cs_id = cell_id_to_cs_id.iloc[:, 0].to_dict()
        scData.metadata.cell_ind_to_cs_ind = {ind: scData.metadata.csIds.index(scData.metadata.cell_id_to_cs_id[cellId])
                                              for
                                              ind, cellId in enumerate(scData.metadata.cellIds)}
    except FileNotFoundError:
        logger.debug("Could not find cell to cellstates mapping. Assuming Bonsai was reconstructed on cells directly.")
        args.cell_id_to_cs_id_file = ''

if len(args.cell_id_to_cs_id_file) == 0:
    scData.metadata.csIds = scData.metadata.cellIds
    scData.metadata.cell_id_to_cs_id = {cell_id: cell_id for cell_id in scData.metadata.cellIds}
    scData.metadata.cell_ind_to_cs_ind = {ind: ind for ind in range(scData.metadata.nCells)}

scData.metadata.nCells = len(scData.metadata.cellIds)
scData.metadata.nCss = len(scData.metadata.csIds)
scData.csIndToVertInd = scData.cellIndToVertInd.copy()
# scData.cssToVerts = scData.cellsToVerts.copy()
scData.cellIndToVertInd = {cell_ind: scData.csIndToVertInd[cs_ind]
    for cell_ind, cs_ind in scData.metadata.cell_ind_to_cs_ind.items() if cs_ind in scData.csIndToVertInd}
scData.metadata.cs_ind_to_cell_inds = {cs_ind: [] for cs_ind in range(scData.metadata.nCss)}
for cell_ind, cs_ind in scData.metadata.cell_ind_to_cs_ind.items():
    scData.metadata.cs_ind_to_cell_inds[cs_ind].append(cell_ind)
scData.metadata.nCellsPerCs = np.array([len(cells) for cs, cells in scData.metadata.cs_ind_to_cell_inds.items()])
# scData.cellsToVerts = {cell_id: scData.cssToVerts[cs_id] for cell_id, cs_id in scData.metadata.cell_id_to_cs_id.items()}

"""This script produces 2 data-files. One with data that doesn't change with different plotting options, which we will
create first. One with precalculated settings to be able to make a first plot. These settings can change when different
options are picked."""
# Merge cells with parent if tParent = 0 but keep track of this info
merge_cells_at_zero_dist(scData)

# if args.use_log1p_dists:
#     tParents = []
#     for vert_ind, node in scData.tree.vert_ind_to_node.items():
#         if node.tParent is not None:
#             tParents.append(node.tParent)
#             node.tParent = np.log(1 + node.tParent)

# Store tree in Newick-format as well
# nwk_str = scData.tree.to_newick(results_path=os.path.join(tree_folder, 'tree.nwk'))
# nwk_str = scData.tree.to_newick(results_path=scData.result_path('tree.nwk'))
nwk_str = scData.tree.to_newick()

# If we get posterior_ltqs and rescale_by_var was True, then we have to undo that rescaling
if reprocess_data:
    if run_configs.rescale_by_var:
        undo_rescaling_by_var = 'geneVariances'
    else:
        undo_rescaling_by_var = np.mean(scData.metadata.geneVariances)
else:
    undo_rescaling_by_var = None

if scData.tree.root.ltqsAIRoot is not None:
    ltqs = np.zeros((scData.metadata.nGenes, scData.nVerts))
    ltqs_vars = np.zeros((scData.metadata.nGenes, scData.nVerts))
    for vert_ind, node in scData.tree.vert_ind_to_node.items():
        if undo_rescaling_by_var is None:
            ltqs[:, vert_ind] = node.ltqsAIRoot
            ltqs_vars[:, vert_ind] = node.getLtqsVars(AIRoot=True)
        elif undo_rescaling_by_var == 'geneVariances':
            ltqs[:, vert_ind] = node.ltqsAIRoot * np.sqrt(scData.metadata.geneVariances)
            ltqs_vars[:, vert_ind] = node.getLtqsVars(AIRoot=True) * scData.metadata.geneVariances
        else:
            ltqs[:, vert_ind] = node.ltqsAIRoot * np.sqrt(undo_rescaling_by_var)
            ltqs_vars[:, vert_ind] = node.getLtqsVars(AIRoot=True) * undo_rescaling_by_var
else:
    ltqs = None
    ltqs_vars = None

node_ids = ['na'] * scData.nVerts
for vert_ind, node in scData.tree.vert_ind_to_node.items():
    node_ids[vert_ind] = node.nodeId

# Initialize anndata object where all data will be stored
# bonvis_data = anndata.AnnData(ltqs, dtype=np.float32)
try:
    bonvis_data_hdf = h5py.File(scData.result_path('bonsai_vis_data.hdf'), 'w')
except BlockingIOError:
    os.remove(scData.result_path('bonsai_vis_data.hdf'))
    bonvis_data_hdf = h5py.File(scData.result_path('bonsai_vis_data.hdf'), 'w')

feature_paths = []
# TODO: Transpose everything again later
data_hdf = bonvis_data_hdf.create_group('data')
if ltqs is not None:
    normalized_hdf = data_hdf.create_group('normalized')
    normalized_hdf.create_dataset('means', data=ltqs)
    feature_paths.append('data/normalized')
    normalized_hdf.attrs['node_ids'] = json.dumps(node_ids)
    normalized_hdf.attrs['gene_ids'] = json.dumps(scData.metadata.geneIds)
    no_variation_genes = np.where(ltqs.min(axis=1) == ltqs.max(axis=1))[0]
    normalized_hdf.create_dataset('no_variation_features', data=no_variation_genes)
    normalized_hdf.create_dataset('variances', data=scData.metadata.geneVariances)
    if ltqs_vars is not None:
        # bonvis_data.layers['ltqs_vars'] = ltqs_vars
        normalized_hdf.create_dataset('vars', data=ltqs_vars)
        zscores = np.sqrt(np.mean((ltqs - np.mean(ltqs, axis=1, keepdims=True)) ** 2 / ltqs_vars, axis=1))
        normalized_hdf.create_dataset('zscores', data=zscores)
else:
    normalized_hdf = data_hdf.create_group('dummy_gene')
    normalized_hdf.create_dataset('means', data=np.linspace(.9, 1.1, len(node_ids))[None, :])
    feature_paths.append('data/dummy_gene')
    normalized_hdf.attrs['node_ids'] = json.dumps(node_ids)
    normalized_hdf.attrs['gene_ids'] = json.dumps(['dummy_gene'])
    no_variation_genes = np.array([])
    normalized_hdf.create_dataset('no_variation_features', data=no_variation_genes)
    normalized_hdf.create_dataset('variances', data=np.array([1.0]))

# Get z-scores of genes
# if ltqs_vars is not None:
#     zscores = np.sqrt(np.mean((ltqs - np.mean(ltqs, axis=1, keepdims=True)) ** 2 / ltqs_vars, axis=1))
# else:
#     zscores = np.zeros(scData.metadata.nGenes)
#     zscores[:] = np.nan
# gene_info_hdf = bonvis_data_hdf.create_group('gene_info')
# gene_info_hdf.create_dataset('gene_variances', data=scData.metadata.geneVariances)


# scData.read_umi_counts()
# if scData.umiCounts is not None:
#     umi_counts = np.zeros((scData.metadata.nGenes, scData.nVerts))
#     for cell_ind, vert_ind in scData.cellIndToVertInd.items():
#         umi_counts[:, vert_ind] += scData.umiCounts[:, cell_ind].T
#     sparse_umis = csr_matrix(umi_counts)
#     data_hdf.create_dataset('raw_data', data=umi_counts)
#     feature_paths.append('data/raw_data')

bonvis_data_hdf.close()

cell_info_df, cs_info_df, data_matrices = scData.get_annotations_with_cs(args.annotation_path)

if cell_info_df is None:
    cell_info_df = pd.DataFrame(index=scData.metadata.cellIds)
cell_ind_to_vert_ind = [-1] * scData.metadata.nCells
for cell_ind, vert_ind in scData.cellIndToVertInd.items():
    cell_ind_to_vert_ind[cell_ind] = vert_ind
cell_info_df['cell_ind_to_vert_ind'] = cell_ind_to_vert_ind

if cs_info_df is None:
    cs_info_df = pd.DataFrame(index=scData.metadata.csIds)
cs_ind_to_vert_ind = [-1] * scData.metadata.nCss
for cs_ind, vert_ind in scData.csIndToVertInd.items():
    cs_ind_to_vert_ind[cs_ind] = vert_ind
cs_info_df['cs_ind_to_vert_ind'] = cs_ind_to_vert_ind
cs_info_df['n_cells_per_cs'] = scData.metadata.nCellsPerCs

cell_info_df.to_hdf(scData.result_path('bonsai_vis_data.hdf'), key='cell_info/cell_info_dict', mode='a', format='table',
                    data_columns=True)
cs_info_df.to_hdf(scData.result_path('bonsai_vis_data.hdf'), key='cs_info/cs_info_dict', mode='a', format='table',
                  data_columns=True)

bonvis_data_hdf = h5py.File(scData.result_path('bonsai_vis_data.hdf'), 'a')
data_hdf = bonvis_data_hdf['data']
for data_label, data_matrix in data_matrices.items():
    new_data_hdf = data_hdf.create_group(data_label)
    data_mat = data_matrix.values.astype(dtype=float)
    new_data_hdf.create_dataset('means', data=data_mat)
    feature_paths.append('data/' + data_label)
    new_data_hdf.attrs['node_ids'] = json.dumps(list(data_matrix.columns))
    new_data_hdf.attrs['gene_ids'] = json.dumps(list(data_matrix.index))
    new_data_hdf.attrs['cell_or_cs'] = 'cell' if data_matrix.shape[1] == scData.metadata.nCells else 'cs'
    no_variation_features = np.where(np.nanmin(data_mat, axis=1) == np.nanmax(data_mat, axis=1))[0]
    new_data_hdf.create_dataset('no_variation_features', data=no_variation_features)
    new_data_hdf.create_dataset('variances', data=np.nanvar(data_mat, axis=1))

# test = pd.read_hdf(scData.result_path('bonsai_vis_data.dat'), key='cell_info')

for feature_path in feature_paths:
    feature_hdf = bonvis_data_hdf[feature_path]
    feature_data = feature_hdf['means']
    if feature_data.shape[1] == scData.tree.nNodes:
        vert_inds_corr_to_cells = np.zeros(scData.metadata.nCells, dtype=int)
        for cell_ind, vert_ind in scData.cellIndToVertInd.items():
            vert_inds_corr_to_cells[cell_ind] = vert_ind
        cell_data = feature_data[:][:, vert_inds_corr_to_cells]
        # cell_data = np.zeros((feature_data.shape[0], scData.metadata.nCells))
        # for cell_ind, vert_ind in scData.cellIndToVertInd.items():
        #     cell_data[:, cell_ind] = feature_data[:, vert_ind]
    elif feature_data.shape[1] == scData.metadata.nCells:
        cell_data = feature_data
    elif feature_data.shape[1] == scData.metadata.nCss:
        cs_inds_corr_to_cells = np.zeros(scData.metadata.nCells, dtype=int)
        for cell_ind, cs_ind in scData.metadata.cell_ind_to_cs_ind.items():
            cs_inds_corr_to_cells[cell_ind] = cs_ind
        cell_data = feature_data[:][:, cs_inds_corr_to_cells]
        # cell_data = np.zeros((feature_data.shape[0], scData.metadata.nCells))
        # for cell_ind, cs_ind in scData.metadata.cell_ind_to_cs_ind.items():
        #     cell_data[:, cell_ind] = feature_data[:, cs_ind]
    else:
        logger.error(
            "Number of columns in feature matrix {} does not match either number of cells, number of cellstates "
            "or number of vertices.".format(feature_path))
    # For determining marker genes, we only want to consider cells that have zero nan's for every gene
    cells_wo_nan = np.where(np.sum(np.isnan(cell_data), axis=0) == 0)[0]
    # Determine ranks for remaining genes
    ranks_per_gene_notnan = np.argsort(np.argsort(cell_data[:, cells_wo_nan], axis=1), axis=1)
    # Note that we thus give a negative number to the rank of all cells that have no data
    ranks_per_gene = np.full_like(cell_data, dtype=int, fill_value=-1)
    ranks_per_gene[:, cells_wo_nan] = ranks_per_gene_notnan
    feature_hdf.create_dataset('cells_wo_nan', data=cells_wo_nan)
    feature_hdf.create_dataset('ranks_per_gene', data=ranks_per_gene)

#
# if (ltqs is not None) and (scData.cellIndToVertInd is not None):
#     cell_ltqs = np.zeros((scData.metadata.nGenes, scData.metadata.nCells))
#     for cell_ind, vert_ind in scData.cellIndToVertInd.items():
#         cell_ltqs[:, cell_ind] = ltqs[:, vert_ind]
#     ranks_per_gene = np.argsort(np.argsort(cell_ltqs, axis=1), axis=1)
#     data_hdf = bonvis_data_hdf['data']
#     ranks_per_gene_hdf = data_hdf.create_dataset('ranks_per_gene', data=ranks_per_gene)
#     ranks_per_gene_hdf.attrs['based_on'] = 'normalized'

# Store list of inds of cells that are alone at a vertex
single_cell_inds = np.array([cells[0] for vert, cells in scData.vertIndToCellInds.items() if (len(cells) == 1)])
multi_at_vert = np.setxor1d(single_cell_inds, np.arange(scData.metadata.nCells), assume_unique=True)

single_cs_inds = np.array([cells[0] for vert, cells in scData.vertIndToCsInds.items() if (len(cells) == 1)])
multi_cs_at_vert = np.setxor1d(single_cs_inds, np.arange(scData.metadata.nCss), assume_unique=True)

cell_info_hdf = bonvis_data_hdf['cell_info']
cell_info_hdf.create_dataset('single_at_vert', data=single_cell_inds)
cell_info_hdf.create_dataset('multi_at_vert', data=multi_at_vert)

cs_info_hdf = bonvis_data_hdf['cs_info']
cs_info_hdf.create_dataset('single_cs_at_vert', data=single_cs_inds)
cs_info_hdf.create_dataset('multi_cs_at_vert', data=multi_cs_at_vert)

vert_info_hdf = bonvis_data_hdf.create_group('vert_info')
vert_info_hdf.create_dataset('n_cells_per_vert', data=scData.nCellsPerVert)
vert_info_hdf.create_dataset('n_css_per_vert', data=scData.nCssPerVert)
vi_to_ci = {vert_ind: cell_ind for vert_ind, cell_ind in scData.vertIndToCellInds.items()}
vi_to_csi = {vert_ind: cs_ind for vert_ind, cs_ind in scData.vertIndToCsInds.items()}
vert_info_hdf.attrs['vert_ind_to_cell_inds_json'] = json.dumps(vi_to_ci)
vert_info_hdf.attrs['vert_ind_to_cs_inds_json'] = json.dumps(vi_to_csi)

# Get tree layout
coords_dict, ly_type_picked = my_tree_layout(scData, True,
                                             filepath=os.path.join(tree_folder, 'layout.csv'),
                                             daylight_subset=None,
                                             eq_daylight=(scData.nVerts < 2000), verbose=run_configs.verbose,
                                             eq_dl_max_stepsize_changes=10, eq_dl_max_steps=30, return_all=True)
node_coords_hdf = bonvis_data_hdf.create_group('layout_coords/node_coords')
for ly_type_ind in coords_dict:
    node_coords_hdf.create_dataset(ly_type_ind, data=coords_dict[ly_type_ind])

# Store metadata as unstructured data
# keys = ['dataset', 'cellIds', 'geneIds', 'geneVariances', 'loglik', 'nCells', 'nGenes', 'pathToOrigData']
# metadata_dict = {key: getattr(scData.metadata, key) for key in keys}
keys = ['dataset', 'cellIds', 'csIds', 'geneIds', 'loglik', 'nCells', 'nCss', 'nGenes', 'pathToOrigData',
        'cell_ind_to_cs_ind', 'cs_ind_to_cell_inds']
metadata_dict = {key: getattr(scData.metadata, key) for key in keys}
metadata_dict['nodeIds'] = node_ids
bonvis_data_hdf.attrs['metadata_json'] = json.dumps(metadata_dict)
bonvis_data_hdf.attrs['feature_paths'] = json.dumps(feature_paths)

# Store information on tree as unstructured metadata
tree_info = {}
tree_info['nwk_str'] = scData.tree.to_newick()
# Store leaf_inds as well
cell_inds = []
multi_cell_inds = []
cs_inds = []
multi_cs_inds = []
int_inds = []
n_leafs = 0
vert_n_cells = np.zeros(scData.tree.nNodes, dtype=int)
for vert_ind, node in scData.tree.vert_ind_to_node.items():
    if not node.isLeaf:
        int_inds.append(vert_ind)
    else:
        n_leafs += 1
    n_cells_at_vert = len(scData.vertIndToCellInds[vert_ind])
    vert_n_cells[vert_ind] = n_cells_at_vert
    if n_cells_at_vert > 0:
        cell_inds.append(vert_ind)
        if n_cells_at_vert > 1:
            multi_cell_inds.append(vert_ind)
    n_css_at_vert = len(scData.vertIndToCsInds[vert_ind])
    if n_css_at_vert > 0:
        cs_inds.append(vert_ind)
        if n_css_at_vert > 1:
            multi_cs_inds.append(vert_ind)
tree_info['cell_inds'] = np.array(cell_inds)
tree_info['int_inds'] = np.array(int_inds)
tree_info['cs_inds'] = np.array(cs_inds)
tree_info['multi_cell_inds'] = np.array(multi_cell_inds)
tree_info['multi_cs_inds'] = np.array(multi_cs_inds)
tree_info['n_cells_per_vert'] = vert_n_cells
tree_info_hdf = bonvis_data_hdf.create_group('tree_info')
tree_info_hdf.attrs['n_leafs'] = n_leafs
nwk_str = scData.tree.to_newick()
tree_info_hdf.attrs["nwk_str"] = nwk_str
tree_info_hdf.create_dataset('cell_inds', data=np.array(cell_inds))
tree_info_hdf.create_dataset('int_inds', data=np.array(int_inds))
tree_info_hdf.create_dataset('cs_inds', data=np.array(cs_inds))
tree_info_hdf.create_dataset('multi_cell_inds', data=np.array(multi_cell_inds))
tree_info_hdf.create_dataset('multi_cs_inds', data=np.array(multi_cs_inds))
tree_info_hdf.create_dataset('n_cells_per_vert', data=vert_n_cells)

# Get edge info
node_id_to_vert_ind = {node.nodeId: vert_ind for vert_ind, node in scData.tree.vert_ind_to_node.items()}
edge_df = scData.tree.get_edge_dataframe(nodeIdToVertInd=node_id_to_vert_ind)
edge_dict = edge_df.to_dict(orient='list')
edge_coords_dict = get_edge_coords(edge_dict, coords_dict, scData.tree.nNodes)
edge_info = {'edge_df_dict': edge_dict}
edge_coords_hdf = bonvis_data_hdf['layout_coords'].create_group('edge_coords')
for ly_type_ind in edge_coords_dict:
    edge_info[ly_type_ind] = edge_coords_dict[ly_type_ind]
    edge_coords_hdf.create_dataset(ly_type_ind, data=edge_coords_dict[ly_type_ind])

bonvis_data_hdf.close()

edge_df.to_hdf(scData.result_path('bonsai_vis_data.hdf'), key='tree_info/edge_df', mode='a', format='fixed',
               data_columns=True)

# Get the first 100 clusters on the tree
node_id_to_n_cells = {}
node_ids_with_cells = []
vert_ind_to_node_id = {}
for ind, node_id in enumerate(node_ids):
    node_id_to_n_cells[node_id] = vert_n_cells[ind]
    vert_ind_to_node_id[ind] = node_id
    if vert_n_cells[ind] > 0:
        node_ids_with_cells.append(node_id)
# node_id_to_n_cells = {node_id: vert_n_cells[ind] for ind, node_id in enumerate(node_ids)}
all_clusterings, cut_edges = get_min_pdists_clustering_from_nwk_str(tree_nwk_str=nwk_str, n_clusters=100,
                                                                    cell_ids=node_ids_with_cells,
                                                                    node_id_to_n_cells=node_id_to_n_cells,
                                                                    footfall=False)

cell_info_dict = cell_info_df.to_dict(orient='list')
cs_info_dict = cs_info_df.to_dict(orient='list')
celltype_info = Celltype_info(cell_info_dict=cell_info_dict,
                              cs_info_dict=cs_info_dict)
cell_id_to_node_id = {scData.metadata.cellIds[cell_ind]: node_ids[vert_ind] for cell_ind, vert_ind in
                      scData.cellIndToVertInd.items()}
cs_id_to_node_id = {scData.metadata.csIds[cs_ind]: node_ids[vert_ind] for cs_ind, vert_ind in
                    scData.csIndToVertInd.items()}

# all_clusterings is a dictionary with keys 'Bnsi_cluster_n=..' and as vals lists of lists of cs-IDs which give the clusters
# We need to convert this into a pandas dataframe with index the cs_ids and entries the cluster-assignments as "cl_{}"
node_ids_multiple_cs_ids = {vert_ind_to_node_id[vert_ind]: [scData.metadata.csIds[cs_ind] for cs_ind in cs_inds] for
                            vert_ind, cs_inds in scData.vertIndToCsInds.items() if len(cs_inds) > 1}
cl_df = get_cluster_assignments(all_clusterings=all_clusterings, node_ids_multiple_cs_ids=node_ids_multiple_cs_ids)
cl_df = cl_df.loc[metadata_dict['csIds']]
# Process distance-based clustering names
cl_df = process_annot_based_clsts(cl_df, cell2annot=None)

all_cl_dfs = [cl_df]

if args.perform_annot_guided_clustering:
    mp_print("Starting annotation-guided clustering.")

    for annot_id, annot_info in celltype_info.annot_infos.items():
        if annot_info.color_type != 'categorical':
            continue
        mp_print("Starting clustering guided by Annotation: {}".format(annot_id))

        # TODO: REMOVE THIS FOR SURE!
        # if annot_id not in ['annot_rna_annotations', 'annot_protein_annotations', 'annot_cell_type_v2']: # , 'annot_chen_group']:
        #     continue
        if annot_id:
            if annot_id.startswith('annot_num'):
                continue

        cell2cs = None
        if annot_info.info_object == 'cell_info_dict':
            cell_to_celltype = cell_info_dict[annot_id]
            to_be_clst_ids = scData.metadata.cellIds
            id_to_node_id = cell_id_to_node_id
            cell2cs = scData.metadata.cell_id_to_cs_id
        elif annot_info.info_object == 'cs_info_dict':
            cell_to_celltype = cs_info_dict[annot_id]
            to_be_clst_ids = scData.metadata.csIds
            id_to_node_id = cs_id_to_node_id
        annotation_dict = dict(zip(to_be_clst_ids, cell_to_celltype))
        if 'NaN' in annot_info.cats:
            annotation_dict = {key: val for key, val in annotation_dict.items() if val != 'NaN'}

        tracking_folder = scData.result_path('annot_based_clustering_stats')
        random_sampling = True
        suffix = '_random.tsv' if random_sampling else '.tsv'
        tracking_path = os.path.join(tracking_folder, annot_id + '_clustering_stats' + suffix)
        Path(tracking_folder).mkdir(parents=True, exist_ok=True)

        clusters, cut_edges, mut_info = get_annotation_based_clustering_from_nwk_str(tree_nwk_str=nwk_str,
                                                                                     annotation_dict=annotation_dict,
                                                                                     node_ids_to_clst=node_ids_with_cells,
                                                                                     cell_id_to_node_id=id_to_node_id,
                                                                                     verbose=True,
                                                                                     random_sampling=True,
                                                                                     tracking_path=tracking_path,
                                                                                     max_moves=1000, cutting_tol=1e-4,
                                                                                     prohibit_small_clsts=False)
        if len(cut_edges):  # i.e. we discard the trivial clusterings
            clustering_name = 'annot_bnsi_cluster_' + annot_id[6:] if annot_id.startswith('annot_') else annot_id
            cl_df_annot = get_cluster_assignments(all_clusterings={clustering_name: clusters},
                                                  node_ids_multiple_cs_ids=node_ids_multiple_cs_ids)
            cl_df_annot = cl_df_annot.loc[metadata_dict['csIds']]
            # Process annot-based clustering results
            cl_df_annot = process_annot_based_clsts(cl_df_annot, cell2annot=annotation_dict, cell2cs=cell2cs)
            all_cl_dfs.append(cl_df_annot)

cl_df = pd.concat(all_cl_dfs, axis=1)
cl_df.to_hdf(scData.result_path('bonsai_vis_data.hdf'), key='cs_info/cluster_info_dict', mode='a', format='fixed',
             data_columns=True)

bonvis_data_hdf.close()
# test = pd.read_hdf(scData.result_path('bonsai_vis_data.dat'), key='cell_info')

"""After this we will create an object that contains all necessary information to make a first tree-visualization.
This will be stored and used to make the initial visualization. After that the information can change."""
bonvis_metadata = Bonvis_metadata(scData.result_path('bonsai_vis_data.hdf'))

bonvis_settings = Bonvis_settings(bonvis_metadata=bonvis_metadata, geometry='hyperbolic', origin_style='root',
                                  ly_type=ly_type_picked)

# start = time.time()
bonvis_settings.to_json(settings_path=scData.result_path('bonsai_vis_settings.json'))
logger.info("Stored preprocessed data in {}".format(scData.result_path('bonsai_vis_settings.json')))
# print("This took {:.2f} seconds.".format(time.time() - start))
