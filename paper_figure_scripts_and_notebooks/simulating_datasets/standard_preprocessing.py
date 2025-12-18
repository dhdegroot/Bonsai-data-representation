from argparse import ArgumentParser

import scanpy as sc
from scipy.io import mmwrite, mmread
import pandas as pd
import os
import numpy as np

parser = ArgumentParser(
    description='Simulates a binary tree.')

# Arguments that define where to find the data and where to store results. Also, whether data comes from Sanity or not
parser.add_argument('--input_dataset', type=str,
                    help='Path to original UMI-count  mtx file. genes x cells')
parser.add_argument('--input_dataset_txt', type=str,
                    help='Path to original UMI-count  txt file. genes x cells, with gene and cell ids')
parser.add_argument('--input_annotation', type=str, required=True,
                    help='Path to cell type annotation, with header, can be multiple columns, tab separated')
parser.add_argument('--input_genes', type=str,required=True,
                    help='Path to gene annotation, no header, one column')
parser.add_argument('--results_folder', type=str,
                    help="Relative path from bonsai_development to base-folder where simulated trees need to be stored.")

args = parser.parse_args()
print(args)


if args.input_dataset:
    print("reading in mtx file {}".format(args.input_dataset))
    gene_expression = mmread(args.input_dataset).toarray()
elif args.input_dataset_txt:
    print("reading in txt file {}".format(args.input_dataset_txt))
    gene_expression = pd.read_csv(args.input_dataset_txt, sep='\t', index_col=0)
    gene_expression = gene_expression.to_numpy()
else:
    print("no input dataset")
    exit

print(gene_expression.shape)

if args.input_annotation is not None:
    # annot_df = pd.read_csv(args.input_annotation, sep='\t')
    annot_df = pd.read_csv(args.input_annotation, sep=',')
    print(annot_df.shape)
if args.input_genes is not None:
    genes_df = pd.read_csv(args.input_genes, header=None, names=["geneName"])
    print(genes_df.shape)

print("creating anndata object")

adata = sc.AnnData(gene_expression.T)   # transpose so cells are rows
print("adata.shape:{}".format(adata.shape))
if args.input_annotation is not None:
    adata.obs = annot_df
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs_names_make_unique()

if args.input_genes is not None:
    adata.var_names = genes_df.geneName
    adata.var_names_make_unique()
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()


adata.layers["counts"] = adata.X.copy()
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, min_counts=200)


# Normalizing to median total counts
sc.pp.normalize_total(adata, exclude_highly_expressed=False, target_sum=None)

# 6) Preprocessing: normalize to total count = 1e6 and log1p
#sc.pp.normalize_total(adata)   # your requested total count
sc.pp.log1p(adata)
# feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Boolean mask for HVGs
hvg_mask = adata.var['highly_variable'].values
# HVG-restricted AnnData (still log1p if .X already is)
adata_hvg = adata[:, hvg_mask].copy()
# The log1p matrix for HVGs:
X_log1p_hvg = adata_hvg.X  # sparse or dense
print("X_log1p_hvg.shape : {}".format(X_log1p_hvg.shape))

hvg_genes = adata.var_names[hvg_mask]

# save:
save_dir = os.path.join(args.results_folder, "processed_raw_cnts_log1p_hvg")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

print("will save in: {}".format(save_dir))
np.savetxt(os.path.join(save_dir, "features.txt"), X_log1p_hvg.T, delimiter="\t")
np.savetxt(os.path.join(save_dir, "geneID.txt"), hvg_genes.values, fmt="%s")
np.savetxt(os.path.join(save_dir, "cellID.txt"), adata.obs["Unnamed: 0"].values, fmt="%s")
#np.savetxt(os.path.join(save_dir, "cellID.txt"), adata.obs["cellbarcode_full"].values, fmt="%s")



