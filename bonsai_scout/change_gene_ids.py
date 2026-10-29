from argparse import ArgumentParser
import pandas as pd
import os
import numpy as np
import json
from datetime import datetime


def get_new_gene_ids(new_gene_ids_file, gene_ids, old_geneIDs_col=None, new_geneIDs_col=None):
    # Returns the list gene_ids where each old gene-ID that appears in new_gene_ids_file is replaced by its new gene-ID
    print("Read in file which contains mapping from old geneID to new geneID: {}".format(new_gene_ids_file))
    filepath = new_gene_ids_file
    ext = os.path.splitext(filepath)[1]
    if ext == '.tsv':
        delim = '\t'
    elif ext == '.csv':
        delim = ','
    if old_geneIDs_col is None:
        header = None
    else:
        header = 0
    gene_annot = pd.read_csv(filepath, delimiter=delim, header=header)

    metadata_geneIds = list(gene_ids)

    # metadata_geneIds = metadata['geneIds'].copy()
    # df = pd.DataFrame({'sig_promoter': metadata['geneIds']})
    # df["id"] = df.index

    # print("there are {} geneIDs in the metadata.json".format(len(df)))

    # out = pd.merge(df, gene_annot, left_on="sig_promoter", right_on=args.geneIDs_col)
    # out = out.sort_values("id")

    # Make new gene names unique
    if new_geneIDs_col is None:
        new_gene_names = gene_annot.iloc[:, 1]
    else:
        new_gene_names = gene_annot[new_geneIDs_col]

    new_gene_names_unq = []
    new_gene_names_counts = {}
    for gene_name in new_gene_names:
        if gene_name not in new_gene_names_counts:
            new_gene_names_counts[gene_name] = 1
            new_gene_names_unq.append(gene_name)
        else:
            new_gene_names_counts[gene_name] += 1
            new_gene_names_unq.append(gene_name + '_promoter{}'.format(new_gene_names_counts[gene_name]))

    if old_geneIDs_col is None:
        old_gene_names = gene_annot.iloc[:, 0]
    else:
        old_gene_names = gene_annot[old_geneIDs_col]

    # Alternative version giving the same result, somewhat slower but understandable
    # start = time.time()
    old_to_new_gene_name = {old_gene_name: new_gene_names_unq[ind] for ind, old_gene_name in
                            enumerate(old_gene_names)}

    new_list = []
    for old_gene_name in metadata_geneIds:
        if old_gene_name in old_to_new_gene_name:
            new_list.append(old_to_new_gene_name[old_gene_name])
        else:
            new_list.append(old_gene_name)
    return new_list


def change_json_file(new_gene_ids_file, path_to_json, old_geneIDs_col=None, new_geneIDs_col=None):
    print("Read in json file which contains metadata: {}".format(path_to_json))
    with open(path_to_json) as f:
        metadata = json.load(f)

    copy_filename = path_to_json[:-5] + "_{}.json".format(datetime.now().strftime('%y%m%d_%H%M%S'))
    print("\nStored old json file at: {}".format(copy_filename))

    with open(copy_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    metadata['geneIds'] = get_new_gene_ids(new_gene_ids_file, metadata['geneIds'], old_geneIDs_col=old_geneIDs_col,
                                           new_geneIDs_col=new_geneIDs_col)
    # print("Check: there are {} geneIDs in the new metadata.json".format(len(metadata['geneIds'])))

    print("\nSaved new metadata json file in : {}".format(path_to_json))

    with open(path_to_json, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("done")


if __name__ == '__main__':
    """Parse input arguments"""
    parser = ArgumentParser(
        description='Change geneIds in metadata.json file')
    parser.add_argument('--path_to_json', type=str, default='.',
                        help='Path to the metadata.json file. This file will be copied to a filename including a '
                             'timestamp. The new metadata-file will be stored in this location.')
    parser.add_argument('--mapping_file', type=str, default='',
                        help='Path to the file which contains mapping of the geneIDs, in csv')
    parser.add_argument('--geneIDs_col', type=str, default='',
                        help='name of the column that contains the geneIDs')
    parser.add_argument('--new_geneIDs_col', type=str, default='',
                        help='name of the column that contains the new geneIDs')

    args = parser.parse_args()

    change_json_file(new_gene_ids_file=args.mapping_file, path_to_json=args.path_to_json,
                     old_geneIDs_col=args.geneIDs_col, new_geneIDs_col=args.new_geneIDs_col)
