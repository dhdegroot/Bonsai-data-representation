import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt

plt.set_loglevel(level='warning')

import matplotlib
import re
import itertools

from matplotlib.collections import LineCollection, EllipseCollection
from matplotlib import cm
from matplotlib import colors
from matplotlib.patches import Wedge, Rectangle, Circle
from matplotlib.collections import PatchCollection
from sklearn.neighbors import KDTree
import sys, os
import json
import h5py
from natsort import natsorted
# parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'tree_layout'))
# sys.path.append(os.path.join(parent_dir, 'downstream_analyses'))

from bonsai_scout.my_tree_layout import Layout_Tree
from downstream_analyses.calc_marker_genes_helpers import calc_marker_genes_single, calc_marker_genes_double, \
    calc_marker_genes_error_bars, calc_marker_genes_error_bars_approx2
from downstream_analyses.get_clusters_max_diameter import get_max_diam_clustering_from_nwk_str, \
    get_footfall_clustering_from_nwk_str, get_cluster_assignments, get_min_pdists_clustering_from_nwk_str

gray = cm.get_cmap('gray')(0.75)
blackish = (0.08578431372549018, 0.08578428015768168, 0.11935208866155156, 1.0)

import logging

FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT, datefmt='%m-%d %H:%M:%S',
                    level=log_level)


class Celltype_info:
    annot_alts = None
    annot_infos = None

    def __init__(self, cell_info_dict=None, cs_info_dict=None, cluster_info_dict=None, annot_alts=None,
                 annot_infos=None, colortype=None, gradient_type='YlOrRd'):
        if (annot_alts is not None) and (annot_infos is not None):
            self.annot_alts = annot_alts
            self.annot_infos = {}
            for annot, annot_info in annot_infos.items():
                if type(annot_info) is dict:
                    annot_info = Annotation_info(**annot_info)
                self.annot_infos[annot] = annot_info
            return

        self.annot_alts = []
        self.annot_infos = {}

        cmap = get_celltype_colors_new(colortype='gradient_colormap', gradientType=gradient_type)
        n_cells = len(cell_info_dict[next(iter(cell_info_dict))])
        bottom_quantile = np.minimum(10 / n_cells, 0.01)
        top_quantile = 1 - bottom_quantile
        for info_key in list(cell_info_dict.keys()) + list(cs_info_dict.keys()) + list(cluster_info_dict.keys()):
            if info_key in cs_info_dict.keys():
                annot_type = 'cellstates'
                info_object = 'cs_info_dict'
                info_dict_curr = cs_info_dict
                hidden = False
            elif info_key in cluster_info_dict.keys():
                annot_type = 'cellstates'
                info_object = 'cluster_info_dict'
                info_dict_curr = cluster_info_dict
                hidden = True
            else:
                annot_type = 'cells'
                info_object = 'cell_info_dict'
                info_dict_curr = cell_info_dict
                hidden = False
            skip = False
            if info_key.startswith('annot_num_'):
                annot_to_color = None
                all_vals = np.array(info_dict_curr[info_key])
                all_vals_notnan = all_vals[~np.isnan(all_vals)]
                cbar_info = {'cmap': cmap, 'vmin': np.quantile(all_vals_notnan, bottom_quantile, overwrite_input=False),
                             'vmax': np.quantile(all_vals_notnan, top_quantile, overwrite_input=False), 'log': False}
                label = info_key[10:].capitalize()
                cats = None
                color_type = 'sequential'
            elif info_key.startswith('annot_'):
                cats = natsorted(list(np.unique(info_dict_curr[info_key])))
                if 'NaN' in cats:
                    n_celltypes = len(cats) - 1
                else:
                    n_celltypes = len(cats)
                celltype_colors = get_celltype_colors_new(n_celltypes, colortype=colortype)
                annot_to_color = {}
                ind = 0
                for cat in cats:
                    if cat == 'NaN':
                        annot_to_color[cat] = cm.get_cmap('gray')(0.75)
                    else:
                        annot_to_color[cat] = celltype_colors(ind)
                        ind += 1
                # annot_to_color = {cat: celltype_colors(ind) for ind, cat in enumerate(cats)}
                cbar_info = {'cmap': None, 'vmin': None, 'vmax': None, 'log': None}
                label = info_key[6:].capitalize()
                color_type = 'categorical'
            else:
                skip = True
            if not skip:
                self.annot_alts.append(info_key)
                self.annot_infos[info_key] = Annotation_info(cats=cats, annot_to_color=annot_to_color, label=label,
                                                             cbar_info=cbar_info, annot_type=annot_type,
                                                             color_type=color_type, info_object=info_object,
                                                             info_key=info_key, hidden=hidden)

    def to_dict(self):
        self_dict = self.__dict__
        class_dicts = ['annot_infos']
        new_dict = {}
        for label in self_dict:
            if label in class_dicts:
                new_dict[label] = {}
                for class_label, class_inst in self_dict[label].items():
                    new_dict[label][class_label] = class_inst.to_dict()
            else:
                new_dict[label] = self_dict[label]
        return new_dict


class Verttype_info:
    # annot = None
    annot_alts = None
    annot_infos = None
    gradient_type = None

    def __init__(self, vert_info_dict=None, annot_alts=None, annot_infos=None, colortype=None, gradient_type='YlOrRd',
                 new_cell_info_dict=None):
        if (annot_alts is not None) and (annot_infos is not None):
            self.annot_alts = annot_alts
            self.annot_infos = {}
            if new_cell_info_dict is not None:
                self.new_cell_info_dict = {info_label: np.array(info_array) for info_label, info_array in
                                           new_cell_info_dict.items()}
            else:
                self.new_cell_info_dict = {}
            for annot, annot_info in annot_infos.items():
                if type(annot_info) is dict:
                    annot_info = Annotation_info(**annot_info)
                self.annot_infos[annot] = annot_info
            return

        self.annot_alts = []
        self.annot_infos = {}

        # First check if there is any annotation stored in vert_info
        annot_type = 'verts'
        cmap = get_celltype_colors_new(colortype='gradient_colormap', gradientType=gradient_type)
        info_object = 'vert_info_dict'

        n_cells = np.sum(vert_info_dict['n_cells_per_vert'])
        # In the color-map, 10 outliers will get saturated colors. This will benefit the distinguishing of the others
        bottom_quantile = np.minimum(10 / n_cells, 0.01)
        top_quantile = 1 - bottom_quantile
        for info_key in vert_info_dict.keys():
            skip = False
            if info_key.startswith('annot_num_'):
                annot_to_color = None
                all_vals = np.array(vert_info_dict[info_key])
                all_vals_notnan = all_vals[~np.isnan(all_vals)]
                cbar_info = {'cmap': cmap, 'vmin': np.quantile(all_vals_notnan, bottom_quantile, overwrite_input=False),
                             'vmax': np.quantile(all_vals_notnan, top_quantile, overwrite_input=False), 'log': False}
                label = info_key[10:].capitalize()
                cats = None
                color_type = 'sequential'
            elif info_key.startswith('annot_'):
                cats = natsorted(list(np.unique(vert_info_dict[info_key])))
                # n_celltypes = len(cats)
                # celltype_colors = get_celltype_colors_new(n_celltypes, colortype=colortype)
                if 'NaN' in cats:
                    n_celltypes = len(cats) - 1
                else:
                    n_celltypes = len(cats)
                celltype_colors = get_celltype_colors_new(n_celltypes, colortype=colortype)
                annot_to_color = {}
                ind = 0
                for cat in cats:
                    if cat == 'NaN':
                        annot_to_color[cat] = cm.get_cmap('gray')(0.75)
                    else:
                        annot_to_color[cat] = celltype_colors(ind)
                        ind += 1
                # annot_to_color = {cat: celltype_colors(ind) for ind, cat in enumerate(cats)}
                cbar_info = {'cmap': None, 'vmin': None, 'vmax': None, 'log': None}
                label = info_key[6:].capitalize()
                color_type = 'categorical'
            else:
                skip = True
            if not skip:
                self.annot_alts.append(info_key)
                self.annot_infos[info_key] = Annotation_info(cats=cats, annot_to_color=annot_to_color, label=label,
                                                             cbar_info=cbar_info, annot_type=annot_type,
                                                             color_type=color_type, info_object=info_object,
                                                             info_key=info_key)

        # Finally add cell_number which is a standard annotation that we always add
        self.new_cell_info_dict = {}
        if vert_info_dict is not None:
            self.annot_alts.append('cell_number')
            # cs_to_vert = np.array(cs_info_dict['cs_ind_to_vert_ind'])
            n_cells_per_vert = vert_info_dict['n_cells_per_vert']
            self.new_cell_info_dict['cell_number'] = n_cells_per_vert
            # cats = natsorted(list(np.unique(n_cells_per_vert)))
            cbar_info = {'cmap': cmap, 'vmin': 1, 'vmax': max(int(np.nanmax(n_cells_per_vert)), 2), 'log': False}
            self.annot_infos['cell_number'] = Annotation_info(cats=None, annot_to_color=None, label='Number of cells',
                                                              cbar_info=cbar_info, annot_type='verts',
                                                              color_type='sequential', info_object='new_cell_info_dict',
                                                              info_key='cell_number')

            # Also add the default sizing
            self.annot_alts.append('binarized_cell_number')
            self.new_cell_info_dict['binarized_cell_number'] = np.minimum(n_cells_per_vert, 1)
            cbar_info = {'cmap': cmap, 'vmin': 1, 'vmax': max(int(np.nanmax(n_cells_per_vert)), 2), 'log': False}
            self.annot_infos['binarized_cell_number'] = Annotation_info(cats=None, annot_to_color=None,
                                                                        label='Default sizing',
                                                                        cbar_info=cbar_info, annot_type='verts',
                                                                        color_type='sequential',
                                                                        info_object='new_cell_info_dict',
                                                                        info_key='binarized_cell_number')

    def to_dict(self):
        self_dict = self.__dict__
        class_dicts = ['annot_infos']
        new_dict = {}
        for label in self_dict:
            if label in class_dicts:
                new_dict[label] = {}
                for class_label, class_inst in self_dict[label].items():
                    new_dict[label][class_label] = class_inst.to_dict()
            elif label == 'new_cell_info_dict':
                new_dict[label] = {}
                for info_label in self_dict[label]:
                    convertible = self_dict[label][info_label]
                    if convertible.dtype == 'int64':
                        converted = tuple(map(int, convertible))
                    elif convertible.dtype == 'float64':
                        converted = tuple(map(float, convertible))
                    else:
                        converted = convertible
                    new_dict[label][info_label] = converted
            else:
                new_dict[label] = self_dict[label]
        return new_dict


class Annotation_info:
    cats = None
    annot_to_color = None
    label = None
    cbar_info = None
    annot_type = None
    color_type = None
    info_object = None
    info_key = None
    hidden = False

    def __init__(self, cats=None, annot_to_color=None, label=None, cbar_info=None, annot_type=None, color_type=None,
                 info_object=None, info_key=None, hidden=False):
        self.cats = cats
        if annot_to_color is not None:
            self.annot_to_color = {}
            for annot, color in annot_to_color.items():
                self.annot_to_color[annot] = tuple(color)
        self.hidden = hidden
        self.label = label
        self.cbar_info = cbar_info
        if ('cmap' in self.cbar_info) and (type(self.cbar_info['cmap']) is str):
            self.cbar_info['cmap'] = cm.get_cmap(self.cbar_info['cmap'])
        self.annot_type = annot_type
        self.color_type = color_type
        self.info_object = info_object
        self.info_key = info_key

    def to_dict(self):
        self_dict = self.__dict__
        new_dict = {}
        for label in self_dict:
            if label == 'cbar_info':
                new_dict['cbar_info'] = {}
                for cbar_label in self_dict['cbar_info']:
                    if cbar_label == 'cmap':
                        cmap = self_dict['cbar_info']['cmap']
                        if cmap is not None:
                            new_dict['cbar_info']['cmap'] = cmap.name
                        else:
                            new_dict['cbar_info']['cmap'] = None
                    else:
                        new_dict['cbar_info'][cbar_label] = self_dict['cbar_info'][cbar_label]
            else:
                new_dict[label] = self_dict[label]

        return new_dict


def get_feature_annot_info(feature_ind, feature_name, info_object, cells_or_verts='verts', gradient_type=None):
    # Then add information from gene expression data
    annot_to_color = None
    annot_type = cells_or_verts
    cats = None
    color_type = 'sequential'
    info_object = info_object
    if gradient_type is None:
        gradient_type = 'YlOrRd'
    cmap = get_celltype_colors_new(colortype='gradient_colormap', gradientType=gradient_type)
    cbar_info = {'cmap': cmap, 'vmin': None, 'vmax': None, 'log': False}
    label = feature_name
    info_key = 'Feature_{}'.format(feature_ind)
    annot_info = Annotation_info(cats=cats, annot_to_color=annot_to_color, label=label, cbar_info=cbar_info,
                                 annot_type=annot_type, color_type=color_type, info_object=info_object,
                                 info_key=info_key)
    return annot_info


class Bonvis_settings:
    # Transformation into the disk
    transf_info = None
    # Layout type
    ly_type = None
    ly_types = None

    # Plotting styles
    node_style = None
    cell_to_celltype = None
    celltype_info = None
    verttype_info = None
    bg_info = None

    # Possible updates to coords by tuning layout
    upd_node_coords = None
    upd_edge_coords = None

    def __init__(self, bonvis_metadata=None, geometry='hyperbolic',
                 origin_style='root', ly_type=None,
                 load_settings_path=None):
        if load_settings_path is not None:
            self.from_json(load_settings_path)
            return
        # if bonvis_data_hdf_path is not None:
        #     self.bonvis_data = h5py.File(bonvis_data_hdf_path, 'r')
        self.transf_info = Transf_info(geometry=geometry, origin_style=origin_style)
        self.set_ly_type(ly_type)
        feature_path = bonvis_metadata.feature_paths[0] if len(bonvis_metadata.feature_paths) else None
        self.node_style = {'radius_int': 0.005, 'radius_cell': 0.015, 'edgecolor': blackish, 'lw_int': .04,
                           'lw_cell': .08, 'color_int': gray, 'radius_int_data': 0.010,
                           'verts_masked': None, 'cells_masked': None, 'use_mask': False, 'gradient_type': 'YlOrRd',
                           'feature_path': feature_path}
        self.celltype_info = Celltype_info(cell_info_dict=bonvis_metadata.cell_info['cell_info_dict'],
                                           cs_info_dict=bonvis_metadata.cs_info['cs_info_dict'],
                                           cluster_info_dict=bonvis_metadata.cs_info['cluster_info_dict'],
                                           gradient_type=self.node_style['gradient_type'])  # , colortype='tab20')
        self.verttype_info = Verttype_info(vert_info_dict=bonvis_metadata.vert_info,
                                           gradient_type=self.node_style['gradient_type'])
        # if len(self.celltype_info.annot_alts) > 1:
        #     possible_init_annots = [annot for annot in self.celltype_info.annot_alts if annot != 'annot_default']
        possible_init_annots = []
        if len(self.celltype_info.annot_alts) > 1:
            possible_init_annots = [annot for annot in self.celltype_info.annot_alts if annot != 'annot_default']
            annot_tuples = []
            for annot_alt in self.celltype_info.annot_alts:
                if annot_alt == 'annot_default':
                    continue
                if annot_alt[:15] == 'annot_cluster_n':
                    continue
                annot_info_alt = self.celltype_info.annot_infos[annot_alt]
                if annot_info_alt.cats is None:
                    continue
                annot_tuples.append((annot_alt, len(annot_info_alt.cats)))
            if len(annot_tuples) > 0:
                # Select "cellTypeName" first, if not present, "Celltype", then the one with fewest categories
                sorted_annots = sorted(annot_tuples, key=lambda x: (not x[0].lower().startswith("annot_celltypename"),
                                                                    not x[0].lower().startswith("annot_celltype"),
                                                                    x[1]))
                possible_init_annots = [sorted_annots[0][0]]
        init_annot = possible_init_annots[0] if len(possible_init_annots) else self.celltype_info.annot_alts[0]
        self.set_annot(annot_label=init_annot)
        self.set_size_style(annot_label='cell_number')
        self.edge_style = {'color': gray, 'linewidth': .25}
        self.upd_node_coords = {}
        self.upd_edge_coords = {}

    def to_json(self, settings_path):
        self_dict = self.to_dict()
        with open(settings_path, "w") as json_file:
            logging.info("Storing settings in {}.".format(settings_path))
            json.dump(self_dict, json_file, indent=4)

    def from_json(self, settings_path):
        with open(settings_path, 'r') as json_file:
            self_dict = json.load(json_file)
        self.from_dict(self_dict)

    def to_dict(self):
        self_dict = self.__dict__
        class_attribs = ['transf_info', 'celltype_info', 'verttype_info']
        subclass_attribs = {'node_style': ['annot_info', 'size_annot_info']}
        np_dicts = ['upd_node_coords', 'upd_edge_coords']
        new_dict = {}
        for label in self_dict:
            if label in subclass_attribs:
                new_dict[label] = {}
                for sub_label in self_dict[label]:
                    if sub_label in subclass_attribs[label]:
                        new_dict[label][sub_label] = self_dict[label][sub_label].to_dict()
                    else:
                        new_dict[label][sub_label] = self_dict[label][sub_label]
            elif label in class_attribs:
                new_dict[label] = self_dict[label].to_dict()
            elif label in np_dicts:
                new_dict[label] = {}
                for np_label, np_array in self_dict[label].items():
                    new_dict[label][np_label] = np_array.tolist()
            else:
                new_dict[label] = self_dict[label]

        return new_dict

    def from_dict(self, json_dict):
        np_dicts = ['upd_node_coords', 'upd_edge_coords']
        for label in json_dict:
            if label == 'transf_info':
                self.transf_info = Transf_info(**json_dict[label])
            elif label == 'celltype_info':
                self.celltype_info = Celltype_info(**json_dict[label])
            elif label == 'verttype_info':
                self.verttype_info = Verttype_info(**json_dict[label])
            elif label == 'node_style':
                self.node_style = json_dict[label]
                self.node_style['annot_info'] = Annotation_info(**self.node_style['annot_info'])
                self.node_style['size_annot_info'] = Annotation_info(**self.node_style['size_annot_info'])
            elif label in np_dicts:
                converted_dict = {}
                for np_label, np_list in json_dict[label].items():
                    converted_dict[np_label] = np.array(np_list)
                setattr(self, label, converted_dict)
            else:
                setattr(self, label, json_dict[label])

    def set_ly_type(self, ly_type):
        self.ly_type = ly_type
        if self.transf_info is not None:
            if ly_type in ['ly_dendrogram', 'ly_dendrogram_ladderized']:
                self.transf_info.no_transform = True
                self.transf_info.geometry = 'flat'
            else:
                self.transf_info.no_transform = False

    def set_annot(self, annot_label=None, annot_info=None):
        if (annot_label is not None) and (annot_label in self.celltype_info.annot_alts):
            self.node_style['annot_info'] = self.celltype_info.annot_infos[annot_label]
        elif (annot_label is not None) and (annot_label in self.verttype_info.annot_alts):
            self.node_style['annot_info'] = self.verttype_info.annot_infos[annot_label]
        elif annot_info is not None:
            self.node_style['annot_info'] = annot_info
        else:
            logging.debug("{} is not a known annotation. Returning {} instead.".format(annot_label,
                                                                                       self.celltype_info.annot_alts[
                                                                                           0]))
            annot = self.celltype_info.annot_alts[0]
            # self.celltype_info.annot = annot
            self.node_style['annot_info'] = self.celltype_info.annot_infos[annot]

    def set_size_style(self, annot_info=None, annot_label=None):
        if annot_info is not None:
            self.node_style['size_annot_info'] = annot_info
        elif (annot_label is not None) and (annot_label in self.celltype_info.annot_alts):
            annot_info = self.celltype_info.annot_infos[annot_label]
            if annot_info.color_type == 'sequential':
                self.node_style['size_annot_info'] = self.celltype_info.annot_infos[annot_label]
        elif (annot_label is not None) and (annot_label in self.verttype_info.annot_alts):
            annot_info = self.verttype_info.annot_infos[annot_label]
            if annot_info.color_type == 'sequential':
                self.node_style['size_annot_info'] = annot_info
        else:
            logging.debug("{} is not a known annotation. Returning {} instead.".format(annot_label, 'cell_number'))
            annot_info = self.verttype_info.annot_infos['cell_number']
            self.node_style['size_annot_info'] = annot_info


# class WedgeDataUnits(Wedge):
#     def __init__(self, *args, **kwargs):
#         _r_data = kwargs.get("r")
#         super().__init__(*args, **kwargs)
#         self._r_data = _r_data
#
#     def _get_r(self):
#         if self.axes is not None:
#             ppd = 72./self.axes.figure.dpi
#             trans = self.axes.transData.transform
#             return ((trans((1, self._r_data))-trans((0, 0)))*ppd)[1]
#         else:
#             return 1
#
#     def _set_r(self, lw):
#         self._r_data = lw
#
#     _r = property(_set_r, _set_r)


class Bonvis_figure:
    bonvis_data = None
    bonvis_data_hdf_path = None
    bonvis_settings = None
    ly_tree = None

    # Coordinate information
    coords_info = None

    # Plotting collections
    bg_obj = None
    int_obj = None
    single_cell_obj = None
    single_cs_obj = None
    single_vert_obj = None
    multi_cell_obj = None
    multi_cs_obj = None
    edge_obj = None

    # Keep track of what is plotted, so that it can be removed when necessary
    is_present = None

    # Figure objects
    fig = None
    ax = None

    fig_leg_df = None
    fig_cbar = None

    marker_genes_df = None
    # number_of_clusters_found = None  # NEW sarah

    def __init__(self, bonvis_data_hdf, bonvis_metadata, bonvis_data_path=None, bonvis_settings=None):
        # Plotting collections
        self.bg_obj = {'flat': {'coll': None, 'patches': None},
                       'hyperbolic': {'coll': None, 'patches': None, 'ax_lims': None}}
        self.int_obj = {'coll': None, 'vert_inds': None}
        self.single_cell_obj = {'coll': None, 'obj_inds': None, 'vert_inds': None}
        self.single_vert_obj = {'coll': None, 'obj_inds': None, 'vert_inds': None}
        self.multi_cell_obj = {'coll': None, 'obj_inds': None, 'vert_inds': None}
        self.single_cs_obj = {'coll': None, 'obj_inds': None, 'vert_inds': None}
        self.multi_cs_obj = {'coll': None, 'obj_inds': None, 'vert_inds': None}
        self.single_obj = None
        self.multi_obj = None
        self.edge_obj = {'coll': None}
        self.vert_to_size = None

        # Keep track of what is plotted, so that it can be removed when necessary
        self.is_present = {'nodes': False, 'edges': False, 'bg': None}

        self.bonvis_data = bonvis_data_hdf
        self.bonvis_data_path = bonvis_data_path
        self.bonvis_settings = bonvis_settings if (bonvis_settings is not None) else Bonvis_settings(
            bonvis_metadata=bonvis_metadata)
        self.marker_genes_df = pd.DataFrame(columns=['marker_genes', 'marker_scores'])
        self.bonvis_metadata = bonvis_metadata
        # Transform node- and edge-coords for plotting
        # Also: store different coords-sets for different layouts
        # Also: reconstruct tree-object from nwk-file to allow for tuning layouts
        self.coords_info = Coords_info(self.bonvis_settings, self.get_bonvis_data(), self.bonvis_metadata)

        # Create background components
        self.get_bg_collection()

        # Create node collection
        self.get_node_collection()

        # Create edge collection
        self.get_edge_collection()

        # self.create_legend()

        # self.create_figure()

    def get_bonvis_data(self):
        if not self.bonvis_data:
            self.bonvis_data = h5py.File(self.bonvis_data_path, 'r')
        return self.bonvis_data

    def get_edge_collection(self):
        edge_style = self.bonvis_settings.edge_style
        edge_coords = self.coords_info.transf_edge_coords_eix
        self.edge_obj['coll'] = LineCollection(edge_coords, colors=edge_style['color'],
                                               linewidths=edge_style['linewidth'], zorder=-1)

    def get_bg_collection(self):
        # Create background for hyperbolic geometry
        gray_cm = cm.get_cmap('gray')

        if self.bonvis_settings.transf_info.geometry == 'hyperbolic':
            # Get interval at which we are going to place lines
            first_line = 0.4
            n_lines = 5
            n_coords = 20

            self.coords_info.get_bg_coords(self.bonvis_settings.transf_info, first_line=first_line, n_lines=n_lines,
                                           n_coords=n_coords)

            line_coll = LineCollection(self.coords_info.transf_bg_edge_coords_eix, colors=gray_cm(0.75), linewidths=.3,
                                       linestyles='--', zorder=-2)
            plotting_space = Circle(xy=(0, 0), radius=1, fill=True, facecolor='white', edgecolor=gray_cm(0.7),
                                    zorder=-3)

            # shade_coll = PatchCollection([plotting_space, overflow_space], match_original=True)
            self.bg_obj['hyperbolic']['patches'] = [plotting_space]
            self.bg_obj['hyperbolic']['coll'] = [line_coll]
        elif self.bonvis_settings.transf_info.geometry == 'flat':
            # Create background for flat geometry
            plotting_space = Rectangle(xy=(-1, -1), width=2, height=2, fill=True, facecolor='white',
                                       edgecolor=gray_cm(0.7),
                                       zorder=-2)
            overflow_space = Rectangle(xy=(-1.05, -1.05), width=2.1, height=2.1, zorder=-3, fill=True,
                                       facecolor=gray_cm(0.95),
                                       edgecolor=None)
            # shade_coll = PatchCollection([plotting_space, overflow_space], match_original=True)
            self.bg_obj['flat']['patches'] = [plotting_space, overflow_space]
            self.bg_obj['flat']['coll'] = []

    def get_node_collection(self):
        annot_info = self.bonvis_settings.node_style['annot_info']
        size_annot_info = self.bonvis_settings.node_style['size_annot_info']

        if annot_info.annot_type == 'cells':
            self.single_obj = self.single_cell_obj
            self.multi_obj = self.multi_cell_obj
            plot_unit = 'cells'
        elif annot_info.annot_type == 'cellstates':
            self.single_obj = self.single_cs_obj
            self.multi_obj = self.multi_cs_obj
            plot_unit = 'cellstates'
        else:
            plot_unit = 'verts'
            self.single_obj = self.single_vert_obj
            self.multi_obj = None

        # Some settings
        node_style = self.bonvis_settings.node_style
        node_coords = self.coords_info.transf_node_coords_nx
        cell_info_dict = self.bonvis_metadata.cell_info['cell_info_dict']
        cs_info_dict = self.bonvis_metadata.cs_info['cs_info_dict']
        # cluster_info_dict = self.bonvis_metadata.cs_info['cluster_info_dict']
        cell_to_vert = np.array(cell_info_dict['cell_ind_to_vert_ind'])
        cs_to_vert = np.array(cs_info_dict['cs_ind_to_vert_ind'])

        # Get information on verts with no cell
        self.int_obj['vert_inds'] = self.bonvis_metadata.cell_info['int_vert_inds']
        self.int_obj['vert_inds_complement'] = self.bonvis_metadata.cell_info['non_int_vert_inds']

        #Get some information on verts with multiple cells
        # if annot_is_cell:
        #     self.multi_obj['obj_inds'] = self.bonvis_metadata.cell_info['multi_at_vert']
        #     self.multi_obj['vert_inds'] = cell_to_vert[self.multi_obj['obj_inds']]
        #     self.multi_obj['obj_ind_to_vert_ind'] = {self.multi_obj['obj_inds'][ind]: self.multi_obj['vert_inds'][ind] for ind in range(len(self.multi_obj['vert_inds']))}
        # else:
        #     self.multi_obj['obj_inds'] = self.bonvis_metadata.cs_info['multi_cs_at_vert']
        #     self.multi_obj['vert_inds'] = cs_to_vert[self.multi_obj['obj_inds']]
        #     self.multi_obj['obj_ind_to_vert_ind'] = {self.multi_obj['obj_inds'][ind]: self.multi_obj['vert_inds'][ind] for ind in range(len(self.multi_obj['vert_inds']))}

        obj_to_celltype, obj_to_color = self.get_color_info(annot_info=annot_info)
        self.vert_to_size = self.get_size_info(size_annot_info=size_annot_info, annot_info=annot_info)

        # We first make nodes for vertices without any cell
        radii = self.vert_to_size[self.int_obj['vert_inds']]
        if plot_unit in ['cells', 'cellstates']:
            facecolors = node_style['color_int']
        elif plot_unit == 'verts':
            facecolors = obj_to_color[self.int_obj['vert_inds']]
        self.int_obj['coll'] = EllipseCollection(radii, radii, 0,
                                                 units='width', offsets=node_coords[self.int_obj['vert_inds'], :],
                                                 facecolors=facecolors,
                                                 edgecolors=node_style['edgecolor'],
                                                 linewidths=node_style['lw_int'])

        # Get some information on verts with one cell
        if plot_unit == 'cells':
            self.single_obj['obj_inds'] = self.bonvis_metadata.cell_info['single_at_vert']
            self.single_obj['vert_inds'] = cell_to_vert[self.single_obj['obj_inds']]
            # In this case self.single_obj['obj_inds'] gives me the cell-inds corresponding to the cells
            # at nodes with only one cell
            facecolors = obj_to_color[self.single_obj['obj_inds']]
            # elif annot_info.annot_type=='cellstates':
            #     # Now we have to map the obj_inds (cell-inds) to cs_inds and then into cell_to_color
            #     if not hasattr(self.bonvis_metadata, 'cell_ind_to_cs_ind_array'):
            #         self.bonvis_metadata.cell_ind_to_cs_ind_array = np.zeros(len(self.bonvis_metadata.cell_ind_to_cs_ind))
            #         for cell_ind, cs_ind in self.bonvis_metadata.cell_ind_to_cs_ind.items():
            #             self.bonvis_metadata.cell_ind_to_cs_ind_array[cell_ind] = cs_ind
            #     cs_inds_corr_to_cells = self.bonvis_metadata.cell_ind_to_cs_ind_array[self.single_obj['obj_inds']]
            #     facecolors = cell_to_color[cs_inds_corr_to_cells]
            # elif annot_info.annot_type=='verts':
            #     # Now we have to map the obj_inds (cell-inds) to vert_inds and then into cell_to_color
            #     vert_inds_corr_to_cells = cell_to_vert[self.single_obj['obj_inds']]
            #     facecolors = cell_to_color[vert_inds_corr_to_cells]
        elif plot_unit == 'cellstates':
            self.single_obj['obj_inds'] = self.bonvis_metadata.cs_info['single_cs_at_vert']
            self.single_obj['vert_inds'] = cs_to_vert[self.single_obj['obj_inds']]
            # In this case self.single_obj['obj_inds'] gives me the cs-inds corresponding to the cellstates
            # at nodes with only one cellstate
            # if annot_info.annot_type=='cellstates':
            facecolors = obj_to_color[self.single_obj['obj_inds']]
            # elif annot_info.annot_type=='verts':
            #     # Now we have to map the obj_inds (cs-inds) to vert_inds and then into cell_to_color
            #     vert_inds_corr_to_cells = cs_to_vert[self.single_obj['obj_inds']]
            #     facecolors = cell_to_color[vert_inds_corr_to_cells]
        elif plot_unit == 'verts':
            # In this case self.single_obj['obj_inds'] gives me the vert-inds corresponding to the vertices
            # that have at least one cell
            self.single_obj['obj_inds'] = self.int_obj['vert_inds_complement']
            self.single_obj['vert_inds'] = self.int_obj['vert_inds_complement']
            facecolors = obj_to_color[self.single_obj['obj_inds']]

        radii = self.vert_to_size[self.single_obj['vert_inds']]
        offsets = node_coords[self.single_obj['vert_inds'], :]


        # Then create collection for verts with single cell
        # if annot_is_cell in ['cells']:
            # In this case self.single_obj['obj_inds'] gives me the cell-inds corresponding to the cells
            # at nodes with only one cell

        # elif annot_is_cell in ['cellstates']:
            # In this case self.single_obj['obj_inds'] gives me the cs-inds corresponding to the cellstates
            # at nodes with only one cellstate
            # radii = self.vert_to_size[self.single_obj['vert_inds']]
            # offsets = node_coords[self.single_obj['vert_inds'], :]
            # facecolors = cell_to_color[self.single_obj['obj_inds']]
        # elif annot_is_cell in ['verts']:
        #     # In this case self.single_obj['obj_inds'] gives me the vert-inds corresponding to the vertices
        #     # that have at least one cell
        #     radii = self.vert_to_size[self.single_obj['vert_inds']]
        #     offsets = node_coords[self.single_obj['vert_inds'], :]
        #     facecolors = cell_to_color[self.single_obj['obj_inds']]
        # elif annot_info.annot_type == 'verts':
        #     radii = self.vert_to_size[self.int_obj['vert_inds_complement']]
        #     offsets = node_coords[self.int_obj['vert_inds_complement']]
        #     facecolors = cell_to_color[self.int_obj['vert_inds_complement']]
        self.single_obj['coll'] = EllipseCollection(radii, radii, 0, units='width', offsets=offsets,
                                                    facecolors=facecolors, edgecolors=node_style['edgecolor'],
                                                    linewidths=node_style['lw_cell'], zorder=5)

        if plot_unit != 'verts':
            self.plot_multi_obj_verts(node_coords=node_coords, obj_to_color=obj_to_color,
                                    node_style=node_style, obj_to_celltype=obj_to_celltype,
                                    annot_type=annot_info.annot_type)

    def plot_multi_obj_verts(self, node_coords=None, obj_to_color=None,
                             node_style=None, obj_to_celltype=None,
                             annot_type=None):
        # cell_info_dict = self.bonvis_metadata.cell_info['cell_info_dict']
        # cs_info_dict = self.bonvis_metadata.cs_info['cs_info_dict']

        # if annot_is_cell:
        #     n_obj_per_vert = self.bonvis_metadata.vert_info['n_cells_per_vert']
        # else:
        #     n_obj_per_vert = self.bonvis_metadata.vert_info['n_css_per_vert']

        # Get some information on special cases where we have more than one cell per vert
        if annot_type == 'cells':
            # obj_inds_at_multi = self.bonvis_metadata.cell_info['multi_at_vert']
            verts_w_multi_objs = self.bonvis_metadata.tree_info['multi_cell_inds']
            vert_ind_to_obj_inds = self.bonvis_metadata.vert_info['vert_ind_to_cell_inds']
            # obj_ind_to_vert_ind = np.array(cell_info_dict['cell_ind_to_vert_ind'])
        elif annot_type == 'cellstates':
            # obj_inds_at_multi = self.bonvis_metadata.cs_info['multi_cs_at_vert']
            verts_w_multi_objs = self.bonvis_metadata.tree_info['multi_cs_inds']
            vert_ind_to_obj_inds = self.bonvis_metadata.vert_info['vert_ind_to_cs_inds']
        elif annot_type == 'verts':
            exit("Annotation-type is verts, but still we have multiple annotation at one vertex. This"
                 "should never happen. Please report this to Daan de Groot.")
            # obj_inds_at_multi = self.bonvis_metadata.cs_info['multi_cs_at_vert']
            # verts_w_multi_objs = self.bonvis_metadata.tree_info['multi_cs_inds']
            # obj_ind_to_vert_ind = np.array(cs_info_dict['cs_ind_to_vert_ind'])
        # vert_inds = obj_to_vert[obj_inds]
        # n_obj_at_vert = n_obj_per_vert[vert_inds]
        # degrees_wedge = 360 / n_obj_at_vert
        vert_inds_per_wedge = []
        all_wedges = []
        all_facecolors = []
        all_coords = []

        for vert_ind in verts_w_multi_objs:
            # Find radius for vertex
            # We divide by two for multi-obj-verts, since the Wedge-object wants a radius, while Ellipse wants a height and width
            radius = self.vert_to_size[vert_ind] / 2
            # Find coordinates for vertex
            coords = node_coords[vert_ind]
            # Find all objects mapping to this vert
            if annot_type != 'verts':
                obj_inds_at_this_vert = np.array(vert_ind_to_obj_inds[str(vert_ind)])
            else:
                obj_inds_at_this_vert = np.array([vert_ind])
            n_objs = len(obj_inds_at_this_vert)

            # Find all colors for these objects
            # if annot_is_cell:
            these_cats = obj_to_celltype[obj_inds_at_this_vert]
            facecolors = obj_to_color[obj_inds_at_this_vert]
            # elif annot_info.annot_type == 'verts':
            #     these_cats = cell_to_celltype[obj_inds_at_this_vert]
            #     facecolors = cell_to_color[obj_inds_at_this_vert]
            # Get unique colors and their counts
            cats, orig_inds, counts = np.unique(these_cats, return_index=True, return_counts=True)
            deg0 = 0
            for count_ind, count in enumerate(counts):
                color = facecolors[orig_inds[count_ind]]
                degf = deg0 + 360 * (count / n_objs)

                # Add wedge to set
                vert_inds_per_wedge.append(vert_ind)
                all_wedges.append(Wedge(center=coords, r=radius, theta1=deg0, theta2=degf, zorder=6))
                all_facecolors.append(color)
                all_coords.append(coords)
                deg0 = degf

        kwargs = {'match_original': False, 'facecolors': all_facecolors,
                  'edgecolors': node_style['edgecolor'], 'linewidths': node_style['lw_cell'], 'zorder': 6}
        self.multi_obj['kwargs'] = kwargs
        self.multi_obj['wedges'] = all_wedges
        # We store the coordinates of the center of each wedge in data-coordinates. At plotting time, we will still do the transform to
        # axes-coordinates. 
        self.multi_obj['coords'] = all_coords
        self.multi_obj['vert_inds_per_wedge'] = vert_inds_per_wedge

        # For the multi-object collections, we only make the collection at plotting time, so we don't do it here anymore
        # self.multi_obj['coll'] = PatchCollection(self.multi_obj['wedges'],
        #                                          **self.multi_obj['kwargs'])
        return

        # # Finally create collection for verts with multiple cells/cellstates
        # radii = vert_to_size[self.multi_obj['vert_inds']]
        # degrees_0 = {}
        # if self.multi_obj['no_wedges']:
        #     self.multi_obj['offsets'] = np.zeros((len(self.multi_obj['vert_inds']), 2))
        #     size_offset = (2 / self.bonvis_metadata.tree_info['n_leafs']) * .8
        # else:
        #     wedges = []
        # for ind, c_ind in enumerate(self.multi_obj['obj_inds']):
        #     v_ind = self.multi_obj['vert_inds'][ind]
        #     if v_ind in degrees_0:
        #         deg0 = degrees_0[v_ind]
        #     else:
        #         deg0 = 90
        #     degf = deg0 + self.multi_obj['degrees_wedge'][ind]
        #     if not self.multi_obj['no_wedges']:
        #         wedge = Wedge(center=node_coords[v_ind], r=radii[ind], theta1=deg0, theta2=degf, zorder=6)
        #         wedges.append(wedge)
        #     else:
        #         deg0_rad = np.radians(deg0)
        #
        #         self.multi_obj['offsets'][ind, :] = size_offset * np.array([np.cos(deg0_rad), np.sin(deg0_rad)])
        #     degrees_0[v_ind] = degf
        #
        # if not self.multi_obj['no_wedges']:
        #     if annot_info.annot_type in ['cells', 'cellstates']:
        #         facecolors = cell_to_color[self.multi_obj['obj_inds']]
        #     elif annot_info.annot_type == 'verts':
        #         facecolors = cell_to_color[self.multi_obj['vert_inds']]
        #     kwargs = {'match_original': False, 'facecolors': facecolors,
        #               'edgecolors': node_style['edgecolor'], 'linewidths': node_style['lw_cell'], 'zorder': 6}
        #     self.multi_obj['kwargs'] = kwargs
        #     self.multi_obj['wedges'] = wedges
        #     self.multi_obj['coll_with_wedges'] = PatchCollection(self.multi_obj['wedges'],
        #                                                          **self.multi_obj['kwargs'])
        #     self.multi_obj['coll'] = self.multi_obj['coll_with_wedges']
        # else:
        #     multi_coords = node_coords[self.multi_obj['vert_inds'], :] + self.multi_obj['offsets']
        #     if annot_info.annot_type == 'cells':
        #         facecolors = cell_to_color[self.multi_obj['obj_inds']]
        #         facecolors[:, 3] = 0.50
        #     elif annot_info.annot_type == 'verts':
        #         facecolors = cell_to_color[self.multi_obj['vert_inds']]
        #     facecolors[:, 3] = 0.50
        #     self.multi_obj['coll_wo_wedges'] = EllipseCollection(radii, radii, 0,
        #                                                          units='width',
        #                                                          offsets=multi_coords,
        #                                                          facecolors=facecolors,
        #                                                          edgecolors=node_style['edgecolor'],
        #                                                          linewidths=node_style['lw_cell'], zorder=6,
        #                                                          linestyles='dashed')
        #     self.multi_obj['coll'] = self.multi_obj['coll_wo_wedges']

    def get_size_info(self, size_annot_info=None, annot_info=None, ):
        # Get size information for all verts
        max_rescaling = 3.5

        size_annot = size_annot_info.info_key
        cell_info_dict = self.bonvis_metadata.cell_info['cell_info_dict']
        cs_info_dict = self.bonvis_metadata.cs_info['cs_info_dict']
        node_style = self.bonvis_settings.node_style
        cell_to_vert = np.array(cell_info_dict['cell_ind_to_vert_ind'])
        cs_to_vert = np.array(cs_info_dict['cs_ind_to_vert_ind'])

        # First determine size of an internal node, which is determined by color-style
        if annot_info.annot_type in ['cells', 'cellstates']:
            radius_internal = node_style['radius_int']
        elif annot_info.annot_type == 'verts':
            radius_internal = node_style['radius_int_data']
        if size_annot_info.info_object == 'cell_info_dict':
            vert_to_size = np.ones(self.bonvis_metadata.n_nodes) * radius_internal
            cell_to_size = np.array(cell_info_dict[size_annot])
            non_nan_mask = ~np.isnan(cell_to_size)
            cell_to_size_non_nan = cell_to_size[non_nan_mask]
            max_cell_size = np.max(cell_to_size_non_nan)
            min_cell_size = np.min(cell_to_size_non_nan)
            norm_size = np.ones_like(cell_to_size) * node_style['radius_cell']
            if max_cell_size != min_cell_size:
                norm_size[non_nan_mask] = node_style['radius_cell'] * (1 + np.sqrt(
                    (cell_to_size_non_nan - min_cell_size) / (max_cell_size - min_cell_size)) * (max_rescaling - 1))
            vert_to_size[cell_to_vert] = norm_size
        elif size_annot_info.info_object == 'cs_info_dict':
            vert_to_size = np.ones(self.bonvis_metadata.n_nodes) * radius_internal
            cs_to_size = np.array(cs_info_dict[size_annot])
            non_nan_mask = ~np.isnan(cs_to_size)
            cs_to_size_non_nan = cs_to_size[non_nan_mask]
            max_cell_size = np.max(cs_to_size_non_nan)
            min_cell_size = np.min(cs_to_size_non_nan)
            norm_size = np.ones_like(cs_to_size) * node_style['radius_cell']
            if max_cell_size != min_cell_size:
                norm_size[non_nan_mask] = node_style['radius_cell'] * (1 + np.sqrt(
                    (cs_to_size_non_nan - min_cell_size) / (max_cell_size - min_cell_size)) * (max_rescaling - 1))
            vert_to_size[cs_to_vert] = norm_size
        elif size_annot_info.info_object == 'new_cell_info_dict':
            # This holds only the number of cells at the moment. Don't do calculations including the zeros of the non-cell-nodes
            vert_to_size_raw = self.bonvis_settings.verttype_info.new_cell_info_dict[size_annot]
            vert_to_size = np.ones(self.bonvis_metadata.n_nodes) * radius_internal

            non_int_vert_to_size = vert_to_size_raw[self.bonvis_metadata.cell_info['non_int_vert_inds']]
            max_vert_size = np.max(non_int_vert_to_size)
            min_vert_size = np.min(non_int_vert_to_size)
            if max_vert_size < 8:
                max_rescaling = 1.5
            else:
                max_rescaling = min(np.sqrt(max_vert_size/min_vert_size), 8)
            # if len(np.unique(vert_to_size_raw)) == 2:
            #     max_rescaling = 1
            if max_vert_size != min_vert_size:
                vert_to_size[self.bonvis_metadata.cell_info['non_int_vert_inds']] = node_style['radius_cell'] * (
                            1 + np.sqrt(
                        (non_int_vert_to_size - min_vert_size) / (max_vert_size - min_vert_size)) * (max_rescaling - 1))
                # vert_to_size[self.bonvis_metadata.cell_info['non_int_vert_inds']] = node_style['radius_cell'] * (
                #             1 + np.log(1 + (np.exp(1) - 1) * 
                #         (non_int_vert_to_size - min_vert_size) / (max_vert_size - min_vert_size)) * (max_rescaling - 1))
            else:
                vert_to_size[self.bonvis_metadata.cell_info['non_int_vert_inds']] = np.ones_like(non_int_vert_to_size) * \
                                                                                    node_style['radius_cell']
        elif size_annot_info.info_object == 'vert_info_dict':
            vert_info_dict = self.bonvis_metadata.vert_info
            vert_to_size_raw = vert_info_dict[size_annot]
            if len(np.unique(vert_to_size_raw)) == 2:
                max_rescaling = 1
            max_vert_size = vert_to_size_raw.max()
            min_vert_size = vert_to_size_raw.min()
            if max_vert_size != min_vert_size:
                vert_to_size = radius_internal * (1 + np.sqrt(
                    (vert_to_size_raw - max_vert_size) / (max_vert_size - min_vert_size)) * (
                                                          max_rescaling * node_style[
                                                      'radius_cell'] / radius_internal - 1))
            else:
                vert_to_size = np.ones_like(vert_to_size_raw) * radius_internal
        return vert_to_size

    def get_color_info(self, annot_info=None):
        cell_info_dict = self.bonvis_metadata.cell_info['cell_info_dict']
        cs_info_dict = self.bonvis_metadata.cs_info['cs_info_dict']
        cluster_info_dict = self.bonvis_metadata.cs_info['cluster_info_dict']
        annot = annot_info.info_key
        node_style = self.bonvis_settings.node_style
        # Get color information for all cells or all verts
        if annot_info.info_object == 'cell_info_dict':
            cell_to_celltype = cell_info_dict[annot]
        elif annot_info.info_object == 'cs_info_dict':
            cell_to_celltype = cs_info_dict[annot]
        elif annot_info.info_object == 'cluster_info_dict':
            cell_to_celltype = cluster_info_dict[annot]
        elif annot_info.info_object == 'vert_info_dict':
            vert_info_dict = self.bonvis_metadata.vert_info
            cell_to_celltype = vert_info_dict[annot]
        elif annot_info.info_object == 'new_cell_info_dict':
            cell_to_celltype = self.bonvis_settings.verttype_info.new_cell_info_dict[annot]
        elif annot_info.info_object[:5] == 'data/':
            feature_ind = int(annot[8:])
            cell_to_celltype = self.bonvis_data[annot_info.info_object]['means'][feature_ind, :]
        else:
            logging.error("Could not find this cell-to-celltype mapping ({}) anywhere".format(annot))
            return None, None
        cell_to_celltype = np.array(cell_to_celltype)

        if annot_info.color_type == 'categorical':
            nan_entries = np.where(cell_to_celltype == 'NaN')[0]
            cell_to_color = np.array([annot_info.annot_to_color[cell_to_celltype[cell]] for cell in
                                      range(len(cell_to_celltype))])
            cell_to_color[nan_entries, :] = node_style['color_int']
        elif annot_info.color_type == 'sequential':
            cbar_info = annot_info.cbar_info
            cell_to_color = np.zeros((len(cell_to_celltype), 4))
            nan_entries = np.isnan(cell_to_celltype)
            cell_to_celltype_notnan = cell_to_celltype[~nan_entries]
            if cbar_info['vmin'] is None:
                bottom_quantile = np.minimum(10 / len(cell_to_celltype), 0.03)
                top_quantile = 1 - bottom_quantile
                cbar_info['vmin'] = np.quantile(cell_to_celltype_notnan, bottom_quantile, overwrite_input=False)
                cbar_info['vmax'] = np.quantile(cell_to_celltype_notnan, top_quantile, overwrite_input=False)
                # When these two values are the same, take the normal min and max
                if cbar_info['vmin'] == cbar_info['vmax']:
                    cbar_info['vmin'] = cell_to_celltype_notnan.min()
                    cbar_info['vmax'] = cell_to_celltype_notnan.max()
            # If the min and max are still the same, all values are apparently the same. We make up some min and max
            if cbar_info['vmin'] == cbar_info['vmax']:
                cbar_info['vmin'] -= 1e-6
                cbar_info['vmax'] += 1e-6
            min_val = cbar_info['vmin']
            max_val = cbar_info['vmax']

            cell_to_colorvals_notnan = (cell_to_celltype_notnan - min_val) / (max_val - min_val)
            cell_to_color_notnan = cbar_info['cmap'](cell_to_colorvals_notnan)
            cell_to_color[~nan_entries, :] = cell_to_color_notnan
            cell_to_color[nan_entries, :] = cm.get_cmap('gray')(0.95)
        else:
            logging.debug("Don't know what color to give, choosing gray.")
            cell_to_celltype = [np.nan] * self.bonvis_metadata.n_cells
            cell_to_color = np.array([cm.get_cmap('gray')] * self.bonvis_metadata.n_cells)

        if ((node_style['verts_masked'] is not None) or (node_style['cells_masked'] is not None)) and node_style[
            'use_mask']:
            alpha = 0.25
            masked_color = list(cm.get_cmap('gray')(0.95))
            masked_color[-1] = alpha
            masked_color = tuple(masked_color)
            if annot_info.annot_type == 'cells':
                # logging.debug("Annotation type is cells!")
                if node_style['cells_masked'] is None:
                    vert_info_dict = self.bonvis_metadata.vert_info
                    node_style['cells_masked'] = []
                    for vert in node_style['verts_masked']:
                        node_style['cells_masked'] += vert_info_dict['vert_ind_to_cell_inds'][str(vert)]
                cell_to_celltype[node_style['cells_masked']] = np.nan
                cell_to_color[node_style['cells_masked']] = masked_color
            elif annot_info.annot_type == 'cellstates':
                # logging.debug("Annotation type is cellstates!")
                css_masked = []
                if node_style['verts_masked'] is not None:
                    vert_info_dict = self.bonvis_metadata.vert_info
                    for vert in node_style['verts_masked']:
                        css_masked += vert_info_dict['vert_ind_to_cs_inds'][str(vert)]
                elif node_style['cells_masked'] is not None:
                    vert_info_dict = self.bonvis_metadata.vert_info
                    for cell in node_style['cells_masked']:
                        css_masked += [self.bonvis_metadata.cell_ind_to_cs_ind[str(cell)]]
                    css_masked = list(np.unique(css_masked))
                cell_to_celltype = cell_to_celltype.astype(str)
                cell_to_celltype[css_masked] = np.nan
                cell_to_color[css_masked] = masked_color
            else:
                if node_style['verts_masked'] is None:
                    verts_masked = []
                    if (node_style['cells_masked'] is not None) and (len(node_style['cells_masked']) != 0):
                        cell_ind_to_vert_ind = np.array(
                            self.bonvis_metadata.cell_info['cell_info_dict']['cell_ind_to_vert_ind'])
                        verts_masked = list(cell_ind_to_vert_ind[np.array(node_style['cells_masked'])])
                        verts_masked = np.union1d(verts_masked, self.bonvis_metadata.cell_info['int_vert_inds'])
                    elif node_style['cells_masked'] is not None: # avoid problematic array indexing for empty mask
                        verts_masked = self.bonvis_metadata.cell_info['int_vert_inds']
                    cell_to_celltype = cell_to_celltype.astype(str)
                    cell_to_celltype[verts_masked] = np.nan
                    cell_to_color[verts_masked] = masked_color
                else:
                    cell_to_celltype = cell_to_celltype.astype(str)
                    cell_to_celltype[node_style['verts_masked']] = np.nan
                    cell_to_color[node_style['verts_masked']] = masked_color
        return cell_to_celltype, cell_to_color

    def reset_figure(self, reset_path):
        # First check if settings should be reset
        logging.debug("reset_path {}".format(reset_path))
        # with open(reset_path, "rb") as f:
        #     bonvis_settings = pickle.load(f)
        bonvis_settings = Bonvis_settings(load_settings_path=reset_path)
        return Bonvis_figure(self.bonvis_data, self.bonvis_metadata, bonvis_settings=bonvis_settings)

    def update_figure(self, geometry=None, zoom=None, click=None, node_style=None, size_style=None, scale_nodes=None,
                      origin=None, ly_type=None, tweak_inds=None, multip_angle=None, reset_layout=None, ax_lims=None,
                      zoom_ax_lims=None, renew_mask=False, verbose=True, scale_edges=None, scale_node_edges=None,
                      flipped_node_ids=[], new_flip_id=False):
        # TODO: Eventually remove this print-statement, nice for debugging
        if verbose:
            logging.info(
                'geometry={}, zoom={}, click={}, node_style={}, size_style={}, scale_nodes={}, scale_edges={}, '
                'scale_node_edges={}, origin={}, ly_type={}, '
                'tweak_inds={}, multip_angle={}, reset_layout={}, ax_lims={}, zoom_ax_lims={}, renew_mask={}, '
                'flipped_node_ids={}, new_flip_id={}'.format(
                    geometry, zoom, click,
                    node_style,
                    size_style,
                    scale_nodes, scale_edges, scale_node_edges, origin,
                    ly_type, tweak_inds,
                    multip_angle,
                    reset_layout, ax_lims, zoom_ax_lims,
                    renew_mask,
                    flipped_node_ids, new_flip_id))

        # First remove all collections from the current figure, the figure will be updated anyhow for the shiny app
        self.remove_artists()

        # Initialize variables that determine what needs to be updated in the collections
        reset_figure = False
        update_colors = False
        update_fig_coords = False
        update_sizes = False
        update_bg = False
        update_edges = False

        # Define some pointers for easier syntax
        transf_info = self.bonvis_settings.transf_info
        celltype_info = self.bonvis_settings.celltype_info

        if renew_mask:
            update_colors = True

        if ly_type is not None:
            update_fig_coords = True
            update_bg = True

            # Check if the ly_type is dendrogram, then set the geometry to flat.
            pattern = re.compile(r'dendrogram')
            if pattern.search(ly_type):
                geometry = 'flat'

            self.coords_info.change_ly_style(self.bonvis_settings, self.bonvis_data, self.bonvis_metadata, ly_type)

        if reset_layout:
            if len(self.coords_info.ly_trees) == 0:
                logging.debug("Tree layout hasn't changed since loading from stored settings. Can't reset it.")
            elif self.bonvis_settings.ly_type == 'ly_eq_daylight':
                self.reset_ed_layout()
            elif self.bonvis_settings.ly_type == 'ly_eq_angle':
                self.coords_info.ly_trees['ly_eq_angle'].resetEqualAngle()
                self.coords_info.set_coords(self.coords_info.ly_trees['ly_eq_angle'].coords, self.bonvis_settings,
                                            self.bonvis_data, self.bonvis_metadata)
                update_fig_coords = True

        if (tweak_inds is not None) or new_flip_id:
            print(flipped_node_ids)
            if self.bonvis_settings.ly_type not in self.coords_info.ly_trees:
                self.coords_info.reconstruct_tree(self.bonvis_metadata, self.bonvis_settings.ly_type)
            ly_tree = self.coords_info.ly_trees[self.bonvis_settings.ly_type]
            if ly_tree.coords is None:
                self.coords_info.add_layout_details_to_tree(ly_type=self.bonvis_settings.ly_type)
            if self.bonvis_settings.ly_type == 'ly_eq_daylight' and ('ly_eq_daylight' in self.coords_info.ly_trees):
                if tweak_inds is not None:
                    self.tweak_ed_layout(tweak_inds)
                else:
                    self.rearrange_branches(flipped_node_ids=flipped_node_ids)
                    ly_tree.equalAngle(verbose=verbose, get_nodelist=True)
                    ly_tree.equalDaylightAll(verbose=True)
                    self.coords_info.set_coords(ly_tree.coords, self.bonvis_settings, self.bonvis_data,
                                                self.bonvis_metadata)
                update_fig_coords = True

            elif self.bonvis_settings.ly_type == 'ly_eq_angle':
                if (tweak_inds is not None) and (multip_angle is not None):
                    self.tweak_ea_layout(tweak_inds, multip_angle)
                elif new_flip_id:
                    self.rearrange_branches(flipped_node_ids=flipped_node_ids)
                    ly_tree.equalAngle(verbose=verbose, get_nodelist=False)
                    self.coords_info.set_coords(ly_tree.coords, self.bonvis_settings, self.bonvis_data,
                                                self.bonvis_metadata)
                update_fig_coords = True
            # elif self.bonvis_settings.ly_type == 'ly_dendrogram':
            #     self.tweak_dendro_layout(tweak_inds, ladderized=False, flipped_node_ids=flipped_node_ids)
            #     update_fig_coords = True
            elif self.bonvis_settings.ly_type == 'ly_dendrogram_ladderized':
                self.tweak_dendro_layout(tweak_inds, ladderized=True, flipped_node_ids=flipped_node_ids)
                update_fig_coords = True

        if geometry is not None:
            update_fig_coords = True
            if transf_info.zoom_per_geometry[geometry] is None:
                edge_df_dict = self.bonvis_metadata.tree_info['edge_dict']
                transf_info.set_geometry(geometry, self.coords_info.node_coords_nx, edge_df_dict)
            else:
                transf_info.set_geometry(geometry)

        if ax_lims is not None:
            x_range = ax_lims[1] - ax_lims[0]
            ax_lims[0] -= .075 * x_range
            ax_lims[1] += .075 * x_range
            y_range = ax_lims[3] - ax_lims[2]
            ax_lims[2] -= .075 * y_range
            ax_lims[3] += .075 * y_range

            ax_lims[0] = max(ax_lims[0], -1.05)
            ax_lims[2] = max(ax_lims[2], -1.05)
            ax_lims[1] = min(ax_lims[1], 1.05)
            ax_lims[3] = min(ax_lims[3], 1.05)

            transf_info.ax_lims = ax_lims
            update_bg = True if transf_info.geometry == 'hyperbolic' else False
        elif transf_info.ax_lims is None:
            transf_info.ax_lims = [-1.05, 1.05, -1.05, 1.05]
            update_bg = True if transf_info.geometry == 'hyperbolic' else False

        if zoom_ax_lims is not None:
            new_ax_lims = [None] * 4
            curr_ax_lims = transf_info.ax_lims
            new_x_range = (curr_ax_lims[1] - curr_ax_lims[0]) * zoom_ax_lims
            x_mean = (curr_ax_lims[1] + curr_ax_lims[0]) / 2
            new_ax_lims[0] = max(x_mean - new_x_range / 2, -1.05)
            new_ax_lims[1] = min(x_mean + new_x_range / 2, 1.05)

            new_y_range = (curr_ax_lims[3] - curr_ax_lims[2]) * zoom_ax_lims
            y_mean = (curr_ax_lims[3] + curr_ax_lims[2]) / 2
            new_ax_lims[2] = max(y_mean - new_y_range / 2, -1.05)
            new_ax_lims[3] = min(y_mean + new_y_range / 2, 1.05)

            transf_info.ax_lims = new_ax_lims
            update_bg = True if transf_info.geometry == 'hyperbolic' else False

        if zoom is not None:
            update_fig_coords = True
            transf_info.set_zoom(multiply_by=zoom)

        if origin is not None:
            # First get new origin in original coordinates
            if not self.bonvis_settings.transf_info.no_transform:
                origin = invert_poincare_transform(origin[None, :], origin=transf_info.origin,
                                                   zoom=transf_info.get_zoom(),
                                                   transform=(transf_info.geometry == 'hyperbolic'))
                transf_info.origin = origin
                update_fig_coords = True

        if scale_nodes is not None:
            update_sizes = True
            self.bonvis_settings.node_style['radius_int'] *= scale_nodes
            self.bonvis_settings.node_style['radius_int_data'] *= scale_nodes
            self.bonvis_settings.node_style['radius_cell'] *= scale_nodes
            self.bonvis_settings.node_style['lw_int'] *= scale_nodes
            self.bonvis_settings.node_style['lw_cell'] *= scale_nodes
        
        if scale_edges is not None:
            update_edges = True
            self.bonvis_settings.edge_style['linewidth'] *= scale_edges

        if scale_node_edges is not None:
            update_sizes = True
            self.bonvis_settings.node_style['lw_int'] *= scale_node_edges
            self.bonvis_settings.node_style['lw_cell'] *= scale_node_edges

        if node_style is not None:
            old_annot_type = self.bonvis_settings.node_style['annot_info'].annot_type
            annot_info = find_annot(new_annot_label=node_style, bonvis_settings=self.bonvis_settings,
                                    bonvis_data=self.bonvis_data, bonvis_metadata=self.bonvis_metadata)
            self.bonvis_settings.set_annot(annot_info=annot_info)
            self.bonvis_settings.cell_to_celltype, _ = self.get_color_info(
                annot_info=self.bonvis_settings.node_style['annot_info'])

            if old_annot_type != self.bonvis_settings.node_style['annot_info'].annot_type:
                update_sizes = True
            if (annot_info is not None):
                update_colors = True
            else:
                logging.error("Did not find annot: {}.".format(node_style))

        # if n_clusters:
        #     old_annot_type = self.bonvis_settings.node_style['annot_info'].annot_type
        #     self.update_clusters(cluster_param=n_clusters, min_pdists=True)
        #     if old_annot_type != self.bonvis_settings.node_style['annot_info'].annot_type:
        #         update_sizes = True
        #     update_colors = True

        if size_style is not None:
            size_annot_info = find_annot(new_annot_label=size_style, bonvis_settings=self.bonvis_settings,
                                         bonvis_data=self.bonvis_data, bonvis_metadata=self.bonvis_metadata)
            self.bonvis_settings.set_size_style(annot_info=size_annot_info)

            if size_annot_info is not None:
                update_sizes = True
            else:
                logging.debug("Did not find size annot: {}.".format(size_style))

        if update_bg:
            curr_geom = self.bonvis_settings.transf_info.geometry
            if (self.bg_obj[curr_geom]['patches'] is None) and (self.bg_obj[curr_geom]['coll'] is None):
                self.get_bg_collection()
            if (curr_geom == 'hyperbolic'):
                self.update_bg()
            # self.update_bg()
        if update_edges:
            self.get_edge_collection()
        if update_sizes:
            self.get_node_collection()
            update_colors = False
            renew_mask = False
        if update_fig_coords:
            self.update_fig_coords()
        if update_colors:
            self.update_colors()
        create_legend = False
        if update_colors or update_sizes:
            create_legend = True
        return create_legend

    def remove_artists(self):
        if self.is_present['nodes']:
            self.int_obj['coll'].remove()
            self.single_obj['coll'].remove()
            if self.multi_obj is not None:
                self.multi_obj['coll'].remove()
            self.is_present['nodes'] = False
        if self.is_present['edges']:
            self.edge_obj['coll'].remove()
            self.is_present['edges'] = False
        if self.is_present['bg'] is not None:
            geom_present = self.is_present['bg']
            for coll in self.bg_obj[geom_present]['coll']:
                coll.remove()
            for patch in self.bg_obj[geom_present]['patches']:
                patch.remove()
            self.is_present['bg'] = None

    def update_clusters(self, cluster_param=10, min_pdists=True, footfall=False, colortype=None):
        # gradient_type = 'YlOrRd' # TODO: give this as an option/argument???
        if colortype is None:
            colortype = 'tab20'

        # my  notes
        # get clusters
        # get_max_diam_clustering()
        # call here function
        # add to annotation info thing
        # cluster_diam

        # get clustering
        if min_pdists:
            logging.debug("Performing minimal distance clustering!")
            # logging.debug(self.bonvis_metadata.tree_info['nwk_str'])
            node_id_to_n_cells = self.bonvis_metadata.tree_info['node_id_to_n_cells']
            clusters_list, cut_edges = get_min_pdists_clustering_from_nwk_str(
                tree_nwk_str=self.bonvis_metadata.tree_info['nwk_str'], n_clusters=cluster_param,
                cell_ids=self.bonvis_metadata.cs_ids, node_id_to_n_cells=node_id_to_n_cells)
        elif footfall:
            logging.debug("Performing maximal footfall clustering!")
            clusters_list, cut_edges = get_footfall_clustering_from_nwk_str(
                tree_nwk_str=self.bonvis_metadata.tree_info['nwk_str'],
                n_clusters=cluster_param, cell_ids=self.bonvis_metadata.cell_ids)
        else:
            logging.debug("Performing max-diameter clustering")
            clusters_list = get_max_diam_clustering_from_nwk_str(tree_nwk_str=self.bonvis_metadata.tree_info['nwk_str'],
                                                                 max_diam_threshold=cluster_param,
                                                                 cell_ids=self.bonvis_metadata.cell_ids)
            cut_edges = None

        logging.debug("number of clusters found: {}".format(len(clusters_list)))
        # num_clusters = len(clusters_list)
        # self.number_of_clusters_found = len(clusters_list)
        # logging.debug(self.number_of_clusters_found)
        # get clusters to cluster-idx
        cl_dict = get_cluster_assignments(clusters_list=clusters_list)

        # get cluster annotation
        # cmap = get_celltype_colors_new(colortype='gradient_colormap', gradientType=gradient_type)
        # n_cells = len(cell_info_dict[next(iter(cell_info_dict))])
        n_css = self.bonvis_metadata.n_Css
        bottom_quantile = np.minimum(10 / n_css, 0.01)
        top_quantile = 1 - bottom_quantile
        annot_type = 'cellstates'
        info_object = 'cs_info_dict'

        # make my categories
        # cats = list(set(cl_dict.values()))
        # cats = list(np.sort(list(set(cl_dict.values()))))
        # natural sorting
        cats = natsorted(list(set(cl_dict.values())))

        n_clusters = len(cats)
        cluster_colors = get_celltype_colors_new(n_clusters, colortype=colortype)
        annot_to_color = {cat: cluster_colors(ind) for ind, cat in enumerate(cats)}
        cbar_info = {'cmap': None, 'vmin': None, 'vmax': None, 'log': None}
        if min_pdists or footfall:
            label = "Cluster_n={}".format(cluster_param)  # TODO change this
            info_key = "annot_cl_n={}".format(cluster_param)  # TODO?
        else:
            label = "Cluster_diam={}".format(cluster_param)  # TODO change this
            info_key = "annot_cl_diam={}".format(cluster_param)  # TODO?

        if cut_edges is not None:
            self.bonvis_metadata.clusters_cut_edges[label] = cut_edges

        cl_annot = Annotation_info(cats=cats, annot_to_color=annot_to_color, label=label,
                                   cbar_info=cbar_info, annot_type=annot_type,
                                   color_type='categorical', info_object=info_object,
                                   info_key=info_key)

        # convert to right order
        cs_id_to_ind = {cs_id: ind for ind, cs_id in enumerate(cl_dict)}
        cs_inds = np.array([cs_id_to_ind[cs_id] for cs_id in self.bonvis_metadata.cs_ids])

        annot_df = pd.DataFrame({"cb_name": cl_dict.keys(),
                                 "clustering_tmp": cl_dict.values()})
        annotation_df = annot_df.iloc[cs_inds, :]
        self.bonvis_metadata.cs_info['cs_info_dict'][info_key] = annotation_df['clustering_tmp'].tolist()
        # logging.debug(annotation_df)

        self.bonvis_settings.set_annot(annot_info=cl_annot)
        self.bonvis_settings.cell_to_celltype, _ = self.get_color_info(
            annot_info=self.bonvis_settings.node_style['annot_info'])
        # TODO: only do this when storing
        self.bonvis_settings.celltype_info.annot_infos[info_key] = cl_annot
        self.bonvis_settings.celltype_info.annot_alts.append(info_key)

    def get_cell_to_celltype(self):
        if self.bonvis_settings.cell_to_celltype is None:
            self.bonvis_settings.cell_to_celltype, _ = self.get_color_info(
                annot_info=self.bonvis_settings.node_style['annot_info'])
        return self.bonvis_settings.cell_to_celltype

    def update_bg(self):
        # Transform background lines from data-coords to axes-coords. In this way we need to update it less
        # Once we have ax-coordinates, we only need to update it when ax-limits change
        if self.coords_info.data_bg_edge_coords_eix is None:
            self.coords_info.bg_data_to_ax(self.ax)
            self.bg_obj['hyperbolic']['ax_lims'] = self.bonvis_settings.transf_info.ax_lims.copy()
            self.bg_obj['hyperbolic']['coll'][0].set_segments(self.coords_info.data_bg_edge_coords_eix)
        else:
            old_ax_lims = self.bg_obj['hyperbolic']['ax_lims']
            new_ax_lims = self.bonvis_settings.transf_info.ax_lims
            for xy_ind in range(2):
                new_range = new_ax_lims[2 * xy_ind + 1] - new_ax_lims[2 * xy_ind + 0]
                mult_factor = (old_ax_lims[2 * xy_ind + 1] - old_ax_lims[2 * xy_ind + 0]) / new_range
                add_factor = (old_ax_lims[2 * xy_ind + 0] - new_ax_lims[2 * xy_ind + 0]) / new_range
                self.coords_info.data_bg_edge_coords_eix[:, :, xy_ind] *= mult_factor
                self.coords_info.data_bg_edge_coords_eix[:, :, xy_ind] += add_factor
            self.bg_obj['hyperbolic']['ax_lims'] = self.bonvis_settings.transf_info.ax_lims.copy()

    def update_colors(self):
        node_style = self.bonvis_settings.node_style
        size_annot_info = self.bonvis_settings.node_style['size_annot_info']
        annot_info = node_style['annot_info']
        node_coords = self.coords_info.transf_node_coords_nx

        obj_to_celltype, obj_to_color = self.get_color_info(annot_info=annot_info)

        if annot_info.annot_type == 'cells':
            plot_unit = 'cells'
        elif annot_info.annot_type == 'cellstates':
            plot_unit = 'cellstates'
        else:
            plot_unit = 'verts'

        if plot_unit in ['cells', 'cellstates']:
            self.single_obj['coll'].set_facecolors(obj_to_color[self.single_obj['obj_inds']])
        elif annot_info.annot_type == 'verts':
            self.int_obj['coll'].set_facecolors(obj_to_color[self.int_obj['vert_inds']])
            self.single_obj['coll'].set_facecolors(obj_to_color[self.single_obj['vert_inds']])

        if plot_unit != 'verts':
            self.plot_multi_obj_verts(node_coords=node_coords, obj_to_color=obj_to_color,
                                      node_style=node_style, obj_to_celltype=obj_to_celltype,
                                      annot_type=annot_info.annot_type)

    def update_fig_coords(self):
        # self.coords_info = Coords_info(self.bonvis_settings.transf_info, self.bonvis_data.obsm['bonsai_coords'],
        #                                self.bonvis_data.uns['edge_info'])
        self.coords_info.transform_coords(self.bonvis_settings.transf_info)
        # Update edge collection
        self.edge_obj['coll'].set_segments(self.coords_info.transf_edge_coords_eix)

        # Update node collections
        node_coords = self.coords_info.transf_node_coords_nx
        self.int_obj['coll'].set_offsets(node_coords[self.int_obj['vert_inds'], :])
        self.single_obj['coll'].set_offsets(node_coords[self.single_obj['vert_inds'], :])

        if self.multi_obj is not None:
            # We re-make the wedge-collection at plotting time (once axes-transformation is known), so here
            # we only have to store the coordinates in "data-coordinates".
            all_coords = []
            for ind, wedge in enumerate(self.multi_obj['wedges']):
                coords = node_coords[self.multi_obj['vert_inds_per_wedge'][ind], :]
                all_coords.append(coords)
            self.multi_obj['coords'] = all_coords
            # self.multi_obj['coll'] = PatchCollection(self.multi_obj['wedges'],
            #                                         **self.multi_obj['kwargs'])

    def create_figure(self, figsize=(12, 12), make_background=True, no_edges=False,
                      verbose=False, fig=None, ax=None):
        logging.debug("Creating figure!")
        if (fig is not None) and (ax is not None):
            self.fig = fig
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_box_aspect(1)
        # self.ax.set_position([0, 0, 1, 1])
        self.ax.axis('off')
        if self.bonvis_settings.transf_info.ax_lims is None:
            self.bonvis_settings.transf_info.ax_lims = [-1.05, 1.05, -1.05, 1.05]
        self.ax.set_xlim(self.bonvis_settings.transf_info.ax_lims[0], self.bonvis_settings.transf_info.ax_lims[1])
        self.ax.set_ylim(self.bonvis_settings.transf_info.ax_lims[2], self.bonvis_settings.transf_info.ax_lims[3])
        plt.subplots_adjust(left=0, right=1.0, bottom=0, top=1.0)

        if make_background:
            curr_geom = self.bonvis_settings.transf_info.geometry
            if (self.bg_obj[curr_geom]['patches'] is None) and (self.bg_obj[curr_geom]['coll'] is None):
                self.get_bg_collection()
            if (curr_geom == 'hyperbolic') and (self.coords_info.data_bg_edge_coords_eix is None):
                self.update_bg()

            for patch in self.bg_obj[curr_geom]['patches']:
                patch.set_transform(self.ax.transData)
                self.ax.add_patch(patch)
            for coll in self.bg_obj[curr_geom]['coll']:
                if curr_geom == 'hyperbolic':
                    coll.set_transform(self.ax.transAxes)
                else:
                    coll.set_transform(self.ax.transData)
                self.ax.add_collection(coll)
            self.is_present['bg'] = self.bonvis_settings.transf_info.geometry

        if not no_edges:
            self.edge_obj['coll'].set_transform(self.ax.transData)
            self.ax.add_collection(self.edge_obj['coll'])
            self.is_present['edges'] = True
            if verbose:
                logging.debug("Plotted all edges.")

        self.int_obj['coll'].set_offset_transform(self.ax.transData)
        self.single_obj['coll'].set_offset_transform(self.ax.transData)
        self.ax.add_collection(self.int_obj['coll'])
        self.ax.add_collection(self.single_obj['coll'])
        if self.multi_obj is not None:
            # The wedges have been created with a center that is in data-coordinates. We want to transform this into axes coordinates.
            # all_wedges = []
            transf = self.ax.transData + self.ax.transAxes.inverted()
            for ind, wedge in enumerate(self.multi_obj['wedges']):
                wedge.set_center(transf.transform(self.multi_obj['coords'][ind]))
            self.multi_obj['coll'] = PatchCollection(self.multi_obj['wedges'], **self.multi_obj['kwargs'])
            self.multi_obj['coll'].set_transform(self.ax.transAxes)
            self.ax.add_collection(self.multi_obj['coll'])
        self.is_present['nodes'] = True
        return self.fig

    def save_fig(self, filepath='bonsai_tree.png', format='.png'):
        self.fig.savefig(fname=filepath, format=format, dpi=300)

    def create_legend(self):
        annot_info = self.bonvis_settings.node_style['annot_info']
        celltype_to_color = annot_info.annot_to_color
        cbar_info = annot_info.cbar_info
        if celltype_to_color is not None:
            # colors.to_hex(color, keep_alpha=True)
            celltype_to_hex = {ind: (" ", celltype) for ind, (celltype, color) in enumerate(celltype_to_color.items())}
            leg_df = pd.DataFrame.from_dict(celltype_to_hex, orient='index', columns=["Color", 'Celltype'])
            color_codes = [colors.to_hex(celltype_to_color[celltype], keep_alpha=True) for celltype in
                           leg_df['Celltype']]

            s = leg_df.style.set_table_attributes('class="dataframe shiny-table table w-auto"').apply(
                color_code_to_color, axis=0, subset='Color', color_codes=color_codes
            ).hide(axis="index"
                   ).hide(axis='columns')
            self.fig_leg_df = s
            self.fig_cbar = None
        elif cbar_info['cmap'] is not None:
            self.fig_leg_df = None
            self.fig_cbar = None

            fig_cbar = plt.figure(figsize=(2, 4))
            ax_cbar = fig_cbar.add_subplot(111)

            norm = colors.Normalize(vmin=0, vmax=1)
            mappable = cm.ScalarMappable(norm=norm, cmap=cbar_info['cmap'])
            cbar = plt.colorbar(mappable, cax=ax_cbar, orientation='vertical')

            tick_list = [mappable.colorbar.vmin + t * (mappable.colorbar.vmax - mappable.colorbar.vmin) for t in
                         cbar.ax.get_yticks()]

            min_val = cbar_info['vmin']
            max_val = cbar_info['vmax']

            def cbar_to_numb(val):
                if cbar_info['log']:
                    return np.exp(val * (max_val - min_val) + min_val)
                return val * (max_val - min_val) + min_val

            tick_labels = ['{:.2e}'.format(cbar_to_numb(tick)) for tick in tick_list]
            cbar.set_ticks(tick_list)
            cbar.set_ticklabels(tick_labels)
            # ax_leg.legend(*leg_handles_labels, loc='center')
            # # hide the axes frame and the x/y labels
            # ax_cbar.axis('off')
            plt.tight_layout()
            self.fig_cbar = fig_cbar
            # self.fig_leg = fig_leg
            # if label is None:
            #     if (self.curr_style_type is not None) and (self.curr_style_type == 'summedUMIs'):
            #         label = 'Summed UMIs'
            #     else:
            #         label = ''
            #
            # if self.cbar_ax is None:
            #     self.original_loc_wo_colorbar = self.ax.get_axes_locator()
            #     divider = make_axes_locatable(self.ax)
            #     cax = divider.append_axes("right", size="10%", pad=0.05)
            #     self.cbar_ax = cax
            # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            # self.cbar = plt.colorbar(mappable, cax=self.cbar_ax, orientation='vertical', label=label)
            # # change colorbar ticks labels and locators
            # tick_list = [mappable.colorbar.vmin + t * (mappable.colorbar.vmax - mappable.colorbar.vmin) for t in
            #              self.cbar.ax.get_yticks()]
            # if self.colorbar_to_numb is None:
            #     def colorbar_to_numb(tick):
            #         return tick
            # else:
            #     colorbar_to_numb = self.colorbar_to_numb
            #
            # tick_labels = ['%.2e' % colorbar_to_numb(tick) for tick in tick_list]
            # self.cbar.set_ticks(tick_list)
            # self.cbar.set_ticklabels(tick_labels)
            # self.colorbar_present = True

        # node_style = self.bonvis_settings.node_style
        # fig_leg = plt.figure(figsize=figsize)
        # ax_leg = fig_leg.add_subplot(111)
        #
        # leg_handles_labels = ax_leg.get_legend_handles_labels()
        # for ind, (celltype, color) in enumerate(celltype_to_color.items()):
        #     circle = Circle((0, 0), radius=node_style['radius_cell'], facecolor=color, label=celltype,
        #                     edgecolor=node_style['edgecolor'], linewidth=node_style['lw_cell'])
        #     leg_handles_labels[0].append(circle)
        #     leg_handles_labels[1].append(celltype)
        #

    def pick_node(self, coords, was_transformed=True):
        if len(coords.shape) == 1:
            coords = coords[None, :]
        if was_transformed:
            transf_info = self.bonvis_settings.transf_info
            if not transf_info.no_transform:
                coords = invert_poincare_transform(coords, origin=transf_info.origin,
                                                   zoom=transf_info.get_zoom(),
                                                   transform=(transf_info.geometry == 'hyperbolic'))
        closest_inds = {}
        closest_ind = -1
        closest_dist = 1e9
        for sub_id, nn_index in self.coords_info.nn_index.items():
            dist, closest = nn_index.query(coords, k=1, return_distance=True)
            closest = self.bonvis_metadata.tree_info[sub_id][closest][0][0]
            dist = dist[0][0]
            if dist < closest_dist:
                closest_ind = closest
                closest_dist = dist
            closest_inds[sub_id] = closest
        closest_inds['all'] = closest_ind
        # closest_inds = {sub_id: nn_index.query(coords, k=1, return_distance=True) for sub_id, nn_index in
        #                 self.coords_info.nn_index.items()}
        # closest_ind = self.coords_info.nn_index.query(coords, k=1, return_distance=False)[0][0]
        logging.debug(self.coords_info.transf_node_coords_nx[closest_inds['int_inds'], :])
        return closest_inds

    def tweak_ed_layout(self, ed_ind):
        ly_tree = self.coords_info.ly_trees['ly_eq_daylight']
        if ly_tree.root.dsNodes is None:
            ly_tree.root.getDsLeafs(ly_tree.nNodes, verbose=True, get_nodelist=True)
        print('dsNodes: {}'.format(self.coords_info.ly_trees['ly_eq_daylight'].root.dsNodes))
        ly_tree.equalDaylightSome(vert_inds=[ed_ind], stepsize=1)
        self.coords_info.set_coords(ly_tree.coords, self.bonvis_settings, self.bonvis_data,
                                    self.bonvis_metadata)

    def reset_ed_layout(self):
        ly_tree = self.ly_trees['ly_eq_daylight']
        old_coords = self.bonvis_data['layout_coords']['node_coords'][self.bonvis_settings.ly_type]
        self.coords_info.set_coords(old_coords, self.bonvis_settings, self.bonvis_data, self.bonvis_metadata)
        ly_tree.coords = old_coords[:]
        ly_tree.set_coords_at_node()
        ly_tree.get_angles_from_coords()

    def rearrange_branches(self, flipped_node_ids=[]):
        ly_tree = self.coords_info.ly_trees[self.bonvis_settings.ly_type]
        ly_tree.root.rearrange_branches_node(flipped_node_ids=flipped_node_ids, nNodes=ly_tree.nNodes)

    def tweak_ea_layout(self, ea_inds, multip_angle):
        if ea_inds[-1] == ea_inds[-2]:
            logging.debug("Two picked nodes are the same. Cannot tweak tree like this.")
            return
        ly_tree = self.coords_info.ly_trees['ly_eq_angle']
        if ly_tree.root.dsNodes is None:
            ly_tree.root.getDsLeafs(ly_tree.nNodes, verbose=True, get_nodelist=True)
        new_origin = ly_tree.increaseEqualAngle(vert_inds=ea_inds, multip_angle=multip_angle,
                                                origin=self.bonvis_settings.transf_info.origin)
        self.bonvis_settings.transf_info.origin = new_origin
        self.coords_info.set_coords(ly_tree.coords, self.bonvis_settings, self.bonvis_data,
                                    self.bonvis_metadata)

    def tweak_dendro_layout(self, new_root_ind, ladderized=False, flipped_node_ids=[]):
        ly_tree = self.coords_info.ly_trees['ly_dendrogram_ladderized']
        if new_root_ind is not None:
            # Make a copy of the tree with only the topology information
            # new_tree = ly_tree.copy(minimal_copy=True)
            # Use this tree to set the new root
            ly_tree.reset_root(new_root_ind)
            ly_tree.root.getDsLeafs(ly_tree.nNodes, verbose=True, get_nodelist=False)
        node_id_to_vert_ind = {node.nodeId: vert_ind for vert_ind, node in ly_tree.get_vert_ind_to_node_DF().items()}
        edge_dict = ly_tree.get_edge_dict(nodeIdToVertInd=node_id_to_vert_ind)
        ly_tree.get_dendrogram(verbose=True, ladderized=ladderized, flipped_node_ids=flipped_node_ids)
        # self.coords_info.ly_tree.coords = ly_tree.coords
        self.coords_info.set_coords(ly_tree.coords, self.bonvis_settings, self.bonvis_data,
                                    self.bonvis_metadata, edge_dict=edge_dict)

    def preprocess_marker_genes(self, bonvis_data_path, curr_subset, curr_subset_2=None, curr_categorical_annot=None):
        logging.debug("update_marker_genes_df(curr_subset={}, curr_subset_2={} curr_categorical_annot={})"
                      .format(curr_subset, curr_subset_2, curr_categorical_annot))
        feature_hdf = self.bonvis_data[self.bonvis_settings.node_style['feature_path']]
        feature_path = self.bonvis_settings.node_style['feature_path']
        time_estimate = None
        marker_info_dict = {'feature_path': self.bonvis_settings.node_style['feature_path']}
        # Determine if we run the script with or without error-bars
        run_with_vars = 'yes' if ('vars' in feature_hdf.keys()) else 'no'

        ds_cell_inds_1, _ = self.get_cell_inds_in_subset(curr_subset, curr_categorical_annot=curr_categorical_annot,
                                                         return_vert_inds=False)
        if ds_cell_inds_1 is None:
            self.marker_genes_df = pd.DataFrame(columns=['marker_genes', 'marker_scores'])
            run_with_vars = 'stop'
            return run_with_vars, None, time_estimate, None
        else:
            ds_cell_inds_1 = np.sort(ds_cell_inds_1)

        if (curr_subset_2 is None) or (curr_subset_2['type'] is None):
            ds_cell_inds_2 = np.setdiff1d(np.arange(self.bonvis_metadata.n_cells), ds_cell_inds_1)
        else:
            ds_cell_inds_2, _ = self.get_cell_inds_in_subset(curr_subset_2,
                                                             curr_categorical_annot=curr_categorical_annot,
                                                             return_vert_inds=False)
            if ds_cell_inds_2 is None:
                ds_cell_inds_2 = np.setdiff1d(np.arange(self.bonvis_metadata.n_cells), ds_cell_inds_1)
            else:
                ds_cell_inds_2 = np.sort(ds_cell_inds_2)

        node_ids = self.bonvis_metadata.cell_ids
        ds_cell_ids_1 = [node_ids[cell_ind] for cell_ind in ds_cell_inds_1]
        ds_cell_ids_2 = [node_ids[cell_ind] for cell_ind in ds_cell_inds_2]
        marker_info_dict['cell_ids_group1'] = ds_cell_ids_1
        marker_info_dict['cell_ids_group2'] = ds_cell_ids_2

        if run_with_vars == 'yes':
            desired_time = 20
            allowed_time = 40
            n_points = desired_time / (
                        5e-8 * (len(ds_cell_inds_1) + len(ds_cell_inds_2)) * self.bonvis_metadata.n_genes)
            n_points = min(max(n_points, 50), 300)
            time_estimate = 5e-8 * (len(ds_cell_inds_1) + len(ds_cell_inds_2)) * self.bonvis_metadata.n_genes * n_points
            logging.info("Time estimate for full marker gene calculation is {} seconds.".format(time_estimate))
            if time_estimate > allowed_time:
                run_with_vars = 'shortcut'
                logging.warning("This is more than the maximally allowed {} seconds. "
                                "Switching to marker gene calculation without taking error-bars into account.\n One could "
                                "run the marker gene calculation outside the app if desired.".format(allowed_time))

        if run_with_vars == 'yes':
            logging.info("Calculating marker genes accounting for the error-bars "
                         "between groups of sizes {} and {}.".format(len(ds_cell_inds_1), len(ds_cell_inds_2)))
            # Determine if the feature matrix has vertex- or cell-information. In the first case, we have to cut the
            # cells out
            if feature_hdf['means'].shape[1] == self.bonvis_metadata.n_cells:
                get_cells_out = False
            elif feature_hdf['means'].shape[1] == self.bonvis_metadata.n_nodes:
                get_cells_out = True
            else:
                logging.error("This feature matrix doesn't have columns equal to number of cells or nodes.\n"
                              "Marker gene detection is therefore not supported (yet).")
                self.marker_genes_df = pd.DataFrame(columns=['marker_genes', 'marker_scores'])
                run_with_vars = 'stop'
                return run_with_vars, None, time_estimate, None

            cells_wo_nan = feature_hdf['cells_wo_nan']
            ds_cell_inds_1 = np.intersect1d(ds_cell_inds_1, cells_wo_nan)
            ds_cell_inds_2 = np.intersect1d(ds_cell_inds_2, cells_wo_nan)
            means_path = os.path.join(feature_path, 'means')
            vars_path = os.path.join(feature_path, 'vars')
            if get_cells_out:
                vert_inds_cells = np.array(self.bonvis_metadata.cell_info['cell_info_dict']['cell_ind_to_vert_ind'])
                ds_cell_inds_1 = np.sort(vert_inds_cells[ds_cell_inds_1])
                ds_cell_inds_2 = np.sort(vert_inds_cells[ds_cell_inds_2])
                # means = feature_hdf['means'][:][:, vert_inds_cells]
                # vars = feature_hdf['vars'][:][:, vert_inds_cells]
            # else:
            #     means = feature_hdf['means'][:]
            #     vars = feature_hdf['vars'][:]
            gene_ids = json.loads(feature_hdf.attrs['gene_ids'])
            marker_genes_tuple = (ds_cell_inds_1, ds_cell_inds_2, bonvis_data_path, means_path, vars_path, gene_ids, n_points)
            # marker_genes_tuple = (ds_cell_inds_1, ds_cell_inds_2, bonvis_data_path, means, vars, gene_ids, n_points)
            return run_with_vars, marker_genes_tuple, time_estimate, marker_info_dict
        else:
            # Note that we only get ranks for cells that have no nan-value for any gene
            # ranks_per_gene = feature_hdf['ranks_per_gene'][:]
            ranks_per_gene_path = os.path.join(feature_path, 'ranks_per_gene')
            cells_wo_nan = feature_hdf['cells_wo_nan'][:]
            n_cells = len(cells_wo_nan)
            ds_cell_inds_1 = np.intersect1d(ds_cell_inds_1, cells_wo_nan)
            gene_ids = json.loads(feature_hdf.attrs['gene_ids'])
            variation_features = np.setdiff1d(np.arange(len(gene_ids)), feature_hdf['no_variation_features'])
            # marker_genes_tuple = (
            # ds_cell_inds_1, ds_cell_inds_2, n_cells, gene_ids, ranks_per_gene, variation_features, cells_wo_nan)
            marker_genes_tuple = (bonvis_data_path, ds_cell_inds_1, ds_cell_inds_2, n_cells, gene_ids,
                                  ranks_per_gene_path, variation_features, cells_wo_nan)

            return run_with_vars, marker_genes_tuple, time_estimate, marker_info_dict

    def set_mask_for_subset(self, curr_subsets, curr_categorical_annot=None):
        # TODO: Make this deal with tuple of subsets
        non_masked_lst = []
        mask_is_on = False
        for curr_subset in curr_subsets:
            if curr_subset['mask_is_on']:
                mask_is_on = True
                subset_cell_inds, cells_or_verts = self.get_cell_inds_in_subset(curr_subset,
                                                                                curr_categorical_annot=curr_categorical_annot,
                                                                                return_vert_inds='flexible')
                if subset_cell_inds is not None:
                    non_masked_lst.append(subset_cell_inds)
            else:
                non_masked_lst.append(np.zeros(0))

        non_masked = np.unique(np.concatenate(non_masked_lst))
        if (non_masked is None) or (len(non_masked) == 0):
            verts_masked = None
            cells_masked = None
        else:
            if cells_or_verts == 'verts':
                verts_masked = np.setdiff1d(self.bonvis_metadata.cell_info['non_int_vert_inds'], non_masked).tolist()
                cells_masked = None
            else:
                cells_masked = np.setdiff1d(np.arange(self.bonvis_metadata.n_cells), non_masked).tolist()
                verts_masked = None

        self.bonvis_settings.node_style['verts_masked'] = verts_masked
        self.bonvis_settings.node_style['cells_masked'] = cells_masked
        self.bonvis_settings.node_style['use_mask'] = mask_is_on

    def get_cell_inds_in_subset(self, curr_subset, curr_categorical_annot=None, return_vert_inds=False):
        ds_cell_inds = None
        ds_vert_inds = None
        if curr_subset['type'] is None:
            ds_cell_inds = np.zeros(0)
            ds_vert_inds = np.zeros(0)
        elif curr_subset['type'] == 'subtree':
            return_vert_inds = False if (return_vert_inds == 'flexible') else return_vert_inds
            subtree_inds = curr_subset['info']
            # logging.debug("subtree_inds {}".format(subtree_inds))
            # Get from tree information which cells are part of the selected subtree
            if not return_vert_inds:
                ds_cell_inds = self.get_cell_inds_subtree(subtree_inds, return_vert_inds=False)
            else:
                ds_vert_inds = self.get_cell_inds_subtree(subtree_inds, return_vert_inds=True)
        elif curr_subset['type'] == 'annot':
            # cell_info_dict = self.bonvis_metadata.cell_info['cell_info_dict']
            annot_cat = curr_subset['info']
            annot_label = curr_categorical_annot
            # logging.debug("annot_cat: {}, annot_type: {}".format(annot_cat, annot_type))
            curr_annot_label = self.bonvis_settings.node_style[
                'annot_info'].label if annot_label is None else annot_label
            curr_annot = find_annot(curr_annot_label, self.bonvis_settings, self.bonvis_data, self.bonvis_metadata)
            curr_info_object = curr_annot.info_object
            if curr_info_object == 'cell_info_dict':
                return_vert_inds = False if (return_vert_inds == 'flexible') else return_vert_inds
                info_dict = self.bonvis_metadata.cell_info['cell_info_dict']
                ds_cell_inds = np.where(np.array(info_dict[curr_annot.info_key]) == annot_cat)[0]
                if return_vert_inds:
                    ds_vert_inds = np.array(info_dict['cell_ind_to_vert_ind'])[ds_cell_inds]
            elif curr_info_object == 'cs_info_dict':
                return_vert_inds = False if (return_vert_inds == 'flexible') else return_vert_inds
                info_dict = self.bonvis_metadata.cs_info['cs_info_dict']
                ds_cs_inds = np.where(np.array(info_dict[curr_annot.info_key]) == annot_cat)[0]
                if return_vert_inds:
                    ds_vert_inds = np.array(info_dict['cs_ind_to_vert_ind'])[ds_cs_inds]
                else:
                    ds_cell_inds = list(itertools.chain.from_iterable(
                        [self.bonvis_metadata.cs_ind_to_cell_inds[str(cs_ind)] for cs_ind in ds_cs_inds]))
            elif curr_info_object == 'cluster_info_dict':
                return_vert_inds = False if (return_vert_inds == 'flexible') else return_vert_inds
                info_dict = self.bonvis_metadata.cs_info['cluster_info_dict']
                ds_cs_inds = np.where(np.array(info_dict[curr_annot.info_key]) == annot_cat)[0]
                if return_vert_inds:
                    ds_vert_inds = np.array(info_dict['cs_ind_to_vert_ind'])[ds_cs_inds]
                else:
                    ds_cell_inds = list(itertools.chain.from_iterable(
                        [self.bonvis_metadata.cs_ind_to_cell_inds[str(cs_ind)] for cs_ind in ds_cs_inds]))
            elif curr_info_object == 'vert_info_dict':
                return_vert_inds = True if (return_vert_inds == 'flexible') else return_vert_inds
                info_dict = self.bonvis_metadata.vert_info
                ds_vert_inds = np.where(np.array(info_dict[curr_annot.info_key]) == annot_cat)[0]
                if not return_vert_inds:
                    ds_cell_inds = list(itertools.chain.from_iterable(
                        [info_dict['vert_ind_to_cell_inds'][cs_ind] for cs_ind in ds_vert_inds]))
            elif curr_info_object == 'new_cell_info_dict':
                return_vert_inds = True if (return_vert_inds == 'flexible') else return_vert_inds
                info_dict = self.bonvis_settings.verttype_info.new_cell_info_dict
                ds_vert_inds = np.where(np.array(info_dict[curr_annot.info_key]) == annot_cat)[0]
                if not return_vert_inds:
                    ds_cell_inds = list(itertools.chain.from_iterable(
                        [info_dict['vert_ind_to_cell_inds'][cs_ind] for cs_ind in ds_vert_inds]))
            else:
                logging.error("Annotation-style {} cannot be found in any info object!".format(curr_annot.label))
                ds_cell_inds = np.zeros(0)
                ds_vert_inds = np.zeros(0)
        else:
            logging.debug("Cannot find cells in selected subset.")
            ds_cell_inds = np.zeros(0)
            ds_vert_inds = np.zeros(0)
        if return_vert_inds:
            if (ds_vert_inds is not None) and len(ds_vert_inds) == 0:
                ds_vert_inds = None
            return ds_vert_inds, 'verts'
        if (ds_cell_inds is not None) and len(ds_cell_inds) == 0:
            ds_cell_inds = None
        return ds_cell_inds, 'cells'

    def get_cell_inds_subtree(self, subtree_inds, return_vert_inds=False):
        if self.bonvis_settings.ly_type not in self.coords_info.ly_trees:
            self.coords_info.reconstruct_tree(self.bonvis_metadata, ly_type=self.bonvis_settings.ly_type,
                                              layout_details=False)

        ly_tree = self.coords_info.ly_trees[self.bonvis_settings.ly_type]
        ancestor_ind = subtree_inds[-2]
        ancestor = ly_tree.get_vert_ind_to_node_DF()[ancestor_ind]
        ancestor.get_dsNodes()

        ds_ind = subtree_inds[-1]
        ds_vert_inds = None
        for child in ancestor.childNodes:
            if ds_ind in child.dsNodes:
                ds_vert_inds = child.dsNodes
                break
        if ds_vert_inds is None:
            ds_vert_inds = np.setdiff1d(np.arange(ly_tree.nNodes), ancestor.dsNodes)
        # logging.debug("ds_vert_inds {}".format(ds_vert_inds))
        if return_vert_inds:
            return ds_vert_inds
        ds_cell_inds = []
        for vert_ind in ds_vert_inds:
            ds_cell_inds += self.bonvis_metadata.vert_info['vert_ind_to_cell_inds'][str(vert_ind)]
        ds_cell_inds = np.array(ds_cell_inds)
        return ds_cell_inds


class Transf_info:
    geometry = None
    origin_style = None
    zoom_per_geometry = None
    origin = None

    no_transform = False

    ax_lims = None

    def __init__(self, geometry='hyperbolic', origin_style='root', zoom=None, origin=None, ax_lims=None,
                 no_transform=None, zoom_per_geometry=None):
        self.origin_style = origin_style
        self.zoom_per_geometry = {'hyperbolic': None, 'flat': None}
        if zoom_per_geometry is not None:
            self.zoom_per_geometry = zoom_per_geometry
        elif zoom is not None:
            self.zoom_per_geometry[geometry] = zoom
        self.set_geometry(geometry)
        if origin is not None:
            self.origin = np.array(origin)
        if ax_lims is not None:
            self.ax_lims = ax_lims
        else:
            self.ax_lims = [-1.05, 1.05, -1.05, 1.05]
        self.no_transform = no_transform

    def to_dict(self):
        self_dict = self.__dict__
        if ('origin' in self_dict) and isinstance(self_dict['origin'], np.ndarray):
            self_dict['origin'] = self_dict['origin'].tolist()
        return self_dict

    def set_geometry(self, geometry, node_coords_nx=None, edge_df_dict=None):
        self.geometry = geometry
        if ((self.get_zoom() is None) or (self.origin is None)) and (node_coords_nx is not None):
            self.get_zoom_origin(node_coords_nx, edge_df_dict)

    def set_zoom(self, zoom=None, multiply_by=None):
        if multiply_by is not None:
            if self.zoom_per_geometry[self.geometry] is not None:
                self.zoom_per_geometry[self.geometry] *= multiply_by
        else:
            self.zoom_per_geometry[self.geometry] = zoom

    def get_zoom(self):
        return self.zoom_per_geometry[self.geometry]

    def get_zoom_origin(self, node_coords_nx, edge_df_dict):
        if self.origin is None:
            if self.origin_style == 'most_connected':
                if edge_df_dict is None:
                    self.origin = None
                else:
                    verts, degrees = np.unique(edge_df_dict['source_ind'] + edge_df_dict['target_ind'],
                                               return_counts=True)
                    most_conn_vert = verts[np.argmax(degrees)]
                    self.origin = node_coords_nx[most_conn_vert, :]
            elif self.origin_style == 'root':
                if edge_df_dict is None:
                    self.origin = None
                else:
                    root_ind = edge_df_dict['source_ind'][0]
                    self.origin = node_coords_nx[root_ind, :]
            else:
                self.origin = None

        if self.origin is None:
            if self.get_zoom() is None:
                zoom, self.origin = get_opt_zoom_origin_poincare(node_coords_nx, self, verbose=False)
                self.set_zoom(zoom=zoom)
            else:
                self.origin = get_centroid_poincare(node_coords_nx, self,
                                                    zoom=self.get_zoom(),
                                                    verbose=False)
        else:
            if self.get_zoom() is None:
                zoom = get_zoom_given_origin(node_coords_nx, self)
                self.set_zoom(zoom=zoom)


def update_marker_genes_df(run_with_vars, marker_genes_tuple):
    print("Starting the update_marker_genes_df")
    if run_with_vars == 'stop':
        return pd.DataFrame(columns=['marker_genes', 'marker_scores'])
    elif run_with_vars == 'yes':
        # ds_cell_inds_1, ds_cell_inds_2, means, vars, gene_ids, n_points = marker_genes_tuple
        ds_cell_inds_1, ds_cell_inds_2, bonvis_data_path, means_path, vars_path, gene_ids, n_points = marker_genes_tuple
        # ds_cell_inds_1, ds_cell_inds_2, bonvis_data_path, means, vars, gene_ids, n_points = marker_genes_tuple
        bonvis_data = h5py.File(bonvis_data_path, 'r')
        means = bonvis_data[means_path]
        vars = bonvis_data[vars_path]
        # marker_genes_OG = calc_marker_genes_error_bars(indices1=ds_cell_inds_1, indices2=ds_cell_inds_2, means=means,
        #                                             vars=vars, gene_ids=gene_ids,
        #                                             n_marker_genes=10)
        marker_genes = calc_marker_genes_error_bars_approx2(indices1=ds_cell_inds_1, indices2=ds_cell_inds_2,
                                                            means=means,
                                                            vars=vars, gene_ids=gene_ids, n_points_total=n_points,
                                                            min_marker_genes=10,
                                                            marker_cutoff=.9)
    else:
        # ds_cell_inds_1, ds_cell_inds_2, n_cells, gene_ids, ranks_per_gene, variation_features, cells_wo_nan = marker_genes_tuple
        bonvis_data_path, ds_cell_inds_1, ds_cell_inds_2, n_cells, gene_ids, ranks_per_gene_path, variation_features, cells_wo_nan = marker_genes_tuple
        bonvis_data = h5py.File(bonvis_data_path, 'r')
        ranks_per_gene = bonvis_data[ranks_per_gene_path]
        if ds_cell_inds_2 is None:
            print("Starting the calc_marker_genes_single")
            marker_genes = calc_marker_genes_single(ds_cell_inds_1, n_cells, gene_ids, ranks_per_gene,
                                                    min_marker_genes=10,
                                                    marker_cutoff=.9,
                                                    gene_subset=variation_features)
        else:
            print("Starting the calc_marker_genes_double")
            ds_cell_inds_2 = np.intersect1d(ds_cell_inds_2, cells_wo_nan)
            marker_genes = calc_marker_genes_double(ds_cell_inds_1, ds_cell_inds_2, n_cells, gene_ids,
                                                    ranks_per_gene,
                                                    min_marker_genes=10,
                                                    marker_cutoff=.9,
                                                    gene_subset=variation_features)

    marker_genes_df = pd.DataFrame.from_dict(marker_genes, orient='index', columns=['marker_scores'])
    marker_genes_df.reset_index(inplace=True)
    marker_genes_df.rename({'marker_scores': 'marker_scores', 'index': 'marker_genes'}, axis=1,
                           inplace=True)
    marker_scores = marker_genes_df['marker_scores'].values
    sorted_markers = np.argsort(np.minimum(marker_scores, 1 - marker_scores))
    marker_genes_df = marker_genes_df.iloc[sorted_markers]
    return marker_genes_df


class Edge_info:
    style_type = None
    edge_collection = None

    def __init__(self, style_type='default'):
        self.style_type = style_type
        # TODO: Complete this


class Bg_info:
    style_type = None
    style_df = None

    def __init__(self, style_type='default'):
        self.style_type = style_type
        # TODO: Complete this


class Coords_info:
    node_coords_nx = None
    edge_coords_eix = None
    transf_node_coords_nx = None
    transf_edge_coords_eix = None

    nn_index = None
    ly_trees = None

    # For hyperbolic curvature, we have some lines as eye-guides
    transf_bg_edge_coords_eix = None
    data_bg_edge_coords_eix = None

    ax_lims = None

    def __init__(self, bonvis_settings, bonvis_data, bonvis_metadata):
        self.ly_trees = {}
        node_coords = bonvis_data['layout_coords']['node_coords']
        bonvis_settings.ly_types = [ly_type for ly_type in node_coords.keys()]
        if bonvis_settings.ly_type is None:
            if 'ly_eq_daylight' in bonvis_settings.ly_types:
                bonvis_settings.set_ly_type('ly_eq_daylight')
            elif 'ly_eq_angle' in bonvis_settings.ly_types:
                bonvis_settings.set_ly_type('ly_eq_angle')
            else:
                bonvis_settings.set_ly_type(bonvis_settings.ly_types[0])
        self.get_coords(bonvis_settings, bonvis_data, bonvis_metadata)
        self.transform_coords(bonvis_settings.transf_info)

        # Initialize a KD-tree which can be used to efficiently query which node is closest to a picked point in the fig
        self.update_nn_index(bonvis_metadata.tree_info, leaf_size=10)

    def reconstruct_tree(self, bonvis_metadata, ly_type, layout_details=True):
        # Reconstruct the tree-object from the newick-string to make tuning of the tree layout possible
        if 'nwk_str' in bonvis_metadata.tree_info:
            self.ly_trees[ly_type] = Layout_Tree()
            node_id_to_vert_ind = {node_id: ind for ind, node_id in enumerate(bonvis_metadata.node_ids)}
            self.ly_trees[ly_type].from_newick(bonvis_metadata.tree_info['nwk_str'], node_id_to_vert_ind)
            self.ly_trees[ly_type].root.storeParent()
            if layout_details:
                self.add_layout_details_to_tree(ly_type)

    def add_layout_details_to_tree(self, ly_type):
        if ly_type not in self.ly_trees:
            logging.debug("Can't add layout-details, tree was not reconstructed yet.")
        ly_tree = self.ly_trees[ly_type]
        ly_tree.coords = self.node_coords_nx[:]
        nNodes = self.node_coords_nx.shape[0]
        # Only do get_nodelist when necessary
        ly_tree.root.getDsLeafs(nNodes, verbose=True, get_nodelist=False)
        ly_tree.set_coords_at_node()
        ly_tree.get_angles_from_coords()

        # We set the coordinates on the tree once, just to make sure that all internal variables are set correctly
        # coords = np.zeros((nNodes, 2))
        # self.ly_tree.root.coords = np.array([0., 0.])
        ly_tree.root.thetaParent = 180
        # coords[self.ly_tree.root.vert_ind, :] = self.ly_tree.root.coords
        ly_tree.root.set_thetaParent()

    def change_ly_style(self, bonvis_settings, bonvis_data, bonvis_metadata, ly_type):
        if ly_type not in bonvis_settings.ly_types:
            logging.debug("This layout-type is not stored. Check this.")
            return
        bonvis_settings.set_ly_type(ly_type)

        self.get_coords(bonvis_settings, bonvis_data, bonvis_metadata)
        self.transform_coords(bonvis_settings.transf_info)
        self.update_nn_index(bonvis_metadata.tree_info, leaf_size=10)
        if ly_type in self.ly_trees:
            self.ly_trees[ly_type].coords = self.node_coords_nx[:]
            self.ly_trees[ly_type].set_coords_at_node()

    def get_coords(self, bonvis_settings, bonvis_data, bonvis_metadata):
        if bonvis_settings.ly_type in bonvis_settings.upd_node_coords:
            self.node_coords_nx = bonvis_settings.upd_node_coords[bonvis_settings.ly_type]
            self.edge_coords_eix = bonvis_settings.upd_edge_coords[bonvis_settings.ly_type]
        else:
            self.node_coords_nx = bonvis_data['layout_coords']['node_coords'][bonvis_settings.ly_type][:]
            self.edge_coords_eix = bonvis_data['layout_coords']['edge_coords'][bonvis_settings.ly_type][:]
        transf_info = bonvis_settings.transf_info
        # logging.debug("get_zoom {}, zoom_per_origin {}".format(transf_info.get_zoom(), transf_info.zoom_per_geometry))
        # logging.debug("origin {}".format(transf_info.origin))

        if (not transf_info.no_transform) and( (transf_info.get_zoom() is None) or (transf_info.origin is None)):
            # Get zoom and origin for transformation, copying from transf_info if known or determining them anew
            # edge_df_dict = bonvis_data['edge_info']['edge_df_dict']
            edge_df_dict = bonvis_metadata.tree_info['edge_dict']
            transf_info.get_zoom_origin(self.node_coords_nx, edge_df_dict)

    def set_coords(self, new_coords, bonvis_settings, bonvis_data, bonvis_metadata, edge_dict=None):
        if edge_dict is None:
            edge_dict = bonvis_metadata.tree_info['edge_dict']
        # Set node-coords
        bonvis_settings.upd_node_coords[bonvis_settings.ly_type] = new_coords
        # Calculate new edge-coords
        vert_coords_dict = {bonvis_settings.ly_type: new_coords}
        new_edge_coords = get_edge_coords(edge_dict, vert_coords_dict, nVerts=new_coords.shape[0])
        bonvis_settings.upd_edge_coords[bonvis_settings.ly_type] = new_edge_coords[bonvis_settings.ly_type]
        self.get_coords(bonvis_settings, bonvis_data, bonvis_metadata)
        self.update_nn_index(bonvis_metadata.tree_info, leaf_size=10)

    def update_nn_index(self, tree_info, leaf_size=20):
        self.nn_index = {}
        self.nn_index['int_inds'] = KDTree(self.node_coords_nx[tree_info['int_inds']], leaf_size=leaf_size,
                                           metric='euclidean')
        self.nn_index['cell_inds'] = KDTree(self.node_coords_nx[tree_info['cell_inds']], leaf_size=leaf_size,
                                            metric='euclidean')

    def get_bg_coords(self, transf_info, first_line=0.4, n_lines=5, n_coords=20):
        # Get the data-scale of how far the lines should be apart
        data_scale = invert_poincare_transform(np.array([first_line, 0]), origin=np.array([0., 0.]),
                                               zoom=transf_info.get_zoom())[0]

        # Get at what distances in the data the next lines should appear
        data_line_dists = data_scale * np.arange(1, n_lines + 1)
        data_line_dists = np.hstack((-data_line_dists, data_line_dists))
        max_coord = data_line_dists[-1]

        # Make these lines in the data-coordinates
        # We need the maximal coordinates
        source_coords = np.zeros((4 * n_lines, 2))
        target_coords = np.zeros((4 * n_lines, 2))

        # Horizontal lines:
        source_coords[:2 * n_lines, 0] = -max_coord
        source_coords[:2 * n_lines, 1] = data_line_dists
        target_coords[:2 * n_lines, 0] = max_coord
        target_coords[:2 * n_lines, 1] = data_line_dists

        # Vertical lines:
        source_coords[2 * n_lines:, 1] = -max_coord
        source_coords[2 * n_lines:, 0] = data_line_dists
        target_coords[2 * n_lines:, 1] = max_coord
        target_coords[2 * n_lines:, 0] = data_line_dists

        bg_edge_coords_eix = np.linspace(source_coords, target_coords, n_coords, axis=1)

        self.transf_bg_edge_coords_eix = transform_coords_poincare(bg_edge_coords_eix, origin=np.array([0., 0.]),
                                                                   zoom=transf_info.get_zoom(),
                                                                   transform=(transf_info.geometry == 'hyperbolic'))

    def bg_data_to_ax(self, ax):
        shape = self.transf_bg_edge_coords_eix.shape
        flattened = self.transf_bg_edge_coords_eix.reshape((-1, 2))
        transformed_flattened = ax.transAxes.inverted().transform(ax.transData.transform(flattened))
        self.data_bg_edge_coords_eix = transformed_flattened.reshape(shape)

    def transform_coords(self, transf_info):
        if transf_info.no_transform:
            self.transf_node_coords_nx = self.node_coords_nx[:].copy()
            self.transf_edge_coords_eix = self.edge_coords_eix[:].copy()
            # if self.bg_edge_coords_eix is not None:
            #     self.bg_transf_edge_coords_eix = self.bg_edge_coords_eix.copy()
            return

        # if (transf_info.origin is None) or (transf_info.get_zoom() is None):
        #     transf_info.get_zoom_origin(node_coords_nx, edge_df_dict)
        if (transf_info.get_zoom() is None) or (transf_info.origin is None):
            # Get zoom and origin for transformation, copying from transf_info if known or determining them anew
            transf_info.get_zoom_origin(self.node_coords_nx, None)
        
        # Once origin and zoom are known, Transform node coords to poincare disc
        self.transf_node_coords_nx = transform_coords_poincare(self.node_coords_nx, origin=transf_info.origin,
                                                               zoom=transf_info.get_zoom(),
                                                               transform=(transf_info.geometry == 'hyperbolic'))

        # Transform coordinates to Poincare disc for edges
        self.transf_edge_coords_eix = transform_coords_poincare(self.edge_coords_eix, origin=transf_info.origin,
                                                                zoom=transf_info.get_zoom(),
                                                                transform=(transf_info.geometry == 'hyperbolic'))

        # If necessary, transform background
        # if transf_info.geometry == 'hyperbolic' and (self.bg_edge_coords_eix is not None):
        #     self.bg_transf_edge_coords_eix = transform_coords_poincare(self.bg_edge_coords_eix,
        #                                                                origin=np.array([0, 0]),
        #                                                                zoom=transf_info.get_zoom(),
        #                                                                transform=(transf_info.geometry == 'hyperbolic'))


"""Helper-functions"""

"""Function for doing hyperbolic disk transformation"""


def transform_coords_poincare(coords, origin=np.zeros(2), zoom=1, transform=True, only_radii=False):
    # TODO: Clean this up
    if not transform:
        new_coords = coords - origin
        new_coords *= zoom
        new_coords[new_coords < -1] = -1.02
        new_coords[new_coords > 1] = 1.02
        # new_coords = np.minimum(np.maximum(new_coords, -1), 1)
        # large_radii = radii > 1
        if only_radii:
            radii = np.sqrt(np.sum(new_coords ** 2, axis=-1))
            return radii
        #     radii[large_radii] = 1.1
        #     return radii
        # new_coords[large_radii] /= np.expand_dims(radii[large_radii], axis=-1)
        # new_coords[large_radii] *= 1.1
        # coords_transformed = np.zeros(new_coords.shape)
        # coords_transformed[nonzeros, :] = radii[nonzeros, np.newaxis] * (
        #         new_coords[nonzeros, :] / np.linalg.norm(new_coords[nonzeros, :], axis=-1, ord=2)[:, np.newaxis])
        return new_coords
    else:
        poinc_coords = coords - origin
        poinc_coords *= zoom
        radii_euclidean_sq = np.sum(poinc_coords ** 2, axis=-1)
        if only_radii:
            return np.sqrt(radii_euclidean_sq) / (1 + np.sqrt(1 + radii_euclidean_sq))
        poinc_coords /= (1 + np.sqrt(1 + np.expand_dims(radii_euclidean_sq, axis=-1)))
        return poinc_coords


def invert_poincare_transform(coords, origin, zoom, transform=True):
    if transform:
        radii_sq = np.sum(coords ** 2, axis=-1)
        coords = coords * 2 / (1 - np.expand_dims(radii_sq, axis=-1))
    return coords / zoom + origin


"""Functions for getting optimal initial zoom and origin position."""


def get_opt_zoom_origin_poincare(coords, transf_info, frac_within=0.8, within_radius=0.8, tol=1e-6, max_iter=20,
                                 verbose=False):
    converged = False
    old_zoom = 1
    origin0 = np.mean(coords, axis=0)
    counter = 0
    while not converged:
        counter += 1
        transf_info.origin = origin0
        new_zoom = get_zoom_given_origin(coords, transf_info, frac_within=frac_within,
                                         within_radius=within_radius)
        origin0 = get_centroid_poincare(coords, transf_info, zoom=new_zoom, verbose=verbose)
        if verbose:
            logging.debug('Zoom is ' + str(new_zoom))
            logging.debug('Origin is ' + str(origin0) + '\n')
        if np.abs(old_zoom - new_zoom) < tol:
            converged = True
        else:
            old_zoom = new_zoom
        if counter == max_iter:
            logging.debug("Maximum number of iterations reached: just returning last found value.")
            logging.debug("Zoom value converged to " + str(np.abs(old_zoom - new_zoom)))
            converged = True
    return new_zoom, origin0


def get_zoom_given_origin(coords, transf_info, frac_within=0.8, within_radius=.8):
    n_points = coords.shape[0]
    n_points_within = int(np.ceil(frac_within * n_points))

    def radius_needed(logzoom):
        zoom = np.exp(logzoom)
        needed_radius = np.sort(transform_coords_poincare(coords, origin=transf_info.origin,
                                                          zoom=zoom, transform=(transf_info.geometry == 'hyperbolic'),
                                                          only_radii=True))[n_points_within - 1] - within_radius
        return needed_radius

    bracket = (-6, 1)
    bracket_ok = False
    while not bracket_ok:
        if radius_needed(bracket[0]) >= 0:
            bracket = (bracket[0] - 1, bracket[1])
        elif radius_needed(bracket[1]) <= 0:
            bracket = (bracket[0], bracket[1] + 1)
        else:
            bracket_ok = True

    opt_result = optimize.root_scalar(radius_needed, bracket=bracket)
    if opt_result.converged:
        result = np.exp(opt_result.root)
    else:
        logging.debug(opt_result.flag)
        logging.debug("Finding optimal zoom failed. Using 0.001 instead.")
        result = 0.001

    return result


def get_centroid_poincare(coords, transf_info, zoom=1, verbose=False):
    origin0 = np.mean(coords, axis=0)

    def dist_fun(ori):
        return np.mean(
            transform_coords_poincare(coords, origin=ori, zoom=zoom, transform=(transf_info.geometry == 'hyperbolic'),
                                      only_radii=True))

    opt_result = optimize.minimize(dist_fun, origin0)
    if opt_result.success:
        if verbose:
            logging.debug(opt_result.message)
        origin = opt_result.x
    else:
        logging.debug(opt_result.message)
        logging.debug("We take the centroid of the graph in Euclidean space instead.")
        origin = origin0
    return origin


# Style helper functions
def get_celltype_colors_new(n_celltypes=None, colortype=None, gradientType='YlOrRd'):
    if (colortype is None) and (n_celltypes <= 8):
        col_HSC = "#0B5345"  # darkgreen
        col_MPP = "#229954"  # green
        col_LMPP = "#48C9B0"  # turchqoise
        col_CMP = "#AF601A"
        # col_UNK = "#E5E7E9"
        col_MEP = "#FE776D"
        col_pDC = "#A690A4"
        col_GMP = "#FCD0A1"
        col_CLP = "#AFD2E9"

        celltype_colors = [col_CLP, col_CMP, col_GMP, col_HSC, col_LMPP, col_MEP, col_MPP, col_pDC]
        celltype_colors = colors.ListedColormap(celltype_colors)
    elif ((colortype is None) or (colortype == 'tab10')) and (n_celltypes <= 10):
        celltype_colors = cm.get_cmap('tab10')
    elif ((colortype is None) or (colortype == 'tab20')) and (n_celltypes <= 20):
        celltype_colors = cm.get_cmap('tab20')
    elif ((colortype is None) and (n_celltypes <= 40)):
        tab20b_colors = plt.cm.tab20b(np.linspace(0, 1, 20))
        tab20c_colors = plt.cm.tab20c(np.linspace(0, 1, 20))
        combined_colors = np.empty((40, 4))
        combined_colors[:20] = tab20b_colors  # Even indices: tab20
        combined_colors[20:] = tab20c_colors
        celltype_colors = colors.ListedColormap(combined_colors, name="tab20b_tab20c_alternating")
    elif (colortype is None):
        celltype_colors, cols = categorical_cmap(10, int(np.ceil(n_celltypes / 10)), cmap="tab10")
    elif colortype == 'offOn':
        tab10 = cm.get_cmap('tab10')
        two_colors = tab10([1, 2])
        two_colors[1, :] = cm.get_cmap('gray')(0.75)
        celltype_colors = colors.ListedColormap(two_colors)
    elif colortype == 'gradient_colormap':
        cmap = cm.get_cmap(gradientType)
        return cmap
    else:
        gradient = cm.get_cmap('hsv')
        celltype_colors = colors.ListedColormap(gradient(np.linspace(0, 1, n_celltypes)))
    return celltype_colors


def get_edge_coords(edge_dict, vert_coords_dict, nVerts=None):
    edge_coords_dict = {}
    for ly_type in vert_coords_dict:
        if ly_type.startswith('ly_dendrogram'):
            vert_coords = vert_coords_dict[ly_type]
            sources = edge_dict['source_ind']
            targets = edge_dict['target_ind']
            edge_coords = np.zeros((len(sources), 3, 2))
            source_coords = vert_coords[sources, :]
            target_coords = vert_coords[targets, :]
            edge_coords[:, 0, :] = source_coords
            edge_coords[:, 2, :] = target_coords
            edge_coords[:, 1, 0] = source_coords[:, 0]
            edge_coords[:, 1, 1] = target_coords[:, 1]
            edge_coords_dict[ly_type] = edge_coords
        else:
            vert_coords = vert_coords_dict[ly_type]
            n_coords = 40 if (nVerts is None) or (nVerts < 20000) else 2
            sources = edge_dict['source_ind']
            targets = edge_dict['target_ind']
            source_coords = vert_coords[sources, :]
            target_coords = vert_coords[targets, :]
            edge_coords_dict[ly_type] = np.linspace(source_coords, target_coords, n_coords, axis=1)
    return edge_coords_dict


# Merge cells that are at distance zero
def merge_cells_at_zero_dist(scData, cell_id_to_cs_id=None):
    # Add information of cell mapping to nodes
    # cellInd -> vertInd -> nodeId -> node
    for _, node in scData.tree.vert_ind_to_node.items():
        node.cell_inds = []
        node.cs_inds = []
    for cell_ind, vert_ind in scData.cellIndToVertInd.items():
        node = scData.tree.vert_ind_to_node[vert_ind]
        node.cell_inds.append(cell_ind)
    for cs_ind, vert_ind in scData.csIndToVertInd.items():
        node = scData.tree.vert_ind_to_node[vert_ind]
        node.cs_inds.append(cs_ind)

    merge_cells_at_zero_dist_node(scData.tree.root, scData)
    scData.nVerts = 0
    scData.tree.vert_ind_to_node, scData.nVerts = scData.tree.root.renumber_verts(vertIndToNode={},
                                                                                  vert_count=0,
                                                                                  old_ind_to_new_ind=None)

    scData.tree.nNodes = scData.nVerts
    # scData.csIndToVertInd = {cs_ind: old_ind_to_new[vert_ind] for cs_ind, vert_ind in scData.csIndToVertInd.items()}
    # We make up some new vertex names
    # scData.vert_names = ['vert_' + str(ind) for ind in range(scData.nVert)]

    # scData.nodeIdToNode = {}
    # scData.cellsToVerts = None
    scData.cellIndToVertInd = {}
    scData.nCellsPerVert = np.zeros(scData.nVerts, dtype=int)
    scData.vertIndToCellInds = {}
    for vert_ind, node in scData.tree.vert_ind_to_node.items():
        # scData.vertIndToNodeId[vert_ind] = node.nodeId
        # scData.nodeIdToNode[node.nodeId] = node
        scData.nCellsPerVert[vert_ind] = len(node.cell_inds)
        scData.vertIndToCellInds[vert_ind] = node.cell_inds
        for cell_ind in node.cell_inds:
            # scData.cellsToVerts[scData.metadata.cellIds[cell_ind]] = scData.vert_names[vert_ind]
            scData.cellIndToVertInd[cell_ind] = vert_ind

    scData.csIndToVertInd = {}
    scData.nCssPerVert = np.zeros(scData.nVerts, dtype=int)
    scData.vertIndToCsInds = {}
    for vert_ind, node in scData.tree.vert_ind_to_node.items():
        scData.nCssPerVert[vert_ind] = len(node.cs_inds)
        scData.vertIndToCsInds[vert_ind] = node.cs_inds
        for cs_ind in node.cs_inds:
            scData.csIndToVertInd[cs_ind] = vert_ind


def merge_cells_at_zero_dist_node(self_node, scData):
    childrenToBeAdded = []
    childIndsToBeDeleted = []
    for ind, child in enumerate(self_node.childNodes):
        merge_cells_at_zero_dist_node(child, scData)
        # TODO: Change this to 0 again.
        if child.tParent == 0:
            childrenToBeAdded += child.childNodes
            childIndsToBeDeleted.append(ind)
            # Merge node-ids
            if len(self_node.cell_inds):
                self_node.nodeId += '_{}'.format(child.nodeId)
            else:
                self_node.nodeId = child.nodeId

            # Add cell_inds
            self_node.cell_inds += child.cell_inds
            self_node.cs_inds += child.cs_inds

            # Adapt nVert
            scData.nVerts -= 1
            scData.tree.nNodes -= 1
    if len(childIndsToBeDeleted) > 0:
        for child in childrenToBeAdded:
            child.parentNode = self_node
        self_node.childNodes = [child for ind, child in enumerate(self_node.childNodes) if
                                ind not in childIndsToBeDeleted]
        self_node.childNodes += childrenToBeAdded
    self_node.isLeaf = (len(self_node.childNodes) == 0)
    self_node.isCell = (len(self_node.cell_inds) > 0)


# def process_cell_merge(self_node, scData, vert_ind_to_node):
#     vert_ind = scData.nVert
#     vert_ind_to_node[vert_ind] = self_node
#     self_node.nodeInd = vert_ind
#     scData.nVert += 1
#     for child in self_node.childNodes:
#         process_cell_merge(child, scData, vert_ind_to_node)
#     return vert_ind_to_node


def set_transform_ellipse_collection(coll, transform):
    """This should usually have been done through coll.set_offset_transform(), but somehow that doesn't work. Therefore
    I checked the source code in the matplotlib package and the following should do the trick."""
    coll._transOffset = transform
    coll._offsets = coll._uniform_offsets
    coll._uniform_offsets = None


def color_code_to_color(dummy, color_codes=None):
    colors = []
    for ind in range(len(dummy)):
        colors.append('background-color:{};width:2em'.format(color_codes[ind]))
    return colors


def find_annot(new_annot_label, bonvis_settings, bonvis_data, bonvis_metadata):
    celltype_info = bonvis_settings.celltype_info
    annot_objs = [annot_info for annot, annot_info in celltype_info.annot_infos.items() if
                  annot_info.label == new_annot_label]
    found_it = False
    if len(annot_objs) == 1:
        found_it = True
        return annot_objs[0]
    if not found_it:
        verttype_info = bonvis_settings.verttype_info
        annot_objs = [annot_info for annot, annot_info in verttype_info.annot_infos.items() if
                      annot_info.label == new_annot_label]
        if len(annot_objs) == 1:
            found_it = True
            return annot_objs[0]

    if not found_it:
        feature_path = bonvis_settings.node_style['feature_path']
        if feature_path is not None:
            feature_hdf = bonvis_data[feature_path]
            feature_ids = json.loads(feature_hdf.attrs['gene_ids'])
            feature_tuples = [(feature_ind, feature_name) for feature_ind, feature_name in
                              enumerate(feature_ids) if feature_name == new_annot_label]
        else:
            feature_tuples = []
        if len(feature_tuples) == 1:
            found_it = True
            feature_ind, feature_name = feature_tuples[0]
            if feature_hdf['means'].shape[1] == bonvis_metadata.n_cells:
                cells_or_verts = 'cells'
            elif feature_hdf['means'].shape[1] == bonvis_metadata.n_Css:
                cells_or_verts = 'cellstates'
            else:
                cells_or_verts = 'verts'
            annot_info = get_feature_annot_info(feature_ind=feature_ind, feature_name=feature_name,
                                                info_object=feature_path, cells_or_verts=cells_or_verts,
                                                gradient_type=bonvis_settings.node_style['gradient_type'])
            return annot_info
    return None


class Bonvis_metadata:
    """
    This class contains attributes from bonvis_data that are not large np-arrays so can be read in all at once:
    dataset
    node_ids
    n_nodes
    gene_ids
    n_genes
    cell_ids
    n_cells
    gene_info_df
    cell_info
    metadata
    feature_paths
    clusters_cut_edges # Info on clustering
    """

    def __init__(self, bonvis_data_hdf_path):
        bonvis_data_hdf = h5py.File(bonvis_data_hdf_path, 'r')
        metadata = json.loads(bonvis_data_hdf.attrs['metadata_json'])
        # Store ids of nodes and ids of genes (and ids of cells)
        self.node_ids = metadata['nodeIds']
        self.gene_ids = metadata['geneIds']
        self.cell_ids = metadata['cellIds']
        self.cs_ids = metadata['csIds']
        self.dataset = metadata['dataset']
        self.n_cells = metadata['nCells']
        self.n_Css = metadata['nCss']
        self.n_nodes = len(self.node_ids)
        self.n_genes = len(self.gene_ids) if self.gene_ids is not None else 0
        if 'cell_ind_to_cs_ind' in metadata:
            self.cell_ind_to_cs_ind = metadata['cell_ind_to_cs_ind']
        if 'cs_ind_to_cell_inds' in metadata:
            self.cs_ind_to_cell_inds = metadata['cs_ind_to_cell_inds']
        self.feature_paths = json.loads(bonvis_data_hdf.attrs['feature_paths'])

        self.clusters_cut_edges = {}
        # Store dataframe with gene-names, ids and gene-variances
        # gene_info_hdf = bonvis_data_hdf['gene_info']
        # self.gene_info_df = pd.DataFrame.from_dict(
        #     {'ids': self.gene_ids, 'zscores': gene_info_hdf['zscores'],
        #      'gene_variances': gene_info_hdf['gene_variances']})
        # self.gene_info_df.sort_values(by='zscores', axis=0, ascending=False, inplace=True)
        # self.no_variation_genes = [self.gene_ids[ind] for ind in gene_info_hdf['no_variation_genes']]
        # Read in dataframe with cell information
        bonvis_data_hdf.close()
        cell_info_dict = pd.read_hdf(bonvis_data_hdf_path, key='cell_info/cell_info_dict').to_dict(orient='list')
        try:
            cs_info_dict = pd.read_hdf(bonvis_data_hdf_path, key='cs_info/cs_info_dict').to_dict(orient='list')
            cluster_info_dict = pd.read_hdf(bonvis_data_hdf_path, key='cs_info/cluster_info_dict').to_dict(
                orient='list')
        except KeyError:
            logging.warning("No key {} in file {}\nThis is likely not a problem, it might merely indicate that you use "
                            "an old run of vis_bonsai_preprocess.py.".format('cs_info/cs_info_dict',
                                                                             bonvis_data_hdf_path))
            cs_info_dict = None
        bonvis_data_hdf = h5py.File(bonvis_data_hdf_path, 'r')
        single_at_vert = bonvis_data_hdf['cell_info/single_at_vert'][:]
        single_cs_at_vert = bonvis_data_hdf['cs_info/single_cs_at_vert'][:]
        multi_at_vert = bonvis_data_hdf['cell_info/multi_at_vert'][:]
        multi_cs_at_vert = bonvis_data_hdf['cs_info/multi_cs_at_vert'][:]

        # Read in vert information
        vert_info_hdf = bonvis_data_hdf['vert_info']
        n_cells_per_vert = vert_info_hdf['n_cells_per_vert'][:]

        int_vert_inds = np.where(n_cells_per_vert == 0)[0]
        non_int_vert_inds = np.setxor1d(int_vert_inds, np.arange(self.n_nodes), assume_unique=True)
        self.cell_info = {'cell_info_dict': cell_info_dict, 'int_vert_inds': int_vert_inds,
                          'non_int_vert_inds': non_int_vert_inds,
                          'single_at_vert': single_at_vert, 'multi_at_vert': multi_at_vert}
        self.cs_info = {'cs_info_dict': cs_info_dict, 'cluster_info_dict': cluster_info_dict,
                        'single_cs_at_vert': single_cs_at_vert,
                        'multi_cs_at_vert': multi_cs_at_vert}

        vert_ind_to_cell_inds = json.loads(vert_info_hdf.attrs['vert_ind_to_cell_inds_json'])
        n_css_per_vert = vert_info_hdf['n_css_per_vert'][:]
        vert_ind_to_cs_inds = json.loads(vert_info_hdf.attrs['vert_ind_to_cs_inds_json'])
        self.vert_info = {'n_cells_per_vert': n_cells_per_vert, 'vert_ind_to_cell_inds': vert_ind_to_cell_inds,
                          'n_css_per_vert': n_css_per_vert, 'vert_ind_to_cs_inds': vert_ind_to_cs_inds}

        # self.metadata = json.loads(bonvis_data_hdf.attrs['metadata_json'])

        nwk_str = bonvis_data_hdf['tree_info'].attrs['nwk_str']
        n_leafs = bonvis_data_hdf['tree_info'].attrs['n_leafs']
        cell_inds = bonvis_data_hdf['tree_info/cell_inds'][:]
        cs_inds = bonvis_data_hdf['tree_info/cs_inds'][:]
        int_inds = bonvis_data_hdf['tree_info/int_inds'][:]
        multi_cs_inds = bonvis_data_hdf['tree_info/multi_cs_inds'][:]
        multi_cell_inds = bonvis_data_hdf['tree_info/multi_cell_inds'][:]
        try:
            n_cells_per_vert = bonvis_data_hdf['tree_info/n_cells_per_vert'][:]
        except KeyError:
            print("Warning: Field n_cells_per_vert doesn't exist yet in tree_info. This is necessary for clustering,"
                  "\nrerun vis_bonsai_preprocess.py to get this field.")
        node_id_to_n_cells = {node_id: n_cells_per_vert[ind] for ind, node_id in enumerate(self.node_ids)}
        bonvis_data_hdf.close()
        edge_dict = pd.read_hdf(bonvis_data_hdf_path, key='tree_info/edge_df').to_dict(orient='list')

        self.tree_info = {'nwk_str': nwk_str, 'cell_inds': cell_inds, 'int_inds': int_inds, 'edge_dict': edge_dict,
                          'n_leafs': n_leafs, 'cs_inds': cs_inds, 'multi_cell_inds': multi_cell_inds,
                          'multi_cs_inds': multi_cs_inds, 'node_id_to_n_cells': node_id_to_n_cells}


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap, cols


def get_placeholder_fig():
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    # Add multi-line centered text
    text = "Bonsai-scout is loading the tree representation of your data.\n\n" \
    "For large datasets, this can take a while. Please be patient."
    ax.text(0, 0, text, ha='center', va='center', fontsize=16)
    return fig
