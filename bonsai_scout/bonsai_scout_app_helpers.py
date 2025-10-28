from datetime import datetime
import json
import h5py
import logging
import numpy as np
import os
from pathlib import Path
import pandas as pd
from ruamel.yaml import YAML
from shiny import ui
import sys

from downstream_analyses.get_cluster_helpers import Cluster_Tree

JOBS_DIR = "/scicore/web/scismara/scismara/www/SCISMARA/jobs"
logging.getLogger("websockets").setLevel(logging.WARNING)

# Some global constants are set that determine behavior
TEMP_MASK_STEPS = 10
TEMP_MASK_SECS = 3

from bonsai_scout.bonsai_scout_helpers import Bonvis_figure, Bonvis_settings, Bonvis_metadata
from bonsai.bonsai_helpers import set_recursion_limits

def store_current_settings(bonvis_settings, settings_path):
    logging.debug("settings_path %r", settings_path)
    dir_path = os.path.dirname(settings_path)
    logging.debug("dir_path %r", dir_path)
    new_settings_path = os.path.join(dir_path,
                                     "bonsai_vis_settings_{}.json".format(datetime.now().strftime('%y%m%d_%H%M%S')))
    logging.debug("new_settings_path %r", new_settings_path)
    bonvis_settings.to_json(settings_path=new_settings_path)

global bonvis_data_objects
bonvis_data_objects = {}

# curvature_info_text = ui.div(
#     ui.p("The hyperbolic layout maps 2D-plane to the unit disk. " \
#          "Distances between points near the boundary of the disk get more and more compressed. " \
#          "This is illustrated by the squares in the background, which are all the same size."),
#     ui.p("The flat geometry does not transform distances, but plots all points and lines that are outside of \
#                  the field of view on the enveloping square.")
# )


def get_feature_info_display(bonvis_data, feature_path):
    if feature_path is None or (len(feature_path) == 0):
        return None
    feature_hdf = bonvis_data[feature_path]
    if 'zscores' in feature_hdf.keys():
        zscore_vals = feature_hdf['zscores'][:]
        if zscore_vals.max() < .01:
            neg_pow = -int(np.floor(np.log10(zscore_vals.max())))
            zscore_vals_norm = zscore_vals * 10 ** neg_pow
        else:
            neg_pow = 0
            zscore_vals_norm = zscore_vals
        zscore_header = 'zscores (*10^(-{})'.format(neg_pow) if neg_pow > 0 else 'zscores'
        feature_display = pd.DataFrame({'ids': json.loads(feature_hdf.attrs['gene_ids']),
                                          zscore_header: zscore_vals_norm})
        feature_display.sort_values(by=zscore_header, axis=0, ascending=False, inplace=True)
    elif 'variances' in feature_hdf.keys():
        variance_vals = feature_hdf['variances'][:]
        if variance_vals.max() < .01:
            neg_pow = -int(np.floor(np.log10(variance_vals.max())))
            variance_vals_norm = variance_vals * 10 ** neg_pow
            print(neg_pow)
        else:
            neg_pow = 0
            variance_vals_norm = variance_vals
        variance_header = 'variances (*10^(-{})'.format(neg_pow) if neg_pow > 0 else 'variances'
        print(variance_header)
        feature_display = pd.DataFrame({'ids': json.loads(feature_hdf.attrs['gene_ids']),
                                        variance_header: variance_vals_norm})
        feature_display.sort_values(by=variance_header, axis=0, ascending=False, inplace=True)
    else:
        feature_display = pd.DataFrame({'ids': json.loads(feature_hdf.attrs['gene_ids'])})
    feature_display = feature_display.round(2)
    return feature_display

class BonvisObjects:
    def __init__(self):
        self.objects = {}

    def __getitem__(self, key):
        if key not in self.objects:
            self.objects[key] = BonvisObject(key)
        return self.objects[key]
    
    def __delitem__(self, key):
        if key in self.objects:
            del self.objects[key]

    def keys(self):
        return self.objects.keys()

class BonvisObject:
    def __init__(self, key):
        self.shiny_paths = {}
        try:
            data = key[1].strip().split("=")
        except:
            logging.debug("Could not process key: %r" % key)
            data = [None]
        if data[0] != "?dir":
            if 'BONSAI_DATA_PATH' in os.environ and 'BONSAI_SETTINGS_PATH' in os.environ:
                self.shiny_paths = {'data_path': os.environ['BONSAI_DATA_PATH'],
                                    'settings_path': os.environ["BONSAI_SETTINGS_PATH"]}
            else:
                raise BaseException("Could not find bonvis data! %r" % key[1])
        if not self.shiny_paths:
            try:
                dataset_id = data[1]
                yaml = YAML()
                with open(os.path.join(JOBS_DIR, dataset_id, "shiny_configs.yaml"), 'r') as file_obj:
                    self.shiny_paths = yaml.load(file_obj)
            except:
                logging.debug("Couldn't find bonvis config file for {}".format(dataset_id))
        self.bonvis_metadata = Bonvis_metadata(self.shiny_paths['data_path'])

        set_recursion_limits(int(2 * self.bonvis_metadata.n_cells))
        self.bonvis_settings = Bonvis_settings(load_settings_path=self.shiny_paths['settings_path'])
        if not (self.shiny_paths['data_path'] in bonvis_data_objects):
            bonvis_data_objects[self.shiny_paths['data_path']] = h5py.File(self.shiny_paths['data_path'], 'r')
        self.bonvis_data_hdf = bonvis_data_objects[self.shiny_paths['data_path']]

        self.bonvis_fig = Bonvis_figure(self.bonvis_data_hdf, self.bonvis_metadata, bonvis_data_path=self.shiny_paths['data_path'],
                                        bonvis_settings=self.bonvis_settings)
        
        # Some changes in the app are counted, such that changes can be detected
        self.click_counters = {'cluster': 0, 'reset_navi': 0, 'crop': 0, 'reset_crop': 0,
                        'tweak': 0, 'flip': 0, 'reset_flip': 0, 'nodes_bigger': 0, 
                        'nodes_smaller': 0, 'more_curve': 0, 'less_curve': 0, 'zoom_in': 0,
                        'zoom_out': 0, 'marker': 0, 'trigger_marker': 0, 'open_marker_gene_warning': 0,
                        'fig': 0, 'marker_info': 0, 'edges_thicker': 0, 'edges_thinner': 0,
                        'node_edges_thicker': 0, 'node_edges_thinner': 0
                        }
        self.mask_counters = {'temp_mask': [11] * 3,  # Variable to track if mask should be turned off in couple of seconds
                        'redraw_mask': 0,  # tracking number of redraws (in figure) done
                        'update_mask': 0,  # tracking number of updates of mask done
                        'set_temp_mask': 0  # tracking number of times mask was set temporarily
                        }
        # Marker gene calculation needs some communication between functions
        self.marker_gene_vars = {'run_with_vars': None, 'marker_genes_tuple': None, 'marker_info_dict': None,
                                 'start_time_modal': None, 'time_estimate': None}
        # We track which branches are flipped
        self.flip_ids = []
        # This is the old position of the rectangle for zooming
        self.old_brush = None
        
        self.annot_infos = dict(self.bonvis_fig.bonvis_settings.celltype_info.annot_infos,
                                **self.bonvis_fig.bonvis_settings.verttype_info.annot_infos)
        self.annotation_dict = {}
        for annot, annot_info in self.annot_infos.items():
            if hasattr(annot_info, 'hidden') and annot_info.hidden:
                continue
            self.annotation_dict[annot_info.label] = annot_info.label

        self.size_annotation_dict = {}
        for annot, annot_info in self.annot_infos.items():
            if hasattr(annot_info, 'hidden') and annot_info.hidden:
                continue
            if annot_info.color_type != 'sequential':
                continue
            if annot_info.annot_type == 'cells' and (len(self.bonvis_metadata.cell_info['multi_at_vert']) > 0):
                continue
            if annot_info.annot_type == 'cellstates' and (len(self.bonvis_metadata.cs_info['multi_cs_at_vert']) > 0):
                continue
            self.size_annotation_dict[annot_info.label] = annot_info.label

        self.ly_types_dictionary = {'ly_eq_angle': 'Equal angle', 'ly_eq_daylight': 'Equal daylight', 'ly_dendrogram': 'Dendrogram',
                                    'ly_dendrogram_ladderized': 'Dendrogram (ladderized)'}
        self.layout_types_dict = {}
        for ly_type in self.bonvis_settings.ly_types:
            if ly_type in self.ly_types_dictionary:
                self.layout_types_dict[ly_type] = self.ly_types_dictionary[ly_type]
            else:
                self.layout_types_dict[ly_type] = ly_type

        self.feature_dict = {feature_path: feature_path[5:] for feature_path in self.bonvis_metadata.feature_paths}
        self.init_geometry = self.bonvis_fig.bonvis_settings.transf_info.geometry
        self.init_layout = self.bonvis_fig.bonvis_settings.ly_type
        self.init_node_style = self.bonvis_fig.bonvis_settings.node_style['annot_info'].label
        self.init_size_style = self.bonvis_fig.bonvis_settings.node_style['size_annot_info'].label
        self.init_selected_annot = 'no_subset'
        self.init_switch_mask = False
        self.init_options_accordion = 'Annotation'
        self.init_annotation_cats = self.bonvis_fig.bonvis_settings.node_style['annot_info'].cats
        self.init_categorical_annot = self.bonvis_fig.bonvis_settings.node_style['annot_info'].label
        self.init_feature_path = self.bonvis_fig.bonvis_settings.node_style['feature_path']
        self.feature_display = get_feature_info_display(self.bonvis_fig.bonvis_data,
                                                        self.bonvis_fig.bonvis_settings.node_style['feature_path'])
        self.max_n_clusters = int(np.max([int(annot_alt.split('annot_cluster_n')[-1]) for 
                                      annot_alt in self.bonvis_fig.bonvis_settings.celltype_info.annot_alts 
                                      if annot_alt.startswith('annot_cluster_n')]))

        self.old_orig = np.array([0, 0])
        self.old_node_style = self.bonvis_fig.bonvis_settings.node_style['annot_info'].label
        self.old_size_style = self.init_size_style

        self.is_big_dataset = self.bonvis_fig.bonvis_metadata.n_nodes > 20000

        """Initial state of variables is stored, such that original state of app can be restored by reset-button."""
        self.orig_zoom = self.bonvis_fig.bonvis_settings.transf_info.zoom_per_geometry.copy()
        self.orig_origin = self.bonvis_fig.bonvis_settings.transf_info.origin
        self.orig_geometry = {self.bonvis_fig.bonvis_settings.ly_type: self.bonvis_fig.bonvis_settings.transf_info.geometry}

        ## get longest path # TODO store this also in bonvis settings, i.e add to preprocessing
        self.cluster_tree = Cluster_Tree()
        self.cluster_tree.from_newick_string(nwk_str=self.bonvis_fig.bonvis_metadata.tree_info['nwk_str'])
        self.longest_path_from_root_to_leaf, self.shortest_path_from_root_to_leaf = self.cluster_tree.root.find_longest_path_between_two_leafs()
        self.num_clusters = None
        self.old_numbers_of_clusters_saved = 0 # keeping track of how many clusters got saved so far
