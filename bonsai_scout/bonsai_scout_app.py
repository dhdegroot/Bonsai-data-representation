import io
import faicons as fa
import h5py
import numpy as np
import os
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML
from shiny import App, reactive, render, req, ui, Session
from shiny.types import SilentException
import shinyswatch
from shinyswatch.theme import minty as shiny_theme
import sys
import re
import json
import time
import asyncio
import concurrent.futures


import logging
FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s'
log_level = logging.WARNING
log_level = logging.DEBUG
logging.basicConfig(format=FORMAT, datefmt='%m-%d %H:%M:%S',
                    level=log_level)

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
#os.chdir(parent_dir)

# TODO: REMOVE THIS LATER. Just for debugging.
# results_folder = '/Users/Daan/Documents/postdoc/collaborations/westendorp_CHKi/bonsai_cellstates_clustered_new'
# results_folder = '/Users/Daan/Documents/postdoc/Private-bonsai/results/hao_satija_2021_paper_figure/cs_summed/bonsai'
# results_folder = '/Users/Daan/Documents/postdoc/Private-bonsai/results/hao_satija_2021_paper_figure/mergedgenes_no_TMono_hao_satija/cs_summed/bonsai'
# results_folder = '/Users/Daan/Documents/postdoc/Private-bonsai/results/hao_satija_2021_paper_figure/spr_super_sure/bonsai_scicore_run'
# settings_filename = 'bonsai_vis_settings.json'
# os.environ['BONSAI_DATA_PATH'] = os.path.abspath(os.path.join(results_folder, 'bonsai_vis_data.hdf'))
# os.environ['BONSAI_SETTINGS_PATH'] = os.path.abspath(os.path.join(results_folder, settings_filename))

from bonsai_scout.bonsai_scout_app_helpers import store_current_settings, \
    get_feature_info_display, BonvisObjects, TEMP_MASK_SECS, TEMP_MASK_STEPS

from bonsai_scout.bonsai_scout_helpers import Bonvis_figure, Bonvis_settings, Bonvis_metadata, update_marker_genes_df, \
    get_placeholder_fig

from downstream_analyses.get_cluster_helpers import Cluster_Tree

"""Some initial variables are loaded"""
css_file = os.path.join(Path(__file__).parent, "my-styles.css")
ICONS = {
    "magnify": fa.icon_svg("magnifying-glass", height='1em', width='1em', margin_left='0em', margin_right='0em',
                           position='relative'),
    "gear": fa.icon_svg("gear", height='1em', width='1em', margin_left='0.5em', margin_right='0.5em',
                        position='relative'),
    "house": fa.icon_svg("house", height='1em', width='1em', margin_left='0em', margin_right='0em',
                         position='relative'),
    "info": fa.icon_svg("circle-info", height='1em', width='1em', margin_left='0.5em', margin_right='0.5em',
                        position='relative'),
    "save": fa.icon_svg("floppy-disk", height='1em', width='1em', margin_left='0em', margin_right='0em',
                         position='relative'),
    "license": fa.icon_svg("creative-commons-nc", height='1em', width='1em', margin_left='0em', margin_right='0em',
                         position='relative'),
    "big_minus": fa.icon_svg("square-minus", height='1.4em', width='1.4em', margin_left='0em', margin_right='0em',
                         position='relative'),
    "minus": fa.icon_svg("square-minus", height='1em', width='1em', margin_left='0em', margin_right='0em',
                         position='relative'),
    "big_plus": fa.icon_svg("square-plus", height='1.4em', width='1.4em', margin_left='0em', margin_right='0em',
                         position='relative'),
    "plus": fa.icon_svg("square-plus", height='1em', width='1em', margin_left='0em', margin_right='0em',
                         position='relative'),
}

# Some other variables are initialized to detect changes

"""User-interface is defined here."""
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.HTML("""
<script>
  if (document.domain == "bonsai.unibas.ch") {
  var _paq = window._paq = window._paq || [];
  /* tracker methods like "setCustomDimension" should be called before "trackPageView" */
  _paq.push(['trackPageView']);
  _paq.push(['enableLinkTracking']);
  (function() {
    var u="https://matomo.sib.swiss/";
    _paq.push(['setTrackerUrl', u+'matomo.php']);
    _paq.push(['setSiteId', '231']);
    var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
    g.async=true; g.src=u+'matomo.js'; s.parentNode.insertBefore(g,s);
  })();
        }
</script>
        """),
        # 1. An accordion giving different options for changing the layout of the tree (zooming, layout-types, ...)
        ui.card(
            ui.card_header(ui.HTML('<strong>Information on selected node:</strong>')),
                ui.output_ui("picked_node_id"),
                max_height="50%", full_screen=True, fill=False
            ),
        ui.accordion(
            # 1.1 First accordion panel handles zooming and changing the curvature
            ui.accordion_panel('Viewing',
                ui.input_action_button("open_viewing_info", "ℹ️ Viewing instructions", class_="btn-info"),
                ui.p(height='1em'),
                # 1.1.1 Optional panel to change curvature and relay origin when plotting on hyperbolic disk
                ui.h6("Drag and zoom:"),
                ui.row(
                    ui.column(3, ui.input_action_button('go_crop', ICONS['magnify'], class_="btn-success")),
                    ui.column(3, ui.input_action_button("zoom_out", ICONS['big_minus'], class_="btn-light d-flex justify-content-center align-items-center me-1")),
                    ui.column(3, ui.input_action_button("zoom_in", ICONS['big_plus'], class_="btn-light d-flex justify-content-center align-items-center me-1")),
                    ui.column(3, ui.input_action_button('reset_crop', ICONS['house'])),
                    style="margin-bottom: 1em;"
                ),
                # ui.row(ui.column(3, ui.input_action_button('reset_crop', ICONS['house'])),style="margin-bottom: 1em;"),
                # ui.p(height='.1em'),
                # ui.input_action_button('reset_crop', ICONS['house']),
                # ui.p(height='1em'),
                # ui.span(
                #     ui.input_action_button("zoom_in", ICONS['big_minus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                #     # ui.input_action_button("nodes_smaller", ICONS['minus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                #     # ui.input_action_button("nodes_bigger", ICONS['plus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                #     ui.input_action_button("zoom_out", ICONS['big_plus'], class_="btn-light d-flex justify-content-center align-items-center"),
                #     style="display: flex; flex-wrap: nowrap; gap: 0.3em;"
                # ),
                ui.panel_conditional(
                    "input.ly_type == 'ly_eq_angle' || input.ly_type == 'ly_eq_daylight'",
                    # ui.tooltip(ui.span("Viewing instructions:", ICONS['info'], style='font-weight:bold'),
                    #     ui.div(
                    #         ui.p(
                    #             "Double-click in the plot to change the center of the hyperbolic disk."),
                    #         ui.p("Use the slider and 'Curve!'-button to change the curvature."),
                    #         ui.p(
                    #             "Use the home-button to reset the original center and curvature.")
                    #     ), placement='right', id='info_geometry',
                    # ),
                    ui.h6("Curve the disk"),
                    ui.row(
                        ui.column(3, ui.input_action_button("less_curve", ICONS['big_minus'], class_="btn-light d-flex justify-content-center align-items-center me-1")),
                        # ui.input_action_button("nodes_smaller", ICONS['minus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                        # ui.input_action_button("nodes_bigger", ICONS['plus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                        ui.column(3, ui.input_action_button("more_curve", ICONS['big_plus'], class_="btn-light d-flex justify-content-center align-items-center")),
                        ui.column(6, ui.input_action_button("reset_geometry", ICONS['house'])),
                        style="margin-bottom: 1em;"
                    ),
                    # ui.input_slider("curvature", "Curvature", min=-3, max=3, value=0, step=0.2),
                    # ui.row(
                    #     ui.column(6, ui.input_action_button("go_curvature", "Curve!", class_="btn-success")),
                    #     ui.column(6, ui.input_action_button("reset_geometry", ICONS['house'])),
                    #     style="margin-bottom: 1em;"
                    # ),
                ),
                # ui.div(height='1em'),
                # 1.1.2 Panel for selecting a rectangle of the plot and zooming in on that
                # ui.tooltip(ui.span("Zooming instructions:", ICONS['info'], style='font-weight:bold;margin-top:2em'),
                #     ui.div(
                #         ui.p("Draw a rectangle in the plot and use the 'Zoom!'-button to zoom."),
                #         ui.p("Click next to the selected region to unselect."),
                #         ui.p("Use the home-button to view the whole tree again.")
                #     ), placement='right', id='info_zoom',
                # ),
                ui.accordion(
                    ui.accordion_panel('Tweak visuals.',
                        ui.h6("Scale node size:"),
                        ui.span(
                            ui.input_action_button("nodes_smaller_fast", ICONS['big_minus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                            # ui.input_action_button("nodes_smaller", ICONS['minus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                            # ui.input_action_button("nodes_bigger", ICONS['plus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                            ui.input_action_button("nodes_bigger_fast", ICONS['big_plus'], class_="btn-light d-flex justify-content-center align-items-center"),
                            style="display: flex; flex-wrap: nowrap; gap: 0.3em;"
                        ),
                        ui.h6("Scale edge width:"),
                        ui.span(
                            ui.input_action_button("edges_thinner", ICONS['big_minus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                            ui.input_action_button("edges_thicker", ICONS['big_plus'], class_="btn-light d-flex justify-content-center align-items-center"),
                            style="display: flex; flex-wrap: nowrap; gap: 0.3em;"
                        ),
                        ui.h6("Scale node-edge width:"),
                        ui.span(
                            ui.input_action_button("node_edges_thinner", ICONS['big_minus'], class_="btn-light d-flex justify-content-center align-items-center me-1"),
                            ui.input_action_button("node_edges_thicker", ICONS['big_plus'], class_="btn-light d-flex justify-content-center align-items-center"),
                            style="display: flex; flex-wrap: nowrap; gap: 0.3em;"
                        ),
                    ),
                open=False, id='plot_params'
                ),
            ),
            # 1.2 Changing the type of layout and the geometry of the disk
            ui.accordion_panel('Layout',
                ui.input_action_button("open_layout_info", "ℹ️ Layout instructions", class_="btn-info"),
                ui.p(height='1em'),
                # 1.2.1 Changing the geometry of the disk from hyperbolic to flat
                # ui.input_selectize("geometry",
                #                     ui.tooltip(ui.span("Disk curvature", ICONS['info']),
                #                                 curvature_info_text),
                #                     {'hyperbolic': 'Hyperbolic', 'flat': 'Flat'},
                #                     multiple=False, selected=None),
                ui.input_selectize("geometry", "Disk curvature", {'hyperbolic': 'Hyperbolic', 'flat': 'Flat'}, multiple=False, selected=None),
                # 1.2.2 Changing the type of layout
                ui.input_selectize('ly_type', 'Tree layout', {},
                                    selected=None),
                # 1.2.3 Tweaking layout
                ui.accordion(
                    ui.accordion_panel('Layout tweaking.',
                                        ui.output_text('tweak_text'),
                                        ui.panel_conditional("input.ly_type == 'ly_eq_angle'",
                                                            ui.p(ui.input_slider("more_angle",
                                                                                "Angle increase log2-foldchange:",
                                                                                min=-3, max=3, value=0,
                                                                                step=0.2))
                                                            ),
                                        ui.input_task_button("go_tweak",
                                                                ui.output_text('tweak_button_text'),
                                                                class_="btn-success"),
                                        ui.p(height='10px'),
                                        ui.row(
                                            ui.column(9, ui.input_task_button("go_flip",
                                                                "Flip branch!",
                                                                class_="btn-success")),
                                            ui.column(3, ui.input_task_button("reset_flip", ICONS['house'])),
                                        ),
                                        ),
                    open=False, id='tweaking'
                               ),
                               ),
            ui.accordion_panel('Downloads',
                ui.HTML("<strong>Download the visualization:</strong>"),
                # ui.input_selectize('figure_format', "Feature type:", {},
                #                                 selected=None),
                ui.input_selectize("fig_format", None, choices=["png", "svg"], selected='.png'),
                ui.download_button("tree_download", "Download", class_="btn-primary"),
                ui.p(),
                ui.output_ui("download_page_link"),
            ),
            open=False, id='sidebar_accordion',
        ),
        # TODO: Add these buttons again.
        # 1.2 Reset settings to original, also resets all app settings
        # ui.input_action_button("reset", "Reset settings"),
        # 1.3 Store current settings, such that user can start from the current settings
        # ui.input_action_button("store", "Store settings"),
        open='always',
    ),
    # Theme code - start
    shinyswatch.theme.minty,
    # ui.head_content(ui.include_css(css_file)),
    # Theme code - end
    ui.row(
        ui.column(8,
                  ui.card(
                    #   ui.card_header(
                    #       #ui.output_text("mytitle"),
                    #       'Bonsai tree',
                    #     #   ui.popover(ICONS["gear"],
                    #     #              ui.input_select(
                    #     #                  "scale_nodes", "Scale node size", {'no': 'No scaling', 'increase_100': "Increase 100%", 'decrease_50': "Decrease 50%", 'increase_25': 'Increase 25%',
                    #     #                                                     'decrease_20': 'Decrease 20%'},
                    #     #                  multiple=False, selected='no'
                    #     #              ),
                    #     #              # ui.input_select("color_map", "Color map", {'default': 'Default',
                    #     #              #                                            'tab20': "Matplotlib's tab20",
                    #     #              #                                            'custom': 'User-provided'},
                    #     #              #                 multiple=False),
                    #     #              title="Choose plotting options.",
                    #     #              placement="top"
                    #     #              ),
                    #       class_="d-flex justify-content-between align-items-center",
                    #       style='font-size:x-large;overflow: auto;',
                    #   ),
                    #   ui.HTML(
                    #       """
                    #       <span><b>Important note:</b> Distances between nodes should always be measured along the
                    #        tree, and not by their positions in the 2D-figure (see: 'Layout -> Layout adjustment')</span>
                    #        """
                    #        ),
                      ui.output_plot("make_tree", click=True, dblclick=True,
                                     brush=ui.brush_opts(reset_on_new=True, opacity=0.1), fill=True),
                      full_screen=True, width='100%', height='100%',
                  )
                  ),
        ui.column(4,
                ui.tooltip(ui.span(ui.h5("Bonsai-scout", ui.span(ICONS['license'], style="font-size: 0.6em; vertical-align: super; margin-left: 0.3em;"))),
                    ui.div(
                        ui.HTML(
                            "<small><em>This tool is free for non-commercial use. "
                            "For commercial licensing, please <a href='mailto:daanhugodegroot@gmail.com'>contact us</a>.</em></small>"
                        ),
                    ), placement='right', id='general_info',
                ),
                ui.span(ui.output_ui("dataset_info")),
                ui.input_action_button("open_general_info", "ℹ️ General information", class_="btn-info"),
                ui.p(),
                ui.card(
                    ui.card_header('Legend', class_="d-flex justify-content-between align-items-center"),
                    ui.output_ui('legend_content'),
                    max_height="50%", full_screen=True, fill=False
                ),
                  ui.accordion(
                      ui.accordion_panel("Annotation",
                                         ui.input_selectize('node_style', "Node color:", {},
                                                            selected=None),
                                         ui.input_selectize('size_style', "Node size:", {},
                                                            selected=None),
                                         ui.accordion(
                                             ui.accordion_panel('Color only subset.',
                                                ui.output_ui('selecting_annotation_cat_annot'),
                                                    ui.row(
                                                        ui.column(3, ui.input_action_button("reset_subset_annot", ICONS['house'])),
                                                        ui.column(9, ui.input_switch('switch_mask_annot', "Show only subset!",
                                                                              value=None, width=None)),
                                                    ),
                                                open=False,
                                                ),
                                             open=False, id='annot_subset'
                                        ),
                                         ),
                      ui.accordion_panel("Gene expression",
                                         ui.input_selectize('feature_path_expr', "Feature type:", {},
                                                            selected=None),
                                         ui.output_data_frame("get_genes_df"),
                                         height='400px'
                                         ),
                      ui.accordion_panel("Marker genes",
                        ui.input_action_button("open_marker_info", "ℹ️ Marker instructions", class_="btn-info"),
                        ui.p(height='1em'),
                        ui.input_selectize('feature_path_mrkr', "Selecting feature type:", {},
                                        selected=None),
                        ui.span("Selecting subsets:", style='font-weight:bold;margin-top:2em'),
                        ui.accordion(
                            ui.accordion_panel("Select subset 1",
                        ui.div(
                            ui.output_ui('selecting_annotation_cat'),
                            ui.panel_conditional("input.selected_annot == 'subtree'",
                                ui.tooltip(ui.span("Select subtree:", ICONS['info'],
                                                style='font-weight:bold;margin-top:2em'),
                                    ui.p("First select a node by clicking in the tree, "
                                            "then click the below buttons to indicate if "
                                            "that node is the ancestor or a downstream "
                                            "node of the subtree."),
                                placement='left', id='info_select_subtree'
                                ),
                                # ui.output_text('select_subtree_text'),
                                ui.output_ui('select_subtree_ui_tree1anc'),
                                ui.output_ui('select_subtree_ui_tree1ds'),
                                #   ui.input_action_button("go_select_subtree",
                                #                          "Subtree!",
                                #                          class_="btn-success")
                            ),
                            ui.row(
                                ui.column(3, ui.input_action_button("reset_subset", ICONS['house'])),
                                ui.column(9, ui.input_switch('switch_mask', "Show only subset!",
                                                            value=None, width=None)),
                            ),
                            style='margin-bottom:2em'
                                         ),
                                        ),
                                        ui.accordion_panel("Select subset 2",
                                         ui.div(
                                             ui.output_ui('selecting_annotation_cat_2'),
                                             ui.panel_conditional("input.selected_annot_2 == 'subtree'",
                                                ui.tooltip(ui.span("Select subtree:", ICONS['info'],
                                                                style='font-weight:bold;margin-top:2em'),
                                                        ui.p("First select a node by clicking in the tree, "
                                                             "then click the below buttons to indicate if "
                                                             "that node is the ancestor or a downstream "
                                                             "node of the subtree."),
                                                        placement='left', id='info_select_subtree_2'
                                                        ),
                                                                  # ui.output_text('select_subtree_text'),
                                                                  ui.output_ui('select_subtree_ui_tree2anc'),
                                                                  ui.output_ui('select_subtree_ui_tree2ds'),
                                                                #   ui.input_action_button("go_select_subtree",
                                                                #                          "Subtree!",
                                                                #                          class_="btn-success")
                                                                  ),
                                             ui.row(
                                                 ui.column(3, ui.input_action_button("reset_subset_2", ICONS['house'])),
                                                 ui.column(9, ui.input_switch('switch_mask_2', "Show only subset!",
                                                                              value=None, width=None)),
                                             ),
                                             style='margin-bottom:2em'
                                         ),
                                        ),
                                        ),
                                        ui.p(),
                                        ui.row(
                                            ui.column(6, ui.input_task_button("go_marker", "Get markers!", class_="btn-success")),
                                            ui.column(6, ui.download_button("marker_info_download", "Download info!", class_="btn-primary")),
                                        ),
                                        ui.p(),
                                        ui.output_data_frame("get_marker_genes_df")
                                        ),
                    ui.accordion_panel("Cluster cells",
                        ui.input_action_button("open_cluster_info", "ℹ️ Clustering instructions", class_="btn-info"),
                        ui.p(height='1em'),
                        ui.div(
                            # ui.tooltip(
                            #     ui.span("Get minimal-distance clusters:", ICONS['info'], style='font-weight:bold'),
                            #     ui.div(
                            #         ui.p("To cluster the leafs in groups\n"
                            #             "we iteratively cut the tree into\n"
                            #             "subtrees, such that the sum of pairwise\n"
                            #             "distances between leafs on subtrees\n"
                            #             "is minimized."),
                            #         ), placement='right', id='info_max_diam_clustering',
                            #     ),
                            # ui.input_slider("cluster_diam", "Logarithm of max diameter", min=np.round(np.log(0.01), 2), max=np.log(1), value=0, step=0.01), # TODO: maybe infer this max value from dataset
                            ui.input_slider("n_clusters", "Number of clusters:", min=2, max=100, value=10),
                            # ui.p("Number of clusters found: {}".format(None)),
                            # ui.output_ui('number_of_clusters_found'),
                            ui.input_action_button("go_clustering", "Cluster!",
                                                                    class_="btn-success"),
                            ui.p(),
                            ui.row(
                                ui.column(6, ui.input_action_button("save_clustering", 'Add as annotation', class_='btn-dark')),
                                ui.column(6, ui.download_button("cluster_download", "Download clustering", class_="btn-primary")),
                                ),
                        ),
                    ),
                      open="Annotation",
                      multiple=False,
                      id='options_accordion',
                      max_height='50%',),
                  style='height:100%;overflow:auto'
                  ),
        style='height: 100%'),
    # title=ui.output_text("mytitle"),#"Bonsai visualization",
    fillable=True,)


bv_objcts = BonvisObjects()


# Execute the extended task logic on a different thread. To use a different
# process instead, use concurrent.futures.ProcessPoolExecutor.
pool = concurrent.futures.ProcessPoolExecutor()


def server(input, output, session: Session):
    # --------------------------------------------------------
    # Reactive calculations and effects
    # --------------------------------------------------------
    user_id = id(session)
    # Initialize reactive values
    picked_inds = reactive.value((None, None,))
    renew_legend = reactive.value(0)
    picked_gene = reactive.value(None)
    feature_path = reactive.value(None)
    node_style = reactive.value(None)
    size_style = reactive.value(None)
    open_marker_gene_warning = reactive.value(0)
    trigger_marker_genes = reactive.value(0)
    trigger_new_fig = reactive.value(0)
    trigger_update_figure = reactive.value(0)
    update_figure_kwargs = reactive.value(None)

    first_tree_plot = reactive.value(True)

    # When subset of cells is selected, we update the following. mask_is_on tracks whether only subset is shown in color
    selected_subset_annot = reactive.value({'type': None, 'info': None, 'mask_is_on': False})
    selected_subset = reactive.value({'type': None, 'info': None, 'mask_is_on': False})
    selected_subset_2 = reactive.value({'type': None, 'info': None, 'mask_is_on': False})
    selected_subsets = [selected_subset_annot, selected_subset, selected_subset_2]

    curr_annotation_cats = reactive.value(None)
    curr_categorical_annot = reactive.value(None)

    set_temp_mask = reactive.value(0)  # tracks how often temp_mask should have been set
    redraw_mask = reactive.value(0)  # tracks how often mask should be redrawn (in figure)
    update_mask = reactive.value(0)  # tracks how often mask should be updated (in bonvis_fig-object)

    # selected_annot_val = reactive.value(None)
    # selected_annot_val_2 = reactive.value(None)

    reset_fig = reactive.value(False)

    # Selected subtree-nodes
    selected_tree_nodes_lst = [reactive.value((None, None)), reactive.value((None, None))]

    ### Get some info ###
    # print("session", session)
    # print("http_conn", session.http_conn.url, session.http_conn.url.path, session.http_conn.url.scheme,  session.http_conn['path'])
    # print("http_conn.headers", session.http_conn.headers)
    # print("http_conn.headers.query_params", session.http_conn.query_params, session.http_conn.query_params is None, type(session.http_conn.query_params))
    # print("http_conn.headers.path_params", session.http_conn.path_params)
    # print("http_conn.headers.app", session.http_conn.app)
    # print("http_conn.headers.base_url", session.http_conn.base_url)
    # print("http_conn.headers.client", session.http_conn.client)
    # print("http_conn.headers.state", session.http_conn.state)
    # print("http_conn.headers.query_params.keys", session.http_conn.query_params.keys())
    # print("http_conn.headers.query_params.items", session.http_conn.query_params.items())
    # import pydoc
    # print("server pydoc %s", pydoc.render_doc( session.http_conn.query_params, "%s"))
    #from starlette.requests import HTTPConnection, Request
    #myrequest = Request
    #print("request", myrequest.method)

    """Initializing the UI-elements"""
    ### M: update some ui elements ###
    # @render.ui
    # def mytitle():
    #     bv_objct = bv_objcts[session.input[".clientdata_url_search"].get()]
    #     return ui.HTML("<strong>Bonsai visualization</strong><br><strong>Dataset:</strong> {:s}<br><strong>Number of cells:</strong> {:d}.<br>".format(bv_objct.bonvis_metadata.dataset,
    #                                                                              bv_objct.bonvis_metadata.n_cells))

    def show_waiting_modal(message="Processing your clicks...", trigger=None):
        modal = ui.modal(
            ui.h3(ui.HTML("<em>Bonsai-scout</em> is rendering...")),
            ui.HTML("<br>{}<br>".format(message)),
            size="m",
            easy_close=False,
            fade=False,
            footer=None,
        )
        ui.modal_show(modal)
        if trigger is not None:
            trigger.set(trigger.get() + 1)

    @reactive.effect
    @reactive.event(first_tree_plot)
    def _():
        # If this is the first tree plot, put up a placeholder while we calculate the first figure
        if first_tree_plot.get():
            logging.debug("RETURNING TO PLACEHODLER!")
            trigger_new_fig.set(trigger_new_fig.get() + 1)
            show_waiting_modal(message="<em>Bonsai-scout</em> is loading your first tree-representation.<br> "
                                       "For large datasets, this can take a while, please be patient.")
        else:
            ui.modal_remove()
            logging.debug("MOVING ON!")

    @render.ui
    def dataset_info():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        # Add some <br> or use a <div> with margin to add extra spacing
        elem = ui.div(
            ui.div(ui.HTML(
                "<strong>Dataset:</strong> {:s}<br>"
                "<strong>Number of cells:</strong> {:d}.<br>"
                "<div style='height: 1em;'></div>"  # Adds a space of 20px
                .format(bv_objct.bonvis_metadata.dataset, bv_objct.bonvis_metadata.n_cells)
            ))
        )
        return elem

    @reactive.effect
    @reactive.event(input.open_general_info)
    def _():
        modal = ui.modal(
            ui.h3(ui.HTML("About <em>Bonsai-scout</em>")),
            ui.h5("Usage of the app"),
            ui.HTML("<strong>Tree interpretation:</strong> Distances between nodes should always be measured along the tree, not by their positions in the 2D-figure!<br>"),
            ui.p(),
            ui.h5("Terms of use:"),
            ui.HTML(
                "<strong><em>Licensing:</strong> </em>Bonsai-scout<em> is free for non-commercial use. "
                "For commercial licensing, please <a href='mailto:daanhugodegroot@gmail.com,daan.degroot@unibas.ch,erik.vannimwegen@unibas.ch'>contact us</a>.</em><br>"
            ),
            ui.HTML(
                '<strong>Acknowledging <em>Bonsai</em>:</strong> When using <em>Bonsai(-scout)</em> for scientific discovery, ' \
                'please cite our <a href="https://www.biorxiv.org" target="_blank">publication</a>.'
            ),
            size="xl",
            easy_close=True,
            fade=True,
            footer=None,
        )
        ui.modal_show(modal)


    @reactive.effect
    @reactive.event(input.open_layout_info)
    def _():
        modal = ui.modal(
            ui.h3(ui.HTML("Changing the <em>Bonsai</em> layout:")),
            ui.h5(ui.HTML("Picking the layout:")),
            ui.HTML("Any tree is defined completely by its topology and its branch lengths, but these can be visualized in various layouts. " \
                "We offer the <em>dendrogram</em> layout, the <em>equal-angle</em> layout, and for smaller datasets the <em>equal-daylight</em> layout.<br>" \
                "Regardless of the layout, object-to-object distances should be measured along the branches of the tree, " \
                "not by the distance between the objects in the 2D-space."),
            ui.p(),
            ui.h5(ui.HTML("Picking the disk geometry (only for circular layouts)")),
            ui.HTML("For circular layouts, one can choose between hyperbolic and flat geometry.<br>"),
            ui.HTML("<strong>Hyperbolic geometry:</strong><br> " \
                "The hyperbolic layout maps the full 2D-plane to the unit disk. " \
                "As a result, distances between points near the boundary of the disk get more and more compressed, " \
                "which is illustrated by the squares in the background, which are all the same size.<br>"),
            ui.HTML("<strong>Flat geometry:</strong><br>" \
                "The flat geometry does not transform distances, but plots all points and lines " \
                "that are outside of the field of view on the enveloping square.<br>" \
                "<em>Distances to these points outside the square are thus not the true distances.</em>"),
            ui.p(),
            ui.h5(ui.HTML("Tweaking the layout:")),
            ui.HTML("The layouts can be tweaked to better explore the data, without changing the underlying tree.<br>"),
            ui.HTML("<strong>Flipping branch order:</strong> With this function, one can pick a node and change the order in which the connecting branches are laid out.<br>"),
            ui.HTML("<strong>Changing the root (only for dendrogram):</strong> The Bonsai likelihood is independent of which point in the tree has been designated as 'the root'. " \
                "However, the root-position does determine which point is plotted on the left-most side in the dendrogram layout. We allow the user to change this root.<br>"),
            ui.HTML("<strong>Improving specific nodes (only for circular layouts):</strong> In the equal-angle and -daylight layouts, it is possible" \
                "to increase the daylight or angle around specific nodes to improve the visual results."),
            size="xl",
            easy_close=True,
            fade=True,
            footer=None,
        )
        ui.modal_show(modal)


    @reactive.effect
    @reactive.event(input.open_marker_info)
    def _():
        modal = ui.modal(
            ui.h3(ui.HTML("Finding marker genes:")),
            ui.HTML("<em>Bonsai-scout</em> allows for finding marker genes/features that " \
                "distinguish two groups of cells/objects. " \
                "For this, we search for features that maximize the probability that, " \
                "when picking random cells from the two groups, the feature is always higher "
                "or always lower in the cell from Group 1 than in the cell from Group 2.<br>"),
            ui.HTML("To find marker features, one first selects which of the feature-types should be used. " \
                "Then, one needs to select two subsets: by 1) picking a subtree, or 2) picking an annotation-category.<br>" \
                "If only one subset is picked, the second subset will be all remaining cells by default. " \
                "Each subset can be reset by clicking the 'Home'-button.<br>"),
            ui.p(),
        ui.h5("Selecting a feature type:"),
        ui.p("Use the dropdown menu under 'Selecting feature type:' to select in which features we will look for markers."),
            ui.p(),
            ui.h5("Picking an annotation-category:"),
            ui.HTML("Note that only categories can be picked that are part of the annotation that is " \
                    "currently activated in the 'Annotation'-tab.<br>"
                    "<em>Note:</em> You can find marker genes on tree-based clusters "
                    "once you have added a clustering as Annotation in the 'Cluster'-tab."),
            ui.p(),
            ui.h5("Selecting a subtree:"),
            ui.HTML("One picks a subtree by 4 clicks: <br>"
                    "1. Click the subtree-ancestor in the <em>Bonsai</em>,<br> "
                    "2. Click the black 'Ancestor'-button in the right sidebar. <br>"
                    "3. Click any node in the subtree (i.e., downstream of the selected ancestor. <br>"
                    "4. Click the black 'Downstream'-button in the right sidebar. <br>"),
            ui.p(),
            ui.h5("Testing the selections, and getting the markers:"),
            ui.HTML("The selection can be tested by using the 'Show only subset'-switch.<br>"),
            ui.HTML("<strong>After selecting the desired subsets, click 'Get markers!' "
                    "to get the marker features with their scores. Scores are between 0 and 1, "
                    "where scores close to 0 or close to 1 indicate strong markers.</strong>"),
            size="xl",
            easy_close=True,
            fade=True,
            footer=None,
        )
        ui.modal_show(modal)


    @reactive.effect
    @reactive.event(input.open_viewing_info)
    def _():
        modal = ui.modal(
            ui.h3(ui.HTML("Changing the <em>Bonsai</em> view:")),
            ui.h5(ui.HTML("Drag and zoom:")),
            ui.HTML("To zoom in on part of the <em>Bonsai</em>, draw a rectangle in the figure and click the 'Zoom!'-button. "
                "The selected rectangle needs to be removed by clicking next to it.<br>"
                "By using the plus- and minus-buttons, one can zoom more or less on the selected region.<br>"
                "Click the home-button to view the whole tree again."),
            ui.p(),
            ui.h5(ui.HTML("Change the origin (only for circular layouts):")),
            ui.HTML("Double-click on a point in the <em>Bonsai</em> to move that point to the center of the figure."),
            ui.p(),
            ui.h5(ui.HTML("Curvature (only for circular layouts):")),
            ui.HTML("By clicking the plus- or minus-buttons, the curvature of the hyperbolic disk can be changed. "
                "More curvature will allocate more space to the center of the figure, compressing distances towards the edges more.<br>"
                "Click the home-button to reset curvature."),
            ui.p(),
            ui.h5(ui.HTML("Tweak visuals:")),
            ui.HTML("This expandable tab allows you to change the sizes of the nodes, the width of the node-edge, "
            "and the thickness of the tree-branches."),
            size="xl",
            easy_close=True,
            fade=True,
            footer=None,
        )
        ui.modal_show(modal)


    @reactive.effect
    @reactive.event(input.open_cluster_info)
    def _():
        modal = ui.modal(
            ui.h3(ui.HTML("Clustering using <em>Bonsai-scout</em>:")),
            ui.p(),
            ui.h5(ui.HTML("Minimal distance clustering:")),
            ui.HTML("To cluster the leafs in groups, we iteratively cut branches to create subtrees. At each step, we cut the branch " \
                "such that the sum of pairwise distances between leafs on the created subtrees is minimized. <br>" \
                "Use the slider to set the number of clusters, and click the 'Cluster'-button to create the clusters."),
            ui.p(),
            ui.h5(ui.HTML("Adding the clustering as a new annotation")),
            ui.HTML("With the 'Add as annotation'-button, the current clustering " \
                "can be added as annotation. By subsequently selecting the clustering in the 'Annotation'-tab, the 'Marker genes'-tab"
                "can now be used to detect marker features for the different clusters."),
            ui.p(),
            ui.h5(ui.HTML("Downloading the clustering:")),
            ui.HTML("By clicking the download button, one can download a '.tsv'-file with a mapping from object-ID to cluster-ID."),
            size="xl",
            easy_close=True,
            fade=True,
            footer=None,
        )
        ui.modal_show(modal)

    # Set all things to their initial values
    @reactive.effect
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        ui.update_selectize(
            "geometry",
            selected=bv_objct.init_geometry,
        )
        ui.update_selectize(
            "ly_type",
            choices=bv_objct.layout_types_dict,
            selected=bv_objct.init_layout,
        )
        ui.update_selectize(
            "node_style",
            choices=bv_objct.annotation_dict,
            selected=bv_objct.init_node_style,
        )
        ui.update_slider("n_clusters", label="Number of clusters:", min=2, max=bv_objct.max_n_clusters, value=10)
        ui.update_selectize(
            "size_style",
            choices=bv_objct.size_annotation_dict,
            selected=bv_objct.init_size_style,
        )
        ui.update_selectize(
            "feature_path_expr",
            choices=bv_objct.feature_dict,
            selected=bv_objct.init_feature_path,
        )
        ui.update_selectize(
            "feature_path_mrkr",
            choices=bv_objct.feature_dict,
            selected=bv_objct.init_feature_path,
        )
        ui.update_switch(
            "switch_mask",
            value=bv_objct.init_switch_mask,
        )
        ui.update_accordion("options_accordion", show=bv_objct.init_options_accordion)
    ### ###

    """Storing or resetting the settings"""
    # TODO: Restore the storing and resetting function
    # @reactive.effect
    # @reactive.event(input.store)
    # def _():
    #     bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
    #     store_current_settings(bv_objct.bonvis_fig.bonvis_settings, bv_objct.shiny_paths["settings_path"])

    # @reactive.effect
    # @reactive.event(input.reset)
    # def _():
    #     bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
    #     global old_reset_cnt
    #     if input.reset() != old_reset_cnt:
    #         old_reset_cnt = input.reset()
    #         reset_fig.set(True)
    #         bv_objct.bonvis_fig = bv_objct.bonvis_fig.reset_figure(bv_objct.shiny_paths['settings_path'])
    #         ui.update_selectize('ly_type', selected=bv_objct.init_layout)
    #         ui.update_selectize('geometry', selected=bv_objct.init_geometry)
    #         ui.update_selectize("node_style", selected=bv_objct.init_node_style)
    #         ui.update_selectize("size_style", selected=bv_objct.init_size_style)
    #         ui.update_selectize('selected_annot', selected=bv_objct.init_selected_annot)
    #         ui.update_selectize('selected_annot_annot', selected=bv_objct.init_selected_annot)
    #         ui.update_switch('switch_mask', value=bv_objct.init_switch_mask)
    #         ui.update_accordion('options_accordion', show=bv_objct.init_options_accordion)
    #         curr_annotation_cats.set(bv_objct.init_annotation_cats)
    #         curr_categorical_annot.set(bv_objct.init_categorical_annot)


    """Change node style"""
    @reactive.effect
    @reactive.event(input.node_style)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        celltype_info = bv_objct.bonvis_fig.bonvis_settings.celltype_info
        annot_info = [annot_info for annot, annot_info in celltype_info.annot_infos.items() if
                      annot_info.label == input.node_style()]
        if len(annot_info) == 1:
            annot_info = annot_info[0]
            if annot_info.color_type == 'categorical':
                curr_categorical_annot.set(annot_info.label)
                curr_annotation_cats.set(annot_info.cats)


    """Change node style"""
    @reactive.effect
    @reactive.event(input.go_clustering)
    def _():
        cluster_node_style = "Cluster_n{}".format(input.n_clusters())
        node_style.set(cluster_node_style)

    # @reactive.effect
    # @reactive.event(input.go_clustering)
    # def _():
    #     bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
    #     n_clusters.set(bv_objct.bonvis_fig.number_of_clusters_found)
        # n_clusters.set(10) # this works#

    # @render.text
    # def number_of_clusters_found():
    #     return "Number of clusters found: {}".format(n_clusters.get())
        # global num_clusters
        # n_clusters.set(num_clusters)
        # return ui.p("hi: {}".format(n_clusters))

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    @reactive.effect
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if input.feature_path_expr() != bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path']:
            bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path'] = input.feature_path_expr()
            ui.update_selectize('feature_path_mrkr',
                                selected=bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path'])
        if input.feature_path_mrkr() != bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path']:
            bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path'] = input.feature_path_mrkr()
            ui.update_selectize('feature_path_expr', selected=bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path'])
        feature_path.set(bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path'])

    @render.data_frame
    def get_genes_df():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if feature_path.get() is None:
            feature_path.set(bv_objct.init_feature_path)
        feature_path.get()
        bv_objct.feature_display = get_feature_info_display(bv_objct.bonvis_fig.bonvis_data,
                                                   bv_objct.bonvis_fig.bonvis_settings.node_style['feature_path'])
        return render.DataGrid(bv_objct.feature_display, selection_mode='row', summary=False, filters=False, width='100%')

    # @reactive.effect
    # @reactive.event(input.go_marker)
    # def _():
    #     bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
    #     if input.go_marker() != bv_objct.click_counters['marker']:
    #         open_marker_gene_warning.set(open_marker_gene_warning.get() + 1)
    #         bv_objct.click_counters['marker'] = input.go_marker()

    @reactive.effect
    @reactive.event(input.go_marker, ignore_init=True)
    def preprocessing_marker_genes():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if input.go_marker() != bv_objct.click_counters['marker']:
            bv_objct.click_counters['marker'] = input.go_marker()
        # if open_marker_gene_warning.get() != bv_objct.click_counters['open_marker_gene_warning']:
            if bv_objct.is_big_dataset:
                modal = ui.modal(
                    ui.h3(ui.HTML("<em>Bonsai's</em> marker gene calculation.")),
                    # ui.HTML("<em>Warning:</em> The full marker score calculation which takes into account " \
                    # "uncertainties on the features would take about {:.0f} seconds.".format(time_estimate)),
                    ui.HTML("<em>Warning:</em> We are currently preprocessing the marker gene calculation. This can "
                            "already take some seconds for big datasets.<br>"
                            "Please be patient, <em>clicking many buttons won't speed it up</em>."),
                    size="xl",
                    easy_close=True,
                    fade=False,
                    footer=None,
                )
                ui.modal_show(modal)
            if curr_categorical_annot.get() is None:
                curr_categorical_annot.set(bv_objct.init_categorical_annot)
            with reactive.isolate():
                curr_subset = selected_subset.get()
                curr_subset_2 = selected_subset_2.get()
                bonvis_data_path = bv_objct.shiny_paths['data_path']
                run_with_vars, marker_genes_tuple, \
                    time_estimate, marker_info_dict = bv_objct.bonvis_fig.preprocess_marker_genes(
                                                            bonvis_data_path,
                                                            curr_subset,
                                                            curr_subset_2=curr_subset_2,
                                                            curr_categorical_annot=curr_categorical_annot.get())
                bv_objct.marker_gene_vars['run_with_vars'] = run_with_vars
                bv_objct.marker_gene_vars['marker_genes_tuple'] = marker_genes_tuple
                bv_objct.marker_gene_vars['marker_info_dict'] = marker_info_dict
                bv_objct.marker_gene_vars['time_estimate'] = time_estimate
                ui.modal_remove()
            if run_with_vars == 'yes':
                bv_objct.click_counters['open_marker_gene_warning'] = open_marker_gene_warning.get()
                modal = ui.modal(
                    ui.h3(ui.HTML("<em>Bonsai's</em> marker gene calculation.")),
                    ui.HTML("This calculation will take about {:.0f} seconds, " \
                    "and will run in the background. App may be a bit slower in the meantime.".format(time_estimate)),
                    size="xl",
                    easy_close=True,
                    fade=False,
                    footer=None,
                )
                ui.modal_show(modal)
            elif run_with_vars == 'shortcut':
                bv_objct.marker_gene_vars['start_time_modal'] = time.time()
                if bv_objct.is_big_dataset:
                    big_dataset_text = "For big datasets, this can still take a while, " \
                    "and it will run in the background. App may be a bit slower in the meantime."
                else:
                    big_dataset_text = ""
                modal = ui.modal(
                    ui.h3(ui.HTML("<em>Bonsai's</em> marker gene calculation.")),
                    # ui.HTML("<em>Warning:</em> The full marker score calculation which takes into account " \
                    # "uncertainties on the features would take about {:.0f} seconds.".format(time_estimate)),
                    ui.HTML("<em>Warning:</em> The full marker score calculation which takes into account "
                            "uncertainties on the features would take about {:.0f} seconds. <br>".format(
                        bv_objct.marker_gene_vars['time_estimate'])),
                    ui.HTML("We will therefore run a faster marker score calculation that ignores the uncertainty. "),
                    ui.HTML(big_dataset_text),
                    ui.p(),
                    ui.h5("Alternative: Running the marker gene detection script offline:"),
                    ui.HTML("You can download the grouping that you picked by clicking the 'Download info'-information, " \
                            "and then using it with the 'calc_marker_genes.py'-script present in the 'downstream_analyses' " \
                            "sub-folder of the <em>Bonsai</em>-GitHub."),
                    size="xl",
                    easy_close=True,
                    fade=False,
                    footer=None,
                )
                ui.modal_show(modal)
            trigger_marker_genes.set(trigger_marker_genes.get() + 1)

    @reactive.effect
    @reactive.event(trigger_marker_genes)
    def launch_core_marker_genes():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if trigger_marker_genes.get() != bv_objct.click_counters['trigger_marker']:
            run_with_vars = bv_objct.marker_gene_vars['run_with_vars']
            marker_genes_tuple = bv_objct.marker_gene_vars['marker_genes_tuple']
            core_get_marker_genes(run_with_vars, marker_genes_tuple)

    @ui.bind_task_button(button_id="go_marker")
    @reactive.extended_task
    async def core_get_marker_genes(run_with_vars, marker_genes_tuple):
        # This is the main calculation that takes very long
        loop = asyncio.get_event_loop()
        logging.info("Starting the update_marker_genes_df-function in a new process")
        return await loop.run_in_executor(pool, update_marker_genes_df, run_with_vars, marker_genes_tuple)

    @render.data_frame
    def get_marker_genes_df():
        marker_genes_df = core_get_marker_genes.result()
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        bv_objct.bonvis_fig.marker_genes_df = marker_genes_df
        bv_objct.click_counters['trigger_marker'] = trigger_marker_genes.get()
        return render.DataGrid(bv_objct.bonvis_fig.marker_genes_df.round(2),
                            selection_mode="row", summary=False, filters=False,
                            width='100%')

    @render.plot(alt="Bonsai plot")
    @reactive.event(trigger_new_fig)
    def make_tree():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        trigger_new_fig.get()
        # logging.debug("trigger_new_fig.get(): {}".format(trigger_new_fig.get()))
        if first_tree_plot.get():
            first_tree_plot.set(False)
        return bv_objct.bonvis_fig.fig

    @reactive.effect
    def preprocess_make_tree():
        req(input.ly_type(), input.feature_path_mrkr(), input.feature_path_expr(), feature_path)
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        logging.debug("user_id: %r url_search: %r", user_id, session.input[".clientdata_url_search"].get())
        logging.debug("bv_objcts.keys(): %r", bv_objcts.keys())
        # Determine whether settings should be reset
        ax_lims = None

        reset_now = None
        if reset_fig.get():
            # global bonvis_fig
            reset_now = True
            reset_fig.set(False)

        if redraw_mask.get() != bv_objct.mask_counters['redraw_mask']:
            renew_mask_fig = True
            bv_objct.mask_counters['redraw_mask'] = redraw_mask.get()
        else:
            renew_mask_fig = None

        # Determine if layout should be changed
        geometry = None
        if input.ly_type() != bv_objct.bonvis_fig.bonvis_settings.ly_type:
            ly_type = input.ly_type()
            if ly_type.startswith('ly_dendrogram'):
                geometry = 'flat'
        else:
            ly_type = None

        # Determine if geometry should be changed
        if bv_objct.bonvis_fig.bonvis_settings.transf_info.geometry != input.geometry():
            pattern = re.compile(r'dendrogram')
            new_ly_type = ly_type if (ly_type is not None) else bv_objct.bonvis_fig.bonvis_settings.ly_type
            if pattern.search(new_ly_type):
                if input.geometry != 'flat':
                    logging.warning("It doesn't make sense to use hyperbolic geometry with dendrogram layout.\n"
                                    "Sticking to flat geometry")
                    ui.update_select('geometry', selected='flat')
                    geometry = None
                else:
                    geometry = input.geometry()
            else:
                # This means the preferred plot geometry has changed. This needs to be udpated
                geometry = input.geometry()
        else:
            geometry = None

        # Determine if layout should be tweaked
        # Take a reactive dependency on the zoom-action button...
        # ...but don't take a reactive dependency on the zoom-slider
        tweak_inds = None
        multip_angle = None
        if input.go_tweak() != bv_objct.click_counters['tweak']:
            if input.ly_type() == 'ly_eq_daylight':
                with reactive.isolate():
                    if picked_inds.get()[-1] is not None:
                        tweak_inds = picked_inds.get()[-1]['int_inds']
            elif input.ly_type() == 'ly_eq_angle':
                with reactive.isolate():
                    if picked_inds.get()[-2] is not None:
                        tweak_inds = (picked_inds.get()[-2]['int_inds'], picked_inds.get()[-1]['all'],)
                        new_multip_angle = 2 ** input.more_angle()
                        if new_multip_angle == 1:
                            picked_inds.set((None, None,))
                        else:
                            multip_angle = new_multip_angle
                            ui.update_slider("more_angle", value=0)
            elif input.ly_type().startswith('ly_dendrogram'):
                with reactive.isolate():
                    if picked_inds.get()[-1] is not None:
                        tweak_inds = picked_inds.get()[-1]['all']
            bv_objct.click_counters['tweak'] = input.go_tweak()

        # Determine if some branch needs to be flipped
        # tweak_inds = None
        # multip_angle = None
        new_flip_id = None
        if input.go_flip() != bv_objct.click_counters['flip']:
            with reactive.isolate():
                if picked_inds.get()[-1] is not None:
                    bv_objct.flip_ids.append(bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-1]['all']])
                    new_flip_id = True
            bv_objct.click_counters['flip'] = input.go_flip()

        if input.reset_flip() != bv_objct.click_counters['reset_flip']:
            with reactive.isolate():
                bv_objct.flip_ids = []
                new_flip_id = True
            bv_objct.click_counters['reset_flip'] = input.reset_flip()

        # Determine if zoom should be changed
        zoom = None
        if input.more_curve() != bv_objct.click_counters['more_curve']:
            zoom = 2
            bv_objct.click_counters['more_curve'] = input.more_curve()
        if input.less_curve() != bv_objct.click_counters['less_curve']:
            zoom = .5
            bv_objct.click_counters['less_curve'] = input.less_curve()

        if input.go_crop() != bv_objct.click_counters['crop']:
            try:
                with reactive.isolate():
                    if input.make_tree_brush() is not None:
                        test_lims = np.array([input.make_tree_brush()[key] for key in ['xmin', 'xmax', 'ymin', 'ymax']])
                        if not np.all(test_lims == bv_objct.old_brush):
                            ax_lims = test_lims
                            bv_objct.old_brush = test_lims
                        # ui.notification_show(
                        #     "Warning: Wedges that indicate multiple cells at a node are not supported on this cropped
                        #     figure. These "
                        #     "nodes will be replaced by multiple nodes with a slight offset, and 50% transparency.",
                        #     action=None, duration=8, close_button=True, id=None, type='warning', session=None)
            except SilentException:
                ax_lims = None
            bv_objct.click_counters['crop'] = input.go_crop()

        zoom_ax_lims = None
        if input.zoom_in() != bv_objct.click_counters['zoom_in']:
            zoom_ax_lims = 1/1.5
            bv_objct.click_counters['zoom_in'] = input.zoom_in()
        if input.zoom_out() != bv_objct.click_counters['zoom_out']:
            zoom_ax_lims = 1.5
            bv_objct.click_counters['zoom_out'] = input.zoom_out()

        # Determine if navigation should be reset
        if input.reset_geometry() != bv_objct.click_counters['reset_navi']:
            bv_objct.click_counters['reset_navi'] = input.reset_geometry()
            if input.ly_type() in ['ly_eq_angle', 'ly_eq_daylight']:
                curr_zoom = bv_objct.bonvis_fig.bonvis_settings.transf_info.get_zoom()
                if (bv_objct.orig_zoom[input.geometry()] is None) or (bv_objct.orig_origin is None):
                    transf_info = bv_objct.bonvis_fig.bonvis_settings.transf_info
                    edge_df_dict = bv_objct.bonvis_metadata.tree_info['edge_dict']
                    # Fit optimal zoom and origin, because it was not known yet
                    transf_info.set_zoom(None)
                    transf_info.origin = None
                    transf_info.get_zoom_origin(bv_objct.bonvis_fig.coords_info.node_coords_nx, edge_df_dict)
                    bv_objct.orig_zoom[input.geometry()] = transf_info.get_zoom()
                    bv_objct.orig_origin = transf_info.origin
                    zoom = 1
                else:
                    zoom = bv_objct.orig_zoom[input.geometry()] / curr_zoom
                    bv_objct.bonvis_fig.bonvis_settings.transf_info.origin = bv_objct.orig_origin
        if input.reset_crop() != bv_objct.click_counters['reset_crop']:
            ax_lims = np.array([-1, 1, -1, 1], dtype=float)
            bv_objct.click_counters['reset_crop'] = input.reset_crop()

        # Determine if nodes should be resized
        scale_nodes = None
        if input.nodes_bigger_fast() != bv_objct.click_counters['nodes_bigger']:
            scale_nodes = 1.5
            bv_objct.click_counters['nodes_bigger'] = input.nodes_bigger_fast()
        if input.nodes_smaller_fast() != bv_objct.click_counters['nodes_smaller']:
            scale_nodes = 1/1.5
            bv_objct.click_counters['nodes_smaller'] = input.nodes_smaller_fast()
        
        # Determine if edges should be resized
        scale_edges = None
        if input.edges_thicker() != bv_objct.click_counters['edges_thicker']:
            scale_edges = 1.5
            bv_objct.click_counters['edges_thicker'] = input.edges_thicker()
        if input.edges_thinner() != bv_objct.click_counters['edges_thinner']:
            scale_edges = 1/1.5
            bv_objct.click_counters['edges_thinner'] = input.edges_thinner()

        # Determine if edges should be resized
        scale_node_edges = None
        if input.node_edges_thicker() != bv_objct.click_counters['node_edges_thicker']:
            scale_node_edges = 1.5
            bv_objct.click_counters['node_edges_thicker'] = input.node_edges_thicker()
        if input.node_edges_thinner() != bv_objct.click_counters['node_edges_thinner']:
            scale_node_edges = 1/1.5
            bv_objct.click_counters['node_edges_thinner'] = input.node_edges_thinner()

        # Determine if nodes should be recolored
        if node_style.get() is None:
            node_style.set(bv_objct.init_node_style)
        if node_style.get() != bv_objct.old_node_style:
            bv_objct.old_node_style = node_style.get()
            ui.update_switch('switch_mask_annot', value=False)
            node_style_upd = node_style.get()
        else:
            node_style_upd = None

        # Determine if sizes should be redone
        if size_style.get() is None:
            size_style.set(bv_objct.init_node_style)
        if size_style.get() != bv_objct.old_size_style:
            bv_objct.old_size_style = size_style.get()
            size_style_upd = size_style.get()
        else:
            size_style_upd = None

        # Determine if origin should be repositioned
        origin = None
        try:
            if input.make_tree_dblclick() is not None:
                new_orig = np.array([input.make_tree_dblclick()[key] for key in ['x', 'y']])
                if (new_orig != bv_objct.old_orig).any() and (np.max(np.abs(new_orig)) < 1.1):
                    origin = new_orig
                    bv_objct.old_orig = new_orig
        except SilentException:
            # Double click was not used yet, so input undefined
            origin = None

        # # Determine if clustering should be performed
        # input.go_clustering()
        # # cluster_diam = None
        # n_clusters = None
        # # Take a reactive dependency on the clustering-action button...
        #     # ...but don't take a reactive dependency on the n_clusters-slider
        # with reactive.isolate():
        #     # new_cluster_diam = np.exp(input.cluster_diam())
        #     if input.go_clustering() != bv_objct.click_counters['cluster']:
        #         bv_objct.click_counters['cluster'] = input.go_clustering()
        #         n_clusters = input.n_clusters()
        #         # ui.update_slider("curvature", value=0)

        kwarg_list = [geometry, zoom, scale_nodes, scale_edges, origin, node_style_upd,
                      size_style_upd, ly_type, tweak_inds, new_flip_id, scale_node_edges,
                      multip_angle, ax_lims, zoom_ax_lims, reset_now, renew_mask_fig]
        if any(v is not None for v in kwarg_list) or (bv_objct.bonvis_fig.fig is None):
            update_figure_kwargs.set(dict(
                geometry=geometry,
                zoom=zoom,
                scale_nodes=scale_nodes,
                scale_edges=scale_edges,
                scale_node_edges=scale_node_edges,
                origin=origin,
                node_style=node_style_upd,
                size_style=size_style_upd,
                ly_type=ly_type,
                renew_mask=renew_mask_fig,
                tweak_inds=tweak_inds,
                flipped_node_ids=bv_objct.flip_ids,
                multip_angle=multip_angle,
                ax_lims=ax_lims,
                zoom_ax_lims=zoom_ax_lims,
                new_flip_id=new_flip_id
            ))
            if bv_objct.is_big_dataset:
                show_waiting_modal("Re-visualizing the Bonsai...", trigger=trigger_update_figure)
            else:
                trigger_update_figure.set(trigger_update_figure.get() + 1)

    @reactive.effect
    @reactive.event(trigger_update_figure)
    def update_figure():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        with reactive.isolate():
            if update_figure_kwargs.get() is None:
                return
            stored_kwargs = update_figure_kwargs.get()
            create_legend = bv_objct.bonvis_fig.update_figure(**stored_kwargs)
            update_figure_kwargs.set(None)
        bv_objct.bonvis_fig.create_figure()
        # logging.debug("Setting trigger_new_fig higher. Currently {}".format(trigger_new_fig.get()))
        trigger_new_fig.set(trigger_new_fig.get() + 1)
        if create_legend:
            renew_legend.set(renew_legend.get() + 1)
        ui.modal_remove()
        # logging.debug("Now {}".format(trigger_new_fig.get()))

    # @reactive.effect
    # @reactive.event(input.fig_format)
    # def _():
    #     bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
    #     bv_objct.filenames['fig'] = 'bonsai_tree_{}.{}'.format(bv_objct.filenames['fig_counter'], input.fig_format())

    @render.download(filename=lambda: make_user_specific_figname())
    async def tree_download():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        bv_objct.click_counters['fig'] += 1
        with io.BytesIO() as buf:
            with reactive.isolate():
                file_extension = input.fig_format()
            bv_objct.bonvis_fig.fig.savefig(buf, format=file_extension, dpi=300)
            yield buf.getvalue()

    def make_user_specific_figname():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        return 'bonsai_tree_{}.{}'.format(bv_objct.click_counters['fig'], input.fig_format())

    # @reactive.effect
    # @reactive.event(input.go_clustering)
    # def _():
    #     # download_cluster_diam = np.exp(input.cluster_diam())
    #     download_n_clsts = input.n_clusters()
    #     cluster_filename = "clustering_{}.tsv".format(download_n_clsts)
    #     logging.debug("go clustering, change filename: {}".format(cluster_filename))

    def make_cluster_filename():
        download_n_clsts = input.n_clusters()
        cluster_filename = "clustering_{}.tsv".format(download_n_clsts)
        return cluster_filename

    @render.download(filename=lambda:make_cluster_filename())
    async def cluster_download():
        # print("Download clustering for max diam {} results in: {}".format(np.exp(input.cluster_diam()), cluster_filename))
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]

        # Add clustering annotation to downloadable node styles
        add_clustering_to_nodestyles(bv_objct)

        cluster_labels = bv_objct.bonvis_metadata.cs_info['cluster_info_dict'].keys()
        info_key = "annot_cluster_n{}".format(input.n_clusters())
        if info_key not in cluster_labels:
            logging.error("Could not find the clustering you want to download. Did you perform the clustering already?")
            return
        # info_key = possible_annot_keys[-1]

        print("downloading: {}".format(info_key))
        annot_list = bv_objct.bonvis_metadata.cs_info['cluster_info_dict'][info_key]
        dfout = pd.DataFrame({"node-ID": bv_objct.bonvis_metadata.cs_ids, "cluster": annot_list})
        yield dfout.to_csv(index=False, sep='\t')

        # test
        # this works
        # df = pd.DataFrame({"test":[0,1,2,3,4]})
        # print(df.shape)
        # print(df)
        # yield df.to_string(index=False)


    @render.download(filename=lambda:make_user_specific_marker_name())
    async def marker_info_download():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        bv_objct.click_counters['marker_info'] += 1
        if bv_objct.marker_gene_vars['marker_info_dict'] is not None:
            logging.info("Download information for currently selected marker groups, " \
                "results in: 'marker_groups_info.json'")
            buffer = io.StringIO()
            json.dump(bv_objct.marker_gene_vars['marker_info_dict'], buffer, indent=4)
            buffer.seek(0)

            yield buffer.read()
        else:
            logging.warning("No marker information yet available. Click 'get_markers' to create this.")

    def make_user_specific_marker_name():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        return 'marker_groups_info_{}.json'.format(bv_objct.click_counters['marker_info'])


    """Saving and updating the clustering to the annotation drop down"""
    @reactive.effect
    @reactive.event(input.save_clustering)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        cluster_node_style = "Cluster_n{}".format(input.n_clusters())

        # Set nodestyle to cluster-node style
        node_style.set(cluster_node_style)

        # Show clustering-annotation in selectable node-styles
        add_clustering_to_nodestyles(bv_objct)


    def add_clustering_to_nodestyles(bv_objct):
        # Get annot-info object 
        celltype_info = bv_objct.bonvis_fig.bonvis_settings.celltype_info
        cluster_node_style = "Cluster_n{}".format(input.n_clusters())
        annot_info = [annot_info for annot, annot_info in celltype_info.annot_infos.items() if
                      annot_info.label == cluster_node_style]
        if len(annot_info) == 1:
            # If found, stop hiding the annotation from view
            annot_info = annot_info[0]
            annot_info.hidden = False
            if annot_info.color_type == 'categorical':
                # Update some reactive values that we need elsewhere
                curr_categorical_annot.set(annot_info.label)
                curr_annotation_cats.set(annot_info.cats)

        # Update annotation dict for showing in the node-style
        bv_objct.annotation_dict = {}
        for annot, annot_info in bv_objct.annot_infos.items():
            if hasattr(annot_info, 'hidden') and annot_info.hidden:
                continue
            bv_objct.annotation_dict[annot_info.label] = annot_info.label
        ui.update_selectize("node_style", choices=bv_objct.annotation_dict,
                            selected=bv_objct.bonvis_fig.bonvis_settings.node_style['annot_info'].label)



    """Changing the legend"""

    @render.ui
    @reactive.event(renew_legend)
    def legend_content():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        logging.debug("Main legend number {}".format(renew_legend.get()))
        bv_objct.bonvis_fig.create_legend()
        if bv_objct.bonvis_fig.bonvis_settings.node_style['annot_info'].color_type == 'categorical':
            return ui.output_table("get_legend_df")
        elif bv_objct.bonvis_fig.bonvis_settings.node_style['annot_info'].color_type == 'sequential':
            return ui.output_plot('get_cbar')

    @render.table
    def get_legend_df():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        # print("Legend number {}".format(renew_legend.get()))
        if renew_legend.get() > -1:
            leg_df = bv_objct.bonvis_fig.fig_leg_df
            return leg_df

    @render.plot
    def get_cbar():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        # print("Color bar number {}".format(renew_legend.get()))
        if renew_legend.get() > -1:
            return bv_objct.bonvis_fig.fig_cbar

    # @render.text
    # def clk():
    #     return input.make_tree_click()

    # @render.text
    # def dblclk():
    #     return input.make_tree_dblclick()

    """Changing the geometry"""

    @reactive.effect
    @reactive.event(input.ly_type)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        # Setting geometry after layout-type change
        if input.ly_type() in bv_objct.orig_geometry:
            ui.update_select('geometry', selected=bv_objct.orig_geometry[input.ly_type()])
        elif input.ly_type() in ['ly_dendrogram', 'ly_dendrogram_ladderized']:
            ui.update_select('geometry', selected='flat')
        elif input.ly_type() in ['ly_eq_angle', 'ly_eq_daylight']:
            ui.update_select('geometry', selected='hyperbolic')

    @reactive.effect
    @reactive.event(input.geometry)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        bv_objct.orig_geometry[input.ly_type()] = input.geometry()

    """Selecting subsets"""

    # @reactive.effect
    # @reactive.event(input.selected_annot, ignore_init=True)
    # def _():
    #     req(input.selected_annot())
    #     selected_annot_val.set(input.selected_annot())

    # @reactive.effect
    # @reactive.event(input.selected_annot_annot, ignore_init=True)
    # def _():
    #     req(input.selected_annot_annot())
    #     ui.update_switch('switch_mask', value=True)
    #     selected_annot_val.set(input.selected_annot_annot())

    # @reactive.effect
    # @reactive.event(input.go_select_subtree)
    # def _():
    #     req(selected_annot_val.get())
    #     if selected_annot_val.get() == 'subtree':
    #         req(picked_inds.get())
    #         if picked_inds.get()[-2] is not None:
    #             marker_inds = (picked_inds.get()[-2]['int_inds'], picked_inds.get()[-1]['all'],)
    #             selected_subset.set({'type': 'subtree', 'info': marker_inds, 'mask_is_on': True})

    #     # Set mask temporarily on through set_temp_mask effect
    #     global temp_mask_counter
    #     if (not input.switch_mask()) or (temp_mask_counter < (TEMP_MASK_STEPS + 1)):
    #         temp_mask_counter = 0
    #         set_temp_mask.set(set_temp_mask.get() + 1)
    #     else:
    #         # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
    #         update_mask.set(update_mask.get() + 1)


    """Three instances of selecting a subset of cells"""
    @render.ui
    def selecting_annotation_cat():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if curr_annotation_cats.get() is None:
            curr_annotation_cats.set(bv_objct.init_annotation_cats)
        selection_dict = {"No subset": {'no_subset': "No subset"}, "Subtree": {"subtree": "Subtree"}}
        if (bv_objct.bonvis_settings.node_style['annot_info'].color_type == 'categorical') and (curr_annotation_cats.get() is not None):
            selection_dict["Annotation"] = {cat: cat for cat in curr_annotation_cats.get()}
        # print(selection_dict)
        return ui.input_selectize('selected_annot', "Pick annotation", selection_dict, selected=bv_objct.init_selected_annot),

    @render.ui
    def selecting_annotation_cat_2():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if curr_annotation_cats.get() is None:
            curr_annotation_cats.set(bv_objct.init_annotation_cats)
        selection_dict = {"All remaining cells": {'no_subset': "All remaining cells"}, "Subtree": {"subtree": "Subtree"}}
        if (bv_objct.bonvis_settings.node_style['annot_info'].color_type == 'categorical') and (curr_annotation_cats.get() is not None):
            selection_dict["Annotation"] = {cat: cat for cat in curr_annotation_cats.get()}
        # print(selection_dict)
        return ui.input_selectize('selected_annot_2', "Pick annotation", selection_dict, selected=bv_objct.init_selected_annot),

    @render.ui
    def selecting_annotation_cat_annot():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if curr_annotation_cats.get() is None:
            curr_annotation_cats.set(bv_objct.init_annotation_cats)
        selection_dict = {"No subset": {'no_subset': "No subset"}}
        if (bv_objct.bonvis_settings.node_style['annot_info'].color_type == 'categorical') and (curr_annotation_cats.get() is not None):
            selection_dict["Annotation"] = {cat: cat for cat in curr_annotation_cats.get()}
        # print(selection_dict)
        return ui.input_selectize('selected_annot_annot', "Pick annotation", selection_dict, selected=bv_objct.init_selected_annot),


    """Three instances of dealing with the selected subset"""
    @reactive.effect
    @reactive.event(input.selected_annot_annot, ignore_init=True)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        subset_ind = 0
        if input.selected_annot_annot() == 'no_subset':
            selected_subsets[subset_ind].set({'type': None, 'info': None, 'mask_is_on': False})
            return
        else:
            selected_subsets[subset_ind].set({'type': 'annot', 'info': input.selected_annot_annot(), 'mask_is_on': True})

        # Set mask temporarily on through set_temp_mask effect
        bv_objct.mask_counters['temp_mask']
        if (not input.switch_mask_annot()) or (bv_objct.mask_counters['temp_mask'][subset_ind] < (TEMP_MASK_STEPS + 1)):
            bv_objct.mask_counters['temp_mask'][subset_ind] = 0
            set_temp_mask.set(set_temp_mask.get() + 1)
        else:
            # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
            update_mask.set(update_mask.get() + 1)

    @reactive.effect
    @reactive.event(input.selected_annot, ignore_init=True)
    def _():
        subset_ind = 1
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if input.selected_annot() != 'subtree':
            if input.selected_annot() == 'no_subset':
                if selected_subsets[subset_ind].get()['mask_is_on'] != input.switch_mask():
                    update_mask.set(update_mask.get() + 1)
                selected_subsets[subset_ind].set({'type': None, 'info': None, 'mask_is_on': False})
                return
            else:
                selected_subsets[subset_ind].set({'type': 'annot', 'info': input.selected_annot(), 'mask_is_on': True})

            # Set mask temporarily on through set_temp_mask effect
            if (not input.switch_mask()) or (bv_objct.mask_counters['temp_mask'][subset_ind] < (TEMP_MASK_STEPS + 1)):
                bv_objct.mask_counters['temp_mask'][subset_ind] = 0
                set_temp_mask.set(set_temp_mask.get() + 1)
            else:
                # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
                update_mask.set(update_mask.get() + 1)

    @reactive.effect
    @reactive.event(input.selected_annot_2, ignore_init=True)
    def _():
        subset_ind = 2
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if input.selected_annot_2() != 'subtree':
            if input.selected_annot_2() == 'no_subset':
                if selected_subsets[subset_ind].get()['mask_is_on'] != input.switch_mask_2():
                    update_mask.set(update_mask.get() + 1)
                selected_subsets[subset_ind].set({'type': None, 'info': None, 'mask_is_on': False})
                return
            else:
                selected_subsets[subset_ind].set({'type': 'annot', 'info': input.selected_annot_2(), 'mask_is_on': True})

            # Set mask temporarily on through set_temp_mask effect
            if (not input.switch_mask_2()) or (bv_objct.mask_counters['temp_mask'][subset_ind] < (TEMP_MASK_STEPS + 1)):
                bv_objct.mask_counters['temp_mask'][subset_ind] = 0
                set_temp_mask.set(set_temp_mask.get() + 1)
            else:
                # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
                update_mask.set(update_mask.get() + 1)

    """Three instances of resetting subset"""
    @reactive.effect
    @reactive.event(input.reset_subset)
    def _():
        # if selected_subset.get()['mask_is_on']:
        #     update_mask.set(update_mask.get() + 1)
        subset_ind = 1
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        bv_objct.mask_counters['set_temp_mask']
        # a, b = selected_tree_nodes_lst[subset_ind-1].get()
        selected_tree_nodes_lst[subset_ind-1].set((None, None))
        bv_objct.mask_counters['temp_mask'][subset_ind] = TEMP_MASK_STEPS + 1
        bv_objct.mask_counters['set_temp_mask'] = set_temp_mask.get()
        selected_subsets[subset_ind].set({'type': None, 'info': None, 'mask_is_on': False})
        ui.update_selectize('selected_annot', selected='no_subset')
        if input.switch_mask():
            ui.update_switch('switch_mask', value=False)
            update_mask.set(update_mask.get() + 1)

    @reactive.effect
    @reactive.event(input.reset_subset_2)
    def _():
        # if selected_subset.get()['mask_is_on']:
        #     update_mask.set(update_mask.get() + 1)
        subset_ind = 2
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        # c, d = selected_tree_nodes_lst[subset_ind-1].get()
        selected_tree_nodes_lst[subset_ind-1].set((None, None))
        bv_objct.mask_counters['temp_mask'][subset_ind] = TEMP_MASK_STEPS + 1
        bv_objct.mask_counters['set_temp_mask'] = set_temp_mask.get()
        selected_subsets[subset_ind].set({'type': None, 'info': None, 'mask_is_on': False})
        ui.update_selectize('selected_annot_2', selected='no_subset')
        if input.switch_mask_2():
            ui.update_switch('switch_mask_2', value=False)
            update_mask.set(update_mask.get() + 1)

    @reactive.effect
    @reactive.event(input.reset_subset_annot)
    def _():
        # if selected_subset.get()['mask_is_on']:
        #     update_mask.set(update_mask.get() + 1)
        subset_ind = 0
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        bv_objct.mask_counters['temp_mask'][subset_ind] = TEMP_MASK_STEPS + 1
        bv_objct.mask_counters['set_temp_mask'] = set_temp_mask.get()
        selected_subsets[subset_ind].set({'type': None, 'info': None, 'mask_is_on': False})
        ui.update_selectize('selected_annot_annot', selected='no_subset')
        if input.switch_mask_annot():
            ui.update_switch('switch_mask_annot', value=False)
            update_mask.set(update_mask.get() + 1)


    """Repetitive part on selecting nodes for subtrees"""
    # TODO: Capture the following repetitiveness in a module
    @render.ui
    def select_subtree_ui_tree1anc():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        selected_tree_nodes = selected_tree_nodes_lst[0]
        if selected_tree_nodes.get()[0] is None:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree1anc", "Ancestor", class_='btn-dark')),
                ui.column(6, None),
            ),
        else:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree1anc", "Ancestor", class_="btn-light")),
                ui.column(6, ui.tags.p(bv_objct.bonvis_metadata.node_ids[selected_tree_nodes.get()[0]])),
            ),

    @reactive.effect
    @reactive.event(input.go_select_tree1anc)
    def _():
        if picked_inds.get()[-1] is None:
            m = ui.modal("No node was selected yet.\n",
                ui.p("Select a node first by clicking in the tree"),
                ui.p("then click the button to indicate if its the ancestor or the downstream node."),
                style='font-size:x-large;font-style:italic',
                easy_close=True, size='xl')
            ui.modal_show(m)
        else:
            selected_tree_nodes = selected_tree_nodes_lst[0]
            a, b = selected_tree_nodes.get()
            selected_tree_nodes.set((picked_inds.get()[-1]['int_inds'], None))

    @render.ui
    def select_subtree_ui_tree1ds():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        selected_tree_nodes = selected_tree_nodes_lst[0]
        if selected_tree_nodes.get()[1] is None:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree1ds", "Downstream", class_='btn-dark')),
                ui.column(6, None),
            ),
        else:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree1ds", "Downstream", class_="btn-light")),
                ui.column(6, ui.tags.p(bv_objct.bonvis_metadata.node_ids[selected_tree_nodes.get()[1]])),
            ),

    @reactive.effect
    @reactive.event(input.go_select_tree1ds)
    def _():
        if picked_inds.get()[-1] is None:
            m = ui.modal("No node was selected yet.\n",
                ui.p("Select a node first by clicking in the tree,\n "
                     "Then click here to assign it as a sub-tree node."),
                style='font-size:x-large;font-style:italic',
                easy_close=True, size='xl')
            ui.modal_show(m)
        else:
            selected_tree_nodes = selected_tree_nodes_lst[0]
            a, b = selected_tree_nodes.get()
            selected_tree_nodes.set((a, picked_inds.get()[-1]['all']))

    @render.ui
    def select_subtree_ui_tree2anc():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        selected_tree_nodes = selected_tree_nodes_lst[1]
        if selected_tree_nodes.get()[0] is None:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree2anc", "Ancestor", class_='btn-dark')),
                ui.column(6, None),
            ),
        else:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree2anc", "Ancestor", class_="btn-light")),
                ui.column(6, ui.tags.p(bv_objct.bonvis_metadata.node_ids[selected_tree_nodes.get()[0]])),
            ),

    @reactive.effect
    @reactive.event(input.go_select_tree2anc)
    def _():
        if picked_inds.get()[-1] is None:
            m = ui.modal("No node was selected yet.\n",
                ui.p("Select a node first by clicking in the tree"),
                ui.p("then click the button to indicate if its the ancestor or the downstream node."),
                style='font-size:x-large;font-style:italic',
                easy_close=True, size='xl')
            ui.modal_show(m)
        else:
            selected_tree_nodes = selected_tree_nodes_lst[1]
            selected_tree_nodes.set((picked_inds.get()[-1]['int_inds'], None))

    @render.ui
    def select_subtree_ui_tree2ds():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        selected_tree_nodes = selected_tree_nodes_lst[1]
        if selected_tree_nodes.get()[1] is None:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree2ds", "Downstream", class_='btn-dark')),
                ui.column(6, None),
            ),
        else:
            return ui.row(
                ui.column(6, ui.input_action_button("go_select_tree2ds", "Downstream", class_="btn-light")),
                ui.column(6, ui.tags.p(bv_objct.bonvis_metadata.node_ids[selected_tree_nodes.get()[1]])),
            ),

    @reactive.effect
    @reactive.event(input.go_select_tree2ds)
    def _():
        if picked_inds.get()[-1] is None:
            m = ui.modal("No node was selected yet.\n",
                ui.p("Select a node first by clicking in the tree,\n "
                     "Then click here to assign it as a sub-tree node."),
                style='font-size:x-large;font-style:italic',
                easy_close=True, size='xl')
            ui.modal_show(m)
        else:
            selected_tree_nodes = selected_tree_nodes_lst[1]
            c, d = selected_tree_nodes.get()
            selected_tree_nodes.set((c, picked_inds.get()[-1]['all']))


    """Function that selects subtree and shows it"""
    @reactive.effect
    @reactive.event(selected_tree_nodes_lst[0])
    def _():
        selected_tree_nodes = selected_tree_nodes_lst[0]
        a, b = selected_tree_nodes.get()
        if input.selected_annot.get() == 'subtree':
            req(a)
            req(b)
            bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
            subset_ind = 1
            marker_inds = (a, b,)
            selected_subsets[subset_ind].set({'type': 'subtree', 'info': marker_inds, 'mask_is_on': True})
            # Set mask temporarily on through set_temp_mask effect
            if (not input.switch_mask()) or (bv_objct.mask_counters['temp_mask'][subset_ind] < (TEMP_MASK_STEPS + 1)):
                bv_objct.mask_counters['temp_mask'][subset_ind] = 0
                set_temp_mask.set(set_temp_mask.get() + 1)
            else:
                # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
                update_mask.set(update_mask.get() + 1)

    @reactive.effect
    @reactive.event(selected_tree_nodes_lst[1])
    def _():
        selected_tree_nodes = selected_tree_nodes_lst[1]
        c, d = selected_tree_nodes.get()
        if input.selected_annot_2.get() == 'subtree':
            req(c)
            req(d)
            bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
            subset_ind = 2
            marker_inds = (c, d,)
            selected_subsets[subset_ind].set({'type': 'subtree', 'info': marker_inds, 'mask_is_on': True})
            # Set mask temporarily on through set_temp_mask effect
            if (not input.switch_mask_2()) or (bv_objct.mask_counters['temp_mask'][subset_ind] < (TEMP_MASK_STEPS + 1)):
                bv_objct.mask_counters['temp_mask'][subset_ind] = 0
                set_temp_mask.set(set_temp_mask.get() + 1)
            else:
                # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
                update_mask.set(update_mask.get() + 1)




    """Masking cells that are not in selected subset"""
    """Setting temporary mask"""
    @reactive.effect()
    def show_mask_temp():
        # This function is called to set mask temporarily. First time, it sets the mask and calls itself after 3
        # seconds. Second time (temp_mask = False) it turns the mask off
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        invalidated = False
        if set_temp_mask.get() != bv_objct.mask_counters['set_temp_mask']:
            for subset_ind in range(len(bv_objct.mask_counters['temp_mask'])):
                if bv_objct.mask_counters['temp_mask'][subset_ind] < TEMP_MASK_STEPS:
                    if not invalidated:
                        reactive.invalidate_later(TEMP_MASK_SECS / TEMP_MASK_STEPS)
                        invalidated = True
                    if bv_objct.mask_counters['temp_mask'][subset_ind] == 0:
                        with reactive.isolate():
                            curr_selected_subset = selected_subsets[subset_ind].get()
                            logging.debug("Switching mask {} on.".format(subset_ind))
                            curr_selected_subset['mask_is_on'] = True
                            selected_subsets[subset_ind].set(curr_selected_subset)
                            update_mask.set(update_mask.get() + 1)
                elif bv_objct.mask_counters['temp_mask'][subset_ind] == TEMP_MASK_STEPS:
                    with reactive.isolate():
                        curr_selected_subset = selected_subsets[subset_ind].get()
                        logging.debug("Switching mask {} off.".format(subset_ind))
                        # if input.use_mask():
                        curr_selected_subset['mask_is_on'] = False
                        selected_subsets[subset_ind].set(curr_selected_subset)
                        # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
                        update_mask.set(update_mask.get() + 1)
                bv_objct.mask_counters['temp_mask'][subset_ind] += 1
            if not invalidated:
                bv_objct.mask_counters['set_temp_mask'] = set_temp_mask.get()

    """Three instances of switching mask permanently on with the switch"""
    @reactive.effect
    @reactive.event(input.switch_mask, ignore_init=True)
    def _():
        subset_ind = 1
        curr_selected_subset = selected_subsets[subset_ind].get()
        if input.switch_mask() and (not curr_selected_subset['mask_is_on']):
            curr_selected_subset['mask_is_on'] = True
            # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
            update_mask.set(update_mask.get() + 1)
        elif (not input.switch_mask()) and (curr_selected_subset['mask_is_on']):
            curr_selected_subset['mask_is_on'] = False
            # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
            update_mask.set(update_mask.get() + 1)

    @reactive.effect
    @reactive.event(input.switch_mask_annot, ignore_init=True)
    def _():
        subset_ind = 0
        curr_selected_subset = selected_subsets[subset_ind].get()
        if input.switch_mask_annot() and (not curr_selected_subset['mask_is_on']):
            curr_selected_subset['mask_is_on'] = True
            # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
            update_mask.set(update_mask.get() + 1)
        elif (not input.switch_mask_annot()) and (curr_selected_subset['mask_is_on']):
            curr_selected_subset['mask_is_on'] = False
            # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
            update_mask.set(update_mask.get() + 1)

    @reactive.effect
    @reactive.event(input.switch_mask_2, ignore_init=True)
    def _():
        subset_ind = 2
        curr_selected_subset = selected_subsets[subset_ind].get()
        if input.switch_mask_2() and (not curr_selected_subset['mask_is_on']):
            curr_selected_subset['mask_is_on'] = True
            # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
            update_mask.set(update_mask.get() + 1)
        elif (not input.switch_mask_2()) and (curr_selected_subset['mask_is_on']):
            curr_selected_subset['mask_is_on'] = False
            # The following calls the update_mask effect which will change the bonvis_fig-object with the new mask
            update_mask.set(update_mask.get() + 1)

    """Updating mask for all the subsets that currently have the mask on"""
    @reactive.effect
    @reactive.event(update_mask)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if curr_categorical_annot.get() is None:
            curr_categorical_annot.set(bv_objct.init_categorical_annot)
        if update_mask.get() != bv_objct.mask_counters['update_mask']:
            # print("Updating mask!")
            curr_subsets = [subset.get() for subset in selected_subsets]
            bv_objct.bonvis_fig.set_mask_for_subset(curr_subsets,
                                                         curr_categorical_annot=curr_categorical_annot.get())

            if input.switch_mask_annot() != curr_subsets[0]['mask_is_on']:
                ui.update_switch('switch_mask_annot', value=curr_subsets[0]['mask_is_on'])
            if input.switch_mask() != curr_subsets[1]['mask_is_on']:
                ui.update_switch('switch_mask', value=curr_subsets[1]['mask_is_on'])
            if input.switch_mask_2() != curr_subsets[2]['mask_is_on']:
                ui.update_switch('switch_mask_2', value=curr_subsets[2]['mask_is_on'])
            bv_objct.mask_counters['update_mask'] = update_mask.get()
            # The following will cause the redrawing of the figure, such that mask-update is shown
            redraw_mask.set(redraw_mask.get() + 1)

    """Selecting node-style here"""
    # One can select node-style through clicking row in gene expression data-frame.
    # ids_ind = feature_display.columns.get_loc('ids')

    @reactive.effect
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        # if bv_objct.feature_display is None:
        #     return
        ids_ind = bv_objct.feature_display.columns.get_loc('ids')
        try:
            data_selected = get_genes_df.data_view(selected=True)
            req(not data_selected.empty)
            with reactive.isolate():
                picked_gene.set(data_selected.iat[0, ids_ind])
                node_style.set(picked_gene.get())
        except IndexError:
            logging.debug("Shiny encountered its bug where it tries to access an old index while the dataframe changed.")

    @reactive.effect
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        # One can select node-style through clicking row in marker genes data-frame.
        marker_ids_ind = bv_objct.bonvis_fig.marker_genes_df.columns.get_loc('marker_genes')

        try:
            data_selected = get_marker_genes_df.data_view(selected=True)
            req(not data_selected.empty)
            with reactive.isolate():
                picked_gene.set(data_selected.iat[0, marker_ids_ind])
                node_style.set(picked_gene.get())
            # req(input.get_marker_genes_df_selected_rows())
            # selected_idx = input.get_marker_genes_df_selected_rows()[0]
            # picked_gene.set(bonvis_fig.marker_genes_df.iat[selected_idx, marker_ids_ind])
        except IndexError:
            logging.debug("Shiny encountered its bug where it tries to access an old index while the dataframe changed.")

    # One can select node-style through selecting annotation in Annotation-tab.
    @reactive.effect
    @reactive.event(input.node_style)
    def _():
        if input.options_accordion()[0] == 'Annotation':
            node_style.set(input.node_style())

    # One can select node-style through selecting annotation in Annotation-tab.
    @reactive.effect
    @reactive.event(input.size_style)
    def _():
        if input.options_accordion()[0] == 'Annotation':
            size_style.set(input.size_style())

    # Once going to different accordion-tab, we switch annotation to the relevant node-style
    @reactive.effect
    @reactive.event(input.options_accordion)
    def _():
        if input.options_accordion()[0] == 'Annotation':
            node_style.set(input.node_style())
        elif input.options_accordion()[0] == 'Gene expression':
            if picked_gene.get() is not None:
                node_style.set(picked_gene.get())

    """Handle clicking in the figure"""

    # Whenever something is clicked we register the closest nodes in a variable picked_inds
    @reactive.effect
    @reactive.event(input.make_tree_click)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if input.make_tree_click() is not None:
            picked_point = np.array([input.make_tree_click()[key] for key in ['x', 'y']])
            picked_inds_old = picked_inds.get()
            if picked_inds_old[0] is not None:
                picked_inds.set((None, bv_objct.bonvis_fig.pick_node(picked_point),))
            else:
                picked_inds.set((picked_inds_old[1], bv_objct.bonvis_fig.pick_node(picked_point),))

    @render.ui()
    def picked_node_id():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if picked_inds.get()[-1] is None:
            return "You didn't pick a node yet."
        else:
            vert_ind = picked_inds.get()[-1]['all']
            text = "Node ID: {}".format(bv_objct.bonvis_metadata.node_ids[vert_ind])
            cell_to_celltype = bv_objct.bonvis_fig.get_cell_to_celltype()
            label = bv_objct.bonvis_settings.node_style['annot_info'].label
            text2 = ''
            if len(cell_to_celltype) == bv_objct.bonvis_metadata.n_cells:
                vert_ind_to_cell_inds = bv_objct.bonvis_metadata.vert_info['vert_ind_to_cell_inds']
                celltypes = cell_to_celltype[vert_ind_to_cell_inds[str(vert_ind)]]
                if len(celltypes) and np.all(celltypes == celltypes[0]):
                    text2 = "{}: {}".format(label, celltypes[0])
            if len(cell_to_celltype) == bv_objct.bonvis_metadata.n_Css:
                vert_ind_to_cs_inds = bv_objct.bonvis_metadata.vert_info['vert_ind_to_cs_inds']
                celltypes = cell_to_celltype[vert_ind_to_cs_inds[str(vert_ind)]]
                if len(celltypes) and np.all(celltypes == celltypes[0]):
                    text2 = "{}: {}".format(label, celltypes[0])
            if len(cell_to_celltype) == bv_objct.bonvis_metadata.n_nodes:
                celltype = cell_to_celltype[vert_ind]
                text2 = "{}: {}".format(label, celltype)
            text = text + '<br>' + text2
            return ui.HTML(text)

    """Tweaking tree layout"""

    # Whenever the tweaking accordion-panel is opened afresh, we reset the picked nodes to None
    @reactive.effect
    @reactive.event(input.tweaking)
    def _():
        if input.tweaking() is not None:
            picked_inds.set((None, None,))

    # Whenever the layout-type is changed, we reset the picked nodes to None
    @reactive.effect
    @reactive.event(input.ly_type)
    def _():
        if input.ly_type() is not None:
            picked_inds.set((None, None,))

    """Render explanatory texts"""

    # Explanatory text for tweaking layouts
    @render.text
    def tweak_text():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if input.ly_type() == 'ly_eq_daylight':
            if picked_inds.get()[-1] is None:
                return "Pick a node by clicking."
            else:
                return "You picked: {}, you can equalize its daylight, " \
                       "or flip downstream branches.".format(
                            bv_objct.bonvis_metadata.node_ids[picked_inds.get()[1]['int_inds']])
        elif input.ly_type() == 'ly_eq_angle':
            if picked_inds.get()[-1] is None:
                return "Select a subtree by picking its ancestor and a downstream node, or pick one node for flipping " \
                       "branches."
            elif picked_inds.get()[-2] is None:
                return "You picked ancestor {}. You can flip the downstream branches, or click any node in the desired subtree to increase the angle.".format(
                    picked_inds.get()[-1]['int_inds'])
            else:
                return "You can reorder the branches downstream of {}, or give more angle to the subtree defined by " \
                       "ancestor {} and downstream node {}.".format(
                    bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-1]['all']],
                    bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-2]['int_inds']],
                    bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-1]['all']])
        elif input.ly_type().startswith('ly_dendrogram'):
            if picked_inds.get()[-1] is None:
                return "Select a new node to become the root, or to flip branch order."
            else:
                return "You picked node {}, choose whether to make it the root, or to flip branches.".format(
                    bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-1]['all']])
        else:
            return None

    # Explanatory text for selecting subtree
    @render.text
    def select_subtree_text():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if picked_inds.get()[-1] is None:
            return "Pick the ancestor and a downstream node."
        elif picked_inds.get()[-2] is None:
            return "Ancestor = {}. Click any downstream node.".format(
                bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-1]['int_inds']])
        else:
            return "Ancestor = {}, downstream node {}.".format(
                bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-2]['int_inds']],
                bv_objct.bonvis_metadata.node_ids[picked_inds.get()[-1]['all']])

    # Text for tweak button
    @render.text()
    def tweak_button_text():
        if input.ly_type() == 'ly_eq_daylight':
            return "More daylight!"
        elif input.ly_type() == 'ly_eq_angle':
            return "Increase angle!"
        elif input.ly_type().startswith('ly_dendrogram'):
            return "New root!"
        else:
            return None

    ids: list[str] = []

    # Warning popup for changing layout
    @reactive.effect
    @reactive.event(input.tweaking)
    def _():
        bv_objct = bv_objcts[(user_id, session.input[".clientdata_url_search"].get())]
        if bv_objct.bonvis_fig.bonvis_settings == 'ly_eq_daylight':
            nonlocal ids
            if input.tweaking() is not None:
                # Save the ID for removal later
                m = ui.modal("Warning: Changing the tree layout may be very slow!\n",
                             ui.p("In particular, at the first adaptation, "
                                  "the tree-object needs to be loaded which will "
                                  "take some time."),
                             style='font-size:x-large;font-style:italic',
                             easy_close=True, size='xl')
                id = ui.modal_show(m)
                ids.append(id)
            else:
                if ids:
                    ui.modal_remove(ids.pop(0))

    @reactive.calc
    def url():
        myurl_search = session.input[".clientdata_url_search"].get()
        data = [None]
        try:
            data = myurl_search.strip().split("=")
        except:
            return ""
        if data[0] != "?dir":
            return ""
        if os.path.exists("/scicore/web/scismara/scismara/www/BONSAI/jobs/{}/downloads/index.html".format(data[1])):
            return "https://bonsai.unibas.ch/BONSAI/jobs/{}/downloads/index.html".format(data[1])
        return ""

    @render.ui
    def download_page_link():
        url_to_follow = url()
        if len(url_to_follow) > 0:
            return_div = ui.div(
                ui.HTML("<strong>Download the <em>Bonsai</em> results:</strong><br>"),
                ui.HTML("Follow the link to download the full <em>Sanity</em>, <em>Cellstates</em>, "
                        "and <em>Bonsai</em> results."),
                ui.p(),
                ui.a("Go to download page", href=url(), class_="btn btn-primary", target="_blank")
            )
        else:
            return_div = ""
        return return_div

    @session.on_ended
    def _():
        for key in list(bv_objcts.keys()):
            if key[0] == user_id:
                del bv_objcts[key]


app = App(app_ui, server)
app.on_shutdown(pool.shutdown)
