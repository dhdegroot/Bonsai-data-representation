"""
Regression tests for the gene-search filter on the "Marker genes" tab.

Same `filters=True` re-enablement as `test_gene_expression_search.py`, but
for `get_marker_genes_df`, which is only populated after running the
"Get markers!" task button.
"""

from playwright.sync_api import Page
from shiny.playwright import controller
from shiny.run import ShinyAppProc


def test_marker_genes_filter_renders_and_survives_tab_switch(
    example_results_folder, app: ShinyAppProc, page: Page
):
    page.goto(app.url)

    accordion = controller.Accordion(page, "options_accordion")
    marker_genes_panel = controller.AccordionPanel(page, "options_accordion", "Marker genes")
    gene_expression_panel = controller.AccordionPanel(
        page, "options_accordion", "Gene expression"
    )
    get_markers_button = controller.InputTaskButton(page, "go_marker")
    marker_table = controller.OutputDataFrame(page, "get_marker_genes_df")

    marker_genes_panel.set(open=True)
    accordion.expect_open(["Marker genes"])

    # Default subsets ("No subset" vs "No subset") are enough to produce a
    # (possibly small) marker table for this smoke test.
    get_markers_button.click()
    get_markers_button.expect_state("busy", timeout=15000)
    get_markers_button.expect_state("ready", timeout=60000)

    marker_table.loc_container.wait_for(state="visible", timeout=15000)

    # filters=True must render a filter row for the marker table too.
    marker_table.loc_column_filter.first.wait_for(state="visible")
    assert marker_table.loc_column_filter.count() >= 1

    filter_input = marker_table.loc_column_filter.nth(0).locator("input")
    filter_input.fill("Gene")
    filter_value_before = filter_input.input_value()

    # The regression scenario: switch away from "Marker genes" and back.
    gene_expression_panel.set(open=True)
    accordion.expect_open(["Gene expression"])
    marker_genes_panel.set(open=True)
    accordion.expect_open(["Marker genes"])

    marker_table.loc_container.wait_for(state="visible", timeout=15000)
    filter_input_after = marker_table.loc_column_filter.nth(0).locator("input")
    assert filter_input_after.input_value() == filter_value_before
    assert page.locator(".shiny-notification, .shiny-output-error").count() == 0
