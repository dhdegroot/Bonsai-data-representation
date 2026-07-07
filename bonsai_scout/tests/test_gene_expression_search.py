"""
Regression tests for the gene-search filter on the "Gene expression" tab.

`render.DataGrid(..., filters=True)` was previously disabled
(`filters=False`) after it was seen to break when switching accordion tabs.
Re-enabled after manually verifying it works with the currently pinned
shiny==1.6.3; these tests lock that behaviour in.
"""

from playwright.sync_api import Page
from shiny.playwright import controller
from shiny.run import ShinyAppProc


def _aria_rowcount(table_locator) -> int:
    value = table_locator.get_attribute("aria-rowcount")
    assert value is not None
    return int(value)


def test_gene_expression_filter_narrows_and_survives_tab_switch(
    example_results_folder, app: ShinyAppProc, page: Page
):
    page.goto(app.url)

    accordion = controller.Accordion(page, "options_accordion")
    gene_expression_panel = controller.AccordionPanel(
        page, "options_accordion", "Gene expression"
    )
    annotation_panel = controller.AccordionPanel(page, "options_accordion", "Annotation")
    gene_table = controller.OutputDataFrame(page, "get_genes_df")
    gene_rows = gene_table.loc_body.locator("> tr")

    gene_expression_panel.set(open=True)
    accordion.expect_open(["Gene expression"])

    # filters=True must render one filter input per column (ids, zscores).
    gene_table.loc_column_filter.first.wait_for(state="visible")
    assert gene_table.loc_column_filter.count() >= 1

    # The table body virtualizes rows (a fixed-size rendered window), so the
    # true (possibly filtered) row count has to be read from `aria-rowcount`
    # on the <table>, not from counting rendered <tr> elements.
    full_row_count = _aria_rowcount(gene_table.loc_table)
    assert full_row_count > 0

    filter_input = gene_table.loc_column_filter.nth(0).locator("input")
    filter_input.fill("1")
    page.wait_for_function(
        """([selector, previousCount]) => {
            const table = document.querySelector(selector);
            if (!table) return false;
            const value = table.getAttribute('aria-rowcount');
            return value !== null && parseInt(value, 10) !== previousCount;
        }""",
        arg=["#get_genes_df table", full_row_count],
        timeout=5000,
    )
    filtered_row_count = _aria_rowcount(gene_table.loc_table)
    assert 0 < filtered_row_count < full_row_count

    # Selecting a filtered row should pick a gene and switch the legend to a
    # numeric colorbar (get_cbar is only rendered for a sequential/gene-
    # expression node style, see legend_content()/get_cbar() in bonsai_scout_app.py).
    gene_rows.nth(0).click()
    page.wait_for_selector("#get_cbar img", timeout=15000)

    filter_value_before = filter_input.input_value()

    # The regression scenario: switch away from "Gene expression" and back.
    annotation_panel.set(open=True)
    accordion.expect_open(["Annotation"])
    gene_expression_panel.set(open=True)
    accordion.expect_open(["Gene expression"])

    # State must have survived the round trip with no broken/empty table.
    filter_input_after = gene_table.loc_column_filter.nth(0).locator("input")
    assert filter_input_after.input_value() == filter_value_before
    assert _aria_rowcount(gene_table.loc_table) == filtered_row_count
    gene_table.expect_selected_rows([0])
    assert page.locator(".shiny-notification, .shiny-output-error").count() == 0
