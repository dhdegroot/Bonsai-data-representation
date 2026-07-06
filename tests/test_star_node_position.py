"""
Reference tests for the "combine children into a parent" primitive:

    findNodeLtqsGivenLeafs           -- vectorised over genes and children
    findNodeLtqsGivenLeafsSingleGene -- same computation for one gene

Given children with positions ltqs_gi, measurement variances ltqsVars_gi and
connecting branch lengths t_i, each child contributes a Gaussian message with
precision wbar = 1 / (var + t). The parent's position is the precision-weighted
average of its children and its precision is the sum of the children's
precisions. This is the core step used everywhere a node's position is 
(re)computed from its neighbours.
"""
import numpy as np
import pytest

from bonsai.bonsai_treeHelpers import findNodeLtqsGivenLeafs, findNodeLtqsGivenLeafsSingleGene


def test_two_children_match_manual_weighted_average():
    ltqs_gi = np.array([[0.0, 10.0]])
    ltqsVars_gi = np.array([[1.0, 3.0]])
    t_i = np.array([0.0, 0.0])

    xr_g, W_g = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)

    w1, w2 = 1 / 1.0, 1 / 3.0
    expected_xr = (0.0 * w1 + 10.0 * w2) / (w1 + w2)
    assert xr_g == pytest.approx([expected_xr])
    assert W_g == pytest.approx([w1 + w2])


def test_precision_is_sum_of_child_precisions():
    rng = np.random.default_rng(1)
    n_genes, n_children = 6, 4
    ltqs_gi = rng.normal(size=(n_genes, n_children))
    ltqsVars_gi = rng.uniform(0.1, 2.0, (n_genes, n_children))
    t_i = rng.uniform(0.0, 1.0, n_children)

    xr_g, W_g, wbar_gi = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i,
                                                 return_wbar_gi=True)

    assert wbar_gi == pytest.approx(1 / (ltqsVars_gi + t_i))
    assert W_g == pytest.approx(np.sum(wbar_gi, axis=1))
    assert xr_g == pytest.approx(np.sum(wbar_gi * ltqs_gi, axis=1) / W_g)


def test_equal_variance_children_average_to_the_mean():
    n_genes, n_children = 3, 5
    rng = np.random.default_rng(2)
    ltqs_gi = rng.normal(size=(n_genes, n_children))
    ltqsVars_gi = np.full((n_genes, n_children), 0.7)
    t_i = np.zeros(n_children)

    xr_g, W_g = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)

    assert xr_g == pytest.approx(np.mean(ltqs_gi, axis=1))
    assert W_g == pytest.approx(np.full(n_genes, n_children / 0.7))


@pytest.mark.parametrize("seed", range(5))
def test_single_gene_matches_vectorised_slice(seed):
    rng = np.random.default_rng(seed)
    n_genes, n_children = 4, 3
    ltqs_gi = rng.normal(size=(n_genes, n_children))
    ltqsVars_gi = rng.uniform(0.1, 2.0, (n_genes, n_children))
    t_i = rng.uniform(0.0, 1.0, n_children)

    xr_g, W_g, wbar_gi, wOverW_gi = findNodeLtqsGivenLeafs(
        ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i,
        return_wbar_gi=True, return_wOverW_gi=True,
    )

    for gene_ind in range(n_genes):
        xr, W, wbar_i, wOverW_i = findNodeLtqsGivenLeafsSingleGene(
            ltqs_gi[gene_ind], ltqsVars_gi[gene_ind], t_i,
            return_wbar_i=True, return_wOverW_i=True,
        )
        assert xr == pytest.approx(xr_g[gene_ind])
        assert W == pytest.approx(W_g[gene_ind])
        assert wbar_i == pytest.approx(wbar_gi[gene_ind])
        assert wOverW_i == pytest.approx(wOverW_gi[gene_ind])


def test_return_flags_control_output_arity():
    ltqs_gi = np.array([[0.0, 1.0, 2.0]])
    ltqsVars_gi = np.array([[1.0, 1.0, 1.0]])
    t_i = np.zeros(3)

    xr_only = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)
    assert len(xr_only) == 2

    with_wbar = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i, return_wbar_gi=True)
    assert len(with_wbar) == 3

    with_both = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i,
                                        return_wbar_gi=True, return_wOverW_gi=True)
    assert len(with_both) == 4
    # The leading values must agree regardless of how much extra info is requested.
    assert with_both[0] == pytest.approx(xr_only[0])
    assert with_both[1] == pytest.approx(xr_only[1])
