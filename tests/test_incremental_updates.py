"""
Reference tests for the incremental-update primitives that let Bonsai update
a parent's position/precision when one child changes, without recomputing the
message-passing sum over *all* children from scratch:

    subtract_contrib_ltqs   -- remove one child's contribution from a parent
    add_contrib_ltqs        -- add one child's contribution to a parent
    getLtqsAfterChildUpdate -- swap one child's contribution for another's, in one step

Note: subtract_contrib_ltqs and add_contrib_ltqs take the parent's 
*precision* (W) as input but return the parent's *variance* (1/W) as output -- 
matching how call sites immediately store the result as (self.ltqs, ltqsVars) 
(see Tree.detach_subtree / add_subtree in bonsai_treeHelpers.py). 
getLtqsAfterChildUpdate, by contrast, takes AND returns precision throughout 
(it's used to update (xrAsIfRoot_g, WAsIfRoot_g) pairs directly). 
The tests below check both the variance/precision convention and the numerical 
identity against findNodeLtqsGivenLeafs computed from scratch.
"""
import numpy as np
import pytest

from bonsai.bonsai_treeHelpers import (
    add_contrib_ltqs,
    findNodeLtqsGivenLeafs,
    getLtqsAfterChildUpdate,
    subtract_contrib_ltqs,
)


def build_star_children(rng, n_genes=5, n_children=3):
    ltqs_gi = rng.normal(size=(n_genes, n_children))
    ltqsVars_gi = rng.uniform(0.2, 1.5, (n_genes, n_children))
    t_i = rng.uniform(0.1, 1.0, n_children)
    return ltqs_gi, ltqsVars_gi, t_i


@pytest.mark.parametrize("seed", range(5))
def test_subtract_matches_recomputing_without_that_child(seed):
    rng = np.random.default_rng(seed)
    ltqs_gi, ltqsVars_gi, t_i = build_star_children(rng)

    xr_full, W_full = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)
    xr_wo, W_wo = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi[:, :-1], ltqsVars_gi=ltqsVars_gi[:, :-1], t_i=t_i[:-1])

    wbar_last = 1 / (ltqsVars_gi[:, -1] + t_i[-1])
    ltqs_sub, var_sub = subtract_contrib_ltqs(xr_full, W_full, ltqs_gi[:, -1], wbar_last)

    assert ltqs_sub == pytest.approx(xr_wo)
    assert 1 / var_sub == pytest.approx(W_wo)


@pytest.mark.parametrize("seed", range(5))
def test_add_matches_recomputing_with_that_child(seed):
    rng = np.random.default_rng(seed)
    ltqs_gi, ltqsVars_gi, t_i = build_star_children(rng)

    xr_full, W_full = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)
    xr_wo, W_wo = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi[:, :-1], ltqsVars_gi=ltqsVars_gi[:, :-1], t_i=t_i[:-1])

    wbar_last = 1 / (ltqsVars_gi[:, -1] + t_i[-1])
    ltqs_add, var_add = add_contrib_ltqs(xr_wo, W_wo, ltqs_gi[:, -1], wbar_last)

    assert ltqs_add == pytest.approx(xr_full)
    assert 1 / var_add == pytest.approx(W_full)


@pytest.mark.parametrize("seed", range(5))
def test_subtract_then_add_round_trips(seed):
    rng = np.random.default_rng(seed)
    ltqs_gi, ltqsVars_gi, t_i = build_star_children(rng)
    xr_full, W_full = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)
    wbar_last = 1 / (ltqsVars_gi[:, -1] + t_i[-1])

    ltqs_sub, var_sub = subtract_contrib_ltqs(xr_full, W_full, ltqs_gi[:, -1], wbar_last)
    ltqs_rt, var_rt = add_contrib_ltqs(ltqs_sub, 1 / var_sub, ltqs_gi[:, -1], wbar_last)

    assert ltqs_rt == pytest.approx(xr_full)
    assert 1 / var_rt == pytest.approx(W_full)


@pytest.mark.parametrize("seed", range(5))
def test_child_update_matches_subtract_then_add_composed(seed):
    # getLtqsAfterChildUpdate should be exactly equivalent to composing
    # subtract_contrib_ltqs (remove the old child) with add_contrib_ltqs (add
    # the new one), just done in one precision-space step instead of two
    # variance<->precision round trips.
    rng = np.random.default_rng(seed)
    ltqs_gi, ltqsVars_gi, t_i = build_star_children(rng)
    xr_full, W_full = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)

    tConn = t_i[-1]
    old_ltqs, old_var = ltqs_gi[:, -1], ltqsVars_gi[:, -1]
    new_ltqs = rng.normal(size=ltqs_gi.shape[0])
    new_var = rng.uniform(0.2, 1.5, ltqs_gi.shape[0])

    got_ltqs, got_W = getLtqsAfterChildUpdate(xr_full, W_full, tConn, old_ltqs, old_var, new_ltqs, new_var)

    wbar_old = 1 / (tConn + old_var)
    wbar_new = 1 / (tConn + new_var)
    ltqs_sub, var_sub = subtract_contrib_ltqs(xr_full, W_full, old_ltqs, wbar_old)
    expected_ltqs, expected_var = add_contrib_ltqs(ltqs_sub, 1 / var_sub, new_ltqs, wbar_new)

    assert got_ltqs == pytest.approx(expected_ltqs)
    assert got_W == pytest.approx(1 / expected_var)


def test_child_update_with_identical_child_is_a_no_op():
    rng = np.random.default_rng(9)
    ltqs_gi, ltqsVars_gi, t_i = build_star_children(rng)
    xr_full, W_full = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)
    tConn = t_i[-1]
    ltqs_last, var_last = ltqs_gi[:, -1], ltqsVars_gi[:, -1]

    got_ltqs, got_W = getLtqsAfterChildUpdate(xr_full, W_full, tConn, ltqs_last, var_last, ltqs_last, var_last)

    assert got_ltqs == pytest.approx(xr_full)
    assert got_W == pytest.approx(W_full)
