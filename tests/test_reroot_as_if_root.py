"""
Reference tests for the "reroot" primitive used throughout Bonsai's tree
optimisation:

    getLtqsAsIfRoot            -- reroot a single node
    getLtqsAsIfRoot_vectorized -- same computation, batched over an "i" axis

Every node keeps its own local position/precision (nodeLtqs_g, nodeW_g) plus
the position/precision of the *global* root (rootLtqs_g, rootW_g), which was
itself computed treating this node as one of the root's children across an
edge of length tConn. getLtqsAsIfRoot answers: "what would this node's
position/precision be if it, instead of the original root, were treated as
the point where all information in the tree meets?"

It does this in two algebraic steps: (1) subtract the node's own contribution
from the root to get the combined information from "everything else", (2) send
that info back across tConn to the node's location and combine it with the
node's own (already-local) info.
"""
import numpy as np
import pytest

from bonsai.bonsai_treeHelpers import getLtqsAsIfRoot, getLtqsAsIfRoot_vectorized


def build_two_child_star(rng, n_genes=5):
    """A root formed from two children: `node` (own info known directly, no
    further edge to cross) and `rest` (a stand-in for the rest of the tree),
    connected to the root via tConn and tConn2 respectively.
    """
    node_ltqs = rng.normal(size=n_genes)
    node_W = rng.uniform(0.5, 2.0, n_genes)
    tConn = 0.4
    rest_ltqs = rng.normal(size=n_genes)
    rest_var = rng.uniform(0.2, 1.0, n_genes)
    tConn2 = 0.3

    wbar_node = 1 / (tConn + 1 / node_W)
    wbar_rest = 1 / (tConn2 + rest_var)
    root_W = wbar_node + wbar_rest
    root_ltqs = (wbar_node * node_ltqs + wbar_rest * rest_ltqs) / root_W

    return dict(node_ltqs=node_ltqs, node_W=node_W, tConn=tConn,
                rest_ltqs=rest_ltqs, rest_var=rest_var, tConn2=tConn2,
                root_ltqs=root_ltqs, root_W=root_W)


@pytest.mark.parametrize("seed", range(5))
def test_reroot_matches_independent_computation(seed):
    # Rerooting at `node` should be identical to directly combining node's
    # own info with a message from `rest` that has crossed tConn + tConn2 of
    # total variance (rest's own edge to the old root, then that root's edge
    # to node) -- computed here via a completely independent code path.
    rng = np.random.default_rng(seed)
    s = build_two_child_star(rng)

    got_ltqs, got_W = getLtqsAsIfRoot(s["node_ltqs"], s["node_W"], s["tConn"], s["root_ltqs"], s["root_W"])

    wbar_rest_at_node = 1 / (s["rest_var"] + s["tConn"] + s["tConn2"])
    expected_W = s["node_W"] + wbar_rest_at_node
    expected_ltqs = (s["node_ltqs"] * s["node_W"] + wbar_rest_at_node * s["rest_ltqs"]) / expected_W

    assert got_ltqs == pytest.approx(expected_ltqs)
    assert got_W == pytest.approx(expected_W)


def test_returns_node_unchanged_when_it_dominates_the_root():
    # If the node's own contribution accounts for (numerically) all of the
    # root's precision, there is no information left to send back, so the
    # function short-circuits and returns the node's own values unchanged.
    rng = np.random.default_rng(11)
    n_genes = 5
    node_ltqs = rng.normal(size=n_genes)
    node_W = rng.uniform(0.5, 2.0, n_genes)
    tConn = 0.4
    wbar_node = 1 / (tConn + 1 / node_W)
    root_W = wbar_node + 1e-15
    root_ltqs = node_ltqs.copy()  # the negligible remainder can't move it

    got_ltqs, got_W = getLtqsAsIfRoot(node_ltqs, node_W, tConn, root_ltqs, root_W)

    assert got_ltqs == pytest.approx(node_ltqs)
    assert got_W == pytest.approx(node_W)
    # And it must be a copy, not the same array the caller can accidentally mutate.
    got_ltqs[0] += 1
    assert got_ltqs[0] != node_ltqs[0]


@pytest.mark.parametrize("seed", range(5))
def test_vectorised_matches_scalar_per_column(seed):
    rng = np.random.default_rng(seed)
    s = build_two_child_star(rng)
    n_genes, n_i = 5, 4

    node_ltqs_gi = rng.normal(size=(n_genes, n_i))
    node_W_gi = rng.uniform(0.5, 2.0, (n_genes, n_i))
    tConn_i = rng.uniform(0.1, 1.0, n_i)

    ltqs_vec, W_vec = getLtqsAsIfRoot_vectorized(node_ltqs_gi, node_W_gi, tConn_i, s["root_ltqs"], s["root_W"])

    for i in range(n_i):
        ltqs_scalar, W_scalar = getLtqsAsIfRoot(node_ltqs_gi[:, i], node_W_gi[:, i], tConn_i[i],
                                                 s["root_ltqs"], s["root_W"])
        assert ltqs_vec[:, i] == pytest.approx(ltqs_scalar)
        assert W_vec[:, i] == pytest.approx(W_scalar)
