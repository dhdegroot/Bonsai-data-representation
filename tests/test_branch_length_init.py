"""
Reference tests for calcTInit, which produces the initial guess handed to the
numerical optimiser when scoring/executing a merge of two nodes with previous
branch lengths tOld1, tOld2 (see calcSingleDLogL and
estimateDerBasedDLogLUB in bonsai_treeHelpers.py).

The two previously-separate branches are replaced by a new shared ancestor
edge (length ~ tAnc0 = min(tOld1, tOld2) / 2) plus two remaining child edges.
`sequential` selects which of the two downstream optimisers will consume the
result, so it also selects the *shape* of the output:
  - sequential=True  -> [t12_init, tAnc0]        (2 values: optimiseT3LeafStarSequential
                                                    only actually uses tAnc0 -- t12_init is
                                                    discarded, since that optimiser
                                                    recomputes the two-leaf optimum itself)
  - sequential=False -> [t1_init, t2_init, tAnc0] (3 values, for optimiseT3LeafStar's
                                                    joint optimisation)
These are only optimiser starting points, not exact answers, so the tests
check structural invariants (shape, non-negativity, internal consistency)
rather than a single "correct" numeric output.
"""
import pytest

from bonsai.bonsai_helpers import calcTInit


@pytest.mark.parametrize("tOld1,tOld2", [(3.0, 5.0), (5.0, 3.0), (1.0, 1.0), (0.0, 0.0), (0.0, 2.0)])
def test_sequential_output_shape_and_consistency(tOld1, tOld2):
    t0_i = calcTInit(tOld1, tOld2, sequential=True)

    assert len(t0_i) == 2
    tAnc0 = min(tOld1, tOld2) / 2
    assert t0_i[1] == pytest.approx(tAnc0)
    assert t0_i[0] == pytest.approx(tOld1 + tOld2 - 2 * tAnc0)
    assert all(t >= 0 for t in t0_i)


@pytest.mark.parametrize("tOld1,tOld2", [(3.0, 5.0), (5.0, 3.0), (1.0, 1.0), (0.0, 0.0), (0.0, 2.0)])
def test_non_sequential_output_shape_and_consistency(tOld1, tOld2):
    t0_i = calcTInit(tOld1, tOld2, sequential=False)

    assert len(t0_i) == 3
    tAnc0 = min(tOld1, tOld2) / 2
    assert t0_i[2] == pytest.approx(tAnc0)
    assert t0_i[0] == pytest.approx(tOld1 - tAnc0)
    assert t0_i[1] == pytest.approx(tOld2 - tAnc0)
    assert all(t >= 0 for t in t0_i)


def test_ancestor_time_is_half_the_shorter_original_branch():
    assert calcTInit(2.0, 10.0, sequential=True)[1] == pytest.approx(1.0)
    assert calcTInit(10.0, 2.0, sequential=False)[2] == pytest.approx(1.0)


def test_is_symmetric_in_tOld1_and_tOld2_up_to_argument_order():
    # Swapping which original branch is "first" swaps which output slot gets
    # which value, but not the overall multiset of times.
    seq_ab = calcTInit(3.0, 7.0, sequential=True)
    seq_ba = calcTInit(7.0, 3.0, sequential=True)
    assert seq_ab[1] == pytest.approx(seq_ba[1])  # tAnc0 depends only on the min
    assert seq_ab[0] == pytest.approx(seq_ba[0])  # t12_init = sum - 2*tAnc0 is symmetric too

    nonseq_ab = calcTInit(3.0, 7.0, sequential=False)
    nonseq_ba = calcTInit(7.0, 3.0, sequential=False)
    assert nonseq_ab[0] == pytest.approx(nonseq_ba[1])
    assert nonseq_ab[1] == pytest.approx(nonseq_ba[0])
    assert nonseq_ab[2] == pytest.approx(nonseq_ba[2])
