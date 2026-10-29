"""
Nearest-neighbour searches of bonsai/bonsai_approxNN.py (area B). Points are given genes x cells; expected
neighbours come from brute-force distance matrices computed here. getFracCorrectNNs also has
tests/test_nn_recall.py.
"""
import numpy as np
import pytest

import bonsai.bonsai_approxNN as approx


def _points(seed, n_genes=5, n_cells=12):
	return np.random.default_rng(seed).normal(size=(n_genes, n_cells))


def _euclidean_order(points):
	diff = points.T[:, None, :] - points.T[None, :, :]
	return np.argsort(np.sqrt((diff ** 2).sum(axis=2)), axis=1, kind='stable')


def _cosine_order(points):
	unit = points / np.linalg.norm(points, axis=0)
	return np.argsort(1.0 - unit.T @ unit, axis=1, kind='stable')


@pytest.mark.parametrize('seed', range(2))
def test_getNNsBruteForce_lists_each_cells_k_nearest_by_euclidean_distance(seed):
	points = _points(seed)
	got = approx.getNNsBruteForce(points, k=4)
	assert got.shape == (12, 4)
	np.testing.assert_array_equal(got, _euclidean_order(points)[:, :4])
	assert np.all(got[:, 0] == np.arange(12))


@pytest.mark.parametrize('metric, order', [('cosine', _cosine_order), ('euclidean', _euclidean_order)])
def test_getNNssklearn_builds_an_index_and_queries_its_own_points(metric, order):
	points = _points(2)
	index, neighbours = approx.getNNssklearn(points.T, k=5, metric=metric)
	np.testing.assert_array_equal(neighbours, order(points)[:, :5])
	assert index.n_samples_fit_ == 12
	# k larger than the number of indexed points minus one is cut down.
	_, neighbours = approx.getNNssklearn(points.T, k=50, metric=metric)
	assert neighbours.shape == (12, 11)


def test_getNNssklearn_adds_query_points_to_a_given_index():
	points = _points(3, n_cells=10)
	index, _ = approx.getNNssklearn(points[:, :8].T, k=3, metric='euclidean')
	index, neighbours = approx.getNNssklearn(points[:, 8:].T, index=index, k=3, metric='euclidean')
	assert index.n_samples_fit_ == 10
	np.testing.assert_array_equal(neighbours, _euclidean_order(points)[8:, :3])
	# Without adding, the query points are searched among the indexed ones only.
	index, _ = approx.getNNssklearn(points[:, :8].T, k=3, metric='euclidean')
	_, neighbours = approx.getNNssklearn(points[:, 8:].T, index=index, k=3, metric='euclidean', addPoints=False)
	assert index.n_samples_fit_ == 8
	order = _euclidean_order(points)
	for row, cell in enumerate((8, 9)):
		np.testing.assert_array_equal(neighbours[row], [ind for ind in order[cell] if ind < 8][:3])


def test_getApproxNNs_keeps_ids_and_grows_the_index():
	points = _points(4, n_cells=10)
	ids = ['c%d' % ind for ind in range(10)]
	index, neighbours = approx.getApproxNNs(points[:, :7], k=3, metric='euclidean', pointsIds=ids[:7])
	assert index.nnType == 'sklearn' and list(index.IDs) == ids[:7]
	np.testing.assert_array_equal(neighbours, _euclidean_order(points[:, :7])[:, :3])
	index, neighbours = approx.getApproxNNs(points[:, 7:], index=index, k=3, metric='euclidean', pointsIds=ids[7:])
	assert list(index.IDs) == ids
	np.testing.assert_array_equal(neighbours, _euclidean_order(points)[7:, :3])
	index, neighbours = approx.getApproxNNs(points[:, :2], index=index, k=50, metric='euclidean', addPoints=False,
	                                        pointsIds=['q0', 'q1'])
	assert list(index.IDs) == ids and neighbours.shape == (2, 9)


def test_removeVect_removes_points_from_a_sklearn_index():
	points = _points(5, n_cells=8)
	index, _ = approx.getApproxNNs(points, k=3, metric='euclidean', pointsIds=np.arange(8) * 10)
	approx.removeVect(index, [1, 5])
	assert index.n_samples_fit_ == 6
	np.testing.assert_array_equal(index.IDs, [0, 20, 30, 40, 60, 70])
	_, neighbours = approx.getNNssklearn(points[:, [0]].T, index=index, k=2, metric='euclidean', addPoints=False)
	kept = [0, 2, 3, 4, 6, 7]
	expected = [kept.index(ind) for ind in _euclidean_order(points)[0] if ind in kept][:2]
	np.testing.assert_array_equal(neighbours[0], expected)
	without_ids, _ = approx.getNNssklearn(points.T, k=2)
	approx.removeNNssklearn(without_ids, [0])
	assert without_ids.n_samples_fit_ == 7 and not hasattr(without_ids, 'IDs')


class FakeFaissIndex:
	nnType = 'faissBrute'

	def __init__(self):
		self.removed = []
		self.IDs = np.array([5, 6, 7])

	def remove_ids(self, inds):
		self.removed.append(list(inds))


def test_removeVect_dispatches_faiss_indices_and_leaves_lsh_alone(capsys):
	index = FakeFaissIndex()
	approx.removeVect(index, [1])
	assert index.removed == [[1]] and list(index.IDs) == [5, 7]
	approx.removeFaiss(index, [0])
	assert index.removed == [[1], [0]] and list(index.IDs) == [7]
	index.nnType = 'faissLSH'
	approx.removeVect(index, [0])
	assert index.removed == [[1], [0]] and list(index.IDs) == [7]
	assert 'not supported' in capsys.readouterr().out


def test_getFracCorrectNNs_is_the_shared_fraction_per_row():
	nns = np.array([[0, 1, 2], [3, 4, 5]])
	true = np.array([[0, 2, 9], [5, 8, 7]])
	assert approx.getFracCorrectNNs(nns, true) == pytest.approx(3 / 6)
