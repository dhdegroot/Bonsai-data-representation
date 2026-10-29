"""
optional_preprocessing/batch_correction/batch_helpers.py (area D): get_batch_annotation, which reads the
batch-annotation file that sanity_batch.py and merge_batches.py take: a tab-separated file without a header,
column 1 the cell-IDs and column 2 the batch of each cell (the scripts' help text).
"""
import numpy as np

from optional_preprocessing.batch_correction.batch_helpers import get_batch_annotation


def _write(path, rows):
	with open(path, 'w') as handle:
		for cell, batch in rows:
			handle.write('%s\t%s\n' % (cell, batch))
	return str(path)


def test_batches_are_sorted_and_every_cell_maps_to_its_batch(tmp_path):
	"""Hand-worked: batch IDs come out sorted, counts follow that order, and every row is data (no header)."""
	rows = [('cellA', 'run2'), ('cellB', 'run10'), ('cellC', 'run2'), ('cellD', 'alpha'), ('cellE', 'run2')]
	batch_ids, cell_ind_to_batch_ind, cell_id_to_batch_id, counts = get_batch_annotation(
		_write(tmp_path / 'batches.tsv', rows))
	assert batch_ids == ['alpha', 'run10', 'run2']
	np.testing.assert_array_equal(counts, [1, 1, 3])
	np.testing.assert_array_equal(np.asarray(cell_ind_to_batch_ind).ravel(), [2, 1, 2, 0, 2])
	assert cell_id_to_batch_id == {'cellA': 'run2', 'cellB': 'run10', 'cellC': 'run2', 'cellD': 'alpha',
	                               'cellE': 'run2'}
	# The index mapping and the ID mapping agree.
	for ind, (cell, _) in enumerate(rows):
		assert batch_ids[np.asarray(cell_ind_to_batch_ind).ravel()[ind]] == cell_id_to_batch_id[cell]


def test_single_batch(tmp_path):
	rows = [('c%d' % ind, 'only') for ind in range(4)]
	batch_ids, cell_ind_to_batch_ind, cell_id_to_batch_id, counts = get_batch_annotation(
		_write(tmp_path / 'batches.tsv', rows))
	assert batch_ids == ['only']
	np.testing.assert_array_equal(counts, [4])
	np.testing.assert_array_equal(np.asarray(cell_ind_to_batch_ind).ravel(), [0, 0, 0, 0])
	assert set(cell_id_to_batch_id.values()) == {'only'}
