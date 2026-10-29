"""
Reference computations of Bonsai's model, written from the model rather than
from Bonsai's code, for trees small enough to handle with dense matrices.

The model (Bonsai paper; README "Running Bonsai on other data-types"): every
feature diffuses independently along the tree as Brownian motion, so a node's
position given its parent's is Gaussian with variance equal to the edge length;
each cell is a leaf whose measurement is Gaussian around the leaf's position
with the given variance; the root's position has a flat (improper) prior. For
Sanity input the Sanity prior has already been divided out of the data, and the
data are rescaled per gene by the gene variance, before the tree sees them.

For one gene, with S the C x C matrix of shared root-to-MRCA path lengths
(S_ii the root-to-leaf length) and V the diagonal of measurement variances,
Sigma = S + V, and integrating the root position z out of N(x; z*1, Sigma):

	log p(x) = -(C-1)/2 log(2 pi) - 1/2 log|Sigma| - 1/2 log(1' Sigma^-1 1)
	           - 1/2 (x' Sigma^-1 x - (1' Sigma^-1 x)^2 / (1' Sigma^-1 1))

Bonsai's Tree.calcLogLComplete reports (checked in test_harness.py and the
area B tests) twice this, summed over genes, without the 2 pi constant:

	bonsai_loglik = 2 * sum_g log p_g + (C - 1) * G * log(2 pi)

and, when given loglikVarCorr, adds that number; Bonsai sets it to
-(C - 1) * sum_g log(v_g), the Jacobian of rescaling every gene by sqrt(v_g).
"""
import math

import numpy as np

from bonsai_testtrees import Node


def _leaf_order(tree, cells):
	leaves = tree.leaves()
	names = [leaf.name for leaf in leaves]
	if cells is None:
		return leaves
	by_name = {leaf.name: leaf for leaf in leaves}
	if set(cells) != set(names):
		raise ValueError('cells do not match the leaves of the tree')
	return [by_name[cell] for cell in cells]


def shared_path_matrix(tree, nodes):
	"""S_ab = summed edge length from the root to the most recent common ancestor of nodes a and b."""
	parent = tree.parents()
	depth = tree.depths()

	def ancestors(node):
		chain = []
		while node is not None:
			chain.append(node)
			node = parent[id(node)]
		return chain

	chains = [ancestors(node) for node in nodes]
	ancestor_sets = [set(id(n) for n in chain) for chain in chains]
	n = len(nodes)
	matrix = np.zeros((n, n))
	for a in range(n):
		for b in range(a, n):
			for anc in chains[a]:
				if id(anc) in ancestor_sets[b]:
					matrix[a, b] = matrix[b, a] = depth[id(anc)]
					break
	return matrix


def reference_loglik_per_gene(tree, means, variances, cells=None):
	"""
	log p(x_g) for every gene, natural units, flat root prior.
	means, variances: dictionaries from leaf name to a vector over genes.
	"""
	leaves = _leaf_order(tree, cells)
	shared = shared_path_matrix(tree, leaves)
	x = np.array([np.asarray(means[leaf.name], dtype=float) for leaf in leaves])
	v = np.array([np.asarray(variances[leaf.name], dtype=float) for leaf in leaves])
	n_cells, n_genes = x.shape
	ones = np.ones(n_cells)
	out = np.zeros(n_genes)
	for g in range(n_genes):
		sigma = shared + np.diag(v[:, g])
		chol = np.linalg.cholesky(sigma)
		logdet = 2.0 * np.sum(np.log(np.diag(chol)))
		inv_ones = np.linalg.solve(sigma, ones)
		inv_x = np.linalg.solve(sigma, x[:, g])
		a = ones @ inv_ones
		b = ones @ inv_x
		quad = x[:, g] @ inv_x - b * b / a
		out[g] = -0.5 * (n_cells - 1) * math.log(2 * math.pi) - 0.5 * logdet - 0.5 * math.log(a) - 0.5 * quad
	return out


def reference_loglik(tree, means, variances, cells=None):
	"""Total log-likelihood over genes, natural units."""
	return float(math.fsum(reference_loglik_per_gene(tree, means, variances, cells=cells)))


def to_bonsai_units(loglik, n_cells, n_genes, loglik_var_corr=None):
	"""Convert a natural log-likelihood to the number Tree.calcLogLComplete reports."""
	out = 2.0 * loglik + (n_cells - 1) * n_genes * math.log(2 * math.pi)
	if loglik_var_corr is not None:
		out += loglik_var_corr
	return out


def reference_bonsai_loglik(tree, means, variances, cells=None, loglik_var_corr=None):
	n_cells = len(tree.leaves())
	n_genes = len(next(iter(means.values())))
	return to_bonsai_units(reference_loglik(tree, means, variances, cells=cells), n_cells, n_genes,
	                       loglik_var_corr=loglik_var_corr)


def reference_posteriors(tree, means, variances):
	"""
	Posterior mean and variance of every named node's position given all cell
	measurements, with the root's position integrated over its flat prior.
	Returns dictionaries from node name to vectors over genes. These are what
	Bonsai writes (rescaled back) to posterior_ltqs_vertByGene.npy and
	posterior_ltqsVars_vertByGene.npy. Zero-length edges are allowed.
	"""
	leaves = tree.leaves()
	nodes = [node for node in tree.iter_nodes() if node.name is not None]
	all_nodes = leaves + [node for node in nodes if not node.is_leaf]
	shared = shared_path_matrix(tree, all_nodes)
	n_leaves = len(leaves)
	k_ll = shared[:n_leaves, :n_leaves]
	k_nl = shared[:, :n_leaves]
	k_nn = shared
	x = np.array([np.asarray(means[leaf.name], dtype=float) for leaf in leaves])
	v = np.array([np.asarray(variances[leaf.name], dtype=float) for leaf in leaves])
	n_genes = x.shape[1]
	ones = np.ones(n_leaves)
	post_mean = np.zeros((len(all_nodes), n_genes))
	post_var = np.zeros((len(all_nodes), n_genes))
	for g in range(n_genes):
		sigma = k_ll + np.diag(v[:, g])
		inv_ones = np.linalg.solve(sigma, ones)
		inv_x = np.linalg.solve(sigma, x[:, g])
		a = ones @ inv_ones
		z_hat = (ones @ inv_x) / a
		gain = np.linalg.solve(sigma, k_nl.T).T
		post_mean[:, g] = z_hat + gain @ (x[:, g] - z_hat)
		cond_var = np.diag(k_nn) - np.einsum('ij,ij->i', gain, k_nl)
		drift = 1.0 - gain @ ones
		post_var[:, g] = cond_var + drift ** 2 / a
	means_out = {node.name: post_mean[ind] for ind, node in enumerate(all_nodes) if node.name is not None}
	vars_out = {node.name: post_var[ind] for ind, node in enumerate(all_nodes) if node.name is not None}
	return means_out, vars_out


def build_bonsai_tree(tree, means, variances):
	"""
	Build a bonsai_treeHelpers.Tree with the topology, edge lengths and leaf data
	of a Node tree, in the current process. Cells get nodeInd 0..C-1 in leaf
	order; internal nodes follow. Sets the bonsai_globals Bonsai's tree code
	reads (nCells, nGenes, nNodes, max_node_ind); use the bonsai_globals_reset
	fixture around it.
	"""
	import bonsai.bonsai_globals as bs_glob
	from bonsai.bonsai_treeHelpers import Tree, TreeNode

	leaves = tree.leaves()
	n_cells = len(leaves)
	n_genes = len(np.asarray(means[leaves[0].name]))
	bs_glob.nCells = n_cells
	bs_glob.nGenes = n_genes
	cell_ind = {leaf.name: ind for ind, leaf in enumerate(leaves)}
	counter = [n_cells]

	def make(node, parent):
		if node.is_leaf:
			out = TreeNode(nodeInd=cell_ind[node.name], childNodes=[], parentNode=parent, isLeaf=True,
			               ltqs=np.array(means[node.name], dtype=float),
			               ltqsVars=np.array(variances[node.name], dtype=float),
			               tParent=node.length, nodeId=node.name, isCell=True)
			return out
		out = TreeNode(nodeInd=counter[0], childNodes=[], parentNode=parent, isLeaf=False,
		               tParent=node.length, nodeId=node.name if node.name else 'internal_%d' % counter[0])
		counter[0] += 1
		out.childNodes = [make(child, out) for child in node.children]
		return out

	bonsai_tree = Tree()
	root = bonsai_tree.root
	root.childNodes = [make(child, root) for child in tree.children]
	n_nodes = sum(1 for _ in tree.iter_nodes())
	bs_glob.nNodes = n_nodes
	bs_glob.max_node_ind = counter[0] - 1
	bonsai_tree.nNodes = n_nodes
	bonsai_tree.max_node_ind = counter[0] - 1
	root.storeParent()
	return bonsai_tree


def star_tree(names, lengths):
	"""A Node star tree: root with one leaf per name at the given edge lengths."""
	return Node('root', None, [Node(name, length) for name, length in zip(names, lengths)])
