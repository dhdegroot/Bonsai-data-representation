"""
A pure-Python newick parser and writer and topology comparison (clades, splits,
Robinson-Foulds distance), written without any of Bonsai's code so that tests
can check Bonsai's trees against it.
"""
import math


class Node:
	"""One vertex of a rooted tree: a name, the length of the edge to its parent and its children."""

	def __init__(self, name=None, length=None, children=None):
		self.name = name
		self.length = length
		self.children = [] if children is None else children

	def __repr__(self):
		return 'Node(%r, %r, %d children)' % (self.name, self.length, len(self.children))

	@property
	def is_leaf(self):
		return len(self.children) == 0

	def iter_nodes(self):
		"""Pre-order (parent before children, children in stored order)."""
		stack = [self]
		while stack:
			node = stack.pop()
			yield node
			stack.extend(reversed(node.children))

	def leaves(self):
		return [node for node in self.iter_nodes() if node.is_leaf]

	def leaf_names(self):
		return [node.name for node in self.leaves()]

	def find(self, name):
		for node in self.iter_nodes():
			if node.name == name:
				return node
		raise KeyError(name)

	def parents(self):
		"""Dictionary from id(node) to its parent node (the root maps to None)."""
		parent = {id(self): None}
		for node in self.iter_nodes():
			for child in node.children:
				parent[id(child)] = node
		return parent

	def depths(self):
		"""Dictionary from id(node) to its summed edge length from this node."""
		depth = {id(self): 0.0}
		for node in self.iter_nodes():
			for child in node.children:
				depth[id(child)] = depth[id(node)] + (child.length or 0.0)
		return depth

	def copy(self):
		return Node(self.name, self.length, [child.copy() for child in self.children])


def parse_newick(text):
	"""
	Parse one newick tree. Supports nested parentheses, names on leaves and
	internal nodes (the root included), ':length' on any node, whitespace
	between tokens and the closing ';'. Quoted names and comments are not
	supported, because neither Bonsai nor this harness writes them.
	"""
	text = text.strip()
	if not text.endswith(';'):
		raise ValueError('newick string does not end with ";"')
	pos = 0

	def skip_ws():
		nonlocal pos
		while pos < len(text) and text[pos].isspace():
			pos += 1

	def read_label():
		nonlocal pos
		start = pos
		while pos < len(text) and text[pos] not in '(),:;' and not text[pos].isspace():
			pos += 1
		return text[start:pos]

	def read_node():
		nonlocal pos
		skip_ws()
		node = Node()
		if text[pos] == '(':
			pos += 1
			while True:
				node.children.append(read_node())
				skip_ws()
				if text[pos] == ',':
					pos += 1
					continue
				if text[pos] == ')':
					pos += 1
					break
				raise ValueError('unexpected %r at position %d' % (text[pos], pos))
		skip_ws()
		label = read_label()
		node.name = label if len(label) else None
		skip_ws()
		if pos < len(text) and text[pos] == ':':
			pos += 1
			skip_ws()
			node.length = float(read_label())
		return node

	root = read_node()
	skip_ws()
	if text[pos] != ';' or pos != len(text) - 1:
		raise ValueError('trailing characters after the tree at position %d' % pos)
	return root


def _float_repr(length):
	return repr(float(length))


def write_newick(root, length_format=_float_repr):
	"""
	Write a tree in the layout Bonsai uses: '(child:t,child:t)name:t' and the root's name then ';'. Lengths are
	written as repr(float(t)), which reads back exactly; repr of a numpy float is 'np.float64(...)' on numpy 2.
	"""

	def write(node, is_root):
		out = ''
		if node.children:
			out = '(' + ','.join(write(child, False) for child in node.children) + ')'
		out += node.name if node.name is not None else ''
		if (not is_root) and (node.length is not None):
			out += ':' + length_format(node.length)
		return out

	return write(root, True) + ';'


def clades(root, min_length=0.0, include_root=False):
	"""
	The set of leaf-name sets below every internal node (rooted clades).
	Internal edges with length <= min_length are collapsed first, so a zero-length
	edge does not create a clade. Leaves themselves are not included.
	"""
	result = set()

	def visit(node, is_root):
		names = set()
		if node.is_leaf:
			names.add(node.name)
			return names
		for child in node.children:
			names |= visit(child, False)
		keep = include_root if is_root else (node.length is None or node.length > min_length)
		if keep:
			result.add(frozenset(names))
		return names

	visit(root, True)
	return result


def splits(root, min_length=0.0):
	"""
	The non-trivial bipartitions of the leaf set (unrooted topology), each given
	as the frozenset of the side that does not contain the smallest leaf name.
	Edges with length <= min_length are collapsed. Rooting, child order and
	degree-two nodes do not change the result.
	"""
	names = frozenset(root.leaf_names())
	if len(names) != len(root.leaf_names()):
		raise ValueError('leaf names are not unique')
	anchor = min(names)
	result = set()
	for clade in clades(root, min_length=min_length, include_root=False):
		side = clade if anchor not in clade else names - clade
		if 1 < len(side) < len(names) - 1:
			result.add(frozenset(side))
	return result


def rf_distance(tree1, tree2, min_length=0.0):
	"""Robinson-Foulds distance: the number of splits found in exactly one of the two trees."""
	if set(tree1.leaf_names()) != set(tree2.leaf_names()):
		raise ValueError('the trees have different leaf sets')
	return len(splits(tree1, min_length) ^ splits(tree2, min_length))


def same_topology(tree1, tree2, min_length=0.0):
	return rf_distance(tree1, tree2, min_length=min_length) == 0


def path_length(root, name1, name2):
	"""Summed edge length on the path between two named nodes."""
	parent = root.parents()
	depth = root.depths()
	node1, node2 = root.find(name1), root.find(name2)
	ancestors = set()
	node = node1
	while node is not None:
		ancestors.add(id(node))
		node = parent[id(node)]
	node = node2
	while id(node) not in ancestors:
		node = parent[id(node)]
	return depth[id(node1)] + depth[id(node2)] - 2 * depth[id(node)]


def total_length(root):
	return math.fsum(node.length for node in root.iter_nodes() if node.length is not None and node is not root)
