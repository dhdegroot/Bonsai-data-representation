from bonsai.bonsai_helpers import *

import bonsai.bonsai_globals as bs_glob
import numpy as np
import time
from scipy.optimize import brentq, minimize
from scipy.spatial import distance
from scipy.sparse import csr_array, lil_array
import itertools
import os
import gc
from itertools import permutations
from bonsai.bonsai_approxNN import getApproxNNs
import pandas as pd
from downstream_analyses.get_clusters_max_diameter import get_min_pdists_clustering_from_nwk_str


class TreeNode:
    nodeInd = None  # identifier index for the node
    vert_ind = None  # will replace nodeInd for most purposes, however nodeInd is still useful in some cases
    nodeId = None
    tParent = None  # diffusion time to parent along tree
    childNodes = None  # list with pointers to child TreeNode objects
    parentNode = None
    isLeaf = None
    isRoot = None
    isCell = None

    # Position information
    ltqs = None  # (Effective) coordinates of the (node) leaf

    # W_g is 1/ltqsVars. We store both here, because it can be beneficial to precalculate this 1/ltqsVars
    # We make sure that these two are consistent by making sure to put the other to None, whenever we update one. That's
    # why they're only accessible through functions getW(), getLtqsVars() and set(LtqsVarsOrW()
    _ltqsVars = None  # (Effective) variances of the (node) leaf
    _W_g = None  # Precision of posterior of node position when all downstream nodes are integrated out

    # Later it will be useful to let every node have its posterior position when all other nodes are integrated out.
    # While ltqs, and ltqsVars are only its position when nodes downstream of it are integrated out
    ltqsAIRoot = None
    _ltqsVarsAIRoot = None
    _W_gAIRoot = None

    prefactor = None  # Prefactor that enters loglikelihood computation that can be started from root
    dLoglikdtParent = None  # Derivative of total tree likelihood w.r.t. diff. time to parent

    nnnLoglik = None  # Loglikelihood increase when next-nearest-neighbours of this node are reconfigured
    cumClosenessNNN = None

    def __init__(self, nodeInd=None, childNodes=None, parentNode=None, isLeaf=False, isRoot=False,
                 ltqs=None, ltqsVars=None, tParent=None, nodeId=None, isCell=False, vert_ind=None):
        self.childNodes = [] if (childNodes is None) else childNodes
        self.nodeInd = nodeInd
        self.nodeId = nodeId
        self.tParent = tParent
        self.parentNode = parentNode
        self.isLeaf = isLeaf
        self.isRoot = isRoot
        self.ltqs = ltqs
        self.setLtqsVarsOrW(ltqsVars=ltqsVars)
        self.isCell = isCell
        self.vert_ind = vert_ind

    def __repr__(self):
        return "TreeNode(\n" \
               "nodeInd = %r \nnodeId = %r \n childNodes = %r \n parentNode = %r \ntParent = %r \nisLeaf = %r " \
               "\nisRoot = %r \nisCell = %r \nltqs = %r \n_ltqsVars = %r \n_W_g = %r \nltqsAIRoot = %r " \
               "\n_ltqsVarsAIRoot = %r \n_W_gAIRoot = %r \nprefactor = %r \ndLoglikdtParent = %r \nnnnLoglik = %r " \
               "\ncumClosenessNNN = %r \n)" \
               % (self.nodeInd, self.nodeId, [child.nodeInd for child in self.childNodes] if self.childNodes else None,
                  self.parentNode.nodeInd if self.parentNode else None,
                  self.tParent, self.isLeaf, self.isRoot,
                  self.isCell, self.ltqs, self._ltqsVars, self._W_g, self.ltqsAIRoot, self._ltqsVarsAIRoot,
                  self._W_gAIRoot, self.prefactor, self.dLoglikdtParent, self.nnnLoglik, self.cumClosenessNNN)

    def to_newick_node(self, use_ids=True):
        nwk_children = []
        if use_ids and self.nodeId is None:
            self.nodeId = 'new_node_id_{}'.format(self.nodeInd)
        for child in self.childNodes:
            nwk_children.append(child.to_newick_node())
        if len(nwk_children) > 0:
            nwk = ','.join(nwk_children)
            nwk = '(' + nwk + ')'
        else:
            nwk = ''
        if self.isRoot:
            own_nwk = self.nodeId if use_ids else 'N' + str(self.nodeInd)
        else:
            own_nwk = self.nodeId + ':' + str(self.tParent) if use_ids else 'N' + str(self.nodeInd) + ':' + str(
                self.tParent)
        nwk = nwk + own_nwk
        if self.isRoot:
            return nwk + ';'
        else:
            return nwk

    def get_edge_dict_node(self, edge_dict, nodeIdToVertInd=None):
        for child in self.childNodes:
            if nodeIdToVertInd is None:
                edge_dict['source'].append(self.nodeId)
                edge_dict['target'].append(child.nodeId)
                edge_dict['dist'].append(child.tParent)
                # new_row = {'source': self.nodeId, 'target': child.nodeId, 'dist': child.tParent}
            else:
                edge_dict['source'].append(self.nodeId)
                edge_dict['target'].append(child.nodeId)
                edge_dict['source_ind'].append(nodeIdToVertInd[self.nodeId])
                edge_dict['target_ind'].append(nodeIdToVertInd[child.nodeId])
                edge_dict['dist'].append(child.tParent)
                # new_row = {'source': self.nodeId, 'source_ind': nodeIdToVertInd[self.nodeId], 'target': child.nodeId,
                #            'target_ind': nodeIdToVertInd[child.nodeId], 'dist': child.tParent}
            # edge_df = edge_df.append(new_row, ignore_index=True)
            if not child.isLeaf:
                edge_dict = child.get_edge_dict_node(edge_dict, nodeIdToVertInd=nodeIdToVertInd)
        return edge_dict

    def copy_node_topology(self):
        node_copy = TreeNode(nodeInd=self.nodeInd, childNodes=[], parentNode=None, isLeaf=self.isLeaf,
                             isRoot=self.isRoot, ltqs=None, ltqsVars=None, tParent=self.tParent, nodeId=self.nodeId,
                             isCell=self.isCell)
        for child in self.childNodes:
            node_copy.childNodes.append(child.copy_node_topology())
        return node_copy

    def countDSLeafs(self, dsLeafs):
        if self.isLeaf:
            dsLeafs += 1
        for child in self.childNodes:
            dsLeafs = child.countDSLeafs(dsLeafs)
        return dsLeafs

    def getTotalTopology(self, centerLeafInd=None, use_cell_ids=False, center_leaf_id=None, with_times=False):
        if self.isRoot:
            self.storeParent()
        centerLeaf, centerLeafFound = self.getCenterLeaf(centerLeafInd=centerLeafInd, use_cell_ids=use_cell_ids,
                                                         center_leaf_id=center_leaf_id)
        if with_times:
            topology = centerLeaf.getTopologyWithTimes(use_cell_ids=use_cell_ids)
        else:
            topology = centerLeaf.getTopology(use_cell_ids=use_cell_ids)
        center_leaf_info = centerLeaf.nodeId if use_cell_ids else centerLeaf.nodeInd
        return topology, center_leaf_info

    def getTopology(self, use_cell_ids=False, skip=[]):
        topologies = []
        if self.isLeaf:
            if use_cell_ids:
                topologies.append(self.nodeId)
            else:
                topologies.append(self.nodeInd)
        for child in self.childNodes:
            if child.nodeInd not in skip:
                if child.isLeaf:
                    if use_cell_ids:
                        topologies.append(child.nodeId)
                    else:
                        topologies.append(child.nodeInd)
                else:
                    topologies.append(child.getTopology(use_cell_ids=use_cell_ids, skip=[self.nodeInd]))
        if not self.isRoot:
            if self.parentNode.nodeInd not in skip:
                topologies.append(self.parentNode.getTopology(use_cell_ids=use_cell_ids, skip=[self.nodeInd]))
        return frozenset(topologies)

    def getTopologyWithTimes(self, use_cell_ids=False, skip=[]):
        topologies = []
        if self.isLeaf:
            if use_cell_ids:
                topologies.append((self.nodeId, self.tParent))
            else:
                topologies.append((self.nodeInd, self.tParent))
        for child in self.childNodes:
            if child.nodeInd not in skip:
                if child.isLeaf:
                    if use_cell_ids:
                        topologies.append((child.nodeId, child.tParent))
                    else:
                        topologies.append((child.nodeInd, child.tParent))
                else:
                    topologies.append(
                        (child.getTopologyWithTimes(use_cell_ids=use_cell_ids, skip=[self.nodeInd]), child.tParent))
        if not self.isRoot:
            if self.parentNode.nodeInd not in skip:
                time = self.parentNode.tParent if (not self.parentNode.isRoot) else -1
                topologies.append(
                    (self.parentNode.getTopologyWithTimes(use_cell_ids=use_cell_ids, skip=[self.nodeInd]), time))
        return frozenset(topologies)

    def getCenterLeaf(self, centerLeafInd=None, use_cell_ids=False, center_leaf_id=None):
        if self.isLeaf:
            if use_cell_ids:
                if (center_leaf_id is None) or (self.nodeId == center_leaf_id):
                    return self, True
                else:
                    return None, False
            else:
                if (centerLeafInd is None) or (self.nodeInd == centerLeafInd):
                    return self, True
                else:
                    return None, False
        centerLeafFound = False
        for child in self.childNodes:
            if not centerLeafFound:
                centerLeaf, centerLeafFound = child.getCenterLeaf(centerLeafInd=centerLeafInd,
                                                                  use_cell_ids=use_cell_ids,
                                                                  center_leaf_id=center_leaf_id)
        return centerLeaf, centerLeafFound

    def get_longest_path_ds(self, ignore_leaf_edge=True):
        if self.isLeaf:
            self.max_dist_ds = 0
            self.max_summed_dist_ds = 0
            return

        max_dists = []
        for child in self.childNodes:
            child.get_longest_path_ds()
            if child.isLeaf and ignore_leaf_edge:
                max_dists.append(0)
            else:
                max_dists.append(child.tParent + child.max_dist_ds)

        max_dists = np.array(max_dists)
        node_idxs_partition = np.argpartition(a=-max_dists,
                                              kth=1)  # all entries left of the kth element are smaller than the kth element, but not sorted
        nodes_with_top_edgelength = node_idxs_partition[:2]
        top_edge_lengths = max_dists[nodes_with_top_edgelength]
        self.max_summed_dist_ds = np.sum(top_edge_lengths)
        self.max_dist_ds = np.max(top_edge_lengths)

    def find_midpoint_pair(self, max_summed_dist):
        # First find child with maximum downstream path
        for child_ind, child in enumerate(self.childNodes):
            if abs(child.max_dist_ds + child.tParent - self.max_dist_ds) < 1e-9:
                break
        # Check if midpoint is between this node and child
        if child.max_dist_ds < max_summed_dist / 2:
            # If so, return this node and the index of the child
            return self, child_ind
        else:
            # Go down to child and repeat
            max_dist_node, child_ind = child.find_midpoint_pair(max_summed_dist)
            return max_dist_node, child_ind

    def get_vert_ind_to_node_node(self, vert_ind_to_node):
        vert_ind_to_node[self.vert_ind] = self
        for child in self.childNodes:
            vert_ind_to_node = child.get_vert_ind_to_node_node(vert_ind_to_node)
        return vert_ind_to_node

    def reorderChildrenRoot(self, maxChild=8, verbose=False):
        # Gather mean of ltqs of children
        nChild = len(self.childNodes)
        childrenLtqs = np.zeros((bs_glob.nGenes, len(self.childNodes)))
        for ind, child in enumerate(self.childNodes):
            childrenLtqs[:, ind] = child.ltqs
        # Get distances between children
        childDists = np.log(1e-6 + distance.pdist(childrenLtqs.T))

        if nChild <= 3:  # In case there is only 3 children, there are no re-orderings possible
            childSeq = self.childNodes
        elif nChild <= maxChild:  # If the root does not have too many children, we can try all configurations
            # Get all permutations of childInds up to cyclic permutations
            allOrders = list(permutations(range(1, nChild)))
            cyclicOrders = np.concatenate(
                (np.array([0] * len(allOrders))[:, np.newaxis], np.array(allOrders)), axis=1)
            # Get rid of orderings that are mirror images
            child1 = cyclicOrders[0, 1]
            child2 = cyclicOrders[0, 2]
            cleanOrders = np.array(list(
                filter(lambda ordering: np.where(ordering == child1)[0] < np.where(ordering == child2)[0],
                       np.array(cyclicOrders))))
            # We minimize the total distance between neighbours
            totalDist = np.zeros(len(cleanOrders))
            for order_ind, ordering in enumerate(cleanOrders):
                for ind in range(len(ordering)):
                    totalDist[order_ind] += childDists[get_pdist_inds(nChild, ordering[ind], ordering[ind - 1])]
            childSeq = [self.childNodes[ind] for ind in cleanOrders[np.argmin(totalDist), :]]
        else:
            if verbose:
                print("Too many branches from root to try all combinations."
                      "Using greedy approach that might not reach global optimum.")
            graph_kruskal = OwnGraph(nChild)
            for cell1 in range(nChild):  # We add all child-child pairs as edges with as length their pdist
                for cell2 in range(cell1 + 1, nChild):
                    graph_kruskal.add_edge(cell1, cell2, childDists[get_pdist_inds(nChild, cell1, cell2)])
            edges = graph_kruskal.kruskal_algo_maxdegree_two(very_verbose=False)  # Then we find the MST using Kruskal
            edges = np.array(edges)
            tmp, counts = np.unique(edges, return_counts=True)
            childSeq = [tmp[counts == 1][0]]  # Find the node with only 1 connection, to start the cycle
            while len(childSeq) < nChild:
                row, col = np.where(childSeq[-1] == edges)
                childSeq = childSeq + list(edges[row, 1 - col])
                edges = np.delete(edges, row, axis=0)
            childSeq = [self.childNodes[ind] for ind in childSeq]
        self.childNodes = childSeq

        # Now order grandchildren according to a similar scheme
        for childInd, child in enumerate(self.childNodes):
            child.reorderChildren(leftNeighbour=self.childNodes[(childInd - 1) % nChild], maxChild=maxChild,
                                  rightNeighbour=self.childNodes[(childInd + 1) % nChild])

    def reset_root_node(self, parent_ind=None, old_tParent=None):
        # Get all connecting nodes
        new_children = []
        new_parent = None
        old_tParents = []
        for node in self.childNodes:
            if node.vert_ind == parent_ind:
                new_parent = node
            else:
                new_children.append(node)
                old_tParents.append(node.tParent)
        if self.parentNode is not None:
            if self.parentNode.vert_ind == parent_ind:
                new_parent = self.parentNode
            else:
                old_tParents.append(self.parentNode.tParent)
                self.parentNode.tParent = old_tParent
                new_children.append(self.parentNode)
        if new_parent is not None:
            self.parentNode = new_parent
        else:
            self.parentNode = None
            self.tParent = None
            self.isRoot = True
        if len(new_children) == 0:
            self.isLeaf = True
        else:
            self.isLeaf = False
            self.childNodes = new_children
        for ind, child in enumerate(self.childNodes):
            child.reset_root_node(parent_ind=self.vert_ind, old_tParent=old_tParents[ind])

    def reorderChildren(self, leftNeighbour, rightNeighbour, maxChild=8):
        starryCounter = 0
        # Gather mean of ltqs of children and left and rightNeighbour
        if self.isLeaf:
            return
        nChild = len(self.childNodes)
        childrenLtqs = np.zeros((bs_glob.nGenes, nChild + 2))
        for ind, child in enumerate(self.childNodes):
            childrenLtqs[:, ind + 1] = child.ltqs
        childrenLtqs[:, 0] = leftNeighbour.ltqs
        childrenLtqs[:, -1] = rightNeighbour.ltqs
        # Get distances between children
        # TODO: Check why this can give a divide by zero warning
        childDists = distance.squareform(np.log(1e-9 + distance.pdist(childrenLtqs.T)))
        if nChild <= maxChild:
            # Get all permutations of childInds up to cyclic permutations
            # Note that we permute 1,..,nChild, since 0 and nChild+1 are given by left and right neighbour
            allOrders = list(permutations(range(1, nChild + 1)))
            allOrders = np.concatenate((np.array([0] * len(allOrders))[:, np.newaxis], np.array(allOrders),
                                        np.array([nChild + 1] * len(allOrders))[:, np.newaxis]), axis=1)
            # We minimize the total distance between neighbours
            totalDist = np.zeros(len(allOrders))
            for order_ind, ordering in enumerate(allOrders):
                for ind in range(1, len(ordering)):
                    totalDist[order_ind] += childDists[ordering[ind], ordering[ind - 1]]
            indSeq = allOrders[np.argmin(totalDist), :][1:-1] - 1
            childSeq = [self.childNodes[ind] for ind in indSeq]
        else:
            starryCounter += 1
            # Delete rows for leftNeighbour and rightNeighbour (first and last row)
            leftNeighbourInd = 0
            rightNeighbourInd = nChild + 1
            childDists = np.delete(childDists, [leftNeighbourInd, rightNeighbourInd], axis=0)
            remainingChildren = np.arange(nChild)

            childSeqLeft = []
            childSeqRight = []

            # Now fill up childSeq by first adding the cell closest to either leftNeighbour or rightNeighbour
            while childDists.shape[0] > 1:
                closestLeft = np.argmin(childDists[:, leftNeighbourInd])  # who is closest to leftNeighbour
                distLeft = childDists[closestLeft, leftNeighbourInd]
                closestRight = np.argmin(childDists[:, rightNeighbourInd])
                distRight = childDists[closestRight, rightNeighbourInd]

                if distRight < distLeft:
                    origChildInd = remainingChildren[closestRight]
                    childSeqRight = [self.childNodes[
                                         origChildInd]] + childSeqRight  # Add child on left of right childSeq
                    rightNeighbourInd = 1 + origChildInd  # Replace rightNeighbourInd by just added child
                    deletedChildInd = closestRight
                else:
                    origChildInd = remainingChildren[closestLeft]
                    childSeqLeft = childSeqLeft + [
                        self.childNodes[origChildInd]]  # Add child on right side of childOrder
                    leftNeighbourInd = 1 + origChildInd  # Replace rightNeighbourInd by just added child
                    deletedChildInd = closestLeft
                childDists = np.delete(childDists, deletedChildInd, axis=0)  # Delete row for just added child
                remainingChildren = np.delete(remainingChildren, deletedChildInd)

            childSeqLeft = childSeqLeft + [self.childNodes[remainingChildren[0]]]  # Add last child in middle
            childSeq = childSeqLeft + childSeqRight

        self.childNodes = childSeq

        # Now order grandchildren according to a similar scheme
        for childInd, child in enumerate(self.childNodes):
            leftNeighbourCh = leftNeighbour if childInd == 0 else self.childNodes[childInd - 1]
            rightNeighbourCh = rightNeighbour if childInd == (nChild - 1) else self.childNodes[childInd + 1]
            child.reorderChildren(leftNeighbour=leftNeighbourCh, rightNeighbour=rightNeighbourCh)

    # Used
    def setLtqsVarsOrW(self, ltqsVars=None, W_g=None, AIRoot=False):
        if not AIRoot:
            if ltqsVars is not None:
                self._ltqsVars = ltqsVars
                self._W_g = None
            elif W_g is not None:
                self._W_g = W_g
                self._ltqsVars = None
        else:
            if ltqsVars is not None:
                self._ltqsVarsAIRoot = ltqsVars
                self._W_gAIRoot = None
            elif W_g is not None:
                self._W_gAIRoot = W_g
                self._ltqsVarsAIRoot = None

    # Used
    def getLtqsVars(self, AIRoot=False, mem_friendly=False):
        if not AIRoot:
            if self._ltqsVars is None:
                if self._W_g is not None:
                    self._ltqsVars = 1 / self._W_g
                    if mem_friendly:
                        self._W_g = None
            return self._ltqsVars
        else:
            if self._ltqsVarsAIRoot is None:
                if self._W_gAIRoot is not None:
                    self._ltqsVarsAIRoot = 1 / self._W_gAIRoot
                    if mem_friendly:
                        self._W_gAIRoot = None
            return self._ltqsVarsAIRoot

    # Used
    def getW(self, AIRoot=False, mem_friendly=False):
        if not AIRoot:
            if self._W_g is None:
                if self._ltqsVars is not None:
                    # If ltqsVars is known, return 1/ltqsVars, if that is None as well, then just return None
                    self._W_g = 1 / self._ltqsVars
                    if mem_friendly:
                        self._ltqsVars = None
            return self._W_g
        else:
            if self._W_gAIRoot is None:
                if self._ltqsVarsAIRoot is not None:
                    self._W_gAIRoot = 1 / self._ltqsVarsAIRoot
                    if mem_friendly:
                        self._ltqsVarsAIRoot = None
            return self._W_gAIRoot

    # Used
    def assignTs(self, t, verbose=False):
        if not self.isRoot:
            self.tParent = t[self.nodeInd]
        for child in self.childNodes:
            child.assignTs(t)

    # Used
    def getTs(self, t):
        if not self.isRoot:
            t[self.nodeInd] = self.tParent
        for child in self.childNodes:
            t = child.getTs(t)
        return t

    # Used
    def getGrads(self, grad):
        if not self.isRoot:
            grad[self.nodeInd] = self.dLoglikdtParent
        for child in self.childNodes:
            grad = child.getGrads(grad)
        return grad

    # Used
    def getInfo(self):
        ltqsVars = self.getLtqsVars()
        return self.tParent, 1 / (self.tParent + ltqsVars), self.ltqs, ltqsVars, self.nodeInd

    # Used
    def getChildMergers(self, mergers):
        if self.isLeaf:
            return self.nodeInd
        childInds = []
        for child in self.childNodes:
            childInds.append(child.getChildMergers(mergers))
        if len(childInds) == 0:
            mp_print("This node was marked as not-a-leaf, but doesn't have children. Check this.")
            self.isLeaf = True
            return self.nodeInd
        myInd = childInds[0]
        for ind in range(len(childInds) - 1):
            if ind == 0:
                tChild1 = self.childNodes[ind].tParent
                tChild2 = self.childNodes[ind + 1].tParent
                mergers.append((
                    childInds[ind], childInds[ind + 1], tChild1,
                    tChild2))
            else:
                tChild = self.childNodes[ind + 1].tParent
                mergers.append((myInd, childInds[ind + 1], 0., tChild))
        return myInd

    # Used
    def getDerivativesDownstream(self, xrAsIfRoot_g, WAsIfRoot_g):
        # The following is the same calculation for all children:
        ltqsTimesWAsIfRoot_g = xrAsIfRoot_g * WAsIfRoot_g

        # Then start loop over children to get all diffusion time derivatives
        for child in self.childNodes:
            # Calculate node position without contribution of this child
            wbarChild_g = 1 / (child.tParent + child.getLtqsVars())
            WWOChild = WAsIfRoot_g - wbarChild_g
            ltqsWOChild = (ltqsTimesWAsIfRoot_g - wbarChild_g * child.ltqs) / WWOChild
            ltqsVarsWOChild = 1 / WWOChild

            # Calculate derivative of loglikelihood w.r.t. diff time to child
            sqDistToChild = (ltqsWOChild - child.ltqs) ** 2
            totalVars = child.getLtqsVars() + ltqsVarsWOChild
            child.dLoglikdtParent = der2LeafTree(child.tParent, totalVars, sqDistToChild)

            if not child.isLeaf:
                # Calculate ltqs and W of child as if root, such that we can move down the tree
                wbarRoot_g = 1 / (child.tParent + ltqsVarsWOChild)
                WChildWithRoot = child.getW() + wbarRoot_g
                ltqsChildWithRoot = (child.ltqs * child.getW() + wbarRoot_g * ltqsWOChild) / WChildWithRoot

                # Move down tree to calculate derivatives from all nodes
                child.getDerivativesDownstream(ltqsChildWithRoot, WChildWithRoot)

    def assignVs(self, optLambdas):
        if self.isLeaf:
            self.ltqs /= np.sqrt(optLambdas)
            self.setLtqsVarsOrW(ltqsVars=self.getLtqsVars() / optLambdas)
        else:
            for child in self.childNodes:
                child.assignVs(optLambdas)

    # Used
    def getLtqsComplete(self, mem_friendly=True):
        if self.isLeaf:
            self.prefactor = 0.
            return

        self.prefactor = 0
        for cInd, child in enumerate(self.childNodes):
            child.getLtqsComplete(mem_friendly=mem_friendly)
            self.prefactor += child.prefactor
        if not mem_friendly:
            ltqsChildren, ltqsVarsChildren, tChildren = self.getInfoChildren()
            self.ltqs, W_g, wbar_gi = findNodeLtqsGivenLeafs(ltqs_gi=ltqsChildren, ltqsVars_gi=ltqsVarsChildren,
                                                             t_i=tChildren, return_wbar_gi=True)
        else:
            self.ltqs, W_g = findNodeLtqsGivenLeafs(childNodes=self.childNodes, return_wbar_gi=False)
            wbar_gi = None
            # self.ltqs, W_g, wbar_gi = getNodeLtqsGivenChildnodes(self.childNodes, mem_friendly=mem_friendly)
        self.setLtqsVarsOrW(W_g=W_g)
        if not mem_friendly:
            self.prefactor += getLoglikAndGradStarTree(ltqs_gi=ltqsChildren, xr_g=self.ltqs, wbar_gi=wbar_gi,
                                                       W_g=self.getW(), returnGrad=False, mem_friendly=False)
        else:
            self.prefactor += getLoglikAndGradStarTree(childNodes=self.childNodes, xr_g=self.ltqs, wbar_gi=wbar_gi,
                                                       W_g=self.getW(mem_friendly=True), returnGrad=False,
                                                       mem_friendly=True)

    def getLogLGradLambdaSingleGene(self, lam, geneInd, loglik, logGrad):
        nChildren = len(self.childNodes)
        ltqsNew_i = np.zeros(nChildren)
        ltqsVarsNew_i = np.zeros(nChildren)
        t_i = np.zeros(nChildren)
        if self.isLeaf:
            # TODO: Maybe make getLtqsVars take gene_ind as argument
            return loglik, logGrad, self.ltqs[geneInd] / np.sqrt(lam), self.getLtqsVars()[geneInd] / lam, self.tParent

        for child_ind, child in enumerate(self.childNodes):
            loglik, logGrad, ltqsNew_i[child_ind], ltqsVarsNew_i[child_ind], t_i[
                child_ind] = child.getLogLGradLambdaSingleGene(lam, geneInd, loglik, logGrad)

        xr, W, wbar_i, wOverW_i = findNodeLtqsGivenLeafsSingleGene(ltqsNew_i, ltqsVarsNew_i, t_i,
                                                                   return_wbar_i=True,
                                                                   return_wOverW_i=True)
        sqDistsTimesWbar_i = wbar_i * (ltqsNew_i - xr) ** 2

        loglik += - np.log(W) + np.sum(np.log(wbar_i) - sqDistsTimesWbar_i)
        logGrad += - np.sum(wbar_i * (ltqsVarsNew_i * (wOverW_i - 1) - t_i * sqDistsTimesWbar_i))
        return loglik, logGrad, xr, 1 / W, self.tParent

    def getLogLGradLambda(self, lam, loglik, logGrad, nGenes):
        if self.isLeaf:
            # TODO: Maybe make getLtqsVars take gene_ind as argument
            return loglik, logGrad, self.ltqs / np.sqrt(lam), self.getLtqsVars() / lam, self.tParent

        nChildren = len(self.childNodes)
        ltqsNew_gi = np.zeros((nGenes, nChildren))
        ltqsVarsNew_gi = np.zeros((nGenes, nChildren))
        t_i = np.zeros(nChildren)
        for child_ind, child in enumerate(self.childNodes):
            loglik, logGrad, ltqsNew_gi[:, child_ind], ltqsVarsNew_gi[:, child_ind], t_i[
                child_ind] = child.getLogLGradLambda(lam, loglik, logGrad, nGenes)

        xr_g, W_g, wbar_gi, wOverW_gi = findNodeLtqsGivenLeafs(ltqs_gi=ltqsNew_gi, ltqsVars_gi=ltqsVarsNew_gi, t_i=t_i,
                                                               return_wbar_gi=True,
                                                               return_wOverW_gi=True)
        sqDistsTimesWbar_gi = wbar_gi * (ltqsNew_gi - xr_g[:, None]) ** 2

        loglik += np.sum(- np.log(W_g) + np.sum(np.log(wbar_gi) - sqDistsTimesWbar_gi, axis=1))
        logGrad += - np.sum(wbar_gi * (ltqsVarsNew_gi * (wOverW_gi - 1) - t_i * sqDistsTimesWbar_gi), axis=1)
        return loglik, logGrad, xr_g, 1 / W_g, self.tParent

    # Used
    def getLtqsNoneOnly(self):
        if self.isLeaf:
            return

        for cInd, child in enumerate(self.childNodes):
            if child.ltqs is None:
                child.getLtqsNoneOnly()
        # ltqsChildren, ltqsVarsChildren, tChildren = self.getInfoChildren()
        # self.ltqs, W_g = findNodeLtqsGivenLeafs(ltqsChildren, ltqsVarsChildren, tChildren, return_wbar_gi=False)
        self.ltqs, W_g = findNodeLtqsGivenLeafs(childNodes=self.childNodes)
        self.setLtqsVarsOrW(W_g=W_g)

    # Used
    def setLtqsUpstream(self):
        # This assumes that the ltqs of this node and all of its siblings are already correct
        if self.isRoot:
            return
        parent = self.parentNode
        # ltqsChilds, ltqsVarsChilds, tChilds = parent.getInfoChildren()
        # parent.ltqs, W_g = findNodeLtqsGivenLeafs(ltqsChilds, ltqsVarsChilds, tChilds, return_wbar_gi=False)
        parent.ltqs, W_g = findNodeLtqsGivenLeafs(childNodes=parent.childNodes, return_wbar_gi=False)
        parent.setLtqsVarsOrW(W_g=W_g)
        parent.setLtqsUpstream()

    # Used
    def getLtqsUponMerge(self):
        # ltqsChilds, ltqsVarsChilds, tChilds = self.getInfoChildren()
        self.ltqs, W_g = findNodeLtqsGivenLeafs(childNodes=self.childNodes, return_wbar_gi=False)
        self.setLtqsVarsOrW(W_g=W_g)

    def gatherInfoDepthFirst(self, info):
        info.append(len(self.childNodes))
        for child in self.childNodes:
            info = child.gatherInfoDepthFirst(info)
        return info

    def gatherInfoDepthFirstGeneral(self, info):
        if self.isLeaf:
            info.append(self.ltqs[:2])
        for child in self.childNodes:
            info = child.gatherInfoDepthFirstGeneral(info)
        return info

    def deleteParentsWithOneChild(self, recalc_ltqs=True):
        toBeDeleted = []
        toBeAdded = []
        for ind, child in enumerate(self.childNodes):
            child.deleteParentsWithOneChild()
            if (len(child.childNodes) == 1) and (not child.isCell):
                toBeDeleted.append(ind)
                gchild = child.childNodes[0]
                toBeAdded.append(gchild)
                gchild.tParent += child.tParent
                gchild.parentNode = self
        self.childNodes = [child for ind, child in enumerate(self.childNodes) if ind not in toBeDeleted]
        for child in toBeAdded:
            self.childNodes.append(child)

        if self.isRoot:
            while len(self.childNodes) == 1:
                # In this case, the root itself has only a single child. Then remove the child and add the
                # children as children of the root
                child = self.childNodes[0]
                for gchild in child.childNodes:
                    gchild.tParent += child.tParent
                    gchild.parentNode = self
                self.childNodes = child.childNodes
            # self.renumberNodes()
            vertIndToNode, bs_glob.nNodes = self.renumber_verts(vertIndToNode={}, vert_count=0)
            if recalc_ltqs:
                self.getLtqsComplete(mem_friendly=True)

    # Used
    def mergeZeroTimeChilds(self):
        # This function finds edges with zero length, then deletes the node downstream of this edge and adds the
        # children of the deleted node as children of the node upstream of the edge
        childrenToBeAdded = []
        childIndsToBeDeleted = []
        for ind, child in enumerate(self.childNodes):
            if not child.isLeaf:
                child.mergeZeroTimeChilds()
                if child.tParent == 0:
                    if child.isCell and (not self.isCell):
                        self.nodeId = child.nodeId
                    childrenToBeAdded += child.childNodes
                    childIndsToBeDeleted.append(ind)
        if len(childIndsToBeDeleted) > 0:
            for child in childrenToBeAdded:
                child.parentNode = self
            self.childNodes = [child for ind, child in enumerate(self.childNodes) if ind not in childIndsToBeDeleted]
            self.childNodes += childrenToBeAdded

    # Used
    def renumberNodes(self):
        # This function renumbers internal nodes to make these indices consistent again after nodes were deleted
        # It assumes that the cells have nodeInds from 0 to nCells-1, and it doesn't change that
        if self.isRoot:
            bs_glob.nNodes = bs_glob.nCells + 1
            self.nodeInd = -1
            self.nodeId = 'root'
        for child in self.childNodes:
            if not child.isLeaf:
                child.renumberNodes()
        if not self.isRoot:
            self.nodeInd = bs_glob.nNodes - 1
            self.nodeId = 'internal_' + str(self.nodeInd)
            bs_glob.nNodes += 1

    def renumber_verts(self, vertIndToNode, vert_count, include_nodeInd=False, old_ind_to_new_ind=None):
        if old_ind_to_new_ind is not None:
            old_ind_to_new_ind[self.vert_ind] = vert_count
        self.vert_ind = vert_count
        if include_nodeInd:
            self.nodeInd = vert_count
        vertIndToNode[self.vert_ind] = self
        vert_count += 1
        for child in self.childNodes:
            if old_ind_to_new_ind is not None:
                vertIndToNode, vert_count, old_ind_to_new_ind = child.renumber_verts(vertIndToNode, vert_count,
                                                                                     include_nodeInd=include_nodeInd,
                                                                                     old_ind_to_new_ind=old_ind_to_new_ind)
            else:
                vertIndToNode, vert_count = child.renumber_verts(vertIndToNode, vert_count,
                                                                 include_nodeInd=include_nodeInd,
                                                                 old_ind_to_new_ind=old_ind_to_new_ind)
        if old_ind_to_new_ind is not None:
            return vertIndToNode, vert_count, old_ind_to_new_ind
        else:
            return vertIndToNode, vert_count

    # Used
    def getInfoChildren(self):
        nChildren = len(self.childNodes)
        ltqsChildren = np.zeros((bs_glob.nGenes, nChildren))
        ltqsVarsChildren = np.zeros((bs_glob.nGenes, nChildren))
        tChildren = np.zeros(nChildren)
        # nodeIndChilds = np.zeros(nChildren + 1, dtype=int)
        for cInd, child in enumerate(self.childNodes):
            ltqsChildren[:, cInd] = child.ltqs
            ltqsVarsChildren[:, cInd] = child.getLtqsVars()
            tChildren[cInd] = child.tParent
        return ltqsChildren, ltqsVarsChildren, tChildren

    def addClosenessNNN(self, dist, src=np.nan):
        dist += 1
        if not self.isRoot:
            nbs = self.childNodes + [self.parentNode]
        else:
            nbs = self.childNodes
        for nb in nbs:
            if (not nb.isLeaf) and (not nb.isRoot) and (nb.nodeInd != src):
                nb.cumClosenessNNN += 1 / dist
                nb.addClosenessNNN(dist=dist, src=self.nodeInd)

    # Used
    def mergeDownstreamChildren(self, xrAsIfRoot_g, WAsIfRoot_g, sequential=True, verbose=False, ellipsoidSize=1.0,
                                nChildNN=-1, kNN=5, mergeDownstream=True, tree=None):
        # If we update the node's position in the process, we keep the old position here. This will help in updating the
        # parent's position. Also, we store in the boolean "someMergeHappened" whether there was an update
        oldLtqs = None
        oldLtqsVars = None
        someMergeHappened = False
        for ind, child in enumerate(self.childNodes):
            if not child.isLeaf:
                mergeHappened, newLtqs, newLtqsVars, oldLtqsChild, oldLtqsVarsChild = child.mergeChildrenRecursive(
                    xrAsIfRoot_g, WAsIfRoot_g, sequential=sequential, verbose=verbose, ellipsoidSize=ellipsoidSize,
                    nChildNN=nChildNN, kNN=kNN,
                    mergeDownstream=mergeDownstream, tree=tree)
                if mergeHappened:  # If merge happened downstream, we need to recalculate W_g and ltqs of current node
                    someMergeHappened = True
                    xrAsIfRoot_g, WAsIfRoot_g = getLtqsAfterChildUpdate(xrAsIfRoot_g, WAsIfRoot_g, child.tParent,
                                                                        oldLtqsChild, oldLtqsVarsChild,
                                                                        child.ltqs, child.getLtqsVars())
                    if not self.isRoot:
                        if oldLtqs is None:
                            oldLtqs = self.ltqs.copy()
                            oldLtqsVars = self.getLtqsVars().copy()
                        self.ltqs, W_g = getLtqsAfterChildUpdate(self.ltqs, self.getW(), child.tParent, oldLtqsChild,
                                                                 oldLtqsVarsChild, child.ltqs, child.getLtqsVars())
                        self.setLtqsVarsOrW(W_g=W_g)
        if someMergeHappened:
            if self.isRoot:
                self.setLtqsVarsOrW(W_g=WAsIfRoot_g)
                self.ltqs = xrAsIfRoot_g
        return someMergeHappened, xrAsIfRoot_g, WAsIfRoot_g, oldLtqs, oldLtqsVars

    # Used
    def mergeDownstreamChildrenSingle(self, xrAsIfRoot_g, WAsIfRoot_g, childNInd, gchildNInd, sequential=True,
                                      verbose=False, ellipsoidSize=1.0, singleProcess=False, random=False,
                                      runConfigs_inherited=None):
        # If we update the node's position in the process, we keep the old position here. This will help in updating the
        # parent's position. Also, we store in the boolean "someMergeHappened" whether there was an update
        oldLtqs = None
        oldLtqsVars = None
        someMergeHappened = False
        for ind, child in enumerate(
                self.childNodes):  # TODO: Make this for-loop by just using child-ind instead of childNodeInd
            if child.nodeInd == childNInd:
                mergeHappened, newLtqs, newLtqsVars, oldLtqsChild, oldLtqsVarsChild = child.mergeChildrenRecursiveSingle(
                    xrAsIfRoot_g, WAsIfRoot_g, gchildNInd, sequential=sequential, verbose=verbose,
                    ellipsoidSize=ellipsoidSize, singleProcess=singleProcess, random=random,
                    runConfigs_inherited=runConfigs_inherited)
                if mergeHappened:  # If merge happened downstream, we need to recalculate W_g and ltqs of current node
                    someMergeHappened = True
                    xrAsIfRoot_g, WAsIfRoot_g = getLtqsAfterChildUpdate(xrAsIfRoot_g, WAsIfRoot_g, child.tParent,
                                                                        oldLtqsChild, oldLtqsVarsChild,
                                                                        child.ltqs, child.getLtqsVars())
                    if not self.isRoot:
                        if oldLtqs is None:
                            oldLtqs = self.ltqs.copy()
                            oldLtqsVars = self.getLtqsVars().copy()
                        self.ltqs, W_g = getLtqsAfterChildUpdate(self.ltqs, self.getW(), child.tParent, oldLtqsChild,
                                                                 oldLtqsVarsChild, child.ltqs, child.getLtqsVars())
                        self.setLtqsVarsOrW(W_g=W_g)
        if someMergeHappened:
            if self.isRoot:
                self.setLtqsVarsOrW(W_g=WAsIfRoot_g)
                self.ltqs = xrAsIfRoot_g
        return someMergeHappened, xrAsIfRoot_g, WAsIfRoot_g, oldLtqs, oldLtqsVars

    # Used
    def mergeChildrenRecursive(self, parentLtqs, parentW_g, sequential=True, verbose=False, ellipsoidSize=1.0,
                               mpiInfo=None, nChildNN=-1, kNN=5, mergeDownstream=True, tree=None):
        """This function loops over all candidate pairs between nodes that are children of the root-node, and checks what
        the likelihood becomes when they are merged."""
        if mpiInfo is None:
            mpiInfo = mpi_wrapper.get_mpi_info()
        if mpiInfo.rank == 0:
            if not self.isRoot:
                # Get the mean and precision of this node's posterior when all other node-positions are integrated out
                xrAsIfRoot_g, WAsIfRoot_g = getLtqsAsIfRoot(self.ltqs, self.getW(), self.tParent, parentLtqs, parentW_g)
            else:
                # For the root we do not need calculation, because already all its connecting nodes were integrated over
                xrAsIfRoot_g = self.ltqs
                WAsIfRoot_g = self.getW()
        else:
            xrAsIfRoot_g = None
            WAsIfRoot_g = None

        # Check whether children can be merged on some downstream node. If so, correct ltqs and W of current node for
        # downstream change
        someMergeHappened, xrAsIfRoot_g, WAsIfRoot_g, oldLtqs, oldLtqsVars = self.mergeDownstreamChildren(
            xrAsIfRoot_g, WAsIfRoot_g, sequential=sequential, verbose=verbose, ellipsoidSize=ellipsoidSize,
            nChildNN=nChildNN, kNN=kNN,
            mergeDownstream=mergeDownstream, tree=tree)

        # Check whether we want to merge any of the children of this node
        nChild = len(self.childNodes)
        expNChild = 3 if self.isRoot else 2
        if nChild == expNChild:
            return someMergeHappened, self.ltqs, self.getLtqsVars(), oldLtqs, oldLtqsVars
        # if (bs_glob.getHessStarTreeJax_jit is None) and (not sequential):
        #     bs_glob.getHessStarTreeJax_jit = jax.jit(getHessStarTreeJax)

        if (oldLtqs is None) and (mpiInfo.rank == 0):
            oldLtqs = self.ltqs.copy()
            oldLtqsVars = self.getLtqsVars().copy()
        changedSomething = self.mergeChildrenUB(xrAsIfRoot_g, WAsIfRoot_g, sequential=sequential, verbose=verbose,
                                                ellipsoidSize=ellipsoidSize, nChildNN=nChildNN, kNN=kNN,
                                                mergeDownstream=mergeDownstream, tree=tree)
        if mpiInfo.rank == 0:
            someMergeHappened = (len(self.childNodes) < nChild) or changedSomething or someMergeHappened
        else:
            someMergeHappened = None
        return someMergeHappened, self.ltqs, self.getLtqsVars(), oldLtqs, oldLtqsVars

    # Used
    def mergeChildrenRecursiveSingle(self, parentLtqs, parentW_g, gchildNInd, sequential=True, verbose=False,
                                     ellipsoidSize=1.0, singleProcess=False, random=False, runConfigs_inherited=None):
        """This function loops over all candidate pairs between nodes that are children of the root-node, and checks what
        the likelihood becomes when they are merged."""
        if not self.isRoot:
            # Get the mean and precision of this node's posterior when all other node-positions are integrated out
            xrAsIfRoot_g, WAsIfRoot_g = getLtqsAsIfRoot(self.ltqs, self.getW(), self.tParent, parentLtqs, parentW_g)
        else:
            # For the root we do not have to calculate, because already all its connecting nodes were integrated over
            xrAsIfRoot_g = self.ltqs
            WAsIfRoot_g = self.getW()

        oldLtqs = self.ltqs.copy()
        oldLtqsVars = self.getLtqsVars().copy()

        changedSomething = self.mergeChildrenUB(xrAsIfRoot_g, WAsIfRoot_g, sequential=sequential, verbose=verbose,
                                                random=random,
                                                ellipsoidSize=ellipsoidSize, specialChild=gchildNInd,
                                                singleProcess=singleProcess,
                                                outputFolder=runConfigs_inherited['outputFolder'])
        someMergeHappened = changedSomething
        return someMergeHappened, self.ltqs, self.getLtqsVars(), oldLtqs, oldLtqsVars

    # Used
    def mergeChildrenUB(self, xrAsIfRoot_g, WAsIfRoot_g, sequential=True, verbose=False, ellipsoidSize=None,
                        nChildUB=10, nChildNN=-1, nNewPairsPar=1000, nDonePairsPar=3000, kNN=5, outputFolder=None,
                        tree=None, redoTimeFrac=0.05, redoNNFrac=0.1, specialChild=None, mergeDownstream=True,
                        singleProcess=False, random=False, tmpTreeInd=None, scData=None, n_mem_friendly=500):
        """This function loops over all candidate pairs between nodes that are children of the root-node, and checks
        what the likelihood becomes when they are merged.
        :param singleProcess: This boolean variable determines whether many processes work together to calculate such
        that mergeChildrenUB can again be parallelized, or whether different processes are doing different instances
        of this function.
        """
        functionStart = time.time()
        # Get number of parallel processes + rank of current process. If singleProcess, it means that this function is
        # run from a single process without communication. In that case, we set mpiInfo.rank = 0, mpiInfo.size=1.
        # In general, only process 0 will keep track of the tree, and will report the correct results. The other
        # processes are just employed when necessary, and given the correct information then.
        mpiInfo = mpi_wrapper.get_mpi_info(singleProcess=singleProcess)
        leaderOfMany = (mpiInfo.rank == 0) and (mpiInfo.size > 1)

        # Process the run configurations, and initialize some variables (only on process 0).
        if mpiInfo.rank == 0:
            changedSomething = False
            infoTuple, initVars, initNoneVars = self.initializeMergeGeneral(sequential=sequential,
                                                                            ellipsoidSize=ellipsoidSize,
                                                                            mpiInfo=mpiInfo, specialChild=specialChild,
                                                                            random=random, nChildNN=nChildNN, kNN=kNN,
                                                                            redoTimeFrac=redoTimeFrac,
                                                                            nChildUB=nChildUB, redoNNFrac=redoNNFrac,
                                                                            nNewPairsPar=nNewPairsPar,
                                                                            nDonePairsPar=nDonePairsPar,
                                                                            outputFolder=outputFolder, tree=tree,
                                                                            tmpTreeInd=tmpTreeInd,
                                                                            n_mem_friendly=n_mem_friendly)
            newAnc, del_node_inds, oldRoot, UBInfo, pairs = initNoneVars
            mergeCounter, tree_folder, tmp_tree_ind, breakOut, pairsDoneInfo = initVars
            chInfo, NNInfo, runConfigs, timeOptInfo = infoTuple
        else:
            epsTuple = None
            breakOut = None

        # Start the merge rounds. This loop will only break once breakOut will become True. All processes start this
        # loop, but process 0 will decide whether the other processes are necessary in this round, or if they can send
        # to the next round already.
        while True:
            # We start timing for this round.
            # if verbose:
            # mp_print("Starting this round, the memory usage of this process is ",
            #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
            if mpiInfo.rank == 0:
                if verbose:
                    mp_print("Merge round start. %d children left." % chInfo['nChild'])
                mergeRoundStart = time.time()

            # At this point, we stop all processes except for process 0. These processes will only continue whenever it
            # gets a message to either A) break out of the while-loop, because the function is done, or B) move to the
            # part of the function where parallel computing is required.
            if mpiInfo.rank != 0:
                computeOrBreak = None
                computeOrBreak = mpi_wrapper.bcast(computeOrBreak, root=0)
                # This will be a string, that can be either "break" or "continue", which mean that the process should
                # quit the function or continue to the parallel part, respectively.
                if computeOrBreak == 'break':
                    changedSomething = None
                    break
                elif computeOrBreak != 'continue':
                    mp_print("That's weird. I got this message: ", computeOrBreak, ALL_RANKS=True)

            # We check whether this function is done, stored in "breakOut" on process 0. If so, all processes break out.
            if (mpiInfo.rank == 0) and breakOut:
                if leaderOfMany:
                    computeOrBreak = 'break'
                    mpi_wrapper.bcast(computeOrBreak, root=0)
                    # chInfo['expNChild'] = chInfo['nChild']
                break

            # TODO: Remove this eventually
            # if manualMerges is not None:
            #     if manualMergeCounter < (len(manualMerges) - 1):
            #         manualMergeCounter += 1
            #         manMergeInd1 = manualMerges[manualMergeCounter, 0]
            #         manMergeInd2 = manualMerges[manualMergeCounter, 1]
            #         maxdLogL = (mergeIndToNodeInd[manMergeInd1], mergeIndToNodeInd[manMergeInd2], 1e9)
            #         foundMax = True
            #     else:
            #         manualMerges = None
            #         runConfigs['useUB'] = True
            #         runConfigs['useUBNow'] = False
            #         runConfigs['getNewUB'] = True

            # We should now decide whether all processes are needed in this round, or only process 0. For that, we need
            # to know whether A) a pairwise comparison will be done, instead of a time-optimisation, B) how many pairs
            # will be treated. If the number of pairs is low, it is faster to do it with only one process.
            # The following function finds out what needs to be done in this round (only has to be done by proc. 0).
            if mpiInfo.rank == 0:
                runConfigs, pairInfoTuple, oldRoot, UBInfo, pairsDoneInfo = self.initializeMergeRound(xrAsIfRoot_g,
                                                                                                      WAsIfRoot_g,
                                                                                                      chInfo['nChild'],
                                                                                                      runConfigs,
                                                                                                      mpiInfo, UBInfo,
                                                                                                      oldRoot,
                                                                                                      specialChild=specialChild,
                                                                                                      timeOptInfo=timeOptInfo,
                                                                                                      newAnc=newAnc,
                                                                                                      del_node_inds=del_node_inds,
                                                                                                      verbose=verbose,
                                                                                                      NNInfo=NNInfo,
                                                                                                      oldPairs=pairs,
                                                                                                      pairsDoneInfo=pairsDoneInfo,
                                                                                                      redoNNFrac=redoNNFrac,
                                                                                                      chInfo=chInfo)

                # If periodic time-optimisation is necessary, do this, store new times, then continue to next round.
                if runConfigs['reoptTimesNow']:
                    if verbose:
                        mp_print("Process %d: Doing a periodic re-optimising of branch lengths "
                                 "connected to the root." % mpiInfo.rank, ALL_RANKS=True)
                    runConfigs['redoTimeAt'] = max(int(chInfo['nChild'] * redoTimeFrac), 10)
                    timeOptInfo['timeOptRoundsAgo'] = 0
                    optimisedTimes, WAsIfRoot_g, xrAsIfRoot_g = self.optimiseConnected(mpiInfo, verbose=verbose,
                                                                                       singleProcess=True)
                    newAnc = None
                    changedSomething = True
                    continue

                if not runConfigs['parNow']:  # So we perform a merge in this round, but not in parallel.
                    mpiInfoTmp = mpi_wrapper.get_mpi_info(singleProcess=True)
                else:  # In this case, we do this round in parallel.
                    if verbose:
                        mp_print("This round will be done in parallel! Number of children left: ", chInfo['nChild'])
                    mpiInfoTmp = mpiInfo
                    # The following command makes the other processes run further to below, to receive necessary info
                    computeOrBreak = 'continue'
                    mpi_wrapper.bcast(computeOrBreak, root=0)
                    # start_comm = time.time()
                    runConfigs, tChildren, ltqsChildren, ltqsVarsChildren = self.sendMergeChildrenInfo(runConfigs,
                                                                                                       pairInfoTuple,
                                                                                                       UBInfo,
                                                                                                       chInfo,
                                                                                                       xrAsIfRoot_g,
                                                                                                       WAsIfRoot_g)
            else:  # i.e. if mpiInfo.rank != 0. The process coming here already means this round will be parallel
                # start_comm = time.time()
                mpiInfoTmp = mpiInfo
                infoTuple, tChildren, coordsTuple = self.receiveMergeChildrenInfo()
                pairInfoTuple, UBInfo, runConfigs, chInfo = infoTuple
                xrAsIfRoot_g, WAsIfRoot_g, ltqsChildren, ltqsVarsChildren = coordsTuple
                # CoordsTuple references the mmap'ed file. Will only be closed when all references are gone.
                del coordsTuple

            # if runConfigs['parNow']:
            # mp_print("After communication, the memory usage of this process is ",
            #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
            # mp_print("This communication took %.2f seconds." % (time.time() - start_comm),
            #          ALL_RANKS=True)

            """Every process that reached this point will go through the same procedure below."""
            # We should only pay attention that process 0 does not try to communicate when mpiInfo.size>1 but
            # runConfigs['parNow'] = False.

            # if useUBNow: Distribute tasks over different CPUs such that all nodes move down the sorted list
            # of UBs else: Distribute tasks over different computing cores, such that we maximize the
            # computational benefit of doing pairs with the same first node on the same CPU
            # mp_print("Memory 0 ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ONLY_RANK=1)
            pairs, nPairs, nNewPairs = pairInfoTuple
            myTasks = getMyTaskNumbers(nPairs, mpiInfoTmp.size, mpiInfoTmp.rank, skippingSteps=runConfigs['useUBNow'])
            nTasks = len(myTasks)
            # mp_print("Memory 1 ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ONLY_RANK=1)

            # if (manualMerges is not None) and (manualMergeCounter < (len(manualMerges) - 1)):
            #     nNewPairs = -1
            #     runConfigs['useUBNow'] = True

            # Prepare variable that keeps track of maximal dLogL so far.
            maxdLogL = (-1, -1, 1e-9)
            maxdLogLZeroRoot = (-1, -1, 1e-9)

            # When we do random merges, we keep track of the dLogLs of different merges and then do Gibbs-sampling
            dLogLDict = {} if random else None

            # Initialize boolean that will keep track whether current maximum is higher than next upper bound
            foundMax = False
            # Keep track of current ind1, such that if same ind1 is in next pair, some calculations can be skipped
            oldInd1 = -1
            indTask = -1
            task = -1
            pairsDone = 0
            oldPairsDone = 0
            # mp_print("Memory 2 ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ONLY_RANK=1)
            start_testing_merges = time.time()
            while True:
                if (indTask + 1) == nTasks:  # This was the last task
                    if not runConfigs['useUBNow']:
                        break  # This process is done.
                    foundMax = True  # This process has found its maximum. Communicate this with others.
                # mp_print("Memory 3 ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ONLY_RANK=1)
                # Communicate if maximum was found
                if runConfigs['useUBNow'] and (foundMax or ((indTask + 1 % 100 == 0) and (task >= nNewPairs))):
                    # Communicate maximum when: this process found maximum or 100 tasks have passed, but only after the
                    # newPairs have all been calculated, i.e., the pairs that include the new ancestor
                    foundMax, allFoundMax, maxdLogL, maxdLogLZeroRoot = communicateMaxs(maxdLogL, maxdLogLZeroRoot,
                                                                                        foundMax, task, nNewPairs,
                                                                                        UBInfo, mpiInfoTmp)
                    if allFoundMax:  # When all processes have found the maximum, quit the while-loop
                        break
                    if foundMax:  # When this process found max but others not, immediately return to communication
                        continue

                # Since no maximum was found, go to next task.
                indTask += 1
                skip, task, nodeInd1, nodeInd2, ind1, ind2 = initializeTask(indTask, myTasks, pairs,
                                                                            chInfo['nodeIndToChildInd'], verbose,
                                                                            mpiInfoTmp, runConfigs)
                if skip:
                    continue

                # First check whether next biggest UB is smaller than currently found max
                if runConfigs['useUBNow'] and (task >= nNewPairs) and (maxdLogL[2] > UBInfo['UBs'][task - nNewPairs]):
                    foundMax = True
                    if verbose:
                        mp_print("Process %d found an optimal solution after comparing %d pairs for which UB was known "
                                 "out of %.0f" % (mpiInfoTmp.rank, oldPairsDone, nTasks - nNewPairs / mpiInfoTmp.size),
                                 ALL_RANKS=True)
                    continue

                # Calculate loglikelihood for current pair and (if runConfigs allow) estimate UB
                # Gather all necessary information on the candidate pair.
                # mp_print("Memory 5 ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ONLY_RANK=1)
                if ind1 != oldInd1:  # If child1 is the same as before, we re-use some information.
                    if not runConfigs['parNow']:
                        # If this round is not run in parallel, information needs to be collected from child-objects
                        child2, sortedNodeInds, child1, child1Tuple, rootMinusFirst_ltqs, rootMinusFirst_W_g = self.getChildInfo(
                            xrAsIfRoot_g, WAsIfRoot_g, ind1, ind2, nodeInd1, recalcInd1=True)
                        tOld1, wbar1_g, ltqs1, ltqsVars1, nodeInd1 = child1Tuple
                    else:
                        # If this round is run in parallel, information was already stored in numpy-objects
                        tOld1 = tChildren[ind1]
                        ltqs1 = ltqsChildren[:, ind1]
                        ltqsVars1 = ltqsVarsChildren[:, ind1]
                        wbar1_g = 1 / (tOld1 + ltqsVars1)
                        rootMinusFirst_W_g = WAsIfRoot_g - wbar1_g
                        rootMinusFirst_ltqs = xrAsIfRoot_g * WAsIfRoot_g - wbar1_g * ltqs1
                        # mp_print("Memory 5a ", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ONLY_RANK=1)
                    oldInd1 = ind1
                elif not runConfigs['parNow']:
                    # If this round is not run in parallel, but child1 remained, we get child2 info from child-objects
                    child2, sortedNodeInds = self.getChildInfo(xrAsIfRoot_g, WAsIfRoot_g, ind1, ind2, nodeInd1,
                                                               recalcInd1=False)
                # Finally, we get child2-info in the same way depending on parallelization.
                if runConfigs['parNow']:
                    tOld2 = tChildren[ind2]
                    ltqs2 = ltqsChildren[:, ind2]
                    ltqsVars2 = ltqsVarsChildren[:, ind2]
                    wbar2_g = 1 / (tOld2 + ltqsVars2)
                    sortedNodeInds = tuple(sorted([nodeInd1, nodeInd2]))
                else:
                    tOld2, wbar2_g, ltqs2, ltqsVars2, nodeInd2 = child2.getInfo()
                # Calculate increase in loglikelihood at optimal diff. times for this pair
                dLogLPair, optTimes, ltqs_gi, ltqsVars_gi, wir_gi = calcSingleDLogL(xrAsIfRoot_g, WAsIfRoot_g, ltqs1,
                                                                                    ltqsVars1, wbar1_g, tOld1,
                                                                                    rootMinusFirst_W_g,
                                                                                    rootMinusFirst_ltqs, ltqs2,
                                                                                    ltqsVars2, wbar2_g, tOld2,
                                                                                    sequential=sequential, tol=1e-4,
                                                                                    returnAll=True)
                pairsDone += 1
                # If useUB, make UB-estimate
                if runConfigs['useUB'] and (task < nNewPairs):
                    # TODO: Decide if you want to use old root position for estimating upper bound or not.
                    # Currently I just use the current root-position to determine a new upper bound. This will only go
                    # wrong when the root is moving out of the ellipsoid centered around the current position, before
                    # it will run out of the original ellipsoid. This is very unlikely.
                    justUseNewRoot = True
                    if justUseNewRoot:
                        rootInfo = {"pos": xrAsIfRoot_g, "prec": WAsIfRoot_g, "nChild": chInfo['nChild']}
                        dLogLOld, ddLogLPerEpsx, ddLogLPerEpsW = estimateDerBasedDLogLUB(rootInfo,
                                                                                         optTimes,
                                                                                         ltqs_gi=ltqs_gi,
                                                                                         ltqsVars_gi=ltqsVars_gi,
                                                                                         wir_gi=wir_gi)
                        UBInfo['tuples'][sortedNodeInds] = (dLogLPair, ddLogLPerEpsx, ddLogLPerEpsW)
                    # else:
                    #     if runConfigs['getNewUB']:  # Using this we use the just calculated optTimes
                    #         dLogLOld, ddLogLPerEpsx, ddLogLPerEpsW = estimateDerBasedDLogLUB(oldRoot,
                    #                                                                          optTimes,
                    #                                                                          ltqs_gi=ltqs_gi,
                    #                                                                          ltqsVars_gi=ltqsVars_gi,
                    #                                                                          wir_gi=wir_gi)
                    #         UBInfo['tuples'][sortedNodeInds] = (dLogLPair, ddLogLPerEpsx, ddLogLPerEpsW)
                    #     else:  # This re-optimises the times based on the old-root position
                    #         dLogLOld, ddLogLPerEpsx, ddLogLPerEpsW = estimateDerBasedDLogLUB(oldRoot,
                    #                                                                          optTimes, child1=child1,
                    #                                                                          child2=child2,
                    #                                                                          sequential=sequential)
                    #     UBInfo['tuples'][sortedNodeInds] = (dLogLOld, ddLogLPerEpsx, ddLogLPerEpsW)
                else:
                    oldPairsDone += 1

                # Compare calculated dLogL with current maximum, and replace if larger
                # TODO: Store optTimes for this pair as well. Saves a recompute at the end.
                if random:
                    dLogLDict[sortedNodeInds] = dLogLPair
                if optTimes[-1] > 1e-6:
                    if dLogLPair > maxdLogL[2]:
                        maxdLogL = (sortedNodeInds[0], sortedNodeInds[1], dLogLPair)
                else:
                    if dLogLPair > maxdLogLZeroRoot[2]:
                        maxdLogLZeroRoot = (sortedNodeInds[0], sortedNodeInds[1], dLogLPair)

            # mp_print("Testing the merge pairs took {} seconds.".format(time.time() - start_testing_merges), DEBUG=True)
            start_post_process = time.time()
            """Optimal pair was found. Processing merge starts here."""
            # Process other than proc. 0 are now only here to communicate their gathered information:
            # - maximum pair
            # - gathered upper bound estimates

            # Communicate maximum
            if mpiInfoTmp.size > 1:
                commTuple = (pairsDone, maxdLogL, maxdLogLZeroRoot)
                allCommTuples = mpi_wrapper.gather(commTuple, root=0)
                if mpiInfo.rank == 0:
                    allPairsDone, maxdLogLTuples, maxdLogLTuplesZeroRoot = tuple(zip(*allCommTuples))
                    pairsDoneInfo['totalPairsDone'] = sum(allPairsDone)
                if random:
                    dLogLDictAll = mpi_wrapper.gather(dLogLDict, root=0)
                    if mpiInfoTmp.rank == 0:
                        for dLogLs in dLogLDictAll:
                            dLogLDict.update(dLogLs)
            else:
                pairsDoneInfo['totalPairsDone'] = pairsDone
                maxdLogLTuples = [maxdLogL]
                maxdLogLTuplesZeroRoot = [maxdLogLZeroRoot]

            if runConfigs['parNow']:
                # if verbose:
                #     mp_print("After this round and before deletion, the memory usage of this process is ",
                #              psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)
                # Numpy arrays are no longer needed, and will be recalculated in next round anyhow
                # The variables below are all pointing to the mmap'ed-file that was used for communication. These
                # pointers need to be deleted before the mmap'ed-file can be deleted from memory
                if mpiInfo.rank == 0:
                    empty_folder(runConfigs['mem_friendly_folder'])
                try:
                    del ltqsChildren
                    del ltqsVarsChildren
                    del ltqs2
                    del ltqsVars2
                    del ltqs1
                    del ltqsVars1
                    # The following variables are also re-computed, but do not take much memory. Probably not worth deleting
                    # del tChildren
                    # del wbar1_g
                    # del rootMinusFirst_W_g
                    # del rootMinusFirst_ltqs
                    gc.collect()
                except UnboundLocalError:
                    pass
                # if verbose:
                #     mp_print("After this round and after deletion, the memory usage of this process is ",
                #              psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, " MB.", ALL_RANKS=True)

            # Processing of this information is only done on process 0.
            if mpiInfoTmp.rank == 0:
                # rightList = np.all([self.childNodes[childInd].nodeInd == nodeInd
                #                     for nodeInd, childInd in chInfo['nodeIndToChildInd'].items()])
                # if not rightList:
                #     print("This went wrong, chInfo was %r \n, self.childNodes was %r " % (chInfo, self.childNodes))
                #     raise_exception = 1 / 0
                # Find optimal inds to merge, and calculate optimal times for this merge
                optNodeInd1, optNodeInd2, newTDict, timeOptInfo, dLogLOpt = self.getOptimalDLogLPairInfoNew(
                    xrAsIfRoot_g, WAsIfRoot_g, maxdLogLTuples, maxdLogLTuplesZeroRoot, chInfo['nodeIndToChildInd'],
                    timeOptInfo, sequential=sequential, verbose=verbose, random=random, dLogLDict=dLogLDict)

                # From the optimal information, it can be concluded that a time-optimization is necessary (for example,
                # when the optimal merge has zero distance from root to new ancestor). In this case, we do this here.
                if timeOptInfo['doTimeOpt']:
                    if runConfigs['useNN']:
                        runConfigs['getNewNN'] = True
                    if optNodeInd1 is None:
                        if verbose:
                            mp_print("Re-optimising branch lengths connected to the root.")
                        runConfigs['redoTimeAt'] = max(int(chInfo['nChild'] * redoTimeFrac), 10)
                        timeOptInfo['timeOptRoundsAgo'] = 0
                        optimisedTimes, WAsIfRoot_g, xrAsIfRoot_g = self.optimiseConnected(mpiInfoTmp, verbose=False,
                                                                                           singleProcess=True)
                    else:
                        xrAsIfRoot_g, WAsIfRoot_g = self.resetChildTimes(xrAsIfRoot_g, WAsIfRoot_g, optNodeInd1,
                                                                         optNodeInd2, newTDict,
                                                                         chInfo['nodeIndToChildInd'])
                        optimisedTimes = True
                    newAnc = None
                else:
                    optimisedTimes = False
                changedSomething = True if optimisedTimes else changedSomething

                if (optNodeInd1 is None) and (not optimisedTimes):
                    if runConfigs['useUBNow']:
                        mp_print("No pair could be merged that increased the loglikelihood. Trying again with new"
                                 "UB estimates.")
                        runConfigs['useUBNow'] = False
                        runConfigs['getNewUB'] = True
                        runConfigs['getNewNN'] = True
                        breakOut = True
                    else:
                        if verbose:
                            mp_print("Process " + str(mpiInfoTmp.rank) + ": No pair could be merged that increased"
                                                                         " the loglikelihood, while the root still has " + str(
                                chInfo['nChild']) +
                                     " children. Leaving the tree as it is now.", ALL_RANKS=singleProcess)
                        chInfo['expNChild'] = chInfo['nChild']
                        breakOut = True

                # if runConfigs['timeStamps']:  # Print message on timing
                #     mergeStart = printTiming("Getting optimal node info", processingStart)
                #     storeStart = mergeStart

                if (not optimisedTimes) and (not breakOut):
                    # Merge these nodes by adding new parent of the two nodes as a child to the current node.
                    # We make sure all position information is updated correctly
                    xrAsIfRoot_g, WAsIfRoot_g, newAnc, del_node_inds = self.mergeNodes(xrAsIfRoot_g, WAsIfRoot_g,
                                                                                       optNodeInd1, optNodeInd2,
                                                                                       newTDict,
                                                                                       chInfo['nodeIndToChildInd'],
                                                                                       dLogLOpt, sequential=sequential,
                                                                                       verbose=verbose,
                                                                                       ellipsoidSize=ellipsoidSize,
                                                                                       mergeDownstream=mergeDownstream,
                                                                                       singleProcess=True,
                                                                                       random=random,
                                                                                       runConfigs=runConfigs,
                                                                                       NNInfo=NNInfo)

                    # if manualMerges is not None:
                    #     # Update mergeIndToNodeInd by overwriting mergeInd1 with direction to ancestor
                    #     mergeIndToNodeInd[manMergeInd1] = self.childNodes[-1].nodeInd
                    # TODO: Optimize this by keeping track of it
                    chInfo['nodeIndToChildInd'] = {child.nodeInd: childInd for childInd, child in
                                                   enumerate(self.childNodes)}
                    chInfo['nChild'] -= 1
                    changedSomething = True
                    # if runConfigs['timeStamps']:  # Print message on timing
                    #     storeStart = printTiming("Merging optimal nodes", mergeStart)
                else:
                    newAnc = None
                    del_node_inds = []

                # if runConfigs['timeStamps']:  # Print message on timing
                #     getUBStart = printTiming("Storing W and xr info", storeStart)
                #     newUBStart = getUBStart

                if runConfigs['getNewUB'] and (not breakOut):
                    pairsDoneInfo[
                        'totalPairsDone'] = -1  # In next round, we do not want to be led by pair-count on previous UB-estimates
                    if verbose:
                        mp_print("Estimating ellipsoid size from root movement in last step.")
                    targeted_UB_steps = max(ellipsoidSize * oldRoot['nChild'], 2)
                    epsx = np.linalg.norm((oldRoot['pos'] - xrAsIfRoot_g) * np.sqrt(oldRoot['prec']),
                                          2) * np.sqrt(targeted_UB_steps)
                    epsW = np.linalg.norm((oldRoot['prec'] - WAsIfRoot_g) / oldRoot['prec'],
                                          2) * targeted_UB_steps
                    # if runConfigs['timeStamps']:  # Print message on timing
                    #     getUBStart = printTiming("Estimating new ellipsoid size", newUBStart)

            # If it was above decided that this round can be stopped, all processes can continue to next round here.
            if mpiInfoTmp.size > 1:
                breakOut = mpi_wrapper.bcast(breakOut, root=0)
            if breakOut:
                continue
            # If no UBs are used, processes other than 0 can continue
            if (mpiInfoTmp.rank > 0) and (not runConfigs['useUB']):
                continue

            # Finally, we use the calculated epsx and epsW to calculate the new upper bounds and add them into the list
            if runConfigs['useUB']:
                if mpiInfoTmp.size > 1:
                    if mpiInfoTmp.rank == 0:
                        epsTuple = (epsx, epsW)
                    epsx, epsW = mpi_wrapper.bcast(epsTuple, root=0)
                # Get upper bounds by combining precalculated information with the chosen epsx and epsW
                if len(UBInfo['tuples']):
                    pairKeys, dLogLUBInfo = zip(*UBInfo['tuples'].items())
                    dLogLDictUB = dict(zip(pairKeys, np.matmul(np.array(dLogLUBInfo), np.array([1, epsx, epsW]))))
                else:
                    dLogLDictUB = {}

                # if runConfigs['timeStamps']:  # Print message on timing
                #     sortingStart = printTiming("Getting upper bounds into dict", getUBStart)

                # Sorting new upper bounds calculated on this process
                if len(dLogLDictUB):
                    pairsUB, dLogLUBs = zip(*sorted(dLogLDictUB.items(), key=lambda x: x[1], reverse=True))
                else:
                    pairsUB = ()
                    dLogLUBs = ()

                # if runConfigs['timeStamps']:  # Print message on timing
                #     communicationStart = printTiming("Sorting upper bounds", sortingStart)

                # Gather the computed upper bounds from all processes
                if mpiInfoTmp.size > 1:  # (not singleProcess) and parallelInfo['parNow']:
                    allPairsUB = mpi_wrapper.gather(pairsUB, root=0)
                    alldLogLsUB = mpi_wrapper.gather(dLogLUBs, root=0)
                    if mpiInfoTmp.rank != 0:
                        # The memory where UBs are stored can be freed up on processes other than zero, because
                        # probably they'll be idle for quite some rounds.
                        try:
                            del pairsUB
                            del dLogLUBs
                            del dLogLDictUB
                            del UBInfo
                            del pairKeys
                            del dLogLUBInfo
                            del myTasks
                            del pairs
                            gc.collect()
                        except UnboundLocalError:
                            pass
                else:
                    allPairsUB = [pairsUB]
                    alldLogLsUB = [dLogLUBs]

                # if runConfigs['timeStamps']:  # Print message on timing
                #     insertStart = printTiming("Communicating UB info", communicationStart)

                # Processes other than 0 have communicated everything. Can proceed to next round.
                if mpiInfoTmp.rank == 0:
                    UBInfo, runConfigs['getNewUB'], ellipsoidSize = getNewUBInfo(xrAsIfRoot_g, WAsIfRoot_g, epsx, epsW,
                                                                                 alldLogLsUB, UBInfo, allPairsUB,
                                                                                 oldRoot, del_node_inds,
                                                                                 nChild=chInfo['nChild'], kNN=kNN,
                                                                                 pairsDoneInfo=pairsDoneInfo,
                                                                                 ellipsoidSize=ellipsoidSize,
                                                                                 verbose=verbose,
                                                                                 newUBObtained=runConfigs[
                                                                                     'obtainedNewPairs'],
                                                                                 mpiInfo=mpiInfo)

                # if runConfigs['timeStamps']:  # Print message on timing
                #  ubCommunicationStart = printTiming("Inserting new UBs and testing if out of ellipsoid", insertStart)

                # if not singleProcess:
                #     bcastUBTuple = mpi_wrapper.bcast(bcastUBTuple, root=0)
                # UBInfo, runConfigs['getNewUB'] = bcastUBTuple

                # if runConfigs['timeStamps']:  # Print message on timing
                #     printTiming("Communicating sorted upper bound lists", ubCommunicationStart)

            if mpiInfo.rank == 0:
                # TODO: Remove eventually, only uncomment this for making animations
                # if (mpiInfo.rank == 0) and bs_glob.nwk_counter and (tree is not None):
                #     tree.to_newick(use_ids=True,
                #                    results_path=os.path.join(bs_glob.nwk_folder,
                #                                              'tree_{}.nwk'.format(bs_glob.nwk_counter)))
                #     bs_glob.nwk_counter += 1

                breakOut = breakOut or (chInfo['nChild'] <= chInfo['expNChild'])
                if verbose:
                    print_text = ", %d pairs were compared using %d processes." % (
                        pairsDoneInfo['totalPairsDone'], mpiInfoTmp.size) \
                        if pairsDoneInfo['totalPairsDone'] >= 0 else '.'
                    mp_print("This round took %.2f seconds%s" % (time.time() - mergeRoundStart, print_text))
                if chInfo['nChild'] % 100 == 0:
                    mp_print("Now %d children left. Function runtime now %.5f seconds." %
                             (chInfo['nChild'], time.time() - functionStart))
                mergeCounter += 1
                if (mergeCounter % 1000 == 0) and (tree is not None) and (tree_folder is not None):
                    scData.storeTreeInFolder(os.path.join(tree_folder, 'greedy_tree_%d' % tmp_tree_ind),
                                             with_coords=False, verbose=verbose, cleanup_tree=False)
                    remove_tree_folders(tree_folder, removeDir=False, notRemove=tmp_tree_ind, base='greedy')
                    tmp_tree_ind += 1
            # mp_print("Post-processing in this round took: {} seconds.".format(time.time() - start_post_process),
            #          DEBUG=True)

        if (mpiInfo.rank == 0) and (tree_folder is not None):
            # scData.storeTreeInFolder(os.path.join(tree_folder, 'greedy_tree_%d' % 1e5), with_coords=False,
            #                          verbose=verbose, nwk=False)
            remove_tree_folders(tree_folder, removeDir=True, base='greedy')
        if (mpiInfo.rank == 0) and (runConfigs['mem_friendly_folder'] is not None):
            remove_folder(runConfigs['mem_friendly_folder'])
        return changedSomething

    def getNewPairs(self, xrAsIfRoot_g, runConfigs, NNInfo=None, newAnc=None, verbose=False,
                    UBInfo=None, specialChild=None, oldPairs=None, chInfo=None, del_node_inds=[]):
        if not runConfigs['useUBNow']:
            # This means we need to recalculate UBs for all pairs
            if not runConfigs['useNN']:
                # This means we need to get all pairs, not only NNs. However, only if we don't have a special node:
                if specialChild is None:
                    new_pairs = list(itertools.combinations([child.nodeInd for child in self.childNodes], 2))
                else:
                    new_pairs = [(specialChild, child.nodeInd) for child in self.childNodes if
                                 child.nodeInd != specialChild]
            else:
                # This means we will use nearest-neighbours
                if runConfigs['getNewNN']:
                    # In this case we calculate new NNs
                    new_pairs, NNInfo = self.getNNPairs(xrAsIfRoot_g, NNInfo, runConfigs['kNN'], verbose=verbose)
                    runConfigs['getNewNN'] = False
                    NNInfo['NNcounter'] = 0
                else:
                    oldPairs_array = np.array(oldPairs)
                    rows_to_be_del = np.where(np.isin(oldPairs_array, del_node_inds).any(axis=1))[0]
                    oldPairs = list(map(tuple, np.delete(oldPairs_array, rows_to_be_del, axis=0)))
                    if newAnc is None:
                        # When no new ancestor was added (e.g. when optimised times instead of merge) -> old UB-pairs
                        new_pairs = oldPairs
                    else:
                        # In this case we should add NN-pairs involving the ancestor
                        new_pairs, oldPairsList = self.get_new_nn_pairs(newAnc, NNInfo, runConfigs, UBInfo=UBInfo,
                                                                        old_pairs_list=[oldPairs])
                        oldPairs = oldPairsList[0]
                        # index, nns = getApproxNNs(newAnc.ltqs[:, None], index=NNInfo['index'],
                        #                           k=2 * runConfigs['kNN'],
                        #                           pointsIds=[newAnc.nodeInd], addPoints=False)
                        # nns = np.array(index.IDs)[nns]
                        # pairs = [(newAnc.nodeInd, NNInfo['leafToChild'][nb]) for nb in nns[0, :] if
                        #          NNInfo['leafToChild'][nb] != newAnc.nodeInd]
                        # if len(pairs) > 0:
                        #     # Update the connectivity matrix for the nearest neighbors with these new pairs
                        #     pairs_array = np.array(pairs)
                        #     NNInfo['conn_mat'][pairs_array[:, 0], pairs_array[:, 1]] = True
                        #     NNInfo['conn_mat'][pairs_array[:, 1], pairs_array[:, 0]] = True
                        #
                        # pairs = [(newAnc.nodeInd, nb_nodeInd) for nb_nodeInd in
                        #          NNInfo['conn_mat'][[newAnc.nodeInd], :].nonzero()[1]]
                        # pairs += list(map(tuple, UBInfo['pairs']))

                        # Then we add the old-NN-pairs, but we sort them efficiently
                        # for del_node_ind in del_node_inds:
                        #     oldPairs = [old_pair for old_pair in oldPairs if del_node_ind not in old_pair]
                        new_pairs += oldPairs
                        new_pairs = sorted(new_pairs)
            # No UBs are used, so all pairs are new
            old_pairs = []
        else:
            # We will use our pre-calculated UBs here. The UB-pairs are ordered descending in dLogLUB, in UBInfo
            if newAnc is None:
                if not runConfigs['getNewNN']:
                    # When no new ancestor was added (e.g. optimised times instead of merge) -> sorted UB-pairs
                    old_pairs = list(map(tuple, UBInfo['pairs']))
                    new_pairs = []
                else:
                    # so we re-use UBs, recalculate NNs
                    # First get all NNs
                    new_pairs, NNInfo = self.getNNPairs(xrAsIfRoot_g, NNInfo, runConfigs['kNN'], verbose=verbose)
                    runConfigs['getNewNN'] = False
                    NNInfo['NNcounter'] = 0
                    # Then we test which UBs were already in the UB-pairs and which ones are new
                    existingUBPairs = []
                    existingUBs = []
                    for ind, pair in enumerate(map(tuple, np.array(UBInfo['pairs']))):
                        try:
                            newInd = new_pairs.index(pair)
                            existingUBPairs.append(pair)
                            existingUBs.append(UBInfo['UBs'][ind])
                            del new_pairs[newInd]
                        except ValueError:
                            # This pair is no longer in the NN-list
                            pass
                    if len(existingUBPairs) == 0:
                        UBInfo['pairs'] = np.zeros((0, 2))
                    else:
                        UBInfo['pairs'] = np.array(existingUBPairs)
                    UBInfo['UBs'] = np.array(existingUBs)
                    # nNewPairs = len(pairs)
                    old_pairs = list(map(tuple, UBInfo['pairs']))
            else:
                # We need to add new pairs (e.g. with new ancestor) first, then the sorted UB-pairs
                if not runConfigs['useNN']:
                    # In this case we just add all new pairs with the new ancestor
                    new_pairs = [(newAnc.nodeInd, child.nodeInd) for child in self.childNodes if
                                 newAnc.nodeInd != child.nodeInd]
                    old_pairs = [*UBInfo['pairs']]
                elif not runConfigs['getNewNN']:
                    new_pairs = self.get_new_nn_pairs(newAnc, NNInfo, runConfigs, UBInfo=UBInfo)
                    old_pairs = list(map(tuple, UBInfo['pairs']))
                else:
                    # so we re-use UBs, recalculate NNs
                    # First get all NNs
                    new_pairs, NNInfo = self.getNNPairs(xrAsIfRoot_g, NNInfo, runConfigs['kNN'], verbose=verbose)
                    runConfigs['getNewNN'] = False
                    NNInfo['NNcounter'] = 0

                    # I here put all new NN-pairs on top, and only after it the ones for which I know a UB already.
                    # Pairs may be listed that are not NN anymore, but these will drop out at next UB calculation round
                    # start = time.time()
                    # pairs = list(set(pairs) - set(map(tuple, np.array(UBInfo['pairs']))))
                    # nNewPairs = len(pairs)
                    # pairs += list(map(tuple, UBInfo['pairs']))
                    # nPairs = len(pairs)
                    # if verbose:
                    #     mp_print(
                    #         "Finding nearest neighbour pairs for which UB is already known with setdiff took %f seconds." % (
                    #                 time.time() - start))

                    start_keeping_known_UBs = time.time()
                    # Then we test which UBs were already in the UB-pairs and which ones are new
                    existingUBPairs = []
                    existingUBs = []
                    for ind, pair in enumerate(map(tuple, np.array(UBInfo['pairs']))):
                        try:
                            newInd = new_pairs.index(pair)
                            existingUBPairs.append(pair)
                            existingUBs.append(UBInfo['UBs'][ind])
                            del new_pairs[newInd]
                        except ValueError:
                            # This pair is no longer in the NN-list
                            pass
                    if len(existingUBPairs) == 0:
                        UBInfo['pairs'] = np.zeros((0, 2))
                    else:
                        UBInfo['pairs'] = np.array(existingUBPairs)
                    UBInfo['UBs'] = np.array(existingUBs)
                    if verbose:
                        mp_print("Finding nearest neighbour pairs for which UB is already known took %f seconds." % (
                                time.time() - start_keeping_known_UBs), DEBUG=True)
                    old_pairs = list(map(tuple, UBInfo['pairs']))

        if runConfigs['useNN']:
            # Test whether there are enough nearest neighbours for each node. Otherwise, get new NNs next round
            unq_node_inds, counts = np.unique(np.array(old_pairs + new_pairs), return_counts=True)
            # TODO: Check if we cannot just keep track of these counts
            few_nn_inds = unq_node_inds[
                np.where(counts <= 0.2 * runConfigs['kNN'])[0]]  # TODO: Only do those problematic ones
            for few_nn_node_ind in few_nn_inds:
                if few_nn_node_ind in chInfo['nodeIndToChildInd']:  # Check if this node is still a child
                    mp_print("NN: Node {} has few nns, recalculating some new NNs for it.".format(few_nn_node_ind),
                             DEBUG=True)
                    child_ind = chInfo['nodeIndToChildInd'][few_nn_node_ind]
                    new_nn_pairs, pairs_list = self.get_new_nn_pairs(self.childNodes[child_ind], NNInfo, runConfigs,
                                                                     UBInfo=UBInfo,
                                                                     old_pairs_list=[new_pairs, old_pairs])
                    new_pairs = new_nn_pairs + pairs_list[0]
                    old_pairs = pairs_list[1]
                    if len(new_nn_pairs) <= 0.4 * runConfigs['kNN']:
                        mp_print("NN: Finding new NN-pairs did not succeed. Updating index for calculating NNs.")
                        new_nn_pairs, pairs_list = self.get_new_nn_pairs(self.childNodes[child_ind], NNInfo, runConfigs,
                                                                         old_pairs_list=[new_pairs],
                                                                         update_nn_index=True,
                                                                         xrAIRoot=xrAsIfRoot_g)
                        new_pairs = new_nn_pairs + pairs_list[0]

                else:
                    mp_print("NN: Somehow, node {} occurs in a pair, "
                             "but it is not a child of the root anymore. "
                             "Check this at some point.".format(few_nn_node_ind), DEBUG=True)

        # if newAnc is not None:
        #     # When using UBs, we should start by computing all dLogLs of the new ancestor with the rest:
        #     # When getNewNN, calculate new NN-pairs, then select those without UB and add them on the top as new pairs
        #     if boolArgs['useNN']:
        #         if not boolArgs['getNewNN']:
        #             NNInfo['NNcounter'] += 1
        #             for deletedInd in del_node_inds:  # Map all nb-connections of merged leafs to their ancestor
        #                 if deletedInd in NNInfo['leafToChild']:
        #                     NNInfo['leafToChild'][deletedInd] = newAnc.nodeInd
        #                 else:
        #                     downstreamLeafs = [leaf for leaf, child in NNInfo['leafToChild'].items() if
        #                                        child == deletedInd]
        #                     for leaf in downstreamLeafs:
        #                         NNInfo['leafToChild'][leaf] = newAnc.nodeInd
        #
        #             # Get nearest neighbours of new ancestor. Take twice as many neighbours as for the other nodes,
        #             # to compensate for no other nodes adding connections to ancestor, and for merged nodes being part
        #             # of ancestor-neighbours
        #             index, nns = getApproxNNs(newAnc.ltqs[:, None], index=NNInfo['index'], k=2 * NNInfo['kNN'],
        #                                       pointsIds=[newAnc.nodeInd], addPoints=False)
        #             nns = np.array(index.IDs)[nns]
        #             pairs = [sorted((newAnc.nodeInd, NNInfo['leafToChild'][nb])) for nb in nns[0, :] if
        #                      NNInfo['leafToChild'][nb] != newAnc.nodeInd]
        #             # Sort pairs to make computation faster
        #             pairs = sorted(pairs)
        #         else:
        #             pairs, NNInfo = self.getNNPairs(xrAsIfRoot_g, NNInfo, verbose=verbose)
        #             boolArgs['getNewNN'] = False
        #             NNInfo['NNcounter'] = 0
        #             # TODO: Take these new pairs and select the pairs for which UB exists, put them in UB['pairs'] with
        #             #  corresponding UB['dLogLs']. New pairs go into pairs, which will be treated first.
        #     else:
        #         # If no NNs are used, just pair the new ancestor to all existing root-children
        #         pairs = [(newAnc.nodeInd, child.nodeInd) for child in self.childNodes[:-1]]
        # else:
        #     # In this case no new ancestor was added because only the times were re-optimised
        #     pairs = []
        #
        # if not boolArgs['useUBNow']:
        #     if boolArgs['useNN'] and boolArgs['getNewNN']:
        #         pairs, NNInfo = self.getNNPairs(xrAsIfRoot_g, NNInfo, verbose=verbose)
        #         boolArgs['getNewNN'] = False
        #         NNInfo['NNcounter'] = 0
        #     # Create a list of all pairs, over which we will loop
        #     elif specialChild is None:
        #         pairs = list(itertools.combinations([child.nodeInd for child in self.childNodes], 2))
        #     # Since we are not using upper bounds. All pairs are new.
        #     nNewPairs = len(pairs)
        # else:
        #     # "pairs" currently contains all pairs with the newly created ancestor. These should be done first.
        #     nNewPairs = len(pairs)
        #     # Append list of ind-pairs ordered descending in dLogLUB. These will be done until dLogLMax>dLogLUB
        #     pairs += [*UBInfo['pairs']]
        #     if boolArgs['useNN']:
        #         # Test whether there are enough nearest neighbours for each node. Otherwise, get new NNs next round
        #         _, counts = np.unique(np.array(pairs), return_counts=True)
        #         boolArgs['getNewNN'] = np.any(counts < 0.2 * NNInfo['kNN'])
        #
        # nPairs = len(pairs)
        nNewPairs = len(new_pairs)
        pairs = new_pairs + old_pairs
        nPairs = len(pairs)
        return pairs, nPairs, nNewPairs

    def initializeMergeRound(self, xrAsIfRoot_g, WAsIfRoot_g, nChild, runConfigs, mpiInfo, UBInfo, oldRoot,
                             timeOptInfo=None, specialChild=None, newAnc=None, del_node_inds=[], verbose=True,
                             NNInfo=None, oldPairs=None, redoNNFrac=0.1, pairsDoneInfo=None, chInfo=None):
        # Determine if the use of UBs and NNs should be stopped because too few children are left
        if nChild <= runConfigs['nChildUB']:
            # If too few children are left we do not use upper bound calculation anymore since it is not helping
            runConfigs['useUB'] = False
            runConfigs['getNewUB'] = False
        if nChild <= runConfigs['nChildNN']:
            # The same holds for using nearest neighbors. It is only necessary for large numbers of children.
            runConfigs['useNN'] = False
        if nChild <= runConfigs['n_mem_friendly']:
            # Under a certain number of children, it is faster to communicate ltqs directly using mpi,
            # not via memmapped file
            runConfigs['mem_friendly'] = False
        # Initialize a dictionary to store pairs and UBs sorted by UB
        UBInfo = {'pairs': np.zeros((0, 2), dtype=int), 'UBs': np.zeros(0)} if (
                runConfigs['getNewUB'] or (not runConfigs['useUB'])) else UBInfo
        # Make new dict to store newly gathered upper bound information (for the pairs including new ancestor)
        UBInfo['tuples'] = {}

        # Determine whether we will use the upper bound information in this round
        if runConfigs['getNewUB'] or (not runConfigs['useUB']):
            # In this round we do not use upper bound information
            runConfigs['useUBNow'] = False
            pairsDoneInfo['pairsDoneList'] = []
            if verbose:
                mp_print("\nStarting new merge round. %s" % (
                    "Estimating new upper bounds." if runConfigs['useUB'] else "Not using upper bound estimation."),
                         ALL_RANKS=(mpiInfo.size == 1))
            if runConfigs['useUB']:
                # In this case, this round will calculate new upper bounds. Therefore we store the current
                # root-information, such that we can notice when the root moves out of the allowed ellipsoid
                oldPos = xrAsIfRoot_g.copy()
                oldPrec = WAsIfRoot_g.copy()
                oldNChild = nChild
                oldRoot = {'pos': oldPos, 'prec': oldPrec, 'nChild': oldNChild}
            else:
                # In this case we do not use upper bounds ever, so no need to store root information.
                oldRoot = None
        else:
            if verbose:
                mp_print("\nStarting new merge round. Using calculated upper bounds.")
            runConfigs['useUBNow'] = True

        # We periodically reoptimize the edge lengths, they may have become sub-optimal due to merges.
        runConfigs['reoptTimesNow'] = (timeOptInfo['timeOptRoundsAgo'] > runConfigs['redoTimeAt']) and (
            not runConfigs['useUBNow']) and self.isRoot

        # TODO: Check whether getting new pairs is handled correctly when time optimization is done.
        # Update NN information when new ancestor was added
        NNInfo = updateNNInfo(runConfigs, NNInfo, del_node_inds, newAnc)

        # Given the information we now have on whether to calc. new UBs, and whether to use NNs. We can now get a list
        # of pairs, in the order that we want to consider them.
        runConfigs['obtainedNewPairs'] = runConfigs['getNewUB'] or runConfigs['getNewNN']
        start_new_pairs = time.time()
        pairs, nPairs, nNewPairs = self.getNewPairs(xrAsIfRoot_g, runConfigs, NNInfo=NNInfo,
                                                    newAnc=newAnc, verbose=verbose, oldPairs=oldPairs,
                                                    UBInfo=UBInfo, specialChild=specialChild, chInfo=chInfo,
                                                    del_node_inds=del_node_inds)
        if verbose:
            mp_print("Finding new pairs took: {} seconds.".format(time.time() - start_new_pairs), DEBUG=True)
        pairInfoTuple = (pairs, nPairs, nNewPairs)
        runConfigs['parNow'] = (mpiInfo.size > 1) and (
                (nNewPairs > runConfigs['nNewPairsPar']) or (
                pairsDoneInfo['totalPairsDone'] > runConfigs['nDonePairsPar']))

        if runConfigs['useNN'] and (NNInfo['NNcounter'] > runConfigs['redoNNAt']):
            if verbose:
                mp_print("Will do a periodic recalculation of nearest neighbours at next round.")
            runConfigs['redoNNAt'] = max(int(nChild * redoNNFrac), 10)
            runConfigs['getNewNN'] = True

        # if nChild == parallelInfo['nextTry']:
        #     parallelInfo['parTry'] = True
        #     parallelInfo['timingCheck'] = 3
        #     parallelInfo['par'] = True
        #     parallelInfo['parNow'] = True
        # # Determine whether we will distribute the tasks over different CPUs or not (communication may be too costly)
        # if parallelInfo['parTry']:
        #     if mpiInfo.rank == 0:
        #         if parallelInfo['timingCheck'] == 0:
        #             if parallelInfo['par']:
        #                 parallelInfo['parTiming'] = np.mean(parallelInfo['timingList'][-2:])
        #                 if parallelInfo['nonparTiming'] is None:  # Check how long non-parallel takes
        #                     parallelInfo['par'] = False
        #                     parallelInfo['timingCheck'] = 3
        #                 else:
        #                     parallelInfo['par'] = parallelInfo['parTiming'] < parallelInfo['nonparTiming']
        #                     mp_print("Time for parallel computing was estimated at %f seconds.\n"
        #                              "Time for non-parallel was estimated at %f seconds." %
        #                              (parallelInfo['parTiming'], parallelInfo['nonparTiming']))
        #                     mp_print("The past 4 merge rounds show that it is faster to run the code %s parallelization." % (
        #                         "with" if parallelInfo['par'] else "without"), ALL_RANKS=True)
        #                     parallelInfo['parTiming'] = None
        #                     parallelInfo['nonparTiming'] = None
        #                     parallelInfo['parTry'] = False
        #                     parallelInfo['nextTry'] = max(int(nChild / 10), 50)
        #             else:
        #                 parallelInfo['nonparTiming'] = np.mean(parallelInfo['timingList'][-2:])
        #                 if parallelInfo['parTiming'] is None:  # Check how long non-parallel takes
        #                     parallelInfo['par'] = True
        #                     parallelInfo['timingCheck'] = 3
        #                 else:
        #                     parallelInfo['par'] = parallelInfo['parTiming'] < parallelInfo['nonparTiming']
        #                     mp_print("Time for parallel computing was estimated at %f seconds.\n"
        #                              "Time for non-parallel was estimated at %f seconds." %
        #                              (parallelInfo['parTiming'], parallelInfo['nonparTiming']))
        #                     mp_print("The past 4 merge rounds show that it is faster to run the code %s parallelization." % (
        #                         "with" if parallelInfo['par'] else "without"), ALL_RANKS=True)
        #                     parallelInfo['parTiming'] = None
        #                     parallelInfo['nonparTiming'] = None
        #                     parallelInfo['parTry'] = False
        #                     parallelInfo['nextTry'] = max(int(nChild / 10), 50)
        #
        #         parallelInfo['parNow'] = (parallelInfo['par'] or (nNewPairs > 1000))  # and (not nNewPairs < 100)
        #
        #     parallelInfo = mpi_wrapper.bcast(parallelInfo, root=0)

        return runConfigs, pairInfoTuple, oldRoot, UBInfo, pairsDoneInfo

    def sendMergeChildrenInfo(self, runConfigs, pairInfoTuple, UBInfo, chInfo, xrAsIfRoot_g, WAsIfRoot_g):
        # Gather all information that should be sent into a tuple containing generic Python-objects, and
        # into some numpy-arrays that will be sent separately (because these can be communicated faster).
        # mp_print("Before communication, memory usage on this process is %.2f MB."
        #          % (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2),
        #          ALL_RANKS=True)
        if runConfigs['mem_friendly']:
            empty_folder(runConfigs['mem_friendly_folder'])
            runConfigs['mem_friendly_files'] = os.path.join(runConfigs['mem_friendly_folder'],
                                                            'data%d' % np.random.randint(1e6))
        infoTuple = (pairInfoTuple, UBInfo, runConfigs, chInfo)
        mpi_wrapper.bcast(infoTuple, root=0)

        ltqsChildren, ltqsVarsChildren, tChildren = self.getInfoChildren()
        if runConfigs['mem_friendly']:
            # This process will store the necessary arrays in binary files using the numpy.save-function.
            # These arrays can then be read using memmap, such that only those parts have to be loaded into
            # memory that are actually required
            files_sent = False
            tries = 0
            while (not files_sent) and (tries <= 10):
                np.save(runConfigs['mem_friendly_files'] + '_ltqs.npy', ltqsChildren, allow_pickle=False)
                np.save(runConfigs['mem_friendly_files'] + '_ltqsVars.npy', ltqsVarsChildren,
                        allow_pickle=False)
                try:
                    ltqsChildren = np.load(runConfigs['mem_friendly_files'] + '_ltqs.npy', mmap_mode='r',
                                           allow_pickle=False)
                    ltqsVarsChildren = np.load(runConfigs['mem_friendly_files'] + '_ltqsVars.npy', mmap_mode='r',
                                               allow_pickle=False)
                    files_sent = True
                    mpi_wrapper.bcast(files_sent, root=0)
                except FileNotFoundError:
                    tries += 1

            if not files_sent:
                mpi_wrapper.bcast(files_sent, root=0)
                mp_print("Trying to communicate data: ltqsChildren with shape {}, and ltqsVarsChildren with"
                         " shape {}, at filepaths: {}, "
                         "but it does not work.".format(ltqsChildren.shape, ltqsVarsChildren.shape,
                                                        runConfigs['mem_friendly_files'] + '_ltqs.npy'),
                         ERROR=True)
                mp_print("Resorting to a less memory-friendly way of communicating.")
            else:
                # Send some smaller numpy arrays (concerning coordinates of root)
                # This sending also functions as a start signal for the other process to start loading the .npy-files
                rootInfo = np.vstack((xrAsIfRoot_g, WAsIfRoot_g))
                mpi_wrapper.Bcast(rootInfo, root=0, type='double')
                mpi_wrapper.Bcast(tChildren, root=0, type='double')
                del rootInfo
                gc.collect()

        if not runConfigs['mem_friendly'] or (not files_sent):
            # The following communication should be faster, but requires the whole numpy arrays to be loaded
            # into memory, instead of loading only what's necessary using the memmap
            # We will communicate 2 numpy arrays with the following layout:
            # ltqsInfo = [ltqsChildren, ltqsAIRoot, ltqsVarsChildren, WAIRoot]
            # and
            # tChildren = [tChildren]
            ltqsInfo = np.concatenate(
                (ltqsChildren, xrAsIfRoot_g[:, None], ltqsVarsChildren, WAsIfRoot_g[:, None]), axis=1)
            mpi_wrapper.Bcast(ltqsInfo, root=0, type='double')
            del ltqsInfo
            gc.collect()
            mpi_wrapper.Bcast(tChildren, root=0, type='double')
        mpi_wrapper.barrier()
        return runConfigs, tChildren, ltqsChildren, ltqsVarsChildren

    def receiveMergeChildrenInfo(self):
        # mp_print("Before communication, memory usage on this process is %.2f MB."
        #          % (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2),
        #          ALL_RANKS=True)
        infoTuple = None
        infoTuple = mpi_wrapper.bcast(infoTuple, root=0)
        pairInfoTuple, UBInfo, runConfigs, chInfo = infoTuple
        nChild = chInfo['nChild']
        files_sent = False
        if runConfigs['mem_friendly']:
            # Receive root coordinates
            files_sent = mpi_wrapper.bcast(files_sent, root=0)
            if files_sent:
                rootInfo = np.empty((2, bs_glob.nGenes))
                tChildren = np.empty(nChild)
                mpi_wrapper.Bcast(rootInfo, root=0, type='double')
                mpi_wrapper.Bcast(tChildren, root=0, type='double')
                xrAsIfRoot_g = rootInfo[0, :]
                WAsIfRoot_g = rootInfo[1, :]
                ltqsChildren = np.load(runConfigs['mem_friendly_files'] + '_ltqs.npy', mmap_mode='r',
                                       allow_pickle=False)
                ltqsVarsChildren = np.load(runConfigs['mem_friendly_files'] + '_ltqsVars.npy', mmap_mode='r',
                                           allow_pickle=False)
                del rootInfo
                gc.collect()

        if not runConfigs['mem_friendly'] or (not files_sent):
            # Prepare numpy arrays to be obtained.
            ltqsInfo = np.empty((bs_glob.nGenes, 2 * nChild + 2))
            tChildren = np.empty(nChild)
            # Receive numpy arrays
            mpi_wrapper.Bcast(ltqsInfo, root=0, type='double')
            mpi_wrapper.Bcast(tChildren, root=0, type='double')

            # Compile child information
            ltqsChildren, xrAsIfRoot_g, ltqsVarsChildren, WAsIfRoot_g = \
                np.array_split(ltqsInfo, indices_or_sections=[nChild, nChild + 1, 2 * nChild + 1], axis=1)
            del ltqsInfo
            gc.collect()
            xrAsIfRoot_g = xrAsIfRoot_g.flatten()
            WAsIfRoot_g = WAsIfRoot_g.flatten()

        coordsTuple = (xrAsIfRoot_g, WAsIfRoot_g, ltqsChildren, ltqsVarsChildren)
        mpi_wrapper.barrier()
        return infoTuple, tChildren, coordsTuple

    def getNNPairs(self, xrAIRoot, NNInfo, kNN, verbose=False):
        start = time.time()
        # We first gather all ltq-information about the children
        ltqsCh, _, _ = self.getInfoChildren()

        # We center the ltq-information around the root
        ltqsCh -= xrAIRoot[:, None]
        NNInfo['subtracted_mean'] = xrAIRoot
        # We ask for the nearest neighbours
        nodeInds = [child.nodeInd for child in self.childNodes]
        # TODO: Check how to get approxNNs beyond the brute-force sklearn one
        # Instead of taking cosine metric, we can also normalize the vectors and use squared-euclidean
        norms = np.linalg.norm(ltqsCh, axis=0)
        np.divide(ltqsCh, norms, out=ltqsCh)
        index, nns = getApproxNNs(ltqsCh, index=None, k=kNN + 1, n_bits_factor=100, metric='sqeuclidean',
                                  pointsIds=nodeInds, addPoints=True, th1=1e9, th2=1e9)
        nns = np.array(nodeInds)[nns]
        # Create unique set of all pairs with at least one nn-connection
        pairs = set()
        for row, nodeInd in enumerate(nodeInds):
            pairs.update(
                [tuple(sorted(pair)) for pair in zip([nodeInd] * (kNN + 1), nns[row, :]) if pair[0] != pair[1]])
        pairs = sorted(pairs)
        # We store the nearest neighbor-information in a connectivity matrix
        # TODO: Clean this up, check if it can be done more efficient
        pairs_array = np.array(pairs)
        # Note that we make the sparse matrix dimensions twice the number of cells. In this way we can store the
        # adjacency between nodes even when we have the maximal number of nodes (in a fully binary tree).
        connectivity_mat = csr_array((np.ones(pairs_array.shape[0]), (pairs_array[:, 0], pairs_array[:, 1])),
                                     shape=(2 * bs_glob.nCells, 2 * bs_glob.nCells), dtype=bool)
        connectivity_mat_T = csr_array((np.ones(pairs_array.shape[0]), (pairs_array[:, 1], pairs_array[:, 0])),
                                       shape=(2 * bs_glob.nCells, 2 * bs_glob.nCells), dtype=bool)
        connectivity_mat = connectivity_mat + connectivity_mat_T
        connectivity_mat = lil_array(connectivity_mat)

        # Take old nearest neighbors and add the newly calculated ones
        NNInfo['conn_mat'] += connectivity_mat

        # We also store information of to what current child-node each node in the knn-index is mapped
        NNInfo['leafToChild'] = {nodeInd: nodeInd for nodeInd in nodeInds}
        NNInfo['index'] = index
        if verbose:
            mp_print("Getting nearest neighbour pairs took %f seconds." % (time.time() - start))
        return pairs, NNInfo

    def get_new_nn_pairs(self, new_node, NNInfo, runConfigs, UBInfo=None, old_pairs_list=None, update_nn_index=False,
                         xrAIRoot=None):
        if UBInfo is not None:
            # TODO: Check if this is necessary: # In this case also delete old UB-information on the "new_node", since we're going to calculate that again.
            to_be_deleted = np.where(UBInfo['pairs'] == new_node.nodeInd)[0]
            UBInfo['pairs'] = np.delete(UBInfo['pairs'], to_be_deleted, axis=0)
            UBInfo['UBs'] = np.delete(UBInfo['UBs'], to_be_deleted)
        if old_pairs_list is not None:
            for ind_pairs, old_pairs in enumerate(old_pairs_list):
                new_node_ind = new_node.nodeInd
                # In this case, we also delete all pairs with the new_node from the old set of pairs
                old_pairs_list[ind_pairs] = [old_pair for old_pair in old_pairs if new_node_ind not in old_pair]
        # Here, we should just get NN-pairs with the new ancestor. Take twice as many neighbours as for the
        # other nodes to compensate for no other nodes adding connections to ancestor
        if not update_nn_index:
            centered_query = (new_node.ltqs - NNInfo['subtracted_mean'])[:, None]
            normalized_query = centered_query / np.linalg.norm(centered_query)
            index, nns = getApproxNNs(normalized_query, index=NNInfo['index'],
                                      k=20 * runConfigs['kNN'], pointsIds=[new_node.nodeInd], addPoints=False)
        else:
            # We gather information about the current children of the root
            ltqsCh, _, _ = self.getInfoChildren()
            # We center the ltq-information around the root
            ltqsCh -= xrAIRoot[:, None]
            NNInfo['subtracted_mean'] = xrAIRoot

            # Instead of taking cosine metric, we can also normalize the vectors and use squared-euclidean
            norms = np.linalg.norm(ltqsCh, axis=0)
            np.divide(ltqsCh, norms, out=ltqsCh)

            # We ask for the nearest neighbours
            nodeInds = [child.nodeInd for child in self.childNodes]
            pointsT = ltqsCh.T
            NNInfo['index'].fit(pointsT)
            NNInfo['index'].IDs = nodeInds

            centered_query = (new_node.ltqs - NNInfo['subtracted_mean'])[:, None]
            normalized_query = centered_query / np.linalg.norm(centered_query)

            index, nns = getApproxNNs(normalized_query, index=NNInfo['index'],
                                      k=20 * runConfigs['kNN'], pointsIds=[new_node.nodeInd], addPoints=False)
            NNInfo['leafToChild'] = {nodeInd: nodeInd for nodeInd in nodeInds}

        nns = np.array(index.IDs)[nns]
        # Make list of unique neighbors that are not the node itself
        non_unique_partners = np.array([NNInfo['leafToChild'][nb] for nb in nns[0, :] if NNInfo['leafToChild'][nb] != new_node.nodeInd])
        seen = set()
        partners = []
        for item in non_unique_partners:
            if item not in seen:
                seen.add(item)
                partners.append(item)

        pairs = [(new_node.nodeInd, partner) for partner in partners]
        if len(pairs) > 0:
            if len(pairs) > runConfigs['kNN']:
                pairs = pairs[:runConfigs['kNN']]
            # Update the connectivity matrix for the nearest neighbors with these new pairs
            pairs_array = np.array(pairs)
            NNInfo['conn_mat'][pairs_array[:, 0], pairs_array[:, 1]] = True
            NNInfo['conn_mat'][pairs_array[:, 1], pairs_array[:, 0]] = True

        pairs = [(new_node.nodeInd, nb_nodeInd) for nb_nodeInd in
                 NNInfo['conn_mat'][[new_node.nodeInd], :].nonzero()[1]]
        if old_pairs_list is not None:
            return pairs, old_pairs_list
        return pairs

    def optimiseConnected(self, mpiInfo, singleProcess=False, verbose=True):
        if (mpiInfo.rank == 0) or singleProcess:
            ltqsCh, ltqsVarsCh, tCh = self.getInfoChildren()
            tStar, loglik, W_g, xr_g = optimiseTStar(ltqsCh, ltqsVarsCh, verbose=verbose, nChilds=len(self.childNodes))
        else:
            tStar = None
            W_g = None
            xr_g = None
        if (not singleProcess) and (mpiInfo.size > 1):
            tStar = mpi_wrapper.bcast(tStar, root=0)
            W_g = mpi_wrapper.bcast(W_g, root=0)
            xr_g = mpi_wrapper.bcast(xr_g, root=0)
        for ind, child in enumerate(self.childNodes):
            child.tParent = tStar[ind]
        self.getLtqsComplete(mem_friendly=True)
        return True, W_g, xr_g

    def getChildInfo(self, xrAsIfRoot_g, WAsIfRoot_g, ind1, ind2, nodeInd1, recalcInd1=True):
        # Gather information on second node
        child2 = self.childNodes[ind2]

        sortedNodeInds = tuple(sorted([nodeInd1, child2.nodeInd]))
        # We check whether for this pair of nodes we already have some reasonable diff. time guess
        # tInit = getNodePairInfo(sortedNodeInds)
        # In case we do sequential optimisation, we also check if t_12 = t_1a+t_2a is known already
        # t12Opt = getNodePairInfo(sortedNodeInds) if boolArgs['sequential'] else None
        if not recalcInd1:
            return child2, sortedNodeInds
        else:
            # Gather information on first node
            child1 = self.childNodes[ind1]
            tOld1, wbar1_g, ltqs1, ltqsVars1, nodeInd1 = child1.getInfo()
            child1Tuple = (tOld1, wbar1_g, ltqs1, ltqsVars1, nodeInd1)

            # Do some computations that are independent of second node:
            rootMinusFirst_W_g = WAsIfRoot_g - wbar1_g
            rootMinusFirst_ltqs = xrAsIfRoot_g * WAsIfRoot_g - wbar1_g * ltqs1
            return child2, sortedNodeInds, child1, child1Tuple, rootMinusFirst_ltqs, rootMinusFirst_W_g

    def initializeMergeGeneral(self, sequential=True, ellipsoidSize=None, mpiInfo=None, specialChild=None,
                               random=False, nChildNN=-1, nChildUB=-1, kNN=50, redoTimeFrac=0.05, redoNNFrac=0.1,
                               outputFolder=None, tree=None, nNewPairsPar=1000, nDonePairsPar=3000, tmpTreeInd=None,
                               n_mem_friendly=500):
        runConfigs = {'sequential': sequential}
        runConfigs['outputFolder'] = outputFolder

        # Initialize boolean that determines if we are estimating dLogL-upper bounds for all pairs
        runConfigs['useUB'] = runConfigs['getNewUB'] = (ellipsoidSize is not None) and (specialChild is None) and (
            not random)

        # Determine what number of children should be merged before ending this loop
        nChild = len(self.childNodes)
        expNChild = 3 if self.isRoot else 2
        # specialChild indicates that we only want to compare merges with a single node
        if specialChild is not None:
            expNChild = max(nChild - 1, expNChild)
        # nodeIndToMergeInd = {child.nodeInd: ind for ind, child in enumerate(self.childNodes)}
        chInfo = {'nChild': nChild, 'expNChild': expNChild}  # , 'nodeIndToMergeInd': nodeIndToMergeInd}

        # Determine if we should use nearest-neighbours as candidates for merges
        runConfigs['useNN'] = (nChildNN >= 0) and (nChild > nChildNN) and (specialChild is None)
        runConfigs['getNewNN'] = runConfigs['useNN']

        # Prepare some information for using only nearest neighbours
        runConfigs['nChildNN'] = nChildNN
        runConfigs['nChildUB'] = nChildUB
        runConfigs['kNN'] = kNN
        NNInfo = {'NNcounter': 0, 'conn_mat': lil_array((2 * bs_glob.nCells, 2 * bs_glob.nCells), dtype=bool),
                  'subtracted_mean': None}

        # Determine how we communicate data between processes (mem-friendly, using memmap) or not
        runConfigs['n_mem_friendly'] = n_mem_friendly
        runConfigs['mem_friendly'] = nChild > n_mem_friendly
        runConfigs['mem_friendly_folder'] = None
        if runConfigs['mem_friendly']:
            if outputFolder is not None:
                runConfigs['mem_friendly_folder'] = os.path.join(outputFolder,
                                                                 'shared_data_%d' % np.random.randint(1e8))
            else:
                runConfigs['mem_friendly_folder'] = os.path.abspath('shared_data_%d' % np.random.randint(1e8))
            Path(runConfigs['mem_friendly_folder']).mkdir(parents=True, exist_ok=True)

        # Create a dictionary that points from unique nodeInd to the index of that node in the list of root-children
        chInfo['nodeIndToChildInd'] = {child.nodeInd: childInd for childInd, child in enumerate(self.childNodes)}

        # TODO: Maybe remove timestamps-option eventually
        timeStamps = False
        runConfigs['timeStamps'] = timeStamps if (mpiInfo.rank == 0) else False

        # Prepare some information for deciding whether to use parallel computing or not
        # parTry = (mpiInfo.size > 1) and (chInfo['nChild'] > 10)
        # parallelInfo = {'timingList': [], 'par': parTry, 'parNow': parTry, 'timingCheck': 2,
        #                 'parTiming': None, 'nonparTiming': None, 'parTry': parTry, 'nextTry': 1e9}
        runConfigs['nNewPairsPar'] = nNewPairsPar
        runConfigs['nDonePairsPar'] = nDonePairsPar
        runConfigs['parNow'] = mpiInfo.size > 1

        # Initialize some more variables
        newAnc = None
        del_node_inds = []
        oldRoot = None
        UBInfo = None
        pairs = None
        timeOptInfo = {'doTimeOpt': False, 'timeOptRoundsAgo': 0}
        if specialChild:
            timeOptInfo['timeOptRoundsAgo'] = 1
        runConfigs['redoTimeAt'] = int(chInfo['nChild'] * redoTimeFrac)
        runConfigs['redoNNAt'] = int(chInfo['nChild'] * redoNNFrac)
        # if manualMerges is not None:
        #     manualMergeCounter = -1
        #     mergeIndToNodeInd = {ind: child.nodeInd for ind, child in enumerate(self.childNodes)}

        # Store initial version of the tree in an intermediate folder where we track the progress
        mergeCounter = 0
        tmp_folder = None
        tmp_tree_ind = None
        if (outputFolder is not None) and (specialChild is None):
            tmp_folder = os.path.join(outputFolder, 'tmp_trees')
            Path(tmp_folder).mkdir(parents=True, exist_ok=True)
            if mpiInfo.rank == 0:
                if tmpTreeInd is not None:
                    tmp_tree_ind = tmpTreeInd + 1
                else:
                    tmp_tree_ind = 0
        breakOut = False
        pairsDoneInfo = {'totalPairsDone': 0, 'pairsDoneList': []}

        initNoneVars = (newAnc, del_node_inds, oldRoot, UBInfo, pairs)
        initVars = (mergeCounter, tmp_folder, tmp_tree_ind, breakOut, pairsDoneInfo)
        infoTuple = (chInfo, NNInfo, runConfigs, timeOptInfo)
        return infoTuple, initVars, initNoneVars

    # Used
    def getOptimalDLogLPairInfoNew(self, xrAsIfRoot_g, WAsIfRoot_g, maxdLogLTuples, maxdLogLTuplesZeroRoot,
                                   nodeIndToChildInd, timeOptInfo, sequential=True, verbose=True,
                                   random=False, dLogLDict=None):
        ind1s, ind2s, maxdLogLs = zip(*maxdLogLTuples)
        optdLogL = maxdLogLTuples[np.argmax(maxdLogLs)]
        dLogLOpt = optdLogL[2]

        ind1sZeroRoot, ind2sZeroRoot, maxdLogLsZeroRoot = zip(*maxdLogLTuplesZeroRoot)
        optdLogLZeroRoot = maxdLogLTuplesZeroRoot[np.argmax(maxdLogLsZeroRoot)]
        dLogLOptZeroRoot = optdLogLZeroRoot[2]

        optimiseTimes = (dLogLOptZeroRoot > dLogLOpt) and (timeOptInfo['timeOptRoundsAgo'] > 0)

        if random:
            pairs, dLogLs = zip(*dLogLDict.items())
            dLogLs = np.array(dLogLs)
            posDLogLInds = np.where(dLogLs > 1e-6)[0]
            if not len(posDLogLInds):
                posDLogLInds = np.arange(len(dLogLs))
            posDLogL = dLogLs[posDLogLInds]
            dLogLsCorr = posDLogL - posDLogL.max()
            dLs = np.exp(dLogLsCorr)
            opt = np.random.choice(posDLogLInds, p=dLs / np.sum(dLs))
            dLogLOpt = dLogLs[opt]
            optdLogL = (pairs[opt][0], pairs[opt][1], dLogLOpt)
            optimiseTimes = False

        if optimiseTimes:
            timeOptInfo['doTimeOpt'] = True
            timeOptInfo['timeOptRoundsAgo'] = 0
            if self.isRoot:
                if verbose:
                    mp_print("It is now optimal to re-optimise the times between nodes %d and %d. \n"
                             "Instead, we will optimise all edge lengths from the root "
                             "and then re-calculate the dLogLs." % (optdLogLZeroRoot[0], optdLogLZeroRoot[1]),
                             ALL_RANKS=True)
                getOptInfo = False
            else:
                if verbose:
                    mp_print("It is now optimal to re-optimise the times between nodes %d and %d. \n"
                             "We will reset these times, without merging them." % (
                                 optdLogLZeroRoot[0], optdLogLZeroRoot[1]), ALL_RANKS=True)
                getOptInfo = True
                optdLogL = optdLogLZeroRoot
        else:
            timeOptInfo['doTimeOpt'] = False
            timeOptInfo['timeOptRoundsAgo'] += 1
            getOptInfo = (dLogLOpt > 1e-6)

        if not getOptInfo:
            newTDict = None
            optNodeInd1 = None
            optNodeInd2 = None
        else:
            optNodeInd1 = optdLogL[0]
            optNodeInd2 = optdLogL[1]
            optInd1 = nodeIndToChildInd[optNodeInd1]
            optInd2 = nodeIndToChildInd[optNodeInd2]

            optChild1 = self.childNodes[optInd1]
            optChild2 = self.childNodes[optInd2]
            # Find optimal times for optimal pair (we repeat this here, while we could look it up but this doesn't
            # significantly increase computation time, and this is useful for later (when we may approximate the
            # optTimes in the first calculation))
            optTimes = getOptTimesSingleDLogLWrapper(xrAsIfRoot_g, WAsIfRoot_g, optChild1, optChild2,
                                                     sequential=sequential, verbose=verbose, tol=1e-9)
            newTDict = {optNodeInd1: optTimes[0], optNodeInd2: optTimes[1], bs_glob.nNodes - 1: optTimes[2]}

        # Communicate merge information with other computing processes
        # if mpiInfo.size > 1:
        #     newTDict = bcast(newTDict, root=0)
        #     optNodeInd1 = bcast(optNodeInd1, root=0)
        #     optNodeInd2 = bcast(optNodeInd2, root=0)
        #     timeOptInfo = bcast(timeOptInfo, root=0)
        #     dLogLOpt = bcast(dLogLOpt, root=0)
        return optNodeInd1, optNodeInd2, newTDict, timeOptInfo, dLogLOpt

    # Used
    def mergeNodes(self, xrAsIfRoot_g, WAsIfRoot_g, nodeInd1, nodeInd2, newTDict, nodeIndToChildInd, dLogLOpt,
                   sequential=True, verbose=False, ellipsoidSize=None, mergeDownstream=True, singleProcess=False,
                   random=False, runConfigs=None, NNInfo=None):

        # TODO: Remove this later. Only uncomment this for printing cell-annotations
        # celltype_info = hasattr(self, 'celltype')

        addAsChild = False
        if (newTDict[nodeInd1] < 1e-6) and (not self.childNodes[nodeIndToChildInd[nodeInd1]].isCell):
            addAsChild = True
            ancNodeInd = nodeInd1
            gchildInds = [nodeInd2]
        elif (newTDict[nodeInd2] < 1e-6) and (not self.childNodes[nodeIndToChildInd[nodeInd2]].isCell):
            addAsChild = True
            ancNodeInd = nodeInd2
            gchildInds = [nodeInd1]
        if addAsChild:
            newNode = self.childNodes[nodeIndToChildInd[ancNodeInd]]

            # Print message on which nodes are merged
            if verbose:
                mp_print("Adding {} as child of {}.".format(self.childNodes[nodeIndToChildInd[gchildInds[0]]].nodeId,
                                                            self.childNodes[nodeIndToChildInd[ancNodeInd]].nodeId))
                # TODO: Remove this later. Only uncomment this for printing cell-annotations
                # if celltype_info:
                #     mp_print(
                #         "Adding {} as child of {}.".format(self.childNodes[nodeIndToChildInd[gchildInds[0]]].celltype,
                #                                            self.childNodes[nodeIndToChildInd[ancNodeInd]].celltype))
                mp_print("\nAdding %d as child of %d with the following times: root -> %d = %f, %d -> %d = %f. "
                         "Increase in loglikelihood: %f per gene.\n" % (
                             gchildInds[0], ancNodeInd, ancNodeInd, newTDict[bs_glob.nNodes - 1], ancNodeInd,
                             gchildInds[0], newTDict[gchildInds[0]], dLogLOpt / bs_glob.nGenes),
                         ALL_RANKS=singleProcess)
        else:
            # Create new node that will be added as ancestor of the children with nodeInd1, nodeInd2
            ancNodeInd = bs_glob.nNodes - 1
            newNode = TreeNode(nodeInd=ancNodeInd)
            newNode.childNodes = []
            gchildInds = [nodeInd1, nodeInd2]
            # Print message on which nodes are merged
            if verbose:
                mp_print("Merging {} and {}.".format(self.childNodes[nodeIndToChildInd[nodeInd1]].nodeId,
                                                     self.childNodes[nodeIndToChildInd[nodeInd2]].nodeId))
                mp_print(
                    "Merging nodes " + str(nodeInd1) + " and " + str(nodeInd2) + " into ancestor " + str(
                        bs_glob.nNodes - 1) + ', with times ' + str([newTDict[nodeInd1], newTDict[nodeInd2], newTDict[
                        bs_glob.nNodes - 1]]) + '. Increase in loglikelihood: ' + str(
                        dLogLOpt / bs_glob.nGenes) + ' per gene.', ALL_RANKS=singleProcess)

                # TODO: Remove this later. Only uncomment this for printing cell-annotations
                # if celltype_info:
                #     mp_print("Merging celltypes {} and {}".format(self.childNodes[nodeIndToChildInd[nodeInd1]].celltype,
                #                                                   self.childNodes[
                #                                                       nodeIndToChildInd[nodeInd2]].celltype))

        newNode.isLeaf = False
        delInds = []
        del_node_inds = []
        W_g = self.getW()
        newSelfLtqs = self.ltqs * W_g
        if not self.isRoot:
            newSelfLtqsAIRoot = xrAsIfRoot_g * WAsIfRoot_g

        if addAsChild:
            # If one child-node remains child of root, its contribution to the root-ltqs should be subtracted here,
            # so that it can be added again when it is updated
            wbarChild_g = 1 / (newNode.tParent + newNode.getLtqsVars())
            W_g -= wbarChild_g
            newSelfLtqs -= wbarChild_g * newNode.ltqs
            if not self.isRoot:
                WAsIfRoot_g -= wbarChild_g
                newSelfLtqsAIRoot -= wbarChild_g * newNode.ltqs
            del_node_inds.append(newNode.nodeInd)

        # Loop over the children that should be merged
        for nodeInd in gchildInds:
            ind = nodeIndToChildInd[nodeInd]
            child = self.childNodes[ind]
            delInds.append(ind)
            del_node_inds.append(nodeInd)
            newNode.childNodes.append(child)

            # We subtract the contribution to the root-ltqs from the children that will be below ancestor
            wbarChild_g = 1 / (child.tParent + child.getLtqsVars())
            W_g -= wbarChild_g
            newSelfLtqs -= wbarChild_g * child.ltqs
            if not self.isRoot:
                WAsIfRoot_g -= wbarChild_g
                newSelfLtqsAIRoot -= wbarChild_g * child.ltqs
            # We store the calculated optimal time from these children to their new ancestor
            child.tParent = newTDict[child.nodeInd]

        # TODO: Remove this later. Only uncomment this for printing cell-annotations
        # if celltype_info:
        #     child_celltypes = [child_node.celltype for child_node in newNode.childNodes]
        #     if len(np.unique(child_celltypes)) == 1:
        #         newNode.celltype = child_celltypes[0]
        #         plot_ciphers = (len(self.childNodes) % 1000) == 0
        #     else:
        #         newNode.celltype = 'unknown'
        #         plot_ciphers = True
        #     plot_ciphers = False

        # We add the optimal time from the new ancestor to the root
        newNode.tParent = newTDict[bs_glob.nNodes - 1]
        # We get the ltqs of the ancestor based on the ltqs of the two merged children
        # TODO: Maybe do the following more efficiently by updating the node-ltqs already above
        newNode.getLtqsUponMerge()

        # TODO: Remove this later. Only uncomment this for printing cell-annotations
        # if celltype_info and plot_ciphers:
        #     root_now = self
        #     if addAsChild:
        #         gchild1 = newNode
        #         gchild2 = self.childNodes[nodeIndToChildInd[gchildInds[0]]]
        #         labels = ['root', 'new ancestor', 'grandchild', 'new ancestor']
        #     else:
        #         gchild1 = self.childNodes[nodeIndToChildInd[gchildInds[0]]]
        #         gchild2 = self.childNodes[nodeIndToChildInd[gchildInds[1]]]
        #         labels = ['root', 'child 1', 'child 2', 'new ancestor']
        #     gene_numbers = np.array([int(gene_id[5:]) for gene_id in bs_glob.geneIds])
        #     fig, ax = plt.subplots(ncols=4, nrows=1)
        #     for gchild_ind, gchild_iter in enumerate([root_now, gchild1, gchild2, newNode]):
        #         image = np.zeros(28 * 28)
        #         image[gene_numbers] = gchild_iter.ltqs
        #         image = np.reshape(image, newshape=(28, 28))
        #         ax[gchild_ind].imshow(image, cmap='gray_r')
        #         ax[gchild_ind].set_title(labels[gchild_ind])
        #
        #         # image = np.zeros(28 * 28)
        #         # image[gene_numbers] = np.sqrt(gchild_iter.getLtqsVars())
        #         # image = np.reshape(image, newshape=(28, 28))
        #         # ax[1, gchild_ind].imshow(image, cmap='gray_r')
        #     plt.tight_layout()
        #     plt.show()

        # We add the contribution of the new ancestor to the position of the root
        wbarParent_g = 1 / (newNode.tParent + newNode.getLtqsVars())
        W_g += wbarParent_g
        self.setLtqsVarsOrW(W_g=W_g)
        newSelfLtqs += wbarParent_g * newNode.ltqs
        self.ltqs = newSelfLtqs / self.getW()
        if not self.isRoot:
            WAsIfRoot_g += wbarParent_g
            newSelfLtqsAIRoot += wbarParent_g * newNode.ltqs
            xrAsIfRoot_g = newSelfLtqsAIRoot / WAsIfRoot_g
        else:
            xrAsIfRoot_g = self.ltqs
            WAsIfRoot_g = self.getW()

        # We don't do this anymore: we make the new ancestor inherit the nearest neighbors from the merged child-nodes
        if runConfigs['useNN'] and (NNInfo is not None):
            # First add the neighbor-connections from the merged children
            NN_conn_mat = NNInfo['conn_mat']
            for gchild_nodeInd in gchildInds:
                # TODO: Check if this is necessary
                # NN_conn_mat[:, [newNode.nodeInd]] += NN_conn_mat[:, [gchild_nodeInd]]
                # Then delete the neighbor-information for the already merged children (out of efficiency)
                NN_conn_mat[:, [gchild_nodeInd]] = 0
                NN_conn_mat[[gchild_nodeInd], :] = 0
            NN_conn_mat[[newNode.nodeInd], :] += NN_conn_mat.T[[newNode.nodeInd], :]
            NN_conn_mat[newNode.nodeInd, newNode.nodeInd] = 0
        # Lastly, we delete the two merged children from the root-childnodes and add the new node
        self.childNodes = [child for ind, child in enumerate(self.childNodes) if ind not in delInds]
        if not addAsChild:
            self.childNodes.append(newNode)
            bs_glob.nNodes += 1

        # If one of children was added below the other. Ask if it wants to merge with one of the other grand-children
        if addAsChild and mergeDownstream:
            someMergeHappened, xrAsIfRoot_g, WAsIfRoot_g, oldLtqs, oldLtqsVars = self.mergeDownstreamChildrenSingle(
                xrAsIfRoot_g, WAsIfRoot_g, childNInd=newNode.nodeInd, gchildNInd=gchildInds[0], sequential=sequential,
                verbose=False, ellipsoidSize=ellipsoidSize, singleProcess=singleProcess, random=random,
                runConfigs_inherited=runConfigs)

        if self.isRoot:
            return self.ltqs, self.getW(), newNode, del_node_inds
        else:
            return xrAsIfRoot_g, WAsIfRoot_g, newNode, del_node_inds

    # Used
    def resetChildTimes(self, xrAsIfRoot_g, WAsIfRoot_g, nodeInd1, nodeInd2, newTDict, nodeIndToChildInd):
        W_g = self.getW()
        newAncestorLtqs = self.ltqs * W_g
        if not self.isRoot:
            newAncLtqsAIRoot = xrAsIfRoot_g * WAsIfRoot_g

        # Loop over the two children that should be merged
        for nodeInd in [nodeInd1, nodeInd2]:
            ind = nodeIndToChildInd[nodeInd]
            child = self.childNodes[ind]

            # We subtract the contribution to the root-ltqs from the two children that will be merged
            wbarChild_g = 1 / (child.tParent + child.getLtqsVars())
            W_g -= wbarChild_g
            newAncestorLtqs -= wbarChild_g * child.ltqs
            if not self.isRoot:
                WAsIfRoot_g -= wbarChild_g
                newAncLtqsAIRoot -= wbarChild_g * child.ltqs
            # We store the calculated optimal time from these children to their new ancestor
            child.tParent = newTDict[child.nodeInd]

            # We add the contribution of the new ancestor to the position of the root
            wbarChild_gNew = 1 / (child.tParent + child.getLtqsVars())
            W_g += wbarChild_gNew
            newAncestorLtqs += wbarChild_gNew * child.ltqs
            if not self.isRoot:
                WAsIfRoot_g += wbarChild_gNew
                newAncLtqsAIRoot += wbarChild_gNew * child.ltqs

        self.setLtqsVarsOrW(W_g=W_g)
        self.ltqs = newAncestorLtqs / self.getW()
        if not self.isRoot:
            xrAsIfRoot_g = newAncLtqsAIRoot / WAsIfRoot_g

        if self.isRoot:
            return self.ltqs, self.getW()
        else:
            return xrAsIfRoot_g, WAsIfRoot_g

    def storeParent(self):
        for child in self.childNodes:
            child.parentNode = self
            if not child.isLeaf:
                child.storeParent()

    def getAIRootInfo(self, ltqsParent, W_gParent):
        if self.isRoot:
            ltqsAIRoot = self.ltqs.copy()
            WAIRoot = self.getW().copy()
        else:
            ltqsAIRoot, WAIRoot = getLtqsAsIfRoot(self.ltqs, self.getW(), self.tParent, ltqsParent, W_gParent)
        self.ltqsAIRoot = ltqsAIRoot
        self.setLtqsVarsOrW(W_g=WAIRoot, AIRoot=True)
        for child in self.childNodes:
            child.getAIRootInfo(ltqsAIRoot, WAIRoot)

    def getAIRootUpstream(self):
        if self.isRoot:
            self.ltqsAIRoot = self.ltqs.copy()
            self.setLtqsVarsOrW(W_g=self.getW().copy(), AIRoot=True)
        else:
            self.parentNode.getAIRootUpstream()
            self.ltqsAIRoot, WAIRoot = getLtqsAsIfRoot(self.ltqs, self.getW(), self.tParent, self.parentNode.ltqsAIRoot,
                                                       self.parentNode.getW(AIRoot=True))
            self.setLtqsVarsOrW(W_g=WAIRoot, AIRoot=True)

    def getNodeList(self, nodeList, returnLeafs=True, returnRoot=True):
        if self.isLeaf and not returnLeafs:
            pass
        elif self.isRoot and not returnRoot:
            pass
        else:
            nodeList.append(self)
        for child in self.childNodes:
            nodeList = child.getNodeList(nodeList, returnLeafs=returnLeafs, returnRoot=returnRoot)
        return nodeList

    def getNodeDict(self, nodeDict, returnLeafs=True, returnRoot=True):
        if self.isLeaf and not returnLeafs:
            pass
        elif self.isRoot and not returnRoot:
            pass
        else:
            nodeDict[self.nodeId] = self
        for child in self.childNodes:
            nodeDict = child.getNodeDict(nodeDict, returnLeafs=returnLeafs, returnRoot=returnRoot)
        return nodeDict

    def getEdgesComplete(self, edgeList, src=np.nan, indMap=None):
        # Store per edge in the tree 1) the source nodeInd, 2) the dst nodeInd, 3) the time between, 4) if dst is leaf
        if self.parentNode is not None:
            if self.parentNode.nodeInd != src:
                nodeInd = self.nodeInd if indMap is None else indMap[self.nodeInd]
                parentNodeInd = self.parentNode.nodeInd if indMap is None else indMap[self.parentNode.nodeInd]
                edgeList.append((nodeInd, parentNodeInd, self.tParent, self.parentNode.isLeaf))
                edgeList = self.parentNode.getEdgesComplete(edgeList, src=self.nodeInd, indMap=indMap)
        for child in self.childNodes:
            if child.nodeInd != src:
                nodeInd = self.nodeInd if indMap is None else indMap[self.nodeInd]
                childNodeInd = child.nodeInd if indMap is None else indMap[child.nodeInd]
                edgeList.append((nodeInd, childNodeInd, child.tParent, child.isLeaf))
                edgeList = child.getEdgesComplete(edgeList, src=self.nodeInd, indMap=indMap)
        return edgeList


class Tree:
    root = None
    loglik = None
    starryYN = None
    nNodes = None

    vert_ind_to_node = None

    def __init__(self):
        self.root = TreeNode(nodeInd=-1, isRoot=True, nodeId='root')
        bs_glob.nNodes = 1

    def __repr__(self):
        return "Tree(root=%r\n\nloglik=%r,nNodes=%r)" % (self.root, self.loglik, self.nNodes)

    # Used
    def buildTree(self, ltqs, ltqsVars, cellIds):
        self.root.childNodes = []
        for ind in range(bs_glob.nCells):
            self.root.childNodes.append(
                TreeNode(nodeInd=ind, childNodes=[], isLeaf=True, ltqs=ltqs[:, ind], tParent=1,
                         ltqsVars=ltqsVars[:, ind], nodeId=cellIds[ind], isCell=True))
        bs_glob.nNodes = bs_glob.nCells + 1

    def copy_tree_topology(self):
        tree_copy = Tree()
        tree_copy.nNodes = self.nNodes
        tree_copy.root = self.root.copy_node_topology()
        return tree_copy

    def to_newick(self, use_ids=True, results_path=None):
        nwk_str = self.root.to_newick_node(use_ids=use_ids)
        if results_path is not None:
            with open(results_path, 'w') as f:
                f.write(nwk_str)
        return nwk_str

    def get_edge_dataframe(self, nodeIdToVertInd=None):
        if nodeIdToVertInd is None:
            edge_dict = {'source': [], 'target': [], 'dist': []}
        else:
            edge_dict = {'source': [], 'source_ind': [], 'target': [], 'target_ind': [], 'dist': []}
        edge_dict = self.root.get_edge_dict_node(edge_dict, nodeIdToVertInd=nodeIdToVertInd)
        edge_df = pd.DataFrame(edge_dict)
        return edge_df

    # Used
    def compile_tree_from_scData_tree(self):
        edge_list = []
        dist_list = []
        if self.root.nodeId is None:
            self.root.nodeId = 'root'
        rootId = self.root.nodeId
        orig_vert_names = {self.root.nodeInd: rootId}
        intCounter = 0
        starryYN = len(self.root.childNodes) > 3
        nodeIndToNode = {}
        edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode = getEdgeDistVertNamesFromNode(self.root,
                                                                                                        edge_list,
                                                                                                        dist_list,
                                                                                                        orig_vert_names,
                                                                                                        intCounter,
                                                                                                        nodeIndToNode)
        return edge_list, dist_list, orig_vert_names, starryYN, nodeIndToNode

    def getEdgeVertInfo(self, coords_folder=None, verbose=False, store_posterior_ltqs=False,
                        undo_rescale_by_var=True, variances=None):
        edgeList, distList, nodeIndToVertId, _, nodeIndToNode = self.compile_tree_from_scData_tree()
        if coords_folder is not None:
            if not store_posterior_ltqs:
                ltqs_file = os.path.join(coords_folder, 'ltqs_vertByGene.npy')
                ltqsVars_file = os.path.join(coords_folder, 'ltqsVars_vertByGene.npy')
            else:
                ltqs_file = os.path.join(coords_folder, 'posterior_ltqs_vertByGene.npy')
                ltqsVars_file = os.path.join(coords_folder, 'posterior_ltqsVars_vertByGene.npy')
        vertInfo = {}
        nodeIndToVertInd = {}
        vertIndCounter = 0

        if coords_folder is not None:
            if store_posterior_ltqs:
                self.root.getAIRootInfo(None, None)
            start = time.time()
            ltqs = []
            ltqsVars = []
            for edge in edgeList:
                for ind, nodeInd in enumerate(edge):
                    if nodeInd not in nodeIndToVertInd:
                        nodeIndToVertInd[nodeInd] = vertIndCounter
                        vertInfo[vertIndCounter] = (nodeInd, nodeIndToVertId[nodeInd])
                        if verbose and (vertIndCounter % 1000 == 0):
                            mp_print("Writing coords of vertex %d to file." % vertIndCounter)
                        if not store_posterior_ltqs:
                            ltqs.append(nodeIndToNode[nodeInd].ltqs)
                            ltqsVars.append(nodeIndToNode[nodeInd].getLtqsVars())
                        else:
                            if undo_rescale_by_var:
                                # This means we have to undo the rescaling that was done before
                                node_ltqs_post = nodeIndToNode[nodeInd].ltqsAIRoot * np.sqrt(variances)
                                node_ltqsVars_post = nodeIndToNode[nodeInd].getLtqsVars(AIRoot=True) * variances
                            else:
                                node_ltqs_post = nodeIndToNode[nodeInd].ltqsAIRoot
                                node_ltqsVars_post = nodeIndToNode[nodeInd].getLtqsVars(AIRoot=True)
                            ltqs.append(node_ltqs_post)
                            ltqsVars.append(node_ltqsVars_post)
                        # ltqsfile.write('\t'.join(np.char.mod('%.8e', nodeIndToNode[nodeInd].ltqs)) + '\n')
                        # varsfile.write('\t'.join(np.char.mod('%.8e', nodeIndToNode[nodeInd].getLtqsVars())) + '\n')
                        vertIndCounter += 1
                    vertInd = nodeIndToVertInd[nodeInd]
                    edge[ind] = vertInd
            ltqs = np.vstack(ltqs)
            ltqsVars = np.vstack(ltqsVars)
            np.save(ltqs_file, ltqs, allow_pickle=False)
            np.save(ltqsVars_file, ltqsVars, allow_pickle=False)
            mp_print("Printing to file took %.2f seconds." % (time.time() - start))
        else:
            for edge in edgeList:
                for ind, nodeInd in enumerate(edge):
                    if nodeInd not in nodeIndToVertInd:
                        nodeIndToVertInd[nodeInd] = vertIndCounter
                        vertInfo[vertIndCounter] = (nodeInd, nodeIndToVertId[nodeInd])
                        vertIndCounter += 1
                    vertInd = nodeIndToVertInd[nodeInd]
                    edge[ind] = vertInd
        return edgeList, distList, vertInfo

    # Used
    def getMergeList(self):
        mergers = []  # This will become a list of tuples with (nodeInd1, nodeInd2, tdiff1, tdiff2)
        self.root.getChildMergers(mergers)
        return mergers

    # Used
    def calcLogLComplete(self, mem_friendly=True, loglikVarCorr=None, recalc=True):
        # Calculate effective positions (Ltqs), error-bars (LtqVars), and prefactors for all children of the root
        if recalc:
            self.root.getLtqsComplete(mem_friendly=mem_friendly)
        if loglikVarCorr is None:
            loglik = self.root.prefactor
        else:
            loglik = self.root.prefactor + loglikVarCorr
        return loglik

    # Used
    def logLGradCompleteLambda(self, logLambda, geneInd, nCells):
        if geneInd < 0:
            nGenes = len(logLambda)
            loglik = -nCells * np.sum(logLambda)
            logGrad = -nCells * np.ones(nGenes)
            lam = np.exp(logLambda)
            loglik, logGrad, _, _, _ = self.root.getLogLGradLambda(lam, loglik, logGrad, nGenes)
        else:
            logLambda = logLambda[0]
            loglik = - nCells * logLambda
            logGrad = -nCells
            lam = np.exp(logLambda)

            loglik, logGrad, _, _, _ = self.root.getLogLGradLambdaSingleGene(lam, geneInd, loglik, logGrad)

        # print("Loglik: " + str(loglik))
        return - loglik, - logGrad

    # Used
    def getLikelihoodGradGivenTimes(self, t, nodeInds, verbose=False, mem_friendly=False):
        t = dict(zip(nodeInds, t))
        self.root.assignTs(t)

        # First calculate the positions (ltqs, W_g) for all nodes in the tree. This also gives us the loglik
        self.root.getLtqsComplete(mem_friendly=mem_friendly)
        loglik = self.root.prefactor

        # Then calculate derivative w.r.t. all diff times
        self.root.getDerivativesDownstream(self.root.ltqs, self.root.getW())

        grad = self.root.getGrads({})
        grad = np.array([grad[ind] for ind in nodeInds])
        if verbose:
            mp_print("Loglik = " + str(loglik), ALL_RANKS=True)
        return loglik, grad

    # Used
    def getNegLikelihoodGradGivenLogTimesWrapper(self, logt, nodeInds, verbose=False, mem_friendly=False):
        t = np.exp(logt)
        loglik, grad = self.getLikelihoodGradGivenTimes(t, nodeInds, verbose=verbose, mem_friendly=mem_friendly)
        return - loglik, - grad * t

    def getNegLikelihoodGradGivenLogTimesWrapper_single_scalar(self, loglamb, logt, nodeInds, verbose=False,
                                                               mem_friendly=False):
        t = np.exp(loglamb + logt)
        loglik, grad = self.getLikelihoodGradGivenTimes(t, nodeInds, verbose=verbose, mem_friendly=mem_friendly)
        return - loglik, np.sum(- grad * t)

    # Used
    def optTimes(self, verbose=False, singleProcess=False, mem_friendly=True, maxiter=1e6, tol=1e-6):
        """

        :param verbose:
        :param singleProcess: This boolean variable determines whether many processes work together to calculate such
        that mergeChildrenUB can again be parallelized, or whether different processes are doing different instances
        of this function.
        :return:
        """
        if mpi_wrapper.is_first_process() or singleProcess:
            nodeInds, times = zip(*self.root.getTs({}).items())
            t0 = np.minimum(np.maximum(np.array(times), 1e-4), 1e4)
            bounds = ((-13.815510557964274, 13.815510557964274),) * len(t0)  # np.log(1e-6) = -13.815510557964274
            optRes = minimize(self.getNegLikelihoodGradGivenLogTimesWrapper, np.log(t0),
                              args=(nodeInds, verbose, mem_friendly,), tol=tol,
                              options={'disp': False, 'maxiter': maxiter}, jac=True, bounds=bounds)
            optTimes = np.exp(optRes.x)
            optTimes[optTimes < (1e-6 + 1e-8)] = 0.
        else:
            optTimes = None
            nodeInds = None
        if not singleProcess:
            nodeInds = mpi_wrapper.bcast(nodeInds, root=0)
            optTimes = mpi_wrapper.bcast(optTimes, root=0)

        optTimes = dict(zip(nodeInds, optTimes))
        self.root.assignTs(optTimes)
        self.root.getLtqsComplete(mem_friendly=True)
        return optTimes

    # Used
    def optTimes_single_scalar(self, verbose=False, singleProcess=False, mem_friendly=True, maxiter=1e6, tol=1e-6):
        """

        :param verbose:
        :param singleProcess: This boolean variable determines whether many processes work together to calculate such
        that mergeChildrenUB can again be parallelized, or whether different processes are doing different instances
        of this function.
        :return:
        """
        if mpi_wrapper.is_first_process() or singleProcess:
            nodeInds, times = zip(*self.root.getTs({}).items())
            t0 = np.minimum(np.maximum(np.array(times), 1e-4), 1e4)
            loglamb0 = 0
            bounds = ((-13.815510557964274, 13.815510557964274),)  # np.log(1e-6) = -13.815510557964274
            optRes = minimize(self.getNegLikelihoodGradGivenLogTimesWrapper_single_scalar, loglamb0,
                              args=(np.log(t0), nodeInds, verbose, mem_friendly,), tol=tol,
                              options={'disp': False, 'maxiter': maxiter}, jac=True, bounds=bounds)
            opt_lamb = np.exp(optRes.x)
            optTimes = opt_lamb * t0
            optTimes[optTimes < (1e-6 + 1e-8)] = 0.
        else:
            optTimes = None
            nodeInds = None
        if not singleProcess:
            nodeInds = mpi_wrapper.bcast(nodeInds, root=0)
            optTimes = mpi_wrapper.bcast(optTimes, root=0)

        optTimes = dict(zip(nodeInds, optTimes))
        self.root.assignTs(optTimes)
        self.root.getLtqsComplete(mem_friendly=True)
        return optTimes

    # Used
    def optGeneVars(self, geneVars, genesSimultaneously=False, verbose=False, singleProcess=False):
        """

        :param genesSimultaneously: boolean. Doing genes simultaneous is faster in this implementation, but can be a bit
        less accurate.
        :param verbose:
        :param singleProcess: This boolean variable determines whether many processes work together to calculate such
        that mergeChildrenUB can again be parallelized, or whether different processes are doing different instances
        of this function.
        :return:
        """
        if mpi_wrapper.is_first_process() or singleProcess:
            nGenes = bs_glob.nGenes
            logLambda0 = np.zeros(nGenes)

            bounds = np.array([[-13.815510557964274, 13.815510557964274]] * nGenes) - np.log(
                geneVars[:, None])  # np.log(1e-6) = -13.815510557964274
            # Try all genes at the same time
            if genesSimultaneously:
                start = time.time()
                optRes2 = minimize(self.logLGradCompleteLambda, x0=logLambda0, bounds=bounds,
                                   jac=True, args=(-1, bs_glob.nCells))
                optLambdas = np.exp(optRes2.x)
                mp_print("Variance optimization (genes simultaneous) took: %.5f seconds." % (time.time() - start))
            else:
                start = time.time()
                optLambdas = np.ones(nGenes)
                for gene_ind in range(nGenes):
                    bounds = np.array([[-13.815510557964274, 13.815510557964274]]) - np.log(
                        geneVars[gene_ind])  # np.log(1e-6)
                    optRes = minimize(self.logLGradCompleteLambda, x0=logLambda0[gene_ind], bounds=bounds,
                                      jac=True, args=(gene_ind, bs_glob.nCells))
                    if verbose and (gene_ind % 100 == 0):
                        mp_print("Completed gene optimization on gene %d out of %d." % (gene_ind, nGenes),
                                 ALL_RANKS=singleProcess)
                    if optRes.success:
                        optLambdas[gene_ind] = np.exp(optRes.x)
                    else:
                        mp_print("Optimizing variance of gene %d did not succeed. Re-setting to old value.")
                        optLambdas[gene_ind] = 1
                mp_print("Variance optimization (per gene) took: %.5f seconds." % (time.time() - start))

            # optRes = minimize(self.getNegLoglikGivenLambdasWrapper, np.log(logLambda0), args=(verbose,),
            #                   options={'disp': False}, jac=True, bounds=bounds)
        else:
            optLambdas = None
        if not singleProcess:
            optLambdas = mpi_wrapper.bcast(optLambdas, root=0)

        self.root.assignVs(optLambdas)
        optGeneVars = geneVars * optLambdas
        self.root.getLtqsComplete(mem_friendly=True)
        return optGeneVars

    def processOptReconfig(self, optEdgeList, nodeIndToNode, nodesList, candidatesList, mostUSInfo,
                           trackCloseness=True):
        mostUSNode = nodeIndToNode[mostUSInfo['mostUSNodeInd']]
        mostUSNode.childNodes = [child for child in mostUSNode.childNodes if
                                 child.nodeInd in mostUSInfo['childrenNotInvolved']]
        if trackCloseness:
            mostUSNode.cumClosenessNNN = 2
            mostUSNode.addClosenessNNN(dist=0, src=optEdgeList[0][1])
        for edge in optEdgeList:
            parentNode = nodeIndToNode[edge[0]]
            if edge[1] not in nodeIndToNode:
                childNode = TreeNode(nodeInd=edge[1], childNodes=[], isLeaf=False, isRoot=False, isCell=False)
                nodesList.append(childNode)
                nodeIndToNode[edge[1]] = childNode
                candidatesList.append(childNode)
                self.nNodes += 1
            childNode = nodeIndToNode[edge[1]]
            childNode.tParent = edge[2]
            childNode.parentNode = parentNode
            if trackCloseness:
                childNode.cumClosenessNNN = 2
            if not edge[3]:  # In this case, the childNode is not a leaf in the subtree and ltqs must be recomputed
                childNode.ltqs = None
                childNode.childNodes = []
            else:
                if trackCloseness:
                    childNode.addClosenessNNN(dist=0, src=parentNode.nodeInd)
            parentNode.childNodes.append(childNode)

        # Recalculate ltqs of all ancestors in the subtree
        mostUSNode.getLtqsNoneOnly()
        # Recalculate ltqs of everything that is upstream of subtree
        mostUSNode.setLtqsUpstream()
        # TODO: Do this only when zerotimechild is added or some edge is removed
        recursionWrap(self.root.mergeZeroTimeChilds)
        recursionWrap(self.root.renumberNodes)
        # self.root.mergeZeroTimeChilds(verbose=verbose)
        # self.root.renumberNodes()
        self.nNodes = bs_glob.nNodes

    def getOptEdgeList(self, dsNode, args, finalOptTimes=False, returnEdgelist=False, singleProcess=True, random=False,
                       trackCloseness=True, mem_friendly=False):
        """

        :param dsNode:
        :param args:
        :param finalOptTimes:
        :param returnEdgelist:
        :param singleProcess: This boolean variable determines whether many processes work together to calculate such
        that mergeChildrenUB can again be parallelized, or whether different processes are doing different instances
        of this function.
        :return:
        """
        # Pick a node (dsNode: downstreamNode) out of the list, then take its parent (usNode) and all
        # nodes connected to these (which may also include a parent that we then call mostUSNode).
        # With these next-nearest-neighbours, re-build the sub-tree, then check if you can improve this sub-tree.
        # This mostUSNode will be the starting point in returning the changed sub-tree in the larger tree. Therefore,
        # we need to store which children of the mostUSNode are being reconfigured, and which are not
        processYN, tree, usNode, dsNodeCopy, mostUSNode, \
                treeIndToOrigInd, mostUSInfo = self.buildTreeNNN(dsNode, returnMostUSInfo=returnEdgelist,
                                                                 mem_friendly=mem_friendly, random=random)
        if not processYN:
            if returnEdgelist:
                return None, None, None
            else:
                return None, None
        topology, centerLeafInd = tree.root.getTotalTopology()
        # Calculate the original loglikelihood from this subtree
        origLoglik = tree.calcLogLComplete()
        # Reconfigure the tree into a star-tree
        for child in dsNodeCopy.childNodes:
            child.tParent += dsNodeCopy.tParent
        tree.root.childNodes = tree.root.childNodes[1:] + tree.root.childNodes[0].childNodes
        bs_glob.nNodes -= 1

        # Do mergeChildren-routine to greedily approximate the optimal tree for this small startree
        tree.optTimes(verbose=False, singleProcess=singleProcess, tol=1e-4)  # TODO: Test if you want to do this.
        tree.root.mergeChildrenUB(tree.root.ltqs, tree.root.getW(), sequential=True,
                                  verbose=False, singleProcess=singleProcess, random=random)

        if trackCloseness:
            dsNode.cumClosenessNNN = 0.

        tree.root.mergeZeroTimeChilds()
        # Get topology of resulting tree
        # Get topology of the subtree, starting from the most upstream node (either usNode or its parent)
        newTopology, centerLeafInd = tree.root.getTotalTopology(centerLeafInd=centerLeafInd)

        if finalOptTimes:
            tree.optTimes(verbose=False, singleProcess=singleProcess,
                          tol=1e-6)  # TODO: Maybe only do this when merge is done.

        newLoglik = tree.calcLogLComplete()
        dsNode.nnnLoglik = newLoglik - origLoglik

        if not returnEdgelist:
            return dsNode.nnnLoglik, topology != newTopology
        # tree.root.storeParent()  # This is already done in getTotalTopology
        edgeList = mostUSNode.getEdgesComplete([], indMap=treeIndToOrigInd)
        return dsNode.nnnLoglik, edgeList, mostUSInfo

    def buildTreeNNN(self, dsNode, returnMostUSInfo=False, mem_friendly=True, random=False):
        dsNeighbours = dsNode.childNodes
        usNode = dsNode.parentNode
        usNeighbours = [node for node in usNode.childNodes if node.nodeInd is not dsNode.nodeInd]

        # TODO: Clean this up
        if (not random) and ((len(dsNeighbours) + len(usNeighbours)) > 30):
            return False, None, None, None, None, None, None
        if mem_friendly:
            dsNode.getAIRootUpstream()
        dsNeighboursLtqs = np.array([child.ltqs for child in dsNeighbours]).T
        dsNeighboursLtqsVars = np.array([child.getLtqsVars() for child in dsNeighbours]).T
        dsTParents = [child.tParent for child in dsNeighbours]

        usNeighboursLtqs = np.array([child.ltqs for child in usNeighbours]).T
        usNeighboursLtqsVars = np.array([child.getLtqsVars() for child in usNeighbours]).T
        usTParents = [child.tParent for child in usNeighbours]
        mostUSInfo = None
        if usNode.parentNode is not None:
            parent = usNode.parentNode
            usTParents.append(usNode.tParent)
            usNeighbours.append(parent)
            if returnMostUSInfo:
                mostUSInfo = {'mostUSNodeInd': parent.nodeInd,
                              'childrenNotInvolved': [ch.nodeInd for ch in parent.childNodes if
                                                      not (ch.nodeInd == usNode.nodeInd)]}
            # Get ltqs of parent where contribution of child is subtracted out
            wbarChild_g = 1 / (usNode.tParent + usNode.getLtqsVars())
            WWOChild = parent.getW(AIRoot=True) - wbarChild_g
            ltqsWOChild = (parent.ltqsAIRoot * parent.getW(AIRoot=True) - wbarChild_g * usNode.ltqs) / WWOChild
            ltqsVarsWOChild = 1 / WWOChild
            usNeighboursLtqs = np.hstack((usNeighboursLtqs, ltqsWOChild[:, None]))
            usNeighboursLtqsVars = np.hstack((usNeighboursLtqsVars, ltqsVarsWOChild[:, None]))
        elif returnMostUSInfo:
            mostUSInfo = {'mostUSNodeInd': usNode.nodeInd, 'childrenNotInvolved': []}

        # Create a new tree object where the usNode is the root and the other nodes are summarizing their downstream
        # information
        nDSnb = len(dsNeighbours)
        nUSnb = len(usNeighbours)
        tree = Tree()
        bs_glob.nNodes = 2 + nDSnb + nUSnb
        treeIndToOrigInd = {-1: usNode.nodeInd}
        tree.root = TreeNode(nodeInd=-1, childNodes=[], isLeaf=False, isRoot=True,
                             ltqs=None, ltqsVars=None, isCell=usNode.isCell)
        treeIndToOrigInd[bs_glob.nNodes - 2] = dsNode.nodeInd
        dsNodeCopy = TreeNode(nodeInd=bs_glob.nNodes - 2, childNodes=[], isLeaf=False, isRoot=False,
                              ltqs=None, ltqsVars=None, isCell=False,
                              tParent=dsNode.tParent, parentNode=tree.root)
        tree.root.childNodes.append(dsNodeCopy)
        for ind, nb in enumerate(usNeighbours):
            treeIndToOrigInd[ind] = nb.nodeInd
            tree.root.childNodes.append(TreeNode(nodeInd=ind, childNodes=[], isLeaf=True, isRoot=False,
                                                 ltqs=usNeighboursLtqs[:, ind], ltqsVars=usNeighboursLtqsVars[:, ind],
                                                 isCell=True, tParent=usTParents[ind]))
        mostUSNode = tree.root.childNodes[-1] if (usNode.parentNode is not None) else tree.root

        for ind, nb in enumerate(dsNeighbours):
            treeIndToOrigInd[ind + nUSnb] = nb.nodeInd
            dsNodeCopy.childNodes.append(TreeNode(nodeInd=ind + nUSnb, childNodes=[], isLeaf=True, isRoot=False,
                                                  ltqs=dsNeighboursLtqs[:, ind], ltqsVars=dsNeighboursLtqsVars[:, ind],
                                                  isCell=True, tParent=dsTParents[ind]))
        # How many ancestors do we expect: for n leaves, we have max. n-2 ancestors. So add some nodeInds if necessary
        for ind in range(nDSnb + nUSnb - 4):
            treeIndToOrigInd[ind + nDSnb + nUSnb + 1] = self.nNodes - 1 + ind
        return True, tree, usNode, dsNodeCopy, mostUSNode, treeIndToOrigInd, mostUSInfo

    def get_vert_ind_to_node_DF(self, update=False):
        if (self.vert_ind_to_node is None) or update:
            vert_ind_to_node = {}
            self.vert_ind_to_node = self.root.get_vert_ind_to_node_node(vert_ind_to_node)
        return self.vert_ind_to_node

    def set_midpoint_root(self):
        # Find longest path below each internal node (discard branches to leafs)
        # Store maximum for each node in some dictionary
        self.root.get_longest_path_ds()

        # Go to node with longest path. Find pair of nodes on path with root in-between
        vert_ind_to_node = self.get_vert_ind_to_node_DF(update=True)
        max_dist_node_tuple = (None, 0)
        for _, node in vert_ind_to_node.items():
            if node.max_summed_dist_ds > max_dist_node_tuple[1]:
                max_dist_node_tuple = (node, node.max_summed_dist_ds)
        max_dist_node, max_summed_dist = max_dist_node_tuple

        # Add new node in-between these pairs at right position
        the_parent, child_ind = max_dist_node.find_midpoint_pair(max_summed_dist)

        # Change child-nodes structure to get this new root
        the_child = the_parent.childNodes[child_ind]
        delta = max_summed_dist / 2 - the_child.max_dist_ds
        # Create new-node that is going to be in-between the_parent and the_child
        midpoint_root_node = TreeNode(vert_ind=-1, nodeInd=-100, childNodes=[the_child], parentNode=the_parent,
                                      isLeaf=False, isRoot=False, tParent=the_child.tParent - delta,
                                      nodeId="midpoint_root", isCell=False)

        # Delete the_child from the_parent
        the_parent.childNodes = [parent_child for curr_ind, parent_child in enumerate(the_parent.childNodes) if
                                 curr_ind != child_ind] + [midpoint_root_node]
        the_child.parentNode = midpoint_root_node
        the_child.tParent = delta

        # Renumber vert_inds on tree such that they are in line with a depth-first search
        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0, include_nodeInd=True)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()

        self.reset_root(new_root_ind=midpoint_root_node.vert_ind)

        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0, include_nodeInd=True)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()

    def set_mindist_root(self, cell_ids):
        # Find parent-node "the_parent" and index of "the_child" that are on both ends of the branch that would be cut
        # to minimize the sum of pairwise distances
        nwk_str = self.to_newick(use_ids=True)
        _, mindist_edge = get_min_pdists_clustering_from_nwk_str(tree_nwk_str=nwk_str, n_clusters=2, cell_ids=cell_ids,
                                                                 get_cell_ids_all_splits=False, node_id_to_n_cells=None,
                                                                 verbose=False)
        child_node_id, parent_node_id = mindist_edge[0]
        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0, include_nodeInd=True)
        for vert_ind, node in vertIndToNode.items():
            if node.nodeId == parent_node_id:
                the_parent = node
            elif node.nodeId == child_node_id:
                the_child = node
        # Check if child is indeed in childNodes of the parent
        child_inds = [ind for ind, child in enumerate(the_parent.childNodes) if child.nodeId == child_node_id]
        if len(child_inds) == 1:
            child_ind = child_inds[0]
        else:
            child_inds = [ind for ind, child in enumerate(the_child.childNodes) if child.nodeId == parent_node_id]
            if len(child_inds) == 1:
                the_parent = the_child
                child_ind = child_inds[0]
            else:
                # If this doesn't succeed, there was an error. We return False, signifying that setting root failed.
                return False

        # Change child-nodes structure to get this new root
        the_child = the_parent.childNodes[child_ind]
        delta = the_child.tParent / 2
        # Create new-node that is going to be in-between the_parent and the_child
        midpoint_root_node = TreeNode(vert_ind=-1, nodeInd=-100, childNodes=[the_child], parentNode=the_parent,
                                      isLeaf=False, isRoot=False, tParent=the_child.tParent - delta,
                                      nodeId="mindist_root", isCell=False)

        # Delete the_child from the_parent
        the_parent.childNodes = [parent_child for curr_ind, parent_child in enumerate(the_parent.childNodes) if
                                 curr_ind != child_ind] + [midpoint_root_node]
        the_child.parentNode = midpoint_root_node
        the_child.tParent = delta

        # Renumber vert_inds on tree such that they are in line with a depth-first search
        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0, include_nodeInd=True)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()

        self.reset_root(new_root_ind=midpoint_root_node.vert_ind)

        vertIndToNode, self.nNodes = self.root.renumber_verts(vertIndToNode={}, vert_count=0, include_nodeInd=True)
        self.vert_ind_to_node = vertIndToNode
        self.root.storeParent()
        return True

    def reset_root(self, new_root_ind):
        vert_ind_to_node = self.get_vert_ind_to_node_DF()
        if list(vert_ind_to_node.values())[1].parentNode is None:
            self.root.storeParent()
        self.root = vert_ind_to_node[new_root_ind]
        # self.root.isRoot = True
        # self.root.tParent = None
        self.root.reset_root_node(parent_ind=None, old_tParent=self.root.tParent)

    def from_newick(self, nwk_str, node_id_to_vert_ind=None):
        # nodes = []
        # Here we will keep a dictionary from level down the tree, to lists of nodes that don't have an assigned parent
        level_to_nodes = {0: []}
        level = 0
        curr_str = ''
        nodeIdY_tParentF = True
        curr_node = None
        new_node = False
        self.nNodes = 0
        internal_node_ids_made_up = 0
        for char in nwk_str:
            if char == '(':
                # We go down the tree one level
                level += 1
                level_to_nodes[level] = []
                # It also means we start a new node
                new_node = True
            elif char == ')':
                # We go back up to the parent level
                level -= 1
                # It also means we start a new node
                new_node = True
            elif char == ':':
                # This indicates a time is coming
                if curr_node is None:
                    # This indicates no node-id is stored in the nwk-file for this node. Make one up
                    curr_node = TreeNode(isLeaf=True)
                    self.nNodes += 1
                    curr_node.level = level
                    level_to_nodes[level].append(curr_node)
                    # nodeIdY_tParentF = True
                    curr_str = 'internal_{}'.format(internal_node_ids_made_up)
                    internal_node_ids_made_up += 1
                    new_node = False

                curr_node.nodeId = curr_str
                if (node_id_to_vert_ind is not None) and (curr_node.nodeId in node_id_to_vert_ind):
                    curr_node.vert_ind = node_id_to_vert_ind[curr_node.nodeId]
                nodeIdY_tParentF = False
                curr_str = ''
            elif char == ',':
                # This indicates the time is complete, new child coming
                new_node = True
            elif char == ';':
                if curr_node is None:
                    # This indicates no node-id is stored in the nwk-file for this node. Make one up
                    curr_node = TreeNode(isLeaf=True)
                    self.nNodes += 1
                    curr_node.level = level
                    level_to_nodes[level].append(curr_node)
                    # nodeIdY_tParentF = True
                    curr_str = 'internal_{}'.format(internal_node_ids_made_up)
                    internal_node_ids_made_up += 1
                    nodeIdY_tParentF = True
                curr_node.isRoot = True
                new_node = True
            else:
                if curr_node is None:
                    curr_node = TreeNode(isLeaf=True)
                    self.nNodes += 1
                    curr_node.level = level
                    level_to_nodes[level].append(curr_node)
                    nodeIdY_tParentF = True
                curr_str += char
                new_node = False

            if new_node:
                if curr_node is not None:
                    if nodeIdY_tParentF:
                        curr_node.nodeId = curr_str
                        if (node_id_to_vert_ind is not None) and (curr_node.nodeId in node_id_to_vert_ind):
                            curr_node.vert_ind = node_id_to_vert_ind[curr_node.nodeId]
                    else:
                        curr_node.tParent = float(curr_str)
                    curr_str = ''
                    # level_to_nodes[level].append(curr_node)
                    # Get all children and add them to this node
                    if (curr_node.level + 1) in level_to_nodes:
                        curr_node.childNodes = level_to_nodes[curr_node.level + 1]
                        if len(level_to_nodes[curr_node.level + 1]) > 0:
                            curr_node.isLeaf = False
                            level_to_nodes[curr_node.level + 1] = []
                    else:
                        curr_node.childNodes = []
                curr_node = None
        self.root = level_to_nodes[0][0]


def getNewUBInfo(xrAsIfRoot_g, WAsIfRoot_g, epsx, epsW, alldLogLsUB, UBInfo, allPairsUB, oldRoot, del_node_inds,
                 nChild=None, kNN=None, pairsDoneInfo=None, ellipsoidSize=None, verbose=False, newUBObtained=None,
                 mpiInfo=None):
    # First add all newly computed upper bounds together
    sortedUBs = np.zeros(0)
    sortedPairs = np.zeros((0, 2), dtype=int)
    for rankInd, dLogLUBs in enumerate(alldLogLsUB):
        if len(dLogLUBs):
            nUBs = len(sortedUBs)
            indices = nUBs - np.searchsorted(sortedUBs, dLogLUBs, sorter=np.arange(nUBs - 1, -1, -1))
            sortedUBs = np.insert(sortedUBs, indices, dLogLUBs)
            sortedPairs = np.insert(sortedPairs, indices, allPairsUB[rankInd], axis=0)

    indices = len(UBInfo['UBs']) - np.searchsorted(UBInfo['UBs'], sortedUBs,
                                                   sorter=np.arange(len(UBInfo['UBs']) - 1, -1, -1))

    UBInfo['UBs'] = np.insert(UBInfo['UBs'], indices, sortedUBs)
    UBInfo['pairs'] = np.insert(UBInfo['pairs'], indices, sortedPairs, axis=0)

    for node in del_node_inds:
        toBeDeleted = np.where(UBInfo['pairs'] == node)[0]
        UBInfo['pairs'] = np.delete(UBInfo['pairs'], toBeDeleted, axis=0)
        UBInfo['UBs'] = np.delete(UBInfo['UBs'], toBeDeleted)

    # Test if root moved out of ellipsoid, because then we need to calculate new UBs
    getNewUB = (np.linalg.norm((oldRoot['pos'] - xrAsIfRoot_g) * np.sqrt(oldRoot['prec']),
                               2) > epsx) or (
                       np.linalg.norm((oldRoot['prec'] - WAsIfRoot_g) / oldRoot['prec'],
                                      2) > epsW)
    if not newUBObtained:
        pairsDoneInfo['pairsDoneList'].append(pairsDoneInfo['totalPairsDone'])
        if len(pairsDoneInfo['pairsDoneList']) > 10:
            avgPairsDone = np.mean(pairsDoneInfo['pairsDoneList'][-10:])
            if avgPairsDone > 0.20 * kNN * nChild / mpiInfo.size:
                getNewUB = True
                ellipsoidSize = ellipsoidSize / 2
                # nChild * ellipsoidSize is the number of targeted steps before new UBs, we want this to be at least 2
                ellipsoidSize = min(max(ellipsoidSize, 2 / nChild, 0.1), 10)
                if verbose:
                    mp_print("Too many pairs were calculated using UB. "
                             "Calculating new upper bounds, and decreasing ellipsoid size. "
                             "Ellipsoid size now %.2f" % ellipsoidSize)
            elif getNewUB and (avgPairsDone < kNN * 5):
                ellipsoidSize *= 2
                # nChild * ellipsoidSize is the number of targeted steps before new UBs, we want this to be at least 2
                ellipsoidSize = min(max(ellipsoidSize, 2 / nChild, 0.1), 10)
                if verbose:
                    mp_print("Very few pairs were calculated per round. "
                             "Ellipsoid size for UB-estimation will be increased. "
                             "Ellipsoid size now %.2f" % ellipsoidSize)
    return UBInfo, getNewUB, ellipsoidSize


def communicateMaxs(maxdLogL, maxdLogLZeroRoot, foundMax, task, nNewPairs, UBInfo, mpiInfo):
    if mpiInfo.size > 1:
        maxdLogLTuples = mpi_wrapper.world_allgather(maxdLogL)
        maxdLogLZeroRootTuples = mpi_wrapper.world_allgather(maxdLogLZeroRoot)
        ind1s, ind2s, maxdLogLs = zip(*maxdLogLTuples)
        ind1sZeroRoot, ind2sZeroRoot, maxdLogLsZeroRoot = zip(*maxdLogLZeroRootTuples)
        maxdLogL = maxdLogLTuples[np.argmax(maxdLogLs)]
        maxdLogLZeroRoot = maxdLogLZeroRootTuples[np.argmax(maxdLogLsZeroRoot)]
        if not foundMax:
            foundMax = (maxdLogL[2] > UBInfo['UBs'][task - nNewPairs])
        foundMaxs = mpi_wrapper.world_allgather(foundMax)
        allFoundMax = min(foundMaxs)
        return foundMax, allFoundMax, maxdLogL, maxdLogLZeroRoot
    else:
        return foundMax, foundMax, maxdLogL, maxdLogLZeroRoot


def initializeTask(indTask, myTasks, pairs, nodeIndToChildInd, verbose, mpiInfo, runConfigs):
    task = myTasks[indTask]
    # Get indices of nodes
    nodeInd1, nodeInd2 = pairs[task]
    # Get corresponding indices in list self.childNodes
    ind1 = nodeIndToChildInd[nodeInd1] if nodeInd1 in nodeIndToChildInd else None
    ind2 = nodeIndToChildInd[nodeInd2] if nodeInd2 in nodeIndToChildInd else None
    skip = True if ((ind1 is None) or (ind2 is None)) else False  # Happens when some node was already merged, but
    # still appears in pairs-list. It seems faster to skip over it here than to update pairs-list
    if (not skip) and (ind1 == ind2):
        skip = True
        logging.error("nodeInd1: {}, nodeInd2: {}".format(nodeInd1, nodeInd2))
        logging.error(
            "There's a mistake somewhere. We are trying to calculate the merging likelihood of a node with itself. "
            "This is now skipped!")
    if skip:
        mp_print("A node-ind showed up in a candidate merge-pair that is no longer one of the root-children. This"
                 "should be impossible. Candidate pairs: ({}, {})".format(nodeInd1, nodeInd2), ERROR=True)
        return skip, task, nodeInd1, nodeInd2, ind1, ind2
    # Print some information on progress
    if verbose and ((indTask + 1) % 5000 == 0):
        mp_print("Process %d: Calculating merger loglikelihoods for pair (%d,%d), "
                 "calculation %d out of%s %.0f."
                 % (mpiInfo.rank, ind1, ind2, indTask + 1, " (potentially)" if runConfigs['useUBNow'] else "",
                    len(myTasks)),
                 ALL_RANKS=True)
    return skip, task, nodeInd1, nodeInd2, ind1, ind2


# TODO: Check if this can be improved
# Used
def findOptimalTSingleCell(ltqs_g, ltqsVars_g, xr_g, t0, W_g):
    # We here find the root of the derivative of the star-tree likelihood w.r.t. to ti. We are almost certain that this
    # is a decreasing function in ti. We therefore first check the derivative at ti=0
    sqDists_g = (xr_g - ltqs_g) ** 2
    derAtZero = np.dot(1 / ltqsVars_g, sqDists_g / ltqsVars_g - 1 + 1 / (ltqsVars_g * W_g))
    if derAtZero < 0:
        return 0.
    else:
        # The derivative is positive at ti=0. Now find ti for which the derivative is negative, so that we can start
        # the brentq-algorithm
        tUBNotFound = True
        tLB = 0.
        tUB = max(t0, 0.01)
        while tUBNotFound:
            if derLogLikStar(tUB, ltqsVars_g, sqDists_g, W_g) < 0:
                tUBNotFound = False
            else:
                tLB = tUB
                tUB *= 10
        tOpt = brentq(derLogLikStar, tLB, tUB, args=(ltqsVars_g, sqDists_g, W_g), xtol=1e-4, rtol=1e-4, maxiter=100,
                      full_output=False, disp=True)
    return tOpt


# Used
def derLogLikStar(t, ltqsVars_g, sqDists_g, W_g):
    tbar_g = ltqsVars_g + t
    wbar_g = 1 / tbar_g
    return np.dot(wbar_g, wbar_g * sqDists_g - 1 + 1 / (tbar_g * W_g))


# Used
def optimiseT3LeafStar(ltqs_gi, ltqsVars_gi, t0_i, verbose=False):
    optRes = minimize(getLogLikAndGradStarTreeWrapper, np.log(np.maximum(t0_i, 1e-4)), args=(ltqsVars_gi, ltqs_gi),
                      bounds=((-16.1180, 10), (-16.1180, 10), (-16.1180, 10)), jac=True, tol=None)

    # TODO: Decide if using Hessian, seems to not do much and may be problematic at t=0
    # negHess = getHessStarTreeJax_jit(optRes.x, ltqsVars_gi, ltqs_gi)
    # detNegHess = np.linalg.det(negHess)
    # optLoglik = -optRes.fun - (np.log(detNegHess) / 2)
    optLoglik = -optRes.fun
    return optLoglik, np.exp(optRes.x), optRes.success


# This may be used in the future
# def getHessStarTreeJax(t, ltqsVars_gi, ltqs_gi):
#     hessFun = jax.jacfwd(jax.jacrev(logLStarTreeJax))
#     return hessFun(t, ltqsVars_gi, ltqs_gi)


# def logLStarTreeJax(t_i, ltqsVars_gi, ltqs_gi):
#     wbar_gi = 1 / (ltqsVars_gi + t_i)
#     W_g = jnp.sum(wbar_gi, axis=1)
#
#     wOverW_gi = jnp.divide(wbar_gi, W_g[:, None])
#     xr_g = jnp.sum(jnp.multiply(wOverW_gi, ltqs_gi), axis=1)
#     sqdistsWbar_gi = jnp.multiply(wbar_gi, (xr_g[:, None] - ltqs_gi) ** 2)
#     loglik = jnp.sum(jnp.sum(jnp.log(wbar_gi), axis=1) - jnp.log(W_g) - jnp.sum(sqdistsWbar_gi, axis=1))
#     # grad = np.sum(wbar_gi * (sqdistsWbar_gi - 1 + wOverW_gi), axis=0)
#     return -loglik


# Used
def optimiseT3LeafStarSequential(ltqs_gi, ltqsVars_gi, t0_i, tol=None, verbose=False):
    # Optimise t12, total diffusion time between the two cells as if these are not connected to the root
    # if t12Opt is None:
    t12Opt, converged = getOptTime2LeafTree(ltqs_gi, ltqsVars_gi, tol=tol)
    if not converged:
        return None, None, None, False
    # Fix t12 = t1a + t2a and optimise t1a < t12 and tar.
    # We will optimize t1a on a linear scale, and tar on log-scale
    optRes = minimize(getLogLikAndGradStarTreeSequentialWrapper, [0.5 * t12Opt, np.log(max(t0_i[1], 1e-4))],
                      args=(t12Opt, ltqsVars_gi, ltqs_gi), bounds=((0, t12Opt), (-16.1180, 10)),
                      jac=True, tol=tol)
    optRes.x[1] = np.exp(optRes.x[1])
    return -optRes.fun, optRes.x, t12Opt, optRes.success


# Used
# def optimiseT3LeafStarSequentialOld(ltqs_gi, ltqsVars_gi, t0_i, tol=None, verbose=False):
#     # Optimise t12, total diffusion time between the two cells as if these are not connected to the root
#     # if t12Opt is None:
#     t12Opt, converged = getOptTime2LeafTree(ltqs_gi, ltqsVars_gi, tol=tol)
#     if not converged:
#         return None, None, None, False
#     # Fix t12 = t1a + t2a and optimise t1a < t12 and tar.
#     # We will optimize t1a on a linear scale, and tar on log-scale
#     optRes = minimize(getLogLikAndGradStarTreeSequentialWrapper, [0.5 * t12Opt, np.log(max(t0_i[1], 1e-4))],
#                       args=(t12Opt, ltqsVars_gi, ltqs_gi), bounds=((0, t12Opt), (np.log(1e-7), np.log(1e7))),
#                       jac=True, tol=tol)
#     optRes.x[1] = np.exp(optRes.x[1])
#     return -optRes.fun, optRes.x, t12Opt, optRes.success


# Used
def getOptTime2LeafTree(ltqs_gi, ltqsVars_gi, tol=1e-7):
    summedLtqsVars_g = ltqsVars_gi[:, 0] + ltqsVars_gi[:, 1]
    sqDists_g = (ltqs_gi[:, 0] - ltqs_gi[:, 1]) ** 2
    lb_bracket = 0
    ub_bracket = lb_bracket + 1
    value_lb = der2LeafTree(lb_bracket, summedLtqsVars_g, sqDists_g)
    if value_lb <= 0:
        # First derivative at lower bound is negative, so lower bound is (local) optimum
        return 0, True

    value_ub = der2LeafTree(ub_bracket, summedLtqsVars_g, sqDists_g)
    bracket_counter = 0
    while value_ub >= 0:
        bracket_counter += 1
        if bracket_counter > 10:
            mp_print("Bisection method could not find good starting bracket in first 10 trys.\n"
                     "Lower bound is now {} (value={}), upper bound is now {} (value={})".format(lb_bracket, value_lb,
                                                                                                 ub_bracket, value_ub),
                     ALL_RANKS=True, ERROR=True)
        lb_bracket = ub_bracket
        ub_bracket *= 10
        value_ub = der2LeafTree(ub_bracket, summedLtqsVars_g, sqDists_g)
        if lb_bracket > 1e6:
            return None, False
        if value_ub > 1e9:
            exit()
    opt_dist, opt_sol = brentq(der2LeafTree, lb_bracket, ub_bracket,
                               (summedLtqsVars_g, sqDists_g),
                               rtol=tol, full_output=True)
    return opt_sol.root, opt_sol.converged


# Used
def der2LeafTree(t12, summedLtqsVars_g, sqDists_g):
    totVar_g = t12 + summedLtqsVars_g
    der = np.sum((-1 + sqDists_g / totVar_g) / totVar_g)
    return der


# Used
def logLGradStarTreeLogT(logt_i, ltqsVars_gi, ltqs_gi):
    t_i = np.exp(logt_i)
    wbar_gi = 1 / (ltqsVars_gi + t_i)
    W_g = np.sum(wbar_gi, axis=1)
    wOverW_gi = np.divide(wbar_gi, W_g[:, None])
    xr_g = np.sum(np.multiply(wOverW_gi, ltqs_gi), axis=1)
    sqdists_gi = (xr_g[:, None] - ltqs_gi) ** 2
    loglik = np.sum(np.sum(np.log(wbar_gi), axis=1) - np.log(W_g) - np.sum(np.multiply(wbar_gi, sqdists_gi), axis=1))
    grad = np.sum(wbar_gi * (sqdists_gi * wbar_gi - 1 + wOverW_gi), axis=0)
    # print("Loglik: " + str(loglik / bs_glob.nGenes))
    return -loglik, - t_i * grad


# Used
def logLGradStarTreeLogLambda(logLambda_g, t_i, ltqs_gi, ltqsVars_gi, nCells):
    lambda_g = np.exp(logLambda_g)

    ltqsNew_gi = ltqs_gi / np.sqrt(lambda_g[:, None])
    ltqsVarsNew_gi = ltqsVars_gi / lambda_g[:, None]
    xr_g, W_g, wbar_gi, wOverW_gi = findNodeLtqsGivenLeafs(ltqs_gi=ltqsNew_gi, ltqsVars_gi=ltqsVarsNew_gi, t_i=t_i,
                                                           return_wbar_gi=True, return_wOverW_gi=True)
    sqDistsTimesWbar_gi = wbar_gi * (ltqsNew_gi - xr_g[:, None]) ** 2

    loglik = np.sum(- nCells * logLambda_g - np.log(W_g) + np.sum(np.log(wbar_gi) - sqDistsTimesWbar_gi, axis=1))
    logGrad = - nCells - np.sum(wbar_gi * (ltqsVarsNew_gi * (wOverW_gi - 1) - t_i * sqDistsTimesWbar_gi), axis=1)

    # print("Loglik: " + str(loglik))
    return - loglik, - logGrad


# Used
def logLGradStarTreeLogLambdaSingleGene(logLambda, t_i, ltqs_i, ltqsVars_i, nCells):
    logLambda = logLambda[0]
    lam = np.exp(logLambda)

    ltqsNew_i = ltqs_i / np.sqrt(lam)
    ltqsVarsNew_i = ltqsVars_i / lam
    xr, W, wbar_i, wOverW_i = findNodeLtqsGivenLeafsSingleGene(ltqsNew_i, ltqsVarsNew_i, t_i, return_wbar_i=True,
                                                               return_wOverW_i=True)
    sqDistsTimesWbar_i = wbar_i * (ltqsNew_i - xr) ** 2

    loglik = - nCells * logLambda - np.log(W) + np.sum(np.log(wbar_i) - sqDistsTimesWbar_i)
    logGrad = - nCells - np.sum(wbar_i * (ltqsVarsNew_i * (wOverW_i - 1) - t_i * sqDistsTimesWbar_i))

    # print("Loglik: " + str(loglik))
    return - loglik, - logGrad


# Used
def optimiseTStar(ltqs_gi, ltqsVars_gi, nChilds=None, verbose=False):
    if nChilds is None:
        nChilds = bs_glob.nCells
    t_i = np.ones(nChilds)
    tnew_i = np.zeros(nChilds)
    converged = False

    # We first try to optimize the diffusion times using an EM-like scheme. If this doesn't work we use a gradient-
    # based approach
    max_steps = 1000
    step_counter = 0
    while (not converged) and (np.max(t_i) < 1e8) and (step_counter < max_steps):
        step_counter += 1
        xr_g, W_g = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)

        for ind in range(nChilds):
            tnew_i[ind] = findOptimalTSingleCell(ltqs_gi[:, ind], ltqsVars_gi[:, ind], xr_g, t_i[ind], W_g)
        changeInT = np.max(np.abs(tnew_i - t_i))
        if verbose:
            mp_print("Max absolute change in ti was: " + str(changeInT))
        if changeInT < 1e-4:
            converged = True
        t_i = tnew_i.copy()
    if not converged:
        mp_print("Iterative procedure for optimising diffusion times on star-tree diverged. "
                 "Trying gradient-based approach.")
        optRes = minimize(logLGradStarTreeLogT, x0=np.log(t_i), jac=True, args=(ltqsVars_gi, ltqs_gi))
        t_i = np.exp(optRes.x)
        if not optRes.success:
            mp_print("Could not optimize diffusion times for star-tree. "
                     "Maybe try including fewer noisy genes, using zscore_cutoff-argument.")
            t_i = np.ones(nChilds)
            xr_g, W_g = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i)
            return t_i, -1e9, W_g, xr_g

    # Found optimum. Now getting information to print and store
    xr_g, W_g, wbar_gi, wOverW_gi = findNodeLtqsGivenLeafs(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i,
                                                           return_wbar_gi=True,
                                                           return_wOverW_gi=True)
    loglik = getLoglikAndGradStarTree(ltqs_gi=ltqs_gi, xr_g=xr_g, wbar_gi=wbar_gi, W_g=W_g, returnGrad=False)
    if verbose:
        mp_print("LogL per Gene: " + str(loglik / bs_glob.nGenes))
        # Real loglikelihood is:
        # (loglik * 0.5) - (bs_glob.nGenes * (bs_glob.nCells - 1)/2)* np.log(2 * math.pi)
    return t_i, loglik, W_g, xr_g


# Used
def optimiseLambdaStar(t_i, ltqs_gi, ltqsVars_gi, geneVars, verbose=False):
    nGenes = bs_glob.nGenes
    nCells = bs_glob.nCells
    logLambda_g = np.zeros(nGenes)

    # bounds = ((-13.815510557964274, 13.815510557964274),) * nGenes  # np.log(1e-6) = -13.815510557964274
    # start = time.time()
    # optRes = minimize(logLGradStarTreeLogLambda, x0=logLambda_g, jac=True, bounds=bounds, args=(t_i, ltqs_gi, ltqsVars_gi, nCells))
    # lambda_g = np.exp(optRes.x)
    # mp_print("Variance optimization took: %.5f seconds." % (time.time() - start))
    # if not optRes.success:
    #     mp_print("Could not optimize gene variances for star-tree. "
    #              "Maybe try including fewer noisy genes, using zscore_cutoff-argument.")
    #     lambda_g = np.ones(nGenes)
    #     return lambda_g, ltqs_gi, ltqsVars_gi

    start = time.time()
    lambda_g = np.zeros(nGenes)
    for gene_ind in range(nGenes):
        bounds = np.array([[-13.815510557964274, 13.815510557964274]]) - np.log(geneVars[gene_ind])  # np.log(1e-6)
        optRes = minimize(logLGradStarTreeLogLambdaSingleGene, x0=logLambda_g[gene_ind], bounds=bounds, jac=True,
                          args=(t_i, ltqs_gi[gene_ind, :], ltqsVars_gi[gene_ind, :], nCells))
        if optRes.success:
            lambda_g[gene_ind] = np.exp(optRes.x)
        else:
            mp_print("Optimizing variance of gene %d did not succeed. Re-setting to old value.")
            lambda_g[gene_ind] = 1
    mp_print("Variance optimization (per gene) took: %.5f seconds." % (time.time() - start))
    return lambda_g


# Used
def getLoglikAndGradStarTree(childNodes=None, ltqs_gi=None, ltqsVars_gi=None, t_i=None, xr_g=None, wbar_gi=None,
                             W_g=None, returnGrad=False, mem_friendly=False):
    """
    :param childNodes: if childNodes is not None, this information about connected nodes is used
    :param ltqs_gi: only necessary when no childNodes. List of coordinates of children.
    :param ltqsVars_gi: only necessary when no childNodes. List of uncertainty-variances on coordinates of children
    :param t_i: only necessary when no childNodes. List of edge-lengths to children.
    :param xr_g: if provided, root position is not recalculated
    :param W_g: if provided, root precision is not recalculated
    :return:
    """
    if childNodes is None:
        mem_friendly = False
        wOverW_gi = None
    if xr_g is None:
        if mem_friendly:
            xr_g, W_g = findNodeLtqsGivenLeafs(childNodes=childNodes, ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i,
                                               return_wbar_gi=False, return_wOverW_gi=False)
            wbar_gi = None
        elif not returnGrad:
            xr_g, W_g, wbar_gi = findNodeLtqsGivenLeafs(childNodes=childNodes, ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi,
                                                        t_i=t_i, return_wbar_gi=True, return_wOverW_gi=False)
        else:
            xr_g, W_g, wbar_gi, wOverW_gi = findNodeLtqsGivenLeafs(childNodes=childNodes, ltqs_gi=ltqs_gi,
                                                                   ltqsVars_gi=ltqsVars_gi, t_i=t_i,
                                                                   return_wbar_gi=True, return_wOverW_gi=True)

    if childNodes is None:
        sqdists_gi = (xr_g[:, None] - ltqs_gi) ** 2
        sqdistsWbar_gi = np.multiply(wbar_gi, sqdists_gi)
        loglik = np.sum(np.sum(np.log(wbar_gi), axis=1) - np.log(W_g) - np.sum(sqdistsWbar_gi, axis=1))
    else:
        loglik = - np.sum(np.log(W_g))
        if returnGrad:
            grad = np.zeros(len(childNodes))
        for cInd, child in enumerate(childNodes):
            if wbar_gi is None:
                wbar_g = 1 / (child.getLtqsVars() + child.tParent)
            else:
                wbar_g = wbar_gi[:, cInd]
            loglik += np.sum(np.log(wbar_g))
            sqdistsWbar_g = np.multiply(wbar_g, (xr_g - child.ltqs) ** 2)
            loglik -= np.sum(sqdistsWbar_g)
            if returnGrad:
                grad[cInd] = np.sum(wbar_g * (sqdistsWbar_g - 1 + np.divide(wbar_g, W_g)))
    if not returnGrad:
        return loglik
    else:
        if childNodes is None:
            if wOverW_gi is None:
                wOverW_gi = np.divide(wbar_gi, W_g[:, None])
            grad = np.sum(wbar_gi * (sqdistsWbar_gi - 1 + wOverW_gi), axis=0)
        return loglik, grad


def estimateDerBasedDLogLUB(rootInfo, tOpt, ltqs_gi=None, ltqsVars_gi=None, wir_gi=None, child1=None,
                            child2=None, sequential=True):
    """
    Function that takes information about a pair and the root, and returns two values that are used to calculate an
    upper bound on the dLogL under the assumption that the root stays in an ellipsoid. The size of this allowed
    ellipsoid scales for the position x_{rg}, epsx/sqrt(n_c W(r)_g), and for the precision W(r)_g as epsW W(r)_g/n_c.

    This function returns the upper bound *per epsx and epsW* so that these can be picked later. So, it returns
    ddLogL/epsx, ddLogL/epsW
    and the upper bound is then determined by
    dLogL + ddLogL/epsx * epsX + ddLogL/epsW * epsW
    """
    xr_g, Wr_g, nChild = rootInfo['pos'], rootInfo['prec'], rootInfo['nChild']
    if ltqs_gi is None:
        tOld1, wbar1_g, ltqs1, ltqsVars1, nodeInd1 = child1.getInfo()
        tOld2, wbar2_g, ltqs2, ltqsVars2, nodeInd2 = child2.getInfo()

        rootMinusFirst_W_g = Wr_g - wbar1_g
        rootMinusFirst_ltqs = xr_g * Wr_g - wbar1_g * ltqs1
        WR_g = rootMinusFirst_W_g - wbar2_g
        ltqsR = (rootMinusFirst_ltqs - wbar2_g * ltqs2) / WR_g
        ltqsVarsR = 1 / WR_g
        ltqs_gi = np.column_stack((ltqs1, ltqs2, ltqsR))
        ltqsVars_gi = np.column_stack((ltqsVars1, ltqsVars2, ltqsVarsR))
        wir_gi = np.column_stack((wbar1_g, wbar2_g, 1 / ltqsVarsR))
        if sequential:
            newLogLik, optTimes, t12Opt, converged = optimiseT3LeafStarSequential(ltqs_gi, ltqsVars_gi, tOpt,
                                                                                  verbose=False)

            if converged:
                optTimes = [optTimes[0], t12Opt - optTimes[0], optTimes[1]]
            else:
                sequential = False
                tOpt = calcTInit(tOld1, tOld2, sequential)
        if not sequential:
            newLogLik, tOpt, converged = optimiseT3LeafStar(ltqs_gi, ltqsVars_gi, tOpt, verbose=False)
        if not converged:
            # Now exiting. If this happens often, I can also just say that these pairs get very low likelihood.
            mp_print("Times cannot be optimized for this pair, probably too many noisy genes are taken into account. "
                     "Try running Bonsai with higher zscore-cutoff.", WARNING=True)
            newLogLik = -np.inf

        wbar_gi = np.column_stack((wbar1_g, wbar2_g, 1 / ltqsVarsR))
        loglik = getLoglikAndGradStarTree(ltqs_gi=ltqs_gi, xr_g=xr_g, wbar_gi=wbar_gi, W_g=Wr_g,
                                          returnGrad=False)
        dLogL = newLogLik - loglik
    else:
        dLogL = None
    # Get derivative w.r.t. W(r)
    wia_gi = 1 / (ltqsVars_gi + tOpt)
    Wa_g = np.sum(wia_gi, axis=1)
    wiaOverWa_gi = np.divide(wia_gi, Wa_g[:, None])

    w1a_g, w2a_g, wRa_g = wia_gi.T
    w1r_g, w2r_g, wRr_g = wir_gi.T
    ltqs1, ltqs2, ltqsR = ltqs_gi.T

    diff12_g = ltqs1 - ltqs2
    diff1R_g = ltqs1 - ltqsR
    diff2R_g = ltqs2 - ltqsR
    diffrR_g = xr_g - ltqsR
    v12_g = diff12_g ** 2
    v1R_g = diff1R_g ** 2
    v2R_g = diff2R_g ** 2

    w1r_g = w1r_g
    w2r_g = w2r_g

    da_g = w1a_g * w2a_g * v12_g + w1a_g * wRa_g * v1R_g + w2a_g * wRa_g * v2R_g
    dr_g = w1r_g * w2r_g * v12_g + w1r_g * wRr_g * v1R_g + w2r_g * wRr_g * v2R_g

    wRaOverW_g = wiaOverWa_gi[:, 2]
    wRrOverW_g = wRr_g / Wr_g

    # Real expression:
    dLdWr = (wRa_g * (1 - wRaOverW_g) - wRr_g * (1 - wRrOverW_g) +
             wRaOverW_g * (w1a_g * w2a_g * v12_g - (1 - wRaOverW_g) * da_g) -
             wRrOverW_g * (w1r_g * w2r_g * v12_g - (1 - wRrOverW_g) * dr_g)) / (wRr_g ** 2) + \
            (2 * diffrR_g / wRr_g) * (
                    wRaOverW_g * (w1a_g * diff1R_g + w2a_g * diff2R_g) - wRrOverW_g * (
                    w1r_g * diff1R_g + w2r_g * diff2R_g))

    # Given this derivative we calculate the maximal increase in dLogL by setting dLUB = |dLdWr * Wr| * epsW / n_c
    ddLogLPerEpsW = np.linalg.norm(dLdWr * Wr_g, 2)

    # WrUB_g = np.maximum(Wr_g + ((dLdWr * Wr_g ** 2) / np.linalg.norm(dLdWr * Wr_g)) * epsW / nChildren,
    #                     1.2 * (wbar1_g + wbar2_g))

    # Get most improving xr
    # The real expression
    dLdxr = 2 * (Wr_g / wRr_g) * (diff1R_g * (w1a_g * wRaOverW_g - w1r_g * wRrOverW_g)
                                  + diff2R_g * (w2a_g * wRaOverW_g - w2r_g * wRrOverW_g))

    # Given this derivative we calculate the maximal increase in dLogL by setting dLUB = |dLdxr/sqrt(Wr)| epsx / n_c
    ddLogLPerEpsx = np.linalg.norm(dLdxr / np.sqrt(Wr_g), 2)
    # Option 2: we calculate the maximal increase in dLogL by moving the W(r) in the best direction possible:
    # xrUB_g = xr_g + (
    #         (dLdxr / Wr_g) / np.linalg.norm(dLdxr / np.sqrt(Wr_g), 2)) * epsx / np.sqrt(nChildren)
    return dLogL, ddLogLPerEpsx, ddLogLPerEpsW


# Used
def getLogLikAndGradStarTreeSequentialWrapper(t_i, t12, ltqsVars_gi, ltqs_gi):
    # This wrapper allows t_i to be the first argument, where t_i has only 2 entries, t_{1a} and t_{ar}. Together with
    # t_{12} = t_{1a} + t_{2a} this gives all times. The wrapper returns the negative of the loglik and grad.
    # It is used in optimising the diffusion times for a merge when args.sequential = True
    t_i = np.array([t_i[0], t12 - t_i[0], np.exp(t_i[1])])
    loglik, grad = getLoglikAndGradStarTree(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i, returnGrad=True)
    grad = np.array([grad[0] - grad[1], t_i[2] * grad[2]])
    return -loglik, -grad


# Used
def getLogLikAndGradStarTreeSequentialWrapperOld(t_i, t12, ltqsVars_gi, ltqs_gi):
    # This wrapper allows t_i to be the first argument, where t_i has only 2 entries, t_{1a} and t_{ar}. Together with
    # t_{12} = t_{1a} + t_{2a} this gives all times. The wrapper returns the negative of the loglik and grad.
    # It is used in optimising the diffusion times for a merge when args.sequential = True
    t_i = np.array([t_i[0], t12 - t_i[0], t_i[1]])
    loglik, grad = getLoglikAndGradStarTree(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i, returnGrad=True)
    grad = np.array([grad[0] - grad[1], grad[2]])
    return -loglik, -grad


# Used
def getLogLikAndGradStarTreeWrapper(t_i, ltqsVars_gi, ltqs_gi):
    # This wrapper allows t_i to be the first argument, and returns the negative of the loglik and grad. It is used in
    # optimising the diffusion times for a merge
    t_i = np.exp(t_i)
    loglik, grad = getLoglikAndGradStarTree(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i, returnGrad=True)
    return -loglik, - t_i * grad


def getLogLikAndGradStarTreeWrapperOld(t_i, ltqsVars_gi, ltqs_gi):
    # This wrapper allows t_i to be the first argument, and returns the negative of the loglik and grad. It is used in
    # optimising the diffusion times for a merge
    loglik, grad = getLoglikAndGradStarTree(ltqs_gi=ltqs_gi, ltqsVars_gi=ltqsVars_gi, t_i=t_i, returnGrad=True)
    return -loglik, -grad


# Used
def findNodeLtqsGivenLeafs(childNodes=None, ltqs_gi=None, ltqsVars_gi=None, t_i=None, return_wbar_gi=False,
                           return_wOverW_gi=False):
    if childNodes is not None:
        # In this case, no coordinates (ltqs_gi etc) are needed, because are taken from children-objects
        W_g = np.zeros(bs_glob.nGenes)
        xr_g = np.zeros(bs_glob.nGenes)
        if return_wbar_gi:
            wbar_gi = np.zeros(bs_glob.nGenes, len(childNodes))
        for cInd, child in enumerate(childNodes):
            wbar_g = 1 / (child.getLtqsVars(mem_friendly=True) + child.tParent)
            if return_wbar_gi:
                wbar_gi[:, cInd] = wbar_g
            W_g += wbar_g
            xr_g += child.ltqs * wbar_g
        xr_g /= W_g
        if return_wOverW_gi:
            wOverW_gi = np.divide(wbar_gi, W_g[:, None])
    else:
        wbar_gi = 1 / (ltqsVars_gi + t_i)
        W_g = np.sum(wbar_gi, axis=1)
        wOverW_gi = np.divide(wbar_gi, W_g[:, None])
        xr_g = np.sum(np.multiply(wOverW_gi, ltqs_gi), axis=1)
    if not return_wbar_gi:
        return xr_g, W_g
    else:
        if not return_wOverW_gi:
            return xr_g, W_g, wbar_gi
        else:
            return xr_g, W_g, wbar_gi, wOverW_gi


# Used
def findNodeLtqsGivenLeafsSingleGene(ltqs_i, ltqsVars_i, t_i, return_wbar_i=False, return_wOverW_i=False):
    wbar_i = 1 / (ltqsVars_i + t_i)
    W = np.sum(wbar_i)
    wOverW_i = np.divide(wbar_i, W)
    xr = np.dot(wOverW_i, ltqs_i)
    if not return_wbar_i:
        return xr, W
    else:
        if not return_wOverW_i:
            return xr, W, wbar_i
        else:
            return xr, W, wbar_i, wOverW_i


# Used
def getLtqsAsIfRoot(nodeLtqs_g, nodeW_g, tConn, rootLtqs_g, rootW_g):
    # To get the ltqs and W of a node where all other node-positions have been integrated out, we can view it as
    # the root. We can shift the root-position along an edge by first subtracting from the root's position the
    # contribution of the node itself, then adding the remaining root-contribution to the position of the node
    # TODO: Check if this can be done more efficiently
    wbarNode_g = 1 / (tConn + 1 / nodeW_g)
    rootMinusNodeW_g = rootW_g - wbarNode_g
    rootMinusNodeLtqs_g = (rootLtqs_g * rootW_g - wbarNode_g * nodeLtqs_g) / rootMinusNodeW_g

    wbarRoot_g = 1 / (tConn + 1 / rootMinusNodeW_g)
    nodePlusRootW_g = nodeW_g + wbarRoot_g
    nodePlusRootLtqs_g = (nodeLtqs_g * nodeW_g + wbarRoot_g * rootMinusNodeLtqs_g) / nodePlusRootW_g
    return nodePlusRootLtqs_g, nodePlusRootW_g


# Used
def getLtqsAfterChildUpdate(nodeLtqs_g, nodeW_g, tConn, oldChLtqs_g, oldChLtqsVars_g, newChLtqs_g, newChLtqsVars_g):
    wbarOldCh_g = 1 / (tConn + oldChLtqsVars_g)
    wbarNewCh_g = 1 / (tConn + newChLtqsVars_g)
    newNodeW_g = nodeW_g - wbarOldCh_g + wbarNewCh_g
    newNodeLtqs_g = (nodeLtqs_g * nodeW_g - wbarOldCh_g * oldChLtqs_g + wbarNewCh_g * newChLtqs_g) / newNodeW_g
    return newNodeLtqs_g, newNodeW_g


# Used
def calcSingleDLogL(xrAsIfRoot_g, WAsIfRoot_g, ltqs1, ltqsVars1, wbar1_g, tOld1, rootMinusFirst_W_g,
                    rootMinusFirst_ltqs, ltqs2, ltqsVars2, wbar2_g, tOld2, sequential=True, onlyTimes=False,
                    returnAll=False, tol=None):
    # t0_i = tInit if tInit is not None else calcTInit(tOld1, tOld2, sequential)
    t0_i = calcTInit(tOld1, tOld2, sequential)

    # Optimise three-leaf star-tree with leafs (ltqs1, topt1 + ltqsVars1) (ltqs2, topt2 + ltqsVars2)
    # (ltqsR, toptR + 1/self.rootW_g)
    # where ltqsR, 1/self.rootW_g are the mean and variance of the root-position when all other leafs
    # (the leafs not in the current pair) are taken into account
    WR_g = rootMinusFirst_W_g - wbar2_g
    ltqsR = (rootMinusFirst_ltqs - wbar2_g * ltqs2) / WR_g
    ltqsVarsR = 1 / WR_g
    ltqs_gi = np.column_stack((ltqs1, ltqs2, ltqsR))
    ltqsVars_gi = np.column_stack((ltqsVars1, ltqsVars2, ltqsVarsR))
    # TODO: Change requested tolerance in t_i
    if sequential:
        newLogLik, optTimes, t12Opt, converged = optimiseT3LeafStarSequential(ltqs_gi, ltqsVars_gi, t0_i, verbose=False,
                                                                              tol=tol)

        # TODO: REMOVE THIS!
        # newLogLik_Old, optTimes_Old, t12Opt_Old, converged_Old = optimiseT3LeafStarSequentialOld(ltqs_gi, ltqsVars_gi,
        #                                                                                          t0_i,
        #                                                                                          verbose=False, tol=tol)
        # if np.max(np.abs(optTimes - optTimes_Old)) > 1e-6:
        #     mp_print("New optTimes: {}, differ from old optTimes {}.".format(optTimes, optTimes_Old), DEBUG=True)
        #     mp_print("New loglik: {}, differ from old loglik {}.".format(newLogLik, newLogLik_Old), DEBUG=True)

        if converged:
            optTimes = [optTimes[0], t12Opt - optTimes[0], optTimes[1]]
        else:
            sequential = False
            t0_i = calcTInit(tOld1, tOld2, sequential)
    if not sequential:
        newLogLik, optTimes, converged = optimiseT3LeafStar(ltqs_gi, ltqsVars_gi, t0_i, verbose=False)

    if not converged:
        # Now exiting. If this happens often, I can also just say that these pairs get very low likelihood.
        mp_print("Times cannot be optimized for this pair, probably too many noisy genes are taken into account. "
                 "Try running Bonsai with higher zscore-cutoff.", WARNING=True)
        newLogLik = -np.inf
        optTimes = None
    if onlyTimes:
        return optTimes
    # Then calculate three-leaf star-tree likelihood with leafs (ltqs1, t1 + ltqsVars1) (ltqs2, t2 + ltqsVars2)
    # (ltqsR, 1/self.rootW_g)
    # wbar_gi = np.column_stack((wbar1_g, wbar2_g, 1 / ltqsVarsR))
    wbar_gi = np.column_stack((wbar1_g, wbar2_g, WR_g))
    loglik = getLoglikAndGradStarTree(ltqs_gi=ltqs_gi, xr_g=xrAsIfRoot_g, wbar_gi=wbar_gi, W_g=WAsIfRoot_g,
                                      returnGrad=False)
    dLogL = newLogLik - loglik
    # Real dLogL is (0.5 * dLogL)
    if returnAll:
        return dLogL, optTimes, ltqs_gi, ltqsVars_gi, wbar_gi
    else:
        return dLogL, optTimes


# Used
def getOptTimesSingleDLogLWrapper(xrAsIfRoot_g, WAsIfRoot_g, optChild1, optChild2, sequential=True, verbose=False,
                                  tol=1e-6):
    # This wrapper is used when computation of optimal times is only done for one pair, so when there is no
    # performance gain by pre-calculating some information on the first child in the pair. As such, this wrapper
    # allows for a one-line call.
    tOld1, wbar1_g, ltqs1, ltqsVars1, optNodeInd1 = optChild1.getInfo()
    tOld2, wbar2_g, ltqs2, ltqsVars2, optNodeInd2 = optChild2.getInfo()

    # We check whether for this pair of nodes we already have some reasonable diff. time guess
    # sortedNodeInds = tuple(sorted([optNodeInd1, optNodeInd2]))
    # tInit = getNodePairInfo(optTDict['all'], sortedNodeInds)
    # In case we do sequential optimisation, we also check if t_12 = t_1a+t_2a is known already
    # t12Opt = getNodePairInfo(optTDict['t12'], sortedNodeInds) if sequential else None

    rootMinusFirst_W_g = WAsIfRoot_g - wbar1_g
    rootMinusFirst_ltqs = xrAsIfRoot_g * WAsIfRoot_g - wbar1_g * ltqs1

    optTimes = calcSingleDLogL(xrAsIfRoot_g, WAsIfRoot_g, ltqs1, ltqsVars1, wbar1_g, tOld1, rootMinusFirst_W_g,
                               rootMinusFirst_ltqs, ltqs2, ltqsVars2, wbar2_g, tOld2, sequential=sequential,
                               onlyTimes=True, tol=tol)
    return optTimes


# def getDistOnTree(node1, node2):
#     """
#     This function calculates the number of edges between two nodes. This assumes getDSInfo() was already called
#     :param node1:
#     :param node2:
#     :return:
#     """
#     if node1.nodeInd in node2.dsInfo['dsNodeInds']:
#         dist = node1.dsInfo['level'] - node2.dsInfo['level']
#     elif node2.nodeInd in node1.dsInfo['dsNodeInds']:
#         dist = node2.dsInfo['level'] - node1.dsInfo['level']
#     elif node1.nodeInd == node2.nodeInd:
#         dist = 0
#     else:
#         dist = node2.dsInfo['level'] + node1.dsInfo['level']
#     return dist


def updateNNInfo(runConfigs, NNInfo, del_node_inds, newAnc):
    if runConfigs['useNN'] and (not runConfigs['getNewNN']) and (newAnc is not None):
        NNInfo['NNcounter'] += 1
        # for deletedInd in del_node_inds:  # Map all nb-connections of merged leafs to their ancestor
        #     if deletedInd in NNInfo['leafToChild']:
        #         NNInfo['leafToChild'][deletedInd] = newAnc.nodeInd
        #     else:
        #         downstreamLeafs = [leaf for leaf, child in NNInfo['leafToChild'].items() if
        #                            child == deletedInd]
        #         for leaf in downstreamLeafs:
        #             NNInfo['leafToChild'][leaf] = newAnc.nodeInd
        for leaf, child in NNInfo['leafToChild'].items():
            if (leaf in del_node_inds) or (child in del_node_inds):
                NNInfo['leafToChild'][leaf] = newAnc.nodeInd
    return NNInfo


def getEdgeDistVertNamesFromNode(node, edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode):
    nodeIndToNode[node.nodeInd] = node
    for child in node.childNodes:
        edge_list.append([node.nodeInd, child.nodeInd])
        dist_list.append(child.tParent)
        if (child.nodeId is None) or child.nodeId[:9] == 'internal_':
            orig_vert_names[child.nodeInd] = "internal_%d" % intCounter
            child.nodeId = "internal_%d" % intCounter
            intCounter += 1
        else:
            orig_vert_names[child.nodeInd] = child.nodeId
        edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode = getEdgeDistVertNamesFromNode(child,
                                                                                                        edge_list,
                                                                                                        dist_list,
                                                                                                        orig_vert_names,
                                                                                                        intCounter,
                                                                                                        nodeIndToNode)
    return edge_list, dist_list, orig_vert_names, intCounter, nodeIndToNode
