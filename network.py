import igraph as ig
from matplotlib import pyplot as plt
import numpy as np
import math
import os

class Network(object):
    DEFAULT_EDGE_COLOR = 'black'
    def __init__(self, network_path, calc_tt_mx=False):
        """Holds the transport network.

        Args:
            network_path (str): the full path to the network file - used to load the transport network.
            calc_tt_mx (boolean): if set to true, it will pre-calculate and store the travel times from all nodes to all other nodes.
        """
        _, ext = os.path.splitext(network_path)
        assert ext == ".gml", "only .gml network files are accepted (currently)"

        # Load the network
        self.network = ig.Graph.Read(network_path)
        # Network layout for plotting (to keep node positions consistent between plots)
        self.network_layout = self.network.layout('kk')
        # Becomes relevant when we want to add edges to the network and plot the new edges.
        self.network.es['color'] = self.DEFAULT_EDGE_COLOR
        self.network.es['label'] = ''
        # If I just do 'id' it returns it as float, hence the weird list comprehension.
        self.network.vs['label'] = [self.network.vs[i].index for i in range(len(self.network.vs))]
        # Keep a list of all the added edges (interventions) to the network.
        self.added_edges = []
        # Calculate the travel time matrix from all nodes to all nodes, store it so we don't have to re-calculate it every time.
        # TODO: think of memory constraints? 
        if calc_tt_mx:
            self.tt_mx = self.shortest_paths(self.network.vs, self.network.vs)

    def shortest_paths(self, orig: list, dest: list):
        """Calculates shortest paths from a list of origins to a list of destinations. Both orig and dest should be lists of node ids.

        Args:
            orig (list): list of origin nodes (ids).
            dest (list): list of destination nodes (ids)

        Returns:
            np.array: 2d matrix where element (i,j) is the shortest path (weighted) from i to j and vice versa.
        """
        # Travel time from all orig nodes to all dest ndoes
        tt_mx = np.ndarray((len(orig), len(dest)))
        for i, o in enumerate(orig):
            for j, d in enumerate(dest):
                sp = self.network.distances(o, d)[0][0]
                if sp == math.inf:
                    print(f'Failed for {o["id"]} - {d["id"]}: No path found between them')
                    continue
                tt_mx[i, j] = sp

        return tt_mx

    def add_edge(self, from_v, to_v, weight):
        """Adds a new edge to the transport network. Note: currently does not support weighted edges.

        Args:
            from_v (int): id of the origin vertex of the new edge.
            to_v (int): id of the destination vertex of the new edge.
        Returns: 
            igraph.Edge: the newly added edge as an Edge object

        """
        edge = self.network.add_edge(from_v, to_v, weight=weight)
        self.added_edges.append(edge)
        # Update the travel time matrix.
        # Reconsider this if we have a big network.
        self.tt_mx = self.shortest_paths(self.network.vs, self.network.vs)
        return edge

    def get_adj_matrix(self):
        """Returns the current node adjacency matrix of the network.

        Returns:
            np.array: thje adjacency matrix.
        """
        return np.array(self.network.get_adjacency().data)

    def weighted_closeness(self, nodes=None, weights=None, normalized=False):
        """Calculates the weighted closeness centrality of a given node and given node weights.

        Args:
            nodes (int or list): node/s to calculate the weighted closeness centrality for. If none, it will calculate the centrality for all nodes.
            weights (list): weights of each node. If none, it will calculate the unweighted centrality of the nodes.
            normalized (bool, optional): if true, values will be normalized by multiplying them with nr of nodes - 1 . Defaults to True.

        Returns:
            np.float64: the calculated weighted closeness centrality.
        """
        if nodes is None:
            nodes = self.network.vs.indices

        if type(nodes) is not list and type(nodes) is not np.ndarray: nodes = [ nodes ]

        if weights is None:
            weights = np.array([1] * len(nodes))

        # For now I want to recalculate the shortest paths every time in this step, otherwise it's dangerous because we don't always update the self.tt_mx.
        # if self.tt_mx is None:
        shortest_paths = np.array(self.network.shortest_paths(target=nodes))
        # else: 
            # shortest_paths = self.tt_mx[:, nodes]
        
        if len(nodes) == 1:
            shortest_paths = shortest_paths.flatten()
            weighted_shortest_paths = shortest_paths * weights
            weighted_closeness = np.divide(1, weighted_shortest_paths.sum())
            # If weight is 0, the weighted_closeness will be inf. Set it to 0.
            if weighted_closeness == np.inf: weighted_closeness = 0
        else:
            weighted_shortest_paths = shortest_paths * np.array(weights)[np.newaxis].T
            weighted_closeness = np.divide(1, weighted_shortest_paths.sum(axis=0))
            # If weight is 0, the weighted_closeness will be inf. Set it to 0.
            weighted_closeness[weighted_closeness == np.inf] = 0
        
        
        # Normalize the way igraph closeness does it.
        if normalized:
            return weighted_closeness * (shortest_paths.shape[0] - 1)
            
        return weighted_closeness
