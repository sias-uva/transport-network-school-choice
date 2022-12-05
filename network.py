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
                    print(f'Failed for {o["node_id"]} - {d["node_id"]}: No path found between them')
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

