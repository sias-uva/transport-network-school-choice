import igraph as ig
import numpy as np
import math
import os

class Network(object):

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

    def add_edge(self, from_v, to_v):
        """Adds a new edge to the transport network. Note: currently does not support weighted edges.

        Args:
            from_v (int): id of the origin vertex of the new edge.
            to_v (int): id of the destination vertex of the new edge.
        """
        self.network.add_edge(from_v, to_v)

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
        # Calculate the travel time matrix from all 
        if calc_tt_mx:
            self.tt_mx = self.shortest_paths(self.network.vs, self.network.vs)