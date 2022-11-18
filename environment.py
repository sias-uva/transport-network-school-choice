import igraph as ig
import pandas as pd

class Environment(object):
    def __init__(self, path, network_file, node_attributes_file, facilities_file):
        super(Environment, self).__init__()
        
        assert path, "env path cannot be None"
        assert network_file, "network_file cannot be None"
        assert node_attributes_file, "node_attributes_file cannot be None"
        assert facilities_file, "facilities_file cannot be None"

        assert network_file.split('.')[1] == "gml", "only .gml network files are accepted"

        # Load the network depending on the file format: .txt for adjacency list, .gml for gml
        self.network = ig.Graph.Read(path / network_file)
        self.node_attributes = pd.read_csv(path / node_attributes_file, index_col=0)
        self.facilities = pd.read_csv(path / facilities_file, index_col=0)


        print('delete this print statement')
