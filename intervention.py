import numpy as np
from network import Network

def create_random_edge(network: Network, edge_weight=1):
    # Get the current adjacency matrix of the network.
    adj_mx = network.get_adj_matrix()
    # Get the indices of the non-connected nodes in the adjacecy matrix.
    candidate_idx = np.where(adj_mx == 0)
    assert len(candidate_idx[0]) > 0, 'Cannot add more edges, all nodes are connected.'
    # Get a random index from the non-connected nodes.
    rand_idx = np.random.choice(candidate_idx[0].shape[0])
    x = candidate_idx[0][rand_idx]
    y = candidate_idx[1][rand_idx]

    assert adj_mx[x][y] == 0, 'Selected nodes are already connected via an edge.'

    network.add_edge(x, y)

    print(f'Added a new edge between nodes {x} and {y}')
