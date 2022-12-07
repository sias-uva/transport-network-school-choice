import numpy as np
from network import Network

def create_random_edge(network: Network, edge_weight=1):
    """Creates a random edge between two nodes that are not connected.

    Args:
        network (Network): the network.
        edge_weight (int, optional): weight of the edge to add. Defaults to 1.

    Returns:
        tuple: (x, y, edge_weight) where x and y are the indices of the nodes to connect and edge_weight is the weight of the edge.
    """
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
    return x, y, edge_weight

def maximize_closeness_centrality(network: Network, node_id: int, edge_weight=1):
    """Returns the edge that maximizes the closeness centrality of the given node.

    Args:
        network (Network): the network.
        node_id (int): the node we want to maximize the closeness centrality of.
        edge_weight (int, optional): weight of the edge to add. Defaults to 1.

    Returns:
        tuple: (x, y, edge_weight) where x and y are the indices of the nodes to connect and edge_weight is the weight of the edge that maximizes the closeness centrality of the given node.
    """
    # Get the node object from the node ID
    node = network.network.vs.find(node_id)

    # Get the current adjacency matrix of the network.
    adj_mx = network.get_adj_matrix()
    # Get the indices of the non-connected nodes in the adjacecy matrix.
    candidate_nodes = np.where(adj_mx[node_id] == 0)

    candidate_edges = [(node_id, cn) for cn in candidate_nodes[0] if cn != node_id]

    if len(candidate_edges) == 0:
        print('Cannot add more edges, all nodes are connected to the given node.')
        return None, None, None

    # assert len(candidate_edges) > 0, 'Cannot add more edges, all nodes are connected to the given node.'

    # Initialize a variable to store the maximum closeness centrality
    max_centrality = 0

    # Initialize a variable to store the edge with the maximum closeness centrality
    max_edge = None

    # Loop through all the possible edges
    for edge in candidate_edges:
        # Add the edge to the graph
        network.network.add_edge(edge[0], edge[1], weight=edge_weight)

        # Calculate the closeness centrality of the input node
        centrality = network.network.closeness(node)

        # If the centrality is greater than the current maximum,
        # update the maximum and the edge with the maximum centrality
        if centrality > max_centrality:
            max_centrality = centrality
            max_edge = edge

        # Remove the edge from the graph
        network.network.delete_edges(edge)

    # Return the edge with the maximum closeness centrality
    return max_edge[0], max_edge[1], edge_weight
