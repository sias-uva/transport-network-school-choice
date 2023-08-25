import numpy as np
from network import Network

def get_candidate_edges(network: Network, node_id: int):
    """Returns a list of candidate edges to add to the network, that are not already conncted to the given node.

    Args:
        network (Network): the network.
        node_id (int): the node to connect to.

    Returns:
        list: list of candidate edges to add to the network, that are not already conncted to the given node.
    """    
    # Get the indices of the non-connected nodes in the adjacecy matrix.
    candidate_nodes = np.where(network.get_adj_matrix()[node_id] == 0)

    candidate_edges = [(node_id, cn) for cn in candidate_nodes[0] if cn != node_id]

    if len(candidate_edges) == 0:
        print('Cannot add more edges, all nodes are connected to the given node.')
        return None
    
    return candidate_edges
    

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
    if len(candidate_idx[0]) <= 0:
        return None, None, None
    # assert len(candidate_idx[0]) > 0, 'Cannot add more edges, all nodes are connected.'
    # Get a random index from the non-connected nodes.
    rand_idx = np.random.choice(candidate_idx[0].shape[0])
    x = candidate_idx[0][rand_idx]
    y = candidate_idx[1][rand_idx]

    assert adj_mx[x][y] == 0, 'Selected nodes are already connected via an edge.'
    return x, y, edge_weight

def maximize_node_centrality(network: Network, node_id: int, centrality_measure: str, group_weights=None, edge_weight=1):
    """Returns the edge that maximizes the given centrality measure of the given node.

    Args:
        network (Network): the network.
        node_id (int): the node we want to maximize the centrality of.
        centrality_measure (str): the centrality measure to maximize for. Accepted values: ['closeness', 'betweenness', 'degree', 'group_closeness', 'group_betweenness', 'group_degree'].
        edge_weight (int, optional): weight of the edge to add. Defaults to 1.

    Returns:
        tuple: (x, y, edge_weight) where x and y are the indices of the nodes to connect and edge_weight is the weight of the edge that maximizes the centrality of the given node.
    """
    assert centrality_measure in ['closeness', 'betweenness', 'degree', 'group_closeness', 'group_betweenness', 'group_degree'], 'Invalid centrality measure.'

    # Get the node object from the node ID
    node = network.network.vs.find(node_id)

    candidate_edges = get_candidate_edges(network, node_id)

    if candidate_edges is None:
        return None, None, None

    # Initialize a variable to store the maximum centrality
    max_centrality = 0
    # Initialize a variable to store the edge with the maximum centrality
    max_edge = None

    # Loop through all the possible edges
    for edge in candidate_edges:
        # Add the edge to the graph
        network.network.add_edge(edge[0], edge[1], weight=edge_weight)

        # Calculate the centrality of the input node
        if centrality_measure == 'closeness':
            centrality = network.network.closeness(node)
        elif centrality_measure == 'betweenness':
            centrality = network.network.betweenness(node)
        elif centrality_measure == 'degree':
            centrality = network.network.degree(node)
        elif centrality_measure == 'group_closeness':
            if group_weights is None:
                raise ValueError('Group weights must be provided to calculate group closeness.')

            centrality = network.weighted_closeness(node_id, weights=group_weights)
        elif centrality_measure == 'group_betweenness':
            if group_weights is None:
                raise ValueError('Group weights must be provided to calculate group betweenness.')

            centrality = network.weighted_betweeness(node_id, weights=group_weights)
        elif centrality_measure == 'group_degree':
            if group_weights is None:
                raise ValueError('Group weights must be provided to calculate group degree.')

            centrality = network.weighted_degree(node_id, weights=group_weights)

        # If the centrality is greater than the current maximum,
        # update the maximum and the edge with the maximum centrality
        if centrality > max_centrality:
            max_centrality = centrality
            max_edge = edge

        # Remove the edge from the graph
        network.network.delete_edges(edge)

    if max_edge is None:
        print(f'Warning - No edge could be added for node {node_id} and centrality measure {centrality_measure}. Probably ll edges lead to 0 centrality.')
        return None, None, None
    # Return the edge with the maximum centrality
    return max_edge[0], max_edge[1], edge_weight

# def maximize_group_node_centrality(network: Network, node_id: int, group_id: int, centrality_measure: str, edge_weight=1):
#     assert centrality_measure in ['group_closeness'], 'Invalid centrality measure.'

#     # Get the node object from the node ID
#     node = network.network.vs.find(node_id)

#     candidate_edges = get_candidate_edges(network, node_id)

#     if candidate_edges is None:
#         return None, None, None

#     # Initialize a variable to store the maximum centrality
#     max_centrality = 0
#     # Initialize a variable to store the edge with the maximum centrality
#     max_edge = None