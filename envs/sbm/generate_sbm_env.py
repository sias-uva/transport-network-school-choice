#%%
import igraph as ig
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
seed = 42

#%% Generate Graph
random.seed(seed)
np.random.seed(seed)
# Number of nodes in each group.
n_communities = 2
nodes_in_community = 6
n_nodes = n_communities * nodes_in_community

community_sizes = [nodes_in_community] * n_communities

total_pop = 500
# Percent of the majority group overall in the population.
maj_pop_pct = 0.5
# Percent of the majority group in the group-dominant nodes.
maj_pop_pct_in_nodes = [0.8]
# probability of in-community and out-community edges
p_in = 0.7
p_out = 0.01
# Capacity of facilities - constant for each facility.
facility_cap = int(total_pop)/2

network_name = f"SBM_{n_communities}_{nodes_in_community}_{p_in}_{p_out}_pop_{total_pop}_{maj_pop_pct}_{maj_pop_pct_in_nodes}"

if not os.path.exists(network_name):
        os.makedirs(network_name)

# Create NxN matrix of probabilities for in edges and out edges for each community
in_out_prefs = np.full((n_communities, n_communities), p_out)
np.fill_diagonal(in_out_prefs, p_in)

for i in range(n_communities):
    n1 = (i + 1) if (i + 1) < n_communities else 0
    n2 = (i - 1) if (i - 1) >= 0 else (n_communities - 1)
    in_out_prefs[i][n1] = p_out
    in_out_prefs[i][n2] = p_out

graph = ig.Graph.SBM(n_nodes, in_out_prefs.tolist(), community_sizes)
# Raw graph, without any node attributes.
graph_raw = copy.deepcopy(graph)

graph.vs['label'] = graph.vs.indices
graph.vs['label_size'] = 10

g0 = []
g1 = []

for i in graph.vs.indices:
    if graph.vs[i].index < community_sizes[0]:
        graph.vs[i]['color'] = 'blue'
        g0.append(i)
    else:
        graph.vs[i]['color'] = 'red'
        g1.append(i)

# ig.plot(graph, vertex_size=20)
ig.plot(graph, vertex_size=20, target=f'./{network_name}/network.pdf')
ig.write(graph, f'./{network_name}/network.gml')
ig.write(graph_raw, f'./{network_name}/network_raw.gml')

# %% Generate Population of agents for each node in the network.
np.random.seed(seed)

graph = ig.Graph.Read(f'./{network_name}/network.gml')

nodes = pd.DataFrame({attr: graph.vs[attr] for attr in graph.vertex_attributes()})
nodes['id'] = nodes['id'].astype(int)
nodes.loc[:, 'closeness'] = graph.closeness()

for comm in ['blue', 'red']:
    comm_subgraph = graph.induced_subgraph(nodes[nodes['color'] == comm].id.to_list())
    nodes.loc[nodes['color'] == comm, 'community_closeness'] = comm_subgraph.closeness()


# Generate 3 populations with 3 different seed values.
for pop_seed in [42, 9845, 9328]:
    np.random.seed(pop_seed)
    nodes = nodes.drop(columns=['g0_pct', 'g1_pct', 'g0_in_node', 'g1_in_node'], errors='ignore')
    nodes.loc[nodes['id'].isin(g0), 'g0_pct'] = np.random.choice(maj_pop_pct_in_nodes, len(g0))
    nodes.loc[nodes['id'].isin(g0), 'g1_pct'] = 1 - nodes.loc[nodes['id'].isin(g0) ,'g0_pct']

    nodes.loc[nodes['g0_pct'].isna(), 'g1_pct'] = np.random.choice(maj_pop_pct_in_nodes, len(g1))
    nodes.loc[nodes['g0_pct'].isna(), 'g0_pct'] = 1 - nodes.loc[nodes['g0_pct'].isna(), 'g1_pct']

    # Group distribution on nodes. The percentage of each group in each node.
    nodes.loc[:, 'g0_in_node'] = nodes['g0_pct'] / nodes['g0_pct'].sum()
    nodes.loc[:, 'g1_in_node'] = nodes['g1_pct'] / nodes['g1_pct'].sum()

    agents = []
    g0_nr = int(total_pop * maj_pop_pct)
    g1_nr = total_pop - g0_nr

    # id = 0
    for i in range(total_pop):
        group = 'g0' if i < g0_nr else 'g1'
        node = np.random.choice(nodes['id'], p=nodes[f'{group}_in_node'])
        # Set tolerance similar to that of the majority population of the node.
        agents.append({'id': i, 'node': node, 'group': group, 
                        'tolerance': nodes[nodes['id'] == node][['g0_pct', 'g1_pct']].iloc[0].max()})

    agents = pd.DataFrame(agents)
    agents.to_csv(f'./{network_name}/population_{pop_seed}.csv', index=False)

#%% Generate facilities
fac_nodes = []

# Add two facilities on each community, one with the highest closeness and one with the lowest closeness.
fac_nodes.append(nodes.loc[nodes['id'] < community_sizes[0]].sort_values('community_closeness').iloc[-1]['id'])
fac_nodes.append(nodes.loc[np.isin(nodes['id'], range(community_sizes[1], community_sizes[0] + community_sizes[1]))].sort_values('community_closeness').iloc[-1]['id'])

# fac_nodes = [6, 13]

facilities = []
for i, f in enumerate(fac_nodes):
    facilities.append({'id': i, 'node': f, 'facility': f'school_{i}', 'capacity': facility_cap, 'quality': 0.5, 'popularity': 0.5})

facilities = pd.DataFrame(facilities)

facilities.to_csv(f'./{network_name}/facilities.csv', index=False)

#%%
A = nodes['g0_in_node'].sum()
B = nodes['g1_in_node'].sum()
DI = 0
for _, nid in nodes.iterrows():
    a = nid['g0_in_node']
    b = nid['g1_in_node']

    DI += np.abs(a / A - b / B)

print(f'Dissimilarity Index: {1 / 2 * DI}')
# %% Plot a raw graph without any decoration, only the nodes where facilities are placed.
for v in graph_raw.vs:
    if np.isin(v.index, facilities['node'].values):
        v['color'] = 'yellow'
        v['label'] = '*'
    else:
        v['color'] = 'red'
        v['label'] = ''

ig.plot(graph_raw, target=f'./{network_name}/network_raw.pdf')

# %% Customize to remove some edges
# graph_custom = copy.deepcopy(graph_raw)
# edge_list = graph_custom.get_edgelist()
# to_delete = [(7, 10), (8, 10), (0, 5), (0, 1)]

# # Convert edge tuples to edge indices
# edge_indices_to_delete = []
# for edge_tuple in to_delete:
#     edge = graph_custom.get_eid(edge_tuple[0], edge_tuple[1], directed=False)
#     if edge != -1:  # Check if the edge exists in the graph
#         edge_indices_to_delete.append(edge)

# graph_custom.delete_edges(edge_indices_to_delete)

# ig.plot(graph_custom)
# ig.plot(graph_custom, target=f'./{network_name}/network_custom.pdf')
# ig.write(graph_custom, f'./{network_name}/network_custom.gml')
