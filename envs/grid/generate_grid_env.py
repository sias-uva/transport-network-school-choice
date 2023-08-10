# %%
import os
from igraph import Graph
import igraph as ig
import numpy as np
import pandas as pd


# Define the dimensions of the grid
# FOR NOW ONLY WORKS WITH SQUARE GRID AND EVEN NUMBER OF ROWS AND COLUMNS
rows = 6
cols = 6
total_pop = 1000
maj_pop_pct = [0.8]
network_name = f"GRID_{rows}x{cols}_{maj_pop_pct}"

if not os.path.exists(network_name):
        os.makedirs(network_name)

# Create an empty graph
grid_graph = Graph(directed=False)

# Add nodes (cells) to the graph and store their vertex IDs in a dictionary
vertices = {(i, j): grid_graph.add_vertex(name=(i, j)) for i in range(rows) for j in range(cols)}

# Define the neighbors' offsets)
neighbors_offsets = [(-1, 0), (-1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (0, 1), (1, 1)]

# Add edges to connect each cell to its immediate neighbors
for i in range(rows):
    for j in range(cols):
        for offset_i, offset_j in neighbors_offsets:
            neighbor_i, neighbor_j = i + offset_i, j + offset_j
            if 0 <= neighbor_i < rows and 0 <= neighbor_j < cols:
                # Get the vertex IDs of the current node and its neighbor
                vertex_id_a = vertices[(i, j)]
                vertex_id_b = vertices[(neighbor_i, neighbor_j)]
                
                # Check if the edge already exists in the graph
                if not grid_graph.are_connected(vertex_id_a, vertex_id_b):
                    grid_graph.add_edge(vertex_id_a, vertex_id_b)

grid_graph.vs['label'] = grid_graph.vs.indices
# Print the graph to see the connections
ig.plot(grid_graph, layout=grid_graph.layout("grid"))

ig.plot(grid_graph, vertex_size=20, target=f'./{network_name}/network.pdf', layout=grid_graph.layout("grid"))
ig.write(grid_graph, f'./{network_name}/network.gml')

# %% Split the network into two groups
np.random.seed(42)
graph = ig.Graph.Read(f'./{network_name}/network.gml')
nodes = pd.DataFrame({attr: graph.vs[attr] for attr in graph.vertex_attributes()})
nodes['id'] = nodes['id'].astype(int)
nodes.loc[:, 'closeness'] = graph.closeness()

g0 = []
g1 = []

N = rows
# Group 0 (g0) in top-left and bottom-right corners
for i in range(N // 2):
    for j in range(N // 2):
        g0.append(i * N + j)
        g0.append(((N - 1 - i) * N + (N - 1 - j)))

# Group 1 (g1) in top-right and bottom-left corners
for i in range(N // 2):
    for j in range(N // 2, N):
        g1.append(i * N + j)
        g1.append(((N - 1 - i) * N + (N - 1 - j)))


# Assign colors to nodes based on their group
for i in graph.vs.indices:
    if np.isin(graph.vs[i].index, g0):
        nodes.loc[nodes['id'] == i, 'color'] = 'blue'
        nodes.loc[nodes['id'] == i, 'group'] = 0
    else:
        nodes.loc[nodes['id'] == i, 'color'] = 'red'
        nodes.loc[nodes['id'] == i, 'group'] = 1

graph.vs['label'] = graph.vs.indices
graph.vs['label_size'] = 10

ig.plot(grid_graph, layout=grid_graph.layout("grid"), vertex_size=20, vertex_color=nodes['color'])
ig.plot(grid_graph, layout=grid_graph.layout("grid"), vertex_size=20, vertex_color=nodes['color'], target=f'./{network_name}/network_groups.pdf')

#%% Generate Population of agents for each node in the network.
nodes.loc[nodes['id'].isin(g0), 'g0_pct'] = np.random.choice(maj_pop_pct, len(g0))
nodes.loc[nodes['id'].isin(g0), 'g1_pct'] = 1 - nodes.loc[nodes['id'].isin(g0) ,'g0_pct']

nodes.loc[nodes['g0_pct'].isna(), 'g1_pct'] = np.random.choice(maj_pop_pct, len(g1))
nodes.loc[nodes['g0_pct'].isna(), 'g0_pct'] = 1 - nodes.loc[nodes['g0_pct'].isna(), 'g1_pct']

# Group distribution on nodes. The percentage of each group in each node.
nodes.loc[:, 'g0_in_node'] = nodes['g0_pct'] / nodes['g0_pct'].sum()
nodes.loc[:, 'g1_in_node'] = nodes['g1_pct'] / nodes['g1_pct'].sum()

#%%
agents = []

id = 0
for g in ['g0', 'g1']:
    for i in range(total_pop // 2):
        node = np.random.choice(nodes['id'], p=nodes[f'{g}_in_node'])
        group = g
        # Set tolerance similar to that of the majority population of the node.
        agents.append({'id': id, 'node': node, 'group': group, 
                       'tolerance': nodes[nodes['id'] == node][['g0_pct', 'g1_pct']].iloc[0].max()})
        id += 1

agents = pd.DataFrame(agents)
agents.to_csv(f'./{network_name}/population.csv', index=False)

# Adjust nodes to include the number of agents in each group and the real percentage of each group in the node.
agents_per_node = agents.groupby(['node', 'group'])['id'].count().unstack() \
                        .rename(columns={'g0': 'g0_nr', 'g1': 'g1_nr'}) \
                        .fillna(0).astype(int)
nodes = nodes.merge(agents_per_node, how='left', left_on='id', right_on='node')
nodes.loc[:, 'g0_pct'] = nodes['g0_nr'] / (nodes['g0_nr'] + nodes['g1_nr'])
nodes.loc[:, 'g1_pct'] = nodes['g1_nr'] / (nodes['g0_nr'] + nodes['g1_nr'])
nodes.loc[:, 'g0_in_node'] = nodes['g0_nr'] / nodes['g0_nr'].sum()
nodes.loc[:, 'g1_in_node'] = nodes['g1_nr'] / nodes['g1_nr'].sum()

#%% Generate facilities
fac_nodes = [7, 10, 25, 28]
facility_cap = total_pop // len(fac_nodes)

facilities = []
for i, f in enumerate(fac_nodes):
    facilities.append({'id': i, 'node': f, 'facility': f'school_{i}', 'capacity': facility_cap, 'quality': 0.5, 'popularity': 0.5})
    
facilities = pd.DataFrame(facilities)

facilities.to_csv(f'./{network_name}/facilities.csv', index=False)

# %% Dissimilaritiy index of population in the network


A = nodes['g0_nr'].sum()
B = nodes['g1_nr'].sum()
DI = 0
for _, nid in nodes.iterrows():
    a = nid['g0_nr']
    b = nid['g1_nr']

    DI += np.abs(a/A - b/B)

print(f'Dissimilarity Index: {1/2 * DI}') 


# %%
