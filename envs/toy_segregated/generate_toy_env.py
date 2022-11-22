#%% This explores and visualizes the basics of the toy segregated example.
import igraph as ig
import geopandas as gpd
import numpy as np
import networkx as nx
import pandas as pd
import math
import matplotlib.pyplot as plt

G = nx.read_adjlist('./network_adjlist.txt')
fig, ax = plt.subplots(figsize=(5, 5))
nx.draw(G, with_labels=True, ax=ax)
fig.savefig('network.png')

# Optional - convert to igraph because calculating shortest paths is faster.
G = ig.Graph.from_networkx(G)

nodes = pd.read_csv('./node_attributes.csv', index_col=0)
facilities = pd.read_csv('./facilities.csv', index_col=0)

# %%
# Calculate travel times between all nodes and all facilities.
# tt_mx.shape = (nr of nodes (origins), nr of facilities (destinations))
orig_nodes = nodes.index.to_list()
dest_nodes = facilities.index.to_list()
tt_mx = np.ndarray((len(orig_nodes), len(dest_nodes)))
for i, o in enumerate(orig_nodes):
    for j, d in enumerate(dest_nodes):
        sp = G.distances(o, d)[0][0]
        if sp == math.inf:
            print(f'Failed for {o["node_id"]} - {d["node_id"]}: No path found between them')
            continue
        tt_mx[i, j] = sp

nodes['nearest_fac_dist'] = tt_mx.min(axis=1)
nodes['nearest_fac_node'] = list(dest_nodes[i] for i in tt_mx.argmin(axis=1))

print(f"Average Distance to nearest facility for grp_0: {np.nansum(nodes['pct_grp_0'] * nodes['nearest_fac_dist']).round(2)}")
print(f"Average Distance to nearest facility for grp_1: {np.nansum(nodes['pct_grp_1'] * nodes['nearest_fac_dist']).round(2)}")

# %% Generate population of Agents from node attributes.
population = {'id': [], 'node': [], 'group': []}
count = -1
for i in nodes.index:
    for j in range(nodes.loc[i, 'pop_grp_0']):
        count += 1
        population['id'].append(count)
        population['node'].append(i)
        population['group'].append('grp_0')

    for k in range(nodes.loc[i, 'pop_grp_1']):
        count += 1
        population['id'].append(count)
        population['node'].append(i)
        population['group'].append('grp_1')
population = pd.DataFrame(population)

population.to_csv('./population.csv', index=False)
# %%
