#%% This explores and visualizes the basics of the toy segregated example.
import igraph as ig
import geopandas as gpd
import numpy as np
import networkx as nx
import pandas as pd
import math

G = nx.read_adjlist('./network_adjlist.txt')
nx.draw(G, with_labels=True)

# Optional - convert to igraph because calculating shortest paths is faster.
G = ig.Graph.from_networkx(G)

nodes = pd.read_csv('./network_node_attributes.csv', index_col=0)
facilities = pd.read_csv('./network_facilities.csv', index_col=0)

nodes['pct_grp_1'] = 1 - nodes['pct_grp_0']
nodes['pop_grp_0'] = nodes['pop'] * nodes['pct_grp_0']
nodes['pop_grp_1'] = nodes['pop'] - nodes['pop_grp_0']

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
# %%
