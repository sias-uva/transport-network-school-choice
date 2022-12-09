# Create an unweighted graph of the Amsterdam neighborhood network.
# This is done in the style of the "Quantifying ethnic segregation in cities through random walks paper."
# Each neighborhood is a node, and neighboring neighborhoods are connected by an (unweighted) edge.

#%%
import pandas as pd
import geopandas as gpd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

ams_nb = gpd.read_file('./ams-diemen-duiven-neighbourhoods.geojson')
ams_ses = pd.read_csv('./ams-diemen-duiven-ses.csv')
# Rename it to avoid confusion between the real and the generated population.
ams_ses = ams_ses.rename(columns={'pop': 'real_pop'})
# %% Create the network from the amsterdam neighborhoods.
graph = ig.Graph()
nb_nodes = []
for row, nb in ams_nb.iterrows():
    node = graph.add_vertex(name=nb['BU_NAAM'], code=nb['BU_CODE'], x=nb['cent_x'], y=nb['cent_y'])
    nb_nodes.append(node)

for i, nb_i in ams_nb.iterrows():
    neighbors = ams_nb[ams_nb.geometry.touches(nb_i.geometry)]

    for j, nb_j in neighbors.iterrows():
        if not graph.are_connected(nb_nodes[i], nb_nodes[j]):
            graph.add_edge(nb_nodes[i], nb_nodes[j])

ig.plot(graph, layout=[(v['y'], v['x']) for v in graph.vs], 
        vertex_size=10, target='network.pdf')
ig.write(graph, 'network.gml')

# %% Generate population of agents for each node in the network.
graph = ig.Graph.Read('network.gml')
ams_nb['node_id'] = [v.index for v in nb_nodes]
generated_pop = 1050
# Keep only relevant ses attributes.
ams_ses = ams_ses[['BU_CODE', 'real_pop', 'nr_dutch', 'nr_w_migr', 'nr_nw_migr']]
ams_ses.loc[:, 'nr_dutch_w_migr'] = ams_ses['nr_dutch'] + ams_ses['nr_w_migr']

# List of socio-economic attributes to base the group definitions on.
ses_groups = ['nr_dutch_w_migr', 'nr_nw_migr']
for g in ses_groups:
    group_pop = ams_ses[g].sum() / ams_ses['real_pop'].sum() * generated_pop
    ams_ses[g + '_in_node'] = ams_ses[g] / ams_ses[g].sum() * group_pop
    ams_ses[g + '_in_node'] = ams_ses[g + '_in_node'].round().astype(int)

ams_nb = ams_nb.merge(ams_ses, on='BU_CODE')
ams_nb['gen_pop'] = ams_nb['nr_dutch_w_migr_in_node'] + ams_nb['nr_nw_migr_in_node']

agents = []
for i, nb in ams_nb.iterrows():
    for g in ses_groups:
        for j in range(nb[g + '_in_node']):
            g = g.replace('_in_node', '')
            g = g.replace('nr_', '')
            agents.append({'id': len(agents), 'node': nb['node_id'], 'group': g})

pd.DataFrame(agents).to_csv('population.csv', index=False)
# %% Generate the facilities - for now just toy data until we have the schools
# facilities = pd.DataFrame(columns=['id', 'node', 'facility', 'capacity', 'quality'])
facilities = []
facilities.append({'id': 0, 'node': 58, 'facility': 'school_0', 'capacity': 300, 'quality': 0.5})
facilities.append({'id': 1, 'node': 290, 'facility': 'school_1', 'capacity': 300, 'quality': 0.5})
facilities.append({'id': 2, 'node': 246, 'facility': 'school_2', 'capacity': 300, 'quality': 0.5})

pd.DataFrame(facilities).to_csv('facilities.csv', index=False)
#%% Plot the environment specifications.
ams_nb['group_pop_diff'] = ams_nb['nr_dutch_w_migr_in_node'] - ams_nb['nr_nw_migr_in_node']
ams_nb['group_pop_ratio'] = ams_nb['nr_dutch_w_migr_in_node'].div(ams_nb['nr_nw_migr_in_node'])
ams_nb.loc[ams_nb['nr_nw_migr_in_node'] > ams_nb['nr_dutch_w_migr_in_node'], 'group_pop_ratio'] = - ams_nb['nr_nw_migr_in_node'].div(ams_nb['nr_dutch_w_migr_in_node'])
ams_nb = ams_nb.replace([np.inf, -np.inf], np.nan)

figsize = (15, 8)
fig, axs = plt.subplots(2, 2, figsize=figsize)
ams_nb.plot('real_pop', ax=axs[0][0], legend=True)
axs[0][0].set_title("Real population")

ams_nb.plot('gen_pop', ax=axs[0][1], legend=True)
axs[0][1].set_title("Generated population")

ams_nb.plot('group_pop_diff', ax=axs[1][0], legend=True, cmap='PiYG')
axs[1][0].set_title("Group population difference (generated)")

ams_nb.plot('group_pop_ratio', ax=axs[1][1], legend=True, cmap='PiYG')
axs[1][1].set_title("Group population ratio (generated)")

fig.suptitle(f"Amsterdam environment: {ams_nb['gen_pop'].sum()} agents, {ams_nb['gen_pop'].count()} nodes, \n groups: {ses_groups}")
fig.savefig('./ams-env.png', dpi=300)

print('Successfullly generated the environment.')
# %%
