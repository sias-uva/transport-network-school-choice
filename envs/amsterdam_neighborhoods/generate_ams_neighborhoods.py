# Create an unweighted graph of the Amsterdam neighborhood network.
# This is done in the style of the "Quantifying ethnic segregation in cities through random walks paper."
# Each neighborhood is a node, and neighboring neighborhoods are connected by an (unweighted) edge.

#%%
import pandas as pd
import geopandas as gpd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

ams_nb = gpd.read_file('./ams-diemen-duiven-neighbourhoods.geojson')
ams_ses = pd.read_csv('./ams-diemen-duiven-ses.csv')
# Rename it to avoid confusion between the real and the generated population.
ams_ses = ams_ses.rename(columns={'pop': 'real_pop'})

# Schools data
ams_schools = gpd.read_file('./ams-schools.geojson', )
ams_schools.geometry = gpd.points_from_xy(x=ams_schools['Lng'], y=ams_schools['Lat'])
# Keep schools with capacity data.
ams_schools = ams_schools.dropna(subset=['Capacity'])
# Attach the neighborhood to each school.
# This will filter out schools that are not in amsterdam.
ams_schools = gpd.sjoin(ams_schools, ams_nb, how="inner", predicate="within").drop(['index_right', 'cent_x', 'cent_y'], axis=1)
# Reset the id after spatial join
ams_schools = ams_schools.drop(['id'], axis=1).reset_index(drop=True).reset_index().rename(columns={'index': 'id'})
# gpd.sjoin(ams_schools, ams_nb, how="inner", op="within")
ams_schools['quality'] = 1
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

graph.vs['label'] = [v.index for v in graph.vs]
graph.vs['label_size'] = 5

ig.plot(graph, layout=[(v['y'], v['x']) for v in graph.vs], 
        vertex_size=10, target='./network.pdf')
ig.write(graph, 'network.gml')

# %% Generate population of agents for each node in the network.
graph = ig.Graph.Read('network.gml')
ams_nb['node_id'] = [v.index for v in nb_nodes]
generated_pop = 1000
# Keep only relevant ses attributes.
ams_ses = ams_ses[['BU_CODE', 'real_pop', 'nr_dutch', 'nr_w_migr', 'nr_nw_migr']]
ams_ses.loc[:, 'nr_dutch_w_migr'] = ams_ses['nr_dutch'] + ams_ses['nr_w_migr']
ams_ses.loc[:, 'real_pop_pct'] = ams_ses['real_pop'] / ams_ses['real_pop'].sum()
ams_nb = ams_nb.merge(ams_ses, on='BU_CODE')
# List of socio-economic attributes to base the group definitions on.
ses_groups = ['nr_dutch_w_migr', 'nr_nw_migr']

#%%
agents = []
for i in range(generated_pop):
    node = np.random.choice(ams_nb['node_id'], p=ams_nb['real_pop_pct'])
    node_ses = ams_nb[ams_nb['node_id'] == node]
    group = np.random.choice(ses_groups, p=[node_ses.iloc[0][ses_groups[0]]/node_ses.iloc[0]['real_pop'],
                                            node_ses.iloc[0][ses_groups[1]]/node_ses.iloc[0]['real_pop']])
    agents.append({'id': i, 'node': node, 'group': group})

ams_agents = pd.DataFrame(agents)
ams_agents.to_csv(f'population_{generated_pop}.csv', index=False)
# %% Generate the facilities - for now just toy data until we have the schools

# facilities = []
# for i, f_i in ams_schools.iterrows():
#     # Skip schools with unknown capacity
#     if f_i['Popularity']:
#         node_id = ams_nb[ams_nb['BU_NAAM'] == f_i['BU_NAAM']].iloc[0]['node_id']
#         facilities.append({'id': i,
#                            'node': node_id,
#                            'facility': f_i['Name'],
#                            'capacity': f_i['Capacity'],
#                            'popularity': f_i['Popularity'],
#                            'quality': f_i['quality']})
# pd.DataFrame(facilities).to_csv('facilities.csv', index=False)

ams_nodes = pd.DataFrame([(n['code'], n.index) for n in nb_nodes], columns=('BU_CODE', 'node_id'))
ams_schools = ams_schools[['id', 'Name', 'BU_CODE', 'Capacity', 'Popularity', 'quality']] \
                        .merge(ams_nodes, on='BU_CODE').rename(columns={'node_id': 'node', 'Name': 'facility', 'Capacity': 'capacity', 'Popularity': 'popularity'})
ams_schools.to_csv('schools.csv', index=False)
#%% Plot the environment real population and generated population side by side, so we can compare the distribution.
ams_nb = ams_nb.merge(ams_agents.groupby('node')['id'].count().rename('gen_pop'), left_on='node_id', right_on='node', how='left')
ams_nb = ams_nb.merge(ams_agents[ams_agents['group'] == ses_groups[0]].groupby('node')['id'].count().rename(f'{ses_groups[0]}_in_node'), left_on='node_id', right_on='node', how='left')
ams_nb = ams_nb.merge(ams_agents[ams_agents['group'] == ses_groups[1]].groupby('node')['id'].count().rename(f'{ses_groups[1]}_in_node'), left_on='node_id', right_on='node', how='left')

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
# %% Create fully connected
# adj_mx = graph.get_adjacency()

# for i in range(adj_mx.shape[0]):
#     print('added edges for node', i)
#     adj_mx = graph.get_adjacency()

#     edges_to_add = [(i, j) for j in range(adj_mx.shape[0]) if adj_mx[i, j] == 0]
    
#     graph.add_edges(edges_to_add)

# ig.plot(graph, layout=[(v['y'], v['x']) for v in graph.vs], 
#         vertex_size=10, target='./network_full.pdf')

# ig.write(graph, 'network_full.gml')
# %%
