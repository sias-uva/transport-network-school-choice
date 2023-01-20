#%%
import igraph as ig
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Generate Graph
random.seed(42)
np.random.seed(42)
g0 = 6
g1 = 6
# sample from probability distribution of population size of the majority of the group of each node.
maj_pop_pct = [0.6, 0.8, 0.9]
# maj_pop_pct = [1]
p_in = 0.7
p_out = 0.01
network_name = f"SBM_{g0}_{g1}_{p_in}_{p_out}"

graph = ig.Graph.SBM(g0+g1, [(p_in, p_out), (p_out, p_in)], [g0, g1])
graph.vs['label'] = graph.vs.indices
graph.vs['label_size'] = 10

for i in graph.vs.indices:
    if graph.vs[i].index < g0:
        graph.vs[i]['color'] = 'blue'
    else:
        graph.vs[i]['color'] = 'red'

# ig.plot(graph, vertex_size=20)
ig.plot(graph, vertex_size=20, target=f'./{network_name}.pdf')
ig.write(graph, f'{network_name}.gml')

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(graph.degree())
fig.suptitle('Degree distrubtion')
# fig.savefig(f'./{network_name}_degree_distribution.png')

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(graph.closeness())
fig.suptitle('Closeness distrubtion')
# fig.savefig(f'./{network_name}_closeness_distribution.png')
# %% Generate Population of agents for each node in the network.
np.random.seed(42)
total_pop = 100
graph = ig.Graph.Read(f'{network_name}.gml')

nodes = pd.DataFrame({attr: graph.vs[attr] for attr in graph.vertex_attributes()})
nodes['id'] = nodes['id'].astype(int)
nodes.loc[:, 'closeness'] = graph.closeness()

nodes.loc[nodes['id'] < g0 ,'g0_pct'] = np.random.choice(maj_pop_pct, g0)
nodes.loc[nodes['id'] < g0 ,'g1_pct'] = 1 - nodes.loc[nodes['id'] < g0 ,'g0_pct']

nodes.loc[nodes['g0_pct'].isna(), 'g1_pct'] = np.random.choice(maj_pop_pct, g1)
nodes.loc[nodes['g0_pct'].isna(), 'g0_pct'] = 1 - nodes.loc[nodes['g0_pct'].isna(), 'g1_pct']

nodes.loc[:, 'g0_in_node'] = nodes['g0_pct'] / nodes['g0_pct'].sum()
nodes.loc[:, 'g1_in_node'] = nodes['g1_pct'] / nodes['g1_pct'].sum()

agents = []
for i in range(total_pop):
    node = np.random.choice(nodes['id'])
    group = np.random.choice(['g0', 'g1'], p=[nodes.loc[nodes['id'] == node, 'g0_pct'].values[0], 
                                              nodes.loc[nodes['id'] == node, 'g1_pct'].values[0]])
    agents.append({'id': i, 'node': node, 'group': group})

agents = pd.DataFrame(agents)
agents.to_csv(f'./population_{network_name}.csv', index=False)

#%% Generate facilities
fac_nodes = []

# Add two facilities on each community, one with the highest closeness and one with the lowest closeness.
fac_nodes.append(nodes.loc[nodes['id'] < g0].sort_values('closeness').iloc[-1]['id'])
# facilities.append(nodes.loc[nodes['id'] < g0].sort_values('closeness').iloc[0]['id'])
fac_nodes.append(nodes.loc[np.isin(nodes['id'], range(g1, g0+g1))].sort_values('closeness').iloc[-1]['id'])
# facilities.append(nodes.loc[np.isin(nodes['id'], range(g1, g0+g1))].sort_values('closeness').iloc[0]['id'])

facilities = []
for i, f in enumerate(fac_nodes):
    facilities.append({'id': i, 'node': f, 'facility': f'school_{i}', 'capacity': 400, 'quality': 0.5})
    
pd.DataFrame(facilities).to_csv(f'facilities_{network_name}.csv', index=False)

# %%
