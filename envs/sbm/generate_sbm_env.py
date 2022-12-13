#%%
import igraph as ig
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Generate Graph
random.seed(42)
g0 = 20
g1 = 20
# sample from probability distribution of population size of the majority of the group of each node.
maj_pop_pct = [0.6, 0.8, 0.9]
p_in = 0.4
p_out = 0.02
network_name = f"SBM_{g0}_{g1}_{p_in}_{p_out}"

graph = ig.Graph.SBM(g0+g1, [(p_in, p_out), (p_out, p_in)], [g0, g1])
graph.vs['label'] = graph.vs.indices
graph.vs['label_size'] = 10

for i in graph.vs.indices:
    if graph.vs[i].index < g0:
        graph.vs[i]['color'] = 'blue'
    else:
        graph.vs[i]['color'] = 'red'

ig.plot(graph, vertex_size=20, target=f'./{network_name}.pdf')
ig.write(graph, f'{network_name}.gml')

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(graph.degree())
fig.suptitle('Degree distrubtion')
fig.savefig(f'./{network_name}_degree_distribution.png')

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(graph.closeness())
fig.suptitle('Closeness distrubtion')
fig.savefig(f'./{network_name}_closeness_distribution.png')
# %% Generate Population of agents for each node in the network.
total_pop = 20000
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
nodes.loc[:, 'population'] = total_pop / nodes.shape[0]

nodes.loc[:, 'g0_pop'] = (nodes['population'] * nodes['g0_in_node']).round().astype(int)
nodes.loc[:, 'g1_pop'] = (nodes['population'] * nodes['g1_in_node']).round().astype(int)

agents = []
for i, row in nodes.iterrows():
    for g in ['g0', 'g1']:
        for j in range(row[g + '_pop']):
            agents.append({'id': len(agents), 'node': row['id'], 'group': g})

agents = pd.DataFrame(agents)
agents.to_csv('./population.csv', index=False)

#%% Generate facilities
total_facilities = 4
facilities = []

# Add two facilities on each community, one with the highest closeness and one with the lowest closeness.
fac1 = nodes.loc[nodes['id'] < g0].sort_values('closeness').iloc[-1]['id']
fac2 = nodes.loc[nodes['id'] < g0].sort_values('closeness').iloc[0]['id']
fac3 = nodes.loc[np.isin(nodes['id'], range(g1, g0+g1))].sort_values('closeness').iloc[-1]['id']
fac4 = nodes.loc[np.isin(nodes['id'], range(g1, g0+g1))].sort_values('closeness').iloc[0]['id']

for i, f in enumerate([fac1, fac2, fac3, fac4]):
    facilities.append({'id': i, 'node': f, 'facility': f'school_{i}', 'capacity': 400, 'quality': 0.5})

pd.DataFrame(facilities).to_csv('facilities.csv', index=False)

# %%
