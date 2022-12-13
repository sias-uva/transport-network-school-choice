#%%
import igraph as ig
import random

#%% Generate Graph
random.seed(42)
g0 = 20
g1 = 20
p_in = 0.5
p_out = 0.02
network_name = f"SBM_{g0}_{g1}_{p_in}_{p_out}"

graph = ig.Graph.SBM(g0+g1, [(p_in, p_out), (p_out, p_in)], [g0, g1])
graph.vs['label'] = graph.vs.indices
graph.vs['label_size'] = 10

for i in graph.vs.indices:
    if graph.vs[i].index < g0:
        graph.vs[i]['color'] = 'blue'

ig.plot(graph, vertex_size=20, target=f'./{network_name}.pdf')
ig.write(graph, f'{network_name}.gml')

# %% Generate Population of agents for each node in the network.
