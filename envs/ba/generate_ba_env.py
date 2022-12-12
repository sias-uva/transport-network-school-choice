#%%
import igraph as ig
import random
random.seed(42)

#%% Generate Graph
graph = ig.Graph.Barabasi(n=15, m=1)
graph.vs['label'] = graph.vs.indices
graph.vs['label_size'] = 10
ig.plot(graph)

ig.plot(graph, vertex_size=20, target='./network.pdf')
ig.write(graph, 'network.gml')

# %% Generate Population of agents for each node in the network.
