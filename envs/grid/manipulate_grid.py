#%% This is a special file for manual modification of the grid network, to control segregation and connections between communities.
import copy
import igraph as ig
import numpy as np
from network import Network
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 22})
TITLE_FONT_SIZE = 28
SUBTITLE_FONT_SIZE = 26
LEGEND_FONT_SIZE = 16

env = 'GRID_5x10_0.5_[0.8]'
graph = ig.read(f'./{env}/network.gml')

def delete_edges(graph, to_delete):
    # Convert edge tuples to edge indices
    edge_indices_to_delete = []
    for edge_tuple in to_delete:
        edge = graph.get_eid(edge_tuple[0], edge_tuple[1], directed=False)
        if edge != -1:  # Check if the edge exists in the graph
            edge_indices_to_delete.append(edge)

    graph.delete_edges(edge_indices_to_delete)

to_delete = [(44, 45), (44, 35), (34, 45), (34, 35), (34, 25), (24, 35), (14, 15), (14, 25), (24, 15), (4, 15), (14, 5), (4, 5), 
             (9, 18), (8, 19), (19, 28), (18, 29), (28, 39), (29, 38), (38, 49), (39, 48),
             (8, 17), (7, 18), (18, 27), (17, 28), (27, 38), (28, 37), (37, 48), (38, 47),
             (6, 17), (7, 16), (16, 27), (17, 26), (26, 37), (27, 36), (36, 47), (37, 46),
             (5, 16), (6, 15), (15, 26), (25, 16), (25, 36), (35, 26), (35, 46), (45, 36),
            ]
graph_new = copy.deepcopy(graph)
delete_edges(graph_new, to_delete)
ig.plot(graph_new, layout=graph_new.layout('grid', width=10))

ig.write(graph_new, f'./{env}/network_disconnected_disadvantage.gml')

# %%
ig.plot(graph_new, layout=graph_new.layout('grid', width=10))

# %%
