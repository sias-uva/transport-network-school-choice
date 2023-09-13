#%%
import igraph as ig
import numpy as np
from network import Network
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 22})
TITLE_FONT_SIZE = 28
SUBTITLE_FONT_SIZE = 26
LEGEND_FONT_SIZE = 16

gridenv = 'grid/GRID_10x10_[0.8]'
amsenv = 'amsterdam_neighborhoods'
# sbmenv = 'sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.5'

sbmenv = 'sbm/SBM_6_6_0.3_0.3_pop_1000_maj_pop_pct_0.8'
#%%
# Load the graph
grid_nw = Network(f'./envs/{gridenv}/network.gml', calc_tt_mx=False)
amsnb_nw = Network(f'./envs/{amsenv}/network.gml', calc_tt_mx=False)
sbm_nw = Network(f'./envs/{sbmenv}/network.gml', calc_tt_mx=False)

grid_facilities = pd.read_csv(f'./envs/{gridenv}/facilities.csv')
amsnb_facilities = pd.read_csv(f'./envs/{amsenv}/schools.csv')
sbm_facilities = pd.read_csv(f'./envs/{sbmenv}/facilities.csv')

grid_pop = pd.read_csv(f'./envs/{gridenv}/population.csv')
grid_groups = grid_pop.group.unique()
grid_grp_dist = [(grid_pop[grid_pop['group'] == gid].groupby('node')['id'].count() 
                  / grid_pop[grid_pop['group'] == gid]['id'].count())
                    .reindex(grid_nw.network.vs.indices, fill_value=0) 
                    for gid in grid_groups]

ams_pop = pd.read_csv(f'./envs/{amsenv}/population_7000.csv')
ams_groups = ams_pop.group.unique()
ams_grp_dist = [(ams_pop[ams_pop['group'] == gid].groupby('node')['id'].count()
                  / ams_pop[ams_pop['group'] == gid]['id'].count()) 
                  .reindex(amsnb_nw.network.vs.indices, fill_value=0)
                  for gid in ams_groups]

sbm_pop = pd.read_csv(f'./envs/{sbmenv}/population.csv')
sbm_groups = sbm_pop.group.unique()
sbm_grp_dist = [(sbm_pop[sbm_pop['group'] == gid].groupby('node')['id'].count()
                  / sbm_pop[sbm_pop['group'] == gid]['id'].count()) 
                  .reindex(sbm_nw.network.vs.indices, fill_value=0)
                  for gid in sbm_groups]
# %%
bins = 10

fig, axs = plt.subplots(3, 3, figsize=(20, 15))
axs[0][0].hist(grid_nw.network.closeness(), bins=bins, edgecolor='black', density=False)
axs[0][0].set_title('Grid - Closeness')
axs[0][1].hist(grid_nw.network.betweenness(), bins=bins, edgecolor='black', density=False)
axs[0][1].set_title('Grid - Betweenness')
axs[0][2].hist(grid_nw.network.degree(), bins=bins, edgecolor='black', density=False)
axs[0][2].set_title('Grid - Degree')

axs[1][0].hist(amsnb_nw.network.closeness(), bins=bins, edgecolor='black', density=False)
axs[1][0].set_title('Amsterdam - Closeness')
axs[1][1].hist(amsnb_nw.network.betweenness(), bins=bins, edgecolor='black', density=False)
axs[1][1].set_title('Amsterdam - Betweenness')
axs[1][2].hist(amsnb_nw.network.degree(), bins=bins, edgecolor='black', density=False)
axs[1][2].set_title('Amsterdam - Degree')


axs[2][0].hist(sbm_nw.network.closeness(), bins=bins, edgecolor='black', density=False)
axs[2][0].set_title('SBM - Closeness')
axs[2][1].hist(sbm_nw.network.betweenness(), bins=bins, edgecolor='black', density=False)
axs[2][1].set_title('SBM - Betweenness')
axs[2][2].hist(sbm_nw.network.degree(), bins=bins, edgecolor='black', density=False)
axs[2][2].set_title('SBM - Degree')

fig.suptitle('Distribution of Centrality Measures', fontsize=32)
fig.tight_layout()
# %%
def plot_histogram(ax, data, label, bin_edges, color, title=None):
    ax.hist(data, bins=bin_edges, edgecolor='black', density=False, alpha=0.5, color=color, label=label)
    ax.axvline(data.mean(), linestyle='dashed', linewidth=2, color=color, alpha=0.5)
    ax.set_title(title)

    return ax
colors = ['#1f77b4', '#ff7f0e']

def plot_all_histograms(networks, nodes, network_names, groups, group_distributions, group_colors, figsize=(20, 15), title=None):
  """Plots a grid of histograms for all centrality measures for the different groups.

  Args:
      networks (list): list of networks to calculate the centrality measures for
      nodes (list): list of nodes to calculate the centrality measures for. Set to [None x len(networks)] to calculate for all nodes.
      network_names (list): list of network names
      groups (list): list of gropus
      group_distributions (list): distribution of groups within the nodes of the network
      group_colors (list): plot color for each group
  """

  fig, axs = plt.subplots(len(networks), 3, figsize=figsize)

  for i, nw in enumerate(networks):
    closeness_bin_edges = np.histogram_bin_edges(
       np.concatenate((nw.weighted_closeness(weights=group_distributions[i][0], nodes=nodes[i]), 
                       nw.weighted_closeness(weights=group_distributions[i][1], nodes=nodes[i]))), bins=bins)
    
    betweenness_bin_edges = np.histogram_bin_edges(
       np.concatenate((nw.weighted_betweeness(weights=group_distributions[i][0], nodes=nodes[i]), 
                       nw.weighted_betweeness(weights=group_distributions[i][1], nodes=nodes[i]))), bins=bins)
    
    degree_bin_edges = np.histogram_bin_edges(
       np.concatenate((nw.weighted_degree(weights=group_distributions[i][0], nodes=nodes[i]), 
                       nw.weighted_degree(weights=group_distributions[i][1], nodes=nodes[i]))), bins=bins)
    
    for grp_dist, grp, color in zip(group_distributions[i], groups[i], group_colors):
      plot_histogram(axs[i][0], nw.weighted_closeness(weights=grp_dist, nodes=nodes[i]), grp, closeness_bin_edges, color, f'{network_names[i]} - Closeness')
      plot_histogram(axs[i][1], nw.weighted_betweeness(weights=grp_dist, nodes=nodes[i]), grp, betweenness_bin_edges, color, f'{network_names[i]} - Betweenness')
      plot_histogram(axs[i][2], nw.weighted_degree(weights=grp_dist, nodes=nodes[i]), grp, degree_bin_edges, color, f'{network_names[i]} - Degree')
  if title is not None:
    fig.suptitle(title)
  else: 
    fig.suptitle('Distribution of Group-Based Centrality Measures')
  axs[1, 1].legend()
  fig.tight_layout()

plot_all_histograms([grid_nw, amsnb_nw, sbm_nw], [None, None, None], ['Grid', 'Amsterdam', 'SBM'], [grid_groups, ams_groups, sbm_groups], [grid_grp_dist, ams_grp_dist, sbm_grp_dist], colors)
plot_all_histograms([grid_nw, amsnb_nw, sbm_nw], [grid_facilities['node'].tolist(), amsnb_facilities['node'].tolist(), sbm_facilities['node'].tolist()], 
                    ['Grid', 'Amsterdam', 'SBM'], [grid_groups, ams_groups, sbm_groups], [grid_grp_dist, ams_grp_dist, sbm_grp_dist], colors, 
                    title='Distribution of Group-Based Centrality Measures | Facilities')

# %% Test  - remove links from the grid envrioment to increase disparity in closeness
# Load the graph
# grid_nw = Network(f'./envs/grid/GRID_10x10_[0.8]/network.gml', calc_tt_mx=False)
# grid_pop = pd.read_csv('./envs/grid/GRID_10x10_[0.8]/population.csv')

# grid_groups = grid_pop.group.unique()
# grid_grp_dist = [(grid_pop[grid_pop['group'] == gid].groupby('node')['id'].count() 
#                   / grid_pop[grid_pop['group'] == gid]['id'].count())
#                     .reindex(grid_nw.network.vs.indices, fill_value=0) 
#                     for gid in grid_groups]

# fig, axs = plt.subplots(1, 3, figsize=(12, 5))

# # Remove the edge from the graph
# edge_list = grid_nw.network.get_edgelist()
# to_delete = [(44, 45), (44, 45), (44, 54), (43, 53), (42, 52), (40, 51), (41, 50), (41, 52), (41, 51), 
#              (40, 50), (34, 35), (24, 25), (14, 15), (51, 42), (52, 43), (42, 53), (43, 52), (44, 53),
#              (43, 54), (44, 35), (34, 45), (24, 35), (34, 25), (24, 15), (14, 25), (14, 5), (4, 15), (4, 5)]

# # to_delete = []

# # Convert edge tuples to edge indices
# edge_indices_to_delete = []
# for edge_tuple in to_delete:
#     edge = grid_nw.network.get_eid(edge_tuple[0], edge_tuple[1], directed=False)
#     if edge != -1:  # Check if the edge exists in the graph
#         edge_indices_to_delete.append(edge)

# grid_nw.network.delete_edges(edge_indices_to_delete)

# bin_edges = np.histogram_bin_edges(np.concatenate((grid_nw.weighted_closeness(weights=grid_grp_dist[0]), 
#                                               grid_nw.weighted_closeness(weights=grid_grp_dist[1]))), bins=bins)

# axs[0].hist(grid_nw.weighted_closeness(weights=grid_grp_dist[0]), bins=bin_edges, edgecolor='black', density=False, alpha=0.5, label=grid_groups[0])
# axs[0].axvline(grid_nw.weighted_closeness(weights=grid_grp_dist[0]).mean(), linestyle='dashed', linewidth=2)
# axs[0].hist(grid_nw.weighted_closeness(weights=grid_grp_dist[1]), bins=bin_edges, edgecolor='black', density=False, alpha=0.5, label=grid_groups[1])
# axs[0].axvline(grid_nw.weighted_closeness(weights=grid_grp_dist[1]).mean(), linestyle='dashed', linewidth=2)
# axs[0].set_title('Closeness')


# bin_edges = np.histogram_bin_edges(np.concatenate((grid_nw.weighted_betweeness(weights=grid_grp_dist[0]), 
#                                               grid_nw.weighted_betweeness(weights=grid_grp_dist[1]))), bins=bins)
# axs[1].hist(grid_nw.weighted_betweeness(weights=grid_grp_dist[0]), bins=bin_edges, edgecolor='black', density=False, alpha=0.5, label=grid_groups[0])
# axs[1].hist(grid_nw.weighted_betweeness(weights=grid_grp_dist[1]), bins=bin_edges, edgecolor='black', density=False, alpha=0.5, label=grid_groups[1])
# axs[1].axvline(grid_nw.weighted_betweeness(weights=grid_grp_dist[0]).mean(), linestyle='dashed', linewidth=2)
# axs[1].axvline(grid_nw.weighted_betweeness(weights=grid_grp_dist[1]).mean(), linestyle='dashed', linewidth=2)
# axs[1].set_title('Betweenness')

# bin_edges = np.histogram_bin_edges(np.concatenate((grid_nw.weighted_degree(weights=grid_grp_dist[0]), 
#                                               grid_nw.weighted_degree(weights=grid_grp_dist[1]))), bins=bins)
# axs[2].hist(grid_nw.weighted_degree(weights=grid_grp_dist[0]), bins=bin_edges, edgecolor='black', density=False, alpha=0.5, label=grid_groups[0])
# axs[2].hist(grid_nw.weighted_degree(weights=grid_grp_dist[1]), bins=bin_edges, edgecolor='black', density=False, alpha=0.5, label=grid_groups[1])
# axs[2].axvline(grid_nw.weighted_degree(weights=grid_grp_dist[0]).mean(), linestyle='dashed', linewidth=2)
# axs[2].axvline(grid_nw.weighted_degree(weights=grid_grp_dist[1]).mean(), linestyle='dashed', linewidth=2)

# axs[2].set_title('Degree')

# ig.plot(grid_nw.network, layout=grid_nw.network.layout("grid"))
# %%
