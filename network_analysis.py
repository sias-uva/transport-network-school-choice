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
#%%
# Load the graph
grid_nw = Network(f'./envs/grid/GRID_10x10_0.7_[0.8]/network.gml', calc_tt_mx=False)
amsnb_nw = Network(f'./envs/amsterdam_neighborhoods/network.gml', calc_tt_mx=False)

grid_pop = pd.read_csv('./envs/grid/GRID_10x10_[0.8]/population.csv')
grid_groups = grid_pop.group.unique()
grid_grp_dist = [(grid_pop[grid_pop['group'] == gid].groupby('node')['id'].count() 
                  / grid_pop[grid_pop['group'] == gid]['id'].count())
                    .reindex(grid_nw.network.vs.indices, fill_value=0) 
                    for gid in grid_groups]

ams_pop = pd.read_csv('./envs/amsterdam_neighborhoods/population_7000.csv')
ams_groups = ams_pop.group.unique()
ams_grp_dist = [(ams_pop[ams_pop['group'] == gid].groupby('node')['id'].count()
                  / ams_pop[ams_pop['group'] == gid]['id'].count()) 
                  .reindex(amsnb_nw.network.vs.indices, fill_value=0)
                  for gid in ams_groups]
# %%
bins = 10

fig, axs = plt.subplots(2, 3, figsize=(20, 15))
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

fig.suptitle('Distribution of Centrality Measures', fontsize=32)
fig.tight_layout()
# %%
fig, axs = plt.subplots(2, 3, figsize=(20, 15))
axs[0][0].hist(grid_nw.weighted_closeness(weights=grid_grp_dist[0]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=grid_groups[0])
axs[0][0].hist(grid_nw.weighted_closeness(weights=grid_grp_dist[1]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=grid_groups[1])
axs[0][0].set_title('Grid - Closeness')
axs[0][1].hist(grid_nw.weighted_betweeness(weights=grid_grp_dist[0]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=grid_groups[0])
axs[0][1].hist(grid_nw.weighted_betweeness(weights=grid_grp_dist[1]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=grid_groups[1])
axs[0][1].set_title('Grid - Betweenness')
axs[0][2].hist(grid_nw.weighted_degree(weights=grid_grp_dist[0]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=grid_groups[0])
axs[0][2].hist(grid_nw.weighted_degree(weights=grid_grp_dist[1]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=grid_groups[1])
axs[0][2].set_title('Grid - Degree')

axs[1][0].hist(amsnb_nw.weighted_closeness(weights=ams_grp_dist[0]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=ams_groups[0])
axs[1][0].hist(amsnb_nw.weighted_closeness(weights=ams_grp_dist[1]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=ams_groups[1])
axs[1][0].set_title('Amsterdam - Closeness')
axs[1][1].hist(amsnb_nw.weighted_betweeness(weights=ams_grp_dist[0]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=ams_groups[0])
axs[1][1].hist(amsnb_nw.weighted_betweeness(weights=ams_grp_dist[1]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=ams_groups[1])
axs[1][1].set_title('Amsterdam - Betweenness')
axs[1][2].hist(amsnb_nw.weighted_degree(weights=ams_grp_dist[0]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=ams_groups[0])
axs[1][2].hist(amsnb_nw.weighted_degree(weights=ams_grp_dist[1]), bins=bins, edgecolor='black', density=False, alpha=0.5, label=ams_groups[1])
axs[1][2].set_title('Amsterdam - Degree')

fig.suptitle('Distribution of Group-Based Centrality Measures')
axs[1, 1].legend()
fig.tight_layout()

# %%
