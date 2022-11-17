#%% This aims at exploring the Amsterdam environment and calculating travel times 
# from neighborhoods (nodes) to points of interest (also nodes) via the transport network (edges).
# Edges in the netwrok are weighed based on the travel time (in minutes) it takes to traverse them.
# igraph is used to calculate shortest routes from node to node, because it is much faster than networkx.
import igraph as ig
import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree
import math
import matplotlib.pyplot as plt
EARTH_RADIUS_M = 6_371_009

def nearest_nodes_to_points(G, X, Y, return_dist=False):
    """OSMNX nearest_nodes function adapted to igraph 
    from https://github.com/gboeing/osmnx/blob/main/osmnx/distance.py: nearest_nodes
    For a given set of geographical locations (X, Y), return the nearest nodes of the given graph G.

    Args:
        G (igraph.Graph): input graph
        X (pandas.Series): X coordinates of the input points
        Y (pandas.Series): Y coordinages of the input points
        return_dist (bool, optional): If True the distance to the nearest node for all points is returned. Defaults to False.

    Returns:
        list: list of nodes of graph G
    """
    if np.isnan(X).any() or np.isnan(Y).any():  # pragma: no cover
        raise ValueError("`X` and `Y` cannot contain nulls")
    
    # node_ids = np.array([node['node_id'] for node in G.vs])
    nodes = np.array([[node['y'], node['x']] for node in G.vs])

    nodes_rad = np.deg2rad(nodes)
    points_rad = np.deg2rad(np.array([Y, X]).T)

    dist, pos = BallTree(nodes_rad, metric='haversine').query(points_rad, k=1)
    dist = dist[:, 0] * EARTH_RADIUS_M  # convert radians -> meters

    nn = G.vs[pos[:, 0].tolist()]
    nn = list(nn)
    dist = dist.tolist()

    if return_dist:
        return nn, dist
    else:
        return nn
# %%
 # Read the transit network - with transfers
# G_transit_transfer = ig.read('./ams_transit_network_transfer_hw.gml')
G_transit = ig.read('./ams_transit_network_no_transfer_hw.gml')

# Read Amsterdam Vaccine Centers
ams_vc = gpd.read_file('./ams-vc-locations.geojson')
# Read Amsterdam neighborhoods
ams_nb = gpd.read_file('./ams-neighbourhoods.geojson')
ams_nb['centroid'] = gpd.points_from_xy(ams_nb.cent_x, ams_nb.cent_y, crs='EPSG:4326')
ams_nb['res_centroid'] = gpd.points_from_xy(ams_nb.res_cent_x, ams_nb.res_cent_y, crs='EPSG:4326')
# Places without residential buildings have no residential centroids. Find them and assign to them the geographical centroid.
ams_nb.loc[ams_nb['res_cent_x'].isna(), 'res_centroid'] = ams_nb[ams_nb['res_cent_x'].isna()]['centroid']

# Read Amsterdam Socio-economic variables

# %% To calculate travel times within the network, each starting and ending point needs to be assigned to a node
# For each neighborhood, get its nearest node in the network.
nb_nodes = nearest_nodes_to_points(G_transit, ams_nb['res_centroid'].x, ams_nb['res_centroid'].y)
# For each VC, get its nearest node in the network.
vc_nodes = nearest_nodes_to_points(G_transit, ams_vc['geometry'].x, ams_vc['geometry'].y)

 # Calculate travel times between all neighborhoods and all vaccine centers.
# tt_mx.shape = (nr of neighborhoods (origins), nr of vaccine centers (destinations))
tt_mx = np.ndarray((len(nb_nodes), len(vc_nodes)))
# Using enumerate as it is the more efficient (this function cannot be vectorized)
for i, o in enumerate(nb_nodes):
    for j, d in enumerate(vc_nodes):
        sp = G_transit.shortest_paths(o, d, weights='travel_time')[0][0]
        if sp == math.inf:
            print(f'Failed for {o["node_id"]} - {d["node_id"]}: No path found between them')
            continue
        tt_mx[i, j] = sp

print(f'Travel time between {ams_nb.iloc[0]["BU_NAAM"]} and {ams_vc.iloc[0]["name"]} is {round(tt_mx[0,0], 2)} minutes.')
print(f'Travel time between {ams_nb.iloc[0]["BU_NAAM"]} and {ams_vc.iloc[1]["name"]} is {round(tt_mx[0,1], 2)} minutes.')
print(f'Travel time between {ams_nb.iloc[1]["BU_NAAM"]} and {ams_vc.iloc[0]["name"]} is {round(tt_mx[1,0], 2)} minutes.')
# %%
