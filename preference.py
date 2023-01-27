import numpy as np

def nearest_k(tt_mx, k):
    """Returns the nearest k facilities, based on the tt_mx (travel-time matrix.)

    Args:
        tt_mx (np.array): 2-D array where (i, j) is the travel time for agent i to facility j
        k (int): nearest k facilities to return

    Returns:
        np.array: indices of nearest k facilities (indices=ids)
    """

    assert k <= tt_mx.shape[1], f'Cannot pick nearest {k} out of {tt_mx.shape[1]} facilities' 
    return tt_mx.argsort()[:, :k], tt_mx


def toy_model(tt_mx, qualities):
    """
    Returns an ordered list of k preferred facilities for all agents. 
    In this preference model, the lowest a utility the more it is prefered.
    Args:
        tt_mx (np.array): 2-D array where (i, j) is the travel time for agent i to facility j
        qualities (np.array): array containing facility quality
    Returns:
        np.array: indices of nearest k facilities (indices=ids)
    """
    #normalize travel times
    tt_mxn = tt_mx / tt_mx.sum(axis=1)[:, np.newaxis]
    tt_mxn = np.divide(tt_mxn, qualities)

    return tt_mxn.argsort(), tt_mxn

def distance_popularity(tt_mx, popularity): 
    """
    Returns an ordered list of k preferred facilities for all agents, based on distance and facility popularity.
    In this preference model, the highest a utility the more it is prefered.
    Args:
        tt_mx (np.array): 2-D array where (i, j) is the travel time for agent i to facility j
        popularity (np.array): array containing facility popularity
    Returns:
        np.array: indices of facilities ordered by preference (indices=ids) and the preference matrix (utility of each facility for each agent)
    """
    # Distance matrix
    d = tt_mx.copy()
    # To avoid division by zero, we add 1 to the travel time matrix
    d += 1
    # normalize distance
    dn = d / d.sum(axis=1)[:, np.newaxis]
    # normalize popularity
    popn = popularity / popularity.sum()
    # Formula is 1 / travel_time / normalized popularity
    # We take the inverse(1/travel_time because) because the lower the travel time the higher the utility should be.
    util = np.divide(popn, dn)

    # Return the indices of the sorted utilities (descending order) and the utility matrix
    # (-util) is used as a trick to sort in descending order
    return (-util).argsort(), util
