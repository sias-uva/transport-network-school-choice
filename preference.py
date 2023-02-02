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
    # TODO: consider using a utility function for the popularity instead of a linear combination of distance and popularity.

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

def distance_composition(tt_mx, population, facilities, M, C_weight):
    """Distance composition preference model, based on the paper "Mechanisms for increased school segregation relative to residential segregation: a model-based analysis" by Dignum et al.
    Utility is a weighed combination of an agent's distance to each facility and its composition (pct of agent's group, which is controlled by a tolerance parameter). 
    The composition weight is controlled by C_weight, and the distance weight is 1-C_weight.

    Args:
        tt_mx (np.array): 2-D array where (i, j) is the travel time for agent i to facility j.
        population (pd.DataFrame): population dataframe, containing the tolerance parameter for each agent.
        facilities (pd.DataFrame): facilities dataframe, containing the composition of each facility.
        M (float): penalty for exceeding the tolerance, as defined in the paper.
        C_weight (float): weight of the composition utility.
    Returns:
        tuple(np.array, np.array): indices of facilities ordered by preference (indices=ids) and the preference matrix (utility of each facility for each agent)
    """
    # Distance - Normalized by the maximum and minimum travel time for each agent.
    D = np.zeros_like(tt_mx)
    D_max = np.broadcast_to(tt_mx.max(axis=1, keepdims=True), (population.shape[0], facilities.shape[0]))
    D_min = np.broadcast_to(tt_mx.min(axis=1, keepdims=True), (population.shape[0], facilities.shape[0]))

    D[tt_mx <= D_max] = (np.divide(D_max - tt_mx, D_max - D_min, out=np.zeros_like(D), where=(D_max - D_min) !=0))[tt_mx <= D_max]

    # Composition - a utility function, for more details on its defininion see the paper.
    x = np.zeros((population.shape[0], facilities.shape[0]))
    t = np.broadcast_to(population['tolerance'].values.reshape(-1, 1), (population.shape[0], facilities.shape[0]))

    for group in population.group.unique():
        x[population[population['group'] == group]['id'].values] = facilities[f'comp_{group}'].values

    C = x/t
    C[x > t] = (M + (1-x)*(1-M)/(1-t))[x > t]

    util = C_weight * C + (1-C_weight) * D
    
    # (-util) is used as a trick to sort in descending order
    return (-util).argsort(), util
