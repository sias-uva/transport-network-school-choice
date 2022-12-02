import numpy as np


def facility_rank_distribution(pref_list, total_facilities, return_avg_pos_by_fac=False):
    """Returns a numpy array of size (facility_size, max nr of preferences) where each element in the array is the number of agents that have that facility as their n-th preference.

    Args:
        pref_list (np.array): array of size (nr of agents, nr of preferences) where each facility is sorted by preference.
        total_facilities (int): total number of facilities.
        return_avg_pos_by_fac (bool, optional): whether to return the average position of each facility in the preference list. Defaults to False.
    Returns:
        - np.array: array of size (facility_size, max nr of preferences) where each element in the array is the number of agents that have that facility as their n-th preference.
        - np.array: (if return_avg_pos_by_fac is True) array of size (facility_size, 1) where each element is the average position of that facility in the preference list.
    """
    # Don't know if we can speed this up, because preference lists can have -1 values for agents that don't have preferences over all facilities.
    pref_position_by_facility = np.zeros((total_facilities, pref_list.shape[1]))
    for i in range(total_facilities):
        for j in range(pref_list.shape[1]):
            pref_position_by_facility[i, j] = np.sum(pref_list[:, j] == i)
    
    if return_avg_pos_by_fac:
        avg_pos_by_fac = (pref_position_by_facility * np.arange(1, pref_position_by_facility.shape[1] + 1)).sum(axis=1)/pref_list.shape[0]
        return pref_position_by_facility, avg_pos_by_fac
    else:    
        return pref_position_by_facility

def facility_capacity(population, facilities, allocation, return_pct=True):
    """Returns an array of percentage of satisfied capacity per facility.

    Args:
        population (pd.DataFrame): population dataframe - should have id column
        facilities (pd.DataFrame): facility dataframe - should have id column
        allocation (np.array): array of shape (total_pop, 1) where each agent is allocated to 1 facility.
        return_pct (bool, optional): whether to return the percentage of filled capacity. Defaults to True.

    Returns:
        array: array of satisfied capacity per facility.
    """
    # TODO - probably best to transfer this assert to the allocation method.
    assert allocation.shape[1] == 1, 'Only one facility should be allocated to each agent.'
    
    alloc_by_facility = []
    capacity_pct = []
    for fid in facilities['id'].values:
        allocs = np.sum(allocation == fid)
        alloc_by_facility.append(allocs)
        capacity_pct.append(allocs / facilities.loc[facilities['id'] == fid]['capacity'].values[0])

    if return_pct:
        return alloc_by_facility, capacity_pct
    else:
        return alloc_by_facility

def facility_group_composition(population, facilities, allocation, return_pct=True):
    """ Returns 2-D array of the group composition in each facility - shape = (nr_facilities, nr_groups) where (i,j) = allocated agents of gorup j in facility i.
    If return_pct is True, returns another 2-D array with the percentage of agents of each group in each facility.

    Args:
        population (pd.DataFrame): population dataframe - should have id column
        facilities (pd.DataFrame): facility dataframe - should have id column
        allocation (np.array): array of shape (total_pop, 1) where each agent is allocated to 1 facility.
        return_pct (bool, optional): whether to return the percentage of each group in each facility. Defaults to True.

    Returns:
        - np.array: allocated agents of per group per facility.
        - list: (if return_pct is True) [allocated agents of per group per facility, percentage of each group in each facility].
    """
    # TODO - probably best to transfer this assert to the allocation method.
    assert allocation.shape[1] == 1, 'Only one facility should be allocated to each agent.'

    # I don't like the for-based solution but too lazy atm.
    alloc_by_facility = np.zeros((facilities.shape[0], population['group'].nunique()))
    for gid, g in enumerate(population['group'].unique()):
        pop_id = population[population['group'] == g]['id'].values

        for fid in facilities['id'].values:
            allocs = np.sum(allocation[pop_id] == fid)
            alloc_by_facility[fid, gid] = allocs

    if return_pct:
        return alloc_by_facility, np.true_divide(alloc_by_facility, alloc_by_facility.sum(axis=1, keepdims=True))
    else:
        return alloc_by_facility

def dissimilarity_index(population, facilities, allocation, group_composition=None):
    """Calculates the dissimilarity index of the whole school allocation. It is calculated as the sum of the dissimilarity index of each facility.
    The index is the sum of the squared differences between the percentage of each group in each facility and the percentage of each group in the whole population.
    More on the index here: https://en.wikipedia.org/wiki/Index_of_dissimilarity

    Args:
        population (pd.DataFrame): population dataframe - should have id column
        facilities (pd.DataFrame): facility dataframe - should have id column
        allocation (np.array): array of shape (total_pop, 1) where each agent is allocated to 1 facility.
        group_composition (np.array, optional): if given, it will not call facility_group_composition to re-calculate it . Defaults to None.

    Returns:
        _type_: _description_
    """
    # For performance issues - we mostly calculate the group composition on a previous step anyway, so we can skip the re-calculation.
    if group_composition is None:
        group_composition = facility_group_composition(population, facilities, allocation, return_pct=False)

    A = group_composition[:, 0].sum()
    B = group_composition[:, 1].sum()
    DI = 0
    for fid in range(facilities.shape[0]):
        a = group_composition[fid, 0]
        b = group_composition[fid, 1]
        DI += np.abs(a/A - b/B)
    
    return 1/2 * DI