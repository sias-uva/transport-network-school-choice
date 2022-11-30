import numpy as np

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

