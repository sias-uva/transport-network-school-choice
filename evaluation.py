import numpy as np

def facility_capacity(population, facilities, allocation):
    """Returns an array of percentage of satisfied capacity per facility.

    Args:
        population (pd.DataFrame): population dataframe - should have id column
        facilities (pd.DataFrame): facility dataframe - should have id column
        allocation (np.array): array of shape (total_pop, 1) where each agent is allocated to 1 facility.

    Returns:
        array: array of satisfied capacity per facility.
    """
    # TODO - probably best to transfer this assert to the allocation method.
    assert allocation.shape[1] == 1, 'Only one facility should be allocated to each agent.'
    
    eval = []
    for fid in facilities['id'].values:
        allocs = np.sum(allocation == fid)
        eval.append(allocs / facilities.loc[facilities['id'] == fid]['capacity'].values[0])

    return eval

def facility_diversity(population, facilities, allocation):
    """ Returns a 2-D array of the group distribution in each facility - shape = (nr_facilities, nr_groups) where (i,j) = percent of group j in facility i.

    Args:
        population (pd.DataFrame): population dataframe - should have id column
        facilities (pd.DataFrame): facility dataframe - should have id column
        allocation (np.array): array of shape (total_pop, 1) where each agent is allocated to 1 facility.

    Returns:
        np.array: array of percent of groups at each facility.
    """
    # TODO - probably best to transfer this assert to the allocation method.
    assert allocation.shape[1] == 1, 'Only one facility should be allocated to each agent.'

    # I don't like the for-based solution but too lazy atm.
    eval = np.zeros((facilities.shape[0], population['group'].nunique()))
    for i_g, g in enumerate(population['group'].unique()):
        pop_id = population[population['group'] == g]['id'].values

        for fid in facilities['id'].values:
            allocs = np.sum(allocation[pop_id] == fid)
            eval[i_g, fid] = allocs

    return np.true_divide(eval, eval.sum(axis=1, keepdims=True))

