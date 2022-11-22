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
    assert allocation.shape[1] == 1, 'Only one facility should be allocated to each agent.'
    
    eval = []
    for fid in facilities['id'].values:
        allocs = np.sum(allocation == fid)
        eval.append(allocs / facilities.loc[facilities['id'] == fid]['capacity'].values[0])

    return eval
