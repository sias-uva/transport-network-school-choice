import numpy as np

def facility_capacity(population, facilities, allocation):
    assert allocation.shape[1] == 1, 'Only one facility should be allocated to each agent.'
    
    eval = []
    for fid in facilities['id'].values:
        allocs = np.sum(allocation == fid)
        eval.append(allocs / facilities.loc[facilities['id'] == fid]['capacity'].values[0])

    return eval