import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import islice

def first_choice(pref_list):
    """Returns an allocation list where each agent is assigned to their first choice.

    Args:
        pref_list (np.array): the preference list, shape = (nr_agents, nr_preferences)

    Returns:
        np.array: the first choice of every agent -- facility index.
    """
    # shape = (nr_agents, 1)
    return pref_list[:, [0]]

def random_serial_dictatorship(pref_list, capacities):
    """Returns a matching where each agent is assigned a facility according to random serial dictatorship
    In random serial dictatorship (RSD), agents arrive randomly and are assigned top available preferred facility.
    Availability is determined by capacity of the facility at the time of the assignment.

    Args:
        pref_list (np.array): the preference list, shape = (nr_agents, nr_preferences)
        capacities (np.array): the capacities of the facilities

    Returns:
        np.array: facility indices of the assigned facility per agent
    """
    assignments = np.empty([len(pref_list), 1])

    lottery = list(range(len(pref_list)))
    random.shuffle(lottery)

    for agent in lottery:
        for facility in pref_list[agent]:
            if capacities[facility] > 0:
                assignments[agent] = np.array([facility])
                capacities[facility] -= 1
                break
    
    # this is done to reset the shuffle of the lottery. (python stuff)
    del lottery
    assert len(np.unique(assignments)) <= len(np.unique(pref_list)), 'Some agents were not assigned and this should not happen, or we should take care of it.'

    return assignments.astype(int)

def hungarian_match(pref_list, capacities):
    """Returns a matching where each agent is assigned a facility using the hungarian algorithm
    In hungarian algorithm, agents are assigned to facilities such that the average utility of each agent is maximized.

    Args:
        pref_list (np.array): the preference list, shape = (nr_agents, nr_preferences)
        capacities (np.array): the capacities of the facilities

    Returns:
        np.array: facility indices of the assigned facility per agent
    """
    total_capacity = int(sum(capacities))
    it = iter(list(range(total_capacity)))  # unique units of facilities
    facility_units = [list(islice(it, 0, int(i))) for i in capacities]
    cost = np.full(shape=(pref_list.shape[0], total_capacity), fill_value=99999, dtype=float)
    assignments = np.empty([len(pref_list), 1])
    for i, a in enumerate(pref_list):
        for r, f in enumerate(a):
            if len(facility_units[f]) > 0:
                for idx in facility_units[f]:
                    cost[i][idx] = r + 1
    row_ind, col_ind = linear_sum_assignment(cost)
    for i, agent in enumerate(row_ind):
        unit = col_ind[i]
        placed = facility_units.index([l for l in facility_units if unit in l][0])
        assignments[agent] = placed
    return assignments.astype(int)