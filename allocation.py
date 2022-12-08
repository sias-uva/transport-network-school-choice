import random
import numpy as np

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

    assert len(np.unique(assignments)) <= len(np.unique(pref_list)), 'Some agents were not assigned and this should not happen, or we should take care of it.'

    return assignments.astype(int)
