def first_choice(pref_list):
    """Returns a pref list where each agent is assigned to their first choice.

    Args:
        pref_list (np.array): the preference list, shape = (nr_agents, nr_preferences)

    Returns:
        np.array: the first choice of every agent -- facility index.
    """
    return pref_list[:, [0]]
