def nearest_k(tt_mx, k):
    """Returns the nearest k facilities, based on the tt_mx (travel-time matrix.)

    Args:
        tt_mx (np.array): 2-D array where (i, j) is the travel time for agent i to facility j
        k (int): nearest k facilities to return

    Returns:
        np.array: indices of nearest k facilities (indices=ids)
    """

    assert k <= tt_mx.shape[1], f'Cannot pick nearest {k} out of {tt_mx.shape[1]} facilities' 
    return tt_mx.argsort()[:, :k]
