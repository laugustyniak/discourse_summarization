def batch_with_indexes(iterable, n=1):
    """
    Create sublist for iterable variable

    Parameters:

        iterable : any iterable structure

        n : int
            Number of sublist to create

    return:
        Tuple with indexes of each sublist based on starting list and list of sublists.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        if ndx + n > l:
            yield (ndx, l), iterable[ndx:min(ndx + n, l)]
        else:
            yield (ndx, ndx + n), iterable[ndx:min(ndx + n, l)]
