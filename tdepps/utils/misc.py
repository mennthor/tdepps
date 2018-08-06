# coding: utf-8

"""
Collection of other helper methods.
"""

import numpy as np


def interval_overlap(a0, a1, b0, b1):
    """
    Returns the overlap between closed intervals ``[a0, a1], [b0, b1]``.

    Parameters
    ----------
    a0, a1 : float or array-like
        First interval(s). Multiple borders can be given as arrays, then both
        ``a0, a1`` must have the same length.
    b0, b1 : float or array-like
        Second interval(s). Multiple borders can be given as arrays, then both
        ``b0, b1`` must have the same length.
        If both ``a0, a1`` and ``b0, b1`` are given as arrays, all arrays must
        have the same length.

    Returns
    -------
    overlap : float or array-like
        Overlap between the intervals, truncated at zero for negativ overlaps.
        If all borders were given as floats, ``overlap`` is also a float.
        Otherwise ``overlap`` is an array of the same length as the given
        array(s).

    Example
    -------
    >>> a0, a1 = 1, 2
    >>> b0 = [0.5  , 2.2, 1.2, 0.8, 0.8, 1.8]
    >>> b1 = [0.8, 2.5, 1.8, 2.2, 1.2, 2.2]
    >>> interval_overlap(a0, a1, b0, b1)
    [0, 0, 0.6, 1, 0.2, 0.2]
    """
    return np.maximum(np.minimum(a1, b1) - np.maximum(a0, b0), 0.)
