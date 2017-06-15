# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import zip
from future import standard_library
standard_library.install_aliases()
import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state


def flatten_list_of_1darrays(l):
    """
    Flattens a list of 1d ndarrays with different lenghts to a single 1D array.

    Parameters
    ----------
    l : list of 1d arrays
        Arrays can have different lenghts but must be 1d.

    Returns
    -------
    arr : array-like
        1d output array. Length is the combined length of all arrays in list.

    Example
    -------
    >>> l = [np.array([1, 2]), np.array([]), np.array([3]), np.array([])]
    >>> arr = flatten_list_of_1darrays(l)
    array([1, 2, 3])
    """
    return np.array([el for arr in l for el in arr])


def fill_dict_defaults(d, required_keys=[], opt_keys={}, noleft=True):
    """
    Populate dictionary with data from a given dict `d`, and check if `d` has
    required and optional keys. Set optionals with default if not present.

    If input `d` is None and `required_keys` is empty, just return `opt_keys`.

    Parameters
    ----------
    d : dict
        Input dictionary containing the data to be checked.
    required_keys : list, optional
        Keys that must be present in `d`. (default: [])
    opt_keys : dict, optional
        Keys that are optional. `opt_keys` provides optional keys and the
        default values, if not present in `d`. (default: {})
    noleft : bool, optional
        If True, raises a KeyError, when `d` contains more keys, than given in
        `required_keys` and `opt_keys`. (default: True)

    Returns
    -------
    out : dict
        Contains all required and optional keys with default values, where
        optional keys where missing.
        If `d` is None, returns only the `opt_keys` dict.
    """
    if d is None:
        if not required_keys:
            return opt_keys
        else:
            raise ValueError("Dict is None, but 'required_keys' is not empty.")

    d = d.copy()  # Copy to not destroy the original dict
    out = {}
    # Set required keys
    for key in required_keys:
        if key in d:
            out[key] = d.pop(key)
        else:
            raise KeyError("Dict is missing required key '{}'.".format(key))
    # Set optional values, if key not given
    for key, val in opt_keys.items():
        out[key] = d.pop(key, val)
    # Complain when extra keys are left and noleft is True
    if d and noleft:
        raise KeyError("Keys ['{}'] not used in dict `d`.".format(
            "', '".join(list(d.keys()))))
    return out


def get_binmids(bins):
    """
    Given a list of bins, return the bin mids.

    Doesn't catch any falsely formatted data so be sure what you do.

    Parameter
    ---------
    bins : list
        Contains one array per binning.

    Returns
    -------
    mids : list
        List with the bin mids for each binning in bins.
        Mids have one point lesse than the input bins.
    """
    m = []
    for b in bins:
        m.append(0.5 * (b[:-1] + b[1:]))
    return m


def rejection_sampling(pdf, bounds, n_samples, max_fvals=None,
                       random_state=None):
    """
    Rejection sampler function to sample from multiple 1D regions at once.

    Algorithm description can be found at [1].
    To maximize efficiency the upper boundary for the random numbers is the
    maximum of the function over the defined region.

    Parameters
    ----------
    func : RateFunction.fun
        func must depend on one array-like argument x and return a list of
        function values with the same shape. Gets called via `func(x)`.
        func must also be interpretable as a pdf, so func >= 0 everywhere.
    bounds : array-like, shape (nsrcs, 2)
        Borders [[xlow, xhig], ...] in which func is sampled per source.
    n_samples : array-like, shape (nsrcs)
        Number of events to sample per source.
    fmax_vals : array-like, shape (nsrcs)
        If given, these values are used as the upper function bounds for each
        sampling interval. This can speed up calculation because we do not have
        to find the same maximum again for every call. Be sure these are right
        values, otherwise nonsense is sampled. (default: None)
    random_state : seed, optional
        Turn seed into a `np.random.RandomState` instance. See
        `sklearn.utils.check_random_state`. (default: None)

    Returns
    -------
    sample : list of arrays, len (nsrcs)
        Sampled events per source. If n_samples is 0 for a source, an empty
        array is included at that position.

    Notes
    -----
    This currently just loops over each given interval. The problem is, that
    each interval can have different amount of events that need to get sampled,
    so we can't use array broadcasting etc. If there's a better/faster method
    go ahead and implement it ;).

    .. [1] https://en.wikipedia.org/wiki/Rejection_sampling#Algorithm
    """
    n_samples = np.atleast_1d(n_samples)

    bounds = np.atleast_2d(bounds)
    if bounds.shape[1] != 2:
        raise ValueError("'bounds' shape must be (nsrcs, 2).")

    if max_fvals is not None:
        max_fvals = np.atleast_1d(max_fvals)
        if len(max_fvals) != bounds.shape[0]:
            raise ValueError("'max_fvals' must have same length as 'bounds'.")

    rndgen = check_random_state(random_state)

    def negpdf(x):
        """Wrapper to use scipy.minimize minimization."""
        return -1. * pdf(x)

    sample = []
    # Just loop over all intervals and append sample arrays to output list
    for i, (bound, nsam) in enumerate(zip(bounds, n_samples)):
        # Get maximum func value in bound to maximize sampling efficiency
        if max_fvals is None:
            fmax = -1. * func_min_in_interval(negpdf, bound)
        else:
            # Use cached values instead, if given
            fmax = max_fvals[i]

        # Draw remaining events until all samples per sourcee are created
        _sample = []
        xlow, xhig = bound[0], bound[1]
        while nsam > 0:
            # Sample x positions r1 and comparators r2, then accept or not
            r1 = (xhig - xlow) * rndgen.uniform(
                0, 1, size=nsam) + xlow
            r2 = fmax * rndgen.uniform(0, 1, nsam)

            accepted = (r2 <= pdf(r1))
            _sample += r1[accepted].tolist()

            nsam = np.sum(~accepted)  # Number of remaining samples to draw

        sample.append(np.array(_sample))

    return sample


def func_min_in_interval(func, interval, nscan=7):
    """
    Find the minimum of `func` the given intervall. A small scan in that range
    is performed prior to the fit to get a reasonable seed.

    Parameters
    ----------
    func : callcable
        Called as a function of 1 array-like parameter `func([x1, ..., xn])`.
    interval : array-like, shape (2)
        Interval boundaries `[min, max]` in which the minimization is performed.
    nscan : how many equidistant scan points prior to the fit to use. More
        points give a better see, but also take more time.

    Returns
    -------
    fmin : float
        Minimum function value in the given interval.
    """
    # Get start seed for minimizer by quick scan in the interval, avoid borders
    x_scan = np.linspace(interval[0], interval[1], nscan)[1:-1]
    min_idx = np.argmin(func(x_scan))
    x0 = x_scan[min_idx]
    # gtol, and ftol are explicitely low, when dealing with low rates.
    xmin = sco.minimize(func, x0, bounds=[interval], method="L-BFGS-B",
                        options={"gtol": 1e-12, "ftol": 1e-12}).x

    return func(xmin)
