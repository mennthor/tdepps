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


def fill_dict_defaults(d, required_keys=[], opt_keys={}):
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
            raise KeyError("Dict is missing key '{}'.".format(key))
    # Set optional values, if key not given
    for key, val in opt_keys.items():
        out[key] = d.pop(key, val)
    # Should have no extra keys left
    if d:
        for key in d.keys():
            raise KeyError("Key '{}' not used in dict.".format(key))
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


def rejection_sampling(pdf, bounds, n_samples, random_state=None):
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
        raise ValueError("`bounds` shape must be (nsrcs, 2).")

    rndgen = check_random_state(random_state)

    def negpdf(x):
        """Wrapper to use scipy.minimize minimization."""
        return -1. * pdf(x)

    sample = []

    # Just loop over all intervals and append sample arrays to output list
    for bound, nsam in zip(bounds, n_samples):
        # Get maximum func value in bound to maximize efficiency
        xlow, xhig = bound[0], bound[1]
        # Start seed for minimizer by quick scan in the interval
        x_scan = np.linspace(bound[0], bound[1], 7)[1:-1]
        max_idx = np.argmax(pdf(x_scan))
        x0 = x_scan[max_idx]
        # x0 = 0.5 * (xlow + xhig)
        # gtol, and ftol are explicitely low, when dealing with low rates.
        xmin = sco.minimize(negpdf, x0, bounds=[bound], method="L-BFGS-B",
                            options={"gtol": 1e-12, "ftol": 1e-12}).x

        fmax = pdf(xmin)

        # Draw remaining events until all samples per sourcee are created
        _sample = []
        while nsam > 0:
            # Sample x positions r1 and comparators r2, then accept or not
            r1 = (xhig - xlow) * rndgen.uniform(
                0, 1, size=nsam) + xlow
            r2 = fmax * rndgen.uniform(0, 1, nsam)

            accepted = (r2 <= pdf(r1))
            _sample += r1[accepted].tolist()  # Concatenate

            nsam = np.sum(~accepted)  # Number of remaining samples to draw

        sample.append(np.array(_sample))

    return sample
