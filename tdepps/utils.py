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
            raise ValueError("'d' is None, but 'required_keys' is not empty.")

    out = {}
    # Set required keys
    for key in required_keys:
        if key in d:
            out[key] = d.pop(key)
        else:
            raise KeyError("'d' is missing key '{}'".format(key))
    # Set optional values, if key not given
    for key, val in opt_keys.items():
        out[key] = d.pop(key, val)
    # Should have no extra keys left
    if d:
        raise KeyError("{} not used in 'd'.".format(d.keys()))
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
    Generic rejection sampling method for ND pdfs with f: RN -> R.
    The ND function `pdf` is sampled in intervals xlow_i <= func(x_i) <= xhig_i
    where i=1, ..., N and n is the desired number of events.

    1. Find maximum of function. This is our upper boundary fmax of f(x)
    2. Loop until we have n valid events, start with m = n
        1. Create m uniformly distributed Nnsrcs points in the Nnsrcs bounds.
           Coordinates of these points are
           r1 = [[x11, ..., x1N ], ..., [xm1, ..., xmN]]
        2. Create m uniformly distributed numbers r2 between 0 and fmax
        3. Calculate the pdf value pdf(r1) and compare to r2
           If r2 <= pdf(r1) accept the event, else reject it
        4. Append only accepted random numbers to the final list

    This generates points that occur more often (or gets less rejection) near
    high values of the pdf and occur less often (get more rejection) where the
    pdf is small.

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
    sample : list of lists, len (nsrcs)
        Sampled events per source. If n_samples is 0 for a source, an empty
        list is included at that position.
    """
    n_samples = np.atleast_1d(n_samples)

    bounds = np.atleast_2d(bounds)
    nsrcs = bounds.shape[0]
    if bounds.shape[1] != 2:
        raise ValueError("`bounds` shape must be (nsrcs, 2).")

    rndgen = check_random_state(random_state)

    def negpdf(x):
        """Wrapper to use scipy.minimize minimization."""
        return -1. * pdf(x)

    # Maximum func values as upper bounds in all tranges to maximize efficiency
    xlow, xhig = bounds[:, 0], bounds[:, 1]
    x0 = 0.5 * (xlow + xhig)  # Start seed for minimizer
    xmin = np.zeros(nsrcs, dtype=np.float)
    for i in range(nsrcs):
        xmin[i] = sco.minimize(negpdf, x0[i], bounds=bounds[[i]]).x
    fmax = pdf(xmin)

    # Draw remaining events until all n samples are created
    sample = []
    while n > 0:
        # Sample positions and comparator, then accepts or not
        r1 = (xhig - xlow) * rndgen.uniform(
            0, 1, size=n * nsrcs).reshape(nsrcs, n) + xlow
        r2 = fmax * rndgen.uniform(0, 1, n)

        accepted = (r2 <= pdf(r1))
        sample += r1[accepted].tolist()

        n = np.sum(~accepted)

    return np.array(sample)
