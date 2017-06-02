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
    Doesn't catch any false formatted data so be sure what you do.

    Parameter
    ---------
    bins : list
        Contains one array per binning.

    Returns
    -------
    mids : list
        List with the bin mids for each binning in bins.
        Mids have on point lesse than the input bins.
    """
    m = []
    for b in bins:
        m.append(0.5 * (b[:-1] + b[1:]))
    return m


def rejection_sampling(pdf, bounds, n, random_state=None):
    """
    Generic rejection sampling method for ND pdfs with f: RN -> R.
    The ND function `pdf` is sampled in intervals xlow_i <= func(x_i) <= xhig_i
    where i=1, ..., N and n is the desired number of events.

    1. Find maximum of function. This is our upper boundary fmax of f(x)
    2. Loop until we have n valid events, start with m = n
        1. Create m uniformly distributed Ndim points in the Ndim bounds.
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
    func : function
        Function from which to sample. func is taking exactly one argument 'x'
        which is a n-dimensional array containing a N dimensional point in each
        entry (as in scipy.stats.multivariat_normal):
            x = [ [x11, ..., x1N], [x21, ..., x2N], ..., [xn1, ..., xnN]
        func must be a pdf, so func >= 0 and area under curve=1.
    xlow, xhig : array-like, shape (ndims, 2)
        Arrays with the rectangular borders of the pdf. The length of xlow and
        xhig must be equal and determine the dimension of the pdf.
    n : int
        Number of events to be sampled from the given pdf.
    random_state : seed, optional
        Turn seed into a np.random.RandomState instance. See
        `sklearn.utils.check_random_state`. (default: None)

    Returns
    -------
    sample : array-like
        A list of the n sampled points
    """
    if n == 0:
        return np.array([], dtype=np.float), 1.
    bounds = np.atleast_2d(bounds)
    dim = bounds.shape[0]

    rndgen = check_random_state(random_state)

    def negpdf(x):
        """To find the maximum we need to invert to use scipy.minimize."""
        return -1. * pdf(x)

    def _pdf(x):
        """PDF must be positive everywhere, so raise error if not."""
        res = pdf(x)
        if np.any(res < 0.):
            raise ValueError("Evaluation of PDF resultet in negative value.")
        return res

    # Get maximum to maximize efficiency
    xlow, xhig = bounds[:, 0], bounds[:, 1]
    x0 = 0.5 * (xlow + xhig)
    optres = sco.minimize(negpdf, x0, bounds=bounds)
    fmax = _pdf(optres.x)

    # Draw remaining events until all n samples are created
    sample = []
    while n > 0:
        # Sample positions and comparator, then accepts or not
        r1 = (xhig - xlow) * rndgen.uniform(
            0, 1, size=n * dim).reshape(dim, n) + xlow
        r2 = fmax * rndgen.uniform(0, 1, n)

        accepted = (r2 <= _pdf(r1))
        sample += r1[accepted].tolist()

        n = np.sum(~accepted)

    return np.array(sample)
