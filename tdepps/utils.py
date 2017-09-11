# coding: utf-8

"""
Collection of repetedly used or large helper functions.
"""

from __future__ import print_function, division, absolute_import
from builtins import zip
from future import standard_library
standard_library.install_aliases()

import numpy as np
import scipy.optimize as sco
from scipy.stats import rv_continuous, chi2


def power_law_flux_per_type(trueE, gamma):
    """
    Returns the unbroken power law flux :math:`\sim E^{-\gamma} per particle
    type (nu, anti-nu), without a normalization.

    Parameters
    ----------
    trueE : array-like
        True particle energy in GeV.
    gamma : float
        Positive power law index.

    Returns
    -------
    flux : array-like
        Per particle flux :math:`\phi = 0.5 * E^{-\gamma}`
    """
    return 0.5 * trueE**(-gamma)


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


def rejection_sampling(pdf, bounds, n_samples, rndgen, max_fvals=None):
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
    rndgen : `np.random.RandomState` instance
        Random number generator instance to have control over the sample chain.
    fmax_vals : array-like, shape (nsrcs)
        If given, these values are used as the upper function bounds for each
        sampling interval. This can speed up calculation because we do not have
        to find the same maximum again for every call. Be sure these are right
        values, otherwise nonsense is sampled. (default: None)

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

        # Draw remaining events until all samples per source are created
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


def rotator(ra1, dec1, ra2, dec2, ra3, dec3):
    """
    Rotate vectors pointing to directions given by pairs `(ra3, dec3)` by the
    rotation defined by going from `(ra1, dec1)` to `(ra2, dec2)`.

    `ra`, `dec` are per event right-ascension in :math:`[0, 2\pi]` and
    declination in :math:`[-\pi/2, \pi/2]`, both in radians.

    Parameters
    ----------
    ra1, dec1 : array-like, shape (nevts)
        The points we start the rotation at.
    ra2, dec2 : array-like, shape (nevts)
        The points we end the rotation at.
    ra3, dec3 : array-like, shape (nevts)
        The points we actually rotate around the axis defined by the directions
        above.

    Returns
    -------
    ra3t, dec3t : array-like, shape (nevts)
        The rotated directions `(ra3, dec3) -> (ra3t, dec3t)`.

    Notes
    -----
    Using quaternion rotation from [1]_. Was a good way to recap this stuff.
    If you are keen, you can show that this is the same as the rotation
    matrix formalism used in skylabs rotator.

    Alternative ways to express the quaternion conjugate product:

    .. code-block::
       A) ((q0**2 - np.sum(qv * qv, axis=1).reshape(qv.shape[0], 1)) * rv +
            2 * q0 * np.cross(qv, rv) +
            2 * np.sum(qv * rv, axis=1).reshape(len(qv), 1) * qv)

       B) rv + 2 * q0 * np.cross(qv, rv) + 2 * np.cross(qv, np.cross(qv, rv))


    .. [1] http://people.csail.mit.edu/bkph/articles/Quaternions.pdf
    """
    def ra_dec_to_quat(ra, dec):
        """
        Convert equatorial coordinates to quaternion representation.

        Parameters
        ----------
        ra, dec : array-like, shape (nevts)
            Per event right-ascension in :math:`[0, 2\pi]` and declination in
            :math:`[-\pi/2, \pi/2]`, both in radians.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            One quaternion per row from each (ra, dec) pair.
        """
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.vstack((np.zeros_like(x), x, y, z)).T

    def quat_to_ra_dec(q):
        """
        Convert quaternions back to quatorial coordinates.

        Parameters
        ----------
        q : array-like, shape (nevts, 4)
            One quaternion per row to convert to a (ra, dec) pair each.

        Returns
        -------
        ra, dec : array-like, shape (nevts)
            Per event right-ascension in :math:`[0, 2\pi]` and declination in
            :math:`[-\pi/2, \pi/2]`, both in radians.
        """
        nv = norm(q[:, 1:])
        x, y, z = nv[:, 0], nv[:, 1], nv[:, 2]
        dec = np.arcsin(z)
        ra = np.arctan2(y, x)
        ra[ra < 0] += 2. * np.pi
        return ra, dec

    def norm(v):
        """
        Normalize a vector, so that the sum over the squared elements is one.

        Also valid for quaternions.

        Parameters
        ----------
        v : array-like, shape (nevts, ndim)
            One vector per row to normalize

        Returns
        -------
        nv : array-like, shape (nevts, ndim)
            Normed vectors per row.
        """
        norm = np.sqrt(np.sum(v**2, axis=1))
        m = (norm == 0.)
        norm[m] = 1.
        vn = v / norm.reshape(v.shape[0], 1)
        assert np.allclose(np.sum(vn[~m]**2, axis=1), 1.)
        return vn

    def quat_mult(p, q):
        """
        Multiply p * q in exactly this order.

        Parameters
        ----------
        p, q : array-like, shape (nevts, 4)
            Quaternions in each row to multiply.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            Result of the quaternion multiplication. One quaternion per row.
        """
        p0, p1, p2, p3 = p[:, [0]], p[:, [1]], p[:, [2]], p[:, [3]]
        q0, q1, q2, q3 = q[:, [0]], q[:, [1]], q[:, [2]], q[:, [3]]
        # This algebra reflects the similarity to the rotation matrices
        a = q0 * p0 - q1 * p1 - q2 * p2 - q3 * p3
        x = q0 * p1 + q1 * p0 - q2 * p3 + q3 * p2
        y = q0 * p2 + q1 * p3 + q2 * p0 - q3 * p1
        z = q0 * p3 - q1 * p2 + q2 * p1 + q3 * p0

        return np.hstack((a, x, y, z))

    def quat_conj(q):
        """
        Get the conjugate quaternion. This means switched signs of the
        imagenary parts `(i,j,k) -> (-i,-j,-k)`.

        Parameters
        ----------
        q : array-like, shape (nevts, 4)
            One quaternion per row to conjugate.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            Conjugated quaternions. One quaternion per row.
        """
        return np.hstack((q[:, [0]], -q[:, [1]], -q[:, [2]], -q[:, [3]]))

    def get_rot_quat_from_ra_dec(ra1, dec1, ra2, dec2):
        """
        Construct quaternion which defines the rotation from a vector
        pointing to `(ra1, dec1)` to another one pointing to `(ra2, dec2)`.

        The rotation quaternion has the rotation angle in it's first
        component and the axis around which is rotated in the last three
        components. The quaternion must be normed :math:`\sum(q_i^2)=1`.

        Parameters
        ----------
        ra1, dec1 : array-like, shape (nevts)
            The points we start the rotation at.
        ra2, dec2 : array-like, shape (nevts)
            The points we end the rotation at.

        Returns
        -------
        out : array-like, shape (nevts, 4)
            One quaternion per row defining the rotation axis and angle for
            each given pair of `(ra1, dec1)`, `(ra2, dec2)`.
        """
        p0 = ra_dec_to_quat(ra1, dec1)
        p1 = ra_dec_to_quat(ra2, dec2)
        # Norm rotation axis for proper quaternion normalization
        ax = norm(np.cross(p0[:, 1:], p1[:, 1:]))

        cos_ang = np.clip((np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2) +
                           np.sin(dec1) * np.sin(dec2)), -1., 1.)

        ang = np.arccos(cos_ang).reshape(cos_ang.shape[0], 1)
        ang /= 2.
        a = np.cos(ang)
        ax = ax * np.sin(ang)
        # Normed because: sin^2 + cos^2 * vec(ax)^2 = sin^2 + cos^2 = 1
        return np.hstack((a, ax[:, [0]], ax[:, [1]], ax[:, [2]]))

    ra1, dec1, ra2, dec2, ra3, dec3 = map(np.atleast_1d,
                                          [ra1, dec1, ra2, dec2, ra3, dec3])
    assert(len(ra1) == len(dec1) == len(ra2) == len(dec2) ==
           len(ra3) == len(dec3))

    # Convert (ra3, dec3) to imaginary quaternion -> (0, vec(ra, dec))
    q3 = ra_dec_to_quat(ra3, dec3)

    # Make rotation quaternion: (angle, vec(rot_axis)
    q_rot = get_rot_quat_from_ra_dec(ra1, dec1, ra2, dec2)

    # Rotate by multiplying q3' = q_rot * q3 * q_rot_conj
    q3t = quat_mult(q_rot, quat_mult(q3, quat_conj(q_rot)))
    # Rotations preserves vectors, so imaganery part stays zero
    assert np.allclose(q3t[:, 0], 0.)

    # And transform back to (ra3', dec3')
    return quat_to_ra_dec(q3t)


class delta_chi2_gen(rv_continuous):
    """
    Class for a probability denstiy function modelled by using a:math`\chi^2`
    distribution for :math:`x > 0` and a constant fraction :math:`1 - \eta`
    of zero trials for :math`x = 0` (like a delta peak at 0).

    Notes
    -----
    The probability density function for `delta_chi2` is:

    .. math::

      \text{PDF}(x|\text{df}, \eta) =
          \begin{cases}
              (1-\eta)                &\text{for } x=0 \\
              \eta\chi^2_\text{df}(x) &\text{for } x>0 \\
          \end{cases}

    `delta_chi2` takes ``df`` and ``eta``as a shape parameter, where ``df`` is
    the standard :math:`\chi^2_\text{df}` degrees of freedom parameter and
    ``1-eta`` is the fraction of the contribution of the delta function at zero.
    """
    def _rvs(self, df, eta):
        # Determine fraction of zeros by drawing from binomial with p=eta
        s = self._size if not len(self._size) == 0 else 1
        nzeros = self._random_state.binomial(n=s, p=(1. - eta), size=None)
        # If None, len of size is 0 and single scalar rvs requested
        if len(self._size) == 0:
            if nzeros == 1:
                return 0.
            else:
                return self._random_state.chisquare(df, size=None)
        # If no zeros or no chi2 is drawn for this trial, only draw one type
        if nzeros == self._size:
            return np.zeros(nzeros, dtype=np.float)
        if nzeros == 0:
            return self._random_state.chisquare(df, size=self._size)
        # All other cases: Draw, concatenate and shuffle to simulate a per
        # random number Bernoulli process with p=eta
        out = np.r_[np.zeros(nzeros, dtype=np.float),
                    self._random_state.chisquare(df,
                                                 size=(self._size - nzeros))]
        self._random_state.shuffle(out)
        return out

    def _pdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., eta * chi2.pdf(x, df=df), 1. - eta)

    def _logpdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., np.log(eta) + chi2.logpdf(x, df=df),
                        np.log(1. - eta))

    def _cdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., (1. - eta) + eta * chi2.cdf(x, df=df),
                        (1. - eta))

    def _logcdf(self, x, df, eta):
        x = np.asarray(x)
        return np.where(x > 0., np.log(1 - eta + eta * chi2.cdf(x, df)),
                        np.log(1. - eta))

    def _sf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.asarray(x)
        return np.where(x > 0., eta * chi2.sf(x, df), 1.)

    def _logsf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.asarray(x)
        return np.where(x > 0., np.log(eta) + chi2.logsf(x, df), 0.)

    def _ppf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.cdf as defined above
        p = np.asarray(p)
        return np.where(p > (1. - eta), chi2.ppf(1 + (p - 1) / eta, df), 0.)

    def _isf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.sf as defined above
        return np.where(p < eta, chi2.isf(p / eta, df), 0.)

    def fit(self, data, *args, **kwds):
        # Wrapper for chi2 fit, estimating eta and fitting chi2 on data > 0
        data = np.asarray(data)
        eta = float(np.count_nonzero(data)) / len(data)
        df, loc, scale = chi2.fit(data[data > 0.], *args, **kwds)
        return df, eta, loc, scale

    def fit_nzeros(self, data, nzeros, *args, **kwds):
        # Same as `fit` but data has only non-zero trials
        data = np.asarray(data)
        ndata = len(data)
        eta = float(ndata) / (nzeros + ndata)
        df, loc, scale = chi2.fit(data, *args, **kwds)
        return df, eta, loc, scale


delta_chi2 = delta_chi2_gen(name="delta_chi2")
