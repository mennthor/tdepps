# coding: utf-8

"""
Collection of repetedly used or large helper functions more or less indepent
of the use as building blocks for the model classes.
"""

from __future__ import print_function, division, absolute_import
from builtins import zip, map
from future import standard_library
standard_library.install_aliases()

import numpy as np
import scipy.optimize as sco
from scipy.stats import rv_continuous, chi2


def fill_dict_defaults(d, required_keys=None, opt_keys=None, noleft=True):
    """
    Populate dictionary with data from a given dict ``d``, and check if ``d``
    has required and optional keys. Set optionals with default if not present.

    If input ``d`` is None and ``required_keys`` is empty, just return
    ``opt_keys``.

    Parameters
    ----------
    d : dict or None
        Input dictionary containing the data to be checked. If is ``None``, then
        a copy of ``opt_keys`` is returned. If ``opt_keys`` is ``None``, a
        ``TypeError`` is raised. If ``d``is ``None`` and ``required_keys`` is
        not, then a ``ValueError`` israised.
    required_keys : list or None, optional
        Keys that must be present  and set in ``d``. (default: None)
    opt_keys : dict or None, optional
        Keys that are optional. ``opt_keys`` provides optional keys and default
        values ``d`` is filled with if not present in ``d``. (default: None)
    noleft : bool, optional
        If True, raises a ``KeyError``, when ``d`` contains etxra keys, other
        than those given in ``required_keys`` and ``opt_keys``. (default: True)

    Returns
    -------
    out : dict
        Contains all required and optional keys, using default values, where
        optional keys were missing. If ``d`` was None, a copy of ``opt_keys`` is
        returned, if ``opt_keys`` was not ``None``.
    """
    if required_keys is None:
        required_keys = []
    if opt_keys is None:
        opt_keys = {}
    if d is None:
        if not required_keys:
            if opt_keys is None:
                raise TypeError("`d` and òpt_keys` are both None.")
            return opt_keys.copy()
        else:
            raise ValueError("`d` is None, but `required_keys` is not empty.")

    d = d.copy()
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
        raise KeyError("Leftover keys ['{}'].".format(
            "', '".join(list(d.keys()))))
    return out


def random_choice(rndgen, CDF, n=None):
    """
    Stripped implementation of ``np.ranom.choice`` without the checks for the
    weights making it significantly faster. If ``CDF`` is not a real CDF this
    will produce non-sense.

    Parameters
    ----------
    rndgen : np.random.RandomState instance
        Random state instance to sample from.
    CDF : array-like
        Correctly normed CDF used to sample from. ``CDF[-1]=1`` and
        ``CDF[i-1]<=CDF[i]``.
    n : int or None, optional
        How many indices to sample. If ``None`` a single int is returned.
        (default: ``None``)

    Returns
    –------
    idx : int or array-like
        The sampled indices of the chosen CDF values. Can be inserted in the
        original array to obtain the values.
    """
    u = np.random.uniform(size=n)
    return np.searchsorted(CDF, u, side="right")


def weighted_cdf(x, val, weights=None):
    """
    Calculate the weighted CDF of data ``x`` with weights ``weights``.

    This calculates the fraction  of data points ``x <= val``, so we get a CDF
    curve when ``val`` is scanned for the same data points.
    The uncertainty is calculated from weighted binomial statistics using a
    Wald approximation.

    Inverse function of ``weighted_percentile``.

    Parameters
    ----------
    x : array-like
        Data values on which the percentile is calculated.
    val : float
        Threshold in x-space to calculate the percentile against.
    weights : array-like
        Weight for each data point. If ``None``, all weights are assumed to be
        1. (default: None)

    Returns
    -------
    cdf : float
        CDF in ``[0, 1]``, fraction of ``x <= val``.
    err : float
        Estimated uncertainty on the CDF value.
    """
    x = np.asarray(x)

    if weights is None:
        weights = np.ones_like(x)
    elif np.any(weights < 0.) or (np.sum(weights) <= 0):
        raise ValueError("Weights must be positive and add up to > 0.")

    # Get weighted CDF: Fraction of weighted values in x <= val
    mask = (x <= val)
    weights = weights / np.sum(weights)
    cdf = np.sum(weights[mask])
    # Binomial error on weighted cdf in Wald approximation
    err = np.sqrt(cdf * (1. - cdf) * np.sum(weights**2))

    return cdf, err


def ThetaPhiToDecRa(theta, phi):
    """
    Convert healpy theta, phi coordinates to equatorial ra, dec coordinates.
    ``phi`` and right-ascension are assumed to be the same value, declination is
    converted with ``dec = pi/2 - theta``.

    Parameters
    ----------
    theta, phi : array-like
        Healpy cooridnates in radian. ``theta ``is i n range ``[0, pi]``,
        ``phi`` in ``[0, 2pi]``.

    Returns
    -------
    dec, ra : array-like
        Equtorial coordinates declination in ``[-pi/2, pi/2]`` and
        right-ascension equal to ``phi`` in ``[0, 2pi]``.
    """
    theta, phi = map(np.atleast_1d, [theta, phi])
    return np.pi / 2. - theta, phi


def cos_angdist(ra1, dec1, ra2, dec2):
    """
    Great circle angular distance in equatorial coordinates.

    Parameters
    ----------
    ra1, dec1 : array-like
        Position(s) of the source point(s)
    ra2, dec2 : array-like
        Position(s) of the target point(s)

    Returns
    -------
    cos_dist : array-like
        Cosine of angular distances from point(s) (ra1, dec1) to (ra2, dec2).
        Shape depends onn input shapes. If both arrays are 1D they must have the
        same length and output is 1D too. General broadcasting rules up to 2D
        should apply.
    """
    cos_dist = (np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2) +
                np.sin(dec1) * np.sin(dec2))
    return np.clip(cos_dist, -1., 1.)


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

        cos_dist = np.clip((np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2) +
                           np.sin(dec1) * np.sin(dec2)), -1., 1.)

        ang = np.arccos(cos_dist).reshape(cos_dist.shape[0], 1)
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
