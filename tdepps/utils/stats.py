# coding: utf-8

"""
Collection of statistics related helper methods.
"""

from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from scipy.stats import rv_continuous, chi2, expon
import scipy.optimize as sco
from .io import logger
log = logger(name="utils.stats", level="ALL")


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
    u = rndgen.uniform(size=n)
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
    x = np.atleast_1d(x)

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


def cdf_nzeros(x, nzeros, vals, sorted=False):
    """
    Returns the CDF value at value ``vals`` for a dataset with ``x > 0`` and
    ``nzeros`` entries that are zero.

    Parameters
    ----------
    x : array-like
        Data values on which the percentile is calculated.
    nzeros : int
        Number of zero trials.
    vals : float or array-like
        Threshold(s) in x-space to calculate the percentile against.
    sorted : bool, optional
        If ``True`` assume ``x`` is sorted ``x[0] <= ... <= x[-1]``.
        Can save time on large arrays but produces wrong results, if array is
        not really sorted.

    Returns
    -------
    cdf : float
        CDF in ``[0, 1]``, fraction of ``x <= vals``.
    """
    x = np.atleast_1d(x)
    vals = np.atleast_1d(vals)
    if not sorted:
        x = np.sort(x)

    ntot = float(len(x) + nzeros)

    # CDF(x<=val) =  Total fraction of values x <= val + given zero fraction
    frac_zeros = nzeros / ntot
    cdf = np.searchsorted(x, vals, side="right") / ntot + frac_zeros
    return cdf


def percentile_nzeros(x, nzeros, q, sorted=False):
    """
    Returns the percentile ``q`` for a dataset with ``x > 0`` and ``nzeros``
    entries that are zero.

    Alternatively do ``np.percentile(np.r_[np.zeros(nzeros), x], q)``, which
    gives the same result when choosing ``interpolation='lower'``.

    Parameters
    ----------
    x : array-like
        Non-zero values.
    nzeros : int
        Number of zero trials.
    q : float
        Percentile in ``[0, 100]``.
    sorted : bool, optional
        If ``True`` assume ``x`` is sorted ``x[0] <= ... <= x[-1]``.
        Can save time on large arrays but produces wrong results, if array is
        not really sorted.

    Returns
    -------
    percentile : float
        The percentile at level ``q``.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q) / 100.
    if not sorted:
        x = np.sort(x)

    ntot = len(x) + nzeros
    idx = (q * ntot).astype(int) - nzeros - 1

    percentile = np.zeros_like(q, dtype=np.float)
    m = (idx >= 0)
    percentile[m] = x[idx[m]]

    return percentile


def fit_chi2_cdf(ts_val, beta, TS, mus):
    """
    Use collection of trials with different injected mean signal events to
    calculate the CDF values above a certain test statistic value ``ts_val``
    and fit a ``chi2`` CDF to it.
    From this ``chi2``function we can get the desired percentile ``beta`` above
    ``ts_val``.

    Parameters
    ----------
    ts_val : float
        Test statistic value of the BG distribution, which is connected to the
        alpha value (Type I error).
    beta : float
        Fraction of signal injected PDF that should lie right of the ``ts_val``.
    mus : array-like
        Mean injected signal events per bunch of trials.
    TS : array-like, shape (len(mus), ntrials_per_mu)
        Test statistic values for each ``mu`` in ``mus``. These are used to
        calculate the CDF values used in the fit.

    Returns
    -------
    mu_bf : float
        Best fit mean injected number of signal events to fullfill the
        tested performance level from ``ts_val`` and ``beta``.
    cdfs : array-like, shape (len(mus))

    pars : tuple
        Best fit parameters ``(df, loc, scale)`` for the ``chi2`` CDF.
    """
    # Get location of beta percentile per bunch of trial TS values
    cdfs = []
    errs = []
    for TSi in TS:
        cdf_i, err_i = weighted_cdf(TSi, val=ts_val, weights=None)
        cdfs.append(cdf_i)
        errs.append(err_i)
    cdfs, errs = np.array(cdfs), np.array(errs)

    def _cdf_func(x, df, loc, scale):
        """ Somehow can't use scs.chi2.cdf directly for curve fit... """
        return chi2.cdf(x, df, loc, scale)

    try:
        pars, _ = sco.curve_fit(
            _cdf_func, xdata=mus, ydata=1. - cdfs, sigma=errs, p0=[1., 1., 1.])
        mu_bf = chi2.ppf(beta, *pars)
    except RuntimeError:
        print(log.WARN("Couldn't find best params, returning `None` instead."))
        mu_bf = None
        pars = None

    return mu_bf, cdfs, pars


class delta_chi2_gen(rv_continuous):
    """
    Class for a probability denstiy distribution modelled by using a ``chi2``
    distribution for ``x > 0`` and a constant fraction ``1 - \eta`` of zero
    trials for ``x = 0`` - like a delta peak at 0.

    Notes
    -----
    The probability density distribution for ``delta_chi2`` is:

    .. math::

      \text{PDF}(x|\text{df}, \eta) =
          \begin{cases}
              (1-\eta)                &\text{for } x=0 \\
              \eta\chi^2_\text{df}(x) &\text{for } x>0 \\
          \end{cases}

    ``delta_chi2`` takes ``df`` and ``eta``as a shape parameter, where ``df``
    is the standard ``chi2_df`` degrees of freedom parameter and ``1-eta`` is
    the fraction of the contribution of the delta distribution at zero.
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
        x = np.atleast_1d(x)
        return np.where(x > 0., eta * chi2.pdf(x, df=df), 1. - eta)

    def _logpdf(self, x, df, eta):
        x = np.atleast_1d(x)
        return np.where(x > 0., np.log(eta) + chi2.logpdf(x, df=df),
                        np.log(1. - eta))

    def _cdf(self, x, df, eta):
        x = np.atleast_1d(x)
        return np.where(x > 0., (1. - eta) + eta * chi2.cdf(x, df=df),
                        (1. - eta))

    def _logcdf(self, x, df, eta):
        x = np.atleast_1d(x)
        return np.where(x > 0., np.log(1 - eta + eta * chi2.cdf(x, df)),
                        np.log(1. - eta))

    def _sf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.atleast_1d(x)
        return np.where(x > 0., eta * chi2.sf(x, df), 1.)

    def _logsf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.atleast_1d(x)
        return np.where(x > 0., np.log(eta) + chi2.logsf(x, df), 0.)

    def _ppf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.cdf as defined above
        p = np.atleast_1d(p)
        return np.where(p > (1. - eta), chi2.ppf(1 + (p - 1) / eta, df), 0.)

    def _isf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.sf as defined above
        return np.where(p < eta, chi2.isf(p / eta, df), 0.)

    def fit(self, data, *args, **kwds):
        # Wrapper for chi2 fit, estimating eta and fitting chi2 on data > 0
        data = np.atleast_1d(data)
        eta = float(np.count_nonzero(data)) / len(data)
        df, loc, scale = chi2.fit(data[data > 0.], *args, **kwds)
        return df, eta, loc, scale

    def fit_nzeros(self, data, nzeros, *args, **kwds):
        # Same as `fit` but data has only non-zero trials
        data = np.atleast_1d(data)
        ndata = len(data)
        eta = float(ndata) / (nzeros + ndata)
        df, loc, scale = chi2.fit(data, *args, **kwds)
        return df, eta, loc, scale


delta_chi2 = delta_chi2_gen(name="delta_chi2")


class emp_with_exp_tail_gen(rv_continuous):
    """
    Class for a probability denstiy distribution modelled by using the empirical
    PDF defined by trial data and a fitted exponetial tail, replacing the
    empirical PDF in a low statistics region.

    Notes
    -----
    The probability density distribution for `delta_chi2` is:

    .. math::

      \text{PDF}(x, t, \lambda) =
          \begin{cases}
              \text{PDF}_\text{emp}(x)    &\text{for } x<=t \\
              \text{CDF}_\text{emp}(t) \cdot
                \lambda e^{-\lambda (x-t)}  &\text{for } x>t \\
          \end{cases}

    Here ``t`` is the threshold after which we switch from th empirical PDF to
    the exponential tail, ``lambda`` is the exponential decay parameter. The
    normalization of the exponential term is adapted to match the total
    normaliazation together with the empirical part.
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
        x = np.atleast_1d(x)
        return np.where(x > 0., eta * chi2.pdf(x, df=df), 1. - eta)

    def _logpdf(self, x, df, eta):
        x = np.atleast_1d(x)
        return np.where(x > 0., np.log(eta) + chi2.logpdf(x, df=df),
                        np.log(1. - eta))

    def _cdf(self, x, df, eta):
        x = np.atleast_1d(x)
        return np.where(x > 0., (1. - eta) + eta * chi2.cdf(x, df=df),
                        (1. - eta))

    def _logcdf(self, x, df, eta):
        x = np.atleast_1d(x)
        return np.where(x > 0., np.log(1 - eta + eta * chi2.cdf(x, df)),
                        np.log(1. - eta))

    def _sf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.atleast_1d(x)
        return np.where(x > 0., eta * chi2.sf(x, df), 1.)

    def _logsf(self, x, df, eta):
        # Note: Define sf(0)=0 and not sf(0)=(1-eta) because 0 is inclusive
        x = np.atleast_1d(x)
        return np.where(x > 0., np.log(eta) + chi2.logsf(x, df), 0.)

    def _ppf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.cdf as defined above
        p = np.atleast_1d(p)
        return np.where(p > (1. - eta), chi2.ppf(1 + (p - 1) / eta, df), 0.)

    def _isf(self, p, df, eta):
        # Derive this from inverting p = delta_chi2.sf as defined above
        return np.where(p < eta, chi2.isf(p / eta, df), 0.)

    def fit(self, data, t, *args, **kwds):
        # Fits the exponential tail after the threshold
        data = np.atleast_1d(data)
        loc, scale = chi2.fit(data[data > 0.], *args, **kwds)
        return df, eta, loc, scale

    def fit_nzeros(self, data, nzeros, t, *args, **kwds):
        # Same as `fit` but given data has only non-zero trials
        data = np.atleast_1d(data)
        ndata = len(data)
        self._nzeros = nzeros
        df, loc, scale = chi2.fit(data, *args, **kwds)
        return df, eta, loc, scale
