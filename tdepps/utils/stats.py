# coding: utf-8

"""
Collection of statistics related helper methods.
"""

from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()

import abc
import json
import numpy as np
from scipy.stats import rv_continuous, norm, chi2, expon, kstest
import scipy.optimize as sco
from .io import logger
log = logger(name="utils.stats", level="ALL")


def random_choice(rndgen, CDF, size=None):
    """
    Stripped implementation of ``np.random.choice`` without the checks for the
    weights making it significantly faster. If ``CDF`` is not a real CDF this
    will produce non-sense.

    Note: In general the CDF is built by cumulative summing up event weights.
          For unweighted events, the weights are all equal.
    Note: Only samples with replacement.

    Parameters
    ----------
    rndgen : np.random.RandomState instance
        Random state instance to sample from.
    CDF : array-like
        Correctly normed CDF used to sample from. ``CDF[-1]=1`` and
        ``CDF[i-1]<=CDF[i]``.
    size : int or None, optional
        How many indices to sample. If ``None`` a single int is returned.
        (default: ``None``)

    Returns
    –------
    idx : int or array-like
        The sampled indices of the chosen CDF values. Can be inserted in the
        original array to obtain the values.
    """
    u = rndgen.uniform(size=size)
    return np.searchsorted(CDF, u, side="right")


def sigma2prob(sig):
    """
    Return the probability for a given gaussian sigma central interval.
    Reverse operation of ``prob2sigma``.

    Parameters
    ----------
    sig : array-like
        Sigma values to convert to central probabities.

    Returns
    -------
    p : array-like
        Central interval probabilities.
    """
    sig = np.atleast_1d(sig)
    return norm.cdf(sig) - norm.cdf(-sig)  # Central interval


def prob2sigma(p):
    """
    Return the corresponding sigma for an assumed total probability of
    ``1-p`` in both tails of a gaussian PDF - so ``p/2`` in each of the tails.
    Reverse operation of ``sigma2prob``

    Parameters
    ----------
    p : array-like
        Probabilities to convert to sigmas.

    Returns
    -------
    sig : array-like
        Sigma values corresponding to ``p``.
    """
    p = np.atleast_1d(p)
    return norm.ppf(p + (1. - p) / 2.)  # (1-p) / 2 in the right tail


def weighted_cdf(x, val, weights=None):
    """
    Calculate the weighted CDF of data ``x`` with weights ``weights``, this
    calculates the summed weights of all data points ``x <= val``.

    The uncertainty is calculated from weighted binomial statistics using a
    Wald approximation:

        var = p * (1 - p) / n

    or in the weighted case:

        var = p * (1 - p) * sum w^22

    which is with w = 1 / n

        var = p * (1 - p) * sum (1 / n^2)
            = p * (1 - p) * (n / n^2)
            = p * (1 - p) / n

    Inverse function of ``weighted_percentile``.

    Parameters
    ----------
    x : array-like
        Data values on which the percentile is calculated.
    val : float
        Threshold in x-space to calculate the percentile against.
    weights : array-like
        Weight for each data point. If ``None``, all weights are assumed to be
        equal. Weights are normalized to ``sum(weights) = 1``. (default: None)

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
    err = np.sqrt(cdf * np.clip((1. - cdf), 0., 1.) * np.sum(weights**2))

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

    # CDF(x<=val) = Total fraction of values x <= val + given zero fraction
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


def fit_chi2_cdf(ts_val, beta, ts, mus, p0=[1., 1., 1.]):
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
    ts : array-like, shape (len(mus), ntrials_per_mu)
        Test statistic values for each ``mu`` in ``mus``. These are used to
        calculate the CDF values used in the fit.
    par0 : list, optional
        Seed values ``[df, loc, scale]`` for the ``chi2`` CDF fit.
        (default: ``[1., 1., 1.]``)

    Returns
    -------
    mu_bf : float
        Best fit mean injected number of signal events to fullfill the
        tested performance level from ``ts_val`` and ``beta``.
    cdfs : array-like, shape (len(mus))

    pars : tuple
        Best fit parameters ``(df, loc, scale)`` for the ``chi2`` CDF.
    """
    # Get location of beta percentile per bunch of trial ts values
    cdfs = []
    errs = []
    for tsi in ts:
        cdf_i, err_i = weighted_cdf(tsi, val=ts_val, weights=None)
        cdfs.append(cdf_i)
        errs.append(err_i)
    cdfs, errs = np.array(cdfs), np.array(errs)

    def _cdf_func(x, df, loc, scale):
        """ Somehow we can't use scs.chi2.cdf directly for curve fit... """
        return chi2.cdf(x, df, loc, scale)

    try:
        # Missing stats get the largest global error estimate for the fit
        m = (errs == 0.)
        errs[m] = np.amax(errs[~m])
        pars, _ = sco.curve_fit(
            _cdf_func, xdata=mus, ydata=1. - cdfs, sigma=errs, p0=p0)
        mu_bf = chi2.ppf(beta, *pars)
    except RuntimeError:
        print(log.WARN("Couldn't find best params, returning `None` instead."))
        mu_bf = None
        pars = None

    return mu_bf, cdfs, pars


def scan_best_thresh(emp_dist, thresh_vals, pval_thresh=0.5):
    """
    Scans thresholds for the ``ExpTailEmpiricalDist`` distribution using a
    KS-test to test how good the tails fits to the data.

    Parameters
    ----------
    emp_dist : ExpTailEmpiricalDist instance
        Distribution object used to fit the tails to the stored data. The
        threshold is set to the best threshold found here after the scan.
    thresh_vals : array-like
        Threshold values to test.
    pval_thresh : float, optional
        The first threshold from the left, for which the scanned p-values are
        larger than ``pval_thresh`` is returned as the best fit threshold.
        (default: 0.5)

    Returns
    -------
    best_thresh : float
        First threshold for which the KS-test p-value was larger than
        ``pval_thresh``.
    best_idx ; int
        Index of the best threshold, can be used to match p-values and scales.
    pvals : array-like
        KS-test p-values for each exponential tail.
    scales : array-like
        Fitted scales for each exponential tail.
    """
    thresh_vals = np.atleast_1d(thresh_vals)

    pvals_ks = np.empty_like(thresh_vals)
    scales = np.empty_like(thresh_vals)
    for i, thresh in enumerate(thresh_vals):
        scales[i] = emp_dist.fit_thresh(thresh)
        pvals_ks[i] = kstest(emp_dist.get_split_data(emp=False)[0], "expon",
                             args=(thresh, scales[i])).pvalue

    # Choose first (=most stats) point where alpha error is above `pval_thresh`.
    # We'd rather say something about beta error, but we don't know that
    best_idx = np.where(pvals_ks > pval_thresh)[0][0]
    best_thresh = thresh_vals[best_idx]
    # Set best threshold to distribution instance
    emp_dist.fit_thresh(best_thresh)
    return best_thresh, best_idx, pvals_ks, scales


def make_equdist_bins(data, lo, hi, weights=None, upper_bound=None,
                      min_evts_per_bin=100):
    """
    Makes as much equidistant bins as possible with the condition, that at least
    ``min_evts_per_bin`` events are left in any bin.

    Parameters
    ----------
    data : array-like
        Data to be binned in a 1D histogram.
    lo, hi : float
        Left and right borders of the binning.
    weights : array-like, shape(len(data),), optional
        Data point weights, if ``None``all weiiiights are 1. If weights are
        given, data is resampled with replacement first to obtain unweighted
        events that get binned. (default: ``None``)
    upper_bound : int, optional
        Upper bound for the search algorithm. If ``None`` twice as much bins as
        needed if the data were uniformly distributed is chosen, which might
        fail. (default: ``None``)
    min_evts_per_bin : int, optional
        Number of events that must be left in any bin. Computed by the sum of
        weights. (default: 100)

    Return
    ------
    b : array-like
        Bin edges that match the ``min_evts_per_bin`` condition.
    """
    def fun(nbins, data, lo, hi, min_evts_per_bin):
        """ Return -1 if any bin has to few events and 1 if all have enough """
        b = np.linspace(lo, hi, int(nbins + 1))
        h, b = np.histogram(data, bins=b, weights=None, density=False)
        if np.any(h < min_evts_per_bin):
            return -1
        else:
            return 1

    if weights is not None:
        # If we had weighted data, resample to get unweighted events
        weights = np.atleast_1d(weights) / np.sum(weights)
        data = np.random.choice(data, p=weights, replace=True, size=len(data))
    if upper_bound is None:
        # Assume uniform distribution, might fail
        upper_bound = 2 * int(len(data) // min_evts_per_bin)

    # Find the root, but only with integer values
    nbins = int(sco.brentq(f=fun, a=1, b=upper_bound,
                           args=(data, lo, hi, min_evts_per_bin)))

    b = np.linspace(lo, hi, nbins + 1)
    h, b = np.histogram(data, bins=b, weights=None, density=False)
    if np.any(h < min_evts_per_bin):
        # If the higher integer nbins end is closest to the solution
        b = np.linspace(lo, hi, nbins)
        h, b = np.histogram(data, bins=b, weights=None, density=False)
    return b


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


class BaseEmpiricalDist(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def data(self):
        """ Data this distribtuion was built from. """
        pass

    @abc.abstractmethod
    def pdf(self):
        """ Calculates the probability density function """
        return

    @abc.abstractmethod
    def cdf(self):
        """ Calculates the cumulative distribution """
        return

    @abc.abstractmethod
    def sf(self):
        """ Calculates the survival function """
        return

    @abc.abstractmethod
    def ppf(self):
        """ Calculates the point percentile function """
        return

    @abc.abstractmethod
    def to_json(self):
        """ Encodes this class in JSON format for human readable storage """
        return

    @abc.abstractmethod
    def from_json(self):
        """ Creates a new instance from a ``to_json`` representation """
        return


class PureEmpiricalDist(BaseEmpiricalDist):
    """
    Class for a probability denstiy distribution modelled by using the empirical
    PDF defined by trial data only, eg. for a post-trial test statistic.
    """
    def __init__(self, data):
        """
        Parameters
        ----------
        data : array-like
            Data points of the empirical distribution.
        """
        self._data = np.sort(data)
        self._min, self._max = self._data[[0, -1]]

    @property
    def data(self):
        return self._data

    def pdf(self, x, dx=1.):
        """
        PDF value(s) at point(s) x. The PDF is modelled using a simple
        histogram.

        Parameters
        ----------
        x : array-like
            Points to evaluate the PDF at.
        dx : float, optional
            Bin stepsize for the histogram used to model the PDF. (default: 1.)

        Returns
        -------
        pdf : array-like
            PDF values.
        """
        x = np.atleast_1d(x)
        bins = np.arange(self._min, self._max + dx, dx)
        h, _ = np.histogram(self._data, bins=bins, density=True)
        # Read hist values bin heights for each x, ignore over-/underflow
        idx = np.digitize(x, bins) - 1
        valid = np.where((idx > -1) & (idx < len(bins) - 1))[0]
        pdf = np.zeros_like(x, dtype=float)
        pdf[valid] = h[idx[valid]]
        return pdf

    def cdf(self, x):
        """
        CDF value(s) at point(s) x using the empirical CDF.

        Parameters
        ----------
        x : array-like
            Points to evaluate the CDF at.

        Returns
        -------
        cdf : array-like
            CDF values.
        """
        return cdf_nzeros(self._data, nzeros=0, vals=x, sorted=True)

    def sf(self, x):
        """
        Survival function values, ``sf = 1 - cdf``. ``sf(0) = 1.``.

        Parameters
        ----------
        x : array-like
            Points to evaluate the survival function at.

        Returns
        -------
        sf : array-like
            Survival function values.
        """
        return 1. - self.cdf(x)

    def ppf(self, q):
        """
        Get distribution values for given percentiles ``q``. The empirical part
        is evaluated using empirical percentiles.

        Parameters
        ----------
        q : array-like
            Percentile(s) in ``[0, 100]``.

        Returns
        -------
        ppf: array-like, shape (len(q))
            Percentile values.
        """
        return percentile_nzeros(self._data, nzeros=0, q=q, sorted=True)

    def to_json(self, dtype=np.float, **json_args):
        """
        Get a representation of this class in JSON format to save on disk.

        Parameters
        ----------
        dtype : numpy data type
            Type to store the data array in. To save space, it might be
            sufficient to save the data 16bit floats and still have enough
            precision. (default: ``np.float``)
        json_args : keyword arguments
            Arguments given to ``json.dumps``.

        Returns
        -------
        json : str
            This class representation in JSON format. Can be used to restore the
            class using the ``from_json``method.

        Note
        ----
        See 'https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html' for
        numpy data types.
        """
        out = {"data": self._data.astype(dtype).tolist(),
               "dtype": "Data array was saved with {}".format(str(dtype))}
        return json.dumps(out, **json_args)

    @staticmethod
    def from_json(fp):
        """
        Build a new class object from a JSON string.

        Parameters
        ----------
        fp : file object
            File object given to ``json.dump``.

        Returns
        -------
        emp_dist : PureEmpiricalDist instance
            A new class instance made from the saved state.
        """
        sav = json.load(fp)
        return PureEmpiricalDist(data=np.array(sav["data"]))


class ExpTailEmpiricalDist(object):
    """
    Class for a probability denstiy distribution modelled by using the empirical
    PDF defined by trial data and a fitted exponetial tail, replacing the
    empirical PDF in a low statistics region. Only suitable for chi2 like test
    statisitics with only positive and / or zero values.

    Notes
    -----
    The probability density distribution for `delta_chi2` is:

    .. math::

      \text{PDF}(x, t, \lambda) =
          \begin{cases}
              \text{PDF}_\text{emp}(x)    &\text{for } 0<=x<t \\
              \text{CDF}_\text{emp}(t) \cdot
                \lambda e^{-\lambda (x-t)}  &\text{for } t<=x \\
          \end{cases}

    Here ``t`` is the threshold after which we switch from th empirical PDF to
    the exponential tail, ``lambda`` is the exponential decay parameter. The
    normalization of the exponential term is adapted to match the total
    normaliazation together with the empirical part.
    """
    def __init__(self, data, nzeros, thresh=0.):
        """
        Parameters
        ----------
        data : array-like
            All data points > 0.
        nzeros : int
            Number of extra zero trials not in ``data``.
        thresh : float, optional
            Threshold after which the exponential tail is used over the
            empirical statistic. (default: 0.)
        """
        data = np.atleast_1d(data)
        if np.any(data < 0):
            raise ValueError("Test statistic `data` points must be >= 0.")
        if nzeros < 0:
            raise ValueError("`nzeros` must be >= 0.")

        # Internal defaults
        self._scale = None
        self._emp_norm = None

        self._data = np.sort(data)
        self._nzeros = nzeros
        self.fit_thresh(thresh)

    @property
    def thresh(self):
        return self._thresh

    @property
    def scale(self):
        return self._scale

    @property
    def data(self):
        return self._data

    @property
    def nzeros(self):
        return self._nzeros

    def fit_thresh(self, thresh):
        """
        Refit the exponential tail for the given threshold.

        Parameters
        ----------
        thresh : float
            Threshold after which the exponential tail is used over the
            empirical statistic.
        """
        if thresh < 0.:
            raise ValueError("`thresh` must be >= 0.")
        is_exp = (self._data >= thresh)
        _, self._scale = expon.fit(self._data[is_exp], floc=thresh)
        # This is the part of the total normalization taken by the empirical PDF
        self._emp_norm = cdf_nzeros(x=self._data, nzeros=self._nzeros,
                                    vals=thresh, sorted=True)
        self._thresh = thresh
        return self._scale

    def pdf(self, x, dx=1.):
        """
        PDF value(s) at point(s) x. The empirical part is evaluated using a
        histogram.

        Parameters
        ----------
        x : array-like
            Points to evaluate the PDF at.
        dx : float, optional
            Bin steps. Bins for the empirical part are chosen to be
            ``numpy.concatenate((numpy.arange(0, thresh, dx) + [thresh]``.
            (default: 1.)

        Returns
        -------
        pdf : array-like
            PDF values.
        """
        x = np.atleast_1d(x)
        pdf = np.empty_like(x, dtype=float)
        is_exp = (x >= self._thresh)
        if not np.all(is_exp):
            bins = np.concatenate((np.arange(0., self._thresh, dx),
                                   [self._thresh]))
            # Create empirical PDF estimator from a histogram
            h, _ = np.histogram(self._data, bins=bins, density=False)
            h[0] += self._nzeros
            norm = np.diff(bins) * np.sum(h)
            h = h * self._emp_norm / norm
            # Read hist values bin heights for each x, ignore over-/underflow
            idx = np.digitize(x, bins) - 1
            valid = np.where((idx > -1) & (idx < len(bins) - 1))[0]
            pdf[~is_exp] = h[idx[valid]]
        # Assign exponential PDF part
        pdf[is_exp] = (1. - self._emp_norm) * expon.pdf(x[is_exp], self._thresh,
                                                        self._scale)
        return pdf

    def cdf(self, x):
        """
        CDF value(s) at point(s) x. The empirical part is evaluated using
        the empirical CDF.

        Parameters
        ----------
        x : array-like
            Points to evaluate the CDF at.

        Returns
        -------
        cdf : array-like
            CDF values.
        """
        x = np.atleast_1d(x)
        cdf = np.empty_like(x, dtype=float)
        is_exp = (x >= self._thresh)

        # Up to threshold, values are from empirical CDF
        cdf[~is_exp] = cdf_nzeros(self._data, self._nzeros, x[~is_exp],
                                  sorted=True)
        # Then use scaled exponential part
        cdf[is_exp] = self._emp_norm + (1. - self._emp_norm) * expon.cdf(
            x[is_exp], self._thresh, self._scale)
        return cdf

    def sf(self, x):
        """
        Survival function values, ``sf = 1 - cdf``. ``sf(0) = 1.``.

        Parameters
        ----------
        x : array-like
            Points to evaluate the survival function at.

        Returns
        -------
        sf : array-like
            Survival function values.
        """
        x = np.atleast_1d(x)
        sf = np.ones_like(x)
        m = (x > 0)
        sf[m] = 1. - self.cdf(x[m])
        return sf

    def ppf(self, q):
        """
        Get distribution values for given percentiles ``q``. The empirical part
        is evaluated using empirical percentiles.

        Parameters
        ----------
        q : array-like
            Percentile(s) in ``[0, 100]``.

        Returns
        -------
        ppf: array-like, shape (len(q))
            Percentile values.
        """
        q = np.atleast_1d(q).astype(np.float) / 100.
        ppf = np.empty_like(q, dtype=np.float)
        is_exp = (q >= self._emp_norm)
        # Up to threshold, percentiles are from empirical percentiles
        ppf[~is_exp] = percentile_nzeros(self._data, self._nzeros,
                                         100. * q[~is_exp], sorted=True)
        # Then use scaled exponential part
        perc_thresh = percentile_nzeros(self._data, self._nzeros,
                                        self._emp_norm, sorted=True)
        q_scaled = (q[is_exp] - self._emp_norm) / (1. - self._emp_norm)
        ppf[is_exp] = perc_thresh + expon.ppf(q_scaled, self._thresh,
                                              self._scale)
        return ppf

    def get_split_data(self, emp=False):
        """
        Get fraction of internally stored data which is used for the empirical
        or the exponential part.

        Parameters
        ----------
        emp : bool, optional
            If ``True`` returns the part of the data which is used for the
            empirical part, else for the exponential part. (default: ``False``)

        Returns
        -------
        data : array-like
            Data array used for the empirical or the exponential part.
        mask : array-like
            Bool mask to mask the full data array ``data = self.data[mask]``.
        """
        is_exp = (self._data >= self._thresh)
        if emp:
            return self._data[~is_exp], ~is_exp
        else:
            return self._data[is_exp], is_exp

    def data_hist(self, dx=1., density=False, which="all"):
        """
        Return a histogram of the internally stored data considering the number
        of zero trials.

        Parameters
        ----------
        dx : float, optional
            Bin steps. Bins for the empirical part are chosen to be
            ``numpy.arange(0, thresh + dx, dx)`` so the might slightly overlap
            in the exponential region (illustration purpose only). (default: 1.)
        density : bool, optional
            If ``True`` area under the histogram is 1 if ``which`` is ``'all'``.
            If ``which`` is ``'emp', 'exp'``, then the histogram parts are
            normalized with respect to the total data and not each on their own.
            (default: False)
        which : str
            Can be ``'all', 'emp', 'exp'``. If ``'all'``, creates histogram of
            all stored data, otherwise only the part used for the empirical or
            the exponential part of the distribution.

        Returns
        -------
        h : array-like
            Histogram values.
        bins : array-like, shape (len(h) + 1)
            Bins edges.
        err : array-like
            Properly scaled ``sqrt(sum(h))`` error of the histogram entries.
        norm : float
            Area under the histogram, is 1, if ``which='all'``, otherwise the
            fraction of the empirical or the exponetial CDF part.
        """
        if which == "all":
            bins = np.arange(0., np.amax(self._data) + dx, dx)
            _norm = 1.
        elif which == "emp":
            bins = np.arange(0., self._thresh + dx, dx)
            _norm = self._emp_norm
        elif which == "exp":
            _start = np.arange(0., self._thresh + dx, dx)[-1]
            bins = np.arange(_start, np.amax(self._data) + dx, dx)
            _norm = 1. - self._emp_norm
        else:
            raise ValueError("`which` can be one of 'all', 'emp', 'exp'.")

        h, bins = np.histogram(self._data, bins, density=False)
        zero_idx = np.digitize(0., bins) - 1
        if (zero_idx > -1) & (zero_idx < len(bins) - 1):
            h[zero_idx] += self._nzeros
        err = np.sqrt(h)
        if density:
            norm = np.diff(bins) * np.sum(h)
            h = h / norm * _norm
            err = err / norm * _norm
        return h, bins, err, _norm

    def to_json(self, fp, dtype=np.float, **json_args):
        """
        Save this class to disk in JSON format.

        Parameters
        ----------
        fp : file object
            File object given to   ``json.dump``.
        dtype : numpy data type
            Type to store the data array in. To save space, the test statisitc
            can be saved in 16bit floats and still have more than sufficient
            precision. (default: ``np.float``)
        json_args : keyword arguments
            Arguments given to ``json.dump``.

        Note
        ----
        See 'https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html' for
        numpy data types.
        """
        out = {"data": self._data.astype(dtype).tolist(),
               "nzeros": self._nzeros,
               "thresh": self._thresh,
               "dtype": "Data array was saved with {}".format(str(dtype))}
        json.dump(out, fp=fp, **json_args)

    @staticmethod
    def from_json(fp):
        """
        Build a new class object from a saved JSON file.

        Parameters
        ----------
        fp : file object
            File object given to ``json.dump``.

        Returns
        -------
        emp_with_exp_tail_dist : ExpTailEmpiricalDist object
            A new class instance made from the saved state.
        """
        sav = json.load(fp)
        return ExpTailEmpiricalDist(data=np.array(sav["data"]),
                                    nzeros=sav["nzeros"],
                                    thresh=sav["thresh"])
