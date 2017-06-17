# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import zip, super
from future import standard_library
standard_library.install_aliases()                                              # noqa

import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state

from .utils import (rejection_sampling, func_min_in_interval,
                    flatten_list_of_1darrays)

import abc     # Abstract Base Class
import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class RateFunction(object):
    """
    Rate Function Base Class

    Base class for rate functions describing time dependent background rates.
    Rate function must be interpretable as a PDF and must not be negative.

    Classes must implement methods:

    - `fun`
    - `integral`
    - `sample`
    - `_get_default_seed`

    Class object then provides public methods:

    - `fun`
    - `integral`
    - `fit`
    - `sample`
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._SECINDAY = 24. * 60. * 60.

    @docs.get_sectionsf("RateFunction.fun", sections=["Parameters", "Returns"])
    @docs.dedent
    @abc.abstractmethod
    def fun(self, t, pars):
        """
        Returns the rate in Hz at a given time t in MJD.

        Parameters
        ----------
        t : array-like, shape (nevts)
            MJD times of experimental data.
        pars : tuple
            Further parameters the function depends on.

        Returns
        -------
        rate : array-like
            Rate in Hz for each time `t`.
        """
        pass

    @docs.get_sectionsf("RateFunction.integral",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    @abc.abstractmethod
    def integral(self, t, trange, pars):
        """
        Integral of rate function in intervals trange around source times t.

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows `[[t0, t1], ...]` in seconds around each time `t`.
        pars : tuple
            Further parameters `self.fun` depends on.

        Returns
        -------
        integral : array-like, shape (nsrcs)
            Integral of `self.fun` within given time windows `trange`.
        """
        pass

    @docs.get_summaryf("RateFunction.sample")
    @docs.get_sectionsf("RateFunction.sample",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    @abc.abstractmethod
    def sample(self, t, trange, pars, n_samples, random_state=None):
        """
        Generate random samples from the rate function for multiple source times
        and time windows.

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows `[[t0, t1], ...]` in seconds around each time `t`.
        pars : tuple
            Further parameters `self.fun` depends on.
        n_samples : array-like, shape (nsrcs)
            Number of events to sample per source.
        random_state : seed, optional
            Turn seed into a `np.random.RandomState` instance. See
            `sklearn.utils.check_random_state`. (default: None)

        Returns
        -------
        times : list of arrays, len (nsrcs)
            Sampled times in MJD of background events per source. If `n_samples`
            is 0 for a source, an empty arrays is placed at that position.
        """
        pass

    @docs.get_summaryf("RateFunction._get_default_seed")
    @docs.get_sectionsf("RateFunction._get_default_seed",
                        sections=["Parameters"])
    @docs.dedent
    @abc.abstractmethod
    def _get_default_seed(self, t, trange, rate_std):
        """
        Default seed values for the specifiv RateFunction fit.

        Parameters
        ----------
        t : array-like
            MJD times of experimental data.
        rate : array-like, shape (len(t))
            Rates at given times `t` in Hz.
        rate_std : array-like, shape(len(t)), optional
            Standard deviations for each datapoint. If None, all are set to 1.
            If rate_std is a good description of the standard deviation, then
            the fit statistics follows a :math:`\chi^2` distribution. But for a
            binned fit this makes less sense, because low standard deviation
            means low statistics, so better use unweighted. (default: None)

        Returns
        -------
        p0 : tuple
            Seed values for each parameter the specific `RateFunction` uses.
        """
        pass

    @docs.get_summaryf("RateFunction.fit")
    @docs.get_sectionsf("RateFunction.fit",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    def fit(self, t, rate, p0=None, rate_std=None, **kwargs):
        """
        Fits the function parameters to experimental data using a weighted
        least squares fit.

        Parameters
        ----------
        t : array-like
            MJD times of experimental data.
        rate : array-like, shape (len(t))
            Rates at given times `t` in Hz.
        p0 : tuple, optional
            Seed values for the fit parameters. If None, default ones are used,
            that may or may not work. (default: None)
        rate_std : array-like, shape(len(t)), optional
            Standard deviations for each datapoint. If None, all are set to 1.
            If rate_std is a good description of the standard deviation, then
            the fit statistics follows a :math:`\chi^2` distribution. But for a
            binned fit this makes less sense, because low standard deviation
            means low statistics, so better use unweighted. (default: None)
        kwargs
            Other keywords are passed to `scipy.optimize.minimize`.

        Returns
        -------
        bf_pars : array-like
            Values of the best fit parameters.
        """
        if rate_std is None:
            rate_std = np.ones_like(rate)

        if p0 is None:
            p0 = self._get_default_seed(t, rate, rate_std)

        # t, rate and rate_std are fixed
        args = (t, rate, rate_std)
        res = sco.minimize(fun=self._lstsq, x0=p0, args=args, **kwargs)

        return res.x

    def _lstsq(self, pars, *args):
        """
        Weighted leastsquares loss: :math:`\sum_i (w_i * (y_i - f_i))^2`

        Parameters
        ----------
        pars : tuple
            Fitparameter for `self.fun` that gets fitted.
        args : tuple
            Fixed values `(t, rate, rate_std)` for the loss function:

            - t, array-like: See :func:`RateFunction.fit`, Parameters
            - rate, array-like, shape (len(t)): See :func:`RateFunction.fit`,
              Parameters
            - rate_std, array-like, shape(len(t)): See :func:`RateFunction.fit`,
              Parameters

            The weights are :math:`w_i = 1/\text{rate_std}_i` which is
            equivalent to the weighted mean, when a constant is fitted.

        Returns
        -------
        loss : float
            The weighted least squares loss for the given `pars` and `args`.
        """
        t, rate, rate_std = args
        w = 1 / rate_std
        fun = self.fun(t, pars)
        return np.sum((w * (rate - fun))**2)

    def _transform_trange_mjd(self, t, trange):
        """
        Transform time window to MJD and check on correct shapes

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows `[[t0, t1], ...]` in seconds around each time `t`.

        Returns
        -------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows `[[t0, t1], ...]` in MJD around each time `t`.
        """
        t = np.atleast_1d(t)
        nsrcs = len(t)
        # Proper braodcasting to process multiple srcs at once
        t = t.reshape(nsrcs, 1)
        trange = np.atleast_2d(trange).reshape(nsrcs, 2)
        return t, t + trange / self._SECINDAY


class SinusRateFunction(RateFunction):
    """
    Sinus Rate Function

    Describes time dependent background rate. Used function is a sinus with:

    .. math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

    depending on 4 parameters:

    - a, float: Amplitude in Hz.
    - b, float: Angular frequency, :math:`\omega = 2\pi/T` with period :math:`T`
      given in 1/MJD.
    - c, float: x-axis offset in MJD.
    - d, float: y-axis offset in Hz

    When `t, trange` where given at object creation the function bounds for
    the rejection sampling step are precalculated in `fit` to save computation
    time on subsequent calls of `sample`.

    Parameters
    ----------
    t : array-like, shape (nsrcs)
        MJD times of sources.
    trange : array-like, shape(nsrcs, 2)
        Time windows `[[t0, t1], ...]` in seconds around each time `t`.
    """
    def __init__(self, t=None, trange=None):
        super(SinusRateFunction, self).__init__()

        self.t, self.trange = None, None
        if (t is not None) and (trange is not None):
            self.t, self.trange = self._transform_trange_mjd(t, trange)
        self._fmax = None
        return

    @docs.dedent
    def fit(self, t, rate, p0=None, rate_std=None, **kwargs):
        """
        %(RateFunction.fit.summary)s

        When `t, trange` where given at object creation the function bounds for
        the rejection sampling step are precalculated here to save computation
        time on subsequent calls of `sample`.

        Parameters
        ----------
        %(RateFunction.fit.parameters)s

        Returns
        -------
        %(RateFunction.fit.returns)s
        """
        # After fitting, cache the maximum of the pdf to avoid recalculating
        # that in the rejection sampling step
        bf_pars = super(SinusRateFunction, self).fit(t, rate, p0, rate_std,
                                                     **kwargs)

        if self.t is not None:
            def negpdf(t):
                return -1. * self.fun(t, bf_pars)

            _fmax = []
            for bound in self.trange:
                _fmax.append(-1. * func_min_in_interval(negpdf, bound))
            self._fmax = flatten_list_of_1darrays(_fmax)

        return bf_pars

    @docs.dedent
    def fun(self, t, pars):
        """
        Returns the rate at a given time in MJD from a sinusodial function.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s

            See :class:`SinusRateFunction`, Summary

        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        a, b, c, d = pars
        return a * np.sin(b * (t - c)) + d

    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Analytic integral of the sinusodial rate function in interval trange.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s

            See :class:`SinusRateFunction`, Summary

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        a, b, c, d = pars

        # Transform time windows to MJD
        t, dts = self._transform_trange_mjd(t, trange)
        t0, t1 = dts[:, 0], dts[:, 1]

        # Split analytic expression for readability only
        per = a / b * (np.cos(b * (t0 - c)) - np.cos(b * (t1 - c)))
        lin = d * (t1 - t0)

        # Match units with secinday = 24 * 60 * 60 s/MJD = 86400 / (Hz*MJD)
        #     [a], [d] = Hz;M [b] = 1/MJD; [c], [t] = MJD
        #     [a / b] = Hz * MJD; [d * (t1 - t0)] = HZ * MJD
        return (per + lin) * self._SECINDAY

    @docs.dedent
    def sample(self, t, trange, pars, n_samples, random_state=None):
        """
        %(RateFunction.sample.summary)s

        For `pars`, see :class:`SinusRateFunction`, summary.

        Parameters
        ----------
        %(RateFunction.sample.parameters)s

        Returns
        -------
        %(RateFunction.sample.returns)s
        """
        n_samples = np.atleast_1d(n_samples)

        def sample_fun(t):
            """Wrapper to have only one argument."""
            return self.fun(t, pars)

        t, dts = self._transform_trange_mjd(t, trange)

        # Samples times for all sources at once
        times = rejection_sampling(sample_fun, bounds=dts, n_samples=n_samples,
                                   max_fvals=self._fmax,
                                   random_state=random_state)

        return times

    @docs.dedent
    def _get_default_seed(self, t, rate, rate_std):
        """
        %(RateFunction._get_default_seed.summary)s

        Motivation for default seed:

        - a0 : Using the maximum amplitude in variation of rate bins.
        - b0 : The expected seasonal variation is 1 year.
        - c0 : Earliest time in `t`.
        - d0 : Weighted averaged rate, which is the best fit value for a
          constant target function.

        Parameters
        ----------
        %(RateFunction._get_default_seed.parameters)s

        Returns
        -------
        p0 : tuple, shape (4)
            Seed values `(a0, b0, c0, d0)`:

            - a0 : :math:`(\max(\text{rate}) + \min(\text{rate})) / 2`
            - b0 : :math:`2\pi / 365`
            - c0 : :math:`\min(t)`
            - d0 : `np.average(rate, weights=rate_std**2)`
        """
        a0 = 0.5 * (np.amax(rate) - np.amin(rate))
        b0 = 2. * np.pi / 365.
        c0 = np.amin(t)
        d0 = np.average(rate, weights=rate_std**2)
        return (a0, b0, c0, d0)


class Sinus1yrRateFunction(SinusRateFunction):
    """
    Sinus Rate Function fixing the period to 1 year.

    Function describes time dependent background rate. Function is a sinus with

    .. math:: f(t|a,c,d) = a \sin((2\pi/\mathrm{365.25}) (t - c)) + d

    depending on 3 parameters:

    - a, float: Amplitude in Hz.
    - c, float: x-axis offset in MJD.
    - d, float: y-axis offset in Hz

    When `t, trange` where given at object creation the function bounds for
    the rejection sampling step are precalculated in `fit` to save computation
    time on subsequent calls of `sample`.

    Parameters
    ----------
    t : array-like, shape (nsrcs)
        MJD times of sources.
    trange : array-like, shape(nsrcs, 2)
        Time windows `[[t0, t1], ...]` in seconds around each time `t`.
    """
    def __init__(self, t=None, trange=None):
        super(Sinus1yrRateFunction, self).__init__(t, trange)

        self._b = 2 * np.pi / 365.25
        return

    @docs.dedent
    def fun(self, t, pars):
        """
        Returns the rate at a given time in MJD from a sinusodial function.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s

            See :class:`Sinus1yrRateFunction`, Summary

        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        a, c, d = pars
        pars = (a, self._b, c, d)  # Just inject the fixed par in the super func
        return super(Sinus1yrRateFunction, self).fun(t, pars)

    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Analytic integral of the sinusodial rate function in interval trange.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s

            See :class:`Sinus1yrRateFunction`, Summary

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        a, c, d = pars
        pars = (a, self._b, c, d)  # Just inject the fixed par in the super func
        return super(Sinus1yrRateFunction, self).integral(t, trange, pars)

    def _get_default_seed(self, t, rate, rate_std):
        """
        %(RateFunction._get_default_seed.summary)s

        Motivation for default seed:

        - a0 : Using the maximum amplitude in variation of rate bins.
        - c0 : Earliest time in `t`.
        - d0 : Weighted averaged rate, which is the best fit value for a
           constant target function.

        Parameters
        ----------
        %(RateFunction._get_default_seed.parameters)s

        Returns
        -------
        p0 : tuple, shape (3)

            See :class:`Sinus1yrRateFunction._get_default_seed`, Returns.
            Here `b` is fixed, so only `(a0, c0, d0)` are returned.
        """
        seed = super(Sinus1yrRateFunction, self)._get_default_seed(t, rate,
                                                                   rate_std)
        # Drop b0 seed, as it's fixed to 1yr here
        return tuple(seed[i] for i in [0, 2, 3])


class ConstantRateFunction(RateFunction):
    """
    Uses a constant rate in Hz at a given time in MJD.

    Uses one parameter:

    - rate, float: Constant rate in Hz.
    """
    def __init__(self):
        super(ConstantRateFunction, self).__init__()
        return

    @docs.dedent
    def fun(self, t, pars):
        """
        Returns a constant rate Hz at any given time in MJD by simply
        broadcasting the input rate.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s

            See :class:`ConstantRateFunction`, Summary

        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        (rate,) = pars
        return np.ones_like(t) * rate

    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Analytic integral of the rate function in interval trange. Because the
        rate function is constant, integral is simply `trange * rate`.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s

            See :class:`ConstantRateFunction`, Summary

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        t, dts = self._transform_trange_mjd(t, trange)
        # Multiply first then diff to avoid roundoff errors
        return (np.diff(dts * self._SECINDAY, axis=1) *
                self.fun(t, pars)).flatten()

    @docs.dedent
    def sample(self, t, trange, pars, n_samples, random_state=None):
        """
        %(RateFunction.sample.summary)s

        Parameters
        ----------
        %(RateFunction.sample.parameters)s

        Returns
        -------
        %(RateFunction.sample.returns)s
        """
        # Just sample uniformly in MJD time windows
        rndgen = check_random_state(random_state)
        t, dts = self._transform_trange_mjd(t, trange)

        # Samples times for all sources at once
        sample = []
        for dt, nsam in zip(dts, n_samples):
            sample.append(rndgen.uniform(dt[0], dt[1], size=nsam))

        return sample

    @docs.dedent
    def _get_default_seed(self, t, rate, rate_std):
        """
        %(RateFunction._get_default_seed.summary)s

        Motivation for default seed:

        - rate0 : Mean of the given rates. This is the anlytic solution to the
          fit, so we seed with the best fit.

        Parameters
        ----------
        %(RateFunction._get_default_seed.parameters)s

        Returns
        -------
        p0 : tuple, shape(1)
            Seed values (rate0):

            - rate0 : `np.mean(rate)`
        """
        return (np.mean(rate), )
