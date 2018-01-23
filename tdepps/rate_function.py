# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import zip, super
from future import standard_library
standard_library.install_aliases()

import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state

from .utils import rejection_sampling, func_min_in_interval

import abc     # Abstract Base Class
import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class RateFunction(object):
    __metaclass__ = abc.ABCMeta

    @docs.get_sectionsf("RateFunction.init", sections=["Parameters"])
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Rate Function Base Class

        Base class for rate functions describing time dependent background
        rates. Rate function must be interpretable as a PDF and must not be
        negative.

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

        Parameters
        ----------
        random_state : seed, optional
            Turn seed into a `np.random.RandomState` instance. See
            `sklearn.utils.check_random_state`. (default: None)
        """
        self.rndgen = random_state
        self._SECINDAY = 24. * 60. * 60.

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

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
    def sample(self, t, trange, pars, n_samples):
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
    def _get_default_seed(self, t, trange, w):
        """
        Default seed values for the specifiv RateFunction fit.

        Parameters
        ----------
        t : array-like
            MJD times of experimental data.
        rate : array-like, shape (len(t))
            Rates at given times `t` in Hz.
        w : array-like, shape(len(t)), optional
            Weights for least squares fit: :math:`\sum_i (w_i * (y_i - f_i))^2`.
            (default: None)

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
    def fit(self, t, rate, p0=None, w=None, **kwargs):
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
        w : array-like, shape(len(t)), optional
            Weights for least squares fit: :math:`\sum_i (w_i * (y_i - f_i))^2`.
            (default: None)
        kwargs
            Other keywords are passed to `scipy.optimize.minimize`.

        Returns
        -------
        bf_pars : array-like
            Values of the best fit parameters.
        """
        if w is None:
            w = np.ones_like(rate)

        if p0 is None:
            p0 = self._get_default_seed(t, rate, w)

        # t, rate and w are fixed
        args = (t, rate, w)
        res = sco.minimize(fun=self._lstsq, x0=p0, args=args, **kwargs)
        self._res = res
        return res.x

    def _lstsq(self, pars, *args):
        """
        Weighted leastsquares loss: :math:`\sum_i (w_i * (y_i - f_i))^2`

        Parameters
        ----------
        pars : tuple
            Fitparameter for `self.fun` that gets fitted.
        args : tuple
            Fixed values `(t, rate, w)` for the loss function:

            - t, array-like: See :py:meth:`RateFunction.fit`, Parameters
            - rate, array-like, shape (len(t)): See :py:meth:`RateFunction.fit`,
              Parameters
            - w, array-like, shape(len(t)): See :py:meth:`RateFunction.fit`,
              Parameters

        Returns
        -------
        loss : float
            The weighted least squares loss for the given `pars` and `args`.
        """
        t, rate, w = args
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
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Sinus Rate Function

        Describes time dependent background rate. Used function is a sinus with:

        .. math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

        depending on 4 parameters:

        - a, float: Amplitude in Hz.
        - b, float: Angular frequency, :math:`\omega = 2\pi/T` with period
          :math:`T` given in 1/MJD.
        - c, float: x-axis offset in MJD.
        - d, float: y-axis offset in Hz

        Parameters
        ----------
        %(RateFunction.init.parameters)s
        """
        super(SinusRateFunction, self).__init__(random_state)
        self._t = None
        self._trange = None
        self._dts = None
        self._fmax = None

        self._names = ["amplitude", "period", "toff", "baseline"]
        return

    @docs.dedent
    def fit(self, t, rate, p0=None, w=None, **kwargs):
        """
        %(RateFunction.fit.summary)s

        Parameters
        ----------
        %(RateFunction.fit.parameters)s

        Returns
        -------
        %(RateFunction.fit.returns)s
        """
        bf_pars = super(SinusRateFunction, self).fit(t, rate, p0, w, **kwargs)
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
        _, dts = self._transform_trange_mjd(t, trange)
        t0, t1 = dts[:, 0], dts[:, 1]

        # Split analytic expression for readability only
        per = a / b * (np.cos(b * (t0 - c)) - np.cos(b * (t1 - c)))
        lin = d * (t1 - t0)

        # Match units with secinday = 24 * 60 * 60 s/MJD = 86400 / (Hz*MJD)
        #     [a], [d] = Hz;M [b] = 1/MJD; [c], [t] = MJD
        #     [a / b] = Hz * MJD; [d * (t1 - t0)] = HZ * MJD
        return (per + lin) * self._SECINDAY

    @docs.dedent
    def sample(self, t, trange, pars, n_samples):
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

        # If we always get the same t and trange, cache the fmax values for
        # faster rejection sampling after the first encounter, else reset
        if not (np.array_equal(t, self._t) and
                np.array_equal(trange, self._trange)):
            # Store the raw input otherwise test is always false
            self._t = t
            self._trange = trange
            _, self._dts = self._transform_trange_mjd(t, trange)
            self._fmax = None

        # Samples times for all sources at once
        times, self._fmax = rejection_sampling(
            sample_fun, bounds=self._dts, n_samples=n_samples,
            rndgen=self._rndgen, max_fvals=self._fmax)

        return times

    @docs.dedent
    def _get_default_seed(self, t, rate, w):
        """
        %(RateFunction._get_default_seed.summary)s

        Motivation for default seed:

        - a0 : Using the width of the central 50\% percentile of the rate
               distribtion (for rates > 0). The sign is determined based on
               wether the average rates in the first two octants based on the
               period seed decrese or increase.
        - b0 : The expected seasonal variation is 1 year.
        - c0 : Earliest time in ``t``.
        - d0 : Weighted averaged rate, which is the best fit value for a
               constant target function.

        Parameters
        ----------
        %(RateFunction._get_default_seed.parameters)s

        Returns
        -------
        p0 : tuple, shape (4)
            Seed values `(a0, b0, c0, d0)`:

            - a0 : ``-np.diff(np.percentile(rate[w > 0], q=[0.25, 0.75])) / 2``
            - b0 : :math:`2\pi / 365`
            - c0 : :math:`\min(t)`
            - d0 : ``np.average(rate, weights=w)``
        """
        a0 = 0.5 * np.diff(np.percentile(rate[w > 0], q=[0.25, 0.75]))[0]
        b0 = 2. * np.pi / 365.
        c0 = np.amin(t)
        d0 = np.average(rate, weights=w)

        # Get the sign of the amplitude a0 depending on wether the average
        # falls or rises in the first 2 octants of the whole period.
        m0 = (c0 <= t) & (t <= c0 + 365. / 8.)
        oct0 = np.average(rate[m0], weights=w[m0])
        m1 = (c0 <= t + 365. / 8.) & (t <= c0 + 365. / 4.)
        oct1 = np.average(rate[m1], weights=w[m1])
        sign = np.sign(oct1 - oct0)

        return (sign * a0, b0, c0, d0)


class SinusFixedRateFunction(SinusRateFunction):
    @docs.dedent
    def __init__(self, p_fix=None, t0_fix=None, random_state=None):
        """
        Sinus Rate Function but with a-priori fixed period.

        Function describes time dependent background rate with fixed period:

        .. math:: f(t|a,b,c,d) = a \sin((2\pi/\mathrm{b}_\mathrm{fix}) (t-c))+d

        depending on 4 parameters:

        - a, float: Amplitude in Hz.
        - b, float: Period in MJD.
        - c, float: x-axis offset in MJD.
        - d, float: y-axis offset in Hz

        Here we can fix the parameters ``b`` and / or ``c`` and only fit the
        other ones.

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        p_fix : float or None
            Fixed period in MJD days for the sine function. If ``None``, the
            parameter is not fixed and fitted. If a float is given, the
            parameter stays fied on that value. (Default: None)
        t0_fix : float
            Fixed start time in MJD days for the sine function. If ``None``, the
            parameter is not fixed and fitted. If a float is given, the
            parameter stays fied on that value. (Default: None)
        trange : array-like, shape(nsrcs, 2)
            Time windows `[[t0, t1], ...]` in seconds around each time `t`.
        %(RateFunction.init.parameters)s
        """
        super(SinusFixedRateFunction, self).__init__(random_state)

        # Process which parameters are fixed and which get fitted
        self._fit_idx = np.ones(4, dtype=bool)
        self._b = None
        self._c = None

        if p_fix is not None:
            if p_fix <= 0.:
                raise ValueError("Fixed period must be >0 days.")
            self._fit_idx[1] = False
            self._b = 2 * np.pi / p_fix

        if t0_fix is not None:
            self._fit_idx[2] = False
            self._c = t0_fix

        self._fit_idx = np.arange(4)[self._fit_idx]

        return

    @docs.dedent
    def fun(self, t, pars):
        """
        Returns the rate at a given time in MJD from a sinusodial function.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s

            See :class:`SinusFixedRateFunction`, Summary

        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).fun(t, pars)

    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Analytic integral of the sinusodial rate function in interval trange.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s

            See :class:`SinusFixedRateFunction`, Summary

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).integral(t, trange, pars)

    def _get_default_seed(self, t, rate, w):
        """
        %(RateFunction._get_default_seed.summary)s

        Motivation for default seed:

        - a0 : Using the maximum amplitude in variation of rate bins.
        - b0 : :math:`2\pi / 365`
        - c0 : Earliest time in `t`.
        - d0 : Weighted averaged rate, which is the best fit value for a
           constant target function.

        Parameters
        ----------
        %(RateFunction._get_default_seed.parameters)s

        Returns
        -------
        p0 : tuple, shape (3)

            See :class:`SinusFixedRateFunction._get_default_seed`, Returns.
            Here ``b`` and / or ``c`` may be fixed, so only the actual fit
            parameter seeds are returned.
        """
        seed = super(SinusFixedRateFunction,
                     self)._get_default_seed(t, rate, w)
        # Drop b0 and / or c0 seed, when it's marked as fixed
        return tuple(seed[i] for i in self._fit_idx)

    def _make_params(self, pars):
        """
        Check which parameters are fixed and insert them where needed to build
        a full parameter set.

        Parameters
        ----------
        pars : tuple
            See See :py:meth:`fun`, Parameters

        Returns
        -------
        pars : tuple
            Fixed parameters inserted in the full argument list.
        """
        if len(pars) != len(self._fit_idx):
            raise ValueError("Given number of parameters does not match the " +
                             "number of free parameters here.")
        # Explicit here, but OK, because we have only 4 combinations
        if self._b is None:
            if self._c is None:
                pars = pars
            pars = (pars[0], pars[1], self._c, pars[2])
        elif self._c is None:
            pars = (pars[0], self._b, pars[1], pars[2])
        else:
            pars = (pars[0], self._b, self._c, pars[1])
        return pars


class SinusFixedConstRateFunction(SinusFixedRateFunction):
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Same as SinusFixedRateFunction, but sampling uniform times in each time
        interval instead of rejection sampling the sine function.

        Here the number of expected events is still following the seasonal
        fluctuations, but within the time windows we sample uniformly (step
        function like). Perfect for small time windows, avoiding rejection
        sampling and thus giving a speed boost.

        Parameters
        ----------
        %(RateFunction.init.parameters)s
        """
        super(SinusFixedConstRateFunction, self).__init__(random_state)
        return

    @docs.dedent
    def sample(self, t, trange, pars, n_samples):
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
        _, dts = self._transform_trange_mjd(t, trange)

        # Samples times for all sources at once
        sample = []
        for dt, nsam in zip(dts, n_samples):
            sample.append(self._rndgen.uniform(dt[0], dt[1], size=nsam))

        return sample


class ConstantRateFunction(RateFunction):
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Uses a constant rate in Hz at a given time in MJD. This one models no
        seasonal fluctuations but describes the rate as a constant.

        Uses one parameter:

        - rate, float: Constant rate in Hz.

        Parameters
        ----------
        %(RateFunction.init.parameters)s
        """
        self._names = ["baseline"]
        super(ConstantRateFunction, self).__init__(random_state)
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
        return np.ones_like(t) * pars[0]

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
                self.fun(t, pars)).ravel()

    @docs.dedent
    def sample(self, t, trange, pars, n_samples):
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
        _, dts = self._transform_trange_mjd(t, trange)

        # Samples times for all sources at once
        sample = []
        for dt, nsam in zip(dts, n_samples):
            sample.append(self._rndgen.uniform(dt[0], dt[1], size=nsam))

        return sample

    @docs.dedent
    def _get_default_seed(self, t, rate, w):
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
