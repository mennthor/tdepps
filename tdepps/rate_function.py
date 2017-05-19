import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state

from anapymods3.stats.sampling import rejection_sampling

import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class RateFunction(object):
    """
    Interface for rate functions describing time dependent background rates.

    Classes must implement functions ["fun", "integral", "fit", "sample"].
    Rate functio must be interpretable as a PDF and must not be negative.
    """
    # Set up globals for shared inherited constants
    _SECINDAY = 24. * 60. * 60.

    def __init__(self):
        self._DESCRIBES = ["fun", "integral", "fit", "sample"]
        print("Interface only. Describes functions: ", self._DESCRIBES)
        return

    @docs.get_sectionsf("RateFunction.fun", sections=["Parameters", "Returns"])
    @docs.dedent
    def fun(self, t, pars):
        """
        Returns the rate in Hz at a given time t in MJD.

        Parameters
        ----------
        t : array-like, shape (n_samples)
            MJD times of experimental data.
        pars : tuple, optional
            Further parameters `fun` depends on.

        Returns
        -------
        rate : array-like
            Rate in Hz for each time t.
        """
        raise NotImplementedError("RateFunction is an interface.")

    @docs.get_sectionsf("RateFunction.integral",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Integral of rate function in interval trange around time t in MJD.

        Parameters
        ----------
        t : float
            Reference time in MJD where the integration interval is aligned to.
        trange : array-like, shape(2)
            Time window `[t0, t1]` in seconds around the source time t.
        pars : tuple, optional
            Further parameters `fun` depends on.

        Returns
        -------
        integral : float
            Integral of `self.fun` within time windows `trange`.
        """
        raise NotImplementedError("RateFunction is an interface.")

    @docs.get_summaryf("RateFunction.sample")
    @docs.get_sectionsf("RateFunction.sample",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    def sample(self, t, trange, pars, n_samples=1, random_state=None):
        """
        Generate random samples from the rate function for one time and a single
        time window.

        Parameters
        ----------
        t : float
            Time of the occurance of the source event in MJD.
        trange : array-like, shape(2)
            Time window `[t0, t1]` in seconds around the source time t.
        pars : tuple, optional
            Further parameters `fun` depends on.
        n_samples : int
            Number of events to sample. (default: 1)
        random_state : RandomState, optional
            A random number generator instance. (default: None)

        Returns
        -------
        times : array-like, shape (n_samples)
            Sampled times in MJD of background events per source. If `n_samples`
            is 0, an empty array is returned.
        """
        raise NotImplementedError("RateFunction is an interface.")

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
            Rates at given times t in Hz.
        p0 : tuple
            Seed values for the fit parameters. If None, default ones are used,
            that may or may not work. (default: None)
        rate_std : array-like, shape(len(t)), optional
            Standard deviations for each datapoint. If None, all are set to 1.
            If `rate_std` is a good description of the standard deviation, then
            the fit statistics follows a chi2 distribution. (default: None)
        kwargs
            Other keywords are passed to `scipy.optimize.minimize`.

        Returns
        -------
        res.x : array-like
            Values of the best fit parameters.
        """
        raise NotImplementedError("RateFunction is an interface.")

    def _lstsq(self, pars, *args):
        """
        Weighted leastsquares loss: sum_i((wi * (yi - fi))**2)

        Parameters
        ----------
        pars : tuple
            Fitparameter for `self.fun` that gets fitted.
        args : tuple
            Fixed values for the loss function: (t, rate, weights)

        Returns
        -------
        loss : float
            The weighted least squares loss for the given `pars` and `args`.
        """
        t, rate, w = args
        fun = self.fun(t, pars)
        return np.sum((w * (rate - fun))**2)


class SinusRateFunction(RateFunction):
    """
    Sinus Rate Function

    Function describes time dependent background rate. Function is a sinus with:

    .. math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

    where:

    - a is the Amplitude in Hz
    - b is the period scale in 1/MJD
    - c is the x-offset in MJD
    - d the y-offset in Hz

    Time is measured in MJD days.
    """
    def __init__(self):
        return

    @docs.dedent
    def fun(self, t, pars):
        """
        Returns the rate at a given time in MJD from a sinusodial function.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s
            Here `pars` are (a, b, c, d) as stated above.


        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        a, b, c, d = pars
        return a * np.sin(b * (t - c)) + d

    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Analytic integral of the sinusodial rate function in interval `trange`.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s
            Here `pars` are (a, b, c, d) as stated above.

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        a, b, c, d = pars

        # Transform time window to MJD
        t0, t1 = self._transform_trange_mjd(t, trange)

        # Split analytic expression for readability only
        per = a / b * (np.cos(b * (t0 - c)) - np.cos(b * (t1 - c)))
        lin = d * (t1 - t0)

        # Match units with secinday = 24 * 60 * 60 s/MJD = 86400 / (Hz*MJD)
        #     [a], [d] = Hz;M [b] = 1/MJD; [c], [t] = MJD
        #     [a / b] = Hz * MJD; [d * (t1 - t0)] = HZ * MJD
        return (per + lin) * RateFunction._SECINDAY

    @docs.dedent
    def sample(self, t, trange, pars, n_samples=1, random_state=None):
        """
        %(RateFunction.sample.summary)s

        Parameters
        ----------
        %(RateFunction.sample.parameters)s

        Returns
        -------
        %(RateFunction.sample.returns)s
        """
        rndgen = check_random_state(random_state)

        # rejection_sampling needs bounds in shape (1, 2)
        trange = self._transform_trange_mjd(t, trange).T

        def sample_fun(t):
            return self.fun(t, pars)

        times = rejection_sampling(sample_fun, bounds=trange, n=n_samples,
                                   random_state=rndgen)[0]

        return times

    @docs.dedent
    def fit(self, t, rate, p0=None, rate_std=None, **kwargs):
        """
        %(RateFunction.fit.summary)s

        Parameters
        ----------
        %(RateFunction.fit.parameters)s

        Returns
        -------
        %(RateFunction.fit.returns)s
        """
        if p0 is None:
            p0 = self._get_default_seed(t, rate, rate_std)

        if rate_std is None:
            rate_std = np.ones_like(t)

        # t, rate and rate_std are fixed
        args = (t, rate, 1. / rate_std)
        res = sco.minimize(fun=self._lstsq, x0=p0, args=args, **kwargs)

        return res.x

    def _transform_trange_mjd(self, t, trange):
        """
        Transform time window to MJD

        Parameters
        ----------
        t : array-like
            MJD times of experimental data.
        trange : array-like, shape(2)
            Time window `[t0, t1]` in seconds around the source times t.

        Returns
        -------
        trange : array-like, shape(2, len(t))
            Reshaped time window `[[t0], [t1]]` in MJD for each time t.
        """
        t = np.atleast_1d(t)
        return t + np.array(trange).reshape(2, 1) / RateFunction._SECINDAY

    def _get_default_seed(self, t, rate, rate_std):
        """
        Default seed values derived from data under assumption that it behaves
        like the fit function.

        Parameters
        ----------
        t : array-like
            MJD times of experimental data.
        rate : array-like, shape (len(t))
            Rate per given time t in Hz.
        rate_std : array-like, shape(len(t)), optional
            Standard deviations for each datapoint. If None, all are set to 1.
            If `rate_std` is a good description of the standard deviation, then
            the fit statistics follows a chi2 distribution. (default: None)

        Returns
        -------
        p0 : tuple
            Seed values (a0, b0, c0, d0):
            - a0: (max(rate) + min(rate)) / 2, max amplitude of data bins
            - b0: 2pi / 365, as the usual seasonal variation is 1 year
            - c0: min(X), earliest time in X
            - d0: np.average(rate, weights=rate_std**2), average rate
        """
        # Assume weights are standard deviation
        rate_std = 1. / rate_std
        if rate_std is None:
            rate_std = np.ones_like(t)

        a0 = 0.5 * (np.amax(rate) - np.amin(rate))
        b0 = 2. * np.pi / 365.
        c0 = np.amin(t)
        d0 = np.average(rate, weights=rate_std**2)
        return [a0, b0, c0, d0]


class SinusRateFunction1yr(SinusRateFunction):
    """
    Sinus Rate Function fixing the period to 1 year.

    Function describes time dependent background rate. Function is a sinus with:

    .. math:: f(t|a,c,d) = a \sin((2\pi/\mathrm{365.25}) (t - c)) + d

    where:

    - a is the Amplitude in Hz
    - c is the x-offset in MJD
    - d the y-offset in Hz

    Time is measured in MJD days.
    """
    def __init__(self):
        self._b = 2 * np.pi / 365.25
        return

    @docs.dedent
    def fun(self, t, pars):
        """
        Returns the rate at a given time in MJD from a sinusodial function.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s
            Here `pars` are (a, c, d) as stated above.

        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        a, c, d = pars
        pars = (a, self._b, c, d)  # Just inject the fixed par in the super func
        return super(SinusRateFunction1yr, self).fun(t, pars)

    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Analytic integral of the sinusodial rate function in interval `trange`.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s
            Here `pars` are (a, c, d) as stated above.

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        a, c, d = pars
        pars = (a, self._b, c, d)  # Just inject the fixed par in the super func
        return super(SinusRateFunction1yr, self).integral(t, trange, pars)

    def _get_default_seed(self, t, rate, rate_std):
        """
        Default seed values derived from data under assumption that it behaves
        like the fit function.

        Parameters
        ----------
        t, rate, rate_std
            See `SinusRateFunction._get_default_seed`, Parameters
        Returns
        -------
        p0
            See `SinusRateFunction._get_default_seed`, Parameters.
            But `b` is fixed here, so only a0, c0 and d0 are seed values.
        """
        seed = super(SinusRateFunction1yr, self)._get_default_seed(t, rate,
                                                                   rate_std)
        # Drop b0 seed, as it's fixed to 1yr here
        return [seed[i] for i in [0, 2, 3]]
