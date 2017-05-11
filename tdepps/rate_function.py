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
            Parameters for the fit function. If not None the local parameters
            are prefered over the global ones. (default: None)

        Returns
        -------
        rate : array-like
            Rate in Hz for each time t.
        """
        raise NotImplementedError("RateFunction is an interface.")

    @docs.get_sectionsf("RateFunction.integral",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    def integral(self, t, pars, trange):
        """
        Integral of rate function at a given time t in MJD.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s
        trange : [float, float] or array-like, shape (len(t), 2)
            Time window(s) in seconds relativ to the given time(s) t.
            - If [float, float], the same window [lower, upper] is used for
              every time t.
            - If array-like [lower, upper] bounds of the time
              window per source in each row.

        Returns
        -------
        integral : float
            Integral over fun integrated over t in trange.
        """
        raise NotImplementedError("RateFunction is an interface.")

    @docs.get_sectionsf("RateFunction.sample",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    def sample(self, t, trange, n_samples=1, random_state=None):
        """
        Generate random samples from the rate function for one time and a single
        time window.

        Parameters
        ----------
        t : float
            Time of the occurance of the source event in MJD.
        trange : [float, float]
            Time window in seconds relativ to the given time t.
        nsamples : int
            Number of events to sample. (default: 1)
        random_state : RandomState, optional
            A random number generator instance. (default: None)

        Returns
        -------
        times : list, length len(t)
            List of samples times in MJD of background events per source.
            For each source i nsamples[i] times are drawn from the rate function.
        """
        raise NotImplementedError("RateFunction is an interface.")

    @docs.get_sectionsf("RateFunction.integral",
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
        if p0 is None:
            p0 = self._get_default_seed(t, rate, rate_std)

        if rate_std is None:
            rate_std = np.ones_like(t)

        # t, rate and rate_std are fixed
        args = (t, rate, 1. / rate_std)
        res = sco.minimize(fun=self._lstsq, x0=p0, args=args, **kwargs)

        self.best_pars = res.x
        self.best_fun = (lambda t: self.fun(t, self.best_pars))
        self.best_integral = (lambda t, trange:
                              self.integral(t, self.best_pars, trange))

        return res.x

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

    ..math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

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
        Returns the rate at a given time in MJD from a sin function.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s

        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        a, b, c, d = pars
        return a * np.sin(b * (t - c)) + d

    @docs.dedent
    def integral(self, t, pars, trange):
        """
        Analytic integral of rate function for rejection sampling norm.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        a, b, c, d = pars
        t, trange = self._prep_times(t, trange)

        # Reshape to handle all at once
        t0 = np.atleast_2d(trange[:, 0]).reshape(len(trange), 1)
        t1 = np.atleast_2d(trange[:, 1]).reshape(len(trange), 1)

        # Split analytic expression for convenience
        per = a / b * (np.cos(b * (t0 - c)) - np.cos(b * (t1 - c)))
        lin = d * (t1 - t0)

        # Match units with secinday = 24 * 60 * 60 s/MJD = 86400 / (Hz*MJD)
        #     [a], [d] = Hz, [b], [c], [ti] = MJD
        #     [a / b] = Hz * MJD, [d * (t1 - t0)] = HZ * MJD
        return (per + lin) * RateFunction._SECINDAY

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

    def _prep_times(self, t, trange):
        """
        Transform times to correct shape and convert to MJD.

        Parameters
        ----------
        t : array-like, shape (n_samples)
            MJD times of experimental data.

        trange : [float, float] or array-like, shape (len(t), 2)
            Time window(s) in seconds relativ to the given time(s) t.
            - If [float, float], the same window [lower, upper] is used for
              every time t.
            - If array-like [lower, upper] bounds of the time
              window per source in each row.

        Returns
        -------
        t : array-like, shape (len(t), 1)
            Reshaped times.
        trange : array-like, shape (len(t), 2)
            One time window in MJD per time t.
        """
        t = np.atleast_1d(t)
        trange = np.array(trange)
        nsrc = len(t)

        # Make shape (nsources, 1) for the times
        t = t.reshape(nsrc, 1)

        # If range is 1D (one for all) reshape it to (nsources, 2)
        if len(trange.shape) == 1:
            trange = np.repeat(trange.reshape(1, 2), repeats=nsrc, axis=0)

        # Prepare time window in MJD
        trange = t + trange / RateFunction._SECINDAY

        return t, trange
