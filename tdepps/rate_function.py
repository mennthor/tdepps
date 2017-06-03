import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state

from .utils import rejection_sampling

import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class RateFunction(object):
    """
    Interface for rate functions describing time dependent background rates.

    Classes must implement methods:
    ["fun", "integral", "sample", "_get_default_seed"].

    Class then provides methos:
    ["fun", "integral", "sample", "fit"]

    Rate function must be interpretable as a PDF despite a normalization and
    thus must not be negative.
    """
    # Set up globals for shared inherited constants
    _SECINDAY = 24. * 60. * 60.

    def __init__(self):
        self._DESCRIBES = ["fun", "integral", "sample", "_get_default_seed"]
        raise NotImplementedError("Interface only. Describes functions: ",
                                  self._DESCRIBES)
        return

    @docs.get_sectionsf("RateFunction.fun", sections=["Parameters", "Returns"])
    @docs.dedent
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
        t : array-like, shape (nsrcs)
            MJD times of sources around which the time ranges get centerd.
        trange : array-like, shape(nsrcs, 2)
            Time windows [[t0, t1], ...] in seconds around each given time t.
        pars : tuple
            Further parameters `self.fun` depends on.

        Returns
        -------
        integral : array-like, shape (nsrcs)
            Integral of `self.fun` within given time windows `trange`.
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
            Time window [t0, t1] in seconds around the source time t.
        pars : tuple
            Further parameters `self.fun` depends on.
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
        p0 : tuple, optional
            Seed values for the fit parameters. If None, default ones are used,
            that may or may not work. (default: None)
        rate_std : array-like, shape(len(t)), optional
            Standard deviations for each datapoint. If None, all are set to 1.
            If `rate_std` is a good description of the standard deviation, then
            the fit statistics follows a chi2 distribution. But for a binned
            fit this makes less sense, becuase low stddec means low statistics,
            so better use unweighted. (default: None)
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

    @docs.get_summaryf("RateFunction._get_default_seed")
    @docs.get_sectionsf("RateFunction._get_default_seed",
                        sections=["Parameters"])
    @docs.dedent
    def _get_default_seed(self, t, trange, rate_std):
        """
        Default seed values for the specifiv RateFunction fit.

        Parameters
        ----------
        t, rate, rate_std
            See `RateFunction.fit`, Parameters

        Returns
        -------
        p0 : tuple
            Seed values for each parameter the specific RateFunction uses.
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
            Fixed values for the loss function: (t, rate, rate_std)

            - t, rate, rate_std : See `RateFunction.fit`, Parameters

            The weights are wi = (1 / rate_std_i) which is equivalent to the
            weighted mean, when a constant is fitted.

        Returns
        -------
        loss : float
            The weighted least squares loss for the given pars and args.
        """
        t, rate, rate_std = args
        w = 1 / rate_std
        fun = self.fun(t, pars)
        return np.sum((w * (rate - fun))**2)

    def _transform_trange_mjd(self, t, trange):
        """
        Transform time window to MJD

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources around which the time ranges get centerd.
        trange : array-like, shape(nsrcs, 2)
            Time windows [[t0, t1], ...] in seconds around each given time t.

        Returns
        -------
        trange : array-like, shape(nsrcs, 2)
            Time windows [[t0, t1], ...] in MJD around each given time t.
        """
        t = np.atleast_1d(t)
        nsrcs = len(t)
        t = t.reshape(nsrcs, 1)
        trange = np.atleast_2d(trange).reshape(nsrcs, 2)
        return t + np.array(trange) / RateFunction._SECINDAY


class SinusRateFunction(RateFunction):
    """
    Sinus Rate Function

    Function describes time dependent background rate. Function is a sinus with:

    .. math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

    Parameters are commonly used in methods and not given at object creation.
    Times t are used in MJD days.

    Parameters
    ----------
    a : float
        Amplitude in Hz.
    b : float
        Angular frequency, omega = 2*pi/T` with period T given in 1/MJD.
    c : float
        x-axis offset in MJD.
    d : float
        y-axis offset in Hz

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
            pars : See `SinusRateFunction`, Parameters

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
            pars : See `SinusRateFunction`, Parameters

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
            pars : See `SinusRateFunction`, Parameters

        Returns
        -------
        %(RateFunction.sample.returns)s
        """
        rndgen = check_random_state(random_state)

        # rejection_sampling needs bounds in shape (1, 2)
        trange = np.atleast_2d(self._transform_trange_mjd(t, trange))

        def sample_fun(t):
            return self.fun(t, pars)

        times = rejection_sampling(sample_fun, bounds=trange, n=n_samples,
                                   random_state=rndgen)

        return times

    @docs.dedent
    def _get_default_seed(self, t, rate, rate_std):
        """
        %(RateFunction._get_default_seed.summary)s

        Motivation for default seed:

        - a0 : Using the maximum amplitude in variation of rate bins.
        - b0 : The expected seasonal variation is 1 year.
        - c0 : Earliest time in t.
        - d0 : Weighted averaged rate, which is the best fit value for a
           constant target function.

        Parameters
        ----------
        %(RateFunction._get_default_seed.parameters)s

        Returns
        -------
        p0 : tuple, shape (4)
            Seed values (a0, b0, c0, d0):

            - a0 : (max(rate) + min(rate)) / 2
            - b0 : 2pi / 365
            - c0 : min(t)
            - d0 : np.average(rate, weights=rate_std**2)
        """
        a0 = 0.5 * (np.amax(rate) - np.amin(rate))
        b0 = 2. * np.pi / 365.
        c0 = np.amin(t)
        d0 = np.average(rate, rate_std=rate_std**2)
        return (a0, b0, c0, d0)


class Sinus1yrRateFunction(SinusRateFunction):
    """
    Sinus Rate Function fixing the period to 1 year.

    Function describes time dependent background rate. Function is a sinus with:

    .. math:: f(t|a,c,d) = a \sin((2\pi/\mathrm{365.25}) (t - c)) + d

    where:

    - a : Amplitude in Hz
    - c : x-offset in MJD
    - d : y-offset in Hz

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
            pars : See `Sinus1yrRateFunction`, Parameters

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
        Analytic integral of the sinusodial rate function in interval `trange`.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s
            pars : See `Sinus1yrRateFunction`, Parameters

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
        - c0 : Earliest time in t.
        - d0 : Weighted averaged rate, which is the best fit value for a
           constant target function.

        Parameters
        ----------
        t, rate, rate_std
            See `SinusRateFunction._get_default_seed`, Parameters

        Returns
        -------
        p0 : tuple, shape (3)
            See `SinusRateFunction._get_default_seed`, Parameters.
            But b is fixed here, so only a0, c0 and d0 are seed values.
        """
        seed = super(Sinus1yrRateFunction, self)._get_default_seed(t, rate,
                                                                   rate_std)
        # Drop b0 seed, as it's fixed to 1yr here
        return tuple(seed[i] for i in [0, 2, 3])


class ConstantRateFunction(RateFunction):
    """
    Returns a constant rate at a given time in MJD.

    Parameters
    ----------
    rate : float
        Constant rate in Hz.
    """
    def __init__(self):
        return

    @docs.dedent
    def fun(self, t, pars):
        """
        Returns a constant rate Hz at any given time in MJD by simply
        broadcasting the input rate.

        Parameters
        ----------
        %(RateFunction.fun.parameters)s
            pars : See `ConstantRateFunction`, Parameters

        Returns
        -------
        %(RateFunction.fun.returns)s
        """
        rate = pars
        return np.ones_like(t) * rate

    @docs.dedent
    def integral(self, t, trange, pars):
        """
        Analytic integral of the rate function in interval trange.

        As rate function is constant, integral is simply trange * rate.

        Parameters
        ----------
        %(RateFunction.integral.parameters)s
            pars : See `ConstantRateFunction`, Parameters

        Returns
        -------
        %(RateFunction.integral.returns)s
        """
        trange = self._transform_trange_mjd(t, trange)
        return (np.diff(trange, axis=1) * RateFunction._SECINDAY *
                self.fun(t, pars))

    @docs.dedent
    def sample(self, t, trange, pars=None, n_samples=1, random_state=None):
        """
        %(RateFunction.sample.summary)s

        Parameters
        ----------
        %(RateFunction.sample.parameters)s

        Returns
        -------
        %(RateFunction.sample.returns)s
        """
        # Just sample uniformly in MJD time window
        rndgen = check_random_state(random_state)
        trange = self._transform_trange_mjd(t, trange)
        return rndgen.uniform(trange[0], trange[1], size=n_samples)

    @docs.dedent
    def _get_default_seed(self, t, rate, rate_std):
        """
        %(RateFunction._get_default_seed.summary)s

        Motivation for default seed:

        - rate0 : Mean of the given rates, this is the anlytic solution to the
          fit, so we seed with the best fit.

        Parameters
        ----------
        %(RateFunction._get_default_seed.parameters)s

        Returns
        -------
        p0 : tuple, shape(1)
            Seed values (rate0):

            - rate0 : mean(rate)
        """
        return np.mean(rate)
