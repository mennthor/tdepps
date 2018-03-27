# coding: utf-8

"""
Base class definitions are here, implementations in `model_toolkit.py`.
"""

from __future__ import print_function

import abc
import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state
from ..utils import logger, fill_dict_defaults
log = logger(name="toolkit", level="ALL")


##############################################################################
# Signal injector classes
##############################################################################
class SignalInjector(object):
    """
    Signal Injector base class
    """
    __metaclass__ = abc.ABCMeta

    _rndgen = None
    _srcs = None

    @property
    def rndgen(self):
        """ numpy RNG instance used to sample """
        return self._rndgen

    @rndgen.setter
    def rndgen(self, rndgen):
        self._rndgen = check_random_state(rndgen)

    @property
    def srcs(self):
        """ Source recarray the injector was fitted to """
        return self._srcs

    @abc.abstractmethod
    def mu2flux(self):
        """ Converts a number of injected events to a flux """
        pass

    @abc.abstractmethod
    def flux2mu(self):
        """ Converts a flux to injected number of events """
        pass

    @abc.abstractmethod
    def fit(self):
        """ Sets up the injector and makes it ready to use """
        pass

    @abc.abstractmethod
    def sample(self):
        """ Get a signal sample for a single trial to use in a LLH object """
        pass


class MultiSignalInjector(SignalInjector):
    """ Interface for managing multiple SignalInjector type classes """
    _names = None

    @property
    @abc.abstractmethod
    def names(self):
        """ Subinjector names, identifies this as a MultiModelInjector """
        pass


##############################################################################
# Time sampler
##############################################################################
class TimeSampler(object):
    """
    Class to collect time samplers used by SignalFluenceInjector
    """
    __metaclass__ = abc.ABCMeta

    _SECINDAY = 24. * 60. * 60
    _rndgen = None

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    @abc.abstractmethod
    def sample():
        pass


##############################################################################
# Background injector classes
##############################################################################
class BGDataInjector(object):
    """
    Background Data Injector Base Class

    Base class for generating events from a given data record array.
    Classes must implement methods:

    - ``fun``
    - ``sample``

    Class object then provides public methods:

    - ``fun``
    - ``sample``

    Parameters
    ----------
    random_state : None, int or np.random.RandomState, optional
        Turn seed into a ``np.random.RandomState`` instance. (default: None)

    Example
    -------
    >>> # Example for a special class which resamples directly from an array
    >>> from tdepps.model_toolkit import DataGPInjector as inj
    >>> # Generate some test data
    >>> n_evts, n_features = 100, 3
    >>> X = np.random.uniform(0, 1, size=(n_evts, n_features))
    >>> X = np.core.records.fromarrays(X.T, names=["logE", "dec", "sigma"])
    >>> # Fit injector and let it resample from the pool of testdata
    >>> inj.fit(X)
    >>> sample = inj.sample(n_samples=1000)
    """
    __metaclass__ = abc.ABCMeta

    _rndgen = None
    _X_names = None
    _n_features = None

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    @abc.abstractmethod
    def fit(self, X):
        """
        Build the injection model with the provided data.

        Parameters
        ----------
        X : record-array
            Data named array.
        """
        pass

    @abc.abstractmethod
    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (default: 1)

        Returns
        -------
        X : record-array
            Generated samples from the fitted model. Has the same names as the
            given record-array X in `fit`.
        """
        pass

    def _check_bounds(self, bounds):
        """
        Check if bounds are OK. Create numerical values when None is given.

        Returns
        -------
        bounds : array-like, shape (n_features, 2)
            Boundary conditions for each dimension. Unconstrained axes have
            bounds ``[-np.inf, +np.inf]``.
        """
        if bounds is None:
            bounds = np.repeat([[-np.inf, np.inf], ],
                               repeats=self._n_features, axis=0)

        bounds = np.array(bounds)
        if bounds.shape[1] != 2 or (bounds.shape[0] != len(self._X_names)):
            raise ValueError("Invalid `bounds`. Must be shape (n_features, 2).")

        # Convert None to +-np.inf depnding on low/hig bound
        bounds[:, 0][bounds[:, 0] == np.array(None)] = -np.inf
        bounds[:, 1][bounds[:, 1] == np.array(None)] = +np.inf

        return bounds

    def _check_X_names(self, X):
        """ Check if given input ``X`` is valid and extract names. """
        try:
            _X_names = X.dtype.names
        except AttributeError:
            raise AttributeError("`X` must be a record array with dtype.names.")

        self._n_features = len(_X_names)
        self._X_names = _X_names

        return X


##############################################################################
# Rate function classes to fit a BG rate model
##############################################################################
class RateFunction(object):
    """
    Rate Function Base Class

    Base class for rate functions describing time dependent background
    rates. Rate function must be interpretable as a PDF and must not be
    negative.

    Classes must implement methods:

    - ``fun``
    - ``integral``
    - ``sample``
    - ``_get_default_seed``

    Class object then provides public methods:

    - ``fun``
    - ``integral``
    - ``fit``
    - ``sample``

    Parameters
    ----------
    random_state : seed, optional
        Turn seed into a ``np.random.RandomState`` instance. See
        ``sklearn.utils.check_random_state``. (default: None)
    """
    __metaclass__ = abc.ABCMeta
    _SECINDAY = 24. * 60. * 60.

    def __init__(self, random_state=None):
        self.rndgen = random_state
        # Get set when fitted
        self._bf_pars = None
        self._bf_fun = None
        self._bf_int = None

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    @property
    def bf_pars(self):
        return self._bf_pars

    @property
    def bf_fun(self):
        return self._bf_fun

    @property
    def bf_int(self):
        return self._bf_int

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
            Rate in Hz for each time ``t``.
        """
        pass

    @abc.abstractmethod
    def integral(self, t, trange, pars):
        """
        Integral of rate function in intervals trange around source times t.

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows ``[[t0, t1], ...]`` in seconds around each time ``t``.
        pars : tuple
            Further parameters :py:meth:`fun` depends on.

        Returns
        -------
        integral : array-like, shape (nsrcs)
            Integral of :py:meth:`fun` within given time windows ``trange``.
        """
        pass

    @abc.abstractmethod
    def sample(self, n_samples, t, trange, pars):
        """
        Generate random samples from the rate function for multiple source times
        and time windows.

        Parameters
        ----------
        n_samples : array-like, shape (nsrcs)
            Number of events to sample per source.
        t : array-like, shape (nsrcs)
            MJD times of sources to sample around.
        trange : array-like, shape(nsrcs, 2)
            Time windows ``[[t0, t1], ...]`` in seconds around each time ``t``.
        pars : tuple
            Parameters :py:meth:`fun` depends on.

        Returns
        -------
        times : list of arrays, len (nsrcs)
            Sampled times in MJD of background events per source. If
            ``n_samples`` is 0 for a source, an empty array is placed at that
            position.
        """
        pass

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
            Seed values for each parameter the specific :py:class:`RateFunction`
            uses as a staerting point in the :py:meth:`fit`.
        """
        pass

    def fit(self, t, rate, p0=None, w=None, **minopts):
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
        minopts : dict, optional
            Minimizer options passed to
            ``scipy.optimize.minimize(method='L-BFGS-B')``. Default settings if
            given as ``None`` or for missing keys are
            ``{'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': int(1e3)}``.

        Returns
        -------
        res : scipy.optimize.OptimizeResult
            Dict wrapper with fot results.
        """
        if w is None:
            w = np.ones_like(rate)

        if p0 is None:
            p0 = self._get_default_seed(t, rate, w)

        # Setup minimizer options
        required_keys = []
        opt_keys = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": int(1e3)}
        minopts["options"] = fill_dict_defaults(
            minopts.get("options", None), required_keys, opt_keys)

        # Scale times to be in range [0, 1] for a proper fit
        bounds = minopts.pop("bounds", None)
        # t, p0, bounds, min_t, max_t = self._scale(t, p0, bounds)
        res = sco.minimize(fun=self._lstsq, x0=p0, args=(t, rate, w),
                           method="L-BFGS-B", bounds=bounds, **minopts)
        # Re-scale fit result back to original scale
        # res = self._rescale(res, min_t, max_t)

        self._bf_fun = (lambda t: self.fun(t, res.x))
        self._bf_int = (lambda t, trange: self.integral(t, trange, res.x))
        self._bf_pars = res.x
        return res

    def _lstsq(self, pars, *args):
        """
        Weighted leastsquares loss: :math:`\sum_i (w_i * (y_i - f_i))^2`

        Parameters
        ----------
        pars : tuple
            Fitparameter for :py:meth:`fun` that gets fitted.
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
