# coding: utf-8

"""
This is a collection of different function and / or classes that can be used in
a modular way to create a PDF and injection model class which can be used in the
LLH analysis.
When creating a new function it should only work on public interfaces and data
provided by the model class. If this can't be realized due to performance /
caching reasons, then consider coding the functionality directly into a model to
benefit from direct class attributes.
"""

from __future__ import print_function, division, absolute_import
from builtins import zip, super
from future import standard_library
standard_library.install_aliases()

import abc
import numpy as np
import scipy.optimize as sco
from sklearn.utils import check_random_state

from awkde import GaussianKDE as KDE

from .utils import rejection_sampling, random_choice, fill_dict_defaults


##############################################################################
# Background injector methods
##############################################################################
class GeneralPurposeInjector(object):
    """
    General Purpose Injector Base Class

    Base class for generating events from a given record array.
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

    def __init__(self, random_state=None):
        self.rndgen = random_state
        # Setup private defaults
        self._X_names = None
        self._n_features = None

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


class KDEGeneralPurposeInjector(GeneralPurposeInjector):
    """
    Adaptive Bandwidth Kernel Density Background Injector.

    Parameters
    ----------
    kde : awkde.GaussianKDE
        Adaptive width KDE model. If an already fitted model is given,
        th ``fit`` step can be called with ``X=None`` to avoid refitting,
        which can take some time when many points with adaptive kernels are
        used.
    random_state : None, int or np.random.RandomState, optional
        Turn seed into a ``np.random.RandomState`` instance. (default: None)
    """
    def __init__(self, kde, random_state=None):
        super(KDEGeneralPurposeInjector, self).__init__(random_state)
        self.kde_model = kde

    @property
    def kde_model(self):
        return self._kde_model

    @kde_model.setter
    def kde_model(self, kde_model):
        if not isinstance(kde_model, KDE):
            raise TypeError("`kde_model` must be an instance of " +
                            "`awkde.GaussianKDE`")
        self._kde_model = kde_model

    # PUBLIC
    def fit(self, X, bounds=None):
        """
        Fit a KDE model to the given data.

        Parameters
        ----------
        X : record-array or list
            Data named array. If list it is checked if the given KDE model is
            already fitted and usable and fits to the given names. This can be
            used to reuse an already existing KDE model for injection. Be
            careful that the fitted model has the data stored in the same order
            as in the list ``X``.
        bounds : None or array-like, shape (n_features, 2)
            Boundary conditions for each dimension. If ``None``,
            ``[-np.inf, +np.inf]`` is used in each dimension.
            (default: ``None``)
        """
        # TODO: Use advanced bounds via mirror method in KDE class.
        # Currently bounds are used to resample events that fall outside
        if hasattr(X, "__iter__"):
            if self._kde_model._std_X is None:
                raise ValueError("Given KDE model is not ready to use and " +
                                 "must be fitted first. Give an explicit " +
                                 "to fit the model.")
            if len(X) != self._kde_model._std_X.shape[1]:
                raise ValueError("Given names `X` do not have the same " +
                                 "dimension as the given KDE instance.")
            if not all(map(lambda s: isinstance(s, str), X)):
                raise TypeError("`X` is not a list of string names.")
            self._n_features = len(X)
            self._X_names = np.array(X)
        elif isinstance(X, np.ndarray):
            X = self._check_X_names(X)
            # Turn record-array in normal 2D array for more general KDE class
            X = np.vstack((X[n] for n in self._X_names)).T
            self._kde_model.fit(X)
        else:
            raise ValueError("`X` is neither None  nor a record array to " +
                             "fit a the given KDE model to.")

        assert (self._n_features == self._kde_model._std_X.shape[1])
        self._bounds = self._check_bounds(bounds)

    def sample(self, n_samples=1):
        """ Sample from a KDE model that has been build on given data. """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty recarray with all keys, when n_samples < 1
        dtype = [(n, float) for n, f in self._X_names]
        if n_samples < 1:
            return np.empty(0, dtype=dtype)

        # Resample until all sample points are inside bounds
        X = []
        bounds = self._bounds
        while n_samples > 0:
            gen = self._kde_model.sample(n_samples, self._rndgen)
            accepted = np.all(np.logical_and(gen >= bounds[:, 0],
                                             gen <= bounds[:, 1]), axis=1)
            n_samples = np.sum(~accepted)
            # Append accepted to final sample
            X.append(gen[accepted])

        # Concat sampled array list and convert to single record-array
        return np.core.records.fromarrays(np.concatenate(X).T, dtype=dtype)


class DataGeneralPurposeInjector(GeneralPurposeInjector):
    """
    Data injector selecting random data events from the given sample.
    """
    def __init__(self, random_state=None):
        super(DataGeneralPurposeInjector, self).__init__(random_state)

    def fit(self, X, weights=None):
        """
        Build the injection model with the provided data. Here the model is
        simply the data itself.

        Parameters
        ----------
        X : record-array
            Data named array.
        weights : array-like, shape(len(X)), optional
            Weights used to sample from ``X``. If ``None`` all weights are
            equal. (default: ``None``)
        """
        self._X = self._check_X_names(X)
        nevts = len(self._X)
        if weights is None:
            weights = np.ones(nevts, dtype=float) / float(nevts)
        elif len(weights) != nevts:
            raise ValueError("'weights' must have same length as `X`.")
        # Normalize sampling weights and create sampling CDF
        CDF = np.cumsum(weights)
        self._CDF = CDF / CDF[-1]

    def sample(self, n_samples=1):
        """
        Sample by choosing random events from the given data.
        """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty array with all keys, when n_samples < 1
        dtype = [(n, float) for n, f in self._X_names]
        if n_samples < 1:
            return np.empty(0, dtype=dtype)

        # Choose uniformly from given data
        idx = random_choice(rndgen=self._rndgen, CDF=self._CDF, n=n_samples)
        return self._X[idx]


class Binned3DKDEGeneralPurposeInjector(GeneralPurposeInjector):
    """
    Injector binning up data space in `[a x b x c]` bins with equal statistics
    and then sampling uniformly from those bins. Only works with 3 dimensional
    data arrays.
    """
    def __init__(self, random_state=None):
        super(Binned3DKDEGeneralPurposeInjector, self).__init__(random_state)

    def fit(self, X, nbins=10, minmax=False):
        """
        Build the injection model with the provided data, dimension fixed to 3.

        Parameters
        ----------
        X : record-array
            Experimental data named array.
        nbins : int or array-like, shape(n_features), optional

            - If int, same number of bins is used for all dimensions.
            - If array-like, number of bins for each dimension is used.

            (default: 10)

        minmax : bool or array-like, optional
            Defines the outermost bin edges for the 2nd and 3rd feature:

                - If False, use the min/max values in the current 1st (2nd)
                  feature bin.
                - If True, use the global min/max values per feature.
                - If array-like: Use the given edges as global min/max values
                  per feature. Must then have shape (3, 2):
                  ``[[min1, max1], [min2, max2], [min3, max3]]``.

            (default: False)

        Returns
        -------
        ax0_bins : array-like, shape (nbins[0] + 1)
            The bin borders for the first dimension.
        ax1_bins : array-like, shape (nbins[0], nbins[1] + 1)
            The bin borders for the second dimension.
        ax2_bins : array-like, shape (nbins[0], nbins[1], nbins[2] + 1)
            The bin borders for the third dimension.
        """
        def bin_equal_stats(data, nbins, minmax=None):
            """
            Bin with nbins of equal statistics by using percentiles.

            Parameters
            ----------
            data : array-like, shape(n_samples)
                The data to bin.
            nbins : int
                How many bins to create, must be smaller than len(data).
            minmax : array-like, shape (2), optional
                If [min, max] these values are used for the outer bin edges. If
                None, the min/max of the given data is used. (default: None)

            Returns
            -------
            bins : array-like
                (nbins + 1) bin edges for the given data.
            """
            if nbins > len(data):
                raise ValueError("Cannot create more bins than datapoints.")
            nbins += 1  # We need 1 more edge than bins
            if minmax is not None:
                # Use global min/max for outermost bin edges
                bins = np.percentile(data, np.linspace(0, 100, nbins)[1:-1])
                return np.hstack((minmax[0], bins, minmax[1]))
            else:
                # Else just use the bounds from the given data
                return np.percentile(data, np.linspace(0, 100, nbins))

        # Turn record-array in normal 2D array as it is easier to handle here
        X = self._check_X_names(X)
        if self._n_features != 3:
            raise ValueError("Only 3 dimensions supported here.")
        X = np.vstack((X[n] for n in self._X_names)).T

        # Repeat bins, if only int was given
        nbins = np.atleast_1d(nbins)
        if (len(nbins) == 1) and (len(nbins) != self._n_features):
            nbins = np.repeat(nbins, repeats=self._n_features)
        elif len(nbins) != self._n_features:
            raise ValueError("Given 'nbins' does not match dim of data.")
        self._nbins = nbins

        # Get bounding box, we sample the maximum distance in each direction
        if minmax is True:
            minmax = np.vstack((np.amin(X, axis=0), np.amax(X, axis=0))).T
        elif isinstance(minmax, np.ndarray):
            if minmax.shape != (self._n_features, 2):
                raise ValueError("'minmax' must have shape (3, 2) if edges " +
                                 "are given explicitely.")
        else:
            minmax = self._n_features * [None]

        # First axis is the main binning and only an 1D array
        ax0_dat = X[:, 0]
        ax0_bins = bin_equal_stats(ax0_dat, nbins[0], minmax[0])

        # 2nd axis array has bins[1] bins per bin in ax0_bins, so it's 2D
        ax1_bins = np.zeros((nbins[0], nbins[1] + 1))
        # 3rd axis is 3D: nbins[2] bins per bin in ax0_bins and ax1_bins
        ax2_bins = np.zeros((nbins[0], nbins[0], nbins[2] + 1))

        # Fill bins by looping over all possible combinations
        for i in range(nbins[0]):
            # Bin left inclusive, except last bin
            m = (ax0_dat >= ax0_bins[i]) & (ax0_dat < ax0_bins[i + 1])
            if (i == nbins[1] - 1):
                m = (ax0_dat >= ax0_bins[i]) & (ax0_dat <= ax0_bins[i + 1])

            # Bin ax1 subset of data in current ax0 bin
            _X = X[m]
            ax1_dat = _X[:, 1]
            ax1_bins[i] = bin_equal_stats(ax1_dat, nbins[1], minmax[1])

            # Directly proceed to axis 2 and repeat procedure
            for k in range(nbins[1]):
                m = ((ax1_dat >= ax1_bins[i, k]) &
                     (ax1_dat < ax1_bins[i, k + 1]))
                if (k == nbins[2] - 1):
                    m = ((ax1_dat >= ax1_bins[i, k]) &
                         (ax1_dat <= ax1_bins[i, k + 1]))

                # Bin ax2 subset of data in current ax0 & ax1 bin
                ax2_bins[i, k] = bin_equal_stats(_X[m][:, 2],
                                                 nbins[2], minmax[2])

        self._ax0_bins = ax0_bins
        self._ax1_bins = ax1_bins
        self._ax2_bins = ax2_bins
        return ax0_bins, ax1_bins, ax2_bins

    def sample(self, n_samples=1):
        """
        Sample pseudo events uniformly from each bin.
        """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty array with all keys, when n_samples < 1
        dtype = [(n, float) for n, f in self._X_names]
        if n_samples < 1:
            return np.empty(0, dtype=dtype)

        # Sample indices to select from which bin is injected
        ax0_idx = self._rndgen.randint(0, self._nbins[0], size=n_samples)
        ax1_idx = self._rndgen.randint(0, self._nbins[1], size=n_samples)
        ax2_idx = self._rndgen.randint(0, self._nbins[2], size=n_samples)

        # Sample uniform in [0, 1] to decide where each point lies in the bins
        r = self._rndgen.uniform(0, 1, size=(n_samples, self._n_features))

        # Get edges of each bin
        ax0_edges = np.vstack((self._ax0_bins[ax0_idx],
                               self._ax0_bins[ax0_idx + 1])).T

        ax1_edges = np.vstack((self._ax1_bins[ax0_idx, ax1_idx],
                               self._ax1_bins[ax0_idx, ax1_idx + 1])).T

        ax2_edges = np.vstack((self._ax2_bins[ax0_idx, ax1_idx, ax2_idx],
                               self._ax2_bins[ax0_idx, ax1_idx, ax2_idx + 1])).T

        # Sample uniformly between selected bin edges
        ax0_pts = ax0_edges[:, 0] + r[:, 0] * np.diff(ax0_edges, axis=1).T
        ax1_pts = ax1_edges[:, 0] + r[:, 1] * np.diff(ax1_edges, axis=1).T
        ax2_pts = ax2_edges[:, 0] + r[:, 2] * np.diff(ax2_edges, axis=1).T

        # Combine and convert to record-array
        return np.core.records.fromarrays(np.vstack((ax0_pts, ax1_pts,
                                                     ax2_pts)).T, dtype=dtype)


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

    def fit(self, t, rate, p0=None, w=None, minopts=None):
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
        bf_pars : array-like
            Values of the best fit parameters.
        """
        if w is None:
            w = np.ones_like(rate)

        if p0 is None:
            p0 = self._get_default_seed(t, rate, w)

        # Setup minimizer options
        required_keys = []
        opt_keys = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": int(1e3)}
        minopts = fill_dict_defaults(minopts, required_keys, opt_keys)

        res = sco.minimize(fun=self._lstsq, x0=p0, args=(t, rate, w),
                           method="L-BFGS-B", options=minopts)
        self._bf_fun = (lambda t: self.fun(t, res.x))
        self._bf_int = (lambda t, trange: self.integral(t, trange, res.x))
        self._bf_pars = res.x
        return res.x

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


class SinusRateFunction(RateFunction):
    """
    Sinus Rate Function

    Describes time dependent background rate. Used function is a sinus with:

    .. math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

    depending on 4 parameters:

    - a, float: Amplitude in Hz.
    - b, float: Angular frequency, ``b = 2*pi / T`` with period ``T`` given in
                1 / (MJD days).
    - c, float: x-axis offset in MJD.
    - d, float: y-axis offset in Hz
    """
    # Just to have some info encoded in the class which params we have
    _PARAMS = ["amplitude", "period", "toff", "baseline"]

    def __init__(self, random_state=None):
        super(SinusRateFunction, self).__init__(random_state)
        # Cached in `fit` for faster rejection sampling
        self._fmax = None
        self._trange = None

    def fit(self, t, rate, srcs, p0=None, w=None, minopts=None):
        """
        Fit the rate model to discrete points ``(t, rate)``. Cache source values
        for fast sampling.

        Parameters
        ----------
        srcs : record-array
            Must have names ``'t', 'dt0', 'dt1'`` describing the time intervals
            around the source times to sample from.
        """
        bf_pars = super(SinusRateFunction, self).fit(t, rate, p0, w, minopts)

        # Cache max function values in the fixed source intervals for sampling
        required_names = ["t", "dt0", "dt1"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` recarray is missing name " +
                                 "'{}'.".format(n))
        _dts = np.vstack((srcs["dt0"], srcs["dt1"])).T
        _, _trange = self._transform_trange_mjd(srcs["t"], _dts)
        self._fmax = self._calc_fmax(_trange[:, 0], _trange[:, 1], pars=bf_pars)
        self._trange = _trange

        return bf_pars

    def fun(self, t, pars):
        """ Full 4 parameter sinus rate function """
        a, b, c, d = pars
        return a * np.sin(b * (t - c)) + d

    def integral(self, t, trange, pars):
        """ Analytic integral, full 4 parameter sinus """
        a, b, c, d = pars

        # Transform time windows to MJD
        _, dts = self._transform_trange_mjd(t, trange)
        t0, t1 = dts[:, 0], dts[:, 1]

        per = a / b * (np.cos(b * (t0 - c)) - np.cos(b * (t1 - c)))
        lin = d * (t1 - t0)

        # Match units with secinday = 24 * 60 * 60 s/MJD = 86400 / (Hz*MJD)
        #     [a], [d] = Hz;M [b] = 1/MJD; [c], [t] = MJD
        #     [a / b] = Hz * MJD; [d * (t1 - t0)] = HZ * MJD
        return (per + lin) * self._SECINDAY

    def sample(self, n_samples):
        """
        Rejection sample from the fitted sinus function

        Parameters
        ----------
        n_samples : array-like
            How many events to sample per source. Length must match length of
            cached source positions if any.
        """
        n_samples = np.atleast_1d(n_samples)
        if self._bf_pars is None:
            raise RuntimeError("Rate function was not fit yet.")
        if len(n_samples) != len(self._fmax):
            raise ValueError("Requested to sample a different number of " +
                             "sources than have been fit")

        # Just loop over all intervals and rejection sample the src regions
        for i, (bound, nsam) in enumerate(zip(self._trange, n_samples)):
            # Draw remaining events until all samples per source are created
            sample = []
            while nsam > 0:
                t = self._rndgen.uniform(bound[0], bound[1], size=nsam)
                y = self._fmax[i] * self._rndgen.uniform(0, 1, size=nsam)

                accepted = (y <= self._bf_fun(t))
                sample += t[accepted].tolist()
                nsam = np.sum(~accepted)  # Number of remaining samples to draw

            sample.append(np.array(sample))

        return sample

    def _get_default_seed(self, t, rate, w):
        """
        Default seed values for the specifiv RateFunction fit.

        Motivation for default seed:

        - a0 : Using the width of the central 50\% percentile of the rate
               distribtion (for rates > 0). The sign is determined based on
               wether the average rates in the first two octants based on the
               period seed decrease or increase.
        - b0 : The expected seasonal variation is 1 year.
        - c0 : Earliest time in ``t``.
        - d0 : Weighted averaged rate, which is the best fit value for a
               constant target function.

        Returns
        -------
        p0 : tuple, shape (4)
            Seed values `(a0, b0, c0, d0)`:

            - a0 : ``-np.diff(np.percentile(rate[w > 0], q=[0.25, 0.75])) / 2``
            - b0 : ``2 * pi / 365``
            - c0 : ``np.amin(t)``
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

    def _calc_fmax(self, t0, t1, pars):
        """
        Get the analytic maximum function value in interval ``[t0, t1]`` cached
        for rejection sampling.
        """
        a, b, c, d = pars
        L = 2. * np.pi / b  # Period length
        # If we start with a negative sine, then the first max is after 3/4 L
        if np.sign(a) == 1:
            step = 1.
        else:
            step = 3.
        # Get dist to first max > c and count how many periods k it is away
        k = np.ceil((t0 - (c + step * L / 4.)) / L)
        # Get the closest next maximum to t0 with t0 <= tmax_k
        tmax_gg_t0 = L / 4. * (step + 4. * k) + c

        # If the next max is <= t1, then fmax must be the global max, else the
        # highest border
        fmax = np.zeros_like(t0) + (np.abs(a) + d)
        m = (tmax_gg_t0 > t1)
        fmax[m] = np.maximum(self.fun(t0[m], pars), self.fun(t1[m], pars))

        return fmax


class SinusFixedRateFunction(SinusRateFunction):
    """
    Same a sinus Rate Function but period and time offset can be fixed for the
    fit and stays constant.
    """
    def __init__(self, p_fix=None, t0_fix=None, random_state=None):
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

    def fun(self, t, pars):
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).fun(t, pars)

    def integral(self, t, trange, pars):
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).integral(t, trange, pars)

    def _get_default_seed(self, t, rate, w):
        """ Same default seeds as SinusRateFunction, but drop fixed params. """
        seed = super(SinusFixedRateFunction,
                     self)._get_default_seed(t, rate, w)
        # Drop b0 and / or c0 seed, when it's marked as fixed
        return tuple(seed[i] for i in self._fit_idx)

    def _make_params(self, pars):
        """
        Check which parameters are fixed and insert them where needed to build
        a full parameter set.

        Returns
        -------
        pars : tuple
            Fixed parameters inserted in the full argument list.
        """
        if len(pars) != len(self._fit_idx):
            raise ValueError("Given number of parameters does not match the " +
                             "number of free parameters here.")
        # Explicit handling OK here, because we have only 4 combinations
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
    """
    Same as SinusFixedRateFunction, but sampling uniform times in each time
    interval instead of rejection sampling the sine function.

    Here the number of expected events is still following the seasonal
    fluctuations, but within the time windows we sample uniformly (step function
    like). Perfect for small time windows, avoiding rejection sampling and thus
    giving a speed boost.
    """
    def __init__(self, random_state=None):
        super(SinusFixedConstRateFunction, self).__init__(random_state)

    def sample(self, n_samples):
        """
        Just sample uniformly in MJD time windows here.

        Parameters
        ----------
        n_samples : array-like
            How many events to sample per source. Length must match length of
            cached source positions if any.
        """
        n_samples = np.atleast_1d(n_samples)
        if self._fmax is None:
            raise RuntimeError("Rate function was not fit yet.")
        if len(n_samples) != len(self._fmax):
            raise ValueError("Requested to sample a different number of " +
                             "sources than have been fit")

        # Samples times for all sources at once
        sample = []
        for dt, nsam in zip(self._trange, n_samples):
            sample.append(self._rndgen.uniform(dt[0], dt[1], size=nsam))

        return sample


class ConstantRateFunction(RateFunction):
    """
    Uses a constant rate in Hz at any given time in MJD. This models no seasonal
    fluctuations but uses the constant average rate.

    Uses one parameter:

    - rate, float: Constant rate in Hz.
    """
    _PARAMS = ["baseline"]

    def __init__(self, random_state=None):
        self.rndgen = random_state
        self._trange = None

    def fit(self, rate, srcs, w=None):
        """ Cache source values for sampling. Fit is the weighted average """
        if w is None:
            w = np.ones_like(rate)

        required_names = ["t", "dt0", "dt1"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` recarray is missing name " +
                                 "'{}'.".format(n))
        _dts = np.vstack((srcs["dt0"], srcs["dt1"])).T
        _, self._trange = self._transform_trange_mjd(srcs["t"], _dts)

        # Analytic solution to fit
        bf_pars = self._get_default_seed(rate, w)
        self._bf_fun = (lambda t: self.fun(t, bf_pars))
        self._bf_int = (lambda t, trange: self.integral(t, trange, bf_pars))
        self._bf_pars = bf_pars
        return bf_pars

    def fun(self, t, pars):
        """ Returns constant rate Hz at any given time in MJD """
        return np.ones_like(t) * pars[0]

    def integral(self, t, trange, pars):
        """
        Analytic integral of the rate function in interval trange. Because the
        rate function is constant, the integral is simply ``trange * rate``.
        """
        t, dts = self._transform_trange_mjd(t, trange)
        # Multiply first then diff to avoid roundoff errors(?)
        return (np.diff(dts * self._SECINDAY, axis=1) *
                self.fun(t, pars)).ravel()

    def sample(self, n_samples):
        n_samples = np.atleast_1d(n_samples)
        if len(n_samples) != len(self._trange):
            raise ValueError("Requested to sample a different number of " +
                             "sources than have been fit")

        # Samples times for all sources at once
        sample = []
        for dt, nsam in zip(self._trange, n_samples):
            sample.append(self._rndgen.uniform(dt[0], dt[1], size=nsam))

        return sample

    def _get_default_seed(self, rate, w):
        """
        Motivation for default seed:

        - rate0 : Mean of the given rates. This is the anlytic solution to the
                  fit, so we seed with the best fit. Weights must be squared
                  though, to get the same result.

        Returns
        -------
        p0 : tuple, shape(1)
            Seed values ``rate0 = np.average(rate, weights=w**2)``
        """
        return (np.average(rate, weights=w**2), )


##############################################################################
# Misc helper methods
##############################################################################
def power_law_flux(trueE, gamma):
    """
    Returns the unbroken power law flux :math:`\sim E^{-\gamma} summed over both
    particle types (nu, anti-nu), without a normalization.

    Parameters
    ----------
    trueE : array-like
        True particle energy in GeV.
    gamma : float
        Positive power law index.

    Returns
    -------
    flux : array-like
        Per nu+anti-nu particle flux :math:`\phi \sim E^{-\gamma}`.
    """
    return trueE**(-gamma)
