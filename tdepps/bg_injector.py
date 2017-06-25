# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()                                              # noqa

import numpy as np
from numpy.lib.recfunctions import drop_fields, append_fields
from sklearn.utils import check_random_state

import anapymods3.stats.KDE as KDE

import abc     # Abstract Base Class
import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class BGInjector(object):
    __metaclass__ = abc.ABCMeta

    @docs.get_sectionsf("BGInjector.init", sections=["Parameters"])
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Background Injector Base Class

        Base class for generating background events with 3 features:
        declination, logE, and directional reconstruction error sigma.

        Classes must implement methods:

        - `fun`
        - `sample`

        Class object then provides public methods:

        - `fun`
        - `sample`

        Parameters
        ----------
        random_state : RandomState, optional
            Turn seed into a `np.random.RandomState` instance. Method from
            `sklearn.utils`. Can be None, int or RndState. (default: None)

        Example
        -------
        >>> import tdepps.bg_injector as BGInj
        >>> data_inj = BGInj.DataBGInjector()
        >>>
        >>> # Generate some test data
        >>> n_evts, n_features = 100, 3
        >>> X = np.random.uniform(0, 1, size=(n_evts, n_features))
        >>> X = np.core.records.fromarrays(X.T, names=["logE", "dec", "sigma"])
        >>>
        >>> # Fit and sample from testdata
        >>> data_inj.fit(X)
        >>> data_sam = data_inj.sample(n_samples=1000)
        """
        self.rndgen = random_state

        # self._X_names = ["ra", "sinDec", "logE", "sigma", "timeMJD"]
        self._X_names = ["logE", "dec", "sigma"]

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    @docs.get_sectionsf("BGInjector.fit", sections=["Parameters", "Returns"])
    @docs.dedent
    @abc.abstractmethod
    def fit(self, X):
        """
        Build the injection model with the provided data

        Parameters
        ----------
        X : record-array
            Experimental data from which background-like event are generated.
            dtypes are ["name", type]. Here X must have names:

            - "dec": Per event declination, [-pi/2, pi/2], coordinates in
              equatorial coordinates, given in radians.
            - "logE": Per event energy proxy, given in log10(1/GeV).
            - "sigma": Per event positional uncertainty, given in radians. It is
              assumed, that a circle with radius sigma contains approximatly
              :math:`1\sigma\approx 0.39` of probability of the reconstrucion
              likelihood space.

            Other names are dropped and not fitted by the injector.
        """
        pass

    @docs.get_sectionsf("BGInjector.sample", sections=["Parameters", "Returns"])
    @docs.dedent
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
            Generated samples from the fitted model. Has the same keys as the
            given record-array X in `fit` and additional names:

            - "ra", float: Equatorial right-ascension coordinates in radains.
              These are sampled uniformly in [0, 2pi].
            - "sinDec", float: Sinus declination in range [-1, 1] transformed
              from sampled declination values.
        """
        pass

    def _add_ra_sin_dec(self, X):
        """
        Adds uniformly sampled right ascension and sinus declination to the
        record array.

        Parameters
        ----------
        X
            Sees:py:meth:`BGInjector.fit`, Parameters

        Returns
        -------
        X
            See :py:meth:`BGInjector.sample`, Returns
        """
        ra = self._rndgen.uniform(0, 2 * np.pi, size=len(X))
        sin_dec = np.sin(X["dec"])

        # Append ra, sin_dec fields to X output array
        return append_fields(X, ["ra", "sinDec"], [ra, sin_dec],
                             dtypes=np.float, usemask=False)

    def _check_bounds(self, bounds):
        """
        Check if bounds are OK. Create numerical values when None is given.

        Parameters
        ----------
        bounds
            See :py:meth:`BGInjector.fit`, Parameters

        Returns
        -------
        bounds : array-like, shape (n_features, 2)
            Boundary conditions for each dimension. Unconstrained axes have
            bounds [-np.inf, +np.inf].
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
        """
        Check if record arrays have the required names specific for PS analysis,
        drops other given names.

        Parameters
        ----------
        X
            See :py:meth:`fit`, Parameters

        Raises
        ------
        ValueError
            When `X` does not have :py:meth:names ["logE", "dec", "sigma"].
        """
        if X.dtype.names is None:
            valid = "['" + "', '".join(self._X_names) + "']"
            raise ValueError("`X` must be a record-array with names " + valid)
        for n in self._X_names:
            if n not in X.dtype.names:
                raise ValueError("`X` is missing name '{}'.".format(n))

        # Drop unneeded fields
        drop = [n for n in X.dtype.names if n not in self._X_names]
        return drop_fields(X, drop, usemask=False)


class KDEBGInjector(BGInjector):
    @docs.dedent
    def __init__(self, glob_bw="silverman", alpha=0.5, diag_cov=False,
                 random_state=None):
        """
        Adaptive Bandwidth Kernel Density Background Injector.

        Parameters are passed to the KDE class. Fitting of the model can take
        some time when adaptive kernels are used.

        Parameters
        ----------
        glob_bw : float or str
            The global bandwidth of the kernel, must be a float > 0 or one of
            `['silverman'|'scott']`. If alpha is not None, this is the bandwidth
            for the first estimate KDE from which the local bandwidth is
            calculated. If `['silverman'|'scott']` a rule of thumb is used to
            estimate the bandwidth. (default: 'silverman')
        alpha : float or None
            If None, only the global bandwidth is used. If `0 <= alpha <= 1`, an
            adaptive local kernel bandwith is used as described in.
            (default: 0.5)
        diag_cov : bool
            If True, only scale by variance, diagonal cov matrix.
            (default: False)
        %(BGInjector.init.parameters)s
        """
        super(KDEBGInjector, self).__init__(random_state)

        self._n_features = None
        # Create KDE model
        self._kde_model = KDE.GaussianKDE(glob_bw=glob_bw, alpha=alpha,
                                          diag_cov=diag_cov)

        return

    @docs.dedent
    def fit(self, X, bounds=None):
        """
        Fit a KDE model to the given data.

        Parameters
        ----------
        %(BGInjector.fit.parameters)s
        bounds : None or array-like, shape (n_features, 2)
            Boundary conditions for each dimension. If None, [-np.inf, +np.inf]
            is used in each dimension. (default: None)

        Returns
        -------
        %(BGInjector.fit.returns)s
        """
        X = self._check_X_names(X)
        self._n_features = len(X.dtype.names)

        # TODO: Use advanced bounds via mirror method in KDE class.
        # Currently bounds are used to resample events that fall outside
        self._bounds = self._check_bounds(bounds)

        # Turn record-array in normal 2D array for more general KDE class
        X = np.vstack((X[n] for n in self._X_names)).T
        self._kde_model.fit(X)
        return

    @docs.dedent
    def sample(self, n_samples=1):
        """
        Sample from a KDE model that has been build on given data.

        The model can only sample data in the previously fit form. So make sure
        to use the same convention everywhere.

        Parameters
        ----------
        %(BGInjector.sample.parameters)s

        Returns
        -------
        %(BGInjector.sample.returns)s
        """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty array with all keys, when n_samples < 1
        if n_samples < 1:
            X = np.empty((0, ),
                         dtype=[(n, np.float) for n in self._X_names])
            return self._add_ra_sin_dec(X)

        # Check which samples are in bounds, redraw those that are not
        X = []
        bounds = self._bounds
        while n_samples > 0:
            gen = self._kde_model.sample(n_samples, self._rndgen)
            accepted = np.all(np.logical_and(gen >= bounds[:, 0],
                                             gen <= bounds[:, 1]), axis=1)
            n_samples = np.sum(~accepted)
            # Append accepted to final sample
            X.append(gen[accepted])

        # Combine and convert to record-array
        X = np.concatenate(X)
        X = np.core.records.fromarrays(X.T, names=self._X_names,
                                       formats=self._n_features * ["float64"])

        return self._add_ra_sin_dec(X)


class DataBGInjector(BGInjector):
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Data Background Injector

        Background Injector selecting random data events from the given sample.

        Parameters
        ----------
        %(BGInjector.init.parameters)s
        """
        super(DataBGInjector, self).__init__(random_state)
        self._n_features = None
        return

    @docs.dedent
    def fit(self, X):
        """
        Build the injection model with the provided data. Here the model is
        simply the data itself.

        Parameters
        ----------
        %(BGInjector.fit.parameters)s
        """
        self.X = self._check_X_names(X)
        self._n_features = len(X.dtype.names)
        return

    @docs.dedent
    def sample(self, n_samples=1):
        """
        Sample by choosing random events from the given data.

        Parameters
        ----------
        %(BGInjector.sample.parameters)s

        Returns
        -------
        %(BGInjector.sample.returns)s
        """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty array with all keys, when n_samples < 1
        if n_samples < 1:
            X = np.empty((0, ),
                         dtype=[(n, np.float) for n in self._X_names])
            return self._add_ra_sin_dec(X)

        # Choose uniformly from given data
        X = self._rndgen.choice(self.X, size=n_samples)

        return self._add_ra_sin_dec(X)


class UniformBGInjector(BGInjector):
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Uniform Background Injector

        Background Injector creating uniform events on the whole sky.
        Created features (in addition to right-ascension) are:

        - logE from a gaussian with mean 3 and stddev 0.5
        - Declination in radian uniformly distributed in sinDec
        - Sigma in radian modeled as :math:`f(x) = 3^2 x exp(-3x)`

        Parameters
        ----------
        %(BGInjector.init.parameters)s
        """
        super(UniformBGInjector, self).__init__(random_state)

        # Values "close" to real data
        self._logE_mean = 3.
        self._logE_sigma = .25
        self._sigma_scale = 3.

        # Model is completely generated, only save n_features for consistency
        self._n_features = 3
        return

    def fit(self):
        """
        Method has no effect: Model generates pseudo data only, no fit needed.
        """
        raise NotImplementedError(
            "Function disabled. Generating pesudo data only.")

    @docs.dedent
    def sample(self, n_samples=1):
        """
        Sample pseudo events that look somewhat similar to data.

        The method on how to sample from x*exp(x) is taken from Pythia [1]_.

        Parameters
        ----------
        %(BGInjector.sample.parameters)s

        Returns
        -------
        %(BGInjector.sample.returns)s

        Notes
        -----
        .. [1] # From pythia8: home.thep.lu.se/~torbjorn/doxygen/Basics_8h_source.html # noqa
        """
        # Return empty array with all keys, when n_samples < 1
        if n_samples < 1:
            X = np.empty((0, ),
                         dtype=[(n, np.float) for n in self._X_names])
            return self._add_ra_sin_dec(X)

        X = np.empty((n_samples, ),
                     dtype=[(n, np.float) for n in self._X_names])

        # Sample logE from gaussian, sinDec uniformly, sigma from x*exp(-x)
        X["logE"] = self._rndgen.normal(self._logE_mean, self._logE_sigma,
                                        size=n_samples)

        X["dec"] = (np.arccos(self._rndgen.uniform(-1, 1, size=n_samples)) -
                    np.pi / 2.)

        u1, u2 = self._rndgen.uniform(size=(2, n_samples))
        X["sigma"] = np.deg2rad(-np.log(u1 * u2) / self._sigma_scale)

        return self._add_ra_sin_dec(X)


class MRichmanBGInjector(BGInjector):
    @docs.dedent
    def __init__(self, random_state=None):
        """
        Injector binning up data space in `[a x b x c]` bins with equal
        statistics and then sampling uniformly from those bins. Data must have 3
        dimensions.

        Parameters
        ----------
        %(BGInjector.init.parameters)s
        """
        super(MRichmanBGInjector, self).__init__(random_state)
        self._n_features = None
        return

    @docs.dedent
    def fit(self, X, nbins=10, minmax=False):
        """
        Build the injection model with the provided data.

        Dimensions fixed to 3. For a variable solution we would need recursion.

        Parameters
        ----------
        %(BGInjector.fit.parameters)s
        nbins : int or array-like, shape(n_features), optional

            - If int, same number of bins is used for all dimensions.
            - If array-like, number of bins for each dimension is used.

            (default: 10)

        minmax : bool, optional
            If True, use global min/max for outermost bin edges. Else use the
            min/max bounds in current data bin. Can be used for a global
            bounding box. (default: False)

        Returns
        -------
        ax0_bins : array-like, shape (nbins[0] + 1)
            The bin borders for the first dimension.
        ax1_bins : array-like, shape (nbins[0], nbins[1] + 1)
            The bin borders for the second dimension.
        ax2_bins : array-like, shape (nbins[0], nbins[1] nbins[2] + 1)
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
        X = np.vstack((X[n] for n in self._X_names)).T

        self._n_features = X.shape[1]
        if self._n_features != 3:
            raise ValueError("Only 3 dimensions supported.")

        # Repeat bins, if only int was given
        nbins = np.atleast_1d(nbins)
        if (len(nbins) == 1) and (len(nbins) != self._n_features):
            nbins = np.repeat(nbins, repeats=self._n_features)
        elif len(nbins) != self._n_features:
            raise ValueError("Given 'nbins' doesn't match dim of data.")
        self._nbins = nbins

        # Get bounding box, we sample the maximum distance in each direction
        if minmax is True:
            minmax = np.vstack((np.amin(X, axis=0), np.amax(X, axis=0))).T
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

    @docs.dedent
    def sample(self, n_samples=1):
        """
        Sample pseudo events uniformly from each bin.

        Parameters
        ----------
        %(BGInjector.sample.parameters)s

        Returns
        -------
        %(BGInjector.sample.returns)s
        """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty array with all keys, when n_samples < 1
        if n_samples < 1:
            X = np.empty((0, ),
                         dtype=[(n, np.float) for n in self._X_names])
            return self._add_ra_sin_dec(X)

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
        X = np.vstack((ax0_pts, ax1_pts, ax2_pts))
        X = np.core.records.fromarrays(X, names=self._X_names,
                                       formats=self._n_features * ["float64"])

        return self._add_ra_sin_dec(X)
