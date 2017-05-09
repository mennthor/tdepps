from ._utils import doc_inherit

import numpy as np
from sklearn.utils import check_random_state

import anapymods3.stats.KDE as KDE


class BGInjector(object):
    """
    Background Injector Interface

    Injects background events obtained from a given data sample.
    Describes a `fit` and a `sample` method.
    """
    def __init__(self):
        self._DESCRIBES = ["fit", "sample"]
        print("Interface only. Describes functions: ", self._DESCRIBES)
        return

    def fit(self, X, bounds=None):
        """
        Build the injection model with the provided data

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        bounds : array-like, shape (n_features, 2)
            Boundary conditions for each dimension. (default: None)
        """
        raise NotImplementedError("BGInjector is an interface.")

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (default: 1)
        random_state : RandomState, optional
            Turn seed into a `np.random.RandomState` instance. Method from
            `sklearn.utils`. Can be None, int or RndState. (default: None)

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            Generated samples from the fitted model.
        """
        raise NotImplementedError("BGInjector is an interface.")

    # Private methods
    def _check_bounds(self, bounds):
        """
        Check if bounds are OK. Create numerical values when None is given.

        Returns
        -------
        bounds : array-like, shape (n_features, 2)
            Boundary conditions for each dimension.
        """
        if bounds is None:
            bounds = np.repeat([[-np.inf, np.inf], ],
                               repeats=self._n_features, axis=0)

        bounds = np.array(bounds)
        if bounds.shape[1] != 2:
            raise ValueError("Invalid 'bounds'. Must be shape (n_features, 2).")

        # Convert None to +-np.inf depnding on low/hig bound
        bounds[:, 0][bounds[:, 0] == np.array(None)] = -np.inf
        bounds[:, 1][bounds[:, 1] == np.array(None)] = +np.inf

        return bounds

    def _check_sample_params(self, n_samples, random_state):
        """
        Check if bounds are OK.

        Returns
        -------
        random_state : np.random.RandomState
            Random state instance created by `sklearn.utils.check_random_state`.
        """
        if n_samples < 1:
            raise ValueError("'n_samples' must be at least 1.")

        return check_random_state(random_state)


class KDEBGInjector(BGInjector):
    """
    Adaptive Bandwidth Kernel Density Background Injector.

    Parameters are passed to the KDE class. Fitting of the model can take some
    time (60min / 100k evts) when adaptive kernels are used.

    Parameters
    ----------
    glob_bw : float or str
        The global bandwidth of the kernel, must be a float > 0 or one of
        ["silverman"|"scott"]. If alpha is not None, this is the bandwidth for
        the first estimate KDE from which the local bandwidth is calculated.
        If ["silverman"|"scott"] a rule of thumb is used to estimate the
        bandwidth. (default: "silverman")
    alpha : float or None
        If None, only the global bandwidth is used. If 0 <= alpha <= 1, an
        adaptive local kernel bandwith is used as described in. (default: 0.5)
    diag_cov : bool
        If True, only scale by variance, diagonal cov matrix. (default: False)
    max_gb : float
        Maximum gigabyte of RAM occupied in evaluating the KDE.
    """
    def __init__(self, glob_bw="silverman", alpha=0.5,
                 diag_cov=False, max_gb=2.):
        # Create KDE model
        self.kde_model = KDE.GaussianKDE(glob_bw=glob_bw, alpha=alpha,
                                         diag_cov=diag_cov, max_gb=max_gb)

        return

    @doc_inherit
    def fit(self, X, bounds=None):
        self._n_features = X.shape[1]
        self._bounds = self._check_bounds(bounds)
        # TODO: Use real bounds via mirror method in KDE class
        self.kde_model.fit(X)
        return

    @doc_inherit
    def sample(self, n_samples=1, random_state=None):
        random_state = self._check_sample_params(n_samples, random_state)
        rng = self._bounds

        # Check which samples are in range, redraw those that are not
        X = []
        while n_samples > 0:
            gen = self.kde_model.sample(n_samples, random_state)
            accepted = np.all(np.logical_and(gen >= rng[:, 0],
                                             gen <= rng[:, 1]), axis=1)
            n_samples = np.sum(~accepted)
            # Append accepted to final sample
            X.append(gen[accepted])

        return np.concatenate(X)


class DataBGInjector(BGInjector):
    """
    Data Background Injector

    Background Injector selecting random data events from the given sample.
    """
    def __init__(self):
        return

    def fit(self, X):
        """
        Build the injection model with the provided data.
        Here the 'model' is simply the data itself.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        self.X = np.copy(X)
        self._n_features = X.shape[1]
        return

    @doc_inherit
    def sample(self, n_samples=1, random_state=None):
        rndgen = self._check_sample_params(n_samples, random_state)
        # Draw indices uniformly from the data
        idx = rndgen.randint(self.X.shape[0], size=n_samples)
        return self.X[idx]


class UniformBGInjector(BGInjector):
    """
    Uniform Background Injector

    Background Injector creating uniform events on the whole sky.
    Created features are:
        - logE from a gaussian with mean 3 and stddev .5
        - Declination in rad uniformly distributed in sinDec
        - Sigma in rad from 3^2 * x * exp(-3 * x)
    """
    def __init__(self):
        # "Close" to real data
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
        raise AttributeError("Function disabled. Generating pesudo data only.")

    @doc_inherit
    def sample(self, n_samples=1, random_state=None):
        random_state = self._check_sample_params(n_samples, random_state)
        X = np.zeros((n_samples, self._n_features), dtype=np.float)

        # Sample logE from gaussian, sinDec uniform, sigma from x*exp(-x)
        X[:, 0] = random_state.normal(self._logE_mean,
                                      self._logE_sigma, size=n_samples)

        X[:, 1] = (np.arccos(random_state.uniform(-1, 1, size=n_samples)) -
                   np.pi / 2.)

        # From pythia8: home.thep.lu.se/~torbjorn/doxygen/Basics_8h_source.html
        u1, u2 = np.random.uniform(size=(2, n_samples))
        X[:, 2] = np.deg2rad(-np.log(u1 * u2) / self._sigma_scale)

        return X
