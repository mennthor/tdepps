from ._utils import doc_inherit

import numpy as np
from sklearn.utils import check_random_state


class BGRateInjector(object):
    """
    Background Rate Injector Interface

    Samples times of BG-like events for a given time frame.
    Describes a `fit` and a `sample` method.
    """
    def __init__(self):
        self._DESCRIBES = ["fit", "sample"]
        print("Interface only. Describes functions: ", self._DESCRIBES)
        return

    def fit(self, X):
        """
        Build the injection model with the provided data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to
            a single data point, each column is a coordinate.
        """
        raise NotImplementedError("BGInjector is an interface.")

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (defaults: 1)
        random_state : RandomState, optional
            A random number generator instance. (default: None)

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            Generated samples from the fitted model.
        """
        raise NotImplementedError("BGInjector is an interface.")


class RunlistBGRateInjector():
    def __init__(self):
        return

    def fit(self, X, bins, fit_seed):
        """
        Build the injection model with the provided data.

        Takes data and a binning derived from a runlist. Bins the data,
        normalizes to a rate in HZ and fits a periodic function over the whole
        time span to it. This function serves as a rate per time model.

        The function is chosen to be a sinus with:

        ..math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

        where

        - a is the Amplitude in Hz
        - b is the period scale in 1/MJD
        - c is the x-offset in MJD
        - d the y-offset in Hz

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to
            a single data point, each column is a coordinate.
        bins : array-like, shape (nbins, 2)
            Time bins, where every row represents the start and end time in MJD
            for a single run. This can be preselected from a goodrun list.
        fit_seed : array-like, shape (4)
            Seed values for the fit function as described above.
        """
        raise NotImplementedError("BGInjector is an interface.")

    @doc_inherit
    def sample(self, n_samples=1, random_state=None):
        rndgen = check_random_state(random_state)
        raise NotImplementedError("BGInjector is an interface.")
        return


# class TimebinBGRateInjector():


# class FunctionBGRateInjector():




















