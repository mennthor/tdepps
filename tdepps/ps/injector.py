# coding: utf-8

"""
Collection of base implementations for a GRB style, time dependent analysis.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.lib.recfunctions import drop_fields

from ..base import BaseBGDataInjector, BaseMultiBGDataInjector
from ..utils import arr2str, logger, dict_map


##############################################################################
# Signal injector classes
##############################################################################
# TODO, almost the same as in GRB case, but with livetime


##############################################################################
# Background injector classes
##############################################################################
class ScrambledBGDataInjector(BaseBGDataInjector):
    """
    Injects background by simply assigning new RA values per trial to saved data
    """
    def __init__(self, random_state=None):
        """
        Parameters
        ----------
        random_state : None, int or np.random.RandomState, optional
            Used as PRNG, see ``sklearn.utils.check_random_state``.
            (default: None)
        """
        self._provided_data = np.array(
            ["dec", "ra", "sigma", "logE"])
        self.rndgen = random_state

        # Defaults for private class variables
        self._X = None
        self._nevts = None

        self._log = logger(name=self.__class__.__name__, level="ALL")

    @property
    def provided_data(self):
        return self._provided_data

    @property
    def srcs(self):
        return None

    def fit(self, X):
        """
        Just store data for resampling.

        Parameters
        ----------
        X : recarray
            Experimental data for BG injection.
        """
        X_names = np.array(X.dtype.names)
        for name in self._provided_data:
            if name not in X_names:
                raise ValueError("`X` is missing name '{}'.".format(name))
        drop = np.isin(X_names, self._provided_data,
                       assume_unique=True, invert=True)
        drop_names = X_names[drop]
        print(self._log.INFO("Dropping names '{}'".format(arr2str(drop_names)) +
                             " from data recarray."))

        self._X = drop_fields(X, drop_names, usemask=False)
        self._nevts = len(self._X)

        return

    def sample(self):
        """
        Resample the whole data set by drawing new RA values.
        """
        self._X["ra"] = self._rndgen.uniform(0, 2. * np.pi, size=self._nevts)
        return self._X


class MultiBGDataInjector(BaseMultiBGDataInjector):
    """
    Container class simply collects all samples from the individual injectors.
    """
    @property
    def names(self):
        return list(self._injs.keys())

    @property
    def injs(self):
        return self._injs

    @property
    def provided_data(self):
        return dict_map(lambda key, inj: inj.provided_data, self._injs)

    @property
    def srcs(self):
        return dict_map(lambda key, inj: inj.srcs, self._injs)

    def fit(self, injs):
        """
        Takes multiple single injectors in a dict and manages them.

        Parameters
        ----------
        injs : dict
            Injectors to be managed by this multi injector class. Names must
            match with dict keys of required multi-LLH data.
        """
        for name, inj in injs.items():
            if not isinstance(inj, BaseBGDataInjector):
                raise ValueError("Injector `{}` ".format(name) +
                                 "is not of type `BaseBGDataInjector`.")

        self._injs = injs

    def sample(self):
        """
        Sample each injector and combine to a dict of recarrays.
        """
        return dict_map(lambda key, inj: inj.sample(), self._injs)
