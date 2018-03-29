# coding: utf-8

from __future__ import absolute_import
import abc
import numpy as np


class BaseModel(object):
    """ Interface for unbinned LLH model type classes """
    __metaclass__ = abc.ABCMeta

    _needed_data = None

    @abc.abstractproperty
    def needed_data(self):
        """ Data recarray attributes this PDF model needs for evaluation """
        pass

    @abc.abstractproperty
    def provided_args(self):
        """ Additional LLH arguments this model provides via `get_args` """
        pass

    @abc.abstractmethod
    def get_soverb(self):
        """ Returns the signal over bg ratio for each given data point """
        pass

    @abc.abstractmethod
    def get_args(self):
        """ Returns fixed argus the LLH needs, which are not data or params """
        pass


# #############################################################################
# GRB style Model
# #############################################################################
class GRBModel(BaseModel):
    """
    Models the PDF part for the GRB LLH function.
    """
    def __init__(self, X, MC, srcs):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        self._needed_data = np.array(
            ["timeMJD", "dec", "ra", "sigma", "logE"])
        self._provided_args = ["src_w", "nb"]

    @property
    def needed_data(self):
        return self._needed_data

    @property
    def provided_args(self):
        return self._provided_args

    def get_args(self):
        pass

    def get_soverb(self):
        pass

    def _select_X(self, X):
        """
        Select events in a band around the source declinations and discard those
        outside, which have a negligible contribution on the the result.
        """
        # TODO
        return X
