# coding: utf-8

"""
Base definition for LLH objects.
"""

from __future__ import division, absolute_import
import abc


class BaseLLH(object):
    """ Interface for LLH type classes """
    __metaclass__ = abc.ABCMeta

    _model = None

    @abc.abstractproperty
    def model(self):
        """ The underlying model this LLH is based on """
        pass

    @abc.abstractproperty
    def needed_args(self):
        """ Additional LLH arguments, must match with model `provided_args` """
        pass

    @abc.abstractmethod
    def lnllh_ratio(self):
        """ Returns the lnLLH ratio given data and params """
        pass

    @abc.abstractmethod
    def fit_lnllh_ratio(self):
        """ Returns the best fit parameter set under given data """
        pass


class BaseMultiLLH(BaseLLH):
    """ Interface for managing multiple LLH type classes """
    _llhs = None

    @abc.abstractproperty
    def llhs(self):
        """ Dict of sub-llhs, identifies this as a MultiLLH """
        pass
