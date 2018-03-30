# coding: utf-8

"""
Base definition for model objects used in LLHs.
"""

from __future__ import absolute_import
import abc


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

    @abc.abstractproperty
    def srcs(self):
        """ Source recarray the model was built for """
        pass

    @abc.abstractmethod
    def get_soverb(self):
        """ Returns the signal over bg ratio for each given data point """
        pass

    @abc.abstractmethod
    def get_args(self):
        """ Returns fixed argus the LLH needs, which are not data or params """
        pass
