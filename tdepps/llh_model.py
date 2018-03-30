# coding: utf-8

from __future__ import absolute_import
import abc
import numpy as np

from ..utils import fill_dict_defaults


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


# #############################################################################
# GRB style Model
# #############################################################################
class GRBModel(BaseModel):
    """
    Models the PDF part for the GRB LLH function.
    """
    def __init__(self, X, MC, srcs, model_opts):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        self._needed_data = np.array(
            ["timeMJD", "dec", "ra", "sigma", "logE"])
        self._provided_args = ["src_w_dec", "src_w_theo" "nb"]

        self.model_opts = model_opts

        self._setup_model()
        self._srcs = srcs

    @property
    def needed_data(self):
        return self._needed_data

    @property
    def provided_args(self):
        return self._provided_args

    @property
    def srcs(self):
        """ Source recarray the injector was fitted to """
        return self._srcs

    @property
    def model_opts(self):
        return self._model_opts

    @model_opts.setter
    def model_opts(self, model_opts):
        required_keys = []
        opt_keys = {"sindec_band": 0.1}
        model_opts = fill_dict_defaults(model_opts, required_keys, opt_keys)
        if model_opts["sindec_band"] < 0.:
            raise ValueError("'sindec_band' must be > 0.")
        self._model_opts = model_opts

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
