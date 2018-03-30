# coding: utf-8

from __future__ import absolute_import

import numpy as np

from ..base import BaseModel
from ..utils import fill_dict_defaults


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
