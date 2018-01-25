# coding: utf-8

from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()

import math
import numpy as np
import scipy.optimize as sco

from .utils import fill_dict_defaults


class GRBLLH(object):
    """
    Stacking GRB LLH

    Stacking weights are a-priori fixed with w_theo * w_dec and only a single
    signal strength parameter ns is fitted.
    """
    def __init__(self, model_pdf, llh_args=None):
        self.model_pdf = model_pdf
        self.llh_args = llh_args

    @property
    def model_pdf(self):
        return self._model_pdf

    @model_pdf.setter
    def model_pdf(self, model_pdf):
        self._model_pdf = model_pdf

    @property
    def llh_args(self):
        return self._llh_args

    @llh_args.setter
    def llh_args(self, llh_args):
        required_keys = []
        opt_keys = {
            "sob_rel_eps": 0,
            "sob_abs_eps": 1e-3,
            "sindec_band": 0.1,
            "ns_bounds": [0., None],
            "minimizer_opts": {
                "ftol": 1e-15, "gtol": 1e-10, "maxiter": int(1e3)
                },
            }
        _llh_args = fill_dict_defaults(llh_args, required_keys, opt_keys)
        if (_llh_args["sob_rel_eps"] < 0 or _llh_args["sob_rel_eps"] > 1):
            raise ValueError("'sob_rel_eps' must be in [0, 1]")
        if _llh_args["sob_abs_eps"] < 0:
            raise ValueError("'sob_abs_eps' must be >= 0.")
        if _llh_args["sindec_band"] < 0.:
            raise ValueError("'sindec_band' must be > 0.")
        if len(_llh_args["ns_bounds"] != 2):
            raise ValueError("'ns_bounds' must be `[lo, hi]`.")
        if type(_llh_args["minimizer_opts"]) is not dict:
            raise ValueError("'minimizer_opts' must be a dictionary.")
        self._llh_args = _llh_args

    # PUBLIC
    def lnllh_ratio(self, ns, X):
        """ Public method wrapper """
        sob = self._soverb(self._select_X(X))
        return self._lnllh_ratio(ns, sob)

    def fit_lnllh_ratio(self, ns0, X):
        """ Fit TS with optimized analytic cases """
        if len(X) == 0:  # Fit is always 0 if no events are given
            return 0., 0.

        # Get the best fit parameter and TS. Analytic cases are handled:
        # For nevts = [1 | 2] we get a [linear | quadratic] equation to solve.
        sob = self._soverb(self._select_X(X))
        nevts = len(sob)

        # Test again, because we applied some threshold cuts
        if nevts == 0:
            return 0., 0.
        elif nevts == 1:
            sob = sob[0]
            if sob <= 1.:  # sob <= 1 => ns <= 0, so fit will be 0
                return 0., 0.
            else:
                ns = 1. - (1. / sob)
                TS = 2. * (-ns + math.log(sob))
            return ns, TS
        elif nevts == 2:
            sum_sob = sob[0] + sob[1]
            if sum_sob <= 1.:  # More complicated to show but same as above
                return 0., 0.
            else:
                a = 1. / (sob[0] * sob[1])
                c = sum_sob * a
                ns = 1. - 0.5 * c + math.sqrt(c * c / 4. - a + 1.)
                TS = 2. * (-ns + np.sum(np.log1p(ns * sob)))
                return ns, TS
        else:  # Fit other cases
            res = sco.minimize(fun=self._neglnllh, x0=[ns0], jac=True,
                               bounds=self._llh_args["bounds"], args=(sob,),
                               method="L-BFGS-B",
                               options=self._llh_args["minimizer_opts"])
            return res.x[0], -res.fun[0]

    # PRIVATE
    def _lnllh_ratio(self, ns, sob):
        """ Calculate TS = 2 * ln(L1 / L0) """
        x = ns * sob
        TS = 2. * (-ns + np.sum(np.log1p(x)))
        # Gradient in ns (chain rule: ln(ns * a + 1)' = 1 / (ns * a + 1) * a)
        ns_grad = 2. * (-1. + np.sum(sob / (x + 1.)))
        return TS, np.array([ns_grad])

    def _neglnllh(self, ns, sob):
        """ Wrapper for minimizing the negative lnLLH ratio """
        lnllh, lnllh_grad = self._lnllh_ratio(ns, sob)
        return -lnllh, -lnllh_grad

    def _soverb(self, X):
        """ Make an additional cut on small sob values to save time """
        if len(X) == 0:  # With no events given, we can skip this step
            return np.empty(0, dtype=np.float)

        # Signal stacking case: Weighted signal sum per source
        sob = self._model_pdf.get_soverb(self._select_X(X))
        args = self._model_pdf.get_args()
        sob = np.sum(sob * args["src_w"] / args["nb"], axis=0)

        # Apply a SoB ratio cut, to save computation time on events that don't
        # contribute anyway. We have a relative and an absolute threshold
        sob_max = np.amax(sob)
        if sob_max > 0:
            sob_rel_mask = (sob / sob_max) < self._llh_args["sob_rel_eps"]
        else:
            sob_rel_mask = np.zeros_like(sob, dtype=bool)
        sob_abs_mask = sob < self._llh_args["sob_abs_eps"]
        return sob[np.logical_not(np.logical_or(sob_rel_mask, sob_abs_mask))]

    def _select_X(self, X):
        """
        Select events in a band around the source declinations and discard those
        outside, which have a negligible contribtion on the the result.
        """
        # TODO
        return X


class MultiGRBLLH(object):
    """
    Class holding multiple GRBLLH objects, implementing the combined LLH
    from all single LLHs.
    """
    def __init__(self):
        self._llhs = {}
        self._ns_weights = None

    @property
    def names(self):
        return list(self._llhs.keys())

    @property
    def llhs(self):
        return self._llhs

    def add_sample(self, name, llh):
        if not isinstance(llh, GRBLLH):
            raise ValueError("`llh` object must be of type GRBLLH.")

        if name in self.names:
            raise KeyError("Name '{}' has already been added. ".format(name) +
                           "Choose a different name.")
        else:
            self._llhs[name] = llh

        # Reset, because weights change for each added sample.
        self._ns_weights = None
