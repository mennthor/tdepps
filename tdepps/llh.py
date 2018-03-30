# coding: utf-8

from __future__ import division, absolute_import

import abc
import math
import numpy as np
import scipy.optimize as sco

from .utils import fill_dict_defaults, all_equal


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


# #############################################################################
# GRB style LLH
# #############################################################################
class GRBLLH(BaseLLH):
    """
    Stacking GRB LLH

    Stacking weights are a-priori fixed with w_theo * w_dec and only a single
    signal strength parameter ns is fitted.
    """
    def __init__(self, llh_model, llh_args=None):
        self._needed_args = ["src_w", "nb"]
        self.model = llh_model
        self.llh_args = llh_args

    @property
    def needed_args(self):
        return self._needed_args

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if not all_equal(self._needed_args, model.provided_args):
            raise(KeyError("Model `provided_args` don't match `needed_args`."))
        self._model = model

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
        if len(_llh_args["ns_bounds"]) != 2:
            raise ValueError("'ns_bounds' must be `[lo, hi]`.")
        if type(_llh_args["minimizer_opts"]) is not dict:
            raise ValueError("'minimizer_opts' must be a dictionary.")
        self._llh_args = _llh_args

    def lnllh_ratio(self, ns, X):
        """ Public method wrapper """
        sob = self._soverb(X)
        return self._lnllh_ratio(ns, sob)

    def fit_lnllh_ratio(self, ns0, X):
        """ Fit TS with optimized analytic cases """
        if len(X) == 0:  # Fit is always 0 if no events are given
            return 0., 0.

        # Get the best fit parameter and TS. Analytic cases are handled:
        # For nevts = [1 | 2] we get a [linear | quadratic] equation to solve.
        sob = self._soverb(X)
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
        sob = self._model.get_soverb(X)
        args = self._model.get_args()
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


class MultiGRBLLH(BaseMultiLLH):
    """
    Class holding multiple GRBLLH objects, implementing the combined GRBLLH
    from all single GRBLLHs.
    """
    def __init__(self):
        self._ns_weights = None

    @property
    def names(self):
        return list(self._llhs.keys())

    @property
    def llhs(self):
        return self._llhs

    @property
    def model(self):
        return {key: llhi.model for key, llhi in self._llhs.items()}

    @property
    def needed_args(self):
        return {key: llhi.needed_args for key, llhi in self._llhs.items()}

    def fit(self, llhs):
        """
        Takes multiple single GRBLLHs in a dict and manages them.

        Parameters
        ----------
        llhs : dict
            LLHs to be managed by this multi LLH class. Names must match with
            dict keys of provided multi-injector data.
        """
        for name, llh in llhs.items():
            if not isinstance(llh, GRBLLH):
                raise ValueError("LLH object " +
                                 "`{}` is not of type `GRBLLH`.".format(name))

        self._llhs = llhs

    def lnllh_ratio(self, ns, X):
        raise NotImplementedError("TODO")

    def fit_lnllh_ratio(self, ns0, X):
        raise NotImplementedError("TODO")
