# coding: utf-8

from __future__ import division, absolute_import

import math
import numpy as np
import scipy.optimize as sco

from ..base import BaseLLH, BaseMultiLLH
from ..utils import fill_dict_defaults, all_equal


class GRBLLH(BaseLLH):
    """
    Stacking GRB LLH

    Stacking weights are a-priori fixed with w_theo * w_dec and only a single
    signal strength parameter ns is fitted.
    """
    def __init__(self, llh_model, llh_opts=None):
        """
        Parameters
        ----------
        llh_model : BaseModel instance
            Model providing LLH args and signal over background ratio.
        llh_opts : dict, optional
            LLH options:
            - 'sob_rel_eps', optional: Relative threshold under which a single
              signal over background ratio is considered zero for speed reasons.
            - 'sob_abs_eps', optional: Absolute threshold under which a single
              signal over background ratio is considered zero for speed reasons.
            - 'ns_bounds', optional: ``[lo, hi]`` bounds for the ``ns`` fit
              parameter.
            - 'minimizer', optional: String selecting a scipy minizer.
            - 'minimizer_opts', optional: Options dict for the scipy minimizer.
        """
        self._needed_args = ["src_w_dec", "src_w_theo" "nb"]
        self.model = llh_model
        self.llh_opts = llh_opts

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
        # Cache fixed src weights over background estimation, shape (nsrcs, 1)
        args = model.get_args()
        src_w = args["src_w_dec"] * args["src_w_theo"]
        self._src_w_over_nb = (src_w[:, None] / np.sum(src_w)) / args["nb"]
        self._model = model

    @property
    def llh_opts(self):
        return self._llh_opts

    @llh_opts.setter
    def llh_opts(self, llh_opts):
        required_keys = []
        opt_keys = {
            "sob_rel_eps": 0,
            "sob_abs_eps": 1e-3,
            "ns_bounds": [0., None],
            "minimizer": "L-BFGS-B",
            "minimizer_opts": {
                "ftol": 1e-15, "gtol": 1e-10, "maxiter": int(1e3)
                },
            }
        llh_opts = fill_dict_defaults(llh_opts, required_keys, opt_keys)
        if (llh_opts["sob_rel_eps"] < 0 or llh_opts["sob_rel_eps"] > 1):
            raise ValueError("'sob_rel_eps' must be in [0, 1]")
        if llh_opts["sob_abs_eps"] < 0:
            raise ValueError("'sob_abs_eps' must be >= 0.")
        if len(llh_opts["ns_bounds"]) != 2:
            raise ValueError("'ns_bounds' must be `[lo, hi]`.")
        if type(llh_opts["minimizer_opts"]) is not dict:
            raise ValueError("'minimizer_opts' must be a dictionary.")
        self._llh_opts = llh_opts

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
            res = sco.minimize(fun=self._neglnllh, x0=[ns0],
                               jac=True, args=(sob,),
                               bounds=self._llh_opts["ns_bounds"],
                               method=self._llh_opts["minimizer"],
                               options=self._llh_opts["minimizer_opts"])
            # Use `abs`: Very rarely, the minimizer doesn't go the boundary
            # when the fit should be zero. Then simply negating the function
            # value would yield a small but negative TS value. `abs` avoids
            # that accepting the "error" on having a very small positive TS
            # val instead of a true zero one. "return res.x[0], -res.fun[0]"
            return res.x[0], abs(res.fun[0])

    def _soverb(self, X):
        """ Make an additional cut on small sob values to save time """
        if len(X) == 0:  # With no events given, we can skip this step
            return np.empty(0, dtype=np.float)

        # Stacking case: Weighted signal sum per source
        sob = self._model.get_soverb(X)

        # TODO: Shape of src_w_over_nb is wrong. Check LLH model get_args.

        sob = np.sum(sob * self._src_w_over_nb, axis=0)

        # Apply a SoB ratio cut, to save computation time on events that don't
        # contribute anyway. We have a relative and an absolute threshold
        sob_max = np.amax(sob)
        if sob_max > 0:
            sob_rel_mask = (sob / sob_max) < self._llh_opts["sob_rel_eps"]
        else:
            sob_rel_mask = np.zeros_like(sob, dtype=bool)
        sob_abs_mask = sob < self._llh_opts["sob_abs_eps"]

        return sob[~(sob_rel_mask | sob_abs_mask)]

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


class MultiGRBLLH(BaseMultiLLH):
    """
    Class holding multiple GRBLLH objects, implementing the combined GRBLLH
    from all single GRBLLHs.
    """
    def __init__(self, llh_opts=None):
        self._ns_weights = None
        self.llh_opts = llh_opts

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

    @property
    def llh_opts(self):
        return self._llh_opts

    @llh_opts.setter
    def llh_opts(self, llh_opts):
        required_keys = []
        opt_keys = {
            "ns_bounds": [0., None],
            "minimizer": "L-BFGS-B",
            "minimizer_opts": {
                "ftol": 1e-15, "gtol": 1e-10, "maxiter": int(1e3)
                },
            }
        llh_opts = fill_dict_defaults(llh_opts, required_keys, opt_keys)
        if len(llh_opts["ns_bounds"]) != 2:
            raise ValueError("'ns_bounds' must be `[lo, hi]`.")
        if type(llh_opts["minimizer_opts"]) is not dict:
            raise ValueError("'minimizer_opts' must be a dictionary.")
        self._llh_opts = llh_opts

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

        # Cache ns plit weights used in combined LLH evaluation
        self._ns_weights = self._ns_split_weights(llhs)
        self._llhs = llhs
        return

    def lnllh_ratio(self, ns, X):
        """
        Combine LLH contribution from fitted single LLH instances.

        Parameters
        ----------
        ns : float
            Total expected signal events ``ns``.
        X : dict of recarrays
            Fixed data to evaluate the LHL at
        """
        # Loop over ln-LLHs and add their contribution
        TS = 0.
        ns_grad = 0.
        # Add up LLHs for each single LLH
        for key, llh in self._llhs.items():
            ns_w = self._ns_weights[key]
            TS_i, ns_grad_i = llh.lnllh_ratio(X[key], ns * ns_w)
            TS += TS_i
            ns_grad += ns_grad_i * ns_w  # Chain rule

        return TS, ns_grad

    def fit_lnllh_ratio(self, ns0, X):
        """
        Fit single ns parameter simultaneously for all LLHs.

        TODO: This relies on calls into private LLH methods directly using sob
              for speed reasons. Maybe we can change that.
        """
        def _neglnllh(ns, sob_dict):
            """ Multi LLH wrapper directly using a dict of sob values """
            TS = 0.
            ns_grad = 0.
            for key, sob in sob_dict.items():
                TS_i, ns_grad_i = self._llhs[key]._lnllh_ratio(
                    ns * self._ns_weights[key], sob)
                TS -= TS_i
                ns_grad -= ns_grad_i * self._ns_weights[key]  # Chain rule
            return TS, ns_grad

        # No events given for any LLH, fit is zero
        if sum(map(len, X.values())) == 0:
            return 0., 0.

        # Get soverb separately for all LLHs
        sob = []
        # sob_dict is only used if we fit, because we need sob unweighted there
        sob_dict = {}
        for key, llh in self._llhs.items():
            sob_i = llh._soverb(X[key])
            sob.append(self._ns_weights[key] * sob_i)
            if len(sob_i) > 0:
                # If sob is empty for a LLH, it would return (0, [0]) anyway,
                # so just add the existing ones. ns_weights are added in
                # correctly in the fit function later
                sob_dict[key] = sob_i

        sob = np.concatenate(sob)
        nevts = len(sob)

        # Test again, because we may have applied sob threshold cuts per LLH
        if nevts == 0:
            return 0., 0.
        elif nevts == 1:
            # Same case as in single LLH because sob is multi year weighted
            sob = sob[0]
            if sob <= 1.:  # sob <= 1 => ns <= 0, so fit will be 0
                return 0., 0.
            else:
                ns = 1. - (1. / sob)
                TS = 2. * (-ns + math.log(sob))
            return ns, TS
        elif nevts == 2:
            # Same case as in single LLH because sob is multi year weighted
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
            res = sco.minimize(fun=_neglnllh, x0=[ns0],
                               jac=True, args=(sob_dict,),
                               bounds=self._llh_opts["ns_bounds"],
                               method=self._llh_opts["minimizer"],
                               options=self._llh_opts["minimizer_opts"])
            if not res.success:
                def _neglnllh_numgrad(ns, sob_dict):
                    """ Use numerical gradient if LINESRCH problem arises. """
                    return _neglnllh(ns, sob_dict)[0]
                res = sco.minimize(fun=_neglnllh_numgrad, x0=[ns0],
                                   jac=True, args=(sob_dict,),
                                   bounds=self._llh_opts["ns_bounds"],
                                   method=self._llh_opts["minimizer"],
                                   options=self._llh_opts["minimizer_opts"])

            # Use `abs`: Very rarely, the minimizer doesn't go the boundary
            # when the fit should be zero. Then simply negating the function
            # value would yield a small but negative TS value. `abs` avoids
            # that accepting the "error" on having a very small positive TS
            # val instead of a true zero one.
            return res.x[0], abs(res.fun[0])

    def _ns_split_weights(self, llhs):
        """
        Set up the ``ns`` splitting weights: The weights simply renormalize the
        source weights for all single LLHs over all samples.

        Parameters
        ----------
        llhs : dict of LLH instances
            Single LLH instances that shall be combined.

        Returns
        -------
        ns_weigths : dict of array-like
            Weight per LLH to split up ``ns`` among different samples.
        """
        ns_weights = {}
        ns_w_sum = 0
        for key, llh in llhs:
            args = llh.model.get_args()
            ns_weights[key] = np.sum(args["src_w_dec"] * args["src_w_theo"])
            ns_w_sum += ns_weights[key]

        # Normalize weights over all sample source weights
        return {key: nsw / ns_w_sum for key, nsw in ns_weights.items()}
