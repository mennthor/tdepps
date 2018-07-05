# coding: utf-8

from __future__ import division, absolute_import

import math
import numpy as np
import scipy.optimize as sco

from ..base import BaseLLH, BaseMultiLLH
from ..utils import fill_dict_defaults, all_equal, dict_map


class PSLLH(BaseLLH):
    """
    Stacking time integrated PS LLH

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
        self._needed_args = ["src_w_dec", "src_w_theo"]
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
        # Shape is (1, nsrcs) for general stacking LLH
        self._src_w = (src_w / np.sum(src_w))[:, None]
        self._model = model

    @property
    def llh_opts(self):
        return self._llh_opts.copy()

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

    def lnllh_ratio(self, ns, X, band_select=True):
        """ Public method wrapper """
        sob = self._soverb(X, band_select=band_select)
        return self._lnllh_ratio(ns, sob)

    def fit_lnllh_ratio(self, ns0, X, band_select=True):
        """ Fit TS with optimized analytic cases """
        def _neglnllh(ns, sob):
            """ Wrapper for minimizing the negative lnLLH ratio """
            lnllh, lnllh_grad = self._lnllh_ratio(ns, sob, nevts)
            return -lnllh, -lnllh_grad

        # If selection cuts are applied the number
        nevts = len(X)
        sob = self._soverb(X, band_select=band_select)
        assert nevts >= len(sob)

        # Get the best fit parameter and TS
        res = sco.minimize(fun=_neglnllh, x0=[ns0],
                           jac=True, args=(sob, nevts),
                           bounds=[self._llh_opts["ns_bounds"]],
                           method=self._llh_opts["minimizer"],
                           options=self._llh_opts["minimizer_opts"])
        ns, ts = res.x[0], -res.fun[0]
        if ts < 0.:
            # Some times the minimizer doesn't go all the way to 0., so
            # TS vals might end up negative for a truly zero fit result
            ts = 0.
        return ns, ts

    def _soverb(self, X, band_select=True):
        """
        Compute the the signal over background stacking sum and make an
        additional cut on small sob values to save time
        """
        # Stacking case: Weighted signal sum per source: shape (nevts,)
        sob = self._model.get_soverb(X, band_select=band_select)
        sob = np.sum(sob * self._src_w, axis=0)
        if len(sob) < 1:
            return np.empty(0, dtype=np.float)

        # Apply a SoB ratio cut, to save computation time on events that don't
        # contribute anyway. There is a relative and an absolute threshold
        sob_max = np.amax(sob)
        if sob_max > 0:
            sob_rel_mask = (sob / sob_max) < self._llh_opts["sob_rel_eps"]
        else:
            sob_rel_mask = np.zeros_like(sob, dtype=bool)
        sob_abs_mask = sob < self._llh_opts["sob_abs_eps"]

        return sob[~(sob_rel_mask | sob_abs_mask)]

    def _lnllh_ratio(self, ns, sob, nevts):
        """
        Calculate TS = 2 * ln(L1 / L0)

        nevts gives the number of original used events. A subset of them may
        have been set to zero due to selection cuts. These terms need to be
        added manually to the LLH and the gradient.
        """
        nsel = len(sob)
        nzero = nevts - nsel
        assert nevts >= nsel

        x = (sob - 1.) / nevts
        alpha = ns * x
        ts = 2. * (np.sum(np.log1p(alpha)) +
                   nzero * np.log1p(-ns / nevts))
        ns_grad = 2. * (np.sum(x / ((alpha) + 1.)) -
                        nzero / (nevts - ns))

        return ts, np.array([ns_grad])


class MultiGRBLLH(BaseMultiLLH):
    """
    Class holding multiple PSLLH objects, implementing the combined PSLLH
    from all single PSLLHs.
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
        return dict_map(lambda key, llh: llh.model, self._llhs)

    @property
    def needed_args(self):
        return dict_map(lambda key, llh: llh.needed_args, self._llhs)

    @property
    def llh_opts(self):
        return self._llh_opts.copy()

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
        Takes multiple single PSLLHs in a dict and manages them.

        Parameters
        ----------
        llhs : dict
            LLHs to be managed by this multi LLH class. Names must match with
            dict keys of provided multi-injector data.
        """
        for name, llh in llhs.items():
            if not isinstance(llh, PSLLH):
                raise ValueError("LLH object " +
                                 "`{}` is not of type `PSLLH`.".format(name))

        # Check if all sources are equal
        srcs = dict_map(lambda k, v: v.model.srcs, llhs)
        keys = srcs.keys()
        ref = keys[0]
        for key in keys[1:]:
            if not np.array_equal(srcs[ref], srcs[key]):
                raise ValueError(
                    "Sources in sample '{}' are not equal to ".format(ref) +
                    "sources in sample '{}'.".format(key))

        nsrcs = len(srcs.values()[0])

        # Cache ns plit weights used in combined LLH evaluation
        self._ns_weights = self._ns_split_weights(llhs, nsrcs)
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
        ts = 0.
        ns_grad = 0.
        # Add up LLHs for each single LLH
        for key, llh in self._llhs.items():
            ns_w = self._ns_weights[key]
            ts_i, ns_grad_i = llh.lnllh_ratio(ns=ns * ns_w, X=X[key])
            ts += ts_i
            ns_grad += ns_grad_i * ns_w  # Chain rule

        return ts, ns_grad

    def fit_lnllh_ratio(self, ns0, X):
        """
        Fit single ns parameter simultaneously for all LLHs.
        """
        def _neglnllh(ns, sob_dict, nevts_dict):
            """ Multi LLH wrapper directly using a dict of sob values """
            ts = 0.
            ns_grad = 0.
            for key, sob in sob_dict.items():
                ts_i, ns_grad_i = self._llhs[key]._lnllh_ratio(
                    ns * self._ns_weights[key], sob, nevts_dict[key])
                ts -= ts_i
                ns_grad -= ns_grad_i * self._ns_weights[key]  # Chain rule
            return ts, ns_grad

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
        nevts_dict = dict_map(lambda k, v: len(v), X)

        res = sco.minimize(fun=_neglnllh, x0=[ns0],
                           jac=True, args=(sob_dict, nevts_dict),
                           bounds=[self._llh_opts["ns_bounds"]],
                           method=self._llh_opts["minimizer"],
                           options=self._llh_opts["minimizer_opts"])
        ns, ts = res.x[0], -res.fun[0]
        if ts < 0.:
            # Some times the minimizer doesn't go all the way to 0., so
            # TS vals might end up negative for a truly zero fit result
            ts = 0.
        return ns, ts

    def _ns_split_weights(self, llhs, nsrcs):
        """
        Set up the ``ns`` splitting weights that allow to fit a global ns in the
        multi LLH case, by splitting the expected signal portion to each sample.

        wj = \sum_{k=1}^{N_\text{srcs}} (P(j | k) * P(k))

        where

            P(k): Relative detection efficiency of src k, normalized over all
                  sources. This is obtained by summing each unnormalized
                  P(j | k) over j and then normalize over all k terms.
            P(j | k): Relative detection efficiency of sample j for source k,
                      normalized over all samples for each source. This is
                      obtained by getting all unnormalized source weights for
                      each sample and then normalizing per source over all
                      samples.

        This can also be written as a matrix equation

            |w_1,  |   |P(j=1,k=0)    ... P(j=1,k=nsrcs)   |   |P(k=0)    |
            |...,  | = |...           ... ...              | * |...       |
            |w_nsam|   |P(j=nsam,k=0) ... P(j=nsam,k=nsrcs)|   |P(k=nsrcs)|

        where the matrix columns are normalized and the P(k) vector is
        normalized separately.

        In code:
            * p_k := P(k), shape (nsrcs,)
            * p_kj := P(j | k), shape (nsamples, nsrcs)

        Parameters
        ----------
        llhs : dict of LLH instances
            Single LLH instances that shall be combined.

        Returns
        -------
        ns_weigths : dict of array-like
            Weight per LLH to split up ``ns`` among different samples.
        """
        nsamples = len(llhs)

        # First get all unnormalized P(k | j)
        p_kj = np.empty((nsamples, nsrcs), dtype=float)
        for j, key in enumerate(llhs.keys()):
            args = llhs[key].model.get_args()
            p_kj[j] = args["src_w_dec"] * args["src_w_theo"]

        # Get the unnormalized P(k)
        p_k = np.sum(p_kj, axis=1)

        # Normalize cols of P(k | j) and P(k) vector
        p_kj = p_kj / np.sum(p_kj, axis=0)
        p_k = p_k / np.sum(p_k)

        # Store ns weight per sample
        ns_weights = {}
        for j, key in enumerate(llhs.keys()):
            ns_weights[key] = np.sum(p_kj[j] * p_k)

        assert np.isclose(np.sum(ns_weights.values()), 1.)
        return ns_weights
