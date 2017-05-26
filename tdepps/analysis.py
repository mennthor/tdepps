import numpy as np
import scipy.optimize as sco

from anapymods3.general.misc import fill_dict_defaults


class TransientsAnalysis(object):
    def __init__(self, srcs, llh):
        """
        Methods to do a transients analysis.

        srcs : recarray
            Source properties: Position, time, time window, weight(?), size(?).
        llh : `tdepps.LLH` instance
            LLH function to test against.
        """
        self.srcs = srcs
        self.n_srcs = len(srcs)
        self.llh = llh
        return

    def do_trials(self, n_trials, theta0,
                  bg_inj=None, bg_rate_inj=None, signal_inj=None,
                  **kwargs):
        """
        Do trials using the given event injectors.

        n_trials : int
            How many trials to perform.
        theta0 : dict
            Seeds for the parameter set {"par_name": value} to evaluate the
            ln-LLH at.
        *_inj : `tdepps.*_injector` instance
            INjector to generate events per trials.
        """
        if signal_inj is not None:
            raise NotImplementedError("Signal injection not yet implemented.")

        # These are set by the event injectors later
        X = None
        args = None

        res = np.empty(n_trials, dtype=np.float)
        for i in range(n_trials):
            res[i] = self.fit_lnllh_ratio_params(X, theta0, args, **kwargs)

        return res

    def fit_lnllh_ratio_params(self, X, theta0, args, **kwargs):
        """
        Fit LLH parameters for a given set of data and args.

        X : recarray
            Event data: Positons, time, angular uncertainty, energy.
        theta0 : dict
            Seeds for the parameter set {"par_name": value} to evaluate the
            ln-LLH at.
        args : dict
            Other fixed parameters {"par_name": value}, the LLH depents on,
            except the src parameter, given at class creation.
        """
        def _llh(x, weights):
            """
            Wrap LLH function. We need to minimize and wrap up params in a dict.
            Also we can test multiple srcs at once by summing the LLHs.
            """
            ns = x[0]

            lnllh = 0
            lnllh_grad = np.zeros(len(x), dtype=np.float)
            for i in range(self.n_srcs):
                theta = {"ns": ns * weights[i]}
                f, g = self.llh.lnllh_ratio(X, theta, args[i])
                lnllh += f
                lnllh_grad += weights[i] * g

            return -1. * lnllh, -1. * lnllh_grad

        # Check if args are OK
        if len(args) != self.n_srcs:
            raise ValueError("Must provide an 'arg' dict for each source.")
        for i in range(self.n_srcs):
            required_keys = ["nb"]
            opt_keys = {}
            args[i] = fill_dict_defaults(args[i], required_keys, opt_keys)

        # Setup src info in args list to be used by LLH
        for i in range(self.n_srcs):
            args[i]["dt"] = [self.srcs[i]["dt0"], self.srcs[i]["dt1"]]
            args[i]["src_t"] = self.srcs[i]["t"]
            args[i]["src_ra"] = self.srcs[i]["ra"]
            args[i]["src_dec"] = self.srcs[i]["dec"]

        # Setup minimizer
        bounds = kwargs.pop("bounds", None)
        theta0 = [val for val in theta0.values()]

        # When using multiple srcs, we weight ns with the expected number of BG
        # events to get the correct number of ns for each src window.
        weights = np.array([arg["nb"] for arg in args])
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:  # Else uniform weights
            weights = np.ones_like(weights) / self.n_srcs

        res = sco.minimize(fun=_llh, x0=theta0, jac=True, args=weights,
                           bounds=bounds, **kwargs)

        return res
