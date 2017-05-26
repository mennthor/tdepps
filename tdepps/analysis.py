import numpy as np
import scipy.optimize as sco

from tdepps.llh import GRBLLH

from anapymods3.general.misc import fill_dict_defaults


class TransientsAnalysis(object):
    def __init__(self, srcs, llh):
        """
        Providing methods to do a transients analysis.

        Parameters
        ----------
        srcs : recarray, shape (n_srcs)
            Source properties, must have names:
            - "ra", float: Right-ascension coordinate of each source in radian
              in intervall [0, 2pi].
            - "dec", float: Declinatiom coordinate of each source in radian in
              intervall [-pi/2, pi/2].
            - "t", float: Time of the occurence of the source event in MJD days.
            - "dt0", "dt1": float: Lower/upper border of the time search window
              in seconds, centered around each source time `t`.
        llh : `tdepps.LLH.GRBLLH` instance
            LLH function used to test the hypothesis, that additional neutrinos
            have been measured accompaning a source event occuring only for a
            limited amount of time, eg. a gamma ray burst (GRB).
        """
        required_names = ["ra", "dec", "t", "dt0", "dt1"]
        if not all([name in required_names for name in srcs.dtype.names]):
            raise ValueError("'srcs' is missing required names.")

        if not isinstance(llh, GRBLLH):
            raise ValueError("'llh' must be an instance of tdepps.llh.GRBLLH.")

        self.srcs = srcs
        self.n_srcs = len(srcs)
        self.llh = llh
        return

    def do_trials(self, n_trials, theta0,
                  bg_inj, bg_rate_inj, signal_inj=None,
                  **kwargs):
        """
        Do pseudo experiment trials using the given event injectors.

        Parameters
        ----------
        n_trials : int
            Number of trials to perform.
        heta0 : dict
            Seeds for the parameter set {"par_name": value} to evaluate the
            ln-LLH at.
            Here GRBLLH depends on:

            - "ns": Number of signal events that we want to fit.

        bg_inj : `tdepps.bg_injector` instance
            Injector to generate background-like pseudo events.
        bg_rate_inj : `tdepps.bg_rate_injector` instance
            Injector to generate the times of background-like pseudo events.
        signal_inj : `tdepps.signal_injector` instance, optional
            Injector to generate signal events. If None, pure background trials
            are done. (default: None)
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
        Fit LLH parameters for a given set of data and parameters.

        Parameters
        ----------
        X : record-array
            Fixed data set the LLH depends on. dtypes are ["name", type].
            Here `X` must have keys:

            - "timeMJD": Per event times in MJD days.
            - "ra", "sinDec": Per event right-ascension positions in equatorial
              coordinates, given in radians and sinus declination in [-1, 1].
            - "logE": Per event energy proxy, given in log10(1/GeV).
            - "sigma": Per event positional uncertainty, given in radians. It is
              assumed, that a circle with radius `sigma` contains approximatly
              :math:`1\sigma` (~0.39) of probability of the reconstrucion
              likelihood space.
        theta0 : dict
            Seeds for the parameter set {"par_name": value} to evaluate the
            ln-LLH at.
            Here GRBLLH depends on:

            - "ns": Number of signal events that we want to fit.

        args : dict
            Other fixed parameters {"par_name": value}, the LLH depents on.
            Here `args` must have keys:

            - "ns": Number of expected background events in the time window.
        """
        def _llh(x, weights):
            """
            Wrapper for the LLH function.

            We need to minimize and wrap up params in a dict.
            Also we can test multiple srcs at once by summing the LLHs.

            Parameters
            ----------
            x : array-like
                The parameters to be fitted. Order is fixed by the initial seed.
            weights : array-like
                Weights for the `ns` parameter when fitting multiple sources at
                once. Weights are expected number of background events.

            Returns
            -------
            lnllh : float
                Value of the ln-LLH ratio at a single set of parameters x.
            lnllh_grad : array-like, shape (len(x))
                Analytic gradient of the ln-LLH ratio for the minimizer.
            """
            # Only for readability
            ns = x[0]

            # For multiple sources we sum up each contribution
            lnllh = 0
            lnllh_grad = np.zeros(len(x), dtype=np.float)
            for i in range(self.n_srcs):
                # Each source contributes a certain amount ot the total ns
                theta = {"ns": ns * weights[i]}
                f, g = self.llh.lnllh_ratio(X, theta, args[i])
                lnllh += f
                lnllh_grad += weights[i] * g  # Chain rule: g(ax)' = a*g'(ax)

            return -1. * lnllh, -1. * lnllh_grad

        # Check if args list is OK. Need arg dict for each src
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
