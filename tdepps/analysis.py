import numpy as np
import scipy.optimize as sco

from tdepps.llh import GRBLLH

from .utils import fill_dict_defaults, flatten_list_of_1darrays


class TransientsAnalysis(object):
    """
    Providing methods to do a transients analysis.

    Parameters
    ----------
    srcs : recarray, shape (n_srcs)
        Source properties, must have names:

        - "ra", float: Right-ascension coordinate of each source in radian in
          intervall [0, 2pi].
        - "dec", float: Declinatiom coordinate of each source in radian in
          intervall [-pi/2, pi/2].
        - "t", float: Time of the occurence of the source event in MJD days.
        - "dt0", "dt1": float: Lower/upper border of the time search window in
          seconds, centered around each source time `t`.
        - "w_theo", float: Theoretical source weight per source, eg. from a
          known gamma flux.

    llh : `tdepps.LLH.GRBLLH` instance
        LLH function used to test the hypothesis, that signal neutrinos have
        been measured accompaning a source event occuring only for a limited
        amount of time, eg. a gamma ray burst (GRB).
    """
    def __init__(self, srcs, llh):
        required_names = ["ra", "dec", "t", "dt0", "dt1", "w_theo"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` is missing name '{}'.".format(n))

        if not isinstance(llh, GRBLLH):
            raise ValueError("`llh` must be an instance of tdepps.llh.GRBLLH.")

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
        theta0 : dict
            Seeds for the parameter set {"par_name": value} to evaluate the
            ln-LLH ratio at.
            Here GRBLLH depends on:

            - "ns": Number of signal events that we want to fit.

        bg_inj : `tdepps.bg_injector` instance
            Injector to generate background-like pseudo events.
        bg_rate_inj : `tdepps.bg_rate_injector` instance
            Injector to generate the times of background-like pseudo events.
        signal_inj : `tdepps.signal_injector` instance, optional
            Injector to generate signal events. If None, pure background trials
            are done. (default: None)

        Returns
        -------
        res : list
            Fit results from each trial.
        """
        if signal_inj is not None:
            raise NotImplementedError("Signal injection not yet implemented.")

        def get_pseudo_events(src_idx):
            _X = []
            times = []
            rnd_ra = []
            args = []

            for src_idx in range(self.n_srcs):
                # Samples times and thus number of bg expectated events
                _t = self.srcs["t"][src_idx]
                dt = [self.srcs["dt0"][src_idx], self.srcs["dt1"][src_idx]]
                _times = bg_rate_inj.sample(t=_t, trange=dt, ntrials=1)[0]
                nb = len(_times)
                times.append(_times)
                args.append({"nb": nb})
                # Sample rest of features
                if nb > 0:
                    _X.append(bg_inj.sample(n_samples=nb))
                    rnd_ra.append(np.random.uniform(0, 2. * np.pi, size=nb))

            names = ["ra", "sinDec", "timeMJD", "logE", "sigma"]
            dtype = [(n, t) for (n, t) in zip(names, len(names) * [np.float])]
            nb_tot = np.sum([d["nb"] for d in args])
            X = np.empty((nb_tot, ), dtype=dtype)
            # Make output array in compatible format
            _X = flatten_list_of_1darrays(_X)
            X["ra"] = flatten_list_of_1darrays(rnd_ra)
            X["sinDec"] = np.sin(_X[:, 1])
            X["logE"] = _X[:, 0]
            X["sigma"] = _X[:, 2]
            X["timeMJD"] = flatten_list_of_1darrays(times)

            return X, args

        res = []
        for i in range(n_trials):
            # Inject background-like events
            X, args = get_pseudo_events()
            res.append(self.fit_lnllh_ratio_params(X, theta0, args, **kwargs))

        return res

    def fit_lnllh_ratio_params(self, X, theta0, args, **kwargs):
        """
        Fit LLH parameters for a given set of data and parameters.

        Parameters
        ----------
        X : record-array
            Fixed data set the LLH depends on. dtypes are ["name", type].
            Here `X` must have names:

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
            ln-LLH at. Here GRBLLH depends on:

            - "ns": Number of signal events that we want to fit.

        args : dict
            Other fixed parameters {"par_name": value}, the LLH depents on.
            Here `args` must have keys:

            - "ns", array-like: Number of expected background events for each
              sources time window.

        kwargs : optional
            Keyword arguments are passed to `scipy.optimize.minimize` [1] using
            the "L-BFGS-B" algorithm. Explicitly set default values are:

            - bounds: None (given bound must be of shape (nparams, 2))
            - ftol: 1e-12 (absolute tolerance of the function value)
            - gtol: 1e-12 (absolute tolerance of one gradient component)
            - maxiter: int(1e5) (Maximum fit iterations)

        Returns
        -------
        res : `scipy.optimise.OptimizationResult`
            Holding the information of the fit. Get best fit point with `res.x`.

        Notes
        -----
        .. [1] https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html_minimize.py#L36-L466 # noqa
        """
        def _llh(theta):
            """
            Wrapper for the LLH function to put params in a dict and returning
            the negative ln-LLH ratio for the minimizer.

            Parameters
            ----------
            theta : array-like
                The parameters to be fitted. Order is fixed by the initial seed.

            Returns
            -------
            lnllh : float
                Value of the ln-LLH ratio at a single set of parameters x.
            lnllh_grad : array-like, shape (len(x))
                Analytic gradient of the ln-LLH ratio for the minimizer.
            """
            theta = {"ns": theta[0]}
            lnllh, lnllh_grad = self.llh.lnllh_ratio(X, theta, args)
            return -1. * lnllh, -1. * lnllh_grad

        # Check if args list is OK.
        required_keys = ["nb"]
        opt_keys = {}
        args = fill_dict_defaults(args, required_keys, opt_keys)

        nb = np.atleast_1d(args["nb"])
        if len(nb) != self.n_srcs:
            raise ValueError("We need a background epectation for each source.")
        args["nb"] = nb

        # Put sources to args
        args["srcs"] = self.srcs

        # Setup minimizer defaults and put rest of kwargs in options dict
        bounds = kwargs.pop("bounds", None)
        ftol = kwargs.pop("ftol", 1e-12)
        gtol = kwargs.pop("gtol", 1e-12)
        maxiter = kwargs.pop("maxiter", int(1e5))
        fit_options = {"ftol": ftol, "gtol": gtol, "maxiter": maxiter}
        for key, val in kwargs.items():
            fit_options[key] = val

        # Wrap up seed and Write it, cut it, paste it, save it,
        #                  Load it, check it, quick, let's fit it
        theta0 = [val for val in theta0.values()]
        res = sco.minimize(fun=_llh, x0=theta0, jac=True, bounds=bounds,
                           method="L-BFGS-B", options=fit_options)

        return res
