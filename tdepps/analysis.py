# coding: utf-8

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()
import numpy as np
from numpy.lib.recfunctions import append_fields
import scipy.optimize as sco
from sklearn.utils import check_random_state


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
                  random_state=None, minimizer_opts=None):
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
        random_state : RandomState, optional
            Turn seed into a `np.random.RandomState` instance. Method from
            `sklearn.utils`. Can be None, int or RndState. (default: None)
        minimizer_opts : dict
            See `fit_lnllh_ratio_params`, Parameters: Options passed to
            `scipy.optimize.minimize` [1] using the "L-BFGS-B" algorithm.
            If key 'bounds' is not explicitely given or None, it is set to
            [0, 2 * nevts_injected] for parameter ns in each trial.

        Returns
        -------
        res : record-array, shape (n_trials)
            Best fit parameters and test statistic for each nonzero trial.
            Has keys:

            - "ns": Best fit values for number of signal events.
            - "TS": Test statisitc for each trial.

        nzeros : int
            How many trials with ns = 0 and TS = 0 occured. This is done to save
            memory, because really a lot of trials are zero.
        """
        if signal_inj is not None:
            raise NotImplementedError("Signal injection not yet implemented.")

        rndgen = check_random_state(random_state)

        # Prepare fixed source parameters for injectors
        srcs = self.srcs
        src_t = srcs["t"]
        src_dt = np.vstack((srcs["dt0"], srcs["dt1"])).T

        # Total injection time window in which the time PDF is defined and
        # nonzero.
        trange = self.llh.get_injection_trange(src_t, src_dt)

        # Number of expected background events in each given time frame
        nb = bg_rate_inj.get_nb(src_t, trange)
        args = {"nb": nb}

        ns, TS = [], []
        nzeros = 0
        for i in range(n_trials):
            # Inject events from given injectors
            times = bg_rate_inj.sample(src_t, trange, poisson=True,
                                       random_state=rndgen)
            times = flatten_list_of_1darrays(times)
            nevts = len(times)

            X = bg_inj.sample(nevts, random_state=rndgen)
            X = append_fields(X, "timeMJD", times, dtypes=np.float,
                              usemask=False)

            # Only store the best fit params and the TS value if nonzero
            _res = self.fit_lnllh_ratio_params(X, theta0, args, minimizer_opts)
            xmin, fmin = _res.x[0], -1. * _res.fun
            if (xmin == 0) and (fmin == 0):
                nzeros += 1
            else:
                ns.append(_res.x[0])
                TS.append(-1. * _res.fun)

        # Make output record array for non zero trials
        res = np.empty((n_trials - nzeros,),
                       dtype=[("ns", np.float), ("TS", np.float)])
        res["ns"] = np.array(ns)
        res["TS"] = np.array(TS)

        return res, nzeros

    def fit_lnllh_ratio_params(self, X, theta0, args, minimizer_opts=None):
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

        minimizer_opts : dict, optional
            Options passed to `scipy.optimize.minimize` [1] using the "L-BFGS-B"
            algorithm. Explicitly set default values are:

            - bounds: None (given bound must be of shape (nparams, 2))
            - ftol: 1e-12 (absolute tolerance of the function value)
            - gtol: 1e-12 (absolute tolerance of one gradient component)
            - maxiter: int(1e5) (Maximum fit iterations)

            (default: None)

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

        # Setup minimizer defaults
        if minimizer_opts is None:
            minimizer_opts = {}
        else:
            minopts = minimizer_opts.copy()
        bounds = minopts.pop("bounds", None)
        ftol = minopts.pop("ftol", 1e-12)
        gtol = minopts.pop("gtol", 1e-12)
        maxiter = minopts.pop("maxiter", int(1e5))
        fit_options = {"ftol": ftol, "gtol": gtol, "maxiter": maxiter}
        for key, val in minopts.items():
            fit_options[key] = val  # Let scipy handle leftover options

        # Wrap up seed and Write it, cut it, paste it, save it,
        #                  Load it, check it, quick, let's fit it
        theta0 = [val for val in theta0.values()]
        res = sco.minimize(fun=_llh, x0=theta0, jac=True, bounds=bounds,
                           method="L-BFGS-B", options=fit_options)

        return res
