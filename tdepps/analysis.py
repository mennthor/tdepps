# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import range, int
from future import standard_library
standard_library.install_aliases()                                              # noqa

import numpy as np
from numpy.lib.recfunctions import append_fields

from tdepps.llh import GRBLLH
from tdepps.utils import fill_dict_defaults, flatten_list_of_1darrays


class TransientsAnalysis(object):
    """
    Providing methods to do a transients analysis.

    Parameters
    ----------
    srcs : recarray, shape (nsrcs)
        Source properties, must have names:

        - 'ra', float: Right-ascension coordinate of each source in radian in
          intervall :math:`[0, 2\pi]`.
        - 'dec', float: Declinatiom coordinate of each source in radian in
          intervall :math:`[-\pi / 2, \pi / 2]`.
        - 't', float: Time of the occurence of the source event in MJD days.
        - 'dt0', 'dt1': float: Lower/upper border of the time search window in
          seconds, centered around each source time `t`.
        - 'w_theo', float: Theoretical source weight per source, eg. from a
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
            raise ValueError("`llh` must be an instance of " +
                             "`tdepps.llh.GRBLLH`.")

        self._srcs = srcs
        self._llh = llh
        return

    @property
    def srcs(self):
        return self._srcs

    @property
    def llh(self):
        return self._llh

    def do_trials(self, n_trials, ns0, bg_inj, bg_rate_inj, signal_inj=None,
                  minimizer_opts=None):
        """
        Do pseudo experiment trials using only background-like events from the
        given event injectors.

        We need to build the background (or null hypothesis) test statistic (TS)
        to estimate the signifiance of a TS result on data.
        We can either sample the TS very often to populate even the range beyond
        5 sigma with sufficent statistics or we can fit an appropriate model
        function to fewer samples, hoping that it decribes the TS well enough.

        Parameters
        ----------
        n_trials : int
            Number of trials to perform.
        ns0 : float
            Fitter seed for the fit parameter ns: number of signal events that
            we expect at the source locations.
        bg_inj : `tdepps.bg_injector` instance
            Injector to generate background-like pseudo events.
        bg_rate_inj : `tdepps.bg_rate_injector` instance
            Injector to generate the times of background-like pseudo events.
        signal_inj : `tdepps.signal_injector.sample` generator, optional
            Injector generator to generate signal events. If None, pure
            background trials are done. (default: None)
        minimizer_opts : dict, optional
            Options passed to `scipy.optimize.minimize` [1] using the 'L-BFGS-B'
            algorithm. If specific key is not given or argument is None, default
            values are set to:

            - 'bounds', array-like, shape (1, 2): Bounds `[[min, max]]` for
              `ns`. Use None for one of min or max when there is no bound in
              that direction. (default: `[[0, None]]`)
            - 'ftol', float: Minimizer stops when the absolute tolerance of the
              function value is `ftol`. (default: 1e-12)
            - 'gtol', float: Minimizer stops when the absolute tolerance of the
              gradient component is `gtol`. (default: 1e-12)
            - maxiter, int: Maximum fit iterations the minimiter performs.
              (default: 1e3)

            (default: None)

        Returns
        -------
        res : record-array, shape (n_trials)
            Best fit parameters and test statistic for each nonzero trial.
            Has keys:

            - 'ns': Best fit values for number of signal events.
            - 'TS': Test statisitc for each trial.

        nzeros : int
            How many trials with `ns = 0` and `TS = 0` occured. This is done to
            save memory, because usually a lot of trials are zero.
        """
        # Setup minimizer defaults and bounds
        if minimizer_opts is None:
            minimizer_opts = {}

        bounds = minimizer_opts.pop("bounds", [[0., None]])

        required_keys = []
        opt_keys = {"ftol": 1e-12,
                    "gtol": 1e-12,
                    "maxiter": int(1e3)}
        minopts = fill_dict_defaults(minimizer_opts, required_keys, opt_keys,
                                     noleft=False)
        assert len(minopts) >= len(opt_keys)

        # Prepare fixed source parameters for injectors
        src_t = self._srcs["t"]
        src_dt = np.vstack((self._srcs["dt0"], self._srcs["dt1"])).T

        # Total injection time window in which the time PDF is defined and
        # nonzero.
        trange = self.llh.time_pdf_def_range(src_t, src_dt)
        assert len(trange) == len(self._srcs)
        assert trange.shape == (len(self._srcs), 2)

        # Number of expected background events in each given time frame
        nb = bg_rate_inj.get_nb(src_t, trange)
        assert len(nb) == len(self._srcs)
        assert nb.shape == (self._srcs.shape)

        # Create args and do trials
        X_names = ["ra", "dec", "timeMJD", "logE", "sigma"]
        dtype = [(n, np.float) for n in X_names]
        args = {"nb": nb, "srcs": self._srcs}
        nzeros = 0
        ns, TS = [], []
        for i in range(n_trials):
            # Inject events from given injectors
            times = bg_rate_inj.sample(src_t, trange, poisson=True)
            times = flatten_list_of_1darrays(times)
            nevts = len(times)

            # If we have no events, fit will be zero
            if nevts == 0:
                nzeros += 1
                continue

            if signal_inj is not None:
                nsig, Xsig = next(signal_inj)

            X = np.empty((nzeros,), dtype=dtype)

            # Else ask LLH what value we have
            X = bg_inj.sample(nevts)
            X = append_fields(X, "timeMJD", times, dtypes=np.float,
                              usemask=False)

            # Only store the best fit params and the TS value if nonzero
            _ns, _TS = self.llh.fit_lnllh_ratio(X, ns0, args, bounds,
                                                minimizer_opts)
            if (_ns == 0) and (_TS == 0):
                nzeros += 1
            else:
                ns.append(_ns)
                TS.append(_TS)

        # Make output record array for non zero trials
        res = np.empty((n_trials - nzeros,),
                       dtype=[("ns", np.float), ("TS", np.float)])
        res["ns"] = np.array(ns)
        res["TS"] = np.array(TS)

        return res, nzeros
