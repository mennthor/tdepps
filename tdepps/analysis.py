# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import range, int, next, zip
from future import standard_library
from future.utils import viewkeys
standard_library.install_aliases()

import numpy as np
from numpy.lib.recfunctions import append_fields, stack_arrays
import scipy.stats as scs
import scipy.optimize as sco

from tdepps.llh import GRBLLH, MultiSampleGRBLLH
from tdepps.utils import (fill_dict_defaults, weighted_cdf,
                          make_ns_poisson_weights)

import sys
if any("jupyter" in arg for arg in sys.argv):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class TransientsAnalysis(object):
    def __init__(self, srcs, llh):
        """
        Providing methods to do a transients analysis.

        Parameters
        ----------
        srcs : dict or dict of recarrays, shape (nsrcs)
            Source properties, must have names:

            - 'ra', float: Right-ascension coordinate of each source in radian
              in intervall :math:`[0, 2\pi]`.
            - 'dec', float: Declinatiom coordinate of each source in radian in
              intervall :math:`[-\pi / 2, \pi / 2]`.
            - 't', float: Time of the occurence of the source event in MJD days.
            - 'dt0', 'dt1': float: Lower/upper border of the time search window
              in seconds, centered around each source time `t`.
            - 'w_theo', float: Theoretical source weight per source, eg. from a
              known gamma flux.

            If given as dictionary each value holds the source information for a
            different sample. ``llh`` must then be of type ``MultiLLH``.
        llh : ``tdepps.LLH.GRBLLH`` or ``MultiSampleGRBLLH`` instance
            LLH function used to test the hypothesis, that signal neutrinos have
            been measured accompaning a source event occuring only for a limited
            amount of time, eg. a gamma ray burst (GRB).
            If ``srcs`` is given as a dict, ``llh`` must be of type
            ``MultiSampleGRBLLH``.
        """
        if not isinstance(srcs, dict):  # Temporary work with dicts
            srcs = {-1: srcs}

        # If we have a multi LLH, keys must match
        if isinstance(llh, MultiSampleGRBLLH):
            if viewkeys(llh._llhs) != viewkeys(srcs):
                raise ValueError("`llh` is a `tdepps.llh.MultiSampleGRBLLH` " +
                                 "but sample names in `llh` and keys in " +
                                 "`srcs` don't match.")
        elif not isinstance(llh, GRBLLH):
            raise ValueError("`llh` must be an instance of " +
                             "`tdepps.llh.GRBLLH` or " +
                             "`tdepps.llh.MultiSampleGRBLLH`.")

        # Check each source recarray's names and test the time windows
        required_names = ["ra", "dec", "t", "dt0", "dt1", "w_theo"]
        _allsrcs = np.empty(0, dtype=[(n, np.float) for n in required_names])
        for n in required_names:
            for key, srcs_i in srcs.items():
                if n not in srcs_i.dtype.names:
                    raise ValueError("Source recarray '" +
                                     "{}' is missing name '{}'.".format(key, n))
                    _allsrcs = stack_arrays(_allsrcs, srcs_i)

        # Test mutual exclusiveness of all source time windows
        t, dt0, dt1 = _allsrcs["t"], _allsrcs["dt0"], _allsrcs["dt1"]
        exclusive = (((t + dt0)[:, None] >= t + dt1) |
                     ((t + dt1)[:, None] <= t + dt0))
        # Fix manually that time windows overlap with themselves
        np.fill_diagonal(exclusive, True)
        # If any entry is False, we have an overlapping window case
        if np.any(exclusive is False):
            window_ids = [[x, y] for x, y in zip(*np.where(~exclusive))]
            raise ValueError("Overlapping time windows: {}.".format(
                ", ".join(["[{:d}, {:d}]".format(*c) for c in
                          window_ids])) + "\nThis is not supported.")

        self._llh = llh
        if (len(srcs) == 1) and (srcs.keys()[0] == -1):
            self._srcs = srcs[-1]
            self._multi = False
        else:
            self._srcs = srcs
            self._multi = True

        return

    @property
    def srcs(self):
        return self._srcs

    @property
    def llh(self):
        return self._llh

    def do_trials(self, n_trials, ns0, bg_inj, bg_rate_inj, signal_inj=None,
                  minimizer_opts=None, full_out=False, verb=False):
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

        full_out : bool, optional
            If ``True`` also return number of injected signal events per trial.
            (default: ``False``)
        verb : bool, optional
            If ``True`` show iteration status with ``tqdm``.
            (default: ``False``)

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
        nsig_all : array
            Only if ``full_out==True`` also return the number of injected signal
            events per trial.
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
        if self._multi:
            src_t = {}
            src_dt = {}
            for key, src_i in self._srcs.items():
                src_t[key] = src_i["t"]
                src_dt[key] = np.vstack((src_i["dt0"], src_i["dt1"])).T
        else:
            src_t = self._srcs["t"]
            src_dt = np.vstack((self._srcs["dt0"], self._srcs["dt1"])).T

        # Total injection time window in which the time PDF is defined and
        # nonzero.
        trange = self._llh.time_pdf_def_range(src_t, src_dt)
        assert len(trange) == len(self._srcs)

        # Number of expected background events in each given time frame
        nb = bg_rate_inj.get_nb(src_t, trange)
        assert len(nb) == len(self._srcs)

        # Create args and do trials
        if self._multi:
            args = {name: {"nb": nb[name], "srcs": self._srcs[name]} for name in
                    self._srcs.keys()}
        else:
            args = {"nb": nb, "srcs": self._srcs}

        # Select iterator depending on `verb` keyword
        if verb:
            trial_iter = tqdm(range(n_trials))
        else:
            trial_iter = range(n_trials)

        nzeros = 0
        ns, TS = [], []
        nsig_all = []
        for i in trial_iter:
            # Inject events from given injectors
            times = bg_rate_inj.sample(src_t, trange, poisson=True)
            if self._multi:
                nevts_split = {n: len(times_i) for n, times_i in times.items()}
                nevts = np.sum([len(times_i) for times_i in times.values()])
                if nevts > 0:
                    X = bg_inj.sample(nevts_split)
                    for key, arr in X.items():
                        X[key] = append_fields(arr, "timeMJD", times[key],
                                               dtypes=np.float, usemask=False)

                if signal_inj is not None:
                    nsig, Xsig, _ = next(signal_inj)
                    nsig_all.append(nsig[0])
                    nevts += nsig[0]
                else:
                    Xsig = None

                # If we have no events at all, fit will be zero
                if nevts == 0:
                    nzeros += 1
                    continue

                # Else ask LLH what value we have
                if Xsig is not None:
                    for key, arr in Xsig.items():
                        X[key] = stack_arrays((X[key], arr), usemask=False)
            else:
                times = np.concatenate(times, axis=0)
                nevts = len(times)

                if nevts > 0:
                    X = bg_inj.sample(nevts)
                    X = append_fields(X, "timeMJD", times, dtypes=np.float,
                                      usemask=False)

                if signal_inj is not None:
                    nsig, Xsig, _ = next(signal_inj)
                    nsig_all.append(nsig[0])
                    nevts += nsig[0]
                else:
                    Xsig = None

                # If we have no events at all, fit will be zero
                if nevts == 0:
                    nzeros += 1
                    continue

                # Else ask LLH what value we have
                if Xsig is not None:
                    X = stack_arrays((X, Xsig), usemask=False)

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

        if full_out:
            return res, nzeros, np.array(nsig_all)
        else:
            return res, nzeros

    def performance(self, ts_val, beta, bg_inj, bg_rate_inj, signal_inj,
                    mu0=-0.1, ntrials=100, tol_perc_err=5e-3, tol_mu_rel=1e-3,
                    maxloops=100, minimizer_opts=None, verb=False):
        """
        Iteratively search for the best fit `mu`, so that a fraction `beta` of
        the scaled PDF lies above the background test statistic value `ts_val`.

        Performance search on a PDF parameter `mu` which is the expectation
        value for a poisson PDF via a second variable defining a test statistic
        which is directly influenced by the choice of `mu`.

        Parameters
        ----------
        ts_val : float
            Test statistic value of the BG distribution, which is connected to
            the alpha value (Type I error).
        beta : float
            Fraction of alternative hypothesis PDF that should lie right of the
            `ts_val`.
        bg_inj : `tdepps.bg_injector` instance
            Injector to generate background-like pseudo events.
        bg_rate_inj : `tdepps.bg_rate_injector` instance
            Injector to generate the times of background-like pseudo events.
        signal_inj : `tdepps.signal_injector.sample` generator
            Injector generator to generate signal events.
        minimizer_opts : dict, optional
            See :py:meth:`do_trials`, Parameters
        mu0 : float, optional
            Seed value to begin the minimization at. If ``< 0`` a region close
            to the minimum is searched for automatically with few trials and
            increasing ``mu`` by ``mu += abs(mu0)``.
            If ``> 0`` this value is taken to start the minimization at.
            (default: ``-0.1``)
        ntrials : int, optional
            How many new trials to make per new iteration. (default: 100)
        tol_perc_err, tol_mu_rel : float, optional
            The iteration stops when BOTH of the following conditons are met:

            - The error on the estimated percentile for the current best fit
              ``mu`` is ``errors[-1] <= tol_perc_err`` AND
            - The relative difference in the best fit ``mus`` is
              ``abs(mus[-1]-mus[-2])/mus[-1]<= tol_mu_rel``.

            Furthermore the conditions must be met in both the last and second
            to last trial loops to avoid a break on accidental fluctuations.
            (default: tol_perc_err: 5e-3, tol_mu_rel: 1e-3)
        maxloops : int, optional
            Break the minimization process after this many loops with ntrials
            trials each. (default: 100)
        verb : bool, optional
            If ``True``print convergence message during fit. (default: False)

        Returns
        -------
        res : dict
            Result dictionary with keys:

            - "mu_bf": Best fit mu, equal to mu[-1].
            - "mu": List of visited mu values during minimization.
            - "ts": List of generated TS values during minimization.
            - "ns": List of generared ns values during minimization.
            - "err": List of errors on the weighted TS percentile per iteration.
            - "perc": List of estimated TS percentiles per iteration.
            - "nloops": Number of iterations needed to converge.
            - "ninitloops": Number of initial scan iterations done.
            - "lastfitres": scipy.optimize.OptimizeResult of the last fit.
            - "converged": Boolean, if ``True`` fit converged within maxloops.
        """
        def loss(mu, ns, ts):
            """
            Logged least squares loss for percentile distance to beta. No
            gradient is returned, because is has a pole at the minimum and the
            minimizer doesn't like that.

            Parameters
            ----------
            mu : float
                Current expectation value for poisson PDF.
            ns : array-like
                ns values from all trials done so far.
            ts : array-like
                Test statistic values from all trials done so far.

            Returns
            -------
            loss : float and array-like
                Value of the loss function at the current mu.
            """
            # Reweight TS trials using the poisson statistics of ns
            perc, _ = get_perc_and_err(mu, ns, ts)
            return np.log10((perc - (1. - beta))**2)

        def append_batch_of_trials(n, mu, ns, ts):
            """Do n trials and append result to ns and ts arrays"""
            sig_gen = signal_inj.sample(mean_mu=mu, poisson=True)
            res, nzeros, nsig = self.do_trials(n, ns0=mu, bg_inj=bg_inj,
                                               bg_rate_inj=bg_rate_inj,
                                               signal_inj=sig_gen,
                                               minimizer_opts=minimizer_opts,
                                               full_out=True, verb=False)
            ns = np.concatenate((ns, nsig))
            ts = np.concatenate((ts, res["TS"], np.zeros(nzeros, dtype=float)))
            return ns, ts

        def get_perc_and_err(mu, ns, ts):
            """
            Get the percentile and its relative error for all trials under
            the current best fit mu.
            """
            w, _ = make_ns_poisson_weights(mu=mu, ns=ns)
            perc, err = weighted_cdf(x=ts, val=ts_val, weights=w)
            return perc, err

        # Keep track of progress
        mus = np.array([], dtype=np.float)
        ts = np.array([], dtype=np.float)
        ns = np.array([], dtype=np.int)
        errors = np.array([], dtype=np.float)
        percs = np.array([], dtype=np.float)
        n_init_loops = 0
        n_loops = 0
        converged = False

        # If no seed given, start initial scan to get close to the minimum
        if mu0 < 0:
            if verb:
                print("Starting intitial scan loops.")
            dmu = -mu0
            n_init_trials = 20  # Not too few but also not too many for 1st scan

            def frac_over_tsval(ts):
                """Fraction of 'n_init_trials' last trials above 'ts_val'"""
                if len(ts) < n_init_trials:
                    return 0.
                return (np.sum(ts[-n_init_trials:] > ts_val) /
                        n_init_trials)

            mu = 1.
            stop = False
            while not stop:
                ns, ts = append_batch_of_trials(n_init_trials, mu, ns, ts)

                # Save progress
                mus = np.append(mus, mu)
                # Err estimation is not reliable here, too few trials, so use
                # worst error = 1 for all trials
                perc, _ = get_perc_and_err(mu, ns, ts)
                errors = np.append(errors, 1.)
                percs = np.append(percs, perc)

                # Go above beta in the last two batches to have enough trials
                # above best fit
                if n_init_loops > 2:
                    stop = ((frac_over_tsval(ts) > beta) and
                            (frac_over_tsval(ts[:-n_init_trials]) > beta))

                mu += dmu
                n_init_loops += 1

            if verb:
                print("Made {} intitial scan loops with {} trials total".format(
                    n_init_loops, len(ts)))
        elif mu0 > 0.:
            # Init and do first batch of trials
            mu = mu0
            # mus = np.append(mus, mu)
            # errors = np.append(errors, 1.)
            # percs = np.append(percs, -1.)
            # ns, ts = append_batch_of_trials(ntrials, mu, ns, ts)
        else:
            raise ValueError("Seed `mu0` can be >0 or <0, but not == 0.")

        if verb:
            print("Starting main trials with mu0 = {}.".format(mu))

        # Process minimizer loop until last two rel. error are below tolerance
        if verb:
            trial_iter = tqdm(range(maxloops))
        else:
            trial_iter = range(maxloops)

        for i in trial_iter:
            # Make new batch of trials
            ns, ts = append_batch_of_trials(ntrials, mu, ns, ts)

            # Now fit the poisson expectation by reusing all (reweighted) trials
            # Bounds: 90% CL central interval around best mu
            bl, bu = scs.poisson.interval(0.90, mu)
            bounds = [bl, max(bu, 1.)]  # Avoid [0, 0] for small mu
            # Do a seed scan prior to fitting to avoid local minima
            seeds = np.arange(*bounds)
            seed = seeds[np.argmin([loss(mui, ns, ts) for mui in seeds])]

            res = sco.minimize(loss, [seed], bounds=[bounds], args=(ns, ts),
                               jac=False, method="L-BFGS-B",
                               options={"ftol": 100, "gtol": 1e-8})
            mu = res.x
            perc, err = get_perc_and_err(mu, ns, ts)

            # Save the progress
            mus = np.append(mus, mu)
            errors = np.append(errors, err)
            percs = np.append(percs, perc)

            # First build up neccessary statisitcs for the tests
            if n_loops > 2:
                # Make some manual tweaks to help the minimizer
                oldmu = mus[-2]
                # New fit suddenly more than 50% above old fit: Truncate change
                if (oldmu > 0.) and (np.abs(mu - oldmu) / oldmu > 0.5):
                    if mu > oldmu:
                        mu = 1.5 * oldmu
                    else:
                        mu = 0.5 * oldmu
                    err = errors[-1]  # Make sure we definitely do another trial
                # Fit is identically to previous fit
                if mu == oldmu:
                    mu = 1.1 * oldmu  # Larger mu: conservative -> more flux
                    err = errors[-1]

                # Err conditions must match in the last and last to last trials
                if (mus[-1] > 0.) and (mus[-2] > 0.):
                    mu_rel_err1 = np.abs(mus[-1] - mus[-2]) / mus[-1]
                    mu_rel_err2 = np.abs(mus[-2] - mus[-3]) / mus[-2]
                else:  # If errs can't be calculated, make sure we don't break
                    mu_rel_err1 = 10 * tol_mu_rel
                    mu_rel_err2 = 10 * tol_mu_rel
                if ((mu_rel_err1 <= tol_mu_rel) and
                        (mu_rel_err2 <= tol_mu_rel) and
                        (errors[-1] < tol_perc_err) and
                        (errors[-2] < tol_perc_err)):
                    if verb:
                        print("Break: below tol_mu_rel and tol_perc_err.")
                    converged = True
                    break

            n_loops += 1

        if (n_loops == maxloops) and verb:
            print("Manual break after {} loops with {} main ".format(
                  n_loops, n_loops * ntrials) + "trials: Reached " +
                  "`maxloops` loops.")

        return {"mu_bf": mus[-1], "mus": mus, "ts": ts, "ns": ns, "err": errors,
                "perc": percs, "nloops": n_loops, "ninitloops": n_init_loops,
                "lastfitres": res, "converged": converged}

    def performance_chi2(self, ts_val, beta, bg_inj, bg_rate_inj, signal_inj,
                         mus, par0=[1., 1., 1.], ntrials=1000, verb=False):
        """
        Make independent trials within given range and fit a ``chi2`` CDF to the
        resulting percentiles, becasue they are surprisingly well described by
        that.

        Parameters
        ----------
        ts_val : float
            Test statistic value of the BG distribution, which is connected to
            the alpha value (Type I error).
        beta : float
            Fraction of alternative hypothesis PDF that should lie right of the
            `ts_val`.
        bg_inj : `tdepps.bg_injector` instance
            Injector to generate background-like pseudo events.
        bg_rate_inj : `tdepps.bg_rate_injector` instance
            Injector to generate the times of background-like pseudo events.
        signal_inj : `tdepps.signal_injector.sample` generator
            Injector generator to generate signal events.
        mus : array-like
            How much mean poisson signal shall be injected.
        par0 : list, optional
            Seed values ``[df, loc, scale]`` for the ``chi2`` CDF fit.
            (default: ``[1., 1., 1.]``)
        ntrials : int, optional
            How many new trials to make per independent trial. (default: 1000)
        verb : bool, optional
            If ``True`` print progress message during fit. (default: False)

        Returns
        -------
        mu_bf : float
            Best fit poisson signal expectation derived from the fitted ``chi2``
            CDF.
        cdfs : array-like, len(mus)
            Calculated CDF values for each trial from the ``chi2`` function.
        TS : list of arrays
            For each bunch ``mu`` trials the resulting test satistic values.
            From these we can in principle  calculate other ``ts_val, beta``
            combinations by calculating new percentiles and refit the ``chi2``
            wthout doing more trials.
        best_pars : tuple, len(3)
            Best fit parameters from the ``chi2`` CDF fit.
        """
        if np.any(mus < 0):
            raise ValueError("`mus` must not have an entry < 0.")

        if verb:
            trial_iter = tqdm(mus)
        else:
            trial_iter = mus

        # Do the trials
        TS = []
        for mui in trial_iter:
            if verb:
                print("Trials for mu = {:.2f}".format(mui))
            sig_gen = signal_inj.sample(mean_mu=mui)
            res, nzeros = self.do_trials(n_trials=ntrials, ns0=mui,
                                         signal_inj=sig_gen, bg_inj=bg_inj,
                                         bg_rate_inj=bg_rate_inj,
                                         verb=False, full_out=False)
            TS.append(np.concatenate((res["TS"],
                                      np.zeros(nzeros, dtype=np.float))))

        # Create the CDF values and fit the chi2
        cdfs = []
        for TSi in TS:
            cdfs.append(weighted_cdf(TSi, val=ts_val)[0])
        cdfs = np.array(cdfs)

        def cdf_func(x, df, loc, scale):
            """Can't use scs.chi2.cdf directly in curve fit."""
            return scs.chi2.cdf(x, df, loc, scale)

        pars, cov = sco.curve_fit(cdf_func, xdata=mus, ydata=1. - cdfs)
        mu_bf = scs.chi2.ppf(beta, *pars)

        return mu_bf, cdfs, TS, pars

    def unblind(self):
        """
        Get the TS value for unblinded on data.

        Parameters
        ----------
        Xon : record-array
            On time data
        ns0 : float
        minimizer_opts : dict, optional

        Returns
        -------
        TS : float
        ns : float
        significance : float
        """
        raise NotImplementedError("Not done yet.")
