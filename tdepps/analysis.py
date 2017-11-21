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
from tqdm import tqdm


class TransientsAnalysis(object):
    def __init__(self, srcs, llh):
        """
        Providing methods to do a transients analysis.

        Parameters
        ----------
        srcs : recarray or dict(name, recarray)
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
              `ns`. Use ``None`` for one of ``min, max`` when there is no bound
              in that direction. (default: ``[[0, None]]``)
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
        # Check that srcs here and in injectors match
        if self._multi:
            names = self._srcs.keys()
            _injt = np.concatenate([bg_rate_inj._injs[n].srcs[0] for
                                    n in names])
            _anat = np.concatenate([self._srcs[n]["t"] for n in names])
            _injtr = np.concatenate([bg_rate_inj._injs[n].srcs[1] for
                                     n in names])
            _anatr = np.concatenate([np.vstack((self._srcs[n]["dt0"],
                                                self._srcs[n]["dt1"])).T for
                                     n in names])
        else:
            _injt = bg_rate_inj.srcs[0]
            _anat = self._srcs["t"]
            _injtr = bg_rate_inj.srcs[1]
            _anatr = np.vstack((self._srcs["dt0"], self._srcs["dt1"])).T
        if not np.array_equal(_injt, _anat):
            raise ValueError("Source times in ana and bg rate injector " +
                             "don't match.")
        if not np.array_equal(_injtr, _anatr):
            raise ValueError("Source dts in ana and bg rate injector " +
                             "don't match.")

        # Setup minimizer defaults and bounds
        bounds, minimizer_opts = self._setup_minopts(minimizer_opts)

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
        nb = bg_rate_inj.get_nb()
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
        ns = []
        TS = []
        nsig_all = []
        if self._multi:  # Use multi LLH syntax
            for i in trial_iter:
                # Inject events from given injectors
                times = bg_rate_inj.sample(poisson=True)
                nevts_split = {n: len(ti) for n, ti in times.items()}
                nevts = np.sum(nevts_split.values())
                if nevts > 0:
                    X = bg_inj.sample(nevts_split)
                    for key, arr in X.items():
                        X[key] = append_fields(arr, "timeMJD", times[key],
                                               dtypes=np.float, usemask=False)
                else:
                    X = None

                if signal_inj is not None:
                    nsig, Xsig, _ = next(signal_inj)
                    nsig_all.append(nsig)
                    nevts += nsig
                else:
                    Xsig = None

                # If we have no events at all, fit will be zero
                if nevts == 0:
                    nzeros += 1
                    if full_out:
                        ns.append(0.)
                        TS.append(0.)
                    continue

                # Else ask LLH what value we have
                if Xsig is not None:
                    if X is not None:
                        for key, arr in Xsig.items():
                            X[key] = stack_arrays((X[key], arr), usemask=False)
                    else:
                        X = Xsig

                # Only store the best fit params and the TS value if nonzero
                _ns, _TS = self.llh.fit_lnllh_ratio(X, ns0, args, bounds,
                                                    minimizer_opts)

                if (_ns == 0) and (_TS == 0):
                    nzeros += 1
                    if full_out:
                        ns.append(0.)
                        TS.append(0.)
                else:
                    ns.append(_ns)
                    TS.append(_TS)
        else:  # Use single LLH and injectors
            for i in trial_iter:
                # Inject events from given injectors
                times = bg_rate_inj.sample(poisson=True)
                times = np.concatenate(times, axis=0)
                nevts = len(times)

                if nevts > 0:
                    X = bg_inj.sample(nevts)
                    X = append_fields(X, "timeMJD", times, dtypes=np.float,
                                      usemask=False)

                if signal_inj is not None:
                    nsig, Xsig, _ = next(signal_inj)
                    nsig_all.append(nsig)
                    nevts += nsig
                else:
                    Xsig = None

                # If we have no events at all, fit will be zero
                if nevts == 0:
                    nzeros += 1
                    if full_out:
                        ns.append(0.)
                        TS.append(0.)
                    continue

                # Else ask LLH what value we have
                if Xsig is not None:
                    X = stack_arrays((X, Xsig), usemask=False)

                # Only store the best fit params and the TS value if nonzero
                _ns, _TS = self.llh.fit_lnllh_ratio(X, ns0, args, bounds,
                                                    minimizer_opts)
                if (_ns == 0.) and (_TS == 0.):
                    nzeros += 1
                    if full_out:
                        ns.append(0.)
                        TS.append(0.)
                else:
                    ns.append(_ns)
                    TS.append(_TS)

        # Make output record array for non zero trials
        if full_out:
            size = n_trials
        else:
            size = n_trials - nzeros
        res = np.empty((size,), dtype=[("ns", np.float), ("TS", np.float)])
        res["ns"] = np.array(ns)
        res["TS"] = np.array(TS)

        if full_out:
            if signal_inj is not None:
                return res, nzeros, nsig_all
            else:
                return res, nzeros, np.empty((0,))
        else:
            return res, nzeros

    def performance_chi2(self, ts_val, beta, bg_inj, bg_rate_inj, signal_inj,
                         mus, par0=[1., 1., 1.], ntrials=1000, verb=False):
        """
        Make independent trials within given range and fit a ``chi2`` CDF to the
        resulting percentiles, becasue they are surprisingly well described by
        that.

        The returned CDF and best fit ``chi2`` values are valid for the given
        combination of ``beta`` and ``ts_val``.
        But it is possible to use the same trial values to calculate performance
        at different values by recalculating the CDF values and refitting a
        ``chi2`` function.

        Parameters
        ----------
        ts_val : float
            Test statistic value of the BG distribution, which is connected to
            the alpha value (Type I error).
        beta : float
            Fraction of signal injected PDF that should lie right of the
            ``ts_val``.
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
        res : dict
            Result dictionary with keys:

            - "mu_bf": Best fit poisson signal expectation derived from the
              fitted ``chi2`` CDF.
            - "mus": Same as input ``mus``.
            - "cdfs": Calculated CDF values for each trial from the ``chi2``
              function.
            - "pars": Best fit parameters from the ``chi2`` CDF fit.
            - "ts": For each bunch ``mu`` trials the resulting test satistic
              values. From these we can in principle  calculate other
              ``ts_val, beta`` combinations by calculating new percentiles and
              refit the ``chi2`` wthout doing more trials.
            - "ns": Same as ``'ts'`` but for the fitted signal parameter.
            - "ninj": Same as ``'ns'`` but the number of injected signal events
              per trial per bunch of trials.
        """
        if np.any(mus < 0):
            raise ValueError("`mus` must not have an entry < 0.")

        if verb:
            trial_iter = tqdm(mus)
        else:
            trial_iter = mus

        # Do the trials
        TS = []
        ns = []
        nsig = []
        for mui in trial_iter:
            sig_gen = signal_inj.sample(mean_mu=mui)
            res, nzeros, nsig_i = self.do_trials(n_trials=ntrials, ns0=mui,
                                                 signal_inj=sig_gen,
                                                 bg_inj=bg_inj,
                                                 bg_rate_inj=bg_rate_inj,
                                                 verb=False, full_out=True)
            TS.append(res["TS"])
            ns.append(res["ns"])
            nsig.append(nsig_i)

        # Create the CDF values and fit the chi2
        mu_bf, cdfs, pars = TransientsAnalysis.fit_chi2_cdf(ts_val, beta, TS,
                                                            mus)

        return {"mu_bf": mu_bf, "ts": TS, "ns": ns, "mus": mus, "ninj": nsig,
                "beta": beta, "tsval": ts_val, "cdfs": cdfs, "pars": pars}

    @staticmethod
    def fit_chi2_cdf(ts_val, beta, TS, mus):
        """
        Use collection of trials with different numbers injected mean signal
        events to calculate the CDF values above a certain test statistic
        value ``ts_val`` and fit a ``chi2`` CDF to it.
        From this ``chi2``function we can get the desired percentile ``beta``
        above ``ts_val``.

        Trials can systematically be made using :py:meth:`performance_chi2`.

        Parameters
        ----------
        ts_val : float
            Test statistic value of the BG distribution, which is connected to
            the alpha value (Type I error).
        beta : float
            Fraction of signal injected PDF that should lie right of the
            ``ts_val```.
        mus : array-like
            How much mean poisson signal shall was injected for each bunch of
            trials.
        TS : array-like, shape (len(mus), ntrials_per_mu)
            Test statistic values for each ``mu`` in ``mus``. These are used to
            calculate the CDF values used in the fit.

        Returns
        -------
        mu_bf : float
            Best fit mean injected number of signal events to fullfill the
            tested performance level from ``ts_val`` and ``beta``.
        cdfs : array-like, shape (len(mus))

        pars : tuple
            Best fit parameters ``(df, loc, scale)`` for the ``chi2`` CDF.
        """
        cdfs = []
        for TSi in TS:
            cdfs.append(weighted_cdf(TSi, val=ts_val)[0])
        cdfs = np.array(cdfs)

        def cdf_func(x, df, loc, scale):
            """Can't use scs.chi2.cdf directly in curve fit."""
            return scs.chi2.cdf(x, df, loc, scale)

        try:
            pars, _ = sco.curve_fit(cdf_func, xdata=mus, ydata=1. - cdfs)
            mu_bf = scs.chi2.ppf(beta, *pars)
        except RuntimeError:
            print("Couldn't find best params, returning `None` instead.")
            mu_bf = None
            pars = None

        return mu_bf, cdfs, pars

    def post_trials(self, n_trials, time_windows, ns0, bg_inj, bg_rate_inj,
                    minimizer_opts=None, verb=False):
        """
        Make trials to create a post trial correction for the selection biased
        p-value for the following case:
        On data, we may want to pick the best p-value from multiple tested time
        ranges with the same set of sources.
        So we can construct a p-value distribution by making BG only trials by
        sampling from the largest time window and fitting with all time window
        PDFs.
        The we pick and store the best p-value for this trial.
        Doing this for many trials will yield a best p-value distribtuion
        against which we can compare the final data p-value.
        Time windows must be fully included in the next bigger ones to make this
        approach work here.

        Parameters
        ----------
        n_trials : int
            Number of trials to perform.
        time_windows : array-like, shape (n_windows, 2)
            Time windows to fit in.
        ns0 : float
            Fitter seed for the fit parameter ns: number of signal events that
            we expect at the source locations.
        bg_inj : `tdepps.bg_injector` instance
            Injector to generate background-like pseudo events.
        bg_rate_inj : `tdepps.bg_rate_injector` instance
            Injector to generate the times of background-like pseudo events in
            the largest (or larger) time window as given in ``time_windows``.
        minimizer_opts : dict, optional
            Options passed to `scipy.optimize.minimize` [1] using the 'L-BFGS-B'
            algorithm. If specific key is not given or argument is None, default
            values are set to:

            - 'bounds', array-like, shape (1, 2): Bounds `[[min, max]]` for
              `ns`. Use ``None`` for one of ``min, max`` when there is no bound
              in that direction. (default: ``[[0, None]]``)
            - 'ftol', float: Minimizer stops when the absolute tolerance of the
              function value is `ftol`. (default: 1e-12)
            - 'gtol', float: Minimizer stops when the absolute tolerance of the
              gradient component is `gtol`. (default: 1e-12)
            - maxiter, int: Maximum fit iterations the minimiter performs.
              (default: 1e3)

            (default: None)

        verb : bool, optional
            If ``True`` show iteration status with ``tqdm``.
            (default: ``False``)
        """
        # Setup minimizer defaults and bounds
        bounds, minimizer_opts = self._setup_minopts(minimizer_opts)

        time_windows = np.atleast_2d(time_windows)
        n_tw = len(time_windows)
        if time_windows.shape[1] != 2:
            raise ValueError("Time window array must have shape (nwindows, 2).")

        # Test if the large windows include the small ones and that no zero or
        # negative sized windows are given
        time_windows = time_windows[np.argsort(np.diff(time_windows).ravel())]

        if np.any(np.diff(time_windows, axis=1) <= 0):
            raise ValueError("Time window lengths must all be > 0.")

        for i, (tw_low, tw_hig) in enumerate(zip(time_windows[:-1],
                                                 time_windows[1:])):
            if (tw_low[0] < tw_hig[0]) or (tw_low[1] > tw_hig[1]):
                raise ValueError("Time window {} ".format(i) +
                                 "is not included in the next larger one")

        assert np.argmax(np.diff(time_windows)) == len(time_windows) - 1
        tw_max = time_windows[-1]

        if verb:
            print("Using {:d} time windows.".format(n_tw))
            print("Sorted time windows are:\n", time_windows)
            print("Maximum time window used for injection:\n", tw_max)

        # Prepare fixed source parameters for injectors, use largest time
        # window. Also prepare estimated BG events for each time window
        nb = []
        if self._multi:
            src_t = {key: src_i["t"] for key, src_i in self._srcs.items()}
            src_dt = {}
            for tw in time_windows:
                for key, src_i in self._srcs.items():
                    src_dt[key] = np.repeat([tw], repeats=len(src_i), axis=0)
                nb.append(bg_rate_inj.get_nb(src_t, src_dt))
            # The last time window should be the largest
            assert np.all(np.diff(np.concatenate(list(src_dt.values())),
                                  axis=1) == np.diff(tw_max))
        else:
            src_t = self._srcs["t"]
            for tw in time_windows:
                src_dt = np.repeat([tw], repeats=len(self._srcs), axis=0)
                nb.append(bg_rate_inj.get_nb(src_t, src_dt))
            assert np.all(np.diff(src_dt, axis=1) == np.diff(tw_max))

        assert len(nb) == n_tw
        if verb:
            print("Estimated background events per time window and src:")
            print(nb)

        # Make sure bg_rate_injector is using the largest time window correctly
        if self._multi:
            names = self._srcs.keys()
            _injtr = np.concatenate([bg_rate_inj._injs[n].srcs[1] for
                                     n in names])
        else:
            _injtr = bg_rate_inj.srcs[1]
        if not np.array_equal(_injtr, np.repeat([tw_max], repeats=len(_injtr),
                                                axis=0)):
            raise ValueError("Source dts in bg rate injector us not using " +
                             "the largest time window.")

        # Total injection time window in which the time PDF is defined and
        # nonzero for the largest time window.
        trange_max = self._llh.time_pdf_def_range(src_t, src_dt)
        assert len(trange_max) == len(self._srcs)

        # Select iterator depending on `verb` keyword
        if verb:
            print("Doing trial fits on {} time windows.".format(n_tw))
            trial_iter = tqdm(range(n_trials))
        else:
            trial_iter = range(n_trials)

        res = np.empty((n_trials, n_tw), dtype=[("ns", np.float),
                                                ("TS", np.float)])
        # Copy the src dict, because we change some stuff in there
        srcs = self._srcs.copy()
        if self._multi:  # Use multi LLH syntax
            for i in trial_iter:
                # Inject events from largest window and then fit with every
                # given time window. The LLH is cutting of any events outside
                # the current time window automatically.
                times = bg_rate_inj.sample(poisson=True)
                nevts_split = {n: len(ti) for n, ti in times.items()}
                nevts = np.sum(list(nevts_split.values()))
                if nevts > 0:
                    X = bg_inj.sample(nevts_split)
                    for key, arr in X.items():
                        X[key] = append_fields(arr, "timeMJD", times[key],
                                               dtypes=np.float, usemask=False)
                else:
                    # If we have no events at all, fit will be zero for all tws
                    res["ns"][i] = np.zeros(n_tw, dtype=np.float)
                    res["TS"][i] = np.zeros(n_tw, dtype=np.float)
                    continue

                # Now do the fit on the same data for all time windows
                for j, tw in enumerate(time_windows):
                    # Setup LLH args with correct time window information
                    for key, src_i in srcs.items():
                        src_i["dt0"] = tw[0]
                        src_i["dt1"] = tw[1]
                    args = {name: {"nb": nb[j][name], "srcs": srcs[name]} for
                            name in self._srcs.keys()}

                    # Data is still from the largest time window
                    _ns, _TS = self.llh.fit_lnllh_ratio(
                        X, ns0, args, bounds, minimizer_opts)
                    res["ns"][i, j] = _ns
                    res["TS"][i, j] = _TS

        else:  # Use single LLH and injectors
            raise NotImplementedError("TODO: Single LLH")
            # for i in trial_iter:
            #     # Inject events from given injectors
            #     times = bg_rate_inj.sample(poisson=True)
            #     times = np.concatenate(times, axis=0)
            #     nevts = len(times)

            #     if nevts > 0:
            #         X = bg_inj.sample(nevts)
            #         X = append_fields(X, "timeMJD", times, dtypes=np.float,
            #                           usemask=False)

            #     if signal_inj is not None:
            #         nsig, Xsig, _ = next(signal_inj)
            #         nsig_all.append(nsig)
            #         nevts += nsig
            #     else:
            #         Xsig = None

            #     # If we have no events at all, fit will be zero
            #     if nevts == 0:
            #         nzeros += 1
            #         if full_out:
            #             ns.append(0.)
            #             TS.append(0.)
            #         continue

            #     # Else ask LLH what value we have
            #     if Xsig is not None:
            #         X = stack_arrays((X, Xsig), usemask=False)

            #     # Only store the best fit params and the TS value if nonzero
            #     _ns, _TS = self.llh.fit_lnllh_ratio(X, ns0, args, bounds,
            #                                         minimizer_opts)
            #     if (_ns == 0.) and (_TS == 0.):
            #         nzeros += 1
            #         if full_out:
            #             ns.append(0.)
            #             TS.append(0.)
            #     else:
            #         ns.append(_ns)
            #         TS.append(_TS)

        return res, time_windows

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

    def _setup_minopts(self, minimizer_opts=None):
        """
        Setup minimizer options and return refined settings and the parameter
        bounds specified seperately.

        Parameters
        ----------
        minimizer_opts : dict or None
            See :py:meth:`do_trials`, Parameters.

        Returns
        -------
        bounds : list
            Bounds for the minimizer, specified seperately.
        minopts : dict
            Refined settings, with defaults plugged in.
        """
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

        return bounds, minopts
