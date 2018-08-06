# coding: utf-8

from __future__ import print_function, division, absolute_import

import numpy as np
from sklearn.utils import check_random_state

from ..utils import fit_chi2_cdf, logger, arr2str, all_equal, dict_map


class PSLLHAnalysis(object):
    """
    Providing methods to do analysis stuff on a PS LLH.

    Parameters
    ----------
    llh : BaseLLH or BaseMultiLLH instance
        LLH model used for testing hypothesis.
    bg_inj : BaseBGDataInjector or BaseMultiBGDataInjector
        Background injector model injecting background-like events in trials.
    sig_inj : BaseSignaalInjector or BaseMultiSignalInjector or None
        Signal injector model injecting signal-like events in trials. If
        ``None`` no signal or performance trials can be done.
        (default: ``None``)
    random_state : None, int or np.random.RandomState, optional
        Used as PRNG, see ``sklearn.utils.check_random_state``.
        (default: ``None``)
    """
    def __init__(self, llh, bg_inj, sig_inj=None, random_state=None):
        self._log = logger(name=self.__class__.__name__, level="ALL")

        self._check_llh_inj_harmony(llh, bg_inj, sig_inj)
        self._llh = llh
        self._bg_inj = bg_inj
        self._sig_inj = sig_inj

        self.rndgen = random_state

    @property
    def llh(self):
        return self._llh

    @property
    def bg_inj(self):
        return self._bg_inj

    @property
    def sig_inj(self):
        return self._sig_inj

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    def _check_llh_inj_harmony(self, llh, bg_inj, sig_inj):
        """
        Check if llh and injectors fit together. Both must be single or multi
        sample type and provide / receive matching record array data object.

        A multi injector has an additional 'names' attribute which must match
        across all injectors and llhs. Each single injector must match its
        provided data with the requirements from the lll receiving it.
        """
        def _all_match(llh, inj):
            """ Check if provided and needed names match for inj, llh pairs """
            if hasattr(llh, "llhs"):
                for key in llh.llhs.keys():
                    model = llh.llhs[key].model
                    inj_i = inj.injs[key]
                    if not all_equal(model.needed_data, inj_i.provided_data):
                        e = "Provided and needed names don't match"
                        e += " for sample '{}'".format(key)
                        e += ": ['{}'] != ['{}'].".format(
                            arr2str(model.needed_data, sep="', '"),
                            arr2str(inj_i.provided_data, sep="', '"))
                        raise KeyError(e)
            else:
                if not all_equal(llh.model.needed_data, inj.provided_data):
                    e = "Provided and needed names don't match"
                    e += ": ['{}'] != ['{}'].".format(
                        arr2str(llh.model.needed_data, sep="', '"),
                        arr2str(inj.provided_data, sep="', '"))
                    raise KeyError(e)

            return True

        # First check if all are single or multi types and if we have a sig inj.
        if sig_inj is None:
            injs = [bg_inj]
            print(self._log.INFO("No signal injector, can only do BG trials."))
        else:
            injs = [bg_inj, sig_inj]

        has_names = ([hasattr(llh, "llhs")] +
                     [hasattr(_inj, "injs") for _inj in injs])

        if all(has_names):
            print(self._log.INFO("Dealing with multi sample modules."))
            # All names must match in the multi case
            all_equ = [all_equal(llh.llhs.keys(), _inj.injs.keys())
                       for _inj in injs]
            if not all(all_equ):
                raise AttributeError("LLH and / or injector names don't match.")
        elif not all(has_names):
            print(self._log.INFO("Dealing with single sample modules."))
        else:
            raise TypeError(
                "Likelihoods and injectors are not all single or multi types.")

        # Now check if exchanged data recarrays are matching
        for _inj in injs:
            try:
                _all_match(llh, bg_inj)
            except KeyError as e:
                print(self._log.ERROR(e))
                raise KeyError("Provided and needed names don't match for " +
                               "likelihoods and injectors.")

        return

    def do_trials(self, n_trials, n_signal=None, ns0=1., poisson=True,
                  full_out=False):
        """
        Do pseudo experiment trials using events from the injector model.

        Parameters
        ----------
        n_trials : int
            Number of trials to perform.
        n_signal : int or None, optional
            Number of mean signal events to inject per trial. If ``None``, no
            signal is injected, only do background trials. (Default: ``None``)
        ns0 : float, optional
            Seed value for the ns fit parameter. (Default: 1.)
        poisson : bool, optional
            If ``True`` sample the injected number of signal events per trial
            from a poisson distribution with mean ``n_signal``.
            (Default: ``True``)
        full_out : bool, optioanl
            If ``True`` also append zero trials to the output array, else only
            return how many zero trials occured. Also return number of injected
            signal events per trial. (Default: ``False``)

        Returns
        -------
        res : record-array
            Has names ``'ts', 'ns'`` and holds the fit results of each trial.
        nzeros : int
            How many zero trials occured.
        nsig : array-like or None
            Contains the number of sampled signal events per trial, only if
            ``full_out`` is ``True``, else ``None`` is returned.
        """
        if n_signal is None:
            nsig_i = 0
        elif not poisson:
            nsig_i = int(n_signal)
        if n_signal == 0:  # Set to None if we don't sample signal anyway
            n_signal = None
            nsig_i = 0

        if self._sig_inj is None and n_signal is not None:
            raise ValueError("This module was not given a signal injector, " +
                             "'n_signal' can only be `None` here.")

        # Define the concatenation dependending on single or multi samples
        if hasattr(self._llh, "llhs"):
            def _concat(X, Xsig):
                return dict_map(lambda k, Xi: np.concatenate((Xi, Xsig[k])), X)
        else:
            def _concat(X, Xsig):
                return np.concatenate((X, Xsig))

        ns, ts, nsig = [], [], []
        nzeros = 0
        for i in range(n_trials):
            X = self._bg_inj.sample()
            if n_signal is not None:
                if poisson:
                    nsig_i = self._rndgen.poisson(n_signal, size=None)
                if nsig_i > 0:
                    Xsig = self._sig_inj.sample(nsig_i)
                    X = _concat(X, Xsig)

            ns_i, ts_i = self.llh.fit_lnllh_ratio(ns0=ns0, X=X)

            if full_out:
                ns.append(ns_i)
                ts.append(ts_i)
                nsig.append(nsig_i)
            elif (ns_i > 0) and (ts_i > 0):
                ns.append(ns_i)
                ts.append(ts_i)
            else:
                nzeros += 1

        # Make output record array for non zero trials
        if full_out:
            size = n_trials
            nsig = np.array(nsig, dtype=int)
        else:
            size = n_trials - nzeros
            nsig = None
        res = np.empty((size,), dtype=[("ns", np.float), ("ts", np.float)])
        res["ns"] = np.array(ns)
        res["ts"] = np.array(ts)
        if full_out:
            # Didn't get saved if full_out, but we have the info to recompute
            nzeros = np.sum(~((ts > 0) & (ns > 0)))

        return res, nzeros, nsig

    def performance(self, ts_val, beta, mus, ns0=1., par0=[1., 1., 1.],
                    n_batch_trials=1000):
        """
        Make independent trials within given range and fit a ``chi2`` CDF to the
        resulting percentiles, because they are surprisingly well described by
        that.

        The returned CDF and best fit ``chi2`` values are valid for the given
        combination of ``beta`` and ``ts_val``.
        But it is possible to use the same trial values to calculate performance
        at different values by recalculating the CDF values and refitting a
        ``chi2`` function, as long as the generated test statisitic is large
        enough for the desired percentiles.

        Parameters
        ----------
        ts_val : float
            Test statistic value of the BG distribution, which is connected to
            the alpha value (Type I error).
        beta : float
            Fraction of signal injected PDF that should lie right of the
            ``ts_val``.
        mus : array-like
            How much mean poisson signal shall be injected.
        ns0 : float, optional
            Seed value for the ns fit parameter. (Default: 1.)
        par0 : list, optional
            Seed values ``[df, loc, scale]`` for the ``chi2`` CDF fit.
            (default: ``[1., 1., 1.]``)
        n_batch_trials : int, optional
            How many trials to make per independent trial batch. (default: 1000)

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
              refit the ``chi2`` without doing more trials.
            - "ns": Same as ``'ts'`` but for the fitted signal parameter.
            - "ninj": Same as ``'ns'`` but the number of injected signal events
              per trial per bunch of trials.
        """
        if np.any(mus < 0):
            raise ValueError("`mus` must not have an entry < 0.")

        # Do the trials
        ts = []
        ns = []
        nsig = []
        for mui in mus:
            print(self._log.INFO("Starting {} ".format(n_batch_trials) +
                                 "signal trials with mu={:.3f}".format(mui)))
            res, nzeros, nsig_i = self.do_trials(n_trials=n_batch_trials,
                                                 n_signal=mui,
                                                 ns0=ns0,
                                                 poisson=True,
                                                 full_out=True)
            ts.append(res["ts"])
            ns.append(res["ns"])
            nsig.append(nsig_i)

        # Create the requested CDF values and fit the chi2
        print(self._log.INFO("Fitting a chi2 CDF to the trial stats."))
        mu_bf, cdfs, pars = fit_chi2_cdf(ts_val, beta, ts, mus)

        return {"mu_bf": mu_bf, "ts": ts, "ns": ns, "mus": mus, "ninj": nsig,
                "beta": beta, "tsval": ts_val, "cdfs": cdfs, "pars": pars}

    def post_trials(self, n_trials, test_llhs, ns0=1.):
        """
        Do post trials for the GRB case when scanning multiple time windows
        which include each other, instead of fitting them.

        Per post-trial, samples are drawn once from the classes BG injector
        time window and the LLH is then tested for all given time windows on the
        same trial data to include the correlated nature of the time windows.
        All tested sources are given the same time window per post-trial per
        time window.

        Parameters
        ----------
        n_trials : int
            Number of post-trials to perform.
        test_llhs : list
            List of LLH classes to test against. This classes BG injector should
            inject the largest time window for sane results.
        ns0 : float, optional
            Seed value for the ns fit parameter. (Default: 1.)

        Returns
        -------
        res : record-array, shape (n_trials, n_test_llhs)
            Has names ``'ts', 'ns'`` and holds the fit results of each
            post-trial for each tested LLH class.
        """
        for i, test_llh in enumerate(test_llhs):
            print(self._log.DEBUG("Test if test-LLH {} is".format(i) +
                  " compatible with internal BG injector."))
            self._check_llh_inj_harmony(test_llh, self.bg_inj, None)

        res = np.empty((n_trials, len(test_llhs)),
                       dtype=[("ns", np.float), ("ts", np.float)])
        for i in range(n_trials):
            # Sample from the stored bg injector
            X = self._bg_inj.sample()
            for j, test_llh in enumerate(test_llhs):
                # Test each given LLH object
                ns_ij, ts_ij = test_llh.fit_lnllh_ratio(ns0=ns0, X=X)
                res["ns"][i, j] = ns_ij
                res["ts"][i, j] = ts_ij

        return res

    def unblind(self, X, test_llhs, bg_pdfs, post_trial_pdf=None, ns0=1.,
                really_unblind=False, n_signal=None):
        """
        Unblind the analysis on data ``X`` by testing each LLH in ``test_llh``.
        Reports fit values for each test LLH, the pre-trial p-values obtained
        from the ``bg_pdfs`` and the final post-trial p-value from the
        ``post_trial_pdf``.

        Parameters
        ----------
        test_llhs : dict
            Dict of LLH classes to test against. Must have time window IDs as
            keys to match correctly against the ``bg_pdfs``.
        bg_pdfs : dict
            Dict of background PDFs as ``tdepps.utils.stats.EmpiricalDist``
            instances to calculate p-values from the test results. Must have
            time window IDs as keys to match correctly against the
            ``test_llhs``.
        post_trial_pdf : tdepps.utils.stats.EmpiricalDist instance or None
            Post trial ``-log10(p)`` distribution for the final p-value
            calculation. Must match the number of tested time windows ot the
            post trial p-value won't make sense. If ``None`` no post_trial
            p-value is returned. (default: ``None``)
        ns0 : float, optional
            Seed value for the ns fit parameter. (Default: 1.)
        really_unblind : bool, optional
            If ``True`` ignores given data ``X`` and makes a BG trial with the
            stored BG injector instead. If ``False`` tests the given data.
            Additional safety line to avoid accidental unblinding.
            (default: ``False``)
        n_signal : int or None, optional
            Number of mean signal events to inject per trial. If ``None``, no
            signal is injected, only do background trials. (Default: ``None``)

        Returns
        -------
        res : dict
            Result dictionary with keys:

            - "ns": Best fit ``ns`` parameters for each tested LLH.
            - "ts": Best fit ``ns`` parameters for each tested LLH.
            - "pvals": Pre-trial p-value for each tested LLH.
            - "best_tw_id": Best time window ID, having the lowest p-value.
            - "post_pval": Pos-trial corrected p-value for the best time window.
            - "time_window_ids": All tested time window IDs.
        """
        if really_unblind:
            print("## Using the given data to unblind ##")
        else:
            print("Using trial data from internal BG injector to fake unblind.")
            X = self._bg_inj.sample()
            if n_signal is None:
                n_signal = 0
            n_signal = self._rndgen.poisson(n_signal, size=None)
            if n_signal > 0:
                if hasattr(self._llh, "llhs"):
                    def _concat(X, Xsig):
                        return dict_map(
                            lambda k, Xi: np.concatenate((Xi, Xsig[k])), X)
                else:
                    def _concat(X, Xsig):
                        return np.concatenate((X, Xsig))
                Xsig = self._sig_inj.sample(n_signal)
                X = _concat(X, Xsig)
            print("  Also injected {} signal events.".format(n_signal))

        res = {"ns": [], "ts": [], "pvals": [], "post_pval": None,
               "time_window_ids": [], "best_idx": None}

        for idx, test_llh in test_llhs.items():
            # Fit LLH and compute pre-trial p-value
            ns_i, ts_i = test_llh.fit_lnllh_ratio(ns0=ns0, X=X)
            pval_i = bg_pdfs[idx].sf(ts_i)[0]

            res["ns"].append(ns_i)
            res["ts"].append(ts_i)
            res["pvals"].append(pval_i)
            res["time_window_ids"].append(idx)

        # Obtain the final post-trial p-value
        res["best_idx"] = np.argmin(res["pvals"])
        if post_trial_pdf is not None:
            best_neg_log10_pval = -np.log10(res["pvals"][res["best_idx"]])
            res["post_pval"] = post_trial_pdf.sf(best_neg_log10_pval)[0]

        return res
