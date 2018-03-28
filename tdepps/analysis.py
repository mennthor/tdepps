# coding: utf-8

from __future__ import print_function, division, absolute_import

import numpy as np
from sklearn.utils import check_random_state
from itertools import repeat

from .utils import fit_chi2_cdf, logger, arr2str


class GRBLLHAnalysis(object):
    """
    Providing methods to do analysis stuff on a GRBLLH.

    Parameters
    ----------
    llh : BaseLLH or BaseMultiLLH instance
        LLH model used for testing hypothesis.
    bg_inj : BaseBGDataInjector
        Background injector model injecting background-like events in trials.
    bg_inj : BaseSignalInjector
        Signal injector model injecting signal-like events in trials.
    random_state : None, int or np.random.RandomState, optional
        Used as PRNG, see ``sklearn.utils.check_random_state``. (default: None)
    """
    def __init__(self, llh, bg_inj, sig_inj, random_state=None):
        self._log = logger(name=self.__class__.__name__, level="ALL")

        self._check_llh_inj_harmony(llh, bg_inj, sig_inj)
        self._llh = llh
        self._bg_inj = bg_inj
        self._sig_inj = sig_inj

        self.rndgen = random_state

    @property
    def bg_inj(self):
        return self._bg_inj

    @property
    def sig_inj(self):
        return self._sig_inj

    # @model_injector.setter
    # def model_injector(self, model_injector):
    #     self._check_inj_llh_harmony(model_injector, self._llh)
    #     self._model_injector = model_injector

    @property
    def llh(self):
        return self._llh

    @llh.setter
    def llh(self, llh):
        self._check_inj_llh_harmony(self._model_injector, llh)
        self.llh = llh

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
        def _all_equal(a1, a2):
            """ ``True`` if a1 and a2 are equal (unsorted test) """
            if (len(a1) == len(a2)) and np.all(np.isin(a1, a2)):
                return True
            return False

        def _all_match(llh, inj, multi):
            if multi:
                keys, models, injs = llh.names, llh.model_pdf, inj.injs
            else:
                keys, models, injs = ["none"], [llh.model_pdf], [inj]

            for key, model, inj in zip(keys, models, injs):
                if not _all_equal(model.needed_data, inj.provided_data):
                    e = "Provided and needed names don't match"
                    if multi:
                        e += " for sample '{}'".format(key)
                    e += ": ['{}'] != ['{}'].".format(
                        arr2str(model.needed_data, sep="', '"),
                        arr2str(inj.provided_data, sep="', '"))
                    raise KeyError(e)
            return True

        # First check if all are single or multi types
        has_names = [hasattr(inst, "names") for inst in [llh, bg_inj, sig_inj]]
        if all(has_names):
            print(self._log.INFO("Dealing with multi types."))
            # All names must match in the multi case
            if not (_all_equal(llh.names, bg_inj.names) and
                    _all_equal(llh.names, sig_inj.names)):
                raise AttributeError("LLH and / or injector names don't match.")
            multi = True
        elif not all(has_names):
            print(self._log.INFO("Dealing with single types."))
            multi = False
        else:
            raise TypeError(
                "LLH and injectors are not all single or multi types.")

        # Now check if exchanged data recarrays are matching
        try:
            _all_match(llh, bg_inj, multi)
        except KeyError as e:
            print(self._log.ERROR(e))
            raise KeyError("Provided and needed names " +
                           "don't match for `llh` and `bg_inj`.")
        try:
            _all_match(llh, sig_inj, multi)
        except KeyError as e:
            print(self._log.ERROR(e))
            raise KeyError("Provided and needed names " +
                           "don't match for `llh` and `sig_inj`.")
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
            Has names ``'TS', 'ns'`` and holds the fit results of each trial.
        nzeros : int
            How many zero trials occured.
        nsig : array-like, optional
            Only if ``full_out`` is ``True``. Then contains the number of
            sampled signal events per trial.
        """
        if n_signal is None:
            n_signal = 0

        if poisson:
            nsig = self._rndgen.poisson(n_signal, size=n_trials)
        else:
            nsig = repeat(0., n_trials)

        ns, TS = [], []
        nzeros = 0
        for i, nsig_i in enumerate(nsig):
            X = self._bg_inj.sample()
            if nsig_i > 0:
                Xsig = self._sig_inj.sample(nsig_i)
                X = np.concatenate((X, Xsig))

            ns_i, TS_i = self.llh.fit_lnllh_ratio(ns0, X)

            if (ns_i == 0) and (TS_i == 0):
                nzeros += 1
                if full_out:
                    ns.append(0.)
                    TS.append(0.)
            else:
                ns.append(ns_i)
                TS.append(TS_i)

        # Make output record array for non zero trials
        if full_out:
            size = n_trials
        else:
            size = n_trials - nzeros
        res = np.empty((size,), dtype=[("ns", np.float), ("TS", np.float)])
        res["ns"] = np.array(ns)
        res["TS"] = np.array(TS)

        if full_out:
            return res, nzeros, np.array(nsig, dtype=int)
        else:
            return res, nzeros

    def make_trial_gen(self, n_signal=0, ns0=1., poisson=True):
        """
        Creates a generator which yields one full dataset per trial.

        Parameters
        ----------
        n_signal : float
            Mean number of signal events to sample each trial.
        ns0 : float
            Seed for the LLh fit for each trial.
        poisson : bool, optional
            If ``True`` sample a new number of signal events to inject for each
            trial from a poisson distribution with mean ``n_signal``. If
            ``False``, ``n_signal`` is ceiled to the next integer.
        """
        raise DeprecationWarning("Just call do_trials.")
        if poisson:
            while True:
                nsig = self._rndgen.poisson(n_signal, size=1)
                X = self._model_injector.get_sample(nsig)
                ns, TS = self.llh.fit_lnllh_ratio(ns0, X)
                yield ns, TS, X
        else:
            nsig = int(np.ceil(nsig))
            while True:
                X = self._model_injector.get_sample(nsig)
                ns, TS = self.llh.fit_lnllh_ratio(ns0, X)
                yield ns, TS, X

    def performance(self, ts_val, beta, mus, par0=[1., 1., 1.], n_trials=1000):
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
        mus : array-like
            How much mean poisson signal shall be injected.
        par0 : list, optional
            Seed values ``[df, loc, scale]`` for the ``chi2`` CDF fit.
            (default: ``[1., 1., 1.]``)
        n_trials : int, optional
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

        # Do the trials
        TS = []
        ns = []
        nsig = []
        for mui in mus:
            res, nzeros, nsig_i = self.do_trials(n_trials=n_trials, ns0=mui,
                                                 full_out=True)
            TS.append(res["TS"])
            ns.append(res["ns"])
            nsig.append(nsig_i)

        # Create the requested CDF values and fit the chi2
        mu_bf, cdfs, pars = fit_chi2_cdf(ts_val, beta, TS, mus)

        return {"mu_bf": mu_bf, "ts": TS, "ns": ns, "mus": mus, "ninj": nsig,
                "beta": beta, "tsval": ts_val, "cdfs": cdfs, "pars": pars}

    def post_trials(self, n_trials, time_windows, ns0):
        """
        Use an analysis object with a BG model with different injection and
        PDF models. Injects BG only for the largest time window but test the
        injected data with a LLH having each of the different timewindows.
        """
        pass
