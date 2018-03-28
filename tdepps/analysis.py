# coding: utf-8

from __future__ import print_function, division, absolute_import

import numpy as np
from sklearn.utils import check_random_state

from .utils import fit_chi2_cdf


class GRBLLHAnalysis(object):
    """
    Providing methods to do analysis stuff on a GRBLLH.
    """
    def __init__(self, model_injector, llh, random_state=None):
        self._check_inj_llh_harmony(model_injector, llh)
        self._model_injector = model_injector
        self._llh = llh
        self.rndgen = random_state

    @property
    def model_injector(self):
        return self._model_injector

    @model_injector.setter
    def model_injector(self, model_injector):
        self._check_inj_llh_harmony(model_injector, self._llh)
        self._model_injector = model_injector

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

    def _check_inj_llh_harmony(inj_mod, llh):
        """
        Check if injector model and llh fit together. Both must be single or
        multi sample type.
        """
        if hasattr(inj_mod, "names"):      # Injector is multi sample
            if not hasattr(llh, "names"):  # But LLH is not
                raise TypeError("'injector_model' is a multi sample injector " +
                                "but 'llh' is a single sample LLH.")
            # Check if exchanged data is compatible. Must be dicts of lists here
            for sam in inj_mod.names:
                for name in inj_mod.provided_data_names[sam]:
                    if name not in llh.needed_data_names[sam]:
                        e = ("'model_injector' for sample '{}' ".format(sam) +
                             "is not providing data name '{}' ".format(name) +
                             "needed for the corresponding LLH model.")
                        raise ValueError(e)
        else:                          # Injector is single sample
            if hasattr(llh, "names"):  # But LLH is not
                raise TypeError("'llh' is a multi sample LLH but the " +
                                "'injector_model' is a single sample injector.")
            # Check if exchanged data is compatible. Must be list here
            for name in inj_mod.provided_data_names:
                if name not in llh.needed_data_names:
                    raise ValueError("'model_injector' is not providing " +
                                     "data name '{}' ".format(name) +
                                     "needed for the LLH model.")

    def do_trials(self, n_trials, n_signal=0, ns0=1., poisson=True,
                  full_out=False):
        """
        Do pseudo experiment trials using events from the injector model.
        """
        gen = self.make_trial_gen(ns0)

        ns, TS = [], []
        nzeros = 0
        for i in range(n_trials):
            # TODO: Get ninj from injector model?
            ns_i, TS_i, X_i = next(gen)
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

    def performance(self, ts_val, beta, mus, par0=[1., 1., 1.], ntrials=1000):
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

        # Do the trials
        TS = []
        ns = []
        nsig = []
        for mui in mus:
            res, nzeros, nsig_i = self.do_trials(n_trials=ntrials, ns0=mui,
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
