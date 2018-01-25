# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import next
from future import standard_library
standard_library.install_aliases()

import numpy as np
import scipy.stats as scs
import scipy.optimize as sco

from tdepps.utils import weighted_cdf


class GRBLLHAnalysis(object):
    """
    Providing methods to do a transients analysis.

    model_injector yields events for the LLH that gets tested
    """
    def __init__(self, model_injector, llh):
        self._check_inj_llh_harmony(model_injector, llh)
        self._model_injector = model_injector
        self._llh = llh

    @property
    def model_injector(self):
        return self._model_injector

    @model_injector.setter
    def model_injector(self, model_injector):
        self._check_inj_llh_harmony(model_injector, self._llh)
        self._model_injector = self._model_injector

    @property
    def llh(self):
        return self._llh

    @llh.setter
    def llh(self, llh):
        self._check_inj_llh_harmony(self._model_injector, llh)
        self.llh = self.llh

    def _check_inj_llh_harmony(inj_mod, llh):
        """
        Check if injector model and llh fit together. Both must be single or
        multi sample type.
        """
        if hasattr(inj_mod, "names"):      # Injector is multi sample
            if not hasattr(llh, "names"):  # But LLH is not
                raise TypeError("'injector_model' is a multi sample injector " +
                                "but 'llh' is a single sample LLH.")
            for sam in inj_mod.names:
                for name in inj_mod.injectors[sam].provided_names:
                    if name not in llh.llhs[sam].model.needed_names:
                        e = ("'model_injector' for sample '{}' ".format(sam) +
                             "is not providing data name '{}' ".format(name) +
                             "needed for the corresponding LLH model.")
                        raise ValueError(e)
        else:                          # Injector is single sample
            if hasattr(llh, "names"):  # But LLH is not
                raise TypeError("'llh' is a multi sample LLH but the " +
                                "'injector_model' is a single sample injector.")
            for name in inj_mod.provided_names:
                if name not in llh.model.needed_names:
                    raise ValueError("'model_injector' is not providing " +
                                     "data name '{}' ".format(name) +
                                     "needed for the LLH model.")

    # PUBLIC
    def do_trials(self, n_trials, ns0, full_out=False):
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

    def make_trial_gen(self, ns0):
        """ Creates a generator which yields on full trial per iteration. """
        while True:
            X = self._model_injector.get_sample()
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
        mu_bf, cdfs, pars = self.fit_chi2_cdf(ts_val, beta, TS, mus)

        return {"mu_bf": mu_bf, "ts": TS, "ns": ns, "mus": mus, "ninj": nsig,
                "beta": beta, "tsval": ts_val, "cdfs": cdfs, "pars": pars}

    def post_trials(self, n_trials, time_windows, ns0):
        """
        Use an analysis object with a BG model with different injection and
        PDF models. Injects BG only for the largest time window but test the
        injected data with a LLH having each of the different timewindows.
        """

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
