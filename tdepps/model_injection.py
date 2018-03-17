# coding: utf-8

from __future__ import print_function, division

import abc
import numpy as np
from sklearn.utils import check_random_state

from .model_toolkit import (fit_time_dec_rate_models,
                            make_time_dec_rate_model_splines)
from .utils import spl_normed, fit_spl_to_hist


class ModelInjector(object):
    """ Interface for ModelInjector type classes """
    __metaclass__ = abc.ABCMeta

    _rndgen = None
    _provided_data_names = None

    @property
    def rndgen(self):
        """ numpy RNG instance used to sample """
        return self._rndgen

    @rndgen.setter
    def rndgen(self, rndgen):
        self._rndgen = check_random_state(rndgen)

    @property
    def provided_data_names(self):
        """ Data attributes this injector provides. """
        return self._provided_data_names

    @abc.abstractmethod
    def fit(self):
        """ Sets up the injector and makes it ready to use """
        pass

    @abc.abstractmethod
    def get_sample(self):
        """ Get a data sample for a single trial to use in a LLH object """
        pass


class MultiModelInjector(ModelInjector):
    """ Interface for managing multiple LLH type classes """
    _names = None

    @property
    @abc.abstractmethod
    def names(self):
        """ Subinjector names, identifies this as a MultiModelInjector """
        pass


class GRBModelInjector(ModelInjector):
    """
    Models the injection part for the GRB LLH, implements: ``get_sample()``.
    This model is used for the GRB-like HESE stacking analysis.

    BG injection is allsky and time and declination dependent:
      1. For each source time build a declination dependent detector profile
         from which the declination is sampled weighted.
      2. For each source the integrated event rate over the time interval is
         used to draw the number of events to sample.
      3. Then the total number of events for the source is sampled from the
         total pool of experimental data weighted in declination.
      4. RA is sampled uniformly in ``[0, 2pi]`` and times are sampled from the
         rate function (uniform for small time windows.)

    Signal injection is done similar to skylab:
      1. Spatial and energy attributes are resampled from MC weighted to the
         detector response for a specific signal model and source position.
    """
    _t = None
    _trange = None
    _nb = None

    def __init__(self, sindec_bins, t_bins, rate_func, rndgen=None):
        # Use this outside as a default
        # hor = 0.25
        # sindec_bins = np.unique(np.concatenate([
        #                         np.linspace(-1., -hor, 5 + 1),    # south
        #                         np.linspace(-hor, +hor, 10 + 1),  # horizon
        #                         np.linspace(+hor, 1., 5 + 1),     # north
        #                         ]))

        # p_fix = 365.
        # t0_fix = np.amin(timesMJD)
        # rate_func = SinusFixedRateFunction(p_fix=p_fix, t0_fix=t0_fix)

        self._sindec_bins = np.atleast_1d(sindec_bins)
        self._t_bins = np.atleast_1d(t_bins)
        self._rate_func = rate_func
        self.rndgen = rndgen

    def fit(self, X, MC, srcs, run_dict):
        """
        Take data, MC and sources and build injection models. This is the place
        to actually stitch together a custom injector from the toolkit modules.

        Parameters
        ----------
        X : recarray
            Experimental data for BG injection.
        MC : recarray
            MC data for signal injection.
        srcs : recarray
            Source information.
        run_dict : dict
            Run information used in combination with data.
        """
        self._provided_data_names = X.dtype.names
        self._names = list(MC.keys())

        self._build_data_injector(X, srcs, run_dict)
        self._build_signal_injector(MC, srcs)

        return

    def get_sample(self, n_signal=None):
        """
        Get a complete data sample for one trial.

        Parameters
        ----------
        n_signal : int or None, opional
            How many signal events to sample in addition to the BG events.
            If ``None`` no signal is sampled. (default: ``None``)

        1. Get expected nb per source from allsky rate spline
        2. Sample times from allsky rate splines
        3. Sample same number of events from data using the CDF per source
           for the declination distribution
        4. Combine to a single recarray X
        5. Concat BG and signal samples
        Internally keep track of which event was injected from which injector
        """
        return

    def _build_data_injector(self, X, srcs, run_dict):
        """
        Create a time and declination dependent background model.

        Fit rate functions to time dependent rate in sindec bins. Normalize PDFs
        over the sindec range and fit splines to the fitted parameter points to
        continiously describe a rate model for a declination. Then choose a
        specific source time and build weights to inject according to the sindec
        dependent rate PDF from the whole pool of BG events.
        """
        sin_dec = np.sin(X["dec"])
        self._t = np.atleast_1d(srcs["t"])
        self._trange = np.vstack((srcs["dt0"], srcs["dt1"])).T
        assert len(self._trange) == len(self._t)

        # Cache expected nb for each source from allsky rate func integral
        pars_allsky = fit_time_dec_rate_models(
            timesMJD=X["timeMJD"], sin_decs=sin_dec, run_dict=run_dict,
            sin_dec_bins=[-1., 1.], rate_rebins=self._t_bins)[0]

        self._nb = self._rate_func.integral(self._t, self._trange, pars_allsky)
        assert len(self._nb) == len(self._t)

        # Build sampling CDFs from dec dependent rate func fits
        pars, std_devs, _ = fit_time_dec_rate_models(
            timesMJD=X["timeMJD"], sin_decs=sin_dec, run_dict=run_dict,
            sin_dec_bins=self._sindec_bins, rate_rebins=self._t_bins)
        # Normalize parameters to sinus declination bins
        norm = np.diff(self._sin_dec_bins)
        for n in ["amp", "base"]:
            pars[n] = pars[n] / norm
            std_devs[n] = std_devs[n] / norm
        # Fix the differential splines norm to match the allsky params. The
        # norm can be a bit off because each sindec fit is done individually.
        spl_norm = {"amp": pars_allsky["amp"], "base": pars_allsky["base"]}
        splines = make_time_dec_rate_model_splines(
            self._sin_dec_bins, pars, std_devs, spl_norm=spl_norm)

        # Build the data sampling weight CDFs by dividing the sindec PDF per
        # source by the data sindec PDF.


        return

    def _build_signal_injector(self, X, MC, srcs):
        """ The specific model for signal injection is encoded here """
        pass

    # def _setup_timedep_bg_rate_splines(self, T, sindec, sindec_bins, t_bins,
    #                                    rate_rec):
    #     """
    #     Create the weight CDFs for a time and declination dependent background
    #     injection model.

    #     Parameters
    #     ----------
    #     T : array-like
    #         Experimental data MJD times.
    #     sindec : array-like
    #         Experimental data sinus of declination.
    #     sindec_bins : array-like
    #         Explicit bin edges for the binning used to construct the dec
    #         dependent model.
    #     t_bins : array-like
    #         Explicit bins edges for the time binning used to fit the rate model.
    #     rate_rec : record-array
    #         Rate information as coming out of ``rebin_rate_rec``. Needs names
    #         ``'start_mjd', 'stop_mjd', 'rate'``.

    #     Returns
    #     -------

    #     """
    #     T = np.atleast_1d(T)
    #     sindec = np.atleast_1d(sindec)
    #     sindec_bins = np.atleast_1d(sindec_bins)

    #     # Rate model only varies in amplitude and baseline
    #     p_fix = 365.
    #     t0_fix = np.amin(T)
    #     rate_func = SinusFixedConstRateFunction(p_fix=p_fix, t0_fix=t0_fix,
    #                                             random_state=self._rndgen)

    #     # First an allsky fit to correctly renormalize the splines later
    #     rates, new_bins, rates_std, _ = rebin_rate_rec(
    #         rate_rec, rate_rec, bins=t_bins, ignore_zero_runs=True)
    #     t_mids = 0.5 * (new_bins[:-1] + new_bins[1:])
    #     fitres = rate_func.fit(t_mids, rates, x0=None, w=1. / rates_std)
    #     names = ["amplitude", "baseline"]
    #     allsky_fitpars = {n: fitres.x[i] for i, n in enumerate(names)}

    #     # For each sindec bin fit a model
    #     fit_pars = []
    #     fit_stds = []
    #     sindec_mids = []
    #     for i, (lo, hi) in enumerate(zip(sindec_bins[:-1], sindec_bins[1:])):
    #         mask = (sindec >= lo) & (sindec < hi)
    #         # Rebin rates and fit the model per sindec bin
    #         rates, new_bins, rates_std, _ = rebin_rate_rec(
    #             rate_rec[mask], bins=t_bins, ignore_zero_runs=True)

    #         t_mids = 0.5 * (new_bins[:-1] + new_bins[1:])
    #         fitres = rate_func.fit(t_mids, rates, x0=None, w=1. / rates_std)

    #         # Save stats for spline interpolation
    #         sindec_mids.append(0.5 * (lo + hi))
    #         fit_stds.append(np.sqrt(np.diag(fitres.hess_inv)))
    #         fit_pars.append(fitres.x)

    #     fit_pars = np.array(fit_pars).T
    #     fit_stds = np.array(fit_stds).T
    #     sindec_mids = np.array(sindec_mids)

    #     # Build a spline to continiously describe the rate model for each dec
    #     splines = {}
    #     lo, hi = sindec_bins[0], sindec_bins[-1]
    #     for i, (bp, std, n) in enumerate(zip(fit_pars, fit_stds, names)):
    #         # Amplitude and baseline must be in units HZ/dec, so that the
    #         # integral over declination gives back the allsky values
    #         norm = np.diff(sindec_bins)
    #         _bp = bp / norm
    #         _std = std / norm
    #         spl, norm, vals, pts = fit_spl_to_hist(_bp, sindec_bins, _std)
    #         splines[n] = self._spl_normed_factory(spl, lo=lo, hi=hi,
    #                                               norm=allsky_fitpars[n])

    #     return splines

    # def _spl_normed_factory(self, spl, lo, hi, norm):
    #     """
    #     Returns a renormalized spline so that
    #     ``int_lo^hi renorm_spl dx = norm``.

    #     Parameters
    #     ----------
    #     spl : scipy.interpolate.UnivariateSpline
    #         Scipy spline object.
    #     lo, hi : float
    #         Borders of the integration range.
    #     norm : float
    #         Norm the renormalized spline should have after renormalizing.

    #     Returns
    #     -------
    #     spl : utils.spl_normed
    #         New spline object with fewer feature set, but normalized over
    #         the range ``[lo, hi]``.
    #     """
    #     return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)


class MultiGRBModelInjector(MultiModelInjector):
    """
    Class holding multiple GRBModelInjector objects, managing the combined
    sampling from all single injectors.
    """
    _injectors = {}

    @property
    def names(self):
        return list(self._injectors.keys())

    @property
    def injectors(self):
        return self._injectors

    def fit(self, injectors):
        """
        Takes multiple single GRBLLHs in a dict and manages them.

        Parameters
        ----------
        injectors : dict
            Injectors to be managed by this multi Injector class. Names must
            match with dict keys needed by tested LLH.
        """
        for name, inj in injectors.items():
            if not isinstance(inj, GRBModelInjector):
                raise ValueError("Injector object `{}`".format(name) +
                                 " is not of type `GRBModelInjector`.")

    def get_sample(self, n_signal=None):
        raise NotImplementedError("TODO")


# ##############################################################################
# Code to sample from multiple single signal injector instances over their
# public methods.
# ##############################################################################
# # We need to re-normalize w_theo over all samples instead of all sources in a
# # single samples for a single injector, because sources are disjunct in each one
# def get_raw_fluxes(injectors):
#     # Split original flux over sources per sample.
#     # List of arrays, fluxes per sample, per source
#     w_theos = [inj._srcs[-1]["w_theo"] / inj._srcs[-1]["w_theo"].sum() for inj in injectors]
#     raw_fluxes = np.array([wts / inj.mu2flux(1.) for inj, wts in zip(injectors, w_theos)])
#     # Renormalize w_theos over all samples
#     w_theos = [inj._srcs[-1]["w_theo"] for inj in injectors]
#     w_theo_sum = np.sum(map(np.sum, w_theos))
#     w_theos = [wt / w_theo_sum for wt in w_theos]
#     # Renormalize fluxes per sample per source with renormalized w_theo weights
#     raw_fluxes = np.array([raw_f * wt for raw_f, wt in zip(raw_fluxes, w_theos)])
#     # Combine to decreased raw flux per sample
#     raw_fluxes = np.array(map(np.sum, raw_fluxes))
#     return raw_fluxes, w_theos

# def get_sample_w(injectors):
#     # Get the renormalized fluxes and normalize as normal now
#     raw_fluxes, _ = get_raw_fluxes(injectors)
#     return raw_fluxes / raw_fluxes.sum()

# def distribute_samples(n, injectors):
#     p = get_sample_w(injectors)
#     return np.random.multinomial(n, p, size=None)

# def mu2flux(mu, injectors):
#     raw_fluxes, _ = get_raw_fluxes(injectors)
#     return mu / np.sum(raw_fluxes)
