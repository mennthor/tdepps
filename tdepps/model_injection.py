# coding: utf-8

from __future__ import print_function, division, absolute_import
from future.utils import viewkeys
from future import standard_library
standard_library.install_aliases()

import numpy as np
from sklearn.utils import check_random_state
import abc

from model_toolkit import (SignalFluenceInjector, ResampleBGDataInjector,
                           SinusFixedConstRateFunction, rebin_rate_rec,
                           make_rate_records)
from utils import spl_normed, fit_spl_to_hist


class Injector(object):
    """ Interface for injection type classes. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit():
        raise NotImplementedError

    @abc.abstractmethod
    def get_sample():
        raise NotImplementedError


class GRBInjectionModel(Injector):
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

    Parameters
    ----------
    X : dict
    MC : dict
    srcs : dict
    """
    def __init__(self, rndgen=None):
        self._bg_injectors = None
        self._sig_injectors = None
        self._names
        # Settings?
        return

    @property
    def names(self):
        return self._names

    @property
    def bg_injectors(self):
        return self._bg_injectors

    @property
    def signal_injectors(self):
        return self._sig_injectors

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    def fit(self, X, MC, srcs, run_dicts):
        """
        Take data, MC and sources and build injection models. This is the place
        to actually stitch together a custom injector from the toolkit modlues.
        """
        # Keys must all be equivalent
        if viewkeys(X) != viewkeys(MC):
            raise ValueError("Keys in `X` and `MC` don't match.")
        if viewkeys(srcs) != viewkeys(MC):
            raise ValueError("Keys in `MC` and `srcs` don't match.")
        if viewkeys(run_dicts) != viewkeys(MC):
            raise ValueError("Keys in `MC` and `run_dicts` don't match.")

        self._names = list(viewkeys(MC))

        self._build_data_injectors(X, srcs, run_dicts)
        self._build_signal_injectors(MC, srcs)

        return

    def get_sample(self, n_signal=None):
        """
        Get a complete data sample for one trial.

        Parameters
        ----------
        n_signal : int or None, opional
            How many signal events shall be sampled additional to the BG events.
            IF ``None`` no events are sampled. If signal is sampled, it just
            get's blend into the background sample. (default: ``None``)
        """
        # TODO: Concat BG and signal samples
        # Internally keep track of which event was injected from which injector
        return

    def _build_data_injectors(self, X, srcs, run_dicts):
        """
        Create a time and declination dependent background model.

        Fit rate functions to time dependent rate in sindec bins. Normalize PDFs
        over the sindec range and fit splines to the fitted parameter points to
        continiously describe a rate model for a declination. Then choose a
        specific source time and build weights to inject according to the sindec
        dependent rate PDF from the whole pool of BG events.
        """
        # Custom sindec binning, equal for all samples, finer at the horizon
        hor = 0.25
        sindec_bins = np.unique(np.concatenate([
                                np.linspace(-1., -hor, 5 + 1),    # south
                                np.linspace(-hor, +hor, 10 + 1),  # horizon
                                np.linspace(+hor, 1., 5 + 1),     # north
                                ]))


        print("Setup background injectors:")
        splines = {}
        for key in self._names:
            print(" - Sample '{}':".format(key))
            # For each sample build a spline model for the rate function params
            # describing the declination dependent bg rate
            T = X[key]["timeMJD"]
            sindec = np.sin(X[key]["dec"])
            # Time binning for stable rate function fits
            t_bins = np.linspace(np.amin(T), np.amax(T), 12 + 1)
            # Make model parameter splines
            rate_rec = make_rate_records(T, run_dicts[key])
            splines[key] = self._setup_timedep_bg_rate_splines(
                T, sindec, sindec_bins, t_bins, rate_rec)
            print("   + Built dec dependent rate spline".format(len()))

            # Get the background rate for each source per sample and setup the
            # background sampler using the correct weights per source

        return

    def _build_signal_injectors(self, X, MC, srcs):
        """ The specific model for signal injection is encoded here """
        # Create single injectors
        _sig_injectors = {}
        for name in self._names:
            pass
        self._sig_injectors = _sig_injectors
        return

    def _setup_timedep_bg_rate_splines(self, T, sindec, sindec_bins, t_bins,
                                       rate_rec):
        """
        Create the weight CDFs for a time and declination dependent background
        injection model.

        Parameters
        ----------
        T : array-like
            Experimental data MJD times.
        sindec : array-like
            Experimental data sinus of declination.
        sindec_bins : array-like
            Explicit bin edges for the binning used to construct the dec
            dependent model.
        t_bins : array-like
            Explicit bins edges for the time binning used to fit the rate model.
        rate_rec : record-array
            Rate information as coming out of ``rebin_rate_rec``. Needs names
            ``'start_mjd', 'stop_mjd', 'rate'``.

        Returns
        -------

        """
        T = np.atleast_1d(T)
        sindec = np.atleast_1d(sindec)
        sindec_bins = np.atleast_1d(sindec_bins)

        # Rate model only varies in amplitude and baseline
        p_fix = 365.
        t0_fix = np.amin(T)
        rate_func = SinusFixedConstRateFunction(p_fix=p_fix, t0_fix=t0_fix,
                                                random_state=self._rndgen)

        # First an allsky fit to correctly renormalize the splines later
        rates, new_bins, rates_std, _ = rebin_rate_rec(
            rate_rec, rate_rec, bins=t_bins, ignore_zero_runs=True)
        t_mids = 0.5 * (new_bins[:-1] + new_bins[1:])
        fitres = rate_func.fit(t_mids, rates, x0=None, w=1. / rates_std)
        names = ["amplitude", "baseline"]
        allsky_fitpars = {n: fitres.x[i] for i, n in enumerate(names)}

        # For each sindec bin fit a model
        fit_pars = []
        fit_stds = []
        sindec_mids = []
        for i, (lo, hi) in enumerate(zip(sindec_bins[:-1], sindec_bins[1:])):
            mask = (sindec >= lo) & (sindec < hi)
            # Rebin rates and fit the model per sindec bin
            rates, new_bins, rates_std, _ = rebin_rate_rec(
                rate_rec[mask], bins=t_bins, ignore_zero_runs=True)

            t_mids = 0.5 * (new_bins[:-1] + new_bins[1:])
            fitres = rate_func.fit(t_mids, rates, x0=None, w=1. / rates_std)

            # Save stats for spline interpolation
            sindec_mids.append(0.5 * (lo + hi))
            fit_stds.append(np.sqrt(np.diag(fitres.hess_inv)))
            fit_pars.append(fitres.x)

        fit_pars = np.array(fit_pars).T
        fit_stds = np.array(fit_stds).T
        sindec_mids = np.array(sindec_mids)

        # Build a spline to continiously describe the rate model for each dec
        splines = {}
        lo, hi = sindec_bins[0], sindec_bins[-1]
        for i, (bp, std, n) in enumerate(zip(fit_pars, fit_stds, names)):
            # Amplitude and baseline must be in units HZ/dec, so that the
            # integral over declination gives back the allsky values
            norm = np.diff(sindec_bins)
            _bp = bp / norm
            _std = std / norm
            spl, norm, vals, pts = fit_spl_to_hist(_bp, sindec_bins, _std)
            splines[n] = self._spl_normed_factory(spl, lo=lo, hi=hi,
                                                  norm=allsky_fitpars[n])

        return splines

        def _spl_normed_factory(self, spl, lo, hi, norm):
            """
            Returns a renormalized spline so that
            ``int_lo^hi renorm_spl dx = norm``.

            Parameters
            ----------
            spl : scipy.interpolate.UnivariateSpline
                Scipy spline object.
            lo, hi : float
                Borders of the integration range.
            norm : float
                Norm the renormalized spline should have after renormalizing.

            Returns
            -------
            spl : utils.spl_normed
                New spline object with fewer feature set, but normalized over
                the range ``[lo, hi]``.
            """
            return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)



# class MultiGRBInjectionModel(object):
#     """
#     Class holding multiple injector model objects. All must have either BG only
#     or BG and signal injectors for each sample.
#     This implements backgorund and signal injection for several samples using
#     correct weighting between the samples.
#     """
#     def __init__(self):
#         self._injectors = {}
#         self._ns_weights = None
#         self._has_signal_model = None

#     @property
#     def names(self):
#         return list(self._injectors.keys())

#     @property
#     def injectors(self):
#         return self._injectors

#     @property
#     def has_signal_model(self):
#         return self._has_signal_model

#     def add_injector(self, name, injector):
#         if not isinstance(injector, GRBInjectionModel):
#             raise ValueError("`injector` object must be of type " +
#                              "GRBInjectionModel.")

#         if name in self.names:
#             raise KeyError("Name '{}' has already been added. ".format(name) +
#                            "Choose a different name.")

#         # Check if new injector is consistent with the already added ones
#         if self._has_signal_model is not None:
#             if (self._has_signal_model is True and
#                     injector.has_signal_model is False):
#                 raise ValueError("Added injectors have signal models, but " +
#                                  "`injector` only has a BG model.")
#             if (self._has_signal_model is False and
#                     injector.has_signal_model is True):
#                 raise ValueError("Added injectors have only BG models, but " +
#                                  "`injector` has a signal model.")
#         else:
#             self._has_signal_model = injector.has_signal_model

#         self._injectors[name] = injector

#     def get_sample(self):
#         # TODO: Concat all samples and adapt MultiLLH so it understands the
#         # format
        # return

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

# has_signal needed? -> get_sample(signal=True)
# class GRBInjectionModel(object):
#     """
#     Wrapper class to combine signal and backgorund models to use as a whole.
#     Single PDFs are replaced by ratios.

#     Implements `get_sample()`, `get_soverb()` and `get_args()`
#     """
#     def __init__(self, bg_model, signal_model=None):
#         if signal_model is None:
#             self._has_signal_model = False
#         else:
#             self._has_signal_model = True

#         self.bg = bg_model
#         self.sig = signal_model

#     @property
#     def has_signal_model(self):
#         return self._has_signal_model


# Old bg_injector multisampler
# class MultiGeneralPurposeInjector(object):
#     """
#     Container class that holds single instances of GeneralPurposeInjectors.
#     """
#     def __init__(self):
#         self._injs = {}
#         return

#     @property
#     def names(self):
#         return list(self._injs.keys())

#     @property
#     def llhs(self):
#         return list(self._injs.values())

#     def add_injector(self, name, inj):
#         """
#         Add a injector object to consider.

#         Parameters
#         ----------
#         name : str
#             Name of the inj object. Should be connected to the dataset used.
#         inj : tdepps.bg_injector.GeneralPurposeInjector
#             BG injector object sampling pseudo BG events.
#         """
#         if not isinstance(inj, GeneralPurposeInjector):
#             raise ValueError("`inj` object must be of type GeneralPurposeInjector.")

#         if inj._n_features is None:
#             raise RuntimeError("Injector must be fitted before adding.")

#         if name in self.names:
#             raise KeyError("Name '{}' has already been added. ".format(name) +
#                            "Choose a different name.")
#         else:
#             self._injs[name] = inj

#         return

#     def sample(self, n_samples=1):
#         """
#         Call each added injector's sample method and wrap the sampled arrays in
#         dictionaries for use in ``MultiSampleGRBLLH``.

#         Parameters
#         ----------
#         n_samples : dict
#             Number of samples to generate per injector. Dictionary keys must
#             match added ``self.names``.

#         Returns
#         -------
#         sam_ev : dictionary
#             Sampled events from each added ``GeneralPurposeInjector``.
#         """
#         if len(self.names) == 0:
#             raise ValueError("No injector has been added yet.")

#         if viewkeys(n_samples) != viewkeys(self._injs):
#             raise ValueError("Given `n_samples` has not the same keys as " +
#                              "stored injectors names.")

#         sam_ev = {}
#         for name in self.names:
#             # Get per sample information
#             inj = self._injs[name]
#             sam_ev[name] = inj.sample(n_samples[name])

#         return sam_ev
