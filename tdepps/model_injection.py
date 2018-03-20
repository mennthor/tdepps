# coding: utf-8

from __future__ import print_function, division

import abc
import numpy as np
from numpy.lib.recfunctions import drop_fields
from sklearn.utils import check_random_state

from .model_toolkit import make_time_dep_dec_splines
from .utils import fit_spl_to_hist, random_choice, arr2str, fill_dict_defaults


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
    _nsrcs = None
    _src_t = None
    _src_trange = None
    _allsky_rate_func = None
    _allsky_pars = None
    _nb = None

    # Debug info
    _bg_inj = None
    _sig_inj = None
    _data_spl = None
    _sin_dec_splines = None
    _param_splines = None
    _best_pars = None
    _best_stddevs = None

    def __init__(self, bg_inj_args, rndgen=None):
        self._provided_data_names = np.array(
            ["timeMJD", "dec", "ra", "sigma", "logE"])

        # Check BG inj args
        req_keys = ["sindec_bins", "rate_rebins"]
        opt_keys = {}
        self.bg_inj_args = fill_dict_defaults(bg_inj_args, req_keys, opt_keys)

        self._sin_dec_bins = np.atleast_1d(self.bg_inj_args["sindec_bins"])
        self._rate_rebins = np.atleast_1d(self.bg_inj_args["rate_rebins"])
        self.rndgen = rndgen

    def fit(self, X, srcs, run_dict, sig_inj):
        """
        Take data, MC and sources and build injection models. This is the place
        to actually stitch together a custom injector from the toolkit modules.

        Parameters
        ----------
        X : recarray
            Experimental data for BG injection.
        srcs : recarray
            Source information.
        run_dict : dict
            Run information used in combination with data.
        sig_inj : SignalFluenceInjector
            Ready to sample SignalFluenceInjector instance
        """
        X_names = np.array(X.dtype.names)
        for name in self._provided_data_names:
            if name not in X_names:
                raise ValueError("`X` is missing name '{}'.".format(name))
        drop = np.isin(X_names, self._provided_data_names,
                       assume_unique=True, invert=True)
        drop_names = X_names[drop]
        print("Dropping not needed names '{}' from data recarray.".format(
            arr2str(drop_names)))

        self.X = drop_fields(X, drop_names, usemask=False)
        self.srcs = srcs
        self._setup_data_injector(self.X, self.srcs, run_dict)

        self._sig_inj = sig_inj

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
        # Get number of BG events to sample in this trial
        expected_evts = self._rndgen.poisson(self._nb)
        # Sample times from rate function for all sources
        times = self._allsky_rate_func.sample(expected_evts)

        sam = []
        for j in range(self._nsrcs):
            nevts = expected_evts[j]
            if nevts > 0:
                # Resample dec, logE, sigma from exp data with each source CDF
                idx = random_choice(self._rndgen, CDF=self._sample_CDFs[j],
                                    n=nevts)
                sam_i = self.X[idx]
                # Sample missing ra uniformly
                sam_i["ra"] = self._rndgen.uniform(0., 2. * np.pi, size=nevts)
                # Append times
                sam_i["timeMJD"] = times[j]
            else:
                sam_i = np.empty((0,), dtype=[(n, float) for n in
                                              self._provided_data_names])
            sam.append(sam_i)

        self._bg_inj = sam

        # Make signal contribution
        if n_signal > 0:
            sig = self._sig_inj.sample(n_signal)
            sam.append(sig)
            self._sig_inj = sig

        # Concat to a single recarray
        return np.concatenate(sam)

    def _setup_data_injector(self, X, srcs, run_dict):
        """
        Create a time and declination dependent background model.

        Fit rate functions to time dependent rate in sindec bins. Normalize PDFs
        over the sindec range and fit splines to the fitted parameter points to
        continiously describe a rate model for a declination. Then choose a
        specific source time and build weights to inject according to the sindec
        dependent rate PDF from the whole pool of BG events.
        """
        ev_t = X["timeMJD"]
        ev_sin_dec = np.sin(X["dec"])
        self._nsrcs = len(srcs)
        self._src_t = np.atleast_1d(srcs["t"])
        self._src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T
        assert len(self._src_trange) == len(self._src_t) == self._nsrcs

        # Get sindec PDF spline for each source, averaged over its time window
        sin_dec_splines, info = make_time_dep_dec_splines(
            ev_t, ev_sin_dec, srcs, run_dict, self._sin_dec_bins, self._rate_rebins)

        # Cache expected nb for each source from allsky rate func integral
        self._param_splines = info["param_splines"]
        self._best_pars = info["best_pars"]
        self._best_stddevs = info["best_stddevs"]
        self._allsky_rate_func = info["allsky_rate_func"]
        self._allsky_pars = info["allsky_best_params"]
        self._nb = self._allsky_rate_func.integral(
            self._src_t, self._src_trange, self._allsky_pars)
        assert len(self._nb) == len(self._src_t)

        # Make sampling CDFs to sample sindecs per source per trial
        # Spline to estimate intrinsic data sindec distribution
        hist = np.histogram(ev_sin_dec, bins=self._sin_dec_bins,
                            density=False)[0]
        stddev = np.sqrt(hist)
        norm = np.diff(self._sin_dec_bins) * np.sum(hist)
        hist = hist / norm
        stddev = stddev / norm
        data_spl = fit_spl_to_hist(hist, bins=self._sin_dec_bins,
                                   stddev=stddev)
        self._sin_dec_splines = sin_dec_splines
        self._data_spl = data_spl

        # Build sampling weights from PDF ratios
        sample_w = np.empty((len(sin_dec_splines), len(ev_sin_dec)),
                            dtype=float)
        for i, spl in enumerate(sin_dec_splines):
            sample_w[i] = spl(ev_sin_dec) / data_spl(ev_sin_dec)

        # Cache fixed sampling CDFs for fast random choice
        CDFs = np.cumsum(sample_w, axis=1)
        self._sample_CDFs = CDFs / CDFs[:, [-1]]

        return


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
