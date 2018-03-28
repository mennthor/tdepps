# coding: utf-8

from __future__ import print_function, division, absolute_import

import abc
import numpy as np
from numpy.lib.recfunctions import drop_fields
from sklearn.utils import check_random_state

from .toolkit import MultiSignalFluenceInjector
from .utils import (fit_spl_to_hist, random_choice, make_time_dep_dec_splines,
                    fill_dict_defaults, arr2str, logger)


class BaseModelInjector(object):
    """ Interface for ModelInjector type classes """
    __metaclass__ = abc.ABCMeta

    _rndgen = None
    _provided_data_names = None
    _sig_inj = None

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

    @property
    def sig_inj(self):
        return self._sig_inj

    @abc.abstractmethod
    def fit(self):
        """ Sets up the injector and makes it ready to use """
        pass

    @abc.abstractmethod
    def get_sample(self):
        """ Get a data sample for a single trial to use in a LLH object """
        pass


class BaseMultiModelInjector(BaseModelInjector):
    """ Interface for managing multiple LLH type classes """
    _names = None

    @property
    @abc.abstractmethod
    def names(self):
        """ Subinjector names, identifies this as a MultiModelInjector """
        pass


# #############################################################################
# GRB style injector
# #############################################################################
class GRBModelInjector(BaseModelInjector):
    """
    Coordinates injection from background and signal injectors.
    """
    def __init__(self, bg_inj_args, rndgen=None):
        # Check BG inj args
        req_keys = ["sindec_bins", "rate_rebins"]
        opt_keys = {"spl_s": None}
        self.bg_inj_args = fill_dict_defaults(bg_inj_args, req_keys, opt_keys)

        self._sin_dec_bins = np.atleast_1d(self.bg_inj_args["sindec_bins"])
        self._rate_rebins = np.atleast_1d(self.bg_inj_args["rate_rebins"])
        self.rndgen = rndgen

        self._provided_data_names = np.array(
            ["timeMJD", "dec", "ra", "sigma", "logE"])

        self._log = logger(name=self.__class__.__name__, level="ALL")

        # Private attribute defaults
        self._nsrcs = None
        self._src_t = None
        self._src_trange = None
        self._allsky_rate_func = None
        self._allsky_pars = None
        self._nb = None

        # Debug info
        self._bg_cur_sam = None
        self._sig_cur_sam = None
        self._data_spl = None
        self._sin_dec_splines = None
        self._param_splines = None
        self._best_pars = None
        self._best_stddevs = None

        return

    def fit(self, bg_inj, sig_inj):
        """
        Take data, MC and sources and build injection models. This is the place
        to actually stitch together a custom injector from the toolkit modules.

        Parameters
        ----------
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
        print(self._INFO_("Dropping names '{}' ".format(arr2str(drop_names)) +
                          "from data recarray."))

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

        self._bg_cur_sam = sam

        # Make signal contribution
        if n_signal > 0:
            sig = self._sig_inj.sample(n_signal)
            sam.append(sig)
            self._sig_cur_sam = sig

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
        print(self._INFO_("Create time dep sindec splines."))
        sin_dec_splines, info = make_time_dep_dec_splines(
            ev_t, ev_sin_dec, srcs, run_dict, self._sin_dec_bins,
            self._rate_rebins, spl_s=self.bg_inj_args["spl_s"])

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
        hist = np.histogram(ev_sin_dec, bins=self._sin_dec_bins,
                            density=False)[0]
        stddev = np.sqrt(hist)
        norm = np.diff(self._sin_dec_bins) * np.sum(hist)
        hist = hist / norm
        stddev = stddev / norm
        weight = 1. / stddev
        # Spline to estimate intrinsic data sindec distribution
        data_spl = fit_spl_to_hist(hist, bins=self._sin_dec_bins, w=weight)[0]
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


class MultiGRBModelInjector(BaseMultiModelInjector):
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

        self._injectors = injectors

        # Additionaly collect sub signal injectors in the multi injector
        # TODO: This injector should group instances of bg and sig injectors
        # but the bg injector is implemented in the GRB injector...
        # When this is outsourced, it can get more modular again.
        self._sig_inj = MultiSignalFluenceInjector()
        self._sig_inj.fit({n: inj.sig_inj for n, inj
                           in self._injectors.items()})

    def get_sample(self, n_signal=None):
        """
        Split background samples manually
        """
        # We need to re-normalize w_theo over all samples instead of all sources in a
        # single samples for a single injector, because sources are disjunct in each one
        pass
