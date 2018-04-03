# coding: utf-8

from __future__ import absolute_import

import numpy as np

from ..backend import soverb_time_box, pdf_spatial_signal
from ..base import BaseModel
from ..utils import (fill_dict_defaults, make_time_dep_dec_splines, logger,
                     make_grid_interp_from_hist_ratio, power_law_flux)


class GRBModel(BaseModel):
    """
    Models the PDF part for the GRB LLH function.
    """
    def __init__(self, X, MC, srcs, run_dict, flux_model, spatial_opts=None,
                 energy_opts=None):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        try:
            flux_model(1.)  # Standard units are E = 1GeV
        except Exception:
            raise TypeError("`model` must be a function `f(trueE)`.")

        # Check and setup spatial PDF options
        req_keys = ["sindec_bins", "rate_rebins"]
        opt_keys = {"select_ev_sigma": 5., "spl_s": None, "n_scan_bins": 50}
        spatial_opts = fill_dict_defaults(spatial_opts, req_keys, opt_keys)

        spatial_opts["sindec_bins"] = np.atleast_1d(spatial_opts["sindec_bins"])
        spatial_opts["rate_rebins"] = np.atleast_1d(spatial_opts["rate_rebins"])

        if spatial_opts["select_ev_sigma"] <= 0.:
            raise ValueError("'select_ev_sigma' must be > 0.")

        if (spatial_opts["spl_s"] is not None and spatial_opts["spl_s"] < 0):
            raise ValueError("'spl_s' must be `None` or >= 0.")

        if spatial_opts["n_scan_bins"] < 20:
            raise ValueError("'n_scan_bins' should be > 20 for proper scans.")

        # Check and setup energy PDF options
        req_keys = ["bins"]
        opt_keys = {"mc_bg_weights": None, "force_logE_asc": True,
                    "edge_fillval": "minmax_col", "interp_col_log": False}
        energy_opts = fill_dict_defaults(energy_opts, req_keys, opt_keys)

        sin_dec_bins, logE_bins = map(np.atleast_1d, energy_opts["bins"])
        energy_opts["bins"] = [sin_dec_bins, logE_bins]

        if (energy_opts["edge_fillval"] not in
                ["minmax", "col", "minmax_col", "min"]):
            raise ValueError("'edge_fillval' must be one of " +
                             "['minmax'|'col'|'minmax_col'|'min'].")

        if energy_opts["edge_fillval"] == "min" and energy_opts["force_y_asc"]:
            raise ValueError("`edge_fillval` is 'min' and 'force_y_asc' is " +
                             "`True`, which doesn't make sense together.")

        if len(energy_opts["bins"]) != 2:
            raise ValueError("Bins for energy hist must be of format " +
                             "`[sin_dec_bins, logE_bins]`.")

        if np.any(sin_dec_bins < -1.) or np.any(sin_dec_bins > 1.):
            raise ValueError("sinDec declination bins for energy hist not " +
                             "in valid range `[-1, 1]`.")

        if energy_opts["mc_bg_w"] is None:
            # Energy interpolator is build on data and MC for signal only
            for sd, name in zip([np.sin(X["dec"]), np.sin(MC["dec"])],
                                ["data", "MC"]):
                if np.any((sd < sin_dec_bins[0]) | (sd > sin_dec_bins[-1])):
                    raise ValueError("dec " + name + " events outside " +
                                     "given bins for energy hist. If this is " +
                                     "intended, please remove them beforehand.")
            for logE, name in zip([X["logE"], MC["logE"]], ["data", "MC"]):
                if np.any((logE < logE_bins[0]) | (logE > logE_bins[-1])):
                    raise ValueError("logE " + name + " events outside " +
                                     "given bins for energy hist. If this is " +
                                     "intended, please remove them beforehand.")
            sin_dec_bg, logE_bg, w_bg = X["dec"], X["logE"], np.ones(len(X))
        else:
            # Energy interpolator is build on MC for both bg and signal
            if len(energy_opts["mc_bg_w"]) != len(MC):
                raise ValueError("Length of MC BG weights and MC must match.")
            if np.any((MC["dec"] < sin_dec_bins[0]) |
                      (MC["dec"] > sin_dec_bins[-1])):
                raise ValueError("dec MC events outside given bins for " +
                                 "energy hist. If this is intended, please " +
                                 "remove them beforehand.")
            if np.any((MC["logE"] < logE_bins[0]) |
                      (MC["logE"] > logE_bins[-1])):
                raise ValueError("logE MC events outside given bins for " +
                                 "energy hist. If this is intended, please " +
                                 "remove them beforehand.")

        self._flux_model = flux_model
        self._spatial_opts = spatial_opts
        self._energy_opts = energy_opts

        self._needed_data = np.array(
            ["timeMJD", "dec", "ra", "sigma", "logE"])
        self._provided_args = ["src_w_dec", "src_w_theo" "nb"]

        # Setup internals for model evaluation
        self._log = logger(name=self.__class__.__name__, level="ALL")
        _out = self._setup_model(X, MC, srcs, run_dict)
        self._nb, self._sin_dec_splines, _, self._energy_interpol = _out

        # Cache repeatedly used values
        self._src_dt = np.vstack((srcs["dt0"], srcs["dt1"])).T
        self._src_dec_col_vec = self._srcs["dec"][:, None]
        self._srcs = srcs

    @property
    def needed_data(self):
        return self._needed_data

    @property
    def provided_args(self):
        return self._provided_args

    @property
    def srcs(self):
        """ Source recarray the injector was fitted to """
        return self._srcs

    @property
    def model_opts(self):
        return self._model_opts

    def get_args(self):
        pass

    def get_soverb(self, X):
        """
        Calculate sob values per source per event for given data X
        """
        # Preselect data to save computation time
        X = self._select_X(X)

        # Make combined PDF term
        sob = (self._soverb_time(X["timeMJD"]) *
               self._soverb_spatial(X["ra"], np.sin(X["dec"]), X["sigma"]) *
               self._soverb_energy(np.sin(X["dec"]), X["logE"]))
        return sob

    def _select_X(self, X):
        """
        Only select events which are closer than ``select_ev_sigma * sigma`` to
        the corresponding source. The gaussian (or kent) spatial signal term
        makes these event not contributing anyway (if nsigma is high enough).
        Selection only in declination bands.
        """
        dec_mask = ((X["dec"] > self._src_dec_col_vec - X["sigma"] *
                     self._spatial_opts["select_ev_sigma"]) &
                    (X["dec"] < self._src_dec_col_vec + X["sigma"] *
                     self._spatial_opts["select_ev_sigma"]))
        return X[dec_mask]

    def _soverb_time(self, t):
        """
        Time signal over background ratio for a simple box model.

        Parameters
        ----------
        t : array-like
            Event times given in MJD for which we want to evaluate the ratio.

        Returns
        -------
        soverb_time_ratio : array-like, shape (nsrcs, len(t))
            Ratio of the time signal and background PDF for each given time `t`
            and per source time `src_t`.
        """
        return soverb_time_box(
            t, self._srcs["t"], self._srcs["dt0"], self._srcs["dt1"])

    def _soverb_spatial(self, ev_ra, ev_sin_dec, ev_sig):
        """
        Spatial signal over background ratio.

        The signal PDF is a 2D Kent distribution (or 2D gaussian), normalized to
        the unit sphere area. It depends on the great circle distance between
        an event and a source postition.

        The background PDF is only declination dependent (detector rotational
        symmetry) and is created from the experimental data sinus declination
        distribution. It only depends on the events declination.

        Parameters
        ----------
        ev_ra, ev_sin_dec : array-like, shape (nevts)
            Event positions in equatorial right-ascension, [0, 2pi] in radian
            and sinus declination, [-1, 1].
        ev_sig : array-like, shape (nevts)
            Event positional reconstruction errors in radian (eg. Paraboloid).

        Returns
        -------
        soverb_spatial_ratio : array-like, shape (nsrcs, nevts)
            Ratio of the spatial signal and background PDF for each given event
            and for each source position.
        """
        sob = pdf_spatial_signal(
            self._srcs["ra"], self._src["dec"], ev_ra, ev_sin_dec, ev_sig,
            self._spatial_args["kent"])
        # Divide by background PDF per source
        for j, sobi in enumerate(sob):
            Bi = np.exp(self._spatial_bg_spls[j](ev_sin_dec)) / (2. * np.pi)
            sob[j] = sobi / Bi

        return sob

    def _soverb_energy(self, ev_sin_dec, ev_logE):
        """
        Energy signal over background ratio.

        Energy has a lot of seperation power, because signal is following an
        astrophysical flux, which becomes dominant at higher energies over the
        flux of atmospheric background neutrinos.

        To account for different source positions on the whole sky, we create
        2D PDFs in sinus declination and in log10 of an energy estimator.

        Outside of the definition range, the PDF is set to zero.

        Parameters
        ----------
        ev_sin_dec
            See :py:meth:`GRBLLH._soverb_spatial`, Parameters
        ev_logE
            See :py:meth:`lnllh_ratio`, Parameters: `X`

        Returns
        -------
        soverb_energy_ratio : array-like, shape (nevts)
            Ratio of the energy signal and background PDF for each given event.
        """
        return np.exp(self._energy_interpol(np.vstack((ev_sin_dec, ev_logE)).T))

    def _setup_model(self, X, MC, srcs, run_dict):
        """
        Create the splines and interpolators used to evaluate the LLH model.

        The energy PDF is built using 2D histogram ratios for bg and signal and
        interpolate the histogram.
        The background spatial PDFs are built per source by fitting a spline
        to a sindec modelas done in the ``grb.TimeDecDependentBGDataInjector``.

        Returns
        -------
        nb : array-like
            Expected background events per source.
        sin_dec_splines : list of splines
            Splines per source describing the spatial background PDF in  in
            ``sin(dec)``.
        spl_info : dict
            Collection of spline and rate fit information.
        energy_interpol : scipy.interpolate.RegularGridInterpolator
            Interpolator returning the energy signal over background ratio for
            ``sin(dec), logE`` pairs.
        """
        # Step 1: Build per source spatial BG estimation and sindec PDFs
        ev_t = X["timeMJD"]
        ev_sin_dec = np.sin(X["dec"])
        src_t = np.atleast_1d(srcs["t"])
        src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T

        sin_dec_bins = self._spatial_opts["sindec_bins"]
        rate_rebins = self._spatial_opts["rate_rebins"]

        # Get sindec PDF spline for each source, averaged over its time window
        print(self._log.INFO("Create time dep sindec splines."))
        sin_dec_splines, spl_info = make_time_dep_dec_splines(
            ev_t, ev_sin_dec, srcs, run_dict, sin_dec_bins, rate_rebins,
            spl_s=self._spatial_opts["spl_s"],
            n_scan_bins=self._spatial_opts["n_scan_bins"])

        # Cache expected nb for each source from allsky rate func integral
        nb = spl_info["allsky_rate_func"].integral(
            src_t, src_trange, spl_info["allsky_best_params"])
        assert len(nb) == len(src_t)

        # Step 2: Build energy PDF interpolator
        # Make histograms, signal weighted to flux model
        sin_dec_bg = np.sin(X["dec"])
        logE_bg = np.sin(X["logE"])
        w_bg = self._energy_opts["mc_bg_w"]
        sin_dec_sig = np.sin(MC["dec"])
        logE_sig = np.sin(MC["logE"])
        w_sig = MC["ow"] * self._flux_model(MC["trueE"])

        h_bg, _, _ = np.histogram2d(sin_dec_bg, logE_bg, weights=w_bg,
                                    bins=self._energy_opts["bins"], normed=True)
        h_sig, _, _ = np.histogram2d(sin_dec_sig, logE_sig, weights=w_sig,
                                     bins=self._energy_opts["bins"],
                                     normed=True)

        energy_interpol = make_grid_interp_from_hist_ratio(
            h_bg, h_sig, bins=self._energy_opts["bins"],
            edge_fillval=self._energy_opts["edge_fillval"],
            interp_col_log=self._energy_opts["interp_col_log"],
            force_y_asc=self._energy_opts["force_y_asc"])

        return nb, sin_dec_splines, spl_info, energy_interpol
