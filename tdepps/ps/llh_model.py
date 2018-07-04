# coding: utf-8

from __future__ import absolute_import

import numpy as np

from ..backend import pdf_spatial_signal
from ..base import BaseModel
from ..utils import fill_dict_defaults, logger
from ..utils import spl_normed, fit_spl_to_hist, make_equdist_bins
from ..utils import make_grid_interp_from_hist_ratio


class PSModel(BaseModel):
    """
    Models the PDF part for the time integrated PS LLH function.
    """
    def __init__(self, model_opts=None):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        # Check and setup spatial PDF options
        req_keys = ["bins", "flux_model"]
        opt_keys = {"select_ev_sigma": 5., "kent": True,
                    "n_mc_evts_min": 500, "n_data_evts_min": 100,
                    "mc_bg_w": None, "force_logE_asc": True,
                    "edge_fillval": "minmax_col", "interp_col_log": False}
        model_opts = fill_dict_defaults(model_opts, req_keys, opt_keys)

        sin_dec_bins, logE_bins = map(np.atleast_1d, model_opts["bins"])
        model_opts["bins"] = [sin_dec_bins, logE_bins]

        if model_opts["select_ev_sigma"] <= 0.:
            raise ValueError("'select_ev_sigma' must be > 0.")

        if model_opts["n_mc_evts_min"] < 1:
            raise ValueError("'n_mc_evts_min' must > 0.")
        if model_opts["n_data_evts_min"] < 1:
            raise ValueError("'n_data_evts_min' must > 0.")

        try:
            model_opts["flux_model"](1.)  # Standard units are E = 1GeV
        except Exception:
            raise TypeError("'flux_model' must be a function `f(trueE)`.")

        if (model_opts["edge_fillval"] not in
                ["minmax", "col", "minmax_col", "min"]):
            raise ValueError("'edge_fillval' must be one of " +
                             "['minmax'|'col'|'minmax_col'|'min'].")

        if model_opts["edge_fillval"] == "min" and model_opts["force_y_asc"]:
            raise ValueError("`edge_fillval` is 'min' and 'force_y_asc' is " +
                             "`True`, which doesn't make sense together.")

        if len(model_opts["bins"]) != 2:
            raise ValueError("Bins for energy hist must be of format " +
                             "`[sin_dec_bins, logE_bins]`.")

        if np.any(sin_dec_bins < -1.) or np.any(sin_dec_bins > 1.):
            raise ValueError("sinDec declination bins for energy hist not " +
                             "in valid range `[-1, 1]`.")

        if self._model_opts["mc_bg_w"] is not None:
            self._model_opts["mc_bg_w"] = np.atleast_1d(
                self._model_opts["mc_bg_w"])

        self._model_opts = model_opts
        self._model_opts = model_opts

        # Hashs and data for cached values in the pur BG case
        self._soverb_energy_cache = None
        self._soverb_energy_hash = None
        self._spatial_bg_cache = None
        self._spatial_bg_hash = None

        self._needed_data = np.array(
            ["dec", "ra", "sigma", "logE"])
        self._provided_args = ["src_w_dec", "src_w_theo"]
        self._SECINDAY = 24. * 60. * 60

        self._log = logger(name=self.__class__.__name__, level="ALL")

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
        return self._model_opts.copy()

    def fit(self, X, livetime_days, MC, srcs):
        """
        Setup the LLH model and make it ready to use.
        """
        if self._model_opts["mc_bg_w"] is not None:
            if len(self._model_opts["mc_bg_w"]) != len(MC):
                raise ValueError("Length of MC BG weights and MC must match.")

        # Setup internals for model evaluation
        self._srcs = srcs
        _out = self._setup_model(X, livetime_days, MC, srcs)
        self._llh_args, self._spatial_bg_spl, self._energy_interpol, _ = _out

        # Cache repeatedly used values
        self._src_dec_col_vec = self._srcs["dec"][:, None]

        # Debug
        self._spl_info = _out[3]

        return

    def get_args(self):
        return self._llh_args

    def get_soverb(self, X, band_select=True):
        """
        Calculate sob values per source per event for given data X
        """
        # Improved caching: Give X and Xsig separately!
        # Preselect data to save computation time
        if band_select:
            X = X[np.any(self._select_X(X), axis=0)]

        # Make combined PDF ratio term. Needs caching for the energy ratio
        sob = (self._soverb_spatial(X["ra"], np.sin(X["dec"]), X["sigma"]) *
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
                     self._model_opts["select_ev_sigma"]) &
                    (X["dec"] < self._src_dec_col_vec + X["sigma"] *
                     self._model_opts["select_ev_sigma"]))
        return dec_mask

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
            self._srcs["ra"], self._srcs["dec"], ev_ra, ev_sin_dec, ev_sig,
            self._model_opts["kent"])

        # Divide by background PDF per event. Needs caching for the bg PDF
        ev_sin_dec.flags.writeable = False
        new_hash = hash(ev_sin_dec.data)
        if (self._spatial_bg_hash is None or self._spatial_bg_hash != new_hash):
            self._spatial_bg_cache = self._spatial_bg_spl(ev_sin_dec)
            self._spatial_bg_hash = new_hash
        sob = sob / self._spatial_bg_cache

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
            Event positions in equatorial sinus declination, [-1, 1].
        ev_logE
            Event log10 energy estimator ``log10(energy proxy)``.

        Returns
        -------
        soverb_energy_ratio : array-like, shape (nevts)
            Ratio of the energy signal and background PDF for each given event.
        """
        ev_sin_dec.flags.writeable = False
        ev_logE.flags.writeable = False
        new_hash = hash(ev_sin_dec.data) + hash(ev_logE.data)

        if (self._soverb_energy_hash is None or
                self._soverb_energy_hash != new_hash):
            self._soverb_energy_cache = np.exp(self._energy_interpol(
                np.vstack((ev_sin_dec, ev_logE)).T))
            self._soverb_energy_hash = new_hash

        return self._soverb_energy_cache

    def _setup_model(self, X, livetime_days, MC, srcs):
        """
        Create the splines and interpolators used to evaluate the LLH model.

        The energy PDF is built using 2D histogram ratios for bg and signal and
        interpolate the histogram.
        The background spatial PDFs are built per source by fitting a spline
        to a sindec modelas done in the ``grb.TimeDecDependentBGDataInjector``.

        Returns
        -------
        llh_args : dict
            Fixed LLH args this model provides via ``get_args()``.
        spatial_bg_spls : list of splines
            Splines per source describing the spatial background PDF in
            ``sin(dec)``, normalized over the full sphere
            ``int_(-1,1)_(0, 2pi) spl dra dsindec = 1``.
        energy_interpol : scipy.interpolate.RegularGridInterpolator
            Interpolator returning the energy signal over background ratio for
            ``sin(dec), logE`` pairs.
        spl_info : dict
            Collection of spline information.
        """
        def spl_normed_factory(spl, lo, hi, norm):
            """ Renormalize spline, so ``int_lo^hi renorm_spl dx = norm`` """
            return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)

        sin_dec_bins = self._model_opts["bins"][0]
        lo, hi = sin_dec_bins[0], sin_dec_bins[-1]

        # Step 1: Make a sindec spline from data for the background PDF.
        # Data is binned finely and the spline is using the errors as to smooth
        # appropriatly
        ev_sin_dec = np.sin(X["dec"])
        _bins = make_equdist_bins(
            ev_sin_dec, lo, hi, weights=None,
            min_evts_per_bin=self._inj_opts["n_data_evts_min"])
        hist = np.histogram(ev_sin_dec, bins=_bins, density=False)[0]
        dA = np.diff(_bins)  # Bin diffs is enough, spline gets renormed later
        stddev = np.sqrt(hist) / dA
        hist = hist / dA
        weight = 1. / stddev
        data_spline = fit_spl_to_hist(hist, bins=_bins, w=weight,
                                      s=len(hist))[0]
        # Renormalize sindec data spline to include the 2pi from RA
        # normalization so it can be used as the BG PDF
        norm = 1. / 2. / np.pi
        data_spline = spl_normed_factory(data_spline, lo, hi, norm=norm)
        print(self._log.INFO("Made {} bins for allsky hist".format(len(_bins))))

        # Step 2: Get source weights from the signal weighted MC sindec spline
        # fitted to a histogram. True dec, to match selection in signal injector
        mc_sin_dec = np.sin(MC["trueDec"])
        w_sig = (MC["ow"] * self._model_opts["flux_model"](MC["trueE"]) *
                 livetime_days * self._SECINDAY)
        _bins = make_equdist_bins(
            mc_sin_dec, lo, hi,
            weights=w_sig, min_evts_per_bin=self._model_opts["n_mc_evts_min"])
        print(self._log.INFO("Made {} bins for allsky hist".format(len(_bins))))
        hist = np.histogram(mc_sin_dec, bins=_bins, weights=w_sig,
                            density=False)[0]
        variance = np.histogram(mc_sin_dec, bins=_bins, weights=w_sig**2,
                                density=False)[0]
        dA = np.diff(_bins)  # Bin diffs is enough, weights get renormed later
        hist = hist / dA
        stddev = np.sqrt(variance) / dA
        weight = 1. / stddev
        mc_spline = fit_spl_to_hist(hist, bins=_bins, w=weight, s=len(hist))[0]
        src_w_dec = mc_spline(np.sin(srcs["dec"]))

        # Step 3: Build energy PDF interpolator
        # Make histograms, signal weighted to flux model. BG is either data
        # or MC weighted by external model_opts["mc_bg_w"] (eg. atmo flux)
        w_bg = self._model_opts["mc_bg_w"]
        if w_bg is None:
            sin_dec_bg = np.sin(X["dec"])
            logE_bg = X["logE"]
        else:
            sin_dec_bg = np.sin(MC["dec"])
            logE_bg = MC["logE"]
        sin_dec_sig = np.sin(MC["dec"])
        logE_sig = MC["logE"]

        _bx, _by = self._model_opts["bins"]
        h_bg, _, _ = np.histogram2d(sin_dec_bg, logE_bg, weights=w_bg,
                                    bins=[_bx, _by], normed=True)
        h_sig, _, _ = np.histogram2d(sin_dec_sig, logE_sig, weights=w_sig,
                                     bins=[_bx, _by], normed=True)

        # Check if events are inside bin ranges
        err = ""
        if np.any((sin_dec_bg < _bx[0]) | (sin_dec_bg > _bx[-1])):
            err += "declinations outside sindec bins for BG energy PDF.\n"
        if np.any((sin_dec_sig < _bx[0]) | (sin_dec_sig > _bx[-1])):
            err += "declinations outside sindec bins for signal energy PDF.\n"
        if np.any((logE_bg < _by[0]) | (logE_bg > _by[-1])):
            err += "logEs outside logE bins for BG energy PDF.\n"
        if np.any((logE_sig < _by[0]) | (logE_sig > _by[-1])):
            err += "logEs outside logE bins for signal energy PDF.\n"
        if err != "":
            err += "If this is intended, please remove them beforehand."
            raise ValueError(err)

        # Create interpolator
        energy_interpol = make_grid_interp_from_hist_ratio(
            h_bg=h_bg, h_sig=h_sig, bins=[_bx, _by],
            edge_fillval=self._model_opts["edge_fillval"],
            interp_col_log=self._model_opts["interp_col_log"],
            force_y_asc=self._model_opts["force_logE_asc"])

        llh_args = {"src_w_dec": src_w_dec, "src_w_theo": srcs["w_theo"]}

        spl_info = {
            "sin_dec_pdf_spline": data_spline,
            "mc_sin_dec_pdf_spline": mc_spline,
        }

        return llh_args, data_spline, energy_interpol, spl_info
