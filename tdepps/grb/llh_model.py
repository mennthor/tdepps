# coding: utf-8

from __future__ import absolute_import

import numpy as np

from ..backend import soverb_time_box, pdf_spatial_signal
from ..base import BaseModel
from ..utils import fill_dict_defaults, logger
from ..utils import spl_normed, fit_spl_to_hist
from ..utils import make_time_dep_dec_splines, make_grid_interp_from_hist_ratio


class GRBModel(BaseModel):
    """
    Models the PDF part for the GRB LLH function.
    """
    def __init__(self, X, MC, srcs, run_dict, spatial_opts=None,
                 energy_opts=None):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        # Check and setup spatial PDF options
        req_keys = ["sindec_bins", "rate_rebins"]
        opt_keys = {"select_ev_sigma": 5., "spl_s": None, "n_scan_bins": 50,
                    "kent": True}
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
        req_keys = ["bins", "flux_model"]
        opt_keys = {"mc_bg_w": None, "force_logE_asc": True,
                    "edge_fillval": "minmax_col", "interp_col_log": False}
        energy_opts = fill_dict_defaults(energy_opts, req_keys, opt_keys)

        sin_dec_bins, logE_bins = map(np.atleast_1d, energy_opts["bins"])
        energy_opts["bins"] = [sin_dec_bins, logE_bins]

        try:
            energy_opts["flux_model"](1.)  # Standard units are E = 1GeV
        except Exception:
            raise TypeError("'flux_model' must be a function `f(trueE)`.")

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

        if energy_opts["mc_bg_w"] is not None:
            energy_opts["mc_bg_w"] = np.atleast_1d(energy_opts["mc_bg_w"])
            if len(energy_opts["mc_bg_w"]) != len(MC):
                raise ValueError("Length of MC BG weights and MC must match.")

        self._spatial_opts = spatial_opts
        self._energy_opts = energy_opts
        self._srcs = srcs

        self._needed_data = np.array(
            ["timeMJD", "dec", "ra", "sigma", "logE"])
        self._provided_args = ["src_w_dec", "src_w_theo" "nb"]

        # Setup internals for model evaluation
        self._log = logger(name=self.__class__.__name__, level="ALL")
        _out = self._setup_model(X, MC, srcs, run_dict)
        self._llh_args, self._spatial_bg_spls, self._energy_interpol, _ = _out

        # Cache repeatedly used values
        self._src_dt = np.vstack((srcs["dt0"], srcs["dt1"])).T
        self._src_dec_col_vec = self._srcs["dec"][:, None]

        # Debug
        self._spl_info = _out[-1]

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
    def spatial_opts(self):
        return self._spatial_opts.copy()

    @property
    def energy_opts(self):
        return self._energy_opts.copy()

    def get_args(self):
        return self._llh_args

    def get_soverb(self, X):
        """
        Calculate sob values per source per event for given data X
        """
        # Preselect data to save computation time
        X = X[np.any(self._select_X(X), axis=0)]

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
        return dec_mask

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
            self._srcs["ra"], self._srcs["dec"], ev_ra, ev_sin_dec, ev_sig,
            self._spatial_opts["kent"])
        # Divide by background PDF per source
        for j, sobi in enumerate(sob):
            sob[j] = sobi / self._spatial_bg_spls[j](ev_sin_dec)

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
        llh_args : dict
            Fixed LLH args this model provides via ``get_args()``.
        spatial_bg_spls : list of splines
            Splines per source describing the spatial background PDF in
            ``sin(dec)``, normalized over the full sphere
            ``int_(-1,1)_(0, 2pi) spl dra dsindec = 1``.
        energy_interpol : scipy.interpolate.RegularGridInterpolator
            Interpolator returning the energy signal over background ratio for
            ``sin(dec), logE`` pairs.
        """
        ev_t = X["timeMJD"]
        ev_sin_dec = np.sin(X["dec"])
        src_t = np.atleast_1d(srcs["t"])
        src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T

        sin_dec_bins = self._spatial_opts["sindec_bins"]
        rate_rebins = self._spatial_opts["rate_rebins"]

        # Step 1: Build per source spatial BG estimation and sindec PDFs
        print(self._log.INFO("Create time dep sindec splines."))
        sin_dec_splines, spl_info = make_time_dep_dec_splines(
            ev_t, ev_sin_dec, srcs, run_dict, sin_dec_bins, rate_rebins,
            spl_s=self._spatial_opts["spl_s"],
            n_scan_bins=self._spatial_opts["n_scan_bins"])

        # Step 2: Cache fixed LLH args
        # Cache expected nb for each source from allsky rate func integral
        nb = spl_info["allsky_rate_func"].integral(
            src_t, src_trange, spl_info["allsky_best_params"])
        assert len(nb) == len(src_t)

        # Get source weights from the signal weighted MC sindec spline fitted
        # to a histogram. True dec, to match selection in signal injector
        w_sig = MC["ow"] * self._energy_opts["flux_model"](MC["trueE"])
        hist = np.histogram(np.sin(MC["trueDec"]), bins=sin_dec_bins,
                            weights=w_sig, density=False)[0]
        variance = np.histogram(np.sin(MC["dec"]), bins=sin_dec_bins,
                                weights=w_sig**2, density=False)[0]
        dA = np.diff(sin_dec_bins)
        hist = hist / dA
        stddev = np.sqrt(variance) / dA
        weight = 1. / stddev
        mc_spline = fit_spl_to_hist(hist, bins=sin_dec_bins, w=weight,
                                    s=self._spatial_opts["spl_s"])[0]
        src_w_dec = mc_spline(np.sin(srcs["dec"]))

        # Renormalize sindec splines to include the 2pi from RA normalization
        # These are used as the per source BG PDFs each normed on its own
        def spl_normed_factory(spl, lo, hi, norm):
            """ Renormalize spline, so ``int_lo^hi renorm_spl dx = norm`` """
            return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)

        lo, hi = sin_dec_bins[0], sin_dec_bins[-1]
        norm = 1. / 2. / np.pi
        spatial_bg_spls = []
        for spl in sin_dec_splines:
            spatial_bg_spls.append(spl_normed_factory(spl, lo, hi, norm=norm))

        # Step 3: Build energy PDF interpolator
        # Make histograms, signal weighted to flux model. BG is either data
        # or MC weighted by external energy_opts["mc_bg_w"] (eg. atmo flux)
        w_bg = self._energy_opts["mc_bg_w"]
        if w_bg is None:
            sin_dec_bg = np.sin(X["dec"])
            logE_bg = X["logE"]
        else:
            sin_dec_bg = np.sin(MC["dec"])
            logE_bg = MC["logE"]
        sin_dec_sig = np.sin(MC["dec"])
        logE_sig = MC["logE"]
        w_sig = MC["ow"] * self._energy_opts["flux_model"](MC["trueE"])

        _bx, _by = self._energy_opts["bins"]
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
            edge_fillval=self._energy_opts["edge_fillval"],
            interp_col_log=self._energy_opts["interp_col_log"],
            force_y_asc=self._energy_opts["force_logE_asc"])

        llh_args = {"src_w_dec": src_w_dec,
                    "src_w_theo": srcs["w_theo"], "nb": nb}

        spl_info["sin_dec_splines"] = sin_dec_splines
        spl_info["mc_sin_dec_pdf_spline"] = mc_spline

        return llh_args, spatial_bg_spls, energy_interpol, spl_info
