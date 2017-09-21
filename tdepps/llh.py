# coding: utf-8

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import math
import numpy as np
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy.ndimage.filters import gaussian_filter

from .utils import fill_dict_defaults, power_law_flux_per_type
import tdepps.backend as backend


class GRBLLH(object):
    r"""
    Implementation of the GRBLLH for time dependent analysis.

    For more information on the extended Likelihood method, see [1]_.
    The used Likelihood is defined as:

    .. math::

      \mathcal{L}(n_S|\{X\}_N) = \frac{(n_S + \langle n_B \rangle)^{-N}}{N!}
                                   \cdot \exp{(-(n_S + \langle n_B \rangle))}
                                   \cdot\prod_{i=1}^N P_i


    Where :math:`n_S` is the number of signal-like events, :math:`\{X\}_N` is
    the set of the :math:`N` given datapoints, and :math:`P_i` are the event
    probabilities of being background- or signal-like:

    .. math::

      P_i = \frac{n_S \cdot S_i + \langle n_B \rangle \cdot B_i}
                 {n_S + \langle n_B \rangle}

    The expected background :math:`\langle n_B \rangle` is derived from the
    detector properties at a given time and is not a free parameter here.

    The ln-LLH is then derived by taking the natural logarithm of the LLH:

    .. math::

      \ln\mathcal{L}(n_S|\{X\}_N) = -(n_S + \langle n_B\rangle) -\ln(N!) +
                            \sum_{i=1}^N \ln(n_S S_i + \langle n_B\rangle B_i)

    Parameters
    ----------
    X : record-array
        Global data set, used to derive per events PDFs. `X` must contain names:

        - "sinDec", float: Per event sinus declination, in `[-1, 1]`.
        - "logE", float: Per event energy proxy, given in log10(1/GeV).

    MC : record-array
        Global Monte Carlo data set, used to derive per event PDFs. `MC` must
        contain the same names as `X` and additionaly the MC truths:

        - "trueE", float: True event energy in GeV.
        - "ow", float: Per event 'neutrino generator' OneWeight [2]_,
          so it is already divided by `nevts * nfiles * type_weight`.
          Units are 'GeV sr cm^2'. Final event weights are obtained by
          multiplying with desired flux per particle type.

    spatial_pdf_args : dict
        Arguments for the spatial signal and background PDF. Must contain keys:

        - "bins", array-like: Explicit bin edges of the sinus declination
          histogram used to fit a spline describing the spatial background PDF.
          Bins must be in range ``[-1, 1]``.
          Bins are used for every 1D hist, including BG PDF, detector weight
          histogram and expected events per ``sin_dec`` histogram.
        - "kent", bool, optional: If ``True``, the signal PDF uses the Kent [3]_
          distribution. A 2D gaussian PDF is used otherwise. (default: ``True``)
        - "k", int, optional: Degree of the smoothing spline used to fit the
          background histogram. Must be ``1 <= k <= 5``. (default: 3)

    energy_pdf_args : dict
        Arguments for the energy PDF ratio. Must contain keys:

        - "bins", array-like: Explicit bin edges of the sinus declination vs
          logE histogram used to interpolate the energy PDF ratio. Must be
          [sin_dec_bins, logE_bins] in ranges [-1, 1] for sinus declination and
          [-inf, +inf] for logE (logspace bins).
        - "gamma", float, optional: Spectral index of the power law
          :math:`E^{-\gamma}` used to weight MC to an astrophisical flux.
          (default: 2.)
        - "mc_bg_weights", array-like or None: If not ``None`` also use MC for
          the BG histogram weighted to the given weights. If ``None``, use data.
        - "fillval", str, optional: What values to use, when the histogram has
          MC but no data in a bin. Then the gaps are filled, by assigning values
          to the histogram edges in each sinDec slice for low/high energies
          seprately and then interpolating inside. Can be one of
          ['minmax'|'minmax_col'|'min'|'col']:

          + 'minmax': Use the lowest/highest global ratio values at the edges.
          + 'col': Next valid value in each colum from the top/bottom is used.
          + 'minmax_col': Like 'minmax' but use min/max value per bin.
          + 'min': Only the lowest global ratio value is used at all edges.

          Listed in order optimistic -> conservative. (default: 'minmax_col')
        - "interpol_log", bool, optional: If ``True``, gaps in the signal over
          background ratio histogram are interpolated linearly in log. Otherwise
          the interpolation is in linear space. (default: ``False``)
        - "smooth_sigma": ``[[sin_dec_bg, logE_bg], [sin_dec_sig, logE_sig]]``.
          Standard deviations for a 2D gaussian smoothing kernel applied to the
          BG and signal histograms in *normal space* before taking the ratio.
          Units are array indices of the corresponding bins arrays.
        - "logE_asc", bool, optional: If ``True`` assume that in each
          ``sin_dec`` bin the energy distribution must be monotonically
          increasing and correct if it is not. This may be justified by the
          shape of the flux PDFs, you should always check the resulting PDF.
          Note: If ``True``, using 'min' in 'fillval' makes no sense so an error
          is thrown. (default: ``False``)

    time_pdf_args : dict, optional
        Arguments for the time PDF ratio. Must contain keys:

        - "nsig", float, optional: The truncation of the gaussian edges of the
          signal PDF to have finite support. Given in units of sigma, must >= 3.
          (default: 4.)
        - "sigma_t_min", float, optional: Minimum sigma of the gaussian edges.
          The gaussian edges of the time signal PDF have at least this sigma.
        - "sigma_t_max", float, optional: Maximum sigma of the gaussian edges.
          The gaussian edges of the time signal PDF have maximal this sigma.

        (default: None)

    llh_args : dict, optional
        Arguments controlling the LLH calculation. Must contain keys:

        - "sob_rel_eps", float, optional: Realative signal over background
          threshold. Events which have `SoB_i / max(SoB) < sob_rel_eps` are not
          further used in the LLH evluation. (default: 0)
        - "sob_abs_abs", float: Absolute signal over background threshold.
          Events which have `SoB_i < sob_abs_eps` are not further used in the
          LLH evluation. (default: 1e-3)

        (default: None)

    Notes
    -----
    .. [1] Barlow, "Statistics - A Guide to the Use of Statistical Methods in
           the Physical Sciences". Chap. 5.4, p. 90. Wiley (1989)
    .. [2] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
    .. [3] https://en.wikipedia.org/wiki/Kent_distribution
    """
    def __init__(self, X, MC, spatial_pdf_args, energy_pdf_args,
                 time_pdf_args=None, llh_args=None):
        # Check if data, MC have all needed names
        #  X Doesn't need logE if MC BG weights are given, but leave it for now.
        X_names = ["sinDec", "logE"]
        for n in X_names:
            if n not in X.dtype.names:
                raise ValueError("`X` is missing name '{}'.".format(n))
        MC_names = X_names + ["trueE", "ow"]
        for n in MC_names:
            if n not in MC.dtype.names:
                raise ValueError("`MC` is missing name '{}'.".format(n))

        # Setup spatial PDF args
        required_keys = ["bins"]
        opt_keys = {"k": 3, "kent": True}
        self._spatial_pdf_args = fill_dict_defaults(spatial_pdf_args,
                                                    required_keys, opt_keys)
        k = self._spatial_pdf_args["k"]
        if (k < 1) or (k > 5):
            raise ValueError("'k' must be integer in [1, 5].")
        # Check if binning is OK
        bins = np.atleast_1d(self._spatial_pdf_args["bins"])
        if np.any(bins < -1) or np.any(bins > 1):
            raise ValueError("Bins for BG spline not in valid range [-1, 1].")
        if np.any(X["sinDec"] < bins[0]) or np.any(X["sinDec"] > bins[-1]):
            raise ValueError("sinDec data events outside given bins. If this" +
                             "is intended, please remove them beforehand.")
        self._spatial_pdf_args["bins"] = bins

        # Setup energy PDF args
        required_keys = ["bins"]
        opt_keys = {"gamma": 2., "mc_bg_weights": None, "fillval": "minmax_col",
                    "interpol_log": False, "smooth_sigma": [[0., 0.], [0., 0.]],
                    "logE_asc": True}
        self._energy_pdf_args = fill_dict_defaults(energy_pdf_args,
                                                   required_keys, opt_keys)
        if (self._energy_pdf_args["fillval"] not in
                ["minmax", "col", "minmax_col", "min"]):
            raise ValueError("'fillval' must be one of " +
                             "['minmax'|'col'|'minmax_col'|'min'].")
        if ((self._energy_pdf_args["fillval"] == "min") and
                (self._energy_pdf_args["logE_asc"])):
            raise ValueError("'fillval'='min' makes no sense when 'logE_asc' " +
                             "is True.")
        sig = np.asarray(self._energy_pdf_args["smooth_sigma"])
        if len(sig) != 2 or sig.shape != (2, 2):
            raise ValueError("`smooth_sigma` must be [[sin_dec_bg, logE_bg]" +
                             ", [sin_dec_sig, logE_sig]].")
        self._energy_pdf_args["smooth_sigma"] = sig
        # Check if binning is OK
        if len(self._energy_pdf_args["bins"]) != 2:
            raise ValueError("Bins for energy hist must be of format " +
                             "[sin_dec_bins, logE_bins].")
        sin_dec_bins = np.atleast_1d(self._energy_pdf_args["bins"][0])
        logE_bins = np.atleast_1d(self._energy_pdf_args["bins"][1])
        if np.any(sin_dec_bins < -1.) or np.any(sin_dec_bins > 1.):
            raise ValueError("sinDec declination bins for energy hist not " +
                             "in valid range [-1, 1].")
        self._energy_pdf_args["bins"] = [sin_dec_bins, logE_bins]

        mc_bg_w = self._energy_pdf_args["mc_bg_weights"]
        if mc_bg_w is None:  # Check both data and MC
            for sd, name in zip([X["sinDec"], MC["sinDec"]], ["data", "MC"]):
                if np.any((sd < sin_dec_bins[0]) | (sd > sin_dec_bins[-1])):
                    raise ValueError("sinDec " + name + " events outside " +
                                     "given bins for energy hist. If this is " +
                                     "intended, please remove them beforehand.")
            for logE, name in zip([X["logE"], MC["logE"]], ["data", "MC"]):
                if np.any((logE < logE_bins[0]) | (logE > logE_bins[-1])):
                    raise ValueError("logE " + name + " events outside " +
                                     "given bins for energy hist. If this is " +
                                     "intended, please remove them beforehand.")
            sin_dec_bg, logE_bg, w_bg = X["sinDec"], X["logE"], np.ones(len(X))
        else:  # Only need to check MC, building 2D hist on MC only
            if len(mc_bg_w) != len(MC):
                raise ValueError("Length of MC BG weights and MC must match.")
            if np.any((MC["sinDec"] < sin_dec_bins[0]) |
                      (MC["sinDec"] > sin_dec_bins[-1])):
                raise ValueError("sinDec MC events outside given bins for " +
                                 "energy hist. If this is intended, please " +
                                 "remove them beforehand.")
            if np.any((MC["logE"] < logE_bins[0]) |
                      (MC["logE"] > logE_bins[-1])):
                raise ValueError("logE MC events outside given bins for " +
                                 "energy hist. If this is intended, please " +
                                 "remove them beforehand.")
            sin_dec_bg, logE_bg, w_bg = MC["sinDec"], MC["logE"], mc_bg_w

        sin_dec_sig, logE_sig = MC["sinDec"], MC["logE"]
        w_sig = MC["ow"] * power_law_flux_per_type(
            MC["trueE"], self._energy_pdf_args["gamma"])

        self._energy_pdf_args["bins"][0] = sin_dec_bins
        self._energy_pdf_args["bins"][1] = logE_bins

        # Setup time PDF args
        required_keys = []
        opt_keys = {"nsig": 4., "sigma_t_min": 2., "sigma_t_max": 30.}
        self._time_pdf_args = fill_dict_defaults(time_pdf_args, required_keys,
                                                 opt_keys)
        nsig = self._time_pdf_args["nsig"]
        if nsig < 3:
            raise ValueError("'nsig' must be >= 3.")

        tmin = self._time_pdf_args["sigma_t_min"]
        tmax = self._time_pdf_args["sigma_t_max"]
        if tmin > tmax or tmin == tmax:
            raise ValueError("'sigma_t_min' must be < 'sigma_t_max'.")
        if tmin <= 0:
            raise ValueError("'sigma_t_min' must be > 0.")

        # Setup LLH args
        required_keys = []
        opt_keys = {"sob_rel_eps": 0, "sob_abs_eps": 1e-3}
        self._llh_args = fill_dict_defaults(llh_args, required_keys, opt_keys)

        rel_eps = self._llh_args["sob_rel_eps"]
        if (rel_eps < 0) or (rel_eps > 1):
            raise ValueError("'sob_rel_eps' must be in [0, 1]")
        if self._llh_args["sob_abs_eps"] < 0:
            raise ValueError("'sob_abs_eps' must be >= 0.")

        # Setup common variables
        self._SECINDAY = 24. * 60. * 60.

        # Create background spline used in the spatial PDF from data
        self._spatial_bg_spl = self._normed_sin_dec_spline(
            sin_dec=X["sinDec"], bins=self._spatial_pdf_args["bins"],
            weights=np.ones(len(X)))

        # Create energy PDF from global data and MC
        self._energy_interpol = self._create_sin_dec_logE_interpolator(
            sin_dec_bg, logE_bg, w_bg, sin_dec_sig, logE_sig, w_sig)

        # Create sin_dec signal spline for the src detector weights from MC
        self._spatial_signal_spl = self._normed_sin_dec_spline(
            sin_dec=MC["sinDec"], bins=self._energy_pdf_args["bins"][0],
            weights=w_sig)

        # Create event expectation signal spline per sin_dec for absolute event
        # expectations used for multiyear weighting
        self._nexpected_signal_spl = self._nexpected_per_sin_dec_spline(
            sin_dec=MC["sinDec"], trueE=MC["trueE"], ow=MC["ow"],
            bins=self._spatial_pdf_args["bins"])

        return

    @property
    def spatial_pdf_args(self):
        return self._spatial_pdf_args

    @spatial_pdf_args.setter
    def spatial_pdf_args(self, arg):
        raise ValueError("`spatial_pdf_args` can't be set. Create new " +
                         "object instead.")

    @property
    def energy_pdf_args(self):
        return self._energy_pdf_args

    @energy_pdf_args.setter
    def energy_pdf_args(self, arg):
        raise ValueError("`energy_pdf_args` can't be set. Create new " +
                         "object instead.")

    @property
    def time_pdf_args(self):
        return self._time_pdf_args

    @time_pdf_args.setter
    def time_pdf_args(self, arg):
        raise ValueError("`time_pdf_args` can't be set. Create new " +
                         "object instead.")

    @property
    def llh_args(self):
        return self._llh_args

    @llh_args.setter
    def llh_args(self, arg):
        raise ValueError("`llh_args` can't be set. Create new object instead.")

    def lnllh_ratio(self, X, ns, args):
        r"""
        Return two times the the natural logarithm of the ratio of Likelihoods
        under the null hypothesis -- here: :math:`n_S = 0` -- and the
        alternative hypothesis that we have non-zero signal contribution.

        The ratio :math:`\Lambda` used here is defined as:

        .. math::

          \Lambda = -2\ln\left(\frac{\mathcal{L}_0}{\mathcal{L}_1}\right)
                  =  2\ln\left(\mathcal{L}_1 - \mathcal{L}_0\right)

        High values of :math:`\Lambda` indicate, that the null hypothesis is
        more unlikely contrary to the alternative.

        For GRBLLH this reduces to:

        .. math::

          \Lambda(n_S) = -n_S + \sum_{i=1}^N\ln
                         \left(\frac{n_S S_i}{\langle n_B\rangle B_i} + 1\right)

        .. note:: Which events contribute to the LLH is controlled using the
          options given in `GRBLLH.llh_args`.

        Parameters
        ----------
        X : record-array, shape (nevts)
            Fixed data set the LLH depends on. dtypes are ["name", type].
            Here `X` must have names:

            - 'timeMJD': Per event times in MJD days.
            - 'ra', 'sinDec': Per event right-ascension positions in equatorial
              coordinates, given in radians in :math:`[0, 2\pi]`and sinus
              declination coordinates in :math:`[-1, 1]`.
            - 'logE': Per event energy proxy, given in
              :math:`\log_{10}(1/\text{GeV})`.
            - 'sigma': Per event positional uncertainty, given in radians. It is
              assumed, that a circle with radius `sigma` contains approximately
              :math:`1\sigma` (~0.39) of probability of the reconstrucion
              likelihood space.

        ns : float
            Fitparameter: number of signal events that we expect at the source
            locations.

        args : dict
            Other fixed parameters {'par_name': value}, the LLH depends on.
            Here `args` must have keys:

            - 'ns', array-like, shape (nsrcs): Number of expected background
              events for each sources time window.
            - 'srcs', record-array, shape (nsrcs): Fixed source parameters,
              must have names:

              + 'ra', float: Right-ascension coordinate of each source in
                radian in intervall :math:`[0, 2\pi]`.
              + 'dec', float: Declinatiom coordinate of each source in radian
                in intervall :math:`[-\pi / 2, \pi / 2]`.
              + 't', float: Time of the occurence of the source event in MJD
                days.
              + 'dt0', 'dt1': float: Lower/upper border of the time search
                window in seconds, centered around each source time `t`.
              + 'w_theo', float: Theoretical source weight per source, eg. from
                a known gamma flux.

        Returns
        -------
        TS : float
            Lambda test statistic, 2 times the natural logarithm of the LLH
            ratio.
        ns_grad : array-like, shape (1)
            Gradient of the test statistic in the fit parameter `ns`.
        """
        sob = self._soverb(X, args)
        return self._lnllh_ratio(ns, sob)

    def fit_lnllh_ratio(self, X, ns0, args, bounds, minimizer_opts):
        """
        Fit the LLH parameter :math:`n_S` for a given set of data and fixed
        LLH arguments.

        The fit is calculated for the test statistic, which is two times the
        natural logarithm of the ratio of Likelihoods under the null hypothesis
        -- here: :math:`n_S = 0` -- and the alternative hypothesis that we have
        non-zero signal contribution.

        Parameters
        ----------
        X : record-array
            See :py:meth:`lnllh_ratio`, Parameters
        ns0 : float
            Fitter seed for the fit parameter ns: number of signal events that
            we expect at the source locations.
        args : dict
            See :py:meth:`lnllh_ratio`, Parameters
        bounds : array-like, shape (1, 2)
            Minimization bounds `[[min, max]]` for `ns`. Use None for one of
            `min` or `max` when there is no bound in that direction.
        minimizer_opts : dict
            Options passed to `scipy.optimize.minimize` [4]_ using the
            "L-BFGS-B" algorithm.

        Returns
        -------
        ns : float
            Best fit parameter number of signal events :math:`n_S`.
        TS : float
            Best fit test statistic value.

        Notes
        -----
        .. [4] https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html_minimize.py#L36-L466
        """
        def _neglnllh(ns):
            """
            Wrapper for the LLH function returning the negative ln-LLH ratio
            suitable for the minimizer.

            Parameters
            ----------
            ns : float
                See :py:meth:`lnllh_ratio`, Parameters

            Returns
            -------
            lnllh : float
                See :py:meth:`lnllh_ratio`, Returns
            lnllh_grad : array-like
                See :py:meth:`lnllh_ratio`, Returns
            """
            lnllh, lnllh_grad = self._lnllh_ratio(ns, sob)
            return -1. * lnllh, -1. * lnllh_grad

        # If no events are given, best fit is always 0, skip all further steps
        if len(X) == 0:
            return 0., 0.

        # Get the best fit parameter and TS. Analytic cases are handled:
        # For nevts = [1 | 2] we get a [linear | quadratic] equation to solve.
        sob = self._soverb(X, args)
        nevts = len(sob)
        # Test again, because we applied some threshold cuts
        if nevts == 0:
            return 0., 0.
        if nevts == 1:
            # Use scalar math functions, they're faster than numpy
            sob = sob[0]
            ns = 1. - (1. / sob)
            if ns <= 0:
                return 0., 0.
            else:
                TS = 2. * (-ns + math.log(sob))
            return ns, TS
        elif nevts == 2:
            a = 1. / (sob[0] * sob[1])
            c = (sob[0] + sob[1]) * a
            ns = 1. - 0.5 * c + math.sqrt(c * c / 4. - a + 1.)
            if ns <= 0:
                return 0., 0.
            else:
                TS, _ = self._lnllh_ratio(ns, sob)
            return ns, TS
        else:
            # Fit other cases
            res = sco.minimize(fun=_neglnllh, x0=[ns0], jac=True, bounds=bounds,
                               method="L-BFGS-B", options=minimizer_opts)

        # Return function value with correct sign
        return res.x[0], -1. * res.fun[0]

    def _lnllh_ratio(self, ns, sob):
        """
        Internal wrapper to calculate the ln-LLH ratio on the already cutted
        sob values, to avoid recalculation.

        Parameters
        ----------
        ns : float
            See :py:meth:`lnllh_ratio`, Parameters
        sob : array-like, shape (nevts)
            Total signal over background ratio for each event, already reduced
            over all sources.

        Returns
        -------
        lnllh : float
            See :py:meth:`lnllh_ratio`, Returns
        lnllh_grad : array-like
            See :py:meth:`lnllh_ratio`, Returns
        """
        # Teststatistic 2 * ln(LLH-ratio)
        x = ns * sob
        TS = 2. * (-ns + np.sum(np.log1p(x)))
        # Gradient in ns (chain rule: ln(ns * a + 1)' = 1 / (ns * a + 1) * a)
        ns_grad = 2. * (-1. + np.sum(sob / (x + 1.)))
        return TS, np.array([ns_grad])

    # #########################################################################
    # Public accessible helper methods
    def src_weights(self, src_dec, src_w_theo):
        """
        Make combined, normalized source weights from the detector exposure and
        a theoretical source weight.

        Parameters
        ----------
        src_dec : array-like, shape (nsrcs)
            Declination coordinate of each source in radian in interval
            ``[-pi/2, pi/2]``.
        src_w_theo : array-like, shape (nsrcs)
            Theoretical source weight per source, eg. from a known gamma flux.

        Returns
        -------
        src_w : array-like, shape (nsrcs, 1)
            Combined normalized weight per source.
        """
        sin_dec = np.sin(src_dec)
        min_sin_dec, max_sin_dec = self._spatial_pdf_args["bins"][[0, -1]]
        if np.any(sin_dec < min_sin_dec) or np.any(sin_dec > max_sin_dec):
            raise ValueError("Requested weight for a sin_dec outside the " +
                             "declination binning.")
        # Get src detector weights form signal sin_dec spline from MC
        src_dec_w = np.exp(self._spatial_signal_spl(np.sin(src_dec)))
        # Make combined src weight by multiplying with the theoretical weights
        src_w = src_dec_w * src_w_theo
        src_w = src_w[:, None] / np.sum(src_w)
        return src_w

    def time_pdf_def_range(self, src_t, dt):
        """
        Returns the time window per source, in which the PDF ratio is defined.

        Parameters
        ----------
        src_t : array-like, shape (nsrcs)
            Times of each source event in MJD days.
        dt : array-like, shape (nsrcs, 2)
            Time windows [start, end] in seconds centered at each src_t in
            which the signal PDF is assumed to be uniform.

        Returns
        -------
        trange : array-like, shape (nsrcs, 2)
            Total time window per source [start, end] in seconds in which the
            time PDF is defined and thus non-zero.
        """
        src_t, dt, sig_t, sig_t_clip = self._setup_time_windows(src_t, dt)

        # Total time window per source in seconds
        trange = np.empty_like(dt, dtype=np.float)
        trange[:, 0] = dt[:, 0] - sig_t_clip
        trange[:, 1] = dt[:, 1] + sig_t_clip

        return trange

    def expect_weights(self, src_dec):
        """
        Returns an event expectation in units ``1/(T sr)`` per declination.

        If integrated over soild angle and multiplied with a flux normalization
        at ``1 GeV`` in units ``1/(GeV cm^2 s sr)``, this yields the total event
        rate from the MC signal sample used to build the PDFs.

        Parameters
        ----------
        src_dec : array-like, shape (nsrcs)
            Declination coordinate of each source in radian in interval
            ``[-pi/2, pi/2]``.

        Returns
        -------
        expect_w : array-like, shape (nsrcs, 1)
            Unnormalized event expectation per given ``src_dec`` in
            ``1/(T sr)``.
        """
        sin_dec = np.sin(src_dec)
        min_sin_dec, max_sin_dec = self._spatial_pdf_args["bins"][[0, -1]]
        if np.any(sin_dec < min_sin_dec) or np.any(sin_dec > max_sin_dec):
            raise ValueError("Requested weight for a sin_dec outside the " +
                             "declination binning.")
        return np.exp(self._nexpected_signal_spl(sin_dec))

    # #########################################################################
    # Signal over background probabilities for time, spatial and energy PDFs
    def _soverb(self, X, args):
        """
        Returns total signal over background ratio for given data X and fixed
        LLH arguments args.

        Parameters
        ----------
        X : record-array
            See :py:meth:`lnllh_ratio`, Parameters
        args : dict
            See :py:meth:`lnllh_ratio`, Parameters

        Returns
        -------
        sob : array-like, shape (nevts)
            Total signal over background ratio for each event, already reduced
            over all sources.

        Note:
        -----
        Which events contribute to the LLH is controlled using the options given
        in `GRBLLH.llh_args`.
        """
        # With no events given, we can skip this step
        if len(X) == 0:
            return np.empty(0, dtype=np.float)

        # Get data values
        t = X["timeMJD"]
        ev_ra = X["ra"]
        ev_sin_dec = X["sinDec"]
        ev_logE = X["logE"]
        ev_sig = X["sigma"]

        # Get other fixed paramters
        nb = args["nb"]
        srcs = args["srcs"]

        # Setup source parameters
        src_t = srcs["t"]
        dt = np.vstack((srcs["dt0"], srcs["dt1"])).T
        src_ra = srcs["ra"]
        src_dec = srcs["dec"]
        src_w_theo = srcs["w_theo"]

        # Per event probabilities
        sob = (self._soverb_time(t, src_t, dt) *
               self._soverb_spatial(src_ra, src_dec, ev_ra,
                                    ev_sin_dec, ev_sig) *
               self._soverb_energy(ev_sin_dec, ev_logE))

        # If mutliple srcs: sum over weighted signal contribution from each src
        # The single src case is automatically included due to broadcasting
        src_w = self.src_weights(src_dec, src_w_theo)
        # Background expecation per source
        sob = np.sum(sob * src_w / nb, axis=0)

        # Apply a SoB ratio cut, to save computation time on events that don't
        # contribute anyway. We have a relative and an absolute threshold
        sob_max = np.amax(sob)
        if sob_max > 0:
            sob_rel_mask = (sob / sob_max) < self._llh_args["sob_rel_eps"]
        else:
            sob_rel_mask = np.zeros_like(sob, dtype=bool)
        sob_abs_mask = sob < self._llh_args["sob_abs_eps"]

        # Only return events surviving both thresholds
        survive = np.logical_not(np.logical_or(sob_rel_mask, sob_abs_mask))
        return sob[survive]

    def _soverb_time(self, t, src_t, dt):
        """
        Time signal over background ratio.

        Signal and background PDFs are each normalized over seconds.
        Signal PDF has gaussian edges to smoothly let it fall of to zero, the
        stddev is dt when dt is in [2, 30]s, otherwise the nearest edge.

        To ensure finite support, the edges of the gaussian are truncated after
        nsig * dt.

        Parameters
        ----------
        t : array-like
            Times given in MJD for which we want to evaluate the ratio.
        src_t : array-like, shape (nsrcs)
            See :py:meth:`time_pdf_def_range`, Parameters
        dt : array-like, shape (nsrcs, 2)
            See :py:meth:`time_pdf_def_range`, Parameters

        Returns
        -------
        soverb_time_ratio : array-like, shape (nsrcs, len(t))
            Ratio of the time signal and background PDF for each given time `t`
            and per source time `src_t`.
        """
        nsig = self._time_pdf_args["nsig"]

        # Setup input to proper shapes
        src_t, dt, sig_t, sig_t_clip = self._setup_time_windows(src_t, dt)

        return backend.soverb_time(t, src_t, dt[:, 0], dt[:, 1],
                                   sig_t, sig_t_clip, nsig)

    def _soverb_spatial(self, src_ra, src_dec, ev_ra, ev_sin_dec, ev_sig):
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
        src_ra, src_dec : array-like, shape (nsrcs)
            Source positions in equatorial right-ascension, [0, 2pi] and
            declination, [-pi/2, pi/2], given in radian.
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
        S = self._pdf_spatial_signal(src_ra, src_dec, ev_ra, ev_sin_dec, ev_sig)
        B = self._pdf_spatial_background(ev_sin_dec)

        return S / B

    def _soverb_energy(self, ev_sin_dec, ev_logE):
        """
        Energy signal over background ratio.

        Energy has a lot of seperation power, because signal is following an
        astrophysical flux, which becomes dominant at higher energies over the
        flux of atmospheric background neutrinos.

        To account for different source positions on the whole sky, we create
        2 dimensional PDFs in sinus declination and in log10(E) of an energy
        estimator.

        The signal PDF is dervided from MC weighted to a specific unbroken poer
        law, the BG PDF is derived from data. A 2D histogram is used to fit a
        2D interpolating spline at the ration that describes a smooth PDF ratio.

        Outside of the definiton range, the PDF is set to zero.

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
        sob = np.zeros_like(ev_sin_dec)
        min_sin_dec, max_sin_dec = self._energy_pdf_args["bins"][0][[0, -1]]
        min_logE, max_logE = self._energy_pdf_args["bins"][1][[0, -1]]

        valid = ((ev_sin_dec >= min_sin_dec) & (ev_sin_dec <= max_sin_dec) &
                 (ev_logE >= min_logE) & (ev_logE <= max_logE))

        # scipy.interpolate.RegularGridInterpolator takes shape (nevts, ndim)
        pts = np.vstack((ev_sin_dec[valid], ev_logE[[valid]])).T
        sob[valid] = np.exp(self._energy_interpol(pts))

        return sob

    # #########################################################################
    # PDFs and PDF helper methods
    def _setup_time_windows(self, src_t, dt):
        """
        Bring the given source times and time windows in proper shape.

        Parameters
        ----------
        src_t : array-like, (nsrcs)
            See :py:meth:`time_pdf_def_range`, Parameters
        dt : array-like, shape (nsrcs, 2)
            See :py:meth:`time_pdf_def_range`, Parameters

        Returns
        -------
        src_t : array-like, (nsrcs)
            See :py:meth:`time_pdf_def_range`, Parameters
        dt : array-like, shape (nsrcs, 2)
            See :py:meth:`time_pdf_def_range`, Parameters
        sig_t : array-like, shape (nsrcs)
            sigma of the gaussian edges of the time signal PDF.
        sig_t_clip : array-like, shape (nsrcs)
            Total length `time_pdf_args['nsig'] * sig_t` of the gaussian edges
            of each time window.
        """
        nsig = self._time_pdf_args["nsig"]
        sigma_t_min = self._time_pdf_args["sigma_t_min"]
        sigma_t_max = self._time_pdf_args["sigma_t_max"]

        src_t = np.atleast_1d(src_t)
        dt = np.atleast_2d(dt)
        assert dt.shape[0] == len(src_t)
        assert dt.shape[1] == 2
        assert np.all(dt[:, 0] < dt[:, 1])

        # Constrain sig_t to given min/max, regardless of uniform time window
        dt_len = np.diff(dt, axis=1).ravel()
        sig_t = np.clip(dt_len, sigma_t_min, sigma_t_max)
        sig_t_clip = nsig * sig_t

        return src_t, dt, sig_t, sig_t_clip

    def _pdf_spatial_background(self, ev_sin_dec):
        """
        Calculate the value of the background PDF for each event.

        PDF is uniform in right-ascension and described by a spline fitted to
        data in sinus declination. Outside of the definiton range, the PDF is
        set to zero. The PDF is normalized is over the whole sphere in ra, dec.

        Parameters
        ----------
        ev_sin_dec
            See :py:meth:`GRBLLH._soverb_spatial`, Parameters

        Returns
        -------
        B : array-like, shape (nevts)
            The value of the background PDF for each event.
        """
        min_sin_dec, max_sin_dec = self._spatial_pdf_args["bins"][[0, -1]]
        if np.any(ev_sin_dec < min_sin_dec) or np.any(ev_sin_dec > max_sin_dec):
            raise ValueError("Requested to evaluate the spatial BG PDF " +
                             "outside it's definition range.")

        return 1. / (2. * np.pi) * np.exp(self._spatial_bg_spl(ev_sin_dec))

    def _pdf_spatial_signal(self, src_ra, src_dec, ev_ra, ev_sin_dec, ev_sig):
        """
        Spatial distance PDF between source position(s) and event positions.

        Signal is assumed to cluster around source position(s).
        The PDF is a convolution of a delta function for the localized sources
        and a Kent (or gaussian) distribution with the events positional
        reconstruction error as width.

        If `spatial_pdf_args["kent"]` is True a Kent distribtuion is used, where
        kappa is chosen, so that the same amount of probability as in the 2D
        gaussian is inside a circle with radius `ev_sig` per event.

        Multiple source positions can be given, to use it in a stacked search.

        Parameters
        -----------
        src_ra : array-like, shape (nsrcs)
            See :py:meth:`GRBLLH._soverb_spatial`, Parameters
        src_dec : array-like, shape (nsrcs)
            See :py:meth:`GRBLLH._soverb_spatial`, Parameters
        ev_ra : array-like, shape (nsrcs)
            See :py:meth:`GRBLLH._soverb_spatial`, Parameters
        ev_sin_dec : array-like, shape (nsrcs)
            See :py:meth:`GRBLLH._soverb_spatial`, Parameters
        ev_sig : array-like, shape (nsrcs)
            See :py:meth:`GRBLLH._soverb_spatial`, Parameters

        Returns
        --------
        S : array-like, shape (nsrcs, nevts)
            Spatial signal probability for each event and each source position.
        """
        assert len(src_ra.shape) == 1
        assert len(src_dec.shape) == 1

        return backend.pdf_spatial_signal(src_ra, src_dec,
                                          ev_ra, ev_sin_dec, ev_sig,
                                          self._spatial_pdf_args["kent"])

    def _create_sin_dec_logE_interpolator(self, sin_dec_bg, logE_bg, w_bg,
                                          sin_dec_sig, logE_sig, w_sig):
        """
        Create a 2D interpolatinon for a PDF ratio in sin(dec), energy proxy.

        The interpolation is done in the *natural logarithm* of the histogram.
        Fit parameters are controlled by the `self._energy_pdf_args` dict.

        Parameters
        ----------
        sin_dec_bg, logE_bg : array-like
            ``sin_dec_bg``: see :py:meth:`GRBLLH._soverb_spatial`, Parameters.
            ``logE_bg``: see :py:meth:`lnllh_ratio`, Parameters: `X`.
            Coordinates per event used to construct the BG histogram.
        w_bg : array-like, shape (len(sind_dec_bg))
            Per event weights to use in construction of the BG histogram.
        sin_dec_sig, logE_sig : array-like
            ``sin_dec_sig``: see :py:meth:`GRBLLH._soverb_spatial`, Parameters.
            ``logE_sig``: see :py:meth:`lnllh_ratio`, Parameters: `X`.
            Coordinates per event used to construct the signal histogram.
        w_sig : array-like, shape (len(sind_dec_bg))
            Per event weights to use in construction of the BG histogram.

        Returns
        -------
        _sin_dec_logE_interpol : scipy.interpolate.RegularGridInterpolator
            2D interpolator for the histogram. Must be evaluated with sin(dec)
            and logE to return the correct ratio values.
        """
        fillval = self._energy_pdf_args["fillval"]

        # Create binmids to fit spline to bin centers
        bins = self._energy_pdf_args["bins"]
        mids = []
        for b in bins:
            mids.append(0.5 * (b[:-1] + b[1:]))

        # Make 2D histograms for BG / signal using the same binning
        bg_h, _, _ = np.histogram2d(sin_dec_bg, logE_bg, bins=bins,
                                    weights=w_bg, normed=True)
        sig_h, _, _ = np.histogram2d(sin_dec_sig, logE_sig, bins=bins,
                                     weights=w_sig, normed=True)

        # Smooth each hist and renormalize
        [sigma_bg, sigma_sig] = self._energy_pdf_args["smooth_sigma"]
        bg_h = gaussian_filter(bg_h, sigma_bg)
        sig_h = gaussian_filter(sig_h, sigma_sig)

        dA = np.diff(bins[0])[:, None] * np.diff(bins[1])[None, :]
        bg_h = bg_h / (np.sum(bg_h) * dA)
        sig_h = sig_h / (np.sum(sig_h) * dA)
        assert np.isclose(np.sum(bg_h * dA), 1.)
        assert np.isclose(np.sum(sig_h * dA), 1.)

        # Check that all 1D sin_dec bins are populated
        _sin_dec_h = np.sum(bg_h, axis=1)
        if np.any(_sin_dec_h <= 0.):
            raise ValueError("Got empty sin_dec bins, this must not happen. " +
                             "Empty bins idx:\n{}".format(
                                 np.arange(len(bins[0]) - 1)[_sin_dec_h <= 0.]))

        # Fill all values where data has non-empty bins
        sob = np.ones_like(bg_h) - 1.
        mask = (bg_h > 0) & (sig_h > 0)
        sob[mask] = sig_h[mask] / bg_h[mask]
        if fillval in ["minmax", "min"]:
            sob_min, sob_max = np.amin(sob[mask]), np.amax(sob[mask])
        # We may have gaps in the hist, where no data OR no MC is. Fill with
        # interpolated values in sin_dec slice.
        # In each sin_dec slice assign values to bins with no data or no MC.
        for i in np.arange(len(bins[0]) - 1):
            # Assumption: sob is rising monotonically in the energy dimension.
            # So we go from top to bottom and rescale all violating bins.
            if self._energy_pdf_args["logE_asc"]:
                sob_m = sob[i] > 0
                masked_sob = sob[i][sob_m]
                for j in range(len(masked_sob) - 1, 0, -1):
                    if masked_sob[j] < masked_sob[j - 1]:
                        masked_sob[j - 1] = masked_sob[j]
                        # # Use mean in linspace if next to next is smaller
                        # if (j > 2) and (masked_sob[j - 2] < masked_sob[j]):
                        #     masked_sob[j - 1] = 0.5 * (masked_sob[j] +
                        #                              masked_sob[j - 2])
                        # else:  # Else just put it on the same value
                        #     masked_sob[j - 1] = masked_sob[j]
                sob[i][sob_m] = masked_sob

            # Get invalid points in sin_dec slice
            m = sob[i] <= 0

            if fillval in ["minmax_col"]:  # min/max per slice instead of global
                sob_min, sob_max = np.amin(sob[i][~m]), np.amax(sob[i][~m])

            # Only fill missing logE border values, rest is interpolated
            # Lower edge: argmax stops at first True, argmin at first False
            low_first_invalid_id = np.argmax(m)
            if low_first_invalid_id == 0:
                # Set lower edge with valid point, depending on 'fillval'
                if fillval == "col":  # Fill with first valid ratio from bottom
                    low_first_valid_id = np.argmin(m)
                    sob[i, 0] = sob[i, low_first_valid_id]
                elif fillval in ["minmax", "minmax_col", "min"]:
                    sob[i, 0] = sob_min  # Fill with global | per bin min

            # Repeat with turned around array for upper edge
            hig_first_invalid_id = np.argmax(m[::-1])
            if hig_first_invalid_id == 0:
                if fillval == "col":  # Fill with first valid ratio from top
                    hig_first_valid_id = len(m) - 1 - np.argmin(m[::-1])
                    sob[i, -1] = sob[i, hig_first_valid_id]
                elif fillval == "min":  # Fill also with global min
                    sob[i, -1] = sob_min
                elif fillval in ["minmax", "minmax_col"]:  # glob. | per bin max
                    sob[i, -1] = sob_max

            # Interpolate in each slice over missing entries
            sob_m = sob[i] > 0
            x = mids[1][sob_m]
            y = sob[i, sob_m]
            if self._energy_pdf_args["interpol_log"]:
                fi = sci.interp1d(x, np.log(y), kind="linear")
                sob[i] = np.exp(fi(mids[1]))
            else:
                fi = sci.interp1d(x, y, kind="linear")
                sob[i] = fi(mids[1])

        # Now fit a 2D interpolating spline to the ratio
        spl = sci.RegularGridInterpolator(mids, np.log(sob), method="linear",
                                          bounds_error=False, fill_value=None)
        return spl

    def _normed_sin_dec_spline(self, sin_dec, bins, weights):
        """
        Fit an interpolating spline to a histogram of ``sin_dec`` and normalize
        so it is a PDF in ``sin_dec``.

        Spline is extrapolated outside it's definition range.

        Parameters
        ----------
        sin_dec : array-like, shape (nevts)
            Equatorial sinus declination coordinates in ``[-1, 1]``.
        bins : array-like, shape (nbins + 1)
            Explicit bin edges to use in the ``sin_dec`` histogram.
        weights : array-like, shape (nevts)
            Weights used in histogram creation.

        Returns
        -------
        sin_dec_spl : scipy.interpolate.InterpolatingSpline
            Spline object interpolating the created ``sin_dec`` histogram.
            Must be evaluated with sin(dec) and exponentiated to give the
            correct PDF values.
        """
        assert np.all((sin_dec >= bins[0]) & (sin_dec <= bins[-1]))

        # Make normalised hist to fit the spline to x, y pairs
        hist, bins = np.histogram(sin_dec, bins=bins, weights=weights,
                                  density=True)

        if np.any(hist <= 0.):
            raise ValueError("Got empty sin_dec hist bins, this must not " +
                             "happen. Empty bins idx:\n{}".format(
                                 np.arange(len(bins) - 1)[hist <= 0.]))

        return self._fit_spline_to_hist(hist, bins)

    def _nexpected_per_sin_dec_spline(self, sin_dec, trueE, ow, bins):
        """
        Make a spline describing the expected number of events for signal from a
        histogram::

        .. math:

          \frac{n_\text{exp}}{T\Delta\Omega} =
            \sum_{i\in\Delta E, \Delta \Omega}
                \frac{\text{ow}_i (E_i/\text{GeV})^{-\gamma}}{{\Delta\Omega}}

        Parameters
        ----------
        sin_dec : array-like, shape (nevts)
            Equatorial sinus declination coordinates in ``[-1, 1]``.
        bins : array-like, shape (nbins + 1)
            Explicit bin edges to use in the ``sin_dec`` histogram.

        Returns
        -------
        eff_area_spl : scipy.interpolate.InterpolatingSpline
            Spline object interpolating the created effective area histogram.
            Must be exponentiated to give the correct values.
        """
        # Not using a normalization: would be global constant for all weigths
        w_sig = ow * power_law_flux_per_type(trueE,
                                             self._energy_pdf_args["gamma"])
        hist, bins = np.histogram(sin_dec, bins=bins, weights=w_sig,
                                  density=False)

        if np.any(hist <= 0.):
            raise ValueError("Got empty sin_dec hist bins, this must not " +
                             "happen. Empty bins idx:\n{}".format(
                                 np.arange(len(bins) - 1)[hist <= 0.]))

        # Make effective area by normalizing by the bin volume
        hist /= np.diff(bins) * 2. * np.pi
        return self._fit_spline_to_hist(hist, bins)

    def _fit_spline_to_hist(self, h, bins):
        """
        Fit an interpolating spline to a histogram in ln-space.

        Spline is extrapolated outside it's definition range.

        Parameters
        ----------
        h : array-like
            Histogram values per bin.
        bins : array-like, shape (len(h) + 1)
            Explicit bin edges for the the histogram.

        Returns
        -------
        spl : scipy.interpolate.InterpolatingSpline
            Spline object interpolating the histogram. Must be exponentiated to
            give the correct histograms values.
        """
        k = self._spatial_pdf_args["k"]
        mids = 0.5 * (bins[:-1] + bins[1:])
        # Add the outermost bin edges to avoid overshoots at the edges
        x = np.concatenate((bins[[0]], mids, bins[[-1]]))
        y = np.log(h)
        y = np.concatenate((y[[0]], y, y[[-1]]))
        return sci.InterpolatedUnivariateSpline(x, y, k=k, ext="extrapolate")

    def __str__(self):
        """
        Use to print all settings: `>>> print(llh_object)`
        """
        rep = "GRBLLH object\n"
        rep += "-------------\n\n"

        def shorten(val, cut=10):
            """Shorten lengthy arguments for print"""
            try:
                length = len(val)
            except TypeError:
                length = 1
            if length > cut:
                val = ("{}".format(val[:int(cut / 2)]).replace("]", "") +
                       " ... {}".format(val[-int(cut / 2):]).replace("[", ""))
            return val

        rep += "Spatial PDF settings:\n"
        for key, val in self._spatial_pdf_args.items():
            rep += "  - {:12s} : {}\n".format(key, shorten(val))

        rep += "\n"

        rep += "Time PDF settings:\n"
        for key, val in self._time_pdf_args.items():
            rep += "  - {:12s} : {}\n".format(key, shorten(val))

        rep += "\n"

        rep += "Energy PDF settings:\n"
        for key, val in self._energy_pdf_args.items():
            rep += "  - {:12s} : {}\n".format(key, shorten(val))

        rep += "\n"

        rep += "LLH settings:\n"
        for key, val in self._llh_args.items():
            rep += "  - {:12s} : {}\n".format(key, shorten(val))

        return rep


class MultiyearGRBLLH(object):
    def __init__(self):
        self._names = []
        self._llhs = []
        return

    @property
    def names(self):
        return self._names

    @property
    def llhs(self):
        return self._llh

    def add_sample(self, name, llh):
        """
        Add a LLH object to consider.

        Parameters
        ----------
        name : str
            Name of the LLH object. Should be connected to the dataset used.
        llh : tdepps.llh.GRBLLH
            LLH object holding all the PDF information for the sample.
        """
        if not isinstance(llh, GRBLLH):
            raise ValueError("`llh` object must be of type GRBLLH.")

        if name in self._names:
            raise KeyError("Name '{}' has already been added. ".format(name) +
                           "Choose a different name.")

        self._names.append(name)
        self._llhs.append(llh)
        return

    def lnllh_ratio(self, X, ns, args):
        r"""
        Calculate the lnllh ratio for the multi year case.

        The total LLH is the sum of all single LLHs operating on disjunct data
        sets each. ``ns`` gets split up over the data sets to regard detection
        efficiency per year.

        Parameters
        ----------
        X : dict of record-arrays
            Fixed data set each LLH depends on, given as a dict. Each value must
            be a record array as used in the single LLH class.
        ns : float
            Number of signal events at the source locations for all years in
            total.
        args : list of dicts
            Other fixed parameters each LLH depends on, given as a dict. Each
            value must be a dict as used in the single LLH class.
            List must have length of number of added samples.

        Returns
        -------
        TS : float
            Lambda test statistic, 2 times the natural logarithm of the LLH
            ratio.
        ns_grad : array-like, shape (1)
            Gradient of the test statistic in the fit parameter `ns`.
        """
        raise NotImplementedError("TODO")
        return

    def fit_lnllh_ratio(self, X, ns0, args, bounds, minimizer_opts):
        """
        Fit the LLH parameter :math:`n_S` for a given set of data and fixed
        LLH arguments for the multi year case

        The total LLH is the sum of all single LLHs operating on disjunct data
        sets each. The fitted ``ns`` parameter gets split up over the data sets
        to regard detection efficiency per year.
        The relative weight is the effective area per year summed over every
        source position. For a single source, this reduces to a single detection
        efficiency per year at the sources position.

        Parameters
        ----------
        X : dict of record-arrays
            Fixed data set each LLH depends on, given as a dict. Each value
            must be a record array as used in the single LLH class.
        ns0 : float
            Fitter seed for the fit parameter ``ns``: number of signal events
            that we expect at the source locations.
        args : list of dicts
            Other fixed parameters each LLH depends on, given as a dict. Each
            value must be a dict as used in the single LLH class.
            List must have length of number of added samples.
        bounds : array-like, shape (1, 2)
            Minimization bounds ``[[min, max]]`` for ``ns``. Use None for one of
            ``min`` or ``max`` when there is no bound in that direction.
        minimizer_opts : dict
            Options passed to ``scipy.optimize.minimize`` [5]_ using the
            "L-BFGS-B" algorithm.

        Returns
        -------
        ns : float
            Best fit parameter number of signal events :math:`n_S`.
        TS : float
            Best fit test statistic value.

        Notes
        -----
        .. [5] https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html_minimize.py#L36-L466
        """
        raise NotImplementedError("TODO")
        return


class MaxBurstGRBLLH(GRBLLH):
    # TODO: Redefine the TS only here. Instead of stacking, evaluate single LLH
    #       For every source seperately. The best fit TS is then max_i(TS_i).
    def __init__(self):
        raise NotImplementedError("Not done yet.")
        return
