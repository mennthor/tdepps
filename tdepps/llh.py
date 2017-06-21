# coding: utf-8

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()                                              # noqa

import math
import numpy as np
import scipy.stats as scs
import scipy.optimize as sco
import scipy.interpolate as sci

from .utils import fill_dict_defaults


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

        - "dec", float: Per event equatorial declination, given in radians.
        - "logE", float: Per event energy proxy, given in log10(1/GeV).

    MC : record-array
        Global Monte Carlo data set, used to derive per event PDFs. `MC` must
        contain the same names as `X` and additionaly the MC truths:

        - "trueE", float: True event energy in GeV.
        - "ow", float: Per event "neutrino generator (NuGen)" OneWeight [2]_,
          already divided by `nevts * nfiles` known from SimProd.
          Units are "GeV sr cm^2". Final event weights are obtained by
          multiplying with desired flux.

    spatial_pdf_args : dict
        Arguments for the spatial signal and background PDF. Must contain keys:

        - "bins", array-like: Explicit bin edges of the sinus declination
          histogram used to fit a spline describing the spatial background PDF.
          Bins must be in range [-1, 1].
        - "kent", bool, optional: If True, the signal PDF uses the Kent [3]_
          distribution. A 2D gaussian PDF is used otherwise. (default: True)
        - "k", int, optional: Degree of the smoothing spline used to fit the
          background histogram. Must be 1 <= k <= 5. (default: 3)

    energy_pdf_args : dict
        Arguments for the energy PDF ratio. Must contain keys:

        - "bins", array-like: Explicit bin edges of the sinus declination vs
          logE histogram used to fit a 2D spline describing the energy PDF
          ratio. Must be [sin_dec_bins, logE_bins] in ranges [-1, 1] for sinus
          declination and [-inf, +inf] for logE.
        - "gamma", float, optional: Spectral index of the power law
          :math:`E^{-\gamma}` used to weight MC to an astrophisical flux.
          (default: 2.)
        - "fillval", str, optional: What values to use, when the histogram has
          MC but no data in a bin. Then the gaps are filled, by assigning values
          to the histogram edges for low/high energies seprately and then
          interpolating inside. Can be one of ["minmax"|"col"]. When "minmax"
          the lowest/highest ratio values are used at the edges, when "col" the
          next valid value in each colum from the top/bottom is used.
          "col" is more conservative, "minmax" more optimistic. (default: "col")
        - "interpol_log", bool, optional: If True, gaps in the signal over
          background ratio histogram are interpolated linearly in ln. Otherwise
          the interpolation is in linear space. (default: False)

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
    .. [2] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight # noqa: 501
    .. [3] https://en.wikipedia.org/wiki/Kent_distribution
    """
    def __init__(self, X, MC, spatial_pdf_args, energy_pdf_args,
                 time_pdf_args=None, llh_args=None):
        # Check if data, MC have all needed names
        X_names = ["dec", "logE"]
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
        self.spatial_pdf_args = fill_dict_defaults(spatial_pdf_args,
                                                   required_keys, opt_keys)
        bins = self.spatial_pdf_args["bins"]
        k = self.spatial_pdf_args["k"]
        if np.any(bins < -1) or np.any(bins > 1):
            raise ValueError("Bins for BG spline not in valid range [-1, 1].")
        if (k < 1) or (k > 5):
            raise ValueError("'k' must be integer in [1, 5].")

        # Setup energy PDF args
        required_keys = ["bins"]
        opt_keys = {"gamma": 2., "fillval": "col", "interpol_log": False}
        self.energy_pdf_args = fill_dict_defaults(energy_pdf_args,
                                                  required_keys, opt_keys)
        if len(self.energy_pdf_args["bins"]) != 2:
            raise ValueError("Bins for energy hist must have shape " +
                             "[sin_dec_bins, logE_bins].")
        sin_dec_bins = np.atleast_1d(self.energy_pdf_args["bins"][0])
        logE_bins = np.atleast_1d(self.energy_pdf_args["bins"][1])
        if np.any(sin_dec_bins < -1) or np.any(sin_dec_bins > 1):
            raise ValueError("Sinus declination bins for energy spline not in" +
                             " valid range [-1, 1].")
        if self.energy_pdf_args["fillval"] not in ["minmax", "col"]:
            raise ValueError("'fillval' must be one of ['minmax'|'col'].")

        # Setup time PDF args
        required_keys = []
        opt_keys = {"nsig": 4., "sigma_t_min": 2., "sigma_t_max": 30.}
        self.time_pdf_args = fill_dict_defaults(time_pdf_args, required_keys,
                                                opt_keys)
        nsig = self.time_pdf_args["nsig"]
        if nsig < 3:
            raise ValueError("'nsig' must be >= 3.")

        tmin = self.time_pdf_args["sigma_t_min"]
        tmax = self.time_pdf_args["sigma_t_max"]
        if tmin > tmax or tmin == tmax:
            raise ValueError("'sigma_t_min' must be < 'sigma_t_max'.")
        if tmin <= 0:
            raise ValueError("'sigma_t_min' must be > 0.")

        # Setup LLH args
        required_keys = []
        opt_keys = {"sob_rel_eps": 0, "sob_abs_eps": 1e-3}
        self.llh_args = fill_dict_defaults(llh_args, required_keys, opt_keys)

        rel_eps = self.llh_args["sob_rel_eps"]
        if (rel_eps < 0) or (rel_eps > 1):
            raise ValueError("'sob_rel_eps' must be in [0, 1]")
        if self.llh_args["sob_abs_eps"] < 0:
            raise ValueError("'sob_abs_eps' must be >= 0.")

        # Setup common variables
        self.energy_pdf_args["bins"] = [sin_dec_bins, logE_bins]
        self._SECINDAY = 24. * 60. * 60.

        # Create background spline used in the spatial PDF from global data
        ev_sin_dec = np.sin(X["dec"])
        ev_bins = self.spatial_pdf_args["bins"]
        self._spatial_bg_spl = self._create_sin_dec_spline(
            sin_dec=ev_sin_dec, bins=ev_bins, mc=None)

        # Create energy PDF from global data and MC
        mc_sin_dec = np.sin(MC["trueDec"])
        self._energy_spl = self._create_sin_dec_logE_spline(
            ev_sin_dec, X["logE"],
            mc_sin_dec, MC["logE"], MC["trueE"], MC["ow"])

        # Create sin_dec signal spline for the src detector weights from MC
        mc_sin_dec = np.sin(MC["dec"])
        mc_bins = self.energy_pdf_args["bins"][0]
        mc_dict = {"trueE": MC["trueE"], "ow": MC["ow"]}
        self._spatial_signal_spl = self._create_sin_dec_spline(
            sin_dec=mc_sin_dec, bins=mc_bins, mc=mc_dict)

        return

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
              coordinates, given in radians and sinus declination in
              :math:`[-1, 1]`.
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
            Other fixed parameters {'par_name': value}, the LLH depents on.
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
            See :func:`lnllh_ratio`, Parameters
        ns0 : float
            Fitter seed for the fit parameter ns: number of signal events that
            we expect at the source locations.
        args : dict
            See :func:`lnllh_ratio`, Parameters
        bounds : array-like, shape (1, 2)
            Minimization bounds `[[min, max]]` for `ns`. Use None for one of
            `min` or `max` when there is no bound in that direction.
        minimizer_opts : dict
            Options passed to `scipy.optimize.minimize` [4] using the "L-BFGS-B"
            algorithm.



        Returns
        -------
        ns : float
            Best fit parameter number of signal events :math:`n_S`.
        TS : float
            Best fit test statistic value.

        Notes
        -----
        .. [4] https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html_minimize.py#L36-L466 # noqa
        """
        def _neglnllh(ns):
            """
            Wrapper for the LLH function returning the negative ln-LLH ratio
            suitable for the minimizer.

            Parameters
            ----------
            ns : float
                See :func:`lnllh_ratio`, Parameters

            Returns
            -------
            lnllh : float
                See :func:`lnllh_ratio`, Returns
            lnllh_grad : array-like
                See :func:`lnllh_ratio`, Returns
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
                TS = -ns + math.log(sob)
            return ns, 2. * TS
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
            See :func:`lnllh_ratio`, Parameters

        Returns
        -------
        lnllh : float
            See :func:`lnllh_ratio`, Returns
        lnllh_grad : array-like
            See :func:`lnllh_ratio`, Returns
        """
        # Teststatistic 2 * ln(LLH-ratio)
        x = ns * sob
        TS = 2. * (-ns + np.sum(np.log1p(x)))
        # Gradient in ns (chain rule: ln(x + 1)' * x')
        ns_grad = 2. * (-1. + np.sum(sob / (x + 1.)))
        return TS, np.atleast_1d(ns_grad)

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
            [-pi/2, pi/2].
        src_w_theo : array-like, shape (nsrcs)
            Theoretical source weight per source, eg. from a known gamma flux.

        Returns
        -------
        src_w : array-like, shape (nsrcs, 1)
            Combined normalized weight per source.
        """
        # Get src detector weights form signal sin_dec spline from MC
        src_dec_w = np.exp(self._spatial_signal_spl(np.sin(src_dec)))

        # Make combined src weight by multiplying with the theoretical weights
        src_w = src_dec_w * src_w_theo
        nsrcs = len(src_dec)
        src_w = src_w.reshape(nsrcs, 1) / np.sum(src_w)
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
        trange[:, 0] = dt[:, 0] - sig_t_clip.ravel()
        trange[:, 1] = dt[:, 1] + sig_t_clip.ravel()

        return trange

    # #########################################################################
    # Signal over background probabilities for time, spatial and energy PDFs
    def _soverb(self, X, args):
        """
        Returns total signal over background ratio for given data X and fixed
        LLH arguments args.

        Parameters
        ----------
        X : record-array
            See :func:`lnllh_ratio`, Parameters
        args : dict
            See :func:`lnllh_ratio`, Parameters

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

        # If mutliple srcs: sum over signal contribution from each src.
        # The single src case is automatically included due to broadcasting
        src_w = self.src_weights(src_dec, src_w_theo)
        nb = nb.reshape(len(nb), 1)
        sob = np.sum(sob * src_w / nb, axis=0)

        # Apply a SoB ratio cut, to save computation time on events that don't
        # contribute anyway. We have a relative and an absolute threshold
        sob_max = np.amax(sob)
        if sob_max > 0:
            sob_rel_mask = (sob / sob_max) < self.llh_args["sob_rel_eps"]
        else:
            sob_rel_mask = np.zeros_like(sob, dtype=bool)
        sob_abs_mask = sob < self.llh_args["sob_abs_eps"]

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
        src_t, dt
            See `time_pdf_def_range`, Parameters

        Returns
        -------
        soverb_time_ratio : array-like, shape (nsrcs, len(t))
            Ratio of the time signal and background PDF for each given time `t`
            and per source time `src_t`.
        """
        nsig = self.time_pdf_args["nsig"]

        # Setup input to proper shapes
        src_t, dt, sig_t, sig_t_clip = self._setup_time_windows(src_t, dt)
        nsrc = dt.shape[0]
        dt_len = np.diff(dt, axis=1)

        # Create signal PDF
        gaus_norm = np.sqrt(2 * np.pi) * sig_t

        # Normalize times from data relative to src_t in seconds
        # Stability: Multiply before subtracting avoids small number rounding(?)
        _t = t * self._SECINDAY - src_t * self._SECINDAY

        # Broadcast
        dt0 = dt[:, 0].reshape(nsrc, 1)
        dt1 = dt[:, 1].reshape(nsrc, 1)
        # Split in PDF regions: gauss rising, uniform, gauss falling
        gr = (_t < dt0) & (_t >= dt0 - sig_t_clip)
        gf = (_t > dt1) & (_t <= dt1 + sig_t_clip)
        uni = (_t >= dt0) & (_t <= dt1)

        # Broadcast
        nevts = len(t)
        _dt0 = np.repeat(dt[:, 0].reshape(nsrc, 1), axis=1, repeats=nevts)
        _dt1 = np.repeat(dt[:, 1].reshape(nsrc, 1), axis=1, repeats=nevts)
        _sig_t = np.repeat(sig_t.reshape(nsrc, 1), axis=1, repeats=nevts)
        _gaus_norm = np.repeat(gaus_norm.reshape(nsrc, 1),
                               axis=1, repeats=nevts)
        # Get pdf values in the masked regions
        pdf = np.zeros_like(_t, dtype=np.float)
        pdf[gr] = scs.norm.pdf(_t[gr], loc=_dt0[gr], scale=_sig_t[gr])
        pdf[gf] = scs.norm.pdf(_t[gf], loc=_dt1[gf], scale=_sig_t[gf])
        # Connect smoothly with the gaussians
        pdf[uni] = 1. / _gaus_norm[uni]

        # Normalize signal distribtuion: Prob in half gaussians + uniform part
        dcdf = (scs.norm.cdf(nsig, loc=0, scale=1) -
                scs.norm.cdf(-nsig, loc=0, scale=1))
        norm = dcdf + dt_len / gaus_norm
        pdf /= norm

        # Calculate the ratio signal / background
        bg_pdf = 1. / (dt_len + 2 * sig_t_clip)
        pdf /= bg_pdf
        return pdf

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
            See `GRBLLH._soverb_spatial`, Parameters
        ev_logE
            See `lnllh_ratio`, Parameters: `X`

        Returns
        -------
        soverb_energy_ratio : array-like, shape (nevts)
            Ratio of the energy signal and background PDF for each given event.
        """
        sob = np.zeros_like(ev_sin_dec)
        min_sin_dec, max_sin_dec = self.energy_pdf_args["bins"][0][[0, -1]]
        min_logE, max_logE = self.energy_pdf_args["bins"][1][[0, -1]]

        valid = ((ev_sin_dec >= min_sin_dec) & (ev_sin_dec <= max_sin_dec) &
                 (ev_logE >= min_logE) & (ev_logE <= max_logE))

        # scipy.interpolate.RegularGridInterpolator takes shape (nevts, ndim)
        pts = np.vstack((ev_sin_dec[valid], ev_logE[[valid]])).T
        sob[valid] = np.exp(self._energy_spl(pts))

        return sob

    # #########################################################################
    # PDFs and PDF helper methods
    def _setup_time_windows(self, src_t, dt):
        """
        Bring the given source times and time windows in proper shape.

        Parameters
        ----------
        src_t, dt
            See `time_pdf_def_range`, Parameters

        Returns
        -------
        src_t, dt
            See `time_pdf_def_range`, Parameters
        sig_t : array-like, shape (nsrcs)
            sigma of the gaussian edges of the time signal PDF.
        sig_t_clip : array-like, shape (nsrcs)
            Total length nsig * sig_t of the gaussian edges of each time window.
        """
        nsig = self.time_pdf_args["nsig"]
        sigma_t_min = self.time_pdf_args["sigma_t_min"]
        sigma_t_max = self.time_pdf_args["sigma_t_max"]

        src_t = np.atleast_1d(src_t)
        nsrc = len(src_t)
        dt = np.atleast_2d(dt)
        if dt.shape[1] != 2:
            raise ValueError("Timeframe 'dt' must be [start, end] in seconds" +
                             " for each source.")
        if dt.shape[0] != nsrc:
            raise ValueError("Length of 'src_t' and 'dt' must be equal.")
        if np.any(dt[:, 0] >= dt[:, 1]):
            raise ValueError("Interval 'dt' must not be negative or zero.")

        # Each src in its own array for proper broadcasting
        src_t = np.atleast_2d(src_t).reshape(nsrc, 1)

        # Constrain sig_t to given min/max, regardless of uniform time window
        dt_len = np.diff(dt, axis=1)
        sig_t = np.clip(dt_len, sigma_t_min, sigma_t_max)
        sig_t_clip = nsig * sig_t

        return src_t, dt, sig_t, sig_t_clip

    def _pdf_spatial_background(self, ev_sin_dec):
        """
        Calculate the value of the background PDF for each event.

        PDF is uniform in right-ascension and described by a spline fitted to
        data in sinus declination. Outside of the definiton range, the PDF is
        set to zero.

        Parameters
        ----------
        ev_sin_dec
            See `GRBLLH._soverb_spatial`, Parameters

        Returns
        -------
        B : array-like, shape (nevts)
            The value of the background PDF for each event.
        """
        # TODO: Maybe better raise a value error? Otherwise ratio is +inf which
        #       might boost sensitivity. Or make the test in soverb_spatial.
        B = np.zeros_like(ev_sin_dec)
        min_sin_dec, max_sin_dec = self.spatial_pdf_args["bins"][[0, -1]]
        valid = (ev_sin_dec >= min_sin_dec) & (ev_sin_dec <= max_sin_dec)
        B[valid] = 1. / (2. * np.pi) * np.exp(self._spatial_bg_spl(
            ev_sin_dec[valid]))

        return B

    def _pdf_spatial_signal(self, src_ra, src_dec, ev_ra, ev_sin_dec, ev_sig):
        """
        Spatial distance PDF between source position(s) and event positions.

        Signal is assumed to cluster around source position(s).
        The PDF is a convolution of a delta function for the localized sources
        and a Kent (or gaussian) distribution with the events positional
        reconstruction error as width.

        If `self.kent` is True a Kent distribtuion is used, where the kappa
        is chosen, so that the same amount of probability as in the 2D gaussian
        is inside a circle with radius `ev_sig` per event.

        Multiple source positions can be given, to use it in a stacked search.

        Parameters
        -----------
        src_ra, src_dec, ev_ra, ev_sin_dec, ev_sig
            See `GRBLLH._soverb_spatial`, Parameters

        Returns
        --------
        S : array-like, shape (nsrcs, nevts)
            Spatial signal probability for each event and each source position.
        """
        nsrcs = len(np.atleast_1d(src_ra))
        # Shape (nsrcs, 1) to use broadcasting to shape (nsrcs, nevts)
        src_ra = np.atleast_2d(src_ra).reshape(nsrcs, 1)
        src_dec = np.atleast_2d(src_dec).reshape(nsrcs, 1)

        # Dot product to get great circle distance for every evt to every src
        cos_dist = (np.cos(src_ra - ev_ra) *
                    np.cos(src_dec) * np.sqrt(1. - ev_sin_dec**2) +
                    np.sin(src_dec) * ev_sin_dec)

        # Handle possible floating precision errors
        cos_dist = np.clip(cos_dist, -1, 1)

        if self.spatial_pdf_args["kent"]:
            # Stabilized version for possibly large kappas
            kappa = 1. / ev_sig**2
            S = (kappa / (2. * np.pi * (1. - np.exp(-2. * kappa))) *
                 np.exp(kappa * (cos_dist - 1.)))
        else:
            # Otherwise use standard symmetric 2D gaussian
            dist = np.arccos(cos_dist)
            ev_sig_2 = 2 * ev_sig**2
            S = np.exp(-dist**2 / ev_sig_2) / (np.pi * ev_sig_2)

        return S

    def _create_sin_dec_logE_spline(self, ev_sin_dec, ev_logE,
                                    mc_sin_dec, mc_logE, trueE, ow):
        """
        Create a 2D interpolating spline describing the energy signal over
        background ratio.

        The spline is fitted to the *natural logarithm* of the histogram, to
        avoid ringing. Normalization is done by normalizing the hist, so it may
        be slightly off, but that's tolerable.

        Fit parameters are controlled by the `self.energy_pdf_args` dict.

        Parameters
        ----------
        ev_sin_dec
            See `GRBLLH._soverb_spatial`, Parameters
        ev_logE
            See `lnllh_ratio`, Parameters: `X`
        mc_sin_dec, mc_logE, trueE, ow
            See `GRBLLH`, Parameters: `MC`

        Returns
        -------
        _sin_dec_logE_spline : scipy.interpolate.RegularGridInterpolator
            2D Spline object interpolating the histogram. Must be evaluated with
            sin(dec) and logE to give the correct ratio values.
        """
        gamma = self.energy_pdf_args["gamma"]
        fillval = self.energy_pdf_args["fillval"]

        # Create binmids to fit spline to bin centers
        bins = self.energy_pdf_args["bins"]
        mids = []
        for b in bins:
            mids.append(0.5 * (b[:-1] + b[1:]))

        if np.any((ev_logE < bins[1][0]) | (ev_logE > bins[1][-1])):
            raise ValueError("Not all logE events fall into given bins. If " +
                             "this is intended, please remove them beforehand.")

        # Weight MC to power law *shape* only, because we normalize anyway to
        # get a PDF
        mc_w = ow * trueE**(-gamma)

        # Make 2D hist from data and from MC, using the same binning
        mc_h, _, _ = np.histogram2d(mc_sin_dec, mc_logE, bins=bins,
                                    weights=mc_w, normed=True)

        bg_h, _, _ = np.histogram2d(ev_sin_dec, ev_logE, bins=bins, normed=True)

        # Check that all 1D sin_dec bins are populated
        _sin_dec_h = np.sum(bg_h, axis=1)
        if np.any(_sin_dec_h <= 0.):
            raise ValueError("Got empty sin_dec bins, this must not happen. " +
                             "Empty bins idx:\n{}".format(
                                 np.arange(len(bins[0]) - 1)[_sin_dec_h <= 0.]))

        # Fill all values where data has non-empty bins
        sob = np.ones_like(bg_h) - 1.
        mask = (bg_h > 0) & (mc_h > 0)
        sob[mask] = mc_h[mask] / bg_h[mask]
        if fillval == "minmax":
            sob_min, sob_max = np.amin(sob[mask]), np.amax(sob[mask])
        # We may have gaps in the hist, where no data OR no MC is. Fill with
        # interpolated values in sin_dec slice.
        # In each sin_dec slice assign values to bins with no data or no MC.
        for i in np.arange(len(bins[0]) - 1):
            # Get invalid points in sin_dec slice
            m = (bg_h[i] <= 0) | (mc_h[i] <= 0)

            # Only fill missing logE border values, rest is interpolated
            # Lower edge: argmax stops at first True, argmin at first False
            low_first_invalid_id = np.argmax(m)
            if low_first_invalid_id == 0:
                # Set lower edge with valid point, depending on 'fillval'
                if fillval == "col":  # Fill with first valid ratio from bottom
                    low_first_valid_id = np.argmin(m)
                    sob[i, 0] = sob[i, low_first_valid_id]
                elif fillval == "minmax":  # Fill with global min
                    sob[i, 0] = sob_min

            # Repeat with turned around array for upper edge
            hig_first_invalid_id = np.argmax(m[::-1])
            if hig_first_invalid_id == 0:
                if fillval == "col":  # Fill with first valid ratio from top
                    hig_first_valid_id = len(m) - 1 - np.argmin(m[::-1])
                    sob[i, -1] = sob[i, hig_first_valid_id]
                elif fillval == "minmax":  # Fill with global max
                    sob[i, -1] = sob_max

            # Interpolate in each slice over missing entries
            m = sob[i] > 0
            x = mids[1][m]
            y = sob[i, m]
            if self.energy_pdf_args["interpol_log"]:
                fi = sci.interp1d(x, np.log(y), kind="linear")
                sob[i] = np.exp(fi(mids[1]))
            else:
                fi = sci.interp1d(x, y, kind="linear")
                sob[i] = fi(mids[1])

        # # Now fit a 2D interpolating spline to the ratio
        spl = sci.RegularGridInterpolator(mids, np.log(sob), method="linear",
                                          bounds_error=False, fill_value=None)
        return spl

    def _create_sin_dec_spline(self, sin_dec, bins, mc=None):
        """
        Fit an interpolating spline to a histogram of sin(dec).

        Spline is extrapolated outside it's definition range.

        Parameters
        ----------
        sin_dec : array-like, shape (nevts)
            Equatorial sinus declination coordinates in [-1, 1].
        bins : array-like, shape (nbins + 1)
            Explicit bin edges to use in the sin_dec histogram.
        mc : dict, optional
            If dict, then it hold additional monte carlo information used to
            create the spline on simulation data. Must then have keys:

            - "trueE", array: True energy in GeV from MC simulation.
            - "ow", array: Neutrino generator oneweights, already divided by the
              number of generated events.

            (default: None)

        Returns
        -------
        sin_dec_spl : scipy.interpolate.InterpolatingSpline
            Spline object interpolating the created histogram. Must be evaluated
            with sin(dec) and exponentiated to give the correct PDF values.

        """
        k = self.spatial_pdf_args["k"]

        if np.any((sin_dec < bins[0]) | (sin_dec > bins[-1])):
            raise ValueError("Not all sinDec events fall into given bins. If " +
                             "this is intended, please remove them beforehand.")

        if mc is not None:
            # Weight MC to power law shape only, because we normalize anyway
            gamma = self.energy_pdf_args["gamma"]
            weights = mc["ow"] * mc["trueE"]**(-gamma)
        else:
            weights = np.ones_like(sin_dec)

        # Make normalised hist to fit the spline to x, y pairs
        hist, bins = np.histogram(sin_dec, bins=bins, weights=weights,
                                  density=True)

        if np.any(hist <= 0.):
            raise ValueError("Got empty ev_sin_dec hist bins, this must not " +
                             "happen. Empty bins idx:\n{}".format(
                                 np.arange(len(bins) - 1)[hist <= 0.]))

        mids = 0.5 * (bins[:-1] + bins[1:])
        # Add the outermost bin edges to avoid overshoots at the edges
        x = np.concatenate((bins[[0]], mids, bins[[-1]]))
        y = np.log(hist)
        y = np.concatenate((y[[0]], y, y[[-1]]))
        return sci.InterpolatedUnivariateSpline(x, y, k=k, ext="extrapolate")

    def __str__(self):
        rep = "GRBLLH object\n"
        rep += "-------------\n\n"

        rep += "Spatial PDF settings:\n"
        for key, val in self.spatial_pdf_args.items():
            rep += "  - {:12s} : {}\n".format(key, val)

        rep += "\n"

        rep += "Time PDF settings:\n"
        for key, val in self.time_pdf_args.items():
            rep += "  - {:12s} : {}\n".format(key, val)

        rep += "\n"

        rep += "Energy PDF settings:\n"
        for key, val in self.energy_pdf_args.items():
            rep += "  - {:12s} : {}\n".format(key, val)

        rep += "\n"

        rep += "LLH settings:\n"
        for key, val in self.llh_args.items():
            rep += "  - {:12s} : {}\n".format(key, val)

        return rep
