import numpy as np
# import numpy.lib.recfunctions
import scipy.stats as scs
import scipy.interpolate as sci
from sklearn.utils import check_random_state

from anapymods3.general.misc import fill_dict_defaults

import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class LLH(object):
    """
    Interface for Likelihood functions used in PS analyses.

    Classes must implement functions ["lnllh", "lnllh_ratio"].
    """
    def __init__(self):
        self._DESCRIBES = ["llh"]
        print("Interface only. Describes functions: ", self._DESCRIBES)
        return

    @docs.get_summaryf("LLH.lnllh")
    @docs.get_sectionsf("LLH.lnllh", sections=["Parameters", "Returns"])
    @docs.dedent
    def lnllh(self, X, theta, args):
        r"""
        Return the natural logarithm ln-Likelihood (ln-LLH) value for a given
        set of data `X` and parameters `theta`.

        The ln-LLH of a parameter `theta` of a parametric probability model
        under the given data `X` is the product of all probability values of
        the given data under the model assumption:

        .. math::

          \ln P(X|\theta) = \ln \mathcal{L}(\theta|X)
                          = \ln \prod_{i=1}^N f(x_i|\theta)
                          = \sum_{i=1}^N \ln f(x_i|\theta)


        The most likely set of parameters `theta` can be found by finding the
        maximum of the ln-LLH function by variation of the parameter set.

        Parameters
        ----------
        X : record-array
            Fixed data set the LLH depends on. dtypes are ["name", type].
        theta : dict
            Parameter set {"par_name": value} to evaluate the ln-LLH at.
        args : dict
            Other fixed parameters {"par_name": value}, the LLH depents on.

        Returns
        -------
        lnllh : float
            Natural logarithm of the LLH for the given `X`, `theta` and `args`.
        """
        raise NotImplementedError("LLH is an interface.")

    @docs.get_summaryf("LLH.lnllh_ratio")
    @docs.get_sectionsf("LLH.lnllh_ratio", sections=["Parameters", "Returns"])
    @docs.dedent
    def lnllh_ratio(self, X, theta, args):
        r"""
        Return the natural logarithm of the ratio of Likelihoods under the null
        hypothesis and the alternative hypothesis.

        The ratio :math:`\Lambda` used here is defined as:

        .. math::

          \Lambda = -2\ln\left(\frac{\mathcal{L}_0}
                                     {\mathcal{L}_1}\right)
                  =  2\ln\left(\mathcal{L}_1 - \mathcal{L}_0\right)


        High values of :math:`\Lambda` indicate, that the null hypothesis is
        more unlikely contrary to the alternative.

        Parameters
        ----------
        X, theta, args
            See LLH.lnllh, Paramters

        Returns
        -------
        lnllh_ratio : float
            Natural logarithm of the LLH ratio for the given `X`, `theta` and
            `args`.
        """
        raise NotImplementedError("LLH is an interface.")


class GRBLLH(LLH):
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
        Global data set, used to derive per events PDFs. X must contain names:

        - "ra", "dec", float: Per event positions in equatorial coordinates,
          given in radians.
        - "logE", float: Per event energy proxy, given in log10(1/GeV).
        - "sigma", float: Per event positional uncertainty, given in radians.
          It is assumed, that a circle with radius `sigma` contains approximatly
          :math:`1\sigma` (\~39\%) of probability of the reconstrucion
          likelihood space.

    MC : record-array
        Global Monte Carlo data set, used to derive per event PDFs. MC must
        contin the same names as X and additionaly the MC truths:

        - "trueRa", "trueDec", float: Per event true positions in equatorial
          coordinates, given in radians.
        - "trueE", float: Per event true energy in GeV.
        - "ow", float: Per event "neutrino generator (NuGen)" OneWeight [2]_,
          already divided by `nevts * nfiles` known from SimProd.
          Units are "GeV sr cm^2". Final events weight are obtained by
          multiplying with desired Flux and data livetime.
    livetime : float
        Livetime in days of the data `X`. Used to properly weight the MC events.
    scramble : bool, optional
        If True, scramble expiremental data `X` in right-ascension for
        blindness. (default: True)
    random_state : seed, optional
        Turn seed into a np.random.RandomState instance. See
        `sklearn.utils.check_random_state`. (default: None)
    bg_spline_args : dict
        Arguments for the creation of the spatial background spline describing
        the sin_dec distribtuion. Must contain keys:

        - "bins", array-like: Explicit bin edges of the histogram created to fit
          a spline describing the spatial background PDF.
        - "k", int, optional: Degree of the smoothing spline. Must be
          1 <= k <= 5. (default: 3)
    signal_pdf_args : dict or None
        Arguments for the spatial signal PDF. Must contain keys:

        - "kent", bool, optional: If True, uses the Kent [3]_ distribution. A 2D
          gaussian PDF is used otherwise. (default: True)

    Notes
    -----
    .. [1] Barlow, "Statistics - A Guide to the Use of Statistical Methods in
           the Physical Sciences". Chap. 5.4, p. 90. Wiley (1989)
    .. [2] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
    .. [3] https://en.wikipedia.org/wiki/Kent_distribution
    """
    def __init__(self, X, MC, livetime, bg_spline_args, signal_pdf_args=None,
                 random_state=None):
        # Check if data and MC have all needed names
        X_names = ["ra", "dec", "logE", "sigma"]
        if not all([n in X.dtype.names for n in X_names]):
            raise ValueError("`X` has not all required names")
        MC_names = X_names + ["trueRa", "trueDec", "trueE", "ow"]
        if not all([n in MC.dtype.names for n in MC_names]):
            raise ValueError("`MC` has not all required names")

        # Setup spatial bg spline args
        required_keys = ["bins"]
        opt_keys = {"k": 3}
        self.bg_spline_args = fill_dict_defaults(bg_spline_args, required_keys,
                                                 opt_keys)

        # Setup spatial signal PDF args
        opt_keys = {"kent": True}
        self.signal_pdf_args = fill_dict_defaults(signal_pdf_args, [], opt_keys)

        # Add sin_dec field for usage in spatial PDF (code taken from skylab)
        # self.X = numpy.lib.recfunctions.append_fields(
        #     X, "sin_dec", np.sin(X["dec"]), dtypes=np.float, usemask=False)
        # self.MC = numpy.lib.recfunctions.append_fields(
        #     MC, "sin_dec", np.sin(MC["dec"]), dtypes=np.float, usemask=False)

        # Setup common variables
        self.rndgen = check_random_state(random_state)
        self.X = X
        self.MC = MC
        self.livetime = livetime
        self._nX = len(X)
        self._nMC = len(MC)
        self._SECINDAY = 24. * 60. * 60.

        # Create PDFs used in the LLH from global data and MC
        _bins = self.bg_spline_args["bins"]
        if np.any(_bins < -1) or np.any(_bins > 1):
            raise ValueError("Bins for BG spline not in sin_dec range [-1, 1].")
        sin_dec = np.sin(self.X["dec"])
        self._spatial_bg_spl = self._create_spatial_bg_spline(sin_dec)

        return

    # Public Methods
    @docs.dedent
    def lnllh(self, X, theta, args):
        """
        %(LLH.lnllh.summary)s

        Parameters
        ----------
        %(LLH.lnllh.parameters)s
            For GRBLLH, args must contain keys:

            - "nb", float: Expected number of background events.

        Returns
        -------
        %(LLH.lnllh.returns)s
        """
        nb = args["nb"]
        return nb

    @docs.dedent
    def lnllh_ratio(self, X, theta, args):
        """
        %(LLH.lnllh_ratio.summary)s

        Parameters
        ----------
        %(LLH.lnllh_ratio.parameters)s

        Returns
        -------
        %(LLH.lnllh_ratio.returns)s
        """

    # Private Methods
    # Time PDFs
    def _soverb_time(self, t, t0, dt, nsig=4.):
        """
        Time signal over background PDF.

        Signal and background PDFs are each normalized over seconds.
        Signal PDF has gaussian edges to smoothly let it fall of to zero, the
        stddev is dt when dt is in [2, 30]s, otherwise the nearest edge.

        To ensure finite support, the edges are truncated after nsig * dt.

        Parameters
        ----------
        t : array-like
            Times given in MJD for which we want to evaluate the ratio.
        t0 : float
            Time of the source event.
        dt : array-like, shape (2)
            Time window [start, end] in seconds centered at t0 in which the
            signal pdf is assumed to be uniform.
        nsig : float, optional
            Clip the gaussian edges at nsig * dt to have finite support.
            (default: 4.)
        """
        dt = np.atleast_1d(dt)
        if len(dt) != 2:
            raise ValueError("Timefram 'dt' must be [start, end] in seconds.")
        if dt[0] >= dt[1]:
            raise ValueError("Interval 'dt' must not be negative or zero.")

        # Normalize times from data relative to t0 in seconds
        # Stability: Multiply before subtracting avoids small number rounding?
        _t = t * self._SECINDAY - t0 * self._SECINDAY

        # Create signal PDF
        # Constrain sig_t to [2, 30]s regardless of uniform time window
        dt_tot = np.diff(dt)
        sig_t = np.clip(dt_tot, 2, 30)
        sig_t_clip = nsig * sig_t
        gaus_norm = (np.sqrt(2 * np.pi) * sig_t)

        # Split in def regions gaus rising, uniform, gaus falling
        gr = (_t < dt[0]) & (_t >= dt[0] - sig_t_clip)
        gf = (_t > dt[1]) & (_t <= dt[1] + sig_t_clip)
        uni = (_t >= dt[0]) & (_t <= dt[1])

        pdf = np.zeros_like(t, dtype=np.float)
        pdf[gr] = scs.norm.pdf(_t[gr], loc=dt[0], scale=sig_t)
        pdf[gf] = scs.norm.pdf(_t[gf], loc=dt[1], scale=sig_t)
        # Connect smoothly with the gaussians
        pdf[uni] = 1. / gaus_norm

        # Normalize signal distribtuion: Prob in gaussians + uniform part
        dcdf = (scs.norm.cdf(dt[1] + sig_t_clip, loc=dt[1], scale=sig_t) -
                scs.norm.cdf(dt[0] - sig_t_clip, loc=dt[0], scale=sig_t))
        norm = dcdf + dt_tot / gaus_norm
        pdf /= norm

        # Calculate the ratio
        bg_pdf = 1. / (dt_tot + 2 * sig_t_clip)
        ratio = pdf / bg_pdf
        return ratio

    # Spatial PDFs
    # def _sob_spatial(src_ra, src_dec, ev_ra, ev_dec, ev_sig, kent=True):
    #     S = spatial_signal(src_ra, src_dec, ev_ra, ev_dec, ev_sig, kent)
    #     B = spatial_background(ev_sin_dec, sindec_log_bg_spline)

    #     SoB = np.zeros_like(S)
    #     B = np.repeat(B[np.newaxis, :], repeats=S.shape[0], axis=0)
    #     m = B > 0
    #     SoB[m] = S[m] / B[m]

    #     return SoB

    def _pdf_spatial_background(self, ev_sin_dec):
        """
        Calculate the value of the background PDF for each event.

        PDF is uniform in right-ascension and described by a spline fitted to
        data in sinus declination.

        Parameters
        ----------
        ev_sin_dec : array-like, shape (nevts)
            Sinus declination coordinates of each event, in range [-1, 1].

        Returns
        -------
        B : array-like, shape (nevts)
            The value of the background PDF for each event.
        """
        return 1. / (2. * np.pi) * np.exp(self._spatial_bg_spl(ev_sin_dec))

    def _pdf_spatial_signal(self, src_ra, src_dec, ev_ra, ev_dec, ev_sig):
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
        src_ra, src_dec : array-like, shape (nsrc)
            Source positions in equatorial right-ascension, [0, 2pi] and
            declination, [-pi/2, pi/2], given in radian.
        ev_ra, ev_dec : array-like, shape (nevts)
            Event positions in equatorial right-ascension, [0, 2pi] and
            declination, [-pi/2, pi/2], given in radian.
        ev_sig : array-like, shape (nevts)
            Event positional reconstruction errors in radian (eg. Paraboloid).

        Returns
        --------
        S : array-like, shape(n_sources, n_events)
            Spatial signal probability for each event and each source.
        """
        # Shape (n_sources, 1), suitable for 1 src or multiple srcs
        src_ra = np.atleast_1d(src_ra)[:, np.newaxis]
        src_dec = np.atleast_1d(src_dec)[:, np.newaxis]

        # Dot product in polar coordinates, broadcasting applies here
        cosDist = (np.cos(src_ra - ev_ra) *
                   np.cos(src_dec) * np.cos(ev_dec) +
                   np.sin(src_dec) * np.sin(ev_dec))

        # Handle possible floating precision errors
        cosDist = np.clip(cosDist, -1, 1)

        if self.signal_pdf_args["kent"]:
            # Stabilized version for possibly large kappas
            kappa = 1. / ev_sig**2
            S = (kappa / (2. * np.pi * (1. - np.exp(-2. * kappa))) *
                 np.exp(kappa * (cosDist - 1.)))
        else:
            # Otherwise use standard symmetric 2D gaussian
            dist = np.arccos(cosDist)
            ev_sig_2 = 2 * ev_sig**2
            S = np.exp(-dist**2 / (ev_sig_2)) / (np.pi * ev_sig_2)

        return S

    def _create_spatial_bg_spline(self, sin_dec):
        """
        Fit an interpolating spline to the a histogram of sin(dec).

        The spline is fitted to the *natural logarithm* of the histogram, to
        avoid ringing. Normalization is done by normalizing the hist, so it may
        be slightly off, but that's tolerable.

        Fit parameters are controlled by the `self.bg_spline_args` dict.

        Parameters
        ----------
        sin_dec
            See _pdf_spatial_background, Parameters

        Returns
        -------
        _spatial_bg_spl : scipy.interpolate.InterpolatingSpline
            Spline object interpolating the histogram. Must be evaluated with
            sin(dec) and exponentiated to give the correct PDF values.
            Spline is interpolated outside it's definition range.
        """
        bins = self.bg_spline_args["bins"]
        k = self.bg_spline_args["k"]

        if np.any((sin_dec < bins[0]) | (sin_dec > bins[-1])):
            raise ValueError("Not all events fall into given bins range. If " +
                             "this is intended, please remove them beforehand.")

        # Make normalised hist to fit the spline to x, y pairs
        hist, bins = np.histogram(sin_dec, bins=bins, density=True)

        if np.any(hist <= 0.):
            raise ValueError("Got empty sin_dec hist bins, this must not " +
                             "happen. Empty bins idx:\n{}".format(
                                 np.arange(len(bins) - 1)[hist <= 0.]))

        mids = 0.5 * (bins[:-1] + bins[1:])
        return sci.InterpolatedUnivariateSpline(mids, np.log(hist), k=k, ext=0)






















