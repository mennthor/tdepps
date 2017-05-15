import numpy as np
import scipy.stats as scs
from sklearn.utils import check_random_state

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
    bg_spline_args : dict
    random_state : seed, optional
        Turn seed into a np.random.RandomState instance. See
        `sklearn.utils.check_random_state`. (default: None)

    Notes
    -----
    .. [1] Barlow, "Statistics - A Guide to the Use of Statistical Methods in
           the Physical Sciences". Chap. 5.4, p. 90. Wiley (1989)
    .. [2] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
    """
    def __init__(self, X, MC, livetime, scramble=True, random_state=None):
        # Check if data and MC have all needed names
        X_names = ["ra", "dec", "logE", "sigma"]
        if not all([n in X.dtype.names for n in X_names]):
            raise ValueError("`X` has not all required names")
        MC_names = X_names + ["trueRa", "trueDec", "trueE", "ow"]
        if not all([n in MC.dtype.names for n in MC_names]):
            raise ValueError("`MC` has not all required names")

        # This part is from skylab, add sinDec field for spatial PDF
        self.X = np.lib.recfunctions.append_fields(
            X, "sinDec", np.sin(X["dec"]), dtypes=np.float, usemask=False)
        self.MC = np.lib.recfunctions.append_fields(
            MC, "sinDec", np.sin(MC["dec"]), dtypes=np.float, usemask=False)

        self.livetime = livetime
        self.rndgen = check_random_state(random_state)

        self._nX = len(X)
        self._nMC = len(MC)
        self._SECINDAY = 24. * 60. * 60.

        # Scramble data if not unblinded. Do this after seed has been set
        if scramble:
            self.X["ra"] = self.rndgen.uniform(0., 2. * np.pi, self._nX)
        else:
            print("\t####################################\n"
                  "\t# Working on >> UNBLINDED << data! #\n"
                  "\t####################################\n")

        # Create PDFs used in the LLH from global data and MC

        return

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
    def _soverb_time(self, t, t0, dt, nsig):
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
        nsig : float
            Clip the gaussian edges at nsig * dt
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

    # Signal PDFs
    def spatial_SoB(src_ra, src_dec, ev_ra, ev_dec, ev_sig,
                    sindec_log_bg_spline, kent=True):
        S = spatial_signal(src_ra, src_dec, ev_ra, ev_dec, ev_sig, kent)
        B = spatial_background(ev_sin_dec, sindec_log_bg_spline)

        SoB = np.zeros_like(S)
        B = np.repeat(B[np.newaxis, :], repeats=S.shape[0], axis=0)
        m = B > 0
        SoB[m] = S[m] / B[m]

        return SoB

    def spatial_background(ev_sin_dec, sindec_log_bg_spline):
        """
        Calculate the value of the backgournd PDF for each event from a previously
        created spline, interpolating the declination distribution of the data.

        Parameters
        ----------
        ev_sin_dec : array-like
            Sinus Declination coordinates of each event, [-1, 1].
        sindec_log_bg_spline : scipy.interpolate.InterpolatingSpline
            Spline returning the logarithm of the bg PDF at given sin_dec values.

        Returns
        -------
        B : array-like
            The value of the background PDF for each event.
        """
        return 1. / 2. / np.pi * np.exp(sindec_log_bg_spline(ev_sin_dec))

    def spatial_signal(self, src_ra, src_dec, ev_ra, ev_dec, ev_sig, kent=True):
        """
        Spatial distance PDF between source position(s) and event positions.

        Signal is assumed to cluster around source position(s).
        The PDF is a convolution of a delta function for the localized sources
        and a Kent (gaussian on a sphere) distribution with the events
        positional reconstruction error as width.

        Multiplie source positions can be given, to use it in a stacked
        search.

        Parameters
        -----------
        src_ra : array-like
            Src positions in equatorial RA in radian: [0, 2pi].
        src_dec : array-like
            Src positions in equatorial DEC in radian: [-pi/2, pi/2].
        ev_ra : array-like
            Event positions in equatorial RA in radian: [0, 2pi].
        ev_dec : array-like
            Event positions in equatorial DEC in radian: [-pi/2, pi/2].
        ev_sig : array-like
            Event positional reconstruction error in radian (eg. Paraboloid).

        Returns
        --------
        S : array-like, shape(n_sources, n_events)
            Spatial signal probability for each event and each source.

        """
        # Shape (n_sources, 1), suitable for 1 src or multiple srcs
        src_ra = np.atleast_1d(src_ra)[:, np.newaxis]
        src_dec = np.atleast_1d(src_dec)[:, np.newaxis]

        # Dot product in polar coordinates
        cosDist = (np.cos(src_ra - ev_ra) *
                   np.cos(src_dec) * np.cos(ev_dec) +
                   np.sin(src_dec) * np.sin(ev_dec))

        # Handle possible floating precision errors
        cosDist = np.clip(cosDist, -1, 1)

        if kent:
            # Stabilized version for possibly large kappas
            kappa = 1. / ev_sig**2
            S = (kappa / (2. * np.pi * (1. - np.exp(-2. * kappa))) *
                 np.exp(kappa * (cosDist - 1. )))
        else:
            # Otherwise use standard symmetric 2D gaussian
            dist = np.arccos(cosDist)
            ev_sig_2 = 2 * ev_sig**2
            S = np.exp(-dist**2 / (ev_sig_2)) / (np.pi * ev_sig_2)

        return S

    def _create_spatial_bg_spline(sinDec, bins=100, range=None, k=3):
        """
        Fit an interpolating spline to the a histogram of sin(dec).

        The spline is fitted to the logarithm of the histogram, to avoid
        ringing. Normalization is done by normalizing the hist, so it may be
        slightly off, but that's tolerable.

        Parameters
        ----------
        sinDec : array-like
            Sinus declination coorcinates of each event, [-1, 1].
        bins : int or array-like
            Binning passed to `np.histogram`. (default: 100)
        range : array-like
            Lower and upper boundary for the histogram. (default: None)
        k : int
            Order of the spline. (default: 3)

        Returns
        -------
        spl : scipy.interpolate.InterpolatingSpline
            Spline object interpolating the histogram. Must be evaluated with
            sin(dec) and exponentiated to give the correct values.
            Spline is interpolating outside it's definition range.
        """
        hist, bins = np.histogram(sinDec, bins=bins,
                                  range=range, density=True)

        if np.any(hist <= 0.):
            estr = ("Declination hist bins empty, this must not happen. Empty " +
                    "bins: {0}".format(np.arange(len(bins) - 1)[hist <= 0.]))
            raise ValueError(estr)
        elif np.any((sinDec < bins[0]) | (sinDec > bins[-1])):
            raise ValueError("Data outside of declination bins!")

        mids = 0.5 * (bins[:-1] + bins[1:])
        return sci.InterpolatedUnivariateSpline(mids, np.log(hist), k=k, ext=0)






















