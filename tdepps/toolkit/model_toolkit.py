# coding: utf-8

"""
This is a collection of different function and / or classes that can be used in
a modular way to create a PDF and injection model class which can be used in the
LLH analysis.
When creating a new function it should only work on public interfaces and data
provided by the model class. If this can't be realized due to performance /
caching reasons, then consider coding the functionality directly into a model to
benefit from direct class attributes.
"""

from __future__ import print_function, division, absolute_import

import abc
import numpy as np
from numpy.lib.recfunctions import drop_fields
import scipy.stats as scs
import healpy as hp
from .model_base import (BaseSignalInjector, BaseBGDataInjector,
                         BaseRateFunction, BaseTimeSampler)
from .model_base import BaseMultiBGDataInjector, BaseMultiSignalInjector
from ..utils import (random_choice, fill_dict_defaults, thetaphi2decra, logger,
                     rotator, fit_spl_to_hist, make_time_dep_dec_splines,
                     arr2str)

try:
    from awkde import GaussianKDE as KDE
except ImportError as e:
    log = logger(name="toolkit", level="ALL")
    print(log.ERROR(e))
    print(log.ERROR("KDE injector is not available, because package " +
                    "`awkde` is missing."))


##############################################################################
# Signal injector classes
##############################################################################
class SignalFluenceInjector(BaseSignalInjector):
    """
    Signal Fluence Injector

    Inject signal events from Monte Carlo data weighted to a specific fluence
    model and inject at given source positions. Fluence is assumed to be per
    "burst" as in GRB models, so the fluence is not depending on the duration
    of a source's time window. Time sampling is done via a ``TimeSampler``.

    Parameters
    ----------
    model : callable
        Function of true energy ``f(MC['trueE'])``, describing the model flux
        that will be injected from. Values are used to weight events to a
        physics scenario by ``w[i] ~ f(MC['trueE'][i] * MC['ow'][i]``. Make
        sure the model fits to the units of the true energy and the OneWeights.
    time_sampler : TimeSampler instance
        Time sampler for sampling the signal times in addition to the spatial
        rotation part done by the injector.
    mode : string, optional
        One of ``['circle'|'band']``. Selects MC events to inject based
        on their true location:

        - 'circle': Select ``MC`` events in circle around each source.
        - 'band': Select ``MC`` events in a declination band around each source.

        (default: 'band')

    sindec_inj_width : float, optional
        Angular size of the regions from which MC events are injected, in
        ``sin(dec)`` to preserve solid angle sizes near the poles.

        - If ``mode`` is ``'band'``, this is half the width of the sinues
          declination band centered at the source positions.
        - If ``mode`` is ``'circle'`` this is the radius of the circular
          selection region in sinues declination.

        (default: 0.035 ~ 2°)

    dec_range : array-like, shape (2), optional
        Global declination interval in which events can be injected in rotated
        coordinates. Events rotated outside are dropped even if selected, which
        drops sensitivity as desired. (default: ``[-pi/2, pi/2]``)
    random_state : None, int or np.random.RandomState, optional
        Used as PRNG, see ``sklearn.utils.check_random_state``. (default: None)
    """
    def __init__(self, flux_model, time_sampler, mode="band",
                 sindec_inj_width=0.035, dec_range=None, random_state=None):
        try:
            flux_model(1.)  # Standard units are E = 1GeV
        except Exception:
            raise TypeError("`model` must be a function `f(trueE)`.")
        self._flux_model = flux_model

        if not isinstance(time_sampler, BaseTimeSampler):
            raise ValueError("`time_sampler` must have type `BaseTimeSampler`.")
        self._time_sampler = time_sampler

        if mode not in ["band", "circle"]:
            raise ValueError("`mode` must be one of ['band', 'circle']")
        self._mode = mode

        if (sindec_inj_width <= 0.) or (sindec_inj_width > np.pi):
            raise ValueError("Injection width must be in (0, pi].")
        self._sindec_inj_width = sindec_inj_width

        if dec_range is None:
            dec_range = np.array([-np.pi / 2., np.pi / 2.])
        else:
            dec_range = np.atleast_1d(dec_range)
            if (dec_range[0] < -np.pi / 2.) or (dec_range[1] > np.pi / 2.):
                raise ValueError("`dec_range` must be in range [-pi/2, pi/2].")
            if dec_range[0] >= dec_range[1]:
                raise ValueError("`dec_range=[low, high]` must be increasing.")
        self._dec_range = dec_range

        self.rndgen = random_state
        self._mc_arr = None
        self._srcs = None

        # Defaults for private class attributes
        self._MC = None
        self._mc_names = None
        self._exp_names = None
        self._src_dt = None

        self._min_dec = None
        self._max_dec = None
        self._omega = None

        self._raw_flux = None
        self._sample_w_CDF = None
        self._w_theo_norm = None

        # Debug attributes
        self._sample_idx = None
        self._skylab_band = False

        self._log = logger(name=self.__class__.__name__, level="ALL")

        return

    @property
    def flux_model(self):
        return self._flux_model

    # No setters, use the `fit` method for that or create a new object
    @property
    def provided_data(self):
        return self._exp_names

    @property
    def mode(self):
        return self._mode

    @property
    def sindec_inj_width(self):
        return self._sindec_inj_width

    @property
    def dec_range(self):
        return self._dec_range

    @property
    def mc_arr(self):
        return self._mc_arr

    def mu2flux(self, mu, per_source=False):
        """
        Convert a given number of events ``mu`` to a corresponding particle flux
        normalization :math:`F_0` in units [GeV^-1 cm^-2].

        The connection between :math:`F_0` and the number of events ``mu`` is:

        .. math:: F_0 = \mu / \sum_i \hat{w}_i

        where :math:`F_0 \sum_i \hat{w}_i` would give the number of events. The
        weights :math:`w_i` are calculated in :py:meth:`_set_sampling_weights`.

        Parameters
        ----------
        mu : float
            Expectation for number of events at the detector.
        per_source : bool, optional
            If ``True`` returns the expected events per source by splitting the
            total flux accroding to the sources' theoretical weights.
            (Default: ``False``)

        Returns
        -------
        flux : float or array-like
            Source flux in total, or split for each source, in unit(s)
            ``[GeV^-1 cm^-2]``.
        """
        if per_source:
            # Split the total mu according to the theoretical source weights
            mu = self._w_theo_norm * mu
        return mu / self._raw_flux

    def flux2mu(self, flux, per_source=False):
        """
        Calculates the number of events ``mu`` corresponding to a given particle
        flux for the current setup:

        .. math:: \mu = F_0 \sum_i \hat{w}_i

        where :math:`F_0 \sum_i \hat{w}_i` would gives the number of events. The
        weights :math:`w_i` are calculated in :py:meth:`_set_sampling_weights`.

        Parameters
        ----------
        flux : float
            Total source flux for all sources, in units ``[GeV^-1 cm^-2]``.
        per_source : bool, optional
            If ``True`` returns the flux per source by splitting the total flux
            according to the expected events per source at detector level.
            (Default: ``False``)

        Returns
        -------
        mu : float or array-like
            Expected number of event in total or per source at the detector.
        """
        if per_source:
            # Split the total mu according to the theoretical source weights
            mu = flux * self._raw_flux_per_src
        else:
            mu = flux * self._raw_flux
        return mu

    def fit(self, srcs, MC, exp_names):
        """
        Fill injector with Monte Carlo events, preselecting events around the
        source positions.

        Parameters
        -----------
        srcs : recarray
            Source properties in a record array, must have names:

            - 'ra', float: Right-ascension coordinate of each source in
              radian in intervall :math:`[0, 2\pi]`.
            - 'dec', float: Declinatiom coordinate of each source in radian
              in intervall :math:`[-\pi / 2, \pi / 2]`.
            - 't', float: Time of the occurence of the source event in MJD
              days.
            - 'dt0', 'dt1': float: Lower/upper border of the time search
              window in seconds, centered around each source time ``t``.
            - 'w_theo', float: Theoretical source weight per source, eg. from
              a known gamma flux.

        MC : recarray
            Structured array describing Monte Carlo events, must contain the
            same names as given in ``exp_names`` and additonally MC truths:

            - 'trueE', float: True event energy in GeV.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator' OneWeight [2]_,
              so it is already divided by ``nevts * nfiles`.
              Units are 'GeV sr cm^2'. Final event weights are obtained by
              multiplying with desired sum flux for nu and anti-nu flux.

        exp_names : list
            Names that must be present in ``MC`` to match names for experimental
            data record array. Must have at least names ``['ra', 'dec']`` needed
            to rotate injected signal events to the source positions. Names
            that are present in ``MC`` but not in ``exp_names`` are not used and
            removed from the sample output. Names other than ``['ra', 'dec']``
            are not treated any further, but only piped through. So if there is
            e.g. ``sinDec`` this must be converted manually after receiving the
            sampled data.

        Notes
        -----
        .. [1] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
        """
        self._srcs, MC, exp_names = self._check_fit_input(srcs, MC, exp_names)
        nsrcs = len(self._srcs)

        # Set injection solid angles for all sources
        self._set_solid_angle()

        # Select events in the injection regions, masking all srcs at once
        if self._mode == "band":
            min_decs = self._min_dec.reshape(nsrcs, 1)
            max_decs = self._max_dec.reshape(nsrcs, 1)
            mc_true_dec = MC["trueDec"]
            inj_mask = ((mc_true_dec > min_decs) & (mc_true_dec < max_decs))
        else:
            # Compare great circle distance to each src with inj_width
            src_ra = np.atleast_2d(srcs["ra"]).reshape(nsrcs, 1)
            src_dec = np.atleast_2d(srcs["dec"]).reshape(nsrcs, 1)
            mc_true_ra = MC["trueRa"]
            mc_true_dec = MC["trueDec"]
            cos_dist = (np.cos(src_ra - mc_true_ra) *
                        np.cos(src_dec) * np.cos(mc_true_dec) +
                        np.sin(src_dec) * np.sin(mc_true_dec))
            cos_r = np.cos(np.arcsin(self._sindec_inj_width))
            inj_mask = (cos_dist > cos_r)

        if not np.any(inj_mask):
            # If no events were selected we don't need this src-MC combination
            raise RuntimeError("No events selected. Check `srcs` and `MC`.")

        # Select unqique MC in all injection regions (include overlap)
        total_mask = np.any(inj_mask, axis=0)
        N_unique = np.count_nonzero(total_mask)
        # Only keep needed MC names (MC truth and those specified by exp_names)
        keep_names = self._mc_names + self._exp_names
        drop_names = [n for n in MC.dtype.names if n not in keep_names]
        self._MC = drop_fields(MC[total_mask], drop_names)
        assert len(self._MC) == N_unique

        # Total number of all selected events, including multi counts
        core_mask = (inj_mask.T[total_mask]).T  # Remove all non-selected
        n_tot = np.count_nonzero(core_mask)     # Equal to count inj_mask

        # Only store selected event IDs in mc_arr to sample from the unique MC
        # pool to save memory.
        # ev_idx: index of evt in _MC; src_idx: index in _srcs for each event
        dtype = [("ev_idx", np.int), ("src_idx", np.int)]
        self._mc_arr = np.empty(n_tot, dtype=dtype)

        _core = core_mask.ravel()  # [src1_mask, src2_mask, ...]
        self._mc_arr["ev_idx"] = np.tile(np.arange(N_unique), nsrcs)[_core]
        # Same src IDs for each selected evt per src (rows in core_mask)
        self._mc_arr['src_idx'] = np.repeat(np.arange(nsrcs),
                                            np.sum(core_mask, axis=1))

        s_ = "Selected {:d} evts at {:d} sources.".format(n_tot, nsrcs) + "\n"
        s_ += len(self._log.INFO()) * " "
        s_ += "- Sources without selected evts: {}".format(
            nsrcs - np.count_nonzero(np.sum(core_mask, axis=1)))
        print(self._log.INFO(s_))

        self._set_sampling_weights()

    def sample(self, n_samples=1):
        """
        Get sampled events from stored MC for each  stored source position.

        Parameters
        -----------
        n_samples : int, optional
            Number of signal events to sample. (Default. 1)

        Returns
        --------
        sam_ev : record-array
            Sampled events from the stored MC pool. Number of events sampled in
            total might be smaller than ``n_samples`` when a ``dec_width`` less
            than the whole sky is used. If ``n_samples<1`` an empty recarray is
            returned.
        """
        if self._mc_arr is None:
            raise ValueError("Injector has not been filled with MC data yet.")

        if n_samples < 1:
            dtype = [(name, float) for name in self._exp_names]
            return np.empty(0, dtype=dtype)

        # Draw IDs from the whole stored pool of MC events
        sam_idx = random_choice(self._rndgen, self._sample_w_CDF, n=n_samples)
        sam_idx = self._mc_arr[sam_idx]

        # Select events from pool and rotate them to corresponding src positions
        sam_ev = self._MC[sam_idx["ev_idx"]]
        src_idx = sam_idx["src_idx"]
        sam_ev = self._rot_and_strip(self._srcs["ra"][src_idx],
                                     self._srcs["dec"][src_idx],
                                     sam_ev)

        # Sample times from time model
        sam_ev["timeMJD"] = self._time_sampler.sample(
            src_t=self._srcs["t"][src_idx], src_dt=self._src_dt[src_idx])

        # Debug purpose
        self._sample_idx = sam_idx
        return sam_ev

    def _set_sampling_weights(self):
        """
        Setup per event sampling weights from the OneWeights.

        Physics weights are calculated for a simple unbroken power law particle
        flux (per particle type) differential in energy and detection area:

        .. math:: dN/(dE dA) = F_0 (E / GeV)^{-\gamma}

        with the normalization :math:`F_0` at 1 GeV in units ``[GeV^-1 cm^-2]``.
        The flux is not differential in time because for GRBs we time integrate.

        Because we inject only from a fraction of the sky from the diffuse MC
        per GRB (band or circle) the per event physics weight are calculated
        using:

        .. math::

          w_i = [\text{ow}]_i \times \left.\frac{dF}{dE}\right|_{E_i} \times
                \frac{w_\text{src}}{\Omega_\text{src}}

        where ``Omega_src``/``w_src`` is the injected solid angle/intrinsic
        weight for the GRB the event :math:`i` is injected at and ``ow`` is the
        NuGen OneWeight per type already divided by
        ``nfiles * nevents * type_weight``.

        We then get the number of expected events n as

        .. math:: n = \sum_i w_i = F_0 \sum_i \hat{w}_i

        where the free to choose normalization :math:`F_0` is explicitly written
        in the last step. See :py:meth:`mu2flux` which calculates the
        fluence from a given number of events from that relation.
        """
        mc = self._MC[self._mc_arr["ev_idx"]]
        src_idx = self._mc_arr["src_idx"]

        # Normalize w_theos to prevent overestimatiing injected fluxes
        self._w_theo_norm = self._srcs["w_theo"] / np.sum(self._srcs["w_theo"])
        assert np.isclose(np.sum(self._w_theo_norm), 1.)

        # Broadcast solid angles and w_theo to corrsponding sources for each evt
        omega = self._omega[src_idx]
        w_theo = self._w_theo_norm[src_idx]
        flux = self._flux_model(mc["trueE"])
        assert len(omega) == len(w_theo) == len(flux)

        w = mc["ow"] * flux / omega * w_theo
        assert len(self._mc_arr) == len(w)

        self._raw_flux_per_src = np.array(
            [np.sum(w[src_idx == j]) for j in np.unique(src_idx)])
        self._raw_flux = np.sum(w)
        assert np.isclose(np.sum(self._raw_flux_per_src), self._raw_flux)

        # Cache sampling CDF used for injecting events from the whole MC pool
        self._sample_w_CDF = np.cumsum(w) / self._raw_flux
        assert np.isclose(self._sample_w_CDF[-1], 1.)

    def _set_solid_angle(self):
        """
        Setup solid angles of injection area for selected MC events and sources.

        For a given set of source positions and an injection mode, we need to
        calculate the solid angle per source from which events where injected
        to be able to correctly weight the injected events to get a flux.

        Sets up private class variables:

        - ``_omega``, dict of arrays: Solid angle in radians of each injection
          region for each source sample.
        - ``_min_dec``, ``_max_dec``, dict of arrays: Upper/lower bounds for
          each declination band in radians for each source sample. Optional,
          only set if `self._mode` is 'band'.
        """
        assert self._mode in ["band", "circle"]
        nsrcs = len(self._srcs)

        if self._mode == "band":
            sinL, sinU = np.sin(self._dec_range)

            if self._skylab_band:
                # Recenter sources somewhat so that bands get bigger at poles
                m = (sinL - sinU + 2. * self._sindec_inj_width) / (sinL - sinU)
                sinU = self._sindec_inj_width * (sinL + sinU) / (sinU - sinL)
                sin_dec = m * np.sin(self._srcs["dec"]) + sinU
            else:
                sin_dec = np.sin(self._srcs["dec"])

            min_sin_dec = np.maximum(sinL, sin_dec - self._sindec_inj_width)
            max_sin_dec = np.minimum(sinU, sin_dec + self._sindec_inj_width)

            self._min_dec = np.arcsin(np.clip(min_sin_dec, -1., 1.))
            self._max_dec = np.arcsin(np.clip(max_sin_dec, -1., 1.))

            # Solid angles of selected events around each source
            self._omega = 2. * np.pi * (max_sin_dec - min_sin_dec)
            assert (len(self._min_dec) == len(self._max_dec) == nsrcs)
        else:
            r = self._inj_width
            self._omega = np.array(nsrcs * [2 * np.pi * (1. - np.cos(r))])
            assert len(self._omega) == nsrcs
        assert np.all((0. < self._omega) & (self._omega <= 4. * np.pi))

    def _rot_and_strip(self, src_ras, src_decs, MC):
        """
        Rotate injected event positions to the sources and strip Monte Carlo
        information from the output array to match exp data array.

        The rotation angles to move the true directions to the sources are used
        to rotate the measured positions by the same amount. Events rotated
        outside ``dec_range`` are removed from the sample.

        Parameters
        ----------
        src_ras, src_decs : array-like, shape (len(MC))
            Sources equatorial positions in right-ascension in ``[0, 2pi]``
            and declination in ``[-pi/2, pi/2]``, both given in radians.
            These are the true coordinates we rotate ``MC`` events to.
        MC : record array
            Structured array with injected  selection from MC pool.

        Returns
        --------
        ev : structured array
            Array with rotated values and removed true ``MC`` information.
        """
        # True positions are used to build the rotation matrix. Then the
        # 'measured' positions are actually rotated
        MC["ra"], MC["dec"] = rotator(MC["trueRa"], MC["trueDec"],  # From
                                      src_ras, src_decs,            # To
                                      MC["ra"], MC["dec"])          # Rotate

        # Remove events that got rotated outside the sin_dec_range
        m = ((MC["dec"] >= self._dec_range[0]) &
             (MC["dec"] <= self._dec_range[1]))
        return drop_fields(MC[m], self._mc_names)

    def _check_fit_input(self, srcs, MC, exp_names):
        """ Check fit input, setup self._exp_names, self._mc_names """
        # Check if each recarray has it's required names
        for n in ["t", "dt0", "dt1", "ra", "dec", "w_theo"]:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` array is missing name '{}'.".format(n))
        self._src_dt = np.vstack((srcs["dt0"], srcs["dt1"])).T

        self._exp_names = list(exp_names)
        for n in ["ra", "dec"]:
            if n not in exp_names:
                raise ValueError("`exp_names` is missing name '{}'.".format(n))

        self._mc_names = ["trueRa", "trueDec", "trueE", "ow"]
        MC_names = self._exp_names + self._mc_names
        for n in MC_names:
            if n not in MC.dtype.names:
                raise ValueError("`MC` array is missing name '{}'.".format(n))

        # Check if sources are outside `dec_range`
        if (np.any(srcs["dec"] < self._dec_range[0]) or
                np.any(srcs["dec"] > self._dec_range[1])):
            raise ValueError("Source position(s) found outside `dec_range`.")
        return srcs, MC, exp_names


class HealpySignalFluenceInjector(SignalFluenceInjector):
    """
    Healpy Signal Injector

    Inject signal events not at a fixed source position but according to a
    healpy prior map.
    If fixed source positions are tested in the analysis, this injection should
    decrease sensitivity systematically.
    Injection mode is constrained to ``band`` here.

    Parameters
    ----------
    model : callable
        Function of true energy ``f(MC['trueE'])``, describing the model flux
        that will be injected from. Values are used to weight events to a
        physics scenario by ``w[i] ~ f(MC['trueE'][i] * MC['ow'][i]``. Make
        sure the model fits to the units of the true energy and the OneWeights.
    time_sampler : TimeSampler instance
        Time sampler for sampling the signal times in addition to the spatial
        rotation part done by the injector.
    sindec_inj_width : float, optional
        This is different form the ``SignalFluenceInjector`` behaviour because
        depending on the prior localization we may select a different bandwidth
        for each source. Here, ``sindec_inj_width`` is the minimum selection
        bandwidth, preventing the band to become too small for very narrow
        priors. It is therefore used in combination with the new attribute
        ``inj_sigma``. (default: 0.035 ~ 2°)
    inj_sigma : float, optional
        Angular size in sigma around each source region from the prior map from
        which MC events are injected. Use in combination with
        ``sindec_inj_width`` to make sure, the injection band is wide enough.
        (default: 3.)
    dec_range : array-like, shape (2), optional
        Global declination interval in which events can be injected in rotated
        coordinates. Events rotated outside are dropped even if selected, which
        drops sensitivity as desired. (default: ``[-pi/2, pi/2]``)
    random_state : None, int or np.random.RandomState, optional
        Used as PRNG, see ``sklearn.utils.check_random_state``. (default: None)
    """
    def __init__(self, model, time_sampler, sindec_inj_width=0.035,
                 inj_sigma=3., dec_range=None, random_state=None):
        if inj_sigma <= 0.:
            raise ValueError("Injection sigma must be >0.")
        self._inj_sigma = inj_sigma

        # Defaults for private class attributes
        self._src_map_CDFs = None
        self._NSIDE = None
        self._NPIX = None
        self._pix2ra = None
        self._pix2dec = None

        # Debug attributes
        self._src_ra = None
        self._src_dec = None

        return super(HealpySignalFluenceInjector, self).__init__(
            model, time_sampler, "band", sindec_inj_width, dec_range,
            random_state)

    @property
    def inj_sigma(self):
        return self._inj_sigma

    def fit(self, srcs, src_maps, MC, exp_names):
        """
        Fill injector with Monte Carlo events, preselecting events in regions
        in the prior maps.

        Parameters
        -----------
        srcs : recarray
            Source properties in a record array, must have names:

            - 'ra', float: Right-ascension coordinate of each source in
              radian in intervall :math:`[0, 2\pi]`.
            - 'dec', float: Declinatiom coordinate of each source in radian
              in intervall :math:`[-\pi / 2, \pi / 2]`.
            - 't', float: Time of the occurence of the source event in MJD
              days.
            - 'dt0', 'dt1': float: Lower/upper border of the time search
              window in seconds, centered around each source time ``t``.
            - 'w_theo', float: Theoretical source weight per source, eg. from
              a known gamma flux.

        src_maps : array-like, shape (nsrcs, NPIX)
            List of valid healpy map arrays per source, all in same resolution
            used as spatial priors for injecting source positions each sample
            step. Maps must be normal space PDFs normalized to area equals one
            on the unit sphere from a positional reconstruction to give a
            probability region of the true source position per source.
        MC : recarray
            Structured array describing Monte Carlo events, must contain the
            same names as given in ``exp_names`` and additonally MC truths:

            - 'trueE', float: True event energy in GeV.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator' OneWeight [2]_,
              so it is already divided by ``nevts * nfiles`.
              Units are 'GeV sr cm^2'. Final event weights are obtained by
              multiplying with desired sum flux for nu and anti-nu flux.

        exp_names : list
            Names that must be present in ``MC`` to match names for experimental
            data record array. Must have at least names ``['ra', 'dec']`` needed
            to rotate injected signal events to the source positions. Names
            that are present in ``MC`` but not in ``exp_names`` are not used and
            removed from the sample output. Names other than ``['ra', 'dec']``
            are not treated any further, but only piped through. So if there is
            e.g. ``sinDec`` this must be converted manually after receiving the
            sampled data.

        Notes
        -----
        .. [1] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
        """
        srcs, MC, exp_names = self._check_fit_input(srcs, MC, exp_names)
        nsrcs = len(srcs)

        # Check if maps match srcs
        if hp.maptype(src_maps) != nsrcs:
                raise ValueError("Not as many prior maps as srcs given.")

        self._NPIX = int(len(src_maps[0]))
        self._NSIDE = hp.npix2nside(self._NPIX)

        # Test if maps are valid PDFs on the unit sphere (m>=0 and sum(m*dA)=1)
        dA = hp.nside2pixarea(self._NSIDE)
        areas = np.array(map(np.sum, src_maps)) * dA
        if not np.allclose(areas, 1.) or np.any(src_maps < 0.):
            raise ValueError("Not all given maps for key are valid PDFs.")

        # Select the injection band depending on the given source prior maps
        decL, decU = self._dec_range
        sinL, sinU = np.sin(self._dec_range)
        # Get band [min dec, max dec] from n sigma contour of the prior maps
        for i, map_i in enumerate(src_maps):
            min_dec, max_dec = self.get_nsigma_dec_band(map_i, self._inj_sigma)
            self._min_dec.append(min_dec)
            self._max_dec.append(max_dec)
            if (min_dec <= srcs["dec"][i]) or (srcs["dec"][i] <= max_dec):
                raise ValueError("Source {} not within {} sigma band ".format(
                    i, self._inj_sigma) + "of corresponding prior map.")

        # Enlarge bands if they got too narrow
        sin_dec = np.sin(srcs["dec"])
        min_sin_dec = np.minimum(np.sin(self._min_dec),
                                 sin_dec - self._sindec_inj_width)
        max_sin_dec = np.maximum(np.sin(self._max_dec),
                                 sin_dec + self._sindec_inj_width)
        # Clip if we ran over the set dec range
        self._min_dec = np.arcsin(np.clip(min_sin_dec, sinL, sinU))
        self._max_dec = np.arcsin(np.clip(max_sin_dec, sinL, sinU))
        assert not np.any(self._max_dec < srcs["dec"])
        assert not np.any(srcs["dec"] < self._min_dec)

        # Pre-compute normalized sampling CDFs from the maps for fast sampling
        CDFs = np.cumsum(src_maps, axis=1)
        self._src_map_CDFs = CDFs / CDFs[:, [-1]]
        assert np.allclose(self._src_map_CDFs[:, -1], 1.)
        assert len(self._src_map_CDFs) == self._NPIX

        # Prepare other fixed elements to save time during sampling
        # Precompute pix2ang conversion, directly in ra, dec
        th, phi = hp.pix2ang(self._NSIDE, np.arange(self._NPIX))
        self._pix2dec, self._pix2ra = thetaphi2decra(th, phi)

        return super(HealpySignalFluenceInjector, self).fit(srcs, MC, exp_names)

    @staticmethod
    def get_nsigma_dec_band(pdf_map, sigma=3.):
        """
        Get the ns sigma declination band around a source position from a
        prior healpy normal space PDF map.

        Parameters
        ----------
        pdf_map : array-like
            Healpy PDF map on the unit sphere.
        sigma : int, optional
            How many sigmas the band should measure. Wilk's theorem is assumed
            to calculate the sigmas.

        Returns
        -------
        min_dec, max_dec : float
            Lower / upper border of the n sigma declination band, in radian.
        """
        # Get n sigma level from Wilk's theorem
        level = np.amax(pdf_map) * scs.chi2.sf(sigma**2, df=2)
        # Select pixels inside contour
        m = (pdf_map >= level)
        pix = np.arange(len(pdf_map))[m]
        # Convert to ra, dec and get min(dec), max(dec) for the band
        NSIDE = hp.get_nside(pdf_map)
        th, phi = hp.pix2ang(NSIDE, pix)
        dec, _ = thetaphi2decra(th, phi)
        return np.amin(dec), np.amax(dec)

    def sample(self, n_samples=1):
        """
        Generator to get sampled events from MC for each source position.
        Each time new source positions are sampled from the prior maps.
        Only performs spatial rotation to the source positions, no time sampling
        is done.

        Parameters
        -----------
        n_samples : n_samples
            How many events to sample.

        Returns
        --------
        sam_ev : record-array
            Sampled events from the stored MC pool. Number of events sampled in
            total might be smaller than ``n_samples`` when a ``dec_width`` less
            than the whole sky is used. If ``n_samples<1`` an empty recarray is
            returned.
        """
        if self._mc_arr is None:
            raise ValueError("Injector has not been filled with MC data yet.")

        if n_samples < 1:
            dtype = [(name, float) for name in self._exp_names]
            return np.empty(0, dtype=dtype)

        # Sample new source positions from prior maps
        src_idx = np.empty(len(self._srcs), dtype=int)
        for i, CDFi in enumerate(self._src_map_CDFs):
            src_idx[i] = random_choice(self._rndgen, CDF=CDFi, n=None)
        src_ras, src_decs = self._pix2ra[src_idx], self._pix2dec[src_idx]

        # Draw IDs from the whole stored pool of MC events
        sam_idx = random_choice(self._rndgen, self._sample_w_CDF, n=n_samples)
        sam_idx = self._mc_arr[sam_idx]

        # Select events from pool and rotate them to corresponding src positions
        sam_ev = self._MC[sam_idx["ev_idx"]]
        src_idx = sam_idx["src_idx"]
        sam_ev = self._rot_and_strip(src_ras[src_idx], src_decs[src_idx],
                                     sam_ev)

        # Sample times from time model
        sam_ev["timeMJD"] = self._time_sampler.sample(
            src_t=self._srcs["t"][src_idx], src_dt=self._src_dt[src_idx])

        # Debug purpose
        self._sample_idx = sam_idx
        self._src_ra, self._src_dec = src_ras, src_decs
        return sam_ev

    def _set_solid_angle(self):
        """
        Setup solid angles of injection area for selected MC events and sources.

        Only sets up the solid angle here to not break inheritance, band
        boundaries have been calculated from prior maps in ``fit``.
        """
        assert self._mode == "band"
        # Solid angles of selected events around each source
        min_sin_dec = np.sin(self._min_dec)
        max_sin_dec = np.sin(self._max_dec)
        self._omega = 2. * np.pi * (max_sin_dec - min_sin_dec)
        assert np.all((0. < self._omega) & (self._omega <= 4. * np.pi))


class MultiSignalFluenceInjector(BaseMultiSignalInjector):
    """
    Collect multiple SignalFluenceInjector classes, implements collective
    sampling, flux2mu and mu2flux methods.
    """
    def __init__(self, random_state=None):
        self.rndgen = random_state

    @property
    def names(self):
        return list(self._injs.keys())

    @property
    def injs(self):
        return list(self._injs.values())

    @property
    def provided_data(self):
        return [inji.model_pdf for inji in self.injs]

    @property
    def srcs(self):
        return [inji.srcs for inji in self.injs]

    def mu2flux(self, mu, per_source=False):
        """
        Convert a given number of events ``mu`` to a corresponding particle flux
        normalization :math:`F_0` in units [GeV^-1 cm^-2].

        Combines mu2flux from multiple injectors to a single output.

        Parameters
        ----------
        mu : float
            Expectation for number of events at the detector.
        per_source : bool, optional
            If ``True`` returns the expected events per source by splitting the
            total flux accroding to the sources' theoretical weights.
            (Default: ``False``)

        Returns
        -------
        flux : float or array-like
            Source flux in total, or split for each source, in unit(s)
            ``[GeV^-1 cm^-2]``.
        """
        raw_fluxes, _, w_theos = self._combine_raw_fluxes(self._injs.values())
        raw_fluxes_sum = np.sum(raw_fluxes)
        if per_source:
            return {key: mu * wts / raw_fluxes_sum for key, wts
                    in zip(self.names, w_theos)}
        return mu / raw_fluxes_sum

    def flux2mu(self, flux, per_source=False):
        """
        Calculates the number of events ``mu`` corresponding to a given particle
        flux for the current setup:

        .. math:: \mu = F_0 \sum_i \hat{w}_i

        where :math:`F_0 \sum_i \hat{w}_i` would gives the number of events. The
        weights :math:`w_i` are calculated in :py:meth:`_set_sampling_weights`.

        Combines flux2mu from multiple injectors to a single output.

        Parameters
        ----------
        flux : float
            Total source flux for all sources, in units ``[GeV^-1 cm^-2]``.
        per_source : bool, optional
            If ``True`` returns the flux per source by splitting the total flux
            according to the expected events per source at detector level.
            (Default: ``False``)

        Returns
        -------
        mu : float or dict
            Expected number of event in total or per source and sample at the
            detector.
        """
        raw_fluxes, raw_fluxes_per_src, _ = self._combine_raw_fluxes(
            self._injs.values())
        if per_source:
            return {key: flux * rfs for key, rfs
                    in zip(self.names, raw_fluxes_per_src)}
        return flux * np.sum(raw_fluxes)

    def fit(self, injectors):
        """
        Takes multiple single SignalFluenceInjectors in a dict and manages them.

        Parameters
        ----------
        injectors : dict
            Injectors to be managed by this multi Injector class. Names must
            match with dict keys needed by tested LLH.
        """
        for name, inj in injectors.items():
            if not isinstance(inj, BaseSignalInjector):
                raise ValueError("Injector object `{}`".format(name) +
                                 " is not of type `BaseSignalInjector`.")

        # Cache sampling weights
        raw_fluxes, _, _ = self._combine_raw_fluxes(injectors.values())
        self._distribute_weights = raw_fluxes / np.sum(raw_fluxes)
        assert np.isclose(np.sum(self._distribute_weights), 1.)

        self._injs = injectors

        return

    def sample(self, n_samples=1):
        """
        Split n_samples across single injectors and combine to combined sample
        after sampling each sub injector.

        Parameters
        ----------
        n_samples : int, optional
            Number of signal events to sample from all injectors in total.
            (Default: 1)

        Returns
        --------
        sam_ev : dict of record-arrays
            Combined samples of all signal injectors. Number of events sampled
            in total might be smaller depending on the sub injector settings.
            If ``n_samples<1`` an empty dict is returned.
        """
        if n_samples < 1:
            return {}

        # Distribute sample size to individual injectors
        n_per_sam = self._rndgen.multinomial(
            n_samples, self._distribute_weights, size=None)

        # Sample per injector
        sam_ev = {}
        for i, (key, inj) in enumerate(self._injs.items()):
            sam_ev[key] = inj.sample(n_samples=n_per_sam[i])

        return sam_ev

    def _combine_raw_fluxes(self, injectors):
        """
        Renormalize raw fluxes from each injector which can be used to
        distribute the amount of signal to sample across multiple injectors.

        Parameters
        ----------
        injectors : dict
            Dictionary of signal injector instances.

        Returns
        -------
        raw_fluxes_per_sam : array-like
            Summed raw flux per injector after renormalizing the theo weights.
        raw_fluxes_per_src : list of arrays
            Raw fluxes per injector and per source after renormalizing the theo
            weights.
        w_renorm : list of arrays
            Renormalized theo weights per sample per source.
        """
        # Normalize theo weights for each sample first
        weights = [inj.srcs["w_theo"] for inj in injectors]
        w_sum_per_sam = map(np.sum, weights)
        w_renorm = [wts / norm for wts, norm in zip(weights, w_sum_per_sam)]

        # Remove normalized theo weights from raw fluxes
        raw_fluxes_per_src = [inj.flux2mu(1., per_source=True) / wts
                              for inj, wts in zip(injectors, w_renorm)]
        assert np.all([len(wts) == len(rfs)
                       for wts, rfs in zip(w_renorm, raw_fluxes_per_src)])

        # Globally renormalize theo weights over all samples
        w_sum = np.sum(w_sum_per_sam)
        w_renorm = [wts / w_sum for wts in weights]

        # Renormalize fluxes per sample per source with renormalized weights
        raw_fluxes_per_src = [raw_f * wts for raw_f, wts
                              in zip(raw_fluxes_per_src, w_renorm)]

        # Combined fluxes per sample are used for sampling weights
        raw_fluxes_per_sam = np.array(map(np.sum, raw_fluxes_per_src))
        return raw_fluxes_per_sam, raw_fluxes_per_src, w_renorm


##############################################################################
# Time sampler
##############################################################################
class UniformTimeSampler(BaseTimeSampler):
    def __init__(self, random_state=None):
        self.rndgen = random_state

    def sample(self, src_t, src_dt):
        """
        Sample times uniformly in on-time signal PDF region.

        Parameters
        ----------
        src_t : array-like (nevts)
            Source time in MJD per event.
        src_dt : array-like, shape (nevts, 2)
            Time window in seconds centered around ``src_t`` in which the signal
            time PDF is assumed to be uniform.

        Returns
        -------
        times : array-like, shape (nevts)
            Sampled times for this trial.
        """
        # Sample uniformly in [0, 1] and scale to time windows per source in MJD
        r = self._rndgen.uniform(0, 1, size=len(src_t))
        times_rel = r * np.diff(src_dt, axis=1).ravel() + src_dt[:, 0]

        return src_t + times_rel / self._SECINDAY


##############################################################################
# Background injector classes
##############################################################################
class KDEBGDataInjector(BaseBGDataInjector):
    """
    Adaptive Bandwidth Kernel Density Background Injector.

    Parameters
    ----------
    kde : awkde.GaussianKDE
        Adaptive width KDE model. If an already fitted model is given,
        the ``fit`` step can be called with ``X=None`` to avoid refitting,
        which can take some time when many points with adaptive kernels are
        used.
    random_state : None, int or np.random.RandomState, optional
        Turn seed into a ``np.random.RandomState`` instance. (default: None)
    """
    def __init__(self, kde, random_state=None):
        self.rndgen = random_state
        self.kde_model = kde

    @property
    def kde_model(self):
        return self._kde_model

    @kde_model.setter
    def kde_model(self, kde_model):
        if not isinstance(kde_model, KDE):
            raise TypeError("`kde_model` must be an instance of " +
                            "`awkde.GaussianKDE`")
        self._kde_model = kde_model

    # PUBLIC
    def fit(self, X, bounds=None):
        """
        Fit a KDE model to the given data.

        Parameters
        ----------
        X : record-array or list
            Data named array. If list it is checked if the given KDE model is
            already fitted and usable and fits to the given names. This can be
            used to reuse an already existing KDE model for injection. Be
            careful that the fitted model has the data stored in the same order
            as in the list ``X``.
        bounds : None or array-like, shape (n_features, 2)
            Boundary conditions for each dimension. If ``None``,
            ``[-np.inf, +np.inf]`` is used in each dimension.
            (default: ``None``)
        """
        # TODO: Use advanced bounds via mirror method in KDE class.
        # Currently bounds are used to resample events that fall outside
        if hasattr(X, "__iter__"):
            if self._kde_model._std_X is None:
                raise ValueError("Given KDE model is not ready to use and " +
                                 "must be fitted first. Give an explicit " +
                                 "to fit the model.")
            if len(X) != self._kde_model._std_X.shape[1]:
                raise ValueError("Given names `X` do not have the same " +
                                 "dimension as the given KDE instance.")
            if not all(map(lambda s: isinstance(s, str), X)):
                raise TypeError("`X` is not a list of string names.")
            self._n_features = len(X)
            self._X_names = np.array(X)
        elif isinstance(X, np.ndarray):
            X = self._check_X_names(X)
            # Turn record-array in normal 2D array for more general KDE class
            X = np.vstack((X[n] for n in self._X_names)).T
            self._kde_model.fit(X)
        else:
            raise ValueError("`X` is neither None  nor a record array to " +
                             "fit a the given KDE model to.")

        assert (self._n_features == self._kde_model._std_X.shape[1])
        self._bounds = self._check_bounds(bounds)

    def sample(self, n_samples=1):
        """ Sample from a KDE model that has been build on given data. """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty recarray with all keys, when n_samples < 1
        if n_samples < 1:
            dtype = [(n, float) for n in self._X_names]
            return np.empty(0, dtype=dtype)

        # Resample until all sample points are inside bounds
        X = []
        bounds = self._bounds
        while n_samples > 0:
            gen = self._kde_model.sample(n_samples, self._rndgen)
            accepted = np.all(np.logical_and(gen >= bounds[:, 0],
                                             gen <= bounds[:, 1]), axis=1)
            n_samples = np.sum(~accepted)
            # Append accepted to final sample
            X.append(gen[accepted])

        # Concat sampled array list and convert to single record-array
        return np.core.records.fromarrays(np.concatenate(X).T, dtype=dtype)

    def _check_bounds(self, bounds):
        """
        Check if bounds are OK. Create numerical values when None is given.

        Returns
        -------
        bounds : array-like, shape (n_features, 2)
            Boundary conditions for each dimension. Unconstrained axes have
            bounds ``[-np.inf, +np.inf]``.
        """
        if bounds is None:
            bounds = np.repeat([[-np.inf, np.inf], ],
                               repeats=self._n_features, axis=0)

        bounds = np.array(bounds)
        if bounds.shape[1] != 2 or (bounds.shape[0] != len(self._X_names)):
            raise ValueError("Invalid `bounds`. Must be shape (n_features, 2).")

        # Convert None to +-np.inf depnding on low/hig bound
        bounds[:, 0][bounds[:, 0] == np.array(None)] = -np.inf
        bounds[:, 1][bounds[:, 1] == np.array(None)] = +np.inf

        return bounds


class ResampleBGDataInjector(BaseBGDataInjector):
    """
    Data injector resampling weighted data events from the given array.
    """
    def __init__(self, random_state=None):
        self.rndgen = random_state

    def fit(self, X, weights=None):
        """
        Build the injection model with the provided data. Here the model is
        simply the data itself.

        Parameters
        ----------
        X : record-array
            Data named array.
        weights : array-like, shape(len(X)), optional
            Weights used to sample from ``X``. If ``None`` all weights are
            equal. (default: ``None``)
        """
        self._X = self._check_X_names(X)
        nevts = len(self._X)
        if weights is None:
            weights = np.ones(nevts, dtype=float) / float(nevts)
        elif len(weights) != nevts:
            raise ValueError("'weights' must have same length as `X`.")
        # Normalize sampling weights and create sampling CDF
        CDF = np.cumsum(weights)
        self._CDF = CDF / CDF[-1]

    def sample(self, n_samples=1):
        """
        Sample by choosing random events from the given data.
        """
        # Return empty array with all keys, when n_samples < 1
        if n_samples < 1:
            dtype = [(n, float) for n in self._X_names]
            return np.empty(0, dtype=dtype)

        # Choose uniformly from given data
        idx = random_choice(rndgen=self._rndgen, CDF=self._CDF, n=n_samples)
        return self._X[idx]


class Binned3DBGDataInjector(BaseBGDataInjector):
    """
    Injector binning up data space in `[a x b x c]` bins with equal statistics
    and then sampling uniformly from those bins. Only works with 3 dimensional
    data arrays.
    """
    def __init__(self, random_state=None):
        self.rdngen = random_state
        self._ax0_bins = None
        self._ax1_bins = None
        self._ax2_bins = None

    def fit(self, X, nbins=10, minmax=False):
        """
        Build the injection model with the provided data, dimension fixed to 3.

        Parameters
        ----------
        X : record-array
            Experimental data named array.
        nbins : int or array-like, shape(n_features), optional

            - If int, same number of bins is used for all dimensions.
            - If array-like, number of bins for each dimension is used.

            (default: 10)

        minmax : bool or array-like, optional
            Defines the outermost bin edges for the 2nd and 3rd feature:

                - If False, use the min/max values in the current 1st (2nd)
                  feature bin.
                - If True, use the global min/max values per feature.
                - If array-like: Use the given edges as global min/max values
                  per feature. Must then have shape (3, 2):
                  ``[[min1, max1], [min2, max2], [min3, max3]]``.

            (default: False)

        Returns
        -------
        ax0_bins : array-like, shape (nbins[0] + 1)
            The bin borders for the first dimension.
        ax1_bins : array-like, shape (nbins[0], nbins[1] + 1)
            The bin borders for the second dimension.
        ax2_bins : array-like, shape (nbins[0], nbins[1], nbins[2] + 1)
            The bin borders for the third dimension.
        """
        def bin_equal_stats(data, nbins, minmax=None):
            """
            Bin with nbins of equal statistics by using percentiles.

            Parameters
            ----------
            data : array-like, shape(n_samples)
                The data to bin.
            nbins : int
                How many bins to create, must be smaller than len(data).
            minmax : array-like, shape (2), optional
                If [min, max] these values are used for the outer bin edges. If
                None, the min/max of the given data is used. (default: None)

            Returns
            -------
            bins : array-like
                (nbins + 1) bin edges for the given data.
            """
            if nbins > len(data):
                raise ValueError("Cannot create more bins than datapoints.")
            nbins += 1  # We need 1 more edge than bins
            if minmax is not None:
                # Use global min/max for outermost bin edges
                bins = np.percentile(data, np.linspace(0, 100, nbins)[1:-1])
                return np.hstack((minmax[0], bins, minmax[1]))
            else:
                # Else just use the bounds from the given data
                return np.percentile(data, np.linspace(0, 100, nbins))

        # Turn record-array in normal 2D array as it is easier to handle here
        X = self._check_X_names(X)
        if self._n_features != 3:
            raise ValueError("Only 3 dimensions supported here.")
        X = np.vstack((X[n] for n in self._X_names)).T

        # Repeat bins, if only int was given
        nbins = np.atleast_1d(nbins)
        if (len(nbins) == 1) and (len(nbins) != self._n_features):
            nbins = np.repeat(nbins, repeats=self._n_features)
        elif len(nbins) != self._n_features:
            raise ValueError("Given 'nbins' does not match dim of data.")
        self._nbins = nbins

        # Get bounding box, we sample the maximum distance in each direction
        if minmax is True:
            minmax = np.vstack((np.amin(X, axis=0), np.amax(X, axis=0))).T
        elif isinstance(minmax, np.ndarray):
            if minmax.shape != (self._n_features, 2):
                raise ValueError("'minmax' must have shape (3, 2) if edges " +
                                 "are given explicitely.")
        else:
            minmax = self._n_features * [None]

        # First axis is the main binning and only an 1D array
        ax0_dat = X[:, 0]
        ax0_bins = bin_equal_stats(ax0_dat, nbins[0], minmax[0])

        # 2nd axis array has bins[1] bins per bin in ax0_bins, so it's 2D
        ax1_bins = np.zeros((nbins[0], nbins[1] + 1))
        # 3rd axis is 3D: nbins[2] bins per bin in ax0_bins and ax1_bins
        ax2_bins = np.zeros((nbins[0], nbins[0], nbins[2] + 1))

        # Fill bins by looping over all possible combinations
        for i in range(nbins[0]):
            # Bin left inclusive, except last bin
            m = (ax0_dat >= ax0_bins[i]) & (ax0_dat < ax0_bins[i + 1])
            if (i == nbins[1] - 1):
                m = (ax0_dat >= ax0_bins[i]) & (ax0_dat <= ax0_bins[i + 1])

            # Bin ax1 subset of data in current ax0 bin
            _X = X[m]
            ax1_dat = _X[:, 1]
            ax1_bins[i] = bin_equal_stats(ax1_dat, nbins[1], minmax[1])

            # Directly proceed to axis 2 and repeat procedure
            for k in range(nbins[1]):
                m = ((ax1_dat >= ax1_bins[i, k]) &
                     (ax1_dat < ax1_bins[i, k + 1]))
                if (k == nbins[2] - 1):
                    m = ((ax1_dat >= ax1_bins[i, k]) &
                         (ax1_dat <= ax1_bins[i, k + 1]))

                # Bin ax2 subset of data in current ax0 & ax1 bin
                ax2_bins[i, k] = bin_equal_stats(_X[m][:, 2],
                                                 nbins[2], minmax[2])

        self._ax0_bins = ax0_bins
        self._ax1_bins = ax1_bins
        self._ax2_bins = ax2_bins
        return ax0_bins, ax1_bins, ax2_bins

    def sample(self, n_samples=1):
        """
        Sample pseudo events uniformly from each bin.
        """
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty array with all keys, when n_samples < 1
        if n_samples < 1:
            dtype = [(n, float) for n in self._X_names]
            return np.empty(0, dtype=dtype)

        # Sample indices to select from which bin is injected
        ax0_idx = self._rndgen.randint(0, self._nbins[0], size=n_samples)
        ax1_idx = self._rndgen.randint(0, self._nbins[1], size=n_samples)
        ax2_idx = self._rndgen.randint(0, self._nbins[2], size=n_samples)

        # Sample uniformly in [0, 1] to decide where each point lies in the bins
        r = self._rndgen.uniform(0, 1, size=(n_samples, self._n_features))

        # Get edges of each bin
        ax0_edges = np.vstack((self._ax0_bins[ax0_idx],
                               self._ax0_bins[ax0_idx + 1])).T

        ax1_edges = np.vstack((self._ax1_bins[ax0_idx, ax1_idx],
                               self._ax1_bins[ax0_idx, ax1_idx + 1])).T

        ax2_edges = np.vstack((self._ax2_bins[ax0_idx, ax1_idx, ax2_idx],
                               self._ax2_bins[ax0_idx, ax1_idx, ax2_idx + 1])).T

        # Sample uniformly between selected bin edges
        ax0_pts = ax0_edges[:, 0] + r[:, 0] * np.diff(ax0_edges, axis=1).T
        ax1_pts = ax1_edges[:, 0] + r[:, 1] * np.diff(ax1_edges, axis=1).T
        ax2_pts = ax2_edges[:, 0] + r[:, 2] * np.diff(ax2_edges, axis=1).T

        # Combine and convert to record-array
        return np.core.records.fromarrays(np.vstack((ax0_pts, ax1_pts,
                                                     ax2_pts)).T, dtype=dtype)


class TimeDecDependentBGDataInjector(BaseBGDataInjector):
    """
    Models the injection part for the GRB LLH.

    BG injection is allsky and time and declination dependent:
      1. For each source time build a declination dependent detector profile
         from which the declination is sampled weighted.
      2. For each source the integrated event rate over the time interval is
         used to draw the number of events to sample.
      3. Then the total number of events for the source is sampled from the
         total pool of experimental data weighted in declination.
      4. RA is sampled uniformly in ``[0, 2pi]`` and times are sampled from the
         rate function (uniform for small time windows.)
    """
    def __init__(self, bg_inj_args, rndgen=None):
        self._provided_data_names = np.array(
            ["timeMJD", "dec", "ra", "sigma", "logE"])

        # Check BG inj args
        req_keys = ["sindec_bins", "rate_rebins"]
        opt_keys = {"spl_s": None}
        self.bg_inj_args = fill_dict_defaults(bg_inj_args, req_keys, opt_keys)

        self._sin_dec_bins = np.atleast_1d(self.bg_inj_args["sindec_bins"])
        self._rate_rebins = np.atleast_1d(self.bg_inj_args["rate_rebins"])
        self.rndgen = rndgen

        # Defaults for private class variables
        self._nsrcs = None
        self._src_t = None
        self._src_trange = None
        self._allsky_rate_func = None
        self._allsky_pars = None
        self._nb = None

        # Debug info
        self._data_spl = None
        self._sin_dec_splines = None
        self._param_splines = None
        self._best_pars = None
        self._best_stddevs = None
        self._bg_cur_sam = None

        self._log = logger(name=self.__class__.__name__, level="ALL")

    def fit(self, X, srcs, run_dict):
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
        """
        X_names = np.array(X.dtype.names)
        for name in self._provided_data_names:
            if name not in X_names:
                raise ValueError("`X` is missing name '{}'.".format(name))
        drop = np.isin(X_names, self._provided_data_names,
                       assume_unique=True, invert=True)
        drop_names = X_names[drop]
        print(self._log.INFO("Dropping names '{}'".format(arr2str(drop_names)) +
                             " from data recarray."))

        self.X = drop_fields(X, drop_names, usemask=False)
        self.srcs = srcs
        self._setup_data_injector(self.X, self.srcs, run_dict)

        return

    def sample(self):
        """
        Get a complete data sample for one trial.

        1. Get expected nb per source from allsky rate spline
        2. Sample times from allsky rate splines
        3. Sample same number of events from data using the CDF per source
           for the declination distribution
        4. Combine to a single recarray X
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

        # Concat to a single recarray
        self._bg_cur_sam = sam
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
        print(self._log.INFO("Create time dep sindec splines."))
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


class MultiBGDataInjector(BaseMultiBGDataInjector):
    @property
    def names(self):
        return list(self._injs.keys())

    @property
    def injs(self):
        return list(self._injs.values())

    @property
    def provided_data(self):
        return [inji.provided_data for inji in self.injs]

    def fit(self, injs):
        """
        Takes multiple single injectors in a dict and manages them.

        Parameters
        ----------
        injs : dict
            Injectors to be managed by this multi injector class. Names must
            match with dict keys of required multi-LLH data.
        """
        for name, inj in injs.items():
            if not isinstance(inj, BaseBGDataInjector):
                raise ValueError("LLH object `{}` ".format(name) +
                                 "is not of type `BaseBGDataInjector`.")

        self._injs = injs

    def sample(self):
        """
        Sample each injector and combine to a dict of recarrays.
        """
        sam = {}
        for key, inj in self._injs.items():
            sam[key] = inj.sample()

        return sam


##############################################################################
# Rate function classes to fit a BG rate model
##############################################################################
class SinusRateFunction(BaseRateFunction):
    """
    Sinus Rate Function

    Describes time dependent background rate. Used function is a sinus with:

    .. math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

    depending on 4 parameters:

    - a, float: Amplitude in Hz.
    - b, float: Angular frequency, ``b = 2*pi / T`` with period ``T`` given in
                1 / (MJD days).
    - c, float: x-axis offset in MJD.
    - d, float: y-axis offset in Hz
    """
    # Just to have some info encoded in the class which params we have
    _PARAMS = ["amplitude", "period", "toff", "baseline"]

    def __init__(self, random_state=None):
        super(SinusRateFunction, self).__init__(random_state)
        # Cached in `fit` for faster rejection sampling
        self._fmax = None
        self._trange = None

    def fit(self, t, rate, srcs, p0=None, w=None, **minopts):
        """
        Fit the rate model to discrete points ``(t, rate)``. Cache source values
        for fast sampling.

        Parameters
        ----------
        srcs : record-array
            Must have names ``'t', 'dt0', 'dt1'`` describing the time intervals
            around the source times to sample from.
        """
        fitres = super(SinusRateFunction, self).fit(t, rate, p0, w, **minopts)

        # Cache max function values in the fixed source intervals for sampling
        required_names = ["t", "dt0", "dt1"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` recarray is missing name " +
                                 "'{}'.".format(n))
        _dts = np.vstack((srcs["dt0"], srcs["dt1"])).T
        _, _trange = self._transform_trange_mjd(srcs["t"], _dts)
        self._fmax = self._calc_fmax(
            _trange[:, 0], _trange[:, 1], pars=fitres.x)
        self._trange = _trange

        return fitres

    def fun(self, t, pars):
        """ Full 4 parameter sinus rate function """
        a, b, c, d = pars
        return a * np.sin(b * (t - c)) + d

    def integral(self, t, trange, pars):
        """ Analytic integral, full 4 parameter sinus """
        a, b, c, d = pars

        # Transform time windows to MJD
        _, dts = self._transform_trange_mjd(t, trange)
        t0, t1 = dts[:, 0], dts[:, 1]

        per = a / b * (np.cos(b * (t0 - c)) - np.cos(b * (t1 - c)))
        lin = d * (t1 - t0)

        # Match units with secinday = 24 * 60 * 60 s/MJD = 86400 / (Hz*MJD)
        #     [a], [d] = Hz;M [b] = 1/MJD; [c], [t] = MJD
        #     [a / b] = Hz * MJD; [d * (t1 - t0)] = HZ * MJD
        return (per + lin) * self._SECINDAY

    def sample(self, n_samples):
        """
        Rejection sample from the fitted sinus function

        Parameters
        ----------
        n_samples : array-like
            How many events to sample per source. Length must match length of
            cached source positions if any.
        """
        n_samples = np.atleast_1d(n_samples)
        if self._bf_pars is None:
            raise RuntimeError("Rate function was not fit yet.")
        if len(n_samples) != len(self._fmax):
            raise ValueError("Requested to sample a different number of " +
                             "sources than have been fit")

        # Just loop over all intervals and rejection sample the src regions
        for i, (bound, nsam) in enumerate(zip(self._trange, n_samples)):
            # Draw remaining events until all samples per source are created
            sample = []
            while nsam > 0:
                t = self._rndgen.uniform(bound[0], bound[1], size=nsam)
                y = self._fmax[i] * self._rndgen.uniform(0, 1, size=nsam)

                accepted = (y <= self._bf_fun(t))
                sample += t[accepted].tolist()
                nsam = np.sum(~accepted)  # Number of remaining samples to draw

            sample.append(np.array(sample))

        return sample

    def _get_default_seed(self, t, rate, w):
        """
        Default seed values for the specifiv RateFunction fit.

        Motivation for default seed:

        - a0 : Using the width of the central 50\% percentile of the rate
               distribtion (for rates > 0). The sign is determined based on
               wether the average rates in the first two octants based on the
               period seed decrease or increase.
        - b0 : The expected seasonal variation is 1 year.
        - c0 : Earliest time in ``t``.
        - d0 : Weighted averaged rate, which is the best fit value for a
               constant target function.

        Returns
        -------
        p0 : tuple, shape (4)
            Seed values `(a0, b0, c0, d0)`:

            - a0 : ``-np.diff(np.percentile(rate[w > 0], q=[0.25, 0.75])) / 2``
            - b0 : ``2 * pi / 365``
            - c0 : ``np.amin(t)``
            - d0 : ``np.average(rate, weights=w)``
        """
        a0 = 0.5 * np.diff(np.percentile(rate[w > 0], q=[0.25, 0.75]))[0]
        b0 = 2. * np.pi / 365.
        c0 = np.amin(t)
        d0 = np.average(rate, weights=w)

        # Get the sign of the amplitude a0 depending on wether the average
        # falls or rises in the first 2 octants of the whole period.
        m0 = (c0 <= t) & (t <= c0 + 365. / 8.)
        oct0 = np.average(rate[m0], weights=w[m0])
        m1 = (c0 <= t + 365. / 8.) & (t <= c0 + 365. / 4.)
        oct1 = np.average(rate[m1], weights=w[m1])
        sign = np.sign(oct1 - oct0)

        return (sign * a0, b0, c0, d0)

    def _calc_fmax(self, t0, t1, pars):
        """
        Get the analytic maximum function value in interval ``[t0, t1]`` cached
        for rejection sampling.
        """
        a, b, c, d = pars
        L = 2. * np.pi / b  # Period length
        # If we start with a negative sine, then the first max is after 3/4 L
        if np.sign(a) == 1:
            step = 1.
        else:
            step = 3.
        # Get dist to first max > c and count how many periods k it is away
        k = np.ceil((t0 - (c + step * L / 4.)) / L)
        # Get the closest next maximum to t0 with t0 <= tmax_k
        tmax_gg_t0 = L / 4. * (step + 4. * k) + c

        # If the next max is <= t1, then fmax must be the global max, else the
        # highest border
        fmax = np.zeros_like(t0) + (np.abs(a) + d)
        m = (tmax_gg_t0 > t1)
        fmax[m] = np.maximum(self.fun(t0[m], pars), self.fun(t1[m], pars))

        return fmax

    def _scale(self, t, p0, bounds):
        """ Scale x axis to [0,1] - times, seeds and bounds - before fitting """
        min_t, max_t = np.amin(t), np.amax(t)
        dt = max_t - min_t

        t_ = (t - min_t) / dt
        b_ = dt * p0[1]
        c_ = (p0[2] - min_t) / dt

        if bounds is not None:
            b_bnds = [dt * bounds[1, 0], dt * bounds[1, 1]]
            c_bnds = [(bounds[2, 0] - min_t) / dt,
                      (bounds[2, 1] - min_t) / dt]

            bounds = [bounds[0], b_bnds, c_bnds, bounds[3]]

        return t_, (p0[0], b_, c_, p0[3]), bounds, min_t, max_t

    def _rescale(self, res, min_t, max_t):
        """ Rescale fitres and errors after fitting """
        dt = (max_t - min_t)
        best_pars = res.x

        b_ = best_pars[1]
        c_ = best_pars[2]

        b = b_ / dt
        c = c_ * dt + min_t

        res.x = np.array([best_pars[0], b, c, best_pars[3]])

        try:
            errs = res.hess_inv
        except AttributeError:
            errs = res.hess_inv.todense()

        # Var[a*x] = a^2*x. Cov[a*x+b, c*y+d] = a*c*Cov[x, y]
        # >>> Only need to scale the variances because dt / dt drops out in Cov
        errs[1, 1] = dt**2 * errs[1, 1]
        errs[2, 2] = errs[2, 2] / dt**2
        res.hess_inv = errs

        return res


class SinusFixedRateFunction(SinusRateFunction):
    """
    Same a sinus Rate Function but period and time offset can be fixed for the
    fit and stays constant.
    """
    def __init__(self, p_fix=None, t0_fix=None, random_state=None):
        super(SinusFixedRateFunction, self).__init__(random_state)

        # Process which parameters are fixed and which get fitted
        self._fit_idx = np.ones(4, dtype=bool)
        self._b = None
        self._c = None

        if p_fix is not None:
            if p_fix <= 0.:
                raise ValueError("Fixed period must be >0 days.")
            self._fit_idx[1] = False
            self._b = 2 * np.pi / p_fix

        if t0_fix is not None:
            self._fit_idx[2] = False
            self._c = t0_fix

        self._fit_idx = np.arange(4)[self._fit_idx]
        self._PARAMS = np.array(super(SinusFixedRateFunction,
                                      self)._PARAMS)[self._fit_idx]

    def fun(self, t, pars):
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).fun(t, pars)

    def integral(self, t, trange, pars):
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).integral(t, trange, pars)

    def _get_default_seed(self, t, rate, w):
        """ Same default seeds as SinusRateFunction, but drop fixed params. """
        seed = super(SinusFixedRateFunction,
                     self)._get_default_seed(t, rate, w)
        # Drop b0 and / or c0 seed, when it's marked as fixed
        return tuple(seed[i] for i in self._fit_idx)

    def _calc_fmax(self, t0, t1, pars):
        """
        Copy and pasted with a minor change to make it work, needs a better
        solution. Problem is, that we need all 4 params first and then only the
        stripped version when using fixed rate functions.
        """
        a, b, c, d = self._make_params(pars)
        L = 2. * np.pi / b  # Period length
        # If we start with a negative sine, then the first max is after 3/4 L
        if np.sign(a) == 1:
            step = 1.
        else:
            step = 3.
        # Get dist to first max > c and count how many periods k it is away
        k = np.ceil((t0 - (c + step * L / 4.)) / L)
        # Get the closest next maximum to t0 with t0 <= tmax_k
        tmax_gg_t0 = L / 4. * (step + 4. * k) + c

        # If the next max is <= t1, then fmax must be the global max, else the
        # highest border
        fmax = np.zeros_like(t0) + (np.abs(a) + d)
        m = (tmax_gg_t0 > t1)
        fmax[m] = np.maximum(self.fun(t0[m], pars), self.fun(t1[m], pars))

        return fmax

    def _scale(self, t, p0, bounds):
        """ Scale x axis to [0,1] - times, seeds and bounds - before fitting """
        if len(p0) != len(self._fit_idx):
            raise ValueError("Given number of parameters does not match the " +
                             "number of free parameters here.")

        min_t, max_t = np.amin(t), np.amax(t)
        dt = max_t - min_t

        t_ = (t - min_t) / dt

        # Explicit handling OK here, because we have only 4 combinations
        if self._b is None:
            if self._c is None:
                return super(SinusFixedRateFunction, self)._scale(t, p0, bounds)
            b_ = dt * p0[1]
            p0 = (p0[0], b_, p0[2])
        elif self._c is None:
            c_ = (p0[2] - min_t) / dt
            p0 = (p0[0], c_, p0[2])
        else:
            # Don't scale
            pass

        if bounds is not None:
            if len(bounds) != len(self._fit_idx):
                raise ValueError("Given number of bounds does not match the " +
                                 "number of free parameters here.")
            if self._b is None:
                if self._c is None:
                    return super(SinusFixedRateFunction, self)._scale(t, p0, bounds)
                b_bnds = [dt * bounds[1, 0], dt * bounds[1, 1]]
                p0 = (p0[0], b_, p0[2])
                bounds = [bounds[0], b_bnds, bounds[2]]
            elif self._c is None:
                c_bnds = [(bounds[1, 0] - min_t) / dt,
                          (bounds[1, 1] - min_t) / dt]
                bounds = [bounds[0], c_bnds, bounds[2]]
            else:
                # Don't scale
                pass

        return t_, p0, bounds, min_t, max_t

    def _rescale(self, res, min_t, max_t):
        """ Rescale fitres and errors after fitting """
        if len(res.x) != len(self._fit_idx):
            raise ValueError("Number of best fit params does not match the " +
                             "number of free parameters here.")

        dt = (max_t - min_t)

        try:
            errs = res.hess_inv
        except AttributeError:
            errs = res.hess_inv.todense()

        if self._b is None:
            if self._c is None:
                return super(SinusFixedRateFunction, self)._rescale(
                    res, min_t, max_t)
            b_ = res.x[1]
            b = b_ / dt
            res.x[1] = b
            errs[1, 1] = dt**2 * errs[1, 1]
        elif self._c is None:
            c_ = res.x[1]
            c = c_ * dt + min_t
            res.x[1] = c
            errs[1, 1] = errs[1, 1] / dt**2
        else:
            # Don't rescale
            pass

        # Var[a*x] = a^2*x. Cov[a*x+b, c*y+d] = a*c*Cov[x, y]
        # >>> Only need to scale the variances because dt / dt drops out in Cov
        res.hess_inv = errs

        return res

    def _make_params(self, pars):
        """
        Check which parameters are fixed and insert them where needed to build
        a full 4 parameter set.

        Returns
        -------
        pars : tuple
            Fixed parameters inserted in the full argument list.
        """
        if len(pars) != len(self._fit_idx):
            raise ValueError("Given number of parameters does not match the " +
                             "number of free parameters here.")
        # Explicit handling OK here, because we have only 4 combinations
        if self._b is None:
            if self._c is None:
                pars = pars
            pars = (pars[0], pars[1], self._c, pars[2])
        elif self._c is None:
            pars = (pars[0], self._b, pars[1], pars[2])
        else:
            pars = (pars[0], self._b, self._c, pars[1])
        return pars


class SinusFixedConstRateFunction(SinusFixedRateFunction):
    """
    Same as SinusFixedRateFunction, but sampling uniform times in each time
    interval instead of rejection sampling the sine function.

    Here the number of expected events is still following the seasonal
    fluctuations, but within the time windows we sample uniformly (step function
    like). Perfect for small time windows, avoiding rejection sampling and thus
    giving a speed boost.
    """
    def __init__(self, p_fix=None, t0_fix=None, random_state=None):
        super(SinusFixedConstRateFunction, self).__init__(
            p_fix, t0_fix, random_state)

    def sample(self, n_samples):
        """
        Just sample uniformly in MJD time windows here.

        Parameters
        ----------
        n_samples : array-like
            How many events to sample per source. Length must match length of
            cached source positions if any.
        """
        n_samples = np.atleast_1d(n_samples)
        if self._fmax is None:
            raise RuntimeError("Rate function was not fit yet.")
        if len(n_samples) != len(self._fmax):
            raise ValueError("Requested to sample a different number of " +
                             "sources than have been fit")

        # Samples times for all sources at once
        sample = []
        for dt, nsam in zip(self._trange, n_samples):
            sample.append(self._rndgen.uniform(dt[0], dt[1], size=nsam))

        return sample


class ConstantRateFunction(BaseRateFunction):
    """
    Uses a constant rate in Hz at any given time in MJD. This models no seasonal
    fluctuations but uses the constant average rate.

    Uses one parameter:

    - rate, float: Constant rate in Hz.
    """
    _PARAMS = ["baseline"]

    def __init__(self, random_state=None):
        self.rndgen = random_state
        self._trange = None

    def fit(self, rate, srcs, w=None):
        """ Cache source values for sampling. Fit is the weighted average """
        if w is None:
            w = np.ones_like(rate)

        required_names = ["t", "dt0", "dt1"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` recarray is missing name " +
                                 "'{}'.".format(n))
        _dts = np.vstack((srcs["dt0"], srcs["dt1"])).T
        _, self._trange = self._transform_trange_mjd(srcs["t"], _dts)

        # Analytic solution to fit
        bf_pars = self._get_default_seed(rate, w)
        self._bf_fun = (lambda t: self.fun(t, bf_pars))
        self._bf_int = (lambda t, trange: self.integral(t, trange, bf_pars))
        self._bf_pars = bf_pars
        return bf_pars

    def fun(self, t, pars):
        """ Returns constant rate Hz at any given time in MJD """
        return np.ones_like(t) * pars[0]

    def integral(self, t, trange, pars):
        """
        Analytic integral of the rate function in interval trange. Because the
        rate function is constant, the integral is simply ``trange * rate``.
        """
        t, dts = self._transform_trange_mjd(t, trange)
        # Multiply first then diff to avoid roundoff errors(?)
        return (np.diff(dts * self._SECINDAY, axis=1) *
                self.fun(t, pars)).ravel()

    def sample(self, n_samples):
        n_samples = np.atleast_1d(n_samples)
        if len(n_samples) != len(self._trange):
            raise ValueError("Requested to sample a different number of " +
                             "sources than have been fit")

        # Samples times for all sources at once
        sample = []
        for dt, nsam in zip(self._trange, n_samples):
            sample.append(self._rndgen.uniform(dt[0], dt[1], size=nsam))

        return sample

    def _get_default_seed(self, rate, w):
        """
        Motivation for default seed:

        - rate0 : Mean of the given rates. This is the anlytic solution to the
                  fit, so we seed with the best fit. Weights must be squared
                  though, to get the same result.

        Returns
        -------
        p0 : tuple, shape(1)
            Seed values ``rate0 = np.average(rate, weights=w**2)``
        """
        return (np.average(rate, weights=w**2), )
