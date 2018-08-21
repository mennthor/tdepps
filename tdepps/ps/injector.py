# coding: utf-8

"""
Collection of base implementations for a GRB style, time dependent analysis.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.lib.recfunctions import drop_fields
import healpy as hp

from ..base import BaseSignalInjector, BaseMultiSignalInjector
from ..base import BaseBGDataInjector, BaseMultiBGDataInjector
from ..utils import arr2str, logger, dict_map, fill_dict_defaults
from ..utils import random_choice
from ..utils import rotator, thetaphi2decra, get_pixel_in_sigma_region


##############################################################################
# Signal injector classes
##############################################################################
class PowerLawFluxInjector(BaseSignalInjector):
    """
    Signal Flux Injector

    Inject signal events from Monte Carlo data weighted to a specific flux model
    and inject at given source positions. The flux is assumed to be a steady
    emission from all sources.

    Events are injected by rotation true event positions from a diffuse MC set
    to the source locations. Events are sampled weighted using an external
    energy dependent flux model.
    """
    def __init__(self, gamma=2., E0=1., inj_opts=None, random_state=None):
        """
        Parameters
        ----------
        gamma, E0 : float
            Postive spectral index ``gamma`` and normalization energy ``E0`` in
            GeV of the assumed unbroken power law that is used to weight the
            simulation events. The power law is

            ..math: \phi(E) = \phi_0 \cdot (E / E0)^{-\gamma}

            Events are weighted to the physics scenario by
            ``w[i] ~ (MC['trueE'][i] / E0)**-gamma * MC['ow'][i]``, where
            ``trueE`` also is in units of GeV.
            (default: gamma = 2., E0 = 1.)
        inj_opts : dict, optional
            Injector options:
            - 'sindec_inj_width', optional: Size in ``sin(angle)`` of the region
              from which MC events are selected and rotated to each source. See
              'mode' for how the MC selection region is chosen corresponding to
              'sindec_inj_width'. (default: ``0.035 ~ sin(2°)``)
            - 'mode', optional: One of ``['circle'|'band']``. Selects MC events
              to inject based on their true location (default: 'band'):
              + 'circle': Select in a circle with radius 'sindec_inj_width'
                around each source.
              + 'band': Select in a declination band with width
                'sindec_inj_width' in each direction symmetrically around each
                source.
            - 'dec_range', optional: Global declination interval in which events
              can be injected in rotated coordinates. Events rotated outside are
              dropped even if selected, which drops sensitivity as desired.
              (default: ``[-pi/2, pi/2]``)
        random_state : None, int or np.random.RandomState, optional
            Used as PRNG, see ``sklearn.utils.check_random_state``.
            (default: None)
        """
        if gamma < 1.:
            raise TypeError("`gamma` should be > 1 for physical injection.")
        if not E0 > 0.:
            raise TypeError("`E0` must be a postive energy in units of GeV.")

        # Check injector options
        req_keys = None
        opt_keys = {"mode": "band", "sindec_inj_width": 0.035,
                    "dec_range": np.array([-np.pi / 2., np.pi / 2.])}
        inj_opts = fill_dict_defaults(inj_opts, req_keys, opt_keys,
                                      noleft="use")
        if inj_opts["mode"] not in ["band", "circle"]:
            raise ValueError("'mode' must be one of ['band', 'circle']")
        _sindw = inj_opts["sindec_inj_width"]
        if (_sindw <= 0.) or (_sindw > np.pi):
            raise ValueError("'sindec_inj_width' width must be in (0, pi].")
        _decrng = np.atleast_1d(inj_opts["dec_range"])
        if (_decrng[0] < -np.pi / 2.) or (_decrng[1] > np.pi / 2.):
            raise ValueError("`dec_range` must be in range [-pi/2, pi/2].")
        if _decrng[0] >= _decrng[1]:
            raise ValueError("`dec_range=[low, high]` must be increasing.")
        inj_opts["dec_range"] = _decrng

        self._gamma = gamma
        self._E0 = E0
        self._inj_opts = inj_opts
        self._provided_data = np.array(
            ["dec", "ra", "sigma", "logE"])
        self._mc_names = np.array(["trueRa", "trueDec", "trueE", "ow"])
        self._srcs = None
        self.rndgen = random_state

        # Defaults for private class attributes
        self._SECINDAY = 24. * 60. * 60.
        self._MC = None
        self._livetime = None
        self._mc_idx = None
        self._raw_flux = None
        self._raw_flux_per_src = None
        self._sample_w_CDF = None
        self._sample_dtype = [(name, float) for name in self._provided_data]

        # Debug attributes
        self._sample_idx = None
        self._skylab_band = False

        self._log = logger(name=self.__class__.__name__, level="ALL")

    @property
    def provided_data(self):
        return self._provided_data

    @property
    def srcs(self):
        return self._srcs

    @property
    def flux_model(self):
        return self._flux_model

    @property
    def inj_opts(self):
        return self._inj_opts.copy()

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
            (default: ``False``)

        Returns
        -------
        flux : float or array-like
            Total source flux normalization for the flux model given at
            initialization, or split for each source, in unit(s)
            ``[GeV^-1 cm^-2]``.
        """
        if per_source:
            # Split the total mu according to the theoretical source weights
            mu = self._w_theo_norm * mu
        return mu / self._raw_flux

    def flux2mu(self, flux, per_source=False):
        """
        Calculates the number of events ``mu`` corresponding to a given particle
        flux normalization for the current setup:

        .. math:: \mu = F_0 \sum_i \hat{w}_i

        where :math:`F_0 \sum_i \hat{w}_i` would give the number of events. The
        weights :math:`w_i` are calculated in :py:meth:`_set_sampling_weights`.

        Parameters
        ----------
        flux : float
            Total source flux normalization for the flux model given at
            intialization for all sources, in units ``[GeV^-1 cm^-2]``.
        per_source : bool, optional
            If ``True`` returns the flux per source by splitting the total flux
            according to the expected events per source at detector level.
            (default: ``False``)

        Returns
        -------
        mu : float or array-like
            Expected number of event in total or per source at the detector.
        """
        if per_source:
            # Split the intrinsic total flux according to each srcs acceptance
            mu = flux * self._raw_flux_per_src
        else:
            mu = flux * self._raw_flux
        return mu

    def fit(self, srcs, MC, livetime):
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
            same names as given in ``provided_data`` and additonal MC truths:
            - 'trueE', float: True event energy in GeV.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator' OneWeight [2]_,
              so it is already divided by ``nevts * nfiles`.
              Units are 'GeV sr cm^2'. Final event weights are obtained by
              multiplying with desired sum flux for nu and anti-nu flux.
        livetimes : float
            Livetime in days of the data, that the simulation ``MC`` represents.

        Notes
        -----
        .. [1] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
        """
        self._check_fit_input(srcs, MC)
        nsrcs = len(srcs)

        # Set injection solid angles for all sources
        omega, min_dec, max_dec = self._set_solid_angle(srcs)

        # Select events in the injection regions, masking all srcs at once
        if self._inj_opts["mode"] == "band":
            min_decs = min_dec.reshape(nsrcs, 1)
            max_decs = max_dec.reshape(nsrcs, 1)
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
            cos_r = np.cos(np.arcsin(self._inj_opts["sindec_inj_width"]))
            inj_mask = (cos_dist > cos_r)

        if not np.any(inj_mask):
            # If no events were selected we don't need this src-MC combination
            raise RuntimeError("No events selected. Check `srcs` and `MC`.")

        # Select unqique MC in all injection regions (include overlap)
        total_mask = np.any(inj_mask, axis=0)
        N_unique = np.count_nonzero(total_mask)
        # Only keep MC names (MC truth and those specified by provided_names)
        keep_names = np.concatenate((self._mc_names, self._provided_data))
        drop_names = [n for n in MC.dtype.names if n not in keep_names]
        self._MC = drop_fields(MC[total_mask], drop_names)
        assert len(self._MC) == N_unique

        # Total number of all selected events, including multi counts
        core_mask = (inj_mask.T[total_mask]).T  # Remove all non-selected
        n_tot = np.count_nonzero(core_mask)     # Equal to count inj_mask

        # Only store selected event IDs in mc_idx to sample from the unique MC
        # pool to save memory.
        # ev_idx: index of evt in _MC; src_idx: index in _srcs for each event
        self._mc_idx = np.empty(n_tot, dtype=[("ev_idx", np.int),
                                              ("src_idx", np.int)])

        _core = core_mask.ravel()  # [src1_mask, src2_mask, ...]
        self._mc_idx["ev_idx"] = np.tile(np.arange(N_unique), nsrcs)[_core]
        # Same src IDs for each selected evt per src (rows in core_mask)
        self._mc_idx['src_idx'] = np.repeat(np.arange(nsrcs),
                                            np.sum(core_mask, axis=1))

        s_ = "Selected {:d} evts at {:d} sources.".format(n_tot, nsrcs) + "\n"
        s_ += len(self._log.INFO()) * " "
        s_ += "- Sources without selected evts: {}".format(
            nsrcs - np.count_nonzero(np.sum(core_mask, axis=1)))
        print(self._log.INFO(s_))

        # Build sampling weigths
        (self._raw_flux, self._raw_flux_per_src, self._w_theo_norm,
            self._sample_w_CDF) = self._set_sampling_weights(
            self._MC, self._mc_idx, srcs, omega, livetime)

        self._srcs = srcs
        self._livetime = livetime

        return

    def sample(self, n_samples=1):
        """
        Get sampled events from stored MC for each stored source position.

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
        if n_samples < 1:
            return np.empty(0, dtype=self._sample_dtype)

        # Draw IDs from the whole stored pool of MC events
        sam_idx = random_choice(self._rndgen, self._sample_w_CDF,
                                size=n_samples)
        sam_idx = self._mc_idx[sam_idx]

        # Select events from pool and rotate them to corresponding src positions
        sam_ev = self._MC[sam_idx["ev_idx"]]
        src_idx = sam_idx["src_idx"]
        sam_ev = self._rot_and_strip(self._srcs["ra"][src_idx],
                                     self._srcs["dec"][src_idx],
                                     sam_ev)

        # Debug purpose
        self._sample_idx = sam_idx
        return sam_ev

    def _set_sampling_weights(self, MC_pool, mc_idx, srcs, omega, livetime):
        """
        Setup per event sampling weights from the OneWeights.

        Physics weights are calculated for a simple unbroken power law particle
        flux (per particle type) differential in energy, detection area and
        time:

        .. math:: dN/(dE dA dt) = F_0 (E / GeV)^{-\gamma}

        with the normalization :math:`F_0` at 1 GeV in units
        ``[GeV^-1 cm^-2 s^-1]``.

        Because we inject only from a fraction of the sky from the diffuse MC
        per source (band or circle) the per event physics weight are calculated
        using:

        .. math::

          w_i = [\text{ow}]_i \times \left.\frac{dF}{dE}\right|_{E_i} \times
                \frac{w_\text{src}}{\Omega_\text{src}} \times T

        where ``Omega_src`` is the injected solid angle and ``w_src`` the
        intrinsic weight for the source the event :math:`i` is injected at and
        ``ow`` is the NuGen OneWeight per type already divided by
        ``nfiles * nevents * type_weight``.

        We then get the number of expected events n as

        .. math:: n = \sum_i w_i = F_0 \sum_i \hat{w}_i

        where the free to choose normalization :math:`F_0` is explicitly written
        in the last step. See :py:meth:`mu2flux` which calculates the
        fluence from a given number of events from that relation.
        """
        mc = MC_pool[mc_idx["ev_idx"]]
        src_idx = mc_idx["src_idx"]

        # Normalize w_theos to split intrinsic injected fluxes
        w_theo_norm = srcs["w_theo"] / np.sum(srcs["w_theo"])
        assert np.isclose(np.sum(w_theo_norm), 1.)

        # Broadcast solid angles and w_theo to corresponding srcs for each event
        omega = omega[src_idx]
        w_theo = w_theo_norm[src_idx]
        flux = (mc["trueE"] / self._E0)**-self._gamma
        assert len(omega) == len(w_theo) == len(flux)

        w = mc["ow"] * flux / omega * w_theo * livetime * self._SECINDAY
        assert len(mc_idx) == len(w)

        raw_flux_per_src = np.array(
            [np.sum(w[src_idx == j]) for j in np.arange(len(srcs))])
        raw_flux = np.sum(w)
        assert np.isclose(np.sum(raw_flux_per_src), raw_flux)

        # Cache sampling CDF used for injecting events from the whole MC pool
        sample_w_CDF = np.cumsum(w) / raw_flux
        assert np.isclose(sample_w_CDF[-1], 1.)

        return raw_flux, raw_flux_per_src, w_theo_norm, sample_w_CDF

    def _set_solid_angle(self, srcs):
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
        nsrcs = len(srcs)
        sin_inj_w = self._inj_opts["sindec_inj_width"]

        if self._inj_opts["mode"] == "band":
            sinL, sinU = np.sin(self._inj_opts["dec_range"])
            if self._skylab_band:
                # Recenter sources somewhat so that bands get bigger at poles
                m = (sinL - sinU + 2. * sin_inj_w) / (sinL - sinU)
                sinU = sin_inj_w * (sinL + sinU) / (sinU - sinL)
                sin_dec = m * np.sin(srcs["dec"]) + sinU
            else:
                sin_dec = np.sin(srcs["dec"])

            min_sin_dec = np.maximum(sinL, sin_dec - sin_inj_w)
            max_sin_dec = np.minimum(sinU, sin_dec + sin_inj_w)

            min_dec = np.arcsin(np.clip(min_sin_dec, -1., 1.))
            max_dec = np.arcsin(np.clip(max_sin_dec, -1., 1.))

            # Solid angles of selected events around each source
            omega = 2. * np.pi * (max_sin_dec - min_sin_dec)
        else:
            min_dec, max_dec = None, None  # No meaning in circle mode
            r = np.arcsin(sin_inj_w)
            omega = np.array(nsrcs * [2 * np.pi * (1. - np.cos(r))])

        assert len(omega) == len(min_dec) == len(max_dec) == nsrcs
        assert np.all((0. < omega) & (omega <= 4. * np.pi))
        return omega, min_dec, max_dec

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
        m = ((MC["dec"] >= self._inj_opts["dec_range"][0]) &
             (MC["dec"] <= self._inj_opts["dec_range"][1]))
        return drop_fields(MC[m], self._mc_names)

    def _check_fit_input(self, srcs, MC):
        """ Check fit input, setup self._provided_data, self._mc_names """
        MC_names = np.concatenate((self._provided_data, self._mc_names))
        for n in MC_names:
            if n not in MC.dtype.names:
                raise ValueError("`MC` array is missing name '{}'.".format(n))

        if (np.any(srcs["dec"] < self._inj_opts["dec_range"][0]) or
                np.any(srcs["dec"] > self._inj_opts["dec_range"][1])):
            raise ValueError("Source position(s) found outside 'dec_range'.")


class HealpyPowerLawFluxInjector(PowerLawFluxInjector):
    """
    Healpy Signal Injector

    Inject signal events not at a fixed source position but according to a
    healpy prior map.
    If fixed source positions are tested in the analysis, this injection should
    decrease sensitivity systematically.
    Injection mode is constrained to ``band`` here.
    """
    def __init__(self, gamma=2., E0=1., inj_opts=None, random_state=None):
        """
        Parameters
        ----------
        gamma, E0 : float
            Postive spectral index ``gamma`` and normalization energy ``E0`` in
            GeV of the assumed unbroken power law that is used to weight the
            simulation events. The power law is

            ..math: \phi(E) = \phi_0 \cdot (E / E0)^{-\gamma}

            Events are weighted to the physics scenario by
            ``w[i] ~ (MC['trueE'][i] / E0)**-gamma * MC['ow'][i]``, where
            ``trueE`` also is in units of GeV.
            (default: gamma = 2., E0 = 1.)
        inj_opts : dict, optional
            Injector options:
            - 'sindec_inj_width', optional: This is different form the
              ``PowerLawFluxInjector`` behaviour because depending on the prior
              localization we may select a different bandwidth for each source.
              Here ``sindec_inj_width`` is the minimum selection bandwidth,
              preventing the band to become too small for very narrow priors.
              It is therefore used in combination with the new setting
              ``inj_sigma``. (default: ``0.035 ~ sin(2°)``)
            - 'inj_sigma', optional: Angular size in prior sigmas around each
              source region from the prior map from which MC events are
              injected. Use in combination with ``sindec_inj_width`` to make
              sure, the injection band is wide enough. (default: 3.)
            - 'dec_range', optional: Global declination interval in which events
              can be injected in rotated coordinates. Events rotated outside are
              dropped even if selected, which drops sensitivity as desired.
              (default: ``[-pi/2, pi/2]``)
        random_state : None, int or np.random.RandomState, optional
            Used as PRNG, see ``sklearn.utils.check_random_state``.
            (default: None)
        """
        req_keys = None
        opt_keys = {"mode": "band", "inj_sigma": 3.}  # Fix band mode for super
        inj_opts = fill_dict_defaults(inj_opts, req_keys, opt_keys,
                                      noleft="use")
        if inj_opts["inj_sigma"] <= 0.:
            raise ValueError("Injection sigma must be > 0.")

        # Defaults for private class attributes
        self._src_map_CDFs = None
        self._NSIDE = None
        self._NPIX = None
        self._pix2ra = None
        self._pix2dec = None
        self._src_idx = None

        # Debug attributes
        self._src_ra = None
        self._src_dec = None

        return super(HealpyPowerLawFluxInjector, self).__init__(
            gamma, E0, inj_opts, random_state)

    def fit(self, srcs, src_maps, MC, livetime):
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
            same names as given in ``provided_data`` and additonal MC truths:
            - 'trueE', float: True event energy in GeV.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator' OneWeight [2]_,
              so it is already divided by ``nevts * nfiles`.
              Units are 'GeV sr cm^2'. Final event weights are obtained by
              multiplying with desired sum flux for nu and anti-nu flux.
        livetimes : float
            Livetime in days of the data, that the simulation ``MC`` represents.

        Notes
        -----
        .. [1] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
        """
        # Check if maps match srcs
        nsrcs = len(srcs)
        if hp.maptype(src_maps) != nsrcs:
                raise ValueError("Not as many prior maps as srcs given.")

        self._NPIX = int(len(src_maps[0]))
        self._NSIDE = hp.npix2nside(self._NPIX)

        # Test if maps are valid PDFs on the unit sphere (m>=0 and sum(m*dA)=1)
        dA = hp.nside2pixarea(self._NSIDE)
        areas = np.array(map(np.sum, src_maps)) * dA
        if not np.allclose(areas, 1.) or np.any(src_maps < 0.):
            raise ValueError("Not all given maps for key are valid PDFs.")

        # Temporary save src maps for the super call. Then del them to save RAM
        self._src_maps = src_maps
        super_out = super(HealpyPowerLawFluxInjector, self).fit(
            srcs, MC, livetime)
        del self._src_maps

        # Pre-compute normalized sampling CDFs from the maps for fast sampling
        CDFs = np.cumsum(src_maps, axis=1)
        self._src_map_CDFs = CDFs / CDFs[:, [-1]]
        assert np.allclose(self._src_map_CDFs[:, -1], 1.)
        assert len(self._src_map_CDFs) == nsrcs
        assert self._src_map_CDFs.shape[1] == self._NPIX

        # Pre-compute pix2ang conversion, directly in ra, dec
        th, phi = hp.pix2ang(self._NSIDE, np.arange(self._NPIX))
        self._pix2dec, self._pix2ra = thetaphi2decra(th, phi)
        self._src_idx = np.empty(nsrcs, dtype=int)

        return super_out

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
        if self._mc_idx is None:
            raise ValueError("Injector has not been filled with MC data yet.")

        if n_samples < 1:
            return np.empty(0, dtype=self._sample_dtype)

        # Sample new source positions from prior maps
        for i, CDFi in enumerate(self._src_map_CDFs):
            self._src_idx[i] = random_choice(self._rndgen, CDF=CDFi, size=None)
        src_ras = self._pix2ra[self._src_idx]
        src_decs = self._pix2dec[self._src_idx]

        # Draw IDs from the whole stored pool of MC events
        sam_idx = random_choice(self._rndgen, self._sample_w_CDF,
                                size=n_samples)
        sam_idx = self._mc_idx[sam_idx]

        # Select events from pool and rotate them to corresponding src positions
        sam_ev = self._MC[sam_idx["ev_idx"]]
        src_idx = sam_idx["src_idx"]
        sam_ev = self._rot_and_strip(src_ras[src_idx], src_decs[src_idx],
                                     sam_ev)

        # Debug purpose
        self._sample_idx = sam_idx
        self._src_ra, self._src_dec = src_ras, src_decs
        return sam_ev

    def _set_solid_angle(self, srcs):
        """
        Setup solid angles of injection area for selected MC events and sources,
        by selecting the injection band depending on the given source prior maps
        and the injection width in sigma.
        """
        nsrcs = len(srcs)
        sin_inj_w = self._inj_opts["sindec_inj_width"]
        decL, decU = self._inj_opts["dec_range"]
        sinL, sinU = np.clip(np.sin(self._inj_opts["dec_range"]), -1., 1.)

        # Get band [min dec, max dec] from n sigma contour of the prior maps
        min_dec, max_dec = np.empty((2, nsrcs), dtype=float)
        for i, map_i in enumerate(self._src_maps):
            decs, _, _ = get_pixel_in_sigma_region(
                map_i, self._inj_opts["inj_sigma"])
            min_dec[i], max_dec[i] = np.amin(decs), np.amax(decs)
            if (max_dec[i] < srcs["dec"][i]) or (srcs["dec"][i] < min_dec[i]):
                raise ValueError(
                    "Source {} not within {} sigma band ".format(
                        i, self._inj_opts["inj_sigma"]) +
                    "of corresponding prior map.")

        # Enlarge bands if they got too narrow
        sin_dec = np.sin(srcs["dec"])
        min_sin_dec = np.minimum(np.sin(min_dec), sin_dec - sin_inj_w)
        max_sin_dec = np.maximum(np.sin(max_dec), sin_dec + sin_inj_w)

        # Clip if we ran over the set dec range
        min_dec = np.arcsin(np.clip(min_sin_dec, sinL, sinU))
        max_dec = np.arcsin(np.clip(max_sin_dec, sinL, sinU))
        assert not np.any((max_dec < srcs["dec"]) | (srcs["dec"] < min_dec))

        # Solid angles of selected events around each source
        min_sin_dec = np.sin(min_dec)
        max_sin_dec = np.sin(max_dec)
        omega = 2. * np.pi * (max_sin_dec - min_sin_dec)
        assert len(omega) == len(min_dec) == len(max_dec) == nsrcs
        assert np.all((0. < omega) & (omega <= 4. * np.pi))
        return omega, min_dec, max_dec


class MultiPowerLawFluxInjector(BaseMultiSignalInjector):
    """
    Collect multiple PowerLawFluxInjector classes, implements collective
    sampling, flux2mu and mu2flux methods.
    """
    def __init__(self, random_state=None):
        self._names = None
        self.rndgen = random_state

    @property
    def names(self):
        return self._names

    @property
    def injs(self):
        return self._injs

    @property
    def provided_data(self):
        return dict_map(lambda key, inj: inj.provided_data, self._injs)

    @property
    def srcs(self):
        return dict_map(lambda key, inj: inj.srcs, self._injs)

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
            (default: ``False``)

        Returns
        -------
        flux : float or array-like
            Source flux in total, or split for each source, in unit(s)
            ``[GeV^-1 cm^-2]``.
        """
        raw_fluxes_per_inj, _, src_theo_w = self._combine_raw_fluxes(self._injs)
        raw_fluxes_sum = sum(list(raw_fluxes_per_inj.values()))
        if per_source:
            mu = mu * src_theo_w
        return mu / raw_fluxes_sum

    def flux2mu(self, flux, per_source=False):
        """
        Calculates the number of events ``mu`` corresponding to a given particle
        flux normalization for the current setup:

        .. math:: \mu = F_0 \sum_i \hat{w}_i

        where :math:`F_0 \sum_i \hat{w}_i` would gives the number of events. The
        weights :math:`w_i` are calculated in :py:meth:`_set_sampling_weights`.

        Combines flux2mu from multiple injectors to a single output.

        Parameters
        ----------
        flux : float
            Total source flux normalization for all sources, in units
            ``[GeV^-1 cm^-2]``.
        per_source : bool, optional
            If ``True`` returns the expected number of events per source by
            splitting the total flux normalization according to the expected
            events per source at detector level. (default: ``False``)

        Returns
        -------
        mu : float or dict
            Expected number of events in total or per source at the detector for
            all injectors combined.
        """
        _out = self._combine_raw_fluxes(self._injs)
        raw_fluxes_per_inj, raw_fluxes_per_src, _ = _out
        if per_source:
            raw_flux = raw_fluxes_per_src
        else:
            raw_flux = sum(list(raw_fluxes_per_inj.values()))
        return flux * raw_flux

    def fit(self, injectors):
        """
        Takes multiple single PowerLawFluxInjector in a dict and manages them.

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

        # For the steady state emitters, all sources need to be equal
        if not self._sources_ok(injectors):
            raise ValueError("Not all source arrays are the same for each " +
                             "injector")

        # Cache sampling weights to distribute the number of requested sampled
        # events to each injector
        raw_fluxes_per_inj, _, _ = self._combine_raw_fluxes(injectors)
        raw_fluxes_sum = sum([rf for rf in raw_fluxes_per_inj.values()])
        self._distribute_weights = dict_map(
            lambda k, rf: rf / raw_fluxes_sum, raw_fluxes_per_inj)
        assert np.isclose(
            sum([w for w in self._distribute_weights.values()]), 1.)

        self._injs = injectors
        self._names = list(self._injs.keys())

        return

    def sample(self, n_samples=1):
        """
        Split n_samples across single injectors and combine to combined sample
        after sampling each sub injector.

        Parameters
        ----------
        n_samples : int, optional
            Number of signal events to sample from all injectors in total.
            (default: 1)

        Returns
        --------
        sam_ev : dict of record-arrays
            Combined samples of all signal injectors. Number of events sampled
            in total might be smaller depending on the sub injector settings.
            If ``n_samples<1`` an empty dict is returned.
        """
        if n_samples < 1:
            return {}

        # Sample per injector by distributing the total samples to each sampler
        # See also: Bohm, Zech: Statistics of weighted poisson, arXiv:1309.1287
        sam_ev = {}
        nsam = 0
        for key in self._names[:-1]:
            # Fake mutlinomial by sampling cascaded binomials to make sure
            # each sampler receives the correct amount of signal to sample
            nsami = self._rndgen.binomial(
                n_samples, self._distribute_weights[key], size=None)
            sam_ev[key] = self._injs[key].sample(n_samples=nsami)
            nsam += nsami
        # The last number of events is determined by sum(nsami) = n_samples
        key = self._names[-1]
        sam_ev[key] = self._injs[key].sample(n_samples=n_samples - nsam)

        return sam_ev

    def _combine_raw_fluxes(self, injectors):
        """
        Renormalize raw fluxes from each injector which can be used to
        distribute the amount of signal to sample across multiple injectors.
        All sources are the same per injector and are in a steady state emission
        scenario, so this differs from the GRB case, where all sources unique
        and non-overlapping in the samples. Because each source is always
        emitting, the combined strength is simply the sum of all samples.

        Parameters
        ----------
        injectors : dict
            Dictionary of signal injector instances.

        Returns
        -------
        raw_fluxes_per_inj : dict of floats
            Raw flux per injector summed over all sources.
        raw_fluxes_per_src : array-like
            Raw fluxes per source summed over all injectors per source in the
            same order as the source record arrays for matching sources.
        src_theo_w : array-like
            Normalized intrinsic source weights, equal for all injectors.
        """
        # Get normalized source weight from one injector
        src_theo_w = injectors[injectors.keys()[0]].srcs["w_theo"]
        src_theo_w = src_theo_w / np.sum(src_theo_w)
        # Get the raw fluxes from each injector and per src per injector
        raw_fluxes = dict_map(
            lambda k, inj: np.array(inj.flux2mu(1., per_source=True)),
            injectors)
        # Sum over injector to get rf per source over all samples
        raw_fluxes_per_src = np.sum([rf for rf in raw_fluxes.values()], axis=0)
        # Sum over sources to get summed rf for each injector
        raw_fluxes_per_inj = dict_map(
            lambda k, rf: np.sum(rf), raw_fluxes)
        return raw_fluxes_per_inj, raw_fluxes_per_src, src_theo_w

    def _sources_ok(self, injectors):
        """
        Checks if all source arrays from the given injectors are equal as
        required by the steady state scenario.

        Parameters
        ----------
        injectors : dict
            Dictionary of signal injector instances.

        Returns
        -------
        sources_equal : bool
        ``True`` if all source record arrays are equal for all injectors,
        ``False`` otherwise.
        """
        srcs = [inj.srcs for inj in injectors.values()]

        for srcs_i in srcs[1:]:
            if not np.array_equal(srcs[0], srcs_i):
                return False

        return True


##############################################################################
# Background injector classes
##############################################################################
class ScrambledBGDataInjector(BaseBGDataInjector):
    """
    Injects background by simply assigning new RA values per trial to saved data
    """
    def __init__(self, random_state=None):
        """
        Parameters
        ----------
        random_state : None, int or np.random.RandomState, optional
            Used as PRNG, see ``sklearn.utils.check_random_state``.
            (default: None)
        """
        self._provided_data = np.array(
            ["dec", "ra", "sigma", "logE"])
        self.rndgen = random_state

        # Defaults for private class variables
        self._X = None
        self._nevts = None

        self._log = logger(name=self.__class__.__name__, level="ALL")

    @property
    def provided_data(self):
        return self._provided_data

    @property
    def srcs(self):
        return None

    def fit(self, X):
        """
        Just store data for resampling.

        Parameters
        ----------
        X : recarray
            Experimental data for BG injection.
        """
        X_names = np.array(X.dtype.names)
        for name in self._provided_data:
            if name not in X_names:
                raise ValueError("`X` is missing name '{}'.".format(name))
        drop = np.isin(X_names, self._provided_data,
                       assume_unique=True, invert=True)
        drop_names = X_names[drop]
        print(self._log.INFO("Dropping names '{}'".format(arr2str(drop_names)) +
                             " from data recarray."))

        self._X = drop_fields(X, drop_names, usemask=False)
        self._nevts = len(self._X)

        return

    def sample(self):
        """
        Resample the whole data set by drawing new RA values.
        """
        self._X["ra"] = self._rndgen.uniform(0, 2. * np.pi, size=self._nevts)
        return self._X


class MultiBGDataInjector(BaseMultiBGDataInjector):
    """
    Container class simply collects all samples from the individual injectors.
    """
    @property
    def names(self):
        return list(self._injs.keys())

    @property
    def injs(self):
        return self._injs

    @property
    def provided_data(self):
        return dict_map(lambda key, inj: inj.provided_data, self._injs)

    @property
    def srcs(self):
        return dict_map(lambda key, inj: inj.srcs, self._injs)

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
                raise ValueError("Injector `{}` ".format(name) +
                                 "is not of type `BaseBGDataInjector`.")

        self._injs = injs

    def sample(self):
        """
        Sample each injector and combine to a dict of recarrays.
        """
        return dict_map(lambda key, inj: inj.sample(), self._injs)
