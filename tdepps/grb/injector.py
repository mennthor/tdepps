# coding: utf-8

"""
Collection of base implementations for a GRB style, time dependent analysis.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from numpy.lib.recfunctions import drop_fields
from numpy.core.records import fromarrays
import healpy as hp

from ..base import BaseSignalInjector, BaseMultiSignalInjector
from ..base import BaseBGDataInjector, BaseMultiBGDataInjector
from ..base import BaseTimeSampler, BaseRateFunction
from ..utils import random_choice
from ..utils import fit_spl_to_hist, make_time_dep_dec_splines, spl_normed
from ..utils import arr2str, fill_dict_defaults, logger, dict_map
from ..utils import rotator, thetaphi2decra, get_pixel_in_sigma_region
from ..utils import make_equdist_bins


##############################################################################
# Signal injector classes
##############################################################################
class SignalFluenceInjector(BaseSignalInjector):
    """
    Signal Fluence Injector

    Inject signal events from Monte Carlo data weighted to a specific flux model
    and inject at given source positions. Flux is assumed to be per "burst", not
    depending on the duration of a source's time window.

    Events are injected by rotation true event positions from a diffuse MC set
    to the source locations. Events are sampled weighted using an external
    energy dependent flux model. Time is also sampled via an external module.
    """
    def __init__(self, flux_model, time_sampler, inj_opts=None,
                 random_state=None):
        """
        Parameters
        ----------
        flux_model : callable
            Function of true energy ``f(MC['trueE'])``, describing the model
            flux that will be injected from. Values are used to weight events to
            a physics scenario by ``w[i] ~ f(MC['trueE'][i] * MC['ow'][i]``.
            Model must fit to the units of true energy and the event weights.
        time_sampler : TimeSampler instance
            Time sampler for sampling the signal times in addition to the
            spatial rotation part done by the injector.
        inj_opts : dict, optional
            Injector options:
            - 'sindec_inj_width', optional: Size in ``sin(angle)`` of the region
              from which MC events are selected and rotated to each source. See
              'mode' for how the region is chosen corresponding to
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
        try:
            flux_model(1.)  # Standard units are E = 1GeV
        except Exception:
            raise TypeError("`flux_model` must be a function `f(trueE)`.")

        if not isinstance(time_sampler, BaseTimeSampler):
            raise ValueError("`time_sampler` must have type `BaseTimeSampler`.")

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

        self._flux_model = flux_model
        self._time_sampler = time_sampler
        self._inj_opts = inj_opts
        self._provided_data = np.array(
            ["time", "dec", "ra", "sigma", "logE"])
        self._mc_names = np.array(["trueRa", "trueDec", "trueE", "ow"])
        self._srcs = None
        self.rndgen = random_state

        # Defaults for private class attributes
        self._MC = None
        self._mc_idx = None
        self._src_dt = None
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
            (default: ``False``)

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

    def fit(self, srcs, MC):
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
            self._MC, self._mc_idx, srcs, omega)

        self._srcs = srcs
        self._src_dt = np.vstack((srcs["dt0"], srcs["dt1"])).T

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

        # Sample times from time model
        sam_ev["time"] = self._time_sampler.sample(
            src_t=self._srcs["time"][src_idx], src_dt=self._src_dt[src_idx])

        # Debug purpose
        self._sample_idx = sam_idx
        return sam_ev

    def _set_sampling_weights(self, MC_pool, mc_idx, srcs, omega):
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
        mc = MC_pool[mc_idx["ev_idx"]]
        src_idx = mc_idx["src_idx"]

        # Normalize w_theos to prevent overestimatiing injected fluxes
        w_theo_norm = srcs["w_theo"] / np.sum(srcs["w_theo"])
        assert np.isclose(np.sum(w_theo_norm), 1.)

        # Broadcast solid angles and w_theo to corrsponding sources for each evt
        omega = omega[src_idx]
        w_theo = w_theo_norm[src_idx]
        flux = self._flux_model(mc["trueE"])
        assert len(omega) == len(w_theo) == len(flux)

        w = mc["ow"] * flux / omega * w_theo
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


class HealpySignalFluenceInjector(SignalFluenceInjector):
    """
    Healpy Signal Injector

    Inject signal events not at a fixed source position but according to a
    healpy prior map.
    If fixed source positions are tested in the analysis, this injection should
    decrease sensitivity systematically.
    Injection mode is constrained to ``band`` here.
    """
    def __init__(self, flux_model, time_sampler, inj_opts=None,
                 random_state=None):
        """
        Parameters
        ----------
        flux_model : callable
            Function of true energy ``f(MC['trueE'])``, describing the model
            flux that will be injected from. Values are used to weight events to
            a physics scenario by ``w[i] ~ f(MC['trueE'][i] * MC['ow'][i]``.
            Model must fit to the units of true energy and the event weights.
        time_sampler : TimeSampler instance
            Time sampler for sampling the signal times in addition to the
            spatial rotation part done by the injector.
        inj_opts : dict, optional
            Injector options:
            - 'sindec_inj_width', optional: This is different form the
              ``SignalFluenceInjector`` behaviour because depending on the prior
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

        return super(HealpySignalFluenceInjector, self).__init__(
            flux_model, time_sampler, inj_opts, random_state)

    def fit(self, srcs, src_maps, MC):
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
        super_out = super(HealpySignalFluenceInjector, self).fit(srcs, MC)
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

        # Sample times from time model
        sam_ev["time"] = self._time_sampler.sample(
            src_t=self._srcs["time"][src_idx], src_dt=self._src_dt[src_idx])

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


class MultiSignalFluenceInjector(BaseMultiSignalInjector):
    """
    Collect multiple SignalFluenceInjector classes, implements collective
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
        raw_fluxes, _, w_theos = self._combine_raw_fluxes(self._injs)
        raw_fluxes_sum = np.sum(list(raw_fluxes.values()))
        if per_source:
            return dict_map(lambda key, wts: mu * wts / raw_fluxes_sum, w_theos)
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
            (default: ``False``)

        Returns
        -------
        mu : float or dict
            Expected number of event in total or per source and sample at the
            detector.
        """
        raw_fluxes, raw_fluxes_per_src, _ = self._combine_raw_fluxes(self._injs)
        if per_source:
            return dict_map(lambda key, rfs: flux * rfs, raw_fluxes_per_src)
        return flux * np.sum(list(raw_fluxes.values()))

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
        raw_fluxes, _, _ = self._combine_raw_fluxes(injectors)
        raw_fluxes_sum = sum([rf for rf in raw_fluxes.values()])
        self._distribute_weights = dict_map(lambda key, rf: rf / raw_fluxes_sum,
                                            raw_fluxes)
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

        Parameters
        ----------
        injectors : dict
            Dictionary of signal injector instances.

        Returns
        -------
        raw_fluxes_per_sam : dict
            Summed raw flux per injector after renormalizing the theo weights.
        raw_fluxes_per_src : dict of arrays
            Raw fluxes per injector and per source after renormalizing the theo
            weights.
        w_renorm : dict of arrays
            Renormalized theo weights per sample per source.
        """
        # Normalize theo weights for each sample first
        weights = dict_map(lambda key, inj: inj.srcs["w_theo"], injectors)
        w_sum_per_sam = dict_map(lambda key, val: np.sum(val), weights)
        w_norm = dict_map(lambda key, wts: wts / w_sum_per_sam[key], weights)
        # Remove normalized theo weights from raw fluxes per sample
        raw_fluxes_per_src = dict_map(
            lambda key, inj: inj.flux2mu(1., per_source=True) / w_norm[key],
            injectors)
        assert np.all([len(w_norm[key]) == len(rf) for key, rf in
                       raw_fluxes_per_src.items()])
        # Globally renormalize theo weights over all samples
        w_sum = np.sum(list(w_sum_per_sam.values()))
        w_renorm = dict_map(lambda key, wts: wts / w_sum, weights)
        # Renormalize fluxes per sample per source with renormalized weights
        raw_fluxes_per_src = dict_map(lambda key, rf: rf * w_renorm[key],
                                      raw_fluxes_per_src)
        # Combined fluxes per sample are used for sampling weights
        raw_fluxes_per_sam = dict_map(lambda key, rf: np.sum(rf),
                                      raw_fluxes_per_src)
        return raw_fluxes_per_sam, raw_fluxes_per_src, w_renorm


##############################################################################
# Time sampler
##############################################################################
class UniformTimeSampler(BaseTimeSampler):
    def __init__(self, random_state=None):
        """
        Samples events times uniformly distributed in a time window around a
        source.

        Parameters
        ----------
        random_state : None, int or np.random.RandomState, optional
            Used as PRNG, see ``sklearn.utils.check_random_state``.
            (default: None)
        """
        self.rndgen = random_state

    def sample(self, src_t, src_dt):
        """
        Sample times uniformly in on-time signal PDF region.

        Parameters
        ----------
        src_t : array-like (nevts)
            Source time in MJD per event used to transform the sampled relativ
            event times to absolute MJD times around each source.
        src_dt : array-like, shape (nevts, 2)
            Time window in seconds centered around ``src_t`` in which the signal
            time PDF is assumed to be uniform.

        Returns
        -------
        times : array-like, shape (nevts)
            Sampled MJD times for this trial.
        """
        # Sample uniformly in [0, 1] and scale to time windows per source in MJD
        r = self._rndgen.uniform(0, 1, size=len(src_t))
        times_rel = r * np.diff(src_dt, axis=1).ravel() + src_dt[:, 0]

        return src_t + times_rel / self._SECINDAY


class SplineTimeSampler(BaseTimeSampler):
    def __init__(self, spl, bins, random_state=None):
        """
        Samples event times from a given spline modeling the time distribution
        relativ to the sources' times assumed at ``t=0``. Sampling is done using
        an empirical CDF.

        Parameters
        ----------
        spl : scipy.interpolate.UnivariateSpline instance
            Spline used to describe the time distribution to inject. The spline
            should describe the distribution relativ to the source's time at
            ``t=0`` and using time in seconds.
        bins : array-like
            Explicit bin edges used to create the empirical CDF by sampling the
            given spline's integral function at the bin edges. This also defines
            how much the sampled values can vary, as only the bin edges can be
            sampled. ``bins[0]`` and ``bins[-1]`` define the definition range of
            the PDF spline.
        random_state : None, int or np.random.RandomState, optional
            Used as PRNG, see ``sklearn.utils.check_random_state``.
            (default: None)

        """
        # Cache the CDF
        bins = np.atleast_1d(bins)
        cdf = np.array([spl.integral(bins[0], ti) for ti in bins])

        self._spl = spl
        self._bins = bins
        self._cdf = cdf / cdf[-1]

        self.rndgen = random_state

    def sample(self, src_t):
        """
        Sample times from given spline in on-time signal PDF region.

        Parameters
        ----------
        src_t : array-like (nevts)
            Source time in MJD per event used to transform the sampled relativ
            event times to absolute MJD times around each source.

        Returns
        -------
        times : array-like, shape (nevts)
            Sampled MJD times for this trial.
        """
        # Sample uniformly in [0, 1] and scale to time windows per source in MJD
        r = self._rndgen.uniform(0, 1, size=len(src_t))
        idx = np.searchsorted(self._cdf, r, side="right") - 1
        times_rel = self._bins[idx]

        return src_t + times_rel / self._SECINDAY


##############################################################################
# Background injector classes
##############################################################################
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
    def __init__(self, inj_opts, random_state=None):
        """
        Parameters
        ----------
        inj_opts : dict
            Injector options:
            - 'sindec_bins': Explicit bin edges used to bin the data in
              sinus(dec) to fit a rate model for each bin.
            - 'rate_rebins': Explicit bin edges used to rebin the rate data
              before fitting the rate model to improve fit stability.
            - 'spl_s', optional: Smoothing factor >0 or None, describes how much
              fitted splines are allowed to stick to data.
              ``scipy.interpolate.UnivariateSpline`` for more info.
              (default: ``None``)
            - 'n_scan_bins', optional: Number of bins used to scan the rate
              model fir chi2 landscape to obtain errors for the spline fit.
              (default: 50)
            - 'n_data_evts_min' : int, optional
              Number of events that must be left in any bin while building the
              allsky ``sin(dec)`` histogram for the declination sampling
              weights. (default: 100)
        random_state : None, int or np.random.RandomState, optional
            Used as PRNG, see ``sklearn.utils.check_random_state``.
            (default: None)
        """
        # Check injector options
        req_keys = ["sindec_bins", "rate_rebins"]
        opt_keys = {"spl_s": None, "n_scan_bins": 50, "n_data_evts_min": 100}
        inj_opts = fill_dict_defaults(inj_opts, req_keys, opt_keys)
        inj_opts["sindec_bins"] = np.atleast_1d(inj_opts["sindec_bins"])
        inj_opts["rate_rebins"] = np.atleast_1d(inj_opts["rate_rebins"])
        if inj_opts["spl_s"] is not None and inj_opts["spl_s"] < 0:
            raise ValueError("'spl_s' must be `None` or >= 0.")
        if inj_opts["n_scan_bins"] < 20:
            raise ValueError("'n_scan_bins' should be > 20 for proper scans.")
        if inj_opts["n_data_evts_min"] < 1:
            raise ValueError("'n_data_evts_min' must > 0.")

        self._provided_data = np.array(
            ["time", "dec", "ra", "sigma", "logE"])
        self._sample_dtype = [(n, float) for n in self._provided_data]
        self._inj_opts = inj_opts
        self.rndgen = random_state

        # Defaults for private class variables
        self._X = None
        self._srcs = None
        self._nsrcs = None
        self._nb = None
        self._sample_CDFs = None

        # Debug info
        self._spl_info = None
        self._sample_idx = None

        self._log = logger(name=self.__class__.__name__, level="ALL")

    @property
    def provided_data(self):
        return self._provided_data

    @property
    def srcs(self):
        return self._srcs

    @property
    def inj_opts(self):
        return self._inj_opts.copy()

    def fit(self, X, srcs, run_list):
        """
        Take data, MC and sources and build injection models. This is the place
        to actually stitch together a custom injector from the toolkit modules.

        Parameters
        ----------
        X : recarray
            Experimental data for BG injection.
        srcs : recarray
            Source information.
        run_list : list of dicts
            List of dicts made from a good run runlist snapshot from [1]_. Must
            be a list of single runs of the following structure

                [{
                  "good_tstart": "YYYY-MM-DD HH:MM:SS",
                  "good_tstop": "YYYY-MM-DD HH:MM:SS",
                  "run": 123456, ...,
                  },
                 {...}, ..., {...}]

            Each run dict must at least have keys ``'good_tstart'``,
            ``'good_tstop'`` and ``'run'``. Times are given in iso formatted
            strings and run numbers as integers as shown above.
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

        _out = self._setup_data_injector(X, srcs, run_list)
        self._nb, self._sample_CDFs, self._spl_info = _out

        self._X = drop_fields(X, drop_names, usemask=False)
        self._srcs = srcs
        self._nsrcs = len(srcs)

        return

    def sample(self, debug=False):
        """
        Get a complete data sample for one trial.

        1. Get expected nb per source from allsky rate spline
        2. Sample times from allsky rate splines
        3. Sample same number of events from data using the CDF per source
           for the declination distribution
        4. Combine to a single recarray X
        """
        # Get number of BG events to sample per source in this trial
        evts_per_src = self._rndgen.poisson(self._nb)
        # Sample times from rate function for all sources
        times = self._spl_info["allsky_rate_func"].sample(evts_per_src)

        sam = []
        ev_idx = []
        src_idx = []
        for j in range(self._nsrcs):
            nevts = evts_per_src[j]
            if nevts > 0:
                # Resample dec, logE, sigma from exp data with each source CDF
                idx = random_choice(self._rndgen, CDF=self._sample_CDFs[j],
                                    size=nevts)
                sam_i = self._X[idx]
                # Sample missing ra uniformly
                sam_i["ra"] = self._rndgen.uniform(0., 2. * np.pi, size=nevts)
                # Append times
                sam_i["time"] = times[j]
                # Debug purpose
                ev_idx.append(idx)
                src_idx.append(nevts * [j])
            else:
                sam_i = np.empty(0, dtype=self._sample_dtype)
            sam.append(sam_i)

        # Debug purpose
        if debug:
            try:
                self._sample_idx = fromarrays(
                    [np.concatenate(ev_idx), np.concatenate(src_idx)],
                    dtype=[("ev_idx", float), ("src_idx", float)])
            except ValueError:
                self._sample_idx = np.empty(0, dtype=[("ev_idx", float),
                                            ("src_idx", float)])

        return np.concatenate(sam)

    def _setup_data_injector(self, X, srcs, run_list):
        """
        Create a time and declination dependent background model.

        Fit rate functions to time dependent rate in sindec bins. Normalize PDFs
        over the sindec range and fit splines to the fitted parameter points to
        continiously describe a rate model for a declination. Then choose a
        specific source time and build weights to inject according to the sindec
        dependent rate PDF from the whole pool of BG events.

        Returns
        -------
        nb : array-like
            Expected background events per source.
        sampling_CDF : array-like, shape (nsrcs, nevts)
            Sampling weight CDF per source for fast random choice sampling.
        spl_info : dict
            Collection of spline and rate fit information. See
            ``util.spline.make_time_dep_dec_splines`` for detailed info.
        """
        # Get sindec rate spline for each source, averaged over its time window
        print(self._log.INFO("Create time dep sindec splines."))
        sin_dec_bins = self._inj_opts["sindec_bins"]
        sin_dec_splines, spl_info = make_time_dep_dec_splines(
            X=X, srcs=srcs, run_list=run_list, sin_dec_bins=sin_dec_bins,
            rate_rebins=self._inj_opts["rate_rebins"],
            spl_s=self._inj_opts["spl_s"],
            n_scan_bins=self._inj_opts["n_scan_bins"])

        # Cache expected nb for each source from allsky rate func integral
        src_t = np.atleast_1d(srcs["time"])
        src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T
        nb = spl_info["allsky_rate_func"].integral(
            src_t, src_trange, spl_info["allsky_best_params"])
        assert len(nb) == len(src_t)

        # Normalize sindec splines to be a PDF in sindec for sampling weights
        def spl_normed_factory(spl, lo, hi, norm):
            """ Renormalize spline, so ``int_lo^hi renorm_spl dx = norm`` """
            return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)

        lo, hi = sin_dec_bins[0], sin_dec_bins[-1]
        sin_dec_pdf_splines = []
        for spl in sin_dec_splines:
            sin_dec_pdf_splines.append(spl_normed_factory(spl, lo, hi, norm=1.))

        # Make sampling CDFs to sample sindecs per source per trial
        # First a PDF spline to estimate intrinsic data sindec distribution
        ev_sin_dec = np.sin(X["dec"])
        _bins = make_equdist_bins(
            ev_sin_dec, lo, hi, weights=None,
            min_evts_per_bin=self._inj_opts["n_data_evts_min"])
        # Spline is interpolating to cover the data densitiy as fine as possible
        # because for resampling we divide by the initial densitiy.
        hist = np.histogram(ev_sin_dec, bins=_bins, density=True)[0]
        data_spl = fit_spl_to_hist(hist, bins=_bins, w=None, s=0)[0]
        data_spl = spl_normed_factory(data_spl, lo, hi, norm=1.)
        print(self._log.INFO("Made {} bins for allsky hist".format(len(_bins))))

        # Build sampling weights from PDF ratios
        sample_w = np.empty((len(sin_dec_pdf_splines), len(ev_sin_dec)),
                            dtype=float)
        _vals = data_spl(ev_sin_dec)
        for i, spl in enumerate(sin_dec_pdf_splines):
            sample_w[i] = spl(ev_sin_dec) / _vals

        # Cache fixed sampling CDFs for fast random choice
        CDFs = np.cumsum(sample_w, axis=1)
        sample_CDFs = CDFs / CDFs[:, [-1]]

        spl_info["sin_dec_splines"] = sin_dec_splines
        spl_info["sin_dec_pdf_splines"] = sin_dec_pdf_splines
        spl_info["data_sin_dec_pdf_spline"] = data_spl
        spl_info["sample_weights"] = sample_w

        return nb, sample_CDFs, spl_info


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
    def __init__(self, random_state=None):
        self.rndgen = random_state

        self._fmax = None
        self._trange = None
        # Just to have some info encoded in the class which params we have
        self._PARAMS = np.array(["amplitude", "period", "toff", "baseline"])

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
        required_names = ["time", "dt0", "dt1"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` recarray is missing name " +
                                 "'{}'.".format(n))
        _dts = np.vstack((srcs["dt0"], srcs["dt1"])).T
        _, _trange = self._transform_trange_mjd(srcs["time"], _dts)
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
        self._PARAMS = self._PARAMS[self._fit_idx]

    def fun(self, t, pars):
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).fun(t, pars)

    def integral(self, t, trange, pars):
        pars = self._make_params(pars)
        return super(SinusFixedRateFunction, self).integral(t, trange, pars)

    def _get_default_seed(self, t, rate, w):
        """ Same default seeds as SinusRateFunction, but drop fixed params. """
        seed = super(SinusFixedRateFunction, self)._get_default_seed(t, rate, w)
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
        super(SinusFixedConstRateFunction, self).__init__(p_fix, t0_fix,
                                                          random_state)

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
    def __init__(self, random_state=None):
        self.rndgen = random_state
        self._trange = None
        self._PARAMS = np.array(["baseline"])

    def fit(self, rate, srcs, w=None):
        """ Cache source values for sampling. Fit is the weighted average """
        if w is None:
            w = np.ones_like(rate)

        required_names = ["time", "dt0", "dt1"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` recarray is missing name " +
                                 "'{}'.".format(n))
        _dts = np.vstack((srcs["dt0"], srcs["dt1"])).T
        _, self._trange = self._transform_trange_mjd(srcs["time"], _dts)

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
