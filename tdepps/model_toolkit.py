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
from builtins import zip, super
from future import standard_library
standard_library.install_aliases()

import abc
from copy import deepcopy
import numpy as np
from numpy.lib.recfunctions import drop_fields
import scipy.optimize as sco
import scipy.stats as scs
from sklearn.utils import check_random_state
import healpy as hp
from astropy.time import Time as astrotime

try:
    from awkde import GaussianKDE as KDE
except ImportError as e:
    print(e)
    print("KDE injector is not available because package `awkde` is missing.")

from .utils import (random_choice, fill_dict_defaults, ThetaPhiToDecRa,
                    rotator, spl_normed, fit_spl_to_hist)


##############################################################################
# Signal injector classes
##############################################################################
class SignalFluenceInjector(object):
    """
    Signal Fluence Injector

    Inject signal events from Monte Carlo data weighted to a specific fluence
    model and inject at given source positions. Fluence is assumed to be per
    "burst" as in GRB models, so the fluence is not depending on the duration
    of a source's time window. Only spatial rotation is done.

    Parameters
    ----------
    model : callable
        Function of true energy ``f(MC['trueE'])``, describing the model flux
        that will be injected from. Values are used to weight events to a
        physics scenario by ``w[i] ~ f(MC['trueE'][i] * MC['ow'][i]``. Make
        sure the model fits to the units of the true energy and the OneWeights.
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
    random_state : seed, optional
        Turn seed into a ``np.random.RandomState`` instance. See
        ``sklearn.utils.check_random_state``. (default: None)
    """
    def __init__(self, model, mode="band", sindec_inj_width=0.035,
                 dec_range=None, random_state=None):
        if not callable(model):
            raise TypeError("`model` must be a function `f(trueE)`.")
        self._model = model

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

        # Defaults for private class variables
        self._MC = None
        self._mc_names = None
        self._exp_names = None
        self._nsrcs = None

        self._min_dec = None
        self._max_dec = None
        self._omega = None

        self._raw_flux = None
        self._sample_w_CDF = None
        self._w_theo_norm = None

        # Debug attributes
        self._sample_idx = None
        self._skylab_band = False

        return

    # No setters, use the `fit` method for that or create a new object
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

    @property
    def srcs(self):
        return self._srcs

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    def mu2flux(self, mu, per_source=False):
        """
        Convert a given number of events ``mu`` to a corresponding particle
        flux normalization :math:`F_0` in units [GeV^-1 cm^-2].

        The connection between :math:`F_0` and the number of events ``mu`` is:

        .. math:: F_0 = \mu / \sum_i \hat{w}_i

        where :math:`F_0 \sum_i \hat{w}_i` would gives the number of events. The
        weights :math:`w_i` are calculated in :py:meth:`_set_sampling_weights`.

        Parameters
        ----------
        mu : float
            Expectation for number of events.

        Returns
        -------
        flux : float or array-like
            Total flux for all sources in unit ``[GeV^-1 cm^-2]``.
        """
        if per_source:
            # Split the total flux according to the thoeretical source weights
            return self._w_theo_norm * mu / self._raw_flux
        return mu / self._raw_flux

    def flux2mu(self, flux):
        """
        Calculates the number of events ``mu`` corresponding to a given particle
        flux for the current setup:

        .. math:: \mu = F_0 \sum_i \hat{w}_i

        where :math:`F_0 \sum_i \hat{w}_i` would gives the number of events. The
        weights :math:`w_i` are calculated in :py:meth:`_set_sampling_weights`.
        """
        return flux * self._raw_flux

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
        keep_names = self._mc_arr + self._exp_names
        drop_names = [n for n in self._MC.dtype.names if n not in keep_names]
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

        print("Selected {:d} evts at {:d} sources.".format(n_tot, nsrcs))
        print("  - Sources without selected evts: {}".format(
            nsrcs - np.count_nonzero(np.sum(core_mask, axis=1))))

        self._set_sampling_weights()

    def sample(self, n_samples=1):
        """
        Get sampled events from stored MC for each  stored source position.

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

        # Draw IDs from the whole stored pool of MC events
        sam_idx = random_choice(self._rndgen, self._sample_w_CDF, n=n_samples)
        sam_idx = self._mc_arr[sam_idx]

        # Select events from pool and rotate them to corresponding src positions
        sam_ev = self._MC[sam_idx["ev_idx"]]
        src_idx = sam_idx["src_idx"]
        sam_ev = self._rot_and_strip(self._srcs["ra"][src_idx],
                                     self._srcs["dec"][src_idx],
                                     sam_ev)
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
        nsrcs = len(self._nsrcs)

        # Normalize w_theos to prevent overestimatiing injected fluxes
        self._w_theo_norm = self._srcs["w_theo"] / np.sum(self._srcs["w_theo"])
        assert np.isclose(np.sum(self._w_theo_norm, 1.))

        # Broadcast solid angles and w_theo to corrsponding sources for each evt
        omega = self._omega[src_idx]
        w_theo = self._w_theo_norm[src_idx]
        flux = self._model(mc["trueE"])
        assert len(omega) == len(w_theo) == len(flux)

        w = mc["ow"] * flux / omega * w_theo
        assert len(self._mc_arr) == len(w)

        self._raw_flux = np.sum(w)
        _raw_flux_per_source = ([np.sum(w[src_idx == j]) for j in
                                np.arange(nsrcs)])
        assert np.allclose(_raw_flux_per_source,
                           self._w_theo_norm * self._raw_flux)
        assert np.isclose(np.sum(self._raw_flux_per_source), self._raw_flux)

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
        for n in ["ra", "dec", "w_theo"]:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` array is missing name '{}'.".format(n))

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
    gamma : float
        Index of an unbroken power law :math:`E^{-\gamma}` which is used to
        describe the energy flux of signal events.
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
    random_state : seed, optional
        Turn seed into a ``np.random.RandomState`` instance. See
        ``sklearn.utils.check_random_state``. (default: ``None``)
    """
    def __init__(self, gamma, sindec_inj_width=0.035, inj_sigma=3.,
                 dec_range=None, random_state=None):
        if inj_sigma <= 0.:
            raise ValueError("Injection sigma must be >0.")
        self._inj_sigma = inj_sigma

        # Set private attributes' default values
        self._src_map_CDFs = None
        self._NSIDE = None
        self._NPIX = None
        self._pix2ra = None
        self._pix2dec = None

        # Debug attributes
        self._src_ra = None
        self._src_dec = None

        return super(HealpySignalFluenceInjector, self).__init__(
            gamma, "band", sindec_inj_width, dec_range, random_state)

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
        self._pix2dec, self._pix2ra = ThetaPhiToDecRa(th, phi)

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
        dec, _ = ThetaPhiToDecRa(th, phi)
        return np.amin(dec), np.amax(dec)

    def sample(self, n_samples=1):
        """
        Generator to get sampled events from MC for each source position.
        Each time new source positions are sampled from the prior maps.

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


##############################################################################
# Background injector classes
##############################################################################
class BGDataInjector(object):
    """
    General Purpose Data Injector Base Class

    Base class for generating events from a given data record array.
    Classes must implement methods:

    - ``fun``
    - ``sample``

    Class object then provides public methods:

    - ``fun``
    - ``sample``

    Parameters
    ----------
    random_state : None, int or np.random.RandomState, optional
        Turn seed into a ``np.random.RandomState`` instance. (default: None)

    Example
    -------
    >>> # Example for a special class which resamples directly from an array
    >>> from tdepps.model_toolkit import DataGPInjector as inj
    >>> # Generate some test data
    >>> n_evts, n_features = 100, 3
    >>> X = np.random.uniform(0, 1, size=(n_evts, n_features))
    >>> X = np.core.records.fromarrays(X.T, names=["logE", "dec", "sigma"])
    >>> # Fit injector and let it resample from the pool of testdata
    >>> inj.fit(X)
    >>> sample = inj.sample(n_samples=1000)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, random_state=None):
        self.rndgen = random_state
        # Setup private defaults
        self._X_names = None
        self._n_features = None

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    @abc.abstractmethod
    def fit(self, X):
        """
        Build the injection model with the provided data.

        Parameters
        ----------
        X : record-array
            Data named array.
        """
        pass

    @abc.abstractmethod
    def sample(self, n_samples=1):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (default: 1)

        Returns
        -------
        X : record-array
            Generated samples from the fitted model. Has the same names as the
            given record-array X in `fit`.
        """
        pass

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

    def _check_X_names(self, X):
        """ Check if given input ``X`` is valid and extract names. """
        try:
            _X_names = X.dtype.names
        except AttributeError:
            raise AttributeError("`X` must be a record array with dtype.names.")

        self._n_features = len(_X_names)
        self._X_names = _X_names

        return X


class KDEBGDataInjector(BGDataInjector):
    """
    Adaptive Bandwidth Kernel Density Background Injector.

    Parameters
    ----------
    kde : awkde.GaussianKDE
        Adaptive width KDE model. If an already fitted model is given,
        th ``fit`` step can be called with ``X=None`` to avoid refitting,
        which can take some time when many points with adaptive kernels are
        used.
    random_state : None, int or np.random.RandomState, optional
        Turn seed into a ``np.random.RandomState`` instance. (default: None)
    """
    def __init__(self, kde, random_state=None):
        super(KDEBGDataInjector, self).__init__(random_state)
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


class ResampleBGDataInjector(BGDataInjector):
    """
    Data injector resampling weighted data events from the given array.
    """
    def __init__(self, random_state=None):
        super(ResampleBGDataInjector, self).__init__(random_state)

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
        if self._n_features is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # Return empty array with all keys, when n_samples < 1
        if n_samples < 1:
            dtype = [(n, float) for n in self._X_names]
            return np.empty(0, dtype=dtype)

        # Choose uniformly from given data
        idx = random_choice(rndgen=self._rndgen, CDF=self._CDF, n=n_samples)
        return self._X[idx]


class Binned3DBGDataInjector(BGDataInjector):
    """
    Injector binning up data space in `[a x b x c]` bins with equal statistics
    and then sampling uniformly from those bins. Only works with 3 dimensional
    data arrays.
    """
    def __init__(self, random_state=None):
        super(Binned3DBGDataInjector, self).__init__(random_state)

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

        # Sample uniform in [0, 1] to decide where each point lies in the bins
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


##############################################################################
# Rate function classes to fit a BG rate model
##############################################################################
class RateFunction(object):
    """
    Rate Function Base Class

    Base class for rate functions describing time dependent background
    rates. Rate function must be interpretable as a PDF and must not be
    negative.

    Classes must implement methods:

    - ``fun``
    - ``integral``
    - ``sample``
    - ``_get_default_seed``

    Class object then provides public methods:

    - ``fun``
    - ``integral``
    - ``fit``
    - ``sample``

    Parameters
    ----------
    random_state : seed, optional
        Turn seed into a ``np.random.RandomState`` instance. See
        ``sklearn.utils.check_random_state``. (default: None)
    """
    __metaclass__ = abc.ABCMeta
    _SECINDAY = 24. * 60. * 60.

    def __init__(self, random_state=None):
        self.rndgen = random_state
        # Get set when fitted
        self._bf_pars = None
        self._bf_fun = None
        self._bf_int = None

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    @property
    def bf_pars(self):
        return self._bf_pars

    @property
    def bf_fun(self):
        return self._bf_fun

    @property
    def bf_int(self):
        return self._bf_int

    @abc.abstractmethod
    def fun(self, t, pars):
        """
        Returns the rate in Hz at a given time t in MJD.

        Parameters
        ----------
        t : array-like, shape (nevts)
            MJD times of experimental data.
        pars : tuple
            Further parameters the function depends on.

        Returns
        -------
        rate : array-like
            Rate in Hz for each time ``t``.
        """
        pass

    @abc.abstractmethod
    def integral(self, t, trange, pars):
        """
        Integral of rate function in intervals trange around source times t.

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows ``[[t0, t1], ...]`` in seconds around each time ``t``.
        pars : tuple
            Further parameters :py:meth:`fun` depends on.

        Returns
        -------
        integral : array-like, shape (nsrcs)
            Integral of :py:meth:`fun` within given time windows ``trange``.
        """
        pass

    @abc.abstractmethod
    def sample(self, n_samples, t, trange, pars):
        """
        Generate random samples from the rate function for multiple source times
        and time windows.

        Parameters
        ----------
        n_samples : array-like, shape (nsrcs)
            Number of events to sample per source.
        t : array-like, shape (nsrcs)
            MJD times of sources to sample around.
        trange : array-like, shape(nsrcs, 2)
            Time windows ``[[t0, t1], ...]`` in seconds around each time ``t``.
        pars : tuple
            Parameters :py:meth:`fun` depends on.

        Returns
        -------
        times : list of arrays, len (nsrcs)
            Sampled times in MJD of background events per source. If
            ``n_samples`` is 0 for a source, an empty array is placed at that
            position.
        """
        pass

    @abc.abstractmethod
    def _get_default_seed(self, t, trange, w):
        """
        Default seed values for the specifiv RateFunction fit.

        Parameters
        ----------
        t : array-like
            MJD times of experimental data.
        rate : array-like, shape (len(t))
            Rates at given times `t` in Hz.
        w : array-like, shape(len(t)), optional
            Weights for least squares fit: :math:`\sum_i (w_i * (y_i - f_i))^2`.
            (default: None)

        Returns
        -------
        p0 : tuple
            Seed values for each parameter the specific :py:class:`RateFunction`
            uses as a staerting point in the :py:meth:`fit`.
        """
        pass

    def fit(self, t, rate, p0=None, w=None, minopts=None):
        """
        Fits the function parameters to experimental data using a weighted
        least squares fit.

        Parameters
        ----------
        t : array-like
            MJD times of experimental data.
        rate : array-like, shape (len(t))
            Rates at given times `t` in Hz.
        p0 : tuple, optional
            Seed values for the fit parameters. If None, default ones are used,
            that may or may not work. (default: None)
        w : array-like, shape(len(t)), optional
            Weights for least squares fit: :math:`\sum_i (w_i * (y_i - f_i))^2`.
            (default: None)
        minopts : dict, optional
            Minimizer options passed to
            ``scipy.optimize.minimize(method='L-BFGS-B')``. Default settings if
            given as ``None`` or for missing keys are
            ``{'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': int(1e3)}``.

        Returns
        -------
        res : scipy.optimize.OptimizeResult
            Dict wrapper with fot results.
        """
        if w is None:
            w = np.ones_like(rate)

        if p0 is None:
            p0 = self._get_default_seed(t, rate, w)

        # Setup minimizer options
        required_keys = []
        opt_keys = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": int(1e3)}
        minopts = fill_dict_defaults(minopts, required_keys, opt_keys)

        res = sco.minimize(fun=self._lstsq, x0=p0, args=(t, rate, w),
                           method="L-BFGS-B", options=minopts)
        self._bf_fun = (lambda t: self.fun(t, res.x))
        self._bf_int = (lambda t, trange: self.integral(t, trange, res.x))
        self._bf_pars = res.x
        return res

    def _lstsq(self, pars, *args):
        """
        Weighted leastsquares loss: :math:`\sum_i (w_i * (y_i - f_i))^2`

        Parameters
        ----------
        pars : tuple
            Fitparameter for :py:meth:`fun` that gets fitted.
        args : tuple
            Fixed values `(t, rate, w)` for the loss function:

            - t, array-like: See :py:meth:`RateFunction.fit`, Parameters
            - rate, array-like, shape (len(t)): See :py:meth:`RateFunction.fit`,
              Parameters
            - w, array-like, shape(len(t)): See :py:meth:`RateFunction.fit`,
              Parameters

        Returns
        -------
        loss : float
            The weighted least squares loss for the given `pars` and `args`.
        """
        t, rate, w = args
        fun = self.fun(t, pars)
        return np.sum((w * (rate - fun))**2)

    def _transform_trange_mjd(self, t, trange):
        """
        Transform time window to MJD and check on correct shapes

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows `[[t0, t1], ...]` in seconds around each time `t`.

        Returns
        -------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows `[[t0, t1], ...]` in MJD around each time `t`.
        """
        t = np.atleast_1d(t)
        nsrcs = len(t)
        # Proper braodcasting to process multiple srcs at once
        t = t.reshape(nsrcs, 1)
        trange = np.atleast_2d(trange).reshape(nsrcs, 2)
        return t, t + trange / self._SECINDAY


class SinusRateFunction(RateFunction):
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

    def fit(self, t, rate, srcs, p0=None, w=None, minopts=None):
        """
        Fit the rate model to discrete points ``(t, rate)``. Cache source values
        for fast sampling.

        Parameters
        ----------
        srcs : record-array
            Must have names ``'t', 'dt0', 'dt1'`` describing the time intervals
            around the source times to sample from.
        """
        bf_pars = super(SinusRateFunction, self).fit(t, rate, p0, w, minopts)

        # Cache max function values in the fixed source intervals for sampling
        required_names = ["t", "dt0", "dt1"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` recarray is missing name " +
                                 "'{}'.".format(n))
        _dts = np.vstack((srcs["dt0"], srcs["dt1"])).T
        _, _trange = self._transform_trange_mjd(srcs["t"], _dts)
        self._fmax = self._calc_fmax(_trange[:, 0], _trange[:, 1], pars=bf_pars)
        self._trange = _trange

        return bf_pars

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

    def _make_params(self, pars):
        """
        Check which parameters are fixed and insert them where needed to build
        a full parameter set.

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
    def __init__(self, random_state=None):
        super(SinusFixedConstRateFunction, self).__init__(random_state)

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


class ConstantRateFunction(RateFunction):
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


##############################################################################
# Misc helper methods
##############################################################################
def power_law_flux(trueE, gamma=2., phi0=1., E0=1.):
    """
    Returns the unbroken power law flux :math:`\sim \phi_0 (E/E_0)^{-\gamma}`
    where the normlaization is summer over both particle types nu and anti-nu.
    Default have no physical meaning. Unit must be adapted to used weights.

    Parameters
    ----------
    trueE : array-like
        True particle energy.
    gamma : float
        Positive power law index. (default: 2.)
    phi0 : float
        Flux normalization. Resembles value at ``E0``. (default: 1.)
    E0 : float
        Support point at which ``phi(E0) = phi0``. (default: 1.)

    Returns
    -------
    flux : array-like
        Per nu+anti-nu particle flux :math:`\phi \sim E^{-\gamma}`.
    """
    return phi0 * (trueE / E0)**(-gamma)


def fit_time_dec_rate_models(timesMJD, sin_decs, run_dict, sin_dec_bins,
                             rate_rebins, minimizer_opts=None):
    """
    Fit amplitude and baseline of a ``SinusFixedRateFunction`` model to data in
    given declination bins to build a time and dec dependent rate model.

    Parameters
    ----------
    timesMJD : array-like
        Experimental per event event times in MJD days.
    sin_decs : array-like
        Experimental per event ``sin(declination)`` values.
    run_dict : dictionary
        Dictionary with run information, matching the experimental data. Can be
        obtained from ``create_run_dict``.
    sin_dec_bins : array-like
        Explicit bin edges in ``sin(dec)`` used to bin ``sin_decs``.
    rate_rebins : array-like
        Explicit bin edges used to rebin the rates before fitting the model to
        achieve more stable fit conditions.
    minimizer_opts : dict, optional
        Options passed to the minimizer in the rate model fitter.
        (default: ``None``)

    Returns
    -------
    best_pars : record-array
        Best fit parameters amplitude and baseline per sinus declination bin,
        has names ``'amp', 'base'``.
    std_devs : record-array
        Standard deviation of amplitude and baseline per sinus declination bin,
        has names ``'amp', 'base'``.
    """
    timesMJD = np.atleast_1d(timesMJD)
    sin_decs = np.atleast_1d(sin_decs)
    sin_dec_bins = np.atleast_1d(sin_dec_bins)
    rate_rebins = np.atleast_1d(rate_rebins)

    if not isinstance(run_dict, dict):
        raise TypeError("`run_dict` must be a dictionary.")

    # Create rate function to model the data: Use only amplitude and baseline as
    # free parameters in the rate model
    p_fix = 365.
    t0_fix = np.amin(timesMJD)
    rate_func = SinusFixedRateFunction(p_fix=p_fix, t0_fix=t0_fix)

    nbins = len(sin_dec_bins) - 1
    names = ["amp", "base"]
    best_pars = np.empty((nbins, ), dtype=[(n, float) for n in names])
    std_devs = np.empty_like(best_pars)
    for i, (lo, hi) in enumerate(zip(sin_dec_bins[:-1], sin_dec_bins[1:])):
        mask = (sin_decs >= lo) & (sin_decs < hi)
        rate_rec = make_rate_records(timesMJD[mask], run_dict, eps=0.,
                                     all_in_err=False)

        # Rebinned fit ((f-y)/std). Weights should be approx. std dev to obtain
        # useful weights for the spline parametrization.
        rates, bins, rates_std, _ = rebin_rate_rec(
            rate_rec, bins=rate_rebins, ignore_zero_runs=True)
        mids = 0.5 * (bins[:-1] + bins[1:])

        fitres = rate_func.fit(mids, rates, x0=None, w=1. / rates_std)

        for j, n in enumerate(names):
            best_pars[n][i] = fitres.x[j]
            std_devs[n][i] = np.sqrt(np.diag(fitres.hess_inv))[j]

    return best_pars, std_devs


def make_time_dec_rate_model_splines(sin_dec_bins, best_pars, std_devs,
                                     spl_norm=None):
    """
    Fit a spline to best fit values from ``fit_time_dec_rate_models``. Build a
    spline model to continiously describe the rate model parameters in
    declination.

    Parameters
    ----------
    sin_dec_bins : array-like
        Explicit bin edges in ``sin(dec)`` used to create the binned best fits.
    best_pars : record-array
        Best fit parameters amplitude and baseline per sinus declination bin,
        has names ``'amp', 'base'``.
    std_devs : record-array
        Standard deviation of amplitude and baseline per sinus declination bin,
        has names ``'amp', 'base'``.
    spl_norm : dict, optional
        If not ``None`` must be a dict with keys ``'amplitude'``, ``'baseline'``
        containing the normalization constant used for each spline, so that the
        integral over the whole declination range yields ``spl_norm`` in each
        case. (Default: ``None``)

    Returns
    -------
    param_splines : dict
        Dictionary with keys ``'amp'``, ``'base'`` containing ``spl_normed``
        objects, which describe the amplitude and baseline parameters used to
        describe the experimental data rates with a time and declination
        dependent rate model.
    """
    def spl_normed_factory(spl, lo, hi, norm):
        """ Renormalize spline, so ``int_lo^hi renorm_spl dx = norm`` """
        return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)

    if spl_norm is not None:
        if not isinstance(spl_norm, dict):
            raise ValueError("`spl_norm` must be a dictionary.")

    param_splines = {}
    lo, hi = sin_dec_bins[0], sin_dec_bins[-1]
    sin_dec_norm = np.diff(sin_dec_bins)
    names = ["amp", "base"]
    for i, (bp, n, std) in enumerate(zip(best_pars, names, std_devs)):
        # Use normalized amplitude and baseline in units HZ/dec
        bp = bp / sin_dec_norm
        std = std / sin_dec_norm
        spl = fit_spl_to_hist(bp, sin_dec_bins, std)

        if spl_norm is not None:
            # Renormalization can only be done with amp and baseline because the
            # are additive across the disjunct sindec bins for the rate model.
            norm = spl_norm[n]

        param_splines[n] = spl_normed_factory(spl, lo=lo, hi=hi, norm=norm)

    return param_splines


def create_run_dict(run_list, filter_runs=None):
    """
    Create a dict of lists from a run list in JSON format (list of dicts).

    Parameters
    ----------
    run_list : list of dicts
            Dict made from a good run runlist snapshot from [1]_ in JSON format.
            Must be a list of single runs of the following structure

                [{
                  "good_tstart": "YYYY-MM-DD HH:MM:SS",
                  "good_tstop": "YYYY-MM-DD HH:MM:SS",
                  "run": 123456, ...,
                  },
                 {...}, ..., {...}]

            Each run dict must at least have keys ``'good_tstart'``,
            ``'good_tstop'`` and ``'run'``. Times are given in iso formatted
            strings and run numbers as integers as shown above.
    filter_runs : function, optional
        Filter function to remove unwanted runs from the goodrun list.
        Called as ``filter_runs(dict)``. Function must operate on a single
        run dictionary element strucutred as shown above. If ``None``, every run
        is used. (default: ``None``)

    Returns
    -------
    run_dict : dict
        Dictionary with run attributes as keys. The values are stored in arrays
        for each key.
    """
    # run_list must be a list of dicts (one dict to describe one run)
    if not np.all(map(lambda item: isinstance(item, dict), run_list)):
        raise TypeError("Not all entries in 'run_list' are dicts.")

    required_names = ["good_tstart", "good_tstop", "run"]
    for i, item in enumerate(run_list):
        for key in required_names:
            if key not in item.keys():
                raise KeyError("Runlist item '{}' ".format(i) +
                               "is missing required key '{}'.".format(key))

    # Filter to remove unwanted runs
    run_list = list(filter(filter_runs, run_list))

    # Convert the run list of dicts to a dict of arrays for easier handling
    run_dict = dict(zip(run_list[0].keys(),
                        zip(*[r.values() for r in run_list])))

    # Dict keys were not necessarly sorted, so sort the new lists after run id
    srt_idx = np.argsort(run_dict["run"])
    for k in run_dict.keys():
        run_dict[k] = np.atleast_1d(run_dict[k])[srt_idx]

    # Convert and add times in MJD float format
    run_dict["good_start_mjd"] = astrotime(run_dict["good_tstart"],
                                           format="iso").mjd
    run_dict["good_stop_mjd"] = astrotime(run_dict["good_tstop"],
                                          format="iso").mjd
    # Add runtimes in MJD days
    run_dict["runtime_days"] = (run_dict["good_stop_mjd"] -
                                run_dict["good_start_mjd"])

    return run_dict


def make_rate_records(T, run_dict, eps=0., all_in_err=False):
    """
    Creates time bins ``[start_MJD_i, stop_MJD_i]`` for each run in ``run_dict``
    and bins the experimental data to calculate the rate for each run. Data
    selection should match the used run list to give reasonable results.

    Parameters
    ----------
    T : array_like, shape (n_samples)
        Per event times in MJD days of experimental data.
    run_dict
        Dictionary with run attributes as keys and values stored in arrays for
        each key. Must at least have keys ``'good_tstart'``, ``'good_tstop'``
        and ``'run'``. Can be created by method ``create_run_dict``.
    eps : float, optional
        Extra margin in mirco seconds added to run bins to account for possible
        floating point errors during binning. (default: 0.)
    all_in_err : bool, optional
        If ``True`` raises an error if not all times ``T`` have been sorted in
        the run bins defined in ``rund_dict``. (default: False)

    Returns
    -------
    rate_rec : recarray, shape(nruns)
        Record array with keys:

        - "run" : int, ID of the run.
        - "rate" : float, rate in Hz in this run.
        - "runtime" : float, livetime of this run in MJD days.
        - "start_mjd" : float, MJD start time of the run.
        - "stop_mjd" : float, MJD end time of the run.
        - "nevts" : int, numver of events in this run.
        - "rates_std" : float, ``sqrt(nevts) / runtime`` scaled poisson standard
          deviation of the rate in Hz in this run.
    """
    _SECINDAY = 24. * 60. * 60.
    T = np.atleast_1d(T)
    if eps < 0.:
        raise ValueError("`eps` must be >0.")

    # Store events in bins with run borders, broadcast for fast masking
    start_mjd = run_dict["good_start_mjd"]
    stop_mjd = run_dict["good_stop_mjd"]
    run = run_dict["run"]

    # Histogram time values in each run manually, eps = micro sec extra margin
    eps *= 1.e-3 / _SECINDAY
    mask = ((T[:, None] >= start_mjd[None, :] - eps) &
            (T[:, None] <= stop_mjd[None, :] + eps))

    evts = np.sum(mask, axis=0)  # Events per run. mask: (dim(T), dim(runs))
    tot_evts = np.sum(evts)      # All selected
    assert tot_evts == np.sum(mask)

    # Sometimes runlists given for the used samples don't seem to include all
    # events correctly contradicting to what is claimed on wiki pages
    if all_in_err and tot_evts > len(T):
        # We seem to have double counted times, try again with eps = 0
        dble_m = (np.sum(mask, axis=0) > 1.)  # Each time in more than 1 run
        dble_t = T[dble_m]
        idx_dble = np.where(np.isin(T, dble_t))[0]
        err = ("Double counted times. Try a smaller `eps` or check " +
               "if there are overlapping runs in `run_dict`.\n")
        err += "  Events selected : {}\n".format(tot_evts)
        err += "  Events in T     : {}\n".format(len(T))
        err += "  Leftover times in MJD:\n    {}\n".format(", ".join(
            ["{}".format(ti) for ti in dble_t]))
        err += "  Indices:\n    {}".format(", ".join(
            ["{}".format(i) for i in idx_dble]))
        raise ValueError()
    elif all_in_err and tot_evts < len(T):
        # We didn't get all events into our bins
        not_cntd_m = (~np.any(mask, axis=0))  # All times not in any run
        left_t = T[not_cntd_m]
        idx_left = np.where(np.isin(T, left_t))[0]
        err = ("Not all events in `T` were sorted in runs. If this is " +
               "intended, please remove them beforehand.\n")
        err += "  Events selected : {}\n".format(tot_evts)
        err += "  Events in T     : {}\n".format(len(T))
        err += "  Leftover times in MJD:\n    {}\n".format(", ".join(
            ["{}".format(ti) for ti in left_t]))
        err += "  Indices:\n    {}".format(", ".join(
            ["{}".format(i) for i in idx_left]))
        raise ValueError(err)

    # Normalize to rate in Hz
    runtime = stop_mjd - start_mjd
    rate = evts / (runtime * _SECINDAY)

    # Calculate poisson sqrt(N) stddev for scaled rates
    rate_std = np.sqrt(evts) / (runtime * _SECINDAY)

    # Create record-array
    names = ["run", "rate", "runtime", "start_mjd",
             "stop_mjd", "nevts", "rate_std"]
    types = [int, np.float, np.float, np.float, np.float, int, np.float]
    dtype = [(n, t) for n, t in zip(names, types)]

    a = np.vstack((run, rate, runtime, start_mjd, stop_mjd, evts, rate_std))
    rate_rec = np.core.records.fromarrays(a, dtype=dtype)
    return rate_rec


def rebin_rate_rec(rate_rec, bins, ignore_zero_runs=True):
    """
    Rebin rate per run information. The binning is right exclusice on the start
    time of an run:
      ``bins[i] <= rate_rec["start_mjd"] < bins[i+1]``.
    Therefore the bin borders are not 100% exact, but the included rates are.
    New bin borders adjustet to start at the first included run are returned, to
    miniimize the error, but we still shouldn't calculate the event numbers by
    multiplying bin widths with rates.

    Parameters
    ----------
    rate_rec : record-array
        Rate information as coming out of RunlistBGRateInjector._rate_rec.
        Needs names ``'start_mjd', 'stop_mjd', 'rate'``.
    bins : array-like or int
        New time binning used to rebin the rates.
    ignore_zero_runs : bool, optional
        If ``True`` runs with zero events are ignored. This method of BG
        estimation doesn't work well, if we have many zero events runs because
        the baseline gets biased towards zero. If this is an effect of the
        events selection then a different method should be used. (Default: True)

    Returns
    -------
    rates : array-like
        Rebinned rates per bin.
    bins : array-like
        Adjusted bins so that the left borders always start at the first
        included run and the last right bin at the end of the last included run.
    rate_std : array-like
        Poisson ``sqrt(N)`` standard error of the rates per bin.
    deadtime : array-like
        How much livetime is 'dead' in the given binning, because runs do not
        start immideiately one after another or there are bad runs that got
        filtered out. Subtracting the missing livetime from the bin width
        enables us to use the resulting time to recreate the event numbers.
    """
    _SECINDAY = 24. * 60. * 60.
    rates = rate_rec["rate"]
    start = rate_rec["start_mjd"]
    stop = rate_rec["stop_mjd"]

    bins = np.atleast_1d(bins)
    if len(bins) == 1:
        # Use min max and equidistant binning if only a number is given
        bins = np.linspace(np.amin(start), np.amax(stop), int(bins[0]) + 1)

    new_bins = np.empty_like(bins)
    rate = np.empty(len(bins) - 1, dtype=float)
    rate_std = np.empty(len(bins) - 1, dtype=float)
    livetime_per_bin = np.empty_like(rate)

    assert np.allclose(rate_rec["nevts"],
                       rate_rec["rate"] * (stop - start) * _SECINDAY)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (lo <= start) & (start < hi)
        if ignore_zero_runs:
            mask = mask & (rates > 0)
        livetime_per_bin[i] = np.sum(stop[mask] - start[mask])
        # New mean rate: sum(all events in runs) / sum(real livetimes in runs)
        nevts = np.sum(rates[mask] * (stop[mask] - start[mask]))
        rate[i] = nevts / livetime_per_bin[i]
        rate_std[i] = np.sqrt(nevts / _SECINDAY) / livetime_per_bin[i]
        # Adapt bin edges
        new_bins[i] = np.amin(start[mask])
    new_bins[-1] = np.amax(stop[mask])

    deadtime = np.diff(new_bins) - livetime_per_bin
    return rate, np.atleast_1d(new_bins), rate_std, deadtime
