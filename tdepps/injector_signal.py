# coding: utf-8

from __future__ import division, print_function, absolute_import
from builtins import dict, zip
from future import standard_library
from future.utils import viewkeys
standard_library.install_aliases()

from copy import deepcopy
import numpy as np
from numpy.lib.recfunctions import drop_fields
import scipy.stats as scs
from sklearn.utils import check_random_state
import healpy as hp

from .utils import rotator, power_law_flux, ThetaPhiToDecRa


class SignalInjector(object):
    """
    Signal Injector

    Generate signal events from Monte Carlo data for specific source
    realizations.

    Parameters
    ----------
    gamma : float
        Index of an unbroken power law :math:`E^{-\gamma}` which is used to
        describe the energy flux of signal events.
    mode : string, optional
        One of ``['circle'|'band']``. Selects MC events to inject based
        on their true location:

        - 'circle': Select ``MC`` events in circle around each source.
        - 'band': Select ``MC`` events in a declination band around each source.

        (default: 'band')

    inj_width : float, optional
        Angular size of the regions from which MC events are injected, in
        radians.

        - If ``mode`` is ``'band'``, this is half the width of the declination
          band centered at the source positions in radian.
        - If ``mode`` is ``'circle'`` this is the radius of the circular
          selection region in radian.

        (default: ``np.deg2rad(2)``)

    sin_dec_range : array-like, shape (2), optional
        Boundaries for which injected events are discarded, when their rotated
        coordinates are outside this bounds. Is useful, when a zenith cut is
        used and the PDFs are not defined on the whole sky.
        (default: ``[-1, 1]``)
    random_state : seed, optional
        Turn seed into a ``np.random.RandomState`` instance. See
        ``sklearn.utils.check_random_state``. (default: None)
    """
    def __init__(self, gamma, mode="band", inj_width=np.deg2rad(2),
                 sin_dec_range=[-1., 1.], random_state=None):
        # Public class members (settable only by constructor)
        if (gamma < 1.) or (gamma > 4.):
            raise ValueError("`gamma` in doubtful range, must be in [1, 4].")
        self._gamma = gamma

        if mode not in ["band", "circle"]:
            raise ValueError("`mode` must be one of ['band', 'circle']")
        self._mode = mode

        if (inj_width <= 0.) or (inj_width > np.pi):
            raise ValueError("Injection width must be in (0, pi].")
        self._inj_width = inj_width

        sin_dec_range = np.atleast_1d(sin_dec_range)
        if (sin_dec_range[0] < -1.) or (sin_dec_range[1] > 1.):
            raise ValueError("`sin_dec_range` must be range [a, b] in [-1, 1].")
        if sin_dec_range[0] >= sin_dec_range[1]:
            raise ValueError("`sin_dec_range=[low, high]` must be increasing.")
        self._sin_dec_range = sin_dec_range

        self.rndgen = random_state

        # Defaults for private class variables
        self._mc_arr = None
        self._MC = None
        self._exp_names = None
        self._srcs = None
        self._nsrcs = None

        self._min_dec = None
        self._max_dec = None
        self._sin_dec_range = np.atleast_1d(sin_dec_range)
        self._omega = None

        self._raw_flux = None
        self._sample_w_CDF = None

        self._key2enum = None
        self._enum2key = None  # Handy for debugging, not actually used
        self._SECINDAY = 24. * 60. * 60.

        # Debug flag. Set True and injection bands get calculated as in skylab
        self._skylab_band = False

        return

    # No setters, use the `fit` method for that or create a new object0c
    @property
    def gamma(self):
        return self._gamma

    @property
    def mode(self):
        return self._mode

    @property
    def inj_width(self):
        return self._inj_width

    @property
    def sin_dec_range(self):
        return self._sin_dec_range

    @property
    def mc_arr(self):
        return self._mc_arr

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    def mu2flux(self, mu):
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

        Multiyear compatible when using dicts of samples, see below.

        Parameters
        -----------
        srcs : recarray or dict(name, recarray)
            Source properties as single record array or as dictionary with
            sample names mapped to record arrays. Keys must match keys in
            ``MC``. Each record array must have names:

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

        MC : recarray or dict(name, recarray)
            Either single structured array describing Monte Carlo events or a
            dictionary with sample names mapped to record arrays. Keys must
            match ``srcs``. Each record array must contain names given in
            ``exp_names`` and additonally:

            - 'trueE', float: True event energy in GeV.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator' OneWeight [2]_,
              so it is already divided by ``nevts * nfiles`.
              Units are 'GeV sr cm^2'. Final event weights are obtained by
              multiplying with desired sum flux for nu and anti-nu flux.

        exp_names : tuple of strings
            All names in the experimental data record array used for other
            classes. Must match with the MC record names. ``exp_names`` is
            required to have at least the names:

            - 'ra': Per event right-ascension coordinate in :math:`[0, 2\pi]`.
            - 'sinDec': Per event sinus declination, in :math`[-1, 1]`.
            - 'logE': Per event energy proxy, given in
              :math`\log_{10}(1/\text{GeV})`.
            - 'sigma': Per event positional uncertainty, given in radians.
            - 'timeMJD': Per event times in MJD days.

        Notes
        -----
        .. [1] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
        """
        srcs, MC, exp_names = self._check_fit_input(srcs, MC, exp_names)
        self._exp_names = exp_names
        self._srcs = srcs
        self._nsrcs = {key: len(srcs_i) for key, srcs_i in srcs.items()}

        # Set injection solid angles for all sources
        self._set_solid_angle()

        # Store selected event ids in mc_arr to sample from a single array
        # ev_idx: event ID per sam., src_idx: src ID per sam., enum: sample ID
        # ev_idx : [ 1, 5,11,47,58,66,70,93, ..., 0, 4, 7,12,24,71,86, ...]
        # src_idx: [ 0, 0, 1, 2, 2, 3, 4, 4, ..., 0, 2, 2, 2, 3, 4, 4, ...]
        # enum   : [ 0, 0, 0, 0, 0, 0, 0, 0, ..., 1, 1, 1, 1, 1, 1, 1, ...]
        dtype = [("ev_idx", np.int), ("src_idx", np.int), ("enum", np.int)]
        self._mc_arr = np.empty(0, dtype=dtype)
        # self._MC: Store unique events selected for sampling per sample
        self._MC = dict()

        # Loop all samples
        for key in self._key2enum.keys():
            _nsrcs = self._nsrcs[key]
            _min_dec = self._min_dec[key]
            _max_dec = self._max_dec[key]
            mc_i = MC[key]
            srcs_i = srcs[key]
            # Select events in the injection regions
            if self._mode == "band":
                # Broadcast to mask all srcs at once
                min_decs = _min_dec.reshape(_nsrcs, 1)
                max_decs = _max_dec.reshape(_nsrcs, 1)
                mc_true_dec = mc_i["trueDec"]
                inj_mask = ((mc_true_dec > min_decs) & (mc_true_dec < max_decs))
            else:
                # Compare great circle distance to each src with inj_width
                src_ra = np.atleast_2d(srcs_i["ra"]).reshape(_nsrcs, 1)
                src_dec = np.atleast_2d(srcs_i["dec"]).reshape(_nsrcs, 1)
                mc_true_ra = mc_i["trueRa"]
                mc_true_dec = mc_i["trueDec"]
                cos_dist = (np.cos(src_ra - mc_true_ra) *
                            np.cos(src_dec) * np.cos(mc_true_dec) +
                            np.sin(src_dec) * np.sin(mc_true_dec))
                cos_r = np.cos(self._inj_width)
                inj_mask = (cos_dist > cos_r)

            if not np.any(inj_mask):
                print("Sample '{:d}': No events were selected!".format(key))
                # Nevertheless add empty slice
                self._MC[key] = mc_i[np.any(inj_mask, axis=0)]
                continue

            # Select all at least in one injection region (include overlap)
            total_mask = np.any(inj_mask, axis=0)
            N_unique = np.count_nonzero(total_mask)
            self._MC[key] = mc_i[total_mask]

            # Total number of selected events, including overlap
            core_mask = (inj_mask.T[total_mask]).T  # Remove all non-selected
            n_tot = np.count_nonzero(core_mask)     # Equal to count inj_mask

            # Append all event IDs (include double selceted) to sampling array
            mc_arr = np.empty(n_tot, dtype=dtype)

            # Bookkeeping: Create IDs to access the events in the sampling step
            _core = core_mask.ravel()  # [src1_mask, src2_mask, ...]
            # Unique IDs regarding all selected events per MC sample
            mc_arr["ev_idx"] = np.tile(np.arange(N_unique), _nsrcs)[_core]
            # Same src IDs for each selected evt per src (rows in core_mask)
            mc_arr['src_idx'] = np.repeat(np.arange(_nsrcs),
                                          np.sum(core_mask, axis=1))
            # Repeat enum ID for each sample
            mc_arr["enum"] = self._key2enum[key] * np.ones(n_tot, dtype=np.int)

            self._mc_arr = np.append(self._mc_arr, mc_arr)

            print("Sample '{}': Selected {:d} evts at {:d} sources.".format(
                key, n_tot, _nsrcs))
            print("  # Sources without selected evts: {}".format(
                _nsrcs - np.count_nonzero(np.sum(core_mask, axis=1))))

        if len(self._mc_arr) < 1:
            raise ValueError("No events were selected. Check `inj_width`.")

        print("Selected {:d} events in total".format(len(self._mc_arr)))

        self._set_sampling_weights()

        return

    def sample(self, mean_mu, poisson=True):
        """
        Generator to get sampled events from MC for each source position.

        Parameters
        -----------
        mu : float
            Expectation value of number of events to sample.
        poisson : bool, optional
            If True, sample the actual number of events from a poisson
            distribution with expectation ``mu``. Otherwise the number of events
            is constant in each trial. (default: True)

        Returns
        --------
        num : int
            Number of events sampled in total
        sam_ev : iterator
            Sampled events for each loop iteration, either as simple array or
            as dictionary for each sample.
        idx : array-like
            Indices mapping the sampled events to the injected MC events in
            ``self._MC`` for debugging purposes.
        """
        if self._mc_arr is None:
            raise ValueError("Injector has not been filled with MC data yet.")

        # Only one sample, src memberships are unambigious
        if len(self._key2enum) == 1 and list(self._key2enum.keys())[0] == -1:
            src_ra = self._srcs[-1]["ra"]
            src_dec = self._srcs[-1]["dec"]
            src_t = self._srcs[-1]["t"]
            src_dt = np.vstack((self._srcs[-1]["dt0"], self._srcs[-1]["dt1"])).T

        n = int(np.around(mean_mu))
        while True:
            if poisson:
                n = self._rndgen.poisson(mean_mu, size=None)

            # If n=0 (no events get sampled) return None
            if n < 1:
                yield n, None, None
                continue

            # Draw IDs from the whole pool of events
            # Stripped version to avoid `choice` checks on the sampling weights,
            # which are set up correctly already in `_set_sampling_weights`
            u = np.random.uniform(size=n)
            sam_idx = np.searchsorted(self._sample_w_CDF, u, side="right")
            sam_idx = self._mc_arr[sam_idx]

            # Check which samples have been injected
            enums = np.unique(sam_idx["enum"])

            # TODO: This is unnecessary to do in the while loop, because once
            # the iterator is built, we already know if we have a single or
            # multiple samples
            # If only one sample: return single recarray
            if len(enums) == 1 and enums[0] == -1:
                sam_ev = np.copy(self._MC[enums[0]][sam_idx["ev_idx"]])
                src_idx = sam_idx["src_idx"]
                sam_ev, m = self._rot_and_strip(
                    src_ra[src_idx], src_dec[src_idx], sam_ev, key=-1)
                sam_ev["timeMJD"] = self._sample_times(
                    src_t[src_idx], src_dt[src_idx])[m]
                yield n, sam_ev, sam_idx[m]
                continue

            # Else return same dict structure as used in `fit`
            sam_ev = dict()
            # Total mask to filter out events rotated outside `sin_dec_range`
            idx_m = np.zeros_like(sam_idx, dtype=bool)
            for key, enum in self._key2enum.items():
                if enum in enums:
                    # Get source positions for the correct sample
                    _src_ra = self._srcs[key]["ra"]
                    _src_dec = self._srcs[key]["dec"]
                    _src_t = self._srcs[key]["t"]
                    _src_dt = np.vstack((self._srcs[key]["dt0"],
                                         self._srcs[key]["dt1"])).T
                    # Select events per sample
                    enum_m = (sam_idx["enum"] == enum)
                    idx = sam_idx[enum_m]["ev_idx"]
                    sam_ev_i = np.copy(self._MC[key][idx])
                    # Broadcast corresponding sources for correct rotation
                    src_idx = sam_idx[enum_m]["src_idx"]
                    sam_ev[key], m = self._rot_and_strip(
                        _src_ra[src_idx], _src_dec[src_idx], sam_ev_i, key=key)
                    sam_ev[key]["timeMJD"] = self._sample_times(
                        _src_t[src_idx], _src_dt[src_idx])[m]
                    # Build up the mask for the returned indices 'sam_idx'
                    _idx_m = np.zeros_like(m)
                    _idx_m[m] = True
                    idx_m[enum_m] = _idx_m
                else:
                    drop_names = [ni for ni in self._MC[key].dtype.names if
                                  ni not in self._exp_names[key]]
                    sam_ev[key] = drop_fields(
                        np.empty((0,), dtype=self._MC[key].dtype), drop_names)

            yield n, sam_ev, sam_idx[idx_m]

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
        src_idx = self._mc_arr["src_idx"]

        # Norm over all w_theo in all samples
        w_theo_sum = 0.
        for srcs in self._srcs.values():
            w_theo_sum += np.sum(srcs["w_theo"])
        assert np.isclose(np.sum(np.concatenate([srcs["w_theo"] / w_theo_sum
                                                 for srcs in
                                                 self._srcs.values()])), 1.)

        w = []
        self._raw_flux_per_sample = {}
        for key in self._key2enum.keys():
            srcs = self._srcs[key]
            mc_i = self._MC[key]
            enum_mask = (self._mc_arr["enum"] == self._key2enum[key])

            src_idx = self._mc_arr[enum_mask]["src_idx"]

            # Broadcast src dependent weight parts to each evt
            omega = self._omega[key][src_idx]
            w_theo = srcs["w_theo"][src_idx] / w_theo_sum
            assert (len(omega) == len(w_theo) == np.sum(enum_mask) ==
                    len(self._mc_arr[enum_mask]))

            ev_idx = self._mc_arr[enum_mask]["ev_idx"]
            flux = power_law_flux(mc_i["trueE"][ev_idx], self._gamma)
            w.append(mc_i["ow"][ev_idx] * flux / omega * w_theo)
            self._raw_flux_per_sample[key] = np.sum(w[-1])

        w = np.concatenate(w, axis=0)
        assert len(w) == len(self._mc_arr)

        self._raw_flux = np.sum(w)
        assert np.isclose(np.sum(list(self._raw_flux_per_sample.values())),
                          self._raw_flux)

        # Sampling weight CDF used for injecting events from the whole selection
        self._sample_w_CDF = np.cumsum(w)
        self._sample_w_CDF = self._sample_w_CDF / self._sample_w_CDF[-1]
        assert np.isclose(self._sample_w_CDF[-1], 1.)
        assert np.allclose(self._sample_w_CDF,
                           np.cumsum(w / np.sum(self._raw_flux)))

        return

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

        self._min_dec = {}
        self._max_dec = {}
        self._omega = {}
        for key, _srcs in self._srcs.items():
            if self._mode == "band":
                sin_dec_bandwidth = np.sin(self._inj_width)
                A, B = self._sin_dec_range

                if self._skylab_band:
                    m = (A - B + 2. * sin_dec_bandwidth) / (A - B)
                    b = sin_dec_bandwidth * (A + B) / (B - A)
                    sin_dec = m * np.sin(_srcs["dec"]) + b
                else:
                    sin_dec = np.sin(_srcs["dec"])

                min_sin_dec = np.maximum(A, sin_dec - sin_dec_bandwidth)
                max_sin_dec = np.minimum(B, sin_dec + sin_dec_bandwidth)

                self._min_dec[key] = np.arcsin(np.clip(min_sin_dec, -1., 1.))
                self._max_dec[key] = np.arcsin(np.clip(max_sin_dec, -1., 1.))

                # Solid angles of selected events around each source
                self._omega[key] = 2. * np.pi * (max_sin_dec - min_sin_dec)
                assert (len(self._min_dec[key]) == len(self._max_dec[key]) ==
                        self._nsrcs[key])
            else:
                r = self._inj_width
                _omega = 2 * np.pi * (1. - np.cos(r))
                self._omega[key] = (np.ones(self._nsrcs[key],
                                            dtype=np.float) * _omega)
                assert len(self._omega[key]) == self._nsrcs[key]
        return

    def _rot_and_strip(self, src_ras, src_decs, MC, key):
        """
        Rotate injected event positions to the sources and strip Monte Carlo
        information from the output array.

        The rotation angles to move the true directions to the sources are used
        to rotate the measured positions by the same amount.
        Events rotated outside ``sin_dec_range`` are removed.

        Parameters
        ----------
        src_ras, src_decs : array-like, shape (len(MC))
            Sources equatorial positions in right-ascension in :math:`[0, 2\pi]`
            and declination in :math:`[-\pi/2, \pi/2]`, both given in radians.
            These are the coordinates we rotate on per event in ``MC``.
        MC : record array
            See :py:meth:<SignalInjector.fit>, Parameters.
        key : dict key
            Which MC sample we are looking at.

        Returns
        --------
        ev : structured array
            Array with rotated values, true MC information is deleted
        m : array-like, shape (len(MC))
            Boolean mask, ``False`` for events that got rotated outside the
            ``sin_dec_range`` and are thus filtered out. Must be applied to the
            sampled times.
        """
        MC["ra"], _dec = rotator(MC["trueRa"], MC["trueDec"],
                                 src_ras, src_decs,
                                 MC["ra"], MC["dec"])

        MC["sinDec"] = np.sin(_dec)
        if "dec" in MC.dtype.names:
            MC["dec"] = _dec

        # Remove events that got rotated outside the sin_dec_range
        m = ((MC["sinDec"] >= self._sin_dec_range[0]) &
             (MC["sinDec"] <= self._sin_dec_range[1]))
        MC = MC[m]

        # Remove all names not in experimental data (= remove MC attributes)
        drop_names = [n for n in MC.dtype.names if
                      n not in self._exp_names[key]]
        return drop_fields(MC, drop_names), m

    def _sample_times(self, src_t, dt):
        """
        Sample times uniformly in on-time signal PDF region.

        Parameters
        ----------
        src_t : array-like (nevts)
            Source time in MJD per event.
        dt : array-like, shape (nevts, 2)
            Time window in seconds centered around ``src_t`` in which the signal
            time PDF is assumed to be uniform.

        Returns
        -------
        times : array-like, shape (nevts)
            Sampled times for this trial.
        """
        # Sample uniformly in [0, 1] and scale to time windows per source in MJD
        r = self._rndgen.uniform(0, 1, size=len(src_t))
        times_rel = r * np.diff(dt, axis=1).ravel() + dt[:, 0]

        return src_t + times_rel / self._SECINDAY

    def _check_fit_input(self, srcs, MC, exp_names):
        """
        Check if input to the ``fit`` method is OK. Parameters like in ``fit``.
        """
        # Work consistently with dict. MC is the blueprint for all others
        if not isinstance(MC, dict):
            MC = {-1: MC}
        if not isinstance(srcs, dict):
            srcs = {-1: srcs}
        if not isinstance(exp_names, dict):
            exp_names = {-1: exp_names}

        # Setup mapping from keys to integer enums and vice-versa
        if len(MC.keys()) == 1 and list(MC.keys())[0] == -1:
            # Trivial wrappers for consistency
            self._key2enum = {-1: -1}
        else:
            self._key2enum = {key: enum for key, enum in
                              zip(MC.keys(), np.arange(len(MC)))}
        # For debugging it's handy to simply get the key from an enum
        self._enum2key = {val: key for key, val in self._key2enum.items()}

        # Keys must be equivalent to MC keys
        if viewkeys(MC) != viewkeys(srcs):
            raise ValueError("Keys in `MC` and `srcs` don't match.")
        if viewkeys(MC) != viewkeys(exp_names):
            raise ValueError("Keys in `MC` and `exp_names` don't match.")

        # Check if each recarray has it's required names
        required_names = ["ra", "dec", "t", "dt0", "dt1", "w_theo"]
        for n in required_names:
            for key, srcs_i in srcs.items():
                if n not in srcs_i.dtype.names:
                    raise ValueError("Source recarray '" +
                                     "{}' is missing name '{}'.".format(key, n))

        required_names = ["ra", "sinDec", "logE", "sigma", "timeMJD"]
        for n in required_names:
            for key, exp_ni in exp_names.items():
                if n not in exp_ni:
                    raise ValueError("`exp_names` is missing name " +
                                     "'{}' at key '{}'.".format(n, key))

        for key, mc_i in MC.items():
            MC_names = exp_names[key] + ("trueRa", "trueDec", "trueE", "ow")
            for n in MC_names:
                if n not in mc_i.dtype.names:
                    e = "MC sample '{}' is missing name '{}'.".format(key, n)
                    raise ValueError(e)

        # Check if `sin_dec_range` is OK with the given source positions
        for key, srcs_i in srcs.items():
            if (np.any(np.sin(srcs_i["dec"]) < self._sin_dec_range[0]) |
                    np.any(np.sin(srcs_i["dec"]) > self._sin_dec_range[1])):
                raise ValueError("Source position(s) outside `sin_dec_range` " +
                                 "in `srcs` array for sample {}.".format(key))

        return srcs, MC, exp_names

    def __str__(self):
        """
        Use to print all settings: `>>> print(sig_inj_object)`
        """
        raise NotImplementedError("Needs some fixes for dicts.")
        rep = "Signal Injector object\n"
        rep += "----------------------\n\n"
        rep += "- gamma     : {:.2f}\n".format(self._gamma)
        rep += "- mode      : '{:5s}'\n".format(self._mode)
        rep += "- inj_width : {:.2f}\n".format(self._inj_width)

        if self._MC is None:
            rep += "\nInjector has not been fitted to MC data yet."
        else:
            rep += "\nMC data info:\n"
            rep += "- Selected {:d} events in total\n".format(len(self._mc_arr))
            for enum, mc_i in self._MC.items():
                rep += "- Sample {:d}:\n".format(enum)
                rep += "  + Left events     : {:d}\n".format(len(mc_i))
                rep += "  + Selected Events : {:d}\n".format(np.sum(
                    self._mc_arr["enum"] == enum))
                rep += "  + True E range    : [{:.2f}, {:.2f}] GeV\n".format(
                    np.amin(mc_i["trueE"]), np.amax(mc_i["trueE"]))
                rep += ("  +     log10(E)    : " +
                        "[{:.2f}, {:.2f}] log10(E/GeV)\n".format(
                            np.amin(np.log10(mc_i["trueE"])),
                            np.amax(np.log10(mc_i["trueE"]))))

            rep += "\nSource info:\n"
            rep += "- Number of sources : {:d}\n".format(self._nsrcs)
            for srci in self._srcs:
                rep += "  + RA {:6.2f}°, DEC {:+6.2f}°".format(
                    np.rad2deg(srci["ra"]), np.rad2deg(srci["dec"]))
                rep += ", w_theo: {:5.2f}".format(srci["w_theo"])
                rep += ", dt: [{:.1f}, {:.1f}]s\n".format(srci["dt0"],
                                                          srci["dt1"])

        return rep


class HealpySignalInjector(SignalInjector):
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
    inj_width : float, optional
        This is different form the `SignalInjector` behaviour because depending
        on the prior localization we may select a different bandwidth for each
        source. ``inj_width`` is the minimum selection bandwidth, preventing the
        band to become too small for very narrow priors. Is therefore used in
        combination with the new attribute ``inj_sigma``.
        (default: ``np.deg2rad(2)``)
    inj_sigma : float, optional
        Angular size in sigma around each source region from the prior map from
        which MC events are injected. Use in combination with ``inj_width``
        to make sure, the injection band is wide enough. (default: ``3.``)
    sin_dec_range : array-like, shape (2), optional
        Boundaries for which injected events are discarded, when their rotated
        coordinates are outside this bounds. Is useful, when a zenith cut is
        used and the PDFs are not defined on the whole sky.
        (default: ``[-1, 1]``)
    random_state : seed, optional
        Turn seed into a ``np.random.RandomState`` instance. See
        ``sklearn.utils.check_random_state``. (default: ``None``)
    """
    def __init__(self, gamma, inj_width=np.deg2rad(2), inj_sigma=3.,
                 sin_dec_range=[-1., 1.], random_state=None):
        if inj_sigma <= 0.:
            raise ValueError("Injection sigma must be >0.")
        self._inj_sigma = inj_sigma

        # Set private attributes' default values
        self._src_map_CDFs = None
        self._NSIDE = None
        self._NPIX = None

        # Debug attributes
        self._src_ra = None
        self._src_dec = None

        return super(HealpySignalInjector, self).__init__(
            gamma, "band", inj_width, sin_dec_range, random_state)

    @property
    def inj_sigma(self):
        return self._inj_sigma

    def fit(self, srcs, src_maps, MC, exp_names):
        """
        Fill injector with Monte Carlo events, preselecting events around the
        source positions.

        Multiyear compatible when using dicts of samples, see below.

        Sets up private class variables:

        - ``_min_dec``, ``_max_dec``, dict of arrays: Upper/lower bounds for
          each declination band in radians for each source sample. Size of the
          bands is determined by ``inj_width`` in sigma and the prior width.

        Parameters
        -----------
        srcs : recarray or dict(name, recarray)
            Source properties as single record array or as dictionary with
            sample names mapped to record arrays. Keys must match keys in
            ``MC``. Each record array must have names:

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

        src_maps : array or dict(name, list of array(s))
            List of valid healpy map array(s) per source per sample, all in same
            resolution used as spatial priors for injecting source positions
            each sample step. Maps must be normal space PDFs normalized to
            area equals one on the unit sphere from a positional reconstruction
            to give a probability region of the true source position per source.
        MC : recarray or dict(name, recarray)
            Either single structured array describing Monte Carlo events or a
            dictionary with sample names mapped to record arrays. Keys must
            match ``srcs``. Each record array must contain names given in
            ``exp_names`` and additonally:

            - 'trueE', float: True event energy in GeV.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator' OneWeight [2]_,
              so it is already divided by ``nevts * nfiles`.
              Units are 'GeV sr cm^2'. Final event weights are obtained by
              multiplying with desired sum flux for nu and anti-nu flux.

        exp_names : tuple of strings
            All names in the experimental data record array used for other
            classes. Must match with the MC record names. ``exp_names`` is
            required to have at least the names:

            - 'ra': Per event right-ascension coordinate in :math:`[0, 2\pi]`.
            - 'sinDec': Per event sinus declination, in :math`[-1, 1]`.
            - 'logE': Per event energy proxy, given in
              :math`\log_{10}(1/\text{GeV})`.
            - 'sigma': Per event positional uncertainty, given in radians.
            - 'timeMJD': Per event times in MJD days.

        Notes
        -----
        .. [1] http://software.icecube.wisc.edu/documentation/projects/neutrino-generator/weightdict.html#oneweight
        """
        srcs, MC, exp_names = self._check_fit_input(srcs, MC, exp_names)

        # Check if prior maps are valid healpy maps and all have same resolution
        # and if there is a map for every source
        if not isinstance(src_maps, dict):
            src_maps = {-1: src_maps}
        if viewkeys(src_maps) != viewkeys(MC):
            raise ValueError("Keys in `MC` and `src_maps` don't match.")

        src_maps = deepcopy(src_maps)
        map_lens = []
        for key, maps_i in src_maps.items():
            if hp.maptype(maps_i) == 0:
                # Always use list of maps, not bare arrays
                maps_i = [maps_i]
                src_maps[key] = maps_i
                print("Wapped single bare map for key '{}'.".format(key))
            if hp.maptype(maps_i) != len(srcs[key]):
                raise ValueError("For sample '{}' ".format(key) +
                                 "there are not as many maps as srcs given.")
            map_lens.append(map(len, maps_i))
        map_lens = np.concatenate(map_lens)
        self._NPIX = int(map_lens[0])
        self._NSIDE = hp.npix2nside(self._NPIX)
        if not hp.isnsideok(self._NSIDE):
            raise ValueError("Given `src_maps` don't have proper resolution.")
        if not np.all(map_lens == self._NPIX):
            raise ValueError("Not all given 'src_maps' have the same length.")

        # Test if maps are valid PDFs on the unit sphere (m>=0 and sum(m*dA)=1)
        dA = hp.nside2pixarea(self._NSIDE)
        for key, maps_i in src_maps.items():
            areas = np.array(map(np.sum, maps_i)) * dA
            if not np.allclose(areas, 1.) or np.any(maps_i < 0.):
                raise ValueError("Not all given maps for key '{}'".format(key) +
                                 " are valid PDFs on the unit sphere.")

        # Select the injection band depending on the given source prior maps
        self._min_dec = {}
        self._max_dec = {}
        min_sin_dec_bandwidth = np.sin(self._inj_width)
        A, B = self._sin_dec_range
        Adec, Bdec = np.arcsin(self._sin_dec_range)
        for key, maps_i in src_maps.items():
            # Get band min / max from n sigma prior contour
            self._min_dec[key] = []
            self._max_dec[key] = []
            for i, map_i in enumerate(maps_i):
                min_dec, max_dec = self.get_nsigma_dec_band(map_i,
                                                            self._inj_sigma)
                assert min_dec <= srcs[key]["dec"][i] <= max_dec
                self._min_dec[key].append(min_dec)
                self._max_dec[key].append(max_dec)
            self._min_dec[key] = np.maximum(self._min_dec[key], Adec)
            self._max_dec[key] = np.minimum(self._max_dec[key], Bdec)
            assert not np.any(self._max_dec[key] < srcs[key]["dec"])
            assert not np.any(srcs[key]["dec"] < self._min_dec[key])

            # Check that all bands are larger than requested minimum size
            if self._skylab_band:
                m = (A - B + 2. * min_sin_dec_bandwidth) / (A - B)
                b = min_sin_dec_bandwidth * (A + B) / (B - A)
                sin_dec = m * np.sin(srcs[key]["dec"]) + b
            else:
                sin_dec = np.sin(srcs[key]["dec"])
            min_sin_dec = np.maximum(A, sin_dec - min_sin_dec_bandwidth)
            max_sin_dec = np.minimum(B, sin_dec + min_sin_dec_bandwidth)
            min_dec = np.arcsin(np.clip(min_sin_dec, -1., 1.))
            max_dec = np.arcsin(np.clip(max_sin_dec, -1., 1.))
            assert (len(min_dec) == len(max_dec) ==
                    len(self._min_dec[key]) == len(self._min_dec[key]))
            self._min_dec[key] = np.minimum(self._min_dec[key], min_dec)
            self._max_dec[key] = np.maximum(self._max_dec[key], max_dec)

        # Pre-compute normalized sampling CDFs from the maps for fast sampling
        self._src_map_CDFs = {}
        for key, maps_i in src_maps.items():
            self._src_map_CDFs[key] = []
            for i, map_i in enumerate(maps_i):
                CDF = np.cumsum(map_i)
                self._src_map_CDFs[key].append(CDF / CDF[-1])
                assert np.isclose(self._src_map_CDFs[key][i][-1], 1.)
                assert np.allclose(self._src_map_CDFs[key][i],
                                   np.cumsum(map_i / np.sum(map_i)))
                assert len(self._src_map_CDFs[key][i]) == self._NPIX

        return super(HealpySignalInjector, self).fit(srcs, MC, exp_names)

    @staticmethod
    def get_nsigma_dec_band(pdf_map, sigma=3.):
        """
        Get the ns sigma declination band around a source position from a
        prior healpy normal space PDF map.

        Parameters
        ----------
        pdf_map : array-like
            Healpy PDF map.
        sigma : int, optional
            How many sigmas the band should measure. For sigmas, Wilk's
            theorem is assumed

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

    def sample(self, mean_mu, poisson=True):
        """
        Generator to get sampled events from MC for each source position.
        Each time new source positions are sampled from the prior maps.

        Parameters
        -----------
        mu : float
            Expectation value of number of events to sample.
        poisson : bool, optional
            If True, sample the actual number of events from a poisson
            distribution with expectation ``mu``. Otherwise the number of events
            is constant in each trial. (default: True)

        Returns
        --------
        num : int
            Number of events sampled in total
        sam_ev : iterator
            Sampled events for each loop iteration, either as simple array or
            as dictionary for each sample.
        idx : array-like
            Indices mapping the sampled events to the injected MC events in
            ``self._MC`` for debugging purposes.
        """
        if self._mc_arr is None:
            raise ValueError("Injector has not been filled with MC data yet.")

        # If poisson is `False`, sample fixed integer number of signal events
        if poisson is False:
            n = int(np.around(mean_mu))
            if n < 1:
                raise ValueError("`poisson` is False and `int(mean_mu) < 1` " +
                                 "so there won't be any events sampled.")

        # Precompute pix2ang conversion, directly in ra, dec
        th, phi = hp.pix2ang(self._NSIDE, np.arange(self._NPIX))
        pix2dec, pix2ra = ThetaPhiToDecRa(th, phi)

        # Prepare other fixed elements to save time during sampling
        if len(self._key2enum) == 1 and list(self._key2enum.keys())[0] == -1:
            # Only one sample, src memberships are unambigious
            src_map_CDFs = self._src_map_CDFs[-1]
            src_t = self._srcs[-1]["t"]
            src_dt = np.vstack((self._srcs[-1]["dt0"],
                                self._srcs[-1]["dt1"])).T
            src_idx = np.zeros(self._nsrcs[-1], dtype=int) - 1

            def sample_src_positions():
                """
                Returns a new set of source position sampled from the given
                prior map PDFs.

                Returns
                -------
                ra, dec : array-like
                    New source positions sampled from each prior map.
                """
                for i, src_map_CDF_i in enumerate(src_map_CDFs):
                    u = np.random.uniform(size=None)
                    src_idx[i] = np.searchsorted(src_map_CDF_i, u, side='right')
                return pix2ra[src_idx], pix2dec[src_idx]
        else:
            src_dt = {}
            src_idx = {}
            for key in self._key2enum.keys():
                src_dt[key] = np.vstack((self._srcs[key]["dt0"],
                                         self._srcs[key]["dt1"])).T
                src_idx[key] = np.zeros(self._nsrcs[key], dtype=int) - 1

            def sample_src_positions():
                """
                Returns a new set of source position sampled from the given
                prior map PDFs.

                Returns
                -------
                ra, dec : dict of arrays
                    Returns an array of source positions for each key in the
                    given source map dictionary.
                """
                src_ras = {}
                src_decs = {}
                for key, src_map_CDFs_i in self._src_map_CDFs.items():
                    for i, src_map_CDF_i in enumerate(src_map_CDFs_i):
                        u = np.random.uniform(size=None)
                        src_idx[key][i] = np.searchsorted(src_map_CDF_i, u,
                                                          side='right')
                    src_ras[key] = pix2ra[src_idx[key]]
                    src_decs[key] = pix2dec[src_idx[key]]

                return src_ras, src_decs

        # Create the generator part
        while True:
            if poisson:
                n = self._rndgen.poisson(mean_mu, size=None)

            # If n=0 (no events get sampled) return None
            if n < 1:
                yield n, None, None
                continue

            # Draw IDs from the whole pool of events
            u = np.random.uniform(size=n)
            sam_idx = np.searchsorted(self._sample_w_CDF, u, side="right")
            sam_idx = self._mc_arr[sam_idx]

            # Check which samples have been injected
            enums = np.unique(sam_idx["enum"])

            # Also draw new src position from prior maps
            src_ra, src_dec = sample_src_positions()
            # Debug attributes
            self._src_ra, self._src_dec = src_ra, src_dec

            # If only one sample: return single recarray
            if len(enums) == 1 and enums[0] == -1:
                sam_ev = np.copy(self._MC[enums[0]][sam_idx["ev_idx"]])
                _src_idx = sam_idx['src_idx']
                sam_ev, m = self._rot_and_strip(
                    src_ra[_src_idx], src_dec[_src_idx], sam_ev, key=-1)
                sam_ev["timeMJD"] = self._sample_times(
                    src_t[_src_idx], src_dt[_src_idx])[m]
                yield n, sam_ev, sam_idx[m]
                continue

            # Else return same dict structure as used in `fit`
            sam_ev = dict()
            # Total mask to filter out events rotated outside `sin_dec_range`
            idx_m = np.zeros_like(sam_idx, dtype=bool)
            for key, enum in self._key2enum.items():
                if enum in enums:
                    # Get source positions for the correct sample
                    _src_ra = src_ra[key]
                    _src_dec = src_dec[key]
                    _src_t = self._srcs[key]["t"]
                    _src_dt = src_dt[key]
                    # Select events per sample
                    enum_m = (sam_idx["enum"] == enum)
                    idx = sam_idx[enum_m]["ev_idx"]
                    sam_ev_i = np.copy(self._MC[key][idx])
                    # Broadcast corresponding sources for correct rotation
                    _src_idx = sam_idx[enum_m]["src_idx"]
                    sam_ev[key], m = self._rot_and_strip(
                        _src_ra[_src_idx], _src_dec[_src_idx], sam_ev_i, key)
                    sam_ev[key]["timeMJD"] = self._sample_times(
                        _src_t[_src_idx], _src_dt[_src_idx])[m]
                    # Build up the mask for the returned indices 'sam_idx'
                    _idx_m = np.zeros_like(m)
                    _idx_m[m] = True
                    idx_m[enum_m] = _idx_m
                else:
                    drop_names = [ni for ni in self._MC[key].dtype.names if
                                  ni not in self._exp_names[key]]
                    sam_ev[key] = drop_fields(
                        np.empty((0,), dtype=self._MC[key].dtype), drop_names)

            yield n, sam_ev, sam_idx[idx_m]

    def _set_solid_angle(self):
        """
        Setup solid angles of injection area for selected MC events and sources.

        Overriden to use only 'band' mode and different upper/lower ranges per
        source, as set up in ``fit``. Only sets up the solid angle here, band
        boundaries have been calculated in ``fit`` to not break inheritance.

        Sets up private class variable:

        - ``_omega``, dict of arrays: Solid angle in radians of each injection
          region for each source sample.
        """
        assert self._mode == "band"
        self._omega = {}
        for key, _srcs in self._srcs.items():
            # Solid angles of selected events around each source
            min_sin_dec = np.sin(self._min_dec[key])
            max_sin_dec = np.sin(self._max_dec[key])
            self._omega[key] = 2. * np.pi * (max_sin_dec - min_sin_dec)
            assert np.all(0. < self._omega[key])
            assert np.all(self._omega[key] <= 4. * np.pi)
            assert (len(self._min_dec[key]) == len(self._max_dec[key]) ==
                    len(self._omega[key]) == self._nsrcs[key])
        return
