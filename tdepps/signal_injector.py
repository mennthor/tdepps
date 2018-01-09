# coding: utf-8

from __future__ import division, print_function, absolute_import
from builtins import dict, int, zip
from future import standard_library
from future.utils import viewkeys
standard_library.install_aliases()

import numpy as np
from numpy.lib.recfunctions import drop_fields
import scipy.stats as scs
from sklearn.utils import check_random_state
import healpy as hp

from .utils import rotator, power_law_flux, ThetaPhi2DecRa


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

    inj_width : float, optinal
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

        if (sin_dec_range[0] < -1.) or (sin_dec_range[1] > 1.):
            raise ValueError("`sin_dec_range` must be range [a, b] in [-1, 1].")
        self._sin_dec_range = sin_dec_range

        self._mc_arr = None
        self.rndgen = random_state

        # Defaults for private class variables (later set ones get None)
        self._MC = None
        self._exp_names = None
        self._srcs = None
        self._nsrcs = None

        self._min_dec = None
        self._max_dec = None
        self._sin_dec_range = np.atleast_1d(sin_dec_range)
        self._omega = None

        self._raw_flux = None
        self._sample_w = None

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
        if not isinstance(MC, dict):  # Work consistently with dicts
            MC = {-1: MC}
            self._keymap = {-1: -1}  # Trivial wrappers for consistency
            self._enummap = {-1: -1}
        else:  # Map dict names to int keys for use as indices and vice-versa
            self._keymap = {key: i for key, i in zip(srcs.keys(),
                                                     np.arange(len(srcs)))}
            self._enummap = {enum: key for key, enum in self._keymap.items()}

        if not isinstance(srcs, dict):  # Work consistently with dicts
            srcs = {-1: srcs}
        else:  # Keys must be equivalent to MC keys
            if viewkeys(MC) != viewkeys(srcs):
                raise ValueError("Keys in `MC` and `srcs` don't match.")

        # Check each recarray's names
        required_names = ["ra", "dec", "t", "dt0", "dt1", "w_theo"]
        for n in required_names:
            for key, srcs_i in srcs.items():
                if n not in srcs_i.dtype.names:
                    raise ValueError("Source recarray '" +
                                     "{}' is missing name '{}'.".format(key, n))

        if not isinstance(exp_names, dict):
            exp_names = {-1: exp_names}
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

        self._exp_names = exp_names

        # Set injection solid angles for all sources
        self._srcs = srcs
        self._nsrcs = {key: len(srcs_i) for key, srcs_i in srcs.items()}
        self._set_solid_angle()

        # Check if sin_dec_range is OK with the used source positions
        for key, srcs_i in srcs.items():
            if (np.any(np.sin(srcs_i["dec"]) < self._sin_dec_range[0]) |
                    np.any(np.sin(srcs_i["dec"]) > self._sin_dec_range[1])):
                raise ValueError("Source position(s) outside `sin_dec_range` " +
                                 "in `srcs` array for sample {}.".format(key))

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
        for key in self._keymap.keys():
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
                inj_mask = cos_dist > cos_r

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
            n_tot = np.count_nonzero(core_mask)  # Equal to count inj_mask

            # Append all events to sampling array
            mc_arr = np.empty(n_tot, dtype=dtype)

            # Bookkeeping: Create id to access the events in the sampling step
            _core = core_mask.ravel()
            # Unique id for selected events per sample and per src
            mc_arr["ev_idx"] = np.tile(np.arange(N_unique), _nsrcs)[_core]
            # Same src id for each selected evt per src (rows in core_mask)
            mc_arr['src_idx'] = np.repeat(np.arange(_nsrcs),
                                          np.sum(core_mask, axis=1))
            # Repeat enum id for each sample
            mc_arr["enum"] = self._keymap[key] * np.ones(n_tot, dtype=np.int)

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

        # Only one sample, src positions are unambigious
        if len(self._keymap) == 1:
            src_ra = self._srcs[-1]["ra"]
            src_dec = self._srcs[-1]["dec"]
            src_t = self._srcs[-1]["t"]
            src_dt = np.vstack((self._srcs[-1]["dt0"], self._srcs[-1]["dt1"])).T

        n = int(np.around(mean_mu))
        while True:
            if poisson:
                n = self._rndgen.poisson(mean_mu, size=None)  # Returns scalar

            # If n=0 (no events get sampled) return None
            if n < 1:
                yield n, None, None
                continue

            # Draw IDs from the whole pool of events
            sam_idx = self._rndgen.choice(self._mc_arr, size=n,
                                          p=self._sample_w)
            enums = np.unique(sam_idx["enum"])

            # If only one sample: return single recarray
            if len(enums) == 1 and enums[0] < 0:
                sam_ev = np.copy(self._MC[enums[0]][sam_idx["ev_idx"]])
                src_idx = sam_idx['src_idx']
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
            for enum in self._enummap.keys():
                key = self._enummap[enum]
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
        for key in self._keymap.keys():
            srcs = self._srcs[key]
            mc_i = self._MC[key]
            enum_mask = (self._mc_arr["enum"] == self._keymap[key])

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

        # Sampling weights used for injecting events from the whole selection
        self._sample_w = w / self._raw_flux
        assert np.isclose(np.sum(self._sample_w), 1.)

        return

    def _set_solid_angle(self):
        """
        Setup solid angles of injection area for selected MC events and sources.

        For a given set of source positions and an injection mode, we need to
        calculate the solid angle per source from which events where injected
        to be able to correctly weight the injected events to get a flux.

        Sets up private class varaibles:

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


class SystematicSignalInjector(SignalInjector):
    """
    Inject signal events not at a fixed source position but according to a
    healpy prior map.
    If fixed source positions are tested in the analysis, this injection should
    decrease sensitivity systematically.
    """
    def fit(self, srcs, src_maps, MC, exp_names):
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

        src_maps : array or dict(name, list of array(s))
            List of valid healpy map array(s) per source per sample, all in same
            resolution used as spatial priors for injecting source positions
            each sample step. Maps must be a normal space PDFs normalized to
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
        # MC structure is the blueprint for all others
        if not isinstance(MC, dict):  # Work consistently with dicts
            MC = {-1: MC}
            self._keymap = {-1: -1}  # Trivial wrappers for consistency
            self._enummap = {-1: -1}
        else:  # Map dict names to int keys for use as indices and vice-versa
            self._keymap = {key: i for key, i in zip(MC.keys(),
                                                     np.arange(len(MC)))}
            self._enummap = {enum: key for key, enum in self._keymap.items()}

        # Work consistently with dicts
        if not isinstance(srcs, dict):
            srcs = {-1: srcs}
        if not isinstance(exp_names, dict):
            exp_names = {-1: exp_names}
        if not isinstance(src_maps, dict):
            src_maps = {-1: src_maps}

        # Keys must be equivalent to MC keys
        if viewkeys(MC) != viewkeys(srcs):
            raise ValueError("Keys in `MC` and `srcs` don't match.")
        if viewkeys(MC) != viewkeys(exp_names):
            raise ValueError("Keys in `MC` and `exp_names` don't match.")
        if viewkeys(src_maps) != viewkeys(exp_names):
            raise ValueError("Keys in `MC` and `src_maps` don't match.")

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

        # Check if prior maps are valid healpy maps and all have same resolution
        # and if there is a map for every source
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
        self.NSIDE = len(map_lens[0])
        if not np.all(map_lens == self.NSIDE):
            raise ValueError("Not all given 'src_maps' have the same length.")
        if not hp.isnsideok(self.NSIDE):
            raise ValueError("Given `src_maps` don't have proper resolution.")

        # Now widen the injection bandwidth to match the 3 sigma band around
        # each source. Use Wilks' theorem to estimate the 3 sigma value.
        # We select from the interval [3sig_band-inj, 3sig_band+inj] per src.
        sigma = scs.chi2.sf(x=3**2, df=2)
        # Normalize maps to sum P = 1 for np.random.choice and read of pixels
        # with map(logllh) - sigma to construct injection band
        pixels = {}
        for key, maps_i in src_maps.items():
            for mapi in maps_i:
                max_idx = np.argmax(mapi)
                max_llh = mapi[max_idx]
                pix_idx = np.where(mapi >= max_llh * sigma)
                # Get max dec coordinate for the selected pixels
                dec, _ = ThetaPhi2DecRa(*hp.pix2ang(self.NSIDE, pix_idx))



        super(SystematicSignalInjector, self).__init__(srcs, MC, exp_names)
        return

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

        # Only one sample, src positions are unambigious
        if len(self._keymap) == 1:
            src_ra = self._srcs[-1]["ra"]
            src_dec = self._srcs[-1]["dec"]
            src_t = self._srcs[-1]["t"]
            src_dt = np.vstack((self._srcs[-1]["dt0"], self._srcs[-1]["dt1"])).T

        n = int(np.around(mean_mu))
        while True:
            if poisson:
                n = self._rndgen.poisson(mean_mu, size=None)  # Returns scalar

            # If n=0 (no events get sampled) return None
            if n < 1:
                yield n, None, None
                continue

            # Draw IDs from the whole pool of events
            sam_idx = self._rndgen.choice(self._mc_arr, size=n,
                                          p=self._sample_w)
            enums = np.unique(sam_idx["enum"])

            # If only one sample: return single recarray
            if len(enums) == 1 and enums[0] < 0:
                sam_ev = np.copy(self._MC[enums[0]][sam_idx["ev_idx"]])
                src_idx = sam_idx['src_idx']
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
            for enum in self._enummap.keys():
                key = self._enummap[enum]
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
