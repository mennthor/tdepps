# coding: utf-8

from __future__ import division, print_function, absolute_import
from builtins import dict, int
from future import standard_library
standard_library.install_aliases()

import numpy as np
from numpy.lib.recfunctions import drop_fields
from sklearn.utils import check_random_state

from .utils import rotator, power_law_flux_per_type


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

        - 'circle' : Select ``MC`` events in circle around a source.
        - 'band' : Select ``MC`` events in a declination band around a source.

        (default: 'band')

    inj_width : float, optinal
        If ``mode`` is 'band', this is half the width of the declination band
        centered at the source positions in radian.
        If ``mode`` is ``circle`` this is the radius of the circle in radian.
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
        per_source : bool, optional
            If True, return the flux per source, which is the total flux
            weighted by the intrinsic weights per source. (default: False)

        Returns
        -------
        flux : float or array-like
            If ``per_source`` is ``True``, return the total flux for all
            sources, otherwise the flux per source. Flux is in unit
            [GeV^-1 cm^-2].
        """
        flux = mu / self._raw_flux
        if per_source:
            w_theo = self._srcs["w_theo"] / np.sum(self._srcs["w_theo"])
            return flux * w_theo
        else:
            return flux

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
        srcs : recarray, shape (nsrcs)
            Source properties, must have names:

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

        MC : recarray or dict(enum, recarray)
            Either single structured array describing Monte Carlo events or a
            dictionary with integer keys `enum` mapped to record arrays.
            `MC` must contain names given in `exp_names` and additonally:

            - 'trueE', float: True event energy in GeV.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator' OneWeight [1]_,
              so it is already divided by ``nevts * nfiles * type_weight``.
              Units are ``[GeV sr cm^2]``. Final event weights are obtained by
              multiplying with desired flux per particle type.

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
        if not isinstance(MC, dict):  # Work consitently with dicts
            MC = {-1: MC}

        required_names = ["ra", "dec", "t", "dt0", "dt1", "w_theo"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` is missing name '{}'.".format(n))

        required_names = ["ra", "sinDec", "logE", "sigma", "timeMJD"]
        for n in required_names:
            if n not in exp_names:
                raise ValueError("`exp_names` is missing name '{}'.".format(n))

        MC_names = exp_names + ("trueRa", "trueDec", "trueE", "ow")
        for n in MC_names:
            for key, mc_i in MC.items():
                if n not in mc_i.dtype.names:
                    e = "MC sample '{}' is missing name '{}'.".format(key, n)
                    raise ValueError(e)

        self._exp_names = exp_names

        # For new srcs set injection solid angle
        self._srcs = srcs
        self._nsrcs = len(srcs)
        self._set_solid_angle()

        # Check if sin_dec_range is OK with the used source positions
        if (np.any(np.sin(srcs["dec"]) < self._sin_dec_range[0]) |
                np.any(np.sin(srcs["dec"]) > self._sin_dec_range[1])):
            raise ValueError("Source position(s) outside `sin_dec_range`.")

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
        assert self._mode in ["band", "circle"]
        for key, mc_i in MC.items():
            # Select events in the injection regions
            if self._mode == "band":
                # Broadcast to mask all srcs at once
                min_decs = self._min_dec.reshape(self._nsrcs, 1)
                max_decs = self._max_dec.reshape(self._nsrcs, 1)
                mc_true_dec = mc_i["trueDec"]
                inj_mask = ((mc_true_dec > min_decs) & (mc_true_dec < max_decs))
            else:
                # Compare great circle distance to each src with inj_width
                src_ra = np.atleast_2d(srcs["ra"]).reshape(self._nsrcs, 1)
                src_dec = np.atleast_2d(srcs["dec"]).reshape(self._nsrcs, 1)
                mc_true_ra = mc_i["trueRa"]
                mc_true_dec = mc_i["trueDec"]
                cos_dist = (np.cos(src_ra - mc_true_ra) *
                            np.cos(src_dec) * np.cos(mc_true_dec) +
                            np.sin(src_dec) * np.sin(mc_true_dec))
                cos_r = np.cos(self._inj_width)
                inj_mask = cos_dist > cos_r

            if not np.any(inj_mask):
                print("Sample {:d}: No events were selected!".format(key))
                self._MC[key] = mc_i[inj_mask.any(axis=0)]  # Add empty slice
                continue

            # Select all at least in one injection region (include overlap)
            total_mask = inj_mask.any(axis=0)
            N_unique = np.count_nonzero(total_mask)
            self._MC[key] = mc_i[total_mask]

            # Total number of selected events, including overlap
            core_mask = (inj_mask.T[total_mask]).T  # Remove all non-selected
            n_tot = np.count_nonzero(core_mask)  # Equal to count inj_mask

            # Append all events to sampling array
            mc_arr = np.empty(n_tot, dtype=dtype)

            # Bookkeeping: Create id to acces the events in the sampling step
            _core = core_mask.ravel()
            # Unique id for selected events per sample and per src
            mc_arr["ev_idx"] = np.tile(np.arange(N_unique), self._nsrcs)[_core]
            # Same src id for each selected evt per src (rows in core_mask)
            mc_arr['src_idx'] = np.repeat(np.arange(self._nsrcs),
                                          np.sum(core_mask, axis=1))
            # Repeat enum id for each sample
            mc_arr["enum"] = key * np.ones(n_tot, dtype=np.int)

            self._mc_arr = np.append(self._mc_arr, mc_arr)

            del mc_arr  # Only needed next loop again, but with different shape

            print("Sample {:d}: Selected {:6d} evts at {:6d} sources.".format(
                key, n_tot, self._nsrcs))
            print("  # Sources without selected evts: {}".format(
                self._nsrcs - np.count_nonzero(np.sum(core_mask, axis=1))))

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
            Number of events
        sam_ev : iterator
            Sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample.
        """
        if self._mc_arr is None:
            raise ValueError("Injector has not been filled with MC data yet.")

        src_ra = self._srcs["ra"]
        src_dec = self._srcs["dec"]
        src_t = self._srcs["t"]
        src_dt = np.vstack((self._srcs["dt0"], self._srcs["dt1"])).T

        n = int(np.around(mean_mu))
        while True:
            if poisson:
                n = self._rndgen.poisson(mean_mu, size=1)

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
                    src_ra[src_idx], src_dec[src_idx], sam_ev)
                sam_ev["timeMJD"] = self._sample_times(
                    src_t[src_idx], src_dt[src_idx])[m]
                yield n, sam_ev, sam_idx[m]
                continue

            # Else return same dict structure as used in fit
            sam_ev = dict()
            for enum in enums:
                # Select events per sample
                idx = sam_idx[sam_idx["enum"] == enum]["ev_idx"]
                sam_ev_i = np.copy(self._MC[enum][idx])
                # Broadcast corresponding sources for correct rotation
                src_idx = sam_idx[sam_idx["enum"] == enum]["src_idx"]
                sam_ev[enum], m = self._rot_and_strip(
                    src_ra[src_idx], src_dec[src_idx], sam_ev_i)
                sam_ev[enum]["timeMJD"] = self._sample_times(
                    src_t[src_idx], src_dt[src_idx])[m]

            yield n, sam_ev, sam_idx[m]

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
        w_theo = self._srcs["w_theo"]

        # Broadcast src dependent weight parts to each evt
        omega = self._omega[src_idx]
        w_theo = (w_theo / np.sum(w_theo))[src_idx]
        assert len(omega) == len(w_theo) == len(self._mc_arr)

        # Calculate physical weights for E^-gamma fluence for all events
        w = []
        for enum, mc_i in self._MC.items():  # Select again per sample
            idx = self._mc_arr[self._mc_arr["enum"] == enum]["ev_idx"]
            flux = power_law_flux_per_type(mc_i["trueE"][idx], self._gamma)
            w.append(mc_i["ow"][idx] * flux)

        # Finalize by dividing with per event injection solid angle
        w = np.concatenate(w, axis=0)
        w *= w_theo
        w /= omega
        assert len(w) == len(self._mc_arr)

        # Total injected fluence and normalized sampling weights
        self._raw_flux = np.sum(w)
        self._sample_w = w / self._raw_flux
        assert np.allclose(np.sum(self._sample_w), 1.)

        return

    def _set_solid_angle(self):
        """
        Setup solid angles of injection area for selected MC events and sources.

        For a given set of source positions and an injection mode, we need to
        calculate the solid angle per source from which events where injected
        to be able to correctly weight the injected events to get a flux.

        Sets up private class varaibles:

        - ``_omega``, array-like: Solid angle in radians of each injection
          region.
        - ``_min_dec``, ``_max_dec``, array-like: Upper/lower bounds for each
          declination band in radians (only if `self._mode` is 'band').
        """
        assert self._mode in ["band", "circle"]

        if self._mode == "band":
            sin_dec_bandwidth = np.sin(self._inj_width)
            A, B = self._sin_dec_range

            if self._skylab_band:
                m = (A - B + 2. * sin_dec_bandwidth) / (A - B)
                b = sin_dec_bandwidth * (A + B) / (B - A)
                sin_dec = m * np.sin(self._srcs["dec"]) + b
            else:
                sin_dec = np.sin(self._srcs["dec"])

            min_sin_dec = np.maximum(A, sin_dec - sin_dec_bandwidth)
            max_sin_dec = np.minimum(B, sin_dec + sin_dec_bandwidth)

            self._min_dec = np.arcsin(np.clip(min_sin_dec, -1., 1.))
            self._max_dec = np.arcsin(np.clip(max_sin_dec, -1., 1.))

            # Solid angles of selected events around each source
            self._omega = 2. * np.pi * (max_sin_dec - min_sin_dec)
            assert (len(self._min_dec) == len(self._max_dec) == self._nsrcs)
        else:
            r = self._inj_width
            _omega = 2 * np.pi * (1. - np.cos(r))
            self._omega = np.ones(self._nsrcs, dtype=np.float) * _omega
            assert len(self._omega) == self._nsrcs
        return

    def _rot_and_strip(self, src_ras, src_decs, MC):
        """
        Rotate injected event positions to the sources and strip Monte Carlo
        information from the output array.

        The rotation angles to move the true directions to the sources are used
        to rotate the measured positions by the same amount.

        Parameters
        ----------
        src_ras, src_decs : array-like, shape (len(MC))
            Sources equatorial positions in right-ascension in :math:`[0, 2\pi]`
            and declination in :math:`[-\pi/2, \pi/2]`, both given in radians.
            These are the coordinates we rotate on per event in ``MC``.
        MC : record array
            See :py:meth:<SignalInjector.fit>, Parameters.

        Returns
        --------
        ev : structured array
            Array with rotated values, true MC information is deleted
        m : array-like
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
        drop_names = [n for n in MC.dtype.names if n not in self._exp_names]
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
        nevts = len(src_t)

        # Sample uniformly in [0, 1] and scale to time windows per source in MJD
        r = self._rndgen.uniform(0, 1, size=nevts)
        times_rel = r * np.diff(dt, axis=1).ravel() + dt[:, 0]

        return src_t + times_rel / self._SECINDAY

    def __str__(self):
        """
        Use to print all settings: `>>> print(sig_inj_object)`
        """
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
