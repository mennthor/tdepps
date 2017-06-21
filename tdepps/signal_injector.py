# coding: utf-8

import numpy as np
from numpy.lib.recfunctions import drop_fields
from sklearn.utils import check_random_state

from .utils import rotator


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
        One of `['circle'|'band']`. Selects MC events to inject based
        on their true location:

        - 'circle' : Select `MC` events in circle around a source.
        - 'band' : Select `MC` events in a declination band around a source.

        (default: 'band')

    inj_width : float, optinal
        If `mode` is 'band', this is half the width of the declination band
        centered at the source positions in radian.
        If `mode` is `circle` this is the radius of the circle in radian.
        (default: `np.deg2rad(2)`)
    """
    def __init__(self, gamma, mode="band", inj_width=np.deg2rad(2)):
        if (gamma < 1.) or (gamma > 4.):
            raise ValueError("`gamma` in doubtful range, must be in [1, 4].")
        self._gamma = gamma

        if mode not in ["band", "circle"]:
            raise ValueError("`mode` must be one of ['band', 'circle']")
        self._mode = mode

        if (inj_width <= 0.) or (inj_width > np.pi):
            raise ValueError("Injection width must be in (0, pi].")
        self._inj_width = inj_width

        # Private defaults
        self._srcs = None
        self._nsrcs = None
        self._sin_dec_range = np.array([-1., 1.])
        self._SECINDAY = 24. * 60. + 60.

        return

    @property
    def gamma(self):
        return self._gamma

    @property
    def mode(self):
        return self._mode

    @property
    def inj_width(self):
        return self._inj_width

    def fit(self, srcs, MC, livetime):
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
              window in seconds, centered around each source time `t`.
            - 'w_theo', float: Theoretical source weight per source, eg. from
              a known gamma flux.

        MC : recarray or dict(enum, recarray)
            Either single structured array describing Monte Carlo events or a
            dictionary with integer keys `enum` mapped to record arrays.
            `MC` must contain names:

            - 'timeMJD': Per event times in MJD days.
            - 'ra', float: Per event equatorial right-ascension, given in
              radian, in :math:`[0, 2\pi]`.
            - 'dec', float: Per event equatorial declination, given in
              radians, in :math:`[-\pi/2, \pi/2]`.
            - 'logE', float: Per event energy proxy, given in log10(1/GeV).
            - 'sigma': Per event positional uncertainty, given in radians. It
              is assumed, that a circle with radius sigma contains
              approximatly :math:`1\sigma\approx 0.39` of probability of the
              reconstrucion likelihood space.
            - 'trueRa', 'trueDec', float: True MC equatorial coordinates.
            - 'trueE', float: True event energy in GeV.
            - 'ow', float: Per event 'neutrino generator (NuGen)' OneWeight
              [2]_, already divided by `nevts * nfiles` known from SimProd.
              Units are 'GeV sr cm^2'. Final event weights are obtained by
              multiplying with desired flux.

        livetime : float or dict(enum, float)
            Livetime in MJD days per sample with same `enum` as in `MC`.
        """
        if isinstance(MC, dict) ^ isinstance(livetime, dict):
            raise TypeError("`MC` and `livetime` must both be a dict or " +
                            "recarray and float.")

        if not isinstance(MC, dict):  # Work consitently with dicts
            MC = {-1: MC}
            livetime = {-1: livetime}

        required_names = ["ra", "dec", "t", "dt0", "dt1", "w_theo"]
        for n in required_names:
            if n not in srcs.dtype.names:
                raise ValueError("`srcs` is missing name '{}'.".format(n))

        MC_names = ["timeMJD", "ra", "dec", "logE", "sigma",
                    "trueRa", "trueDec", "trueE", "ow"]
        for n in MC_names:
            if n not in MC.dtype.names:
                raise ValueError("`MC` is missing name '{}'.".format(n))

        # For new srcs set injection solid angle
        self._srcs = srcs
        self._nsrcs = len(srcs)
        self._set_solid_angle()

        # self.mc_arr: Store selected event ids to sample from a single array
        # ev_idx: event ID, src_idx: src ID, enum: sample ID
        dtype = [("ev_idx", np.int), ("src_idx", np.int), ("enum", np.int)]
        self.mc_arr = np.empty(0, dtype=dtype)
        # self.MC: Store unique events selected for sampling per sample
        self.MC = dict()

        # Broadcast to mask all srcs at once
        _min_sin_decs = np.sin(self._min_dec).reshape(self._nsrcs, 1)
        _max_sin_decs = np.sin(self._max_dec).reshape(self._nsrcs, 1)

        # Loop all samples
        assert self.mode in ["band", "circle"]
        for key, mc_i in MC.items():
            # Select events in the injection regions
            if self.mode == "band":
                _mc_sin_dec = np.sin(mc_i["trueDec"])
                inj_mask = ((_mc_sin_dec > _min_sin_decs) &
                            (_mc_sin_dec < _max_sin_decs))
            else:
                # Compare great circle distance to each src with inj_width
                src_ra = np.atleast_2d(srcs["ra"]).reshape(self._nsrcs, 1)
                src_dec = np.atleast_2d(srcs["dec"]).reshape(self._nsrcs, 1)
                mc_ra = mc_i["trueRa"]
                mc_dec = mc_i["trueDec"]
                cos_dist = (np.cos(src_ra - mc_ra) *
                            np.cos(src_dec) * np.cos(mc_dec) +
                            np.sin(src_dec) * np.sin(mc_dec))
                cos_r = np.cos(self.inj_width)
                inj_mask = cos_dist < cos_r

            print(inj_mask.shape)

            if not np.any(inj_mask):
                print("Sample {:d}: No events were selected!".format(key))
                self.MC[key] = mc_i[inj_mask.any(axis=0)]  # Add empty slice
                continue

            # Select all at least in one injection region (include overlap)
            total_mask = inj_mask.any(axis=0)
            N_unique = np.count_nonzero(total_mask)
            self.MC[key] = mc_i[total_mask]

            # Total number of selected events, including overlap
            core_mask = (inj_mask.T[total_mask]).T  # Remove all non-selected
            n_tot = np.count_nonzero(core_mask)  # Equal to count inj_mask

            # Append all events to sampling array
            mc_arr = np.empty(n_tot, dtype=dtype)

            # Create id to acces the events in the sampling step
            _core = core_mask.ravel()
            # Unique id for selected events, one src after another
            mc_arr["ev_idx"] = np.tile(np.arange(N_unique), self._nsrcs)[_core]
            # Same src id for each selected evt per src (rows in core_mask)
            mc_arr['src_idx'] = np.repeat(np.arange(self._nsrcs),
                                          np.sum(core_mask, axis=1))
            # Repeat enum id for each sample
            mc_arr["enum"] = key * np.ones(n, dtype=np.int)

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            print("Sample {:d}: Selected {:6d} evts at {:6d} sources.".format(
                key, n, self._nsrcs))
            print("  # Sources without selected evts: {}".format(
                np.count_nonzero(np.sum(core_mask, axis=1))))

        if len(self.mc_arr) < 1:
            raise ValueError("No events were selected. Check `inj_width`.")

        print("Selected {:d} events in total".format(len(self.mc_arr)))

        self._set_sampling_weights()

        return

    def sample(self, mean_mu, poisson=True, random_state=None):
        """
        Generator to get sampled events from MC for each source position.

        Parameters
        -----------
        mu : float
            Expectation value of number of events to sample.
        poisson : bool, optional
            If True, sample the actual number of events from a poisson
            distribution with expectation `mu`. Otherwise the number of events
            is constant in each trial. (default: True)

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            Sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample.
        """
        rndgen = check_random_state(random_state)

        src_ra = self.srcs["ra"]
        src_dec = self.srcs["dec"]
        src_t = self.srcs["timeMJD"]
        src_dt = np.vstack((self.srcs["dt0"], self.srcs["dt1"])).T

        while True:
            if poisson:
                n = rndgen.poisson(mean_mu, size=1)
            else:
                n = int(np.around(mean_mu))

            # If n=0 (no events get sampled) return None
            if n < 1:
                yield n, None
                continue

            # Draw IDs from the whole pool of events
            sam_idx = self.random.choice(self.mc_arr, size=n, p=self._norm_w)

            # Get the actual events from the sampled IDs
            enums = np.unique(sam_idx["enum"])

            # If only one sample: return single recarray
            if len(enums) == 1 and enums[0] < 0:
                sam_ev = np.copy(self.MC[enums[0]][sam_idx["idx"]])
                sam_ev = self._rot_and_strip(src_ra[sam_idx['src_idx']],
                                             src_dec[sam_idx['src_idx']],
                                             sam_ev)
                sam_ev["timeMJD"] = self._sample_times(src_t, src_dt, rndgen)
                yield n, sam_ev
                continue

            # Else return same dict structure as used in fit
            sam_ev = dict()
            for enum in enums:
                # Filter events per enum
                idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                sam_ev_i = np.copy(self.MC[enum][idx])
                # Broadcast corresponding sources for correct rotation
                src_ind = sam_idx[sam_idx["enum"] == enum]["src_idx"]
                sam_ev[enum] = self._rot_and_strip(src_ra[src_ind],
                                                   src_dec[src_ind],
                                                   sam_ev_i)
                sam_ev[enum]["timeMJD"] = self._sample_times(src_t, src_dt,
                                                             rndgen)

            yield n, sam_ev

    def _set_sampling_weights(self):
        """
        Setup per event sampling weights from the oneweights.

        Physics weights per event (in Hz) are the ratio of expected fluence
        to the generated fluence per event:

        .. math:

          w_i &= \frac{dF_i / (\text{GeV cm}^2\text{ s sr})}
                      {dF^0_i / (\text{GeV cm}^2\text{ sr})} \\
              &= \frac{\text{ow}_i}{N_\text{gen}} \cdot \Phi
                 \frac{E_i^{-\gamma}}{\Omega_\text{inj}}

        because OneWeight :math:`ow` is defined as the inverse generating flux
        times the nugen specific interaction probability. :math:`\Phi` is the
        fluence normalization in (GeV cm^2) we search for by injecting events to
        pure background and see how the TS transfroms.

        Detector acceptance weights are automatically accounted for in the
        Monte Carlo sample. It simply has more events in regions with higher
        acceptance.

        Also different acceptances in different samples are already taken care
        of with the OneWeight factorm which is already divided by the number of
        generated events per sample. So if the acceptance was lower the
        simulation just threw away more events resulting in lower OneWeights.

        So we only include the theoretical weights here manually.
        """
        w_theo = self._srcs["w_theo"].astype(np.float)
        w_theo /= np.sum(w_theo)
        assert np.allclose(np.sum(w_theo), 1.)

        # Broadcast src dependent weight parts to each evt
        omega = (self._omega / self.w_theo)[self.mc_arr['src_idx']]
        assert len(omega) == len(self.mc_arr)

        # Calculate physical weights for E^-gamma fluence for all events
        w = []
        for mc_i in self.MC.values():
            _w = ((mc_i["ow"] * mc_i["trueE"]**(-self.gamma))
                  [self.mc_arr["idx"]])
            w.append(_w / omega)

        w = np.array(w)
        assert len(w) == len(self.mc_arr)

        # Total injected fluence and normalized sampling weights
        self._raw_fluence = np.sum(w)
        self._sample_w = w / self._raw_fluence
        assert np.allclose(np.sum(self._sample_w), 1.)

        return

    def _set_solid_angle(self):
        """
        Setup solid angles of injection area for selected MC events and sources.

        For a given set of source positions and an injection mode, we need to
        calculate the solid angle per source from which events where injected
        to be able to correctly weight the injected events to get a flux.

        Sets up private class varaibles:

        - _omega, array-like: Solid angle in radians of each injection region.
        - _min_dec, _max_dec, array-like: Upper/lower bounds for each
          declination band in radians (only if `self.mode` is 'band').
        """
        assert self.mode in ["band", "circle"]

        if self.mode == "band":
            sin_dec_bandwidth = np.sin(self.inj_width)
            A, B = self._sin_dec_range
            sin_dec = np.sin(self._srcs["dec"])

            min_sin_dec = np.maximum(A, sin_dec - sin_dec_bandwidth)
            max_sin_dec = np.minimum(B, sin_dec + sin_dec_bandwidth)

            self._min_dec = np.arcsin(np.clip(min_sin_dec, -1., 1.))
            self._max_dec = np.arcsin(np.clip(max_sin_dec, -1., 1.))

            # Solid angles of selected events around each source
            self._omega = 2. * np.pi * (max_sin_dec - min_sin_dec)
            assert (len(self._min_dec) == len(self._max_dec) == self._nsrcs)
        else:
            r = self.inj_width
            _omega = 2 * np.pi * (1. - np.cos(r))
            self._omega = np.ones(self._nsrcs, dtype=np.float) * _omega

            assert len(self._omega) == self._nsrcs
        return

    def _rot_and_strip(self, src_ras, src_decs, rec):
        """
        Rotate injected event positions to the sources and strip Monte Carlo
        information from the output array.

        The rotation angles to move the true directions to the sources are used
        to rotate the measured positions by the same amount.

        Parameters
        ----------
        src_ras, src_decs : array-like, shape (len(rec))
            Sources equatorial positions in right-ascension in :math:`[0, 2\pi]`
            and declination in :math:`[-\pi/2, \pi/2]`, both given in radians.
            These are the coordinates we rotate on per event in `ev`.
        rec : record array
            See :py:meth:<SignalInjector.fit>, Parameters.

        Returns
        --------
        ev : structured array
            Array with rotated value, true information is deleted
        """
        rec["ra"], _dec = rotator(rec["trueRa"], rec["trueDec"],
                                  src_ras, src_decs,
                                  rec["ra"], rec["dec"])

        rec["sinDec"] = np.sin(_dec)
        if "dec" in rec.names:
            rec["dec"] = _dec

        # Remove Monte Carlo information from sampled events
        MC_names = ["trueRa", "trueDec", "trueE", "ow"]
        return drop_fields(rec, MC_names)

    def _sample_times(self, src_t, dt, rndgen):
        """
        Sample times uniformly in on-time signal PDF region.

        Parameters
        ----------
        src_t : array-like (nevts)
            Source time in MJD per event.
        dt : array-like, shape (nevts, 2)
            Time window in seconds centered around `src_t` in which the signal
            time PDF is assumed to be uniform.
        rndgen : `np.random.RandomState` instance
            Random number generator instance.

        Returns
        -------
        times : array-like, shape (nevts)
            Sampled times for this trial.
        """
        # Transform time window to MJD and check on correct shapes
        src_t = np.atleast_1d(src_t)
        nsrcs = len(src_t)
        # Proper braodcasting to process all srcs at once
        src_t = src_t.reshape(nsrcs, 1)
        dt = np.atleast_2d(dt).reshape(nsrcs, 2)
        trange = src_t + dt / self._SECINDAY

        return rndgen.uniform(trange[:, 0], trange[:, 1])
