# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import dict, open, filter, zip
from future import standard_library
standard_library.install_aliases()                                              # noqa

import os
import json
import numpy as np
from astropy.time import Time as astrotime

import abc     # Abstract Base Class
import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class BGRateInjector(object):
    """
    Background Rate Injector Base Class

    Samples times of background-like events for a given time frame.

    Classes must implement methods:

    - `fun`
    - `sample`

    Class object then provides public methods:

    - `fun`
    - `sample`
    - `get_nb`
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._SECINDAY = 24. * 60. * 60.
        self.best_pars = None
        self.best_estimator = None
        self.best_estimator_integral = None

    @docs.get_summaryf("BGRateInjector.fit")
    @docs.get_sectionsf("BGRateInjector.fit",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    @abc.abstractmethod
    def fit(self, T):
        """
        Build the injection model with times from experimental data.

        Additionally creates new class attributes:

        - best_pars, tuple: Best fit parameters.
        - best_estimator, callable: Rate function with `best_pars` plugged in.
        - best_estimator_integral, callable: Rate function integral with
          `best_pars` plugged in.

        Parameters
        ----------
        T : array_like, shape (n_samples)
            Per event times in MJD days of experimental data.
        """
        pass

    @docs.get_summaryf("BGRateInjector.sample")
    @docs.get_sectionsf("BGRateInjector.sample",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    @abc.abstractmethod
    def sample(self, t, trange, poisson=True, random_state=None):
        """
        Generate random samples from the fitted model for multiple source event
        times and corrseponding time frames at once.

        Sample size can be drawn from a poisson distribution each time, with
        expectation value determined by the time window, so each call might
        produce a different number of events.

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows [[t0, t1], ...] in seconds around each given time t.
        poisson : bool, optional
            If True, sample the number of events per src using a poisson
            distribution, with expectation from the background expectation in
            each time window. Otherwise they are rounded to the next integer to
            the expectation value and alway the same. (default: True)
        random_state : seed, optional
            Turn seed into a `np.random.RandomState` instance. See
            `sklearn.utils.check_random_state`. (default: None)

        Returns
        -------
        event_times : list of arrays, length (nsrcs)
            The times in MJD for each sampled srcs. Arrays might be empty, when
            the background expectation is low for small time windows.
        """
        pass

    def get_nb(self, t, trange):
        """
        Return the expected number of events from integrating the rate function
        in the given time ranges.

        Parameters
        ----------
        t : array-like, shape (nsrcs)
            MJD times of sources.
        trange : array-like, shape(nsrcs, 2)
            Time windows [[t0, t1], ...] in seconds around each given time t.

        Returns
        -------
        nb : array-like, shape (nsrcs)
            Expected number of background events for each sources time window.
        """
        if self.best_pars is None:
            raise RuntimeError("Injector was not fit to data yet.")

        t, trange = self._prep_t_trange(t, trange)

        # BG expectations are the integrals over each time frames, shape (nsrcs)
        return self.best_estimator_integral(t, trange)

    def _prep_t_trange(self, t, trange):
        """
        Put t, trange in needed shapes and check dimensions

        Parameters
        ----------
        t, trange
            See `BGRateInjector.sample`, Parameters

        Returns
        -------
        t, trange
            See `BGRateInjector.sample`, Parameters
        """
        t = np.atleast_1d(t)
        trange = np.atleast_2d(trange)
        nsrcs = len(t)
        if trange.shape != (nsrcs, 2):
            raise ValueError("`trange` shape must be (nsrcs, 2).")

        return t, trange

    def _set_best_fit(self, pars):
        """
        Sets new class attributes with the best fit parameters.

        Creates new class attributes:

        - best_pars, tuple: Best fit parameters.
        - best_estimator, callable: Rate function with `best_pars` plugged in.
        - best_estimator_integral, callable: Rate function integral with
          `best_pars` plugged in.

        Parameters
        ----------
        pars : tuple
            Best fit parameters plugged into `fun` and `integral`.
        """
        self.best_pars = pars
        self.best_estimator = (lambda t: self.rate_func.fun(t, pars))
        self.best_estimator_integral = (
            lambda t, trange: self.rate_func.integral(t, trange, pars))

        return self.best_pars


class RunlistBGRateInjector(BGRateInjector):
    """
    Runlist Background Rate Injector

    Creates the time depending rate from a given runlist.
    Parameters are passed to `create_goodrun_dict` method.

    Parameters
    ----------
    runlist : str
        Path to a valid good run runlist snapshot from [1]_ in JSON format.
        Must have keys 'latest_snapshot' and 'runs'.
    filter_runs : function
        Filter function to remove unwanted runs from the goodrun list.
        Called as `filter_runs(run)`. Function must operate on a single
        dictionary argument, with keys:
        ['good_i3', 'good_it', 'good_tstart', 'good_tstop', 'run', 'reason_i3',
        'reason_it', 'source_tstart', 'source_tstop', 'snapshot', 'sha'].
    rate_func : `rate_function.RateFunction` instance
        Class defining the function to describe the time dependent
        background rate. Must provide functions
        ['fun', 'integral', 'fit', 'sample'].

    Notes
    -----
    .. [1] https://live.icecube.wisc.edu/snapshots/
    """
    def __init__(self, runlist, filter_runs, rate_func):
        super(RunlistBGRateInjector, self).__init__()

        # Create a goodrun list from the JSON snapshot
        runlist = os.path.abspath(runlist)
        self.goodrun_dict = self.create_goodrun_dict(runlist, filter_runs)

        self.rate_func = rate_func
        return

    @docs.dedent
    def fit(self, T, x0=None, remove_zero_runs=False, **kwargs):
        """
        %(BGRateInjector.fit.summary)s

        Takes data and a binning derived from the runlist. Bins the data,
        normalizes to a rate in HZ and fits a RateFunction over the whole
        time span to it. This function serves as a rate per time model.

        Parameters
        ----------
        %(BGRateInjector.fit.parameters)s
        x0 : array-like, optional
            Seed values for the fit function as described above. If None,
            defaults from `RateFunction` are used. (default: None)
        remove_zero_runs : bool, optional
            If True, remove all runs with zero events and adapt the livetime.
            (default: False)
        kwargs
            Other arguments are passed to the `scipy.optimize.minimize` method.

        Returns
        -------
        best_estimator : function
            Rate function with the best fit parameters plugged in.
        """
        # Put data into run bins to fit them
        h = self._create_runtime_hist(T, self.goodrun_dict, remove_zero_runs)
        rate = h["rate"]
        rate_std = h["rate_std"]
        binmids = 0.5 * (h["start_mjd"] + h["stop_mjd"])

        resx = self.rate_func.fit(binmids, rate, p0=None, rate_std=rate_std)

        # Setup wrappers for functions with best fit pars plugged in
        return self._set_best_fit(resx)

    @docs.dedent
    def sample(self, t, trange, poisson=True, random_state=None):
        """
        %(BGRateInjector.sample.summary)s

        Parameters
        ----------
        %(BGRateInjector.sample.parameters)s

        Returns
        -------
        %(BGRateInjector.sample.returns)s
        """
        if self.best_pars is None:
            raise RuntimeError("Injector was not fit to data yet.")

        # BG expectations are the integrals over each time frames, shape (nsrcs)
        t, trange = self._prep_t_trange(t, trange)
        expect = self.get_nb(t, trange)
        nsrcs = len(t)

        # Get number of actual events to sample times for
        if poisson:
            nevents = np.random.poisson(lam=expect, size=nsrcs)
        else:  # If one expectation is < 0.5 no event is sampled for that src
            nevents = np.round(expect).astype(int)

        # Sample all nevents for this trial from the rate function at once
        return self.rate_func.sample(t, trange, self.best_pars,
                                     n_samples=nevents,
                                     random_state=random_state)

    def create_goodrun_dict(self, runlist, filter_runs):
        """
        Create a dict of lists from a runlist in JSON format.
        Each entry in each list is one run.

        Parameters
        ----------
        runlist, filter_runs
            See `RunlistBGRateInjector`, Parameters

        Returns
        -------
        goodrun_dict : dict
            Dictionary with run attributes as keys. The values are stored in
            arrays in each key.
        """
        with open(runlist, 'r') as jsonFile:
            goodruns = json.load(jsonFile)

        if not all([k in goodruns.keys() for k in ["latest_snapshot", "runs"]]):
            raise ValueError("Runlist misses 'latest_snapshot' or 'runs'")

        # This is a list of dicts (one dict per run)
        goodrun_list = goodruns["runs"]

        # Filter to remove unwanted runs
        goodrun_list = list(filter(filter_runs, goodrun_list))

        # Convert the run list of dicts to a dict of arrays for easier handling
        goodrun_dict = dict(zip(goodrun_list[0].keys(),
                                zip(*[r.values() for r in goodrun_list])))
        for k in goodrun_dict.keys():
            goodrun_dict[k] = np.array(goodrun_dict[k])

        # Add times to MJD floats
        goodrun_dict["good_start_mjd"] = astrotime(
            goodrun_dict["good_tstart"]).mjd
        goodrun_dict["good_stop_mjd"] = astrotime(
            goodrun_dict["good_tstop"]).mjd

        # Add runtimes in MJD days
        goodrun_dict["runtime_days"] = (goodrun_dict["good_stop_mjd"] -
                                        goodrun_dict["good_start_mjd"])

        return goodrun_dict

    # Private Methods
    def _create_runtime_hist(self, T, goodrun_dict, remove_zero_runs=False):
        """
        Creates time bins [start_MJD_i, stop_MJD_i] for each run i and bins the
        experimental data to calculate the rate for each run.

        Parameters
        ----------
        T, remove_zero_runs
            See `RunlistBGRateInjector.fit`, Parameters
        goodrun_dict
            See `RunlistBGRateInjector.create_goodrun_dict`, Returns

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
            - "rates_std" : float, sqrt(N) stddev of the rate in Hz in this run.
        """
        # Store events in bins with run borders
        start_mjd = goodrun_dict["good_start_mjd"]
        stop_mjd = goodrun_dict["good_stop_mjd"]
        run = goodrun_dict["run"]

        tot_evts = 0
        # Histogram time values in each run manually
        evts = np.zeros_like(run, dtype=int)
        for i, (start, stop) in enumerate(zip(start_mjd, stop_mjd)):
            mask = (T >= start) & (T < stop)
            evts[i] = np.sum(mask)
            tot_evts += np.sum(mask)

        # Crosscheck, if we got all events and didn't double count
        if not tot_evts == len(T):
            print("Events selected : ", tot_evts)
            print("Events in T     : ", len(T))
            raise ValueError("Not all events in 'T' were sorted in bins. If " +
                             "this is intended, please remove them beforehand.")

        if remove_zero_runs:
            # Remove all zero event runs and update livetime
            m = (evts > 0)
            _livetime = np.sum(stop_mjd[~m] - start_mjd[~m])
            evts, run = evts[m], run[m]
            start_mjd, stop_mjd = start_mjd[m], stop_mjd[m]
            print("Removing runs with zero events")
            print("  Number of runs with 0 events : {:d}".format(np.sum(~m)))
            print("  Total livetime of those runs : {:.3f} d".format(_livetime))

        # Normalize to rate in Hz
        runtime = stop_mjd - start_mjd
        rate = evts / (runtime * self._SECINDAY)

        # Calculate 1 / sqrt(N) stddev for scaled rates
        rate_std = np.sqrt(rate) / np.sqrt(runtime * self._SECINDAY)

        # Create record-array
        names = ["run", "rate", "runtime", "start_mjd",
                 "stop_mjd", "nevts", "rate_std"]
        types = [int, np.float, np.float, np.float, np.float, int, np.float]
        dtype = [(n, t) for n, t in zip(names, types)]

        a = np.vstack((run, rate, runtime, start_mjd, stop_mjd, evts, rate_std))
        rate_rec = np.core.records.fromarrays(a, dtype=dtype)

        # Final total livetime and store stuff
        self.livetime = np.sum(rate_rec["runtime"])
        self.rate_rec = rate_rec

        return rate_rec
