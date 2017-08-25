# coding: utf-8

from __future__ import print_function, division, absolute_import
from builtins import dict, open, filter, zip
from future import standard_library
standard_library.install_aliases()                                              # noqa

import os
import json
import numpy as np
from astropy.time import Time as astrotime
from sklearn.utils import check_random_state

import abc     # Abstract Base Class
import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class BGRateInjector(object):
    __metaclass__ = abc.ABCMeta

    @docs.get_sectionsf("BGRateInjector.init", sections=["Parameters"])
    @docs.dedent
    def __init__(self, rate_func, random_state=None):
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

        Additionally creates new class attributes after calling the fit method:

            - livetime, float: Livetime in days of the given data.
            - best_pars, tuple: Best fit parameters.
            - best_estimator, callable: Rate function with `best_pars` plugged
              in.
            - best_estimator_integral, callable: Rate function integral with
              `best_pars` plugged in.

        Parameters
        ----------
        rate_func : `rate_function.RateFunction` instance
            Class defining the function to describe the time dependent
            background rate. Must provide functions
            ['fun', 'integral', 'fit', 'sample'].
        random_state : seed, optional
            Turn seed into a `np.random.RandomState` instance. See
            `sklearn.utils.check_random_state`. (default: None)

        Example
        -------
        >>> import tdepps.bg_rate_injector as BGRateInj
        >>>
        >>> # Make a simple run filter list
        >>> filter_runs = lambda run: (run["good_i3"] == True)
        >>>
        >>> # Chose a rate function and a path to a goodrunlist
        >>> rate_func = RateFunc.Sinus1yrRateFunction()
        >>> runlist = "/path/to/goodrunlist.json"
        >>> runlist_inj = BGRateInj.RunlistBGRateInjector(runlist, filter_runs,
                                                          rate_func)
        """
        self._rate_func = rate_func
        self.rndgen = random_state

        self._SECINDAY = 24. * 60. * 60.
        self._livetime = None
        self._best_pars = None
        self._best_estimator = None
        self._best_estimator_integral = None

        return

    @property
    def rndgen(self):
        return self._rndgen

    @rndgen.setter
    def rndgen(self, random_state):
        self._rndgen = check_random_state(random_state)

    @property
    def rate_func(self):
        return self._rate_func

    @rate_func.setter
    def rate_func(self, arg):
        raise ValueError("`rate_func` can't be set. Create new object instead.")

    @property
    def livetime(self):
        if self._livetime is None:
            raise RuntimeWarning("BGRateInjector has not been fitted yet.")
        return self._livetime

    @livetime.setter
    def livetime(self, livetime):
        raise ValueError("Use `fit` to calculate a new livetime.")

    @property
    def best_pars(self):
        if self._best_pars is None:
            raise RuntimeWarning("BGRateInjector has not been fitted yet.")
        return self._best_pars

    @best_pars.setter
    def best_pars(self, best_pars):
        raise ValueError("Use `fit` to get new best fit parameters.")

    @property
    def best_estimator(self):
        if self._best_estimator is None:
            raise RuntimeWarning("BGRateInjector has not been fitted yet.")
        return self._best_estimator

    @best_estimator.setter
    def best_estimator(self, best_estimator):
        raise ValueError("Use `fit` to get a new best estimator function.")

    @property
    def best_estimator_integral(self):
        if self._best_estimator_integral is None:
            raise RuntimeWarning("BGRateInjector has not been fitted yet.")
        return self._best_estimator_integral

    @best_estimator_integral.setter
    def best_estimator_integral(self, best_estimator_integral):
        raise ValueError("Use `fit` to get a new best estimator integral.")

    @docs.get_summaryf("BGRateInjector.fit")
    @docs.get_sectionsf("BGRateInjector.fit",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    @abc.abstractmethod
    def fit(self, T):
        """
        Build the injection model with times from experimental data.

        Additionally creates new class attributes:

        - livetime, float: Livetime in days of the given data.
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

    def sample(self, t, trange, poisson=True):
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

        Returns
        -------
        event_times : list of arrays, length (nsrcs)
            The times in MJD for each sampled srcs. Arrays might be empty, when
            the background expectation is low for small time windows.
        """
        # BG expectations are the integrals over each time frames, shape (nsrcs)
        t, trange = self._prep_t_trange(t, trange)
        expect = self.get_nb(t, trange)
        nsrcs = len(t)

        # Get number of actual events to sample times for
        if poisson:
            nevents = self._rndgen.poisson(lam=expect, size=nsrcs)
        else:  # If one expectation is < 0.5 no event is sampled for that src
            nevents = np.round(expect).astype(int)

        # Sample all nevents for this trial from the rate function at once
        return self._rate_func.sample(t, trange, self._best_pars,
                                      n_samples=nevents)

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
        if self._best_pars is None:
            raise RuntimeError("Injector was not fit to data yet.")

        t, trange = self._prep_t_trange(t, trange)

        # BG expectations are the integrals over each time frames, shape (nsrcs)
        return self._best_estimator_integral(t, trange)

    def _prep_t_trange(self, t, trange):
        """
        Put t, trange in needed shapes and check dimensions

        Parameters
        ----------
        t, trange
            See :py:meth:`BGRateInjector.sample`, Parameters

        Returns
        -------
        t, trange
            See :py:meth:`BGRateInjector.sample`, Parameters
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
        self._best_pars = pars
        self._best_estimator = (lambda t: self._rate_func.fun(t, pars))
        self._best_estimator_integral = (
            lambda t, trange: self._rate_func.integral(t, trange, pars))

        return self._best_estimator


class RunlistBGRateInjector(BGRateInjector):
    @docs.dedent
    def __init__(self, rate_func, runlist, filter_runs=None, random_state=None):
        """
        Runlist Background Rate Injector

        Creates the time depending rate from a given runlist.
        Parameters are passed to `create_goodrun_dict` method.

        Parameters
        ----------
        %(BGRateInjector.init.parameters)s
        runlist : dict
            Dict made from a good run runlist snapshot from [1]_ in JSON format.
            Must have key 'runs' at top level which has a list that lists all
            runs as dictionaries::

                {
                  "runs":[
                        { ...,
                          "good_tstart": "YYYY-MM-DD HH:MM:SS",
                          "good_tstop": "YYYY-MM-DD HH:MM:SS",
                          "run": 123456,
                          ... },
                        {...}, ..., {...}
                    ]
                }

            Each run dict must at least have keys 'good_tstart', 'good_tstop'
            and 'run'. Times are given in iso formatted strings and run numbers
            as integers as shown above.
        filter_runs : function, optional
            Filter function to remove unwanted runs from the goodrun list.
            Called as `filter_runs(run)`. Function must operate on a single
            run dictionary element. If None, every run is used. (default: None)

        Notes
        -----
        .. [1] https://live.icecube.wisc.edu/snapshots/
        """
        super(RunlistBGRateInjector, self).__init__(rate_func, random_state)

        # Create a goodrun list from the JSON snapshot
        runlist = os.path.abspath(runlist)
        if filter_runs is None:
            def filter_runs(run):
                return True

        self._goodrun_dict = self.create_goodrun_dict(runlist, filter_runs)

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
        h = self._create_runtime_hist(T, self._goodrun_dict, remove_zero_runs)

        rate = h["rate"]
        # Use relativ poisson error as LSQ weights, but ignore empty bins
        w = np.zeros_like(rate)
        m = (rate > 0)
        w[m] = rate[m] / np.sqrt(rate[m])
        binmids = 0.5 * (h["start_mjd"] + h["stop_mjd"])

        resx = self._rate_func.fit(binmids, rate, p0=x0, w=w, **kwargs)

        # Set wrappers for functions with best fit pars plugged in and livetime
        # class variables as promised
        self._livetime = np.sum(h["runtime"])
        return self._set_best_fit(resx)

    def create_goodrun_dict(self, runlist, filter_runs):
        """
        Create a dict of lists from a runlist in JSON format.
        Each entry in each list is one run.

        Parameters
        ----------
        runlist, filter_runs
            See :py:meth:`RunlistBGRateInjector`, Parameters

        Returns
        -------
        goodrun_dict : dict
            Dictionary with run attributes as keys. The values are stored in
            arrays in each key.
        """
        if "runs" not in runlist.keys():
            raise ValueError("Runlist misses key 'runs' on top level")

        # This is a list of dicts (one dict per run)
        goodrun_list = runlist["runs"]

        # Filter to remove unwanted runs
        goodrun_list = list(filter(filter_runs, goodrun_list))

        # Convert the run list of dicts to a dict of arrays for easier handling
        goodrun_dict = dict(zip(goodrun_list[0].keys(),
                                zip(*[r.values() for r in goodrun_list])))
        for k in goodrun_dict.keys():
            goodrun_dict[k] = np.array(goodrun_dict[k])

        # Add times to MJD floats
        goodrun_dict["good_start_mjd"] = astrotime(
            goodrun_dict["good_tstart"], format="iso").mjd
        goodrun_dict["good_stop_mjd"] = astrotime(
            goodrun_dict["good_tstop"], format="iso").mjd

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
            See :py:meth:`RunlistBGRateInjector.fit`, Parameters
        goodrun_dict
            See :py:meth:`RunlistBGRateInjector.create_goodrun_dict`, Returns

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

        self._rate_rec = rate_rec
        return rate_rec


class BinnedBGRateInjector(BGRateInjector):
    def __init__(self, rate_func, random_state=None):
        """
        Binned Background Rate Injector

        Creates the time depending rate from already binned rates.

        Parameters
        ----------
        %(BGRateInjector.init.parameters)s
        """
        super(BinnedBGRateInjector, self).__init__(rate_func, random_state)
        return

    @docs.dedent
    def fit(self, tbins, rate, w=None, x0=None, **kwargs):
        """
        %(BGRateInjector.fit.summary)s

        Takes data and a binning derived from the runlist. Bins the data,
        normalizes to a rate in HZ and fits a RateFunction over the whole
        time span to it. This function serves as a rate per time model.

        Parameters
        ----------
        tbins : array-like, shape (nruns, 2)
            Explicit edges `[[start1, stop1], ..., [startN, stopN]]` of the
            pre-created time bins in MJD days. First columns is start MJD,
            second column is stop MJD for each run.
        rate : array-like, shape (nruns)
            Rates at given times `t` in Hz.
        w : array-like, shape(nruns), optional
            Weights for least squares fit: :math:`\sum_i (w_i * (y_i - f_i))^2`.
            (default: None)
        x0 : array-like, optional
            Seed values for the fit function as described above. If None,
            defaults from `RateFunction` are used. (default: None)
        kwargs
            Other arguments are passed to the `scipy.optimize.minimize` method.

        Returns
        -------
        best_estimator : function
            Rate function with the best fit parameters plugged in.
        """
        if tbins.shape != (len(rate), 2):
            raise ValueError("`tbins` must have shape (nruns, 2) and the " +
                             "same length as `rate`.")

        start_mjd = tbins[:, 0]
        stop_mjd = tbins[:, 1]
        binmids = 0.5 * (start_mjd + stop_mjd)
        resx = self._rate_func.fit(binmids, rate, p0=x0, w=w, **kwargs)

        # Set wrappers for functions with best fit pars plugged in and livetime
        # class variables as promised
        self._livetime = np.sum(stop_mjd - start_mjd)
        return self._set_best_fit(resx)
