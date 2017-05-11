import os
import json
import numpy as np
from sklearn.utils import check_random_state
from astropy.time import Time as astrotime

import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class BGRateInjector(object):
    """
    Background Rate Injector Interface

    Samples times of BG-like events for a given time frame.
    Describes a `fit` and a `sample` method.
    """
    # Set up globals for shared inherited constants
    _SECINDAY = 24. * 60. * 60.

    def __init__(self):
        self._DESCRIBES = ["fit", "sample"]
        print("Interface only. Describes functions: ", self._DESCRIBES)
        return

    @docs.get_summaryf("BGRateInjector.fit")
    @docs.get_sectionsf("BGRateInjector.fit",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    def fit(self, X):
        """
        Build the injection model with the provided data.

        Parameters
        ----------
        X : array_like, shape (n_samples)
            MJD times of experimental data.
        """
        raise NotImplementedError("BGInjector is an interface.")

    @docs.get_summaryf("BGRateInjector.sample")
    @docs.get_sectionsf("BGRateInjector.sample",
                        sections=["Parameters", "Returns"])
    @docs.dedent
    def sample(self, t, trange, n_samples=1, random_state=None):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        t : array-like
            Times of the occurance of each source event in MJD.
        trange : [float, float] or array_like, shape (len(t), 2)
            Time window(s) in seconds relativ to the given time(s) t.
            - If [float, float], the same window [lower, upper] is used for
              every time t.
            - If array-like [lower, upper] bounds of the time
              window per source in each row.
        n_samples : int, optional
            Number of samples to generate. (default: 1)
        random_state : RandomState, optional
            A random number generator instance. (default: None)

        Returns
        -------
        nevents : array-like, shape (len(t), ntrials)
            The number of events to inject for each trial in each source time
            window.
        """
        raise NotImplementedError("BGInjector is an interface.")


class RunlistBGRateInjector():
    @docs.get_sectionsf("RunlistBGRateInjector",
                        sections=["Parameters", "Notes"])
    @docs.dedent
    def __init__(self, runlist, filter_runs, rate_func):
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
            dictionary `run`, with keys:

                ['good_i3', 'good_it', 'good_tstart', 'good_tstop', 'run',
                 'reason_i3', 'reason_it', 'source_tstart', 'source_tstop',
                 'snapshot', 'sha']
        rate_func : `rate_function.RateFunction` instance
            Class defining the function to describe the time dependent
            background rate. Must provide functions ["fun", "integral", "fit"].

        Notes
        -----
        .. [1] https://live.icecube.wisc.edu/snapshots/
        """
        # Class defaults
        self.best_estimator = None

        # Create a goodrun list from the JSON snapshot
        runlist = os.path.abspath(runlist)
        self.goodrun_dict = self.create_goodrun_dict(runlist, filter_runs)

        self.rate_func = rate_func
        return

    @docs.dedent
    def fit(self, X, x0=None, remove_zero_runs=False, **kwargs):
        """
        %(BGRateInjector.fit.summary)s

        Takes data and a binning derived from the runlist. Bins the data,
        normalizes to a rate in HZ and fits a periodic function over the whole
        time span to it. This function serves as a rate per time model.

        Parameters
        ----------
        %(BGRateInjector.fit.parameters)s
        x0 : array-like, optional
            Seed values for the fit function as described above. If None,
            defaults are used. (default: None)
        remove_zero_runs : bool, optional
            If True, remove all runs with zero events and adapt the livetime.
            (default: False)
        kwargs
            Other arguments are passed to the `scipy.optimize.minimize` method.

        Returns
        -------
        rate_fun : function
            Function with the best fit parameters plugged in.
        """
        # Put data into run bins to fit them
        h = self._create_runtime_hist(X, self.goodrun_dict, remove_zero_runs)
        rate = h["rate"]
        rate_std = h["rate_std"]
        binmids = 0.5 * (h["start_mjd"] + h["stop_mjd"])

        resx = self.rate_func.fit(binmids, rate, p0=None, rate_std=rate_std)

        self.best_pars = resx
        self.best_estimator = self.rate_func.best_fun

        return self.best_estimator

    @docs.dedent
    def sample(self, t, trange, n_samples=1, random_state=None):
        """
        %(BGRateInjector.sample.summary)s

        Parameters
        ----------
        %(BGRateInjector.sample.parameters)s

        Returns
        -------
        %(BGRateInjector.sample.returns)s
        """
        if self.best_estimator is None:
            raise RuntimeError("Injector was not fit to data yet.")

        rndgen = check_random_state(random_state)

        # Expectations are the integrals over each time frames
        expect = self.rate_func.best_integral(t, trange)

        # Sample event numbers from poisson distribution
        nevents = rndgen.poisson(lam=expect, size=(len(t), n_samples))

        # For each event number sample times from the best fit rate function
        pdf = self.best_estimator

        t, trange = _prep_times(t, trange)

        sample = []
        nsamples = np.atleast_1d(nsamples)

        for i, ni in enumerate(nsamples):
            sam, _ = rejection_sampling(_pdf, bounds=trange, n=ni)
            sample.append(sam)

        return nevents

    @docs.get_sectionsf("RunlistBGRateInjector.create_goodrun_dict",
                        sections=["Returns"])
    @docs.dedent
    def create_goodrun_dict(self, runlist, filter_runs):
        """
        Create a dict of lists from a runlist in JSON format.
        Each entry in each list is one run.

        Parameters
        ----------
        %(RunlistBGRateInjector.parameters)s

        Returns
        -------
        goodrun_dict : dict
            Dictionary with run attributes as keys. The values are stored in
            lists in each key. One list item is one run.

        Notes
        -----
        %(RunlistBGRateInjector.notes)s
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
    def _create_runtime_hist(self, X, goodrun_dict, remove_zero_runs=False):
        """
        Creates time bins [start_MJD_i, stop_MJD_i] for each run i and bin the
        experimental data to calculate the rate for each run.

        Parameters
        ----------
        %(BGRateInjector.fit.parameters)s
        %(RunlistBGRateInjector.create_goodrun_dict.returns)s
        remove_zero_runs : bool, optional
            If True, remove all runs with zero events and adapt the livetime.
            (default: False)

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
            mask = (X >= start) & (X < stop)
            evts[i] = np.sum(mask)
            tot_evts += np.sum(mask)

        # Crosscheck, if we got all events and didn't double count
        if not tot_evts == len(X):
            print("Events selected : ", tot_evts)
            print("Events in X     : ", len(X))
            raise ValueError("Not all events in 'X' were sorted in bins. If " +
                             "this is intended, please remove them beforehand.")

        if remove_zero_runs:
            # Remove all zero event runs and update livetime
            m = (evts > 0)
            _livetime = np.sum(stop_mjd[~m] - start_mjd[~m])
            evts, run = evts[m], run[m]
            start_mjd, stop_mjd = start_mjd[m], stop_mjd[m]
            print("Removing runs with zero events")
            print("  Number of runs with 0 events : {:d}".format(np.sum(~m)))
            print("  Total livetime of those runs : {:.3f} d".format(_livetime))

        # Normalize to rate in Hz
        runtime = stop_mjd - start_mjd
        rate = evts / (runtime * BGRateInjector._SECINDAY)

        # Calculate 1 / sqrt(N) stddev for scaled rates
        rate_std = np.sqrt(rate) / np.sqrt(runtime * BGRateInjector._SECINDAY)

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
