from ._utils import doc_inherit

import os
import numpy as np
from sklearn.utils import check_random_state
import json
from astropy.time import Time as astrotime


class BGRateInjector(object):
    """
    Background Rate Injector Interface

    Samples times of BG-like events for a given time frame.
    Describes a `fit` and a `sample` method.
    """
    def __init__(self):
        self._DESCRIBES = ["fit", "sample"]
        print("Interface only. Describes functions: ", self._DESCRIBES)
        return

    def fit(self, X):
        """
        Build the injection model with the provided data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to
            a single data point, each column is a coordinate.
        """
        raise NotImplementedError("BGInjector is an interface.")

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the fitted model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (defaults: 1)
        random_state : RandomState, optional
            A random number generator instance. (default: None)

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            Generated samples from the fitted model.
        """
        raise NotImplementedError("BGInjector is an interface.")


class RunlistBGRateInjector():
    """
    Runlist Background Rate Injector

    Creates the time depending rate from a given runlist.
    Parameters are passed to `create_goodrun_dict` method.
    """
    def __init__(self, runlist, filter_runs):
        # Create a goodrun list from the JSON snapshot
        runlist = os.path.abspath(runlist)
        self.goodrun_dict = self.create_goodrun_dict(runlist, filter_runs)

        # Create runtime bins and rates from the goodrun list
        self._create_runtime_bins(self.goodrun_dict)
        return

    def fit(self, X, fit_seed):
        """
        Build the injection model with the provided data.

        Takes data and a binning derived from the runlist. Bins the data,
        normalizes to a rate in HZ and fits a periodic function over the whole
        time span to it. This function serves as a rate per time model.

        The function is chosen to be a sinus with:

        ..math:: f(t|a,b,c,d) = a \sin(b (t - c)) + d

        where

        - a is the Amplitude in Hz
        - b is the period scale in 1/MJD
        - c is the x-offset in MJD
        - d the y-offset in Hz

        Parameters
        ----------
        X : array_like, shape (n_samples)
            MJD times of experimental data.
        fit_seed : array-like, shape (4)
            Seed values for the fit function as described above.
            If None, defaults are used:
            - a0: (max(rate) - min(rate)) / 2 of data bins
            - b0: 2pi / 365, as the usual seasonal variation is 1 year
            - c0: min(X), first datapoint

        Returns
        -------
        rate_fun : function
            Function with the best fit parameters plugged in.
        """
        rates, bins = self._create_runtime_hist()

    @doc_inherit
    def sample(self, n_samples=1, random_state=None):
        rndgen = check_random_state(random_state)
        raise NotImplementedError("BGInjector is an interface.")
        return

    def create_goodrun_dict(self, runlist, filter_runs):
        """
        Create a dict of lists. Each entry in each list is one run.

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

        Returns
        -------
        goodrun_dict : dict
            Dictionary with run attributes as keys. The values are stored in
            lists in each key. One list item is one run.

        Notes
        -----
        .. [1] https://live.icecube.wisc.edu/snapshots/
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

        self.livetime = np.sum(goodrun_dict["runtime_days"])

        return goodrun_dict

    # Private Methods
    def _create_runtime_bins(self, X):
        # Store events in bins with run borders
        exp_times = exp["timeMJD"]
        start_mjd = inc_run_arr["start_mjd"]
        stop_mjd = inc_run_arr["stop_mjd"]

        tot = 0
        evts_in_run = {}
        for start, stop , runid in zip(start_mjd, stop_mjd, inc_run_arr["runID"]):
            mask = (exp_times >= start) & (exp_times < stop)
            evts_in_run[runid] = exp[mask]
            tot += np.sum(mask)

        # Crosscheck, if we got all events and counted nothing double
        print("Do we have all events? ", tot == len(exp))
        print("  Events selected : ", tot)
        print("  Events in exp   : ", len(exp))

        # Create binmids and histogram values in each bin
        binmids = 0.5 * (start_mjd + stop_mjd)
        h = np.zeros(len(binmids), dtype=np.float)

        for i, evts in enumerate(evts_in_run.values()):
            h[i] = len(evts)

        # Now remove the 120 runs with zero rate that come from the differences
        # in the runlist. See side_test for more
        m = (h > 0)
        print("\nRuns with 0 events :", np.sum(~m))
        print("Runtime in those runs: ", np.sum(inc_run_arr["stop_mjd"][~m] -
                                                inc_run_arr["start_mjd"][~m]))

        # Remove all zero event runs (artifacts from new run list) and calc the rate
        stop_mjd, start_mjd = stop_mjd[m], start_mjd[m]
        h = h[m]

        print("\nHave all events after removing zero rates? ", np.sum(h) == len(exp))
        print("  Events selected : ", int(np.sum(h)))
        print("  Events in exp   : ", len(exp))

        # Normalize to rate in Hz and calc yerrors for fitting later
        h /= ((stop_mjd - start_mjd) * secinday)
        binmids = binmids[m]
        yerr = np.sqrt(h) / np.sqrt((stop_mjd - start_mjd) * secinday)


# class TimebinBGRateInjector():
    # bins : array-like, shape (nbins, 2)
    #     Time bins, where every row represents the start and end time in MJD
    #     for a single run. This can be preselected from a goodrun list.


# class FunctionBGRateInjector():




















