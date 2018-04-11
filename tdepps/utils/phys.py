# coding: utf-8

"""
Collection of statistics related helper methods.
"""

from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from astropy.time import Time as astrotime

from .io import fill_dict_defaults


def power_law_flux(trueE, gamma=2., phi0=1., E0=1.):
    """
    Returns the unbroken power law flux :math:`\sim \phi_0 (E/E_0)^{-\gamma}`
    where the normalization is summed over both particle types nu and anti-nu.
    Defaults have no physical meaning, units must be adapted to used weights.

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
    return rate, new_bins, rate_std, deadtime


def make_src_records(dict_list, dt0, dt1):
    """
    Make a source record array from a list of source dict entries.

    Parameters
    ----------
    dict_list : list of dict
        One dict per source, must have keys ``'ra', 'dec', 'mjd'`` and
        optionally ``'w_theo'``.
    dt0 : float

    dt1 : float

    Parameters
    ----------
    src_recs : record_array
        Record array with source information, has names
        ``'ra', 'dec', 'mjd', 'dt0', 'dt1' 'w_theo'``.
    """
    nsrcs = len(dict_list)
    dtype = [("ra", float), ("dec", float), ("time", float),
             ("dt0", float), ("dt1", float), ("w_theo", float)]
    src_recs = np.empty((nsrcs,), dtype=dtype)
    for i, src in enumerate(dict_list):
        src = fill_dict_defaults(src, required_keys=["ra", "dec", "mjd"],
                                 opt_keys={"w_theo": 1.}, noleft=False)
        src_recs["ra"][i] = src["ra"]
        src_recs["dec"][i] = src["dec"]
        src_recs["time"][i] = src["mjd"]
        src_recs["w_theo"][i] = src["w_theo"]
        # Fill current time window
        src_recs["dt0"][i] = dt0
        src_recs["dt1"][i] = dt1

    return src_recs
