# coding: utf-8

"""
Collection of statistics related helper methods.
"""

from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from astropy.time import Time as astrotime

from .io import fill_dict_defaults, logger
log = logger(name="utils.phys", level="ALL")


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


def make_rate_records(ev_runids, run_list):
    """
    Creates time bins ``[start_MJD_i, stop_MJD_i]`` for each run in ``run_dict``
    and bins the experimental data to calculate the rate for each run.

    Data selection should match the used run list to give reasonable results.
    Runs wih livetime 0 or no events get zero rate.

    Parameters
    ----------
    ev_runids : array_like, shape (n_samples)
        Per event run IDs to macth events and runs in ``run_list``.
    run_list : list of dicts
            List of dicts made from a good run runlist snapshot from [1]_. Must
            be a list of single runs of the following structure

                [{
                  "good_tstart": "YYYY-MM-DD HH:MM:SS",
                  "good_tstop": "YYYY-MM-DD HH:MM:SS",
                  "run": 123456, ...,
                  },
                 {...}, ..., {...}]

            Each run dict must at least have keys ``'good_tstart'``,
            ``'good_tstop'`` and ``'run'``. Times are given in iso formatted
            strings and run numbers as integers as shown above.

    Returns
    -------
    rate_rec : recarray, shape(nruns)
        Record array with keys:

        - "run" : int, ID of the run.
        - "nevts" : int, numver of events in this run.
        - "rate" : float, rate in Hz in this run.
        - "rate_std" : float, ``sqrt(nevts) / runtime`` scaled poisson standard
                       deviation of the rate in Hz in this run.
        - "runtime" : float, livetime of this run in MJD days.
        - "start_mjd" : float, MJD start time of the run.
        - "stop_mjd" : float, MJD end time of the run.
    """
    _SECINDAY = 24. * 60. * 60.

    # run_list must be a list of dicts (one dict to describe one run)
    if not np.all(map(lambda item: isinstance(item, dict), run_list)):
        raise TypeError("Not all entries in 'run_list' are dicts.")

    required_names = ["good_tstart", "good_tstop", "run"]
    for i, item in enumerate(run_list):
        for key in required_names:
            if key not in item.keys():
                raise KeyError("Runlist item '{}' ".format(i) +
                               "is missing required key '{}'.".format(key))

    # Convert the run list of dicts to a dict of arrays for easier handling
    run_dict = dict(zip(run_list[0].keys(),
                        zip(*[r.values() for r in run_list])))

    # Dict keys were not necessarly sorted, so sort the new lists after run id
    srt_idx = np.argsort(run_dict["run"])
    for k in required_names:
        run_dict[k] = np.atleast_1d(run_dict[k])[srt_idx]

    # Convert and add times in MJD float format
    start_mjds = astrotime(run_dict["good_tstart"], format="iso").mjd
    stop_mjds = astrotime(run_dict["good_tstop"], format="iso").mjd
    runs = run_dict["run"]

    # Get number of events per run
    ev_runids = np.atleast_1d(ev_runids)
    _nruns = len(runs)
    evts = np.empty(_nruns, dtype=np.int)
    for i, runid in enumerate(runs):
        evts[i] = np.sum(ev_runids == runid)

    # Normalize to rate in Hz, zero livetime runs (1 evt only) get zero rate
    runtime = stop_mjds - start_mjds
    rate = np.zeros_like(runtime, dtype=float)
    mask = (runtime > 0.)
    runtime_mjd = runtime[mask] * _SECINDAY
    rate[mask] = evts[mask] / runtime_mjd
    print(log.INFO("{} / {} runs with zero livetime.".format(np.sum(~mask),
                                                             _nruns)))

    # Calculate poisson sqrt(N) stddev for scaled rates
    rate_std = np.zeros_like(runtime, dtype=float)
    rate_std[mask] = np.sqrt(evts[mask]) / runtime_mjd

    # Create record-array
    names = ["run", "nevts", "rate", "rate_std",
             "runtime", "start_mjd", "stop_mjd"]
    types = [np.int, np.int, np.float, np.float, np.float, np.float, np.float]
    a = np.vstack((runs, evts, rate, rate_std, runtime, start_mjds, stop_mjds))

    return np.core.records.fromarrays(a, dtype=list(zip(names, types)))


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
        Rate information as build with ``utils.phys.make_rate_records``. Needs
        names ``'start_mjd', 'stop_mjd', 'rate'``.
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

    _m = (stop - start > 0)
    assert np.allclose(rate_rec["nevts"][_m],
                       rate_rec["rate"][_m] * (stop - start)[_m] * _SECINDAY)

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
                                 opt_keys={"w_theo": 1.}, noleft="drop")
        src_recs["ra"][i] = src["ra"]
        src_recs["dec"][i] = src["dec"]
        src_recs["time"][i] = src["mjd"]
        src_recs["w_theo"][i] = src["w_theo"]
        # Fill current time window
        src_recs["dt0"][i] = dt0
        src_recs["dt1"][i] = dt1

    return src_recs
