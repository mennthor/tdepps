# coding: utf-8

"""
Collection of io releated helper methods.
"""

from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from astropy.time import Time as astrotime


class logger(object):
    """ Bad attempt to a simple logger. TODO: Replace with logging module """
    _modes = ["DEBUG", "INFO", "WARN", "ERROR", "ALL", "NONE"]

    def __init__(self, name="", level="INFO"):
        if level not in self._modes:
            raise ValueError("`level` mus be one of {}".format(
                arr2str(self._modes)))
        self._level = level
        self._name = name

    def INFO(self, s=""):
        if self._level in ["ALL", "INFO"]:
            return "INFO :: {} :: {}".format(self._name, s)

    def DEBUG(self, s=""):
        if self._level in ["ALL", "DEBUG"]:
            return "DEBUG :: {} :: {}".format(self._name, s)

    def WARN(self, s=""):
        if self._level in ["ALL", "WARN"]:
            return "WARNING :: {} :: {}".format(self._name, s)

    def ERROR(self, s=""):
        if self._level in ["ALL", "ERROR"]:
            return "ERROR :: {} :: {}".format(self._name, s)


def all_equal(a1, a2):
    """ ``True`` if ``a1`` and ``a2`` are equal (unsorted test) """
    if (len(a1) == len(a2)) and np.all(np.isin(a1, a2)):
        return True
    return False


def arr2str(arr, sep=", ", fmt="{}"):
    """
    Make a string from a list seperated by ``sep`` and each item formatted
    with ``fmt``.
    """
    return sep.join([fmt.format(v) for v in arr])


def dict_map(func, d):
    """
    Applies func to each dict value, returns results in a dict with the same
    keys as the original d.

    Parameters
    ----------
    func : callable
        Function ``func(key, val)`` applied to all ke, value pairs in ``d``.
    d : dict
        Dictionary which values are to be mapped.

    Returns
    -------
    out : dict
        New dict with same key as ``d`` and ``func`` applied to ``d.items()``.
    """
    return {key: func(key, val) for key, val in d.items()}


def fill_dict_defaults(d, required_keys=None, opt_keys=None, noleft=True):
    """
    Populate dictionary with data from a given dict ``d``, and check if ``d``
    has required and optional keys. Set optionals with default if not present.

    If input ``d`` is None and ``required_keys`` is empty, just return
    ``opt_keys``.

    Parameters
    ----------
    d : dict or None
        Input dictionary containing the data to be checked. If is ``None``, then
        a copy of ``opt_keys`` is returned. If ``opt_keys`` is ``None``, a
        ``TypeError`` is raised. If ``d``is ``None`` and ``required_keys`` is
        not, then a ``ValueError`` israised.
    required_keys : list or None, optional
        Keys that must be present  and set in ``d``. (default: None)
    opt_keys : dict or None, optional
        Keys that are optional. ``opt_keys`` provides optional keys and default
        values ``d`` is filled with if not present in ``d``. (default: None)
    noleft : bool, optional
        If True, raises a ``KeyError``, when ``d`` contains etxra keys, other
        than those given in ``required_keys`` and ``opt_keys``. (default: True)

    Returns
    -------
    out : dict
        Contains all required and optional keys, using default values, where
        optional keys were missing. If ``d`` was None, a copy of ``opt_keys`` is
        returned, if ``opt_keys`` was not ``None``.
    """
    if required_keys is None:
        required_keys = []
    if opt_keys is None:
        opt_keys = {}
    if d is None:
        if not required_keys:
            if opt_keys is None:
                raise TypeError("`d` and òpt_keys` are both None.")
            return opt_keys.copy()
        else:
            raise ValueError("`d` is None, but `required_keys` is not empty.")

    d = d.copy()
    out = {}
    # Set required keys
    for key in required_keys:
        if key in d:
            out[key] = d.pop(key)
        else:
            raise KeyError("Dict is missing required key '{}'.".format(key))
    # Set optional values, if key not given
    for key, val in opt_keys.items():
        out[key] = d.pop(key, val)
    # Complain when extra keys are left and noleft is True
    if d and noleft:
        raise KeyError("Leftover keys ['{}'].".format(
            "', '".join(list(d.keys()))))
    return out


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
