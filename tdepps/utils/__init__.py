# coding: utf-8

"""
Collection of helper methods used to build functionality in the toolkit or
directly in the top level code.
"""

from __future__ import absolute_import

from .coords import thetaphi2decra, cos_angdist, rotator

from .io import dict_map, fill_dict_defaults
from .io import arr2str, all_equal, logger

from .phys import power_law_flux, make_src_records
from .phys import create_run_dict, make_rate_records, rebin_rate_rec

from .spline import spl_normed, make_spl_edges, fit_spl_to_hist
from .spline import make_time_dep_dec_splines, make_grid_interp_from_hist_ratio
from .spline import get_stddev_from_scan

from .stats import random_choice, weighted_cdf, delta_chi2, fit_chi2_cdf
