# coding: utf-8

"""
Collection of helper methods used to build functionality in the toolkit or
directly in the top level code.
"""

from __future__ import absolute_import

from .coords import thetaphi2decra, cos_angdist, rotator
from .io import (arr2str, fill_dict_defaults, logger, create_run_dict,
                 all_equal, dict_map)
from .phys import power_law_flux, make_rate_records, rebin_rate_rec
from .spline import (make_spl_edges, fit_spl_to_hist, get_stddev_from_scan,
                     spl_normed, make_time_dep_dec_splines)
from .stats import random_choice, weighted_cdf, delta_chi2, fit_chi2_cdf
