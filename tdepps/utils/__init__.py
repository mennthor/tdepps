# coding: utf-8

"""
Collection of helper methods used to build functionality in the toolkit or
directly in the top level code.
"""

from __future__ import absolute_import

from .coords import thetaphi2decra, cos_angdist, rotator
from .coords import get_pixel_in_sigma_region

from .io import dict_map, fill_dict_defaults
from .io import arr2str, all_equal, logger

from .misc import interval_overlap

from .phys import power_law_flux, make_src_records
from .phys import make_rate_records, rebin_rate_rec
from .phys import flux_model_factory

from .spline import spl_normed, make_spl_edges, fit_spl_to_hist
from .spline import make_time_dep_dec_splines, make_grid_interp_from_hist_ratio
from .spline import get_stddev_from_scan

from .stats import random_choice, delta_chi2, fit_chi2_cdf
from .stats import weighted_cdf, cdf_nzeros, percentile_nzeros
from .stats import emp_with_exp_tail_dist, scan_best_thresh, emp_dist
from .stats import sigma2prob, prob2sigma
from .stats import make_equdist_bins
