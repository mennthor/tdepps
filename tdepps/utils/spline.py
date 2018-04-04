# coding: utf-8

"""
Collection of spline related helper methods.
"""

from __future__ import print_function, division, absolute_import
from builtins import map
from future import standard_library
standard_library.install_aliases()

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
from scipy.stats import chi2

from .phys import make_rate_records, rebin_rate_rec
from .io import logger
log = logger(name="utils.spline", level="ALL")


def make_spl_edges(vals, bins, w=None):
    """
    Make well behaved edge conditions for a spline fit by linear extrapolation.

    Parameters
    ----------
    vals : array-like
        Histogram values for the spline fit. Edges are created to make well
        behaved boundary conditions.
    bins : array-like, shape (len(vals), )
        Histogram bin edges.
    w : array-like or None
        Additional weight array that may be used in a spline fit later. If not
        ``None``, ``w`` is shaped as the values array, the outermost points get
        the minimal weight from the original entries: ``w[[0, -1]] = min(w)``.

    Returns
    -------
    vals : array-like
        y-values for the spline fit, same length as ``pts``.
    pts : array-like
        x-values for the spline fit.
    w : array-like or None
        Weights for the spline fit, same length as ``pts``.
    """
    vals = np.atleast_1d(vals)
    bins = np.atleast_1d(bins)
    if len(vals) != len(bins) - 1:
        raise ValueError("Bin egdes must have length `len(vals) + 1`.")
    if w is not None:
        w = np.atleast_1d(w)
        if len(w) != len(vals):
            raise ValueError("Weights must have same length as vals")
        # Give appended artificial points the lowest priority
        w = np.concatenate(([np.amin(w)], w, [np.amin(w)]))

    # Linear extrapolate outer bin edges to avoid uncontrolled edge behaviour:
    # f(x_i+1 + (x_i+1 - x_i) / 2)
    #  = (y_i+1 - y_i) / (x_i+1 + x_i) * (x_i+1 + (x_i+1 - x_i) / 2) + y_i
    val_l = (3. * vals[0] - vals[1]) / 2.
    val_r = (3. * vals[-1] - vals[-2]) / 2.

    vals = np.concatenate(([val_l], vals, [val_r]))
    mids = 0.5 * (bins[:-1] + bins[1:])
    pts = np.concatenate((bins[[0]], mids, bins[[-1]]))
    return vals, pts, w


def fit_spl_to_hist(h, bins, w=None, s=None, k=3, ext="raise"):
    """
    Takes histogram values and bin edges and returns a spline fitted through the
    bin mids.

    Parameters
    ----------
    h : array-like
        Histogram values.
    bins : array-like
        Histogram bin edges.
    w : array-like or None
        Weights to use for the smoothing condition, eg. standard deviation of
        histogram entries. If ``None`` the spline is interpolating (``s=0``).
        (Default: ``None``)
    s, k, ext
        Arguments passed to ``scipy.interpolate.UnivariateSpline``.

    Returns
    -------
    spl : scipy.interpolate.UnivariateSpline
        Spline object fitted to the histogram values at the binmids.
    """
    h, bins = map(np.atleast_1d, [h, bins])
    if len(h) != len(bins) - 1:
        raise ValueError("Bin edges must have length `len(h) + 1`.")

    if w is None:
        s = 0
    else:
        if len(h) != len(w):
            raise ValueError("Length of weights and histogram muste be equal.")
        if np.any(w <= 0):
            raise ValueError("Given weights have unallowed entries <= 0.")
        w = np.atleast_1d(w)
        if s is None:
            # Set dof via s here, because `make_spl_edges` adds in edge points
            s = len(h)

    vals, pts, w = make_spl_edges(h, bins, w=w)
    return sci.UnivariateSpline(pts, vals, s=s, w=w, k=k, ext=ext), vals, pts, w


def make_time_dep_dec_splines(ev_t, ev_sin_dec, srcs, run_dict, sin_dec_bins,
                              rate_rebins, spl_s=None, n_scan_bins=50):
    """
    Make a declination PDF spline averaged over each sources time window.

    Parameters
    ----------
    ev_t : array-like, shape (nevts)
        Experimental per event event times in MJD days.
    ev_sin_dec : array-like, shape (nevts)
        Experimental per event ``sin(declination)`` values.
    srcs : record-array
        Must have names ``'t', 'dt0', 'dt1'`` describing the time intervals
        around the source times to sample from.
    run_dict : dictionary
        Dictionary with run information, matching the experimental data. Can be
        obtained from ``create_run_dict``.
    sin_dec_bins : array-like
        Explicit bin edges in ``sin(dec)`` used to bin ``sin_decs``.
    rate_rebins : array-like
        Explicit bin edges used to rebin the rates before fitting the model to
        achieve more stable fit conditions.
    spl_s : float, optional
        Smoothing condition for the parameter spline fits, controlling how much
        the spline is allowed to deviate from the data points, where ``spl_s=0``
        means interpolation. If ``None`` the default for
        ``scipy.interpolate.UnivariateSpline`` is used. (Default: ``None``)

    Returns
    -------
    sin_dec_splines : list of scipy.interpolate.UnivariateSpline
        For each given source time and range, a declination PDF spline, so that
        the integral over the spline over the given sin dec range is 1.
    info : dict
        Collection of various intermediate results and fit information.
    """
    # http://stackabuse.com/python-circular-imports
    from ..grb import SinusFixedConstRateFunction

    ev_t = np.atleast_1d(ev_t)
    ev_sin_dec = np.atleast_1d(ev_sin_dec)
    src_t = np.atleast_1d(srcs["t"])
    src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T
    sin_dec_bins = np.atleast_1d(sin_dec_bins)
    rate_rebins = np.atleast_1d(rate_rebins)

    rate_rec = make_rate_records(run_dict=run_dict, T=ev_t)
    norm = np.diff(sin_dec_bins)

    # 1) Get phase offset from allsky fit for good amp and baseline fits.
    #    Only fix the period to 1 year, as expected from seasons.
    p_fix = 365.
    rate_func_allsky = SinusFixedConstRateFunction(p_fix=p_fix)
    #    Fit amp, phase and base using rebinned rates
    rates, new_rate_bins, rates_std, _ = rebin_rate_rec(
        rate_rec, bins=rate_rebins, ignore_zero_runs=True)
    rate_bin_mids = 0.5 * (new_rate_bins[:-1] + new_rate_bins[1:])
    weights = 1. / rates_std
    #    Empirical seeds to get a good fit
    min_rate, max_rate = np.amin(rates), np.amax(rates)
    min_ev_t = np.amin(ev_t)
    seed_reb = (-0.5 * (max_rate - min_rate),  # Optimized for southern hemisp.
                min_ev_t,
                np.average(rates, weights=weights))
    bounds = [[-2. * max_rate, 2. * max_rate],
              [seed_reb[1] - 180., seed_reb[1] + 180.],
              [0., None]]
    fitres_allsky = rate_func_allsky.fit(t=rate_bin_mids, rate=rates,
                                         srcs=srcs, p0=seed_reb, w=weights,
                                         bounds=bounds)
    #    This is used to fix the time phase approximately at the correct
    #    baseline in the following fits, because a 3 param fit yields large
    #    errors, due to strong correlation between amp and phase, so we can't
    #    use them to build a proper spline representation.
    phase_bf_fix = fitres_allsky.x[1]

    # 2) For each sin_dec_bin fit a separate rate model in amp and base.
    rate_func = SinusFixedConstRateFunction(p_fix=p_fix, t0_fix=phase_bf_fix)
    names = ["amp", "base"]
    nbins = len(sin_dec_bins) - 1
    best_pars = np.empty((nbins, ), dtype=[(n, float) for n in names])
    std_devs = np.empty_like(best_pars)
    best_pars_norm = np.empty_like(best_pars)
    std_devs_norm = np.empty_like(std_devs)
    for i, (lo, hi) in enumerate(zip(sin_dec_bins[:-1], sin_dec_bins[1:])):
        print(log.INFO("sindec bin {} / {}".format(i + 1, nbins)))
        # Only make rates for the current bin and fit rate func in amp and base
        mask = (ev_sin_dec >= lo) & (ev_sin_dec < hi)
        rate_rec = make_rate_records(ev_t[mask], run_dict, eps=0.,
                                     all_in_err=False)
        rates, _, rates_std, _ = rebin_rate_rec(
            rate_rec, bins=rate_rebins, ignore_zero_runs=True)
        weights = 1. / rates_std
        min_rate, max_rate = np.amin(rates), np.amax(rates)
        p0 = (-0.5 * (max_rate - min_rate), np.average(rates, weights=weights))
        bounds = [[-2. * max_rate, 2. * max_rate], [0., None]]
        fitres = rate_func.fit(t=rate_bin_mids, rate=rates, srcs=srcs, p0=p0,
                               w=weights, bounds=bounds)

        # Scan the LLH to get stdev estimates.
        args = (rate_bin_mids, rates, weights)
        # Use empirical range estimates, amp seems to have larger errors
        rngs = np.array([fitres.x[0] / 10., fitres.x[1] / 100.])
        stds = get_stddev_from_scan(rate_func._lstsq, args, bfs=fitres.x,
                                    rngs=rngs, nbins=n_scan_bins)[0]

        # Store normalized best pars and fit stddevs to build a spline model
        for j, n in enumerate(names):
            best_pars[n][i] = fitres.x[j]
            std_devs[n][i] = stds[j]
            best_pars_norm[n][i] = fitres.x[j] / norm[i]
            std_devs_norm[n][i] = stds[j] / norm[i]

    # 3) Interpolate discrete fit points with a continous smoothing spline
    def spl_normed_factory(spl, lo, hi, norm):
        """ Renormalize spline, so ``int_lo^hi renorm_spl dx = norm`` """
        return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)

    param_splines = {}
    lo, hi = sin_dec_bins[0], sin_dec_bins[-1]
    sin_dec_pts = np.linspace(lo, hi, 1000)
    norm_allsky = {names[0]: fitres_allsky.x[0],
                   names[-1]: fitres_allsky.x[-1]}
    for n in names:
        # Use normalized amplitude and baseline in units HZ/dec
        weights = 1. / std_devs_norm[n]
        # Small feedback loop to catch large 2nd derivates, meaning extremely
        # large fluctuations between the points. Empirical correction values...
        OK = False
        spl_s_ = spl_s
        while not OK:
            spl = fit_spl_to_hist(
                best_pars_norm[n], sin_dec_bins, weights, s=spl_s_)[0]
            spl2 = spl.derivative(n=2)
            derivative2 = np.abs(spl2(sin_dec_pts))
            # Empirically found value that is working OK
            OK = np.all(derivative2 < 1.)
            if not OK:
                spl_s_ *= 0.9
                print(log.INFO("Degraded smoothing factor, 2nd derivative " +
                               " was {:.2f}".format(np.amax(derivative2))))
        # Renormalize to match the allsky params, because the model is additive
        scale = norm_allsky[n] / spl.integral(lo, hi)
        best_pars_norm[n] = best_pars_norm[n] * scale
        std_devs_norm[n] = std_devs_norm[n] * scale
        param_splines[n] = spl_normed_factory(
            spl, lo=lo, hi=hi, norm=norm_allsky[n])

    # 4) For each source time window build a sindec PDF spline.
    #    Each spline is averaged over the time range, so this only works for
    #    reasonably small windows in which fluctuations don't get averaged out.
    #    Get all rate model params from splines at sin_dec support points. Nr of
    #    is arbitrary and chosen to have enough resolution in sin_dec, using
    #    linear splines as they are fastest and good enough with many pts.
    def spl_factory(x, y, k=1, ext="raise"):
        """ Factory returning a new UnivariateSpline object """
        return sci.InterpolatedUnivariateSpline(x, y, k=k, ext=ext)

    # Broadcast params to get the rate func vals for each sindec
    amp = param_splines["amp"](sin_dec_pts)
    base = param_splines["base"](sin_dec_pts)
    sin_dec_splines = []
    for ti, tri in zip(src_t, src_trange):
        # Average over source time window
        vals = rate_func.integral(t=ti, trange=tri, pars=(amp, base))
        spl = sci.InterpolatedUnivariateSpline(
            sin_dec_pts, vals, k=1, ext="raise")
        # Leave splines unnormalized, unit is then events / dec
        sin_dec_splines.append(spl_factory(sin_dec_pts, vals))

    info = {"allsky_rate_func": rate_func_allsky,
            "allsky_best_params": fitres_allsky.x,
            "param_splines": param_splines,
            "best_pars": best_pars,
            "best_stddevs": std_devs,
            "best_pars_norm": best_pars_norm,
            "best_stddevs_norm": std_devs_norm}
    return sin_dec_splines, info


def get_stddev_from_scan(func, args, bfs, rngs, nbins=50):
    """
    Scan the rate_func chi2 fit LLH to get stddevs for the best fit params a, d.
    Using matplotlib contours and averaging to approximately get the variances.
    Method tries to adapt the scan ranges automatically to get the best contour
    resolution.
    Note: This is not a true LLH profile scan in both variables.

    Parameters
    ----------
    func : callable
        Loss function to be scanned, used to obtain the best fit. Function
        is called as done with a scipy fitter, ``func(x, *args)``.
    args : tuple
        Args passed to the loss function ``func``. For a rate function, this is
        ``(mids, rates, weights)``.
    bfs : array-like, shape (2)
        Best fit result parameters.
    rngs : list
        Parameter ranges to scan: ``[bf[i] - rng[i], bf[i] + rng[i]]``.
    nbins : int, optional
        Number of bins in each dimension to scan. (Default: 100)

    Returns
    -------
    stds : array-like, shape (2)
        Approximate standard deviations (symmetric) for each fit parameter,
        obtained using Wilks' theorem on the scanned space.
    llh : array-like, shape (nbins, nbins)
        Scanned LLH values.
    grid : list
        X, Y grid, same shape as ``llh``.
    """
    def _scan_llh(bf_x, rng_x, bf_y, rng_y):
        """ Scan LLH and return contour vertices """
        x_bins = np.linspace(bf_x - rng_x, bf_x + rng_x, nbins)
        y_bins = np.linspace(bf_y - rng_y, bf_y + rng_y, nbins)
        x, y = np.meshgrid(x_bins, y_bins)
        AA, DD = map(np.ravel, [x, y])
        llh = np.empty_like(AA)
        for i, (ai, di) in enumerate(zip(AA, DD)):
            llh[i] = func((ai, di), *args)
        llh = llh.reshape(x.shape)
        # Get the contour points and average over min, max per parameter
        one_sigma_level = np.amin(llh) - chi2.logsf(df=2, x=[1**2])
        cntr = plt.contour(x, y, llh, one_sigma_level)
        plt.close("all")
        paths = [lincol.vertices for lincol in cntr.collections[0].get_paths()]
        return paths, llh, [x, y]

    def _is_path_closed(paths, rng_x, rng_y):
        """
        We want the contour to be fully contained. Means there is only one path
        and the first and last point are close together.
        Returns ``True`` if contour is closed.
        """
        closed = False
        if len(paths) == 1:
            vertices = paths[0]
            # If no contour is made, only 1 vertex is returned -> invalid
            if len(vertices) > 1:
                max_bin_dist = np.amax([rng_x / float(nbins),
                                        rng_y / float(nbins)])
                closed = np.allclose(vertices[0], vertices[-1],
                                     atol=max_bin_dist, rtol=0.)
        return closed

    def _get_stds_from_path(path):
        """ Create symmetric stddevs from the path vertices """
        x, y = path[:, 0], path[:, 1]
        # Average asymmetricities in both direction
        x_min, x_max = np.amin(x), np.amax(x)
        y_min, y_max = np.amin(y), np.amax(y)
        return 0.5 * (x_max - x_min), 0.5 * (y_max - y_min)

    # Scan the LLH, adapt scan range if contour is not closed
    bf_x, bf_y = bfs
    rng_x, rng_y = rngs
    closed = False
    # We do something wrong, when not converging after so much iterations
    n_rescans = 0
    RAISE_N_RESCANS = 100
    while not closed:
        # Reset after each scan. Default is scaling up, when range is too small.
        scalex, scaley = 10., 10.
        # Get contour from scanned LLH space
        paths, llh, grid = _scan_llh(bf_x, rng_x, bf_y, rng_y)
        if _is_path_closed(paths, rng_x, rng_y):
            vertices = paths[0]
            # Estimate scale factors to get contour in optimum resolution and
            # check if the range is zoomed way to far out
            diffx = np.abs(np.amax(vertices[:, 0]) - np.amin(vertices[:, 0]))
            diffy = np.abs(np.amax(vertices[:, 1]) - np.amin(vertices[:, 1]))
            scalex = diffx / rng_x
            scaley = diffy / rng_y
            # Contour can be closed, but extremely zoomed out in only one param
            if not np.allclose([scalex, scaley], 1., atol=0.5, rtol=0.):
                print(log.INFO(
                    "LLH contour is very distorted in one direction"))
                closed = False
            else:
                # Rescan valid contour 2 times to use optimal scan resolution.
                # 2nd scan is useful when resolution was coarse in 1st try
                for i in range(2):
                    std_x, std_y = _get_stds_from_path(vertices)
                    rng_x = std_x * 1.05  # Allow a little padding
                    rng_y = std_y * 1.05
                    paths, llh, grid = _scan_llh(bf_x, rng_x, bf_y, rng_y)
                    # Recheck if path is still valid
                    closed = _is_path_closed(paths, rng_x, rng_y)
        # Must always be checked because path can get invalid in rescaling step
        if not closed:
            print(log.INFO("Open or no LLH contour, rescale scan ranges."))
            rng_x *= scalex
            rng_y *= scaley

        n_rescans += 1
        if n_rescans > RAISE_N_RESCANS:
            raise RuntimeError(log.INFO(
                "LLH scan not converging. Check seeds!"))

    vertices = paths[0]
    stds = np.array(_get_stds_from_path(vertices))
    return stds, llh, grid


def make_grid_interp_from_hist_ratio(h_bg, h_sig, bins, edge_fillval,
                                     interp_col_log, force_y_asc):
    """
    Create a 2D regular grind interpolator to describe the ratio
    ``h_sig / h_bg`` of two 2D histograms. The interpolation is done in the
    natural logarithm of the histogram ratio

    When the original histograms have empty entries a 2 step method of filling
    them is used: First all edge values in each x colum are filled. Then missing
    entries within each coumn are interpolated using existing and the new edge
    entries. Because this is only done in ``y`` direction, each histogram column
    (x bin) needs at least one entry.

    Parameters
    ----------
    h_bg : array-like, shape (len(x_bins), len(y_bins))
        Histogram for variables ``x, y`` for the background distribution.
    h_sig : array-like, shape (len(x_bins), len(y_bins))
        Histogram for variables ``x, y`` for the signal distribution.
    bins : list of array-like
        Explicit x and y bin edges used to make the histogram.
    edge_fillval : string, optional
        Fill values to use when the background histogram has no entries at
        the higest and lowest bins per column:
        - 'minmax': Use the low/high global ratio vals at the bottom/top edges.
        - 'col': Next valid value in each colum from the top/bottom is used.
        - 'minmax_col': Like 'minmax' but use min/max value per bin.
        - 'min': Only the lowest global ratio value is used at all edges.
        Filling is always done in y direction only. Listed in order from
        optimistic to conservative. (default: 'minmax_col')
    interp_col_log : bool, optional
        If ``True``, remaining gaps after filling the edge values in the signal
        over background ratio histogram are interpolated linearly in
        ``log(ratio)`` per column. Otherwise the interpolation is in linear
        space per column. (default: ``False``)
    force_y_asc : bool, optional
        If ``True``, assume that in each column the distribution ``y`` must be
        monotonically increasing. If it is not, a conservative approach is used
        and going from the top to the bottom edge per column, each value higher
        than its predecessor is shifted to its predecessor's value until we
        arrive at the bottom edge.
        Note: If ``True``, using 'min' in ``edge_fillval`` makes no sense, so a
        ``ValueError`` is thrown. (default: ``False``)

    Returns
    -------
    interp : scipy.interpolate.RegularGridInterpolator
        2D interpolator for the logarithm of the histogram ratio:
        ``interp(x, y) = log(h_sig / h_bg)(x, y)``. Exponentiate to obtain the
        original ratios. Interpolator returns 0, if points outside given
        ``bins`` domain are requested.
    """
    if edge_fillval not in ["minmax", "col", "minmax_col", "min"]:
        raise ValueError("`edge_fillval` must be one of " +
                         "['minmax'|'col'|'minmax_col'|'min'].")
    if edge_fillval == "min" and force_y_asc:
        raise ValueError("`edge_fillval` is 'min' and 'force_y_asc' is " +
                         "`True`, which doesn't make sense together.")

    # Create binmids to fit spline to bin centers
    x_bins, y_bins = map(np.atleast_1d, bins)
    x_mids, y_mids = map(lambda b: 0.5 * (b[:-1] + b[1:]), [x_bins, y_bins])
    nbins_x, nbins_y = len(x_mids), len(y_mids)

    # Check if hist shape fits to given binning
    if h_bg.shape != h_sig.shape:
        raise ValueError("Histograms don't have the same shape.")
    if h_bg.shape != (nbins_x, nbins_y):
        raise ValueError("Hist shapes don't match with number of bins.")

    # Check if hists are normed and do so if they are not
    dA = np.diff(x_bins)[:, None] * np.diff(y_bins)[None, :]
    if not np.isclose(np.sum(h_bg * dA), 1.):
        h_bg = h_bg / (np.sum(h_bg) * dA)
    if not np.isclose(np.sum(h_sig * dA), 1.):
        h_sig = h_sig / (np.sum(h_sig) * dA)
    assert np.isclose(np.sum(h_bg * dA), 1.)
    assert np.isclose(np.sum(h_sig * dA), 1.)

    # Check that all x bins in the bg hist have at least one entry
    mask = (np.sum(h_bg, axis=1) <= 0.)
    if np.any(mask):
        raise ValueError("Got empty x bins, this must not happen. Empty " +
                         "bins idx:\n{}".format(np.arange(nbins_x)[mask]))

    # Step 1: Construct simple ratio where we have valid entries
    sob = np.ones_like(h_bg) - 1.  # Use invalid value for init
    mask = (h_bg > 0) & (h_sig > 0)
    sob[mask] = h_sig[mask] / h_bg[mask]
    # Step 2: First fill all y edge values per column where no valid values
    # are, then interpolate missing inner ratios per column.
    if edge_fillval in ["minmax", "min"]:
        sob_min, sob_max = np.amin(sob[mask]), np.amax(sob[mask])
    for i in np.arange(nbins_x):
        if force_y_asc:
            # Rescale valid bins from top to bottom, so that b_i >= b_(i+1)
            mask = (sob[i] > 0)
            masked_sob = sob[i][mask]
            for j in range(len(masked_sob) - 1, 0, -1):
                if masked_sob[j] < masked_sob[j - 1]:
                    masked_sob[j - 1] = masked_sob[j]
            sob[i][mask] = masked_sob

        # Get invalid points in current column
        m = (sob[i] <= 0)

        if edge_fillval == "minmax_col":
            # Use min/max per slice instead of global min/max
            sob_min, sob_max = np.amin(sob[i][~m]), np.amax(sob[i][~m])

        # Fill missing top/bottom edge values, rest is interpolated later
        # Lower edge: argmax stops at first True, argmin at first False
        low_first_invalid_id = np.argmax(m)
        if low_first_invalid_id == 0:
            # Set lower edge with valid point, depending on 'edge_fillval'
            if edge_fillval == "col":
                # Fill with first valid ratio from bottom for this column
                low_first_valid_id = np.argmin(m)
                sob[i, 0] = sob[i, low_first_valid_id]
            elif edge_fillval in ["minmax", "minmax_col", "min"]:
                # Fill with global min or with min for this col
                sob[i, 0] = sob_min

        # Repeat with turned around array for upper edge
        hig_first_invalid_id = np.argmax(m[::-1])
        if hig_first_invalid_id == 0:
            if edge_fillval == "col":
                # Fill with first valid ratio from top for this column
                hig_first_valid_id = len(m) - 1 - np.argmin(m[::-1])
                sob[i, -1] = sob[i, hig_first_valid_id]
            elif edge_fillval == "min":
                # Fill also with global min
                sob[i, -1] = sob_min
            elif edge_fillval in ["minmax", "minmax_col"]:
                # Fill with global max or with max for this col
                sob[i, -1] = sob_max

        # Interpolate missing entries in the current column
        mask = (sob[i] > 0)
        _x = y_mids[mask]
        _y = sob[i, mask]
        if interp_col_log:
            col_interp = sci.interp1d(_x, np.log(_y), kind="linear")
            sob[i] = np.exp(col_interp(y_mids))
        else:
            col_interp = sci.interp1d(_x, _y, kind="linear")
            sob[i] = col_interp(y_mids)

    # Step 3: Construct a 2D interpolator for the log(ratio).
    # Repeat values at the edges (y then x) to cover full bin domain, so the
    # interpolator can throw an error outside the domain
    sob_full = np.zeros((nbins_x + 2, nbins_y + 2), dtype=sob.dtype) - 1.
    for j, col in enumerate(sob):
        sob_full[j + 1] = np.concatenate([col[[0]], col, col[[-1]]])
    sob_full[0] = sob_full[1]
    sob_full[-1] = sob_full[-2]
    # Build full support points
    pts_x = np.concatenate((x_bins[[0]], x_mids, x_bins[[-1]]))
    pts_y = np.concatenate((y_bins[[0]], y_mids, y_bins[[-1]]))

    interp = sci.RegularGridInterpolator([pts_x, pts_y], np.log(sob_full),
                                         method="linear", bounds_error=False,
                                         fill_value=0.)
    return interp


class spl_normed(object):
    """
    Simple wrapper to make and handle a normalized UnivariateSpline.

    The given spline is normalized so that integral over ``[lo, hi]`` is
    ``norm``. There might be a better way by directly inheriting from
    ``UnivariateSpline``, but this class is OK, if we don't need the full spline
    feature set.

    Note: Not all spline methods are available.

    Parameters
    ----------
    spl : scipy.interpolate.UnivariateSpline instance
        A spline object that shall be normlaized.
    norm : float
        The value the new spline's integral should have over ``lo, hi``.
    lo, hi : float
        Lower and upper integration borders over which the integral should be
        ``norm``.
    """
    def __init__(self, spl, norm, lo, hi):
        self._spl = spl
        if spl.integral(a=lo, b=hi) == 0:
            raise ValueError("Given spline has integral 0, can't scale it.")
        self._scale = norm / spl.integral(a=lo, b=hi)
        if np.isclose(self._scale, 1.):
            self._scale = 1.

    def __call__(self, x, nu=0, ext=None):
        return self._scale * self._spl(x, nu, ext)

    def antiderivative(self, n=1):
        return self._scale * self._spl.antiderivative(n)

    def derivative(self, n=1):
        return self._scale * self._spl.derivative(n)

    def derivatives(self, x):
        return self._scale * self._spl.derivatives(x)

    def get_coeffs(self):
        return self._scale * self._spl.get_coeffs()

    def get_knots(self):
        return self._spl.get_knots()

    def get_residual(self):
        raise NotImplementedError("Don't knwo how to do this.")

    def integral(self, a, b):
        return self._scale * self._spl.integral(a, b)

    def roots(self, ):
        return self._spl.roots()

    def set_smoothing_factor(self, s):
        raise NotImplementedError("Don't knwo how to do this.")
