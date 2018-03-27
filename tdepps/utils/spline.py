# coding: utf-8

"""
Collection of spline related helper methods.
"""

from __future__ import print_function, division, absolute_import
from builtins import map
from future import standard_library
standard_library.install_aliases()

import numpy as np
from matplotlib import _cntr as contour
import scipy.interpolate as sci
from scipy.stats import chi2

from . import logger
log = logger(name="utils.spline", level="ALL")


def make_spl_edges(vals, bins, w=None):
    """
    Make nicely behaved edge conditions for a spline fit.

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
                              rate_rebins, spl_s=None):
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
    for i, (lo, hi) in enumerate(zip(sin_dec_bins[:-1], sin_dec_bins[1:])):
        print(_INFO_("sindec bin {} / {}".format(i + 1, nbins)))
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
                                    rngs=rngs, nbins=50)[0]

        # Store normalized best pars and fit stddevs to build a spline model
        for j, n in enumerate(names):
            best_pars[n][i] = fitres.x[j] / norm[i]
            std_devs[n][i] = stds[j] / norm[i]

    # 3) Interpolate discrete fit points with a continous smoothing spline
    def spl_normed_factory(spl, lo, hi, norm):
        """ Renormalize spline, so ``int_lo^hi renorm_spl dx = norm`` """
        return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)

    param_splines = {}
    lo, hi = sin_dec_bins[0], sin_dec_bins[-1]
    norm_allsky = {
        names[0]: fitres_allsky.x[0], names[-1]: fitres_allsky.x[-1]}
    for n in names:
        # Use normalized amplitude and baseline in units HZ/dec
        weights = 1. / std_devs[n]
        spl = fit_spl_to_hist(best_pars[n], sin_dec_bins, weights, s=spl_s)[0]
        # Renormalize to match the allsky params, because the model is additive
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

    sin_dec_pts = np.linspace(lo, hi, 1000)
    # Broadcast params to get the rate func vals for each sindec
    amp = param_splines["amp"](sin_dec_pts)
    base = param_splines["base"](sin_dec_pts)
    sin_dec_splines = []
    for ti, tri in zip(src_t, src_trange):
        vals = rate_func.integral(t=ti, trange=tri, pars=(amp, base))
        spl = sci.InterpolatedUnivariateSpline(
            sin_dec_pts, vals, k=1, ext="raise")
        norm = spl.integral(lo, hi)
        sin_dec_splines.append(spl_factory(sin_dec_pts, vals / norm))

    info = {"allsky_rate_func": rate_func_allsky,
            "allsky_best_params": fitres_allsky.x,
            "param_splines": param_splines,
            "best_pars": best_pars,
            "best_stddevs": std_devs}
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
        # Call undocumented base of plt.contour, to avoid creating a figure ...
        cntr = contour.Cntr(x, y, llh)
        paths = cntr.trace(level0=one_sigma_level)
        paths = paths[:len(paths) // 2]  # First half of list has the vertices
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
