#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>


// Time critical function are coded in C++ here.
// pybind11 allows us to just write snippets in C++ and integrate them easily
// and intuitively in out python package.
// I tried my best to make the C++ code a readable as possible and only use it,
// where we would have to use heavy broadcasting in python otherwise.

static const double pi = 4. * std::atan(1.);
static const double sqrt2 = std::sqrt(2.);
static const double sqrt2pi = std::sqrt(2. * pi);
static const double secinday = 24. * 60. * 60.;


// Actual C++ snippets. See docstrings at the bottom for argument info.
namespace py = pybind11;


template <typename T>
inline T gaus_pdf_nonorm(T x, T mean, T sigma) {
    // Returns value of the 1D Gaussian PDF w/o normalization term at x.
    T x_ = x - mean;
    return std::exp(-0.5 * x_ * x_ / (sigma * sigma));
}


template <typename T>
py::array_t<T> pdf_spatial_signal(const py::array_t<T> src_ra,
                                  const py::array_t<T> src_dec,
                                  const py::array_t<T> ev_ra,
                                  const py::array_t<T> ev_sin_dec,
                                  const py::array_t<T> ev_sig,
                                  const bool kent) {
    auto b_src_ra = src_ra.template unchecked<1>();
    auto b_src_dec = src_dec.template unchecked<1>();
    auto b_ev_ra = ev_ra.template unchecked<1>();
    auto b_ev_sin_dec = ev_sin_dec.template unchecked<1>();
    auto b_ev_sig = ev_sig.template unchecked<1>();
    auto nevts = ev_ra.shape(0);
    auto nsrcs = src_ra.shape(0);

    // Precompute sin(src_dec) and cos(src_dec)
    std::vector<T> sin_src_dec(nsrcs);
    std::vector<T> cos_src_dec(nsrcs);
    for (unsigned j=0; j < nsrcs; ++j){
        sin_src_dec[j] = std::sin(b_src_dec(j));
        cos_src_dec[j] = std::cos(b_src_dec(j));
    }

    // Allocate output array: shape (nsrcs, nevts)
    py::array_t<T> S({nsrcs, nevts});
    auto b_S = S.template mutable_unchecked<2>();

    // Compute cos(great circle dist) for every point to every src and with that
    // the actual PDF value (kent or gaussian) at that distance
    T cos_dist;
    T kappa;
    for (unsigned j = 0; j < nsrcs; ++j) {
        for (unsigned i = 0; i < nevts; ++i) {
            // Derivation: Calc vec(x)*vec(x) in spherical equatorial coords
            cos_dist = std::cos(b_src_ra(j) - b_ev_ra(i)) *
                       cos_src_dec[j] *
                       std::sqrt(1. - b_ev_sin_dec(i) * b_ev_sin_dec(i)) +
                       sin_src_dec[j] * b_ev_sin_dec(i);
            // Handle possible rounding errors
            cos_dist = std::max((T)-1., std::min((T)1., cos_dist));

            if (kent) {
            // Stabilized version for possibly large kappas
            kappa = 1. / (b_ev_sig(i) * b_ev_sig(i));
            b_S(j, i) = kappa / (2. * pi * (1. - std::exp(-2. * kappa))) *
                        std::exp(kappa * (cos_dist - 1.));
            }
            else {
                // TODO: Type correctly.
                // T dist = std::acos(cos_dist);
                // T ev_sig_2 = 2. * pi * b_ev_sig(i) * b_ev_sig(i);
                // S(j, i) = std::exp(-dist * dist / ev_sig_2) / ev_sig_2;
                // 2D gaussian PDF with great circle distance evt <-> src
                // S(j, i) = gaus_pdf_nonorm(std::acos(cos_dist),
                //                           0., b_ev_sig(i)) /
                //           (2. * pi * b_ev_sig(i) * b_ev_sig(i));
            }
        }
    }

    return S;
}


template <typename T>
py::array_t<T> soverb_time (const py::array_t<T>& t,
                            const py::array_t<T>& src_t,
                            const py::array_t<T>& dt0,
                            const py::array_t<T>& dt1,
                            const py::array_t<T>& sig_t,
                            const py::array_t<T>& sig_t_clip,
                            const T nsig) {
    // Gather input shape info (b = buffer, the way pybind11 handles data)
    auto b_t = t.template unchecked<1>();
    auto b_src_t = src_t.template unchecked<1>();
    auto b_dt0 = dt0.template unchecked<1>();
    auto b_dt1 = dt1.template unchecked<1>();
    auto b_sig_t = sig_t.template unchecked<1>();
    auto b_sig_t_clip = sig_t_clip.template unchecked<1>();
    auto nevts = t.shape(0);
    auto nsrcs = src_t.shape(0);

    // Allocate output array: shape (nsrcs, nevts)
    py::array_t<T> soverb({nsrcs, nevts});
    auto b_soverb = soverb.template mutable_unchecked<2>();

    // Precompute gaussian norm for each src
    std::vector<T> gaus_norm(nsrcs);
    for (unsigned j = 0; j < nsrcs; ++j) {
        gaus_norm[j] = 1. / (sqrt2pi * b_sig_t(j));
    }

    // Precompute signal PDF total norm: gaussian + uniform part
    std::vector<T> sig_pdf_norm(nsrcs);
    // CDF(sigma) - CDF(-sigma). https://en.wikipedia.org/wiki/Error_function
    const T dcdf = std::erf(nsig / sqrt2);
    for (unsigned j = 0; j < nsrcs; ++j) {
        sig_pdf_norm[j] = 1. / (dcdf + (b_dt1(j) - b_dt0(j)) * gaus_norm[j]);
    }

    // Precompute bg PDF: Uniform in total src time window length
    std::vector<T> bg_pdf(nsrcs);
    for (unsigned j = 0; j < nsrcs; ++j) {
        bg_pdf[j] = 1. / (b_dt1(j) - b_dt0(j) + 2. * b_sig_t_clip(j));
    }

    // Loop all events, and all sources per event and cal the time soverb value.
    T sig_pdf;
    T t_rel;
    for (unsigned i = 0; i < nevts; ++i) {
        for (unsigned j = 0; j < nsrcs; ++j) {
            // Normaliez event time relative to src time in seconds
            t_rel = b_t[i] * secinday - b_src_t[j] * secinday;
            // 4 cases: outside, rising edge, uniform, falling edge. Sorted
            // after most probable occurrence to skip as much as possible
            if ((t_rel < b_dt0(j) - b_sig_t_clip(j)) ||
                (t_rel > b_dt1(j) + b_sig_t_clip(j))){  // outside
                b_soverb(j, i) = 0.;
                continue;
            }
            else if ((t_rel >= b_dt0(j)) && (t_rel <= b_dt1(j))) {  // uniform
                sig_pdf = gaus_norm[j];
            }
            else if ((t_rel < b_dt0(j)) &&  // rising edge at dt0
                     (t_rel >= b_dt0(j) - b_sig_t_clip(j))){
                sig_pdf = gaus_norm[j] *
                          gaus_pdf_nonorm(t_rel, b_dt0(j), b_sig_t(j));
            }
            else {  // falling edge at dt1
                sig_pdf = gaus_norm[j] *
                          gaus_pdf_nonorm(t_rel, b_dt1(j), b_sig_t(j));
            }
            b_soverb(j, i) = sig_pdf_norm[j] * sig_pdf / bg_pdf[j];
        }
    }

    return soverb;
}


PYBIND11_PLUGIN(backend) {
    py::module m("backend", R"pbdoc(
        Pybind11 C++ backend for tdepps
        -------------------------------

        .. currentmodule:: tdepps_backend

        .. autosummary::
           :toctree: _generate

           soverb_time
           pdf_spatial_signal
    )pbdoc");

    auto soverb_time_docstr = R"pbdoc(
                    Time signal over background ratio.

                    Signal and background PDFs are each normalized over seconds.
                    Signal PDF has gaussian edges to smoothly let it fall of to
                    zero, the standard deviation is controlled by
                    `sigma_t_[min|max]` in `time_pdf_args`.

                    To ensure finite support, the edges of the gaussian are
                    truncated after `time_pdf_args['nsig'] * dt`.

                    Parameters
                    ----------
                    t : array-like, shape (nevts)
                        Times given in MJD for which we want to evaluate the
                        ratio.
                    src_t : array-like, shape (nsrcs)
                        Times of each source event in MJD days.
                    dt0 : array-like, shape (nsrcs)
                        Time windows start time in seconds relative to each
                        `src_t`. Marks the start point of the interval per src
                        in which the signal PDF is assumed to be uniform.
                    dt1 : array-like, shape (nsrcs)
                        Time windows end time in seconds relative to each
                        `src_t`. Marks the end point of the interval per src
                        in which the signal PDF is assumed to be uniform.
                    sig_t : array-like, shape (nsrcs)
                        sigma of the gaussian edges of the time signal PDF.
                    sig_t_clip : array-like, shape (nsrcs)
                        Total length `time_pdf_args['nsig'] * sig_t` of the
                        gaussian edges of each time window.
                    nsig : int
                        At how many sigmas the gaussian egdes are clipped.

                    Returns
                    -------
                    soverb_time_ratio : array-like, shape (nsrcs, nevts)
                        Ratio of the time signal and background PDF for each
                        given time `t` and per source time `src_t`.
                  )pbdoc";

    auto pdf_spatial_signal_docstr = R"pbdoc(
                    Spatial distance PDF between source position(s) and event
                    positions.

                    Signal is assumed to cluster around source position(s).
                    The PDF is a convolution of a delta function for the
                    localized sources and a Kent (or gaussian) distribution with
                    the events positional
                    reconstruction error as width.

                    If `spatial_pdf_args["kent"]` is True a Kent distribtuion is
                    used, where kappa is chosen, so that the same amount of
                    probability as in the 2D gaussian is inside a circle with
                    radius `ev_sig` per event.

                    Multiple source positions can be given, to use it in a
                    stacked search.

                    Parameters
                    -----------
                    src_ra, src_dec : array-like, shape (nsrcs)
                        Source positions in equatorial right-ascension, [0, 2pi]
                        and declination, [-pi/2, pi/2], given in radian.
                    ev_ra, ev_sin_dec : array-like, shape (nevts)
                        Event positions in equatorial right-ascension, [0, 2pi]
                        in radian and sinus declination, [-1, 1].
                    ev_sig : array-like, shape (nevts)
                        Event positional reconstruction errors in radian
                        (eg. Paraboloid).
                    kent : bool
                        If True, the signal PDF uses the Kent distribution,
                        otherwise 2D gaussian PDF is used.

                    Returns
                    --------
                    S : array-like, shape (nsrcs, nevts)
                        Spatial signal probability for each event and each
                        source position.
                  )pbdoc";

    // Define the actual template types
    m.def("soverb_time", &soverb_time<double>, soverb_time_docstr,
          py::arg("t"), py::arg("src_t"), py::arg("dt0"), py::arg("dt1"),
          py::arg("sig_t"), py::arg("sig_t_clip"), py::arg("nsig"));
    m.def("soverb_time", &soverb_time<float>, "",
          py::arg("t"), py::arg("src_t"), py::arg("dt0"), py::arg("dt1"),
          py::arg("sig_t"), py::arg("sig_t_clip"), py::arg("nsig"));

    m.def("pdf_spatial_signal", &pdf_spatial_signal<double>,
          pdf_spatial_signal_docstr,
          py::arg("src_ra"), py::arg("src_dec"), py::arg("ev_ra"),
          py::arg("ev_sin_dec"), py::arg("ev_sig"),  py::arg("kent"));
    m.def("pdf_spatial_signal", &pdf_spatial_signal<float>, "",
          py::arg("src_ra"), py::arg("src_dec"), py::arg("ev_ra"),
          py::arg("ev_sin_dec"), py::arg("ev_sig"),  py::arg("kent"));

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
    return m.ptr();
}
