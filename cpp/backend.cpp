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


namespace py = pybind11;

// Actual C++ snippets. See docstrings at the bottom for argument info.
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

    // Precompute src dec and

}


template <typename T>
inline T gaus_pdf_nonorm(T x, T mean, T sigma) {
    // Returns value of the 1D Gaussian PDF w/o normalization term at x.
    T x_ = x - mean;
    return std::exp(-0.5 * x_ * x_ / (sigma * sigma));
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
    T dcdf = std::erf(nsig / sqrt2);
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

    // Define the actual template types
    m.def("soverb_time", &soverb_time<double>, soverb_time_docstr,
          py::arg("t"), py::arg("src_t"), py::arg("dt0"), py::arg("dt1"),
          py::arg("sig_t"), py::arg("sig_t_clip"), py::arg("nsig"));
    m.def("soverb_time", &soverb_time<float>, "",
          py::arg("t"), py::arg("src_t"), py::arg("dt0"), py::arg("dt1"),
          py::arg("sig_t"), py::arg("sig_t_clip"), py::arg("nsig"));

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
    return m.ptr();
}
