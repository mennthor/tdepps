# coding: utf-8

from __future__ import print_function, division, absolute_import
from future.utils import viewkeys
from future import standard_library
standard_library.install_aliases()

from model_toolkit import SignalFluenceInjector, ResampleBGDataInjector


class GRBInjectionModel(object):
    """
    Models the injection part for the LLH tests. Implements: `get_sample()`.

    Parameters
    ----------
    X : dict
    MC : dict
    srcs : dict
    """
    def __init__(self):
        self._bg_injectors = None
        self._sig_injectors = None
        self._names
        # Settings?
        return

    @property
    def names(self):
        return self._names

    @property
    def bg_injectors(self):
        return self._bg_injectors

    @property
    def signal_injectors(self):
        return self._sig_injectors

    def fit(self, X, MC, srcs):
        """
        Take data, MC and sources and build injection models. This is the place
        to actually stitch together a custom injector from the toolkit modlues.
        """
        # Keys must all be equivalent
        if viewkeys(X) != viewkeys(MC):
            raise ValueError("Keys in `X` and `MC` don't match.")
        if viewkeys(MC) != viewkeys(srcs):
            raise ValueError("Keys in `MC` and `srcs` don't match.")

        self._names = list(viewkeys(X))

        self._build_data_injectors(X, srcs)
        self._build_signal_injectors(MC, srcs)

        return

    def get_sample(self, n_signal=None):
        """
        Get a complete data sample for one trial.

        Parameters
        ----------
        n_signal : int or None, opional
            How many signal events shall be sampled additional to the BG events.
            IF ``None`` no events are sampled. If signal is sampled, it just
            get's blend into the background sample. (default: ``None``)
        """
        # TODO: Concat BG and signal samples
        # Internally keep track of which event was injected from which injector
        return

    def _build_data_injectors(self, X, srcs):
        """ The specific model for BG injection is encoded here """
        return

    def _build_signal_injectors(self, X, MC, srcs):
        """ The specific model for signal injection is encoded here """
        # Create single injectors
        _sig_injectors = {}
        for name in self._names:
            pass
        self._sig_injectors = _sig_injectors
        return


# class MultiGRBInjectionModel(object):
#     """
#     Class holding multiple injector model objects. All must have either BG only
#     or BG and signal injectors for each sample.
#     This implements backgorund and signal injection for several samples using
#     correct weighting between the samples.
#     """
#     def __init__(self):
#         self._injectors = {}
#         self._ns_weights = None
#         self._has_signal_model = None

#     @property
#     def names(self):
#         return list(self._injectors.keys())

#     @property
#     def injectors(self):
#         return self._injectors

#     @property
#     def has_signal_model(self):
#         return self._has_signal_model

#     def add_injector(self, name, injector):
#         if not isinstance(injector, GRBInjectionModel):
#             raise ValueError("`injector` object must be of type " +
#                              "GRBInjectionModel.")

#         if name in self.names:
#             raise KeyError("Name '{}' has already been added. ".format(name) +
#                            "Choose a different name.")

#         # Check if new injector is consistent with the already added ones
#         if self._has_signal_model is not None:
#             if (self._has_signal_model is True and
#                     injector.has_signal_model is False):
#                 raise ValueError("Added injectors have signal models, but " +
#                                  "`injector` only has a BG model.")
#             if (self._has_signal_model is False and
#                     injector.has_signal_model is True):
#                 raise ValueError("Added injectors have only BG models, but " +
#                                  "`injector` has a signal model.")
#         else:
#             self._has_signal_model = injector.has_signal_model

#         self._injectors[name] = injector

#     def get_sample(self):
#         # TODO: Concat all samples and adapt MultiLLH so it understands the
#         # format
        # return

        # # We need to re-normalize w_theo over all samples instead of all sources in a
        # # single samples for a single injector, because sources are disjunct in each one
        # def get_raw_fluxes(injectors):
        #     # Split original flux over sources per sample.
        #     # List of arrays, fluxes per sample, per source
        #     w_theos = [inj._srcs[-1]["w_theo"] / inj._srcs[-1]["w_theo"].sum() for inj in injectors]
        #     raw_fluxes = np.array([wts / inj.mu2flux(1.) for inj, wts in zip(injectors, w_theos)])
        #     # Renormalize w_theos over all samples
        #     w_theos = [inj._srcs[-1]["w_theo"] for inj in injectors]
        #     w_theo_sum = np.sum(map(np.sum, w_theos))
        #     w_theos = [wt / w_theo_sum for wt in w_theos]
        #     # Renormalize fluxes per sample per source with renormalized w_theo weights
        #     raw_fluxes = np.array([raw_f * wt for raw_f, wt in zip(raw_fluxes, w_theos)])
        #     # Combine to decreased raw flux per sample
        #     raw_fluxes = np.array(map(np.sum, raw_fluxes))
        #     return raw_fluxes, w_theos

        # def get_sample_w(injectors):
        #     # Get the renormalized fluxes and normalize as normal now
        #     raw_fluxes, _ = get_raw_fluxes(injectors)
        #     return raw_fluxes / raw_fluxes.sum()

        # def distribute_samples(n, injectors):
        #     p = get_sample_w(injectors)
        #     return np.random.multinomial(n, p, size=None)

        # def mu2flux(mu, injectors):
        #     raw_fluxes, _ = get_raw_fluxes(injectors)
        #     return mu / np.sum(raw_fluxes)

# has_signal needed? -> get_sample(signal=True)
# class GRBInjectionModel(object):
#     """
#     Wrapper class to combine signal and backgorund models to use as a whole.
#     Single PDFs are replaced by ratios.

#     Implements `get_sample()`, `get_soverb()` and `get_args()`
#     """
#     def __init__(self, bg_model, signal_model=None):
#         if signal_model is None:
#             self._has_signal_model = False
#         else:
#             self._has_signal_model = True

#         self.bg = bg_model
#         self.sig = signal_model

#     @property
#     def has_signal_model(self):
#         return self._has_signal_model


# Old bg_injector multisampler
# class MultiGeneralPurposeInjector(object):
#     """
#     Container class that holds single instances of GeneralPurposeInjectors.
#     """
#     def __init__(self):
#         self._injs = {}
#         return

#     @property
#     def names(self):
#         return list(self._injs.keys())

#     @property
#     def llhs(self):
#         return list(self._injs.values())

#     def add_injector(self, name, inj):
#         """
#         Add a injector object to consider.

#         Parameters
#         ----------
#         name : str
#             Name of the inj object. Should be connected to the dataset used.
#         inj : tdepps.bg_injector.GeneralPurposeInjector
#             BG injector object sampling pseudo BG events.
#         """
#         if not isinstance(inj, GeneralPurposeInjector):
#             raise ValueError("`inj` object must be of type GeneralPurposeInjector.")

#         if inj._n_features is None:
#             raise RuntimeError("Injector must be fitted before adding.")

#         if name in self.names:
#             raise KeyError("Name '{}' has already been added. ".format(name) +
#                            "Choose a different name.")
#         else:
#             self._injs[name] = inj

#         return

#     def sample(self, n_samples=1):
#         """
#         Call each added injector's sample method and wrap the sampled arrays in
#         dictionaries for use in ``MultiSampleGRBLLH``.

#         Parameters
#         ----------
#         n_samples : dict
#             Number of samples to generate per injector. Dictionary keys must
#             match added ``self.names``.

#         Returns
#         -------
#         sam_ev : dictionary
#             Sampled events from each added ``GeneralPurposeInjector``.
#         """
#         if len(self.names) == 0:
#             raise ValueError("No injector has been added yet.")

#         if viewkeys(n_samples) != viewkeys(self._injs):
#             raise ValueError("Given `n_samples` has not the same keys as " +
#                              "stored injectors names.")

#         sam_ev = {}
#         for name in self.names:
#             # Get per sample information
#             inj = self._injs[name]
#             sam_ev[name] = inj.sample(n_samples[name])

#         return sam_ev
