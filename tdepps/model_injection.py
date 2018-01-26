# coding: utf-8


class GRBInjectionModel(object):
    """
    Models the injection part for the LLH tests. Implements: `get_sample()`.
    """
    def __init__(self, X, MC, srcs):
        """
        Build PDFs for sampling BG and/or signal contributions.
        The sampled PDFs must not necessarily match the tested PDF model.
        """
        return


class MultiGRBModel(object):
    """
    Class holding multiple injector model objects. All must have either BG only
    or BG and signal injectors for each sample.
    """
    def __init__(self):
        self._injectors = {}
        self._ns_weights = None
        self._has_signal_model = None

    @property
    def names(self):
        return list(self._injectors.keys())

    @property
    def injectors(self):
        return self._injectors

    @property
    def has_signal_model(self):
        return self._has_signal_model

    def add_injector(self, name, injector):
        if not isinstance(injector, GRBModel):
            raise ValueError("`injector` object must be of type GRBModel.")

        if name in self.names:
            raise KeyError("Name '{}' has already been added. ".format(name) +
                           "Choose a different name.")

        # Check if new injector is consistent with the already added ones
        if self._has_signal_model is not None:
            if (self._has_signal_model is True and
                    injector.has_signal_model is False):
                raise ValueError("Added injectors have signal models, but " +
                                 "`injector` only has a BG model.")
            if (self._has_signal_model is False and
                    injector.has_signal_model is True):
                raise ValueError("Added injectors have only BG models, but " +
                                 "`injector` has a signal model.")
        else:
            self._has_signal_model = injector.has_signal_model

        self._injectors[name] = injector

    def get_sample(self):
        # TODO: Concat all samples and adapt MultiLLH so it understands the
        # format
        return


# has_signal needed?
# class GRBModel(object):
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
