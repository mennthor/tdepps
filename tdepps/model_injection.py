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


class GRBModel(object):
    """
    Wrapper class to combine signal and backgorund models to use as a whole.
    Single PDFs are replaced by ratios.

    Implements `get_sample()`, `get_soverb()` and `get_args()`
    """
    def __init__(self, bg_model, signal_model=None):
        if signal_model is None:
            self._has_signal_model = False
        else:
            self._has_signal_model = True

        self.bg = bg_model
        self.sig = signal_model

    @property
    def has_signal_model(self):
        return self._has_signal_model


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