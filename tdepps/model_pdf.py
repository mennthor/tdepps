# coding: utf-8


class GRBInjectionModel(object):
    """
    Models the PDF part for the LLH function. Implements: `get_soverb()` and
    `get_args()`.
    """
    def __init__(self, X, MC, srcs):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        return
