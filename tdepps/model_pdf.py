# coding: utf-8

import abc


class PDF(object):
    """ Interface for LLH PDF type classes. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_soverb():
        raise NotImplementedError

    @abc.abstractmethod
    def get_args():
        raise NotImplementedError


class GRBPDF(PDF):
    """
    Models the PDF part for the GRB LLH function. Implements: ``get_soverb()``
    and ``get_args()``.
    """
    def __init__(self, X, MC, srcs):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        return
