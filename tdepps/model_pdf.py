# coding: utf-8

import abc


class PDF(object):
    """ Interface for unbinned LLH PDF type classes """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_soverb(self):
        """ Returns the signal over bg ratio for each given data point """
        pass

    @abc.abstractmethod
    def get_args(self):
        """ Returns fixed argus the LLH needs, which are not data or params """
        pass


# #############################################################################
# GRB style PDFs
# #############################################################################
class GRBPDF(PDF):
    """
    Models the PDF part for the GRB LLH function.
    """
    def __init__(self, X, MC, srcs):
        """
        Build PDFs that decsribe BG and/or signal-like events in the LLH.
        The LLH PDFs must not necessarily match the injected data model.
        """
        return

    def get_args(self):
        pass

    def get_soverb(self):
        pass
