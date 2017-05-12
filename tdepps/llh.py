import numpy as np

import docrep  # Reuse docstrings
docs = docrep.DocstringProcessor()


class LLH(object):
    """
    Interface for Likelihood functions used in PS analyses.

    Classes must implement functions ["lnllh", "lnllh_ratio"].
    """
    def __init__(self):
        self._DESCRIBES = ["llh"]
        print("Interface only. Describes functions: ", self._DESCRIBES)
        return

    @docs.get_summaryf("LLH.lnllh")
    @docs.get_sectionsf("LLH.lnllh", sections=["Parameters", "Returns"])
    @docs.dedent
    def lnllh(self, X, theta):
        r"""
        Return the natural logarithm ln-Likelihood (ln-LLH) value for a given
        set of data `X` and parameters `theta`.

        The Likelihood of a parameter `theta` of a parametric probability model
        under the given data `X` is the product of all probability values of
        the given data under the model assumption:

        .. math::

          P(X|\theta) = \mathcal{L}(\theta|X) = \prod_{i=1}^N f(x_i|\theta)


        The most likely set of parameters `theta` can be found by finding the
        maximum of the LLH function by variation of the parameter set.

        Parameters
        ----------
        X : array-like, shape (n_points, n_dim)
            Fixed data points, each row is a single nD point.
        theta : array-like, shape (n_params)
            Parameter set to evaluate the ln-LLH at.

        Returns
        -------
        lnllh : float
            Natural logarithm of the LLH for a given `X` and `theta`.
        """
        raise NotImplementedError("LLH is an interface.")

    @docs.get_summaryf("LLH.lnllh_ratio")
    @docs.get_sectionsf("LLH.lnllh_ratio", sections=["Parameters", "Returns"])
    @docs.dedent
    def lnllh_ratio(self, X, theta):
        r"""
        Return the natural logarithm ratio of the Likelihood under the null
        hypothesis and the alternative hypothesis.

        The ratio :math:`\Lambda` used here is defined as:

        .. math::

          \Lambda = -2\ln\left(\frac{\mathcal{L}_0}
                                     {\mathcal{L}_1}\right)
                  =  2\ln\left(\mathcal{L}_1 - \mathcal{L}_0\right)


        High values of :math:`\Lambda` indicate, that the null hypothesis is
        more unlikely contrary to the alternative.

        Parameters
        ----------
        X : array-like, shape (n_points, n_dim)
            Fixed data points, each row is a single nD point.
        theta : array-like, shape (n_params)
            Parameter set to evaluate the ln-LLH at.

        Returns
        -------
        lnllh_ratio : float
            Natural logarithm of the LLH for a given `X` and `theta`.
        """
        raise NotImplementedError("LLH is an interface.")


class GRBLLH(LLH):
    r"""
    Implementation of the GRBLLH for time dependent analysis.

    For more information on the extended Likelihood method, see [1]_.
    The used Likelihood is defined as:

    .. math::

      \mathcal{L}(N|n_S,\theta) = \frac{(n_S + \langle n_B \rangle)^{-N}}{N!}
                                   \cdot \exp{(-(n_S + \langle n_B \rangle))}
                                   \cdot\prod_{i=1}^N P_i


    Where :math:`n_S` is the number of signal-like events and :math:`P_i` are
    the events probabilities to be background- or signal-like:

    .. math::

      P_i = \frac{n_S \cdot S_i + \langle n_B \rangle \cdot B_i}
                 {n_S + \langle n_B \rangle}


    Other parameters :math:`\theta` might be used in the per event signal and
    background PDFs :math:`S_i` and :math:`B_i`.

    The ln-LLH is then derived by taking the natural logarithm of the LLH:

    .. math::

      \ln\mathcal{L}(N | n_S, \theta) = -(n_S + \langle n_B\rangle) -\ln(N!) +
                            \sum_{i=1}^N \ln(n_S S_i + \langle n_B\rangle B_i)


    Notes
    -----
    .. [1] Barlow, "Statistics - A Guide to the Use of Statistical Methods in
           the Physical Sciences". Chap. 5.4, p. 90. Wiley (1989)
    """
    def __init__(self):
        # Make it explicit
        return

    @docs.dedent
    def lnllh(self, X, theta):
        """
        %(LLH.lnllh.summary)s

        Parameters
        ----------
        %(LLH.lnllh.parameters)s

        Returns
        -------
        %(LLH.lnllh.returns)s
        """





















