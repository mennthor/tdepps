Description
===========

Motivation
----------

What do we want to do and how can we do it?


Likelihood Formalism
--------------------

Describing the Likelihood and event PDFs.

We are given a source event occurrence (can be GRB, GW, HESE or anything
else) at a given position in space and time.
We want to search for a
significant contribution of other events, within a predefined region in
time and space around the source events.
For this we need to derive the
expected signal and background contributions in that frame.

The Likelihood that describes this scenario can be derived from counting
statistics.
If we expect :math:`n_S` signal and :math:`n_B` background
events in the given frame, then the probability of observing :math:`N`
events is given by a Poisson PDF:

.. math::


       P_\text{Poisson}(N\ |\ n_S + n_B) = \mathcal{L}(N | n_S, n_b) = \frac{(n_S + n_B)^{-N}}{N!}\cdot \exp{-(n_S + n_B)}

We want to fit for the number of signal events :math:`n_S` in the frame.
But each event doesn't have the same probability of contributing to
either signal or background, because we don't have that information on a
per event basis. So we include prior information on a per event basis to
account for that.

.. math::


       \mathcal{L}(N | n_S, n_B) = \frac{(n_S + n_B)^{-N}}{N!}\cdot \exp{-(n_S + n_B)} \cdot \prod_{i=1}^N P_i

Also the simple Poisson PDF above only has one parameter, the total
number of events, which can be fit for. So we need to resolve this
degeneracy in :math:`n_S`, :math:`n_B` by giving additional information.
For that we include a weighted combination of the probability for an
event to be signal, denoted by the PDF :math:`S_i` and for it to
background, denoted by :math:`B_i`. Because the simple counting
probabilities are :math:`n_S / (n_S + n_B)` to count a signal event and
likewise :math:`n_B / (n_S + n_B)` to count a background event we
construct the per event prior :math:`P_i` as:

.. math::


       P_i = \frac{n_S}{n_S + n_B}\cdot S_i + \frac{n_B}{n_S + n_B}\cdot B_i
           = \frac{n_S \cdot S_i + n_B \cdot B_i}{n_S + n_B}

Note, that for equal probabilities :math:`S_i` and :math:`B_i`, we
simply and up with the normal Poisson counting statistic.

Plugging that back into the likelihood we get:

.. math::


       \mathcal{L}(N | n_S, n_B) = \frac{(n_S + n_B)^{-N}}{N!}\cdot \exp{(-(n_S + n_B))} \cdot \prod_{i=1}^N \frac{n_S \cdot S_i + n_B \cdot B_i}{n_S + n_B}

Taking the natural logarithm to get the log-likelihood we arrive at:

.. math::


       \ln\mathcal{L}(N | n_S, n_B) = -(n_S + n_B) -\ln(N!) + \sum_{i=1}^N \ln((n_S + n_B) P_i)

If we weight up :math:`n_S` then every events signal PDF is contributing
a bit more than the background PDF. So the fitter tries to find the
combination of :math:`n_S` and :math:`n_B` that maximizes the
likelihood.

To further simplify, we can use a measured and fixed background
expectation rate :math:`\langle n_B\rangle` and fit only for the number
of signal events. Then we only fit for the number of signal events
:math:`n_S`.
Also we are only interested in the number of signal events.
The fixed background rate can be extracted from data by using the pdf of a larger timescale and average over that (or fit a function) to ensure that local fluctuations don't matter.

Then we end up with our full Likelihood (the denominator in :math:`P_i`
cancels with the term from the Poisson PDF):

.. math::


       \ln\mathcal{L}(N | n_S) = -(n_S + \langle n_B\rangle) -\ln(N!) + \sum_{i=1}^N \ln(n_S S_i + \langle n_B\rangle B_i)

For the test statistic :math:`\Lambda = -2T` we want to test the hypothesis of having no signal :math:`n_S=0` vs. the alternative with a free parameter
:math:`n_S`:

.. math::


       T &= \ln\frac{\mathcal{L}_0}{\mathcal{L}_1}
          = \ln\frac{\mathcal{L}(n_S=0)}{\mathcal{L}{\hat{n}_S}} \\
         &= -(\hat{n}_S + \langle n_B\rangle) -\ln(N!) +
              \sum_{i=1}^N \ln(\hat{n}_S S_i + \langle n_B\rangle B_i) \\
         &\phantom{=} +\langle n_B\rangle +\ln(N!) -
               \sum_{i=1}^N \ln(\langle n_B\rangle B_i) \\
         &= -\hat{n}_S + \sum_{i=1}^N
             \ln\left( \frac{\hat{n}_S S_i}{\langle n_B\rangle B_i} + 1 \right)

The per event PDFs :math:`S_i` and :math:`B_i` can depend on arbitrary
parameters. The common choise here is to use a time, energy proxy and
spatial proxy depency which has most seperation power:

.. math::


       S_i(x_i, t_i, E_i) = S_T(t_i) \cdot S_S(x_i) \cdot S_E(E_i) \\
       B_i(x_i, t_i, E_i) = B_T(t_i) \cdot B_S(x_i) \cdot B_E(E_i)

Because the Likelihood only contains ratios of the PDF, we only have to
construct functions of the signal to background ratio for each time,
spatial and energy distribution.

For the energy PDFs :math:`S_E, B_E` we use a 2D representation in
reconstructed energy and declination because this has the most
seperation power (see coenders & skylab models). The spatial part
:math:`S_S, B_S` is only depending on the distance from source to event,
not on the absilute position on the sphere. The time part
:math:`S_T, B_T` is equivalent to that, only using the distance in time
between source event and event.


Inference
---------

How we get our best fit parameters from likelihood fitting.


Performance
-----------

Describe sensitivity and limits.


Code Example
------------

Show a representative example how the code links to the descriptions here.

