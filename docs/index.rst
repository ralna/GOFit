.. GOFit documentation master file, created by
   sphinx-quickstart on Wed Apr 13 10:00:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GOFit
=====
Global Optimization for Fitting problems
----------------------------------------

**Release:** |release|

**Date:** |today|

**Author:** `Jaroslav Fowkes <jaroslav.fowkes@stfc.ac.uk>`_

GOFit is a package of C++ algorithms with python interfaces designed for the global optimization of parameters in curve fitting, i.e. for nonlinear least-squares problems arising from curve fitting. GOFit was developed with scientific curve fitting problems in mind [1]_ but is also applicable to general curve fitting problems provided they can be formulated as nonlinear least-squares problems.

GOFit is released under the New BSD License.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   install
   example
   algorithms

* :ref:`genindex`

.. * :ref:`modindex`
   * :ref:`search`

References
----------

.. [1]
   O'Flynn, M., Fowkes, J. and Gould, N. (2022) Global optimization of crystal field parameter fitting in Mantid, *RAL Technical Reports*, RAL-TR-2022-002. https://doi.org/10.5286/raltr.2022002
