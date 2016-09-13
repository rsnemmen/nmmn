.. nmmn documentation master file, created by
   sphinx-quickstart on Sun Sep 11 23:53:27 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`nmmn <https://github.com/rsnemmen/nmmn>`_
==================================================

Tools for astronomy, data analysis, time series, numerical simulations and more! These are modules I wrote which I find useful -- for whatever reason -- in my research.

List of submodules available:

* `astro`: astronomy 
* `dsp`: signal processing
* `lsd`: misc. operations on arrays, lists, dictionaries and sets
* `stats`: statistical methods
* `plots`: custom plots
* `fermi`: Fermi LAT analysis methods
* `bayes`: Bayesian tools for dealing with posterior distributions
* `grmhd`: tools for dealing with GRMHD numerical simulations

`Code available on Github <https://github.com/rsnemmen/nmmn>`_.

Usage
-------

Example 1: Remove all `nan` and `inf` (:math:`\infty`) elements from a numpy array.

>>> import nmmn.lsd, numpy
>>> x=numpy.array([1,2,numpy.nan,numpy.inf])
>>> xok=nmmn.lsd.delweird(x)


Contents
---------

.. toctree::
   :maxdepth: 4

   nmmn


Todo
-----

- [ ] need more examples


Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

