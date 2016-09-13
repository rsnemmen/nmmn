nmmn
======

Tools for astronomy, data analysis, time series, numerical simulations, gamma-ray astronomy and more! These are modules I wrote which I find useful -- for whatever reason -- in my research.

List of submodules available ([more info here](http://rsnemmen.github.io/nmmn/)):

* `astro`: astronomy
* `dsp`: signal processing
* `lsd`: misc. operations on arrays, lists, dictionaries and sets
* `stats`: statistical methods
* `plots`: custom plots
* `fermi`: Fermi LAT analysis methods
* `bayes`: Bayesian tools for dealing with posterior distributions
* `grmhd`: tools for dealing with GRMHD numerical simulations

Very basic [documentation](http://rsnemmen.github.io/nmmn/) for the modules.

# Usage

Example 1: Remove all `nan` and `inf` elements from a `numpy` array.

```python
import nmmn.lsd, numpy
x=numpy.array([1,2,numpy.nan,numpy.inf])
xok=nmmn.lsd.delweird(x)
```

More examples coming...

# Installation

The command line script can be installed via

    python setup.py install

You may need to run the last command with `sudo`.

Install the package with a symlink, so that changes to the source files will be immediately available:

    python setup.py develop


# TODO

* [x] repackage into public and private modules
* [x] install instructions
* [x] license
* [x] requirements
* [x] ~~possibly broken intra-package references? let's see~~
* [x] documentation with Sphinx
* [x] ~~fix documentation with Read the Docs~~
* [ ] need more examples of how to use the modules

# License

See LICENSE file.

If you have suggestions of improvements, by all means please contribute with a pull request!  :)

The MIT License (MIT). Copyright (c) 2016 [Rodrigo Nemmen](http://rodrigonemmen.com)

[Visit the author's web page](http://rodrigonemmen.com/) and/or follow him on twitter ([@nemmen](https://twitter.com/nemmen)).