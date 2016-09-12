Miscellaneous tools: nmmn
===================

Miscellaneous modules for:

* `astro`: astronomy
* `dsp`: signal processing
* `lsd`: misc. operations on arrays, lists, dictionaries and sets
* `stats`: statistical methods
* `plots`: custom plots
* `fermi`: Fermi LAT analysis methods
* `bayes`: Bayesian tools for dealing with posterior distributions
* `grmhd`: tools for dealing with GRMHD numerical simulations

These are modules I wrote which I find useful -- for whatever reason -- in my research.

# Installation

The command line script can be installed via

    python setup.py install

You may need to run the last command with `sudo`.

Install the package with a symlink, so that changes to the source files will be immediately available:

    python setup.py develop

# Documentation

~~[Documentation for the module](http://nmmn.readthedocs.io/en/latest/#) is kept updated on Read the Docs.~~

I am trying to render the documentation nicely with [Read the Docs](http://nmmn.readthedocs.io/en/latest/#) but so far did not manage to get it working properly. In the meantime, you will have to browse the docstrings of the modules and methods for the documentation.

# TODO

* [x] repackage into public and private modules
* [x] install instructions
* [x] license
* [x] requirements
* [x] ~~possibly broken intra-package references? let's see~~
* [x] documentation with Sphinx
* [ ] fix documentation with Read the Docs

# License

See LICENSE file.

If you have suggestions of improvements, by all means please contribute with a pull request!  :)

The MIT License (MIT). Copyright (c) 2016 [Rodrigo Nemmen](http://rodrigonemmen.com)

[Visit the author's web page](http://rodrigonemmen.com/) and/or follow him on twitter ([@nemmen](https://twitter.com/nemmen)).