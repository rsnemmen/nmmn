`nmmn` package
================

Tools for astronomy, data analysis, time series, numerical simulations, gamma-ray astronomy and more! These are modules I wrote which I find useful—for whatever reason—in my research.

List of modules available ([more info here](http://rsnemmen.github.io/nmmn/)):

* `astro`: astronomy
* `dsp`: signal processing
* `lsd`: misc. operations on arrays, lists, dictionaries and sets
* `stats`: statistical methods
* [`sed`: spectral energy distributions](./docs/SEDs.ipynb)
* `plots`: custom plots
* `fermi`: Fermi LAT analysis methods
* `bayes`: Bayesian tools for dealing with posterior distributions
* `grmhd`: tools for dealing with GRMHD numerical simulations

Very basic [documentation](http://rsnemmen.github.io/nmmn/) for the package. Generated with Sphinx.

# Installation

You have a couple of options to install the module:

### 1. Install using `pip`:

```
pip install nmmn
```


### 2. Install the module on the system’s python library path: 

```
git clone https://github.com/rsnemmen/nmmn.git
cd nmmn
python setup.py install
```

### 3. Install the package with a symlink, so that changes to the source files will be immediately available:

```
git clone https://github.com/rsnemmen/nmmn.git
cd nmmn
python setup.py develop
```

This last method is preferred if you want the latest, bleeding-edge updates in the repo. You may need to run the last command with `sudo`.

## Updating

If you installed with `pip` (method 1), to upgrade the package to the latest stable version use

    pip install --upgrade nmmn

If you installed with the `setup.py` script and the `develop` option (method 3), use

    cd /path/to/nmmn
    git pull

# Usage

First import the specific module that you want to use:

    import nmmn.lsd

Then call the method you need. For example, to remove all `nan` and `inf` elements from a `numpy` array:

```python
import numpy as np

# generates some array with nan and inf
x=np.array([1,2,np.nan,np.inf])

# removes strange elements
xok=nmmn.lsd.delweird(x)
```

For more examples, please refer to the [examples doc](examples.md).

# TODO

* [x] need more examples of how to use the modules
* [x] add IFU data cubes method (refer to [ifscube](https://ifscube.readthedocs.io/en/latest/))

# License

See `LICENSE` file.

If you have suggestions of improvements, by all means please contribute with a pull request!  :)

The MIT License (MIT). Copyright (c) 2020 [Rodrigo Nemmen](http://rodrigonemmen.com)

[Visit the author's web page](https://rodrigonemmen.com/) and/or follow him on twitter ([@nemmen](https://twitter.com/nemmen)).