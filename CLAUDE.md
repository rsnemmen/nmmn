# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nmmn** is a miscellaneous scientific computing Python library for astronomy, data analysis, signal processing, and numerical simulations by Rodrigo Nemmen. Published on PyPI as `nmmn`.

## Development Setup

```bash
# Development install (symlink, edits take effect immediately)
python setup.py develop

# Regular install
python setup.py install
```

## Building and Publishing

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

## Documentation

```bash
# Build Sphinx docs
cd docs && make html
```

## Module Architecture

All modules live in `nmmn/`:

| Module | Purpose |
|--------|---------|
| `astro.py` | Redshift/distance conversions, flux unit conversions, physical constants (CGS) |
| `lsd.py` | Array/list/set operations: NaN removal (`delnan`), set operations, bootstrapping, normalization |
| `dsp.py` | Digital signal processing: peak detection, smoothing, Lomb-Scargle periodogram, wavelet transforms |
| `stats.py` | Statistics: scatter, confidence/prediction bands, goodness-of-fit, p-values |
| `bayes.py` | Bayesian statistics: posterior visualization, emcee/PyMC integration |
| `sed.py` | Spectral Energy Distributions: `SED` class for multi-wavelength data, bolometric luminosity |
| `plots.py` | Custom plotting: 2D KDE plots, multi-histograms, Mathematica/Parula colormaps |
| `grmhd.py` | GRMHD simulation I/O: RAISHIN, Pluto, HARM VTK file readers |
| `fermi.py` | Fermi LAT gamma-ray analysis: Enrico output, count/model/residual maps |
| `misc.py` | Coordinate transforms: polar↔cartesian, spherical↔cartesian |
| `ml.py` | ML utilities: multi-class ROC AUC, transformer tokenization debugging |
| `finance.py` | Financial tools: candlestick plots, returns calculation (yfinance) |

## Key Dependencies

Core: `numpy`, `scipy`, `matplotlib`, `astropy`, `uncertainties`
Optional (per module): `aplpy`, `cosmolopy`, `emcee`, `PyMC`, `peakutils`, `wavelets`, `yfinance`, `scikit-learn`, `plotly`, `transformer_lens`

## Conventions

- Functions include docstrings with parameter descriptions and often usage examples
- There is no automated test suite — test changes manually or via the example notebooks in `docs/`
- The `sed.py` module is the most complex (~56 KB); the `SED` class is the primary interface for multi-wavelength astronomical data
- `peakdetect.py` is an auxiliary module (third-party algorithm), not part of the main API
