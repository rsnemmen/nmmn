# Repository Guidelines

## Project Structure & Module Organization

`nmmn` is a Python scientific-computing package. Source modules live in `nmmn/`, with one file per domain: `astro.py`, `dsp.py`, `stats.py`, `sed.py`, `plots.py`, `grmhd.py`, `fermi.py`, `ml.py`, `finance.py`, and related helpers. Package metadata and dependency groups are defined in `pyproject.toml`. Sphinx documentation is under `docs/`, examples are in `examples.md` and `docs/SEDs.ipynb`, and rendered/example images are stored in `figures/`. Generated packaging outputs such as `build/`, `dist/`, and `nmmn.egg-info/` should not be edited by hand.

## Build, Test, and Development Commands

- `pip install -e .`: install the package in editable mode for local development.
- `pip install -e ".[dsp]"`: install an optional dependency group; other groups include `finance`, `grmhd`, `fermi`, `bayes`, and `ml`.
- `python -m build`: build source and wheel distributions into `dist/`.
- `cd docs && make html`: build the Sphinx documentation. Note that `docs/Makefile` writes to a configured output directory.
- `python -m pytest`: run tests if a test suite is added.

## Coding Style & Naming Conventions

Prefer PEP 8 Python style for new and modified code: 4-space indentation, lowercase module names, `snake_case` functions and variables, and concise docstrings. Existing modules often expose small numerical utility functions; keep APIs simple and avoid broad refactors when fixing a focused issue. Preserve public function names unless a compatibility plan is included. Use explicit imports where practical, and keep optional-dependency imports local to functions when they are not part of the core install.

## Testing Guidelines

There is currently no committed automated test suite. For behavior changes, add focused tests under `tests/` using `pytest`, with file names like `test_lsd.py` and test functions named `test_<behavior>`. Cover numerical edge cases such as `nan`, `inf`, empty arrays, unit conversions, and optional dependency absence. When tests are not practical, document the manual check performed, including commands and sample inputs.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit messages such as `fix deprecated APIs and clean up packaging` and `cleaned up TODO.md`. Follow that style: describe the change, keep the subject concise, and mention the affected module when useful. Pull requests should include a brief summary, motivation or linked issue, test/manual-verification notes, and screenshots only for changes affecting figures, plots, or documentation output.

## Agent-Specific Instructions

Do not edit generated directories (`build/`, `dist/`, `nmmn.egg-info/`) unless the task explicitly concerns packaging artifacts. Prefer small, reviewable changes and update docs or examples when public behavior changes.
