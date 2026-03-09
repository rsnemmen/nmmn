# nmmn Improvement Plan

## Priority 1 — CRITICAL: Broken on Modern NumPy/SciPy

- [x] `scipy.stats.randint.rvs()` removed in scipy 1.14 → `nmmn/lsd.py:277,281` → Use `numpy.random.randint()`
- [x] `numpy.float` removed in NumPy 1.20 → `nmmn/lsd.py:397,402` → Use `float()`
- [x] `scipy.stats.scoreatpercentile` removed in scipy 0.18 → `nmmn/bayes.py:151,162` → Use `numpy.percentile()`
- [x] `zip()` returns iterator in Python 3; unpacking fails → `nmmn/dsp.py:29,31` → Wrapped with `tuple()`
- [~] `numpy.fromstring()` removed in NumPy 1.20 → `nmmn/grmhd.py:137–148` → **Not applicable**: usage with explicit `sep='\n'` is still valid in modern NumPy

## Priority 2 — HIGH: Packaging & Dependencies

- [x] Delete stale `requirements.txt` (2018, conflicts with pyproject.toml)
- [x] Add missing optional dep groups to `pyproject.toml`: `ml` (scikit-learn, transformer_lens), `bayes` (emcee, PyMC)

## Priority 3 — MEDIUM: Correctness & Safety

- [ ] `eval()` on user-supplied string → `nmmn/dsp.py:83` → Replace with dict mapping window names → numpy functions
- [ ] Off-by-one in `lsd.crop()` → `nmmn/lsd.py:335` → Change `j[-1]` → `j[-1]+1`, same for `i`
- [ ] File open without context manager → `nmmn/misc.py:391–393` → Use `with open(...)`

## Priority 4 — MEDIUM: Missing Infrastructure

- [ ] No test suite — add `tests/` with pytest smoke tests per module
- [ ] No CI/CD — add `.github/workflows/` for test + PyPI publish
- [ ] `docs/nmmn.rst` missing `nmmn.ml` module entry
- [ ] `docs/conf.py` uses deprecated Read the Docs mocking (lines 23–32)

## Priority 5 — LOW: Housekeeping

- [ ] `nmmn/__init__.py` docstring doesn't mention `ml` module
- [ ] README copyright year is 2020
- [ ] `.ipynb_checkpoints/` not in `.gitignore`
- [ ] 40+ PEP 8 violations (`if x == True`, `if x != None`) in `plots.py`, `bayes.py`, `astro.py`
