from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def test_removed_api_tokens_are_absent():
    root = Path(__file__).resolve().parents[1] / "nmmn"
    removed_tokens = (
        "np.Inf",
        "numpy.Inf",
        "normed=",
        "scipy.random",
        "scipy.stats.f_value",
        "scipy.integrate.trapz",
        "from scipy import fft",
    )

    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text()
        for token in removed_tokens:
            if token in text:
                offenders.append(f"{path.name}: {token}")

    assert offenders == []


def test_peakdetect_import_and_basic_detection():
    from nmmn import peakdetect

    x = np.linspace(0, 4 * np.pi, 200)
    y = np.sin(x)

    max_peaks, min_peaks = peakdetect.peakdetect(y, x, lookahead=20, delta=0.1)

    assert len(max_peaks) >= 1
    assert len(min_peaks) >= 1


def test_stats_random_mode_and_ftest():
    from nmmn import stats

    samples = stats.random(0.25, 0.75, 100)
    assert samples.shape == (100,)
    assert np.all(samples >= 0.25)
    assert np.all(samples < 0.75)

    mode = stats.mode(np.array([0, 0, 0, 1, 2, 3]))
    assert abs(mode) < 0.5

    rss1, rss2, n, p1, p2 = 20.0, 10.0, 30, 2, 4
    fstat, pvalue, conf = stats.ftest(rss1, rss2, n, p1, p2)
    expected = ((rss1 - rss2) / (p2 - p1)) / (rss2 / (n - p2))
    assert fstat == expected
    assert 0 <= pvalue <= 1
    assert np.isfinite(conf)


def test_sed_integration_methods_smoke():
    from nmmn import sed

    lognu = np.linspace(8, 22, 200)
    ll = np.full_like(lognu, 42.0)
    spectrum = sed.SED(lognu=lognu, ll=ll, logfmt=1)

    assert spectrum.bol() > 0
    lumx, gammax = spectrum.xrays()
    assert lumx > 0
    assert np.isfinite(gammax)
    assert spectrum.ion() > 0


def test_joint_plots_accept_current_histogram2d_api():
    from nmmn import bayes, plots

    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(scale=0.2, size=200)

    plots.jointplot(x, y)
    bayes.jointplotx(x, y, binscon=4)

    plt.close("all")
