"""Causal, drift-invariant feature matrix, calibration and threshold helpers.

Two design rules, both enforced by the test suite:

1. **Causality.**  Every column at row *i* uses only rows ``<= i``.  A centred
   rolling window would hand the detector the future and inflate every score.

2. **Drift invariance.**  Every column is a ratio of a short-window statistic to
   a long-window statistic (or a rolling z-score), never an absolute level.
   An absolute ``level`` column standardised on the probation prefix stays
   extreme forever after a persistent step change, which turns every row after
   the step into a false alarm — the standard way a model fitted on a prefix
   dies on a streaming signal.

Calibration (`pit_scores`) is a *kernel* CDF rather than a hard empirical CDF;
see its docstring for the failure that fixes.  Thresholds (`tail_threshold`) are
raw quantiles, not calibrated ones, because a probability-scale score saturates
near 1 and stops being usable as a cut-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

WINDOWS = (5, 15, 60)


def build_features(values: np.ndarray, windows: tuple[int, ...] = WINDOWS
                   ) -> tuple[np.ndarray, list[str]]:
    s = pd.Series(np.asarray(values, dtype=float))
    short, mid, long = windows

    def roll(w, how):
        return getattr(s.rolling(w, min_periods=2), how)()

    mean_s, mean_m, mean_l = roll(short, "mean"), roll(mid, "mean"), roll(long, "mean")
    std_s, std_m, std_l = roll(short, "std"), roll(mid, "std"), roll(long, "std")
    std_s_safe = std_s.replace(0.0, np.nan)
    std_m_safe = std_m.replace(0.0, np.nan)
    std_l_safe = std_l.replace(0.0, np.nan)

    cols: dict[str, pd.Series] = {
        f"z_{short}": (s - mean_s) / std_s_safe,
        f"z_{mid}": (s - mean_m) / std_m_safe,
        f"z_{long}": (s - mean_l) / std_l_safe,
        "diff_n": s.diff() / std_m_safe,
        "ewm_resid_n": (s - s.ewm(span=mid, adjust=False).mean()) / std_m_safe,
        "dev_up": (s - roll(long, "max")) / std_l_safe,
        "dev_dn": (s - roll(long, "min")) / std_l_safe,
        "mean_gap": (mean_s - mean_l) / std_l_safe,
        "std_ratio": np.log((std_s / std_l_safe).clip(lower=1e-6)),
    }

    names = list(cols)
    X = np.column_stack([cols[c].to_numpy(dtype=float) for c in names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def standardize(X: np.ndarray, prob_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score using statistics of the probation (unlabelled) prefix only."""
    ref = X[:prob_len] if prob_len > 1 else X
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _calibration_map(raw: np.ndarray, prob_len: int, grid: int = 6000):
    """Parzen (Gaussian-kernel) CDF of the probation-period scores, on a grid.

    Returns ``(xs, cdf)`` — a monotone map from raw score to probability — or
    ``None`` when the reference period is degenerate.

    The grid is built from *quantiles of the observed scores*, not from a uniform
    linspace.  A uniform grid over a series with extreme outliers (a PCA
    reconstruction error spanning 1e-3 to 1e6) puts almost no resolution where
    the probation scores actually live, and the interpolated tail then reads
    ~0 for a 99.9th-percentile threshold — so the threshold fires on a large
    fraction of the stream instead of 0.1% of it.  Quantile knots put resolution
    where the data is dense and still carry the extreme values explicitly.
    """
    raw = np.asarray(raw, dtype=float)
    ref = np.sort(raw[:prob_len]) if prob_len > 1 else np.sort(raw)
    m = ref.size
    if m < 2 or ref[-1] - ref[0] < 1e-12:
        return None

    sigma = float(ref.std())
    bandwidth = max(1.06 * sigma * m ** (-1 / 5), 1e-9)

    knots = np.unique(np.quantile(raw, np.linspace(0.0, 1.0, grid)))
    if knots.size < 2:
        return None
    xs = np.concatenate(([knots[0] - 4 * bandwidth], knots,
                         [knots[-1] + 4 * bandwidth]))
    cdf = ndtr((xs[:, None] - ref[None, :]) / bandwidth).mean(axis=1)
    return xs, np.maximum.accumulate(np.clip(cdf, 1e-12, 1.0 - 1e-12))


def pit_scores(raw: np.ndarray, prob_len: int, grid: int = 3000) -> np.ndarray:
    """Probability integral transform against the probation period.

    Maps any detector's arbitrary raw scale to [0, 1] using *only* unlabelled
    data: the output is the probability that a typical probation-period row
    scores below this one.  Used for ranking and for NAB-style scoring, so every
    detector's score lives on one comparable scale.

    Why a kernel CDF and not the empirical one: a hard ECDF quantises every
    score into 1/750 steps at NAB's probation length, so the handful of
    genuinely extreme rows all collapse onto exactly 1.0 and NAB's single
    corpus-wide threshold can no longer rank them against each other.

    A constant score over the reference period (flatlined sensor, detector that
    reports a constant) carries no information about what is abnormal, so it maps
    to 0 rather than to an arbitrary rank — mapping it to ranks would turn a dead
    sensor into a perfect ramp of "anomalies".
    """
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        return raw
    fitted = _calibration_map(raw, prob_len, grid)
    if fitted is None:
        return np.zeros_like(raw)
    xs, cdf = fitted
    return np.interp(raw, xs, cdf)


def calibrated_value(raw: np.ndarray, prob_len: int, value: float,
                     grid: int = 3000) -> float:
    """Where a raw threshold sits on the calibrated scale (monotone, so exact)."""
    fitted = _calibration_map(np.asarray(raw, dtype=float), prob_len, grid)
    if fitted is None:
        return 0.0
    xs, cdf = fitted
    return float(np.interp(value, xs, cdf))


def tail_threshold(raw: np.ndarray, prob_len: int, quantile: float) -> float:
    """Per-stream deployable threshold: the ``quantile`` of the raw scores the
    detector produced during its unlabelled learning period.

    An exact quantile, so "fire above it" means "fire on the top
    (1 - quantile) fraction of what this stream looked like before we started
    watching" — a rule an operator can set without any labels.
    """
    ref = raw[:prob_len] if prob_len > 1 else raw
    return float(np.quantile(ref, quantile))
