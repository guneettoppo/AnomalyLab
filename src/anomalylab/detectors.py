"""Detectors.  All of them are unsupervised: they are fitted on the probation
prefix (NAB's unlabelled learning period, first 15% capped at 750 rows) and then
score the whole stream.

Families covered
----------------
rolling_zscore   robust univariate deviation (the classic baseline)
rolling_gauss    NAB's windowed-Gaussian tail test, re-implemented
iforest          Isolation Forest over the causal feature matrix
pca_recon        PCA reconstruction error, with per-feature attributions
hbos             histogram-based outlier score, with per-feature attributions
consensus        mean of the four PIT-calibrated scores above
null             constant 0.5 (NAB's null reference, lower bound)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .features import build_features, pit_scores, standardize
from .io import Series

SEED = 42


@dataclass
class DetectorOutput:
    raw: np.ndarray                          # arbitrary scale, higher = more anomalous
    contributions: np.ndarray | None = None  # (n, n_features) signed feature evidence
    feature_names: list[str] = field(default_factory=list)
    notes: str = ""


class Detector:
    name = "base"
    uses_features = True

    def fit_predict(self, series: Series) -> DetectorOutput:  # pragma: no cover - interface
        raise NotImplementedError

    def _matrix(self, series: Series) -> tuple[np.ndarray, list[str], np.ndarray]:
        X, names = build_features(series.values)
        Z, _, _ = standardize(X, series.probation_len())
        return X, names, Z


class RollingZScore(Detector):
    """|z| of the raw value over a short rolling window, smoothed over 3 points."""

    name = "rolling_zscore"
    uses_features = False

    def __init__(self, window: int = 15, smooth: int = 3) -> None:
        self.window = window
        self.smooth = smooth

    def fit_predict(self, series: Series) -> DetectorOutput:
        X, names, _ = self._matrix(series)
        z = np.abs(X[:, names.index(f"z_{self.window}")])
        raw = np.convolve(z, np.ones(self.smooth) / self.smooth, mode="full")[: len(z)]
        return DetectorOutput(raw=raw, notes=f"|z| over {self.window} rows, {self.smooth}-point smoothing")


class RollingGaussian(Detector):
    """NAB's windowed Gaussian: two-sided tail probability of the newest value
    against the previous ``window`` values, scored as -log10(p)."""

    name = "rolling_gauss"

    def __init__(self, window: int = 60, min_periods: int = 15) -> None:
        self.window = window
        self.min_periods = min_periods

    def fit_predict(self, series: Series) -> DetectorOutput:
        y = pd.Series(series.values)
        hist = y.shift(1)                                  # strictly past values
        mu = hist.rolling(self.window, min_periods=self.min_periods).mean()
        sd = hist.rolling(self.window, min_periods=self.min_periods).std()
        sd = sd.where(sd > 1e-9, 1e-9)
        z = (y - mu).abs() / sd
        p = 2.0 * (1.0 - norm.cdf(z.to_numpy(dtype=float)))
        raw = -np.log10(np.clip(p, 1e-12, 1.0))
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return DetectorOutput(raw=raw, notes=f"{self.window}-row Gaussian tail, -log10(p)")


class IsolationForestDetector(Detector):
    name = "iforest"

    def __init__(self, n_estimators: int = 200, max_samples: int = 256) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit_predict(self, series: Series) -> DetectorOutput:
        _, names, Z = self._matrix(series)
        pl = series.probation_len()
        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, max(pl, 16)),
            contamination="auto",
            random_state=SEED,
            n_jobs=1,
        )
        model.fit(Z[:pl])
        raw = -model.score_samples(Z)
        return DetectorOutput(raw=raw, feature_names=names,
                              notes=f"{self.n_estimators} trees on {len(names)} causal features")


class PCAReconstruction(Detector):
    name = "pca_recon"

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components

    def fit_predict(self, series: Series) -> DetectorOutput:
        _, names, Z = self._matrix(series)
        pl = series.probation_len()
        if float(Z[:pl].std()) < 1e-9:
            # dead/flat sensor: there is no normal subspace to reconstruct from
            return DetectorOutput(raw=np.zeros(series.n),
                                  contributions=np.zeros_like(Z), feature_names=names,
                                  notes="degenerate (zero-variance) probation period")
        k = max(1, min(self.n_components, Z.shape[1] - 1, pl - 1))
        model = PCA(n_components=k, random_state=SEED)
        model.fit(Z[:pl])
        recon = model.inverse_transform(model.transform(Z))
        resid = Z - recon
        raw = (resid ** 2).sum(axis=1)
        return DetectorOutput(raw=raw, contributions=resid, feature_names=names,
                              notes=f"PCA k={k} fitted on probation, squared reconstruction error")


class HBOS(Detector):
    name = "hbos"

    def __init__(self, bins: int = 20) -> None:
        self.bins = bins

    def fit_predict(self, series: Series) -> DetectorOutput:
        _, names, Z = self._matrix(series)
        pl = series.probation_len()
        ref = Z[:pl]
        n_feat = Z.shape[1]
        contrib = np.zeros_like(Z)
        for j in range(n_feat):
            col = ref[:, j]
            lo, hi = np.percentile(col, [0.5, 99.5])
            if hi - lo < 1e-9:
                lo, hi = lo - 1.0, hi + 1.0
            hist, edges = np.histogram(col, bins=self.bins, range=(lo, hi))
            dens = (hist + 1.0) / (hist.sum() + self.bins)  # Laplace smoothing
            idx = np.clip(np.digitize(Z[:, j], edges) - 1, 0, self.bins - 1)
            contrib[:, j] = -np.log(dens[idx])
        raw = contrib.mean(axis=1)
        return DetectorOutput(raw=raw, contributions=contrib, feature_names=names,
                              notes=f"{self.bins}-bin histogram per feature, mean -log p")


class Consensus(Detector):
    """Fusion of the PIT-calibrated member scores.

    ``kind="mean"`` requires broad agreement (fewer alarms, loses single-detector
    wins); ``kind="max"`` is an OR-fusion (catches more incidents, pays for it in
    false alarms).  Members are PIT-calibrated first so no detector's raw scale
    dominates the average.
    """

    name = "consensus_mean"

    def __init__(self, members: list[Detector] | None = None, kind: str = "mean") -> None:
        self.kind = kind
        self.name = f"consensus_{kind}"
        self.members = members or [
            RollingZScore(), IsolationForestDetector(), PCAReconstruction(), HBOS(),
        ]

    def fit_predict(self, series: Series) -> DetectorOutput:
        pl = series.probation_len()
        parts, contribs, names = [], None, []
        for det in self.members:
            out = det.fit_predict(series)
            parts.append(pit_scores(out.raw, pl))
            if out.contributions is not None and out.feature_names:
                contribs = out.contributions
                names = out.feature_names
        stack = np.vstack(parts)
        raw = stack.max(axis=0) if self.kind == "max" else stack.mean(axis=0)
        return DetectorOutput(raw=raw, contributions=contribs, feature_names=names,
                              notes=f"{self.kind} of PIT scores of " +
                                    ", ".join(m.name for m in self.members))


class NullDetector(Detector):
    name = "null"
    uses_features = False

    def fit_predict(self, series: Series) -> DetectorOutput:
        return DetectorOutput(raw=np.full(series.n, 0.5), notes="constant 0.5")


def build_detectors() -> dict[str, Detector]:
    dets: list[Detector] = [
        NullDetector(), RollingZScore(), RollingGaussian(),
        IsolationForestDetector(), PCAReconstruction(), HBOS(),
    ]
    dets.append(Consensus([d for d in dets if d.name in
                           ("rolling_zscore", "iforest", "pca_recon", "hbos")], kind="mean"))
    dets.append(Consensus([d for d in dets if d.name in
                           ("rolling_zscore", "iforest", "pca_recon", "hbos")], kind="max"))
    return {d.name: d for d in dets}


def explain(out: DetectorOutput, row: int, k: int = 5) -> list[tuple[str, float]]:
    """Top-k features behind a detector's score at ``row``.

    For PCA/HBOS the contributions are the model's own per-feature terms; for
    detectors that do not expose one (Isolation Forest), nothing is reported
    rather than inventing an attribution.
    """
    if out.contributions is None or not out.feature_names:
        return []
    vals = out.contributions[row]
    order = np.argsort(np.abs(vals))[::-1][:k]
    return [(out.feature_names[i], float(vals[i])) for i in order]
