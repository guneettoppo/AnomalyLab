"""Smoke and correctness tests.  Run:  python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomalylab import nab_scoring as ns                              # noqa: E402
from anomalylab.detectors import (HBOS, IsolationForestDetector,     # noqa: E402
                                  PCAReconstruction, RollingGaussian,
                                  RollingZScore)
from anomalylab.features import build_features, pit_scores           # noqa: E402
from anomalylab.io import load_labels, load_series                   # noqa: E402

KEY = "realKnownCause/machine_temperature_system_failure.csv"


class TestScoringPort(unittest.TestCase):
    """The ported scorer must reproduce NAB's own published per-file score."""

    def test_matches_reference(self):
        det = "numenta"
        ref_path = ROOT / "reference" / f"{det}_machine_temperature_system_failure.csv"
        if not ref_path.exists():
            self.skipTest("reference results not downloaded")
        series = load_series(KEY, load_labels())
        ref = pd.read_csv(ref_path, parse_dates=["timestamp"])
        published = pd.read_csv(ROOT / "reference" / f"{det}_standard_scores.csv")
        published = published[(published.Profile == "standard") &
                              (published.File == KEY)]
        threshold = float(published.Threshold.iloc[0])
        sw = ns.sweep_file(series.ticks, ref.anomaly_score.to_numpy(float), series.windows)
        self.assertAlmostEqual(sw.row(threshold)[0], float(published.Score.iloc[0]), places=9)

    def test_sigmoid_endpoints(self):
        self.assertAlmostEqual(ns.scaled_sigmoid(-1.0), 0.986612, places=5)
        self.assertAlmostEqual(ns.scaled_sigmoid(0.0), 0.0, places=6)
        self.assertAlmostEqual(ns.scaled_sigmoid(3.5), -1.0, places=9)

    def test_probation_cap(self):
        self.assertEqual(ns.probation_length(4032), 604)
        self.assertEqual(ns.probation_length(100000), 750)


class TestFeatures(unittest.TestCase):
    def test_causality(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=400).cumsum()
        X1, _ = build_features(y)
        y2 = y.copy()
        y2[300:] += 100.0                    # mutate the future only
        X2, _ = build_features(y2)
        np.testing.assert_allclose(X1[:300], X2[:300], atol=1e-9)

    def test_pit_is_calibrated_on_probation(self):
        rng = np.random.default_rng(1)
        raw = rng.normal(size=2000)
        pit = pit_scores(raw, 500)
        self.assertGreaterEqual(pit.min(), 0.0)
        self.assertLessEqual(pit.max(), 1.0)
        self.assertAlmostEqual(float(np.mean(pit[:500] < 0.5)), 0.5, delta=0.06)


class TestCalibration(unittest.TestCase):
    def test_deployable_threshold_fires_at_requested_rate(self):
        from anomalylab.features import calibrated_value, tail_threshold
        rng = np.random.default_rng(3)
        raw = np.concatenate([rng.normal(size=5000), rng.pareto(1.5, 200) * 1e4])
        pl = 600
        thr = tail_threshold(raw, pl, 0.999)
        pred = raw > thr
        self.assertLess(pred[:pl].mean(), 0.004)          # ~0.1% of the learning period
        cal = pit_scores(raw, pl)
        cv = calibrated_value(raw, pl, thr)
        self.assertLess(abs((cal > cv).mean() - pred.mean()), 0.02)

    def test_flat_signal_is_not_all_anomalous(self):
        raw = np.full(1000, 0.5)
        self.assertFalse(pit_scores(raw, 600).any())


class TestDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = load_series(KEY, load_labels())

    def test_all_detectors_produce_finite_scores(self):
        for det in (RollingZScore(), RollingGaussian(), IsolationForestDetector(),
                    PCAReconstruction(), HBOS()):
            out = det.fit_predict(self.series)
            self.assertEqual(len(out.raw), self.series.n, det.name)
            self.assertTrue(np.isfinite(out.raw).all(), det.name)
            score = pit_scores(out.raw, self.series.probation_len())
            self.assertGreaterEqual(score.min(), 0.0)
            self.assertLessEqual(score.max(), 1.0)

    def test_pca_exposes_contributions(self):
        out = PCAReconstruction().fit_predict(self.series)
        self.assertIsNotNone(out.contributions)
        self.assertEqual(out.contributions.shape, (self.series.n, len(out.feature_names)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
