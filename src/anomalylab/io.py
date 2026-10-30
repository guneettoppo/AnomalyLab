"""Corpus and label loading for the NAB dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LABELS = ROOT / "labels" / "combined_windows.json"

CATEGORIES = [
    "artificialNoAnomaly",
    "artificialWithAnomaly",
    "realAWSCloudwatch",
    "realAdExchange",
    "realKnownCause",
    "realTraffic",
    "realTweets",
]


@dataclass
class Series:
    """One NAB data stream plus its ground-truth windows."""

    key: str                 # "realKnownCause/machine_temperature_system_failure.csv"
    ticks: np.ndarray        # int64 nanoseconds, monotonic
    values: np.ndarray       # float64
    windows: list[tuple[int, int]]   # inclusive [start, end] row indices

    @property
    def name(self) -> str:
        return self.key.split("/")[-1].replace(".csv", "")

    @property
    def category(self) -> str:
        return self.key.split("/")[0]

    @property
    def n(self) -> int:
        return len(self.values)

    def label_mask(self) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        for a, b in self.windows:
            m[a:b + 1] = True
        return m

    def probation_len(self) -> int:
        """NAB's unlabeled learning period: 15% of the file, capped at 750 rows."""
        return int(min(np.floor(0.15 * self.n), 0.15 * 5000))


def list_series(root: Path = DATA) -> list[str]:
    return sorted(str(p.relative_to(root)).replace("\\", "/")
                  for p in root.rglob("*.csv"))


def load_labels(path: Path = LABELS) -> dict[str, list[list[str]]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_series(key: str, labels: dict | None = None, root: Path = DATA) -> Series:
    labels = load_labels() if labels is None else labels
    df = pd.read_csv(root / key, parse_dates=["timestamp"]).sort_values("timestamp")
    ticks = df["timestamp"].values.astype("datetime64[ns]").astype("int64")
    values = df["value"].to_numpy(dtype=float)

    index = {t: i for i, t in enumerate(ticks)}
    windows: list[tuple[int, int]] = []
    for start, end in labels.get(key, []):
        s = int(np.datetime64(start, "ns").astype("int64"))
        e = int(np.datetime64(end, "ns").astype("int64"))
        if s in index and e in index:
            windows.append((index[s], index[e]))
    windows.sort()
    return Series(key=key, ticks=ticks, values=values, windows=windows)


def load_corpus(root: Path = DATA) -> list[Series]:
    labels = load_labels()
    return [load_series(k, labels, root) for k in list_series(root)]
