"""Evaluation beyond NAB's single number.

NAB's application score answers "how much is this detector worth to an operator
who tolerates some false alarms" but it hides three things an on-call engineer
cares about:

* how many alarms per day the operator actually has to look at;
* whether each labelled incident was caught at all (NAB's score can be positive
  while a window is missed entirely);
* what the detector looks like at a threshold you could deploy, rather than at
  the threshold chosen with knowledge of the labels.
"""
from __future__ import annotations

import numpy as np

DAY_NS = 86_400 * 10 ** 9


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as inclusive (start, end) index pairs."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1] - 1)) for i in range(0, len(edges), 2)]


def point_metrics(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def window_metrics(pred: np.ndarray, windows: list[tuple[int, int]]) -> dict[str, float]:
    """Were the incidents caught, and how quickly?"""
    if not windows:
        return {"windows_hit": 0.0, "window_recall": float("nan"), "median_delay_pts": float("nan")}
    hits, delays = 0, []
    for a, b in windows:
        seg = np.flatnonzero(pred[a:b + 1])
        if seg.size:
            hits += 1
            delays.append(int(seg[0]))
    return {
        "windows_hit": float(hits),
        "window_recall": hits / len(windows),
        "median_delay_pts": float(np.median(delays)) if delays else float("nan"),
    }


def alarm_load(pred: np.ndarray, label: np.ndarray, ticks: np.ndarray) -> dict[str, float]:
    """Operator-facing load: false alarms per day and alarm episodes total."""
    outside = pred & ~label
    episodes = _runs(outside)
    span_days = max((ticks[-1] - ticks[0]) / DAY_NS, 1e-6)
    return {
        "false_alarms": float(len(episodes)),
        "false_alarms_per_day": len(episodes) / span_days,
        "alarm_episodes": float(len(_runs(pred))),
        "span_days": float(span_days),
    }


def evaluate(pred: np.ndarray, label: np.ndarray, windows: list[tuple[int, int]],
             ticks: np.ndarray) -> dict[str, float]:
    out = {}
    out.update(point_metrics(pred, label))
    out.update(window_metrics(pred, windows))
    out.update(alarm_load(pred, label, ticks))
    return out
