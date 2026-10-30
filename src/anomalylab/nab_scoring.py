"""Faithful port of NAB's application-based scoring profile.

Source of truth: ``nab/sweeper.py`` and ``config/profiles.json`` in
github.com/numenta/NAB (MIT licensed, (c) Numenta Inc.).  The scoring idea is:

* every labelled anomaly window is worth ``tpWeight`` if the detector fires
  inside it, and costs ``fnWeight`` if it never does;
* each detection takes the score of its *position* in the window: a hit at the
  start of the window is worth 0.9866/0.9866 = 1.0, a hit at the right edge is
  worth 0.0, and it decays as a scaled sigmoid;
* a false positive outside a window costs ``fpWeight`` times a sigmoid of its
  distance past the most recent window, so early false positives hurt less than
  late ones, and the cost saturates at ``fpWeight``;
* only the best detection inside a window counts (a window cannot be earned
  twice), while every false positive is summed.

Thresholds are chosen by sweeping every observed score, exactly as NAB's
``Sweeper.calcScoreByThreshold`` does, and the corpus threshold is the single
threshold maximising the summed per-file score (NAB optimises one threshold per
detector over the whole corpus, not per file).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PROFILES: dict[str, dict[str, float]] = {
    "standard": {"tpWeight": 1.0, "fnWeight": 1.0, "fpWeight": 0.11},
    "reward_low_FP_rate": {"tpWeight": 1.0, "fnWeight": 1.0, "fpWeight": 0.22},
    "reward_low_FN_rate": {"tpWeight": 1.0, "fnWeight": 2.0, "fpWeight": 0.11},
}
DEFAULT_PROFILE = "standard"
PROBATION_PERCENT = 0.15


def scaled_sigmoid(relative_position: float) -> float:
    """NAB's window position -> score mapping (see module docstring)."""
    if relative_position > 3.0:
        return -1.0
    return 2.0 / (1.0 + math.exp(5.0 * relative_position)) - 1.0


_MAX_TP = scaled_sigmoid(-1.0)  # 0.98661, the best score a hit inside a window can earn


def probation_length(n_rows: int, percent: float = PROBATION_PERCENT) -> int:
    return int(min(np.floor(percent * n_rows), percent * 5000))


@dataclass
class FileSweep:
    """Per-file threshold sweep plus the per-row sweep scores NAB writes to disk."""

    point_scores: np.ndarray      # weighted S(t), shape (n,)
    window_names: np.ndarray      # object array, None outside any window
    thresholds: np.ndarray        # ascending, one row per distinct score level
    score: np.ndarray
    tp: np.ndarray
    tn: np.ndarray
    fp: np.ndarray
    fn: np.ndarray

    def at_threshold(self, threshold: float, strict: bool = False) -> int:
        """Index of the row NAB would report for ``threshold``.

        NAB keeps the *smallest* swept threshold that is >= the requested one, so
        a detection needs ``score > threshold`` — except that a swept threshold
        that equals an observed score level also captures every row sitting
        exactly on it.  ``strict=True`` drops that behaviour, which is what you
        want for a threshold picked as a quantile: rows exactly at the quantile
        are not above it.
        """
        side = "right" if strict else "left"
        i = int(np.searchsorted(self.thresholds, threshold, side=side))
        return min(i, len(self.thresholds) - 1)

    def row(self, threshold: float, strict: bool = False) -> tuple[float, int, int, int, int]:
        i = self.at_threshold(threshold, strict)
        return (float(self.score[i]), int(self.tp[i]), int(self.tn[i]),
                int(self.fp[i]), int(self.fn[i]))


def sweep_file(ticks: np.ndarray, scores: np.ndarray, windows: list[tuple[int, int]],
               profile: str = DEFAULT_PROFILE,
               probation_percent: float = PROBATION_PERCENT) -> FileSweep:
    """Reproduce ``Sweeper.calcSweepScore`` + ``calcScoreByThreshold``."""
    cm = PROFILES[profile]
    tp_w, fn_w, fp_w = cm["tpWeight"], cm["fnWeight"], cm["fpWeight"]
    n = len(scores)
    scores = np.asarray(scores, dtype=float)

    prob_len = probation_length(n, probation_percent)

    point_score = np.empty(n, dtype=float)
    point_window = np.empty(n, dtype=object)
    point_window[:] = None

    wins = [(int(a), int(b)) for a, b in windows]
    wi = 0
    cur = None
    cur_name = None
    cur_width = 0.0
    cur_right = None
    prev_right = None
    prev_width = None

    for i in range(n):
        # entering a window (NAB compares timestamps; the index is equivalent)
        if wi < len(wins) and i == wins[wi][0]:
            cur = wins[wi]
            cur_name = f"{cur[0]}"
            cur_right = cur[1]
            cur_width = float(cur_right - cur[0] + 1)
            wi += 1

        if cur is not None:
            position = -(cur_right - i + 1) / cur_width
            unweighted = scaled_sigmoid(position)
            weighted = unweighted * tp_w / _MAX_TP
        else:
            if prev_right is None:
                unweighted = -1.0
            else:
                numerator = abs(prev_right - i)
                denominator = float(prev_width - 1)
                unweighted = scaled_sigmoid(numerator / denominator if denominator > 0 else 3.0)
            weighted = unweighted * fp_w

        point_score[i] = weighted
        point_window[i] = cur_name if i >= prob_len else "probationary"

        if cur is not None and i == cur[1]:
            prev_right, prev_width = i, int(cur_width)
            cur, cur_name, cur_width, cur_right = None, None, None, None

    # ---- calcScoreByThreshold -------------------------------------------------
    scorable = [i for i in range(n) if point_window[i] != "probationary"]
    order = sorted(scorable, key=lambda i: scores[i], reverse=True)

    perf: dict[str, float] = {"fp": 0.0}
    for i in scorable:
        wn = point_window[i]
        if wn is not None:
            perf[wn] = -fn_w

    tn = sum(1 for i in scorable if point_window[i] is None)
    fn = sum(1 for i in scorable if point_window[i] is not None)
    tp = fp = 0

    ths: list[float] = []
    sc: list[float] = []
    tps: list[int] = []
    tns: list[int] = []
    fps: list[int] = []
    fns: list[int] = []

    cur_threshold = 1.1
    for i in order:
        s = scores[i]
        if s != cur_threshold:
            ths.append(cur_threshold)
            sc.append(sum(perf.values()))
            tps.append(tp); tns.append(tn); fps.append(fp); fns.append(fn)
            cur_threshold = s
        if point_window[i] is not None:
            tp += 1
            fn -= 1
            wn = point_window[i]
            perf[wn] = max(perf[wn], point_score[i])
        else:
            fp += 1
            tn -= 1
            perf["fp"] += point_score[i]

    ths.append(cur_threshold)
    sc.append(sum(perf.values()))
    tps.append(tp); tns.append(tn); fps.append(fp); fns.append(fn)

    # recorded from high threshold to low; reverse into ascending order
    return FileSweep(
        point_scores=point_score,
        window_names=point_window,
        thresholds=np.asarray(ths[::-1], dtype=float),
        score=np.asarray(sc[::-1], dtype=float),
        tp=np.asarray(tps[::-1]), tn=np.asarray(tns[::-1]),
        fp=np.asarray(fps[::-1]), fn=np.asarray(fns[::-1]),
    )


def corpus_score(sweeps: dict[str, FileSweep], threshold: float) -> float:
    return float(sum(sweeps[k].row(threshold)[0] for k in sweeps))


def optimize_threshold(sweeps: dict[str, FileSweep],
                       n_candidates: int = 20000) -> tuple[float, float]:
    """Single corpus-wide threshold maximising the summed file score.

    NAB optimises one threshold per detector over the whole corpus (not per
    file), which is what a deployment looks like.  Each file's score is a step
    function of the threshold, so the optimum lies on one of the observed score
    levels; I evaluate a quantile grid of those levels, weighted towards the
    upper tail where the useful thresholds live, and take the maximum.
    """
    all_thresholds = np.concatenate([sweeps[k].thresholds for k in sweeps])
    half = max(n_candidates // 2, 2)
    qs = np.concatenate([np.linspace(0.0, 1.0, half), np.linspace(0.9, 1.0, half)])
    candidates = np.unique(np.quantile(all_thresholds, qs))

    total = np.zeros(len(candidates))
    for sweep in sweeps.values():
        idx = np.clip(np.searchsorted(sweep.thresholds, candidates, side="left"),
                      0, len(sweep.thresholds) - 1)
        total += sweep.score[idx]
    best = int(np.argmax(total))
    return float(candidates[best]), float(total[best])


def normalize_score(raw: float, null_raw: float, oracle_raw: float) -> float:
    """NAB's 0-100 rescaling between a null and an oracle detector."""
    denom = oracle_raw - null_raw
    if denom <= 0:
        return 0.0
    return float(100.0 * (raw - null_raw) / denom)


def oracle_raw_score(sweeps: dict[str, FileSweep]) -> float:
    """Upper bound of the profile: every window earned at its best point, no FPs.

    A detector scoring 1.0 inside every labelled window and 0 elsewhere earns
    exactly ``tpWeight`` per window at its best threshold, so the ceiling is the
    number of labelled windows in the scored corpus.
    """
    total = 0
    for s in sweeps.values():
        names = {w for w in s.window_names if w not in (None, "probationary")}
        total += len(names)
    return float(total)


def null_raw_score(sweeps: dict[str, FileSweep]) -> float:
    """Raw score of NAB's null detector (constant 0.5): nothing detected."""
    return float(sum(s.score[-1] for s in sweeps.values()))
