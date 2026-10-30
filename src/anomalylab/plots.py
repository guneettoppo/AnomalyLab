"""Figures for the report. Swiss layout: flat paper, ink lines, one red accent,
no chartjunk, no rounded corners, no gradients."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = "#FAFAF9"
INK = "#111111"
ACCENT = "#E53E3E"
MUTED = "#8A8A85"
RULE = "#DEDED8"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": ["Inter", "Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "600",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def leaderboard(board: pd.DataFrame, path: Path) -> Path:
    b = board[board.detector != "null"].sort_values("nab_score")
    fig, ax = plt.subplots(figsize=(7.6, 0.46 * len(b) + 1.6))
    y = np.arange(len(b))
    h = 0.36
    ax.barh(y + h / 2, b.nab_score, height=h, color=INK, label="NAB score, label-chosen global threshold")
    ax.barh(y - h / 2, b.nab_score_deployable, height=h, color=ACCENT,
            label="same detector, per-stream deployable threshold")
    ax.set_yticks(y, b.detector)
    ax.set_xlabel("normalised NAB score  (0 = null detector, 100 = oracle; below 0 = worse than silence)")
    ax.axvline(0, color=INK, lw=1)
    for yi, v in zip(y + h / 2, b.nab_score):
        ax.text(v + (0.8 if v >= 0 else -0.8), yi, f"{v:.1f}", va="center", fontsize=8,
                color=INK, ha="left" if v >= 0 else "right")
    for yi, v in zip(y - h / 2, b.nab_score_deployable):
        ax.text(v + (0.8 if v >= 0 else -0.8), yi, f"{v:.1f}", va="center", fontsize=8,
                color=ACCENT, ha="left" if v >= 0 else "right")
    lo = min(-14.0, float(b.nab_score_deployable.min()) * 1.15)
    hi = max(float(b.nab_score.max()), 1.0) * 1.18
    ax.set_xlim(lo, hi)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.set_title("Anomaly detection on 58 NAB streams", pad=12)
    return _save(fig, path)


def category_heatmap(per_series: pd.DataFrame, path: Path) -> Path:
    d = per_series[per_series.detector != "null"]
    piv = d.pivot_table(index="detector", columns="category",
                        values="nab_file_score_oracle", aggfunc="mean")
    order = ["artificialWithAnomaly", "realAWSCloudwatch", "realAdExchange",
             "realKnownCause", "realTraffic", "realTweets"]
    piv = piv[[c for c in order if c in piv.columns]]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(piv) + 1.6))
    vmax = float(np.nanmax(np.abs(piv.values))) or 1.0
    im = ax.imshow(piv.values, cmap="RdGy", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [c.replace("real", "") for c in piv.columns], rotation=25,
                  ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=PAPER if v > vmax * 0.55 else INK)
    ax.set_title("Mean per-file NAB score by stream family", pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean score")
    return _save(fig, path)


def detections(examples: list[dict], path: Path) -> Path:
    n = len(examples)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 5.0), sharex="col",
                             gridspec_kw={"height_ratios": [1.35, 1]})
    axes = np.atleast_2d(axes)
    for j, ex in enumerate(examples):
        t = pd.to_datetime(ex["ticks"])
        a0 = axes[0, j]
        a0.plot(t, ex["values"], color=INK, lw=0.8)
        for (s, e) in ex["windows"]:
            a0.axvspan(t[s], t[e], color=ACCENT, alpha=0.16, lw=0)
        a0.set_title(ex["name"], fontsize=9)
        a0.set_ylabel("value")
        a0.tick_params(axis="x", labelrotation=30, labelsize=7)

        a1 = axes[1, j]
        k = 15
        smooth = np.convolve(ex["score"], np.ones(k) / k, mode="full")[: len(ex["score"])]
        a1.plot(t, smooth, color=ACCENT, lw=1.0, label=f"score, {k}-point trailing mean")
        a1.axhline(ex.get("threshold", 0.5), color=INK, ls=(0, (3, 3)), lw=0.9)
        a1.set_ylim(-0.02, 1.02)
        a1.set_ylabel("calibrated score")
        a1.tick_params(axis="x", labelrotation=30, labelsize=7)
        hits = ex["hits"]
        if len(hits):
            a1.plot(t[hits], ex["score"][hits], "o", ms=2.4, color=INK, lw=0, alpha=0.85)
        if j == 0:
            a1.legend(loc="lower left", ncols=2, fontsize=7)
    fig.suptitle("Consensus detector at the deployable threshold\nred band = labelled incident, "
                 "dots = alarms, dashed line = per-stream threshold",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def evidence(names: list[str], values: np.ndarray, path: Path, title: str) -> Path:
    order = np.argsort(np.abs(values))
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(order) + 1.3))
    colors = [ACCENT if v > 0 else MUTED for v in values[order]]
    ax.barh(range(len(order)), values[order], color=colors, height=0.62)
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    ax.axvline(0, color=RULE, lw=1)
    ax.set_xlabel("signed reconstruction residual (standardised units)")
    ax.set_title(title, pad=10)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    return _save(fig, path)


def tradeoff(board: pd.DataFrame, path: Path) -> Path:
    b = board[(board.detector != "null") & np.isfinite(board.mean_average_precision)]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sizes = 40 + 260 * (b.window_recall.fillna(0))
    ax.scatter(b.false_alarms_per_day, b.mean_average_precision, s=sizes,
               color=[ACCENT if str(d).startswith("consensus") else INK for d in b.detector],
               alpha=0.9, lw=0)
    for _, r in b.iterrows():
        ax.annotate(r.detector, (r.false_alarms_per_day, r.mean_average_precision),
                    textcoords="offset points", xytext=(7, 5), fontsize=8, color=INK)
    ax.set_xlabel("false alarms per stream-day")
    ax.set_ylabel("mean average precision")
    ax.set_title("Detection quality against operator load\nbubble size = share of incidents caught",
                 pad=12)
    ax.grid(True)
    ax.set_axisbelow(True)
    return _save(fig, path)
