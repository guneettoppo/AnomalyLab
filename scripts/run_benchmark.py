"""Run the full benchmark: every detector over all 58 NAB streams, two thresholds.

Operating point 1 — NAB-oracle: the single corpus-wide threshold that maximises
the standard profile's score.  This is what NAB reports, and it is chosen *with
the labels*, so it is an upper bound on what a detector is worth.

Operating point 2 — deployable: fire above the 99.9th percentile of the raw
scores the detector produced during its own unlabelled learning period (NAB's
probation prefix).  No labels, no tuned constant: a per-stream rule an operator
could actually configure.  Reported next to the oracle so the honest gap is
visible.

Outputs
-------
outputs/per_series.csv   one row per (detector, series)
outputs/leaderboard.csv  one row per detector
outputs/alarms.csv       consensus alarms with their feature evidence
reports/metrics.json     machine-readable copy of the leaderboard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomalylab import nab_scoring as ns                                  # noqa: E402
from anomalylab.detectors import build_detectors, explain                # noqa: E402
from anomalylab.features import calibrated_value, pit_scores, tail_threshold  # noqa: E402
from anomalylab.io import load_corpus                                    # noqa: E402
from anomalylab.metrics import evaluate                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONSENSUS = ("consensus_mean", "consensus_max")


def score_series(dets: dict, series) -> dict[str, tuple[np.ndarray, object]]:
    """Run every base detector once, then build both fusions from the cache."""
    pl = series.probation_len()
    cache: dict[str, tuple[np.ndarray, object]] = {}
    for name, det in dets.items():
        if name in CONSENSUS:
            continue
        out = det.fit_predict(series)
        cache[name] = (out.raw, out)
    members = [(n, out) for n, (raw, out) in cache.items() if n != "null"]
    stack = np.vstack([pit_scores(out.raw, pl) for _, out in members])
    mean_raw = stack.mean(axis=0)
    max_raw = stack.max(axis=0)
    # feature evidence comes from whichever member exposes per-feature terms
    donor = next((out for _, out in members if out.contributions is not None), members[0][1])
    for name, raw in zip(CONSENSUS, (mean_raw, max_raw)):
        if name in dets:
            cache[name] = (raw, type(donor)(
                raw=raw, contributions=donor.contributions,
                feature_names=donor.feature_names))
    return {n: cache[n] for n in dets if n in cache}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=list(ns.PROFILES))
    ap.add_argument("--quantile", type=float, default=0.999,
                    help="deployable threshold = this quantile of probation-period scores")
    ap.add_argument("--detectors", default="")
    ap.add_argument("--limit", type=int, default=0, help="only first N series (debug)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    corpus = load_corpus()
    if args.limit:
        corpus = corpus[: args.limit]
    dets = build_detectors()
    if args.detectors:
        wanted = [d.strip() for d in args.detectors.split(",") if d.strip()]
        dets = {k: v for k, v in dets.items() if k in wanted}

    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    rows: list[dict] = []
    alarms: list[dict] = []
    sweeps: dict[str, dict[str, ns.FileSweep]] = {k: {} for k in dets}
    t0 = time.time()

    for ci, series in enumerate(corpus, 1):
        pl = series.probation_len()
        label = series.label_mask()
        results = score_series(dets, series)

        for name in dets:
            raw, out = results[name]
            score = pit_scores(raw, pl)   # null maps to a constant 0, as it should
            sw = ns.sweep_file(series.ticks, score, series.windows, args.profile)
            sweeps[name][series.key] = sw

            thr = tail_threshold(raw, pl, args.quantile)
            pred = raw > thr
            pred[:pl] = False
            m = evaluate(pred, label, series.windows, series.ticks)
            tail = label[pl:]
            ap_pts = (float(average_precision_score(tail, score[pl:]))
                      if tail.any() else float("nan"))

            rows.append({
                "detector": name, "series": series.key, "category": series.category,
                "n": series.n, "windows": len(series.windows), "probation": pl,
                "nab_file_score_oracle": float(sw.score.max()),
                "nab_file_score_deployable": sw.row(
                    calibrated_value(raw, pl, thr), strict=True)[0],
                "deployable_threshold": thr,
                "average_precision": ap_pts,
                **{f"m_{k}": v for k, v in m.items()},
            })

            if name == "consensus_mean" and series.windows:
                hits = np.flatnonzero(pred & label)
                for row_i in hits[:3]:
                    why = explain(out, int(row_i), 4)
                    alarms.append({
                        "series": series.key,
                        "timestamp": pd.Timestamp(int(series.ticks[row_i])),
                        "score": round(float(score[row_i]), 4),
                        "value": round(float(series.values[row_i]), 6),
                        "evidence_from": "pca_recon reconstruction residual",
                        "evidence": "; ".join(f"{f}={v:+.2f}" for f, v in why),
                    })
        print(f"[{ci}/{len(corpus)}] {series.key:58s} {time.time()-t0:6.1f}s")

    per_series = pd.DataFrame(rows)

    # ---- corpus-level scores --------------------------------------------------
    null_raw = ns.null_raw_score(sweeps["null"])
    oracle_raw = ns.oracle_raw_score(sweeps["null"])
    board = []
    for name in dets:
        t_opt, raw = ns.optimize_threshold(sweeps[name])
        sub = per_series[per_series.detector == name]
        dep_raw = float(sub.nab_file_score_deployable.sum())
        board.append({
            "detector": name,
            "nab_score": ns.normalize_score(raw, null_raw, oracle_raw),
            "nab_score_deployable": ns.normalize_score(dep_raw, null_raw, oracle_raw),
            "oracle_threshold": t_opt,
            "deployable_quantile": args.quantile,
            "raw_corpus_score": raw,
            "mean_average_precision": float(np.nanmean(sub.average_precision)),
            "mean_precision": float(np.nanmean(sub.m_precision)),
            "mean_recall": float(np.nanmean(sub.m_recall)),
            "mean_f1": float(np.nanmean(sub.m_f1)),
            "window_recall": float(np.nansum(sub.m_windows_hit) /
                                   max(float(np.nansum(sub.windows)), 1)),
            "false_alarms": float(np.nansum(sub.m_false_alarms)),
            "false_alarms_per_day": float(np.nansum(sub.m_false_alarms) /
                                          max(float(np.nansum(sub.m_span_days)), 1e-9)),
            "median_delay_pts": (float(np.nanmedian(sub.m_median_delay_pts))
                                 if np.isfinite(np.nanmedian(sub.m_median_delay_pts))
                                 else float("nan")),
        })
    board_df = pd.DataFrame(board).sort_values("nab_score", ascending=False)

    per_series.to_csv(ROOT / "outputs" / "per_series.csv", index=False)
    board_df.to_csv(ROOT / "outputs" / "leaderboard.csv", index=False)
    pd.DataFrame(alarms).to_csv(ROOT / "outputs" / "alarms.csv", index=False)
    with open(ROOT / "reports" / "metrics.json", "w") as fh:
        json.dump({
            "profile": args.profile,
            "deployable_quantile": args.quantile,
            "n_series": len(corpus),
            "n_windows": int(per_series[per_series.detector == "null"].windows.sum()),
            "null_raw": null_raw,
            "oracle_raw": oracle_raw,
            "leaderboard": board_df.to_dict(orient="records"),
            "runtime_seconds": round(time.time() - t0, 1),
        }, fh, indent=2)

    pd.set_option("display.width", 220)
    print()
    print(board_df[["detector", "nab_score", "nab_score_deployable", "mean_average_precision",
                    "window_recall", "mean_precision", "false_alarms",
                    "false_alarms_per_day"]].round(3).to_string(index=False))
    print(f"\nwrote outputs/ + reports/metrics.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
