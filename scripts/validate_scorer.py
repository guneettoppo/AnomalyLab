"""Validate the ported NAB scorer against NAB's own published results.

NAB ships, for every detector, one results CSV per series (`anomaly_score`
column) plus `<detector>_standard_scores.csv` holding the per-file score it
reported at the corpus-optimal threshold.  If my port is faithful, feeding the
shipped `anomaly_score` column through `sweep_file()` must reproduce those
per-file scores, and the normalised corpus score must match
`results/final_results.json`.

Run:  python scripts/validate_scorer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomalylab import nab_scoring as ns            # noqa: E402
from anomalylab.io import load_labels, load_series, list_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"
DETECTORS = ["windowedGaussian", "random", "numenta"]
# The shipped `random` per-file results cover fewer rows than the data files
# (TN counts do not even add up to NAB's own Total_Count), so only its corpus
# score is a meaningful reference.
PER_FILE_STRICT = {"windowedGaussian", "numenta"}


def check_detector(det: str, labels, by_stem) -> tuple[float, float, float]:
    published = pd.read_csv(REF / f"{det}_standard_scores.csv", keep_default_na=False)
    published = published[published["Profile"] == "standard"]
    pub = {str(r.File).replace(".csv", "").split("/")[-1]: r for r in published.itertuples()}
    threshold = float(published["Threshold"].iloc[0])

    sweeps: dict[str, ns.FileSweep] = {}
    worst, checked = 0.0, 0
    for path in sorted(REF.glob(f"{det}_*.csv")):
        if path.name.endswith("_scores.csv"):
            continue
        stem = path.stem.replace(f"{det}_", "", 1)
        key = by_stem.get(stem)
        if key is None:
            continue
        ref = pd.read_csv(path, parse_dates=["timestamp"])
        series = load_series(key, labels)
        if len(ref) != series.n:
            print(f"  length mismatch {key}: {len(ref)} vs {series.n}")
            continue
        sw = ns.sweep_file(series.ticks, ref["anomaly_score"].to_numpy(float), series.windows)
        sweeps[key] = sw
        if stem in pub:
            worst = max(worst, abs(sw.row(threshold)[0] - float(pub[stem].Score)))
            checked += 1

    raw = ns.corpus_score(sweeps, threshold)
    norm = ns.normalize_score(raw, ns.null_raw_score(sweeps), ns.oracle_raw_score(sweeps))
    with open(REF / "final_results.json") as fh:
        official = json.load(fh)[det]["standard"]
    print(f"{det:18s} files={checked:3d}  max|file diff|={worst:.2e}  "
          f"ours={norm:8.4f}  NAB={official:8.4f}  delta={abs(norm - official):.3f}")
    return worst, norm, official


def main() -> int:
    labels = load_labels()
    by_stem = {Path(k).stem: k for k in list_series()}
    ok = True
    for det in DETECTORS:
        try:
            worst, norm, official = check_detector(det, labels, by_stem)
        except FileNotFoundError as exc:
            print(f"{det}: missing reference file ({exc})")
            ok = False
            continue
        ok &= (abs(norm - official) < 0.01)
        if det in PER_FILE_STRICT:
            ok &= worst < 1e-6
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
