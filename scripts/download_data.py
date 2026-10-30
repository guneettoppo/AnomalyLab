"""Download the NAB corpus (data + labels) and the reference results used to
validate our scorer implementation.

Everything lands under data/ and labels/ so the rest of the pipeline is offline.
Uses urllib with retries because this network drops connections often.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/numenta/NAB/master/"
API = "https://api.github.com/repos/numenta/NAB/git/trees/master?recursive=1"
UA = {"User-Agent": "anomalylab-downloader"}
REF_DETECTORS = ["windowedGaussian", "random", "numenta"]


def fetch(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} tries: {url} ({last})")


def main() -> int:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "labels").mkdir(exist_ok=True)
    (ROOT / "reference").mkdir(exist_ok=True)

    tree = json.loads(fetch(API))
    paths = [n["path"] for n in tree["tree"]]
    data_files = [p for p in paths if p.startswith("data/") and p.endswith(".csv")]
    ref_files = []
    for det in REF_DETECTORS:
        ref_files += [p for p in paths
                      if p.startswith(f"results/{det}/") and p.endswith(".csv")]
    ref_files.append("results/final_results.json")
    ref_files.append("labels/combined_windows.json")

    print(f"downloading {len(data_files)} series, {len(ref_files)} reference files")
    for i, rel in enumerate(data_files, 1):
        dest = ROOT / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fetch(RAW + rel))
        if i % 10 == 0 or i == len(data_files):
            print(f"  {i}/{len(data_files)} series")

    for rel in ref_files:
        if rel.startswith("labels/"):
            dest = ROOT / rel
        else:
            dest = ROOT / "reference" / Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.write_bytes(fetch(RAW + rel))

    n_series = len(list((ROOT / "data").rglob("*.csv")))
    print(f"done: {n_series} series on disk, labels at labels/combined_windows.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
