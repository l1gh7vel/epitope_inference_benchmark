#!/usr/bin/env python3
"""
Rebuild an ESMFold2 fold manifest from the structures actually on disk.

The fold driver appends to `fold_manifest.csv` as it goes, so a run that is
sharded and then resumed after a timeout leaves a manifest describing only the
last shards' work while the cif directory holds everything. Measured on the Task
III spike arm: 50 manifest rows against 300 cifs. Scoring from the manifest then
drops five sixths of the data, and the loss is invisible because the adapter
reports only what it found.

The cif filenames carry the two things the adapter needs, the row index and the
antibody name, in the form `<row_index>_<name>.cif`. Disk is the ground truth
here, so the manifest is regenerated from it.

Confidence columns are left blank: they live only in the driver's own output and
cannot be recovered from a structure file. Nothing downstream of the contact
scoring uses them.

    PYTHONPATH=scripts python scripts/esm_manifest_from_cifs.py \
        --cif_dir work/esm_task3cov/cifs --out work/esm_task3cov/fold_manifest_all.csv
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", default=None,
                    help="existing manifest to take confidence columns from where it "
                         "covers a row")
    a = ap.parse_args()

    known = {}
    if a.merge and Path(a.merge).exists():
        for r in csv.DictReader(open(a.merge)):
            known[str(r.get("row_id"))] = r

    cifs = sorted(Path(a.cif_dir).glob("*.cif"))
    cols = ["row_id", "name", "status", "iptm", "ptm", "plddt_mean", "out_path"]
    n_merged = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in cifs:
            stem = c.stem
            name = stem.split("_", 1)[1] if "_" in stem else stem
            row = {"row_id": stem, "name": name, "status": "ok", "out_path": str(c)}
            src = known.get(stem)
            if src:
                n_merged += 1
                for k in ("iptm", "ptm", "plddt_mean"):
                    row[k] = src.get(k, "")
            w.writerow(row)
    print(f"  {len(cifs)} structures on disk -> {a.out}")
    print(f"  confidence columns recovered for {n_merged} of them from {a.merge}")


if __name__ == "__main__":
    main()
