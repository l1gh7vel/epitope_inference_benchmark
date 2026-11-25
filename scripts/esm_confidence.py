#!/usr/bin/env python3
"""Build the per-case ESMFold2 iPTM, pTM and pLDDT table."""
import csv
import sys
from pathlib import Path

import env

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ESM = env.ESM_PREFIX
OUT = PROJECT / "results" / "esm_confidence_per_case.csv"

# (task tag, manifest, how the row_id maps onto our manifest)
SOURCES = [("task2", ESM / "results_rbm/fold_manifest.csv", "row_index"),
           ("task3", ESM / "results_env/fold_manifest.csv", "row_index")]


def parse(path):
    """Yield (row_id, iptm, ptm, plddt) for successful folds, both layouts."""
    rows = list(csv.reader(open(path)))
    hdr = rows[0]
    i_status = hdr.index("status")
    out = []
    for r in rows[1:]:
        if len(r) <= i_status:
            continue
        if r[i_status] == "ok":                 # current layout
            off = i_status
        elif i_status >= 1 and r[i_status - 1] == "ok":
            off = i_status - 1                  # older layout, shifted left by one
        else:
            continue
        try:
            out.append((r[0], float(r[off + 1]), float(r[off + 2]), float(r[off + 3])))
        except (ValueError, IndexError):
            continue
    return out


def main():
    sys.path.insert(0, str(PROJECT / "config"))
    frames = []
    for task, path, _how in SOURCES:
        if not path.exists():
            print(f"  {task}: {path} absent, skipped")
            continue
        recs = parse(path)
        man = pd.read_csv(PROJECT / "data" / "manifests" / f"{task}_manifest.csv")
        by_row = {int(r.row_index): r.case_id for _, r in man.iterrows()} \
            if "row_index" in man else {}
        n_map = 0
        for row_id, iptm, ptm, plddt in recs:
            idx = int(str(row_id).split("_")[0])
            cid = by_row.get(idx)
            if cid is None:
                continue
            frames.append({"case_id": cid, "task": task, "esm_iptm": iptm,
                           "esm_ptm": ptm, "esm_plddt": plddt})
            n_map += 1
        print(f"  {task}: {len(recs)} successful folds in the manifest, "
              f"{n_map} mapped onto our cases")
    if not frames:
        raise SystemExit("no ESMFold2 confidence rows were available; output unchanged")
    d = pd.DataFrame(frames).drop_duplicates("case_id")
    required = {"case_id", "task", "esm_iptm", "esm_ptm", "esm_plddt"}
    if not required.issubset(d.columns):
        raise SystemExit("ESMFold2 confidence table is incomplete; output unchanged")
    tmp = OUT.with_suffix(".csv.tmp")
    d.to_csv(tmp, index=False)
    tmp.replace(OUT)
    print(f"\n{len(d)} rows -> {OUT}")
    for t, g in d.groupby("task"):
        print(f"  {t:8s} iPTM {g.esm_iptm.mean():.3f}  pTM {g.esm_ptm.mean():.3f}  "
              f"pLDDT {g.esm_plddt.mean():.3f}")


if __name__ == "__main__":
    main()
