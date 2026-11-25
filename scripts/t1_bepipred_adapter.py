#!/usr/bin/env python3
"""
Turn Task I BepiPred-3.0 output into per-case score vectors.

How this differs from the other tasks' adapter. On Tasks II and III there is one
antigen, so `bepipred_adapter.py` writes one identical vector into every case.
Task I has 1,529 antigens, so each case gets the vector for its own antigen. The
method is antibody-blind in the same way either way: every antibody that shares an
antigen gets a bit-identical prediction. That is what makes BepiPred the reference
point, and on Task I it is the only antibody-blind baseline besides random,
because the RBD/RBM region masks have no meaning when the antigens are arbitrary.

The per-residue score is used instead of the rolling-average "linear epitope
score", because the rolling window smears signal across neighbouring positions and
the gold here is a per-residue contact set.

A case whose antigen was excluded upstream (over the length cap, or containing a
non-standard residue) gets no file, and the evaluator reports the coverage.
Substituting a zero vector would put those cases at the floor and make BepiPred
look worse than it is.

    PYTHONPATH=scripts:config python scripts/t1_bepipred_adapter.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                        # noqa: E402

SCORE_COL = "BepiPred-3.0 score"
LINEAR_COL = "BepiPred-3.0 linear epitope score"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(PROJECT / "work/task1/bepipred/out/raw_output.csv"))
    ap.add_argument("--out_dir", default=str(PROJECT / "work/task1/scores_bepipred"))
    ap.add_argument("--linear", action="store_true",
                    help="use the rolling-average linear epitope score instead")
    a = ap.parse_args()

    raw = Path(a.raw)
    if not raw.exists():
        sys.exit(f"no BepiPred-3.0 output at {raw} -- has that stage run?")
    d = pd.read_csv(raw)
    col = LINEAR_COL if a.linear else SCORE_COL
    by_ag = {str(k): g[col].to_numpy(float) for k, g in d.groupby("Accession")}
    print(f"  {len(by_ag)} antigens in {raw.name}")

    man = pd.read_csv(cfg.MANIFEST)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_missing = n_len = 0
    for _, r in man[man.usable].iterrows():
        s = by_ag.get(str(r.antigen_id))
        if s is None:
            n_missing += 1
            continue
        # A length mismatch means the vector is not aligned to this row's labels,
        # so it is refused rather than truncated: a trim would shift every
        # downstream residue index.
        if len(s) != int(r.antigen_len):
            print(f"    {r.case_id}: {len(s)} scores vs antigen {r.antigen_len} -- skipped")
            n_len += 1
            continue
        np.savez_compressed(out / f"{r.case_id}.npz", score=s)
        n_ok += 1
    print(f"  wrote {n_ok} score vectors -> {out}")
    print(f"  no BepiPred output for {n_missing} cases "
          f"(antigen over the length cap or containing a non-standard residue)")
    if n_len:
        print(f"  {n_len} cases refused on a length mismatch")


if __name__ == "__main__":
    main()
