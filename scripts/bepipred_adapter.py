#!/usr/bin/env python3
"""
Turn BepiPred-3.0 output into our per-case score format.

BepiPred-3.0 sees only the antigen sequence. It has no way to take an antibody as
input, so it emits one score vector per antigen, which we reuse for every
antibody. That is the definition of the antibody-blind class, and it makes
BepiPred the reference point the benchmark is built around:

  * its antibody-swap gap is zero in expectation, because the prediction does not
    change when the antibody changes;
  * on Task III its argmax domain is the same for every antibody, so it is an
    "always predict domain X" strategy, discovered rather than imposed.

Writing one identical vector into every case's npz is the faithful encoding of
what the method does.

    EIB_TASK=task2 PYTHONPATH=scripts python scripts/bepipred_adapter.py \
        --raw work/bepipred/out/raw_output.csv --accession spike \
        --out_dir work/scores_bepipred
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import cfg

SCORE_COL = "BepiPred-3.0 score"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="BepiPred raw_output.csv")
    ap.add_argument("--accession", required=True, help="which antigen row set to use")
    ap.add_argument("--out_dir", help="per-case npz directory (Tasks I-III)")
    # The DMS-family cost scripts read one vector rather than one file per case.
    # dms_04/dms_08 load `cfg.WORK / <arm>.npy` directly, because on those arms
    # the antigen is a single protein and a per-case copy of the same 504 or 856
    # numbers would be six to twelve identical files. This writes that one vector.
    # The content is the same column either way, so the two outputs cannot
    # disagree: `work/dms/bepipred_env.npy` is byte-identical to the
    # `BepiPred-3.0 score` column of work/bepipred/out/raw_output.csv for
    # Accession `env`, verified 2026-11-04.
    ap.add_argument("--npy", help="instead, write the antigen's score vector to "
                                  "this single .npy (the DMS-family convention)")
    ap.add_argument("--linear", action="store_true",
                    help="use the rolling-average 'linear epitope score' instead of the "
                         "per-residue score. Default is the per-residue score, because "
                         "the rolling average smears signal across a window and our gold "
                         "is a per-residue contact set.")
    a = ap.parse_args()

    d = pd.read_csv(a.raw)
    d = d[d.Accession.astype(str) == a.accession]
    if d.empty:
        raise SystemExit(f"accession {a.accession!r} not in {a.raw}; "
                         f"have {sorted(pd.read_csv(a.raw).Accession.unique())}")
    col = "BepiPred-3.0 linear epitope score" if a.linear else SCORE_COL
    score = d[col].to_numpy(float)
    if len(score) != cfg.ANTIGEN_LEN:
        raise SystemExit(f"{a.accession}: {len(score)} residues but this task's antigen is "
                         f"{cfg.ANTIGEN_LEN} -- wrong antigen or wrong EIB_TASK.")

    if bool(a.out_dir) == bool(a.npy):
        raise SystemExit("give exactly one of --out_dir (per-case npz) or --npy "
                         "(one vector, the DMS-family convention)")
    if a.npy:
        out = Path(a.npy); out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, score)
        print(f"wrote {len(score)} residues ({col}) -> {out}")
        print(f"  antigen {a.accession}: score range "
              f"{score.min():.4f}-{score.max():.4f}, mean {score.mean():.4f}")
        return

    man = pd.read_csv(cfg.MANIFEST)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for cid in man.case_id:
        # identical vector for every antibody -- see module docstring
        np.savez_compressed(out / f"{cid}.npz", contact_mean=score, contact_top=score,
                            n_samples=1)
    print(f"wrote {len(man)} identical score vectors ({col}) -> {out}")
    print(f"  antigen {a.accession}: {len(score)} residues, "
          f"score range {score.min():.4f}-{score.max():.4f}, mean {score.mean():.4f}")


if __name__ == "__main__":
    main()
