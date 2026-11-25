#!/usr/bin/env python3
"""
Score ESMFold2-predicted complexes through our harness, in our npz format.

The adapter reuses stage 04's own `split_chains` and `score_one_structure` -- same
chain selection, same heavy-atom minimum distance, same contact = 1/(1+d) -- so
the ESMFold2 and AF3 readouts are computed identically. Nothing about the contact
definition is re-derived here.

One difference between the two. AF3 gives 25 structures per complex (5 seeds x 5
diffusion samples), which we ensemble; ESMFold2 as run gives one. So for ESM,
`contact_mean` == `contact_top`, with nothing to average. That is a property of
how the two models were run, and it should be stated wherever the two are
compared: AF3's number is an ensemble, ESM's is a single structure.

Mapping to our cases is by row index into the same source CSV, cross-checked
against the antibody name, so a reordered manifest cannot misalign scores.

    PYTHONPATH=scripts python scripts/esm_adapter.py \
        --cif_dir $EIB_ESM_PREFIX/results_env/cifs \
        --fold_manifest $EIB_ESM_PREFIX/results_env/fold_manifest.csv \
        --out_dir work/task3/scores_esm
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import cfg
import importlib.util as _iu
_spec = _iu.spec_from_file_location("stage04", Path(__file__).with_name("04_score_structures.py"))
_s04 = _iu.module_from_spec(_spec); _spec.loader.exec_module(_s04)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--fold_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name_col", default="name")
    a = ap.parse_args()

    ours = pd.read_csv(cfg.MANIFEST)
    by_row = {int(r.row_index): r for _, r in ours.iterrows()}
    fm = pd.read_csv(a.fold_manifest)
    cif_dir, out = Path(a.cif_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    done = skipped = mismatch = 0
    for _, r in fm.iterrows():
        row_id = str(r.row_id)
        ridx = int(row_id.split("_")[0])
        if ridx not in by_row:
            skipped += 1
            continue
        ours_row = by_row[ridx]
        # Cross-check the antibody name when both sides carry one (Task III does,
        # Task II's manifest does not). A reordered source CSV would otherwise
        # misalign every score.
        if a.name_col in fm.columns and "ab_name" in ours.columns:
            if norm(r[a.name_col]) != norm(ours_row.ab_name):
                mismatch += 1
                continue
        cif = cif_dir / f"{row_id}.cif"
        if not cif.exists():
            skipped += 1
            continue
        d, _ = _s04.score_one_structure(cif, do_bsa=False)
        if len(d) != cfg.ANTIGEN_LEN:
            print(f"  LENGTH MISMATCH {row_id}: antigen {len(d)} != {cfg.ANTIGEN_LEN}, skipping")
            skipped += 1
            continue
        contact = 1.0 / (1.0 + d)
        np.savez_compressed(out / f"{ours_row.case_id}.npz",
                            contact_mean=contact,   # one structure: mean == top
                            contact_top=contact,
                            dmin_mean=d, dmin_top=d,
                            n_samples=1)
        done += 1

    print(f"scored {done} ESM structures -> {out}")
    print(f"  skipped {skipped} (no cif / not in our manifest) | name mismatches {mismatch}")
    if mismatch:
        print("  NB mismatches mean the row indices disagree -- investigate before trusting these.")


if __name__ == "__main__":
    main()
