#!/usr/bin/env python3
"""Score the DMS-panel ESMFold2 structures through OUR harness.

Same principle as `esm_adapter.py` / `t1_esm_adapter.py`: reuse stage 04's own
`split_chains` and `score_one_structure`, so the contact readout (heavy-atom
minimum distance, contact = 1/(1+d)) is computed identically to AF3's and the
comparison is like-for-like. Nothing about the contact definition is re-derived.

The antigen is fixed per arm, so a length other than cfg.ANTIGEN_LEN means the
wrong chain was picked or the fold was truncated. That is refused, not trimmed:
trimming would shift every residue index against the gold positions.

ESMFold2 produces ONE structure, so contact_mean == contact_top -- there is
nothing to average. State that wherever it is compared with AF3's 25-structure
ensemble.

WHICH ARM: EIB_TASK, the same switch stages 01-04 follow, defaulting to `dms` so
the HIV call is byte-for-byte what it was before the three-arm split. rsv_f and
niv fold their own antigens into work/<arm>/esm/cifs and keep their own manifest,
so nothing here is HIV-specific beyond that default.

    EIB_TASK=dms   PYTHONPATH=scripts:config python scripts/dms_esm_adapter.py
    EIB_TASK=rsv_f PYTHONPATH=scripts:config python scripts/dms_esm_adapter.py
    EIB_TASK=niv   PYTHONPATH=scripts:config python scripts/dms_esm_adapter.py
"""
import argparse, importlib, importlib.util as _iu, os, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
cfg = importlib.import_module(os.environ.get("EIB_TASK", "dms"))   # noqa: E402

_spec = _iu.spec_from_file_location("stage04", Path(__file__).with_name("04_score_structures.py"))
_s04 = _iu.module_from_spec(_spec); _spec.loader.exec_module(_s04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", default=str(cfg.WORK / "esm" / "cifs"))
    ap.add_argument("--out_dir", default=str(cfg.WORK / "scores_esm"))
    a = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST).set_index("case_id")
    cif_dir, out = Path(a.cif_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = badlen = 0
    for cif in sorted(cif_dir.glob("*.cif")):
        cid = cif.stem
        if cid not in man.index:
            skipped += 1; continue
        d, _ = _s04.score_one_structure(cif, do_bsa=False)
        if len(d) != cfg.ANTIGEN_LEN:
            print(f"  LENGTH MISMATCH {cid}: scored {len(d)} vs {cfg.ANTIGEN_LEN} -- refused")
            badlen += 1; continue
        contact = 1.0 / (1.0 + d)
        np.savez_compressed(out / f"{cid}.npz", score=contact, contact_mean=contact,
                            contact_top=contact, dmin=d, n_samples=1)
        done += 1
    print(f"scored {done} ESMFold2 structures -> {out}")
    print(f"  skipped {skipped} (not in manifest) | length mismatches refused {badlen}")


if __name__ == "__main__":
    main()
