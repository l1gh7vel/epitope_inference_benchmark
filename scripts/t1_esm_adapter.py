#!/usr/bin/env python3
"""
Score the Task I ESMFold2 structures through OUR harness, in the Task I npz format.

Same principle as `esm_adapter.py`: reuse stage 04's own `split_chains` and
`score_one_structure`, so the contact readout (heavy-atom minimum distance,
contact = 1/(1+d)) is computed identically to AF3's and the comparison is
like-for-like. Nothing about the contact definition is re-derived.

TWO TASK I DIFFERENCES from that adapter:
  * mapping is direct -- our own driver wrote `<case_id>.cif`, so there is no
    row-index cross-check to get wrong;
  * the antigen length is PER ROW, not `cfg.ANTIGEN_LEN`. A length mismatch is
    therefore a real signal (wrong chain picked, truncated fold) and is refused
    rather than trimmed, because trimming would shift every residue index against
    the gold labels.

ESMFold2 produces ONE structure, so `contact_mean == contact_top` -- there is
nothing to average. State that wherever it is compared with AF3's 25-structure
ensemble.

    EIB_TASK=task1 PYTHONPATH=scripts:config python scripts/t1_esm_adapter.py

NB EIB_TASK=task1 is REQUIRED: stage 04 resolves its own `cfg` through `common`,
which reads EIB_TASK and defaults to task2. Without it the shared length assertion
fires against Spike's 1,273 residues on every Task I antigen.
"""
import argparse
import importlib.util as _iu
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
import task1 as cfg                                        # noqa: E402

_spec = _iu.spec_from_file_location("stage04", Path(__file__).with_name("04_score_structures.py"))
_s04 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_s04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", default=str(PROJECT / "work/task1/esm/cifs"))
    ap.add_argument("--out_dir", default=str(PROJECT / "work/task1/scores_esm"))
    a = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST).set_index("case_id")
    cif_dir, out = Path(a.cif_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    done = skipped = badlen = 0
    for cif in sorted(cif_dir.glob("*.cif")):
        cid = cif.stem
        if cid not in man.index:
            skipped += 1
            continue
        d, _ = _s04.score_one_structure(cif, do_bsa=False)
        want = int(man.loc[cid, "antigen_len"])
        if len(d) != want:
            print(f"  LENGTH MISMATCH {cid}: scored {len(d)} vs antigen {want} -- refused")
            badlen += 1
            continue
        contact = 1.0 / (1.0 + d)
        np.savez_compressed(out / f"{cid}.npz", score=contact,
                            contact_mean=contact, contact_top=contact,
                            dmin=d, n_samples=1)
        done += 1
    print(f"scored {done} ESMFold2 structures -> {out}")
    print(f"  skipped {skipped} (not in the manifest) | length mismatches refused {badlen}")


if __name__ == "__main__":
    main()
