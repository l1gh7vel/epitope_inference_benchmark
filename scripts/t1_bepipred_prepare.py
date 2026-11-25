#!/usr/bin/env python3
"""
Write the Task I antigen FASTA for BepiPred-3.0, and say what it will cost.

BepiPred-3.0 is the one method in the evaluated set that scales to the whole of
Task I without folding anything: it reads only the antigen sequence, so the unit
of work is the 1,627 distinct antigens rather than the 4,585 complexes. Every
complex sharing an antigen gets the same vector, which is the definition of the
antibody-blind class.

Two exclusions, both written out.

  Length. BepiPred runs ESM-2 over the whole sequence in one pass and attention is
  quadratic, so a 14,507-residue antigen is an out-of-memory case rather than a
  slow one. Antigens above the cap go to `antigens_too_long.fasta`. Antigens
  within the 1,400 aa evaluated set cap are unaffected.

  Non-standard residues. BepiPred hard-fails on anything outside the 20 standard
  amino acids: it exits on the first offending sequence and folds nothing at all.
  Measured: 12 of 1,627 antigens contain `X` (141 occurrences), affecting 17 of
  4,516 complexes. These are excluded rather than repaired. Substituting X with a
  real residue would invent sequence that the per-residue epitope labels are
  indexed against, so a one-character "fix" would shift the gold labels of exactly
  the rows it touched. Losing 0.4% of the full-set arm is cheaper than that.

    PYTHONPATH=scripts:config python scripts/t1_bepipred_prepare.py [--max_len 2000]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                        # noqa: E402

OUT = PROJECT / "work" / "task1" / "bepipred"


def ag_id(seq):
    """Stable id for one antigen sequence, the FASTA header BepiPred echoes back."""
    return "ag_" + hashlib.sha1(seq.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=2000,
                    help="skip antigens longer than this (ESM-2 attention is quadratic)")
    a = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    man["antigen_id"] = man.antigen_sequence.map(ag_id)
    man.to_csv(cfg.MANIFEST, index=False)          # persist the id for the adapter

    ag = man[man.usable].drop_duplicates("antigen_sequence")
    std = ag.antigen_sequence.map(lambda s: not (set(s) - VALID_AA))
    keep = ag[(ag.antigen_len <= a.max_len) & std]
    skip = ag[(ag.antigen_len > a.max_len) & std]
    nonstd = ag[~std]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "antigens.fasta", "w") as fh:
        for _, r in keep.iterrows():
            fh.write(f">{r.antigen_id}\n{r.antigen_sequence}\n")
    for name, frame in [("antigens_too_long.fasta", skip),
                        ("antigens_nonstandard.fasta", nonstd)]:
        if len(frame):
            with open(OUT / name, "w") as fh:
                for _, r in frame.iterrows():
                    fh.write(f">{r.antigen_id}\n{r.antigen_sequence}\n")

    rows_keep = int(man[man.usable & man.antigen_id.isin(set(keep.antigen_id))].shape[0])
    print(f"  {len(ag)} distinct antigens over {int(man.usable.sum())} usable complexes")
    print(f"  writing {len(keep)} antigens  ({rows_keep} complexes, "
          f"{100 * rows_keep / man.usable.sum():.1f}%)")
    print(f"  skipping {len(skip)} antigens over {a.max_len} aa -> antigens_too_long.fasta")
    print(f"  skipping {len(nonstd)} antigens with non-standard residues "
          f"-> antigens_nonstandard.fasta  (BepiPred exits on the first one, so one "
          f"bad sequence costs the whole run)")
    print(f"  excluded complexes in total: {int(man.usable.sum()) - rows_keep}")
    print(f"  residues to embed: {int(keep.antigen_len.sum()):,}  "
          f"(longest kept {int(keep.antigen_len.max())})")
    print(f"\n  -> {OUT / 'antigens.fasta'}")
    print("  run: the Task I BepiPred-3.0 stage over the chunks written above")


if __name__ == "__main__":
    main()
