#!/usr/bin/env python3
"""
Task II leakage: which of our antibodies did WALLE already see in AsEP TRAIN?

Both arms are computed here, so the whole check is reproducible from the manifest
and so it handles MERGED cases: after stage 00 collapses duplicate (VH, VL)
inputs, one case can carry several PDB ids, and a hit on ANY of them counts.

Two arms, unioned:
  PDB arm       any of the case's PDB ids is an AsEP TRAIN complex.
  sequence arm  some AsEP TRAIN complex has VH identity >= THRESH to ours AND
                carries our antigen. This catches the same antibody redeposited
                under a different id, which is still memorisation -- and it matters
                enormously: it roughly doubled the leaked set over PDB alone.

AsEP defines TWO splits over the same 1,723 complexes (epitope_ratio, ordinary
generalisation; epitope_group, unseen test epitopes), with different train sets, so
every number is computed per split. Checkpoints jumping-sweep-22 and wild-sweep-1
are group-split; the other eight are ratio-split.

    PYTHONPATH=scripts python scripts/walle_leakage.py     (needs the `walle` env: torch)
Writes data/manifests/walle_leakage.csv
"""
import json
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch

from common import cfg, same_antigen

THRESH = 0.95
META = cfg.PROJECT / "work" / "walle" / "asep_meta"


def best_identity(vh, pool):
    best = 0.0
    for _, h in pool:
        if not h:
            continue
        sm = SequenceMatcher(None, vh, h)
        if sm.real_quick_ratio() < THRESH or sm.quick_ratio() < THRESH:
            continue
        best = max(best, sm.ratio())
    return best


def main():
    ref = open(cfg.ANTIGEN_FASTA).read().split("\n", 1)[1].replace("\n", "").strip()
    seqs = json.load(open(META / "asep_seqs.json"))
    ids = [l.strip() for l in open(META / "asepv1-AbDb-IDs.txt") if l.strip()]
    sd = torch.load(META / "split_dict.pt")

    man = pd.read_csv(cfg.MANIFEST)
    # after merging, a case owns every PDB id it was built from
    pdbs = (man.merged_pdb_ids if "merged_pdb_ids" in man.columns
            else man.pdb_id.astype(str))
    case_pdbs = [set(str(p).lower().split(";")) for p in pdbs]
    out = man[["case_id", "epitope_class", "chain_role_ok"]].copy()
    out["pdb_id"] = list(pdbs)
    print(f"{len(man)} cases | {int(man.chain_role_ok.sum())} headline | AsEP {len(ids)} complexes")

    for method in ["epitope_ratio", "epitope_group"]:
        train = [ids[i] for i in sd[method]["train"].tolist()]
        train_pdb = {a.split("_")[0].lower() for a in train}
        pool = [(a, seqs[a]["H"]) for a in train
                if a in seqs and same_antigen(seqs[a].get("ag") or "", ref)]
        tag = "ratio" if method == "epitope_ratio" else "group"
        print(f"\n[{method}] {len(train)} train complexes | {len(pool)} carry our antigen")

        out[f"leaked_{tag}"] = [int(bool(p & train_pdb)) for p in case_pdbs]
        ident = [best_identity(str(v), pool) for v in man.vh_seq]
        out[f"leaked_{tag}_seq"] = [int(b >= THRESH) for b in ident]
        out[f"leaked_{tag}_maxid"] = [round(b, 3) for b in ident]
        out[f"leaked_{tag}_any"] = ((out[f"leaked_{tag}"] == 1) |
                                    (out[f"leaked_{tag}_seq"] == 1)).astype(int)
        h = out[out.chain_role_ok]
        print(f"   PDB arm {int(h[f'leaked_{tag}'].sum()):3d} | sequence arm "
              f"{int(h[f'leaked_{tag}_seq'].sum()):3d} | union "
              f"{int(h[f'leaked_{tag}_any'].sum()):3d} / {len(h)}  "
              f"(clean {len(h) - int(h[f'leaked_{tag}_any'].sum())})")

    path = cfg.MANIFESTS / "walle_leakage.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
