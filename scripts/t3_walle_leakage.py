#!/usr/bin/env python3
"""
Task III leakage: which of our antibodies did WALLE already see in AsEP TRAIN?

Structurally different from the Task II analysis, and the difference matters:

  Task II  cases ARE solved complexes, so we could match on PDB id first and then
           add sequence matching on top (which nearly doubled the leaked set,
           60 -> 104: PDB matching ALONE badly under-counts).
  Task III cases have NO solved complex -- that is the premise of the task. CATNAP
           and CoV-AbDab give a sequence and an epitope bin, nothing to match a PDB
           id against. So leakage here is necessarily PURE SEQUENCE matching, and
           there is no PDB arm to cross-check it with.

Two strengths are reported, because they answer different questions:

  same-antigen  an AsEP TRAIN complex has VH identity >= THRESH *and* its antigen
                is this arm's antigen. The model saw this antibody bound to this
                antigen, i.e. it could have memorised this epitope. This is the
                strict definition, and matches what Task II reports.
  any-antigen   VH identity >= THRESH against ANY train complex. Weaker: the model
                saw the antibody, but against a different antigen, so it could have
                learned the paratope without the answer. Reported so the strict
                number can be read in context, never as the headline.

    EIB_TASK=task3    PYTHONPATH=scripts python scripts/t3_walle_leakage.py
    EIB_TASK=task3cov PYTHONPATH=scripts python scripts/t3_walle_leakage.py

Writes data/manifests/<task>_walle_leakage.csv (one row per case).
"""
import json
import os
from difflib import SequenceMatcher

import pandas as pd
import torch

from common import antigen_coverage, cfg, same_antigen

THRESH = 0.95          # VH identity to call it "the same antibody"
WALLE_META = cfg.PROJECT / "work" / "walle" / "asep_meta"   # Task II tree holds the AsEP metadata


def best_identity(vh, pool):
    """Highest VH identity against a pool of (id, heavy_seq), with cheap prefilters."""
    best, who = 0.0, ""
    for a, h in pool:
        if not h:
            continue
        sm = SequenceMatcher(None, vh, h)
        # real_quick_ratio is a pure length bound; quick_ratio is a multiset bound.
        # Both are upper bounds on ratio(), so skipping below THRESH is exact.
        if sm.real_quick_ratio() < THRESH or sm.quick_ratio() < THRESH:
            continue
        r = sm.ratio()
        if r > best:
            best, who = r, a
    return best, who


def main():
    task = os.environ.get("EIB_TASK", "task2")
    ref = open(cfg.ANTIGEN_FASTA).read().split("\n", 1)[1].replace("\n", "").strip()
    seqs = json.load(open(WALLE_META / "asep_seqs.json"))
    ids = [l.strip() for l in open(WALLE_META / "asepv1-AbDb-IDs.txt") if l.strip()]
    sd = torch.load(WALLE_META / "split_dict.pt")

    man = pd.read_csv(cfg.MANIFEST)
    out = man[["case_id", "ab_name", "bin"]].copy()
    print(f"{task}: {len(man)} cases | antigen {getattr(cfg, 'ANTIGEN_NAME', '?')} "
          f"({cfg.ANTIGEN_LEN} aa) | AsEP {len(ids)} complexes")

    for method in ["epitope_ratio", "epitope_group"]:
        train_ids = [ids[i] for i in sd[method]["train"].tolist()]
        all_pool = [(a, seqs[a]["H"]) for a in train_ids if a in seqs]
        same_pool = [(a, h) for a, h in all_pool if same_antigen(seqs[a]["ag"], ref)]
        print(f"\n[{method}] {len(train_ids)} train complexes | "
              f"{len(same_pool)} with THIS arm's antigen")

        tag = "ratio" if method == "epitope_ratio" else "group"
        for label, pool in [("same", same_pool), ("any", all_pool)]:
            if not pool:
                out[f"leaked_{tag}_{label}"] = 0
                out[f"maxid_{tag}_{label}"] = 0.0
                print(f"   {label}-antigen : pool empty -> 0 leaked")
                continue
            res = [best_identity(str(vh), pool) for vh in man.vh_seq]
            out[f"leaked_{tag}_{label}"] = [int(b >= THRESH) for b, _ in res]
            out[f"maxid_{tag}_{label}"] = [round(b, 3) for b, _ in res]
            out[f"match_{tag}_{label}"] = [w for _, w in res]
            n = int(out[f"leaked_{tag}_{label}"].sum())
            print(f"   {label}-antigen : {n}/{len(man)} leaked at VH id >= {THRESH} "
                  f"(clean {len(man) - n})")
            if n:
                print(f"      by bin: "
                      f"{out[out[f'leaked_{tag}_{label}'] == 1]['bin'].value_counts().to_dict()}")

    # Only a handful of cases match at all, so print each one with the matched
    # complex's antigen coverage. Grafted-epitope scaffolds (e.g. 10E8 bound to an
    # MPER peptide on a carrier) score LOW coverage yet are genuine leakage, so the
    # same-antigen flag must be audited by eye on this small set, not trusted blind.
    for tag in ["ratio", "group"]:
        col = f"leaked_{tag}_any"
        if col not in out or not out[col].sum():
            continue
        print(f"\n[{tag}] audit of every VH match (id >= {THRESH}):")
        for _, r in out[out[col] == 1].iterrows():
            a = r[f"match_{tag}_any"]
            c = antigen_coverage(seqs[a]["ag"], ref) if a in seqs else 0.0
            verdict = "OUR ANTIGEN" if c >= 0.30 else ("grafted/other? CHECK" if c > 0.02 else "different antigen")
            print(f"   {r.ab_name:16s} {r['bin']:6s} id={r[f'maxid_{tag}_any']:.3f} "
                  f"train={a:10s} ag_cov={c:.3f}  {verdict}")

    path = cfg.MANIFESTS / f"{task}_walle_leakage.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    strict = [c for c in out.columns if c.startswith("leaked_") and c.endswith("_same")]
    if strict and out[strict].to_numpy().sum() == 0:
        print("\nNo same-antigen leakage under either split: every Task III case is\n"
              "clean for WALLE, so the whole set IS the clean set and no separate\n"
              "clean/whole split is needed for this arm.")


if __name__ == "__main__":
    main()
