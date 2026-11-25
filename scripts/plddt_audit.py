#!/usr/bin/env python3
"""
Input-quality control: per-chain and per-bin pLDDT of the AF3 structures.

Why this matters, and why it is not the same question as iPTM:

  * pLDDT  = per-residue confidence in the LOCAL FOLD. This is what WALLE consumes,
             because WALLE is pose-independent (it reads the antibody and antigen as
             two separate structures and never sees inter-chain geometry). If pLDDT
             were poor, a WALLE result would be input-limited rather than a model
             property.
  * iPTM   = confidence in the INTERFACE POSE. Irrelevant to WALLE; decisive for
             AF3's own contact-based epitope prediction.

The load-bearing use is per-BIN pLDDT on Task III. A bin that AF3 fails to predict
AND folds badly is ambiguous -- bad input or bad model? A bin it fails to predict
while folding it CONFIDENTLY is an unambiguous model failure. Run this before
attributing any per-bin failure to the model.

AF3 writes per-atom pLDDT into the mmCIF B-factor column.

    PYTHONPATH=scripts python scripts/plddt_audit.py [--n 40] [--full]
"""
import argparse
import glob
import json
import importlib
import random
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))

DATASETS = [("Task II (the structures WALLE consumed)", "task2", "work/af3_out"),
            ("Task III HIV Env", "task3", "work/task3/af3_out"),
            ("Task III SARS-CoV-2 spike", "task3cov", "work/task3cov/af3_out")]


def per_chain_plddt(cif):
    import gemmi
    st = gemmi.read_structure(cif)
    st.setup_entities()
    return {ch.name: np.array([np.mean([a.b_iso for a in r]) for r in ch])
            for ch in st[0] if len(ch)}


def rep_model(case_dir):
    f = sorted(glob.glob(case_dir + "*seed-1_sample-0*model.cif"))
    return (f or sorted(glob.glob(case_dir + "*model.cif")) or [None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="cases sampled per dataset")
    ap.add_argument("--full", action="store_true", help="use every case (slow)")
    ap.add_argument("--json", default=None, help="also write the per-bin table here")
    a = ap.parse_args()
    random.seed(0)
    out_json = {}

    for label, task, root in DATASETS:
        cases = sorted(glob.glob(str(PROJECT / root) + "/*/"))
        if not cases:
            print(f"\n{label}: no AF3 outputs at {root}")
            continue
        sample = cases if a.full else random.sample(cases, min(a.n, len(cases)))
        cfg = importlib.import_module(task)

        chains, ag = {}, []
        for c in sample:
            f = rep_model(c)
            if not f:
                continue
            d = per_chain_plddt(f)
            for k, v in d.items():
                chains.setdefault(k, []).append(v.mean())
            if "A" in d and len(d["A"]) == cfg.ANTIGEN_LEN:
                ag.append(d["A"])

        print(f"\n=== {label} -- {len(sample)} of {len(cases)} cases ===")
        for k in sorted(chains):
            v = np.array(chains[k])
            print(f"  chain {k}: mean pLDDT {v.mean():5.1f}  (case range {v.min():.0f}-{v.max():.0f})")

        # Per-bin pLDDT is only defined for the Task III arms, which have bins.
        if ag and getattr(cfg, "BIN_RANGES", None):
            M = np.vstack(ag)
            print(f"  antigen pLDDT by bin (n={len(ag)} structures):")
            print(f"    {'bin':8s} {'mean':>6s} {'%>70':>7s}   interpretation")
            out_json.setdefault(task, {"antigen": getattr(cfg, "ANTIGEN_NAME", task),
                                       "n_structures": len(ag), "bins": {}})
            for b, spans in cfg.BIN_RANGES.items():
                m = np.zeros(cfg.ANTIGEN_LEN, bool)
                for lo, hi in spans:
                    m[lo - 1:hi] = True
                sub = M[:, m]
                conf = 100 * (sub > 70).mean()
                out_json[task]["bins"][b] = {"mean_plddt": float(sub.mean()),
                                             "pct_above_70": float(conf),
                                             "per_structure_mean": [float(x) for x in sub.mean(1)]}
                note = ("well folded -- a failure here is the MODEL" if conf > 70 else
                        "borderline" if conf > 40 else
                        "POORLY folded -- a failure here is partly the INPUT")
                print(f"    {b:8s} {sub.mean():6.1f} {conf:6.1f}%   {note}")
            print(f"    {'ALL':8s} {M.mean():6.1f} {100*(M>70).mean():6.1f}%")

    if a.json:
        Path(a.json).write_text(json.dumps(out_json, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    print("\npLDDT scale: >90 very high | 70-90 confident | 50-70 low | <50 disordered")


if __name__ == "__main__":
    main()
