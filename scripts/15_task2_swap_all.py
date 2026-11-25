#!/usr/bin/env python3
"""
Antibody-swap conditioning test for EVERY Task II method, on the same antibodies.

The swap test is the benchmark's core device: score each prediction against its own
antibody's epitope, then against other antibodies' epitopes. gap = own - cross. An
antibody-blind method scores EXACTLY zero -- its prediction does not change, so the
terms cancel algebraically. BepiPred-3.0 and the fixed region masks are in here
precisely to show that zero empirically.

    PYTHONPATH=scripts python scripts/15_task2_swap_all.py
Writes results/task2/swap_all_methods.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "share"))
from swap_test import swap_test                      # noqa: E402
from common import cfg                               # noqa: E402
import walle                                             # noqa: E402


def vecs(d, cases, key="contact_mean"):
    out = {}
    for c in cases:
        p = Path(d) / f"{c}.npz"
        if p.exists():
            z = np.load(p)
            if key in z:
                out[c] = z[key].astype(float)
    return out


def main():
    man = pd.read_csv(cfg.MANIFEST)
    man = man[man.chain_role_ok].reset_index(drop=True)
    z = np.load(cfg.LABELS_NPZ, allow_pickle=True)
    lab = {c: z["labels"][i] for i, c in enumerate(z["case_id"])}
    cases = man.case_id.tolist()

    methods = {}
    # antibody-blind: a fixed region, identical for every antibody
    for name, (lo, hi) in [("RBD mask 319-541", (319, 541)), ("RBM mask 436-507", (436, 507))]:
        v = np.zeros(cfg.ANTIGEN_LEN); v[lo - 1:hi] = 1.0
        methods[name] = {c: v for c in cases}
    methods["BepiPred-3.0"] = vecs(PROJECT / "work/scores_bepipred", cases)
    methods["ESMFold2"] = vecs(PROJECT / "work/scores_esm", cases)
    # contact_top, NOT contact_mean. The headline convention is the TOP-RANKED
    # structure, so that AlphaFold3, WALLE (which reads only the top-ranked model)
    # and ESMFold2 (which produces one structure) are all read off one structure.
    # The ensemble is computed as a separate, clearly named series, so the same
    # label is never used for two different quantities.
    for label, key in [("AlphaFold3", "contact_top"),
                       ("AlphaFold3 (5x5 ensemble)", "contact_mean")]:
        methods[label] = vecs(PROJECT / "work/scores", cases, key=key) or \
            vecs(cfg.SCORES, cases, key=key)
    # WALLE: the project checkpoint (config/walle.py)
    ck = walle.HEADLINE_CHECKPOINT
    methods[f"WALLE ({ck.replace('-sweep','')})"] = vecs(
        PROJECT / f"work/walle/scores/af3/{ck}", cases, key="walle_score")

    rows, per_case_rows = [], []
    for name, d in methods.items():
        common = [c for c in cases if c in d]
        if len(common) < 20:
            print(f"  {name}: only {len(common)} cases, skipping")
            continue
        S = np.vstack([d[c] for c in common])
        L = np.vstack([lab[c] for c in common])
        st, per_case = swap_test(S, L, n_perm=1000, seed=0)
        # Keep the PER-ANTIBODY gaps, not just their mean. The mean alone asserts a
        # uniform effect, and we already know the gap is not uniform (AlphaFold3's
        # win-rate is 0.47 -- it conditions on a subset, not on every antibody).
        # fig3f draws these as distributions.
        per_case = per_case.assign(method=name, case_id=common)
        per_case_rows.append(per_case)
        st["method"] = name
        st["blind"] = name in ("RBD mask 319-541", "RBM mask 436-507", "BepiPred-3.0")
        rows.append(st)
        print(f"  {name:24s} n={st['n_antibodies']:3d} gap {st['swap_gap']:+.4f} "
              f"p={st['p_value']:.4f} win-rate {st['win_rate']:.2f}")

    out = PROJECT / "results" / "task2" / "swap_all_methods.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out_pc = PROJECT / "results" / "task2" / "swap_all_methods_per_case.csv"
    pd.concat(per_case_rows, ignore_index=True).to_csv(out_pc, index=False)
    print(f"\nwrote {out}\nwrote {out_pc}")


if __name__ == "__main__":
    main()
