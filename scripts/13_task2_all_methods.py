#!/usr/bin/env python3
"""
One tidy per-antibody table for EVERY Task II method, so figures never recompute.

Emits results/task2/all_methods_per_case.csv with one row per (antibody, method)
and its AUPRC against that antibody's gold epitope. Every method is scored the SAME
way on the SAME antibodies: same gold labels, same 1,273-position frame, same AUPRC.

    PYTHONPATH=scripts python scripts/13_task2_all_methods.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import cfg
import walle

PROJECT = Path(__file__).resolve().parents[1]


def auprc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def from_dir(d, case_ids, key="contact_mean"):
    out = {}
    for cid in case_ids:
        p = Path(d) / f"{cid}.npz"
        if p.exists():
            z = np.load(p)
            if key in z:
                out[cid] = z[key].astype(float)
    return out


def main():
    man = pd.read_csv(cfg.MANIFEST)
    man = man[man.chain_role_ok].reset_index(drop=True)
    z = np.load(cfg.LABELS_NPZ, allow_pickle=True)
    lab = {c: z["labels"][i] for i, c in enumerate(z["case_id"])}
    cases = man.case_id.tolist()
    group = dict(zip(man.case_id, man.epitope_class))

    pc = pd.read_csv(PROJECT / "results/task2/full/per_case_metrics.csv").set_index("case_id")
    rows = []

    # --- antibody-blind baselines: already per-case in the stage-05 table ---
    for meth, col in [("Random\n(AUPRC floor)", "pos_rate"),
                      ("RBD mask 319-541", "rbd_mask_auprc"),
                      ("RBM mask 436-507", "rbm_mask_auprc")]:
        for cid in cases:
            if cid in pc.index:
                rows.append({"case_id": cid, "epitope_class": group[cid],
                             "method": meth, "auprc": float(pc.loc[cid, col])})

    # --- BepiPred-3.0: antibody-blind, ONE identical vector reused per antibody ---
    for cid, s in from_dir(PROJECT / "work/scores_bepipred", cases).items():
        rows.append({"case_id": cid, "epitope_class": group[cid],
                     "method": "BepiPred-3.0", "auprc": auprc(lab[cid], s)})

    # --- ESMFold2: one predicted structure per complex ---
    for cid, s in from_dir(PROJECT / "work/scores_esm", cases).items():
        rows.append({"case_id": cid, "epitope_class": group[cid],
                     "method": "ESMFold2", "auprc": auprc(lab[cid], s)})

    # --- WALLE: the project checkpoint, the same on every task (config/walle.py) ---
    ck = walle.HEADLINE_CHECKPOINT
    wp_all = pd.read_csv(PROJECT / "results/task2/walle/af3/per_case.csv")
    series = [(ck, f"WALLE ({ck.replace('-sweep','')})")]
    for ckpt, label in series:
        wp = wp_all[(wp_all.method == "WALLE") & (wp_all.checkpoint == ckpt)]
        for _, r in wp.iterrows():
            if r.case_id in group:
                rows.append({"case_id": r.case_id, "epitope_class": group[r.case_id],
                             "method": label, "auprc": float(r.auprc)})

    # --- AlphaFold3 ---
    # HEADLINE is the TOP-RANKED structure, not the 25-sample ensemble, so that AF3
    # sits on the same convention as WALLE (which reads the top-ranked model) and
    # ESMFold2 (which produces one structure). The ensemble is a series of its own,
    # drawn as a marker, and applies wherever an ensemble readout is compared.
    for cid in cases:
        if cid in pc.index:
            rows.append({"case_id": cid, "epitope_class": group[cid],
                         "method": "AlphaFold3", "auprc": float(pc.loc[cid, "contact_top_auprc"])})
            rows.append({"case_id": cid, "epitope_class": group[cid],
                         "method": "AlphaFold3 (5x5 ensemble)",
                         "auprc": float(pc.loc[cid, "contact_mean_auprc"])})

    d = pd.DataFrame(rows).dropna(subset=["auprc"])

    # BepiPred-3.0 and ESMFold2 are read from work/, which holds the per-residue
    # score arrays and is not part of a published clone. from_dir returns an
    # empty mapping when it is absent, so without this the table would be
    # written with those two methods missing and nothing would say so. Every
    # method the panels read has to be present, or the existing table stands.
    want = {"Random\n(AUPRC floor)", "RBD mask 319-541", "RBM mask 436-507",
            "BepiPred-3.0", "ESMFold2", f"WALLE ({ck.replace('-sweep','')})",
            "AlphaFold3", "AlphaFold3 (5x5 ensemble)"}
    absent = sorted(want - set(d.method))
    if absent:
        raise SystemExit(
            f"missing per-case scores for {', '.join(absent)}; these read the "
            f"per-residue arrays under work/, which a published clone does not "
            f"carry. Output unchanged.")

    out = PROJECT / "results" / "task2" / "all_methods_per_case.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"wrote {out}")
    print(d.groupby("method").agg(n=("auprc", "size"), mean=("auprc", "mean"),
                                  median=("auprc", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
