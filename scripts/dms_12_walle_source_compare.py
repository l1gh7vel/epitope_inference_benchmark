#!/usr/bin/env python3
"""WALLE on the DMS panel: AF3-predicted input vs DEPOSITED-complex input.

Scores the two sources with the SAME metrics stage 05/09 use (AUPRC, precision@L,
rank-biserial against the gold structural epitope), on THREE residue pools, because
the pools are not the same, and the residue pool can reverse the
ranking:

  native   -- each source on its own surface mask. NOT a fair head-to-head; it is
              here only to show how far apart the pools are.
  matched  -- the intersection of the two surface masks, i.e. residues BOTH inputs
              resolve AND WALLE scores in both. This is the fair comparison.
  ref856   -- the whole 856-position HXB2 frame with unscored residues at 0, the
              `full` convention the Fig 4 escape readout actually ranks on. This is
              where a construct that covers 3% of the frame is at its most
              misleading, which is the point of printing it.

Mean and median are reported side by side and never reconciled. The paired test
is a one-sided Wilcoxon signed-rank, asking only whether the experimental input
is better.

    EIB_ROOT=<root> python scripts/dms_12_walle_source_compare.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import average_precision_score

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "config"))
import dms as cfg                                           # noqa: E402

MAN = PROJECT / "data" / "manifests"
WALLE_SCORES = cfg.WORK / "walle" / "scores"
OUT = PROJECT / "results" / "dms_case" / "walle_source"


def auprc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def p_at_l(y, s):
    L = int(y.sum())
    if L == 0:
        return np.nan
    return float(y[np.argsort(s)[::-1][:L]].sum()) / L


def rank_biserial(y, s):
    x, z = s[y == 1], s[y == 0]
    if not len(x) or not len(z):
        return np.nan
    u = mannwhitneyu(x, z, alternative="two-sided").statistic
    return float(2 * (u / (len(x) * len(z))) - 1)


def load(src, ckpt, cid, L):
    p = WALLE_SCORES / src / ckpt / f"{cid}.npz"
    if not p.exists():
        return None, None
    z = np.load(p)
    return z["walle_score"].astype(float), z["surf_mask"].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="treasured-sweep-8")
    ap.add_argument("--all_checkpoints", action="store_true")
    a = ap.parse_args()

    L = cfg.ANTIGEN_LEN
    man = pd.read_csv(cfg.MANIFEST)
    lab = np.load(MAN / "dms_labels.npz")
    rep = json.loads((MAN / "dms_gold_report.json").read_text())
    af3_ok = sorted(d.name for d in (WALLE_SCORES / "af3").iterdir() if d.is_dir())
    ckpts = af3_ok if a.all_checkpoints else [a.checkpoint]

    rows = []
    for ckpt in ckpts:
        for r in man.itertuples():
            cid, ab = r.case_id, r.antibody
            a_s, a_m = load("af3", ckpt, cid, L)
            e_s, e_m = load("experimental", ckpt, cid, L)
            if a_s is None or e_s is None:
                continue
            y = lab[f"{ab}__gold"].astype(bool)
            pools = {"native_af3": a_m, "native_exp": e_m,
                     "matched": a_m & e_m, "ref856": np.ones(L, bool)}
            for pool_name, pool in pools.items():
                for src, s in (("af3", a_s), ("experimental", e_s)):
                    if pool_name == "native_af3" and src != "af3":
                        continue
                    if pool_name == "native_exp" and src != "experimental":
                        continue
                    yy, ss = y[pool], s[pool]
                    if yy.sum() == 0 or yy.sum() == len(yy):
                        rows.append(dict(checkpoint=ckpt, case_id=cid, antibody=ab,
                                         arm=r.arm, source=src, pool=pool_name,
                                         n_pool=int(pool.sum()), n_gold=int(yy.sum()),
                                         auprc=np.nan, rankbis=np.nan, p_at_L=np.nan))
                        continue
                    rows.append(dict(checkpoint=ckpt, case_id=cid, antibody=ab,
                                     arm=r.arm, source=src, pool=pool_name,
                                     n_pool=int(pool.sum()), n_gold=int(yy.sum()),
                                     auprc=auprc(yy, ss), rankbis=rank_biserial(yy, ss),
                                     p_at_L=p_at_l(yy, ss)))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_case.csv", index=False)

    # -------- summary: mean AND median, per pool / arm / source, with n ----------
    summ = (df.groupby(["checkpoint", "pool", "arm", "source"])
              .agg(n=("auprc", "size"),
                   n_pool_mean=("n_pool", "mean"),
                   auprc_mean=("auprc", "mean"), auprc_median=("auprc", "median"),
                   rankbis_mean=("rankbis", "mean"), rankbis_median=("rankbis", "median"),
                   p_at_L_mean=("p_at_L", "mean"), p_at_L_median=("p_at_L", "median"))
              .reset_index())
    summ.to_csv(OUT / "summary.csv", index=False)

    # -------- one-sided paired tests on the matched pool -------------------------
    tests = []
    for ckpt in ckpts:
        for pool in ["matched", "ref856"]:
            for arm in ["vhvl", "vh"]:
                sub = df[(df.checkpoint == ckpt) & (df.pool == pool) & (df.arm == arm)]
                piv = sub.pivot_table(index="antibody", columns="source", values="auprc")
                piv = piv.dropna()
                if len(piv) < 3:
                    continue
                for metric in ["auprc", "rankbis", "p_at_L"]:
                    pv = sub.pivot_table(index="antibody", columns="source",
                                         values=metric).dropna()
                    d = pv["experimental"] - pv["af3"]
                    if (d == 0).all():
                        p = np.nan
                    else:
                        p = float(wilcoxon(pv["experimental"], pv["af3"],
                                           alternative="greater").pvalue)
                    tests.append(dict(checkpoint=ckpt, pool=pool, arm=arm,
                                      metric=metric, n=len(pv),
                                      exp_mean=float(pv["experimental"].mean()),
                                      af3_mean=float(pv["af3"].mean()),
                                      exp_median=float(pv["experimental"].median()),
                                      af3_median=float(pv["af3"].median()),
                                      n_exp_better=int((d > 0).sum()),
                                      p_one_sided_exp_better=p))
    tt = pd.DataFrame(tests)
    tt.to_csv(OUT / "paired_tests.csv", index=False)

    pd.set_option("display.width", 250)
    for ckpt in ckpts:
        print(f"\n=== {ckpt} ===")
        s = summ[summ.checkpoint == ckpt]
        print(s.drop(columns="checkpoint").round(3).to_string(index=False))
        print("\none-sided Wilcoxon, H1: experimental > af3")
        print(tt[tt.checkpoint == ckpt].drop(columns="checkpoint").round(4).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
