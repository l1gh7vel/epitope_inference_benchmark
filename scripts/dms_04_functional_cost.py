"""Functional cost of mispredicting an epitope.

An epitope prediction is usually scored against a structural gold standard, which
treats every contact residue as equally worth getting right. This asks a different
question: do the residues a model ranks highest carry ESCAPE POTENTIAL -- the
mutations that actually cost the antibody its neutralisation?

Construction, for one (antibody, escape dataset, method):

  * Rank all 856 Env positions by the method's score.
  * For each k, take the top k and run Mann-Whitney U of escape at those positions
    against escape at every other MEASURED position. Report -log10 p and the
    rank-biserial correlation.
  * The GOLD EPITOPE gives the same two numbers, and is drawn as a horizontal
    reference. It is the value a structurally perfect prediction would earn, so
    the vertical gap between a method's curve and that line IS the functional cost
    of its errors.

Two statistics, deliberately, because they answer different questions:
  -log10 p  is what was asked for, but it depends on k through statistical power,
            so a curve rising with k can rise for free.
  rank-biserial is the effect size (probability a top-k site out-escapes a
            non-top-k one, rescaled to [-1,1]). It does NOT depend on n, so it is
            the curve to read across k and across antibodies.
Both are reported for both tails: one-sided greater (the directional question)
and two-sided (which covers an answer that can come out negative).

The random floor is simulated: N_RANDOM random rankings per k. For a top-k drawn
at random the expected -log10 p is above zero (the median of -log10 U is
log10 2 = 0.30), so an eyeballed "near zero" baseline would be wrong.

Unmeasured sites are excluded from the test, never imputed as zero.
"""
import argparse, json, sys
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "config"))
import dms as cfg                                           # noqa: E402

MAN = PROJECT / "data" / "manifests"
RES = PROJECT / "results" / "dms_case"
KS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400]
N_RANDOM = 400
STAT = "mean"

# subdir of cfg.WORK -> npz key. WALLE is read STRAIGHT FROM ITS OWN TREE
# (work/dms/walle/scores/<source>/<checkpoint>/) rather than from a copied-in
# scores_walle/, so there is exactly one artefact per case and no second copy to
# go stale. `af3` is the structure source -- the same AF3 folds AlphaFold3 and
# ESMFold2 are scored from, so the three see the same input; `decent-sweep-2` is
# the project-wide headline checkpoint (BLOSUM62 node features), the one Tasks I
# and II quote. Other checkpoints sit beside it and can be swapped in here.
WALLE_SRC, WALLE_CKPT = "af3", "treasured-sweep-8"   # headline since 2026-11-02
METHOD_DIRS = {"AlphaFold3": ("scores", "contact_top"),
               "ESMFold2": ("scores_esm", "contact_top"),
               "WALLE": (f"walle/scores/{WALLE_SRC}/{WALLE_CKPT}", "walle_score")}


def mwu(values, sel, measured):
    """MWU of `values` at sel vs not-sel, over measured sites only."""
    a = values[sel & measured]
    b = values[(~sel) & measured]
    if len(a) < 3 or len(b) < 3:
        return dict(n=len(a), rb=np.nan, p_gt=np.nan, p_2s=np.nan)
    U, p1 = mannwhitneyu(a, b, alternative="greater")
    _, p2 = mannwhitneyu(a, b, alternative="two-sided")
    return dict(n=len(a), rb=2 * U / (len(a) * len(b)) - 1, p_gt=p1, p_2s=p2)


def topk_mask(score, k, L):
    m = np.zeros(L, bool)
    # ties broken by position so the mask is deterministic and exactly size k
    m[np.lexsort((np.arange(L), -score))[:k]] = True
    return m


def load_method_scores(case_id, L):
    out = {}
    for name, (sub, key) in METHOD_DIRS.items():
        p = cfg.WORK / sub / f"{case_id}.npz"
        if p.exists():
            d = np.load(p)
            k = key if key in d else ("score" if "score" in d else None)
            if k and len(d[k]) == L:
                out[name] = np.asarray(d[k], float)
    bp = cfg.WORK / "bepipred_env.npy"
    if bp.exists():
        out["BepiPred-3.0"] = np.load(bp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="vhvl", choices=["vhvl", "vh"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    esc = np.load(MAN / "dms_escape.npz")
    lab = np.load(MAN / "dms_labels.npz")
    gold_rep = json.loads((MAN / "dms_gold_report.json").read_text())
    man = pd.read_csv(cfg.MANIFEST)
    man = man[man.arm == a.arm]
    L = cfg.ANTIGEN_LEN
    rng = np.random.default_rng(a.seed)
    datasets = sorted({k.split("|")[1] for k in esc.files})

    curve_rows, gold_rows, missing = [], [], []
    for r in man.itertuples():
        ab = r.antibody
        gold = lab[f"{ab}__gold"].astype(bool)
        scores = load_method_scores(r.case_id, L)
        if not scores:
            missing.append(r.case_id)
        for ds in datasets:
            key = f"{ab}|{ds}|{STAT}"
            if key not in esc.files:
                continue
            v = esc[key]; measured = ~np.isnan(v)
            g = mwu(v, gold, measured)
            gold_rows.append(dict(antibody=ab, arm=a.arm, dataset=ds,
                                  epitope_class=gold_rep[ab]["epitope_class"],
                                  n_gold=g["n"], rb=g["rb"],
                                  neglog10_p_greater=-np.log10(g["p_gt"]),
                                  neglog10_p_two_sided=-np.log10(g["p_2s"])))
            for name, sc in scores.items():
                for k in KS:
                    m = topk_mask(sc, k, L)
                    s = mwu(v, m, measured)
                    curve_rows.append(dict(antibody=ab, arm=a.arm, dataset=ds,
                        epitope_class=gold_rep[ab]["epitope_class"], method=name,
                        k=k, n_measured_in_topk=s["n"], rb=s["rb"],
                        neglog10_p_greater=-np.log10(s["p_gt"]),
                        neglog10_p_two_sided=-np.log10(s["p_2s"]),
                        overlap_with_gold=int((m & gold).sum()),
                        recall_of_gold=float((m & gold).sum() / max(gold.sum(), 1))))
            # simulated random floor, same k grid, same measured mask
            for k in KS:
                rb, p1, p2 = [], [], []
                for _ in range(N_RANDOM):
                    m = np.zeros(L, bool); m[rng.choice(L, k, replace=False)] = True
                    s = mwu(v, m, measured)
                    rb.append(s["rb"]); p1.append(s["p_gt"]); p2.append(s["p_2s"])
                curve_rows.append(dict(antibody=ab, arm=a.arm, dataset=ds,
                    epitope_class=gold_rep[ab]["epitope_class"], method="Random",
                    k=k, n_measured_in_topk=np.nan, rb=float(np.nanmean(rb)),
                    neglog10_p_greater=float(np.nanmean(-np.log10(p1))),
                    neglog10_p_two_sided=float(np.nanmean(-np.log10(p2))),
                    overlap_with_gold=np.nan, recall_of_gold=np.nan))

    RES.mkdir(parents=True, exist_ok=True)
    cur = pd.DataFrame(curve_rows); gol = pd.DataFrame(gold_rows)
    cur.to_csv(RES / f"functional_cost_curves_{a.arm}.csv", index=False)
    gol.to_csv(RES / f"functional_cost_gold_{a.arm}.csv", index=False)
    print(f"arm={a.arm}: {cur.method.nunique()} methods "
          f"{sorted(cur.method.unique())}, {gol.antibody.nunique()} antibodies, "
          f"{cur.dataset.nunique()} datasets -> {RES}")
    if missing:
        print(f"no scores yet for {len(missing)} case(s): {missing[:6]}")
    if len(cur):
        piv = (cur[cur.k == 25].pivot_table(index=["antibody", "dataset"],
                                            columns="method", values="rb"))
        print("\nrank-biserial at k=25:")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
