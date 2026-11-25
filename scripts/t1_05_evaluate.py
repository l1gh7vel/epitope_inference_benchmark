#!/usr/bin/env python3
"""
Task I evaluation: per-fold performance, because a pooled number hides the point.

What Task I asks. Tasks II and III hold the antigen fixed and vary the antibody.
Task I sweeps all of SAbDab and asks whether a method's accuracy is a property of
the method or of which antigens it happened to be shown. A single pooled AUPRC
cannot answer that; a per-fold distribution can, which is why the headline is
violins by fold rather than one bar.

Fold 1 is essentially one antigen. The folds were balanced by complex count (917
each) rather than by antigen diversity, and antigen-identity clustering swept
nearly every SARS-CoV-2 spike complex into fold 1: 29 antigen clusters against
~280 in each of the others. So fold 1 behaving differently is a fact about SAbDab
rather than about the method being scored. Any claim of "unstable across folds"
has to survive dropping fold 1.

The floor is per case rather than global. Every antigen has a different length and
a different epitope size, so the random-AUPRC floor (= positive rate) varies case
by case, from under 1% to over 10%, and a single global floor line would be wrong
for nearly every case. Two things are therefore reported: raw AUPRC, and AUPRC
divided by that case's own floor (enrichment over random), which is the one that
can be compared across antigens.

Also reported: a within-antigen swap test. Task I is the only task with many
antigens carrying several antibodies, so "does the prediction move with the
antibody" can be asked while holding the antigen exactly fixed, which is the
cleanest form of the conditioning test in the benchmark. It is defined only for
antigens with two or more antibodies whose epitopes differ.

    PYTHONPATH=scripts:config python scripts/t1_05_evaluate.py \
        --scores_dir work/task1/scores_bepipred --method BepiPred-3.0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                        # noqa: E402


def rank_biserial(y, s):
    """Prevalence-free: P(positive ranks above negative), rescaled to [-1, 1]."""
    y = np.asarray(y, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    auc = (r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return 2 * auc - 1


def load_score(path):
    """One score vector from an npz, whichever key wrote it.

    The adapters write "score"; stage 04 writes AlphaFold3's own keys; stage 08
    writes WALLE's. contact_top is preferred over contact_mean because the
    top-ranked structure is the headline convention everywhere in this project.
    The key choice lives here, so that "score" is not hardcoded in one reader and
    not another.

    WALLE is read straight from its own output tree rather than through an adapter,
    so there is no second copy of a 500-case score set to drift:
        --scores_dir work/task1/walle/scores/experimental/decent-sweep-2
    """
    z = np.load(path)
    key = next((k for k in ("score", "contact_top", "contact_mean", "walle_score")
                if k in z), None)
    return None if key is None else z[key].astype(float)


def load_labels():
    z = np.load(cfg.LABELS_NPZ, allow_pickle=True)
    return {c: z[c] for c in z.files if c != "case_ids"}


def per_case(man, labels, scores_dir):
    rows = []
    for _, r in man.iterrows():
        f = scores_dir / f"{r.case_id}.npz"
        if not f.exists():
            continue
        s = load_score(f)
        if s is None:
            continue
        y = labels.get(r.case_id)
        if y is None or len(y) != len(s) or y.sum() == 0 or y.sum() == len(y):
            continue
        floor = float(y.mean())
        ap = float(average_precision_score(y, s))
        rows.append({"case_id": r.case_id, "fold": int(r.fold),
                     "antigen_id": r.antigen_id, "cluster": int(r.Cluster),
                     "antigen_len": int(len(y)), "n_epitope": int(y.sum()),
                     "auprc": ap, "floor": floor, "enrichment": ap / floor,
                     "rankbis": rank_biserial(y, s),
                     })
    return pd.DataFrame(rows)


def within_antigen_swap(man, labels, scores_dir, pc):
    """Own-epitope AUPRC minus the same prediction scored against a different
    antibody's epitope on the same antigen.

    Holding the antigen fixed removes every confound the Task II version still
    carries (antigen length, epitope size, fold quality). For an antibody-blind
    method this is exactly zero by the same algebra as Task II: the score vector
    does not change, so own and cross cancel over the group.
    """
    keep = set(pc.case_id)
    rows = []
    for ag, g in man[man.case_id.isin(keep)].groupby("antigen_id"):
        if len(g) < 2:
            continue
        cids = list(g.case_id)
        ys = {c: labels[c] for c in cids}
        ss = {c: load_score(scores_dir / f"{c}.npz") for c in cids}
        for c in cids:
            # exclude antibodies sharing this exact epitope: they are the same
            # question, and scoring against them would understate the gap
            others = [o for o in cids if o != c and not np.array_equal(ys[o], ys[c])]
            if not others:
                continue
            own = average_precision_score(ys[c], ss[c])
            cross = np.mean([average_precision_score(ys[o], ss[c]) for o in others])
            rows.append({"case_id": c, "antigen_id": ag, "n_others": len(others),
                         "own": float(own), "cross": float(cross),
                         "gap": float(own - cross)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--keep_shared_antibodies", action="store_true",
                    help="keep antibodies whose heavy chain appears in more than one "
                         "fold. OFF by default: the folds are antigen-disjoint but not "
                         "antibody-disjoint, and a shared antibody correlates two folds "
                         "that the per-fold comparison treats as independent.")
    ap.add_argument("--arm", default="full", choices=["full", "intersection"],
                    help="full: every eligible case this method scored (per-method "
                         "coverage). intersection: the cases every method in "
                         "cfg.INTERSECTION_METHODS has a full-arm score for; the "
                         "cross-method comparison uses this arm. Run the full arm for "
                         "all methods first, since the intersection is read from those "
                         "tables.")
    a = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    man = man[man.usable]
    # The antigen length cap is part of the evaluated set for every method and
    # both arms. AF3 and ESMFold2 are only ever given length-eligible cases to
    # fold; BepiPred-3.0 and WALLE score every case and would otherwise enter the
    # table with antigens outside the evaluated set.
    man = man[man.antigen_len <= cfg.MAX_ANTIGEN_LEN]
    if a.arm == "intersection":
        keep = None
        for m in cfg.INTERSECTION_METHODS:
            ids = set(pd.read_csv(cfg.RESULTS / f"per_case_{m.replace(' ', '_')}_full.csv").case_id)
            keep = ids if keep is None else keep & ids
        man = man[man.case_id.isin(keep)]
    if not a.keep_shared_antibodies and "vh_dedup_keep" in man:
        n_before = len(man)
        man = man[man.vh_dedup_keep]
        print(f"  antibody-disjoint filter: dropped {n_before - len(man)} of {n_before} "
              f"rows whose heavy chain also appears in another fold")
    labels = load_labels()
    sdir = Path(a.scores_dir)
    n_eligible = len(man)
    # Counted here rather than inside per_case(), so the summary can say how many
    # eligible cases had no score file at all.
    n_missing_score = int(sum(not (sdir / f"{c}.npz").exists() for c in man.case_id))

    pc = per_case(man, labels, sdir)
    if pc.empty:
        sys.exit(f"no scored cases found in {sdir}")
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{a.method.replace(' ', '_').replace('/', '_')}_{a.arm}"
    pc.to_csv(cfg.RESULTS / f"per_case_{tag}.csv", index=False)

    print(f"\n{a.method}  ({a.arm} arm)")
    print(f"  scored {len(pc)} of {n_eligible} eligible cases "
          f"({100 * len(pc) / n_eligible:.1f}%)")
    print(f"    no score file: {n_missing_score}   other skips "
          f"(no score key, label mismatch, single class): "
          f"{n_eligible - n_missing_score - len(pc)}")
    print(f"  AUPRC        mean {pc.auprc.mean():.4f}  median {pc.auprc.median():.4f}")
    print(f"  random floor mean {pc.floor.mean():.4f}   <- varies per antigen, so "
          f"compare the ENRICHMENT")
    print(f"  enrichment   mean {pc.enrichment.mean():.2f}x median "
          f"{pc.enrichment.median():.2f}x")
    print(f"  rank-biserial mean {pc.rankbis.mean():+.3f}")
    print("\n  per fold (fold 1 is ~one antigen -- see module docstring):")
    g = pc.groupby("fold").agg(n=("case_id", "size"), clusters=("cluster", "nunique"),
                               auprc=("auprc", "median"), floor=("floor", "median"),
                               enrich=("enrichment", "median"))
    print(g.round(4).to_string())

    swap = within_antigen_swap(man, labels, sdir, pc)
    # Record where the scores came from. Without it the per-case table carries no
    # record of which checkpoint produced it, so a change of headline checkpoint
    # could leave Task I behind undetected. fig2c asserts this against the label
    # it draws.
    summary = {"method": a.method, "arm": a.arm, "n": len(pc),
               "n_eligible": n_eligible,
               "n_skipped_missing_score": n_missing_score,
               "scores_dir": str(sdir), "checkpoint": sdir.name,
               "auprc_mean": float(pc.auprc.mean()),
               "auprc_median": float(pc.auprc.median()),
               "floor_mean": float(pc.floor.mean()),
               "enrichment_mean": float(pc.enrichment.mean()),
               "enrichment_median": float(pc.enrichment.median()),
               "rankbis_mean": float(pc.rankbis.mean()),
               "per_fold": {int(k): {kk: float(vv) for kk, vv in v.items()}
                            for k, v in g.to_dict(orient="index").items()}}
    if not swap.empty:
        swap.to_csv(cfg.RESULTS / f"swap_{tag}.csv", index=False)
        summary["swap_gap"] = float(swap.gap.mean())
        summary["swap_n"] = int(len(swap))
        print(f"\n  within-antigen swap gap: {swap.gap.mean():+.4f} "
              f"over {len(swap)} antibodies on {swap.antigen_id.nunique()} antigens")
        print("    (exactly 0 for an antibody-blind method, by the same algebra as "
              "Task II)")
    (cfg.RESULTS / f"metrics_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  -> {cfg.RESULTS}/per_case_{tag}.csv, metrics_{tag}.json")


if __name__ == "__main__":
    main()
