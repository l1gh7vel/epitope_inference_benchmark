#!/usr/bin/env python3
"""Evaluate Task III residue scores as antigen-domain predictions.

The mean residue score is computed within each configured domain and the
highest-scoring domain is the prediction. Outputs include accuracy, kappa,
macro recall, per-domain recall, per-case rank-biserial and label-permutation
tests. Every method is evaluated on the configured 300-case cohort.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, matthews_corrcoef

from common import cfg


def bin_masks():
    out = {}
    for b, ranges in cfg.BIN_RANGES.items():
        m = np.zeros(cfg.ANTIGEN_LEN, bool)
        for lo, hi in ranges:
            m[lo - 1:hi] = True
        out[b] = m
    return out


def auprc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def p_at_l(y, s):
    """Fraction of the top-L residues inside the domain, L = domain size."""
    L = int(y.sum())
    if L == 0:
        return np.nan
    return float(y[np.argsort(s)[::-1][:L]].sum()) / L


def rank_biserial(y, s):
    """2*AUC-1 via Mann-Whitney U. 0 = no localisation, <0 = wrong region."""
    x, z = s[y == 1], s[y == 0]
    if not len(x) or not len(z):
        return np.nan
    u = mannwhitneyu(x, z, alternative="two-sided").statistic
    return float(2 * (u / (len(x) * len(z))) - 1)


def load_scores(case_ids, variant="contact_mean", scores_dir=None):
    scores_dir = scores_dir or cfg.SCORES
    scores, have = [], np.zeros(len(case_ids), bool)
    for i, cid in enumerate(case_ids):
        p = scores_dir / f"{cid}.npz"
        if p.exists():
            z = np.load(p)
            scores.append(z[variant].astype(float) if variant in z
                          else np.zeros(cfg.ANTIGEN_LEN))
            have[i] = variant in z
        else:
            scores.append(np.zeros(cfg.ANTIGEN_LEN))
    return np.vstack(scores), have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="contact_top",
                    help="array key inside each .npz (AF3: contact_top; WALLE: walle_score)")
    ap.add_argument("--scores_dir", default=None,
                    help="directory of per-case .npz (default: cfg.SCORES, i.e. AF3)")
    ap.add_argument("--out_dir", default=None,
                    help="where to write results (default: cfg.RESULTS)")
    ap.add_argument("--walle", default=None, metavar="CHECKPOINT",
                    help="shorthand: score WALLE instead of AF3. Sets --scores_dir to "
                         "<WORK>/walle/scores/<source>/<CHECKPOINT>, --variant walle_score, "
                         "and --out_dir <RESULTS>/walle/<CHECKPOINT>.")
    ap.add_argument("--method", default=None,
                    help="label written to the summary, metrics.json and figure titles. "
                         "Required unless the scores are the AF3 default or --walle is set.")
    args = ap.parse_args()

    # WALLE writes elsewhere and under a different array key than AF3; this keeps
    # the two from ever overwriting each other's results directory.
    if args.walle:
        args.scores_dir = str(cfg.WORK / "walle" / "scores" / "af3" / args.walle)
        args.variant = "walle_score"
        args.out_dir = args.out_dir or str(cfg.RESULTS / "walle" / args.walle)

    # Display name for the summary header, metrics.json and figure titles. It was
    # once derived from --variant alone, so a BepiPred or ESMFold2 run that kept the
    # default variant was written as "AF3". Anything that is not the AF3 default
    # must now say what it is.
    if args.method is None:
        if args.walle:
            args.method = f"WALLE {args.walle}"
        elif args.scores_dir is None or Path(args.scores_dir) == cfg.SCORES:
            args.method = "AF3" if args.variant == "contact_top" else f"AF3 {args.variant}"
        else:
            ap.error("--method is required when --scores_dir is not the AF3 default")
    scores_dir = Path(args.scores_dir) if args.scores_dir else cfg.SCORES
    out_dir = Path(args.out_dir) if args.out_dir else cfg.RESULTS

    man = pd.read_csv(cfg.MANIFEST)
    masks = bin_masks()
    bins = list(masks)
    scores, have = load_scores(man.case_id.tolist(), args.variant, scores_dir)
    print(f"scores : {scores_dir}  [{args.variant}]")
    keep = have
    print(f"cases: {len(man)} total | {int(keep.sum())} scored")
    if keep.sum() == 0:
        raise SystemExit("No scored cases yet -- run stages 01-04 first.")

    rows = []
    for i in np.flatnonzero(keep):
        s = scores[i]
        # Mean score per residue within each bin: size-normalised, so a large bin
        # cannot win simply by covering more residues.
        agg = {b: float(s[m].mean()) for b, m in masks.items()}
        order = sorted(agg, key=agg.get, reverse=True)
        true_bin = man.bin[i]
        y = masks[true_bin].astype(int)
        rows.append({
            "case_id": man.case_id[i], "ab_name": man.ab_name[i],
            "true_bin": true_bin, "pred_bin": order[0],
            "correct": int(order[0] == true_bin),
            "correct_top2": int(true_bin in order[:2]),
            "rank_of_true": order.index(true_bin) + 1,
            "domain_rankbis": rank_biserial(y, s),
            # Task II-style residue-level metrics, with "residue is in the correct
            # bin" as the positive label. The floor differs per bin (bins are
            # 16-40 residues), so the enrichment over floor is the comparable
            # number, not the raw AUPRC.
            "domain_auprc": auprc(y, s),
            "domain_floor": float(y.mean()),
            "domain_p_at_L": p_at_l(y, s),
            # Bin-swap: this antibody's score against its OWN bin vs against the
            # other bins. The Task II swap test, adapted -- "own" and "cross" are
            # bins rather than individual epitopes, because 116 antibodies share
            # CD4bs and per-antibody swapping would be degenerate.
            "own_bin_auprc": auprc(y, s),
            "cross_bin_auprc": float(np.mean([auprc(masks[b].astype(int), s)
                                              for b in bins if b != true_bin])),
            "trimer_dependent": bool(man.trimer_dependent[i]),
            **{f"score_{b}": agg[b] for b in bins},
        })
    pc = pd.DataFrame(rows)
    pc["domain_enrichment"] = pc.domain_auprc / pc.domain_floor
    pc["bin_swap_gap"] = pc.own_bin_auprc - pc.cross_bin_auprc
    out_dir.mkdir(parents=True, exist_ok=True)
    pc.to_csv(out_dir / "per_case_domain.csv", index=False)

    conf = pd.crosstab(pc.true_bin, pc.pred_bin).reindex(index=bins, columns=bins, fill_value=0)
    conf.to_csv(out_dir / "confusion.csv")

    # ---- antibody-blind floors: a domain accuracy is meaningless without them ----
    counts = pc.true_bin.value_counts()
    majority = counts.max() / len(pc)          # always predict the commonest bin
    # Empirical noise floor: score random vectors through the SAME aggregation.
    # This is not 1/n_bins -- averaging noise over a small bin has higher variance,
    # so small bins win the argmax more often than chance, which drags accuracy
    # below the naive uniform value. Use the measured floor, not the theoretical one.
    rng = np.random.default_rng(0)
    hits = 0
    N_FLOOR_REPS = 200   # 20 gave SE ~0.006, the same size as the effects being read
    for _ in range(N_FLOOR_REPS):
        for tb in pc.true_bin:
            r = rng.random(cfg.ANTIGEN_LEN)
            hits += max(masks, key=lambda b: r[masks[b]].mean()) == tb
    noise_floor = hits / (N_FLOOR_REPS * len(pc))
    # Per-bin random recall. Essential context: bins differ in size (CD4bs is 11
    # residues, V2 is 40) and a noisy argmax favours SMALL bins, so each bin has a
    # different chance level. Without this column "CD4bs recall 0.92" and "V3 recall
    # 0.01" cannot be judged.
    rand_rec = {b: [] for b in bins}
    rng2 = np.random.default_rng(1)
    for _ in range(N_FLOOR_REPS):
        for tb in pc.true_bin:
            r = rng2.random(cfg.ANTIGEN_LEN)
            rand_rec[tb].append(max(masks, key=lambda b: r[masks[b]].mean()) == tb)
    rand_rec = {b: float(np.mean(v)) if v else float("nan") for b, v in rand_rec.items()}
    uniform = 1.0 / len(bins)

    acc = pc.correct.mean()

    # ---- chance-corrected accuracy (Cohen's kappa) ----------------------------
    # Raw accuracy is inflated by the fact that CD4bs is 38% of the set: a model
    # that leans on the commonest bin scores well without using the antibody.
    # kappa = (observed - expected-by-chance) / (1 - expected-by-chance), where the
    # chance term uses the model's OWN prediction marginals, so kappa is 0 for any
    # antibody-blind strategy and 1 for a perfect one. This is the domain-level
    # analogue of "AUPRC minus its floor".
    # MCC uses the full confusion matrix. Kappa uses the true and predicted
    # marginals. Both are reported because the class frequencies are unequal.
    mcc = float(matthews_corrcoef(pc.true_bin, pc.pred_bin))

    p_true = pc.true_bin.value_counts(normalize=True)
    p_pred = pc.pred_bin.value_counts(normalize=True)
    p_chance = float(sum(p_true.get(b, 0) * p_pred.get(b, 0) for b in bins))
    kappa = (acc - p_chance) / (1 - p_chance) if p_chance < 1 else float("nan")

    # ---- macro-averaged recall (balanced accuracy) ---------------------------
    # Plain accuracy is dominated by the largest class. Macro recall weights
    # every bin equally, so a
    # model that ignores the rare classes cannot hide.
    per_bin_recall = {b: pc[pc.true_bin == b].correct.mean()
                      for b in bins if (pc.true_bin == b).any()}
    macro_recall = float(np.mean(list(per_bin_recall.values())))

    # ---- label-permutation null ----------------------------------------------
    # Shuffle which antibody owns which bin and recompute accuracy. This destroys
    # only the prediction<->antibody correspondence (the marginals, the bin sizes
    # and the predictions all stay fixed), which is exactly "does the answer depend
    # on the antibody?" -- the same logic as the Task II swap test.
    perm_rng = np.random.default_rng(0)
    preds = pc.pred_bin.to_numpy()
    truth = pc.true_bin.to_numpy()
    null = np.array([(preds == perm_rng.permutation(truth)).mean() for _ in range(2000)])
    p_value = float((np.sum(null >= acc) + 1) / (len(null) + 1))
    z = float((acc - null.mean()) / (null.std() + 1e-12))

    lines = [
        f"Task III domain localisation ({args.method}, {cfg.ANTIGEN_NAME}, "
        f"{getattr(cfg, 'ANTIBODY_INPUT', 'VH')})",
        f"n = {len(pc)} antibodies over {len(bins)} bins",
        "",
        f"argmax-domain accuracy : {acc:.3f}",
        f"macro-averaged recall  : {macro_recall:.3f}  (every bin weighted equally)",
        f"Matthews corr (MCC)    : {mcc:+.3f}  (chance-corrected, full confusion matrix; 0 = antibody-blind)",
        f"Cohen's kappa          : {kappa:+.3f}  (chance-corrected via marginals; 0 = antibody-blind)",
        f"top-2 accuracy         : {pc.correct_top2.mean():.3f}",
        f"mean rank of true bin  : {pc.rank_of_true.mean():.2f} / {len(bins)}",
        f"domain rank-biserial   : mean {pc.domain_rankbis.mean():+.3f} "
        f"| median {pc.domain_rankbis.median():+.3f}",
        "",
        "residue-level view (Task II metrics vs the domain label):",
        f"  domain AUPRC         : mean {pc.domain_auprc.mean():.3f} "
        f"| median {pc.domain_auprc.median():.3f}",
        f"  its floor (bin size) : mean {pc.domain_floor.mean():.3f}  -> enrichment "
        f"{pc.domain_enrichment.mean():.1f}x   (floors differ per bin: compare the enrichment)",
        f"  precision@L          : mean {pc.domain_p_at_L.mean():.3f} "
        f"| median {pc.domain_p_at_L.median():.3f}",
        f"  bin-swap gap         : own {pc.own_bin_auprc.mean():.3f} - cross "
        f"{pc.cross_bin_auprc.mean():.3f} = {pc.bin_swap_gap.mean():+.3f}  "
        f"(win-rate {(pc.bin_swap_gap > 0).mean():.2f})",
        "",
        "antibody-blind floors:",
        f"  always-commonest-bin : {majority:.3f}  (bin {counts.idxmax()}, n={counts.max()})",
        f"  random score vector  : {noise_floor:.3f}  "
        f"(measured over {N_FLOOR_REPS} replicates)",
        f"  chance agreement     : {p_chance:.3f}  (given the prediction marginals)",
        "",
        f"conditioning (label-permutation null, 2000 shuffles):",
        f"  p = {p_value:.4f}   z = {z:+.1f}   null accuracy {null.mean():.3f}",
        "  null: predicted domain is independent of antibody identity",
        "",
        "per-bin accuracy (recall):",
    ]
    per_bin_rows = []
    for b in bins:
        sub = pc[pc.true_bin == b]
        if len(sub) == 0:
            continue
        flag = "  [trimer-dependent epitope; input contains one protomer]" \
            if b in cfg.TRIMER_DEPENDENT else ""
        # Recall and precision are reported together because predicted class
        # frequencies may differ.
        tp = int(((pc.true_bin == b) & (pc.pred_bin == b)).sum())
        npred = int((pc.pred_bin == b).sum())
        rec = sub.correct.mean()
        prec = tp / npred if npred else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if npred and (prec + rec) > 0 else 0.0
        verdict = "BELOW random" if rec < rand_rec[b] else ""
        lines.append(f"  {b:10s} n={len(sub):3d}  recall {rec:.3f} (random {rand_rec[b]:.3f})  "
                     f"precision {prec:.3f} ({tp}/{npred} predicted)  F1 {f1:.3f}  "
                     f"rankbis {sub.domain_rankbis.mean():+.3f}  {verdict}{flag}")
        per_bin_rows.append({
            "bin": b, "n": int(len(sub)), "recall": float(rec),
            "random_recall": float(rand_rec[b]),
            "precision": (float(prec) if npred else None),
            "tp": tp, "n_predicted": npred, "f1": float(f1),
            "rankbis": float(sub.domain_rankbis.mean()),
            "below_random": bool(rec < rand_rec[b]),
            "trimer_dependent": b in cfg.TRIMER_DEPENDENT,
        })
    lines += ["", f"macro recall vs random macro: {macro_recall:.3f} vs "
              f"{np.mean(list(rand_rec.values())):.3f}",
              "Recall and precision are reported together because predicted class frequencies differ.",
              f"predictions made per bin: "
              f"{pc.pred_bin.value_counts().reindex(bins).fillna(0).astype(int).to_dict()}"]
    lines += ["", "confusion matrix (rows = true, cols = predicted):", conf.to_string()]

    # Machine-readable twin of summary.txt. The figure script reads THIS, never the
    # prose -- so a reworded summary line can never silently change a figure.
    metrics = {
        "task": __import__("os").environ.get("EIB_TASK", "task2"),
        "variant": args.variant,
        "method": args.method,
        "n": int(len(pc)),
        "bins": bins,
        "accuracy": float(acc),
        "macro_recall": float(macro_recall),
        "random_macro_recall": float(np.mean(list(rand_rec.values()))),
        "kappa": float(kappa),
        "mcc": mcc,
        "top2": float(pc.correct_top2.mean()),
        "mean_rank_true": float(pc.rank_of_true.mean()),
        "rankbis_mean": float(pc.domain_rankbis.mean()),
        "rankbis_median": float(pc.domain_rankbis.median()),
        "floor_majority": float(majority),
        "floor_majority_bin": str(counts.idxmax()),
        "floor_random": float(noise_floor),
        "floor_uniform": float(uniform),
        "chance_agreement": float(p_chance),
        "perm_p": float(p_value), "perm_z": float(z),
        "perm_null_mean": float(null.mean()),
        "n_predicted_per_bin": {b: int((pc.pred_bin == b).sum()) for b in bins},
        "per_bin": per_bin_rows,
        "confusion": conf.to_dict(orient="index"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    txt = "\n".join(lines)
    (out_dir / "summary.txt").write_text(txt + "\n")
    print("\n" + txt)
    print(f"\nwrote {out_dir}/per_case_domain.csv, confusion.csv, summary.txt, metrics.json")


if __name__ == "__main__":
    main()
