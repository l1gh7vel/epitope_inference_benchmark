#!/usr/bin/env python3
"""
Stage 09 -- evaluate WALLE against gold, head-to-head with AF3.

Reuses the stage-05 metrics (AUPRC, precision@L, rank-biserial, swap test) so WALLE
and AF3 are scored *identically* and are directly comparable. For every checkpoint
it reports:
  - both scoring conventions: `full` (all 1273, buried=0) and `surface` (only the
    residues WALLE scored, using WALLE's own surface mask; AF3 restricted to the
    same set),
  - both leakage conditions: `whole` (all headline cases) and `clean` (cases whose
    PDB is NOT in that checkpoint's AsEP train split -- ratio vs group split differ;
    see data/manifests/walle_leakage.csv), and
  - RBM / non-RBM split, plus AF3's numbers on the identical case set.

The swap (conditioning) test is run per checkpoint in the full/whole condition --
the headline "does the predicted epitope move with the antibody?" question.

    PYTHONPATH=scripts python scripts/09_evaluate_walle.py --source af3

Writes results/task2/walle/<source>/summary.csv (per-case metrics aggregated) and
swap.csv, and prints a comparison table.

Overlap is defined by PDB identifier or at least 95% VH-sequence identity to a
spike-antigen training complex; see data/manifests/walle_leakage.csv.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common import cfg, load_labels

# reuse stage-05 metric functions verbatim (identical scoring to AF3)
_s = importlib.util.spec_from_file_location("s05", Path(__file__).parent / "05_evaluate_task2.py")
s05 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s05)
auprc, p_at_l, rank_biserial, swap_test = s05.auprc, s05.p_at_l, s05.rank_biserial, s05.swap_test

GROUP_SPLIT_CKPTS = {"jumping-sweep-22", "wild-sweep-1"}   # trained on the epitope_group split
WALLE_SCORES = cfg.WORK / "walle" / "scores"


def load_walle(ckpt, case_ids, source):
    scores, surf, have = [], [], np.zeros(len(case_ids), bool)
    for i, cid in enumerate(case_ids):
        p = WALLE_SCORES / source / ckpt / f"{cid}.npz"
        if p.exists():
            z = np.load(p)
            scores.append(z["walle_score"].astype(float))
            surf.append(z["surf_mask"].astype(bool))
            have[i] = True
        else:
            scores.append(np.zeros(cfg.ANTIGEN_LEN))
            surf.append(np.zeros(cfg.ANTIGEN_LEN, bool))
    return np.vstack(scores), np.vstack(surf), have


def load_af3(case_ids):
    out, have = [], np.zeros(len(case_ids), bool)
    for i, cid in enumerate(case_ids):
        p = cfg.SCORES / f"{cid}.npz"
        if p.exists():
            out.append(np.load(p)["contact_mean"].astype(float))
            have[i] = True
        else:
            out.append(np.zeros(cfg.ANTIGEN_LEN))
    return np.vstack(out), have


def per_case(score, labels, valid, keep, groups):
    """AUPRC / rank-biserial / precision@L per kept case, on the valid positions."""
    rows = []
    for i in np.flatnonzero(keep):
        m = valid[i]
        y = labels[i][m]
        if y.sum() == 0 or y.sum() == m.sum():      # no signal to grade
            continue
        s = score[i][m]
        rows.append({"case_idx": int(i), "epitope_class": groups[i], "auprc": auprc(y, s),
                     "rankbis": rank_biserial(y, s), "p_at_L": p_at_l(y, s)})
    return pd.DataFrame(rows)


def summarise(df, method, ckpt, conv, leak):
    """Metrics overall and per RBM group -> tidy rows.

    BOTH mean and median are reported. They differ a lot for rank-biserial (its
    per-antibody distribution is skewed), and the AF3 deck figures 2-4 are box
    plots showing MEDIANS while these comparison tables/figures use MEANS -- so
    quoting one against the other is not a fair comparison. Bare `auprc` /
    `rankbis` / `p_at_L` stay the means for backward compatibility; the `*_median`
    columns are the medians."""
    out = []
    for grp, sub in [("all", df)] + [(g, df[df.epitope_class == g]) for g in cfg.REPORTED_CLASSES]:
        if len(sub) == 0:
            continue
        out.append({"method": method, "checkpoint": ckpt, "convention": conv,
                    "leakage": leak, "group": grp, "n": len(sub),
                    "auprc": sub.auprc.mean(), "rankbis": sub.rankbis.mean(),
                    "p_at_L": sub.p_at_L.mean(),
                    "auprc_median": sub.auprc.median(),
                    "rankbis_median": sub.rankbis.median(),
                    "p_at_L_median": sub.p_at_L.median()})
    return out


def compare_sources(args):
    """AF3-predicted vs experimental INPUT structures, on a matched residue set.

    WALLE scores surface residues only, and the two inputs do not have the same
    surface set. An experimental structure is missing every unresolved residue
    (Task II median 999 of 1,273 resolved), so it yields fewer graph nodes than
    the AlphaFold3 structure, which has all 1,273. Scoring each condition against
    all 1,273 compares two different residue pools, and the pool difference alone
    moves AUPRC: the condition with fewer scored residues looks different for
    reasons unrelated to structure quality.

    Per case, take the intersection of the two surface masks and score
    everything -- WALLE on AF3 input, WALLE on experimental input, and AF3's own
    contact score -- on exactly that set. AF3 is included on the same footing so the
    panel reads as three methods on one residue pool rather than two incomparable
    columns.
    """
    man = pd.read_csv(cfg.MANIFEST)
    case_ids, labels, _ = load_labels()
    assert list(man.case_id) == list(case_ids)
    groups = man.epitope_class.to_numpy(dtype=object)
    headline = man.chain_role_ok.to_numpy(dtype=bool, copy=True)
    af3_score, af3_have = load_af3(case_ids)

    ckpts = sorted(d.name for d in (WALLE_SCORES / "af3").iterdir() if d.is_dir())
    rows, pc_rows = [], []
    for ckpt in ckpts:
        a_score, a_surf, a_have = load_walle(ckpt, case_ids, "af3")
        e_score, e_surf, e_have = load_walle(ckpt, case_ids, "experimental")
        keep = headline & a_have & e_have & af3_have
        if keep.sum() == 0:
            print(f"  {ckpt}: not scored in both conditions, skipping")
            continue
        both = a_surf & e_surf                      # the matched residue pool
        n_a, n_e, n_b = a_surf[keep].sum(1).mean(), e_surf[keep].sum(1).mean(), both[keep].sum(1).mean()
        for method, score in [("WALLE (AF3-predicted input)", a_score),
                              ("WALLE (experimental input)", e_score),
                              ("AF3 contact score", af3_score)]:
            pc = per_case(score, labels, both, keep, groups)
            rows += summarise(pc, method, ckpt, "matched-surface", "whole")
            # the summary hides that this is a PAIRED comparison -- the same cases
            # scored from two inputs -- so keep the per-case rows for the figure and
            # for a signed-rank test on the pairs.
            pc_rows.append(pc.assign(method=method, checkpoint=ckpt,
                                     case_id=[case_ids[i] for i in pc.case_idx]))
        print(f"  {ckpt}: n={int(keep.sum())} | surface residues AF3 {n_a:.0f}, "
              f"experimental {n_e:.0f}, matched {n_b:.0f}")

    out = cfg.RESULTS / "walle" / "source_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pcout = cfg.RESULTS / "walle" / "source_comparison_per_case.csv"
    pd.concat(pc_rows, ignore_index=True).to_csv(pcout, index=False)
    print(f"\nwrote {out}\nwrote {pcout}")
    print("Every row is scored on the residues BOTH input structures resolve, so the\n"
          "three methods are directly comparable and no residue-pool artefact remains.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="af3", choices=["af3", "experimental"])
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--no_swap", action="store_true")
    ap.add_argument("--compare_sources", action="store_true",
                    help="AF3-predicted vs experimental INPUT structures, scored on the "
                         "residues both resolve (see compare_sources()). Writes "
                         "source_comparison.csv and ignores --source.")
    args = ap.parse_args()

    if args.compare_sources:
        compare_sources(args)
        return

    man = pd.read_csv(cfg.MANIFEST)
    leak = pd.read_csv(cfg.MANIFESTS / "walle_leakage.csv").set_index("case_id")
    case_ids, labels, obs = load_labels()
    assert list(man.case_id) == list(case_ids)
    groups = man.epitope_class.to_numpy(dtype=object)
    headline = man.chain_role_ok.to_numpy(dtype=bool, copy=True)

    ckpts = sorted(d.name for d in (WALLE_SCORES / args.source).iterdir() if d.is_dir())
    af3_score, af3_have = load_af3(case_ids)
    print(f"{len(ckpts)} checkpoints | {int(headline.sum())} headline cases | "
          f"AF3 scored {int(af3_have.sum())}")

    rows, swap_rows, per_case_rows = [], [], []
    for ckpt in ckpts:
        w_score, w_surf, w_have = load_walle(ckpt, case_ids, args.source)
        # "_any" = in that split's AsEP TRAIN set by PDB id OR by VH sequence
        # identity >= 0.95 against a Spike-antigen training complex (the sequence
        # arm catches the same antibody redeposited under a different PDB).
        base = "leaked_group" if ckpt in GROUP_SPLIT_CKPTS else "leaked_ratio"
        leak_col = f"{base}_any" if f"{base}_any" in leak.columns else base
        leaked = leak.loc[case_ids, leak_col].to_numpy().astype(bool)
        base_keep = headline & w_have & af3_have
        if base_keep.sum() == 0:
            print(f"  {ckpt}: no scored cases yet, skipping")
            continue

        for conv in ["full", "surface"]:
            for cond in ["whole", "clean"]:
                keep = base_keep & (~leaked if cond == "clean" else True)
                # valid positions: full = all 1273; surface = WALLE's surface set
                for method, score in [("WALLE", w_score), ("AF3", af3_score)]:
                    valid = (w_surf if conv == "surface"
                             else np.ones((len(case_ids), cfg.ANTIGEN_LEN), bool))
                    pcdf = per_case(score, labels, valid, keep, groups)
                    rows += summarise(pcdf, method, ckpt, conv, cond)
                    # keep the raw per-antibody values for the headline cell only --
                    # distribution figures need them, summaries cannot be un-averaged
                    if conv == "full" and cond == "whole":
                        pcdf = pcdf.assign(method=method, checkpoint=ckpt)
                        pcdf["case_id"] = [case_ids[j] for j in pcdf.case_idx]
                        per_case_rows.append(pcdf)

        # swap / conditioning test: full positions, whole set (headline question)
        if not args.no_swap:
            ki = np.flatnonzero(base_keep)
            for method, score in [("WALLE", w_score), ("AF3", af3_score)]:
                st, _ = swap_test(np.vstack([score[i] for i in ki]),
                                  np.vstack([labels[i] for i in ki]),
                                  groups[ki], args.n_perm, np.random.default_rng(0))
                st.update({"method": method, "checkpoint": ckpt})
                swap_rows.append(st)
            print(f"  {ckpt}: swap gap WALLE {swap_rows[-2]['swap_gap']:+.3f} "
                  f"(p={swap_rows[-2]['p_value']:.3f}) | AF3 {swap_rows[-1]['swap_gap']:+.3f}")

    out = cfg.RESULTS / "walle" / args.source
    out.mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    if per_case_rows:
        pd.concat(per_case_rows).to_csv(out / "per_case.csv", index=False)
        print(f"  per-antibody values -> {out/'per_case.csv'}")
    if swap_rows:
        pd.DataFrame(swap_rows).to_csv(out / "swap.csv", index=False)

    # ---- headline table: full/whole, overall + per group ----
    print(f"\n=== WALLE vs AF3 (full convention, whole set), mean AUPRC / rank-biserial ===")
    hl = summ[(summ.convention == "full") & (summ.leakage == "whole")]
    for ckpt in ckpts:
        sub = hl[hl.checkpoint == ckpt]
        if sub.empty:
            continue
        w = sub[(sub.method == "WALLE") & (sub.group == "all")]
        a = sub[(sub.method == "AF3") & (sub.group == "all")]
        if w.empty:
            continue
        print(f"  {ckpt:20s} WALLE AUPRC {w.auprc.iloc[0]:.3f} rb {w.rankbis.iloc[0]:+.2f} "
              f"| AF3 AUPRC {a.auprc.iloc[0]:.3f} rb {a.rankbis.iloc[0]:+.2f}  (n={int(w.n.iloc[0])})")
    print(f"\nwrote {out/'summary.csv'} and swap.csv")


if __name__ == "__main__":
    main()
