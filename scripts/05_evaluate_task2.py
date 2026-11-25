#!/usr/bin/env python3
"""
Stage 05 -- Task II evaluation: AF3 vs gold epitopes, vs antibody-blind floors.

Three questions, in order of how much they matter:

 1. Does AF3 find the epitope at all?
    Per antibody: AUPRC, precision@L (L = number of gold epitope residues) and
    rank-biserial r vs the gold labels. Every one is reported next to its own
    random floor (the positive rate), because on a 1273-residue antigen with
    ~17 epitope residues the floor is ~0.013 and a raw AUPRC means nothing
    without it.

 2. Does it beat a method that ignores the antibody?
    The antibody-blind baselines are random, the RBD mask (319-541) and the
    RBM mask (436-507), each scored against every antibody's own gold epitope.
    The RBM mask reaches ~0.30 AUPRC on this antigen purely because the RBM is
    immunodominant. That is the number to beat, not the random floor.

 3. Is it conditioning on the antibody, or on the antigen?
    RBM vs non-RBM split: a surface-propensity model does well on RBM binders
    and badly on the rest. Plus an explicit antibody-swap test -- score each
    prediction against other antibodies' epitopes on the same antigen. If
    own == cross the prediction never moved with the antibody, and a
    label-permutation null says whether any gap is real.

    python scripts/05_evaluate_task2.py
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

import sys
from pathlib import Path

from swap_test import swap_test as _shared_swap_test   # noqa: E402
from common import cfg, load_labels, window_mask       # noqa: E402

# Score variants written by stage 04. contact_top reads the top-ranked structure.
SCORE_VARIANTS = ["contact_mean", "contact_freq", "bsa_mean", "contact_top", "bsa_top"]
PRIMARY = "contact_top"

# Two series, assigned in fixed order from the validated categorical palette
# (slots 1 and 2). Group identity is also carried by the axis labels, so it is
# never colour alone.
COLOUR = {"RBM": "#2a78d6", "nonRBM": "#eb6834"}
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

# Panel order is shared across methods.
PANELS = [(f"{PRIMARY}_auprc", "AUPRC"), (f"{PRIMARY}_p_at_L", "precision@L"),
          (f"{PRIMARY}_rankbis", "rank-biserial r"), ("bsa_mean_auprc", "BSA AUPRC"),
          ("rbm_mask_auprc", "RBM-mask AUPRC\n(antibody-blind)"),
          ("pos_rate", "positive rate\n(random floor)")]


def plot_groups(per_case, path):
    """RBM vs non-RBM per metric, with the antibody-blind floors drawn in.

    The floors are the point of the figure: an AUPRC box sitting below the
    dashed RBM-mask line means AF3 lost to a fixed mask that never saw the
    antibody, which is the result the benchmark exists to detect.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = per_case.pos_rate.median()
    rbm_ceiling = per_case.rbm_mask_auprc.median()

    fig, axes = plt.subplots(1, len(PANELS), figsize=(2.5 * len(PANELS), 4.2),
                             facecolor=SURFACE)
    for ax, (col, title) in zip(axes, PANELS):
        if col not in per_case:
            ax.axis("off")
            continue
        groups = [per_case.loc[per_case.epitope_class == g, col].dropna().to_numpy()
                  for g in COLOUR]
        bp = ax.boxplot(groups, positions=[0, 1], widths=0.58, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color=INK, lw=2),
                        whiskerprops=dict(color=INK_MUTED, lw=1),
                        capprops=dict(color=INK_MUTED, lw=1),
                        boxprops=dict(lw=0))
        for patch, g in zip(bp["boxes"], COLOUR):
            # 2px surface ring keeps adjacent fills from touching.
            patch.set(facecolor=COLOUR[g], alpha=0.85, edgecolor=SURFACE, lw=2)
        rng = np.random.default_rng(0)
        for i, d in enumerate(groups):
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(d)), d, s=12, zorder=3,
                       color=INK, alpha=0.35, edgecolor="none")

        # Reference lines only where they mean something: the floors belong on
        # the model's own AUPRC panels, not on the panels that *are* the floors.
        if col.endswith("auprc") and col not in ("rbm_mask_auprc", "rbd_mask_auprc"):
            ax.axhline(floor, color=INK_MUTED, ls=":", lw=1.5, zorder=1)
            ax.axhline(rbm_ceiling, color=INK_MUTED, ls="--", lw=1.5, zorder=1)
        if "rankbis" in col:
            ax.axhline(0, color=INK_MUTED, ls="--", lw=1.5, zorder=1)

        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(COLOUR), fontsize=9, color=INK)
        ax.tick_params(axis="y", labelsize=8, colors=INK_MUTED)
        ax.grid(axis="y", color=INK_MUTED, alpha=0.15, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK_MUTED)
            ax.spines[s].set_linewidth(0.8)

    handles = [matplotlib.patches.Patch(facecolor=c, edgecolor=SURFACE, lw=2, label=g)
               for g, c in COLOUR.items()]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               labelcolor=INK, ncol=2, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("AF3 epitope localisation vs gold labels, by antibody class\n"
                 "dotted = random floor   dashed = RBM-mask baseline (antibody-blind)",
                 fontsize=11, color=INK, x=0.42)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


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
    """2*AUC-1 via Mann-Whitney U. 0 = no localisation, <0 = anti-localised."""
    x, z = s[y == 1], s[y == 0]
    if not len(x) or not len(z):
        return np.nan
    u = mannwhitneyu(x, z, alternative="two-sided").statistic
    return float(2 * (u / (len(x) * len(z))) - 1)


def load_scores(case_ids, keep):
    """Stack per-case score vectors; returns (scores dict, boolean mask of loaded cases)."""
    out = {v: [] for v in SCORE_VARIANTS}
    have = np.zeros(len(case_ids), bool)
    for i, cid in enumerate(case_ids):
        path = cfg.SCORES / f"{cid}.npz"
        if not keep[i] or not path.exists():
            for v in SCORE_VARIANTS:
                out[v].append(np.zeros(cfg.ANTIGEN_LEN))
            continue
        z = np.load(path)
        for v in SCORE_VARIANTS:
            out[v].append(z[v] if v in z else np.zeros(cfg.ANTIGEN_LEN))
        have[i] = True
    return {v: np.vstack(a) for v, a in out.items()}, have


def swap_test(scores, labels, groups, n_perm, seed=0):
    """Task II's view of the antibody-swap test.

    The implementation is not duplicated here: two copies of a test whose whole
    value is that its blind case is EXACTLY zero are two chances to drift. This
    wrapper only renames the summary
    keys to the ones Task II's CSVs already read, and
    attaches the RBM/non-RBM group so the per-case table can be split by epitope
    class. The statistic, the exclusion rule and the null are the shared ones.
    """
    st, per_case = _shared_swap_test(scores, labels, n_perm=n_perm, seed=seed,
                                     pair_fn=auprc)
    # Rename IN PLACE so the CSV column order is unchanged too: swap_winrate is
    # Task II's long-standing name for win_rate, n_cases for n_antibodies, and
    # n_cases is read downstream.
    rename = {"win_rate": "swap_winrate", "n_antibodies": "n_cases"}
    st = {rename.get(k, k): v for k, v in st.items()}
    return st, per_case.assign(epitope_class=groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_flagged", action="store_true",
                    help="Keep rows whose vh_sequence column is not a heavy chain.")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    case_ids, labels, obs = load_labels()
    assert list(man.case_id) == list(case_ids), "manifest and labels are out of sync"

    keep = np.ones(len(man), bool) if args.include_flagged \
        else man.chain_role_ok.to_numpy(dtype=bool, copy=True)
    scores, have = load_scores(case_ids, keep)
    keep &= have
    print(f"cases: {len(man)} total | {int(keep.sum())} scored and kept "
          f"({int((~man.chain_role_ok).sum())} chain-role flagged, "
          f"{int((~have).sum())} unscored)")
    if keep.sum() == 0:
        raise SystemExit("No scored cases yet -- run stage 04 first.")

    valid = np.ones_like(labels, bool)

    rbm_mask = window_mask(*cfg.RBM).astype(float)
    rbd_mask = window_mask(*cfg.RBD).astype(float)

    # ------------------------------------------------- per-case metrics ----
    rows = []
    for i in np.flatnonzero(keep):
        m = valid[i]
        y = labels[i][m]
        rec = {"case_id": case_ids[i], "epitope_class": man.epitope_class[i],
               "n_epitope": int(y.sum()), "n_scored_positions": int(m.sum()),
               "pos_rate": float(y.mean()),
               "rbm_mask_auprc": auprc(y, rbm_mask[m]),
               "rbd_mask_auprc": auprc(y, rbd_mask[m])}
        for v in SCORE_VARIANTS:
            s = scores[v][i][m]
            rec[f"{v}_auprc"] = auprc(y, s)
            rec[f"{v}_p_at_L"] = p_at_l(y, s)
            rec[f"{v}_rankbis"] = rank_biserial(y, s)
        rows.append(rec)
    per_case = pd.DataFrame(rows)

    out = cfg.RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(out / "per_case_metrics.csv", index=False)

    # --------------------------------------- antibody-blind baseline table ----
    # Per-antibody and pooled summaries answer different questions.
    # pooled concatenates every antibody's residues into one ranking.
    ki = np.flatnonzero(keep)
    y_pool = np.concatenate([labels[i][valid[i]] for i in ki])
    blind = []
    for name, vec in [("random", None), ("RBD mask 319-541", rbd_mask), ("RBM mask 436-507", rbm_mask)]:
        if vec is None:
            per_ab, pooled = float(per_case.pos_rate.mean()), float(y_pool.mean())
        else:
            col = "rbd_mask_auprc" if "RBD" in name else "rbm_mask_auprc"
            per_ab = float(per_case[col].mean())
            pooled = auprc(y_pool, np.concatenate([vec[valid[i]] for i in ki]))
        blind.append({"baseline": name, "per_antibody_mean": per_ab, "pooled": pooled})
    for v in SCORE_VARIANTS:
        blind.append({"baseline": f"AF3 {v}",
                      "per_antibody_mean": float(per_case[f"{v}_auprc"].mean()),
                      "pooled": auprc(y_pool, np.concatenate([scores[v][i][valid[i]] for i in ki]))})
    blind = pd.DataFrame(blind)
    blind.to_csv(out / "blind_baselines.csv", index=False)

    # ------------------------------------------- RBM vs non-RBM contrast ----
    summ = []
    metrics = ([f"{v}_{m}" for v in SCORE_VARIANTS for m in ("auprc", "p_at_L", "rankbis")]
               + ["rbm_mask_auprc", "rbd_mask_auprc", "pos_rate"])
    for col in metrics:
        a = per_case.loc[per_case.epitope_class == "RBM", col].dropna()
        b = per_case.loc[per_case.epitope_class == "nonRBM", col].dropna()
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else np.nan
        summ.append({"metric": col, "RBM_median": a.median(), "nonRBM_median": b.median(),
                     "RBM_n": len(a), "nonRBM_n": len(b), "mwu_p": p})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "group_summary.csv", index=False)

    # --------------------------------------------------- conditioning ----
    # The swap test always runs on the full 1273-residue frame: cross-scoring
    # needs one shared coordinate system, and each row's observed mask differs.
    swap, swap_per_case = swap_test(np.vstack([scores[PRIMARY][i] for i in ki]),
                                    np.vstack([labels[i] for i in ki]),
                                    per_case.epitope_class.to_numpy(),
                                    args.n_perm, 0)
    pd.DataFrame([swap]).to_csv(out / "swap_test.csv", index=False)
    swap_per_case.assign(case_id=per_case.case_id).to_csv(out / "swap_per_case.csv", index=False)

    plot_groups(per_case, str(out / "rbm_vs_nonrbm"))

    # ------------------------------------------------------------ report ----
    print(f"\n{'=' * 72}\nTask II -- AF3 vs gold epitopes  [{args.tag}]\n{'=' * 72}")
    print(f"scored positions per antibody: median {int(per_case.n_scored_positions.median())}"
          f" / {cfg.ANTIGEN_LEN}\n")
    print("AUPRC, antibody-blind floors first (higher = better):")
    print(blind.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nRBM vs non-RBM (median per antibody), primary score = {PRIMARY}:")
    show = summ[summ.metric.str.startswith(PRIMARY) |
                summ.metric.isin(["rbm_mask_auprc", "pos_rate"])]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nAntibody-swap conditioning test (own vs other antibodies' epitopes):")
    for k, v in swap.items():
        print(f"  {k:14s} {v:.4f}" if isinstance(v, float) else f"  {k:14s} {v}")
    print("\n  gap > 0 with a small p means the predicted site moves with the "
          "antibody.\n  gap ~ 0 means AF3 is predicting the antigen, not the complex.")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
