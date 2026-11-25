#!/usr/bin/env python3
"""Generate the ESM-2 antibody-representation similarity panel.

The analysis compares cosine similarity of mean-pooled whole-VH and CDR-only
embeddings with an antigen-sequence control. Pairs are matched within five-
residue length bins; antigen controls use one sequence per 90%-identity cluster.

    PYTHONPATH=scripts:config python scripts/29_esm_degeneracy_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import figstyle                                                  # noqa: E402
from figstyle import COL2, fit_save, panel_letter                # noqa: E402

RES = PROJECT / "results" / "esm_degeneracy"
OUT = PROJECT / "results" / "paper_figures"

GREY_D, GREY_M, GREY_L = "#6E6E69", "#9A9A95", "#C7C7C2"
INK, MUTED, SURFACE = "#1A1A1A", "#5C5C57", "#FFFFFF"
# Okabe-Ito. Nothing here is a method, so no method colour is borrowed.
C_AB, C_CDR, C_AG = "#0072B2", "#009E73", "#E69F00"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, **figstyle.BASE_RC,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": False, "legend.frameon": False,
})

HEADLINE = "esm2_650M"
ORDER = ["esm2_8M", "esm2_35M", "esm2_150M", "esm2_650M", "esm2_3B"]
NICE = {"esm2_8M": "8M", "esm2_35M": "35M", "esm2_150M": "150M",
        "esm2_650M": "650M", "esm2_3B": "3B"}
# The antigen control is the cluster-deduplicated set: one sequence per CD-HIT
# 90% cluster, 994 sequences. `antigen_exact` is also computed and is in the
# results CSV.
SETS = [("antibody", C_AB, "antibodies"),
        ("antibody_cdr", C_CDR, "antibodies, CDR only"),
        ("antigen", C_AG, "antigens (control)")]


def main():
    H = pd.read_csv(RES / "cosine_hist_by_model.csv")
    S = pd.read_csv(RES / "summary_by_model.csv")
    E = pd.read_csv(RES / "epitope_split_by_model.csv")
    bcols = [c for c in H.columns if c.startswith("b") and c[1:].isdigit()]
    edges = np.linspace(0.0, 1.0, len(bcols) + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 3, figsize=(COL2 * 0.78, 1.95),
                             gridspec_kw={"wspace": 0.46,
                                          "width_ratios": [1.25, 1.0, 1.0]})

    # ---- (a) the distributions, at one model scale --------------------------
    ax = axes[0]
    meds = []
    for name, col, lab in SETS:
        r = H[(H.model == HEADLINE) & (H.item_set == name)]
        if not len(r):
            continue
        v = r[bcols].values[0].astype(float)
        # Heights are per cent of pairs. The bins are a fixed 0.005 wide, so
        # per cent of that set's pairs has the same shape as a probability
        # density and reads without holding the bin width in mind.
        v = 100 * v / v.sum()
        ax.plot(ctr, v, color=col, lw=1.2, zorder=3)
        ax.fill_between(ctr, 0, v, color=col, alpha=0.16, lw=0, zorder=2)
        # The median is marked on each curve. The three distributions overlap
        # heavily in the 0.95-1.0 region, and all three are strongly
        # left-skewed, so the median sits well left of the mode that the eye
        # reads as the centre. The value comes from the summary CSV, which
        # stores it exactly. Histogram bins are 0.005 wide, so the antibody
        # median (0.9783) and the CDR-only median (0.9761) fall in bins whose
        # centres both print "0.978".
        r = S[(S.model == HEADLINE) & (S.item_set == name) & (S.bucket_w == 5)
              & (S.standardised == 0 if "standardised" in S.columns else True)]
        med = float(r["median"].iloc[0])
        meds.append((med, float(np.interp(med, ctr, v)), col))
    # The key sits inside the empty left half, clear of the y tick labels. The
    # three series are identified once for the whole figure; panels (b) and (c)
    # reuse the same colours and carry no key.
    handles = [plt.Line2D([], [], color=c, lw=1.4) for _n, c, _l in SETS]
    handles.append(plt.Line2D([], [], color=INK, lw=0.6, ls=(0, (1.2, 1.2))))
    ax.legend(handles, [l for _n, _c, l in SETS] + ["medians"],
              loc="upper left", fontsize=5.8,
              handlelength=1.1, handletextpad=0.4, borderpad=0.0,
              labelspacing=0.3, bbox_to_anchor=(0.04, 1.02))
    # x runs 0.6 to 1.0. Below 0.6 every series is flat against zero, and the
    # full 0-1 range compresses the region where they differ into a third of the
    # axis. The lower limit is stated in the legend.
    # The medians are drawn after the curves, as full-height black dotted rules
    # with the value in a reserved band of white space above the curves. One
    # black rule named once in the key is less ink than three coloured ones, and
    # a per-series colour leaves the antibody median invisible inside its own
    # filled curve; black on white also clears the contrast floor a pale series
    # colour would not. The three distributions fill the panel from the axis to
    # the peak, so the y limit is set to 1.35x the tallest curve and the labels
    # live in the top third of that headroom, clear of every line.
    ymax = max(y for _m, y, _c in meds) if meds else 1.0
    ytop = ymax * 1.35
    ax.set_ylim(0, ytop)
    order = sorted(meds)
    # Each value sits beside its own rule, on one row, offset to one side so
    # that no digit lands on a dotted line:
    #   * a median with no neighbour within 0.02 goes left of its rule;
    #   * where two are close, the lower goes left and the higher goes right, so
    #     the pair opens outwards and the gap between the rules stays empty.
    # The labels are 5.8 pt, inside Nature's 5-7 pt window and the size this
    # project uses for secondary annotation.
    GAP = 2.0                       # points of clearance between digit and rule
    for k, (med, _y, _c) in enumerate(order):
        ax.axvline(med, color=INK, lw=0.6, ls=(0, (1.2, 1.2)), zorder=6)
        hi_of_close_pair = any(0 < med - m2 < 0.02
                               for j, (m2, _b, _c2) in enumerate(order) if j != k)
        right = hi_of_close_pair
        ax.annotate(f"{med:.3f}", xy=(med, ytop * 0.90),
                    xytext=(GAP if right else -GAP, 0), textcoords="offset points",
                    fontsize=5.8, color=INK, zorder=7,
                    ha="left" if right else "right", va="bottom")
    ax.set_xlim(0.60, 1.0)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(0, None)
    ax.set_xlabel("pairwise cosine similarity")
    ax.set_ylabel("% of pairs", fontsize=6.5)

    # ---- (b) does it hold across model scale --------------------------------
    ax = axes[1]
    prim = S[S.bucket_w == 5]
    for name, col, _lab in SETS:
        y = [float(prim[(prim.model == m) & (prim.item_set == name)]["median"].iloc[0])
             for m in ORDER if len(prim[(prim.model == m) & (prim.item_set == name)])]
        x = np.arange(len(y))
        ax.plot(x, y, "-", color=col, lw=1.2, zorder=3)
        ax.plot(x, y, "o", color=col, ms=2.4, zorder=4)
    ax.set_xticks(range(len(ORDER)), [NICE[m] for m in ORDER], fontsize=5.8)
    # y runs 0.70 to 1.00 at two decimals. A 0.85 to 1.00 range puts the antigen
    # control on the floor of the panel and fills it edge to edge, which
    # exaggerates differences of 0.002 between the two antibody series.
    ax.set_ylim(0.70, 1.0)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels([f"{t:.2f}" for t in
                        [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]])
    ax.set_xlabel("ESM-2 parameters")
    ax.set_ylabel("median cosine", fontsize=6.5)

    # ---- (c) does the embedding know which epitope the antibody binds -------
    # Among antibodies against the same antigen and in the same length bin,
    # split pairs into same-epitope (overlap coefficient >= 0.5) and
    # different-epitope (strictly disjoint, no shared residue). AUC = P(cosine
    # of a random same-epitope pair > cosine of a random different-epitope
    # pair), ties half, as Mann-Whitney U / (n_same n_diff). One antigen gives
    # one AUC, so the statistic is the median over the 67 antigens that have
    # both kinds of pair. 0.5 is chance and is drawn as the rule.
    ax = axes[2]
    prim_e = E[(E.scope == "same_length_bucket") & (E.bucket_w == 5)]
    y = [float(prim_e[prim_e.model == m]["auc_same_over_diff_per_antigen_median"].iloc[0])
         for m in ORDER if len(prim_e[prim_e.model == m])]
    x = np.arange(len(y))
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (2.0, 1.5)), zorder=2)
    ax.plot(x, y, "-", color=C_AB, lw=1.2, zorder=3)
    ax.plot(x, y, "o", color=C_AB, ms=2.4, zorder=4)
    ax.text(len(ORDER) - 1, 0.515, "chance", fontsize=5.3, color=MUTED,
            ha="right", va="bottom")
    ax.set_xticks(range(len(ORDER)), [NICE[m] for m in ORDER], fontsize=5.8)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("ESM-2 parameters")
    ax.set_ylabel("per-antigen AUC,\nsame vs different epitope", fontsize=6.5)

    # ---- is panel (c) an anisotropy artefact? ----------------------------
    # `epitope_test()` is called outside the standardisation loop, so panel (c)
    # is computed on raw embeddings, and a raw ESM-2 cosine is dominated by the
    # cone effect. Standardising per dimension is the published remedy, and it
    # leaves the effect in place: same 67 antigens, same thresholds, same length
    # bin, ESM-2 650M went 0.708 -> 0.800 (one-sided Wilcoxon 9.8e-5 -> 8.3e-7)
    # and ProteinBERT 0.760 -> 0.775 (1.1e-5 -> 1.2e-6). The data are in
    # results/esm_degeneracy_std_epitope/ and results/proteinbert_degeneracy/,
    # and the ESM-2 re-run asserts its raw arm reproduces the existing CSV to
    # 1e-6 before the standardised number is used. The control is not drawn,
    # because practice does not standardise these embeddings.

    # Sub-panel letters a-c, one per axes, drawn in axes coordinates at PANEL_PT
    # bold, the same way 12_paper_figures.titled_panel does it. This file is one
    # Extended Data figure holding three sub-panels, and figstyle.panel_letter
    # positions against the whole figure's corner, so calling it once per axes
    # would stack the letters in one place. All three letters are drawn because
    # the footnote refers to panel c by name.
    for ax_, lab_ in zip(axes, "abc"):
        ax_.text(0.0, 1.0, lab_, transform=ax_.transAxes,
                 fontsize=figstyle.PANEL_PT, fontweight="bold", color=INK,
                 ha="right", va="bottom")
    got = fit_save(fig, OUT, "figE_esm_degeneracy", (183, 46))

    print(f"\n  headline model {HEADLINE}, length bin width 5")
    for name, _c, lab in SETS:
        r = prim[(prim.model == HEADLINE) & (prim.item_set == name)]
        if len(r):
            r = r.iloc[0]
            print(f"    {lab:<24} n_items {int(r.n_items):>5}  pairs {int(r.n_pairs):>9}"
                  f"  median {r['median']:.4f}  IQR {r.iqr:.4f}"
                  f"  frac>0.99 {r['frac_gt_0.99']:.3f}")
    r = prim_e[prim_e.model == HEADLINE].iloc[0]
    print(f"    epitope split: {int(r.n_same)} same / {int(r.n_diff)} different / "
          f"{int(r.n_intermediate)} intermediate pairs over {int(r.n_antigens_auc)} antigens; "
          f"per-antigen AUC median {r.auc_same_over_diff_per_antigen_median:.3f}, "
          f"{int(r.n_antigens_same_higher)}/{int(r.n_antigens_paired)} antigens in the "
          f"predicted direction, one-sided Wilcoxon P = {r.wilcoxon_p_one_sided:.1e}")
    print(f"  panel written at {got[0]:.1f} x {got[1]:.1f} mm")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
def fig_model_compare():
    """ESM-2 against ProteinBERT, raw and standardised: a 2 x 2 small multiple.

    This is a separate panel rather than extra curves on E8(a). E8(a)'s x axis
    is clipped to 0.60-1.00 because all three ESM-2 series live there.
    ProteinBERT's antigen control has a median of 0.449 and falls off that axis
    entirely, and its antibody and shuffled curves land on top of ESM-2's
    antigen curve at a different meaning.

    Rows are the model, columns are the readout.
      * The right column is nearly the same picture twice. Standardised, the
        series come out in the same order with the same signs in both models,
        and CDR-only collapses far below whole-VH in both. The degeneracy is a
        property of mean-pooling ~125 positions of which ~31 differ, rather than
        an ESM-2 artefact.
      * The left column is two different pictures. ESM-2 puts unrelated
        antigens 0.084 from real antibodies; ProteinBERT puts them ~0.14 away
        on a scale with several times the spread. "Raw cosine carries no
        information" is a statement about ESM-2's readout, not about protein
        language models. Against the shuffled control: ESM-2's shuffled
        sequences sit 0.0008 from its real antibodies (0.9775 against 0.9783),
        ProteinBERT's sit 0.141 away.
    The standardised view is plotted here, where two models make the column
    comparable; E8 carries it in prose.

    Absolute values do not transfer, and the legend says so. ProteinBERT is
    128-dimensional against ESM-2's 1280, and cosine between random vectors has
    s.d. about 1/sqrt(d), so the null spread is ~3x wider. The orderings and the
    signs of the gaps are what carry across.
    """
    pb_dir = RES.parent / "proteinbert_degeneracy"
    if not (pb_dir / "cosine_hist_by_model.csv").exists():
        print("  no ProteinBERT results yet -- skipping the model comparison")
        return
    H_e = pd.read_csv(RES / "cosine_hist_by_model.csv")
    H_p = pd.read_csv(pb_dir / "cosine_hist_by_model.csv")
    H_p = H_p[H_p.track == "local"]
    bcols = [c for c in H_e.columns if c.startswith("b") and c[1:].isdigit()]

    # The shuffled series is not drawn. The shuffle is a permutation of
    # amino-acid order -- esm_ab_degeneracy.py takes one uniform permutation of
    # each VH with seed 0, so length and amino-acid composition are kept and
    # only the order is destroyed -- which makes it a readout sanity check
    # rather than a control on the question the panel asks, whether the
    # representation separates one antibody from another. The cost is that the
    # left column's point then rests on the antigen control alone, which is
    # weaker because antigens differ from antibodies in length and composition
    # as well as in identity. The shuffled numbers (ESM-2's shuffles sit 0.0008
    # from its real antibodies, ProteinBERT's 0.141 away) are in
    # fig_model_compare's docstring, and the legend carries them.
    SERIES = [("antibody", C_AB, "antibodies"),
              ("antibody_cdr", C_CDR, "antibodies, CDR only"),
              ("antigen", C_AG, "antigens (control)")]
    # Every ESM-2 size, plus ProteinBERT, raw only. The standardised comparison
    # lives in panel d of the ESM-2 figure, as the epitope-split AUC with both
    # models on one pair of axes. This figure carries the raw readout across
    # model scale and across architecture, which is what makes "raw cosine
    # carries no information" an ESM-2 statement rather than a statement about
    # language models.
    ROWS = [(NICE[m] if m in NICE else m, H_e, m) for m in ORDER]
    ROWS.append(("ProteinBERT", H_p, "proteinbert_local"))

    ncol = 3
    nrow = int(np.ceil(len(ROWS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL2 * 0.86, 1.35 * nrow),
                             gridspec_kw={"wspace": 0.26, "hspace": 0.62},
                             squeeze=False)
    for i, (mname, H, mkey) in enumerate(ROWS):
        ax = axes[i // ncol][i % ncol]
        for name, col, _lab in SERIES:
            q = H[(H.model == mkey) & (H.item_set == name)
                  & (H.standardised == 0)]
            if not len(q):
                continue
            row = q.iloc[0]
            edges = np.linspace(float(row.lo), float(row.hi), len(bcols) + 1)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            v = row[bcols].values.astype(float)
            v = 100 * v / v.sum()
            ax.plot(ctr, v, color=col, lw=1.0, zorder=3)
            ax.fill_between(ctr, 0, v, color=col, alpha=0.14, lw=0, zorder=2)
        # One x range for the five ESM-2 panels, so model scale is readable
        # across them, and ProteinBERT on its own, because its antigen control
        # sits near 0.45 and sharing an axis with it compresses every ESM-2
        # panel into the right-hand fifth.
        ax.set_xlim(0.2 if mkey != "proteinbert_local" else -0.1, 1.0)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=5.5)
        ax.set_title(mname, fontsize=6.5, pad=3, loc="center")
        if i % ncol == 0:
            ax.set_ylabel("% of pairs", fontsize=6.0)
        if i // ncol == nrow - 1:
            ax.set_xlabel("pairwise cosine similarity", fontsize=6.5)
    for j in range(len(ROWS), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [plt.Line2D([], [], color=c, lw=1.4) for _n, c, _l in SERIES]
    fig.legend(handles, [l for _n, _c, l in SERIES], loc="lower center",
               ncol=3, fontsize=5.8, frameon=False, handlelength=1.1,
               columnspacing=1.4, bbox_to_anchor=(0.5, -0.03))
    # No panel letter. This is Extended Data 6, a figure in its own right rather
    # than a sub-panel of the degeneracy figure.
    return fit_save(fig, OUT, "figE_model_degeneracy_compare", (180, 86))


if __name__ == "__main__":
    fig_model_compare()
