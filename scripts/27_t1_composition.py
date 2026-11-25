#!/usr/bin/env python3
"""Generate the Task I benchmark-composition and coverage panels.

The analysis uses 4,391 distinct antibodies across 1,130 antigen clusters.
Antigen coverage is the union of gold-epitope positions within each antigen,
summed across antigens. The sampling panel compares uniform antibody sampling
with round-robin antigen-stratified sampling at the same antibody budget.

    PYTHONPATH=scripts:config python scripts/27_t1_composition.py
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
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                              # noqa: E402

import figstyle                                                  # noqa: E402
from figstyle import COL2, fit_save, panel_letter                # noqa: E402
from t1_06_epitope_redundancy import load                        # noqa: E402

OUT = PROJECT / "results" / "paper_figures"
RES = cfg.RESULTS

GREY_D, GREY_M, GREY_L = "#6E6E69", "#9A9A95", "#C7C7C2"
INK, MUTED, SURFACE = "#1A1A1A", "#5C5C57", "#FFFFFF"
# Okabe-Ito. Nothing in this panel is a method, so it must not borrow a method
# colour -- a reader arriving from 2c reads #2C6FAD as AlphaFold3.
C_ANTIGEN, C_ANTIBODY, C_COVER = "#C7C7C2", "#0072B2", "#E69F00"
C_OBS, C_CF = "#0072B2", "#009E73"
C_SPIKE, C_ENV = "#D55E00", "#CC79A7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, **figstyle.BASE_RC,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": False, "legend.frameon": False,
})

SEED = 0
N_DRAW_REPS = 20          # draw orderings averaged for panel (ii)
# Bin edges on antibodies per antigen entity. The two largest entities get their
# own named bin, which the data permits unambiguously: no cluster falls between
# 134 and 831, so "HIV-1 Env" and "spike" are singleton bins rather than ranges
# dominated by one member, and the axis names them.
# Type size floor. Nature states a 5 pt minimum ("Minimum text size: 5 pt"), and
# matplotlib renders a requested 5.0-5.3 pt slightly under at this figure scale,
# so nothing here goes below 5.5 pt.
BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"),
        (10, 19, "10-19"), (20, 99, "20-99"),
        (100, 199, "Env\n(132)"), (800, 10 ** 6, "spike\n(832)")]

# Dataset-scale comparison, panel (iii). Only the benchmarked methods are drawn,
# not every method in the field. Each entry is (label, count, unit, kind). The
# unit sits in the label because the papers count different objects, and one axis
# without the units compares unlike things.
#   SAbDab      derived by us from the official all-summary file, see PROVENANCE
#   this work   our Task I evaluated set
#   AsEP/WALLE  the AsEP paper, section 5.3, verified
#   BepiPred    BP3C50ID, the set the released model uses; verified from PMC9679979
#   AlphaFold3  its antibody-antigen test set; AF3 has no epitope training set
#   ESMFold2    PDB and AFDB; model card gives Sept 2021, preprint 2021-09-30
# Every bar is the same quantity where it can be: how many antibody-antigen
# structures were available to that method, counted the same way from the same
# vendored SAbDab summary (paired VH+VL, protein antigen, unique PDB entries) and
# cut at the method's own data cutoff. AlphaFold3 and ESMFold2 have no
# antibody-specific set, so they are shown at the PDB up to their cutoff date.
#   AlphaFold3's 71 is an evaluation set, and beside two training sizes it reads
#   as "AF3 saw 20x less data", the reverse of the truth. What AF3 had is the
#   whole PDB up to 2021-09-30, which contained 2,743 antibody-antigen entries by
#   our count. The two methods with a curated antibody-specific training set are
#   marked as such; the two without are shown at what the PDB offered them.
# kind: "reference" = the database today, "ours" = our evaluation set,
#       "curated"   = a deliberately built antibody-antigen training set,
#       "pdb"       = no antibody-specific set; this is what the PDB held by then
SCALE = [
    ("SAbDab today (2026-10-28)", "PDB entries", 6694, "reference"),
    # Antigen counts come from the manifest. The composition table is written
    # from a cohort of its own; the manifest gives 132 gp120 antibodies over this
    # cohort's 4,391 rows and 1,130 clusters. The bar is labelled "our evaluation
    # set", so it carries the evaluated set:
    # `usable & antigen_len <= 1400 & vh_dedup_keep`.
    ("our evaluation set", "complexes", 3908, "ours"),
    ("AlphaFold3: PDB to 2021-09-30", "PDB entries", 2743, "pdb"),
    # Same basis as AlphaFold3. The biohub/ESMFold2 model card gives a training
    # data cutoff of September 2021 for the checkpoint this project loads, and
    # the preprint puts the day at 2021-09-30, so both bars cover the same span
    # (snapshot 1ebf0e34), so the antibody-antigen entries the PDB held by then
    # is the same count. Neither model has an antibody-specific training set.
    ("ESMFold2: PDB to 2021-09-30", "PDB entries", 2743, "pdb"),
    ("WALLE / AsEP, curated train", "complexes", 1383, "curated"),
    ("BepiPred-3.0, curated train", "antigens", 343, "curated"),
]

def build():
    """One row per antigen entity: antibody count, and the entity's epitope union.

    The grouping key is `Cluster` rather than `antigen_id`, so that constructs of
    one antigen are merged. `antigen_id` is a hash of the exact antigen sequence,
    so one biological antigen deposited as several constructs becomes several
    rows: SARS-CoV-2 spike is 20 antigen_ids and HIV-1 Env gp120 is 9. An overlap
    computation needs that split, because epitope vectors are indexed into their
    own sequence and cannot be intersected across constructs (see
    t1_06_epitope_redundancy.py). A composition count needs the merge, because
    the split reports 20 spike antigens where a reader would say one. `Cluster`
    is the antigen sequence-identity clustering, already in the manifest and
    already the basis of the Task I split, so this introduces no new threshold.
    It merges spike to 832 antibodies in one entity and gp120 to 132. 1,627
    antigen_ids become 1,130 clusters.

    The epitope-union column is not plotted. A union over exact-match residue
    positions counts a site once however many antibodies hit it, so an antigen
    whose epitopes are deposited twice contributes no extra coverage, and the
    series would measure distinct surface rather than redundancy of deposition.
    The union is computed per antigen_id and summed within a cluster, because
    positions are not comparable across constructs.
    """
    man, labels = load(keep_dup_ab=False, apply_fold_dedup=False)
    per_ag = {}
    for a, g in man.groupby("antigen_id"):
        M = np.stack([labels[c] for c in g.case_id]).astype(bool)
        per_ag[a] = (M, int(g.Cluster.iloc[0]))
    rows = []
    for cl, g in man.groupby("Cluster"):
        ids = g.antigen_id.unique()
        rows.append(dict(cluster=int(cl),
                         antigen_name=g.antigen_name.mode().iloc[0],
                         n_antibodies=int(len(g)),
                         n_antigen_ids=int(len(ids)),
                         union_size=int(sum(per_ag[a][0].any(0).sum() for a in ids))))
    D = pd.DataFrame(rows)
    D["pct_antibodies"] = 100 * D.n_antibodies / D.n_antibodies.sum()
    return man, D.sort_values("n_antibodies", ascending=False).reset_index(drop=True)


def concentration(D):
    """Lorenz-style concentration of antibodies over antigen entities.

    Rank antigen entities by how many antibodies they carry, most first. Plot the
    cumulative share of antibodies against the share of antigens. Even deposition
    is the 1:1 diagonal exactly, so the panel needs no simulation and no seed.
    Half the antigen entities carry exactly one antibody, and the knee is at the
    far left: one entity holds 18.9% of the antibodies and 76 entities (6.7%)
    hold half of them.
    """
    c = np.sort(D.n_antibodies.values)[::-1]
    x = np.arange(1, len(c) + 1) / len(c)
    return 100 * x, 100 * np.cumsum(c) / c.sum()


def fig2d(man, D):
    n_ab = int(D.n_antibodies.sum())
    n_cl = len(D)

    fig, axes = plt.subplots(1, 3, figsize=(COL2 * 0.99, 2.05),
                             gridspec_kw={"wspace": 0.50,
                                          "width_ratios": [1.45, 1.0, 1.30]})

    # ---------------- (i) which antigens own the set ---------------------------------
    ax = axes[0]
    xs = np.arange(len(BINS))
    s_ag, s_ab = [], []
    for lo, hi, _ in BINS:
        sl = D[(D.n_antibodies >= lo) & (D.n_antibodies <= hi)]
        s_ag.append(100 * len(sl) / n_cl)
        s_ab.append(float(sl.pct_antibodies.sum()))
    w = 0.38
    # The key sits top-right, outside the data and frameless. Labels on the bars
    # would overlap their neighbours.
    ax.bar(xs - w / 2, s_ag, w * 0.94, color=C_ANTIGEN, edgecolor=GREY_D, lw=0.4,
           zorder=3, label="antigens")
    ax.bar(xs + w / 2, s_ab, w * 0.94, color=C_ANTIBODY, edgecolor=C_ANTIBODY,
           lw=0.4, zorder=3, label="antibodies")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.1,
              handletextpad=0.4, borderpad=0.0, labelspacing=0.25,
              bbox_to_anchor=(1.03, 1.06))
    ax.set_ylim(0, 60)
    ax.set_xticks(xs, [b[2] for b in BINS], fontsize=5.8, rotation=45,
                  ha="right", rotation_mode="anchor")
    ax.set_xlabel("antibodies per antigen")
    ax.set_ylabel("% of total", fontsize=6.5)

    # ---------------- (ii) how concentrated is the deposition ----------------
    ax = axes[1]
    cx, cy = concentration(D)
    ax.plot([0, 100], [0, 100], color=GREY_L, lw=0.8, ls=(0, (1.8, 1.8)), zorder=1)
    ax.plot(cx, cy, color=C_OBS, lw=1.4, zorder=4)
    # The two entities that own the left-hand knee are dots, identified in the
    # key rather than labelled on the curve. Both sit within one pixel of the y
    # axis, where leadered text crowds the axis and its lines cross the curve.
    ax.plot([cx[0]], [cy[0]], "o", ms=3.0, color=C_SPIKE, zorder=6,
            label="spike (832 antibodies)")
    ax.plot([cx[1]], [cy[1]], "o", ms=3.0, color=C_ENV, zorder=6,
            label="HIV-1 Env (132)")
    ax.legend(loc="lower right", fontsize=5.5, handlelength=0.8,
              handletextpad=0.4, borderpad=0.0, labelspacing=0.3,
              bbox_to_anchor=(1.02, -0.03))
    # The half-the-antibodies point is marked on the x axis under the rule.
    half = int(np.searchsorted(cy, 50.0))
    ax.plot([cx[half], cx[half]], [0, cy[half]], color=GREY_M, lw=0.6,
            ls=(0, (1.6, 1.6)), zorder=2)
    # The explanatory text belongs in the legend. A number on the axis reads as a
    # coordinate; the same number written across the panel reads as a caption.
    # The mark is a label under the axis rather than an extra tick. A tick would
    # add a scale mark at a value that is not part of the scale, and its label
    # would sit hard against the "0". The scale keeps its round ticks; the number
    # is annotation, offset right of its own rule so the two do not collide, in
    # grey at the smallest size this project uses.
    mark = round(cx[half], 1)
    ax.set_xticks([0, 25, 50, 75, 100])
    # Position: 2.5 pt left of the rule, 5 pt below the axis, so that the first
    # digit sits under the dotted line. ha="left", so the offset moves the
    # label's left edge: at 5.5 pt a digit is ~3 pt wide, so -2.5 puts the "6"
    # spanning roughly -2.5 to +0.5 about the rule, centred on it. The y offset
    # of -5 keeps it between the axis and the tick-number row rather than below
    # that row; measured, it clears both "0" and "25" (gaps 8.4 and 4.8 px at
    # -11, and the horizontal gap does not change with the vertical offset).
    ax.annotate(f"{mark:g}%", xy=(mark, 0), xytext=(-2.5, -5),
                textcoords="offset points", fontsize=5.5, color=GREY_D,
                ha="left", va="top", annotation_clip=False)
    ax.set_xlim(0, 100); ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of antigens, most antibodies first")
    ax.set_ylabel("cumulative % of antibodies", fontsize=6.5)

    # ---------------- (iii) how big is this, against what we run --------------
    ax = axes[2]
    col = {"reference": GREY_L, "ours": C_ANTIBODY, "curated": GREY_D,
           "pdb": GREY_M}
    y = np.arange(len(SCALE))[::-1] * 1.0
    ax.set_xscale("log"); ax.set_xlim(40, 120000)
    # Tick labels are written out in full. Matplotlib's 10^n mathtext sets the
    # superscript at 70% of the tick size, 4.9 pt from a 7 pt base, below
    # Nature's 5 pt floor, and a reader of a size chart wants the number itself.
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100", "1,000", "10,000", "100,000"], fontsize=5.8)
    ax.minorticks_off()
    ax.barh(y, [t[2] for t in SCALE], 0.42,
            color=[col[t[3]] for t in SCALE],
            edgecolor=[col[t[3]] for t in SCALE], lw=0.4, zorder=3)
    # Labels sit above each bar. As y tick labels they need a column about as
    # wide as the plot, which runs them across the neighbouring panel.
    for yi, t in zip(y, SCALE):
        ax.text(45, yi + 0.30, t[0], fontsize=5.5, va="bottom", ha="left",
                color=INK, zorder=5)
        ax.text(t[2] * 1.15, yi, f"{t[2]:,} {t[1]}", fontsize=5.5, va="center",
                ha="left", color=MUTED, zorder=5)
    ax.set_yticks([])
    ax.set_ylim(-1.0, len(SCALE) - 0.25)
    ax.set_xlabel("number of structures")
    ax.set_ylabel("data available to each method", fontsize=6.5, labelpad=2)
    # The y axis rule is drawn. The bars start at the axis, so without it the
    # panel has no left edge and the bars float.
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_position(("data", 40))

    # Each sub-plot answers its own question and the legend has a block apiece,
    # so each carries its own letter. There is no file-level caption, because one
    # centred title over three panels would be a title for none of them.
    for ax_, L in zip(axes, "def"):
        figstyle.sub_letter(ax_, L)
    got = fit_save(fig, OUT, "fig2d_task1_composition", (183, 48))

    print(f"\n  roster: {n_ab} antibodies over {n_cl} antigen entities (clusters)")
    for (lo, hi, nm), a, b in zip(BINS, s_ag, s_ab):
        print(f"    {nm.replace(chr(10),' '):<16} antigens {a:5.1f}%  antibodies {b:5.1f}%")
    cx, cy = concentration(D)
    for f in (1, 6, 11, 56, 113):
        print(f"  top {cx[f-1]:5.2f}% of antigens ({f:>3}) hold {cy[f-1]:5.1f}% of antibodies")
    print(f"  panel written at {got[0]:.1f} x {got[1]:.1f} mm")
    return got


def main():
    man, D = build()
    RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES / "composition_per_antigen_full.csv", index=False)
    fig2d(man, D)


if __name__ == "__main__":
    main()
