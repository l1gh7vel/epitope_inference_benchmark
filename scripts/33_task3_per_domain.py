#!/usr/bin/env python3
"""Generate Task III per-domain recall, rank-biserial and bubble panels.

HIV-1 Env contains 300 antibodies: CD4bs 116, V3 99, V2 45, MPER 27
and FP 13. The main panel shows CD4bs and V3 (215/300); the extended panel
shows all five domains. SARS-CoV-2 spike contains 100 antibodies in each of
NTD, RBD and S2, all of which appear in both panels.

Recall is compared with each domain's measured random-score recall using a
one-sided exact binomial test. The threshold is 0.05/32 for eight domains and
four methods. Rank-biserial is reported per antibody.

    PYTHONPATH=scripts python scripts/33_task3_per_domain.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import binomtest

import figstyle
from figstyle import COL2, fit_save, panel_letter
import walle

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "paper_figures"

# --- the paper palette, identical to scripts/12_paper_figures.py -------------
AF3_C, WALLE_C, THIRD_C = "#2C6FAD", "#C1553B", "#7B5EA7"
GREY_D, GREY_M, GREY_L = "#6E6E69", "#9A9A95", "#C7C7C2"
BEPI_C = "#BDBDB8"
INK, MUTED, SURFACE = "#1A1A1A", "#5C5C57", "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    **figstyle.BASE_RC,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.2,
    "legend.handleheight": 0.9, "legend.borderpad": 0.2,
    "legend.labelspacing": 0.3,
})

ARMS = [("HIV-1 Env", "task3"), ("SARS-CoV-2 spike", "task3cov")]
#: domains main carries, per arm. See the module docstring for the derivation.
MAIN_DOMAINS = {"task3": ["CD4bs", "V3"], "task3cov": ["NTD", "RBD", "S2"]}
ALPHA = 0.05 / 32          # Bonferroni over 8 domains x 4 methods; see docstring
LIFT_MAX = 0.6             # opacity saturates here, same on both arms
TINY, SMALL = 5.4, 6.0     # Nature floor is 5.0; nothing here goes below 5.4


def save(fig, name, mm):
    return fit_save(fig, OUT, name, mm)


def _stars(p, alpha=ALPHA):
    """One star or n.s., never ***  (project rule: a star is a verdict)."""
    return "*" if p < alpha else "n.s."


# ------------------------------------------------------------------ loading ---
def walle_ckpt():
    """The project WALLE checkpoint, from config/walle.py -- never hardcoded."""
    return walle.HEADLINE_CHECKPOINT


def arm_table(task):
    """Per-domain recall, chance recall and per-case rank-biserial for one arm.

    Returns (bins, counts, chance, rows) where `rows` is the ordered method
    list: (label, colour, blind?, {domain: (recall, k, n)}, per_case df).
    Order is the paper's order everywhere: antibody-blind first.
    """
    base = PROJECT / "results" / task
    af3 = json.loads((base / "metrics.json").read_text())
    bins = af3["bins"]
    chance = {b["bin"]: b["random_recall"] for b in af3["per_bin"]}
    counts = {b["bin"]: b["n"] for b in af3["per_bin"]}

    wal = {f.parent.name: json.loads(f.read_text())
           for f in sorted((base / "walle").glob("*/metrics.json"))}
    ck = walle_ckpt()
    if not (ck and ck in wal):
        ck = max(wal.items(), key=lambda kv: kv[1]["accuracy"])[0] if wal else None

    spec = [("BepiPred-3.0", "bepipred", BEPI_C, True),
            ("ESMFold2", "esmfold2", THIRD_C, False)]
    if ck:
        spec.append((f"WALLE {ck.replace('-sweep', '')}", f"walle/{ck}",
                     WALLE_C, False))
    spec.append(("AlphaFold3", None, AF3_C, False))

    rows = []
    for label, sub, col, blind in spec:
        d = base / sub if sub else base
        m = json.loads((d / "metrics.json").read_text())
        rec = {}
        for b in m["per_bin"]:
            n = int(b["n"])
            rec[b["bin"]] = (float(b["recall"]), int(round(b["recall"] * n)), n)
        pc = d / "per_case_domain.csv"
        rows.append((label, col, blind, rec,
                     pd.read_csv(pc) if pc.exists() else None))
    return bins, counts, chance, rows, af3


def load_all():
    return {task: arm_table(task) for _t, task in ARMS}


# --------------------------------------------------------- per-domain bars ----
#: Okabe-Ito, the same palette scripts/22_fig4_domain_distribution.py
#: sets for the predicted-domain stack, so a domain is one colour in both.
DOMAIN_COLS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def _misrouting(row, dom, bins):
    """{destination domain: share} for the antibodies of `dom` this method got
    WRONG, largest first. Read from the method's own per-case table.

    Returns {} when the table is absent, which is how a method with no per-case
    output degrades to the plain recall bar rather than to a wrong one.
    """
    pc = row[4]
    if pc is None or "true_bin" not in pc or "pred_bin" not in pc:
        return {}
    sub = pc[pc.true_bin == dom]
    if not len(sub):
        return {}
    wrong = sub[sub.pred_bin != dom].pred_bin.value_counts()
    n = len(sub)
    return {b: wrong[b] / n for b in sorted(wrong.index, key=lambda b: -wrong[b])}


def _domain_axes(ax, dom, n_dom, rows, chance, show_y, ylab, stacked=False,
                 bins=None):
    """One subpanel: recall per method on one domain, with its chance line.

    With `stacked`, the bar does not stop at the recall. The filled bottom
    segment is the recall, and above it the wrong predictions are stacked as
    outlined segments identified by the domain they went to, so the column
    totals 1.0 and reads as "of the n antibodies that really bind this domain,
    where did this method put them".

    The stack shows where the rest of each method's predictions went. ESMFold2
    on V3 is the worked case: 24 of 99 correct, and of the 75 it misses, 36 go
    to CD4bs and 34 to FP. A recall bar alone shows that a method misses; the
    stack shows that each method empties its misses into one sink domain, as
    BepiPred-3.0 does, which is the degeneracy claim.
    """
    x = np.arange(len(rows))
    vals = [r[3].get(dom, (np.nan, 0, 0))[0] for r in rows]
    ax.bar(x, vals, 0.66, color=[r[1] for r in rows], edgecolor=SURFACE,
           linewidth=0.7, zorder=3)
    if stacked:
        for xi, r in zip(x, rows):
            base = r[3].get(dom, (np.nan, 0, 0))[0]
            if np.isnan(base):
                continue
            for dest, frac in _misrouting(r, dom, bins).items():
                # Segments are coloured by destination rather than labelled in
                # the bar. A 7 mm bar cannot hold "CD4bs" at a legible size, and
                # four of them per column would be a wall of type. The palette
                # is the one Figure 4b's predicted-domain stack sets, at reduced
                # opacity so a misroute segment never reads as a method's own
                # bar.
                ax.bar(xi, frac, 0.66, bottom=base,
                       facecolor=DOMAIN_COLS[bins.index(dest) % len(DOMAIN_COLS)]
                       if dest in bins else MUTED,
                       alpha=0.42, edgecolor=INK, linewidth=0.4, zorder=3)
                base += frac

    # that domain's own chance recall rather than 1/n_bins; see the docstring
    ax.axhline(chance[dom], color=INK, lw=0.7, ls=(0, (2.4, 1.6)), zorder=4)

    # one verdict row at a constant height, so the marks read across methods
    # rather than each riding its own bar
    for xi, r in zip(x, rows):
        rec, k, n = r[3].get(dom, (np.nan, 0, 0))
        if np.isnan(rec) or n == 0:
            continue
        # The k/n label sits just above the bar on a white halo. A bar near the
        # chance level puts its label straight on the dashed line (spike RBD
        # 28/100, Env MPER 3/27), and a stroke keeps the line visible either
        # side of the glyphs where a bbox would erase it. A stacked column
        # prints the numerator alone. The denominator is in the panel title
        # ("V3  n = 99"), and "106/116" at 4.6 pt is wider than a 7 mm bar, so
        # four per domain collide with each other and with the segment labels
        # above.
        lab = f"{k}" if stacked else f"{k}/{n}"
        ax.text(xi, rec + 0.015, lab, ha="center", va="bottom",
                fontsize=TINY, color=MUTED, zorder=7,
                path_effects=[pe.withStroke(linewidth=1.8, foreground=SURFACE)])
        if r[2]:                      # blind: at chance by construction, no test
            continue
        p = binomtest(k, n, chance[dom], alternative="greater").pvalue
        ax.text(xi, 1.17, _stars(p), ha="center", va="center", fontsize=SMALL,
                color=INK if p < ALPHA else MUTED, zorder=6)

    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_xticks(x)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if show_y:
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=TINY)
        ax.set_ylabel(ylab, fontsize=SMALL)
    else:
        ax.set_yticklabels([])
    ax.spines["left"].set_bounds(0, 1.0)
    ax.set_title(f"{dom}  n = {n_dom}", loc="center", pad=2.5, fontsize=SMALL)


def _bar_legend(rows):
    """Shared method key. Replaces 4 x 5 rotated tick labels, which do not fit.

    It takes no axes. Handles built on a 1e-6 dummy axes parked at the figure
    origin keep their position in get_tightbbox even though axis("off") hides
    the spines, so every panel's ink box is stretched down to the bottom-left
    corner and fit_save pads around that, leaving 13 mm of empty page under the
    main panel.
    """
    h = [Patch(facecolor=r[1], edgecolor=SURFACE, lw=0.6, label=r[0])
         for r in rows]
    h.append(Line2D([0], [0], color=INK, lw=0.7, ls=(0, (2.4, 1.6)),
                    label="chance recall for that domain"))
    return h


def fig_per_domain(data, which):
    """Recall per domain.

    which='main'     the five main domains on one 183 mm strip, Figure 4d;
    which='task3'    every HIV-1 Env domain, with the misrouting stack;
    which='task3cov' every SARS-CoV-2 spike domain, with the misrouting stack.
    """
    if which == "main":
        grid = [(title, task, MAIN_DOMAINS[task]) for title, task in ARMS]
        # The drawn letter is "d". The file stem (fig4e_...) is an identifier
        # rather than a slot, so the letter follows the figure-slot table, which
        # is the numbering authority; figstyle.fit_save's mismatch guard warns
        # if the two disagree.
        name, mm, letter = "fig4e_task3_per_domain_main", (183, 48), "d"
        ncol = sum(len(g[2]) for g in grid)
        fig = plt.figure(figsize=(COL2, 1.9))
        gs = fig.add_gridspec(1, ncol + 1, wspace=0.30,
                              width_ratios=[1] * 2 + [0.30] + [1] * 3,
                              bottom=0.22, top=0.88, left=0.055, right=0.995)
        # top is 0.88: a larger gap above the panel would be wasted, since the
        # arm title is one line.
        slots = [(0, 0), (0, 1), (0, 3), (0, 4), (0, 5)]
    else:
        # One arm per panel. A 5 + 3 grid over two rows of one 180 x 95 mm
        # figure has no room for the misrouting stack, in which every column
        # carries up to four labelled segments.
        # `which` is the task name here: 'task3' -> HIV-1 Env, panel b;
        # 'task3cov' -> SARS-CoV-2 spike, panel c.
        title = next(t for t, task in ARMS if task == which)
        grid = [(title, which, data[which][0])]
        name = f"figE_task3_per_domain_{'hiv' if which == 'task3' else 'cov'}"
        letter = "b" if which == "task3" else "c"
        ndom = len(data[which][0])
        mm = (180, 62)
        fig = plt.figure(figsize=(COL2, 2.5))
        gs = fig.add_gridspec(1, ndom + 2, wspace=0.30,
                              width_ratios=[1] * ndom + [0.25, 1.9],
                              bottom=0.10, top=0.90, left=0.075, right=0.995)
        slots = [(0, i) for i in range(ndom)]

    rows_ref = None
    bins_ref = []
    k = 0
    for title, task, doms in grid:
        bins, counts, chance, rows, _af3 = data[task]
        rows_ref = rows
        bins_ref = bins
        for j, dom in enumerate(doms):
            r, c = slots[k]; k += 1
            ax = fig.add_subplot(gs[r, c])
            first = j == 0
            _domain_axes(ax, dom, counts[dom], rows, chance, first,
                         f"{title}\nshare of n",
                         stacked=which != "main", bins=bins)
            if k == 1:
                panel_letter(ax, letter)

    # Shared method key. It replaces 4 rotated tick labels under every subpanel,
    # 20 of them in the main layout. In the 'all' layout it occupies the two
    # empty slots of row 2; in the one-row main layout it is a fig.legend,
    # because a dedicated legend axes there leaves a 15 mm band of empty page
    # between the bars and the key.
    h = _bar_legend(rows_ref)
    if which == "main":
        fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.18),
                   ncol=5, fontsize=TINY, handlelength=1.3, columnspacing=1.1,
                   borderpad=0.2)
    else:
        # The misroute segments are identified by colour, so the key names the
        # domains rather than the device.
        h.append(Patch(facecolor="none", edgecolor="none",
                       label="misses went to:"))
        h += [Patch(facecolor=DOMAIN_COLS[i % len(DOMAIN_COLS)], alpha=0.42,
                    edgecolor=INK, lw=0.4, label=b)
              for i, b in enumerate(bins_ref)]
        lax = fig.add_subplot(gs[0, -1]); lax.axis("off")
        lax.legend(handles=h, loc="center left", ncol=1, fontsize=TINY,
                   handlelength=1.3, borderpad=0.2)
    return save(fig, name, mm)


# ----------------------------------------------- per-domain rank-biserial -----
def fig_per_domain_rankbis(data):
    """Per-case domain rank-biserial, restricted to each domain's antibodies.

    The continuous counterpart of the recall panel. Read the shape and the mass
    below zero rather than the median. The statistic rewards a degenerate
    predictor asymmetrically, since a method that always answers one domain is
    strongly positive on that domain's cases and strongly negative on every
    other, so a tall positive violin in one column with deep negatives elsewhere
    is the signature of degeneracy rather than of a method that is good at one
    domain.
    """
    fig = plt.figure(figsize=(COL2, 3.8))
    gs = fig.add_gridspec(2, 5, wspace=0.30, hspace=0.34,
                          bottom=0.07, top=0.96, left=0.075, right=0.995)
    slots = [(0, i) for i in range(5)] + [(1, i) for i in range(3)]
    rng = np.random.default_rng(0)
    k = 0
    rows_ref = None
    for title, task in ARMS:
        bins, counts, chance, rows, _af3 = data[task]
        rows_ref = rows
        bins_ref = bins
        for j, dom in enumerate(bins):
            r, c = slots[k]; k += 1
            ax = fig.add_subplot(gs[r, c])
            for xi, (lab, col, blind, _rec, pc) in enumerate(rows):
                if pc is None:
                    continue
                v = pc.loc[pc.true_bin == dom, "domain_rankbis"].dropna().values
                if not len(v):
                    continue
                if len(np.unique(v)) > 1:
                    parts = ax.violinplot([v], positions=[xi], widths=0.72,
                                          showextrema=False, showmedians=False)
                    b = parts["bodies"][0]
                    b.set_facecolor(col); b.set_alpha(0.30)
                    b.set_edgecolor(col); b.set_linewidth(0.6)
                ax.plot(xi + rng.uniform(-0.13, 0.13, len(v)), v, ".", ms=1.1,
                        color=col, alpha=0.45, zorder=3)
                ax.plot([xi - 0.30, xi + 0.30], [np.median(v)] * 2, color=col,
                        lw=1.5, zorder=4)
            ax.axhline(0, color=INK, lw=0.7, zorder=2)
            ax.set_ylim(-1.15, 1.15)
            ax.spines["left"].set_bounds(-1, 1)
            # No bottom spine. There are no x tick labels, since the key names
            # the methods, so it carries nothing, and at ylim -1.15 it is a
            # detached rule 0.15 below the -1 tick that reads as a second axis.
            ax.spines["bottom"].set_visible(False)
            ax.set_xlim(-0.65, len(rows) - 0.35)
            ax.set_xticks(range(len(rows)))
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if j == 0:
                ax.set_yticks([-1, 0, 1], ["-1", "0", "1"], fontsize=TINY)
                ax.set_ylabel(f"{title}\ndomain rank-biserial", fontsize=TINY)
            else:
                ax.set_yticks([-1, 0, 1]); ax.set_yticklabels([])
            ax.set_title(f"{dom}  n = {counts[dom]}", loc="center", pad=2.5,
                         fontsize=SMALL)
            if k == 1:
                panel_letter(ax, "d")   # panel d of Extended Data 2
    lax = fig.add_subplot(gs[1, 3:5]); lax.axis("off")
    h = [Patch(facecolor=r[1], edgecolor=r[1], alpha=0.45, lw=0.6, label=r[0])
         for r in rows_ref]
    h.append(Line2D([0], [0], color=INK, lw=0.7, label="no preference (0)"))
    lax.legend(handles=h, loc="center left", ncol=1, fontsize=TINY,
               handlelength=1.3, borderpad=0.2)
    return save(fig, "figE_task3_per_domain_rankbis", (180, 95))


# ------------------------------------------------------------- bubble grid ----
def fig_bubbles(data):
    """Domains x methods in one grid. Area = recall, opacity = lift over chance.

    Both channels are named under the grid, because a bubble whose size and
    opacity each mean something is unreadable the moment the reader has to guess
    which is which.
    """
    # rows: Env's 5 domains, a spacer, spike's 3. Columns are identical across
    # arms so a method reads as one vertical strip over both antigens.
    row_labels, row_y, row_arm = [], [], []
    y = 0.0
    arm_span = {}
    for title, task in ARMS:
        bins = data[task][0]
        y0 = y
        for b in bins:
            row_labels.append(b); row_y.append(y); row_arm.append(task)
            y += 1.0
        arm_span[task] = (y0, y - 1)
        y += 0.9                              # gap between arms

    # columns: two blind floors, then the methods (BepiPred is itself blind)
    _b, _c, _ch, rows0, _a = data["task3"]
    cols = [("Random\nscores", GREY_L, True), ("Always\ncommonest", GREY_D, True)]
    cols += [(r[0].replace("WALLE ", "WALLE\n"), r[1], r[2]) for r in rows0]
    n_blind = 2 + sum(1 for r in rows0 if r[2])

    # The grid and the key are separate axes at fixed figure fractions. A key
    # axes that runs under the grid prints its size swatches on top of the S2
    # row and carries the shaded antibody-blind band down into the caption.
    fig = plt.figure(figsize=(COL2 * 0.66, 3.0))
    # left is 0.105: a larger margin would hold room for a y label that is short.
    ax = fig.add_axes([0.105, 0.275, 0.890, 0.665])

    for ri, (dom, yy, task) in enumerate(zip(row_labels, row_y, row_arm)):
        bins, counts, chance, rows, af3 = data[task]
        maj = af3["floor_majority_bin"]
        series = [("Random\nscores", GREY_L, chance[dom]),
                  ("Always\ncommonest", GREY_D, 1.0 if dom == maj else 0.0)]
        series += [(r[0], r[1], r[3].get(dom, (np.nan,))[0]) for r in rows]
        for ci, (_lab, col, rec) in enumerate(series):
            if rec is None or (isinstance(rec, float) and np.isnan(rec)):
                continue
            lift = rec - chance[dom]
            if rec <= 0:
                # a fixed hairline ring marks a measured zero (not a missing value)
                ax.plot(ci, -yy, "o", ms=1.9, mfc="none", mec=col, mew=0.7,
                        zorder=3)
                continue
            s = 8.0 + 190.0 * rec           # area proportional to recall
            if lift <= 0:
                ax.scatter([ci], [-yy], s=s, facecolors="none", edgecolors=col,
                           linewidths=0.8, zorder=3)
            else:
                a = 0.28 + 0.72 * min(lift, LIFT_MAX) / LIFT_MAX
                ax.scatter([ci], [-yy], s=s, color=col, alpha=a,
                           edgecolors="none", zorder=3)

    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(-max(row_y) - 0.45, 0.9)
    # drawn after the limits so the band spans the grid and stops there
    ax.axvspan(-0.6, n_blind - 0.5, color=GREY_L, alpha=0.18, zorder=0, lw=0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=TINY)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks([-v for v in row_y])
    ax.set_yticklabels(row_labels, fontsize=TINY)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.55, 0.70, "antibody-blind", fontsize=TINY, color=MUTED,
            ha="left", va="center", zorder=6)

    # arm names down the left, outside the domain labels
    for title, task in ARMS:
        y0, y1 = arm_span[task]
        ax.text(-1.75, -(y0 + y1) / 2, title, fontsize=TINY, color=INK,
                rotation=90, ha="center", va="center")

    # ---- the key. Both channels are named; neither is left to be inferred ----
    # The key sits in the bottom right rather than spanning the full width.
    kax = fig.add_axes([0.42, 0.0, 0.575, 0.215]); kax.axis("off")
    kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    # Each channel is named DIRECTLY ABOVE its own swatches. The first version
    # put both captions under both swatch rows, which left the reader to work
    # out which caption belonged to which group -- the one thing a two-channel
    # bubble panel must not ask.
    kax.text(0.015, 0.99, "area = recall", fontsize=TINY, va="top", color=INK)
    for i, rv in enumerate((0.25, 0.5, 1.0)):
        kax.scatter([0.055 + 0.070 * i], [0.62], s=8.0 + 190.0 * rv,
                    color=GREY_D, alpha=0.55, edgecolors="none")
        kax.text(0.055 + 0.070 * i, 0.36, f"{rv:g}", fontsize=TINY,
                 ha="center", va="top", color=MUTED)
    kax.text(0.345, 0.99, "fill opacity = recall \u2212 chance recall",
             fontsize=TINY, va="top", color=INK)
    for i, (lv, lab) in enumerate(((0.05, "0.05"), (0.3, "0.3"), (0.6, "\u22650.6"))):
        a = 0.28 + 0.72 * lv / LIFT_MAX
        kax.scatter([0.385 + 0.070 * i], [0.62], s=110, color=GREY_D, alpha=a,
                    edgecolors="none")
        kax.text(0.385 + 0.070 * i, 0.36, lab, fontsize=TINY, ha="center",
                 va="top", color=MUTED)
    kax.scatter([0.625], [0.62], s=110, facecolors="none", edgecolors=GREY_D,
                linewidths=0.8)
    kax.text(0.625, 0.36, "\u2264 chance", fontsize=TINY, ha="center", va="top",
             color=MUTED)
    # No prose lines here; they belong in the legend. The definitions of recall
    # and chance recall, and the note that spike's balance makes the
    # majority-class domain a tie-break, are sentences, so they belong with the
    # typeset legend.
    #
    # THE SWATCHES STAY, and the distinction is the point: a size and opacity
    # key is a KEY, not prose. Bubble area and fill opacity cannot be decoded
    # from a sentence -- a reader has to compare a drawn bubble against a drawn
    # reference -- so Nature's "keys in the legend where possible" does not
    # reach them. The open ring is in the same class: it encodes a verdict
    # (at or below chance) that has no size or opacity.
    panel_letter(ax, "e")          # slot fig4e; see the note on the main panel
    return save(fig, "fig4f_task3_domain_bubbles", (124, 76))


# ----------------------------------------------------------------- console ----
def report(data):
    """The per-domain table, printed so the panels can be checked against it."""
    print("\nper-domain recall vs that domain's own chance recall "
          f"(one-sided exact binomial, Bonferroni alpha = {ALPHA:.4f})")
    for title, task in ARMS:
        bins, counts, chance, rows, af3 = data[task]
        print(f"\n  {title}   n = {af3['n']}   pooled accuracy "
              f"{af3['accuracy']:.3f}  (majority floor {af3['floor_majority']:.3f})")
        hdr = f"    {'domain':7s} {'n':>4s} {'chance':>7s}"
        print(hdr + "".join(f" {r[0].split()[0][:11]:>19s}" for r in rows))
        for b in bins:
            cells = []
            for r in rows:
                rec, k, n = r[3].get(b, (np.nan, 0, 0))
                if n == 0:
                    cells.append(f"{'--':>19s}"); continue
                p = binomtest(k, n, chance[b], alternative="greater").pvalue
                mark = "*" if (p < ALPHA and not r[2]) else (
                    " " if r[2] else ("v" if rec < chance[b] else " "))
                cells.append(f"{rec:.2f} ({k:3d}/{n:3d}){mark:>2s}".rjust(19))
            print(f"    {b:7s} {counts[b]:4d} {chance[b]:7.3f}" + "".join(cells))
    print("\n    * beats that domain's chance recall at alpha = 0.05/32")
    print("    v BELOW that domain's chance recall")


def main():
    figstyle.check_font()
    print(f"WALLE checkpoint: {walle_ckpt()}")
    data = load_all()
    report(data)
    print(f"\nwriting panels to {OUT}")
    fig_per_domain(data, "main")
    fig_per_domain(data, "task3")
    fig_per_domain(data, "task3cov")
    fig_per_domain_rankbis(data)
    fig_bubbles(data)


if __name__ == "__main__":
    main()
