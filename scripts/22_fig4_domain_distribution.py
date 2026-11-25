#!/usr/bin/env python3
"""Plot true and predicted Task III domain distributions.

    PYTHONPATH=scripts:config python scripts/22_fig4_domain_distribution.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
import figstyle                                            # noqa: E402
from figstyle import COL2, fit_save, panel_letter          # noqa: E402

_s = importlib.util.spec_from_file_location("pf", PROJECT / "scripts" / "12_paper_figures.py")
pf = importlib.util.module_from_spec(_s); _s.loader.exec_module(pf)

OUT = pf.OUT
INK, MUTED, SURFACE = "#1A1A1A", "#5C5C57", "#FFFFFF"
GREY_L = "#C7C7C2"

# Okabe-Ito, which stays separable under the common forms of colour vision
# deficiency. The domains are a categorical axis with no natural order, and none of
# these is a method colour, so a reader cannot mistake a domain for a model. This
# sets the domain palette for the paper: the confusion matrices encode recall on a
# Blues ramp and the accuracy panel uses method colours. The fifth colour is
# #D55E00; #56B4E9 sits too close to #0072B2 in adjacent bars.
DOMAIN_COLS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, **figstyle.BASE_RC,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": False, "legend.frameon": False,
})


def build(arms, ckpt=None):
    fig, axes = plt.subplots(1, len(arms), figsize=(COL2 * 0.95, 1.95),
                             squeeze=False, gridspec_kw={"wspace": 0.44})
    for k, (title, af3, wal, extra) in enumerate(arms):
        ax = axes[0][k]
        roster, _ = pf.t3_methods(pf.TASK_OF[title], af3, wal, extra, ckpt)
        bins = af3["bins"]
        cols = DOMAIN_COLS[:len(bins)]

        # "true" first, then each method. The true distribution is the reference a
        # degenerate predictor has to be read against, so it is the leftmost bar.
        labels, dists, lab_cols = ["true"], [], [INK]
        conf0 = np.array([[af3["confusion"][t][p] for p in bins] for t in bins], float)
        dists.append(conf0.sum(axis=1) / conf0.sum())          # row sums = true
        for lab, m, _pc, _col, _b in roster:
            c = np.array([[m["confusion"][t][p] for p in bins] for t in bins], float)
            labels.append(pf.short_label(lab).replace("\n", " "))
            lab_cols.append(_col)
            dists.append(c.sum(axis=0) / c.sum())              # col sums = predicted

        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        D = np.array(dists)
        for d, (b, col) in enumerate(zip(bins, cols)):
            vals = D[:, d]
            # Segments carry no edge, and the white keylines that separate them
            # are drawn as explicit horizontal rules. An edge on the bottom
            # segment falls half below y = 0 and sits on the axis spine, which
            # reads as the bar overlapping its own axis.
            ax.bar(x, vals, 0.70, bottom=bottom, color=col, alpha=1.0,
                   edgecolor="none", linewidth=0, zorder=3)
            if d:                     # skipped for the first segment; rule along its base
                ax.hlines(bottom, x - 0.35, x + 0.35, color=SURFACE, lw=0.6,
                          zorder=4)
            # Segments carry no labels. Nature's guidance asks that reversed
            # type (white on a colour) be "avoided where possible", and the same
            # for lettering over shaded areas; 5.5 pt type sits 0.5 pt above
            # their 5 pt floor, and they reduce figures at production. Writing
            # into segments >= 16% tall would also leave a thin segment's
            # identity carried by colour alone. The key labels every domain,
            # including the ones too thin to write in.
            bottom += vals
        # The key is in the artwork. Nature's guide asks for keys in the figure
        # "rather than having colour descriptions in the figure caption, as
        # readers with colour blindness can struggle to interpret these". A
        # coloured swatch beside black text is the pattern it calls Recommended.
        handles = [Patch(facecolor=c, edgecolor="none") for c in cols]
        ax.legend(handles, list(bins), loc="center left",
                  bbox_to_anchor=(1.005, 0.5), fontsize=6.0, handlelength=0.9,
                  handleheight=0.9, handletextpad=0.4, labelspacing=0.32,
                  borderpad=0.0, frameon=False)
        # The bottom spine is drawn over the bars. The bars sit at zorder 3 and
        # the axis spine defaults below them, so a filled segment paints over the
        # black rule at y = 0 and the bar reads as overlapping its own axis.
        ax.spines["bottom"].set_zorder(6)
        # The true bar is the reference, not a result; rule it off from the methods.
        ax.axvline(0.5, color=INK, lw=0.7, zorder=4)
        ax.set_xticks(x, labels, fontsize=6, rotation=45, ha="right",
                      rotation_mode="anchor")
        # Method names are black. Nature's research figure guide lists "Coloured
        # text; keylines, keys, etc. should be used instead" under "We avoid",
        # and says plainly "Avoid coloured text"; where colour is not necessary
        # for meaning they recolour text to black at production. AlphaFold2's
        # Nature Fig. 1a is the same chart shape, method names under coloured
        # bars, and sets every label black. Black is also the legible choice
        # here, because BepiPred-3.0's colour is a near-white grey marking an
        # antibody-blind floor rather than a model. The labels also carry no
        # path effect, because in matplotlib a path-effect stroke converts the
        # glyphs to vector outlines, which strips them from the PDF text layer
        # and defeats the pdf.fonttype=42 setting figstyle.py enforces.
        del lab_cols
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0])
        if k == 0:
            ax.set_ylabel("share of antibodies\nassigned", fontsize=6.5)
        ax.set_title(f"{title}, n = {af3['n']}", fontsize=6.5, color=INK, pad=3)
        if k == 0:
            panel_letter(ax, "b")
    fit_save(fig, OUT, "fig4b_task3_domain_distribution", (183, 42))


if __name__ == "__main__":
    figstyle.check_font()
    arms = []
    for name, task in [("HIV-1 Env", "task3"), ("SARS-CoV-2 spike", "task3cov")]:
        r = pf.load_t3(task)
        if r:
            arms.append((name, r[0], r[1], r[2]))
    _pc, wal, _sw = pf.load_task2()
    build(arms, pf.t3_ckpt(wal))
