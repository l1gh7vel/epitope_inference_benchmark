#!/usr/bin/env python3
"""Generate Figure 1b escape profiles for PG9, 10-1074 and 10E8.

Blue marks structural contacts, green marks contacts that are also significant
escape sites, and yellow marks escape sites outside the structural epitope.
Tracks use BG505.T332N Env positions and antibody-specific y-axis scales.

    PYTHONPATH=scripts:config python scripts/24_fig1b_escape_tracks.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
import figstyle                                            # noqa: E402
from figstyle import fit_save                              # noqa: E402

OUT = PROJECT / "results" / "paper_figures"
MAN = PROJECT / "data" / "manifests"

INK, MUTED, SURFACE = "#1A1A1A", "#5C5C57", "#FFFFFF"
# The source figure's own three-way key, in colour-vision-safe equivalents.
CONTACT, BOTH, ESCAPE = "#0072B2", "#009E73", "#E69F00"
LINE = "#4A4A46"

SHOW = [("PG9", "V1V2 apex"), ("10-1074", "V3-glycan"), ("10E8", "MPER")]
# Ticks sit at round HXB2 numbers. A reader of an Env figure navigates by round
# site numbers and by the landmarks (V3 is ~296-331, MPER ~660-683). Letting the
# array index drive the ticks puts them at 31, 184, 332, 483, 633, which carry
# no meaning in HXB2 numbering.
XTICKS = [100, 200, 300, 400, 500, 600, 700]

# Per-track x window. Each track has its own x axis, zoomed to a range set by
# hand rather than derived from the data. The three ranges are of similar width,
# so the rows sit at comparable magnification and one row's peak cannot be made
# to look wider than another's:
#
#     PG9       25-350   (325 sites)   marked region 156-174
#     10-1074  150-500   (350 sites)   marked region 204-441
#     10E8     450-710   (260 sites)   marked region 609-683
#
# Each window carries ~300 sites of context rather than ~90, so a 17-site region
# such as PG9's is not magnified until baseline noise looks like signal.
#
# The cost is that the three rows do not share a coordinate system, so the panel
# cannot be read as three antibodies on one map. The legend says in words that
# the targets are disjoint.
WINDOWS = [(25, 350), (150, 500), (450, 710)]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, **figstyle.BASE_RC,
    "axes.edgecolor": INK, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})


def main():
    figstyle.check_font()
    lab = np.load(MAN / "dms_labels.npz")
    fun = np.load(MAN / "dms_functional_sites.npz")
    esc = np.load(MAN / "dms_escape.npz")

    # sharex=False, because the three rows cover different ranges. hspace is
    # 1.05 to clear the tick labels and bottom spine that every row carries.
    fig, axes = plt.subplots(len(SHOW), 1, figsize=(3.3, 1.95), sharex=False,
                             gridspec_kw={"hspace": 1.05})
    for r, (ab, cls) in enumerate(SHOW):
        ax = axes[r]
        v = esc[f"{ab}|Dingens2019_BG505|mean"].astype(float)
        meas = np.isfinite(v)
        pos = np.arange(1, len(v) + 1)
        g = lab[f"{ab}__gold"].astype(bool)          # structural contact
        s = fun[f"{ab}__escape"].astype(bool)        # significant escape

        top = float(np.nanmax(v))
        ax.plot(pos[meas], v[meas], color=LINE, lw=0.45, zorder=3)

        # Markers below the baseline, as the source figure does it. The three
        # classes share one lane, with the colour changing site by site. They
        # are mutually exclusive, verified at zero sites in more than one, so
        # one lane shows all three without overlap. PG9's contact-only sites
        # (156-158, 165-167, 169, 171) and its both-sites (159, 161, 168, 170)
        # interleave along the sequence, so separate bands would draw a block of
        # blue above a block of green in the same x region and suggest the same
        # residue was in two classes.
        #
        # Each contiguous run is one bar in data coordinates, so a mark is as
        # wide as the residues it covers at any zoom: a run of one stays one
        # site wide, a run of ten looks ten sites long. A per-site vline instead
        # carries a fixed stroke width in points, which has nothing to do with
        # how wide one site is in data units. On a 325-site window at 90 mm one
        # site is ~0.2 pt, so a 0.6 pt stroke overdraws its own site and still
        # leaves a gap to the next one, and sites 156-158 render as three
        # separate ticks when they are one stretch of sequence.
        y0 = -top * 0.155
        h = top * 0.125
        PAD = 0.60                 # sites either side of the run
        for mask, col in ((g & ~s, CONTACT), (g & s, BOTH), (s & ~g, ESCAPE)):
            q = pos[mask]
            if len(q):
                # split into maximal runs of consecutive sites
                brk = np.flatnonzero(np.diff(q) > 1) + 1
                spans = [(r[0] - PAD, (r[-1] - r[0]) + 2 * PAD)
                         for r in np.split(q, brk) if len(r)]
                ax.broken_barh(spans, (y0 - h / 2, h), facecolors=col,
                               edgecolors="none", clip_on=True, zorder=5)

        ax.set_ylim(-top * 0.30, top * 1.15)
        ax.set_yticks([0, round(top, 3)])
        ax.tick_params(labelsize=5.0, pad=1.2)
        # Every row draws its own bottom spine and its own tick labels, because
        # every row is a different stretch of Env.
        lo, hi = WINDOWS[r]
        ax.set_xlim(lo, hi)
        ax.set_xticks([t for t in XTICKS if lo < t <= hi])
        ax.spines["bottom"].set_visible(True)
        ax.tick_params(bottom=True, labelbottom=True, labelsize=5.0, pad=1.2)
        ax.text(0.004, 1.02, ab, transform=ax.transAxes, fontsize=6.0,
                color=INK, va="bottom", ha="left")
        ax.text(0.16, 1.02, cls, transform=ax.transAxes, fontsize=5.5,
                color=MUTED, va="bottom", ha="left")

    # One x label, on the bottom row. The three windows differ but the quantity
    # they plot does not, so one label covers all three rows.
    axes[-1].set_xlabel("HIV-1 Env site (HXB2 numbering)", fontsize=6.0, labelpad=1.5)
    fig.supylabel("excess fraction surviving", fontsize=6.0, x=0.012)

    # The key is three words against three swatches, so it is direct-labelled in
    # the band above the top track rather than boxed.
    ax = axes[0]
    for x, col, s in [(0.40, CONTACT, "contact"), (0.60, BOTH, "both"),
                      (0.76, ESCAPE, "escape")]:
        ax.plot([x], [1.055], marker="s", ms=1.8, lw=0, color=col,
                transform=ax.transAxes, clip_on=False, zorder=6)
        ax.text(x + 0.022, 1.055, s, transform=ax.transAxes, fontsize=5.5,
                color=col, va="center", ha="left")
    # No panel letter. This panel is the lower half of Figure 1b, and the bold
    # "b" is drawn on the escape-workflow schematic above it.
    fit_save(fig, OUT, "fig1b_escape_tracks", (90, 52))


if __name__ == "__main__":
    main()
