"""Plot recall of experimentally significant viral escape sites.

Predicted residues are ranked by score; curves show recall through rank k.
Where a structural complex is available, the fixed structural epitope is shown
at its size-matched k. HIV-1 uses nine antibody panels; RSV and Nipah use one
row in the configured antibody order.

    EIB_TASK=dms   PYTHONPATH=scripts:config python scripts/dms_09_binary_cost_figure.py
    EIB_TASK=rsv_f PYTHONPATH=scripts:config python scripts/dms_09_binary_cost_figure.py
    EIB_TASK=niv   PYTHONPATH=scripts:config python scripts/dms_09_binary_cost_figure.py
"""
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np, pandas as pd
from matplotlib.lines import Line2D

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "scripts"))

# The WALLE checkpoint is resolved in ONE place, dms_08, which is also what wrote
# the curves this reads; importing it keeps the label on the legend and the rows
# in the csv from ever naming different checkpoints.
from dms_08_binary_cost import walle_label                  # noqa: E402

# Which arm. Read the same way dms_08 reads it, so the two can never disagree
# about which tables are being drawn.
ARM = os.environ.get("EIB_TASK", "dms")
RES_REL = {"dms": "dms_case", "rsv_f": "rsv_case", "niv": "niv_case"}
if ARM not in RES_REL:
    raise SystemExit(f"EIB_TASK={ARM!r}: this script knows {sorted(RES_REL)}")
# Both follow the variant: an alt run reads the curves that run's dms_08 wrote and
# writes its panels beside them, leaving results/ exactly as the default left it.
RES = PROJECT / "results" / RES_REL[ARM]
OUT = PROJECT / "results" / "paper_figures"

import figstyle                                            # noqa: E402
from figstyle import COL2, fit_save                        # noqa: E402
AF3_C, WALLE_C, THIRD_C = "#2C6FAD", "#C1553B", "#7B5EA7"
BEPI_C, GREY_L = "#B4B4AE", "#C7C7C2"
INK, MUTED, SURFACE = "#1A1A17", "#6E6E69", "#FFFFFF"
COL = {"AlphaFold3": AF3_C, "ESMFold2": THIRD_C, "WALLE": WALLE_C,
       "BepiPred-3.0": BEPI_C, "Random": GREY_L}
ORDER = ["BepiPred-3.0", "ESMFold2", "WALLE", "AlphaFold3"]

# Two antibodies are dropped from the panel. The grid is ordered by hypergeometric
# fold enrichment descending, so "the last two" is well defined:
#
#     VRC01     fold  2.698   P = 0.320   5 escape sites   CD4bs
#     3BNC117   fold  1.520   P = 0.494  15 escape sites   CD4bs
#
# THE CONSEQUENCE HAS TO BE STATED WHEREVER THIS PANEL APPEARS, because dropping
# them biases the panel. They are the ONLY TWO of the nine whose escape sites are
# NOT significantly concentrated inside the structural epitope. That concentration
# is the premise the whole DMS arm rests on, so removing the two cases where it
# fails makes the premise look stronger than the data supports. The panel
# shows the seven antibodies for which the premise
# holds and does not show "the nine".
#
# Extended Data Fig. 3a draws all nine. It is the panel that
# shows WHY these two fail -- their escape sites sit visibly away from their
# contacts -- and both are CD4-binding-site antibodies with large epitopes and
# few escape sites, which is a property of the biology rather than of our gold
# epitopes. Dropping them there too would hide the reason.
#
# PER ARM. RSV drops palivizumab and suptavumab. Neither has a deposited F complex -- palivizumab has only the Fab alone,
# 2HWZ, and suptavumab none -- so neither has a gold structural epitope and
# neither had a dashed reference line. They are also the two that WALLE cannot be
# run on from a deposited structure, so dropping them is what lets 5c run WALLE
# on experimental input for every antibody it draws. Their escape data is not
# lost: it is in results/rsv_case/.
DROP_ANTIBODIES_BY_ARM = {
    "dms": ("VRC01", "3BNC117"),
    # 1A2 IS EXCLUDED as well, because its gold epitope captures few of its
    # escape sites, so there is little point showing it. Its declared escape sites do not sit on its own solved
    # contacts -- 1 of 10 overlap, fold 2.1, P = 0.39 -- so its gold line is at
    # 0.1 and there is no ceiling for a method to be measured against.
    "rsv_f": ("Palivizumab", "Suptavumab", "1A2"),
    # NIPAH KEEPS ONLY THE TWO ANTIBODIES WITH A GENUINE NIPAH COMPLEX
    # HENV-32 (6VY4) and m102.4 (6CMG) have no
    # deposited Nipah structure at all -- both are cross-reactive and their only
    # solved complexes are with HENDRA G, measured at 75.1% and 77.1% antigen
    # identity to Nipah, so their gold epitopes are mapped across about a
    # quarter of the sequence. m102.4 carried two further compromises: 6CMG is
    # the clone m102.3, and its light chain in our fold is a proxy differing at
    # 15 positions. nAH1.3 (7TXZ, 99.8%) and HENV-26 (6VY5, 100.0%) keep theirs.
    "niv": ("HENV-32", "m102.4"),
}
DROP_ANTIBODIES = DROP_ANTIBODIES_BY_ARM.get(ARM, ())

# ------------------------------------------- template-availability marks ----
# The legend and the figure both mark which complexes AlphaFold3 may have seen
# (well studied, deposited early) and which are very recent.
#
# WHAT THE MARK MEANS, exactly. AlphaFold3's `--max_template_date` is left at its
# 2021-09-30 default, and leakage is reported rather than suppressed, so a
# complex DEPOSITED BEFORE that date was available to it as a template and one
# deposited after was not. The mark is about the STRUCTURE's release date. It is
# not a claim about the training set, which we cannot see -- only about what the
# template search could have returned.
#
# FILLED IS THE "SEEN" CASE, OPEN THE RECENT ONE, which matches the open-ring
# convention the Task III bubble panel already uses for the weaker verdict.
#
# This matters most on RSV. Its four antibodies split two and two. At k = 50
# the two pre-cutoff antibodies reach 0.7-0.9 recall with both co-folders, on
# their gold ceiling; the two from a 2026 structure reach 0.0-0.2 against a
# random floor of 0.10. The two pre-cutoff antibodies are also the two clinical
# ones, so deposition date and clinical status are confounded on this arm.
# Squares rather than circles. A filled circle is already taken on this panel: it is
# the dot on the gold line marking k equal to the epitope's size, and two
# filled circles meaning different things in one key is the kind of thing a
# reader only notices after misreading it. Liberation Sans has the filled and
# open squares and not the diamonds, so diamonds are not used.
SEEN, RECENT = "\u25a0", "\u25a1"          # filled square, open square
#: antibody -> PDB id and release date of the complex the gold epitope comes
#: from. HIV's seven are uniformly pre-cutoff, which is itself worth showing.
TEMPLATE_DATE = {
    "dms": {"PG9": ("3U2S", "2011-10-12"), "10E8": ("4G6F", "2012-09-12"),
            "PGT145": ("5V8L", "2017-06-14"), "VRC34": ("6NC3", "2019-01-16"),
            "10-1074": ("6CCB", "2018-03-07"), "PGT121": ("4TVP", "2014-11-05"),
            "PGT151": ("6DCQ", "2018-06-20"),
            "VRC01": ("3NGB", "2010-08-11"), "3BNC117": ("4JPV", "2013-05-15")},
    "rsv_f": {"Nirsevimab": ("5UDC", "2017-04-12"),
              "Clesrovimab": ("6OUS", "2019-07-24"),
              "1A2": ("9LLY", "2026-01-21"), "1B6": ("9LLY", "2026-01-21")},
    "niv": {"nAH1.3": ("7TXZ", "2022-03-09"), "HENV-26": ("6VY5", "2021-01-06"),
            "HENV-32": ("6VY4", "2020-12-30"), "m102.4": ("6CMG", "2018-04-18")},
}
AF3_TEMPLATE_CUTOFF = "2021-09-30"


def template_mark(ab):
    """SEEN / RECENT glyph for an antibody, or "" if we have no date for it."""
    spec = TEMPLATE_DATE.get(ARM, {}).get(ab)
    if not spec:
        return ""
    return SEEN if spec[1] < AF3_TEMPLATE_CUTOFF else RECENT
# Extended Data slot: half an Extended Data page, which is 180 x 170 mm.
ED = (180, 83)

# THE VIRUS BLOCKS OF FIGURE 5. One row of antibodies each, both at the same
# height so 5c and 5d stack evenly under their BioRender strips.
#
# WHY A ROW AND NOT A GRID, in millimetres. Figure 5 is three virus blocks under
# one shared 183 x 30 mm header; each block also carries its own 183 x 22 mm
# strip. That is 30 + 3 x 22 = 96 mm of schematic before a single curve is drawn,
# against a ~190 mm target, which leaves ~94 mm for three result panels. A row is
# what fits. The cost is column width -- 30.5 mm on RSV, 36.6 mm on Nipah -- which
# is why the title is three short lines.
#
# The key's position. Nipah has four antibodies in a five-slot row, so its spare
# slot takes the key at no cost in height, exactly as the HIV panel's tenth slot
# does. RSV fills all six of its slots, so it carries no key: one key for the
# figure, in the block that has room for it. Override with --key / --no-key.
VIRUS_PANEL = {
    # FOUR antibodies in a FIVE-slot row since the two ungolded ones were
    # dropped, so RSV now has a spare slot and takes the key. Nipah keeps its
    # own key; whether Figure 5 ships one key or three is a layout choice.
    "rsv_f": dict(stem="fig5c_rsv_escape_recall", ncol=4, mm=(183, 42), key=True,
                  ylab="escape sites captured"),
    # TWO antibodies, so a 3-slot row: two panels and the key.
    "niv": dict(stem="fig5d_niv_escape_recall", ncol=3, mm=(183, 42), key=True,
                ylab="escape sites captured"),
}

# DECLARED panel order, fixed and independent of any result. See the module
# docstring. None means "sort by gold fold enrichment descending", which is what
# the HIV grid has always done and which an arm with ungolded antibodies cannot.
PANEL_ORDER = {
    "dms": None,
    "rsv_f": ["Nirsevimab", "Clesrovimab", "Palivizumab", "Suptavumab", "1A2", "1B6"],
    "niv": ["nAH1.3", "HENV-26", "HENV-32", "m102.4"],
}
TARGET_WORD = {"escape": "escape sites"}
TARGET_YLAB = {"escape": "escape sites captured"}

plt.rcParams.update({
    **figstyle.BASE_RC,
    "figure.dpi": 600, "savefig.dpi": 600,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": SURFACE})


def draw(stat, ylab, fname, letter, arm="vhvl", logy=False, targets="escape",
         xmax=None, ncol=3, mm=(183, 134), name_ckpt=False, key=True):
    """xmax caps k and switches the axis to LINEAR.

    ncol/mm pick the GRID. Nine antibodies go 3 x 3 at 183 x 134 mm when the grid
    is a figure in its own right (Figure 4), and 5 x 2 at 180 x 83 mm when two
    grids have to share one Extended Data page.

    Why 5 x 2 and not a shorter 3 x 3. The height of a
    row is dominated by things that do not shrink. A three-line title at 6.2 pt is
    8.2 mm and a row of tick labels ~3 mm whatever the page is, so a 3 x 3 inside
    83 mm leaves 3 * H = 83 - 3*11 - 3.5 - 12 (legend) = 34.5 mm, i.e. 11.5 mm of
    plot per panel -- less than four y-tick labels. A 5 x 2 pays that fixed cost
    twice instead of three times AND has a TENTH slot, which takes the legend, so
    the 12 mm strip under the grid disappears too: 2 * H = 83 - 2*11 - 3.5 - 3
    gives H = 27 mm. The cost is column width, 34 mm instead of 58, which is why
    the title is broken onto three lines below.

    Nature BE, "Panel arrangement": "Consider the content and legibility of each
    panel and let that define its size within the figure."

    The log axis exists because k ran to 400, two orders of magnitude. Capped at
    the size of the largest gold epitope (49 residues, VRC01) every panel covers
    one order of magnitude, where a log axis only distorts the spacing of the
    points that matter. So: capped -> linear, uncapped -> log.
    """
    suf = "" if targets == "escape" else f"_{targets}"
    cur = pd.read_csv(RES / f"binary_cost_curves_{arm}{suf}.csv")
    gol = pd.read_csv(RES / f"binary_cost_gold_{arm}{suf}.csv").set_index("antibody")
    if xmax is not None:
        cur = cur[cur.k <= xmax]
    order_ = PANEL_ORDER[ARM]
    if order_ is None:
        order_ = list(gol.sort_values("fold", ascending=False).index)
    abs_ = [a for a in order_
            if a in set(cur.antibody) and a not in DROP_ANTIBODIES]
    missing = sorted(set(cur.antibody) - set(abs_) - set(DROP_ANTIBODIES))
    if missing:
        # An antibody with escape data that PANEL_ORDER does not name would be
        # dropped silently, which is the one failure this panel cannot afford.
        raise SystemExit(f"{ARM}: {missing} have curves but are not in "
                         f"PANEL_ORDER[{ARM!r}] -- add them or name them in "
                         f"DROP_ANTIBODIES, do not let them vanish")
    n = len(abs_)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(mm[0] / 25.4, 1.9 * nrow),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.40 if ncol > 3 else 0.34,
                                          "hspace": 0.72 if ncol > 3 else 0.62})
    # The last row is centred and the key sits at its right. With 7 antibodies
    # in a 3-wide grid an uncentred last
    # row held one panel hard left and the key beside it, which read as a row
    # that had been cut off. Panels in the final partial row are shifted right by
    # half the gap, the key goes in the row's LAST column, and the slots on
    # either side stay blank.
    full = (n // ncol) * ncol              # panels in the complete rows
    rem = n - full                         # panels in the final partial row
    # This applies only when the last row holds one panel. A single panel hard left with the
    # key beside it reads as a row that was cut off, so it is pushed right until
    # it meets the key's column -- in a 3-wide row that lands it in the middle,
    # which is the case this was asked for. With TWO panels left over it shifted
    # the Extended Data k = 400 grids (7 antibodies in 5 columns) away from the
    # left margin for no reason, so those stay where they were.
    pad = max(0, ncol - 1 - rem) if rem == 1 else 0
    slot = [i if i < full else full + pad + (i - full) for i in range(n)]
    key_slot = (nrow * ncol - 1) if rem else n

    for i, ab in enumerate(abs_):
        ax = axes[slot[i] // ncol][slot[i] % ncol]
        sub = cur[cur.antibody == ab]
        for m in ORDER + ["Random"]:
            d = sub[sub.method == m].sort_values("k")
            if not len(d):
                continue
            yv = -np.log10(d[stat]) if stat == "p" else d[stat]
            ax.plot(d.k, yv, lw=1.3 if m != "Random" else 1.0,
                    color=COL.get(m, MUTED), zorder=4 if m != "Random" else 2,
                    ls="-" if m != "Random" else (0, (2, 1.5)))
        # Not every antibody has a solved complex. RSV palivizumab (Fab-alone)
        # and suptavumab (no PDB entry) have escape data and no structural
        # epitope, so they are drawn WITHOUT the dashed reference rather than
        # left out of the panel. `has_gold` is the table's own flag; the NaN
        # test covers the HIV tables, which predate the flag and have none.
        gv = float(gol.loc[ab, stat])
        has_gold = bool(gol.loc[ab, "has_gold"]) if "has_gold" in gol.columns \
            else np.isfinite(gv)
        if has_gold and np.isfinite(gv):
            gv = -np.log10(gv) if stat == "p" else gv
            ax.axhline(gv, color=INK, lw=1.0, ls=(0, (3, 1.6)), zorder=5)
            # THE DOT IS AT x = n_gold, the number of residues in the solved
            # structural epitope, and y = the gold line's own recall. It is the
            # one k at which the model's top-k set and the structural epitope
            # are the same SIZE, so it is the only point on the panel where the
            # two are compared like for like. The count is printed beside it
            # because a reader cannot read it off a log x
            # axis, and the key names it.
            ng = int(gol.loc[ab, "n_gold"])
            ax.plot([ng], [gv], marker="o", ms=2.6, color=INK, zorder=6)
            ax.annotate(f"{ng}", xy=(ng, gv), xytext=(3.0, -1.0),
                        textcoords="offset points", ha="left", va="top",
                        fontsize=5.6, color=INK, zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground=SURFACE)])
        if xmax is None:
            ax.set_xscale("log")
            ax.set_xticks([5, 20, 100, 400])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        else:
            ax.set_xlim(0, xmax)
            ax.set_xticks(np.arange(0, xmax + 1, 10 if xmax <= 50 else 100))
        # Three lines at ncol=5: "gp120-gp41 interface  (12 escape sites)" is 39
        # characters, ~34 mm at 6.2 pt, which is the WHOLE column. Split, the
        # longest line is "gp120-gp41 interface" at ~17.5 mm and clears easily.
        sep = "\n" if ncol > 3 else "  "
        if ARM == "dms":
            title = (f"{ab} {template_mark(ab)}\n{gol.loc[ab,'epitope_class']}{sep}"
                     f"({int(gol.loc[ab,'n_sig'])} {TARGET_WORD[targets]})")
        else:
            # THREE LINES ALWAYS, the third often blank, so the antibody names
            # sit on one line across the row -- a title anchored at the axes top
            # grows UPWARD, so a two-line title would raise its name above its
            # neighbours'. The escape-set size is not repeated per panel here:
            # it is the same declared top 10 for every antibody on these arms
            # and belongs in the legend, where the third line is needed for the
            # reason a panel has no gold line.
            # `note` is empty for most antibodies, and pandas reads an empty
            # csv field back as NaN, not "" -- which printed the string "nan"
            # on four of six RSV panels the first time this ran.
            nt = gol.loc[ab, "note"] if "note" in gol.columns else ""
            nt = "" if pd.isna(nt) else str(nt)
            title = f"{ab} {template_mark(ab)}\n{gol.loc[ab,'epitope_class']}\n{nt}"
        ax.set_title(title, fontsize=6.2, pad=2.5)
        if i % ncol == 0:
            ax.set_ylabel(ylab, fontsize=6.3)
        elif ARM != "dms":
            # ONE ROW: every panel has the same 0-1 recall axis, so repeating
            # six sets of tick labels spends ~3 mm of a 30 mm column on saying
            # the same thing six times. Ticks stay, labels go.
            ax.set_yticklabels([])
        if i // ncol == nrow - 1 and ARM == "dms":
            ax.set_xlabel("top k predicted residues", fontsize=6.3)
        ax.tick_params(labelsize=5.8)
        if stat == "recall":
            ax.set_ylim(-0.04, 1.04)
        else:
            ax.axhline(-np.log10(0.05), color=MUTED, lw=0.5, ls=":", zorder=1)
        if i == 0 and letter:
            figstyle.panel_letter(ax, letter)
    for j in set(range(nrow * ncol)) - set(slot):
        axes[j // ncol][j % ncol].axis("off")
    if ARM != "dms":
        # One x label under the row, centred on the figure, rather than the same
        # five words repeated under every panel.
        # y is NEGATIVE on purpose. y=0 is the bottom of the figure BOX, which
        # on a one-row panel is where the x tick labels already are -- the label
        # landed on top of "20 30 40". fit_save measures the tight bbox and pads
        # the page to it, so ink below the box is kept, not clipped.
        fig.supxlabel("top k predicted residues", fontsize=6.3, y=-0.13)
    h = [Line2D([], [], color=COL[m], lw=1.4,
                label=walle_label(force=name_ckpt) if m == "WALLE" else m)
         for m in ORDER if (cur.method == m).any()]
    wide = "gold structural epitope (dot = its size in residues)"
    h += [Line2D([], [], color=GREY_L, lw=1.0, ls=(0, (2, 1.5)), label="Random"),
          Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.6)),
                 label=wide if ncol == 3 else wide.replace(" (dot", "\n(dot"))]
    if stat != "recall":
        h += [Line2D([], [], color=MUTED, lw=0.5, ls=":", label="p = 0.05")]
    # The template-availability key, drawn only when this arm actually splits --
    # on a uniform arm it would be one entry restating every title.
    marks = {template_mark(a) for a in abs_} - {""}
    if len(marks) > 1:
        h += [Line2D([], [], color=INK, lw=0, marker="s", ms=2.8,
                     label=f"complex deposited before "
                           f"{AF3_TEMPLATE_CUTOFF}"),
              Line2D([], [], color=INK, lw=0, marker="s", ms=2.8, mfc="none",
                     mew=0.9, label="deposited after")]
    elif marks == {SEEN}:
        h += [Line2D([], [], color=INK, lw=0, marker="s", ms=2.8,
                     label=f"all deposited before "
                           f"{AF3_TEMPLATE_CUTOFF}")]
    if not key:
        h = []
    if not h:
        pass
    elif n < nrow * ncol:
        # The legend goes in the empty slot. Nine panels in a ten-slot grid leave
        # one, and a key placed there costs no height at all -- where the strip
        # under the grid cost ~12 mm, which is 14% of an 83 mm half-page.
        spare = axes[key_slot // ncol][key_slot % ncol]
        spare.legend(handles=h, loc="center", ncol=1, frameon=False, fontsize=6.2,
                     handlelength=1.6, labelspacing=0.5, borderpad=0.0)
    else:
        fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.2,
                   bbox_to_anchor=(0.5, -0.02), columnspacing=1.6)
    fit_save(fig, OUT, fname, mm)
    print(f"wrote {OUT / fname}.{{pdf,png}}")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # Put the checkpoint in the WALLE legend entry even when this is the default
    # variant. For a panel drawn to sit beside another checkpoint's panel, "WALLE"
    # alone does not say which of the ten it is.
    ap.add_argument("--name_checkpoint", action="store_true")
    # Which virus block carries the figure's key. The default is per arm (see
    # VIRUS_PANEL); this moves it without editing the table.
    ap.add_argument("--key", dest="key", action="store_true", default=None)
    ap.add_argument("--no-key", dest="key", action="store_false")
    a = ap.parse_args()
    tg, suf = "escape", ""
    nc = a.name_checkpoint
    ylab = TARGET_YLAB[tg]

    if ARM != "dms":
        # One row, VH+VL, k <= 50 linear. No panel letter is drawn and the
        # top of the page is left free, because the letter belongs to the
        # schematic strip above this panel in the assembled figure.
        sp = VIRUS_PANEL[ARM]
        if not (RES / f"binary_cost_curves_vhvl.csv").exists():
            raise SystemExit(f"no curves in {RES} -- run dms_08 on this arm first")
        draw("recall", sp["ylab"], sp["stem"], "", "vhvl", targets=tg, xmax=50,
             ncol=sp["ncol"], mm=sp["mm"], name_ckpt=nc,
             key=sp["key"] if a.key is None else a.key)
        return

    for arm in ("vhvl", "vh"):
        if not (RES / f"binary_cost_curves_{arm}{suf}.csv").exists():
            continue
        if arm == "vhvl":
            # Two x ranges of the SAME curves. k<=50 (linear) is the candidate main
            # panel: 50 is the size of the largest gold epitope, so it is the range
            # over which the dashed reference is a fair comparison. k<=400 (log) is
            # the supplementary version that shows where the curves eventually go.
            # Figure 4 is a figure of its own and keeps the roomy 3x3 at 183 mm;
            # Extended Data pairs two grids on one page, so those go 5x2 at 180 mm
            # -- the Extended Data maximum, which is 180 and not the 183 that main
            # figures get ("Maximum page dimensions are 180 mm wide by 170 mm
            # tall", research-figure-guide.nature.com/figures/
            # extended-data-formatting-guidelines/).
            draw("recall", ylab, f"fig4_{tg}_recall_k50", "", arm, targets=tg,
                 xmax=50, name_ckpt=nc)
            # "b" and "c": panels b and c of Extended Data 3, which groups the three
            # escape detail panels under one figure. The letters must match the
            # figure-slot table, which is the numbering authority.
            draw("recall", ylab, f"fig4_{tg}_recall_k400", "b", arm, targets=tg,
                 ncol=5, mm=ED, name_ckpt=nc)
            draw("p", "\u2212log10 Fisher p", f"figS7_{tg}_p_k400", "b", arm,
                 targets=tg, ncol=5, mm=ED, name_ckpt=nc)
        else:
            draw("recall", ylab, f"figS8a_{tg}_recall_VHonly", "c", arm, targets=tg,
                 ncol=5, mm=ED, name_ckpt=nc)
            draw("p", "\u2212log10 Fisher p", f"figS8b_{tg}_p_VHonly", "b", arm,
                 targets=tg, ncol=5, mm=ED, name_ckpt=nc)


if __name__ == "__main__":
    main()
