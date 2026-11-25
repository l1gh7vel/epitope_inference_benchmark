#!/usr/bin/env python3
"""Generate the Task II, Task III and structural-quality paper panels.

    PYTHONPATH=scripts python scripts/12_paper_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import figstyle
from figstyle import COL1, COL2, fit_save
import walle

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "paper_figures"
AMP = PROJECT / "results" / "task2" / "all_methods_per_case.csv"

# --- validated categorical palette (see module docstring) ---
AF3_C, WALLE_C, THIRD_C = "#2C6FAD", "#C1553B", "#7B5EA7"
# achromatic: baselines are floors, not models
GREY_D, GREY_M, GREY_L = "#6E6E69", "#9A9A95", "#C7C7C2"
# BepiPred-3.0 is a real method, not a floor, so it gets its own lighter grey --
# grey because it is antibody-blind, lighter so it does not read as a mask.
BEPI_C = "#BDBDB8"
# title -> results dir, so the Task III panels can find their per-case tables
TASK_OF = {"HIV-1 Env": "task3", "SARS-CoV-2 spike": "task3cov"}
INK, MUTED, SURFACE = "#1A1A1A", "#5C5C57", "#FFFFFF"


plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    **figstyle.BASE_RC,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.3, "legend.handleheight": 0.9,
    "legend.borderpad": 0.2, "legend.labelspacing": 0.35,
    "hatch.linewidth": 0.6,
})
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#FFFFFF", "#D3E2F2", AF3_C, "#12385C"])


def save(fig, name, mm):
    """Write the panel at its final print size -- see scripts/figstyle.py.

    `mm` is the slot this panel occupies in the assembled figure, in
    millimetres. The page comes out that size, so the panel is placed at
    100% and never scaled.

    Main figures use Nature's double-column width, 183 mm. Extended Data has a
    narrower specification, 180 mm wide by 170 mm tall, the height leaving room
    for a legend underneath. Every figS* panel here is therefore 180 mm wide
    and every main-figure panel 183 mm.
    """
    return fit_save(fig, OUT, name, mm)


def method_colour(m):
    """Colour follows the entity, in every panel.

    The antibody-blind methods are achromatic: they are the floor, not a
    competing model. BepiPred-3.0 gets a lighter grey than the masks because it is a
    real predictor rather than a fixed region, but it stays grey because it cannot
    take an antibody as input at all.
    """
    if m in BLIND:
        return {"Random\n(AUPRC floor)": GREY_L, "RBD mask 319-541": GREY_M,
                "RBM mask 436-507": GREY_D, "BepiPred-3.0": BEPI_C}.get(m, GREY_M)
    if m.startswith("ESM"):
        return THIRD_C
    if m.startswith("WALLE"):
        return WALLE_C
    return AF3_C


def short_label(m):
    """Axis labels that fit. The full spec lives in the figure legend, not the art.

    At 183 mm with eight methods a tick slot is ~23 mm wide, and
    "(treasured-8, authors' pick)" is wider than that, so the qualifier is
    dropped from the tick label. Nature FG: "avoid ... overlapping text".
    Which checkpoint the authors picked is a sentence for the legend.
    """
    return (m.replace("Random\n(AUPRC floor)", "Random")
             .replace(" 319-541", "\n319\u2013541")
             .replace(" 436-507", "\n436\u2013507")
             .replace(", authors' pick", "")
             # ", AsEP best, ratio split" is the wording written by
             # 13_task2_all_methods.py. Both wordings are stripped, because the
             # committed results/task2/all_methods_per_case.csv may carry either
             # one.
             .replace(", AsEP best, ratio split", "")
             .replace("WALLE (", "WALLE\n("))


from figstyle import panel_letter                           # noqa: E402,F401


def titled_panel(ax, letter, title, **kw):
    """A left-aligned title with the panel letter as its own text object.

    Nature asks for panel letters at 8 pt bold, upright. A letter written into
    the title string takes the title's size and weight instead, so the letter is
    drawn separately and the title is indented to clear it.
    """
    ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=figstyle.PANEL_PT,
            fontweight="bold", va="bottom", ha="left")
    # Centred and bold, matching every other panel title. The letter keeps its
    # own position at the axes' left corner, so centring the title does not move it.
    ax.set_title(title, loc="center", pad=4, fontweight="bold",
                 fontsize=figstyle.TITLE_PT, **kw)


def pair_bars(ax, x, solid, hatched, colors, width=0.34):
    """Two statistics of the same quantity: solid + same-hue hatched."""
    b1 = ax.bar(x - width / 2, solid, width, color=colors, edgecolor=SURFACE,
                linewidth=0.8, zorder=2)
    b2 = ax.bar(x + width / 2, hatched, width, color="none", edgecolor=colors,
                linewidth=0.8, hatch="////", zorder=2)
    return b1, b2


# ------------------------------------------------------------------- Task II ---
def load_task2():
    pc = pd.read_csv(PROJECT / "results/task2/full/per_case_metrics.csv")
    man = pd.read_csv(PROJECT / "data/manifests/task2_manifest.csv")
    pc = pc.merge(man[["case_id", "chain_role_ok"]], on="case_id").query("chain_role_ok")
    wal = pd.read_csv(PROJECT / "results/task2/walle/af3/summary.csv")
    wal = wal[(wal.convention == "full") & (wal.leakage == "whole") & (wal.group == "all")]
    swap = pd.read_csv(PROJECT / "results/task2/walle/af3/swap.csv")
    return pc, wal, swap


def t3_ckpt(wal=None):
    """The project WALLE checkpoint, held fixed across every panel and task."""
    return walle.HEADLINE_CHECKPOINT


# ------------------------------------------------------------------ Task III ---
def load_t3(task):
    p = PROJECT / "results" / task / "metrics.json"
    if not p.exists():
        return None
    af3 = json.loads(p.read_text())
    wal = {}
    for f in sorted((PROJECT / "results" / task / "walle").glob("*/metrics.json")):
        wal[f.parent.name] = json.loads(f.read_text())
    # the other two method classes, where they were runnable on this arm
    extra = {}
    for name, sub in [("BepiPred-3.0", "bepipred"), ("ESMFold2", "esmfold2")]:
        q = PROJECT / "results" / task / sub / "metrics.json"
        if q.exists():
            extra[name] = json.loads(q.read_text())
    return af3, wal, extra


def _t3_bin_labels(task):
    """Per-residue domain label vectors for one Task III arm, from its config."""
    import importlib.util as _iu
    spec = _iu.spec_from_file_location(task, PROJECT / "config" / f"{task}.py")
    mod = _iu.module_from_spec(spec); spec.loader.exec_module(mod)
    L = mod.ANTIGEN_LEN
    out = {}
    for b, spans in mod.BIN_RANGES.items():
        v = np.zeros(L, bool)
        for lo, hi in spans:
            v[lo - 1:hi] = True
        out[b] = v
    return out, L


def _rank_biserial(y, sc):
    """2*AUC-1, identical definition to t3_05_evaluate.rank_biserial."""
    x, z = sc[y], sc[~y]
    if not len(x) or not len(z):
        return np.nan
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(x, z, alternative="two-sided").statistic
    return float(2 * (u / (len(x) * len(z))) - 1)


def t3_blind_rankbis(task, true_bins, majority_bin, seed=0):
    """Per-case domain rank-biserial for the two antibody-blind baselines.

    These are the same two floors the accuracy panel shows, carried onto the
    distribution panel so the two can be read against each other:

      always-<majority bin>  a binary mask: every residue of one fixed domain ranked
            above every other residue. It is the strongest blind strategy, and on
            this statistic it is degenerate by construction -- exactly +1 when that
            domain happens to be the right answer and a single constant negative
            value, -(size of that bin)/(residues outside the true bin), when it is
            not. A blind method therefore cannot produce a spread, only a coin flip
            between two outcomes.
      random scores  a fresh uniform ranking per case; centred on zero with real
            spread, which is what "no information" actually looks like here.
    """
    bins, L = _t3_bin_labels(task)
    rng = np.random.default_rng(seed)
    mask = bins[majority_bin]
    always, rand = [], []
    for tb in true_bins:
        y = bins[tb]
        always.append(_rank_biserial(y, mask.astype(float)))
        rand.append(_rank_biserial(y, rng.random(L)))
    return np.array(always, float), np.array(rand, float)


# Which domain the "always <domain>" baseline answers when the arm is tied.
# SARS-CoV-2 spike is exactly balanced -- NTD 100 / RBD 100 / S2 100 -- so the
# majority-class baseline scores 0.3333 whichever of the three it names, and
# metrics.json's `floor_majority_bin` = "NTD" is a value_counts tie-break rather
# than a finding. The spike majority-domain baseline is drawn as "Always RBD".
# RBD is the immunodominant and most-studied domain, so it is the baseline a
# reader would think of and beating it is the meaningful claim.
# HIV-1 Env is not tied: CD4bs is the majority at 116/300.
MAJORITY_TIE_BREAK = {"task3cov": "RBD"}


def majority_bin(task, af3):
    """The domain the always-<domain> floor answers, with ties named explicitly."""
    return MAJORITY_TIE_BREAK.get(task, af3["floor_majority_bin"])


def t3_methods(task, af3, wal, extra, ckpt=None):
    """One ordered method list per arm: (label, metrics, per_case df, colour, blind).

    Order is the same everywhere in the paper -- antibody-blind first, then the
    classes that can condition -- so a reader can compare panels by position, and the
    method keys are identical across arms so the two rows of fig4d line up column
    by column. `per_case` is None for a method that has no per-case score vector (the
    two floors are scalars, not predictors).

    The configured WALLE checkpoint is used for both antigens. The range across
    all released checkpoints is retained for the accuracy panel.
    """
    base = PROJECT / "results" / task
    bw = max(wal.items(), key=lambda kv: kv[1]["accuracy"]) if wal else None
    use = ckpt if (ckpt and ckpt in wal) else (bw[0] if bw else None)

    def pc(sub):
        f = base / sub / "per_case_domain.csv" if sub else base / "per_case_domain.csv"
        return pd.read_csv(f) if f.exists() else None

    out = []
    if "BepiPred-3.0" in extra:
        out.append(("BepiPred-3.0", extra["BepiPred-3.0"], pc("bepipred"), BEPI_C, True))
    if "ESMFold2" in extra:
        out.append(("ESMFold2", extra["ESMFold2"], pc("esmfold2"), THIRD_C, False))
    if use:
        out.append((f"WALLE\n{use.replace('-sweep', '')}", wal[use],
                    pc(f"walle/{use}"), WALLE_C, False))
    out.append(("AlphaFold3", af3, pc(None), AF3_C, False))
    return out, (bw if bw and bw[0] != use else None)


T3_ROSTER = ["BepiPred-3.0", "ESMFold2", "WALLE", "AlphaFold3"]


def roster_key(label):
    return label.split("\n")[0]


def chance_corrected(m):
    """Return observed hits, chance-expected hits, kappa and permutation P."""
    n = int(m["n"])
    return dict(n=n, hits=int(round(m["accuracy"] * n)),
                expected=int(round(m["chance_agreement"] * n)),
                kappa=float(m.get("kappa", float("nan"))),
                p=float(m.get("perm_p", float("nan"))))


def fig4c(arms, ckpt=None):
    """Task III: the headline accuracy and the distribution behind it.

    argmax-domain accuracy is binary per case, so it has no distribution to show.
    The top row presents it directly and carries the comparison to the
    antibody-blind floors. The bottom row shows per-case domain rank-biserial,
    which is continuous, so the spread and the failures are visible rather than
    averaged away.

    Rank-biserial is the one Task III metric that is prevalence-free, hence the
    only one comparable to Task II, but three things flatter it here. (i) The
    positives are a contiguous surface patch and the score field is smooth, so
    neighbouring residues are not independent and the effective n is far below the
    residue count. (ii) Large bins are picked up by area alone. (iii) It rewards a
    degenerate predictor asymmetrically: a method that always answers CD4bs scores
    strongly positive on CD4bs cases and strongly negative elsewhere (measured:
    S2 -0.83, V2 -0.23), so its mean can look reasonable while the method is doing
    nothing. Read the shape and the mass below zero, not the mean. Task III
    positives are ~4% of residues against Task II's 1.3%, so a higher number here
    is not "better".
    """
    n_arm = len(arms)
    # Method labels sit on the lower row only, so the top row carries no tick
    # labels and needs no clearance beneath it. 0.28 leaves room for the bottom
    # row's title line and nothing more.
    fig, axes = plt.subplots(2, n_arm, figsize=(COL2 * 1.0, 3.6), squeeze=False,
                             gridspec_kw={"wspace": 0.22, "hspace": 0.52,
                                          "height_ratios": [1.0, 1.15]})
    for k, (title, af3, wal, extra) in enumerate(arms):
        meth, _best = t3_methods(TASK_OF[title], af3, wal, extra, ckpt)

        # ---- top: accuracy against every antibody-blind floor ----
        ax = axes[0][k]
        ckpt_range = None
        rows = [("Random\nscores", af3["floor_random"], GREY_L),
                (f'Always\n{majority_bin(TASK_OF[title], af3)}',
                 af3["floor_majority"], GREY_D)]
        rows += [(lab, m["accuracy"], col) for lab, m, _pc, col, _b in meth]
        x = np.arange(len(rows))
        # Shaded band and label over the antibody-blind columns, matching Fig 3b,
        # 3c and 3d. The band names what the grey means: random scores, always
        # answering the commonest domain, and BepiPred-3.0, which reads the
        # antigen sequence alone and never sees the antibody. The count is
        # derived from the row labels rather than hard-coded, so it cannot drift
        # from the order the rows are built in.
        blind_names = ("Random", "Always", "BepiPred")
        n_blind_4c = sum(1 for r in rows
                         if str(r[0]).replace("\n", " ").startswith(blind_names))
        ax.axvspan(-0.55, n_blind_4c - 0.45, color=GREY_L, alpha=0.18, zorder=0,
                   lw=0)
        # Left-aligned inside the band, clear of the significance mark that sits
        # over BepiPred-3.0.
        ax.text(-0.42, 0.97, "antibody-blind", fontsize=6.5,
                color=MUTED, ha="left", va="top", zorder=6)
        ax.bar(x, [r[1] for r in rows], 0.62, color=[r[2] for r in rows],
               edgecolor=SURFACE, linewidth=0.8, zorder=2)
        # Value labels sit above whatever is drawn in each column, which for
        # WALLE is the checkpoint range bar rather than the bar top.
        wal_top = max((v["accuracy"] for v in wal.values()), default=None) if wal else None
        for xi, r in zip(x, rows):
            top = max(r[1], wal_top) if (r[0].startswith("WALLE") and wal_top) else r[1]
            ax.text(xi, top + 0.016, f"{r[1]:.3f}", ha="center", va="bottom",
                    fontsize=6)
        # Significance marks on this panel as well as on the confusion-matrix
        # panel. Three of the eight tests are significant, two at P = 0.0005.
        #
        # The statistic is the one 4b uses: chance-corrected agreement with both
        # marginals held fixed, P by label permutation, one-sided, since the
        # question has a direction. The two panels report the same eight
        # comparisons, so they carry the same test.
        #
        # Bonferroni over eight tests (2 arms x 4 methods) -> alpha = 0.05 / 8 =
        # 0.00625, and it changes an answer. Spike ESMFold2 at P = 0.024 is
        # nominal only and gets n.s. here; Env ESMFold2 and Env AlphaFold3 at
        # P = 0.0005 clear it. The blind floors carry no mark, being at chance by
        # construction, so testing them against chance is not a question.
        alpha_dom = 0.05 / 8
        # All marks on one row at the top of the axes. A constant row keeps a
        # mark off a neighbouring bar's body and makes the marks readable as a
        # row of verdicts, which is how they are used.
        marks = []
        for xi, (_lab, m, _pc, _col, _b) in zip(x[len(rows) - len(meth):], meth):
            en = chance_corrected(m)
            marks.append((xi, _stars(en["p"], alpha_dom), en["p"]))
        # Fixed y near the top of the 0-1 accuracy axis. The per-bar value labels
        # sit at bar height + 0.016 and no accuracy on either arm exceeds 0.45,
        # so 0.90 clears every bar, every value label and every WALLE range
        # whisker.
        y_mark = 0.90
        for xi, txt, pv in marks:
            ax.text(xi, y_mark, txt, ha="center", va="center", fontsize=6.5,
                    color=INK if pv < alpha_dom else MUTED)

        # No reference lines. The blind floors are already bars on this axis, so a
        # dashed line would duplicate them, and the legend names which bars they
        # are.
        # WALLE: one checkpoint is one draw, so the spread over the ten released
        # checkpoints is drawn as a vertical range bar on its column.
        if wal:
            xi = [i for i, r in enumerate(rows) if r[0].startswith("WALLE")]
            accs = sorted(v["accuracy"] for v in wal.values())
            if xi:
                # Drawn on the column, clear of the neighbouring bars' values.
                ax.vlines(xi[0], accs[0], accs[-1], color=INK, lw=1.0, zorder=5)
                for yv in (accs[0], accs[-1]):
                    ax.plot([xi[0] - 0.12, xi[0] + 0.12], [yv] * 2, color=INK, lw=1.0,
                            zorder=5)
                ckpt_range = (len(accs), accs[0], accs[-1])
        # The method columns are identical to the rank-biserial row immediately
        # below, so the shared columns are labelled once, on the lower row.
        # The two rows share one x range, set explicitly on both. Left to
        # autoscale, this row would take the bar extent (-0.31 to n-0.69 at
        # width 0.62) while the lower row runs -0.6 to n-0.4, so the bars would
        # not sit over their own violins and the antibody-blind band, drawn in
        # data coordinates on both rows, would land in two different places.
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xticks(x)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        # Y label on the left arm only. Both arms plot the same quantity on the
        # same scale. The two rows also have different tick-label widths (0.0-1.0
        # against -1.0-1.0), so their labels sit at different x, and on the right
        # arm the pair lands ~2 mm apart in the middle of the figure and reads as
        # one run-on string. Nature FG asks to avoid overlapping text.
        if k == 0:
            ax.set_ylabel("domain accuracy")
        # The vertical bar spanning WALLE's column is the range over all ten
        # released checkpoints. Defined in the caption, not keyed in the plot.
        ax.text(0.0, 1.02, f"{title}, n = {af3['n']}, {len(af3['bins'])} domains",
                transform=ax.transAxes, fontsize=figstyle.TITLE_PT, va="bottom", ha="left")
        if k == 0:
            panel_letter(ax, "c")          # Fig 4c, after the confusion matrices

        # ---- bottom: the per-case distribution behind it ----
        # The same two antibody-blind floors as the panel above, carried onto this
        # statistic so the columns line up and "is this method beating blind?" can be
        # read down a column rather than across two different baselines.
        ax = axes[1][k]
        from scipy.stats import wilcoxon
        scored = [(lab, p, col) for lab, _m, p, col, _b in meth if p is not None]
        base = []
        if scored:
            tb = scored[0][1].true_bin.tolist()
            maj = majority_bin(TASK_OF[title], af3)
            alw, rnd = t3_blind_rankbis(TASK_OF[title], tb, maj)
            base = [("Random\nscores", rnd, GREY_L, None),
                    (f"Always\n{maj}", alw, GREY_D, None)]
        cols_all = base + [(lab, p.domain_rankbis.to_numpy(), col, p)
                           for lab, p, col in scored]
        x = np.arange(len(cols_all))
        # The same band on this row. The two rows share their columns exactly, so
        # a band on one row only would say the grouping applies to accuracy but
        # not to rank-biserial.
        n_blind_rb = sum(1 for c in cols_all
                         if str(c[0]).replace("\n", " ")
                         .startswith(("Random", "Always", "BepiPred")))
        ax.axvspan(-0.55, n_blind_rb - 0.45, color=GREY_L, alpha=0.18, zorder=0,
                   lw=0)
        rng = np.random.default_rng(0)
        for xi, (lab, v, col, p) in zip(x, cols_all):
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            if len(v) > 2 and np.ptp(v) > 1e-9:
                parts = ax.violinplot([v], positions=[xi], widths=0.74,
                                      showextrema=False)
                parts["bodies"][0].set_facecolor(col)
                parts["bodies"][0].set_alpha(0.30)
                parts["bodies"][0].set_edgecolor(col)
                parts["bodies"][0].set_linewidth(0.8)
            if p is not None:
                # Points split by whether the domain call was correct, which makes
                # the bimodality legible: a method scores strongly positive on the
                # domain it answers and strongly negative on the ones it does not,
                # so a single mean hides two populations. A correct point is drawn
                # in that method's own colour.
                ok = p.correct.to_numpy().astype(bool)
                jit = xi + rng.uniform(-0.10, 0.10, len(v))
                ax.plot(jit[~ok], v[~ok], ".", ms=1.6, color=GREY_D, alpha=0.40,
                        zorder=3)
                ax.plot(jit[ok], v[ok], ".", ms=1.6, color=col, alpha=0.75, zorder=4)
            else:
                jit = xi + rng.uniform(-0.10, 0.10, len(v))
                ax.plot(jit, v, ".", ms=1.6, color=col, alpha=0.55, zorder=3)
            ax.plot([xi - 0.26, xi + 0.26], [np.median(v)] * 2, color=col, lw=1.9, zorder=5)
            # Annotations live above the axes rather than in reserved data space.
            # The axis is the range of the statistic, -1 to 1, the grey
            # antibody-blind band stops with it, matching the band on the row
            # above, and both annotation rows sit in the figure margin in axes
            # fraction, where the column titles already live.
            ax.annotate(f"{np.median(v):+.2f}", xy=(xi, 1.02),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=5.8, color=col,
                        annotation_clip=False)
            # Significance on this row as well, against always-<domain>, and
            # reported whatever it returns.
            #
            # One item is one antibody. The statistic is that antibody's domain
            # rank-biserial. The comparator is the same antibody's rank-biserial
            # under the always-<majority domain> baseline, so the test is paired
            # and signed-rank is the right family, weighting by magnitude rather
            # than by sign. One-sided, method > blind, since the question has a
            # direction. Bonferroni 0.05/8 over the eight tests, matching the
            # accuracy row above so the two rows carry the same correction.
            #
            # always-<domain> is degenerate on this statistic: exactly +1 when
            # its one fixed domain is the right answer and a single constant
            # negative otherwise -- two values on Env, three on spike. The paired
            # differences are therefore dominated by the ~40% of cases where the
            # baseline scores +1 and every real method loses by a wide margin.
            # Nothing clears it on either arm, BepiPred-3.0 included. A method
            # that localises well on average and still cannot beat answering one
            # fixed domain is the finding.
            if p is not None and len(base):
                # Rebuild the baseline in per-case order for a paired test.
                alw_m, _rnd_m = t3_blind_rankbis(
                    TASK_OF[title], p.true_bin.tolist(),
                    majority_bin(TASK_OF[title], af3))
                blind_v = np.asarray(alw_m, float)
                if len(blind_v) == len(v):
                    pv_rb = wilcoxon(v, blind_v, alternative="greater",
                                     zero_method="wilcox").pvalue
                    ax.annotate(_stars(pv_rb, 0.05 / 8), xy=(xi, 1.17),
                                xycoords=("data", "axes fraction"),
                                ha="center", va="bottom", fontsize=6.5,
                                color=INK if pv_rb < 0.05 / 8 else MUTED,
                                annotation_clip=False)
        ax.axhline(0, color=INK, lw=0.7, zorder=1)
        ax.set_xticks(x, [c[0].replace("\n", " ") for c in cols_all], fontsize=5.8,
                      rotation=24, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(cols_all) - 0.4)
        # The axis is the range of the statistic. Rank-biserial is bounded at -1
        # and +1, so anything above 1 is empty by construction. Holding the
        # annotation rows inside the axes would need a higher limit, and axvspan
        # spans the axes in fraction rather than in data, so the grey
        # antibody-blind band would rise with it and overshoot the band on the
        # accuracy row. Both annotation rows sit in the margin above instead.
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks(np.arange(-1, 1.01, 0.5))
        if k == 0:
            ax.set_ylabel("rank-biserial")
        # Point colouring (grey = domain call incorrect, the method's own colour =
        # correct) is defined in the caption. A key in the plot would need one
        # swatch per method, since "correct" is whichever colour that method
        # already carries.

    # The two y labels are put on one vertical line. matplotlib places a y label
    # just outside its own widest tick label, and the two rows have different
    # widths -- "0.0" to "1.0" on top against "-1.0" to "1.0" below -- so the
    # minus sign pushes the lower label further left and the pair sits on two
    # verticals. The rows share a column and therefore a width, so one
    # axes-fraction x is one figure x for both. The labels are moved after the
    # draw, when the tick text has real extents, and the leftmost of the two
    # natural positions wins so neither label lands on its own ticks.
    fig.canvas.draw()
    xs = []
    for row in (0, 1):
        a = axes[row][0]
        box = a.get_window_extent()
        lefts = [t.get_window_extent().x0 for t in a.get_yticklabels()
                 if t.get_text()]
        if lefts:
            xs.append((min(lefts) - box.x0) / box.width)
    if xs:
        lab_x = min(xs) - 0.014
        for row in (0, 1):
            axes[row][0].yaxis.set_label_coords(lab_x, 0.5)
    save(fig, "fig4c_task3_accuracy", (183, 50))                  # Fig 4c


def fig4b_binary(arms, ckpt=None):
    """Binary confusion for the main figure: per true domain, right or wrong.

    The main figure carries the binary confusion matrix and the multiclass one
    goes to Extended Data.

    Task III's prediction is multiclass -- which of 3 (spike) or 5 (Env) domains
    does this antibody bind -- so "binary" here means the true domain stays on
    the rows and only the prediction is binarised into correct / incorrect. It is
    the multiclass matrix with the off-diagonal summed, so it answers which
    domains are recoverable at all while dropping the part that makes the 5 x 5
    unreadable at 183 mm, namely which wrong domain was chosen.

    What that drops is the identity of the confusion, i.e. the heavy column that
    shows a method always answering CD4bs, which is why the multiclass matrix is
    kept in Supplementary and why the degeneracy claim cites it rather than this
    panel. This panel still shows the fact of degeneracy: a fully degenerate
    method appears as one row at 100% with every other row at 0%, and
    BepiPred-3.0 is exactly that (V3 99/99 with 0 for all four other Env domains;
    RBD 100/100 with 0 for NTD and S2), as is WALLE decent-2 on spike (RBD
    97/100). Which wrong domain absorbs the misses needs the multiclass matrix.

    The layout is recall per domain rather than four separate two-column
    matrices, which would spend most of their width on white space and repeat the
    domain labels four times. One heatmap per arm, rows = true domain, columns =
    method, cell = fraction of that domain's antibodies placed in it, puts the
    four methods on a shared row so a domain can be read across methods, which is
    the comparison the panel is for.

    The chance correction under each column is the statistic used everywhere else
    in Figure 4 -- kappa with both marginals fixed, label-permutation P,
    Bonferroni over the eight tests -- so 4b, 4c and Supplementary agree.
    """
    n_arm = len(arms)
    rosters = [t3_methods(TASK_OF[a[0]], a[1], a[2], a[3], ckpt)[0] for a in arms]
    ncol = len(T3_ROSTER)
    hr = [len(a[1]["bins"]) for a in arms]
    # hspace is a fraction of the mean axes height, and these two matrices are
    # short (5 and 4 rows). The gap only has to clear one line of 5.4 pt kappa
    # text drawn at y = 1.04 above the lower matrix, which is ~0.08 in.
    fig, axes = plt.subplots(n_arm, 1, squeeze=False,
                             figsize=(COL2 * 0.62, 0.17 * sum(hr) + 0.66),
                             gridspec_kw={"hspace": 0.28, "height_ratios": hr})
    for r, ((title, af3, _w, _e), roster) in enumerate(zip(arms, rosters)):
        bins = af3["bins"]
        n = len(bins)
        by_key = {roster_key(x[0]): x for x in roster}
        keys = [k for k in T3_ROSTER if k in by_key]
        rec = np.full((n, len(keys)), np.nan)
        cnt = [[""] * len(keys) for _ in range(n)]
        for c, key in enumerate(keys):
            _lab, m, _pc, _col, _b = by_key[key]
            for i, t in enumerate(bins):
                row = m["confusion"][t]
                tot = sum(row.values())
                if tot:
                    rec[i, c] = row[t] / tot
                    cnt[i][c] = f"{row[t]}/{tot}"
        ax = axes[r][0]
        im = ax.imshow(rec, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for c in range(len(keys)):
                if not np.isnan(rec[i, c]):
                    ax.text(c, i, cnt[i][c], ha="center", va="center", fontsize=5.4,
                            color=SURFACE if rec[i, c] > 0.55 else INK)
        ax.set_yticks(range(n), bins, fontsize=5.8)
        # Method labels on the bottom arm only. Both arms carry the same four
        # columns in the same order, and a rotated upper copy would run into the
        # lower matrix's own row labels.
        ax.set_xticks(range(len(keys)))
        if r == n_arm - 1:
            ax.set_xticklabels([by_key[k][0].replace("\n", " ") for k in keys],
                               fontsize=5.8, rotation=22, ha="right",
                               rotation_mode="anchor")
        else:
            ax.set_xticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_ylabel(f"{title}\ntrue domain", fontsize=6.5)
        # kappa + one star under each method column, same threshold as 4c
        for c, key in enumerate(keys):
            en = chance_corrected(by_key[key][1])
            # Above each column, in axes fraction. get_xaxis_transform() takes y
            # in axes fraction, so the value is 1.04 and not a row index.
            ax.text(c, 1.04, f"$\\kappa$ {en['kappa']:+.2f} {_stars(en['p'], 0.05 / 8)}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.4, color=INK if en["p"] < 0.05 / 8 else MUTED)
        if r == 0:
            # "a": panel a of Extended Data 2, which groups the Task III
            # per-domain panels under one figure.
            panel_letter(ax, "a")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03,
                      shrink=0.6)
    cb.set_label("fraction of true domain recovered", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_linewidth(0.4)
    save(fig, "fig4b_task3_binary_confusion", (120, 62))          # ED 2a


def fig4d(arms, ckpt=None):
    """Confusion matrices for every method, with chance correction under each.

    The matrix alone cannot say whether a method is doing better than chance,
    because a degenerate predictor produces a plausible-looking heavy column.
    Hits, expected hits under the same marginals, Cohen's kappa and a label-
    permutation test make that distinction without treating expected counts as an
    independent sample.
    """
    n_arm = len(arms)
    rosters = [t3_methods(TASK_OF[a[0]], a[1], a[2], a[3], ckpt)[0] for a in arms]
    ncol = len(T3_ROSTER)
    hr = [len(a[1]["bins"]) for a in arms]
    # imshow(aspect="equal") sizes each axes from its data, so the figure height
    # has to match or the rows float in whitespace. 0.26 in per cell with 2.0 in
    # of per-row overhead keeps the counts at their point size without a ring of
    # empty page around them. hspace holds the three-line chance-correction
    # block, which at 0.46 runs through the spike row's method titles.
    fig, axes = plt.subplots(n_arm, ncol, figsize=(COL2 * 1.08, 0.26 * sum(hr) + 2.0),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.34, "hspace": 0.70,
                                          "height_ratios": hr})
    for r, ((title, af3, _w, _e), roster) in enumerate(zip(arms, rosters)):
        bins = af3["bins"]
        n = len(bins)
        by_key = {roster_key(x[0]): x for x in roster}
        for c, key in enumerate(T3_ROSTER):
            ax = axes[r][c]
            if key not in by_key:
                ax.axis("off")
                # name why the cell is empty
                ax.text(0.5, 0.5, f"{key}\n{title}\n\nfolds still running",
                        ha="center", va="center", fontsize=6.5, color=MUTED,
                        transform=ax.transAxes, linespacing=1.6)
                continue
            lab, m, _pc, col, _b = by_key[key]
            conf = np.array([[m["confusion"][t][p] for p in bins] for t in bins], float)
            rt = conf.sum(1, keepdims=True)
            frac = np.divide(conf, rt, out=np.zeros_like(conf), where=rt > 0)
            im = ax.imshow(frac, cmap=SEQ, vmin=0, vmax=1, aspect="equal")
            # The CoV matrices are 3x3 while the Env matrices above are 5x5.
            # With equal-aspect images matplotlib centres the smaller image inside
            # its subplot by default. Anchor west so every matrix's left edge
            # aligns with the one above.
            ax.set_anchor("W")
            for i in range(n):
                for j in range(n):
                    if rt[i, 0] == 0:
                        continue
                    ax.text(j, i, f"{int(conf[i, j])}", ha="center", va="center",
                            fontsize=5.8, color=SURFACE if frac[i, j] > 0.55 else INK,
                            fontweight="bold" if i == j else "normal")
            npred = [m["n_predicted_per_bin"][b] for b in bins]
            ax.set_xticks(range(n), bins, fontsize=5.6, rotation=45, ha="right")
            ax.set_yticks(range(n), bins, fontsize=5.6)
            for j, tl in enumerate(ax.get_xticklabels()):
                if npred[j] / m["n"] > 1.5 / n:     # a bin predicted far too often
                    tl.set_color(AF3_C)
                    tl.set_fontweight("bold")
            # Bold black, the same rule as figE4b and fig4b. Nature's
            # research-figure guide lists "Coloured text" under "We avoid" and
            # says to use keylines and keys instead. A confusion matrix is drawn
            # in a sequential colourmap, so the method colour would appear
            # nowhere else in the cell it titles.
            ax.set_title(lab.replace("\n", " "), loc="center", pad=3, fontsize=6.8,
                         color=INK, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{title}\ntrue domain", fontsize=6.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            en = chance_corrected(m)
            # The same Bonferroni threshold as 4c above (8 tests), so the two
            # panels agree about spike ESMFold2 (P = 0.024).
            sig = _stars(en["p"], 0.05 / 8)
            # Three short lines rather than two long ones, which at four columns
            # run into the neighbouring panel's caption.
            # An offset in axes fraction is not a constant distance here. The two
            # rows have height_ratios 5 and 3, so the Env axes is taller in
            # inches and a fixed offset drops its caption further down the page
            # than the spike row's. Scaling by 1/n makes the physical drop equal
            # on both rows; 1.15 rather than 0.9 because the rotated tick labels
            # descend into that gap and are not counted by the axes box.
            ax.text(0.5, -1.15 / n,
                    f"{en['hits']} / {en['n']} correct\n"
                    f"{en['expected']} expected by chance\n"
                    f"$\\kappa$ = {en['kappa']:+.3f}   {sig}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
                    color=INK if en["p"] < 0.05 / 8 else MUTED, linespacing=1.5)
            if r == 0 and c == 0:
                # Fig 4b: the confusion matrices come before the accuracy panel so
                # the reader sees the degeneracy (which domain each method defaults
                # to) before seeing any accuracy number. 4a is the Task III
                # schematic.
                panel_letter(ax, "b")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.015,
                      shrink=0.55, anchor=(0.0, 0.75), panchor=False)
    cb.set_label("fraction of true domain (recall)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    # "predicted domain" only; the statistics are defined in the legend, not here.
    # The y position drops it clear of the bottom row's own kappa annotations.
    fig.text(0.5, -0.085, "predicted domain", ha="center", fontsize=6.8,
             color=INK)
    save(fig, "fig4d_task3_confusion", (183, 92))                 # ED 2e


def figS1a(path):
    """Supplementary: was the input even folded well enough to ask the question?"""
    d = json.loads(Path(path).read_text())
    arms = [(v["antigen"], v["bins"]) for k, v in d.items() if v.get("bins")]
    if not arms:
        return
    ws = [len(b) for _, b in arms]
    fig, axes = plt.subplots(1, len(arms), figsize=(COL2 * 0.66, 2.3),
                             gridspec_kw={"width_ratios": ws, "wspace": 0.30})
    axes = np.atleast_1d(axes)
    for k, (ax, (title, bins)) in enumerate(zip(axes, arms)):
        names = list(bins)
        x = np.arange(len(names))
        vals = [bins[b]["mean_plddt"] for b in names]
        pts = [bins[b]["per_structure_mean"] for b in names]
        # The bands are a key, not a background. Nature's guide puts keys in the
        # legend or in a discrete key rather than drawn over the plot, so there
        # are no shaded axhspans across the data area. The bar values, printed
        # above each bar below, let a reader place a domain against the key
        # without any shading.
        #
        # The band edges follow the AlphaFold DB convention, checked against the
        # RCSB AlphaFold viewer legend and an AlphaFold2 best-practice review but
        # not against DeepMind's or EBI's own pages, so the legend attributes it
        # to the AlphaFold DB convention. Same scale in AF2/AF3.
        BANDS = [(90, 100, "very high"), (70, 90, "confident"),
                 (50, 70, "low"), (30, 50, "very low")]
        ax.bar(x, vals, 0.6, color=GREY_D, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        # The value above each bar, so a reader can check the number against the
        # band key rather than estimating it off the axis. Nature's figure
        # guidance asks that text be 5-7 pt and not overlap, so these sit above
        # the bar top and above the per-structure points, at 5.5 pt.
        for xi, v_, pp in zip(x, vals, pts):
            top = max([v_] + list(pp)) if len(pp) else v_
            ax.annotate(f"{v_:.0f}", xy=(xi, top), xytext=(0, 2.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.5, color=INK, zorder=5)
        for xi, p in zip(x, pts):                       # per-structure spread
            ax.plot(np.full(len(p), xi), p, ".", ms=1.6, color=INK, alpha=0.30, zorder=3)
        ax.set_xticks(x, names, fontsize=6, rotation=30, ha="right")
        ax.set_ylim(30, 100)
        pass  # the key is a stacked bar, built once after the loop
        ax.set_ylabel("antigen pLDDT")
        ax.set_title(title, loc="left", pad=4, fontsize=7)
        if k == 0:
            panel_letter(ax, "a")
    # The key is a stacked bar at the top right, giving the ranges and the
    # confidence each is read as. Its segments share the panels' own y quantity
    # -- the pLDDT bands on the same 30-100 scale the bars are measured on -- so
    # a reader reads a bar's height straight across into a band without
    # converting anything. Drawn in its own axes so it costs the data panels no
    # width.
    #
    # The key is outlined and divided at the band boundaries, with no grey
    # fills: the bars are one flat colour, so graded greys would imply an
    # encoding the panel does not use. The range sits inside each box and the
    # confidence word outside, to the right.
    kx = fig.add_axes([0.905, 0.12, 0.030, 0.80])
    for lo_, hi_, lab in BANDS:
        kx.bar(0, hi_ - lo_, bottom=lo_, width=1.0, color="none",
               edgecolor=INK, linewidth=0.6)
        rng = f">{lo_}" if hi_ >= 100 else f"{lo_}\u2013{hi_}"
        kx.text(0, (lo_ + hi_) / 2, rng, fontsize=5.0, color=INK,
                ha="center", va="center")
        kx.text(0.60, (lo_ + hi_) / 2, lab, fontsize=5.0, color=MUTED,
                ha="left", va="center")
    kx.set_ylim(30, 100); kx.set_xlim(-0.5, 0.5)
    kx.set_xticks([]); kx.set_yticks([])
    for sp in kx.spines.values():
        sp.set_visible(False)
    kx.set_title("pLDDT\n(AlphaFold DB)", fontsize=5.3, color=MUTED, pad=2,
                 loc="left")
    save(fig, "figS1a_plddt_by_bin", (180, 36))                   # ED 4a


# ------------------------------------------------- confidence vs performance ---
CONF = PROJECT / "results/confidence_per_case.csv"


def load_conf(task):
    if not CONF.exists():
        return None
    d = pd.read_csv(CONF)
    return d[d.task == task].set_index("case_id")


def figS1b():
    """Does AF3's own confidence predict how well a method does? (Task II)

    Two confidences with two meanings, never substituted for one another.
      pLDDT is per-residue confidence in the local fold. It is what WALLE depends on,
        because WALLE is pose-independent: it reads antibody and antigen as separate
        graphs and never sees inter-chain geometry.
      iPTM is confidence in the interface pose. It is what AF3's own contact-based
        epitope prediction depends on, and it should be irrelevant to WALLE.
    The panel therefore states a falsifiable prediction: iPTM should track AF3 and
    not WALLE. ESMFold2 is the control -- it builds its own structure and never sees
    AF3's, so any correlation there is case difficulty (some epitopes are simply
    easier) rather than AF3-specific docking. Read the AF3 column against the
    ESMFold2 column, not against zero.
    """
    from scipy.stats import spearmanr
    conf = load_conf("task2")
    p = AMP
    if conf is None or not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.method != "AlphaFold3 (5x5 ensemble)"]
    # BepiPred-3.0 is the difficulty control. A rise in AUPRC with AlphaFold3's
    # iPTM could mean AF3 docked this antibody well and so its epitope is right,
    # which is the claim, or it could mean the case is easy for everyone and AF3
    # is also confident about easy cases. BepiPred-3.0 reads the antigen sequence
    # alone, and the antigen is the same spike in all 186 cases, so its score
    # vector is identical for every antibody and its AUPRC varies only with where
    # that antibody's epitope happens to sit. Its correlation with AF3's iPTM is
    # therefore the case-difficulty component with nothing else in it. Flat
    # BepiPred column => the AF3 column is docking quality; rising BepiPred
    # column => that much of the AF3 column is difficulty. ESMFold2 is the weaker
    # control, since it builds its own structure but still reads the antibody.
    cols = [m for m in ["AlphaFold3", *sorted(m for m in d.method.unique()
                                              if m.startswith("WALLE")), "ESMFold2",
                        "BepiPred-3.0"]
            if m in set(d.method)]
    # Task II antigen pLDDT is tightly clustered (every case is the same spike,
    # confidently folded), so a fixed 55-100 axis would render the panel as one
    # vertical stripe. Scale it to the data instead; iPTM keeps its natural 0-1.
    pl = conf.plddt_antigen.dropna()
    pad = max(0.6, 0.08 * (pl.max() - pl.min()))
    measures = [("plddt_antigen", "AlphaFold3 antigen pLDDT",
                 (pl.min() - pad, pl.max() + pad)),
                ("iptm", "AlphaFold3 iPTM", (0, 1))]
    # ESMFold2's own confidence is a different quantity, since it folds its own
    # structure and never sees AlphaFold3's. Including it turns the ESMFold2
    # column from a bare control into a testable claim: AlphaFold3's iPTM should
    # not predict ESMFold2's epitope, whereas ESMFold2's own iPTM should.
    econf = None
    ep = PROJECT / "results/esm_confidence_per_case.csv"
    if ep.exists():
        e = pd.read_csv(ep)
        econf = e[e.task == "task2"].set_index("case_id")
        # Whose confidence this is belongs in the legend rather than in the axis
        # label.
        measures.append(("esm_iptm", "ESMFold2 iPTM", (0, 1)))
    # Each row of this grid is a different x variable, so each row carries its
    # own x label and the gap between rows has to hold one. The next row's axes
    # patch is opaque white and paints over anything in that gap.
    #
    # hspace 1.40 and a left margin of 0.055 rather than matplotlib's default
    # 0.125, for two reasons:
    #   * vertical. Each row carries its own x label and the row below it opens
    #     with a right-aligned rho/P line sitting 3% above its frame, so the two
    #     strings meet in the gap. The hspace plus labelpad=1.5 below closes it
    #     from both sides, using the free space between the tick numbers and the
    #     x label.
    #   * horizontal. The page is cropped to the tight bbox, so the empty band
    #     runs from the shared y label to the y tick labels, and at left=0.125
    #     that band is ~11 mm of blank page. At 0.055 the y label sits just
    #     outside the ticks instead of at the page edge.
    fig, axes = plt.subplots(len(measures), len(cols),
                             figsize=(COL2 * 0.95, 2.45 * len(measures)),
                             squeeze=False, gridspec_kw={"wspace": 0.26, "hspace": 1.40})
    fig.subplots_adjust(left=0.055, right=0.995)
    for r, (key, lab, xlim) in enumerate(measures):
        for c, m in enumerate(cols):
            ax = axes[r][c]
            src = econf if key.startswith("esm_") else conf
            if src is None or key not in src:
                ax.axis("off")
                continue
            sub = d[d.method == m].set_index("case_id").join(src[[key]], how="inner")
            sub = sub.dropna(subset=[key, "auprc"])
            col = method_colour(m)
            ax.plot(sub[key], sub.auprc, "o", ms=2.6, color=col, alpha=0.50,
                    mew=0, zorder=2)
            rho, pv = spearmanr(sub[key], sub.auprc)
            # Above the axes, not inside them. In the panels where the mass sits
            # at low x and high AUPRC there is no empty corner, so an annotation
            # inside the frame would sit on the points it describes whatever its
            # background. On one line above the frame it covers nothing, at a
            # cost of ~3 mm of height. Right-aligned on the same line the column
            # title occupies on the left, so the two share one band above the
            # axes.
            ax.text(1.0, 1.03,
                    f"$\\rho$ = {rho:+.3f}, p = {pv:.1e} {_stars(pv)}",
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=6,
                    zorder=6, color=INK if pv < 0.05 else MUTED)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.03, 1.03)
            # labelpad 1.5: the default ~4 pt leaves a gap the label does not
            # need, at the expense of the row below.
            ax.set_xlabel(lab, fontsize=6.5, labelpad=1.5)
            if r == 0:
                # short_label() drops ", authors' pick", 15 characters of the
                # widest title in the panel: at 183 mm "WALLE (treasured-8,
                # authors' pick)" runs into "ESMFold2" in the column beside it.
                # Which checkpoint the authors picked belongs in the legend,
                # exactly as it does on Fig. 2c.
                # Bold black, not the method's colour. Nature's research figure
                # guide lists "Coloured text; keylines, keys, etc. should be used
                # instead" under "We avoid", and says "Avoid coloured text". The
                # points below already carry the colour.
                # pad 13: the title is left-aligned and the rho/P line is
                # right-aligned on the same baseline, so on the long names
                # ("WALLE (decent-2)") the two would meet in the middle. The
                # title sits on its own line above the statistics.
                ax.set_title(short_label(m).replace("\n", " "), loc="left", pad=13,
                             fontsize=7, color=INK, fontweight="bold")

    # One y label for the whole grid, not one per row. The rotated string is
    # ~20 mm at 7 pt and a row's axes is ~15 mm tall, so per-row labels would
    # overlap each other. supylabel is a figure-level artist, so it is also drawn
    # on top of the axes patches rather than under them. x=0.016 keeps it just
    # outside the tick labels, which already sit close to the spine.
    fig.supylabel("per-antibody AUPRC", fontsize=figstyle.BASE_PT, x=0.016)
    # This call comes after supylabel, outside the loop. panel_letter measures
    # the figure's existing ink to place the letter just outside it, so anything
    # added later on the left would leave the letter indented.
    panel_letter(axes[0][0], "b")
    save(fig, "figS1b_task2_confidence_vs_auprc", (180, 64))      # ED 4b


def figS1c(arms):
    """Task III: is the domain call explained by how confident AF3 was? (both arms)

    Split per case by whether the domain call was correct. If the confident cases
    are the ones it gets right, the failure is a confidence-calibrated docking
    failure rather than a localisation failure, which is what the HIV arm shows
    (CD4bs iPTM 0.68, every other bin 0.21-0.34). The test is Mann-Whitney rather
    than Wilcoxon, because correct and incorrect are independent groups of cases
    here and are not paired measurements of the same case.
    """
    from scipy.stats import mannwhitneyu, spearmanr
    measures = [("iptm", "iPTM", (0, 1)), ("plddt_antigen", "antigen pLDDT", None)]
    # Tight spacing. No row carries an x label under it, so hspace holds no
    # text. wspace is small for the same reason: these panels have a short
    # shared y range and short tick labels.
    fig, axes = plt.subplots(len(arms), len(measures),
                             figsize=(COL2 * 0.70, 1.85 * len(arms)), squeeze=False,
                             gridspec_kw={"wspace": 0.22, "hspace": 0.62})
    for r, (title, af3, _w, _e) in enumerate(arms):
        task = TASK_OF[title]
        conf = load_conf(task)
        pc = pd.read_csv(PROJECT / "results" / task / "per_case_domain.csv")
        pc = pc.set_index("case_id").join(conf, how="inner")
        for c, (key, lab, fixed) in enumerate(measures):
            ax = axes[r][c]
            groups = [pc[pc.correct == v][key].dropna().to_numpy() for v in (0, 1)]
            # pLDDT on one antigen spans a few units, so a fixed 50-100 axis would
            # render a real difference as two flat lines. Scale it to the data.
            if fixed is not None:
                ylim = fixed
            else:
                lo, hi = pc[key].min(), pc[key].max()
                pad = max(0.4, 0.10 * (hi - lo))
                ylim = (lo - pad, hi + pad)
            cols = [GREY_M, AF3_C]
            for xi, (v, col) in enumerate(zip(groups, cols)):
                if not len(v):
                    continue
                parts = ax.violinplot([v], positions=[xi], widths=0.7, showextrema=False)
                parts["bodies"][0].set_facecolor(col)
                parts["bodies"][0].set_alpha(0.30)
                parts["bodies"][0].set_edgecolor(col)
                parts["bodies"][0].set_linewidth(0.8)
                ax.plot([xi - 0.25, xi + 0.25], [np.median(v)] * 2, color=col, lw=1.8,
                        zorder=4)
            if all(len(g) > 2 for g in groups):
                # Two-sided. "Are the correct calls the confident ones?" has no
                # pre-committed direction: anti-predictive confidence would be a
                # finding in itself, and the CoV arm is an explicit null test.
                pv = mannwhitneyu(groups[1], groups[0], alternative="two-sided").pvalue
                ax.text(0.5, 1.02, f"correct vs incorrect: p = {pv:.1e} {_stars(pv)}",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=6,
                        color=INK if pv < 0.05 else MUTED)
            rho, prho = spearmanr(pc[key], pc.domain_rankbis, nan_policy="omit")
            # The x label carries no Spearman rho against domain rank-biserial.
            # The panel's question is whether confidence separates correct from
            # incorrect domain calls, which the two groups and their test answer;
            # a correlation against a different quantity, printed where the axis
            # label goes, would be a second result competing with the first. The
            # number is computed here and belongs in the legend, so `rho` and
            # `prho` stay in scope for it.
            ax.set_xticks([0, 1], [f"incorrect\n(n={len(groups[0])})",
                                   f"correct\n(n={len(groups[1])})"], fontsize=6.5)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(*ylim)
            ax.set_ylabel(lab)
            if c == 0:
                # -0.26: the y label is a single short word, so the arm name
                # needs no more clearance than that.
                ax.text(-0.26, 0.5, title, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=figstyle.TITLE_PT, fontweight="bold")
            if r == 0 and c == 0:
                panel_letter(ax, "c")
    save(fig, "figS1c_task3_confidence_vs_correct", (180, 58))    # ED 4c


def fig3e_by_group(pc, per_case, ckpt,
                   name="fig3e_task2_by_group_distribution"):
    """Task II per-antibody AUPRC split by epitope class, all methods. (Panel 3e.)

    The non-RBM group is the antibodies whose epitope lies wholly inside the RBD.
    The RBM/non-RBM split is defined on the motif alone and says nothing about the
    RBD, so a non-RBM antibody can bind outside the RBD entirely, and the RBD mask
    scored against such an antibody is a baseline that cannot be right rather than
    one that has to be beaten. The group is the 45 antibodies for which the RBD
    mask could in principle be correct. The classes are `epitope_class` in the
    Task II manifest.

    The split is needed because the blind masks are defined as a region, so their
    score is a property of whether the antibody happens to bind inside that region
    rather than of the method. The RBM mask medians 0.222 on RBM binders and
    exactly the random floor outside it. A pooled number is therefore a weighted
    average of an almost-oracle and an exact floor, weighted by the panel's
    RBM:non-RBM ratio, which is a property of the dataset.

    The reference for the asterisks differs between panels. Within each panel the
    test is against the strongest antibody-blind baseline in that panel, chosen on
    the mean: the RBM mask on RBM binders (0.217) and the RBD mask outside
    (0.062). One reference for both would make the non-RBM asterisks meaningless,
    because the RBM mask sits exactly at the floor there and beating it is
    automatic. The cost is that asterisks are not comparable between the two
    panels, which the legend has to say.
    """
    from scipy.stats import wilcoxon
    d = pd.read_csv(AMP)
    d = d[d.method != "AlphaFold3 (5x5 ensemble)"]
    # Each antibody carries its class in the manifest; the panel draws the two
    # classes that the masks can be scored against. Antibodies of the other two
    # classes appear in every panel that does not split by class.
    man = pd.read_csv(PROJECT / "data/manifests/task2_manifest.csv")
    cls = dict(zip(man.case_id, man.epitope_class))
    d = d.assign(epitope_class=d.case_id.map(cls))
    order = [m for m in BLIND if m in set(d.method)] + \
        [m for m in ["ESMFold2"] if m in set(d.method)] + \
        sorted([m for m in d.method.unique() if m.startswith("WALLE")]) + ["AlphaFold3"]
    order = [m for m in order if m in set(d.method)]
    col = {m: method_colour(m) for m in order}

    # Tall, with the median printed per violin, matching the companion panel so
    # a reader comparing the two reads the same quantity off both.
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 1.15, 4.3), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    right_title = "non-RBM, epitope wholly inside the RBD"
    for k, (ax, (title, g)) in enumerate(zip(axes, [("RBM-binding antibodies", "RBM"),
                                                    (right_title, "nonRBM")])):
        sub = d[d.epitope_class == g]
        vals = {m: sub[sub.method == m].set_index("case_id").auprc for m in order}
        ref = max((m for m in BLIND if m in vals), key=lambda m: vals[m].mean())
        x = np.arange(len(order))
        data = [vals[m].to_numpy() for m in order]
        parts = ax.violinplot(data, positions=x, widths=0.74, showextrema=False)
        for body, m in zip(parts["bodies"], order):
            body.set_facecolor(col[m]); body.set_alpha(0.28)
            body.set_edgecolor(col[m]); body.set_linewidth(0.7)
        rng = np.random.default_rng(0)
        for xi, m, v in zip(x, order, data):
            ax.plot(xi + rng.uniform(-0.10, 0.10, len(v)), v, ".", ms=1.5,
                    color=col[m], alpha=0.45, zorder=3)
            ax.plot([xi - 0.26, xi + 0.26], [np.median(v)] * 2, color=col[m], lw=1.8,
                    zorder=4)
            ax.text(xi, 1.015, f"{np.median(v):.3f}", ha="center", va="bottom",
                    fontsize=5.4, color=MUTED)
        for xi, m in zip(x, order):
            if m == ref:
                ax.text(xi, 1.10, "ref", ha="center", va="bottom", fontsize=6,
                        color=MUTED, style="italic")
                continue
            common = vals[ref].index.intersection(vals[m].index)
            p = wilcoxon(vals[m][common], vals[ref][common], alternative="greater").pvalue
            ax.text(xi, 1.10, _stars(p), ha="center", va="bottom", fontsize=6.5,
                    color=INK if p < 0.05 else MUTED)
        ax.axvspan(-0.55, len(BLIND) - 0.45, color=GREY_L, alpha=0.18, zorder=0, lw=0)
        ax.set_xticks(x, [short_label(m).replace("\n", " ") for m in order],
                      fontsize=6, rotation=32, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.55, len(order) - 0.45)
        ax.set_ylim(-0.02, 1.22)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        # above the axes, clear of the significance row
        ax.text(0.0, 1.17, f"{title}, n = {sub.case_id.nunique()}",
                transform=ax.transAxes, fontsize=figstyle.TITLE_PT, va="bottom", ha="left")
        # named on BOTH panels: the band means the same thing in each
        ax.text(len(BLIND) / 2 - 0.5, 0.96, "antibody-blind", fontsize=6.5,
                color=MUTED, ha="center", va="top")
        if k == 0:
            ax.set_ylabel("AUPRC")
            panel_letter(ax, "d")
    save(fig, name, (183, 56))                                    # Fig 3d


def fig3f_conditioning(path, per_case_path=None):
    """Task II conditioning for every method: the antibody-swap gap. (Panel 3f.)

    Drawn as a distribution. Each antibody contributes its own (own - cross) gap,
    so a single bar would assert a uniform effect the data does not support.
    AlphaFold3's win-rate is 0.47, so it conditions on a subset of antibodies
    rather than a little on every one. The violin shows that directly: most of the
    mass sits at zero and the mean is carried by a tail. Read the shape, then the
    mean, in that order.

    The three antibody-blind methods have a measured mean gap of exactly 0.0000,
    whereas their per-antibody gaps are not zero and the violins are wide. For a
    blind predictor M[i,j] = f(j), so own_i = f(i) and cross_i = mean over j != i
    of f(j). own_i therefore varies with which epitope it is scored against, and
    it is the average over antibodies that cancels, S/n - (nS-S)/(n(n-1)) = 0. The
    blind violins are the null shape -- the spread a method with no conditioning
    still shows, purely because epitopes differ in how easy they are -- centred
    exactly on zero. A conditioning method has to beat that shape, not just zero.

    AlphaFold3 wins on 51% of antibodies, barely more often than the blind RBM
    mask's 47%. Its +0.117 mean comes from the size of its wins on a subset rather
    than from winning more often.

    There is no "random" violin. Random here is the analytic expected AUPRC of a
    random ranking, which equals the epitope prevalence, and a scalar has nothing
    to swap. An actual random score vector would be antibody-blind in the same
    sense as the three shown, so it would add a fourth exact zero and no
    information.
    """
    d = pd.read_csv(path)
    ENS = "AlphaFold3 (5x5 ensemble)"
    ens = d[d.method == ENS]
    ens_gap = float(ens.swap_gap.iloc[0]) if len(ens) else None
    d = d[d.method != ENS].set_index("method")
    order = [m for m in ["RBD mask 319-541", "RBM mask 436-507", "BepiPred-3.0",
                         "ESMFold2"] if m in d.index]
    order += sorted(m for m in d.index if m.startswith("WALLE"))
    order += [m for m in ["AlphaFold3"] if m in d.index]
    n_blind = int(d.loc[order].blind.sum())

    pc = None
    if per_case_path and Path(per_case_path).exists():
        pc = pd.read_csv(per_case_path)

    # Half width, sharing a row with the all-methods violin. Six categories of a
    # signed quantity centred on zero; the width is set by the annotation row,
    # which is ~11 mm, so a 94 mm slot gives each category 12.5 mm and they
    # clear.
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    x = np.arange(len(order))
    for xi, m in zip(x, order):
        col = method_colour(m)
        v = (pc[pc.method == m].gap.to_numpy() if pc is not None
             else np.array([d.loc[m, "swap_gap"]]))
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        # A blind method has zero spread; violinplot cannot take that, and forcing
        # it would invent a width that means nothing.
        if len(v) < 3 or np.ptp(v) < 1e-9:
            ax.plot([xi - 0.30, xi + 0.30], [float(np.mean(v))] * 2, color=col,
                    lw=2.2, solid_capstyle="butt", zorder=4)
        else:
            parts = ax.violinplot([v], positions=[xi], widths=0.74,
                                  showextrema=False)
            b = parts["bodies"][0]
            b.set_facecolor(col); b.set_alpha(0.32)
            b.set_edgecolor(col); b.set_linewidth(0.8)
            ax.plot([xi - 0.26, xi + 0.26], [float(np.median(v))] * 2, color=col,
                    lw=1.8, zorder=4)
        ax.plot([xi], [float(d.loc[m, "swap_gap"])], marker="D", ms=3.0,
                color=INK, zorder=5)

    ax.axhline(0, color=INK, lw=0.8, zorder=3)
    lo, hi = ax.get_ylim()
    # Headroom for one annotation row. More than this leaves a dead band across
    # the panel.
    ax.set_ylim(lo, hi + 0.11 * (hi - lo))
    lo, hi = ax.get_ylim()
    y_num = hi - 0.05 * (hi - lo)
    for xi, m in zip(x, order):
        g, pv = d.loc[m, "swap_gap"], d.loc[m, "p_value"]
        blind = bool(d.loc[m, "blind"])
        # no star on a blind method: its p-value tests a mean that is zero by
        # construction, so "not significant" there is arithmetic, not evidence.
        ax.text(xi, y_num, f"{g:+.4f}" + ("" if blind else f" {_stars(pv)}"),
                ha="center", va="center", fontsize=5.8, color=INK)
    # No win-rate row. Such a row would print, for each method, the fraction of
    # antibodies whose own-epitope score beat the mean of the others'. For an
    # antibody-blind method that number is forced by the cohort and says nothing
    # about the method: with one fixed score vector, own(i) - cross(i) > 0
    # reduces to f(i) > mean(f), so the win rate is the fraction of epitopes that
    # are easier than average for that vector, and it exceeds one half whenever
    # the median of f exceeds its mean. Measured, the identity is exact -- RBD
    # mask 0.5699, RBM mask 0.4677, BepiPred-3.0 0.4086, reproduced to four
    # decimals by fraction(f > mean f), and reproduced for no method that reads
    # the antibody. In a row it would put "wins 57%" over the RBD mask directly
    # above AlphaFold3's "wins 51%", reading as the mask conditioning better than
    # AlphaFold3, which it cannot, its mean gap being zero by construction. The
    # per-case unevenness is stated in the legend, where it can carry that
    # explanation.
    if ens_gap is not None and "AlphaFold3" in order:
        xi = order.index("AlphaFold3")
        ax.plot([xi - 0.34, xi + 0.34], [ens_gap] * 2, color=INK, lw=1.0,
                ls=(0, (2, 1.5)), zorder=5)
        # No in-plot key. Nature's formatting guide: "Where possible, text,
        # including keys to symbols, should be provided in the legend rather than
        # on the figure itself." At 94 mm a two-line key also collides with the
        # "antibody-blind" band label. The dashed line and the diamond are
        # defined in the caption.
    ax.axvspan(-0.55, n_blind - 0.45, color=GREY_L, alpha=0.18, zorder=0, lw=0)
    # At the top of the band, matching Fig 3b and 3d, so a reader scanning the
    # three panels finds the band named in the same place, and below the
    # per-column value row rather than level with it. The mean gaps are printed
    # across the top of the axes, and 0.11 of the range clears that row.
    ax.text((n_blind - 1) / 2, hi - 0.11 * (hi - lo), "antibody-blind", fontsize=6.5,
            color=MUTED, ha="center", va="top")
    ax.set_xticks(x, [short_label(m).replace("\n", " ") for m in order], fontsize=6,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(order) - 0.45)
    # A y label is drawn along the panel's height, so its length is a floor on
    # that height. "antibody-swap gap, per antibody (AUPRC)" is ~38 mm at 7 pt
    # against this panel's 50 mm slot, so the label is kept short.
    ax.set_ylabel("antibody-swap gap (AUPRC)")
    panel_letter(ax, "c")
    save(fig, "fig3f_task2_conditioning_all_methods", (94, 50))   # Fig 3d


def figS2(summary):
    """Extended: WALLE loses ground once cases it saw in training are removed.

    The comparison is the ratio to AF3 rather than clean-vs-whole absolutes. The
    clean subset is intrinsically easier -- AF3, which never saw AsEP, scores
    higher on it -- so AF3 measures the subset's difficulty and dividing by it
    cancels that out. A ratio that falls is WALLE losing something AF3 did not,
    namely training exposure.
    """
    d = summary[(summary.convention == "full") & (summary.group == "all")]
    rows = []
    for ck, g in d.groupby("checkpoint"):
        r = {}
        for lk in ("whole", "clean"):
            w = g[(g.leakage == lk) & (g.method == "WALLE")]
            a = g[(g.leakage == lk) & (g.method == "AF3")]
            if len(w) and len(a):
                r[lk] = w.auprc.iloc[0] / a.auprc.iloc[0]
        if len(r) == 2:
            rows.append((ck, r["whole"], r["clean"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(COL1 * 1.15, 2.6))
    for ck, a, b in rows:
        ax.plot([0, 1], [a, b], "-", color=WALLE_C, lw=0.9, alpha=0.75, zorder=2)
        ax.plot([0, 1], [a, b], "o", color=WALLE_C, ms=3, zorder=3)
    # greedy de-collision: several checkpoints land within a hair of each other
    span = max(b for _, _, b in rows) - min(b for _, _, b in rows)
    gap, last = span * 0.055, -np.inf
    for ck, a, b in sorted(rows, key=lambda r: r[2]):
        y = max(b, last + gap)
        last = y
        if abs(y - b) > 1e-9:
            ax.plot([1.0, 1.03], [b, y], "-", color=GREY_L, lw=0.5, zorder=1)
        ax.text(1.04, y, ck.replace("-sweep", ""), fontsize=5.5, color=MUTED, va="center")
    ax.set_xticks([0, 1], ["all cases\n(n=186)", "unseen by WALLE\n(n=119)"])
    ax.set_xlim(-0.15, 1.5)
    ax.set_ylabel("WALLE AUPRC / AlphaFold3 AUPRC")
    mean_ch = 100 * np.mean([(b - a) / a for _, a, b in rows])
    ax.text(0.02, 0.03, f"{len(rows)}/{len(rows)} decline, mean {mean_ch:+.1f}%",
            transform=ax.transAxes, fontsize=6.5, color=INK)
    # No panel letter. This is a single-panel figure, and Nature's rule is about
    # "separate panels in multi-part figures", which a single-panel figure has
    # none of.
    save(fig, "figS2_walle_leakage_ratio", (89, 62))              # not placed



# ------------------------------------------------------- input-structure axis ---
AF3_TEMPLATE_CUTOFF = "2021-09-30"      # AF3's --max_template_date default; left as-is


def seen_flags():
    """Per Task II case, could each method have seen this complex already?

    The split is method-specific. "Seen" has a different definition for each
    method, so the definitions are not collapsed into one axis:
      * AlphaFold3 -- the PDB entry was released BEFORE AF3's template cutoff
        (2021-09-30, left at the default and reported rather than suppressed), so the
        solved complex was available to it. This is a proxy for training exposure,
        not proof of it.
      * ESMFold2 -- the same PDB-date split, at 2021-09-30. Two sources, and
        they do not agree to the day. The model card bundled with the weights
        loaded here gives a Data Cutoff of "Sept 2021", a month with no day. The
        preprint's supplement (A.2.9.1) says "For benchmarking, we train on PDB
        structures released prior to 2021-09-30", and A.2.9.2 builds the
        validation sets "between 2021-09-30 and 2023-01-13, corresponding to the
        training and evaluation date cutoffs" -- but neither sentence names the
        RELEASED checkpoint, and the day-level one is qualified "for
        benchmarking". 2021-09-30 is the end of the month the card gives, so the
        split is correct under either reading, which is why it is used. The two
        limits are different in kind -- AlphaFold3's bounds what it may look up
        at run time, ESMFold2's what it was fitted on -- and the legend says so.
      * WALLE -- exact membership of the AsEP train split (PDB id or VH identity
        >=0.95 against a Spike-antigen training complex). This is exact, because we
        have the split.
      * the blind methods have no training at all, so they get no split. Their flat
        distribution across both halves is the control that says the two halves are
        not simply of different difficulty.
    A merged case counts as seen if any of its source PDB ids does, matching the rule
    used throughout the leakage analysis.
    """
    man = pd.read_csv(PROJECT / "data/manifests/task2_manifest.csv")
    man = man[man.chain_role_ok]
    dates = pd.read_csv(PROJECT / "data/manifests/pdb_release_dates.csv")
    dmap = dict(zip(dates.pdb_id.str.upper(), dates.released.astype(str)))

    def earliest(pdb_field):
        ds = [dmap.get(x.strip().upper()) for x in str(pdb_field).split(";")]
        ds = [d for d in ds if d and d != "nan"]
        return min(ds) if ds else None

    man["released"] = man.pdb_id.map(earliest)
    man["pdb_pre_cutoff"] = man.released.lt(AF3_TEMPLATE_CUTOFF).astype(float)
    man.loc[man.released.isna(), "pdb_pre_cutoff"] = np.nan

    lk = PROJECT / "data/manifests/walle_leakage.csv"
    if lk.exists():
        l = pd.read_csv(lk)
        col = "leaked_ratio_any" if "leaked_ratio_any" in l else "leaked_ratio"
        man = man.merge(l[["case_id", col]].rename(columns={col: "walle_leaked"}),
                        on="case_id", how="left")
    return man.set_index("case_id")


def figS3_structure_source(ckpt):
    """(a) does WALLE want a real antigen structure?  (b) does having seen the
    complex already explain the score?

    Panel a turns on the residue pool. An experimental structure is missing every
    unresolved residue, so it exposes ~616 surface residues where the AF3 structure
    exposes ~1,157, and scoring each condition on its own pool compares two
    different questions with the pool difference alone moving AUPRC. Everything
    here is scored on the 579 residues both inputs resolve, for the 183 cases that
    have both. Paired, so the test is signed-rank.

    Panel b is the leakage question, per method, on its own definition of "seen"
    (see seen_flags). Task II is the only arm where this can be asked: every Task
    II case is a solved complex and no Task III case has one, so the comparison
    would otherwise be Task II against Task III.
    """
    from scipy.stats import wilcoxon, mannwhitneyu
    pcp = PROJECT / "results/task2/walle/source_comparison_per_case.csv"
    amp = AMP
    if not pcp.exists():
        print("  (no source_comparison_per_case.csv -- run scripts/09 --compare_sources)")
        return
    d = pd.read_csv(pcp)
    # Two rows. A single row of four is 236.5 mm wide, over Nature's 183 mm
    # maximum, and scaling 236.5 -> 183 is 77.4%, which takes the 5.5 pt n=
    # annotations in panel d to 4.3 pt, under the 5 pt floor. a|b|c on top and d
    # across the bottom is the same content at 183 mm with every type size
    # untouched.
    fig = plt.figure(figsize=(COL2, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.78],
                          height_ratios=[1.0, 1.05], wspace=0.34, hspace=0.46)

    # ---------------- (a) AF3-predicted vs experimental antigen ----------------
    ax = fig.add_subplot(gs[0, 0])
    sub = d[d.checkpoint == ckpt]
    # Short tick labels, since the full names run into each other in this narrow
    # panel. The input condition is spelled out in the legend.
    # The checkpoint name goes in the tick label. It names which WALLE the first
    # two columns are, so it belongs under "WALLE" and over the input condition
    # rather than above the panel, where it would read as a title.
    series = [(f"WALLE\n{ckpt.replace('-sweep', '')}\nAF3 ag.",
               "WALLE (AF3-predicted input)", WALLE_C),
              (f"WALLE\n{ckpt.replace('-sweep', '')}\nexp. ag.",
               "WALLE (experimental input)", "#E2A08C"),
              ("AlphaFold3\ncontact", "AF3 contact score", AF3_C)]
    wide = {lab: sub[sub.method == key].set_index("case_id").auprc
            for lab, key, _c in series if (sub.method == key).any()}
    x = np.arange(len(wide))
    rng = np.random.default_rng(0)
    for xi, ((lab, v), (_l, _k, col)) in enumerate(zip(wide.items(), series)):
        parts = ax.violinplot([v.to_numpy()], positions=[xi], widths=0.72,
                              showextrema=False)
        parts["bodies"][0].set_facecolor(col)
        parts["bodies"][0].set_alpha(0.30)
        parts["bodies"][0].set_edgecolor(col)
        parts["bodies"][0].set_linewidth(0.8)
        ax.plot(xi + rng.uniform(-0.09, 0.09, len(v)), v, ".", ms=1.6, color=col,
                alpha=0.40, zorder=3)
        ax.plot([xi - 0.26, xi + 0.26], [v.median()] * 2, color=col, lw=1.9, zorder=4)
        ax.text(xi, 1.035, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=6,
                color=col)
    keys = list(wide)
    if len(keys) >= 2:
        a, b = wide[keys[0]], wide[keys[1]]
        common = a.index.intersection(b.index)
        pv = wilcoxon(a[common], b[common]).pvalue         # two-sided: no direction
        # Clear of the means. The per-violin means are printed at y = 1.035 and
        # run to about 1.08 at 6 pt, so the bracket sits at 1.18 and the y limit
        # is raised to match.
        ax.plot([0, 0, 1, 1], [1.18, 1.22, 1.22, 1.18], color=INK, lw=0.5, clip_on=False)
        ax.text(0.5, 1.23, f"{_stars(pv)}  p={pv:.2f}", ha="center", va="bottom",
                fontsize=6, color=INK if pv < 0.05 else MUTED)
    ax.set_xticks(x, keys, fontsize=6.3)
    ax.set_ylim(-0.03, 1.34)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("AUPRC on the matched residue pool")
    titled_panel(ax, "a", "Predicted against experimental antigen")

    # ------------- (b) the same question across all ten checkpoints -----------
    # One checkpoint is one draw. decent-2 prefers the AF3 antigen by a
    # significant +0.028, which does not generalise: averaged over the ten
    # released checkpoints the two inputs are indistinguishable (0.176 vs 0.176)
    # and the experimental antigen is ahead on 6 of 10. The full set is shown
    # beside panel a for that reason.
    ax = fig.add_subplot(gs[0, 1])
    piv = (d[d.method.str.startswith("WALLE")]
           .groupby(["checkpoint", "method"]).auprc.mean().unstack())
    ka, ke = "WALLE (AF3-predicted input)", "WALLE (experimental input)"
    sub_title = "b   All 10 WALLE checkpoints"
    if ka in piv and ke in piv:
        for _ck, r in piv.iterrows():
            ax.plot([0, 1], [r[ka], r[ke]], "-", color=WALLE_C, lw=0.8, alpha=0.7, zorder=2)
            ax.plot([0, 1], [r[ka], r[ke]], "o", color=WALLE_C, ms=2.8, zorder=3)
        # The mean over the ten checkpoints, which is the panel's claim: averaged
        # over all ten the two inputs are indistinguishable (0.176 vs 0.176).
        # Drawn thin and dashed with a marker at each end, so it reads as a
        # distinct series without covering the ten slopes it summarises.
        ax.plot([0, 1], [piv[ka].mean(), piv[ke].mean()], color=INK, lw=0.9,
                ls=(0, (2.4, 1.2)), zorder=5)
        ax.plot([0, 1], [piv[ka].mean(), piv[ke].mean()], "s", color=INK,
                ms=2.6, zorder=6)
        ax.text(1.06, piv[ke].mean(), "mean", fontsize=5.5, color=INK,
                va="center", ha="left")
        nbetter = int((piv[ke] > piv[ka]).sum())
        # Goes in the title because ten slope lines leave no empty corner in the plot.
        sub_title += f"\nexperimental ahead on {nbetter}/{len(piv)}"
    ax.set_xticks([0, 1], ["AF3\nantigen", "experi-\nmental"], fontsize=6.3)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("mean AUPRC")
    titled_panel(ax, "b", "All 10 WALLE checkpoints")


    # ---------------- (d) seen vs unseen, per method --------------------------
    ax = fig.add_subplot(gs[1, :])
    flags = seen_flags()
    dm = pd.read_csv(amp)
    dm = dm[dm.method != "AlphaFold3 (5x5 ensemble)"]
    order = [m for m in BLIND if m in set(dm.method)] \
        + [m for m in ["ESMFold2"] if m in set(dm.method)] \
        + sorted(m for m in dm.method.unique() if m.startswith("WALLE")) + ["AlphaFold3"]
    order = [m for m in order if m in set(dm.method)]

    def split_for(m):
        """Which 'seen' definition applies (None = draw one unsplit box).

        A method is split where "seen" can be defined from something we hold.
        AlphaFold3 has a published template cutoff, 2021-09-30, a run-time limit
        on which deposited structures it may use. ESMFold2's model card gives a
        training data cutoff of September 2021 for the biohub/ESMFold2
        checkpoint this pipeline loads, and its preprint puts the day at
        2021-09-30; the split uses 2021-09-30, the end of that month, so it
        holds under either. WALLE has its AsEP training split. The
        antibody-blind methods are unsplit because they have no training.
        """
        if m in BLIND:
            return None
        if m.startswith("WALLE"):
            return ("walle_leaked", "in AsEP train", "unseen")
        return ("pdb_pre_cutoff", "PDB pre-cutoff", "post-cutoff")

    x = np.arange(len(order))
    w = 0.34
    for xi, m in zip(x, order):
        v = dm[dm.method == m].set_index("case_id").auprc
        col = method_colour(m)
        spec = split_for(m)
        if spec is None:                       # no training: one distribution, centred
            ax.boxplot([v.to_numpy()], positions=[xi], widths=0.52, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=0.35, edgecolor=col, lw=0.8),
                       medianprops=dict(color=INK, lw=1.1),
                       whiskerprops=dict(color=MUTED, lw=0.7),
                       capprops=dict(color=MUTED, lw=0.7))
            continue
        key, _l1, _l0 = spec
        f = flags[key].reindex(v.index)
        for j, (mask, hatch) in enumerate([(f == 1, "////"), (f == 0, None)]):
            vv = v[mask.fillna(False)].to_numpy()
            if not len(vv):
                continue
            pos = xi + (j - 0.5) * w
            bp = ax.boxplot([vv], positions=[pos], widths=w * 0.82, showfliers=False,
                            patch_artist=True,
                            medianprops=dict(color=INK, lw=1.1),
                            whiskerprops=dict(color=MUTED, lw=0.7),
                            capprops=dict(color=MUTED, lw=0.7))
            b = bp["boxes"][0]
            b.set_edgecolor(col); b.set_linewidth(0.9)
            if hatch:
                b.set_facecolor("none"); b.set_hatch(hatch)
            else:
                b.set_facecolor(col); b.set_alpha(0.35)
            ax.text(pos, 0.86, f"n={len(vv)}", ha="center", va="bottom", fontsize=figstyle.TINY_PT,
                    color=MUTED, rotation=90)
        g1 = v[(f == 1).fillna(False)].to_numpy()
        g0 = v[(f == 0).fillna(False)].to_numpy()
        if len(g1) > 2 and len(g0) > 2:
            # Independent groups of antibodies, so Mann-Whitney rather than
            # signed-rank. Two-sided: there is no prior that having seen a
            # complex must help, and a method doing worse on complexes it has
            # seen would be a reportable result.
            pv = mannwhitneyu(g1, g0, alternative="two-sided").pvalue
            ax.text(xi, 1.045, _stars(pv), ha="center", va="bottom", fontsize=6,
                    color=INK if pv < 0.05 else MUTED)
    ax.set_xticks(x, [short_label(m).replace("\n", " ") for m in order], fontsize=6,
                  rotation=32, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.02, 1.16)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_ylabel("per-antibody AUPRC")
    ax.legend(handles=[Patch(facecolor="none", edgecolor=INK, hatch="////",
                             label="method could have seen this complex"),
                       Patch(facecolor=GREY_L, edgecolor=INK, label="it could not"),
                       Patch(facecolor=GREY_L, edgecolor=INK, alpha=0.35,
                             label="no training → not split")],
              loc="upper left", fontsize=5.8, ncol=1)
    titled_panel(ax, "c", "Every trained method, split on possible prior exposure")
    save(fig, "figS3_structure_source_and_exposure", (180, 110))  # ED 1

def main():
    print("Task II panels")
    pc, wal, swap = load_task2()
    # One checkpoint for every task and panel, from config/walle.py.
    ck = ck3 = t3_ckpt()
    print(f"  WALLE checkpoint, Task II panels: {ck}")
    print(f"  WALLE checkpoint, Task III panels: {ck3}")
    pcp = PROJECT / "results/task2/walle/af3/per_case.csv"
    if AMP.exists():
        for kind in ("violin", "box"):
            fig3d_dist(kind)
        fig3e_by_group(pc, pd.read_csv(pcp) if pcp.exists() else None, ck)
    figS2(pd.read_csv(PROJECT / "results/task2/walle/af3/summary.csv"))
    sw = PROJECT / "results" / "task2" / "swap_all_methods.csv"
    if sw.exists():
        fig3f_conditioning(sw, PROJECT / "results" / "task2" / "swap_all_methods_per_case.csv")

    print("Task III panels")
    arms3 = []
    for name, task in [("HIV-1 Env", "task3"), ("SARS-CoV-2 spike", "task3cov")]:
        r = load_t3(task)
        if r:
            arms3.append((name, r[0], r[1], r[2]))
    if arms3:
        fig4b_binary(arms3, ck3)       # ED 2a
        fig4c(arms3, ck3)              # Fig 4c (main)
        fig4d(arms3, ck3)              # ED 2e, the full confusion matrices

    print("Supplementary")
    p = OUT / "plddt_by_bin.json"
    if p.exists():
        figS1a(p)
    if CONF.exists():
        figS1b()
        if arms3:
            figS1c(arms3)
    else:
        print("  (no confidence table -- run scripts/17_confidence_table.py)")
    figS3_structure_source(ck)
    print(f"\nfigures in {OUT}")



# --------------------------------------------------------------- fig 3d v2 ----
BLIND = ["Random\n(AUPRC floor)", "RBD mask 319-541", "RBM mask 436-507", "BepiPred-3.0"]
REF = "RBM mask 436-507"       # strongest antibody-blind baseline = the bar to clear


def _stars(p, alpha=0.05):
    """One star or n.s., never *** / ** / *.

    One star marks significant and n.s. marks not significant, across the
    manuscript.

    The star does not encode a magnitude, so the exact P appears in the figure
    legend for every panel that draws one. A reader cannot recover "how
    significant" from the panel, which is intended: a graded *** invites the
    over-reading that one-sided testing is meant to stop.

    `alpha` is a parameter rather than a constant because two panels carry a
    multiple-comparison correction: Task III's eight domain tests use Bonferroni
    0.05/8 = 0.00625, under which spike ESMFold2 (P = 0.024) is not significant
    and gets no star.
    """
    return "*" if p < alpha else "n.s."


def fig3d_dist(kind="violin"):
    """Task II AUPRC distributions for every method, blind baselines greyed.

    Distributions rather than bars. Per-antibody AUPRC is strongly right-skewed:
    most antibodies sit near the floor and a minority are essentially solved, so a
    bar of means reports a number no individual antibody has.

    The greyed methods are antibody-blind. They cannot take an antibody as input at
    all, so they return the same answer for every one, and they are the bar to
    clear rather than competitors. BepiPred-3.0 is grey for that reason, not
    because it scores low.

    Significance is a Wilcoxon signed-rank of each method against the strongest
    blind baseline (the RBM mask), which is the benchmark's question: does using
    the antibody beat not using it? Signed-rank rather than Mann-Whitney because
    the samples are paired -- the same 186 antibodies scored by both methods -- so
    pairing removes between-antibody variation (some epitopes are simply easier),
    which MWU would throw away.
    """
    from scipy.stats import wilcoxon
    d = pd.read_csv(AMP)
    secondary = "AlphaFold3 (5x5 ensemble)"     # drawn as a marker, not its own column
    order = BLIND + [m for m in ["ESMFold2"] if m in set(d.method)] \
        + sorted([m for m in d.method.unique() if m.startswith("WALLE")]) + ["AlphaFold3"]
    order = [m for m in order if m in set(d.method)]
    vals = {m: d[d.method == m].set_index("case_id").auprc for m in order}
    ref = vals[REF]

    colour = {m: method_colour(m) for m in order}

    # Half width. This panel shares a row with the antibody-swap panel, so it is
    # drawn for an 89 mm slot rather than the full 183 mm. Nature BE: "Consider
    # the content and legibility of each panel and let that define its size
    # within the figure." Eight violins of a 0-1 quantity need room for eight
    # method names, which rotation provides, rather than 183 mm.
    fig, ax = plt.subplots(figsize=(COL1 * 0.95, 2.4))
    x = np.arange(len(order))
    data = [vals[m].to_numpy() for m in order]
    if kind == "violin":
        parts = ax.violinplot(data, positions=x, widths=0.72, showextrema=False)
        for body, m in zip(parts["bodies"], order):
            body.set_facecolor(colour[m]); body.set_alpha(0.30)
            body.set_edgecolor(colour[m]); body.set_linewidth(0.8)
        for xi, m in zip(x, order):
            ax.plot([xi - 0.26, xi + 0.26], [np.median(vals[m])] * 2,
                    color=colour[m], lw=2.0, zorder=4)
    else:
        bp = ax.boxplot(data, positions=x, widths=0.55, showfliers=False,
                        patch_artist=True, medianprops=dict(color=INK, lw=1.2))
        for patch, m in zip(bp["boxes"], order):
            patch.set_facecolor(colour[m]); patch.set_alpha(0.35)
            patch.set_edgecolor(colour[m]); patch.set_linewidth(0.9)
        for el in ("whiskers", "caps"):
            for ln in bp[el]:
                ln.set_color(MUTED); ln.set_linewidth(0.7)
    rng = np.random.default_rng(0)
    for xi, m in zip(x, order):
        v = vals[m].to_numpy()
        ax.plot(xi + rng.uniform(-0.09, 0.09, len(v)), v, ".", ms=1.6,
                color=colour[m], alpha=0.40, zorder=3)
    # AF3 is reported on its top-ranked structure, matching WALLE and ESMFold2.
    # The marker shows where the 25-structure ensemble would put the median, so
    # the choice is visible.
    if secondary in set(d.method) and "AlphaFold3" in order:
        xi = x[order.index("AlphaFold3")]
        ens = d[d.method == secondary].auprc.median()
        ax.plot([xi - 0.34, xi + 0.34], [ens] * 2, color=INK, lw=1.0,
                ls=(0, (2, 1.5)), zorder=6)
        # No in-plot key. Nature's formatting guide asks for keys in the legend
        # rather than on the figure, and at 89 mm this one would be the widest
        # object in the panel. The dashed rule is defined in the caption.

    # Significance against the strongest blind baseline, one-sided (greater).
    # The benchmark's question is directional -- does using the antibody beat not
    # using it -- and a two-sided test would put stars on the random floor and the
    # RBD mask for being significantly worse than the RBM mask, which reads as if
    # they had won.
    print(f"\n  Task II, vs {REF}, Wilcoxon one-sided greater (n={len(ref)} antibodies):")
    for xi, m in zip(x, order):
        if m == REF:
            continue
        common = ref.index.intersection(vals[m].index)
        w = wilcoxon(vals[m][common], ref[common], alternative="greater").pvalue
        print(f"    {m:24s} median {vals[m].median():.3f} | p={w:.2e} {_stars(w)}")
        ax.text(xi, 1.06, _stars(w), ha="center", va="bottom", fontsize=6.5,
                color=INK if w < 0.05 else MUTED)
    ax.text(x[order.index(REF)], 1.06, "ref", ha="center", va="bottom",
            fontsize=6, color=MUTED, style="italic")

    # Rotated, one line each. At 89 mm a tick slot is ~9.5 mm and "BepiPred-3.0"
    # is ~13 mm at 6 pt, so upright labels collide whatever the line breaks.
    # Rotation turns the labels into parallel lines whose perpendicular
    # separation is 9.5*sin45 = 6.7 mm against a ~2.3 mm cap height, so they
    # cannot touch at any width. The cost is height: the longest label, "WALLE
    # (treasured-8)" at ~22 mm, projects 22*sin45 = 15.6 mm downwards.
    ax.set_xticks(x, [short_label(m).replace("\n", " ") for m in order],
                  fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(-0.02, 1.14)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.axhspan(-0.02, 1.14, xmin=0, xmax=(len(BLIND)) / len(order),
               color=GREY_L, alpha=0.18, zorder=0, lw=0)
    # At the top of the band, matching 3c and 3d, so the three Figure 3 panels
    # name the band in the same place. The blind distributions all die out below
    # 0.45, so 0.96 is clear of them, and the significance row sits higher still
    # at 1.06.
    ax.text(len(BLIND) / 2 - 0.5, 0.96, "antibody-blind", fontsize=6.5, color=MUTED,
            ha="center", va="center")
    panel_letter(ax, "b")
    save(fig, f"fig3d_task2_all_methods_{kind}", (89, 50))        # Fig 3b


if __name__ == "__main__":
    main()
