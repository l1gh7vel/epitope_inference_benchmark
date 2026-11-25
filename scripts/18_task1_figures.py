#!/usr/bin/env python3
"""Generate the Task I antigen-group composition and performance panels.

Panel 2c is drawn over the 3,804 complexes every method scored, so the four
distributions are comparable.
Random AUPRC is computed per case because antigen length and epitope prevalence
vary. Performance is shown by antigen group and relative to that case-specific
floor.

    PYTHONPATH=scripts:config python scripts/18_task1_figures.py --arm evaluated
    PYTHONPATH=scripts:config python scripts/18_task1_figures.py --arm full
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                        # noqa: E402
import walle                                               # noqa: E402

import figstyle                                            # noqa: E402
from figstyle import COL1, COL2, fit_save, panel_letter    # noqa: E402

OUT = PROJECT / "results" / "paper_figures"

AF3_C, WALLE_C, THIRD_C = "#2C6FAD", "#C1553B", "#7B5EA7"
GREY_D, GREY_M, GREY_L = "#6E6E69", "#9A9A95", "#C7C7C2"
BEPI_C = "#BDBDB8"
INK, MUTED, SURFACE = "#1A1A1A", "#5C5C57", "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    **figstyle.BASE_RC,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": False, "legend.frameon": False,
})

# Same colour rule as every other figure: the entity keeps its colour, and the
# antibody-blind methods are achromatic because they are the floor, not a model.
# WALLE carries its CHECKPOINT in the label wherever it appears. The AsEP
# authors publish ten checkpoints and name no recommended one, so "WALLE" alone
# is ambiguous between models that differ by more than the gap between methods.
# The label names the checkpoint, from config/walle.py.
# FIG 2C SIGNIFICANCE COMPARATOR. The Task III accuracy panel tests against the
# always-<commonest domain> baseline and this panel tests against the random
# floor; each is the strongest simple baseline for its own task.
#
# "floor"  each method against its OWN per-case random floor, paired per
#          antibody within each antigen group, one-sided, Bonferroni over the
#          20 method x group tests. This is what is drawn.
# "BepiPred-3.0"  the same construction against the strongest ANTIBODY-BLIND
#          method on Task I.
#
# WHAT THE FLOOR TEST ACTUALLY SHOWS, as measured: all 20 of 20 tests
# are significant, including BepiPred-3.0, which takes no antibody as input at
# all. Beating a per-case prevalence floor is a low bar and every method clears
# it in every group, so the drawn row is uniform and carries no information
# beyond "nothing here is at chance". That is worth stating once; it is not
# worth twenty stars. The alternative discriminates -- against BepiPred-3.0,
# ESMFold2 is n.s. in group 1 (the spike group) and group 5 and significant in
# 2, 3 and 4, while AlphaFold3 and WALLE clear it everywhere. Switching is one
# line, deliberately, so the choice stays visible rather than baked in.
FIG2C_REF = "floor"

WALLE_LABEL = f"WALLE ({walle.HEADLINE_CHECKPOINT.replace('-sweep', '')})"
# THE FLOOR AND BEPIPRED-3.0 ARE THE TWO ENDS OF THE GREY RAMP, not neighbours
# on it, and each is a different shade of grey that is consistent across all
# figures. The floor is GREY_D, the darkest; BepiPred-3.0 is BEPI_C, which is the
# colour it carries in every Task II and Task III panel, so Task I is
# consistent with the rest of the paper rather than using a Task I scheme. The
# two are #6E6E69 against #BDBDB8, which separates at point size where adjacent
# steps of the ramp did not.
COLOUR = {"Random (per-case floor)": GREY_D, "BepiPred-3.0": BEPI_C,
          "ESMFold2": THIRD_C, "WALLE": WALLE_C, WALLE_LABEL: WALLE_C,
          "AlphaFold3": AF3_C}
ORDER = ["Random (per-case floor)", "BepiPred-3.0", "ESMFold2", "WALLE", "AlphaFold3"]


def save(fig, name, mm):
    """Write the panel at its FINAL print size -- see scripts/figstyle.py."""
    return fit_save(fig, OUT, name, mm)


def load(arm="full"):
    """Every method's per-case table for this arm, plus the shared floor column."""
    out = {}
    # This glob turns every matching table into a violin, so nothing else may be
    # written beside the per-method tables under this name.
    for f in sorted(cfg.RESULTS.glob(f"per_case_*_{arm}.csv")):
        method = f.stem[len("per_case_"):-len(f"_{arm}")].replace("_", " ")
        out[method] = pd.read_csv(f)
    return out


def violin(ax, x, v, colour, width=0.70, pts=True, rng=None, outline=False):
    """outline=True draws an unfilled violin.

    Used for the random floor. It has to sit next to BepiPred-3.0, which is also
    achromatic (it is antibody-blind), and two similar greys side by side are not
    separable in print -- so the floor is a dashed outline and the method is filled.
    """
    if len(v) < 2:
        return
    parts = ax.violinplot([v], positions=[x], widths=width, showextrema=False)
    b = parts["bodies"][0]
    if outline:
        b.set_facecolor("none"); b.set_alpha(1.0)
        b.set_edgecolor(GREY_D); b.set_linewidth(0.8); b.set_linestyle((0, (2.5, 1.5)))
    else:
        b.set_facecolor(colour); b.set_alpha(0.32)
        b.set_edgecolor(colour); b.set_linewidth(0.7)
    if pts and rng is not None:
        # The floor's points carry a touch more weight than a method's: its
        # values are packed into the bottom of the axis, so at alpha 0.40 they
        # merged into their own median rule.
        ax.plot(x + rng.uniform(-0.09, 0.09, len(v)), v, ".",
                ms=1.5 if outline else 1.3, color=colour,
                alpha=0.55 if outline else 0.40, zorder=3)
    ax.plot([x - 0.24, x + 0.24], [np.median(v)] * 2, color=colour, lw=1.7, zorder=4)


def fig2c(arm="full"):
    """Plot per-case AUPRC and case-specific random floors by antigen group."""
    data = load(arm)
    if not data:
        print("  no Task I per-case tables yet")
        return
    # GUARD: the WALLE table must come from the checkpoint this panel LABELS.
    # Every figure that resolves its checkpoint at draw time follows the
    # selected checkpoint, but Task I reads a per-case table produced by a
    # separate t1_05_evaluate run. If that run used a different checkpoint, the
    # panel would carry one label over another checkpoint's numbers (mean 0.2508
    # for decent-2 against 0.2648 for treasured-8). The table records the
    # checkpoint, and this refuses to draw a panel whose label does not match it.
    import json as _json
    _mj = cfg.RESULTS / f"metrics_WALLE_{arm}.json"
    if _mj.exists():
        _ck = _json.loads(_mj.read_text()).get("checkpoint")
        _want = walle.HEADLINE_CHECKPOINT
        if _ck and _want and _ck != _want:
            raise SystemExit(
                f"fig2c would mislabel WALLE: the per-case table was built from "
                f"{_ck!r} but this variant draws {_want!r}. Re-run\n"
                f"  scripts/t1_05_evaluate.py --scores_dir "
                f"work/task1/walle/scores/experimental/{_want} "
                f"--method WALLE --arm {arm}")
    methods = [m for m in ORDER[1:] if m in data]
    if not methods:
        return
    # On the full arm, drop any method scored on a different cohort.
    # A table holding no more than half the largest one is a different cohort,
    # and drawing it beside the others would put two cohorts in one panel. The
    # filter is on the DATA, not a hardcoded method list.
    if arm == "full":
        big = max(len(data[m]) for m in methods)
        kept = [m for m in methods if len(data[m]) > 0.5 * big]
        dropped = [m for m in methods if m not in kept]
        if dropped:
            print(f"  full arm: omitting {', '.join(dropped)} -- their 'full' "
                  f"table is a smaller cohort ({', '.join(str(len(data[m])) for m in dropped)} "
                  f"rows against {big})")
        methods = kept
    folds = sorted(data[methods[0]].fold.unique())
    floor_src = max(data.values(), key=len)
    # The manifest, restricted to exactly the arm being drawn, for the tick
    # labels below. Read here rather than inside the label loop so a missing
    # manifest degrades to the old two-line label instead of raising.
    # The length cap belongs here. Without it the tick labels would be computed
    # on usable & vh_dedup_keep, 4,177 rows, rather than on the cohort the tick
    # labels describe, so group 2 would print 19% where its own panel's cohort
    # gives 21%.
    try:
        _m = pd.read_csv(cfg.MANIFEST)
        man_sub = _m[_m.usable & _m.vh_dedup_keep &
                     (_m.antigen_len <= cfg.MAX_ANTIGEN_LEN)]
    except Exception:
        man_sub = None

    rng = np.random.default_rng(0)
    # Bonferroni over every method x group test drawn on this panel.
    global ALPHA_2C
    n_tests = len(methods) * len(folds)
    ALPHA_2C = 0.05 / max(n_tests, 1)
    fig, ax = plt.subplots(figsize=(COL2 * 0.95, 3.9))
    series = ["Random (per-case floor)"] + methods
    step = len(series) + 1.0
    for fi, f in enumerate(folds):
        base = fi * step
        for si, s_ in enumerate(series):
            x = base + si
            if s_.startswith("Random"):
                v = floor_src[floor_src.fold == f].floor.to_numpy()
                mult = 1.0
            else:
                sub = data[s_][data[s_].fold == f]
                v = sub.auprc.to_numpy()
                mult = float(np.median(sub.enrichment))
            # Points are drawn for the floor too. With BepiPred-3.0 at the far
            # end of the grey ramp the two achromatic series do not collide,
            # and a floor drawn without points would make a 57-93-value
            # distribution read as a single line.
            violin(ax, x, v, COLOUR[s_], rng=rng, pts=True,
                   outline=s_.startswith("Random"))
            if len(v) >= 2:
                # median multiple of that case's OWN floor: how much better than
                # random, per antigen, rather than against a pooled baseline
                star = ""
                if not s_.startswith("Random"):
                    sub_m = data[s_][data[s_].fold == f]
                    if FIG2C_REF == "floor":
                        a_, b_ = sub_m.auprc, sub_m.floor
                    else:
                        r_ = data[FIG2C_REF]
                        r_ = r_[r_.fold == f].set_index("case_id")
                        a0 = sub_m.set_index("case_id")
                        j = a0.index.intersection(r_.index)
                        a_, b_ = a0.loc[j, "auprc"], r_.loc[j, "auprc"]
                    if len(a_) >= 2 and not np.allclose(a_, b_):
                        pv = wilcoxon(a_, b_, alternative="greater",
                                      zero_method="wilcox").pvalue
                        star = " *" if pv < ALPHA_2C else " n.s."
                ax.annotate(f"{mult:.1f}x{star}", xy=(x, 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6,
                            color=MUTED if s_.startswith("Random") else COLOUR[s_])
        # NO SHADED BAND behind antigen group 1. The tick label already says "mostly spike" and
        # gives the cluster count, so the band was a second, heavier telling of
        # something the axis states in words. It also competed with the grey of
        # the two achromatic series, which are grey because they are floors.

    ax.set_ylabel("per-antibody AUPRC")
    ax.set_ylim(-0.03, 1.02)
    labs = []
    for f in folds:
        # The cluster count describes the GROUP, so it comes from the same
        # evaluated-set manifest fig2b counts, not from the table drawn here.
        # Sourcing it from the drawn data made 2c print 237-243 against 2b's
        # 244-249 for the same five groups once 2c moved to the intersection.
        n_cl = (int(man_sub[man_sub.fold == f].Cluster.nunique())
                if man_sub is not None and "Cluster" in man_sub
                else floor_src[floor_src.fold == f].cluster.nunique())
        # The label says "group". The `fold` COLUMN keeps its name because it is in
        # the manifest and renaming data breaks every reader of it. "fold" has
        # three meanings in this repo (antigen group, protein folding,
        # fold-change) and only the first is renamed.
        # THIRD LINE NAMES THE TOP ANTIGEN FAMILY AND ITS SHARE, ALWAYS
        # (e.g. HIV-1 Env, x%), rather than a count of distinct antigens.
        #
        # The unit is a family, not a raw name or a cluster. Raw `antigen_name`
        # strings are near-singletons outside group 1 -- the commonest in
        # group 2 is 2% -- because one antigen is deposited under many names
        # and many constructs. Collapsing to a family (spike, HIV-1 Env,
        # influenza HA, ...) is what makes the label mean something to a
        # reader: group 2 is 20% HIV-1 Env, which is the useful fact, while its
        # largest single CLUSTER is 1%.
        # Two matching traps are avoided here. Matching the bare substring
        # "envelope glyco" and calling the result "HIV-1 Env" sweeps in
        # herpesvirus glycoprotein B and L, Epstein-Barr gp350 and
        # cytomegalovirus UL130 -- none of them HIV. Calling every protein
        # named "spike" SARS-CoV-2 spike puts three OTHER coronavirus spikes
        # (1,349-1,353 residues, accessions R9UQ53 / Q696P8 / A0A2R4KP93) under
        # that name in group 5.
        # HIV is therefore matched only on gp120 / gp41 / gp160 / an explicit "hiv",
        # which are unambiguous, and the spike family resolves its DISPLAY name
        # from the dominant UniProt accession in the group: P0DTC2 is SARS-CoV-2
        # spike, anything else is named generically.
        FAMILY = [("spike", "spike"), ("gp120", "HIV-1 Env"),
                  ("gp160", "HIV-1 Env"), ("gp41", "HIV-1 Env"),
                  ("hiv", "HIV-1 Env"),
                  ("hemagglutinin", "influenza HA"), ("neuraminidase", "influenza NA"),
                  ("nucleoprotein", "nucleoprotein"), ("capsid", "viral capsid"),
                  ("lysozyme", "lysozyme"), ("interleukin", "interleukin"),
                  ("mhc", "MHC"), ("insulin", "insulin")]
        #: accessions whose species can be named on the page without guessing.
        ACCESSION_NAME = {"P0DTC2": "SARS-CoV-2 spike"}

        def _family(nm):
            nm = str(nm).lower()
            for key, fam in FAMILY:
                if key in nm:
                    return fam
            return None

        def _display(fam, rows):
            """Family name to print, resolved by accession where that is safe."""
            if fam != "spike":
                return fam
            if "uniprot_id" in rows:
                vc = rows.uniprot_id.dropna().value_counts()
                if len(vc) and vc.iloc[0] > len(rows) / 2:
                    return ACCESSION_NAME.get(vc.index[0], "coronavirus spike")
            return "coronavirus spike"

        # The family is a property of the antigen, not of its name string.
        # Matching the name alone misses rows whose depositor called the same
        # sequence something else: 29 complexes in group 1 carry the identical
        # spike sequence under "surface glycoprotein", "SARS-CoV-2 RBD" or
        # "unknown", so a name-only match returned 92.4% for group 1 while 2b,
        # which counts the antigen SEQUENCE, printed 94.4% beside it. A family
        # containing fewer complexes than one of its own sequences is a
        # contradiction on the page. The family is therefore decided per
        # antigen cluster by a majority vote of the names within it and then
        # propagated to every row of that cluster, which recovers 61 rows and
        # takes group 1 to 95.7%. No group's top family changes.
        if man_sub is not None and "antigen_id" in man_sub:
            _named = man_sub.assign(_f=man_sub.antigen_name.fillna("?").map(_family))
            _named = _named.dropna(subset=["_f"])
            _vote = _named.groupby("antigen_id")._f.agg(
                lambda col: col.value_counts().index[0])
        else:
            _vote = None

        gsub = man_sub[man_sub.fold == f] if man_sub is not None else None
        third = None
        if gsub is not None and len(gsub):
            famcol = (gsub.antigen_id.map(_vote) if _vote is not None
                      else gsub.antigen_name.fillna("?").map(_family))
            fams = famcol.dropna().value_counts()
            if len(fams):
                fam, n_ = fams.index[0], int(fams.iloc[0])
                sub = gsub[famcol == fam]
                nm = _display(fam, sub)
                third = f"{nm}, {100 * n_ / len(gsub):.0f}%"
                # THE ACCESSION. A family is matched by NAME, so HIV-1 Env is
                # named under three groups and reads as one protein split
                # across them, which would break the antigen-disjoint design.
                # The commonest UniProt accession says otherwise: group 1 is
                # P0DTC2 over and over in a single cluster, while each HIV-1
                # Env group is many clusters (below 90% sequence identity to
                # one another) led by a different accession, none shared. Only
                # the top accession is drawn, and it carries the cluster count
                # with it, because one accession out of twelve clusters means
                # something different from one out of one.
                if "uniprot_id" in sub:
                    acc = sub.uniprot_id.dropna().value_counts()
                    if len(acc):
                        # ACCESSION PLUS THE OTHER CLUSTERS, counted in
                        # CLUSTERS and not in accessions, e.g. "HIV-1 Env
                        # (UP: Q2N0S6 & 10 others)"; a single-cluster group
                        # prints the accession alone rather than "& 0 others".
                        # Clusters are the unit the split is built
                        # on, and they are the unit that carries the point: one
                        # cluster means the same antigen over and over, eleven
                        # means eleven different ones. Accessions would say the
                        # opposite here -- group 1 is a single cluster holding
                        # nine accessions, because 90% identity groups the
                        # sarbecoviruses together.
                        n_other = sub.Cluster.nunique() - 1
                        third += ("\n(UP: " + acc.index[0]
                                  + (")" if n_other == 0 else
                                     f" & {n_other} other)" if n_other == 1
                                     else f" & {n_other} others)"))
        lab = f"group {f}"
        if third:
            lab += f"\n{third}"
        labs.append(lab)
    ax.set_xticks([fi * step + (len(series) - 1) / 2 for fi in range(len(folds))],
                  labs, fontsize=6.5)
    ax.set_xlim(-0.8, (len(folds) - 1) * step + len(series) - 0.2)
    handles = [Patch(facecolor="none", edgecolor=GREY_D, ls=(0, (2.5, 1.5)),
                     label="Random (per-case floor)")]
    handles += [Patch(facecolor=COLOUR[s_], alpha=0.5, edgecolor=COLOUR[s_],
                      label=WALLE_LABEL if s_ == "WALLE" else s_)
                for s_ in series if not s_.startswith("Random")]
    ax.legend(handles=handles, loc="lower center", ncol=len(series), fontsize=6.2,
              handlelength=1.3, columnspacing=1.2, bbox_to_anchor=(0.5, 1.10))
    panel_letter(ax, "c")
    save(fig, f"fig2c_task1_per_fold_{arm}", (183, 52))        # Fig 2c


def fig2b(arm="full"):
    """Antigen group composition -- the panel that makes group 1 legible.

    THE ARM MUST MATCH 2c's. The bars and the legend beside them count antigen
    clusters, so both have to be read against the same cohort: a count from one
    cohort quoted against a count from another is a category error. `evaluated`
    is the cohort 2c's tick labels count clusters over -- usable, within the 1,400-residue antigen
    cap, and one occurrence of each cross-group heavy chain.
    """
    man = pd.read_csv(cfg.MANIFEST)
    if arm == "evaluated":
        man = man[man.usable & (man.antigen_len <= cfg.MAX_ANTIGEN_LEN)]
    else:
        man = man[man.usable]
    # Match 2c, which drops antibodies shared between folds. Showing the composition
    # before the filter and the performance after it would put two different cluster
    # counts for the same fold on two adjacent panels.
    if "vh_dedup_keep" in man:
        man = man[man.vh_dedup_keep]
    folds = sorted(man.fold.unique())
    # TWO sub-panels. The median antigen length is not drawn, because cluster count and the share of the commonest antigen already
    # carry the result, which is that group 1 is effectively one antigen family.
    # Length is a property of spike and not of the group construction.
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.62, 1.95),
                             gridspec_kw={"wspace": 0.42})
    x = np.arange(len(folds))
    # Y-LABELS ARE WRAPPED, and that is a height constraint, not a style choice.
    # A y-label is set along the panel's HEIGHT, so its string length is a FLOOR
    # on that height: "share of the commonest antigen" is 30 characters, ~34 mm
    # at 6.5 pt, which on its own forbids any panel shorter than ~40 mm. This
    # panel's slot in the re-laid Figure 2 is 30 mm. Two lines halve the run and
    # cost ~2.5 mm of width, which at 183 mm across three sub-panels is free.
    for ax, (key, lab, fn) in zip(axes, [
            ("Cluster", "distinct antigen\nclusters", lambda g: g.Cluster.nunique()),
            ("share", "share of the\ncommonest antigen",
             lambda g: g.antigen_id.value_counts().iloc[0] / len(g))]):
        v = [fn(man[man.fold == f]) for f in folds]
        # NO COLOUR. Highlighting group 1 in WALLE's red would be wrong for two
        # reasons: the red is a METHOD colour everywhere else in the paper, so a
        # red bar on a panel with no methods in it reads as WALLE; and the
        # panel's own axis labels and 2c's tick labels already say which group
        # is the spike group, so the highlight would be a third telling.
        ax.bar(x, v, 0.62, color=GREY_M, edgecolor=SURFACE, linewidth=0.8)
        for xi, vi in zip(x, v):
            ax.text(xi, vi, f"{vi:.2f}" if key == "share" else f"{vi:.0f}",
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x, [str(f) for f in folds])
        ax.set_xlabel("antigen group")
        ax.set_ylabel(lab, fontsize=6.5)
    panel_letter(axes[0], "b")
    save(fig, f"fig2b_task1_fold_composition_{arm}", (120, 30))  # Fig 2b


def fig2d_redundancy(arm="full"):
    """Within-antigen epitope redundancy, the mechanism behind the Task I argument.

    Task I reports a per-antibody average over all of SAbDab.
    That average can only distinguish a conditioned model from a blind one if the
    antibodies raised against one antigen bind different places. If they pile onto
    one patch, "predict the popular patch" scores well and the average cannot tell
    the two apart. Panels 2b and 2c show the average; this one shows whether the
    average means anything. Gold labels only -- no method, no prediction, nothing
    folded.

    Three sub-panels rather than one number, because the answer is
    direction-dependent. Convergence is large against a naive null, coverage saturates
    quickly, and against a null restricted to the antigenic region it disappears.
    Reporting only the first would overclaim and only the third would underclaim.

    Every number is in the caption rather than on the panel. Nature's guide
    asks for text and keys in the legend rather than on the figure, and this panel
    would otherwise carry three annotation blocks and two keys. The series are
    direct-labelled at the right-hand end of the data, which the figure guide
    prefers over a detached key.

    No method colours. Nothing in this panel is a method, so it does not borrow
    AlphaFold3's blue -- a reader who has seen 2c reads that blue as AlphaFold3.
    One neutral slate for observed, grey for the null.
    """
    # The epitope-redundancy panel is not placed in any figure and its
    # tables are not published, so skip rather than fail when they are
    # absent. It still runs wherever they exist.
    if not (cfg.RESULTS / f"redundancy_pair_hist_antigen_{arm}.csv").exists():
        print("  epitope-redundancy tables absent; panel skipped")
        return
    R = cfg.RESULTS
    hist = pd.read_csv(R / f"redundancy_pair_hist_antigen_{arm}.csv")
    sat = pd.read_csv(R / f"redundancy_saturation_antigen_{arm}.csv")
    H = json.loads((R / f"redundancy_headline_antigen_{arm}.json").read_text())

    # Okabe-Ito blue and vermillion. The figure guide asks for colour-vision-safe
    # encoding, and this pair is the standard recommendation; neither is a method
    # colour, so a reader coming from 2c cannot mistake either for AlphaFold3.
    # The previous slate-against-grey failed for a specific reason: the null is a
    # DASHED LINE drawn over filled bars, and two desaturated neighbours of the
    # same hue made it impossible to follow where it crossed them.
    OBS, NUL = "#0072B2", "#D55E00"

    per = pd.read_csv(R / f"redundancy_per_antigen_{'antigen'}_{arm}.csv")

    fig, axes = plt.subplots(1, 4, figsize=(COL2 * 0.95, 2.2),
                             gridspec_kw={"wspace": 0.52})

    # ---- left: distribution of pairwise overlap, observed against the null ----
    # The OVERLAP COEFFICIENT (shared / smaller epitope) is plotted instead of
    # Jaccard, because the two
    # epitopes in a pair routinely differ several-fold in size and Jaccard charges
    # that difference as disagreement. Jaccard is reported in the caption.
    ax = axes[0]
    ctr = 0.5 * (hist.bin_left + hist.bin_right)
    w = (hist.bin_right - hist.bin_left).iloc[0]
    tot_o, tot_n = hist.obs_OC.sum(), hist.n1_OC.sum()
    ax.bar(ctr, hist.obs_OC / tot_o, w * 0.92, color=OBS, alpha=0.60,
           edgecolor=OBS, linewidth=0.4, zorder=3)
    ax.step(np.r_[hist.bin_left, 1.0], np.r_[hist.n1_OC / tot_n, 0], where="post",
            color=NUL, lw=1.3, ls=(0, (2.2, 1.1)), zorder=6)
    ax.axvspan(0.5, 1.0, color=GREY_L, alpha=0.30, lw=0, zorder=1)
    # Direct labels placed ON the two series rather than stacked as a key. Both
    # series are only distinguishable in the first two bins -- past 0.2 the null is
    # flat against zero -- so the labels go there, beside the tall null step and
    # the tall observed bar, not out in the empty right half where neither is.
    ax.text(0.085, float(hist.n1_OC.iloc[0] / tot_n), "null", fontsize=5.8,
            color=NUL, ha="left", va="center")
    ax.text(0.085, float(hist.obs_OC.iloc[0] / tot_o), "observed", fontsize=5.8,
            color=OBS, ha="left", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    ax.set_xlabel("pairwise overlap coefficient")
    ax.set_ylabel("fraction of pairs")

    # ---- middle: ABSOLUTE coverage of the antigen as antibodies accumulate ----
    # THE AXIS IS ABSOLUTE COVERAGE, and the reason is the whole point of the
    # panel.
    #
    # Plotting union-after-j divided by union-after-ALL, each curve normalised by
    # its OWN final union, would force BOTH curves to exactly 1.0 at the
    # right-hand edge, so the plot would end where agreement was guaranteed by
    # arithmetic rather than measured, and a statement such as "ten antibodies
    # recover 85.5%" would quote the single weakest point on the curve. It also
    # invites the misreading "random draws of antibodies get a lot of coverage",
    # which is close to the opposite of the finding.
    #
    # Instead: union of the first j epitopes as a fraction of ANTIGEN LENGTH. Nothing
    # forces the two curves together, and they carry a built-in control the
    # normalised version threw away -- AT j = 1 OBSERVED AND NULL ARE EQUAL BY
    # CONSTRUCTION (0.0674 vs 0.0673), because the segment-shift null preserves
    # epitope size, so one epitope covers the same fraction of the antigen
    # wherever it is placed. The curves start together and fan apart to 0.254
    # against 0.444 by j = 10. The separation IS the convergence, measured in
    # units a reader can interpret, and it grows rather than vanishing.
    #
    # Still restricted to the 38 antigens with >= 10 antibodies: the curve is only
    # defined out to j = 10 for an antigen that HAS 10, and using every antigen at
    # every j would swap the cohort at every step along the x axis.
    ax = axes[1]
    ax.plot(sat.j, sat.frac_len_mean, "-", color=OBS, lw=1.3, zorder=3)
    ax.plot(sat.j, sat.frac_len_mean, "o", color=OBS, ms=2.2, zorder=4)
    ax.plot(sat.j, sat.null_frac_len_mean, ls=(0, (2.2, 1.1)), color=NUL, lw=1.3,
            zorder=2)
    # The shared origin is the control, so it is marked rather than left to be
    # noticed: a hairline from the common start point.
    ax.plot([1], [float(sat.frac_len_mean.iloc[0])], "o", mfc="none", mec=INK,
            ms=4.5, mew=0.7, zorder=6)
    j10 = sat[sat.j == 10].iloc[0]
    ax.text(9.7, j10.null_frac_len_mean + 0.03, "null", fontsize=5.8, color=NUL,
            ha="right", va="bottom")
    ax.text(9.7, j10.frac_len_mean - 0.03, "observed", fontsize=5.8, color=OBS,
            ha="right", va="top")
    # The axis runs 0 to 0.5 because the data never exceeds 0.45.
    ax.set_ylim(0, 0.50); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticks(range(2, 11, 2)); ax.set_xlim(0.4, 10.6)
    ax.set_xlabel("antibodies drawn")
    # The label names the plotted quantity: the fraction of antigen residues covered.
    ax.set_ylabel("fraction of antigen covered")

    # ---- right: the answer depends on the null, so all three nulls are shown ----
    ax = axes[2]
    # SHORT tick labels only. "within the / antigenic region" is 2.4x the width of
    # its 45 mm column at 6 pt and ran straight through "cross antigen" next to it;
    # Nature's figure guide asks that text not overlap. Each null is named in full
    # in the legend, which is where a three-line definition belongs anyway.
    tests = [("segment\nshift", "n1"), ("cross-\nantigen", "n2"),
             ("within\nregion", "n3")]
    xs = np.arange(len(tests))
    vals = [H["tests"][k]["frac_antigens_above_null"] for _, k in tests]
    ps = [H["tests"][k]["p_one_sided"] for _, k in tests]
    # The third bar does NOT clear chance. Drawing it in the same slate as the
    # other two would make the panel read as three positive results.
    for xi, v in zip(xs, vals):
        up = v > 0.5
        # Grey and not the null's vermillion, because this bar is a result that failed to
        # clear chance and is not the null itself.
        ax.bar([xi], [v], 0.58, color=OBS if up else GREY_L, alpha=0.75 if up else 1.0,
               edgecolor=OBS if up else GREY_D, linewidth=0.5, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (2.0, 1.5)), zorder=2)
    for x, v, pv in zip(xs, vals, ps):
        lab = "n.s." if pv >= 0.05 else ("$P$ < 0.001" if pv < 1e-3 else f"$P$ = {pv:.3f}")
        ax.text(x, v + 0.03, f"{v:.2f}\n{lab}", ha="center", va="bottom",
                fontsize=5.5, color=INK, linespacing=1.3, zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
    # The 0.5 rule carries no "chance" label on the panel. At this size the word sat
    # on the first bar, and the caption says what the rule is.
    ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(xs, [n for n, _ in tests], fontsize=6.0)
    ax.set_ylabel("fraction of antigens\nabove null", fontsize=6.5)

    # ---- right: WHY most of SAbDab cannot answer this question at all ----
    # The count finding is shown at the end, in its own panel.
    #
    # The three panels to the left are all computed on antigens carrying TWO OR
    # MORE antibodies, because an overlap needs a pair. That set is a minority of
    # the data and the panel should say so rather than leave it to the caption.
    # Two series show the INVERSION between them. The antigens
    # are nearly all in the left-hand bin and the evidence is nearly all in the
    # right-hand one.
    #   974 of 1,627 antigens (59.9%) carry exactly ONE antibody and contribute no
    #       pair at all -- for them the question "do different antibodies bind
    #       different places" has no second antibody to ask it of.
    #   38 antigens (2.3%) carry >= 10 and own 99.2% of all 334,453 pairs.
    # So every per-pair number in the left panel is, to within rounding, a
    # statement about a handful of antigens; the per-antigen views used in panels
    # 2 and 3 give each antigen one vote and are the ones to read.
    ax = axes[3]
    n_ag_all = int(H["counts"]["full"]["antigens"])
    n_single = n_ag_all - int(H["n_groups"])
    bk = [("1", None), ("2", (2, 3)), ("3-4", (3, 5)), ("5-9", (5, 10)),
          ("\u226510", (10, 10 ** 6))]
    tot_pairs = float(per.n_pairs.sum())
    f_ag, f_pr = [], []
    for _lab, b in bk:
        if b is None:
            f_ag.append(100.0 * n_single / n_ag_all); f_pr.append(0.0)
        else:
            sl = per[(per.n_antibodies >= b[0]) & (per.n_antibodies < b[1])]
            f_ag.append(100.0 * len(sl) / n_ag_all)
            f_pr.append(100.0 * float(sl.n_pairs.sum()) / tot_pairs)
    xs = np.arange(len(bk))
    ax.bar(xs - 0.19, f_ag, 0.36, color=GREY_D, edgecolor=GREY_D, lw=0.4, zorder=3)
    ax.bar(xs + 0.19, f_pr, 0.36, color=OBS, alpha=0.75, edgecolor=OBS, lw=0.4,
           zorder=3)
    # Direct labels on the two tallest bars -- the two numbers that carry the
    # panel -- rather than a key, which this figure does not use anywhere.
    # Left-aligned off the first bar rather than centred over it: centred, the word
    # is wider than the bar and its left half hung past the axis spine.
    ax.text(-0.01, f_ag[0] + 4, "antigens", fontsize=5.8, color=GREY_D, ha="left",
            va="bottom")
    ax.text(len(bk) - 1 + 0.19, f_pr[-1] - 5, "pairs", fontsize=5.8, color=OBS,
            ha="center", va="top")
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(xs, [lab for lab, _ in bk], fontsize=6.0)
    ax.set_xlabel("antibodies per antigen")
    ax.set_ylabel("% of total", fontsize=6.5)

    panel_letter(axes[0], "d")
    save(fig, f"fig2d_task1_epitope_redundancy_{arm}", (183, 50))  # Fig 2d


def main():
    ap = argparse.ArgumentParser(
        description="Task I panels 2b and 2c. The placed Figure 2 needs two runs: "
                    "--arm evaluated (2b) and --arm full (2c).")
    ap.add_argument("--arm", choices=["full", "evaluated", "intersection"],
                    default="full",
                    help="intersection: fig2c on the 3,804 cases every method "
                         "scored, which is the shipped 2c, because the panel is "
                         "a cross-method comparison and four cohorts in one "
                         "panel are not comparable. evaluated: fig2b, the "
                         "shipped 2b. full: fig2c on each method's own "
                         "evaluated cases, kept for the per-method view.")
    arm = ap.parse_args().arm
    print(f"Task I panels ({arm} arm)")
    if arm != "intersection":
        fig2b(arm)
    if arm == "evaluated":
        # fig2c reads the per-case tables, which the other arms write.
        print("  fig2c is not drawn for --arm evaluated; "
              "run --arm intersection for it")
    else:
        fig2c(arm)
    for f in sorted(cfg.RESULTS.glob(f"metrics_*_{arm}.json")):
        m = json.loads(f.read_text())
        print(f"  {m['method']:16s} n={m['n']:4d}  AUPRC {m['auprc_mean']:.4f} "
              f"| enrichment {m['enrichment_mean']:.2f}x "
              f"| swap gap {m.get('swap_gap', float('nan')):+.4f}")


if __name__ == "__main__":
    main()
