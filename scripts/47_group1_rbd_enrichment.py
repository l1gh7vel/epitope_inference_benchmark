#!/usr/bin/env python3
"""Antigen group 1's epitopes are concentrated in the RBD.

The Results report an overenrichment of the SARS-CoV-2 spike antigen in the
first cluster, and that the corresponding epitopes in that cluster were enriched
for the receptor binding domain (RBD, residues 319-541). Fig. 2b carries the
antigen overrepresentation; this figure draws the epitope enrichment inside that
antigen.

One row is one antigen group 1 complex from the evaluated set whose antigen is
the 1,273-residue spike and which carries a gold epitope: 844 of the group's 890
complexes. The other 46 are coronavirus antigens of a different length and are
left out rather than aligned, because a position index only means the same thing
within one antigen frame.

Per spike position, the fraction of those 844 antibodies whose gold epitope
contains it. Two summaries follow, each against the share expected if epitope
residues fell uniformly along the sequence, which is the region's own length as
a fraction of 1,273. The RBD is 223 of 1,273 residues, 17.5%; the RBM is 72,
5.7%. The test is a one-sided binomial on epitope residues, one-sided because
the claim has a direction.

Measured: 25,997 of 28,374 epitope residues fall inside the RBD, 91.6% against
17.5% expected, a 5.2-fold enrichment; 55.6% fall inside the RBM against 5.7%
expected, 9.8-fold. Per antibody, 755 of 844 have an epitope lying entirely
inside the RBD and 55 have no RBD residue at all.

    PYTHONPATH=scripts:config EIB_TASK=task1 python scripts/47_group1_rbd_enrichment.py
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "config"))
import figstyle                                                  # noqa: E402
from figstyle import COL2, fit_save                              # noqa: E402
pf = importlib.import_module("12_paper_figures")                 # noqa: E402
import task1 as cfg                                              # noqa: E402
INK, MUTED = pf.INK, pf.MUTED
OUT = PROJECT / "results" / "paper_figures"
RBD, RBM, L = (319, 541), (436, 507), 1273


def main():
    man = pd.read_csv(cfg.MANIFEST)
    z = np.load(cfg.LABELS_NPZ, allow_pickle=True)
    sel = man[(man.usable) & (man.vh_dedup_keep) & (man.af3_eligible)]
    g1 = sel[sel.fold == 1]
    spike = g1[g1.antigen_len == L]
    freq = np.zeros(L)
    rows = []
    for _, r in spike.iterrows():
        if r.case_id not in z:
            continue
        v = np.asarray(z[r.case_id])
        idx = np.flatnonzero(v)
        if not len(idx):
            continue
        freq[idx] += 1
        p1 = idx + 1
        rows.append((len(p1), int(((p1 >= RBD[0]) & (p1 <= RBD[1])).sum()),
                     int(((p1 >= RBM[0]) & (p1 <= RBM[1])).sum())))
    d = pd.DataFrame(rows, columns=["n_epi", "n_rbd", "n_rbm"])
    n = len(d)
    freq /= n
    tot, inr, inm = d.n_epi.sum(), d.n_rbd.sum(), d.n_rbm.sum()
    ed, em = (RBD[1] - RBD[0] + 1) / L, (RBM[1] - RBM[0] + 1) / L
    pd_, pm = (binomtest(inr, tot, ed, alternative="greater").pvalue,
               binomtest(inm, tot, em, alternative="greater").pvalue)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(COL2 * 1.08, 1.9),
        gridspec_kw={"width_ratios": [2.6, 1.0], "wspace": 0.28})

    # (a) where the epitopes are, along the antigen
    x = np.arange(1, L + 1)
    ax.fill_between(x, freq, color=MUTED, alpha=0.5, lw=0)
    ax.plot(x, freq, color=INK, lw=0.6)
    ax.axvspan(*RBD, color="#2C6FAD", alpha=0.13, lw=0, zorder=0)
    ax.axvspan(*RBM, color="#2C6FAD", alpha=0.20, lw=0, zorder=0)
    ax.set_xlim(1, L)
    ax.set_ylim(0, max(0.3, freq.max() * 1.25))
    ax.set_xlabel("SARS-CoV-2 spike residue", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("fraction of the group's\nantibodies in contact", fontsize=6.5)
    ax.annotate("RBD", (np.mean(RBD), freq.max() * 1.18), ha="center",
                fontsize=6, color="#2C6FAD")
    ax.annotate("RBM", (np.mean(RBM), freq.max() * 1.02), ha="center",
                fontsize=5.6, color="#2C6FAD")
    ax.set_title(f"antigen group 1, n = {n} spike complexes", loc="center",
                 fontsize=figstyle.TITLE_PT, pad=3)

    # (b) observed against the share expected from each region's length
    labs = ["RBD\n319-541", "RBM\n436-507"]
    obs = [100 * inr / tot, 100 * inm / tot]
    exp = [100 * ed, 100 * em]
    xx = np.arange(2)
    ax2.bar(xx - 0.19, obs, width=0.36, color="#2C6FAD", alpha=0.85, lw=0,
            label="observed")
    ax2.bar(xx + 0.19, exp, width=0.36, color="#C7C7C2", lw=0,
            label="expected from\nthe region's length")
    for i, (o, e) in enumerate(zip(obs, exp)):
        ax2.text(i, o + 3, f"{o / e:.1f}x", ha="center", va="bottom",
                 fontsize=6.2, color=INK)
    ax2.set_xticks(xx, labs, fontsize=6)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("% of epitope residues", fontsize=6.5)
    ax2.legend(fontsize=5.4, frameon=False, loc="upper right",
               handlelength=1.0, borderpad=0.1, labelspacing=0.25)
    ax2.set_title("epitope residues, pooled", loc="center",
                  fontsize=figstyle.TITLE_PT, pad=3)

    # This panel is Figure 3b. It carries its own letter because the slot has no
    # registered title, so fit_save's caption block never runs.
    figstyle.panel_letter(ax, "a")
    fit_save(fig, OUT, "figR_group1_rbd_enrichment", (183, 50))
    print(f"  n = {n} complexes, {tot} epitope residues")
    print(f"  RBD {inr} = {100 * inr / tot:.1f}% against {100 * ed:.1f}% expected"
          f"  -> {(inr / tot) / ed:.2f}x, one-sided binomial P = {pd_:.3g}")
    print(f"  RBM {inm} = {100 * inm / tot:.1f}% against {100 * em:.1f}% expected"
          f"  -> {(inm / tot) / em:.2f}x, P = {pm:.3g}")
    frac = d.n_rbd / d.n_epi
    print(f"  entirely inside the RBD: {(frac == 1).sum()}/{n}"
          f" | none inside it: {(frac == 0).sum()}")
    pd.DataFrame({"position": np.arange(1, L + 1),
                  "fraction_in_contact": freq}).to_csv(
        PROJECT / "results/task1/group1_epitope_profile.csv", index=False)


if __name__ == "__main__":
    main()
