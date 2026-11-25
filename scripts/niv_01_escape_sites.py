#!/usr/bin/env python3
"""Per-site antibody escape for the Nipah RBP, and the BINARY escape call this arm declares.

Larsen et al. (Cell 188:2480-2494.e22, 2025) publish continuous escape only --
see `data/source/PROVENANCE.md`, "NO PUBLISHED BINARY ESCAPE CALL". The HIV arm
inherits Dingens' FDR site lists verbatim; the RSV arm had to declare a rule
(`rsv_01_escape_sites.py`); so does this one. Everything quoted below was
measured on `data/source/dms_larsen2025/` on 2026-11-01 and regenerates with
`--sweep` and `--structures`.

What this arm can and cannot be
-------------------------------
FOUR antibodies, not six. Only nAH1.3, HENV-26, HENV-32 and m102.4 have a
recoverable (VH, VL); HENV-103 and HENV-117 have no deposited structure and no
published sequence, so they cannot be folded. Their per-site statistics ARE
written here (they cost nothing and are a useful internal consistency check --
HENV-103's hotspots sit on HENV-32's structural epitope, which is what the
competition groups predict), but `foldable` in the JSON is the list of four.

*** n = 4 CANNOT CARRY A SIGNIFICANCE TEST. *** A one-sided Wilcoxon signed-rank
at n = 4 has a minimum attainable P of 1/2^4 = 0.0625, i.e. it cannot reach 0.05
even if every antibody moves the right way by any margin. No test is attached to
this arm anywhere, and none should be added. It is built as a CASE ILLUSTRATION:
four antibodies against one antigen whose escape sets are nearly disjoint
(mean pairwise Jaccard 0.019 at the declared k), shown per antibody. Project rule
7 (one-sided only) still applies to anything that does get a test elsewhere.

The site statistic, and why the floor at zero changes the result
-------------------------------------------------------------
    stat(site) = sum over the <= 19 mutations at that site of max(escape_mean, 0)

`escape_mean` IS SIGNED (range -1.652 to +3.531) and NEGATIVE MEANS THE MUTATION
*INCREASES* NEUTRALISATION -- it is the opposite of escape, not weak escape.
48-64% of all per-mutation values are negative, and for HENV-32 the negative mass
(-1120) is nearly 5x the positive mass (+235). Summing the signed values would
let a site's genuine escape be cancelled by sensitising mutations at the same
position, which answers a different (two-sided) question. The question here has a
direction -- "does mutating this site let the virus escape THIS antibody" -- so
the negative half is floored, not subtracted. In practice the two agree on 8-10
of the top 10 sites per antibody (Spearman 0.73-0.80 over all sites), so this is
a principled choice rather than a load-bearing one; it is stated because it must
be.

A site whose every mutation is NaN stays NaN and is never selected (site 565 for
four antibodies, site 146 for three). It is not floored to 0, because an unmeasured site
is not a no-escape site. Same rule as `dms_01_escape` and `rsv_01_escape_sites`.

SUM, NOT MEAN OR MAX -- AND THIS ONE *IS* LOAD-BEARING
------------------------------------------------------
Against the solved 4.5 A contact sets (below), a per-site MEAN or MAX scores
BETTER than the sum at k = 10, and this docstring says so:

    statistic        mean enrichment over the 4 solved cases, k = 10
    sum, floored          9.3x     <- DECLARED
    mean, floored        12.5x
    max, floored         13.3x
    sum, signed           9.3x

The sum is still what is declared, because the mean's and max's advantage is
bought from sites where almost nothing was measurable, and "how many mutations
were measurable at this site" is not an innocent covariate here. Measured on all
four antibodies: Spearman(number of mutations measured at a site, that site's
mean `entry_CHO_bEFNB3`) = 0.91-0.92. Sites with < 5 measured mutations average
-2.66 on cell entry (near-lethal); sites with >= 15 average -0.21. 46 of the 510
sites carry fewer than 5 measurements.

So a per-site mean is an average over a sample whose SIZE is a near-perfect proxy
for functional constraint, and the extra epitope hits it buys are exactly at
those sites: requiring >= 8 measured mutations collapses the mean back onto the
sum (HENV-26 4/10, HENV-32 3/10, m102.4 4/10 -- the sum's own numbers). Cell
entry in CHO-bEFNB3 is the MATCHED tolerance axis of this arm (all antibody
selections were followed by infection of CHO-bEFNB3 cells) and is the variable
the functional-cost analysis treats as independent. Choosing the mean would wire
that axis into the escape target set and make the cost analysis circular. The sum
is also the authors' own collapse -- their visualisation rule speaks of a "summed
escape score" -- and it is what the RSV arm declared, so the two arms stay
commensurable.

The declared rule
-----------------
    *** A site is an ESCAPE SITE for an antibody if its floored-sum statistic is
        among that antibody's TOP 10 of the 510 scanned sites (top 1.96%).
        Fixed rank, per antibody, ties broken by ascending site number. ***

`--topk` changes it; 10 is the declared default and the only value a headline
number may use without saying otherwise.

WHY FIXED RANK -- every value-keyed rule deletes an antibody
------------------------------------------------------------
`--sweep` prints all of this.

  * The HIV arm's gamma-tail / BH-FDR rule (`dms_10_functional_sites.find_sig_sel`)
    gives **nAH1.3 ZERO sites at FDR 0.01, 0.05 and 0.10** and three at 0.20 --
    and nAH1.3 is by a wide margin the SHARPEST antibody in the panel (max/median
    86x, 8/10 of its top 10 inside its own solved epitope, 16.3x enrichment). The
    fit absorbs its signal into the bulk. The gamma fit is formally REJECTED by a
    KS test for all six antibodies (p from 2e-4 to 7e-36), so this is not a
    borderline call. At the other end it gives m102.4 29 sites at FDR 0.01 -- a
    10x spread in set size across antibodies.
  * An absolute cut: stat >= 3 gives HENV-32 7 sites and m102.4 27; stat >= 5
    gives HENV-32 2 and nAH1.3 21; stat >= 8 gives HENV-26 and HENV-117 zero.
  * Robust z (median + 1.4826*MAD): z >= 8 gives HENV-32 8 and nAH1.3 24.
  * The authors' own VISUALISATION rule (50% of max per-mutation escape, OR 75%
    of max summed escape) gives set sizes 3, 4, 5, 7, 8, 22 -- a 7x spread, and
    its own methods scope it to figure-making, so it is not an escape call and is
    not used as one.

Every one of these makes the per-antibody base rate vary by 4-10x, which would
make per-antibody AUPRC and its random floor incomparable across antibodies --
the same category error that is forbidden for numbers
taken on different cohorts. Fixed rank pins the base rate at 10/510 = 1.96% for all four.

WHY k = 10 -- AND WHAT THE EVIDENCE DOES *NOT* SHOW
---------------------------------------------------
Mean enrichment of the selected set inside the solved 4.5 A contact set, over the
four antibodies that have one:

    k         5     8    10    12    15    20    25    30    40    50
    mean   13.3  10.4   9.3   8.2   8.3   8.7   7.7   7.3   6.0   5.3

*** There is no elbow at 10. *** The curve falls to ~8-9x by k = 12 and then
PLATEAUS to k = 25 before resuming its decline. Stated plainly: the structural
evidence picks a REGIME (k roughly 5-25 is all "concentrated"), it does not pick
a point, and anyone who reads an elbow into this curve is reading noise at n = 4.
k = 10 is therefore chosen on three grounds that are declared, not fitted:

  1. COMMENSURABILITY. The RSV arm declares top 10 of 504 sites (1.98%); 10 of
     510 here is 1.96%. Identical base rate, so the two DMS arms' AUPRCs and
     random floors can sit on one axis. The HIV arm's inherited Dingens sets
     carry 4-15 sites, so k = 10 keeps all three arms in one range.
  2. NEAR-DISJOINTNESS, which is the whole point of the arm. Over the four
     foldable antibodies: k = 10 gives union 38/510 and mean pairwise Jaccard
     0.019; k = 20 gives union 69 and Jaccard 0.053; k = 25 gives union 83 and
     Jaccard 0.068. A target set that is the same for every antibody cannot test
     antibody conditioning at all (that is the defect of the HIV CONSTRAINT arm,
     `dms_10_functional_sites.py`); the smaller k, the sharper the test.
  3. The value at rank 10 is 9.4-33.1x each antibody's own median, against
     6.1-9.6x at rank 25 -- rank 10 is still in the tail for every antibody.

PER-ANTIBODY, AT THE DECLARED k = 10, AGAINST THE SOLVED COMPLEX
-----------------------------------------------------------------
    antibody   PDB    antigen   |4.5A epitope|   in epitope   enrichment
    nAH1.3     7TXZ   Nipah          25            8/10         16.3x
    HENV-26    6VY5   Nipah          31            4/10          8.2x
    m102.4     6CMG   **Hendra**     29            3/10          7.7x
    HENV-32    6VY4   **Hendra**     30            3/10          5.1x

Two caveats that must travel with those numbers. (a) m102.4's and HENV-32's
contact sets come from HENDRA G complexes and are mapped onto Nipah numbering by
global alignment at 75-77% identity, so they are cross-species references, not
Nipah epitopes. (b) 6CMG is m102.**3**, not m102.4 (see the manifest's
`sequence_is_proxy`). Neither case is a null the way RSV's 1A2 is -- 3/10 at
5-8x is real enrichment, 5-8 standard deviations above the hypergeometric mean --
but HENV-32 is the weakest case in the panel and should be labelled as such.

Sanity check that the Hendra->Nipah mapping is right rather than plausible: the
m102.4 and HENV-26 contact sets both come out as 501/504-507/528-533/554-559/
579-588, which is the ephrin receptor-binding site of the Nipah RBP. Both are
published receptor-blocking antibodies. A frame error of even one residue would
not land there.

WHY THE `region` COLUMN IS NOT USED AS EVIDENCE
-----------------------------------------------
`site_numbering_map.csv` annotates Head / Stalk / Neck / Linker, and the RSV arm
leaned on the equivalent column. It cannot do any work here. Background shares
over the 510 scanned sites are Head 0.798, Stalk 0.147, Neck 0.031, Linker 0.024,
so the maximum attainable enrichment for a Head-binding antibody is 1.25x. All
four solved epitopes are in the Head. Measured at k = 10 the Head fraction is
8-10/10 for every antibody including the two with no structure -- i.e. 1.0-1.25x,
indistinguishable. The sweep is printed by `--sweep` for the record and is
reported as UNINFORMATIVE, not as support.

Frame, and the stop codon in the FASTA
------------------------------------------------
*** `protein.fasta` is 533 CHARACTERS BUT THE PROTEIN IS 532 RESIDUES: the last
character is `*`, a stop codon. *** `site_numbering_map.csv` carries the matching
row (sequential 533 / reference 603, `sequential_wt` = `*`, `reference_wt` = `-`),
so a naive "the map and the fasta agree 533/533" check PASSES on it and a naive
`len(fasta)` gives 533. Feeding that string to AF3 fails outright --
`ValueError: Protein must contain only letters` -- which is how this was caught
(smp job 24496519, 2026-11-01), and it would have been far worse if it had
silently shifted a mask by one. The sequence is stripped to the 532 real residues
everywhere in this arm, `FOLD_LEN` is 532, and `load()` asserts it.

So: the folded antigen is the 532 aa ectodomain, sequential 1-532 == reference
71-602 (offset exactly 70 at every row). 510 of those 532 positions were
scanned; 22 sites inside 71-602 are (107, 121, 158, 162, 216, 240, 253, 382, 387, 395, 487, 489, 499, 500,
503, 504, 506, 533, 557, 574, 579, 581). The npz carries BOTH frames -- 510-long
vectors over the scanned sites, and 532-long masks over the fold frame with a
`scanned_mask` saying which positions were measured at all -- so no consumer has
to reconstruct the offset and nobody silently scores an unscanned position as a
negative.

    PYTHONPATH=scripts python scripts/niv_01_escape_sites.py
    PYTHONPATH=scripts python scripts/niv_01_escape_sites.py --sweep
    PYTHONPATH=scripts python scripts/niv_01_escape_sites.py --structures
"""
import argparse, itertools, json, os, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
SRC = PROJECT / "data" / "source" / "dms_larsen2025"
REF_DIR = PROJECT / "work" / "niv" / "ref"
OUT = PROJECT / "data" / "manifests"

# All six measured antibodies. `FOLDABLE` is the subset with a recoverable
# (VH, VL); the other two have no structure and no published sequence, so they
# can never enter the fold manifest. They are kept here because their statistics
# are free and act as an internal check (see docstring).
ANTIBODIES = ["nAH1.3", "HENV-26", "HENV-32", "m102.4", "HENV-103", "HENV-117"]
FOLDABLE = ["nAH1.3", "HENV-26", "HENV-32", "m102.4"]
NOT_FOLDABLE = {
    "HENV-103": "no PDB entry and no published sequence (EMD-23488 is 15 A with "
                "no fitted model; its Fabs are a 12E8 placeholder and a predicted Fv)",
    "HENV-117": "no PDB entry and no published sequence",
}

TOPK = 10                 # *** the declared rule: top 10 of 510 sites = top 1.96% ***
SITE_OFFSET = 70          # fasta index i (1-based) == reference site i + 70
FOLD_LEN = 532            # the REAL protein; protein.fasta has 533 chars, last is `*`
CONTACT_CUT = 4.5         # project-wide, every task

# Solved complexes. Chain roles were read off `_entity_poly` (see
# work/niv/ref/README.md), not guessed from the letters. `species` is the antigen
# in THAT structure -- two of the four are Hendra G and are mapped onto Nipah
# numbering by global alignment at 75-77% identity.
COMPLEXES = {
    "nAH1.3":  dict(cif="7txz.cif", ab=set("EFHL"), ag=set("ABCD"), species="Nipah"),
    "HENV-26": dict(cif="6vy5.cif", ab=set("HL"),   ag=set("A"),    species="Nipah"),
    "HENV-32": dict(cif="6vy4.cif", ab=set("CDHL"), ag=set("AB"),   species="Hendra"),
    "m102.4":  dict(cif="6cmg.cif", ab=set("BC"),   ag=set("A"),    species="Hendra"),
}


# ------------------------------------------------------------------ load ----
def antigen_sequence():
    """The 532 aa RBP ectodomain, with the trailing stop codon removed.

    `protein.fasta` ends in `*`. AF3 rejects it outright and a mask built on
    len(fasta) would be one position too long, so stripping happens HERE and
    every consumer (including niv_02_build_manifest) imports this rather than
    reading the fasta itself.
    """
    raw = "".join(l.strip() for l in (SRC / "protein.fasta").read_text().splitlines()
                  if not l.startswith(">"))
    seq = raw.rstrip("*")
    assert seq.isalpha() and len(seq) == FOLD_LEN, \
        f"antigen is {len(seq)} aa after stripping stops, expected {FOLD_LEN}"
    return seq


def load():
    e = pd.read_csv(SRC / "RBP_mutation_effects_antibody_escape.csv")
    m = pd.read_csv(SRC / "site_numbering_map.csv")
    fasta = antigen_sequence()
    real = m[m.sequential_wt.str.isalpha()]
    assert all(fasta[r.sequential_site - 1] == r.sequential_wt for r in real.itertuples()), \
        "protein.fasta is not in the sequential frame of site_numbering_map.csv"
    assert len(real) == FOLD_LEN, f"{len(real)} real rows in the numbering map, expected {FOLD_LEN}"
    assert (m.reference_site - m.sequential_site == SITE_OFFSET).all(), \
        f"reference/sequential offset is not {SITE_OFFSET}"
    assert set(e.antibody) == set(ANTIBODIES), f"unexpected antibodies: {set(e.antibody)}"
    sites = np.sort(e.site.unique())
    stat = pd.DataFrame({ab: site_statistic(e, ab) for ab in ANTIBODIES}).reindex(sites)
    region = m.set_index("reference_site")["region"].reindex(sites).fillna("none")
    return e, stat, region, fasta


def site_statistic(e, antibody):
    """Floored sum: sum over the mutations at a site of max(escape_mean, 0).

    `min_count=1` keeps a site NaN when every mutation there is NaN rather than
    scoring it 0 -- an unmeasured site is not a no-escape site.
    """
    d = e[e.antibody == antibody]
    return d.assign(_v=d.escape_mean.clip(lower=0)).groupby("site")["_v"].sum(min_count=1)


def escape_mask(stat, k=TOPK):
    """Top-k sites by the statistic. Ties broken by ascending site number."""
    order = stat.sort_values(ascending=False, kind="mergesort").index[:k]
    return stat.index.isin(order)


# ------------------------------------------- solved structural epitopes ----
def _one_letter(res):
    info = __import__("gemmi").find_tabulated_residue(res.name)
    if info is None or not info.is_amino_acid():
        return None
    return info.one_letter_code.upper()


def _chain_atoms(st, names):
    """Heavy-atom coordinates + (chain, seqid) keys + per-chain (seqids, sequence)."""
    import gemmi
    P, K, per = [], [], {}
    for ch in st[0]:
        if ch.name not in names:
            continue
        nums, letters = [], []
        for r in ch:
            ol = _one_letter(r)
            if ol is None:
                continue
            nums.append(r.seqid.num); letters.append(ol)
            for a in r:
                if a.element == gemmi.Element("H"):
                    continue
                P.append([a.pos.x, a.pos.y, a.pos.z]); K.append((ch.name, r.seqid.num))
        per[ch.name] = (nums, "".join(letters))
    return np.array(P), K, per


def _map_to_nipah(nums, letters, fasta):
    """{depositor seqid -> Nipah reference site} by global alignment to protein.fasta.

    Needed because 6VY4 and 6CMG are HENDRA G: the two glycoproteins differ in
    length and the depositor numbering is not interchangeable. Also run on the
    Nipah structures, where it must come out as the identity -- that is the
    check, not an assumption.
    """
    import gemmi
    res = gemmi.align_string_sequences(list(letters), list(fasta), [],
                                       gemmi.AlignmentScoring())
    out, qi, ti = {}, 0, 0
    for n, op in re.findall(r"(\d+)([MID])", res.cigar_str()):
        n = int(n)
        if op == "M":
            for _ in range(n):
                out[nums[qi]] = ti + 1 + SITE_OFFSET
                qi += 1; ti += 1
        elif op == "I":          # residue present in the structure, absent from Nipah
            qi += n
        else:                    # gap in the structure
            ti += n
    return out, res.calculate_identity(1)


def structural_epitopes(fasta, verbose=False):
    """4.5 A heavy-atom antigen contacts per antibody, in Nipah reference numbering."""
    import gemmi
    from scipy.spatial import cKDTree
    out = {}
    for ab, spec in COMPLEXES.items():
        p = REF_DIR / spec["cif"]
        if not p.exists():
            print(f"  {ab}: {p} missing -- see work/niv/ref/README.md"); continue
        st = gemmi.read_structure(str(p)); st.setup_entities(); st.remove_ligands_and_waters()
        A, _, _ = _chain_atoms(st, spec["ab"])
        G, GK, per = _chain_atoms(st, spec["ag"])
        maps = {}
        for cn, (nums, letters) in per.items():
            maps[cn], ident = _map_to_nipah(nums, letters, fasta)
            if verbose:
                print(f"  {ab:9s} {spec['cif']} antigen chain {cn}: {len(nums)} res, "
                      f"{ident:.1f}% identity to Nipah ({spec['species']})")
        idx = cKDTree(A).query_ball_point(G, CONTACT_CUT)
        hits = {maps[GK[i][0]][GK[i][1]] for i, v in enumerate(idx)
                if v and GK[i][1] in maps[GK[i][0]]}
        out[ab] = sorted(hits)
    return out


# ------------------------------------- the HIV arm's rule, as a DIAGNOSTIC ----
def _bh(p, fdr):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.clip(np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    out = np.empty(n); out[o] = q
    return out, out <= fdr


def find_sig_sel(v, fdr=0.01):
    """dms_tools2 `robust_hist`, as in dms_10_functional_sites. Diagnostic only."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    try:
        edges = np.histogram(v, bins="fd")[1]
    except ValueError:
        edges = np.histogram(v, bins="doane")[1]
    heights, edges = np.histogram(v, bins=edges, density=True)
    bins = (edges[:-1] + edges[1:]) / 2
    scale = v.var(ddof=1) / v.mean()
    fit = optimize.least_squares(
        lambda x: stats.gamma.pdf(bins, x[0], scale=x[1], loc=x[2]) - heights,
        np.array([v.mean() / scale, scale, 0.0]), loss="soft_l1")
    P = stats.gamma(fit.x[0], scale=fit.x[1], loc=fit.x[2]).sf(v)
    Q, sig = _bh(P, fdr)
    return sig, fit.x


# ----------------------------------------------------------------- sweeps ----
def sweep(e, stat, region, fasta):
    n = len(stat)
    print(f"=== the site statistic over {n} scanned sites ===")
    for ab in ANTIBODIES:
        x = stat[ab].values
        print(f"  {ab:10s} n={np.isfinite(x).sum():3d} mean={np.nanmean(x):6.2f} "
              f"median={np.nanmedian(x):5.2f} sd={np.nanstd(x):5.2f} "
              f"skew={stats.skew(x, nan_policy='omit'):5.2f} "
              f"kurt={stats.kurtosis(x, nan_policy='omit'):7.2f} "
              f"max={np.nanmax(x):6.2f} max/median={np.nanmax(x)/np.nanmedian(x):6.1f}")

    print("\n=== the SIGN of escape_mean: why the floor at zero matters ===")
    print("  negative = the mutation INCREASES neutralisation, i.e. the opposite of escape")
    for ab in ANTIBODIES:
        d = e[e.antibody == ab].escape_mean
        print(f"  {ab:10s} n_mut={len(d)} frac<0={float((d < 0).mean()):.3f} "
              f"range [{d.min():6.3f}, {d.max():5.3f}] "
              f"sum_neg={d[d < 0].sum():8.1f} sum_pos={d[d > 0].sum():7.1f}")
    signed = pd.DataFrame({ab: e[e.antibody == ab].groupby("site")["escape_mean"].sum(min_count=1)
                           for ab in ANTIBODIES}).reindex(stat.index)
    print("  floored sum vs signed sum, top-10 overlap / Spearman over all sites:")
    for ab in ANTIBODIES:
        ov = len(set(stat[ab].nlargest(TOPK).index) & set(signed[ab].nlargest(TOPK).index))
        rho = stats.spearmanr(stat[ab], signed[ab], nan_policy="omit").statistic
        print(f"    {ab:10s} {ov}/{TOPK}   rho={rho:.3f}")

    print("\n=== SUM vs MEAN vs MAX, and why the sum is declared despite scoring worse ===")
    ep = structural_epitopes(fasta)
    variants = {
        "sum, floored":  lambda d: d.assign(_v=d.escape_mean.clip(lower=0)).groupby("site")["_v"].sum(min_count=1),
        "mean, floored": lambda d: d.assign(_v=d.escape_mean.clip(lower=0)).groupby("site")["_v"].mean(),
        "max, floored":  lambda d: d.assign(_v=d.escape_mean.clip(lower=0)).groupby("site")["_v"].max(),
        "sum, signed":   lambda d: d.groupby("site")["escape_mean"].sum(min_count=1),
    }
    for name, fn in variants.items():
        cells, tot = [], 0.0
        for ab, E in ep.items():
            s = fn(e[e.antibody == ab]).reindex(stat.index)
            E = set(E) & set(int(i) for i in stat.index)
            ov = len(set(s.nlargest(TOPK).index) & E); exp = TOPK * len(E) / n
            tot += ov / exp; cells.append(f"{ab}:{ov}/{TOPK}({ov/exp:4.1f}x)")
        mark = "   <- DECLARED" if name == "sum, floored" else ""
        print(f"  {name:14s} mean {tot/len(ep):5.2f}x   " + "  ".join(cells) + mark)

    f = pd.read_csv(SRC / "RBP_mutation_effects_cell_entry_CHO-bEFNB3.csv")
    ent = f.groupby("site")["entry_CHO_bEFNB3"].mean()
    print("  the mean/max advantage is bought at low-coverage sites, and coverage IS tolerance:")
    for ab in FOLDABLE:
        d = e[e.antibody == ab]
        cnt = d.groupby("site")["escape_mean"].apply(lambda s: s.notna().sum())
        j = pd.concat([cnt.rename("n"), ent.rename("ent")], axis=1).dropna()
        print(f"    {ab:10s} Spearman(n measured, mean entry_CHO_bEFNB3)="
              f"{stats.spearmanr(j.n, j.ent).statistic:.3f}  "
              f"mean entry at n<5: {j[j.n < 5].ent.mean():6.2f}  at n>=15: {j[j.n >= 15].ent.mean():6.2f}")
    print(f"    sites with <5 mutations measured: {int((e[e.antibody == FOLDABLE[0]].groupby('site').escape_mean.apply(lambda s: s.notna().sum()) < 5).sum())} of {n}")
    print("  requiring >=8 measured mutations collapses the MEAN rule back onto the SUM:")
    for mincov in (0, 3, 5, 8):
        cells = []
        for ab, E in ep.items():
            d = e[e.antibody == ab]
            cnt = d.groupby("site")["escape_mean"].apply(lambda s: s.notna().sum()).reindex(stat.index)
            s = variants["mean, floored"](d).reindex(stat.index).where(cnt >= mincov)
            E = set(E) & set(int(i) for i in stat.index)
            ov = len(set(s.nlargest(TOPK).index) & E)
            cells.append(f"{ab}:{ov}/{TOPK}")
        print(f"    mean, n_mut>={mincov}:  " + "  ".join(cells))

    print("\n=== the HIV arm's gamma-tail / BH-FDR rule, applied here (DIAGNOSTIC ONLY) ===")
    for ab in ANTIBODIES:
        v = stat[ab].values
        counts = []
        for fdr in (0.01, 0.05, 0.10, 0.20):
            sig, par = find_sig_sel(v, fdr); counts.append(int(sig.sum()))
        ks = stats.kstest(v[np.isfinite(v)], "gamma", args=(par[0], par[2], par[1]))
        flag = "  gamma fit REJECTED (KS p<0.05)" if ks.pvalue < 0.05 else ""
        print(f"  {ab:10s} FDR.01={counts[0]:3d} .05={counts[1]:3d} .10={counts[2]:3d} "
              f".20={counts[3]:3d}   KS p={ks.pvalue:.3g}{flag}")
    print("  nAH1.3 gets ZERO sites up to FDR 0.10 and is the SHARPEST antibody here.")

    print("\n=== SWEEP B: absolute cut on the statistic (set size per antibody) ===")
    for cut in (1, 2, 3, 5, 8, 12, 20):
        print(f"  stat>={cut:5.1f}  " +
              "  ".join(f"{ab}:{int((stat[ab] >= cut).sum())}" for ab in ANTIBODIES))

    print("\n=== SWEEP C: robust z = (stat - median)/(1.4826*MAD) ===")
    for z in (3, 4, 5, 6, 8, 10):
        cells = []
        for ab in ANTIBODIES:
            v = stat[ab].values; med = np.nanmedian(v)
            mad = np.nanmedian(np.abs(v - med)) * 1.4826
            cells.append(f"{ab}:{int(np.nansum((v - med) / mad >= z))}")
        print(f"  z>={z:3d}  " + "  ".join(cells))

    print("\n=== SWEEP D: the authors' own VISUALISATION rule (NOT an escape call) ===")
    print("  'max escape within 50% of the antibody max' OR 'summed escape within 75% of max'")
    for ab in ANTIBODIES:
        d = e[e.antibody == ab]
        mx = d.groupby("site")["escape_mean"].max().reindex(stat.index)
        a = (mx >= 0.5 * mx.max()); b = (stat[ab] >= 0.75 * stat[ab].max())
        print(f"  {ab:10s} 50%-of-max-mutation: {int(a.sum()):3d}   "
              f"75%-of-max-summed: {int(b.sum()):3d}   union: {int((a | b).sum()):3d}")

    print("\n=== SWEEP E: fixed rank, the DECLARED family ===")
    print("  (A) structural: mean enrichment inside the solved 4.5 A contact set, 4 antibodies")
    for k in (5, 8, 10, 12, 15, 20, 25, 30, 40, 50):
        vals = []
        for ab, E in ep.items():
            E = set(E) & set(int(i) for i in stat.index)
            ov = len(set(stat[ab].nlargest(k).index) & E)
            vals.append(ov / (k * len(E) / n))
        mark = "   <- DECLARED" if k == TOPK else ""
        print(f"    k={k:3d}  mean={np.mean(vals):5.2f}x  " +
              "  ".join(f"{ab}:{len(set(stat[ab].nlargest(k).index) & (set(E_) & set(int(i) for i in stat.index)))}/{k}"
                        for ab, E_ in ep.items()) + mark)
    print("    NO ELBOW: ~8-9x from k=12 to k=25. This picks a regime, not a point.")

    print("  (B) region concentration -- UNINFORMATIVE here, printed for the record")
    share = region.value_counts() / n
    print(f"    background share over {n} sites: " +
          "  ".join(f"{r} {share[r]:.3f}" for r in share.index))
    print(f"    every solved epitope is in the Head, whose share is {share.get('Head', 0):.3f}, "
          f"so max attainable enrichment is {1/share.get('Head', 1):.2f}x")
    for k in (5, 10, 15, 20, 25, 40):
        print(f"    k={k:3d}  " + "  ".join(
            f"{ab}:Head {int((region.loc[stat[ab].nlargest(k).index] == 'Head').sum())}/{k}"
            for ab in ANTIBODIES))

    print("  (C) disjointness of the selected sets (the point of the arm)")
    for k in (5, 10, 15, 20, 25):
        s6 = {ab: set(stat[ab].nlargest(k).index) for ab in ANTIBODIES}
        s4 = {ab: s6[ab] for ab in FOLDABLE}
        j6 = [len(s6[a] & s6[b]) / len(s6[a] | s6[b]) for a, b in itertools.combinations(ANTIBODIES, 2)]
        j4 = [len(s4[a] & s4[b]) / len(s4[a] | s4[b]) for a, b in itertools.combinations(FOLDABLE, 2)]
        mark = "   <- DECLARED" if k == TOPK else ""
        print(f"    k={k:3d}  foldable 4: union={len(set().union(*s4.values())):3d}/{n} "
              f"meanJaccard={np.mean(j4):.3f}   all 6: union={len(set().union(*s6.values())):3d} "
              f"meanJaccard={np.mean(j6):.3f}{mark}")

    print("  (D) value at rank k, relative to that antibody's own median")
    for k in (5, 10, 15, 20, 25):
        print(f"    k={k:3d}  " + "  ".join(
            f"{ab}:{stat[ab].nlargest(k).min()/stat[ab].median():5.1f}x" for ab in ANTIBODIES))


def validate_structures(stat, fasta):
    ep = structural_epitopes(fasta, verbose=True)
    n = len(stat)
    print(f"\n=== solved {CONTACT_CUT} A epitopes, in Nipah reference numbering ===")
    for ab, E in ep.items():
        print(f"  {ab:9s} ({COMPLEXES[ab]['species']:6s}, {COMPLEXES[ab]['cif']}) "
              f"|epitope|={len(E):3d}  {E}")
    print(f"\n=== declared sites vs those epitopes ===")
    for k in (5, 8, 10, 12, 15, 20, 25, 50):
        print(f"  --- k={k}")
        for ab, E in ep.items():
            E = set(E) & set(int(s) for s in stat.index)
            sel = set(stat[ab].nlargest(k).index)
            exp = k * len(E) / n
            print(f"      {ab:9s} |epitope|={len(E):3d}  in epitope {len(sel & E):2d}/{k}  "
                  f"expected {exp:4.1f}  enrich {len(sel & E)/exp:5.2f}x")
    print("\nHendra-derived sets (HENV-32, m102.4) are cross-species references mapped by "
          "alignment;\nthe m102.4/HENV-26 sets landing on the ephrin receptor-binding site "
          "(501, 504-507,\n528-533, 554-559, 579-588) is the check that the frame is right.")


# ------------------------------------------------- gold labels, as masks ----
def write_labels(fasta, stat):
    """Write the solved 4.5 A contact sets as FOLD-FRAME boolean masks.

    `structural_epitopes` already computes these -- it is what the json's
    `epitope_size` and `structural_enrichment` are built from -- but it only ever
    reported COUNTS, so the per-residue gold vector this arm needs for a recall
    panel did not exist on disk. The RSV arm has the same thing as
    `rsv_labels.npz`; this writes the Nipah twin in the same shape, same key
    spelling (`{antibody}__gold`), so one reader serves both.

    FRAME. The masks are FOLD-frame, length 532, 0-based index = reference site
    - 71, i.e. the frame every score vector in work/niv is in. Reference sites
    outside 71-602 cannot be represented and are counted out loud rather than
    dropped silently -- there are none today, and if a structure is ever swapped
    for one with a longer construct the count is the thing that catches it.

    NOT intersected with the scanned set. The gold epitope is a property of the
    structure, not of the DMS; a consumer that needs it restricted to assayed
    positions does that intersection itself (dms_08 does).
    """
    ep = structural_epitopes(fasta, verbose=True)
    sites = stat.index.values.astype(int)
    scanned = np.zeros(FOLD_LEN, bool)
    scanned[sites - SITE_OFFSET - 1] = True
    arrays = {"site_numbers": sites, "scanned_mask": scanned}
    print(f"\n  {'antibody':10s} {'pdb':6s} {'n_contact':>9s} {'in frame':>8s} "
          f"{'scanned':>7s}")
    for ab, hits in ep.items():
        idx = np.array([h - SITE_OFFSET - 1 for h in hits], int)
        keep = (idx >= 0) & (idx < FOLD_LEN)
        m = np.zeros(FOLD_LEN, bool); m[idx[keep]] = True
        arrays[f"{ab}__gold"] = m
        print(f"  {ab:10s} {COMPLEXES[ab]['cif'][:4].upper():6s} {len(hits):9d} "
              f"{int(keep.sum()):8d} {int((m & scanned).sum()):7d}")
        if (~keep).any():
            print(f"    !! {int((~keep).sum())} contacted sites fall outside the "
                  f"{FOLD_LEN}-residue fold frame and are NOT in the mask")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "niv_labels.npz", **arrays)
    print(f"\nwrote {OUT/'niv_labels.npz'} ({len(ep)} of {len(ANTIBODIES)} "
          f"antibodies have a solved complex)")


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topk", type=int, default=TOPK,
                    help=f"sites per antibody (DECLARED DEFAULT {TOPK} = top 1.96%% of 510)")
    ap.add_argument("--sweep", action="store_true",
                    help="print the full threshold sensitivity sweep and diagnostics")
    ap.add_argument("--structures", action="store_true",
                    help="print the solved-complex epitopes and the overlap table")
    ap.add_argument("--labels", action="store_true",
                    help="write niv_labels.npz (the solved 4.5 A contact sets as "
                         "fold-frame masks) and nothing else")
    args = ap.parse_args()

    e, stat, region, fasta = load()
    if args.sweep:
        sweep(e, stat, region, fasta); return
    if args.structures:
        validate_structures(stat, fasta); return
    if args.labels:
        write_labels(fasta, stat); return

    ep = structural_epitopes(fasta)
    n = len(stat)
    share = region.value_counts() / n
    sites = stat.index.values.astype(int)
    scanned = np.zeros(FOLD_LEN, bool)
    scanned[sites - SITE_OFFSET - 1] = True

    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {"site_numbers": sites,
              "fold_index": sites - SITE_OFFSET,      # 1-based index into protein.fasta
              "scanned_mask": scanned}
    meta, rows = {}, []
    for ab in ANTIBODIES:
        m = escape_mask(stat[ab], args.topk)
        sel = [int(s) for s in stat.index[m]]
        fold_mask = np.zeros(FOLD_LEN, bool)
        fold_mask[np.array(sel) - SITE_OFFSET - 1] = True
        arrays[f"{ab}|escape"] = m
        arrays[f"{ab}|stat"] = stat[ab].values
        arrays[f"{ab}|escape_fold"] = fold_mask

        E = set(ep.get(ab, ())) & set(int(s) for s in stat.index)
        in_ep = len(set(sel) & E) if E else None
        enr = (in_ep / (args.topk * len(E) / n)) if E else None
        head = int((region.loc[sel] == "Head").sum())
        meta[ab] = dict(
            rule=f"top {args.topk} of {n} scanned sites ({100*args.topk/n:.2f}%) "
                 f"of the floored-sum statistic",
            foldable=ab in FOLDABLE,
            not_foldable_reason=NOT_FOLDABLE.get(ab),
            n_sites=int(m.sum()), threshold=float(stat[ab][m].min()),
            median_stat=float(stat[ab].median()), sites=sel,
            n_head=head, head_background=float(share.get("Head", np.nan)),
            structure=COMPLEXES[ab]["cif"].upper()[:4] if ab in COMPLEXES else None,
            structure_species=COMPLEXES[ab]["species"] if ab in COMPLEXES else None,
            epitope_size=len(E) if E else None,
            n_in_structural_epitope=in_ep,
            structural_enrichment=round(enr, 2) if enr else None)
        rows.append(dict(antibody=ab, foldable=ab in FOLDABLE, k=int(m.sum()),
                         cut=round(float(stat[ab][m].min()), 2),
                         median=round(float(stat[ab].median()), 2),
                         head=f"{head}/{args.topk}",
                         epitope=("-" if not E else f"{in_ep}/{args.topk}"),
                         enrich=("-" if not enr else f"{enr:.1f}x"),
                         pdb=meta[ab]["structure"] or "-",
                         antigen=meta[ab]["structure_species"] or "-"))

    np.savez_compressed(OUT / "niv_escape_sites.npz", **arrays)
    (OUT / "niv_escape_sites.json").write_text(json.dumps(dict(
        source="Larsen et al., Cell 188:2480-2494.e22 (2025); "
               "RBP_mutation_effects_antibody_escape.csv",
        statistic="sum over the mutations at a site of max(escape_mean, 0); "
                  "escape_mean is SIGNED and negative means increased neutralisation",
        rule=f"top {args.topk} of {n} scanned sites, fixed rank, per antibody",
        topk=args.topk, n_scanned_sites=int(n), fold_len=FOLD_LEN,
        frame=f"fold frame is the {FOLD_LEN} aa ectodomain = protein.fasta with its "
              f"trailing '*' stop codon stripped; index i (1-based) == RBP reference "
              f"site i + {SITE_OFFSET}; scanned reference sites 71-602 with 22 gaps",
        no_significance_test="n=4 foldable antibodies; a one-sided Wilcoxon at n=4 "
                             "has minimum attainable P = 0.0625. This arm is a case "
                             "illustration and carries no test.",
        antibodies=ANTIBODIES, foldable=FOLDABLE, not_foldable=NOT_FOLDABLE,
        contact_cutoff=CONTACT_CUT, per_antibody=meta), indent=2))

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {OUT/'niv_escape_sites.npz'} and {OUT/'niv_escape_sites.json'}")
    print("NOTE: n=4 foldable antibodies. No significance test is attached to this arm.")


if __name__ == "__main__":
    main()
