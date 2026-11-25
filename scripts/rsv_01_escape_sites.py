#!/usr/bin/env python3
"""Per-site Fab escape for RSV F, and the binary escape call this arm declares.

Simonich et al. 2026 publish no binary escape call. The HIV arm inherits the
authors' FDR site lists verbatim; here there is nothing to inherit, so the rule
below is ours and is declared rather than fitted. Everything in this docstring was measured on
`data/source/dms_simonich2026/all_antibodies.csv` (9,824 rows, 504 sites, 26-529
in full-length F numbering, no gaps) and regenerates with `--sweep`.

The site statistic (the authors' own)
----------------------------------------------
`summaries_config.yml` sets `init_site_escape_stat: sum` and
`init_floor_escape_at_zero: true`, so a site's statistic is

    stat(site) = sum over the <=20 mutations at that site of max(escape, 0)

computed on the **Fab** columns (the IgG columns are not read).
The per-mutation values in `all_antibodies.csv` are already the authors'
`stat: escape_median`, already filtered by `min_times_seen: 2`,
`min_frac_models: 0.67` and the per-selection `escape_std` ceiling (1.0-2.6).
No filter of ours is applied on top; only the threshold below is ours.
RSM01 is excluded for want of a public antibody sequence, leaving six
antibodies: Nirsevimab, Clesrovimab, Palivizumab, Suptavumab, 1A2, 1B6. Missing
measurements are NaN and are skipped rather than floored to 0; in practice all 504
sites carry a value for all six antibodies, so no site is dropped.

Why the HIV arm's gamma-tail / FDR rule does not transfer
-------------------------------------------
`dms_10_functional_sites.find_sig_sel` (dms_tools2 `robust_hist`) fits a gamma to
the site statistic and calls the upper tail at BH-FDR. Its assumption is that
almost every site is null and the null piles up near zero. That assumption fails
here, for two reasons, and the rule misbehaves accordingly:

  * Flooring at zero before summing ~20 mutations turns noise into a positive
    offset: the median site scores 0.66-2.03 depending on the antibody, so the
    null is a bulk sitting well away from zero rather than a spike at it.
  * The fitted shape varies widely between antibodies (a = 0.97 to 5.25), so the
    fit absorbs a diffuse antibody's real signal into the bulk.

Running it anyway (`--sweep` prints this):

    antibody      FDR.01  FDR.05  FDR.10  FDR.20   gamma KS p
    Nirsevimab        9      10      11      12      8e-05  (fit rejected)
    Clesrovimab       6       8      10      13      0.051
    Palivizumab       8      10      11      15      0.017  (fit rejected)
    Suptavumab        3       3       3       4      0.078
    1A2               0       0       0       0      0.047  (fit rejected)
    1B6              19      22      25      33      0.055

1A2 gets zero sites at any FDR up to 0.2 and suptavumab gets three. Including 1A2
is a fixed decision, so an FDR rule cannot be used: it would cut the arm from six
antibodies to four or five. The same holds for every rule keyed to the value of
the statistic rather than its rank -- an absolute cut at stat>=4 gives 1A2 one
site and at stat>=5 gives it none; robust-z (median+MAD) at z>=4 gives it one and
at z>=6 none. See `--sweep`.

The declared rule
-----------------
    A site is an escape site for an antibody if its Fab site statistic is among
    that antibody's top 10 of the 504 sites (top 2.0%). Fixed rank, per antibody,
    ties broken by site number.

`--topk` changes it; 10 is the declared default and the only value any headline
number may use without saying otherwise.

Fixed rank is the only family of rules that leaves all six antibodies usable.
Every value-keyed rule deletes antibodies: the gamma-FDR above gives 1A2 zero
sites at any FDR up to 0.2, an absolute cut at stat>=4 gives it one and at
stat>=5 none, robust-z at z>=4 gives it one and at z>=6 none (`--sweep` prints all
three). It also fixes the base rate at 10/504 = 1.98% for every antibody, so
per-antibody AUPRC and the random floor are directly comparable; a variable-size
target set would make a cross-antibody comparison the same category error as
quoting numbers taken on different cohorts against each other. And 10 sites sits
inside the 4-15 the HIV arm's inherited Dingens sets carry, so the two arms stay
commensurable.

Why 10 and not more: the region concentration sweep
-----------------------------------------------------
Each antibody's own antigenic region is declared from the published epitope rather
than inferred (`OWN_REGION` below). A top-10-dominant heuristic assigns 1A2 to
region_II, which 9LLY contradicts. Background share of each region, over the 504
scanned sites: region_0 0.133, region_II 0.048, region_IV 0.099, region_V 0.141.

Fraction of the selected sites falling inside the antibody's own region, mean
over the six, by k:

    k        6     8    10    12    15    20    25    30    40
    mean  0.81  0.73  0.70  0.65  0.62  0.53  0.45  0.41  0.35

The curve falls monotonically and has no elbow, so this picks a regime rather than
a point: by k = 20-25 the mean is at 0.45-0.53 and still dropping, so the set is
taking in bulk rather than a tail. The value cut agrees -- rank 25 sits at only
2.5-4.3x the per-antibody median, where rank 10 sits at 2.9-10.1x. k = 10 is the
largest round k still in the concentrated regime, at 5-14x over background. Per
antibody at k = 10:

    antibody      own region   in-region   background   enrichment
    Nirsevimab    region_0       10/10        0.133        7.5x
    Palivizumab   region_II       9/10        0.048       18.9x
    1B6           region_0        9/10        0.133        6.8x
    Clesrovimab   region_IV       7/10        0.099        7.1x
    Suptavumab    region_V        6/10        0.141        4.3x
    1A2           region_IV       1/10        0.099        1.0x   <- see below

`region` is a lower bound on correctness either way: the regions are coarse and
several true footprint sites are labelled `none` (clesrovimab's 235/240/421).

The 1A2 exception
----------------------------------------------------
1A2 is a near-null case, and it is a property of the measurement rather than of
the threshold. Its site statistic is barely tailed: max/median 5.2x, against 15.5x
to 45.0x for the other five, skew 0.89 and kurtosis 1.47 against 4.3-10.3 skew
elsewhere. Its top 10 scatters over region_IV, region_II, region_V, region_0 and
five sites with no region at all; only its single strongest site, 445, is in
region_IV. 1A2 stays in the panel, but it must not influence the threshold, and
any number computed for it in this arm is a null by construction -- a perfect
epitope predictor scores at the floor on 1A2. Label it wherever it appears, so the
panel is not read as "the method failed on 1A2".

Second check: the solved structural epitopes
-----------------------------------------------------------
`region` is a coarse proxy. Four of the six antibodies have a solved F complex
(PROVENANCE.md, "RSV structures"), so the declared sets can be checked against the
real 4.5 A heavy-atom contact set, taken over the whole trimer:
5UDC = nirsevimab, 6OUS = clesrovimab, 9LLY chains H/L = 1A2 and D/I = 1B6
(matched by VH/VL sequence against `RSV_mAb_AAseq.csv` rather than by name).
This is an independent line of evidence and it reaches the same place:

    antibody    |epitope|   in epitope at k=10   enrichment   (at k=25, for contrast)
    Nirsevimab      25           9/10               18.1x        14/25   11.3x
    Clesrovimab     26           7/10               13.6x        11/25    8.5x
    1B6             27           7/10               13.1x        13/25    9.7x
    1A2             24           1/10                2.1x         1/25    0.84x

Enrichment against the true contact set is ~1.4x higher at k=10 than at k=25 for
all three real cases, which is the verdict the region sweep gives by a route that
does not use the `region` column at all.

1A2 fails this check at every k: 1/10, 1/15, 1/20, 1/25, 2/50. Its 9LLY contacts
are 160-161 / 182-184 (site V) and 426-468 (site IV), so it is a site IV/V
antibody, and its escape does not sit there. No choice of k repairs this.

The six sets are nearly disjoint, which is what makes this arm worth running: mean
pairwise Jaccard 0.033 at k = 10, union 51 of 504 sites (10%). The largest overlap
is nirsevimab against 1B6, and they genuinely share site 0. The HIV constraint
arm's target set is identical for all nine antibodies and therefore cannot test
antibody conditioning (`dms_10_functional_sites.py`). This one can.

Frame. Masks are 504-long over `F_protein.fasta`, index i (1-based) == F site
i + 25, so sites 26-529 in full-length F numbering with an offset of exactly 25 at
every row (verified in PROVENANCE.md). The site numbers are also written out
explicitly so no consumer has to reconstruct the offset.

    PYTHONPATH=scripts python scripts/rsv_01_escape_sites.py
    PYTHONPATH=scripts python scripts/rsv_01_escape_sites.py --sweep
"""
import argparse, itertools, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
SRC = PROJECT / "data" / "source" / "dms_simonich2026"
OUT = PROJECT / "data" / "manifests"

# Fab (not IgG). RSM01 is omitted because no public antibody sequence exists.
ANTIBODIES = ["Nirsevimab", "Clesrovimab", "Palivizumab", "Suptavumab", "1A2", "1B6"]
TOPK = 10                 # the declared rule: top 10 of 504 sites = top 2%

# Each antibody's own antigenic region, declared from the published epitope rather
# than inferred from the data. 9LLY puts 1A2's contacts in site IV (426-468) and
# site V (160-161, 182-184). Suptavumab and palivizumab have no F complex to check
# against, so theirs are the canonical literature assignments.
OWN_REGION = {"Nirsevimab": "region_0", "Clesrovimab": "region_IV",
              "Palivizumab": "region_II", "Suptavumab": "region_V",
              "1A2": "region_IV", "1B6": "region_0"}
SITE_OFFSET = 25          # fasta index i (1-based) == F site i + 25
N_SITES = 504


def site_statistic(df, antibody):
    """The authors' own site statistic: sum over mutations of Fab escape, floored at 0.

    `min_count=1` keeps a site NaN if every mutation there is NaN, rather than
    scoring it 0, the same rule as `dms_01_escape`. An unmeasured site is not a
    no-escape site.
    """
    col = f"{antibody}-Fab escape"
    v = df[col].clip(lower=0)
    return df.assign(_v=v).groupby("site")["_v"].sum(min_count=1)


def escape_mask(stat, k=TOPK):
    """Top-k sites by the statistic. Ties broken by ascending site number."""
    order = stat.sort_values(ascending=False, kind="mergesort").index[:k]
    return stat.index.isin(order)


# ---- the HIV arm's rule, reimplemented here to show it does not apply --------
def _bh(p, fdr):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.clip(np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    out = np.empty(n); out[o] = q
    return out, out <= fdr


def find_sig_sel(v, fdr=0.01):
    """dms_tools2 `robust_hist`, as in dms_10_functional_sites. Diagnostic only."""
    v = np.asarray(v, float)
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


def load():
    df = pd.read_csv(SRC / "all_antibodies.csv")
    sites = np.sort(df.site.unique())
    assert len(sites) == N_SITES and sites[0] == 26 and sites[-1] == 529
    assert (df.site - df.sequential_site == SITE_OFFSET).all(), "site offset is not 25"
    stat = pd.DataFrame({ab: site_statistic(df, ab) for ab in ANTIBODIES}).reindex(sites)
    region = df.drop_duplicates("site").set_index("site")["region"].reindex(sites).fillna("none")
    return stat, region


def own_region(stat, region, ab):
    """The declared own region. Takes `stat`/`region` to keep the call sites
    uniform; it does not look at the data (see OWN_REGION)."""
    return OWN_REGION[ab]


def sweep(stat, region):
    n = len(stat)
    print("=== distribution of the site statistic (504 sites) ===")
    for ab in ANTIBODIES:
        x = stat[ab].values
        print(f"  {ab:12s} n={np.isfinite(x).sum():3d} mean={np.nanmean(x):5.2f} "
              f"median={np.nanmedian(x):5.2f} sd={np.nanstd(x):5.2f} "
              f"skew={stats.skew(x):5.2f} kurt={stats.kurtosis(x):6.2f} max={np.nanmax(x):6.2f}")

    print("\n=== the HIV arm's gamma-tail/FDR rule, applied here (DIAGNOSTIC ONLY) ===")
    for ab in ANTIBODIES:
        v = stat[ab].values
        counts = []
        for fdr in (0.01, 0.05, 0.10, 0.20):
            sig, par = find_sig_sel(v, fdr)
            counts.append(int(sig.sum()))
        ks = stats.kstest(v, "gamma", args=(par[0], par[2], par[1]))
        flag = "  gamma fit REJECTED (KS p<0.05)" if ks.pvalue < 0.05 else ""
        print(f"  {ab:12s} FDR.01={counts[0]:3d} .05={counts[1]:3d} .10={counts[2]:3d} "
              f".20={counts[3]:3d}   KS p={ks.pvalue:.3g}{flag}")

    own = {ab: own_region(stat, region, ab) for ab in ANTIBODIES}
    share = region.value_counts() / n
    print("\n=== SWEEP A: fixed rank (the declared family). k sites per antibody ===")
    print("  own regions are DECLARED, not inferred: " + str(OWN_REGION))
    print("  mean in-own-region fraction over the six, by k:")
    for k in (6, 8, 10, 12, 15, 20, 25, 30, 40):
        v = [float((region.loc[stat[ab].nlargest(k).index] == OWN_REGION[ab]).sum()) / k
             for ab in ANTIBODIES]
        mark = "   <- DECLARED" if k == TOPK else ""
        print(f"    k={k:3d}  mean={np.mean(v):.2f}{mark}")
    for k in (6, 10, 15, 20, 25, 50):
        print(f"  --- k={k}  (top {100*k/n:.1f}%)")
        for ab in ANTIBODIES:
            sel = stat[ab].nlargest(k).index
            hit = int((region.loc[sel] == own[ab]).sum())
            thr = float(stat[ab].nlargest(k).min())
            print(f"      {ab:12s} cut>={thr:6.2f}  in {own[ab]:10s} {hit:2d}/{k} "
                  f"({100*hit/k:5.1f}%)  enrich {hit/k/share[own[ab]]:5.2f}x")
        sets = {ab: set(stat[ab].nlargest(k).index) for ab in ANTIBODIES}
        js = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
              for a, b in itertools.combinations(ANTIBODIES, 2)]
        print(f"      union={len(set().union(*sets.values())):3d}/{n}  "
              f"mean pairwise Jaccard={np.mean(js):.3f}")

    print("\n=== SWEEP B: absolute cut on the statistic (set sizes; in-own-region) ===")
    for cut in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        cells = []
        for ab in ANTIBODIES:
            sel = stat.index[stat[ab] >= cut]
            cells.append(f"{ab}:{len(sel)}({int((region.loc[sel]==own[ab]).sum())})")
        print(f"  stat>={cut:4.1f}  " + "  ".join(cells))

    print("\n=== SWEEP C: robust z = (stat - median)/(1.4826*MAD) ===")
    for z in (3, 4, 5, 6, 8):
        cells = []
        for ab in ANTIBODIES:
            v = stat[ab].values
            med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826
            sel = stat.index[(v - med) / mad >= z]
            cells.append(f"{ab}:{len(sel)}({int((region.loc[sel]==own[ab]).sum())})")
        print(f"  z>={z}  " + "  ".join(cells))

    print("\nSweeps B and C are the reason the declared rule is fixed-RANK: both "
          "drop 1A2 to 0-6 sites,\nand 1A2 is in the panel by decision.")


# --- the harder sanity check: the solved 4.5 A structural epitopes -------------
# Chain assignments were made by VH/VL sequence identity against
# RSV_mAb_AAseq.csv, because names mislead: 9LLY labels 1A2 as H/L and 1B6 as D/I,
# the opposite of what the chain letters suggest. Re-derive before changing.
REF_DIR = PROJECT / "work" / "rsv_f" / "ref"
COMPLEXES = {
    "Nirsevimab":  ("5udc.cif", set("HLBCEG"), set("FAD")),
    "Clesrovimab": ("6ous.cif", set("MNOPQRSTUVWX"), set("ABCDEFGHIJKL")),
    "1A2":         ("9lly.cif", set("HLGNJO"), set("ABC")),
    "1B6":         ("9lly.cif", set("DIEKFM"), set("ABC")),
}
CONTACT_CUT = 4.5


def structural_epitopes():
    import gemmi
    from scipy.spatial import cKDTree

    def heavy_atoms(st, chains):
        P, R = [], []
        for ch in st[0]:
            if ch.name not in chains:
                continue
            for r in ch:
                for a in r:
                    if a.element == gemmi.Element("H"):
                        continue
                    P.append([a.pos.x, a.pos.y, a.pos.z]); R.append(r.seqid.num)
        return np.array(P), np.array(R)

    out = {}
    for ab, (fn, abc, fc) in COMPLEXES.items():
        p = REF_DIR / fn
        if not p.exists():
            print(f"  {ab}: {p} missing -- download it into work/rsv_f/ref/ first"); continue
        st = gemmi.read_structure(str(p)); st.setup_entities(); st.remove_ligands_and_waters()
        A, _ = heavy_atoms(st, abc); F, Fr = heavy_atoms(st, fc)
        idx = cKDTree(A).query_ball_point(F, CONTACT_CUT)
        out[ab] = sorted({int(Fr[i]) for i, v in enumerate(idx) if v})
    return out


def validate_structures(stat):
    ep = structural_epitopes()
    n = len(stat)
    print(f"=== declared sites vs the solved {CONTACT_CUT} A structural epitopes (trimer union) ===")
    for k in (6, 10, 15, 20, 25, 50):
        print(f"  --- k={k}")
        for ab, E in ep.items():
            E = set(E) & set(int(s) for s in stat.index)
            sel = set(stat[ab].nlargest(k).index)
            ov = sel & E
            exp = k * len(E) / n
            print(f"      {ab:12s} |epitope|={len(E):3d}  in epitope {len(ov):2d}/{k}  "
                  f"expected {exp:4.1f}  enrich {len(ov)/exp:5.2f}x")
    print("\n1A2 never exceeds ONE site in its own epitope at any k up to 25 (2 at k=50):\nits escape is not epitope-localised at all. See the docstring.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", action="store_true",
                    help="regenerate the structural-epitope validation table")
    ap.add_argument("--topk", type=int, default=TOPK,
                    help=f"sites per antibody (DECLARED DEFAULT {TOPK} = top 5%% of 504)")
    ap.add_argument("--sweep", action="store_true",
                    help="print the threshold sensitivity sweep and the gamma-FDR diagnostic")
    args = ap.parse_args()

    stat, region = load()
    if args.structures:
        validate_structures(stat)
        return
    if args.sweep:
        sweep(stat, region)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    share = region.value_counts() / len(stat)
    rows = []
    for ab in ANTIBODIES:
        m = escape_mask(stat[ab], args.topk)
        arrays[f"{ab}|escape"] = m
        arrays[f"{ab}|stat"] = stat[ab].values
        sel = stat.index[m]
        own = own_region(stat, region, ab)
        hit = int((region.loc[sel] == own).sum())
        meta[ab] = dict(rule=f"top {args.topk} of {len(stat)} sites "
                             f"({100*args.topk/len(stat):.2f}%) of the floored-sum Fab statistic",
                        n_sites=int(m.sum()), threshold=float(stat[ab][m].min()),
                        median_stat=float(stat[ab].median()), sites=[int(s) for s in sel],
                        own_region=own, n_in_own_region=hit,
                        enrichment=float(hit / args.topk / share[own]))
        rows.append(dict(antibody=ab, k=int(m.sum()), cut=round(float(stat[ab][m].min()), 2),
                         median=round(float(stat[ab].median()), 2), own_region=own,
                         in_region=f"{hit}/{args.topk}",
                         enrich=round(float(hit / args.topk / share[own]), 2)))

    arrays["site_numbers"] = stat.index.values.astype(int)
    np.savez_compressed(OUT / "rsv_escape_sites.npz", **arrays)
    (OUT / "rsv_escape_sites.json").write_text(json.dumps(
        dict(source="Simonich et al. 2026, all_antibodies.csv (Fab columns)",
             statistic="sum over mutations of max(escape, 0), the authors' own",
             frame=f"504 sites, index i (1-based) == F site i + {SITE_OFFSET}; sites 26-529",
             topk=args.topk, antibodies=ANTIBODIES, per_antibody=meta), indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {OUT/'rsv_escape_sites.npz'} and {OUT/'rsv_escape_sites.json'}")


if __name__ == "__main__":
    main()
