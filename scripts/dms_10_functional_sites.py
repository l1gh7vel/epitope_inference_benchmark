"""The escape target set Figure 4 is scored against, and the evidence for it.

Figure 4 asks whether a method's top-k residues contain the positions where a
mutation has a real consequence for the antibody. The target set is the authors'
own escape call, used verbatim: Dingens et al. fit a gamma distribution to the
per-site `avgfracsurvive` and take the upper tail at FDR 0.01, giving 4-15 sites
per antibody. No threshold of ours enters it.

Escape is positive selection only, by design.
The mirror construction would be sites of significant NEGATIVE selection --
mutations the antibody depletes, which sensitise the virus. `fracsurvive` cannot
express it, being clipped at zero (the minimum is exactly 0.0000 in all nine
files), but the authors also publish signed `*_sitediffsel.csv`, so the question
is answerable. It was asked and the answer is no. `--investigate` regenerates all
of the following:

  1. The significance rule reimplements EXACTLY. Running `find_sig_sel` on the
     authors' own `avgfracsurvive` input recovers all nine published site lists
     with zero differences, so what follows is not an artefact of the fit.
  2. Applied to the negative tail it finds essentially nothing: 7 of 9 antibodies
     have ZERO significant negative sites even at FDR 0.2, and the other two have
     1-3. There is no concentrated negative tail to call.
  3. That is not because the negative signal is noise. It is MORE reproducible
     between replicates than the positive one (Spearman 0.58-0.72 against
     0.31-0.55). It is real, but DIFFUSE -- spread over the whole protein rather
     than concentrated at a few positions, which is precisely what a gamma-tail
     rule cannot call, since that rule assumes most sites are null.
  4. The diffuse negative signal is a property of the SITE, not of the antibody:
     its between-ANTIBODY correlation (0.75) exceeds its within-antibody
     between-replicate correlation (0.64), so no antibody-specific component
     survives above noise. Every antibody's negative signal is the same signal.

A target set that is identical for every antibody cannot test antibody
conditioning at all, since an antibody-blind method is under no handicap against
it. That is why the target set is the per-antibody escape call.

Writes `data/manifests/dms_functional_sites.npz` (boolean masks, 856-long, one per
antibody) and `dms_functional_sites.json` (sizes and the measured-site counts).

    PYTHONPATH=scripts python scripts/dms_10_functional_sites.py
    PYTHONPATH=scripts python scripts/dms_10_functional_sites.py --investigate
"""
import argparse, itertools, json, sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "config"))
import dms as cfg                                                    # noqa: E402

SRC = PROJECT / "data" / "source"
D19 = SRC / "dms_dingens2019"
SIG = D19 / "significant_sites"
DIFF = D19 / "diffsel"
MAN = PROJECT / "data" / "manifests"

# file stem -> the name used everywhere else in the project
NAMES = {"101074": "10-1074", "10E8": "10E8", "3BNC117": "3BNC117", "PG9": "PG9",
         "PGT121": "PGT121", "PGT145": "PGT145", "PGT151": "PGT151",
         "VRC01": "VRC01", "VRC34": "VRC34"}
FDR = 0.01          # the authors' FDR, used for the escape calls


# --------------------------------------------------------------- the authors' rule
def _bh(p, fdr):
    """Benjamini-Hochberg, matching statsmodels' multipletests(..., 'fdr_bh')."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.clip(np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[o] = q
    return out, out <= fdr


def find_sig_sel(v, fdr=FDR):
    """dms_tools2.plot.findSigSel, method='robust_hist', reimplemented.

    Bin with the Freedman-Diaconis rule, fit a gamma to the histogram by least
    squares under a soft-L1 loss, take the survival function as a P value, then BH.
    Verified against the authors' published output: see `--validate`.
    """
    v = np.asarray(v, float)
    try:
        edges = np.histogram(v, bins="fd")[1]
    except ValueError:                       # 'fd' fails on many identical points
        edges = np.histogram(v, bins="doane")[1]
    heights, edges = np.histogram(v, bins=edges, density=True)
    bins = (edges[:-1] + edges[1:]) / 2
    scale = v.var(ddof=1) / v.mean()
    fit = optimize.least_squares(
        lambda x: stats.gamma.pdf(bins, x[0], scale=x[1], loc=x[2]) - heights,
        np.array([v.mean() / scale, scale, 0.0]), loss="soft_l1")
    g = stats.gamma(fit.x[0], scale=fit.x[1], loc=fit.x[2])
    P = g.sf(v)
    Q, sig = _bh(P, fdr)
    return P, Q, sig, fit.x


# ------------------------------------------------------------------------ loading
def published_escape(stem):
    """The authors' significant-site list, verbatim. No threshold of ours enters.

    `isdigit` drops sites with a letter suffix, which are insertions relative to
    HXB2 and have no slot in the linear reference frame everything else is indexed
    against (same reason, same handling as `dms_01_escape`). This costs exactly one
    site in the whole dataset: PGT151 publishes 13, site 322a is an insertion, so
    the mask carries 12.
    """
    return {int(t) for t in (SIG / f"{stem}_SignificantSitesList.txt").read_text().split(",")
            if t.strip().isdigit()}


def median_diffsel():
    """Per-antibody median across replicates of the signed site diffsel."""
    out = {}
    for stem, ab in NAMES.items():
        fs = sorted(f for f in DIFF.glob("*_sitediffsel.csv")
                    if f.name.startswith(stem + "-"))
        ds = [pd.read_csv(f, dtype={"site": str}).set_index("site") for f in fs]
        out[ab] = (pd.DataFrame({c: pd.concat([d[c] for d in ds], axis=1).median(axis=1)
                                 for c in ("positive_diffsel", "negative_diffsel")}),
                   [f.name.split("_")[0] for f in fs])
    return out


# ------------------------------------------------------------------- the two checks
def validate():
    """Our fit must reproduce the authors' published lists before we trust it."""
    ok = 0
    for stem in NAMES:
        d = pd.read_csv(DIFF / f"summary_{stem}-mediansitefracsurvive.csv",
                        dtype={"site": str})
        _, _, sig, _ = find_sig_sel(d.avgfracsurvive.values, FDR)
        mine, theirs = set(d.site[sig]), {s.strip() for s in
                                          (SIG / f"{stem}_SignificantSitesList.txt").read_text().split(",")
                                          if s.strip()}
        ok += mine == theirs
        print(f"  {stem:9s} ours {len(mine):2d} / theirs {len(theirs):2d}  "
              f"identical={mine == theirs}")
    print(f"  reproduced {ok}/9 exactly")
    return ok == 9


def investigate():
    """Regenerate the evidence that there is no negative escape tail to call."""
    print("\n1. our fit vs the authors' published escape calls")
    validate()

    med = median_diffsel()
    print("\n2. the same rule applied to the NEGATIVE tail of diffsel")
    print(f"   {'antibody':10s} {'reps':>4s} " + "".join(f"  FDR{f:<5g}" for f in (0.01, 0.05, 0.1, 0.2)))
    for ab, (m, reps) in sorted(med.items()):
        v = (-m.negative_diffsel).clip(lower=0).values
        P, _, _, _ = find_sig_sel(v, 0.01)
        row = "".join(f"{int(_bh(P, f)[1].sum()):9d}" for f in (0.01, 0.05, 0.1, 0.2))
        print(f"   {ab:10s} {len(reps):4d} {row}")

    print("\n3. reproducibility of each tail, mean Spearman over replicate pairs")
    print(f"   {'antibody':10s} {'positive':>9s} {'negative':>9s}")
    for stem, ab in sorted(NAMES.items(), key=lambda kv: kv[1]):
        fs = sorted(f for f in DIFF.glob("*_sitediffsel.csv") if f.name.startswith(stem + "-"))
        ds = [pd.read_csv(f, dtype={"site": str}).set_index("site") for f in fs]
        rp, rn = [], []
        for a, b in itertools.combinations(range(len(ds)), 2):
            ix = ds[a].index.intersection(ds[b].index)
            rp.append(stats.spearmanr(ds[a].positive_diffsel[ix], ds[b].positive_diffsel[ix]).statistic)
            rn.append(stats.spearmanr(ds[a].negative_diffsel[ix], ds[b].negative_diffsel[ix]).statistic)
        print(f"   {ab:10s} {np.mean(rp):9.3f} {np.mean(rn):9.3f}")

    abs_ = sorted(med)
    ix = med[abs_[0]][0].index
    for ab in abs_:
        ix = ix.intersection(med[ab][0].index)
    N = pd.DataFrame({ab: med[ab][0].negative_diffsel[ix] for ab in abs_})
    P = pd.DataFrame({ab: med[ab][0].positive_diffsel[ix] for ab in abs_})
    iu = np.triu_indices(len(abs_), 1)
    print("\n4. between-ANTIBODY Spearman (compare with the between-replicate column above)")
    print(f"   positive {P.corr('spearman').to_numpy()[iu].mean():.3f}"
          f"   negative {N.corr('spearman').to_numpy()[iu].mean():.3f}")
    print("   -> the negative tail agrees BETWEEN antibodies better than a single")
    print("      antibody agrees with itself between replicates: no specific component.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--investigate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        return validate()
    if a.investigate:
        return investigate()

    L = cfg.ANTIGEN_LEN
    esc_arr = np.load(MAN / "dms_escape.npz")
    masks, meta = {}, {}
    for stem, ab in NAMES.items():
        measured = ~np.isnan(esc_arr[f"{ab}|Dingens2019_BG505|mean"])
        escape = np.zeros(L, bool)
        for s in published_escape(stem):
            if 1 <= s <= L:
                escape[s - 1] = True
        escape &= measured
        masks[f"{ab}__escape"] = escape
        meta[ab] = dict(n_escape=int(escape.sum()), n_measured=int(measured.sum()))

    out_npz = MAN / "dms_functional_sites.npz"
    np.savez_compressed(out_npz, **masks)
    (MAN / "dms_functional_sites.json").write_text(json.dumps(
        dict(fdr=FDR, per_antibody=meta), indent=2))
    print(pd.DataFrame(meta).T.to_string())
    print(f"\n-> {out_npz}")


if __name__ == "__main__":
    main()
