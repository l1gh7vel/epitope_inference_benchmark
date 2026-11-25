"""Per-site escape potential in the HXB2 frame, from both Bloom-lab papers.

  Dingens et al. 2019 (Immunity), "antigenic atlas"  -- 9 bnAbs, BG505.T332N Env,
      metric = median EXCESS fraction surviving across replicates.
  Radford & Bloom 2025 (JVI)                         -- 2 bnAbs x 2 Envs
      (BF520 clade A, TRO.11 clade B), metric = escape_mean.

Both number sites by HXB2, which is what makes them commensurable with env.fa
and with the structures. Verified two ways: Radford's own site_numbering_map
matches env.fa at 843/843 integer reference positions, and Dingens' wild-type
residues match Radford's BF520 wild-type at 521/611 sites versus ~9% for any
shifted frame.

The 521/611 is evidence about the FRAME, not the strain. Dingens is BG505.T332N
and Radford's is BF520: two different clade A Envs, differing at 15.0% of shared
wild-type sites. ~85% agreement is what that comparison should give, and reading
it as a same-strain check would be an error. The npz keys say
`Dingens2019_BG505`, which is the strain the data is from. `Radford2025_BF520`, in the same archive,
IS genuinely BF520 and is a different dataset; never conflate the two.

A site absent from a file is a site with no trustworthy measurement -- NaN, not
zero. Writing zero would fill unmeasured sites with the most extreme "no escape"
value and bias every downstream test.

Sites with a letter suffix are insertions relative to HXB2, have no reference
position, and are dropped -- counted and reported, never silently.
"""
import argparse, json
import os
from pathlib import Path
import numpy as np, pandas as pd

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
ENV_FA = PROJECT / "data" / "source" / "env.fa"
D19 = PROJECT / "data" / "source" / "dms_dingens2019"
R25 = PROJECT / "data" / "source" / "dms_radford2025"
OUT = PROJECT / "data" / "manifests"

DINGENS = {"VRC01": "VRC01", "3BNC117": "3BNC117", "10-1074": "101074",
           "PGT121": "PGT121", "PGT151": "PGT151", "PG9": "PG9",
           "10E8": "10E8", "PGT145": "PGT145", "VRC34": "VRC34"}
RADFORD = {("3BNC117", "BF520"): "BF520_Env_3BNC117_mut_effects_filtered.csv",
           ("3BNC117", "TRO11"): "TRO11_Env_3BNC117_mut_effects_filtered.csv",
           ("10-1074", "BF520"): "BF520_Env_10-1074_mut_effects_filtered.csv",
           ("10-1074", "TRO11"): "TRO11_Env_10-1074_mut_effects_filtered.csv"}
STATS = ("mean", "sum", "max")


def collapse(d, valcol, L):
    s = d.site.astype(str)
    ins = s[~s.str.fullmatch(r"\d+")].nunique()
    d = d[s.str.fullmatch(r"\d+")].copy()
    d["pos"] = d.site.astype(str).astype(int)
    d = d[(d.pos >= 1) & (d.pos <= L)]
    out = {}
    for st in STATS:
        g = d.groupby("pos")[valcol].agg(st)
        v = np.full(L, np.nan); v[g.index.values - 1] = g.values
        out[st] = v
    return out, len(d), int(ins)


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    env = "".join(l.strip() for l in open(ENV_FA) if not l.startswith(">"))
    L = len(env)
    arrays, meta, rows = {}, {}, []

    for ab, fname in DINGENS.items():
        d = pd.read_csv(D19 / f"{fname}_medianmutfracsurvive.csv", dtype={"site": str})
        got, nmut, nins = collapse(d, "mutfracsurvive", L)
        for st in STATS:
            arrays[f"{ab}|Dingens2019_BG505|{st}"] = got[st]
        n = int((~np.isnan(got["mean"])).sum())
        meta[f"{ab}|Dingens2019_BG505"] = dict(paper="Dingens2019", strain="BG505.T332N",
            metric="median excess fraction surviving", n_mutations=nmut,
            n_sites=n, n_insertion_sites_dropped=nins)
        rows.append(dict(antibody=ab, dataset="Dingens2019_BG505", n_sites=n, n_mut=nmut))

    for (ab, strain), fn in RADFORD.items():
        d = pd.read_csv(R25 / fn, dtype={"site": str})
        got, nmut, nins = collapse(d, "escape_mean", L)
        key = f"Radford2025_{strain}"
        for st in STATS:
            arrays[f"{ab}|{key}|{st}"] = got[st]
        n = int((~np.isnan(got["mean"])).sum())
        meta[f"{ab}|{key}"] = dict(paper="Radford2025", strain=strain,
            metric="escape_mean", n_mutations=nmut, n_sites=n,
            n_insertion_sites_dropped=nins)
        rows.append(dict(antibody=ab, dataset=key, n_sites=n, n_mut=nmut))

    np.savez_compressed(OUT / "dms_escape.npz", **arrays)
    (OUT / "dms_escape_meta.json").write_text(json.dumps(meta, indent=2))
    df = pd.DataFrame(rows).sort_values(["dataset", "antibody"])
    print(df.to_string(index=False))
    print(f"\n{len(arrays)//len(STATS)} (antibody, dataset) escape maps written")


if __name__ == "__main__":
    main()
