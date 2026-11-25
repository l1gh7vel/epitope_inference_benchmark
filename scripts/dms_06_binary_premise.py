"""Premise check under the AUTHORS' OWN binary definition of an escape site.

`dms_03_premise.py` treats escape as CONTINUOUS and asks a rank question: is escape
higher at the structural epitope than elsewhere? That avoids choosing a threshold,
and it is the only option for Radford 2025, whose metric is on a different scale
from Dingens'. But it is not the definition Dingens et al. themselves use.

They define SITES OF SIGNIFICANT ESCAPE by fitting a gamma distribution to the
per-site mean excess fraction surviving and taking sites above a q = 0.01 cut
(`SignificantEscape/<AB>_avgfracsurvive_0.01_fit.pdf` in their repo). The resulting
site lists are published, so this script uses THEIR lists verbatim rather than
reimplementing the fit -- there is no risk of our reimplementation disagreeing with
the paper.

The test becomes a 2x2 over measured sites:

                        significant escape    not
    in gold epitope              a             b
    not in epitope               c             d

and a one-sided Fisher exact (greater). Fisher is legitimate here, whereas it was not for the Task III
domain-call table, because the units are SITES, each site is counted
once, and the two classifications come from independent sources -- the epitope from
a crystal structure, the escape call from a mutational scan. Nothing in one row is
a deterministic function of the other, which is exactly the flaw that killed the
odds ratio in fig4d.

Only the nine Dingens antibodies appear: Radford 2025 publishes no significant-site
calls, so this definition simply does not exist for those datasets.
"""
import json
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
MAN = PROJECT / "data" / "manifests"
SIG = PROJECT / "data" / "source" / "dms_dingens2019" / "significant_sites"
RES = PROJECT / "results" / "dms_case"

# their file stem -> our antibody id
NAMES = {"VRC01": "VRC01", "3BNC117": "3BNC117", "101074": "10-1074",
         "PGT121": "PGT121", "PGT151": "PGT151", "PG9": "PG9",
         "10E8": "10E8", "PGT145": "PGT145", "VRC34": "VRC34"}


def sig_sites(stem):
    txt = (SIG / f"{stem}_SignificantSitesList.txt").read_text().strip()
    out = set()
    for tok in txt.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out


def main():
    esc = np.load(MAN / "dms_escape.npz")
    lab = np.load(MAN / "dms_labels.npz")
    rep = json.loads((MAN / "dms_gold_report.json").read_text())
    rows = []
    for stem, ab in NAMES.items():
        sig = sig_sites(stem)
        v = esc[f"{ab}|Dingens2019_BG505|mean"]
        measured = ~np.isnan(v)                      # the 661 sites they assayed
        gold = lab[f"{ab}__gold"].astype(bool)
        L = len(v)
        is_sig = np.zeros(L, bool)
        for s in sig:
            if 1 <= s <= L:
                is_sig[s - 1] = True
        # restrict EVERYTHING to measured sites: an unassayed site is neither a
        # significant escape site nor evidence against one.
        g, s_, m = gold & measured, is_sig & measured, measured
        a = int((g & s_).sum()); b = int((g & ~s_).sum())
        c = int((~g & s_ & m).sum()); d = int((~g & ~s_ & m).sum())
        orr, p_gt = fisher_exact([[a, b], [c, d]], alternative="greater")
        _, p_2s = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        rows.append(dict(antibody=ab, epitope_class=rep[ab]["epitope_class"],
                         n_sig_sites=int(s_.sum()), n_gold_measured=int(g.sum()),
                         overlap=a, expected=round(float(g.sum()) * s_.sum() / m.sum(), 2),
                         odds_ratio=round(float(orr), 2) if np.isfinite(orr) else np.inf,
                         p_greater=float(p_gt), p_two_sided=float(p_2s)))
    d = pd.DataFrame(rows).sort_values("p_greater")
    RES.mkdir(parents=True, exist_ok=True)
    d.to_csv(RES / "premise_binary_dingens.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.to_string(index=False))
    n_sig = int((d.p_greater < 0.05).sum())
    bonf = 0.05 / len(d)
    print(f"\nsignificant one-sided p<0.05: {n_sig}/{len(d)}")
    print(f"surviving Bonferroni (alpha={bonf:.4f}): {int((d.p_greater < bonf).sum())}/{len(d)}")
    print(f"-> compare with the continuous rank test in premise.csv")


if __name__ == "__main__":
    main()
