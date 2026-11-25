"""Premise check: is the structural epitope where the escape potential is?

Nothing downstream means anything unless two things hold.

  (1) The gold epitopes of antibodies targeting DIFFERENT sites are disjoint.
      If every antibody's epitope were the same residues, a model could score well
      without conditioning on the antibody at all -- which is the failure mode this
      whole benchmark exists to detect.
  (2) Escape potential is elevated AT the gold epitope relative to the rest of Env.
      If it is not, then "the prediction missed the escape sites" carries no
      functional meaning for that antibody, because the structural epitope does not
      carry it either.

Test: Mann-Whitney U on per-site escape, gold vs every other MEASURED site.
Reported BOTH one-sided (greater, the directional question) and two-sided, because
a two-sided p covers a result that can come out negative -- and
three of these do. Effect size is the rank-biserial correlation, which is just
2U/(n1*n2)-1: the probability a random gold site out-escapes a random non-gold
site, rescaled to [-1,1]. It does not depend on the escape units, so it is
comparable across two papers that measure escape differently.

Sites with no measurement are excluded, never imputed as zero.
"""
import json, sys
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
MAN = PROJECT / "data" / "manifests"
RES = PROJECT / "results" / "dms_case"
STAT = "mean"


def main():
    esc = np.load(MAN / "dms_escape.npz")
    lab = np.load(MAN / "dms_labels.npz")
    gold_rep = json.loads((MAN / "dms_gold_report.json").read_text())

    datasets = sorted({k.split("|")[1] for k in esc.files})
    rows = []
    for tag, rep in gold_rep.items():
        ab = rep["crosscheck_of"] or tag
        g = lab[f"{tag}__gold"].astype(bool)
        for ds in datasets:
            key = f"{ab}|{ds}|{STAT}"
            if key not in esc.files:
                continue
            v = esc[key]; m = ~np.isnan(v)
            a, b = v[g & m], v[(~g) & m]
            if len(a) < 3:
                continue
            U, p_gt = mannwhitneyu(a, b, alternative="greater")
            _, p_2s = mannwhitneyu(a, b, alternative="two-sided")
            rows.append(dict(
                gold_source=tag, antibody=ab, dataset=ds,
                epitope_class=rep["epitope_class"], pdb=rep["pdb"],
                resolution=rep["resolution"], is_crosscheck=bool(rep["crosscheck_of"]),
                n_gold=len(a), n_background=len(b),
                median_gold=round(float(np.median(a)), 5),
                median_background=round(float(np.median(b)), 5),
                rank_biserial=round(2 * U / (len(a) * len(b)) - 1, 4),
                p_greater=float(p_gt), p_two_sided=float(p_2s),
                neglog10_p_greater=round(float(-np.log10(p_gt)), 3),
                neglog10_p_two_sided=round(float(-np.log10(p_2s)), 3)))
    df = pd.DataFrame(rows)

    # (1) pairwise epitope overlap, primary gold sets only
    prim = [t for t, r in gold_rep.items() if not r["crosscheck_of"]]
    jac = pd.DataFrame(index=prim, columns=prim, dtype=float)
    for i in prim:
        for j in prim:
            A, B = set(gold_rep[i]["gold"]), set(gold_rep[j]["gold"])
            jac.loc[i, j] = round(len(A & B) / len(A | B), 3)

    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "premise.csv", index=False)
    jac.to_csv(RES / "epitope_overlap_jaccard.csv")

    main_df = df[~df.is_crosscheck]
    n_sig = int((main_df.p_greater < 0.05).sum())
    bonf = 0.05 / len(main_df)
    summary = dict(n_tests=len(main_df), n_significant_one_sided=n_sig,
                   bonferroni_alpha=bonf,
                   n_surviving_bonferroni=int((main_df.p_greater < bonf).sum()),
                   n_significant_two_sided=int((main_df.p_two_sided < 0.05).sum()),
                   stat=STAT)
    (RES / "premise_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print("=== (2) escape elevated at the gold epitope? ===")
    show = ["antibody", "dataset", "epitope_class", "n_gold", "rank_biserial",
            "neglog10_p_greater", "neglog10_p_two_sided"]
    print(main_df.sort_values("rank_biserial", ascending=False)[show].to_string(index=False))
    print(f"\nsignificant one-sided p<0.05: {n_sig}/{len(main_df)}; "
          f"surviving Bonferroni (alpha={bonf:.4f}): {summary['n_surviving_bonferroni']}")
    cc = df[df.is_crosscheck]
    if len(cc):
        print("\n=== construct / resolution controls ===")
        print(cc[["gold_source", "dataset", "n_gold", "rank_biserial",
                  "neglog10_p_greater", "neglog10_p_two_sided"]].to_string(index=False))
    print("\n=== (1) pairwise gold-epitope overlap (Jaccard) ===")
    print(jac.to_string())


if __name__ == "__main__":
    main()
