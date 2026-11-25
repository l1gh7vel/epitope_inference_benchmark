#!/usr/bin/env python3
"""Is a predicted RSV F structure prefusion or postfusion? The pre-check that
decides how many antibodies the RSV arm can carry.

Why this gates the arm. The Simonich 2026 scan is on wild-type, non-stabilised
Long-strain F (PROVENANCE.md: S155/S190/V207/S290, so no DS-Cav1), because the
readout is F-mediated cell entry and a locked F does not enter. Wild-type F is a
metastable spring that collapses to the postfusion six-helix bundle. Antigenic
sites 0 and V exist only on the prefusion apex. So if a structure predictor
returns postfusion F for this sequence, nirsevimab, suptavumab and 1B6 -- the
three antibodies whose declared escape sites sit in site 0 / site V
(`rsv_01_escape_sites.py`) -- are being scored against a surface that is not in
the model.

The discriminator, measured on references. Prefusion 5UDC chain F against
postfusion 3RRR protomer A+B, CA atoms, full-length F numbering (both deposited in
that frame):

    measure                                      prefusion   postfusion
    CA-RMSD to 5UDC-F over shared residues          0.00        45.20
    CA-RMSD to 3RRR-A+B over shared residues       45.20         0.00
    site 0 lobes 62-69 vs 196-212, centroid        10.84        23.07
    site 0 lobes, minimum CA-CA                     4.36         7.15
    HRA 155-210 end-to-end                         42.72        80.61
    HRA(155-210) to HRB(476-516), minimum CA-CA    41.38        12.70
    protomer max CA-CA extent                     113.33       175.23
    protomer radius of gyration                    30.46        47.61

What a postfusion fold would cost, measured as local-superposition CA-RMSD of each
antigenic region between 5UDC and 3RRR. Global superposition says nothing here,
because the two states differ by 45 A RMSD, so every residue moves ~33 A:

    site II   (254-277)                 0.50 A   preserved
    site III  (46-54, 301-348)          0.92 A   preserved
    site I    (389-421)                 1.38 A   preserved
    site 0    lobe 62-69 alone          1.91 A   each lobe keeps its own fold
    site 0    lobe 196-212 alone        1.72 A      "
    site 0    both lobes together      10.65 A   destroyed as a composite surface
    site IV   (422-471)                11.90 A   destroyed
    site V    (55-62,146-194,287-300)  43.56 A   destroyed

Per-antibody, over the solved 4.5 A contact sets: nirsevimab 12.5 A, 1A2 23.7 A,
1B6 26.2 A, clesrovimab 27.1 A. Site IV goes as well, which takes clesrovimab and
1A2 with it, so only palivizumab (site II) survives a postfusion fold intact. A
postfusion verdict would leave one antibody.

Site 0's first lobe (62-69) is resolved in postfusion F: all eight residues are
present in 3RRR's F2 chain. What changes is where they are -- the lobes separate
from 10.8 A to 23.1 A between centroids, while the minimum CA-CA goes 4.4 -> 7.1
A. The minimum-distance form of the site 0 test is therefore weak, a 2.8 A gap,
and is not used alone; the centroid form is better, and the three global measures
(RMSD, HRA extension, Rg) are unambiguous. This module votes over all of them.

Verdict rule: prefusion if RMSD-to-5UDC < RMSD-to-3RRR by at least 10 A and HRA
end-to-end < 60 A; postfusion if the reverse; ambiguous otherwise. Both components
are reported whatever the verdict, so a split vote is visible rather than averaged
away.

    PYTHONPATH=scripts python scripts/rsv_03_conformer.py --refs
    PYTHONPATH=scripts python scripts/rsv_03_conformer.py \
        --model work/rsv_f/af3_out/rsvF_long_trimer/rsvF_long_trimer_model.cif
"""
import argparse, glob, json, os
from pathlib import Path

import gemmi
import numpy as np

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
REF = PROJECT / "work" / "rsv_f" / "ref"
PRE_CIF, POST_CIF = REF / "5udc.cif", REF / "3rrr.cif"
PRE_CHAINS, POST_CHAINS = {"F"}, {"A", "B"}        # one protomer each (3RRR is cleaved: F2+F1)

LOBE_A, LOBE_B = (62, 69), (196, 212)   # antigenic site 0, its two lobes
HRA, HRB = (155, 210), (476, 516)       # the heptad repeats that form the postfusion bundle
ECTO = (26, 513)                        # window for the global measures
SITE_OFFSET = 25                        # DMS fasta index i (1-based) == F site i + 25


def ca_map(st, chains=None, offset=0):
    """Residue number -> CA coordinate. `offset` shifts a model numbered 1..N into F numbering."""
    out = {}
    for ch in st[0]:
        if chains is not None and ch.name not in chains:
            continue
        for r in ch:
            a = r.find_atom("CA", "*")
            if a is not None:
                out[r.seqid.num + offset] = np.array([a.pos.x, a.pos.y, a.pos.z])
    return out


def _seg(m, lo, hi):
    return np.array([m[i] for i in range(lo, hi + 1) if i in m])


def rmsd(ref, m, window=None):
    """Kabsch CA-RMSD over residues present in both (optionally inside `window`)."""
    common = sorted(set(ref) & set(m))
    if window:
        common = [c for c in common if window[0] <= c <= window[1]]
    if len(common) < 10:
        return None, len(common)
    X = np.array([m[c] for c in common]); Y = np.array([ref[c] for c in common])
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return float(np.sqrt((((Xc @ R) - Yc) ** 2).sum(1).mean())), len(common)


def measure(m, pre, post):
    o = {"n_residues": len(m)}
    A, B = _seg(m, *LOBE_A), _seg(m, *LOBE_B)
    o["site0_lobeA_n"], o["site0_lobeB_n"] = len(A), len(B)
    if len(A) and len(B):
        D = np.linalg.norm(A[:, None] - B[None], axis=-1)
        o["site0_min_CA"] = float(D.min())
        o["site0_centroid"] = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    H, R_ = _seg(m, *HRA), _seg(m, *HRB)
    if len(H) and len(R_):
        o["HRA_HRB_min_CA"] = float(np.linalg.norm(H[:, None] - R_[None], axis=-1).min())
    if len(H) > 2:
        o["HRA_end_to_end"] = float(np.linalg.norm(H[0] - H[-1]))
    P = np.array([m[i] for i in sorted(m) if ECTO[0] <= i <= ECTO[1]])
    o["max_CA_CA_ectodomain"] = float(np.linalg.norm(P[:, None] - P[None], axis=-1).max())
    o["Rg_ectodomain"] = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
    o["rmsd_to_prefusion"], o["n_pre"] = rmsd(pre, m, ECTO)
    o["rmsd_to_postfusion"], o["n_post"] = rmsd(post, m, ECTO)
    o["verdict"] = verdict(o)
    return o


def verdict(o):
    dr = o["rmsd_to_postfusion"] - o["rmsd_to_prefusion"]     # >0 means closer to prefusion
    hra = o.get("HRA_end_to_end", np.nan)
    if dr > 10 and hra < 60:
        return "PREFUSION"
    if dr < -10 and hra > 60:
        return "POSTFUSION"
    return f"AMBIGUOUS (dRMSD={dr:+.1f}, HRA end-to-end={hra:.1f})"


def load(path):
    st = gemmi.read_structure(str(path)); st.setup_entities(); st.remove_ligands_and_waters()
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", default=[],
                    help="a predicted mmCIF; repeatable. Globs allowed.")
    ap.add_argument("--chains", default=None,
                    help="comma-separated chain ids to treat as ONE protomer (default: each chain separately)")
    ap.add_argument("--offset", type=int, default=0,
                    help="add to model residue numbers to reach full-length F numbering "
                         f"(use {SITE_OFFSET} for a model of F_protein.fasta, which starts at site 26)")
    ap.add_argument("--refs", action="store_true", help="also print the two reference protomers")
    ap.add_argument("--json", help="write all rows here")
    args = ap.parse_args()

    pre = ca_map(load(PRE_CIF), PRE_CHAINS)
    post = ca_map(load(POST_CIF), POST_CHAINS)

    rows = []
    if args.refs or not args.model:
        for lbl, m in (("REF 5UDC prefusion (chain F)", pre),
                       ("REF 3RRR postfusion (A+B)", post)):
            rows.append({"label": lbl, **measure(m, pre, post)})

    for pat in args.model:
        for path in sorted(glob.glob(pat)) or [pat]:
            st = load(path)
            groups = ([set(args.chains.split(","))] if args.chains
                      else [{c.name} for c in st[0] if any(r.find_atom("CA", "*") for r in c)])
            for g in groups:
                m = ca_map(st, g, args.offset)
                if len(m) < 100:
                    continue
                rows.append({"label": f"{Path(path).parent.name}/{Path(path).name} chain {'+'.join(sorted(g))}",
                             **measure(m, pre, post)})

    w = max(len(r["label"]) for r in rows)
    print(f"{'structure':<{w}}  {'verdict':<10} {'RMSDpre':>8} {'RMSDpost':>9} "
          f"{'s0cent':>7} {'s0min':>6} {'HRAe2e':>7} {'HRA-HRB':>8} {'Rg':>6} {'extent':>7}")
    for r in rows:
        print(f"{r['label']:<{w}}  {r['verdict'][:10]:<10} {r['rmsd_to_prefusion']:8.2f} "
              f"{r['rmsd_to_postfusion']:9.2f} {r.get('site0_centroid', float('nan')):7.2f} "
              f"{r.get('site0_min_CA', float('nan')):6.2f} {r.get('HRA_end_to_end', float('nan')):7.2f} "
              f"{r.get('HRA_HRB_min_CA', float('nan')):8.2f} {r['Rg_ectodomain']:6.2f} "
              f"{r['max_CA_CA_ectodomain']:7.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
