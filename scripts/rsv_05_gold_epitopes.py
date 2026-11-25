#!/usr/bin/env python3
"""Gold STRUCTURAL epitopes for the RSV F arm -- for the four antibodies that have one.

This is the RSV equivalent of `dms_00_gold_epitopes.py`, and it differs from it in
one way that has to be stated up front rather than discovered downstream:

*** ONLY 4 OF THE 6 ANTIBODIES HAVE A SOLVED F COMPLEX. ***

    Nirsevimab   5UDC  3.45 A  X-ray    released 2017-05-17   PRE  AF3 cutoff
    Clesrovimab  6OUS  3.40 A  X-ray    released 2019-10-30   PRE  AF3 cutoff
    1A2          9LLY  3.18 A  cryo-EM  released 2026-01-21   POST AF3 cutoff
    1B6          9LLY  3.18 A  cryo-EM  released 2026-01-21   POST AF3 cutoff
    Palivizumab  2HWZ is the Fab ALONE (1.8 A, 2007). NO F complex exists.
    Suptavumab   no PDB entry at all.

No gold epitope is invented for palivizumab or suptavumab. They get NO entry in
`rsv_labels.npz` and `has_gold=False` in the report, so a consumer that needs a
structural label excludes them by asking, while the escape-scored comparisons --
which is what this arm is actually for -- keep all six. Inventing an epitope for
them (e.g. "site II is 254-277, call that palivizumab's epitope") would turn a
LITERATURE REGION into a per-residue gold standard and make the benchmark score
itself against its own annotation.

THE CHAIN TRAP, honoured here and re-verified by sequence
---------------------------------------------------------
9LLY labels **1A2 as chains H+L and 1B6 as chains D+I** -- the opposite of what
the letters suggest (`work/rsv_f/ref/README.md`). The assignment is made by
VH/VL identity against `RSV_mAb_AAseq.csv`, not by name, and this module
DERIVES it at run time (`--verify_chains`, on by default): every Fab chain
named below must match its antibody's VH or VL at >= MIN_AB_IDENTITY coverage, or
the run refuses. A silent chain swap here would hand 1A2's epitope to 1B6 and the
arm would be quietly wrong in both directions at once.

Measured coverage (difflib containment of the CSV sequence in the deposited
chain, 2026-11-01): 0.99-1.00 for all sixteen Fab chains used.

NUMBERING FRAME -- the same one everything else in this arm uses
----------------------------------------------------------------
Masks are 504-long over `F_protein.fasta`; 0-based index i == full-length F site
i + 26, i.e. sites 26-529. Depositor numbering in all four structures IS
full-length F numbering, so depositor residue numbers are site numbers directly.
The script checks this: every F chain's residue identity against
`F_protein.fasta` must clear MIN_IDENTITY (measured 0.94-0.98; chance is ~0.06).

Contacts outside 26-529 are DROPPED and counted. That matters for 6OUS, whose F1
chains run to 544 because the construct carries a fibritin foldon -- those are not
F and must not become gold sites. Glycans and ligands are removed before contact
counting: unlike HIV Env, no RSV F antigenic site is glycan-defined, so there is
no `covale` attribution step here.

The headline definition, and the two further readings reported beside it
------------------------------------------------------------------------
Every one of these structures carries 3 or 6 copies of the same Fab on the same
site of different protomers. Three readings are possible and all three are
computed; the headline is the first, matching `dms_00_gold_epitopes.py` so the
RSV and HIV arms are scored the same way:

  best_copy  the single Fab copy with the most antigen contacts. "One antibody on
             one site", which is what a per-antibody epitope means and what a
             predicted 1:1 complex can possibly reproduce.  <-- HEADLINE
  union      every copy pooled. This is what `rsv_01_escape_sites.py --structures`
             used for its validation table, so that table's |epitope| column is
             the union and is not this file's n_gold unless they coincide.
  per-copy spread is reported for all copies, so a structure where the copies
             disagree is visible rather than averaged away.

Monomer or trimer: the cross-protomer measurement that settles it
-----------------------------------------------------------------
The complexes this arm folds next can present F as one protomer (504 tokens, the
Task III convention, and 504 + VH + VL ~ 745 fits the project's 1,536 AF3 bucket)
or as the trimer (1,512 tokens, and 1,512 + VH + VL ~ 1,750 does NOT). The
question that settles it is empirical: does any of these four Fabs actually touch
more than one protomer? This module measures it -- `cross_protomer_frac` is the
fraction of the best copy's contacted F residues that are reached only on a
protomer other than its primary one -- and prints it. Do not choose the assembly
without reading that number.

    PYTHONPATH=scripts python scripts/rsv_05_gold_epitopes.py
    PYTHONPATH=scripts python scripts/rsv_05_gold_epitopes.py --cutoff 5.0
"""
import argparse, csv, difflib, io, json, os
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
SRC = PROJECT / "data" / "source" / "dms_simonich2026"
REF_DIR = PROJECT / "work" / "rsv_f" / "ref"
OUT = PROJECT / "data" / "manifests"

SITE_LO, SITE_HI = 26, 529          # the scanned ectodomain; index i == site i + 26
N_SITES = 529 - 26 + 1              # 504
CONTACT_CUTOFF = 4.5                # project-wide, every task
MIN_IDENTITY = 0.40                 # F chain vs F_protein.fasta; measured 0.94-0.98
MIN_AB_IDENTITY = 0.85              # Fab chain vs RSV_mAb_AAseq.csv; measured 0.99-1.00

# `assemblies` groups protomer INDICES into biological trimers. It exists because
# 6OUS has TWO trimers in one asymmetric unit, and a Fab bound to trimer 2 sits
# 4.5 A from a protomer of trimer 1 -- a CRYSTAL-LATTICE contact, not an epitope.
# Measured 2026-11-01, Fab U/V: 17 sites on its own protomer (I+J), 4 on its
# in-trimer neighbour (K+L), and 3 on E+F, which belongs to the OTHER trimer.
# Those 3 would have entered the gold set as real epitope residues. Contacts
# outside the primary protomer's own assembly are therefore counted separately as
# `lattice` and EXCLUDED. 5UDC and 9LLY each hold one trimer, so this is a no-op
# for them and the field is still stated rather than defaulted.
#
# Protomer grouping is DECLARED, not guessed, and it is non-obvious for 6OUS,
# whose F is cleaved into F2 (26-103) and F1 (137-544) as separate chains. The
# pairing below was established by heavy-atom contact count between every F2 and
# every F1 chain (2026-11-01): the declared partner carries 318-345 atom contacts
# and the best non-partner 0-42, a >7x separation, and the twelve chains fall into
# two disjoint trimers {A,B,C,D,E,F} and {G,H,I,J,K,L}. The naive test -- distance
# from the F2 C-terminus to the F1 N-terminus -- gives the WRONG answer here,
# because the p27 peptide (104-136) is excised so the two ends are 18-33 A apart
# and the nearest F1 N-terminus often belongs to the neighbouring protomer.
CASES = {
    "Nirsevimab": dict(
        pdb="5udc", resolution=3.45, method="X-ray", released="2017-05-17",
        epitope_class="region_0", pre_cutoff=True,
        construct="RSV A2 F DS-Cav1 prefusion-stabilised trimer + MEDI8897 Fab",
        protomers=[["F"], ["A"], ["D"]], assemblies=[[0, 1, 2]],
        fabs=[("H", "L"), ("B", "C"), ("E", "G")]),
    "Clesrovimab": dict(
        pdb="6ous", resolution=3.40, method="X-ray", released="2019-10-30",
        epitope_class="region_IV", pre_cutoff=True,
        construct="RSV F1 fused to a fibritin trimerisation domain + RB1 (MK-1654) Fab; "
                  "TWO trimers in the asymmetric unit, F cleaved into F2 and F1 chains",
        protomers=[["A", "B"], ["C", "D"], ["E", "F"],
                   ["G", "H"], ["I", "J"], ["K", "L"]],
        assemblies=[[0, 1, 2], [3, 4, 5]],
        fabs=[("M", "N"), ("O", "P"), ("Q", "R"), ("S", "T"), ("U", "V"), ("W", "X")]),
    "1A2": dict(
        pdb="9lly", resolution=3.18, method="cryo-EM", released="2026-01-21",
        epitope_class="region_IV", pre_cutoff=False,
        construct="RSV F prefusion trimer + 1A2 and 1B6 Fabs on the same particle",
        protomers=[["A"], ["B"], ["C"]], assemblies=[[0, 1, 2]],
        fabs=[("H", "L"), ("G", "N"), ("J", "O")],
        note="MEASURED NULL (rsv_01_escape_sites.py): 1A2's declared escape sites do "
             "not sit on these contacts -- 1/10 overlap at k=10, against 9/10 for "
             "nirsevimab. Any number computed for 1A2 in this arm is a null by "
             "construction, not a method failure."),
    "1B6": dict(
        pdb="9lly", resolution=3.18, method="cryo-EM", released="2026-01-21",
        epitope_class="region_0", pre_cutoff=False,
        construct="RSV F prefusion trimer + 1A2 and 1B6 Fabs on the same particle",
        protomers=[["A"], ["B"], ["C"]], assemblies=[[0, 1, 2]],
        fabs=[("D", "I"), ("E", "K"), ("F", "M")]),
}

# The two antibodies with no structural epitope. Recorded, never invented.
NO_STRUCTURE = {
    "Palivizumab": dict(epitope_class="region_II", pre_cutoff=True,
                        reason="Fab-alone structure 2HWZ (1.8 A, 2007); no F complex "
                               "has ever been deposited. The Fab coordinates predate "
                               "the AF3 template cutoff but carry no epitope."),
    "Suptavumab": dict(epitope_class="region_V", pre_cutoff=None,
                       reason="no PDB entry at all, Fab or complex."),
}

# Canonical antigenic-region ranges, as shipped with the DMS data
# (data/source/PROVENANCE.md, "Antigenic regions ship with the data").
REGIONS = {"region_0": [(63, 96), (195, 227)],
           "region_II": [(254, 277)],
           "region_IV": [(422, 471)],
           "region_V": [(55, 62), (146, 194), (287, 300)]}


def f_sequence():
    return "".join(l.strip() for l in (SRC / "F_protein.fasta").read_text().splitlines()
                   if not l.startswith(">"))


def mab_sequences():
    """mAb -> {'VH':..., 'VL':...}. The CSV has a UTF-8 BOM on its header."""
    raw = (SRC / "RSV_mAb_AAseq.csv").read_text().lstrip("﻿")
    return {r["mAb"]: {"VH": r["VH"].strip(), "VL": r["VL"].strip()}
            for r in csv.DictReader(io.StringIO(raw))}


# The ONE naming reconciliation this dataset needs. The escape tables, the
# figures and every other RSV script call it "Clesrovimab"; the antibody-sequence
# CSV calls it "MK-1654 (Clesrovimab)" -- the development code plus the INN. They
# are the same molecule (MK-1654 = clesrovimab = the PDB's "RB1"). Mapped here
# ONCE, explicitly, rather than by a startswith/contains rule that would silently
# pick a different row if the file ever gains another MK- entry.
CSV_NAME = {"Clesrovimab": "MK-1654 (Clesrovimab)"}


def olc(res):
    t = gemmi.find_tabulated_residue(res.name)
    return t.one_letter_code.upper() if t and t.is_amino_acid() else None


def chain_seq(chain):
    return "".join(c for c in (olc(r) for r in chain) if c)


def containment(query, subject):
    """Fraction of `query` explained by >=4-residue blocks shared with `subject`.

    This is containment rather than similarity, for the reason common.py records: a global ratio
    is bounded by the length difference, and a deposited Fab chain carries ~100
    extra constant-domain residues that the CSV's variable domain does not.
    autojunk=False is mandatory on protein alphabets (common.py, trap 1).
    """
    if not query:
        return 0.0
    sm = difflib.SequenceMatcher(None, query, subject, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks() if b.size >= 4) / len(query)


def heavy_atoms(model, chains):
    """(coords[N,3], site_number[N]) over non-hydrogen atoms of `chains`."""
    P, S = [], []
    for name in chains:
        for r in model[name]:
            if olc(r) is None:
                continue
            for a in r:
                if a.element == gemmi.Element("H"):
                    continue
                P.append([a.pos.x, a.pos.y, a.pos.z]); S.append(r.seqid.num)
    return np.asarray(P, float), np.asarray(S, int)


def f_identity(model, chains, fseq):
    """Residue-identity of these F chains against F_protein.fasta, in site frame."""
    ok = tot = 0
    for name in chains:
        for r in model[name]:
            c = olc(r)
            if c is None or r.seqid.icode.strip():
                continue
            n = r.seqid.num
            if SITE_LO <= n <= SITE_HI:
                tot += 1; ok += (c == fseq[n - SITE_LO])
    return (ok / tot if tot else 0.0), tot


def region_of(site):
    for name, spans in REGIONS.items():
        if any(lo <= site <= hi for lo, hi in spans):
            return name
    return "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF,
                    help=f"heavy-atom contact distance in A (default {CONTACT_CUTOFF}, "
                         "the project-wide value used by every other task)")
    ap.add_argument("--no_verify_chains", action="store_true",
                    help="skip the VH/VL identity re-derivation of the 9LLY chain "
                         "assignment. Do not use; the check is the trap guard.")
    a = ap.parse_args()

    fseq = f_sequence()
    assert len(fseq) == N_SITES, f"F_protein.fasta is {len(fseq)} aa, expected {N_SITES}"
    mabs = mab_sequences()

    labels, report = {}, {}
    for ab, spec in CASES.items():
        path = REF_DIR / f"{spec['pdb']}.cif"
        if not path.exists():
            raise SystemExit(f"{ab}: {path} missing. See work/rsv_f/ref/README.md.")
        st = gemmi.read_structure(str(path))
        st.setup_entities(); st.remove_ligands_and_waters()
        model = st[0]
        present = {c.name for c in model}

        # --- frame check: is depositor numbering full-length F numbering? -----
        ident = {}
        for prot in spec["protomers"]:
            for cn in prot:
                assert cn in present, f"{ab}: F chain {cn} not in {spec['pdb']}"
                ident[cn], n = f_identity(model, [cn], fseq)
                if ident[cn] < MIN_IDENTITY:
                    raise SystemExit(
                        f"{ab}: F chain {cn} identity {ident[cn]:.2f} over {n} residues "
                        f"< {MIN_IDENTITY}; numbering is not full-length F. Refusing.")

        # --- chain-assignment check: the 9LLY trap -----------------------------
        csv_name = CSV_NAME.get(ab, ab)
        ab_cov = {}
        if not a.no_verify_chains:
            if csv_name not in mabs:
                raise SystemExit(f"{ab}: no row '{csv_name}' in RSV_mAb_AAseq.csv")
            for h, l in spec["fabs"]:
                for cn, role in ((h, "VH"), (l, "VL")):
                    assert cn in present, f"{ab}: Fab chain {cn} not in {spec['pdb']}"
                    cov = containment(mabs[csv_name][role], chain_seq(model[cn]))
                    ab_cov[f"{cn}:{role}"] = round(cov, 3)
                    if cov < MIN_AB_IDENTITY:
                        raise SystemExit(
                            f"{ab}: chain {cn} matches {csv_name} {role} at only "
                            f"{cov:.2f} (< {MIN_AB_IDENTITY}). The chain assignment is "
                            "wrong -- see the 9LLY trap in this module's docstring.")

        # --- contacts, per Fab copy -------------------------------------------
        prot_atoms = [heavy_atoms(model, p) for p in spec["protomers"]]
        prot_trees = [cKDTree(P) for P, _ in prot_atoms]

        asm_of = {i: ai for ai, members in enumerate(spec["assemblies"]) for i in members}
        assert set(asm_of) == set(range(len(spec["protomers"]))), \
            f"{ab}: every protomer must belong to exactly one assembly"

        copies = []
        for h, l in spec["fabs"]:
            A, _ = heavy_atoms(model, [h, l])
            tree = cKDTree(A)
            per_prot, dropped = [], 0
            for (P, S), _t in zip(prot_atoms, prot_trees):
                hit = tree.query_ball_point(P, a.cutoff)
                sites = {int(S[i]) for i, v in enumerate(hit) if v}
                dropped += sum(1 for s in sites if not SITE_LO <= s <= SITE_HI)
                per_prot.append({s for s in sites if SITE_LO <= s <= SITE_HI})
            primary = int(np.argmax([len(s) for s in per_prot]))
            own = asm_of[primary]
            # Only the primary protomer's OWN trimer is biology; anything on
            # another assembly is crystal packing (see the `assemblies` note).
            in_asm = [s for i, s in enumerate(per_prot) if asm_of[i] == own]
            lattice = set().union(*[s for i, s in enumerate(per_prot)
                                    if asm_of[i] != own] or [set()])
            allsites = set().union(*in_asm)
            cross = allsites - per_prot[primary]
            copies.append(dict(fab=[h, l], n=len(allsites), sites=sorted(allsites),
                               primary_protomer="+".join(spec["protomers"][primary]),
                               assembly=own,
                               n_on_primary=len(per_prot[primary]),
                               n_cross_protomer=len(cross),
                               cross_protomer_sites=sorted(cross),
                               n_lattice_excluded=len(lattice - allsites),
                               lattice_sites_excluded=sorted(lattice - allsites),
                               n_dropped_outside_frame=dropped))

        copies.sort(key=lambda c: -c["n"])
        best = copies[0]
        union = sorted(set().union(*[set(c["sites"]) for c in copies]))
        gold = best["sites"]                       # HEADLINE: the best single copy

        vec = np.zeros(N_SITES, np.int8)
        vec[[s - SITE_LO for s in gold]] = 1
        labels[f"{ab}__gold"] = vec
        uvec = np.zeros(N_SITES, np.int8)
        uvec[[s - SITE_LO for s in union]] = 1
        labels[f"{ab}__gold_union"] = uvec

        comp = {}
        for s in gold:
            comp[region_of(s)] = comp.get(region_of(s), 0) + 1

        report[ab] = dict(
            has_gold=True, pdb=spec["pdb"].upper(), resolution=spec["resolution"],
            method=spec["method"], released=spec["released"],
            epitope_class=spec["epitope_class"], construct=spec["construct"],
            af3_template_pre_cutoff=spec["pre_cutoff"],
            leakage=("pre_cutoff_complex" if spec["pre_cutoff"] else "post_cutoff_complex"),
            cutoff=a.cutoff, frame=f"sites {SITE_LO}-{SITE_HI}, {N_SITES} long",
            f_chain_identity={k: round(v, 3) for k, v in ident.items()},
            fab_chain_coverage=ab_cov,
            headline="best_copy", fab_copy_used=best["fab"],
            n_gold=len(gold), gold=gold,
            n_gold_union=len(union), gold_union=union,
            copy_epitope_sizes=[c["n"] for c in copies],
            primary_protomer=best["primary_protomer"],
            n_on_primary_protomer=best["n_on_primary"],
            cross_protomer_sites=best["cross_protomer_sites"],
            cross_protomer_frac=round(best["n_cross_protomer"] / max(best["n"], 1), 3),
            lattice_sites_excluded=best["lattice_sites_excluded"],
            n_dropped_outside_frame=best["n_dropped_outside_frame"],
            region_composition=comp,
            measured_null=(ab == "1A2"),
            note=spec.get("note"))

    for ab, spec in NO_STRUCTURE.items():
        report[ab] = dict(has_gold=False, pdb=None, resolution=None, method=None,
                          released=None, epitope_class=spec["epitope_class"],
                          af3_template_pre_cutoff=spec["pre_cutoff"],
                          leakage=("no_complex_fab_only" if ab == "Palivizumab"
                                   else "no_structure"),
                          n_gold=0, gold=[], reason=spec["reason"],
                          measured_null=False)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "rsv_labels.npz", site_numbers=np.arange(SITE_LO, SITE_HI + 1),
                        **labels)
    (OUT / "rsv_gold_report.json").write_text(json.dumps(report, indent=2))

    print(f"contact cutoff {a.cutoff} A, frame sites {SITE_LO}-{SITE_HI} ({N_SITES})\n")
    for ab, r in report.items():
        if not r["has_gold"]:
            print(f"{ab:12s} NO STRUCTURAL EPITOPE  ({r['leakage']})")
            print(f"             {r['reason']}")
            continue
        print(f"{ab:12s} {r['pdb']} {r['resolution']:>5}A {r['method']:<7} "
              f"{r['leakage']:<20} gold={r['n_gold']:3d} (union {r['n_gold_union']}) "
              f"copies={r['copy_epitope_sizes']} cross-protomer={r['cross_protomer_frac']:.2f}")
        print(f"             F-chain identity {list(r['f_chain_identity'].values())}  "
              f"regions {r['region_composition']}")
        print(f"             {r['gold']}")
    print(f"\n-> {OUT/'rsv_labels.npz'}  and  {OUT/'rsv_gold_report.json'}")
    print("\nREAD cross_protomer_frac BEFORE choosing the monomer or trimer assembly.")


if __name__ == "__main__":
    main()
