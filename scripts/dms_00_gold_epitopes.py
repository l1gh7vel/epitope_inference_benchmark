"""Gold epitopes for every DMS-mapped antibody, in the HXB2 frame of env.fa.

One structure per antibody; the contacted Env positions ARE the gold epitope.
Four things make this non-trivial and each is handled explicitly, never silently:

  1. HXB2 numbering. HIV Env structures are conventionally deposited in HXB2
     numbering, so depositor residue numbers are reference positions directly --
     the convention the Task III configuration uses, and the one that avoids
     the difflib alignment failure. Validated per chain by residue
     identity against env.fa; a chain below MIN_IDENTITY is refused.
  2. Glycans. 10-1074, PGT121, PGT145 and PG9 are glycan-dependent; their
     defining contact (e.g. the N332 glycan) is a sugar, not an amino acid.
     Every contact to a sugar is attributed back to the Asn it is covalently
     linked to via the mmCIF `covale` records.
  3. Scaffolded and peptide constructs. 3U2S presents V1V2 grafted on a 1FD6
     scaffold and numbers the scaffold CONTINUOUSLY with Env: identity is 0.53
     inside V1V2 and 0.05 beyond residue 200. Without an explicit Env span those
     scaffold contacts would be mapped onto real HXB2 positions in C2 and become
     fabricated gold sites. Hence `span` per case.
  4. Multiple copies. Trimer structures carry several Fab copies; we take the ONE
     copy with the most antigen contacts rather than the union, so the epitope is
     "one antibody on one site". Copy-to-copy spread is reported.

Antigen chains are ALL Env chains, so a cross-protomer epitope (PGT145 at the
apex, 3BNC117 reaching the neighbouring V3) is captured; in the HXB2 frame the
protomers collapse onto the same positions, which is what a per-site epitope means.
"""
import argparse, json, collections
import os
from pathlib import Path
import gemmi, numpy as np, pandas as pd

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
ENV_FA = PROJECT / "data" / "source" / "env.fa"
# Use the configured structure directory shared by the DMS pipeline.
import sys as _sys
_sys.path.insert(0, str(PROJECT / "config"))
import dms as _dmscfg                                           # noqa: E402
PDB_DIR = _dmscfg.PDB_DIR
assert PDB_DIR.name == "pdb", PDB_DIR
OUT = PROJECT / "data" / "manifests"
CONTACT_CUTOFF = 4.5
MIN_IDENTITY = 0.40

SUGARS = {"NAG", "BMA", "MAN", "FUC", "GAL", "SIA", "NDG", "GLC", "XYS", "BGC", "A2G"}

# antibody -> structure spec. `pairs` lists every (heavy, light) copy.
CASES = {
 "VRC01":   dict(pdb="3ngb", ag=["G","A","D","I"], pairs=[("H","L"),("B","C"),("E","F"),("J","K")],
                 res=2.68, cls="CD4bs", construct="gp120 core, 93TH057 (clade A/E)"),
 "3BNC117": dict(pdb="4jpv", ag=["G"], pairs=[("H","L")],
                 res=2.83, cls="CD4bs", construct="gp120 core, 93TH057 with loop D/V5 swapped",
                 caveat="monomeric core: the adjacent-protomer V3 contacts reported "
                        "for 3BNC117 cannot form in this construct."),
 "10-1074": dict(pdb="6ccb", ag=["C","G","A","B"], pairs=[("D","E"),("H","L")],
                 res=6.50, cls="V3-glycan", construct="253-11 SOSIP trimer",
                 caveat="6.5 A: side-chain placement is unreliable at this resolution; "
                        "6UDJ at 2.5 A is carried as a cross-check."),
 "PGT121":  dict(pdb="4tvp", ag=["G","B"], pairs=[("H","L")],
                 res=3.10, cls="V3-glycan", construct="BG505 SOSIP.664 + PGT122",
                 caveat="the deposited Fab is PGT122, the clonal relative of PGT121; "
                        "no PGT121-Env complex is deposited."),
 "PGT151":  dict(pdb="6dcq", ag=["A","B","C","D","E","F"], pairs=[("H","L"),("M","N")],
                 res=3.10, cls="gp120-gp41 interface", construct="full-length Env PC64M18C043"),
 "PG9":     dict(pdb="3u2s", ag=["C","G"], pairs=[("A","B"),("H","L")], span=(118, 200),
                 res=1.80, cls="V1V2 apex", construct="V1V2 of ZM109 on a 1FD6 scaffold",
                 caveat="scaffolded V1V2, not a trimer; residues >200 are scaffold and are "
                        "excluded by `span`. V1V2 is also where HXB2 mapping is least certain."),
 "10E8":    dict(pdb="4g6f", ag=["P","F"], pairs=[("H","L"),("B","D")],
                 res=2.10, cls="MPER", construct="gp41 MPER peptide (656-683)",
                 caveat="isolated peptide, so only MPER could ever be contacted; the gold "
                        "epitope is complete for MPER but cannot include anything outside it."),
 "PGT145":  dict(pdb="5v8l", ag=["A","B","C","D","E","F"], pairs=[("J","N")],
                 res=4.30, cls="V1V2 apex", construct="BG505 SOSIP.664 trimer apex",
                 caveat="4.3 A, and a trimer-axis epitope spanning all three protomers."),
 "VRC34":   dict(pdb="6nc3", ag=["A","B","C","I","D","J","E","K","F","M","G","N"],
                 pairs=[("H","L"),("O","T"),("P","U"),("Q","V"),("R","W"),("S","X")],
                 res=4.50, cls="fusion peptide", construct="AMC011 v4.2 SOSIP"),

 # --- construct controls: the same antibody on a TRIMER instead of a monomeric
 # gp120 core. Dingens 2019 reports that 3BNC117 escape is driven partly by its
 # HCDR3 inter-protomer contacts (V3 sites 304/308/316-320, and 207/209), which a
 # gp120 monomer CANNOT contain. These test whether the CD4bs premise failure is
 # an artefact of construct choice or genuine biology.
 "VRC01_trimer":   dict(pdb="6v8x", ag=["A","B","E","F","I","J"],
                 pairs=[("C","D"),("G","H"),("K","L")],
                 res=3.00, cls="CD4bs", construct="BG505 F14 SOSIP trimer",
                 crosscheck_of="VRC01"),
 "3BNC117_trimer": dict(pdb="5v8l", ag=["A","B","C","D","E","F"],
                 pairs=[("G","K"),("H","L"),("I","M")],
                 res=4.30, cls="CD4bs", construct="BG505 SOSIP.664 trimer",
                 crosscheck_of="3BNC117"),
 # cross-check only, not a separate antibody
 "10-1074_6UDJ": dict(pdb="6udj", ag=["G","C","J","M","P","F"], pairs=[("A","B"),("K","L"),("Q","R")],
                 res=2.50, cls="V3-glycan", construct="BG505 SOSIP.664 (resolution cross-check)",
                 crosscheck_of="10-1074"),
}


def is_aa(r):
    t = gemmi.find_tabulated_residue(r.name)
    return bool(t) and t.is_amino_acid()


def olc(r):
    return gemmi.find_tabulated_residue(r.name).one_letter_code.upper()


def glycan_links(path):
    blk = gemmi.cif.read(str(path)).sole_block()
    tab = blk.find("_struct_conn.", ["conn_type_id",
                                     "ptnr1_label_asym_id", "ptnr1_auth_asym_id",
                                     "ptnr1_auth_seq_id", "ptnr1_label_comp_id",
                                     "ptnr2_label_asym_id", "ptnr2_auth_asym_id",
                                     "ptnr2_auth_seq_id", "ptnr2_label_comp_id"])
    link = {}
    for r in tab:
        if r[0] != "covale":
            continue
        for a, b in ((1, 5), (5, 1)):
            if r[a + 3] == "ASN" and r[b + 3] in SUGARS:
                try:
                    link[r[b]] = (r[a + 1], int(r[a + 2]))
                except ValueError:
                    pass
    return link


def epitope_of_pair(st, ns, link, ag_chains, heavy, light, span, cutoff):
    """Contacted Env positions for one Fab copy -> {'direct':set,'glycan':set}."""
    model = st[0]
    lo, hi = span
    found = {"direct": set(), "glycan": set()}
    for cn in (heavy, light):
        if cn not in [c.name for c in model]:
            continue
        for res in model[cn]:
            if not is_aa(res):
                continue
            for atom in res:
                for mark in ns.find_atoms(atom.pos, "\0", radius=cutoff):
                    cra = mark.to_cra(model)
                    if cra.atom.pos.dist(atom.pos) > cutoff:
                        continue
                    tgt, tch = cra.residue, cra.chain.name
                    if is_aa(tgt) and tch in ag_chains:
                        n = tgt.seqid.num
                        if lo <= n <= hi:
                            found["direct"].add(n)
                    elif tgt.name in SUGARS:
                        asn = link.get(tgt.subchain) or link.get(tch)
                        if asn and asn[0] in ag_chains and lo <= asn[1] <= hi:
                            found["glycan"].add(asn[1])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=CONTACT_CUTOFF)
    args = ap.parse_args()
    env = "".join(l.strip() for l in open(ENV_FA) if not l.startswith(">"))
    L = len(env)
    labels, report, manifest = {}, {}, []

    for tag, spec in CASES.items():
        path = PDB_DIR / f"{spec['pdb']}.cif"
        st = gemmi.read_structure(str(path)); st.setup_entities(); st.remove_waters()
        ns = gemmi.NeighborSearch(st, 5.0).populate()
        link = glycan_links(path)
        span = spec.get("span", (1, L))
        present = {c.name for c in st[0]}
        ag = [c for c in spec["ag"] if c in present]

        ids = {}
        for cn in ag:
            ok = tot = 0
            for r in st[0][cn]:
                if not is_aa(r) or r.seqid.icode.strip():
                    continue
                n = r.seqid.num
                if span[0] <= n <= span[1] and 1 <= n <= L:
                    tot += 1; ok += olc(r) == env[n - 1]
            if tot:
                ids[cn] = round(ok / tot, 3)
        if ids and max(ids.values()) < MIN_IDENTITY:
            raise SystemExit(f"{tag}: best Env-chain identity {max(ids.values())} < "
                             f"{MIN_IDENTITY}; numbering is not HXB2, refusing.")

        copies = []
        for h, l in spec["pairs"]:
            f = epitope_of_pair(st, ns, link, ag, h, l, span, args.cutoff)
            copies.append(((h, l), f, len(f["direct"] | f["glycan"])))
        copies.sort(key=lambda x: -x[2])
        (bh, bl), best, nbest = copies[0]
        sizes = [c[2] for c in copies]

        direct, glyc = sorted(best["direct"]), sorted(best["glycan"])
        gold = sorted(set(direct) | set(glyc))
        vec = np.zeros(L, np.int8); vec[[p - 1 for p in gold]] = 1
        labels[f"{tag}__gold"] = vec
        vd = np.zeros(L, np.int8); vd[[p - 1 for p in direct]] = 1
        labels[f"{tag}__gold_direct"] = vd

        seqs = {}
        for role, cn in (("H", bh), ("L", bl)):
            seqs[role] = "".join(olc(r) for r in st[0][cn] if is_aa(r))

        report[tag] = dict(pdb=spec["pdb"].upper(), resolution=spec["res"],
                           epitope_class=spec["cls"], construct=spec["construct"],
                           cutoff=args.cutoff, span=list(span),
                           env_chain_identity=ids, fab_copy_used=[bh, bl],
                           copy_epitope_sizes=sizes,
                           n_direct=len(direct), n_glycan_only=len(set(glyc) - set(direct)),
                           n_gold=len(gold), gold=gold, glycan_sites=glyc,
                           caveat=spec.get("caveat"), crosscheck_of=spec.get("crosscheck_of"))
        if "crosscheck_of" not in spec:
            manifest.append(dict(case_id=tag, pdb=spec["pdb"].upper(),
                                 resolution=spec["res"], epitope_class=spec["cls"],
                                 vh_sequence=seqs["H"], vl_sequence=seqs["L"],
                                 n_gold=len(gold)))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "dms_labels.npz", **labels)
    pd.DataFrame(manifest).to_csv(OUT / "dms_manifest.csv", index=False)
    (OUT / "dms_gold_report.json").write_text(json.dumps(report, indent=2))

    for tag, r in report.items():
        mark = "  (cross-check)" if r["crosscheck_of"] else ""
        print(f"{tag:14s} {r['pdb']} {r['resolution']:>5}A  gold={r['n_gold']:3d} "
              f"(direct {r['n_direct']}, +glycan {r['n_glycan_only']})  "
              f"copies={r['copy_epitope_sizes']}  id={list(r['env_chain_identity'].values())[:3]}{mark}")
        print(f"               {r['gold']}")


if __name__ == "__main__":
    main()
