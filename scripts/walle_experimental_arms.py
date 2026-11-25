#!/usr/bin/env python3
"""WALLE on the RSV F and Nipah RBP arms from the DEPOSITED complexes.

This is the RSV/Nipah twin of `scripts/dms_11_walle_experimental.py`, and it
exists for the same reason that one does: `08_walle_infer.py --source
experimental` cannot run these arms.

WHY STAGE 08's EXPERIMENTAL PATH DOES NOT SUFFICE
-----------------------------------------------------------------------------
Stage 08 wants `pdb_id`, `heavy_chain_id_src`, `light_chain_id_src` and
`antigen_chain_src` on the manifest row, takes ONE deposited chain as the
antigen, and maps it onto the reference frame by pairwise ALIGNMENT. All four
assumptions break here:

  1. NEITHER MANIFEST CARRIES THOSE COLUMNS. `rsv_manifest.csv` has `pdb` and
     nothing else; `niv_manifest.csv` has `heavy_chain`/`light_chain` but no
     antigen chain. The antigen chain of the RSV arm is not even a property of
     the antibody -- it is the property of the Fab COPY whose contacts defined
     the gold epitope, and that copy is recorded in `rsv_gold_report.json`.
  2. ONE DEPOSITED CHAIN IS NOT ONE PROTOMER, in 6OUS. That construct is cleaved
     into F2 (26-103) and F1 (137-544) as SEPARATE chains, so one F protomer is
     two deposited chains -- exactly the gp120/gp41 situation dms_11 documents
     for HIV Env. Taking the best-contacted single chain would have handed
     clesrovimab F1 alone.
  3. ALIGNMENT IS THE WRONG MAPPER FOR RSV. Depositor numbering in 5UDC, 6OUS
     and 9LLY IS full-length F numbering, which is the reference frame of
     `F_protein.fasta` (site i == index i + 26; `rsv_05_gold_epitopes.py`
     verifies this at run time, measuring 0.94-0.98 residue identity). Aligning
     instead would have to cope with 6OUS's fibritin foldon (residues past 529)
     and would map it onto real F sites.
  4. ALIGNMENT IS THE *ONLY* MAPPER FOR TWO OF THE FOUR NIPAH CASES, and it is
     not a mapper this script may choose for itself: 6VY4 and 6CMG are HENDRA G,
     and the arm's gold contacts were already placed in Nipah numbering by
     `niv_01_escape_sites._map_to_nipah`. That function is imported and used
     unchanged, so WALLE's antigen frame is the same frame the arm's own escape
     and structural-epitope tables live in. Re-deriving the mapping here would
     be a second, possibly disagreeing, copy of it.

Which antigen chains are scored
-----------------------------------------------------------------------------
Both arms fold a MONOMER (config/rsv_f.py ANTIGEN_ASSEMBLY = "monomer";
config/niv.py ANTIGEN_COPIES = 1), so the experimental antigen is ONE protomer.

  * RSV: the protomer is DECLARED, not re-derived. `rsv_gold_report.json` records
    `fab_copy_used` and `primary_protomer` for each of the four solved cases, and
    those are taken verbatim ("I+J" -> chains I and J). This is deliberately NOT
    dms_11's greedy contact assembly: 6OUS has two trimers in the asymmetric unit
    and its F2 chains carry ~0 Fab contacts, so a contact-ranked greedy would pick
    an F2 chain by name order and could pair clesrovimab's F1 with the F2 of a
    DIFFERENT protomer. `rsv_05_gold_epitopes.py` established the F2/F1 pairing by
    atom-contact count (declared partner 318-345 contacts, best non-partner 0-42)
    and this script reuses that answer rather than re-deriving a weaker one.
  * Nipah: the protomer is the single antigen chain the chosen Fab copy actually
    touches. Measured 2026-11-04 over all candidate chains of each structure:
    nAH1.3 -> A only (B, C, D have 0 heavy-atom contacts), HENV-26 -> A,
    HENV-32 -> B (A has 0), m102.4 -> A. Every case is 100% single-chain, so
    there is no assembly choice to make; the fraction of contacts falling outside
    the chosen chain is recorded as `frac_contacts_off_protomer` so that a future
    structure which is not single-chain cannot pass unnoticed.

The cost of the monomer convention, which the AF3 arm pays identically and which
belongs in the legend rather than being averaged away: three of the four solved
RSV epitopes are 19-26% quaternary (`cross_protomer_frac` in the manifest), so
roughly a fifth of those gold epitopes cannot be reached on one protomer by
EITHER source. None of the four Nipah epitopes is quaternary.

Two limitations, reported rather than repaired
-----------------------------------------------------------------------------
  * `_map_to_nipah` is a GLOBAL alignment, and on both Hendra chains its first
    residue is pulled onto the N-terminus of the Nipah frame as an end-gap
    artefact: 6VY4 chain B maps depositor 187 -> reference 71 and then 188-602
    -> 188-602 (offset 0 for 415 of 416 residues); 6CMG chain A maps 176 -> 71
    and then 177-602 -> 177-602 (offset 0 for 426 of 427). ONE residue per
    structure therefore lands on reference site 71 where it does not belong. It
    is counted here as `n_offset_outliers` and left in place, because the gold
    structural epitopes of this arm were built through the same function and
    silently diverging from them would be worse than one misplaced node. If it
    is ever repaired, repair it in `niv_01_escape_sites._map_to_nipah` so both
    move together.
  * RSV 6OUS chain F1 runs to 544 because of the fibritin trimerisation domain.
    Residues past site 529 are outside the 504-long frame, map to -1, and are
    dropped from the score vector while remaining in the coordinates handed to
    DSSP -- the same treatment dms_11 gives out-of-span HIV residues.

Two antibodies have no experimental run and are not given one. Palivizumab's
only structure, 2HWZ, is the Fab ALONE (`leakage = no_complex_fab_only`) and
suptavumab has no PDB entry at all (`no_structure`). They keep their AF3 scores
and are simply absent from `experimental/`; `has_gold` is the column that says so.

    EIB_TASK=rsv_f python scripts/walle_experimental_arms.py --checkpoint all
    EIB_TASK=niv   python scripts/walle_experimental_arms.py --checkpoint all

Writes <cfg.WORK>/walle/scores/experimental/<checkpoint>/<case>.npz, the same npz
schema stage 08 and dms_11 write, so any reader can point at a different <source>.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
TASK = os.environ.get("EIB_TASK", "")
if TASK not in ("rsv_f", "niv"):
    raise SystemExit("set EIB_TASK=rsv_f or EIB_TASK=niv. The HIV DMS arm has its "
                     "own script, scripts/dms_11_walle_experimental.py.")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W = _load("walle_infer", PROJECT / "scripts" / "08_walle_infer.py")
cfg = W.cfg
sys.path.insert(0, str(PROJECT / "config"))
import walle                                                      # noqa: E402
SCORES_ROOT = W.SCORES_ROOT
PAIR_CACHE = W.PAIR_CACHE
SOURCE = "experimental"


def _aa_residues(chain):
    """Amino-acid residues of a gemmi chain, insertion codes excluded.

    A residue with an insertion code has no integer position in either reference
    frame, so it could only be mapped by guessing. There are none in these eight
    structures; the filter is here so a re-download that gains one fails visibly
    in the identity check rather than shifting the frame.
    """
    import gemmi
    out = []
    for r in chain:
        t = gemmi.find_tabulated_residue(r.name)
        if t and t.is_amino_acid() and not r.seqid.icode.strip():
            out.append(r)
    return out


def _one(res):
    return W._THREE2ONE.get(res.name.upper(), "X")


# --------------------------------------------------------------------------- #
#  Arm specifications -- read from each arm's OWN declaration, never re-declared #
# --------------------------------------------------------------------------- #
def rsv_specs():
    """antibody -> spec, from rsv_05_gold_epitopes.CASES + rsv_gold_report.json."""
    R5 = _load("rsv05", PROJECT / "scripts" / "rsv_05_gold_epitopes.py")
    rep = json.loads(cfg.GOLD_REPORT.read_text())
    fseq = "".join(l.strip() for l in open(cfg.ANTIGEN_FASTA) if not l.startswith(">"))
    assert len(fseq) == cfg.ANTIGEN_LEN, (len(fseq), cfg.ANTIGEN_LEN)

    def to_ref(chain):
        """{deposited seqid -> 0-based index into the 504-long F frame}.

        Depositor numbering IS full-length F numbering in all three structures
        (re-checked below against F_protein.fasta), so this is an offset, not an
        alignment: site n -> index n - SITE_LO.
        """
        return {r.seqid.num: r.seqid.num - cfg.SITE_LO for r in _aa_residues(chain)
                if cfg.SITE_LO <= r.seqid.num <= cfg.SITE_HI}

    out = {}
    for ab, r in rep.items():
        if not r.get("has_gold"):
            out[ab] = dict(skip=r["leakage"], reason=r.get("reason", ""))
            continue
        case = R5.CASES[ab]
        out[ab] = dict(cif=cfg.REF_DIR / f"{case['pdb']}.cif", pdb=case["pdb"].upper(),
                       fab=tuple(r["fab_copy_used"]),
                       ag_declared=r["primary_protomer"].split("+"),
                       ag_candidates=[c for p in case["protomers"] for c in p],
                       to_ref=to_ref, ref_seq=fseq,
                       leakage=r["leakage"], cross_protomer_frac=r["cross_protomer_frac"])
    return out


def niv_specs():
    """antibody -> spec, from niv_01_escape_sites.COMPLEXES + niv_manifest.csv."""
    import pandas as pd
    N1 = _load("niv01", PROJECT / "scripts" / "niv_01_escape_sites.py")
    fasta = N1.antigen_sequence()
    assert len(fasta) == cfg.ANTIGEN_LEN, (len(fasta), cfg.ANTIGEN_LEN)
    man = pd.read_csv(cfg.MANIFEST)
    # chain roles are a property of the ANTIBODY, and the vh rows leave the light
    # chain blank, so both arms take them from the vhvl row.
    chains = {r.antibody: (str(r.heavy_chain), str(r.light_chain))
              for r in man[man.arm == "vhvl"].itertuples()}

    def make_to_ref(fa):
        def to_ref(chain):
            """{deposited seqid -> 0-based index into the 532-long RBP frame}.

            Via the arm's own `_map_to_nipah`, which returns a REFERENCE SITE in
            71-602; index = site - SITE_OFFSET - 1. Identity on the two Nipah
            structures, a 75-77% global alignment on the two Hendra ones.
            """
            res = _aa_residues(chain)
            nums = [r.seqid.num for r in res]
            letters = "".join(_one(r) for r in res)
            m, ident = N1._map_to_nipah(nums, letters, fa)
            return ({k: v - cfg.SITE_OFFSET - 1 for k, v in m.items()
                     if 1 <= v - cfg.SITE_OFFSET <= cfg.ANTIGEN_LEN}, ident)
        return to_ref

    out = {}
    for ab, spec in N1.COMPLEXES.items():
        h, l = chains[ab]
        out[ab] = dict(cif=cfg.REF_DIR / spec["cif"], pdb=spec["cif"][:4].upper(),
                       fab=(h, l), ag_declared=None,
                       ag_candidates=sorted(spec["ag"]),
                       to_ref=make_to_ref(fasta), ref_seq=fasta,
                       species=spec["species"])
    return out


SPECS = {"rsv_f": rsv_specs, "niv": niv_specs}[TASK]


# --------------------------------------------------------------------------- #
#  Build one experimental complex                                               #
# --------------------------------------------------------------------------- #
def fab_contacts(st, model, candidates, bh, bl, cutoff):
    """Heavy-atom contacts from the chosen Fab copy to each candidate antigen chain.

    Same construction as dms_11.fab_contacts: a NeighborSearch over the whole
    structure, walked from the Fab's atoms, counting hits per antigen chain.
    """
    import gemmi
    ns = gemmi.NeighborSearch(st, 5.0).populate()
    cnt = {c: 0 for c in candidates}
    present = {c.name for c in model}
    for cn in (bh, bl):
        if cn not in present:
            continue
        for res in _aa_residues(model[cn]):
            for atom in res:
                for mark in ns.find_atoms(atom.pos, "\0", radius=cutoff):
                    cra = mark.to_cra(model)
                    if cra.atom.pos.dist(atom.pos) > cutoff:
                        continue
                    if cra.chain.name in cnt:
                        cnt[cra.chain.name] += 1
    return cnt


def build_experimental(row, spec, wd):
    """Write <case>_exp.pdb with chains A/H/L; return (path, obs, to_ref, meta)."""
    import gemmi
    ab = row.antibody
    bh, bl = spec["fab"]
    st = gemmi.read_structure(str(spec["cif"]))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    model = st[0]
    present = {c.name for c in model}
    cands = [c for c in spec["ag_candidates"] if c in present]
    cnt = fab_contacts(st, model, cands, bh, bl, cfg.CONTACT_CUTOFF)

    # ---- per-chain reference maps (and, for Nipah, the alignment identity) ----
    maps, ident = {}, {}
    for cn in cands:
        m = spec["to_ref"](model[cn])
        if isinstance(m, tuple):
            m, ident[cn] = m
        maps[cn] = m

    # ---- which chains form the ONE protomer that is scored --------------------
    if spec["ag_declared"] is not None:
        # RSV: the protomer rsv_05_gold_epitopes established for this Fab copy.
        chosen = [c for c in spec["ag_declared"]]
        missing = [c for c in chosen if c not in present]
        if missing:
            raise ValueError(f"{ab}: declared protomer chain(s) {missing} absent "
                             f"from {spec['pdb']}")
        taken = set()
        for cn in chosen:                      # a declared protomer must not self-collide
            s = set(maps[cn].values())
            if s & taken:
                raise ValueError(f"{ab}: declared protomer chains {chosen} overlap in "
                                 f"the reference frame; refusing to score a chimera")
            taken |= s
    else:
        # Nipah: the single chain the Fab touches. Measured single-chain for all
        # four cases; the off-protomer fraction below is the guard if that changes.
        if not any(cnt.values()):
            raise ValueError(f"{ab}: Fab {bh}/{bl} contacts no candidate antigen chain")
        chosen = [max(cnt, key=lambda c: (cnt[c], c))]
    off = sum(v for k, v in cnt.items() if k not in chosen)
    frac_off = off / max(sum(cnt.values()), 1)

    # N-to-C along the reference, NOT in contact order. `seqres2atmseq` fails
    # outright on a chain whose numbering runs backwards -- the bug dms_11 hit on
    # PGT151 and VRC34, and the reason 6OUS must emit F2 (26-103) before F1 (137-).
    chosen = sorted(chosen, key=lambda c: min(maps[c].values()))

    # ---- antigen: the chosen protomer, into one chain A -----------------------
    out = gemmi.Structure()
    out.add_model(gemmi.Model("1"))
    ch = gemmi.Chain("A")
    to_ref, ag_res = [], []
    for cn in chosen:
        for res in _aa_residues(model[cn]):
            to_ref.append(maps[cn].get(res.seqid.num, -1))
            ag_res.append(res)
    for res in ag_res:
        ch.add_residue(res)
    out[0].add_chain(ch)
    obs = {"A": "".join(_one(r) for r in ag_res)}

    # ---- antibody chains, trimmed to the manifest's variable domain ------------
    want = [(bh, "H", row.vh_seq)]
    has_l = bool(row.has_light) and isinstance(row.light_seq, str) and row.light_seq
    if has_l:
        want.append((bl, "L", row.light_seq))
    for src_id, new_id, target in want:
        if src_id not in present:
            raise ValueError(f"{ab}: chain {src_id} absent from {spec['pdb']}")
        seq, res = W._chain_seq(model[src_id])
        res = W._trim_to_domain(res, seq, target)
        if len(res) < 60:
            raise ValueError(f"{ab}: chain {src_id} trimmed to {len(res)} residues")
        c = gemmi.Chain(new_id)
        for r in res:
            c.add_residue(r)
        out[0].add_chain(c)
        obs[new_id] = "".join(_one(r) for r in res)

    out.setup_entities()
    p = wd / f"{row.case_id}_exp.pdb"
    out.write_pdb(str(p))

    # ---- frame check: do the mapped residues actually match the reference? -----
    ref = spec["ref_seq"]
    to_ref = np.asarray(to_ref, int)
    mapped = to_ref[to_ref >= 0]
    idn = float(np.mean([obs["A"][i] == ref[t] for i, t in enumerate(to_ref) if t >= 0])) \
        if len(mapped) else 0.0
    # TWO DIFFERENT DIAGNOSTICS, which an earlier version of this script conflated
    # into one and mislabelled:
    #   n_reference_gaps_gt20 -- places where consecutive mapped residues jump more
    #     than 20 reference positions. These are ordinary and expected: unmodelled
    #     loops, and in 6OUS the excised p27 peptide between F2 (ends 103) and F1
    #     (starts 137). NOT an error.
    #   n_offset_outliers -- residues whose (deposited seqid - reference index)
    #     differs from the modal offset of this case. For an offset mapping (RSV)
    #     this is 0 by construction. For the ALIGNED Hendra chains it is exactly
    #     the end-gap artefact the module docstring describes: one residue, pulled
    #     onto the N-terminus of the Nipah frame. Counted, never silently dropped,
    #     because this arm's gold labels came through the same function.
    d = np.diff(mapped)
    gaps = int(np.sum(np.abs(d - 1) > 20)) if len(d) else 0
    offs = np.array([r.seqid.num - t for r, t in zip(ag_res, to_ref) if t >= 0], int)
    outliers = int(np.sum(offs != np.bincount(offs - offs.min()).argmax() + offs.min())) \
        if len(offs) else 0

    meta = dict(antibody=ab, pdb=spec["pdb"], fab_copy=[bh, bl],
                ag_chains_used=chosen, contacts={k: int(v) for k, v in cnt.items()},
                frac_contacts_off_protomer=round(frac_off, 4),
                n_ag_res=len(ag_res), n_mapped=int(len(mapped)),
                frac_reference=round(len(set(mapped.tolist())) / cfg.ANTIGEN_LEN, 4),
                identity_to_reference=round(idn, 3),
                n_offset_outliers=outliers, n_reference_gaps_gt20=gaps)
    for k in ("leakage", "cross_protomer_frac", "species"):
        if k in spec:
            meta[k] = spec[k]
    if ident:
        meta["alignment_identity_pct"] = {k: round(float(v), 1) for k, v in ident.items()}
    return p, obs, to_ref, meta


def build_pair(row, spec, wd, overwrite=False):
    """Graph + masks for one case. Identical to dms_11.build_pair's second half,
    which is itself stage 08's block -- same checks, same cache schema."""
    from asep.preprocess import create_pair_graph_data
    cid = row.case_id
    cache = PAIR_CACHE / SOURCE / f"{cid}.pt"
    if cache.exists() and not overwrite:
        return torch.load(cache)

    pdb, obs, to_ref, meta = build_experimental(row, spec, wd)
    ab_ids = ["H", "L"] if "L" in obs else ["H"]
    fasta = wd / f"{cid}.fasta"
    with open(fasta, "w") as fh:
        for k in ab_ids:
            fh.write(f">{cid}|{k}\n{obs[k]}\n")
        fh.write(f">{cid}|A\n{obs['A']}\n")

    W._SURF_CAPTURE.clear()
    pair = create_pair_graph_data(
        job_id=cid, ab_structure=pdb, ag_structure=pdb,
        ab_chain_id=ab_ids, ag_chain_id=["A"],
        ab_seqres=fasta, ag_seqres=fasta,
        esm2_ckpt_path=Path("/dev/null"), is_complex=False, log_level="ERROR")

    surf = W._SURF_CAPTURE[0].astype(np.uint8)
    cdr = W._SURF_CAPTURE[1].astype(np.uint8)
    if len(surf) != len(obs["A"]):
        raise ValueError(f"surf mask len {len(surf)} != antigen seqres {len(obs['A'])}")
    if int(surf.sum()) != pair.x_g.shape[0]:
        raise ValueError(f"surf sum {int(surf.sum())} != ag nodes {pair.x_g.shape[0]}")
    if int(cdr.sum()) != pair.x_b.shape[0]:
        raise ValueError(f"cdr sum {int(cdr.sum())} != ab nodes {pair.x_b.shape[0]}")

    blob = {"pair": pair, "surf": surf, "cdr": cdr,
            "ab_seq": "".join(obs[k] for k in ab_ids), "ag_seq": obs["A"],
            "to_ref": to_ref, "meta": meta}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache)
    return blob


# --------------------------------------------------------------------------- #
def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="run name under walle-inference/assets, or 'all'. The "
                         "default is the project checkpoint in config/walle.py.")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not getattr(cfg, "HAS_EXPERIMENTAL", False):
        raise SystemExit(f"config {TASK} has HAS_EXPERIMENTAL=False")

    want = a.checkpoint or walle.HEADLINE_CHECKPOINT

    W.install_patches()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = SPECS()
    man = pd.read_csv(cfg.MANIFEST)

    ckpts = sorted(d.name for d in W.ASSETS.iterdir() if d.is_dir()) \
        if want == "all" else [want]
    models = {c: W.load_model(c, device) for c in ckpts}
    print(f"EIB_TASK={TASK} | device {device} | {len(ckpts)} checkpoint(s) | "
          f"{len(man)} rows | antigen frame {cfg.ANTIGEN_LEN}")

    wd = Path(tempfile.mkdtemp())
    metas, failed, skipped = {}, [], []
    for row in man.itertuples():
        cid, ab = row.case_id, row.antibody
        spec = specs.get(ab)
        if spec is None or "skip" in spec:
            why = spec["skip"] if spec else "no entry in this arm's complex table"
            skipped.append((cid, why))
            print(f"  {cid}: NO EXPERIMENTAL RUN -- {why}")
            continue
        outs = {c: SCORES_ROOT / SOURCE / c / f"{cid}.npz" for c in ckpts}
        if not a.overwrite and all(o.exists() for o in outs.values()):
            continue
        try:
            blob = build_pair(row, spec, wd, a.overwrite)
        except Exception as e:
            failed.append((cid, f"preprocess: {e}"))
            print(f"  {cid}: preprocess FAIL -- {e}")
            continue
        m = blob.get("meta", {})
        metas[cid] = {**m, "n_surface_nodes": int(blob["surf"].sum()),
                      "n_cdr_nodes": int(blob["cdr"].sum())}
        print(f"  {cid}: {m.get('pdb')} fab {m.get('fab_copy')} antigen "
              f"{m.get('ag_chains_used')} {m.get('n_mapped')}/{m.get('n_ag_res')} mapped, "
              f"frac_ref={m.get('frac_reference')}, id={m.get('identity_to_reference')}, "
              f"off_protomer={m.get('frac_contacts_off_protomer')}, "
              f"surf={int(blob['surf'].sum())}, cdr={int(blob['cdr'].sum())}")
        for c in ckpts:
            if outs[c].exists() and not a.overwrite:
                continue
            try:
                model, nft = models[c]
                score, surf = W.score_case(blob, model, nft, device, cfg.ANTIGEN_LEN)
                if score.shape != (cfg.ANTIGEN_LEN,):
                    raise ValueError(f"score vector {score.shape}, expected "
                                     f"({cfg.ANTIGEN_LEN},)")
                outs[c].parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(outs[c], walle_score=score, surf_mask=surf,
                                    n_ag_nodes=int(surf.sum()),
                                    n_ab_nodes=int(blob["cdr"].sum()))
            except Exception as e:
                failed.append((f"{cid}/{c}", f"score: {e}"))
                print(f"  {cid}/{c}: score FAIL -- {e}")

    if metas:
        p = SCORES_ROOT / SOURCE / "build_report.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(metas)
        p.write_text(json.dumps(old, indent=2))
    print(f"\n-> {SCORES_ROOT / SOURCE}")
    if skipped:
        print(f"{len(skipped)} rows with no solved complex (expected, not a failure):")
        for k, why in skipped:
            print(f"  {k}: {why}")
    if failed:
        print(f"{len(failed)} failures:")
        for k, why in failed:
            print(f"  {k}: {why}")


if __name__ == "__main__":
    main()
