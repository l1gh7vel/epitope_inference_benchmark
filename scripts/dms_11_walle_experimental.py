#!/usr/bin/env python3
"""WALLE on the DMS panel from the DEPOSITED complexes, not from AF3 folds.

What this script does that `scripts/08_walle_infer.py --source experimental`
cannot
-----------------------------------------------------------------------------
Stage 08's experimental path was written for Tasks I/II, whose manifests carry
`pdb_id`, `heavy_chain_id_src`, `light_chain_id_src`, `antigen_chain_src`, and
whose antigen is a single deposited chain mapped onto the reference by pairwise
ALIGNMENT. The DMS manifest (`data/manifests/dms_manifest.csv`) has none of those
columns -- it carries `pdb` and nothing else -- and HIV Env structures break both
of stage 08's remaining assumptions:

  1. ONE DEPOSITED CHAIN IS NOT ONE PROTOMER. A SOSIP trimer deposits gp120 and
     gp41 of the same protomer as two chains (4TVP G+B, 6DCQ E+F, 5V8L D+B,
     6NC3 B+A, 6CCB G+A). AF3 folds the whole 856-aa gp160 protomer as one chain,
     so the matched experimental antigen is gp120+gp41 of ONE protomer. Taking
     the single best-contacted chain would have handed PGT151 only gp41
     (139 residues, 62% of its gold epitope) and VRC34 only gp41 (75% of gold).
  2. ALIGNMENT IS THE WRONG MAPPER HERE. HIV Env structures are deposited in
     HXB2 numbering, which IS the reference frame of `data/source/env.fa`
     (verified: `scripts/dms_00_gold_epitopes.py` uses exactly this, and the
     per-chain identity against env.fa is 0.74-0.89 for genuine Env chains).
     A global alignment would have to cope with a 93TH057 gp120 core, a ZM109
     V1V2 graft on a 1FD6 scaffold and a 28-residue MPER peptide, and the
     scaffold in particular would be mapped onto real HXB2 positions in C2 and
     fabricate antigen nodes. So: depositor numbering, exactly as dms_00 does,
     with `span` honoured for 3U2S.

Method choices made here
-----------------------------------------------------------------------------
  * Antigen = a greedy single-protomer assembly. Env chains are taken in
    descending order of contacts to the selected Fab copy and kept only while
    their reference positions do not collide with chains already taken. gp120
    and gp41 of one protomer are numbered disjointly, so they are both kept;
    a second protomer collides and is refused. This is deliberate: it matches
    the monomeric gp160 AF3 folds, at the cost of the cross-protomer part of
    an apex epitope (PGT145, 3BNC117). Both arms pay that cost.
  * Fab copy = `fab_copy_used` from `data/manifests/dms_gold_report.json`, i.e.
    the same copy the gold epitope was read from.
  * to_ref = depositor seqid - 1, restricted to [span] and [1, 856]. Residues
    with an insertion code, and the 1FD6 scaffold outside PG9's span, map to -1
    and are dropped -- they have no reference position.
  * Antibody chains are trimmed to the manifest's variable-domain sequence with
    stage 08's own `_trim_to_domain`, and WALLE is handed the OBSERVED sequences
    as SEQRES so its seqres->atmseq step is an identity map (stage 08's rule).

Everything downstream of the graph -- embeddings, node features, model, the
summed bipartite edge probability -- is stage 08's code, imported, not copied.

    EIB_ROOT=<worktree> python scripts/dms_11_walle_experimental.py --checkpoint all

Writes <EIB_ROOT>/work/dms/walle/scores/experimental/<checkpoint>/<case>.npz,
the same npz schema stage 08 writes, so `dms_08_binary_cost.py` and
`09_evaluate_walle.py` can read it by pointing at a different <source>.
"""
import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
os.environ.setdefault("EIB_TASK", "dms")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W = _load("walle_infer", PROJECT / "scripts" / "08_walle_infer.py")
G0 = _load("dms_gold", PROJECT / "scripts" / "dms_00_gold_epitopes.py")
cfg = W.cfg

SCORES_ROOT = W.SCORES_ROOT
PAIR_CACHE = W.PAIR_CACHE
SOURCE = "experimental"


def protomer_chains(model, cnt, span, L):
    """Env chains forming ONE protomer: greedy by contacts, no reference collision."""
    def refpos(cn):
        s = set()
        for res in model[cn]:
            if not G0.is_aa(res) or res.seqid.icode.strip():
                continue
            n = res.seqid.num
            if span[0] <= n <= span[1] and 1 <= n <= L:
                s.add(n)
        return s

    chosen, taken = [], set()
    for cn in sorted(cnt, key=lambda c: -cnt[c]):
        s = refpos(cn)
        if not s or (s & taken):
            continue
        chosen.append((min(s), cn))
        taken |= s
    # Emit N-to-C along gp160 (gp120 then gp41), NOT in contact order. chain A is
    # written as one polypeptide, and `seqres2atmseq` fails outright on a chain
    # whose residue numbering runs backwards -- which is what happened for PGT151
    # (6DCQ gp41 F before gp120 E) and VRC34 (6NC3 gp41 B before gp120 A), the two
    # cases whose Fab contacts gp41 more than gp120.
    return [cn for _, cn in sorted(chosen)]


def fab_contacts(st, model, ag, bh, bl, cutoff=4.5):
    """Heavy-atom contacts from the chosen Fab copy to each candidate Env chain."""
    ns = __import__("gemmi").NeighborSearch(st, 5.0).populate()
    cnt = {c: 0 for c in ag}
    present = {c.name for c in model}
    for cn in (bh, bl):
        if cn not in present:
            continue
        for res in model[cn]:
            if not G0.is_aa(res):
                continue
            for atom in res:
                for mark in ns.find_atoms(atom.pos, "\0", radius=cutoff):
                    cra = mark.to_cra(model)
                    if cra.atom.pos.dist(atom.pos) > cutoff:
                        continue
                    if cra.chain.name in cnt and G0.is_aa(cra.residue):
                        cnt[cra.chain.name] += 1
    return cnt


def build_experimental(row, spec, rep, env, wd):
    """Write <case>_exp.pdb with chains A/H/L; return (path, obs_seqs, to_ref, meta)."""
    import gemmi
    L = len(env)
    ab = row.antibody
    bh, bl = rep[ab]["fab_copy_used"]
    span = tuple(rep[ab]["span"])
    src = cfg.PDB_DIR / f"{spec['pdb']}.cif"
    st = gemmi.read_structure(str(src))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    model = st[0]
    present = {c.name for c in model}
    ag = [c for c in spec["ag"] if c in present]
    cnt = fab_contacts(st, model, ag, bh, bl)
    chosen = protomer_chains(model, cnt, span, L)
    if not chosen:
        raise ValueError(f"{ab}: no Env chain maps into the reference frame")

    out = gemmi.Structure()
    out.add_model(gemmi.Model("1"))

    # --- antigen: one protomer, possibly several deposited chains, into chain A
    ch = gemmi.Chain("A")
    to_ref, ag_res = [], []
    for cn in chosen:
        for res in model[cn]:
            if not G0.is_aa(res):
                continue
            n = res.seqid.num
            ok = (not res.seqid.icode.strip()) and span[0] <= n <= span[1] and 1 <= n <= L
            to_ref.append(n - 1 if ok else -1)
            ag_res.append(res)
    for res in ag_res:
        ch.add_residue(res)
    out[0].add_chain(ch)
    obs = {"A": "".join(W._THREE2ONE.get(r.name.upper(), "X") for r in ag_res)}

    # --- antibody chains, trimmed to the manifest's variable domain
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
        obs[new_id] = "".join(W._THREE2ONE.get(r.name.upper(), "X") for r in res)

    out.setup_entities()
    p = wd / f"{row.case_id}_exp.pdb"
    out.write_pdb(str(p))

    to_ref = np.asarray(to_ref, int)
    mapped = to_ref[to_ref >= 0]
    ident = float(np.mean([obs["A"][i] == env[t] for i, t in enumerate(to_ref) if t >= 0]))
    meta = dict(antibody=ab, pdb=spec["pdb"].upper(), fab_copy=[bh, bl],
                env_chains_used=chosen, contacts={k: int(v) for k, v in cnt.items()},
                span=list(span), n_ag_res=len(ag_res), n_mapped=int(len(mapped)),
                frac_reference=round(len(set(mapped.tolist())) / L, 4),
                identity_to_env_fa=round(ident, 3))
    return p, obs, to_ref, meta


def build_pair(row, spec, rep, env, wd, overwrite=False):
    from asep.preprocess import create_pair_graph_data
    cid = row.case_id
    cache = PAIR_CACHE / SOURCE / f"{cid}.pt"
    if cache.exists() and not overwrite:
        return torch.load(cache)

    pdb, obs, to_ref, meta = build_experimental(row, spec, rep, env, wd)
    has_l = "L" in obs
    ab_ids = ["H", "L"] if has_l else ["H"]
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
    ab_seq = "".join(obs[k] for k in ab_ids)
    if len(surf) != len(obs["A"]):
        raise ValueError(f"surf mask len {len(surf)} != antigen seqres {len(obs['A'])}")
    if int(surf.sum()) != pair.x_g.shape[0]:
        raise ValueError(f"surf sum {int(surf.sum())} != ag nodes {pair.x_g.shape[0]}")
    if int(cdr.sum()) != pair.x_b.shape[0]:
        raise ValueError(f"cdr sum {int(cdr.sum())} != ab nodes {pair.x_b.shape[0]}")

    blob = {"pair": pair, "surf": surf, "cdr": cdr, "ab_seq": ab_seq,
            "ag_seq": obs["A"], "to_ref": to_ref, "meta": meta}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache)
    return blob


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="decent-sweep-2",
                    help="run name under walle-inference/assets, or 'all'")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not getattr(cfg, "HAS_EXPERIMENTAL", False):
        raise SystemExit("config has HAS_EXPERIMENTAL=False")

    W.install_patches()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = "".join(l.strip() for l in open(cfg.ENV_FASTA) if not l.startswith(">"))
    assert len(env) == cfg.ANTIGEN_LEN, (len(env), cfg.ANTIGEN_LEN)
    rep = json.loads((cfg.MANIFESTS / "dms_gold_report.json").read_text())
    man = pd.read_csv(cfg.MANIFEST)

    ckpts = sorted(d.name for d in W.ASSETS.iterdir() if d.is_dir()) \
        if a.checkpoint == "all" else [a.checkpoint]
    models = {c: W.load_model(c, device) for c in ckpts}
    print(f"device {device} | {len(ckpts)} checkpoint(s) | {len(man)} cases")

    wd = Path(tempfile.mkdtemp())
    metas, failed = {}, []
    for row in man.itertuples():
        cid = row.case_id
        spec = G0.CASES[row.antibody]
        outs = {c: SCORES_ROOT / SOURCE / c / f"{cid}.npz" for c in ckpts}
        if not a.overwrite and all(o.exists() for o in outs.values()):
            continue
        try:
            blob = build_pair(row, spec, rep, env, wd, a.overwrite)
        except Exception as e:
            failed.append((cid, f"preprocess: {e}"))
            print(f"  {cid}: preprocess FAIL -- {e}")
            continue
        m = blob.get("meta", {})
        metas[cid] = {**m, "n_surface_nodes": int(blob["surf"].sum()),
                      "n_cdr_nodes": int(blob["cdr"].sum())}
        print(f"  {cid}: Env chains {m.get('env_chains_used')} "
              f"{m.get('n_mapped')}/{m.get('n_ag_res')} mapped, "
              f"frac_ref={m.get('frac_reference')}, id={m.get('identity_to_env_fa')}, "
              f"surf={int(blob['surf'].sum())}, cdr={int(blob['cdr'].sum())}")
        for c in ckpts:
            if outs[c].exists() and not a.overwrite:
                continue
            try:
                model, nft = models[c]
                score, surf = W.score_case(blob, model, nft, device, cfg.ANTIGEN_LEN)
                outs[c].parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(outs[c], walle_score=score, surf_mask=surf,
                                    n_ag_nodes=int(surf.sum()),
                                    n_ab_nodes=int(blob["cdr"].sum()))
            except Exception as e:
                failed.append((f"{cid}/{c}", f"score: {e}"))
                print(f"  {cid}/{c}: score FAIL -- {e}")

    if metas:
        p = SCORES_ROOT / SOURCE / "build_report.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(metas)
        p.write_text(json.dumps(old, indent=2))
    print(f"\n-> {SCORES_ROOT / SOURCE}")
    if failed:
        print(f"{len(failed)} failures:")
        for k, why in failed:
            print(f"  {k}: {why}")


if __name__ == "__main__":
    main()
