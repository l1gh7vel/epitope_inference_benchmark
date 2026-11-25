#!/usr/bin/env python3
"""
Stage 08 -- run WALLE (AsEP) on our complexes -> per-residue antigen scores.

Task-switchable via EIB_TASK (task2 = Spike / Task II, task3 = HIV Env,
task3cov = SARS-CoV-2 Spike). The antigen frame, manifest and output tree all come
from the selected config, so nothing below is Spike-specific.

Runs in the `walle` conda env, Docker-free. Native fair-esm / antiberty embeddings
replace WALLE's stock Docker embedding calls; these were validated numerically
identical to AsEP's own precomputed embeddings (max|diff| ~5e-6). WALLE's model,
weights, graph construction (4.5 A contacts, DSSP surface, ANARCI CDRs) and forward
pass are UNCHANGED. We only read its continuous per-antigen-node score -- the sum
of predicted bipartite edge probabilities, i.e. the quantity WALLE thresholds at
>3 -- and place it in the task's reference antigen frame, so it is graded by the
same harness as AF3 (scripts/05 for Task II, t3_05 for Task III).

Two outputs per case, so that both scoring conventions can be reported:
  walle_score : full ANTIGEN_LEN vector, buried residues = 0 (comparable to AF3 `full`)
  surf_mask   : ANTIGEN_LEN uint8, the surface residues WALLE actually scored
                (surface-only convention; also lets AF3 be restricted to the set)

Embeddings + graph are the expensive part and do NOT depend on the checkpoint, so
each case's PairData is cached once and reused across checkpoints.

    python scripts/08_walle_infer.py --checkpoint whole-sweep-72 --source af3
    python scripts/08_walle_infer.py --checkpoint whole-sweep-72 --source af3 --start 0 --end 50
    python scripts/08_walle_infer.py --checkpoint all --source af3          # every checkpoint
    EIB_TASK=task3cov python scripts/08_walle_infer.py --checkpoint all     # Task III CoV

Writes <cfg.WORK>/walle/scores/<source>/<checkpoint>/<case>.npz

NB WALLE is POSE-INDEPENDENT: asep reads the antibody and the antigen as two
separate structures and encodes each with only intra-entity edges
(`encode()` sees x_b/edge_index_b and x_g/edge_index_g), then decodes all pairs.
No inter-chain geometry enters the forward pass, and DSSP runs on the antigen
alone, so the antibody cannot occlude its surface. Consequently AF3's iPTM
(interface-pose confidence) is irrelevant here -- what matters is monomer fold
quality. Do not reason about WALLE results in terms of docking accuracy.
"""
import argparse
import importlib.util
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# ---- config (import the plain config module; avoid scripts/common heavy deps) ----
# Task-switchable exactly like scripts/common.py, but loaded by path so this script
# keeps working inside the `walle` conda env, which does not have common.py's deps.
#     EIB_TASK=task2 (default) | task3 | task3cov
_PROJECT = Path(__file__).resolve().parent.parent
_TASK = os.environ.get("EIB_TASK", "task2")
_spec = importlib.util.spec_from_file_location(f"{_TASK}cfg", _PROJECT / "config" / f"{_TASK}.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

# The downloaded checkpoints are TASK-INDEPENDENT (one copy of WALLE's released
# weights), so they live in the shared work/walle tree. Only the derived per-case
# artefacts -- the PairData cache and the score npz files -- are per task.
WALLE_SHARED = _PROJECT / "work" / "walle"
ASSETS       = WALLE_SHARED / "walle-inference" / "assets"
WALLE_DIR    = cfg.WORK / "walle"
SCORES_ROOT  = WALLE_DIR / "scores"
PAIR_CACHE   = WALLE_DIR / "pair_cache"

# --------------------------------------------------------------------------- #
#  Native embeddings (replace the Docker calls) -- validated identical to AsEP  #
# --------------------------------------------------------------------------- #
_ESM = {"model": None, "bc": None}
_AB = {"runner": None}


def _esm():
    if _ESM["model"] is None:
        import esm
        m, alph = esm.pretrained.esm2_t12_35M_UR50D()
        m.eval()
        _ESM["model"], _ESM["bc"] = m, alph.get_batch_converter()
    return _ESM["model"], _ESM["bc"]


def _antiberty():
    if _AB["runner"] is None:
        from antiberty import AntiBERTyRunner
        _AB["runner"] = AntiBERTyRunner()
    return _AB["runner"]


def native_esm2(*, antigen_name, antigen_seqres, model_name=None, esm2_ckpt_host_path=None):
    """ESM2 esm2_t12_35M_UR50D layer-12 per-token embedding (BOS/EOS stripped) -> (L,480)."""
    seq = list(antigen_seqres.values())[0]
    model, bc = _esm()
    _, _, toks = bc([(antigen_name, seq)])
    with torch.no_grad():
        rep = model(toks, repr_layers=[12])["representations"][12]
    return rep[0, 1:len(seq) + 1], {}


def native_igfold(*, antibody_name, antibody_seqres):
    """AntiBERTy per-residue embedding (CLS/SEP stripped), chains concatenated -> (L,512)."""
    embs = _antiberty().embed(list(antibody_seqres.values()))
    ab = torch.cat([e[1:-1] for e in embs], dim=0)
    return {"tensors": {"bert_embs": ab}}, {}


# 3-letter -> 1-letter (newer biopython dropped Polypeptide.three_to_one)
_THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
              "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
              "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
              "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O"}

# captures the antigen seqres->surface mask (first downstream-mask call in main())
_SURF_CAPTURE = []


def install_patches():
    """Monkeypatch asep at runtime -- no edits to WALLE's source files."""
    import asep.preprocess as pp
    pp.run_esm2_docker_container = native_esm2
    pp.run_igfold_docker_container = native_igfold

    import Bio.PDB.Polypeptide as _PP
    if not hasattr(_PP, "three_to_one"):
        _PP.three_to_one = lambda x: _THREE2ONE.get(str(x).upper(), "X")

    # Wrap the seqres->downstream-mask mapper to capture its first result per case
    # (main() calls it for the ANTIGEN surface mask before the antibody CDR mask),
    # so we can scatter surface-node scores back to the full antigen frame without
    # editing WALLE's code.
    orig = pp.map_seqres_to_atmseq_with_downstream_mask

    def wrapped(*a, **k):
        r = orig(*a, **k)
        _SURF_CAPTURE.append(np.asarray(r))
        return r

    pp.map_seqres_to_atmseq_with_downstream_mask = wrapped

    _patch_graphein_unk()


def _patch_graphein_unk():
    """Fix a graphein bug that kills any structure containing a modified residue.

    graphein's `add_dssp_df` replaces DSSP's "UNK" rows with the real residue name
    from the coordinates:

        dssp_dict.loc[dssp_dict["aa"] == "UNK", "aa"] = (
            G.graph["pdb_df"].loc[
                G.graph["pdb_df"].residue_number.isin(<the UNK resnums>)
            ]["residue_name"].values)

    `pdb_df` is PER ATOM and spans every chain, so the right-hand side has one entry
    per matching ATOM in the whole structure while the left-hand side has one per UNK
    RESIDUE in this chain. The two almost never have the same length, and pandas
    raises "Must have equal len keys and value when setting with an iterable" --
    killing the case outright. In Task I this is mostly selenomethionine,
    which is routine in X-ray structures.

    The patch replaces that one assignment with a per-residue lookup built from the
    same `pdb_df`, which is what the original line was trying to express. It is a
    strict superset of the old behaviour: with no UNK rows the block is a no-op
    either way, so no case that already succeeds can change.
    """
    from graphein.protein.features.nodes import dssp as _gd

    if getattr(_gd, "_abc_unk_patched", False):
        return
    _orig_add = _gd.add_dssp_df

    def add_dssp_df(G, dssp_config, *a, **k):
        """Resolve UNK rows here, so the original's own repair has nothing to do."""
        orig_parse = _gd.parse_dssp_df

        def parse_dssp_df(d):
            out = orig_parse(d)
            pdb_df = G.graph.get("pdb_df")
            if "aa" not in out.columns or pdb_df is None:
                return out
            # EXACTLY the rows the original tries to repair. graphein maps the
            # 1-letter code to 3-letter and then looks for "UNK"; only "X" maps to
            # "UNK". DSSP's lowercase disulfide cysteines map to NaN instead, so the
            # original never touches them and neither do we -- which is what keeps
            # this a strict superset rather than a behaviour change.
            unk = out["aa"] == "X"
            if not unk.any():
                return out
            # one row per (chain, residue_number) -- the lookup the original meant,
            # instead of an isin() over every ATOM of every chain
            lut = (pdb_df.drop_duplicates(subset=["chain_id", "residue_number"])
                         .set_index(["chain_id", "residue_number"])["residue_name"])
            keys = list(zip(out.loc[unk, "chain"], out.loc[unk, "resnum"]))
            resolved = [_THREE2ONE.get(str(lut[kk]).upper(), "X") if kk in lut.index
                        else None for kk in keys]
            # A DSSP row graphein's own pdb_df does not contain is a residue graphein
            # EXCLUDED from the graph (nonstandard, or lacking backbone atoms). It
            # therefore corresponds to no node, contributes RSA to nothing, and is
            # dropped -- which is a no-op for every node that does exist. Leaving it
            # in is precisely what breaks the original. These are what actually
            # occurs in practice; genuine modified residues that graphein DOES keep
            # are resolved by name above.
            out.loc[unk, "aa"] = [r if r is not None else "?" for r in resolved]
            return out[out["aa"] != "?"].reset_index(drop=True)

        _gd.parse_dssp_df = parse_dssp_df
        try:
            return _orig_add(G, dssp_config, *a, **k)
        finally:
            _gd.parse_dssp_df = orig_parse

    _gd.add_dssp_df = add_dssp_df
    _gd._abc_unk_patched = True
    # graphein.protein.features.nodes.dssp.rsa() resolves add_dssp_df from the
    # module globals at call time, so rebinding the attribute above is enough.


# --------------------------------------------------------------------------- #
#  Inputs                                                                       #
# --------------------------------------------------------------------------- #
def _read_antigen_ref():
    """The task's single reference antigen, or None where each row carries its own.

    Tasks II and III hold the antigen fixed, so there is one frame for the whole
    run. Task I sweeps all of SAbDab -- `cfg.ANTIGEN_LEN is None` marks that, and
    main() then takes the frame from each row's own `antigen_sequence`."""
    if getattr(cfg, "ANTIGEN_LEN", None) is None:
        return None
    txt = open(cfg.ANTIGEN_FASTA).read().split("\n", 1)[1]
    return txt.replace("\n", "").strip()


def _af3_structure(case_id):
    """Top-ranked AF3 model for a case (fall back to first diffusion sample)."""
    d = cfg.AF3_OUT / case_id
    top = d / f"{case_id}_model.cif"
    if top.exists():
        return top
    s = sorted(d.glob("seed-*_sample-*/*_model.cif"))
    return s[0] if s else None


PDB_DIR = WALLE_DIR / "pdb"
_AA3 = {v: k for k, v in _THREE2ONE.items()}


def _chain_seq(chain):
    """(one-letter sequence, residue list) for the amino-acid residues of a chain."""
    import gemmi
    res = [r for r in chain if gemmi.find_tabulated_residue(r.name)
           and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
    return "".join(_THREE2ONE.get(r.name.upper(), "X") for r in res), res


def _trim_to_domain(res, seq, target):
    """Keep only the residues matching our variable-domain sequence.

    Deposited antibody chains are often full Fabs, but AsEP/WALLE (and our AF3 run)
    use the variable domain, and AntiBERTy expects one. The vh/light sequences in the input tables
    were derived from these structures, so an exact substring usually hits; fall
    back to a local alignment when the structure has gaps or mutations.
    """
    i = seq.find(target)
    if i >= 0:
        return res[i:i + len(target)]
    from Bio import Align
    al = Align.PairwiseAligner(mode="local", open_gap_score=-10, extend_gap_score=-0.5,
                               mismatch_score=-1, match_score=2)
    a = al.align(seq, target)[0]
    s, e = int(a.aligned[0][0][0]), int(a.aligned[0][-1][1])
    return res[s:e]


def _align_to_reference(obs, ref):
    """Map each observed antigen residue onto its reference-Spike index (-1 if none).

    A global alignment, so variant substitutions still map (they align as
    mismatches) while insertions/deletions are handled as gaps."""
    from Bio import Align
    al = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-0.5,
                               open_gap_score=-10, extend_gap_score=-0.5,
                               target_end_gap_score=0.0, query_end_gap_score=0.0)
    a = al.align(obs, ref)[0]
    out = np.full(len(obs), -1, int)
    for (os_, oe), (rs, re_) in zip(a.aligned[0], a.aligned[1]):
        out[os_:oe] = np.arange(rs, re_)
    return out


def _experimental_pdb(row, cid, wd):
    """Write a PDB holding just antigen/heavy/light as chains A/H/L.

    Returns (pdb_path, observed_antigen_sequence)."""
    import gemmi
    stem = str(row.pdb_id).lower()
    # Task I stores gzipped (many are cryo-EM spike, tens of MB each);
    # Tasks II/III store plain .cif. gemmi reads either.
    src = next((PDB_DIR / f"{stem}{e}" for e in (".cif", ".cif.gz")
                if (PDB_DIR / f"{stem}{e}").exists()), None)
    if src is None:
        raise FileNotFoundError(f"no downloaded structure for {stem}")
    st = gemmi.read_structure(str(src))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_waters()
    model = st[0]

    # Task II/III manifests record the DEPOSITED chain ids under *_src (their own
    # chain labels are already remapped to A/H/L); Task I's manifest carries the
    # deposited ids directly. Same quantity, different column name.
    def _chain(*names):
        for n in names:
            v = getattr(row, n, None)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return str(v)
        raise ValueError(f"no chain id column among {names}")

    want = [(_chain("antigen_chain_src", "antigen_chain"), "A", None),
            (_chain("heavy_chain_id_src", "heavy_chain_id"), "H", row.vh_seq)]
    # Task I is VH-only by construction (its source data has no light chain), so
    # the deposited light chain is deliberately NOT pulled in: including it would
    # give WALLE an input the other Task I methods were never given.
    if getattr(row, "has_light", False) and isinstance(getattr(row, "light_seq", None), str) \
            and row.light_seq:
        want.append((_chain("light_chain_id_src", "light_chain_id"), "L", row.light_seq))

    out = gemmi.Structure()
    out.add_model(gemmi.Model("1"))
    obs = {}
    for src_id, new_id, target in want:
        if src_id not in [c.name for c in model]:
            raise ValueError(f"chain {src_id} not in {src.name}")
        seq, res = _chain_seq(model[src_id])
        if not res:
            raise ValueError(f"chain {src_id} has no amino acids")
        if target:
            res = _trim_to_domain(res, seq, target)
            if len(res) < 60:
                raise ValueError(f"chain {src_id} trimmed to {len(res)} residues")
        ch = gemmi.Chain(new_id)
        for r in res:
            ch.add_residue(r)
        out[0].add_chain(ch)
        # SEQRES we hand WALLE is the OBSERVED sequence, so its seqres->atmseq step
        # is an identity map and cannot fail on variants or unmodelled loops.
        obs[new_id] = "".join(_THREE2ONE.get(r.name.upper(), "X") for r in res)
    out.setup_entities()
    p = wd / f"{cid}_exp.pdb"
    out.write_pdb(str(p))
    return p, obs


def build_pair_data(row, source, ag_ref, wd):
    """Build one case's graph (cached). Returns a blob with the PairData, the
    antigen surface mask (ANTIGEN_LEN) and antibody CDR mask, plus the node sequences,
    so node features for ANY checkpoint type (PLM / one-hot / BLOSUM) can be
    reconstructed at score time."""
    import gemmi
    from asep.preprocess import create_pair_graph_data

    cid = row.case_id
    cache = PAIR_CACHE / source / f"{cid}.pt"
    if cache.exists():
        return torch.load(cache)

    ag_pdb = None                         # None => antigen comes from the same file
    if source == "af3":
        cif = _af3_structure(cid)
        if cif is None:
            raise FileNotFoundError("no AF3 structure")
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        pdb = wd / f"{cid}.pdb"
        st.write_pdb(str(pdb))
        ag_seq = ag_ref                   # AF3 antigen chain A IS the reference antigen
        to_ref = np.arange(cfg.ANTIGEN_LEN)  # already in frame
    else:
        # Experimental deposited structure: pull the three relevant chains out of
        # the RCSB mmCIF and relabel them A/H/L.
        #
        # The antigen is a CONSTRUCT (often RBD-only) and frequently a VARIANT
        # (Omicron etc.), so it is NOT a substring of our reference Spike. Passing
        # the reference as SEQRES makes WALLE's seqres2atmseq step fail outright
        # for some cases. WALLE therefore receives the observed sequence, and its
        # internal alignment is trivial -- and do the construct -> reference mapping
        # ourselves by aligning to the reference, which tolerates the variant
        # substitutions. `to_ref` carries that mapping downstream.
        pdb, obs = _experimental_pdb(row, cid, wd)
        ag_seq = obs["A"]
        to_ref = _align_to_reference(ag_seq, ag_ref)

    # antibody chains present. Some Task II rows are VH-only; Task I is VH-only
    # throughout and its manifest has no light-chain columns at all.
    _has_l = bool(getattr(row, "has_light", False)) and \
        isinstance(getattr(row, "light_seq", None), str) and bool(getattr(row, "light_seq", ""))
    ab_ids = ["H", "L"] if _has_l else ["H"]
    # For experimental structures use the OBSERVED antibody sequences too, for the
    # same reason as the antigen (deposited chains carry variants/unmodelled gaps).
    ab_seqs = ({k: obs[k] for k in ab_ids} if source == "experimental"
               else {"H": row.vh_seq, **({"L": row.light_seq} if "L" in ab_ids else {})})
    fasta = wd / f"{cid}.fasta"
    with open(fasta, "w") as fh:
        for k in ab_ids:
            fh.write(f">{cid}|{k}\n{ab_seqs[k]}\n")
        fh.write(f">{cid}|A\n{ag_seq}\n")

    _SURF_CAPTURE.clear()
    pair = create_pair_graph_data(
        job_id=cid, ab_structure=pdb, ag_structure=(ag_pdb or pdb),
        ab_chain_id=ab_ids, ag_chain_id=["A"],
        ab_seqres=fasta, ag_seqres=fasta,
        esm2_ckpt_path=Path("/dev/null"), is_complex=False, log_level="ERROR")

    # main() calls the downstream-mask mapper twice, in order: [0] antigen surface,
    # [1] antibody CDR.
    surf = _SURF_CAPTURE[0].astype(np.uint8)          # over the antigen SEQRES given
    cdr = _SURF_CAPTURE[1].astype(np.uint8)
    ab_seq = "".join(ab_seqs[k] for k in ab_ids)      # H then L (asep concat order)
    if len(surf) != len(ag_seq):
        raise ValueError(f"surf mask len {len(surf)} != antigen seqres {len(ag_seq)}")
    if int(surf.sum()) != pair.x_g.shape[0]:
        raise ValueError(f"surf sum {int(surf.sum())} != ag nodes {pair.x_g.shape[0]}")
    if int(cdr.sum()) != pair.x_b.shape[0]:
        raise ValueError(f"cdr sum {int(cdr.sum())} != ab nodes {pair.x_b.shape[0]}")

    blob = {"pair": pair, "surf": surf, "cdr": cdr, "ab_seq": ab_seq, "ag_seq": ag_seq,
            "to_ref": np.asarray(to_ref, int)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, cache)
    return blob


# --------------------------------------------------------------------------- #
#  Model + scoring                                                              #
# --------------------------------------------------------------------------- #
def load_model(ckpt_name, device):
    from omegaconf import OmegaConf
    from asep.train_model import create_model
    cdir = ASSETS / ckpt_name
    conf = OmegaConf.load(cdir / "config.yaml")
    model = create_model(conf)
    sd = torch.load(cdir / "rank_0.pt", map_location="cpu")
    model.load_state_dict(sd["model_state_dict"])
    model.to(device).eval()
    # node feature type this checkpoint was trained with
    nft = conf["dataset"]["node_feat_type"]        # 'pre_cal' | 'one_hot' | 'custom'
    if nft == "custom":
        # 'custom' is not a synonym for BLOSUM62. AsEP's config lets a checkpoint
        # name ANY embedding function, so hardcoding embed_blosum62 here would
        # silently encode a future checkpoint's nodes with the wrong matrix and
        # still produce a finite score. Resolve the method the checkpoint actually
        # names, and let _node_features refuse anything it has no encoder for.
        meth = _custom_method_name(conf)
        if meth is None:
            raise ValueError(
                f"checkpoint {ckpt_name}: node_feat_type='custom' but its config "
                f"names no custom_embedding_method_src.method_name, so there is no "
                f"way to know which embedding to use. Refusing to guess BLOSUM62.")
        nft = f"custom:{meth}"
    return model, nft


def _custom_method_name(conf):
    """The embedding function a 'custom' checkpoint names, or None.

    AsEP stores it per entity (antibody and antigen each have their own block).
    Both must agree -- a checkpoint that encoded the two sides with different
    matrices is not something this script can reproduce.
    """
    ds = conf["dataset"]
    names = set()
    for key in ("ab", "ag"):
        src = (ds.get(key) or {}).get("custom_embedding_method_src") or {}
        if src.get("method_name"):
            names.add(str(src["method_name"]))
    if len(names) > 1:
        raise ValueError(f"checkpoint encodes antibody and antigen with different "
                         f"custom embeddings {sorted(names)}; not reproducible here")
    return names.pop() if names else None


def _node_features(blob, nft):
    """Rebuild (x_b, x_g) for a checkpoint's node_feat_type, using asep's EXACT
    encoders so features aren't scrambled. PLM ('pre_cal') uses the cached
    embeddings; one-hot / BLOSUM ('custom') encode the full sequence then slice by
    the CDR / surface mask -- identical to asep.data.asepv1_dataset."""
    if nft == "pre_cal":
        return blob["pair"].x_b, blob["pair"].x_g
    cdr = blob["cdr"].astype(bool)
    surf = blob["surf"].astype(bool)
    if nft == "one_hot":
        from asep.data.asepv1_dataset import aa_seq_to_one_hot_tensor as enc
    elif nft == "custom:embed_blosum62":           # decent-sweep-2
        from asep.data.embedding.blosum62 import embed_blosum62 as enc
    elif nft.startswith("custom:"):
        raise ValueError(
            f"checkpoint uses custom embedding {nft.split(':', 1)[1]!r}, for which "
            f"this script has no encoder. Add it here rather than falling back to "
            f"BLOSUM62 -- the wrong matrix produces plausible but wrong scores.")
    else:
        raise ValueError(f"unknown node_feat_type {nft}")
    x_b = enc(blob["ab_seq"])[cdr].float()
    x_g = enc(blob["ag_seq"])[surf].float()
    return x_b, x_g


def score_case(blob, model, nft, device, n_ref=None):
    """WALLE continuous per-antigen score, scattered onto the reference frame."""
    from asep.preprocess import pyg_data_to_batch
    x_b, x_g = _node_features(blob, nft)           # reads original PLM for pre_cal
    pair = blob["pair"].clone()                    # clone so we never mutate the cache
    pair.x_b, pair.x_g = x_b, x_g                  # swap in the right node features
    batch = pyg_data_to_batch(pair).to(device)
    with torch.no_grad():
        pred = model(batch)
    p = pred["edge_index_bg_pred"][0]              # (n_ab, n_ag)
    n_ag = pair.x_g.shape[0]
    # Sum over the antibody axis. The assertion fixes the expected orientation
    # when the antibody and antigen node counts happen to be equal.
    assert p.shape == (pair.x_b.shape[0], n_ag), p.shape
    node_score = p.sum(axis=0).detach().cpu().numpy().astype(np.float32)
    # Node i corresponds to the i-th surface position of the antigen SEQRES we gave
    # WALLE; to_ref maps that onto our reference antigen frame (identity for AF3
    # structures, an alignment for experimental constructs/variants; -1 = no match).
    local = np.flatnonzero(blob["surf"] == 1)
    # Read the task-generic key and the compatible spike-specific cache key.
    tgt = blob["to_ref" if "to_ref" in blob else "to_spike"][local]
    ok = tgt >= 0
    n_ref = cfg.ANTIGEN_LEN if n_ref is None else n_ref
    full = np.zeros(n_ref, np.float32)
    full[tgt[ok]] = node_score[ok]
    surf_ref = np.zeros(n_ref, np.uint8)
    surf_ref[tgt[ok]] = 1
    return full, surf_ref


# --------------------------------------------------------------------------- #
def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="whole-sweep-72",
                    help="run name under walle-inference/assets, or 'all'")
    ap.add_argument("--source", default="af3",
                    choices=["af3", "experimental"],
                    help="af3 = both chains from the AF3 prediction; experimental = both "
                         "from the deposited complex (Tasks I and II)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--headline_only", action="store_true",
                    help="only rows with valid antibody chain roles")
    ap.add_argument("--usable_only", action="store_true",
                    help="Task I: restrict to exactly the rows t1_05_evaluate "
                         "will grade: usable and retained by the heavy-chain deduplication")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Task III antibodies have no solved complex -- that is the premise of the task
    # (CATNAP / CoV-AbDab give a bin, not a structure). Fail loudly rather than
    # producing an empty run.
    if args.source == "experimental" and not getattr(cfg, "HAS_EXPERIMENTAL", False):
        raise SystemExit(
            f"--source experimental is not available for EIB_TASK={_TASK}: these "
            f"antibodies have no solved complexes. Use --source af3.")

    install_patches()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ag_ref = _read_antigen_ref()

    man = pd.read_csv(cfg.MANIFEST)
    if args.headline_only:
        man = man[man.chain_role_ok].reset_index(drop=True)
    if args.usable_only:
        keep = man.usable.astype(bool)
        if "vh_dedup_keep" in man:
            keep &= man.vh_dedup_keep.astype(bool)
        man = man[keep].reset_index(drop=True)
    end = len(man) if args.end < 0 else min(args.end, len(man))

    ckpts = sorted(d.name for d in ASSETS.iterdir() if d.is_dir()) \
        if args.checkpoint == "all" else [args.checkpoint]
    models = {c: load_model(c, device) for c in ckpts}   # c -> (model, node_feat_type)
    print(f"device {device} | {len(ckpts)} checkpoint(s) | cases [{args.start}:{end}] of {len(man)}")

    wd = Path(tempfile.mkdtemp())
    done, failed = 0, []
    for i in range(args.start, end):
        row = man.iloc[i]
        cid = row.case_id
        # all requested checkpoints already scored?
        outs = {c: SCORES_ROOT / args.source / c / f"{cid}.npz" for c in ckpts}
        if not args.overwrite and all(o.exists() for o in outs.values()):
            continue
        # One antigen for the whole task (II/III), or this row's own (Task I).
        row_ref = row.antigen_sequence if ag_ref is None else ag_ref
        try:
            blob = build_pair_data(row, args.source, row_ref, wd)
        except Exception as e:
            failed.append((cid, f"preprocess: {e}"))
            print(f"  {cid}: preprocess FAIL -- {e}")
            continue
        for c in ckpts:
            if outs[c].exists() and not args.overwrite:
                continue
            try:
                model, nft = models[c]
                score, surf = score_case(blob, model, nft, device, len(row_ref))
                outs[c].parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(outs[c], walle_score=score, surf_mask=surf,
                                    n_ag_nodes=int(surf.sum()), n_ab_nodes=int(blob["cdr"].sum()))
            except Exception as e:
                failed.append((f"{cid}/{c}", f"score: {e}"))
                print(f"  {cid}/{c}: score FAIL -- {e}")
        done += 1
        if done % 10 == 0:
            print(f"  ...{done} cases done")

    print(f"\nscored {done} cases x {len(ckpts)} ckpts -> {SCORES_ROOT/args.source}")
    if failed:
        print(f"{len(failed)} failures:")
        for k, why in failed[:15]:
            print(f"  {k}: {why}")


if __name__ == "__main__":
    main()
