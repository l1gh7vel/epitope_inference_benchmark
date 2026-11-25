#!/usr/bin/env python3
"""
Stage 04 -- turn AF3 predicted complexes into per-residue epitope scores.

AF3 outputs structures, not epitope probabilities, so we read an epitope off the
predicted pose. Two readouts, both standard, both computed for every diffusion
sample so the ensemble gives a continuous score rather than a binary call:

  contact : 1 / (1 + d_min), where d_min is the shortest heavy-atom distance from
            an antigen residue to any antibody heavy atom. Monotone in distance,
            so it is rank-equivalent to -d_min (all AUPRC/rank metrics need) but
            bounded in (0, 1].
  BSA     : per-residue buried surface area, SASA(antigen alone) minus
            SASA(antigen in the complex), both from the same predicted
            coordinates. Weighted by how much surface the antibody actually
            occludes, unlike a distance cutoff that counts glancing contacts
            equally.

Ensembling over the 5 diffusion samples also gives contact_freq -- the fraction
of samples in which a residue is within CONTACT_CUTOFF. That is the closest thing
AF3 offers to a calibrated per-residue confidence.

Indexing is 1:1 and needs no alignment: chain A of the prediction IS the 1273-aa
P0DTC2 sequence we submitted, so antigen residue i maps to Spike position i.

    python scripts/04_score_structures.py              # all cases, resumable
    python scripts/04_score_structures.py --start 0 --end 50
"""

import argparse
import json

import freesasa
import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import cfg

# Element -> van der Waals radius (A). We feed freesasa raw coordinates rather
# than a PDB file, so radii are assigned by element; its own classifier expects
# standard PDB atom naming, which predicted mmCIFs do not always honour.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90, "P": 1.80}
DEFAULT_VDW = 1.70


def _is_aa(res):
    info = gemmi.find_tabulated_residue(res.name)
    return bool(info) and info.is_amino_acid()


def split_chains(cif_path):
    """
    Return (antigen_residues, antibody_residues) from a predicted complex.

    Chains are matched by the ids we assigned in the input JSON (A / H / L). The
    length-based fallback exists only so a renamed chain fails loudly on the
    length assertion downstream rather than silently scoring the wrong chain.
    """
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    by_name = {ch.name: [r for r in ch if _is_aa(r)] for ch in st[0]}
    by_name = {k: v for k, v in by_name.items() if v}

    ag = by_name.get(cfg.CHAIN_ANTIGEN)
    ab = [r for name in (cfg.CHAIN_HEAVY, cfg.CHAIN_LIGHT) for r in by_name.get(name, [])]
    if ag is None or not ab:
        # Fallback: antigen is the chain closest to the expected length.
        ordered = sorted(by_name.items(), key=lambda kv: abs(len(kv[1]) - cfg.ANTIGEN_LEN))
        ag = ordered[0][1]
        ab = [r for _, rs in ordered[1:] for r in rs]
    return ag, ab


def heavy_atoms(residues):
    """(coords[N,3], radii[N], residue_index[N]) for all non-hydrogen atoms."""
    xyz, rad, idx = [], [], []
    for i, res in enumerate(residues):
        for a in res:
            el = a.element.name.upper()
            if el in ("H", "D"):
                continue
            xyz.append((a.pos.x, a.pos.y, a.pos.z))
            rad.append(VDW.get(el, DEFAULT_VDW))
            idx.append(i)
    return np.asarray(xyz, float), np.asarray(rad, float), np.asarray(idx, int)


def min_distance_per_residue(ag_xyz, ag_idx, n_ag, ab_xyz):
    """Shortest antibody heavy-atom distance for each antigen residue."""
    d_atom, _ = cKDTree(ab_xyz).query(ag_xyz, k=1)
    d_res = np.full(n_ag, np.inf)
    np.minimum.at(d_res, ag_idx, d_atom)
    return d_res


def sasa_per_atom(xyz, radii):
    res = freesasa.calcCoord(xyz.ravel().tolist(), radii.tolist())
    return np.fromiter((res.atomArea(i) for i in range(res.nAtoms())),
                       dtype=float, count=res.nAtoms())


def bsa_per_residue(ag_xyz, ag_rad, ag_idx, n_ag, ab_xyz, ab_rad):
    """
    SASA(antigen alone) - SASA(antigen within the complex), summed per residue.

    Both states come from the same predicted coordinates -- the antigen is not
    relaxed after removing the antibody. For "how much surface does this pose
    bury", that is the correct comparison.
    """
    free = sasa_per_atom(ag_xyz, ag_rad)
    both = sasa_per_atom(np.vstack([ag_xyz, ab_xyz]),
                         np.concatenate([ag_rad, ab_rad]))[:len(ag_xyz)]
    per_res = np.zeros(n_ag)
    np.add.at(per_res, ag_idx, np.clip(free - both, 0.0, None))
    return per_res


def score_one_structure(cif_path, do_bsa=True):
    ag_res, ab_res = split_chains(cif_path)
    # Tasks II and III have ONE antigen, so a length that disagrees with the config
    # means the wrong chain was picked or the fold is truncated -- worth failing on.
    # Task I has a different antigen per row, so there is no task-level length to
    # check against (cfg.ANTIGEN_LEN is None) and the caller does the per-row check.
    if cfg.ANTIGEN_LEN is not None and len(ag_res) != cfg.ANTIGEN_LEN:
        raise ValueError(f"antigen has {len(ag_res)} residues, expected {cfg.ANTIGEN_LEN}")
    ag_xyz, ag_rad, ag_idx = heavy_atoms(ag_res)
    ab_xyz, ab_rad, _ = heavy_atoms(ab_res)
    if not len(ab_xyz):
        raise ValueError("no antibody heavy atoms")

    d = min_distance_per_residue(ag_xyz, ag_idx, len(ag_res), ab_xyz)
    bsa = bsa_per_residue(ag_xyz, ag_rad, ag_idx, len(ag_res), ab_xyz, ab_rad) \
        if do_bsa else np.zeros(len(ag_res))
    return d, bsa


def case_confidence(case_dir, case_id):
    """iPTM / pTM / ranking score, for filtering unreliable interfaces later."""
    out = {}
    summary = case_dir / f"{case_id}_summary_confidences.json"
    if summary.exists():
        with open(summary) as fh:
            s = json.load(fh)
        out["iptm"] = s.get("iptm")
        out["ptm"] = s.get("ptm")
        out["ranking_score"] = s.get("ranking_score")
        # Chain-pair iPTM isolates the antibody-antigen interface specifically,
        # which is the part we care about; whole-complex iPTM is diluted by the
        # (easy) intra-Fab interface.
        cp = s.get("chain_pair_iptm")
        if cp and len(cp) >= 2:
            out["iptm_ag_ab"] = float(np.max([cp[0][j] for j in range(1, len(cp))]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--no_bsa", action="store_true", help="Skip SASA (about 5x faster).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    end = len(man) if args.end < 0 else min(args.end, len(man))
    cfg.SCORES.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.start, end):
        r = man.iloc[i]
        case_dir = cfg.AF3_OUT / r.case_id
        out_npz = cfg.SCORES / f"{r.case_id}.npz"
        if out_npz.exists() and not args.overwrite:
            continue
        if not case_dir.is_dir():
            print(f"  {r.case_id}: no output directory, skipping")
            continue

        # Every diffusion sample, plus the top-ranked model AF3 promotes. AF3
        # names each sample <case>_seed-S_sample-N_model.cif inside its own dir.
        samples = sorted(case_dir.glob("seed-*_sample-*/*_model.cif"))
        top = case_dir / f"{r.case_id}_model.cif"
        if not samples and top.exists():
            samples = [top]

        ds, bsas, failed = [], [], []
        for cif in samples:
            try:
                d, b = score_one_structure(cif, do_bsa=not args.no_bsa)
                ds.append(d)
                bsas.append(b)
            except Exception as e:
                failed.append(f"{cif.parent.name}: {e}")
        if not ds:
            print(f"  {r.case_id}: all {len(samples)} samples failed -- {failed[:1]}")
            continue

        D = np.vstack(ds)                       # [n_samples, 1273]
        B = np.vstack(bsas)
        d_top, b_top = (score_one_structure(top) if top.exists() else (D[0], B[0]))

        np.savez_compressed(
            out_npz,
            # ensemble over diffusion samples
            contact_mean=(1.0 / (1.0 + D)).mean(0),
            contact_freq=(D <= cfg.CONTACT_CUTOFF).mean(0),
            dmin_mean=D.mean(0),
            bsa_mean=B.mean(0),
            # single top-ranked model, for comparability with 1-structure methods
            contact_top=1.0 / (1.0 + d_top),
            dmin_top=d_top,
            bsa_top=b_top,
            n_samples=len(ds),
        )

        conf = case_confidence(case_dir, r.case_id)
        rows.append({
            "case_id": r.case_id, "n_samples": len(ds), "n_failed": len(failed),
            "min_dmin": float(D.min()),
            "n_contact_top": int((d_top <= cfg.CONTACT_CUTOFF).sum()),
            "total_bsa_top": float(b_top.sum()),
            **conf,
        })
        print(f"  {r.case_id}: {len(ds)} samples | closest approach "
              f"{D.min():5.2f} A | {int((d_top <= cfg.CONTACT_CUTOFF).sum()):3d} contact "
              f"residues | iptm {conf.get('iptm', float('nan'))}")

    if rows:
        # One summary file per shard range, so concurrent array tasks never write
        # the same file. A full (unsharded) run keeps the plain name. The per-case
        # .npz files are the real output and are race-free (one file per case);
        # stage 05 reads those, not this summary. Merge shards later with:
        #   pandas.concat(map(read_csv, glob("scoring_summary_*.csv")))
        sharded = args.start != 0 or args.end >= 0
        name = f"scoring_summary_{args.start}-{end}.csv" if sharded else "scoring_summary.csv"
        pd.DataFrame(rows).to_csv(cfg.SCORES / name, index=False)
        print(f"\nscored {len(rows)} cases -> {cfg.SCORES / name}")
    else:
        print("\nnothing new to score")


if __name__ == "__main__":
    main()
