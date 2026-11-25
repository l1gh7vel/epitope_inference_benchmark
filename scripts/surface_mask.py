#!/usr/bin/env python3
"""
Antigen surface masks (SASA > 0) per case, from the AF3-predicted structure.

Surface = an antigen residue with any solvent-accessible area when the antigen
chain is considered ALONE (antibody removed) -- i.e. intrinsically exposed. This
matches WALLE's convention (RSA_THR = 0.0: any exposure => surface) and is
computed antigen-alone so that interface/epitope residues (which the antibody
would occlude in the complex) still count as surface.

Why: WALLE only scores surface residues, so the "surface-only" comparison needs
AF3 restricted to the same surface set. This produces that mask for AF3. Both
methods can then be evaluated surface-only, alongside the all-1273 convention
(both are computed so either can be reported).

Reuses the stage-04 structure helpers (freesasa, gemmi) so the SASA definition is
identical to the pipeline's BSA calculation.

    python scripts/surface_mask.py                 # all cases -> one npz
    python scripts/surface_mask.py --overwrite

Writes work/scores/af3_surface_masks.npz {case_id, surf_mask [N, 1273]}.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common import cfg

# Reuse stage-04's split_chains / heavy_atoms / sasa_per_atom so the surface
# definition matches the pipeline exactly (same freesasa radii, same chain split).
_spec = importlib.util.spec_from_file_location(
    "s04", Path(__file__).parent / "04_score_structures.py")
s04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s04)


def antigen_surface_mask(cif_path):
    """Return (surf_mask[uint8, N], per_res_sasa[float, N]) for the antigen alone."""
    ag_res, _ = s04.split_chains(cif_path)
    if len(ag_res) != cfg.ANTIGEN_LEN:
        raise ValueError(f"antigen has {len(ag_res)} residues, expected {cfg.ANTIGEN_LEN}")
    xyz, rad, idx = s04.heavy_atoms(ag_res)            # antigen ALONE (no antibody)
    atom_area = s04.sasa_per_atom(xyz, rad)
    per_res = np.zeros(len(ag_res))
    np.add.at(per_res, idx, atom_area)
    return (per_res > 0.0).astype(np.uint8), per_res


def _pick_cif(case_dir, case_id):
    """Prefer the top-ranked model; fall back to the first diffusion sample."""
    top = case_dir / f"{case_id}_model.cif"
    if top.exists():
        return top
    samples = sorted(case_dir.glob("seed-*_sample-*/*_model.cif"))
    return samples[0] if samples else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = cfg.SCORES / "af3_surface_masks.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} exists; use --overwrite to rebuild")
        return

    man = pd.read_csv(cfg.MANIFEST)
    cfg.SCORES.mkdir(parents=True, exist_ok=True)
    ids, masks, failed = [], [], []
    for _, r in man.iterrows():
        cif = _pick_cif(cfg.AF3_OUT / r.case_id, r.case_id)
        if cif is None:
            failed.append((r.case_id, "no structure"))
            continue
        try:
            m, _ = antigen_surface_mask(cif)
            ids.append(r.case_id)
            masks.append(m)
            print(f"  {r.case_id}: {int(m.sum()):4d}/{cfg.ANTIGEN_LEN} surface")
        except Exception as e:
            failed.append((r.case_id, str(e)))
            print(f"  {r.case_id}: FAIL {e}")

    np.savez_compressed(out, case_id=np.array(ids), surf_mask=np.stack(masks))
    frac = np.stack(masks).mean()
    print(f"\nwrote {out}: {len(ids)} cases, {len(failed)} failed")
    print(f"mean surface fraction: {frac:.3f} (~{int(frac*cfg.ANTIGEN_LEN)} of {cfg.ANTIGEN_LEN} residues)")
    if failed:
        for c, why in failed[:10]:
            print(f"  FAILED {c}: {why}")


if __name__ == "__main__":
    main()
