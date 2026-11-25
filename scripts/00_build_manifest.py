#!/usr/bin/env python3
"""
Stage 00 -- build and validate the Task II case manifest.

Turns rbm_nonrbm_fold_input.csv into one row per AF3 case, with the exact
sequences that will be folded and the cache keys of the chains they need. Also
extracts the gold epitope labels and the structure-observed mask into a compact
.npz so the manifest itself stays readable.

Nothing downstream re-reads the source CSV: the manifest is the contract.

    python scripts/00_build_manifest.py
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import cfg, chain_key, clean_seq, domain_kind, read_fasta_single, trim_light_to_variable


def epitope_class(epitope, frac_in_rbm, rbm_lo, rbm_hi, rbd_lo, rbd_hi):
    """Which epitope class one merged gold epitope belongs to.

    See merge_duplicate_inputs for what the four values mean and why the
    non-motif side is restricted to the receptor-binding domain.
    """
    if 0.05 < frac_in_rbm < 0.95:
        return "ambiguous"
    if frac_in_rbm >= 0.95:
        return "RBM"
    pos = np.flatnonzero(epitope) + 1                       # 1-based antigen numbering
    inside = bool(((pos >= rbd_lo) & (pos <= rbd_hi)).all())
    return "nonRBM" if inside else "nonRBM_outside_RBD"


def merge_duplicate_inputs(man, labels, obs_masks):
    """Collapse rows whose (VH, VL) sequences are EXACTLY equal into one case.

    The model sees only VH, VL and antigen sequence. Rows sharing those inputs
    define one prediction. Collapsing them prevents repeated antibodies from
    receiving extra weight and prevents one prediction from being graded against
    several separate labels.

    Gold epitopes and observed-residue masks are unioned across structures in a
    group. A residue contacted or resolved in any member is retained.

    Two groups mix RBM and non-RBM members, so the class is recomputed from the
    MERGED epitope.

    `epitope_class` is the classification every consumer reads, in four values:

      RBM                 the epitope lies in the receptor-binding motif.
      nonRBM              no epitope residue is in the motif and every residue is
                          inside the receptor-binding domain.
      nonRBM_outside_RBD  no epitope residue is in the motif and at least one lies
                          outside the domain.
      ambiguous           the merged epitope straddles the motif boundary.

    The motif/non-motif split says nothing about the domain around it, so a
    non-motif epitope can sit outside the RBD entirely. Scoring such an antibody
    against the RBD mask measures nothing: the mask is not a baseline that has to
    be beaten, it is one that cannot be right. Comparisons against that mask
    therefore use `nonRBM`, the antibodies for which it could in principle be
    correct.
    """
    key = man.vh_seq.astype(str) + "||" + man.light_seq.fillna("").astype(str)
    lo, hi = cfg.RBM                                    # single source of truth: config/task2.py
    rbd_lo, rbd_hi = cfg.RBD
    keep_idx, out_rows = [], []
    for _, pos in pd.Series(range(len(man))).groupby(key.to_numpy()):
        pos = list(pos)
        first = pos[0]
        row = man.iloc[first].to_dict()
        merged = np.any(labels[pos], axis=0).astype(np.uint8)
        obs = np.any(obs_masks[pos], axis=0).astype(np.uint8)
        frac = merged[lo - 1:hi].sum() / max(1, merged.sum())
        row.update({
            "n_merged": len(pos),
            "merged_from": ";".join(man.case_id.iloc[pos]),
            "merged_pdb_ids": ";".join(sorted(set(str(x) for x in man.pdb_id.iloc[pos]))),
            "n_epitope_residues": int(merged.sum()),
            "n_obs_residues": int(obs.sum()),
            "frac_in_rbm": float(frac),
            "epitope_class": epitope_class(merged, frac, lo, hi, rbd_lo, rbd_hi),
        })
        out_rows.append(row)
        keep_idx.append(pos)

    new_man = pd.DataFrame(out_rows).reset_index(drop=True)
    new_lab = np.stack([np.any(labels[p], axis=0).astype(np.uint8) for p in keep_idx])
    new_obs = np.stack([np.any(obs_masks[p], axis=0).astype(np.uint8) for p in keep_idx])
    n_dup = int((new_man.n_merged > 1).sum())
    print(f"\nmerged duplicate inputs: {len(man)} rows -> {len(new_man)} distinct (VH, VL) cases")
    print(f"  {n_dup} cases merged from >1 row (largest {int(new_man.n_merged.max())})")
    print(f"  epitope size: median {int(np.median(new_lab.sum(1)))} residues "
          f"(positive rate {new_lab.mean():.4f})")
    print("  epitope_class: " + ", ".join(
        f"{k} {v}" for k, v in new_man.epitope_class.value_counts().items()))
    return new_man, new_lab, new_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trim_light", type=int, default=int(cfg.TRIM_LIGHT_TO_VARIABLE),
                    help="1 = cut the light-chain constant domain (see config).")
    ap.add_argument("--merge_duplicate_inputs", type=int, default=1,
                    help="1 = collapse rows with an identical (VH, VL) to one case, "
                         "unioning their gold epitopes.")
    args = ap.parse_args()

    spike = read_fasta_single(cfg.SPIKE_FASTA)
    df = pd.read_csv(cfg.SPIKE_CSV)
    print(f"source: {cfg.SPIKE_CSV.name}  {df.shape[0]} rows x {df.shape[1]} cols")

    # ---- global assertions: if any of these fail the benchmark frame is wrong ----
    assert len(spike) == cfg.ANTIGEN_LEN, f"spike.fa is {len(spike)} aa, expected {cfg.ANTIGEN_LEN}"
    assert (df.antigen_sequence == spike).all(), "some rows carry an antigen != spike.fa"
    assert (df.antigen_length == cfg.ANTIGEN_LEN).all(), "antigen_length is not uniform"

    rows, labels, obs_masks = [], [], []
    problems = []

    for i, r in df.iterrows():
        case_id = f"t2_{i:03d}_{str(r.pdb_id).lower()}"

        y = np.asarray(json.loads(r.epitope_labels), dtype=np.uint8)
        obs = np.asarray(json.loads(r.antigen_mask), dtype=np.uint8)
        contacts = json.loads(r.vh_antigen_contacts)

        # Per-row invariants. Collected rather than raised so one bad row does
        # not hide the others; the run aborts at the end if anything failed.
        if len(y) != cfg.ANTIGEN_LEN:
            problems.append((case_id, f"label length {len(y)}"))
        if len(obs) != cfg.ANTIGEN_LEN:
            problems.append((case_id, f"antigen_mask length {len(obs)}"))
        if y.sum() != r.n_epitope_residues:
            problems.append((case_id, "n_epitope_residues disagrees with labels"))
        if np.any(y & (1 - obs)):
            problems.append((case_id, "epitope residue outside the observed mask"))
        if contacts and max(j for _, j in contacts) >= cfg.ANTIGEN_LEN:
            problems.append((case_id, "contact antigen index out of range"))

        vh, vh_bad = clean_seq(r.vh_sequence)
        light, light_bad = clean_seq(r.light_sequence)

        # A "light chain" shorter than MIN_LIGHT_LEN is a modelled fragment, not
        # a domain; folding it would add noise, so it is dropped and recorded.
        light_note = ""
        if light and len(light) < cfg.MIN_LIGHT_LEN:
            light_note = f"dropped_short_light({len(light)})"
            light = ""
        light_trimmed = False
        if light and args.trim_light:
            light, light_trimmed = trim_light_to_variable(light)

        rows.append({
            "case_id": case_id,
            "row_index": i,
            "pdb_id": r.pdb_id,
            "frac_in_rbm": r.frac_in_rbm,
            "n_epitope": int(y.sum()),
            "n_contacts_vh": int(r.n_contacts),
            "n_obs_residues": int(obs.sum()),
            "vh_seq": vh,
            "vh_len": len(vh),
            "vh_kind": domain_kind(vh),          # expected 'heavy'
            "vh_key": chain_key("vh", vh),
            "light_seq": light,
            "light_len": len(light),
            "light_trimmed": light_trimmed,
            "light_key": chain_key("vl", light) if light else "",
            "light_kind": domain_kind(light),    # expected 'light' or 'none'
            "has_light": bool(light),
            # False when the vh_sequence column does not hold a heavy chain.
            # These cases still get folded (so the run is complete and auditable)
            # but are excluded from headline metrics.
            "chain_role_ok": domain_kind(vh) == "heavy",
            # AF3 compiles per token-count bucket; 1536 is the relevant one here.
            "n_tokens": cfg.ANTIGEN_LEN + len(vh) + len(light),
            "heavy_chain_id_src": r.heavy_chain_id,
            "light_chain_id_src": r.light_chain_id,
            "antigen_chain_src": r.antigen_chain,
            "resolution": r.resolution,
            "method": r.method,
            "organism": r.organism,
            "nonstandard_aa": vh_bad + light_bad,
            "note": light_note,
        })
        labels.append(y)
        obs_masks.append(obs)

    if problems:
        for c, why in problems[:20]:
            print(f"  FAIL {c}: {why}")
        raise SystemExit(f"{len(problems)} row-level validation failures; not writing manifest.")

    man = pd.DataFrame(rows)
    labels = np.stack(labels)
    obs_masks = np.stack(obs_masks)
    if args.merge_duplicate_inputs:
        man, labels, obs_masks = merge_duplicate_inputs(man, labels, obs_masks)

    cfg.MANIFESTS.mkdir(parents=True, exist_ok=True)
    man.to_csv(cfg.MANIFEST, index=False)
    np.savez_compressed(cfg.LABELS_NPZ,
                        case_id=np.array(man.case_id),
                        labels=labels,
                        obs_mask=obs_masks)

    # ------------------------------------------------------------- report ----
    print(f"\nvalidation: all {len(man)} rows passed")
    print(f"  epitope classes : {man.epitope_class.value_counts().to_dict()}")
    print(f"  with light chain: {int(man.has_light.sum())} / {len(man)} "
          f"(trimmed {int(man.light_trimmed.sum())})")
    print(f"  vh_sequence is    : {man.vh_kind.value_counts().to_dict()}")
    print(f"  light_sequence is : {man.light_kind.value_counts().to_dict()}")
    print(f"  chain_role_ok     : {int(man.chain_role_ok.sum())} / {len(man)} "
          f"({int((~man.chain_role_ok).sum())} flagged for exclusion from headline metrics)")
    print(f"  short lights dropped: {int((man.note != '').sum())}")
    print(f"  tokens          : min {man.n_tokens.min()} / median "
          f"{int(man.n_tokens.median())} / max {man.n_tokens.max()} "
          f"({int((man.n_tokens > 1536).sum())} above the 1536 bucket)")
    print(f"  unique chains   : {man.vh_key.nunique()} VH + "
          f"{man.loc[man.has_light, 'light_key'].nunique()} light + 1 antigen")
    print(f"  observed antigen: median {int(np.median(man.n_obs_residues))} / "
          f"{cfg.ANTIGEN_LEN} residues resolved in the source structure")
    print(f"\nwrote {cfg.MANIFEST}")
    print(f"wrote {cfg.LABELS_NPZ}")


if __name__ == "__main__":
    main()
