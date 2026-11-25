#!/usr/bin/env python3
"""
ESMFold2 on Task I -- one model load per shard, not per row.

What this driver does. It reads a manifest in which every row carries its own
antigen and folds each (VH, VL, antigen) complex in turn. The model is loaded once
per shard, because loading the ~6-shard checkpoint costs roughly 1 GPU-minute of
pure checkpoint reading, which over hundreds of complexes would dominate the run.
Per-row antigens under a single model load is the property this task needs.

The folding call uses the same builder and settings as the Task II and Task III
ESMFold2 folds -- num_loops=8, num_sampling_steps=50, seed=0 -- so all three tasks
are directly comparable. The settings belong to the model; this file supplies the
loop around it.

The evaluated set is `usable & vh_dedup_keep & af3_eligible` = 3,908 complexes,
the same definition t1_01_make_chain_jobs.py states in full.

Measured cost. A 484-fold run (job 4000227, four shards) took 14.9 GPU-hours =
1.85 min/fold, so the evaluated set is ~120 GPU-hours if the length mix matched.
It does not match: the evaluated set carries 1,077 rows over 1,000 residues (spike
alone is ~800). Budget ~200.

Sharding is by length, round-robin. Cost per fold rises steeply with complex
length, and case_id order correlates with antigen, so a contiguous slice of the
full evaluated set hands one shard every spike row. Shard i takes rows i, i+N,
i+2N, ... of the length-sorted evaluated set, which gives every shard the same
length profile by construction. Contiguous shards of 125 rows took 8h28m against
1h52m, a 4.5x spread for equal row counts. --start/--end mean a contiguous slice
of the evaluated set.

Per-row failures are caught and recorded rather than fatal: one OOM at row 12 must
not cost the other 88 rows in the shard. Resumes by skipping rows whose cif exists.

    PYTHONPATH=scripts:config python scripts/t1_esm_fold.py --start 0 --end 125
    PYTHONPATH=scripts:config python scripts/t1_esm_fold.py \
        --shard 3 --n_shards 10
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                        # noqa: E402

OUT = PROJECT / "work" / "task1" / "esm"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1,
                    help="round-robin shard index over the LENGTH-SORTED roster; "
                         "requires --n_shards. Mutually exclusive with --start/--end.")
    ap.add_argument("--n_shards", type=int, default=0)
    ap.add_argument("--len_band", default="",
                    help="LO:HI on vh_len+antigen_len, applied BEFORE sharding. "
                         "Lets the short complexes run on 48 GB l40s cards and the "
                         "long ones on h200, which is the only partition where a "
                         "1,573-residue complex is known to fit (16 "
                         "failures in an earlier run were all nonstandard residues, never OOM, so "
                         "the l40s ceiling is untested rather than known-bad).")
    ap.add_argument("--start", type=int, default=0, help="first roster row (incl)")
    ap.add_argument("--end", type=int, default=-1, help="last roster row (excl), -1 = all")
    ap.add_argument("--num_loops", type=int, default=8)
    ap.add_argument("--num_sampling_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_tag", default="0")
    a = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    sub = man[man.usable & man.vh_dedup_keep & man.af3_eligible]
    sub = sub.sort_values("case_id").reset_index(drop=True)

    if a.len_band:
        lo, hi = (int(x) for x in a.len_band.split(":"))
        L = sub.vh_len + sub.antigen_len
        sub = sub[(L >= lo) & (L < hi)].reset_index(drop=True)
        print(f"len_band [{lo},{hi}) -> {len(sub)} rows", flush=True)

    if a.shard >= 0:
        if a.n_shards <= 0:
            ap.error("--shard requires --n_shards")
        # Length-sorted, then dealt round-robin: equal row counts and equal
        # length profiles. Sorting is on the complex rather than the antigen,
        # because the fold sees VH + antigen as one chain set.
        order = (sub.assign(_L=sub.vh_len + sub.antigen_len)
                    .sort_values(["_L", "case_id"]).index.to_numpy())
        rows = sub.loc[order[a.shard::a.n_shards]]
        span = f"shard {a.shard}/{a.n_shards} round-robin, {len(rows)} rows, " \
               f"complex len {int((rows.vh_len + rows.antigen_len).min())}-" \
               f"{int((rows.vh_len + rows.antigen_len).max())}"
    else:
        end = len(sub) if a.end < 0 else min(a.end, len(sub))
        rows = sub.iloc[a.start:end]
        span = f"contiguous rows [{a.start}:{end})"

    (OUT / "cifs").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifests" / f"shard_{a.shard_tag}.csv"
    print(f"shard {a.shard_tag}: roster ({len(sub)} rows), {span}",
          flush=True)

    import torch
    from esm.models.esmfold2 import (ESMFold2InputBuilder, ProteinInput,
                                     StructurePredictionInput)
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    builder = ESMFold2InputBuilder()
    print("model loaded", flush=True)

    n_ok = n_skip = n_fail = 0
    for _, r in rows.iterrows():
        cif = OUT / "cifs" / f"{r.case_id}.cif"
        if cif.exists() and cif.stat().st_size > 0:
            n_skip += 1
            continue
        vh, ag = str(r.vh_seq).upper(), str(r.antigen_sequence).upper()
        rec = {"case_id": r.case_id, "fold": r.fold, "vh_len": len(vh),
               "ag_len": len(ag), "complex_len": len(vh) + len(ag),
               "status": "", "iptm": "", "ptm": "", "plddt_mean": "", "error": ""}
        # ESMFold2's tokenizer has no symbol for X; the label vector is indexed
        # against this exact sequence, so a substitution would shift the gold.
        if (set(vh) | set(ag)) - VALID_AA:
            rec["status"] = "nonstandard_residue"
            n_fail += 1
        else:
            try:
                spi = StructurePredictionInput(sequences=[
                    ProteinInput(id="H", sequence=vh),
                    ProteinInput(id="A", sequence=ag)])
                with torch.no_grad():
                    res = builder.fold(model, spi, num_loops=a.num_loops,
                                       num_sampling_steps=a.num_sampling_steps,
                                       num_diffusion_samples=1, seed=a.seed)
                    cif.write_text(res.complex.to_mmcif())
                    rec["iptm"] = float(res.iptm) if res.iptm is not None else ""
                    rec["ptm"] = float(res.ptm) if res.ptm is not None else ""
                    rec["plddt_mean"] = (float(np.asarray(res.plddt).mean())
                                         if res.plddt is not None else "")
                rec["status"] = "ok"
                n_ok += 1
            except Exception as e:                  # one bad row must not end the shard
                oom = "out of memory" in str(e).lower()
                rec["status"] = "oom" if oom else "fold_error"
                rec["error"] = str(e)[:200]
                n_fail += 1
                torch.cuda.empty_cache()
        new = not mpath.exists()
        with open(mpath, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rec))
            if new:
                w.writeheader()
            w.writerow(rec)
        print(f"[{r.case_id}] {rec['status']} len={rec['complex_len']} "
              f"iptm={rec['iptm']}", flush=True)

    print(f"\nshard {a.shard_tag} done: ok {n_ok}, already present {n_skip}, "
          f"failed {n_fail}  -> {mpath}")


if __name__ == "__main__":
    main()
