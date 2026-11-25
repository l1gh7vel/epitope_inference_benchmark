#!/usr/bin/env python3
"""
Task I stage 03 -- assemble one complex's AF3 input from cached chain blocks.

Task I needs its own stage 03. The shared one reads `cfg.ANTIGEN_FASTA`, one
antigen for the whole task, and looks up a single cached antigen block. Task I has
one antigen per row, so both the sequence and the cache key come from the manifest
row.

Chains are A = antigen, H = VH. There is no light chain in this dataset, which
matches Task III rather than Task II.

Each chain carries its own cached unpairedMsa / pairedMsa / templates, so AF3 runs
with --norun_data_pipeline and goes straight to inference. The assembled file is
large (the antigen MSA dominates), so it is written to node-local scratch at job
time rather than kept on /ix.

The evaluated set is `usable & vh_dedup_keep & af3_eligible` = 3,908 complexes,
the same definition t1_01_make_chain_jobs.py and t1_esm_fold.py state. --index
indexes the evaluated set sorted by case_id.

--list prints a shard's case ids instead of assembling, because the evaluated set
array cannot be one task per case: this cluster's MaxArraySize is 500 and the
evaluated set leaves ~3,450 to fold. The sbatch asks for its shard's list and
loops. Sharding is round-robin over the length-sorted evaluated set rather than a
contiguous slice, for the reason t1_esm_fold.py records: AF3 inference cost rises
steeply with token count and case_id order correlates with antigen, so a
contiguous slice hands one shard every spike row. Measured over 500 complexes
(72.1 GPU-h): elapsed goes as complex_len^1.5 with 13% mean error, from 92 s at
174 residues to 2,117 s at 1,573.

    PYTHONPATH=scripts:config python scripts/t1_03_assemble_af3_input.py \
        --index 0 --out $SLURM_SCRATCH/case.json
    python scripts/t1_03_assemble_af3_input.py \
        --shard 0 --n_shards 50 --skip_done --list
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
import task1 as cfg                                        # noqa: E402
from common import af3_job, af3_protein, write_json        # noqa: E402


def load_block(key):
    path = cfg.MSA_CACHE / f"{key}.json"
    if not path.exists():
        sys.exit(f"ERROR: no cached MSA for chain {key} ({path}). "
                 f"Run t1_01 + stage 02 first.")
    with open(path) as fh:
        return json.load(fh)


def roster():
    """The evaluated set, sorted by case_id.

    `usable & vh_dedup_keep & af3_eligible`, which is 3,908 rows and is the same
    predicate t1_esm_fold.py applies. Sorting by case_id is what makes --index
    reproducible across runs and across methods.
    """
    man = pd.read_csv(cfg.MANIFEST)
    sub = man[man.usable & man.vh_dedup_keep & man.af3_eligible]
    return sub.sort_values("case_id").reset_index(drop=True)


def already_folded(case_id):
    """True when AF3 has written this case's top-ranked model. Same guard the
    sbatch uses, so the listing and the per-case skip can never disagree."""
    return (cfg.AF3_OUT / case_id / f"{case_id}_model.cif").exists()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int,
                   help="0-based row of the roster, sorted by case_id")
    g.add_argument("--case_id")
    g.add_argument("--list", action="store_true",
                   help="print the shard's case ids, one per line, and exit")
    ap.add_argument("--shard", type=int, default=-1,
                    help="round-robin shard index over the LENGTH-SORTED roster")
    ap.add_argument("--n_shards", type=int, default=0)
    ap.add_argument("--skip_done", action="store_true",
                    help="with --list, drop cases whose AF3 model.cif already "
                         "exists, so shards are balanced on work left to do")
    ap.add_argument("--out")
    ap.add_argument("--print_name", action="store_true")
    args = ap.parse_args()

    sub = roster()

    if args.list:
        if args.shard < 0 or args.n_shards <= 0:
            ap.error("--list requires --shard and --n_shards")
        # Length-sorted, then dealt round-robin: equal row counts and equal
        # length profiles, so shards inside one array finish together.
        #
        # Shard membership is computed on the whole evaluated set, and
        # --skip_done is applied afterwards. Sharding the not-yet-done set would
        # make a shard's membership depend on when it was listed, and an array
        # throttled to 10 concurrent lists its shards minutes to hours apart, so
        # two shards would be handed the same case and two GPUs would write the
        # same AF3 output directory at once. Dropping finished cases after the
        # deal costs nothing, because the already-folded cases are length-diverse
        # and round-robin spreads them evenly.
        order = (sub.assign(_L=sub.vh_len + sub.antigen_len)
                    .sort_values(["_L", "case_id"]).index.to_numpy())
        mine = sub.loc[order[args.shard::args.n_shards]]
        if args.skip_done:
            mine = mine[~mine.case_id.map(already_folded)]
        for cid in mine.case_id:
            print(cid)
        return

    if not args.out:
        ap.error("--out is required unless --list is given")

    if args.index is not None:
        if args.index >= len(sub):
            sys.exit(f"ERROR: index {args.index} >= {len(sub)} rows "
                     f"in the roster")
        row = sub.iloc[args.index]
    else:
        hit = sub.loc[sub.case_id == args.case_id]
        if hit.empty:
            sys.exit(f"ERROR: case not found in the roster: "
                     f"{args.case_id}")
        row = hit.squeeze()

    proteins = [
        af3_protein("A", row.antigen_sequence, load_block(row.ag_key)),
        af3_protein("H", row.vh_seq, load_block(row.vh_key)),
    ]
    write_json(args.out, af3_job(row.case_id, proteins))

    if args.print_name:
        print(row.case_id)
    else:
        print(f"{row.case_id}: 2 chains, {len(row.antigen_sequence) + len(row.vh_seq)} "
              f"tokens (fold {row.fold}, antigen {row.antigen_len} aa) -> {args.out}")


if __name__ == "__main__":
    main()
