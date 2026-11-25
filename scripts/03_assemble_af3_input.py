#!/usr/bin/env python3
"""
Stage 03 -- assemble the AF3 input JSON for one complex from cached chain blocks.

Chains are always A = antigen (full-length Spike), H = heavy, L = light (omitted
when the row has none). Because each chain carries its cached unpairedMsa /
pairedMsa / templates, AF3 can then be run with --norun_data_pipeline and goes
straight to inference.

The assembled file is about 85 MB, so batch jobs write it to node-local scratch.

    python scripts/03_assemble_af3_input.py --index 0 --out $SLURM_SCRATCH/case.json
    python scripts/03_assemble_af3_input.py --case_id t2_000_8ywe --out /tmp/x.json
"""

import argparse
import json
import sys

import pandas as pd

from common import af3_job, af3_protein, cfg, chain_key, read_fasta_single, write_json


def load_block(key):
    path = cfg.MSA_CACHE / f"{key}.json"
    if not path.exists():
        sys.exit(f"ERROR: no cached MSA for chain {key} ({path}). Run stages 01-02 first.")
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="0-based row of the manifest (for Slurm arrays).")
    g.add_argument("--case_id", help="case_id from the manifest.")
    ap.add_argument("--out", required=True, help="Where to write the assembled JSON.")
    ap.add_argument("--print_name", action="store_true",
                    help="Print only the AF3 job name (used by the sbatch to locate output).")
    args = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    row = man.iloc[args.index] if args.index is not None else \
        man.loc[man.case_id == args.case_id].squeeze()
    if isinstance(row, pd.DataFrame) or row.empty:
        sys.exit(f"ERROR: case not found: {args.case_id}")

    spike = read_fasta_single(cfg.ANTIGEN_FASTA)
    # The antigen's cache key is normally derived from its sequence. One arm's is
    # not: the RSV F antigen was hand-named `ag_rsvF_long` by the prefusion
    # pre-check before that arm had a config, and its (expensive) MSA is cached
    # under that name. A config may therefore state the key; the default is the
    # derived one, so no existing task changes.
    ag_key = getattr(cfg, "ANTIGEN_CHAIN_KEY", None) or chain_key("ag", spike)
    proteins = [
        af3_protein(cfg.CHAIN_ANTIGEN, spike, load_block(ag_key)),
        af3_protein(cfg.CHAIN_HEAVY, row.vh_seq, load_block(row.vh_key)),
    ]
    if row.has_light:
        proteins.append(af3_protein(cfg.CHAIN_LIGHT, row.light_seq, load_block(row.light_key)))

    write_json(args.out, af3_job(row.case_id, proteins))

    if args.print_name:
        print(row.case_id)
    else:
        print(f"{row.case_id}: {len(proteins)} chains, {row.n_tokens} tokens "
              f"({'VH+VL' if row.has_light else 'VH only'}) -> {args.out}")


if __name__ == "__main__":
    main()
