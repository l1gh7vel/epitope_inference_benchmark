#!/usr/bin/env python3
"""Stage 01 for the RSV F arm -- one single-chain AF3 MSA job per unique chain.

Same purpose as `scripts/01_make_chain_jobs.py`: MSA search is ~99% of AF3's wall
clock here, so each distinct chain is searched once, cached by stage 02, and
spliced into complexes by stage 03 with --norun_data_pipeline.

Why this arm needs its own copy of stage 01
-------------------------------------------------------------------------------
The shared stage 01 derives the antigen's cache key as `chain_key("ag", seq)`,
`ag_<sha1[:10]>`. The RSV antigen's key is the hand-assigned `ag_rsvF_long`, named
by `rsv_02_assemble_fold.py` for the prefusion pre-check before this arm had a
config, and its MSA is already searched and sitting in
`work/rsv_f/msa_raw/ag_rsvF_long/`.

Running the shared stage 01 here would emit a new `ag_<hash>.json` whose MSA has
never been searched, and its stale-pruning step would delete `ag_rsvF_long.json`,
costing a repeated ~25 CPU-minute antigen search and leaving an orphaned cache. So
the antigen key comes from `cfg.ANTIGEN_CHAIN_KEY`, and the antigen job file is
written and never pruned.

The array-index contract
------------------------------------------------------------------
The MSA stage maps array task N to the Nth file of
`ls work/rsv_f/chain_jobs/*.json | sort`. index.csv is written in that same sorted
order, so row N of index.csv is array task N. Add, remove or rename a job file and
every index after it shifts. This script prints the exact array range to submit,
and it prints which index the already-searched antigen occupies so it can be left
out of the submission.

Sorted order puts `ag_` first, then `vh_`, then `vl_`, so the antigen is index 0
and the twelve antibody chains are 1..12.

Why the antigen is skipped by hand rather than by the sbatch's own guard
------------------------------------------------------------------------------
The sbatch's idempotence guard looks for `msa_raw/${KEY,,}/${KEY,,}_data.json`,
which lowercases the key. The directory AF3 wrote is `ag_rsvF_long`, mixed case,
because AF3 3.0.2 preserves the job name's case. The guard therefore misses for
this one chain and would redo the search. Submit `--array=1-12` rather than
`0-12`. (`scripts/02_build_msa_cache.py` accepts either case, so the harvest finds
this chain.)

    PYTHONPATH=scripts:config EIB_TASK=rsv_f python scripts/rsv_07_make_chain_jobs.py
"""
import os, sys
from pathlib import Path

import pandas as pd

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
import rsv_f as cfg                                              # noqa: E402
from common import af3_job, af3_protein, read_fasta_single, write_json   # noqa: E402


def main():
    man = pd.read_csv(cfg.MANIFEST)
    ag = read_fasta_single(cfg.ANTIGEN_FASTA)

    chains = {cfg.ANTIGEN_CHAIN_KEY: ag}
    for r in man.itertuples():
        chains[r.vh_key] = r.vh_seq
        if r.has_light:
            chains[r.light_key] = r.light_seq

    cfg.CHAIN_JOBS.mkdir(parents=True, exist_ok=True)
    index = []
    for key in sorted(chains):
        seq = chains[key]
        # Single-chain job. Chain id "A" carries no meaning; the real id is
        # assigned when stage 03 splices the block into a complex. seeds=[1]
        # because --norun_inference never reaches the diffusion stage.
        write_json(cfg.CHAIN_JOBS / f"{key}.json",
                   af3_job(key, [af3_protein("A", seq)], seeds=[1]))
        index.append(dict(chain_key=key, kind=key.split("_")[0], length=len(seq)))

    stale = [p for p in cfg.CHAIN_JOBS.glob("*.json") if p.stem not in chains]
    for p in stale:
        p.unlink()
    if stale:
        print(f"pruned {len(stale)} stale job file(s): {[p.stem for p in stale]}")

    idx = pd.DataFrame(index)
    idx.to_csv(cfg.CHAIN_JOBS / "index.csv", index=False)
    print(idx.to_string())
    print(f"\n{len(man)} fold cases -> {len(chains)} unique chains "
          f"({len(man) + int(man.has_light.sum()) + len(man) - len(chains)} "
          "searches avoided by dedup + the cached antigen)")

    raw = cfg.MSA_RAW / cfg.ANTIGEN_CHAIN_KEY / f"{cfg.ANTIGEN_CHAIN_KEY}_data.json"
    ag_i = int(idx.index[idx.chain_key == cfg.ANTIGEN_CHAIN_KEY][0])
    print(f"\nantigen {cfg.ANTIGEN_CHAIN_KEY} is array index {ag_i}; its MSA is "
          f"{'ALREADY SEARCHED' if raw.exists() else 'MISSING'} ({raw})")
    todo = [i for i in idx.index if i != ag_i] if raw.exists() else list(idx.index)
    print(f"chains still needing an MSA: array indices {min(todo)}-{max(todo)} "
          f"({len(todo)} chains, {idx.loc[todo, 'length'].min()}-"
          f"{idx.loc[todo, 'length'].max()} aa)")
    print("\nSUBMIT ON smp, NOT htc -- htc-htc-n is at its cpu=1613 per-ACCOUNT cap\n"
          "with Task I. smp has an identical, unused pool and is CPU-only:\n\n"
          f"  sbatch --clusters=smp --partition=smp --qos=smp-smp-n \\\n"
          f"         --time=3-00:00:00 --mem=32G --array={min(todo)}-{max(todo)}%6 \\\n"
          f"         --export=ALL,EIB_TASK=rsv_f\n\n"
          "then:  squeue -M smp -u $USER     (squeue DEFAULTS TO htc and will not\n"
          "                                   show these jobs without -M smp)")


if __name__ == "__main__":
    main()
