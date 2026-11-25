#!/usr/bin/env python3
"""
Task I stage 01 -- one single-chain AF3 job per unique sequence.

The paper runs Task I across all antigen groups. The evaluated set is defined here once
and every later stage reads the same definition:

    usable  AND  vh_dedup_keep  AND  af3_eligible   ->  3,908 complexes
            (890 / 733 / 737 / 788 / 760 across the five antigen groups)

  usable        : >= 3 epitope residues, VH > 60 aa -- the evaluated set t1_05 scores.
  vh_dedup_keep : removes heavy chains appearing in more than one group, so the
                  groups are antibody-disjoint as well as antigen-disjoint.
  af3_eligible  : antigen_len <= 1400, the AF3 token-bucket cap. It is LOAD-BEARING
                  and it costs 269 of the 4,177 otherwise-eligible rows. Those rows
                  are reported rather than dropped silently: BepiPred-3.0 folds
                  nothing and still covers them, so they stay in the BepiPred
                  full-set arm. The cap is NOT lowered to
                  1,200 -- at 1,200 group 1 collapses from 890 rows to 35,
                  because spike is 1,273.

  Cost of the evaluated set: 4,978 distinct chains, 1,403 antigen and 3,575 VH.
  At about 25 min per antigen and 22 min per VH on 16 CPUs, searching all of them
  is roughly 1,900 node-hours. Each chain is searched once and cached, so a rerun
  pays only for chains the cache does not hold. This is why the stage runs on a
  CPU partition and is the longest in the pipeline.

WHY TASK I NEEDS ITS OWN STAGE 01. The shared stage 01 assumes ONE antigen for the
whole task: it reads `cfg.ANTIGEN_FASTA` and emits a single antigen chain job that
every complex then reuses. That assumption is exactly what Task I removes -- each
complex carries its own antigen -- so the antigen side goes from 1 MSA search to
one per distinct antigen, and the manifest has to carry a per-row antigen key.
MSA search is ~99% of AF3's wall clock here (~25 min for a mid-sized antigen).

Antibody chains are VH-only, matching Task III: these rows have no light chain.

    PYTHONPATH=scripts:config python scripts/t1_01_make_chain_jobs.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
import task1 as cfg                                        # noqa: E402
from common import af3_job, af3_protein, chain_key, write_json   # noqa: E402


# Columns that must all be True for a complex to be on the evaluated set.
ROSTER = ["usable", "vh_dedup_keep", "af3_eligible"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    man["vh_key"] = [chain_key("vh", s) for s in man.vh_seq]
    man["ag_key"] = [chain_key("ag", s) for s in man.antigen_sequence]
    man.to_csv(cfg.MANIFEST, index=False)

    mask = man[ROSTER].all(axis=1)
    sub = man[mask]
    print(f"roster = {' & '.join(ROSTER)}  ->  {len(sub)} complexes")
    print(sub.fold.value_counts().sort_index().to_string())
    chains = {}
    for _, r in sub.iterrows():
        chains[r.ag_key] = r.antigen_sequence
        chains[r.vh_key] = r.vh_seq

    cfg.CHAIN_JOBS.mkdir(parents=True, exist_ok=True)
    index = []
    # Sorted by key so that row N of index.csv is exactly Slurm array task N -- the
    # sbatch picks its chain with `ls work/.../chain_jobs/*.json | sort`. Keep the
    # two orderings identical or the "rerun these array ids" hint points at the
    # wrong chains.
    for key in sorted(chains):
        seq = chains[key]
        write_json(cfg.CHAIN_JOBS / f"{key}.json",
                   af3_job(key, [af3_protein("A", seq)], seeds=[1]))
        index.append({"chain_key": key, "kind": key.split("_")[0], "length": len(seq)})

    # Prune job files this manifest does not reference: the Slurm array maps task
    # id -> Nth sorted filename, so one leftover JSON shifts every index after it
    # and a real chain silently never runs.
    stale = [p for p in cfg.CHAIN_JOBS.glob("*.json") if p.stem not in chains]
    for p in stale:
        p.unlink()
    if stale:
        print(f"pruned {len(stale)} stale job file(s)")

    idx = pd.DataFrame(index)
    idx.to_csv(cfg.CHAIN_JOBS / "index.csv", index=False)
    print(f"{len(sub)} complexes -> {len(chains)} unique chains "
          f"({2 * len(sub) - len(chains)} MSA searches avoided)")
    print(idx.groupby("kind").agg(n=("chain_key", "size"), min_len=("length", "min"),
                                  max_len=("length", "max"),
                                  total_res=("length", "sum")).to_string())
    print(f"\nwrote {len(chains)} job JSONs to {cfg.CHAIN_JOBS}")
    # MaxArraySize on this cluster is 500, so an evaluated set of this size is submitted as
    # consecutive chunks with EIB_ARRAY_OFFSET. Antigens sort before VH ('ag_' <
    # 'vh_'), so the chunks that need --mem=96G are exactly the leading ones.
    n_ag = int((idx.kind == "ag").sum())
    print(f"\nchains 0-{n_ag - 1} are ANTIGENS (need --mem=96G); "
          f"{n_ag}-{len(chains) - 1} are VH (--mem=32G is ample)")
    print("next: submit in chunks of 500 with EIB_ARRAY_OFFSET, e.g.")
    print("  sbatch --mem=96G --array=0-499%40 "
          "--export=ALL,EIB_TASK=task1,EIB_ARRAY_OFFSET=0")


if __name__ == "__main__":
    main()
