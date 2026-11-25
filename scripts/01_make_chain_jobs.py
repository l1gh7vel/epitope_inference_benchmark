#!/usr/bin/env python3
"""Emit one AlphaFold3 data-pipeline job per unique chain sequence.

The antigen and repeated antibody chains are searched once and cached for all
manifest cases. Outputs are `work/chain_jobs/<chain_key>.json` and `index.csv`.
"""

import pandas as pd

from common import af3_job, af3_protein, cfg, chain_key, read_fasta_single, write_json


def main():
    man = pd.read_csv(cfg.MANIFEST)
    spike = read_fasta_single(cfg.ANTIGEN_FASTA)

    chains = {chain_key("ag", spike): spike}
    for _, r in man.iterrows():
        chains[r.vh_key] = r.vh_seq
        if r.has_light:
            chains[r.light_key] = r.light_seq

    cfg.CHAIN_JOBS.mkdir(parents=True, exist_ok=True)
    index = []
    # Sorted by key so that row N of index.csv is exactly Slurm array task N:
    # the sbatch picks its chain with `ls work/chain_jobs/*.json | sort`. Keep
    # these two orderings identical or the "rerun these array ids" hint in
    # stage 02 points at the wrong chains. ('ag_' sorts first, which is also
    # convenient -- the antigen is by far the slowest job.)
    for key in sorted(chains):
        seq = chains[key]
        # A single-chain job. Chain id is "A" here and carries no meaning; the
        # real id is assigned when stage 03 splices the block into a complex.
        write_json(cfg.CHAIN_JOBS / f"{key}.json",
                   af3_job(key, [af3_protein("A", seq)], seeds=[1]))
        index.append({"chain_key": key, "kind": key.split("_")[0], "length": len(seq)})

    # Prune job files the current manifest does not reference. The Slurm array
    # maps task id -> Nth sorted filename, so a leftover JSON from an earlier run
    # silently shifts every index after it and one real chain never gets submitted.
    # (Exactly that happened on Task III after a sequence-trim change.)
    stale = [p for p in cfg.CHAIN_JOBS.glob("*.json") if p.stem not in chains]
    for p_ in stale:
        p_.unlink()
    if stale:
        print(f"pruned {len(stale)} stale job file(s): {[p_.stem for p_ in stale]}")

    idx = pd.DataFrame(index)
    idx.to_csv(cfg.CHAIN_JOBS / "index.csv", index=False)

    print(f"{len(man)} cases -> {len(chains)} unique chains "
          f"({(3 * len(man) - len(chains))} MSA searches avoided)")
    print(idx.groupby("kind").agg(n=("chain_key", "size"),
                                  min_len=("length", "min"),
                                  max_len=("length", "max")).to_string())
    print(f"\nwrote {len(chains)} job JSONs to {cfg.CHAIN_JOBS}")
    print(f"index: {cfg.CHAIN_JOBS / 'index.csv'}")
    print(f"\nnext: run the MSA stage, scripts/02_build_msa_cache.py, over "
          f"chain indices 0-{len(chains) - 1}")


if __name__ == "__main__":
    main()
