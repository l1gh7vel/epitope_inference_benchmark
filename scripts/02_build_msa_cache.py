#!/usr/bin/env python3
"""
Stage 02 -- harvest per-chain MSA blocks from the stage-01 output.

AF3 writes <output_dir>/<name>/<name>_data.json: the input JSON echoed back with
unpairedMsa, pairedMsa and templates filled in for every chain. Each chain block
is computed independently of the others (its own jackhmmer/hmmsearch run against
the same databases), which is what makes it safe to splice one chain's block into
a different complex later.

We pull out the single chain block and store it on its own, so stage 03 can
assemble a complex without re-reading a ~85 MB file per chain.

    python scripts/02_build_msa_cache.py           # harvest + report gaps
    python scripts/02_build_msa_cache.py --prune   # also delete work/msa_raw
"""

import argparse
import json
import os
import shutil

import pandas as pd

from common import cfg, write_json

REQUIRED = ("unpairedMsa", "pairedMsa", "templates")


def harvest_one(key):
    """Return the chain block for `key`, or None if the AF3 run is missing/incomplete.

    AF3 3.0.2 PRESERVES THE CASE of the job name in its output directory, so the
    harvest must match on the name as given rather than on `key.lower()`. Every
    chain key this pipeline generates is already lowercase (`ag_<hash>` /
    `vh_<hash>`), but the RSV arm's antigen chain is hand-named `ag_rsvF_long`
    by `rsv_02_assemble_fold.py`. For that one chain the lowercase path does not
    exist and the harvest reported "no _data.json" for a search that had in fact
    finished. Try the key as written as well; lowercase stays first so nothing
    that worked before changes.
    """
    data = next((p for p in (cfg.MSA_RAW / k / f"{k}_data.json"
                             for k in (key.lower(), key)) if p.exists()), None)
    if data is None:
        return None, "no _data.json"
    with open(data) as fh:
        obj = json.load(fh)
    prots = [s["protein"] for s in obj["sequences"] if "protein" in s]
    if len(prots) != 1:
        return None, f"expected 1 chain, found {len(prots)}"
    block = prots[0]
    missing = [f for f in REQUIRED if f not in block]
    if missing:
        return None, f"missing {missing}"
    # Chain id is assigned at assembly time, so drop it here to avoid confusion.
    block.pop("id", None)
    return block, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true",
                    help="Delete work/msa_raw after a fully successful harvest.")
    args = ap.parse_args()

    idx = pd.read_csv(cfg.CHAIN_JOBS / "index.csv")
    cfg.MSA_CACHE.mkdir(parents=True, exist_ok=True)

    done, cached, failed = 0, 0, []
    for _, r in idx.iterrows():
        out = cfg.MSA_CACHE / f"{r.chain_key}.json"
        if out.exists() and out.stat().st_size > 0:
            cached += 1
            continue
        block, why = harvest_one(r.chain_key)
        if block is None:
            failed.append((r.chain_key, r.kind, why))
            continue
        write_json(out, block)
        done += 1

    total = len(idx)
    print(f"chains: {total} | newly harvested {done} | already cached {cached} | "
          f"missing {len(failed)}")
    if failed:
        by_kind = pd.Series([k for _, k, _ in failed]).value_counts().to_dict()
        print(f"  missing by kind: {by_kind}")
        for key, kind, why in failed[:15]:
            print(f"    {key} ({kind}): {why}")
        if len(failed) > 15:
            print(f"    ... and {len(failed) - 15} more")
        # Array indices of the chains still to run, so the rerun is a one-liner.
        todo = sorted(idx.index[idx.chain_key.isin([k for k, _, _ in failed])])
        # This cluster's MaxArraySize is 500, so a flat id list is not always a
        # valid submission -- Task I has 914 chains. Emit one chunk per 500-id
        # window, with the EIB_ARRAY_OFFSET the sbatch expects, so the printed
        # command can be pasted as-is rather than silently rejected.
        task = os.environ.get("EIB_TASK", "task2")
        for lo in range(0, max(todo) + 1, 500):
            chunk = [i - lo for i in todo if lo <= i < lo + 500]
            if not chunk:
                continue
            env = f"--export=ALL,EIB_TASK={task}"
            if lo:
                env += f",EIB_ARRAY_OFFSET={lo}"
            print(f"\n  rerun: sbatch {env} --array=$(echo "
                  f"'{','.join(map(str, chunk))}')")
        return

    print(f"\nall {total} chain blocks cached in {cfg.MSA_CACHE}")
    if args.prune:
        shutil.rmtree(cfg.MSA_RAW, ignore_errors=True)
        print(f"pruned {cfg.MSA_RAW}")
    else:
        print(f"(rerun with --prune to delete {cfg.MSA_RAW} and reclaim disk)")


if __name__ == "__main__":
    main()
