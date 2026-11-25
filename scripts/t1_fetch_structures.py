#!/usr/bin/env python3
"""Download the deposited mmCIFs for Task I, so WALLE can read real structures.

Every Task I case IS a solved complex -- that is where the fold data comes from --
so unlike Tasks II/III there is no reason to hand WALLE a predicted structure. This
fetches one gzipped mmCIF per distinct `pdb_id` into `work/task1/walle/pdb/`.

Gzipped on purpose: many Task I complexes are cryo-EM spike structures, which are tens
of megabytes each uncompressed. gemmi reads .cif.gz transparently.

    python scripts/t1_fetch_structures.py
"""
import argparse, sys, time, urllib.error, urllib.request
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                                  # noqa: E402

OUT = cfg.WORK / "walle" / "pdb"
URL = "https://files.rcsb.org/download/{}.cif.gz"


def fetch(pdb, dest, tries=3):
    for t in range(tries):
        try:
            with urllib.request.urlopen(URL.format(pdb.upper()), timeout=120) as r:
                data = r.read()
            if len(data) < 200:
                raise ValueError(f"suspiciously small ({len(data)} B)")
            dest.write_bytes(data)
            return True, len(data)
        except Exception as e:                                   # noqa: BLE001
            if t == tries - 1:
                return False, str(e)
            time.sleep(2 * (t + 1))
    return False, "unreachable"


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST)
    pdbs = sorted(man.pdb_id.dropna().astype(str).str.lower().unique())
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(man)} rows -> {len(pdbs)} distinct structures -> {OUT}")

    ok = skip = 0
    failed = []
    for i, p in enumerate(pdbs, 1):
        dest = OUT / f"{p}.cif.gz"
        if dest.exists() and dest.stat().st_size > 200:
            skip += 1
            continue
        good, info = fetch(p, dest)
        if good:
            ok += 1
        else:
            failed.append((p, info))
            print(f"  {p}: FAIL {info}", flush=True)
        if i % 25 == 0:
            print(f"  ...{i}/{len(pdbs)}  downloaded {ok}  cached {skip}  failed {len(failed)}",
                  flush=True)

    print(f"\ndownloaded {ok}, already present {skip}, failed {len(failed)}")
    if failed:
        print("failures:")
        for p, why in failed:
            print(f"  {p}: {why}")


if __name__ == "__main__":
    main()
