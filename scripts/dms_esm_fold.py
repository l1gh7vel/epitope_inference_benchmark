#!/usr/bin/env python3
"""ESMFold2 co-folds for the DMS panel -- one model load per shard.

The FOLDING CALL IS COPIED EXACTLY from `t1_esm_fold.py` (itself copied from
the upstream `fold_one_row_vhvl.py`): same builder, num_loops=8,
num_sampling_steps=50, seed=0. So these structures stay directly comparable with
the Task I/II/III ESMFold2 folds. Only the loop around it is ours.

Handles both arms from one manifest: `vhvl` emits chains H, L, A and `vh` emits
H, A. Chain ids match the AF3 convention in config/dms.py so the scorer does not
care which folder produced a structure.

Per-row failures are caught and recorded, never fatal. Resumes by skipping rows
whose cif already exists.

Task selection is `EIB_TASK` (default "dms"), the same switch scripts/common.py
uses, so this one script now serves the HIV Env DMS panel, the RSV F arm
(EIB_TASK=rsv_f) and the Nipah G arm (EIB_TASK=niv). It needs only four things
from a config -- MANIFEST, ANTIGEN_FASTA, WORK and the three CHAIN_* ids -- and
writes to <cfg.WORK>/esm/{cifs,manifests}/.

    PYTHONPATH=scripts:config EIB_TASK=dms   python scripts/dms_esm_fold.py --shard_tag 0
    PYTHONPATH=scripts:config EIB_TASK=rsv_f python scripts/dms_esm_fold.py --shard_tag 0
    PYTHONPATH=scripts:config EIB_TASK=niv   python scripts/dms_esm_fold.py --shard_tag 0
"""
import argparse, csv, importlib, os, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
# TASK-AGNOSTIC. A hard-coded `import dms as cfg` / `OUT = work/dms/esm` would
# lock out the RSV F and Nipah G arms. EIB_TASK selects the config exactly as scripts/common.py does,
# and the output tree is derived from that config's WORK. The default is "dms",
# and config/dms.py sets WORK = PROJECT/"work"/"dms", so with EIB_TASK unset (or
# set to dms, which the DMS fold stage does via PYTHONPATH + no EIB_TASK)
# every path and every fold is byte-identical to before -- verified by diffing
# the resolved (cfg module, MANIFEST, ANTIGEN_FASTA, OUT) tuple before and after.
cfg = importlib.import_module(os.environ.get("EIB_TASK", "dms"))     # noqa: E402

OUT = cfg.WORK / "esm"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--num_loops", type=int, default=8)
    ap.add_argument("--num_sampling_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_tag", default="0")
    a = ap.parse_args()

    man = pd.read_csv(cfg.MANIFEST).sort_values("case_id").reset_index(drop=True)
    end = len(man) if a.end < 0 else min(a.end, len(man))
    rows = man.iloc[a.start:end]
    ag = "".join(l.strip() for l in open(cfg.ANTIGEN_FASTA) if not l.startswith(">")).upper()
    (OUT / "cifs").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifests" / f"shard_{a.shard_tag}.csv"
    print(f"shard {a.shard_tag}: rows [{a.start}:{end}) of {len(man)}", flush=True)

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
        vh = str(r.vh_seq).upper()
        vl = str(r.light_seq).upper() if r.has_light and isinstance(r.light_seq, str) else ""
        chains = [ProteinInput(id=cfg.CHAIN_HEAVY, sequence=vh)]
        if vl:
            chains.append(ProteinInput(id=cfg.CHAIN_LIGHT, sequence=vl))
        chains.append(ProteinInput(id=cfg.CHAIN_ANTIGEN, sequence=ag))
        rec = {"case_id": r.case_id, "antibody": r.antibody, "arm": r.arm,
               "complex_len": len(vh) + len(vl) + len(ag), "status": "",
               "iptm": "", "ptm": "", "plddt_mean": "", "error": ""}
        # ESMFold2's tokenizer has no symbol for X; the label vector is indexed
        # against this exact sequence, so a substitution would shift the gold.
        if (set(vh) | set(vl) | set(ag)) - VALID_AA:
            rec["status"] = "nonstandard_residue"; n_fail += 1
        else:
            try:
                spi = StructurePredictionInput(sequences=chains)
                with torch.no_grad():
                    res = builder.fold(model, spi, num_loops=a.num_loops,
                                       num_sampling_steps=a.num_sampling_steps,
                                       num_diffusion_samples=1, seed=a.seed)
                    cif.write_text(res.complex.to_mmcif())
                    rec["iptm"] = float(res.iptm) if res.iptm is not None else ""
                    rec["ptm"] = float(res.ptm) if res.ptm is not None else ""
                    rec["plddt_mean"] = (float(np.asarray(res.plddt).mean())
                                         if res.plddt is not None else "")
                rec["status"] = "ok"; n_ok += 1
            except Exception as e:
                oom = "out of memory" in str(e).lower()
                rec["status"] = "oom" if oom else "fold_error"
                rec["error"] = str(e)[:200]; n_fail += 1
                torch.cuda.empty_cache()
        new = not mpath.exists()
        with open(mpath, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rec))
            if new:
                w.writeheader()
            w.writerow(rec)
        print(f"[{r.case_id}] {rec['status']} len={rec['complex_len']} iptm={rec['iptm']}", flush=True)

    print(f"\nshard {a.shard_tag} done: ok {n_ok}, present {n_skip}, failed {n_fail} -> {mpath}")


if __name__ == "__main__":
    main()
