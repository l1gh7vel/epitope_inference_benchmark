#!/usr/bin/env python3
"""ESMFold2 on the wild-type Long F ectodomain -- the second half of the
prefusion/postfusion pre-check.

AF3 and ESMFold2 are independent structure sources in this benchmark, so the
pre-check has to be answered for both: an AF3 prefusion
verdict says nothing about what the ESM column of the RSV arm would be scoring
against. ESMFold2 reads sequence only -- no MSA, no templates -- so it has a
different reason to prefer one conformer than AF3 does.

Folds the monomer and the trimer, matching `rsv_02_assemble_fold.py`, and the
output goes to `rsv_03_conformer.py` with `--offset 25` (the fasta starts at F
site 26). The FOLDING CALL IS COPIED EXACTLY from `dms_esm_fold.py` (num_loops=8,
num_sampling_steps=50, seed=0) so these structures stay comparable with every
other ESMFold2 structure in the project.

    EIB_TASK=rsv_f python scripts/rsv_04_esm_fold.py
"""
import argparse, os
from pathlib import Path
import numpy as np

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
FASTA = PROJECT / "data" / "source" / "dms_simonich2026" / "F_protein.fasta"
OUT = PROJECT / "work" / "rsv_f" / "esm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_loops", type=int, default=8)
    ap.add_argument("--num_sampling_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seq = "".join(l.strip() for l in FASTA.read_text().splitlines()
                  if not l.startswith(">")).upper()
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from esm.models.esmfold2 import (ESMFold2InputBuilder, ProteinInput,
                                     StructurePredictionInput)
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    builder = ESMFold2InputBuilder()
    print("model loaded", flush=True)

    for name, ids in (("rsvF_long_monomer", ["A"]), ("rsvF_long_trimer", ["A", "B", "C"])):
        cif = OUT / f"{name}.cif"
        if cif.exists() and cif.stat().st_size > 0:
            print(f"{name}: present, skipping", flush=True); continue
        spi = StructurePredictionInput(
            sequences=[ProteinInput(id=i, sequence=seq) for i in ids])
        try:
            with torch.no_grad():
                res = builder.fold(model, spi, num_loops=a.num_loops,
                                   num_sampling_steps=a.num_sampling_steps,
                                   num_diffusion_samples=1, seed=a.seed)
            cif.write_text(res.complex.to_mmcif())
            plddt = float(np.asarray(res.plddt).mean()) if res.plddt is not None else float("nan")
            print(f"{name}: ok  len={len(ids)*len(seq)}  plddt={plddt:.1f}  ptm={res.ptm}", flush=True)
        except Exception as e:
            print(f"{name}: FAILED {str(e)[:300]}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
