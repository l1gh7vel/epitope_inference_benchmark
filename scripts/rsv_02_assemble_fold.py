#!/usr/bin/env python3
"""Assemble the AF3 inputs for the RSV F prefusion/postfusion pre-check.

ONE unique chain -- the wild-type Long-strain F ectodomain from
`data/source/dms_simonich2026/F_protein.fasta` (504 aa, sites 26-529) -- so ONE
MSA search (the MSA stage with EIB_TASK=rsv_f), reused by both
assemblies written here:

  rsvF_long_trimer   3 copies, chains A/B/C, 1,512 tokens. THE PRIMARY. F is an
                     obligate trimer and the prefusion state is a trimeric
                     metastable spring; a protomer alone is not the biological
                     object. 1,512 also lands inside the same 1,536-token AF3
                     bucket the rest of the project uses.
  rsvF_long_monomer    1 copy, 504 tokens. Control: it says whether any
                     prefusion/postfusion verdict is carried by the trimer
                     contacts or is already present in one chain.

Both are scored by `rsv_03_conformer.py`, which works per protomer, so the two
are directly comparable.

The F0 construct is left UNCLEAVED and the signal peptide and TM/tail are
absent, because that is exactly the sequence the deep mutational scan measured. Folding the stabilised PDB construct instead
would answer a different question.

    PYTHONPATH=scripts python scripts/rsv_02_assemble_fold.py --out_dir work/rsv_f/af3_in
"""
import argparse, json, os
from pathlib import Path

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
FASTA = PROJECT / "data" / "source" / "dms_simonich2026" / "F_protein.fasta"
WORK = PROJECT / "work" / "rsv_f"
KEY = "ag_rsvF_long"
SEEDS = [1, 2, 3, 4, 5]          # project convention: 5 seeds x 5 diffusion samples


def msa_block():
    """The chain block AF3 echoed back from the MSA-only run, MSAs and templates filled."""
    # AF3 3.0.2 keeps the job name's case in its output directory; the Task II
    # pipeline's stage 02 lowercases it. Accept either rather than depend on it.
    cands = [WORK / "msa_raw" / k / f"{k}_data.json" for k in (KEY, KEY.lower())]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        raise SystemExit("ERROR: MSA not finished -- none of "
                         f"{[str(c) for c in cands]} exist. Run the MSA stage first.")
    obj = json.loads(p.read_text())
    prots = [s["protein"] for s in obj["sequences"] if "protein" in s]
    assert len(prots) == 1, f"expected 1 chain, found {len(prots)}"
    b = prots[0]
    for f in ("unpairedMsa", "pairedMsa", "templates"):
        assert f in b, f"MSA block missing {f}"
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(WORK / "af3_in"))
    args = ap.parse_args()
    seq = "".join(l.strip() for l in FASTA.read_text().splitlines() if not l.startswith(">"))
    b = msa_block()
    assert b["sequence"] == seq, "cached MSA is for a different sequence"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, ids in (("rsvF_long_trimer", ["A", "B", "C"]), ("rsvF_long_monomer", ["A"])):
        entry = {"id": ids, "sequence": seq,
                 **{f: b[f] for f in ("unpairedMsa", "pairedMsa", "templates")}}
        job = {"name": name, "sequences": [{"protein": entry}],
               "modelSeeds": SEEDS, "dialect": "alphafold3", "version": 1}
        (out / f"{name}.json").write_text(json.dumps(job))
        print(f"{name}: {len(ids)} x {len(seq)} = {len(ids)*len(seq)} tokens -> {out/(name+'.json')}")


if __name__ == "__main__":
    main()
