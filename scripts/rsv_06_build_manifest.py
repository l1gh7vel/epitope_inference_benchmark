#!/usr/bin/env python3
"""Fold manifest for the RSV F arm: ONE ROW PER (antibody, chain arm). 6 x 2 = 12.

The RSV equivalent of `dms_02_build_manifest.py`, and it carries the same columns
so the two DMS arms can be read side by side. What is added is the three things
that are uniform in the HIV arm and are NOT uniform here, each as its own column
rather than as a caveat in a legend (`config/rsv_f.py` argues all three):
`has_gold`, `leakage`, `measured_null`.

What one row is
---------------
One (antibody, chain arm) pair = one AF3/ESMFold2 complex to fold and one
prediction to score. Two arms per antibody, never mixed:

  vhvl  heavy + light, the Task II convention. All six antibodies have BOTH
        chains published, so this is the models' best shot.
  vh    heavy only, the Task III convention, so this arm can be read against the
        existing HIV Env numbers.

Running both separates "the model cannot condition on the antibody" from "we
crippled its input" -- the confound the VH-only HIV Env results would otherwise
carry into any cross-arm statement.

THE ONE NAMING RECONCILIATION, made explicitly and recorded in the manifest
--------------------------------------------------------------------------
`RSV_mAb_AAseq.csv` has a UTF-8 BOM on its header (so a naive read gives a column
called '\\ufeffmAb' and every lookup misses), and it names one antibody

    "MK-1654 (Clesrovimab)"          <- the antibody-sequence CSV
    "Clesrovimab"                    <- the escape tables, the figures, and
                                        every other script in this arm
    "RB1"                            <- what the PDB calls it in 6OUS

all three being the same molecule (MK-1654 = clesrovimab = RB1). The mapping is a
one-entry dict (`CSV_NAME`), not a startswith/contains rule, so a file that later
gains another MK- row cannot silently redirect it. The CSV spelling is kept in the
manifest column `mab_csv_name` so the join back to the source file is auditable.
The other five names are byte-identical across the two files.

Chain trimming, a no-op here, checked rather than assumed
-----------------------------------------------------------------
`trim_heavy_to_variable` / `trim_light_to_variable` are applied anyway, because
they are what Tasks II/III use and the manifest must be produced the same way.
Measured: all twelve chains pass through untouched (`vh_trimmed`/`light_trimmed`
are False for every row) -- the published CSV already carries variable domains,
120-128 aa VH and 106-113 aa VL. So AF3 sees an Fv, the project convention,
and it sees it without any cut being made.

A REAL FINDING FROM THE DOMAIN CHECK, worth not rediscovering
--------------------------------------------------------------
`domain_kind` classifies 11 of the 12 chains correctly and returns **unknown for
1A2's light chain**. The data are correct and the row is kept: 1A2's
framework-2 reads **WYLHK** where the canonical kappa motif is WYQQK, so
`common.FR2_LIGHT` (`W[YF][QLH]Q`) does not fire. The sequence is a bona fide
kappa V domain -- it is 113 aa, it carries the C23/C88 cysteines and the
FGPGTKVDIR J-region, and chain L of 9LLY matches it at coverage 1.00. The column
`domain_kind` records what the classifier said so the anomaly is visible; it is
reported, never used as a filter.

Token budget, for both assemblies
----------------------------------------------------------
`n_tokens_monomer` = 504 + VH (+ VL) ~= 630-745, inside the project's 1,536 bucket.
`n_tokens_trimer`  = 1,512 + VH (+ VL) ~= 1,640-1,753, ABOVE it.
`config/rsv_f.py` states the measurement that decides between them
(`cross_protomer_frac` 0.04 / 0.19 / 0.21 / 0.26) and the code change the trimer
would require in stage 04. The MSA stage is identical either way, so this does
not block it. `n_tokens` is an alias of whichever `cfg.ANTIGEN_ASSEMBLY` names,
so a consumer that just wants "the" number gets the declared one.

    PYTHONPATH=scripts:config EIB_TASK=rsv_f python scripts/rsv_06_build_manifest.py
"""
import csv, io, json, os, sys
from pathlib import Path

import pandas as pd

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
import rsv_f as cfg                                                   # noqa: E402
from common import (chain_key, clean_seq, domain_kind, read_fasta_single,   # noqa: E402
                    trim_heavy_to_variable, trim_light_to_variable)

ARMS = ("vhvl", "vh")

# The evaluated set, in the order rsv_01_escape_sites.py declares it. RSM01 is the
# seventh antibody in the scan and is absent on purpose: no public sequence.
ANTIBODIES = ["Nirsevimab", "Clesrovimab", "Palivizumab", "Suptavumab", "1A2", "1B6"]

# See the docstring. One entry, stated, not inferred.
CSV_NAME = {"Clesrovimab": "MK-1654 (Clesrovimab)"}


def mab_sequences():
    """mAb -> {'VH','VL'} from RSV_mAb_AAseq.csv. The header carries a UTF-8 BOM."""
    raw = cfg.MAB_CSV.read_text().lstrip("﻿")
    return {r["mAb"]: r for r in csv.DictReader(io.StringIO(raw))}


def main():
    gold = json.loads(cfg.GOLD_REPORT.read_text())
    esc = json.loads((cfg.MANIFESTS / "rsv_escape_sites.json").read_text())
    mabs = mab_sequences()
    ag = read_fasta_single(cfg.ANTIGEN_FASTA)
    assert len(ag) == cfg.ANTIGEN_LEN, f"antigen is {len(ag)} aa, expected {cfg.ANTIGEN_LEN}"
    assert esc["topk"] == cfg.ESCAPE_TOPK, \
        f"escape set was built at k={esc['topk']} but config declares {cfg.ESCAPE_TOPK}"

    rows = []
    for ab in ANTIBODIES:
        csv_name = CSV_NAME.get(ab, ab)
        if csv_name not in mabs:
            raise SystemExit(f"{ab}: no row '{csv_name}' in {cfg.MAB_CSV}. "
                             "If the file changed, fix CSV_NAME explicitly -- do not "
                             "add a fuzzy match.")
        g = gold[ab]
        e = esc["per_antibody"][ab]

        h_raw, h_bad = clean_seq(mabs[csv_name]["VH"])
        l_raw, l_bad = clean_seq(mabs[csv_name]["VL"])
        assert not (h_bad or l_bad), f"{ab}: non-standard residues in the published sequence"
        vh, vh_trimmed = trim_heavy_to_variable(h_raw)
        vl, vl_trimmed = trim_light_to_variable(l_raw)

        for arm in ARMS:
            has_light = arm == "vhvl"
            n_ab = len(vh) + (len(vl) if has_light else 0)
            rows.append(dict(
                case_id=f"{ab}__{arm}", antibody=ab, arm=arm,
                mab_csv_name=csv_name,
                pdb=g["pdb"] or "", resolution=g["resolution"] or "",
                epitope_class=g["epitope_class"],
                vh_seq=vh, vh_len=len(vh), vh_trimmed=vh_trimmed,
                vh_key=chain_key("vh", vh), vh_domain_kind=domain_kind(vh),
                light_seq=vl if has_light else "",
                light_len=len(vl) if has_light else 0,
                light_trimmed=vl_trimmed if has_light else "",
                light_key=chain_key("vl", vl) if has_light else "",
                light_domain_kind=domain_kind(vl) if has_light else "",
                has_light=has_light,
                # --- structural gold standard: present for only 4 of the 6 ---
                has_gold=g["has_gold"], n_gold=g["n_gold"],
                gold_definition=g.get("headline", ""),
                cross_protomer_frac=g.get("cross_protomer_frac", ""),
                # --- escape target set: present for all 6, this arm's headline ---
                n_escape=e["n_sites"], escape_rule=e["rule"],
                escape_own_region=e["own_region"],
                escape_in_own_region=e["n_in_own_region"],
                # --- the three non-uniformities ---
                leakage=g["leakage"],
                af3_template_pre_cutoff=g["af3_template_pre_cutoff"],
                measured_null=(ab in cfg.MEASURED_NULL),
                # --- token budget, both assemblies (see the docstring) ---
                n_tokens_monomer=cfg.ANTIGEN_LEN + n_ab,
                n_tokens_trimer=3 * cfg.ANTIGEN_LEN + n_ab,
                n_tokens=cfg.ANTIGEN_COPIES * cfg.ANTIGEN_LEN + n_ab,
                antigen_assembly=cfg.ANTIGEN_ASSEMBLY))

    man = pd.DataFrame(rows)
    cfg.MANIFESTS.mkdir(parents=True, exist_ok=True)
    man.to_csv(cfg.MANIFEST, index=False)

    print(man[["case_id", "arm", "vh_len", "light_len", "has_gold", "n_gold",
               "n_escape", "leakage", "measured_null", "n_tokens_monomer",
               "n_tokens_trimer"]].to_string(index=False))
    chains = sorted(set(man.vh_key) | {k for k in man.light_key if k})
    print(f"\n{len(man)} fold cases over {man.antibody.nunique()} antibodies x "
          f"{len(ARMS)} arms; {len(chains)} distinct antibody chains needing an MSA")
    print(f"antigen chain {cfg.ANTIGEN_CHAIN_KEY} is already searched "
          f"({'present' if (cfg.MSA_RAW / cfg.ANTIGEN_CHAIN_KEY).exists() else 'MISSING'} "
          f"in {cfg.MSA_RAW})")
    print(f"gold structural epitope for {int(man.has_gold.sum()) // len(ARMS)} of "
          f"{man.antibody.nunique()} antibodies; "
          f"missing for {sorted(set(man[~man.has_gold].antibody))}")
    print(f"leakage: {man.drop_duplicates('antibody').leakage.value_counts().to_dict()}")
    print(f"tokens: monomer max {man.n_tokens_monomer.max()}, "
          f"trimer max {man.n_tokens_trimer.max()} (AF3 bucket 1536)")
    print(f"\n-> {cfg.MANIFEST}")


if __name__ == "__main__":
    main()
