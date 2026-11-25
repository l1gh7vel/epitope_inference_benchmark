#!/usr/bin/env python3
"""Fold manifest for the Nipah RBP escape arm: one row per (antibody, chain arm).

Modelled on `scripts/dms_02_build_manifest.py`. Two arms per antibody, never
mixed:
  vhvl -- heavy + light, the Task II convention. Both chains are solved for all
          four antibodies, so this is the models' best shot.
  vh   -- heavy only, the Task III convention, so the panel can be read against
          the existing HIV Env and RSV numbers.
Running both separates "the model cannot condition on the antibody" from "we
crippled its input".

*** n = 4 AND NO SIGNIFICANCE TEST. *** Only four of the six scanned antibodies
have a recoverable (VH, VL). A one-sided Wilcoxon signed-rank at n = 4 has a
minimum attainable P of 1/2^4 = 0.0625, so no test on this arm can reach 0.05
however large the effect. This is built to be presented as a CASE ILLUSTRATION,
per antibody, with no test attached. Do not add one.

Where the sequences come from, and how each one is checked
---------------------------------------------------------------------
The paper publishes NO antibody sequence -- all six rows of its Key Resources
Table carry identifier "N/A" (PROVENANCE.md). Every sequence here is read out of
a deposited structure's `_entity_poly.pdbx_seq_one_letter_code_can`, i.e. the
full construct, NOT the modelled residues, so chain breaks in the crystal do not
silently delete residues from the Fv. Release dates, source organisms and the
antigen's species are all read from the same CIF rather than hardcoded, so the
leakage flags cannot drift from the files.

    antibody   PDB    antibody chains   antigen in THAT structure   species
    nAH1.3     7TXZ   E (H) / F (L)     Nipah                       MURINE
    HENV-26    6VY5   H / L             Nipah                       human
    HENV-32    6VY4   C (H) / D (L)     **HENDRA**                  human
    m102.4     6CMG   C (H) / B (L)     **HENDRA**                  human

HENV-103 and HENV-117 are measured by the DMS and CANNOT be folded: no PDB entry
and no published sequence. They are written into `niv_escape_sites.*` and are
absent here. That is the whole reason this arm is n = 4 and not n = 6.

m102.4 IS A PROXY ON ITS LIGHT CHAIN AND THE MANIFEST SAYS SO
--------------------------------------------------------------
6CMG is m102.**3**, not m102.4. Per Xu et al. (PLoS Pathog 2013, PMC3795035) the
two share an IDENTICAL heavy chain and their light chains differ at 15 positions;
the true m102.4 VL exists only as a TIFF sequence alignment in that paper's
Figure S6, which has not been read. So:

  * the `vh` arm of m102.4 is EXACT -- the heavy chain is m102.4's own;
  * the `vhvl` arm of m102.4 is a PROXY, and `sequence_is_proxy` is True on that
    row only, with `proxy_note` carrying the 15-position caveat.

Never substitute m102.3 for m102.4 silently. If m102.4 moves a conclusion, the
proxy flag is the first thing to quote.

LEAKAGE: 3 OF 4 ARE TEMPLATE-VISIBLE, AND A CLEAN SUBSET DOES NOT EXIST
------------------------------------------------------------------------
Initial release dates against AF3's 2021-09-30 cutoff (the run keeps the
default and reports leakage rather than suppressing it):

    7TXZ 2022-03-09  CLEAN      6VY5 2021-01-06  LEAKED
    6VY4 2020-12-30  LEAKED     6CMG 2018-04-18  LEAKED

One clean antibody is not a subset. Report the WALLE/AF3 ratio, never
clean-vs-whole absolutes, because the clean subset is intrinsically easier. State
the 3/4 on any number from this arm.

Chain trimming
--------------
Antibody chains are cut to the variable domain so AF3 sees an Fv, not a Fab
(the project convention). `common.trim_light_to_variable` handles all four light chains
unchanged (human kappa PSVFIFPPS, human lambda PTVTLFPPS, murine kappa
PTVSIFPPS all match its CL motif). The HEAVY side needs a wider net than
`common.CH1_START`, which anchors on `STKGPSV[FL]PLAP`, and TWO of these four
heavy chains do not match it:

  * 7TXZ is a full-length MURINE IgG (458 aa, with a C-terminal StrepII tag).
    Its CH1 begins `AKTTPPSVYPLAPG` -- no STKGPSV at all.
  * 6VY4's heavy constant begins `ASFKGPSVFPLAPSS` -- an F where the motif wants
    a T.

The extra motifs are defined HERE rather than in `common.py` on purpose: that
regex is load-bearing for Tasks I/II/III and the DMS arm, where ~150 rows already
depend on its exact behaviour, and widening it would silently re-trim them. The
local regex is applied only as a FALLBACK, after `common.trim_heavy_to_variable`
has had its chance, so any chain the shared helper already handles is trimmed
identically to every other task. Every row records `vh_trim_rule` saying which
path it took, and the script asserts that each trimmed VH ends in the J-segment
motif `[WF]G.G[TA]` -- a trim that silently did nothing would otherwise send a
450 aa IgG to AF3 and look merely expensive.

    PYTHONPATH=scripts python scripts/niv_02_build_manifest.py
    EIB_TASK=niv PYTHONPATH=scripts python scripts/01_make_chain_jobs.py
"""
import json
import re
import sys
from pathlib import Path

import gemmi
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (chain_key, clean_seq,  # noqa: E402
                    trim_heavy_to_variable, trim_light_to_variable)
# The antigen is read through niv_01, because read_fasta_single fails here: protein.fasta
# ends in a `*` stop codon, AF3 rejects it ("Protein must contain only letters"),
# and a mask built on len(fasta) would be one position too long. One definition.
from niv_01_escape_sites import antigen_sequence  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
import niv as cfg  # noqa: E402

ARMS = ("vhvl", "vh")

# Heavy constant-region N-termini that `common.CH1_START` does not cover.
# FALLBACK ONLY -- see the docstring for why this is not fixed in common.py.
#   AKTTPPSV[YF]PLAP  murine IgG1 CH1 (7TXZ)
#   [A-Z]{2}[TF]KGPSV[FL]PLAP  human IgG1 CH1 incl. the ASFKGPSV variant (6VY4)
CH1_EXTRA = re.compile(r"AKTTPPSV[YF]PLAP|[A-Z]{2}[TF]KGPSV[FL]PLAP")
# J-segment / FR4 end of a V domain. Used as an assertion, not as a cut point.
J_END = re.compile(r"[WF]G.G[TA]")

# Chain roles read off `_entity_poly` (see work/niv/ref/README.md), never guessed
# from the letters. Where a structure holds two copies of the Fab, the first is
# taken; the entity sequence is identical between copies by construction.
CASES = {
    "nAH1.3":  dict(cif="7txz.cif", heavy="E", light="F",
                    proxy=False, proxy_note=None),
    "HENV-26": dict(cif="6vy5.cif", heavy="H", light="L",
                    proxy=False, proxy_note=None),
    "HENV-32": dict(cif="6vy4.cif", heavy="C", light="D",
                    proxy=False, proxy_note=None),
    "m102.4":  dict(cif="6cmg.cif", heavy="C", light="B",
                    proxy=True,
                    proxy_note="6CMG is m102.3, not m102.4. Heavy chain IDENTICAL "
                               "(so the vh arm is exact); light chains differ at 15 "
                               "positions (Xu et al., PLoS Pathog 2013, PMC3795035). "
                               "True m102.4 VL exists only as a TIFF alignment in that "
                               "paper's Fig S6, unread."),
}
# Antigen chain in each structure, used only to report which virus it is.
ANTIGEN_CHAIN = {"7txz.cif": "A", "6vy5.cif": "A", "6vy4.cif": "A", "6cmg.cif": "A"}
# Which species each source organism string means for the antigen.
SPECIES_OF = {"nipah henipavirus": "Nipah", "hendra henipavirus": "Hendra",
              "hendra virus": "Hendra"}


def cif_facts(path):
    """Release date, deposition date, and per-chain (sequence, source organism)."""
    block = gemmi.cif.read(str(path)).sole_block()
    revisions = [r.str(0) for r in
                 block.find("_pdbx_audit_revision_history.", ["revision_date"])]
    deposited = block.find_value("_pdbx_database_status.recvd_initial_deposition_date")
    src = {}
    for r in block.find("_entity_src_gen.", ["entity_id", "pdbx_gene_src_scientific_name"]):
        src[r.str(0)] = r.str(1)
    chains, organisms = {}, {}
    for r in block.find("_entity_poly.", ["entity_id", "pdbx_strand_id",
                                          "pdbx_seq_one_letter_code_can"]):
        seq = "".join(r.str(2).split())
        for c in r.str(1).split(","):
            chains[c.strip()] = seq
            organisms[c.strip()] = src.get(r.str(0), "?")
    return dict(released=min(revisions) if revisions else None,
                deposited=(deposited or "").strip("?'\" ") or None,
                chains=chains, organisms=organisms)


def trim_heavy(seq):
    """Variable domain of a heavy chain. Shared helper first, local fallback second."""
    vh, trimmed = trim_heavy_to_variable(seq)
    if trimmed:
        return vh, "common.CH1_START"
    m = CH1_EXTRA.search(seq)
    if m:
        return seq[:m.start()], "niv_02.CH1_EXTRA"
    return seq, "none (already a V domain)"


def main():
    antigen = antigen_sequence()
    # The shared stages read cfg.ANTIGEN_FASTA with common.read_fasta_single, which
    # does not strip the vendored fasta's trailing `*`. Write the cleaned copy here
    # so stage 01 and stage 03 cannot pick up the stop codon. See config/niv.py.
    cfg.ANTIGEN_FASTA.parent.mkdir(parents=True, exist_ok=True)
    cfg.ANTIGEN_FASTA.write_text(f">niv_rbp_ectodomain_71_602\n{antigen}\n")
    assert len(antigen) == cfg.ANTIGEN_LEN, f"antigen is {len(antigen)} aa, expected {cfg.ANTIGEN_LEN}"
    gold = json.loads(cfg.ESCAPE_JSON.read_text())
    assert gold["topk"] == 10, "escape rule changed; rerun niv_01 and re-read this manifest"

    rows = []
    for ab, spec in CASES.items():
        assert ab in gold["foldable"], f"{ab} is not in the foldable list of niv_01"
        facts = cif_facts(cfg.REF_DIR / spec["cif"])
        pdb = spec["cif"][:4].upper()

        h_raw, _ = clean_seq(facts["chains"][spec["heavy"]])
        l_raw, _ = clean_seq(facts["chains"][spec["light"]])
        vh, vh_rule = trim_heavy(h_raw)
        vl, vl_trimmed = trim_light_to_variable(l_raw)
        for name, s in (("VH", vh), ("VL", vl)):
            assert J_END.search(s[-18:]), \
                f"{ab} {name} does not end in a J-segment motif after trimming: {s[-25:]}"
            assert 95 <= len(s) <= 140, f"{ab} {name} is {len(s)} aa after trimming"

        ag_org = facts["organisms"].get(ANTIGEN_CHAIN[spec["cif"]], "?").lower()
        antigen_species = SPECIES_OF.get(ag_org, ag_org)
        ab_org = facts["organisms"].get(spec["heavy"], "?")
        leaked = facts["released"] <= cfg.MAX_TEMPLATE_DATE

        for arm in ARMS:
            has_light = arm == "vhvl"
            # m102.4's HEAVY chain is m102.4's own; only the light chain is m102.3's,
            # so the vh arm is exact and must not be flagged as a proxy.
            proxy = spec["proxy"] and has_light
            rows.append(dict(
                case_id=f"{ab}__{arm}", antibody=ab, arm=arm,
                pdb=pdb, heavy_chain=spec["heavy"],
                light_chain=spec["light"] if has_light else "",
                antigen_in_structure=antigen_species,
                antibody_species=ab_org,
                pdb_deposited=facts["deposited"], pdb_released=facts["released"],
                af3_template_cutoff=cfg.MAX_TEMPLATE_DATE, leaked=bool(leaked),
                sequence_is_proxy=bool(proxy),
                proxy_note=(spec["proxy_note"] if proxy else ""),
                vh_seq=vh, vh_len=len(vh), vh_trim_rule=vh_rule,
                vh_key=chain_key("vh", vh),
                light_seq=vl if has_light else "",
                light_len=len(vl) if has_light else 0,
                light_trimmed=bool(vl_trimmed) if has_light else False,
                light_key=chain_key("vl", vl) if has_light else "",
                has_light=has_light,
                n_gold=gold["per_antibody"][ab]["n_sites"],
                n_tokens=len(antigen) + len(vh) + (len(vl) if has_light else 0)))

    man = pd.DataFrame(rows)
    cfg.MANIFESTS.mkdir(parents=True, exist_ok=True)
    man.to_csv(cfg.MANIFEST, index=False)

    cols = ["case_id", "arm", "pdb", "antigen_in_structure", "antibody_species",
            "pdb_released", "leaked", "sequence_is_proxy", "vh_len", "vh_trim_rule",
            "light_len", "n_gold", "n_tokens"]
    print(man[cols].to_string(index=False))
    chains = set(man.vh_key) | {k for k in man.light_key if k}
    print(f"\n{len(man)} fold cases ({man.antibody.nunique()} antibodies x {len(ARMS)} arms), "
          f"{len(chains)} distinct antibody chains needing an MSA")
    print(f"antigen chain {chain_key('ag', antigen)} ({len(antigen)} aa) needs one more")
    print(f"max tokens {man.n_tokens.max()} (AF3 bucket 1536)")
    print(f"leakage: {int(man[man.arm=='vhvl'].leaked.sum())}/{man.antibody.nunique()} "
          f"antibodies are template-visible at {cfg.MAX_TEMPLATE_DATE} -- "
          "report the WALLE/AF3 ratio, not clean-vs-whole absolutes")
    print("n = 4 antibodies. NO significance test is attached to this arm.")
    print(f"\nwrote {cfg.MANIFEST}")


if __name__ == "__main__":
    main()
