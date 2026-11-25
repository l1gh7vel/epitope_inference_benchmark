#!/usr/bin/env python3
"""Derive the SAbDab counts that Fig 2d(iii) prints, rather than hardcoding them.

The counts are derived here so the filtering rule can be checked and the numbers
regenerate when SAbDab updates. The question they must answer is whether the
SAbDab total is filtered for proper (protein) antigens rather than lipids and
other non-protein antigens.

`antigen_type` is a pipe-separated multi-value field rather than a single label:
PROTEIN, PROTEIN|SUGAR, PROTEIN|HAPTEN|ION and so on. The filter is therefore "the
antigen has a protein component", PROTEIN being one of the tokens, rather than
`antigen_type == "PROTEIN"`.

That distinction decides whether this benchmark's two main antigens are in the
count at all. Measured on the vendored 2026-10-28 summary:

    rule                                   unique PDB entries
    antigen_type == "PROTEIN" exactly                   5,034
    antigen_type CONTAINS "PROTEIN"                     6,694   <- what we print
    any antigen of any type                             8,497

Strict equality drops every glycoprotein, because a glycosylated antigen is typed
PROTEIN|SUGAR. For HIV-1 Env that is 1,162 instances down to 353, and for
SARS-CoV-2 spike 2,628 down to 1,974. A "protein antigen" filter that discards
most of Env is the wrong filter.

What is excluded: 1,822 PDB entries whose antigen has no protein component --
peptides (993 instances), haptens (664), ions (303), sugars (114) and their
combinations. There is no LIPID token in SAbDab's vocabulary; the non-protein
vocabulary is PEPTIDE / HAPTEN / ION / SUGAR / NUCLEIC-ACID. The count is
filtered, and the small-molecule and sugar-only complexes are out.

    python scripts/32_sabdab_counts.py
"""
import os
from pathlib import Path

import pandas as pd

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
SRC = PROJECT / "data" / "source" / "sabdab_2026-11-01" / "sabdab_all_summary.csv.gz"
OUT = PROJECT / "results" / "task1" / "sabdab_counts.csv"


def load():
    # on_bad_lines="skip": a handful of rows carry an unescaped comma inside a
    # free-text field. They are dropped rather than repaired, because editing a
    # vendored upstream file costs more than losing a few rows from a count that
    # is already quoted to the nearest entry.
    return pd.read_csv(SRC, sep=",", low_memory=False, on_bad_lines="skip")


def main():
    d = load()
    paired = d.VH.notna() & d.VL.notna()
    tokens = d.antigen_type.fillna("").str.upper().str.split("|")
    has_protein = tokens.apply(lambda t: "PROTEIN" in t)
    has_antigen = d.antigen_type.notna()

    rows = [
        ("all instances", "instances", len(d)),
        ("all PDB entries", "PDB entries", d.PDB.nunique()),
        ("with an antigen of any type", "instances", int(has_antigen.sum())),
        ("paired VH+VL", "PDB entries", d[paired].PDB.nunique()),
        ("paired VH+VL, any antigen", "PDB entries",
         d[paired & has_antigen].PDB.nunique()),
        ("paired VH+VL, PROTEIN component", "instances",
         int((paired & has_protein).sum())),
        ("paired VH+VL, PROTEIN component", "PDB entries",
         d[paired & has_protein].PDB.nunique()),
        ("excluded: no protein component", "PDB entries",
         d[paired & has_antigen & ~has_protein].PDB.nunique()),
    ]
    out = pd.DataFrame(rows, columns=["set", "unit", "n"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"SAbDab, vendored summary {SRC.name}")
    for s, u, n in rows:
        print(f"  {s:<34} {n:>7,}  {u}")

    headline = d[paired & has_protein].PDB.nunique()
    print(f"\n  Fig 2d(iii) prints {headline:,} PDB entries. "
          f"{'MATCHES' if headline == 6694 else 'DOES NOT MATCH'} the hardcoded 6,694.")

    print("\n  excluded antigen types (no protein component):")
    exc = d[paired & has_antigen & ~has_protein].antigen_type.value_counts()
    for t, n in exc.head(8).items():
        print(f"    {t:<18} {n:>5} instances")

    print("\n  why CONTAINS and not ==, on this benchmark's own antigens:")
    strict = d.antigen_type.fillna("").str.upper().eq("PROTEIN")
    for name in ("spike", "envelope glycoprotein"):
        m = d.antigen_name.fillna("").str.lower().str.contains(name)
        print(f"    {name:<22} kept {int((m & paired & has_protein).sum()):>5}"
              f"   under == only {int((m & paired & strict).sum()):>5}")
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    main()
