#!/usr/bin/env python
"""
CDR spans for every distinct antibody heavy chain in the project.

What this produces
---------------
The planned CDR-only pooling analysis mean-pools ESM-2 embeddings over CDR
positions instead of over all ~120 VH positions, to test whether framework
residues swamp the CDR signal.  That needs, for each VH sequence, the index
range of CDR-H1/H2/H3 *inside that exact sequence string* -- not inside some
renumbered or trimmed version of it.  Nothing in this repository commits to an
antibody numbering scheme, so this script is where the numbering enters.

INTERPRETER (required -- the analysis env has no ANARCI/abnumber/HMMER)
-----------------------------------------------------------------------
    $EIB_PYTHON_GPU scripts/28_cdr_spans.py

The `walle` env carries the `anarci` and `abnumber` packages and the HMMER
suite.  ANARCI shells out to `hmmscan`, so that env's `bin/` must be on PATH;
the script prepends it itself (derived from sys.executable), so no `conda
activate` is needed.

What one row is
---------------
One row = one DISTINCT VH sequence under ONE numbering scheme.  A sequence that
appears in many manifest rows is numbered once.  Both schemes are emitted for
every sequence, so each sequence contributes exactly two rows (scheme='imgt'
and scheme='chothia').  No "recommended" scheme is chosen here; the two differ
systematically (Chothia's CDR definition is tighter) and the caller decides.

Indices are 0-based into the input `vh_seq` string, and **ends are EXCLUSIVE**
(python slice convention): `vh_seq[cdr1_start:cdr1_end]` IS CDR-H1.

The self-check, and the code structure it requires
----------------------------------------------
abnumber's `Chain.seq` is NOT the input string.  ANARCI numbers only the
variable domain, so abnumber silently drops any residue N-terminal to the first
numbered position and parks the C-terminal remainder in `Chain.tail`.  Positions
are therefore indices into `Chain.seq`, which can be offset from the input.
That offset is exactly the kind of silent frame shift that would make a
CDR-pooling result look like a real signal while pooling framework residues.

So the code never trusts the tool's indices:

  1. regions come from `Position.get_region()` over the ordered `Chain.positions`,
     giving an index range inside `Chain.seq`;
  2. the offset of `Chain.seq` inside the input is recovered by exact substring
     search, and REJECTED unless it occurs exactly once;
  3. every emitted span is then re-read out of the input and compared, residue
     for residue, against the CDR string the tool reported
     (`vh_seq[start:end] == chain.cdrN_seq`).

A sequence that fails numbering, or fails any of those checks, is written with
EMPTY spans, `ok=0` and a `note` giving the reason.  Nothing is dropped
silently, because a dropped antibody is an invisible cohort change -- and this
project's HIV arm is full of heavily somatically hypermutated bnAbs on which
motif-based heavy-chain detection already fails at ~22%.  The per-scheme
failure count printed at the end is a result, not a diagnostic.

`vh_key` uses the repo's existing convention, `scripts/common.chain_key("vh",
seq)` = `"vh_" + sha1(seq)[:10]`, so these tables join straight onto the
manifests' own `vh_key` column.

INPUTS
------
Every manifest under data/manifests/ carrying a `vh_seq` column (task1, task2,
task3, task3cov, task3covvl, dms).  Missing files are reported and skipped.

OUTPUTS
-------
data/manifests/cdr_spans.csv
    vh_key, vh_len, scheme, cdr1_start, cdr1_end, cdr2_start, cdr2_end,
    cdr3_start, cdr3_end, cdr_total_len, ok, note
data/manifests/cdr_spans_seqs.csv
    vh_key, vh_seq   (so downstream code joins without re-reading every manifest)

Usage:
    $EIB_PYTHON_GPU scripts/28_cdr_spans.py
"""

import hashlib
import os
import sys
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

# ANARCI shells out to hmmscan; put this interpreter's own bin/ on PATH so the
# script runs without `conda activate`.
_BIN = str(Path(sys.executable).parent)
os.environ["PATH"] = _BIN + os.pathsep + os.environ.get("PATH", "")

import pandas as pd  # noqa: E402
from abnumber.chain import Chain  # noqa: E402

# abnumber warns once per batch about multi-domain constructs; the same fact is
# recorded per-sequence in the `note` column, so the stderr copy is just noise.
warnings.filterwarnings("ignore", message=".*multiple_domains.*")

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "data" / "manifests"
MANIFESTS = [
    "task1_manifest.csv",
    "task2_manifest.csv",
    "task3_manifest.csv",
    "task3cov_manifest.csv",
    "task3covvl_manifest.csv",
    "dms_manifest.csv",
]
SCHEMES = ["imgt", "chothia"]
BATCH = 400  # sequences per ANARCI/hmmscan call

OUT_SPANS = MANIFEST_DIR / "cdr_spans.csv"
OUT_SEQS = MANIFEST_DIR / "cdr_spans_seqs.csv"

COLUMNS = [
    "vh_key", "vh_len", "scheme",
    "cdr1_start", "cdr1_end",
    "cdr2_start", "cdr2_end",
    "cdr3_start", "cdr3_end",
    "cdr_total_len", "ok", "note",
]


def chain_key(kind, seq):
    """Same convention as scripts/common.chain_key -- 'vh_<sha1[:10]>'."""
    return f"{kind}_{hashlib.sha1(seq.encode()).hexdigest()[:10]}"


def collect_sequences():
    """Distinct vh_seq across every manifest that has the column."""
    seqs = OrderedDict()   # vh_key -> seq
    report = []
    for name in MANIFESTS:
        path = MANIFEST_DIR / name
        if not path.exists():
            report.append((name, "MISSING", 0, 0))
            continue
        df = pd.read_csv(path)
        if "vh_seq" not in df.columns:
            report.append((name, "no vh_seq column", len(df), 0))
            continue
        vals = [str(s).strip().upper() for s in df["vh_seq"].dropna()]
        vals = [s for s in vals if s and s != "NAN"]
        for s in vals:
            seqs.setdefault(chain_key("vh", s), s)
        report.append((name, "ok", len(df), len(set(vals))))
    return seqs, report


def region_span(chain):
    """
    Index range of each CDR inside chain.seq, from the ordered position list.

    Returns {region: (start, end_exclusive)} and raises ValueError if a region's
    positions are not contiguous (which would make a single span meaningless).
    """
    spans = {}
    for i, pos in enumerate(chain.positions):
        reg = pos.get_region()
        if reg not in ("CDR1", "CDR2", "CDR3"):
            continue
        if reg not in spans:
            spans[reg] = [i, i + 1]
        else:
            if i != spans[reg][1]:
                raise ValueError(f"{reg} positions are not contiguous")
            spans[reg][1] = i + 1
    return {k: tuple(v) for k, v in spans.items()}


def spans_for_chain(seq, chain):
    """
    Verified 0-based, end-exclusive spans into `seq`.

    Raises ValueError with a human-readable reason on any failure; the caller
    turns that into an ok=0 row rather than dropping the sequence.
    """
    numbered = chain.seq
    if not numbered:
        # Seen on the ultralong (50-60 aa) bovine-type CDR-H3 chains under
        # Chothia: the scheme has no room for the insertion, so ANARCI returns
        # zero numbered positions rather than an error. Catch it explicitly --
        # an empty string "occurs" everywhere and would otherwise be reported
        # as an ambiguous offset.
        raise ValueError("scheme produced an empty numbering (0 numbered positions)")
    occurrences = seq.count(numbered)
    if occurrences == 0:
        raise ValueError("numbered region is not a substring of vh_seq")
    if occurrences > 1:
        raise ValueError("numbered region occurs more than once in vh_seq (ambiguous offset)")
    offset = seq.index(numbered)

    local = region_span(chain)
    reported = {
        "CDR1": chain.cdr1_seq,
        "CDR2": chain.cdr2_seq,
        "CDR3": chain.cdr3_seq,
    }

    out = {}
    for reg in ("CDR1", "CDR2", "CDR3"):
        if reg not in local:
            raise ValueError(f"{reg} has no numbered positions")
        s, e = local[reg]
        gs, ge = s + offset, e + offset
        # THE self-check: read the span back out of the INPUT string.
        if seq[gs:ge] != reported[reg]:
            raise ValueError(
                f"{reg} self-check failed: vh_seq[{gs}:{ge}]={seq[gs:ge]!r} "
                f"!= tool {reported[reg]!r}"
            )
        out[reg] = (gs, ge)
    return out


def blank_row(key, seq, scheme, note):
    return {
        "vh_key": key, "vh_len": len(seq), "scheme": scheme,
        "cdr1_start": "", "cdr1_end": "",
        "cdr2_start": "", "cdr2_end": "",
        "cdr3_start": "", "cdr3_end": "",
        "cdr_total_len": "", "ok": 0, "note": note,
    }


def run_scheme(seqs, scheme):
    """Number every sequence under one scheme; return list of row dicts."""
    rows = []
    keys = list(seqs)
    for i in range(0, len(keys), BATCH):
        chunk = keys[i:i + BATCH]
        sub = {k: seqs[k] for k in chunk}
        try:
            chains, errors = Chain.batch(sub, scheme=scheme)
        except Exception as exc:                      # whole-batch failure
            for k in chunk:
                rows.append(blank_row(k, seqs[k], scheme, f"batch error: {exc}"))
            continue
        for k in chunk:
            seq = seqs[k]
            if k not in chains or chains[k] is None:
                reason = str(errors.get(k, "ANARCI returned no numbering")).strip()
                rows.append(blank_row(k, seq, scheme, f"anarci: {reason}"))
                continue
            chain = chains[k]
            try:
                sp = spans_for_chain(seq, chain)
            except Exception as exc:
                rows.append(blank_row(k, seq, scheme, str(exc)))
                continue
            total = sum(e - s for s, e in sp.values())
            note = ""
            ctype = getattr(chain, "chain_type", "H")
            if ctype != "H":
                # Not an error: the manifests are known to carry a few rows whose
                # vh_seq column holds a light chain. Flag it, do not drop it.
                note = f"chain_type={ctype} (not a heavy chain)"
            rows.append({
                "vh_key": k, "vh_len": len(seq), "scheme": scheme,
                "cdr1_start": sp["CDR1"][0], "cdr1_end": sp["CDR1"][1],
                "cdr2_start": sp["CDR2"][0], "cdr2_end": sp["CDR2"][1],
                "cdr3_start": sp["CDR3"][0], "cdr3_end": sp["CDR3"][1],
                "cdr_total_len": total, "ok": 1, "note": note,
            })
        print(f"  [{scheme}] {min(i + BATCH, len(keys))}/{len(keys)}", flush=True)
    return rows


def short_reason(note):
    """Group failure notes into a handful of buckets for the summary."""
    n = note.lower()
    if "antibody domains" in n:
        # Not a numbering failure: the vh_seq column holds a multi-domain
        # construct (scFv / scFv-Fc), so "the VH" is not well defined for it.
        return "multi-domain construct (scFv etc.), single domain not resolvable"
    if n.startswith("anarci:"):
        return "ANARCI could not number the sequence"
    if "empty numbering" in n:
        return "scheme produced an empty numbering (ultralong CDR-H3 under Chothia)"
    if "not a substring" in n:
        return "numbered region not found in input string"
    if "more than once" in n:
        return "ambiguous offset (numbered region repeats)"
    if "self-check failed" in n:
        return "span self-check failed"
    if "not contiguous" in n:
        return "CDR positions not contiguous"
    if "no numbered positions" in n:
        return "a CDR had no numbered positions"
    if n.startswith("batch error"):
        return "ANARCI batch error"
    return note[:60]


def main():
    seqs, report = collect_sequences()

    print("Manifests read")
    for name, status, nrows, nuniq in report:
        print(f"  {name:28s} {status:20s} rows={nrows:<6d} unique vh_seq={nuniq}")
    print(f"\nDistinct vh_seq across all manifests: {len(seqs)}")

    all_rows = []
    for scheme in SCHEMES:
        print(f"\nNumbering under {scheme} ...")
        all_rows += run_scheme(seqs, scheme)

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df = df.sort_values(["scheme", "vh_key"]).reset_index(drop=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_SPANS, index=False)
    pd.DataFrame(
        [{"vh_key": k, "vh_seq": v} for k, v in seqs.items()]
    ).to_csv(OUT_SEQS, index=False)

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY  (rows attempted: {len(seqs)} sequences x {len(SCHEMES)} schemes = {len(df)})")
    print("=" * 70)
    for scheme in SCHEMES:
        sub = df[df.scheme == scheme]
        good = sub[sub.ok == 1]
        bad = sub[sub.ok == 0]
        print(f"\n[{scheme}]  numbered OK: {len(good)}/{len(sub)} "
              f"({100.0 * len(good) / max(len(sub), 1):.2f}%)   "
              f"failed: {len(bad)} ({100.0 * len(bad) / max(len(sub), 1):.2f}%)")
        if len(bad):
            for reason, n in Counter(bad.note.map(short_reason)).most_common():
                print(f"    {n:5d}  {reason}")
                ex = bad[bad.note.map(short_reason) == reason].iloc[0]
                print(f"           e.g. {ex.vh_key} (len {ex.vh_len}): {str(ex.note)[:110]}")
        nonheavy = good[good.note.str.contains("chain_type=", na=False)]
        if len(nonheavy):
            print(f"    note: {len(nonheavy)} numbered sequences are NOT heavy chains "
                  f"({', '.join(sorted(set(nonheavy.note)))})")
        if not len(good):
            continue
        tot = good.cdr_total_len.astype(int)
        vhl = good.vh_len.astype(int)
        print(f"    total CDR length  min={tot.min()}  median={tot.median():.1f}  max={tot.max()}")
        print(f"    VH length         min={vhl.min()}  median={vhl.median():.1f}  max={vhl.max()}")
        print(f"    median(CDR total / VH len) = {(tot / vhl).median():.4f}")
        for c, lo, hi in (("cdr1", "cdr1_start", "cdr1_end"),
                          ("cdr2", "cdr2_start", "cdr2_end"),
                          ("cdr3", "cdr3_start", "cdr3_end")):
            L = good[hi].astype(int) - good[lo].astype(int)
            print(f"    {c} length       min={L.min()}  median={L.median():.1f}  max={L.max()}")

    # IMGT vs Chothia, on sequences numbered OK under BOTH
    piv = df[df.ok == 1].pivot_table(index="vh_key", columns="scheme",
                                     values="cdr_total_len", aggfunc="first")
    if set(SCHEMES).issubset(piv.columns):
        piv = piv.dropna()
        d = (piv["imgt"] - piv["chothia"]).abs()
        print(f"\n[imgt vs chothia]  sequences numbered OK under both: {len(piv)}")
        print(f"    median total CDR length  imgt={piv['imgt'].median():.1f}  "
              f"chothia={piv['chothia'].median():.1f}")
        print(f"    median |imgt - chothia| = {d.median():.1f} residues")
        print(f"    differ by > 5 residues: {int((d > 5).sum())} "
              f"({100.0 * (d > 5).mean():.2f}%)")

    # Implausible CDR-H3
    print("\n[implausible CDR-H3]  (length 0, or > 40)")
    flagged = 0
    for scheme in SCHEMES:
        good = df[(df.scheme == scheme) & (df.ok == 1)].copy()
        if not len(good):
            continue
        good["h3"] = good.cdr3_end.astype(int) - good.cdr3_start.astype(int)
        odd = good[(good.h3 == 0) | (good.h3 > 40)]
        for _, r in odd.iterrows():
            print(f"    {scheme:8s} {r.vh_key}  CDR-H3 length {int(r.h3)}  (VH len {int(r.vh_len)})")
            flagged += 1
    if not flagged:
        print("    none")

    print(f"\nWrote {OUT_SPANS}  ({len(df)} rows)")
    print(f"Wrote {OUT_SEQS}  ({len(seqs)} rows)")


if __name__ == "__main__":
    main()
