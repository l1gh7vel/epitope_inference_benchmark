#!/usr/bin/env python3
"""
Task I manifest: the 5 antigen-identity folds, audited and stratified.

What this does
  * vendors the upstream fold CSVs into data/source/task1_folds (with sha256) so
    the pipeline runs from this repo alone and the upstream tree stays read-only;
  * concatenates them into one manifest with a `fold` column and a ragged label
    array per case (antigen lengths differ, so a rectangular npz is impossible);
  * trims the heavy chain to the variable domain by motif rather than by length. A
    fifth of these rows carry the whole IgG heavy chain including CH1, and
    length-based trimming would decapitate genuine ultralong-CDRH3 antibodies;
  * audits the split rather than trusting it. Two properties are checked and
    written into the manifest rather than assumed:
      - antigen clusters must not span groups (zero overlap is found);
      - antibodies should not span folds either, if the intent is
        "non-overlapping antibodies", and they do. `vh_in_other_fold` marks every
        such row.

Every exclusion is a flag rather than a deletion. Nothing is dropped from the
manifest; `usable` and `af3_eligible` say what each row is eligible for, so a
later decision to include or exclude is one filter away and is visible in the
file.

    PYTHONPATH=scripts:config python scripts/t1_00_build_manifest.py [--vendor]
"""
import argparse
import ast
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
import task1 as cfg                                        # noqa: E402
import common                                              # noqa: E402

# Seed for the random choice of which fold keeps each cross-fold heavy chain.
DEDUP_SEED = 0


def vendor():
    """Copy the upstream fold CSVs in, recording sha256. Upstream is read-only."""
    cfg.FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted(cfg.FOLDS_UPSTREAM.glob("test_fold_*.csv")):
        dst = cfg.FOLDS_DIR / f.name
        shutil.copy2(f, dst)
        h = hashlib.sha256(dst.read_bytes()).hexdigest()
        lines.append(f"{f.name}  {h}  <- {f}")
        print(f"  vendored {f.name} ({dst.stat().st_size / 1e6:.1f} MB)")
    (cfg.FOLDS_DIR / "PROVENANCE.txt").write_text(
        "Task I folds, copied unmodified from the read-only upstream tree.\n"
        "Built by the upstream benchmarks/make_cluster_folds.py: antigen sequence-identity\n"
        "clusters dealt into 5 folds.\n\n" + "\n".join(lines) + "\n")


def load_folds():
    src = cfg.FOLDS_DIR if any(cfg.FOLDS_DIR.glob("test_fold_*.csv")) else cfg.FOLDS_UPSTREAM
    fs = sorted(src.glob("test_fold_*.csv"))
    if not fs:
        sys.exit(f"no fold CSVs under {src} -- run with --vendor first")
    print(f"  reading {len(fs)} folds from {src}")
    return pd.concat([pd.read_csv(f).assign(fold=i + 1) for i, f in enumerate(fs)],
                     ignore_index=True)


def audit_split(d):
    """Check the two properties the split is supposed to have. Report, never fix."""
    cl = {f: set(g.Cluster) for f, g in d.groupby("fold")}
    shared_cl = sum(len(cl[a] & cl[b]) for a in cl for b in cl if a < b)
    # The sum of pairwise overlaps double-counts any VH spanning three or more
    # folds, so it is not the count to quote. The distinct count is reported below.
    vh = {f: set(g.vh_seq) for f, g in d.groupby("fold")}
    shared_vh = sum(len(vh[a] & vh[b]) for a in vh for b in vh if a < b)
    print(f"\n  SPLIT AUDIT")
    print(f"    antigen clusters shared between folds : {shared_cl}"
          f"   {'OK -- antigen-disjoint as intended' if shared_cl == 0 else 'PROBLEM'}")
    print(f"    VH overlaps summed over fold pairs    : {shared_vh}"
          f"   (double-counts VH in 3+ folds; use the line below)")
    # Per row: does this exact VH also appear in a different fold? Compared on the
    # trimmed variable domain rather than the deposited sequence. Two antibodies
    # that differ only in their constant region are the same antibody for this
    # benchmark, because only the variable domain is ever folded or scored. The raw
    # column undercounts: 215 shared sequences over 578 rows, against 225 over 658.
    counts = d.groupby("vh_seq").fold.nunique()
    d["vh_in_other_fold"] = d.vh_seq.map(counts).gt(1)
    # Which fold each shared VH is kept in, when the folds are made
    # antibody-disjoint by keeping one occurrence. The fold is chosen at random
    # among the folds that VH appears in, with a fixed seed. Keeping the
    # lowest-numbered one would retain every shared antibody that touches fold 1
    # and thin the others, and fold 1 is already the pathological fold (one antigen
    # cluster, 97% of it), so a lowest-fold rule would concentrate the retained
    # duplicates where they do the most damage. Random keep is unbiased across
    # folds.
    rng_d = np.random.default_rng(DEDUP_SEED)
    keep_fold = {vh: int(rng_d.choice(sorted(set(g))))
                 for vh, g in d.groupby("vh_seq").fold}
    d["vh_dedup_keep"] = (~d.vh_in_other_fold) | (d.fold == d.vh_seq.map(keep_fold))
    n_seq = int((d.groupby("vh_seq").fold.nunique() > 1).sum())
    print(f"    VH sequences appearing in >1 fold     : {n_seq}")
    print(f"    rows those account for                : {int(d.vh_in_other_fold.sum())}"
          f" of {len(d)} ({100 * d.vh_in_other_fold.mean():.1f}%)")
    print(f"    rows if deduped to one fold per VH    : {int(d.vh_dedup_keep.sum())} "
          f"kept, {int((~d.vh_dedup_keep).sum())} dropped")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", action="store_true",
                    help="copy the upstream fold CSVs into data/source first")
    a = ap.parse_args()
    if a.vendor:
        vendor()

    d = load_folds()
    print(f"  {len(d)} complexes, {d.antigen_sequence.nunique()} distinct antigens, "
          f"{d.vh_sequence.nunique()} distinct VH, {d.Cluster.nunique()} clusters")

    # ---- antibody chain: trim by motif, classify, flag rather than drop ----
    cleaned = [common.clean_seq(s) for s in d.vh_sequence]
    d["nonstandard_aa"] = [n for _s, n in cleaned]
    trimmed = [common.trim_heavy_to_variable(s) for s, _n in cleaned]
    d["vh_seq"] = [t[0] for t in trimmed]
    d["vh_trimmed"] = [t[1] for t in trimmed]
    d["vh_len"] = d.vh_seq.str.len()
    d["vh_kind"] = d.vh_seq.map(common.domain_kind)
    print(f"\n  heavy chains: {int(d.vh_trimmed.sum())} carried a constant region and "
          f"were trimmed at CH1")
    print("  VH framework-2 classification (recorded, not used as a filter -- "
          "hypermutated frameworks evade the motif):")
    for k, n in d.vh_kind.value_counts().items():
        print(f"    {k:8s} {n}")

    # ---- labels: ragged, one vector per case ----
    labels = [np.asarray(ast.literal_eval(s), dtype=np.int8) for s in d.epitope_labels]
    d["antigen_len"] = [len(x) for x in labels]
    d["n_epitope"] = [int(x.sum()) for x in labels]
    bad = d.antigen_len != d.antigen_sequence.str.len()
    if bad.any():
        print(f"  WARNING {int(bad.sum())} rows: label vector length != antigen length")

    # ---- eligibility ----
    # Stable per-antigen id. It lives here rather than in a downstream script
    # because several of them key on it, so a manifest rebuild carries it.
    d["antigen_id"] = ["ag_" + hashlib.sha1(s.encode()).hexdigest()[:12]
                       for s in d.antigen_sequence]
    d["vh_key"] = [common.chain_key("vh", s) for s in d.vh_seq]
    d["ag_key"] = [common.chain_key("ag", s) for s in d.antigen_sequence]
    d["case_id"] = [f"t1_{i:04d}_{p}_{c}" for i, (p, c)
                    in enumerate(zip(d.pdb_id, d.antigen_chain))]
    d["usable"] = (d.n_epitope >= cfg.MIN_EPITOPE_RESIDUES) & (d.vh_len > 60)
    d["af3_eligible"] = d.usable & (d.antigen_len <= cfg.MAX_ANTIGEN_LEN)
    print(f"\n  usable (>= {cfg.MIN_EPITOPE_RESIDUES} epitope residues, VH > 60 aa): "
          f"{int(d.usable.sum())} of {len(d)}")
    print(f"  of those, antigen <= {cfg.MAX_ANTIGEN_LEN} aa (AF3 token bucket): "
          f"{int(d.af3_eligible.sum())}")
    print(f"  antigen length: median {d.antigen_len.median():.0f}, "
          f"max {d.antigen_len.max()}, over {cfg.MAX_ANTIGEN_LEN}: "
          f"{int((d.antigen_len > cfg.MAX_ANTIGEN_LEN).sum())}")

    # ---- how degenerate is each fold? ----
    # The folds were balanced by complex count (917 each) rather than by antigen
    # diversity, and antigen-identity clustering swept nearly every SARS-CoV-2
    # spike complex into fold 1. So fold 1 is ~one antigen family asked 917 times
    # while the others span ~280. Any per-fold comparison has to be read with that
    # in mind: fold 1 being an outlier is a property of SAbDab rather than of the
    # method being scored.
    print("\n  FOLD COMPOSITION (folds are balanced by count, NOT by diversity)")
    for f, g in d.groupby("fold"):
        top = g.antigen_name.value_counts()
        print(f"    fold {f}: {len(g)} complexes, {g.Cluster.nunique():4d} antigen "
              f"clusters, median antigen {g.antigen_len.median():.0f} aa   "
              f"| commonest: {top.index[0][:34]} x{top.iloc[0]}")

    d = audit_split(d)

    cols = ["case_id", "fold", "pdb_id", "antigen_chain", "heavy_chain_id",
            "antigen_name", "antigen_type", "Cluster", "vh_seq", "vh_len",
            "vh_kind", "vh_trimmed", "antigen_sequence", "antigen_len",
            "antigen_id", "vh_key", "ag_key",
            "n_epitope", "n_contacts", "nonstandard_aa", "resolution", "method", "date",
            "uniprot_id", "source_pdb_ids", "n_source_structures",
            "usable", "af3_eligible", "vh_in_other_fold",
            "vh_dedup_keep"]
    out = d[[c for c in cols if c in d]].copy()
    cfg.MANIFESTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.MANIFEST, index=False)

    np.savez_compressed(cfg.LABELS_NPZ,
                        case_ids=np.array(d.case_id),
                        **{c: l for c, l in zip(d.case_id, labels)})
    print(f"\n  wrote {cfg.MANIFEST}  ({len(out)} rows)")
    print(f"  wrote {cfg.LABELS_NPZ}  (ragged, one vector per case)")


if __name__ == "__main__":
    main()
