#!/usr/bin/env python3
"""
Task III stage 00 -- build the case manifest and domain gold labels from CATNAP.

Serves BOTH Task III arms from one code path, selected by EIB_TASK:
  task3     HIV Env    -- vendored `catnap_bins_seq.csv` (314 rows, 5 bins after
                          folding GLYCAN_V3 into V3)
  task3cov  SARS-CoV-2 -- vendored `covabdab_bins.csv`  (300 rows, 3 bins,
                          100 each; some entries are nanobodies)
Both give: antibody name, epitope bin, VH sequence. Unlike Task II there is no
solved complex, so
the gold is not a residue-level contact set but the antibody's **epitope bin** --
a region of Env. We expand each bin to its Env residue set via cfg.BIN_RANGES and
store that as a 856-long 0/1 vector, so the downstream scorer can reuse the same
"score vector vs label vector" machinery as Task II.

Nothing downstream re-reads the source CSV: the manifest is the contract.

    EIB_TASK=task3    python scripts/t3_00_build_manifest.py   # HIV Env
    EIB_TASK=task3cov python scripts/t3_00_build_manifest.py   # SARS-CoV-2 Spike
"""
import numpy as np
import pandas as pd

from common import (cfg, chain_key, clean_seq, domain_kind, read_fasta_single,
                    trim_heavy_to_variable)


def bin_mask(bin_name):
    """1-based inclusive ranges -> 0/1 vector over the Env protomer."""
    m = np.zeros(cfg.ANTIGEN_LEN, np.uint8)
    for lo, hi in cfg.BIN_RANGES[bin_name]:
        m[lo - 1:hi] = 1
    return m


def main():
    env = read_fasta_single(cfg.ANTIGEN_FASTA)
    assert len(env) == cfg.ANTIGEN_LEN, f"env.fa is {len(env)} aa, expected {cfg.ANTIGEN_LEN}"

    df = pd.read_csv(cfg.CASES_CSV)
    print(f"source: {cfg.CASES_CSV.name}  {len(df)} rows")

    masks = {b: bin_mask(b) for b in cfg.BINS}
    for b, m in masks.items():
        spans = ", ".join(f"{lo}-{hi}" for lo, hi in cfg.BIN_RANGES[b])
        print(f"  bin {b:10s} {int(m.sum()):4d} residues  [{spans}]")

    rows, labels, problems = [], [], []
    seen = set()          # (name, bin) -> drop exact repeats
    for i, r in df.iterrows():
        name, bn = str(r["Name"]), str(r["bin"]).strip()
        bn = cfg.BIN_ALIASES.get(bn, bn)          # e.g. GLYCAN_V3 -> V3
        if bn in cfg.DROP_BINS:
            problems.append((name, f"dropped: junk bin '{bn}'"))
            continue
        if bn not in cfg.BIN_RANGES:
            problems.append((name, f"dropped: unknown bin '{bn}'"))
            continue
        if any(name.startswith(prefix)
               for prefix in getattr(cfg, "DROP_ANTIBODY_PREFIXES", ())):
            problems.append((name, "dropped: outside evaluation cohort"))
            continue
        vh, vh_bad = clean_seq(r["vh_sequence"])
        # Some rows carry the whole IgG heavy chain; cut the constant region so
        # AF3 folds an antibody variable domain, as in Task II. Motif-based, so
        # bovine ultralong-CDRH3 domains are left intact.
        vh, vh_trimmed = trim_heavy_to_variable(vh)
        if len(vh) < 80:
            problems.append((name, f"dropped: VH only {len(vh)} aa"))
            continue

        # CATNAP repeats 9 antibodies verbatim (same name, same bin, same VH).
        # The upstream scorer drops these too. Keeping them would fold the same complex
        # twice and double-weight those antibodies in every metric.
        if (name, bn) in seen:
            problems.append((name, f"dropped: duplicate (name, bin={bn})"))
            continue
        seen.add((name, bn))

        case_id = f"t3_{i:03d}_{''.join(ch for ch in name if ch.isalnum())[:16].lower()}"
        rows.append({
            "case_id": case_id,
            "row_index": i,
            "ab_name": name,
            "bin": bn,
            "trimer_dependent": bn in cfg.TRIMER_DEPENDENT,
            "vh_seq": vh,
            "vh_len": len(vh),
            "vh_trimmed": vh_trimmed,
            "vh_kind": domain_kind(vh),               # expected 'heavy'
            "vh_key": chain_key("vh", vh),
            # VH only by design -- keep the columns so stages 01/03 are unchanged.
            "light_seq": "", "light_len": 0, "light_key": "", "has_light": False,
            # NB: unlike Task II we do NOT filter on the FR2 motif. HIV bnAbs are
            # extraordinarily somatically hypermutated, including in the framework,
            # so the germline-based motif misses ~22% of genuine heavy chains here
            # (e.g. WIRL/WIRR/WVQQ instead of WVRQ). CATNAP's vh_sequence column is
            # curated, so we trust the source and keep every row, recording
            # `vh_kind` for audit rather than using it to drop data.
            "chain_role_ok": True,
            "fr2_vh_kind": domain_kind(vh),
            "n_domain_residues": int(masks[bn].sum()),
            "n_tokens": cfg.ANTIGEN_LEN * cfg.ENV_COPIES + len(vh),
            "light_chain_type": r.get("Light chain type", ""),
            # CoV-AbDab marks single-domain antibodies. A nanobody genuinely has no
            # light chain, so for those rows VH-only is the complete molecule rather
            # than a deliberate simplification.
            "is_vhh": bool(r.get("is_vhh", False)),
            "nonstandard_aa": vh_bad,
        })
        labels.append(masks[bn])

    man = pd.DataFrame(rows)
    cfg.MANIFESTS.mkdir(parents=True, exist_ok=True)
    man.to_csv(cfg.MANIFEST, index=False)
    np.savez_compressed(cfg.LABELS_NPZ,
                        case_id=np.array(man.case_id),
                        labels=np.stack(labels),
                        bin=np.array(man.bin))

    print(f"\nkept {len(man)} / {len(df)} rows ({len(problems)} dropped)")
    print(f"  bins            : {man['bin'].value_counts().to_dict()}")
    print(f"  FR2 motif says  : {man.fr2_vh_kind.value_counts().to_dict()}  "
          f"(informational only -- not used to drop rows; bnAbs are hypermutated)")
    print(f"  heavy chains trimmed to VH: {int(man.vh_trimmed.sum())}")
    print(f"  unique VH chains: {man.vh_key.nunique()} (+1 antigen)")
    print(f"  tokens          : {man.n_tokens.min()}-{man.n_tokens.max()} "
          f"(Task II was ~1506; smaller bucket here)")
    if man.is_vhh.any():
        print(f"  nanobodies (VHH): {int(man.is_vhh.sum())} -- VH-only is complete for these")
    print(f"  trimer-dependent: {int(man.trimer_dependent.sum())} cases "
          f"({sorted(cfg.TRIMER_DEPENDENT)}) -- expected to underperform on one protomer")
    for n, why in problems[:10]:
        print(f"  DROP {n}: {why}")
    print(f"\nwrote {cfg.MANIFEST}\nwrote {cfg.LABELS_NPZ}")


if __name__ == "__main__":
    main()
