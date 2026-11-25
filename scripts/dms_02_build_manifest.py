"""Fold manifest for the DMS panel: one row per (antibody, chain arm).

Two arms per antibody, never mixed:
  vhvl -- heavy + light, the Task II convention. Both chains are solved for every
          one of these antibodies, so this is the models' best shot.
  vh   -- heavy only, the Task III convention, so the panel can be read against
          the existing HIV Env numbers.

Running both separates "the model cannot condition on the antibody" from "we
crippled its input", which VH-only HIV Env results would otherwise
confound.

Antibody chains come straight out of the deposited Fab and are trimmed to the
variable domain with the same helpers Tasks II/III use, so AF3 sees an Fv and not
a Fab, the project convention. The antigen is always env.fa, whose MSA is already
cached from Task III.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (cfg, chain_key, clean_seq, read_fasta_single,  # noqa: E402
                    trim_heavy_to_variable, trim_light_to_variable)

ARMS = ("vhvl", "vh")


def main():
    gold = pd.read_csv(cfg.MANIFEST if cfg.MANIFEST.exists() else
                       cfg.MANIFESTS / "dms_manifest.csv")
    env = read_fasta_single(cfg.ANTIGEN_FASTA)
    rows = []
    for r in gold.itertuples():
        h_raw, _ = clean_seq(r.vh_sequence)
        l_raw, _ = clean_seq(r.vl_sequence)
        vh, vh_trimmed = trim_heavy_to_variable(h_raw)
        vl, vl_trimmed = trim_light_to_variable(l_raw)
        for arm in ARMS:
            has_light = arm == "vhvl"
            rows.append(dict(
                case_id=f"{r.case_id}__{arm}", antibody=r.case_id, arm=arm,
                pdb=r.pdb, resolution=r.resolution, epitope_class=r.epitope_class,
                vh_seq=vh, vh_len=len(vh), vh_trimmed=vh_trimmed,
                vh_key=chain_key("vh", vh),
                light_seq=vl if has_light else "", light_len=len(vl) if has_light else 0,
                light_key=chain_key("vl", vl) if has_light else "",
                has_light=has_light,
                n_gold=r.n_gold,
                n_tokens=len(env) + len(vh) + (len(vl) if has_light else 0)))
    man = pd.DataFrame(rows)
    man.to_csv(cfg.MANIFEST, index=False)
    print(man[["case_id", "arm", "vh_len", "light_len", "n_tokens", "n_gold"]].to_string(index=False))
    chains = set(man.vh_key) | {k for k in man.light_key if k}
    print(f"\n{len(man)} fold cases, {len(chains)} distinct antibody chains needing an MSA")
    print(f"antigen chain {chain_key('ag', env)} is already cached from Task III")
    print(f"max tokens {man.n_tokens.max()} (AF3 bucket 1536)")


if __name__ == "__main__":
    main()
