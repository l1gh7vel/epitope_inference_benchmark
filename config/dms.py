"""
DMS functional-cost panel configuration.

NINE HIV-1 bnAbs with solved complexes AND a published deep mutational scan of
escape on the same antigen, from TWO Bloom-lab papers (both ingested by
`scripts/dms_01_escape.py`, which holds the live per-paper tables):

    Dingens et al. 2019 (Immunity), "antigenic atlas" -- the panel that DEFINES
        the cases: VRC01, 3BNC117, 10-1074, PGT121, PGT151, PG9, 10E8, PGT145,
        VRC34 on BG505.T332N Env. It is also the only source of a BINARY call of
        significant escape sites (gamma tail at FDR 0.01), which Figure 4's
        escape target set uses verbatim.
    Radford & Bloom 2025 (JVI) -- continuous escape for 3BNC117 and 10-1074 on
        two Envs (BF520 clade A, TRO.11 clade B). Publishes no significant-site
        calls, so it feeds the continuous arm only.

Nine antibodies x two chain arms = the 18 rows of `data/manifests/dms_manifest.csv`.

Everything is expressed in the HXB2 reference frame of `data/source/env.fa` --
the same 856-aa protomer Task III folds. Using one frame is
also what makes the different strains in play commensurable: the structures are
93TH057/BG505, the DMS is on BG505.T332N (Dingens 2019) and BF520/TRO.11 (Radford
2025), and we fold HXB2. Verified: the
authors' own site_numbering_map agrees with env.fa at 843/843 integer reference
positions, and both structures' depositor numbering is HXB2 (identity 0.75-0.79
against env.fa, the same range the Task III configuration uses for 5FYK/4TVP).

TWO ARMS, never mixed (`arm` in the manifest):
  * vhvl -- heavy + light, the Task II convention. Both chains are solved here.
  * vh   -- heavy only, the Task III convention, so this panel can be read
            against the existing HIV Env numbers.
Running both separates "the model cannot condition" from "we crippled its input",
which VH-only HIV Env results would otherwise confound.

Token budget: 856 (Env) + 132 (VH) + 105 (VL) = 1,093 -> the same 1,536 AF3
bucket as Tasks II/III.
"""

import os
from pathlib import Path

import env

# Project root. EIB_ROOT overrides it; the fallback is this file's own repo
# location. Resolve every path from here rather than from a literal, so one
# definition moves with the repository.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT / "data" / "source"
ENV_FASTA = DATA_DIR / "env.fa"
ANTIGEN_FASTA = ENV_FASTA
DMS_DIR = DATA_DIR / "dms_radford2025"

AF3_MODEL_DIR = env.AF3_MODEL_DIR
AF3_DB_DIR = env.AF3_DB_DIR

WORK = PROJECT / "work" / "dms"
CHAIN_JOBS = WORK / "chain_jobs"
MSA_RAW = WORK / "msa_raw"
# Own cache, but SEEDED from Task III: the Env chain (ag_d1cb3a73d5) is the
# identical sequence and its MSA is the single most expensive artefact in the
# pipeline, so it is copied in rather than re-searched. Keeping a separate
# directory stops this panel writing antibody MSAs into Task III's cache.
MSA_CACHE = WORK / "msa_cache"
AF3_OUT = WORK / "af3_out"
SCORES = WORK / "scores"
PDB_DIR = WORK / "pdb"

MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "dms_manifest.csv"
LABELS_NPZ = MANIFESTS / "dms_labels.npz"
ESCAPE_NPZ = MANIFESTS / "dms_escape.npz"

RESULTS = PROJECT / "results" / "dms_case"
FIGURES = RESULTS / "figures"

ARM_TAG = "dms"
ANTIGEN_NAME = "HIV-1 Env"
ANTIGEN_LEN = 856

CHAIN_ANTIGEN, CHAIN_HEAVY, CHAIN_LIGHT = "A", "H", "L"
MODEL_SEEDS = [1, 2, 3, 4, 5]
DIFFUSION_SAMPLES = 5
MAX_TEMPLATE_DATE = "2021-09-30"
TRIM_LIGHT_TO_VARIABLE = True
MIN_LIGHT_LEN = 80
CONTACT_CUTOFF = 4.5           # project-wide, every task

PYTHON = env.PYTHON

# Unlike Tasks II/III these cases DO have solved complexes -- that is the point.
HAS_EXPERIMENTAL = True

# ------------------------------------------------------------- DMS data ----
# No script reads these tables; `scripts/dms_01_escape.py` carries its own
# copies (DINGENS, RADFORD, STATS) with the paths it needs to ingest both
# papers in one pass. The NEGATIVE_CONTROL note at the bottom of this block is
# the one definition kept only here.
# (antibody, strain) -> published per-mutation escape csv.
DMS_FILES = {
    ("3BNC117", "BF520"): "BF520_Env_3BNC117_mut_effects_filtered.csv",
    ("3BNC117", "TRO11"): "TRO11_Env_3BNC117_mut_effects_filtered.csv",
    ("10-1074", "BF520"): "BF520_Env_10-1074_mut_effects_filtered.csv",
    ("10-1074", "TRO11"): "TRO11_Env_10-1074_mut_effects_filtered.csv",
}
# Site-level escape = this collapse over the mutations measured at a site.
ESCAPE_SITE_STAT = "mean"
# 3BNC117 x TRO.11 carries no site-level escape signal at its own structural
# epitope (MWU p=0.54): TRO.11 cannot tolerate the loop D mutations at all, so
# their escape is unmeasurable -- stated in the paper, not an artefact of ours.
# Kept and reported as a NEGATIVE CONTROL on the assay, never quietly dropped.
NEGATIVE_CONTROL = {("3BNC117", "TRO11")}
