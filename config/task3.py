"""Task III HIV-1 Env domain-localisation configuration.

The 300-antibody benchmark uses one Env protomer and the antibody heavy chain.
Predictions are evaluated by argmax-domain accuracy, Cohen's kappa, and
per-antibody domain rank-biserial correlation. A single protomer cannot form the
quaternary V2 apex epitope.
"""

import os
from pathlib import Path

import env

# ---------------------------------------------------------------- paths ----
# Project root. EIB_ROOT overrides it; the fallback is this file's own repo
# location. Resolve every path from here rather than from a literal, so one
# definition moves with the repository.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))

# Read-only inputs used to build the frozen public manifest. These source files
# are not distributed with the public repository; data/source/PROVENANCE.md
# records how to obtain and verify them.
DATA_DIR = PROJECT / "data" / "source"
CASES_CSV = DATA_DIR / "catnap_bins_seq.csv"      # 314 rows: Name, bin, vh_sequence, ...
DOMAINS_JSON = DATA_DIR / "env.domains.json"      # UniProt P04578 regions (provenance)
ENV_FASTA = DATA_DIR / "env.fa"                   # HXB2 gp160 protomer, 856 aa
ANTIGEN_FASTA = ENV_FASTA                         # generic alias used by stages 01/03

# AlphaFold 3 resources (identical to Task II).
AF3_MODEL_DIR = env.AF3_MODEL_DIR
AF3_DB_DIR = env.AF3_DB_DIR

# Generated state (gitignored). Separate tree from Task II so nothing collides.
WORK = PROJECT / "work" / "task3"
CHAIN_JOBS = WORK / "chain_jobs"
MSA_RAW = WORK / "msa_raw"
MSA_CACHE = WORK / "msa_cache"
AF3_OUT = WORK / "af3_out"
SCORES = WORK / "scores"

MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "task3_manifest.csv"
LABELS_NPZ = MANIFESTS / "task3_labels.npz"       # per-case domain label vectors

RESULTS = PROJECT / "results" / "task3"

# ------------------------------------------------------------- antigen ----
# Short tag + display name for this Task III arm. Figure filenames embed the
# tag so the two antigens have distinct output names.
ARM_TAG = "hiv"
ANTIGEN_NAME = "HIV-1 Env"
ANTIBODY_INPUT = "VH"

ANTIGEN_LEN = 856                 # HXB2 gp160 protomer; env.fa is already in this frame
ENV_COPIES = 1                    # one protomer; a trimer variant is not run

# gp120 / gp41 split, for reporting only.
GP120 = (1, 511)
GP41 = (512, 856)

# ---- epitope bins -> Env residue ranges (1-based, inclusive, HXB2 numbering) ----
# SOURCE: the vendored `env.domains.json` = UniProt/Swiss-Prot P04578 (HIV-1 HXB2 Env)
# curated Region features. env.fa is already in this frame, so these map 1:1.
#
# GLYCAN_V3 annotations are merged into V3. CD4bs is represented by the curated
# UniProt CD4-binding loop at residues 364-374.
BIN_RANGES = {
    "V3":    [(296, 330)],      # UniProt "V3" (+ GLYCAN_V3 merged in)
    "V2":    [(157, 196)],      # UniProt "V2" -- V1V2 apex, trimer-dependent
    "CD4bs": [(364, 374)],      # UniProt "CD4-binding loop" (see caveat above)
    "FP":    [(512, 532)],      # UniProt "Fusion peptide", gp41 N-terminus
    "MPER":  [(662, 683)],      # UniProt "MPER; binding to GalCer"
}
BINS = list(BIN_RANGES)
# CATNAP bin label -> our bin. GLYCAN_V3 folds into V3.
BIN_ALIASES = {"GLYCAN_V3": "V3"}
DROP_BINS = {"immid"}             # one junk row in the CSV
DROP_ANTIBODY_PREFIXES = ("VRC07-523",)

# Bins whose epitope only fully exists on the assembled trimer -- expect these to
# underperform under ENV_COPIES = 1 and say so when reporting. (Carrying this
# caveat is deliberate; a 3-copy variant is not run.)
TRIMER_DEPENDENT = {"V2"}

# -------------------------------------------------------- AF3 settings ----
CHAIN_ANTIGEN, CHAIN_HEAVY, CHAIN_LIGHT = "A", "H", "L"
MODEL_SEEDS = [1, 2, 3, 4, 5]
DIFFUSION_SAMPLES = 5
MAX_TEMPLATE_DATE = "2021-09-30"
TRIM_LIGHT_TO_VARIABLE = True      # unused here (VH only) but kept for stage parity
MIN_LIGHT_LEN = 80
CONTACT_CUTOFF = 4.5

PYTHON = env.PYTHON

# Is there an experimental complex per case? Gates scripts/08 --source experimental.
# CATNAP gives an epitope bin, not a structure -- no solved complex.
HAS_EXPERIMENTAL = False
