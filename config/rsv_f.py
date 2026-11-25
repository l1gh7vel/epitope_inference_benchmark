"""RSV F escape-arm configuration.

The arm contains Nirsevimab, Clesrovimab, Palivizumab, Suptavumab, 1A2 and 1B6.
RSM01 is excluded because no public antibody sequence is available. Each antibody
has ten declared escape sites. Four antibodies have a solved F complex; structural
metrics use those four. Template exposure and the 1A2 measured-null flag are stored
per case.

The ``vhvl`` and ``vh`` manifest arms use heavy-plus-light and heavy-only inputs.
The antigen is one copy of wild-type, non-stabilized RSV F residues 26--529.
The native trimer has cross-protomer contacts that a monomeric input cannot form.
"""

import os
from pathlib import Path

import env

# Project root. EIB_ROOT overrides the repository location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT / "data" / "source"
DMS_DIR = DATA_DIR / "dms_simonich2026"
ANTIGEN_FASTA = DMS_DIR / "F_protein.fasta"
ESCAPE_CSV = DMS_DIR / "all_antibodies.csv"
MAB_CSV = DMS_DIR / "RSV_mAb_AAseq.csv"

AF3_MODEL_DIR = env.AF3_MODEL_DIR
AF3_DB_DIR = env.AF3_DB_DIR

WORK = PROJECT / "work" / "rsv_f"
CHAIN_JOBS = WORK / "chain_jobs"
MSA_RAW = WORK / "msa_raw"
MSA_CACHE = WORK / "msa_cache"
AF3_OUT = WORK / "af3_out"
SCORES = WORK / "scores"
REF_DIR = WORK / "ref"            # deposited F structures, gitignored (see its README.md)

MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "rsv_manifest.csv"
LABELS_NPZ = MANIFESTS / "rsv_labels.npz"          # gold structural epitopes, 4 of 6
ESCAPE_NPZ = MANIFESTS / "rsv_escape_sites.npz"    # declared top-10 escape sets, 6 of 6
GOLD_REPORT = MANIFESTS / "rsv_gold_report.json"

RESULTS = PROJECT / "results" / "rsv_case"
FIGURES = RESULTS / "figures"

ARM_TAG = "rsv"
ANTIGEN_NAME = "RSV F"
ANTIGEN_LEN = 504                 # F_protein.fasta; == sites 26-529
SITE_OFFSET = 25                  # 1-based fasta index i == F site i + 25
SITE_LO, SITE_HI = 26, 529

# The hand-named antigen chain key. It differs from chain_key("ag", seq): the key was
# assigned by `rsv_02_assemble_fold.py` before this config existed and its MSA is
# already searched and sitting in work/rsv_f/msa_raw/ag_rsvF_long/. Stages that
# would otherwise recompute the key read this instead, so the finished search is
# not thrown away. Note the mixed case -- AF3 3.0.2 keeps the job name's case in
# its output directory name, and stage 02 lowercases when it looks for it.
ANTIGEN_CHAIN_KEY = "ag_rsvF_long"

CHAIN_ANTIGEN, CHAIN_HEAVY, CHAIN_LIGHT = "A", "H", "L"
MODEL_SEEDS = [1, 2, 3, 4, 5]
DIFFUSION_SAMPLES = 5
MAX_TEMPLATE_DATE = "2021-09-30"
TRIM_LIGHT_TO_VARIABLE = True
MIN_LIGHT_LEN = 80
CONTACT_CUTOFF = 4.5           # project-wide, every task

PYTHON = env.PYTHON

# Four of the six have a solved complex, so this is True "in part" -- always ask
# the manifest's `has_gold` for a specific antibody rather than this flag.
HAS_EXPERIMENTAL = True

# The model input contains one F protomer. Three solved epitopes have a
# 19--26% cross-protomer contact fraction; `cross_protomer_frac` in the manifest
# records this ceiling for interpretation.
ANTIGEN_ASSEMBLY = "monomer"
ANTIGEN_COPIES = 1

# ------------------------------------------------------------ escape rule ----
# DECLARED, not fitted, and not the authors'. See rsv_01_escape_sites.py for the
# full argument; the short version is that every value-keyed rule (gamma-FDR,
# absolute cut, robust-z) drops 1A2 to 0-1 sites and 1A2 is in the panel by
# Fixed-rank selection retains ten sites for every antibody.
ESCAPE_TOPK = 10                  # top 10 of 504 sites = top 1.98%, per antibody
ESCAPE_STATISTIC = "sum over mutations of max(Fab escape, 0)"   # the authors' own
MEASURED_NULL = {"1A2"}
