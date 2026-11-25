"""
Task III-CoV (SARS-CoV-2 Spike epitope-domain localisation) configuration.

The second Task III antigen. Same question as the HIV Env arm -- given an antibody
with no solved complex, can a predictor localise its epitope to the right *region*?
-- but on Spike, with CoV-AbDab's three coarse bins.

Reuses the shared pipeline unchanged via `EIB_TASK=task3cov`.

Differences from the Env arm worth knowing:
  * 300 antibodies, PERFECTLY BALANCED across 3 bins (NTD/RBD/S2, 100 each). That
    makes the antibody-blind floor 1/3 = 0.333 instead of Env's 0.383, and it means
    macro-recall and plain accuracy nearly coincide.
  * Only 3 bins, but they are BIG (NTD 290 residues, RBD 223, S2 588 = 87% of the
    protein between them), so this is a coarser question than Env's.
  * Some entries are NANOBODIES (`is_vhh`) -- single-domain, genuinely no light
    chain, which fits the VH-only design exactly rather than being a compromise.
  * Same antigen as Task II, so the Spike MSA is already cached and is reused.
"""

import os
from pathlib import Path

import env

# ---------------------------------------------------------------- paths ----
# Project root. EIB_ROOT overrides it; the fallback is this file's own repo
# location. Resolve every path from here rather than from a literal, so one
# definition moves with the repository.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))

# Vendored copies of the upstream read-only inputs (data/source/PROVENANCE.md
# records origin + sha256). Local so the pipeline is reproducible from this
# repo alone; the upstream tree stays read-only and is never written to.
DATA_DIR = PROJECT / "data" / "source"
CASES_CSV = DATA_DIR / "covabdab_bins.csv"        # 300 rows: Name, bin, vh_sequence, is_vhh
DOMAINS_JSON = DATA_DIR / "spike.domains.json"    # UniProt P0DTC2 regions (provenance)
SPIKE_FASTA = DATA_DIR / "spike.fa"
ANTIGEN_FASTA = SPIKE_FASTA

AF3_MODEL_DIR = env.AF3_MODEL_DIR
AF3_DB_DIR = env.AF3_DB_DIR

WORK = PROJECT / "work" / "task3cov"
CHAIN_JOBS = WORK / "chain_jobs"
MSA_RAW = WORK / "msa_raw"
MSA_CACHE = WORK / "msa_cache"
AF3_OUT = WORK / "af3_out"
SCORES = WORK / "scores"

MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "task3cov_manifest.csv"
LABELS_NPZ = MANIFESTS / "task3cov_labels.npz"

RESULTS = PROJECT / "results" / "task3cov"

# ------------------------------------------------------------- antigen ----
# Short tag + display name for this Task III arm. Figure filenames embed the
# tag so the two antigens have distinct output names.
ARM_TAG = "cov"
ANTIGEN_NAME = "SARS-CoV-2 Spike"

ANTIGEN_LEN = 1273                # full-length Spike, same frame as Task II
ENV_COPIES = 1                    # one protomer (Spike is also a trimer in reality)

# ---- epitope bins -> Spike residue ranges (1-based, inclusive) ----
# SOURCE: the vendored `spike.domains.json` = UniProt/Swiss-Prot P0DTC2 features.
#   NTD <- Domain "BetaCoV S1-NTD"                 14-303
#   RBD <- Region "Receptor-binding domain (RBD)"  319-541
#   S2  <- not a single UniProt feature; it is everything C-terminal of the S1/S2
#          furin cleavage site at 685/686, which is the standard definition and the
#          one already used in this project's Task II docs. The UniProt S2 landmarks
#          (fusion peptides 816-855, HR1 920-970, HR2 1163-1202) all fall inside it.
BIN_RANGES = {
    "NTD": [(14, 303)],
    "RBD": [(319, 541)],
    "S2":  [(686, 1273)],
}
BINS = list(BIN_RANGES)
BIN_ALIASES = {}
DROP_BINS = set()

# Spike is a trimer too, but all three of these bins are largely within one
# protomer, so single-copy folding is less of a compromise here than for Env's
# V2 apex. Recorded for symmetry.
TRIMER_DEPENDENT = set()

# -------------------------------------------------------- AF3 settings ----
CHAIN_ANTIGEN, CHAIN_HEAVY, CHAIN_LIGHT = "A", "H", "L"
MODEL_SEEDS = [1, 2, 3, 4, 5]
DIFFUSION_SAMPLES = 5
MAX_TEMPLATE_DATE = "2021-09-30"
TRIM_LIGHT_TO_VARIABLE = True
MIN_LIGHT_LEN = 80
CONTACT_CUTOFF = 4.5

PYTHON = env.PYTHON

# Is there an experimental complex per case? Gates scripts/08 --source experimental.
# CoV-AbDab gives an epitope bin, not a structure.
HAS_EXPERIMENTAL = False
