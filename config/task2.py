"""
Task II (Spike RBM vs non-RBM) configuration.

Every path and tunable the AF3 pipeline uses lives here, so no script hard-codes
a location. Plain Python instead of YAML: no parser, no extra dependency, and it
can be imported directly by the scripts.

Import from anywhere with:
    import sys; sys.path.insert(0, "<project>/scripts")
    from common import cfg       # every pipeline stage imports it this way
"""

import os
from pathlib import Path

import env

# ---------------------------------------------------------------- paths ----
# Project root. EIB_ROOT overrides it; the fallback is this file's own repo
# location. Resolve every path from here rather than from a literal, so one
# definition moves with the repository.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))

# External inputs used only to rebuild the frozen public manifest.
# data/source/PROVENANCE.md records their origin and checksum.
DATA_DIR = PROJECT / "data" / "source"
SPIKE_CSV = DATA_DIR / "rbm_nonrbm_fold_input.csv"
SPIKE_FASTA = DATA_DIR / "spike.fa"
ANTIGEN_FASTA = SPIKE_FASTA   # generic alias: stages 01/03 are task-agnostic

# AlphaFold 3 resources. The module (alphafold/3.0.2) is loaded in the sbatch
# files, not here, since only Slurm needs it.
#   AF3_MODEL_DIR must be a directory *containing* af3.bin (not the file itself).
#   AF3_DB_DIR is the CRCD-hosted genetic database bundle.
AF3_MODEL_DIR = env.AF3_MODEL_DIR
AF3_DB_DIR = env.AF3_DB_DIR

# Generated state (gitignored).
WORK = PROJECT / "work"
CHAIN_JOBS = WORK / "chain_jobs"    # single-chain AF3 JSONs for the MSA stage
MSA_RAW = WORK / "msa_raw"          # raw AF3 data-pipeline output
MSA_CACHE = WORK / "msa_cache"      # one harvested chain block per unique chain
AF3_OUT = WORK / "af3_out"          # one directory per predicted complex
SCORES = WORK / "scores"            # one .npz of per-residue scores per case

MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "task2_manifest.csv"
LABELS_NPZ = MANIFESTS / "task2_labels.npz"    # gold labels + structure masks

RESULTS = PROJECT / "results" / "task2"

PYTHON = env.PYTHON

# ------------------------------------------------------------- antigen ----
ANTIGEN_ID = "P0DTC2"          # SARS-CoV-2 Spike, UniProt
ANTIGEN_LEN = 1273             # every epitope_labels vector must be this long

# Windows in 1-based P0DTC2 numbering, inclusive.
RBM = (436, 507)
RBD = (319, 541)

# The epitope classes reported as groups. `epitope_class` takes four values
# (scripts/00_build_manifest.py defines them); the other two, nonRBM_outside_RBD
# and ambiguous, are carried in the all-antibody row and in no group.
REPORTED_CLASSES = ("RBM", "nonRBM")

# ------------------------------------------------------- AF3 parameters ----
CHAIN_ANTIGEN = "A"
CHAIN_HEAVY = "H"
CHAIN_LIGHT = "L"

# A "seed" reruns the whole network (the expensive trunk pass + sampling); a
# "diffusion sample" reruns only the cheap coordinate head from fresh noise,
# reusing one trunk pass. So samples are near-free variety within one reading of
# the complex, while seeds give independent readings (different MSA subsampling
# in the trunk) at ~100 s each. For an ensemble epitope score we want both kinds
# of variation, and the cost is modest: 5 seeds x 5 samples = 25 structures per
# complex, about 500 s GPU per complex.
MODEL_SEEDS = [1, 2, 3, 4, 5]
DIFFUSION_SAMPLES = 5

# AlphaFold3's published default and training cutoff.
MAX_TEMPLATE_DATE = "2021-09-30"

# Light chains in the source CSV are inconsistent: 183 are variable domains
# (~110 aa) but 114 include the constant domain (~210 aa), which pushes those
# complexes past AF3's 1536-token compilation bucket. Trimming to the variable
# domain makes the light chain consistent with the VH column (already trimmed)
# and keeps every complex in one bucket.
TRIM_LIGHT_TO_VARIABLE = True
MIN_LIGHT_LEN = 60             # below this the "light chain" is a fragment; drop it

# --------------------------------------------------------------- scoring ----
CONTACT_CUTOFF = 4.5           # A, heavy-atom; WALLE/AsEP convention
# (BSA is ranked as a continuous per-residue value, no threshold: AUPRC depends
#  on the top of the ranking, where burial is large, so a noise floor is moot.)

# Is there an experimental complex per case? Gates scripts/08 --source experimental.
# Task II cases come from solved antibody-Spike complexes.
HAS_EXPERIMENTAL = True
