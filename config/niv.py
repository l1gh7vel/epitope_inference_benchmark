"""Nipah virus receptor-binding protein escape-arm configuration.

The arm contains four antibodies with recoverable VH and VL sequences: nAH1.3,
HENV-26, HENV-32 and m102.4. Results are shown per antibody without a significance
test because n = 4. Three deposited complexes predate the AlphaFold3 template
cutoff and one postdates it; leakage is reported per case.

The ``vhvl`` and ``vh`` manifest arms use heavy-plus-light and heavy-only inputs,
respectively. The antigen is one copy of the 532-residue ectodomain, corresponding
to reference residues 71--602. The native receptor-binding protein is tetrameric,
so this input cannot form a quaternary epitope.
"""

import os
from pathlib import Path

import env

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT / "data" / "source"
DMS_DIR = DATA_DIR / "dms_larsen2025"
# The VENDORED fasta is 533 CHARACTERS but 532 RESIDUES: its last character is a
# `*` stop codon, which AF3 rejects outright ("Protein must contain only
# letters") and which would make any mask built on len(fasta) one position too
# long. The shared stages (01 chain jobs, 03 AF3 assembly) read ANTIGEN_FASTA
# with `common.read_fasta_single`, which does no such stripping and must not be
# taught to -- it is load-bearing for four other tasks. So ANTIGEN_FASTA points
# at a CLEANED copy that `scripts/niv_02_build_manifest.py` writes into work/,
# and the vendored original stays reachable as ANTIGEN_FASTA_RAW.
ANTIGEN_FASTA_RAW = DMS_DIR / "protein.fasta"
ESCAPE_CSV = DMS_DIR / "RBP_mutation_effects_antibody_escape.csv"
NUMBERING_CSV = DMS_DIR / "site_numbering_map.csv"
# The MATCHED tolerance axis: every antibody selection was followed by infection
# of CHO-bEFNB3 cells, so this is the functional-cost axis for this arm. The
# bEFNB2 / binding files are the unmatched ones and must be labelled if used.
FUNCTIONAL_CSV = DMS_DIR / "RBP_mutation_effects_cell_entry_CHO-bEFNB3.csv"

AF3_MODEL_DIR = env.AF3_MODEL_DIR
AF3_DB_DIR = env.AF3_DB_DIR

WORK = PROJECT / "work" / "niv"
ANTIGEN_FASTA = WORK / "antigen_532.fa"             # written by niv_02; see above
CHAIN_JOBS = WORK / "chain_jobs"
MSA_RAW = WORK / "msa_raw"
MSA_CACHE = WORK / "msa_cache"
AF3_OUT = WORK / "af3_out"
SCORES = WORK / "scores"
REF_DIR = WORK / "ref"                              # deposited complexes, gitignored

MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "niv_manifest.csv"
ESCAPE_NPZ = MANIFESTS / "niv_escape_sites.npz"
ESCAPE_JSON = MANIFESTS / "niv_escape_sites.json"
# The solved 4.5 A contact sets as fold-frame masks, keyed `{antibody}__gold`,
# the same spelling rsv_labels.npz uses. Written by
# `niv_01_escape_sites.py --labels`, which is also where they are derived: the
# counts behind them were always in niv_escape_sites.json (`epitope_size`), but
# the per-residue vector a recall panel needs did not exist on disk until
# 2026-11-04. All four foldable antibodies have one.
LABELS_NPZ = MANIFESTS / "niv_labels.npz"

RESULTS = PROJECT / "results" / "niv_case"
FIGURES = RESULTS / "figures"

ARM_TAG = "niv"
ANTIGEN_NAME = "Nipah virus RBP (G)"
ANTIGEN_LEN = 532                  # protein.fasta minus its trailing `*`
SITE_OFFSET = 70                   # fasta index i (1-based) == reference site i + 70
ANTIGEN_COPIES = 1                 # one protomer; see the quaternary caveat above

CHAIN_ANTIGEN, CHAIN_HEAVY, CHAIN_LIGHT = "A", "H", "L"
MODEL_SEEDS = [1, 2, 3, 4, 5]
DIFFUSION_SAMPLES = 5
MAX_TEMPLATE_DATE = "2021-09-30"
TRIM_LIGHT_TO_VARIABLE = True
MIN_LIGHT_LEN = 80
CONTACT_CUTOFF = 4.5

PYTHON = env.PYTHON

# All four cases have a solved complex -- that is the point of the arm, and it is
# what lets the declared escape rule be checked against real contacts. Two of the
# four are HENDRA complexes, so "experimental" here is cross-species for those.
HAS_EXPERIMENTAL = True

# n = 4. Do not add a test. See the module docstring.
SIGNIFICANCE_TEST = None
