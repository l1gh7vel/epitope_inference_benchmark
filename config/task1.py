"""
Task I (antigen-split generalisation) configuration.

Tasks II and III hold the antigen fixed and vary the antibody. Task I does the
opposite: it sweeps across the whole of SAbDab and asks whether a method's
accuracy is stable across antigen families, or whether it is carried by a few
over-represented antigens. SAbDab is heavily skewed -- 857 of the 4,585
complexes carry SARS-CoV-2 spike (UniProt P0DTC2), the antigen the other two
tasks are built on -- so a method tuned to spike can look strong on a pooled
average and weak everywhere else.

The split comes with the source data. Complexes are clustered by antigen
sequence identity and the clusters are dealt into five antigen groups, carried
in the manifest's `fold` column. The groups share no antigen cluster, so no
antigen family spans a group boundary. They are not antibody-disjoint: 223 VH
sequences appear in more than one group, across 658 rows, recorded per row as
`vh_in_other_fold` and reported rather than silently fixed.

Nothing in this benchmark is trained, so the groups are evaluation strata, not
train/test splits. Reporting per-group AUPRC as a distribution answers whether a
method's score is a property of the method or of the antigens it was shown,
which a single pooled number cannot.

Methods that produce a structure run on the evaluated set: the usable complexes
whose antigen fits the length cap, 3,908 of them (see MAX_ANTIGEN_LEN).
BepiPred-3.0 reads the antigen sequence alone and runs on the full usable set.
Comparisons across methods use the intersection of cases every method scored
(INTERSECTION_METHODS); each method is also reported on its own full arm. The
RBD and RBM region masks do not apply here, because Task I's antigens are
arbitrary and there is no fixed region to mask; its antibody-blind baselines are
BepiPred-3.0 and random. Cross-method claims use the intersection arm,
per-method claims that method's own full arm, and each number states which.
"""

import os
from pathlib import Path

import env

# ---------------------------------------------------------------- paths ----
# Project root. EIB_ROOT overrides it; the fallback is this file's own repo
# location. Resolve every path from here rather than from a literal, so one
# definition moves with the repository.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT / "data" / "source"

# The source group CSVs, vendored by scripts/t1_00_build_manifest.py --vendor.
# Upstream is read-only and never written to; its location is EIB_UPSTREAM_FOLDS
# and the sha256 of each vendored copy is in data/source/task1_folds/PROVENANCE.txt.
FOLDS_UPSTREAM = env.UPSTREAM_FOLDS
FOLDS_DIR = DATA_DIR / "task1_folds"
N_FOLDS = 5

AF3_MODEL_DIR = env.AF3_MODEL_DIR
AF3_DB_DIR = env.AF3_DB_DIR

WORK = PROJECT / "work" / "task1"
CHAIN_JOBS = WORK / "chain_jobs"
MSA_RAW = WORK / "msa_raw"
MSA_CACHE = WORK / "msa_cache"
AF3_OUT = WORK / "af3_out"
SCORES = WORK / "scores"

MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "task1_manifest.csv"
LABELS_NPZ = MANIFESTS / "task1_labels.npz"      # ragged: one array per case

RESULTS = PROJECT / "results" / "task1"

# ------------------------------------------------------------- antigen ----
# Unlike Tasks II and III there is no single antigen: each case carries its own.
HAS_EXPERIMENTAL = True          # every Task I case IS a solved complex
ANTIGEN_NAME = "SAbDab (all antigens)"
# There is no task-level antigen length: every row carries its own. Stage 04 skips
# its chain-length assertion when this is None, and the Task I adapters do the
# per-row check instead -- a mismatch there is refused, never trimmed, because
# trimming would shift every residue index against that row's gold labels.
ANTIGEN_LEN = None

# Chain ids, matching Task III: A = antigen, H = VH. There is no light chain in
# this dataset, so CHAIN_LIGHT exists only because the shared stage-04 splitter
# reads it; nothing here ever writes an L chain.
# Heavy-atom contact cutoff, in angstroms. 4.5 is the WALLE/AsEP convention and is
# what Tasks II and III use; it must match or "contact" means a different thing in
# each task and no number is comparable across them.
CONTACT_CUTOFF = 4.5

CHAIN_ANTIGEN = "A"
CHAIN_HEAVY = "H"
CHAIN_LIGHT = "L"

# AF3 token bucket. VH (~120) + antigen must fit in the 1,536-token bucket Task II
# and Task III already use, so runtime per complex is already characterised.
# 1,400 is chosen deliberately, not for headroom: full-length spike is 1,273 aa and
# FOLD 1 IS ALMOST ENTIRELY SPIKE (916 of 917 rows, median antigen length 1,273,
# only 29 antigen clusters against ~280 in every other fold). A 1,200 cap silently
# deletes fold 1 -- it drops that fold from 916 eligible complexes to 42 -- which
# would remove the single most informative fold from the AF3 arm. At 1,400 the
# eligible set is 4,227 of 4,516 usable, evenly spread across folds.
MAX_ANTIGEN_LEN = 1400
# Below this an AUPRC is dominated by one or two residues and is mostly noise.
MIN_EPITOPE_RESIDUES = 3

# The methods whose intersection defines the cross-method arm.
INTERSECTION_METHODS = ["AlphaFold3", "BepiPred-3.0", "ESMFold2", "WALLE"]

# ---------------------------------------------------------------- AF3 run ----
# SAME as Tasks II and III (5 seeds x 5 diffusion samples = 25 structures), on
# purpose. The headline readout is the top-ranked structure, so a smaller ensemble
# would have been cheaper -- but then Task I's AlphaFold3 number would not be the
# same quantity as Task II's and Task III's, and the 25-structure ensemble is also
# reported as a marker. Comparability across tasks is worth the inference time
# (~100 s per complex, so ~14 GPU-hours for the 500).
MODEL_SEEDS = [1, 2, 3, 4, 5]
DIFFUSION_SAMPLES = 5
