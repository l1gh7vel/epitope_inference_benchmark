"""
Shared helpers for the Task II AF3 pipeline.

Nothing here does I/O on its own; the numbered scripts drive everything. Keeping
sequence cleaning / chain-keying in one place matters because a chain's cache key
is a hash of its sequence -- if two scripts clean a sequence differently, the MSA
cache silently misses.
"""

import hashlib
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
# Which task's config to load. Stages 01-04 (chain jobs, MSA cache, AF3 assembly,
# structure scoring) are task-agnostic, so Task III reuses them unchanged:
#     EIB_TASK=task3 python scripts/01_make_chain_jobs.py
import importlib  # noqa: E402
import os  # noqa: E402
cfg = importlib.import_module(os.environ.get("EIB_TASK", "task2"))

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Heavy vs light V-domain discrimination by the framework-2 motif at ~Kabat 36-39.
# This is far more reliable than the N-terminus: these sequences come from
# structures, so 1-2 leading residues are often unresolved (QVQLVE -> VQLVE),
# which defeats an anchored ^[EQ]V[QK]L test on ~40% of rows for no good reason.
#   heavy : WVRQ (human), WVKQ/WMKQ (murine), WFRQ/WYRQ (camelid VHH)
#   light : WYQQ / WFQQ
FR2_HEAVY = re.compile(r"W[VIAFYLGM][RK]Q")
FR2_LIGHT = re.compile(r"W[YF][QLH]Q")

# N-terminus of the antibody light *constant* domain (CL). One regex covers
# human kappa (PSVFIFPPS), mouse kappa (PTVSIFPPS) and lambda (PSVTLFPPS /
# PTVTLFPPS). The CL is preceded by a ~6-residue linker (RTVAAP, RADAAP,
# GQPKANP), so we cut that far upstream to end cleanly at the V domain.
CL_START = re.compile(r"P[ST]V[TFS][ILS]FPPS")
CL_LINKER = 6

# Heavy-chain CH1 N-terminus (human IgG: ASTKGPSVFPLAP; the leading residue is
# sometimes mutated (PSTKGPSV, SSTKGPSV), so we anchor on STKGPSV.PLAP. Cuts a full-length heavy
# chain down to its variable domain. NB: trim by MOTIF, never by length -- bovine
# ultralong-CDRH3 antibodies (several HIV bnAbs in CATNAP) have genuine ~165 aa
# variable domains, and a length rule would amputate their binding region.
CH1_START = re.compile(r"[A-Z]?STKGPSV[FL]PLAP")


def read_fasta_single(path):
    """Return the sequence from a single-record FASTA, whitespace stripped."""
    lines = Path(path).read_text().splitlines()
    return "".join(l.strip() for l in lines if l and not l.startswith(">"))


def clean_seq(s):
    """Uppercase, strip all whitespace. Returns (sequence, n_nonstandard)."""
    if s is None:
        return "", 0
    s = "".join(str(s).strip().upper().split())
    if s in ("NAN", "NONE"):
        return "", 0
    return s, sum(1 for c in s if c not in VALID_AA)


def trim_light_to_variable(seq):
    """
    Drop the light-chain constant domain, returning (trimmed, was_trimmed).

    Leaves the sequence untouched if no CL motif is found, so a light chain that
    is already a variable domain passes through unchanged.
    """
    m = CL_START.search(seq)
    if not m:
        return seq, False
    cut = max(0, m.start() - CL_LINKER)
    return seq[:cut], True


def trim_heavy_to_variable(seq):
    """
    Drop the heavy-chain constant region (CH1 onward), returning (trimmed, was_trimmed).

    Some CATNAP rows carry the *entire* IgG heavy chain (~460 aa) rather than just
    VH. Folding the constant domains wastes tokens and is not what the paratope
    needs. Cut at the CH1 N-terminus; if no CH1 motif is present the sequence is
    returned untouched, so genuine long variable domains (bovine ultralong CDR-H3)
    survive intact.
    """
    m = CH1_START.search(seq)
    if not m:
        return seq, False
    return seq[:m.start()], True


def domain_kind(seq):
    """
    Classify an antibody V domain as 'heavy', 'light', 'both' or 'unknown'.

    Catches rows where the light chain was written into the vh_sequence
    column. Such rows are flagged and never silently dropped, because what to do
    with them is an analysis decision, not a parsing one.
    """
    if not seq:
        return "none"
    h, l = bool(FR2_HEAVY.search(seq)), bool(FR2_LIGHT.search(seq))
    if h and not l:
        return "heavy"
    if l and not h:
        return "light"
    return "both" if h else "unknown"


def chain_key(kind, seq):
    """
    Stable cache key for one unique chain sequence, e.g. 'vh_1f3a9c2b0d'.

    The hash is over the exact string that goes into the AF3 JSON, so identical
    sequences share one MSA no matter which rows they came from.
    """
    return f"{kind}_{hashlib.sha1(seq.encode()).hexdigest()[:10]}"


def af3_protein(chain_id, seq, msa_block=None):
    """
    Build one AF3 'protein' entry.

    With msa_block (a cached chain dict carrying unpairedMsa / pairedMsa /
    templates) the entry is complete and AF3 can run with --norun_data_pipeline.
    Without it, AF3 will run its own data pipeline for this chain.
    """
    entry = {"id": chain_id, "sequence": seq}
    if msa_block:
        for field in ("unpairedMsa", "pairedMsa", "templates"):
            if field in msa_block:
                entry[field] = msa_block[field]
    return {"protein": entry}


def af3_job(name, proteins, seeds=None):
    """Wrap protein entries into a complete AF3 input JSON object."""
    return {
        "name": name,
        "sequences": proteins,
        "modelSeeds": list(seeds if seeds is not None else cfg.MODEL_SEEDS),
        "dialect": "alphafold3",
        "version": 1,
    }


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def window_mask(lo, hi, length=None):
    """Boolean mask over 1-based positions [lo, hi] inclusive."""
    length = length or cfg.ANTIGEN_LEN
    pos = np.arange(1, length + 1)
    return (pos >= lo) & (pos <= hi)


def load_labels():
    """
    Load gold epitope labels and structure masks written by 00_build_manifest.py.

    Returns (case_ids, labels[N,1273] uint8, obs_mask[N,1273] uint8) where
    obs_mask marks residues actually resolved in the source experimental
    structure.
    """
    z = np.load(cfg.LABELS_NPZ, allow_pickle=True)
    return list(z["case_id"]), z["labels"], z["obs_mask"]


# --------------------------------------------------------------------- #
#  Antigen identity: "is this AsEP antigen OUR antigen?"                 #
# --------------------------------------------------------------------- #
# Builds the leakage pool, because only training complexes bearing our antigen
# can leak our epitope. Two traps, both of which silently
# UNDER-count leakage (the dangerous direction: it inflates the 'clean' set):
#
#  1. difflib's `autojunk=True` (the DEFAULT) treats any character occurring in
#     >1% of a sequence >=200 long as junk. A protein alphabet has 20 letters, so
#     essentially EVERY residue is 'popular' and get_matching_blocks() returns
#     nothing. Measured: a gp120 core vs Env gave coverage 0.000 with autojunk on,
#     0.762 with it off. Always pass autojunk=False for sequences.
#  2. A global similarity threshold cannot work when the construct and the
#     reference differ in length: quick_ratio() is bounded by 2*len(a)/(len(a)+
#     len(b)), so a 353 aa gp120 core vs 856 aa gp160 maxes out at 0.584 and can
#     never clear 0.85 however identical it is.
#
# So we measure CONTAINMENT: the fraction of the AsEP antigen explained by our
# reference. Similarity would depend on length; containment is length-agnostic and handles constructs
# (RBD-only, gp120 core, gp41). Separation is clean and bimodal -- for Spike the
# count is 402 at every threshold from 0.15 to 0.50, nothing in between -- so the
# exact cut is not delicate.
MIN_BLOCK = 4          # ignore incidental 1-3 residue coincidences
ANTIGEN_COV_THRESH = 0.30


def antigen_coverage(ag, ref, min_block=MIN_BLOCK):
    """Fraction of `ag` covered by contiguous blocks (>= min_block) shared with `ref`."""
    if not ag:
        return 0.0
    sm = SequenceMatcher(None, ag, ref, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks() if b.size >= min_block) / len(ag)


def same_antigen(ag, ref, min_len=20, thresh=ANTIGEN_COV_THRESH):
    """Is this AsEP antigen our antigen (full length, a construct, or a domain)?"""
    if not ag or len(ag) < min_len:
        return False
    return antigen_coverage(ag, ref) >= thresh
