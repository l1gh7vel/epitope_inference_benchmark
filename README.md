# epitope_inference_benchmark

A benchmark and meta-analysis of whether epitope prediction is
antibody-conditioned: does the predicted site move when the antibody changes?

It evaluates AlphaFold3, ESMFold2, WALLE and BepiPred-3.0. Nothing is trained
here. BepiPred-3.0 and fixed antigen-region masks provide antibody-blind
comparisons, and an antibody-blind method has an antibody-swap gap of exactly
zero, which is the anchor that says the conditioning test is wired up correctly.

## Tasks

| task | question | evaluated set |
|---|---|---|
| I | Does accuracy vary across antigen sequence groups? | 3,804 complexes scored by every method |
| II | On one antigen, does the prediction follow the antibody? | 186 distinct antibody inputs on all 1,273 SARS-CoV-2 spike residues |
| III | Can a method localise binding to the correct antigen domain without a solved complex? | 300 antibodies on HIV-1 Env and 300 on SARS-CoV-2 spike |

The functional analyses compare predictions with published mutational escape
measurements for HIV-1 Env, RSV F and Nipah virus receptor-binding protein.

## Contents

```text
config/          task definitions and portable path configuration
data/manifests/  frozen evaluated sets and per-residue label arrays
data/source/     provenance for the external inputs, which are not redistributed
demo/            six-case CPU-only example
results/         per-case tables and summary statistics
scripts/         analysis and figure programs
```

## What reproduces, and what does not

Every number in the paper is computed from the per-case tables in `results/`,
and those tables ship. So a clone rebuilds every figure and every summary
statistic without downloading anything.

Recomputing the per-case tables themselves is a different matter. They are
derived from per-residue score arrays, one per antibody-antigen pair, which come
to roughly 170 GB across all arms and are not published. Nor are the source
tables used to build the manifests, the deposited and predicted structures, the
alignments, the model weights or the AlphaFold3 databases. Getting back to the
per-case tables therefore means re-running structure prediction and scoring with
the inputs named in `data/source/PROVENANCE.md`, which records the source,
immutable identifier, SHA-256 checksum, terms and expected local filename for
each one.

Every program that needs what is not here says so and stops without writing, so
a missing input is never silently absorbed into a smaller result.

## Installation

```bash
conda env create -f environment.yml
conda activate epitope-bench
```

Python 3.14 with numpy, pandas, scipy, scikit-learn, matplotlib, gemmi and
freesasa. Structure prediction and embedding additionally need an older Python
and torch, obtained from each tool's own distribution: AlphaFold3 (weights from
Google DeepMind under their licence), ESMFold2, BepiPred-3.0, WALLE with the
AsEP checkpoints (`pip install git+https://github.com/biochunan/AsEP-dataset`),
ESM-2 and ProteinBERT.

Machine-specific locations are read from `EIB_*` environment variables, resolved
in `config/env.py`: `EIB_ROOT`, `EIB_PYTHON`, `EIB_PYTHON_GPU`,
`EIB_AF3_MODEL_DIR`, `EIB_AF3_DB_DIR`, `EIB_AF3_MODULE`, `EIB_ESM_PREFIX`,
`EIB_BEPIPRED_CLI`, `EIB_HF_HOME`. Set only the ones the stage you are running
needs; a stage that needs an unset variable stops and names it.

## Demo

```bash
python demo/run_demo.py
```

Under a minute on one core, no GPU and no downloads. It scores six SARS-CoV-2
spike antibodies, runs the antibody-swap test on them, and prints values that
should match `demo/expected_output.txt` exactly. If they do not, the environment
differs from the one the published numbers came from.

## Recomputing the reported results

These run on a clone as it is, reading the per-case tables and writing the
figure panels to `results/paper_figures/`:

```bash
PYTHONPATH=scripts:config python scripts/12_paper_figures.py
PYTHONPATH=scripts:config python scripts/18_task1_figures.py
PYTHONPATH=scripts:config python scripts/33_task3_per_domain.py
```

The evaluators that produce the per-case tables read the per-residue score
arrays, so they need a populated `work/` and will stop without it. `EIB_TASK`
selects which task module they use:

```bash
PYTHONPATH=scripts:config python scripts/05_evaluate_task2.py
EIB_TASK=task3 PYTHONPATH=scripts:config python scripts/t3_05_evaluate.py
EIB_TASK=task3cov PYTHONPATH=scripts:config python scripts/t3_05_evaluate.py
```

The pipeline is staged, and each stage is one program: `00_build_manifest.py`
defines the evaluated set, `01_make_chain_jobs.py` emits one alignment job per
unique chain, `02_build_msa_cache.py` harvests the alignments,
`03_assemble_af3_input.py` assembles one complex from cached chains,
`04_score_structures.py` turns structures into per-residue scores, and
`05_evaluate_task2.py` turns scores into metrics. Task I has `t1_*` variants,
because every complex there carries a different antigen. Each module's docstring
carries its parameters and its cost.

Roughly 99% of AlphaFold3's wall clock is the sequence search, about 25 minutes
for spike at 8 CPUs against about 100 seconds of inference, so each unique chain
is searched once and cached. Inference averages 519 s per complex and rises with
length as about length^1.5. Predicted structures and the alignment cache come to
roughly 170 GB across all arms.

## Evaluation conventions

Gold epitopes are antigen residues with a heavy atom within 5 Å of an antibody
heavy or light chain. `contact_top` is the primary structural score, the maximum
contact score across the structure ensemble at each residue; `contact_mean`
averages the ensemble and is reported beside it. Per-case AUPRC is primary for
residue-level prediction, with precision at the gold epitope size and
rank-biserial correlation secondary. Random AUPRC equals the positive-residue
prevalence of each case. Task III averages residue scores within each configured
domain and assigns the domain with the highest mean. WALLE uses
`treasured-sweep-8` in every task; the checkpoint is fixed by the Task II
validation rule and is not selected separately for each test set.

Tests are one-sided only where the stated alternative is directional. Multiple
tests within a panel use the correction stated in its legend. Per-antibody
distributions are right-skewed, so every summary states whether it is a mean or
a median, and the two are never reconciled into one number.

| quantity | n | use |
|---|---:|---|
| Task I complexes | 3,804 | scored by every method; the comparison set |
| Task I antigens | 1,403 | antigen-sequence grouping |
| Task I composition set | 4,391 antibodies; 1,130 clusters | benchmark composition panels |
| Task II merged inputs | 193 | one row per distinct antibody input |
| Task II scored inputs | 186 | headline analysis |
| Task II RBM | 94 | epitope intersects spike residues 436--507 |
| Task II non-RBM-in-RBD | 45 | epitope lies within residues 319--541 and outside the RBM |
| Task II outside-RBD | 45 | epitope includes a residue outside residues 319--541 |
| Task II ambiguous | 2 | epitope straddles the RBM boundary |
| Task III HIV-1 Env | 300 | every method; CD4bs 116, V3 99, V2 45, MPER 27, FP 13 |
| Task III spike | 300 | every method; NTD 100, RBD 100, S2 100 |

Task I compares methods over the 3,804 complexes every method scored. Each
method is also reported on its own evaluated cases, in
`results/task1/metrics_<method>_full.json`. A number from one arm is not
comparable with a number from the other.

## Licence and terms

The code and documentation in this repository are released under the MIT
licence. External datasets retain the terms of their sources, recorded per input
in `data/source/PROVENANCE.md`. AlphaFold3 weights are distributed separately by
Google DeepMind; results derived from AlphaFold3 output are subject to the
[AlphaFold3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md)
and should cite Abramson et al., *Nature* (2024),
<https://doi.org/10.1038/s41586-024-07487-w>.
