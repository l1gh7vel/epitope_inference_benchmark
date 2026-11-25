#!/usr/bin/env python3
"""A worked example: score six antibodies and run the antibody-swap test.

Six SARS-CoV-2 spike antibodies from Task II, with AlphaFold3's per-residue
scores and their gold epitopes, in `example_six_antibodies.npz`. They were
chosen across the range of the full cohort: two the method recovers almost
exactly, two near the median and two at the random floor. That spread is the
shape the benchmark is about, and six antibodies are enough to show it.

The demo computes the two quantities the manuscript rests on.

  AUPRC per antibody, on `contact_top`, which reads the top-ranked AlphaFold3
  structure and is the headline readout of this benchmark. It is computed
  against that antibody's own gold epitope over all 1,273 spike positions, with
  unresolved positions counted as non-epitope.

  The antibody-swap gap. Each prediction is held fixed and graded a second time
  against the other antibodies' epitopes; the gap is the first minus the mean of
  the second. A prediction that does not depend on the antibody scores a gap of
  zero, by algebra rather than by approximation, which the random control at the
  end demonstrates.

Runs in under a minute on one core. No GPU, no structures, no downloads.

    python demo/run_demo.py

The printed values should match `expected_output.txt` exactly. If they do not,
the environment differs from the one the published numbers came from, and
The README names the environment those numbers came from.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "example_six_antibodies.npz"


def swap_gap(cases, score, gold):
    """Own-epitope AUPRC minus the mean over the other antibodies' epitopes.

    Antibodies sharing an identical epitope are excluded from each other's cross
    set. The cross term is meant to be a stranger's epitope, and an identical
    one is not.
    """
    key = {c: gold[c].tobytes() for c in cases}
    out = {}
    for c in cases:
        own = average_precision_score(gold[c], score[c])
        cross = [average_precision_score(gold[o], score[c])
                 for o in cases if key[o] != key[c]]
        out[c] = (own, float(np.mean(cross)))
    return out


def main():
    if not DATA.exists():
        sys.exit(f"missing {DATA}")
    z = np.load(DATA, allow_pickle=True)
    cases = [str(c) for c in z["case_ids"]]
    score = {c: z[f"{c}__score"].astype(float) for c in cases}
    gold = {c: z[f"{c}__gold"].astype(bool) for c in cases}

    print(f"{len(cases)} antibodies, {len(next(iter(score.values())))} antigen "
          "positions each\n")
    print(f"{'antibody':<16}{'epitope':>8}{'AUPRC':>9}{'floor':>8}"
          f"{'cross':>9}{'gap':>9}")
    gaps = swap_gap(cases, score, gold)
    for c in cases:
        own, cross = gaps[c]
        floor = gold[c].mean()
        print(f"{c:<16}{int(gold[c].sum()):>8}{own:>9.4f}{floor:>8.4f}"
              f"{cross:>9.4f}{own - cross:>+9.4f}")

    mean_gap = float(np.mean([o - x for o, x in gaps.values()]))
    print(f"\nmean antibody-swap gap: {mean_gap:+.4f}")

    # The control. One fixed vector for every antibody is what an antibody-blind
    # method produces, and its gap cancels to zero.
    rng = np.random.default_rng(0)
    fixed = rng.random(len(next(iter(score.values()))))
    blind = swap_gap(cases, {c: fixed for c in cases}, gold)
    blind_gap = float(np.mean([o - x for o, x in blind.values()]))
    # The cancellation is exact in arithmetic and lands near 1e-18 in floating
    # point, whose SIGN depends on summation order. Printed with an explicit
    # sign that would read as -0.0000 on one machine and +0.0000 on another,
    # and this output is compared byte for byte in CI.
    if abs(blind_gap) < 1e-12:
        blind_gap = 0.0
    print(f"the same gap for one fixed vector: {blind_gap:+.4f}")


if __name__ == "__main__":
    main()
