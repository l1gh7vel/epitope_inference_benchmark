#!/usr/bin/env python3
"""
The antibody-swap conditioning test.
=====================================

The question it answers
-----------------------
An epitope predictor can score well on average for the wrong reason: if one site on
an antigen is immunodominant, always pointing at that site is right often enough to
look competent. Such a model is not predicting "where does this antibody bind" -- it
is predicting "where do antibodies bind on this antigen". Accuracy, AUPRC and AUROC
cannot tell those two apart, because both produce good scores.

The swap test separates them directly. For each antibody we score its prediction
twice:

    own   = the prediction scored against the epitope THAT antibody actually binds
    cross = the same prediction scored against epitopes other antibodies bind

    gap = mean(own - cross)

If the prediction genuinely depends on the antibody, it should fit its own antibody's
epitope better than a stranger's, so gap > 0. If the model emits (nearly) the same
answer regardless of antibody, own and cross are drawn from the same distribution and
gap ~ 0.

The reference point: an ANTIBODY-BLIND method scores a gap of EXACTLY zero, and the
algebra is worth seeing because it is the anchor for the whole test. Its prediction
is identical for every antibody, so M[i, j] = f(j) depends only on which epitope it
is graded against. With all epitopes distinct:

    own_i   = f(i)
    cross_i = (S - f(i)) / (n - 1)          where S = sum_j f(j)
    mean_i (own_i - cross_i) = S/n - (nS - S)/(n(n-1)) = S/n - S/n = 0

exactly, not approximately. (Verified numerically: -1.4e-18, i.e. float noise.)

The ONE thing that perturbs it is the duplicate-epitope exclusion below. When several
antibodies share an identical gold epitope they are excluded from one another's cross
set as a group, which removes a different number of terms for different antibodies and
breaks the cancellation by a tiny amount. In the self-check at the bottom, 5 of 60
antibodies share an epitope and the blind gap comes out at -0.00008 instead of 0. The
sign is arbitrary -- it depends on whether the excluded members happen to sit above or
below the mean.

So: zero is a hard algebraic anchor, not an empirical approximation. The permutation
null below is still needed, but for a different reason -- to calibrate how large a gap
a genuinely conditioned-looking but actually chance arrangement could produce.

The permutation null
--------------------
A positive gap can arise by chance: epitopes differ in size and in how "easy" they
are, so even a random score vector can fit some epitopes better than others. We
therefore compare the observed gap against a null built by permuting which epitope
is assigned as "own".

The permutation holds fixed everything that is not the effect of interest -- the
predictions, the set of epitopes, their sizes, their difficulty -- and destroys only
the correspondence between antibody and epitope. That is exactly the null hypothesis
"the predicted site does not depend on which antibody was given".

The cross term is recomputed inside every permutation. Holding it fixed would test
the own term alone rather than the own-minus-cross gap. See `cross_for()` below.

Antibodies that share an identical gold epitope are excluded from each other's
cross set. Scoring a prediction against a copy of its own epitope is not a swap,
and counting it as one inflates cross and shrinks the gap. We group by exact
epitope and exclude the whole group.

Reading the output
------------------
    gap        the effect size, in the same units as the per-pair score (here AUPRC)
    p, z       is the gap larger than permutation chance
    win_rate   the fraction of individual antibodies with own > cross

win_rate deserves attention. A gap can be positive in aggregate while most individual
antibodies show no effect, if a minority show a large one. Report both: "real in
aggregate, uneven per case" is a different claim from "conditions on every antibody".

INPUTS
------
scores : (n_antibodies, n_positions) float
    One predicted score per antigen position per antibody. Higher = more likely
    epitope. Any monotone score works; nothing here assumes a probability.
labels : (n_antibodies, n_positions) {0,1}
    The gold epitope of each antibody, on the SAME antigen coordinate frame. All
    antibodies must share one frame or the swap is meaningless.

The pairwise score function is AUPRC here, but the test is agnostic -- substitute any
"how well does this score vector fit this label vector" function.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def auprc(y, s):
    """Area under the precision-recall curve. Undefined without both classes."""
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return float(average_precision_score(y, s))


def swap_test(scores, labels, n_perm=1000, seed=0, pair_fn=auprc):
    """Run the antibody-swap conditioning test.

    Returns (summary dict, per-antibody DataFrame).
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(int)
    n = len(scores)
    rng = np.random.default_rng(seed)

    # M[i, j] = how well antibody i's PREDICTION fits antibody j's EPITOPE.
    # The diagonal is "own"; everything else is a potential swap.
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = pair_fn(labels[j], scores[i])

    # Antibodies sharing an identical gold epitope form one group. Scoring against a
    # copy of your own epitope is not a swap, so a group is excluded from its own
    # members' cross sets.
    gid = pd.factorize(pd.Series([labels[i].tobytes() for i in range(n)]))[0]
    n_groups = gid.max() + 1
    group_size = np.bincount(gid, minlength=n_groups)

    row_sum = M.sum(1)                                  # total over all j
    group_sum = np.zeros((n, n_groups))                 # per-group totals, for exclusion
    np.add.at(group_sum.T, gid, M.T)

    def cross_for(assigned):
        """Mean fit against every epitope DISTINCT from the assigned one.

        `assigned[i]` is the index of the epitope currently treated as antibody i's
        own. Under permutation this changes, so the excluded group changes too --
        which is why cross must be recomputed rather than held fixed.
        """
        g = gid[assigned]
        return (row_sum - group_sum[np.arange(n), g]) / (n - group_size[g])

    own = np.diag(M)
    cross = cross_for(np.arange(n))                     # identity = the true pairing
    gap = float(np.mean(own - cross))
    win_rate = float(np.mean(own > cross))

    # Null: shuffle which epitope each antibody "owns", recompute the whole gap.
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        null[p] = np.mean(M[np.arange(n), perm] - cross_for(perm))

    # +1 in numerator and denominator: a permutation test can never report p = 0,
    # only "smaller than 1/(n_perm+1)".
    p_value = float((np.sum(null >= gap) + 1) / (n_perm + 1))

    per_case = pd.DataFrame({"own": own, "cross": cross, "gap": own - cross})
    return {
        "swap_own": float(own.mean()),
        "swap_cross": float(cross.mean()),
        "swap_gap": gap,
        "win_rate": win_rate,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": p_value,
        "z": float((gap - null.mean()) / (null.std() + 1e-12)),
        "n_antibodies": n,
        "n_distinct_epitopes": int(n_groups),
    }, per_case


if __name__ == "__main__":
    # Self-check on synthetic data: the test must separate a conditioned predictor
    # from an antibody-blind one, and must give EXACTLY zero for the blind case.
    rng = np.random.default_rng(0)
    n, L, size = 60, 400, 15
    labels = np.zeros((n, L), int)
    starts = rng.integers(0, L - size, n)
    for i, s in enumerate(starts):
        labels[i, s:s + size] = 1

    # (a) conditioned: peak at this antibody's own epitope
    cond = np.stack([np.exp(-0.5 * ((np.arange(L) - (s + size / 2)) / 25.0) ** 2)
                     for s in starts]) + 0.05 * rng.random((n, L))
    # (b) antibody-blind: the SAME vector for everyone
    blind = np.tile(np.exp(-0.5 * ((np.arange(L) - L / 3) / 25.0) ** 2), (n, 1))

    for name, sc in [("antibody-conditioned", cond), ("antibody-blind", blind)]:
        st, _ = swap_test(sc, labels, n_perm=500)
        print(f"{name:22s} gap {st['swap_gap']:+.4f}  p={st['p_value']:.4f}  "
              f"win_rate {st['win_rate']:.2f}  (n={st['n_antibodies']})")
