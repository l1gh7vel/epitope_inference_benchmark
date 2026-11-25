#!/usr/bin/env python3
"""
Task I supplementary: is epitope coverage REDUNDANT across the antibodies that
SAbDab raises against one antigen?

The question, and why it is the benchmark's premise
  This benchmark exists because an epitope predictor can be antibody-BLIND -- predict one
  popular surface patch, ignore the antibody, and still score well. That strategy
  only pays if the antibodies in the data actually converge on the same patch. This
  script measures that convergence directly, on gold labels alone: no method, no
  prediction, nothing folded. If different antibodies against one antigen tile the
  surface, "predict the popular patch" loses and a pooled per-antibody average is a
  fair evaluation. If they pile onto the same patch, the pooled average cannot tell
  a conditioned model from a blind one, and Tasks II/III (antigen fixed, antibody
  varied) are the only place the distinction is visible.

What is compared, and the grouping that makes it valid
  The gold labels are per-residue vectors INDEXED INTO THAT ROW'S OWN antigen
  sequence. Two epitopes can only be intersected index-by-index if the two rows
  carry the SAME sequence, so the primary grouping is `antigen_id` (a hash of the
  exact antigen sequence), not `Cluster`. `Cluster` is a sequence-IDENTITY cluster:
  its members differ in length and in numbering, so intersecting their label
  vectors by index would silently compare residue 300 of one construct with residue
  300 of another. The cluster view is still reported (`--group cluster95`) but only
  over rows that are the same length AND >= 95% identical position-by-position --
  an alignment-free compatibility test, stated here because it is a threshold:
  below that, indices are not interchangeable and the comparison is meaningless.
  Rows that pass are pooled into a sub-group; `antigen_id` groups are a subset of
  these by construction (identity 1.0), so cluster95 can only ADD pairs.

Rows included, and two deduplications that pull in opposite directions
  * `usable` only (>= 3 epitope residues, VH > 60 aa) -- the same filter
    `t1_05_evaluate.py` uses, so the rows are the evaluated set Task I scores.
  * WITHIN-ANTIGEN VH DEDUP IS APPLIED BY DEFAULT (`--keep_duplicate_antibodies`
    turns it off). The same heavy chain deposited twice against the same antigen
    contributes a pair with Jaccard ~1.0 that is evidence about the PDB, not about
    antibody convergence. 125 of 4,516 usable rows are such repeats. Keeping them
    inflates every overlap statistic.
  * `vh_dedup_keep` IS NOT APPLIED by default (`--apply_fold_dedup` turns it on).
    That flag exists to make the five FOLDS antibody-disjoint. The folds are
    antigen-disjoint, so a VH appearing in two folds is one antibody binding two
    DIFFERENT antigens -- irrelevant to within-antigen overlap, and dropping it
    would delete a real antibody from an antigen's evaluated set for a reason that has
    nothing to do with this question. It is reported as a robustness line.

THE METRICS (two, because they answer different questions)
  For every unordered pair (A, B) of epitopes on one antigen:
    Jaccard         J  = |A n B| / |A u B|      -- symmetric agreement
    overlap coeff.  OC = |A n B| / min(|A|,|B|) -- "is the smaller epitope a subset
                                                   of the larger", which is the
                                                   shape a blind predictor exploits
  Thresholds, chosen and stated rather than tuned:
    OC >= 0.50  "overlapping"      -- the majority of the smaller epitope is shared
    J  >= 0.80  "near-identical"   -- essentially the same patch
    J  >= 0.25  reported for continuity with the only prior overlap number in this
                repo: the first deck's own overlap panel flagged a merged
                Task II case below jaccard_min 0.25 as worth inspecting. The DMS arm's
                `results/dms_case/epitope_overlap_jaccard.csv` (dms_03_premise.py)
                computes the same set Jaccard for 9 HIV bnAbs; this script is that
                statistic generalised to every SAbDab antigen, plus a null.
  MEAN AND MEDIAN ARE REPORTED SEPARATELY AND NEVER AVERAGED TOGETHER (project
  rule 1): the pair distribution is bimodal-ish -- a mass of near-zero pairs and a
  mass of near-duplicates -- so the two numbers say different true things.

TWO VIEWS, BECAUSE ONE ANTIGEN OWNS THE DATA (project rule: say which pool)
  * PER-PAIR  : every within-antigen pair pooled. 96% of all pairs are SARS-CoV-2
                spike, so this view is essentially "what does spike look like".
  * PER-ANTIGEN: each antigen contributes ONE number (its median over its own
                pairs), then summarised across antigens. Spike is one row here.
  Neither is wrong; quoting one without saying which is.

THE NULLS -- the load-bearing choice, because overlap alone means nothing
  Three, each destroying a different thing. All share the fixed seed below.
  N1 SEGMENT-SHIFT (primary). Decompose each observed epitope into its contiguous
     runs along the sequence, then re-place those runs, same lengths, same count,
     uniformly at random and non-overlapping on the same antigen. Preserves epitope
     size and its sequence-level fragmentation (a 3D patch is a handful of linear
     segments); destroys only WHERE on the antigen it sits. So the question it
     answers is exactly "do these antibodies agree on a LOCATION".
  N2 CROSS-ANTIGEN TRANSPLANT. Replace each epitope with one drawn from a different
     antigen's repertoire, mapped on by proportional index rescaling
     (i -> floor(i * L_here / L_donor)). Preserves the global shape of real
     epitopes; destroys the fact that these epitopes were raised against THIS
     antigen. Caveat: rescaling changes sizes slightly, so it is the secondary null.
  N3 UNION-RESTRICTED (deliberately conservative, antigens with >= 5 antibodies
     only). Resample each epitope uniformly from the UNION of all observed epitopes
     on that antigen. It asks: given an epitope has to land somewhere antibodies
     ever bind, is the agreement still above chance? It is partly circular -- the
     union is built from the same epitopes, so for small evaluated sets the null is forced
     to look like the data, which is why it is restricted to n >= 5 and read as a
     floor, not as the answer.
  TEST: per-antigen paired Wilcoxon signed-rank, ONE-SIDED (observed > null), over
  antigens -- directional because the hypothesis has a direction (project rule 7).

SATURATION (does coverage spread as antibodies accumulate?)
  For each antigen, average over 200 RANDOM ORDERINGS of its antibodies (so the
  curve is not an artefact of row order) the size of the union of the first j
  epitopes. Two curves: a FIXED-COMPOSITION one over antigens with >= 10
  antibodies (so the antigen set does not change along the x axis -- a curve over
  "all antigens with >= j" silently swaps the population at every step), and an
  all-antigen one with its own n printed at each j. Union is reported both as a
  fraction of antigen length and as a fraction of that antigen's final union.

SURFACE / EXPOSURE -- NOT COMPUTED, AND NOT FAKED
  The answerable form of "is the overlap more than you would expect given epitopes
  must be on the surface" needs a solved structure and SASA per antigen. This repo
  has deposited structures for 493 PDBs only (`work/task1/walle/pdb`) against 1,627
  usable antigens, so a surface-restricted null is NOT cheaply available. N3 is the
  structure-free stand-in: the observed union is an empirical lower bound on the
  antigenic surface. Stated as a caveat, not as a surface result.

Evaluated set
  All 4,516 usable rows. This is where the question has n.

    PYTHONPATH=scripts:config python scripts/t1_06_epitope_redundancy.py
    PYTHONPATH=scripts:config python scripts/t1_06_epitope_redundancy.py \
        --group cluster95
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "config"))
import task1 as cfg                                        # noqa: E402

SEED = 0                 # fixed; every null and every ordering draws from it
N_NULL_REPS = 50         # null replicates per antigen (each one re-places EVERY
                         # epitope on that antigen, so the pooled pair count per
                         # replicate is already large; 50 leaves Monte-Carlo error
                         # far below the observed/null separation)
MAX_NULL_PAIRS = 2_000_000   # cap on null pairs per group: replicates are reduced
                         # for big evaluated sets, where one replicate is already 3e5 pairs
N_ORDERS = 200           # random orderings for the saturation curve
OC_OVERLAP = 0.50        # "overlapping": majority of the smaller epitope shared
J_NEAR_IDENT = 0.80      # "near-identical patch"
J_PRIOR_ART = 0.25       # continuity with the flag line of the first deck
ARM = "full"            # label carried in the output files
IDENTITY_MIN = 0.95      # cluster95: index-comparability threshold
MIN_N_UNION_NULL = 5     # N3 is too circular below this evaluated set size
SATURATION_MIN_N = 10    # fixed-composition saturation curve


# ------------------------------------------------------------------ data ----
def load(keep_dup_ab, apply_fold_dedup):
    man = pd.read_csv(cfg.MANIFEST)
    n_all = len(man)
    man = man[man.usable]
    if apply_fold_dedup:
        man = man[man.vh_dedup_keep]
    if not keep_dup_ab:
        n_before = len(man)
        man = (man.sort_values(["antigen_id", "vh_seq", "n_epitope", "case_id"],
                               ascending=[True, True, False, True])
                  .drop_duplicates(["antigen_id", "vh_seq"]))
        print(f"  within-antigen VH dedup: {n_before - len(man)} repeat depositions "
              f"of the same heavy chain on the same antigen dropped")
    z = np.load(cfg.LABELS_NPZ, allow_pickle=True)
    labels = {c: np.asarray(z[c], dtype=np.int8) for c in man.case_id}
    print(f"  roster: {len(man)} rows of {n_all} ({man.antigen_id.nunique()} "
          f"distinct antigens, {man.Cluster.nunique()} clusters)")
    return man, labels


def groups_antigen(man):
    """One group per EXACT antigen sequence. Indices are trivially comparable."""
    return {f"{ag}": g for ag, g in man.groupby("antigen_id") if len(g) >= 2}


def groups_cluster95(man):
    """Sequence-identity clusters, split so every member of a group is index-
    comparable with the group's representative: same length and >= IDENTITY_MIN
    positionwise identity. Greedy against the first-seen representative, which is
    deterministic given the sort and avoids transitive chaining (A~B, B~C, A!~C)."""
    out = {}
    for cl, g in man.groupby("Cluster"):
        if len(g) < 2:
            continue
        for L, gl in g.groupby("antigen_len"):
            reps = []                               # [(seq_array, [row indices])]
            for idx, row in gl.sort_values("case_id").iterrows():
                s = np.frombuffer(row.antigen_sequence.encode(), dtype=np.uint8)
                for rseq, members in reps:
                    if float((rseq == s).mean()) >= IDENTITY_MIN:
                        members.append(idx)
                        break
                else:
                    reps.append((s, [idx]))
            for k, (_s, members) in enumerate(reps):
                if len(members) >= 2:
                    out[f"cl{cl}_L{L}_{k}"] = man.loc[members]
    return out


# ----------------------------------------------------------- pair metrics ----
def pair_stats(M):
    """M: (n, L) 0/1. Returns flat arrays of Jaccard and overlap coefficient over
    all unordered pairs. Done as a matmul because the spike group is 800 x 1273."""
    Mf = M.astype(np.float32)
    inter = Mf @ Mf.T
    s = np.diag(inter).copy()
    iu = np.triu_indices(M.shape[0], 1)
    i_ab = inter[iu]
    union = s[iu[0]] + s[iu[1]] - i_ab
    smaller = np.minimum(s[iu[0]], s[iu[1]])
    return i_ab / union, i_ab / smaller


def summarise_pairs(J, OC):
    return dict(n_pairs=int(len(J)),
                jaccard_median=float(np.median(J)), jaccard_mean=float(J.mean()),
                oc_median=float(np.median(OC)), oc_mean=float(OC.mean()),
                frac_oc_ge_50=float((OC >= OC_OVERLAP).mean()),
                frac_j_ge_80=float((J >= J_NEAR_IDENT).mean()),
                frac_j_ge_25=float((J >= J_PRIOR_ART).mean()))


# ----------------------------------------------------------------- nulls ----
def runs(v):
    """Contiguous runs of 1s in a 0/1 vector, as a list of lengths."""
    d = np.diff(np.concatenate(([0], v.astype(np.int8), [0])))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()


def place_segments(lengths, L, rng):
    """Uniform random NON-OVERLAPPING placement of segments of these lengths on a
    sequence of length L (stars-and-bars: choose m of slack+m slots)."""
    m = len(lengths)
    slack = L - int(sum(lengths))
    v = np.zeros(L, np.int8)
    if m == 0 or slack < 0:
        return v
    cuts = np.sort(rng.choice(slack + m, size=m, replace=False))
    cum = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(int)
    for st, ln in zip(cuts - np.arange(m) + cum, lengths):
        v[st:st + ln] = 1
    return v


def null_segment_shift(M, rng):
    L = M.shape[1]
    segs = [runs(r) for r in M]
    return np.stack([place_segments(s, L, rng) for s in segs])


def null_union_restricted(M, rng):
    idx = np.flatnonzero(M.sum(0) > 0)
    out = np.zeros_like(M)
    for i, r in enumerate(M):
        k = min(int(r.sum()), len(idx))
        out[i, rng.choice(idx, size=k, replace=False)] = 1
    return out


def null_cross_antigen(M, L, donor_pool, rng):
    """Each epitope replaced by a different antigen's epitope, index-rescaled."""
    out = np.zeros_like(M)
    for i in range(M.shape[0]):
        d = donor_pool[rng.integers(len(donor_pool))]
        pos = np.flatnonzero(d)
        mapped = np.unique(np.floor(pos * (L / len(d))).astype(int))
        out[i, mapped[mapped < L]] = 1
    return out


# ------------------------------------------------------------ saturation ----
def saturation_curve(M, rng, n_orders=N_ORDERS, jmax=None):
    """Mean union size after j antibodies, averaged over random orderings."""
    n, L = M.shape
    jmax = n if jmax is None else min(jmax, n)
    acc = np.zeros(jmax)
    for _ in range(n_orders):
        order = rng.permutation(n)
        cum = np.zeros(L, bool)
        for j in range(jmax):
            cum |= M[order[j]].astype(bool)
            acc[j] += cum.sum()
    return acc / n_orders


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="antigen", choices=["antigen", "cluster95"],
                    help="antigen = exact antigen sequence (indices trivially "
                         "comparable); cluster95 = identity cluster split to "
                         "equal-length, >=95%% identical sub-groups")
    ap.add_argument("--keep_duplicate_antibodies", action="store_true",
                    help="keep repeat depositions of the SAME heavy chain on the "
                         "same antigen (off by default: they are PDB redundancy, "
                         "not antibody convergence)")
    ap.add_argument("--apply_fold_dedup", action="store_true",
                    help="also apply vh_dedup_keep. OFF by default -- that flag "
                         "makes FOLDS antibody-disjoint and is about antibodies "
                         "that bind two different antigens, which is irrelevant "
                         "here and only deletes real rows from a roster.")
    ap.add_argument("--reps", type=int, default=N_NULL_REPS)
    ap.add_argument("--orders", type=int, default=N_ORDERS)
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    man, labels = load(a.keep_duplicate_antibodies, a.apply_fold_dedup)
    groups = (groups_antigen if a.group == "antigen" else groups_cluster95)(man)
    print(f"  grouping={a.group}: {len(groups)} groups with >= 2 antibodies, "
          f"covering {sum(len(g) for g in groups.values())} rows")

    # (d) counts
    full = pd.read_csv(cfg.MANIFEST)
    full = full[full.usable]
    counts = {}
    for nm, d in (("full", full),):
        ga, gc = d.groupby("antigen_id").size(), d.groupby("Cluster").size()
        counts[nm] = dict(rows=int(len(d)),
                          antigens=int(ga.size), antigens_ge2=int((ga >= 2).sum()),
                          rows_in_antigens_ge2=int(ga[ga >= 2].sum()),
                          clusters=int(gc.size), clusters_ge2=int((gc >= 2).sum()),
                          rows_in_clusters_ge2=int(gc[gc >= 2].sum()))
    print("\n  (d) HOW MUCH OF THE DATA EVEN HAS THE QUESTION (usable rows)")
    for nm, c in counts.items():
        print(f"    {nm:9s}: {c['antigens_ge2']}/{c['antigens']} antigens carry >=2 "
              f"antibodies ({c['rows_in_antigens_ge2']}/{c['rows']} rows); "
              f"{c['clusters_ge2']}/{c['clusters']} clusters do "
              f"({c['rows_in_clusters_ge2']} rows)")

    # donor pool for N2: every usable epitope, keyed by the antigen it came from
    donor_by_ag = {}
    for ag, g in man.groupby("antigen_id"):
        donor_by_ag[ag] = [labels[c] for c in g.case_id]
    all_ags = list(donor_by_ag)

    rows, pooled = [], {k: [] for k in ("obs_J", "obs_OC", "n1_J", "n1_OC",
                                        "n2_J", "n2_OC", "n3_J", "n3_OC",
                                        "obs5_J", "obs5_OC")}
    sat_rows = []
    for gid, g in groups.items():
        M = np.stack([labels[c] for c in g.case_id]).astype(np.int8)
        n, L = M.shape
        J, OC = pair_stats(M)
        pooled["obs_J"].append(J)
        pooled["obs_OC"].append(OC)
        if M.shape[0] >= MIN_N_UNION_NULL:
            # N3 only exists for these groups, so the pooled N3 row has to be read
            # against THIS observed pool, not against the all-group one
            pooled["obs5_J"].append(J)
            pooled["obs5_OC"].append(OC)
        union = (M.sum(0) > 0)
        cnt = M.sum(0)                                  # antibodies per residue
        top_dec = max(1, int(np.ceil(0.10 * L)))
        rec = dict(group=gid, arm=ARM, grouping=a.group,
                   antigen_id=g.antigen_id.iloc[0],
                   antigen_name=str(g.antigen_name.iloc[0]),
                   cluster=int(g.Cluster.iloc[0]), n_antibodies=n, antigen_len=L,
                   epitope_size_median=float(np.median(M.sum(1))),
                   union_size=int(union.sum()),
                   union_frac=float(union.mean()),
                   hotspot_frac=float(cnt.max() / n),
                   top_decile_share=float(np.sort(cnt)[::-1][:top_dec].sum()
                                          / max(cnt.sum(), 1)))
        rec.update(summarise_pairs(J, OC))

        # --- nulls, same group, reps averaged -------------------------------
        # Replicates are capped so a group contributes at most ~MAX_NULL_PAIRS
        # null pairs: spike already yields 3.2e5 pairs per replicate, and 50 of
        # those would pin its null median to far more decimal places than anything
        # here is read to -- while costing 400 MB. Small antigens keep all --reps.
        own = set(g.antigen_id)
        donors = [e for ag in all_ags if ag not in own for e in donor_by_ag[ag]]
        n_pairs = max(len(J), 1)
        reps_eff = int(np.clip(np.ceil(MAX_NULL_PAIRS / n_pairs), 1, a.reps))
        nulls = {"n1": [], "n2": [], "n3": []}
        sat_null = np.zeros(n)
        hot_null, dec_null = 0.0, 0.0
        for rep in range(reps_eff):
            Mn = null_segment_shift(M, rng)
            nulls["n1"].append(pair_stats(Mn))
            sat_null += saturation_curve(Mn, rng, n_orders=1)
            cn = Mn.sum(0)
            hot_null += cn.max() / n
            dec_null += np.sort(cn)[::-1][:top_dec].sum() / max(cn.sum(), 1)
            if donors:
                nulls["n2"].append(pair_stats(
                    null_cross_antigen(M, L, donors, rng)))
            if n >= MIN_N_UNION_NULL:
                nulls["n3"].append(pair_stats(null_union_restricted(M, rng)))
        sat_null /= reps_eff
        rec["null_reps_used"] = reps_eff
        for key, vals in nulls.items():
            if not vals:
                rec[f"{key}_jaccard_median"] = np.nan
                rec[f"{key}_jaccard_mean"] = np.nan
                rec[f"{key}_oc_median"] = np.nan
                rec[f"{key}_frac_oc_ge_50"] = np.nan
                continue
            Jn = np.concatenate([v[0] for v in vals])
            On = np.concatenate([v[1] for v in vals])
            # the POOLED null distribution takes ONE replicate per group, so N1 and
            # N2 carry the same pair count and the same group composition as the
            # observed pool; N3 exists only for n >= MIN_N_UNION_NULL groups and so
            # is read against `observed_n_ge_5`, never against `observed`
            pooled[f"{key}_J"].append(vals[0][0])
            pooled[f"{key}_OC"].append(vals[0][1])
            rec[f"{key}_jaccard_median"] = float(np.median(Jn))
            rec[f"{key}_jaccard_mean"] = float(Jn.mean())
            rec[f"{key}_oc_median"] = float(np.median(On))
            rec[f"{key}_frac_oc_ge_50"] = float((On >= OC_OVERLAP).mean())
        rec["n1_union_frac"] = float(sat_null[-1] / L)
        rec["n1_hotspot_frac"] = float(hot_null / reps_eff)
        rec["n1_top_decile_share"] = float(dec_null / reps_eff)
        rows.append(rec)

        # --- saturation, observed, random orderings -------------------------
        sat = saturation_curve(M, rng, n_orders=a.orders)
        for j in range(n):
            sat_rows.append(dict(group=gid, n_antibodies=n, antigen_len=L, j=j + 1,
                                 union_obs=float(sat[j]),
                                 union_obs_frac_len=float(sat[j] / L),
                                 union_obs_frac_final=float(sat[j] / sat[-1]),
                                 union_null=float(sat_null[j]),
                                 union_null_frac_len=float(sat_null[j] / L)))

    per_ag = pd.DataFrame(rows).sort_values("n_antibodies", ascending=False)
    sat_df = pd.DataFrame(sat_rows)
    # fraction of that group's FINAL null union, taken over j within the group
    # (not within a single-j slice, which would make every value 1.0)
    sat_df["union_null_frac_final"] = (
        sat_df.union_null / sat_df.groupby("group").union_null.transform("max"))
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.group}_{ARM}"
    per_ag.to_csv(cfg.RESULTS / f"redundancy_per_antigen_{tag}.csv", index=False)

    # ---- pooled pair distributions, binned (figure-ready, small) ----------
    bins = np.round(np.arange(0, 1.0001, 0.05), 3)
    hist = {"bin_left": bins[:-1], "bin_right": bins[1:]}
    for k, v in pooled.items():
        hist[k] = (np.histogram(np.concatenate(v), bins=bins)[0]
                   if v else np.zeros(len(bins) - 1, int))
    pd.DataFrame(hist).to_csv(cfg.RESULTS / f"redundancy_pair_hist_{tag}.csv",
                              index=False)

    # ---- saturation summary: fixed composition, then all antigens ---------
    fixed = sat_df[sat_df.n_antibodies >= SATURATION_MIN_N]
    sat_sum = []
    for j in range(1, SATURATION_MIN_N + 1):
        s = fixed[fixed.j == j]
        allj = sat_df[sat_df.j == j]
        sat_sum.append(dict(
            j=j, n_groups_fixed=int(s.group.nunique()),
            frac_final_mean=float(s.union_obs_frac_final.mean()),
            frac_final_median=float(s.union_obs_frac_final.median()),
            frac_len_mean=float(s.union_obs_frac_len.mean()),
            frac_len_median=float(s.union_obs_frac_len.median()),
            null_frac_len_mean=float(s.union_null_frac_len.mean()),
            null_frac_final_mean=float(s.union_null_frac_final.mean()),
            n_groups_all=int(allj.group.nunique()),
            frac_final_mean_all=float(allj.union_obs_frac_final.mean())))
    sat_sum = pd.DataFrame(sat_sum)
    sat_sum.to_csv(cfg.RESULTS / f"redundancy_saturation_{tag}.csv", index=False)

    # ---- tests: per-antigen paired, ONE-SIDED (observed > null) -----------
    tests = {}
    for key, lab in (("n1", "segment-shift"), ("n2", "cross-antigen"),
                     ("n3", "union-restricted")):
        avail = per_ag.dropna(subset=[f"{key}_jaccard_median"])
        # Ties carry no sign, scipy drops them, and so does this: the medians below
        # are the medians of the set the TEST saw. Most ties are antigens where the
        # observed and the null pair medians are both exactly 0, so the tested
        # median is systematically higher than the median over all antigens -- both
        # are reported and they are not interchangeable.
        d = avail[avail.jaccard_median != avail[f"{key}_jaccard_median"]]
        if len(d) < 5:
            continue
        st, p = wilcoxon(d.jaccard_median, d[f"{key}_jaccard_median"],
                         alternative="greater")
        tests[key] = dict(null=lab, n_antigens_tested=int(len(d)),
                          n_antigens_available=int(len(avail)),
                          n_antigens_tied=int(len(avail) - len(d)),
                          obs_median_over_tested=float(d.jaccard_median.median()),
                          null_median_over_tested=float(
                              d[f"{key}_jaccard_median"].median()),
                          obs_median_over_available=float(
                              avail.jaccard_median.median()),
                          frac_antigens_above_null=float(
                              (d.jaccard_median > d[f"{key}_jaccard_median"]).mean()),
                          wilcoxon_stat=float(st), p_one_sided=float(p))

    def pool(k):
        return np.concatenate(pooled[k]) if pooled[k] else np.array([np.nan])

    head = dict(
        arm=ARM, grouping=a.group, seed=SEED, null_reps=a.reps,
        orderings=a.orders,
        thresholds=dict(oc_overlapping=OC_OVERLAP, j_near_identical=J_NEAR_IDENT,
                        j_prior_art=J_PRIOR_ART, identity_min=IDENTITY_MIN,
                        union_null_min_n=MIN_N_UNION_NULL,
                        saturation_fixed_min_n=SATURATION_MIN_N),
        filters=dict(within_antigen_vh_dedup=not a.keep_duplicate_antibodies,
                     fold_dedup_applied=a.apply_fold_dedup, usable_only=True),
        counts=counts,
        n_groups=len(groups), n_rows_in_groups=int(per_ag.n_antibodies.sum()),
        per_pair=dict(observed=summarise_pairs(pool("obs_J"), pool("obs_OC")),
                      observed_n_ge_5=summarise_pairs(pool("obs5_J"),
                                                      pool("obs5_OC")),
                      null_segment_shift=summarise_pairs(pool("n1_J"), pool("n1_OC")),
                      null_cross_antigen=summarise_pairs(pool("n2_J"), pool("n2_OC")),
                      null_union_restricted=summarise_pairs(pool("n3_J"),
                                                            pool("n3_OC"))),
        per_antigen=dict(
            jaccard_median_of_medians=float(per_ag.jaccard_median.median()),
            jaccard_mean_of_medians=float(per_ag.jaccard_median.mean()),
            oc_median_of_medians=float(per_ag.oc_median.median()),
            frac_oc_ge_50_median=float(per_ag.frac_oc_ge_50.median()),
            frac_oc_ge_50_mean=float(per_ag.frac_oc_ge_50.mean()),
            union_frac_median=float(per_ag.union_frac.median()),
            union_frac_mean=float(per_ag.union_frac.mean()),
            null_union_frac_median=float(per_ag.n1_union_frac.median()),
            hotspot_frac_median=float(per_ag.hotspot_frac.median()),
            null_hotspot_frac_median=float(per_ag.n1_hotspot_frac.median()),
            top_decile_share_median=float(per_ag.top_decile_share.median()),
            null_top_decile_share_median=float(per_ag.n1_top_decile_share.median())),
        tests=tests,
        saturation=sat_sum.to_dict(orient="records"))
    # the antigens that actually have a EVALUATED SET. A 2-antibody antigen cannot show
    # saturation or concentration; these can, and they are where the claim lives.
    big = per_ag[per_ag.n_antibodies >= SATURATION_MIN_N]
    head["per_antigen_n_ge_10"] = dict(
        n_antigens=int(len(big)), n_rows=int(big.n_antibodies.sum()),
        jaccard_median_of_medians=float(big.jaccard_median.median()),
        jaccard_mean_of_medians=float(big.jaccard_median.mean()),
        frac_oc_ge_50_median=float(big.frac_oc_ge_50.median()),
        frac_oc_ge_50_mean=float(big.frac_oc_ge_50.mean()),
        union_frac_median=float(big.union_frac.median()),
        union_frac_mean=float(big.union_frac.mean()),
        null_union_frac_median=float(big.n1_union_frac.median()),
        hotspot_frac_median=float(big.hotspot_frac.median()),
        null_hotspot_frac_median=float(big.n1_hotspot_frac.median()),
        top_decile_share_median=float(big.top_decile_share.median()),
        null_top_decile_share_median=float(big.n1_top_decile_share.median()))
    (cfg.RESULTS / f"redundancy_headline_{tag}.json").write_text(
        json.dumps(head, indent=2))

    # ------------------------------------------------------------- print ----
    p = head["per_pair"]
    print(f"\n  (a) PAIRWISE OVERLAP -- PER-PAIR view "
          f"({p['observed']['n_pairs']} within-antigen pairs; "
          f"spike dominates this pool)")
    for nm, k in (("observed", "observed"),
                  ("null N1 segment-shift", "null_segment_shift"),
                  ("null N2 cross-antigen", "null_cross_antigen"),
                  (f"observed, n>={MIN_N_UNION_NULL} only", "observed_n_ge_5"),
                  ("null N3 union-restricted", "null_union_restricted")):
        s = p[k]
        print(f"    {nm:26s} J median {s['jaccard_median']:.4f} mean "
              f"{s['jaccard_mean']:.4f} | OC median {s['oc_median']:.4f} mean "
              f"{s['oc_mean']:.4f} | OC>={OC_OVERLAP:.2f} {s['frac_oc_ge_50']:.3f} "
              f"| J>={J_NEAR_IDENT:.2f} {s['frac_j_ge_80']:.3f}")
    q = head["per_antigen"]
    print(f"\n  (a) PER-ANTIGEN view (each of {len(per_ag)} antigens = 1 point, "
          f"its own median)")
    print(f"    Jaccard  median-of-medians {q['jaccard_median_of_medians']:.4f}  "
          f"mean-of-medians {q['jaccard_mean_of_medians']:.4f}")
    print(f"    frac of pairs with OC >= {OC_OVERLAP:.2f}: median "
          f"{q['frac_oc_ge_50_median']:.3f}  mean {q['frac_oc_ge_50_mean']:.3f}")
    print(f"\n  (b) CONCENTRATION: union covers median "
          f"{q['union_frac_median']:.3f} of antigen residues "
          f"(null N1 {q['null_union_frac_median']:.3f}); hottest residue is hit by "
          f"median {q['hotspot_frac_median']:.3f} of an antigen's antibodies "
          f"(null {q['null_hotspot_frac_median']:.3f}); hottest 10% of residues "
          f"carry median {q['top_decile_share_median']:.3f} of all epitope "
          f"residue-instances (null {q['null_top_decile_share_median']:.3f})")
    print(f"\n  (b) SATURATION, fixed composition (antigens with >= "
          f"{SATURATION_MIN_N} antibodies, n={sat_sum.n_groups_fixed.iloc[0]}), "
          f"{a.orders} random orderings")
    print(sat_sum[["j", "n_groups_fixed", "frac_final_mean", "null_frac_final_mean",
                   "frac_len_mean", "null_frac_len_mean", "n_groups_all",
                   "frac_final_mean_all"]].round(4).to_string(index=False))
    b = head["per_antigen_n_ge_10"]
    print(f"\n  (b) ANTIGENS WITH A ROSTER (>= {SATURATION_MIN_N} antibodies, "
          f"n={b['n_antigens']} antigens / {b['n_rows']} rows): Jaccard "
          f"median-of-medians {b['jaccard_median_of_medians']:.4f}, "
          f"OC>={OC_OVERLAP:.2f} in median {b['frac_oc_ge_50_median']:.3f} of pairs; "
          f"union covers median {b['union_frac_median']:.3f} of the antigen vs "
          f"{b['null_union_frac_median']:.3f} under N1")
    print(f"\n  (c) TESTS -- per-antigen paired Wilcoxon, one-sided observed > null")
    for k, t in tests.items():
        print(f"    {t['null']:17s} tested n={t['n_antigens_tested']:4d} of "
              f"{t['n_antigens_available']:4d} ({t['n_antigens_tied']} tied) | obs "
              f"{t['obs_median_over_tested']:.4f} vs null "
              f"{t['null_median_over_tested']:.4f} | above-null "
              f"{t['frac_antigens_above_null']:.3f} | p={t['p_one_sided']:.3g}")
    print(f"\n  wrote {cfg.RESULTS}/redundancy_*_{tag}.{{csv,json}}")


if __name__ == "__main__":
    main()
