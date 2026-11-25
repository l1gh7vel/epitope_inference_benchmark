#!/usr/bin/env python3
"""
Figure 5B supplementary: whether the antibody embedding degeneracy is a property
of the data or of the one ESM-2 size the slide used, and what a median cosine of
0.99 means.

The claim under test
  Fig 5B is one histogram: pairwise cosine similarity between ESM-2 embeddings of
  SAbDab antibody sequences, pooled over all pairs within length buckets, labelled
  "ESM-2 150M (med 0.990)".  With Fig 5A (18.7% of rows are spike; the top 15
  antigens are 29.2% of rows) the thesis is data degeneracy: the benchmark is less
  diverse than its 4,585 rows suggest.
  A follow-up claim is that the cosine embeddings from different parameter-count
  ESM models are indistinguishable.  That is a robustness claim about 5B -- run the
  same histogram at several ESM-2 sizes and the picture (median ~0.99, everything
  alike) should hold.  Each model produces its own within-dataset cosine
  distribution, and those distributions are what is compared.  No vector from one
  model is compared with a vector from another, so cosine across different widths
  never arises.  Whether the models agree with each other, via RSA/CKA, is a
  different experiment and is not touched here.

The controls
  A median cosine of 0.99 among antibodies is on its own uninterpretable.  Two
  things can produce it, with opposite consequences for the paper:
    (a) Real biology.  Antibody variable domains are ~85% conserved framework and
        differ over three short CDR loops.  A mean-pooled embedding of 120 residues
        of which ~15 differ must look alike.
    (b) No spread in the embedding.  Mean-pooled ESM-2 cosines are high for almost
        any pair of proteins, because the mean token is dominated by a shared
        "protein-ness" direction.  Then 0.99 is a property of the readout rather
        than of SAbDab.
  Three controls separate the two, on the same statistic, bucketing and models:
    Antigen contrast.  Distinct SAbDab antigens, genuinely unrelated proteins.  If
      they also sit near 0.99, case (b) holds and the antibody number says nothing
      specific about antibodies.
    Shuffled floor.  Each antibody sequence permuted (same length, same amino-acid
      composition, order destroyed).  It separates "the model encodes sequence
      order" from "the model encodes composition", and is a floor for the readout
      rather than a null for the biology.
    Epitope split.  Among antibodies raised against the same antigen, do pairs that
      bind different epitopes embed less similarly than pairs that bind the same
      epitope?  If not, the representation does not encode what the antibody
      targets, which is the quantity an antibody-conditioned predictor has to read.

Item sets (all from data/manifests/task1_manifest.csv)
  antibody  : `usable` rows (the evaluated set Task I scores) with vh_kind ==
              "heavy" and a vh_seq over the standard 20 letters, deduplicated on
              exact vh_seq.  VH only, because the manifest carries no light chain
              -- Task I is a VH-only benchmark -- so this is the antibody
              representation the pipeline can build.  Exact repeat depositions
              would contribute cosine 1.0 pairs that are evidence about the PDB
              rather than about antibody diversity.  The non-deduplicated count is
              reported alongside as `antibody_rows`, so the slide's own convention
              is recoverable.
              vh_kind "both"/"unknown"/"light" usable rows are dropped because a
              VH+VL concatenate and a VH are not the same object and mean-pooling
              mixes them; the loss is reported in `provenance`.
  antigen   : distinct antigens (dedup on ag_key), standard 20 letters, length <=
              ANTIGEN_LEN_MAX (compute cap; the 3B model on a 14,507-residue chain
              is not affordable and such chains are a handful).
  shuffled  : one uniform permutation of each antibody sequence, seed SEED.
  Every eligible sequence is embedded.

Length bucketing
  Mean-pooled embedding cosine rises with sequence length -- a longer mean is
  closer to the corpus mean direction -- so pooling pairs across lengths confounds
  the statistic.  BUCKET = floor(len / BUCKET_W), fixed-width bins of
  BUCKET_W = 5 residues, and only within-bucket pairs are used.  The identical
  absolute rule is applied to antibodies, antigens and shuffles, so the three are
  comparable.  Sensitivity: bucket_w 1 (exact length match) and bucket_w 0 (no
  bucketing at all) are reported in the summary CSV as separate rows.

The epitope split, and the thresholds it inherits
  Machinery and thresholds come from scripts/t1_06_epitope_redundancy.py, so the
  two analyses cut the data the same way:
    * grouping is `antigen_id` (a hash of the exact antigen sequence), never
      `Cluster` -- gold labels are indexed into each row's own antigen sequence, so
      two epitopes may only be intersected index-by-index when the sequences match;
    * within-antigen repeat depositions of the same vh_seq are dropped;
    * overlap coefficient OC = |A n B| / min(|A|,|B|), with the same threshold
      OC >= 0.50 = "same epitope" ("the majority of the smaller epitope is
      shared");
    * "different epitope" is the strict complement at the other end: |A n B| == 0,
      the two epitopes disjoint.  An OC < 0.50 group would include half-overlapping
      pairs and blur the contrast the test is for.  The intermediate pairs are
      counted and reported.
  Pairs are additionally restricted to the same length bucket, for the reason
  above.
  Two tests, because pair-level pooling is pseudo-replication (96% of
  within-antigen pairs in SAbDab are spike -- see t1_06):
    primary  per-antigen paired one-sided Wilcoxon signed-rank over antigens that
             have >= 1 pair in each group: median cosine(same-epitope) >
             median cosine(different-epitope).  One antigen, one number, so spike
             cannot carry it.
    pooled   one-sided Mann-Whitney U over all qualifying pairs, reported with the
             pseudo-replication warning and repeated with the single largest
             antigen removed.
  Tests are one-sided throughout, because the hypothesis has a direction: if the
  embedding encodes the target, same-epitope pairs should be more similar.  Also
  reported, because dichotomising throws information away: the Spearman
  correlation between a pair's OC and its embedding cosine, pooled and per-antigen.

Embedding call
  From scripts/08_walle_infer.py::native_esm2, which is what the pipeline runs for
  WALLE's `pre_cal` node features: fair-esm, the model's final representation
  layer, BOS/EOS stripped, fp32, one sequence per forward pass (no padding, so no
  batching artefact).  Per model that is layer 6 / 12 / 30 / 33 / 36 -- the
  pipeline's 35M layer-12 choice generalised as "last layer".  Any other layer
  would be a different experiment.
  Pooling is the mean over residues, then cosine, which is what Fig 5B does (a
  per-sequence vector is needed and the slide's x range matches mean-pooling).
  CLS-token pooling would give different absolute numbers.  The controls are
  pooled identically, so the contrast is unaffected.

Models: ESM-2 8M / 35M / 150M / 650M / 3B.  35M is what the pipeline uses; 150M is
  what the slide shows.  Weights are reused from work/esm_scale/torch_home.

Reported per model per item set: n items, n pairs, median, mean, q25, q75, IQR, and
  the fraction of pairs above 0.95 and above 0.99.  Median and mean are reported
  separately and never reconciled.

Seed 0 for the shuffle; nothing else is random.

    $EIB_PYTHON_GPU scripts/esm_ab_degeneracy.py --device cuda

Writes results/esm_degeneracy/.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
MANIFESTS = PROJECT / "data" / "manifests"
MANIFEST = MANIFESTS / "task1_manifest.csv"
LABELS_NPZ = PROJECT / "data" / "manifests" / "task1_labels.npz"
OUT = PROJECT / "results" / "esm_degeneracy"
TORCH_HOME = PROJECT / "work" / "esm_scale" / "torch_home"

HIST_BINS = np.linspace(0.0, 1.0, 201)   # 0.005-wide bins, shared x axis
SEED = 0
BUCKET_W = 5                 # length-bucket width, residues (primary)
BUCKET_SENS = (1, 0)         # sensitivity widths; 0 = no bucketing at all
ANTIGEN_LEN_MAX = 1000       # compute cap on the antigen control
OC_SAME = 0.50               # "same epitope", inherited from t1_06
FRAC_THRESH = (0.95, 0.99)   # the two tail fractions reported
AA20 = "ACDEFGHIKLMNPQRSTVWY"

# fair-esm name -> (loader attr, final repr layer, nominal width)
MODELS = [
    ("esm2_8M",   "esm2_t6_8M_UR50D",    6,   320),
    ("esm2_35M",  "esm2_t12_35M_UR50D",  12,  480),   # the pipeline's model
    ("esm2_150M", "esm2_t30_150M_UR50D", 30,  640),   # the slide's model
    ("esm2_650M", "esm2_t33_650M_UR50D", 33,  1280),
    ("esm2_3B",   "esm2_t36_3B_UR50D",   36,  2560),
]


# --------------------------------------------------------------------------- #
#  item sets                                                                    #
# --------------------------------------------------------------------------- #
def build_items():
    man = pd.read_csv(MANIFEST)
    n_all = len(man)
    man = man[man.usable].copy()
    n_usable = len(man)

    ab = man[man.vh_kind == "heavy"].copy()
    n_heavy = len(ab)
    ab = ab[ab.vh_seq.map(lambda s: set(str(s)) <= set(AA20))].copy()
    ab_rows = ab                                  # row-level (slide's convention)
    ab_uniq = (ab.sort_values("case_id")
                 .drop_duplicates("vh_seq")[["vh_seq", "vh_len"]]
                 .reset_index(drop=True))

    # One antigen per sequence-identity cluster, not per exact sequence. Dedup on
    # `ag_key` removes only exact duplicates, so the 20 deposited spike constructs
    # and the 9 gp120 constructs survive as separate "unrelated proteins", which
    # puts 17.5% of the control's pairs above cosine 0.99 with a spike at 1.0. A
    # control whose job is to say what unrelated proteins look like cannot contain
    # near-identical variants of one antigen. `Cluster` is the CD-HIT 90%-identity
    # clustering that defines the antigen groups, so this introduces no new
    # threshold. `antigen` counts one sequence per cluster; `antigen_exact` counts
    # every distinct exact sequence. Both are reported, so their difference
    # measures the size of the near-duplicate effect.
    ag_exact = man.drop_duplicates("ag_key").copy()
    ag = man.drop_duplicates("Cluster").copy()
    def _clean_ag(df):
        df = df[df.antigen_sequence.map(lambda s: set(str(s)) <= set(AA20))]
        n_std = len(df)
        df = df[df.antigen_len <= ANTIGEN_LEN_MAX]
        return (df[["antigen_sequence", "antigen_len"]]
                .rename(columns={"antigen_sequence": "seq", "antigen_len": "len"})
                .reset_index(drop=True), n_std)
    ag_uniq, n_ag_std = _clean_ag(ag)
    ag_exact_uniq, n_ag_exact_std = _clean_ag(ag_exact)

    rng = np.random.default_rng(SEED)
    shuf = []
    for s in ab_uniq.vh_seq:
        a = np.frombuffer(s.encode(), dtype=np.uint8).copy()
        rng.shuffle(a)
        shuf.append(a.tobytes().decode())
    shuf_uniq = pd.DataFrame({"seq": shuf, "len": ab_uniq.vh_len.values})

    prov = dict(
        manifest_rows=int(n_all), usable_rows=int(n_usable),
        usable_vh_kind_heavy=int(n_heavy),
        heavy_rows_std20=int(len(ab_rows)),
        heavy_rows_dropped_nonstd=int(n_heavy - len(ab_rows)),
        usable_rows_dropped_not_heavy=int(n_usable - n_heavy),
        antibody_unique_vh=int(len(ab_uniq)),
        antigen_unique_std20=int(n_ag_std),
        antigen_exact_unique_std20=int(n_ag_exact_std),
        antigen_exact_kept=int(len(ag_exact_uniq)),
        antigen_kept_le_len_max=int(len(ag_uniq)),
        antigen_dropped_too_long=int(n_ag_std - len(ag_uniq)),
        antigen_len_max=ANTIGEN_LEN_MAX,
    )
    sets = {
        "antibody":      ab_uniq.rename(columns={"vh_seq": "seq", "vh_len": "len"}),
        "antibody_rows": ab_rows[["vh_seq", "vh_len"]].rename(
                             columns={"vh_seq": "seq", "vh_len": "len"}).reset_index(drop=True),
        "antigen":       ag_uniq,
        "antigen_exact": ag_exact_uniq,
        "shuffled_ab":   shuf_uniq,
    }
    return sets, ab_rows, prov


# --------------------------------------------------------------------------- #
#  CDR positions                                                                #
# --------------------------------------------------------------------------- #
def load_cdr_idx(scheme="imgt"):
    """{vh_seq -> 0-based indices of its CDR residues} from data/manifests.

    The scheme is IMGT. Both IMGT and Chothia spans are emitted by
    scripts/28_cdr_spans.py; they differ by a near-constant +5 residues (mean
    4.99, sd 0.31, median 5, only 11 of 4,371 sequences differ by more than 5),
    so the choice barely moves the window. It changes which sequences survive:
    13 carry an ultralong CDR-H3 of 50-63 residues (bovine-type knob-domain
    antibodies, cysteine-rich, several raised against HIV Env), and ANARCI
    returns an empty numbering for all 13 of them under Chothia. Chothia would
    drop the antibodies whose CDR-H3 dominates binding most, the cohort most
    favourable to the hypothesis under test. IMGT keeps them.
    """
    sp = pd.read_csv(MANIFESTS / "cdr_spans.csv")
    sq = pd.read_csv(MANIFESTS / "cdr_spans_seqs.csv")
    key2seq = dict(zip(sq.vh_key, sq.vh_seq))
    sp = sp[(sp.scheme == scheme) & (sp.ok == 1)]
    idx, tot = {}, {}
    for r in sp.itertuples():
        s = key2seq.get(r.vh_key)
        if s is None:
            continue
        a = np.concatenate([np.arange(int(r.cdr1_start), int(r.cdr1_end)),
                            np.arange(int(r.cdr2_start), int(r.cdr2_end)),
                            np.arange(int(r.cdr3_start), int(r.cdr3_end))])
        assert a.max() < len(s), f"span past end of {r.vh_key}"
        idx[s] = a
        tot[s] = int(r.cdr_total_len)
    return idx, tot


# --------------------------------------------------------------------------- #
#  embeddings -- scripts/08_walle_infer.py::native_esm2, final layer, mean-pooled
# --------------------------------------------------------------------------- #
def embed(attr, layer, seqs, device, tag="", cdr_idx=None):
    """Mean-pooled final-layer ESM-2 vector per sequence.

    cdr_idx: optional {sequence -> np.array of 0-based residue indices}. When a
    sequence appears there, a second vector is produced from the same forward
    pass, meaning over those positions only. Returned as a separate dict.

    The CDR positions are pooled from the full VH rather than from a stitched
    CDR string. Stitching changes the model input as well as the readout: ESM
    would be asked to encode a ~31 residue chimera that is not a protein, with no
    framework context, and a difference would then be attributable to either
    change. Pooling over CDR positions of the full-length VH changes only the
    readout, so a difference is attributable to pooling alone. The stitched
    variant is a robustness check and is not what this computes.
    """
    import esm
    model, alph = getattr(esm.pretrained, attr)()
    model.eval().to(device)
    bc = alph.get_batch_converter()
    out, out_cdr = {}, {}
    t0 = time.time()
    for i, s in enumerate(seqs):
        _, _, toks = bc([("x", s)])
        with torch.no_grad():
            rep = model(toks.to(device), repr_layers=[layer])["representations"][layer]
        # Strip BOS/EOS: residue j of s is token j+1.
        res = rep[0, 1:len(s) + 1].float()
        out[s] = res.mean(0).cpu().numpy().astype(np.float32)
        if cdr_idx is not None and s in cdr_idx:
            idx = cdr_idx[s]
            out_cdr[s] = res[idx].mean(0).cpu().numpy().astype(np.float32)
        if (i + 1) % 1000 == 0:
            print(f"    [{tag}] {i+1}/{len(seqs)}  {time.time()-t0:.0f}s", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out, out_cdr


# --------------------------------------------------------------------------- #
#  pairwise cosine within length buckets                                        #
# --------------------------------------------------------------------------- #
def standardise(V, mu, sd):
    """Per-dimension standardisation, the published remedy for the cone effect.

    Transformer embeddings are anisotropic: all vectors share a large common
    direction, so every pair of proteins, related or not, has a cosine inflated
    toward 1. A raw median of 0.98 among antibodies cannot be read as "nearly
    identical" without knowing how much of it is that shared direction. The fix
    in the literature removes it: for each of the d dimensions, subtract that
    dimension's mean and divide by its s.d., both computed across the corpus,
    and only then take the cosine.

    The corpus is shared. mu and sd are computed once over every sequence
    embedded for a given model -- antibodies, antigens and shuffles together --
    rather than per item set. Standardising each set by its own statistics would
    remove a different direction from each and make the three distributions
    incomparable, which is what this panel needs them to be. Reported as a
    separate row (standardised=1), never replacing the raw numbers, which are
    what the original slide showed and what a reader of the literature expects.
    """
    return (V - mu) / np.clip(sd, 1e-6, None)


def bucket_cosines(V, lengths, bucket_w):
    """V: (n,d) float32.  Returns the pooled within-bucket pair cosines."""
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    keys = np.zeros(len(V), dtype=np.int64) if bucket_w == 0 else (np.asarray(lengths) // bucket_w)
    chunks = []
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        if len(idx) < 2:
            continue
        G = V[idx] @ V[idx].T
        iu = np.triu_indices(len(idx), k=1)
        chunks.append(G[iu].astype(np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def summarize(c):
    if len(c) == 0:
        return dict(n_pairs=0)
    q25, med, q75 = np.percentile(c, [25, 50, 75])
    d = dict(n_pairs=int(len(c)), median=float(med), mean=float(c.mean()),
             q25=float(q25), q75=float(q75), iqr=float(q75 - q25),
             p01=float(np.percentile(c, 1)), p99=float(np.percentile(c, 99)))
    for t in FRAC_THRESH:
        d[f"frac_gt_{t}"] = float((c > t).mean())
    return d


# --------------------------------------------------------------------------- #
#  epitope split                                                                #
# --------------------------------------------------------------------------- #
def epitope_pairs(ab_rows, bucket_w):
    """Within-antigen antibody pairs with their epitope overlap and bucket key.

    Grouping, dedup and the OC >= 0.50 threshold are t1_06_epitope_redundancy.py's.
    """
    z = np.load(LABELS_NPZ, allow_pickle=True)
    rows = (ab_rows.sort_values(["antigen_id", "vh_seq", "n_epitope", "case_id"],
                                ascending=[True, True, False, True])
                   .drop_duplicates(["antigen_id", "vh_seq"]))
    recs = []
    for ag, g in rows.groupby("antigen_id"):
        if len(g) < 2:
            continue
        cases = list(g.case_id)
        seqs = list(g.vh_seq)
        lens = list(g.vh_len)
        sets_ = [set(np.flatnonzero(np.asarray(z[c], dtype=np.int8)).tolist()) for c in cases]
        for i in range(len(cases)):
            for j in range(i + 1, len(cases)):
                a, b = sets_[i], sets_[j]
                if not a or not b:
                    continue
                inter = len(a & b)
                oc = inter / min(len(a), len(b))
                jac = inter / len(a | b)
                ki = 0 if bucket_w == 0 else lens[i] // bucket_w
                kj = 0 if bucket_w == 0 else lens[j] // bucket_w
                recs.append((ag, seqs[i], seqs[j], inter, oc, jac, ki == kj))
    return pd.DataFrame(recs, columns=["antigen_id", "seq_i", "seq_j",
                                       "inter", "oc", "jaccard", "same_bucket"])


def epitope_test(pairs, emb, model_name, scope, bucket_w):
    """Cosine of same-epitope (OC >= OC_SAME) vs different-epitope (disjoint) pairs."""
    p = pairs if scope == "all_length_pairs" else pairs[pairs.same_bucket]
    if len(p) == 0:
        return None
    cache = {}

    def vec(s):
        if s not in cache:
            v = emb[s]
            cache[s] = v / max(float(np.linalg.norm(v)), 1e-12)
        return cache[s]

    cos = np.empty(len(p), np.float32)
    for k, (si, sj) in enumerate(zip(p.seq_i.values, p.seq_j.values)):
        cos[k] = float(vec(si) @ vec(sj))
    p = p.assign(cos=cos)

    same = p[p.oc >= OC_SAME]
    diff = p[p.inter == 0]
    mid = p[(p.oc < OC_SAME) & (p.inter > 0)]

    # Primary: per-antigen paired, one-sided (same > diff)
    per_ag = []
    for ag, g in p.groupby("antigen_id"):
        s = g[g.oc >= OC_SAME].cos
        d = g[g.inter == 0].cos
        if len(s) and len(d):
            per_ag.append((ag, float(s.median()), float(d.median())))
    w_p, w_stat, n_ag = np.nan, np.nan, len(per_ag)
    if n_ag >= 5:
        a = np.array([r[1] for r in per_ag])
        b = np.array([r[2] for r in per_ag])
        if np.any(a != b):
            st, pv = wilcoxon(a, b, alternative="greater")
            w_stat, w_p = float(st), float(pv)
    n_ag_same_higher = int(sum(1 for r in per_ag if r[1] > r[2]))

    # Effect size. The raw median gap is unreadable against a cosine distribution
    # whose whole IQR is ~0.01, so report the rank statistic the U test is built
    # from: AUC = P(cosine of a same-epitope pair > cosine of a different-epitope
    # pair), ties counted half.  0.5 is "the embedding cannot tell them apart",
    # 1.0 is perfect separation.  Also per-antigen, so spike cannot carry it.
    def auc(a, b):
        if not len(a) or not len(b):
            return np.nan
        u = mannwhitneyu(a, b, alternative="greater").statistic
        return float(u / (len(a) * len(b)))

    auc_pooled = auc(same.cos.values, diff.cos.values)
    auc_ag = [auc(g[g.oc >= OC_SAME].cos.values, g[g.inter == 0].cos.values)
              for _, g in p.groupby("antigen_id")]
    auc_ag = [a for a in auc_ag if not np.isnan(a)]

    # Pooled (pseudo-replicated), and repeated without the biggest antigen
    u_p = np.nan
    if len(same) and len(diff):
        u_p = float(mannwhitneyu(same.cos, diff.cos, alternative="greater").pvalue)
    big = p.antigen_id.value_counts().idxmax()
    q = p[p.antigen_id != big]
    u_p_nobig, med_s_nb, med_d_nb = np.nan, np.nan, np.nan
    s2, d2 = q[q.oc >= OC_SAME], q[q.inter == 0]
    if len(s2) and len(d2):
        u_p_nobig = float(mannwhitneyu(s2.cos, d2.cos, alternative="greater").pvalue)
        med_s_nb, med_d_nb = float(s2.cos.median()), float(d2.cos.median())

    rho, rho_p = spearmanr(p.oc.values, p.cos.values)
    rho_ag = [float(spearmanr(g.oc.values, g.cos.values)[0])
              for _, g in p.groupby("antigen_id")
              if len(g) >= 10 and g.oc.nunique() > 1]

    return dict(
        model=model_name, scope=scope, bucket_w=bucket_w,
        n_pairs=int(len(p)), n_antigens=int(p.antigen_id.nunique()),
        n_same=int(len(same)), n_diff=int(len(diff)), n_intermediate=int(len(mid)),
        same_median=float(same.cos.median()) if len(same) else np.nan,
        same_mean=float(same.cos.mean()) if len(same) else np.nan,
        diff_median=float(diff.cos.median()) if len(diff) else np.nan,
        diff_mean=float(diff.cos.mean()) if len(diff) else np.nan,
        delta_median=float(same.cos.median() - diff.cos.median()) if len(same) and len(diff) else np.nan,
        delta_mean=float(same.cos.mean() - diff.cos.mean()) if len(same) and len(diff) else np.nan,
        auc_same_over_diff_pooled=auc_pooled,
        auc_same_over_diff_per_antigen_median=float(np.median(auc_ag)) if auc_ag else np.nan,
        n_antigens_auc=len(auc_ag),
        n_antigens_paired=int(n_ag), n_antigens_same_higher=n_ag_same_higher,
        wilcoxon_stat=w_stat, wilcoxon_p_one_sided=w_p,
        mwu_p_one_sided_pooled=u_p,
        biggest_antigen=str(big), biggest_antigen_pairs=int((p.antigen_id == big).sum()),
        mwu_p_one_sided_nobiggest=u_p_nobig,
        same_median_nobiggest=med_s_nb, diff_median_nobiggest=med_d_nb,
        spearman_oc_vs_cos=float(rho), spearman_p=float(rho_p),
        spearman_per_antigen_median=float(np.median(rho_ag)) if rho_ag else np.nan,
        n_antigens_spearman=len(rho_ag),
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default="all")
    ap.add_argument("--bucket_w", type=int, default=BUCKET_W)
    ap.add_argument("--cdr_scheme", default="imgt", choices=("imgt", "chothia"),
                    help="CDR definition for the CDR-only pooling arm. IMGT by "
                         "default; see load_cdr_idx() for why, and note Chothia "
                         "drops all 13 ultralong-CDR-H3 antibodies.")
    ap.add_argument("--debug_n", type=int, default=0,
                    help="DEBUG ONLY: truncate every item set to N sequences so the "
                         "wiring can be smoke-tested on CPU. Never use for results.")
    args = ap.parse_args()

    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    torch.hub.set_dir(str(TORCH_HOME / "hub"))
    # A debug run writes to its own directory. `--debug_n` truncates every item
    # set, so its summary_by_model.csv, headline.json and histograms must not
    # land on top of a full run's.
    global OUT
    if args.debug_n:
        OUT = OUT.parent / (OUT.name + "_DEBUG")
        print(f"[DEBUG] writing to {OUT} -- real results are not touched", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sets, ab_rows, prov = build_items()
    print("[items] " + json.dumps(prov), flush=True)
    if args.debug_n:
        print(f"[DEBUG] truncating every item set to {args.debug_n}", flush=True)
        sets = {k: v.head(args.debug_n).reset_index(drop=True) for k, v in sets.items()}
        keep = set(sets["antibody"].seq) | set(sets["antibody_rows"].seq)
        ab_rows = ab_rows[ab_rows.vh_seq.isin(keep)]

    pairs = epitope_pairs(ab_rows, args.bucket_w)
    print(f"[epitope] {len(pairs)} within-antigen pairs over "
          f"{pairs.antigen_id.nunique()} antigens; "
          f"{int(pairs.same_bucket.sum())} in the same length bucket", flush=True)

    # CDR-only pooling, shown against the whole-VH embedding. Same sequences, same
    # forward pass, different pooling window; embed()'s docstring covers why the
    # input is the full VH rather than a stitched CDR string.
    cdr_idx, cdr_tot = load_cdr_idx(args.cdr_scheme)
    ab = sets["antibody"]
    have = ab.seq.isin(cdr_idx)
    sets["antibody_cdr"] = pd.DataFrame({
        "seq": ab.seq[have].values,
        # Bucket on CDR length rather than VH length. The length confound this
        # bucketing controls is a property of how many vectors the mean averages,
        # and for this set that is the CDR window rather than the whole domain.
        "len": [cdr_tot[s] for s in ab.seq[have]]})
    print(f"[cdr] scheme={args.cdr_scheme}: {int(have.sum())} of {len(ab)} unique "
          f"VH have spans ({len(ab) - int(have.sum())} unnumbered, mostly "
          f"multi-domain scFv constructs); CDR window median "
          f"{int(np.median(sets['antibody_cdr']['len']))} residues", flush=True)

    # every unique sequence that must be embedded, once per model
    all_seqs = sorted({s for k in ("antibody", "antigen", "antigen_exact",
                                   "shuffled_ab") for s in sets[k].seq})
    print(f"[embed] {len(all_seqs)} unique sequences per model", flush=True)

    models = MODELS if args.models == "all" else [m for m in MODELS if m[0] in args.models.split(",")]
    rows, ep_rows, hist_rows = [], [], []
    for name, attr, layer, width in models:
        t0 = time.time()
        emb, emb_cdr = embed(attr, layer, all_seqs, args.device, tag=name,
                             cdr_idx=cdr_idx)
        print(f"[esm] {name} -> {width}d ({time.time()-t0:.0f}s)", flush=True)

        # Corpus statistics for standardisation: once per model, over every
        # sequence this model embedded. CDR vectors get their own, because they
        # are a mean over a different number of positions and so sit in a
        # different place in the space.
        _all = np.stack([emb[k] for k in emb])
        MU, SD = _all.mean(0), _all.std(0)
        if emb_cdr:
            _allc = np.stack([emb_cdr[k] for k in emb_cdr])
            MUC, SDC = _allc.mean(0), _allc.std(0)
        else:
            MUC, SDC = MU, SD

        for set_name in ("antibody", "antibody_cdr", "antibody_rows", "antigen",
                         "antigen_exact", "shuffled_ab"):
            df = sets[set_name]
            src = emb_cdr if set_name == "antibody_cdr" else emb
            V0 = np.stack([src[s] for s in df.seq])
            mu, sd = (MUC, SDC) if set_name == "antibody_cdr" else (MU, SD)
            for std_flag, V in ((0, V0), (1, standardise(V0, mu, sd))):
              for bw in (args.bucket_w,) + BUCKET_SENS:
                c = bucket_cosines(V, df["len"].values, bw)
                rows.append(dict(model=name, width=width, layer=layer,
                                 item_set=set_name, n_items=len(df), bucket_w=bw,
                                 standardised=std_flag, **summarize(c)))
                # Histogram, at the primary bucket width only. The pairs are kept
                # rather than reduced to five summary statistics because the
                # panel needs the shape: the CDR-only arm moves the median by
                # 0.002 while frac > 0.99 falls from 0.107 to 0.010. Fixed bins
                # over [0, 1] so every model and item set share an x axis
                # exactly.
                if bw == args.bucket_w and len(c):
                    # Standardised cosines span [-1, 1], raw ones sit near 1, so
                    # the two cannot share HIST_BINS. Each gets its own edges and
                    # the row records which.
                    edges = HIST_BINS if not std_flag else np.linspace(-1.0, 1.0,
                                                                       len(HIST_BINS))
                    h, _ = np.histogram(c, bins=edges)
                    hist_rows.append(dict(model=name, item_set=set_name,
                                          standardised=std_flag,
                                          lo=float(edges[0]), hi=float(edges[-1]),
                                          n_pairs=int(len(c)),
                                          **{f"b{i}": int(v) for i, v in enumerate(h)}))
        for scope in ("same_length_bucket", "all_length_pairs"):
            r = epitope_test(pairs, emb, name, scope, args.bucket_w)
            if r:
                ep_rows.append(r)
        print(f"  [{name}] done ({time.time()-t0:.0f}s)", flush=True)

        # write incrementally so a late failure does not lose finished models
        pd.DataFrame(rows).to_csv(OUT / "summary_by_model.csv", index=False)
        pd.DataFrame(ep_rows).to_csv(OUT / "epitope_split_by_model.csv", index=False)
        pd.DataFrame(hist_rows).to_csv(OUT / "cosine_hist_by_model.csv", index=False)

    sm = pd.DataFrame(rows)
    ep = pd.DataFrame(ep_rows)
    # The headline block reports the raw numbers only; the standardised rows are
    # a separate arm and live in the CSV. Without this filter every lookup below
    # returns two rows and collapses to a Series.
    prim = sm[(sm.bucket_w == args.bucket_w) & (sm.standardised == 0)]
    cols = ["n_items", "n_pairs", "median", "mean", "iqr",
            "frac_gt_0.95", "frac_gt_0.99"]
    headline = dict(
        question="is Fig 5B's antibody-embedding degeneracy stable across ESM-2 "
                 "model scale, and do the controls make 0.99 interpretable",
        seed=SEED, bucket_w=args.bucket_w,
        pooling="mean over residues of the final representation layer",
        oc_same_threshold=OC_SAME, provenance=prov, per_model={})
    for name, _, _, _ in models:
        g = prim[prim.model == name].set_index("item_set")
        headline["per_model"][name] = {
            s: {k: g.loc[s, k] for k in cols if k in g.columns} for s in g.index}
    headline["epitope_split"] = ep[ep.scope == "same_length_bucket"].to_dict("records")
    (OUT / "headline.json").write_text(
        json.dumps(headline, indent=2, default=lambda o: float(o)))
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
