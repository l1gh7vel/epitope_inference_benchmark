#!/usr/bin/env python3
"""
E8, second model family: whether the antibody embedding degeneracy survives a
change of protein language model, or is a fact about ESM-2.

What this measures, and why it imports the other script
  scripts/esm_ab_degeneracy.py answers the question for ESM-2 at five sizes.  Its
  result, on the merged item sets, is a median pairwise cosine of mean-pooled
  final-layer representations within equal-length bins:

      item set                raw      per-dimension standardised
      antibodies (whole VH)   0.978    0.774
      antibodies, CDR only    0.976    0.068
      shuffled control        0.978    0.495
      antigens (control)      0.894    0.319

  Two claims follow, both about a readout rather than about ESM-2:
    (i)  raw cosine carries no information -- the shuffled control sits 0.0008
         from the real antibodies, so 0.978 is the anisotropy of mean-pooled
         transformer features rather than a fact about antibodies;
    (ii) after per-dimension standardisation the series separate, and CDR-only
         pooling collapses below the shuffled control, so the CDRs do distinguish
         antibodies and the degeneracy comes from mean-pooling ~125 positions of
         which ~31 differ.
  A claim about a readout becomes a claim about representations once it holds for
  a second, architecturally different model.  That is this script.

  Everything that defines the measurement -- the item sets, the length bucketing,
  the cosine, the standardisation, the summary statistics, the epitope split and
  its thresholds -- is imported from esm_ab_degeneracy.py.  The value of a second
  model is that it is the same measurement, and a copy that later drifts would
  destroy the comparison.  That file's module docstring carries the construction;
  it is not repeated here.

  The import needs a three-line shim.  esm_ab_degeneracy.py does `import torch` at
  module scope, and torch is absent from the ProteinBERT venv.  torch is used only
  inside its embed(), which this script never calls, so a stub module is injected
  before the file is exec'd.  The explicit assertion on the functions taken makes
  the script fail loudly if that file ever uses torch outside embed().

Which representation is pooled
  ProteinBERT is not a stack of transformer blocks.  Each of its 6 blocks carries
  two parallel tracks (Brandes et al. 2022):
    * a local track, (sequence_length, 128), updated by two dilated 1-D
      convolutions plus a broadcast from the global track;
    * a global track, a single 512-d vector per sequence, updated by dense layers
      plus a global-attention head that queries every sequence position.
  So "the final representation layer" is ambiguous in a way it is not for ESM-2,
  and the choice is stated here rather than defaulted into.

  Primary is the local track of the last block, the output of layer
  `seq-merge2-norm-block6`, with the <START>/<END> tokens stripped and a mean
  taken over residue positions.  That is the structural analogue of what the
  ESM-2 run pools: the final per-residue hidden state of the encoder, taken
  before the language-model head that projects to vocabulary logits.  It is also
  the only track on which the CDR-only arm is definable, because CDR-only pooling
  is a choice of which residue positions to average and the global track has no
  residue axis.

  Secondary is the global track of the last block, `global-merge2-norm-block6`,
  512-d, one vector per sequence with no pooling.  It is reported as a separate
  `model` row (proteinbert_global) because it is ProteinBERT's own answer to
  "give me one vector for this protein", and a reader will ask whether the
  degeneracy is an artefact of mean-pooling specifically.  One caveat travels with
  every number from it: the model's annotation input is a zero vector here
  (proteinbert.InputEncoder.encode_X zeroes it, the standard inference path for an
  unannotated sequence), so the global track starts identical for every sequence
  and differentiates only through global attention over the local track.  It is
  not an independent view of the protein.  The CDR-only arm is omitted for it.

No padding, and the loop structure that follows
  ProteinBERT is built for a fixed input length; <PAD> tokens are real tokens that
  the convolutions convolve over and that the global-attention softmax normalises
  across.  A padded batch is a different computation from an unpadded one, and the
  padding amount would be a free parameter of the result.  The ESM-2 run avoids
  the equivalent problem by running one sequence per forward pass.  Here every op
  in the network (Conv1D, Dense, LayerNormalization, GlobalAttention) is
  independent across the batch axis, so sequences of identical length batch with
  bit-identical results to running them one at a time.  Sequences are grouped by
  exact length and one model is instantiated per distinct length (649 of them over
  the 8,174 sequences).  Model instantiation, not inference, is the cost; the
  whole run is ~35 min on CPU and needs no GPU.

How the environment was built (work/proteinbert/ is gitignored, so this is the
record).  ProteinBERT is TensorFlow/Keras, which nothing else in this repo is, so
it has its own venv rather than anything being added to the analysis or walle
envs:

    BASE=/software/rhel9/manual/install/python/tensorflow_218_311/bin/python3  # 3.11
    $BASE -m venv work/proteinbert/venv
    work/proteinbert/venv/bin/pip install 'tensorflow==2.15.1' 'numpy<2' pandas scipy h5py
    git clone https://github.com/nadavbra/protein_bert.git work/proteinbert/src
    git clone https://github.com/nadavbra/shared_utils.git   # the empty submodule
    cp -r shared_utils/* work/proteinbert/src/proteinbert/shared_utils/
    work/proteinbert/venv/bin/pip install --no-deps work/proteinbert/src

  Three constraints in that recipe:
    * TensorFlow is pinned to 2.15.1 because 2.16 switched tf.keras to Keras 3,
      under which ProteinBERT's GlobalAttention layer does not build (it uses
      K.int_shape and a list-of-shapes build signature).
    * The code is github master (commit 69a1122) rather than the PyPI
      `protein-bert` 1.0.1 sdist. 1.0.1 calls `optimizer(lr=...)`, removed in this
      Keras, and its model dump URL is a dead FTP host.
    * `--no-deps`, because setup.py requires tensorflow-addons, which is archived
      and pins an older TF. Nothing in the package imports it.

Weights
  The pretrained dump distributed by the authors,
  epoch_92400_sample_23500000.pkl (sha256 cffb53e2...acbf), pulled from
  Brandes-Lab/proteinbert_data_files and cached under work/proteinbert/models/.
  Verified on load by checking that the model's sequence head reconstructs an
  unmasked VH sequence (1.00 against ~0.05 for random weights), which tests that
  the weight vector was loaded in the right order rather than testing the model.

Outputs, in the same schema as results/esm_degeneracy/ so the two directories can
be concatenated:
  summary_by_model.csv       one row per (model, item_set, bucket_w, standardised)
  epitope_split_by_model.csv one row per (model, scope, standardised)
  cosine_hist_by_model.csv   fixed-bin histograms at the primary bucket width
  headline.json              the raw-arm headline block
  Two columns are supersets of the ESM file: `standardised` is added to the
  epitope table (the ESM run computes the epitope AUC on raw embeddings only --
  epitope_test() is called outside its standardisation loop -- so its 0.708 is a
  raw number), and `track` records local against global.

    work/proteinbert/venv/bin/python scripts/proteinbert_ab_degeneracy.py

Nothing under results/esm_degeneracy/ is read or written.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
OUT = PROJECT / "results" / "proteinbert_degeneracy"
MODEL_DIR = PROJECT / "work" / "proteinbert" / "models"
DUMP_NAME = "epoch_92400_sample_23500000.pkl"

LOCAL_LAYER = "seq-merge2-norm-block6"    # final per-residue hidden state
GLOBAL_LAYER = "global-merge2-norm-block6"  # final whole-sequence vector
N_BLOCKS = 6
D_LOCAL, D_GLOBAL = 128, 512

# A sanity VH used only to confirm the weight dump loaded in the right order.
PROBE_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADS"
            "VKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGRYFDYWGQGTLVTVSS")


# --------------------------------------------------------------------------- #
#  import the ESM script's construction verbatim                                #
# --------------------------------------------------------------------------- #
def load_esm_module():
    """Exec scripts/esm_ab_degeneracy.py with a stub `torch`, return the module.

    The stub carries a `Tensor` attribute and is removed afterwards. scipy's
    array-API dispatch does `getattr(sys.modules['torch'], 'Tensor')` whenever it
    promotes an argument, so a bare empty module turns every later scipy call
    into an AttributeError. Deleting the stub once the file is exec'd puts scipy
    back on its normal no-torch path.
    """
    injected = "torch" not in sys.modules
    if injected:
        stub = types.ModuleType("torch")
        stub.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = stub
    try:
        path = Path(__file__).resolve().parent / "esm_ab_degeneracy.py"
        spec = importlib.util.spec_from_file_location("esm_ab_degeneracy", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if injected:
            del sys.modules["torch"]
    for fn in ("build_items", "load_cdr_idx", "standardise", "bucket_cosines",
               "summarize", "epitope_pairs", "epitope_test"):
        assert hasattr(mod, fn), f"esm_ab_degeneracy.py does not define {fn}"
    return mod


# --------------------------------------------------------------------------- #
#  embeddings                                                                   #
# --------------------------------------------------------------------------- #
def build_submodel(mg, seq_len, keras):
    """A two-output view of the pretrained model at one exact input length."""
    m = mg.create_model(seq_len, compile=False)
    return keras.models.Model(
        inputs=m.inputs,
        outputs=[m.get_layer(LOCAL_LAYER).output, m.get_layer(GLOBAL_LAYER).output])


def verify_weights(mg, enc, keras):
    """Weight-order check: the sequence head must reproduce an unmasked input.

    This tests that the flat weight list in the pickle was assigned to the right
    variables: a wrong assignment gives ~1/26 agreement, a right one gives ~1.0.
    Reconstructing an input with nothing masked is easy for any trained denoising
    model, so it says nothing about ProteinBERT itself.
    """
    from proteinbert.tokenization import index_to_token
    L = len(PROBE_VH)
    m = mg.create_model(L + 2, compile=False)
    X = enc.encode_X([PROBE_VH], L + 2)
    out_seq, _ = m.predict(X, verbose=0)
    pred = out_seq[0, 1:L + 1].argmax(-1)
    rec = "".join(index_to_token.get(int(i), "?") for i in pred)
    acc = float(np.mean([a == b for a, b in zip(rec, PROBE_VH)]))
    print(f"[weights] unmasked-reconstruction agreement {acc:.3f} "
          f"(random weights give ~0.04)", flush=True)
    assert acc > 0.9, "pretrained weights did not load correctly"
    return acc


def embed_all(seqs, cdr_idx, max_tokens=200_000):
    """{seq -> local mean vector}, {seq -> CDR mean vector}, {seq -> global vector}.

    Sequences are grouped by exact length so that no <PAD> token enters a
    forward pass; see the module docstring.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras
    from proteinbert import load_pretrained_model

    mg, enc = load_pretrained_model(
        local_model_dump_dir=str(MODEL_DIR), local_model_dump_file_name=DUMP_NAME,
        download_model_dump_if_not_exists=False)
    verify_weights(mg, enc, keras)

    by_len = {}
    for s in seqs:
        by_len.setdefault(len(s), []).append(s)

    out_loc, out_cdr, out_glb = {}, {}, {}
    t0 = time.time()
    done = 0
    for k, L in enumerate(sorted(by_len)):
        group = by_len[L]
        sub = build_submodel(mg, L + 2, keras)
        bs = max(1, min(len(group), max_tokens // (L + 2)))
        for i in range(0, len(group), bs):
            chunk = group[i:i + bs]
            X = enc.encode_X(chunk, L + 2)
            loc, glb = sub.predict(X, batch_size=len(chunk), verbose=0)
            # Strip <START>/<END>: residue j of s is token j+1, exactly as the
            # ESM run strips BOS/EOS.
            res = loc[:, 1:L + 1, :].astype(np.float32)
            for j, s in enumerate(chunk):
                out_loc[s] = res[j].mean(0)
                out_glb[s] = glb[j].astype(np.float32)
                if s in cdr_idx:
                    out_cdr[s] = res[j][cdr_idx[s]].mean(0)
            done += len(chunk)
        del sub
        keras.backend.clear_session()
        if (k + 1) % 50 == 0:
            print(f"    [embed] {k+1}/{len(by_len)} lengths, {done}/{len(seqs)} "
                  f"seqs, {time.time()-t0:.0f}s", flush=True)
    print(f"[embed] {done} sequences in {time.time()-t0:.0f}s", flush=True)
    return out_loc, out_cdr, out_glb


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket_w", type=int, default=None)
    ap.add_argument("--cdr_scheme", default="imgt", choices=("imgt", "chothia"))
    ap.add_argument("--debug_n", type=int, default=0,
                    help="DEBUG ONLY: truncate every item set to N sequences. "
                         "Writes to a _DEBUG directory; never use for results.")
    args = ap.parse_args()

    E = load_esm_module()
    bucket_w = args.bucket_w if args.bucket_w is not None else E.BUCKET_W

    global OUT
    if args.debug_n:
        OUT = OUT.parent / (OUT.name + "_DEBUG")
        print(f"[DEBUG] writing to {OUT} -- real results are not touched", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sets, ab_rows, prov = E.build_items()
    print("[items] " + json.dumps(prov), flush=True)
    if args.debug_n:
        sets = {k: v.head(args.debug_n).reset_index(drop=True) for k, v in sets.items()}
        keep = set(sets["antibody"].seq) | set(sets["antibody_rows"].seq)
        ab_rows = ab_rows[ab_rows.vh_seq.isin(keep)]

    pairs = E.epitope_pairs(ab_rows, bucket_w)
    print(f"[epitope] {len(pairs)} within-antigen pairs over "
          f"{pairs.antigen_id.nunique()} antigens; "
          f"{int(pairs.same_bucket.sum())} in the same length bucket", flush=True)

    cdr_idx, cdr_tot = E.load_cdr_idx(args.cdr_scheme)
    ab = sets["antibody"]
    have = ab.seq.isin(cdr_idx)
    sets["antibody_cdr"] = pd.DataFrame({
        "seq": ab.seq[have].values,
        "len": [cdr_tot[s] for s in ab.seq[have]]})
    print(f"[cdr] scheme={args.cdr_scheme}: {int(have.sum())} of {len(ab)} unique VH "
          f"have spans; CDR window median "
          f"{int(np.median(sets['antibody_cdr']['len']))} residues", flush=True)

    all_seqs = sorted({s for k in ("antibody", "antigen", "antigen_exact",
                                   "shuffled_ab") for s in sets[k].seq})
    print(f"[embed] {len(all_seqs)} unique sequences, "
          f"{len({len(s) for s in all_seqs})} distinct lengths", flush=True)

    emb_loc, emb_cdr, emb_glb = embed_all(all_seqs, cdr_idx)

    rows, ep_rows, hist_rows = [], [], []
    tracks = [("proteinbert_local", "local", D_LOCAL, emb_loc,
               ("antibody", "antibody_cdr", "antibody_rows", "antigen",
                "antigen_exact", "shuffled_ab")),
              ("proteinbert_global", "global", D_GLOBAL, emb_glb,
               ("antibody", "antibody_rows", "antigen", "antigen_exact",
                "shuffled_ab"))]

    for name, track, width, emb, item_sets in tracks:
        # Standardisation corpus: every sequence this track embedded, antibodies
        # antigens and shuffles together -- the ESM run's rule, unchanged.
        _all = np.stack([emb[k] for k in emb])
        MU, SD = _all.mean(0), _all.std(0)
        if track == "local" and emb_cdr:
            _allc = np.stack([emb_cdr[k] for k in emb_cdr])
            MUC, SDC = _allc.mean(0), _allc.std(0)
        else:
            MUC, SDC = MU, SD

        for set_name in item_sets:
            df = sets[set_name]
            src = emb_cdr if set_name == "antibody_cdr" else emb
            V0 = np.stack([src[s] for s in df.seq])
            mu, sd = (MUC, SDC) if set_name == "antibody_cdr" else (MU, SD)
            for std_flag, V in ((0, V0), (1, E.standardise(V0, mu, sd))):
                for bw in (bucket_w,) + E.BUCKET_SENS:
                    c = E.bucket_cosines(V, df["len"].values, bw)
                    rows.append(dict(model=name, track=track, width=width,
                                     layer=N_BLOCKS, item_set=set_name,
                                     n_items=len(df), bucket_w=bw,
                                     standardised=std_flag, **E.summarize(c)))
                    if bw == bucket_w and len(c):
                        edges = (E.HIST_BINS if not std_flag else
                                 np.linspace(-1.0, 1.0, len(E.HIST_BINS)))
                        h, _ = np.histogram(c, bins=edges)
                        hist_rows.append(dict(model=name, track=track,
                                              item_set=set_name,
                                              standardised=std_flag,
                                              lo=float(edges[0]), hi=float(edges[-1]),
                                              n_pairs=int(len(c)),
                                              **{f"b{i}": int(v) for i, v in enumerate(h)}))

        # Epitope split, both arms. The ESM run computes this on raw embeddings
        # only. Feeding epitope_test() a standardised embedding dict changes
        # nothing about the test -- it re-normalises each vector and takes the
        # cosine either way -- so the same function answers both questions.
        std_emb = {s: E.standardise(v, MU, SD) for s, v in emb.items()}
        for std_flag, src in ((0, emb), (1, std_emb)):
            for scope in ("same_length_bucket", "all_length_pairs"):
                r = E.epitope_test(pairs, src, name, scope, bucket_w)
                if r:
                    ep_rows.append(dict(track=track, standardised=std_flag, **r))

        pd.DataFrame(rows).to_csv(OUT / "summary_by_model.csv", index=False)
        pd.DataFrame(ep_rows).to_csv(OUT / "epitope_split_by_model.csv", index=False)
        pd.DataFrame(hist_rows).to_csv(OUT / "cosine_hist_by_model.csv", index=False)
        print(f"  [{name}] done", flush=True)

    sm = pd.DataFrame(rows)
    ep = pd.DataFrame(ep_rows)
    prim = sm[(sm.bucket_w == bucket_w) & (sm.standardised == 0)]
    cols = ["n_items", "n_pairs", "median", "mean", "iqr",
            "frac_gt_0.95", "frac_gt_0.99"]
    headline = dict(
        question="does the ESM-2 antibody-embedding degeneracy reproduce in a "
                 "second, architecturally different protein language model",
        model="ProteinBERT (Brandes et al. 2022), pretrained dump "
              "epoch_92400_sample_23500000.pkl",
        representation={
            "proteinbert_local": f"{LOCAL_LAYER}, mean over residue positions "
                                 "(START/END stripped) -- the analogue of ESM-2's "
                                 "final per-residue layer",
            "proteinbert_global": f"{GLOBAL_LAYER}, one 512-d vector per sequence; "
                                  "annotation input is all zeros at inference"},
        seed=E.SEED, bucket_w=bucket_w,
        oc_same_threshold=E.OC_SAME, provenance=prov, per_model={})
    for name, _, _, _, _ in tracks:
        g = prim[prim.model == name].set_index("item_set")
        headline["per_model"][name] = {
            s: {k: g.loc[s, k] for k in cols if k in g.columns} for s in g.index}
    headline["epitope_split"] = ep[ep.scope == "same_length_bucket"].to_dict("records")
    (OUT / "headline.json").write_text(
        json.dumps(headline, indent=2, default=lambda o: float(o)))
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
