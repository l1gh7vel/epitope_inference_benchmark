"""Functional cost, scored against the AUTHORS' significant escape SITES.

Same question and same shape as `dms_04`/`dms_05`, one ingredient swapped. There the
outcome was CONTINUOUS escape and the test was a rank test. Here the outcome is the
binary call the DMS authors publish -- a site either is or is not a site of
significant escape -- and the test is a hypergeometric (one-sided Fisher):

    of the k residues a method ranks highest, how many are significant escape
    sites, against how many would land there by chance?

The gold structural epitope is run through the identical test and drawn as the
dashed reference, so the vertical gap is still the functional cost -- just measured
against a site list rather than a gradient.

THREE ARMS, ONE SCRIPT, chosen by EIB_TASK (the same switch stages 01-04 follow).
`dms` (HIV-1 Env, the default) is unchanged byte for byte; `rsv_f` and `niv` were
added 2026-11-04 for Figure 5c and 5d. Everything that differs between them is in
ARM_SPEC and in `arm_cases()` -- the ranking, the hypergeometric and the random
floor below them are one implementation for all three. What differs, and why each
one would silently produce a wrong number if it were assumed instead:

  * THE ESCAPE STATISTIC IS NOT THE SAME COLLAPSE. HIV collapses the mutations at
    a site with a MEAN (`--escape_stat mean`, the dms.py default); RSV and Nipah
    use their authors' own FLOORED SUM, sum_i max(escape_i, 0). It is not a
    rescaling of the mean -- the floor removes the negative (increased
    neutralisation) tail and the sum weights a site by how many mutations were
    measured there. `--escape_stat` is declared per arm and an arm refuses a
    statistic it does not have, rather than defaulting to `mean` and scoring the
    wrong vector.
  * THE NPZ KEYS DIFFER. HIV: `{antibody}|{dataset}|{stat}` for escape and
    `{antibody}__{target}` for the target set, in two files. RSV/Nipah keep both
    in one file keyed `{antibody}|stat` and `{antibody}|escape`, and Nipah's
    escape mask comes in two frames -- `|escape` over the 510 SCANNED sites and
    `|escape_fold` over the 532 fold positions. Only the fold frame is
    commensurable with a score vector, so that is the one read here.
  * NOT EVERY ANTIBODY HAS A GOLD STRUCTURAL EPITOPE. RSV palivizumab has a
    Fab-alone structure (2HWZ) and suptavumab has no PDB entry at all. They have
    escape data, they are in the panel, and they have no dashed reference line.
    `has_gold` is asked; `lab[f"{ab}__gold"]` is never indexed unconditionally.
  * THE k GRID IS SIZED TO THE ANTIGEN. 856 residues (Env) against 504 (RSV F)
    and 532 (Nipah RBP). See KS_* below.
  * BEPIPRED IS OPTIONAL. It is one vector per ANTIGEN, so it exists only where
    that antigen has been run; a missing file drops the series rather than
    faking a floor.

TWO DIFFERENCES FROM dms_04 THAT MATTER:

  * Ranking is restricted to MEASURED sites. The continuous version ranked all 856
    positions and then dropped unmeasured ones from the test, which makes the
    effective k vary between methods. With a hypergeometric the universe has to be
    fixed, so top-k here means "top k of the assayed positions" (661 on HIV, all
    504 on RSV, 510 of 532 on Nipah).
  * On the HIV arm only the nine Dingens antibodies appear. Radford & Bloom
    publish no significant-site call, so this definition does not exist for that
    arm. The continuous test uses every assayed site's escape value and needs no
    call of which sites count as significant, so it is defined on every arm; it
    measures how well the predicted ranking orders escape across the whole assayed
    range, which is the readout the question asks for.

The WALLE checkpoint is the project checkpoint, `treasured-sweep-8`, from
config/walle.py, the same one every task uses. This arm has never ranked the ten
AsEP checkpoints on escape recall, so no other choice is made here.

Three readouts per (antibody, method, k):
    p        one-sided Fisher, the direct analogue of the rank test's p
    fold     observed overlap / expected overlap -- effect size, n-independent
    recall   fraction of the significant escape sites captured, which is the
             plainest reading of "did the prediction include the escape spots"

    EIB_TASK=dms   PYTHONPATH=scripts:config python scripts/dms_08_binary_cost.py
    EIB_TASK=rsv_f PYTHONPATH=scripts:config python scripts/dms_08_binary_cost.py
    EIB_TASK=niv   PYTHONPATH=scripts:config python scripts/dms_08_binary_cost.py
"""
import importlib, json, sys
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

# Same rule as config/*.py: EIB_ROOT wins, else this file's own repo location.
PROJECT = Path(os.environ.get("EIB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / "config"))
sys.path.insert(0, str(PROJECT / "scripts"))
# Which arm. `dms` is the default so every existing invocation, and dms_13 which
# loads this module directly, behaves exactly as before.
ARM = os.environ.get("EIB_TASK", "dms")
cfg = importlib.import_module(ARM)                           # noqa: E402
import walle                                              # noqa: E402

MAN = PROJECT / "data" / "manifests"
SIG = PROJECT / "data" / "source" / "dms_dingens2019" / "significant_sites"

# k grids, one per ANTIGEN LENGTH. The HIV grid runs to 400 of 856 positions
# (47%). RSV F is 504 and Nipah RBP 532, so the same grid would be asking what
# fraction of the escape sites sit in three quarters of the protein, which is a
# question with a known answer. Both virus grids are the HIV grid truncated at
# 250 -- half of each antigen, the same share of the protein the HIV grid's 400
# is -- and they keep every k the HIV grid has below that, so a k on one arm is
# the same k on another. The drawn panel is capped at k = 50 on all three arms
# (dms_09), which is above the largest gold epitope on each: 49 residues on HIV,
# 27 on RSV, 30 on Nipah (counting, as the panel's dot does, only the gold
# residues inside the assayed universe).
KS_856 = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400]
KS_HALF = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250]
N_RANDOM = 400
NAMES = {"VRC01": "VRC01", "3BNC117": "3BNC117", "101074": "10-1074",
         "PGT121": "PGT121", "PGT151": "PGT151", "PG9": "PG9",
         "10E8": "10E8", "PGT145": "PGT145", "VRC34": "VRC34"}

# Per arm: where the tables go, the k grid, which site-collapse statistics the
# arm's escape file actually contains (the FIRST is the declared default), which
# target sets exist, and the name of the antigen-level BepiPred-3.0 vector if one
# has been run. `stat_words` is the one-line description that goes into the csv,
# so a table can always say what its escape numbers are a collapse of.
ARM_SPEC = {
    "dms": dict(
        results="dms_case", ks=KS_856,
        stats=("mean", "sum", "max"),
        stat_words={"mean": "mean over the mutations at a site of the Dingens 2019 "
                            "avgfracsurvive",
                    "sum": "sum over the mutations at a site of avgfracsurvive",
                    "max": "max over the mutations at a site of avgfracsurvive"},
        targets=("escape",),
        target_words="Dingens et al. gamma-tail FDR 0.01 significant-site call",
        bepipred="bepipred_env.npy"),
    "rsv_f": dict(
        results="rsv_case", ks=KS_HALF,
        stats=("floored_sum",),
        stat_words={"floored_sum": "sum over the mutations at a site of "
                                   "max(Fab escape, 0) -- the authors' own floored "
                                   "SUM, not a mean"},
        targets=("escape",),
        target_words="declared top 10 of 504 sites per antibody "
                     "(scripts/rsv_01_escape_sites.py)",
        bepipred="bepipred_rsvF.npy"),
    "niv": dict(
        results="niv_case", ks=KS_HALF,
        stats=("floored_sum",),
        stat_words={"floored_sum": "sum over the mutations at a site of "
                                   "max(escape_mean, 0) -- the authors' own floored "
                                   "SUM, not a mean"},
        targets=("escape",),
        target_words="declared top 10 of the 510 scanned sites per antibody "
                     "(scripts/niv_01_escape_sites.py)",
        bepipred="bepipred_rbp.npy"),
}
if ARM not in ARM_SPEC:
    raise SystemExit(f"EIB_TASK={ARM!r}: this script knows {sorted(ARM_SPEC)}")
SPEC = ARM_SPEC[ARM]
RES = PROJECT / "results" / SPEC["results"]
KS = SPEC["ks"]


WALLE_CKPT = walle.HEADLINE_CHECKPOINT
# The structure WALLE reads. `af3` is the same AF3 fold AlphaFold3 and ESMFold2
# are scored from, so the three methods see one input; `experimental` exists on
# the RSV and Nipah arms for the cases with a solved complex and is the subject
# of dms_12/dms_13, not of the headline panel.
WALLE_SRC = os.environ.get("EIB_WALLE_SOURCE", "af3")


def walle_label(ckpt=None, force=False):
    """Legend label for the WALLE series.

    Bare "WALLE" by default, so the legend names the checkpoint in words rather
    than in the key.

    `force` names it regardless. A panel generated to be put NEXT TO another
    checkpoint's panel has to say which one it is even when it is the default,
    and after 2026-11-02 the default is itself one side of that comparison.
    """
    ck = ckpt or WALLE_CKPT
    if force:
        return f"WALLE ({ck.replace('-sweep', '')})"
    return "WALLE"


# WALLE is read STRAIGHT FROM ITS OWN TREE
# (work/<arm>/walle/scores/<source>/<checkpoint>/) rather than from a copied-in
# scores_walle/, so there is exactly one artefact per case and no second copy to
# go stale. All ten released checkpoints are scored under that tree on every arm,
# so the pick is a lookup, not a re-run.
METHOD_DIRS = {"AlphaFold3": ("scores", "contact_top"),
               "ESMFold2": ("scores_esm", "contact_top"),
               "WALLE": (f"walle/scores/{WALLE_SRC}/{WALLE_CKPT}", "walle_score")}


def sig_sites(stem):
    """The Dingens significant-site list, read from their published txt. HIV only."""
    return {int(t) for t in (SIG / f"{stem}_SignificantSitesList.txt").read_text().split(",")
            if t.strip().isdigit()}


def enrich(sel, is_sig, universe):
    """One-sided Fisher + fold enrichment + recall, all within `universe`."""
    a = int((sel & is_sig).sum()); b = int((sel & ~is_sig & universe).sum())
    c = int((~sel & is_sig & universe).sum()); d = int((~sel & ~is_sig & universe).sum())
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    k, n_sig, n = a + b, int((is_sig & universe).sum()), int(universe.sum())
    exp = k * n_sig / n if n else np.nan
    return dict(overlap=a, n_sig=n_sig, expected=round(exp, 3), p=float(p),
                fold=float(a / exp) if exp > 0 else np.nan,
                recall=float(a / n_sig) if n_sig else np.nan)


def load_scores(case_id, L):
    out = {}
    for name, (sub, key) in METHOD_DIRS.items():
        p = cfg.WORK / sub / f"{case_id}.npz"
        if p.exists():
            d = np.load(p)
            k = key if key in d else ("score" if "score" in d else None)
            if k and len(d[k]) == L:
                out[name] = np.asarray(d[k], float)
    # BepiPred-3.0 is ONE vector per antigen, reused for every antibody -- that is
    # the definition of the antibody-blind class, not a shortcut. It is also
    # OPTIONAL: an arm whose antigen has not been through BepiPred simply has no
    # blind floor on this panel, which is stated rather than faked.
    bp = cfg.WORK / SPEC["bepipred"]
    if bp.exists():
        v = np.load(bp)
        if len(v) == L:
            out["BepiPred-3.0"] = v
        else:
            print(f"  !! {bp.name} is {len(v)} long, antigen is {L} -- series dropped")
    return out


# --------------------------------------------------------------- the arms ----
# RSV's manifest names the antigenic sites `region_0` / `region_II` / ...; the
# literature and the figure call them "site 0", "site II". One mapping, here.
RSV_SITE = {"region_0": "site 0", "region_I": "site I", "region_II": "site II",
            "region_III": "site III", "region_IV": "site IV", "region_V": "site V"}


def arm_cases(chain_arm, targets, stat):
    """Per-antibody inputs for this arm: the universe, the target set, the gold.

    One record per antibody, all three masks in the ANTIGEN FRAME of this arm
    (856 / 504 / 532) and all already intersected with the universe, so the
    caller does not have to know which arm it is on. `gold` is None where no
    solved complex exists; `note` is the one phrase that has to appear on that
    antibody's panel for its numbers to be read correctly.
    """
    man = pd.read_csv(cfg.MANIFEST)
    man = man[man.arm == chain_arm]
    L = cfg.ANTIGEN_LEN
    out = []

    if ARM == "dms":
        esc = np.load(MAN / "dms_escape.npz")
        tgt = np.load(MAN / "dms_functional_sites.npz")
        lab = np.load(MAN / "dms_labels.npz")
        rep = json.loads((MAN / "dms_gold_report.json").read_text())
        for r in man.itertuples():
            ab = r.antibody
            if ab not in NAMES.values():        # Radford antibodies: no binary call
                continue
            v = esc[f"{ab}|Dingens2019_BG505|{stat}"]
            universe = ~np.isnan(v)             # the 661 assayed positions
            out.append(dict(
                antibody=ab, case_id=r.case_id, universe=universe,
                is_sig=tgt[f"{ab}__{targets}"].astype(bool) & universe,
                gold=lab[f"{ab}__gold"].astype(bool) & universe,
                epitope_class=rep[ab]["epitope_class"], note=""))

    elif ARM == "rsv_f":
        esc = np.load(cfg.ESCAPE_NPZ)
        lab = np.load(cfg.LABELS_NPZ)
        for r in man.itertuples():
            ab = r.antibody
            # Every one of the 504 sites is assayed on this arm; the isfinite
            # test is kept rather than assumed, because it is what makes the
            # universe mean the same thing it means on the other two arms.
            universe = np.isfinite(esc[f"{ab}|stat"])
            gold = (lab[f"{ab}__gold"].astype(bool) & universe
                    if f"{ab}__gold" in lab.files else None)
            # Why this antibody's panel reads the way it does, in one phrase.
            if ab == "Suptavumab":
                note = "no solved complex"
            elif ab == "Palivizumab":
                note = "Fab-alone structure"
            elif bool(r.measured_null):
                note = "measured null"
            else:
                note = ""
            out.append(dict(
                antibody=ab, case_id=r.case_id, universe=universe,
                is_sig=esc[f"{ab}|escape"].astype(bool) & universe, gold=gold,
                epitope_class=RSV_SITE.get(r.epitope_class, r.epitope_class),
                note=note))

    elif ARM == "niv":
        esc = np.load(cfg.ESCAPE_NPZ)
        lab = np.load(cfg.LABELS_NPZ) if cfg.LABELS_NPZ.exists() else None
        # 510 of the 532 fold positions were scanned. The other 22 have no escape
        # measurement at all, so they are outside the universe exactly as HIV's
        # unassayed positions are.
        universe = esc["scanned_mask"].astype(bool)
        for r in man.itertuples():
            ab = r.antibody
            gold = (lab[f"{ab}__gold"].astype(bool) & universe
                    if lab is not None and f"{ab}__gold" in lab.files else None)
            out.append(dict(
                antibody=ab, case_id=r.case_id, universe=universe,
                # |escape is indexed over the 510 SCANNED sites; |escape_fold is
                # the same set in the 532 fold frame, which is the only one
                # commensurable with a score vector.
                is_sig=esc[f"{ab}|escape_fold"].astype(bool) & universe, gold=gold,
                # No epitope-class taxonomy is published for this arm, so the
                # second title line carries the provenance that actually changes
                # how the gold line is read: WHICH complex it came from and
                # whether that complex is Nipah or Hendra G.
                epitope_class=f"{r.pdb}, {r.antigen_in_structure} G",
                note="proxy light chain" if bool(r.sequence_is_proxy) else ""))

    for rec in out:
        for key in ("universe", "is_sig"):
            assert len(rec[key]) == L, f"{rec['antibody']}: {key} is not {L} long"
        assert rec["gold"] is None or len(rec["gold"]) == L
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="vhvl", choices=["vhvl", "vh"])
    ap.add_argument("--seed", type=int, default=0)
    # Which set of "sites that matter" to score against. On the HIV arm it is
    # built by dms_10, which also documents why the negative direction is
    # mutational constraint rather than a negative escape tail. escape reproduces
    # the behaviour this script had before the option existed, bit for bit. The
    # virus arms have escape only.
    ap.add_argument("--targets", default="escape",
                    choices=["escape", "constraint", "union"])
    # The site-collapse statistic is set per arm, never defaulted: HIV's escape file
    # holds mean/sum/max of avgfracsurvive, RSV's and Nipah's hold ONE vector,
    # their authors' floored sum, and scoring that as if it were a mean is a
    # silently wrong number rather than an error. Each arm declares what it has.
    ap.add_argument("--escape_stat", default=None,
                    help=f"{ARM}: {' | '.join(SPEC['stats'])} (default {SPEC['stats'][0]})")
    a = ap.parse_args()

    if a.targets not in SPEC["targets"]:
        raise SystemExit(f"--targets {a.targets} does not exist on arm {ARM}: "
                         f"it has {SPEC['targets']}")
    stat = a.escape_stat or SPEC["stats"][0]
    if stat not in SPEC["stats"]:
        raise SystemExit(f"--escape_stat {stat} does not exist on arm {ARM}: its "
                         f"escape file holds {SPEC['stats']}. Scoring the wrong "
                         f"collapse would be a plausible wrong number, not an error.")

    L = cfg.ANTIGEN_LEN
    rng = np.random.default_rng(a.seed)
    curves, golds = [], []

    for rec in arm_cases(a.arm, a.targets, stat):
        ab, universe, is_sig, gold = (rec["antibody"], rec["universe"],
                                      rec["is_sig"], rec["gold"])
        g = dict(overlap=np.nan, n_sig=int(is_sig.sum()), expected=np.nan,
                 p=np.nan, fold=np.nan, recall=np.nan) if gold is None else \
            enrich(gold, is_sig, universe)
        row = dict(antibody=ab, arm=a.arm, epitope_class=rec["epitope_class"],
                   n_gold=0 if gold is None else int(gold.sum()), **g)
        if ARM != "dms":
            # Only the arms that HAVE antibodies without a solved complex carry
            # these; adding them to the HIV table would change a committed file
            # for no gain.
            row.update(has_gold=gold is not None, note=rec["note"])
        golds.append(row)

        idx = np.where(universe)[0]                    # rank ONLY measured sites
        for name, sc in load_scores(rec["case_id"], L).items():
            order = idx[np.lexsort((idx, -sc[idx]))]
            for k in KS:
                sel = np.zeros(L, bool); sel[order[:k]] = True
                curves.append(dict(antibody=ab, arm=a.arm, method=name, k=k,
                                   epitope_class=rec["epitope_class"],
                                   **enrich(sel, is_sig, universe)))
        for k in KS:
            acc = []
            for _ in range(N_RANDOM):
                sel = np.zeros(L, bool); sel[rng.choice(idx, k, replace=False)] = True
                acc.append(enrich(sel, is_sig, universe))
            curves.append(dict(antibody=ab, arm=a.arm, method="Random", k=k,
                               epitope_class=rec["epitope_class"],
                               overlap=float(np.mean([x["overlap"] for x in acc])),
                               n_sig=acc[0]["n_sig"], expected=acc[0]["expected"],
                               p=float(np.mean([x["p"] for x in acc])),
                               fold=float(np.nanmean([x["fold"] for x in acc])),
                               recall=float(np.mean([x["recall"] for x in acc]))))

    RES.mkdir(parents=True, exist_ok=True)
    suf = "" if a.targets == "escape" else f"_{a.targets}"
    pd.DataFrame(curves).to_csv(RES / f"binary_cost_curves_{a.arm}{suf}.csv", index=False)
    pd.DataFrame(golds).to_csv(RES / f"binary_cost_gold_{a.arm}{suf}.csv", index=False)
    cur = pd.DataFrame(curves)
    # Provenance beside the tables, so a reader never has to infer which collapse
    # produced them or which antibodies had no gold line.
    (RES / f"binary_cost_meta_{a.arm}{suf}.json").write_text(json.dumps(dict(
        task=ARM, antigen=cfg.ANTIGEN_NAME, antigen_len=int(L), chain_arm=a.arm,
        escape_stat=stat, escape_statistic=SPEC["stat_words"][stat],
        target_set=a.targets, target_set_definition=SPEC["target_words"],
        ks=KS, n_random=N_RANDOM, seed=a.seed,
        walle_source=WALLE_SRC, walle_checkpoint=WALLE_CKPT,
        antibodies=sorted(cur.antibody.unique().tolist()),
        methods=sorted(cur.method.unique().tolist()),
        without_gold=[r["antibody"] for r in golds
                      if not r.get("has_gold", True)]), indent=2))
    print(f"arm={ARM}/{a.arm}: {cur.antibody.nunique()} antibodies, "
          f"{sorted(cur.method.unique())}; escape_stat={stat}; "
          f"WALLE={WALLE_SRC}/{WALLE_CKPT} -> {RES}")
    piv = cur[cur.k == 25].pivot_table(index="antibody", columns="method", values="recall")
    print(f"\nrecall of {a.targets} sites at k=25:")
    print(piv.round(3).to_string())
    print("\ngold epitope recall:")
    cols = ["n_gold", "overlap", "n_sig", "recall", "fold"]
    if ARM != "dms":
        cols = ["has_gold"] + cols + ["note"]
    print(pd.DataFrame(golds).set_index("antibody")[cols].round(3).to_string())


if __name__ == "__main__":
    main()
