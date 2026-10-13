"""Every module together on synthetic data: hold-out, folds, pair out of fold, comparison, exhibits, evidence file.

This is the shape the 2026 run script will take once clean.py exists; only the data-specific parts (loading,
cleaning, the incumbent, the segments, the size columns) change.
"""
import numpy as np
import pandas as pd

from src.evidence import (adjustment_summary, ae_by_segment, disagreement_check, dislocation, double_lift, gini,
                          lift_table, score_table, seed_scores)
from src.glm import ForwardGLM
from src.pipeline import ModelPair, PredictionRanges, ReferralRule
from src.trees import TreeModel
from src.validation import SEEDS, fitting_mask, lock_holdout, make_folds, oof_predictions, open_holdout, paired_by_seed

FAST = dict(num_boost_round=200, learning_rate=0.05)


def run(d, y, family, t, workdir, offset=None, test_share=0.2, segment="region", size_cols=("size",), seeds=SEEDS):
    y = np.asarray(y, dtype=float)
    n_all = len(y)
    off_all = np.zeros(n_all) if offset is None else np.asarray(offset)
    # a "test file": the last rows, never used for fitting (like the competition's test inputs)
    n_test = int(test_share * n_all)
    tr, te = d.iloc[:-n_test].reset_index(drop=True), d.iloc[-n_test:].reset_index(drop=True)
    y_tr, off_tr, off_te = y[:-n_test], off_all[:-n_test], off_all[-n_test:]
    ids = np.arange(len(y_tr))

    lock = workdir / "holdout.json"
    lock_holdout(ids, t["holdout_share"], lock)
    fit = fitting_mask(lock, ids)
    d_f, y_f, o_f = tr[fit].reset_index(drop=True), y_tr[fit], off_tr[fit]
    folds = make_folds(len(y_f), seeds=seeds)

    pair = ModelPair(ForwardGLM.from_thresholds(family, t), TreeModel(family, FAST))
    current = ForwardGLM(family, min_gain=1.0, min_level_count=int(t["glm_min_level_count"]), forced=["x1"])  # stand-in incumbent

    def fp(a, b, s):
        p = pair.clone().fit(d_f.iloc[a], y_f[a], o_f[a]).predict(d_f.iloc[b], o_f[b])
        import copy
        p["current"] = copy.deepcopy(current).fit(d_f.iloc[a], y_f[a], o_f[a]).predict_link(d_f.iloc[b], o_f[b])
        return p

    oof = oof_predictions(fp, len(y_f), folds)
    s0 = seeds[0]
    o = oof[s0]
    ev = {}
    ev["seed_scores"] = seed_scores(family, y_f, oof, ["current", "glm", "gbm", "blend"])
    ev["compare"] = paired_by_seed(y_f, {s: v.current.values for s, v in oof.items()}, {s: v.blend.values for s, v in oof.items()},
                                   family=family, n_boot=300)

    final = pair.clone().fit(d_f, y_f, o_f)
    cur_final = current.fit(d_f, y_f, o_f)
    hold = open_holdout(lock, ids, reason="synthetic end-to-end test")
    d_h, y_h, o_h = tr[hold].reset_index(drop=True), y_tr[hold], off_tr[hold]
    app, ph = final.predict(d_f, o_f), final.predict(d_h, o_h)
    ev["score_table"] = score_table(family, {
        "current": dict(apparent=(y_f, cur_final.predict_link(d_f, o_f)), cv=(y_f, o.current), holdout=(y_h, cur_final.predict_link(d_h, o_h))),
        "proposed": dict(apparent=(y_f, app.blend), cv=(y_f, o.blend), holdout=(y_h, ph.blend))})

    mu, mu_cur = family.inverse(o.blend), family.inverse(o.current)
    expo = np.exp(o_f) if offset is not None else None
    ev["lift"] = dict(proposed=lift_table(y_f, mu, exposure=expo), current=lift_table(y_f, mu_cur, exposure=expo))
    ev["double_lift"] = double_lift(y_f, mu_cur, mu, exposure=expo)
    ev["ae_segments"] = {segment: ae_by_segment(y_f, mu, d_f[segment], exposure=expo)}
    ev["gini"] = dict(proposed=gini(y_f, mu, expo), current=gini(y_f, mu_cur, expo))
    ev["disagreement"] = disagreement_check(family, y_f, oof, top_share=0.02)
    rng = PredictionRanges.from_thresholds(family, t).fit(y_f, o.blend)
    ev["ranges"] = rng.coverage(y_f, o.blend, folds[s0])
    ev["glm_path"] = final.glm.path
    ev["glm_coef"] = final.glm.coef_table().to_dict("records")

    pt = final.predict(te, off_te)
    ref = ReferralRule.from_thresholds(t, size_cols=list(size_cols), segment_col=segment).fit(d_f).apply(te, pt)
    ev["adjustment"] = adjustment_summary(final.adjustment(pt))
    ev["dislocation"] = dislocation(family.inverse(pt.blend), cur_final.predict(te, off_te))
    ev["referral"] = dict(n=len(te), n_referred=int(ref.referred.sum()), n_out_of_evidence=int(ref.out_of_evidence.sum()),
                          n_gap=int(ref.gap_flag.sum()),
                          rows=ref[ref.referred][["size_ratio", "unseen", "gap", "reason", "treatment"]].head(10).to_dict("records"))
    rows = {r["model"]: r for r in ev["seed_scores"]["rows"]}
    st = {r["model"]: r for r in ev["score_table"]}
    ev["summary"] = dict(title="Synthetic test run", team="test", metric=family.score_name, public_lb="(not submitted)",
                         n_seeds=len(seeds), current_cv=rows["current"]["mean"], proposed_cv=rows["blend"]["mean"],
                         n_holdout=int(hold.sum()), current_holdout=st["current"]["holdout"], proposed_holdout=st["proposed"]["holdout"],
                         within5=ev["dislocation"]["within5"], n_referred=ev["referral"]["n_referred"], n_test=len(te))
    return ev, oof
