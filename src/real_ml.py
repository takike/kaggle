"""LightGBM residual correction on top of the geosteering aligner (real data).

Per toe row, learn  (true_TVT - align_pred)  from features that include the
alignment's own confidence (Viterbi path cost, GR match residual, deviation
instability).  This lets the model shrink toward the structural trend exactly where
the alignment is unreliable -- the tail cases that dominate the pooled RMSE.

GroupKFold by case (a well is never split) mirrors the held-out-wells leaderboard.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from real_io import list_cases, load_case, split_known_pred  # noqa: E402
from predict_real import predict_case  # noqa: E402

import lightgbm as lgb

LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.04,
    num_leaves=63, min_child_samples=200, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=10.0, n_estimators=900, verbosity=-1,
)


def _roll(s, w, fn):
    r = s.rolling(w, center=True, min_periods=max(2, w // 4))
    return getattr(r, fn)()


def case_features(h, diag):
    """Feature frame for the toe rows of one case (+ align_pred, true if present)."""
    known, pred = split_known_pred(h)
    MD = h["MD"].values; Z = h["Z"].values; GR = h["GR"].values
    n = len(h)
    align_pred = h["TVT_input"].values.copy()
    e = diag["e"]
    align_pred[pred] = (diag["trend"] + e - Z)[pred]

    grs = pd.Series(GR)
    dZ = np.gradient(Z, MD)
    f = pd.DataFrame({
        "dist_past_ps": diag["dist_past_ps"],
        "frac_toe": diag["dist_past_ps"] / max(1.0, diag["dist_past_ps"].max()),
        "known_frac": known.mean(),
        "n_toe": int(pred.sum()),
        "Z": Z, "dZ": dZ, "absdZ": np.abs(dZ), "d2Z": np.gradient(dZ, MD),
        "GR": GR, "GR_grad": grs.diff().fillna(0).values,
        "GR_m11": _roll(grs, 11, "mean").values,
        "GR_s41": _roll(grs, 41, "std").values,
        "GR_m121": _roll(grs, 121, "mean").values,
        "e": e, "abs_e": np.abs(e),
        "path_cost": diag["path_cost"], "log_cost": np.log1p(diag["path_cost"]),
        "cost_m61": _roll(pd.Series(diag["path_cost"]), 61, "mean").values,
        "e_s61": _roll(pd.Series(e), 61, "std").values,
        "gr_match": diag["gr_match_resid"], "abs_grm": np.abs(diag["gr_match_resid"]),
        "r": diag["r"], "trend": diag["trend"],
        "to_ref_lo": (diag["trend"] + e - Z) - diag["ref_lo"],
        "to_ref_hi": diag["ref_hi"] - (diag["trend"] + e - Z),
    })
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    f["__align"] = align_pred
    if "TVT" in h.columns:
        f["__true"] = h["TVT"].values
    f["__pred_mask"] = pred
    return f


def build_dataset(split_dir, cases=None, verbose=True, cloud=None, **pkw):
    """Build the per-toe-row feature dataset.

    If `cloud` (a StructuralCloud) is given, the aligner uses the offset-well dip
    prior, excluding each case from its own dip estimate (leave-one-well-out).
    """
    cases = cases or list_cases(split_dir)
    feats, groups = [], []
    t0 = time.time()
    if cloud is not None:
        from offset import dip_trend
    for k, cid in enumerate(cases):
        h, tw = load_case(split_dir, cid)
        known, pred = split_known_pred(h)
        if pred.sum() < 10:
            continue
        tr = dip_trend(h, cloud, exclude_case=hash(cid) & 0xffffffff) \
            if cloud is not None else None
        _, diag = predict_case(h, tw, return_diag=True, trend_override=tr, **pkw)
        f = case_features(h, diag)
        f = f[f["__pred_mask"]].drop(columns="__pred_mask")
        f["__case"] = cid
        feats.append(f)
        if verbose and (k + 1) % 100 == 0:
            print(f"  features {k+1}/{len(cases)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(feats, ignore_index=True)
    return df


def pooled(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


FEATURE_COLS = None  # set in oof_eval


# Absolute-coordinate features do NOT generalise across cases (each well sits at a
# different depth; GroupKFold hands the model novel ranges) -> they overfit and
# corrupt well-aligned cases.  Keep only relative geometry, GR texture, and
# alignment-confidence features.
EXCLUDE = {"Z", "trend", "GR", "GR_m11", "GR_m121", "r"}


def oof_eval(df, n_splits=5, train_subsample=2, seed=0):
    """GroupKFold OOF: compare alignment vs alignment+ML residual (pooled RMSE)."""
    global FEATURE_COLS
    FEATURE_COLS = [c for c in df.columns
                    if not c.startswith("__") and c not in EXCLUDE]
    X = df[FEATURE_COLS].values.astype(np.float32)
    align = df["__align"].values
    true = df["__true"].values
    resid = true - align
    groups = df["__case"].values
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(gkf.split(X, resid, groups)):
        trs = tr[::train_subsample]
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(X[trs], resid[trs])
        oof[va] = m.predict(X[va])
    final = align + oof
    return {
        "align_pooled": pooled(true - align),
        "ml_pooled": pooled(true - final),
        "oof": oof, "final": final, "align": align, "true": true, "case": groups,
    }


def fit_residual_model(train_dir, cases=None, train_subsample=2, seed=0,
                       cloud=None, **pkw):
    """Train the residual LGBM on train cases; return (model, feature_cols)."""
    df = build_dataset(train_dir, cases, cloud=cloud, **pkw)
    feat_cols = [c for c in df.columns if not c.startswith("__") and c not in EXCLUDE]
    X = df[feat_cols].values.astype(np.float32)
    resid = (df["__true"] - df["__align"]).values
    m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
    m.fit(X[::train_subsample], resid[::train_subsample])
    return m, feat_cols


def predict_residual(model, feat_cols, h, diag):
    """Per-row residual correction for the toe rows of one case (0 elsewhere)."""
    f = case_features(h, diag)
    pred = f["__pred_mask"].values
    out = np.zeros(len(h))
    out[pred] = model.predict(f[pred][feat_cols].values.astype(np.float32))
    return out


def per_case_rmse(true, pred, case):
    d = pd.DataFrame({"e2": (true - pred) ** 2, "case": case})
    g = d.groupby("case")["e2"].mean() ** 0.5
    return g


if __name__ == "__main__":
    ncases = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all
    use_offset = "--offset" in sys.argv
    allc = list_cases("data_real/train")
    cases = allc if ncases == 0 else allc[:ncases]
    cloud = None
    if use_offset:
        from offset import StructuralCloud
        print("building structural cloud ...", flush=True)
        cloud = StructuralCloud.from_dir("data_real/train")
    print(f"building features for {len(cases)} cases "
          f"(offset prior={'on' if cloud else 'off'}) ...", flush=True)
    df = build_dataset("data_real/train", cases, cloud=cloud)
    print(f"rows: {len(df)}  features: {sum(1 for c in df.columns if not c.startswith('__'))}")
    res = oof_eval(df)
    print(f"\nalignment      pooled toe-RMSE : {res['align_pooled']:.3f}")
    print(f"alignment + ML pooled toe-RMSE : {res['ml_pooled']:.3f}")
    a = per_case_rmse(res["true"], res["align"], res["case"])
    m = per_case_rmse(res["true"], res["final"], res["case"])
    print(f"per-case median  align {a.median():.2f} -> ml {m.median():.2f}")
    print(f"per-case p90     align {a.quantile(.9):.2f} -> ml {m.quantile(.9):.2f}")
    print(f"per-case max     align {a.max():.2f} -> ml {m.max():.2f}")
    print("\nalpha sweep  final = align + alpha*oof:")
    for al in (0.0, 0.15, 0.3, 0.5, 0.8, 1.0):
        fp = res["align"] + al * res["oof"]
        pc = per_case_rmse(res["true"], fp, res["case"])
        print(f"  alpha={al:.2f}  pooled {pooled(res['true']-fp):.3f}  "
              f"median {pc.median():.2f}  p90 {pc.quantile(.9):.2f}")
    try:
        imp = pd.Series(lgb.LGBMRegressor(random_state=0, **LGB_PARAMS).fit(
            df[FEATURE_COLS].values.astype(np.float32),
            (df['__true']-df['__align']).values).feature_importances_,
            index=FEATURE_COLS).sort_values(ascending=False)
        print("\ntop features:\n" + imp.head(10).to_string())
    except Exception as ex:
        print("imp skip", ex)
