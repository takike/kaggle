"""End-to-end ROGII TVT pipeline + experiment harness.

Stages:  data -> geosteering alignment -> features -> LightGBM residual model
         -> blend -> along-hole smoothing.

Runs on synthetic data by default (no Kaggle creds needed).  Point ``--real`` at a
folder of real CSVs (after editing config.COLS / config.FILES) to run for-real.

CV is GroupKFold by well: a well is never split across train/val, mirroring the
held-out-wells nature of the real leaderboard and preventing within-well leakage.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
import config as C          # noqa: E402
from align import align_dataframe  # noqa: E402
from features import build_features  # noqa: E402

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# Ensemble of geosteering aligners.  Diverse (lam, window, prior) configs lock
# the GR cycle slightly differently; averaging is robust to the occasional well
# that an individual config locks one cycle off.  Their disagreement (std) is a
# strong confidence signal for the downstream model.
ALIGN_ENSEMBLE = [
    dict(lam=24, win_ft=22, prior_w=0),   # no prior: trusts GR shape fully
    dict(lam=28, win_ft=22, prior_w=0),
    dict(lam=32, win_ft=24, prior_w=2),
    dict(lam=40, win_ft=26, prior_w=4),
    dict(lam=48, win_ft=20, prior_w=8),   # strong prior: anchors hard to the zone
]


def align_ensemble(df, tw, cols, configs=None, **base_kw):
    """Run the aligner ensemble; return mean tvt_align (+ disagreement std)."""
    configs = configs or ALIGN_ENSEMBLE
    preds, costs, deltas = [], [], []
    for c in configs:
        r = align_dataframe(df, tw, cols, **{**base_kw, **c})
        preds.append(r["tvt_align"]); costs.append(r["align_cost"]); deltas.append(r["delta"])
    P = np.array(preds)
    return {
        "tvt_align": P.mean(0),
        "tvt_align_std": P.std(0),          # ensemble disagreement = uncertainty
        "delta": np.mean(deltas, 0),
        "align_cost": np.mean(costs, 0),
    }


def load_synth(n_wells=14, seed=42):
    from synth import generate
    df, tw = generate(n_wells=n_wells, seed=seed)
    return df, tw


def load_real(data_dir):
    f = C.FILES
    train = pd.read_csv(os.path.join(data_dir, f["train"]))
    tw = pd.read_csv(os.path.join(data_dir, f["typewell"]))
    return train, tw


def smooth_per_well(df, pred, cols, win=15):
    """Light along-hole median smoothing (structure/TVT are spatially coherent)."""
    out = pd.Series(pred, index=df.index).copy()
    for _, idx in df.groupby(cols["well"], sort=False).groups.items():
        idx = np.asarray(idx)
        sub = df.loc[idx].sort_values(cols["md"])
        sm = out.loc[sub.index].rolling(win, center=True, min_periods=1).median()
        out.loc[sub.index] = sm.values
    return out.values


LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=63, min_child_samples=80, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=5.0, n_estimators=1200, verbosity=-1,
)


def cv_oof(df, feats, target, groups, residual_base=None, n_splits=5, seed=0):
    """GroupKFold OOF predictions from LightGBM.

    If residual_base is given, the model learns (target - base) and OOF = base + pred.
    """
    y = target.values.astype(float)
    base = np.zeros(len(df)) if residual_base is None else np.asarray(residual_base, float)
    ytrain = y - base
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    X = feats.values.astype(np.float32)
    for tr, va in gkf.split(X, ytrain, groups):
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(X[tr], ytrain[tr],
              eval_set=[(X[va], ytrain[va])],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[va] = m.predict(X[va])
    return base + oof


def run(df, tw, label="synthetic", n_splits=5, align_kw=None):
    cols = C.COLS
    align_kw = align_kw or {}
    t0 = time.time()
    print(f"\n=== {label}: {df[cols['well']].nunique()} wells, {len(df)} rows ===")

    print("[1] geosteering alignment (ensemble) ...", flush=True)
    al = align_ensemble(df, tw, cols, **align_kw)
    print(f"    align done in {time.time()-t0:.1f}s")

    have_target = cols["target"] in df.columns
    results = {}
    if have_target:
        y = df[cols["target"]]
        # Baseline 0: predict per-well mean is not available at test (no labels);
        # global-mean baseline is the honest naive score.
        results["naive_global_mean"] = rmse(y, np.full(len(y), y.mean()))
        results["align_only"] = rmse(y, al["tvt_align"])

    print("[2] features ...", flush=True)
    feats = build_features(df, cols, al)

    if not HAVE_LGB:
        print("LightGBM unavailable; alignment-only results:")
        for k, v in results.items():
            print(f"    {k:24s} RMSE {v:.3f}")
        return al, feats, results

    groups = df[cols["well"]].values
    if have_target:
        print("[3] LightGBM (direct tvt) ...", flush=True)
        oof_direct = cv_oof(df, feats, y, groups, n_splits=n_splits)
        results["lgbm_direct"] = rmse(y, oof_direct)

        print("[4] LightGBM (residual on alignment) ...", flush=True)
        oof_resid = cv_oof(df, feats, y, groups,
                           residual_base=al["tvt_align"], n_splits=n_splits)
        results["lgbm_residual"] = rmse(y, oof_resid)

        # Blend align + residual model, then smooth.
        blend = 0.5 * al["tvt_align"] + 0.5 * oof_resid
        results["blend_align_resid"] = rmse(y, blend)
        for w in (9, 15, 25):
            sm = smooth_per_well(df, oof_resid, cols, win=w)
            results[f"resid_smooth_{w}"] = rmse(y, sm)

        print(f"\n--- {label} RMSE summary ({time.time()-t0:.1f}s) ---")
        for k, v in sorted(results.items(), key=lambda kv: kv[1]):
            print(f"    {k:24s} {v:.3f}")
    return al, feats, results


def make_submission(train, test, tw, out_path="submission.csv", align_kw=None):
    """Train on full train, predict TVT for test, write Kaggle submission CSV."""
    cols = C.COLS
    align_kw = align_kw or {}
    print("aligning train+test ...", flush=True)
    al_tr = align_ensemble(train, tw, cols, **align_kw)
    al_te = align_ensemble(test, tw, cols, **align_kw)
    f_tr = build_features(train, cols, al_tr)
    f_te = build_features(test, cols, al_te).reindex(columns=f_tr.columns, fill_value=0.0)
    y = train[cols["target"]].values.astype(float)
    base_tr = al_tr["tvt_align"]; base_te = al_te["tvt_align"]
    m = lgb.LGBMRegressor(random_state=0, **LGB_PARAMS)
    m.fit(f_tr.values.astype(np.float32), y - base_tr)
    pred = base_te + m.predict(f_te.values.astype(np.float32))
    pred = smooth_per_well(test, pred, cols, win=15)
    idc = cols["id"] if cols["id"] in test.columns else None
    sub = pd.DataFrame({(idc or "id"): (test[idc] if idc else np.arange(len(test))),
                        cols["target"]: pred})
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(sub)} rows)")
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="path to real CSV dir")
    ap.add_argument("--wells", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    if args.real:
        df, tw = load_real(args.real)
        run(df, tw, label="real", n_splits=args.splits)
    else:
        df, tw = load_synth(n_wells=args.wells, seed=args.seed)
        run(df, tw, label="synthetic", n_splits=args.splits)
