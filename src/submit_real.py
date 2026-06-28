"""Generate a submission for the real ROGII competition.

For each test case, predict TVT on the toe (rows where TVT_input is NaN) with the
geosteering aligner, and emit rows in the required `<caseid>_<rowindex>` id format.

Usage:
    python src/submit_real.py --data data_real --split test --out submission.csv

This is a Code Competition (hidden test); the same per-case function is what a
Kaggle inference notebook would call.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(__file__))
from real_io import list_cases, load_case, split_known_pred  # noqa: E402
from predict_real import predict_case  # noqa: E402


def build_submission(data_dir, split="test", out="submission.csv", verbose=True,
                     ml=False, alpha=0.15, train_split="train", **predict_kw):
    sdir = os.path.join(data_dir, split)
    cases = list_cases(sdir)

    res_model = feat_cols = None
    if ml:  # opt-in: train the light residual model on the train split
        from real_ml import fit_residual_model
        if verbose:
            print("training residual model on train split ...", flush=True)
        res_model, feat_cols = fit_residual_model(
            os.path.join(data_dir, train_split), **predict_kw)

    ids, tvts = [], []
    t0 = time.time()
    for k, cid in enumerate(cases):
        h, tw = load_case(sdir, cid)
        known, pred = split_known_pred(h)
        if ml:
            pr, diag = predict_case(h, tw, return_diag=True, **predict_kw)
            if diag is not None:
                from real_ml import predict_residual
                pr = pr + alpha * predict_residual(res_model, feat_cols, h, diag)
        else:
            pr = predict_case(h, tw, **predict_kw)
        pidx = np.where(pred)[0]
        for j in pidx:
            ids.append(f"{cid}_{j}")
            tvts.append(pr[j])
        if verbose and (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(cases)} cases  {time.time()-t0:.0f}s", flush=True)
    sub = pd.DataFrame({"id": ids, "tvt": tvts})
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {len(sub)} rows from {len(cases)} cases "
          f"({time.time()-t0:.0f}s)")
    return sub


def align_submission_ids(sub, sample_path):
    """Reorder/format to match sample_submission ids exactly (safety)."""
    samp = pd.read_csv(sample_path)
    merged = samp[["id"]].merge(sub, on="id", how="left")
    missing = merged["tvt"].isna().sum()
    if missing:
        print(f"WARNING: {missing} sample ids not predicted; filling 0")
        merged["tvt"] = merged["tvt"].fillna(0.0)
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--ml", action="store_true",
                    help="add the light LightGBM residual blend (trains on train split)")
    ap.add_argument("--alpha", type=float, default=0.15)
    args = ap.parse_args()
    sub = build_submission(args.data, args.split, args.out, ml=args.ml, alpha=args.alpha)
    samp = os.path.join(args.data, "sample_submission.csv")
    if os.path.exists(samp):
        merged = align_submission_ids(sub, samp)
        merged.to_csv(args.out, index=False)
        print(f"aligned to sample_submission: {len(merged)} rows")
        print(merged.head(3).to_string())
