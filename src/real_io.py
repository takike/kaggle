"""Load the real ROGII competition cases and define the heel/toe split.

Each case = <id>__horizontal_well.csv (+ <id>__typewell.csv).
Horizontal columns (test): MD, X, Y, Z, GR, TVT_input
  +train-only: ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA (formation tops), TVT (target)
Typewell columns: TVT, GR (+train: Geology)

Task: TVT is known up to the Prediction Start (PS) point (== TVT_input non-NaN,
a contiguous heel); predict TVT for the toe (TVT_input is NaN).  Metric: pooled
RMSE of (true_TVT - pred_TVT) over predicted points.

Key quantity: the structural offset  C = TVT + Z  is slowly varying along MD.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def list_cases(split_dir):
    hs = sorted(glob.glob(os.path.join(split_dir, "*__horizontal_well.csv")))
    return [os.path.basename(h).split("__")[0] for h in hs]


def load_case(split_dir, cid):
    h = pd.read_csv(os.path.join(split_dir, f"{cid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(split_dir, f"{cid}__typewell.csv"))
    h = h.sort_values("MD").reset_index(drop=True)
    tw = tw.sort_values("TVT").reset_index(drop=True)
    # Clean GR: interpolate the occasional NaN along MD / TVT.
    h["GR"] = h["GR"].interpolate(limit_direction="both")
    tw["GR"] = tw["GR"].interpolate(limit_direction="both")
    return h, tw


def ps_index(h):
    """Row index of the Prediction Start: first row where TVT_input is NaN."""
    isna = h["TVT_input"].isna().values
    if not isna.any():
        return len(h)
    return int(np.argmax(isna))


def split_known_pred(h):
    """Boolean masks for known (heel) and to-predict (toe) rows."""
    known = h["TVT_input"].notna().values
    pred = ~known
    return known, pred
