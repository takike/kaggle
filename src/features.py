"""Feature engineering for TVT refinement.

The geosteering aligner (align.py) already produces a strong ``tvt_align``.  These
features give a gradient-boosted model the context to (a) refine that estimate and
(b) recover where alignment is locally unreliable: trajectory geometry, multi-scale
GR texture, and alignment confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _roll(s: pd.Series, win: int, fn: str):
    r = s.rolling(win, center=True, min_periods=max(2, win // 4))
    return getattr(r, fn)()


def build_features(df: pd.DataFrame, cols: dict, align: dict) -> pd.DataFrame:
    """Return a feature frame aligned row-for-row with ``df``.

    ``align`` is the dict from align_dataframe (delta, tvt_align, align_cost).
    """
    g = pd.DataFrame(index=df.index)
    well = cols["well"]
    md, tvd, gr, x = cols["md"], cols["tvd"], cols["gr"], cols["x"]

    g["tvt_align"] = align["tvt_align"]
    g["delta"] = align["delta"]
    g["align_cost"] = align["align_cost"]
    g["log_align_cost"] = np.log1p(align["align_cost"])
    if "tvt_align_std" in align:        # ensemble disagreement = uncertainty
        g["tvt_align_std"] = align["tvt_align_std"]

    g[tvd] = df[tvd].values
    g[gr] = df[gr].values
    if cols["inc"] in df.columns:
        g["inc"] = df[cols["inc"]].values

    parts = []
    for wid, idx in df.groupby(well, sort=False).groups.items():
        idx = np.asarray(idx)
        sub = df.loc[idx].sort_values(md)
        oi = sub.index
        f = pd.DataFrame(index=oi)

        tvd_s = sub[tvd]
        gr_s = sub[gr]
        md_s = sub[md]

        # Trajectory geometry.
        dmd = md_s.diff().replace(0, np.nan)
        f["dtvd_dmd"] = (tvd_s.diff() / dmd).fillna(0.0)
        f["curv"] = f["dtvd_dmd"].diff().fillna(0.0)
        f["tvd_centered"] = tvd_s - tvd_s.mean()          # per-well, label-free
        f["x_frac"] = (sub[x] - sub[x].min()) / max(1.0, (sub[x].max() - sub[x].min()))
        f["md_frac"] = (md_s - md_s.min()) / max(1.0, (md_s.max() - md_s.min()))

        # Multi-scale GR texture.
        for w in (7, 21, 61, 151):
            f[f"gr_mean_{w}"] = _roll(gr_s, w, "mean")
            f[f"gr_std_{w}"] = _roll(gr_s, w, "std")
        f["gr_grad"] = gr_s.diff().fillna(0.0)
        f["gr_rank"] = gr_s.rank(pct=True)                 # GR percentile within well
        f["gr_minus_mean61"] = gr_s - f["gr_mean_61"]

        # Alignment context (smoothness / local variability of the estimate).
        ta = pd.Series(align["tvt_align"], index=df.index).loc[oi]
        f["tvt_align_grad"] = ta.diff().fillna(0.0)
        f["tvt_align_std61"] = _roll(ta, 61, "std")
        ac = pd.Series(align["align_cost"], index=df.index).loc[oi]
        f["align_cost_mean61"] = _roll(ac, 61, "mean")

        parts.append(f)

    feats = pd.concat(parts).reindex(df.index)
    out = pd.concat([g, feats], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
