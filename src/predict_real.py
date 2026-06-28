"""Per-case geosteering predictor for the real ROGII data.

For each case: predict TVT on the toe (after the Prediction Start point) by
aligning the horizontal GR onto a reference GR(TVT) curve, tracking the smooth
structural offset C = TVT + Z with a Viterbi DP anchored to the known heel.

Reference GR(TVT) is built from:
  * the typewell GR(TVT), affinely calibrated to the horizontal GR scale using the
    heel overlap (the two logs come from different tools / scales), and
  * the heel's own GR(TVT) where it covers (higher resolution, same tool).
"""
from __future__ import annotations

import numpy as np

from align import _l1_distance_transform
from real_io import split_known_pred


def _bin_curve(tvt, gr, grid):
    """Average GR into TVT grid bins (heel TVT is non-monotonic)."""
    out = np.full(len(grid), np.nan)
    if len(tvt) == 0:
        return out
    step = grid[1] - grid[0]
    idx = np.round((tvt - grid[0]) / step).astype(int)
    ok = (idx >= 0) & (idx < len(grid))
    idx = idx[ok]; g = gr[ok]
    if len(idx) == 0:
        return out
    sums = np.bincount(idx, weights=g, minlength=len(grid))
    cnts = np.bincount(idx, minlength=len(grid))
    nz = cnts > 0
    out[nz] = sums[nz] / cnts[nz]
    return out


def build_reference(h, tw, known, grid_step=0.5, heel_weight=0.7):
    tvt_k = h["TVT_input"].values[known]
    gr_k = h["GR"].values[known]
    tw_tvt = tw["TVT"].values
    tw_gr = tw["GR"].values

    lo = min(np.nanmin(tvt_k), tw_tvt.min()) - 5
    hi = max(np.nanmax(tvt_k), tw_tvt.max()) + 5
    grid = np.arange(lo, hi + grid_step, grid_step)

    # Calibrate typewell GR -> horizontal GR scale on the heel TVT overlap.
    o = (tw_tvt >= np.nanmin(tvt_k)) & (tw_tvt <= np.nanmax(tvt_k))
    if o.sum() > 20:
        tw_at_heel = np.interp(tvt_k, tw_tvt, tw_gr)
        a, b = np.polyfit(tw_at_heel, gr_k, 1)
        if not np.isfinite(a) or a <= 0:
            a, b = np.nanstd(gr_k) / (np.nanstd(tw_gr) + 1e-9), 0.0
    else:
        a = np.nanstd(gr_k) / (np.nanstd(tw_gr) + 1e-9)
        b = np.nanmean(gr_k) - a * np.nanmean(tw_gr)
    ref_tw = np.interp(grid, tw_tvt, a * tw_gr + b)

    ref_heel = _bin_curve(tvt_k, gr_k, grid)
    # light fill of small heel gaps
    s = np.where(np.isnan(ref_heel))[0]
    ref = ref_tw.copy()
    hv = ~np.isnan(ref_heel)
    ref[hv] = heel_weight * ref_heel[hv] + (1 - heel_weight) * ref_tw[hv]
    return grid, ref


def predict_case(h, tw, stride=15, win_ft=34.0, win_sub=3.0, lam=100.0,
                 e_pad=200.0, c_step=0.5, anchor_ft=120.0, heel_weight=0.7,
                 shape_norm=False):
    """Return predicted TVT for every row of h (meaningful on toe rows)."""
    known, pred = split_known_pred(h)
    MD = h["MD"].values.astype(float)
    Z = h["Z"].values.astype(float)
    GR = h["GR"].values.astype(float)
    n = len(h)
    ps = int(np.argmax(pred)) if pred.any() else n
    if ps >= n:
        return h["TVT_input"].values.copy()

    grid_ref, ref = build_reference(h, tw, known, heel_weight=heel_weight)

    C_true_known = (h["TVT_input"].values + Z)  # known C at heel
    C_ps = C_true_known[ps - 1]

    # Structural drift rate r (dC/dMD) from the tail of the heel -> trend prior.
    tail = np.where(known)[0]
    tail = tail[tail >= max(0, ps - 500)]
    if len(tail) > 20:
        r = np.polyfit(MD[tail], C_true_known[tail], 1)[0]
    else:
        r = 0.0
    r = float(np.clip(r, -0.5, 0.5))

    # States = deviation e of C from the heel trend line (centred, bounded), so
    # the search never clips no matter how far C drifts.  trend[i] = C_ps + r*dMD.
    trend = C_ps + r * (MD - MD[ps - 1])
    egrid = np.arange(-e_pad, e_pad + c_step, c_step)
    S = len(egrid)

    # Control points: a stretch of heel (for anchoring) + the whole toe.
    start = max(0, ps - int(anchor_ft))
    ctrl = list(range(start, n, stride))
    if ctrl[-1] != n - 1:
        ctrl.append(n - 1)
    ctrl = np.array(ctrl)

    half = win_ft / 2.0
    sub = max(1, int(round(win_sub)))
    emis = np.empty((len(ctrl), S))
    ANCHOR = 1e6
    for ci, i in enumerate(ctrl):
        if known[i]:
            # Pin deviation to the known structural offset here.
            emis[ci] = ANCHOR * (egrid - (C_true_known[i] - trend[i])) ** 2
            continue
        lo = np.searchsorted(MD, MD[i] - half)
        hi = np.searchsorted(MD, MD[i] + half)
        wj = np.arange(lo, hi, sub)
        if len(wj) < 3:
            wj = np.array([i])
        # C = trend[j] + e ;  tvt_j = C - Z_j  (trend varies across the window)
        tvt = (trend[wj] - Z[wj])[:, None] + egrid[None, :]   # W x S
        predgr = np.interp(tvt.ravel(), grid_ref, ref).reshape(tvt.shape)
        gw = GR[wj][:, None]
        if shape_norm:        # match GR *signature* (shape), not absolute level
            gw = gw - gw.mean()
            predgr = predgr - predgr.mean(axis=0, keepdims=True)
        emis[ci] = np.mean((gw - predgr) ** 2, axis=0)

    # Viterbi with L1 transition on C.
    back = np.empty((len(ctrl), S), dtype=int)
    cost = emis[0].copy()
    back[0] = np.arange(S)
    for ci in range(1, len(ctrl)):
        prop, arg = _l1_distance_transform(cost, lam, c_step)
        cost = emis[ci] + prop
        back[ci] = arg
    path = np.empty(len(ctrl), dtype=int)
    path[-1] = int(np.argmin(cost))
    for ci in range(len(ctrl) - 1, 0, -1):
        path[ci - 1] = back[ci, path[ci]]
    e_ctrl = egrid[path]

    C_full = trend + np.interp(MD, MD[ctrl], e_ctrl)
    tvt_pred = C_full - Z
    # Keep known rows exact.
    out = h["TVT_input"].values.copy()
    out[pred] = tvt_pred[pred]
    return out
