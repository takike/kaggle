"""
Geosteering log-correlation: align a lateral GR log against the typewell to
recover the structural shift, then derive TVT.

Model
-----
At a lateral sample with true vertical depth ``tvd`` and measured GR ``gr``, the
typewell GR at depth ``tvd - delta`` should match, where ``delta`` is the local
*structural relief* (TVD of the datum marker minus its typewell datum).  ``delta``
varies *slowly* along the well (geology is structurally smooth), whereas the bit's
stratigraphic position TVT changes quickly as the well undulates.

So we search, per along-hole location, the ``delta`` that makes a short window of
the lateral GR best match the typewell GR shape, regularised so ``delta`` changes
smoothly along measured depth.  This is solved exactly with a 1-D dynamic program
(Viterbi) whose transition cost is L1 in ``delta`` -- evaluated in O(states) per
step via an L1 distance transform, so the whole well is fast.

Output per lateral sample:
  * ``delta``       structural relief estimate
  * ``tvt_align``   = (tvd - delta) - DATUM   (calibrated to labels downstream)
  * ``align_cost``  matching residual (low = confident lock)
"""
from __future__ import annotations

import numpy as np

from synth import DATUM


def _l1_distance_transform(cost, lam, step):
    """Generalised distance transform with L1 metric, with backpointers.

    Returns (out, arg) where out[j] = min_k cost[k] + lam*step*|j-k| and
    arg[j] is the argmin index k (used for Viterbi backtracking).
    """
    n = cost.shape[0]
    out = cost.copy()
    arg = np.arange(n)
    pen = lam * step
    # forward pass
    for j in range(1, n):
        cand = out[j - 1] + pen
        if cand < out[j]:
            out[j] = cand
            arg[j] = arg[j - 1]
    # backward pass
    for j in range(n - 2, -1, -1):
        cand = out[j + 1] + pen
        if cand < out[j]:
            out[j] = cand
            arg[j] = arg[j + 1]
    return out, arg


def align_well(md, tvd, gr, tw_tvd, tw_gr,
               stride=20, win_ft=24.0, win_sub=3.0,
               delta_step=0.5, delta_pad=70.0, lam=32.0, datum=DATUM,
               prior_w=8.0, tvt_center=0.0, tvt_scale=25.0):
    """Align one lateral well; returns dict of per-sample arrays.

    Parameters chosen for ~1 ft sampling.  ``lam`` trades matching fidelity for
    structural smoothness (higher = stiffer structure).
    """
    md = np.asarray(md, float)
    tvd = np.asarray(tvd, float)
    gr = np.asarray(gr, float)
    order = np.argsort(tw_tvd)
    tw_tvd = np.asarray(tw_tvd, float)[order]
    tw_gr = np.asarray(tw_gr, float)[order]
    n = len(md)

    # Control points (downsampled along measured depth).
    ctrl = np.arange(0, n, stride)
    if ctrl[-1] != n - 1:
        ctrl = np.append(ctrl, n - 1)

    # Global delta grid.  delta = tvd - tvd_type, and tvt is bounded, so delta
    # sits within +/- delta_pad of (tvd - datum).
    base_lo = (tvd - datum).min() - delta_pad
    base_hi = (tvd - datum).max() + delta_pad
    grid = np.arange(base_lo, base_hi + delta_step, delta_step)
    S = len(grid)

    half = win_ft / 2.0
    sub = max(1, int(round(win_sub)))

    # Emission cost matrix: (n_ctrl, S)
    emis = np.empty((len(ctrl), S))
    for ci, i in enumerate(ctrl):
        lo = np.searchsorted(md, md[i] - half)
        hi = np.searchsorted(md, md[i] + half)
        wj = np.arange(lo, hi, sub)
        if len(wj) < 3:
            wj = np.array([i])
        tvd_w = tvd[wj]
        gr_w = gr[wj]
        # depths into typewell: tvd_w - delta  (W x S)
        depths = tvd_w[:, None] - grid[None, :]
        pred = np.interp(depths.ravel(), tw_tvd, tw_gr).reshape(depths.shape)
        emis[ci] = np.mean((gr_w[:, None] - pred) ** 2, axis=0)
        # Soft zone prior: the bit stays within the target window, so favour the
        # GR cycle whose implied tvt is near the (label-estimated) zone centre.
        # This breaks the quasi-periodic marker ambiguity that otherwise locks a
        # well one cycle off.
        if prior_w:
            tvt_cand = (tvd[i] - grid) - datum
            emis[ci] += prior_w * ((tvt_cand - tvt_center) / tvt_scale) ** 2

    # Viterbi over control points with L1 transition on delta.
    back = np.empty((len(ctrl), S), dtype=int)
    cost = emis[0].copy()
    back[0] = np.arange(S)
    for ci in range(1, len(ctrl)):
        prop, arg = _l1_distance_transform(cost, lam, delta_step)
        cost = emis[ci] + prop
        back[ci] = arg
    # Backtrack.
    path = np.empty(len(ctrl), dtype=int)
    path[-1] = int(np.argmin(cost))
    for ci in range(len(ctrl) - 1, 0, -1):
        path[ci - 1] = back[ci, path[ci]]
    delta_ctrl = grid[path]
    cost_ctrl = emis[np.arange(len(ctrl)), path]

    # Interpolate delta + cost back to every sample.
    delta = np.interp(md, md[ctrl], delta_ctrl)
    align_cost = np.interp(md, md[ctrl], cost_ctrl)
    tvt_align = (tvd - delta) - datum
    return {"delta": delta, "tvt_align": tvt_align, "align_cost": align_cost}


def align_dataframe(df, typewell, cols, **kw):
    """Run alignment per well over a dataframe; returns arrays aligned to df rows."""
    import pandas as pd
    out = {"delta": np.empty(len(df)), "tvt_align": np.empty(len(df)),
           "align_cost": np.empty(len(df))}
    tw_tvd = typewell[cols["tw_tvd"]].to_numpy()
    tw_gr = typewell[cols["tw_gr"]].to_numpy()
    for _, idx in df.groupby(cols["well"]).groups.items():
        idx = np.asarray(idx)
        sub = df.loc[idx]
        o = sub.sort_values(cols["md"]).index
        r = align_well(df.loc[o, cols["md"]].to_numpy(),
                       df.loc[o, cols["tvd"]].to_numpy(),
                       df.loc[o, cols["gr"]].to_numpy(),
                       tw_tvd, tw_gr, **kw)
        for k in out:
            out[k][df.index.get_indexer(o)] = r[k]
    return out
