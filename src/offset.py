"""Offset-well structural prior.

The structural offset C = TVT + Z is NOT a shared spatial surface across wells
(they are landed in different stratigraphic zones / use different typewell datums,
so absolute C differs even at the same X/Y).  But the *dip* — the spatial gradient
(dC/dX, dC/dY) — IS shared by neighbouring wells (the task slides: "geological dips
behave similarly in neighbouring wells").

So we estimate the local dip gradient from neighbouring wells via a regression
`C ~ a*X + b*Y + (per-well offset dummies)` over a spatial neighbourhood — the
dummies absorb the per-well zone/datum offsets, leaving the shared (a, b).  We then
build the toe trend by propagating from the known heel anchor along the toe's actual
X/Y path:  trend(i) = C_PS + a*(X_i - X_PS) + b*(Y_i - Y_PS).

This prior is ~2x better than extrapolating the heel's own 1-D dip (toe-C RMSE ~14
vs ~31) and is fed to the aligner as `trend_override`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from real_io import list_cases, load_case, split_known_pred


class StructuralCloud:
    """Spatial index of (X, Y, C) points from a set of (train) wells."""

    def __init__(self, X, Y, C, case_id):
        self.X = np.asarray(X); self.Y = np.asarray(Y)
        self.C = np.asarray(C); self.case = np.asarray(case_id)
        self.tree = cKDTree(np.c_[self.X, self.Y])

    @classmethod
    def from_dir(cls, train_dir, cases=None, stride=20):
        cases = cases or list_cases(train_dir)
        Xs, Ys, Cs, ids = [], [], [], []
        for cid in cases:
            h, _ = load_case(train_dir, cid)
            s = slice(None, None, stride)
            Xs.append(h["X"].values[s]); Ys.append(h["Y"].values[s])
            Cs.append((h["TVT"].values + h["Z"].values)[s])
            ids.append(np.full(np.shape(h["X"].values[s]), hash(cid) & 0xffffffff))
        return cls(np.concatenate(Xs), np.concatenate(Ys),
                   np.concatenate(Cs), np.concatenate(ids))

    def regional_dip(self, cx, cy, exclude_case=None,
                     radii=(3000.0, 5000.0, 8000.0, 13000.0),
                     min_wells=4, min_pts=50, maxpts=4000):
        """Shared dip gradient (a, b) around (cx, cy).

        Expands the search radius until enough distinct neighbouring wells are
        found, so wells in sparser parts of the field still get a dip prior (rather
        than falling back to the heel trend).  Returns (a, b, R_used) or None.
        """
        for R in radii:
            idx = np.asarray(self.tree.query_ball_point([cx, cy], R))
            if len(idx) == 0:
                continue
            if exclude_case is not None:
                idx = idx[self.case[idx] != exclude_case]
            if len(idx) >= min_pts and len(np.unique(self.case[idx])) >= min_wells:
                break
        else:
            return None
        if len(idx) > maxpts:
            idx = np.random.default_rng(0).choice(idx, maxpts, replace=False)
        xs, ys, cs, ws = self.X[idx], self.Y[idx], self.C[idx], self.case[idx]
        uw, inv = np.unique(ws, return_inverse=True)
        D = np.zeros((len(idx), 2 + len(uw)))
        D[:, 0] = xs - cx; D[:, 1] = ys - cy
        D[np.arange(len(idx)), 2 + inv] = 1.0
        coef, *_ = np.linalg.lstsq(D, cs, rcond=None)
        return float(coef[0]), float(coef[1]), float(R)


def dip_trend(h, cloud, exclude_case=None, **dip_kw):
    """Full-length trend array from the offset-well dip prior, or None to fall back."""
    known, pred = split_known_pred(h)
    if not pred.any():
        return None
    ps = int(np.argmax(pred))
    X = h["X"].values; Y = h["Y"].values; Z = h["Z"].values
    C_ps = (h["TVT_input"].values + Z)[ps - 1]
    cx, cy = X[pred].mean(), Y[pred].mean()
    dip = cloud.regional_dip(cx, cy, exclude_case=exclude_case, **dip_kw)
    if dip is None:
        return None
    a, b, _R = dip
    return C_ps + a * (X - X[ps - 1]) + b * (Y - Y[ps - 1])
