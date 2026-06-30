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

    @staticmethod
    def _poly_terms(dx, dy, order):
        """Polynomial basis (no intercept; that is the per-well dummy)."""
        cols = [dx, dy]
        if order >= 2:
            cols += [dx * dx, dy * dy, dx * dy]
        return np.column_stack(cols)

    def regional_fit(self, cx, cy, exclude_case=None, order=1,
                     radii=(3000.0, 5000.0, 8000.0, 13000.0),
                     min_wells=4, min_pts=50, maxpts=4000):
        """Fit the shared structural surface C ~ poly(X-cx, Y-cy) + per-well offset.

        Expands the search radius until enough distinct neighbouring wells are found.
        The per-well offset dummies absorb each well's zone/datum offset, leaving the
        shared polynomial coefficients.  ``order=1`` -> planar dip (a,b);
        ``order=2`` -> + curvature (a,b,c,d,e).  Returns (coefs, order) or None.
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
        P = self._poly_terms(xs - cx, ys - cy, order)
        uw, inv = np.unique(ws, return_inverse=True)
        D = np.zeros((len(idx), P.shape[1] + len(uw)))
        D[:, :P.shape[1]] = P
        D[np.arange(len(idx)), P.shape[1] + inv] = 1.0
        coef, *_ = np.linalg.lstsq(D, cs, rcond=None)
        return coef[:P.shape[1]], order

    def regional_dip(self, cx, cy, **kw):
        """Backward-compatible planar dip: returns (a, b) or None."""
        r = self.regional_fit(cx, cy, order=1, **kw)
        return (float(r[0][0]), float(r[0][1])) if r is not None else None, float(R)


def dip_trend(h, cloud, exclude_case=None, order=1, **dip_kw):
    """Full-length trend array from the offset-well dip prior, or None to fall back.

    Propagates the shared structural surface from the known heel anchor along the
    toe's actual X/Y path:  trend(i) = C_PS + [surface(X_i,Y_i) - surface(X_PS,Y_PS)].
    """
    known, pred = split_known_pred(h)
    if not pred.any():
        return None
    ps = int(np.argmax(pred))
    X = h["X"].values; Y = h["Y"].values; Z = h["Z"].values
    C_ps = (h["TVT_input"].values + Z)[ps - 1]
    cx, cy = X[pred].mean(), Y[pred].mean()
    fit = cloud.regional_fit(cx, cy, exclude_case=exclude_case, order=order, **dip_kw)
    if fit is None:
        return None
    coef, order = fit
    surf = lambda px, py: cloud._poly_terms(px - cx, py - cy, order) @ coef  # noqa: E731
    return C_ps + (surf(X, Y) - surf(X[ps - 1], Y[ps - 1]))
