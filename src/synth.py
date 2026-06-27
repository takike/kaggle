"""
Physics-faithful synthetic data generator for the ROGII Wellbore Geology
Prediction competition (geosteering / TVT estimation).

Why this exists
---------------
This sandbox has no Kaggle credentials, so the real competition CSVs cannot be
downloaded.  To still develop and *validate* a real solution end-to-end, we
simulate the same forward model that produces the real data:

  * A 1-D stratigraphic Gamma-Ray (GR) template `gr_template(s)` describing the
    GR signature of the rock column as a function of stratigraphic position `s`
    (measured vertically from a datum marker).  Low GR ~ clean sand/carbonate,
    high GR ~ shale.
  * A vertical *typewell*: a reference well that logged GR vs TVD.  Because it is
    vertical, its GR(TVD) is exactly the template shifted by the datum depth D:
        gr_type(tvd) = gr_template(tvd - D)
  * A *lateral* (near-horizontal) well that wanders up and down through the rock.
    At measured depth MD it has a horizontal displacement x(MD) and a true
    vertical depth tvd(MD).  The local structural surface z0(x) (TVD of the datum
    marker at horizontal position x) tilts/folds/faults across the field.
  * The bit's stratigraphic position (the TARGET) is
        tvt(MD) = tvd(MD) - z0(x(MD))
    and the measured lateral GR is the template read at that position:
        gr_lat(MD) = gr_template(tvt(MD)) + noise = gr_type(tvt(MD) + D) + noise

The inference task (what the solution must invert): given gr_lat, tvd, x and the
typewell gr_type(tvd), recover tvt.  Equivalently, find the slowly-varying
structural relief delta(x) = z0(x) - D such that the lateral GR aligns with the
typewell, then tvt = (tvd - D) - delta.

The generated tables mirror a typical Kaggle geosteering schema.  Real column
names may differ slightly; `config.COLS` maps logical names -> real names so the
rest of the pipeline is data-source agnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATUM = 8000.0  # TVD (ft) of the datum marker at the typewell (tvt = 0 there)


def make_gr_template(rng: np.random.Generator, s_min=-120.0, s_max=120.0,
                     step=0.25):
    """Build a stratigraphic GR template gr_template(s) on a fine grid.

    Combines blocky layers (steps) with finer wiggles so the curve has both
    coarse markers and high-frequency texture an aligner can lock onto.
    """
    s = np.arange(s_min, s_max + step, step)
    gr = np.full_like(s, 70.0)

    # Blocky layering: a handful of beds with characteristic GR levels.
    n_layers = rng.integers(10, 16)
    bounds = np.sort(rng.uniform(s_min, s_max, n_layers - 1))
    levels = rng.uniform(25, 145, n_layers)  # sand(low) .. shale(high)
    idx = np.searchsorted(bounds, s)
    gr = levels[idx]

    # Smooth the hard steps a little (transition zones).
    gr = _smooth(gr, win=9)

    # Add multi-scale wiggles for texture (distinctive correlation features).
    for _ in range(6):
        amp = rng.uniform(3, 12)
        wl = rng.uniform(2, 25)  # wavelength in ft
        ph = rng.uniform(0, 2 * np.pi)
        gr = gr + amp * np.sin(2 * np.pi * s / wl + ph)

    gr = np.clip(gr, 8, 165)
    return s, gr


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return a
    k = np.ones(win) / win
    return np.convolve(a, k, mode="same")


def _smooth_random_walk(rng, n, scale, smooth_win):
    """A bounded, smooth 1-D path (used for tvt path and structure)."""
    steps = rng.normal(0, 1, n)
    path = np.cumsum(steps)
    path = _smooth(path, smooth_win)
    path = path - path.mean()
    sd = path.std()
    if sd > 0:
        path = path / sd
    return path * scale


def gen_well(well_id, rng, gr_s, gr_template, length_ft=None):
    """Generate one lateral well's table of per-foot samples."""
    if length_ft is None:
        length_ft = rng.integers(4000, 9000)
    n = int(length_ft)  # ~1 ft sampling
    md = np.arange(n, dtype=float) + rng.uniform(6000, 9000)

    # Horizontal displacement: lateral advances ~1 ft per ft of MD.
    x = np.cumsum(np.full(n, 1.0) * rng.uniform(0.96, 1.0)) + rng.uniform(0, 500)

    # Structural surface z0(x): regional dip + folds (+ optional fault).
    dip = rng.uniform(-0.03, 0.03)            # ft TVD per ft horizontal
    fold = _smooth_random_walk(rng, n, scale=rng.uniform(8, 25), smooth_win=601)
    z0 = DATUM + dip * (x - x[0]) + fold
    if rng.random() < 0.5:  # throw a fault somewhere
        fpos = rng.integers(int(0.3 * n), int(0.8 * n))
        z0[fpos:] += rng.uniform(-18, 18)

    # The driller's actual stratigraphic path = TARGET tvt (kept in a zone).
    tvt = _smooth_random_walk(rng, n, scale=rng.uniform(10, 22), smooth_win=351)
    tvt += rng.uniform(-8, 8)  # offset within the target window

    tvd = z0 + tvt
    # Inclination from trajectory geometry (~90 deg, horizontal).
    dtvd = np.gradient(tvd)
    dmd = np.gradient(md)
    inc = 90.0 - np.degrees(np.arcsin(np.clip(dtvd / np.maximum(dmd, 1e-6), -1, 1)))

    # Measured lateral GR = template at tvt + measurement noise.
    gr_lat = np.interp(tvt, gr_s, gr_template) + rng.normal(0, 4.0, n)
    gr_lat = np.clip(gr_lat, 5, 175)

    return pd.DataFrame({
        "well_id": well_id,
        "md": md,
        "tvd": tvd,
        "x": x,
        "inc": inc,
        "gr": gr_lat,
        "tvt": tvt,           # target (dropped for test)
        "_z0": z0,            # ground-truth structure (debug only)
    })


def make_typewell(gr_s, gr_template):
    """Typewell GR vs TVD (vertical reference).  gr_type(tvd)=template(tvd-D)."""
    tvd = gr_s + DATUM
    return pd.DataFrame({"tvd": tvd, "gr": gr_template})


def generate(n_wells=14, seed=42):
    """Generate a full synthetic dataset: typewell + many lateral wells."""
    rng = np.random.default_rng(seed)
    gr_s, gr_template = make_gr_template(rng)
    typewell = make_typewell(gr_s, gr_template)
    wells = [gen_well(f"W{ i:03d}".replace(" ", ""), rng, gr_s, gr_template)
             for i in range(n_wells)]
    df = pd.concat(wells, ignore_index=True)
    return df, typewell


if __name__ == "__main__":
    df, tw = generate()
    print("wells:", df.well_id.nunique(), "rows:", len(df))
    print(df.groupby("well_id").size().describe())
    print("tvt stats:", df.tvt.describe()[["min", "mean", "max", "std"]].to_dict())
    print("gr stats:", df.gr.describe()[["min", "mean", "max"]].to_dict())
    print("typewell rows:", len(tw), "gr range",
          round(tw.gr.min(), 1), round(tw.gr.max(), 1))
