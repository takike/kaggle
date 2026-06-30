# Offset-well dip prior

The task slides state geological **dip behaves similarly in neighbouring wells**.
We exploit this — but carefully.

## What does NOT work
The structural offset `C = TVT + Z` is **not** a shared spatial surface: wells are
landed in different stratigraphic zones and use different typewell datums, so the
*absolute* `C` differs even at the same (X, Y). Interpolating `C` from neighbours
(kNN, even with a per-well constant offset) is far worse than the heel trend:

| prior | toe-`C` RMSE |
|---|---|
| heel linear trend | ~31 |
| offset-well kNN of `C` (calibrated) | ~98 |

## What works: borrow the *dip gradient*, not the absolute level
Estimate the shared spatial gradient `(∂C/∂X, ∂C/∂Y)` from neighbouring wells via a
regression over a 3000-ft neighbourhood:

```
C ~ a·X + b·Y + (one offset dummy per well)
```

The per-well dummies absorb each well's zone/datum offset, leaving the **shared dip
`(a, b)`**. Propagate it from the known heel anchor along the toe's actual X/Y path:

```
trend(i) = C_PS + a·(X_i − X_PS) + b·(Y_i − Y_PS)
```

This prior alone is ~2× better than the heel's own 1-D dip:

| prior | toe-`C` RMSE |
|---|---|
| heel linear trend | ~31 |
| **offset-well dip gradient** | **~14** |

## Effect on the full pipeline (leave-one-well-out, 250 cases)

Final toe-RMSE after the GR aligner, swapping only the prior:

| prior | pooled | per-case median | p90 | max | dip fallbacks |
|---|---|---|---|---|---|
| heel trend | 20.07 | 6.67 | 30.41 | 91.91 | — |
| **offset-well dip (adaptive radius)** | **12.55** | **5.09** | **21.69** | **51.00** | 0 / 250 |

A ~37% pooled reduction — the single largest improvement in the project — and it
helps the whole distribution (median, p90 and max all drop; the worst case falls
from 92 to 51). The **adaptive search radius** (3k→5k→8k→13k ft) removed all
fallbacks (was 30/250 at a fixed 3 k ft) for a small extra gain (12.59→12.55,
median 5.16→5.09). (150-case run agreed: 18.09 → 11.88.)

This is now the **default** prior in `predict_case` (`trend_override`) and
`submit_real.py` (`--no-offset` to disable). Leave-one-well-out excludes the target;
for the real hidden test the cloud is built from all train wells.

Reproduce: `python experiments/validate_offset.py 150`

## Tuning round — what did NOT help (honest negatives)

After adopting the planar dip prior, two further levers were tested (150 cases,
leave-one-well-out):

- **Aligner params already optimal.** Sweeping `lam ∈ {40,70,100}` × `e_pad ∈
  {80,120,160,200}` on the dip prior: `lam=100` is best at every `e_pad`, and `e_pad`
  is insensitive at `lam=100` (pooled 11.08–11.17). Lower `lam` (looser structure)
  is consistently worse. Current defaults stand. (`experiments/tune_aligner.py`)
- **Quadratic dip surface is worse, not better.** `order=2` (add X², Y², XY curvature
  terms) overfits and *diverges on extrapolation*: the toe extends well beyond the
  fitted neighbourhood, where a 2nd-order surface blows up, while a plane extrapolates
  stably. prior-`C` RMSE 28.8 vs 13.2, final pooled 18.97 vs 11.84. Planar `order=1`
  is the sweet spot. (`experiments/test_quadratic.py`)

Takeaway: the solution is at a stable optimum — planar offset-well dip prior +
`lam=100` aligner (+ optional `α≈0.3` residual). Added model complexity does not help.
