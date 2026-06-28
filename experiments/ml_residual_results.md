# LightGBM residual + confidence correction — explored, not adopted

Following the two proposed "close the gap" directions:
1. confidence-aware correction, and
2. a LightGBM residual over all cases (GroupKFold by case),

I built `src/real_ml.py`: per toe row, learn `true_TVT − align_pred` from 19
**relative/confidence** features (distance past PS, GR texture, the Viterbi path
cost, GR-match residual, deviation instability, distance to typewell coverage
edges, …), GroupKFold by case, then `final = align + α·oof`.

## Result (200 cases, 971,862 toe rows, GroupKFold OOF)

| α (ML weight) | pooled toe-RMSE | per-case median | per-case p90 |
|---|---|---|---|
| 0.00 (alignment only) | 16.341 | **6.28** | 26.83 |
| 0.15 | 16.209 | 6.70 | 27.13 |
| 0.30 | **16.162** | 6.75 | 27.28 |
| 0.50 | 16.230 | 7.41 | 27.76 |
| 0.80 | 16.607 | 8.14 | 27.72 |
| 1.00 | 17.034 | 8.88 | 27.19 |

## Conclusion — honest negative result

The ML residual buys at best a **~1% pooled improvement (16.34 → 16.16)** and only
by **degrading the median case (6.28 → 6.75+)**. There is no α that improves both.

Why: on the ~80% of cases the aligner already locks well, the residual is
near-zero noise, so a global model injects error into good cases faster than it
fixes the idiosyncratic tail. The first attempt was even worse because absolute
coordinates (`Z`, `trend`) topped the importance and overfit per-case depth ranges;
restricting to relative/confidence features removed that but still did not yield a
reliable gain.

**Decision:** keep the geosteering aligner as the production prediction (best
median, competitive pooled). `real_ml.py` is retained as a reproducible experiment.

The genuinely promising remaining lever is **offset wells** (the task explicitly
notes neighbouring wells share structural dip): use X/Y to borrow the toe dip from
nearby wells instead of extrapolating the heel trend alone — a better *prior* for
the aligner, which is more likely to help the tail than a post-hoc residual.

Reproduce: `python src/real_ml.py 200`  (or `0` for all 773 cases).
