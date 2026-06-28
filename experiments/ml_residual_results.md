# LightGBM residual + confidence correction

Following the two proposed "close the gap" directions:
1. confidence-aware correction, and
2. a LightGBM residual over all cases (GroupKFold by case),

`src/real_ml.py`: per toe row, learn `true_TVT − align_pred` from 19
**relative/confidence** features (distance past PS, GR texture, the Viterbi path
cost, GR-match residual, deviation instability, distance to typewell coverage
edges, …), GroupKFold by case, then `final = align + α·oof`.

## Results (GroupKFold OOF)

**Full data — all 773 cases, 3,783,989 toe rows:**

| α (ML weight) | pooled toe-RMSE | per-case median | per-case p90 |
|---|---|---|---|
| 0.00 (alignment only) | 17.721 | 7.20 | 29.99 |
| **0.15** | **17.458** | **7.03** | **29.09** |
| 0.30 | 17.311 | 7.32 | 28.46 |
| 0.50 | 17.303 | 8.03 | 28.28 |
| 1.00 | 18.196 | 10.16 | 28.17 |

**Small sample — 200 cases, 971,862 rows** (for contrast):

| α | pooled | median |
|---|---|---|
| 0.00 | 16.341 | 6.28 |
| 0.30 | 16.162 | 6.75 |
| 1.00 | 17.034 | 8.88 |

## Conclusion

The benefit **depends on training-set size**:
- With 200 cases, any ML weight degrades the median (the model overfits the
  idiosyncratic tail and injects noise into the ~80% well-aligned cases).
- With all 773 cases, a **light blend `α≈0.15` improves every metric** — pooled
  17.72→17.46, median 7.20→7.03, p90 29.99→29.09 — a small (~1.5% pooled) but
  consistent gain. Heavier weights (α≥0.5) still hurt the median.

So the residual is a **mild, data-hungry refinement**, not a major lever: the
geosteering aligner remains the dominant component (it does ~95% of the work,
43.5→17.7), and the ML shaves a little more once enough wells are available.

**Decision:** keep the aligner as the core; expose the light residual blend as an
**opt-in final stage** (`submit_real.py --ml --alpha 0.15`). An early version that
included absolute coordinates (`Z`, `trend`) overfit per-case depth ranges and hurt
everything; restricting to relative/confidence features was necessary.

The largest remaining lever is **offset wells** (the task notes neighbouring wells
share structural dip): borrow the *toe* dip from nearby wells via X/Y as a better
prior for the aligner, rather than extrapolating the heel trend alone.

Reproduce: `python src/real_ml.py 0`  (full)  or  `python src/real_ml.py 200`.
