# Ablation — held-out-wells RMSE (GroupKFold OOF)

Synthetic geosteering data, 14 wells, averaged over seeds [42, 1, 7, 100, 5].

| Stage | mean RMSE | per-seed |
|---|---|---|
| naive: global mean | **16.592** | 15.11, 17.62, 16.83, 16.58, 16.82 |
| align: single, no prior | **4.034** | 6.39, 2.68, 4.50, 3.23, 3.36 |
| align: single + prior | **4.791** | 4.62, 3.20, 7.98, 3.84, 4.32 |
| align: ensemble | **3.364** | 4.40, 2.01, 3.86, 3.17, 3.38 |
| ML: LGBM direct | **4.340** | 4.88, 3.15, 4.90, 4.12, 4.65 |
| ML: ensemble + LGBM residual | **3.250** | 4.25, 1.90, 3.67, 3.11, 3.33 |
| FINAL: + along-hole smoothing | **3.248** | 4.25, 1.90, 3.66, 3.10, 3.33 |
