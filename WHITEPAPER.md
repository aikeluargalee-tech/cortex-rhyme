# CORTEX-RHYME: Empirical State-Space Conditioning for Bitcoin Market Microstructure

**Version:** 1.1.0 | **License:** MIT (Open-Source Public Release)  
**CoreTex Internal Designation:** `MOD 63 — Historical Regime Analogue & Expectancy Engine`

## Abstract
Spot price levels are non-stationary across macro cycles, whereas derivatives leverage mechanics, carry cost, volatility clustering, and order-book liquidity exhibit recurring structural behavior. **CORTEX-RHYME** is a zero-dependency state-space conditioning engine designed to answer one empirical question: *When Bitcoin's perpetual funding, basis, 24h open interest delta, Garman-Klass volatility, and order-book flow looked statistically identical to the present hour, how did forward return distributions over +4h, +24h, and +72h deviate from unconditional background noise (`baseline_all_hours`)?*

Crucially, **CORTEX-RHYME** is engineered not as a standalone directional oracle, but as a Layer 1 Base-Rate Prior for expectancy and position sizing governed by strict de-clustering, look-ahead embargoes, and tail-asymmetry auditing.

## State-Space Dimensions
At every hourly close $t$, CORTEX-RHYME constructs a 5D state vector:
$$X_t = [z_{\text{funding}}(t), z_{\text{basis}}(t), z_{\text{OI}}(t), z_{\text{vol}}(t), z_{\text{flow}}(t)]$$

| Dimension | Metric | Mechanism | Source (Keyless OKX v5) |
| :--- | :--- | :--- | :--- |
| 1. Carry Stress | `funding_rate` | Crowding cost & leverage imbalance | `/api/v5/public/funding-rate-history` |
| 2. Term Structure | `basis_bps` | Perp mark vs. spot index premium/discount | `/api/v5/market/history-mark-price-candles` vs. index |
| 3. Positioning | `oi_delta_24h` | Net 24h leverage buildup / unwinding | `/api/v5/rubik/stat/contracts/open-interest-history` |
| 4. Turbulence | `gk_vol_24h` | Annualized 24h Garman-Klass volatility | `/api/v5/market/history-candles` (1H) |
| 5. Flow / Liquidity | `taker_buy_ratio` | Taker buy/sell aggression (+ live L2 skew) | `/api/v5/rubik/stat/taker-volume` & `/api/v5/market/books` |

## Mathematical & Statistical Safeguards
1. **Strict Causal Normalization:** Every feature is standardized strictly against its trailing 168-hour window `[t-168, t]`—zero future leakage.
2. **Regularized 5D Mahalanobis Distance:** Uses Gauss-Jordan matrix inversion with partial pivoting and Tikhonov ridge regularization (`lambda = 0.05`) estimated strictly prior to the 72h forward embargo, eliminating double-counting between collinear funding and basis signals.
3. **24-Hour Minimum De-Clustering:** Candidate matches are ranked by ascending Mahalanobis distance and admitted only if `|i - j| >= 24 hours` for all already-selected analogues.
4. **72-Hour Forward Embargo:** Excludes the trailing 72 hours so every candidate possesses a finalized forward return horizon.
5. **Honest Sample Quality Grading:** Calibrated for 5D Mahalanobis ($\sqrt{5} \approx 2.24$): `thin` (`N < 10` or `avg_dist > 2.35`), `moderate` (`N < 18` or `avg_dist > 1.65` or `history_days < 90`), and `solid` otherwise.
6. **Tail-Asymmetry Auditing:** Separates `median_up_pct` from `median_down_pct` (`tail_skew_ratio`) alongside baseline-adjusted win-rate lift (`up_share_lift_pts`).

## Disclaimer
Observational historical statistics only. Not a price prediction or financial advice.
