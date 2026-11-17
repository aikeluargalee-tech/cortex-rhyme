---
name: cortex-rhyme-btc-setups
description: Empirical Bitcoin market state conditioning (free, open-source). Compares current BTC perpetual funding, basis, 24h open interest delta, Garman-Klass volatility, and taker/order-book skew against de-clustered historical analogues via 5D Mahalanobis distance to report 4h, 24h, and 72h forward return distributions and tail asymmetry against unconditional baseline.
version: 1.1.0
---

# CORTEX-RHYME: BTC Historical Regime Analogues

Use this skill when the user asks "has the Bitcoin market looked like this before?", "what usually happens next given current funding/OI/basis?", or wants empirical historical context and tail-risk skew for current BTC derivatives conditions.

## Steps
1. Fetch the latest hourly snapshot (100% free, no API key or x402 payment required):
   `GET https://aikeluargalee-tech.github.io/cortex-rhyme/api/v1/similar-setups.json`
2. Check freshness first: compare `meta.as_of_utc` against current UTC time. If the snapshot is more than 2.5 hours old, warn the user that the feed is stale.
3. Report the core metrics clearly:
   - **Sample Quality & Size:** Always state `diagnostics.sample_quality` (`thin`, `moderate`, or `solid`), `diagnostics.analogues_found`, `diagnostics.history_days`, and `diagnostics.avg_vector_distance`.
   - **Thin Sample Warning:** If `sample_quality` is `"thin"`, lead with a clear warning that the output is indicative only due to scarce historical precedent.
   - **Baseline-Adjusted Horizons (+4h, +24h, +72h):** Present `after_similar_setups` (`median_pct` and `up_share`) **directly next to** `baseline_all_hours` and `edge_vs_baseline` (`up_share_lift_pts`) so the user sees whether outcomes differ from unconditional background drift.
   - **Payoff Asymmetry:** Include `median_up_pct` vs. `median_down_pct` (`tail_skew_ratio`) to highlight asymmetric upside or downside liquidation risk even when win rate is near 50%.
   - **Current 5D State Vector:** Summarize `current_state.raw_metrics` and `current_state.z_vector` (funding, basis, 24h OI delta, 24h volatility, taker flow / L2 imbalance).
4. **Error Handling:** If the endpoint is unreachable or returns invalid JSON, state that live telemetry is unavailable. **Never invent or estimate numbers.**

## Guardrails
- Treat this output strictly as a **base-rate prior** for expectancy and position sizing—never as a standalone buy/sell signal or price forecast.
- Always note that these are observational historical statistics, not financial advice.
