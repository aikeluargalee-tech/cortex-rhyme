# CORTEX-RHYME

**Open-Source Empirical State-Space Conditioning for Bitcoin Market Microstructure (MOD 63)**

[![Hourly Sync](https://github.com/aikeluargalee-tech/cortex-rhyme/actions/workflows/hourly_sync.yml/badge.svg)](https://github.com/aikeluargalee-tech/cortex-rhyme/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-stdlib%20only-green.svg)]()

**CORTEX-RHYME** answers one empirical question every hour:
> *"When Bitcoin's perpetual funding, basis, open interest velocity, Garman-Klass volatility, and order-book flow looked statistically identical to right now, how did forward returns over +4h, +24h, and +72h behave compared to an unconditional random hour?"*

Built with **zero external Python dependencies**, hosted at **$0/month** via GitHub Actions + GitHub Pages, and accessible to both human traders and AI agents via [`SKILL.md`](./SKILL.md).

## Free Public Endpoints (No API Keys or Wallets)
- **Live Web HUD:** `https://aikeluargalee-tech.github.io/cortex-rhyme/`
- **Static JSON API:** `https://aikeluargalee-tech.github.io/cortex-rhyme/api/v1/similar-setups.json`
- **Agent Skill Spec:** [`SKILL.md`](./SKILL.md)

## Why CORTEX-RHYME Differs From Naive Analogue Tools
1. **Strict Causal Normalization:** Every feature is standardized via trailing 168-hour (7-day) Z-scores `[t-168, t]`—zero future volatility leakage.
2. **Regularized 5D Mahalanobis Distance:** Uses pure-Python Gauss-Jordan matrix inversion with Tikhonov ridge regularization (`λ = 0.05`) so collinear signals (like perpetual funding and basis) are never double-counted.
3. **24-Hour De-Clustering:** Enforces a hard 24-hour minimum spacing between selected historical analogues so a single trending weekend cannot masquerade as 20 independent setups.
4. **72-Hour Forward Embargo:** Excludes the trailing 72 hours from candidate selection so every match has finalized +4h, +24h, and +72h outcomes.
5. **Tail-Asymmetry Auditing:** Reports `median_up_pct` vs. `median_down_pct` (`tail_skew_ratio`) alongside `baseline_all_hours` win-rate lift, exposing asymmetric liquidation cliffs even when win rate sits near 50%.

## Quickstart
```bash
# 1. Run quantitative sanity tests (zero dependencies)
python3 -m unittest discover -s tests -p "test_*.py" -v

# 2. Seed trailing 60 days from keyless OKX public v5 endpoints
python3 src/seed_and_update.py

# 3. (Optional) Bulk backfill 2+ years of OKX historical CSV/ZIP archives
python3 src/seed_and_update.py --import-csv data/raw_csv

# 4. Compute analogues and output public/api/v1/similar-setups.json
python3 src/cortex_rhyme.py
```

## Disclaimer
All metrics published by CORTEX-RHYME are **observational historical statistics** intended strictly as a base-rate prior for expectancy and risk sizing—never a standalone buy/sell signal or financial advice.
