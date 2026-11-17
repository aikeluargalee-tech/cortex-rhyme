#!/usr/bin/env python3
"""
CORTEX-RHYME — Quantitative Sanity & De-Correlation Test Suite
Verifies causal Z-scores, 72h embargo, 24h de-clustering, thin sample grading,
tail asymmetry math, and Mahalanobis collinearity de-correlation.
"""
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cortex_rhyme  # noqa: E402


class TestCortexRhymeSanity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_cortex.db")
        self.out_json = os.path.join(self.tmp_dir.name, "similar-setups.json")
        self.db_patcher = patch.object(cortex_rhyme, "DB_PATH", self.db_path)
        self.json_patcher = patch.object(cortex_rhyme, "OUT_JSON_PATH", self.out_json)
        self.db_patcher.start()
        self.json_patcher.start()

    def tearDown(self) -> None:
        self.db_patcher.stop()
        self.json_patcher.stop()
        self.tmp_dir.cleanup()

    def _seed_synthetic_db(self, rows: list[dict]) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE hourly_state (
                ts_ms INTEGER PRIMARY KEY,
                iso_utc TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume_ccy REAL NOT NULL,
                mark_close REAL,
                index_close REAL,
                basis_bps REAL,
                funding_rate REAL,
                oi_usd REAL,
                oi_delta_24h REAL,
                taker_buy_ratio REAL,
                gk_vol_24h REAL,
                live_l2_imbalance REAL
            );
        """)
        for r in rows:
            conn.execute("""
                INSERT INTO hourly_state (
                    ts_ms, iso_utc, open, high, low, close, volume_ccy,
                    basis_bps, funding_rate, oi_delta_24h, gk_vol_24h, taker_buy_ratio,
                    live_l2_imbalance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["ts_ms"],
                r["iso_utc"],
                r["close"],
                r["close"] * 1.005,
                r["close"] * 0.995,
                r["close"],
                1000.0,
                r["basis_bps"],
                r["funding_rate"],
                r["oi_delta_24h"],
                r["gk_vol_24h"],
                r["taker_buy_ratio"],
                r.get("live_l2_imbalance", 0.0),
            ))
        conn.commit()
        conn.close()

    def _generate_baseline_series(self, n_hours: int = 600) -> list[dict]:
        start_dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(n_hours):
            dt = start_dt + timedelta(hours=i)
            ts_ms = int(dt.timestamp() * 1000)
            phase = (i % 48) / 48.0
            wave = (phase - 0.5) * 2.0
            ortho = math.sin(i * 0.37)
            price = 70000.0 + (i * 5.0) + (wave * 400.0)
            rows.append({
                "ts_ms": ts_ms,
                "iso_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "close": round(price, 2),
                "funding_rate": 0.0001 + (wave * 0.00005),
                "basis_bps": 5.0 + (wave * 2.0) + (ortho * 0.3),
                "oi_delta_24h": 1.0 + (ortho * 1.5),
                "gk_vol_24h": 0.45 + (abs(wave) * 0.10),
                "taker_buy_ratio": 0.50 + (ortho * 0.03),
            })
        return rows

    def test_01_causal_z_scores_have_zero_lookahead_leakage(self) -> None:
        series_a = [float(i % 17) + (i * 0.05) for i in range(400)]
        series_b = list(series_a)
        for idx in range(250, 400):
            series_b[idx] = 999999.0
        z_a = cortex_rhyme.causal_z_scores(series_a, window=168)
        z_b = cortex_rhyme.causal_z_scores(series_b, window=168)
        for idx in range(250):
            self.assertEqual(z_a[idx], z_b[idx])

    def test_02_declustering_enforces_minimum_24h_spacing(self) -> None:
        rows = self._generate_baseline_series(n_hours=600)
        target_state = {
            "funding_rate": 0.00045,
            "basis_bps": 18.0,
            "oi_delta_24h": 6.5,
            "gk_vol_24h": 0.85,
            "taker_buy_ratio": 0.62,
        }
        for h in range(300, 312):
            rows[h].update(target_state)
        rows[-1].update(target_state)
        self._seed_synthetic_db(rows)
        result = cortex_rhyme.run_engine(k=20, declustering_hours=24, max_distance=1.5)
        matched_times = [
            datetime.strptime(m["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")
            for m in result["top_analogues"]
        ]
        for i in range(len(matched_times)):
            for j in range(i + 1, len(matched_times)):
                gap_hours = abs((matched_times[i] - matched_times[j]).total_seconds()) / 3600.0
                self.assertGreaterEqual(gap_hours, 24.0)

    def test_03_embargo_excludes_recent_72_hours(self) -> None:
        rows = self._generate_baseline_series(n_hours=500)
        extreme_state = {
            "funding_rate": -0.0008,
            "basis_bps": -22.0,
            "oi_delta_24h": -8.5,
            "gk_vol_24h": 1.15,
            "taker_buy_ratio": 0.34,
        }
        rows[len(rows) - 40].update(extreme_state)
        rows[-1].update(extreme_state)
        self._seed_synthetic_db(rows)
        result = cortex_rhyme.run_engine(k=20, embargo_hours=72, max_distance=2.65)
        latest_allowed_dt = datetime.strptime(
            rows[len(rows) - 1 - 72]["iso_utc"], "%Y-%m-%dT%H:%M:%SZ"
        )
        for match in result["top_analogues"]:
            match_dt = datetime.strptime(match["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")
            self.assertLess(match_dt, latest_allowed_dt)

    def test_04_outlier_shock_triggers_thin_sample_grade(self) -> None:
        rows = self._generate_baseline_series(n_hours=600)
        rows[-1].update({
            "funding_rate": -0.0025,
            "basis_bps": -65.0,
            "oi_delta_24h": -18.0,
            "gk_vol_24h": 1.95,
            "taker_buy_ratio": 0.22,
        })
        self._seed_synthetic_db(rows)
        result = cortex_rhyme.run_engine(k=20, max_distance=2.35)
        self.assertLess(result["diagnostics"]["analogues_found"], 10)
        self.assertEqual(result["diagnostics"]["sample_quality"], "thin")

    def test_05_tail_asymmetry_exposes_downside_cliff(self) -> None:
        returns = [0.5, -4.0, 0.4, -3.8, 0.6, -4.2]
        summary = cortex_rhyme.summarize_horizon(returns)
        self.assertEqual(summary["up_share"], 50.0)
        self.assertEqual(summary["median_up_pct"], 0.5)
        self.assertEqual(summary["median_down_pct"], -4.0)
        self.assertEqual(summary["tail_skew_ratio"], 0.12)

    def test_06_mahalanobis_decorrelates_collinear_funding_and_basis(self) -> None:
        pool = []
        for i in range(200):
            x = math.sin(i * 0.2)
            pool.append([x, 0.95 * x + 0.05 * math.cos(i), math.cos(i * 0.3), math.sin(i * 0.5), math.cos(i * 0.7)])
        stats = cortex_rhyme.compute_regularized_precision_matrix(pool, ridge=0.05)
        inv_cov = stats["inv_cov"]
        self.assertGreater(stats["funding_basis_corr"], 0.90)
        origin = [0.0, 0.0, 0.0, 0.0, 0.0]
        comoving_vec = [1.0, 1.0, 0.0, 0.0, 0.0]
        divergent_vec = [1.0, -1.0, 0.0, 0.0, 0.0]
        dist_comoving = cortex_rhyme.mahalanobis_distance(origin, comoving_vec, inv_cov)
        dist_divergent = cortex_rhyme.mahalanobis_distance(origin, divergent_vec, inv_cov)
        self.assertLess(dist_comoving, dist_divergent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
