#!/usr/bin/env python3
"""
CORTEX-RHYME — Step 2: Regularized 5D Mahalanobis State-Space Engine (MOD 63)
Reads aligned 1H telemetry from data/cortex_rhyme.db, computes causal 168h
rolling Z-scores, estimates a strictly causal regularized 5x5 covariance
matrix, and ranks historical analogues via Mahalanobis distance.
"""
import json
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "cortex_rhyme.db")
OUT_JSON_PATH = os.path.join(BASE_DIR, "..", "public", "api", "v1", "similar-setups.json")

FEATURES = [
    "funding_rate",
    "basis_bps",
    "oi_delta_24h",
    "gk_vol_24h",
    "taker_buy_ratio",
]
PRIOR_SCALES = [1.15, 1.0, 1.15, 1.0, 1.10]


def causal_z_scores(values: List[float], window: int = 168) -> List[float]:
    """Computes strictly causal trailing Z-scores [i-window, i] with zero lookahead."""
    z_out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window)
        slice_vals = values[start : i + 1]
        if len(slice_vals) < 24:
            z_out.append(0.0)
            continue
        mu = statistics.mean(slice_vals)
        sigma = statistics.stdev(slice_vals)
        raw_z = (values[i] - mu) / sigma if sigma > 1e-12 else 0.0
        z_out.append(max(-4.0, min(4.0, raw_z)))
    return z_out


def invert_matrix_gauss_jordan(matrix: List[List[float]]) -> List[List[float]]:
    """Inverts an NxN square matrix using Gauss-Jordan elimination with partial pivoting."""
    n = len(matrix)
    aug = [
        [float(matrix[r][c]) for c in range(n)] + [1.0 if r == c else 0.0 for c in range(n)]
        for r in range(n)
    ]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix encountered during Mahalanobis inversion.")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [val / pivot_val for val in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if factor != 0.0:
                    aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def compute_regularized_precision_matrix(
    vectors: List[List[float]],
    ridge: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes the regularized inverse covariance matrix (precision matrix)
    strictly over historical candidate vectors (zero lookahead).
    """
    n_samples = len(vectors)
    dim = len(FEATURES)
    if n_samples < 30:
        inv_cov = [[1.0 if r == c else 0.0 for c in range(dim)] for r in range(dim)]
        return {"inv_cov": inv_cov, "funding_basis_corr": 0.0}

    means = [sum(vec[d] for vec in vectors) / n_samples for d in range(dim)]
    cov = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for vec in vectors:
        diffs = [vec[d] - means[d] for d in range(dim)]
        for r in range(dim):
            for c in range(r, dim):
                cov[r][c] += diffs[r] * diffs[c]

    denom = max(n_samples - 1, 1)
    for r in range(dim):
        for c in range(r, dim):
            val = cov[r][c] / denom
            cov[r][c] = val
            cov[c][r] = val

    std_f = math.sqrt(max(cov[0][0], 1e-12))
    std_b = math.sqrt(max(cov[1][1], 1e-12))
    fb_corr = cov[0][1] / (std_f * std_b)

    for d in range(dim):
        # Prevent zero-padded historical backfill columns from collapsing variance and hyper-amplifying inv_cov
        cov[d][d] = max(cov[d][d], 1.0) + ridge

    inv_cov = invert_matrix_gauss_jordan(cov)
    return {
        "inv_cov": inv_cov,
        "funding_basis_corr": round(fb_corr, 3),
    }


def mahalanobis_distance(
    vec_a: List[float],
    vec_b: List[float],
    inv_cov: List[List[float]],
) -> float:
    """Computes sqrt((a - b)^T * Sigma^-1 * (a - b))."""
    diff = [a - b for a, b in zip(vec_a, vec_b)]
    dim = len(diff)
    quad_form = 0.0
    for r in range(dim):
        row_sum = sum(inv_cov[r][c] * diff[c] for c in range(dim))
        quad_form += diff[r] * row_sum
    return math.sqrt(max(quad_form, 0.0))


def grade_sample_quality(n_analogues: int, avg_distance: float, history_days: float) -> str:
    """Honest quality grading calibrated for 5D Mahalanobis distance."""
    if n_analogues < 10 or avg_distance > 2.35:
        return "thin"
    if n_analogues < 18 or avg_distance > 1.65 or history_days < 90.0:
        return "moderate"
    return "solid"


def summarize_horizon(returns: List[float]) -> Dict[str, Any]:
    """Computes directional win rate, median return, and upside/downside tail asymmetry."""
    if not returns:
        return {
            "median_pct": 0.0,
            "up_share": 0.0,
            "median_up_pct": 0.0,
            "median_down_pct": 0.0,
            "tail_skew_ratio": 0.0,
        }
    ups = [r for r in returns if r > 0]
    downs = [r for r in returns if r < 0]
    med_up = statistics.median(ups) if ups else 0.0
    med_down = statistics.median(downs) if downs else 0.0
    skew_ratio = round(abs(med_up / med_down), 2) if abs(med_down) > 1e-9 else 0.0
    return {
        "median_pct": round(statistics.median(returns), 2),
        "up_share": round((len(ups) / len(returns)) * 100.0, 1),
        "median_up_pct": round(med_up, 2),
        "median_down_pct": round(med_down, 2),
        "tail_skew_ratio": skew_ratio,
    }


def load_aligned_rows(db_path: str) -> List[Dict[str, Any]]:
    """Loads chronological rows where all 5 state features are non-null."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Run seed_and_update.py first.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT
            ts_ms, iso_utc, close,
            funding_rate, basis_bps, oi_delta_24h, gk_vol_24h, taker_buy_ratio,
            live_l2_imbalance
        FROM hourly_state
        WHERE funding_rate IS NOT NULL
          AND basis_bps IS NOT NULL
          AND oi_delta_24h IS NOT NULL
          AND gk_vol_24h IS NOT NULL
          AND taker_buy_ratio IS NOT NULL
        ORDER BY ts_ms ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def run_engine(
    k: int = 20,
    z_window: int = 168,
    embargo_hours: int = 72,
    declustering_hours: int = 24,
    max_distance: float = 2.65,
) -> Dict[str, Any]:
    rows = load_aligned_rows(DB_PATH)
    n_total = len(rows)
    if n_total < (z_window + embargo_hours + 48):
        raise RuntimeError(f"Insufficient aligned rows ({n_total}) for causal window + embargo.")

    z_matrix: Dict[str, List[float]] = {
        feat: causal_z_scores([float(r[feat]) for r in rows], window=z_window)
        for feat in FEATURES
    }

    def get_scaled_vector(idx: int) -> List[float]:
        return [
            z_matrix[feat][idx] * scale
            for feat, scale in zip(FEATURES, PRIOR_SCALES)
        ]

    current_idx = n_total - 1
    current_row = rows[current_idx]
    current_vec = get_scaled_vector(current_idx)

    cutoff_idx = current_idx - embargo_hours
    historical_pool_vectors = [get_scaled_vector(i) for i in range(z_window, cutoff_idx)]
    cov_stats = compute_regularized_precision_matrix(historical_pool_vectors, ridge=0.05)
    inv_cov = cov_stats["inv_cov"]

    baseline_returns: Dict[int, List[float]] = {4: [], 24: [], 72: []}
    candidates: List[Dict[str, Any]] = []

    for i in range(z_window, cutoff_idx):
        p0 = float(rows[i]["close"])
        if p0 <= 0:
            continue
        ret_4h = ((float(rows[i + 4]["close"]) - p0) / p0) * 100.0
        ret_24h = ((float(rows[i + 24]["close"]) - p0) / p0) * 100.0
        ret_72h = ((float(rows[i + 72]["close"]) - p0) / p0) * 100.0
        baseline_returns[4].append(ret_4h)
        baseline_returns[24].append(ret_24h)
        baseline_returns[72].append(ret_72h)

        hist_vec = get_scaled_vector(i)
        dist = mahalanobis_distance(current_vec, hist_vec, inv_cov)
        if dist <= max_distance:
            candidates.append({
                "idx": i,
                "iso_utc": rows[i]["iso_utc"],
                "close": round(p0, 2),
                "distance": dist,
                "ret_4h": ret_4h,
                "ret_24h": ret_24h,
                "ret_72h": ret_72h,
            })

    candidates.sort(key=lambda x: x["distance"])
    selected: List[Dict[str, Any]] = []
    for cand in candidates:
        if all(abs(cand["idx"] - s["idx"]) >= declustering_hours for s in selected):
            selected.append(cand)
        if len(selected) == k:
            break

    n_found = len(selected)
    history_days = round(n_total / 24.0, 1)
    avg_dist = statistics.mean([s["distance"] for s in selected]) if n_found > 0 else 999.0
    quality = grade_sample_quality(n_found, avg_dist, history_days)

    after_setups = {
        "4h": summarize_horizon([s["ret_4h"] for s in selected]),
        "24h": summarize_horizon([s["ret_24h"] for s in selected]),
        "72h": summarize_horizon([s["ret_72h"] for s in selected]),
    }
    baseline_all = {
        "4h": summarize_horizon(baseline_returns[4]),
        "24h": summarize_horizon(baseline_returns[24]),
        "72h": summarize_horizon(baseline_returns[72]),
    }

    edge_vs_baseline = {}
    for horizon in ("4h", "24h", "72h"):
        edge_vs_baseline[horizon] = {
            "up_share_lift_pts": round(
                after_setups[horizon]["up_share"] - baseline_all[horizon]["up_share"], 1
            ),
            "median_lift_pct": round(
                after_setups[horizon]["median_pct"] - baseline_all[horizon]["median_pct"], 2
            ),
        }

    output_payload: Dict[str, Any] = {
        "meta": {
            "engine": "CORTEX-RHYME",
            "version": "1.1.0",
            "distance_metric": "regularized_mahalanobis_5d",
            "instrument": "BTC-USDT-SWAP",
            "status": "live",
            "as_of_utc": current_row["iso_utc"],
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "history_sources": ["OKX Public v5 Keyless Archive"],
            "disclaimer": "Observational historical statistics only. Not a prediction or financial advice.",
        },
        "current_state": {
            "btc_price": round(float(current_row["close"]), 2),
            "raw_metrics": {
                "funding_rate_pct": round(float(current_row["funding_rate"]) * 100.0, 4),
                "basis_bps": round(float(current_row["basis_bps"]), 2),
                "oi_delta_24h_pct": round(float(current_row["oi_delta_24h"]), 2),
                "gk_vol_24h_ann_pct": round(float(current_row["gk_vol_24h"]) * 100.0, 2),
                "taker_buy_ratio": round(float(current_row["taker_buy_ratio"]), 4),
                "live_l2_imbalance": current_row["live_l2_imbalance"],
            },
            "z_vector": {
                feat: round(z_matrix[feat][current_idx], 3) for feat in FEATURES
            },
        },
        "diagnostics": {
            "analogues_found": n_found,
            "target_k": k,
            "history_days": history_days,
            "baseline_hours_evaluated": len(baseline_returns[24]),
            "sample_quality": quality,
            "avg_vector_distance": round(avg_dist, 3),
            "funding_basis_correlation": cov_stats["funding_basis_corr"],
            "embargo_hours": embargo_hours,
            "declustering_hours": declustering_hours,
        },
        "after_similar_setups": after_setups,
        "baseline_all_hours": baseline_all,
        "edge_vs_baseline": edge_vs_baseline,
        "top_analogues": [
            {
                "timestamp_utc": s["iso_utc"],
                "btc_price": s["close"],
                "distance": round(s["distance"], 3),
                "ret_4h_pct": round(s["ret_4h"], 2),
                "ret_24h_pct": round(s["ret_24h"], 2),
                "ret_72h_pct": round(s["ret_72h"], 2),
            }
            for s in selected[:10]
        ],
    }

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(
        f"[OK] Generated {OUT_JSON_PATH} | Mahalanobis Dist | "
        f"Funding-Basis Corr: {cov_stats['funding_basis_corr']} | "
        f"Quality: {quality.upper()} (n={n_found}, avg_dist={avg_dist:.2f})"
    )
    return output_payload


if __name__ == "__main__":
    run_engine()
