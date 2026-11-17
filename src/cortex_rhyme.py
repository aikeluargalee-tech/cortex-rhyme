#!/usr/bin/env python3
"""
CORTEX-RHYME — Dual-Lens 5D/3D Mahalanobis State-Space Engine (MOD 63 v1.2.0)
Reads aligned 1H telemetry from data/cortex_rhyme.db and computes:
  1. Lens A (5D_MICROSTRUCTURE): Full 5D Mahalanobis analogues across genuine
     derivatives rows (funding, basis, 24h OI delta, 24h GK vol, taker flow).
  2. Lens B (3D_DEEP_CARRY_VOL): Deep multi-year 3D Mahalanobis analogues
     (funding, basis, 24h GK vol) across the full historical archive (720d+).
  3. Automatic THIN-to-3D arbitration when 5D analogues drop below threshold.
  4. Causal 200D SMA (4,800h) trend-regime conditioning & 24h conditional EV.
"""

import json
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "cortex_rhyme.db")
OUT_JSON_PATH = os.path.join(BASE_DIR, "..", "public", "api", "v1", "similar-setups.json")

FEATURES_5D = [
    "funding_rate",
    "basis_bps",
    "oi_delta_24h",
    "gk_vol_24h",
    "taker_buy_ratio",
]
PRIOR_SCALES_5D = [1.15, 1.0, 1.15, 1.0, 1.10]

FEATURES_3D = [
    "funding_rate",
    "basis_bps",
    "gk_vol_24h",
]
PRIOR_SCALES_3D = [1.20, 1.05, 1.10]

SMA_200D_HOURS = 4800


def causal_z_scores(values: List[float], window: int = 168) -> List[float]:
    """Computes strictly causal trailing Z-scores [i-window, i] winsorized to [-4, +4]."""
    z_out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window)
        slice_vals = values[start : i + 1]
        if len(slice_vals) < 24:
            z_out.append(0.0)
            continue
        mu = statistics.fmean(slice_vals)
        sigma = statistics.stdev(slice_vals)
        raw_z = (values[i] - mu) / sigma if sigma > 1e-12 else 0.0
        z_out.append(max(-4.0, min(4.0, raw_z)))
    return z_out


def compute_causal_sma_regimes(closes: List[float], window: int = SMA_200D_HOURS) -> List[Tuple[float, str]]:
    """
    Computes strictly causal trailing 200D (4,800h) SMA in O(N) time using prefix sums.
    Zero future data ever enters bar i's moving average or regime label.
    """
    n = len(closes)
    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + closes[i]

    regimes: List[Tuple[float, str]] = []
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        sma_val = (prefix[i + 1] - prefix[start]) / count
        label = "ABOVE_200D_SMA" if closes[i] >= sma_val else "BELOW_200D_SMA"
        regimes.append((sma_val, label))
    return regimes


def has_genuine_5d_telemetry(row: Dict[str, Any]) -> bool:
    """
    Distinguishes genuine observed derivatives rows from neutral-filled historical
    CSV backfill rows (where oi_usd is NULL, oi_delta_24h == 0.0, taker_buy_ratio == 0.5).
    """
    if row.get("oi_usd") is not None:
        return True
    if abs(float(row["oi_delta_24h"])) > 1e-9:
        return True
    if abs(float(row["taker_buy_ratio"]) - 0.5) > 1e-9:
        return True
    return False


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
    Computes a regularized inverse covariance matrix (precision matrix) strictly
    over eligible historical vectors prior to the 72h embargo (zero lookahead).
    """
    dim = len(vectors[0]) if vectors else len(FEATURES_5D)
    n_samples = len(vectors)

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
    fb_corr = cov[0][1] / (std_f * std_b) if dim >= 2 else 0.0

    for d in range(dim):
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


def grade_sample_quality(
    n_analogues: int,
    avg_distance: float,
    history_days: float,
    dim: int = 5,
) -> str:
    """Calibrates distance thresholds to dimensionality (E[chi_5] ~ 2.24, E[chi_3] ~ 1.73)."""
    thin_dist = 2.35 if dim == 5 else 1.85
    solid_dist = 1.65 if dim == 5 else 1.25

    if n_analogues < 10 or avg_distance > thin_dist:
        return "thin"
    if n_analogues < 18 or avg_distance > solid_dist or history_days < 90.0:
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


def derive_posture_verdict(
    quality: str,
    summary_24h: Dict[str, Any],
    lift_24h_pts: float,
) -> Tuple[float, str]:
    """Computes 24h conditional expected return and maps it to a MOD 62/63 posture tag."""
    if quality == "thin":
        return 0.0, "OUTLIER_THIN_ZERO_WEIGHT"

    p_up = summary_24h["up_share"] / 100.0
    ev_24h = (p_up * summary_24h["median_up_pct"]) + ((1.0 - p_up) * summary_24h["median_down_pct"])
    ev_rounded = round(ev_24h, 3)

    if ev_rounded >= 0.25 and lift_24h_pts >= 5.0 and summary_24h["tail_skew_ratio"] >= 0.90:
        return ev_rounded, "POSITIVE_SKEW_LOOP_A"
    if ev_rounded <= -0.25 and lift_24h_pts <= -5.0:
        return ev_rounded, "NEGATIVE_SKEW_LOOP_B"
    return ev_rounded, "NEUTRAL_CHOP"


def decluster_select(
    candidates: List[Dict[str, Any]],
    k: int,
    declustering_hours: int,
) -> List[Dict[str, Any]]:
    """Selects top-k closest analogues enforcing a minimum 24h spacing."""
    ordered = sorted(candidates, key=lambda x: x["distance"])
    selected: List[Dict[str, Any]] = []
    for cand in ordered:
        if all(abs(cand["idx"] - s["idx"]) >= declustering_hours for s in selected):
            selected.append(cand)
        if len(selected) == k:
            break
    return selected


def load_aligned_rows(db_path: str) -> List[Dict[str, Any]]:
    """Loads chronological rows where all 5 state features are non-null."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Run seed_and_update.py first.")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT
            ts_ms, iso_utc, close, oi_usd,
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


def evaluate_lens(
    lens_id: str,
    features: List[str],
    scales: List[float],
    rows: List[Dict[str, Any]],
    z_matrix: Dict[str, List[float]],
    sma_regimes: List[Tuple[float, str]],
    eligible_mask: List[bool],
    current_idx: int,
    cutoff_idx: int,
    z_window: int,
    k: int,
    declustering_hours: int,
    embargo_hours: int,
    max_distance: float,
) -> Dict[str, Any]:
    """Evaluates a single state-space lens (5D or 3D) with full & 200D-regime conditioning."""
    dim = len(features)

    def get_vec(idx: int) -> List[float]:
        return [z_matrix[feat][idx] * scale for feat, scale in zip(features, scales)]

    current_vec = get_vec(current_idx)
    current_regime = sma_regimes[current_idx][1]

    eligible_indices = [
        i for i in range(z_window, cutoff_idx) if eligible_mask[i]
    ]
    pool_vectors = [get_vec(i) for i in eligible_indices]
    cov_stats = compute_regularized_precision_matrix(pool_vectors, ridge=0.05)
    inv_cov = cov_stats["inv_cov"]

    baseline_all_ret: Dict[int, List[float]] = {4: [], 24: [], 72: []}
    baseline_regime_ret: Dict[int, List[float]] = {4: [], 24: [], 72: []}
    candidates_all: List[Dict[str, Any]] = []
    candidates_regime: List[Dict[str, Any]] = []

    for i in eligible_indices:
        p0 = float(rows[i]["close"])
        if p0 <= 0:
            continue

        ret_4h = ((float(rows[i + 4]["close"]) - p0) / p0) * 100.0
        ret_24h = ((float(rows[i + 24]["close"]) - p0) / p0) * 100.0
        ret_72h = ((float(rows[i + 72]["close"]) - p0) / p0) * 100.0
        row_regime = sma_regimes[i][1]

        baseline_all_ret[4].append(ret_4h)
        baseline_all_ret[24].append(ret_24h)
        baseline_all_ret[72].append(ret_72h)

        if row_regime == current_regime:
            baseline_regime_ret[4].append(ret_4h)
            baseline_regime_ret[24].append(ret_24h)
            baseline_regime_ret[72].append(ret_72h)

        dist = mahalanobis_distance(current_vec, get_vec(i), inv_cov)
        if dist <= max_distance:
            entry = {
                "idx": i,
                "iso_utc": rows[i]["iso_utc"],
                "close": round(p0, 2),
                "trend_regime": row_regime,
                "distance": dist,
                "ret_4h": ret_4h,
                "ret_24h": ret_24h,
                "ret_72h": ret_72h,
            }
            candidates_all.append(entry)
            if row_regime == current_regime:
                candidates_regime.append(entry)

    selected_all = decluster_select(candidates_all, k=k, declustering_hours=declustering_hours)
    selected_regime = decluster_select(candidates_regime, k=k, declustering_hours=declustering_hours)

    eligible_total_count = sum(1 for flag in eligible_mask if flag)
    history_days = round(eligible_total_count / 24.0, 1)

    n_found = len(selected_all)
    avg_dist = statistics.fmean([s["distance"] for s in selected_all]) if n_found > 0 else 999.0
    quality = grade_sample_quality(n_found, avg_dist, history_days, dim=dim)

    n_regime = len(selected_regime)
    avg_dist_regime = statistics.fmean([s["distance"] for s in selected_regime]) if n_regime > 0 else 999.0
    regime_quality = grade_sample_quality(n_regime, avg_dist_regime, history_days, dim=dim)

    after_all = {
        "4h": summarize_horizon([s["ret_4h"] for s in selected_all]),
        "24h": summarize_horizon([s["ret_24h"] for s in selected_all]),
        "72h": summarize_horizon([s["ret_72h"] for s in selected_all]),
    }
    base_all = {
        "4h": summarize_horizon(baseline_all_ret[4]),
        "24h": summarize_horizon(baseline_all_ret[24]),
        "72h": summarize_horizon(baseline_all_ret[72]),
    }
    edge_all = {
        h: {
            "up_share_lift_pts": round(after_all[h]["up_share"] - base_all[h]["up_share"], 1),
            "median_lift_pct": round(after_all[h]["median_pct"] - base_all[h]["median_pct"], 2),
        }
        for h in ("4h", "24h", "72h")
    }

    after_reg = {
        "4h": summarize_horizon([s["ret_4h"] for s in selected_regime]),
        "24h": summarize_horizon([s["ret_24h"] for s in selected_regime]),
        "72h": summarize_horizon([s["ret_72h"] for s in selected_regime]),
    }
    base_reg = {
        "4h": summarize_horizon(baseline_regime_ret[4]),
        "24h": summarize_horizon(baseline_regime_ret[24]),
        "72h": summarize_horizon(baseline_regime_ret[72]),
    }
    edge_reg = {
        h: {
            "up_share_lift_pts": round(after_reg[h]["up_share"] - base_reg[h]["up_share"], 1),
            "median_lift_pct": round(after_reg[h]["median_pct"] - base_reg[h]["median_pct"], 2),
        }
        for h in ("4h", "24h", "72h")
    }

    ev_24h, posture = derive_posture_verdict(
        quality=quality,
        summary_24h=after_all["24h"],
        lift_24h_pts=edge_all["24h"]["up_share_lift_pts"],
    )

    return {
        "lens_id": lens_id,
        "dimensions": features,
        "diagnostics": {
            "analogues_found": n_found,
            "target_k": k,
            "history_days": history_days,
            "baseline_hours_evaluated": len(baseline_all_ret[24]),
            "sample_quality": quality,
            "avg_vector_distance": round(avg_dist, 3),
            "funding_basis_correlation": cov_stats["funding_basis_corr"],
            "conditional_ev_24h_pct": ev_24h,
            "posture_verdict": posture,
            "regime_analogues_found": n_regime,
            "regime_sample_quality": regime_quality,
            "regime_avg_distance": round(avg_dist_regime, 3),
            "embargo_hours": embargo_hours,
            "declustering_hours": declustering_hours,
        },
        "after_similar_setups": after_all,
        "baseline_all_hours": base_all,
        "edge_vs_baseline": edge_all,
        "regime_matched_setups": after_reg,
        "regime_baseline_hours": base_reg,
        "regime_edge_vs_baseline": edge_reg,
        "top_analogues": [
            {
                "timestamp_utc": s["iso_utc"],
                "btc_price": s["close"],
                "trend_regime": s["trend_regime"],
                "distance": round(s["distance"], 3),
                "ret_4h_pct": round(s["ret_4h"], 2),
                "ret_24h_pct": round(s["ret_24h"], 2),
                "ret_72h_pct": round(s["ret_72h"], 2),
            }
            for s in selected_all[:10]
        ],
    }


def run_engine(
    k: int = 20,
    z_window: int = 168,
    embargo_hours: int = 72,
    declustering_hours: int = 24,
    max_distance: Optional[float] = None,
) -> Dict[str, Any]:
    rows = load_aligned_rows(DB_PATH)
    n_total = len(rows)

    if n_total < (z_window + embargo_hours + 48):
        raise RuntimeError(f"Insufficient aligned rows ({n_total}) for causal window + embargo.")

    z_matrix: Dict[str, List[float]] = {
        feat: causal_z_scores([float(r[feat]) for r in rows], window=z_window)
        for feat in FEATURES_5D
    }
    closes = [float(r["close"]) for r in rows]
    sma_regimes = compute_causal_sma_regimes(closes, window=SMA_200D_HOURS)

    current_idx = n_total - 1
    current_row = rows[current_idx]
    cutoff_idx = current_idx - embargo_hours

    current_sma_200d, current_trend_regime = sma_regimes[current_idx]
    sma_dist_pct = ((closes[current_idx] - current_sma_200d) / current_sma_200d) * 100.0 if current_sma_200d > 0 else 0.0

    mask_5d = [has_genuine_5d_telemetry(r) for r in rows]
    mask_3d = [True] * n_total

    max_dist_5d = max_distance if max_distance is not None else 2.65
    max_dist_3d = min(max_distance, 2.05) if max_distance is not None else 2.05

    lens_5d = evaluate_lens(
        lens_id="5D_MICROSTRUCTURE",
        features=FEATURES_5D,
        scales=PRIOR_SCALES_5D,
        rows=rows,
        z_matrix=z_matrix,
        sma_regimes=sma_regimes,
        eligible_mask=mask_5d,
        current_idx=current_idx,
        cutoff_idx=cutoff_idx,
        z_window=z_window,
        k=k,
        declustering_hours=declustering_hours,
        embargo_hours=embargo_hours,
        max_distance=max_dist_5d,
    )

    lens_3d = evaluate_lens(
        lens_id="3D_DEEP_CARRY_VOL",
        features=FEATURES_3D,
        scales=PRIOR_SCALES_3D,
        rows=rows,
        z_matrix=z_matrix,
        sma_regimes=sma_regimes,
        eligible_mask=mask_3d,
        current_idx=current_idx,
        cutoff_idx=cutoff_idx,
        z_window=z_window,
        k=k,
        declustering_hours=declustering_hours,
        embargo_hours=embargo_hours,
        max_distance=max_dist_3d,
    )

    q5 = lens_5d["diagnostics"]["sample_quality"]
    q3 = lens_3d["diagnostics"]["sample_quality"]
    n5 = lens_5d["diagnostics"]["analogues_found"]
    n3 = lens_3d["diagnostics"]["analogues_found"]

    fallback_triggered = False
    fallback_reason: Optional[str] = None
    active_lens = lens_5d

    if q5 == "thin" and (q3 in ("solid", "moderate") or n3 > n5):
        active_lens = lens_3d
        fallback_triggered = True
        fallback_reason = (
            f"5D lens graded THIN (n={n5}); promoted {lens_3d['diagnostics']['history_days']}d "
            f"3D Carry-Vol lens (grade={q3.upper()}, n={n3})."
        )

    active_diag = dict(active_lens["diagnostics"])
    active_diag["active_lens"] = active_lens["lens_id"]
    active_diag["fallback_triggered"] = fallback_triggered
    active_diag["fallback_reason"] = fallback_reason

    output_payload: Dict[str, Any] = {
        "meta": {
            "engine": "CORTEX-RHYME",
            "version": "1.2.0",
            "active_lens": active_lens["lens_id"],
            "distance_metric": "regularized_mahalanobis_dual_lens",
            "instrument": "BTC-USDT-SWAP",
            "status": "live",
            "as_of_utc": current_row["iso_utc"],
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "history_sources": ["OKX Public v5 Keyless Archive"],
            "disclaimer": "Observational historical statistics only. Not a prediction or financial advice.",
        },
        "current_state": {
            "btc_price": round(float(current_row["close"]), 2),
            "regime_context": {
                "trend_regime": current_trend_regime,
                "sma_200d": round(current_sma_200d, 2),
                "distance_from_sma_200d_pct": round(sma_dist_pct, 2),
            },
            "raw_metrics": {
                "funding_rate_pct": round(float(current_row["funding_rate"]) * 100.0, 4),
                "basis_bps": round(float(current_row["basis_bps"]), 2),
                "oi_delta_24h_pct": round(float(current_row["oi_delta_24h"]), 2),
                "gk_vol_24h_ann_pct": round(float(current_row["gk_vol_24h"]) * 100.0, 2),
                "taker_buy_ratio": round(float(current_row["taker_buy_ratio"]), 4),
                "live_l2_imbalance": current_row["live_l2_imbalance"],
            },
            "z_vector": {
                feat: round(z_matrix[feat][current_idx], 3) for feat in FEATURES_5D
            },
        },
        "diagnostics": active_diag,
        "after_similar_setups": active_lens["after_similar_setups"],
        "baseline_all_hours": active_lens["baseline_all_hours"],
        "edge_vs_baseline": active_lens["edge_vs_baseline"],
        "regime_matched_setups": active_lens["regime_matched_setups"],
        "regime_baseline_hours": active_lens["regime_baseline_hours"],
        "regime_edge_vs_baseline": active_lens["regime_edge_vs_baseline"],
        "top_analogues": active_lens["top_analogues"],
        "lenses": {
            "5D_MICROSTRUCTURE": lens_5d,
            "3D_DEEP_CARRY_VOL": lens_3d,
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(
        f"[OK] Generated {OUT_JSON_PATH} | Active: {active_lens['lens_id']} | "
        f"Trend: {current_trend_regime} | Quality: {active_diag['sample_quality'].upper()} "
        f"(n={active_diag['analogues_found']}, avg_dist={active_diag['avg_vector_distance']}) | "
        f"24h EV: {active_diag['conditional_ev_24h_pct']}% ({active_diag['posture_verdict']})"
    )
    return output_payload


if __name__ == "__main__":
    run_engine()
