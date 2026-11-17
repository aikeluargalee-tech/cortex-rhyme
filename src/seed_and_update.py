#!/usr/bin/env python3
"""
CORTEX-RHYME — Step 1: Zero-Dependency Keyless OKX Telemetry Collector
Fetches aligned 1H BTC-USDT-SWAP & BTC-USDT spot index telemetry into SQLite:
  1. Swap OHLCV (for returns & Garman-Klass realized volatility)
  2. Mark vs. Index Price (for Perp-to-Spot Basis in bps)
  3. Funding Rate History (forward-filled across 8h intervals)
  4. Open Interest History (for 24h OI % delta)
  5. Taker Buy/Sell Volume (historical order-flow aggression proxy)
  6. Live Top-20 L2 Order Book Skew (captured on each hourly snapshot)
"""
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASE_URL = "https://www.okx.com"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cortex_rhyme.db")
HEADERS = {
    "User-Agent": "CORTEX-RHYME-OpenTelemetry/1.1",
    "Accept": "application/json",
}


def http_get_json(path: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Dict[str, Any]:
    """Resilient GET request with exponential backoff for rate limits."""
    url = f"{BASE_URL}{path}"
    if params:
        clean_params = {k: v for k, v in params.items() if v is not None}
        url = f"{url}?{urllib.parse.urlencode(clean_params)}"
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("code") == "0":
                    return payload
                raise RuntimeError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
            if attempt == retries - 1:
                raise
            sleep_sec = 1.5 * (2 ** attempt)
            print(f"[WARN] Retry {attempt + 1}/{retries} for {path} ({exc}) — sleeping {sleep_sec:.1f}s")
            time.sleep(sleep_sec)
    return {"data": []}


def init_db(db_path: str) -> sqlite3.Connection:
    """Initializes the SQLite archive with an indexed hourly_state table."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hourly_state (
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
    conn.commit()
    return conn


def floor_to_hour_ms(ts_ms: int) -> int:
    return (ts_ms // 3_600_000) * 3_600_000


def fetch_paginated_candles(endpoint: str, inst_id: str, target_bars: int = 1440) -> Dict[int, List[float]]:
    """Fetches historical 1H candles backwards using `after` cursor."""
    records: Dict[int, List[float]] = {}
    after_cursor: Optional[str] = None
    while len(records) < target_bars:
        res = http_get_json(endpoint, {
            "instId": inst_id,
            "bar": "1H",
            "limit": "100",
            "after": after_cursor,
        })
        rows = res.get("data", [])
        if not rows:
            break
        for r in rows:
            ts = floor_to_hour_ms(int(r[0]))
            vals = [float(r[1]), float(r[2]), float(r[3]), float(r[4])]
            if len(r) > 6:
                vals.append(float(r[6]))
            records[ts] = vals
        oldest_ts = int(rows[-1][0])
        if after_cursor == str(oldest_ts):
            break
        after_cursor = str(oldest_ts)
        time.sleep(0.22)
    return records


def fetch_funding_history(inst_id: str = "BTC-USDT-SWAP", pages: int = 4) -> Dict[int, float]:
    """Fetches realized 8h funding rates (up to ~3 months)."""
    funding_map: Dict[int, float] = {}
    after_cursor: Optional[str] = None
    for _ in range(pages):
        res = http_get_json("/api/v5/public/funding-rate-history", {
            "instId": inst_id,
            "limit": "100",
            "after": after_cursor,
        })
        rows = res.get("data", [])
        if not rows:
            break
        for r in rows:
            ts = floor_to_hour_ms(int(r["fundingTime"]))
            funding_map[ts] = float(r["realizedRate"] or r["fundingRate"])
        after_cursor = rows[-1]["fundingTime"]
        time.sleep(0.22)
    return funding_map


def fetch_open_interest_history(target_bars: int = 1440) -> Dict[int, float]:
    """Fetches hourly Open Interest in USD (max 1,440 bars on OKX)."""
    oi_map: Dict[int, float] = {}
    end_cursor: Optional[str] = None
    while len(oi_map) < target_bars:
        res = http_get_json("/api/v5/rubik/stat/contracts/open-interest-history", {
            "instId": "BTC-USDT-SWAP",
            "period": "1H",
            "end": end_cursor,
        })
        rows = res.get("data", [])
        if not rows:
            break
        for r in rows:
            ts = floor_to_hour_ms(int(r[0]))
            oi_map[ts] = float(r[3])
        oldest_ts = int(rows[-1][0])
        if end_cursor == str(oldest_ts):
            break
        end_cursor = str(oldest_ts)
        time.sleep(0.25)
    return oi_map


def fetch_taker_ratio_history(target_bars: int = 1440) -> Dict[int, float]:
    """Fetches hourly taker buy/sell volume ratio for BTC."""
    taker_map: Dict[int, float] = {}
    end_cursor: Optional[str] = None
    while len(taker_map) < target_bars:
        res = http_get_json("/api/v5/rubik/stat/taker-volume", {
            "ccy": "BTC",
            "instType": "CONTRACTS",
            "period": "1H",
            "end": end_cursor,
        })
        rows = res.get("data", [])
        if not rows:
            break
        for r in rows:
            ts = floor_to_hour_ms(int(r[0]))
            sell_vol = float(r[1])
            buy_vol = float(r[2])
            total = buy_vol + sell_vol
            taker_map[ts] = (buy_vol / total) if total > 0 else 0.5
        oldest_ts = int(rows[-1][0])
        if end_cursor == str(oldest_ts):
            break
        end_cursor = str(oldest_ts)
        time.sleep(0.25)
    return taker_map


def fetch_live_l2_imbalance(inst_id: str = "BTC-USDT-SWAP", depth: int = 20) -> Optional[float]:
    """Captures top-20 level order-book notional imbalance [-1.0, +1.0]."""
    try:
        res = http_get_json("/api/v5/market/books", {"instId": inst_id, "sz": str(depth)})
        book = res.get("data", [{}])[0]
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        bid_notional = sum(float(p) * float(sz) for p, sz, *_ in bids)
        ask_notional = sum(float(p) * float(sz) for p, sz, *_ in asks)
        total = bid_notional + ask_notional
        return round((bid_notional - ask_notional) / total, 4) if total > 0 else 0.0
    except Exception as exc:
        print(f"[WARN] Live L2 snapshot skipped: {exc}")
        return None


def compute_garman_klass_24h(ohlc_window: List[List[float]]) -> float:
    """Computes annualized Garman-Klass volatility from trailing 24 hourly bars."""
    if len(ohlc_window) < 2:
        return 0.0
    sum_gk = 0.0
    c1 = 0.5
    c2 = 2.0 * math.log(2.0) - 1.0
    for o, h, l, c, *_ in ohlc_window:
        if o <= 0 or l <= 0 or h <= 0 or c <= 0:
            continue
        log_hl = math.log(h / l)
        log_co = math.log(c / o)
        sum_gk += c1 * (log_hl ** 2) - c2 * (log_co ** 2)
    variance_per_hour = max(sum_gk / len(ohlc_window), 0.0)
    return math.sqrt(variance_per_hour * 8760.0)


def run_sync(target_bars: int = 1440) -> None:
    conn = init_db(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM hourly_state")
    existing_count = cur.fetchone()[0]
    fetch_bars = 72 if existing_count >= 1200 else target_bars
    print(f"[INFO] Existing DB rows: {existing_count} | Fetching {fetch_bars} hourly bars from OKX...")

    swap_candles = fetch_paginated_candles("/api/v5/market/history-candles", "BTC-USDT-SWAP", fetch_bars)
    mark_candles = fetch_paginated_candles("/api/v5/market/history-mark-price-candles", "BTC-USDT-SWAP", fetch_bars)
    index_candles = fetch_paginated_candles("/api/v5/market/history-index-candles", "BTC-USDT", fetch_bars)
    funding_map = fetch_funding_history("BTC-USDT-SWAP", pages=2 if fetch_bars == 72 else 4)
    oi_map = fetch_open_interest_history(fetch_bars)
    taker_map = fetch_taker_ratio_history(fetch_bars)
    live_l2 = fetch_live_l2_imbalance("BTC-USDT-SWAP", depth=20)

    current_hour_ms = floor_to_hour_ms(int(time.time() * 1000))
    sorted_ts = sorted(ts for ts in swap_candles.keys() if ts < current_hour_ms)
    sorted_funding_ts = sorted(funding_map.keys())
    last_known_funding = funding_map[sorted_funding_ts[0]] if sorted_funding_ts else 0.0

    upserted = 0
    for ts in sorted_ts:
        if ts in funding_map:
            last_known_funding = funding_map[ts]
        o, h, l, c, vol = swap_candles[ts]
        mark_c = mark_candles[ts][3] if ts in mark_candles else c
        idx_c = index_candles[ts][3] if ts in index_candles else c
        basis_bps = ((mark_c - idx_c) / idx_c) * 10_000.0 if idx_c > 0 else 0.0
        oi_usd = oi_map.get(ts)
        taker_ratio = taker_map.get(ts)
        iso_utc = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        l2_val = live_l2 if ts == sorted_ts[-1] else None

        cur.execute("""
            INSERT INTO hourly_state (
                ts_ms, iso_utc, open, high, low, close, volume_ccy,
                mark_close, index_close, basis_bps, funding_rate,
                oi_usd, taker_buy_ratio, live_l2_imbalance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ts_ms) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume_ccy=excluded.volume_ccy,
                mark_close=excluded.mark_close,
                index_close=excluded.index_close,
                basis_bps=excluded.basis_bps,
                funding_rate=excluded.funding_rate,
                oi_usd=COALESCE(excluded.oi_usd, hourly_state.oi_usd),
                taker_buy_ratio=COALESCE(excluded.taker_buy_ratio, hourly_state.taker_buy_ratio),
                live_l2_imbalance=COALESCE(excluded.live_l2_imbalance, hourly_state.live_l2_imbalance)
        """, (
            ts, iso_utc, o, h, l, c, vol,
            mark_c, idx_c, round(basis_bps, 3), last_known_funding,
            oi_usd, taker_ratio, l2_val
        ))
        upserted += 1
    conn.commit()

    cur.execute("SELECT ts_ms, open, high, low, close, volume_ccy, oi_usd FROM hourly_state ORDER BY ts_ms ASC")
    all_rows = cur.fetchall()
    for idx, row in enumerate(all_rows):
        ts_ms = row[0]
        window_24 = [list(r[1:6]) for r in all_rows[max(0, idx - 23):idx + 1]]
        gk_vol = compute_garman_klass_24h(window_24)
        oi_now = row[6]
        oi_prev = all_rows[idx - 24][6] if idx >= 24 else None
        oi_delta = ((oi_now - oi_prev) / oi_prev * 100.0) if (oi_now and oi_prev and oi_prev > 0) else None
        cur.execute("""
            UPDATE hourly_state
            SET gk_vol_24h = ?, oi_delta_24h = COALESCE(?, oi_delta_24h)
            WHERE ts_ms = ?
        """, (round(gk_vol, 5), round(oi_delta, 4) if oi_delta is not None else None, ts_ms))
    conn.commit()

    cur.execute("SELECT COUNT(*), MIN(iso_utc), MAX(iso_utc) FROM hourly_state WHERE oi_delta_24h IS NOT NULL")
    valid_count, min_dt, max_dt = cur.fetchone()
    conn.close()
    print(f"[OK] Upserted {upserted} rows. Total valid state vectors: {valid_count} ({min_dt} -> {max_dt})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--import-csv":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from import_historical_csv import bulk_import
        raw_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(DB_PATH), "raw_csv")
        bulk_import(raw_dir=raw_folder)
    run_sync(target_bars=1440)
