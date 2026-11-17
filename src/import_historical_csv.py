#!/usr/bin/env python3
"""
CORTEX-RHYME — Bulk Multi-Regime Historical CSV/ZIP Importer
Ingests 2+ years of historical OKX CSV/ZIP files from data/raw_csv/ into
data/cortex_rhyme.db so the analogue engine spans bull, bear, and chop cycles.
"""
import argparse
import csv
import glob
import io
import os
import sys
import zipfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seed_and_update import (
    DB_PATH,
    compute_garman_klass_24h,
    floor_to_hour_ms,
    init_db,
)


def parse_timestamp_ms(raw_val: str) -> Optional[int]:
    """Parses epoch ms, epoch seconds, or ISO-8601 UTC strings into floored hour ms."""
    val = raw_val.strip()
    if not val:
        return None
    try:
        num = float(val)
        ms = int(num) if num > 1e11 else int(num * 1000)
        return floor_to_hour_ms(ms)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            dt = datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
            return floor_to_hour_ms(int(dt.timestamp() * 1000))
        except ValueError:
            continue
    return None


def iter_csv_rows_from_path(filepath: str) -> Iterable[List[str]]:
    """Yields raw CSV rows from either a plain .csv or a .zip containing .csv files."""
    if filepath.lower().endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zf:
            for member in sorted(zf.namelist()):
                if member.lower().endswith(".csv"):
                    with zf.open(member, "r") as raw_f:
                        text_f = io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace")
                        for row in csv.reader(text_f):
                            if row:
                                yield row
    else:
        with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
            for row in csv.reader(f):
                if row:
                    yield row


def classify_file(filename: str, header: List[str]) -> str:
    """Identifies OKX archive type from filename and header columns."""
    name_low = os.path.basename(filename).lower()
    head_low = ",".join(h.strip().lower() for h in header)
    if "funding" in name_low or "funding_rate" in head_low or "realized_rate" in head_low:
        return "funding"
    if "mark" in name_low:
        return "mark_candles"
    if "index" in name_low:
        return "index_candles"
    if "oi" in name_low or "open_interest" in head_low:
        return "oi_taker"
    return "swap_candles"


def parse_candle_rows(rows: Iterable[List[str]]) -> Dict[int, Tuple[float, float, float, float, float]]:
    """Extracts {ts_ms: (open, high, low, close, vol_ccy)} from OKX kline CSV rows."""
    out: Dict[int, Tuple[float, float, float, float, float]] = {}
    for row in rows:
        if len(row) < 5:
            continue
        ts = parse_timestamp_ms(row[0])
        if ts is None:
            continue
        try:
            o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            vol = float(row[6]) if len(row) > 6 and row[6].strip() else (float(row[5]) if len(row) > 5 else 0.0)
            out[ts] = (o, h, l, c, vol)
        except ValueError:
            continue
    return out


def parse_funding_rows(rows: Iterable[List[str]]) -> Dict[int, float]:
    """Extracts {ts_ms: realized_funding_rate} from OKX funding rate CSV rows."""
    out: Dict[int, float] = {}
    header_map: Dict[str, int] = {}
    for row in rows:
        if not header_map and any("funding" in c.lower() or "time" in c.lower() for c in row):
            header_map = {c.strip().lower(): idx for idx, c in enumerate(row)}
            continue
        ts_idx = header_map.get("fundingtime", header_map.get("timestamp", 1 if len(row) >= 4 else 0))
        rate_idx = header_map.get("realizedrate", header_map.get("fundingrate", 2 if len(row) >= 3 else 1))
        if len(row) <= max(ts_idx, rate_idx):
            continue
        ts = parse_timestamp_ms(row[ts_idx])
        if ts is None:
            continue
        try:
            rate_str = row[rate_idx].strip()
            if not rate_str and rate_idx > 0:
                rate_str = row[rate_idx - 1].strip()
            out[ts] = float(rate_str)
        except ValueError:
            continue
    return out


def parse_oi_taker_rows(rows: Iterable[List[str]]) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """Extracts {ts_ms: (oi_usd, taker_buy_ratio)} from optional custom hourly CSVs."""
    out: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    header_map: Dict[str, int] = {}
    for row in rows:
        if not header_map and any("oi" in c.lower() or "time" in c.lower() for c in row):
            header_map = {c.strip().lower(): idx for idx, c in enumerate(row)}
            continue
        ts_idx = header_map.get("ts_ms", header_map.get("timestamp", 0))
        oi_idx = header_map.get("oi_usd", header_map.get("oi", 1))
        taker_idx = header_map.get("taker_buy_ratio", header_map.get("taker_ratio", 2))
        ts = parse_timestamp_ms(row[ts_idx]) if len(row) > ts_idx else None
        if ts is None:
            continue
        oi_val = float(row[oi_idx]) if len(row) > oi_idx and row[oi_idx].strip() else None
        taker_val = float(row[taker_idx]) if len(row) > taker_idx and row[taker_idx].strip() else None
        out[ts] = (oi_val, taker_val)
    return out


def bulk_import(raw_dir: str, db_path: str = DB_PATH, neutral_fill_missing_derivatives: bool = True) -> None:
    files = sorted(
        glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
        + glob.glob(os.path.join(raw_dir, "**", "*.zip"), recursive=True)
    )
    if not files:
        print(f"[WARN] No .csv or .zip files found in {raw_dir}")
        return

    swap_all: Dict[int, Tuple[float, float, float, float, float]] = {}
    mark_all: Dict[int, Tuple[float, float, float, float, float]] = {}
    index_all: Dict[int, Tuple[float, float, float, float, float]] = {}
    funding_all: Dict[int, float] = {}
    oi_taker_all: Dict[int, Tuple[Optional[float], Optional[float]]] = {}

    for fp in files:
        row_iter = iter_csv_rows_from_path(fp)
        first_row = next(row_iter, [])
        if not first_row:
            continue
        ftype = classify_file(fp, first_row)

        def full_iter(head=first_row, rest=row_iter):
            yield head
            yield from rest

        if ftype == "funding":
            parsed = parse_funding_rows(full_iter())
            funding_all.update(parsed)
            print(f"[READ] {os.path.basename(fp)} -> funding ({len(parsed)} settlements)")
        elif ftype == "mark_candles":
            parsed = parse_candle_rows(full_iter())
            mark_all.update(parsed)
            print(f"[READ] {os.path.basename(fp)} -> mark_candles ({len(parsed)} bars)")
        elif ftype == "index_candles":
            parsed = parse_candle_rows(full_iter())
            index_all.update(parsed)
            print(f"[READ] {os.path.basename(fp)} -> index_candles ({len(parsed)} bars)")
        elif ftype == "oi_taker":
            parsed = parse_oi_taker_rows(full_iter())
            oi_taker_all.update(parsed)
            print(f"[READ] {os.path.basename(fp)} -> oi_taker ({len(parsed)} bars)")
        else:
            parsed = parse_candle_rows(full_iter())
            swap_all.update(parsed)
            print(f"[READ] {os.path.basename(fp)} -> swap_candles ({len(parsed)} bars)")

    if not swap_all:
        print("[ERROR] At least one Swap 1H Candlestick CSV/ZIP is required as the primary time index.")
        return

    conn = init_db(db_path)
    cur = conn.cursor()
    sorted_ts = sorted(swap_all.keys())
    sorted_fund_ts = sorted(funding_all.keys())
    last_funding = funding_all[sorted_fund_ts[0]] if sorted_fund_ts else 0.0
    upserted = 0

    for ts in sorted_ts:
        if ts in funding_all:
            last_funding = funding_all[ts]
        o, h, l, c, vol = swap_all[ts]
        mark_c = mark_all[ts][3] if ts in mark_all else c
        idx_c = index_all[ts][3] if ts in index_all else c
        basis_bps = ((mark_c - idx_c) / idx_c) * 10_000.0 if idx_c > 0 else 0.0
        oi_usd, taker_ratio = oi_taker_all.get(ts, (None, None))
        if neutral_fill_missing_derivatives and taker_ratio is None:
            taker_ratio = 0.5
        iso_utc = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        cur.execute("""
            INSERT INTO hourly_state (
                ts_ms, iso_utc, open, high, low, close, volume_ccy,
                mark_close, index_close, basis_bps, funding_rate,
                oi_usd, taker_buy_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ts_ms) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume_ccy=excluded.volume_ccy,
                mark_close=COALESCE(excluded.mark_close, hourly_state.mark_close),
                index_close=COALESCE(excluded.index_close, hourly_state.index_close),
                basis_bps=excluded.basis_bps,
                funding_rate=excluded.funding_rate,
                oi_usd=COALESCE(excluded.oi_usd, hourly_state.oi_usd),
                taker_buy_ratio=COALESCE(excluded.taker_buy_ratio, hourly_state.taker_buy_ratio)
        """, (
            ts, iso_utc, o, h, l, c, vol,
            mark_c, idx_c, round(basis_bps, 3), last_funding,
            oi_usd, taker_ratio
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
        if oi_now and oi_prev and oi_prev > 0:
            oi_delta = round((oi_now - oi_prev) / oi_prev * 100.0, 4)
        else:
            oi_delta = 0.0 if neutral_fill_missing_derivatives else None
        cur.execute("""
            UPDATE hourly_state
            SET gk_vol_24h = ?, oi_delta_24h = ?
            WHERE ts_ms = ?
        """, (round(gk_vol, 5), oi_delta, ts_ms))
    conn.commit()

    cur.execute("SELECT COUNT(*), MIN(iso_utc), MAX(iso_utc) FROM hourly_state")
    total_rows, min_dt, max_dt = cur.fetchone()
    conn.close()
    print(f"[OK] Bulk imported {upserted} bars. DB now holds {total_rows} hourly rows ({min_dt} -> {max_dt})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk import historical OKX CSV/ZIP archives into CORTEX-RHYME SQLite DB.")
    parser.add_argument("--raw-dir", default=os.path.join(os.path.dirname(DB_PATH), "raw_csv"))
    parser.add_argument("--strict-oi", action="store_true")
    args = parser.parse_args()
    bulk_import(raw_dir=args.raw_dir, neutral_fill_missing_derivatives=not args.strict_oi)
