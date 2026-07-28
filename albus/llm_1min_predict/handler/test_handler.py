"""Benchmark: compare 3 approaches for loading & 1-min slicing HK L2 tick data.

Usage:
    python handler/test_handler.py
    python handler/test_handler.py --date 2026-01-05
    python handler/test_handler.py --symbols 1,3,5,8
    python handler/test_handler.py --rounds 5
"""

import os
import time
import statistics
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb

SILVER_BASE = "/hk-l2-data-lake/data/lake/silver/hk_l2"
KLINE_DIR = "/home/albus/Python_Codes/qlib/qlib_data/golden_data_1min/kline_1min"

AM_RANGE = ("09:30:00", "11:59:00")
PM_RANGE = ("13:00:00", "15:59:00")
PRICE_SCALE = 1000.0
BASE_COLS = ["open", "high", "low", "close", "volume", "turnover"]


def hk_calendar(date_str: str) -> pd.DatetimeIndex:
    am = pd.date_range(f"{date_str} {AM_RANGE[0]}", f"{date_str} {AM_RANGE[1]}", freq="1min")
    pm = pd.date_range(f"{date_str} {PM_RANGE[0]}", f"{date_str} {PM_RANGE[1]}", freq="1min")
    return am.append(pm)


def list_symbols(date_str: str) -> list:
    sym_dir = os.path.join(SILVER_BASE, "tickex", f"date={date_str}")
    if not os.path.isdir(sym_dir):
        return []
    return sorted(s.replace("symbol=", "") for s in os.listdir(sym_dir) if s.startswith("symbol="))


# ---------------------------------------------------------------------------
# Helpers (shared post-processing)
# ---------------------------------------------------------------------------

def _symbol_tick_path(date_str: str, symbol: str) -> str:
    return os.path.join(SILVER_BASE, "tickex", f"date={date_str}", f"symbol={symbol}", "part-00000.snappy.parquet")


def _finalize_bar(bar: pd.DataFrame, date_str: str, symbol: str) -> pd.DataFrame:
    """Price scaling, cumulative diff, calendar reindex, add instrument index."""
    for c in ("open", "high", "low", "close"):
        bar[c] = bar[c] / PRICE_SCALE
    bar["volume"] = bar["volume"].diff().fillna(bar["volume"])
    bar["turnover"] = bar["turnover"].diff().fillna(bar["turnover"])
    cal = hk_calendar(date_str)
    bar = bar.reindex(cal)
    bar["volume"] = bar["volume"].fillna(0)
    bar["turnover"] = bar["turnover"].fillna(0)
    bar["close"] = bar["close"].ffill()
    bar.index.name = "datetime"
    bar["instrument"] = symbol
    return bar.set_index("instrument", append=True)


# ---------------------------------------------------------------------------
# Approach 1 — load pre-processed 1-min parquet
# ---------------------------------------------------------------------------

def approach1_preprocessed(date_str: str, symbols: list = None) -> pd.DataFrame:
    path = os.path.join(KLINE_DIR, f"{date_str}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=BASE_COLS)
    df = pd.read_parquet(path)
    if symbols is not None:
        df = df[df.index.get_level_values("instrument").isin(symbols)]
    return df[BASE_COLS]


# ---------------------------------------------------------------------------
# Approach 2 — pandas: ParquetFile → to_pandas → groupby
# ---------------------------------------------------------------------------

def _pandas_one_symbol(date_str: str, symbol: str):
    path = _symbol_tick_path(date_str, symbol)
    if not os.path.exists(path):
        return None
    pf = pq.ParquetFile(path)
    df = pf.read(columns=["timestamp", "price", "volume", "turnover"]).to_pandas()
    if df.empty:
        return None
    df = df[df["price"] > 0]
    if df.empty:
        return None
    df["bin"] = pd.to_datetime(df["timestamp"]).dt.floor("1min")
    bar = df.groupby("bin", sort=True).agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), volume=("volume", "last"), turnover=("turnover", "last"),
    )
    return _finalize_bar(bar, date_str, symbol)


def approach2_pandas(date_str: str, symbols: list) -> pd.DataFrame:
    parts = [_pandas_one_symbol(date_str, s) for s in symbols]
    parts = [p for p in parts if p is not None]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


# ---------------------------------------------------------------------------
# Approach 3 — DuckDB: read_parquet → SQL date_trunc → GROUP BY
# ---------------------------------------------------------------------------

def _duckdb_one_symbol(date_str: str, symbol: str):
    path = _symbol_tick_path(date_str, symbol)
    if not os.path.exists(path):
        return None
    sql = f"""
        SELECT date_trunc('minute', timestamp::TIMESTAMP) AS bin,
               first(price) AS open, max(price) AS high, min(price) AS low,
               last(price) AS close, last(volume) AS volume,
               last(turnover) AS turnover
        FROM read_parquet('{path}')
        WHERE price > 0
        GROUP BY bin ORDER BY bin
    """
    bar = duckdb.query(sql).df()
    if bar.empty:
        return None
    bar = bar.set_index("bin")
    return _finalize_bar(bar, date_str, symbol)


def approach3_duckdb(date_str: str, symbols: list) -> pd.DataFrame:
    parts = [_duckdb_one_symbol(date_str, s) for s in symbols]
    parts = [p for p in parts if p is not None]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(date_str: str, symbol_counts: list, rounds: int = 3):
    all_symbols = list_symbols(date_str)
    if not all_symbols:
        print(f"No symbols found for {date_str}")
        return

    has_preprocessed = os.path.exists(os.path.join(KLINE_DIR, f"{date_str}.parquet"))
    print(f"\n===== Speed Benchmark: {date_str}, {len(all_symbols)} symbols available =====")
    if not has_preprocessed:
        print("  [kline_1min file not found — skipping approach 1]")
    print(f"{'n_symbols':>10}", end="")
    for n in symbol_counts:
        print(f"{f'  {n:>3} sym':>14}", end="")
    print()

    approaches = [
        ("1. Preprocessed (kline_1min)", approach1_preprocessed),
        ("2. Pandas (raw tick)       ", approach2_pandas),
        ("3. DuckDB (raw tick)       ", approach3_duckdb),
    ]

    for label, func in approaches:
        print(f"{label:>10}", end="")
        for n in symbol_counts:
            symbols = [s for s in all_symbols if os.path.exists(
                _symbol_tick_path(date_str, s)
            )][:n]
            if len(symbols) < n:
                print(f"{'  N/A':>14}", end="")
                continue

            times = []
            for _ in range(rounds):
                t0 = time.perf_counter()
                df = func(date_str, symbols)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            result_rows = len(df) if not df.empty else 0
            print(f"  {mean:>6.1f}±{std:.1f}ms", end="")
        print()

    # verify approaches produce same result
    sym_test = [s for s in all_symbols if os.path.exists(_symbol_tick_path(date_str, s))][:2]
    if len(sym_test) >= 2:
        df1 = approach1_preprocessed(date_str, sym_test)
        df2 = approach2_pandas(date_str, sym_test)
        df3 = approach3_duckdb(date_str, sym_test)
        for name, df in [("preprocessed", df1), ("pandas", df2), ("duckdb", df3)]:
            print(f"  {name}: {len(df)} rows, index={df.index.names}, cols={list(df.columns)}")
        # compare (duckdb vs pandas) — close enough check
        if not df2.empty and not df3.empty:
            diff = (df2[BASE_COLS] - df3[BASE_COLS]).abs().max().max()
            print(f"  max diff (pandas vs duckdb): {diff:.6f}")
        if not df1.empty and not df2.empty:
            diff = (df1[BASE_COLS] - df2[BASE_COLS]).abs().max().max()
            print(f"  max diff (preprocessed vs pandas): {diff:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-01-05")
    parser.add_argument("--symbols", default="1,3,5,8")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    symbol_counts = [int(x) for x in args.symbols.split(",")]
    benchmark(args.date, symbol_counts, args.rounds)
