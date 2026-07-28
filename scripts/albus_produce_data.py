import os
import sys
import argparse
import time
import pandas as pd
import pandas._libs.tslibs.offsets as _off
_off.prefix_mapping["day"] = _off.prefix_mapping["D"]
_off.prefix_mapping["DAY"] = _off.prefix_mapping["D"]
import numpy as np
import gc
import pyarrow.parquet as pq
from pathlib import Path
from multiprocessing import Pool
import shutil
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dump_bin import DumpDataAll

SILVER_BASE = "/hk-l2-data-lake/data_deprecate/lake/silver/hk_l2"
AM_RANGE = ("09:30:00", "11:59:00")
PM_RANGE = ("13:00:00", "15:59:00")
OUT_COLS = [
    "date", "symbol",
    "open", "high", "low", "close", "volume", "turnover", "vwap",
    "trade_count", "buy_count", "sell_count", "unknown_dir_count",
    "unique_brokers", "avg_bid_volume", "avg_ask_volume",
    "paused", "factor", "change", "paused_num",
]
EXTRA_COLS = [
    "buy_count", "sell_count", "unknown_dir_count", "unique_brokers",
    "avg_bid_volume", "avg_ask_volume",
]


def get_hk_calendar(date_str: str, freq: str):
    am = pd.date_range(f"{date_str} {AM_RANGE[0]}", f"{date_str} {AM_RANGE[1]}", freq=freq)
    pm = pd.date_range(f"{date_str} {PM_RANGE[0]}", f"{date_str} {PM_RANGE[1]}", freq=freq)
    return list(am) + list(pm)


def parse_hk_time(time_int, date_str):
    s = str(int(time_int)).zfill(6)
    return f"{date_str} {s[:2]}:{s[2:4]}:{s[4:6]}"


def process_symbol_interval(date_str: str, symbol: str, csv_dir: str, freq_minutes: int, freq: str):
    tickex_path = os.path.join(SILVER_BASE, "tickex", f"date={date_str}", f"symbol={symbol}", "part-00000.snappy.parquet")
    res_path = os.path.join(SILVER_BASE, "traderesumes", f"date={date_str}", f"symbol={symbol}", "part-00000.snappy.parquet")

    if not os.path.exists(tickex_path):
        return

    pf = pq.ParquetFile(tickex_path)
    df = pf.read(columns=["timestamp", "price", "volume", "turnover"]).to_pandas()
    if df.empty:
        return
    # drop zero-price & zero-volume junk ticks
    df = df[(df["price"] != 0) | (df["volume"] != 0)]
    if df.empty:
        return
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df["time_bin"] = df["datetime"].dt.floor(f"{freq_minutes}min")

    grp = df.groupby("time_bin", sort=True)
    agg = grp.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
        turnover=("turnover", "sum"),
        trade_count=("price", "count"),
    ).reset_index()
    agg["vwap"] = np.where(agg["volume"] > 0, agg["turnover"] / agg["volume"], np.nan)
    for c in EXTRA_COLS:
        agg[c] = 0

    if os.path.exists(res_path):
        pf_res = pq.ParquetFile(res_path)
        df_res = pf_res.read(columns=["time", "dir", "brokerno", "bidvolume", "askvolume"]).to_pandas()
        if not df_res.empty:
            df_res["time_bin"] = pd.Series(
                pd.to_datetime([parse_hk_time(t, date_str) for t in df_res["time"]])
            ).dt.floor(f"{freq_minutes}min")
            df_res = df_res.dropna(subset=["time_bin"])
            if not df_res.empty:
                grp_res = df_res.groupby("time_bin", sort=True)
                agg_res = grp_res.agg(
                    buy_count=("dir", lambda x: (x == 1).sum()),
                    sell_count=("dir", lambda x: (x == 2).sum()),
                    unknown_dir_count=("dir", lambda x: (x == 0).sum()),
                    unique_brokers=("brokerno", "nunique"),
                    avg_bid_volume=("bidvolume", "mean"),
                    avg_ask_volume=("askvolume", "mean"),
                ).reset_index()
                agg = agg.merge(agg_res, on="time_bin", how="left", suffixes=("", "_r"))
                for c in EXTRA_COLS:
                    rc = f"{c}_r"
                    if rc in agg.columns:
                        agg[c] = agg[rc].fillna(0)
                        agg.drop(columns=[rc], inplace=True)

    cal = pd.DataFrame({"time_bin": get_hk_calendar(date_str, freq)})
    result = cal.merge(agg, on="time_bin", how="left")
    result = result.sort_values("time_bin").reset_index(drop=True)

    VOL_COLS = ["volume", "turnover", "trade_count", "buy_count", "sell_count",
                "unknown_dir_count", "unique_brokers", "avg_bid_volume", "avg_ask_volume"]
    result[VOL_COLS] = result[VOL_COLS].fillna(0)

    result["date"] = result["time_bin"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["symbol"] = symbol

    result["paused"] = (result["volume"] == 0).astype(int)
    result["factor"] = 1.0
    result["change"] = np.nan
    result["paused_num"] = np.nan

    result = result[OUT_COLS]

    csv_path = os.path.join(csv_dir, f"{symbol}.csv")
    result.to_csv(csv_path, index=False, mode="a", header=not os.path.exists(csv_path))

    del pf, df, agg, result


def post_process_minute_csv(csv_dir: str, symbols: list):
    for sym in tqdm(symbols, desc="  post-process minute"):
        csv_path = os.path.join(csv_dir, f"{sym}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["_dt"] = pd.to_datetime(df["date"])
        df.sort_values("_dt", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # change: close / prev_close - 1 (minute-level, ffill close for computation only)
        df["change"] = df["close"].ffill().pct_change()

        # paused_num: consecutive non-paused day counter
        df["_day"] = df["_dt"].dt.date
        all_data = []
        all_nan_nums = 0
        not_nan_nums = 0
        for _day, _df in df.groupby("_day", sort=True, group_keys=False):
            if (_df["paused"] == 1).all():
                all_nan_nums += 1
                not_nan_nums = 0
                _df["paused_num"] = not_nan_nums
                if all_data:
                    all_data.append(_df)
            else:
                all_nan_nums = 0
                not_nan_nums += 1
                _df["paused_num"] = not_nan_nums
                all_data.append(_df)
        # drop trailing paused days
        all_data = all_data[: len(all_data) - all_nan_nums]
        if all_data:
            df = pd.concat(all_data, sort=False)
        else:
            continue
        df.drop(columns=["_dt", "_day"], inplace=True)
        df = df[OUT_COLS]
        df.to_csv(csv_path, index=False)


def aggregate_csv_to_day(csv_dir: str, day_csv_dir: str, symbols: list):
    os.makedirs(day_csv_dir, exist_ok=True)
    for sym in tqdm(symbols, desc="  aggregate→day"):
        src = os.path.join(csv_dir, f"{sym}.csv")
        if not os.path.exists(src):
            continue
        df = pd.read_csv(src)
        if df.empty:
            continue
        df["_dt"] = pd.to_datetime(df["date"])
        df["_day"] = df["_dt"].dt.strftime("%Y-%m-%d")

        def safe_div(x, y):
            return np.where(y > 0, x / y, np.nan)

        def nan_safe_sum(s):
            return np.nan if s.isna().all() else s.sum()

        daily = df.groupby("_day", sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", nan_safe_sum),
            turnover=("turnover", nan_safe_sum),
            trade_count=("trade_count", "sum"),
            buy_count=("buy_count", "sum"),
            sell_count=("sell_count", "sum"),
            unknown_dir_count=("unknown_dir_count", "sum"),
            unique_brokers=("unique_brokers", "max"),
            avg_bid_volume=("avg_bid_volume", "mean"),
            avg_ask_volume=("avg_ask_volume", "mean"),
            paused=("paused", "min"),
            factor=("factor", "first"),
            paused_num=("paused_num", "first"),
        ).reset_index()
        daily["vwap"] = safe_div(daily["turnover"], daily["volume"])
        daily["date"] = daily["_day"]
        daily["symbol"] = sym
        daily["change"] = np.nan
        daily = daily[OUT_COLS]

        dst = os.path.join(day_csv_dir, f"{sym}.csv")
        daily.to_csv(dst, index=False, mode="a", header=not os.path.exists(dst))


def post_process_day_csv(csv_dir: str, symbols: list):
    for sym in tqdm(symbols, desc="  post-process day"):
        csv_path = os.path.join(csv_dir, f"{sym}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["_dt"] = pd.to_datetime(df["date"])
        df.sort_values("_dt", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["change"] = df["close"].ffill().pct_change()
        df.drop(columns=["_dt"], inplace=True)
        df.to_csv(csv_path, index=False)


def dump_csv_to_qlib(csv_dir: str, qlib_dir: str, freq: str, workers: int):
    DumpDataAll(
        data_path=csv_dir,
        qlib_dir=qlib_dir,
        freq=freq,
        date_field_name="date",
        symbol_field_name="symbol",
        exclude_fields="symbol",
        max_workers=workers,
    ).dump()


def remove_zombie_stocks(qlib_dir: str, freq: str):
    features_dir = os.path.join(qlib_dir, "features")
    if not os.path.isdir(features_dir):
        return
    removed = []
    for sym in os.listdir(features_dir):
        vol_file = os.path.join(features_dir, sym, f"volume.{freq}.bin")
        if not os.path.exists(vol_file):
            continue
        with open(vol_file, "rb") as f:
            raw = f.read()
        vals = np.frombuffer(raw[4:], dtype="<f4")
        active_ratio = (vals > 0).sum() / max(len(vals), 1)
        if active_ratio < 0.01:
            removed.append(sym)
            shutil.rmtree(os.path.join(features_dir, sym))

    if removed:
        removed_set = set(removed)
        instr_path = os.path.join(qlib_dir, "all.txt")
        if os.path.exists(instr_path):
            with open(instr_path) as f:
                lines = f.readlines()
            with open(instr_path, "w") as f:
                for line in lines:
                    parts = line.split("\t")
                    if parts[0] not in removed_set:
                        f.write(line)
        remaining = len(os.listdir(features_dir)) if os.path.isdir(features_dir) else 0
        print(f"  Removed {len(removed)} zombie stocks ({remaining} remaining)")


def run_tasks_parallel(tasks: list, csv_dir: str, freq_minutes: int, interval: str,
                       workers: int, date_str: str, hang_timeout: int = 120, poll_interval: int = 0.2):
    pool = Pool(processes=workers)
    async_results = []
    for d, s in tasks:
        async_results.append((d, s, pool.apply_async(
            process_symbol_interval, (d, s, csv_dir, freq_minutes, interval)
        )))
    with tqdm(total=len(async_results), desc=f"  {date_str}") as pbar:
        last_remaining = len(async_results)
        stuck = 0
        while async_results:
            time.sleep(poll_interval)
            ready = [(d, s, r) for d, s, r in async_results if r.ready()]
            for rd, rs, r in ready:
                try:
                    r.get(timeout=0.1)
                except Exception as e:
                    print(f"  ERROR {rd} {rs}: {e}")
                pbar.update()
            async_results = [(d, s, r) for d, s, r in async_results if not r.ready()]
            if len(async_results) == last_remaining:
                stuck += poll_interval
                if stuck >= hang_timeout:
                    print(f"  No progress for {hang_timeout}s — {len(async_results)} hung, terminating pool")
                    break
            else:
                stuck = 0
                last_remaining = len(async_results)
        for rd, rs, r in async_results:
            print(f"  TIMEOUT {rd} {rs}: worker hung, skipped")
            pbar.update()
    pool.terminate()
    pool.join()


def get_all_tasks(silver_base: str, dates: list, limit_syms: int = None):
    tasks, all_symbols_set = [], set()
    for d in dates:
        sym_dir = os.path.join(silver_base, "tickex", f"date={d}")
        if not os.path.isdir(sym_dir):
            continue
        syms = sorted(s.replace("symbol=", "") for s in os.listdir(sym_dir) if s.startswith("symbol="))
        if limit_syms:
            syms = syms[:limit_syms]
        for s in syms:
            tasks.append((d, s))
        all_symbols_set.update(syms)
    return tasks, sorted(all_symbols_set)


def main():
    
    parser = argparse.ArgumentParser(description="Build HK L2 tick data into qlib bin format")
    parser.add_argument("--interval", default="day", choices=["1min", "5min", "day"],
                        help="Target bar frequency (default: 1min). day is also always generated.")
    parser.add_argument("--test", action="store_true", help="Test mode: 2 days, 10 symbols")
    parser.add_argument("--dates", type=int, default=None, help="Number of dates")
    parser.add_argument("--symbols", type=int, default=None, help="Number of symbols per date")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8), help="Parallel workers")
    args = parser.parse_args()

    interval = args.interval
    freq_minutes = int(interval.replace("min", "")) if interval != "day" else 0
    csv_dir = f"/tmp/tick_to_{interval}_csv"
    day_csv_dir = "/tmp/tick_to_day_csv"
    qlib_dir = str(Path(__file__).resolve().parent.parent / "qlib_data" / f"silver_data_{interval}")

    os.makedirs(qlib_dir, exist_ok=True)
    for d in [csv_dir, day_csv_dir]:
        os.makedirs(d, exist_ok=True)
        for f in Path(d).glob("*.csv"):
            f.unlink()
            
    dates = sorted(d.replace("date=", "") for d in os.listdir(os.path.join(SILVER_BASE, "tickex")) if d.startswith("date="))
    limit_dates = args.dates or (2 if args.test else len(dates))
    dates = dates[:limit_dates]
    limit_syms = args.symbols or (10 if args.test else None)

    tasks, all_symbols = get_all_tasks(SILVER_BASE, dates, limit_syms)
    print(f"Tasks: {len(tasks)} ({len(dates)} dates, {len(all_symbols)} symbols)")

    if interval == "day":
        for date_idx, d in enumerate(dates):
            sym_dir = os.path.join(SILVER_BASE, "tickex", f"date={d}")
            syms = sorted(s.replace("symbol=", "") for s in os.listdir(sym_dir) if s.startswith("symbol="))
            if limit_syms:
                syms = syms[:limit_syms]
            day_tasks = [(d, s) for s in syms]
            print(f"[{date_idx+1}/{len(dates)}] {d} — {len(day_tasks)} symbols (day)")
            run_tasks_parallel(day_tasks, day_csv_dir, freq_minutes, interval, args.workers, d)
        print(f"\nPost-processing day CSV (change)...")
        post_process_day_csv(day_csv_dir, all_symbols)
        print(f"Dumping day → qlib bin (full)...")
        dump_csv_to_qlib(day_csv_dir, qlib_dir, "day", args.workers)
        # remove_zombie_stocks(qlib_dir, "day")
    else:
        for date_idx, d in enumerate(dates):
            sym_dir = os.path.join(SILVER_BASE, "tickex", f"date={d}")
            syms = sorted(s.replace("symbol=", "") for s in os.listdir(sym_dir) if s.startswith("symbol="))
            if limit_syms:
                syms = syms[:limit_syms]
            intra_tasks = [(d, s) for s in syms]
            print(f"[{date_idx+1}/{len(dates)}] {d} — {len(intra_tasks)} symbols ({interval})")
            run_tasks_parallel(intra_tasks, csv_dir, freq_minutes, interval, args.workers, d)
        print(f"\nPost-processing minute CSV (change, paused_num)...")
        post_process_minute_csv(csv_dir, all_symbols)
        print(f"Dumping {interval} → qlib bin (full)...")
        dump_csv_to_qlib(csv_dir, qlib_dir, interval, args.workers)
        # remove_zombie_stocks(qlib_dir, interval)
        print(f"Aggregating {interval} → day CSV...")
        aggregate_csv_to_day(csv_dir, day_csv_dir, all_symbols)
        print(f"Post-processing day CSV (change)...")
        post_process_day_csv(day_csv_dir, all_symbols)
        print(f"Dumping day → qlib bin (full)...")
        dump_csv_to_qlib(day_csv_dir, qlib_dir, "day", args.workers)
        # remove_zombie_stocks(qlib_dir, "day")

    print(f"\nDone! Qlib data saved to: {qlib_dir}")
    print(f"  calendars: {interval}.txt, day.txt")
    print(f"  instruments: all.txt")
    print(f"  features: <symbol>/*.{interval}.bin, *.day.bin")


if __name__ == "__main__":
    main()
