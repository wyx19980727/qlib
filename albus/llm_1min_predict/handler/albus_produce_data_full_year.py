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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from scripts.dump_bin import DumpDataAll

SILVER_BASE = "/hk-l2-data-lake/data/lake/silver/hk_l2"
AM_RANGE = ("09:30:00", "11:59:00")
PM_RANGE = ("13:00:00", "15:59:00")
PRICE_SCALE = 1000.0
OUT_COLS = [
    "date", "symbol",
    "open", "high", "low", "close", "volume",
    "change", "factor", "paused", "paused_num", "vwap",
]


def get_hk_calendar(date_str: str, freq: str):
    am = pd.date_range(f"{date_str} {AM_RANGE[0]}", f"{date_str} {AM_RANGE[1]}", freq=freq)
    pm = pd.date_range(f"{date_str} {PM_RANGE[0]}", f"{date_str} {PM_RANGE[1]}", freq=freq)
    return list(am) + list(pm)


def process_symbol_interval(date_str: str, symbol: str, csv_dir: str, freq_minutes: int, freq: str):
    tickex_path = os.path.join(SILVER_BASE, "tickex", f"date={date_str}", f"symbol={symbol}", "part-00000.snappy.parquet")

    if not os.path.exists(tickex_path):
        return

    pf = pq.ParquetFile(tickex_path)
    df = pf.read(columns=["timestamp", "price", "volume", "turnover"]).to_pandas()
    if df.empty:
        return
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
        cum_vol=("volume", "last"),
        cum_turn=("turnover", "last"),
    ).reset_index()
    for c in ("open", "high", "low", "close"):
        agg[c] = agg[c] / PRICE_SCALE
    agg["volume"] = agg["cum_vol"].diff()
    agg.iloc[0, agg.columns.get_indexer(["volume"])] = agg.iloc[0]["cum_vol"]
    agg["turnover"] = agg["cum_turn"].diff()
    agg.iloc[0, agg.columns.get_indexer(["turnover"])] = agg.iloc[0]["cum_turn"]

    cal = pd.DataFrame({"time_bin": get_hk_calendar(date_str, freq)})
    result = cal.merge(agg, on="time_bin", how="left")
    result = result.sort_values("time_bin").reset_index(drop=True)

    result["volume"] = result["volume"].fillna(0)
    result["turnover"] = result["turnover"].fillna(0)

    result["date"] = result["time_bin"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["symbol"] = symbol

    result["paused"] = (result["volume"] == 0).astype(int)
    result["factor"] = 1.0
    result["change"] = np.nan
    result["paused_num"] = np.nan
    # vwap = Δturnover / Δvolume（真元价，见 patch_silver_vwap.py 顶部推导）
    result["vwap"] = np.where(result["volume"] > 0, result["turnover"] / result["volume"], np.nan)

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

        df["change"] = df["close"].ffill().pct_change()

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
        all_data = all_data[: len(all_data) - all_nan_nums]
        if all_data:
            df = pd.concat(all_data, sort=False)
        else:
            continue
        df.drop(columns=["_dt", "_day"], inplace=True)
        df = df[OUT_COLS]
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
    parser.add_argument("--interval", default="1min", choices=["1min", "5min"],
                        help="Target bar frequency (default: 1min)")
    parser.add_argument("--test", action="store_true", help="Test mode: 2 days, 10 symbols")
    parser.add_argument("--dates", type=int, default=None, help="Number of dates")
    parser.add_argument("--symbols", type=int, default=None, help="Number of symbols per date")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8), help="Parallel workers")
    args = parser.parse_args()

    interval = args.interval
    freq_minutes = int(interval.replace("min", ""))
    csv_dir = f"/tmp/tick_to_{interval}_csv"
    qlib_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "qlib_data" / f"silver_data_{interval}")

    os.makedirs(csv_dir, exist_ok=True)
    for f in Path(csv_dir).glob("*.csv"):
        f.unlink()
    # Clean qlib_dir to avoid appending to stale bin files
    if Path(qlib_dir).exists():
        shutil.rmtree(qlib_dir)
    os.makedirs(qlib_dir, exist_ok=True)
            
    dates = sorted(d.replace("date=", "") for d in os.listdir(os.path.join(SILVER_BASE, "tickex")) if d.startswith("date="))
    limit_dates = args.dates or (2 if args.test else len(dates))
    dates = dates[:limit_dates]
    limit_syms = args.symbols or (10 if args.test else None)

    tasks, all_symbols = get_all_tasks(SILVER_BASE, dates, limit_syms)
    print(f"Tasks: {len(tasks)} ({len(dates)} dates, {len(all_symbols)} symbols)")

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

    print(f"\nDone! Qlib data saved to: {qlib_dir}")
    print(f"  calendars: {interval}.txt")
    print(f"  instruments: all.txt")
    print(f"  features: <symbol>/*.{interval}.bin")


if __name__ == "__main__":
    main()
