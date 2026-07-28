"""增量补丁：从 tickex 原始 parquet 计算**真 vwap** = Σ(price·volume)/Σvolume，
补齐 silver_data_1min 缺失的 vwap.1min.bin 文件。

背景：
    albus_produce_data_full_year.py 聚合 silver 时（process_symbol_interval 第44行）只读了
    ["timestamp", "price", "volume"]，丢掉了 tickex 原始 parquet 中的 turnover 字段，导致
    silver_data_1min 的 features/<sym>/ 目录下没有 vwap.1min.bin。原版 Alpha158 在缺失 \$vwap
    字段时不会报错（graceful fallback 返回 NaN），但 VWAP0 这一整列实际全是 NaN —— 是一个
    隐藏 bug。

    真实 vwap 无法靠 OHLC 算出（必须量加权），但可由 tickex 的 turnover 字段直接得到：
        per_minute_vwap = Δturnover / Δvolume / PRICE_SCALE
                     = Σ(tick_price × tick_volume) / Σ(tick_volume) / 1000
    这比 Simpson 近似 (\$open+2·\$high+2·\$low+\$close)/6 更准 —— Simpson 只用价格，
    忽略成交量加权。

本脚本是 **增量补丁**，与 albus_produce_data_full_year.py 互补：
    - 不重跑全量聚合（避免 139 天 × 3172 标的重写 OHLCV+change/factor/paused/paused_num）
    - 只补 vwap.1min.bin 一个文件到每个标的的 features 目录
    - 其他字段保持不动

输出格式（与已有 close.1min.bin 完全一致）：
    features/<sym>/vwap.1min.bin:
        [4B float32 header = 起始 calendar index (=0)]
        + [N × 4B float32 body 对齐到全局 1min calendar]
        全局 N = calendar 总分钟数（silver_data_1min 当前 = 43890）
        缺失或停牌 slot 填 NaN（与 close.1min.bin 同策略）

用法：
    # 冒烟：少量标的
    python patch_silver_vwap.py --test

    # 全量补齐（约 20 分钟，按 8 进程估算）
    python patch_silver_vwap.py --workers 8

    # 只补指定日期
    python patch_silver_vwap.py --dates 2026-01-05,2026-01-06

    # resume：跳过已存在且大小正确的 vwap.1min.bin
    python patch_silver_vwap.py --resume
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# 复用上游脚本常量（避免硬编码不一致）
SILVER_BASE = "/hk-l2-data-lake/data/lake/silver/hk_l2"
TICKEX_DIR = os.path.join(SILVER_BASE, "tickex")

SILVER_QLIB_DIR = "/home/albus/Python_Codes/qlib/qlib_data/silver_data_1min"
FEATURES_DIR = os.path.join(SILVER_QLIB_DIR, "features")

AM_RANGE = ("09:30:00", "11:59:00")
PM_RANGE = ("13:00:00", "15:59:00")
# 真实 vwap 推导（约简）：
#   tickex.price (int) 单位千元，silver 的 close=price/1000 即元相符
#   tickex.volume (int) 单位股
#   tickex.turnover (int) = price × volume / 1000  (经验证，第 0 行 53900×76000=4096.4M，turnover=4096400)
# 真元 vwap = Σ(price_yuan × volume) / Σvolume = Σ((price_int/1000)×volume)/Σvolume
#           = (Σ(price_int × volume) / 1000) / Σvolume
#           = (turnover × 1000 / 1000) / Σvolume = turnover / Σvolume
# 即 vwap(元) = Δturnover / Δvolume，不要再除 PRICE_SCALE。
PRICE_SCALE = 1000.0  # 仅用于本备忘注释；vwap 计算不再用

FREQ = "1min"  # 本脚本只处理 1min 频率

# 读取全局 calendar 一次（进程内引用）
_CALENDAR_CACHE: pd.DatetimeIndex = None  # type: ignore


def get_global_calendar() -> pd.DatetimeIndex:
    """读取 silver_data_1min/calendars/1min.txt，返回全局 1min calendar。"""
    global _CALENDAR_CACHE
    if _CALENDAR_CACHE is None:
        p = os.path.join(SILVER_QLIB_DIR, "calendars", "1min.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"calendar not found: {p}")
        with open(p) as f:
            times = [line.strip() for line in f if line.strip()]
        _CALENDAR_CACHE = pd.to_datetime(times)
    return _CALENDAR_CACHE


def get_hk_calendar(date_str: str) -> pd.DatetimeIndex:
    """单日港股 330 分钟 calendar（09:30-11:59 + 13:00-15:59）。"""
    am = pd.date_range(f"{date_str} {AM_RANGE[0]}", f"{date_str} {AM_RANGE[1]}", freq=FREQ)
    pm = pd.date_range(f"{date_str} {PM_RANGE[0]}", f"{date_str} {PM_RANGE[1]}", freq=FREQ)
    return am.append(pm)


# --------------------------------------------------------------------------- #
# 单标的处理：读所有日期 tickex → 聚合 vwap → 对齐全局 calendar → 写 bin
# --------------------------------------------------------------------------- #

def process_one_symbol(
    symbol: str,
    available_dates: List[str],
    out_bin_path: str,
    expected_body_len: int,
    resume: bool = False,
) -> Tuple[str, int, str]:
    """Aggregate vwap for one symbol across all dates, write vwap.1min.bin.

    Returns
    -------
    (symbol, status_code, message)
      status_code: 0 = OK, 1 = SKIP (resume), 2 = EMPTY, 3 = ERROR
    """
    out_path = Path(out_bin_path)
    if resume and out_path.exists() and out_path.stat().st_size == 4 + expected_body_len * 4:
        return (symbol, 1, "skip existing")

    try:
        cal_all = get_global_calendar()
        # 单标的的全局 vwap 数组，初值全 NaN
        vwap_full = np.full(len(cal_all), np.nan, dtype="<f4")

        for d in available_dates:
            tickex = os.path.join(TICKEX_DIR, f"date={d}", f"symbol={symbol}", "part-00000.snappy.parquet")
            if not os.path.exists(tickex):
                continue

            try:
                pf = pq.ParquetFile(tickex)
                df = pf.read(columns=["timestamp", "price", "volume", "turnover"]).to_pandas()
            except Exception:
                # turnover 字段在某些 parquet 可能不存在；跳过当日
                continue
            if df.empty:
                continue

            # 与 albus_produce_data_full_year.py 一致的过滤
            df = df[(df["price"] != 0) | (df["volume"] != 0)]
            if df.empty:
                continue

            df["datetime"] = pd.to_datetime(df["timestamp"])
            df["bin"] = df["datetime"].dt.floor("1min")

            agg = df.groupby("bin", sort=True).agg(
                cum_vol=("volume", "last"),
                cum_turn=("turnover", "last"),
            )
            # 还原 per-minute 增量（tickex 中 volume/turnover 是累计量）
            vol_per_min = agg["cum_vol"].diff().fillna(agg["cum_vol"])
            turn_per_min = agg["cum_turn"].diff().fillna(agg["cum_turn"])

            # 真元 vwap = Δturnover / Δvolume（见顶部推导），0 量分钟 → NaN
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap_min = turn_per_min / vol_per_min.replace(0, np.nan)
            vwap_min = vwap_min.astype(np.float32)

            # 对齐单日 330 分钟 calendar（缺失 slot 保持 NaN）
            day_cal = get_hk_calendar(d)
            vwap_day = vwap_min.reindex(day_cal)

            # 把当日 vwap 填回全局 calendar 对应位置
            day_start = cal_all.get_loc(day_cal[0])
            # 若 day_cal 是 cal_all 的子集，切片直接对齐
            vwap_full[day_start : day_start + len(day_cal)] = vwap_day.values

        # 写 bin：4B header (=0 起始 index) + N×4B body
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = np.array([0], dtype="<f4")
        with open(out_path, "wb") as f:
            f.write(header.tobytes())
            f.write(vwap_full.astype("<f4").tobytes())

        non_nan = int(np.isfinite(vwap_full).sum())
        if non_nan == 0:
            return (symbol, 2, "all NaN (no tickex data)")

        return (symbol, 0, f"wrote {non_nan}/{len(vwap_full)} non-NaN values")

    except Exception as e:  # noqa: BLE001
        return (symbol, 3, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 主控
# --------------------------------------------------------------------------- #

def list_all_symbols() -> List[str]:
    """从 silver features 目录列出所有 silver 已收录的标的。"""
    if not os.path.isdir(FEATURES_DIR):
        return []
    syms = sorted(
        d for d in os.listdir(FEATURES_DIR)
        if os.path.isdir(os.path.join(FEATURES_DIR, d))
    )
    return syms


def list_tickex_dates() -> List[str]:
    """从 tickex 目录列出所有可用日期。"""
    if not os.path.isdir(TICKEX_DIR):
        return []
    return sorted(
        d.replace("date=", "") for d in os.listdir(TICKEX_DIR) if d.startswith("date=")
    )


def run_parallel(
    symbols: List[str],
    dates: List[str],
    workers: int,
    expected_body_len: int,
    resume: bool,
    hang_timeout: int = 300,
):
    pbar_desc = "patch vwap"
    expected_size = 4 + expected_body_len * 4
    out_dir = Path(FEATURES_DIR)

    tasks = []
    for sym in symbols:
        out_bin = out_dir / sym / f"vwap.{FREQ}.bin"
        if resume and out_bin.exists() and out_bin.stat().st_size == expected_size:
            continue
        tasks.append((sym, out_bin))

    if not tasks:
        print("Nothing to do (all symbols already have valid vwap.1min.bin).")
        return

    print(f"Tasks: {len(tasks)} symbols × {len(dates)} dates, workers={workers}")

    pool = Pool(processes=workers)
    async_results = [
        (sym, pool.apply_async(process_one_symbol, (sym, dates, str(bin_path), expected_body_len, resume)))
        for sym, bin_path in tasks
    ]

    stats = {0: 0, 1: 0, 2: 0, 3: 0}
    with tqdm(total=len(async_results), desc=pbar_desc) as pbar:
        last_remaining = len(async_results)
        stuck = 0.0
        poll = 0.2
        while async_results:
            time.sleep(poll)
            ready_idx = []
            for i, (sym, r) in enumerate(async_results):
                if r.ready():
                    ready_idx.append((i, sym, r))
            for i, sym, r in ready_idx:
                try:
                    sym_r, code, msg = r.get(timeout=0.1)
                    stats[code] = stats.get(code, 0) + 1
                    if code == 3:
                        print(f"  ERROR {sym_r}: {msg}")
                except Exception as e:  # noqa: BLE001
                    stats[3] += 1
                    print(f"  ERROR {sym}: {e}")
                pbar.update()
            ready_set = {x[0] for x in ready_idx}
            async_results = [t for i, t in enumerate(async_results) if i not in ready_set]
            if len(async_results) == last_remaining:
                stuck += poll
                if stuck >= hang_timeout:
                    print(f"No progress {hang_timeout}s — {len(async_results)} hung, terminating")
                    break
            else:
                stuck = 0.0
                last_remaining = len(async_results)
    pool.terminate()
    pool.join()

    print(
        f"\nDone. OK={stats.get(0,0)}  Skipped={stats.get(1,0)}  "
        f"Empty={stats.get(2,0)}  Error={stats.get(3,0)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Patch silver_data_1min with real vwap.1min.bin")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    parser.add_argument("--test", action="store_true", help="Test: 5 symbols only")
    parser.add_argument("--symbols", type=int, default=None, help="limit per-symbol count")
    parser.add_argument("--dates", default="", help="comma-separated date list (default: all)")
    parser.add_argument("--resume", action="store_true", help="skip symbols whose vwap.bin already exists")
    parser.add_argument("--silver-qlib-dir", default=SILVER_QLIB_DIR,
                        help="silver qlib bin 输出根目录 (calendars/instruments/features)")
    args = parser.parse_args()

    # 若修改了 silver_qlib_dir，更新 FEATURES_DIR 常量
    if args.silver_qlib_dir != SILVER_QLIB_DIR:
        global FEATURES_DIR
        FEATURES_DIR = os.path.join(args.silver_qlib_dir, "features")

    # 全局 calendar
    try:
        cal = get_global_calendar()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    expected_body_len = len(cal)
    print(f"Global calendar: {expected_body_len} 1min bars, "
          f"{cal[0]} → {cal[-1]}")

    # 日期候选
    all_dates = list_tickex_dates()
    if args.dates:
        date_list = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        date_list = all_dates
    if not date_list:
        print("No tickex dates available.")
        return
    print(f"Dates: {len(date_list)} ({date_list[0]} → {date_list[-1]})")

    # 标的候选：从 silver features 目录取（已收录过的）
    symbols = list_all_symbols()
    if args.test:
        symbols = symbols[:5]
    elif args.symbols:
        symbols = symbols[: args.symbols]
    if not symbols:
        print("No symbols in silver features dir; run albus_produce_data_full_year.py first.")
        return
    print(f"Symbols: {len(symbols)}")

    run_parallel(
        symbols=symbols,
        dates=date_list,
        workers=args.workers,
        expected_body_len=expected_body_len,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()