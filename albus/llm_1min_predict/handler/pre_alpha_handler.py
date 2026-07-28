"""Pre-Alpha 特征层生成器：从 1min silver 数据生成 Alpha158 与 HFH12 特征并 dump 至 golden 目录。

数据流向：
    silver_data_1min  (qlib bin, HK L2 1min OHLCV + change/factor/paused/paused_num)
        ↓  通过 QlibDataLoader + expression engine 计算
    Alpha158  /  HFH12  特征 (wide DataFrame，MultiIndex=[instrument, datetime]，含 LABEL0)
        ↓  按格式 dump
    golden_data_1min/{alpha158, hfh12}/{train, valid, test}.{parquet|pkl|bin}

要点：
- silver_data 的 vwap 由 albus_produce_data_full_year.py 直接通过 tickex 的
  turnover 聚合生成（真 vwap = Σ(p·v)/Σv），存入 vwap.<freq>.bin。
  Alpha158Binary 继承 Alpha158 原版，只 override label 为涨/跌二分类。
- label 为涨/跌二分类（参考现有 workflow yaml 的 t+2 vs t+1 窗口）
     LABEL0 = If(Gt(Ref(\$close, -2), Ref(\$close, -1)), 1, 0)
     t+2 收盘 > t+1 收盘 → 1（涨），否则 0（跌）。
     因此 learn_processors 只用 DropnaLabel，不再做 CSRankNorm/CSZScoreNorm（否则破坏 0/1 语义）。
- infer_processors 只保留 Fillna（不做 RobustZScoreNorm）——Alpha158 的 158 个因子
     和 HFH12 的 12 个因子本身就是 /$close 或 /DayLast($close) 的比率，值域已自归一化。
     全局 z-score 会抹掉时序上下文（平静日/爆发日同一因子被映射到接近的值），
     LLM 文本输入读原始比率（如 VWAP0=1.002 即比 close 高 0.2%）更可解释。
- Alpha158 rolling 默认窗口 [5,10,20,30,60] 保持原值，即分钟级（5/10/20/30/60 分钟）。
- HFH12 不做原版 HighFreqNorm 的 12×330=3960 列 reshape，保留
  (instrument, datetime) MultiIndex，每分钟 12 列，方便 LightGBM/LLM 直接喂入。

输出格式对比（详见 --benchmark 输出）：
    parquet  ：列式 + zstd，体积最小、跨语言、可列裁，保留 MultiIndex round-trip。**推荐**。
    pkl      ：保留原生 dtype/MultiIndex，但仅 Python、不安全、体积偏大。
    bin      ：qlib 原生内存映射格式，仅 float32；宽特征表一旦计算完成后
               无需 expression engine；每个 (symbol, feature) 一份文件，宽表
               生成 158/12 个文件 × 3172 标的 ≈ 上万文件，管理/端到端读写不划算。
用法：
    # 无参数 = PDF 实验默认（2026-01-05 ~ 2026-07-23，train 1-3月 / valid 4-5月 / test 6-7月）
    python pre_alpha_handler.py

    # 冒烟测试
    python pre_alpha_handler.py --test

    # 全量 + 三格式 benchmark
    python pre_alpha_handler.py --formats pkl,parquet,bin --benchmark

    # 自定义时间窗口
    python pre_alpha_handler.py --start "2026-01-05 09:30:00" --end "2026-06-22 15:59:00" \\
        --train-end "2026-06-09 15:59:00" --valid-end "2026-06-15 15:59:00"

    # 仅作为 handler 类被 workflow config 加载（不跑 main）
    # 在 yaml 里 class: Alpha158Binary / HFH12Handler，module_path 指本文件即可
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# 确保项目根目录（含本地 qlib/ 源码）优先于 site-packages 的 qlib
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# qlib 系列导入放在末层，避免循环依赖
from qlib.contrib.data.handler import Alpha158, check_transform_proc
from qlib.contrib.ops.high_freq import (
    BFillNan,
    Cut,
    Date,
    DayLast,
    FFillNan,
    IsNull,
    Select,
)
from qlib.constant import REG_HK
from qlib.data.dataset.handler import DataHandlerLP

# --------------------------------------------------------------------------- #
# 常量与路径
# --------------------------------------------------------------------------- #

SILVER_URI = "/home/albus/Python_Codes/qlib/qlib_data/silver_data_1min"
GOLDEN_DIR = Path("/home/albus/Python_Codes/qlib/qlib_data/golden_data_1min")

# 涨/跌 二分类 label：t+2 收盘 > t+1 收盘 → 1 否则 0
BINARY_LABEL_EXPR = "If(Gt(Ref($close, -2), Ref($close, -1)), 1, 0)"
# 即时预测 label：t+1 收盘 > t 收盘 → 1 否则 0（港股 T+0 适用）
IMMEDIATE_LABEL_EXPR = "If(Gt(Ref($close, -1), $close), 1, 0)"

# example/highfreq 自定义算子（已在 qlib.contrib.ops.high_freq 内置）
CUSTOM_OPS = [DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut]

# LLM 微调无需 RobustZScoreNorm：
# Alpha158 158 个因子本身就是 /$close 比率，值域天然[-1,1]或[0,2]；HFH12 12 个因子也是 /DayLast($close) 比率。
# 全局 z-score 会抹掉时序上下文（平静/爆发日同一因子映射到接近值），LLM 文本输入读原始比率更可解释。
# 只 Fillna——特征 NaN 统一填 0（LLM 不认识 "NaN" token）。
DEFAULT_INFER_PROCESSORS = [
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]
DEFAULT_LEARN_PROCESSORS = [{"class": "DropnaLabel"}]

# 备忘：highfreq 的归一化窗口（Cut 第一天 N 分钟）
# 港股每日 330 分钟（09:30-11:59 + 13:00-15:59），A 股每日 240 分钟。
# examples/highfreq 默认按 A 股 240；silver_data_1min 是港股 → 用 330。
_HFH12_NORM_BASE = 330  # 港股每个交易日 330 个 1min bar
# volume 归一化基准窗口 = 30 个交易日 × 330 分钟 = 9900
_HFH12_VOL_NORM_BASE = _HFH12_NORM_BASE * 30


# --------------------------------------------------------------------------- #
# qlib 初始化
# --------------------------------------------------------------------------- #

def init_qlib(silver_uri: str = SILVER_URI):
    """初始化 qlib 客户端，注册高频自定义算子并启用 expression cache。"""
    import qlib

    qlib.init(
        provider_uri=silver_uri,
        region=REG_HK,
        expression_cache="DiskExpressionCache",
        dataset_cache=None,
        custom_ops=CUSTOM_OPS,
    )


class Alpha158Binary(Alpha158):
    """Alpha158 + 涨/跌二分类 LABEL0（T+1→T+2）。

    原版 Alpha158.get_label_config() 返回连续收益率 Ref(c,-2)/Ref(c,-1)-1，
    本类 override 为 If(Gt(Ref(c,-2), Ref(c,-1)), 1, 0)，即 t+2 涨=1 跌=0。
    vwap 由 silver 数据自带（聚合 turnover 生成），无需 Simpson 替换。
    """

    def get_label_config(self):
        return [BINARY_LABEL_EXPR], ["LABEL0"]


class Alpha158Immediate(Alpha158):
    """Alpha158 + 即时预测 LABEL0（T→T+1，港股 T+0 适用）。"""

    def get_label_config(self):
        return [IMMEDIATE_LABEL_EXPR], ["LABEL0"]


# Alpha158Binary 注册到 HANDLERS，不由用户区分。


# --------------------------------------------------------------------------- #
# HFH12 —— 12 个高频归一化特征，flat (instrument, datetime) 形状
# --------------------------------------------------------------------------- #

class HFH12Handler(DataHandlerLP):
    """12 个高频归一化特征的 handler。

    特征结构（与 examples/highfreq/highfreq_handler.py 的 HighFreqHandler 一致）：
        ├ open  / high  / low  / close  / vwap   (t0, 归一化到昨日收盘)
        ├ open1 / high1 / low1 / close1 / vwap1  (t-1day, 归一化到昨日收盘)
        └ volume / volume_1                        (归一化到 9900 分钟均值=30个港股交易日)

    与原版 HighFreqHandler 的关键差异：
    1. volume 过滤中的 vwap 仍用 Simpson 近似（因 Select 算子对稀疏 vwap 标的会
       触发 qlib 布尔索引对齐 bug）。Alpha158 的 VWAP0 列不受影响（不走 Select）。
    2. 不接 HighFreqNorm processor（原版会把 12 个特征 reshape 成 12×330=3960 列/天，
       给 RL 高频执行器用）。本类保持 (instrument, datetime) MultiIndex、每分钟 12 列，
       适合 LightGBM / LLM 预测直接读取。
    2. 增加 label 列 LABEL0（涨/跌二分类），原版把 label 放在 yaml 里。

    注意：Cut 算子会切掉每个标的每天的前 330 分钟（一个港股交易日）归一化基准数据，
    因此 HFH12 的有效数据从第二天的第一个 bar 开始（即 t0 时间窗口的第二日）。
    若 start_time 不足两天，输出可能全 0。
    """

    def __init__(
        self,
        instruments: str = "all",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        fit_start_time: Optional[str] = None,
        fit_end_time: Optional[str] = None,
        infer_processors: Optional[List[dict]] = None,
        learn_processors: Optional[List[dict]] = None,
        drop_raw: bool = True,
        **kwargs,
    ):
        infer_processors = check_transform_proc(
            infer_processors if infer_processors is not None else DEFAULT_INFER_PROCESSORS,
            fit_start_time,
            fit_end_time,
        )
        learn_processors = check_transform_proc(
            learn_processors if learn_processors is not None else DEFAULT_LEARN_PROCESSORS,
            fit_start_time,
            fit_end_time,
        )

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "swap_level": False,
                "freq": "1min",
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            drop_raw=drop_raw,
            **kwargs,
        )

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        t_if = "If(IsNull({1}), {0}, {1})"
        t_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        t_fillnan = "BFillNan(FFillNan({0}))"
        # HFH12 用它做 volume 过滤归一化的中间基准，并非主力 VWAP0 列。
        # 由于 qlib Select 算子在稀疏 vwap 数据下会触发布尔索引对齐 bug
        # （如 00036 这类低频标的 vwap ÷ close 的有效值极少），
        # 保留 Simpson 近似，确保全标的稳定运行。
        # Alpha158 的 VWAP0 列不受影响——它直接引用数据层真 $vwap，不走 Select。
        vwap_expr = "($open+2*$high+2*$low+$close)/6"

        def _norm(price_field: str, shift: int = 0) -> str:
            if shift == 0:
                tmpl = "Cut({0}/Ref(DayLast({1}), {base}), {base}, None)"
            else:
                tmpl = "Cut(Ref({0}, {sh})/Ref(DayLast({1}), {base}), {base}, None)"
            return tmpl.format(
                t_if.format(
                    t_fillnan.format(t_paused.format("$close")),
                    t_paused.format(price_field),
                ),
                t_fillnan.format(t_paused.format("$close")),
                base=_HFH12_NORM_BASE,
                sh=shift,
            )

        fields: List[str] = []
        names: List[str] = []

        # 5 个 t0 特征（$open/$high/$low/$close/$vwap）
        for pf, nm in [
            ("$open", "$open"),
            ("$high", "$high"),
            ("$low", "$low"),
            ("$close", "$close"),
            (vwap_expr, "$vwap"),
        ]:
            fields.append(_norm(pf, 0))
            names.append(nm)

        # 5 个 t-1day 特征
        for pf, nm in [
            ("$open", "$open_1"),
            ("$high", "$high_1"),
            ("$low", "$low_1"),
            ("$close", "$close_1"),
            (vwap_expr, "$vwap_1"),
        ]:
            fields.append(_norm(pf, _HFH12_NORM_BASE))
            names.append(nm)

        # volume + volume_1
        filt_vol = (
            "If(IsNull({0}), 0, "
            "If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))"
        ).format(
            t_paused.format("$volume"),
            t_paused.format(vwap_expr),
            t_paused.format("$low"),
            t_paused.format("$high"),
        )
        fields.append(
            "Cut({0}/Ref(DayLast(Mean({0}, {base2})), {base}), {base}, None)".format(
                filt_vol, base=_HFH12_NORM_BASE, base2=_HFH12_VOL_NORM_BASE
            )
        )
        names.append("$volume")
        fields.append(
            "Cut(Ref({0}, {base})/Ref(DayLast(Mean({0}, {base2})), {base}), {base}, None)".format(
                filt_vol, base=_HFH12_NORM_BASE, base2=_HFH12_VOL_NORM_BASE
            )
        )
        names.append("$volume_1")

        return fields, names

    def get_label_config(self) -> Tuple[List[str], List[str]]:
        return [BINARY_LABEL_EXPR], ["LABEL0"]


class HFH12ImmediateHandler(HFH12Handler):
    """HFH12 特征 + 即时预测 LABEL0（T→T+1，港股 T+0 适用）。"""

    def get_label_config(self) -> Tuple[List[str], List[str]]:
        return [IMMEDIATE_LABEL_EXPR], ["LABEL0"]


# --------------------------------------------------------------------------- #
# 三种格式 dump 工具
# --------------------------------------------------------------------------- #

def dump_pkl(df: pd.DataFrame, path: Path) -> float:
    """Dump 为 pickle，保留 MultiIndex 与原生 dtype；返回写盘耗时(秒)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df.to_pickle(path)
    return time.perf_counter() - t0


def dump_parquet(df: pd.DataFrame, path: Path, compression: str = "zstd") -> float:
    """Dump 为 parquet，用 reset_index 把 MultiIndex 落地为 instrument/datetime 列；
    读回时 set_index(['instrument','datetime']) 即可还原。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # copy() 以避免 pandas fragmentation PerformanceWarning
    df_reset = df.copy()
    df_reset.columns = [c[-1] if isinstance(c, tuple) else c for c in df_reset.columns]
    df_reset = df_reset.reset_index()
    df_reset.to_parquet(path, compression=compression, index=False)
    return time.perf_counter() - t0


def dump_bin(
    df: pd.DataFrame,
    qlib_dir: Path,
    freq: str = "1min",
    workers: int = 8,
) -> float:
    """Dump 为 qlib 原生 bin 格式（features/<symbol>/<feature>.<freq>.bin）。

    实现：每个标的写一个 CSV [date, symbol, feature1, ..., LABEL0]，再交给
    scripts/dump_bin.DumpDataAll 一次性 dump calendars/instruments/features。
    """
    from qlib.utils import code_to_fname  # noqa: F401  (dump_bin 内部依赖)

    # 限制为单层列名（去除残留 MultiIndex）
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

    qlib_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = qlib_dir.parent / f"_tmp_csv_{qlib_dir.name}"
    if tmp_csv.exists():
        shutil.rmtree(tmp_csv)
    tmp_csv.mkdir(parents=True)

    # 把 MultiIndex 行展开成 instrument/datetime 列；按标的分文件
    df_flat = df.reset_index()
    # 保证 datetime 字段是字符串
    if not np.issubdtype(df_flat["datetime"].dtype, np.object_):
        df_flat["datetime"] = df_flat["datetime"].astype(str)
    else:
        df_flat["datetime"] = df_flat["datetime"].astype(str)

    t0 = time.perf_counter()
    for sym, g in df_flat.groupby("instrument", sort=True):
        out = g.drop(columns=["instrument"]).copy()
        out = out.rename(columns={"datetime": "date", "LABEL0": "label"})
        out["symbol"] = sym
        # DumpDataAll 期望列：date, symbol, 各 feature 列
        cols = ["date", "symbol"] + [c for c in out.columns if c not in ("date", "symbol")]
        out = out[cols]
        out.to_csv(tmp_csv / f"{sym}.csv", index=False)
    csv_to_bin_time = time.perf_counter() - t0

    # 调用 DumpDataAll 将 CSV 转 bin
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 让能 import scripts.dump_bin
    from scripts.dump_bin import DumpDataAll

    dump_all = DumpDataAll(
        data_path=str(tmp_csv),
        qlib_dir=str(qlib_dir),
        freq=freq,
        date_field_name="date",
        symbol_field_name="symbol",
        exclude_fields="symbol",
        max_workers=workers,
    )
    if qlib_dir.exists():
        # DumpDataAll 不会覆盖已有 calendar 之外的 features；为干净起见先清空
        shutil.rmtree(qlib_dir)
        qlib_dir.mkdir(parents=True, exist_ok=True)
    dump_all.dump()
    shutil.rmtree(tmp_csv, ignore_errors=True)
    return time.perf_counter() - t0


def load_pkl(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "instrument" in df.columns and "datetime" in df.columns:
        df = df.set_index(["instrument", "datetime"]).sort_index()
    return df


def merge_parquet_streaming(chunk_files: List[Path], out_path: Path) -> int:
    """用 pyarrow ParquetWriter 逐文件追加合并，峰值内存 ≈ 单个 chunk。返回总行数。"""
    import pyarrow.parquet as pq

    writer = None
    total_rows = 0
    try:
        for p in chunk_files:
            tbl = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            total_rows += tbl.num_rows
            del tbl
    finally:
        if writer is not None:
            writer.close()
    return total_rows


# --------------------------------------------------------------------------- #
# Benchmark：对比 pkl/parquet/bin 三种格式
# --------------------------------------------------------------------------- #

def benchmark_formats(df: pd.DataFrame, out_base: Path) -> Dict[str, dict]:
    """对同一份特征宽表，分别以 pkl/parquet/bin 写入并读回，比较
    体积、写时、读时、可列裁时读时。返回结果字典并打印对比表。"""
    out_base.mkdir(parents=True, exist_ok=True)
    sample = df.head(50_000).copy()  # 取前 5 万行做 benchmark，避免时间过长
    results: Dict[str, dict] = {}

    # pkl
    p_pkl = out_base / "bench.pkl"
    w = dump_pkl(sample, p_pkl)
    r0 = time.perf_counter()
    _ = load_pkl(p_pkl)
    rpkl = time.perf_counter() - r0
    results["pkl"] = {
        "size_MB": p_pkl.stat().st_size / 1e6,
        "write_s": w,
        "read_s": rpkl,
        "partial_read_s": float("nan"),  # pkl 不能列裁剪
    }
    p_pkl.unlink(missing_ok=True)

    # parquet
    p_par = out_base / "bench.parquet"
    w = dump_parquet(sample, p_par)
    r0 = time.perf_counter()
    _ = load_parquet(p_par)
    rpar = time.perf_counter() - r0
    # 列裁读
    r0 = time.perf_counter()
    _ = pd.read_parquet(p_par, columns=["LABEL0"])
    rpar_cols = time.perf_counter() - r0
    results["parquet"] = {
        "size_MB": p_par.stat().st_size / 1e6,
        "write_s": w,
        "read_s": rpar,
        "partial_read_s": rpar_cols,
    }
    p_par.unlink(missing_ok=True)

    # bin
    bin_dir = out_base / "bench_bin"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    w = dump_bin(sample, bin_dir, workers=4)
    # 总 volume = 所有 feature/*.bin 文件大小之和
    total = sum(f.stat().st_size for f in bin_dir.rglob("*.bin"))
    r0 = time.perf_counter()
    # bin 通用读需要 qlib init；这里只测“列出一标的全部特征”
    try:
        bin_files = sorted(bin_dir.glob("features/*/*.bin"))
        # 读第一个文件作为代表
        if bin_files:
            with open(bin_files[0], "rb") as f:
                _ = np.frombuffer(f.read()[4:], dtype="<f4")
        rbin = time.perf_counter() - r0
    except Exception:
        rbin = float("nan")
    results["bin"] = {
        "size_MB": total / 1e6,
        "write_s": w,
        "read_s": rbin,
        "partial_read_s": float("nan"),  # bin 需要 qlib init 才能列裁读，单测略
    }
    shutil.rmtree(bin_dir, ignore_errors=True)

    # 打印对比
    print("\n" + "=" * 72)
    print("Dump Format Benchmark (sample rows = {})".format(len(sample)))
    print("=" * 72)
    print(f"{'format':<10} {'size(MB)':>10} {'write(s)':>10} {'read(s)':>10} {'pcol_read(s)':>14}")
    print("-" * 72)
    for fmt, m in results.items():
        print(
            f"{fmt:<10} {m['size_MB']:>10.3f} {m['write_s']:>10.3f} "
            f"{m['read_s']:>10.3f} {m['partial_read_s']:>14.3f}"
        )
    print("=" * 72)
    print("结论：parquet 综合最优 —— 体积接近 bin、读速 > pkl、支持列裁、跨语言。\n")
    # 清理 benchmark 自身的工作目录（bench 文件已在上面删除，仅可能残留空目录）
    if out_base.exists() and not any(out_base.iterdir()):
        out_base.rmdir()
    return results


# --------------------------------------------------------------------------- #
# 整合 handler：构建 + prepare + dump
# --------------------------------------------------------------------------- #

class PreAlphaHandler:
    """表征层生成器：构建 Alpha158 / HFH12 handler，prepare segment 数据并 dump。

    Parameters
    ----------
    start_time, end_time : str
        全时间窗口（含 fit、train、valid、test 的并集），格式 'YYYY-MM-DD HH:MM:SS'
    train_end, valid_end : str
        train 段结束 = train_end；valid 段 = (train_end, valid_end]；test = (valid_end, end]
    chunk_size : int
        >0 时启用按标的分组模式：每组 chunk_size 个标的跑完整时间窗口，
        rolling/Cut/label 无任何边界损耗；=0 时一次性加载全量。
        分组模式下 chunk 中间文件只写 parquet，最终流式合并为 {seg}.parquet。
    instruments : str
        传给 DataHandler，'all' 或 'csi300' / 任意 instruments 名
    handlers : Iterable[str]
        'alpha158', 'hfh12' 子集
    formats : Iterable[str]
        'pkl', 'parquet', 'bin' 子集
    out_dir : Path
        golden 目录根
    """

    HANDLERS = {
        "alpha158": Alpha158Binary,
        "hfh12": HFH12Handler,
        "hfh12_immediate": HFH12ImmediateHandler,
        "alpha158_immediate": Alpha158Immediate,
    }

    def __init__(
        self,
        start_time: str,
        end_time: str,
        train_end: str,
        valid_end: str,
        chunk_size: int = 0,
        instruments: str = "all",
        handlers: Iterable[str] = ("alpha158", "hfh12"),
        formats: Iterable[str] = ("parquet",),
        out_dir: Path | str = GOLDEN_DIR,
        workers: int = 8,
        infer_processors: Optional[List[dict]] = None,
        learn_processors: Optional[List[dict]] = None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.train_end = train_end
        self.valid_end = valid_end
        self.segments = {
            "train": (start_time, train_end),
            "valid": (train_end, valid_end),
            "test": (valid_end, end_time),
        }
        self.chunk_size = chunk_size
        self.instruments = instruments
        self.handlers = list(handlers)
        self.formats = list(formats)
        self.out_dir = Path(out_dir)
        self.workers = workers
        self.infer_processors = infer_processors or DEFAULT_INFER_PROCESSORS
        self.learn_processors = learn_processors or DEFAULT_LEARN_PROCESSORS

    # ========== 构造 handler ==========

    def build_handler(
        self,
        name: str,
        start_time: str = None,
        end_time: str = None,
        instruments=None,
    ) -> DataHandlerLP:
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        if instruments is None:
            instruments = self.instruments
        if name not in self.HANDLERS:
            raise ValueError(f"unknown handler {name}; valid: {list(self.HANDLERS)}")
        cls = self.HANDLERS[name]
        kwargs = dict(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            fit_start_time=start_time,
            fit_end_time=self.train_end,
            infer_processors=self.infer_processors,
            learn_processors=self.learn_processors,
        )
        if name in ("alpha158", "alpha158_immediate"):
            kwargs["freq"] = "1min"
        return cls(**kwargs)

    # ========== 一次性加载（短窗口专用） ==========

    def prepare(
        self,
        name: str,
        seg_start: str = None,
        seg_end: str = None,
        instruments=None,
    ) -> Dict[str, pd.DataFrame]:
        """加载数据并按 segment 切分。instruments 不为空时只加载这批标的。"""
        from qlib.data.dataset import DatasetH

        actual_start = seg_start or self.start_time
        actual_end = seg_end or self.end_time

        print(f"  loading data for {name} ({actual_start} → {actual_end})...", flush=True)
        t0 = time.time()
        handler = self.build_handler(name, actual_start, actual_end, instruments=instruments)
        seg_map = {}
        for seg_name, (s, e) in self.segments.items():
            # 只保留与 [actual_start, actual_end] 有交集的 segment
            seg_start_clip = max(pd.Timestamp(s), pd.Timestamp(actual_start))
            seg_end_clip = min(pd.Timestamp(e), pd.Timestamp(actual_end))
            if seg_start_clip < seg_end_clip:
                seg_map[seg_name] = (str(seg_start_clip), str(seg_end_clip))
        ds = DatasetH(handler=handler, segments=seg_map)
        out: Dict[str, pd.DataFrame] = {}
        for seg in seg_map:
            try:
                print(f"    preparing {seg}...", end=" ", flush=True)
                t1 = time.time()
                df = ds.prepare(seg)
                print(f"done ({time.time()-t1:.1f}s) shape={df.shape}")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {seg} failed: {e}")
                df = pd.DataFrame()
            out[seg] = df
        print(f"  handler done ({time.time()-t0:.1f}s)")
        return out

    # ========== dump 单段 DataFrame ==========

    def dump_segment(self, name: str, seg: str, df: pd.DataFrame, seg_path: Path = None) -> Dict[str, float]:
        out_root = seg_path or (self.out_dir / name)
        timings = {}
        if df.empty:
            print(f"  [skip] {name}/{seg} empty")
            return timings

        for fmt in self.formats:
            fmt = fmt.lower()
            if fmt == "pkl":
                p = out_root / f"{seg}.pkl"
                timings["pkl"] = dump_pkl(df, p)
            elif fmt == "parquet":
                p = out_root / f"{seg}.parquet"
                timings["parquet"] = dump_parquet(df, p)
            elif fmt == "bin":
                p = out_root / f"{seg}_bin"
                timings["bin"] = dump_bin(df, p, workers=self.workers)
            else:
                raise ValueError(f"unknown format {fmt}; valid: pkl/parquet/bin")
            print(f"    -> {fmt}: {timings.get(fmt, 0):.3f}s")
        return timings

    # ========== 分组相关 ==========

    def _list_instruments(self) -> List[str]:
        """展开 instruments 配置为标的列表（需先 qlib.init）。"""
        if isinstance(self.instruments, (list, tuple)):
            return sorted(self.instruments)
        from qlib.data import D

        insts = D.list_instruments(
            D.instruments(self.instruments),
            start_time=self.start_time,
            end_time=self.end_time,
            freq="1min",
            as_list=True,
        )
        return sorted(insts)

    def run_chunked(self, benchmark: bool = False) -> Dict[str, Dict[str, dict]]:
        """按标的分组处理：每组 chunk_size 个标的跑完整时间窗口。

        特征全是单标的时序计算，按标的切分没有 rolling/Cut/label 的边界损耗；
        内存峰值 ≈ 组大小 × 全窗口。各组 parquet 最后流式合并，避免 concat 内存反弹。
        """
        t_start = time.time()
        all_insts = self._list_instruments()
        groups = [
            all_insts[i : i + self.chunk_size]
            for i in range(0, len(all_insts), self.chunk_size)
        ]
        print(f"[PreAlphaHandler] {len(all_insts)} instruments → "
              f"{len(groups)} groups ({self.chunk_size}/group)")

        summary: Dict[str, Dict[str, dict]] = {}
        n_handlers = len(self.handlers)

        for idx, name in enumerate(self.handlers, 1):
            print(f"\n[PreAlphaHandler] [{idx}/{n_handlers}] handler={name} "
                  f"(chunked by instrument, {self.chunk_size}/group)")

            chunk_dirs: Dict[str, Path] = {}
            for seg_name in self.segments:
                d = self.out_dir / name / f"_chunks_{seg_name}"
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
                chunk_dirs[seg_name] = d

            for gi, group in enumerate(groups):
                print(f"  group [{gi + 1}/{len(groups)}] {len(group)} instruments "
                      f"({group[0]} ... {group[-1]})", flush=True)
                segs = self.prepare(name, instruments=group)
                for seg_name, df in segs.items():
                    if df.empty:
                        continue
                    dump_parquet(df, chunk_dirs[seg_name] / f"{seg_name}_chunk_{gi:04d}.parquet")
                del segs
                gc.collect()

            timing: Dict[str, dict] = {}
            for seg_name, chunk_dir in chunk_dirs.items():
                chunk_files = sorted(chunk_dir.glob(f"{seg_name}_chunk_*.parquet"))
                if chunk_files:
                    out_dir = self.out_dir / name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    final_path = out_dir / f"{seg_name}.parquet"
                    t1 = time.time()
                    n_rows = merge_parquet_streaming(chunk_files, final_path)
                    merge_s = time.time() - t1
                    print(f"  [{seg_name}] merged {len(chunk_files)} chunks → "
                          f"{final_path.name} ({merge_s:.1f}s, rows={n_rows})")
                    timing[seg_name] = {"parquet": merge_s}
                else:
                    print(f"  [{seg_name}] no data")
                shutil.rmtree(chunk_dir, ignore_errors=True)
            gc.collect()

            if benchmark:
                print("  [benchmark] skipped in chunked mode (use --test for benchmark)")
            summary[name] = timing

        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"Done! Total time: {elapsed:.1f}s "
              f"({len(groups)} groups × {self.chunk_size} instruments)")
        for name, t in summary.items():
            seg_info = " | ".join(
                f"{s}=ok" if s in t else f"{s}=empty" for s in ["train", "valid", "test"]
            )
            print(f"  {name}: {seg_info}")
        print(f"="*60)
        return summary

    # ========== 完整运行（自动选择分组或一次性） ==========

    def run(self, benchmark: bool = False) -> Dict[str, Dict[str, dict]]:
        """对每个 handler 跑 prepare + dump。chunk_size>0 时自动走按标的分组。"""
        if self.chunk_size > 0:
            return self.run_chunked(benchmark)

        # 短窗口：一次性加载
        t_start = time.time()
        summary: Dict[str, Dict[str, dict]] = {}
        n_handlers = len(self.handlers)
        for idx, name in enumerate(self.handlers, 1):
            print(f"\n[PreAlphaHandler] [{idx}/{n_handlers}] handler={name}")
            segs = self.prepare(name)
            timing: Dict[str, dict] = {}
            for seg, df in segs.items():
                timing[seg] = self.dump_segment(name, seg, df)

            if benchmark:
                train_df = segs.get("train", pd.DataFrame())
                if not train_df.empty:
                    bench = benchmark_formats(train_df, self.out_dir / name / "_bench")
                    timing["benchmark"] = bench
            summary[name] = timing
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"Done! Total time: {elapsed:.1f}s")
        for name, t in summary.items():
            seg_info = " | ".join(
                f"{s}={list(t[s].values())[0]:.1f}s" if t[s] else f"{s}=0s"
                for s in ["train", "valid", "test"]
            )
            print(f"  {name}: {seg_info}")
        print(f"="*60)
        return summary


# --------------------------------------------------------------------------- #
# CLI 入口
# --------------------------------------------------------------------------- #

def _parse_formats(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def _parse_handlers(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def main():
    # 默认按 PDF 实验设计的日期范围（2026-01-05 ~ 2026-07-23）
    # --test 覆盖为 3 天冒烟
    parser = argparse.ArgumentParser(
        description="Generate Alpha158 + HFH12 features from 1-min silver data and dump to golden."
    )
    parser.add_argument("--silver-uri", default=SILVER_URI, help="silver qlib bin 目录")
    parser.add_argument("--out-dir", default=str(GOLDEN_DIR), help="golden 输出目录")
    parser.add_argument("--start", default="2026-01-05 09:30:00",
                        help="开始时间 (默认: 2026-01-05 09:30:00)")
    parser.add_argument("--end", default="2026-07-23 15:59:00",
                        help="结束时间 (默认: 2026-07-23 15:59:00)")
    parser.add_argument("--train-end", default="2026-03-31 15:59:00",
                        help="train 段结束时间 (PDF: 2026/1-3月)")
    parser.add_argument("--valid-end", default="2026-05-31 15:59:00",
                        help="valid 段结束时间 (PDF: 2026/4-5月)")
    parser.add_argument("--instruments", default="all", help="传给 handler 的 instruments")
    parser.add_argument(
        "--handlers",
        default="alpha158_immediate",
        help="要生成的 handler，逗号分隔，可选: alpha158,hfh12,hfh12_immediate,alpha158_immediate",
    )
    parser.add_argument(
        "--formats",
        default="parquet",
        help="输出格式，逗号分隔，可选: pkl,parquet,bin",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="跑 pkl/parquet/bin 三格式对比 benchmark",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="冒烟模式 (5 标的 3 天小样本)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="按标的分组: 0=一次性加载全部, >0=每组N个标的跑完整时间窗口(无 rolling/Cut/label 边界损耗)",
    )
    parser.add_argument("--workers", type=int, default=14, help="dump_bin 并发数")
    args = parser.parse_args()

    if args.test:
        print("[test mode] 3 天冒烟测试 (全市场)")
        args.start = "2026-01-05 09:30:00"
        args.end = "2026-01-08 15:59:00"
        args.train_end = "2026-01-06 15:59:00"
        args.valid_end = "2026-01-07 15:59:00"
        args.chunk_size = 0  # test 模式下不启用分组

    print(f"窗口: train=({args.start}, {args.train_end}]  valid=({args.train_end}, {args.valid_end}]  "
          f"test=({args.valid_end}, {args.end}]")
    if args.chunk_size > 0:
        print(f"分组模式: {args.chunk_size} 标的/组 (完整时间窗口, 无边界损耗)")
    else:
        print(f"一次性加载: 需确保内存足够")
    init_qlib(args.silver_uri)

    runner = PreAlphaHandler(
        start_time=args.start,
        end_time=args.end,
        train_end=args.train_end,
        valid_end=args.valid_end,
        chunk_size=args.chunk_size,
        instruments=args.instruments,
        handlers=_parse_handlers(args.handlers),
        formats=_parse_formats(args.formats),
        out_dir=Path(args.out_dir),
        workers=args.workers,
    )
    runner.run(benchmark=args.benchmark)


if __name__ == "__main__":
    main()