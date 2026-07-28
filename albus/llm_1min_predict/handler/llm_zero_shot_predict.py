"""Qwen3.5-0.8B zero-shot 1min 涨/跌预测（PDF 实验组 A：基线零样本）。

数据流向：
    golden_data_1min/{hfh12|alpha158|hfh12_immediate|alpha158_immediate}/{test}.parquet
        ↓  按行采样 → 因子数值转文本 prompt（保留 3 位小数）
    Qwen3.5-0.8B (transformers, CPU/GPU, zero-shot)
        ↓  输出 JSON {"direction": "up"|"down", "confidence": 0-1}
     output/exp_YYYYMMDD_HHMMSS/{out}_detail.csv / {out}_summary.csv / {out}.log
     (Accuracy / F1 / Precision / Recall / Avg_Time)

与 PDF 的差异：不是 30 个 L2 微观结构因子，而是 HFH12（12 个高频归一化因子，默认）
或 Alpha158（158 因子）。支持两组 label 版本：
  - hfh12 / alpha158:  LABEL0 = If(Gt(Ref($close,-2), Ref($close,-1)), 1, 0)
                       即 t+2 收盘 > t+1 收盘 → 1（涨）。
  - hfh12_immediate / alpha158_immediate:  LABEL0 = If(Gt(Ref($close,-1), $close), 1, 0)
                       即 t+1 收盘 > t 收盘 → 1（涨，港股 T+0 适用）。

用法（conda qlib 环境）：
    python llm_zero_shot_predict.py                                                   # 默认：hfh12_immediate, test, 全量, 4只
    python llm_zero_shot_predict.py --handler hfh12
    python llm_zero_shot_predict.py --sample 50 --segment valid
    python llm_zero_shot_predict.py --model Qwen/Qwen3.5-0.8B --csv-out my_result
    python llm_zero_shot_predict.py --time-slot both                                   # 4只×开收盘30min
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

GOLDEN_DIR = Path("/home/albus/Python_Codes/qlib/qlib_data/golden_data_1min")
OUTPUT_BASE = Path(__file__).resolve().parent / "output" / "zero-shot_output"
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_LABEL = "Qwen3.5-0.8B (zero-shot)"
DEFAULT_NUM_WORKERS = 14


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("zero_shot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

# --------------------------------------------------------------------------- #
# 因子释义（写进 prompt，应对 PDF“基座模型不懂因子含义”挑战）
# --------------------------------------------------------------------------- #

HFH12_DESC = {
    "$open": "当日该分钟开盘价 / 昨日收盘价",
    "$high": "当日该分钟最高价 / 昨日收盘价",
    "$low": "当日该分钟最低价 / 昨日收盘价",
    "$close": "当日该分钟收盘价 / 昨日收盘价",
    "$vwap": "当日该分钟 Simpson 近似 VWAP=$(open+2·high+2·low+close)/6 / 昨日收盘价",
    "$open_1": "昨日同一分钟开盘价 / 昨日收盘价",
    "$high_1": "昨日同一分钟最高价 / 昨日收盘价",
    "$low_1": "昨日同一分钟最低价 / 昨日收盘价",
    "$close_1": "昨日同一分钟收盘价 / 昨日收盘价",
    "$vwap_1": "昨日同一分钟 Simpson 近似 VWAP / 昨日收盘价",
    "$volume": "当日该分钟成交量（异常tick先置0） / 近30个港股交易日分钟均量",
    "$volume_1": "昨日同一分钟成交量（异常tick先置0） / 近30个港股交易日分钟均量",
}

ALPHA158_LEGEND = (
    "Alpha158 因子分 3 类："
    "A) K-bar 9 个：KMID/KLEN/KMID2=实体/全幅比，"
    "KUP/KUP2=上影比，KLOW/KLOW2=下影比，KSFT/KSFT2=收盘偏离度；"
    "B) Raw Price 4 个：OPEN0/HIGH0/LOW0/VWAP0 = 当前分钟开/高/低/VWAP ÷ 收盘价；"
    "C) Rolling 29 类 × 5 窗口 [5,10,20,30,60 分钟]=145 个："
    "趋势组(ROC/MA/STD/BETA/RSQR/RESI/MAX/MIN)，"
    "分位组(QTLU/QTLD/RANK/RSV)，"
    "Aroon组(IMAX/IMIN/IMXD)，价量相关(CORR/CORD)，"
    "涨跌频次(CNTP/CNTN/CNTD)，RSI类(SUMP/SUMN/SUMD)，"
    "量能组(VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD)。"
    "数字后缀=窗口长度(分钟)，多数因子已 ÷当前收盘价去单位。"
)

PROMPT_HEADER = (
    "你是一位高频量化交易专家。以下是当前时刻（{timestamp}）"
    "港股 {instrument} 的 {n} 个高频因子数值（保留3位小数，数值精度很重要）：\n"
)

PROMPT_FOOTER = (
    "\n请根据以上数据，预测下一分钟 {instrument} 的价格方向（涨/跌）。\n"
    '请只输出JSON格式: {{"direction": "up" 或 "down", "confidence": 0-1之间的数值}}'
)


def build_prompt(row: pd.Series, feature_cols: list[str], handler: str) -> str:
    instrument, timestamp = row.name[0], str(row.name[1])
    lines = [PROMPT_HEADER.format(timestamp=timestamp, instrument=instrument, n=len(feature_cols))]
    if handler in ("alpha158", "alpha158_immediate"):
        lines.append(ALPHA158_LEGEND + "\n")
        for i, c in enumerate(feature_cols, 1):
            lines.append(f"{i}. {c}: {row[c]:.3f}")
    else:
        for i, c in enumerate(feature_cols, 1):
            desc = HFH12_DESC.get(c, "")
            lines.append(f"{i}. {c}（{desc}）: {row[c]:.3f}")
    lines.append(PROMPT_FOOTER.format(instrument=instrument))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 模型加载与推理（参考 peft_compare.py 的 z35 分支）
# --------------------------------------------------------------------------- #

def load_model(name_or_path: str, device: str = "cpu", dtype: str = "float32"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    with tqdm(total=3, desc="Loading model", disable=False) as pbar:
        pbar.set_description_str("Loading model: tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        pbar.update(1)

        pbar.set_description_str("Loading model: config")
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        text_config = getattr(config, "text_config", None)
        pbar.update(1)

        pbar.set_description_str("Loading model: weights")
        # 多模态模型（如 Qwen3.5-0.8B）含 text_config 时，传 text_config 给
        # AutoModelForCausalLM.from_pretrained，只构造纯文本 Qwen3_5ForCausalLM，
        # 跳过 vision/patch_merger 等视觉权重，减少加载时间与内存占用。
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            config=text_config if text_config is not None else config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model.to(device)
        model.eval()
        pbar.update(1)

    return model, tokenizer


def _format_prompt(tokenizer, prompt: str) -> str:
    """对单个 prompt 套 chat template，兼容 enable_thinking 不支持的旧版。"""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except (TypeError, ValueError):
        # 旧版模板不支持 enable_thinking 参数
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def inference(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    import torch

    formatted = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def inference_batch(model, tokenizer, prompts: list[str],
                    max_new_tokens: int = 64) -> list[str]:
    """批量 left-padded greedy decoding。

    CPU greedy + 正确 attention_mask 下结果与单条逐行一致：
      - padding_side="left" 让所有序列的 prompt 末端对齐到 generate 起点
      - attention_mask 屏蔽 pad token 不进 logits
      - 每行输出为 outputs[:, input_len:] 截掉 padded prompt 部分
    """
    import torch

    if not prompts:
        return []

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        try:
            enc = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            enc = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        inputs = {k: v.to(model.device) for k, v in enc.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = outputs[:, input_len:]
        return tokenizer.batch_decode(gen, skip_special_tokens=True)
    finally:
        tokenizer.padding_side = prev_side


def parse_prediction(text: str) -> tuple[int | None, float]:
    """从模型输出解析 (direction, confidence)。direction: 1=up 0=down, None=解析失败。"""
    # 去掉 <think> 段（qwen3 thinking 模式兜底）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        d = str(obj.get("direction", "")).strip().lower()
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = min(max(conf, 0.0), 1.0)
        if d in ("up", "涨"):
            return 1, conf
        if d in ("down", "跌"):
            return 0, conf
    low = text.lower()
    if "up" in low or "涨" in text:
        return 1, 0.5
    if "down" in low or "跌" in text:
        return 0, 0.5
    return None, 0.5


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #

def load_segment(handler: str, segment: str, golden_dir: Path,
                 num_workers: int = DEFAULT_NUM_WORKERS) -> pd.DataFrame:
    path = golden_dir / handler / f"{segment}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先运行 pre_alpha_handler.py 生成 golden 数据"
        )
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads = {num_workers}")
    df = con.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
    con.close()
    if "instrument" in df.columns and "datetime" in df.columns:
        df = df.set_index(["instrument", "datetime"]).sort_index()
    return df


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != "LABEL0"]
    df = df.dropna(subset=["LABEL0"])
    # 过滤全 0 行（停牌/Cut 边界填充），因子无信息量
    df = df[(df[feature_cols] != 0).any(axis=1)]
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed).sort_index()
    return df


# --------------------------------------------------------------------------- #
# 指标（binary up/down，参考 process_peft_compare.py 风格）
# --------------------------------------------------------------------------- #

def binary_metrics(gts: list[int], preds: list[int],
                   times: list[float]) -> dict:
    total = len(gts)
    avg_time = sum(times) / len(times) if times else 0.0
    accuracy = sum(1 for g, p in zip(gts, preds) if g == p) / total if total else 0.0

    def _per_class(pos):
        tp = sum(1 for g, p in zip(gts, preds) if g == pos and p == pos)
        fp = sum(1 for g, p in zip(gts, preds) if g != pos and p == pos)
        fn = sum(1 for g, p in zip(gts, preds) if g == pos and p != pos)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    p1, r1, f1 = _per_class(1)
    p0, r0, f0 = _per_class(0)
    return {
        "Accuracy": round(accuracy, 4),
        "F1": round((f1 + f0) / 2, 4),
        "Precision": round((p1 + p0) / 2, 4),
        "Recall": round((r1 + r0) / 2, 4),
        "Avg_Time_s": round(avg_time, 2),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B zero-shot 1min up/down prediction")
    parser.add_argument("--handler", default="alpha158_immediate",
                        choices=["hfh12", "alpha158", "hfh12_immediate", "alpha158_immediate"],
                        help="特征集 (默认: hfh12_immediate)")
    parser.add_argument("--segment", default="test", choices=["train", "valid", "test"],
                        help="数据段 (默认: test，PDF 要求所有组共用 test 评估)")
    parser.add_argument("--golden-dir", default=str(GOLDEN_DIR), help="golden 数据目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名或本地路径")
    parser.add_argument("--device", default="cpu", help="推理设备 (默认: cpu, 可选: cpu/cuda)")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
                        help="模型权重精度 (默认: float32; CPU 无 AVX512_BF16 时 fp32 最快，GPU 上 bf16 更快)")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="DuckDB 并行线程数 (默认: 14)")
    parser.add_argument("--sample", type=int, default=0, help="采样行数, 0=全量 (默认: 0)")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8,
                        help="批量推理大小 (默认: 8; HFH12 短 prompt 可调 16, alpha158 长 prompt 建议 4-8)")
    parser.add_argument("--stocks", default="00700,09988,03690,00005",
                        help="逗号分隔的标的代码，如 00700,09988 (默认: 00700,09988,03690,00005)")
    parser.add_argument("--time-slot", default="all", choices=["open", "close", "both", "all"],
                        help="时段: open=开盘30min, close=尾盘30min, both=开+尾, all=全天 (默认: all)")
    parser.add_argument("--save-raw", action="store_true",
                        help="detail CSV 中保存模型原始输出文本 (默认: 不保存)")
    parser.add_argument("--csv-out", default=None,
                        help="输出 CSV 前缀 (默认: llm_zero_shot_{handler}_{segment})")
    args = parser.parse_args()

    exp_dir = OUTPUT_BASE / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_out_base = args.csv_out or f"llm_zero_shot_{args.handler}_{args.segment}"
    log = setup_logging(exp_dir / f"{csv_out_base}.log")

    df = load_segment(args.handler, args.segment, Path(args.golden_dir), args.workers)
    log.info(f"Loaded {args.handler}/{args.segment}: {df.shape[0]} rows × {df.shape[1]} cols")

    # stock filter
    if args.stocks:
        stock_list = [s.strip() for s in args.stocks.split(",")]
        before = len(df)
        df = df[df.index.get_level_values("instrument").isin(stock_list)]
        log.info(f"  Filter stocks {stock_list}: {before} → {len(df)} rows")

    # time-slot filter
    if args.time_slot != "all":
        times = df.index.get_level_values("datetime").time
        if args.time_slot == "open":
            mask = (times >= pd.Timestamp("09:30").time()) & (times <= pd.Timestamp("09:59").time())
        elif args.time_slot == "close":
            mask = (times >= pd.Timestamp("15:30").time()) & (times <= pd.Timestamp("15:59").time())
        else:  # both
            mask = (
                ((times >= pd.Timestamp("09:30").time()) & (times <= pd.Timestamp("09:59").time()))
                | ((times >= pd.Timestamp("15:30").time()) & (times <= pd.Timestamp("15:59").time()))
            )
        before = len(df)
        df = df[mask]
        log.info(f"  Filter time-slot {args.time_slot}: {before} → {len(df)} rows")

    df = sample_rows(df, args.sample, args.seed)
    feature_cols = [c for c in df.columns if c != "LABEL0"]
    n = len(df)
    gts = df["LABEL0"].astype(int).tolist()
    log.info(f"Sampled {n} rows (up={sum(gts)}, down={n - sum(gts)})")

    model, tokenizer = load_model(args.model, args.device, args.dtype)

    preds: list[int] = []
    confs: list[float] = []
    times: list[float] = []
    raws: list[str] = []
    gts_clean: list[int] = []
    kept_rows: list[tuple] = []
    parse_fail = 0

    correct_count = 0
    pbar = tqdm(total=n, desc="Inferencing", unit="inf", disable=False, mininterval=0.1)
    batch: list[tuple[int, tuple, pd.Series]] = []

    def flush_batch(batch_rows: list[tuple[int, tuple, pd.Series]]) -> None:
        nonlocal parse_fail, correct_count
        if not batch_rows:
            return
        prompts = [build_prompt(row, feature_cols, args.handler)
                   for _, _, row in batch_rows]
        t0 = time.perf_counter()
        try:
            contents = inference_batch(model, tokenizer, prompts, args.max_new_tokens)
            if len(contents) != len(batch_rows):
                raise RuntimeError(
                    f"batch decode 长度不匹配: in={len(batch_rows)} out={len(contents)}")
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"ERROR batch@{batch_rows[0][0]}: {e} -> fallback to single")
            # 兜底逐行，保证跑通的行被记录
            contents = []
            for p in prompts:
                try:
                    contents.append(inference(model, tokenizer, p, args.max_new_tokens))
                except Exception as e2:  # noqa: BLE001
                    tqdm.write(f"  single ERROR: {e2}")
                    contents.append("")
        elapsed = time.perf_counter() - t0
        n_in = len(batch_rows)
        # 每行分摊时间（用于 Avg-Time 口径保持一致）
        per_row_time = elapsed / n_in

        for (idx, row_name, _row), content in zip(batch_rows, contents):
            pred, conf = parse_prediction(content)
            if pred is None:
                parse_fail += 1
                pbar.update(1)
                continue
            gt = gts[idx]
            correct = pred == gt
            if correct:
                correct_count += 1
            preds.append(pred)
            confs.append(conf)
            times.append(per_row_time)
            raws.append(content.replace("\n", "\\n"))
            gts_clean.append(gt)
            kept_rows.append(row_name)
            pbar.set_postfix(
                acc=f"{correct_count / len(preds):.2%}",
                avg=f"{sum(times[-50:]) / min(len(times), 50):.2f}s",
                pred="up" if pred == 1 else "down",
                mark="✓" if correct else "✗",
            )
            pbar.update(1)

    for idx, (_, row) in enumerate(df.iterrows()):
        batch.append((idx, row.name, row))
        if len(batch) >= args.batch:
            flush_batch(batch)
            batch.clear()
    flush_batch(batch)
    pbar.close()

    n_valid = len(preds)

    # ---- detail（仅保留有效行） ----
    detail_cols = {
        "instrument": [r[0] for r in kept_rows],
        "datetime": [str(r[1]) for r in kept_rows],
        "ground_truth": ["up" if g == 1 else "down" for g in gts_clean],
        "pred": ["up" if p == 1 else "down" for p in preds],
        "confidence": confs,
        "correct": [p == g for p, g in zip(preds, gts_clean)],
        "time_s": [round(t, 3) for t in times],
    }
    if args.save_raw:
        detail_cols["raw_output"] = raws
    detail = pd.DataFrame(detail_cols)

    # ---- summary（单行，含元信息与指标） ----
    m = binary_metrics(gts_clean, preds, times)
    pos = sum(gts_clean)
    parse_fail_rate = parse_fail / n if n else 0.0
    pos_rate = pos / n_valid if n_valid else 0.0
    summary = pd.DataFrame([{
        "Model": MODEL_LABEL,
        "Handler": args.handler,
        "Segment": args.segment,
        "Device": args.device,
        "Dtype": args.dtype,
        "Batch": args.batch,
        "n_total": n,
        "n_valid": n_valid,
        "parse_fail": parse_fail,
        "parse_fail_rate": round(parse_fail_rate, 4),
        "up": pos,
        "down": n_valid - pos,
        "pos_rate": round(pos_rate, 4),
        **m,
    }])

    sep = "=" * 70
    log.info(f"\n{sep}")
    log.info(f"  n_total={n}  n_valid={n_valid}  parse_fail={parse_fail}  "
             f"parse_fail_rate={parse_fail_rate:.2%}  "
             f"up={pos}  down={n_valid - pos}  pos_rate={pos_rate:.2%}")
    log.info(f"  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
             f"Prec={m['Precision']:.4f}  Rec={m['Recall']:.4f}  "
             f"Avg={m['Avg_Time_s']:.2f}s")
    log.info(sep)

    summary_path = exp_dir / f"{csv_out_base}_summary.csv"
    detail_path = exp_dir / f"{csv_out_base}_detail.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved: {summary_path}\n       {detail_path}")


if __name__ == "__main__":
    main()
