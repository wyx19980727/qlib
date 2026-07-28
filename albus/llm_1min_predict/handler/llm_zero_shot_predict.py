"""Qwen3.5-0.8B zero-shot 1min 涨/跌预测（PDF 实验组 A：基线零样本）。

数据流向：
    golden_data_1min/{hfh12|alpha158}/{test}.parquet   (pre_alpha_handler.py 产出)
        ↓  按行采样 → 因子数值转文本 prompt（保留 3 位小数）
    Qwen3.5-0.8B (transformers, CPU/GPU, zero-shot)
        ↓  输出 JSON {"direction": "up"|"down", "confidence": 0-1}
    {out}_detail.csv / {out}_summary.csv   (Accuracy / F1 / Precision / Recall / AUC / Avg_Time)

与 PDF 的差异：不是 30 个 L2 微观结构因子，而是 HFH12（12 个高频归一化因子，默认）
或 Alpha158（158 因子）。label 为 LABEL0 = If(Gt(Ref($close,-2), Ref($close,-1)), 1, 0)，
即 t+2 收盘 > t+1 收盘 → 1（涨）。

用法（conda qlib 环境）：
    python llm_zero_shot_predict.py                          # HFH12, test 段, 采样 200 行
    python llm_zero_shot_predict.py --handler alpha158
    python llm_zero_shot_predict.py --sample 50 --segment valid
    python llm_zero_shot_predict.py --model Qwen/Qwen3.5-0.8B --csv-out my_result
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

GOLDEN_DIR = Path("/home/albus/Python_Codes/qlib/qlib_data/golden_data_1min")
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_LABEL = "Qwen3.5-0.8B-base (zero-shot)"

# --------------------------------------------------------------------------- #
# 因子释义（写进 prompt，应对 PDF“基座模型不懂因子含义”挑战）
# --------------------------------------------------------------------------- #

HFH12_DESC = {
    "$open": "当日该分钟开盘价 / 昨日收盘价",
    "$high": "当日该分钟最高价 / 昨日收盘价",
    "$low": "当日该分钟最低价 / 昨日收盘价",
    "$close": "当日该分钟收盘价 / 昨日收盘价",
    "$vwap": "当日该分钟成交量加权均价 / 昨日收盘价",
    "$open_1": "昨日同一分钟开盘价 / 昨日收盘价",
    "$high_1": "昨日同一分钟最高价 / 昨日收盘价",
    "$low_1": "昨日同一分钟最低价 / 昨日收盘价",
    "$close_1": "昨日同一分钟收盘价 / 昨日收盘价",
    "$vwap_1": "昨日同一分钟成交量加权均价 / 昨日收盘价",
    "$volume": "当日该分钟成交量 / 近30日分钟均量",
    "$volume_1": "昨日同一分钟成交量 / 近30日分钟均量",
}

ALPHA158_LEGEND = (
    "因子命名规则：KMID/KLEN/KUP/KLOW/KSFT 等为K线形态类（当根K线实体、影线比例）；"
    "OPEN0/HIGH0/LOW0/VWAP0 为当前分钟价格/收盘价比率；"
    "ROC*/MA*/STD*/BETA*/RSQR*/RESI*/MAX*/MIN*/QTLU*/QTLD*/RANK*/RSV*/"
    "IMAX*/IMIN*/IMXD*/CORR*/CORD*/CNTP*/CNTN*/CNTD*/SUMP*/SUMN*/SUMD*/"
    "VMA*/VSTD*/WVMA*/VSUMP*/VSUMN*/VSUMD* 为滚动窗口统计类，"
    "数字后缀表示窗口长度（分钟），值多为相对当前收盘价的归一化比率。"
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
    if handler == "alpha158":
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

def load_model(name_or_path: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            config=text_config if text_config is not None else config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        pbar.update(1)

    return model, tokenizer


def inference(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except (TypeError, ValueError):
        # 旧版模板不支持 enable_thinking 参数
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


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

def load_segment(handler: str, segment: str, golden_dir: Path) -> pd.DataFrame:
    path = golden_dir / handler / f"{segment}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先运行 pre_alpha_handler.py 生成 golden 数据"
        )
    df = pd.read_parquet(path)
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

def binary_metrics(gts: list[int], preds: list[int], scores: list[float],
                   times: list[float], positive: int) -> dict:
    tp = sum(1 for g, p in zip(gts, preds) if g == positive and p == positive)
    fp = sum(1 for g, p in zip(gts, preds) if g != positive and p == positive)
    fn = sum(1 for g, p in zip(gts, preds) if g == positive and p != positive)
    total = len(gts)
    accuracy = sum(1 for g, p in zip(gts, preds) if g == p) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    try:
        from sklearn.metrics import roc_auc_score

        y = [1 if g == positive else 0 for g in gts]
        s = scores if positive == 1 else [1 - x for x in scores]
        auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    avg_time = sum(times) / len(times) if times else 0.0
    return {
        "Accuracy": round(accuracy, 4), "F1": round(f1, 4),
        "Precision": round(precision, 4), "Recall": round(recall, 4),
        "AUC": round(auc, 4) if auc == auc else "",
        "Avg_Time_s": round(avg_time, 2),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main():
    import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B zero-shot 1min up/down prediction")
    parser.add_argument("--handler", default="alpha158", choices=["hfh12", "alpha158"],
                        help="特征集 (默认: hfh12)")
    parser.add_argument("--segment", default="test", choices=["train", "valid", "test"],
                        help="数据段 (默认: test，PDF 要求所有组共用 test 评估)")
    parser.add_argument("--golden-dir", default=str(GOLDEN_DIR), help="golden 数据目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名或本地路径")
    parser.add_argument("--sample", type=int, default=1000, help="采样行数, 0=全量 (默认: 200)")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--csv-out", default=None,
                        help="输出 CSV 前缀 (默认: llm_zero_shot_{handler}_{segment})")
    args = parser.parse_args()

    csv_out_base = args.csv_out or f"llm_zero_shot_{args.handler}_{args.segment}"

    df = load_segment(args.handler, args.segment, Path(args.golden_dir))
    print(f"Loaded {args.handler}/{args.segment}: {df.shape[0]} rows × {df.shape[1]} cols")
    df = sample_rows(df, args.sample, args.seed)
    feature_cols = [c for c in df.columns if c != "LABEL0"]
    n = len(df)
    gts = df["LABEL0"].astype(int).tolist()
    print(f"Sampled {n} rows (up={sum(gts)}, down={n - sum(gts)})")

    model, tokenizer = load_model(args.model)

    preds: list[int] = []
    confs: list[float] = []
    times: list[float] = []
    raws: list[str] = []
    parse_fail = 0

    correct_count = 0
    pbar = tqdm(total=n, desc="Inferencing", unit="inf", disable=False, mininterval=0.1)
    for _, row in df.iterrows():
        prompt = build_prompt(row, feature_cols, args.handler)
        t0 = time.perf_counter()
        try:
            content = inference(model, tokenizer, prompt, args.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"ERROR {row.name}: {e}")
            content = ""
        elapsed = time.perf_counter() - t0

        pred, conf = parse_prediction(content)
        if pred is None:
            parse_fail += 1
            pred, conf = 0, 0.5  # 解析失败按 down/无置信处理
        correct = pred == gts[len(preds)]
        if correct:
            correct_count += 1
        preds.append(pred)
        confs.append(conf)
        times.append(elapsed)
        raws.append(content.replace("\n", "\\n"))
        pbar.set_postfix(
            acc=f"{correct_count / len(preds):.2%}",
            avg=f"{sum(times[-50:]) / min(len(times), 50):.2f}s",
            pred="up" if pred == 1 else "down",
            mark="✓" if correct else "✗",
        )
        pbar.update(1)
    pbar.close()

    # score = P(up)：pred=up 时为 conf，pred=down 时为 1-conf
    scores = [c if p == 1 else 1 - c for p, c in zip(preds, confs)]

    # ---- detail ----
    detail = pd.DataFrame({
        "instrument": [i for i, _ in df.index],
        "datetime": [str(t) for _, t in df.index],
        "ground_truth": ["up" if g == 1 else "down" for g in gts],
        "pred": ["up" if p == 1 else "down" for p in preds],
        "confidence": confs,
        "score_up": scores,
        "correct": [p == g for p, g in zip(preds, gts)],
        "time_s": [round(t, 3) for t in times],
        "raw_output": raws,
    })

    # ---- summary ----
    rows = [{
        "Model": "DATASET",
        "Handler": args.handler,
        "Class": f"up={sum(gts)} down={n - sum(gts)} parse_fail={parse_fail}",
        "Accuracy": "", "F1": "", "Precision": "", "Recall": "", "AUC": "", "Avg_Time_s": "",
    }]
    for cls_name, positive in [("overall(up)", 1), ("down", 0)]:
        m = binary_metrics(gts, preds, scores, times, positive)
        rows.append({"Model": MODEL_LABEL, "Handler": args.handler, "Class": cls_name, **m})
    summary = pd.DataFrame(rows)

    print("\n" + "=" * 96)
    print(f"{'Class':<14} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7} {'Time':>7}")
    print("-" * 96)
    for r in rows[1:]:
        auc_s = f"{r['AUC']:.3f}" if r["AUC"] != "" else "  n/a"
        print(f"{r['Class']:<14} {r['Accuracy']:>7.3f} {r['F1']:>7.3f} "
              f"{r['Precision']:>7.3f} {r['Recall']:>7.3f} {auc_s:>7} {r['Avg_Time_s']:>6.2f}s")
    print("=" * 96)

    out_dir = Path(__file__).resolve().parent
    summary_path = out_dir / f"{csv_out_base}_summary.csv"
    detail_path = out_dir / f"{csv_out_base}_detail.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {summary_path}\n       {detail_path}")


if __name__ == "__main__":
    main()
