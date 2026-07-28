import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

import qlib
from qlib.constant import REG_HK
from qlib.utils import init_instance_by_config
from qlib.contrib.data.highfreq_handler import HighFreqGeneralHandler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from highfreq.highfreq_ops import DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class HighFreqWithLabelHandler(HighFreqGeneralHandler):
    def __init__(self, label_expr=None, label_name="LABEL0", **kwargs):
        self.label_expr = label_expr or ["Ref($close, -2) / Ref($close, -1) - 1"]
        self.label_name = label_name
        super().__init__(**kwargs)

    def get_feature_config(self):
        features = super().get_feature_config()
        label = (self.label_expr, [self.label_name])
        return {"feature": features, "label": label}


def load_label_data(from_qlib=False):
    if not from_qlib:
        patterns = [
            "/home/albus/Python_Codes/qlib/mlruns/qlib_hk_1min_lightgbm/*/artifacts/label.pkl",
            "/home/albus/Python_Codes/qlib/mlruns/qlib_hk_1min_hflightgbm/*/artifacts/label.pkl",
        ]
        for pattern in patterns:
            fps = glob.glob(pattern)
            if fps:
                print(f"Loading label from: {fps[0]}")
                with open(fps[0], "rb") as f:
                    return pickle.load(f)

    print("Computing labels via qlib (train/valid/test)...")

    qlib.init(
        provider_uri=os.path.join(os.path.dirname(__file__), "..", "qlib_data", "silver_data_1min"),
        region=REG_HK,
        custom_ops=[DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut],
    )

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "HighFreqWithLabelHandler",
                "module_path": "__main__",
                "kwargs": {
                    "start_time": "2026-05-22 09:30:00",
                    "end_time": "2026-06-18 15:59:00",
                    "fit_start_time": "2026-05-22 09:30:00",
                    "fit_end_time": "2026-06-11 15:59:00",
                    "instruments": "all",
                    "freq": "1min",
                    "day_length": 330,
                    "drop_raw": False,
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                },
            },
            "segments": {
                "train": ("2026-05-22 09:30:00", "2026-06-11 15:59:00"),
                "valid": ("2026-06-11 15:59:00", "2026-06-16 15:59:00"),
                "test": ("2026-06-16 15:59:00", "2026-06-18 15:59:00"),
            },
        },
    }
    dataset = init_instance_by_config(dataset_config)
    return load_all_segments(dataset)


def load_all_segments(dataset):
    print("Preparing train/valid/test labels...")
    result = {}
    for seg in ["train", "valid", "test"]:
        df = dataset.prepare(seg, col_set="label")
        result[seg] = df
        print(f"  {seg}: {df.shape[0]:>8} rows, {df.shape[1]} cols")
    return result


def per_stock_summary(label_df):
    vals = label_df.values.flatten()
    idx = label_df.index
    stocks = idx.get_level_values("instrument")
    dts = idx.get_level_values("datetime")

    df = pd.DataFrame({"label": vals, "stock": stocks, "datetime": dts})

    summary = df.groupby("stock").agg(
        n_rows=("label", "size"),
        nan_count=("label", lambda x: np.isnan(x).sum()),
        zero_count=("label", lambda x: (x == 0).sum()),
    )
    summary["valid_count"] = summary["n_rows"] - summary["nan_count"]
    summary["nan_ratio"] = summary["nan_count"] / summary["n_rows"]
    summary["zero_ratio_exnan"] = (
        summary["zero_count"] / summary["valid_count"].clip(lower=1)
    )
    summary["zero_ratio_exnan"] = summary["zero_ratio_exnan"].clip(upper=1)
    return summary, df


def plot_segment_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"train": "steelblue", "valid": "orange", "test": "coral"}
    for ax, metric in zip(axes, ["nan_ratio", "zero_ratio_exnan", "valid_count"]):
        for seg_name, res in results.items():
            summary = res["summary"]
            if metric == "valid_count":
                mask = summary[metric] > 0
                vals = summary.loc[mask, metric]
            else:
                vals = summary[metric]
            ax.hist(vals, bins=40, alpha=0.5, label=seg_name, color=colors.get(seg_name, "gray"))
        ax.set_xlabel(metric)
        ax.set_ylabel("Stocks")
        ax.set_title(metric)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_comparison.png", dpi=150)
    plt.close()
    print("  Saved: segment_comparison.png")


def plot_nan_ratio_histogram(summary, seg_name=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(summary["nan_ratio"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(
        summary["nan_ratio"].mean(), color="red", ls="--",
        label=f"mean={summary['nan_ratio'].mean():.3f}",
    )
    ax.axvline(
        summary["nan_ratio"].median(), color="orange", ls="--",
        label=f"median={summary['nan_ratio'].median():.3f}",
    )
    ax.set_xlabel("NaN Ratio")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("Per-Stock NaN Ratio Distribution\n(What DropnaLabel removes)")
    ax.legend()

    ax = axes[1]
    sorted_r = np.sort(summary["nan_ratio"])
    ax.plot(sorted_r, np.arange(1, len(sorted_r) + 1) / len(sorted_r), "b-", lw=2)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("NaN Ratio")
    ax.set_ylabel("Cumulative Fraction of Stocks")
    ax.set_title("CDF of NaN Ratio")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fn = f"01_nan_ratio_{seg_name}.png" if seg_name else "01_nan_ratio_distribution.png"
    plt.savefig(OUTPUT_DIR / fn, dpi=150)
    plt.close()
    print(f"  Saved: {fn}")


def plot_zero_ratio_histogram(summary, seg_name=""):
    mask = summary["valid_count"] > 0
    subs = summary[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(subs["zero_ratio_exnan"], bins=50, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(
        subs["zero_ratio_exnan"].mean(), color="red", ls="--",
        label=f"mean={subs['zero_ratio_exnan'].mean():.3f}",
    )
    ax.axvline(
        subs["zero_ratio_exnan"].median(), color="orange", ls="--",
        label=f"median={subs['zero_ratio_exnan'].median():.3f}",
    )
    ax.set_xlabel("Zero Ratio (excl. NaN)")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("Per-Stock Zero-Label Ratio Distribution\n(Excluding NaN samples)")
    ax.legend()

    ax = axes[1]
    sorted_r = np.sort(subs["zero_ratio_exnan"])
    ax.plot(sorted_r, np.arange(1, len(sorted_r) + 1) / len(sorted_r), "b-", lw=2)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Zero Ratio (excl. NaN)")
    ax.set_ylabel("Cumulative Fraction of Stocks")
    ax.set_title("CDF of Zero Ratio")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fn = f"02_zero_ratio_{seg_name}.png" if seg_name else "02_zero_ratio_distribution.png"
    plt.savefig(OUTPUT_DIR / fn, dpi=150)
    plt.close()
    print(f"  Saved: {fn}")


def plot_nan_vs_zero_scatter(summary):
    mask = summary["valid_count"] > 0
    subs = summary[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        subs["nan_ratio"], subs["zero_ratio_exnan"],
        c=subs["valid_count"], cmap="viridis", alpha=0.6, s=8,
    )
    plt.colorbar(sc, ax=ax, label="Valid Sample Count")
    ax.set_xlabel("NaN Ratio")
    ax.set_ylabel("Zero Ratio (excl. NaN)")
    ax.set_title("NaN Ratio vs Zero Ratio (per Stock)")

    corr = subs["nan_ratio"].corr(subs["zero_ratio_exnan"])
    ax.text(
        0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_nan_vs_zero_scatter.png", dpi=150)
    plt.close()
    print("  Saved: 03_nan_vs_zero_scatter.png")


def plot_daily_valid_sample_count(df):
    df = df.copy()
    df["date"] = df["datetime"].dt.date

    daily = df.groupby("date").agg(
        total=("label", "count"),
        nan_count=("label", lambda x: np.isnan(x).sum()),
        zero_count=("label", lambda x: (x == 0).sum()),
    )
    daily["valid"] = daily["total"] - daily["nan_count"]
    daily["nonzero"] = daily["valid"] - daily["zero_count"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily["total"], "gray", alpha=0.5, label="Total (all stocks)")
    ax.plot(daily.index, daily["valid"], "steelblue", lw=2, label="Valid (non-NaN)")
    ax.plot(daily.index, daily["nonzero"], "coral", lw=2, label="Non-zero label")
    ax.fill_between(daily.index, daily["valid"], daily["nonzero"], color="coral", alpha=0.15)
    ax.fill_between(daily.index, daily["total"], daily["valid"], color="gray", alpha=0.15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("Daily Sample Count (2979 stocks × 330 1min bars/day)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.text(
        0.02, 0.98,
        f"Avg valid/day: {daily['valid'].mean():.0f}\nAvg nonzero/day: {daily['nonzero'].mean():.0f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_daily_sample_count.png", dpi=150)
    plt.close()
    print("  Saved: 04_daily_sample_count.png")


def plot_daily_valid_sample_count(df, seg_name=""):
    df = df.copy()
    df["date"] = df["datetime"].dt.date

    daily = df.groupby("date").agg(
        total=("label", "count"),
        nan_count=("label", lambda x: np.isnan(x).sum()),
        zero_count=("label", lambda x: (x == 0).sum()),
    )
    daily["valid"] = daily["total"] - daily["nan_count"]
    daily["nonzero"] = daily["valid"] - daily["zero_count"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily["total"], "gray", alpha=0.5, label="Total (all stocks)")
    ax.plot(daily.index, daily["valid"], "steelblue", lw=2, label="Valid (non-NaN)")
    ax.plot(daily.index, daily["nonzero"], "coral", lw=2, label="Non-zero label")
    ax.fill_between(daily.index, daily["valid"], daily["nonzero"], color="coral", alpha=0.15)
    ax.fill_between(daily.index, daily["total"], daily["valid"], color="gray", alpha=0.15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stocks")
    ax.set_title(f"Daily Sample Count — {seg_name or ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.98,
        f"Avg valid/day: {daily['valid'].mean():.0f}\nAvg nonzero/day: {daily['nonzero'].mean():.0f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fn = f"04_daily_{seg_name}.png" if seg_name else "04_daily_sample_count.png"
    plt.savefig(OUTPUT_DIR / fn, dpi=150)
    plt.close()
    print(f"  Saved: {fn}")


def plot_label_distribution(df, seg_name=""):
    vals = df["label"].values
    vals = vals[~np.isnan(vals)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(vals, bins=200, color="steelblue", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1, label="label=0")
    ax.set_xlabel("Forward Return (Ref($close,-2)/Ref($close,-1)-1)")
    ax.set_ylabel("Frequency")
    ax.set_title("Raw Label Distribution (all valid samples)")
    ax.set_yscale("symlog")
    ax.legend()

    ax = axes[1]
    mask = (vals > -0.01) & (vals < 0.01)
    ax.hist(vals[mask], bins=200, color="steelblue", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Forward Return (zoomed ±1%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Label Distribution (zoomed, ±1%)")

    stats_text = (
        f"n = {len(vals):,}\n"
        f"mean = {np.mean(vals):.6f}\n"
        f"std = {np.std(vals):.6f}\n\n"
        f"zero ratio = {(vals == 0).sum() / len(vals):.4f}"
    )
    ax.text(
        0.95, 0.95, stats_text, transform=ax.transAxes, va="top", ha="right",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fn = f"05_label_dist_{seg_name}.png" if seg_name else "05_label_distribution.png"
    plt.savefig(OUTPUT_DIR / fn, dpi=150)
    plt.close()
    print(f"  Saved: {fn}")


def plot_heatmap_top_stocks(df, n_stocks=30):
    stock_valid = df.groupby("stock")["label"].apply(
        lambda x: (~np.isnan(x)).sum()
    )
    top = stock_valid.nlargest(n_stocks).index.tolist()

    sub = df[df["stock"].isin(top)].copy()
    sub["date"] = sub["datetime"].dt.date
    sub["is_nan"] = np.isnan(sub["label"])
    sub["is_zero"] = sub["label"] == 0

    nan_pivot = sub.pivot_table(
        index="stock", columns="date", values="is_nan", aggfunc="mean"
    )
    zero_pivot = sub.pivot_table(
        index="stock", columns="date", values="is_zero", aggfunc="mean"
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    sns.heatmap(
        nan_pivot, ax=axes[0], cmap="RdYlBu_r", vmin=0, vmax=1,
        cbar_kws={"label": "NaN Ratio"},
    )
    axes[0].set_title(f"NaN Ratio per Day — Top {n_stocks} Stocks")
    axes[0].set_ylabel("Stock")
    axes[0].set_xlabel("Date")

    sns.heatmap(
        zero_pivot, ax=axes[1], cmap="RdYlBu_r", vmin=0, vmax=1,
        cbar_kws={"label": "Zero Ratio"},
    )
    axes[1].set_title(f"Zero Ratio per Day — Top {n_stocks} Stocks")
    axes[1].set_ylabel("Stock")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_heatmap_top_stocks.png", dpi=150)
    plt.close()
    print("  Saved: 06_heatmap_top_stocks.png")


def print_summary_stats(summary):
    mask = summary["valid_count"] > 0
    subs = summary[mask]

    print()
    print("=" * 64)
    print("SUMMARY STATISTICS")
    print("=" * 64)
    print(f"Total stocks:                     {len(summary)}")
    print(f"Stocks with any valid data:       {mask.sum()}")
    print(f"Stocks with NO valid data:        {(~mask).sum()}")
    print()

    print("NaN ratio per stock:")
    print(f"  mean:   {summary['nan_ratio'].mean():.4f}")
    print(f"  median: {summary['nan_ratio'].median():.4f}")
    n90 = (summary['nan_ratio'] > 0.9).sum()
    n50 = (summary['nan_ratio'] > 0.5).sum()
    print(f"  >90%:   {n90} stocks ({n90 / len(summary) * 100:.1f}%)")
    print(f"  >50%:   {n50} stocks ({n50 / len(summary) * 100:.1f}%)")
    print()

    if len(subs) > 0:
        print("Zero ratio (ex-NaN) per stock (among stocks with valid data):")
        print(f"  mean:   {subs['zero_ratio_exnan'].mean():.4f}")
        print(f"  median: {subs['zero_ratio_exnan'].median():.4f}")
        z50 = (subs['zero_ratio_exnan'] > 0.5).sum()
        print(f"  >50%:   {z50} stocks ({z50 / len(subs) * 100:.1f}%)")
    print()

    total_rows = summary["n_rows"].sum()
    total_nan = summary["nan_count"].sum()
    total_valid = summary["valid_count"].sum()
    total_zero = summary["zero_count"].sum()
    print("Global statistics:")
    print(f"  stock-datetime slots:  {total_rows:,}")
    print(f"  NaN:                   {total_nan:,} ({total_nan / total_rows * 100:.1f}%)")
    print(f"  valid (non-NaN):       {total_valid:,} ({total_valid / total_rows * 100:.1f}%)")
    print(f"  of which zero:         {total_zero:,} ({total_zero / total_valid * 100:.1f}%)")
    print(f"  of which nonzero:      {total_valid - total_zero:,} ({(total_valid - total_zero) / total_valid * 100:.1f}%)")
    print("=" * 64)


def export_tables(results):
    out = OUTPUT_DIR
    seg_order = ["train", "valid", "test"]

    comparison_rows = []
    for seg_name in seg_order:
        if seg_name not in results:
            continue
        summary = results[seg_name]["summary"]
        df = results[seg_name]["df"]

        prefix = f"{seg_name}_"
        per_stock = summary.copy()
        per_stock = per_stock.rename(columns={
            "n_rows": "total_bars", "nan_count": "nan_count",
            "zero_count": "zero_count", "valid_count": "valid_bars",
            "nan_ratio": "nan_ratio", "zero_ratio_exnan": "zero_ratio",
        })
        per_stock = per_stock[["total_bars", "nan_count", "valid_bars", "zero_count", "nan_ratio", "zero_ratio"]]
        per_stock.to_csv(out / f"{prefix}per_stock.csv")
        print(f"  Saved: {prefix}per_stock.csv ({len(per_stock)} stocks)")

        daily = df.copy()
        daily["date"] = daily["datetime"].dt.date
        daily_tbl = daily.groupby("date").agg(
            total_bars=("label", "count"),
            nan_count=("label", lambda x: np.isnan(x).sum()),
            zero_count=("label", lambda x: (x == 0).sum()),
        )
        daily_tbl["valid_bars"] = daily_tbl["total_bars"] - daily_tbl["nan_count"]
        daily_tbl["nonzero_bars"] = daily_tbl["valid_bars"] - daily_tbl["zero_count"]
        daily_tbl["nan_pct"] = (daily_tbl["nan_count"] / daily_tbl["total_bars"] * 100).round(1)
        daily_tbl["zero_pct"] = (daily_tbl["zero_count"] / daily_tbl["valid_bars"] * 100).round(1)
        daily_tbl.to_csv(out / f"{prefix}daily.csv")
        print(f"  Saved: {prefix}daily.csv ({len(daily_tbl)} days)")

        buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        nan_b = pd.cut(summary["nan_ratio"], bins=buckets, include_lowest=True)
        nan_dist = nan_b.value_counts().sort_index()
        vm = summary["valid_count"] > 0
        zero_b = pd.cut(summary.loc[vm, "zero_ratio_exnan"], bins=buckets, include_lowest=True)
        zero_dist = zero_b.value_counts().sort_index()
        bucket_tbl = pd.DataFrame({
            "bucket": [f"{b:.0%}-{buckets[i+1]:.0%}" for i, b in enumerate(buckets[:-1])],
            "nan_stocks": nan_dist.values,
            "zero_stocks": np.pad(zero_dist.values, (0, len(nan_dist) - len(zero_dist)), constant_values=0),
        })
        bucket_tbl.to_csv(out / f"{prefix}bucket.csv", index=False)
        print(f"  Saved: {prefix}bucket.csv")

        row = {
            "segment": seg_name,
            "total_stocks": len(summary),
            "stocks_with_data": vm.sum(),
            "stocks_no_data": (~vm).sum(),
            "total_rows": int(summary["n_rows"].sum()),
            "nan_rows": int(summary["nan_count"].sum()),
            "nan_pct": round(summary["nan_count"].sum() / summary["n_rows"].sum() * 100, 1),
            "valid_rows": int(summary["valid_count"].sum()),
            "zero_rows": int(summary["zero_count"].sum()),
            "zero_pct": round(summary["zero_count"].sum() / summary["valid_count"].sum() * 100, 1),
            "nonzero_rows": int(summary["valid_count"].sum() - summary["zero_count"].sum()),
            "nonzero_pct": round((summary["valid_count"].sum() - summary["zero_count"].sum()) / summary["valid_count"].sum() * 100, 1),
        }
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out / "segments_comparison.csv", index=False)
    print(f"  Saved: segments_comparison.csv")

    print("\n=== Segment Comparison ===")
    print(comparison.to_string(index=False))

    print(f"\nAll tables saved to: {out}/")


def main():
    data = load_label_data(from_qlib=True)
    if isinstance(data, dict):
        # Already a dict of segment → DataFrame (from qlib)
        labels = data
        print(f"Loaded segments: {list(labels.keys())}")
    else:
        # Single DataFrame from MLflow pickle (test set only)
        dataset = load_label_data.__wrapped__() if hasattr(load_label_data, '__wrapped__') else None
        labels = {"test": data}

    results = {}
    all_dfs = []
    for seg_name, seg_df in labels.items():
        print(f"\n{'='*64}")
        print(f"SEGMENT: {seg_name}")
        print(f"{'='*64}")
        summary, full_df = per_stock_summary(seg_df)
        results[seg_name] = {"summary": summary, "df": full_df}
        all_dfs.append(full_df.assign(_segment=seg_name))
        print_summary_stats(summary)

        plot_nan_ratio_histogram(summary, seg_name)
        plot_zero_ratio_histogram(summary, seg_name)
        plot_label_distribution(full_df, seg_name)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    for seg_name, res in results.items():
        plot_daily_valid_sample_count(res["df"], seg_name)

    plot_heatmap_top_stocks(combined_df, n_stocks=30)
    export_tables(results)
    plot_segment_comparison(results)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
