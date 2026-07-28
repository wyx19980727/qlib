import os
import sys
import struct
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

FIELDS_1MIN = [
    "open", "high", "low", "close", "volume", "vwap",
    "trade_count", "buy_count", "sell_count", "unknown_dir_count",
    "unique_brokers", "avg_bid_volume", "avg_ask_volume",
    "paused", "paused_num", "change", "factor",
]

PRICE_FIELDS = {"open", "high", "low", "close", "vwap"}
VOLUME_FIELDS = {"volume", "trade_count", "buy_count", "sell_count",
                 "unknown_dir_count", "unique_brokers",
                 "avg_bid_volume", "avg_ask_volume"}


def _scan_symbol(args):
    features_dir, symbol, freq = args
    sym_dir = features_dir / symbol
    result = {
        "symbol": symbol,
        "n_fields": 0,
        "ok": True,
        "missing_fields": [],
        "nan_ratios": {},
        "start_index": None,
        "length": None,
        "length_mismatch_fields": [],
        "negatives": [],
    }
    if not sym_dir.is_dir():
        result["missing_fields"] = FIELDS_1MIN[:]
        result["ok"] = False
        return result

    for field in FIELDS_1MIN:
        f = sym_dir / f"{field}.{freq}.bin"
        if not f.exists():
            result["missing_fields"].append(field)
            result["ok"] = False
            continue
        raw = f.read_bytes()
        if len(raw) < 4:
            result["missing_fields"].append(field)
            result["ok"] = False
            continue
        start = np.frombuffer(raw[:4], dtype="<f4")[0]
        vals = np.frombuffer(raw[4:], dtype="<f4")
        result["n_fields"] += 1
        result["nan_ratios"][field] = float(np.isnan(vals).sum() / max(len(vals), 1))

        if result["start_index"] is None:
            result["start_index"] = int(start)
        elif int(start) != result["start_index"]:
            result["length_mismatch_fields"].append(f"start_{field}")

        if result["length"] is None:
            result["length"] = len(vals)
        elif len(vals) != result["length"]:
            result["length_mismatch_fields"].append(f"len_{field}")

        if field in PRICE_FIELDS:
            nz = vals[~np.isnan(vals)]
            if len(nz) > 0 and np.any(nz < 0):
                result["negatives"].append(field)
    return result


class DataHealthCheck:
    def __init__(
        self,
        qlib_dir: str,
        freq: str = "1min",
        max_workers: int = 16,
        nan_warn_threshold: float = 0.95,
    ):
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.features_dir = self.qlib_dir / "features"
        self.freq = freq
        self.max_workers = max_workers
        self.nan_warn_threshold = nan_warn_threshold

        assert self.features_dir.is_dir(), f"features dir not found: {self.features_dir}"
        symbols = sorted(
            d.name for d in self.features_dir.iterdir() if d.is_dir()
        )
        self.symbols = symbols
        logger.info(f"Found {len(symbols)} symbols under {self.features_dir}")

        self.results = []

    def check_features_dir_lowercase(self):
        bad = []
        for name in os.listdir(self.features_dir):
            fp = os.path.join(self.features_dir, name)
            if os.path.isdir(fp) and name != name.lower():
                bad.append(name)
        if bad:
            df = pd.DataFrame({"non_lowercase_dir": bad})
            logger.warning(f"{len(bad)} directories with uppercase names")
            return df
        logger.info("All directories are lowercase")
        return None

    def check_field_integrity(self):
        missing_map = {}
        for r in self.results:
            if r["missing_fields"]:
                missing_map[r["symbol"]] = r["missing_fields"]
        if not missing_map:
            logger.info("All symbols have complete field set")
            return None
        rows = []
        for sym, fields in missing_map.items():
            rows.append({"symbol": sym, "missing_count": len(fields),
                         "missing_fields": ",".join(fields)})
        df = pd.DataFrame(rows).set_index("symbol")
        logger.warning(f"{len(df)} symbols with missing fields")
        return df

    def check_start_index(self):
        bad = []
        for r in self.results:
            if r["length_mismatch_fields"]:
                bad.append({"symbol": r["symbol"],
                            "issues": ",".join(r["length_mismatch_fields"])})
        if bad:
            df = pd.DataFrame(bad).set_index("symbol")
            logger.warning(f"{len(df)} symbols with start_index/length mismatch")
            return df
        logger.info("All fields have aligned start_index and length")
        return None

    def check_nan_ratio(self):
        rows = []
        for r in self.results:
            for field, ratio in r["nan_ratios"].items():
                rows.append({
                    "symbol": r["symbol"],
                    "field": field,
                    "nan_ratio": round(ratio, 4),
                    "n_vals": r["length"],
                    "flag": "HIGH" if ratio > self.nan_warn_threshold else "",
                })
        df = pd.DataFrame(rows)

        pivot = df.pivot_table(
            index="field", values="nan_ratio", aggfunc=["mean", "std", "min", "max"]
        )
        pivot.columns = ["nan_mean", "nan_std", "nan_min", "nan_max"]
        pivot = pivot.round(4)

        high_fields = df[df["flag"] == "HIGH"]
        logger.info(f"Field-wise avg NaN ratio:\n{pivot.to_string()}")
        if len(high_fields) > 0:
            logger.warning(
                f"{len(high_fields)} symbol-field pairs have NaN ratio > {self.nan_warn_threshold}"
            )
        return df, pivot

    def check_negative_prices(self):
        bad = []
        for r in self.results:
            for f in r["negatives"]:
                bad.append({"symbol": r["symbol"], "field": f})
        if bad:
            df = pd.DataFrame(bad).set_index("symbol")
            logger.warning(f"{len(df)} symbols have negative price values")
            return df
        logger.info("No negative price values found")
        return None

    def _run_scan(self):
        logger.info(f"Scanning {len(self.symbols)} symbols with {self.max_workers} workers...")
        args_list = [(self.features_dir, s, self.freq) for s in self.symbols]
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self.results = list(tqdm(
                executor.map(_scan_symbol, args_list),
                total=len(args_list),
                desc="Scanning",
            ))

    def export_csv(self, out_dir, nan_df, field_pivot):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if nan_df is not None:
            nan_df.to_csv(out / "field_nan_detail.csv", index=True)
            logger.info(f"  Saved: {out / 'field_nan_detail.csv'} ({len(nan_df)} rows)")

        if field_pivot is not None:
            field_pivot.to_csv(out / "field_nan_summary.csv")
            logger.info(f"  Saved: {out / 'field_nan_summary.csv'}")

        integrity = self.check_field_integrity()
        if integrity is not None:
            integrity.to_csv(out / "missing_files.csv")
            logger.info(f"  Saved: {out / 'missing_files.csv'}")

        start_issues = self.check_start_index()
        if start_issues is not None:
            start_issues.to_csv(out / "start_index_issues.csv")
            logger.info(f"  Saved: {out / 'start_index_issues.csv'}")

        neg = self.check_negative_prices()
        if neg is not None:
            neg.to_csv(out / "negative_prices.csv")
            logger.info(f"  Saved: {out / 'negative_prices.csv'}")

        case = self.check_features_dir_lowercase()
        if case is not None:
            case.to_csv(out / "case_issues.csv")
            logger.info(f"  Saved: {out / 'case_issues.csv'}")

    def check_data(self, out_dir: str = None):
        logger.info(f"=== Data Health Check ===")
        logger.info(f"  qlib_dir: {self.qlib_dir}")
        logger.info(f"  freq:     {self.freq}")
        logger.info(f"  symbols:  {len(self.symbols)}")

        self._run_scan()

        nan_df, field_pivot = self.check_nan_ratio()

        print()
        print(f"{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        total = len(self.results)
        ok = sum(1 for r in self.results if r["ok"])
        partial = sum(1 for r in self.results if not r["ok"] and r["n_fields"] > 0)
        missing_all = sum(1 for r in self.results if r["n_fields"] == 0)
        print(f"  Total symbols:       {total}")
        print(f"  All fields present:  {ok}")
        print(f"  Partial fields:      {partial}")
        print(f"  No bin files:        {missing_all}")

        print()
        print("Field-wise average NaN ratio:")
        for field in FIELDS_1MIN:
            vals = [r["nan_ratios"].get(field, 1.0) for r in self.results if field in r["nan_ratios"]]
            if vals:
                avg = np.mean(vals) * 100
                flag = " ⚠" if avg > self.nan_warn_threshold * 100 else ""
                print(f"  {field:<20} {avg:>6.1f}% NaN{flag}")

        integrity = self.check_field_integrity()
        if integrity is not None:
            print(f"\n  Missing files: {len(integrity)} symbols affected")

        start_issues = self.check_start_index()
        if start_issues is not None:
            print(f"  Start index/length mismatch: {len(start_issues)} symbols")

        neg = self.check_negative_prices()
        if neg is not None:
            print(f"  Negative prices: {len(neg)} symbols")

        case = self.check_features_dir_lowercase()
        if case is not None:
            print(f"  Uppercase dirs: {len(case)}")

        if out_dir:
            self.export_csv(out_dir, nan_df, field_pivot)

        logger.info("Data health check complete.")


if __name__ == "__main__":
    fire.Fire(DataHealthCheck)
