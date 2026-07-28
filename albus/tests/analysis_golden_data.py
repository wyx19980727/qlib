import os
import time
import duckdb
import pyarrow.parquet as pq

GOLDEN_DATA_DIR = "/home/albus/Python_Codes/qlib/qlib_data/golden_data_1min"


def _stats_for(con, files, split_label):
    fl = ", ".join(f"'{f}'" for f in files)
    row = con.execute(f"""
        SELECT
            COUNT(DISTINCT datetime::DATE) AS num_days,
            MIN(datetime::DATE)            AS min_date,
            MAX(datetime::DATE)            AS max_date,
            COUNT(DISTINCT instrument)     AS num_instruments,
            COUNT(*)                       AS total_rows
        FROM read_parquet([{fl}])
    """).fetchone()

    minutes_df = con.execute(f"""
        SELECT datetime::DATE AS d, COUNT(DISTINCT datetime::TIME) AS cnt
        FROM read_parquet([{fl}])
        GROUP BY d
        ORDER BY d
    """).fetchdf()

    mins = minutes_df["cnt"].tolist()
    disk = sum(os.path.getsize(f) for f in files)

    return {
        "split": split_label,
        "num_days": row[0],
        "date_range": (row[1], row[2]),
        "num_instruments": row[3],
        "total_rows": row[4],
        "minutes_per_day": {
            "min": min(mins),
            "max": max(mins),
            "avg": sum(mins) / len(mins),
        },
        "disk_gb": disk / 1024**3,
    }


def _compute_sample_bytes(files):
    pf = pq.ParquetFile(files[0])
    table = pf.read_row_group(0)
    return table.nbytes / table.num_rows, pf.metadata.num_columns - 2


def analyze_dataset(dataset_name: str, num_workers: int = 4) -> dict:
    t0 = time.time()
    data_dir = os.path.join(GOLDEN_DATA_DIR, dataset_name)
    splits = ["train", "valid", "test"]
    files = {s: os.path.join(data_dir, f"{s}.parquet") for s in splits}

    con = duckdb.connect(":memory:")
    con.execute(f"SET threads = {num_workers}")

    per_split = []
    for s in splits:
        per_split.append(_stats_for(con, [files[s]], s))

    total_stats = _stats_for(con, list(files.values()), "total")
    con.close()

    sample_per_row, n_feat = _compute_sample_bytes(list(files.values()))
    elapsed = time.time() - t0

    return {
        "name": dataset_name,
        "num_features": n_feat,
        "est_memory_gb": total_stats["total_rows"] * sample_per_row / 1024**3,
        "elapsed_sec": elapsed,
        "splits": per_split + [total_stats],
    }


def _print_stats(info):
    print(f"{'='*60}")
    print(f"  Dataset: {info['name']}")
    print(f"{'='*60}")
    for sp in info["splits"]:
        label = f"[{sp['split']}]"
        print(f"  {label:<10} {sp['num_days']:>4} days  "
              f"{str(sp['date_range'][0]):>12} ~ {sp['date_range'][1]}  "
              f"{sp['num_instruments']:>5} stocks  "
              f"{sp['total_rows']:>12,} rows  "
              f"m/d {sp['minutes_per_day']['avg']:.0f} "
              f"({sp['minutes_per_day']['min']}-{sp['minutes_per_day']['max']})  "
              f"{sp['disk_gb']:.2f} GB")
    print(f"  {'':12} Feature columns:  {info['num_features']}")
    print(f"  {'':12} Est. memory:      {info['est_memory_gb']:.3f} GB")
    print(f"  {'':12} Read time:        {info['elapsed_sec']:.1f} s")
    print()


if __name__ == "__main__":
    for name in ["alpha158", "hfh12"]:
        info = analyze_dataset(name, num_workers=14)
        _print_stats(info)
