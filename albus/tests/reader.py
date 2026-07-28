"""High-performance data reader backed by DuckDB with simple public API."""

import os
import zipfile
from typing import List, Optional, Tuple
from datetime import datetime

import duckdb
import pandas as pd

from .config import (
    BRONZE_DATA_TYPE_TO_FILE,
    DATA_TYPE_TO_DIR_NAME,
    DEFAULT_BRONZE_DIR,
    DEFAULT_DATA_LAKE_DIR,
    SUPPORTED_DATA_TYPES,
    TIME_COLUMN_NAME,
)


class DataLakeReader:
    """DuckDB-based Parquet reader with predicate pushdown support.

    Directly queries Parquet files on disk without loading full datasets
    into memory. Supports filtering by symbol, date, data type and
    intraday time range.
    """

    def __init__(self, base_dir: str) -> None:
        """Initialize reader with data lake root directory.

        Args:
            base_dir: Root path of the target data layer.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.con = duckdb.connect(database=":memory:")

        # for key, value in DUCKDB_SETTINGS.items():
        #     self.con.execute(f"SET {key} = {value};")
        # self.con.execute(f"SET {key} = {value};")

    def _build_partition_path(self, data_type: str, symbol: str, date: str) -> str:
        """Build hive partition path targeting a specific symbol and date.

        Uses the directory-partitioned layout directly instead of a broad
        glob, so DuckDB only discovers files in the relevant partition.

        Args:
            data_type: Logical data type identifier.
            symbol: Stock symbol code.
            date: Trade date in ``YYYY-MM-DD`` format.

        Returns:
            Glob path scoped to one partition.
        """
        dir_name = DATA_TYPE_TO_DIR_NAME[data_type]
        return os.path.join(
            self.base_dir, dir_name,
            f"date={date}", f"symbol={symbol}",
            "*.parquet",
        )

    def query(
        self,
        symbol: str,
        date: str,
        data_type: str = "tickex",
        time_range: Optional[Tuple[str, str]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Execute filtered query against Parquet files.

        Args:
            symbol: Stock symbol code.
            date: Trade date in ``YYYY-MM-DD`` format.
            data_type: Data category identifier, default is ``tickex``.
            time_range: Optional intraday time window (start, end).
            columns: Optional list of columns to load. Loads all if None.

        Returns:
            Filtered result DataFrame.

        Raises:
            ValueError: If ``data_type`` is not supported.
        """
        if data_type not in SUPPORTED_DATA_TYPES:
            raise ValueError(
                f"Unsupported data type: {data_type}. "
                f"Supported values: {SUPPORTED_DATA_TYPES}"
            )

        partition_path = self._build_partition_path(data_type, symbol, date)
        select_cols = "*" if columns is None else ", ".join(columns)

        sql = f"""
            SELECT {select_cols}
            FROM read_parquet('{partition_path}')
            WHERE 1=1
        """

        if time_range:
            start, end = time_range
            sql += f" AND {TIME_COLUMN_NAME} BETWEEN '{start}' AND '{end}'"

        return self.con.execute(sql).fetchdf()

    def close(self) -> None:
        """Release DuckDB connection and associated resources."""
        self.con.close()

    def __enter__(self) -> "DataLakeReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Simplified public API for team usage
# ---------------------------------------------------------------------------

def get_data(
    symbol: str,
    date: str,
    data_type: str = "tickex",
    time_range: Optional[Tuple[str, str]] = None,
    base_dir: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One-line function to fetch data from the data lake.

    This is the primary entry point for other team members. No knowledge
    of Parquet paths or DuckDB internals is required.

    Args:
        symbol: Stock symbol code.
        date: Trade date in ``YYYY-MM-DD`` format.
        data_type: Data category identifier.
        time_range: Optional full datetime window filter (start, end).
        base_dir: Override default data lake root path.
        columns: Optional list of columns to load.

    Returns:
        Filtered result DataFrame.

    Example::
        >>> from datetime import datetime
        >>> s = datetime(2026,6,1,9,30,0)
        >>> e = datetime(2026,6,1,10,30,0)
        >>> df = get_data("00700", "2026-06-01", time_range=(s,e))
    """
    base_dir = base_dir or DEFAULT_DATA_LAKE_DIR
    with DataLakeReader(base_dir) as reader:
        return reader.query(symbol, date, data_type, time_range, columns=columns)


def get_bronze_data(
    symbol: str,
    date: str,
    data_type: str = "tickex",
    base_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Read raw data directly from the bronze (raw) layer.

    Bronze data is stored as CSV files inside per-symbol zip archives::

        <bronze_root>/<date>/HKSE_<symbol>/<data_type>.csv.zip

    Args:
        symbol: Stock symbol code (e.g. ``"00700"``).
        date: Trade date in ``YYYY-MM-DD`` format.
        data_type: Data type identifier (tickex, trades, etc.).
        base_dir: Override default bronze root path.

    Returns:
        DataFrame with lowercased column names.  Returns an empty DataFrame
        if the file does not exist.

    Raises:
        ValueError: If ``data_type`` is not supported.
    """
    if data_type not in SUPPORTED_DATA_TYPES:
        raise ValueError(
            f"Unsupported data type: {data_type}. "
            f"Supported values: {SUPPORTED_DATA_TYPES}"
        )

    base_dir = base_dir or DEFAULT_BRONZE_DIR
    filename = BRONZE_DATA_TYPE_TO_FILE[data_type]
    zip_path = os.path.join(base_dir, date, f"HKSE_{symbol}", filename)

    if not os.path.isfile(zip_path):
        return pd.DataFrame()

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    df.columns = df.columns.str.lower()
    return df


if __name__ == "__main__":
    # Example usage
    """Local test entry, cover all core parameter combinations."""
    print("===== Test 1: Full tickex data, all columns =====")
    df1 = get_data("00700", "2026-06-01", data_type="tickex")
    print(df1.head(), "\n")

    print("===== Test 2: Specify partial columns =====")
    df2 = get_data(
        "00700",
        "2026-06-01",
        data_type="tickex",
        columns=["timestamp", "symbol", 'price', "volume"]
    )
    print(df2.head(), "\n")

    print("===== Test 3: Filter with full datetime range (date + time) =====")
    start = datetime(2026, 6, 1, 9, 30, 0)
    end = datetime(2026, 6, 1, 10, 30, 0)
    df3 = get_data(
        "00700",
        "2026-06-01",
        data_type="tickex",
        time_range=(start, end),
        columns=["timestamp", "price"]
    )
    print(df3.head(), "\n")

    print("===== Test 4: Switch data type to traderesumes =====")
    df4 = get_data("00700", "2026-06-01", data_type="traderesumes")
    print(df4.head(), "\n")

    # print("===== Test 5: Custom data lake directory =====")
    # custom_path = "./custom_silver"
    # df5 = get_data("00700", "2026-06-01", base_dir=custom_path)
    # print(df5.head())