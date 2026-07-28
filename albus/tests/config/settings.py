"""Global configuration constants for the data lake package."""

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Data type definitions
# ---------------------------------------------------------------------------
SUPPORTED_DATA_TYPES: List[str] = [
    "tickex",
    "traderesumes",
    "orderadd",
    "ordermodifydelete",
]

# Mapping from logical data type name to physical directory name
DATA_TYPE_TO_DIR_NAME: Dict[str, str] = {
    "tickex": "tickex",
    "traderesumes": "traderesumes",
    "orderadd": "orderadd",
    "ordermodifydelete": "ordermodifydelete",
}

# Mapping from logical data type to bronze-layer zip filename
BRONZE_DATA_TYPE_TO_FILE: Dict[str, str] = {
    "tickex": "tickex.csv.zip",
    "traderesumes": "traderesumes.csv.zip",
    "orderadd": "orderadd.csv.zip",
    "ordermodifydelete": "ordermodifydelete.csv.zip",
}

# ---------------------------------------------------------------------------
# Data schema constants
# ---------------------------------------------------------------------------
TIME_COLUMN_NAME: str = "timestamp"

# ---------------------------------------------------------------------------
# DuckDB runtime settings
# ---------------------------------------------------------------------------
# DUCKDB_SETTINGS: Dict[str, str] = {
#     "hive_partitioning": "true",
# }

# ---------------------------------------------------------------------------
# Default path resolution
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_MODULE_DIR)
PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)

# Default silver layer path, overridable via environment variable
DEFAULT_DATA_LAKE_DIR: str = os.environ.get(
    "DATA_LAKE_DIR",
    os.path.join(PROJECT_ROOT, "data", "lake", "silver", 'hk_l2'),
)

# Default bronze (raw) layer path, overridable via environment variable
DEFAULT_BRONZE_DIR: str = os.environ.get(
    "BRONZE_DIR",
    os.path.join(PROJECT_ROOT, "data", "lake", "raw", "HKL2"),
)