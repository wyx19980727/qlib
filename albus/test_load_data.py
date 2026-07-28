import time
import sys
from pathlib import Path

remote_lake = Path("/data_lake")
sys.path.insert(0, str(remote_lake))

from data_lake.reader import get_data


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[time] {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


if __name__ == "__main__":
    get_data_timed = timeit(get_data)
    df = get_data_timed("00001", "2026-05-22", data_type="tickex")
    print(f"shape: {df.shape}")
    print(df.head())
    print(df.dtypes)
