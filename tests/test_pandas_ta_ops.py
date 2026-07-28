import numpy as np
import pandas as pd
import talib
import pytest


@pytest.fixture(scope="module")
def sample_series():
    n = 200
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    volume = np.random.randint(100000, 500000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    return {
        "close": pd.Series(close, index=idx),
        "high": pd.Series(high, index=idx),
        "low": pd.Series(low, index=idx),
        "volume": pd.Series(volume, index=idx),
    }


def test_talib_basics():
    """Verify TA-Lib is accessible and produces expected output shapes."""
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    rsi = talib.RSI(arr, timeperiod=5)
    assert len(rsi) == 10
    assert np.isnan(rsi[0])
    assert np.isnan(rsi[4])
    assert not np.isnan(rsi[5])


def test_rsi_output(sample_series):
    c = sample_series["close"].values.astype(np.float64)
    rsi = talib.RSI(c, timeperiod=14)
    assert len(rsi) == len(c)
    assert all(0 <= x <= 100 for x in rsi[~np.isnan(rsi)])
    assert sum(np.isnan(rsi)) == 14


def test_macd_multioutput(sample_series):
    c = sample_series["close"].values.astype(np.float64)
    macd, signal, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    assert len(macd) == len(signal) == len(hist) == len(c)
    assert sum(np.isnan(macd)) == 33


def test_bbands_output(sample_series):
    c = sample_series["close"].values.astype(np.float64)
    upper, middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    assert len(upper) == len(middle) == len(lower) == len(c)
    assert np.all(upper[20:] >= middle[20:])
    assert np.all(middle[20:] >= lower[20:])


def test_atr_output(sample_series):
    h = sample_series["high"].values.astype(np.float64)
    l = sample_series["low"].values.astype(np.float64)
    c = sample_series["close"].values.astype(np.float64)
    atr = talib.ATR(h, l, c, timeperiod=14)
    assert len(atr) == len(c)
    assert sum(np.isnan(atr)) == 14
    assert np.all(atr[~np.isnan(atr)] > 0)


def test_adx_output(sample_series):
    h = sample_series["high"].values.astype(np.float64)
    l = sample_series["low"].values.astype(np.float64)
    c = sample_series["close"].values.astype(np.float64)
    adx = talib.ADX(h, l, c, timeperiod=14)
    assert len(adx) == len(c)
    assert all(0 <= x <= 100 for x in adx[~np.isnan(adx)] if not np.isnan(x))


def test_stoch_output(sample_series):
    h = sample_series["high"].values.astype(np.float64)
    l = sample_series["low"].values.astype(np.float64)
    c = sample_series["close"].values.astype(np.float64)
    slowk, slowd = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    assert len(slowk) == len(slowd) == len(c)
    assert all(0 <= x <= 100 for x in slowk[~np.isnan(slowk)])


def test_aroon_output(sample_series):
    h = sample_series["high"].values.astype(np.float64)
    l = sample_series["low"].values.astype(np.float64)
    aroondown, aroonup = talib.AROON(h, l, timeperiod=14)
    assert len(aroondown) == len(aroonup) == len(h)
    assert all(0 <= x <= 100 for x in aroonup[~np.isnan(aroonup)])


def test_sar_output(sample_series):
    h = sample_series["high"].values.astype(np.float64)
    l = sample_series["low"].values.astype(np.float64)
    sar = talib.SAR(h, l, acceleration=0.02, maximum=0.2)
    assert len(sar) == len(h)


def test_obv_output(sample_series):
    c = sample_series["close"].values.astype(np.float64)
    v = sample_series["volume"].values.astype(np.float64)
    obv = talib.OBV(c, v)
    assert len(obv) == len(c)


def test_adosc_output(sample_series):
    h = sample_series["high"].values.astype(np.float64)
    l = sample_series["low"].values.astype(np.float64)
    c = sample_series["close"].values.astype(np.float64)
    v = sample_series["volume"].values.astype(np.float64)
    adosc = talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10)
    assert len(adosc) == len(c)


def test_linearreg_output(sample_series):
    c = sample_series["close"].values.astype(np.float64)
    slope = talib.LINEARREG_SLOPE(c, timeperiod=14)
    angle = talib.LINEARREG_ANGLE(c, timeperiod=14)
    assert len(slope) == len(angle) == len(c)


def test_ht_trendline(sample_series):
    c = sample_series["close"].values.astype(np.float64)
    ht = talib.HT_TRENDLINE(c)
    assert len(ht) == len(c)


@pytest.mark.parametrize("period", [5, 14, 20])
def test_sma_various_periods(sample_series, period):
    c = sample_series["close"].values.astype(np.float64)
    sma = talib.SMA(c, timeperiod=period)
    assert sum(np.isnan(sma)) == period - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
