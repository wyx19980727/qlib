# Copyright (c) Albus.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import talib

from qlib.data.base import Expression, ExpressionOps
from qlib.data.ops import ElemOperator, Operators


class _PTABase(ExpressionOps):
    _lookback = 0

    def get_longest_back_rolling(self):
        return self._lookback

    def get_extended_window_size(self):
        lft, rgt = 0, 0
        for feat in self._features:
            l, r = feat.get_extended_window_size()
            lft = max(lft, l)
            rgt = max(rgt, r)
        return lft + self._lookback, rgt


class _PTA1(_PTABase):
    """Single-feature TA-Lib op with timeperiod."""

    def __init__(self, feature, timeperiod=14):
        self._timeperiod = timeperiod
        self._lookback = timeperiod
        self._features = [feature]
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self._features[0].load(instrument, start_index, end_index, freq)
        arr = np.asarray(series, dtype=np.float64)
        result = self._talib_fn(arr, self._timeperiod)
        return pd.Series(result, index=series.index)


class _PTA1NoPeriod(_PTABase):
    """Single-feature TA-Lib op without period param (e.g. HT_TRENDLINE)."""

    def __init__(self, feature):
        self._lookback = 0
        self._features = [feature]
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self._features[0].load(instrument, start_index, end_index, freq)
        arr = np.asarray(series, dtype=np.float64)
        result = self._talib_fn(arr)
        return pd.Series(result, index=series.index)


class _PTA1Stddev(_PTABase):
    """STDDEV: close, timeperiod, nbdev."""

    def __init__(self, feature, timeperiod=5, nbdev=1.0):
        self._lookback = timeperiod
        self._features = [feature]
        self._timeperiod = timeperiod
        self._nbdev = nbdev
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self._features[0].load(instrument, start_index, end_index, freq)
        arr = np.asarray(series, dtype=np.float64)
        result = talib.STDDEV(arr, self._timeperiod, self._nbdev)
        return pd.Series(result, index=series.index)


class _PTA3(_PTABase):
    """HLC triple-feature TA-Lib op (ATR, ADX, CCI, WILLR, NATR, PLUS_DI, MINUS_DI, DX)."""

    def __init__(self, high, low, close, timeperiod=14):
        self._lookback = timeperiod
        self._features = [high, low, close]
        self._timeperiod = timeperiod
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        arrays = []
        for feat in self._features:
            s = feat.load(instrument, start_index, end_index, freq)
            arrays.append(np.asarray(s, dtype=np.float64))
        result = self._talib_fn(*arrays, self._timeperiod)
        return pd.Series(result, index=self._features[0].load(instrument, start_index, end_index, freq).index)


class _PTA2(_PTABase):
    """Pair-feature TA-Lib op (OBV, BETA, CORREL)."""

    def __init__(self, feature_left, feature_right, timeperiod=14):
        self._lookback = timeperiod if hasattr(self, '_no_lookback') else timeperiod
        self._features = [feature_left, feature_right]
        self._timeperiod = timeperiod
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        s0 = self._features[0].load(instrument, start_index, end_index, freq)
        s1 = self._features[1].load(instrument, start_index, end_index, freq)
        a0 = np.asarray(s0, dtype=np.float64)
        a1 = np.asarray(s1, dtype=np.float64)
        result = self._talib_fn(a0, a1, self._timeperiod)
        return pd.Series(result, index=s0.index)


class _PTA2NoPeriod(_PTABase):
    """Pair-feature, no period (OBV)."""

    def __init__(self, feature_left, feature_right):
        self._lookback = 1
        self._features = [feature_left, feature_right]
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        s0 = self._features[0].load(instrument, start_index, end_index, freq)
        s1 = self._features[1].load(instrument, start_index, end_index, freq)
        a0 = np.asarray(s0, dtype=np.float64)
        a1 = np.asarray(s1, dtype=np.float64)
        result = self._talib_fn(a0, a1)
        return pd.Series(result, index=s0.index)


class _PTA4(_PTABase):
    """OHLCV quadruple-feature TA-Lib op (MFI, AD, ADOSC)."""

    def __init__(self, high, low, close, volume, timeperiod=14):
        self._lookback = timeperiod
        self._features = [high, low, close, volume]
        self._timeperiod = timeperiod
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        arrays = []
        for feat in self._features:
            s = feat.load(instrument, start_index, end_index, freq)
            arrays.append(np.asarray(s, dtype=np.float64))
        result = self._talib_fn(*arrays, self._timeperiod)
        idx = self._features[0].load(instrument, start_index, end_index, freq).index
        return pd.Series(result, index=idx)


class _PTA4NoPeriod(_PTABase):
    """OHLCV, no period (AD)."""

    def __init__(self, high, low, close, volume):
        self._lookback = 1
        self._features = [high, low, close, volume]
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        arrays = []
        for feat in self._features:
            s = feat.load(instrument, start_index, end_index, freq)
            arrays.append(np.asarray(s, dtype=np.float64))
        result = self._talib_fn(*arrays)
        idx = self._features[0].load(instrument, start_index, end_index, freq).index
        return pd.Series(result, index=idx)


class _PTA4ADOSC(_PTABase):
    """ADOSC: high, low, close, volume, fastperiod, slowperiod."""

    def __init__(self, high, low, close, volume, fastperiod=3, slowperiod=10):
        self._lookback = slowperiod
        self._features = [high, low, close, volume]
        self._fastperiod = fastperiod
        self._slowperiod = slowperiod
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        arrays = []
        for feat in self._features:
            s = feat.load(instrument, start_index, end_index, freq)
            arrays.append(np.asarray(s, dtype=np.float64))
        result = talib.ADOSC(*arrays, self._fastperiod, self._slowperiod)
        idx = self._features[0].load(instrument, start_index, end_index, freq).index
        return pd.Series(result, index=idx)


class _PTAHL2(_PTABase):
    """HL pair (AROON, AROONOSC)."""

    def __init__(self, high, low, timeperiod=14):
        self._lookback = timeperiod
        self._features = [high, low]
        self._timeperiod = timeperiod
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        s_h = self._features[0].load(instrument, start_index, end_index, freq)
        s_l = self._features[1].load(instrument, start_index, end_index, freq)
        a_h = np.asarray(s_h, dtype=np.float64)
        a_l = np.asarray(s_l, dtype=np.float64)
        result = self._talib_fn(a_h, a_l, self._timeperiod)
        return pd.Series(result, index=s_h.index)


class _PTASAR(_PTABase):
    """PSAR: high, low, acceleration, maximum."""

    def __init__(self, high, low, acceleration=0.02, maximum=0.2):
        self._lookback = 1
        self._features = [high, low]
        self._acceleration = acceleration
        self._maximum = maximum
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        s_h = self._features[0].load(instrument, start_index, end_index, freq)
        s_l = self._features[1].load(instrument, start_index, end_index, freq)
        a_h = np.asarray(s_h, dtype=np.float64)
        a_l = np.asarray(s_l, dtype=np.float64)
        result = talib.SAR(a_h, a_l, self._acceleration, self._maximum)
        return pd.Series(result, index=s_h.index)

# ========== MACD (multi-output: 0=MACD, 1=MACDsignal, 2=MACDhist) ==========

class _MACDBase(_PTABase):
    def __init__(self, feature, fastperiod=12, slowperiod=26, signalperiod=9):
        self._lookback = slowperiod + signalperiod
        self._features = [feature]
        self._fastperiod = fastperiod
        self._slowperiod = slowperiod
        self._signalperiod = signalperiod
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        s = self._features[0].load(instrument, start_index, end_index, freq)
        a = np.asarray(s, dtype=np.float64)
        result = talib.MACD(a, self._fastperiod, self._slowperiod, self._signalperiod)
        return pd.Series(result[self._output_idx], index=s.index)

class PTA_MACD(_MACDBase): _output_idx = 0
class PTA_MACDh(_MACDBase): _output_idx = 2
class PTA_MACDs(_MACDBase): _output_idx = 1

# ========== STOCH (multi-output: 0=slowk, 1=slowd) ==========

class _STOCHBase(_PTABase):
    def __init__(self, high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
        self._lookback = fastk_period + slowd_period
        self._features = [high, low, close]
        self._fastk_period = fastk_period
        self._slowk_period = slowk_period
        self._slowk_matype = slowk_matype
        self._slowd_period = slowd_period
        self._slowd_matype = slowd_matype
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        arrays = []
        for feat in self._features:
            s = feat.load(instrument, start_index, end_index, freq)
            arrays.append(np.asarray(s, dtype=np.float64))
        result = talib.STOCH(*arrays, self._fastk_period, self._slowk_period, self._slowk_matype, self._slowd_period, self._slowd_matype)
        idx = self._features[0].load(instrument, start_index, end_index, freq).index
        return pd.Series(result[self._output_idx], index=idx)

class PTA_STOCHk(_STOCHBase): _output_idx = 0
class PTA_STOCHd(_STOCHBase): _output_idx = 1

# ========== BBANDS (multi-output: 0=upper, 1=middle, 2=lower) ==========

class _BBANDSBase(_PTABase):
    def __init__(self, feature, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0):
        self._lookback = timeperiod
        self._features = [feature]
        self._timeperiod = timeperiod
        self._nbdevup = nbdevup
        self._nbdevdn = nbdevdn
        self._matype = matype
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        s = self._features[0].load(instrument, start_index, end_index, freq)
        a = np.asarray(s, dtype=np.float64)
        result = talib.BBANDS(a, self._timeperiod, self._nbdevup, self._nbdevdn, self._matype)
        return pd.Series(result[self._output_idx], index=s.index)

class PTA_BBU(_BBANDSBase): _output_idx = 0
class PTA_BBM(_BBANDSBase): _output_idx = 1
class PTA_BBL(_BBANDSBase): _output_idx = 2

# ========== AROON (multi-output: 0=aroondown, 1=aroonup) ==========

class _AROONBase(_PTABase):
    def __init__(self, high, low, timeperiod=14):
        self._lookback = timeperiod
        self._features = [high, low]
        self._timeperiod = timeperiod
        ExpressionOps.__init__(self)

    def _load_internal(self, instrument, start_index, end_index, freq):
        s_h = self._features[0].load(instrument, start_index, end_index, freq)
        s_l = self._features[1].load(instrument, start_index, end_index, freq)
        a_h = np.asarray(s_h, dtype=np.float64)
        a_l = np.asarray(s_l, dtype=np.float64)
        result = talib.AROON(a_h, a_l, self._timeperiod)
        idx = s_h.index
        return pd.Series(result[self._output_idx], index=idx)

class PTA_AROOND(_AROONBase): _output_idx = 0
class PTA_AROONU(_AROONBase): _output_idx = 1

# ========== Single-feature ops ==========

class PTA_RSI(_PTA1): _talib_fn = staticmethod(talib.RSI)
class PTA_SMA(_PTA1): _talib_fn = staticmethod(talib.SMA)
class PTA_EMA(_PTA1): _talib_fn = staticmethod(talib.EMA)
class PTA_WMA(_PTA1): _talib_fn = staticmethod(talib.WMA)
class PTA_DEMA(_PTA1): _talib_fn = staticmethod(talib.DEMA)
class PTA_KAMA(_PTA1): _talib_fn = staticmethod(talib.KAMA)
class PTA_ROC(_PTA1): _talib_fn = staticmethod(talib.ROC)
class PTA_CMO(_PTA1): _talib_fn = staticmethod(talib.CMO)
class PTA_HT_TRENDLINE(_PTA1NoPeriod): _talib_fn = staticmethod(talib.HT_TRENDLINE)

class PTA_STDEV(_PTA1Stddev):
    pass

# ========== HLC triple-feature ops ==========

class PTA_ATR(_PTA3): _talib_fn = staticmethod(talib.ATR)
class PTA_NATR(_PTA3): _talib_fn = staticmethod(talib.NATR)
class PTA_WILLR(_PTA3): _talib_fn = staticmethod(talib.WILLR)
class PTA_CCI(_PTA3): _talib_fn = staticmethod(talib.CCI)
class PTA_ADX(_PTA3): _talib_fn = staticmethod(talib.ADX)
class PTA_PLUS_DI(_PTA3): _talib_fn = staticmethod(talib.PLUS_DI)
class PTA_MINUS_DI(_PTA3): _talib_fn = staticmethod(talib.MINUS_DI)
class PTA_DX(_PTA3): _talib_fn = staticmethod(talib.DX)
class PTA_TRANGE(_PTABase):
    _lookback = 1
    def __init__(self, high, low, close):
        self._features = [high, low, close]
        ExpressionOps.__init__(self)
    def _load_internal(self, instrument, start_index, end_index, freq):
        s_h = self._features[0].load(instrument, start_index, end_index, freq)
        s_l = self._features[1].load(instrument, start_index, end_index, freq)
        s_c = self._features[2].load(instrument, start_index, end_index, freq)
        result = talib.TRANGE(np.asarray(s_h, dtype=np.float64), np.asarray(s_l, dtype=np.float64), np.asarray(s_c, dtype=np.float64))
        return pd.Series(result, index=s_h.index)

# ========== HL pair ops ==========

class PTA_AROONOSC(_PTAHL2): _talib_fn = staticmethod(talib.AROONOSC)

# ========== Pair ops (2 features) ==========

class PTA_OBV(_PTA2NoPeriod): _talib_fn = staticmethod(talib.OBV)
class PTA_BETA(_PTA2): _talib_fn = staticmethod(talib.BETA)
class PTA_CORREL(_PTA2): _talib_fn = staticmethod(talib.CORREL)

# ========== Quadruple ops (OHLCV) ==========

class PTA_MFI(_PTA4): _talib_fn = staticmethod(talib.MFI)
class PTA_AD(_PTA4NoPeriod): _talib_fn = staticmethod(talib.AD)

class PTA_ADOSC(_PTA4ADOSC):
    pass

# ========== Linear regression ==========

class _PTA1Linreg(_PTABase):
    def __init__(self, feature, timeperiod=14):
        self._lookback = timeperiod
        self._features = [feature]
        self._timeperiod = timeperiod
        ExpressionOps.__init__(self)

class PTA_LRSLOPE(_PTA1Linreg):
    def _load_internal(self, instrument, start_index, end_index, freq):
        s = self._features[0].load(instrument, start_index, end_index, freq)
        a = np.asarray(s, dtype=np.float64)
        result = talib.LINEARREG_SLOPE(a, self._timeperiod)
        return pd.Series(result, index=s.index)

class PTA_LRANGLE(_PTA1Linreg):
    def _load_internal(self, instrument, start_index, end_index, freq):
        s = self._features[0].load(instrument, start_index, end_index, freq)
        a = np.asarray(s, dtype=np.float64)
        result = talib.LINEARREG_ANGLE(a, self._timeperiod)
        return pd.Series(result, index=s.index)

# ========== SAR ==========

class PTA_SAR(_PTASAR):
    pass


PTAOps = [
    # single-feature (close)
    PTA_RSI,
    PTA_SMA, PTA_EMA, PTA_WMA, PTA_DEMA, PTA_KAMA,
    PTA_ROC, PTA_CMO,
    PTA_HT_TRENDLINE,
    PTA_STDEV,
    PTA_LRSLOPE, PTA_LRANGLE,
    # HLC
    PTA_ATR, PTA_NATR, PTA_WILLR, PTA_CCI,
    PTA_ADX, PTA_PLUS_DI, PTA_MINUS_DI, PTA_DX,
    PTA_TRANGE,
    # HL
    PTA_AROONOSC,
    # OHLCV
    PTA_MFI, PTA_AD, PTA_ADOSC,
    # pair
    PTA_OBV, PTA_BETA, PTA_CORREL,
    # PSAR
    PTA_SAR,
    # multi-output
    PTA_MACD, PTA_MACDh, PTA_MACDs,
    PTA_STOCHk, PTA_STOCHd,
    PTA_BBU, PTA_BBM, PTA_BBL,
    PTA_AROOND, PTA_AROONU,
]
