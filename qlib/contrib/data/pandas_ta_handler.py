# Copyright (c) Albus.
# Licensed under the MIT License.

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec
from ...data.dataset.processor import Processor


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class AlphaPTADL(QlibDataLoader):
    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        fields = []
        names = []

        # Section A: kbar (9) — reused from Alpha158
        fields += [
            "($close-$open)/$open",
            "($high-$low)/$open",
            "($close-$open)/($high-$low+1e-12)",
            "($high-Greater($open, $close))/$open",
            "($high-Greater($open, $close))/($high-$low+1e-12)",
            "(Less($open, $close)-$low)/$open",
            "(Less($open, $close)-$low)/($high-$low+1e-12)",
            "(2*$close-$high-$low)/$open",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
        ]
        names += ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]

        # Section B: price (4)
        for field in ["OPEN", "HIGH", "LOW", "VWAP"]:
            f = field.lower()
            fields += [f"${f}/$close"]
            names += [f"{field}0"]

        # Section C: overlap — SMA, EMA, WMA, DEMA, KAMA x [10, 30]
        for ma_name in ["SMA", "EMA", "WMA", "DEMA", "KAMA"]:
            for d in [10, 30]:
                fields += [f"PTA_{ma_name}($close, {d})/$close - 1"]
                names += [f"{ma_name}{d}"]

        # Section D: momentum
        for d in [10, 14]:
            fields += [f"(PTA_RSI($close, {d}) - 50)/100"]
            names += [f"RSI{d}"]
        for d in [10]:
            fields += [f"PTA_ROC($close, {d})/100"]
            names += [f"ROC{d}"]
        fields += [f"PTA_CMO($close, 14)/100"]; names += ["CMO14"]
        fields += [f"(PTA_WILLR($high, $low, $close, 14) + 50)/100"]; names += ["WILLR14"]
        fields += [f"PTA_CCI($high, $low, $close, 14)/200"]; names += ["CCI14"]
        fields += [f"PTA_MFI($high, $low, $close, $volume, 14)/100 - 0.5"]; names += ["MFI14"]
        fields += [f"PTA_MACD($close, 12, 26, 9)/$close"]; names += ["MACD_12_26_9"]
        fields += [f"PTA_MACDh($close, 12, 26, 9)/$close"]; names += ["MACDh_12_26_9"]
        fields += [f"PTA_MACDs($close, 12, 26, 9)/$close"]; names += ["MACDs_12_26_9"]
        fields += [f"PTA_STOCHk($high, $low, $close, 14)/100"]; names += ["STOCHk14"]
        fields += [f"PTA_STOCHd($high, $low, $close, 14)/100"]; names += ["STOCHd14"]

        # Section E: volatility
        fields += [f"PTA_ATR($high, $low, $close, 14)/$close"]; names += ["ATR14"]
        fields += [f"PTA_NATR($high, $low, $close, 14)/100"]; names += ["NATR14"]
        fields += [f"PTA_BBU($close, 20, 2.0)/$close"]; names += ["BBU20"]
        fields += [f"PTA_BBM($close, 20, 2.0)/$close"]; names += ["BBM20"]
        fields += [f"PTA_BBL($close, 20, 2.0)/$close"]; names += ["BBL20"]
        fields += [f"PTA_STDEV($close, 20, 1.0)/$close"]; names += ["STDEV20"]
        fields += [f"PTA_TRANGE($high, $low, $close)/$close"]; names += ["TRANGE"]

        # Section F: trend
        for fn, name in [
            ("PTA_ADX", "ADX"), ("PTA_PLUS_DI", "PLUS_DI"), ("PTA_MINUS_DI", "MINUS_DI"),
            ("PTA_DX", "DX"),
        ]:
            fields += [f"{fn}($high, $low, $close, 14)/100"]
            names += [f"{name}14"]
        fields += [f"PTA_AROONU($high, $low, 14)/100"]; names += ["AROONU14"]
        fields += [f"PTA_AROOND($high, $low, 14)/100"]; names += ["AROOND14"]
        fields += [f"PTA_AROONOSC($high, $low, 14)/100"]; names += ["AROONOSC14"]
        fields += [f"($close - PTA_SAR($high, $low, 0.02, 0.2))/$close"]; names += ["SAR"]

        # Section G: volume
        fields += [f"Delta(PTA_OBV($close, $volume), 5)/(Mean($volume, 5)+1e-12)"]; names += ["OBV"]
        fields += [f"Delta(PTA_AD($high, $low, $close, $volume), 5)/(Mean($volume, 5)+1e-12)"]; names += ["AD"]
        fields += [f"PTA_ADOSC($high, $low, $close, $volume, 3, 10)"]; names += ["ADOSC_3_10"]

        # Section H: statistics
        fields += [f"PTA_BETA($close, $volume, 5)"]; names += ["BETA5"]
        fields += [f"PTA_CORREL($close, Log($volume+1), 5)"]; names += ["CORREL5"]
        fields += [f"PTA_LRSLOPE($close, 10)/$close"]; names += ["LRSLOPE10"]
        fields += [f"PTA_LRANGLE($close, 10)/90"]; names += ["LRANGLE10"]
        fields += [f"PTA_HT_TRENDLINE($close)/$close - 1"]; names += ["HT_TRENDLINE"]

        return fields, names


class AlphaPTA(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="1min",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        return AlphaPTADL.get_feature_config()

    def get_label_config(self):
        return ["Ref($close, -5)/Ref($close, -1) - 1"], ["LABEL0"]


class AlphaPTAvwap(AlphaPTA):
    def get_label_config(self):
        return ["Ref($vwap, -5)/Ref($vwap, -1) - 1"], ["LABEL0"]
