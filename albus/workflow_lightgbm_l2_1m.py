import os
import sys
import json
import pandas as pd
import qlib
from datetime import datetime
from qlib.constant import REG_HK
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, HFSignalRecord, PortAnaRecord
from qlib.contrib.report import analysis_model, analysis_position
import plotly.graph_objects as go

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()


_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(_out_dir, exist_ok=True)
_log_path = os.path.join(_out_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f"[Log] Saving to: {_log_path}")

PROVIDER_URI = os.path.join(os.path.dirname(__file__), "..", "qlib_data", "silver_data_1min")

START_TIME = "2026-05-22 09:30:00"
TRAIN_END = "2026-06-09 15:59:00"
VALID_END = "2026-06-15 15:59:00"
END_TIME = "2026-06-22 15:59:00"

DATASET_CONFIG = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": START_TIME,
                "end_time": END_TIME,
                "fit_start_time": START_TIME,
                "fit_end_time": TRAIN_END,
                "instruments": "all",
                "freq": "1min",
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
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
            "train": (START_TIME, TRAIN_END),
            "valid": (TRAIN_END, VALID_END),
            "test": (VALID_END, END_TIME),
        },
    },
}

MODEL_CONFIG = {
    # "class": "HFLGBModel",
    # "module_path": "qlib.contrib.model.highfreq_gdbt_model",
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "learning_rate": 0.01,
        "max_depth": 8,
        "num_leaves": 150,
        "lambda_l1": 1.5,
        "lambda_l2": 1,
        "num_threads": 20,
    },
}

# MODEL_CONFIG = {
#     # "class": "HFLGBModel",
#     # "module_path": "qlib.contrib.model.highfreq_gdbt_model",
#     "class": "LGBModel",
#     "module_path": "qlib.contrib.model.gbdt",
#     "kwargs": {
#         "loss": "mse",
#         "colsample_bytree": 0.8879,
#         "learning_rate": 0.0421,
#         "subsample": 0.8789,
#         "lambda_l1": 205.6999,
#         "lambda_l2": 580.9768,
#         "max_depth": 8,
#         "num_leaves": 210,
#         "num_threads": 20,
#     },
# }



def _fix_fig(fig):
    return go.Figure(json.loads(fig.to_json()))


def main():
    global _log_file, _log_path
    qlib.init(
        provider_uri=PROVIDER_URI,
        region=REG_HK,
    )

    model = init_instance_by_config(MODEL_CONFIG)
    dataset = init_instance_by_config(DATASET_CONFIG)

    example_df = dataset.prepare("train")
    print(f"Train data shape: {example_df.shape}")

    with R.start(experiment_name="workflow_l2_1min_alpha158"):
        R.log_params(**flatten_dict({"model": MODEL_CONFIG, "dataset": DATASET_CONFIG}))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        ba_rid = recorder.id
        print(f"Recorder id: {ba_rid}")

        _mlflow_dir = recorder.get_local_dir()
        _final_log = os.path.join(_mlflow_dir, "train.log")
        _log_file.flush()
        _log_file.close()
        os.rename(_log_path, _final_log)
        _log_file = open(_final_log, "a")
        sys.stdout = _Tee(sys.__stdout__, _log_file)
        sys.stderr = _Tee(sys.__stderr__, _log_file)
        print(f"[Log] Moved to: {_final_log}")

        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = HFSignalRecord(recorder)
        sar.generate()

        # === Portfolio Analysis ===
        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                },
            },
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "signal": (model, dataset),
                    "topk": 50,
                    "n_drop": 5,
                },
            },
            "backtest": {
                "start_time": VALID_END,
                "end_time": "2026-06-18 15:59:00",
                "account": 100000000,
                "benchmark": "02800",
                "exchange_kwargs": {
                    "freq": "1min",
                    "limit_threshold": None,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        par = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq="day")
        par.generate()

        # === Analysis Graphs ===
        out_dir = _mlflow_dir
        os.makedirs(out_dir, exist_ok=True)

        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        label_df = dataset.prepare("test", col_set="label")
        label_df.columns = ["label"]

        try:
            figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
            for i, fig in enumerate(figs):
                _fix_fig(fig).write_image(os.path.join(out_dir, f"report_normal_df_{i}.png"))

            figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
            for i, fig in enumerate(figs):
                _fix_fig(fig).write_image(os.path.join(out_dir, f"risk_analysis_{i}.png"))
        except Exception:
            print("Skipping portfolio graphs: insufficient backtest data for plotting")

        pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
        try:
            figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
            for i, fig in enumerate(figs):
                _fix_fig(fig).write_image(os.path.join(out_dir, f"score_ic_{i}.png"))
        except Exception:
            print("Skipping score_ic graphs: image save failed (missing Chrome/kaleido)")

        try:
            figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
            for i, fig in enumerate(figs):
                _fix_fig(fig).write_image(os.path.join(out_dir, f"model_performance_{i}.png"))
        except Exception:
            print("Skipping model_performance graphs: image save failed (missing Chrome/kaleido)")

        print(f"Done. Experiment: workflow_l2_1min_alpha158, recorder: {ba_rid}")


if __name__ == "__main__":
    main()
