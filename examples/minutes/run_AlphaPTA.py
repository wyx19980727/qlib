"""
AlphaPTA — TA-Lib based minute-level indicator workflow.

Runs LightGBM on AlphaPTA features (57 indicators) vs 5-bar forward label.
Register TA-Lib ops as qlib custom_ops via qlib.init().

Usage:
    python run_AlphaPTA.py
"""
import os
import sys
import json
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.contrib.report import analysis_model, analysis_position
from qlib.contrib.ops.pandas_ta_ops import PTAOps
import plotly.graph_objects as go

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

PROVIDER_URI = os.path.join(os.path.dirname(__file__), "..", "..", "qlib_data", "silver_data_1min")

BENCHMARK_STOCKS = pd.read_csv(
    os.path.join(PROVIDER_URI, "instruments", "all.txt"),
    sep="\t", header=None, dtype=str
)[0].tolist()

START_TIME = "2026-05-22 09:30:00"
TRAIN_END = "2026-06-11 15:59:00"
VALID_END = "2026-06-16 15:59:00"
END_TIME = "2026-06-18 15:59:00"

DATASET_CONFIG = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "AlphaPTA",
            "module_path": "qlib.contrib.data.pandas_ta_handler",
            "kwargs": {
                "start_time": START_TIME,
                "end_time": END_TIME,
                "fit_start_time": START_TIME,
                "fit_end_time": TRAIN_END,
                "instruments": "all",
                "freq": "1min",
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
            "train": (START_TIME, TRAIN_END),
            "valid": (TRAIN_END, VALID_END),
            "test": (VALID_END, END_TIME),
        },
    },
}

MODEL_CONFIG = {
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


def _fix_fig(fig):
    return go.Figure(json.loads(fig.to_json()))


def main():
    qlib.init(
        provider_uri=PROVIDER_URI,
        region=REG_CN,
        custom_ops=PTAOps,
    )

    print(f"Registered {len(PTAOps)} TA-Lib custom ops.")
    model = init_instance_by_config(MODEL_CONFIG)
    dataset = init_instance_by_config(DATASET_CONFIG)

    example_df = dataset.prepare("train")
    print(f"Train data shape: {example_df.shape}")
    print(f"Feature columns ({example_df.columns.get_level_values(0).nunique()} groups):",
          example_df.columns.get_level_values(0).unique().tolist())

    with R.start(experiment_name="workflow_AlphaPTA_1min"):
        R.log_params(**flatten_dict({"model": MODEL_CONFIG, "dataset": DATASET_CONFIG}))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        ba_rid = recorder.id
        print(f"Recorder id: {ba_rid}")

        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        sar.generate()

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
                "end_time": "2026-06-18",
                "account": 100000000,
                "benchmark": BENCHMARK_STOCKS,
                "exchange_kwargs": {
                    "freq": "1min",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        par = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq="day")
        par.generate()

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
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
        except Exception:
            print("Skipping portfolio graphs: insufficient backtest data")

        pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
        try:
            figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
            for i, fig in enumerate(figs):
                _fix_fig(fig).write_image(os.path.join(out_dir, f"model_performance_{i}.png"))
        except Exception:
            print("Skipping model_performance graphs")

        print(f"Done. Experiment: workflow_AlphaPTA_1min, recorder: {ba_rid}")


if __name__ == "__main__":
    main()
