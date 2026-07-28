import os
import sys
import json
import gc
import shutil
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import qlib
import mlflow
from mlflow.tracking import MlflowClient
from qlib.constant import REG_HK
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, HFSignalRecord, PortAnaRecord
from qlib.contrib.report import analysis_model, analysis_position
from qlib.contrib.data.highfreq_handler import HighFreqGeneralHandler
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from highfreq.highfreq_ops import DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
            except ValueError:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass


class HighFreqWithLabelHandler(HighFreqGeneralHandler):
    def __init__(self, label_expr=None, label_name="LABEL0", **kwargs):
        self.label_expr = label_expr or ["Ref($close, -2) / Ref($close, -1) - 1"]
        self.label_name = label_name
        super().__init__(**kwargs)

    def get_feature_config(self):
        features = super().get_feature_config()
        label = (self.label_expr, [self.label_name])
        return {"feature": features, "label": label}


from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

class CachedDatasetH(DatasetH):
    """DatasetH wrapper backed by precomputed cache."""

    def __init__(self, cache_dict, handler_key, segments_dict):
        # 不调用 super().__init__()

        self._cache = cache_dict
        self._hk = handler_key

        # DatasetH 会用到
        self.segments = segments_dict

        # 保留 DatasetH 的成员，避免某些代码访问时报错
        self.handler = None
        self.fetch_kwargs = {}

    def prepare(
        self,
        segments,
        col_set=None,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ):
        
        if isinstance(segments, str):
            return self._get_one(segments, col_set, data_key)

        elif isinstance(segments, (list, tuple)):
            return [
                self._get_one(seg, col_set, data_key)
                for seg in segments
            ]

        raise NotImplementedError

    def _get_one(self, seg, col_set, data_key):
        key = (self._hk, seg, data_key)

        if key not in self._cache:
            raise KeyError(f"Cache miss: {key}")

        df = self._cache[key]

        if col_set is None:
            return df.copy()

        return df[col_set].copy()


PROVIDER_URI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qlib_data", "silver_data_1min")
CACHE_DIR = os.path.join(PROVIDER_URI, "cache")
MLFLOW_URI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mlruns")
_PRE_COMPUTED_CACHE = {}

# Default (full) time segments
START_TIME = "2026-05-22 09:30:00"
TRAIN_END = "2026-06-09 15:59:00"
VALID_END = "2026-06-15 15:59:00"
END_TIME = "2026-06-22 15:59:00"
BACKTEST_END = "2026-06-18 15:59:00"

# Test (minimal) time segments — 3 days train, 1 day valid, 1 day test
TEST_START_TIME = "2026-05-22 09:30:00"
TEST_TRAIN_END = "2026-05-27 15:59:00"
TEST_VALID_END = "2026-05-28 15:59:00"
TEST_END_TIME = "2026-05-29 15:59:00"
TEST_BACKTEST_END = "2026-05-28 15:59:00"

_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(_out_dir, exist_ok=True)

_LABEL_EXPR = ["Ref($close, -2) / Ref($close, -1) - 1"]
_INFER_PROCS = [
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]
_LEARN_PROCS = [
    {"class": "DropnaLabel"},
    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
]


def build_dataset_config(handler_key, times):
    base = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "segments": {
                "train": (times["start"], times["train_end"]),
                "valid": (times["train_end"], times["valid_end"]),
                "test": (times["valid_end"], times["end"]),
            },
        },
    }
    if handler_key == "hfh":
        base["kwargs"]["handler"] = {
            "class": "HighFreqWithLabelHandler",
            "module_path": "__main__",
            "kwargs": {
                "start_time": times["start"],
                "end_time": times["end"],
                "fit_start_time": times["start"],
                "fit_end_time": times["train_end"],
                "instruments": "all",
                "freq": "1min",
                "day_length": 330,
                "drop_raw": False,
                "infer_processors": _INFER_PROCS,
                "learn_processors": _LEARN_PROCS,
            },
        }
    elif handler_key == "alpha158":
        base["kwargs"]["handler"] = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": times["start"],
                "end_time": times["end"],
                "fit_start_time": times["start"],
                "fit_end_time": times["train_end"],
                "instruments": "all",
                "freq": "1min",
                "label": _LABEL_EXPR,
                "infer_processors": _INFER_PROCS,
                "learn_processors": _LEARN_PROCS,
            },
        }
    return base


PARAMS = {
    "hforiginal": {
        "learning_rate": 0.01,
        "max_depth": 8,
        "num_leaves": 150,
        "lambda_l1": 1.5,
        "lambda_l2": 1,
        "num_threads": 20,
        "early_stopping_rounds": 50,
        "num_boost_round": 1000,
    },
    "qlibdefault": {
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
        "early_stopping_rounds": 50,
        "num_boost_round": 1000,
    },
}

MODELS = {
    "lgbm": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "loss": "mse",
    },
    "hflightgbm": {
        "class": "HFLGBModel",
        "module_path": "qlib.contrib.model.highfreq_gdbt_model",
        "loss": "binary",
    },
}


def build_model_config(model_key, param_key):
    model = MODELS[model_key]
    params = dict(PARAMS[param_key])
    return {
        "class": model["class"],
        "module_path": model["module_path"],
        "kwargs": {
            "loss": model["loss"],
            **params,
        },
    }


EXPERIMENTS = [
    ("qlib_hk_1min_lgbm_hfh_hforiginal",            "hfh",      "lgbm",       "hforiginal"),
    ("qlib_hk_1min_lgbm_hfh_qlibdefault",            "hfh",      "lgbm",       "qlibdefault"),
    ("qlib_hk_1min_hflightgbm_hfh_hforiginal",       "hfh",      "hflightgbm", "hforiginal"),
    ("qlib_hk_1min_hflightgbm_hfh_qlibdefault",      "hfh",      "hflightgbm", "qlibdefault"),
    ("qlib_hk_1min_lgbm_alpha158_hforiginal",        "alpha158", "lgbm",       "hforiginal"),
    ("qlib_hk_1min_lgbm_alpha158_qlibdefault",       "alpha158", "lgbm",       "qlibdefault"),
    ("qlib_hk_1min_hflightgbm_alpha158_hforiginal",  "alpha158", "hflightgbm", "hforiginal"),
    ("qlib_hk_1min_hflightgbm_alpha158_qlibdefault", "alpha158", "hflightgbm", "qlibdefault"),
]


def _clean_experiment(exp_name):
    """Delete existing experiment by name (MLflow) or by named directory (after rename)."""
    named_dir = os.path.join(MLFLOW_URI, exp_name)
    if os.path.exists(named_dir) or os.path.islink(named_dir):
        if os.path.islink(named_dir):
            os.unlink(named_dir)
        else:
            shutil.rmtree(named_dir)
        print(f"  Cleaned old experiment: {exp_name}")

    client = MlflowClient()
    try:
        exp = client.get_experiment_by_name(exp_name)
        if exp is not None:
            from mlflow.exceptions import MlflowException
            try:
                client.delete_experiment(exp.experiment_id)
                print(f"  Cleaned old MLflow experiment: {exp_name}")
            except MlflowException as e:
                print(f"  Note: {e}")
    except Exception:
        pass


def _fix_fig(fig):
    return go.Figure(json.loads(fig.to_json()))


def run_single_experiment(exp_name, handler_key, model_key, param_key, times, dataset=None):
    temp_log = os.path.join(_out_dir, f"{exp_name}.log")
    log_file = open(temp_log, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, log_file)
    sys.stderr = _Tee(old_stderr, log_file)

    model = None
    success = False

    try:
        dataset_config = build_dataset_config(handler_key, times)
        model_config = build_model_config(model_key, param_key)

        print(f"=== Experiment: {exp_name} ===")
        print(f"Handler: {handler_key}, Model: {model_key}, Params: {param_key}")
        print(f"Model config: {json.dumps(model_config, indent=2, default=str)}")

        model = init_instance_by_config(model_config)
        if dataset is None:
            dataset = init_instance_by_config(dataset_config)
            example_df = dataset.prepare("train")
            print(f"Train data shape: {example_df.shape}")
            del example_df

        # _clean_experiment(exp_name)  # Not needed — mlruns/ is cleaned at startup
        with R.start(experiment_name=exp_name):
            R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))
            evals_result = {}
            model.fit(dataset, verbose_eval=50, evals_result=evals_result)
            best_iter = getattr(model.model, "best_iteration", None)
            if best_iter is not None:
                print(f"[Training] Early stopping at round {best_iter}, best_score={model.model.best_score}")
            R.save_objects(**{"params.pkl": model})

            recorder = R.get_recorder()
            ba_rid = recorder.id
            print(f"Recorder id: {ba_rid}")

            mlflow_dir = recorder.get_local_dir()
            final_log = os.path.join(mlflow_dir, "train.log")
            log_file.flush()
            log_file.close()
            os.rename(temp_log, final_log)
            log_file = open(final_log, "a")
            sys.stdout = _Tee(old_stdout, log_file)
            sys.stderr = _Tee(old_stderr, log_file)
            print(f"[Log] Moved to: {final_log}")

            # Save loss history
            if evals_result:
                loss_rows = []
                for ds_name, metrics in evals_result.items():
                    if isinstance(metrics, dict):
                        for metric_name, values in metrics.items():
                            for step, val in enumerate(values):
                                loss_rows.append({"step": step, "dataset": ds_name, "metric": metric_name, "value": val})
                    elif isinstance(metrics, (list, tuple)):
                        for step, val in enumerate(metrics):
                            loss_rows.append({"step": step, "dataset": ds_name, "metric": "loss", "value": val})
                loss_df = pd.DataFrame(loss_rows)
                loss_df.to_csv(os.path.join(mlflow_dir, "evals_result.csv"), index=False)
                print(f"[Loss] Saved evals_result.csv ({len(loss_rows)} rows)")

                try:
                    fig = go.Figure()
                    for ds_name, metrics in evals_result.items():
                        if isinstance(metrics, dict):
                            for metric_name, values in metrics.items():
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(values))), y=values,
                                    mode="lines", name=f"{ds_name}_{metric_name}"
                                ))
                        elif isinstance(metrics, (list, tuple)):
                            fig.add_trace(go.Scatter(
                                x=list(range(len(metrics))), y=metrics,
                                mode="lines", name=ds_name
                            ))
                    fig.update_layout(
                        title="Training / Validation Loss",
                        xaxis_title="Boosting round",
                        yaxis_title="Loss",
                    )
                    fig.write_image(os.path.join(mlflow_dir, "loss_curve.png"))
                    print(f"[Loss] Saved loss_curve.png")
                except Exception as e:
                    print(f"[Loss] Skipping loss curve plot: {e}")
            else:
                print("[Loss] No evals_result available")

            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # CachedDatasetH is not DatasetH → SignalRecord skips saving label.pkl
            if not isinstance(dataset, DatasetH):
                raw_label = dataset.prepare("test", col_set="label")
                print(f"[Label] raw_label type={type(raw_label).__name__}, "
                      f"shape={raw_label.shape if hasattr(raw_label, 'shape') else 'N/A'}")
                recorder.save_objects(**{"label.pkl": raw_label})

            try:
                sar = HFSignalRecord(recorder)
                sar.generate()
            except Exception as e:
                print(f"HFSignalRecord analysis failed: {e}")

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
                    "start_time": times["valid_end"],
                    "end_time": times["backtest_end"],
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

            try:
                par = PortAnaRecord(recorder, port_analysis_config, risk_analysis_freq="day")
                par.generate()
            except Exception as e:
                print(f"Portfolio analysis failed: {e}")

            out_dir = mlflow_dir

            pred_df = recorder.load_object("pred.pkl")
            label_df = dataset.prepare("test", col_set="label")
            label_df.columns = ["label"]

            try:
                report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
                analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
                figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
                for i, fig in enumerate(figs):
                    _fix_fig(fig).write_image(os.path.join(out_dir, f"report_normal_df_{i}.png"))
                figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
                for i, fig in enumerate(figs):
                    _fix_fig(fig).write_image(os.path.join(out_dir, f"risk_analysis_{i}.png"))
            except Exception:
                print("Skipping portfolio graphs")

            pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
            try:
                figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
                for i, fig in enumerate(figs):
                    _fix_fig(fig).write_image(os.path.join(out_dir, f"score_ic_{i}.png"))
            except Exception:
                print("Skipping score_ic graphs")

            try:
                figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
                for i, fig in enumerate(figs):
                    _fix_fig(fig).write_image(os.path.join(out_dir, f"model_performance_{i}.png"))
            except Exception:
                print("Skipping model_performance graphs")

            shutil.copy(__file__, os.path.join(mlflow_dir, "albus_bash_run.py"))
            config_snapshot = {
                "experiment_name": exp_name,
                "handler": handler_key,
                "model": model_key,
                "params": param_key,
                "model_config": model_config,
                "dataset_config": dataset_config,
                "time_segments": times,
            }
            with open(os.path.join(mlflow_dir, "experiment_config.json"), "w") as f:
                json.dump(config_snapshot, f, indent=2, default=str)

            print(f"Done. Experiment: {exp_name}, recorder: {ba_rid}")
            success = True

        # Rename MLflow numeric ID directory to experiment name
        exp_id_dir = os.path.join(MLFLOW_URI, str(recorder.experiment_id))
        named_dir = os.path.join(MLFLOW_URI, exp_name)
        if os.path.isdir(exp_id_dir) and not os.path.exists(named_dir):
            os.rename(exp_id_dir, named_dir)
            print(f"  Renamed: {recorder.experiment_id} → {exp_name}")

    except Exception as e:
        print(f"ERROR in experiment {exp_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if model is not None:
            del model
        gc.collect()
        print(f"  Cleanup done for {exp_name}")
        return success


def _precompute_handler(handler_key, times, use_disk_cache):
    """Pre-compute features for a handler type, cache in memory and optionally on disk."""
    print(f"\n{'='*60}")
    print(f"Pre-computing features for handler: {handler_key}")
    print(f"{'='*60}")

    dataset_config = build_dataset_config(handler_key, times)
    dataset = init_instance_by_config(dataset_config)

    segments_dict = {
        "train": (times["start"], times["train_end"]),
        "valid": (times["train_end"], times["valid_end"]),
        "test": (times["valid_end"], times["end"]),
    }

    cache = {}
    os.makedirs(CACHE_DIR, exist_ok=True)

    for seg in ["train", "valid", "test"]:
        for dk_key, dk_val in [("DK_L", DataHandlerLP.DK_L)]:
            cache_key = (handler_key, seg, dk_val)
            cache_path = os.path.join(CACHE_DIR, f"{handler_key}_{seg}_{dk_key}.parquet")

            if use_disk_cache and os.path.exists(cache_path):
                print(f"  Loading {handler_key}/{seg}/{dk_key} from disk...")
                df = pd.read_parquet(cache_path)
            else:
                print(f"  Computing {handler_key}/{seg}/{dk_key}...")
                df = dataset.prepare(seg, col_set=["feature", "label"], data_key=dk_val)
                if use_disk_cache:
                    print(f"  Saving to disk cache...")
                    df.to_parquet(cache_path)

            cache[cache_key] = df
            print(f"    Shape: {df.shape}")

        # DK_I only for test (used by predict)
        if seg == "test":
            dk_key, dk_val = "DK_I", DataHandlerLP.DK_I
            cache_key = (handler_key, seg, dk_val)
            cache_path = os.path.join(CACHE_DIR, f"{handler_key}_{seg}_{dk_key}.parquet")

            if use_disk_cache and os.path.exists(cache_path):
                print(f"  Loading {handler_key}/{seg}/{dk_key} from disk...")
                df = pd.read_parquet(cache_path)
            else:
                print(f"  Computing {handler_key}/{seg}/{dk_key}...")
                df = dataset.prepare(seg, col_set=["feature", "label"], data_key=dk_val)
                if use_disk_cache:
                    df.to_parquet(cache_path)

            cache[cache_key] = df
            print(f"    Shape: {df.shape}")

    del dataset
    gc.collect()
    _PRE_COMPUTED_CACHE[handler_key] = cache
    print(f"Handler {handler_key} cache ready.")
    print(f"  Memory cache entries: {len(cache)} ({sum(df.memory_usage(deep=True).sum() for df in cache.values()) // 1024**3} GB)")


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--test", action="store_true", help="Quick test with minimal time range")
    parser.add_argument("--precompute", action="store_true",
                        help="Force pre-compute features (recompute disk cache)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable both disk and in-memory caching")
    args = parser.parse_args()

    if args.test:
        times = {
            "start": TEST_START_TIME,
            "train_end": TEST_TRAIN_END,
            "valid_end": TEST_VALID_END,
            "end": TEST_END_TIME,
            "backtest_end": TEST_BACKTEST_END,
        }
        mode = "TEST"
    else:
        times = {
            "start": START_TIME,
            "train_end": TRAIN_END,
            "valid_end": VALID_END,
            "end": END_TIME,
            "backtest_end": BACKTEST_END,
        }
        mode = "FULL"

    use_disk_cache = not args.no_cache
    use_mem_cache = not args.no_cache

    if args.precompute:
        use_disk_cache = True

    print(f"Starting batch experiments: {len(EXPERIMENTS)} total ({mode} mode)")
    print(f"Time segments: train={times['start']}~{times['train_end']}, "
          f"valid={times['train_end']}~{times['valid_end']}, "
          f"test={times['valid_end']}~{times['end']}")
    print(f"Backtest: {times['valid_end']}~{times['backtest_end']}")
    print(f"Disk cache: {'ON' if use_disk_cache else 'OFF'}, "
          f"Memory cache: {'ON' if use_mem_cache else 'OFF'}")
    print()

    # Suppress MLflow noise and start fresh
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("qlib").setLevel(logging.WARNING)
    if os.path.exists(MLFLOW_URI):
        shutil.rmtree(MLFLOW_URI)
    os.makedirs(MLFLOW_URI, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URI)
    qlib.init(
        provider_uri=PROVIDER_URI,
        region=REG_HK,
        custom_ops=[DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut],
    )

    results = []
    seen_handlers = set()

    for i, (exp_name, handler_key, model_key, param_key) in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Starting: {exp_name}")

        if handler_key not in seen_handlers and use_mem_cache:
            # First time for this handler: pre-compute and cache
            _precompute_handler(handler_key, times, use_disk_cache)
            seen_handlers.add(handler_key)

        if use_mem_cache and handler_key in seen_handlers:
            # Reuse from in-memory cache
            handler_cache = _PRE_COMPUTED_CACHE.setdefault(handler_key, {})
            segments_dict = {
                "train": (times["start"], times["train_end"]),
                "valid": (times["train_end"], times["valid_end"]),
                "test": (times["valid_end"], times["end"]),
            }
            dataset = CachedDatasetH(handler_cache, handler_key, segments_dict)
            success = run_single_experiment(
                exp_name, handler_key, model_key, param_key, times,
                dataset=dataset
            )
        else:
            success = run_single_experiment(
                exp_name, handler_key, model_key, param_key, times
            )

        status = "OK" if success else "FAIL"
        results.append((exp_name, status))
        print(f"[{i+1}/{len(EXPERIMENTS)}] {status}: {exp_name}")

    print(f"\nAll {len(EXPERIMENTS)} experiments completed.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'#':<4} {'Experiment':<52} {'Status':<8}")
    print("-" * 70)
    for i, (exp_name, status) in enumerate(results):
        print(f"  {i+1:<4} {exp_name:<52} {status:<8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
