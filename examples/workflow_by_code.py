#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK, get_dataset_config

import pandas as pd
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D

if __name__ == "__main__":
    # use default data
    provider_uri = "./qlib_data/cn_data/qlib_bin"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    
    # # 缩短时间范围以减少内存使用
    # task = {
    #     "model": CSI300_GBDT_TASK["model"],
    #     "dataset": get_dataset_config(
    #         train=("2017-01-01", "2017-12-31"),
    #         valid=("2018-01-01", "2018-12-31"),
    #         test=("2019-01-01", "2020-08-01"),
    #         handler_kwargs={"start_time": "2017-01-01", "end_time": "2020-08-01", "instruments": CSI300_GBDT_TASK["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"]},
    #     ),
    # }
    
    # # import ipdb; ipdb.set_trace()
    
    # # task = {
    # #     "model": CSI300_GBDT_TASK["model"],
    # #     "dataset": get_dataset_config(
    # #         train=("2017-01-01", "2017-01-31"),
    # #         valid=("2018-01-01", "2018-01-31"),
    # #         test=("2019-01-01", "2019-01-31"),
    # #         handler_kwargs={"start_time": "2017-01-01", "end_time": "2020-01-01", "instruments": CSI300_GBDT_TASK["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"]},
    # #     ),
    # # }
    
    # model = init_instance_by_config(task["model"])
    # dataset = init_instance_by_config(task["dataset"])
    
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    #import ipdb; ipdb.set_trace()
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])


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
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())
    
    #import ipdb; ipdb.set_trace()

    # start exp
    with R.start(experiment_name="workflow"):
        import ipdb; ipdb.set_trace()
        R.log_params(**flatten_dict(CSI300_GBDT_TASK))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        print(f"backtest recorder id: {ba_rid}")
        
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        # analyze graphs
        #import ipdb; ipdb.set_trace()
        # recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
        print(recorder)
        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

        # ## analysis position
        
        # import ipdb; ipdb.set_trace()

        # ### report
        # figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
        # for i, fig in enumerate(figs):
        #     fig.write_image(f"report_normal_df_{i}.png")

        # ### risk analysis
        # figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
        # for i, fig in enumerate(figs):
        #     fig.write_image(f"risk_analysis_{i}.png")

        # ### analysis model
        # label_df = dataset.prepare("test", col_set="label")
        # label_df.columns = ["label"]

        # ### score IC
        # pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
        # figs = analysis_position.score_ic_graph(pred_label)
        # for i, fig in enumerate(figs):
        #     fig.write_image(f"score_ic_{i}.png")

        # ### model performance
        # figs = analysis_model.model_performance_graph(pred_label)
        # for i, fig in enumerate(figs):
        #     fig.write_image(f"model_performance_{i}.png")
