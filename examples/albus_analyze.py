import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R

from qlib.contrib.report import analysis_model, analysis_position

qlib.init(provider_uri="./qlib_data/cn_data/qlib_bin", region=REG_CN)

experiment_name = "workflow"

exp = R.get_exp(experiment_name=experiment_name)
runs = exp.list_recorders()
if not runs:
    raise ValueError(f"实验 '{experiment_name}' 下没有 run")

sorted_rids = sorted(runs, key=lambda rid: runs[rid].start_time, reverse=True)
latest_rid = sorted_rids[0]
print(f"可用 runs: {list(runs.keys())}")
print(f"使用最新的 recorder_id = {latest_rid}")

recorder = exp.get_recorder(recorder_id=latest_rid)
print("Recorder:", recorder)

pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

label_df = recorder.load_object("label.pkl")
if label_df is not None and isinstance(label_df, pd.DataFrame):
    label_df = label_df.iloc[:, 0].to_frame("label")
else:
    label_df = pred_df.iloc[:, 0].to_frame("label")
    label_df[:] = None


import json
import plotly.graph_objects as go

def _fix_fig(fig):
    """Serialize via PlotlyJSONEncoder (handles Timestamps), reload -> no more Timestamps."""
    return go.Figure(json.loads(fig.to_json()))


### report
figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
for i, fig in enumerate(figs):
    _fix_fig(fig).write_image(f"report_normal_df_{i}.png")

### risk analysis
figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
for i, fig in enumerate(figs):
    _fix_fig(fig).write_image(f"risk_analysis_{i}.png")

### score IC
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
for i, fig in enumerate(figs):
    _fix_fig(fig).write_image(f"score_ic_{i}.png")

### model performance
figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
for i, fig in enumerate(figs):
    _fix_fig(fig).write_image(f"model_performance_{i}.png")