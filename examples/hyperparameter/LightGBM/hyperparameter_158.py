import qlib
import optuna
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.tests.config import CSI300_DATASET_CONFIG

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def objective(trial):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 1),
                "subsample": trial.suggest_uniform("subsample", 0, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                "max_depth": 10,
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }
    evals_result = dict()
    model = init_instance_by_config(task["model"])
    model.fit(dataset, evals_result=evals_result)
    return min(evals_result["valid"]["l2"])


if __name__ == "__main__":
    provider_uri = "/home/albus/Python_Codes/qlib/qlib_data/cn_data/qlib_bin"
    qlib.init(provider_uri=provider_uri, region="cn")

    dataset = init_instance_by_config(CSI300_DATASET_CONFIG)

    study = optuna.create_study(study_name="LGBM_158", storage="sqlite:///db.sqlite3", load_if_exists=True)
    study.optimize(objective, n_jobs=6, n_trials=100, show_progress_bar=True)
