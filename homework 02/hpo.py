# hpo.py ── complete, self-contained version
import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# ── MLflow experiment ───────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")

# ── helpers ────────────────────────────────────────────────────────────
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ── CLI ────────────────────────────────────────────────────────────────
@click.command()
@click.option("--data_path",  default="./output",
              help="Folder with train.pkl / val.pkl / test.pkl")
@click.option("--num_trials", default=15,
              help="Number of Hyperopt evaluations")
def main(data_path: str, num_trials: int):

    # -------------------------------------------------------------------
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val,   y_val   = load_pickle(os.path.join(data_path, "val.pkl"))
    print(f"[INFO] Loaded data – Train shape {X_train.shape}, Val shape {X_val.shape}")

    # -------------------------------------------------------------------
    def objective(params):
        """
        Train & evaluate one RandomForestRegressor.
        Each call becomes one MLflow run.
        """
        with mlflow.start_run():                  # standalone run
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            rmse   = root_mean_squared_error(y_val, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

            # console feedback
            print(f"[TRIAL] params={params}  ->  rmse={rmse:.4f}")

            return {"loss": rmse, "status": STATUS_OK}

    # -------------------------------------------------------------------
    search_space = {
        "max_depth":        scope.int(hp.quniform("max_depth",        1, 20, 1)),
        "n_estimators":     scope.int(hp.quniform("n_estimators",    10, 50, 1)),
        "min_samples_split":scope.int(hp.quniform("min_samples_split",2, 10, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1,  4, 1)),
        "n_jobs":           -1,
        "random_state":     42,
    }

    print(f"[INFO] Starting optimisation with {num_trials} trials …")
    rstate = np.random.default_rng(42)
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
        show_progressbar=True,
    )
    print(f"[DONE] Best hyper-params: {best}")

# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()