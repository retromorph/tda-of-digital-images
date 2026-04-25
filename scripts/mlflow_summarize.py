import os

import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

# Override with: MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "PHTX_Alpha_Main")

client = MlflowClient(tracking_uri)
experiment_id = client.get_experiment_by_name(experiment).experiment_id

runs = client.search_runs([experiment_id], order_by=["params.dataset DESC", "start_time ASC"])

df = pd.DataFrame(columns=["dataset", "model", "seed", "start_time", "acc_test_at_val_best"])
for i, run in enumerate(runs):
    df.loc[len(df)] = [
        run.data.params["dataset"],
        run.data.params["model"],
        run.data.params["seed"],
        run.info.start_time,
        run.data.metrics["acc_test_at_val_best"] * 100,
    ]

df_agg = df.groupby(["dataset", "model"], sort=False)["acc_test_at_val_best"].agg(["mean", "std"]).reset_index()
print("{:9} {:7} {}".format("Dataset", "Model", "Accuracy"))
for i, row in df_agg.iterrows():
    if (i % 5) == 0:
        print("\r")
    print("{:9} {:7} {:.2f} ± {:.2f}".format(row["dataset"], row["model"], row["mean"], row["std"]))
