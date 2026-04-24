import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

experiment = "PHTX_Alpha_Main"

client = MlflowClient("http://192.168.31.240:10001")
experiment_id = client.get_experiment_by_name(experiment).experiment_id

# query
# runs = client.search_runs([experiment_id], "params.seed='0'", order_by=["params.dataset DESC", "start_time ASC"])
runs = client.search_runs([experiment_id], order_by=["params.dataset DESC", "start_time ASC"])

# create, populate df
df = pd.DataFrame(columns=["dataset", "model", "seed", "start_time", "acc_test_at_val_best"])
for i, run in enumerate(runs):
    df.loc[len(df)] = [run.data.params["dataset"], run.data.params["model"], run.data.params["seed"], run.info.start_time, run.data.metrics["acc_test_at_val_best"]*100]

# print(df)

# # current seed
# for i, row in df.iterrows():
#     print("{:8}".format(row["dataset"]), "{:6}".format(row["model"]), row["seed"], "{:.2f}".format(row["acc_test_at_val_best"]))
#     if ((i+1)%5)==0:
#         print("\r")

# mean
df_agg = df.groupby(["dataset", "model"], sort=False)["acc_test_at_val_best"].agg(["mean", "std"]).reset_index()
print("{:9} {:7} {}".format("Dataset", "Model", "Accuracy"))
for i, row in df_agg.iterrows():
    if ((i)%5)==0:
        print("\r")
    print("{:9} {:7} {:.2f} ± {:.2f}".format(row["dataset"], row["model"], row["mean"], row["std"]))
    