import argparse
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _metric_col(metric_name):
    return "metrics.{}".format(metric_name)


def _param_col(param_name):
    return "params.{}".format(param_name)


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def expected_experiment_name(root_name, task_name):
    return "{}_{}".format(root_name, task_name)


def gather_rows(cfg, metric_name):
    records = []
    for task in cfg["tasks"]:
        task_name = task["name"]
        exp_name = expected_experiment_name(cfg["experiment"], task_name)

        for method_name, method_cfg in cfg["methods"].items():
            if "project" in method_cfg:
                full_exp_name = "{}_{}".format(method_cfg["project"], exp_name)
            else:
                # Persformer-family scripts use hardcoded PERSFORMER_<experiment>
                full_exp_name = "PERSFORMER_{}".format(exp_name)

            try:
                runs = mlflow.search_runs(
                    experiment_names=[full_exp_name],
                    output_format="pandas",
                )
            except Exception:
                runs = pd.DataFrame()

            if runs.empty:
                continue

            run_filter = method_cfg.get("run_filter", {})
            for k, v in run_filter.items():
                col = _param_col(k)
                if col not in runs.columns:
                    runs = runs.iloc[0:0]
                    break
                runs = runs[runs[col].astype(str) == str(v)]
            if runs.empty:
                continue

            mcol = _metric_col(metric_name)
            scol = _param_col("seed")
            if mcol not in runs.columns:
                continue
            if scol not in runs.columns:
                runs[scol] = None

            for _, row in runs.iterrows():
                records.append(
                    {
                        "task": task_name,
                        "dataset": task["dataset"],
                        "transform": task.get("transform", None),
                        "power": task.get("power", 0.0),
                        "method": method_name,
                        "seed": row.get(scol, None),
                        "metric": row.get(mcol, None),
                        "run_id": row.get("run_id", None),
                        "experiment": full_exp_name,
                    }
                )

    return pd.DataFrame(records)


def summarize(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_valid = df.dropna(subset=["metric"]).copy()
    if df_valid.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    per_task = (
        df_valid.groupby(["task", "method"], as_index=False)["metric"]
        .agg(mean="mean", std="std", n="count")
        .sort_values(["task", "mean"], ascending=[True, False])
    )

    per_task["rank"] = per_task.groupby("task")["mean"].rank(method="min", ascending=False)

    best = per_task[per_task["rank"] == 1].copy()
    win_rate = (
        best.groupby("method", as_index=False)["task"].count().rename(columns={"task": "wins"})
    )
    total_tasks = per_task["task"].nunique()
    win_rate["win_rate"] = win_rate["wins"] / total_tasks

    avg_rank = (
        per_task.groupby("method", as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "avg_rank"})
        .sort_values("avg_rank", ascending=True)
    )

    leaderboard = avg_rank.merge(win_rate, on="method", how="left").fillna({"wins": 0, "win_rate": 0.0})
    return per_task, leaderboard, df_valid


def main():
    parser = argparse.ArgumentParser(description="Aggregate preliminary benchmark MLflow results.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "exp" / "config" / "preliminary_clean.yaml"),
        help="Benchmark config used for runs.",
    )
    parser.add_argument(
        "--metric",
        default="acc_test_at_val_best",
        help="Metric name in MLflow (without 'metrics.' prefix).",
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "exp" / "artifacts"),
        help="Directory for CSV outputs.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    df = gather_rows(cfg, args.metric)
    per_task, leaderboard, raw = summarize(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.config).stem
    raw_path = out_dir / "{}_raw.csv".format(stem)
    per_task_path = out_dir / "{}_per_task.csv".format(stem)
    leaderboard_path = out_dir / "{}_leaderboard.csv".format(stem)

    raw.to_csv(raw_path, index=False)
    per_task.to_csv(per_task_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)

    print("Saved:")
    print(" - {}".format(raw_path))
    print(" - {}".format(per_task_path))
    print(" - {}".format(leaderboard_path))

    if leaderboard.empty:
        print("\nNo runs found for given config/metric.")
    else:
        print("\nLeaderboard:")
        print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
