import mlflow


class MLFlowLogger():

    def __init__(self, url, project, params=None):
        super().__init__()

        mlflow.set_tracking_uri(uri=url)
        mlflow.set_experiment(project)

        mlflow.start_run()
        self.run_id = mlflow.active_run().info.run_id
        mlflow.log_params(params, run_id=self.run_id)

    # def __del__(self):
    #     mlflow.end_run()

    def end(self):
        mlflow.end_run()

    def log(self, metrics, step=0):
        mlflow.log_metrics(metrics, step=step, run_id=self.run_id)
