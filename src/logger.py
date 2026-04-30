import mlflow


class MLFlowLogger:
    def __init__(self, url, project, params=None, run_name=None, tags=None):
        super().__init__()
        mlflow.set_tracking_uri(uri=url)
        mlflow.set_experiment(project)

        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id

        if params:
            flat = {str(k): str(v) for k, v in params.items() if v is not None}
            mlflow.log_params(flat, run_id=self.run_id)
        if tags:
            safe_tags = {str(k): str(v) for k, v in tags.items() if v is not None}
            mlflow.set_tags(safe_tags)

    def end(self):
        mlflow.end_run()

    def log(self, metrics, step=0):
        mlflow.log_metrics(metrics, step=step, run_id=self.run_id)

    def log_params(self, params):
        if not params:
            return
        mlflow.log_params({str(k): str(v) for k, v in params.items() if v is not None}, run_id=self.run_id)

    def log_tags(self, tags):
        if not tags:
            return
        mlflow.set_tags({str(k): str(v) for k, v in tags.items() if v is not None})
