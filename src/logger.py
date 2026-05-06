import json

import mlflow

# MLflow's default `mlflow.entities.Param.MAX_LENGTH` is 6000, and in some
# tracking-server installations it is even smaller (250/500). We stay well below
# both to avoid silent server-side rejections of long values.
_MAX_PARAM_VALUE_LEN = 250


def _to_param_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            text = str(value)
    if len(text) > _MAX_PARAM_VALUE_LEN:
        head = text[: _MAX_PARAM_VALUE_LEN - len(_TRUNC_SUFFIX)]
        return f"{head}{_TRUNC_SUFFIX}"
    return text


_TRUNC_SUFFIX = "...(truncated)"


def _normalize_params(params) -> dict:
    if not params:
        return {}
    out = {}
    for k, v in params.items():
        if v is None:
            continue
        out[str(k)] = _to_param_value(v)
    return out


def _normalize_tags(tags) -> dict:
    if not tags:
        return {}
    return {str(k): str(v) for k, v in tags.items() if v is not None}


class MLFlowLogger:
    """Thin wrapper over MLflow that auto-truncates long param values."""

    def __init__(self, url, project, params=None, run_name=None, tags=None):
        super().__init__()
        mlflow.set_tracking_uri(uri=url)
        mlflow.set_experiment(project)

        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id

        flat = _normalize_params(params)
        if flat:
            mlflow.log_params(flat, run_id=self.run_id)
        safe_tags = _normalize_tags(tags)
        if safe_tags:
            mlflow.set_tags(safe_tags)

    def end(self):
        mlflow.end_run()

    def log(self, metrics, step=0):
        if not metrics:
            return
        mlflow.log_metrics(metrics, step=step, run_id=self.run_id)

    def log_params(self, params):
        flat = _normalize_params(params)
        if flat:
            mlflow.log_params(flat, run_id=self.run_id)

    def log_tags(self, tags):
        safe_tags = _normalize_tags(tags)
        if safe_tags:
            mlflow.set_tags(safe_tags)
