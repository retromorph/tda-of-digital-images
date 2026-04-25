# TDA of digital images

Эксперименты с **Persistent Homology Transformer** (код модели в [`src/`](src/), сценарии обучения в [`exp/`](exp/)).

## Настройка окружения

Нужны [Python 3.12+](https://www.python.org/downloads/) и [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync
```

Опционально Jupyter:

```sh
uv sync --extra jupyter
uv run jupyter notebook
```

Запуск экспериментов (из корня репозитория):

```sh
uv run python exp/run_phtx.py --dataset MNIST --help
uv run python exp/exp_main.py
uv run python exp/exp_invariance.py   # augmentation sweep (see exp/config/invariance.yaml)
```

### MLflow

По умолчанию раннеры пишут в локальное хранилище `file:./mlruns`. Чтобы указать свой сервер:

```sh
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python exp/run_phtx.py --dataset MNIST --epochs 1
```

Сводка по эксперименту (имя эксперимента можно задать через `MLFLOW_EXPERIMENT_NAME`):

```sh
MLFLOW_EXPERIMENT_NAME=MyExperiment uv run python scripts/mlflow_summarize.py
```

### PyTorch: CPU и CUDA

По умолчанию `uv sync` подтягивает **CPU**-сборки `torch` с PyPI. Для **CUDA** задайте подходящий индекс PyTorch в проекте (см. [документацию PyTorch](https://pytorch.org/get-started/locally/) и [uv: indexes](https://docs.astral.sh/uv/concepts/projects/dependencies/#index-urls)) и переустановите зависимости.

## Датасеты

2019 CoREMOF https://mof.tech.northwestern.edu/databases

2025 CoREMOF https://zenodo.org/records/15621349

## Модели


