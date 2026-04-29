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
uv run python exp/runners/run_persformer.py --dataset MNIST --help
uv run python exp/runners/run_latent_persformer.py --dataset MNIST --help
uv run python exp/runners/run_linear_persformer.py --dataset MNIST --help
uv run python exp/pipelines/legacy/main.py
uv run python exp/pipelines/ablations/invariance.py   # augmentation sweep (see exp/config/ablations/invariance.yaml)
uv run python exp/runners/run_persistence_image.py --dataset MNIST --epochs 1
uv run python exp/runners/run_persistence_landscape.py --dataset MNIST --epochs 1
uv run python exp/runners/run_persistence_silhouette.py --dataset MNIST --epochs 1
uv run python exp/pipelines/smoke/fixed_encoders.py
uv run python exp/pipelines/benchmark/preliminary_benchmark.py --config exp/config/benchmark/preliminary_clean.yaml --dry_run
uv run python exp/pipelines/benchmark/aggregate_preliminary_results.py --config exp/config/benchmark/preliminary_clean.yaml
```

### Classical fixed encoders

Добавлены фиксированные векторизации диаграмм в `src/fixed_encoders/`:
- `PersistenceImage` (2D сетка), классификатор: небольшой `2D CNN`
- `PersistenceLandscape` (k слоёв), классификатор: небольшой `1D CNN`
- `PersistenceSilhouette` (1 кривая), классификатор: небольшой `1D CNN`

Режимы взвешивания для `PersistenceImage` и `PersistenceSilhouette`:
- `none` — вес `1`
- `linear` — вес `persistence`
- `power` — вес `persistence ** weight_power`

### MLflow

По умолчанию раннеры пишут в локальное хранилище `file:./mlruns`. Чтобы указать свой сервер:

```sh
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python exp/runners/run_persformer.py --dataset MNIST --epochs 1
```

Сводка по эксперименту (имя эксперимента можно задать через `MLFLOW_EXPERIMENT_NAME`):

```sh
MLFLOW_EXPERIMENT_NAME=MyExperiment uv run python scripts/mlflow_summarize.py
```

Оркестратор `preliminary_benchmark.py` и отчёт `aggregate_preliminary_results.py` логируют runtime-артефакты (manifest/CSV) напрямую в MLflow artifacts.

### PyTorch: CPU и CUDA

По умолчанию `uv sync` подтягивает **CPU**-сборки `torch` с PyPI. Для **CUDA** задайте подходящий индекс PyTorch в проекте (см. [документацию PyTorch](https://pytorch.org/get-started/locally/) и [uv: indexes](https://docs.astral.sh/uv/concepts/projects/dependencies/#index-urls)) и переустановите зависимости.

## Датасеты

2019 CoREMOF https://mof.tech.northwestern.edu/databases

2025 CoREMOF https://zenodo.org/records/15621349

2D-porous-media-images-main https://github.com/gengshaoyang/2D-porous-media-images?utm_source=chatgpt.com

## Модели


