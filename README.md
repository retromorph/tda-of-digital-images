# TDA of digital images

Экспериментальное окружение для **Topological Data Analysis** в анализе изображений: сравнение моделей и фильтраций поверх Persistent Homology Transformer (`src/`), запуск/оркестрация экспериментов в `exp/`.

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

## Запуск одного эксперимента

Единая точка входа — `runner.py` со структурированным YAML и dotlist-overrides.

```sh
# полный конфиг по умолчанию
uv run python runner.py --config exp/config/unified/default.yaml

# точечные оверрайды
uv run python runner.py \
  --config exp/config/unified/default.yaml \
  --override seed=0 \
  --override device=cpu \
  --override dataset.name=MNIST \
  --override model.name=MLP \
  --override 'model.args={"d_hidden": 256, "num_layers": 2, "dropout": 0.1, "activation": "GELU", "alpha": 0.0}' \
  --override training.budget.value=10 \
  --override training.scheduler.name=warmup_cosine \
  --override training.scheduler.warmup_epochs=2 \
  --override training.early_stop.patience=5 \
  --override logging.experiment=MyExperiment/mnist
```

Полная схема конфигурации описана в `src/experiment/config.py` (`ExperimentConfig`).
Главные секции:

- `dataset.{name, args, test_time}` — какой датасет и какие test-time трансформации применять
- `filtration.{name, args, diagram_idx, positional_encoding}` — фильтрация PH (registry в `src/filtrations/`)
- `encoder.{name, args}` — для моделей с input_kind=encoded (registry в `src/encoders/`)
- `model.{name, args}` — модель из `src/models/base.py` (registry)
- `training.{batch_size, optimizer, scheduler, budget, early_stop, grad_accum, max_grad_norm}`
- `logging.{experiment, tags}`

Старые скрипты `exp/runners/run_*.py` остаются как тонкие шимы поверх единого `runner.py` (см. `exp/runners/_shim.py`).

## Sweeps и benchmark

Унифицированный оркестратор — `exp.pipelines.orchestrator`:

```sh
uv run python -m exp.pipelines.orchestrator \
  --config exp/config/benchmark/preliminary_clean.yaml --dry_run

uv run python -m exp.pipelines.orchestrator \
  --config exp/config/benchmark/preliminary_clean.yaml

# фильтр по методу/таску
uv run python -m exp.pipelines.orchestrator \
  --config exp/config/benchmark/preliminary_clean.yaml \
  --only_method persformer --only_task mnist_clean
```

Быстрый smoke поверх той же логики:

```sh
uv run python -m exp.pipelines.orchestrator --config exp/config/smoke/preliminary_quick.yaml
uv run python -m exp.pipelines.smoke.fixed_encoders
```

Отдельные пайплайны для аблейшнов перенесены на тот же оркестратор:

```sh
uv run python -m exp.pipelines.legacy.main
uv run python -m exp.pipelines.ablations.invariance
uv run python -m exp.pipelines.ablations.n_filters
uv run python -m exp.pipelines.ablations.directions
```

Каждый sweep также пишет в MLflow артефакт-манифест (`orchestrator_manifest.json`) с историей вызванных команд.

## Кэш диаграмм

Диаграммы фильтраций кешируются по стабильному хешу пары `(filtration_name, filtration_params)`:

- `data/cache/diagrams/<dataset>/<cache_key>/{train,val,test[_t-<transform>-<power>]}_seed-<n>.pkl`
- test-сплит дополнительно учитывает `transform_str` и `power`, чтобы invariance-эксперименты не подсунули чистые диаграммы для зашумлённых картинок
- payload содержит `version` (`CACHE_VERSION`) для будущих миграций

## Classical fixed encoders

Векторизации диаграмм в `src/fixed_encoders/`:
- `PersistenceImage` (2D сетка) → `PERSISTENCE_CNN2D`
- `PersistenceLandscape` (k слоёв) → `PERSISTENCE_CNN1D`
- `PersistenceSilhouette` (1 кривая) → `PERSISTENCE_CNN1D`

Закодированные фичи также кэшируются на диск (`src/fixed_encoders/feature_dataset.py`) с версионированием по содержимому энкодера.

## Trainer

`src/trainer.py` — единый трейнер для классификации и регрессии. Управление полностью через `OptimConfig` + `TrainConfig`:

- AdamW + опциональные `cosine` / `warmup_cosine`
- gradient accumulation (`train_config.grad_accum`)
- gradient clipping (`train_config.max_grad_norm > 0`)
- early stopping по `loss_val` (`train_config.early_stop_patience`, `early_stop_min_delta`)
- per-epoch логирование `loss_*`, `acc_*` / `rmse_*`/`mae`/`r2`, `lr` и `loss_test_at_*_best`

## MLflow

По умолчанию раннеры пишут в локальный sqlite backend `sqlite:///mlruns/mlflow.db`. Артефакты раннов хранятся в `mlruns/artifacts`. Чтобы использовать свой MLflow сервер без ручного `export`, создайте локальный `.env` из шаблона:

```sh
cp .env.example .env
uv run python runner.py --config exp/config/unified/default.yaml --override training.budget.value=1
```

Сводка по эксперименту:

```sh
MLFLOW_EXPERIMENT_NAME=MyExperiment uv run python scripts/mlflow_summarize.py
```

Локальный UI:

```sh
uv run mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root file:./mlruns/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

## PyTorch: CPU и CUDA

По умолчанию `uv sync` подтягивает CPU-сборки `torch`. Для CUDA задайте подходящий индекс PyTorch в проекте (см. [документацию PyTorch](https://pytorch.org/get-started/locally/) и [uv: indexes](https://docs.astral.sh/uv/concepts/projects/dependencies/#index-urls)) и переустановите зависимости.

## Датасеты

- 2019 CoREMOF — https://mof.tech.northwestern.edu/databases
- 2025 CoREMOF — https://zenodo.org/records/15621349
- 2D-porous-media-images — https://github.com/gengshaoyang/2D-porous-media-images
- MedMNIST / Torchvision MNIST family / NIST SD04 — собираются по `src/datasets/sources/*.py`
