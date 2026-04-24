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
```

### PyTorch: CPU и CUDA

По умолчанию `uv sync` подтягивает **CPU**-сборки `torch` с PyPI. Для **CUDA** задайте подходящий индекс PyTorch в проекте (см. [документацию PyTorch](https://pytorch.org/get-started/locally/) и [uv: indexes](https://docs.astral.sh/uv/concepts/projects/dependencies/#index-urls)) и переустановите зависимости.

## Датасеты

2019 CoREMOF https://mof.tech.northwestern.edu/databases

2025 CoREMOF https://zenodo.org/records/15621349

## Модели


