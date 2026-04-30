def get_dataset_cfg(dataset_str):
    from src.datasets.sources import fingerprints, medmnist, porous, synthetic, torchvision_mnist_family  # noqa: F401
    from src.registry import DATASETS

    loader_factory = DATASETS.get(dataset_str)
    loader = loader_factory()

    return {
        "dataset_loader": loader,
    }
