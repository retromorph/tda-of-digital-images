from dataclasses import dataclass

from src.models.deepsets import DeepSets
from src.models.latent_persformer import LatentPersformer
from src.models.linear_persformer import LinearPersformer
from src.models.mlp import MLP
from src.models.persistence_cnn1d import PersistenceCNN1D
from src.models.persistence_cnn2d import PersistenceCNN2D
from src.models.persformer import Persformer
from src.models.resnet import ResNet
from src.models.vit import ViT
from src.registry import MODELS


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_kind: str
    build: callable
    forward_takes_mask: bool = False


def _register(name: str, input_kind: str, forward_takes_mask: bool = False):
    def deco(fn):
        @MODELS(name)
        def _factory():
            return ModelSpec(
                name=name,
                input_kind=input_kind,
                build=fn,
                forward_takes_mask=forward_takes_mask,
            )

        return fn

    return deco


@_register("MLP", input_kind="flat")
def _build_mlp(meta, **cfg):
    return MLP(meta.dim**2, 1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("ResNet", input_kind="image")
def _build_resnet(meta, **cfg):
    in_channels = 3 if meta.color == "rgb" else 1
    return ResNet(in_channels=in_channels, d_output=1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("ViT", input_kind="image")
def _build_vit(meta, **cfg):
    return ViT(meta.dim, 1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("PHTS", input_kind="diagram", forward_takes_mask=True)
def _build_deepsets(meta, **cfg):
    return DeepSets(9, 1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("PERSFORMER", input_kind="diagram", forward_takes_mask=True)
def _build_persformer(meta, **cfg):
    return Persformer(d_out=1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("LINEAR_PERSFORMER", input_kind="diagram", forward_takes_mask=True)
def _build_linear_persformer(meta, **cfg):
    return LinearPersformer(d_out=1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("LATENT_PERSFORMER", input_kind="diagram", forward_takes_mask=True)
def _build_latent_persformer(meta, **cfg):
    return LatentPersformer(d_out=1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("PERSISTENCE_CNN1D", input_kind="encoded")
def _build_persistence_cnn1d(meta, **cfg):
    return PersistenceCNN1D(d_output=1 if meta.task == "regression" else meta.n_classes, **cfg)


@_register("PERSISTENCE_CNN2D", input_kind="encoded")
def _build_persistence_cnn2d(meta, **cfg):
    return PersistenceCNN2D(d_output=1 if meta.task == "regression" else meta.n_classes, **cfg)


def get_model_spec(name: str) -> ModelSpec:
    return MODELS.get(name)()
