from src.encoders.base import (  # noqa: F401
    persistence_image_encoder,
    persistence_landscape_encoder,
    persistence_silhouette_encoder,
)
from src.registry import ENCODERS


def create_encoder(name: str, **kwargs):
    factory = ENCODERS.get(name)
    return factory(**kwargs)


__all__ = ["ENCODERS", "create_encoder"]
