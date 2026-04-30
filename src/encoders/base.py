from src.fixed_encoders import (
    PersistenceImageEncoder,
    PersistenceLandscapeEncoder,
    PersistenceSilhouetteEncoder,
)
from src.registry import ENCODERS


@ENCODERS("persistence_image")
def persistence_image_encoder(**kwargs):
    return PersistenceImageEncoder(**kwargs)


@ENCODERS("persistence_landscape")
def persistence_landscape_encoder(**kwargs):
    return PersistenceLandscapeEncoder(**kwargs)


@ENCODERS("persistence_silhouette")
def persistence_silhouette_encoder(**kwargs):
    return PersistenceSilhouetteEncoder(**kwargs)
