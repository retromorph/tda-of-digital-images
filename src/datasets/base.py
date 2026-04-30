from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetMeta:
    task: str
    n_classes: int | None
    color: str
    image_size: tuple[int, int]
    target_stats: dict | None = None

    @property
    def dim(self) -> int:
        return int(self.image_size[0])

    @property
    def num_channels(self) -> int:
        return 3 if self.color == "rgb" else 1
