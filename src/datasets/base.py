from dataclasses import dataclass
from enum import Enum


class TaskKind(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ColorMode(str, Enum):
    GRAY = "gray"
    RGB = "rgb"


@dataclass(frozen=True)
class DatasetMeta:
    task: TaskKind
    n_classes: int | None
    color: ColorMode
    image_size: tuple[int, ...]
    target_stats: dict | None = None
    label_offset: int = 0

    def __post_init__(self):
        object.__setattr__(self, "task", TaskKind(self.task))
        object.__setattr__(self, "color", ColorMode(self.color))
        object.__setattr__(self, "image_size", tuple(self.image_size))
        if len(self.image_size) not in (2, 3):
            raise ValueError(
                f"image_size must have 2 (H, W) or 3 (D, H, W) entries, got {self.image_size!r}"
            )

    @property
    def num_channels(self) -> int:
        return 3 if self.color == ColorMode.RGB else 1

    @property
    def image_ndim(self) -> int:
        return len(self.image_size)
