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
    image_size: tuple[int, int]
    target_stats: dict | None = None
    label_offset: int = 0

    def __post_init__(self):
        object.__setattr__(self, "task", TaskKind(self.task))
        object.__setattr__(self, "color", ColorMode(self.color))

    @property
    def num_channels(self) -> int:
        return 3 if self.color == ColorMode.RGB else 1
