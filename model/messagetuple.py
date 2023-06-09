from typing import NamedTuple
import numpy as np


class DetectionMetadata(NamedTuple):
    group: str
    boxes: object
    # class_id: int
    # label: str
    # score: float
    # box: np.ndarray
