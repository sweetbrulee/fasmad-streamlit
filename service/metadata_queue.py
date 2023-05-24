import queue
import numpy as np
from typing import List, NamedTuple


class DetectionMetadata(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


class MetadataQueueService:
    _thread_safety_queue: "queue.Queue[List[DetectionMetadata]]" = queue.Queue()

    def __init__(self):
        # singleton class, cannot be initialized
        raise NotImplementedError("This is a singleton class, cannot be initialized")

    @classmethod
    def use_queue(cls):
        return cls._thread_safety_queue
