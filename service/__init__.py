from .kernel.firedetection.interface import start_fire_detect2
from .message_queue import MetadataQueueService


class InferenceService:
    def __init__(self):
        # singleton class, cannot be initialized
        raise NotImplementedError("This is a singleton class, cannot be initialized")


class FireDetection(InferenceService):
    @classmethod
    def create(cls, frame):
        return start_fire_detect2(frame)
