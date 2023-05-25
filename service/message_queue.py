import queue
from typing import List
from typing_extensions import override

from model.messagetuple import DetectionMetadata




class MessageQueueService:
    def __init__(self):
        # singleton class, cannot be initialized
        raise NotImplementedError("This is a singleton class, cannot be initialized")
    
    @classmethod
    def use_queue(cls):
        pass


class MetadataQueueService(MessageQueueService):
    _thread_safety_queue: "queue.Queue[List[DetectionMetadata]]" = queue.Queue()
    
    @override
    @classmethod
    def use_queue(cls):
        return cls._thread_safety_queue
